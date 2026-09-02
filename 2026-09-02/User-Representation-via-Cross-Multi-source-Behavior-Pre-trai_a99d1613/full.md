# User Representation via Cross Multi-source Behavior Pre-training for Mobile Games

Chengqi Yang<sup>†‡</sup>, Yiran Qiao<sup>†‡</sup>, Feng Liu<sup>§</sup>, Xingyu Lou<sup>§</sup>, Zijun Zhou<sup>§</sup>, Xiaoyun Mo<sup>§</sup> Changwang Zhang<sup>§</sup>, Jiayuan Xu<sup>†‡</sup>, Jun Wang<sup>§</sup>, Xiang Ao<sup>†§¶∗</sup>

<sup>†</sup> State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences (CAS), Beijing, China.

<sup>‡</sup> University of Chinese Academy of Sciences, CAS, Beijing, China.

<sup>§</sup> OPPO Research Institute, Shenzhen, China

<sup>¶</sup> Institute of Intelligent Computing Technology, CAS, Suzhou, China.

Abstract—User representation pre-training has become a fundamental paradigm for alleviating data sparsity in downstream personalization tasks. However, existing studies predominantly focus on single-app or app-level behaviors, overlooking the inherently cross-source and multi-granular nature of user activities on mobile devices. At the device level, user intent emerges from complex interactions among heterogeneous behavior sources and hierarchical action structures, posing challenges that cannot be addressed by conventional app-centric modeling. To tackle this issue, we propose CM-PTM, a novel Cross Multi-source Behavior Pre-Training Model tailored for mobile game user representation learning on device-level behavioral logs. CM-PTM employs hierarchical cascaded mask-then-predict proxy tasks that first infer the source of the next behavior and then progressively refine predictions at the app-action level. This design enables unified modeling of cross-source dependencies and fine-grained behavioral dynamics within a single pre-training paradigm. Extensive experiments on large-scale real-world mobile datasets demonstrate that CM-PTM effectively captures users’ endogenous interests and consistently delivers significant performance gains on downstream mobile game recommendation tasks.

Index Terms—User Behavior Modeling, Pre-training Model, Multi-granular Behavior Modeling, Cross-source Modeling

## I. INTRODUCTION

User representation modeling [1] aims at learning dense vector embeddings that capture user characteristics and interests from historical behaviors. The learned user representations can be applied to various downstream user-related tasks, such as user profile prediction [2], personalized recommendation [3], fraud detection [4], and click-through-rate prediction [5].

Inspired by the success of pre-training models (PTMs) in NLP [6], [7] and CV [8], [9], some studies have begun to leverage large-scale unlabeled behavioral data to pre-train user representations via self-supervised learning. For example, several methods [10], [11] treat user behavior sequences as word sequences, adopting masked-behavior prediction and next-behavior prediction objectives. Some studies [12]–[14] employ contrastive learning by maximizing the consistency between original and augmented views of user behaviors.

Despite the success of previous studies, we observe that they are predominantly confined to single-source or app-level behavior modeling [10], [12], [13], [15]. In this paper, we aim to learn user representations by leveraging all available behaviors from a user’s mobile device. Real-world mobile devices naturally aggregate user behaviors from multiple heterogeneous sources with diverse granularity and accessibility constraints. Therefore, device-level user representation learning cannot be reduced to a simple extension of app-centric modeling, but instead constitutes a distinct problem setting with intrinsic structural complexity.

Specifically, under privacy and permission constraints, mobile devices typically observe two categories of behavioral data: (1) system-level applications (e.g., App Store, Game Center, Browser) that are pre-installed with the OS, for which providers have privileged access to fine-grained interaction data; (2) third-party applications installed by users, for which providers can only observe coarse-grained events such as installation, launch, and uninstall, without access to in-app interactions. Notably, user interactions with third-party applications are far more frequent than those with system-level ones, leading to a pronounced mismatch between behavioral volume and behavioral granularity. This asymmetry introduces substantial challenges in learning high-quality user representations from mobile device data.

More fundamentally, user intentions on mobile devices often emerge from cross-source interactions spanning heterogeneous behavior streams. Figure 1 illustrates a real user case from our dataset. The user’s interactions with the games Honor of Kings and Zombie Frontier span different sources, including the third-party App, the App Store, and the Browser. A closer inspection reveals that the user may have encountered an advertisement for Zombie Frontier on TikTok, explored its details via the App Store and Browser, but ultimately did not download the game, suggesting weak installation intent despite multi-step engagement across sources.

This example underscores the potential and complexity of constructing comprehensive user representations by integrating behaviors from multiple, semantically diverse sources. Unlike prior studies that typically assume single-source behavior sequences, mobile device data entails cross-source interactions where user intentions are distributed across behaviors of differing types and granularities. Directly aggregating all behaviors into a single sequence [11], [12], [16] fails to account for these structural distinctions, often degrading model performance (as shown in the ablation study). Moreover, cross-source interleaving weakens local behavioral continuity within individual sources, undermining the core assumption of strong intra-sequence dependencies underlying many pretraining frameworks [10], [11], [17].

![](images/93b640355da3192297bfa13b04d8f6f06478d06dac01dc4388c32c716c5b4f70.jpg)  
Fig. 1. An illustrative example of user behavior on a mobile device from our real-world dataset. ‘Browser’ indicates interactions within the pre-installed browser; ‘Appstore’ denotes activities in the system App Store; ‘Thirdparty Apps’ represent coarse-grained interactions regarding user-installed applications.

To address these issues, we propose a novel user representation pre-training framework tailored to mobile game recommendation on device-level multi-source logs, termed CM-PTM (Cross Multi-source Behavior Pre-Training Model). CM-PTM takes as input multi-source, multi-granularity user behavior sequences organized by their respective data sources. Motivated by a hierarchical decision structure for progressive intent refinement, CM-PTM employs a cascaded selfsupervised prediction strategy. Given a user’s historical behaviors, it first predicts the source of the next behavior, then progressively infers the App category, the specific App ID, and the user’s operation type within the App if applicable. Each stage is conditioned on the outcome of the previous one, enabling the model to incrementally refine its understanding of user intent across heterogeneous behavior streams. Crucially, CM-PTM is designed as a system-level architecture for asymmetric cross-source fusion rather than as a collection of new attention primitives, enabling unified representation learning under the volume–granularity mismatch inherent to mobile device logs. We conducted extensive experiments on large-scale real-world datasets in the game recommendation scenario, demonstrating that CM-PTM achieves substantial performance improvements.

Our contributions can be summarized as follows:

• We investigate user representation pre-training based on behavioral data from mobile devices. In this scenario, user intentions may span multiple behavior sequences with different sources and granularities, which limits the direct applicability of conventional single-sequence pre-training

paradigms to this setting.

• We propose a pre-training approach called CM-PTM, which is built upon the mask-then-predict framework. CM-PTM incorporates several hierarchical cascaded proxy tasks specifically designed to capture cross-source associations, thereby improving the representation of user behavior across multiple sources.

• Both extensive offline experiments and online A/B tests are conducted to demonstrate the effectiveness and robustness of our proposed approach in large-scale real-world industrial scenarios.

## II. RELATED WORK

In this section, we briefly review several existing works, including user representation pre-training, multi-behavior modeling and cross-domain recommendation.

## A. User Representation Pre-training

The pre-training paradigm has been widely used in NLP [18]–[20] and CV [21], [22], with [23] attributing its effectiveness to the mask-then-predict strategy. Inspired by these successes, the pre-training paradigm has been used to learn user representations from large-scale unlabeled behaviors. Existing methods can be categorized into generative and discriminative methods. Generative methods typically disrupt and reconstruct user behavior sequences. Commonly used strategies include mask behavior prediction (MBP) [4], [11], [24]–[26] and next behavior prediction (NBP) [10], [11], [27]. [28], [29] replaced NBP with a behavior distribution prediction to reduce the negative influence of noise in user behaviors. Discriminative methods maximize the similarity between two behavioral views through contrastive learning. For example, [30], [31] maximize the similarity between multi-modal or multi-granularity behavioral views, and other studies use data augmentation strategies to obtain different views. Commonly used data augmentation strategies include explicit methods, such as masking [13], [16], cropping [32], and substitution [14], and implicit methods, where the same behavior sequence is fed to the model twice with different dropout masks [6], [12], [33], [34]. However, these pretraining methods cannot capture the cross-source correlations, making them less suitable for our scenario.

## B. Multi-Behavior Modeling

Multi-behavior modeling aims to leverage various kinds of user behaviors to provide a more comprehensive understanding of user interests. [35]–[39] have demonstrated that multi-behavior modeling outperforms single-behavior approaches. [40]–[42] use graph neural networks (GNNs) to mine the complex relationships between heterogeneous user behaviors while [31], [43] use Transformer-based architectures [44] to model different behavior sequences, then capture the correlation between heterogeneous behaviors. Additionally, [45] combined GNN-based and Transformer-based methods to model users’ periodic behavior patterns and the dependencies between multiple behaviors. Although these approaches have achieved good results, they all focus on the behavior of a single source, potentially limiting their ability in our settings.

## C. Cross-Domain Recommendation

The core concept of cross-domain recommendation (CDR) is to integrate and analyze user behaviors and preferences in different domains to provide personalized recommendations. Some CDR methods integrate user behaviors of different granularities from multiple domains to improve the accuracy of the recommendation system [46]. Mainstream methods include RNN-based, GNN-based and attention-based methods. [47] proposed an RNN-based long short-term memory fusion module that integrates knowledge extracted from multiple domains. GNN-based methods usually use graph convolution to aggregate neighborhood information [48] or graph attention networks [49] to fuse cross-domain semantic associations. [50] introduces differential privacy protection for cross-domain knowledge transfer on this basis. Attention-based methods usually optimize cross-domain feature interactions through hierarchical attention mechanisms [51], domain adaptation [52], and multiple contrastive learning [53]. However, existing methods focus on how to efficiently integrate multi-granularity behaviors from multiple domains without explicitly capturing the cross-source correlations of user behaviors.

## III. PROBLEM STATEMENT

We assume that a substantial amount of behavioral data from n different sources can be accessed on mobile devices. Let $\mathcal { U } = \{ u _ { 1 } , u _ { 2 } , . . . , u _ { | \mathcal { U } | } \}$ denote the set of users, and $\nu _ { k }$ where $k \ = \ 1 , . . . , n .$ , denotes the set of all the behaviors from the k-th source, respectively. For each $u \in \mathcal { U } , S _ { u } ^ { k } =$ $\{ v _ { < u , 1 > } ^ { k } , . . . , v _ { < u , t > } ^ { k } , . . . , v _ { < u , | S _ { * } ^ { k } | > } ^ { k } \}$ denotes their chronologically ordered sequence of behaviors from the k-th source.

Notation. Throughout the paper, italic $u \in \mathcal { U }$ denotes a user entity only. Bold $\pmb { u } ^ { k } \in \hat { \mathbb { R } } ^ { 2 \bar { d } }$ denotes the source-specific user representation used in pre-training, and bold $\textbf { \em u } \in \mathbb { R } ^ { d _ { u } }$ denotes the concatenated downstream representation; neither bold symbol denotes the user ID. Subscripts such as u in $\boldsymbol { S _ { u } ^ { k } }$ $\boldsymbol { E } _ { u } ^ { k }$ , and $\mathcal { L } ^ { u }$ always refer to the user entity.

Our task aims to learn a function $\mathcal { F } _ { \theta }$ that maps multigranularity user behaviors from multiple sources into a dense user representation, formulated as

$$
\pmb { u } = \mathcal { F } _ { \theta } ( S _ { u } ^ { 1 } , . . . , S _ { u } ^ { n } ) \in \mathbb { R } ^ { d _ { u } } ,\tag{1}
$$

where u denotes the learned downstream representation for user $u \in \mathcal { U } , d _ { u }$ denotes its dimension, and θ denotes the model parameters.

## IV. METHODOLOGY

In this section, we describe CM-PTM in detail, as illustrated in Figure 2. It comprises two components: the multigranularity user model and the cross-source cascaded prediction module. The former integrates fine-grained and coarsegrained behaviors through hierarchical attention mechanisms, while the latter gradually refines the prediction of users next behavior, from source prediction to fine-grained action prediction.

## A. Multi-Granularity User Model

The key challenge lies in simultaneously preserving intrasource behavioral continuity and capturing inter-source intent correlations under heterogeneous granularity. Unlike conventional multi-behavior modeling where all actions are observed at comparable granularity within a unified platform, devicelevel logs exhibit a severe mismatch between behavioral volume and granularity across system apps (fine-grained) and third-party apps (coarse-grained). Our primary contribution is therefore not the invention of novel attention operators, but an architectural design tailored to this asymmetric industrial constraint. We first use a behavior embedding module to transform behavior sequences into fixed-length embedding sequences. Then, to integrate multi-granularity behaviors from multiple sources under this asymmetry, we design a granularity-aware self-attention mechanism (GASA) and a cross-granularity fusion attention mechanism (CGFA), whose integration explicitly targets asymmetric cross-source fusion—a setting where standard Transformers that treat all behavior tokens uniformly fail to model effectively—and fuse their outputs as the user representation.

1) Behavior Embedding: In this paper, we collect user behaviors from OPPO device-level logs under three distinct sources: third-party GameApps, Appstore, and Gamecenter. Our method is instantiated on this fixed multi-source, multigranularity setting in mobile game recommendation.

For $u \in \mathcal { U }$ , we obtain the behavior embeddings of user u as:

$$
\begin{array} { r l } & { \pmb { { E } } _ { u } ^ { k } = \mathrm { E m b e d d i n g } ( \mathcal { S } _ { u } ^ { k } ) } \\ & { \quad \quad = [ { \pmb { e } } _ { < u , 1 > } ^ { k } , . . . , { \pmb { e } } _ { < u , t > } ^ { k } , . . . , { \pmb { e } } _ { < u , | S _ { u } ^ { k } | > } ^ { k } ] \in \mathbb { R } ^ { | S _ { u } ^ { k } | \times d } , } \end{array}\tag{2}
$$

where d is the embedding dimension, $\boldsymbol { S } _ { u } ^ { k }$ represents the behavior sequence of user u from the k-th source, [·] denotes the concatenation of per-event embeddings, and $e _ { < u , t > } ^ { k }$ is the embedding of the t-th behavioral event in source k.

We set the maximum length of the embedding sequence $\boldsymbol { E } _ { u } ^ { k }$ to l. In cases where $\boldsymbol { E } _ { u } ^ { k }$ exceeds the maximum length l, only the most recent l behaviors are retained; otherwise, zero padding extends it to length l. Positional encoding is then added:

$$
\begin{array} { r } { \pmb { E } _ { u } ^ { k } = \mathrm { P a d O r T r u n c a t e } ( \pmb { E } _ { u } ^ { k } ; l ) + \pmb { P } ^ { k } , } \end{array}\tag{3}
$$

where $P ^ { k } \in \mathbb { R } ^ { l \times d }$ denotes the positional embedding.

2) Granularity-Aware Self-Attention: In order to enhance the model’s ability to perceive the granularity of user behavior and capture the behavioral correlations within a single source, following [17], we design a granularity-aware self-attention module (GASA). GASA employs a multi-head self-attention to identify the correlations between behaviors of the same granularity. For $\mathbf { \delta } _ { E _ { u } ^ { i } }$

$$
\pmb { Q } _ { i } ^ { h } = \pmb { E } _ { u } ^ { i } \cdot \pmb { W } _ { i } ^ { Q h } , \pmb { K } _ { i } ^ { h } = \pmb { E } _ { u } ^ { i } \cdot \pmb { W } _ { i } ^ { K h } , \pmb { V } _ { i } ^ { h } = \pmb { E } _ { u } ^ { i } \cdot \pmb { W } _ { i } ^ { V h } ,
$$

$$
{ \mathrm { A t t n } } _ { i } ^ { h } = { \mathrm { s o f t m a x } } ( \frac { \boldsymbol { Q } _ { i } ^ { h } \cdot \left( \boldsymbol { K } _ { i } ^ { h } \right) ^ { \top } } { \sqrt { d _ { k } } } ) \cdot \boldsymbol { V } _ { i } ^ { h } , h \in [ 1 , \cdots , H ] ,\tag{4}
$$

$$
\pmb { g } _ { u } ^ { i } = \mathrm { M e a n } \_ \mathrm { p o o l i n g } ( \mathrm { C o n c a t } ( \mathrm { A t t n } _ { i } ^ { 1 } , . . . , \mathrm { A t t n } _ { i } ^ { H } ) \cdot \pmb { W } _ { i } ^ { O } )
$$

![](images/ed12d57f4e4a4f9d302e1e3f7e9e04967d5ca4d7b186b403f7e26c2840ab25a2.jpg)  
Fig. 2. The overall architecture of the proposed CM-PTM framework. The lower part depicts the multi-granularity user encoder, which derives user representations from historical behaviors in different sources. The upper part shows the cross-source cascaded prediction module, consisting of inter-source and intra-source stages. Bold $\mathbf { \Delta } _ { \mathbf { \pmb { u } } } { } ^ { k }$ in the figure denotes the source-specific representation in Eq. (6); italic u denotes the user entity (Sec. III).

where $\pmb { W } _ { i } ^ { Q h } , \pmb { W } _ { i } ^ { K h } , \pmb { W } _ { i } ^ { V h } \in \mathbb { R } ^ { d \times d _ { k } } , \pmb { W } _ { i } ^ { O } \in \mathbb { R } ^ { d \times d }$ are learnable weight matrices, H is the number of attention heads, and $d _ { k } = d / H$

3) Cross-Granularity Fusion Attention: Since the user’s specific intentions may span multiple sources, we further design a cross-granularity fusion attention module (CGFA) to capture the cross-source behavioral correlations. CGFA employs multi-head cross-attention to integrate the macrolevel behavior patterns, as reflected in coarse-grained behaviors, with the micro-level insights derived from fine-grained behaviors. For $\pmb { { E } } _ { u } ^ { 1 }$ , we compute

$$
\begin{array} { r l } & { \quad \pmb { Q } _ { 1 } ^ { h } = \pmb { E } _ { u } ^ { 1 } \cdot \pmb { W } _ { 1 } ^ { \boldsymbol { Q } h } , \pmb { K } _ { 2 , 3 } ^ { h } = \pmb { E } _ { u } ^ { 2 , 3 } \cdot \pmb { W } _ { 2 , 3 } ^ { K h } , \pmb { V } _ { 2 , 3 } ^ { h } = \pmb { E } _ { u } ^ { 2 , 3 } \cdot \pmb { W } _ { 2 , 3 } ^ { V h } , } \\ & { \mathrm { A t t n } _ { 1 } ^ { h } = \mathrm { s o f t m a x } ( \frac { \pmb { Q } _ { 1 } ^ { h } \cdot ( \pmb { K } _ { 2 , 3 } ^ { h } ) ^ { T } } { \sqrt { d _ { k } } } ) \cdot \pmb { V } _ { 2 , 3 } ^ { h } , h \in [ 1 , \cdots , H ] , } \\ & { \quad \pmb { c } _ { u } ^ { 1 } = \mathrm { M e a n } _ { \mathrm { - p o o l i n g } } ( \mathrm { C o n c a t } ( \mathrm { A t t n } _ { 1 } ^ { 1 } , . . . , \mathrm { A t t n } _ { 1 } ^ { H } ) \cdot \pmb { W } _ { 1 } ^ { O } ) } \end{array}\tag{5}
$$

where ${ W } _ { 1 } ^ { Q h } , { W } _ { 2 , 3 } ^ { K h } , { W } _ { 2 , 3 } ^ { V h } \in \mathbb { R } ^ { d \times d _ { k } } , { W } _ { 1 } ^ { O } \in \mathbb { R } ^ { d \times d }$ are learnable weight matrices, $\bar { E } _ { u } ^ { \bar { 2 } , 3 }$ is the concatenation of $ { \boldsymbol { E } } _ { u } ^ { 2 }$ and $\pmb { { \cal E } } _ { u } ^ { 3 }$ , H is the number of attention heads, and $d _ { k } = d / H$ . We can get $\boldsymbol { c } _ { u } ^ { 2 }$ and $ { c _ { u } } ^ { 3 }$ in a similar way as $\boldsymbol { c } _ { u } ^ { 1 }$

4) User Representation: Finally, we concatenate the outputs of GASA and CGFA to generate user representations with both source-specific and cross-source association knowledge. For pre-training, to learn source-specific behavior patterns and avoid the negative influence of heterogeneous behaviors from other sources, we independently use the representations from the three sources to complete the pre-training task. For downstream tasks, user representations are expected to encapsulate as much knowledge as possible, ensuring that downstream models can flexibly exploit rich local dependencies and global cross-source correlations. Therefore, we concatenate all representations and feed the resulting vector into the downstream model. The representations of user $u \in \mathcal { U }$ are:

$$
\begin{array} { l l } { { \mathrm { P r e - t r a i n i n g : ~ } } } & { { { \pmb u } ^ { k } = \mathrm { C o n c a t } ( \pmb { c } _ { u } ^ { k } , \pmb { g } _ { u } ^ { k } ) , \ { k = 1 , 2 , 3 } } } \\ { { \mathrm { D o w n s t r e a m : ~ } } } & { { { \pmb u } = \mathrm { C o n c a t } ( \pmb { c } _ { u } ^ { 1 } , \pmb { c } _ { u } ^ { 2 } , \pmb { c } _ { u } ^ { 3 } , \pmb { g } _ { u } ^ { 1 } , \pmb { g } _ { u } ^ { 2 } , \pmb { g } _ { u } ^ { 3 } ) } } \end{array}\tag{6}
$$

## B. Cross-Source Cascaded Prediction

In our method, we propose a cross-source cascaded prediction framework that follows a hierarchical decision structure. Unlike parallel multi-task learning, the cascaded design explicitly conditions fine-grained predictions on high-level intent, which reduces the hypothesis space at each stage and mitigates label noise caused by cross-source ambiguity. Specifically, we decouple the standard next behavior prediction (NBP) task [10], [11], [27] into two sequential phases, an intersource phase that predicts the source of the next behavior (e.g., opening the AppStore), followed by an intra-source phase that refines the specific action within the selected source (e.g., searching for a game). This cascaded design explicitly models high-level intent before fine-grained action, aligning with natural decision patterns. We detail each phase below.

(i) Inter-Source Phase. This phase addresses the high-level task of behavior source prediction (T ), aiming to determine in which source (e.g., GameApps, Appstore, Gamecenter) the user’s next behavior will occur. Given a source-specific user embedding $\mathbf { \Delta } _ { \mathbf { \ b { u } } } { } ^ { k }$ , we apply a shared source prediction tower $f _ { s r c } ( . )$ to generate logits $r _ { 1 } ^ { k } ,$ followed by a softmax to obtain the predicted distribution $\bar { \hat { p } } _ { s r c } ^ { k }$ . To support subsequent tasks, an $I n f o$ module $f _ { i n f o } ^ { 1 } ( \cdot )$ generates a context vector $z _ { 1 } ^ { k }$ summarizing the current task state as prior knowledge as follows.

$$
\begin{array} { r l } & { ~ { \pmb r } _ { 1 } ^ { k } = f _ { s r c } ( { \pmb u } ^ { k } ) , } \\ & { \hat { \pmb p } _ { s r c } ^ { k } = \mathrm { s o f t m a x } ( \pmb r _ { 1 } ^ { k } ) , } \\ & { ~ z _ { 1 } ^ { k } = f _ { i n f o } ^ { 1 } ( \pmb r _ { 1 } ^ { k } ) , } \end{array}\tag{7}
$$

During pre-training, the source label is derived from the held-out next behavior in the globally chronological multisource log. Let $s _ { u } ~ \in ~ \{ 1 , 2 , 3 \}$ denote the source of this target behavior. We apply the same source supervision to each source-specific representation $\mathbf { \Delta } _ { \mathbf { \ b { u } } ^ { k } }$ , encouraging each source encoder to capture cross-source transition patterns. Here $f _ { s r c } ( . )$ denotes the source tower network and $f _ { i n f o } ^ { 1 } ( \cdot )$ refers to the $I n f o$ network; both are implemented as 2-layer MLPs with ReLU activation.

(ii) Intra-Source Phase. After modeling the behavior source, we next refine the intra-source behavior predictions. Directly predicting the next action might be too difficult due to the large action space. Therefore, we decompose the prediction of the user’s specific action into three cascaded subtasks. Rather than passing hard predicted labels to subsequent tasks, the cascade propagates a differentiable task state $z _ { i } ^ { k } !$ each sub-task conditions its prediction on the prior state $z _ { i - 1 } ^ { k ^ { - } }$ through an attention fusion step before softmax. The sub-goals in this phase include App category prediction (T<sub>2</sub>), App ID prediction $( T _ { 3 } )$ , and Operation prediction $( T _ { 4 } )$ . We describe each sub-task below.

$T _ { 2 }$ predicts the category of the App that the user will interact with next. $T _ { 3 }$ predicts the specific App ID. Once the App category is determined, the next step is to predict the exact App ID; conditioned on the category context from $T _ { 2 } , T _ { 3 }$ becomes more manageable. $T _ { 4 }$ predicts the specific operation type of the user’s next behavior when operation labels are available and semantically meaningful. For Appstore and Gamecenter behaviors, both click and search events mainly reflect positive user intent in our setting; distinguishing them with $T _ { 4 }$ therefore provides limited additional benefit for learning user interests, so this task is skipped to reduce computational complexity. In contrast, for third-party GameApps, install, uninstall, and launch operations can reflect different user attitudes, making $T _ { 4 }$ useful for user interest modeling.

For each applicable sub-task $T _ { i } ( i = 2 , 3 , 4 )$ , a task-specific tower $f _ { T } ^ { i }$ first maps the source-specific embedding $\mathbf { \Delta } _ { \mathbf { \ b { u } } ^ { k } }$ to features $\boldsymbol { r } _ { i } ^ { k }$ . The prior state $z _ { i - 1 } ^ { k } { \mathrm { - } } \mathrm { w i t h } z _ { 1 } ^ { k }$ from $T _ { 1 }$ when $i = 2 \mathrm { - } \mathrm { i } \mathrm { s }$ fused with $\boldsymbol { r } _ { i } ^ { k }$ via $f _ { a t t n } ^ { i }$ to obtain the task context $\pmb q _ { i } ^ { k }$ , from which $\hat { \pmb { p } } _ { i } ^ { k }$ is predicted. For $i \in \{ 2 , 3 \} , \ f _ { i n f o } ^ { i }$ updates $z _ { i } ^ { k }$ for the next sub-task; when $i = 4 , z _ { 4 } ^ { k }$ is omitted because $T _ { 4 }$ is the last intra-source sub-task on GameApps. Formally,

$$
\begin{array} { r l } & { { \boldsymbol { r } } _ { i } ^ { k } = f _ { T } ^ { i } ( { \boldsymbol { u } } ^ { k } ) , } \\ & { { \boldsymbol { q } } _ { i } ^ { k } = f _ { a t t n } ^ { i } ( { \boldsymbol { r } } _ { i } ^ { k } , { \boldsymbol { z } } _ { i - 1 } ^ { k } ) , } \\ & { \hat { \boldsymbol { p } } _ { i } ^ { k } = \mathrm { s o f t m a x } ( W _ { i } { \boldsymbol { q } } _ { i } ^ { k } + b _ { i } ) , } \\ & { { \boldsymbol { z } } _ { i } ^ { k } = f _ { i n f o } ^ { i } ( { \boldsymbol { q } } _ { i } ^ { k } ) , \quad i \in \{ 2 , 3 \} , } \end{array}\tag{8}
$$

where i indexes the applicable intra-source tasks among $\{ 2 , 3 , 4 \}$ , and $W _ { i } , b _ { i }$ are task-specific output parameters. Here $f _ { T } ^ { i } ( \cdot )$ represents the network of the $T _ { i }$ tower and $f _ { i n f o } ^ { i } ( \cdot )$ denotes the i-th Info module; $f _ { s r c } , ~ f _ { i n f o } ^ { i }$ , and $f _ { T } ^ { i }$ are all implemented as 2-layer MLPs with ReLU activation. $f _ { a t t n } ^ { i } ( \cdot )$ represents the i-th Attention module, implemented by selfattention.

## C. Model Training

Let $\mathscr { A } _ { 1 } , \mathscr { A } _ { 2 } , \mathscr { A } _ { 3 }$ represent the prediction task sets for thirdparty GameApps, Appstore, and Gamecenter in the intrasource stage, respectively. Specifically, $A _ { 1 } ~ = ~ \{ T _ { 2 } , T _ { 3 } , T _ { 4 } \}$ for third-party GameApps, while $\mathcal { A } _ { 2 } = \mathcal { A } _ { 3 } = \{ T _ { 2 } , T _ { 3 } \}$ for Appstore and Gamecenter, where $T _ { 4 }$ is omitted for the latter two sources as discussed above. Losses for the two stages can be expressed as follows:

$$
\mathcal { L } _ { I N T E R } = - \sum _ { k = 1 } ^ { 3 } \log ( \hat { p } _ { s r c , s _ { u } } ^ { k } ) ,\tag{9}
$$

$$
\mathcal { L } _ { I N T R A } = - \sum _ { k = 1 } ^ { 3 } \sum _ { T \in \mathcal { A } _ { k } } \sum _ { i = 0 } ^ { N ( T ) - 1 } ( y _ { T } ^ { k } = i ) \mathrm { l o g } ( \hat { p } _ { T } ^ { k } ) ,\tag{10}
$$

where $N ( T )$ denotes the vocabulary size for sub-task $T ,$ i.e., the total number of candidate categories for that task; for example, when $T$ is App ID prediction $( T _ { 3 } ) , \ N ( T _ { 3 } )$ is the number of unique candidate apps in the corresponding behavior source. $\hat { p } _ { T } ^ { k }$ denotes the predicted probability distribution for task $T$ on source k.

The overall loss function of our pre-training tasks is:

$$
\mathcal { L } = \lambda \mathcal { L } _ { I N T E R } + ( 1 - \lambda ) \mathcal { L } _ { I N T R A } ,\tag{11}
$$

where λ is a hyper-parameter balancing the two stages.

To reduce computational complexity, we only set training and validation sets during the pre-training stage, and evaluate the effectiveness of user representations in downstream scenarios.

## V. EXPERIMENTS

We conduct extensive experiments on large-scale real-world industrial datasets to evaluate the effectiveness of our proposed CM-PTM. CM-PTM follows a distinct two-stage paradigm rather than end-to-end multi-task supervised learning. In Stage 1, the model is trained via self-supervised mask-thenpredict proxy tasks to learn universal user representations. In Stage 2 (downstream adaptation), the pre-trained representations are frozen or fine-tuned and fed into a LightGBM [54] model together with user profile features for downstream mobile game recommendation.

## A. Experimental Setting

1) Datasets: All datasets in our experiments are collected from OPPO, one of the largest smartphone manufacturers in the world, including user behaviors over the 180 days preceding the sampling date. The data is strictly anonymized and fully authorized by users under signed agreements, ensuring compliance with data privacy and security regulations. User behaviors consist of the following sources and types: ‘Launch’, ‘Install’ and ‘Uninstall’ for third-party game apps; ‘Search and ‘Click’ for both App Store and Game Center.

Pre-training Dataset. We select a one-week sampling interval for pre-training from 2023/07/13 to 2023/07/19 to ensure data richness and diversity (see Table I for details).

TABLE I  
STATISTICS OF THE PRE-TRAINING DATASET.
<table><tr><td>Statistics</td><td>Pre-training dataset</td></tr><tr><td># users</td><td>1,947,683</td></tr><tr><td># apps</td><td>83,341</td></tr><tr><td>avg. # gameapp operation behaviors</td><td>144.93</td></tr><tr><td>avg. # appstore click behaviors</td><td>62.21</td></tr><tr><td>avg. # appstore search behaviors</td><td>22.60</td></tr><tr><td>avg. # gamecenter click behaviors</td><td>3.84</td></tr><tr><td>avg. # gamecenter search behaviors</td><td>1.12</td></tr></table>

Downstream Task Datasets. We design downstream tasks on mobile game recommendation, and there is no temporal overlap between pre-training data and downstream task data. Although our experiments focus on the game domain, all evaluations are conducted on mobile game recommendation tasks defined below.

Specifically, we construct six distinct tasks $( G _ { 1 } \sim G _ { 6 } )$ . We select popular blockbuster games for these tasks to ensure sufficient positive and negative interaction samples within the strict sampling windows, enabling robust and statistically significant offline evaluations.

Before describing the tasks, we first clarify several key concepts. Old games refer to games that were released at least six months ago. New games refer to games released within the past month. Old users are users who played the game during the past six months but not within the last two weeks. New users are users who never played the game before. Attract means that the user is labeled as positive if they played the target game within the specified evaluation window.

The three kinds of downstream scenarios are as follows: (i) Attract new users for old games. This scenario refers to predicting whether a new user will play an old game. We select three popular old games: Mini World $\left( G _ { 1 } \right)$ , Eggy Party $\left( G _ { 2 } \right)$ and $S k y \colon$ Children of $L i g h t \ ( G _ { 3 } )$

The training sets for $G _ { 1 } ~ \sim ~ G _ { 3 }$ were sampled from 2023/07/20 to 2023/07/22. Positive samples are new users who played the game during the sampling period, while negative samples are those who had not played the game by the end of that period. The positive-to-negative ratio is 1 : 2.

The test sets were sampled from 2023/07/23 to 2023/07/25. We randomly selected millions of new users and labeled them based on whether they played the game in the following week. (ii) Attract old users for old games. This scenario refers to predicting whether an old user will play an old game again. We select a widely popular game, Honor of Kings $\left( G _ { 4 } \right)$

The training set for $G _ { 4 }$ was sampled from 2023/07/23 to 2023/07/25. Positive samples include old users who played the game during the sampling period, while negative samples are those who had not played the game by the end of that period. The positive-to-negative ratio is 1 : 2 as well.

The test set was sampled from 2023/07/26 to 2023/07/28, consisting of millions of randomly selected users disjoint from the training and validation sets, labeled by whether they played the game during the following week. (iii) Attract new users for new games. This scenario refers to predicting whether a new user will play a new game. We select two popular games released in July 2023: Crystal of Atlan (G<sub>5</sub>) and Journey to West: Burning Soul $\left( G _ { 6 } \right)$

STATISTICS OF THE DATASETS FOR DOWNSTREAM TASKS.  
TABLE II
<table><tr><td></td><td># Training Set</td><td># Test Set</td><td>% Test Pos. Rate</td></tr><tr><td> $G _ { 1 }$ </td><td>236,609</td><td>4,995,000</td><td>0.018%</td></tr><tr><td> $G _ { 2 }$ </td><td>766,978</td><td>4,995,000</td><td>0.087%</td></tr><tr><td> $G _ { 3 }$ </td><td>72,923</td><td>4,995,000</td><td>0.003%</td></tr><tr><td> $G _ { 4 }$ </td><td>1,950,447</td><td>2,850,000</td><td>0.579%</td></tr><tr><td> $G _ { 5 }$ </td><td>675,033</td><td>4,985,000</td><td>0.219%</td></tr><tr><td> $G _ { 6 }$ </td><td>364,235</td><td>4,973,234</td><td>0.024%</td></tr></table>

The training sets for $G _ { 5 }$ and $G _ { 6 }$ were sampled from 2023/07/24 to 2023/07/25. The sampling rules are the same as those in (i) and (ii).

Table II records the statistical information for the downstream datasets mentioned above.

2) Baselines: We compare CM-PTM with representative generative pre-training methods and discriminative pretraining methods. (a) Generative User Representation Pretraining Methods. PeterRec [10] applies masked behavior prediction, and PTUM [11] applies masked behavior prediction and next-K-behavior prediction to the behavior sequence. BERT4Rec [17] uses bidirectional self-attention to model the behavior sequence. (b) Discriminative User Representation Pre-training Methods. CLUE [12] uses implicitly augmented views for contrastive pre-training. CCL [16] proposes a maskand-fill data augmentation strategy and uses contrastive learning to train the model. AdaptSSR [15] proposes an adaptive data augmentation self-supervised ranking method to model user representations. (c) Integrated User Representation Pretraining Method. MAFN [29] adopts both generative and discriminative methods to learn users’ long-term and shortterm interests. All baselines are provided with the same raw behavior data. For methods that do not support multi-source inputs, behaviors are chronologically merged into a single sequence.

3) Evaluation Metrics: We choose two widely used metrics, AUC and $\mathbf { R } @ \mathbf { P } _ { N } .$ , to evaluate the model performance. The first metric, AUC, is the area under the ROC curve. The second metric, R@P<sub>N</sub>, indicates the maximum recall achieved under the constraint that precision is at least N. In this work, we set $N = 0 . 9 5$

4) Implementation Details: Our pre-training model is implemented with PyTorch [55] under a Linux environment. Adam [56] is selected as the optimizer. Considering the results of Optuna optimization and previous works [15], [16], we set the embedding dimension d to 64 and the user representation dimension $d _ { u }$ to $6 \times 6 4$ . The batch size, learning rate, maximum length of each sequence l, number of attention heads $H ,$ and λ in Eq. (11) are set to 512, 1e−4, 256, 2, and $1 / 2 ,$ , respectively. During pre-training, the training and validation sets are split in a 9 : 1 ratio.

TABLE III  
OVERALL PERFORMANCE OF USER REPRESENTATIONS GENERATED BY VARIOUS PRE-TRAINING METHODS ON DOWNSTREAM TASKS.
<table><tr><td rowspan="2">Pre-training Method</td><td colspan="2"> $G _ { 1 }$ </td><td colspan="2"> $G _ { 2 }$ </td><td colspan="2"> $G _ { 3 }$ </td><td colspan="2"> $G _ { 4 }$ </td><td colspan="2"> $G _ { 5 }$ </td><td colspan="2"> $G _ { 6 }$ </td></tr><tr><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td><td>AUC</td><td> $\mathbf { R } @ \mathbf { P } _ { 0 . 9 5 }$ </td></tr><tr><td>None</td><td>0.7920</td><td>0.3164</td><td>0.7543</td><td>0.2382</td><td>0.8553</td><td>0.4268</td><td>0.8008</td><td>0.2569</td><td>0.8906</td><td>0.4698</td><td>0.9730</td><td>0.8546</td></tr><tr><td>Bert4Rec</td><td>0.8370</td><td>0.4579</td><td>0.8109</td><td>0.3836</td><td>0.8779</td><td>0.5786</td><td>0.8546</td><td>0.3485</td><td>0.8819</td><td>0.5054</td><td>0.9867</td><td>0.9034</td></tr><tr><td>PeterRec</td><td>0.8258</td><td>0.4681</td><td>0.8113</td><td>0.3376</td><td>0.8697</td><td>0.5792</td><td>0.8590</td><td>0.3607</td><td>0.8948</td><td>0.5103</td><td>0.9829</td><td>0.9201</td></tr><tr><td>PTUM</td><td>0.8235</td><td>0.4458</td><td>0.8126</td><td>0.3415</td><td>0.8676</td><td>0.5244</td><td>0.8263</td><td>0.2969</td><td>0.8911</td><td>0.4729</td><td>0.9822</td><td>0.8710</td></tr><tr><td>CLUE</td><td>0.8198</td><td>0.4476</td><td>0.8036</td><td>0.3791</td><td>0.8689</td><td>0.5183</td><td>0.8347</td><td>0.3126</td><td>0.8916</td><td>0.4793</td><td>0.9803</td><td>0.8839</td></tr><tr><td>CCL</td><td>0.8215</td><td>0.4563</td><td>0.8097</td><td>0.3814</td><td>0.8721</td><td>0.5253</td><td>0.8406</td><td>0.3263</td><td>0.8921</td><td>0.4907</td><td>0.9835</td><td>0.8864</td></tr><tr><td>AdaptSSR</td><td>0.8247</td><td>0.3969</td><td>0.8017</td><td>0.3152</td><td>0.8871</td><td>0.5548</td><td>0.8325</td><td>0.3236</td><td>0.8509</td><td>0.4252</td><td>0.9793</td><td>0.9190</td></tr><tr><td>MAFN</td><td>0.8366</td><td>0.4571</td><td>0.8064</td><td>0.3528</td><td>0.8679</td><td>0.5633</td><td>0.8367</td><td>0.3575</td><td>0.8859</td><td>0.4916</td><td>0.9856</td><td>0.9072</td></tr><tr><td>CM-PTM</td><td>0.8542</td><td>0.4829</td><td>0.8411</td><td>0.4091</td><td>0.8976</td><td>0.5915</td><td>0.8724</td><td>0.3817</td><td>0.9056</td><td>0.5176</td><td>0.9859</td><td>0.9288</td></tr><tr><td>Improvement</td><td>+1.72%</td><td>+1.48%</td><td>+2.85%</td><td>+2.55%</td><td>+1.05%</td><td>+1.23%</td><td>+1.34%</td><td>+2.10%</td><td>+1.08%</td><td>+0.73%</td><td>-0.08%</td><td>+0.87%</td></tr></table>

Bold font indicates the best-performing method. Underline indicates the second-best results. ‘None’ indicates that only the profile features are input to LightGBM, with no user representation provided. Improvements over the best baselines are significant (paired t-test over 5 runs, p-value < 0.05).

## B. Overall Performance on Downstream Tasks

Table III presents the performance comparison results.

First, for $G _ { 1 }$ to $G _ { 5 } ,$ , CM-PTM significantly outperforms all baseline models, achieving an AUC improvement of 1.05% to 2.85%. This gain can be attributed to the synergy between the cross-source cascaded prediction framework and the hierarchical attention mechanism.

Second, generative pre-training methods generally outperform discriminative counterparts. This is because generative methods directly model the behavior generation process, encouraging the learning of temporal dependencies across heterogeneous or homogeneous sources. In contrast, discriminative methods typically rely on behavior perturbation, e.g., masking or replacing actions, which may disrupt cross-source correlations, such as substituting an AppStore search with a GameApp launch, or distort key behavioral intents, replacing installation with uninstallation as an example. These are conjectured to be possible reasons for degraded intent representation and prediction accuracy.

Finally, we observe that CM-PTM’s AUC is suboptimal for $G _ { 6 } .$ However, its $\mathrm { R @ P _ { 0 . 9 5 } }$ performance is still far ahead, indicating a better ability to identify potential positive samples. This ability is even more crucial in practical new game recommendation scenarios, as new games need to attract sufficient positive users to build momentum and reputation.

On the whole, the results demonstrate the outstanding performance of CM-PTM in various game scenarios, attributed to its effective capability to capture user intentions across different sources.

## C. Ablation Study

Next, we conduct an ablation study to dissect the contributions of different components of CM-PTM, categorized into three groups: data ablation, sequence ablation, and task ablation. Table IV presents results on both $G _ { 1 }$ and $G _ { 5 }$

(i) Data ablation. To examine the importance of jointly modeling multi-granularity behaviors from multiple sources, we selectively input either fine-grained or coarse-grained behavior data while keeping all other settings fixed. As shown in

TABLE IV  
THE AUC AND R@ $\mathrm { { P _ { 0 . 9 5 } } }$ PERFORMANCE WITH DIFFERENT COMPONENTS REMOVED. THE BEST RESULTS ARE IN BOLD.
<table><tr><td rowspan="2">Ablation Group</td><td rowspan="2">Variants</td><td colspan="2"> $G _ { 1 }$ </td><td colspan="2"> $G _ { 5 }$ </td></tr><tr><td>AUC</td><td> $\overline { { { \bf R } ( \omega { \bf P } _ { 0 . 9 5 } } }$ </td><td>AUC</td><td> $\overline { { { \bf R } \ @ { \bf P } _ { 0 . 9 5 } } }$ </td></tr><tr><td>None</td><td>CM-PTM</td><td>0.8542</td><td>0.4829</td><td>0.8956</td><td>0.5176</td></tr><tr><td rowspan="2">Data</td><td>w/o fine-grained</td><td>0.8485</td><td>0.4723</td><td>0.8728</td><td>0.4404</td></tr><tr><td>w/o coarse-grained</td><td>0.8278</td><td>0.4388</td><td>0.8717</td><td>0.4413</td></tr><tr><td>Sequence</td><td>w/o multi-seq</td><td>0.8366</td><td>0.4539</td><td>0.8804</td><td>0.4547</td></tr><tr><td rowspan="5">Task</td><td>w/o  $T _ { 1 }$ </td><td>0.8490</td><td>0.4775</td><td>0.8769</td><td>0.4440</td></tr><tr><td>w/o  $T _ { 2 }$ </td><td>0.8515</td><td>0.4796</td><td>0.8934</td><td>0.4946</td></tr><tr><td>w/o  $T _ { 3 }$ </td><td>0.8477</td><td>0.4664</td><td>0.8919</td><td>0.4861</td></tr><tr><td>w/o  $T _ { 4 }$ </td><td>0.8534</td><td>0.4818</td><td>0.8825</td><td>0.4476</td></tr><tr><td>w/o cascaded</td><td>0.8307</td><td>0.4435</td><td>0.8726</td><td>0.4393</td></tr></table>

Table IV, using only fine-grained behaviors decreases AUC by 2.64 and 2.39 percentage points on $G _ { 1 }$ and $G _ { 5 } ,$ , respectively, while using only coarse-grained behaviors decreases AUC by 0.57 and 2.28 percentage points. These results robustly demonstrate that combining behaviors of different granularities from multiple sources is essential for accurately capturing user representations in mobile device scenarios.

(ii) Sequence ablation. To investigate the limitations of single-sequence modeling, we construct a unified behavior sequence by chronologically merging all available behaviors and employ self-attention as the backbone model. The AUC drops by 1.76% on $G _ { 1 }$ and 1.52% on $G _ { 5 }$ , highlighting the inadequacy of conventional single-sequence approaches [12], [15], [29] in capturing the complexity of multi-granularity, cross-source behaviors in mobile environments.

(iii) Task ablation. To validate the effectiveness of our cascaded prediction framework, we remove each sub-task $T _ { i }$ from the task set $( T _ { 1 } ~ \sim ~ T _ { 4 } ,$ denoted w/o T ) or replace the cascaded design with parallel predictions (w/o cascaded) while keeping all other settings unchanged. As reported in Table IV, T<sub>1</sub> contributes substantially, improving AUC by 0.52% and 1.87% for $G _ { 1 }$ and $G _ { 5 } ,$ respectively. For old games, among the four tasks, the contribution of $T _ { 1 }$ ranks second only to App ID prediction (T<sub>3</sub>), whose importance in user behavior modeling is self-evident. For new games, among the four tasks, removing $T _ { 1 }$ leads to the largest performance decrease. Moreover, each sub-task as well as their cascade relationships contribute positively to the overall performance. These results support the rationale of our approach to capture cross-source intent through the hierarchical decision structure, especially behavior source prediction. The cascaded design consistently outperforms the parallel alternative, demonstrating that progressively conditioning predictions on prior stages effectively captures cross-source dependencies across heterogeneous behavior streams.

![](images/0329fb947b8a3ebb459c044b9589813b4c633f8a5d0720a00bc2382eefba900f.jpg)  
Fig. 3. Two case study users and part of their behaviors.

## D. Case Study

We present two representative case studies in Figure 3. User A was correctly identified as a positive sample. From 2023/01/26 to 2023/07/24, this user had interacted with three sandbox games in third-party GameApps: Terraria, Mini World, and Minecraft. Due to the strong preference for sandbox games, with only one click on Honor ofKings in Appstore, our model successfully captured their interest in Honor of Kings. This case illustrates the usefulness of cross-source intent modeling in mobile device environments.

User B was correctly identified as a negative sample. This user developed a strong interest in Crystal of Atlan, searching for and downloading it on its release day (2023/07/13). After the similar game Journey to West: Burning Soul was released on 2023/07/22, this user searched for it without downloading it, returning to play Crystal of Atlan. Through cross-source analysis, our model successfully identified the correlations between these behaviors, and then inferred that this user had limited interest in Journey to West: Burning Soul. Notably, all baseline models produced false-positive predictions for this user, highlighting the superiority of our model in capturing cross-source user intent.

## E. Effects of Hyper-parameters

We set a hyper-parameter λ in Section IV-C to explore the relative importance of inter-source phase and intra-source phase. In this section, we select four downstream tasks to investigate the effects of different values of λ. Figure 4 illustrates the results. For attracting new users, whether for an old or new game, setting $\lambda = 1 / 2$ performs best among the tested values, indicating that it is necessary to balance the optimization of the two phases. However, for old users, placing different emphasis on each phase may yield better outcomes. This is because old users have already formed their personal preferences: some users may be accustomed to downloading immediately after searching in the Appstore (stable crosssource associations), while others may continuously browse related recommendations after searching and only download when they find something satisfying (uncertain cross-source associations). Therefore, the model needs to prioritize the two phases to adapt to their established preferences.

TABLE V  
ONLINE A/B TESTING RESULTS.
<table><tr><td>Method</td><td>AUC</td><td>R@P0.95</td></tr><tr><td>DeepFM</td><td>0.8064</td><td>0.3549</td></tr><tr><td>DeepFM + CM-PTM</td><td>0.8192</td><td>0.3656</td></tr><tr><td>Improvement</td><td>+1.28%</td><td>+1.07%</td></tr></table>

TABLE VI  
OFFLINE TRAINING TIME FOR ALL COMPARED METHODS.
<table><tr><td>Method</td><td>CM-PTM</td><td>Bert4Rec</td><td>PeterRec</td><td>PTUM</td></tr><tr><td>Training time (h)</td><td>5.74</td><td>3.86</td><td>4.32</td><td>3.31</td></tr><tr><td>Method</td><td>CLUE</td><td>CCL</td><td>AdaptSSR</td><td>MAFN</td></tr><tr><td>Training time (h)</td><td>3.02</td><td>2.87</td><td>3.24</td><td>6.53</td></tr></table>

## F. Online A/B Test

We also conduct an online A/B test on OPPO’s mobile game recommendation system, and the results are shown in Table V. Before deployment, CM-PTM was retrained on fresh, recent streaming behavioral data to reflect the latest user activity patterns. We randomly sampled 2.89 million active users over a two-week period (ranging from 2025/01/05 to 2025/01/18) and split test traffic evenly by user ID between the two models to compare our CM-PTM against the baseline model DeepFM [57]. Compared to DeepFM, our pre-training method improves AUC and $\mathrm { R @ P _ { 0 . 9 5 } }$ by 1.28% and 1.07%, respectively, which validates the practicality of our method in real industrial scenarios on OPPO devices.

## G. Efficiency Comparison

We finally compare the training time of CM-PTM with other baselines and analyze its online inference efficiency.

For offline training, all methods are pre-trained on two NVIDIA A100 GPUs. As shown in Table VI, CM-PTM requires more training time than most generative baselines but remains below MAFN and within an acceptable industrial training budget. This is because we adopt several strategies to reduce computational complexity: (1) no separate test set is constructed during pre-training; (2) the operation prediction task $( T _ { 4 } )$ is skipped for Appstore and Gamecenter behaviors; and (3) although each source-specific embedding ${ \pmb u } ^ { k } \ \left( k \ = \right.$ 1, 2, 3) is pre-trained independently, its dimensionality is set to only 1/3 of $d _ { u }$ , introducing little additional computational burden. These strategies allow CM-PTM to maintain competitive training efficiency while modeling multi-source, multigranular behaviors.

![](images/24043e7dc9a366735e82300b26a3a46763de9ad34514b736ec489ca6b30ac6a2.jpg)  
(a) $G _ { 2 }$

![](images/8326e995e8d0caaad53d76d090e0bcfc590c26de1d861c924b6075c1d57c484c.jpg)  
(b) $G _ { 3 }$

![](images/eb4fbf9f0c029eb7461b288b63a561e98367061544ae7b99cfdfc8f612b270ae.jpg)  
(c) $G _ { 4 }$

![](images/de2fe3f9a724c9ed02f62d7e748dd3bc213c79236457e1f0f1c8ccfca3c867d7.jpg)  
(d) $G _ { 5 }$  
Fig. 4. The impact of different values of λ.

For online inference, CM-PTM augments the downstream representations by only $d _ { u }$ dimensions, which is negligible compared to the existing thousand-dimensional features, yet consistently delivers substantial performance gains. This demonstrates that the model introduces minimal overhead in real-time inference.

In summary, CM-PTM incurs higher training cost than most generative baselines yet remains within an acceptable industrial budget, while introducing minimal inference overhead and providing significant improvements in downstream effectiveness, highlighting its practicality for large-scale industrial deployment.

## VI. CONCLUSION

In this paper, we propose CM-PTM, a novel pre-training framework for learning device-level user representations for mobile game recommendation from multi-source, multigranular behavioral data on mobile devices. Inspired by a hierarchical decision structure for progressive intent refinement, CM-PTM decomposes behavior modeling into cascaded proxy tasks that capture both inter-source dependencies and finegrained intra-source patterns in a progressive manner. Extensive experiments on large-scale industrial datasets from OPPO for mobile game recommendation demonstrate that CM-PTM consistently outperforms strong baselines in offline evaluations and online A/B tests, highlighting its effectiveness in capturing cross-source user intent and its practical applicability in realworld OPPO recommendation scenarios. While our current model relies on positional embeddings to capture relative sequential order within short-term windows, explicitly incorporating time-aware mechanisms (e.g., temporal encoding of absolute time intervals) is a promising direction for future work. Our work provides a foundation for future research in device-level user representation pre-training for mobile games.

## ETHICS STATEMENT

This study was conducted in collaboration with OPPO Research Institute. The study was approved by OPPO’s internal data governance board. Data were anonymized before modeling, and no raw textual search queries or content-level features were used. All data were collected with explicit user consent during service initialization. The dataset is restricted to three behavior sources at the mobile-device level: thirdparty GameApps, where only coarse-grained external signals such as installation, uninstallation, and launching are utilized, excluding sensitive in-app content and fine-grained interaction logs; App Store and Game Center, where data usage is limited to search and click-through events within these system applications. We only model event types and anonymized identifiers and deliberately adopt a design based on aggregated high-level cross-source signals to minimize privacy risks.

## ACKNOWLEDGMENTS

The research work is supported by the National Natural Science Foundation of China under Grant Nos. 62576333, 62406307, and U2436209, the Strategic Priority Research Program of the Chinese Academy of Sciences under Grant No. XDB0680201, the Beijing Natural Science Foundation (F251001), and the Innovation Funding of ICT, CAS under Grant No. E461060.

## REFERENCES

[1] Y. Ni, D. Ou, S. Liu, X. Li, W. Ou, A. Zeng, and L. Si, “Perceive your users in depth: Learning universal user representations from multiple e-commerce tasks,” in KDD, 2018, pp. 596–605.

[2] J. Tang, L. Yao, D. Zhang, and J. Zhang, “A combination approach to web user profiling,” ACM TKDD, vol. 5, no. 1, pp. 1–44, 2010.

[3] L. Zheng, C. Li, C.-T. Lu, J. Zhang, and P. S. Yu, “Deep distribution network: Addressing the data sparsity issue for top-n recommendation,” in SIGIR, 2019, pp. 1081–1084.

[4] C. Liu, Y. Gao, L. Sun, J. Feng, H. Yang, and X. Ao, “User behavior pre-training for online fraud detection,” in KDD, 2022, pp. 3357–3365.

[5] H. Liu, J. Lu, X. Zhao, S. Xu, H. Peng, Y. Liu, Z. Zhang, J. Li, J. Jin, Y. Bao et al., “Kalman filtering attention for user behavior modeling in ctr prediction,” NeurIPS, vol. 33, pp. 9228–9238, 2020.

[6] T. Gao, X. Yao, and D. Chen, “Simcse: Simple contrastive learning of sentence embeddings,” arXiv preprint arXiv:2104.08821, 2021.

[7] L. Wu, J. Li, Y. Wang, Q. Meng, T. Qin, W. Chen, M. Zhang, T.-Y. Liu et al., “R-drop: Regularized dropout for neural networks,” NeurIPS, vol. 34, pp. 10 890–10 905, 2021.

[8] X. Chen and K. He, “Exploring simple siamese representation learning,” in CVPR, 2021, pp. 15 750–15 758.

[9] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin, “Unsupervised learning of visual features by contrasting cluster assignments,” NeurIPS, vol. 33, pp. 9912–9924, 2020.

[10] F. Yuan, X. He, A. Karatzoglou, and L. Zhang, “Parameter-efficient transfer from sequential behaviors for user modeling and recommendation,” in SIGIR, 2020, pp. 1469–1478.

[11] C. Wu, F. Wu, T. Qi, J. Lian, Y. Huang, and X. Xie, “Ptum: Pre-training user model from unlabeled user behaviors via self-supervision,” arXiv preprint arXiv:2010.01494, 2020.

[12] M. Cheng, F. Yuan, Q. Liu, X. Xin, and E. Chen, “Learning transferable user representations with sequential behaviors via contrastive pre-training,” in ICDM. IEEE, 2021, pp. 51–60.

[13] X. Xie, F. Sun, Z. Liu, S. Wu, J. Gao, J. Zhang, B. Ding, and B. Cui, “Contrastive learning for sequential recommendation,” in ICDE, 2022, pp. 1259–1273.

[14] Z. Liu, Y. Chen, J. Li, P. S. Yu, J. McAuley, and C. Xiong, “Contrastive self-supervised sequential recommendation with robust augmentation,” arXiv preprint arXiv:2108.06479, 2021.

[15] Y. Yu, Q. Liu, K. Zhang, Y. Zhang, C. Song, M. Hou, Y. Yuan, Z. Ye, Z. Zhang, and S. L. Yu, “Adaptssr: Pre-training user model with augmentation-adaptive self-supervised ranking,” arXiv preprint arXiv:2310.09706, 2023.

[16] S. Bian, W. X. Zhao, K. Zhou, J. Cai, Y. He, C. Yin, and J.-R. Wen, “Contrastive curriculum learning for sequential user behavior modeling via data augmentation,” in CIKM, 2021, pp. 3737–3746.

[17] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang, “Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer,” in CIKM, 2019, pp. 1441–1450.

[18] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[19] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le, “Xlnet: Generalized autoregressive pretraining for language understanding,” NeurIPS, vol. 32, 2019.

[20] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, “Deep contextualized word representations. arxiv preprint,” arXiv preprint arXiv:1802.05365, 2018.

[21] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in ICML. PMLR, 2020, pp. 1691– 1703.

[22] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in CVPR, 2022, pp. 16 000– 16 009.

[23] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

[24] X. Chen, D. Liu, C. Lei, R. Li, Z.-J. Zha, and Z. Xiong, “Bert4sessrec: Content-based video relevance prediction with bidirectional encoder representations from transformer,” in ACM MM, 2019, pp. 2597–2601.

[25] C. Xiao, R. Xie, Y. Yao, Z. Liu, M. Sun, X. Zhang, and L. Lin, “Uprec: User-aware pre-training for recommender systems,” arXiv preprint arXiv:2102.10989, 2021.

[26] Z. Qiu, X. Wu, J. Gao, and W. Fan, “U-bert: Pre-training user representations for improved recommendation,” in AAAI, vol. 35, no. 5, 2021, pp. 4320–4327.

[27] L. Chen, F. Yuan, J. Yang, X. He, C. Li, and M. Yang, “User-specific adaptive fine-tuning for cross-domain recommendations,” IEEE TKDE, 2021.

[28] C. Fu, W. Wu, X. Zhang, J. Hu, J. Wang, and J. Zhou, “Robust user behavioral sequence representation via multi-scale stochastic distribution prediction,” in CIKM, 2023, pp. 4567–4573.

[29] Y. Zhang, M. Hou, K. Zhang, Y. Yuan, C. Song, Z. Ye, E. Chen, and Y. Yu, “Pre-training general user representation with multi-type app behaviors,” in IJCAI, 2024, pp. 5535–5544.

[30] J. Zuo, J. Hong, F. Zhang, C. Yu, H. Zhou, C. Gao, N. Sang, and J. Wang, “Plip: Language-image pre-training for person representation learning,” NeurIPS, vol. 37, pp. 45 666–45 702, 2024.

[31] X. Lin, J. Luo, J. Pan, W. Pan, Z. Ming, X. Liu, S. Huang, and J. Jiang, “Multi-sequence attentive user representation learning for sideinformation integrated sequential recommendation,” in WSDM, 2024, pp. 414–423.

[32] Q. Sun, J. Gu, X. Xu, R. Xu, K. Liu, B. Yang, H. Liu, and H. Xu, “Learning interest-oriented universal user representation via self-supervision,” in ACM MM, 2022, pp. 7270–7278.

[33] R. Qiu, Z. Huang, H. Yin, and Z. Wang, “Contrastive learning for representation degeneration problem in sequential recommendation,” in WSDM, 2022, pp. 813–823.

[34] S. Luo, Y. Xiao, X. Zhang, Y. Liu, W. Ding, and L. Song, “Perfedrec++: Enhancing personalized federated recommendation with self-supervised pre-training,” ACM TIST, vol. 15, no. 5, pp. 1–24, 2024.

[35] C. Zhou, J. Bai, J. Song, X. Liu, Z. Zhao, X. Chen, and J. Gao, “Atrank: An attention-based user behavior modeling framework for recommendation,” in AAAI, vol. 32, no. 1, 2018.

[36] W. Lin, L. Sun, Q. Zhong, C. Liu, J. Feng, X. Ao, and H. Yang, “Online credit payment fraud detection via structure-aware hierarchical recurrent neural network.” in IJCAI, 2021, pp. 3670–3676.

[37] W. Wang, W. Zhang, S. Liu, Q. Liu, B. Zhang, L. Lin, and H. Zha, “Beyond clicks: Modeling multi-relational item graph for session-based target behavior prediction,” in WWW, 2020, pp. 3056–3062.

[38] B. Jin, C. Gao, X. He, D. Jin, and Y. Li, “Multi-behavior recommendation with graph convolutional networks,” in SIGIR, 2020, pp. 659–668.

[39] X. Zhao, Y. Wang, B. Chen, J. Gao, Y. Wang, X. Li, P. Jia, Q. Liu, H. Guo, and R. Tang, “Joint modeling in recommendations: A survey,” arXiv preprint arXiv:2502.21195, 2025.

[40] Y. Wu, R. Xie, Y. Zhu, X. Ao, X. Chen, X. Zhang, F. Zhuang, L. Lin, and Q. He, “Multi-view multi-behavior contrastive learning in recommendation,” in DASFAA. Springer, 2022, pp. 166–182.

[41] J. Xu, C. Wang, C. Wu, Y. Song, K. Zheng, X. Wang, C. Wang, G. Zhou, and K. Gai, “Multi-behavior self-supervised learning for recommendation,” in SIGIR, 2023, pp. 496–505.

[42] L. Zhang, W. Zhang, L. Wu, M. He, and H. Zhao, “Shgcn: Socially enhanced heterogeneous graph convolutional network for multi-behavior prediction,” ACM Transactions on the Web, vol. 18, no. 1, pp. 1–27, 2023.

[43] S. Elsayed, A. Rashed, and L. Schmidt-Thieme, “Multi-behavioral sequential recommendation,” in RecSys, 2024, pp. 902–906.

[44] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” NeurIPS, vol. 30, 2017.

[45] Z. Shao, S. Wang, W. Lu, W. Zhang, H. Guan, and L. Zhao, “Filterenhanced hypergraph transformer for multi-behavior sequential recommendation,” in ICASSP. IEEE, 2024, pp. 6575–6579.

[46] H. Zhang, M. Cheng, Q. Liu, J. Jiang, X. Wang, R. Zhang, C. Lei, and E. Chen, “A comprehensive survey on cross-domain recommendation: Taxonomy, progress, and prospects,” arXiv preprint arXiv:2503.14110, 2025.

[47] Y.-C. Chen and W.-C. Lee, “A novel cross-domain recommendation with evolution learning,” ACM Trans. Internet Technol., vol. 24, no. 1, pp. 1–23, 2024.

[48] L. Guo, L. Tang, T. Chen, L. Zhu, Q. V. H. Nguyen, and H. Yin, “Dagcn: A domain-aware attentive graph convolution network for sharedaccount cross-domain sequential recommendation,” arXiv preprint arXiv:2105.03300, 2021.

[49] Y. Li, L. Hou, and J. Li, “Preference-aware graph attention networks for cross-domain recommendations with collaborative knowledge graph,” ACM TOIS, vol. 41, no. 3, pp. 1–26, 2023.

[50] Z. Yang, Z. Peng, Z. Wang, J. Qi, C. Chen, W. Pan, C. Wen, C. Wang, and X. Fan, “Federated graph learning for cross-domain recommendation,” arXiv preprint arXiv:2410.08249, 2024.

[51] R. Liang, Q. Zhang, J. Wang, and J. Lu, “A hierarchical attention network for cross-domain group recommendation,” IEEE TNNLS, vol. 35, no. 3, pp. 3859–3873, 2022.

[52] Y. Jiang, Q. Li, H. Zhu, J. Yu, J. Li, Z. Xu, H. Dong, and B. Zheng, “Adaptive domain interest network for multi-domain recommendation,” in CIKM, 2022, pp. 3212–3221.

[53] H. Ma, R. Xie, L. Meng, X. Chen, X. Zhang, L. Lin, and J. Zhou, “Triple sequence learning for cross-domain recommendation,” ACM TOIS, vol. 42, no. 4, pp. 1–29, 2024.

[54] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.- Y. Liu, “Lightgbm: A highly efficient gradient boosting decision tree,” NeurIPS, vol. 30, 2017.

[55] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” NeurIPS, vol. 32, 2019.

[56] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[57] H. Guo, R. Tang, Y. Ye, Z. Li, and X. He, “Deepfm: a factorizationmachine based neural network for ctr prediction,” arXiv preprint arXiv:1703.04247, 2017.