# Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation

Jinwu Hu<sup>∗</sup>, Langyu Bian<sup>∗</sup>, Jian Sun, Dongjin Yang, Chao Shuai, Yufeng Wang, Kai Zhou, Bin Xiao<sup>†</sup>, Mingkui Tan<sup>†</sup>, Member, IEEE, Hongmin Cai<sup>†</sup>, Senior Member, IEEE, Wei Luo<sup>†</sup>

Abstract—Open-set test-time adaptation (TTA) aims to adapt source-trained models to unlabeled test streams where covariateshifted in-distribution (csID) samples and covariate-shifted outof-distribution (csOOD) samples may coexist. Existing TTA methods are typically designed under a closed-set assumption and therefore lack explicit mechanisms for reliably distinguishing csID from csOOD samples. Consequently, csOOD samples may be mistakenly included in confidence-enhancing adaptation, producing harmful gradients that degrade known-class performance. To address this, we propose Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation (UAPA), a plug-and-play method for reliable open-set adaptation. Specifically, we first introduce Uncertainty-Driven ID/OOD Separation, which combines a classifier-prototype score with a mini-batchnormalized evidence-derived uncertainty score to distinguish csID from csOOD samples. We then propose a Dynamic Prototype Refinement strategy that periodically refines class-specific target prototypes using buffered csID features to progressively follow the estimated class directions. Lastly, we introduce Contrastive Prototype Entropy Adaptation, which applies entropy minimization to csID samples and entropy maximization to csOOD samples, while prototype-level contrastive learning aligns csID features with their corresponding target prototypes and reduces the maximum similarity between csOOD features and knownclass prototypes. Empirically, UAPA outperforms the strongest baselines on the CIFAR benchmarks and Tiny-ImageNet-C, improving average OSCR by 2.08 and 1.28 percentage points, respectively.

Index Terms—Open-set recognition, out-of-distribution detection, prototype learning, test-time adaptation

## I. INTRODUCTION

D <sup>EEP</sup> <sup>neural</sup> <sup>networks</sup> <sup>have</sup> <sup>achieved</sup> <sup>remarkable</sup> <sup>progress</sup>in visual recognition tasks [1–6], yet their reliability in visual recognition tasks [1-6], yet their reliability can substantially degrade when deployed in open-world environments [7–10]. In real-world test streams, known-class samples may undergo appearance-level changes caused by weather variations, motion blur, sensor noise, or other visual corruptions while retaining the source label space, a setting known as covariate shift [11]. Meanwhile, test samples may also come from classes outside the source label space, leading to semantic shift [12]. In practice, these two shifts are often coupled: unknown-class samples may appear under the same degraded conditions as shifted known-class samples. For example, an autonomous driving system may encounter familiar vehicles under fog or motion blur, while also observing unseen obstacles or rare objects that were absent during training. A reliable deployed model is therefore expected not only to adapt to shifted known classes, but also to avoid being misled by shifted unknown classes [13–17].

![](images/2361ca5061c4542bd53bbfee1206d52ea0d91c504a07a136c007cb3dbc87c20d.jpg)  
Fig. 1: OSCR comparison on CIFAR-100-C before and after incorporating UAPA into representative TTA methods [18– 21]. Values above the arrows denote absolute OSCR gains in percentage points.

To improve robustness under covariate shift, test-time adaptation (TTA) updates a source-trained model online using unlabeled test data [18, 19, 21–23]. Representative TTA methods combine unsupervised adaptation objectives, such as entropy minimization [18, 19] or prediction consistency [23, 24], with update strategies such as reliability-based sample selection [19, 22]. These designs enable the model to adapt to shifted test distributions without requiring access to the original source data. However, many TTA methods are developed under an implicit closed-set assumption, where each test sample is assumed to belong to one of the source classes despite possible changes in the test distribution. Under this assumption, encouraging confident predictions or selecting reliable samples can help the model adapt to shifted known-class samples.

In open-set test streams, this closed-set assumption no longer holds because the target stream may contain both known-class samples with appearance or quality shifts, referred to as covariate-shifted in-distribution (csID) samples, and unknown-class samples outside the source label space under similar shifts, referred to as covariate-shifted out-ofdistribution (csOOD) samples [15, 16, 25]. This setting raises two key challenges. 1) Negative adaptation caused by csOOD samples. Entropy minimization encourages confident predictions for target samples. If csOOD samples are adapted in the same way as csID samples, they may be pushed toward overconfident known-class predictions, yielding harmful gradients that distort decision boundaries and cause negative adaptation. 2) Unreliable separation between csID and csOOD samples. The model must decide which samples can support known-class adaptation and which should be excluded from confidence-enhancing updates. This is difficult because severe covariate shifts can make csID samples uncertain, whereas some csOOD samples may appear confident. Thus, relying on a single confidence, entropy, or prototype-similarity score [15] can produce unreliable partitions and unstable adaptation.

To address these challenges, we propose Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation (UAPA), a plug-and-play method (see Fig. 1) for reliabilityaware adaptation in open-set test streams. The key idea of UAPA is to determine the adaptation role of each test sample according to both its geometric relation to known classes and an evidence-derived uncertainty score computed from the online-model logits. Specifically, UAPA first introduces Uncertainty-Driven ID/OOD Separation, which combines a classifier-prototype score with a mini-batch-normalized evidence-derived uncertainty score to partition incoming samples into csID and csOOD subsets. It then performs Dynamic Prototype Refinement (DPR), which accumulates selected csID features in class-wise buffers and periodically updates class-specific target prototypes. Finally, UAPA develops Contrastive Prototype Entropy Adaptation (CPEA), which applies different entropy objectives to the estimated csID and csOOD subsets and uses progressively initialized target prototypes to provide stage-dependent, prototype-relative regularization. In this way, UAPA jointly performs sample separation, prototype maintenance, and direction-aware adaptation to suppress negative adaptation in open-set test-time adaptation. Our main contributions are as follows:

• We propose an uncertainty-driven ID/OOD separation mechanism for open-set test-time adaptation. By jointly considering a classifier-prototype score and a mini-batchnormalized evidence-derived uncertainty score, UAPA incorporates complementary geometric and uncertaintyrelated cues for csID/csOOD partitioning. We characterize the conditional separability and bounded-perturbation stability of the fused score, while clarifying the minibatch dependence of the normalized uncertainty score and its distinction from GMM-based posterior gating.

• We propose Dynamic Prototype Refinement and Contrastive Prototype Entropy Adaptation for open-set testtime adaptation. DPR periodically updates class-specific target prototypes using buffered csID features, while CPEA combines csID entropy minimization, csOOD entropy maximization, and prototype-level contrastive learning to suppress negative adaptation. We show that DPR induces one-step angular non-expansion toward the buffered class direction, while CPEA encourages pseudolabel-relative margin enlargement for selected csID samples and nearest-prototype separation for selected csOOD samples in the hyperspherical feature space.

• Extensive experiments on CIFAR-C and Tiny-ImageNet-C benchmarks demonstrate that UAPA consistently improves open-set classification performance across different backbones and state-of-the-art TTA baselines. Further plug-and-play and efficiency analyses show that UAPA can be integrated into existing TTA methods with low additional computational cost.

## II. RELATED WORK

## A. Test-Time Adaptation

Test-time adaptation (TTA) [17–19, 21, 22, 26–32] adapts a source-trained model during inference without accessing source data, using lightweight online updates that often restrict learning to normalization statistics or a small subset of parameters. It differs from test-time training (TTT), which performs an auxiliary self-supervised update before prediction and requires training-time modifications [33]. A foundational approach is entropy minimization, as in TENT [18], which updates the model batch by batch to increase prediction confidence under covariate or domain shift. EATA [19] selects reliable, non-redundant samples to avoid harmful updates and forgetting. COME [21] replaces plain entropy minimization with a conservative variant grounded in evidential or Dirichlet uncertainty to curb overconfidence. Self-Bootstrapping [26] proposes a versatile consistency objective that extends TTA beyond classification to tasks such as segmentation and detection. A<sup>2</sup>Memory [17] improves memory-based online TTA for vision-language models via attribute-centric soft assignment and class-wise associative memory, reducing memory contamination and enabling sample-adaptive prototype retrieval. Despite these advances, most TTA methods are designed for closed-set covariate or domain shifts, where test samples are assumed to belong to the source label space.

## B. Open-Set Test-Time Adaptation

Open-set test-time adaptation (OSTTA) [15, 20, 34–37] considers target streams that include unknown classes in addition to covariate-shifted known classes. Closed-set TTA often fails in this setting because uniform entropy minimization treats unknown inputs as trustworthy, which pushes them toward over-confident known predictions and degrades the decision boundary [20]. A common strategy is to separate target samples into covariate-shifted in-distribution (csID) and covariateshifted out-of-distribution (csOOD) and to assign different updates. UniEnt [15] performs mixture-based posterior gating on a prototype-similarity or uncertainty score to mine pseudo-csID and pseudo-csOOD, then minimizes entropy for the former and maximizes entropy for the latter, with a weighted variant to reduce partition noise. STAMP [34] avoids risky mini-batch updates by optimizing over a stable memory bank that is refreshed with low-entropy and label-consistent samples, improving robustness under streaming conditions. AEO [35] proposes adaptive entropy-aware optimization for multimodal OSTTA, enlarging the entropy gap between known and unknown samples to better identify unknown classes under multimodal distribution shifts. ROSETTA [16] studies the ID-OOD trade-off in OSTTA and aims to improve csOOD detection while preserving csID classification performance.

## C. Out-of-Distribution Detection

Out-of-distribution (OOD) detection aims to identify samples that deviate from the training distribution, preventing overconfident predictions on unseen or abnormal inputs [38– 43]. Maximum Softmax Probability (MSP) [38] uses the maximum softmax probability as a confidence score, providing a baseline for detecting misclassified and OOD samples. ODIN [39] introduces temperature scaling and input perturbations to enlarge the confidence gap between in-distribution and OOD samples, improving the reliability of softmax-based detection. Energy-based OOD detection [40] replaces softmax confidence with an energy score derived from logits, addressing the limitation that softmax probabilities may be poorly calibrated for OOD inputs. PRO [44] leverages perturbation robustness to construct more discriminative OOD scores, based on the observation that OOD confidence is more sensitive to adversarial perturbations. W-DOE [42] improves openworld OOD generalization by expanding auxiliary OOD coverage through implicit data synthesis and distribution-agnostic optimization. Unlike conventional OOD detection methods that mainly focus on post-hoc scoring for a fixed model, UAPA integrates OOD estimation with online adaptation, using prototype-based similarity and a mini-batch-normalized evidence-derived uncertainty score to separate csID/csOOD samples and suppress csOOD-induced negative adaptation.

## D. Contrastive Learning

Contrastive learning learns discriminative representations by pulling semantically similar samples closer and pushing dissimilar samples apart in the embedding space [45–52]. Sim-CLR [48] demonstrates that strong data augmentation, large batch sizes, and nonlinear projection heads enable contrastive representation learning. MoCo [49] introduces a momentum encoder and a dynamic queue to maintain a large and consistent dictionary of negative samples, alleviating the dependence on extremely large batches. Beyond negative-sample-based methods, BYOL [50] and SimSiam [51] adopt asymmetric Siamese architectures with prediction heads and stopgradient operations, enabling representation learning without explicit negative pairs. ReSSL [47] extends instance-level contrastive learning by modeling inter-instance relations through similarity-distribution matching under weak augmentations. CHMatch [52] combines robust adaptive thresholding with contrastive hierarchical matching to select reliable samples and learn discriminative representations.

## III. PROBLEM FORMULATION

Let $P ( x )$ denote the distribution of the source training samples $\{ x _ { i } ^ { s } \} _ { i = 1 } ^ { N _ { s } }$ , where $x _ { i } ^ { s } \sim P ( x )$ . The base model $f _ { \theta ^ { \circ } } ( x )$ is trained on the labeled source dataset $\mathcal { D } _ { s } = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } } \subset$ $\mathcal { X } \times \mathcal { Y } _ { s }$ , where $\mathcal { V } _ { s } = \{ 1 , \dots , \mathcal { C } _ { S } \}$ is the set of $\mathcal { C } _ { S }$ known classes. After training, the model $f _ { \theta ^ { \circ } } ( x )$ is expected to fit the source data distribution. At inference time, the model is expected to perform well on in-distribution test data, namely, $x \ \sim \ P ( x )$ . However, in real-world deployments, due to possible distribution shifts between training and test data, we may encounter target samples drawn from a shifted distribution $Q ( x )$ , where $Q ( x ) \neq P ( x )$ . The types of distribution shifts that can occur during inference are as follows. 1) Covariate $S h i f t .$ The samples share the same set of classes but differ in their distribution due to variations in style or domain. 2) Semantic Shift: The target domain contains novel classes that were not present during training on the source domain. In this case, the source label space is a strict subset of the target label space, $i . e . , \mathcal { V } _ { s } \subset \mathcal { V } _ { t }$ , where $\mathcal { V } _ { t }$ denotes the target label space. These shifts partition the target data stream into two categories: 1) Covariate-shifted In-Distribution (csID). Samples $x ^ { t } \in B _ { t }$ whose true labels $y ^ { t }$ belong to the known source classes, i.e., $y ^ { t } \in \mathcal { V } _ { s } . \ 2 )$ Covariate-shifted Out-of-Distribution (csOOD). Samples $x ^ { t } \in B _ { t }$ whose true labels $y ^ { t }$ are from novel, unknown classes, i.e., $y ^ { t } \in \mathcal { V } _ { t } \setminus \mathcal { V } _ { s }$ . For csOOD samples, predictions over the known source classes can be unreliable, leading to poor open-set performance.

Test-time adaptation (TTA) [18, 19, 21, 53] seeks to improve model performance in the target domain by adapting the model using only test data. Specifically, given a set of target test samples $\{ x _ { j } \} _ { j = 1 } ^ { M } ,$ where $x _ { j } \sim Q ( x )$ and $Q ( x ) \neq P ( x )$ , one needs to adapt $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ to improve prediction performance on the target data. To achieve this, many methods update the model by minimizing an unsupervised objective defined on test samples:

$$
\operatorname* { m i n } _ { \overline { { \theta } } } \mathcal { L } ( x ; \theta ) , \quad x \sim Q ( x ) ,\tag{1}
$$

where $\overline { { \theta } }$ denotes the subset of parameters in θ that is updated during TTA. The learning objective $\mathcal { L } ( \cdot )$ is often formulated as an entropy minimization problem.

However, in open-set scenarios where both csID and csOOD samples coexist, traditional TTA methods struggle to perform. If the same closed-set adaptation objective is applied to all target samples, csOOD samples may be forced toward overconfident predictions over known classes and generate harmful gradients during online updates. As a result, they can distort decision boundaries, contaminate adaptation statistics or prototypes, and degrade the performance on csID samples. The goal of open-set TTA is therefore to improve classification on latent csID samples, identify latent csOOD samples, and prevent the latter from inducing harmful adaptation updates.

## IV. UNCERTAINTY-AWARE PROTOTYPE LEARNING

In this paper, we propose Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation (UAPA), a plugand-play method for open-set TTA, which adapts the model using samples selected as csID while limiting harmful updates from samples selected as csOOD. The overall framework is illustrated in Fig. 2, and the pipeline is summarized in Alg. 1. We first introduce Uncertainty-Driven ID/OOD Separation (cf. Sec. IV-A), which combines a classifier-prototype score with a mini-batch-normalized evidence-derived uncertainty score to partition incoming samples into csID and csOOD subsets. We then propose a Dynamic Prototype Refinement strategy (cf. Sec. IV-B), which accumulates selected csID features in class-wise buffers and periodically refines class-specific target prototypes. Lastly, we introduce Contrastive Prototype Entropy Adaptation $( c f .$ Sec. IV-C), a unified objective that minimizes entropy for csID samples and maximizes entropy for csOOD samples, while further using contrastive learning to increase pseudo-label-relative similarity margins for csID features and reduce the maximum similarity between csOOD features and

![](images/deafe39c899efa9d9760eb106a36c0959b78bfb8091e1ff32a424d008a21b5c8.jpg)  
Fig. 2: Overview of UAPA. For each incoming batch $B _ { t }$ , UAPA combines the classifier-prototype score $S _ { \mathrm { p r o t o } }$ with the mini batch-normalized evidence-derived uncertainty score $S _ { \mathrm { E U } }$ SEU to compute a separation score $S ( x _ { t } )$ , which partitions samples into csID and csOOD subsets using threshold $\tau .$ Contrastive Prototype Entropy Adaptation applies entropy minimization to csID samples, entropy maximization to csOOD samples, and prototype-level contrastive learning to regularize feature representations. Dynamic Prototype Refinement periodically updates class-specific target prototypes using buffered csID features.

```latex
Algorithm 1 Pipeline of the proposed UAPA
1: Input: Test stream $\mathcal { D } _ { \mathrm { t e s t } } ,$ , source model M , batch size B, and
hyperparameters $\lambda , \tau , \gamma , \alpha _ { 0 } , \alpha _ { 1 } , \rho , \beta , \eta , \tau _ { 0 } ,$ and ϵ.
2: Initialize: Online model $M _ { 1 }  M _ { 0 } ,$ reference model $M _ { \mathrm { r e f } } $
Copy $( M _ { 0 } )$ , target prototypes $\mu _ { c }  \mathbf { 0 } ,$ feature buffers $\mathcal { F } _ { c } \gets \emptyset$
for all $^ { c , }$ and effective-batch counter $k  0 .$
3: for each incoming mini-batch $B _ { t } \subset \mathcal { D } _ { \mathrm { t e s t } }$ do
4: Obtain predictions and online features for samples in $B _ { t } .$
5: Compute $S _ { \mathrm { p r o t o } }$ using $M _ { \mathrm { r e f } }$ and classifier prototypes.
6: Compute the raw evidence-derived uncertainty score $U _ { \mathrm { E U } }$ and
its mini-batch-normalized score S using online predictions.
7: Compute $S = \lambda S _ { \mathrm { p r o t o } } + ( 1 - \lambda )$ S<sub>EU</sub> and split $B _ { t }$ into $B _ { t , \mathrm { c s I D } }$
and $B _ { t , \mathrm { c s O O D } }$ using threshold $\tau .$
8: Store csID features in class-wise buffers according to their
pseudo-labels.
9: Compute $\mathcal { L } _ { t , \mathrm { e n t r o p y } }$ and $\mathcal { L } _ { t , \cdot }$ <sub>contrast</sub>.
10: Update the online model by minimizing $\begin{array} { r l } { \mathcal { L } _ { t , \mathrm { t o t a l } } } & { { } = } \end{array}$
$\mathcal { L } _ { t , \mathrm { { e n t r o p y } } } + \gamma \mathcal { L } _ { t , \mathrm { { ~ } } }$ <sub>,contrast</sub>.
11: if $B _ { t , \mathrm { c s I D } } \neq \emptyset$ then
12: $k \gets k + 1 .$
13: if k mod $\eta = 0$ then
14: Update each target prototype with a non-empty buffer
using momentum $\beta .$
15: Clear all class-wise feature buffers.
16: Output: Pre-update predictions for all test samples.
```

initialized target prototypes.

## A. Uncertainty-Driven ID/OOD Separation

In open-set test-time adaptation, distinguishing between csID and csOOD samples is crucial for ensuring robust adaptation. This is particularly important because minimizing entropy on unknown-class samples can push them toward overconfident known-class predictions and induce harmful updates. Therefore, we propose the Uncertainty-Driven ID/OOD Separation, which combines prototype similarity with an evidence-derived uncertainty score to distinguish csID from csOOD samples. Specifically, we design an ID/OOD separation score for each sample $x _ { t } ,$ denoted by $S ( x _ { t } )$ , based on two criteria: 1) the similarity of the sample to known-class classifier prototypes, and 2) an evidence-derived uncertainty score computed from the online-model logits.

Prototype-Based Similarity. To estimate whether a target sample is close to the known source classes, we use the classwise classifier weights as known-class prototypes. Specifically, we maintain a non-optimized reference copy $M _ { \mathrm { r e f } } \ =$ $( g _ { \mathrm { r e f } } , h _ { \mathrm { r e f } } )$ initialized from the source model. The learnable parameters of this reference model are not updated during testtime adaptation, and $g _ { \mathrm { r e f } } ( x _ { t } )$ is used to extract features for prototype-based scoring. Let $w _ { c }$ denote the classifier weight corresponding to class c in the reference prediction head, where $c \in \{ 1 , \ldots , C \}$ . Since each classifier weight encodes a class-specific decision direction, it can be regarded as a prototype of the corresponding known class. We compute the prototype-based score by measuring the maximum cosine similarity between the sample feature and all classifier prototypes:

$$
S _ { \mathrm { p r o t o } } ( x _ { t } ) = 1 - \operatorname* { m a x } _ { c } \left( \frac { \langle g _ { \mathrm { r e f } } ( x _ { t } ) , w _ { c } \rangle } { \| g _ { \mathrm { r e f } } ( x _ { t } ) \| _ { 2 } \| w _ { c } \| _ { 2 } } \right) .\tag{2}
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product and $\| \cdot \| _ { 2 }$ denotes the $L _ { 2 }$ norm. A lower $S _ { \mathrm { p r o t o } } ( x _ { t } )$ indicates that the sample feature is close to a known-class classifier prototype, whereas a higher value suggests that the sample is less aligned with known classes and is more likely to be csOOD.

Uncertainty Estimation. Inspired by Evidential Deep Learning [54, 55], we incorporate uncertainty estimation to assess model confidence in its predictions. Specifically, we estimate evidence-derived uncertainty from the online model logits $\mathbf { h } \ = \ h _ { \phi _ { t } } ( g _ { \theta _ { t } } ( x _ { t } ) )$ . We first transform the logits into Dirichlet concentration parameters:

$$
\alpha _ { c } ( x _ { t } ) = \mathrm { s o f t p l u s } ( h _ { c } ( x _ { t } ) ) + 1 , \qquad T ( x _ { t } ) = \sum _ { c = 1 } ^ { C } \alpha _ { c } ( x _ { t } ) ,\tag{3}
$$

where $T ( x _ { t } )$ denotes the total Dirichlet strength. The raw evidence-derived uncertainty score is computed as

$$
\begin{array} { l l r } & { U _ { \mathrm { E U } } ( x _ { t } ) = \displaystyle \frac { C } { T ( x _ { t } ) + \epsilon } , } \\ & { S _ { \mathrm { E U } } ( x _ { t } ) = \displaystyle \frac { U _ { \mathrm { E U } } ( x _ { t } ) - \operatorname* { m i n } _ { x \in { \mathcal { B } _ { t } } } U _ { \mathrm { E U } } ( x ) } { \operatorname* { m a x } _ { x \in { \mathcal { B } _ { t } } } U _ { \mathrm { E U } } ( x ) - \operatorname* { m i n } _ { x \in { \mathcal { B } _ { t } } } U _ { \mathrm { E U } } ( x ) + \epsilon } , } \end{array}\tag{4}
$$

where $\epsilon \ > \ 0$ is a small constant for numerical stability. Before score fusion, the raw uncertainty values are min–max normalized within the current mini-batch to obtain a bounded, batch-relative score that is more compatible in scale with the classifier-prototype score. A larger $S _ { \mathrm { E U } } ( x _ { t } )$ indicates greater relative uncertainty within the current mini-batch.

ID/OOD Separation. After computing the classifierprototype score and the mini-batch-normalized evidencederived uncertainty score, we compute the joint separation score as

$$
S ( x _ { t } ) = \lambda S _ { \mathrm { p r o t o } } ( x _ { t } ) + ( 1 - \lambda ) S _ { \mathrm { E U } } ( x _ { t } ) ,\tag{5}
$$

where $\lambda ~ \in ~ [ 0 , 1 ]$ controls the relative contribution of the geometric and the evidence-derived uncertainty score. Finally, a threshold τ is applied to the combined score $S ( x _ { t } )$ to classify the sample as either csID or csOOD:

$$
x _ { t } \in \left\{ \begin{array} { l l } { \mathrm { c s I D } , } & { { \mathrm { i f } } \ S ( x _ { t } ) < \tau , } \\ { \mathrm { c s O O D } , } & { { \mathrm { i f } } \ S ( x _ { t } ) \geq \tau . } \end{array} \right.\tag{6}
$$

## B. Dynamic Prototype Refinement Strategy

To better track the evolving csID distribution while reducing the influence of csOOD samples, we introduce a Dynamic Prototype Refinement (DPR) strategy. DPR maintains a set of class-specific target prototypes $\{ \mu _ { c } \} _ { c = 1 } ^ { C }$ in the feature space, which provide class-level anchors for the subsequent contrastive prototype adaptation. Unlike the classifier prototypes used for ID/OOD separation, these target prototypes are refined online from reliable csID samples. In our implementation, all target prototypes are initialized as zero vectors and are updated after every $\eta$ effective mini-batches, where an effective mini-batch contains at least one sample identified as csID. We set $\eta = 1 0$ by default.

For each incoming mini-batch, we first identify the csID subset $B _ { t , \mathrm { c s I D } }$ using the ID/OOD separation strategy (see Sec. $\mathrm { I V } { \cdot } \mathrm { A } )$ . For each selected sample $x \in \mathrm { \Delta } B _ { t , \mathrm { c s I D } }$ , we extract its feature representation from the feature extractor of the current online model before the final prediction head, $i . e .$ $z _ { x } = g _ { \theta _ { t } } ( x )$ . We then assign a pseudo-label according to the current model prediction:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c } h _ { \varphi _ { t } } ^ { c } ( g _ { \theta _ { t } } ( x ) ) ,\tag{7}
$$

where $h _ { \varphi _ { t } } ^ { c } ( \cdot )$ denotes the logit of class c and $\hat { y }$ is the pseudolabel assigned to sample $x .$ . The extracted features are stored in class-wise buffers according to their pseudo-labels. Let $\mathcal { F } _ { c }$ denote the feature buffer for class c within the current update interval. When a sample $\boldsymbol { x } \in B _ { t , \mathrm { c s I D } }$ is pseudo-labeled as class $c ,$ its feature $z _ { x }$ is added to $\mathcal { F } _ { c } . \mathrm { ~ A ~ }$ prototype-update counter is advanced only when the current mini-batch contributes at least one selected csID feature. After $\eta$ such effective minibatches, the normalized buffered mean is computed for each non-empty buffer:

$$
\mu _ { c } ^ { \mathrm { b u f } } = \frac { \frac { 1 } { | \mathcal { F } _ { c } | } \sum _ { z \in \mathcal { F } _ { c } } z } { \left\| \frac { 1 } { | \mathcal { F } _ { c } | } \sum _ { z \in \mathcal { F } _ { c } } z \right\| _ { 2 } } .\tag{8}
$$

The prototype of class c is updated with a momentum rule:

$$
\mu _ { c } \gets \beta \mu _ { c } + ( 1 - \beta ) \mu _ { c } ^ { \mathrm { b u f } } ,\tag{9}
$$

where $\beta ~ \in ~ [ 0 , 1 ]$ is the momentum coefficient. Since prototypes are initialized as zero vectors, if $\mu _ { c }$ has not been initialized by any previous non-empty buffer, we directly set

$$
\mu _ { c }  \mu _ { c } ^ { \mathrm { b u f } } .\tag{10}
$$

If $\mathcal { F } _ { c }$ is empty during the current update interval, $\mu _ { c }$ remains unchanged. After each update, the prototype is normalized as

$$
\mu _ { c }  \frac { \mu _ { c } } { \| \mu _ { c } \| _ { 2 } } .\tag{11}
$$

Finally, all class-wise buffers are cleared for the next update interval. This effective-batch-based refinement allows the target prototypes to gradually follow the shifted csID distribution while avoiding prototype updates when no reliable csID features are available. These refined prototypes serve as reliable anchors for contrastive prototype entropy adaptation.

## C. Contrastive Prototype Entropy Adaptation

To ensure effective adaptation on an unlabeled target stream that mixes csID and csOOD samples, we introduce Contrastive Prototype Entropy Adaptation (CPEA). CPEA combines sample-level entropy optimization with prototypelevel contrastive learning to promote confident adaptation on csID samples while suppressing csOOD-induced negative adaptation. The overall objective is defined as

$$
\mathcal { L } _ { t , \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { e n t r o p y } } + \gamma \mathcal { L } _ { \mathrm { c o n t r a s t } } ,\tag{12}
$$

where $\gamma \geq 0$ is the trade-off hyperparameter.

Entropy Optimization. Inspired by previous work [15, 56], we apply different entropy objectives to samples separated by Eq. (6). Specifically, entropy minimization on csID samples encourages confident predictions for shifted known-class samples and supports known-class adaptation. In contrast, entropy maximization on csOOD samples discourages overconfident assignment to any known class, thereby reducing harmful updates induced by unknown-class samples. The entropy losses are defined as

$$
\mathcal { L } _ { t , \mathrm { c s I D } } = \frac { 1 } { \left| \mathcal { B } _ { t , \mathrm { c s I D } } \right| } \sum _ { x \in \mathcal { B } _ { t , \mathrm { c s I D } } } H ( f _ { \theta _ { t } } ( x ) ) ,\tag{13}
$$

$$
\mathcal { L } _ { t , \mathrm { c s O O D } } = \frac { 1 } { | \mathcal { B } _ { t , \mathrm { c s O O D } } | } \sum _ { x \in \mathcal { B } _ { t , \mathrm { c s O O D } } } H ( f _ { \theta _ { t } } ( x ) ) ,\tag{14}
$$

$$
\mathcal { L } _ { t , \mathrm { e n t r o p y } } = \mathcal { L } _ { t , \mathrm { c s I D } } - \alpha _ { 0 } \mathcal { L } _ { t , \mathrm { c s O O D } } - \alpha _ { 1 } H ( \bar { f } _ { \theta _ { t } } ) ,\tag{15}
$$

where $\begin{array} { r } { H ( f _ { \theta _ { t } } ( x ) ) = - \sum _ { c = 1 } ^ { C } f _ { \theta _ { t } } ^ { c } ( x ) } \end{array}$ log $f _ { \theta _ { t } } ^ { c } ( x )$ is entropy of the softmax output $f _ { \theta _ { t } } ( x )$ , and $\begin{array} { r } { \bar { f } _ { \theta _ { t } } = \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { x \in \mathcal { B } _ { t } } f _ { \theta _ { t } } ( x ) } \end{array}$ is average softmax output over the batch $B _ { t }$ . Here, $\alpha _ { 0 }$ controls entropy maximization on selected csOOD samples, whereas $\alpha _ { 1 }$ controls the diversity regularizer on the mean batch prediction, which discourages prediction collapse onto a small subset of known classes.

Contrastive Optimization. To refine the feature space for csID samples and improve csOOD detection, we employ a contrastive optimization that aligns csID features with their corresponding class prototypes while repelling csOOD features from all prototypes. For samples identified as csID by the ID/OOD separation module, we assign pseudo-labels using the online model prediction and apply an InfoNCE-style loss to pull their normalized features toward the corresponding dynamic prototype:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \frac { 1 } { \vert \mathcal { B } _ { t , \mathrm { c s l D } } \vert } \sum _ { x \in \mathcal { B } _ { t , \mathrm { c s l D } } } \log \frac { \exp ( \cos \theta ( z , \mu _ { \hat { y } } ) / \tau _ { 0 } ) } { \sum _ { c = 1 } ^ { C } \exp ( \cos \theta ( z , \mu _ { c } ) / \tau _ { 0 } ) } ,\tag{16}
$$

where $z$ is the normalized feature, $\mu _ { \hat { y } }$ is the prototype for the pseudo-label $\hat { y } ,$ , and cos θ is the cosine similarity. For csOOD samples, we introduce a repulsion term that minimizes their maximum cosine similarity to any prototype, effectively pushing their features away from csID class representations:

$$
\mathcal { L } _ { \mathrm { O O D - r e p u l s e } } = \frac { 1 } { | \mathcal { B } _ { t , \mathrm { c s O O D } } | } \sum _ { x \in \mathcal { B } _ { t , \mathrm { c s O O D } } } \operatorname* { m a x } _ { c } \cos \theta ( z , \mu _ { c } ) .\tag{17}
$$

The contrastive loss in Eq. (12) is as follows:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t } } = \mathcal { L } _ { \mathrm { I n f o N C E } } + \rho \mathcal { L } _ { \mathrm { O O D - r e p u l s e } } ,\tag{18}
$$

where $\rho > 0$ controls the contribution of the csOOD repulsion term. We set $\rho ~ = ~ 0 . 1$ in all experiments. When the csID subset is empty, its entropy term and the entire contrastive loss are set to zero, whereas for an empty csOOD subset, only its entropy and OOD-repulsion terms are set to zero, with InfoNCE retained.

## D. Theoretical Motivations and Geometric Interpretation

This subsection provides a theoretical characterization of the three principal mechanisms underlying UAPA. We first establish a sufficient margin condition for the separability of the fused ID/OOD score and analyze its stability under bounded score perturbations. We then prove the one-step angular non-expansion of DPR toward the current buffered class direction. Finally, through similarity-gradient analysis, we characterize the stage-dependent prototype-relative effects of CPEA under zero-initialized target prototypes and establish the full nearest-prototype max–min geometric equivalence once all target prototypes have been initialized.

a) Conditional interpretation of the evidence-derived score: Recall from Eqs. (3)–(4) that UAPA first computes the raw evidence-derived uncertainty score $U _ { \mathrm { E U } } ( x ) = C / ( T ( x ) +$ ϵ) and then applies mini-batch min–max normalization. The raw score is strictly decreasing with respect to the total Dirichlet strength:

$$
\frac { \partial U _ { \mathrm { E U } } } { \partial T } = - \frac { C } { ( T + \epsilon ) ^ { 2 } } < 0 .\tag{19}
$$

Consequently, if an ID–OOD pair satisfies $T ( x _ { \mathrm { i d } } ) \quad >$ $T ( x _ { \mathrm { o o d } } )$ , then $U _ { \mathrm { E U } } ( x _ { \mathrm { i d } } ) < U _ { \mathrm { E U } } ( x _ { \mathrm { o o d } } )$ . In particular, suppose that there exist constants $m _ { \mathrm { i d } }$ and $m _ { \mathrm { o o d } }$ such that

$$
T ( x _ { \mathrm { i d } } ) \geq m _ { \mathrm { i d } } > m _ { \mathrm { o o d } } \geq T ( x _ { \mathrm { o o d } } )\tag{20}
$$

for all csID and csOOD samples under consideration. This uniform strength gap is sufficient for the raw score to rank every csID sample below every csOOD sample. This result is conditional on the learned logits and does not imply that a source-trained classifier necessarily satisfies $\operatorname { E q . }$ (20).

For a fixed mini-batch, the normalization in Eq. (4) applies the same strictly increasing affine transformation to all raw scores in the batch. It therefore preserves the ordering of unequal raw scores:

$$
U _ { \mathrm { E U } } ( x _ { i } ) < U _ { \mathrm { E U } } ( x _ { j } ) \Longrightarrow S _ { \mathrm { E U } } ( x _ { i } ) < S _ { \mathrm { E U } } ( x _ { j } ) .\tag{21}
$$

However, the extrema used in the normalization depend on the current mini-batch. A raw-score threshold τ<sub>EU</sub> is mapped in batch $B _ { t }$ to

$$
\widetilde { \tau _ { t } } = \frac { \tau _ { \mathrm { E U } } - \operatorname* { m i n } _ { x \in \mathcal { B } _ { t } } U _ { \mathrm { E U } } ( x ) } { \operatorname* { m a x } _ { x \in \mathcal { B } _ { t } } U _ { \mathrm { E U } } ( x ) - \operatorname* { m i n } _ { x \in \mathcal { B } _ { t } } U _ { \mathrm { E U } } ( x ) + \epsilon } ,\tag{22}
$$

which is generally batch dependent. Moreover, although softmax $( \mathbf { h } + a \mathbf { 1 } ) = \mathrm { s o f t m a x } ( \mathbf { h } )$ , the corresponding value of $U _ { \mathrm { E U } }$ generally changes under the transformation $\mathbf { h } \mapsto \mathbf { h } + a \mathbf { 1 }$ Therefore, $U _ { \mathrm { E U } }$ should be interpreted as an evidence-derived uncertainty score that depends on the logit parameterization, rather than as a calibrated estimate of epistemic uncertainty. The normalization yields $S _ { \mathrm { E U } } ~ \in ~ [ 0 , 1 )$ , whereas $S _ { \mathrm { p r o t o } } ~ \in$ [0, 2]. Accordingly, λ balances two cues defined over different numerical ranges. We next characterize the fused score used by UAPA in Eq. (5). For analysis only, let $\mathcal { T } _ { t }$ and ${ \mathcal { O } } _ { t }$ denote the non-empty latent csID and csOOD subsets of $B _ { t }$ , respectively. For each pair $( x _ { i } , x _ { o } ) \in \mathcal { T } _ { t } \times \mathcal { O } _ { t }$ , define

$$
\Delta _ { \mathrm { p r o t o } } ( x _ { o } , x _ { i } ) = S _ { \mathrm { p r o t o } } ( x _ { o } ) - S _ { \mathrm { p r o t o } } ( x _ { i } ) ,
$$

$$
\begin{array} { r } { \Delta _ { \mathrm { E U } } ( x _ { o } , x _ { i } ) = S _ { \mathrm { E U } } ( x _ { o } ) - S _ { \mathrm { E U } } ( x _ { i } ) , } \end{array}\tag{23}
$$

(24)

and define the minimum fused pairwise margin as

$$
M _ { t } ( \lambda ) = \operatorname* { m i n } _ { \stackrel { x _ { i } \in \mathcal { I } _ { t } } { x _ { o } \in \mathcal { O } _ { t } } } \left[ \lambda \Delta _ { \mathrm { p r o t o } } ( x _ { o } , x _ { i } ) + ( 1 - \lambda ) \Delta _ { \mathrm { E U } } ( x _ { o } , x _ { i } ) \right] .\tag{25}
$$

Proposition 1 (Conditional separability of the fused score). If $M _ { t } ( \lambda ) > 0$ , then

$$
\operatorname* { m a x } _ { x _ { i } \in \mathcal { T } _ { t } } S ( x _ { i } ) < \operatorname* { m i n } _ { x _ { o } \in \mathcal { O } _ { t } } S ( x _ { o } ) .\tag{26}
$$

Consequently, every threshold

$$
\tau _ { t } \in \left( \operatorname* { m a x } _ { x _ { i } \in \mathcal { T } _ { t } } S ( x _ { i } ) , \operatorname* { m i n } _ { x _ { o } \in \mathcal { O } _ { t } } S ( x _ { o } ) \right]\tag{27}
$$

separates the two subsets in batch t according to Eq. (6).

Proof. By Eqs. (5) and (25), every $( x _ { i } , x _ { o } ) \ \in \ { \mathcal { T } } _ { t } \times { \mathcal { O } } _ { t }$ satisfies

$$
S ( x _ { o } ) - S ( x _ { i } ) \geq M _ { t } ( \lambda ) > 0 .\tag{28}
$$

Taking the minimum over $x _ { o }$ and the maximum over $x _ { i }$ yields Eq. (26). The decision rule then gives Eq. (27). □

A single fixed threshold is valid over a stream whenever

$$
\operatorname* { s u p } _ { t } \operatorname* { m a x } _ { x _ { i } \in \mathcal { I } _ { t } } S ( x _ { i } ) < \tau \leq \operatorname* { i n f } _ { t } \operatorname* { m i n } _ { x _ { o } \in \mathcal { O } _ { t } } S ( x _ { o } ) .\tag{29}
$$

This proposition also makes the role of cue complementarity explicit. If both pairwise cue margins are positive, every convex combination preserves the ordering; if one cue is reversed, separation depends on λ and the two margin magnitudes. If the two component scores are perturbed by at most $\delta _ { \mathrm { p r o t o } }$ and $\delta _ { \mathrm { E U } }$ per sample, the ordering remains robust whenever $M _ { t } ( \lambda ) > 2 [ \lambda \delta _ { \mathrm { p r o t o } } + ( 1 - \lambda ) \delta _ { \mathrm { E U } } ]$

Compared with GMM-based gating, the raw evidencederived score $U _ { \mathrm { E U } } ( x )$ is computed directly from the sample logits without fitting additional mixture parameters. Its normalized counterpart $S _ { \mathrm { E U } } ( x )$ remains batch-relative because it depends on the extrema of the current mini-batch. Nevertheless, the normalization does not require an additional distributionfitting stage. Combined with the classifier-prototype score, it provides predictive and geometric cues for csID/csOOD partitioning in streaming test-time adaptation.

b) Directional alignment induced by DPR: Consider class c with an initialized target prototype and a nonzero buffered feature mean. Let $\bar { \mu _ { c } } , \bar { \mu _ { c } ^ { \mathrm { b u f } } } \in \bar { \mathbb { S } } ^ { d - 1 }$ denote their normalized directions. Combining Eqs. (9)–(11) gives

$$
v _ { c } = \beta \mu _ { c } + ( 1 - \beta ) \mu _ { c } ^ { \mathrm { b u f } } , \qquad \mu _ { c } ^ { + } = \frac { v _ { c } } { \| v _ { c } \| _ { 2 } } , \qquad \beta \in [ 0 , 1 ] .\tag{30}
$$

Proposition 2 (One-step angular non-expansion under DPR). Under Eq. (30), if $\mu _ { c } \neq - \mu _ { c } ^ { \mathrm { b u f } }$ , then

$$
\angle ( \mu _ { c } ^ { + } , \mu _ { c } ^ { \mathrm { b u f } } ) \leq \angle ( \mu _ { c } , \mu _ { c } ^ { \mathrm { b u f } } ) .\tag{31}
$$

Proof. Let $\theta = \angle ( \mu _ { c } , \mu _ { c } ^ { \mathrm { b u f } } ) \in [ 0 , \pi )$ . For $\theta = 0$ , the result is immediate. For $\theta \in \mathsf { \Gamma } ( 0 , \pi )$ , there exists a unit vector e orthogonal to $\mu _ { c } ^ { \mathrm { b u f } }$ in the two-dimensional subspace spanned by the two directions such that

$$
\mu _ { c } = \cos \theta \mu _ { c } ^ { \mathrm { b u f } } + \sin \theta e .\tag{32}
$$

Substituting this expression into Eq. (30) gives

$$
v _ { c } = \left[ \beta \cos \theta + \left( 1 - \beta \right) \right] \mu _ { c } ^ { \mathrm { b u f } } + \beta \sin \theta e .\tag{33}
$$

Let $\phi = \angle ( \mu _ { c } ^ { + } , \mu _ { c } ^ { \mathrm { b u f } } )$ . The assumption $\mu _ { c } \neq - \mu _ { c } ^ { \mathrm { b u f } }$ ensures that $v _ { c } \neq 0 ,$ , so ϕ is well defined. Direct calculation yields

$$
\sin ( \theta - \phi ) = \frac { ( 1 - \beta ) \sin \theta } { \| v _ { c } \| _ { 2 } } \ge 0 .\tag{34}
$$

Because $\theta - \phi \in ( - \pi , \pi )$ , the above inequality implies $\theta - \phi \geq$ 0, and hence $\phi \leq \theta$ . This proves Eq. (31). □

Proposition 2 establishes one-step alignment toward the buffered class direction, rather than toward the unobserved target-class center. Therefore, interpreting DPR as tracking the target distribution requires that the ID/OOD separation and pseudo-label assignments yield a buffered direction $\mu _ { c } ^ { \mathrm { b u f } }$ that sufficiently approximates the csID class direction.

c) Prototype-relative geometry induced by CPEA: The entropy terms in Eqs. (13)–(15) operate on output probabilities and contain no explicit feature–prototype distance. Their prototype-relative geometric effect is therefore indirect. At adaptation step t, let

$$
\mathcal { V } _ { t } = \{ c \in \{ 1 , \dots , C \} : \| \mu _ { c } \| _ { 2 } > 0 \}
$$

denote the set of initialized target prototypes. In the implementation, the contrastive terms are evaluated over all C prototype coordinates. Under the epsilon-stabilized rowwise normalization used in the implementation, every nonzero prototype is normalized whereas each zero prototype remains unchanged. Therefore, $\| \mu _ { c } \| _ { 2 } = 1$ for $c \in \mathcal { V } _ { t }$ , whereas $\boldsymbol { \mu _ { c } } = \mathbf { 0 }$ for $c \notin \mathcal { V } _ { t }$ . Let $z \in \mathbb { S } ^ { d - \mathrm { 1 } }$ and $s _ { c } = \langle z , \mu _ { c } \rangle$ for $c \in \{ 1 , \ldots , C \}$

For a selected csID sample with pseudo-label yˆ, define

$$
q _ { c } = \frac { \exp ( s _ { c } / \tau _ { 0 } ) } { \sum _ { j = 1 } ^ { C } \exp ( s _ { j } / \tau _ { 0 } ) } .\tag{35}
$$

The single-sample InfoNCE term satisfies

$$
\frac { \partial \ell _ { \mathrm { I n f o N C E } } } { \partial s _ { \hat { y } } } = \frac { q _ { \hat { y } } - 1 } { \tau _ { 0 } } \leq 0 , \qquad \frac { \partial \ell _ { \mathrm { I n f o N C E } } } { \partial s _ { c } } = \frac { q _ { c } } { \tau _ { 0 } } \geq 0 , \quad c \neq \hat { y }\tag{36}
$$

and the ambient Euclidean gradient with respect to the normalized coordinate z is

$$
\nabla _ { z } \ell _ { \mathrm { I n f o N C E } } = \frac { 1 } { \tau _ { 0 } } \left( \sum _ { c = 1 } ^ { C } q _ { c } \mu _ { c } - \mu _ { \hat { y } } \right) .\tag{37}
$$

When $\hat { y } \in \mathcal { V } _ { t }$ , minimizing the InfoNCE term favors a larger relative margin between the initialized pseudo-label prototype and its competitors. Since, for each $c \in \mathcal { V } _ { t }$ ，

$$
\| z - \mu _ { c } \| _ { 2 } ^ { 2 } = 2 - 2 \langle z , \mu _ { c } \rangle ,\tag{38}
$$

increasing $s _ { \hat { y } }$ reduces the corresponding Euclidean and angular distances on the unit sphere. With correct pseudo-labels and informative prototypes, this provides a prototype-centered regularization effect; it does not by itself guarantee global intraclass compactness. When $\hat { y } \notin \mathcal { V } _ { t }$ , the pseudo-label prototype remains zero, so the loss provides no attraction toward that uninitialized prototype. If $\nu _ { t } = \varnothing$ , all similarities are zero, and the InfoNCE loss equals log C with zero gradient with respect to z. For a selected csOOD sample in a mini-batch where the contrastive loss is active, the implemented repulsion term is

$$
\ell _ { \mathrm { r e p } } ( z ) = \operatorname* { m a x } _ { 1 \leq c \leq C } \langle z , \mu _ { c } \rangle .\tag{39}
$$

When all target prototypes are initialized, i.e., $\begin{array} { r l } { \mathcal { V } _ { t } } & { { } = } \end{array}$ $\{ 1 , \ldots , C \}$ , Eq. (38) gives

$$
\arg \operatorname* { m i n } _ { z \in \mathbb { S } ^ { d - 1 } } \operatorname* { m a x } _ { 1 \leq c \leq C } \langle z , \mu _ { c } \rangle = \arg \operatorname* { m a x } _ { z \in \mathbb { S } ^ { d - 1 } } \operatorname* { m i n } _ { 1 \leq c \leq C } \| z - \mu _ { c } \| _ { 2 } ^ { 2 } .\tag{40}
$$

Thus, in the fully initialized regime, minimizing the repulsion term is equivalent to maximizing the minimum squared distance between a selected csOOD feature and the targetprototype set.

If $\nu _ { t } = \sigma$ , all target prototypes are zero and $\ell _ { \mathrm { r e p } } ( z ) = 0$ When $\emptyset \neq \mathcal { V } _ { t } \subsetneq \{ 1 , . . . , C \}$ , at least one target prototype remains uninitialized. The corresponding zero similarities yield

$$
\ell _ { \mathrm { r e p } } ( z ) = \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { c \in \mathcal { V } _ { t } } \langle z , \mu _ { c } \rangle \right\} .\tag{41}
$$

TABLE I: Performance comparison on CIFAR-10-C and CIFAR-100-C using a WideResNet-40-2 backbone. ↑ indicates higher is better, ↓ lower is better. Best in bold and second best in underlined. ‘Ours’ denotes TENT with our UAPA.
<table><tr><td rowspan="2">Method</td><td colspan="4">CIFAR-10-C</td><td colspan="4">CIFAR-100-C</td><td colspan="4">Average</td></tr><tr><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>Source</td><td>81.73</td><td>77.89</td><td>79.47</td><td>68.44</td><td>53.25</td><td>60.55</td><td>94.97</td><td>39.87</td><td>67.49</td><td>69.22</td><td>87.22</td><td>54.16</td></tr><tr><td>BN Adapt [57]</td><td>84.20</td><td>80.40</td><td>76.85</td><td>72.13</td><td>57.16</td><td>72.45</td><td>84.29</td><td>47.10</td><td>70.68</td><td>76.43</td><td>80.57</td><td>59.62</td></tr><tr><td>CoTTA [58]</td><td>85.74</td><td>85.81</td><td>72.77</td><td>77.17</td><td>56.50</td><td>77.06</td><td>80.98</td><td>49.00</td><td>71.12</td><td>81.44</td><td>76.88</td><td>63.09</td></tr><tr><td>TENT [18]</td><td>79.12</td><td>67.79</td><td>95.76</td><td>58.27</td><td>50.19</td><td>59.42</td><td>93.54</td><td>36.25</td><td>64.66</td><td>63.61</td><td>94.65</td><td>47.26</td></tr><tr><td>EATA [19]</td><td>76.17</td><td>55.43</td><td>92.88</td><td>46.66</td><td>61.34</td><td>87.05</td><td>56.06</td><td>57.27</td><td>68.76</td><td>71.24</td><td>74.47</td><td>51.97</td></tr><tr><td>SAR [22]</td><td>78.41</td><td>66.17</td><td>96.13</td><td>56.79</td><td>59.67</td><td>71.70</td><td>83.56</td><td>48.68</td><td>69.04</td><td>68.94</td><td>89.85</td><td>52.74</td></tr><tr><td>COME [21]</td><td>85.31</td><td>75.34</td><td>84.01</td><td>69.29</td><td>58.09</td><td>72.28</td><td>79.24</td><td>47.49</td><td>71.70</td><td>73.81</td><td>81.63</td><td>58.39</td></tr><tr><td>OSTTA [20]</td><td>84.24</td><td>72.67</td><td>76.98</td><td>64.98</td><td>60.03</td><td>75.09</td><td>82.78</td><td>51.23</td><td>72.14</td><td>73.88</td><td>79.88</td><td>58.11</td></tr><tr><td>UniEnt [15]</td><td>84.34</td><td>92.15</td><td>37.25</td><td>80.24</td><td>59.01</td><td>88.07</td><td>57.17</td><td>55.95</td><td>71.68</td><td>90.11</td><td>47.21</td><td>68.10</td></tr><tr><td>ROSETTA [16]</td><td>84.90</td><td>85.06</td><td>66.00</td><td>76.34</td><td>59.53</td><td>87.02</td><td>61.61</td><td>56.07</td><td>72.22</td><td>86.04</td><td>63.81</td><td>66.21</td></tr><tr><td>UAPA (Ours)</td><td>84.53</td><td>96.18</td><td>13.57</td><td>82.27</td><td>60.14</td><td>93.26</td><td>27.59</td><td>58.08</td><td>72.34</td><td>94.72</td><td>20.58</td><td>70.18</td></tr></table>

TABLE II: Performance comparison on Tiny-ImageNet-C across different backbones (ResNet-50 and ViT-Base). ‘-’ indicate the experiment was not applicable. Best in bold and second best in underlined. ‘Ours’ is TENT with our UAPA.
<table><tr><td rowspan="2">Method</td><td colspan="4">ResNet-50</td><td colspan="4">ViT-Base</td><td colspan="4">Average</td></tr><tr><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>Source</td><td>22.36</td><td>48.11</td><td>90.80</td><td>15.81</td><td>22.09</td><td>48.45</td><td>94.85</td><td>16.29</td><td>22.23</td><td>48.28</td><td>92.83</td><td>16.05</td></tr><tr><td>BN Adapt [57]</td><td>33.79</td><td>56.43</td><td>87.80</td><td>25.60</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoTTA [58]</td><td>32.68</td><td>47.60</td><td>92.27</td><td>22.45</td><td>16.17</td><td>40.99</td><td>95.38</td><td>10.30</td><td>24.43</td><td>44.30</td><td>93.83</td><td>16.38</td></tr><tr><td>TENT [18]</td><td>25.45</td><td>52.22</td><td>94.29</td><td>17.48</td><td>26.45</td><td>52.70</td><td>95.13</td><td>19.65</td><td>25.95</td><td>52.46</td><td>94.71</td><td>18.57</td></tr><tr><td>EATA [19]</td><td>34.46</td><td>55.86</td><td>92.52</td><td>25.92</td><td>27.95</td><td>52.61</td><td>95.14</td><td>21.31</td><td>31.21</td><td>54.24</td><td>93.83</td><td>23.62</td></tr><tr><td>SAR [22]</td><td>34.41</td><td>54.18</td><td>93.74</td><td>24.99</td><td>22.34</td><td>48.64</td><td>94.81</td><td>16.37</td><td>28.38</td><td>51.41</td><td>94.28</td><td>20.68</td></tr><tr><td>COME [21]</td><td>34.28</td><td>49.27</td><td>93.92</td><td>23.65</td><td>22.40</td><td>48.80</td><td>94.70</td><td>16.41</td><td>28.34</td><td>49.04</td><td>94.31</td><td>20.03</td></tr><tr><td>OSTTA [20]</td><td>34.75</td><td>53.00</td><td>95.29</td><td>25.54</td><td>27.02</td><td>49.25</td><td>96.06</td><td>20.17</td><td>30.89</td><td>51.13</td><td>95.68</td><td>22.86</td></tr><tr><td>UniEnt [15]</td><td>34.45</td><td>61.77</td><td>91.53</td><td>27.75</td><td>24.24</td><td>46.86</td><td>96.11</td><td>17.67</td><td>29.35</td><td>54.32</td><td>93.82</td><td>22.71</td></tr><tr><td>ROSETTA [16]</td><td>34.09</td><td>65.62</td><td>87.25</td><td>27.75</td><td>23.85</td><td>48.78</td><td>96.43</td><td>16.84</td><td>28.97</td><td>57.20</td><td>91.84</td><td>22.30</td></tr><tr><td>UAPA (Ours)</td><td>33.85</td><td>67.47</td><td>81.06</td><td>28.10</td><td>28.53</td><td>52.87</td><td>95.37</td><td>21.70</td><td>31.19</td><td>60.17</td><td>88.22</td><td>24.90</td></tr></table>

The repulsion term therefore penalizes only the maximum positive similarity between the selected csOOD feature and the initialized prototypes. Together with InfoNCE, it provides stage-dependent, prototype-relative regularization for csID alignment and csOOD separation.

## V. EXPERIMENTS

## A. Experimental Settings

Datasets. Our experimental setup is designed to construct open-set test streams containing both csID and csOOD samples, where the test data is a composite of both csID and csOOD samples. The csID component is sourced from three standard corruption benchmarks: CIFAR-10-C, CIFAR-100- C, and Tiny-ImageNet-C [59], where we consistently use the most severe corruption setting, level 5. To create the matching csOOD component, we adopt the protocol from OSTTA [20] and apply identical corruption functions to the original SVHN and ImageNet-O test sets, generating SVHN-C and ImageNet-O-C. The SVHN-C provides the csOOD samples for the CIFAR-C benchmarks, while ImageNet-O-C is paired with Tiny-ImageNet-C, thus forming the final mixed-distribution test sets. In addition, to evaluate robustness under small-gap settings, we follow the STAMP [34] and construct CIFAR-10/CIFAR-100 splits with an 8:2 ratio of ID to OOD classes.

Evaluation protocols. Following previous studies [20], the model adapts to sequentially arriving domains without parameter resets. For each mini-batch, predictions are made before the model is updated, ensuring no future information influences current outputs. In the main evaluation, each minibatch contains equal numbers of csID and csOOD samples. Additional experiments evaluate varying csOOD ratios from 10% to 90%. Performance is quantified using four metrics. Accuracy (Acc) measures classification performance on csID samples. Open-set detection capability is evaluated via AU-ROC and FPR@95TPR. To assess the trade-off between classification and detection, we report the Open-Set Classification Rate (OSCR) for a balanced view.

Baselines. We compare UAPA with representative closedset TTA methods, including BN Adapt [57], CoTTA [58], TENT [18], EATA [19], SAR [22], and COME [21], as well as open-set TTA methods, including OSTTA [20], UniEnt [15] and ROSETTA [16].

Implementation details. Following previous studies [20], experiments on CIFAR benchmarks adopt WideResNet-40- 2 [60] pretrained from RobustBench [61]. For Tiny-ImageNet-C, we evaluate on two representative architectures, ResNet-50 [1] and ViT-Base [2], to verify architectural generality. ResNet-50 is trained on Tiny-ImageNet using SGD for 50 epochs, while ViT-Base is fine-tuned on the same dataset using Adam for 100 epochs. During TTA, all methods use the Adam optimizer with batch size 100. For CNN-based models, adaptation is applied to Batch Normalization affine parameters, while for ViT-based models we adapt Layer Normalization affine parameters, following standard practice in TTA. Learning rates for all baselines follow their official implementations or standard settings, including 0.001 for TENT, EATA, and OSTTA; 0.01 for CoTTA; and 0.0001 for SAR and COME. For UAPA, we use a learning rate of 0.001. The prototype momentum, contrastive temperature, OOD-repulsion coefficient, and prototype-update interval are fixed as $\beta = 0 . 9 , \tau _ { 0 } = 0 . 1 , \rho = 0 . 1 ,$ , and $\eta = 1 0$ , respectively. The remaining trade-off parameters $\left( \lambda , \tau , \gamma , \alpha _ { 0 } , \alpha _ { 1 } \right)$ are kept fixed within each benchmark. The configurations on Tiny-ImageNet-C with ResNet and ViT: (0.9, 0.65, 0.2, 1.0, 0.1) and (0.7, 0.9, 0.1, 1.0, 1.0). Experiments are conducted on an 80G memory-sized NVIDIA A800 GPU using PyTorch 1.12.1. We use a fixed random seed of 1 for all experiments. Unless otherwise specified, UAPA is instantiated with TENT as the base adaptation method in the main experiments.

TABLE III: Ablation experiments on CIFAR-10-C and CIFAR-100-C using a WideResNet-40-2 backbone. $\mathbf { \Delta } ^ { 6 } \mathrm { P S } ^ { \mathbf { , } }$ denotes Prototype-based Similarity, ‘UE’ denotes Uncertainty Estimation, ‘DPR’ denotes Dynamic Prototype Refinement strategy, and ‘CL’ denotes Contrastive Learning Optimization.
<table><tr><td rowspan="2">Method</td><td colspan="4">CIFAR-10-C</td><td colspan="4">CIFAR-100-C</td><td colspan="4">Average</td></tr><tr><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>TENT</td><td>79.12</td><td>67.79</td><td>95.76</td><td>58.27</td><td>50.19</td><td>59.42</td><td>93.54</td><td>36.25</td><td>64.66</td><td>63.61</td><td>94.65</td><td>47.26</td></tr><tr><td>w/o PS</td><td>78.46</td><td>85.40</td><td>36.21</td><td>69.22</td><td>53.60</td><td>90.14</td><td>27.18</td><td>50.16</td><td>66.03</td><td>87.77</td><td>31.70</td><td>59.69</td></tr><tr><td>w/o UE</td><td>84.87</td><td>89.88</td><td>45.13</td><td>79.10</td><td>56.96</td><td>94.33</td><td>26.07</td><td>55.51</td><td>70.92</td><td>92.11</td><td>35.60</td><td>67.31</td></tr><tr><td>w/o DPR</td><td>84.35</td><td>93.57</td><td>28.10</td><td>80.95</td><td>59.64</td><td>86.75</td><td>51.70</td><td>55.32</td><td>72.00</td><td>90.16</td><td>39.90</td><td>68.14</td></tr><tr><td>w/o CL</td><td>84.38</td><td>93.28</td><td>26.05</td><td>80.66</td><td>60.03</td><td>89.58</td><td>39.40</td><td>56.59</td><td>72.21</td><td>91.43</td><td>32.73</td><td>68.63</td></tr><tr><td>Ours</td><td>84.53</td><td>96.18</td><td>13.57</td><td>82.27</td><td>60.14</td><td>93.26</td><td>27.59</td><td>58.08</td><td>72.34</td><td>94.72</td><td>20.58</td><td>70.18</td></tr></table>

## B. Comparison Experiments

CIFAR benchmarks. As shown in Table I, UAPA consistently improves open-set detection performance over strong baselines on CIFAR-10-C and CIFAR-100-C. On CIFAR-10-C, UAPA achieves competitive classification accuracy (84.53%) while obtaining the best AUROC (96.18%) and the lowest FPR@95TPR (13.57%). Compared to UniEnt, this corresponds to a 63.6% relative reduction in false positives, indicating improved separation between csID and csOOD samples. On CIFAR-100-C, UAPA further achieves the best performance across all OOD detection metrics, reducing FPR@95TPR to 27.59%, which is 50.8% lower than EATA. These results demonstrate consistent improvements in open-set robustness under severe distribution shifts.

Tiny-ImageNet-C. As shown in Table II, UAPA achieves strong performance across both ResNet-50 and ViT-Base backbones. On ResNet-50, it attains the highest AUROC and OSCR, while maintaining competitive accuracy. On ViT-Base, UAPA achieves the best performance in accuracy and OSCR, with an OSCR of 21.70%, outperforming all compared methods. Overall, the results verify the effectiveness and generality of UAPA across architectures.

## C. Ablation Studies

Effectiveness of Uncertainty-Driven ID/OOD Separation. We evaluate the contribution of the two components in the proposed ID/OOD separation module by removing Uncertainty Estimation (w/o UE) and Prototype-based Similarity (w/o PS), respectively. As shown in Table III, removing either component degrades the overall performance. Without Uncertainty Estimation, the average AUROC decreases from 94.72% to 92.11%, and the average FPR@95TPR increases from 20.58% to 35.60%, indicating that evidence-derived uncertainty is important for identifying uncertain csOOD samples. Removing Prototype-Based Similarity causes a larger drop in OSCR, from 70.18% to 59.69%, suggesting that classifierprototype similarity provides complementary geometric cues for separating csID and csOOD samples.

TABLE IV: Results under the small-gap classes setting, where CIFAR-10/CIFAR-100 classes are split into 8:2 (ID: OOD). This setting poses a more challenging scenario.
<table><tr><td>Method</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td rowspan="3">Source BN Adapt [57] CoTTA [58]</td><td>66.81</td><td>46.75</td><td>93.99</td><td>36.13</td></tr><tr><td>72.70</td><td>62.06</td><td>91.30</td><td>52.15</td></tr><tr><td>72.74</td><td>61.65</td><td>89.96</td><td>51.42</td></tr><tr><td rowspan="4">TENT [18] EATA [19] SAR [22]</td><td>73.68</td><td>61.16</td><td>94.77</td><td>51.55</td></tr><tr><td>73.16</td><td>61.34</td><td>87.49</td><td>50.52</td></tr><tr><td>73.53</td><td>62.01</td><td>91.86</td><td>52.63</td></tr><tr><td>72.62</td><td>62.92</td><td>90.86</td><td>52.81</td></tr><tr><td>COME [21] OSTTA [20]</td><td>73.75</td><td>59.40</td><td>86.88</td><td>50.32</td></tr><tr><td>UniEnt [15]</td><td>73.70</td><td>62.21</td><td>86.69</td><td>52.68</td></tr><tr><td>ROSETTA [16]</td><td>71.78</td><td>61.44</td><td>93.00</td><td>51.96</td></tr><tr><td>UAPA (Ours)</td><td>74.47</td><td>64.30</td><td>88.03</td><td>54.64</td></tr></table>

Effectiveness of Dynamic Prototype Refinement Strategy. We further remove the DPR strategy (w/o DPR) to evaluate the role of online prototype updates. Compared with the complete method, w/o DPR reduces the average AUROC from 94.72% to 90.16% and increases FPR@95TPR from 20.58% to 39.90%. This indicates that periodically refining target prototypes with selected csID samples helps maintain reliable class anchors under distribution shifts and reduces prototype contamination from csOOD samples.

Effectiveness of Contrastive Prototype Entropy Adaptation. We evaluate the contribution of contrastive optimization by removing the contrastive learning term (w/o CL). From Table III, removing this term reduces the average OSCR from 70.18% to 68.63% and increases FPR@95TPR from 20.58% to 32.73%. These results suggest that prototype-level contrastive learning improves feature separation by pulling csID features toward class prototypes while reducing the similarity between csOOD features and known-class prototypes.

TABLE V: Performance comparison of different ID/OOD separation strategies on CIFAR-10-C and CIFAR-100-C using a WideResNet-40-2 backbone. “Ours (w/ GMM)” replaces the proposed Uncertainty-Driven ID/OOD Separation with a UniEntstyle GMM gate, while keeping all other components unchanged.
<table><tr><td rowspan="2">Method</td><td colspan="4">CIFAR-10-C</td><td colspan="4">CIFAR-100-C</td><td colspan="4">Average</td></tr><tr><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>Source</td><td>81.73</td><td>77.89</td><td>79.47</td><td>68.44</td><td>53.25</td><td>60.55</td><td>94.97</td><td>39.87</td><td>67.49</td><td>69.22</td><td>87.22</td><td>54.16</td></tr><tr><td>TENT [18]</td><td>79.12</td><td>67.79</td><td>95.76</td><td>58.27</td><td>50.19</td><td>59.42</td><td>93.54</td><td>36.25</td><td>64.66</td><td>63.61</td><td>94.65</td><td>47.26</td></tr><tr><td>Ours (w/ GMM [15])</td><td>84.45</td><td>95.67</td><td>18.16</td><td>82.12</td><td>59.20</td><td>92.30</td><td>33.64</td><td>56.72</td><td>71.83</td><td>93.99</td><td>25.90</td><td>69.42</td></tr><tr><td>UAPA (Ours)</td><td>84.53</td><td>96.18</td><td>13.57</td><td>82.27</td><td>60.14</td><td>93.26</td><td>27.59</td><td>58.08</td><td>72.34</td><td>94.72</td><td>20.58</td><td>70.18</td></tr></table>

![](images/4f2346ff3c82e25630d25eb3d818c0858060bea08b8e491d1a4b944fa8351372.jpg)  
(c) UniEnt  
(d) UAPA (Ours)  
Fig. 3: t-SNE visualization of CIFAR-10-C and SVHN-C features, where SVHN-C samples are treated as csOOD. Yellow denotes csOOD samples, while the remaining colors denote different csID classes.

## D. More Discussions

Robustness under small-gap settings. We evaluate UAPA under the small-gap setting, where ID and OOD classes are semantically closer and thus harder to distinguish. As shown in Table IV, UAPA achieves the best Acc, AUROC, and OSCR among the compared methods, while remaining competitive in FPR@95TPR. Specifically, UAPA improves AUROC by 1.38 percentage points and OSCR by 1.83 percentage points over the second-best methods. These results indicate that UAPA remains effective when ID and OOD classes are difficult to separate. The improvement is consistent with the design of uncertainty-driven csID/csOOD separation and contrastive prototype adaptation, which provide complementary cues for handling subtle open-set differences.

![](images/9ea67fa0f0814aa3b9c0836d53767dfcb340fb3395797801db6245c953120557.jpg)  
Fig. 4: OSCR comparison on CIFAR-10-C under varying csOOD ratios. Values above UAPA denote its gains in percentage points over the strongest baseline at each ratio.

t-SNE visualization. To illustrate the effects of different methods on csID classification and csOOD detection, we visualize the feature representations of CIFAR-10-C test samples, with SVHN-C test samples serving as csOOD samples using t-SNE [62], as shown in Fig. 3. As observed, after adaptation with TENT, the feature distributions of known-class (csID) and unknown-class (csOOD) samples are mixed. Although UniEnt shows some improvement in distinguishing csID and csOOD samples, there is still considerable overlap, with some csID samples being mixed with csOOD samples. In contrast, UAPA produces more compact csID clusters and a clearer visual separation between csID and csOOD samples. This qualitative observation is consistent with the prototype-relative feature regularization introduced by CPEA, which attracts csID features toward their corresponding target prototypes and reduces the maximum similarity between csOOD features and known-class prototypes.

Comparison with GMM-based ID/OOD Separation. To evaluate our ID/OOD separation mechanism, we construct a variant that replaces it with a UniEnt-style Gaussian Mixture Model (GMM) gate while keeping all other components unchanged. As shown in Table V, the GMM-based variant outperforms the Source and TENT baselines across all metrics, showing that the overall framework remains effective with an alternative separation strategy. Nevertheless, the full model achieves better results than the GMM-based variant across all metrics on both CIFAR-10-C and CIFAR-100-C. On average, the proposed separation mechanism reduces FPR@95TPR by 5.32 percentage points and improves OSCR by 0.76 percentage points over the GMM-based alternative. These results suggest that combining the classifier-prototype score with the minibatch-normalized evidence-derived uncertainty score yields a more effective csID/csOOD partition than the evaluated GMM-based alternative under the current protocol.

TABLE VI: Plug-and-play performance of UAPA on CIFAR-10-C, CIFAR-100-C, and their average results.
<table><tr><td rowspan="2">Method</td><td colspan="4">CIFAR-10-C</td><td colspan="4">CIFAR-100-C</td><td colspan="4">Average</td></tr><tr><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>TENT</td><td>79.12</td><td>67.79</td><td>95.76</td><td>58.27</td><td>50.19</td><td>59.42</td><td>93.54</td><td>36.25</td><td>64.66</td><td>63.61</td><td>94.65</td><td>47.26</td></tr><tr><td>+ UAPA</td><td>84.53</td><td>96.18</td><td>13.57</td><td>82.27</td><td>60.14</td><td>93.26</td><td>27.59</td><td>58.08</td><td>72.34</td><td>94.72</td><td>20.58</td><td>70.18</td></tr><tr><td>EATA</td><td>76.17</td><td>55.43</td><td>92.88</td><td>46.66</td><td>61.34</td><td>87.05</td><td>56.06</td><td>57.27</td><td>68.76</td><td>71.24</td><td>74.47</td><td>51.97</td></tr><tr><td>+ UAPA</td><td>84.69</td><td>92.75</td><td>27.24</td><td>80.23</td><td>61.51</td><td>92.16</td><td>34.30</td><td>59.07</td><td>73.10</td><td>92.46</td><td>30.77</td><td>69.65</td></tr><tr><td>COME</td><td>85.31</td><td>75.34</td><td>84.01</td><td>69.29</td><td>58.09</td><td>72.28</td><td>79.24</td><td>47.49</td><td>71.70</td><td>73.81</td><td>81.63</td><td>58.39</td></tr><tr><td>+ UAPA</td><td>84.87</td><td>79.53</td><td>73.28</td><td>71.54</td><td>59.93</td><td>76.56</td><td>78.61</td><td>51.48</td><td>72.40</td><td>78.05</td><td>75.95</td><td>61.51</td></tr><tr><td>OSTTA</td><td>84.24</td><td>72.67</td><td>76.98</td><td>64.98</td><td>60.03</td><td>75.09</td><td>82.78</td><td>51.23</td><td>72.14</td><td>73.88</td><td>79.88</td><td>58.11</td></tr><tr><td>+ UAPA</td><td>84.55</td><td>92.58</td><td>26.15</td><td>79.81</td><td>60.59</td><td>89.54</td><td>38.48</td><td>57.15</td><td>72.57</td><td>91.06</td><td>32.32</td><td>68.48</td></tr></table>

TABLE VII: Results of different methods on Tiny-ImageNet-C with ResNet-50 backbone. We report Accuracy (Acc), AUROC, FPR@95TPR, OSCR, and the per-image additional inference time introduced by UAPA (in milliseconds).
<table><tr><td>Method</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td><td>Time (ms)↓</td></tr><tr><td>TENT [18]</td><td>25.45</td><td>52.22</td><td>94.29</td><td>17.48</td><td></td></tr><tr><td>+ UAPA</td><td>33.85</td><td>67.47</td><td>81.06</td><td>28.10</td><td>+0.22</td></tr><tr><td>EATA [19]</td><td>34.46</td><td>55.86</td><td>92.52</td><td>25.92</td><td></td></tr><tr><td>+ UAPA</td><td>35.47</td><td>56.94</td><td>88.72</td><td>27.43</td><td>+0.26</td></tr><tr><td>OSTTA [20]</td><td>34.75</td><td>53.00</td><td>95.29</td><td>25.54</td><td></td></tr><tr><td>+ UAPA</td><td>34.87</td><td>53.34</td><td>92.95</td><td>25.68</td><td>+0.31</td></tr><tr><td>COME [21]</td><td>34.28</td><td>49.27</td><td>93.92</td><td>23.65</td><td></td></tr><tr><td>+ UAPA</td><td>35.19</td><td>55.61</td><td>88.80</td><td>26.56</td><td>+0.33</td></tr></table>

Robustness to varying csOOD ratios. To examine the sensitivity of UAPA to open-set stream composition, we vary the csOOD proportion in each test mini-batch from 10% to 90% on CIFAR-10-C while keeping the remaining settings unchanged. As shown in Fig. 4, UAPA achieves the highest OSCR at every evaluated ratio. Its gains over the strongest competing method are 1.41, 1.62, 2.06, 3.28, and 0.61 percentage points at csOOD ratios of 10%, 30%, 50%, 70%, and 90%, respectively. The performance of UAPA remains stable from 10% to 70% csOOD. Although its OSCR decreases when csOOD samples dominate the stream at a ratio of 90%, it still performs best among the compared methods. These results show that UAPA remains effective across varying csID/csOOD compositions, including highly imbalanced test streams.

Plug-and-Play Compatibility of UAPA. To assess the plug-and-play compatibility of our proposed UAPA, we incorporate it into four representative adaptation methods, including TENT [18], EATA [19], OSTTA [20], and COME [21]. For each baseline, we retain its original adaptation objective, update strategy, and method-specific components, while incorporating UAPA’s sample separation, prototype refinement, and contrastive regularization, unless otherwise stated. As shown in Fig. 1, Table VI, and Table VII, incorporating UAPA improves open-set detection and OSCR across all evaluated baselines. On the CIFAR benchmarks, combining UAPA with TENT increases the average OSCR from 47.26% to 70.18% and reduces FPR@95TPR from 94.65% to 20.58%. It also improves the average OSCR of EATA, COME, and OSTTA by 17.68, 3.12, and 10.37 percentage points, respectively. Although the magnitude of improvement varies across baselines, the consistent gains indicate that UAPA is compatible with different adaptation objectives. On Tiny-ImageNet-C, UAPA also improves the OSCR of all four baselines while introducing only about 0.3 ms of additional processing time per image. In particular, when combined with TENT, it improves OSCR by 10.62 percentage points with an additional latency of 0.22 ms per image. These results indicate a favorable performance– latency trade-off and support the use of UAPA as a plug-in module for open-set test-time adaptation.

TABLE VIII: Results on Tiny-ImageNet-C with ViT-Base under the single-class-per-batch setting.
<table><tr><td>Method</td><td>Acc↑</td><td>AUROC↑</td><td>FPR@95TPR↓</td><td>OSCR↑</td></tr><tr><td>Source</td><td>22.09</td><td>48.45</td><td>94.85</td><td>16.29</td></tr><tr><td>BN Adapt [57]</td><td>22.09</td><td>51.96</td><td>94.52</td><td>16.54</td></tr><tr><td>CoTTA [58]</td><td>15.13</td><td>42.04</td><td>95.45</td><td>9.40</td></tr><tr><td>TENT [18]</td><td>14.86</td><td>47.92</td><td>96.05</td><td>10.07</td></tr><tr><td>EATA [19]</td><td>22.26</td><td>51.98</td><td>94.64</td><td>16.64</td></tr><tr><td>SAR [22]</td><td>22.24</td><td>51.76</td><td>94.58</td><td>16.50</td></tr><tr><td>COME [21]</td><td>22.34</td><td>52.05</td><td>94.34</td><td>16.61</td></tr><tr><td>OSTTA [20]</td><td>23.05</td><td>50.83</td><td>95.32</td><td>17.00</td></tr><tr><td>UniEnt [15]</td><td>22.63</td><td>52.04</td><td>94.83</td><td>16.76</td></tr><tr><td>ROSETTA [16]</td><td>22.71</td><td>49.31</td><td>96.12</td><td>16.24</td></tr><tr><td>UAPA (Ours)</td><td>23.31</td><td>52.68</td><td>94.54</td><td>17.63</td></tr></table>

Robustness under the Single-Class-per-Batch Setting. We evaluate the proposed UAPA under a single-class-per-batch setting, where each test mini-batch contains samples from only one class. This setting represents an extreme form of batch-level class imbalance, in which each update observes only partial class information and lacks within-batch class diversity. In UAPA, prototype similarity is computed using the normalized weights of the pretrained classifier head, which provide class references independent of the class composition of the current mini-batch. Meanwhile, DPR accumulates features from samples identified as csID in class-wise buffers and leaves prototypes of unobserved classes unchanged, thereby avoiding unsupported prototype updates under incomplete batch observations. From Table VIII, UAPA achieves the highest Acc, AUROC, and OSCR in this setting. Nevertheless, FPR@95TPR remains high at 94.54%, indicating that reliable csOOD rejection is still challenging under the single-class-perbatch setting. Compared with the second-best results, UAPA improves Acc by 0.26 percentage points and OSCR by 0.63 percentage points. The results suggest that the combination of batch-independent classifier prototypes and buffered targetprototype refinement helps UAPA remain effective when each mini-batch contains only limited class information.

![](images/aede428e23e042b2aade0789e0ec8916e1a170ad8b21c61e6fcc4083e3926b89.jpg)  
(a) Effect of λ

![](images/2b6f62214730110791d41edf148943b8ea938774582b5bbcb2163ee35f6e6920.jpg)  
(b) Effect of τ  
ResNet-50 ViT-Base

![](images/43a092bec00a52eb63f8489d2ded4346edfeb392481dbc5b7aeaadb260d589dd.jpg)  
(c) Effect of γ  
Fig. 5: Effect of key hyperparameters on OSCR on Tiny-ImageNet-C with ResNet-50 and ViT-Base backbones. We vary one hyperparameter at a time while keeping the others fixed at their selected values.

Hyperparameter Analysis. We examine the effects of the score-balance weight λ, separation threshold τ, and contrastive-loss weight γ on Tiny-ImageNet-C. As shown in Fig. 5, UAPA maintains competitive performance over a reasonably broad range around the selected values. The contrastive-loss weight γ has a relatively mild effect on both backbones. The score-balance weight λ exhibits a broad highperforming range, although ResNet-50 benefits more from larger values. The separation threshold τ shows backbonedependent preferred ranges because it directly controls the csID/csOOD partition and the resulting score distributions differ across architectures. Overall, the selected configurations lie within stable operating regions rather than isolated optima.

## VI. CONCLUSION

In this paper, we propose Uncertainty-Aware Prototype Learning for Open-Set Test-Time Adaptation (UAPA), a plugand-play method for test streams containing both csID and csOOD samples. Specifically, UAPA combines a classifierprototype score with a mini-batch-normalized evidencederived uncertainty score to separate csID and csOOD samples. UAPA further employs Dynamic Prototype Refinement to update class-specific target prototypes from buffered features of selected csID samples. Contrastive Prototype Entropy Adaptation then combines class-conditional entropy optimization with prototype-level contrastive learning to increase pseudolabel-relative similarity margins for csID features and reduce the maximum similarity between csOOD features and initialized known-class prototypes. Experimental results on CIFAR-C and Tiny-ImageNet-C with both CNN and ViT backbones demonstrate the effectiveness of UAPA, consistently improving open-set classification and detection with limited additional computational cost.

## REFERENCES

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. IEEE Computer Society, 2016, pp. 770–778.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[3] A. Conti, E. Fini, M. Mancini, P. Rota, Y. Wang, and E. Ricci, “Vocabulary-free image classification and semantic segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 6, pp. 6127–6138, 2026.

[4] J. Wu, Z. Li, W. Sun, J. Yin, L. Nie, and Z. Lin, “Clusmatch: Improving deep clustering by unified positive and negative pseudo-label learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 11, pp. 9688–9701, 2025.

[5] Z. Zhuang, Z. Wang, S. Chen, L. Liu, H. Luo, and M. Tan, “Robust 3d semantic occupancy prediction with calibration-free spatial transformation,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–16, 2026.

[6] Z. Wang, Z. Huang, Y. Gao, N. Wang, and S. Liu, “Mv2dfusion: Leveraging modality-specific object semantics for multi-modal 3d detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 609–623, 2026.

[7] K. Zhou, Z. Liu, Y. Qiao, T. Xiang, and C. C. Loy, “Domain generalization: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 4, pp. 4396–4415, 2023.

[8] Z. Feng, C. Xu, and D. Tao, “Open-set hypothesis transfer with semantic consistency,” IEEE Trans. Image Process., vol. 30, pp. 6473–6484, 2021.

[9] J. Wang, Y. Xu, J. Tang, Z. Gong, B. Tao, C. W. de Silva, and X. Bai, “Vicinal gaussian transform: Rethinking source-free domain adaptation through source-informed label consistency,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 2, pp. 2030–2047, 2026.

[10] P. Du, Y. Wang, Y. Sun, L. Wang, Y. Liao, G. Zhang, E. Ding, Y. Wang, J. Wang, and S. Liu, “Lami-detr: Open-vocabulary detection with language model instruction,” in Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXIII, ser. Lecture

Notes in Computer Science, vol. 15081. Springer, 2024, pp. 312–328.

[11] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. Earnshaw, I. S. Haque, S. M. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang, “WILDS: A benchmark of in-the-wild distribution shifts,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 5637–5664.

[12] C. Geng, S. Huang, and S. Chen, “Recent advances in open set recognition: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 10, pp. 3614–3631, 2021.

[13] H. Shi, M. Hayat, Y. Wu, and J. Cai, “Proposalclip: Unsupervised open-category object proposal generation via exploiting CLIP cues,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 2022, pp. 9601–9610.

[14] W. Zheng, S. Gong, and T. Xiang, “Towards open-world person re-identification by one-shot group-based verification,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 38, no. 3, pp. 591–606, 2016.

[15] Z. Gao, X. Zhang, and C. Liu, “Unified entropy optimization for open-set test-time adaptation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024. IEEE, 2024, pp. 23 975–23 984.

[16] W. Zhao, J. Li, X. Dong, Y. Tian, Y. Xiang, and Y. Guo, “Mitigating the id-ood tradeoff in open-set test-time adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, June 2026, pp. 6789–6798.

[17] Y. Zhang, H. Wang, F. Zhang, Q. Mi, M. Yang, Y. Wang, J. Wang, H. Li, and Z. Lin, “Learning to memorize with attributive and associative memory for online test-time adaptation of vision-language models,” in Forty-third International Conference on Machine Learning, 2026.

[18] D. Wang, E. Shelhamer, S. Liu, B. A. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[19] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Efficient test-time model adaptation without forgetting,” in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 16 888–16 905.

[20] J. Lee, D. Das, J. Choo, and S. Choi, “Towards open-set testtime adaptation utilizing the wisdom of crowds in entropy minimization,” in IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023. IEEE, 2023, pp. 16 380–16 389.

[21] Q. Zhang, Y. Bian, X. Kong, P. Zhao, and C. Zhang, “COME: test-time adaption by conservatively minimizing entropy,” in The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. Open-Review.net, 2025.

[22] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, “Towards stable test-time adaptation in dynamic wild world,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[23] M. Tan, G. Chen, J. Wu, Y. Zhang, Y. Chen, P. Zhao, and S. Niu, “Uncertainty-calibrated test-time model adaptation without forgetting,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 8, pp. 6274–6289, 2025.

[24] G. Chen, S. Niu, D. Chen, J. Yang, Z. Zhang, M. Tan, P. Wu, and Z. Shen, “Zerosiam: An efficient asymmetry for test-

time entropy optimization without collapse,” in International Conference on Learning Representations, 2026.

[25] X. Xia, B. Han, N. Wang, J. Deng, J. Li, Y. Mao, and T. Liu, “Extended \$t\$t: Learning with mixed closed-set and open-set noisy labels,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 3, pp. 3047–3058, 2023.

[26] S. Niu, G. Chen, P. Zhao, T. Wang, P. Wu, and Z. Shen, “Selfbootstrapping for versatile test-time adaptation,” in Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, ser. Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025, pp. 46 611–46 628.

[27] M. Yang, Y. Li, C. Zhang, P. Hu, and X. Peng, “Test-time adaptation against multi-modal reliability bias,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

[28] C. Fuchs, M. Zanella, and C. D. Vleeschouwer, “Online gaussian test-time adaptation of vision-language models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2025, Nashville, TN, USA, June 11-15, 2025. Computer Vision Foundation / IEEE, 2025, pp. 128–137.

[29] Q. Deng, S. Niu, R. Zhang, Y. Chen, R. Zeng, J. Chen, and X. Hu, “Learning to generate gradients for test-time adaptation via test-time training layers,” in AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA. AAAI Press, 2025, pp. 16 235–16 243.

[30] L. Zhou, M. Ye, S. Li, N. Li, X. Zhu, L. Deng, H. Liu, and Z. Lei, “Bayesian test-time adaptation for vision-language models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025. Computer Vision Foundation / IEEE, 2025, pp. 29 999–30 009.

[31] Z. Deng, G. Chen, S. Niu, H. Luo, S. Zhang, Y. Yang, R. Chen, W. Luo, and M. Tan, “Test-time model adaptation for quantized neural networks,” in Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025, pp. 7258–7267.

[32] J. Hu, Z. Zhang, G. Chen, X. Wen, C. Shuai, W. Luo, B. Xiao, Y. Li, and M. Tan, “Test-time learning for large language models,” in Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025.

[33] Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt, “Test-time training with self-supervision for generalization under distribution shifts,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 9229–9248.

[34] Y. Yu, L. Sheng, R. He, and J. Liang, “STAMP: outlier-aware test-time adaptation with stable memory replay,” in Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXI, ser. Lecture Notes in Computer Science, vol. 15139. Springer, 2024, pp. 375–392.

[35] H. Dong, E. N. Chatzi, and O. Fink, “Towards robust multimodal open-set test-time adaptation via adaptive entropy-aware optimization,” in The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[36] X. Wang, J. Lin, C. Chen, L. Tang, Y. Huang, C. Wang, L. Ye, and Y. Yao, “Activate and adapt: A two-stage framework for open-set model adaptation,” Trans. Mach. Learn. Res., vol. 2025, 2025.

[37] M. Sreenivas and S. Biswas, “Efficient open set single image test time adaptation of vision language models,” Trans. Mach. Learn. Res., vol. 2025, 2025.

[38] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,”

in 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.

[39] S. Liang, Y. Li, and R. Srikant, “Enhancing the reliability of out-of-distribution image detection in neural networks,” in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.

[40] W. Liu, X. Wang, J. D. Owens, and Y. Li, “Energy-based outof-distribution detection,” in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[41] X. Cheng, X. Jiang, and Z. Li, “A unified interpretation of training-time out-of-distribution detection,” in IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025. IEEE, 2025, pp. 2142– 2151.

[42] Q. Wang, B. Han, Y. Liu, C. Gong, T. Liu, and J. Liu, “W-DOE: wasserstein distribution-agnostic outlier exposure,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3530–3545, 2025.

[43] Z. Hu, Q. Wang, X. Liu, L. Lan, and B. Han, “On the two facets to conquer wild out-of-distribution detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 7, pp. 8060–8074, 2026.

[44] W. Chen, R. A. Yeh, S. Mou, and Y. Gu, “Leveraging perturbation robustness to enhance out-of-distribution detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025. Computer Vision Foundation / IEEE, 2025, pp. 4724– 4733.

[45] C. Du, Y. Wang, S. Song, and G. Huang, “Probabilistic contrastive learning for long-tailed visual recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 9, pp. 5890–5904, 2024.

[46] J. Li, Y. Wang, X. Zhang, D. Jiang, W. Dai, C. Li, H. Xiong, and Q. Tian, “Contrastive learning via variational information bottleneck,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 9, pp. 7410–7427, 2025.

[47] M. Zheng, S. You, F. Wang, C. Qian, C. Zhang, X. Wang, and C. Xu, “Weak augmentation guided relational self-supervised learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 8502–8516, 2024.

[48] T. Chen, S. Kornblith, M. Norouzi, and G. E. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 1597–1607.

[49] K. He, H. Fan, Y. Wu, S. Xie, and R. B. Girshick, “Momentum contrast for unsupervised visual representation learning,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 2020, pp. 9726–9735.

[50] J. Grill, F. Strub, F. Altche, C. Tallec, P. H. Richemond,´ E. Buchatskaya, C. Doersch, B. A. Pires, Z. Guo, M. G. Azar,<sup>´</sup> B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent - A new approach to self-supervised learning,” in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[51] X. Chen and K. He, “Exploring simple siamese representation learning,” in IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021. Computer Vision Foundation / IEEE, 2021, pp. 15 750–15 758.

[52] J. Wu, H. Yang, T. Gan, N. Ding, F. Jiang, and L. Nie, “CHMATCH: contrastive hierarchical matching and robust adaptive threshold boosted semi-supervised learning,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 2023, pp. 15 762–15 772.

[53] Y. Chen, S. Niu, Y. Wang, S. Xu, H. Song, and M. Tan, “Towards robust and efficient cloud-edge elastic model adaptation via selective entropy distillation,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

[54] M. Sensoy, L. M. Kaplan, and M. Kandemir, “Evidential deep learning to quantify classification uncertainty,” in Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montreal, Canada´ , 2018, pp. 3183–3193.

[55] H. Ma, J. Chen, J. T. Zhou, G. Wang, and C. Zhang, “Estimating llm uncertainty with evidence,” arXiv preprint arXiv:2502.00290, 2025.

[56] J. Katz-Samuels, J. B. Nakhleh, R. D. Nowak, and Y. Li, “Training OOD detectors in their natural habitats,” in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 10 848–10 865.

[57] Y. Li, N. Wang, J. Shi, J. Liu, and X. Hou, “Revisiting batch normalization for practical domain adaptation,” in 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017.

[58] Q. Wang, O. Fink, L. V. Gool, and D. Dai, “Continual test-time domain adaptation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 2022, pp. 7191–7201.

[59] D. Hendrycks and T. G. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

[60] S. Zagoruyko and N. Komodakis, “Wide residual networks,” in Proceedings of the British Machine Vision Conference 2016, BMVC 2016, York, UK, September 19-22, 2016. BMVA Press, 2016.

[61] F. Croce, M. Andriushchenko, V. Sehwag, E. Debenedetti, N. Flammarion, M. Chiang, P. Mittal, and M. Hein, “Robustbench: a standardized adversarial robustness benchmark,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021.

[62] L. v. d. Maaten and G. Hinton, “Visualizing data using t-sne,” Journal of machine learning research, vol. 9, no. Nov, pp. 2579– 2605, 2008.