# IECG: Weakly-Supervised Interpretability-Enhancing Foundation Model for Electrocardiogram

Huynh Dang Nguyen<sup>1</sup> Trong-Thang Pham<sup>2</sup> Anh-Thuan Do<sup>3,†</sup> Phuong-Nhi Dinh<sup>4,†</sup> Brijesh Patel<sup>5</sup> Ngan Le<sup>2,∗</sup> Van Nguyen<sup>1,∗</sup>

<sup>1</sup>Quantum AI & Cyber Security Institute (QACI) - FPT Corporation, Vietnam   
<sup>2</sup>University of Arkansas, USA <sup>3</sup>Hospital Haiphong Medical University, Vietnam   
<sup>4</sup>Pham Ngoc Thach University of Medicine, Vietnam <sup>5</sup>Indiana University, USA {HuynhND11, VanNTH19}@fpt.com, {tp030, thile}@uark.edu 2151010317@hpmu.edu.vn, 2051010735@pnt.edu.vn, bpatel9@iuhealth.org

## Abstract

Electrocardiography (ECG) is a cornerstone of cardiac diagnosis, yet most machine learning methodsfor ECG analysis lack interpretability, limiting their clinical adoption. We present IECG, a weakly-supervised ECG foundation model that jointly performs diagnostic classification and abnormality localization without requiring manual abnormal region annotations, overcoming a long-standing bottleneck in interpretable ECG learning. Central to IECG is a weakly-supervised attention strategy that explicitly incorporates clinical priors by deriving abnormality heatmap pseudo-labels from established ECG clinical knowledge. These pseudo-labels guide the model’s CLS-token attention through a KL-divergence loss, encouraging it to focus on diagnostically relevant waveform regions. During pretraining, IECG is jointly optimized with clinically guided attention supervision, ECG-text contrastive alignment, and report captioning objectives, enabling it to learn rich multimodal representations while producing localized waveform evidence. By explicitly aggregating features from clinically guided attention regions, IECG grounds its diagnostic predictions in interpretable waveform evidence without sacrificing classification performance. Extensive experiments demonstrate that IECG achieves state-of-the-art abnormal region localization while maintaining strong classification performance, advancing trustworthy foundation models for ECG analysis.

## 1. Introduction

Deep learning architectures for electrocardiogram (ECG) analysis [21, 23, 24, 37] have achieved remarkable success in automated cardiac diagnosis. However, despite their strong predictive performance, these models remain largely uninterpretable: they predict what abnormality is present but fail to explain where the supporting evidence resides within the ECG waveform. This limitation substantially hinders their clinical adoption, as physicians rely not only on diagnostic predictions but also on verifying the underlying waveform morphology. An ECG analysis model should therefore identify both the predicted diagnosis and the localized waveform evidence supporting that prediction.

Despite its importance, abnormality localization remains an unsolved problem for ECG foundation models. The primary challenge is the lack of localized supervision. Largescale ECG datasets such as MIMIC-IV-ECG [5] and PTB-XL [32] provide only recording-level diagnostic labels and contain no annotations identifying the waveform regions corresponding to each diagnosis. Consequently, existing models are optimized solely for classification and cannot learn to consistently associate predictions with clinically meaningful waveform regions. While various methods can highlight irregular beats [4, 8], they fail to provide specific diagnoses associated with those localized regions.

In this work, we address this challenge by exploiting a key observation: although localized annotations are unavailable, diagnostic criteria for many ECG abnormalities are well established in clinical practice. For example, a PR interval exceeding 200 ms is a diagnostic criterion for firstdegree AV block, while ST-segment elevation greater than 0.1 mV indicates myocardial injury. We formalize these clinical heuristics into an automatic pseudo-label generation pipeline that produces abnormality heatmaps directly from ECG signals. By explicitly incorporating these clinical priors, we transform localization from a manual annotation problem into a weakly-supervised learning problem.

Building upon this insight, we propose IECG, an attention-enhanced interpretable ECG foundation model that jointly performs classification and abnormality localization. During training, clinically derived abnormality heatmaps supervise the model’s attention through a Kullback-Leibler (KL) divergence objective, encouraging attention to align with diagnostically relevant waveform regions while learning disease classification. As a result, IECG produces both diagnostic predictions and localized visual evidence supporting those predictions. For example, when diagnosing atrial fibrillation, the model highlights the irregularly absent P-wave regions that cardiologists naturally inspect during clinical interpretation.

Extensive experiments demonstrate that IECG achieves state-of-the-art (SOTA) abnormal region localization, reaching 75.55% AUROC and outperforming existing attribution methods, including approaches specifically designed for this task, while archiving competitive classification performance. These findings suggest that clinically guided attention supervision can significantly enhance the interpretability of ECG foundation models without requiring manual attention annotations.

Our main contributions are summarized as follows:

• We propose IECG, an ECG foundation model that jointly performs classification and abnormality localization without requiring manual attention annotations.

• We introduce a weakly-supervised attention supervision framework that incorporates clinical priors to automatically generate abnormality heatmap pseudo-labels from established ECG diagnostic heuristics, aligning model attention with clinically meaningful waveform regions.

• We demonstrate SOTA abnormality localization and competitive classification performance, highlighting the effectiveness of clinically guided attention supervision for improving the interpretability of ECG foundation models.

## 2. Related Work

## 2.1. Foundation models for ECG

Foundation models have become a dominant approach for ECG analysis, typically pretrained on large-scale datasets and adapted to downstream tasks through transfer learning [21, 22, 30, 34]. However, existing approaches focus primarily on representation learning and predictive performance without explicitly addressing interpretability.

Recent work has explored both signal-only and multimodal pretraining strategies. METS [14], MERL [17], and TolerantECG [24] adopt multimodal alignment to learn joint representations between ECG signals and diagnostic reports. These methods improve semantic representation quality but do not explicitly localize diagnostic evidence within the waveform. In contrast, signal-only selfsupervised methods such as ST-MEM [23], ECG-FM [21], and HeartLang [11] leverage masked reconstruction or contrastive learning to capture intrinsic spatio-temporal representations of ECG signals.

Existing ECG foundation models focus on representation, yet they do not explicitly solve where in the waveform the evidence for a diagnosis lies. In contrast, IECG is designed to enhance interpretability by construction: it introduces CGAL to generate attention pseudo-labelsfrom clinical heuristics and uses these labels to directly supervise the model’s CLS-token attention via a KL-divergence loss, enabling anomaly localization without manual attention annotation while preserving strong downstream performance.

## 2.2. Interpretable AI and Localization

Explainable AI aims to enhance transparency in deep learning systems, which is particularly important in the medical domain, where clinicians must understand a model’s reasoning before trusting its predictions.

Saliency-based methods [29, 33] compute gradients of the model output with respect to the input to estimate feature importance. While these approaches can highlight influential regions, they often produce noisy and unstable explanations. In contrast, Grad-CAM [25, 28] generates coarse-resolution heatmaps by leveraging gradients and intermediate feature maps. However, in the context of ECG signals, such heatmaps primarily capture temporal importance and do not explicitly disentangle contributions from individual leads, which limits their clinical interpretability.

For Transformer architectures, attention calibration [20, 39] introduces auxiliary objectives to guide attention distributions toward more meaningful and interpretable patterns. Existing approaches typically derive such supervision signals from model-internal mechanisms, such as perturbationbased importance estimation or structural constraints, rather than relying on explicit ground-truth attention annotations.

However, these generic calibration strategies are not tailored to domain-specific interpretability requirements. In the ECG domain, clinically meaningful attention should align with abnormal waveform regions, yet such supervision signals are not readily available in existing datasets Our work addresses this gap by automatically generating pseudo-labels from clinical heuristics, enabling domainspecific attention calibration without manual labeling.

## 3. Model Architecture

As illustrated in Figure 1, IECG consists of three main components: the ECG Encoder, ECG-Text Alignment, and CGAL. Given an ECG signal $\mathbf { X } \in \mathbb { R } ^ { L \times T }$ and its corresponding text report R, IECG learns multimodal representations while simultaneously learning to focus on clinically

![](images/2739d64939a5a6be4c40c1c5a31b1268d3c05e9b9360302e14c40e32b9ab2667.jpg)  
Figure 1. IECG overview. The ECG Encoder learns representations through ECG-Text Alignment and a self-supervised clinical abnormality localization mechanism with CGAL pipeline.

relevant regions.

The ECG Encoder (center) processes the raw 12-lead ECG through patching, Spatial-Temporal (ST) embedding (Section 3.1), and a Vision Transformer (ViT) to produce ECG representations. The CLS token aggregates global information, and its attention scores over the ECG patches form the basis for interpretability. ECG-Text Alignment (left) aligns ECG representations with text reports preprocessed by the Criteria Feature Retrieval (CFR) module, using a frozen Text Encoder and a Multimodal Text Decoder optimized through contrastive $( \mathcal { L } _ { c o n t r a s t } )$ and captioning $( \mathcal { L } _ { c a p t i o n } )$ losses (Section 3.2). CGAL (right) addresses the annotation bottleneck by automatically generating pseudolabeled abnormal heatmaps. The process begins by segmenting ECG waveforms, after which the Abnormal Seeking module identifies irregular regions using clinical heuristics. Subsequently, the Criteria Filtering module refines these regions into pseudo labels based on clinical criteria retrieved by CFR. The attention loss $( \mathcal { L } _ { a t t n } )$ then guides the CLS attention toward these clinically meaningful regions (Section 3.3).

## 3.1. Spatial-Temporal Embedding

Given a raw 12-lead ECG signal $\mathbf { X } \in \mathbb { R } ^ { L \times T }$ , where $L = 1 2$ denotes the number of leads and $T = 5 0 0 0$ is the temporal length (10 seconds at 500Hz), we first apply a linear patch embedding layer. Each lead is divided into $P = T / s =$ 100 non-overlapping patches of size $s = 5 0 ,$ , yielding patch embeddings $\mathbf { Z } ^ { ( l ) } \in \mathbb { R } ^ { P \times D }$ for each lead $l \in \{ 0 , \ldots , L - 1 \}$ }, where D is the hidden dimension. We then flatten across leads by concatenating all patches sequentially (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6):

$$
{ \bf Z } = [ { \bf Z } ^ { ( 0 ) } ; { \bf Z } ^ { ( 1 ) } ; \ldots ; { \bf Z } ^ { ( L - 1 ) } ] \in \mathbb { R } ^ { N \times D } ,\tag{1}
$$

where $N = L \times P = 1 2 0 0$ is the number of tokens.

![](images/a0c2548550c4d9672ecf9978dccc859566c72b90e98833f3c8ea02aae126a6ac.jpg)  
Figure 2. Each ECG token is enriched with Spatial and Temporal embeddings in addition to Positional embeddings, providing explicit lead identity and time frame information.

In addition to the Transformer’s positional embedding $\mathbf { E } \in \mathbb { R } ^ { N \times D }$ , we incorporate the Spatial-Temporal (ST) embedding proposed by Jin et al. [11] to enhance the model’s ability to capture lead-specific and temporal information, as illustrated in Figure 2. For spatial embedding, we define $\mathbf { E } ^ { s } \in \mathbb { R } ^ { L \times D }$ , assigning a unique embedding to each lead $( { \mathrm { e . g . , ~ } } E _ { \mathrm { I } } ^ { s } , E _ { \mathrm { I I } } ^ { s } , \dots , E _ { \mathrm { V 6 } } ^ { s } )$ . For temporal embedding, we define $\mathbf { E } ^ { t } \in \overset {  } { \mathbb { R } } ^ { P \times D }$ , encoding the position within each lead $( \mathrm { i . e . , } E _ { 0 } ^ { t } , E _ { 1 } ^ { t } , \ldots , E _ { P - 1 } ^ { t } )$ . The input to the Transformer is:

$$
\mathbf { H } _ { n } ^ { ( 0 ) } = \mathbf { Z } _ { n } + \mathbf { E } _ { n } + \mathbf { E } _ { l ( n ) } ^ { s } + \mathbf { E } _ { t ( n ) } ^ { t }\tag{2}
$$

where $l ( n ) ~ = ~ \lfloor n / P \rfloor$ and $t ( n ) ~ = ~ n$ mod P for $n =$ $0 , \ldots , N - 1$ . Here ${ \bf H } _ { n } ^ { ( 0 ) } \in \mathbb { R } ^ { D }$ is the n-th token’s input representation (0-indexed), combining its patch embedding with positional embedding $( E _ { 0 } , E _ { 1 } , \dots , E _ { N - 1 } )$ , spatial embedding, and temporal embedding.

## 3.2. ECG Signal and Text Report Alignment

Once the ECG tokens are enriched with spatial-temporal information, they are passed through the ECG Encoder, a standard Vision Transformer with K layers. A learnable CLS token $\mathbf { h } _ { c l s } ^ { ( 0 ) } \in \mathbb { R } ^ { D }$ is prepended to the sequence:

$$
\tilde { \mathbf { H } } ^ { ( 0 ) } = [ \mathbf { h } _ { c l s } ^ { ( 0 ) } ; \mathbf { H } ^ { ( 0 ) } ] \in \mathbb { R } ^ { ( N + 1 ) \times D } .\tag{3}
$$

The encoder applies K transformer layers with multi-head self-attention (MHSA) and feed-forward networks (FFN):

$$
\tilde { \mathbf { H } } ^ { ( k ) } = \mathrm { F F N } ( \mathrm { M H S A } ( \tilde { \mathbf { H } } ^ { ( k - 1 ) } ) ) , \quad k = 1 , \dots , K .\tag{4}
$$

The final CLS token $\mathbf { h } _ { c l s } ^ { ( K ) } \in \mathbb { R } ^ { D }$ serves as the global ECG representation E for downstream tasks.

Following previous ECG pretraining approaches [35, 37], we employ two pretraining objectives: contrastive loss for representation learning and captioning loss for semantic alignment. Before encoding, text reports are processed by the Criteria Feature Retrieval (CFR) module [24], which retrieves relevant diagnostic criteria to enrich each diagnosis with their corresponding features. Following CoCa [36], the Multimodal Text Decoder employs causal self-attention and cross-attention over ECG encoder outputs for autoregressive report generation. The ECG Encoder and Multimodal Text Decoder are jointly optimized while the Text Encoder remains frozen.

Optimizing the contrastive loss enables alignment between ECG features $\textbf { E } \in \ \mathbb { R } ^ { D }$ and corresponding text embeddings $\textbf { T } \in \ \mathbb { R } ^ { D }$ . For a batch of B samples, let $\{ \mathbf { E } _ { i } , \mathbf { T } _ { i } \} _ { i = 1 } ^ { B }$ denote the paired ECG and text representations, and τ the temperature parameter:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { c o n t r a s t } = - \frac { 1 } { 2 B } ( \sum _ { i = 1 } ^ { B } \log \frac { e ^ { \mathbf { E } _ { i } ^ { \top } \mathbf { T } _ { i } / \tau } } { \sum _ { j = 1 } ^ { B } e ^ { \mathbf { E } _ { i } ^ { \top } \mathbf { T } _ { j } / \tau } }  } } \\ & { } & {  + \sum _ { i = 1 } ^ { B } \log \frac { e ^ { \mathbf { T } _ { i } ^ { \top } \mathbf { E } _ { i } / \tau } } { \sum _ { j = 1 } ^ { B } e ^ { \mathbf { T } _ { i } ^ { \top } \mathbf { E } _ { j } / \tau } } ) } \end{array}\tag{5}
$$

Optimizing the captioning loss encourages the ECG encoder to produce representations that enable accurate report generation. Given the ECG token sequence ${ \bf H } ^ { ( K ) } \in$ $\mathbf { \mathbb { R } } ^ { ( N ) \times D }$ from the encoder (excluding CLS token) and target text tokens $\mathbf { t } = ( t _ { 1 } , \dots , t _ { M } )$ of length M, the Multimodal Text Decoder with parameters θ autoregressively predicts each token, with the corresponding loss function:

$$
\mathcal { L } _ { c a p t i o n } = - \mathbb { E } \Bigg [ \sum _ { m = 1 } ^ { M } \log P _ { \theta } ( t _ { m } | t _ { 1 : m - 1 } , \mathbf { H } ^ { ( K ) } ) \Bigg ]\tag{6}
$$

## 3.3. Clinically-Guided Attention Localization (CGAL)

CGAL is the key component that solves the annotation bottleneck. It generates attention pseudo-labels during training by encoding established clinical knowledge into a rulebased algorithm. The overall procedure is detailed in Section A of the Supplementary Material.

The first stage, Wave Segmenting, focuses on segmenting the ECG waveform into its constituent components. We begin by denoising the ECG signal through successive application of a high-pass filter, notch filter, and low-pass filter. A pretrained UNet3+ model [6, 13], trained on SemiSegECG [26] and RDB [19] datasets and frozen during our training, produces a segmentation map ${ \textbf { S } } \in$ $\{ 0 , 1 , 2 , 3 \} ^ { T }$ , where labels correspond to background (0), P wave (1), QRS complex (2), and T wave (3). The detailed performance of this segmentation module is reported in Section B of the Supplementary material.

The second stage, Abnormal Seeking, identifies candidate abnormal regions using clinical definitions from the Life in the Fastlane (LITFL) ECG library [1]. Firstly, when P wave is not visible (e.g., in atrial fibrillation), we insert a pseudo P wave between the preceding T wave and the current Q wave, and highlight this region to indicate the absence of a true P wave. Based on the segmented waveform boundaries, clinically relevant segments and intervals are computed, including the PR interval, defined as the interval from P wave onset to QRS onset, as well as the QT interval and the ST segment. For QT interval analysis, we apply Bazett’s correction to ensure clinical validity: $Q T c = Q T / \sqrt { R R }$ , where $Q T$ is the measured interval and RR is the preceding R-R interval. This correction accounts for the dependence of the QT interval on heart rate. Finally, regions exceeding clinically established amplitude and duration thresholds, as defined in Section C of the Supplementary material, are identified and flagged as potential abnormalities.

![](images/6c8e869f30753fe3da61f7ec6559a84d7e0108f23e95d472bb14eff07f63c79c.jpg)  
Figure 3. The Criteria Feature Retrieval (CFR) module retrieves diagnostic criteria by comparing the input diagnosis embedding with a pre-constructed vector database.

The third stage, Criteria Filtering, filters these candidates using the CFR module [24], illustrated in Figure 3. CFR maintains a vector database $\mathcal { D } ~ = ~ \{ ( \mathbf { d } _ { j } , c _ { j } ) \} _ { i = 1 } ^ { | \mathcal { D } | }$ of diagnosis-criteria pairs sourced from the LITFL ECG library [1], where $\mathbf { d } _ { j } \in \mathbb { R } ^ { D }$ is the embedding of diagnosis j and $c _ { j }$ is its defining criteria (e.g., “prolonged PR interval $> 2 0 0 \mathrm { m s } ^ { , , }$ for first-degree AV block). Given an input diagnosis with embedding $\mathbf { q } \in \mathbb { R } ^ { D }$ , the CFR module retrieves the most similar criteria via cosine similarity.

We retain only those flagged regions that match the retrieved criteria. For example, regions corresponding to prolonged PR intervals are preserved for first degree atrioventricular block, while unrelated findings such as ST segment abnormalities are discarded. When no matching criteria can be retrieved, a fallback procedure is applied, which uses the input query itself to identify candidate abnormal regions, as detailed in Section D of the Supplementary material.

The selected regions are then converted into a binary token mask $\mathbf { m } \in \{ 0 , 1 \} ^ { L \cdot T }$ , where a value of 1 indicates that the corresponding token lies within a flagged region in $\mathcal { R }$ and a value of 0 denotes all other positions. When multiple diagnoses are associated with a signal, the final mask is obtained by applying a logical OR operation across the diagnosis-specific masks.

Attention Supervision. From the preceding stages, we obtain a binary mask $\mathbf { m } _ { i } \in \{ 0 , 1 \} ^ { L \cdot T }$ for sample i, where $m _ { i , k } ~ = ~ 1$ indicates an abnormal region at timestamp k. However, since the attention token got downsampled after proccessed by ECG encoder, we apply a MaxPool operation with kernel size and stride same as patch size s. We further normalize this mask into a target probability distribution $\mathbf { p } _ { i } \in \mathbb { R } ^ { N }$

$$
\tilde { \mathbf { m } _ { i } } = \mathbf { M a x P o o l } ( \mathbf { m } _ { i } , \mathrm { k e r n e l . s i z e } = s , \mathrm { s t r i d e } = s ) \in \mathbb { R } ^ { N }\tag{7}
$$

$$
\begin{array} { r } { \mathbf p _ { i , n } = \tilde { \mathbf m } _ { i , n } / \sum _ { j = 0 } ^ { N - 1 } \tilde { \mathbf m } _ { i , j } , \quad n = 0 , \dots , N - 1 } \end{array}\tag{8}
$$

To extract the model’s global attention pattern, we compute the aggregated Attention Rollout across all transformer layers [2]. This technique accounts for the flow of information through residual connections and successive selfattention operations. For a given layer $k \in \{ 1 , \ldots , K \}$ let $\mathbf { W } ^ { ( k ) } \in \mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$ denote the attention matrix averaged across all H heads:

$$
\mathbf { W } ^ { ( k ) } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } ^ { ( k , h ) } \mathbf { K } ^ { ( k , h ) } } { \sqrt { d _ { h } } } \right)\tag{9}
$$

To incorporate residual connections, the attention weights are augmented with an identity matrix and re-normalize [2]:

$$
\tilde { \mathbf { W } } ^ { ( k ) } = ( \mathbf { W } ^ { ( k ) } + \mathbf { I } ) / 2\tag{10}
$$

The rollout attention matrix $\mathbf { R } ^ { ( k ) }$ is computed recursively by multiplying the augmented attention matrices from the current layer with the rollout matrix from the previous layer:

$$
\mathbf { R } ^ { ( k ) } = \tilde { \mathbf { W } } ^ { ( k ) } \mathbf { R } ^ { ( k - 1 ) }\tag{11}
$$

where $\mathbf { R } ^ { ( 0 ) } = \mathbf { I }$ . The final aggregated attention scores from the CLS token to the N ECG tokens are extracted from the CLS-token row of the final rollout matrix $\mathbf { R } ^ { ( K ) }$ , denoted as $\mathbf { A } _ { i } ^ { r o l l o u t } \in \mathbb { R } ^ { N }$ . The attention loss aligns this aggregated distribution with the pseudo-label via KL divergence:

$$
\mathcal { L } _ { a t t n } = \mathbb { E } \Bigg [ \sum _ { i = 1 } ^ { B } \sum _ { n = 0 } ^ { N - 1 } \mathbf { p } _ { i , n } \log ( \mathbf { p } _ { i , n } / \mathbf { A } _ { i , n } ^ { r o l l o u t } ) \Bigg ]\tag{12}
$$

By utilizing attention rollout, the supervision signal explicitly guides the hierarchical information aggregation process of the entire Transformer, rather than constraining only the final block. This design ensures that intermediate layers progressively learn to filter and route relevant morphological features toward the CLS token, leading to more robust and coherent localization.

To further evaluate the effectiveness of this design, we additionally consider a variant in which attention supervision is applied only to the CLS-token attention scores from the final Transformer layer.

## 3.4. Training Objective

The overall training loss combines all objectives:

$$
\mathcal { L } _ { t o t a l } = \alpha \mathcal { L } _ { c o n t r a s t } + \beta \mathcal { L } _ { c a p t i o n } + \gamma \mathcal { L } _ { a t t n }\tag{13}
$$

where $\alpha , \beta$ and γ control the relative contributions of contrastive learning, captioning, and attention supervision, respectively. For samples where a valid heatmap cannot be constructed by CGAL, attention supervision is omitted, and training proceeds using only ${ \mathcal { L } } _ { \mathrm { c o n t r a s t } }$ and $\mathcal { L } _ { \mathrm { c a p t i o n } }$

Through this composite objective, contrastive loss aligns ECG features with textual representations, captioning loss enables accurate generation of clinical descriptions, and attention-guiding loss regularizes the model to focus on medically relevant regions.

## 3.5. Down-stream Evaluation

After pretraining, the ECG encoder can be deployed for a variety of downstream tasks. One direct application is zero-shot abnormal region localization, achieved by collecting the output attention scores from input signal. CGAL is omitted during inference since the model has already learned to localize relevant regions. In addition, by attaching a linear classification layer on top of the encoder, the model can be further fine-tuned for classification. Importantly, since only the classifier head is updated while the encoder remains frozen (linear probing), the ability to attend to clinically contributing regions is preserved.

## 4. Experiment

## 4.1. Datasets

We pretrain on MIMIC-IV-ECG [5], comprising over 800,000 12-lead ECG recordings (10-second duration) from approximately 160,000 patients. During preprocessing, we replace NaN/Inf values with zero. CGAL generates attention pseudo-labels without external annotations.

For abnormality localization, we benchmark using the dataset proposed by Jiang et al. [7], which we refer to as $\mathrm { E C G A D _ { d a t a } }$ , consistent with the model’s name $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ The dataset consists of 400 PTB-XL test samples annotated by two cardiologists. Among these, only 190 samples are abnormal and include heatmap annotations. We use these annotated cases for benchmarking.

For classification evaluation, since $\mathrm { E C G A D _ { d a t a } }$ is a subset of PTB-XL [32], we utilize the corresponding labels of these samples for this task. In addition, we evaluate on the full PTB-XL dataset, a clinical 12-lead ECG corpus comprising 21,799 recordings from 18,869 patients, using the predefined train, validation, and test splits. We benchmark on All-level, which includes all 71 diagnostic categories.

## 4.2. Configuration

We use a 1D 12-layer Vision Transformer with 12 attention heads, a hidden width of 768, and a patch size of 50 as the ECG encoder. The text encoder (MedCPT [12]) remains frozen. We flatten the 12-lead ECG signal to ensure the CLS token attends to all tokens. Training uses learning rate 1e − 4, 20 epochs, batch size 32 on 4× A100 GPUs. We set $\alpha = 1 , \beta = 1$ (following [35]), and $\gamma = 0 . 5$

## 4.3. Metrics

Abnormality localization. To evaluate whether the attention scores of IECG accurately highlight relevant regions within each data sample, we frame this as a ranking task where the model should assign higher attention to abnormal tokens. We employ Precision at K (P@K) (↑) where $\mathrm { K } = 5 0 \%$ of the total token count (K=30,000 for $L \times T =$ 60, 000), as the large sequence length makes fixed small K values less discriminative. We also report point-level Area Under the Receiver Operating Characteristic Curve (AU-ROC) (↑), following the protocol in [7].

Classification.: We evaluate classification performance using FMax (↑), the maximum F1 score over all thresholds [31], summarizing precision–recall balance. We also report macro-averaged AUROC (↑), which measures overall discriminative capability across all classes regardless of class imbalance. For clarity, we denote localization and classification AUROC as $\mathrm { { A U R O C } _ { \mathrm { { l o c } } } }$ and $\mathbf { A U R O C } _ { \mathrm { c l s } } ,$ , respectively.

Faithfulness.: We evaluate the faithfulness of the generated attention maps using Insertion (↑) and Deletion (↓) tests [27]. Specifically, we measure AUROC after progressively inserting or deleting the top-k attended regions, where $k \in \{ 1 0 \% , 2 0 \% , . . . , 1 0 0 \% \}$

## 4.4. Baselines

We compare IECG against a diverse set of baseline methods that cover both abnormality localization and classification.

To ensure a fair comparison, we retrain METS [14], MERL [17], and TolerantECG [24] using the same encoder architecture and training dataset as IECG. For ECG-FM [21] and HeartLang [11], we use the official pretrained checkpoint. In addition, since ST-MEM [23] was not originally trained on the MIMIC-IV-ECG dataset [5], we retrain it to match our configuration.

During evaluation, ST-MEM and ECG-FM do not include a CLS token. Following the design choices in their original implementations, where token representations are aggregated using average pooling before being passed to a classification head. We adopt the same aggregation strategy to derive the corresponding attention scores. In contrast, because METS, MERL, HeartLang, and TolerantECG share the same architecture as our model, their attention scores are extracted using the same procedure as ours.

For XAI4ECG [33] and ECG-XPLAIM [25], which generate attribution maps for interpretability, we retrain the models following their released implementations. However, ECG-XPLAIM is trained in a task-specific supervised setting and is designed for a limited set of classification tasks rather than the full multi-label All-level setting in PTB-XL. As a result, its training protocol is not directly compatible with our classification benchmark; hence classification and faithfulness results are not reported for this model.

In contrast, $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ [7] is specifically designed for abnormality localization, and we use the publicly available checkpoint weights. Following [9], the latent representations produced by $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ are passed through a ResNet classifier for downstream classification.

In all cases, we directly use the attribution or localization maps produced by each method as attention scores for evaluation.

## 5. Results and Analysis

## 5.1. Localization and Classification Results

To evaluate the model’s ability to identify abnormalities, the pretrained IECG is directly applied to $\mathrm { E C G A D _ { \mathrm { d a t a } } }$ for inference, and the task is formulated as a binary pointwise classification problem [7]. The ground truth labels are derived from expert annotations, which ensures that the evaluation remains independent of the pseudo labels used during training. To align the predicted attention heatmaps with the ECG input and the corresponding labels, we upsample the heatmaps using interpolation prior to metric computation.

For classification, since the samples in this dataset are derived from PTB-XL, we retrieve the corresponding labels for each sample. We then perform classification by appending a linear classification head to each model and training only this head on the PTB-XL training split, while keeping the backbone parameters frozen. The trained classifiers are subsequently evaluated on the $\mathrm { E C G A D _ { d a t a } }$ samples.

Table 1 summarizes the performance of all methods on both abnormality localization and classification tasks, as well as explanation faithfulness. Overall, both IECG variants achieve the best localization performance across all metrics while obtaining highly competitive classification and faithfulness results, demonstrating its ability to jointly optimize predictive accuracy and clinically meaningful interpretability.

In terms of abnormality localization, both variants of IECG consistently outperform all baseline methods by a clear margin across P@50% and $\mathrm { { \bf A U R O C } _ { \mathrm { l o c } } }$ . In particular, IECG-Rollout achieves the best $\mathrm { { A U R O C } _ { \mathrm { { l o c } } } }$ of 75.55%, exceeding $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ (71.62%), which is specifically designed for localization, by a notable margin. This result highlights the effectiveness of incorporating CGAL during training, enabling the model to focus on diagnostically relevant waveform regions without requiring manual annotations. The improvement in P@50% further indicates that IECG ranks abnormalities more accurately, which is important for clinical interpretability.

In contrast, foundation models such as METS, MERL, and ECG-FM exhibit localization performance that is closer to random, with $\mathrm { { \bf A U R O C } _ { \mathrm { l o c } } }$ ranging from 50% to 61%. This suggests that optimizing solely for classification does not guarantee meaningful attention behavior. Although explainability-focused methods such as XAI4ECG and ECG-XPLAIM provide moderate improvements over some base models, their performance remains substantially below that of IECG, indicating that post hoc explanation methods may lack precise alignment with clinically relevant regions.

As for classification, IECG-Last achieves the highest FMax of 42.41%, while TolerantECG attains the highest $\mathrm { \mathbf { A U R O C } _ { c l s } }$ of 90.26%, with IECG-Last being highly competitive at 90.03%. This marginal gap suggests that introducing attention supervision does not significantly compromise classification performance. Instead, IECG preserves strong discriminative capability while substantially improving interpretability. In terms of explanation faithfulness, IECG-Rollout attains the highest Insertion score of 45.86%, confirming that the highlighted regions are indeed crucial for the model’s predictions, though some baselines report lower Deletion scores.

![](images/2cfabe48da8701d0eb89fe461e2c1066faf5c183526b45f478af8f5c34391d85.jpg)

Table 1. Abnormality localization and Classification results on $\mathrm { E C G A D _ { d a t a } }$ . Bold indicates the best result, and underline denotes the second-best result. For IECG, attention supervision is applied to either the attention scores from the final Transformer layer (IECG-Last) or the layer-wise aggregated attention obtained via attention rollout (IECG-Rollout).
<table><tr><td rowspan=4 colspan=1>Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>Interpretable</td><td rowspan=3 colspan=1>AbnormalityP@50% (↑)</td><td rowspan=3 colspan=1>localization $\mathbf { A U R O C _ { l o c } \left( \uparrow \right) }$ </td><td rowspan=2 colspan=5>Classification</td><td></td><td></td></tr><tr><td rowspan=2 colspan=2>FaithfulnessInsertion (↑) Deletion (↓)</td></tr><tr><td rowspan=1 colspan=1>FMax (↑)</td><td rowspan=1 colspan=4> $\mathbf { A U R O C _ { \mathrm { c l s } } \left( \uparrow \right) }$ </td></tr><tr><td rowspan=1 colspan=1>ST-MEM [23]</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $1 6 . 5 6 \pm 1 . 2 9$ </td><td rowspan=1 colspan=1> $5 0 . 0 7 \pm \mathrm { 0 . 6 5 }$ </td><td rowspan=1 colspan=1> $1 6 . 1 8 \pm \phantom { 0 } 0 . 1 2$ </td><td rowspan=1 colspan=4> $5 9 . 0 7 \ \pm \mathrm { 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $2 3 . 2 5 \pm \mathrm { 0 . 3 8 }$ </td><td rowspan=1 colspan=1>21.57 ±0.37</td></tr><tr><td rowspan=1 colspan=1>METS [14]</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>18.89 ±1.26</td><td rowspan=1 colspan=1>59.25 ±1.24</td><td rowspan=1 colspan=1>40.19 ±0.33</td><td rowspan=1 colspan=4>88.05 ±0.30</td><td rowspan=1 colspan=1>38.42 ±0.58</td><td rowspan=1 colspan=1>21.24 ±0.48</td></tr><tr><td rowspan=1 colspan=1>MERL [17]</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $1 9 . 7 3 \pm 1 . 2 8$ </td><td rowspan=1 colspan=1> $6 1 . 1 3 \pm 1 . 2 3$ </td><td rowspan=1 colspan=1>41.82 ±0.57</td><td rowspan=1 colspan=4> $8 8 . 8 0 \pm \mathrm { 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $3 5 . 5 0 \pm \mathrm { 0 . 2 8 }$ </td><td rowspan=1 colspan=1>20.07 ±0.54</td></tr><tr><td rowspan=1 colspan=1>ECG-FM [21]</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $1 6 . 2 7 \pm 1 . 2 1$ </td><td rowspan=1 colspan=1> $5 0 . 2 9 \pm \mathrm { 0 . 4 3 }$ </td><td rowspan=1 colspan=1> $3 4 . 3 7 \pm 0 . 4 3 $ </td><td rowspan=1 colspan=4> $8 3 . 4 1 \pm 0 . 5 2 $ </td><td rowspan=1 colspan=1> $2 5 . 2 3 \pm 0 . 3 6$ </td><td rowspan=6 colspan=1>12.09 ±0.1420.76 ±0.3422.75 ±1.02 $2 5 . 4 3 \pm \mathrm { 0 . 5 4 }$  $2 2 . 7 0 \pm \mathrm { 0 . 6 5 }$ </td></tr><tr><td rowspan=2 colspan=1>HeartLang [11]TolerantECG [24]</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $1 7 . 0 8 \pm 1 . 2 8$ </td><td rowspan=1 colspan=1> $5 0 . 3 2 \pm 1 . 1 2$ </td><td rowspan=1 colspan=1> $2 7 . 5 8 \pm 0 . 4 7$ </td><td rowspan=1 colspan=4> $7 7 . 0 7 \pm 0 . 6 4$ </td><td rowspan=1 colspan=1> $2 3 . 5 0 \pm \mathrm { 0 . 3 1 }$ </td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1> $1 8 . 2 5 \pm 1 . 2 2$ </td><td rowspan=1 colspan=1>59.15 ±1.21</td><td rowspan=1 colspan=1> $4 2 . 2 6 \pm 0 . 2 2$ </td><td rowspan=1 colspan=4> $\mathbf { 9 0 . 2 6 \_ } \mathbf { _ { 0 . 3 0 } }$ </td><td rowspan=4 colspan=1>35.89 ±0.86 $2 7 . 3 9 \pm 0 . 7 1$  $3 2 . 6 2 \pm 1 . 3 6$ </td></tr><tr><td rowspan=3 colspan=1>XAI4ECG [33]ECG-XPLAIM [25] $\mathrm { E C G A D } _ { \mathrm { m o d e l } } \ [ 7 ]$ </td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $1 6 . 6 3 \pm \mathrm { 0 . 0 8 }$ </td><td rowspan=1 colspan=1>51.44 ±0.32</td><td rowspan=1 colspan=1> $4 0 . 2 1 \pm 1 . 0 9$ </td><td rowspan=3 colspan=4> $8 8 . 0 2 \pm 0 . 4 4 $  $7 5 . 0 1 \pm 1 . 2 8$ </td></tr><tr><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $1 7 . 4 3 \pm \time 1 0 . 2 5$ </td><td rowspan=1 colspan=1>52.56 ±0.93</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1> $2 2 . 2 8 \pm 1 . 3 7$ </td><td rowspan=1 colspan=1> $7 1 . 6 2 \pm 1 . 2 1$ </td><td rowspan=1 colspan=1> $2 8 . 3 0 \pm \mathrm { 0 . 6 9 }$ </td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=1>IECG-LastIECG-Rollout</td><td rowspan=2 colspan=1>√√</td><td rowspan=2 colspan=1> $\underline { { 2 ^ { 3 } . 6 7 \pm 1 . 4 8 } }$  $2 4 . 4 2 \pm 1 . 4 8$ </td><td rowspan=1 colspan=1> $\underline { { 7 3 . 9 4 \pm 0 . 9 9 } }$ </td><td rowspan=1 colspan=1> $\pm 2 . 4 1 \pm \bf { 0 . 1 3 }$ </td><td rowspan=1 colspan=3>90.</td><td rowspan=2 colspan=1> $\underline { { 9 0 . 0 3 \pm 0 . 2 7 } }$ 89.09 ±0.49</td><td rowspan=2 colspan=1> $\underline { { 4 3 . 2 7 \pm 0 . 2 0 } }$  ${ \bf 4 5 . 8 6 _ { \pm 0 . 5 5 } }$ </td><td rowspan=2 colspan=1>26.18 ±0.42 $3 3 . 1 5 \pm 1 . 1 5$ </td></tr><tr><td rowspan=1 colspan=1>75.55 ±1.00</td><td rowspan=1 colspan=1> $4 2 . 0 0 \pm \mathrm { 0 . 4 3 }$ </td><td></td><td></td><td></td></tr></table>

Figure 4. Qualitative comparison of abnormality localization across different ECG conditions. For each example, the top panel shows the ECG waveform with ground-truth abnormalities highlighted in blue. The middle and bottom rows present the localization heatmaps produced by $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ and $\mathbb { T E C G } ,$ respectively. Compared to $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ , IECG produces more accurate and temporally consistent activations that better align with clinically relevant abnormalities across diverse conditions.

Overall, these results demonstrate that explicitly guiding attention using clinically informed pseudo labels produced by CGAL allows IECG to achieve a strong balance between localization, classification, and faithfulness. The proposed approach not only maintains state-of-the-art classification performance but also significantly improves the reliability and interpretability of the model’s predictions.

For additional classification evaluation on the full PTB-XL dataset, Table 2 presents the classification results on the PTB-XL dataset. Overall, IECG achieves the best performance across all metrics, demonstrating strong generalization when evaluated on large-scale ECG data.

Specifically, IECG-Last achieves the best performance across both evaluation metrics, attaining the highest FMax of 42.43% and $\mathbf { A U R O C } _ { \mathrm { c l s } }$ of 92.21%. Notably, IECG outperforms TolerantECG in FMax by a clear margin (42.43% vs 40.76%), highlighting its superior precision-recall balance and robustness under class imbalance. Compared to other foundation models, including METS, MERL, and ECG-FM, both variants of IECG consistently demonstrate substantial improvements.

Table 2. Classification result evaluated on PTB-XL dataset. Bold indicates the best result, and underline indicates the second-best result.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>FMax</td><td rowspan=1 colspan=2> $\mathbf { A U R O C } _ { \mathbf { c l s } }$ </td></tr><tr><td rowspan=1 colspan=1>ST-MEM [23]</td><td rowspan=1 colspan=1> $1 1 . 0 5 \pm 0 . 0 3$ </td><td rowspan=1 colspan=2> $6 4 . 3 9 \pm \mathrm { 0 . 0 4 }$ </td></tr><tr><td rowspan=1 colspan=1>METS [14]</td><td rowspan=1 colspan=1> $3 9 . 8 8 \ \pm \mathrm { 0 . 2 3 }$ </td><td rowspan=1 colspan=2> $9 0 . 8 2 \pm 0 . 1 7$ </td></tr><tr><td rowspan=1 colspan=1>MERL [17]</td><td rowspan=1 colspan=1> $4 0 . 0 0 \pm \mathrm { 0 . 2 7 }$ </td><td rowspan=1 colspan=2> $9 1 . 6 6 \pm 0 . 1 7$ </td></tr><tr><td rowspan=1 colspan=1>ECG-FM [21]</td><td rowspan=1 colspan=1> $3 1 . 0 0 \pm \phantom { 0 } 0 . 3 3$ </td><td rowspan=1 colspan=2> $8 5 . 1 0 \pm \phantom { } 0 . 3 2$ </td></tr><tr><td rowspan=2 colspan=1>HeartLang [10]TolerantECG [24]</td><td rowspan=1 colspan=1> $2 6 . 0 3 \pm \mathrm { 0 . 4 7 }$ </td><td rowspan=1 colspan=2>80.10</td></tr><tr><td rowspan=1 colspan=1> $4 0 . 7 6 \pm \phantom { 0 } 0 . 3 8$ </td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>IECG-Last $\mathbb { I E C G - R o l l o u t }$ </td><td rowspan=1 colspan=1> $\mathbf { 4 2 . 4 3 \_ } _ { \pm 0 . 1 4 }$ </td><td rowspan=2 colspan=2> $\mathbf { 9 2 . 2 1 \_ } \mathbf { 0 . 1 4 }$  $9 2 . 0 4 \pm 0 . 2 8 $ </td></tr><tr><td rowspan=1 colspan=1> $4 1 . 6 2 \pm 0 . 3 8 $ </td></tr></table>

These results confirm that incorporating attention supervision through CGAL does not compromise classification performance. On the contrary, it enables the model to learn more informative representations that translate into state-ofthe-art results for downstream classification tasks.

Table 3. Effect of captioning loss on localization and classification.
<table><tr><td>Configuration</td><td> $\mathbf { A U R O C _ { l o c } }$  </td><td> $\mathbf { A U R O C _ { c l s } ^ { E C G A D } }$ </td><td>—  $\mathbf { A U R O C } _ { \mathrm { c l s } } ^ { \mathbf { P T B - X L } }$ </td></tr><tr><td> $\beta = 1 \mathrm { ( D e f a u l t ) }$ </td><td> $7 5 . 5 9 \pm 1 . 0 0$ </td><td> $\underline { { 8 9 . 0 9 \pm 0 . 4 9 } }$ </td><td> $\mathbf { 9 } 2 . \mathbf { 0 4 } _ { \pm 0 . 2 9 }$ </td></tr><tr><td> $\beta = 2$ </td><td> $\underline { { 7 5 . 2 8 \pm 0 . 9 5 } }$ </td><td> $\mathbf { 8 9 . 3 0 \ } _ { \pm 0 . 2 6 }$ </td><td> $9 1 . 3 6 \pm \mathrm { 0 . 1 3 }$ </td></tr></table>

Since IECG-Rollout achieves the best localization performance, we use this variant for all subsequent experiments and analyses.

## 5.2. Qualitative Analysis of Abnormality Localization

Figure 4 presents qualitative comparisons of abnormal region localization across multiple ECG conditions. For each example, we visualize the ground-truth abnormalities alongside the attention heatmaps generated by $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ and IECG. $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ often produces sparse and fragmented activations, with limited alignment to the annotated abnormal segments. In contrast, IECG generates more consistent and localized attention patterns that better overlap with the ground-truth regions.

Notably, IECG demonstrates improved robustness across diverse conditions, including bundle branch blocks, myocardial infarction, and non-specific ischemic changes. The attention maps produced by IECG are smoother and more temporally coherent, suggesting that the proposed CGAL mechanism effectively guides the model to focus on clinically relevant waveform regions.

## 6. Ablation Studies

We conduct ablation experiments to evaluate the contribution of each component in IECG. To comprehensively assess both interpretability and predictive performance, we benchmark both abnormality localization and classification on $\mathrm { E C G A D _ { d a t a } }$ dataset and additional classification on PTB-XL. This setup enables us to analyze how different design choices affect localization quality and classification accuracy under realistic clinical evaluation settings.

Captioning. Table 3 evaluates the effect of the captioning loss weight β on localization and classification. The default setting, $\beta = 1$ , achieves the best localization performance and the highest PTB-XL classification performance, while remaining competitive on ECGAD classification. Increasing $\beta$ to 2 slightly improves ECGAD classification but leads to a small decline in both localization and PTB-XL classification performance. These results suggest that stronger captioning supervision does not consistently improve diagnostic performance and that $\beta = 1$ provides a favorable balance between localization and classification.

Table 4. Effect of attention supervision on localization and classification.
<table><tr><td>Configuration</td><td> $\mathbf { A U R O C } _ { \mathrm { l o c } }$ </td><td> $\mathbf { A U R O C } _ { \mathrm { c l s } } ^ { \mathbf { E C G A D } }$ </td><td> $\mathbf { A U R O C _ { c l s } ^ { P T B - X L } }$ </td></tr><tr><td> $\gamma = 0$ </td><td> $6 1 . 3 3 \pm 1 . 2 7$ </td><td> $\mathbf { 8 9 . 4 1 \_ } \mathbf { _ { 0 . 2 3 } }$ </td><td> $9 2 . 1 4 \pm \mathrm { 0 . 0 9 }$ </td></tr><tr><td> $\gamma = 0 . 5 \ : ( \mathrm { D e f a u l t } )$ </td><td> $7 5 . 5 5 \pm 1 . 0 0$ </td><td> $8 9 . 0 9 \pm \mathrm { 0 . 4 9 }$ </td><td> $9 2 . 0 3 \pm \mathrm { 0 . 2 3 }$ </td></tr><tr><td> $\gamma = 1 . 0$ </td><td> ${ 7 5 . 3 1 \pm 0 . 9 7 }$ </td><td> $\underline { { 8 9 . 3 8 \pm 0 . 4 0 } }$ </td><td> $\mathbf { 9 2 . 6 1 \_ } \mathbf { _ { 0 . 1 2 } }$ </td></tr></table>

Attention Supervision. Table 4 shows the effect of attention supervision controlled by $\gamma .$ Removing attention guidance $( \gamma = 0 )$ maintains strong classification performance but leads to a sharp drop in localization, with $\mathrm { { A U R O C } _ { \mathrm { { l o c } } } }$ decreasing to $6 1 . 3 3 \%$ Introducing strong attention supervision $( \gamma = 1 . 0 )$ substantially improves localization and yields the highest classification performance on PTB-XL (92.61%). Our default setting $( \gamma = 0 . 5 )$ provides the most effective balance by achieving the highest localization score (75.55%) while keeping competitive classification accuracy across both datasets.

## 7. Conclusion

We propose IECG, an ECG foundation model that achieves strong diagnostic performance while enhancing clinically meaningful interpretability through abnormality localization, all without requiring manual attention annotations. IECG is trained using a weakly-supervised mechanism that derives abnormality heatmap pseudo-labels from clinical heuristics. This approach guides the model’s attention toward diagnostically relevant regions, effectively mitigating the localized annotation bottleneck in ECG data. By aggregating information from these clinically guided regions, IECG improves the grounding of its predictions in relevant waveform features. Experiments demonstrate that IECG delivers state-of-the-art performance on abnormality localization while maintaining robust classification capabilities.

## Limitations

First, CGAL’s pseudo-label generation relies on predefined clinical rules and the CFR database. It may miss rare conditions or abnormalities tied to patient-specific factors (e.g., age, gender), though incorporating free-form reports partially mitigates this.

Second, while IECG enhances interpretability, it is not inherently fully interpretable. Its single global attention map highlights the union of all abnormal regions, preventing it from attributing specific waveform segments to individual diagnoses in multi-diagnosis ECGs.

Future work will incorporate broader clinical guidelines to improve pseudo-label specificity and explore explicit mappings between localized regions and individual diagnoses to enable fine-grained, diagnosis-specific localization.

## References

[1] Life in the fast lane, 2008. 4, 2

[2] Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4190–4197, Online, 2020. Association for Computational Linguistics. 5

[3] ACLS Certification Association. First degree av block criteria, 2025. 3

[4] Nhat-Tan Bui, Dinh-Hieu Hoang, Thinh Phan, Minh-Triet Tran, Brijesh Patel, Donald Adjeroh, and Ngan Le. Tsrnet: Simple framework for real-time ecg anomaly detection with multimodal time and spectrogram restoration network, 2024. 1

[5] Brian Gow, Tom Pollard, Larry A. Nathanson, Alistair Johnson, Benjamin Moody, Chrystinne Fernandes, Nathaniel Greenbaum, Jonathan W. Waks, Parastou Eslami, Tanner Carbonati, Ashish Chaudhari, Elizabeth Herbst, Dana Moukheiber, Seth Berkowitz, Roger Mark, and Steven Horng. Mimic-iv-ecg: Diagnostic electrocardiogram matched subset (version 1.0), 2023. 1, 5, 6, 2

[6] Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, and Jian Wu. Unet 3+: A full-scale connected unet for medical image segmentation, 2020. 4, 1

[7] Aofan Jiang, Chaoqin Huang, Qing Cao, Shuang Wu, Zi Zeng, Kang Chen, Ya Zhang, and Yanfeng Wang. Multi-scale cross-restoration framework for electrocardiogram anomaly detection, 2023. 5, 6, 7

[8] Aofan Jiang, Chaoqin Huang, Qing Cao, Shuang Wu, Zi Zeng, Kang Chen, Ya Zhang, and Yanfeng Wang. Multi-scale cross-restoration framework for electrocardiogram anomaly detection, 2023. 1

[9] Aofan Jiang, Chaoqin Huang, Qing Cao, Yuchen Xu, Zi Zeng, Kang Chen, Ya Zhang, and Yanfeng Wang. Selfsupervised anomaly detection pretraining enhances long-tail ecg diagnosis, 2024. 6

[10] Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, and Shenda Hong. Reading your heart: Learning ecg words and sentences via pre-training ecg language model, 2025. 7, 6

[11] Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, and Shenda Hong. Reading your heart: Learning ecg words and sentences via pre-training ecg language model, 2025. 2, 3, 6, 7

[12] Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana Yeganova, W John Wilbur, and Zhiyong Lu. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. Bioinformatics, 39(11):btad651, 2023. 5

[13] Chankyu Joung, Mijin Kim, Taejin Paik, Seong-Ho Kong, Seung-Young Oh, Won Kyeong Jeon, Jae-Hu Jeon, Joong-Sik Hong, Wan-Joong Kim, Woong Kook, Myung-Jin Cha, and Otto van Koert. Deep learning based ecg segmentation for delineation of diverse arrhythmias. PLOS ONE, 19(6): e0303178, 2024. 4, 1

[14] Jun Li, Che Liu, Sibo Cheng, Rossella Arcucci, and Shenda

Hong. Frozen language model helps ecg zero-shot learning, 2023. 2, 6, 7

[15] Life in the Fast Lane (LITFL). First degree heart block, 2024. 3

[16] Life in the Fast Lane (LITFL). Left bundle branch block (lbbb), 2024. 3

[17] Che Liu, Zhongwei Wan, Cheng Ouyang, Anand Shah, Wenjia Bai, and Rossella Arcucci. Zero-shot ecg classification with multimodal learning and test-time clinical knowledge enhancement, 2024. 2, 6, 7

[18] Feifei Liu, Chengyu Liu, Lina Zhao, Xiangyu Zhang, Xiaoling Wu, Xiaoyan Xu, Yulin Liu, Caiyun Ma, Shoushu Wei, Zhiqiang He, Jianqing Li, and Eddie Ng. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. Journal of Medical Imaging and Health Informatics, 8:1368– 1373, 2018. 6

[19] Yuhang Liu, Peng Zhang, and Fan Lin. Resting ECG Seg mentation Dataset. 2025. 4

[20] Yu Lu, Jiali Zeng, Jiajun Zhang, Shuangzhi Wu, and Mu Li. Attention calibration for transformer in neural machine translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1288–1298, Online, 2021. Association for Computational Linguistics. 2

[21] Kaden McKeen, Sameer Masood, Augustin Toma, Barry Ru bin, and Bo Wang. Ecg-fm: An open electrocardiogram foundation model, 2025. 1, 2, 6, 7

[22] Temesgen Mehari and Nils Strodthoff. Self-supervised rep resentation learning from 12-lead ecg data. Computers in Biology and Medicine, 141:105114, 2022. 2

[23] Yeongyeon Na, Minje Park, Yunwon Tae, and Sunghoon Joo. Guiding masked representation learning to capture spatiotemporal relationship of electrocardiogram, 2024. 1, 2, 6, 7

[24] Huynh Dang Nguyen, Trong-Thang Pham, Ngan Le, and Van Nguyen. Tolerantecg: A foundation model for imperfect electrocardiogram. In Proceedings of the 33rd ACM International Conference on Multimedia, page 8097–8105, New York, NY, USA, 2025. Association for Computing Machinery. 1, 2, 3, 4, 6, 7

[25] Panteleimon Pantelidis, Samuel Ruiperez-Campillo, Julia E.´ Vogt, Alexios Antonopoulos, Ioannis Gialamas, George E. Zakynthinos, Michael Spartalis, Polychronis Dilaveris, Jose Millet, Theodore G. Papaioannou, Evangelos Oikonomou, and Gerasimos Siasos. Ecg-xplaim: explainable locally adaptive artificial intelligence model for arrhythmia detection from large-scale electrocardiogram data. Frontiers in Cardiovascular Medicine, Volume 12 - 2025, 2025. 2, 6, 7

[26] Minje Park, Jeonghwa Lim, Taehyung Yu, and Sunghoon Joo. Semisegecg: A multi-dataset benchmark for semisupervised semantic segmentation in ecg delineation, 2025. 4, 1

[27] Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models, 2018. 6

[28] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Journal of Computer Vision, 128(2):336–359, 2019. 2

[29] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps, 2014. 2

[30] Junho Song, Jong-Hwan Jang, DongGyun Hong, Joon myoung Kwon, and Yong-Yeon Jo. Crema: A contrastive regularized masked autoencoder for robust ecg diagnostics across clinical domains, 2025. 2

[31] Nils Strodthoff, Patrick Wagner, Tobias Schaeffter, and Wojciech Samek. Deep learning for ecg analysis: Benchmarks and insights from ptb-xl, 2020. 6

[32] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Wojciech Samek, and Tobias Schaeffter. PTB-XL, a large publicly available electrocardiography dataset. PhysioNet, 2022. Version 1.0.3. 1, 5

[33] Patrick Wagner, Temesgen Mehari, Wilhelm Haverkamp, and Nils Strodthoff. Explaining deep learning for ecg analysis: Building blocks for auditing and knowledge discovery. Computers in Biology and Medicine, 176:108525, 2024. 2, 6, 7

[34] Yue Wang, Xu Cao, Yaojun Hu, Haochao Ying, Hongxia Xu, Ruijia Wu, James Matthew Rehg, Jimeng Sun, Jian Wu, and Jintai Chen. Anyecg: Foundational models for multitask cardiac analysis in real-world settings, 2025. 2

[35] Han Yu, Peikun Guo, and Akane Sano. Ecg semantic integrator (esi): A foundation ecg model pretrained with llmenhanced cardiological text, 2024. 3, 5

[36] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models, 2022. 3

[37] Yubao Zhao, Jiaju Kang, Tian Zhang, Puyu Han, and Tong Chen. Ecg-chat: A large ecg-language model for cardiac disease diagnosis, 2025. 1, 3, 5

[38] Jianwei Zheng, Hangyuan Guo, and Huimin Chu. A large scale 12-lead electrocardiogram database for arrhythmia study. PhysioNet, 2022. Version 1.0.0. 6

[39] Peilin Zhou, Qichen Ye, Yueqi Xie, Jingqi Gao, Shoujin Wang, Jae Boum Kim, Chenyu You, and Sunghun Kim. Attention calibration for transformer-based sequential recommendation, 2024. 2

## A. CGAL Pseudo-Label Generation algorithm

Algorithm 1 outlines the procedure for curating pseudo-heatmap labels that allocate diagnostic regions. The design ensures that pseudo-labels are both clinically grounded and diagnosis-specific. Before identifying abnormal regions, Stage 1 employs a pretrained UNet3+ [6, 13] to segment the ECG signal into P wave, QRS complex, and T wave components. Stage 2 identifies candidate abnormal regions based on waveform segmentation and interval analysis, leveraging AddPseudoPwave (Algorithm 2) to heuristically insert missing P-wave segments and ComputeIntervals (Algorithm 3) to derive clinically relevant intervals including PR interval, QT interval, and ST segmenet. Stage 3 refines these candidate regions by intersecting them with diagnostic criteria retrieved from CFR, while also applying fallback rules (discussed in Section D) to enhance robustness against database incompleteness. As a result, the generated binary attention masks provide reliable supervision signals that guide the encoder to focus on clinically meaningful regions, thereby improving anomaly localization and interpretability in downstream tasks.

Algorithm 1 CGAL Pseudo-Label Generation   
Require: ECG signal X, Diagnosis list D   
Ensure: Binary attention mask m ∈ {0, 1}<sup>N</sup>   
1: // Stage 1: Wave Segmenting   
2: S ← UNet3+(X) {segment into P, QRS, T}   
3: // Stage 2: Abnormal Seeking   
4: (S, R) ← AddPseudoPwave(S)   
5: (PR, QT, ST) ← ComputeIntervals(S)   
6: QTc ← CorrectQT(QT, S)   
7: R ← R ∩ FlagAbnormal(S, PR, QT, QTc, ST)   
8: // Stage 3: Criteria Filtering   
9: m ← ZerosLike(X)   
10: for d in D do   
11: criteria ← CFR(d)   
12: R ← R ∩ criteria {keep diagnosis-relevant}   
13: m ← m ∨ CreateMask(R, N)   
14: end for   
15: return m

## B. UNet3+ performance

Accurate segmentation is a critical prerequisite for reliable ECG analysis, as it enables precise identification of individual wave components within the input signal. Inadequate performance at this stage can propagate errors into subsequent modules, leading to incorrect pseudo-heatmap label curation and causing the encoder to attend to partially erroneous regions. Table S1 summarizes the comparative performance of three segmentation approaches, namely Pro-

Algorithm 2 AddPseudoPwave   
Require: Segmentation mask S   
Ensure: Modified segmentation S, abnormal highlight set   
R   
1: (QRS , QRS ) ← GetOnOffset(S = 2)   
2: (T<sub>on</sub>, T<sub>off</sub>) ← GetOnOffset(S = 3)   
3: for i ∈ {1, . . . , len(QRS )} do   
4: start ← T [i − 1]   
5: end ← QRS [i]   
6: if 1 ∈/ S[start : end] then   
7: area ← end − start   
8: startP ← start + area   
9: endP ← end − area   
10: S[startP : endP] ← 1 {Mark pseudo-P wave re  
gion}   
11: R ← R ∪ Flag(startP, endP)   
12: end if   
13: end for   
14: return S, R

Algorithm 3 ComputeIntervals   
Require: Segmentation mask S   
Ensure: PR interval, QT interval and ST segment   
1: // Step 1: Detect onset/offset indices   
2: (P , P ) ← GetOnOffset(S = 1)   
3: (QRS , QRS ) ← GetOnOffset(S = 2)   
4: (T , T ) ← GetOnOffset(S = 3)   
5: // Step 2: Initialize empty intervals   
6: PR ← zeros(len(S))   
7: QT ← zeros(len(S))   
8: ST ← zeros(len(S))   
9: // Step 3: Construct intervals   
10: for each (start, end) in (P , QRS ) do   
11: PR[start : end] ← 1   
12: end for   
13: for each (start, end) in (QRS , T<sub>off</sub>) do   
14: QT[start : end] ← 1   
15: end for   
16: for each (start, end) in (QRS , T<sub>on</sub>) do   
17: ST[start : end] ← 1   
18: end for   
19: return PR, QT, ST

Mac, NeuroKit, and UNet3+, in detecting P, QRS, and T wave boundaries. Evaluation was conducted on the Semi-Seg-ECG test set [26] using the F1-score metric [13], which provides a balanced measure of precision and recall for boundary localization.

Across all wave boundaries, UNet3+ demonstrates substantially higher accuracy, achieving near-perfect F1-scores for QRS onset and offset (99.88% and 99.89%, respectively). In contrast, ProMac and NeuroKit exhibit moderate performance, particularly struggling with T-wave offset detection (67.62% and 63.97%). These results highlight the importance of robust segmentation: errors in boundary detection disproportionately affect downstream anomaly localization, as misaligned pseudo-labels can misguide the encoder’s attention. The superior performance of UNet3+ underscores its suitability as the backbone segmentation model in our pipeline, ensuring reliable wave delineation and minimizing error propagation in subsequent stages.

Table S1. Performance comparison of segmentation methods on ECG component boundary localization, evaluated using F1-score on the Semi-Seg-ECG test set.
<table><tr><td>Method</td><td>P onset</td><td>P offset</td><td>QRS onset</td><td>QRS offset</td><td>T onset</td><td>T offset</td></tr><tr><td>ProMac</td><td>84.72</td><td>84.91</td><td>77.05</td><td>78.08</td><td>81.70</td><td>67.62</td></tr><tr><td>NeuroKit</td><td>84.78</td><td>84.60</td><td>79.62</td><td>80.92</td><td>79.52</td><td>63.97</td></tr><tr><td>UNet3+</td><td>96.68</td><td>96.75</td><td>99.88</td><td>99.89</td><td>97.82</td><td>96.66</td></tr></table>

Table S2. F1 score performance of the UNet3+ across multiple datasets in the SemiSegECG test set.
<table><tr><td rowspan=1 colspan=4>Dataset</td><td rowspan=1 colspan=1>P onset</td><td rowspan=1 colspan=1>P offset</td><td rowspan=1 colspan=1>QRS onset</td><td rowspan=1 colspan=1>QRS offset</td><td rowspan=1 colspan=1>T onset</td><td rowspan=1 colspan=1>T offset</td></tr><tr><td rowspan=1 colspan=4>ISP</td><td rowspan=1 colspan=1>97.09</td><td rowspan=1 colspan=1>97.16</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>99.50</td><td rowspan=2 colspan=1>99.4895.73</td></tr><tr><td rowspan=3 colspan=4>ZhejiangLUDBQTDB</td><td rowspan=1 colspan=1>Zhej</td><td rowspan=1 colspan=1>ejiang</td><td rowspan=1 colspan=1>ng</td><td rowspan=1 colspan=1>94.56</td><td rowspan=1 colspan=1>94.60</td><td rowspan=1 colspan=1>99.84</td></tr><tr><td rowspan=1 colspan=1>96.49</td><td rowspan=1 colspan=1>96.57</td><td rowspan=1 colspan=1>99.86</td><td rowspan=1 colspan=1>99.86</td><td rowspan=1 colspan=1>99.60</td><td rowspan=2 colspan=1>99.2789.14</td></tr><tr><td rowspan=1 colspan=1>97.74</td><td rowspan=1 colspan=1>99.43</td><td rowspan=1 colspan=1>99.43</td><td rowspan=1 colspan=1>95.52</td></tr></table>

For a more detailed breakdown, table S2 presents the F1 score performance of the segmentation model across multiple datasets within the SemiSegECG test set. The model achieves consistently high accuracy across all waveform components, with near perfect performance on QRS onset and offset detection across all datasets. P wave segmentation shows slightly lower but still strong results, particularly on the Zhejiang dataset. T wave segmentation exhibits greater variability, with noticeable performance drops on QTDB, especially for T wave offset. Overall, these results demonstrate the robustness of the segmentation model across diverse datasets, while highlighting the increased difficulty of accurately detecting subtle waveform boundaries.

## C. Abnormal criteria in ECG signal

In the second stage of the disease heatmap curation pipeline, we identify and highlight abnormal regions across individual waves and leads. To achieve this, we collect diagnostic conditions from LITFL [1] and subsequently verify them with clinical experts to ensure medical validity. These curated conditions serve as the foundation for constructing reliable heatmaps that emphasize clinically relevant abnormalities. A summary of the identified conditions is presented in Table S3.

Table S3. Abnormal criteria for each ECG waveform component. These thresholds are used in CGAL Stage 2 to flag candidate abnormal regions.
<table><tr><td>Component</td><td>Abnormal Conditions</td></tr><tr><td>P wave</td><td>Missing (absent P wave) Amplitude &gt; 0.25 mV in limb leads Amplitude &gt; 0.15 mV in precordial leads Duration &gt; 120 ms Inverted in leads I, II, III; Upright in lead aVR</td></tr><tr><td>QRS complex</td><td>Duration &gt; 120 ms Abnormal morphology (RSR&#x27;, QR, rS patterns)</td></tr><tr><td>T wave</td><td>Amplitude &gt; 0.5 mV in limb leads Amplitude &gt; 1.0 mV in precordial leads Inverted in normally upright leads (I, II, V2–V6)</td></tr><tr><td>PR interval</td><td>Duration &gt; 200 ms (prolonged) Duration &lt; 120 ms (shortened)</td></tr><tr><td>QT interval</td><td>Corrected duration &gt; 440 ms (prolonged) Corrected duration &lt; 350 ms (shortened)</td></tr><tr><td>ST segment</td><td>Elevation &gt; 0.1 mV (&gt; 0.05 mV in V2, V3) Depression &lt; —0.5 mV from isoelectric line</td></tr></table>

Table S4. Examples of lowest-scoring queries in CFR retrieval.
<table><tr><td>Query</td><td>Retrieved diagnosis</td><td>Retrieve score</td></tr><tr><td>Nonspecific ST changes Marked ST depression in V lead Short PR interval ST depression V1-V3 Prolonged PR interval ST-T changes rSr&#x27; pattern in V1 or V2 Inferior nonspecific ST-T changes Abnormally tall P in V1 or V2</td><td>Septal hypertrophy Pericarditis Extreme Axis Deviation Poor R Wave Progression R Wave Peak Time Digitalis Effect Low QRS voltage Septal hypertrophy Raised Intracranial Pressure Benign Early Repolarisation</td><td>-14.26 -10.04 -9.29 -8.83 -8.04 -7.53 -4.20 -3.83 -3.31</td></tr></table>

## D. CFR quality and fallback

The CFR component is designed to retrieve clinically relevant diagnostic criteria. To evaluate its performance, we collect all diagnostic classes from the MIMIC-IV-ECG dataset [5] and use them as query inputs to measure the acceptance rate. The results show an acceptance rate of 20.18%, with an average similarity score of 84.95 among the accepted queries. This indicates that, while the retrieved criteria are generally of high quality when matched, the overall retrieval coverage remains limited.

Analysis of the retrieval errors reveals that many query reports already describe clinical criteria rather than diagnostic labels, and therefore do not directly correspond to entries in the CFR database. Consequently, such inputs cannot be effectively matched to diagnosis–criteria pairs. Table S4 provides examples of low scoring queries, where clinically valid descriptions such as “nonspecific ST changes” or “prolonged PR interval” are not retrieved.

To mitigate these limitations, we adopt a fallback strategy: when diagnostic features cannot be retrieved, the diagnosis name itself is used to guide abnormal heatmap filtering. Specifically, if a query explicitly mentions a wave component (e.g., “PR interval,” “T-waves”), we retain the corresponding highlights from Stage 2 of CGAL. For broader diagnostic terms such as “ischaemia,” “ischemic,” or “infarction,” we emphasize the ST segment and T wave, while terms like “atrial” direct attention to the P wave. In addition, ECG lead group phrases (e.g., anterior, septal, inferior) are leveraged to ensure that abnormal heatmap labels highlight the correct anatomical regions. This strategy helps reduces false-negative annotations and produces more accurate pseudo-labels for attention supervision.

## E. Ablation on Spatial-Temporal Embedding

To evaluate the contribution of the Spatial-Temporal (ST) embedding (detailed in Section 3.1 of the main text), we compare our default model against a variant trained without the ST embedding components $( \mathbf { E } ^ { s }$ and E<sup>t</sup>). The ST embedding provides explicit lead identity and time frame information, enriching each token with spatial and temporal context.

Table S5 summarizes the localization and classification performance of both configurations.

Table S5. Effect of Spatial-Temporal embedding on localization and classification.
<table><tr><td>Configuration 1</td><td> $\mathbf { A U R O C _ { l o c } }$ </td><td> $\mathbf { A U R O C } _ { \mathrm { c l s } } ^ { \mathbf { E C G A D } }$ </td><td>AUROCPTB-XL cls</td></tr><tr><td>w/o ST embedding</td><td> $7 5 . 0 1 \pm 0 . 8 4$ </td><td>90.31 ±0.24</td><td>92.08 ±0.30</td></tr><tr><td>w/ ST embedding (Default)</td><td> $7 5 . 5 5 \pm 1 . 0 0$ </td><td>89.09 ±0.49</td><td> $9 2 . 0 3 \pm \mathrm { 0 . 2 3 }$ </td></tr></table>

Removing the ST embedding results in slightly higher classification performance, achieving an $\mathrm { A U R O C } _ { \mathrm { c l s } } ^ { \mathbf { \bar { E } C G A D } }$ of 90.31% and an AUROC<sup>PTB-XL</sup> of 92.08%. However, this omission leads to reduced localization accuracy (75.01%). In contrast, incorporating the ST embedding improves localization performance to 75.55%, while still maintaining competitive classification results. This suggests that the ST embedding provides useful structural information that enhances the model’s ability to capture spatial and temporal dependencies across leads. Such explicit spatial-temporal grounding proves particularly beneficial for precisely identifying localized abnormalities, confirming its value in optimizing the model for clinical interpretability.

## F. Qualitative comparison

To further assess the quality of abnormal region localization, we provide visual comparisons across multiple ECG samples with diverse diagnostic conditions, as shown in Figures S1 to S4. These examples illustrate how well the predicted attention maps align with expert annotations across both temporal and lead dimensions. Each example shows a 12-lead ECG signal with corresponding localization results. For each lead, the waveform is shown together with ground truth annotations (blue), baseline predictions from $\mathrm { E C G A D } _ { \mathrm { m o d e l } }$ (red), and predictions from IECG (green). Rather than relying solely on aggregate metrics, this analysis offers a more detailed view of model behavior, highlighting strengths and failure modes in complex and multi-label scenarios.

## G. Limitations of Ground-Truth Abnormal Annotations

Although IECG achieves strong localization performance, the overall $\mathrm { { \bf A U R O C } _ { \mathrm { l o c } } }$ could be constrained by inaccuracies and incompleteness in the ground-truth annotations of $\mathrm { E C G A D _ { d a t a } }$ . In particular, the provided abnormal region labels do not always capture all clinically relevant waveform features, especially for conditions characterized by temporal abnormalities rather than distinct morphological changes.

A representative example is shown in Figure S5, where the ECG exhibits complete left bundle branch block (LBBB) and first-degree atrioventricular (AV) block. The annotations correctly highlight the prolonged QRS complexes associated with LBBB, which is defined by a widened QRS duration (≥ 120 ms) reflecting delayed ventricular depolarization [16]. However, they fail to capture the prolonged PR interval (> 200 ms) that characterizes first-degree AV block, which corresponds to delayed conduction between atrial and ventricular activation [3, 15].

Clinically, these abnormalities manifest in distinct regions of the waveform: LBBB primarily affects the QRS complex, while first-degree AV block is defined by prolongation of the PR interval preceding the QRS. As a result, the ground-truth heatmaps, which focus on QRS-based abnormalities, omit diagnostically important temporal features in the PR segment.

In contrast, IECG highlights both the widened QRS complexes and the prolonged PR intervals, producing attention maps that more closely reflect the full set of clinically relevant abnormalities. Such discrepancies indicate that the ground-truth annotations do not fully capture all relevant diagnostic evidence, causing the model to be penalized for correctly identifying waveform regions that are not labeled. This limitation partially explains the gap between the strong qualitative alignment of IECG and its quantitative evaluation scores.

## H. Evaluating the Effectiveness of CGAL Pseudo-labels

To further understand the contribution of the CGAL pipeline, we isolate its performance by evaluating the generated pseudo-labels directly against the expert ground-truth annotations in the ECGAD dataset. In this experiment, referred to as Just CGAL, we construct abnormal heatmaps exclusively using the rule-based clinical heuristics and CFR module (as detailed in Section 3.3 of the main text), entirely bypassing the Transformer model’s attention rollout.

![](images/013841db7f2629a4e551daf38a5bb4790ede05ea8c7190df04f52746b914d8f6.jpg)  
Figure S1. First-degree atrioventricular block with sinus rhythm. IECG consistently highlights prolonged PR intervals across multiple leads, showing stronger alignment with ground truth compared to the baseline, which exhibits more dispersed and less structured localization.

anterolateral myocardial infarction, first degree AV block, non-specific intraventricular conduction disturbance (block), prolonged PR interval, sinus rhythm  
![](images/1bfc71712730741d9a5e65728132c1e890a6978e1259dd528f49ab5d33a4ae48.jpg)  
Figure S2. Anterolateral myocardial infarction with first-degree AV block and intraventricular conduction disturbance. As expected clinically, IECG assigns substantially higher attention weight to the anterolateral leads (I, aVL, and V4–V6) to capture the myocardia infarction, while simultaneously maintaining strong focus on the prolonged PR intervals across the waveform to identify the AV block. This demonstrates the model’s ability to accurately localize both spatially-restricted morphological abnormalities and global tempora conduction delays within the same sample, whereas the baseline exhibits fragmented and less anatomically specific activations.

We compare this heuristic-only baseline against the learned representations of our model, both with and without attention supervision. Table S6 presents the localization performance $( \mathrm { A U R O C _ { \mathrm { l o c } } ) }$ for each configuration.

Table S6. Comparison of localization performance between the heuristic-based CGAL pseudo-labels and the trained models.
<table><tr><td>Method</td><td> $\mathbf { A U R O C _ { l o c } }$ </td></tr><tr><td>IECG w/o Attention Supervision  $( \gamma = 0 )$ </td><td>61.33%</td></tr><tr><td>Just CGAL (Rule-based Pseudo-labels)</td><td>70.13%</td></tr><tr><td>IECG  $( \gamma = 1 . 0 )$ </td><td>75.31%</td></tr><tr><td>IECG (Default, γ = 0.5)</td><td>75.55%</td></tr></table>

The results yield two key observations. First, the heuristic-derived heatmaps from Just CGAL achieve an $\mathrm { { A U R O C } _ { \mathrm { { l o c } } } }$ of 70.13%, significantly outperforming the baseline IECG trained without attention supervision (61.33%). This validates the quality and clinical relevance of the generated pseudo-labels. Second, and most importantly, when these pseudo-labels are used to supervise the model during training, the resulting models consistently surpass the performance of their own pseudo-label “teacher,” achieving 75.31% $( \gamma = 1 . 0 )$ and 75.55% $( \gamma = 0 . 5 )$ . This demonstrates that the weakly-supervised learning process allows the model to generalize beyond the rigid, rule-based heuristics of CGAL, learning a more robust and comprehensive understanding of morphological abnormalities from largescale data regardless of the exact supervision weight.

![](images/91d1ed8876d5efb672b9fd40c9601f493d7017c007b1bfd41139a36d9b35fd8d.jpg)  
Figure S3. Inferior myocardial infarction with first-degree AV block. To accurately diagnose the inferior infarction, IECG distinctly concentrates its attention on the inferior leads (II), precisely matching the expected anatomical region of the ischemic injury. Concurrently it correctly highlights the prolonged PR intervals to account for the atrioventricular conduction delay. In contrast, the baseline model fails to prioritize the inferior leads and produces activations that lack clear anatomical or temporal justification.

![](images/7df78fc4538fafee3d1717275ab37d82a12b76921ce157ed18099f72a19ca3c0.jpg)  
Figure S4. Complete right bundle branch block (RBBB) with left anterior fascicular block (LAFB). IECG accurately targets the broadened QRS complexes characteristic of RBBB, placing prominent attention on the right precordial leads (V1–V3). Furthermore, to identify LAFB, the model correctly distributes its focus toward the inferior leads (II, III, aVF) and high lateral leads (I, aVL), which are critical for recognizing the associated left axis deviation. This precise, anatomically-grounded localization contrasts with the baseline, which generates uncoordinated activations that fail to clearly reflect the underlying multi-fascicular conduction pathologies.

## I. Report Generation

To evaluate the capabilities of our learned ECG representations on complex generative tasks, we assess IECG on ECG report generation. Following the implementation of ECG-Chat [37], we utilize a Large Language Model (LLM) initialized with Vicuna-13B (v1.5) weights and align our frozen ECG encoder to the LLM using a projection layer. We evaluate the generated reports using standard natural language generation metrics, including BLEU-4, ROUGE-L, METEOR, and BERT-score (F1).

To understand the impact of CGAL on generative performance, we compare our default IECG with a variant trained without attention supervision (w/o CGAL). Both models are evaluated on the PTB-XL test set, using the corresponding English-translated reports as references. The results are

summarized in Table S7.

Table S7. Report generation performance on the PTB-XL dataset.
<table><tr><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=1>BLEU-4</td><td rowspan=1 colspan=1>ROUGE-L</td><td rowspan=1 colspan=1>METEOR</td><td rowspan=1 colspan=1>BERT-score</td></tr><tr><td rowspan=2 colspan=1>w/o CGAL (γ = 0)IECG (Default)</td><td rowspan=1 colspan=1>12.57</td><td rowspan=1 colspan=1>34.49</td><td rowspan=1 colspan=1>33.61</td><td rowspan=1 colspan=1>89.25</td></tr><tr><td rowspan=1 colspan=1>12.14</td><td rowspan=1 colspan=1>34.28</td><td rowspan=1 colspan=1>32.56</td><td rowspan=1 colspan=1>89.20</td></tr></table>

As shown in Table S7, the model trained without CGAL achieves slightly higher scores across traditional n-gram based metrics such as BLEU-4 and METEOR. However, the performance of the default IECG remains highly competitive, demonstrating identical semantic similarity as measured by BERT-Score (89.20%) and only a marginal decrease in ROUGE-L (34.28 vs 34.49). These findings indicate that while explicitly constraining the model’s attention to clinically relevant regions via CGAL imposes a slight regularization effect on exact text matching, it preserves the rich semantic representations necessary for accurate and coherent clinical report generation.

complete left bundle branch block, first degree AV block, prolonged PR interval, sinus bradycardi  
![](images/4400dd9fb42054139ba93f6dcdf3687d96ea2625ed20b23492f2a7eb43c87a6e.jpg)  
Figure S5. Example illustrating annotation incompleteness in $\mathrm { E C G A D _ { d a t a } }$ using Lead II. The ECG exhibits complete left bundle branch block (LBBB) and first-degree atrioventricular (AV) block, as indicated by prolonged QRS duration (e.g., 172–176 ms) and prolonged PR interval (e.g., 204–224 ms), respectively. The ground-truth (GT) heatmap highlights regions corresponding to widened QRS complexes associated with LBBB but fails to capture the prolonged PR intervals that define first-degree AV block. In contrast, IECG accurately highlights both QRS and PR regions, reflecting clinically relevant abnormalities across both morphological (ventricular depolarization) and temporal (atrioventricular conduction delay) features. This example demonstrates that missing annotations of temporal abnormalities can lead to underestimation of localization performance.

## J. Classification on CPSC and Chapman Datasets

To further demonstrate the robustness and generalization ability of our learned representations, we evaluate IECG on two additional downstream classification benchmarks: CPSC 2018 [18] and Chapman [38]. For fair comparison, we adopt the same data splitting strategy for these datasets as established in MERL [17].

Table S8 presents the classification results. Consistent with our findings on $_ { \mathrm { P T B - X L } , }$ IECG achieves highly competitive performance across both datasets, outperforming all baselines except TolerantECG, which it trails by a small margin. Specifically, on the CPSC dataset, IECG achieves an AUROC of 96.32% and an FMax of 80.06%, while on the Chapman dataset, it achieves an AUROC of 95.32% and an FMax of 52.15%. These results confirm that incorporating attention supervision through CGAL for interpretability does not significantly compromise the model’s diagnostic classification capabilities, enabling it to learn informative features that generalize well to diverse unseen datasets.

Table S8. Classification results evaluated on the CPSC and Chapman datasets. Bold indicates the best result, and underline indicates the second-best result.
<table><tr><td rowspan=2 colspan=3>Model</td><td rowspan=1 colspan=2>CPSC</td><td rowspan=1 colspan=2>Chapman</td></tr><tr><td rowspan=1 colspan=1>FMax</td><td rowspan=1 colspan=1>AUROC</td><td rowspan=1 colspan=1>FMax</td><td rowspan=1 colspan=1>AUROC</td></tr><tr><td rowspan=7 colspan=3>ST-MEM [23]HeartLang [10]XAI4ECG [33]ECG-FM [21]ECGAD [7]MERL [17]METS [14]</td><td rowspan=1 colspan=1> $2 6 . 9 5 \pm \mathrm { 0 . 0 9 }$ </td><td rowspan=1 colspan=1> $6 3 . 0 3 \pm \mathrm { 0 . 1 1 }$ </td><td rowspan=1 colspan=1> $1 2 . 4 7 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1>62.22 ±0.80</td></tr><tr><td rowspan=1 colspan=1> $3 6 . 4 4 \pm 0 . 6 1$ </td><td rowspan=1 colspan=1> $7 0 . 1 3 \pm 0 . 4 5$ </td><td rowspan=1 colspan=1> $2 9 . 5 2 \pm \mathrm { 0 . 6 5 }$ </td><td rowspan=1 colspan=1>84.15 ±0.64</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1> $5 7 . 4 6 \pm  { 0 . 6 5 }$ 64.94 ±0.38</td><td rowspan=1 colspan=1>85.74 ±0.40</td><td rowspan=2 colspan=1>29.21 ±0.7844.03 ±0.71</td><td rowspan=2 colspan=1>82.10 ±0.5890.76 ±0.12</td></tr><tr><td rowspan=1 colspan=1>90.24 ±0.12</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.89 ±0.90</td><td rowspan=1 colspan=1>91.59 ±0.82</td><td rowspan=1 colspan=1>34.41 ±0.73</td><td rowspan=1 colspan=1>87.88 ±0.38</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1> $7 7 . 9 8 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1>95.51 ±0.06</td><td rowspan=1 colspan=1> $5 0 . 4 3 \pm \mathrm { 0 . 2 0 }$ </td><td rowspan=1 colspan=1>94.69 ±0.12</td></tr><tr><td rowspan=1 colspan=1> $7 9 . 1 6 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $9 6 . 1 2 \scriptstyle \pm 0 . 0 7$ </td><td rowspan=1 colspan=1> $4 9 . 6 7 \pm \mathrm { 0 . 2 1 }$ </td><td rowspan=1 colspan=1>95.08 ±0.05</td></tr><tr><td rowspan=1 colspan=3>TolerantECG [24]</td><td rowspan=1 colspan=1> $\mathbf { 8 1 . 5 3 \_ 6 } \mathbf { _ { \pm 0 . 4 0 } }$ </td><td rowspan=1 colspan=1> $\mathbf { 9 6 . 7 3 \_ } _ { \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 5 3 . 7 0 \Omega _ { \pm 0 . 2 3 } }$ </td><td rowspan=1 colspan=1>95.88 ±0.03</td></tr><tr><td rowspan=1 colspan=3>IECG (Ours)</td><td rowspan=1 colspan=1> $\underline { { 8 0 . 0 6 \pm 0 . 3 6 } }$ </td><td rowspan=1 colspan=1> $9 6 . 3 2 \pm 0 . 0 7$ </td><td rowspan=1 colspan=1> $5 2 . 1 5 \pm 0 . 3 6 $ </td><td rowspan=1 colspan=1> $9 5 . 3 2 \pm 0 . 0 5$ </td></tr></table>

## K. Trade-off Between Interpretability and Downstream Performance

A central challenge in developing medical AI systems is balancing raw predictive performance with clinical interpretability. Across the experiments detailed in the main text and this supplementary material, we consistently observe a highly favorable trade-off when incorporating CGAL.

Impact on Classification: As demonstrated in the main text’s ablation studies, training IECG without attention supervision $( \gamma = 0 )$ maximizes classification metrics, achieving an $\mathbf { A U R O C } _ { \mathrm { c l s } }$ of 92.42% on PTB-XL and 90.87% on ECGAD. However, this comes at the severe cost of interpretability, with localization performance dropping to 61.33%. By introducing CGAL with $\gamma = 0 . 5 ,$ , the model gains a massive +14.22% absolute improvement in localization $( \mathrm { A U R O C _ { \mathrm { l o c } } }$ of 75.55%). This structural regularization imposes only a marginal penalty on diagnostic power: $\mathrm { \mathbf { A U R O C } _ { c l s } }$ remains highly competitive at 92.21% on PTB-XL and 89.09% on ECGAD. Furthermore, the robust generalization of this balanced representation is confirmed on external benchmarks, where IECG maintains second-best overall performance on both the CPSC (96.32%) and Chapman (95.32%) datasets.

Impact on Report Generation: A similar regularizing effect is observed in complex generative tasks. As shown in the previous section on Report Generation, removing CGAL yields slightly higher exact text-matching scores (e.g., BLEU-4 of 12.57% vs. 12.14%). Nonetheless, the default IECG retains near-identical semantic coherence (BERT-score of 89.20% vs. 89.25%).

Architectural Contributions to the Trade-off: This favorable balance is fundamentally enabled by our specific architectural design choices, namely Attention Rollout and Spatial-Temporal (ST) embedding. As demonstrated in the main text’s experiments, applying attention supervision solely to the last attention layer yields a slightly higher classification $\mathbf { A U R O C } _ { \mathrm { c l s } }$ on ECGAD (90.03%) compared to

Attention Rollout (89.09%), but results in reduced localization accuracy (73.94% vs. 75.55%). Similarly, as detailed in Section E, removing the ST embedding marginally improves classification (90.31% on ECGAD) but degrades localization (75.01% vs. 75.55%). Both Attention Rollout and ST embedding introduce structural priors that regularize the network, intentionally trading a small fraction of raw predictive capacity to achieve more precise and clinically interpretable spatial-temporal grounding.

Conclusion: These results indicate that explicitly constraining the model’s attention to align with clinical heuristics inherently acts as an anatomical and temporal regularizer. While this regularization slightly bounds the unconstrained optimization of downstream objectives, the resulting trade-off is highly advantageous. IECG secures stateof-the-art or highly competitive predictive performance across diverse tasks while guaranteeing that its decisions are grounded in clinically valid waveform evidence, which is a critical prerequisite for clinical trust and real-world medical deployment.

## L. Ethical considerations

This work uses publicly available datasets (MIMIC-IV-ECG, PTB-XL, ECGAD, CPSC and Chapman) that have undergone institutional de-identification procedures. No personally identifiable information was accessed. IECG is intended to support clinical decision-making and is not a substitute for professional medical judgment. All predictions and recommendations must be reviewed and validated by qualified healthcare professionals before any clinical action is taken.