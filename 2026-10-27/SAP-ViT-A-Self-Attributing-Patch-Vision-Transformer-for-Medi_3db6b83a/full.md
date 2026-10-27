# SAP-ViT: A Self-Attributing Patch Vision Transformer for Medical Image Classification

WEI ZHANG, WOLFGANG NEJDL, MING TANG

L3S Research Center, Leibniz University Hannover, 30167 Hannover, Germany

Corresponding author: Wei Zhang (wei.zhang@l3s.de).

ABSTRACT Vision Transformer (ViT) classifiers commonly rely on a global classification (CLS) token for image-level prediction, which is poorly aligned with local interpretability requirements in medical imaging. We propose SAP-ViT, a Self-Attributing Patch Vision Transformer that replaces CLS-based classification with direct aggregation of last-layer patch embeddings. This design makes class predictions decomposable into patch-level contributions as part of the forward pass. To encourage concise and readable evidence, SAP-ViT applies sparsity regularization to the patch-weight distribution. Since self-attention mixes information across patch tokens, we further introduce an input-space back-mapping mechanism to estimate which input patches contribute to influential output patch embeddings. We evaluate SAP-ViT qualitatively and quantitatively on multiple medical image classification datasets spanning different imaging modalities. The results show that SAP-ViT maintains strong classification performance while providing consistent and interpretable patch-level explanations across datasets. Ablation results further show that both sparsity regularization and the alignment component of input-space back-mapping contribute to the final explanation quality. Overall, SAP-ViT provides a favorable trade-off between predictive accuracy and interpretability for ViT-based medical image classification.

INDEX TERMS Interpretability, Vision transformer, Medical image classification.

## INTRODUCTION

Transformer-based models have recently been explored5 across diverse medical imaging tasks [1]. In medical image3 classification, models are increasingly expected not only to<sub>0</sub> make accurate predictions but also to provide interpretable1 evidence for those predictions. This requirement is espe-<sup>6</sup> cially important in clinical settings, where model outputs may<sup>2</sup> support high-stakes decisions and where clinicians need tov inspect whether the model relies on medically meaningful<sup>i</sup> image regions.

Vision Transformer (ViT) [2] classifiers have shown stronga performance in image classification, but standard ViT classifiers usually rely on a global classification (CLS) token for prediction. The CLS token aggregates information across patch tokens and produces an image-level representation, which is effective for classification but poorly aligned with local interpretability. In medical imaging, where disease evidence is often spatially localized, it is desirable to understand how individual image regions contribute to the prediction.

A natural solution is to make the prediction itself patchdecomposable. However, patch-level attribution in ViT is non-trivial because self-attention [3] mixes information across patch tokens. As a result, an influential last-layer patch embedding may contain information not only from its corresponding input patch but also from other regions of the image. Therefore, patch-level attribution in ViT requires both a decomposable prediction mechanism and a way to relate influential output patch embeddings back to input-space evidence.

In this work, we propose SAP-ViT, a Self-Attributing Patch Vision Transformer for medical image classification. SAP-ViT replaces CLS-based classification with direct aggregation of last-layer patch embeddings. Each patch embedding is assigned a learned aggregation weight, allowing the class prediction to be decomposed into patch-level contributions as part of the forward pass. We use the term self-attributing because attribution is built into the prediction mechanism itself, rather than being produced only by an external posthoc explainer.

To encourage concise and readable explanations, SAP-ViT applies sparsity regularization to the patch-weight distribution. This encourages the model to rely on a compact set of informative patches rather than distributing evidence diffusely across the entire image. Since self-attention mixes information across tokens, SAP-ViT further introduces an input-space back-mapping mechanism. This module estimates which input patches contribute to influential output patch embeddings, providing an approximate input-space attribution of the model’s patch-level decision evidence.

Our contributions are threefold:

First, we propose a patch-additive ViT classifier that predicts directly from last-layer patch embeddings rather than the final CLS representation, enabling explicit decomposition of class predictions into patch-level contributions.

Second, we introduce sparsity regularization on patch weights to encourage the model to rely on a compact set of informative image regions.

Third, we propose an input-space back-mapping mechanism that estimates which input patches contribute to influential output patch embeddings, addressing the token-mixing effect of self-attention.

## II. RELATED WORK

## A. POST-HOC EXPLANATION METHODS

Post-hoc explanation methods explain a trained classifier without modifying its prediction mechanism. Gradient-based methods estimate importance from the sensitivity of a class score to the input or an intermediate representation. Vanilla gradient saliency directly differentiates the class score with respect to the input, integrated Gradients accumulates gradients along a path from a reference input, SmoothGrad averages gradients over perturbed inputs, and Gradient-weighted Class Activation Mapping (Grad-CAM) combines gradients with intermediate activations to produce localization maps [4]–[7]. More recently, LeGrad adapts gradient-based explanation to ViTs by estimating patch relevance through featureformation sensitivity [8]. Although these approaches do not require changes to the model architecture, their explanations may be noisy, sensitive to the selected baseline or layer, or affected by gradient saturation.

Perturbation-based methods instead estimate importance by measuring how predictions change when parts of the input are modified. Representative approaches include occlusion analysis, Randomized Input Sampling for Explanation (RISE), meaningful perturbation, differentiable patch masking for ViTs, Local Interpretable Model-agnostic Explanations (LIME), and SHapley Additive exPlanations (SHAP) [9]–[12], [29], [30]. Perturbation provides a direct way to examine the model response to selected image regions, but it can be computationally expensive and sensitive to the masking, sampling, superpixel, or reference-distribution strategy. Moreover, heavily perturbed images may depart from the distribution on which the classifier was trained.

For ViTs, attention-based methods exploit the information available in the Transformer architecture. Attention rollout estimates information flow by aggregating attention matrices across layers [15], while transformer attribution and attention-aware relevance propagation combine attention with gradients or relevance-propagation rules to produce class-specific explanations [17], [18]. However, attention can represent information routing rather than contribution to the prediction [13], [14]. More generally, these methods produce explanations after the prediction has been computed, whereas SAP-ViT incorporates a patch-level decomposition into its prediction mechanism.

## B. INTERPRETABLE-BY-DESIGN MODELS

Interpretable-by-design models construct predictions from components whose contributions can be inspected directly. BagNet predicts by aggregating evidence from local image regions, making it possible to examine the spatial evidence supporting the output [31]. Neural Additive Models extend additive prediction by learning a nonlinear contribution function for each input feature, while Sum-of-Parts constructs predictions from learned feature groups [32], [33]. These approaches provide important foundations for decomposable prediction, but they are not specifically designed for contextualized ViT patch embeddings, where self-attention repeatedly mixes information across image regions.

Several architectural approaches have been developed specifically for ViT interpretation. The Interpretability-Aware Vision Transformer (IA-ViT) jointly trains an interpreter and predictor to generate an explanation alongside the prediction [19]. Disentangling Visual Transformers aims to make patch representations more interpretable by reducing token entanglement [20], while Prompt-CAM introduces class-specific prompts and uses prompt attention to localize discriminative regions [21]. These methods improve ViT interpretability through a separate interpreter, modified token representations, or prompt attention. In contrast, SAP-ViT uses a linear classification head so that each class logit can be decomposed into patch-specific terms. The contribution to a particular class is jointly determined by the learned patch weight, the corresponding patch embedding, and the class-specific parameters of the classification head.

Recent work has also investigated patch selection for regularizing and interpreting ViTs in medical imaging. Nilsson and Azizpour learn stochastic token selection for echocardiography video analysis and left ventricular ejection fraction prediction, demonstrating the relevance of patch-level selection in medical applications [36]. Their approach interprets the model through the selected patch tokens, whereas SAP-ViT focuses on decomposing class predictions through contextualized last-layer patch embeddings and approximately mapping this evidence back to the input space.

## C. INTERPRETABILITY IN ViT

Two properties of ViTs are particularly important for local interpretation. First, a CLS token is commonly used as the global representation for classification. Through selfattention, it interacts with the patch tokens and aggregates information from the image, making a CLS-based prediction difficult to decompose into local contributions [2], [34]. Second, self-attention repeatedly mixes information across patch tokens. Consequently, a last-layer patch embedding may contain information from several input regions rather than only its spatially corresponding input patch. The emergence of highnorm or special-purpose tokens further demonstrates that token representations should not automatically be interpreted as direct spatial evidence [35]. Therefore, the location of an influential output patch token does not necessarily identify the input region responsible for its contribution.

Existing attention- and relevance-based methods attempt to trace information flow through the Transformer [15], [17], [18], but they do not make the class prediction itself additive over patch-level evidence. Conversely, weighting the last-layer patch embeddings and applying a linear classification head produces a decomposable class prediction, but the resulting output embeddings remain contextualized through self-attention. Their weights therefore do not by themselves establish which input regions supplied the information used by each output embedding. This leaves a specific interpretability gap: connecting patch-additive prediction to input-space evidence while explicitly accounting for token mixing. SAP-ViT addresses this gap through patchadditive prediction, sparsity regularization, and input-space back-mapping. The back-mapping provides an approximate association between contextualized output evidence and input patches rather than guaranteed recovery of causal input contributions.

## III. METHOD

We propose SAP-ViT, a Self-Attributing Patch Vision Transformer for medical image classification. Figure 1 presents an overview of the complete framework and its three main components: patch-additive prediction, sparsity regularization, and input-space back-mapping. The goal of SAP-ViT is to make the image-level prediction decomposable into patchlevel contributions while accounting for token mixing in self attention.

SAP-ViT does not use the final CLS representation for classification. Given an input image, we split it into a sequence of patch tokens and process them using the pretrained Transformer encoder, in which the CLS token remains present during encoding. The final prediction is computed exclusively from the last-layer patch embeddings. This design allows the class score to be decomposed into contributions from individual output patch tokens.

However, because self-attention mixes information across tokens, an output patch embedding should not be interpreted as representing only its corresponding input patch. To address this issue, SAP-ViT introduces an input-space back-mapping module that estimates which input patches contribute to influential output patch embeddings.

## A. PRELIMINARIES

A Vision Transformer (ViT) processes an image as a sequence of patch tokens. Given an input image, we divide it into M non-overlapping patches. Each patch is projected into a feature space and combined with positional encoding, producing the input patch-token sequence

$$
( e _ { 0 } ^ { 0 } , \ldots , e _ { 0 } ^ { M - 1 } )
$$

where $e _ { 0 } ^ { j }$ denotes the embedding of the j-th input patch.

SAP-ViT uses a Transformer encoder with N layers. Each layer takes a sequence of tokens as input and outputs a sequence of updated tokens. Due to the self-attention mechanism, each output token can integrate information from other tokens in the sequence. Therefore, after multiple Transformer layers, each patch embedding becomes context-aware.

Unlike standard ViT classifiers that use the final CLS representation for image-level classification, SAP-ViT computes its prediction exclusively from the final-layer patch embeddings. Suppressing the CLS state for notational simplicity, the patch-token states are written as

$$
\bigl [ e _ { i } ^ { 0 } , \ldots , e _ { i } ^ { M - 1 } \bigr ] = T _ { i } \bigl ( \bigl [ e _ { i - 1 } ^ { 0 } , \ldots , e _ { i - 1 } ^ { M - 1 } \bigr ] \bigr ) , \quad i = 1 , \ldots , N\tag{1}
$$

where $T _ { i }$ denotes the i-th Transformer layer.

The final-layer patch embeddings are denoted as

$$
( e _ { N } ^ { 0 } , \ldots , e _ { N } ^ { M - 1 } )
$$

Each $e _ { N } ^ { j }$ may contain information not only from its corresponding input patch $e _ { 0 } ^ { j } { \mathrm { . } }$ , but also from other input patches. This token-mixing property is important for SAP-ViT: it motivates both the patch-additive prediction mechanism and the input-space back-mapping module introduced below.

## B. PATCH-ADDITIVE PREDICTION

Given the final-layer patch embeddings $( e _ { N } ^ { 0 } , \ldots , e _ { N } ^ { M - 1 } )$ SAP-ViT computes the image-level prediction by aggregating patch-level evidence instead of using the CLS representation for classification. For each output patch embedding $e _ { N } ^ { i }$ , a small multilayer perceptron (MLP) produces an unnormalized patch score $r _ { i } \mathrm { : }$

$$
r _ { i } = f _ { \omega } ( e _ { N } ^ { i } ) , \quad i = 0 , \dots , M - 1 ,\tag{2}
$$

where $f _ { \omega }$ denotes the patch-scoring MLP.

The scores are normalized using a softmax function to obtain the aggregation weights:

$$
\alpha _ { i } = \frac { \exp ( r _ { i } ) } { \sum _ { j = 0 } ^ { M - 1 } \exp ( r _ { j } ) } , \quad i = 0 , \ldots , M - 1 .\tag{3}
$$

The final image representation is computed as a weighted sum of the final-layer patch embeddings:

$$
z = \sum _ { i = 0 } ^ { M - 1 } \alpha _ { i } e _ { N } ^ { i } .\tag{4}
$$

The class prediction is obtained using a linear classification head h parameterized by W and b:

$$
\ell = h ( z ) = W z + b , \qquad \hat { c } = \arg \operatorname* { m a x } _ { c } \ell _ { c } ,\tag{5}
$$

where $\ell _ { c }$ denotes the logit for class $c , W _ { c } \in \mathbb { R } ^ { d }$ denotes the class-specific weight vector such that $W _ { c } ^ { \top }$ is the c-th row of W, and $b _ { c }$ is the corresponding bias. Because the classification head is linear, the class logit can be decomposed as

$$
\ell _ { c } = b _ { c } + \sum _ { i = 0 } ^ { M - 1 } \underbrace { \alpha _ { i } W _ { c } ^ { \top } e _ { N } ^ { i } } _ { a _ { i , c } } ,\tag{6}
$$

where $a _ { i , c }$ is the contribution of the i-th last-layer patch embedding to class c. The normalized aggregation weight $\alpha _ { i }$ determines how strongly the patch embedding enters the image representation, but it is not alone a class-specific contribution. The contribution to class c jointly depends on $\alpha _ { i } ,$ the contextualized patch embedding $e _ { N } ^ { i }$ , and the class-specific classifier parameters $W _ { c }$

![](images/af5d1bf3f31a629ff430a815d43195c1b78e2b4334ac87f574a3f2dde8a6ee25.jpg)  
FIGURE 1: Overview of SAP-ViT. The input image is divided into patches and processed by the ViT to obtain contextualized patch embeddings. SAP-ViT learns a weight for each output patch, while sparsity regularization encourages the model to focus on a compact set of informative patches. The weighted patch embeddings are additively aggregated and passed to a linear classification head to obtain the predicted class. Because self-attention mixes information across patches, an influential lastlayer patch embedding $e _ { N } ^ { k }$ may contain both local information from its corresponding input patch embedding $e _ { 0 } ^ { k }$ and contextua information from other image regions, represented by the latent context embedding $\overline { { \hat { e } ^ { k } } }$ . This motivates input-space backmapping, which uses the pair $( e _ { 0 } ^ { k } , e _ { N } ^ { k } )$ to estimate the contextual representation $\hat { e } ^ { k }$ through a conditional VAE. The recovered context representation is compared with all input patch embeddings, and the resulting matching maps for the top-K outpu patches are combined to produce an approximate input-space attribution map.

## C. SPARSITY REGULARIZATION FOR SELECTIVE EVIDENCE

To make the patch-level evidence easier to interpret, we encourage SAP-ViT to rely on a compact set of informative patch tokens rather than distributing the aggregation weights diffusely across all patches. The strength of this sparsity bias is controlled by a regularization hyperparameter.

We apply an entropy regularization term to the normalized aggregation weights:

$$
\mathcal { L } _ { \mathrm { s p a r s e } } = H ( \alpha ) = - \sum _ { i = 0 } ^ { M - 1 } \alpha _ { i } \log \alpha _ { i } .\tag{7}
$$

The final classifier-training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { s p a r s e } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ is the classification loss and $\lambda _ { 1 }$ controls the strength of sparsity regularization.

Minimizing the entropy encourages the aggregationweight distribution to become more concentrated, so that a smaller number of output patch embeddings receive high weights. This improves the readability of the patch-level evidence while still allowing the model to use multiple patches when necessary.

## D. INPUT-SPACE BACK-MAPPING

The patch-additive prediction mechanism identifies influential final-layer patch embeddings. However, due to selfattention, an influential output patch embedding should not be interpreted as representing only its corresponding input patch. For the k-th input patch embedding $e _ { 0 } ^ { k } .$ , the Transformer encoder produces the corresponding final-layer embedding $e _ { N } ^ { k }$ . While $e _ { 0 } ^ { k }$ represents the local input patch, $e _ { N } ^ { k }$ may also contain contextual information from other patches. Therefore, when $e _ { N } ^ { k }$ receives a high aggregation weight, we further estimate which input patches are associated with the contextual information contained in this output embedding.

SAP-ViT introduces an input-space back-mapping module for this purpose. Given the pair $( e _ { 0 } ^ { k } , e _ { N } ^ { k } )$ , the module estimates a context embedding $\hat { e } ^ { k }$ that captures information in $e _ { N } ^ { k }$ beyond the local input patch $e _ { 0 } ^ { k }$ . The embeddings $e _ { 0 } ^ { k } , e _ { N } ^ { k }$ and $\hat { e } ^ { \check { k } }$ have the same dimensionality. We model this context estimation using a conditional variational autoencoder (conditional VAE) [22], [23]. After training, the projected encoder mean is compared with all input patch embeddings to obtain input-space match scores. A higher match score indicates that an input patch is more aligned with the contextual signal represented in $e _ { N } ^ { k }$

## 1) Conditional VAE

We model $e _ { N } ^ { k }$ as a function of its local input patch $e _ { 0 } ^ { k }$ and a latent context embedding $\hat { e } ^ { k }$ . The encoder $q _ { \phi } ( \hat { e } ^ { k } \mid e _ { 0 } ^ { k } , e _ { N } ^ { k } )$ produces a Gaussian posterior, and the decoder $p _ { \theta } ( e _ { N } ^ { k } \mid \hat { e } ^ { k } , e _ { 0 } ^ { k } )$

reconstructs $e _ { N } ^ { k }$ from $( \hat { e } ^ { k } , e _ { 0 } ^ { k } )$ . The training objective $\mathcal { L } _ { 1 }$ of conditional VAE consists of a reconstruction term and a Kullback–Leibler (KL) regularization term:

$$
\begin{array} { r } { \mathcal { L } _ { 1 } = \mathbb { E } _ { k } \Big [ \mathbb { E } _ { q _ { \phi } ( \hat { e } ^ { k } \mid e _ { 0 } ^ { k } , e _ { N } ^ { k } ) } \big [ - \log p _ { \theta } ( e _ { N } ^ { k } \mid \hat { e } ^ { k } , e _ { 0 } ^ { k } ) \big ] } \\ { + \beta \mathrm { K L } \big ( q _ { \phi } ( \hat { e } ^ { k } \mid e _ { 0 } ^ { k } , e _ { N } ^ { k } ) \mid \mid p ( \hat { e } ^ { k } ) \big ) \Big ] . } \end{array}\tag{9}
$$

where $p ( \hat { e } ^ { k } ) = \mathcal { N } ( 0 , I )$ is the Gaussian prior and $\beta$ controls the strength of the KL regularization.

## 2) Alignment

The latent embedding $\hat { e } ^ { k }$ learned by the conditional VAE is not guaranteed to be directly comparable with the input patch embeddings, because latent variables are only identifiable up to transformations. Consequently, cosine similarity between $\hat { e } ^ { k }$ and input tokens $e _ { 0 } ^ { j }$ may not be meaningful without additional constraints. Since our goal is to retrieve input patches aligned with the contextual information in $e _ { N } ^ { k }$ , we introduce an alignment term that places the estimated context embedding in the input-token space.

Specifically, we take the encoder mean $\mu _ { \phi } ( e _ { 0 } ^ { k } , e _ { N } ^ { k } )$ and pass it through a lightweight projection head $g _ { \psi } ( \cdot )$ that maps the latent representation into the input-token space. The alignment loss penalizes the angular discrepancy between the projected context representation and the corresponding input patch embedding:

$$
\mathcal { L } _ { 2 } = \mathbb { E } _ { k } \left[ 1 - \cos \bigl ( g _ { \psi } ( \mu _ { \phi } ( e _ { 0 } ^ { k } , e _ { N } ^ { k } ) ) , e _ { 0 } ^ { k } \bigr ) \right] .\tag{10}
$$

The full objective for the back-mapping module is

$$
{ \mathcal { L } } _ { \mathrm { m a p } } = { \mathcal { L } } _ { 1 } + \lambda _ { 2 } { \mathcal { L } } _ { 2 } ,\tag{11}
$$

where $\lambda _ { 2 } ~ > ~ 0$ controls the strength of the alignment term. The first term encourages the estimated context embedding to reconstruct $e _ { N } ^ { k }$ together with the local patch embedding $e _ { 0 } ^ { k } { \mathrm { . } }$ , while the alignment term provides an inductive bias that makes the estimated context comparable to input patch embeddings.

## 3) Scoring

After training the back-mapping module, we compare the projected encoder mean for the k-th output patch with each input patch embedding $e _ { 0 } ^ { j }$

$$
s _ { k  j } = \cos ( g _ { \psi } ( \mu _ { \phi } ( e _ { 0 } ^ { k } , e _ { N } ^ { k } ) ) , e _ { 0 } ^ { j } ) .\tag{12}
$$

The score $s _ { k \to j }$ measures the alignment between the $j \mathrm { - t h }$ input patch and the projected latent representation inferred from the output patch embedding $e _ { N } ^ { k }$ . We use these scores to produce an input-space attribution map. This back-mapping should be interpreted as an approximate attribution mechanism, rather than a guaranteed recovery of the true causal contribution of each input patch.

For input-space back-mapping, we select output patches using the normalized aggregation weights $\alpha _ { i }$ . These weights are input-dependent but class-independent and quantify the relative importance assigned to each patch when constructing the pooled image representation. Let K denote the hyperparameter controlling the number of selected last-layer output patches, and let

Algorithm 1 Training and inference workflow of SAP-ViT   
Input: Training set D, test image x, and number of selected   
output patches K   
Output: Predicted class ˆc and input-space attribution map A   
1: Train SAP-ViT using patch-additive prediction with spar  
sity regularization by minimizing the objective in Eq. (8).   
2: Freeze the trained SAP-ViT and train the input-space   
back-mapping module by minimizing the objective in   
Eq. (11).   
3: For x, compute the class prediction and normalized patch   
weights according to Eqs. (2)–(5).   
4: Select the K highest-weight last-layer output patches   
according to Eq. (13).   
5: for each selected output patch $k \in \mathcal { K }$ do   
6: Compute its context embedding and input-patch   
matching scores according to Eq. (12).   
7: end for   
8: Obtain A by summing the matching-score maps accord  
ing to Eq. (13).   
9: return ˆc and A

$$
\mathcal { K } = \mathrm { T o p K } \left( \{ \alpha _ { i } \} _ { i = 0 } ^ { M - 1 } , K \right) , \quad \quad A _ { j } = \sum _ { k \in \mathcal { K } } s _ { k \to j } ,\tag{13}
$$

where K contains the indices of the K highest-weight output patches and $A _ { j }$ is the aggregated attribution score of the j-th input patch.

Algorithm 1 summarizes the complete training and inference workflow. SAP-ViT is trained first, and its parameters are then frozen while the input-space back-mapping module is trained. During inference, the K highest-weight output patches are selected, and their input-space matching-score maps are summed to produce the final attribution map.

## IV. EXPERIMENT

## A. SETUP

## 1) Datasets

We evaluate SAP-ViT on four public medical imaging datasets spanning three imaging modalities: brain magnetic resonance imaging (MRI), chest X-ray, and ultrasound. The datasets cover both binary and multi-class classification tasks. The brain tumor MRI dataset [24] is a four-class classification task including glioma, meningioma, no tumor, and pituitary tumor. The COVID chest X-ray dataset [25], [26] is a fourclass task including COVID, lung opacity, normal, and viral pneumonia. The pneumonia chest X-ray dataset [27] is a binary classification task including normal and pneumonia. The breast ultrasound dataset [28] is a three-class task including normal, benign, and malignant.

For each dataset, we resize all images to a fixed resolution and apply the same preprocessing pipeline across SAP-ViT and all baselines. We split each dataset into training, validation, and test sets, and report results on the test set. Table 1 summarizes the number of classes and the number of images used for each dataset.

TABLE 1: Dataset statistics.
<table><tr><td>Dataset</td><td>Modality</td><td># Classes</td><td># Images</td></tr><tr><td>Brain tumor MRI</td><td>MRI</td><td>4</td><td>7023</td></tr><tr><td>COVID chest X-ray</td><td>Chest X-ray</td><td>4</td><td>21165</td></tr><tr><td>Pneumonia chest X-ray</td><td>Chest X-ray</td><td>2</td><td>5856</td></tr><tr><td>Breast ultrasound</td><td>Ultrasound</td><td>3</td><td>780</td></tr></table>

## 2) Predictive and explainability baselines

For predictive-performance comparison, we include two model-level baselines: a standard CLS-based vanilla ViT and IA-ViT [19]. The vanilla ViT uses the same Transformer configuration and training protocol as SAP-ViT but retains the CLS token and the standard CLS-based classification head. IA-ViT jointly trains an interpreter and a predictor. SAP-ViT, vanilla ViT, and IA-ViT are compared in terms of classification performance.

For explanation evaluation, we compare SAP-ViT and IA-ViT with four post-hoc explanation methods: Grad-CAM [7], attention rollout [15], LIME [29] , and SHAP [30]. All four post-hoc methods are applied to the same trained vanilla ViT described above. They therefore do not constitute separate predictive models and are used solely for explanation evaluation. The classification performance of their shared vanilla ViT backbone is reported only once.

To support reproducibility, the complete implementation code and all hyperparameter configurations for SAP-ViT are publicly available in our GitHub repository.

## B. RESULTS

## 1) Model performance

Table 2 reports classification accuracy for the models evaluated under the same experimental protocol. Among these implemented models, SAP-ViT achieves the highest mean accuracy on all four datasets, including brain tumor MRI, COVID chest X-ray, pneumonia chest X-ray, and breast ultrasound. Additional classification metrics are reported in Appendix D. As shown in Table 8, SAP-ViT also achieves the highest mean macro-averaged precision, sensitivity, and F1 score among the implemented models on all four datasets.

Compared with IA-ViT, SAP-ViT obtains substantially higher accuracy on all four datasets. This indicates that the proposed patch-level self-attributing design does not compromise predictive performance compared with the interpretableby-design baseline. Compared with the vanilla ViT model used for the post-hoc explanation baselines, SAP-ViT also achieves higher accuracy across all datasets, showing that the proposed architecture remains competitive with standard ViT-based classification while providing intrinsic patch-level attribution.

To contextualize SAP-ViT’s predictive performance, Appendix C compares SAP-ViT with representative recent studies [37]–[39] evaluated on the same brain tumor MRI dataset. Despite differences in data splits, preprocessing, and training protocols, SAP-ViT achieves classification performance comparable to that reported by these studies. Its principal advantage lies in combining competitive predictive performance with an intrinsic patch-attribution mechanism, rather than universally outperforming existing classifiers in accuracy. We next evaluate the faithfulness of the resulting attribution maps.

## 2) Explanation evaluation

We evaluate explanation faithfulness using insertion (I), deletion (D), and their difference (I − D). A higher insertion score indicates that the most important regions identified by the explanation are sufficient to recover the model prediction when progressively inserted into a masked image. A lower deletion score indicates that removing the most important regions more strongly suppresses the prediction. Therefore, a larger I − D gap reflects explanations that are both more sufficient under insertion and more necessary under deletion.

These perturbation-based metrics evaluate explanation faithfulness with respect to the model’s prediction under the selected insertion and deletion protocol. They do not establish that the highlighted regions are clinically correct, causally related to the disease, or consistent with expert interpretation. Establishing clinical validity would require expert assessment or ground-truth lesion annotations, which are beyond the scope of this study.

As shown in Table 3, SAP-ViT achieves the highest mean I − D across all datasets, indicating the most consistent overall explanation quality among the evaluated methods. In addition, SAP-ViT obtains the best I−D score on three of four datasets: brain tumor MRI, COVID chest X-ray, and breast ultrasound.

Compared with the interpretable-by-design baseline IA-ViT, SAP-ViT produces substantially larger I − D gaps on all datasets. IA-ViT obtains near-zero or negative I − D values, suggesting that its explanation maps do not consistently identify regions that are both sufficient and necessary for the prediction. In contrast, SAP-ViT explicitly decomposes the prediction into sparse patch-level contributions, leading to more discriminative attribution maps.

Compared with post-hoc baselines, SAP-ViT also shows stronger overall performance. It outperforms LIME, SHAP, Grad-CAM, and Rollout on brain tumor MRI and COVID chest X-ray, and achieves the best result on breast ultrasound. These results suggest that integrating attribution into the prediction mechanism can improve explanation faithfulness while maintaining competitive performance.

SAP-ViT does not achieve the best I − D score on the pneumonia chest X-ray dataset. This suggests that, although SAP-ViT provides the strongest overall performance across datasets, its sparse patch-level attribution mechanism may not be equally advantageous in all imaging settings. Pneumonia chest X-ray may represent a case where discriminative evidence is more spatially distributed, making it less aligned with explanations that emphasize a small number of highly influential patches.

TABLE 2: Model performance measured by classification accuracy.
<table><tr><td></td><td>Brain tumor MRI</td><td>COVID chest X-ray</td><td>Pneumonia chest X-ray</td><td>Breast ultrasound</td></tr><tr><td>SAP-ViT IA-ViT</td><td> $0 . 9 9 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 5 5 \pm 0 . 0 0 3$   $0 . 6 5 8 \pm 0 . 0 6 6$ </td><td> $0 . 9 0 9 \pm 0 . 0 1 9$   $0 . 7 1 3 \pm 0 . 0 6 7$ </td><td> $0 . 8 7 8 \pm 0 . 0 1 9$   $0 . 5 5 9 \pm 0 . 0 0 6$ </td></tr><tr><td>Vanilla ViT</td><td> $0 . 9 0 5 \pm 0 . 0 2 0$ </td><td></td><td></td><td></td></tr><tr><td>(post-hoc backbone)</td><td> $0 . 9 7 3 \pm 0 . 0 0 8$ </td><td> $0 . 9 2 9 \pm 0 . 0 0 6$ </td><td> $0 . 8 8 8 \pm 0 . 0 2 5$ </td><td> $0 . 7 8 0 \pm 0 . 0 7 4$ </td></tr></table>

TABLE 3: Explanation evaluation. I denotes the insertion score, D denotes the deletion score, and I − D represents the gap between insertion and deletion. The last column reports the mean I − D across all datasets.
<table><tr><td></td><td></td><td>Brain tumor MRI</td><td>COVID chest X-ray</td><td>Pneumonia chest X-ray</td><td>Breast ultrasound</td><td>Mean  $I - D$ </td></tr><tr><td rowspan="3">SAP-ViT</td><td> $D$ </td><td> $0 . 4 0 0 \pm 0 . 0 2 0$ </td><td> $0 . 4 4 9 \pm 0 . 0 2 5$ </td><td> $0 . 6 5 9 \pm 0 . 0 1 3$ </td><td> $0 . 5 2 2 \pm 0 . 0 2 4$ </td><td>一</td></tr><tr><td> $I$ </td><td> $0 . 7 3 3 \pm 0 . 0 0 7$ </td><td> $0 . 6 7 7 \pm 0 . 0 3 3$ </td><td> $0 . 7 8 9 \pm 0 . 0 2 5$ </td><td> $0 . 6 9 3 \pm 0 . 0 2 2$ </td><td></td></tr><tr><td> $I - D$ </td><td> ${ \bf 0 . 3 3 3 \pm 0 . 0 2 2 }$ </td><td> ${ \bf 0 . 2 2 8 \pm 0 . 0 3 5 }$ </td><td> $0 . 1 3 0 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 1 7 1 \pm 0 . 0 1 2 }$ </td><td>0.216</td></tr><tr><td rowspan="3">IA-ViT</td><td> $D$ </td><td> $0 . 4 2 3 \pm 0 . 0 1 5$ </td><td> $0 . 4 2 5 \pm 0 . 0 4 3$ </td><td> $0 . 5 8 0 \pm 0 . 0 5 6$ </td><td> $0 . 4 0 8 \pm 0 . 0 1 0$ </td><td>一</td></tr><tr><td> $I$ </td><td> $0 . 4 1 6 \pm 0 . 0 2 0$ </td><td> $0 . 4 3 9 \pm 0 . 0 3 1$ </td><td> $0 . 5 7 7 \pm 0 . 0 2 3$ </td><td> $0 . 4 1 2 \pm 0 . 0 2 5$ </td><td></td></tr><tr><td> $I - D$ </td><td> $- 0 . 0 0 7 \pm 0 . 0 1 3$ </td><td> $0 . 0 1 4 \pm 0 . 0 3 1$ </td><td> $- 0 . 0 0 3 \pm 0 . 0 4 7$ </td><td> $0 . 0 0 4 \pm 0 . 0 2 7$ </td><td>0.002</td></tr><tr><td rowspan="3">LIME</td><td> $D$ </td><td> $0 . 3 8 0 \pm 0 . 0 1 5$ </td><td> $0 . 3 6 3 \pm 0 . 0 4 2$ </td><td> $0 . 6 0 8 \pm 0 . 0 4 0$ </td><td> $0 . 5 0 2 \pm 0 . 0 5 9$ </td><td>一</td></tr><tr><td> $I$ </td><td> $0 . 5 1 8 \pm 0 . 0 4 1$ </td><td> $0 . 4 8 4 \pm 0 . 0 6 7$ </td><td> $0 . 8 5 6 \pm 0 . 0 3 8$ </td><td> $0 . 5 6 8 \pm 0 . 0 2 0$ </td><td></td></tr><tr><td> $I - D$ </td><td> $0 . 1 3 8 \pm 0 . 0 3 0$ </td><td> $0 . 1 2 1 \pm 0 . 0 4 7$ </td><td> $\mathbf { 0 . 2 4 8 \pm 0 . 0 6 6 }$ </td><td> $0 . 0 6 6 \pm 0 . 0 4 8$ </td><td>0.143</td></tr><tr><td rowspan="3">SHAP</td><td> $D$ </td><td> $0 . 4 0 1 \pm 0 . 0 2 4$ </td><td> $0 . 4 1 3 \pm 0 . 0 5 1$ </td><td> $0 . 7 1 4 \pm 0 . 0 0 8$ </td><td> $0 . 5 8 1 \pm 0 . 0 1 5$ </td><td>一</td></tr><tr><td> $I$ </td><td> $0 . 4 9 0 \pm 0 . 0 3 1$ </td><td> $0 . 4 8 1 \pm 0 . 0 6 3$ </td><td> $0 . 7 9 1 \pm 0 . 0 3 0$ </td><td> $0 . 5 6 8 \pm 0 . 0 3 4$ </td><td>1</td></tr><tr><td> $I - D$ </td><td> $0 . 0 8 9 \pm 0 . 0 1 0$ </td><td> $0 . 0 6 8 \pm 0 . 0 2 1$ </td><td> $0 . 0 7 6 \pm 0 . 0 2 8$ </td><td> $- 0 . 0 1 3 \pm 0 . 0 2 4$ </td><td>0.055</td></tr><tr><td rowspan="3">Grad-CAM</td><td> $D$ </td><td> $0 . 4 0 2 \pm 0 . 0 2 3$ </td><td> $0 . 4 6 1 \pm 0 . 0 6 0$ </td><td> $0 . 6 9 4 \pm 0 . 0 3 3$ </td><td> $0 . 5 0 6 \pm 0 . 0 6 2$ </td><td>一</td></tr><tr><td> $I$ </td><td> $0 . 5 1 9 \pm 0 . 0 5 3$ </td><td> $0 . 5 1 5 \pm 0 . 0 3 1$ </td><td> $0 . 8 8 1 \pm 0 . 0 3 6$ </td><td> $0 . 6 6 9 \pm 0 . 0 2 5$ </td><td></td></tr><tr><td> $I - D$ </td><td> $0 . 1 1 7 \pm 0 . 0 3 9$ </td><td> $0 . 0 5 5 \pm 0 . 0 4 9$ </td><td> $0 . 1 8 7 \pm 0 . 0 4 9$ </td><td> $0 . 1 6 3 \pm 0 . 0 5 1$ </td><td>0.131</td></tr><tr><td rowspan="3">Rollout</td><td> $D$ </td><td> $0 . 3 7 1 \pm 0 . 0 1 8$ </td><td> $0 . 4 5 2 \pm 0 . 0 4 8$ </td><td> $0 . 6 8 9 \pm 0 . 0 1 9$ </td><td> $0 . 5 3 3 \pm 0 . 0 3 4$ </td><td></td></tr><tr><td> $I$ </td><td> $0 . 5 5 4 \pm 0 . 0 3 5$ </td><td> $0 . 5 9 5 \pm 0 . 0 5 3$ </td><td> $0 . 8 2 5 \pm 0 . 0 1 0$ </td><td> $0 . 5 9 9 \pm 0 . 0 3 6$ </td><td>一</td></tr><tr><td> $I - D$ </td><td> $0 . 1 8 3 \pm 0 . 0 2 2$ </td><td> $0 . 1 4 3 \pm 0 . 0 4 4$ </td><td> $0 . 1 3 6 \pm 0 . 0 1 7$ </td><td> $0 . 0 6 7 \pm 0 . 0 1 4$ </td><td>0.132</td></tr></table>

To further examine whether this dataset-dependent behavior is related to image characteristics, we conducted a modelindependent patch-level radiomics-style concentration analysis. For each dataset, we sampled up to 500 images, computed simple patch-level texture features, summarized their spatial concentration per image, and averaged the resulting metrics at the dataset level. We report normalized entropy and Gini coefficient as complementary measures. Higher normalized entropy indicates a more spatially diffuse patch-level texture distribution, whereas higher Gini coefficient indicates stronger spatial concentration.

As shown in Table 4, breast ultrasound has the lowest normalized entropy and the highest Gini coefficient, followed by brain tumor MRI, indicating that their patch-level texture responses are relatively spatially concentrated. SAP-ViT achieves the best I − D scores on both datasets, which is consistent with its main advantage: sparse patch aggregation can isolate a compact set of influential output patches, while input-space back-mapping approximately relates their contextualized information to the corresponding input regions.

In contrast, both chest X-ray datasets exhibit higher normalized entropy and lower Gini values, indicating more spatially diffuse texture responses. Pneumonia chest X-ray has the highest normalized entropy and the lowest Gini coefficient, providing one possible explanation for why SAP-

![](images/708629cd29000e7e2a5f354c728ab8f96ea9f7545197c0ba2968d334994885f9.jpg)  
FIGURE 2: Qualitative attribution comparison on a Brain Tumor MRI example. The ground-truth label is meningioma. SAP-ViT produces a compact attribution map concentrated around the tumor region, while the baseline methods generate either less localized or more diffuse saliency patterns across surrounding anatomical structures and background regions.

TABLE 4: Model-independent patch-level radiomics-style concentration analysis. Entropy<sub>norm</sub> measures the spatial spread of patch-level texture responses, where higher values indicate more diffuse responses. Gini measures concentration, where higher values indicate stronger spatial concentration.
<table><tr><td>Dataset</td><td> $\operatorname { E n t r o p y } _ { \mathrm { n o r m } }$ </td><td>Gini</td></tr><tr><td>Brain tumor MRI</td><td>0.901</td><td>0.476</td></tr><tr><td>COVID chest X-ray</td><td>0.949</td><td>0.333</td></tr><tr><td>Pneumonia chest X-ray</td><td>0.953</td><td>0.322</td></tr><tr><td>Breast ultrasound</td><td>0.705</td><td>0.488</td></tr></table>

TABLE 5: Ablation study of SAP-ViT on the Brain Tumor MRI dataset.
<table><tr><td>Variant</td><td>Deletion ↓</td><td>Insertion ↑</td><td> $I - D \uparrow$ </td></tr><tr><td>SAP-ViT w/o sparsity</td><td> $0 . 5 5 3 \pm 0 . 0 5 2$ </td><td> $0 . 4 9 3 \pm 0 . 0 2 3$ </td><td> $- 0 . 0 6 0 \pm 0 . 0 5 8$ </td></tr><tr><td>SAP-ViT w/o alignment</td><td> $0 . 4 9 0 \pm 0 . 0 7 9$ </td><td> $0 . 5 7 7 \pm 0 . 0 8 1$ </td><td> $0 . 0 8 7 \pm 0 . 1 5 8$ </td></tr><tr><td>Full SAP-ViT</td><td> $0 . 4 0 0 \pm 0 . 0 2 0$ </td><td> $0 . 7 3 3 \pm 0 . 0 0 7$ </td><td> $0 . 3 3 3 \pm 0 . 0 2 2$ </td></tr></table>

ViT does not achieve the best $I - D$ score on this dataset: its sparsity bias may under-represent information distributed across broad image regions. Nevertheless, SAP-ViT achieves the best I −D score on the COVID chest X-ray dataset despite its relatively diffuse texture responses. This result indicates that SAP-ViT is not restricted to strongly localized image patterns and can still identify compact, prediction-relevant evidence within globally diffuse images.

The radiomics-style analysis is complementary rather than conclusive, because these model-independent texture statistics do not directly measure class-discriminative evidence. Overall, the results suggest that SAP-ViT is particularly advantageous when informative patterns can be represented by a compact set of patches, while remaining effective across datasets with substantially different spatial characteristics.

## 3) Ablation study

Table 5 reports the ablation study on the Brain Tumor MRI dataset. Removing the sparsity regularization leads to a substantial drop in explanation quality, with the $I \mathrm { ~ - ~ } D$ score becoming negative. This indicates that without sparsity, the model no longer identifies compact and discriminative evidence regions, and the resulting attribution becomes less faithful under insertion and deletion evaluation.

Removing the alignment term also degrades performance, although the drop is less severe than removing sparsity. This suggests that the alignment component is important for making the recovered context embedding comparable to the input patch space, thereby improving the quality of input-space back-mapping. The full SAP-ViT achieves the best deletion, insertion, and I − D scores, showing that both sparsity and alignment contribute to the final explanation quality.

To examine whether these findings generalize across datasets, we further conduct the same ablation study on COVID chest X-ray, pneumonia chest X-ray, and breast ultrasound. The corresponding results are reported in Appendix A, specifically in Table 6. The results show a consistent trend across all three datasets: removing sparsity causes the largest degradation, while removing alignment also reduces explanation quality compared with the full SAP-ViT.

## 4) Qualitative attribution analysis

Figure 2 shows a qualitative comparison on the Brain Tumor MRI dataset. The example contains a visible tumor region in the left intracranial area. SAP-ViT produces a compact and highly localized attribution map, with the strongest response concentrated around the tumor region and limited activation elsewhere. This suggests that SAP-ViT identifies a small set of discriminative patches that are spatially aligned with the lesion.

In contrast, IA-ViT also produces sparse high-saliency regions, but these regions are less aligned with the tumor and appear around regions outside the visually apparent tumor area, such as the skull boundary and lower head region. LIME highlights a broader superpixel region that partially overlaps with the tumor, but the attribution also extends to surrounding brain areas, making the localization less precise. SHAP produces fragmented and widespread superpixel regions across the image, including large areas outside the lesion and background, which makes the explanation harder to interpret as localized medical evidence.

Grad-CAM and Rollout generate smoother heatmaps, but their saliency is more diffuse than SAP-ViT. Grad-CAM assigns high responses to multiple regions around the skull boundary and image periphery, while Rollout spreads attribution over a larger part of the brain and nearby anatomical structures. Although these methods capture some relevant image regions, their explanations are less spatially specific. Overall, this qualitative example shows that SAP-ViT provides more localized and visually interpretable attribution than the evaluated baselines.

Additional qualitative comparisons for other datasets are provided in Appendix B. Specifically, Figs. 3, 4, and 5 present the corresponding attribution comparisons. Across these examples, SAP-ViT generally produces more compact attribution patterns, whereas the baseline methods tend to generate less localized or more diffuse responses.

## C. LIMITATIONS AND APPLICABILITY

SAP-ViT is most applicable when the discriminative evidence can be represented by a relatively compact set of image regions. Its entropy regularization introduces a sparsity bias rather than an assumption that is universally valid. When evidence is spatially distributed or multifocal, strong sparsity may suppress complementary regions and reduce explanation faithfulness, as suggested by the pneumonia chest Xray results. The regularization strength therefore represents a dataset-dependent trade-off between attribution concentration and coverage of distributed evidence. Furthermore, input-space back-mapping provides approximate associations rather than causal attribution, while insertion and deletion evaluate faithfulness to model behavior rather than clinical correctness. Finally, the present evaluation is limited to four public datasets spanning MRI, chest X-ray, and ultrasound; further evaluation is required before generalizing the findings to other medical imaging settings.

## V. CONCLUSION

In this paper, we proposed SAP-ViT, a self-attributing Vision Transformer for medical image classification. Rather than relying on a global CLS token for prediction, SAP-ViT directly aggregates last-layer patch embeddings, enabling class predictions to be decomposed into patch-level contributions. To further improve interpretability, SAP-ViT incorporates a sparsity regularization term that encourages the model to focus on a compact set of informative patches, together with an input-space back-mapping mechanism that mitigates the ambiguity introduced by token mixing in self-attention.

Experiments on multiple medical imaging datasets demonstrate that SAP-ViT maintains strong classification performance while improving the interpretability of ViT predictions. Overall, SAP-ViT provides a favorable trade-off between predictive accuracy and interpretability, making it a promising framework for transparent ViT-based medical image classification.

## APPENDIX A ADDITIONAL ABLATION RESULTS

Table 6 reports the ablation results on the three additional datasets. Consistent with the brain-tumor MRI results presented in the main text, the complete SAP-ViT outperforms the variants without sparsity regularization or alignment.

TABLE 6: Additional ablation studies of SAP-ViT on the COVID chest X-ray, Pneumonia chest X-ray, and Breast Ultrasound datasets.
<table><tr><td>Variant</td><td>Deletion ↓</td><td>Insertion ↑</td><td>I −D↑</td></tr><tr><td>COVID chest X-ray</td><td></td><td></td><td></td></tr><tr><td>SAP-ViT w/o sparsity</td><td> $0 . 5 5 9 \pm 0 . 0 5 6$ </td><td> $0 . 5 9 8 \pm 0 . 0 8 4$ </td><td> $0 . 0 3 9 \pm 0 . 1 1 3$ </td></tr><tr><td>SAP-ViT w/o alignment</td><td> $0 . 4 5 4 \pm 0 . 0 2 4$ </td><td> $0 . 6 5 6 \pm 0 . 0 4 6$ </td><td> $0 . 2 0 2 \pm 0 . 0 4 4$ </td></tr><tr><td>Full SAP-ViT</td><td> $0 . 4 4 9 \pm 0 . 0 2 5$ </td><td> $0 . 6 7 7 \pm 0 . 0 3 3$ </td><td> $0 . 2 2 8 \pm 0 . 0 3 5$ </td></tr><tr><td>Pneumonia chest X-ray</td><td></td><td></td><td></td></tr><tr><td>SAP-ViT w/o sparsity</td><td> $0 . 7 1 6 \pm 0 . 0 2 4$ </td><td> $0 . 7 7 0 \pm 0 . 0 2 2$ </td><td> $0 . 0 5 4 \pm 0 . 0 2 2$ </td></tr><tr><td>SAP-ViT w/o alignment</td><td> $0 . 6 6 8 \pm 0 . 0 2 4$ </td><td> $0 . 7 7 4 \pm 0 . 0 3 8$ </td><td> $0 . 1 0 6 \pm 0 . 0 4 4$ </td></tr><tr><td>Full SAP-ViT</td><td> $0 . 6 5 9 \pm 0 . 0 1 3$ </td><td> $0 . 7 8 9 \pm 0 . 0 2 5$ </td><td> $0 . 1 3 0 \pm 0 . 0 2 4$ </td></tr><tr><td>Breast ultrasound</td><td></td><td></td><td></td></tr><tr><td>SAP-ViT w/o sparsity</td><td> $0 . 5 5 4 \pm 0 . 0 1 5$ </td><td> $0 . 6 5 6 \pm 0 . 0 2 9$ </td><td> $0 . 1 0 3 \pm 0 . 0 3 1$ </td></tr><tr><td>SAP-ViT w/o alignment</td><td> $0 . 5 2 8 \pm 0 . 0 3 7$ </td><td> $0 . 6 8 9 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 1 \pm 0 . 0 2 5$ </td></tr><tr><td>Full SAP-ViT</td><td> $0 . 5 2 2 \pm 0 . 0 2 4$ </td><td> $0 . 6 9 3 \pm 0 . 0 2 2$ </td><td> $0 . 1 7 1 \pm 0 . 0 1 2$ </td></tr></table>

## APPENDIX B ADDITIONAL QUALITATIVE ATTRIBUTIONEXAMPLES

Figures 3, 4, and 5 provide qualitative attribution comparisons on the three additional datasets. Within each comparison, all methods are applied to the same input image. These examples illustrate the spatial characteristics of the attribution maps and are intended to complement the dataset-level quantitative evaluation.

![](images/098361e421a7aaacffc5b90dd67542969f6baada4a9f8b4f2f3ac79a19cc2209.jpg)  
FIGURE 3: Qualitative attribution comparison on a chest radiograph from the COVID chest X-ray dataset with the ground-truth label lung opacity. SAP-ViT produces a compact attribution map concentrated in an upper-lung region exhibiting visible opacity, whereas the baseline methods generate less localized or more diffuse saliency patterns across the lungs, surrounding anatomical structures, and background.

![](images/2ce04dd21623fc7b8785162964c73e7e470150469df6263c0045ea9b5333b767.jpg)  
FIGURE 4: Qualitative attribution comparison on an example from the pneumonia chest X-ray dataset with the ground-truth label pneumonia. SAP-ViT produces a compact attribution map concentrated in an upper-lung region exhibiting visible opacity, whereas the baseline methods generate less localized or more diffuse saliency patterns across the lungs, surrounding anatomical structures, and background.

## APPENDIX C COMPARISON WITH PUBLISHED CLASSIFICATION RESULTS

Table 7 compares SAP-ViT with representative recent classification results reported on the same brain tumor MRI dataset. Because the studies use different experimental protocols, the values provide literature context rather than a controlled headto-head comparison.

## APPENDIX D ADDITIONAL CLASSIFICATION METRICS

Table 8 supplements the accuracy results in Table 2 with macro-averaged precision, sensitivity, and F1 score. Macro-

![](images/f1423c9fb940879592b476e8efe0c6770414a796c562265d9fd5bba36f664746.jpg)  
FIGURE 5: Qualitative attribution comparison on a breast ultrasound example. The ground-truth label is malignant. SAP-ViT produces a compact attribution map concentrated around the lesion, with its strongest hotspot overlapping the boundary of the supplied tumor mask. In contrast, the baseline methods generate either less localized or more diffuse saliency patterns across the surrounding tissue and background regions.

TABLE 7: Comparison with representative published classification results.
<table><tr><td>Study</td><td>Model</td><td>Accuracy (%)</td></tr><tr><td>Disci et al. [39]</td><td>Xception</td><td>98.73</td></tr><tr><td>Haque et al. [38]</td><td>NeuroNet19</td><td>99.30</td></tr><tr><td>Shahin [37]</td><td>Fine-tuned ResNet34</td><td>99.66</td></tr><tr><td>SAP-ViT (ours)</td><td>SAP-ViT</td><td> $9 9 . 3 0 \pm 0 . 3 0 $ </td></tr></table>

averaging assigns equal importance to every class, providing a class-balanced assessment of predictive performance.

TABLE 8: Additional classification metrics for models evaluated under the same experimental protocol.
<table><tr><td>Model</td><td>Precision</td><td>Sensitivity</td><td>F1</td></tr><tr><td colspan="4">Brain tumor MRI</td></tr><tr><td>SAP-ViT</td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td></tr><tr><td>IA-ViT</td><td> $0 . 9 0 3 \pm 0 . 0 2 0$ </td><td> $0 . 9 0 0 \pm 0 . 0 2 2$ </td><td> $0 . 9 0 1 \pm 0 . 0 2 1$ </td></tr><tr><td>Vanilla ViT</td><td> $0 . 9 7 2 \pm 0 . 0 0 8$ </td><td> $0 . 9 7 1 \pm 0 . 0 0 9$ </td><td> $0 . 9 7 1 \pm 0 . 0 0 9$ </td></tr><tr><td colspan="4">COVID chest X-ray</td></tr><tr><td>SAP-ViT</td><td> $0 . 9 6 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 4$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 2$ </td></tr><tr><td>IA-ViT</td><td> $0 . 5 6 0 \pm 0 . 1 9 3$ </td><td> $0 . 4 9 5 \pm 0 . 1 5 1$ </td><td> $0 . 4 6 2 \pm 0 . 1 7 6$ </td></tr><tr><td>Vanilla ViT</td><td> $0 . 9 4 4 \pm 0 . 0 0 3$ </td><td> $0 . 9 2 9 \pm 0 . 0 0 9$ </td><td> $0 . 9 3 5 \pm 0 . 0 0 6$ </td></tr><tr><td colspan="4">Pneumonia chest X-ray</td></tr><tr><td>SAP-ViT</td><td> $0 . 9 3 4 \dot { \pm } 0 . 0 1 1$ </td><td> $0 . 8 8 0 \pm 0 . 0 2 6$ </td><td> $0 . 8 9 7 \pm 0 . 0 2 4$ </td></tr><tr><td>IA-ViT</td><td> $0 . 6 7 2 \pm 0 . 2 0 8$ </td><td> $0 . 6 4 4 \pm 0 . 1 0 8$ </td><td> $0 . 6 1 6 \pm 0 . 1 5 9$ </td></tr><tr><td>Vanilla ViT</td><td> $0 . 9 2 0 \pm 0 . 0 1 3$ </td><td> $0 . 8 5 3 \pm 0 . 0 3 4$ </td><td> $0 . 8 7 2 \pm 0 . 0 3 2$ </td></tr><tr><td colspan="4">Breast ultrasound</td></tr><tr><td>SAP-ViT</td><td> $0 . 8 7 1 \pm 0 . 0 1 8$ </td><td> $0 . 8 6 5 \pm 0 . 0 2 8$ </td><td> $0 . 8 6 6 \pm 0 . 0 1 9$ </td></tr><tr><td>IA-ViT</td><td> $0 . 2 1 5 \pm 0 . 0 6 5$ </td><td> $0 . 3 4 1 \pm 0 . 0 1 8$ </td><td> $0 . 2 5 6 \pm 0 . 0 4 0$ </td></tr><tr><td>Vanilla ViT</td><td> $0 . 8 0 5 \pm 0 . 0 5 1$ </td><td> $0 . 7 2 1 \pm 0 . 1 2 7$ </td><td> $0 . 7 3 3 \pm 0 . 1 1 1$ </td></tr></table>

## REFERENCES

[1] Khawaji, A., Martin, R.J. and Daspin, D., 2025. High-Resolution Medical Image Generation With Leak-Prevention Mechanism Using Quantum Transformer Learning. IEEE Access, 13, pp.206381-206401.

[2] Dosovitskiy, A., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. Attention is all you need. Advances in neural information processing systems, 30.

[4] Simonyan, K., Vedaldi, A. and Zisserman, A., 2013. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

[5] Sundararajan, M., Taly, A. and Yan, Q., 2017, July. Axiomatic attribution for deep networks. In International conference on machine learning (pp. 3319-3328). PMLR.

[6] Smilkov, D., Thorat, N., Kim, B., Viégas, F. and Wattenberg, M., 2017. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

[7] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017. Grad-cam: Visual explanations from deep networks via gradient based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

[8] Bousselham, W., Boggust, A., Chaybouti, S., Strobelt, H. and Kuehne, H., 2025. Legrad: An explainability method for vision transformers via feature formation sensitivity. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 20336-20345).

[9] Zeiler, M.D. and Fergus, R., 2014, September. Visualizing and understanding convolutional networks. In European conference on computer vision (pp. 818-833). Cham: Springer International Publishing.

[10] Petsiuk, V., Das, A. and Saenko, K., 2018. Rise: Randomized in put sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421.

[11] Fong, R.C. and Vedaldi, A., 2017. Interpretable explanations of black boxes by meaningful perturbation. In Proceedings of the IEEE international conference on computer vision (pp. 3429-3437).

[12] Nalmpantis, A., Panagiotopoulos, A., Gkountouras, J., Papakostas, K. and Aziz, W., 2023. Vision diffmask: Faithful interpretation of vision transformers with differentiable patch masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3756-3763).

[13] Jain, S. and Wallace, B.C., 2019, June. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 3543-3556).

[14] Serrano, S. and Smith, N.A., 2019, July. Is attention interpretable?. In Proceedings of the 57th annual meeting of the association for computationa linguistics (pp. 2931-2951).

[15] Abnar, S. and Zuidema, W., 2020. Quantifying attention flow in transformers. arXiv preprint arXiv:2005.00928.

[16] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R. and Samek, W., 2015. On pixel-wise explanations for non-linear classifier de cisions by layer-wise relevance propagation. PloS one, 10(7), p.e0130140.

[17] Chefer, H., Gur, S. and Wolf, L., 2021. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 782-791).

[18] Achtibat, R., Hatefi, S.M.V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S. and Samek, W., 2024. Attnlrp: attention-aware layer-wise relevance propagation for transformers. arXiv preprint arXiv:2402.05602.

[19] Qiang, Y., Li, C., Khanduri, P. and Zhu, D., 2023. Interpretability-aware vision transformer. arXiv preprint arXiv:2309.08035.

[20] Jeanneret, G., Simon, L. and Jurie, F., 2025. Disentangling Visual Transformers: Patch-level Interpretability for Image Classification. In Proceed ings of the Computer Vision and Pattern Recognition Conference (pp. 2679-2689).

[21] Chowdhury, A., Paul, D., Mai, Z., Gu, J., Zhang, Z., Mehrab, K.S., Campolongo, E.G., Rubenstein, D., Stewart, C.V., Karpatne, A. and Berger-Wolf, T., 2025. Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 4375-4385).

[22] Diederik, P.K. and Max, W., 2019. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4), pp.307- 392.

[23] Sohn, K., Lee, H. and Yan, X., 2015. Learning structured output representation using deep conditional generative models. Advances in neural information processing systems, 28.

[24] Msoud Nickparvar. (2021). Brain Tumor MRI Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886

[25] M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

[26] Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

[27] Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3

[28] Al-Dhabyani, W., Gomaa, M., Khaled, H. and Fahmy, A., 2020. Dataset of breast ultrasound images. Data in brief, 28, p.104863.

[29] Ribeiro, M.T., Singh, S. and Guestrin, C., 2016, August. " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

[30] Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

[31] Brendel, W. and Bethge, M., 2019. Approximating cnns with bag-oflocal-features models works surprisingly well on imagenet. arXiv preprint arXiv:1904.00760.

[32] Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R. and Hinton, G.E., 2021. Neural additive models: Interpretable machine learning with neural nets. Advances in neural information processing systems, 34, pp.4699-4711.

[33] You, W., Qu, H., Gatti, M., Jain, B. and Wong, E., 2025, October. Sum-ofparts: Self-attributing neural networks with end-to-end learning of feature groups. In Forty-second International Conference on Machine Learning.

[34] Marouani, A., Siméoni, O., Jégou, H., Bojanowski, P. and Vo, H.V., 2026. Revisiting [cls] and patch token interaction in vision transformers. arXiv preprint arXiv:2602.08626.

[35] Darcet, T., Oquab, M., Mairal, J. and Bojanowski, P., 2024, May. Vision transformers need registers. In International conference on learning representations (Vol. 2024, pp. 2632-2652).

[36] Nilsson, A. and Azizpour, H., 2024, July. Regularizing and Interpreting Vision Transformer by Patch Selection on Echocardiography Data. In CHIL (pp. 155-168).

[37] Shahin, A., 2025. Fine-tuned ResNet34 for efficient brain tumor classification. Scientific Reports, 15(1), p.36910.

[38] Haque, R., Hassan, M.M., Bairagi, A.K. and Shariful Islam, S.M., 2024. NeuroNet19: an explainable deep neural network model for the classification of brain tumors using magnetic resonance imaging data. Scientific reports, 14(1), p.1524.

[39] Disci, R., Gurcan, F. and Soylu, A., 2025. Advanced brain tumor classification in MR images using transfer learning and pre-trained deep CNN models. Cancers, 17(1), p.121.

![](images/347b084d5500f6f6fcd0d3caa119449938760a5d18fd599594ad2612a0e0e57f.jpg)

WEI ZHANG received the B.Sc. degree in Computer Science from Xidian University, China, and the M.Sc. degree in Computer Science from Leibniz University Hannover, Germany. He is currently pursuing the Ph.D. degree with the Faculty of Electrical Engineering and Computer Science at Leibniz University Hannover. He is working as a Research Assistant with the L3S Research Center, Hannover. He is involved in the CAIMed project (Lower Saxony Center for AI and Causal Methods

in Medicine), where he focuses on uncovering the underlying biological mechanisms of post-acute infection symptoms, specifically long COVID. His research interests include interpretability of neural networks, particularly in the areas of graph neural networks and vision transformers; the integration of causality and reinforcement learning; and foundation models in biomedicine, with a focus on predicting gene expression profiles under gene perturbations.

![](images/355cd8ad14a3834728b5dd8ca5b5833b09a25e35fe7113de097d3e0d95cce59e.jpg)

WOLFGANG NEJDL was born in 1960. He received the Dipl.Ing. and Dr.Techn. degrees in computer science from Vienna University of Technology, in 1984 and 1988, respectively. He has been a Full Professor of computer science with Leibniz University Hannover, since 1995. He was an Assistant Professor in Vienna, from 1988 to 1992, and an Associate Professor at RWTH Aachen University, from 1992 to 1995. He was a Visiting Researcher/a Professor with Xerox PARC, Stanford University, the University of Illinois at UrbanaChampaign, EPFL Lausanne, PUC Rio, Trento, and Politecnico di Milano. He heads the L3S Research Center, www.L3S.de, and the Data Science Institute/Knowledge-Based Systems at Leibniz University. From 2014 to 2019, he was a Principal Investigator of the ERC Advanced Grant ALEXANDRIA, where he worked on foundations for temporal search, exploration, and analytics in web archives. He has published more than 440 scientific articles listed at DBLP, with an H-index (based on Google Scholar) of 78. His research interests include information retrieval/search, web science, artificial intelligence, social and semantic web, digital libraries, and technology enhanced learning.

![](images/93ed9aa3d9cdd184255604f5192d2c0625aebfb51677ff89f692bf68c4cc3f76.jpg)

MING TANG received the M.Sc. degree in informatics from the Free University of Berlin, Ger many, and the Ph.D. degree in systems biology from the Johns Hopkins University School of Medicine, USA. She is currently a Group Leader with the L3S Research Center, Faculty of Electrical Engineering and Computer Science, Leib niz University Hannover. Her research interests include biomedical foundation models and agentic AI for biomedicine. She is involved in the CAIMed

project (AI and Causal Methods in Medicine) and was previously involved in the International Future Lab for Artificial Intelligence, funded by the German Federal Ministry of Education and Research (BMBF).