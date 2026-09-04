# ON THE INTERACTION BETWEEN MODEL COMPRESSION AND TEST-TIME ADAPTATION

Francesco Corti   
Graz University of Technology   
Graz, Austria   
francesco.corti@tugraz.at

Cecilia Mascolo University of Cambridge Cambridge, United Kingdom cm542@cam.ac.uk

Dong Wang   
Graz University of Technology Graz, Austria   
dong.wang@tugraz.at

Young D. Kwon Samsung AI Center-Cambridge University of Cambridge Cambridge, United Kingdom yd.kwon@samsung.com

Olga Saukh Graz University of Technology Complexity Science Hub Graz, Vienna, Austria saukh@tugraz.at

## ABSTRACT

Deep neural networks deployed in the wild must be both efficient and adaptable, requiring model compression and test-time adaptation (TTA). While both are well studied in isolation, their interaction remains poorly understood. We systematically analyze how structured compression affects a model’s ability to adapt under distribution shift. Using ResNet-18 and ViT-Base on CIFAR-10-C and ImageNet-C, we evaluate multiple compression methods combined with standard TTA techniques. We introduce a diagnostic framework that examines representational expressivity and adaptation subspace compatibility. Our results reveal a consistent gap: although compressed models retain high accuracy under supervised adaptation, their TTA performance degrades significantly with increasing compression. We show that this stems from reduced representational diversity and structural constraints that limit recoverability. These effects strongly depend on the compression method, highlighting the need to design compression strategies that preserve adaptability.

## 1 INTRODUCTION

Deployed deep neural networks (DNNs) often operate under resource constraints and distribution shifts, necessitating both compression to meet memory and latency budgets and adaptation to cope with changing environments (Han et al., 2015; Sun et al., 2020; Wang et al., 2021). While compression and model adaptation have been extensively studied in isolation, their interaction remains largely unexplored (Liang et al., 2025; Choudhary et al., 2020). In particular, it is unclear whether compression strategies can preserve the capacity of a model to adapt to distribution shifts or task variations (Liebenwein et al., 2021).

This interaction is especially critical in edge settings, where adaptation incurs substantial memory overhead due to gradient computation and the need to store intermediate activations (Dong et al., 2026; Gomez et al., 2017; Xiao et al., 2026a). For example, on resource-constrained devices such as the Raspberry Pi Zero, adaptation via entropy minimization leads to out-of-memory errors even at small batch sizes (Jia et al., 2024). Consequently, compression is not only desirable for efficient inference but also a prerequisite for enabling on-device adaptation (Lin et al., 2022).

Structured compression methods reduce model size by removing or merging entire computational units (Li et al., 2017; Wang et al., 2025). These methods can be divided into data-dependent methods, which leverage calibration data to estimate unit importance (Sun et al., 2024; Molchanov et al., 2019; LeCun et al., 1989), and data-free methods, which rely on weight statistics (Li et al., 2017) or structural redundancies (Wang et al., 2025).

Test-time adaptation (TTA) has emerged as a promising paradigm for adapting deep models using only unlabeled incoming data (Liang et al., 2025; 2020; Wang et al., 2021). Existing approaches can be broadly categorized into backpropagation-based (BP) and backpropagation-free (BP-free) methods. BP-based methods update model parameters via gradient-based optimization, typically fine-tuning a subset of parameters (Wang et al., 2021; Niu et al., 2023; Gong et al., 2023). While achieving strong adaptation performance, they incur substantial memory overhead due to gradient computation and activation storage, limiting their applicability on resource-constrained devices (Jia et al., 2024).

![](images/df1bd40d86fa0abc689eaa1ee9bb4fce33a71fbcbae7c783f05dd07b13ea83b8.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/4142807af71d47af463fe50a6a2c59681d6f0baaf0b908b5db89f7ce9f6886a4.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/20caa6aca73d184e0d9b49d2981171f0cb7aedacb2d680bc9a76924ee9b3efca.jpg)  
(c) ViT-Base on ImageNet-C

![](images/95a47bad36ed4df70dd4e8cd2dd2de604aee23bd52e482f440dc06fdd06e293e.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/ade5441370660dec4b669902e6e16bb4ec5fec3b4d5f8f842194adbf7b6cabbf.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/368889e4239ed0958636c14cfa31807c30e3e23e97973daea29a9251aecbccaf.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 1: The gap between test-time adaptation (TTA) and supervised fine-tuning (Oracle) widens with increasing compression, particularly on ImageNet-C. We compare post-compression TTA against supervised fine-tuning across architectures and datasets, averaged across all corruptions. At low sparsity levels, TTA remains competitive, suggesting that adaptation signals are largely preserved. However, as compression increases, the gap between TTA and supervised fine-tuning grows, suggesting that compressed architectures increasingly hinder effective adaptation with TTA. This effect is especially pronounced on ImageNet-C. Shaded regions indicate ±σ across corruption types. Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression (Folding, Mag-ℓ<sub>2</sub>). TTA is performed using SAR (Niu et al., 2023) for ResNet-18 and SPA (Niu et al., 2025) for ViT-Base.

In contrast, BP-free methods avoid gradient computation by adapting parameters during the forward pass (Mirza et al., 2022), modifying lightweight input prompts (Niu et al., 2024) or correcting embedding distortions via layer-wise alignment (Xiao et al., 2026b), making them more suitable for edge deployment. Despite this progress, existing studies predominantly assume dense models and do not account for compression (Mirza et al., 2022; Wang et al., 2021; Niu et al., 2023; Wang et al., 2022; Niu et al., 2022; Sun et al., 2020). This assumption is misaligned with resource-constrained deployment scenarios, where models are routinely compressed to satisfy hardware and efficienc constraints (Fang et al., 2023; Ashkboos et al., 2024; Ma et al., 2023).

Recent studies have shown that compression degrades learned representation quality (Corti et al., 2022), and erodes robustness under distribution shift (Liebenwein et al., 2021). Yet existing work that explicitly considers the compression and adaptation interaction remains limited to quantization (Xiao et al., 2024; Deng et al., 2025). Whether structurally pruned or folded models preserve the representational and geometrical properties required for test-time adaptation remains an open question. In this work, we address the following question:

## Does structured compression degrade a model’s ability to adapt at test time, or can suitable compression strategies preserve adaptability under distribution shift and task variation?

This question is closely related to continual learning: models deployed at the edge must retain plasticity, i.e., the ability to adapt to incoming data over time (Wang et al., 2022). However, compression applied to meet deployment constraints may inadvertently erode this capacity. If compression reduces feature diversity, misaligns the unsupervised adaptation signal from the supervised gradient direction, or restricts the useful update subspace, the standard compress-and-deploy pipeline can yield models that maintain high accuracy on source data but lose the ability to adapt. We refer to this phenomenon as silent plasticity loss.

Figure 1 provides empirical evidence of this effect: while compressed models retain strong performance under supervised adaptation, their test-time adaptation performance degrades substantially as compression increases, leading to a widening gap between the two. This gap indicates that, despite preserved predictive performance, compression can impair the model’s ability to effectively utilize unlabeled test-time signals. A per-corruption decomposition of this gap (Fig. 6) shows that the widening trend generalizes across most corruption types rather than being driven by a subset. To isolate the compression-induced component of this widening from the supervised-unsupervised signal-quality difference already present for the dense model, Fig. 10 reports the Oracle-TTA gap after subtracting its zero-compression value.

We introduce a framework to diagnose how compression interacts with TTA along two complementary dimensions (Sec. 3). First, diversity/expressivity: compression may reduce the variability and richness of internal representations, limiting the range of functions the model can realize during TTA. We quantify this using representation similarity between dense and compressed models (Kornblith et al., 2019), as well as activation entropy across feature maps and tokens (Skean et al., 2025). Second, subspace compatibility: TTA updates are restricted to a small parameter subspace (e.g., normalization layers), and compression may misalign this subspace with useful update directions. We quantify this via gradient cosine similarity between unsupervised and supervised updates, and provide a theoretical analysis of objective-induced gradient degeneracy in both entropy-based and consistency-based TTA objectives.

Our empirical analysis across convolutional and transformer architectures reveals the following insights:

• The gap between TTA and supervised adaptation (Oracle) widens with increasing compression, suggesting that TTA depends on structural properties of dense models that are not preserved under compression.

• TTA degrades representations already at low compression levels and fails to recover them at higher sparsity, whereas supervised adaptation consistently restores them, highlighting TTA ’s sensitivity to compression induced representational distortions.

• Compression induces two distinct failure modes of gradient alignment between TTA and supervised updates: gradient degeneracy, where near-uniform predictions attenuate the TTA signal, and active divergence, where the TTA gradient opposes the supervised gradient direction even at minimal sparsity. We show theoretically that both entropy- and consistency-based TTA objectives exhibit gradient collapse under compression, while the supervised cross-entropy gradient can remain more informative at the same sparsity.

• The severity of these effects depends on the compression method: Wanda and Taylor better preserve recoverable structure, while OBD and $\mathbf { M a g - } \ell _ { 2 }$ lead to pronounced degradation; folding achieves more stable recovery.

• These effects are substantially more pronounced on ImageNet-C than on CIFAR-10-C, highlighting the increased difficulty of adaptation under complex tasks.

## 2 RELATED WORK

Structured Model Compression. Structured pruning removes entire computational units (e.g., convolutional filters (Li et al., 2017) or attention heads (Michel et al., 2019)), enabling efficient inference on commodity hardware (Li et al., 2017). Selection criteria range from magnitude-based heuristics (Li et al., 2017) to activation-aware scores (Sun et al., 2024), first-order Taylor approximations (Molchanov et al., 2019), and second-order Hessian-based methods (LeCun et al., 1989). Beyond pruning, model folding clusters similar channels and replaces them with shared representations (Wang et al., 2025; Saukh et al., 2026; Son et al., 2018). Existing benchmarks primarily evaluate compressed models by their accuracy on the source distribution (Liebenwein et al., 2021; Hooker et al., 2020). This protocol does not capture whether compressed models retain the capacity to adapt under distribution shift. As reported in Fig. 1, these properties can diverge, indicating that standard compression benchmarks provide an incomplete view of model quality.

Test-Time Adaptation. Test-time adaptation (TTA) adapts deployed models to distribution shifts using only unlabeled data. TENT (Wang et al., 2021) updates batch-normalization parameters via entropy minimization; SAR (Niu et al., 2023) extends this with sharpness-aware optimization and reliable sample filtering; EATA (Niu et al., 2022) incorporates an anti-forgetting regularizer to preserve important weights. For vision transformers, SPA (Niu et al., 2025) enforces consistency between clean and augmented inputs. In contrast, BP-free methods eliminate gradient computation at test time: FOA (Niu et al., 2024) uses a derivative-free covariance matrix adaptation of new prompt embeddings; PEA (Xiao et al., 2026b) corrects the embedding distortions at each intermediate layer through a covariance alignment procedure.

Loss Landscape Geometry and Mode Connectivity. Solutions found by SGD in overparameterized networks are often connected by low-loss curves in weight space (Garipov et al., 2018; Draxler et al., 2018), and become linearly connected when accounting for permutation symmetries (Entezari et al., 2022; Ainsworth et al., 2023). Frankle et al. (2020) formalize linear mode connectivity (LMC) and show that pruned subnetworks recover full accuracy only when they are stable to SGD noise. Recent work by Saukh et al. (2026) extends this perspective to model folding, showing how clustering-based compression reshapes the loss landscape geometry. While these works characterize how compression alters the loss landscape, they do not address whether adaptation can compensate for these changes.

Compression-Adaptation Interaction. Plasticity loss in deep networks has been linked to loss landscape properties even without compression (Lyle et al., 2023; Dohare et al., 2024). Compression can further amplify representational biases, e.g., toward low-frequency subgroups (Hooker et al., 2020). Liebenwein et al. (2021) suggest that overparameterization may be necessary to preserve robustness under distribution shift.

Despite these insights, the mechanistic question remains open: which structural properties must a compression method preserve to enable effective adaptation?

## 3 DIAGNOSTIC FRAMEWORK

Given a classifier $f ( x ; \theta )$ with pretrained parameters $\theta _ { 0 } ,$ trained on a source distribution ${ \mathcal P } ,$ we consider deployment under a shifted target distribution $\mathcal { Q } \neq \mathcal { P }$ . To improve performance on Q, test-time adaptation (TTA) updates the model parameters as:

$$
\theta ^ { * } = \theta _ { 0 } + \Delta , \quad \Delta \in \mathcal { U } ,\tag{1}
$$

where U denotes the set of admissible update directions $( e . g .$ ., affine normalization parameters). In contrast, we refer to Oracle as supervised adaptation over the same subspace U, using labeled target data.

In practice, models are often compressed prior to deployment. We formalize structured compression as a projection Π : $\mathbb { R } ^ { D } \to \mathbb { R } ^ { d }$ with $d \ll D$ , mapping the original parameters to a compressed representation $\phi _ { 0 } = \Pi ( \theta _ { 0 } )$ . This formulation captures two common compression strategies:

• Structured Pruning: Π acts as a coordinate selection operator, applying binary masks to channels or attention heads and retaining only a subset of parameters indexed by $S \subset \{ 1 , \ldots , D \}$

• Folding: Π performs cluster-based aggregation (Wang et al., 2025; Son et al., 2018), grouping similar channels and replacing each cluster with its centroid. This reduces dimensionality via linear combinations rather than explicit removal.

To enable a controlled comparison, we use equal per-layer compression ratios, resulting in matched parameter budgets across methods. The central question is how structured compression affects the model’s ability to adapt to Q. While compression reduces model capacity, its interaction with adaptation is governed by more subtle mechanisms, which we analyze next.

Models and Datasets. We evaluate the diagnostic framework on both convolutional and self-attention architectures. Specifically, we use ResNet-18 (He et al., 2016) checkpoints trained on CIFAR-10 (11.1M parameters) and adapted to CIFAR-10-C (Hendrycks & Dietterich, 2019), as well as ResNet-18 (11.7M parameters) and ViT-Base (Dosovitskiy et al., 2021) (86M parameters) checkpoints trained on ImageNet and adapted to ImageNet-C (Hendrycks & Dietterich, 2019). All experiments are conducted at severity level 5 across the 15 corruption types defined in CIFAR-10-C and ImageNet-C. Multi-seed and severity-3 experiments are reported in Appendix A.

Compression Methods. We analyze five structured compression strategies: magnitude-based $\ell _ { 2 }$ pruning (Li et al., 2017), Wanda (Sun et al., 2024) (combining weight magnitudes with activation norms), Taylor-based pruning (Molchanov et al., 2019), Optimal Brain Damage (OBD) (LeCun et al., 1989) using Hessian-based importance scores, and model folding via k-means channel clustering (Wang et al., 2025).

The compression ratio r denotes the uniform fraction of channels removed per layer. For ResNet-18, r corresponds to the fraction of convolutional filters (output channels) removed. For ViT-Base, r specifies the fraction of hidden units removed from the feed-forward network layers in each transformer block. We evaluate $r \in \{ 0 . 0 , 0 . 0 1 , \ldots , 0 . 9 5 \}$ for ResNet-18 (sparsity $s \in [ 0 \% , 9 4 . 9 \% ] )$ and $r \in \{ 0 . 0 , 0 . 1 , \ldots , 0 . 7 \}$ for ViT-Base (sparsity $s \in [ 0 \% , 4 5 . 8 \% ] )$ . An analysis of the memory use, latency, FLOPs and adaptation time of the compressed models is reported in Fig. 12.

For compressed ResNet-18 models, we apply REPAIR (Jordan et al., 2023) to re-estimate batch normalization statistics using four batches of training data, following Wang et al. (2025). In contrast, ViT-Base uses layer normalization (Ba et al., 2016), which computes statistics per sample and does not require post-compression re-estimation.

Adaptation Methods. We focus on backpropagation-based TTA methods, as our diagnostic framework relies on gradient-based analysis (e.g., gradient cosine similarity and objective-induced degeneracy), which requires backpropagation gradients to exist. For ResNet-18, we use Sharpness-Aware and Reliable (SAR) entropy minimization (Niu et al., 2023) with the original hyperparameters. For ViT-Base, we employ Self-Bootstrapping Adaptation (SPA) (Niu et al., 2025), which optimizes a consistency objective between clean and augmented inputs. Together, these two methods cover TTA entropy-based and consistency-based objectives; whether backpropagation-free methods (Niu et al., 2024; Xiao et al., 2026b) exhibit different sensitivity to compression is an important direction for future work.

To isolate the effect of structured compression from the adaptation objective, we introduce supervised Oracle variants that share the same optimization settings (optimizer, trainable parameters, learning rate, and number of adaptation steps) as their analyzed TTA counterparts. For ResNet-18, Oracle-SAR replaces the entropy-based objectives in both inner and outer loops with supervised cross-entropy. For ViT-Base, Oracle-SPA replaces all self-generated signals with supervised cross-entropy losses across all views. We emphasize that the Oracle is not intended as a fair comparison with TTA, but as a diagnostic upper bound on the recoverability of the adaptation subspace U.

![](images/70ebef1c2ce87334c39da3067679303132ccf683e8c0bb3173af278e825df765.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/c0e6119b0ecdb1f21b9c133e115d08bfe15d40726f956b042cb7c2d4d8431840.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/baaab0d6c8c80bd5bfcdc148ae0343d8777a5420018345d39c1b8a7cda76acf0.jpg)  
(c) ViT-Base on ImageNet-C

![](images/602ae7d49c33f39d2b5803afacbc0456e88ba45d3ac06519a9860af829fd0ee7.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/4827a9a03b1fc6348479dcf8f13d84c85d666b6d2eb7c148154f401e592b6e56.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/a45f28271ffb8d9b8633052823fd398de1a6115acee71a627a79bdb8b4670630.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 2: Test-time adaptation (TTA) degrades representations more than supervised adaptation (Oracle), with the gap depending on the compression method. We measure worst-layer CKA between dense and compressed models before (x-axis) and after adaptation (y-axis). Points above the identity line (y=x) indicate that adaptation brings representations closer to the dense model. Across all settings, both TTA and Oracle generally improve representations relative to the compressed model (most points lie above the identity line). However, TTA exhibits substantially weaker recovery than Oracle, with many points lying closer to or below the identity line, especially as compression increases. Notably, this degradation is already visible at low sparsity levels, indicating early loss of adaptable representations. The extent of this effect depends on the compression method: Wanda and Taylor preserve representations that remain partially recoverable, whereas OBD and Mag- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$ exhibit markedly weaker recovery under TTA. Folding retains higher recoverability at larger sparsity. This gap is particularly strong on ImageNet-C, where TTA fails to recover representations even at moderate sparsity. Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression (Folding, Mag-ℓ<sub>2</sub>).

## 3.1 EXPRESSIVITY AND DIVERSITY

Structured compression can reduce the diversity of internal representations and collapse their effective rank, thereby restricting the range of functions the model can realize during adaptation. We ask: To what extent can adaptation recover the representational expressivity and diversity lost due to structured compression? To answer this question, we quantify representational recovery using Centered Kernel Alignment (CKA) to measure alignment with the dense model and Activation Map Entropy (AME) to capture the diversity of layer-wise activations. We additionally test whether the representational collapse is an artifact of the similarity index, and whether model size explains the loss of adaptability.

Centered Kernel Alignment. We quantify representational similarity between dense and compressed models using Centered Kernel Alignment (CKA) (Kornblith et al., 2019), computed on post-residual hidden representations and summarized by the worst-layer CKA across all layers. This worst-layer statistic identifies the most degraded layer under compression. As reported in Tishby & Zaslavsky (2015), information lost at any intermediate layer cannot be recovered by subsequent layers; the most degraded layer acts as a representational bottleneck that may constrain the model’s capacity for adaptation. The information-bottleneck principle (Tishby & Zaslavsky, 2015) governs the mutual information between the input and the layer-l activation, not the kernel alignment that CKA measures; we use the information-bottleneck argument as an intuition for the monotone non-increase of recoverable information along the forward pass and motivate the worst-layer statistic empirically by reporting its correlation with adaptation accuracy (Fig. 7).

![](images/0935787f16827e9999ce81916dd3335b93cf9a960bd3c69b61e0a8c770fa2be0.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/ac57dcbfe761ef39620a088104c07ba64b4b7f79b709c9c548e1369bb492dc1a.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/3645445cdd433e7955adde23dab35c6be927c0b34662264549a272d9831abe43.jpg)  
(c) ViT-Base on ImageNet-C

![](images/145716f413e36b1010c62e61ca90064e66202421e1608a77b6f124c8e774ea44.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/57b60b9c405955fe74c142d0ae1ccc3da075ee2b09177bcfb9d4f2aecdb6ef14.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/cf5880cb0d4c49c0f55b79d83013d1d3904cdab53350c71331627692e7bf1a16.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 3: Test-time adaptation (TTA) only partially restores output expressivity, with recovery strongly dependent on the compression method. We measure expressivity via Activation Map Entropy (AME) and report the $\ell _ { 2 }$ distance between entropy vectors of compressed and dense models before (x-axis) and after adaptation (y-axis). Lower values indicate outputs closer to the dense model. Points below the identity line (y=x) correspond to successful recovery of output expressivity, the lower the better. Across all settings, both TTA and supervised adaptation (Oracle) reduce the AME distance, indicating partial restoration of output diversity. However, Oracle consistently achieves stronger and more stable recovery, while TTA exhibits weaker and less consistent improvements, especially at higher compression levels and on ImageNet-C. The recovery behavior varies across compression methods. Wanda and Taylor enable moderate recovery under TTA, whereas OBD and Mag- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$ show unstable recovery, often failing to restore output expressivity. In contrast, Folding yields more consistent recovery patterns, but still remains below Oracle. Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression (Folding, Mag-ℓ ).

On ResNet-18 (Fig. 2b, 2e), we observe that TTA exhibits representational collapse already at very low sparsity (as early as 2% on ImageNet-C), while Oracle consistently recovers representations closer to the dense model. This gap persists and widens with increasing compression, indicating that TTA fails to restore the most degraded layers. This behavior is consistent with a self-reinforcing degradation under entropy minimization (Niu et al., 2023): since gradients are derived from the model’s predictions damaged by compression, prediction errors may influence the TTA adaptation signal, reinforcing representational distortions rather than correcting them. In contrast, Oracle relies on ground-truth labels, which may provide a more robust gradient signal and explain its consistently stronger representation recovery. We formalize this hypothesis in the gradient degeneracy analysis (Sec. 3.2).

On ViT-Base (Fig. 2f), the impact of the compression method becomes more pronounced. At 39% sparsity, Mag- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$ leads to near-complete representational collapse (CKA ≈ 0), whereas Folding recovers substantially higher post-adaptation alignment with the dense model. This difference may arise from how the two methods transform the activation Gram matrices analyzed by CKA (Kornblith et al., 2019): Mag-ℓ removes entire units and their contributions, while Folding aggregates correlated units via clustering (Wang et al., 2025), better preserving the variance structure of the representations. This observation is consistent with Hu et al. (2024), who show that pruning in the original feature dimension disrupts principal components that span multiple dimensions, motivating projection-based approaches.

Activation Map Entropy. While CKA captures representational expressivity via alignment with the dense model (Kornblith et al., 2019), it does not characterize the spectral diversity of layer-wise activations. To complement this, we define Activation Map Entropy (AME) building on the matrix-based entropy framework of Skean et al. (2025) as a measure of representational diversity. For each layer l, AME is defined as the Shannon entropy of the normalized eigenvalues of the activation Gram matrix $G _ { l } \in \mathbb { R } ^ { C _ { l } \times C _ { l } }$ , quantifying the effective rank of the representation (Roy & Vetterli, 2007).

Structured compression reduces $G _ { l }$ from $C _ { l } \times C _ { l }$ to $C _ { l } ^ { \prime } \times C _ { l } ^ { \prime } ;$ , imposing an upper bound on entropy at log $C _ { l } ^ { \prime } .$ . Since TTA methods (Niu et al., 2023; Wang et al., 2021) update only normalization parameters, they cannot increase the rank of $G _ { l }$ beyond $C _ { l } ^ { \prime }$ . Compression therefore imposes an upper bound on the recoverable diversity of representations. Consistent with prior observations that pruning reduces activation entropy (Liao et al., 2024), we expect this reduction to propagate to the eigenspectrum of $G _ { l }$

Empirically, on CIFAR-10-C (Fig. 3a, 3d), both Oracle and TTA recover AME at low sparsity (15%), but diverge at higher sparsity (35%, 55%), where points form horizontal saturation bands. These bands reflect the entropy ceiling imposed by reduced dimensionality, limiting further recovery regardless of the adaptation method. On ImageNet-C with ResNet-18 (Fig. 3b, 3e), this limitation becomes more pronounced: at 35% sparsity, most points lie below the identity line, indicating partial recovery, with Oracle consistently achieving tighter and lower-distance clusters than TTA. This suggests that while both methods operate under the same dimensionality constraint, Oracle more effectively redistributes the available eigenvalue mass toward the dense reference.

On ViT-Base (Fig. 3c, 3f), AME distances are substantially smaller and less sensitive to compression. This can be attributed to FFN-only compression, which preserves the post-residual dimensionality fixed at 768 (Dosovitskiy et al., 2021), keeping the size of $G _ { l }$ invariant. As a result, the entropy ceiling remains largely unchanged, and residual connections further mitigate the impact of compression on the activation spectrum. Nevertheless, compression methods still differ: Folding achieves lower post-adaptation AME distances than Mag-ℓ and consistently outperforms TTA, aligning with the trends observed in the CKA analysis (Fig. 2f).

Metric Similarity Analysis. To further support our analysis, we adopt the spectral-ratio pseudo-distance $d _ { \mathrm { S R } }$ (Cayco-Gajic & Pellegrino, 2026), which compares the intrinsic geometry of neural representations under the manifold hypothesis, whereas CKA compares their extrinsic geometry. Following Cayco-Gajic & Pellegrino (2026), we apply $d _ { \mathrm { S R } }$ to the Riemannian pullback metrics of the dense and the compressed model, which measure how strongly each model’s output responds to small perturbations of the input, rather than to the activation Gram matrices $G _ { l }$ analyzed by CKA and AME. d ranks post-compression representation consistently with worst-layer CKA (Spearman $\rho = - 0 . 9 2 5$ on ResNet-18/ImageNet-C and −0.665 on ViT-Base/ImageNet-C; the negative sign is expected when a distance metric is compared with a similarity; see Fig. 17), indicating that the representational collapse observed on ImageNet-C is not an artifact of the similarity index (Davari et al., 2023).

Disentangling compression from capacity. We compare compressed against smaller-dense models of equal architecture trained from scratch: at increasing sparsity, TTA and its Oracle adaptation show an earlier collapse compared to the smaller-dense models (Fig. 19), indicating that the loss of adaptability is induced by the structured compression rather than by the reduced model size.

Structured compression imposes intrinsic limits on both representational alignment and diversity that adaptation cannot fully overcome. While Oracle approaches these limits, TTA remains constrained by degraded signals and reduced capacity, resulting in systematically weaker recovery.

## 3.2 SUBSPACE COMPATIBILITY

Structured compression reduces the set of trainable parameters available during adaptation and may also alter how well the remaining update directions align with the target-domain loss. We ask: Does compression reduce thefraction of useful adaptation signal accessible within the admissible update subspace, and can this help explain the growing gap between unsupervised test-time adaptation and supervised target-domain adaptation?

Gradient cosine similarity. Structured compression reduces the dimensionality of the admissible update $\Delta \in { \mathcal { U } }$ (Eq. 1) by eliminating adaptable parameters corresponding to pruned (Li et al., 2017) or folded (Wang et al., 2025) units. Since pruned deep networks suffer disproportionately under distribution shift (Liebenwein et al., 2021), adaptability may depend not only on the preserved dim $( \mathcal { U } )$ but also on whether the adaptation gradient directions preserve their alignment with the gradient of the supervised loss on the target domain. We quantify this alignment via the cosine similarity between the accumulated TTA and Oracle gradients, restricted to U:

$$
\cos ( \mathbf { g } _ { \mathrm { T T A } } , \mathbf { g } _ { \mathrm { O r a c l e } } ) = { \frac { \mathbf { g } _ { \mathrm { T T A } } \cdot \mathbf { g } _ { \mathrm { O r a c l e } } } { \left\| \mathbf { g } _ { \mathrm { T T A } } \right\| \cdot \left\| \mathbf { g } _ { \mathrm { O r a c l e } } \right\| } } ,\tag{2}
$$

where $\begin{array} { r } { \mathbf { g } _ { \mathrm { T T A } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla _ { \phi } \mathcal { L } _ { \mathrm { T T A } } ( x _ { i } ; \phi _ { 0 } ) } \end{array}$ and $\begin{array} { r } { \mathbf { g } _ { \mathrm { 0 r a c l e } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla _ { \phi } \mathcal { L } _ { \mathrm { C E } } ( x _ { i } , y _ { i } ; \phi _ { 0 } ) } \end{array}$ are the adaptation set gradients averaged over all $N$ corruption samples. We compute a single cosine similarity metric on these adaptation set gradients, rather than averaging per-sample (Chen et al., 2025) or per-task (Yu et al., 2020) similarities, to assess whether the TTA loss provides a meaningful proxy of the supervised optimization direction at the pre-adaptation parameters $\phi _ { 0 }$ . The metric is undefined when either gradient norm vanishes, which implies an adaptation collapse regime, a condition we analyze theoretically below.

![](images/577a5e90dac6072173b8c0f52eb8f077c40cbb5e935c394a9d19b60211da28e5.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/abdbab12bc53c23ff8ab1175e4e9bd213de9899a2d54bb9823f3fffc7527dd1a.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/882716d9a6b1a98f7cb366e66aeb75eff6b7a3e333c3da3880d1da100e707791.jpg)  
(c) ViT-Base on ImageNet-C

![](images/be1a0763b49cdfa8469fdf2626b4d47a5794f7b0374c548c89e24fe07d7bd338.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/6cd591310757a6ce6c6dc7ef7ce29aa201c008ee1c8c1e6da30f946e86afa364.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/1825d46d4bad1b245e57de4becc82607400901051ccbc9cf29594b5769df03a0.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 4: Structured compression increasingly misaligns the TTA gradient from the supervised gradient direction (Oracle). We report cosine similarity $\cos ( \mathbf { g } _ { \mathrm { T T A } } , \mathbf { g } _ { \mathrm { O r a c l e } } )$ (Eq. 2, top panels) and gradient $\ell _ { 2 }$ norms (bottom panels), at the post-compression, pre-adaptation parameters ϕ<sub>0</sub> averaged across all corruptions. At low sparsity, the TTA and Oracle gradients are positively aligned, suggesting that the TTA objective initially provides a reliable surrogate for the supervised adaptation direction. As sparsity increases, this alignment deteriorates and can reverse sign, indicating that the TTA updates conflict with the supervised ones, a more severe failure mode than the loss of adaptation signal $( \cos \approx 0 )$ . TTA gradient norms vanish (degeneracy) or remain large while misaligned (active divergence), both leading to adaptation collapse reflecting the TTA accuracy collapse observed in Fig. 1; lines terminate when the metric becomes undefined at high sparsities. The SPA objective on ViT-Base preserves weakly positive alignment across all tested sparsities, consistent with the smaller Oracle-TTA gap in Fig. 1. Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression (Folding, Mag- $\boldsymbol { \cdot } ( \boldsymbol { \mathbf { \rho } } _ { 2 } )$ .

Empirically, Fig. 4 reveals three distinct regimes of gradient alignment under structured compression. On CIFAR-10-C (Fig. 4a, 4d), the TTA and Oracle gradients are positively aligned at low sparsity, indicating that the TTA updates initially point in a direction consistent with the supervised optimization objective. As sparsity increases, however, the cosine similarity crosses zero and becomes strongly negative, reaching its minimum value between 70% and 80% sparsity across all compression criteria. This sign reversal suggests that the TTA update actively drives the model away from the supervised optimum, constituting a more severe failure mode than the simple loss of an adaptation signal, which would correspond to cos $\approx 0 .$ . This phenomenon is compounded by the progressively widening discrepancy in gradient norms. The emergence of this misalignment coincides with the sparsity level at which TTA accuracy begins to deteriorate sharply (Fig. 1), offering a gradient lens explanation for the observed adaptation degradation. At extreme sparsities $( \geq 7 5 \% )$ , the metric becomes undefined when the TTA gradient norm vanishes for all corruptions, e.g., Mag-ℓ<sub>2</sub> beyond 85%. These results support our entropy gradient analysis (Eq. 4, 5): at equal sparsity, compressed models driven to near-uniform predictions produce weak adaptation signals, in contrast to the supervised signals which do not vanish.

On ImageNet-C with ResNet-18 (Fig. 4b, 4e), gradient alignment is negative already at 1%, suggesting that SAR’s loss objective may be a poor surrogate of the supervised loss on this harder distribution shift, even before significant model capacity is removed by compression. The gradient norms in the bottom panel reveal a failure mode distinct from CIFAR-10-C: $\| \mathbf { g } _ { \mathrm { T T A } } \|$ systematically exceeds $\left. \mathbf { g } _ { \mathrm { { O r a c l e } } } \right.$ across all sparsities. This result, combined with the negative cosine similarity, indicates an active divergence regime: the SAR-based objective produces large updates in a direction that conflicts with the supervised one. This adaptation regime, associated with structured compression, may actively damage the model consistent with the severe Oracle-TTA gap previously observed (Fig. 1). A causal ablation of this mechanism (Fig. 18) indicates that the confident-but-wrong samples admitted by SAR’s reliability filter drive the misalignment: excluding them with a correctness filter computed from the ground-truth labels restores the alignment to ≈ +0.9 across all compression criteria and architectures.

On ViT-Base with SPA (Fig. 4c, 4f), for data-dependent criteria, cosine similarity remains weakly positive across all sparsities, suggesting that the consistency-based SPA objective maintains partial alignment with the supervised direction even at 46% sparsity. No sign reversal occurs, consistent with the smaller Oracle-TTA accuracy gap observed in Fig. 1. Data-free compression exhibits more unstable adaptation behavior: Mag- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ produces negative mean alignment at several sparsity levels, and both Folding and Mag- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ exhibit high variance across corruptions. Gradient alignment remains fragile and corruption-dependent under data-free compression, where no calibration data guides unit selection. Decomposing these curves by corruption family (Figs. 13 –16) shows similar gradient alignment regimes within each family.

![](images/42765c2f263459588a1e161b4c5573133783b22c4924f58414e6d1395be0ddc2.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/19f71f2de502d12942bb54d826ad786d63e8370a0dccd546d528329741151c49.jpg)

![](images/0f4dae3d6504f0b0b35dc6f27b4f803cfa6e57cee8cf382e8dd023f3dca276fb.jpg)  
(b) ResNet-18 on ImageNet-C  
(c) ViT-Base on ImageNet-C

![](images/2316bda2e62eace9d9e594cfd2f441b05600bebea2bf6bb21cbb11b67fc76e23.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/54519773e156ccd8055535d5fe228374ab38f8b27f82c92be7c204b3129038ef.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/f502840ce64e049ec21cca9c7d1dc23bf1f869a7bf3633f99e39ef0f060fe03a.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 5: Prediction entropy $H ( \mathbf { p } )$ increases with sparsity in most settings, empirically supporting the gradient degeneracy mechanism of Eq. 3 and Eq. 5. Each dot represents the mean prediction entropy $\begin{array} { r } { H ( \mathbf { p } ) = - \sum _ { c } p _ { c } } \end{array}$ ln $p _ { c }$ of a single corruption type at severity $5 ;$ solid lines and shaded bands show the mean and ±σ across all corruptions. The dashed line marks ${ \dot { H } } _ { m a x } = \ln C$ (uniform distribution). Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression $( \mathbf { M a g - } \ell _ { 2 } ,$ , Folding).

Objective-Induced Gradient Degeneracy. Even if the adaptation subspace is aligned with the target-domain loss, the TTA objective may fail to produce informative gradients under structured compression. We show that this degeneracy arises in both entropy- and consistency-based TTA objectives.

Entropy minimization. TTA methods such as TENT (Wang et al., 2021) and SAR (Niu et al., 2023) optimize:

$$
\mathcal { L } _ { \mathrm { e n t } } ( p ) = - \sum _ { c = 1 } ^ { C } p _ { c } \log p _ { c } ,\tag{3}
$$

where $p = ( p _ { 1 } , \dotsc , p _ { C } )$ is the predictive distribution. Using $\textstyle \sum _ { c } p _ { c } = 1$ and thus $\begin{array} { r } { \sum _ { c } \nabla _ { \phi } p _ { c } = 0 } \end{array}$ , differentiating gives:

$$
\nabla _ { \phi } \mathcal { L } _ { \mathrm { e n t } } = - \sum _ { c = 1 } ^ { C } ( \log p _ { c } + 1 ) \nabla _ { \phi } p _ { c } = - \sum _ { c = 1 } ^ { C } \log ( C p _ { c } ) \nabla _ { \phi } p _ { c } .\tag{4}
$$

When $p = ( 1 / C , \dots , 1 / C )$ , all coefficients $\log ( C p _ { c } )$ vanish and $\nabla _ { \phi } \mathcal { L } _ { \mathrm { e n t } } = \mathbf { 0 }$ . More generally:

$$
\| \nabla _ { \phi } \mathcal { L } _ { \mathrm { e n t } } \| _ { 2 } \leq \left( \sum _ { c = 1 } ^ { C } | \log ( C p _ { c } ) | \right) \operatorname* { m a x } _ { c } \| \nabla _ { \phi } p _ { c } \| _ { 2 } .\tag{5}
$$

If compression pushes predictions toward a near-uniform regime, the entropy signal weakens (gradient degeneracy), as observed on CIFAR-10-C (Fig. 4a, 4d). Conversely, when compression preserves confident but incorrect predictions, $| \log ( C p _ { c } ) |$ remains large, producing misaligned high-norm gradients (active divergence), as observed on ImageNet-C (Fig. 4b, 4e). Prediction entropy measurements indicate that $H ( \mathbf { p } )$ increasingly approaches $H _ { m a x }$ with higher sparsity in most settings (Fig. 5), further supporting the gradient degeneracy mechanism under compression.

KL-consistency. SPA (Niu et al., 2025) minimizes the KL divergence between a detached clean prediction $q =$ softmax $( f ( x ; \phi ) )$ and an augmented-view prediction $p _ { \mathrm { a u g } } = \operatorname { s o f t m a x } ( g ( x _ { \mathrm { a u g } } ; \phi ) )$ :

$$
\mathcal { L } _ { \mathrm { K L } } = \sum _ { c = 1 } ^ { C } q _ { c } \log \frac { q _ { c } } { p _ { \mathrm { a u g } , c } } .\tag{6}
$$

Since q is detached, $\begin{array} { r } { \nabla _ { \phi } \mathcal { L } _ { \mathrm { K L } } = - \sum _ { c } ( q _ { c } / p _ { \mathrm { a u g } , c } ) \nabla _ { \phi } p _ { \mathrm { a u g } , c } . } \end{array}$ . When $q \approx p _ { \mathrm { a u g } } ,$ each ratio $q _ { c } / p _ { \mathrm { a u g , } c } \approx 1$ , and together with $\begin{array} { r } { \sum _ { c } \nabla _ { \phi } p _ { \mathrm { a u g } , c } = \mathbf { 0 } } \end{array}$ this implies that the gradient vanishes. Compression triggers this by driving both views toward uniform predictions.

Supervised cross-entropy (Oracle). By contrast, the supervised gradient for a labeled example $( x , y )$

$$
\nabla _ { \phi } \mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { p _ { y } } \nabla _ { \phi } p _ { y } ,\tag{7}
$$

concentrates on a single direction scaled by $1 / p _ { y }$ , which grows as confidence drops. Even when compression degrades predictions, supervised updates thus remain informative where entropy or consistency signals vanish. Together, Eqs. 4–7 reveal two complementary mechanisms behind the post-compression Oracle-TTA gap: (i) compression removes or misaligns parameter directions useful for target-domain adaptation, and (ii) unsupervised TTA objectives simultaneously lose their gradient signal as predictions approach an uninformative regime.

## 4 DISCUSSION, OUTLOOK AND LIMITATIONS

This paper analyzes the interaction between structured compression and test-time adaptation, a combination that is common in edge deployment but has received limited attention. Our study reveals that the standard compress-and-deploy pipeline can induce silent plasticity loss: compressed models that preserve source-domain accuracy yet progressively lose the ability to adapt at test time via unsupervised objectives. This finding extends prior observations that compression harms robustness under distribution shift (Liebenwein et al., 2021) and disproportionately affects underrepresented subgroups (Hooker et al., 2020), by showing that degradation extends to adaptability. Our theoretical analysis establishes that the gradients of both entropy-based and consistency-based TTA objectives can degenerate under compression, while the supervised cross-entropy gradient remains more informative at the same sparsity (Eq. 4, Eq. 7).

Practical guidelines. Our results suggest two actionable considerations. First, the compression criterion should be selected for adaptability, not source accuracy alone; data-dependent criteria (Wanda, Taylor) preserve higher representational recoverability (Fig. 2) and gradient alignment (Fig. 4) than Mag-ℓ<sub>2</sub> and OBD, while Folding retains more adaptability among data-free methods. Second, the compression budget and TTA objective should be co-designed: on CIFAR-10-C, gradient alignment reverses sign between 45% and 65% sparsity for the pruning criteria (near 75% for Folding); on ImageNet-C, entropy-based objectives report active divergence already at minimal sparsity, whereas consistency-based objectives maintain higher alignment across all analyzed sparsities.

Outlook. There are several directions to extend this work. First, although our empirical evaluation covers both backpropagation-based and backpropagation-free TTA approaches (FOA (Niu et al., 2024), PEA (Xiao et al., 2026b)), our diagnostics are mainly gradient-based and do not yet cover backpropagation-free adaptation methods; we leave this as an open direction. The corresponding accuracy curves are reported in Fig. 9: the benefit of backpropagationfree adaptation likewise collapses toward the no-adaptation baseline as compression increases. Second, designing compression-aware TTA objectives that preserve adaptability under compression is left for future work. Third, extending this analysis to unstructured pruning, quantization, and their combinations would provide a more complete picture of the compression-adaptation interaction.

Limitations. Our study considers two architectures (ResNet-18 and ViT-Base) and two (one backpropagation-based, one backpropagation-free) TTA methods per architecture, covering both entropy- and consistency-based objectives; the generalization to larger models $( e . g .$ , large language models) remains for future work. While the experiments are primarily conducted at maximum corruption severity $( i . e . , 5 )$ , we additionally report results at intermediate severity 3 (Fig. 11) and across multiple seeds (Fig. 8). We hope this work sparks interest in designing structured compression strategies that explicitly preserve adaptability, narrowing the gap between efficient deployment and test-time adaptation.

## ACKNOWLEDGMENTS

This research was funded in part by the Austrian Science Fund (FWF) within the DENISE doctoral school (grant DOI: 10.55776/DFH5) and the FFG COMET K1 Center “Pro<sup>2</sup>Future II” (Cognitive and Sustainable Products and Production Systems of the Future), Contract No. 911655. The results presented in this paper were computed using the computational resources of the HLR Zentralen Informatikdienstes of Graz University of Technology and the Austrian Scientific Computing (ASC) infrastructure.

## REFERENCES

Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In The Eleventh International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=CQsmMYmlP5T.

Saleh Ashkboos, Maximilian Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns. In International Conference on Learning Representations, volume 2024, pp. 11682–11701, 2024.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

N. Alex Cayco-Gajic and Arthur Pellegrino. Geometry-aware similarity metrics for neural representations on riemannian and statistical manifolds. arXiv preprint arXiv:2603.28764, 2026.

Ziyang Chen, Yiwen Ye, Yongsheng Pan, and Yong Xia. Gradient alignment improves test-time adaptation for medical image segmentation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pp. 2429–2437, 2025.

Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Sarangapani. A comprehensive survey on model compression and acceleration: T. choudhary et al. Artificial Intelligence Review, 53(7):5113–5155, 2020.

Francesco Corti, Rahim Entezari, Sara Hooker, Davide Bacciu, and Olga Saukh. Studying the impact of magnitude pruning on contrastive learning methods. arXiv preprint arXiv:2207.00200, 2022.

MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. Reliability of CKA as a similarity measure in deep learning. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=8HRvyxc606.

Zeshuai Deng, Guohao Chen, Shuaicheng Niu, Hui Luo, Shuhai Zhang, Yifan Yang, Renjie Chen, Wei Luo, and Mingkui Tan. Test-time model adaptation for quantized neural networks. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 7258–7267, 2025.

Shibhansh Dohare, J Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A Rupam Mahmood, and Richard S Sutton. Loss of plasticity in deep continual learning. Nature, 632(8026):768–774, 2024.

Jiaheng Dong, Hong Jia, Soumyajit Chatterjee, Abhirup Ghosh, James Bailey, and Ting Dang. E-bats: Efficient backpropagation-free test-time adaptation for speech foundation models. Advances in neural information processing systems, 38:141099–141126, 2026.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht. Essentially no barriers in neural network energy landscape. In International conference on machine learning, pp. 1309–1318. PMLR, 2018.

Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=dNigytemkL.

Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16091–16101. IEEE, 2023.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International conference on machine learning, pp. 3259–3269. PMLR, 2020.

Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.

Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. Advances in neural information processing systems, 30, 2017.

Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, and Sung-Ju Lee. Sotta: Robust test-time adaptation on noisy data streams. Advances in Neural Information Processing Systems, 36:14070–14093, 2023.

Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id=HJz6tiCqYm.

Sara Hooker, Nyalleng Moorosi, Gregory Clark, Samy Bengio, and Emily Denton. Characterising bias in compressed models. arXiv preprint arXiv:2010.03058, 2020.

Yuxuan Hu, Jing Zhang, Zhe Zhao, Chen Zhao, Xiaodong Chen, Cuiping Li, and Hong Chen. Sp3: Enhancing structured pruning via pca projection. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 3150–3170, 2024.

Hong Jia, Young D Kwon, Alessio Orsino, Ting Dang, Domenico Talia, and Cecilia Mascolo. Tinytta: Efficient test-time adaptation via early-exit ensembles on edge devices. Advances in Neural Information Processing Systems, 37:43274–43299, 2024.

Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur. REPAIR: REnormalizing permuted activations for interpolation repair. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=gU5sJ6ZggcX.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519–3529. PMlR, 2019.

Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems, 2, 1989.

Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id= rJqFGTslg.

Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In International conference on machine learning, pp. 6028–6039. PMLR, 2020.

Jian Liang, Ran He, and Tieniu Tan. A comprehensive survey on test-time adaptation under distribution shifts. International Journal ofComputer Vision, 133(1):31–64, 2025.

Zhu Liao, Victor Quetu, Van-Tam Nguyen, and Enzo Tartaglione. Nepenthe: Entropy-based pruning as a neural network´ depth’s reducer. arXiv e-prints, pp. arXiv–2404, 2024.

Lucas Liebenwein, Cenk Baykal, Brandon Carter, David Gifford, and Daniela Rus. Lost in pruning: The effects of pruning neural networks beyond test accuracy. Proceedings ofMachine Learning and Systems, 3:93–138, 2021.

Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chen Wang, Chuang Gan, and Song Han. On-device training under 256kb memory. Advances in Neural Information Processing Systems, 35:22941–22954, 2022.

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, and Will Dabney. Understanding plasticity in neural networks. In International conference on machine learning, pp. 23190–23211. PMLR, 2023.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. Advances in neural information processing systems, 36:21702–21720, 2023.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? Advances in neural information processing systems, 32, 2019.

M Jehanzeb Mirza, Jakub Micorek, Horst Possegger, and Horst Bischof. The norm must go on: Dynamic unsupervised domain adaptation by normalization. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14745–14755. IEEE, 2022.

Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 11256–11264. IEEE, 2019.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In International conference on machine learning, pp. 16888–16905. PMLR, 2022.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=g2YraF75Tj.

Shuaicheng Niu, Chunyan Miao, Guohao Chen, Pengcheng Wu, and Peilin Zhao. Test-time model adaptation with only forward passes. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=qz1Vx1v9iK.

Shuaicheng Niu, Guohao Chen, Peilin Zhao, Tianyi Wang, Pengcheng Wu, and Zhiqi Shen. Self-bootstrapping for versatile test-time adaptation. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=Li4rieeClO.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pp. 606–610. IEEE, 2007.

Olga Saukh, Dong Wang, Haris Siki <sup>ˇ</sup> c, Yun Cheng, and Lothar Thiele. Cut less, fold more: Model compression through´ the lens of projection geometry. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=JV9CEtKLQF.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=WGXb7UdvTX.

Georg Slamanig, Francesco Corti, and Olga Saukh. From llms to edge: Parameter-efficient fine-tuning on edge devices, 2025. URL https://arxiv.org/abs/2507.23536.

Sanghyun Son, Seungjun Nah, and Kyoung Mu Lee. Clustering convolutional kernels to compress deep neural networks. In European Conference on Computer Vision, pp. 225–240. Springer, 2018.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=PxoFut3dWW.

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In International conference on machine learning, pp. 9229–9248. PMLR, 2020.

Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pp. 1–5. Ieee, 2015.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, 2021. URL https:// openreview.net/forum?id=uXl3bZLkr3c.

Dong Wang, Haris Siki<sup>ˇ</sup> c, Lothar Thiele, and Olga Saukh. Forget the data and fine-tuning! just fold the network´ to compress. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=W2Wkp9MQsF.

Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7191–7201. IEEE, 2022.

Junrui Xiao, Zhikai Li, Lianwei Yang, Yiduo Mei, and Qingyi Gu. Ttaq: Towards stable post-training quantization in continuous domain adaptation. arXiv preprint arXiv:2412.09899, 2024.

MA Xiao, Young D. Kwon, and Dong Ma. Efficient Test-Time Adaptation via Decoupled BN Update For Edge Devices. In Third Workshop on Test-Time Updates (Main Track), 2026a. URL https://openreview.net/forum? id=35oSXGUjDH.

MA Xiao, Young D. Kwon, Pan Zhou, and Dong Ma. Architecture-agnostic test-time adaptation via backprop-free embedding alignment. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=7kLNGaAHaw.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 3fe78a8acf5fda99de95303940a2420c-Paper.pdf.

## A APPENDIX

![](images/0e5147273e0655614c7684eed035f6453823aad8f43f19cde69fe9b741d3c4e0.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/951a490ef92fae3780a09207c019764b99ab609d15f72997194b363e1d7304f5.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/32d4003d8f1cdc24c047f5a35c4f5353ecccf7aff99b6286312a19b3814466ab.jpg)  
(c) ViT-Base on ImageNet-C

![](images/4186043cbea02d68898c0a2bf7a46370b6fc18b9d545491bab86c4b889c985b9.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/56eb3ed152807fe51d2ecfe2b21a4ba5c900b5dfcfaca2fd5708a10ff987a8ce.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/83e236308e552ae60dbe2e06496b7b425d36577253be2d46d9790309404e076b.jpg)  
(f) ViT-Base on ImageNet-C

Figure 6: The per-corruption adaptation gap $\Delta _ { \mathrm { g a p } } = ( \mathbf { O r a c l e - T T A } ) _ { \mathbf { c o m p r e s s e d } }$ generally widens with increasing compression, providing empirical support for silentplasticity loss. Each dot represents the adaptation gap for a single corruption type at a given sparsity level; solid lines and shaded bands show the mean and ±σ across all corruptions. Top row (a–c): data-dependent pruning criteria (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression criteria (Folding, Mag-ℓ<sub>2</sub>).  
![](images/a5ab2ea8f34cff0894d40e4719971b50f7356923d9e4fe23c088651122b523cb.jpg)  
(a) Worst-layer CKA (severity 3).

![](images/105bb41d97a5f84000db257f06c88881668b0e0170a6c7c87bf5d11c427e3bf4.jpg)  
(b) Worst-layer CKA (severity 5).  
Figure 7: Before adaptation, the worst-layer CKA of the compressed model is positively correlated with its postadaptation accuracy, strongly on ViT-Base/ImageNet-C and only weakly on ResNet-18. Each bar reports the Spearman correlation ρ between the pre-TTA worst-layer CKA at ϕ and the post-adaptation accuracy of SAR ResNet-18 or SPA ViT-Base over all corruptions. Black error bars are 95% confidence intervals. Left to right: severity 3, severity 5 corruptions level.

Fig. 6 decomposes the averaged Oracle–TTA adaptation gap observed in Fig. 1 into individual corruptions. In Fig. 20 and Fig. 21, marker shape encodes sparsity, color encodes post-adaptation accuracy; thick black edge denotes Oracle, gray edge denotes TTA. Points above the identity line indicate representational recovery after adaptation in Fig. 20, in Fig. 21 points below the identity line indicate entropy-distance adaptation recovery. Across all figures, Top row (a–c): data-dependent pruning criteria (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression criteria (Folding, Mag-ℓ<sub>2</sub>).

Multi-seed evaluation. We train three checkpoints per (architecture, dataset, pre-training paradigm) with different seeds. Fig. 11 reports the multi-seed analysis of CKA for severity 3 corruptions, supporting our previous severity 5

![](images/e641ed78cd1038f0853bf268342a422c125b3f5b652be595ac260c399450e0ec.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/c4669b0bdc383e21f463f6f1ce88a50c9faa9e1b426da0cc85d96eb4ad3d8091.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/a0a46d64fca19c8840450d6b82a76548ff48e6ec10a60a75dec14f6d30746991.jpg)  
(c) ViT-Base on ImageNet-C

![](images/77c6f3c061b12aa30f0112d2648d5d3240807d32c55129dac36ecc5569fc68ab.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/2719ce354279d2dc223b548ead1dcf32a7f362747352e07c04fc10386a6e2faf.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/e8fa93993aa79ad5c157efbb5595b7eb0e5abbe722253cd0491375d0fb8a7282.jpg)  
(f) ViT-Base on ImageNet-C

Figure 8: Multi-seed gap between test-time adaptation (SAR) and the matched supervised baseline (Oracle-SAR), with SAR applied to both architectures. SAR and Oracle-SAR share the same optimization settings and differ only in the loss definitions. Shaded regions indicate ±σ across three random seeds of the per-seed corruption-averaged accuracy. Same layout as Fig. 1.

![](images/461135a4f6eab96da21ac7e7ab2d2763c1e587529959e5799302ece75a105727.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/a28e603b16edea6822f873e7007558fb64ba91512fe946f2a796aa186b3fe678.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/6f2ffc969487529dcf692fd08a74b1f4c36c26c3e827fb98790e930ada003678.jpg)  
(c) ViT-Base on ImageNet-C

![](images/990aa85b5ffe17ab15ad7d123bbc7135e334d22c9f75c8790526e4488fcbcb33.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/6ab87514a79b02d2238085a70313fd60f8bbc4cc0d105b8079e13aa25d4ba822.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/233394c284734a6258825b83415324a15b6149e6c323f5af019f39c5ea0690e7.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 9: Adaptation accuracy curves under backpropagation-free TTA (PEA and FOA) methods. PEA (Xiao et al., 2026b) and FOA (Niu et al., 2024) post-adaptation accuracy averaged across all corruptions with severity 5. PEA is applied to both ResNet-18 and ViT-Base; FOA is applied to ViT-Base only since it relies on learnable prompt embeddings, not available for the ResNet-18 models analyzed in this work. Same layout as Fig. 1: Top row (a–c): data-dependent pruning (Wanda, Taylor, OBD). Bottom row (d–f): data-free compression (Folding, Mag-ℓ ).

findings. Shaded bands of Fig. 8 are the across-seed standard deviation of the per-seed corruption-averaged accuracy. These checkpoints follow a different pre-training recipe than the single checkpoint of Fig. 1, shifting the absolute post-adaptation accuracy of Fig. 8 relative to Fig. 1. Nevertheless, the compression-induced Oracle-TTA gap and

![](images/c96429cb277cd94bba554103c04a8db1558114998711d3fb0bd30a54807a267f.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/83ba395c2ec9f5a2069cdccde6ca85aca3aee3b09ff40500988c3b9a6a57fcc3.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/da9b508f961f099d02fdf482c35f336d10abd06af9d0c2b79cc32a8cc8db9a35.jpg)  
(c) ViT-Base on ImageNet-C

![](images/398cb96ec970b9570e1986f619d58f02af1dc7c41691e666418ca82e258e598f.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/ea1ad16c3ee307c907595c122bb2fa32f36b0aaeb6ec3ccea97142d49d1fb4ba.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/6649d6376e2af28a57d7cc304063da24200b5e286cc2fcf0721a3299b857e755.jpg)  
(f) ViT-Base on ImageNet-C

Figure 10: Compression-induced Oracle–TTA gap ∆G(r) = [Oracle − TTA](r) − [Oracle − TTA](r = 0.0). Solid lines are corruption-averaged means with ±σ bands across all corruptions; a positive value indicates that compression amplifies the Oracle–TTA gap beyond the dense baseline. Same layout as Fig. 1.

![](images/b3580d47327c54feab16e8af90d975c690c393727aa5338d0b6fad989e302c5e.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/ae51872f68fa56455883c7cf18ba4e3306cdc43e5fa2827110fcb7d6d0f239f8.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/299218fcae445d698b36b2119c7fe65743fa0d9514fe88c334c533e19f5abce1.jpg)  
(c) ViT-Base on ImageNet-C

![](images/ab01c804ede2d23cf20dc35134ad16a00d102fa1bc82635c94605570aab90654.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/65157e08a73c4c7046769a08f9265f7f2cc10e7ba8999306e83f6a4c1fd6fde7.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/4b857739053991e57c6558c15f2468231558af8f0afbd71303745a500352bf08.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 11: Multi-seed worst-layer CKA, dense-prune-TTA, severity 3. Same layout as Fig. 2.

its widening trend persist under both recipes, indicating that the effect reflects structured compression rather than a particular pre-training run.

![](images/77198257b9cc953de2684cb1b4fad227e40463ac8281a527864a0db570cd812f.jpg)  
(a) RN-18 CIFAR: Latency

![](images/b9db966bfaa582fdb8dab922beab3eacd199b2539af3b6d380e43ea0e7ae4e75.jpg)  
(b) RN-18 CIFAR: Adapt. time

![](images/505e4657aa802d2486235e05b68ea561d4463863a715d52a425eb7f982ad94d1.jpg)  
(c) RN-18 CIFAR: FLOPS

![](images/96f5eb07e969300f5d40cf5c21abf17b1a6933391b4d3295ca6f99e8fec526d4.jpg)  
(d) RN-18 CIFAR: Memory

![](images/33718e96d09c135e9c3d7c5440314b103b752ae7a4dbff3b17c22a40a16ebdce.jpg)  
(e) RN-18 ImageNet: Latency

![](images/c4a7ee2e282f15aa524c7dc92172c0b1ed995fc45a5f417575b0f47adc73ac6c.jpg)  
(f) RN-18 ImageNet: Adapt. time

![](images/827c0b089fd1f8d0df5d9b21d5962f3fb27a06c421dd301add0eca6f4baf399e.jpg)  
(g) RN-18 ImageNet: FLOPS

![](images/bb4e40b2353ae9891238a22dedf65759ec0e65f718869e01b5cc20688c7542c1.jpg)  
(h) RN-18 ImageNet: Memory

![](images/dc5ae0d387a4c2fcc4aa18f60b9cc620b8d9ff023a8acb325a3f285d4453c26a.jpg)  
(i) ViT-Base: Latency

![](images/fe6f26844d008f1ee66adceab75af8b735278d6a7432d88623bc5255c9510c4b.jpg)  
(j) ViT-Base: Adapt. time

![](images/38478cbcb5040530f29e3a54cd1039f06c490cb6b2855c9739982979fb391c92.jpg)  
(k) ViT-Base: FLOPS

![](images/bee9f0ca3003eb57fc889afe7d2dfa685e3adb77e88f0e6e6eeccb99d8586d77.jpg)  
(l) ViT-Base: Memory

Figure 12: Compute, memory and latency profile across compression ratios. For each architecture we report, as a function of sparsity, the CPU inference latency (ms), the full adaptation-step time (s). The analytical FLOPs and memory usage (GB) per adaptation step are also reported. All measurements use batch size B=64 on three channel inputs at the pretraining data resolution. The measurements are obtained on a single AMD EPYC 9654 96-core processor single-threaded. We use the evaluation protocol described in Slamanig et al. (2025). Under the uniform per-layer compression, for each compression ratio all the five structured compression methods studied in the paper yield architecturally equal smaller-dense ResNet-18 and ViT-Base networks, thus a single curve per architecture suffices; we report Magnitude- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ as a representative structured compression method.

![](images/c2ed74325f08b814d5ced158b84bfc7e961bb5066f6dde64b25368f93606eb47.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/877563b29427cfc3889b31839d29eff4dc25fb1f40efee7738b5822f189ea234.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/639c52a7c72b3f1f58c9dd27253cfc134ada7640ddebecac4490c8617b8ab389.jpg)  
(c) ViT-Base / ImageNet-C.

![](images/b2d195962c653ce13eea0e9a2d8cf59cad6bfc7b978d7ae5e8e5c48e78d5d95b.jpg)  
(d) ResNet-18 / CIFAR-10-C.

![](images/7afbea7d266ed043187c700124d2564164b119f201282692815db583fb6eb178.jpg)  
(e) ResNet-18 / ImageNet-C.

![](images/ee78c9fd9c7e1649e126b180a2b29951e084102067d780a4ab0cfd3c9b5c2e09.jpg)  
(f) ViT-Base / ImageNet-C.  
Figure 13: Gradient alignment restricted to the Noise corruption family (Gaussian, Shot, Impulse). Same visualization scheme as Fig. 4; ±σ band across the three Noise corruptions.

![](images/9c26aae32c7a23e8fc77785f3f845f2b015cb6e39693a4407118bc29ac67e523.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/82923a7b3372b718e103ef095e33034ae6b1f22a82280ad82be9ed2917d63804.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/0751b25bd57cc043cece85434cd9c7a4c74ca9044662448966794ea5150bc9b4.jpg)  
(c) ViT-Base / ImageNet-C.

![](images/3572d2de5e44bf5e256ed785e7b4669eb141032026d2ae06db264995c9b87649.jpg)  
(d) ResNet-18 / CIFAR-10-C.

![](images/8aa2b5711e5c593fad2c28caaf1961cf82baa797358fadce275530b5a3ef7bc0.jpg)  
(e) ResNet-18 / ImageNet-C.

![](images/db45ed9890af57c9fd4a569f38cf87a471713333cff2edcb9d8766978f79a978.jpg)  
(f) ViT-Base / ImageNet-C.

Figure 14: Gradient alignment restricted to the Blur corruption family (Defocus, Glass, Motion, Zoom). Same visualization scheme as Fig. 4; ±σ band across the four Blur corruptions.

![](images/044507392908a2fe608ac3ae7470fa15599d63f6d9dd97488fe879bb4a71d4c5.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/31e9418c8367b411dd45e30e082c6096608a1cc38d3197640b03b51d10bdb410.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/3dfd46662c68c636ca1eb4535a4d16b851638a081cf9727bf0c87819dd51ff94.jpg)  
(c) ViT-Base / ImageNet-C.

![](images/474bb96996b7f566f0984daabed86df235dc95cc85fc322bf99364fe1058f1e3.jpg)  
(d) ResNet-18 / CIFAR-10-C.

![](images/8bd7fe1337a0037e1feab0b05fd90ab93135cc349c389bda2fcb1dc2428fadae.jpg)  
(e) ResNet-18 / ImageNet-C.

![](images/a9116f095f74fd55fe8bbb2b2469ffb166a684b78cf11502ad0a8adf311c4a47.jpg)  
(f) ViT-Base / ImageNet-C.

Figure 15: Gradient alignment restricted to the Weather corruption family (Snow, Frost, Fog, Brightness). Same visualization scheme as Fig. 4; ±σ band across the four Weather corruptions.

![](images/be974debeb1b4e7327de7a076255c6bbff04295893a3a80add79c5858258f8d9.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/e822741bd02eb9c7e9979a37062ed5010dc23eee45b124767720c50721f6fb0f.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/b9a486695836ccda2577fe5154e0cd350a803adc285701e00f9f793d8aed9da8.jpg)  
(c) ViT-Base / ImageNet-C.

![](images/6b0a8a7a17f6361d3435d4af6be8b05f8b69246ebf002a1c07c9cfdbe0767a1a.jpg)  
(d) ResNet-18 / CIFAR-10-C.

![](images/e7c5d7ba49036ed394d4123f725691a612e0d2c38237559b8893963805e5fa69.jpg)  
(e) ResNet-18 / ImageNet-C.

![](images/8b30d5c98f98b01d9267e53409bb80ddf74df0b773daea98628b5f53776c261a.jpg)  
(f) ViT-Base / ImageNet-C.

Figure 16: Gradient alignment restricted to the Digital corruption family (Contrast, Elastic, Pixelate, JPEG). Same visualization scheme as Fig. 4; ±σ band across the four Digital corruptions.

![](images/ad95f62abe24a2107661c50a7e7e682641ae49ddb09175346cb26223a5bf0707.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/ce9dfc8e04978feec656681e1725aa77c2c51350f0c92748c6a3d1856ab22af5.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/42ba4dbbf0b84780e85f325c49f95e8afb5a8a5883665dcc82d30c64d7c23a4b.jpg)  
(c) ViT-Base / ImageNet-C.  
Figure 17: Geometry-aware similarity agrees with worst-layer CKA representation metric on ImageNet-C. Each point is one (criterion, sparsity, corruption) cell at severity 5 post-compression pre-adaptation (n equals to the number of total samples considered per panel; colors denote the compression criterion). The x-axis is the worst-layer CKA between the dense and the compressed model, the y-axis is the spectral-ratio pseudo-distance d (Cayco-Gajic & Pellegrino, 2026) computed on the same batches; since $d _ { \mathrm { S R } }$ is a distance, agreement corresponds to negative rank correlation.

![](images/46bd1d9e0e6c7aaa9027af263b0a8953b0dbac89390b225d5637d614346b12d8.jpg)  
(a) ResNet-18 / CIFAR-10-C.

![](images/e5e1b34144abf3536f74294b873eafc803d862ef67acfe39dec43a5da2b12920.jpg)  
(b) ResNet-18 / ImageNet-C.

![](images/d9d38476a06ef7d8cbc62de76a217fa910703814916e4d757c8a2204f58698bb.jpg)  
(c) ResNet-18 / CIFAR-10-C.

![](images/a8f9ad166d4a4a2a6c77ed78406e4f0286ace004590b3b30b8b0b17125b82d64.jpg)  
(d) ResNet-18 / ImageNet-C.

Figure 18: Removing the confident-but-wrong samples that pass SAR’s reliability filters recovers gradient alignment. We report $\cos ( \mathbf { g } _ { \mathrm { T T A } } , \mathbf { g } _ { \mathrm { O r a c l e } } )$ at ϕ (Eq. 2), averaged across all corruptions, with colors as in Fig. 4. Dashed lines reproduce the cosine-similarity curves of Fig. 4 unchanged; solid lines restrict the reliable set to correctly classified samples and evaluate the Oracle-SAR gradient over the same set. Excluding the confident-wrong samples restores the alignment to $\approx + 0 . 9$ in every considered scenario.

![](images/412de8fed7ef32c11dced4ff996cb4a50e8fd04a725d4aa9b14b491cd2e6599f.jpg)  
(a) ResNet-18 / CIFAR-10-C

![](images/4e73d93cb65b6ff6dee4c5de64301931116fee68a09b53c82d5193015a8e774e.jpg)  
(b) ResNet-18 / ImageNet-C

![](images/a2afe39b05fec1088842d88791568cf43c4eca2378f778c9fddefd3c55eece8f.jpg)  
(c) ViT-Base / ImageNet-C

![](images/11d1beb14ecc4bf3bb1d5b3b4d6db2c7430cc87f6c5157e130b3498c51147405.jpg)  
(d) ResNet-18 / CIFAR-10-C

![](images/31a91b2dc067291a4d17755a01220245df83820f3f72d805b7d12d11e2850bb2.jpg)  
(e) ResNet-18 / ImageNet-C

![](images/0d1c1366f38bf821feda708a396d66516517b9ef342078b358ad86cd3b4105e1.jpg)  
(f) ViT-Base / ImageNet-C  
Figure 19: Test-time adaptation collapses with increasing compression for the pruned models, unlike the smallerdense models of equal parameter budget trained from scratch. Same layout as Fig. 1 using SAR and Oracle-SAR on all the displayed architectures. The gray curves show the method-agnostic smaller-dense TTA reference. At low sparsity, TTA on the compressed models remains competitive with the references; as compression increases, TTA and its Oracle both collapse on every criterion, while the smaller-dense models retain adaptation gain. Together, these observations suggest that the loss of adaptability is induced by the structured compression rather than by the reduced model size.

![](images/969a973640db1cff77285086288a3736a7c6fb72461c9f6200b6e2034adb3b3a.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/a6d5927125abf57c7b53a2b6fa0e4eb306eb3c00db2aeb79e6eb657637084497.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/3524a9c3d9b6d2035ce4f5ba7be4159f577d3c770362c6eb33320d8a870f364b.jpg)  
(c) ViT-Base on ImageNet-C

![](images/2cf4516c3d61cde54c7a39c1e3b97d4c75492908233997b821ac7efc6e711f61.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/0638f6f4737d3ce5bfb6bc846cb3d96739f3a2fad04d8a7cae49c92c26544b39.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/8571988719cfbacec9abbc4adcab11d73a8e58729c4ec9b01760f0d30a91cf85.jpg)  
(f) ViT-Base on ImageNet-C

Figure 20: Test-time adaptation (TTA) degrades representations more than supervised fine-tuning (Oracle), with the gap depending on the compression method.

![](images/1c95d95a82419878e066f97fcab8f3ad945b9da777ac0d0b86853518bda702b6.jpg)  
(a) ResNet-18 on CIFAR-10-C

![](images/4e973b69830b587302120984ca81f430e321ef710d363fdea586b856d34a33be.jpg)  
(b) ResNet-18 on ImageNet-C

![](images/1bff656e0b3c33fc9e68278b617200760025c1b663ae58af15c0173677b0de8b.jpg)  
(c) ViT-Base on ImageNet-C

![](images/976081a07c17be4c37bf81f4cfaf194d0e64f10e0bf662af0f16cbb466eaa170.jpg)  
(d) ResNet-18 on CIFAR-10-C

![](images/55780f7957c06b5f3a9e0facaf767c5e415df3bbac22e33b2589c54eb56ce7f4.jpg)  
(e) ResNet-18 on ImageNet-C

![](images/c9a60e9a128bb6149d0532eadcd315803978f28cbb45da4d2ad15ea89f8fabeb.jpg)  
(f) ViT-Base on ImageNet-C  
Figure 21: Test-time adaptation (TTA) only partially restores output expressivity, with recovery strongly dependent on the compression method.