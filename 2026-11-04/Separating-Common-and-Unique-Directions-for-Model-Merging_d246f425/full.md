# Separating Common and Unique Directions for Model Merging

Shiting Wang<sup>1,2,3</sup> Yingjie Zou<sup>1,2,4</sup> Ke Li<sup>1,2,5</sup> <sup>∗</sup>

<sup>1</sup>Earlham Institute, United Kingdom

<sup>2</sup>Living Systems Institute, University of Exeter, United Kingdom <sup>3</sup>Department of Computer Science, University of Exeter, United Kingdom

<sup>5</sup>School of Computing Sciences, University of East Anglia, United Kingdom stwang0106@gmail.com, yz1033@exeter.ac.uk, ke.li@earlham.ac.uk

## Abstract

Model merging combines multiple fine-tuned models without additional training, and arithmetic methods have emerged as a simple yet effective way to merge models without data. However, existing arithmetic methods treat task vectors in parameter space and apply a single element-wise rule to directions that can differ in mergeability. In a shared directional basis, we find a consistent two-regime structure: a low-dispersion common bulk shared across tasks and a high-dispersion task-dominated tail that carries disproportionate energy. These findings motivate Common–Unique Subspace Decomposition (CUSD), a data-free pre-merging method that separates task vectors into common and unique directional components, merges the common components with downstream arithmetic operator, and retains unique residuals separately. Experiments across 11 base models, four model families, and four arithmetic-based merging operators show that CUSD improves 42 of 44 model–operator combinations and achieves up to 22.5% improvement over the corresponding baselines. Our code is publicly available at https://github.com/stwang666/CUSD

## 1 Introduction

Model merging aims to combine K task-specific checkpoints, fine-tuned from a common base model, into a single multi-task model without access to the original task data or joint retraining [1–8]. This setting is increasingly relevant when training data cannot be shared, checkpoints are produced by different owners, or serving one model per task is too expensive [9, 10]. Arithmetic-based merging methods are a central paradigm in this setting because they require only model weights. They represent each task by a task vector $\tau _ { i } = \theta _ { i } - \theta _ { \mathrm { b a s e } }$ [11] and construct a merged update using non-iterative operations such as addition, sign selection, sparsification, or rescaling [11–15]. Despite substantial progress [16, 17], arithmetic merging often remains below the corresponding task-specific models, indicating that residual inter-task interference is not fully resolved by current rules.

We identify one source of this interference: current arithmetic methods are largely coordinate-wise and uniform. They decide how to add, trim, drop, or rescale scalar entries of task vectors, but they do not first determine whether those entries belong to update directions that are consistently shared across tasks or dominated by only a few tasks. This distinction matters because parameter updates are not merely collections of independent coordinates. In matrix-structured layers, a direction represents a coordinated update pattern across many parameter entries. A direction used comparably by all tasks is a candidate for aggregation, whereas a direction dominated by one task may need to be preserved, damped, or separated from the shared merge. Applying the same arithmetic rule to both types of directions can either suppress task-dominated signal or inject task-specific interference into the common update.

![](images/37ebd8fd6026d46d4d5f6d4dd477450e2a6ee9865c794c23c64acf9e7b6b6638.jpg)  
Dk = stdi(Σik)

![](images/63dfa81769b096885e57b566f931dc3f228d2f175f7c21d62fdeca6da92013dc.jpg)  
Dispersion decile (%)

![](images/a3d4936241418d05e8f5d81a61c2d2c5d9b376882a84d36a988e06e48094de67.jpg)  
Dk = stdi(Σik)

![](images/f3228e9aaa2e64c7068a747b6447ae05acfd92421a4fc44ad15717b5fb9d17f5.jpg)  
Dispersion decile (%)  
Figure 1: Direction-level heterogeneity in task vectors. Motivating examples from representative matrix layers of Llama-3.2-3B (mlp.gate\_proj) and ViT-B/16 (mlp.c\_proj). (a, c) Distribution of cross-task dispersion scores over shared directions. (b, d) Share of total directional energy in each dispersion-ranked decile.

To expose this structure, we analyze task-vector matrices in a shared directional basis. We use energy to denote the update magnitude carried by task vectors along a shared direction. For each shared direction, we measure the task-wise energy and its dispersion across tasks. Figure 1 shows two motivating examples from Llama-3.2-3B [18] and ViT-B/16 [19]. In both cases, most directions form a low-dispersion bulk, while a small high-dispersion tail carries a disproportionate fraction of the total directional energy. In the shown layers, the highest-dispersion decile accounts for 53% and 77% of total energy, respectively. Thus, the high-dispersion tail is not negligible noise: it contains substantial update magnitude, but its energy is unevenly distributed across tasks. This observation motivates separating directions by their cross-task energy dispersion before applying an arithmetic merging rule. Appendix A.1 provides additional cross-architecture visualizations showing that the same two-regime pattern appears broadly across layers, model families, and task groups.

We propose Common–Unique Subspace Decomposition (CUSD), a data-free pre-merging framework for arithmetic model merging. For each matrix-structured layer, CUSD constructs a shared orthogonal directional basis from the task-vector matrices using a regularized Higher-Order Generalized Singular Value Decomposition derived (HO-GSVD-derived) construction [20, 21]. In this shared basis, CUSD computes a cross-task dispersion score for each direction and partitions directions into two sets: common directions, whose energy is comparable across tasks, and unique directions, whose energy is dominated by one or a few tasks. We use “unique” in this operational sense: these directions are task-dominated residuals, not assumed to encode semantically exclusive features.

Given this partition, the common components $C _ { i }$ and unique residuals $R _ { i }$ can be obtained from K task vectors. It then applies the downstream arithmetic merging operator $\mathcal { F }$ only to the $C _ { i }$ and controls the $R _ { i }$ with a separate scalar coefficient $\beta$ to get merged task vector $\tau _ { m }$ :

$$
\tau _ { m } = \mathcal { F } ( C _ { 1 } , \dots , C _ { K } ) + \beta \sum _ { i = 1 } ^ { K } R _ { i } ,
$$

This design makes CUSD a pre-merging module rather than a replacement for existing arithmetic operators. The same decomposition depends only on the task vectors and can therefore be computed once and reused across different arithmetic merging rules [11, 14, 15]. Compared with per-task SVD methods [22, 23], CUSD avoids comparing directions chosen independently for each task; compared with post-merge subspace corrections [24], it separates task-dominated directions before the merging rule has mixed them; and compared with data-dependent routing or modular approaches [25], it produces a single static merged model using weights alone. Our main contributions are summarized as follows:

• We identify and quantify a direction-level two-regime structure in task-vector matrices: a low-dispersion bulk of broadly shared directions and a high-dispersion, task-dominated tail that carries disproportionate energy.

• We introduce CUSD, a data-free pre-merging method that uses a HO-GSVD-derived construction to obtain a shared directional basis, separates common components from unique residuals, applies any downstream arithmetic merging operator only to the common components, and controls the unique residuals through a single coefficient $\beta .$

• We evaluate CUSD across 11 base models, four model families, and four arithmetic-based merging operators, showing that CUSD consistently improves data-free arithmetic merging across the evaluated settings.

## 2 Related Work

Current model merging research falls into two categories: data-free methods that rely only on model parameters or task vectors, and data-dependent methods that require additional data during merging.

## 2.1 Data-Free Model Merging

Arithmetic-based model merging. These methods apply uniform element-wise rules to model weights or task vectors $\{ \tau _ { i } \}$ without data or iterative optimization. Task Arithmetic (TA) [11] merges via scaled addition $\begin{array} { r } { \dot { \theta } _ { \mathrm { m e r g e d } } = \theta _ { \mathrm { b a s e } } + \lambda \sum _ { i } \tau _ { i } ; } \end{array}$ Model Soups [12] averages checkpoints. TIES-Merging [13] resolves sign conflicts through trimming, majority vote, and disjoint averaging. DARE [14] randomly prunes delta parameters with rescaling. Consensus Merging [15] identifies and suppresses catastrophic and selfish parameters based on cross-task agreement patterns. Additional methods in this class include PCB-Merging [26] and Breadcrumbs [27]. LiNeS [17] applies layerincreasing scalar scaling to the merged weights to prevent catastrophic forgetting as a post-merge arithmetic correction. All these operators decide which per-parameter values to retain, drop, or rescale; none examines the directional geometry across task vectors.

SVD-based model merging. A separate line of data-free methods moves from element-wise rules to subspace operations. TSV-Merge [23] performs per-task SVD [22] and aligns singular vectors via Procrustes rotation; KnOTS [28] applies a similar alignment strategy to LoRA [29] adapters. Both produce $K$ separate bases that must be post-hoc aligned, and the alignment step introduces approximation error. Iso-Merging [16] decomposes the merged TA output into common and taskspecific subspaces with isotropic scaling. The common subspace is defined by the top-k singular vectors of the summed task matrices, while task-specific subspaces are extracted from each task’s residual after projecting out the common subspace. Subspace Boosting [24] diagnoses rank collapse in the merged output and repairs the dominant singular subspace, using HO-GSVD [20, 21] as an interpretability tool. These post-merge methods tie their decomposition to a specific merge operator, so the result cannot be reused across methods. Moreover, they repair the merged output after interference has already occurred rather than preventing it. CUSD decomposes each task vector into common and unique subspaces before any merging operator is applied. Because the decomposition depends only on $\left\{ \tau _ { i } \right\}$ , it pairs with any arithmetic-based method and gives common and unique components independent hyperparameters.

Optimization-based model merging. Unlike the non-iterative operators above, methods in this class exploit intrinsic properties of task vectors to construct optimization objectives solved iteratively. AWD [30] proves that inter-task interference is minimized when task vectors are mutually orthogonal, and proposes adaptive weight disentanglement to push them toward orthogonality. DOGE [31] formulates merging as a constrained optimization problem, interprets task vectors as accumulated gradients, and solves it via adaptive projected gradient descent. WUDI-Merging [32] shows that a linear layer’s task vectors span an approximate linear subspace of the corresponding inputs, and uses this property to construct a data-free loss that minimizes the interference upper bound. DOP [33] combines data-space approximation with orthogonal projection and designs a data-free loss balancing stability and plasticity for continual merging. Each method optimizes toward a specific structural prior, such as orthogonality (AWD), gradient alignment (DOGE), or subspace preservation (WUDI, DOP), and its effectiveness depends on how well that prior matches the actual task-vector geometry.

## 2.2 Data-Dependent Model Merging

A second category of methods uses data-derived signals, including Fisher information [34], activation statistics, or unlabeled test samples, to guide the merging process. Fisher Merging [35] weights parameters by diagonal Fisher information estimated from labeled training data. RegMean [36] minimizes a linear regression reconstruction loss using activation inner-product matrices. ZipIt [37] computes activation statistics on unlabeled data to align and fuse redundant features across networks.

AdaMerging [38] learns task-wise or layer-wise merging coefficients via entropy minimization on unlabeled test samples. Representation Surgery [39] learns a linear transformation to repair merged representations using unsupervised data. Twin-Merging [25] decomposes each task vector via per-task SVD into shared and exclusive components, then trains a test-time router on task data to dynamically weight the exclusive components at inference. These methods require Fisher statistics, activations, or unlabeled samples beyond model weights alone and address a different setting from the data-free problem considered here.

## 3 Method

![](images/14711c9ef9edbc91dbccb81a1fff76bc1829a9d12fef7e89ca62fa29dfcec641.jpg)  
Figure 2: CUSD pipeline overview. Task vectors $A _ { i }$ are projected into an HO-GSVD-derived shared right basis V and directional energies Σ. A std-based dispersion score with a KDE-based threshold partitions directions into common (C) and unique (U). CUSD merges common components with $\mathcal { F }$ and retains unique residuals through $\beta \sum _ { i = 1 } ^ { K } { \bar { R _ { i } } }$

## 3.1 Preliminaries

Given a pre-trained base model $\theta _ { \mathrm { b a s e } }$ and K task-specific models $\theta _ { 1 } , \ldots , \theta _ { K }$ fine-tuned on tasks $1 , \ldots , K$ , the task vector for task i is $\tau _ { i } = \theta _ { i } - \theta _ { \mathrm { b a s e } } \left[ 1 1 \right]$ . Arithmetic-based merging can be written as

$$
\tau _ { m } = \mathcal { F } ( \tau _ { 1 } , \dots , \tau _ { K } ) , \qquad \theta _ { \mathrm { m e r g e d } } = \theta _ { \mathrm { b a s e } } + \tau _ { m } ,\tag{1}
$$

where $\tau _ { m }$ denotes the merged task vector, $\mathcal { F }$ denotes a downstream merging operator and $\theta _ { \mathrm { { m e r g e d } } }$ indicates the merged model. This work targets data-free arithmetic-based operators, which instantiate $\mathcal { F }$ as non-iterative per-parameter rules applied to $\{ \tau _ { i } \}$

CUSD does not change this formulation; instead, it pre-processes the task vectors before $\mathcal { F }$ is applied. These operators act on scalar entries of $\{ \tau _ { i } \}$ and therefore do not expose the matrix-level directional structure of a task update. CUSD instead projects each matrix-structured layer into a shared directional basis, where directions with comparable cross-task energy can be separated from directions dominated by one or a few tasks (Figure 2). For each retained layer ℓ with matrix-structured weights, we denote the corresponding per-task matrices by $A _ { 1 } ^ { ( \ell ) } , \ldots , A _ { K } ^ { ( \ell ) } \in \mathbb { R } ^ { m \times n }$ ; all subsequent operations are applied independently per layer, and we suppress ℓ for clarity.

## 3.2 HO-GSVD-Derived Shared Basis

A standard SVD decomposes a single matrix as $A = U \Sigma V ^ { \top }$ . Applied separately to $K$ task matrices, it produces unrelated right bases. Classical HO-GSVD instead seeks a shared right factor and is commonly written as

$$
A _ { i } = U _ { i } \Sigma _ { i } V ^ { \top } , \qquad i \in [ K ] ,
$$

which motivates comparing task matrices in a common right basis. CUSD uses the regularized and symmetrized Gram-quotient construction in Appendix A.2 to obtain an orthonormal shared basis $V = [ v _ { 1 } , \ldots , v _ { n } ]$ , which is required for Eq. 4.

For each shared direction, CUSD records the directional energy $\sigma _ { i k } = \| A _ { i } v _ { k } \| _ { 2 }$ and collects these values into $\Sigma = [ \sigma _ { i k } ] \in \mathbb { R } ^ { K \times n }$

Algorithm 1 CUSD: Common–Unique Subspace Decomposition   
Require: $\tau _ { 1 } , \dots , \tau _ { K }$ (task vectors); $\mathcal F ( \cdot )$ (any arithmetic-based method); $\beta \in [ 0 , 1 ]$   
Ensure: $\theta _ { \mathrm { { m e r g e d } } }$ (merged model)   
1: for each retained layer ℓ with matrix-structured weights do   
2: V, Σ ← SharedBasis ${ \big ( } A _ { 1 } ^ { ( \ell ) } , \ldots , A _ { K } ^ { ( \ell ) } { \big ) }$ {Alg. 2}   
3: $\mathcal { C } , \mathcal { U } \gets \mathbf { C l a s s i f y } ( \Sigma )$ {std-based dispersion + KDE; §3.3}   
4: $P c  V c V _ { c } ^ { \top } ; P u  I - P c$ {Orthogonal projectors}   
5: for $i = 1 , \ldots , K$ do   
6: $C _ { i }  A _ { i } P c ; ~ R _ { i }  A _ { i } P _ { \mathcal { U } }$ {Prop. 2}   
7: end for   
8: $\tau _ { m } ^ { ( \ell ) } \gets \mathcal { F } ( C _ { 1 } , \dots , C _ { K } ) \ + \ \beta \sum _ { i = 1 } ^ { K } R _ { i }$ {Eq. 5}   
9: end for   
10: return $\theta _ { \mathrm { m e r g e d } }  \theta _ { \mathrm { b a s e } } + \tau _ { m }$

Because all tasks are evaluated along the same directions, these energies are directly comparable across tasks. This shared-basis construction is computed once for each retained matrix-structured layer and reused across merging methods.

## 3.3 The CUSD Algorithm

Std-based dispersion score. Given the matrix Σ, CUSD measures the cross-task imbalance of each shared direction. For direction k, the std-based dispersion score is

$$
\begin{array} { r } { d _ { k } = \mathrm { S t d } _ { i \in [ K ] } \bigl ( \tilde { \sigma } _ { i k } \bigr ) , \qquad \tilde { \sigma } _ { i k } = \left\{ \begin{array} { r l r } { \sigma _ { i k } , } & { \mathrm { a b s o l u t e ~ s c o r i n g } , } \\ { \frac { \sigma _ { i k } } { K ^ { - 1 } \sum _ { i } \sigma _ { i k } } , } & { \mathrm { r e l a t i v e ~ s c o r i n g } . } \end{array} \right. } \end{array}\tag{2}
$$

Directions with comparable cross-task energy receive small $d _ { k } ,$ , whereas directions dominated by one or a few tasks receive large $d _ { k }$ . Relative scoring removes the per-direction mean energy scale and measures relative cross-task imbalance. We compare the two scoring choices empirically in Section 4.3.

KDE. The threshold separating common from unique is determined automatically from the distribution of $\{ d _ { k } \} _ { k = 1 } ^ { n }$ via kernel density estimation (KDE) with Scott’s bandwidth [40] (full details appear in Appendix A.3). Let δ denote the detected valley. CUSD then sets

$$
{ \mathcal { C } } = \{ k : d _ { k } < \delta \} , \qquad { \mathcal { U } } = \{ k : d _ { k } \geq \delta \} .\tag{3}
$$

Subspace merging. Given the partition $( \mathcal { C } , \mathcal { U } )$ , let $V _ { \mathcal { C } }$ and $V _ { \mathcal { U } }$ denote the columns of V indexed by C and U, respectively. Define

$$
P _ { \mathcal { C } } = V _ { \mathcal { C } } V _ { \mathcal { C } } ^ { \top } , \qquad P _ { \mathcal { U } } = V _ { \mathcal { U } } V _ { \mathcal { U } } ^ { \top } = I - P _ { \mathcal { C } } , \qquad C _ { i } = A _ { i } P _ { \mathcal { C } } , \qquad R _ { i } = A _ { i } P _ { \mathcal { U } } .\tag{4}
$$

Because $P _ { C }$ and $P _ { \cal { U } }$ are complementary orthogonal projectors, this gives an exact orthogonal decomposition: $C _ { i } + R _ { i } = A _ { i }$ and $\langle C _ { i } , \dot { R _ { i } } \rangle _ { F } = \bar { 0 } ;$ ; proof in Appendix $\mathrm { A . 4 }$ . The common component $C _ { i }$ contains directions that are consistently emphasized across tasks, whereas the unique residual $R _ { i }$ contains directions whose energy is concentrated in one or a few tasks and therefore reflects task-specific variation. CUSD therefore applies the merging operator $\mathcal { F }$ only to $\{ C _ { i } \}$ , while the unique residuals $\{ R _ { i } \}$ are summed and scaled by $\beta \in [ 0 , \bar { 1 } ]$ . The resulting layer-wise merged task vector is

$$
\tau _ { m } = \mathcal { F } ( C _ { 1 } , \dots , C _ { K } ) + \beta \sum _ { i = 1 } ^ { K } R _ { i } .\tag{5}
$$

Setting $\beta = 0$ discards all unique directions; $\beta = 1$ retains them in full. Algorithm 1 summarizes the complete per-layer CUSD procedure. The decomposition (lines 2–5) depends only on the task vectors and the classification rule, not on ${ \mathcal F } .$ . This pre-processing is computed once and reused with any arithmetic-based merging method.

Overall, CUSD augments an arithmetic-based merging operator with three components: a shared-basis decomposition, a std-based common–unique partition, and a $\beta \mathrm { . }$ -controlled unique residual.

## 4 Experiments

## 4.1 Experimental Setup

Models and tasks. We evaluate CUSD on 11 base models covering vision encoders, text encoders, and LLMs: CLIP ViT-B/32, ViT-B/16, and ViT-L/14 [19, 41]; RoBERTa-Base and RoBERTa-Large [42]; and MergeBench Llama and Gemma models [18, 43, 44]. The evaluation follows prior model- merging benchmarks on eight image-classification datasets [32, 38], eight GLUE NLU tasks [32, 45], and MergeBench code, math, and multilingual LLM tasks [43]. Full model and dataset details are provided in Appendix A.5.

Baselines and implementation. We pair CUSD with four representative data-free arithmeticbased merging methods: Task Arithmetic [11], TIES-Merging [13], DARE [14], and Consensus Merging [15]. For each method, we report the plain merge and the corresponding CUSD-preprocessed merge; TSV-Merge [23] and Iso- Merging [16] are included in the subspace-based comparison. We also evaluate CUSD preprocessing with WUDI-Merging [32] and ACE-Merging [7]. CUSD is applied to matrix-structured linear-layer task vectors. Unless otherwise specified, baseline hyperparameters follow the original papers or MergeBench defaults and are kept fixed between plain and CUSDpreprocessed runs. Additional implementation details are provided in Appendix A.5, and Section 4.3 reports the scoring mode and β sensitivity used in the main experiments.

## 4.2 Main Results

Overall performance. Table 1 reports average normalized performance across all model families. Parenthesized values denote changes from the corresponding plain merge, and boldface marks the best score in each model column. CUSD improves 42 of the 44 model–operator combinations, with one tie and one small decrease. The largest gain is 22.5 normalized points on RoBERTa-Large, and all ViT and RoBERTa settings improve. On LLMs, CUSD improves 22 of 24 settings. Detailed task-level results in Appendix A.6 show that these gains often reflect an improved multi-task trade-off: small drops on already strong tasks are accompanied by larger gains on other tasks, consistent with separating unique directions before arithmetic aggregation.

Table 1: Overall performance across all models. We report average normalized scores.
<table><tr><td>Method</td><td>with CUSD</td><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-L/14</td><td>RoBERTa-B</td><td>RoBERTa-L</td><td>Llama-3B</td><td>Llama-8B</td><td>Llama-3B-Inst Llama-8B-Inst</td><td></td><td>Gemma-2b</td><td>Gemma-9b</td></tr><tr><td rowspan="2">Fine-tuned</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>×</td><td>71.8</td><td>76.8</td><td>87.6</td><td>53.5</td><td>58.3</td><td>69.6</td><td>93.7</td><td>73.5</td><td>82.5</td><td>94.2</td><td>92.0</td></tr><tr><td>Task Arithmetic</td><td>√</td><td>85.0 (+13.2)</td><td>88.9 (+12.1)</td><td>93.0 (+5.4)</td><td>68.7 (+15.1)</td><td>70.0 (+11.7)</td><td>69.6 (0.0)</td><td>94.1 (+0.4)</td><td>75.8 (+2.3)</td><td>86.5 (+4.0)</td><td>95.7 (+1.5)</td><td>94.8 (+2.8)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>74.0</td><td>78.5</td><td>88.5</td><td>51.1</td><td>47.3</td><td>69.2</td><td>93.7</td><td>72.7</td><td>82.6</td><td>94.1</td><td>91.1</td></tr><tr><td>√</td><td>87.0 (+13.0)</td><td>90.3 (+11.8)</td><td>94.8 (+6.3)</td><td>69.3 (+18.2)</td><td>69.8 (+22.5)</td><td>69.7 (+0.5)</td><td>94.1 (+0.4)</td><td>75.5 (+2.8)</td><td>86.2 (+3.6)</td><td>97.3 (+3.2)</td><td>95.3 (+4.2)</td></tr><tr><td rowspan="2">DARE</td><td>×</td><td>71.7</td><td>76.6</td><td>87.5</td><td>54.0 58.8</td><td>68.9</td><td>94.1</td><td></td><td>73.4</td><td>83.1</td><td>94.1</td><td>91.2</td></tr><tr><td>√</td><td>84.8 (+13.2)</td><td>88.8 (+12.2)</td><td>92.9 (+5.4)</td><td>68.3 (+14.3)</td><td>69.5 (+10.7)</td><td>70.3 (+1.4)</td><td>93.8 (-0.3)</td><td>75.4 (+2.0)</td><td>86.4 (+3.3)</td><td>96.2 (+2.1)</td><td>94.8 (+3.6)</td></tr><tr><td>Consensus</td><td>X √</td><td>73.2 85.0 (+11.8)</td><td>78.0 88.8 (+10.8)</td><td>88.0 92.8 (+4.9)</td><td>55.2 67.3 (+12.1)</td><td>61.9 70.0 (+8.1)</td><td>67.4 70.6 (+3.2)</td><td>93.8 93.9 (+0.1)</td><td>76.2 76.6 (+0.4)</td><td>86.2 87.5 (+1.3)</td><td>88.8 93.1 (+4.3)</td><td>94.6 95.5 (+0.9)</td></tr></table>

Results on encoder models. On vision encoders, CUSD improves all four operators on all three ViT models, with average gains of about 12 points on ViT-B/32 and ViT-B/16. Appendix Tables 8–11 show a trade-off pattern: with TIES-Merging, CUSD reduces the already strong MNIST score by only 0.1 points while improving three weaker tasks by 8.5–10.7 points. On text encoders, CUSD improves every operator on both RoBERTa models, and the detailed GLUE tables show that STS-B changes from negative plain-merge scores to positive CUSD scores across operators. These trends support the view that common–unique separation improves the multi-task balance in encoder merging.

Results on LLMs. On generative LLMs, CUSD improves most settings. The strongest gains appear on the Llama Instruct and Gemma models, while Llama-3.1-8B leaves little headroom because plain merges already reach normalized averages around 94; the only negative LLM change is a 0.3-point decrease for DARE on this model. The role of partial unique- residual scaling in LLMs is examined in the β ablation (Section 4.3) and analysis (Section 4.4). Full task-level LLM results are provided in Appendix Tables 13–18.

Comparison with other merging methods. Table 2 compares CUSD+TA with TSV-M, Iso-C, and Iso-CTS across four model families. For LLMs, the SVD baselines report the best-observed scores from post-hoc scaling searches on the test sets. Iso-CTS performs best on ViT-B/16 and Iso-C on Gemma-2-2b, whereas CUSD+TA achieves the highest scores on RoBERTa-Base and Llama-3.2-3B.

Table 2: Comparison with subspace-based merging methods. Average normalized scores are reported.
<table><tr><td>Method</td><td>ViT B/16</td><td>RoBERTa Llama Base</td><td>3.2-3B</td><td>Gemma 2-2b</td></tr><tr><td>Fine-tuned</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>TSV-M</td><td>93.4</td><td>45.9</td><td>64.5</td><td>96.1</td></tr><tr><td>Iso-C</td><td>94.5</td><td>42.9</td><td>65.7</td><td>97.7</td></tr><tr><td>Iso-CTS</td><td>95.6</td><td>42.8</td><td>64.5</td><td>97.2</td></tr><tr><td>CUSD+TA</td><td>88.9</td><td>68.7</td><td>69.6</td><td>95.7</td></tr></table>

Table 3: Effect of CUSD preprocessing on WUDI-Merging and ACE-Merging. Average normalized scores are reported.
<table><tr><td rowspan="2">Method</td><td rowspan="2">ViT B/16</td><td rowspan="2">RoBERTa Llama Gemma</td><td rowspan="2">3.2-3B</td><td rowspan="2">2-2b</td></tr><tr><td>Base</td></tr><tr><td>Fine-tuned</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>WUDI</td><td>95.6</td><td>69.3</td><td>65.6</td><td>93.6</td></tr><tr><td>CUSD+WUDI</td><td>94.0</td><td>80.7</td><td>66.9</td><td>94.9</td></tr><tr><td>ACE</td><td>96.4</td><td>85.6</td><td>61.9</td><td>93.7</td></tr><tr><td>CUSD+ACE</td><td>94.7</td><td>81.1</td><td>63.3</td><td>98.6</td></tr></table>

We further evaluate CUSD preprocessing with WUDI-Merging and ACE-Merging (Table 3). CUSD improves WUDI in three of four settings and ACE on both LLMs, while encoder results are mixed.

## 4.3 Ablation Study

Dispersion scoring choice. We compare absolute and relative scoring (Eq. 2). Relative scoring performs better in six of the eight LLM model–operator settings, while absolute scoring performs better with Task Arithmetic on both models (Table 4). These results support relative scoring as the preferred choice for the evaluated LLM settings. Section 4.4 examines how cross-task directionalenergy patterns motivate the choice between absolute and relative scoring.

Table 4: Absolute vs. relative dispersion scoring on decoder LLMs. We report normalized average performance; green values denote gains over the corresponding plain merge.
<table><tr><td rowspan="2">Method</td><td colspan="4">Llama-3.2-3B</td><td colspan="4">Gemma-2-2b</td></tr><tr><td>w/o CUSD</td><td> $\mathbf { C U S D } \left( s _ { \mathrm { a b s } } \right)$ </td><td> $\mathbf { C U S D } \left( s _ { \mathrm { r e l } } \right)$ </td><td> $s _ { \mathrm { r e l } } \cdot s _ { \mathrm { a b s } }$ </td><td>w/o CUSD</td><td> $\mathbf { C U S D } \left( s _ { \mathrm { a b s } } \right)$ </td><td> $\mathbf { C U S D } \left( s _ { \mathrm { r e l } } \right)$ </td><td> $s _ { \mathrm { r e l } } \cdot s _ { \mathrm { a b s } }$ </td></tr><tr><td>Task Arithmetic</td><td>69.6</td><td> ${ \bf 7 0 . 0 ( + 0 . 4 ) }$ </td><td> $6 9 . 6 ( + 0 . 0 ) $ </td><td>-0.4</td><td>94.2</td><td> $\mathbf { 9 6 . 9 } ( + 2 . 7 )$ </td><td> $9 5 . 7 ( + 1 . 5 ) $ </td><td>-1.2</td></tr><tr><td>TIES-Merging</td><td>69.2</td><td> $6 9 . 5 ( + 0 . 3 ) $ </td><td> ${ \bf 6 9 . 7 ( + 0 . 5 ) }$ </td><td>0.2</td><td>94.1</td><td> $9 5 . 9 ( + 1 . 8 ) $ </td><td> $\mathbf { 9 7 . 3 } ( + 3 . 2 )$ </td><td>1.4</td></tr><tr><td>DARE</td><td>68.9</td><td> $6 9 . 6 ( + 0 . 7 ) $ </td><td> $7 0 . 3 ( + 1 . 4 )$ </td><td>0.7</td><td>94.1</td><td> $9 5 . 8 ( + 1 . 7 ) $ </td><td> $\mathbf { 9 6 . 2 } ( + 2 . 1 ) $ </td><td>0.4</td></tr><tr><td>Consensus</td><td>67.4</td><td> $7 0 . 0 ( + 2 . 6 ) $ </td><td> $7 0 . 6 ( + 3 . 2 )$ </td><td>0.6</td><td>88.8</td><td> $9 1 . 7 ( + 2 . 9 ) $ </td><td> $\mathbf { 9 3 . 1 } ( + 4 . 3 )$ </td><td>1.4</td></tr></table>

Direction selection. To examine whether the choice of unique directions matters, we compare CUSD with Random, Reversed (lowest dispersion), and Top-energy selection. In the matched controls, each method selects the same number of unique directions as CUSD in every layer. We also test Top-energy with fixed selection fractions. CUSD outperforms all controls on both backbones (Table 5); Top-energy is more competitive than Random and Reversed, consistent with the energy concentration of CUSD’s unique directions.

Common–unique merging design. We apply $\mathcal { F }$ to common components only (Common, Eq. 5), unique components only (Unique), or both separately (Both). Common performs best in all four reported settings (Table 6), supporting common aggregation with separately controlled unique-residual retention.

Unique-residual coefficient. The coefficient $\beta$ controls the retention strength of the unique residual. We sweep $\beta \in \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ to examine how sensitive CUSD is to this residual contribution. Figure 3 shows that encoders favor small $\beta ,$ whereas decoder LLMs benefit from moderate residual retention, typically around $\beta \in [ 0 . 2 , 0 . 5 ]$ . Section 4.4 analyzes this difference. Since per-operator sweeps in Figure 13 show consistent trends across downstream merge operators, Figure 3 reports the average over the four methods.

We further examine how $\beta$ interacts with the downstream hyperparameter $\lambda ,$ using Task Arithmetic as a case study. For $\begin{array} { r } { \mathrm { C U S D + T A } , \tau _ { m } = \lambda \sum _ { i } C _ { i } + \beta \sum _ { i } R _ { i } } \end{array}$ , which reduces to plain TA when $\beta = \lambda$

Table 5: Direction-selection ablation for CUSD+TA. Average accuracy (%) is reported; Random shows mean ± standard deviation over three seeds.
<table><tr><td>Model</td><td>Random</td><td>Reversed</td><td>Top-energy (matched)</td><td>Top-energy (0.2%)</td><td>Top-energy (0.5%)</td><td>Top-energy (1.0%)</td><td>CUSD</td></tr><tr><td>ViT-B/32</td><td> $5 9 . 2 0 \pm 0 . 0 8$ </td><td>59.57</td><td>73.78</td><td>74.75</td><td>73.68</td><td>70.97</td><td>76.48</td></tr><tr><td>ViT-B/16</td><td> $6 5 . 6 4 \pm 1 . 1 7$ </td><td>65.09</td><td>81.07</td><td>80.84</td><td>80.34</td><td>78.97</td><td>81.74</td></tr></table>

Table 19 reports a post-hoc joint sweep on ViT-B/32. CUSD+TA outperforms plain TA at the same λ in every tested configuration with $\bar { \lambda } \geq 0 . 3$ and $\beta < \lambda$ . These results show that reducing the unique-residual contribution relative to the common component can improve merging performance in this setting.

![](images/8d6aed2edeaaf13fb146d48af243fecde3fbcba26d41b3c0c0502df200fa321c.jpg)

<table><tr><td rowspan="8">ViT/KDE ViT/Fixed/</td><td colspan="4">KDE bandwidth</td></tr><tr><td>Scott</td><td>Silverman</td><td>0.5×</td><td>1.5x</td></tr><tr><td>88.9</td><td>88.9</td><td>89.1</td><td>86.6</td></tr><tr><td>67.6</td><td>70.2</td><td>73.7</td><td>79.4</td></tr><tr><td>69.6 LLM/KDE</td><td>68.6</td><td>67.5</td><td>67.3</td></tr><tr><td>65.4 LLM/Fixed/</td><td>66.1</td><td>66.6</td><td>66.7</td></tr><tr><td>0.80</td><td>0.85</td><td>0.90</td><td>65 0.95</td></tr><tr><td colspan="4">Fixed percentile</td></tr></table>

Figure 3: Sensitivity to the unique-residual retention coefficient $\beta$ across all 11 models.  
Figure 4: Average performance under KDEbased thresholds and fixed percentile thresholds. For KDE rows, columns denote bandwidth choices; for fixed-threshold rows, columns denote percentile cutoffs.

Adaptive KDE thresholding. CUSD uses KDE valley detection to determine the common–unique partition from the dispersion-score distribution, avoiding a fixed global cutoff. Figure 4 compares this adaptive threshold with fixed percentile thresholds. KDE thresholding consistently outperforms fixed thresholds on both ViT and LLM settings, with the largest gap on ViT where fixed percentiles are sensitive to the chosen cutoff. The result is also stable across standard KDE bandwidth choices: Scott’s and Silverman’s rules give nearly identical performance, and moderate bandwidth changes preserve the overall advantage of KDE. This supports using a data-driven layer-adaptive partition rather than tuning a fixed percentile threshold.

Table 6: Merging-design ablation. Average normalized scores are reported; Common denotes CUSD.
<table><tr><td>Model</td><td>Operator Common</td><td></td><td>Unique</td><td>Both</td></tr><tr><td>ViT-B/16</td><td>TIES</td><td>90.3</td><td>73.0</td><td>72.4 76.3</td></tr><tr><td>ViT-B/16</td><td>DARE</td><td>88.8</td><td>76.4</td><td></td></tr><tr><td>Gemma-2-2b TIES Gemma-2-2b DARE</td><td></td><td>97.3 96.2</td><td>94.2 93.6</td><td>95.2 94.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: One-time CUSD decomposition cost.
<table><tr><td>Model</td><td>Tasks Layers</td><td>Total time</td><td>Time per layer</td><td>GPU peak</td></tr><tr><td>ViT-B/32</td><td>8</td><td>48 16.5s</td><td>0.343s</td><td>1.1 GB</td></tr><tr><td>ViT-B/16</td><td>8</td><td>48 16.3s</td><td>0.340s</td><td>1.1 GB</td></tr><tr><td>ViT-L/14</td><td>8</td><td>96 48.5s</td><td>0.505s</td><td>1.9 GB</td></tr><tr><td>Llama-3B</td><td>3</td><td>196 67.5s</td><td>0.345s</td><td>5.0 GB</td></tr><tr><td>Llama-8B</td><td>3</td><td>224 217.2s</td><td>0.970s</td><td>15.4 GB</td></tr></table>

## 4.4 Analysis

What CUSD separates. The common–unique partition gives an operational view of the two-regime structure observed in Figure 1 and further visualized in Appendix A.1: directions assigned to the common subspace form a broad low-dispersion bulk, whereas directions assigned to the unique residual are often fewer but carry substantially higher per-direction energy. Figure 5 confirms this pattern across the evaluated model families. Unique directions consistently have higher per-direction energy than common directions, especially in encoder models. This shows that CUSD does not merely discard weak or low-energy directions; it separates an energy-concentrated residual subspace whose directions are unevenly supported across tasks. As summarized in Figure 6, this concentration is particularly strong in encoders: fewer than 1% of directions account for roughly 12–20% of the directional energy.

![](images/183d0ae64dfe6cfa060bd6a151f61293d0a85bec0a889302947d43b737c0f503.jpg)  
Figure 5: Layer-wise mean energy per direction in common and unique subspaces.

![](images/4482ebb2a468b652ca828b22cdc55b5b1b8287a2f7ee3f0b101e2f4d06787867.jpg)  
Figure 6: Fractions of directions and total energy assigned to the unique subspace.

Understanding dispersion scoring. For decoder LLMs, small absolute dispersion does not necessarily mean that a direction is uniformly balanced across tasks. If such directions were simply shared evenly by all tasks, mean normalization would keep their relative dispersion small. Instead, Figure 8 shows that the low-dispersion mass under $s _ { \mathrm { a b s } }$ is substantially reduced under $s _ { \mathrm { r e l } }$ . Together with the task-imbalanced energy patterns in Figure 7, this indicates that many low-absolute-dispersion directions are low or moderate energy directions that are nevertheless dominated by specific tasks. Relative scoring removes this raw-scale suppression and exposes these directions as high-dispersion.

The encoder case is different. In ViT, the heatmap in Figure 7 shows more balanced directional energies across tasks, except for a small high-energy region with visible task dominance. In this regime, $s _ { \mathrm { a b s } }$ preserves the energy-concentration signal, whereas $s _ { \mathrm { r e l } }$ gives low-energy directions the same scale as high-energy ones and can amplify small relative fluctuations. The ViT distributions in Figure 8 are consistent with this effect: relative scoring introduces a longer high-dispersion tail despite the comparatively balanced heatmap. This can weaken the separation between high-energy task-dominated directions and low-energy variations. This supports our main-experiment choice of using $s _ { \mathrm { a b s } }$ for the evaluated encoder settings and $s _ { \mathrm { r e l } }$ for the evaluated LLM settings. Appendix Figures 11 and 12 provide additional layer heatmaps.

![](images/0b4e83484e26f41fb05c2ffbd42a4c8cb9e1bc62ee9aaba63c9104695eb8526f.jpg)  
Figure 7: Per-task directional singular values for representative ViT, Llama, and Gemma layers. Each panel shows the HO-GSVD singular values $\sigma _ { i k }$ of task vectors along the top shared directions.

Unique residual scale across model families. Figure 6 shows that the unique subspace differs across the evaluated model families and task groups. In encoders, the unique subspace is extremely sparse: it contains fewer than 1% of directions but carries roughly 12–20% of the directional energy. This creates a concentrated residual component, so increasing $\beta$ can quickly reintroduce task-specific interference; this explains why encoder performance in Figure 3 is sensitive to $\beta$ and why $\beta = 0$ is a stable default. In decoder LLMs, the unique subspace is much larger and carries a substantial fraction of the energy, from 17.3% to 81.2%. Removing this broader residual component entirely can discard task-specific generative information, while moderate residual scaling preserves part of it without directly merging it into the common component. This explains why LLMs prefer nonzero $\beta$ and remain less sensitive over a moderate range of $\mathit { \Pi } _ { \beta }$ values.

Interference reduction. To examine whether CUSD reduces interference inside the merged update, we measure a normalized interference proxy (NI), defined in Appendix A.9. Figure 9 compares DARE and CUSD-DARE on the three ViT backbones. CUSD consistently lowers NI across blocks, with average reductions of 10.0%, 9.0%, and 10.2% on ViT-B/32, ViT-B/16, and ViT-L/14, respectively. This supports the proposed mechanism: separating common and unique directions before merging reduces the parameter-space interference that arises when arithmetic rules are applied directly to task-dominated directions. The same trend holds for other downstream merge operators, as reported in Appendix A.9.

![](images/174b20615ff258781c117aca072400cb4b1a6b86119231307b41067254485ec0.jpg)  
(a) Absolute dispersion across all blocks/directions.

![](images/d2f60383c69ec06343ed36396e967ad06373dfc9cdcdd99e9608e0fc0cb97214.jpg)  
(b) Relative dispersion across all blocks/directions.  
Figure 8: Distribution of dispersion scores under absolute and relative scoring. Mean normalization changes the dispersion distribution, exposing scale-suppressed task imbalance in the evaluated LLM settings while amplifying low-energy relative fluctuations in the evaluated encoder settings.

![](images/f0c01b57880090e9d35c5dba75670c1c6a95cdf54568a018b2c27a577544ebc0.jpg)  
(a) ViT-B/32.

![](images/9fd5ae05930ccefaffd263d120f488eeecd133ec10236852b7277bed2b3582f0.jpg)  
(b) ViT-B/16.

![](images/ffb9f02a47ac7de2b956164f58f0797faccfb929f3f2d6eb88db1727be4f859b.jpg)  
(c) ViT-L/14.  
Figure 9: Per-block interference reduction. DARE and CUSD-DARE are compared on ViT models using the normalized interference proxy (NI); lower values indicate less deviation from the taskspecific updates.

Computation cost. CUSD adds a one-time shared basis decomposition whose common components and unique residuals are reused across merge operators. Table 7 shows that ViT decompositions take 16.3–48.5s with at most 1.9GB peak GPU memory, while Llama-3.1-8B takes 217.2s with 15.4GB peak memory. The cost is therefore practical and amortized across downstream arithmetic methods.

## 5 Conclusion

This work studies data-free arithmetic merging from a direction-level perspective. In an HO-GSVDderived shared basis, task-vector matrices exhibit broadly common directions and high-dispersion unique directions whose energy is concentrated in only a subset of tasks. CUSD uses this observation to decompose task vectors before arithmetic aggregation, merge only the common components, and control the unique residuals separately. Since the decomposition depends only on model weights, the same common components and unique residuals can be reused across downstream arithmetic merge operators.

Across vision encoders, text encoders, and decoder LLMs, CUSD improves most model–operator combinations and reduces parameter-space interference. The ablations further show that effective unique-residual scaling varies across the evaluated model families and task groups, supporting the separate control of common aggregation and unique residuals. These results support separating common and task-dominated directions before arithmetic merging.

## Acknowledgments and Disclosure of Funding

This work was supported by the UKRI Future Leaders Fellowship under grants MR/S017062/1 and MR/X011135/1; the National Natural Science Foundation of China under grants 62376056 and 62076056; Isambard AI Open Access (AIOA - 01 - 2026-06-04 - EvoRNA: Predictive Scaling of Evolution-Aware and Multimodal RNA Foundation Models for Crop Resilience); Modal Labs through a research grant of compute credits; the Royal Society Faraday Discovery Fellowship (FDF/S2/251014); BBSRC Transformative Research Technologies (UKRI1875); the Royal Society International Exchanges Award (IES/R3/243136); the Kan Tong Po Fellowship (KTP/R1/231017); the Amazon Research Award; and the Alan Turing Fellowship.

## References

[1] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications, and opportunities. ACM Computing Surveys, 58(8):1–41, 2026.

[2] Mingyang Song and Mao Zheng. Model merging in the era of large language models: Methods, applications, and future directions. arXiv preprint arXiv:2603.09938, 2026.

[3] Yuxuan Yao, LIU Shuqi, Zehua Liu, Qintong Li, LIU Mingyang, Xiongwei Han, Zhijiang Guo, Han Wu, and Linqi Song. Activation-guided consensus merging for large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[4] Yuanyi Wang, Yanggan Gu, Yiming Zhang, Qi Zhou, Zhaoyi Yan, Congkai Xie, Xinyao Wang, Jianbo Yuan, and Hongxia Yang. Model merging scaling laws in large language models. arXiv preprint arXiv:2509.24244, 2025.

[5] Amin Heyrani Nobari, Kaveh Alim, Ali ArjomandBigdeli, Akash Srivastava, Faez Ahmed, and Navid Azizan. Activation-informed merging of large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[6] Marawan Gamal Abdel Hameed, Derek Tam, Pascal Jr Tikeng Notsawo, Colin Raffel, and Guillaume Rabusseau. Model merging via data-free covariance estimation. arXiv preprint arXiv:2604.01329, 2026.

[7] Bo Xu, Haotian Wu, Hehai Lin, Weiquan Huang, Beier Zhu, Yao Shu, and Chengwei Qin. Ace-merging: Data-free model merging with adaptive covariance estimation. arXiv preprint arXiv:2603.02945, 2026.

[8] Wenju Sun, Qingyong Li, Wen Wang, Yangliao Geng, and Boyang Li. Task arithmetic in trust region: A training-free model merging approach to navigate knowledge conflicts. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 5178–5187, 2025.

[9] Karsten Roth, Vishaal Udandarao, Sebastian Dziadzio, Ameya Prabhu, Mehdi Cherti, Oriol Vinyals, Olivier Hénaff, Samuel Albanie, Matthias Bethge, and Zeynep Akata. A practitioner’s guide to continual multimodal pretraining. arXiv preprint arXiv:2408.14471, 2024.

[10] Prateek Yadav, Tu Vu, Jonathan Lai, Alexandra Chronopoulou, Manaal Faruqui, Mohit Bansal, and Tsendsuren Munkhdalai. What matters for model merging at scale? arXiv preprint arXiv:2410.03617, 2024.

[11] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022.

[12] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.

[13] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in neural information processing systems, 36:7093–7115, 2023.

[14] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning, 2024.

[15] Ke Wang, Nikolaos Dimitriadis, Guillermo Ortiz-Jimenez, François Fleuret, and Pascal Frossard. Localizing task information for improved model merging and compression. arXiv preprint arXiv:2405.07813, 2024.

[16] Daniel Marczak, Simone Magistri, Sebastian Cygert, Bartłomiej Twardowski, Andrew D Bagdanov, and Joost Van De Weijer. No task left behind: Isotropic model merging with common and task-specific subspaces. arXiv preprint arXiv:2502.04959, 2025.

[17] Ke Wang, Nikolaos Dimitriadis, Alessandro Favero, Guillermo Ortiz-Jimenez, François Fleuret, and Pascal Frossard. Lines: Post-training layer scaling prevents forgetting and enhances model merging. In The Thirteenth International Conference on Learning Representations, 2025.

[18] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[20] Sri Priya Ponnapalli, Michael A Saunders, Charles F Van Loan, and Orly Alter. A higher-order generalized singular value decomposition for comparison of global mrna expression from multiple organisms. PloS one, 6(12):e28072, 2011.

[21] Idris Kempf, Paul J Goulart, and Stephen R Duncan. A higher-order generalized singular value decomposition for rank-deficient matrices. SIAM Journal on Matrix Analysis and Applications, 44(3):1047–1072, 2023.

[22] Virginia Klema and Alan Laub. The singular value decomposition: Its computation and some applications. IEEE Transactions on automatic control, 25(2):164–176, 1980.

[23] Antonio Andrea Gargiulo, Donato Crisostomi, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, and Emanuele Rodola. Task singular vectors: Reducing task interference in model merging. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18695–18705, 2025.

[24] Ronald Skorobogat, Karsten Roth, and Mariana-Iuliana Georgescu. Subspace-boosted model merging. arXiv preprint arXiv:2506.16506, 2025.

[25] Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng. Twin-merging: Dynamic integration of modular expertise in model merging. Advances in Neural Information Processing Systems, 37:78905–78935, 2024.

[26] Guodong Du, Junlin Lee, Jing Li, Runhua Jiang, Yifei Guo, Shuyang Yu, Hanting Liu, Sim K Goh, Ho-Kin Tang, Daojing He, et al. Parameter competition balancing for model merging. Advances in Neural Information Processing Systems, 37:84746–84776, 2024.

[27] MohammadReza Davari and Eugene Belilovsky. Model breadcrumbs: Scaling multi-task model merging with sparse masks. In European Conference on Computer Vision, pages 270–287. Springer, 2024.

[28] George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. Model merging with svd to tie the knots. arXiv preprint arXiv:2410.19735, 2024.

[29] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[30] Feng Xiong, Runxi Cheng, Wang Chen, Zhanqiu Zhang, Yiwen Guo, Chun Yuan, and Ruifeng Xu. Multi-task model merging via adaptive weight disentanglement. arXiv preprint arXiv:2411.18729, 2024.

[31] Yongxian Wei, Anke Tang, Li Shen, Zixuan Hu, Chun Yuan, and Xiaochun Cao. Modeling multitask model merging as adaptive projective gradient descent. arXiv preprint arXiv:2501.01230, 2025.

[32] Runxi Cheng, Feng Xiong, Yongxian Wei, Wanyun Zhu, and Chun Yuan. Whoever started the interference should end it: Guiding data-free model merging via task vectors. arXiv preprint arXiv:2503.08099, 2025.

[33] Enneng Yang, Anke Tang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, and Jie Zhang. Continual model merging without data: Dual projections for balancing stability and plasticity. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[34] Jorma J Rissanen. Fisher information and stochastic complexity. IEEE transactions on information theory, 42(1):40–47, 2002.

[35] Michael S Matena and Colin A Raffel. Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems, 35:17703–17716, 2022.

[36] Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. arXiv preprint arXiv:2212.09849, 2022.

[37] George Stoica, Daniel Bolya, Jakob Bjorner, Pratik Ramesh, Taylor Hearn, and Judy Hoffman. Zipit! merging models from different tasks without training. arXiv preprint arXiv:2305.03053, 2023.

[38] Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guibing Guo, Xingwei Wang, and Dacheng Tao. Adamerging: Adaptive model merging for multi-task learning. arXiv preprint arXiv:2310.02575, 2023.

[39] Enneng Yang, Li Shen, Zhenyi Wang, Guibing Guo, Xiaojun Chen, Xingwei Wang, and Dacheng Tao. Representation surgery for multi-task model merging. In Proceedings of the 41st International Conference on Machine Learning, pages 56332–56356, 2024.

[40] David W Scott. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons, 2015.

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[42] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

[43] Yifei He, Siqi Zeng, Yuzheng Hu, Rui Yang, Tong Zhang, and Han Zhao. Mergebench: A benchmark for merging domain-specialized llms. arXiv preprint arXiv:2505.10833, 2025.

[44] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

[45] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP workshop BlackboxNLP: Analyzing and interpreting neural networks for NLP, pages 353–355, 2018.

[46] Shi-Ting Wang, Jin-Hua Zheng, Yuan Liu, Juan Zou, and Sheng-Xiang Yang. An extended fuzzy decision variables framework for solving large-scale multiobjective optimization problems. Information Sciences, 643:119221, 2023.

[47] Yingjie Zou, Yuan Liu, Juan Zou, Shengxiang Yang, and Jinhua Zheng. An evolutionary algorithm based on dynamic sparse grouping for sparse large scale multiobjective optimization. Information Sciences, 631:449–467, 2023.

[48] Tianmeng Hu, Yongzheng Cui, Biao Luo, and Ke Li. RIDER: 3D RNA inverse design with reinforcement learning-guided diffusion. In Proc. International Conference on Learning Representations, volume 2026, pages 79321–79346, 2026.

[49] Tianmeng Hu, Biao Luo, and Ke Li. CocoRNA: Collective RNA design with cooperative multi-agent reinforcement learning. In Proc. International Conference on Machine Learning, volume 306, pages 45495–45519, 2026.

[50] Wanli Ma, Tianmeng Hu, and Ke Li. Reflex: Rna ensemble generation via flexibility-calibrated stochastic bridge. arXiv preprint arXiv:2610.26261, 2026.

[51] Tianmeng Hu, Biao Luo, and Ke Li. TANGO: RNA topology and geometry co-design. arXiv preprint arXiv:2610.20277, 2026. URL https://arxiv.org/abs/2610.20277v1.

## A Appendix

## A.1 Additional Directional-Structure Evidence

Figure 10 extends the motivation analysis in Figure 1 to all evaluated model families, including ViT, RoBERTa, Llama, and Gemma. Each panel shows the task-vector directional energy distribution in the HO-GSVD-derived shared basis for a representative layer. Across architectures and task groups, the distributions consistently contain a low-dispersion bulk together with a high-dispersion tail of task-dominated directions. This recurring pattern supports the central premise of CUSD: arithmetic merging should first separate directions that are broadly common across tasks from directions whose energy is concentrated in one or a few task-specific models.

## A.2 HO-GSVD-Derived Shared-Basis Details

This appendix gives the regularized HO-GSVD-derived shared-basis construction used to compute V and Σ for each matrix-structured layer. The construction follows the pairwise Gram-quotient idea of HO-GSVD, with ridge regularization and symmetrization for neural-network task-vector matrices. The symmetrization step produces an orthonormal basis suitable for CUSD’s projection step, so CUSD does not rely on an exact classical HO-GSVD factorization. Algorithm 2 summarizes the implemented procedure.

Let $A _ { 1 } , \ldots , A _ { K }$ denote the task-vector matrices for one matrix-structured layer, where $A _ { i } \in \mathbb { R } ^ { m \times n }$ and all tasks share the same column dimension n. For a fixed $\pi > 0$ , define

$$
G _ { i } = A _ { i } ^ { \top } A _ { i } + \pi I _ { n } , \qquad i = 1 , \ldots , K .\tag{6}
$$

Each $G _ { i }$ is symmetric positive definite, so all quotient terms below are well defined. We form the regularized pairwise quotient average

$$
\widetilde { S } _ { \pi } = \frac { 1 } { K ( K - 1 ) } \sum _ { 1 \le i < j \le K } \left( G _ { j } ^ { - 1 } G _ { i } + G _ { i } ^ { - 1 } G _ { j } \right) .\tag{7}
$$

In practice, the products $G _ { j } ^ { - 1 } G _ { i }$ are computed by solving linear systems rather than explicitly forming matrix inverses. Since ${ \widetilde { S } } _ { \pi }$ is not guaranteed to be symmetric, CUSD uses

$$
S _ { \pi } = \frac { 1 } { 2 } \left( \widetilde { S } _ { \pi } + \widetilde { S } _ { \pi } ^ { \top } \right)\tag{8}
$$

and computes the orthogonal eigendecomposition

$$
\begin{array} { r } { S _ { \pi } = V \boldsymbol { \Lambda } V ^ { \top } , \qquad V ^ { \top } V = I _ { n } . } \end{array}\tag{9}
$$

The columns of $V = [ v _ { 1 } , \ldots , v _ { n } ]$ are sorted by descending eigenvalue. The symmetrization step preserves the pairwise quotient signal while producing an orthonormal basis for the projection step in CUSD.

For each task i and shared direction $v _ { k }$ , CUSD records the raw directional energy

$$
\sigma _ { i k } = \| A _ { i } v _ { k } \| _ { 2 } , \qquad \Sigma = [ \sigma _ { i k } ] \in \mathbb R ^ { K \times n } .\tag{10}
$$

These raw directional energies form the matrix Σ used by CUSD for dispersion scoring.

Proposition 1 (Well-defined shared-basis construction). For any $\pi > 0 ,$ , the CUSD shared-basis construction in Algorithm 2 is well defined and returns an orthonormal basis V.

Proof. For each task matrix $A _ { i } ,$ , the matrix $G _ { i } = A _ { i } ^ { \top } A _ { i } + \pi I _ { n }$ is symmetric positive definite because $\pi > 0$ . Hence all linear systems involving $G _ { i }$ in Eq. 7 are well defined. The symmetrized quotient matrix $S _ { \pi } = ( \widetilde { S } _ { \pi } + \widetilde { S } _ { \pi } ^ { \bar { \top } } ) / 2$ is real symmetric. By the spectral theorem, it admits an eigendecomposition $S _ { \pi } = V \Lambda V ^ { \top }$ with $V ^ { \top } V = I _ { n }$ . Thus the construction returns an orthonormal shared basis. □

The orthogonality of V is the key property used in the lossless decomposition proof in Appendix A.4.

![](images/557474b7d8391adb655d0441c1cd1748955774d3a532636eb30eafab599efe26.jpg)  
ViT-L/14 (block 20 / mlp.c\_proj): dispersion distribution

ViT-B/32: energy concentrates in high-dispersion direction  
![](images/f46244631a53db1d5e0512fea6e9f191278651e0f330cef8612f8da04267954a.jpg)  
ViT-L/14: energy concentrates in high-dispersion directions

![](images/41cbe6876eeb103e3760d67931754d953653aa40e9fa7560eb575a19dc4d5952.jpg)

![](images/6c4e71bfd431ce4587c186a0bf483451d4e16268704dbddac7b1e93b5a449eb5.jpg)

![](images/ca427ec2282c3d2235afff415a30e52cc354936ee0487c67683a77c43f9bd5f8.jpg)  
RoBERTa-Large (block 8 / ffn.down): dispersion distribution

RoBERTa-Base: energy concentrates in high-dispersion directions  
![](images/3f01b924ef780a9758900f254fc83f6b3bdde0387aaa456742bee7ba3836472c.jpg)

![](images/9b141427f119fdf6b11ead98288ff1b2b580dc9e7855c5841a410747ae68c625.jpg)

RoBERTa-Large: energy concentrates in high-dispersion directions  
![](images/7cee2dd380c8e5519785bc9f92727a625dffc37dbca7a5de20d7b63dad3d6b2b.jpg)

![](images/156ec2addafdcdc9d3188dbe59cf521f6e0fb42c8c3b4f8d93131b98419bde89.jpg)

Llama-3.1-8B: energy concentrates in high-dispersion directions  
![](images/7442ca58e87da2ea14721f2fcba1766cc15589782903e6946caf9c8014861f42.jpg)

Gemma-2-2b (block 20 / mlp.down): dispersion distribution  
![](images/f75a4e409e3bcecbef51e4086feb0ae9514745a1f4ee89bef559c31f8b911571.jpg)

Gemma-2-2b: energy concentrates in high-dispersion directions  
![](images/3014a98374d2fbba803fd7ec3ca3a25b80733a44579a9b2d9d1738c9ca30b388.jpg)

![](images/4dda25a406c09d3ed881f760c238f829ddd70bfd6308d2ba88f8c1351b67af13.jpg)

Gemma-2-9b: energy concentrates in high-dispersion directions  
![](images/b1d36acf4a28baf275e743540a8b6640275a90552ee9fc76de134eea4a7e08b0.jpg)  
Figure 10: Direction-level heterogeneity in task vectors. Directional energy structure across all evaluated model families.

```tcl
Algorithm 2 Regularized HO-GSVD-Derived Shared-Basis Construction
Require: $A _ { 1 } , \ldots , A _ { K }$ with $A _ { i } \in \mathbb { R } ^ { m _ { i } \times n } ; K \geq 2 ; \pi = 1 0 ^ { - 3 }$
Ensure: $V \in \mathbb { R } ^ { n \times n }$ (orthogonal shared basis); $\overline { { \Sigma } } \in \mathbb { R } ^ { K \times n }$ (directional-energy magnitudes)
1: Regularized Grams: $\bar { G } _ { i }  A _ { i } ^ { \top } A _ { i } + \pi I _ { n }$ for $i = 1 , \ldots , K$
2: Pairwise quotient average: ${ \widetilde { S } } _ { \pi } \gets \frac { 1 } { K ( K - 1 ) } \sum _ { 1 \leq i < j \leq K } ( \mathrm { s o l v e } ( G _ { j } , G _ { i } ) + \mathrm { s o l v e } ( G _ { i } , G _ { j } ) )$
3: Symmetrization: $S _ { \pi } \gets \frac { 1 } { 2 } \left( \widetilde { S } _ { \pi } + \widetilde { S } _ { \pi } ^ { \top } \right)$
4: Orthogonal eigendecomposition: $S _ { \pi } = V \Lambda V ^ { \top }$ ; sort columns of $V$ by descending eigenvalue
5: Magnitude extraction: ${ \sigma _ { i k } } \gets \| A _ { i } v _ { k } \| _ { 2 }$ for all $i \in [ K ]$ and $k \in [ n ]$
6: return V, $\Sigma = \left[ \sigma _ { i k } \right]$
```

## A.3 KDE Classification Details

CUSD selects the common–unique threshold from the empirical distribution of dispersion scores using kernel density estimation (KDE) [40]. This provides a layer-adaptive threshold without manually fixing a global cutoff.

Density Estimation Framework Given a set of n dispersion scores $\mathcal { D } = \{ d _ { k } \} _ { k = 1 } ^ { n }$ , we construct the density estimate ${ \hat { f } } ( d )$ using a Gaussian kernel:

$$
{ \hat { f } } ( d ) = { \frac { 1 } { n h } } \sum _ { k = 1 } ^ { n } { \frac { 1 } { \sqrt { 2 \pi } } } \exp \left( - { \frac { ( d - d _ { k } ) ^ { 2 } } { 2 h ^ { 2 } } } \right) .\tag{11}
$$

The bandwidth h is set by Scott’s rule for univariate data:

$$
h = \hat { \sigma } n ^ { - 1 / 5 } ,\tag{12}
$$

where $\hat { \sigma }$ denotes the sample standard deviation of $\mathcal { D } .$ . We evaluate ${ \hat { f } } ( d )$ over a uniform grid of $N = 1 { , } 0 0 0$ points covering $[ \operatorname* { m i n } _ { k } d _ { k } , \operatorname* { m a x } _ { k } d _ { k } ]$

Threshold Selection and Decision Rule The algorithm identifies candidate thresholds by locating local minima of the estimated density. Let V be the set of grid points corresponding to these local minima. If V is nonempty, CUSD selects the valley closest to the midpoint of the score range:

$$
\delta = \arg \operatorname* { m i n } _ { v \in \mathcal { V } } \left| v - \frac { \operatorname* { m i n } _ { k } d _ { k } + \operatorname* { m a x } _ { k } d _ { k } } { 2 } \right| .\tag{13}
$$

The final partition is

$$
{ \mathcal { C } } = \{ k : d _ { k } < \delta \} , \qquad { \mathcal { U } } = \{ k : d _ { k } \geq \delta \} .
$$

If no valid valley is found, CUSD applies a conservative no-valley policy and assigns all directions in that layer to the common set C.

## A.4 Lossless Orthogonal Decomposition

Proposition 2. Let $P c = V c V _ { c } ^ { \top }$ be the orthogonal projector onto the common directions, and let $\bar { P } _ { \mathcal { U } } = I - P _ { \mathcal { C } }$ . Set $C _ { i } = A _ { i } P _ { C }$ and $R _ { i } = A _ { i } P _ { \ l }$ . Then for every task i: $( i ) C _ { i } + R _ { i } = A _ { i } ;$ (ii) $\langle C _ { i } , R _ { i } \rangle _ { F } = 0 .$

Proof. The claim uses only the fact that the CUSD basis V is orthonormal. Let $V _ { \mathcal { C } }$ contain the columns of V indexed by ${ \dot { \mathcal { C } } } .$ Since $V ^ { \top } V = I$ , the selected columns also satisfy $V _ { \mathcal { C } } ^ { \top } V _ { \mathcal { C } } = I$ . Hence

$$
P _ { \mathcal { C } } ^ { \top } = ( V _ { \mathcal { C } } V _ { \mathcal { C } } ^ { \top } ) ^ { \top } = P _ { \mathcal { C } } , \qquad P _ { \mathcal { C } } ^ { 2 } = V _ { \mathcal { C } } ( V _ { \mathcal { C } } ^ { \top } V _ { \mathcal { C } } ) V _ { \mathcal { C } } ^ { \top } = P _ { \mathcal { C } } .
$$

Thus $P _ { C }$ is the orthogonal projector onto the span of the common directions. Since $P _ { \mathcal { U } } = I - P _ { \mathcal { C } }$ , we also have

$$
P u + P c = I , \qquad P u P c = ( I - P c ) P c = P c - P c ^ { 2 } = 0 .
$$

Therefore,

$$
C _ { i } + R _ { i } = A _ { i } P _ { \mathscr { C } } + A _ { i } P _ { \mathscr { U } } = A _ { i } ( P _ { \mathscr { C } } + P _ { \mathscr { U } } ) = A _ { i } .
$$

For orthogonality under the Frobenius inner product,

$$
\langle C _ { i } , R _ { i } \rangle _ { F } = \operatorname { t r } ( C _ { i } ^ { \top } R _ { i } ) = \operatorname { t r } \left( P c A _ { i } ^ { \top } A _ { i } P _ { \mathcal { U } } \right) .
$$

Using cyclic invariance of the trace,

$$
\mathrm { t r } \big ( P _ { \mathcal { C } } A _ { i } ^ { \top } A _ { i } P _ { \mathcal { U } } \big ) = \mathrm { t r } \big ( P _ { \mathcal { U } } P _ { \mathcal { C } } A _ { i } ^ { \top } A _ { i } \big ) = 0 ,
$$

because $P _ { \mathcal { U } } P _ { \mathcal { C } } = 0$ . Hence $\langle C _ { i } , R _ { i } \rangle _ { F } = 0$ for every task i.

## A.5 Experimental Setup

Models. We evaluate CUSD on 11 base models from four model families (ViT, RoBERTa, Llama, and Gemma), covering vision encoders, text encoders, and decoder-only LLMs. For vision experiments, we use CLIP ViT-B/32, ViT-B/16, and ViT-L/14 models [19, 41]. For Natural Language Understanding (NLU) experiments, we use RoBERTa-Base and RoBERTa-Large [42]. For generative LLM experiments, we use the MergeBench fine-tuned models [43] based on Llama-3.2-3B, Llama-3.1-8B, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-2b, and Gemma-2-9b [18, 44].

Datasets. For vision tasks, we use eight image classification datasets following prior work [32, 38]: Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, and SVHN. For NLU tasks, we follow prior model-merging evaluations [32] and use the eight GLUE tasks [45]: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, and RTE. For generative language tasks, we follow MergeBench [43] and evaluate code generation with Humaneval+ and MBPP+, mathematical reasoning with GSM8k, and multilingual understanding with M-MMLU, M-ARC, and M-Hellaswag.

Baselines. We evaluate CUSD with four representative data-free arithmetic-based merging methods: Task Arithmetic [11], TIES-Merging [13], DARE [14], and Consensus Merging [15]. For DARE and Consensus Merging, we use their Task Arithmetic instantiations. For each method, we report the plain merged model and the corresponding CUSD-preprocessed variant. We also include TSV-Merge [23] and Iso-Merging [16] for comparison with subspace-based merging methods. Fine-tuned task-specific models are included as reference upper bounds where available.

Implementation details. CUSD operates on matrix-structured linear-layer task vectors. We use absolute scoring for encoder models (ViT and RoBERTa) and relative scoring for decoder LLMs. Section 4.3 compares the dispersion scoring modes and analyzes sensitivity to the unique-residual coefficient β. Figure 3 provides empirical guidance on suitable $\beta$ ranges for the evaluated model families. Given fixed hyperparameters, CUSD uses only model weights during decomposition and merging. Baseline hyperparameters follow the original papers or MergeBench defaults and are kept identical between plain and CUSD-preprocessed runs.

## A.6 Main Results Details

Metrics. We report normalized scores relative to the corresponding fine-tuned task-specific references. For ViT and RoBERTa results in Table 1, task scores are first averaged in their original metric scale and the resulting average is normalized by the corresponding fine-tuned average. For LLMs, following [43], each benchmark score is normalized by the corresponding fine-tuned benchmark score before averaging.

Results on vision tasks. Tables 8–10 provide task-level results for the three ViT models used in the vision experiments. Each table reports the plain arithmetic merge and the corresponding CUSD merge on the eight image classification tasks. CUSD improves all four downstream operators on the three CLIP ViT backbones. The gains are largest on ViT-B/32 and ViT-B/16, where the average improvement across operators is about 12 normalized points, and remain positive on ViT-L/14, with an average gain of about 5 normalized points. The task-level results show a clear trade-off. For example, on ViT-L/14 with TIES-Merging, CUSD slightly reduces the already strong MNIST accuracy by 0.1 points, but improves EuroSAT, RESISC45, and DTD by 10.7, 8.5, and 9.2 points, respectively, raising the average task accuracy by 5.9 points. This pattern is consistent with CUSD reducing the influence of task-dominated directions that would otherwise bias the arithmetic merge, thereby improving the overall multi-task balance. TIES-Merging with CUSD achieves the best normalized average on all three ViT models. Detailed task-level results are provided in Appendix Tables 8–10.

Table 8: Multi-task performance of merged ViT-L/14 models on eight tasks.
<table><tr><td>Method</td><td>with CUSD</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>GTSRB</td><td>MNIST</td><td></td><td>RESISC45</td><td>SUN397</td><td>SVHN</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>92.5</td><td>84.2</td><td>91.0</td><td>99.2</td><td>99.7</td><td></td><td>99.7</td><td>83.6</td><td>98.1</td><td>93.5</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>× √</td><td>80.5</td><td>64.6</td><td>75.6</td><td>86.7</td><td>99.0</td><td>86.1</td><td></td><td>72.9</td><td>89.7</td><td>81.9</td></tr><tr><td></td><td>87.2 (+6.7)</td><td>71.9 (+7.2)</td><td>87.7 (+12.0)</td><td>92.6 (+6.0)</td><td>98.7</td><td>(-0.3)</td><td>94.5 (+8.4)</td><td>76.1 (+3.2)</td><td>87.2 (-2.5)</td><td>87.0 (+5.1)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>81.1</td><td>65.3</td><td>77.8</td><td>87.5</td><td>99.1</td><td></td><td>87.2</td><td>73.3</td><td>90.5</td><td>82.7</td></tr><tr><td>√</td><td>87.7 (+6.6)</td><td>74.5 (+9.2)</td><td>88.4 (+10.7)</td><td>95.0 (+7.5)</td><td>99.0</td><td>(-0.1) 95.7</td><td>(+8.5)</td><td>77.2 (+3.9)</td><td>91.2 (+0.8)</td><td>88.6 (+5.9)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>80.4</td><td>64.0</td><td>75.6</td><td>86.7</td><td>99.0</td><td></td><td>86.1</td><td>72.8</td><td>89.7</td><td>81.8</td></tr><tr><td>√</td><td>87.1 (+6.7)</td><td>71.6 (+7.6)</td><td>87.3 (+11.8)</td><td>92.7 (+6.0)</td><td>98.7</td><td>(-0.3) 94.4</td><td>(+8.3)</td><td>76.0 (+3.3)</td><td>87.2 (-2.6)</td><td>86.9 (+5.1)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>80.8</td><td>65.0</td><td>76.7</td><td>87.0</td><td>99.0</td><td></td><td>86.6</td><td>73.0</td><td>89.7</td><td>82.2</td></tr><tr><td>√</td><td>87.2 (+6.4)</td><td>71.4 (+6.4)</td><td>87.6 (+10.9)</td><td>92.4 (+5.4)</td><td>98.7</td><td>(-0.4) 94.3</td><td>(+7.7)</td><td>75.9 (+2.9)</td><td>86.8 (-2.9)</td><td>86.8 (+4.6)</td></tr></table>

Table 9: Multi-task performance of merged ViT-B/16 models on eight tasks.
<table><tr><td>Method</td><td>with CUSD</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>GTSRB</td><td>MNIST</td><td></td><td>RESISC45</td><td>SUN397</td><td>SVHN</td><td>Avg.</td></tr><tr><td rowspan="2">Fine-tuned</td><td></td><td>86.5</td><td>82.1</td><td>90.2</td><td>99.2</td><td>99.8</td><td></td><td>99.7</td><td>80.6</td><td>97.9</td><td>92.0</td></tr><tr><td>×</td><td>59.0</td><td>51.4</td><td>64.9</td><td>76.3</td><td>98.5</td><td></td><td>71.3</td><td>54.3</td><td>89.2</td><td>70.6</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>√</td><td>75.6 (+16.5)</td><td>62.1 (+10.7)</td><td>85.7 (+20.8)</td><td>87.1</td><td>(+10.9) 97.9</td><td>(-0.5)</td><td>89.2 (+17.9)</td><td>71.6 (+17.3)</td><td>84.7 (-4.5)</td><td>81.7 (+11.1)</td></tr><tr><td>X</td><td>60.7</td><td>52.3</td><td>68.5</td><td>77.1</td><td>98.8</td><td></td><td>73.4</td><td>56.6</td><td>90.2</td><td>72.2</td></tr><tr><td rowspan="2">TIES-Merging</td><td>√</td><td>74.7 (+14.0)</td><td>65.2 (+12.9)</td><td>86.8 (+18.3)</td><td>90.3 (+13.3)</td><td>98.4</td><td>(-0.4)</td><td>90.2 (+16.8)</td><td>70.7 (+14.1)</td><td>88.4 (-1.9)</td><td>83.1 (+10.9)</td></tr><tr><td>X</td><td>59.2</td><td>51.3</td><td>64.8</td><td>75.6</td><td>98.4</td><td></td><td>71.3</td><td>54.2</td><td>89.2</td><td>70.5</td></tr><tr><td rowspan="2">DARE</td><td>√</td><td>75.3 (+16.1)</td><td>62.1 (+10.8)</td><td>85.6 (+20.8)</td><td>87.1</td><td>(+11.4) 97.8</td><td>(-0.6)</td><td>89.2 (+18.0)</td><td>71.6 (+17.4)</td><td>84.9 (4.3)</td><td>81.7 (+11.2)</td></tr><tr><td></td><td>60.8</td><td>52.1</td><td>67.1</td><td>77.1</td><td>98.6</td><td></td><td></td><td>56.2</td><td>89.5</td><td></td></tr><tr><td>Consensus</td><td>X √</td><td>75.7 (+15.0)</td><td>62.1 (+9.9)</td><td>85.7 (+18.5)</td><td>86.7</td><td>(+9.6) 97.9</td><td>(-0.7)</td><td>72.9 89.5 (+16.6)</td><td>71.7 (+15.5)</td><td>84.4 (-5.1)</td><td>71.8 81.7 (+9.9)</td></tr></table>

Table 10: Multi-task performance of merged ViT-B/32 models on eight tasks.
<table><tr><td>Method</td><td>with CUSD</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>GTSRB</td><td>MNIST</td><td>RESISC45</td><td>SUN397</td><td>SVHN</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>78.0</td><td>79.4</td><td>88.9</td><td>98.7</td><td>99.7</td><td>99.7</td><td>78.3</td><td>97.5</td><td>90.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>× √</td><td>46.4</td><td>48.5</td><td>64.2</td><td>71.4</td><td>97.2</td><td>64.1</td><td>46.2</td><td>79.3</td><td>64.7</td></tr><tr><td></td><td>65.1 (+18.8)</td><td>60.4 (+12.0)</td><td>81.9 (+17.7)</td><td>79.8 (+8.5)</td><td>96.5 (-0.7)</td><td>83.4 (+19.3)</td><td>65.5 (+19.4)</td><td>79.3 (0.0)</td><td>76.5 (+11.9)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X √</td><td>48.8</td><td>49.7</td><td>66.9</td><td>72.8</td><td>97.7</td><td>66.7</td><td>49.0</td><td>81.3</td><td>66.6</td></tr><tr><td></td><td>65.6 (+16.8)</td><td>63.7 (+14.0)</td><td>83.8 (+16.9)</td><td>83.7 (+10.9)</td><td>96.8 (-0.9)</td><td>87.1 (+20.4)</td><td>66.1 (+17.1)</td><td>79.8 (-1.5)</td><td>78.3 (+11.7)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>45.9</td><td>48.6</td><td>64.2</td><td>71.4</td><td>97.2</td><td>63.7</td><td>45.9</td><td>79.1</td><td>64.5</td></tr><tr><td>√</td><td>64.6 (+18.7)</td><td>60.3 (+11.7)</td><td>81.8 (+17.6)</td><td>79.9 (+8.5)</td><td>96.5 (-0.7)</td><td>83.2 (+19.5)</td><td>65.5 (+19.6)</td><td>79.1 (-0.1)</td><td>76.4 (+11.9)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>48.2</td><td>49.6</td><td>65.7</td><td>72.2</td><td>97.3</td><td>65.7</td><td>48.4</td><td>80.0</td><td></td></tr><tr><td>√</td><td>65.3 (+17.1)</td><td>60.5 (+10.9)</td><td>82.0 (+16.3)</td><td>79.6 (+7.4)</td><td>96.4 (-0.9)</td><td>83.5 (+17.8)</td><td>65.8 (+17.4)</td><td>79.2 (-0.8)</td><td>65.9 76.5 (+10.7)</td></tr></table>

Results on NLU tasks. Tables 11 and 12 provide task-level GLUE results for RoBERTa-Base and RoBERTa-Large. Each table reports the plain merge and the corresponding CUSD merge for all four downstream arithmetic operators. CUSD also improves every merging operator on both RoBERTa-Base and RoBERTa-Large. The effect is especially strong for operators whose plain merges suffer from severe degradation, such as TIES-Merging on RoBERTa-Large, where the normalized average increases from 47.3 to 69.8. A striking example is STS-B, the semantic textual similarity task: all plain merges yield negative normalized scores, whereas CUSD turns them positive across every operator in both RoBERTa-Base and RoBERTa-Large. These results indicate that common–unique separation is not limited to vision encoders and can also recover lost performance in discriminative language models. Detailed GLUE results are reported in Appendix Tables 12 and 11.

Table 11: Multi-task performance of merged RoBERTa-Base models on the 8-task GLUE benchmark.
<table><tr><td>Method</td><td>with CUSD</td><td>CoLA</td><td>MNLI</td><td>MRPC</td><td></td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST-2</td><td>STS-B</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>55.1</td><td>87.5</td><td>88.0</td><td></td><td></td><td>89.4</td><td>67.5</td><td>93.8</td><td>88.3</td><td>82.8</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X √</td><td>0.0</td><td>64.6</td><td>79.5</td><td></td><td></td><td>71.3</td><td>49.1</td><td>71.1</td><td>-56.8</td><td>44.3</td></tr><tr><td></td><td>6.6 (+6.6)</td><td>63.6 (-1.0)</td><td>78.2</td><td>(-1.3) 74.1 73.2</td><td>(-1.4)</td><td>37.6 (-33.7)</td><td>50.9 (+1.8)</td><td>87.7 (+16.6)</td><td>55.9 (+112.6)</td><td>56.8 (+12.5)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X √</td><td>0.0</td><td>64.9</td><td>79.0 (-2.8)</td><td></td><td></td><td>76.4 36.6</td><td>46.2</td><td>65.9</td><td>-67.2</td><td>42.3</td></tr><tr><td></td><td>6.6 (+6.6)</td><td>62.1</td><td>77.9</td><td>(-1.1) 63.6</td><td>(-9.6)</td><td>(-39.8)</td><td>53.4 (+7.2)</td><td>90.3 (+24.3)</td><td>68.4 (+135.6)</td><td>57.4 (+15.1)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>4.6</td><td>64.6</td><td>79.4</td><td></td><td>75.6</td><td>71.7</td><td>48.7</td><td>71.3</td><td>-58.6</td><td>44.7</td></tr><tr><td>√</td><td>6.6 (+1.9)</td><td>63.0</td><td>78.1</td><td>(-1.3) 73.6</td><td>(-2.1) 37.4</td><td>(-34.3)</td><td>50.9 (+2.2)</td><td>87.7 (+16.4)</td><td>54.8 (+113.3)</td><td>56.5 (+11.8)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>4.6</td><td>66.9</td><td>78.0</td><td></td><td>78.6</td><td>71.8</td><td>46.6</td><td>75.5</td><td>-56.3</td><td>45.7</td></tr><tr><td>√</td><td>6.6 (+1.9)</td><td>59.4</td><td>77.3</td><td>(-0.7)</td><td>66.6 (-12.0)</td><td>33.3 (-38.5)</td><td>53.1 (+6.5)</td><td>87.7 (+12.3)</td><td>61.6 (+117.9)</td><td>55.7 (+10.0)</td></tr></table>

Table 12: Multi-task performance of merged RoBERTa-Large models on the 8-task GLUE benchmark.
<table><tr><td>Method</td><td>with CUSD</td><td>CoLA</td><td>MNLI</td><td>MRPC</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST-2</td><td>STS-B</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td>一</td><td>59.1</td><td>90.5</td><td>89.0</td><td>94.5</td><td>90.5</td><td>70.0</td><td>96.6</td><td>91.7</td><td>85.2</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X √</td><td>0.2</td><td>75.7</td><td>48.6</td><td>69.5</td><td>82.1</td><td>58.8</td><td>85.1</td><td>-22.4</td><td>49.7</td></tr><tr><td></td><td>9.1 (+8.8)</td><td>72.2 (-3.4)</td><td>27.9 (-20.8)</td><td>83.5 (+14.0)</td><td>59.5 (-22.7)</td><td>70.8 (+11.9)</td><td>92.7 (+7.6)</td><td>62.0 (+84.3)</td><td>59.7 (+10.0)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>-1.4</td><td>71.0</td><td>56.3</td><td>53.3</td><td>84.6</td><td>54.5</td><td>72.1</td><td>-67.9</td><td>40.3</td></tr><tr><td>√</td><td>5.7 (+7.1)</td><td>74.5 (+3.5)</td><td>29.1 (-27.2)</td><td>85.3 (+32.0)</td><td>58.6 (-26.0)</td><td>70.0 (+15.5)</td><td>92.6 (+20.4)</td><td>60.3 (+128.2)</td><td>59.5 (+19.2)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>2.4</td><td>75.6</td><td>48.8</td><td>70.2</td><td>82.1</td><td>59.2</td><td>85.7</td><td>-23.2</td><td>50.1</td></tr><tr><td>√</td><td>9.6 (+7.2)</td><td>71.7 (-3.9)</td><td>26.9 (-21.9)</td><td>82.9 (+12.7)</td><td>59.1 (-23.0)</td><td>69.7 (+10.5)</td><td>92.4 (+6.8)</td><td>61.3 (+84.5)</td><td>59.2 (+9.1)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>2.4</td><td>76.3</td><td>40.1</td><td>76.8</td><td>80.3</td><td>57.8</td><td>89.1</td><td>-0.9</td><td>52.7</td></tr><tr><td>√</td><td>9.9 (+7.5)</td><td>71.7 (-4.6)</td><td>27.9 (-12.2)</td><td>83.4 (+6.6)</td><td>59.9 (-20.4)</td><td>70.8 (+13.0)</td><td>92.7 (+3.6)</td><td>61.1 (+62.1)</td><td>59.7 (+6.9)</td></tr></table>

Results on generative LLMs. Tables 13–18 provide task-level results for the six LLMs: Llama-3.2- 3B, Llama-3.1-8B, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-2B, and Gemma-2-9B. Scores are normalized by the corresponding fine-tuned task-specific models before averaging. On generative LLMs, CUSD improves most settings, although the margins are more model-dependent than on encoder models. The strongest gains appear on Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and the Gemma models. Llama-3.1-8B is the least responsive case: plain arithmetic merges already reach normalized averages around 94, leaving little headroom, and the only negative change is a small 0.3-point decrease for DARE. In the evaluated LLM settings, partial unique-residual scaling is often beneficial; this behavior is examined in the β ablation (Section 4.3) and the analysis of unique directions across models (Section 4.4). Full task-level LLM results are provided in the Appendix Tables 13–18.

Table 13: Performance of merged Llama-3.2-3B models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks; normalized scores and their average are reported.
<table><tr><td>Method</td><td>with CUSD</td><td>Humaneval+</td><td>MBPP+</td><td>GSM8k</td><td></td><td>M-MMLU</td><td>M-ARC</td><td>M-HellaSwag</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>45.7</td><td>59.4</td><td>39.2</td><td></td><td>87.0</td><td>93.4</td><td>93.0</td><td>69.6</td></tr><tr><td>√</td><td>45.7 (0.0)</td><td>52.8 (-6.6)</td><td>45.8 (+6.6)</td><td>86.0 (-1.0)</td><td></td><td>94.0 (+0.6)</td><td>93.2 (+0.2)</td><td>69.6 (0.0)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>46.1</td><td>58.5</td><td>37.9</td><td></td><td>87.2</td><td>93.1</td><td>92.6</td><td>69.2</td></tr><tr><td>√</td><td>45.2 (-0.9)</td><td>53.3 (-5.2)</td><td>47.1 (+9.2)</td><td></td><td>85.8 (-1.4)</td><td>94.0 (+0.9)</td><td>93.0 (+0.4)</td><td>69.7 (+0.5)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>47.3</td><td>53.8</td><td>38.8</td><td>87.2</td><td></td><td>93.3</td><td>93.0</td><td>68.9</td></tr><tr><td>√</td><td>47.9 (+0.6)</td><td>57.1 (+3.3)</td><td>42.4 (+3.6)</td><td>86.8 (-0.4)</td><td></td><td>94.2 (+0.9)</td><td>93.2 (+0.2)</td><td>70.3 (+1.4)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>47.9</td><td>59.9</td><td>24.4</td><td></td><td>87.8</td><td>91.5</td><td>92.8</td><td>67.4</td></tr><tr><td>√</td><td>52.1 (+4.2)</td><td>51.9 (-8.0)</td><td>47.1 (+22.7)</td><td></td><td>85.7 (-2.1)</td><td>93.9 (+2.4)</td><td>93.1 (+0.3)</td><td>70.6 (+3.2)</td></tr></table>

Table 14: Performance of merged Llama-3.1-8B models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks; normalized scores and their average are reported.
<table><tr><td>Method</td><td>with CUSD</td><td>Humaneval+</td><td>MBPP+</td><td>GSM8k</td><td>M-MMLU</td><td>M-ARC</td><td>M-HellaSwag</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>82.5</td><td>100.0</td><td>93.7</td><td>95.5</td><td>89.7</td><td>100.8</td><td>93.7</td></tr><tr><td>√</td><td>82.6 (+0.1)</td><td>102.4 (+2.4)</td><td>91.3 (-2.4)</td><td>96.1 (+0.6)</td><td>90.9 (+1.2)</td><td>101.3 (+0.5)</td><td>94.1 (+0.4)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>82.6</td><td>100.8</td><td>94.2</td><td>95.2</td><td>88.8</td><td>100.4</td><td>93.7</td></tr><tr><td>√</td><td>83.0 (+0.4)</td><td>104.0 (+3.2)</td><td>88.2 (-6.0)</td><td>97.1 (+1.9)</td><td>91.2 (+2.4)</td><td>101.2 (+0.8)</td><td>94.1 (+0.4)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>84.6</td><td>101.2</td><td>92.7</td><td>95.1</td><td>90.3</td><td>100.7</td><td>94.1</td></tr><tr><td>√</td><td>83.2 (-1.4)</td><td>100.4 (-0.8)</td><td>91.2 (-1.5)</td><td>96.0 (+0.9)</td><td>91.0 (+0.7)</td><td>101.2 (+0.5)</td><td>93.8 3 (-0.3)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>83.8</td><td>99.2</td><td>89.5</td><td>97.1</td><td>92.3</td><td>101.1</td><td>93.8</td></tr><tr><td>√</td><td>83.4 (-0.4)</td><td>100.8 (+1.6)</td><td>91.9 (+2.4)</td><td>96.2 (-0.9)</td><td>90.4 (-1.9)</td><td>100.9 (-0.2)</td><td>93.9 (+0.1)</td></tr></table>

Table 15: Performance of merged Llama-3.2-3B-Instruct models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks; normalized scores and their average are reported.
<table><tr><td>Method</td><td></td><td>with CUSD Humaneval+</td><td>MBPP+</td><td>GSM8k</td><td>M-MMLU</td><td>M-ARC</td><td></td><td>M-HellaSwag</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td>一</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>55.5</td><td>46.5</td><td>64.2</td><td>85.5</td><td>96.2</td><td></td><td>93.3</td><td>73.5</td></tr><tr><td>√</td><td>61.3 (+5.8)</td><td>54.0 (+7.5)</td><td>62.4 (-1.8)</td><td>86.4 (+0.9)</td><td>97.2 (+1.0)</td><td></td><td>93.6 (+0.3)</td><td>75.8 (+2.3)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>53.8</td><td>44.3</td><td>63.9</td><td>85.1</td><td>95.9</td><td></td><td>93.4</td><td>72.7</td></tr><tr><td>√</td><td>58.0 (+4.2)</td><td>57.5 (+13.2)</td><td>61.1 (-2.8)</td><td>86.4 (+1.3)</td><td>96.4 (+0.5)</td><td></td><td>93.6 (+0.2)</td><td>75.5 (+2.8)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>51.7</td><td>50.0</td><td>64.6</td><td>85.0</td><td>95.9</td><td></td><td>93.3</td><td>73.4</td></tr><tr><td>√</td><td>57.6 (+5.9)</td><td>56.6 (+6.6)</td><td>62.4 (-2.2)</td><td>85.9 (+0.9)</td><td>96.3 (+0.4)</td><td></td><td>93.4 (+0.1)</td><td>75.4 (+2.0)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>57.8</td><td>59.3</td><td>63.8</td><td>86.3</td><td>96.5</td><td></td><td>93.8</td><td>76.2</td></tr><tr><td>√</td><td>60.2 (+2.4)</td><td>60.2 (+0.9)</td><td>62.5 (-1.3)</td><td>86.5 (+0.2)</td><td>96.4 (-0.1)</td><td></td><td>93.7 (-0.1)</td><td>76.6 (+0.4)</td></tr></table>

Table 16: Performance of merged Llama-3.1-8B-Instruct models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks; normalized scores and their average are reported.
<table><tr><td>Method</td><td>with CUSD Humaneval+</td><td></td><td>MBPP+</td><td>GSM8k</td><td>M-MMLU</td><td>M-ARC</td><td>M-HellaSwag</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td>一</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>55.5</td><td>87.9</td><td>87.6</td><td>84.2</td><td>87.3</td><td>92.4</td><td>82.5</td></tr><tr><td>√</td><td>70.7 (+15.2)</td><td>91.4 (+3.5)</td><td>83.8 (-3.8)</td><td>88.3 (+4.1)</td><td>90.7 (+3.4)</td><td>94.0 (+1.6)</td><td>86.5 (+4.0)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>56.9</td><td>87.9</td><td>87.8</td><td>84.5</td><td>86.7</td><td>92.1</td><td>82.6</td></tr><tr><td>√</td><td>68.8 (+11.9)</td><td>92.6 (+4.7)</td><td>85.2 (-2.6)</td><td>87.3 (+2.8)</td><td>89.6 (+2.9)</td><td>93.8 (+1.7)</td><td>86.2 (+3.6)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>59.7</td><td>87.5</td><td>87.9</td><td>83.8</td><td>87.3</td><td>92.3</td><td>83.1</td></tr><tr><td>√</td><td>71.7 (+12.0)</td><td>91.8 (+4.3)</td><td>84.9 (-3.0)</td><td>86.8 (+3.0)</td><td>89.5 (+2.2)</td><td>93.7 (+1.4)</td><td>86.4 (+3.3)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>68.6</td><td>93.4</td><td>85.5</td><td>86.6</td><td>89.2</td><td>93.7</td><td>86.2</td></tr><tr><td>√</td><td>73.7 (+5.1)</td><td>94.9 (+1.5)</td><td>82.7 (-2.8)</td><td>88.7 (+2.1)</td><td>91.2 (+2.0)</td><td>94.0 (+0.3)</td><td>87.5 (+1.3)</td></tr></table>

Table 17: Performance of merged Gemma-2-2b models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks; normalized scores and their average are reported.
<table><tr><td>Method</td><td>with CUSD Humaneval+</td><td></td><td>MBPP+</td><td>GSM8k</td><td>M-MMLU</td><td>M-ARC</td><td>M-HellaSwag</td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>80.6</td><td>94.9</td><td>75.0</td><td>106.8</td><td>104.0</td><td>104.0</td><td>94.2</td></tr><tr><td>√</td><td>83.7 (+3.1)</td><td>98.0 (+3.1)</td><td>76.4 (+1.4)</td><td>107.4 (+0.6)</td><td>104.5 (+0.5)</td><td>104.1 (+0.1)</td><td>95.7 (+1.5)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>80.4</td><td>90.3</td><td>79.5</td><td>107.1</td><td>103.3</td><td>103.8</td><td>94.1</td></tr><tr><td>√</td><td>90.8 (+10.4)</td><td>94.4 (+4.1)</td><td>83.2 (+3.7)</td><td>107.6 (+0.5)</td><td>104.0 (+0.7)</td><td>103.9 (+0.1)</td><td>97.3 (+3.2)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>80.6</td><td>91.8</td><td>76.7</td><td>107.6</td><td>104.0</td><td>104.2</td><td>94.1</td></tr><tr><td>√</td><td>87.3 (+6.7)</td><td>94.9 (+3.1)</td><td>78.6 (+1.9)</td><td>108.1 (+0.5)</td><td>104.1 (+0.1)</td><td>104.1 (-0.1)</td><td>96.2 (+2.1)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>60.5</td><td>94.9</td><td>68.8</td><td>103.2</td><td>102.7</td><td>102.8</td><td>88.8</td></tr><tr><td>√</td><td>77.5 (+17.0)</td><td>90.3 (-4.6)</td><td>75.8 (+7.0)</td><td>107.2 (+4.0)</td><td>104.3 (+1.6)</td><td>103.7 (+0.9)</td><td>93.1 (+4.3)</td></tr></table>

Table 18: Performance of merged Gemma-2-9b models. The task-specific checkpoints are fine-tuned for coding, math, and multilingual tasks;normalized scores and their average are reported.
<table><tr><td>Method</td><td>with CUSD Humaneval+</td><td></td><td>MBPP+</td><td>GSM8k</td><td>M-MMLU</td><td>M-ARC</td><td>M-HellaSwag</td><td></td><td>Avg.</td></tr><tr><td>Fine-tuned</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td></tr><tr><td rowspan="2">Task Arithmetic</td><td>X</td><td>73.5</td><td>102.8</td><td>105.0</td><td>96.3</td><td>83.4</td><td></td><td>90.7</td><td>92.0</td></tr><tr><td>√</td><td>72.4 (-1.1)</td><td>106.8 (+4.0)</td><td>101.3 (-3.7)</td><td>99.5 (+3.2)</td><td>90.7 (+7.3)</td><td></td><td>98.1 (+7.4)</td><td>94.8 (+2.8)</td></tr><tr><td rowspan="2">TIES-Merging</td><td>X</td><td>71.5</td><td>102.0</td><td>105.0</td><td>96.2</td><td>81.8</td><td></td><td>90.3</td><td>91.1</td></tr><tr><td>√</td><td>71.5 (0.0)</td><td>109.2 (+7.2)</td><td>99.6 (-5.4)</td><td>100.3 (+4.1)</td><td>91.5 (+9.7)</td><td>99.6</td><td>(+9.3)</td><td>95.3 (+4.2)</td></tr><tr><td rowspan="2">DARE</td><td>X</td><td>72.8</td><td>99.6</td><td>105.5</td><td>96.0</td><td>82.9</td><td></td><td>90.3</td><td>91.2</td></tr><tr><td>√</td><td>72.1 (-0.7)</td><td>106.8 (+7.2)</td><td>101.7 (-3.8)</td><td>99.5 (+3.5)</td><td>90.7 (+7.8)</td><td>98.2</td><td>(+7.9)</td><td>94.8 (+3.6)</td></tr><tr><td rowspan="2">Consensus</td><td>X</td><td>80.7</td><td>105.2</td><td>100.2</td><td>99.3</td><td>88.6</td><td></td><td>93.7</td><td>94.6</td></tr><tr><td>√</td><td>74.4 (-6.3)</td><td>107.6 (+2.4)</td><td>100.6 (+0.4)</td><td>100.3 (+1.0)</td><td>91.7 (+3.1)</td><td></td><td>98.5 (+4.8)</td><td>95.5 (+0.9)</td></tr></table>

![](images/3ddf6f7e7236af66590f6c8c696bfadde2f1091393dd73a299f964a81f3b1e6c.jpg)  
Figure 11: ViT-B/16 directional singular-value heatmaps. Each panel shows $\log _ { 1 0 } \sigma _ { i , k }$ for one layer component, with directions sorted by energy.

## A.7 Dispersion Scoring Analysis

Figure 11 and Figure $1 2$ provide additional layer heatmaps for the representative ViT and Llama settings used in the dispersion-scoring analysis.

## A.8 Additional Hyperparameter Sensitivity Analysis

Figure 13 reports $\beta$ sensitivity separately for each downstream merging method. This complements the averaged sweep in Figure 3 and checks whether the observed trend is consistent across Task Arithmetic, TIES-Merging, DARE, and Consensus Merging.

![](images/3ed2a65dd0941ffa7ad3b202b22dd1ed8e0d68bdac22d803c3418e237b998314.jpg)

![](images/bf4db3952b2ed341638a7b48c11121c21b100eb89a283ccbbcef0d52aac8c049.jpg)

![](images/f64bb557b7b163fa47ab4381991f9e09eb23d28646858b8dae9cdd40c52d515d.jpg)  
Figure 13: Per-method sensitivity to $\beta$ across downstream arithmetic-based merging operators.

![](images/e386b7e4f105bdd70a1585f3fa9ba43ce59da31c9e846eeb02dc1f31d555478e.jpg)  
Figure 12: Llama-3.2-3B directional singular-value heatmaps. Each panel shows $\log _ { 1 0 } \sigma _ { i , k }$ for one layer component, with directions sorted by energy.

Table 19: Joint sensitivity to the Task Arithmetic scale λ and unique-residual coefficient $\beta$ on ViT-B/32. We report average accuracy (%) from a post-hoc sweep on the test sets. The final column reports plain TA at the same $\lambda ;$ bold entries outperform this baseline. <sup>†</sup> denotes the hyperparameter configuration used in the original submission.
<table><tr><td> $\lambda \backslash \beta$ </td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td></td><td>1.0 Plain TA</td></tr><tr><td>0.1</td><td></td><td>57.4361.74</td><td></td><td>63.33 62.22</td><td>59.15</td><td>53.95</td><td>47.22</td><td>39.87</td><td>33.62</td><td>28.21</td><td>23.91</td><td>61.74</td></tr><tr><td>0.2</td><td></td><td>66.62 69.0769.3667.3562.75</td><td></td><td></td><td></td><td>56.56</td><td>49.11</td><td>41.37</td><td>34.97</td><td>29.3124.91</td><td></td><td>69.36</td></tr><tr><td>0.3</td><td></td><td>72.4973.8772.9470.0564.5957.76</td><td></td><td></td><td></td><td></td><td>49.88</td><td>42.29</td><td>35.91</td><td>30.3625.85</td><td></td><td>70.05</td></tr><tr><td>0.4</td><td>76.48†76.5074.65 70.48 64.65 57.4349.84 42.52 36.31 31.18 26.94</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.65†</td></tr><tr><td>0.5</td><td></td><td>77.64 77.0274.15 69.7063.4556.2749.0342.3636.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>831.94 27.80</td><td></td><td>56.27</td></tr><tr><td>0.6</td><td></td><td></td><td></td><td></td><td></td><td>77.33 75.82 72.47 67.53 61.25 54.54 47.9741.74 36.49 32.35 28.43</td><td></td><td></td><td></td><td></td><td></td><td>47.97</td></tr><tr><td>0.7</td><td></td><td></td><td></td><td></td><td>75.53 73.54 69.82 64.63 58.9052.5646.4540.91 36.3632.47 28.86</td><td></td><td></td><td></td><td></td><td></td><td></td><td>40.91</td></tr><tr><td>0.8</td><td></td><td></td><td></td><td></td><td>72.48 70.22 66.47 61.61 56.11 50.44 45.04 40.05 36.13 32.33 28.99</td><td></td><td></td><td></td><td></td><td></td><td></td><td>36.13</td></tr><tr><td>0.9</td><td></td><td></td><td></td><td></td><td>68.78 66.37 62.71 58.20 53.2548.21 43.42 39.15 35.57 32.20 29.06</td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.20</td></tr><tr><td>1.0</td><td></td><td></td><td></td><td></td><td>64.46 62.11 58.83 54.94 50.45 46.04 41.90 38.28 34.92 31.87 28.98</td><td></td><td></td><td></td><td></td><td></td><td></td><td>28.98</td></tr></table>

## A.9 Additional Per-Block Interference Analysis

For a merged task vector $\tau _ { m } ^ { ( \ell ) }$ at layer ℓ and task-specific task vectors $\{ \tau _ { i } ^ { ( \ell ) } \} _ { i = 1 } ^ { K }$ , we measure a normalized interference proxy (NI) as

$$
\mathrm { N I } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \Vert \tau _ { m } - \tau _ { i } \Vert _ { F } ^ { 2 } } { \Vert \tau _ { i } \Vert _ { F } ^ { 2 } } .\tag{14}
$$

For the per-block plots, we aggregate the decomposed matrix-structured layers in each block. We use this normalized distance as a parameter-space proxy for interference: a larger value means that the

merged update deviates more from the task-specific updates. Figure 14, Figure 15, and Figure 16 provide the per-block interference analysis for the remaining merging methods on ViT models.

![](images/a382ad6819ae1868cced4361dc4f912e4de2b10524e8532cb8c0eb794f3824ce.jpg)  
(a) ViT-B/32.

![](images/85e28c49e26cbbd57cb02d42bb0776e2c62e3e6bbffe7c3a03a7bb39611d9774.jpg)  
(b) ViT-B/16.

![](images/f2ee8043443bd254ec1141a66a39cc4e71d2ad956782d35e350b48000be2544c.jpg)  
(c) ViT-L/14.

Figure 14: Per-block interference analysis for Task Arithmetic on ViT models.  
![](images/69a97d42095889257d4ca4e634bf69631f81bacb5960cceb73fa1e321cab9f8b.jpg)  
(a) ViT-B/32.

![](images/e4088e5b00a52678fc1131311914b110b2537cbe393bb8b5528b037fd8f14b5f.jpg)  
(b) ViT-B/16.

![](images/ae3aef84dc1742e4e7705586cec3c020dc8d4d29b685fe99bfa886749b6f7d31.jpg)  
(c) ViT-L/14.

Figure 15: Per-block interference analysis for TIES-Merging on ViT models.  
![](images/d247bf990cce149809a448742a067d86c4bc3d708472f0120ca9cc0b7bb4b298.jpg)  
(a) ViT-B/32.

![](images/c7ce96e0032fc497ea77e5eae11f8ddb5e5a2ac6a746fd3b4c5026171b20df08.jpg)  
(b) ViT-B/16.

![](images/480b83bccfdb5a4d573b821669a9aef1cba75984d3ac173160bde1af6b30113f.jpg)  
(c) ViT-L/14.  
Figure 16: Per-block interference analysis for Consensus Merging on ViT models.

## A.10 Limitations

We acknowledge the potential limitations of our work. Performance gains vary across methods and models, and SVD-based baselines remain stronger in some evaluated settings. CUSD uses only model weights during decomposition and merging, but selecting $\beta$ can require validation data. CUSD is evaluated in the standard homogeneous-merging setting, where all task-specific checkpoints are fine-tuned from the same base architecture. It does not address heterogeneous merging across different architectures. Our evaluation also focuses on dense models at the ViT, RoBERTa, Llama, and Gemma scales considered in this work; substantially larger checkpoints, such as 70B-class LLMs, and Mixture-of-Experts architectures are left for future work. Future work could also explore multiobjective optimization for balancing common aggregation and unique retention [46, 47]. We further plan to investigate merging compatible models for RNA design [48–51].

## A.11 Broader Impacts

Positive Societal Impacts By improving data-free arithmetic model merging, CUSD has the potential for several positive societal impacts. Because CUSD uses only model weights during decomposition and merging, it can support privacy-sensitive and federated settings where checkpoints are produced by different owners and the underlying data cannot be shared, e.g., when fine-tuned models from hospitals, financial institutions, or independent research groups must be combined into a single multi-task model without disclosing proprietary or personal data. By more reliably merging task-specific checkpoints into one multi-task model, CUSD reduces the need to deploy and serve a separate fine-tuned model per task, which lowers storage, memory, inference compute, and energy consumption, and is particularly beneficial in resource-constrained environments such as edge devices and small organizations. Because CUSD operates as a pre-merging module that augments existing arithmetic operators (Task Arithmetic, TIES, DARE, Consensus) rather than replacing them, practitioners can adopt it without retraining or redesigning their pipelines, which broadens access to high-quality multi-task model assembly across the ViT, RoBERTa, Llama, and Gemma families considered in this work and similar architectures. Together, these properties can accelerate research and deployment of capable multi-task models in domains such as scientific discovery, medical decision support, multilingual NLP, and educational tools, and can make advanced AI capabilities available to a broader range of users and institutions.

Negative Societal Impacts Despite these benefits, the advances presented also carry potential negative societal impacts. Improved data-free merging makes it easier to combine the capabilities of independently distributed checkpoints, which could be leveraged for malicious purposes, such as merging a safety-aligned model with a checkpoint fine-tuned for harmful or unsafe behavior in order to circumvent alignment, or assembling specialized capabilities for generating disinformation, sophisticated phishing content, or deceptive media at scale. Because CUSD merges models without inspecting their training data, biases, unsafe behaviors, or memorized content present in any constituent checkpoint can be inherited and aggregated in the merged model, potentially leading to unfair or discriminatory outcomes when these models are deployed in decision-making pipelines. The reduced cost of obtaining strong multi-task models from publicly available checkpoints might also lower the barrier to repurposing released models in ways that their original developers did not intend, and may contribute to the concentration of capabilities in the hands of actors with the largest pools of high-quality fine-tuned weights. Mitigation strategies—such as releasing merging tools alongside responsible-use guidelines, evaluating merged models for safety, fairness, and bias before deployment, developing detectors for content produced by merged models, and applying safety filters or alignment procedures after merging—are important to address these risks.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The assertions made in the abstract and introduction are fully supported by our methodology and empirical findings.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to Appendix A.10 to see our related discussion.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Please refer to Appendix A.4 to see our theoretical proof.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We conduct detailed descriptions of our experimental setup in Section 4. We will also release the source code to further facilitate reproducibility.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: Please refer to the source code that we provided in the supplementary materials, and it will be publicly available.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Please refer to Section 4 for detailed empirical settings.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper does not report error bars, but experiments were conducted across multiple model architectures and sizes to validate the robustness of the method.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We report the time consumption of our proposed method and computed resources used in the experimental setup section.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in this paper fully conforms to the NeurIPS Code of Ethics in every respect.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss both the potential positive and negative societal impacts of our work in Appendix A.11.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The methods primarily discussed in our paper do not involve data or models with a high risk of misuse.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have appropriately cited the papers relevant to our work.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Please refer to the source code that we provided in the supplementary materials, and it will be publicly available.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [N/A]

Justification: The core methodology of this paper does not involve LLMs as an important, original, or non-standard component. LLMs from the Llama and Gemma families appear only as experimental subjects on which the proposed merging method is evaluated.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.