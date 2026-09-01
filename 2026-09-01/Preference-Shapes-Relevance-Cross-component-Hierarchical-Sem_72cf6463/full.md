# Preference Shapes Relevance: Cross-component Hierarchical Semantic Alignment for Personalized Generative Retrieval

Gaoming Zhang<sup>1</sup> Angqing Jiang<sup>1</sup> Jianchun Song<sup>2</sup> Kena Qi<sup>2</sup>

Dayao Chen<sup>2</sup> Wei Lin<sup>2</sup> Defu Lian<sup>1</sup>\*

<sup>1</sup>University of Science and Technology of China

<sup>2</sup>Meituan

{zzzgm, philipgaq}@mail.ustc.edu.cn liandefu@ustc.edu.cn {songjianchun, qikena, chendayao, linwei31}@meituan.com

## Abstract

Generative Retrieval (GR) has emerged as a promising paradigm by mapping queries directly to Semantic IDs (SIDs) with powerful representation capabilities for candidate items. However, existing SIDs derived solely from item content create a semantic gap, failing to align dynamic query intents with static item representations. Furthermore, current generative paradigms rarely model user behavior sequences and are always bottlenecked by the high inference latency of beam-search autoregressive decoding. To address these challenges, we propose Cross-component Hierarchical semantic Alignment for Personalized generative retrieval (CHAP), a novel personalized GR framework from a hierarchical perspective. First, we design a Hierarchical Semantic Alignment module to align query’s latent space with item’s quantization path and synchronize multi-granular semantics. Second, we construct a personalized GR framework that models user behavior by synergizing discrete SIDs for structural guidance and continuous representations for fine-grained semantic refinement. Notably, we introduce a Residual Cascading Generation mechanism to restrict the costly multi-step Transformer Decoder to a single pass inference, boosting inference throughput while mitigating information loss. Extensive experiments on three public datasets, one proprietary industrial dataset, and online A/B tests demonstrate CHAP’s superiority, validating the effectiveness and practical value of our approach. The code is publicly available at https://github.com/zzzgm/CHAP.

## 1 Introduction

Modern Information Retrieval (IR) systems traditionally rely on the "Index-Retrieve-Rank" cascade pipeline (Wang et al., 2011; Zhou et al., 2021), employing sparse methods like BM25 (Robertson et al., 2009) for keyword matching and dense methods like DPR (Karpukhin et al., 2020) for semantic vector mapping. However, static models often struggle to adapt to dynamic, personalized user intents (Liu et al., 2020). Recently, Generative Retrieval (GR) has emerged as a disruptive paradigm alongside the advancement of Large Language Models (LLMs) (Tang et al., 2023b; Li et al., 2025). By adopting an end-to-end approach that transforms retrieval into a sequence generation task which directly generates relevant item identifiers (ItemIDs) from user queries, GR eliminates the need for complex external indices. This paradigm significantly enhances scalability and demonstrates superior performance in long-tail queries and coldstart problems (Kuo et al., 2024; Yuan et al., 2024).

In the GR architecture, generating Semantic IDs (SIDs) with robust representational capabilities remains a core challenge (Zhang et al., 2025b; Liu et al., 2025). Existing models like TIGER (Rajput et al., 2023) and HierGR (Zhang et al., 2025a) train SIDs solely on item corpora, lacking explicit query perception and failing to bridge the inherent semantic gap between user intents and item descriptions, which leads to poor generalization on complex queries (Zhang et al., 2025c; Zheng et al., 2025). Furthermore, current methods exhibit critical limitations in structural utilization, personalization, and inference efficiency. Treating quantized SIDs merely as flat symbol sequences ignores the natural "Coarse-to-Fine" hierarchy, wasting its benefits for feature generalization and semantic stability (Williams et al., 2020; Si et al., 2024). Relying solely on discrete sparse IDs also discards fine-grained continuous vector details, hindering the integration of user behavior sequences (Yang et al., 2025). Crucially, standard autoregressive GR paradigms suffer from severe computational bottlenecks during online serving; repeatedly executing the heavy Transformer Decoder (especially the costly Cross-Attention) for each hierarchical step causes prohibitive latency, hindering large-scale deployment in latency-sensitive industrial systems.

To address these limitations, we propose Crosscomponent Hierarchical semantic Alignment for Personalized generative retrieval (CHAP), a novel framework which performs hierarchical alignment on pre-trained SIDs and efficiently fuses sparse and dense representations in a sequence model.

Specifically, CHAP initializes SIDs using a Residual Quantized Variational Autoencoder (RQ-VAE) (Rajput et al., 2023). To bridge the queryitem semantic gap while preventing codebook collapse, we design a multi-module Hierarchical Semantic Alignment paradigm which includes a cross-sample alignment module to proactively pull the query’s potential representation towards the target’s quantization space, a hierarchical-aware contrastive learning module to enforce coarse-tofine alignment, and a soft probability distillation module to regularize the output distribution of the encoder via KL divergence. Integrating these modules ensures the generated SIDs can robustly comprehend complex query semantics.

Furthermore, CHAP constructs a Personalized Cascading Generative Sequence Model that implements multi-granular modeling of user queries and personalized behavior sequences by crosscascading aligned sparse SIDs with its raw dense vector. Unlike standard autoregressive approaches, CHAP implements a Residual Cascading Generation strategy which decouples heavy historical intent routing from hierarchical code generation. By restricting the costly multi-step Transformer Decoder to a single-pass inference and offloading the layer-wise generation to lightweight residual blocks, this mechanism not only mitigates information loss but also acts as an efficient decoding accelerator, significantly boosting inference throughput. Our main contributions are as follows:

• We propose a Hierarchical Semantic Alignment paradigm, effectively bridging the inherent Semantic Gap between user intents and static item codes.

• We design a Personalized GR Architecture equipped with a Residual Cascading Generation mechanism, which resolves the inherent latency bottleneck of autoregressive GR while effectively mitigating information loss through coarse-to-fine structural guidance.

• Extensive experiments on three public datasets and one proprietary industrial dataset, as well as online A/B tests demonstrate the superiority of our method.

## 2 Related Works

Sparse retrieval methods like BM25 (Robertson et al., 2009) and TF-IDF (Ramos et al., 2003) rely on exact term matching but suffer from vocabulary mismatch. Conversely, dense retrieval approaches (e.g., DPR (Karpukhin et al., 2020), ANCE (Xiong et al., 2020)) map inputs into a shared vector space, enhancing semantic matching but necessitating costly external indices like HNSW (Malkov and Yashunin, 2018). Personalized retrieval tailors results by incorporating user contexts. QEM (Ai et al., 2019a) investigates personalization mechanisms, while DREM (Ai et al., 2019b) models dynamic user-product relationships via knowledge graphs. Embedding-based approaches like ZAM (Ai et al., 2019a), HEM (Ai et al., 2017), and TEM (Bi et al., 2020) integrate user profiles and history into unified embeddings.

Generative retrieval internalizes corpus knowledge to directly generate ItemIDs. DSI (Tay et al., 2022) pioneered this paradigm, followed by semantic enhancements in SE-DSI (Tang et al., 2023a). TIGER (Rajput et al., 2023) first utilized RQ-VAE for SIDs. Regarding architecture, CO-BRA (Yang et al., 2025) integrates sparse and dense vectors, while $\mathrm { C A T - I D ^ { 2 } }$ (Liu et al., 2025), and MERGE (Zhang et al., 2025b) optimize ID generation by exploiting multi-level correlations.

## 3 Preliminaries

## 3.1 Problem Definition

Given a user’s interaction history S = $\{ i _ { 1 } , \dotsc , i _ { N } \}$ and a current query q, Personalized GR aims to retrieve items by directly generating their identifiers such as a unique hierarchical SID $\mathbf { c } = ( c ^ { 0 } , \dots , c ^ { L - 1 } )$ derived from a codebook. The objective is to learn a probabilistic model θ that maximizes the probability of generating the target SID given the context:

$$
\begin{array} { l } { { \displaystyle P ( i | q , S ; \theta ) = P ( \mathbf { c } | q , S ; \theta ) } } \\ { ~ } \\ { { \displaystyle = \prod _ { l = 0 } ^ { L - 1 } P ( c ^ { l } | c ^ { < l } , q , S ; \theta ) } . } \end{array}\tag{1}
$$

where $c ^ { < l }$ denotes the generated prefix tokens at layer l. Inference is to generate the target SID with the highest joint probability.

![](images/75b28737fb2c7467ae348b05529550b30b4276b6b6be9a04b972e59990f3795a.jpg)  
Figure 1: Schematic of SID generation via RQ-VAE.

## 3.2 Semantic ID Generation via RQ-VAE

To assign meaningful discrete identifiers to items effectively, we employ RQ-VAE (Rajput et al., 2023) (Fig. 1). This approach generates hierarchical SIDs by recursively quantizing item embeddings in a coarse-to-fine manner, which serves as the foundational representation for our proposed framework.

Given an item i, a pre-trained language encoder first encodes its textual content into a dense embedding d<sub>i</sub>. A deep neural network encoder $E ( \cdot )$ then maps this embedding to a latent representation $\mathbf { z } = E ( \mathbf { d } _ { i } )$ . The quantization process utilizes L residual layers, where each layer l is equipped with a codebook $\mathcal { C } ^ { l } = \{ { \mathbf { e } _ { k } ^ { l } } \} _ { k = 1 } ^ { K }$ containing K code words. At each step l, quantization is performed by selecting the code word nearest to the current residual vector $\mathbf { r } ^ { l }$ (where $\mathbf { r } ^ { 0 } = \mathbf { z } )$ . The Semantic Token $c ^ { l }$ for layer l is determined as:

$$
c ^ { l } = \arg \operatorname* { m i n } _ { k } \| { \bf r } ^ { l } - { \bf e } _ { k } ^ { l } \| _ { 2 } ^ { 2 } , \quad { \bf r } ^ { l + 1 } = { \bf r } ^ { l } - { \bf e } _ { c ^ { l } } ^ { l } .\tag{2}
$$

After L iterations, we obtain a discrete coarseto-fine SID $( c ^ { 0 } , \ldots , c ^ { L - 1 } )$ . The quantized latent representation zˆ is then approximated by the sum of the selected codebook vectors: $\begin{array} { r } { \hat { \mathbf { z } } = \dot { \sum } _ { l = 0 } ^ { L - 1 } \mathbf { e } _ { c ^ { l } } ^ { l } } \end{array}$

The training of RQ-VAE aims to reconstruct the original input while optimizing the codebooks. The decoder $D ( \cdot )$ attempts to reconstruct the input $\mathbf { d } _ { i }$ from zˆ, resulting in the reconstruction loss:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \| \mathbf { d } _ { i } - D ( \hat { \mathbf { z } } ) \| _ { 2 } ^ { 2 } .\tag{3}
$$

To ensure the stability of quantization, RQ-VAE employs a split quantization objective consisting of a codebook loss to update codebook vectors towards the residuals and a commitment loss to constrain the encoder output:

$$
\mathcal { L } _ { \mathrm { c o d e b o o k } } = \sum _ { l = 0 } ^ { L - 1 } \| s g [ \mathbf { r } ^ { l } ] - \mathbf { e } _ { c ^ { l } } ^ { l } \| _ { 2 } ^ { 2 }\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { c o m m i t } } = \beta \sum _ { l = 0 } ^ { L - 1 } \Vert \mathbf { r } ^ { l } - s g [ \mathbf { e } _ { c ^ { l } } ^ { l } ] \Vert _ { 2 } ^ { 2 } ,\tag{5}
$$

where $s g [ \cdot ]$ denotes the stop-gradient operation which prevents gradient update during backpropagation, and $\beta$ is a hyperparameter balancing the commitment loss. The final training objective is:

$$
{ \mathcal { L } } _ { \mathrm { R Q - V A E } } = { \mathcal { L } } _ { \mathrm { r e c o n } } + { \mathcal { L } } _ { \mathrm { c o d e b o o k } } + { \mathcal { L } } _ { \mathrm { c o m m i t } } .\tag{6}
$$

## 4 Methodology

## 4.1 Hierarchical Semantic Alignment (HSA)

Directly applying an item-pretrained RQ-VAE to retrieval faces a semantic gap, as its encoder $E ( \cdot )$ is optimized solely for item reconstruction. To bridge this, we propose a Hierarchical Semantic Alignment (HSA) paradigm (Fig. 2). To prevent semantic drift and catastrophic forgetting, we freeze the pre-trained codebook, forcing the query quantization path directly into the stable item manifold.

Cross-Sample Alignment (CSA). To pull the query’s latent representation into the item’s quantization space, we repurpose RQ-VAE losses by swapping input-target variables. Given a query q and target $t ,$ the Cross-Commitment loss forces the query’s residual $\mathbf { r } _ { q } ^ { l }$ to align with the target’s selected code $\mathbf { e } _ { c ^ { l } } ^ { l }$

$$
\mathcal { L } _ { \mathrm { C C } } = \beta \sum _ { l = 0 } ^ { L - 1 } \| \mathbf { r } _ { q } ^ { l } - s g [ \mathbf { e } _ { c ^ { l } } ^ { l } ] \| _ { 2 } ^ { 2 } .\tag{7}
$$

The Cross-Reconstruction loss ensures the target’s raw content $\mathbf { d } _ { t }$ is reconstructable from the query’s quantized representation $\hat { \mathbf { z } } _ { q } \mathrm { . }$

$$
\mathcal { L } _ { \mathrm { C R } } = \Vert \mathbf { d } _ { t } - D ( \hat { \mathbf { z } } _ { q } ) \Vert _ { 2 } ^ { 2 } .\tag{8}
$$

The total CSA loss is $\mathcal { L } _ { \mathrm { C S A } } = \mathcal { L } _ { \mathrm { C C } } + \mathcal { L } _ { \mathrm { C R } }$

Hierarchical-Aware Contrastive Learning (HCL). To explicitly exploit the “Coarse-to-Fine” SID structure, we align the query’s partial quantized representation $\begin{array} { r } { \bar { \bf z } _ { q } ^ { ( l ) } = \sum _ { i = 0 } ^ { \bar { l } } \bar { \bf e } _ { c _ { q } ^ { i } } ^ { i } } \end{array}$ with the target’s partial quantized representation $\hat { \mathbf { z } } _ { t } ^ { ( l ) } ~ = ~$ $\textstyle \sum _ { i = 0 } ^ { \bar { l } } \mathbf { e } _ { c _ { t } ^ { i } } ^ { i }$ accumulated up to depth l:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { H C L } } = \displaystyle - \sum _ { l = 0 } ^ { L - 1 } \log \frac { \exp ( s _ { l } ^ { + } / \tau ) } { \sum _ { j \in \mathcal { B } } \exp ( s _ { l j } / \tau ) } , } \\ & { \quad s _ { l } ^ { + } = \sin \Bigl ( \hat { \mathbf { z } } _ { q } ^ { ( l ) } , s g [ \hat { \mathbf { z } } _ { t } ^ { ( l ) } ] \Bigr ) , } \\ & { \quad s _ { l j } = \sin \Bigl ( \hat { \mathbf { z } } _ { q } ^ { ( l ) } , s g [ \hat { \mathbf { z } } _ { j } ^ { ( l ) } ] \Bigr ) . } \end{array}\tag{9}
$$

where B is the batch containing the positive target and negatives, and τ is the temperature.

![](images/b5a02d1c38de75a20045251c56ef80a1fedc8bf5d22622be6c2faed22d8e8b90.jpg)  
Figure 2: Overall architecture of CHAP, featuring HSA to bridge the query-item gap (left), and dual-view Personalized Sequence Modeling with Residual Cascading Generation for efficient coarse-to-fine retrieval (right).

Soft Probability Distillation. To prevent drastic latent shifts caused by hard quantization assignments, we distill the soft code assignment distribution from the frozen item-side quantization path of the target item to the trainable query encoder via KL divergence:

$$
\mathcal { L } _ { \mathrm { D i s t i l l } } = \sum _ { l = 0 } ^ { L - 1 } { \mathrm { K L } } \left( P _ { \mathrm { i t e m } } ( \cdot | t ) \parallel P _ { \mathrm { q u e r y } } ( \cdot | q ) \right) ,\tag{10}
$$

where the soft assignment probability is defined by the Euclidean distance to the residual:

$$
P ( c ^ { l } = k | q ) = \frac { \exp \left( - \| \mathbf { r } _ { q } ^ { l } - \mathbf { e } _ { k } ^ { l } \| _ { 2 } ^ { 2 } \right) } { \sum _ { j = 1 } ^ { K } \exp \left( - \| \mathbf { r } _ { q } ^ { l } - \mathbf { e } _ { j } ^ { l } \| _ { 2 } ^ { 2 } \right) } .\tag{11}
$$

The comprehensive HSA objective is optimized via a weighted sum:

$$
{ \mathcal { L } } _ { \mathrm { H S A } } = { \mathcal { L } } _ { \mathrm { C S A } } + \gamma \cdot { \mathcal { L } } _ { \mathrm { H C L } } + \delta \cdot { \mathcal { L } } _ { \mathrm { D i s t i l l } } .\tag{12}
$$

## 4.2 Personalized Sequence Modeling

Dual-View Sequence Modeling. To capture the multi-granularity preferences of users and generate high-fidelity retrieval representations, we construct an input sequence where each interaction synergizes an aligned discrete SID and a continuous dense vector. For a user history ${ \cal S } = \{ i _ { 1 } , \dots , i _ { N } \}$ and query q, the input is organized as:

$$
\begin{array} { r } { X _ { i n } = [ \mathbf { v } _ { [ \complement \sqcup \mathrm { S } ] } , \mathbf { v } _ { q } ^ { \mathrm { s p a r s e } } , \mathbf { v } _ { q } ^ { \mathrm { d e n s e } } , } \\ { \mathbf { v } _ { i _ { 1 } } ^ { \mathrm { s p a r s e } } , \mathbf { v } _ { i _ { 1 } } ^ { \mathrm { d e n s e } } , \mathbf { \cdot } \mathbf { \cdot } \mathbf { \cdot } ] . } \end{array}\tag{13}
$$

The sparse view $\mathbf { v } ^ { \mathrm { s p a r s e } }$ maps the aggregated codebook vectors $( \sum \mathbf { e } _ { c ^ { l } } ^ { l } )$ into a unified dense space, while $\mathbf { v } ^ { \mathrm { d e n s e } }$ corresponds to the raw language model embedding. To maintain training-inference consistency and prevent exposure bias, the unified decoder is initialized exclusively with the query’s dual representation $( \mathbf { v } _ { q } ^ { \mathrm { s p a r s e } } , \mathbf { v } _ { q } ^ { \mathrm { d e n s e } } )$ . This explicitly forces the decoder to “look back” via crossattention to retrieve personalized results.

Hybrid Optimization. Our multi-task framework optimizes the Sparse Generation Loss for hierarchical SID classification across all L levels:

$$
\mathcal { L } _ { \mathrm { s p a r s e } } = - \sum _ { l = 0 } ^ { L - 1 } \log \left( \frac { \exp ( s _ { c ^ { l } } ^ { l } ) } { \sum _ { k = 1 } ^ { K } \exp ( s _ { k } ^ { l } ) } \right) ,\tag{14}
$$

where $\mathbf { s } ^ { l } \in \mathbb { R } ^ { K }$ is the logit vector at level l. Simultaneously, to avoid Mean Squared Error instability, we employ an InfoNCE-based Dense Contrastive Loss $( { \mathcal { L } } _ { \mathrm { d e n s e } } )$ over the batch set $B = \{ \mathbf { v } ^ { + } \} \cup B ^ { - } ;$

$$
\mathcal { L } _ { \mathrm { d e n s e } } = - \log \frac { \exp ( \sin ( \hat { \mathbf { v } } , \mathbf { v } ^ { + } ) / \tau ) } { \sum _ { \mathbf { v } ^ { \prime } \in B } \exp ( \sin ( \hat { \mathbf { v } } , \mathbf { v } ^ { \prime } ) / \tau ) } ,\tag{15}
$$

treating the predicted dense vector vˆ (introduced in Sec. 4.3) as the anchor. The joint objective $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s p a r s e } } + \mathcal { L } _ { \mathrm { d e n s e } }$ creates a robust curriculum learning effect, where sparse structural constraints simplify the dense ranking optimization.

## 4.3 Residual Cascading Generation and Retrieval

To achieve a balance between structural correctness and fine-grained semantic precision, we adopt a “Coarse-to-Fine” strategy to coordinate the prediction of discrete SIDs and continuous dense vectors. Formally, we decompose the generation probability of a target item i, given the context X, into two stages: the joint probability of the L hierarchical Semantic Tokens, followed by the conditional probability of the dense vector (operationalized by a cosine-softmax score over generated candidates):

$$
P ( i | X ) = \underbrace { \prod _ { l = 0 } ^ { L - 1 } P ( c _ { i } ^ { l } | c _ { i } ^ { < l } , X ) } _ { \mathrm { S p a r s e G e n e r a t i o n } } \cdot \underbrace { P ( \mathbf { v } _ { i } | c _ { i } ^ { 0 : L - 1 } , X ) } _ { \mathrm { D e n s e R e f i n e m e n t } } .\tag{16}
$$

An entropy-based justification for this cascaded sparse-dense factorization is provided in App. B.

Coarse-to-Fine Residual Generation. To avoid the prohibitive latency of standard autoregressive decoding, we propose a Single-pass Residual Generation Mechanism. The Transformer Decoder executes only once over the query’s input tokens. Specifically, let $\mathbf { h } _ { \mathrm { d e c } } ^ { \mathrm { s p a r s e } }$ and $\mathbf { h } _ { \mathrm { d e c } } ^ { \mathrm { d e n s e } }$ denote the output hidden states corresponding to the discrete query SID position and the continuous query embedding position, respectively. Utilizing $\bar { \mathbf { h } } _ { \mathrm { d e c } } ^ { \mathrm { s p a r s e } }$ as a global structural context anchor, subsequent states for predicting level l are updated via lightweight residual blocks:

$$
\mathbf { h } ^ { ( l + 1 ) } = \mathrm { R e s B l o c k } \left( \mathrm { C o n c a t } [ \mathbf { h } _ { \mathrm { d e c } } ^ { \mathrm { s p a r s e } } , \mathbf { e } _ { c } ^ { l } ] \right) + \mathbf { h } ^ { ( l ) } .\tag{17}
$$

This decoupled design acts as a powerful decoding accelerator. To align the continuous vector with this discrete structure, we inject the reconstructed quantization $\begin{array} { r } { \hat { \mathbf { z } } = \sum _ { l = 0 } ^ { L - 1 } \mathbf { e } _ { c } ^ { l } } \end{array}$ into the dense prediction head conditioned on $\bar { \mathbf { h } } _ { \mathrm { d e c } } ^ { \mathrm { d e n s e } }$

$$
\hat { \mathbf { v } } = \mathrm { P r e d H e a d } \left( { \mathrm { C o n c a t } } [ \mathbf { h } _ { \mathrm { d e c } } ^ { \mathrm { d e n s e } } , \hat { \mathbf { z } } ] \right) ,\tag{18}
$$

achieving an elegant balance between structural correctness and fine-grained semantic precision.

Candidate Scoring. During inference, we maximize throughput via a parallel sampling strategy to generate M candidate SIDs. We calculate the accumulated log-probability of the sampled hierarchical path as the sparse signal $S _ { \mathrm { s p a r s e } } .$ , and the cosine similarity between the predicted dense vector vˆ corresponding to the sampled SID and the candidate item’s raw vector $\mathbf { v _ { i } }$ as the dense signal $S _ { \mathrm { d e n s e } } .$ Both signals are normalized using a temperaturescaled Softmax over the candidate pool $\mathcal { T } _ { \mathrm { c a n d } }$

$$
\mathcal { N } ( S ) = \frac { \exp ( S / \tau ) } { \sum _ { j \in \mathcal { Z } _ { \mathrm { c a n d } } } \exp ( { S _ { j } / \tau } ) } .\tag{19}
$$

The final score fuses both dual-view intents:

$$
S _ { \mathrm { f i n a l } } ( i ) = \mathcal { N } \left( S _ { \mathrm { s p a r s e } } ( i ) \right) \cdot \mathcal { N } \left( S _ { \mathrm { d e n s e } } ( i ) \right) .\tag{20}
$$

Table 1: Statistics of the datasets in our experiments.
<table><tr><td>Dataset</td><td>#Items</td><td>#Queries</td><td>#Q-I Pairs</td><td>Avg. Hist.</td></tr><tr><td>ESCI-us</td><td>1,215,854</td><td>97,346</td><td>1,818,825</td><td></td></tr><tr><td>KuaiSearch</td><td>6,634,118</td><td>295,561</td><td>682,228</td><td>10.37</td></tr><tr><td>Amazon</td><td>35,772</td><td>9,070</td><td>9,070</td><td>40.12</td></tr><tr><td>Local-Life</td><td>10,145,542</td><td>3,089,026</td><td>3,089,026</td><td>7.47</td></tr></table>

## 5 Experiments

We analyze CHAP’s effectiveness and address these research questions in this section: RQ1: How does CHAP perform against state-of-the-art baselines? RQ2: How does semantic alignment improve generated ItemID quality? RQ3: How do core components contribute across intent-wise query segments? RQ4: Does our residual architecture resolve GR latency bottlenecks? RQ5: How robust is CHAP to codebook and decoding hyperparameters? RQ6: Can CHAP drive tangible business growth in live online production?

## 5.1 Experimental Settings

## 5.1.1 Datasets

To validate the effectiveness of CHAP, we conduct extensive experiments on four real-world datasets. ESCI-us (Reddy et al., 2022) features authentic Amazon search queries with complex linguistic patterns and strict attribute constraints. KuaiSearch (Li et al., 2026) provides a massive e-commerce search corpus from Kuaishou, preserving real queries, long-term user behavior sequences, and long-tail product distributions. Amazon (Cai et al., 2025) integrates diverse user profiles and historical web behaviors to rigorously evaluate highly personalized search instructions. Lastly, Local-Life is a large-scale industrial dataset collected from 30 days of user interaction logs on a leading platform, covering heterogeneous search intents with raw natural language texts. Detailed descriptions and comprehensive statistics for all four datasets are provided in Tab. 1 and App. C.

## 5.1.2 Baselines

To comprehensively evaluate the performance of CHAP, we compare it against a diverse set of 14 representative baselines across four categories. For sparse retrieval, we select BM25 (Robertson et al., 2009), Doc2Query (Nogueira et al., 2019), and DeepCT (Dai and Callan, 2019). For dense retrieval, we employ DPR (Karpukhin et al., 2020), Sen-T5 (Ni et al., 2022), and MPNet (Song et al., 2020). For personalized retrieval scenarios, we compare against TEM (Bi et al., 2020) and CoPPS (Dai et al., 2023). Finally, for Generative Retrieval (GR), we evaluate against state-ofthe-art models including DSI (Tay et al., 2022), NCI (Wang et al., 2022), TIGER (Rajput et al., 2023), LTRGR (Li et al., 2024), MERGE (Zhang et al., 2025b), and COBRA (Yang et al., 2025). Detailed descriptions are provided in App. D.

Table 2: Performance comparison on four datasets. Bold indicates the result significantly outperforms all baselines with paired t-tests at $p < 0 . 0 5$ level, and underlined values denote the second-best results.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="3">ESCI-us</td><td colspan="3">KuaiSearch</td><td colspan="3">Amazon</td><td colspan="3">Local-Life</td></tr><tr><td>R@10</td><td>R@50</td><td>NDCG@50</td><td>R@10</td><td>R@50</td><td>HR@50</td><td>R@10</td><td>R@50</td><td>NDCG@50</td><td>R@10</td><td>R@50</td><td>MRR@10</td></tr><tr><td rowspan="3">Sparse</td><td>BM25</td><td>0.0480</td><td>0.0932</td><td>0.0811</td><td>0.0706</td><td>0.1564</td><td>0.2088</td><td>0.4807</td><td>0.6780</td><td>0.3887</td><td>0.2531</td><td>0.3415</td><td>0.0997</td></tr><tr><td>Doc2Query</td><td>0.0551</td><td>0.1410</td><td>0.1015</td><td>0.0784</td><td>0.1772</td><td>0.2382</td><td>0.5064</td><td>0.6904</td><td>0.4039</td><td>0.2971</td><td>0.3959</td><td>0.1374</td></tr><tr><td>DeepCT</td><td>0.0626</td><td>0.1486</td><td>0.0997</td><td>0.0760</td><td>0.1898</td><td>0.2531</td><td>0.5869</td><td>0.7622</td><td>0.4708</td><td>0.3718</td><td>0.4387</td><td>0.1913</td></tr><tr><td rowspan="3">Dense</td><td>DPR</td><td>0.0552</td><td>0.1765</td><td>0.1104</td><td>0.0826</td><td>0.2079</td><td>0.2769</td><td>0.2010</td><td>0.3413</td><td>0.1666</td><td>0.3153</td><td>0.4820</td><td>0.1582</td></tr><tr><td>Sen-T5</td><td>0.0458</td><td>0.1363</td><td>0.0845</td><td>0.0673</td><td>0.1622</td><td>0.2227</td><td>0.5690</td><td>0.7967</td><td>0.4376</td><td>0.2360</td><td>0.3802</td><td>0.0956</td></tr><tr><td>MPNet</td><td>0.0294</td><td>0.0879</td><td>0.0531</td><td>0.0619</td><td>0.1279</td><td>0.1970</td><td>0.5961</td><td>0.7755</td><td>0.4719</td><td>0.1820</td><td>0.3210</td><td>0.0815</td></tr><tr><td rowspan="2">Personalized</td><td></td><td>0.0213</td><td>0.0545</td><td>0.0402</td><td>0.0852</td><td>0.2131</td><td></td><td>0.4814</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TEM</td><td>0.0246</td><td>0.0878</td><td>0.0458</td><td>0.0877</td><td>0.2153</td><td>0.2776 0.2832</td><td>0.4854</td><td>0.7301 0.8004</td><td>0.3535 0.3699</td><td>0.3551 0.4265</td><td>0.5401</td><td>0.1716 0.2403</td></tr><tr><td rowspan="8">Generative</td><td>CoPPS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.5971</td><td></td></tr><tr><td>DSI</td><td>0.0217</td><td>0.0552</td><td>0.0433</td><td>0.0623</td><td>0.1369</td><td>0.2018</td><td>0.4181</td><td>0.6202</td><td>0.3414</td><td>0.1868</td><td>0.3056</td><td>0.0836</td></tr><tr><td>NCI</td><td>0.0432</td><td>0.1006</td><td>0.0803</td><td>0.0591</td><td>0.1270</td><td>0.1781</td><td>0.3765</td><td>0.5836</td><td>0.3139</td><td>0.2540</td><td>0.3951</td><td>0.1014</td></tr><tr><td>LTRGR</td><td>0.0316</td><td>0.0944</td><td>0.0629</td><td>0.0688</td><td>0.1501</td><td>0.2184</td><td>0.5143</td><td>0.7025</td><td>0.4062</td><td>0.2608</td><td>0.4644</td><td>0.1120</td></tr><tr><td>TIGER</td><td>0.0487</td><td>0.1335 0.1903</td><td>0.0814</td><td>0.0731 0.0858</td><td>0.1796</td><td>0.2427</td><td>0.5387</td><td>0.7292 0.7819</td><td>0.4237 0.4523</td><td>0.2659</td><td>0.4068</td><td>0.1073</td></tr><tr><td>MERGE</td><td>0.0612 0.0585</td><td>0.1950</td><td>0.1070 0.1126</td><td>0.0890</td><td>0.2033 0.2074</td><td>0.2625 0.2803</td><td>0.5697 0.6205</td><td>0.8474</td><td>0.4906</td><td>0.3850 0.5020</td><td>0.5828 0.6057</td><td>0.2389</td></tr><tr><td>COBRA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2810</td></tr><tr><td>CHAP (Ours)</td><td>0.1127</td><td>0.2452</td><td>0.2454</td><td>0.0944</td><td>0.2289</td><td>0.2949</td><td>0.7358</td><td>0.9416</td><td>0.5191</td><td>0.5803</td><td>0.8085</td><td>0.3302</td></tr></table>

Table 3: Ablation study of CHAP on Local-Life.
<table><tr><td>Variant</td><td>R@10</td><td>R@50</td><td>MRR@10</td><td>QPS</td></tr><tr><td>CHAP (Full Model)</td><td>0.5803</td><td>0.8089</td><td>0.3302</td><td>78.9</td></tr><tr><td colspan="5">(A) Hierarchical Semantic Alignment (HSA)</td></tr><tr><td>w/o HSA</td><td>0.5115</td><td>0.7488</td><td>0.2845</td><td>78.9</td></tr><tr><td>w/o LCSA</td><td>0.5164</td><td>0.7602</td><td>0.2721</td><td>78.9</td></tr><tr><td>w/o LHCL</td><td>0.5218</td><td>0.7534</td><td>0.2858</td><td>78.9</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { D i s t i l l } }$ </td><td>0.5692</td><td>0.7895</td><td>0.3063</td><td>78.9</td></tr><tr><td colspan="5">(B) Sequence Modeling w/o Dense Refinement</td></tr><tr><td>(C) Residual Cascading Generation</td><td>0.4352</td><td>0.6455</td><td>0.2241</td><td>81.2</td></tr><tr><td colspan="5">w/o Residual Gen. (Autoregressive)</td></tr><tr><td>w/o Self-Attention (Cross-Attention)</td><td>0.5285 0.5712</td><td>0.7610 0.8062</td><td>0.2849</td><td>40.5</td></tr><tr><td></td><td>0.4651</td><td></td><td>0.3237</td><td>84.8</td></tr><tr><td>w/o Decoder (Pooling + MLP)</td><td></td><td>0.6836</td><td>0.2584</td><td>112.4</td></tr></table>

## 5.1.3 Evaluation Metrics

We evaluate retrieval performance using four standard metrics: Recall (R@K) for the overall macroscopic coverage of relevant candidates; Hit Ratio (HR@K) for the success rate of retrieving at least one target in sparse or single-interaction scenarios; Mean Reciprocal Rank (MRR@K) for assessing strict positional ranking quality in single-target datasets; and Normalized Discounted Cumulative Gain (NDCG@K) for handling fine-grained permutations in multi-target datasets. Detailed mathematical formulations and selection justifications are provided in App. E.

## 5.1.4 Implementation Details

We utilize BERT-base (Devlin et al., 2019) for raw text representation and T5-base (Raffel et al., 2020) for the generative retrieval backbone (Chinese-BERT-base and mT5-base for Chinese datasets). In the SID learning stage, the RQ-VAE is configured with a codebook depth of $L = 3$ and a codebook size of $K = 5 1 2$ . For the Hierarchical Semantic Alignment, the loss weights $\gamma$ and $\delta$ are set to 0.01 and 0.001, respectively. During inference, we employ a parallel sampling strategy to efficiently generate $M = 5 0$ diverse candidate SIDs, and set the temperature $\tau = 1 . 1$ for the joint probability normalization operator $\mathcal { N } ( \cdot )$ . Comprehensive details and explicit implementation settings for all baseline models are provided in App. F.

## 5.2 Overall Results (RQ1)

Tab. 2 presents the comparative results across the four evaluated datasets. CHAP consistently establishes a new state-of-the-art, significantly outperforming strong baselines from all categories, including advanced GR models like COBRA and MERGE. Specifically, while standard GR methods like TIGER struggle with the inherent semantic gap—often lagging behind traditional BM25, our Hierarchical Semantic Alignment effectively bridges this gap, synergizing the semantic comprehension of dense retrievers with the structural precision of GR. Furthermore, traditional personalized baselines frequently suffer from relevance drift, allowing historical behaviors to overshadow current search intents. In contrast, CHAP dynamically refines query relevance through its dual-view sequence modeling. This prevents relevance drift and ensures retrieved items are precisely aligned with both the explicit query constraints and the user’s fine-grained intents, yielding robust generalization across diverse, real-world search scenarios.

Notably, unlike COBRA (Yang et al., 2025), which mandates the generation of multiple dense representations—incurring excessive training costs and prohibitive real-time inference overhead, CHAP achieves superior efficiency by elegantly reusing the item’s raw representations for sequence modeling. Furthermore, MERGE (Zhang et al., 2025b) heavily relies on aligning multiple positive samples and varying relevance levels under a single query, exhibiting limited generalization on datasets with single positive targets or uniform relevance levels. In contrast, our semantic alignment paradigm is specifically designed to bridge the gap between complex, multi-intent queries and targets.

## 5.3 Ablation Study

Tab. 3 details our ablation study on Local-Life, incorporating Queries Per Second (QPS) to concurrently assess industrial efficiency. Detailed variant configurations are provided in App. G.

## 5.3.1 Analysis of Generated SIDs (RQ2)

Replacing our aligned encoder with a fixed pretrained encoder (w/o HSA) causes significant degradation, confirming the critical necessity of proactive alignment in bridging the semantic gap. Furthermore, we observe a performance drop when removing the continuous vectors (w/o Dense Refinement). This validates our dual-view design intuition: sparse SIDs act strictly as structural skeletons for coarse retrieval, while dense vectors are indispensable for carrying the fine-grained semantics required for precise personalized matching.

Furthermore, visualization of the latent space encoded by DNN encoder of RQ-VAE after PCA dimensionality reduction (from ESCI-us, shown in Fig. 3) intuitively confirms this advantage: while vanilla RQ-VAE suffers from a visible semantic gap with loosely scattered relevant items, CHAP tightly clusters relevant items (orange) directly around their corresponding queries (red), as highlighted by the dashed boxes, while explicitly pushing irrelevant items (blue) further away. This demonstrates the efficacy of our alignment paradigm in drawing dynamic user intents and static item semantics into a highly unified space. Additional qualitative diagnostics are provided in App. I.

![](images/1207e90186a60bb4ced330d27bec006ff116b49b449bb2a9e40234340501a2e1.jpg)  
Figure 3: Visualization of Items under three queries.

![](images/3489a3926b0974f16031de026a5ca402ea9efe1b6ef1e9aac683c4f65cd4b5b1.jpg)  
Figure 4: Intent-wise query segment analysis.

## 5.3.2 Intent-wise Query Analysis (RQ3)

To further dissect the source of CHAP’s superiority, we extract four intent-wise query segments from the Local-Life dataset: General, Ambiguous, Cold-Start, and LongTail (detailed in App. H). We explicitly evaluate the absolute MRR@10 improvements brought by individually integrating each core component over the vanilla RQ-VAE baseline (Fig. 4).

The segment-wise results corroborate our design intuitions, revealing the orthogonal strengths of each module. Introducing the dual-view Sequence Modeling delivers the most dominant boost on the Ambiguous segment. This confirms that for highly generic queries, dense historical contextualization is indispensable for intent disambiguation. However, its isolated contribution drops on the ColdStart and LongTail segments, as cold entities inherently lack sufficient interaction histories for sequence models to exploit. Conversely, integrating the HSA module addresses these sparse segments, suggesting that proactive semantic mapping bridges the representation gap for rare queries and new items, independent of popularity biases. Meanwhile, the Residual Generation mechanism yields a stable, structural baseline uplift across all slices by mitigating layer-wise decoding decay.

## 5.3.3 Efficiency Analysis (RQ4)

We deeply deconstruct our decoding architecture to evaluate its necessity and efficiency. Degenerating to a standard autoregressive paradigm (w/o Residual Gen.) harms accuracy and halves inference throughput (QPS), indicating our residual design is a powerful decoding accelerator. Removing the decoder’s self-attention (w/o Self-Attention) improves QPS with minimal precision loss, showing explicit residual pathways effectively replace self-attention for hierarchical modeling. However, entirely removing the decoder (w/o Decoder) maximizes QPS but collapses MRR. This highlights the indispensable role of cross-attention for dynamic history routing across lengthy user sequences.

Table 4: Efficiency comparison on Local-Life. ItemID and Seq. denote the training time for ItemID construction and the GR model, respectively. Total is their sum.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>ItemID (H) Seq. (H)Total (H)</td><td rowspan=1 colspan=1>QPS</td></tr><tr><td rowspan=1 colspan=1>DSI</td><td rowspan=1 colspan=1>1.3       16.5     17.8</td><td rowspan=1 colspan=1>37.5</td></tr><tr><td rowspan=1 colspan=1>NCI</td><td rowspan=1 colspan=1>1.3       34.0     35.3</td><td rowspan=1 colspan=1>26.2</td></tr><tr><td rowspan=1 colspan=1>LTRGR</td><td rowspan=1 colspan=1>1.5       31.9     33.4</td><td rowspan=1 colspan=1>34.7</td></tr><tr><td rowspan=1 colspan=1>TIGER</td><td rowspan=1 colspan=1>0.4       18.2     18.6</td><td rowspan=1 colspan=1>39.2</td></tr><tr><td rowspan=1 colspan=1>MERGE</td><td rowspan=1 colspan=1>0.8       19.3     20.1</td><td rowspan=1 colspan=1>36.8</td></tr><tr><td rowspan=1 colspan=1>COBRA</td><td rowspan=1 colspan=1>0.4       26.4     26.8</td><td rowspan=1 colspan=1>32.4</td></tr><tr><td rowspan=1 colspan=1>CHAP (Ours)</td><td rowspan=1 colspan=1>0.5       23.8     24.3</td><td rowspan=1 colspan=1>78.9</td></tr></table>

![](images/b5da4c22bc10faf5d299e80e301e7d015587cbf6430419b5900cf6eb43205923.jpg)  
(a) Alignment Weights

![](images/f0b59f30c05690cfc55cf5545e83640db2d29c04a151605d21138e4b80bf91d2.jpg)  
(b) Codebook Size  
Figure 5: Effect of hyperparameters on retrieval tasks.

To further contextualize the efficiency (Tab. 4), we evaluated the computational overhead using an 8×A100 (80G) GPU cluster for training and a single A100 (80G) GPU for inference QPS. During training, advanced baselines like NCI and LTRGR incur excessive computational overhead due to massive query augmentation or complex list-wise optimization. In contrast, CHAP maintains a highly competitive duration, presenting a cost-effective trade-off for modeling dual-view sequences. During online inference, while standard GR models and dense-heavy frameworks (e.g., COBRA) are severely bottlenecked by repeated layer-wise decoding, our decoupled architecture, combined with parallel sampling, achieves substantially higher QPS, satisfying the strict low-latency requirements of large-scale industrial serving.

![](images/41fa047e2e283a4b7dc2705b9bd993d2ed4cb343cae86065a560ca0066a3b15c.jpg)  
Figure 6: Impact of normalization temperature τ and candidate set width M on accuracy and diversity.

Table 5: Online A/B testing results over a 14-day period. All metrics are statistically significant with paired t-tests at $p < 0 . 0 5$ level.
<table><tr><td>Metric</td><td>Relative Improvement</td></tr><tr><td>UV-CTR (Click-Through Rate)</td><td>+0.77%</td></tr><tr><td>UV-CXR (Conversion Rate)</td><td>+2.09%</td></tr><tr><td>Pay Orders (Order Volume)</td><td>+2.98%</td></tr></table>

## 5.4 Hyperparameter Analysis (RQ5)

We analyze the sensitivity of critical hyperparameters on the Local-Life dataset to evaluate CHAP’s robustness. Regarding training objectives, the alignment weights exhibit an inverted U-shape trend, peaking at $\gamma ~ = ~ 0 . 0 1$ and $\delta ~ = ~ 0 . 0 0 1$ to achieve optimal semantic alignment without disrupting the primary reconstruction stability. For the codebook scale, performance peaks at depth $L = 3$ and size $K = 5 1 2$ . Notably, CHAP demonstrates exceptional robustness to these structural parameters; while extreme settings induce minor metric fluctuations, our framework avoids the performance collapse typically observed in standard GR models under suboptimal codebook assignments. During inference, we regulate the accuracydiversity equilibrium by varying the normalization temperature τ across parallel sampling sizes M. Ranking accuracy follows an inverted U-shape, suffering from over-confident sharp distributions at low τ and excessive exploration noise at high τ. Ultimately, CHAP achieves an optimal balance between semantic discrimination and candidate diversity at $\tau = 1 . 1$ with $M = 5 0$ . Detailed analysis is provided in Fig. 5 and Fig. 6.

## 5.5 Online A/B Test (RQ6)

To verify CHAP’s practical effectiveness and scalability, we conducted a rigorous 14-day online A/B test on a leading production local-life service platform, covering 20% of user traffic. CHAP was deployed via NVIDIA Triton to decouple modules for high-throughput real-time inference, rather than relying on query caching. Compared to the highly optimized online production retrieval system, CHAP achieved consistent improvements: UV-CTR increased by 0.77%, demonstrating that HSA enhances semantic relevance. Moreover, conversion metrics saw substantial lifts, with UV-CXR rising by 2.09% and Pay Order Volume growing by 2.98%, showing that CHAP not only attracts user clicks but, crucially, accurately captures latent user intent to drive final purchasing decisions, yielding tangible business growth in industrial scenarios.

## 6 Conclusion

In this work, we propose CHAP, a novel personalized GR framework designed to overcome the inherent semantic gap and inference latency bottlenecks of existing GR paradigms. By introducing the HSA module, CHAP effectively maps dynamic user intents directly into the static item quantization space. Furthermore, our dual-view sequence modeling, coupled with a single-pass Residual Cascading Generation mechanism, decouples historical intent routing from layer-wise decoding. This architecture not only ensures fine-grained semantic precision but also acts as a powerful decoding accelerator. Extensive experiments across four diverse real-world datasets, alongside a rigorous online A/B test on a leading local-life platform, demonstrate that CHAP establishes a new state-of-the-art. It achieves a balance between exact intent alignment and industrial-grade inference throughput.

## Limitations

Although CHAP demonstrates strong empirical effectiveness and industrial efficiency, several limitations remain for future exploration. First, in the HSA stage, freezing the item-side codebook provides a stable quantization manifold and prevents semantic drift. However, this design may also prevent item representations from being dynamically updated with collaborative filtering signals or query-side feedback, which could limit the upper bound of personalization in highly dynamic environments. Second, our dual-view personalized sequence modeling is mainly validated with hierarchical SIDs generated by RQ-VAE-style quantization. While this setting is representative for recent generative retrieval systems, the generalization ability of the overall framework to other types of ItemIDs, such as lexical identifiers, learned atomic IDs, or tree-based semantic IDs, still requires systematic investigation. Finally, the Residual Cascading Generation module substantially improves inference efficiency by avoiding repeated heavy decoder passes. Nevertheless, how the acceleration ratio evolves when scaling model size, codebook depth, and candidate budget remains an open question. Further compression techniques, such as model distillation for the residual generation blocks, may provide an additional path toward more efficient large-scale deployment.

## Ethical Considerations

This research was reviewed and approved by the relevant technical ethics committee. All data collection procedures followed the platform’s compliance requirements and were conducted with user permission or authorization where applicable. Before being used for model training, evaluation, and analysis, all user-related data were anonymized or de-identified to remove personally identifiable information. Our experiments rely only on aggregated statistics and anonymized behavioral signals, and do not attempt to infer or recover individual identities. The online A/B test was conducted under the platform’s standard safety and compliance protocols to minimize potential risks to user experience. The open-source implementation of CHAP is released under a CC BY-NC-SA 4.0 license, explicitly disclaiming liability for direct commercial deployment to ensure strict non-commercial, research-only usage.

## Acknowledgements

The work is supported by grants from New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123403), National Natural Science Foundation of China (No. U24A20253), Scientific Research Innovation Capability Support Project for Young Faculty, and the Meituan Research Fund.

## References

Qingyao Ai, Daniel N Hill, SVN Vishwanathan, and W Bruce Croft. 2019a. A zero attention model for personalized product search. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 379–388.

Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W Bruce Croft. 2017. Learning a hierarchical embedding model for personalized product search. In Proceedings of the 40th International ACM SI-GIR Conference on Research and Development in Information Retrieval, pages 645–654.

Qingyao Ai, Yongfeng Zhang, Keping Bi, and W Bruce Croft. 2019b. Explainable product search with a dynamic relation embedding model. ACM Transactions on Information Systems (TOIS), 38(1):1–29.

Vladimir Baikalov, Iskander Bagautdinov, and Sergey Muravyov. 2026. Mitigating collaborative semantic id staleness in generative retrieval. arXiv preprint arXiv:2604.13273.

Keping Bi, Qingyao Ai, and W Bruce Croft. 2020. A transformer-based embedding model for personalized product search. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1521– 1524.

Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu, Xiaoyu Shen, Wenjie Li, and Tat-Seng Chua. 2025. Large language models empowered personalized web agents. In Proceedings ofthe ACM on Web Conference 2025, pages 198–215.

Tadeusz Calinski and Jerzy Harabasz. 1974. A den-´ drite method for cluster analysis. Communications in Statistics-theory and Methods, 3(1):1–27.

Shitong Dai, Jiongnan Liu, Zhicheng Dou, Haonan Wang, Lin Liu, Bo Long, and Ji-Rong Wen. 2023. Contrastive learning for user sequence representation in personalized product search. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 380–389.

Zhuyun Dai and Jamie Callan. 2019. Context-aware sentence/passage term importance estimation for first stage retrieval. arXiv preprint arXiv:1910.10687.

David L Davies and Donald W Bouldin. 1979. A cluster separation measure. IEEE transactions on pattern analysis and machine intelligence, (2):224–227.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186.

Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun Liu. 2024. Scaling laws for dense retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1339–1349.

Junchen Fu, Xuri Ge, Alexandros Karatzoglou, Ioannis Arapakis, Suzan Verberne, Joemon M Jose, and Zhaochun Ren. 2026. Differentiable semantic id for generative recommendation. arXiv preprint arXiv:2601.19711.

Zitian Guo, Yupeng Hou, Clark Mingxuan Ju, Neil Shah, and Julian McAuley. 2026. Mlps are efficient distilled generative recommenders. arXiv preprint arXiv:2605.12617.

Ruining He and Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pages 507–517.

Angqing Jiang, Gaoming Zhang, Jianchun Song, Kena Qi, Dayao Chen, Wei Lin, and Defu Lian. 2026a. Think-to-personalize: Unifying reasoning and retrieval for user-centric personalized dense retrieval. arXiv preprint arXiv:2608.18855.

Jie Jiang, Xinxun Zhang, Enming Zhang, Yuling Xiong, Jun Zhang, Jingwen Wang, Huan Yu, Yuxiang Wang, Hao Wang, Xiao Yan, and 1 others. 2026b. End-to-end semantic id generation for generative advertisement recommendation. arXiv preprint arXiv:2602.10445.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39– 48.

Tzu-Lin Kuo, Tzu-Wei Chiu, Tzung-Sheng Lin, Sheng-Yang Wu, Chao-Wei Huang, and Yun-Nung Chen. 2024. A survey of generative information retrieval. arXiv preprint arXiv:2406.01197.

Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and Zhicheng Dou. 2025. From matching to generation: A survey on generative information retrieval. ACM Transactions on Information Systems, 43(3):1–62.

Yongqi Li, Nan Yang, Liang Wang, Furu Wei, and Wenjie Li. 2024. Learning to rank in generative retrieval. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 8716–8723.

Yupeng Li, Ben Chen, Mingyue Cheng, Zhiding Liu, Xuxin Zhang, Chenyi Lei, and Wenwu Ou. 2026. Kuaisearch: A large-scale e-commerce search dataset for recall, ranking, and relevance. arXiv preprint arXiv:2602.11518.

Jingjing Liu, Chang Liu, and Nicholas J Belkin. 2020. Personalization in text information retrieval: A survey. Journal ofthe Associationfor Information Science and Technology, 71(3):349–369.

Jiongnan Liu, Zhicheng Dou, Guoyu Tang, and Sulong Xu. 2023. Jdsearch: A personalized product search dataset with real queries and full interactions. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2945–2952.

Xiaoyu Liu, Fuwei Zhang, Yiqing Wu, Xinyu Jia, Zenghua Xia, Fuzhen Zhuang, Zhao Zhang, Fei Jiang, and Wei Lin. 2025. Cat-id<sup>2</sup>: Category-tree integrated document identifier learning for generative retrieval in e-commerce. arXiv preprint arXiv:2511.01461.

Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836.

Fengran Mo, Abbas Ghaddar, Kelong Mao, Mehdi Rezagholizadeh, Boxing Chen, Qun Liu, and Jian-Yun Nie. 2024. Chiq: Contextual history enhancement for improving query rewriting in conversational search. arXiv preprint arXiv:2406.05013.

Thong Nguyen and Andrew Yates. 2023. Generative retrieval as dense retrieval. arXiv preprint arXiv:2306.11397.

Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-t5: Scalable sentence encoders from pretrained text-to-text models. In Findings ofthe associationfor computational linguistics: ACL 2022, pages 1864–1874.

Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. arXiv preprint arXiv:1904.08375.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, and 1 others. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems, 36:10299–10315.

Juan Ramos and 1 others. 2003. Using tf-idf to determine word relevance in document queries. In Proceedings ofthefirst instructional conference on machine learning, volume 242, pages 29–48. New Jersey, USA.

Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas, Anlu Xing, and Karthik Subbian. 2022. Shopping queries dataset: A large-scale ESCI benchmark for improving product search.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Stephen Robertson, Hugo Zaragoza, and 1 others. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Peter J Rousseeuw. 1987. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal ofcomputational and applied mathematics, 20:53–65.

Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, and Kun Gai. 2024. Generative retrieval with semantic tree-structured identifiers and contrastive learning. In Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 154–163.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. Mpnet: Masked and permuted pretraining for language understanding. Advances in neural information processing systems, 33:16857– 16867.

Zhongxiang Sun, Zihua Si, Xiaoxue Zang, Dewei Leng, Yanan Niu, Yang Song, Xiao Zhang, and Jun Xu. 2023. Kuaisar: A unified search and recommendation dataset. In Proceedings ofthe 32nd ACM international conference on information and knowledge management, pages 5407–5411.

Yubao Tang, Ruqing Zhang, Jiafeng Guo, Jiangui Chen, Zuowei Zhu, Shuaiqiang Wang, Dawei Yin, and Xueqi Cheng. 2023a. Semantic-enhanced differentiable search index inspired by learning strategies. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4904–4913.

Yubao Tang, Ruqing Zhang, Jiafeng Guo, and Maarten de Rijke. 2023b. Recent advances in generative information retrieval. In Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 294–297.

Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, and 1 others. 2022. Transformer memory as a differentiable search index. Advances in Neural Information Processing Systems, 35:21831–21843.

Lidan Wang, Jimmy Lin, and Donald Metzler. 2011. A cascade ranking model for efficient ranked retrieval. In Proceedings ofthe 34th international ACM SIGIR

conference on Research and development in Information Retrieval, pages 105–114.

Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, and 1 others. 2022. A neural corpus indexer for document retrieval. Advances in Neural Information Processing Systems, 35:25600–25614.

Will Williams, Sam Ringer, Tom Ash, David MacLeod, Jamie Dougherty, and John Hughes. 2020. Hierarchical quantized autoencoders. Advances in Neural Information Processing Systems, 33:4524–4535.

Yanjing Wu, Yinfu Feng, Jian Wang, Wenji Zhou, Yunan Ye, Rong Xiao, and Jun Xiao. 2024. Hi-gen: Generative retrieval for large-scale personalized ecommerce search. arXiv preprint arXiv:2404.15675.

Ming Xia, Zhiqin Zhou, Guoxin Ma, and Dongmin Huang. 2026. Unleash the potential of long semantic ids for generative recommendation. arXiv preprint arXiv:2602.13573.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808.

Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, and 1 others. 2025. Sparse meets dense: Unified generative recommendations with cascaded sparse-dense representations. arXiv preprint arXiv:2503.02453.

Peiwen Yuan, Xinglin Wang, Shaoxiong Feng, Boyuan Pan, Yiwei Li, Heda Wang, Xupeng Miao, and Kan Li. 2024. Generative dense retrieval: Memory can be a burden. arXiv preprint arXiv:2401.10487.

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings ofthe 44th international ACM SIGIR con ference on research and development in information retrieval, pages 1503–1512.

Fuwei Zhang, Xiaoyu Liu, Xinyu Jia, Yingfei Zhang, Zenghua Xia, Fei Jiang, Fuzhen Zhuang, Wei Lin, and Zhao Zhang. 2025a. Hiergr: Hierarchical semantic representation enhancement for generative retrieval in food delivery search. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 6: Industry Track), pages 444–455.

Fuwei Zhang, Xiaoyu Liu, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Fuzhen Zhuang, Wei Lin, and Zhao Zhang. 2025b. Multi-level relevance document identifier learning for generative retrieval. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10066–10080.

Yingchen Zhang, Ruqing Zhang, Jiafeng Guo, Wenjun Peng, Sen Li, Fuyu Lv, and Xueqi Cheng. 2025c. C2t-id: Converting semantic codebooks to textual document identifiers for generative search. In Proceedings ofthe 2025 Annual International ACM SI-GIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 331–336.

Carolina Zheng, Minhui Huang, Dmitrii Pedchenko, Kaushik Rangadurai, Siyu Wang, Fan Xia, Gaby Nahum, Jie Lei, Yang Yang, Tao Liu, and 1 others. 2025. Enhancing embedding representation stability in recommendation systems with semantic id. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems, pages 954–957.

Fan Zhou, Xovee Xu, Goce Trajcevski, and Kunpeng Zhang. 2021. A survey of information cascade analysis: Models, predictions, and recent advances. ACM Computing Surveys (CSUR), 54(2):1–36.

Yujia Zhou, Zhicheng Dou, and Ji-Rong Wen. 2023. Enhancing generative retrieval with reinforcement learning from relevance feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12481–12490.

Jianbo Zhu, Xing Fang, Jing Wang, Mingmin Jin, Bokang Wang, Guangxin Song, Zhenyu Xie, and Junjie Bai. 2026. Efficient generative retrieval for e-commerce search with semantic cluster ids and expert-guided rl. arXiv preprint arXiv:2605.14434.

## A Related Works

Sparse retrieval methods like BM25 (Robertson et al., 2009) and TF-IDF (Ramos et al., 2003) rely on exact term matching but suffer from vocabulary mismatch. Conversely, dense retrieval approaches (e.g., DPR (Karpukhin et al., 2020), ANCE (Xiong et al., 2020)) map inputs into a shared vector space, enhancing semantic matching but necessitating costly external indices like HNSW (Malkov and Yashunin, 2018). To address these limitations, STAR (Zhan et al., 2021) introduces stable negative sampling, while ColBERT (Khattab and Zaharia, 2020) and Sentence-BERT (Reimers and Gurevych, 2019) refine fine-grained similarity and semantic understanding. Recent work also investigates scaling laws in dense models (Fang et al., 2024).

Personalized retrieval tailors results by incorporating user contexts. QEM (Ai et al., 2019a) investigates personalization mechanisms, while DREM (Ai et al., 2019b) models dynamic user-product relationships via knowledge graphs. Embedding-based approaches like ZAM (Ai et al., 2019a), HEM (Ai et al., 2017), and TEM (Bi et al., 2020) integrate user profiles and history into unified embeddings. In conversational domains, CHIQ (Mo et al., 2024) enhances accuracy by leveraging historical search queries. TTP (Jiang et al., 2026a) introduces a unified reasoning and retrieval framework for personalized dense retrieval.

Generative retrieval internalizes corpus knowledge to directly generate item identifiers. DSI (Tay et al., 2022) pioneered this paradigm, followed by semantic enhancements in SE-DSI (Tang et al., 2023a). Various optimization strategies have emerged: SEATER (Si et al., 2024) utilizes balanced k-ary trees; Tied-Atomic (Nguyen and Yates, 2023) connects GR with dense retrieval; Gen-RRL (Zhou et al., 2023) employs reinforcement learning; and LTRGR (Li et al., 2024) incorporates ranking tasks. TIGER (Rajput et al., 2023) first utilized RQ-VAE for SIDs. Regarding architecture, COBRA (Yang et al., 2025) integrates sparse and dense vectors, while Hi-gen (Wu et al., 2024), CAT-ID<sup>2</sup> (Liu et al., 2025), and MERGE (Zhang et al., 2025b) optimize ID generation by exploiting multi-level correlations.

Recent and concurrent studies have further revisited Semantic ID based generative retrieval from several complementary perspectives. A line of work improves the construction of Semantic IDs beyond item-only reconstruction. For example,

CQ-SID introduces category-aware and query-item contrastive objectives for e-commerce generative retrieval (Zhu et al., 2026), while DIGER (Fu et al., 2026) and UniSID (Jiang et al., 2026b) explore differentiable or end-to-end Semantic ID learning to better align identifier assignment with downstream recommendation objectives. Another direction studies the dynamics and scalability of Semantic IDs, such as mitigating collaborative Semantic ID staleness under evolving user-item interactions (Baikalov et al., 2026) and exploiting longer Semantic IDs to enhance representation capacity while controlling decoding cost (Xia et al., 2026). Closely related to our efficiency motivation, MLPbased distilled generative recommenders show that attention-heavy autoregressive decoders can be distilled into lightweight MLP-based heads for faster Semantic ID generation (Guo et al., 2026), suggesting that much of the decoder-side computation is redundant once global user context and prefix structure are properly preserved. These studies are largely orthogonal to CHAP: they mainly focus on identifier learning, temporal ID maintenance, or decoder distillation, whereas CHAP targets personalized generative retrieval by explicitly aligning query representations with the item quantization path, jointly modeling sparse Semantic IDs and dense representations, and using residual cascading generation to reduce autoregressive latency while preserving fine-grained personalized relevance.

## B Proof of Cascaded Sparse-Dense Modeling

Theorem 1 (Entropy reduction of CHAP’s cascaded modeling). Let X denote the personalized context encoded from $X _ { i n }$ , and let $\mathbf { C } _ { i }$ $( C _ { i } ^ { 0 } , \ldots , C _ { i } ^ { L - 1 } )$ and $\mathbf { V } _ { i }$ denote the random variables of the target item’s hierarchical SID and dense vector, respectively. Consider an independent dual-view factorization

$$
{ \cal P } _ { \mathrm { i n d } } ( { \bf C } _ { i } , { \bf V } _ { i } | X ) = \prod _ { l = 0 } ^ { L - 1 } { \cal P } ( C _ { i } ^ { l } | X ) \cdot { \cal P } ( { \bf V } _ { i } | X ) ,\tag{21}
$$

and CHAP’s cascaded sparse-dense factorization

$$
\begin{array} { l } { { \displaystyle { P } _ { \mathrm { c h a p } } ( \mathbf { C } _ { i } , \mathbf { V } _ { i } | X ) = \prod _ { l = 0 } ^ { L - 1 } P ( C _ { i } ^ { l } | C _ { i } ^ { < l } , X ) } } \\ { ~ \cdot P ( \mathbf { V } _ { i } | \mathbf { C } _ { i } , X ) . } \end{array}\tag{22}
$$

Denote their total uncertainties by $\mathcal { H } _ { \mathrm { i n d } }$ and ${ \mathcal { H } } _ { \mathrm { c h a p } } .$ where $H ( \cdot )$ is used for discrete SIDs and $h ( \cdot )$ for the continuous dense vector. Then

$$
\mathcal { H } _ { \mathrm { c h a p } } ( \mathbf { C } _ { i } , \mathbf { V } _ { i } | X ) \leq \mathcal { H } _ { \mathrm { i n d } } ( \mathbf { C } _ { i } , \mathbf { V } _ { i } | X ) .\tag{23}
$$

Equality holds if and only if each SID level is conditionally independent of its prefix given X, and $\mathbf { V } _ { i }$ is conditionally independent of $\mathbf { C } _ { i }$ given X.

Proof. Writing $P _ { \mathrm { i n d } }$ and $P _ { \mathrm { c h a p } }$ for the two distributions above, the independent dual-view entropy decomposes as

$$
\begin{array} { l } { \displaystyle \mathcal { H } _ { \mathrm { i n d } } = - \mathbb { E } _ { P _ { \mathrm { i n d } } } [ \log P _ { \mathrm { i n d } } ] } \\ { \displaystyle \quad = \sum _ { l = 0 } ^ { L - 1 } H ( C _ { i } ^ { l } | X ) } \\ { \displaystyle \qquad + h ( \mathbf { V } _ { i } | X ) . } \end{array}\tag{24}
$$

For CHAP’s cascaded factorization, we have

$$
\begin{array} { r l } & { \mathcal { H } _ { \mathrm { c h a p } } = - \mathbb { E } _ { P _ { \mathrm { c h a p } } } [ \log P _ { \mathrm { c h a p } } ] } \\ & { \quad \quad = \displaystyle \sum _ { l = 0 } ^ { L - 1 } H ( C _ { i } ^ { l } | C _ { i } ^ { < l } , X ) } \\ & { \quad \quad \quad + h ( \mathbf { V } _ { i } | \mathbf { C } _ { i } , X ) . } \end{array}\tag{25}
$$

By the non-negativity of conditional mutual information,

$$
H ( C _ { i } ^ { l } | X ) - H ( C _ { i } ^ { l } | C _ { i } ^ { < l } , X ) = I ( C _ { i } ^ { l } ; C _ { i } ^ { < l } | X ) \ge 0 ,\tag{26}
$$

and

$$
h ( \mathbf { V } _ { i } | X ) - h ( \mathbf { V } _ { i } | \mathbf { C } _ { i } , X ) = I ( \mathbf { V } _ { i } ; \mathbf { C } _ { i } | X ) \geq 0 .\tag{27}
$$

Substituting these inequalities into the two entropy decompositions yields

$$
\begin{array}{c} \mathcal { H } _ { \mathrm { i n d } } - \mathcal { H } _ { \mathrm { c h a p } } = \sum _ { l = 0 } ^ { L - 1 } I ( C _ { i } ^ { l } ; C _ { i } ^ { < l } | X )  \\ { + I ( \mathbf { V } _ { i } ; \mathbf { C } _ { i } | X ) \ge 0 . } \end{array}\tag{28}
$$

Therefore, CHAP’s coarse-to-fine SID generation followed by dense refinement does not increase the modeling uncertainty compared with independently predicting sparse and dense views. The equality condition follows directly from the equality condition of conditional mutual information: all terms above are zero only when $C _ { i } ^ { l } \perp C _ { i } ^ { < l } \perp X$ for every level l and $\mathbf { V } _ { i } \perp \mathbf { C } _ { i } \mid X$ . This completes the proof.

## C Dataset Details

In this section, we provide detailed descriptions of the four datasets used to evaluate the proposed CHAP framework. The detailed statistics are summarized in Tab. 6.

ESCI-us. We utilize the large version of the English subset from the ESCI<sup>1</sup> (Reddy et al., 2022) dataset. Unlike the widely used Amazon Product Reviews (He and McAuley, 2016) dataset, which necessitates constructing synthetic queries, ESCI contains authentic queries derived from actual user logs on Amazon, filled with the noise and complexity of real-world search traffic. Each query in ESCI is associated with products labeled by relevance levels: Exact (E), Substitute (S), Complement (C), and Irrelevant (I). This dataset is characterized by complex linguistic patterns, including negation conditions (e.g., “not” and “without”) and intricate attribute constraints (e.g., dimensions, price, functionality). These factors result in a low text overlap between queries and product descriptions, making it a rigorous benchmark for evaluating a model’s semantic understanding capabilities. During sequence modeling, we treat items labeled $\mathbf { \tilde { e } } \mathbf { \tilde { e } } ^ { , , }$ and $\mathbf { \tilde { s } } \mathbf { \tilde { s } } \mathbf { \Psi }$ as positive samples. Notably, unlike existing works (Liu et al., 2025; Zhang et al., 2025b) that utilized the small version of the dataset and applied specific cleaning and preprocessing steps, we directly adopt the large version with raw query and item texts. We strictly follow the official predefined train and test splits without any additional preprocessing to rigorously test the model’s robustness and generalization capabilities.

KuaiSearch. KuaiSearch<sup>2</sup> (Li et al., 2026) is a large-scale e-commerce search dataset built upon real user search interactions from the Kuaishou platform. In contrast to existing datasets that often rely on heuristically constructed queries or aggressively filter out inactive entities, KuaiSearch preserves authentic user queries and natural-language product texts without popularity-based filtering. This ensures the retention of cold-start users and long-tail products, accurately reflecting real-world e-commerce long-tail distributions. The whole dataset contains approximately 330,000 users, 18,000,000 products, and 2,500,000 real search queries. Furthermore, it provides extensive longterm user behavior sequences and systematically covers the core stages of industrial search pipelines.

Table 6: Detailed statistics of the four datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">#Items</td><td rowspan="2">#Queries</td><td rowspan="2">#Q-I Pairs</td><td colspan="2">Train</td><td colspan="2">Test</td><td rowspan="2">Avg. Hist.</td></tr><tr><td>#Queries</td><td>#Q-I Pairs</td><td>#Queries</td><td>#Q-I Pairs</td></tr><tr><td>ESCI-us</td><td>1,215,854</td><td>97,346</td><td>1,818,825</td><td>74,888</td><td>1,393,063</td><td>22,458</td><td>425,762</td><td></td></tr><tr><td>KuaiSearch</td><td>6,634,118</td><td>295,561</td><td>682,228</td><td>266,004</td><td>613,696</td><td>29,557</td><td>68,532</td><td>10.37</td></tr><tr><td>Amazon</td><td>35,772</td><td>9,070</td><td>9,070</td><td>6,896</td><td>6,896</td><td>2,174</td><td>2,174</td><td>40.12</td></tr><tr><td>Local-Life</td><td>10,145,542</td><td>3,089,026</td><td>3,089,026</td><td>2,984,294</td><td>2,984,294</td><td>104,732</td><td>104,732</td><td>7.47</td></tr></table>

In our experiments, we specifically utilize the recall subset. During data construction, we treat both clicked and purchased items as positive samples. Strictly following the methodology described in the original paper, we apply identical data cleaning steps—filtering out entries that lacked explicit search queries or had no positive samples—and subsequently partition the cleaned dataset into training and testing sets using a random 9:1 split.

Amazon. To evaluate our model’s capacity for fine-grained personalized retrieval, we utilize the Amazon search dataset from the Personalized Web Agent Benchmark (PersonalWAB)<sup>3</sup> (Cai et al., 2025). Unlike traditional static search benchmarks, PersonalWAB is explicitly designed to assess the integration of user profiles and historical web behaviors in understanding customized instructions. The dataset originates from real-world Amazon interaction logs and encompasses diverse user profiles alongside tens of thousands of historical web behaviors. It features highly personalized, humansynthesized natural language instructions tailored to each user’s profile, requiring models to dynamically infer specific user preferences. This dataset presents a unique challenge for generative retrieval models, as it necessitates a deep semantic fusion of the current query, user profiles, and heterogeneous historical behaviors to achieve accurate personalized product search. To fully preserve the complexity and noise of these customized instructions, we utilized this dataset in its original form and directly adopted the officially pre-defined train and test splits without performing any data cleaning or preprocessing.

Local-Life. To assess performance in a dynamic, personalized environment, we introduce a largescale industrial dataset collected from 30 days of user interaction logs on a leading Local-Life service platform. Compared to public datasets like KuaiSAR (Sun et al., 2023) or JDSearch (Liu et al., 2023), which only provide encrypted token IDs preventing semantic verification, this dataset comprises raw natural language text. This transparency is crucial for verifying our model’s ability to align semantics between queries and items. The dataset covers diverse scenarios (e.g., Dining, Entertainment, Travel) and heterogeneous search intents (e.g., Brand Search, Ambiguous Intent, Exact Item Search, and Long-tail or Cold-start Search), comprehensively reflecting real-world user behaviors. We construct interaction sequences from the first 29 days for training and the last day for testing.

## D Baseline Descriptions

To rigorously evaluate the proposed CHAP framework, we comprehensively compare it against 14 representative baselines categorized into sparse retrieval, dense retrieval, personalized retrieval, and generative retrieval (GR) methods.

Sparse Retrieval Methods.

• BM25 (Robertson et al., 2009): A classical probabilistic information retrieval model that relies on exact lexical matching based on term frequency and inverse document frequency.

• Doc2Query (Nogueira et al., 2019): A document expansion technique that utilizes a sequence-to-sequence model to predict potential queries for each document prior to indexing, thereby enriching the vocabulary and mitigating the vocabulary mismatch problem.

• DeepCT (Dai and Callan, 2019): A deep contextualized term weighting framework that leverages BERT-based text representations to estimate the document-specific semantic importance of terms, translating them into enhanced term weights for first-stage sparse retrieval.

## Dense Retrieval Methods.

• DPR (Karpukhin et al., 2020): Dense Passage Retriever, a highly effective bi-encoder architecture optimized via in-batch and hard negative contrastive learning to map queries and items into a shared semantic vector space.

• Sen-T5 (Ni et al., 2022): A scalable sentence encoding model built upon the T5 framework, pre-trained on massive text-to-text and contrastive tasks to generate high-quality, continuous text representations.

• MPNet (Song et al., 2020): A robust unified pre-training framework that successfully combines the advantages of masked language modeling (e.g., BERT) and permuted language modeling (e.g., XLNet) for enhanced document embeddings.

## Personalized Retrieval Methods.

• TEM (Bi et al., 2020): Transformer-based Embedding Model, which explicitly encodes user search history alongside current queries to construct dynamic user profiles, aligning retrieval representations with personalized intents.

• CoPPS (Dai et al., 2023): Contrastive learning for Personalized Product Search, a framework that discriminates fine-grained user intents by applying multi-view contrastive learning over user behavior sequence representations.

## Generative Retrieval (GR) Methods.

• DSI (Tay et al., 2022): A pioneering end-toend differentiable search index that directly generates relevant document identifiers using a Transformer memory based on semantic clustering.

• NCI (Wang et al., 2022): Neural Corpus Indexer, a strong GR baseline that utilizes a prefix-aware decoder and a query generation strategy to optimize hierarchical semantic identifiers effectively.

• TIGER (Rajput et al., 2023): A generative recommender and retrieval model that leverages a Residual Quantized Variational AutoEncoder (RQ-VAE) to generate hierarchical, semantically rich discrete IDs strictly based on item textual features.

• LTRGR (Li et al., 2024): Learning-to-Rank in Generative Retrieval, which integrates ranking-aware principles into the generation process by optimizing the log-probabilities of generation according to target ranking metrics like ListMLE.

• MERGE (Zhang et al., 2025b): A recent generative retrieval framework that aims to optimize identifier assignments by aligning multiple positive target items across varying relevance signals for multi-level ranking.

• COBRA (Yang et al., 2025): A state-of-theart cascading generative framework that unites sparse identifier generation and dense representation regression within a joint optimization landscape to improve retrieval coverage and accuracy.

## E Evaluation Metrics Details

To comprehensively evaluate the retrieval accuracy and ranking quality of the proposed CHAP framework, we employ four widely adopted metrics: Recall, Hit Ratio (HR), Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG). Here, we detail their formulations and the specific rationales for selecting them based on the characteristics of different datasets.

Recall (R@K). Recall evaluates the model’s ability to retrieve all relevant items. It provides a macroscopic view of coverage, which is essential for measuring overall semantic alignment.

$$
\mathsf { R } @ K = \frac { 1 } { | Q | } \sum _ { q \in Q } \frac { | \mathcal { R } _ { q } \cap \hat { \mathcal { I } } _ { q , K } | } { | \mathcal { R } _ { q } | } ,\tag{29}
$$

where Q is the set of queries, $\mathcal { R } _ { q }$ denotes the ground-truth relevant items for query q, and $\hat { \mathcal { T } } _ { q , K }$ represents the top-K retrieved candidates.

Hit Ratio (HR@K). Hit Ratio measures the probability that a user’s target item successfully appears in the top-K retrieved list. It is formulated as an indicator function:

$$
\mathrm { H R } @ K = \frac { 1 } { | Q | } \sum _ { q \in Q } \mathbb { I } ( | \mathcal { R } _ { q } \cap \hat { \mathcal { T } } _ { q , K } | > 0 ) ,\tag{30}
$$

where I(·) evaluates to 1 if the condition is true and 0 otherwise. Rationale: In practical industrial scenarios (e.g., e-commerce purchases or local-life orders), user intent is often highly focused on a single specific item. HR effectively reflects the success rate of fulfilling this distinct user intent, making it highly suitable for evaluating performance on datasets like Local-Life and KuaiSearch.

Mean Reciprocal Rank (MRR@K). MRR strictly assesses the ranking quality by focusing on the position of thefirst relevant item retrieved:

$$
\mathrm { M R R } @ \mathrm { { K } } = \frac { 1 } { | Q | } \sum _ { q \in Q } \frac { 1 } { \mathrm { { r a n k } } _ { q } ^ { * } } ,\tag{31}
$$

where rank<sup>∗</sup> is the rank position of the first relevant item in the top-K list (if no relevant item is in the top K, the score is 0). Rationale: For single-target or exact-intent datasets (such as PersonalWAB and Local-Life), users typically stop browsing once they find the exact match. MRR rigorously penalizes models that place the target item lower down the list, directly correlating with user satisfaction and online CTR.

Normalized Discounted Cumulative Gain (NDCG@K). NDCG evaluates the ranking quality by considering both the presence and the position of multiple relevant items, factoring in multi-level relevance scores:

$$
\mathrm { D C G } @ K = \sum _ { i = 1 } ^ { K } \frac { 2 ^ { r e l _ { i } } - 1 } { \log _ { 2 } ( i + 1 ) } ,\tag{32}
$$

$$
\mathsf { N D C G @ } K = \frac { \mathsf { D C G @ } K } { \mathsf { I D C G @ } K } ,\tag{33}
$$

where $r e l _ { i }$ is the graded relevance score of the item at rank $i ,$ and IDCG@K is the ideal DCG obtained by perfectly sorting all relevant items. Rationale: For datasets like ESCI-us, a single query often maps to multiple items with varying relevance levels (e.g., Exact, Substitute, Complement). NDCG is uniquely capable of capturing these fine-grained grading differences, rewarding models that place highly relevant items at the top while tolerating lower-ranked partially relevant items.

## F Implementation Details for CHAP and Baselines

To ensure reproducibility and address concerns regarding experimental fairness, particularly about candidate generation budgets and ranking protocols, we provide the comprehensive implementation and optimization details for our proposed CHAP framework and all evaluated baseline models.

## F.1 Training Details of CHAP

The training of CHAP consists of two primary stages. In the SID Learning Stage, we pre-train the RQ-VAE to discretize item embeddings. We use the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 1024, training the autoencoder for 500 epochs with the commitment loss weight $\beta = 0 . 5$ to ensure semantic stability and avoid codebook collapse. In the Hierarchical Semantic Alignment and Generation Stage, we freeze the pre-trained item codebook and fine-tune the Transformer backbone along with the alignment modules. This stage is trained for 50 epochs using the AdamW optimizer with a reduced learning rate of $1 \times 1 0 ^ { - 5 }$ , a warm-up ratio of 0.1, and a linear learning rate decay schedule. The maximum sequence length for the dual-view user behavior input is truncated to 512 tokens to balance context modeling and computational memory. All experiments are implemented in PyTorch and conducted on eight NVIDIA A100 (80GB) GPUs.

## F.2 Implementation Settings of Baselines

To ensure a strictly fair, apples-to-apples comparison, all deep learning-based baselines are equipped with backbones of comparable capacity to CHAP and are provided with identical raw textual features and historical sequence lengths. Specifically, for English datasets, we uniformly use BERT-base and T5-base; for Chinese datasets, we replace them with Chinese-BERT-base and mT5-base to handle the specific tokenization and vocabulary. BERTbase has about 110M parameters, Chinese-BERTbase about 102M, T5-base about 220M, and mT5- base about 580M. All GR baselines reported in Tab. 4 use the same-capacity T5-base or mT5-base backbone as CHAP, so the efficiency comparison mainly reflects framework-level design differences rather than backbone scale.

Sparse Retrieval. We utilize the Pyserini toolkit for BM25, setting parameters $k _ { 1 } ~ = ~ 1 . 2$ and $b = 0 . 7 5$ . For Doc2Query, we generate 40 synthetic queries per document using a pre-trained T5 (or mT5) model before indexing. For DeepCT, we employ the respective BERT-base models to project contextualized token embeddings into term weights.

Dense and Personalized Retrieval. For DPR, Sen-T5, MPNet, TEM, and CoPPS, we reproduce their results on this dataset using the official opensource implementations and default training strategies. For personalized models (TEM and CoPPS), we explicitly feed the same historical interaction sequences (up to the exact same length as used in CHAP) into their sequence encoders to construct the user profile embeddings. Crucially, their dualtower inner-product search is accelerated using the FAISS library to perform exhaustive vector search over the entire corpus. This ensures they are evaluated under an unbounded candidate budget, making the comparison strictly fair against CHAP, which operates under a restricted generative candidate pool.

Generative Retrieval (GR) and Ranking Fairness. To eliminate discrepancies caused by model scale and evaluate the intrinsic capability of each framework, all GR baselines (DSI, NCI, TIGER, LTRGR, MERGE, COBRA) are implemented using the same T5-base (or mT5-base) backbone. For ItemID construction in DSI and NCI, we employ Hierarchical K-Means on BERT embeddings to build semantic tree IDs. For TIGER and CO-BRA, we utilize their default RQ-VAE architecture configured with depth $L = 3$ and codebook size $K = 5 1 2$ , strictly matching our CHAP’s dimensionality.

To directly address potential concerns regarding evaluation fairness, particularly the candidate generation budgets: no external heavy rerankers (e.g., Cross-Encoders) are applied to any baseline or our CHAP model. During inference, we ensure a rigorously fair evaluation by restricting all GR models to the exact same candidate budget $( M = 5 0 )$ . Crucially, rather than enforcing an identical decoding algorithm that might artificially impair baseline performance (e.g., generating invalid IDs), we allow all GR baselines to employ their optimal Constrained Beam Search with a beam width of 50. In contrast, leveraging our decoupled architecture, CHAP efficiently generates 50 candidates via a Parallel Sampling strategy. For pure generative baselines, their final ranking relies exclusively on the beam search probabilities over their 50 candidates; for hybrid baselines (e.g., COBRA, MERGE), they utilize their internal scoring mechanisms over the same pool. Similarly, CHAP simply fuses its internally generated sparse probabilities and dense similarities in a parameter-free manner within its 50 samples.

By aligning the language backbone, enforcing identical candidate budgets $( M = 5 0 )$ , maintaining the optimal decoding integrity of baselines, and restricting all models to internal parameter-free scoring, we guarantee that CHAP’s superior ranking accuracy and remarkable inference throughput (QPS) stem entirely from our core architectural innovations.

## G Ablation Variant Definitions

In our ablation study (Sec. 5.3), we systematically deconstruct CHAP to validate the contribution of each module. The specific configurations of the ablated variants are defined as follows:

(A) Hierarchical Semantic Alignment (HSA) Variants:

• w/o HSA: Replaces our trainable, aligned query encoder with a standard RQ-VAE encoder whose weights are strictly frozen after the item-side pre-training stage.

• w/o $\mathcal { L } _ { \mathrm { { C S A } } } \mathrm { { : } }$ Excludes the cross-sample alignment losses (both Cross-Commitment and Cross-Reconstruction), relying only on contrastive and distillation objectives.

• w/o L<sub>HCL</sub>: Removes the hierarchical-aware contrastive learning objective, eliminating the explicit multi-granular alignment penalty.

• w/o $\mathcal { L } _ { \mathrm { D i s t i l l } } \colon$ Omits the soft probability distillation module that regularizes the fine-tuned encoder’s output distribution.

## (B) Sequence Modeling Variants:

• w/o Dense Refinement: Eliminates the dense continuous vector from both the input sequence and the generation target. The model relies entirely on discrete sparse SIDs for sequence modeling and ranking.

## (C) Residual Cascading Generation Variants:

• w/o Residual Gen. (Autoregressive): Replaces the lightweight layer-wise residual fusion mechanism with a standard flat autoregressive generation, forcing the heavy Transformer Decoder to execute sequentially for all L layers.

• w/o Self-Attention: Retains the residual cascading mechanism but structurally removes all Masked Self-Attention sub-layers within the Transformer Decoder, retaining only the Cross-Attention mechanism to interact with the encoded history.

• w/o Decoder (Pooling + MLP): Entirely removes the Transformer Decoder. Instead, it relies on mean pooling over the encoder’s outputs to derive a static global context vector, which is then fed directly into the residual blocks for SID and dense vector prediction.

## H Definitions of Intent-wise Query Segments

In industrial Local-Life search scenarios, user intents are highly heterogeneous. To rigorously evaluate the model’s robustness and the specific contribution of each architectural component, we extract four intent-wise query segments from the Local-Life dataset based on query attributes and historical interaction frequencies. Notably, to ensure a rigorous and realistic evaluation, the Ambiguous, Cold-Start, and LongTail subsets share the exact same entire candidate item pool (i.e., over 10 million items) as the General set during retrieval. The detailed statistics of these segments are summarized in Tab. 7.

• General: This segment is exactly the entire set of the Local-Life dataset. It encompasses the natural flow of daily user search traffic, reflecting the macroscopic ranking ability of the model across all scenarios.

• Ambiguous: This segment comprises short, highly ambiguous queries with generalized intents (e.g., “Dinner”, “Nearby attractions”, “Weekend entertainment”). These queries typically map to a vast number of potential candidate items across multiple categories. Excelling in this segment rigorously tests the model’s Personalized Sequence Modeling capability, as accurate retrieval strictly requires deducing explicit user preferences from historical behaviors to disambiguate the broad intent.

• ColdStart: This segment targets queries that exclusively lead to items introduced to the platform within the last 14 days. These items have extremely sparse or zero historical click logs. Standard generative models typically struggle on these data because they lack collaborative training signals for the new SIDs and fail to integrate dense representations for semantic fallback. This scenario directly evaluates the model’s pure zero-shot semantic matching capability.

• LongTail: This segment focuses on queries whose historical interaction frequencies fall into the bottom 10% of the entire search logs (e.g., highly specific long-tail constraints, niche store names, or complex negated conditions). Similar to the ColdStart segment, standard GR models often exhibit poor performance on these data because they do not combine dense representations to capture precise textual semantics, falling back instead on memorized popularity bias. In contrast, excelling in this segment heavily relies on our model’s intrinsic Hierarchical Semantic Alignment (HSA) and dual-view design to accurately map complex linguistics.

## I Additional Diagnostics for Semantic Alignment

In addition to visualization, we use clusteringoriented quantitative diagnostics to measure whether the aligned representation space becomes more compact and separable. We consider Silhouette Score (Rousseeuw, 1987), Calinski-Harabasz Index (Calinski and Harabasz´ , 1974), and Davies-Bouldin Index (Davies and Bouldin, 1979) over coarse product-category clusters constructed from the sampled encoder representations. Higher Silhouette and Calinski-Harabasz scores, together with a lower Davies-Bouldin score, indicate better intra-category compactness and inter-category separation. Tab. 8 reports the corresponding metrics computed from 10,000 sampled encoder representations in ESCI-us using product-category labels as cluster assignments.

Following the matrix-style diagnostic in prior cascaded retrieval analysis, Fig. 7 further provides a similarity-matrix view of the same trend. RQ-VAE shows a relatively noisy off-diagonal background, suggesting that item-only quantization leaves residual similarity between unrelated product categories. MERGE produces a clearer diagonal structure but still contains coarse diagonal blocks, indicating that its multi-level merging improves local consistency while retaining some coarse-grained semantic entanglement. CHAP exhibits a sharper localneighborhood diagonal and a lighter off-diagonal background, suggesting that HSA better concentrates category-consistent representations while reducing spurious cross-category similarity.

Table 7: Detailed statistics of the intent-wise query segments extracted from the Local-Life dataset.
<table><tr><td rowspan="2">Segment</td><td rowspan="2">#Items</td><td rowspan="2">#Queries</td><td rowspan="2">#Q-I Pairs</td><td colspan="2">Train</td><td colspan="2">Test</td><td rowspan="2">Avg. Hist.</td></tr><tr><td>#Queries</td><td>#Q-I Pairs</td><td>#Queries</td><td>#Q-I Pairs</td></tr><tr><td>General</td><td>10,145,542</td><td>3,089,026</td><td>3,089,026</td><td>2,984,294</td><td>2,984,294</td><td>104,732</td><td>104,732</td><td>7.47</td></tr><tr><td>Ambiguous</td><td>10,145,542</td><td>10,000</td><td>10,000</td><td></td><td></td><td>10,000</td><td>10,000</td><td>6.21</td></tr><tr><td>ColdStart</td><td>10,145,542</td><td>5,000</td><td>5,000</td><td></td><td></td><td>5,000</td><td>5,000</td><td>5.74</td></tr><tr><td>LongTail</td><td>10,145,542</td><td>10,000</td><td>10,000</td><td></td><td></td><td>10,000</td><td>10,000</td><td>10.16</td></tr></table>

![](images/cf2c7c202b93c2ce088b247de02a5725f3a886f84f342c812fa58157e4621ed0.jpg)  
Figure 7: Similarity matrix diagnostics over 10,000 sampled queries from ESCI-us. The figure compares pairwise similarity matrices of RQ-VAE, MERGE, and CHAP, where CHAP exhibits a clearer local-neighborhood structure and fewer coarse diagonal blocks after semantic alignment.

<table><tr><td>Method</td><td>Silhouette ↑</td><td>CH Index ↑</td><td>DB Index ↓</td></tr><tr><td>RQ-VAE</td><td>0.218</td><td>612.4</td><td>1.684</td></tr><tr><td>MERGE</td><td>0.331</td><td>948.7</td><td>1.153</td></tr><tr><td>CHAP</td><td>0.387</td><td>1136.2</td><td>0.912</td></tr></table>

Table 8: Semantic alignment diagnostics computed from 10,000 sampled encoder representations in ESCI-us.

## J SID-Level Visualization Examples

Following the qualitative SID analyses in MERGE (Zhang et al., 2025b), we further inspect how CHAP organizes items across hierarchical SID layers. We use the real item-to-SID mappings from RQ-VAE, MERGE, and CHAP on ESCI-us, and select representative queries whose candidate items contain multiple relevance levels.

Fig. 8 visualizes the layer-wise SID concentration across six representative ESCI-us queries. For each query, we count the number of unique SID prefixes occupied by query-relevant items at each layer. Lower curves therefore indicate that relevant items are assigned to fewer hierarchical branches. Compared with RQ-VAE and MERGE, CHAP consistently produces more concentrated layer-wise distributions, especially at deeper SID layers. This trend suggests that HSA encourages the SID hierarchy to better follow query-conditioned relevance rather than only item-side lexical similarity.

To examine the semantic meaning of SID layers, Fig. 9 summarizes frequent product-title keywords along one aligned camera-related SID path. The first-layer prefix (133) captures a coherent coarsegrained visual-device cluster, including camera, security camera, webcam, night vision, and motion detection products. The second-layer prefix (133, 121) narrows this cluster toward webcam and USB video-camera products, while the third-layer prefix (133, 121, 293) further concentrates on webcam products with microphone, desktop, laptop, and streaming attributes. This coarse-to-fine keyword evolution is consistent with CHAP’s hierarchical alignment objective.

Tab. 9 provides item-level examples for a separate query, “crustless sandwich maker”. Since CHAP is initialized from and fine-tuned upon the RQ-VAE codebook, the RQ-VAE and CHAP SID columns are comparable. In contrast, MERGE learns a separate codebook with its alignment loss, so its numeric SIDs should only be interpreted within MERGE’s own codebook space rather than compared token by token with RQ-VAE or CHAP. In these selected examples, exact sandwich cutter and sealer products are mapped to the same CHAP SID, whereas irrelevant items such as generic cookie cutters, handheld sandwich grills, and craft makers remain separated. This example illustrates

![](images/a5f04ca302dbbad9f656941e3727ca13cd06bd6961dedbb1c4c8a36e4597bdce.jpg)

Figure 8: Layer-wise SID concentration over multiple representative ESCI-us queries. The curves are computed from the real item-to-SID mappings of RQ-VAE, MERGE, and CHAP. The y-axis reports the number of unique SID prefixes among query-relevant items, where lower values indicate more compact hierarchical assignments.  
![](images/941cb5fc58a79ebe4895e101bff0f80a475e10d8ffd16eb4d9f7345fb4a10fcc.jpg)  
Figure 9: Product-title keyword-cloud visualization along an aligned camera-related SID path. The prefixes become progressively more specific from a broad camera-related cluster to webcam-oriented products and then to webcam products with microphone and streaming attributes.

how aligned SIDs improve the consistency of gen- the original work of the human authors.   
erated ItemIDs for query-relevant items.

## K AI Usage Disclosure

In compliance with the conference guidelines regarding the use of artificial intelligence tools, we transparently disclose our light usage of AI assistants during the preparation of this manuscript. Specifically, AI tools were strictly limited to serving as utility aids for language polishing (e.g., correcting grammar, refining sentence structures, and improving readability for non-native speakers) and basic programming support (e.g., generating boilerplate scripts for data visualization).

The AI assistants did not generate any novel scientific claims, experimental designs, or core methodology. All research conceptualization, model implementation, data analysis, and the intellectual contributions of this paper are exclusively

<table><tr><td>Label</td><td>Product Title</td><td>RQ-VAE SID</td><td>MERGE SID</td><td>CHAP SID</td></tr><tr><td>E</td><td>Sandwich Cutter and Sealer for Kids</td><td>(14, 127, 331)</td><td>(287, 375, 142)</td><td>(14, 127, 294)</td></tr><tr><td>E</td><td>Decruster Sandwich Maker Set</td><td>(14, 502, 331)</td><td>(287, 64, 411)</td><td>(14, 127, 331)</td></tr><tr><td>E</td><td>DIY Uncrustables Sandwich Cutters</td><td>(160, 127, 393)</td><td>(287, 64, 411)</td><td>(14, 127, 331)</td></tr><tr><td>E</td><td>Fun Sandwich Cutters for Lunch Box</td><td>(14, 127, 441)</td><td>(287, 64, 156)</td><td>(14, 127, 226)</td></tr><tr><td>I</td><td>Cookie Cutters with Irreverent Phrases</td><td>(435, 359, 375)</td><td>(93, 382, 27)</td><td>(435, 241, 88)</td></tr><tr><td>I</td><td>Toas-Tite Handheld Sandwich Grill</td><td>(472, 154, 266)</td><td>(316, 45, 390)</td><td>(472, 308, 119)</td></tr><tr><td>I</td><td>Cricut Champagne Maker</td><td>(46, 491, 265)</td><td>(446, 220, 108)</td><td>(46, 73, 402)</td></tr><tr><td>Q</td><td>crustless sandwich maker</td><td>(14, 127, 393)</td><td>(287, 64, 411)</td><td>(14, 127, 331)</td></tr></table>

Table 9: Comparison of generated SIDs under the query “crustless sandwich maker” using RQ-VAE, MERGE, and CHAP.