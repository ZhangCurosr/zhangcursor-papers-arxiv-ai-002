# Tlow: Flow-based Item Tokenizer for Recommendation

Nian Li<sup>∗</sup> Tsinghua University Shenzhen, China

Lingling Yi Tencent Inc. Shenzhen, China

Chonggang Song<sup>∗</sup> Tencent Inc. Shenzhen, China

Yong Li   
Tsinghua University   
Beijing, China

Jingtao Ding Tsinghua University Beijing, China

Qingmin Liao Shenzhen International Graduate School, Tsinghua University Shenzhen, China

## Abstract

Item tokenizer encodes semantic embeddings into token IDs to replace the randomly assigned item IDs used in traditional recommendation models, fundamentally addressing the problems of excessive parameters and cold starts. However, the most common tokenizer, RQ-VAE, sufers from low decoding eficiency due to the inherent dependencies among its codebooks. Meanwhile, eficient indepen dent tokenizers such as optimized product quantization (OPQ) still struggle with dimensional correlations and distribution complexity of semantic embeddings. In this work, we propose a flow-based item Tokenizer (Tlow) to transform raw semantic embeddings into a latent space where embeddings conform to a unified standard normal distribution, achieving dual advantages of dimensional independence and distributional simplicity. Independent tokenization performed on these latent embeddings yields semantically clear token IDs. Additionally, we introduce a novel codebook guidance to align the codebook space with the token embedding space, further aiding the learning of more semantically distinct token embeddings. Ofline experiments on four public datasets demonstrate that Tlow’s tokenization and codebook guidance significantly improve recommendation performance. The improvement on cross-domain and multi-modal recommendations also proves the efectiveness of item tokenization in a simplified embedding space. Online experiments for a multi-modal retrieval task on China’s largest social media platform WeChat validate Tlow’s powerful distribution transformation capability. The retrieval model based on token IDs improves user CTR by 10.32% globally and by 11.64% for new items. Our codes are available at https://github.com/wjjln/Tlow.

## CCS Concepts

• Information systems → Retrieval models and ranking.

## Keywords

Recommender System; Item Tokenizer; Flow-based Model

ACM Reference Format: Nian Li, Chonggang Song, Jingtao Ding, Lingling Yi, Yong Li, and Qing min Liao. 2026. Tlow: Flow-based Item Tokenizer for Recommendation. In

<sup>∗</sup>Both authors contribute equally to this work.

Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management(CIKM’26), November07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3799682.3840087

## 1 Introduction

Traditional recommender systems often assign a random ID embedding to each item, with learning embeddings as the model’s core objective. However, embeddings learned by ID-based models (GNNs [29], RNNs [3], or self-attention [8, 40]) are completely independent of one another, so the number of parameters grows linearly with the number of items, limiting model capacity in practical online recommender systems. Moreover, the cold-start issue is another significant drawback, i.e., the model struggles to recommend new items with few interaction records.

Inspired by the training paradigm of large language models, some studies now tokenize items based on their semantic embeddings to generate a sequence of discrete token IDs (also called semantic IDs) [4, 5, 22], creating a globally shared token vocabulary. Sequential recommendation models then learn token embeddings by decoding the token IDs of the next item. This approach relies on token embeddings to model atomic information in the semantic space of all items, efectively constraining the model’s parameter size while naturally resolving the cold-start issue. A representative work, TIGER [22], uses residual-quantized variational auto-encoder (RQ-VAE) [33] as its tokenizer, applying hierarchical encoding to semantic embeddings. This creates strong correlations between codebooks, so decoding the next token ID relies on previously decoded IDs, causing an eficiency bottleneck: the number ofinference steps must equal the number of codebooks. Moreover, the codebook collapse and conflicts common in RQ-VAE [12] make its industrial application require substantial experience and expertise [39]. Pursuing independent tokenization to enable eficient parallel decoding is thus a natural next step, and this decoding-eficiency advantage over RQ-VAE-based sequential decoding has been empirically validated in [5]. Existing methods using product quantization (PQ) [7] directly partition semantic embeddings and encode each part independently [4], but still face two critical challenges:

• Dimension correlations. The correlations between the dimensions ofthe semantic embeddings violate the dimensional-independence assumption required for independent tokenization. Although optimized product quantization (OPQ) [2] alleviates this by decomposing the raw space into subspaces via an orthogonal transformation, dimensions within each subspace remain correlated [5]. When dimensions are correlated, embeddings lie on a complex,

skewed manifold, but independent quantization imposes a gridlike structure over it, leading to poor quantization [35].

• Complex embedding distribution. The unknown, complex distribution harms the semantic representational ability of embeddings [13] and causes large quantization errors when tokenized directly. Specifically, these embeddings are often highly anisotropic, occupying a non-uniform cone rather than being evenly spread [7], making standard quantization codebooks an in eficient fit, especially in cross-domain or multi-modal scenarios where embeddings from diverse sources form distinct, separated clusters.

In this work, we propose Tlow, a flow-based item tokenizer that transforms raw semantic embeddings into latent embeddings conforming to a standard normal distribution. Compared to the raw, correlated, and anisotropic embedding space, the transformed latent space better satisfies the dimensional-independenc assumption underlying PQ-based independent quantization [2, 7], mitigating the quantization error induced by correlated and skewed distributions. The resulting latent embeddings combine dimensional independence with distributional simplicity, facilitating more accurate tokenization and more semantically distinct token embeddings. Specifically, Tlow processes semantic embeddings with a multiscale architecture [9], built from basic units called “A Step of Flow”, each comprising three layers: Activation Normalization (ActNorm), Invertible Linear, and Afine Coupling. We apply PQ on the latent embeddings to generate token IDs for each item. Figure 1 compares 16-bit (i.e., 16-part) OPQ and Tlow tokenization on the “CDs and Vinyl” dataset from Amazon Reviews [17], showing albums sharing the same first-position token ID. Although these albums are all related to the “Pop” genre, OPQ groups in many albums of diferent genres (e.g., “Metal”, “British Invasion”, “Jazz”), whereas Tlow’s tokenization yields distinct, clear semantics, with most albums genuinely “Pop”. Tlow also exhibits significantly lower genre diversity than OPQ, as indicated by fewer categories and lower entropy. We then use user behavioral data to train the sequential recommenda tion model and the globally shared token embeddings, introducing a novel codebook guidance that aligns the token embedding space with the codebook space so token embeddings capture distinct semantics and atomic information.

We conduct extensive experiments on four Amazon Reviews datasets from diferent commodity categories. Tlow significantly improves recommendation performance given the same sequential model, validating the efectiveness oftokenization within a standard normal distribution space; ablation studies further confirm the importance of the latent embeddings’ semantic information and the codebook guidance. Performance gains in cross-domain and multimodal recommendations further support Tlow’s efectiveness in simplifying complex semantic embedding distributions, although the improvement is not uniformly largest across every domain and modality metric. On China’s largest social media platform WeChat, we conduct online A/B testing for a multi-modal picture<sup>1</sup> retrieval task, where replacing item IDs with Tlow’s token IDs yields significant gains in conversion metrics, especially for newly published pictures.

![](images/a71d51e1c0b75b8bec8cfb750c09fec51c37d1a163e10fc376ec53771d0d6563.jpg)  
Figure 1: A Comparison of semantic clarity of token IDs generated by OPQ and Tlow tokenization.

## 2 Tlow

In this work, we formulate the task as sequential recommendations. Specifically, based on each user’s historical interaction sequence $\{ i _ { 1 } , i _ { 2 } , \cdots , i _ { T } \}$ , the recommendation model aims to predict the next interaction item $i _ { T + 1 } ,$ where � is the sequence length. Traditional ID-based models assign a random ID for each item and learn ID embeddings through sequential models like Transformer decoder. However, ID-based models sufer from inherent problems of cold-start issues, excessive parameters, and so on. Item tokenizer leverages the semantic embedding x to transform each item into a sequence of token IDs, where x is obtained from pre-trained models with item features (e.g., title, cover image) as the input. In this way, tokenization-based sequential models have the objective of decoding token IDs of the next item, similar to the training paradigm of large language models. This approach overcomes the aforementioned fundamental problems by learning a shared set of token embeddings.

## 2.1 Overview

As stated in the introduction, dimensional correlations and complex distribution of semantic embeddings lead to challenges of independent tokenization. In this work, we propose Tlow with a multi-scale architecture [9] to transform semantic embedding $\textbf { x } \in \mathbb { R } ^ { d _ { s } }$ into latent embedding $\mathbf { z } \in \mathbb { R } ^ { d _ { s } }$ conforming to a standard normal distribution, where $d _ { s }$ is the embedding dimension. We can perform more accurate independent tokenization on z, generating token IDs with clearer semantics. Like all flow-based models, Tlow’s log-likelihood objective is also to minimize:

$$
\mathcal { L } _ { f } = - \frac { 1 } { \vert \boldsymbol { X } \vert } \sum _ { \mathbf { x } \in \boldsymbol { X } } \log p _ { \boldsymbol { \theta } } ( \mathbf { x } ) ,\tag{1}
$$

where X denotes the set ofsemantic embeddings ofall the items and � are learnable model parameters. The probability density function of Tlow can be further written as:

$$
\log p _ { \theta } ( \mathbf { x } ) = \log p _ { \theta } ( \mathbf { z } ) + \log | \operatorname* { d e t } ( \mathrm { d } \mathbf { z } / \mathrm { d } \mathbf { x } ) | ,\tag{2}
$$

where dz/dx is the Jacobian matrix of the transformation from x to z, and det(·) denotes the matrix determinant.

Tlow contains multiple blocks, each consisting of a multi-step flow where each step includes three transformation layers, i.e.,

ActNorm, Invertible Linear, and Afine Coupling layer. Figure 2 illustrates the model architecture of Tlow.

## 2.2 A Step of Flow

A step of flow consists of the following three transformation layers:

• ActNorm Layer. For the input $\mathbf { x } _ { 0 } ~ \in ~ \mathbb { R } ^ { d }$ , this layer performs activation normalization for stable training as follows,

$$
\mathbf { x } _ { 1 } = \mathbf { s } \odot ( \mathbf { x } _ { 0 } + \mathbf { t } ) ,\tag{3}
$$

where ⊙ denotes the element-wise product and s, t ∈ $\mathbb { R } ^ { d }$ are learnable scale and bias parameters, respectively. The log-determinant of this layer is:

$$
\log \vert \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 1 } / \mathrm { d } \mathbf { x } _ { 0 } ) \vert = \sum _ { j = 1 } ^ { d } \log \vert \pmb { s } _ { j } \vert .\tag{4}
$$

• Invertible Linear Layer. For the input $\mathbf { x } _ { 1 } .$ , this layer learns an invertible linear transformation to allow the dimensions of the embedding to influence one another:

$$
\mathbf { x } _ { 2 } = \mathbf { W } \mathbf { x } _ { 1 } = ( \mathbf { P } \mathbf { L } \mathbf { U } ) \mathbf { x } _ { 1 } .\tag{5}
$$

The linear matrix $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ is decomposed into the product of a permutation matrix P, a lower triangular matrix $\mathbf { L } ,$ and an upper triangular matrix U to facilitate the computation of the determinant. We keep the diagonal elements of L as ones, while making the diagonal of U a set of learnable parameters w $\epsilon \mathbb { R } ^ { d }$ In this way, the log-determinant of this layer is:

$$
\begin{array} { l } { { \log | \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 2 } / \mathrm { d } \mathbf { x } _ { 1 } ) | = \log | \operatorname* { d e t } ( \mathbf { W } ) | } } \\ { { \displaystyle \quad = \log | \operatorname* { d e t } ( \mathbf { P } ) | + \log | \operatorname* { d e t } ( \mathbf { L } ) | + \log | \operatorname* { d e t } ( \mathbf { U } ) | } } \\ { { \displaystyle \quad = \sum _ { j = 1 } ^ { d } \log | \mathbf { w } _ { j } | } . } \end{array}\tag{6}
$$

• Afine Coupling Layer. For the input $\mathbf { X } _ { 2 } ,$ , this layer first divides it into two halves $\mathbf { x } _ { 2 } ^ { a }$ and $\mathbf { x } _ { 2 } ^ { b } ,$ , and performs the transformation as follows:

$$
\begin{array} { r l } & { \mathbf { x } _ { 3 } = [ \mathbf { x } _ { 2 } ^ { a } , \mathbf { s } ^ { b } \odot ( \mathbf { x } _ { 2 } ^ { b } + \mathbf { t } ^ { b } ) ] , } \\ & { [ \mathbf { s } ^ { b } , \mathbf { t } ^ { b } ] = g ( \mathbf { x } _ { 2 } ^ { a } ) , } \end{array}\tag{7}
$$

where $g : \mathbb { R } ^ { \frac { d } { 2 } }  \mathbb { R } ^ { d }$ is a MLP to generate scale and bias parameters based on $\mathbf { x } _ { 2 } ^ { a }$ to transform $\mathbf { x } _ { 2 } ^ { \bar { b } }$ . The log-determinant of this layer is:

$$
\log \vert \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 3 } / \mathrm { d } \mathbf { x } _ { 2 } ) \vert = \sum _ { j = 1 } ^ { d / 2 } \log \vert \mathbf { s } _ { j } ^ { b } \vert .\tag{8}
$$

Combining these three layers together, the log-determinant of a step of flow is:

$$
\Delta = \log | \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 1 } / \mathrm { d } \mathbf { x } _ { 0 } ) | + \log | \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 2 } / \mathrm { d } \mathbf { x } _ { 1 } ) | + \log | \operatorname* { d e t } ( \mathrm { d } \mathbf { x } _ { 3 } / \mathrm { d } \mathbf { x } _ { 2 } ) | .\tag{9}
$$

## 2.3 Flow-based Tokenizer

Tlow contains � blocks and each block consists of � steps of flow. For the �-th block, the output $\mathbf { z } _ { n } \in \mathbb { R } ^ { \frac { d _ { s } } { 2 ^ { n - 1 } } }$ is divided into two halves $\mathbf { z } _ { n } ^ { a } \in \mathbb { R } ^ { \frac { d _ { s } } { 2 ^ { n } } }$ and $\mathbf { z } _ { n } ^ { b } \in \mathbb { R } ^ { \frac { d _ { s } } { 2 ^ { n } } }$ , which serve as the input of the next block

and partial output (i.e., partial z) of Tlow, respectively. We estimate the log-likelihood of $\mathbf { z } _ { n } ^ { b }$ as follows:

$$
\begin{array} { l } { \log p ( \mathbf { z } _ { n } ^ { b } ) = \log N ( \mathbf { z } _ { n } ^ { b } ; \pmb { \mu } _ { n } , \pmb { \sigma } _ { n } ^ { 2 } ) , } \\ { \left[ \pmb { \mu } _ { n } , \pmb { \sigma } _ { n } ^ { 2 } \right] = h ( \mathbf { z } _ { n } ^ { a } ) , } \end{array}\tag{10}
$$

where $h : \mathbb { R } ^ { \frac { d _ { s } } { 2 ^ { n } } }  \mathbb { R } ^ { \frac { d _ { s } } { 2 ^ { n - 1 } } }$ is an MLP module to predict the mean $\pmb { \mu } _ { n }$ and variance $\sigma _ { n } ^ { 2 }$ of $\mathbf { z } _ { n } ^ { b }$ based on $\mathbf { z } _ { n } ^ { a } .$ . Combining the outputs of all the blocks, we obtain final output of Tlow $\mathbf { z } = [ \mathbf { z } _ { 1 } ^ { b } , \mathbf { z } _ { 2 } ^ { b } , \cdot \cdot \cdot , \mathbf { z } _ { N } ^ { b } ] \in \mathbb { R } ^ { d _ { s } }$ Note that ${ \bf z } _ { N } ^ { b } = { \bf z } _ { N }$ since we do not need to divide the output of the last block. Moreover, the input of the first block is x. We then formulate:

$$
\log p _ { \boldsymbol { \theta } } ( \mathbf { z } ) = \sum _ { n = 1 } ^ { N } \log p ( \mathbf { z } _ { n } ^ { b } ) .\tag{11}
$$

In addition, the log-determinant of Tlow is calculated as follows:

$$
\log | \operatorname* { d e t } ( \mathrm { d } \mathbf { z } / \mathrm { d } \mathbf { x } ) | = \sum _ { n = 1 } ^ { N } \sum _ { m = 1 } ^ { M } \Delta _ { n , m } ,\tag{12}
$$

where $\Delta _ { n , m }$ denotes the log-determinant of the �-th step of flow in the �-th block. After training Tlow with the log-likelihood objective $\mathcal { L } _ { f } .$ the semantic embedding x is transformed into a latent embedding z conforming to a standard normal distribution, where the dimensions are independent of each other.

For item tokenization, we perform the PQ on z to obtain � token IDs. Specifically, z is divided into � parts $[ { \bf z } _ { 1 } , { \bf z } _ { 2 } , \cdots , { \bf z } _ { C } ]$ , each of which is independently encoded using K-means to obtain a corresponding codebook $C _ { k } = \{ \mathbf { c } _ { k } ^ { 1 } , \mathbf { c } _ { k } ^ { 2 } , \cdot \cdot \cdot , \mathbf { c } _ { k } ^ { S } \}$ . Here, $\mathbf { c } _ { k } ^ { j } \in \mathbb { R } ^ { \frac { d _ { s } } { C } }$ is the �-th quantized codeword and � denotes the codebook size. For the �-th tokenization, the token ID is defined as:

$$
c _ { k } = \underset { j \in \{ 1 , 2 , \cdots , S \} } { \arg \operatorname* { m i n } } \ \| \mathbf { z } _ { k } - \mathbf { c } _ { k } ^ { j } \| ^ { 2 } .\tag{13}
$$

In this way, Tlow tokenizes each item into discrete IDs $[ c _ { 1 } , c _ { 2 } , \cdots , c _ { C } ]$

## 2.4 Token IDs for Recommendation

Given token IDs of all the items, a sequential model $( e . g .$ , Transformer decoder) aims to generate the IDs of the next item, with a set of token embeddings learned. Specifically, there is a one-to-one correspondence between the token embeddings $\mathbf { E } \in \mathbb { R } ^ { C \times S \times d _ { m } }$ and codebooks $\mathbf { C } \in \mathbb { R } ^ { C \times S \times \frac { d _ { s } } { C } }$ , where $d _ { m }$ denotes the dimension of token embeddings. Traditional ID embedding can be replaced with the aggregated token embeddings $\mathbf { e } \in \mathbb { R } ^ { d _ { m } }$ for each item:

$$
\mathbf { e } = \frac { 1 } { C } \sum _ { k = 1 } ^ { C } \mathbf { E } _ { k , c _ { k } } .\tag{14}
$$

The sequential model further encodes the user’s historical embedding sequence $\left[ \mathbf { e } _ { i _ { 1 } } , \mathbf { e } _ { i _ { 2 } } , \cdots \right]$ and outputs a final hidden state h $\in \mathbb { R } ^ { d _ { m } }$ . Following [5], we predict the log-likelihood of interaction with the target item $i _ { t }$ with token IDs $\left[ c _ { 1 } ^ { t } , c _ { 2 } ^ { t } , \cdots , c _ { C } ^ { t } \right]$ as follows:

$$
\log p ( i _ { t } | \mathbf { h } ) = \sum _ { k = 1 } ^ { C } \log p \left( c _ { k } ^ { t } | \mathbf { h } \right) = \sum _ { k = 1 } ^ { C } \log \frac { \exp \left( \mathbf { E } _ { k , c _ { k } ^ { t } } ^ { \top } g _ { k } ( \mathbf { h } ) / \tau \right) } { \sum _ { j = 1 } ^ { S } \exp \left( \mathbf { E } _ { k , j } ^ { \top } g _ { k } ( \mathbf { h } ) / \tau \right) } ,\tag{15}
$$

where � is the temperature hyperparameter. The �-th project head $g _ { k } : \mathbb { R } ^ { d _ { m } }  \mathbb { R } ^ { d _ { m } }$ decodes the �-th token ID of the next item. This

![](images/d88c7e9fb95e6a26c15099a67105ac95f42ae0e630e1ec8d3435727fc722f366.jpg)  
Figure 2: The illustration of Tlow Architecture with $N = 3$ blocks and $M = 2$ flows, where the dimension of semantic embedding is $d _ { s } = 4$ (left). An example of “A Step of Flow” processing embedding with the dimension � = 4 (right).

decoding approach owes to the independence between diferent parts of the latent embedding z. Furthermore, the training loss for recommendation can be formulated as:

$$
\mathcal { L } _ { r e c } = - \sum _ { ( \mathbf { h } , i _ { t } ) } \log p ( i _ { t } | \mathbf { h } ) .\tag{16}
$$

## 2.5 Learning Token Embeddings with Codebook Guidance

Since Tlow tokenizes items based on latent embeddings, the quantized codebooks imply clearer semantics contained in the whole set of items. In other words, each codeword in a codebook encodes atomic information derived from the decomposition of items’ semantics, as the illustration in Figure 1. Therefore, learning token embeddings with the codebook guidance will help enhance recommendations. Unlike independent alignment for each item or behavior sequence used in existing works [14, 28], we propose to directly align two latent spaces spanned by token embeddings and codebooks. The token space Φ $\bar { \in } \mathbb { R } ^ { C \times S \times S }$ and codebook space $\Psi \in \mathbb { R } ^ { C \times S \times S }$ are defined based on cosine similarities between token embeddings and codewords as follows:

$$
\Phi _ { k , i , j } = \frac { \mathbf { E } _ { k , i } ^ { \top } \mathbf { E } _ { k , j } } { \Vert \mathbf { E } _ { k , i } \Vert \cdot \Vert \mathbf { E } _ { k , j } \Vert } , \Psi _ { k , i , j } = \frac { \mathbf { C } _ { k , i } ^ { \top } \mathbf { C } _ { k , j } } { \Vert \mathbf { C } _ { k , i } \Vert \cdot \Vert \mathbf { C } _ { k , j } \Vert } .\tag{17}
$$

Furthermore, we implement the codebook guidance as a simple but efective MSE loss:

$$
\mathcal { L } _ { s i m } = \mathrm { M S E } \left( \left| \Phi - \Psi \right| \right) .\tag{18}
$$

Finally, the sequential model and token embeddings are learned by training the combined loss $\mathcal { L } = \mathcal { L } _ { r e c } + \lambda \mathcal { L } _ { s i m } .$ , where � is the degree of codebook guidance.

## 3 Experiments

We conduct experiments under three scenarios to evaluate the effectiveness of our proposed Tlow, including general, cross-domain, and multi-modal sequential recommendation.

## 3.1 Experimental Setup

3.1.1 Datasets. We use four commodity categories in Amazon Reviews dataset [17] for experiments, including “Sports and Outdoors (Sports), “Beauty”, “Toys and Games (Toys)”, and “CDs and Vinyl (CDs)”. Following existing works [4, 5, 22], users’ reviews are regarded as item interactions and arranged chronologically as historical item sequences. We also use a processed dataset “Cloth-Sports” from a state-of-the-art (SOTA) work [15] for cross-domain sequential recommendation. This dataset is collected from “Clothing Shoes and Jewelry” and “Sports and Outdoors” categories in Amazon dataset. Most of the users in this dataset overlap across two categories. The statistics of used datasets are shown in Table 1, where Avg. Length means the length of historical sequence averaged over all the users.

Table 1: Statistics of used datasets.
<table><tr><td>Datasets</td><td>#Users</td><td>#Items</td><td>#Interactions</td><td>Avg. Length</td></tr><tr><td>Sports</td><td>35,598</td><td>18,357</td><td>260,739</td><td>8.32</td></tr><tr><td>Beauty</td><td>22,363</td><td>12,101</td><td>176,139</td><td>8.87</td></tr><tr><td>Toys</td><td>19,412</td><td>11,924</td><td>148,185</td><td>8.63</td></tr><tr><td>CDs</td><td>75,258</td><td>64,443</td><td>1,022,334</td><td>14.58</td></tr><tr><td>Cloth</td><td>9,933</td><td>3,278</td><td>97,741</td><td>10.71</td></tr><tr><td>Sports</td><td>4,263</td><td>1,021</td><td>11,879</td><td></td></tr></table>

The last item in a historical sequence is used for testing and the second-to-last item for validation, which is the widely adopted leave-one-out evaluation protocol in existing works [5, 8, 38].

3.1.2 Baselines. We compare Tlow with two types of baselines following existing works [5, 14, 22]. Traditional ID-based models include Caser [26], GRU4Rec [3], HGN [16], BERT4Rec [25], SAS-Rec [8], FDSA [37], and S<sup>3</sup>-Rec [41], which adopt diferent neural architectures (e.g., CNNs, RNNs, and Transformers) to model user behavior sequences with randomly assigned item IDs. Tokenizationbased models include VQRec [4], TIGER [22], ETEGRec [14], RecJPQ HSTU [34], and RPG [5], which leverage various quantization techniques (e.g., PQ, RQ-VAE, and OPQ) to tokenize item semantic embeddings for generative recommendation.

3.1.3 Evaluation Metrics. We use widely adopted ranking metrics Recall@� (R@�) and NDCG@� (N@�) for performance evaluation, where � is set as 5 and 10 following [5, 22]. All the items are included for full ranking to avoid sampling bias when calculating evaluation metrics [11].

3.1.4 Implementation Details. We implement our Tlow with Pytorch. All the experimental setups for training and evaluation follow RPG [5], thus we use baseline results from the original paper and rerun RPG. For a fair comparison, all the models use the extracted item semantic embeddings by the text encoder sentence-t5-base [18], leading to $d _ { s } = 7 6 8$ . We set the number of blocks and the steps of flow in Tlow as $N = 4$ and $M = 4 .$ In fact, the recommendation performance is not sensitive to the number ofblocks and flows, because Tlow can easily transform the semantic embeddings into latent embeddings conforming to a standard normal distribution. The number of codebooks $C \in \{ 1 6 , 3 2 , 6 4 , 9 6 , 1 2 8 \}$ and the codebook size � is fixed as 256 following RPG. The temperature $\tau \in \{ 0 . 0 3 , 0 . 0 5 , 0 . 0 7 \}$ when decoding token IDs. For the hyperparameters of the sequential model, we keep them consistent with those of RPG to ensure a similar number of model parameters: a 2-layer GPT-2 with the token embedding dimension of $d _ { m } = 4 4 8 .$ Since Tlow decodes all � token IDs of the next item independently within a single forward pass (Eq. 15), in the same manner as RPG, it retains the same decoding parallelism as RPG and thus the same decoding-eficiency advantage over RQ-VAE-based sequential decoding (e.g., TIGER) that has already been empirically validated in [5]; we therefore do not repeat separate latency/throughput benchmarking in this work.

## 3.2 Overall Performance

The overall performance comparison is shown in Table 2. It is obvious that tokenization-based models using item semantics significantly outperform purely ID-based models. These results prove the efectiveness of item tokenization for more accurate modeling of user behaviors. Moreover, Tlow performs best on all the metrics across four datasets, demonstrating better tokenization in the transformed latent space through a flow-based model. It is worth noting that when using the same semantic embeddings generated by sentence-t5-base, RPG with its independent tokenizer does not outperform TIGER consistently. This also demonstrates the cru cial role of Tlow’s tokenization on latent embeddings conforming to a standard normal distribution.

## 3.3 Ablation Study

We conduct further experiments to validate the efectiveness of two critical modules in Tlow: the transformed latent embeddings z and the codebook guidance.

• Random z. Latent embeddings z are sampled directly from a standard normal distribution instead of being transformed from semantic embeddings by Tlow.

• w/o $\mathcal { L } _ { s i m } .$ The codebook guidance is removed by setting its corresponding degree $\lambda = 0$

The results in Table 3 demonstrate the efectiveness of codebook guidance for enhanced recommendation through learning a semantically clearer latent space of token embeddings. Furthermore, the tokenization based on randomly sampled z leads to a significant performance drop. This confirms that Tlow retains rich semantic information necessary for behaviors modeling when transforming the embedding x.

## 3.4 Cross-domain and Multi-modal Recommendation

In this section, we further validate the necessity and efectiveness of transforming semantic embeddings from diferent distributional spaces into a unified space for tokenization.

3.4.1 Cross-domain Recommendation. We choose the current SOTA cross-domain model LLM4CDSR [15] as our baseline, which leverages LLMs to bridge the domain gap with item semantic embeddings and hierarchical user profiling. Besides, RPG is also included to demonstrate the efectiveness of Tlow when tokenizing items from diferent domains. For the implementation of RPG and Tlow, we directly use the mixed item sequences merged from two domains for model training.

The results in Table 4 demonstrates the significant improvement of Tlow over both LLM4CDSR and RPG. It is efective for two reasons: first, tokenizing items to create a shared vocabulary of token IDs and training their embeddings naturally bridges the cross-domain gap. Second, Tlow’s ability to transform embedding distributions is highly efective for improving recommendations across diferent domains.

3.4.2 Multi-modal Recommendation. We chose the current SOTA multi-modal model HM4SR [36] as our baseline, which leverages interactive and temporal mixture of experts to capture user dynamic interests. Since HM4SR utilizes a large amount of additional side-information (including interaction timestamp, item category, semantic embeddings as additional input), we replace the ID embeddings in HM4SR with the token embeddings e generated by Tlow and RPG to directly compare the efectiveness of tokenization.

We conduct experiments on the “Sports” dataset and generate additional image embeddings using the advanced CLIP model OpenCLIP ViT-H/14 [21], where the image is downloaded from product URLs of each item. The tokenization on text and image embeddings are performed independently, and both token IDs are merged as final token IDs. The results in Table 5 show that RPG’s tokenization brings limited and inconsistent gains over HM4SR in this setting with combined text-image distributions (e.g., it ties HM4SR on R@5 and underperforms on N@5 and N@10). In contrast, Tlow consistently improves over both baselines on all four metrics after performing the distribution transformation, although the absolute margin over HM4SR (e.g., R@10 from 0.0469 to 0.0521) is moderate in this single-domain multi-modal setting; we expect larger gains with richer or more heterogeneous modalities.

## 3.5 Cold-Start Recommendation

Focusing on item semantics, tokenization-based models inherently have an advantage in cold-start recommendation. Therefore, we further investigate whether Tlow’s tokenization can enhance recommendation performance in cold-start scenarios. We conduct our validation from both user and item perspectives. Specifically, users are grouped by their number of historical interactions (interaction depth), while items in the test set are grouped by their frequency in the training set (popularity). We then evaluate the recommendation performance for these diferent groups on the largest “CDs” dataset.

Figure 3 shows the performance comparison between Tlow and RPG, where Tlow performs significantly better across all the groups. It indicates that due to Tlow’s ability to transform any embeddings (including long-tail ones) into a standard normal distribution, its tokenization on items across all popularity levels is more comprehensive. This further leads to more accurate behavioral modeling for users with varying interaction depths.

Table 2: Performance comparison on general recommendation. Underline denotes the best baseline and bold denotes better performance than the best baseline at a significance level of $\dot { p } < 0 . 0 1$ under paired t-test.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="4">Sports and Outdoors</td><td colspan="4">Beauty</td><td colspan="4">Toys and Games</td><td colspan="4">CDs and Vinyl</td></tr><tr><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td></tr><tr><td></td><td>Caser</td><td>0.0116</td><td>0.0072</td><td>0.0194</td><td>0.0097</td><td>0.0205</td><td>0.0131</td><td>0.0347</td><td>0.0176</td><td>0.0166</td><td>0.0107</td><td>0.0270</td><td>0.0141</td><td>0.0116</td><td>0.0073</td><td>0.0205</td><td>0.0101</td></tr><tr><td></td><td>GRU4Rec</td><td>0.0129</td><td>0.0086</td><td>0.0204</td><td>0.0110</td><td>0.0164</td><td>0.0099</td><td>0.0283</td><td>0.0137</td><td>0.0097</td><td>0.0059</td><td>0.0176</td><td>0.0084</td><td>0.0195</td><td>0.0120</td><td>0.0353</td><td>0.0171</td></tr><tr><td></td><td>HGN</td><td>0.0189</td><td>0.0120</td><td>0.0313</td><td>0.0159</td><td>0.0325</td><td>0.0206</td><td>0.0512</td><td>0.0266</td><td>0.0321</td><td>0.0221</td><td>0.0497</td><td>0.0277</td><td>0.0259</td><td>0.0153</td><td>0.0467</td><td>0.0220</td></tr><tr><td>m</td><td>BERT4Rec</td><td>0.0115</td><td>0.0075</td><td>0.0191</td><td>0.0099</td><td>0.0203</td><td>0.0124</td><td>0.0347</td><td>0.0170</td><td>0.0116</td><td>0.0071</td><td>0.0203</td><td>0.0099</td><td>0.0326</td><td>0.0201</td><td>0.0547</td><td>0.0271</td></tr><tr><td></td><td>SASRec</td><td>0.0233</td><td>0.0154</td><td>0.0350</td><td>0.0192</td><td>0.0387</td><td>0.0249</td><td>0.0605</td><td>0.0318</td><td>0.0463</td><td>0.0306</td><td>0.0675</td><td>0.0374</td><td>0.0351</td><td>0.0177</td><td>0.0619</td><td>0.0263</td></tr><tr><td>FDSA S3-Rec</td><td></td><td>0.0182</td><td>0.0122</td><td>0.0288</td><td>0.0156</td><td>0.0267</td><td>0.0163</td><td>0.0407</td><td>0.0208</td><td>0.0228</td><td>0.0140</td><td>0.0381</td><td>0.0189</td><td>0.0226</td><td>0.0137</td><td>0.0378</td><td>0.0186</td></tr><tr><td></td><td></td><td>0.0251</td><td>0.0161</td><td>0.0385</td><td>0.0204</td><td>0.0387</td><td>0.0244</td><td>0.0647</td><td>0.0327</td><td>0.0443</td><td>0.0294</td><td>0.0700</td><td>0.0376</td><td>0.0213</td><td>0.0130</td><td>0.0375</td><td>0.0182</td></tr><tr><td></td><td>RecJPQ</td><td>0.0141</td><td>0.0076</td><td>0.0220</td><td>0.0102</td><td>0.0311</td><td>0.0167</td><td>0.0482</td><td>0.0222</td><td>0.0331</td><td>0.0182</td><td>0.0484</td><td>0.0231</td><td>0.0075</td><td>0.0046</td><td>0.0138</td><td>0.0066</td></tr><tr><td>Toztion</td><td>VQ-Rec</td><td>0.0208</td><td>0.0144</td><td>0.0300</td><td>0.0173</td><td>0.0457</td><td>0.0317</td><td>0.0664</td><td>0.0383</td><td>0.0497</td><td>0.0346</td><td>0.0737</td><td>0.0423</td><td>0.0352</td><td>0.0238</td><td>0.0520</td><td>0.0292</td></tr><tr><td></td><td>TIGER</td><td>0.0264</td><td>0.0181</td><td>0.0400</td><td>0.0225</td><td>0.0454</td><td>0.0321</td><td>0.0648</td><td>0.0384</td><td>0.0521</td><td>0.0371</td><td>0.0712</td><td>0.0432</td><td>0.0492</td><td>0.0329</td><td>0.0748</td><td>0.0411</td></tr><tr><td>ETEGRec</td><td></td><td>0.0175</td><td>0.0114</td><td>0.0281</td><td>0.0149</td><td>0.0404</td><td>0.0277</td><td>0.0587</td><td>0.0337</td><td>0.0209</td><td>0.0136</td><td>0.0339</td><td>0.0178</td><td>0.0309</td><td>0.0204</td><td>0.0461</td><td>0.0253</td></tr><tr><td>HSTU</td><td></td><td>0.0258</td><td>0.0165</td><td>0.0414</td><td>0.0215</td><td>0.0469</td><td>0.0314</td><td>0.0704</td><td>0.0389</td><td>0.0433</td><td>0.0281</td><td>0.0669</td><td>0.0357</td><td>0.0417</td><td>0.0275</td><td>0.0638</td><td>0.0346</td></tr><tr><td></td><td>RPG</td><td>0.0296</td><td>0.0203</td><td>0.0428</td><td>0.0246</td><td>0.0533</td><td>0.0366</td><td>0.0753</td><td>0.0437</td><td>0.0509</td><td>0.0357</td><td>0.0765</td><td>0.0440</td><td>0.0486</td><td>0.0328</td><td>0.0693</td><td>0.0395</td></tr><tr><td></td><td>Tlow</td><td>0.0307</td><td>0.0207</td><td>0.0477</td><td>0.0261</td><td>0.0545</td><td>0.0377</td><td>0.0786</td><td>0.0454</td><td>0.0590</td><td>0.0395</td><td>0.0864</td><td>0.0482</td><td>0.0541</td><td>0.0362</td><td>0.0801</td><td>0.0446</td></tr><tr><td></td><td>Impr. (%)</td><td>3.72</td><td>1.97</td><td>11.45</td><td>6.10</td><td>2.25</td><td>3.01</td><td>4.38</td><td>3.89</td><td>11.52</td><td>9.70</td><td>9.80</td><td>11.36</td><td>9.96</td><td>10.03</td><td>7.09</td><td>8.52</td></tr></table>

Table 3: Ablation study of Tlow.
<table><tr><td rowspan="2">Model</td><td colspan="2">Sports</td><td colspan="2">Beauty</td><td colspan="2">Toys</td><td colspan="2">CDs</td></tr><tr><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>Tlow</td><td>0.0477</td><td>0.0261</td><td>0.0786</td><td>0.0454</td><td>0.0864</td><td>0.0482</td><td>0.0801</td><td>0.0446</td></tr><tr><td>w/o  $\mathcal { L } _ { s i m }$ </td><td>0.0449</td><td>0.0248</td><td>0.0764</td><td>0.0436</td><td>0.0826</td><td>0.0471</td><td>0.0756</td><td>0.0425</td></tr><tr><td>Random z</td><td>0.0202</td><td>0.0103</td><td>0.0623</td><td>0.0363</td><td>0.0571</td><td>0.0341</td><td>0.0147</td><td>0.0074</td></tr></table>

Table 4: Performance comparison on cross-domain recommendation.
<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">Cloth</td><td colspan="2">Sports</td></tr><tr><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>LLM4CDSR</td><td>0.4620</td><td>0.2803</td><td>0.4220</td><td>0.2637</td><td>0.5507</td><td>0.3172</td></tr><tr><td>RPG</td><td>0.4766</td><td>0.3457</td><td>0.4487</td><td>0.3353</td><td>0.5385</td><td>0.3686</td></tr><tr><td>Tlow</td><td>0.5558</td><td>0.4395</td><td>0.5669</td><td>0.4568</td><td>0.5312</td><td>0.4014</td></tr></table>

Table 5: Performance comparison on multi-modal recommendation.
<table><tr><td>Model</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td></tr><tr><td>HM4SR</td><td>0.0326</td><td>0.0231</td><td>0.0469</td><td>0.0277</td></tr><tr><td>RPG</td><td>0.0326</td><td>0.0216</td><td>0.0501</td><td>0.0272</td></tr><tr><td>Tlow</td><td>0.0343</td><td>0.0233</td><td>0.0521</td><td>0.0290</td></tr></table>

## 3.6 Online Experiments

3.6.1 Online Setup. We verify Tlow’s online performance on China’s largest social media platform WeChat, in the picture recommendation scenario involving image-text multi-modal features, comparing sequential models that use Tlow’s tokenized embeddings e against those using randomly assigned item-ID embeddings. We construct users’ behavior sequences of length 500 with their recently clicked items, and feed the sequence of item embeddings into a 12-layer Transformer decoder to predict the user’s interaction tendency toward the next item. Both sequential models are integrated as retrieval pathways in addition to the primary DSSM-based realtime retrieval method, and the online serving scheme is shown in Figure 4. During the online serving phase, user embeddings and item embeddings are stored in dedicated embedding servers. Upon user access, the corresponding user embedding is retrieved via a real-time lookup operation, and candidate pictures with the highest similarity scores are then dynamically identified by the Similarity Server [1] to construct the retrieval results. We conduct both single-domain and cross-domain online experiments:

![](images/c2ead3662395048825238222a360011d75525b101677a3c8d081b004d0a5579d.jpg)

![](images/6b8d60886805abf0a511cff6c8d78897c814d459179ce99fc84e3b6bde0eb694.jpg)  
Figure 3: Performance comparison under diferent user and item groups.

• Single-domain: Only users’ clicked pictures are used to con struct behavior sequences for model training.

• Cross-domain: Both users’ clicked pictures and articles are mixed to construct behavior sequences for model training. This is to verify the efectiveness of Tlow’s tokenization in cross-domain scenarios.

Table 6: Online metrics improvement (%) of pathway comparison in single-domain and cross-domain scenarios.
<table><tr><td rowspan="2">Scenario</td><td colspan="2">Overall</td><td colspan="2">New Item</td></tr><tr><td>CTR</td><td>UCTR</td><td>CTR</td><td>UCTR</td></tr><tr><td>Single-domain</td><td>4.79</td><td>10.32</td><td>8.46</td><td>11.64</td></tr><tr><td>Cross-domain</td><td>6.23</td><td>7.20</td><td>9.09</td><td>9.45</td></tr></table>

3.6.2 Training and Inference. In the stage of training Tlow, we find that the model converges after training on embeddings of approximately 1 million items (pictures and articles). Then Tlow can generate token IDs for other unseen items by performing semantic tokenization. As a result, the training time of Tlow is negligible, and the inference eficiency is such that tokenization for tens of millions of newly published items per day can be completed in just a few minutes on a single GPU. In the deployment, we set the number of codebooks � = 16 and codebook size � = 256. Therefore, the number of token embeddings that the Tlow-based sequential model needs to learn is only $C \times S = 4 0 9 6 .$ , far fewer than the tens of millions of item embeddings in the baseline model. The Tlow-based sequential model is initialized with all parameters of the baseline model except the item-ID embeddings. Before being deployed for online A/B test, both models are trained for two weeks on identical datasets with more than 70 and 200 million interaction records per day in single-domain and cross-domain scenarios, respectively.

3.6.3 Online Performance. After deploying the aforementioned Tlow-based and ID-based models as supplemental retrieval pathways, we evaluate the conversion eficacy of both pathways across the entire picture corpus and specifically on cold-start pictures exposed on the same day as publication. The following metrics are reported:

• CTR (Click-Through Rate) measures the click-through rate of items exposed via this specific retrieval pathway.

• UCTR (User Click-Through Rate) averages all the users’ click through rate of items exposed via this pathway.

Results are shown in Table 6. Due to privacy considerations, we report only the relative diference for all metric comparisons between the models. As we can see, the sequential model employing Tlow’s tokenization demonstrates 4.79% and 6.23% higher CTR compared to the one using random IDs in single-domain and crossdomain scenarios, respectively. Furthermore, we observe a more substantial improvement of 10.32% and 7.20% in UCTR, indicating its efectiveness in retrieving potentially relevant content to a broader user base. These findings validate Tlow’s powerful capabilities on multi-modal and cross-domain embedding transformation in large-scale industrial recommender systems. To further validate Tlow’s efectiveness on cold-start recommendations, we perform evaluations exclusively on newly published pictures each day. The results reveal that the advantages of Tlow become markedly more pronounced when evaluated solely on these cold-start pictures, since random ID embeddings require extensive user interaction data for training, whereas Tlow utilizes inherent semantic features available immediately upon a picture’s publication.

![](images/84c30ee8d0516c7a9a76922876ba2465dfa929fd80d0674d4788347bf8840bfd.jpg)  
Figure 4: Online serving of Tlow’s tokenization.

Beyond pathway comparisons, core online metrics from A/B testing also improve significantly: in the single-domain scenario, 24-hour overall and newly-published picture CTR increase by 0.78% and 1.94%, respectively; in the cross-domain scenario, per-capita picture CTR improves by 1.05%, accompanied by a 1.15% decrease in top-tier accounts’ exposure share, indicating that Tlow substantially benefits long-tail picture exposure.

## 4 Related Work

## 4.1 ID-based Recommendation

ID-based recommendation originates from collaborative filtering, evolving from matrix factorization [10] to GNN-based models [29]. For sequential recommendation, various architectures have been explored, including Markov Chains [23], RNNs and CNNs [3, 26, 31, 32], and self-attention mechanisms [8, 36, 40]. Multi-modal information such as text and images has also been incorporated to enrich item representations [27, 36].

## 4.2 Tokenization-based Recommendation

TIGER [22] pioneers the use of RQ-VAE to tokenize item semantic embeddings, followed by works that introduce alignment strategies to enhance tokenization quality [14, 28, 30]. However, the hierarchical nature of RQ-VAE leads to low decoding eficiency due to codebook correlations. To enable parallel decoding, PQ-based independent tokenization methods have been proposed [4, 5], while parameter-free tokenizers based on heuristic algorithms remain limited in performance [6, 19, 24]. Despite this progress, existing independent tokenizers still face challenges from non-independent embedding dimensions and complex embedding distributions, which motivates our flow-based approach.

## 5 Conclusion

We propose Tlow, a flow-based item tokenizer that transforms semantic embeddings into a standard normal distribution space, achieving dimensional independence and distributional simplicity for more accurate independent tokenization. Tlow significantly improves recommendation performance across general, cross-domain, multi-modal, and cold-start scenarios in both ofline and online experiments.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China under U24B20180.

## References

[1] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jef Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The Faiss library. (2024). arXiv:2401.08281 [cs.LG]

[2] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence 36, 4 (2013), 744–755.

[3] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[4] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings ofthe ACM Web Conference 2023. 1162–1171.

[5] Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, and Julian McAuley. 2025. Generating Long Semantic IDs in Parallel for Recommendation. arXiv preprint arXiv:2506.05781 (2025).

[6] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to index item ids for recommendation foundation models. In Proceedings ofthe Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region. 195–204.

[7] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117–128.

[8] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

[9] Durk P Kingma and Prafulla Dhariwal. 2018. Glow: Generative flow with in vertible 1x1 convolutions. Advances in neural information processing systems 31 (2018).

[10] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30–37.

[11] Walid Krichene and Stefen Rendle. 2020. On sampled metrics for item recommendation. In Proceedings ofthe 26th ACM SIGKDD international conference on knowledge discovery & data mining. 1748–1757.

[12] Zhirui Kuai, Zuxu Chen, Huimu Wang, Mingming Li, Dadong Miao, Binbin Wang, Xusong Chen, Li Kuang, Yuxing Han, Jiaxing Wang, et al. 2024. Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval. arXiv preprint arXiv:2407.21488 (2024).

[13] Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. 2020. On the sentence embeddings from pre-trained language models. arXiv preprint arXiv:2011.05864 (2020).

[14] Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, and Wayne Xin Zhao. 2024. Generative Recommender with End-to-End Learnable Item Tokenization. arXiv preprint arXiv:2409.05546 (2024).

[15] Qidong Liu, Xiangyu Zhao, Yejing Wang, Zijian Zhang, Howard Zhong, Chong Chen, Xiang Li, Wei Huang, and Feng Tian. 2025. Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation. arXiv preprint arXiv:2504.18383 (2025).

[16] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In Proceedings ofthe 25th ACM SIGKDD international conference on knowledge discovery & data mining. 825–833.

[17] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings ofthe 38th international ACM SIGIR conference on research and development in information retrieval. 43–52.

[18] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang. 2021. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877 (2021).

[19] Aleksandr V Petrov and Craig Macdonald. 2023. Generative sequential recommendation with gptrec. arXiv preprint arXiv:2306.11114 (2023).

[20] Aleksandr V Petrov and Craig Macdonald. 2024. RecJPQ: training large-catalogue sequential recommenders. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 538–547.

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

[22] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al.

2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[23] Stefen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web. 811–820.

[24] Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, and Kun Gai. 2024. Generative retrieval with semantic tree-structured identifiers and contrastive learning. In Proceedings ofthe 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region. 154–163.

[25] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

[26] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings ofthe eleventh ACM international conference on web search and data mining. 565–573.

[27] Zhulin Tao, Yinwei Wei, Xiang Wang, Xiangnan He, Xianglin Huang, and Tat Seng Chua. 2020. Mgat: Multimodal graph attention network for recommendation. Information Processing & Management 57, 5 (2020), 102277.

[28] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024. Learnable item tokenization for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2400–2409.

[29] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 165–174.

[30] Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024. Eager: Two-stream generative recommender with behavior-semantic collaboration. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3245–3254.

[31] Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Jiajie Xu, Victor S Sheng S. Sheng, Zhiming Cui, Xiaofang Zhou, and Hui Xiong. 2019. Recurrent convolutional neural network for sequential recommendation. In The world wide web conference. 3398–3404.

[32] An Yan, Shuo Cheng, Wang-Cheng Kang, Mengting Wan, and Julian McAuley. 2019. CosRec: 2D convolutional neural networks for sequential recommenda tion. In Proceedings ofthe 28th ACM international conference on information and knowledge management. 2173–2176.

[33] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing 30 (2021), 495–507.

[34] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024).

[35] Jin Zhang, Qi Liu, Defu Lian, Zheng Liu, Le Wu, and Enhong Chen. 2022. Anisotropic additive quantization for fast inner product search. In Proceedings of the AAAI conference on Artificial Intelligence, Vol. 36. 4354–4362.

[36] Shengzhe Zhang, Liyi Chen, Dazhong Shen, Chao Wang, and Hui Xiong. 2025. Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation. In Proceedings ofthe ACM on Web Conference 2025. 3672–3682.

[37] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level deeper self attention network for sequential recommendation.. In IJCAI. 4320–4326.

[38] Wayne Xin Zhao, Zihan Lin, Zhichao Feng, Pengfei Wang, and Ji-Rong Wen. 2022. A revisiting study of appropriate ofline evaluation for top-N recommendation algorithms. ACM Transactions on Information Systems 41, 2 (2022), 1–41.

[39] Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qian qian Wang, Qigen Hu, Rui Huang, Shiyao Wang, et al. 2025. OneRec Technical Report. arXiv preprint arXiv:2506.13695 (2025).

[40] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1059–1068.

[41] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM international conference on information & knowledge management. 1893–1902.