# Molecular Prior Networks

Pawel Gomoluch<sup>1</sup>, Anthony Bradley<sup>1,2</sup>

<sup>1</sup>DaltonTx <sup>2</sup>Department of Chemistry, University of Liverpool pgomoluch@daltontx.com, abradley@daltontx.com

## Abstract

Predicting affinity between small molecules and their protein targets remains one of the central problems in drug discovery. Due to the substantial cost of wet-lab experiments, machine learning models of drug-target affinity are usually trained on relatively small datasets, ranging from tens to hundreds of molecules. Given the inherent difficulty of generalization between different targets and chemical series, existing datasets of protein-ligand affinities are typically not used for new projects. At the same time, tabular foundation models, including Prior-data Fitted Networks, have recently shown how transformers can be used to solve machine learning problems in a single forward pass of the network, after being trained over a large corpus of, typically, synthetic datasets. Inspired by that line of work, we propose a novel affinity modelling approach, in which the labels of the query molecules are predicted from the labelled support molecules in a single forward pass of a transformer model. The model is trained on thousands of existing affinity datasets, enabling it to capture the general patterns of molecular activity across different targets, chemical series and dataset sizes. For the smallest support sets, our model matches the state of the art on the regression variant of the FS-Mol benchmark, while being faster than existing methods, thanks to the lack of taskspecific training. A comparison against a reduced variant of our architecture, which amounts to kernel regression in a meta-learned feature space, shows that the predictive performance of the transformer model is only partially explained by simple molecular similarity.

## 1 Introduction

Recent developments in tabular foundation models have demonstrated a remarkable ability of deep neural networks to generalize between tabular datasets and perform inference on unseen datasets in a single forward pass. Models such as TabPFN [Hollmann et al., 2023, Grinsztajn et al., 2025] and TabICL [Qu et al., 2025], trained over thousands of synthetic datasets, have achieved state-of-theart performance on real-world tabular machine learning problems. In this work, we propose using a similar transformer-based approach to build drug-target affinity models. Instead of synthetically generated datasets, we use the wealth of existing affinity datasets to imbue our model with the relevant priors. Over the course of training, the datasets are randomly split into support and query sets and the model is tasked with predicting the labels of the query molecules, using the information contained in the support molecules and their labels. At inference, the model is presented with an unseen dataset including both labelled and unlabelled examples, and predicts the missing labels in a single forward pass, without the need for any task-specific training.

Since Prior-data Fitted Networks have been hypothesized to approximate non-parametric inference [Muller et al., 2022], we also introduce a reduced variant of our approach, in which similarity be-¨ tween projected molecule representations is used directly to compute a weighted mean of the support labels. While this model outperforms common baselines, it remains behind the full transformer model.

The source code of our models and experiments and the trained model weights are publicly available from https://github.com/daltontx/molpriornet.

## 2 Related work

By using a large collection of molecular property datasets to improve predictive performance on unseen datasets, our work naturally falls within the field of molecular meta-learning. In our experiments, we show that our model matches state-of-the-art predictive performance on the 16-shot regression tasks from the FS-Mol benchmark [Stanley et al., 2021], while saving computation time due to the lack of task-specific training.

The general idea of using attention between labelled and unlabelled molecules is not entirely new. Restricted forms of attention have previously been explored as part of a number of more complex architectures. For instance, the sum of support labels weighted by the similarity of learned embeddings is the output mechanism for the iterative refinement LSTMs in Altae-Tran et al. [2017]. An attention layer with values representing the support labels is also a key component of the Attentive Neural Process [Kim et al., 2019], applied to molecular property prediction by Lee et al. [2022].

A single layer of attention between query and support molecules forms the cross-attention module of MHNfs [Schimunek et al., 2023, 2025]. It is then followed by a similarity module, which uses molecule similarity to compute a weighted mean of the support labels for each query molecule, in a manner similar to our kernel baseline. Together with the context module, in which a Modern Hopfield Network is used to retrieve related molecules from a subsample of the FS-Mol dataset, this architecture leads to very strong performance in the 16-shot classification tasks of the FS-Mol benchmark. Unlike MHNfs, our main model processes features and labels jointly at every layer and does not use the similarities to directly average support labels. This results in a conceptually simpler architecture, while also allowing for stacking multiple transformer layers, which we find essential for predictive performance. Furthermore, our model does not rely on external context beyond the information stored in the weights of the network. This is especially important for inference on larger datasets, as the cost of attending to a large context scales linearly with the sizes of the support and query sets. Our work is also generally geared towards regression problems, since we are primarily interested in accurate ranking of candidate molecules.

In bridging parametric and non-parametric modelling, our work is also related to deep-kernel Gaussian processes, and in particular to ADKF-IFT [Chen et al., 2023], the state of the art in few-shot molecular regression. ADKF-IFT combines a meta-trained graph neural network (GNN) encoder with a Gaussian process fit to individual tasks and generally outperforms other approaches to dataefficient adaptation of GNNs. The optimization scheme based on the Implicit Function Theorem enables efficient meta-learning despite the nested optimization setting.

## 3 Approach

## 3.1 Molecular Prior Network

The Molecular Prior Network (MPN) is a transformer that takes as input a set of molecules, partitioned into a labelled support set $\boldsymbol { S } = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } }$ and an unlabelled query set $\mathcal { Q } = \{ x _ { j } ^ { q } \} _ { j = 1 } ^ { N _ { q } }$ and jointly predicts the labels $\hat { y } _ { j } ^ { q }$ of all query molecules in a single forward pass. Each molecule is represented by a fixed-length binary fingerprint $x \in \{ 0 , 1 \} ^ { F }$ ; labels $y \in \mathbb { R }$ are real-valued regression targets. Unlike TabPFN, where individual features are tokenized, we treat each molecule as a single token and keep the feature ordering fixed across datasets. This lets the model operate under fixed semantics of individual fingerprint bits and enables learning molecular patterns that generalize across targets.

Following the original TabPFN architecture, attention is asymmetric and only the support molecules are attended $^ { \mathrm { t o , } }$ so that evidence flows only from the labelled molecules. Otherwise, the support and query molecules are processed by the same stack of layers. The predictions $\hat { y } _ { i } ^ { q }$ are read off the hidden states of the query molecules at the final layer by a single linear head. While the support hidden states are not used for the output, they are repeatedly attended to and updated, allowing the network to construct task-specific support representations conditioned on the labels.

Input embedding. Molecules are embedded into a d-dimensional token space by a shared linear map, while support labels are normalized by subtracting the task support mean and dividing by the standard deviation, and embedded by a separate linear map into a $d _ { y }$ -dimensional space (we use $d _ { y } = d / 4 )$

$$
h _ { i } ^ { ( 0 ) } = W _ { h } x _ { i } + b _ { h } , \quad i \in \{ 1 , \ldots , N _ { s } + N _ { q } \} , \qquad e _ { i } ^ { y } = W _ { y } y _ { i } ^ { s } , \quad i \in \{ 1 , \ldots , N _ { s } \} ,\tag{1}
$$

where the first $N _ { s }$ tokens correspond to support molecules and the remaining $N _ { q }$ to queries. No positional encoding is added, so the model predictions are invariant to permutations of the support set.

Transformer block. The stack consists of $L$ residual blocks. Within block ℓ we first form value tokens by concatenating the current support hidden states with the label embeddings,

$$
\begin{array} { r } { v _ { i } ^ { ( \ell ) } = \big [ h _ { i } ^ { ( \ell ) } ; e _ { i } ^ { y } \big ] \in \mathbb { R } ^ { d + d _ { y } } , \qquad i \in \{ 1 , \dots , N _ { s } \} . } \end{array}\tag{2}
$$

Attention is then applied with pre-normalization, $\tilde { h } ^ { ( \ell ) } = \mathrm { L a y e r N o r m } ( h ^ { ( \ell ) } )$ , and a learned temperature τ:

$$
h ^ { ( \ell + \frac { 1 } { 2 } ) } = h ^ { ( \ell ) } + \mathrm { A t t n } \Big ( Q = \tilde { h } ^ { ( \ell ) } / \tau , { \cal K } = \tilde { h } _ { 1 : N _ { s } } ^ { ( \ell ) } , { \cal V } = v ^ { ( \ell ) } \Big ) ,\tag{3}
$$

where Attn is single-head scaled dot-product attention with learned projections $W _ { Q } , W _ { K } , W _ { V }$ and output projection $\bar { W } _ { O }$

$$
\mathrm { A t t n } ( Q , K , V ) _ { i } = W _ { O } \sum _ { j = 1 } ^ { N _ { s } } \alpha _ { i j } W _ { V } v _ { j } , \qquad \alpha _ { i j } = { \frac { \exp \bigl ( \langle W _ { Q } q _ { i } , W _ { K } k _ { j } \rangle / \sqrt { d } \bigr ) } { \sum _ { j ^ { \prime } = 1 } ^ { N _ { s } } \exp \bigl ( \langle W _ { Q } q _ { i } , W _ { K } k _ { j ^ { \prime } } \rangle / \sqrt { d } \bigr ) } } .\tag{4}
$$

Note that labels enter the computation exclusively through the values of attention, via the concatenation in Eq. (2). The original label embeddings are re-injected at every layer, which we have found to improve the model’s performance.

In line with modern transformer architectures [Karpathy, 2025], the attention is followed by a small feed-forward network with squared ReLU activation and a residual connection,

$$
h ^ { ( \ell + 1 ) } = h ^ { ( \ell + \frac { 1 } { 2 } ) } + W _ { 2 } \Big [ \mathrm { R e L U } \big ( W _ { 1 } \mathrm { L a y e r N o r m } ( h ^ { ( \ell + \frac { 1 } { 2 } ) } ) \big ) \Big ] ^ { 2 } ,\tag{5}
$$

where the inner dimension of the FFN is 4d.

Output head. After L blocks, predictions are read off the query tokens by a linear head,

$$
\hat { y } _ { j } ^ { q } = W _ { o } h _ { N _ { s } + j } ^ { ( L ) } , \qquad j \in \{ 1 , \dotsc , N _ { q } \} .\tag{6}
$$

## 3.2 Kernel baseline

We hypothesize that the MPN may be effectively implementing a form of non-parametric inference, determining the query labels as weighted sums of the labels of the nearest support molecules. To test this, we introduce a reduced architecture which makes this structure explicit. The resulting kernel MPN discards the transformer stack, and instead predicts each query label as a similarity-weighted average of the support labels, using a single meta-learned feature projection.

Given the same support set $\boldsymbol { S } = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } }$ and query set $\mathcal { Q } = \{ x _ { j } ^ { q } \} _ { j = 1 } ^ { N _ { q } }$ as before, the kernel baseline embeds all molecules with a shared linear projection $\phi : \mathbb { R } ^ { F } \overset { \cdot } {  } \mathbb { R } ^ { d }$ , and $L ^ { 2 }$ -normalizes the result to the unit sphere. Similarity between a query and a support molecule is measured by the cosine of the angle between their normalized projections, scaled by a learned temperature τ:

$$
s _ { j i } = \frac { \phi ( x _ { j } ^ { q } ) ^ { \top } \phi ( x _ { i } ^ { s } ) } { \tau } , \quad \mathrm { w h e r e } \quad \phi ( x ) = \frac { W _ { \phi } x } { \lVert W _ { \phi } x \rVert _ { 2 } } .\tag{7}
$$

Predictions are a softmax-weighted mean of the support labels,

$$
\hat { y } _ { j } ^ { q } = \sum _ { i = 1 } ^ { N _ { s } } \alpha _ { j i } y _ { i } ^ { s } , \qquad \alpha _ { j i } = \frac { \exp ( s _ { j i } ) } { \sum _ { i ^ { \prime } = 1 } ^ { N _ { s } } \exp ( s _ { j i ^ { \prime } } ) } ,\tag{8}
$$

which makes the model an instance of kernel regression with an exponential-cosine kernel and a meta-learned feature projection $\phi$

Table 1: Kendall’s τ on the FS-Mol test tasks as a function of the support set size $N _ { s }$ . The $N _ { s } = 2 5 6$ column is computed on the subset of tasks (43 out of 111) with enough samples to form a 256- molecule support set.
<table><tr><td rowspan="2">Model</td><td colspan="5">Support set size  $N _ { s }$ </td></tr><tr><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr><td>Random forest</td><td>0.138</td><td>0.198</td><td>0.255</td><td>0.310</td><td>0.354</td></tr><tr><td>GP-ST</td><td>0.163</td><td>0.214</td><td>0.262</td><td>0.315</td><td>0.358</td></tr><tr><td>ADKF-IFT</td><td>0.298</td><td>0.349</td><td>0.391</td><td>0.434</td><td>0.456</td></tr><tr><td>TabPFN</td><td>0.146</td><td>0.190</td><td>0.228</td><td>0.276</td><td>0.322</td></tr><tr><td>TabICL (500)</td><td>0.150</td><td>0.197</td><td>0.247</td><td>0.304</td><td>0.350</td></tr><tr><td>TabICL (2048)</td><td>0.156</td><td>0.200</td><td>0.250</td><td>0.309</td><td>0.358</td></tr><tr><td>kernel MPN</td><td>0.233</td><td>0.282</td><td>0.326</td><td>0.369</td><td>0.379</td></tr><tr><td>MPN (4 layers)</td><td>0.299</td><td>0.326</td><td>0.352</td><td>0.382</td><td>0.395</td></tr><tr><td>MPN (16 layers)</td><td>0.298</td><td>0.335</td><td>0.368</td><td>0.398</td><td>0.411</td></tr></table>

## 3.3 Training

Both of the architectures described above are trained end-to-end by meta-learning over a corpus of existing affinity datasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$ . Each dataset $\mathcal D _ { k } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { k } }$ corresponds to a single protein target and comprises pairs of molecular fingerprints and real-valued activity measurements. We denote the full set of learnable parameters by θ and the network’s prediction on query $x _ { j } ^ { q }$ given support set S by $\hat { y } _ { j } ^ { q } ( S ; \theta )$

At each training step, a dataset D is sampled from the corpus and randomly partitioned into a support set S and a query set Q of sizes drawn from a distribution over $( N _ { s } , N _ { q } )$ . The parameters are updated to minimize the mean squared error of the predicted query labels,

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { ( \mathcal { S } , \mathcal { Q } ) \sim \mathcal { D } } \frac { 1 } { N _ { q } } \sum _ { j = 1 } ^ { N _ { q } } \big ( \hat { y } _ { j } ^ { q } ( \mathcal { S } ; \theta ) - y _ { j } ^ { q } \big ) ^ { 2 } .\tag{9}
$$

## 4 Evaluation

## 4.1 Experimental setup

We evaluate the MPN models on the regression variant of the FS-Mol benchmark [Stanley et al., 2021, Chen et al., 2023]. We train on the 4938 training tasks, using the logarithmic regression targets (i.e., the pIC50 values). To avoid spending a disproportionate amount of time on a small group of datasets with a large number of samples, we randomly subsample sets larger than 500 molecules to 500 samples. The support size is uniformly sampled from the set {16, 32, 64, 128, 256} and capped at 80% of the entire dataset, so that at least 20% of the dataset is used as the query. Subsampling and splitting of individual datasets is performed independently at each epoch. We use the mean absolute error computed on the 40 validation tasks to select the best model within the training run, and report evaluation metrics averaged across the 111 test tasks and 10 different random splits for each task and support set size.

The molecules are featurized with Morgan fingerprints of radius 2 and length 2048. The attentionbased models use a hidden dimension of 1024 while the kernel baseline linearly projects the input without changing its dimensionality. All MPN models are trained with the Adam optimizer, with a learning rate of $1 0 ^ { - 4 }$ for the kernel variant and $1 0 ^ { - 5 }$ for the attention-based MPNs, batches of 10 tasks, and gradient norm clipped at 100. The kernel MPN is trained for a total of 100 epochs and the attention-based MPN for 400 epochs. In each case, we keep the checkpoint with the lowest validation MAE. The total time required to train a 16-layer MPN is approximately 4 days on a single NVIDIA A10G GPU.

We compare our models against ADKF-IFT [Chen et al., 2023], using a model checkpoint provided by the authors, TabPFNv2 [Hollmann et al., 2025] (fingerprints folded to 500 bits to match the model’s limit), and TabICLv2 [Qu et al., 2026] (both 500 and 2048-bit fingerprints). We also include the baselines of a tuned random forest and a Gaussian process with a Tanimoto kernel (GP-ST), fitted independently on the support set of every task, both of which we rerun without any changes from Chen et al. [2023]. We report the Kendall rank correlation coefficient τ between true and predicted affinities, which aligns with our interest in accurate ranking of the molecules.

![](images/ea1c38c0a2f1d7ab93dec496bf34cb93039aa0b05cd70106eae46ef9e5462018.jpg)  
Figure 1: Kendall’s τ on the FS-Mol test tasks as a function of the support set size. Error bars: standard deviation over 10 seeds.

![](images/da787d23cca05b1413f4b25e3fe537e611b8f9922b0d983e91532c174d31e4c4.jpg)  
Figure 2: Joint fit and inference time as a function of the support set size (100 query molecules, 10 CPU cores, median of 3 runs).

## 4.2 Results

The Kendall rank correlation coefficient for test tasks is reported in Table 1, with the corresponding plots in Figure 1. Overall, the strongest method remains ADKF-IFT, but for the smallest support set size of 16, both attention-based MPN models are able to match its performance. The kernel baseline turns out to be a surprisingly strong model, given that it performs no fitting on the evaluation task and reduces inference to a single matrix multiplication followed by a softmax operation and a weighted sum. It outperforms both random forests and basic Gaussian processes at every support set size. This supports the hypothesis of Section 3.2: a substantial fraction of the predictive performance achievable on FS-Mol can be expressed as kernel regression in a meta-learned feature space. The comparison between MPNs and general tabular foundation models shows that in settings for which a domain-specific pre-training corpus is available, it can lead to stronger performance than generic priors.

To illustrate the computational efficiency of MPNs, we pick one of the largest test datasets in FS-Mol and plot the joint time needed for fitting and inference on a fixed query set of 100 samples, as a function of the support set size. The results are shown in Figure 2. MPNs are capable of performing inference over thousands of molecules in a matter of seconds (attention) or hundredths of a second (kernel), which should allow future work to apply the same architecture across a wider range of support set sizes. We remark that both MPN and ADKF-IFT are roughly two times faster on a GPU, but GPU memory becomes a bottleneck for ADKF-IFT beyond 1000 samples.

## 5 Conclusion

In this work, we introduce the Molecular Prior Network (MPN), a new approach to protein-ligand affinity modelling. Inspired by Prior-data Fitted Networks, the MPN jointly processes labelled support and unlabelled query molecules given as inputs and predicts the labels of the query molecules, all in a single forward pass of the transformer model.

The MPN architecture opens a number of future research directions. One of them is explicit modelling of prediction uncertainty, which would enable the use of the model in active learning and Bayesian optimization settings. Another possibility is going beyond experimental affinity values, and training on a wider range of both experimentally and computationally derived molecular properties, turning MPNs into a foundation model for molecular property prediction. An orthogonal question is the representation of the input molecules: Morgan fingerprints could be augmented with additional descriptors or replaced with a graph neural network encoder, trained end-to-end with the transformer layers.

## References

H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. Pande. Low data drug discovery with one-shot learning. ACS central science, 3(4):283–293, 2017.

W. Chen, A. Tripp, and J. M. Hernandez-Lobato. Meta-learning adaptive deep kernel Gaussian pro-´ cesses for molecular property prediction. In The Eleventh International Conference on Learning Representations, 2023.

L. Grinsztajn, K. Floge, O. Key, F. Birkel, P. Jund, B. Roof, B. J ¨ ager, D. Safaric, S. Alessi, A. Hayler,¨ et al. TabPFN-2.5: Advancing the state of the art in tabular foundation models. arXiv preprint arXiv:2511.08667, 2025.

N. Hollmann, S. Muller, K. Eggensperger, and F. Hutter. TabPFN: A transformer that solves small ¨ tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, 2023.

N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. K¨ orfer, S. B. Hoo, R. T. Schirrmeister,¨ and F. Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637 (8045):319–326, 2025. doi: 10.1038/s41586-024-08328-6.

A. Karpathy. nanochat: The best ChatGPT that \$100 can buy, 2025. URL https://github.com/ karpathy/nanochat.

H. Kim, A. Mnih, J. Schwarz, M. Garnelo, A. Eslami, D. Rosenbaum, O. Vinyals, and Y. W. Teh. Attentive neural processes. In International Conference on Learning Representations, 2019.

E. Lee, J. Yoo, H. Lee, and S. Hong. MetaDTA: Meta-learning-based drug-target binding affinity prediction. In ICLR2022 Machine Learningfor Drug Discovery, 2022.

S. Muller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter. Transformers can do Bayesian¨ inference. In International Conference on Learning Representations, 2022.

J. Qu, D. Holzmuller, G. Varoquaux, and M. Le Morvan. TabICL: A tabular foundation model for¨ in-context learning on large data. In International Conference on Machine Learning, 2025.

J. Qu, D. Holzmuller, G. Varoquaux, and M. Le Morvan. TabICLv2: A better, faster, scalable, and¨ open tabular foundation model. arXiv preprint arXiv:2602.11139, 2026.

J. Schimunek, P. Seidl, L. Friedrich, D. Kuhn, F. Rippmann, S. Hochreiter, and G. Klambauer. Context-enriched molecule representations improve few-shot drug discovery. In The Eleventh International Conference on Learning Representations, 2023.

J. Schimunek, S. Luukkonen, and G. Klambauer. MHNfs: Prompting in-context bioactivity predictions for low-data drug discovery. Journal of Chemical Information and Modeling, 65(9): 4243–4250, 2025.

M. Stanley, J. F. Bronskill, K. Maziarz, H. Misztela, J. Lanini, M. Segler, N. Schneider, and M. Brockschmidt. FS-Mol: A few-shot learning dataset of molecules. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.