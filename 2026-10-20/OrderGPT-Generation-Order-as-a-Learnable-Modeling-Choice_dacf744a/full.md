# OrderGPT: Generation Order as a Learnable Modeling Choice

Chenhe Yang<sup>1</sup>, Yuhuan Lyu<sup>1</sup> and Tianyang Hu<sup>∗1</sup> <sup>1</sup>SAGA Lab, The Chinese University of Hong Kong, Shenzhen

Generative modeling of discrete data has two popular paradigms: autoregressive models such as GPT, which factorize a sequence in a left-to-right (L2R) order, and masked difusion models, which are essentially any-order autoregressive modeling. Rather than competing methods, these are the two extreme points of a single axis—the generation order. For structured discrete domains without a natural traversal, both extremes may be suboptimal, yet the intermediate choice—a single well-adapted order—remains largely unexplored. We introduce OrderGPT, which builds on Any-Order GPT to learn a task-level generation order without canonical-order supervision. Our key observation is that predicting variables from randomly sampled reveal prefixes produces recurring attention patterns in the backbone that provide a model-dependent signal of predictive dependencies. We aggregate this signal into a reciprocal dependency graph and use a spectral teacher to derive continuous generation priorities. A lightweight OrderMLP integrated into the GPT backbone learns to predict these priorities from attention, incorporating order learning into the architecture within a single training run. Training gradually shifts toward the learned order, which is then frozen for subsequent autoregressive training and generation. The same mechanism adapts across modalities: adaptively it recovers near-L2R dependency structure in language, learns anatomically organized factorizations on human pose, and discovers a factorization on tabular data that places statistically and semantically related fields nearby, while improving downstream modeling metrics across all domains. The code is available at GitHub

## 1. Introduction

Autoregressive generative models represent a joint distribution as a sequence of conditional prediction problems. This formulation underlies modern language models and has also been applied to visual tokens (Esser et al., 2021), human poses (Lucas et al., 2022), tabular data (Yang et al., 2019), graphs (You et al., 2018), and other structured domains (Uria et al., 2016). Although the chain rule permits the variables (or tokens) to be generated in any order, the choice of order matters for finite models: it determines the context available at each step, the dificulty of the conditional predictions, and the path along which errors propagate during generation (Bengio et al., 2015). Generation order is therefore an important modeling choice rather than merely an implementation detail.

Current methods largely occupy two endpoints of this design space. Conventional autoregressive models, such as GPT (Radford et al., 2019; Brown et al., 2020), commit to a single prescribed factorization, typically the L2R order for language. When this order matches the structure of the data, it provides a strong inductive bias and allows the model to concentrate its capacity on a stable family of prediction problems. Masked difusion models (MDMs) take the opposite approach (Lou et al., 2023; Shi et al., 2024; Sahoo et al., 2024; Nie et al., 2026; Xie et al., 2026) by reconstructing masked variables from arbitrary observed subsets without specifying a canonical traversal. Their objective can be interpreted as any-order autoregressive training that averages over randomly sampled reveal permutations (Ou et al., 2025; Xue et al., 2025). This flexibility is useful when no natural order exists, but it spreads a finite model’s capacity across many factorizations of potentially unequal dificulty without identifying a stable order for training and generation.

![](images/dcbd6ea51161b22e1adcac1b2a905c2df7135fc79089ae68dac25f0f44564c20.jpg)  
Figure 1 | OrderGPT augments the AO-GPT backbone with an Order MLP. The Spectral Teacher constructs a reciprocal afinity graph from aggregated AO-GPT attention, recovers a Fiedler axis, and orients it using directed attention. The resulting continuous priorities supervise the Order MLP, whose outputs are sorted to obtain the task-level generation order used by the backbone.

GPTs and MDMs can be viewed as two extremes along a common axis; both may be suboptimal for structured domains without a natural one-dimensional traversal. This motivates an intermediate alternative: learn a single task-level generation order and then use it in autoregressive factorization. Learning such an order is nontrivial: the search space is factorial, canonical-order supervision is generally unavailable, and the most useful factorization may evolve with the model.

The decoder-only GPT architecture provides a natural backbone for this goal. Its computational architecture and its conventional L2R factorization are conceptually separable: the same decoder can support diferent generation orders while retaining the GPT training and generation framework. AO-GPT (Xue et al., 2025) makes this flexibility explicit by supporting arbitrary reveal orders within a decoder-only Transformer. It therefore provides both the architectural foundation for learning a generation order and an opportunity to observe predictive relationships across diferent reveal contexts. Building on this backbone, we introduce OrderGPT, which incorporates a lightweight OrderMLP into AO-GPT to learn a task-level generation order without canonical-order supervision. Figure 1 summarizes the OrderGPT pipeline.

Our key observation is that predicting the same variable from diferent randomly sampled prefixes produces recurring attention patterns that provide model-dependent evidence of predictive relationships. We aggregate these patterns to construct a reciprocal dependency graph. Our Spectral Teacher recovers a continuous axis from the graph Laplacian, placing strongly related variables nearby, then uses directed attention to orient it, favoring attended context before its prediction targets. The resulting continuous generation priorities supervise OrderMLP, which learns to predict them from aggregated attention across successive windows, consolidating recurring dependency evidence. Sorting its predictions yields a task-level order that gradually replaces random orders during backbone training. The final order is frozen for subsequent autoregressive training and generation. OrderGPT thus integrates order learning into the GPT architecture within a single training run.

We evaluate OrderGPT across three domains with diferent structural properties. In language, it learns an order close to L2R without L2R supervision and lowers perplexity relative to matched any-order baselines. On H3WB human-pose generation (Zhu et al., 2023), where no unique one-dimensional traversal exists, it learns anatomically coherent factorizations and improves density fitting and freerunning generation. On the CENSUS tabular dataset (Meek et al., 2001), it places related fields nearby, improves density estimation, and reduces tail cardinality-estimation errors.

## 2. Preliminaries

Generation order as a modeling variable Let $\mathbf { x } = ( x _ { 1 } , \dots , x _ { D } )$ denote � target variables, c optional observed conditions, and $\pi = ( \pi _ { 1 } , \ldots , \pi _ { D } ) $ a permutation of variable IDs. Treating � as an explicit modeling variable gives the order-conditioned autoregressive factorization $p _ { \theta } ( \mathbf { x } \mid \mathbf { c } ; \pi ) =$ $\textstyle \prod _ { t = 1 } ^ { D } p _ { \theta } ( x _ { \pi _ { t } } \mid x _ { \pi _ { < t } } , \mathbf { c } )$ . �-GPT (Pannatier et al., 2024) implements this formulation by providing positional encodings for both the input variable and the next variable to predict. Causal attention follows the specified reveal order, while variable identities remain fixed across permutations. This makes the generation order independently configurable within the same decoder architecture.

AO-GPT backbone Xue et al. (2025) provides an implementation of any-order GPT within a decoderonly Transformer architecture. Specifically, it injects the target-position embedding into every Transformer block through adaptive layer normalization (adaLN). The embedding determines separate scale, shift, and residual-gating parameters for the attention and feed-forward branches, allowing the prediction target to modulate computation throughout the network. The backbone retains causal attention along the reveal order. For a specified permutation, it supports likelihood evaluation in a single forward pass and key–value caching during sequential generation. Given a distribution �<sub>AO</sub> over reveal permutations, the corresponding order-averaged objective is

$$
\mathcal { L } _ { \mathrm { A O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { ( \mathbf { x } , \mathbf { c } ) } \mathbb { E } _ { \pi \sim q _ { \mathrm { A O } } } \left[ - \sum _ { t = 1 } ^ { D } \log p _ { \boldsymbol { \theta } } \big ( x _ { \pi _ { t } } \mid x _ { \pi _ { < t } } , \mathbf { c } \big ) \right] .\tag{1}
$$

In the standard any-order setting, �<sub>AO</sub> is uniform over all �! permutations. For discrete data, the objective in Eq. (1) is equivalent to the standard masked difusion training objective at the formulation level (Xue et al., 2025; Ou et al., 2025).

Learning generation orders In encoder-only MDM literature, it is known that training contains dificult infilling subproblems and that a suitable decoding order exists (Kim et al., 2025; Peng et al., 2025; Garg et al., 2025; Hong et $\mathrm { a l . , }$ 2026). A closely related approach is LO-ARM (Wang et al., 2025), which treats the ordering as latent variables and learns a state-dependent order policy that sequentially selects the next variable. The policy is trained jointly with the autoregressive model through a variational lower bound on the log-likelihood. In comparison, OrderGPT learns a single task-level permutation $\pi ^ { * }$ that captures dependency structure shared across samples. It recovers this permutation from attention in an AO-GPT backbone, without introducing a sequential order policy or optimizing a distribution over orders. Once learned, the global order is frozen and reused for ordinary autoregressive training and generation, providing an explicit structural inductive bias without per-instance order search.

![](images/d5cbb18876a17402db57dcebfbc0d26f0c7eb8f56ee39e376ee4fdba338ea55c.jpg)

![](images/75f49f1658851cdf453afd1c338fa1b33a8bb72a632aa15e05c010318c91b22a.jpg)  
Figure 2 | Attention patterns under the current and original indexing. (a, b) WikiText103 Seq80 attention. (c, d) H3WB attention. The original-frame panels reorder the same values only for visualization, highlighting local tokens and anatomical structure. Each current/original pair shares a color scale.

## 3. Attention Reveals Order Preference

Generation order controls the context available to each prediction. We first show how this choice can afect the conditional distributions that a model with limited output capacity can represent. We then use attention from random-order training to estimate relationships between variables. Finally, we convert these relationships into a generation order.

## 3.1. Order Matters under Limited Capacity

The chain rule allows any generation order, but diferent orders require diferent conditional predictions. We formalize the resulting capacity tradeof in a toy setting, showing that a suitable order can achieve a strictly lower population loss than uniform random-order training at the same model capacity.

We consider categorical variables forming a Markov chain along a dependency path �. This path describes the data’s dependencies and need not match the original variable indexing. To make the capacity constraint explicit, all predictions use a shared categorical output layer with � features. Each prediction can still condition on the entire revealed prefix. We measure performance by expected negative log-likelihood per variable. Following the chain makes a neighboring variable available for each prediction after the first, so the model only needs to represent the local one-step conditionals. An order that skips an intermediate variable can instead require a longer-range conditional obtained by averaging over unrevealed states. Our construction makes the local conditionals representable within the shared output capacity, while a skipped-intermediate conditional has unavoidable approximation error.

Proposition 1 (Informal). Let $\mathcal { L } _ { \mathrm { c h a i n } } ^ { \star } ( C )$ denote the best achievable loss under the chain order, and let $\mathcal { L } _ { \mathrm { r a n d o m } } ^ { \star } ( C )$ denote the best average loss achievable by one predictor across uniformly sampled orders, within the same predictor family. There exist full-support reversible categorical Markov chains and capacity limits � for which the true joint distribution is exactly representable within this predictor family under the chain order and its reversal, whereas

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r a n d o m } } ^ { \star } ( C ) \geq \mathcal { L } _ { \mathrm { c h a i n } } ^ { \star } ( C ) + \Delta _ { C } , \qquad \Delta _ { C } > 0 . } \end{array}\tag{2}
$$

Uniform random ordering encounters the dificult prediction problems with positive probability, producing a non-vanishing gap in the best achievable population loss. At the stated capacity, this gap persists even with unlimited data and optimal fitting. Appendix A.2 gives the precise model definition, capacity conditions, construction, and proof.

The chain order and its reversal are efective because they make an informative neighboring variable available for each prediction after the first. This example motivates selecting orders that supply informative context for later predictions. In real data, the relevant relationships are unknown; we next investigate attention as a model-dependent signal for identifying them.

## 3.2. From Attention to a Reciprocal Dependency Graph

To identify predictive relationships that can guide order learning, we examine attention in an AO-GPT backbone trained with random reveal orders under Eq. (1). These orders expose each target to diferent prefixes, allowing us to examine which contextual variables repeatedly receive attention when available.

At the start of each run, we randomly relabel the variables and keep these IDs fixed across examples. Original IDs are used only for post-hoc evaluation. Figure 2 compares the same attention values under relabeled and original variable IDs. Restoring the original indexing reveals a diagonal band in language, indicating stronger attention between nearby text positions. In H3WB, distinct attention blocks align with body/feet, face, and the left and right hands. These observations motivate using attention as a model-dependent signal of relationships between variables.

We aggregate attention across training examples and random generation orders into a directed matrix �. Each entry $A _ { i j }$ averages attention to variable � when predicting variable $i ,$ using only predictions where � is already observed. To capture relationships supported in both directions, we construct a symmetric reciprocal dependency graph:

$$
W _ { i j } = \sqrt { A _ { i j } A _ { j i } } \quad ( i \ne j ) , \qquad W _ { i i } = 0 .\tag{3}
$$

The geometric mean favors mutually strong attention over one-sided links. The reciprocal graph � captures which variables are strongly related, while the directed matrix � preserves information about which variables are attended to when predicting others. We use these complementary signals to construct and orient a generation order.

## 3.3. Laplacian Recovery and Attention-Based Orientation

We now turn the reciprocal dependency graph � into a generation order. Motivated by the dependencyfollowing orders in Section 3.1, we seek an arrangement that places strongly connected variables nearby. We assign each variable a coordinate on a line and use a graph-Laplacian objective to penalize separation between connected variables. Sorting these coordinates yields a candidate order; we then use directed attention to choose its orientation.

Let $D _ { W }$ be the diagonal matrix with $\begin{array} { r } { ( D _ { W } ) _ { i i } = \sum _ { j } W _ { i j } } \end{array}$ . The graph Laplacian is $L = D _ { W } - W$ . Let z collect the variable coordinates, with $\mathcal { Z } _ { i }$ assigned to variable �. We measure the separation between connected variables by

$$
\mathcal { E } _ { W } ( \mathbf { z } ) = \mathbf { z } ^ { \mathsf { T } } L \mathbf { z } = \frac { 1 } { 2 } \sum _ { i , j } W _ { i j } ( z _ { i } - z _ { j } ) ^ { 2 } .\tag{4}
$$

We minimize this energy while fixing the coordinates to have zero mean and unit length (Fogel et al.,

2015):

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { v } _ { 2 } \in \mathrm { a r g } \quad \underset { \sum _ { i } z _ { i } = 0 , \parallel \mathbf { z } \parallel _ { 2 } = 1 } { \operatorname* { m i n } } \mathbf { z } ^ { \mathsf { T } } L \mathbf { z } , } \\ & { \pi ^ { \mathrm { r a w } } = \mathrm { a r g s o r t } ( \mathbf { v } _ { 2 } ) . } \end{array} } \end{array}\tag{5}
$$

Sorting the entries of $\mathbf { v } _ { 2 }$ in ascending order gives the candidate generation order $\pi ^ { \mathrm { { r a w } } }$ . For a connected graph, $\mathbf { v } _ { 2 }$ is a Fiedler vector, associated with the second-smallest eigenvalue of � (Atkins et al., 1998). When the second-smallest eigenvalue is not repeated and all coordinates are distinct, the order is unique up to reversal.

We use the directed attention matrix � to choose between these two directions. For an order �, let $S ( \pi )$ be the total attention from later variables to earlier ones:

$$
\begin{array} { l } { { \displaystyle S ( \pi ) = \sum _ { 1 \leq s < t \leq D } A _ { \pi _ { t } , \pi _ { s } } } , \ } \\ { { \displaystyle \pi ^ { s \mathrm { p e c } } \in \arg \operatorname* { m a x } _ { \pi \in \{ \pi ^ { \mathrm { r a w } } , \mathrm { r e v e r s e } ( \pi ^ { \mathrm { r a w } } ) \} } S ( \pi ) } . \ }  \end{array}\tag{6}
$$

After reordering � by $\pi ^ { \mathrm { r a w } }$ , the forward score is the sum below the diagonal, and the reverse score is the sum above it. We select the higher-scoring direction as $\pi ^ { \mathrm { { s p e c } } }$ . Equal scores indicate no directional preference. This choice uses only the training attention matrix. Appendix A.4 provides the derivation. We orient the Fiedler coordinates so that variables appearing earlier in $\pi ^ { \mathrm { { s p e c } } }$ have larger values. We call the complete procedure the Spectral Teacher.

Order recovery For the Markov-chain example in Section 3.1, suppose the attention graph is sufficiently close to a graph connecting neighboring variables along the chain. Spectral sorting then recovers the chain order up to reversal. Both directions retain the advantage over random orders at the same output capacity. A trained model can achieve this loss advantage if it learns the corresponding conditional predictions accurately enough. Appendix A.3 gives the precise conditions and proof. On WikiText103 Seq80, the Spectral Teacher recovers an order close to L2R, with Kendall’s $\tau = 0 . 9 8 9$ L2R is used only as a reference for evaluating the recovered order. Section 4.2 compares the Spectral Teacher’s predictive performance and computational cost with those of the Order MLP.

## 4. OrderGPT

Section 3 shows that recurring attention patterns in AO-GPT provide a model-dependent signal of predictive dependencies, which the Spectral Teacher converts into continuous generation priorities. We now incorporate this supervision into a trainable component of the GPT architecture, so that the model can learn its factorization as the backbone develops and use it to guide subsequent training within the same run.

## 4.1. Order MLP Distillation

We augment the AO-GPT backbone with a lightweight OrderMLP that maps the directed attention matrix � to one scalar priority per variable, with larger values indicating earlier generation. Training a shared mapping across successive attention windows encourages the module to capture recurring dependency structure and reduces its reliance on individual spectral estimates.

Supervision and training objective The Spectral Teacher rescales the Fiedler coordinates, oriented by Eq. (6), to [0, 1], assigning larger priorities to earlier variables. We train OrderMLP by minimizing the mean squared error between predicted and teacher priorities. The attention input is detached, so this loss updates only OrderMLP. Sorting its predictions in descending order yields the candidate generation order $\hat { \pi } _ { \mathrm { M L P } }$

![](images/ba9f86b3cf044833e7c48e8ed7238a17a2e81aa394d65f32997378edd0a83238.jpg)

![](images/7619d6bc4e43bcc030ba92b38eca0861026104b3276c7e613e8decacf3f5ff92.jpg)  
Figure 3 | Order MLP efectiveness on WikiText103 Seq80. (a) Validation loss under Order MLP, Spectral Teacher, Random-per-sample, Fixed-random AR, and Canonical AR. Vertical guides mark MLP distillation (10k–18k), annealing of the learned-order (18k–32k), and full use of the frozen learned order after 32k. The final learned axis has a positive post-hoc original-frame Kendall’s $\tau = 0 . 9 9 9$ . (b) Distillation MSE between Order-MLP priorities and oriented Fiedler targets.

## 4.2. Integrated Training Recipe

As shown in Figure 1, AO-GPT initially trains with random reveal orders. During this phase, the Spectral Teacher derives generation priorities from AO-GPT’s attention. The Order MLP learns to predict these priorities and converts them into a candidate generation order. AO-GPT continues to use random reveal orders throughout this initial distillation phase.

We then gradually increase the probability of training AO-GPT with the learned order, while sampling random orders otherwise. During the configured update window, the Order MLP can refresh the candidate order as AO-GPT’s attention evolves. At the end of this window, we freeze the final accepted order as $\pi ^ { * }$ . AO-GPT reuses this order across examples for subsequent autoregressive training and generation.

Efectiveness Figure 3(a) shows that Order MLP attains a lower final validation loss than Spectral Teacher, and closely approaches Canonical AR. The learned axes have positive post-hoc Kendall’s � compared with natural L2R: $\tau = 0 . 9 8 9$ for Spectral Teacher and $\tau = 0 . 9 9 9$ for Order MLP. Figure 3(b) provides complementary evidence: after an initial transient, the distillation MSE decreases and remains low across successive Spectral Teacher windows. Because the same MLP is optimized across these windows, its shared parameters consolidate recurring structure rather than copying each small, window-specific rank fluctuation into the executable order. This helps explain the lower validation loss and stronger final order agreement in panel a.

Eficiency We isolate order production with an optimized fixed-checkpoint replay at the matched Seq80 refresh cadence. The Spectral Teacher’s spectral path requires $O ( D ^ { 2 } )$ storage for the graph and $O ( D ^ { 3 } )$ time for the dense eigendecomposition; the frozen MLP consumes the already aggregated attention matrix and replaces the repeated spectral solve with a forward pass and sorting. In the replay, the Spectral Teacher requires 242.727 ms per optimizer iteration, whereas the frozen Order MLP requires 190.992 ms. After normalization by their paired Random-per-sample baselines, the per-iteration replay time is 21.336% lower for the Order MLP than for the Spectral Teacher.

Table 1 | Zero-shot perplexity (↓) on language benchmarks for approximately 125M-parameter models. Published baselines are reproduced from Table 5 of AO-GPT (Xue et al., 2025). <sup>†</sup> denotes a model reproduced by the AO-GPT authors, and <sup>‡</sup> denotes AO-GPT trained with an additional 10% of L2Rordered data. <sup>∗</sup> denotes our AO-GPT baseline trained for 400k updates without language-model EMA or a prescribed L2R training mixture. Published rows follow their original settings and provide numerical context rather than a compute-matched ranking. Fixed-random AR (throughout) uses one presampled random permutation throughout training and inference. Fixed-random AR + curriculum follows OrderGPT’s random-order warm-up and order-use annealing schedule, with a presampled random permutation as its terminal order for subsequent training and inference.
<table><tr><td>Model and inference order</td><td>LAMBADA</td><td>WikiText-2</td><td>PTB</td><td>WikiText-103</td><td>1BW</td></tr><tr><td>Published L2R evaluation</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEDD</td><td>49.410</td><td>41.190</td><td>118.740</td><td>41.700</td><td>72.600</td></tr><tr><td>RADD</td><td>49.090</td><td>38.260</td><td>107.780</td><td>38.410</td><td>63.330</td></tr><tr><td>GPT-2</td><td>45.040</td><td>42.430</td><td>138.430</td><td>41.600</td><td>75.200</td></tr><tr><td>σ-GPT†</td><td>68.610</td><td>57.660</td><td>146.870</td><td>55.540</td><td>90.980</td></tr><tr><td>AO-GPT</td><td>52.460</td><td>42.100</td><td>135.960</td><td>40.970</td><td>71.730</td></tr><tr><td>Published order-agnostic evaluation</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEDD</td><td>50.920</td><td>41.840</td><td>114.240</td><td>40.620</td><td>79.290</td></tr><tr><td>RADD</td><td>50.270</td><td>38.260</td><td>110.380</td><td>35.900</td><td>74.280</td></tr><tr><td>σ-GPT†</td><td>65.830</td><td>53.080</td><td>138.610</td><td>50.750</td><td>87.710</td></tr><tr><td> $\mathsf { A O - G P T ^ { \ddagger } }$ </td><td>59.930</td><td>46.330</td><td>141.920</td><td>45.440</td><td>84.360</td></tr><tr><td> $\mathrm { A O { - } G P T ^ { \ddagger } + E n s e m b l e \ ( } 8 )$ </td><td>55.620</td><td>42.770</td><td>126.080</td><td>42.020</td><td>77.620</td></tr><tr><td>AO-GPT‡ + Ensemble (64)</td><td>54.920</td><td>42.240</td><td>123.490</td><td>41.510</td><td>76.730</td></tr><tr><td colspan="6">Our 400k-update evaluation (no language-model EMA)</td></tr><tr><td>AO-GPT*+ Random inference</td><td>69.345</td><td>57.606</td><td>161.210</td><td>55.635</td><td>90.168</td></tr><tr><td> $_ { \mathsf { A O - G P T } ^ { * } }$  + Canonical L2R inference</td><td>70.262</td><td>60.273</td><td>163.053</td><td>58.535</td><td>91.069</td></tr><tr><td>Fixed-random AR (throughout)</td><td>71.140</td><td>57.830</td><td>160.790</td><td>56.310</td><td>90.390</td></tr><tr><td>Fixed-random AR + curriculum</td><td>67.690</td><td>53.600</td><td>149.530</td><td>51.800</td><td>86.870</td></tr><tr><td>OrderGPT (Order MLP)</td><td>50.962</td><td>45.125</td><td>132.753</td><td>39.217</td><td>70.046</td></tr></table>

## 5. Empirical Validation

We evaluate OrderGPT on language modeling, continuous human-pose generation, and categorical tabular modeling. These domains cover discrete sequences, structured continuous variables, and variables without a prescribed topology. We test whether learning task-level order improves performance over random-order training. Our main comparison is between OrderGPT and AO-GPT. Within each domain, the two methods use the same backbone capacity, data processing, and training budget.

Canonical AR uses L2R in language and the original dataset indexing in H3WB and CENSUS. Fixedrandom AR tests whether an arbitrary fixed order provides similar gains. In language, we evaluate the same AO-GPT<sup>∗</sup> model with random and L2R inference orders to test whether changing inference order alone reproduces the gains. Fixed-random AR + curriculum follows OrderGPT’s warm-up and annealing schedule but ends with a presampled random order, testing whether the schedule alone

Table 2 | Performance on H3WB masked-I2D. Internal results are reported as mean ± sample standard deviation across three independently trained model seeds $( n = 3$ per method). Each run is evaluated on the same 10,000-pose test split. NLL is averaged per joint, and MPJPE values are in millimeters. Published results are taken from Table 4 of Zhu et al. (2023) and provide external numerical context. Lower is better for every metric.
<table><tr><td>Method</td><td>Training data</td><td>NLL/joint ↓</td><td>Whole↓</td><td>Body ↓</td><td>Face-local ↓</td><td>Hand-local ↓</td></tr><tr><td>Large SimpleBaseline</td><td>H3WB</td><td>一</td><td>131.40</td><td>131.60</td><td>19.80</td><td>44.80</td></tr><tr><td>Jointformer</td><td>H3WB</td><td>1</td><td>109.20</td><td>103.00</td><td>19.80</td><td>53.50</td></tr><tr><td>Jointformer</td><td> $\mathrm { H 3 W B + T 3 W B }$ </td><td>一</td><td>84.20</td><td>80.10</td><td>16.30</td><td>44.50</td></tr><tr><td>Canonical AR</td><td>H3WB</td><td> $- 1 1 . 0 5 \pm 0 . 0 2$ </td><td> $8 0 . 8 3 \pm 1 . 8 0 $ </td><td> $7 4 . 8 6 \pm 0 . 5 9$ </td><td> $1 0 . 8 7 \pm 0 . 6 6$ </td><td> $3 7 . 4 1 \pm 3 . 8 7$ </td></tr><tr><td>Fixed-random AR</td><td>H3WB</td><td> $- 1 0 . 6 8 \pm 0 . 0 9$ </td><td> $9 9 . 2 6 \pm 1 1 . 8 7 $ </td><td> $8 2 . 6 9 \pm 8 . 5 3 $ </td><td> $1 3 . 1 4 \pm 2 . 6 8$ </td><td> $3 6 . 3 6 \pm 1 . 7 1$ </td></tr><tr><td>AO-GPT</td><td>H3WB</td><td> $- 1 0 . 5 9 \pm 0 . 0 7$ </td><td> $1 1 5 . 2 3 \pm 1 5 . 9 8$ </td><td> $8 9 . 9 7 \pm 8 . 4 0 $ </td><td> $1 4 . 6 7 \pm 2 . 9 5$ </td><td> $3 8 . 1 8 \pm 0 . 4 2 $ </td></tr><tr><td>OrderGPT</td><td>H3WB</td><td> $- 1 1 . 3 0 \pm 0 . 0 4$ </td><td> $7 9 . 4 4 \pm 1 . 7 6$ </td><td> $7 2 . 4 0 \pm 0 . 7 3$ </td><td> ${ \bf 1 0 . 1 6 \pm 0 . 7 4 }$ </td><td> ${ \bf 3 0 . 0 3 \pm 0 . 5 2 }$ </td></tr></table>

Table 3 | Performance in density modeling and cardinality estimation on CENSUS. Internal results are reported as mean ± sample standard deviation across three independently trained model seeds $( n = 3$ per method). Test NLL is averaged per variable on the held-out split and displayed in units of $1 0 ^ { - 3 }$ . Published rows reproduce the reported 10,000-particle results (Yang et al., 2019; Liang et al., 2020). Internal Q-error uses the same 1,000-query workload with 10,000 particles per query. Lower is better, and bold denotes the lowest internal mean.
<table><tr><td>Method</td><td>Test NLL/variable (×10−³) ↓ Q-error P50 ↓</td><td></td><td></td><td>Q-error P99 ↓ Maximum Q-error ↓</td></tr><tr><td>Published references</td><td></td><td></td><td></td><td></td></tr><tr><td>Progressive Sampling</td><td></td><td> $1 . 0 3 0 \pm 0 . 0 1 0$ </td><td> $1 . 4 9 0 \pm 0 . 0 8 0$ </td><td> $8 . 8 6 0 \pm 1 4 . 0 0 0$ </td></tr><tr><td>Variable Skipping</td><td></td><td> $1 . 0 9 0 \pm 0 . 0 1 0$ </td><td> $2 . 4 8 0 \pm 0 . 2 9 0$ </td><td> $1 4 . 7 0 0 \pm 7 . 0 0 0$ </td></tr><tr><td>Multi-Order + Skipping</td><td></td><td> $1 . 2 0 0 \pm 0 . 2 4 0$ </td><td> $3 . 2 0 0 \pm 0 . 2 9 0$ </td><td> $1 2 . 6 0 0 \pm 5 . 0 0 0$ </td></tr><tr><td>Internal evaluations</td><td></td><td></td><td></td><td></td></tr><tr><td>Canonical AR</td><td> $3 2 5 . 8 1 8 \pm 0 . 0 2 0$ </td><td> $1 . 0 2 9 \pm 0 . 0 0 3$ </td><td> $1 . 4 5 9 \pm 0 . 0 0 7$ </td><td> $2 . 8 6 2 \pm 1 . 1 3 7$ </td></tr><tr><td>Fixed-random AR</td><td> $3 2 5 . 9 8 3 \pm 0 . 0 1 9$ </td><td> $1 . 0 3 2 \pm 0 . 0 0 4$ </td><td> $1 . 5 1 4 \pm 0 . 1 0 0$ </td><td> $4 . 8 4 4 \pm 0 . 7 5 1$ </td></tr><tr><td>AO-GPT</td><td> $3 3 2 . 6 6 5 \pm 0 . 1 4 8$ </td><td> $1 . 0 4 6 \pm 0 . 0 1 2$ </td><td> $1 . 6 7 1 \pm 0 . 0 0 6$ </td><td> $1 0 . 5 0 0 \pm 4 . 4 7 8$ </td></tr><tr><td>OrderGPT</td><td> $\mathbf { 3 2 5 . 4 8 8 \pm 0 . 0 7 3 }$ </td><td> $\mathbf { 1 . 0 2 7 \pm 0 . 0 0 1 }$ </td><td> $1 . 4 4 2 \pm 0 . 0 1 3$ </td><td> $2 . 3 3 3 \pm 1 . 0 1 4$ </td></tr></table>

Note. For the positive cardinalities in this workload, with estimated cardinality ˆ� and true cardinality �, Q-error = max $( \hat { c } / c , c / \hat { c } ) ;$ ; its ideal value is one.

reproduces the gains.

## 5.1. Language Modeling

We train an approximately 125M-parameter decoder-only model on OpenWebText with a context length of 1024, and align the five-dataset evaluation suite with Table 5 of AO-GPT (Xue et al., 2025).

Learned order The final learned order closely matches L2R, with Kendall’s $\tau = 0 . 9 9 7$ after mapping variables back to their original positions. L2R is used only for this post-hoc comparison and does not guide order learning.

Improved performance For zero-shot benchmarking, OrderGPT and our AO-GPT<sup>∗</sup> baseline are trained for 400k updates.<sup>1</sup> Following AO-GPT, we report zero-shot perplexity on LAMBADA, WikiText-2, Penn

Treebank (PTB), WikiText-103, and the One Billion Word Benchmark (1BW). We evaluate the same trained AO-GPT<sup>∗</sup> model under either random inference orders or canonical L2R inference, whereas OrderGPT uses its frozen task-level generation order. As shown in Table 1, OrderGPT achieves lower perplexity than all internal baselines on all five benchmarks. For AO-GPT<sup>∗</sup>, L2R inference yields higher perplexity than random inference on every benchmark. Changing the inference order alone therefore does not reproduce OrderGPT’s gains. Fixed-random AR + curriculum improves over both AO-GPT<sup>∗</sup> with random inference and Fixed-random AR (throughout), but remains worse than OrderGPT on all five benchmarks. Thus, the gains observed with OrderGPT are not fully reproduced by a curriculum that terminates in an arbitrary fixed order.

## 5.2. Human-Pose Generation

The H3WB masked-I2D task (Zhu et al., 2023) requires the model to generate a complete 3D wholebody pose from incomplete 2D joint observations. Because human joints have local anatomical dependencies but no uniquely correct one-dimensional traversal, the task tests whether OrderGPT can learn a useful continuous-variable factorization without using the original joint numbering.

Learned order After mapping the final learned orders back to the original joint coordinates, we observe a consistent coarse anatomical-region traversal of face → body/feet → hands. This recurring pattern difers from the canonical H3WB indexing, which follows body/feet → face → hands. The shared region-level pattern suggests an ordering preference specific to the model and task.

Improved performance We next test whether the complete learned-order pipeline improves both density fitting and closed-loop pose generation. Teacher-forced negative log-likelihood (NLL) measures density fitting, whereas mean per-joint position error (MPJPE) measures 3D coordinate accuracy after a complete autoregressive rollout.<sup>2</sup> Averaged over the reported runs, OrderGPT achieves the lowest reported mean NLL and mean MPJPE in all four MPJPE columns among the internal methods in Table 2. These results reflect the complete learned-order pipeline, while Canonical AR already contains an anatomical-layout prior through the H3WB joint numbering.

## 5.3. Tabular Estimation

CENSUS (Meek et al., 2001) is a categorical tabular dataset where table columns have no natural generation order, making this a test of learning an order from their statistical dependencies. Alongside density modeling, we evaluate cardinality estimation, which predicts how many table rows satisfy a query.

Learned order Across independent runs, variables with strong mutual attention tend to appear close together in the learned order. We quantify this pattern using a locality score that measures average separation in generation positions. Lower scores indicate closer placement. For a representative run, the score is 17.910 for the learned order, 23.170 for the canonical column order, and 23.000 in expectation under random ordering. Original column identities and names are used only for this post-hoc interpretation.

Improved performance We measure density modeling with mean test NLL and cardinality estimation with Q-error. Q-error measures the factor by which a query’s row count is overestimated or underestimated. We evaluate 1,000 queries using 10,000 progressive-sampling particles per query. The median (P50) summarizes typical errors, while the 99th percentile (P99) and maximum capture larger errors. OrderGPT has the lowest reported mean on every metric among the internal methods in Table 3. Relative to AO-GPT, the largest reduction among the Q-error statistics is in the mean maximum. Compared with Canonical AR, the diferences are smaller, particularly for median Q-error. Thus, the complete OrderGPT pipeline improves both density modeling and cardinality estimation over AO-GPT, with particularly large gains in reducing the largest query errors.

## 6. Discussion

Our results support treating generation order as a learnable model choice within decoder-only autoregressive models. OrderGPT integrates attention-based order learning into the GPT backbone, learning a task-level factorization automatically without canonical-order supervision. Across language, human pose, and tabular data, the learned orders capture modality-specific structure and improve downstream performance, suggesting that a learned, stable factorization provides a useful middle ground between prescribed orders and uniform random-order training.

The current method shares one order across all examples, making it most suitable for tasks with recurring dependency structure. A global order may become restrictive when dependencies vary substantially across samples, and the attention signal itself remains conditioned on the model and reveal policy. Future work could combine reusable global order structure with context-dependent selection based only on variables available during generation.

## References

Jonathan E. Atkins, Erik G. Boman, and Bruce Hendrickson. A spectral algorithm for seriation and the consecutive ones problem. SIAM Journal on Computing, 28(1):297–310, 1998. doi: 10.1137/S0097539795285771.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 12868–12878. IEEE, 2021.

Fajwel Fogel, Rodolphe Jenatton, Francis Bach, and Alexandre d’Aspremont. Convex relaxations for permutation problems. SIAM Journal on Matrix Analysis and Applications, 36(4):1465–1488, 2015. doi: 10.1137/130947362.

Prateek Garg, Bhavya Kohli, and Sunita Sarawagi. Masked difusion models are secretly learned-order autoregressive models. arXiv preprint arXiv:2511.19152, 2025.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. OpenWebText corpus, 2019. URL http://Skylion007.github.io/OpenWebTextCorpus.

Chunsan Hong, Sanghyun Lee, and Jong Chul Ye. Unifying masked difusion models with various generation orders and beyond. arXiv preprint arXiv:2602.02112, 2026.

Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham M. Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked difusions. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 30749–30768. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/kim25ah.html.

Eric Liang, Zongheng Yang, Ion Stoica, Pieter Abbeel, Yan Duan, and Xi Chen. Variable skipping for autoregressive range density estimation. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 6040–6049. PMLR, 2020. URL https://proceedings.mlr.press/v119/liang20b.html.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete difusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Thomas Lucas, Fabien Baradel, Philippe Weinzaepfel, and Grégory Rogez. Posegpt: Quantizationbased 3d human motion generation and forecasting. In European Conference on Computer Vision, pages 417–435. Springer, 2022.

Chris Meek, Bo Thiesson, and David Heckerman. US Census Data (1990). UCI Machine Learning Repository, 2001. URL https://archive.ics.uci.edu/dataset/116/us+census+data+1 990. Dataset.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017. URL https://openreview.net /forum?id=Byj72udxe.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. Advances in Neural Information Processing Systems, 38:50608–50646, 2026.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete difusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, volume 2025, pages 64972–65009, 2025.

Arnaud Pannatier, Evann Courdier, and François Fleuret. �-gpts: A new approach to autoregressive models. In Joint European conference on machine learning and knowledge discovery in databases, pages 143–159. Springer, 2024.

Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Avishek Joey Bose, Alexander Tong, and Pranam Chatterjee. Path planning for masked difusion model sampling. arXiv preprint arXiv:2502.03540, 2025.

Alec Radford, Jef Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https: //cdn.openai.com/better-language-models/language-models.pdf.

Lixiang Ru, Yibing Zhan, Baosheng Yu, and Bo Du. Learning afinity from attention: End-to-end weakly-supervised semantic segmentation with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16846–16855, June 2022.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Mariano Marroquin, Justin T Chiu, Alexander M Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked difusion for discrete data. Advances in neural information processing systems, 37:103131– 103167, 2024.

Benigno Uria, Marc-Alexandre Côté, Karol Gregor, Iain Murray, and Hugo Larochelle. Neural autoregressive distribution estimation. Journal of Machine Learning Research, 17(205):1–37, 2016. URL https://jmlr.org/papers/v17/16-272.html.

Zhe Wang, Jiaxin Shi, Nicolas Heess, Arthur Gretton, and Michalis Titsias. Learning-order autoregressive models with application to molecular graph generation. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 64282–64301. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/wang25cn.html.

Tianyu Xie, Shuchen Xue, Zijin Feng, Tianyang Hu, Jiacheng Sun, Zhenguo Li, and Cheng Zhang. Variational autoencoding discrete difusion with enhanced dimensional correlations modeling. In International Conference on Learning Representations, volume 2026, pages 39109–39132, 2026.

Shuchen Xue, Tianyu Xie, Tianyang Hu, Zijin Feng, Jiacheng Sun, Kenji Kawaguchi, Zhenguo Li, and Zhi-Ming Ma. Any-order gpt as masked difusion model: Decoupling formulation and architecture. arXiv preprint arXiv:2506.19935, 2025.

Zongheng Yang, Eric Liang, Amog Kamsetty, Chenggang Wu, Yan Duan, Xi Chen, Pieter Abbeel, Joseph M. Hellerstein, Sanjay Krishnan, and Ion Stoica. Deep unsupervised cardinality estimation. Proceedings of the VLDB Endowment, 13(3):279–292, 2019. doi: 10.14778/3368289.3368294. URL https://www.vldb.org/pvldb/vol13/p279-yang.pdf.

Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, and Jure Leskovec. Graphrnn: Generating realistic graphs with deep auto-regressive models. In International conference on machine learning, pages 5708–5717. PMLR, 2018.

Yi Yu, Tengyao Wang, and Richard J. Samworth. A useful variant of the Davis–Kahan theorem for statisticians. Biometrika, 102(2):315–323, 2015. doi: 10.1093/biomet/asv008.

Yue Zhu, Nermin Samet, and David Picard. H3WB: Human3.6M 3d wholebody dataset and benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20166–20177, 2023. doi: 10.1109/ICCV51070.2023.01845. URL https://openaccess.thecvf.com/cont ent/ICCV2023/html/Zhu\_H3WB\_Human3.6M\_3D\_WholeBody\_Dataset\_and\_Benchmark\_ ICCV\_2023\_paper.html.

## Supplementary Material

OrderGPT: Generation Order as a Learnable Modeling Choice

## A. Theoretical Analysis and Complete Proofs

We prove an order gap for a specified categorical output class, then give suficient conditions connecting its beneficial axis to the Spectral Teacher and Order MLP. All logarithms are natural, and risks are negative log-likelihoods per variable.

## A.1. Setup and finite-capacity risk

Let � be a strictly positive distribution on $\{ 1 , \ldots , V \} ^ { D }$ , and let $\mathfrak { S } _ { D }$ be the permutations of the � variable IDs. The history $h _ { \pi , t }$ contains target ID $\pi _ { t }$ and the complete ordered prefix with its values and IDs. Consider the shared output class

$$
\mathcal { F } _ { C } = \left\{ q ( a \mid h ) = \operatorname { s o f t m a x } \bigl ( U \phi ( h ) + b \bigr ) _ { a } : U \in \mathbb { R } ^ { V \times C } , \ b \in \mathbb { R } ^ { V } \right\} ,\tag{A.1}
$$

where �, � are shared across all histories, and $\phi$ may be any finite-valued function into $\mathbb { R } ^ { C }$ . Thus $C$ limits the shared logit-feature dimension, while the complete prefix remains available. This is an output-interface capacity model related to the softmax bottleneck viewpoint.<sup>3</sup> The order-separation construction is proved below.

Define

$$
\begin{array} { l } { \displaystyle Q _ { \boldsymbol { q } , \pi } ( \boldsymbol { x } ) = \prod _ { t = 1 } ^ { D } \boldsymbol { q } ( x _ { \pi _ { t } } \mid h _ { \pi , t } ) , } \\ { \displaystyle R ( \boldsymbol { q } , \pi ) = - \frac { 1 } { D } \mathbb { E } _ { P } \log Q _ { \boldsymbol { q } , \pi } ( \boldsymbol { X } ) = \frac { H ( P ) } { D } + \frac { 1 } { D } \mathrm { K L } ( P \| Q _ { \boldsymbol { q } , \pi } ) . } \end{array}\tag{A.2}
$$

The last equality follows from the chain rule and cross-entropy identity; unrestricted exact conditionals give $H ( P ) / D$ for every order. The fixed-order and uniform any-order optima are

$$
\begin{array} { r l } & { R _ { C } ^ { \star } ( P , \pi ) = \underset { q \in \mathcal { F } _ { C } } { \operatorname* { i n f } } R ( q , \pi ) , } \\ & { R _ { \mathrm { A O } , C } ^ { \star } ( P ) = \underset { q \in \mathcal { F } _ { C } } { \operatorname* { i n f } } \mathbb { E } _ { \Pi \sim \mathrm { U n i f } ( \mathfrak { S } _ { D } ) } R ( q , \Pi ) . } \end{array}\tag{A.3}
$$

The latter infimum remains outside the expectation because one predictor is shared across orders. It corresponds to Eq. (1) divided by �, specialized to uniform orders and no additional conditioning variables. For the chain construction below, the main-text quantities satisfy $\mathcal { L } _ { \mathrm { c h a i n } } ^ { \star } ( C ) = R _ { C } ^ { \star } ( P , \rho )$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { r a n d o m } } ^ { \star } ( C ) = R _ { \mathrm { A O } , C } ^ { \star } ( P ) } \end{array}$

## A.2. A strict order gap under a shared softmax bottleneck

Theorem A.1 (finite-capacity order separation). For every $D \geq 3 , V \geq 4$ , and $2 \leq C \leq V - 2$ , there exist a strictly positive � and an axis $\rho \in \mathfrak { S } _ { D }$ such that $Y _ { t } = X _ { \rho _ { t } }$ is a stationary reversible first-order Markov chain and

$$
R _ { C } ^ { \star } ( P , \rho ) = R _ { C } ^ { \star } ( P , \mathrm { r e v } ( \rho ) ) = \frac { H ( P ) } { D } ,
$$

$$
R _ { { \mathrm { A O } } , C } ^ { \star } ( P ) \geq { \frac { H ( P ) } { D } } + \Gamma _ { C } , \qquad \Gamma _ { C } > 0 .\tag{A.4}
$$

The two chain directions require at most two shared features for exact representation, whereas an order beginning with $( \rho _ { 1 } , \rho _ { 3 } )$ requires at least $V - 1$

Proof: construction and the easy directions. For distinct real $s _ { 1 } , \ldots , s _ { V }$ and finite $\beta _ { 1 } , \ldots , \beta _ { V }$ , set

$$
Z _ { a } = \sum _ { b = 1 } ^ { V } \exp ( \beta _ { b } + s _ { a } s _ { b } ) , T _ { a b } = \frac { \exp ( \beta _ { b } + s _ { a } s _ { b } ) } { Z _ { a } } ,\tag{A.5}
$$

$$
\nu _ { a } = \frac { \exp ( \beta _ { a } ) Z _ { a } } { \sum _ { j } \exp ( \beta _ { j } ) Z _ { j } } , \qquad P ( y ) = \nu _ { y _ { 1 } } \prod _ { t = 2 } ^ { D } T _ { y _ { t - 1 } , y _ { t } } .
$$

All probabilities are positive, and

$$
\nu _ { a } T _ { a b } = \frac { \exp ( \beta _ { a } + \beta _ { b } + s _ { a } s _ { b } ) } { \sum _ { j } \exp ( \beta _ { j } ) Z _ { j } } = \nu _ { b } T _ { b a }\tag{A.6}
$$

proves stationarity and reversibility. Choose $U _ { a } = ( s _ { a } , \log Z _ { a } , 0 , . . . , 0 ) , b _ { a } = \beta _ { a }$ , and

$$
\phi ( h ) = \left\{ \begin{array} { l l } { { ( 0 , 1 , 0 , \ldots , 0 ) , } } & { { \mathrm { e m p t y p r e f i x , } } } \\ { { ( s _ { y _ { \mathrm { l a s t } } } , 0 , 0 , \ldots , 0 ) , } } & { { \mathrm { n o n e m p t y p r e f i x . } } } \end{array} \right.\tag{A.7}
$$

Here $y _ { \mathrm { l a s t } }$ is the state of the most recently revealed variable. The empty-prefix output equals $\nu ,$ and subsequent outputs equal the required rows of �. The Markov property and reversibility make these the full-prefix conditionals in both chain directions, so this shared predictor attains $H ( P ) / D$ with two features. Other members of $\mathcal { F } _ { C }$ are not restricted to reading one prefix variable.

Shared-output rank constraint. Stack features for � histories into $\Phi \in \mathbb { R } ^ { N \times C }$ and predictions into $Q \in \mathbb { R } ^ { N \times V }$ . With $H _ { n } = I _ { n } - \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } / n$ and row log-normalizers $c ,$ the entrywise logarithm satisfies

$$
\begin{array} { c } { { \log Q = \Phi U ^ { \mathsf { T } } + \mathbf { 1 } _ { N } b ^ { \mathsf { T } } - c \mathbf { 1 } _ { V } ^ { \mathsf { T } } , } } \\ { { H _ { N } \log Q H _ { V } = ( H _ { N } \Phi ) ( U ^ { \mathsf { T } } H _ { V } ) , \qquad \mathrm { r a n k } ( H _ { N } \log Q H _ { V } ) \leq C . } } \end{array}\tag{A.8}
$$

Double centering removes shared biases and row normalizers, yielding a constraint on log-probability tables.

Lemma A.1 (centered logarithmic kernel). For distinct positive $x _ { 1 } , \ldots , x _ { V }$ and distinct positive $y _ { 1 } , \ldots , y _ { V } .$ , let $F _ { i j } = \log ( 1 + x _ { i } y _ { j } )$ . Then rank $\left( H _ { V } F H _ { V } \right) = V - 1$

Proof. If the rank were smaller, there would be a nonzero $z \in \mathbf { 1 } _ { V } ^ { \perp }$ with $F z = c \mathbf { 1 } _ { V }$ . Then

$$
f ( x ) = \sum _ { j } z _ { j } \log ( 1 + x y _ { j } ) - c , \qquad f ^ { \prime } ( x ) = \sum _ { j } { \frac { z _ { j } } { x + 1 / y _ { j } } }\tag{A.9}
$$

has � distinct positive zeros, so $f ^ { \prime }$ has at least $V - 1$ by Rolle’s theorem. Multiplying $f ^ { \prime }$ by $\textstyle \prod _ { j } ( x + 1 / y _ { j } )$ gives a polynomial of degree at most $V - 2 ,$ , since $\begin{array} { r } { \sum _ { j } z _ { j } = 0 } \end{array}$ . The denominator has no positive root, so this polynomial must vanish identically. Evaluating it at each $- 1 / y _ { j }$ forces $z _ { j } = 0 _ { : }$ , since the poles are distinct, a contradiction. The centering matrices already bound the rank above by $V - 1$ , proving equality.

A dificult two-step conditional withfinite parameters. In Eq. (A.5), take $s _ { 1 } = 0 , s _ { 2 } = 1 , \beta _ { 1 } = \beta _ { 2 } = 0$ and $\beta _ { a } = - M$ for $a \ge 3$ , keeping the remaining $s _ { a }$ distinct and fixed. For $K = T ^ { 2 }$ , multiplication gives

$$
\log K _ { a c } = \beta _ { c } - \log Z _ { a } + \log \left[ \sum _ { b } \frac { \exp ( \beta _ { b } ) } { Z _ { b } } \exp \bigl ( s _ { b } ( s _ { a } + s _ { c } ) \bigr ) \right] .\tag{A.10}
$$

Double centering removes the first two terms. As $M  \infty , Z _ { 1 }  2 , Z _ { 2 }  1 + e _ { \cdot }$ and summands with $b \geq 3$ vanish, so the remaining matrix converges entrywise to

$$
- \log 2 + \log \left( 1 + \frac { 2 } { 1 + e } \exp ( s _ { a } + s _ { c } ) \right) .\tag{A.11}
$$

Apply Lemma A.1 with $x _ { a } = \exp ( s _ { a } )$ and $y _ { c } = 2 \exp ( s _ { c } ) / ( 1 + e )$ . The centered limit has a nonzero minor of order $V - 1$ , which remains nonzero for suficiently large finite � by continuity. Fix such a finite $M ;$ all probabilities remain positive and

$$
\mathrm { r a n k } ( H _ { V } \log K H _ { V } ) = V - 1 .\tag{A.12}
$$

The limit only proves existence of finite parameters, without taking logarithms of zero-probability transitions.

Strictly positive conditional KL gap. An order beginning with $( \rho _ { 1 } , \rho _ { 3 } )$ requires $P ( Y _ { 3 } = c \mid Y _ { 1 } = a ) = K _ { a c }$ at its second step. Let $Q _ { C }$ contain the prediction tables $Q _ { a c }$ realizable by $\mathcal { F } _ { C }$ on these � target-prefix histories, and define

$$
\ell _ { C } : = \operatorname* { i n f } _ { Q \in Q _ { C } } \sum _ { a } \nu _ { a } \operatorname { K L } ( K _ { a , : } \| Q _ { a , : } ) .\tag{A.13}
$$

We claim $\ell _ { C } > 0$ . Otherwise there is a sequence $Q ^ { ( n ) } \in Q _ { C }$ whose weighted KL tends to zero. Since every $\nu _ { a } > 0$ , each row KL tends to zero, and Pinsker’s inequality gives $Q ^ { ( n ) } \to K$ entrywise. Strict positivity of � makes the entrywise logarithm continuous at $K ,$ hence $H _ { V }$ log $Q ^ { ( n ) } H _ { V } \to H _ { V } \log K H _ { V }$ Every matrix in this sequence has rank at most � by Eq. (A.8). The set of matrices of rank at most � is closed, because all minors of order $C + 1$ vanish on it. The limit would therefore have rank at most $C \leq V - 2$ , contradicting Eq. (A.12). This proves a positive KL gap, without assuming that the infimum is attained.

From conditional KL to any-order risk. The chain rule for KL gives

$$
\operatorname { K L } ( P \| Q _ { q , \pi } ) = \sum _ { t = 1 } ^ { D } \mathbb { E } _ { P } \operatorname { K L } \left( P ( X _ { \pi _ { t } } \mid X _ { \pi _ { < t } } ) \parallel q ( \cdot \mid h _ { \pi , t } ) \right) .\tag{A.14}
$$

All terms are nonnegative, and a uniform permutation begins with $( \rho _ { 1 } , \rho _ { 3 } )$ with probability $1 / [ D ( D -$ 1)]. On that event, its second conditional KL is at least $\ell _ { C }$ after averaging over $Y _ { 1 } \sim \nu$ . Consequently, for every shared predictor,

$$
\mathbb { E } _ { \Pi } R ( q , \Pi ) \ge \frac { H ( P ) } { D } + \Gamma _ { C } , \qquad \Gamma _ { C } = \frac { \ell _ { C } } { D ^ { 2 } ( D - 1 ) } > 0 .\tag{A.15}
$$

Taking the infimum preserves this bound; the explicit two-feature predictor establishes equality with $H ( P ) / D$ in both chain directions and completes Theorem A.1. Equation (2) therefore holds with $\Delta _ { C } = \Gamma _ { C }$

This is a representational result for Eq. (A.1), not a lower bound on arbitrary network parameter counts or a training-speed guarantee. A subclass with the same output interface inherits the lower bound, while its chain-order upper bound additionally requires that it implement Eq. (A.7). The theorem does not make every fixed random order worse or quantify the size of the gap in the experimental datasets.

## A.3. Graph alignment and conditional spectral recovery

We write $\widehat { A }$ for the empirical matrix � defined by the exposure-corrected estimator in Appendix B.1, Eq. (B.2). Its reciprocal graph $\widehat { W }$ is symmetric and nonnegative, with zero diagonal. The following condition connects that graph to the same axis $\rho$ certified by Theorem A.1.

Assumption A.1 (alignment with the beneficial axis). Let $W _ { \rho }$ be the unweighted path adjacency matrix on consecutive pairs $\left( \rho _ { t } , \rho _ { t + 1 } \right)$ . Assume

$$
\widehat { W } = \alpha W _ { \rho } + \beta ( \mathbf { 1 1 } ^ { \mathsf { T } } - I ) + E , \qquad \alpha > 0 , \quad \beta \geq 0 ,\tag{A.16}
$$

where � is symmetric with zero diagonal and $\widehat { W }$ remains nonnegative. For any symmetric weight matrix $B ,$ write $L ( B ) = \mathrm { d i a g } ( B 1 ) - B $ , and set $L _ { \rho } = L ( W _ { \rho } )$ and $\eta = \| L ( E ) \| _ { \mathrm { o p } }$ . The substantive assumption is the bound on $\eta / \alpha$ below; $\alpha , \beta$ are graph coeficients unrelated to the categorical biases $\beta _ { a }$

Theorem A.2 (recovery of the chain axis). Define

$$
( \nu _ { \rho } ) _ { \rho _ { t } } = - \sqrt { \frac { 2 } { D } } \cos \left( \frac { \pi ( t - \frac { 1 } { 2 } ) } { D } \right) , \quad t = 1 , \ldots , D ,\tag{A.17}
$$

and the positive spectral and coordinate gaps

$$
\begin{array} { r l } & { g _ { \rho } = \underset { t < D } { \operatorname* { m i n } } \{ \lambda _ { 2 } ( L _ { \rho } ) , \lambda _ { 3 } ( L _ { \rho } ) - \lambda _ { 2 } ( L _ { \rho } ) \} , } \\ & { m _ { \rho } = \underset { t < D } { \operatorname* { m i n } } \big ( ( \nu _ { \rho } ) _ { \rho _ { t + 1 } } - ( \nu _ { \rho } ) _ { \rho _ { t } } \big ) . } \end{array}\tag{A.18}
$$

Under Assumption A.1, if

$$
{ \frac { \eta } { \alpha } } < \operatorname* { m i n } \left\{ { \frac { g _ { \rho } } { 4 } } , { \frac { m _ { \rho } g _ { \rho } } { 8 \sqrt { 2 } } } \right\} ,\tag{A.19}
$$

then $\widehat { L } = L ( \widehat { W } )$ has a simple Fiedler eigenvalue, and sorting a unit Fiedler vector $\widehat { \nu }$ recovers $\rho$ up to reversal.

Proof. The path eigenvalues are $\lambda _ { k + 1 } ( L _ { \rho } ) = 2 ( 1 - \cos ( k \pi / D ) )$ for $k = 0 , \ldots , D - 1$ . Substitution verifies that Eq. (A.17) is its unit, zero-mean Fiedler vector, with coordinates strictly increasing along $\rho .$ . Thus both gaps are positive, and the path is recovered exactly, as in spectral seriation (Atkins et al., 1998; Fogel et al., 2015). Linearity gives

$$
\widehat { L } = L _ { 0 } + L ( E ) , \qquad L _ { 0 } = \alpha L _ { \rho } + \beta ( D I - { \bf 1 1 } ^ { \mathsf { T } } ) .\tag{A.20}
$$

On $\mathbf { 1 ^ { \perp } }$ , the background adds $\beta D$ times the identity, preserving eigenvectors and giving $\lambda _ { k } ( L _ { 0 } ) =$ $\alpha \lambda _ { k } ( L _ { \rho } ) + \beta D$ for $k \geq 2$ . Its Fiedler eigenvalue is separated from zero and the remaining spectrum by at least $\alpha g _ { \rho }$ . Weyl’s inequality and $\eta < \alpha g _ { \rho } / 4$ ensure that $\widehat { L }$ is connected and its Fiedler eigenvalue remains simple.

Projecting the perturbed eigenvector equation onto $\upsilon _ { \rho } ^ { \perp } .$ , the inverse of $L _ { 0 } - \lambda _ { 2 } ( \widehat { L } ) I$ on that subspace has norm at most $( \alpha g _ { \rho } - \eta ) ^ { - 1 }$ . Hence the acute angle $\theta$ between the eigenvector lines satisfies sin $\theta \leq \eta / ( \alpha g _ { \rho } - \eta ) \leq 2 \eta / ( \alpha g _ { \rho } )$ , the standard Davis–Kahan argument (Yu et al., 2015). After choosing a common sign $s \in \{ - 1 , 1 \}$ ,

$$
\| s \widehat { \boldsymbol { \nu } } - \boldsymbol { \nu } _ { \rho } \| _ { \infty } \leq \| s \widehat { \boldsymbol { \nu } } - \boldsymbol { \nu } _ { \rho } \| _ { 2 } \leq \sqrt { 2 } \sin \theta \leq \frac { 2 \sqrt { 2 } \eta } { \alpha g _ { \rho } } < \frac { m _ { \rho } } { 4 } .\tag{A.21}
$$

Every consecutive coordinate gap remains greater than $m _ { \rho } / 2 .$ , proving recovery up to reversal.

Uniform background does not change the exact ordering; nonuniform background must be included in �. If the eigensolver contributes an additional sign-aligned coordinate error $\varepsilon _ { \mathrm { e i g } }$ , the same proof applies when

$$
\eta < \frac { \alpha g _ { \rho } } { 4 } , \qquad \frac { 2 \sqrt { 2 } \eta } { \alpha g _ { \rho } } + \varepsilon _ { \mathrm { e i g } } < \frac { m _ { \rho } } { 2 } .\tag{A.22}
$$

These are deterministic conditions, not finite-sample guarantees that a trained Transformer satisfies them. In particular, low predictive loss alone need not identify raw attention, because value representations or other computational paths may compensate for its weights. More attention samples may reduce estimation error but do not automatically remove structural bias or checkpoint drift.

## A.4. Attention-based orientation

Let $\pi ^ { \mathrm { r a w } } = \mathrm { a r g s o r t } _ { \mathrm { a s c } } ( \widehat { \nu } )$ . For target-to-observed �, define $\begin{array} { r } { S _ { A } ( \pi ) = \sum _ { s < t } A _ { \pi _ { t } , \pi _ { s } } } \end{array}$ and let $B _ { t s } = A _ { \pi _ { t } ^ { \mathrm { r a w } } , \pi _ { s } ^ { \mathrm { r a w } } }$

Proposition A.1 (triangular and antisymmetric forms). The forward and reverse scores are

$$
S _ { A } ( \pi ^ { \mathrm { r a w } } ) = \sum _ { t > s } B _ { t s } , \qquad S _ { A } ( \mathrm { r e v } ( \pi ^ { \mathrm { r a w } } ) ) = \sum _ { t < s } B _ { t s } .\tag{A.23}
$$

Writing $\begin{array} { r } { M _ { A } = \sum _ { i \neq j } A _ { i j } } \end{array}$ and $A ^ { - } = ( A - A ^ { \mathsf { T } } ) / 2$ , every order satisfies

$$
\begin{array} { l } { { \displaystyle S _ { A } ( \pi ) + S _ { A } ( \mathrm { r e v } ( \pi ) ) = M _ { A } , } } \\ { { \displaystyle S _ { A } ( \pi ) - S _ { A } ( \mathrm { r e v } ( \pi ) ) = \sum _ { s < t } ( A _ { \pi _ { t } , \pi _ { s } } - A _ { \pi _ { s } , \pi _ { t } } ) = 2 \sum _ { s < t } A _ { \pi _ { t } , \pi _ { s } } ^ { - } . } } \end{array}\tag{A.24}
$$

Proof. Reversal swaps the earlier and later member of each unordered pair, interchanging its lowerand upper-triangular entries. Adding the scores counts both directed entries once, while subtracting them gives their signed diference. Summing over pairs proves both identities.

Writing $\pi _ { T } = \pi ^ { \mathrm { s p e c } }$ for the Spectral Teacher order defined in the main text, the Teacher selects

$$
\pi _ { T } \in \mathop { \mathrm { a r g } } _ { \pi \in \{ \pi ^ { \mathrm { r a w } } , \mathrm { r e v } ( \pi ^ { \mathrm { r a w } } ) \} } S _ { \widehat { A } } ( \pi ) .\tag{A.25}
$$

Diagonal entries are excluded, and symmetric � gives a tie. The score compares prefix-compatible attention on the existing matrix, without new backbone evaluations or an identity relating the score to NLL. The reciprocal graph is invariant under $A \mapsto A ^ { \mathsf { T } }$ , whereas the score diference changes sign. For Theorem A.1, both recovered directions are already beneficial, so any maximizer, including either tied candidate, preserves the capacity advantage. For distributions with only one beneficial direction, an additional relation between that direction and the attention score would be needed.

## A.5. Priority normalization and preservation by the Order MLP

Using the main text’s convention that earlier variables have larger priorities, define

$$
z ^ { \mathrm { e a r l y } } = \left\{ \begin{array} { l l } { - \widehat { \nu } , } & { \pi _ { T } = \pi ^ { \mathrm { r a w } } , } \\ { \widehat { \nu } , } & { \pi _ { T } = \mathrm { r e v } ( \pi ^ { \mathrm { r a w } } ) , } \end{array} \right. \qquad r _ { i } = \frac { z _ { i } ^ { \mathrm { e a r l y } } - \mathrm { m i n } _ { j } z _ { j } ^ { \mathrm { e a r l y } } } { \mathrm { m a x } _ { j } z _ { j } ^ { \mathrm { e a r l y } } - \mathrm { m i n } _ { j } z _ { j } ^ { \mathrm { e a r l y } } } .\tag{A.26}
$$

Theorem A.2 ensures distinct coordinates, so descending sorting of � gives $\pi _ { T }$ . With a unique score maximizer, the arbitrary Fiedler sign only swaps candidate names. In a tie, selecting the lexicographically smaller current-ID sequence is one sign-invariant convention; either candidate remains certified.

Proposition A.2 (ranking fidelity). For the final Teacher target, let

$$
\Delta _ { r } = \operatorname* { m i n } _ { t < D } ( r _ { \pi _ { T , t } } - r _ { \pi _ { T , t + 1 } } ) > 0 .\tag{A.27}
$$

Descending sorting of the final MLP output $\widehat { r }$ recovers $\pi _ { T }$ if

$$
\| \widehat { r } - r \| _ { \infty } < \frac { \Delta _ { r } } { 2 } .\tag{A.28}
$$

A suficient condition on the same output vector is

$$
\mathrm { M S E } ( \widehat { r } , r ) : = D ^ { - 1 } \sum _ { i } ( \widehat { r } _ { i } - r _ { i } ) ^ { 2 } < \frac { \Delta _ { r } ^ { 2 } } { 4 D } .\tag{A.29}
$$

Proof. For $e = \| { \widehat { r } } - r \| _ { \infty } .$ , each consecutive diference is at least $\Delta _ { r } - 2 e > 0$ , preserving the complete order. The MSE condition follows from $\| \widehat { \boldsymbol { r } } - \boldsymbol { r } \| _ { \infty } \leq \| \widehat { \boldsymbol { r } } - \boldsymbol { r } \| _ { 2 } = \sqrt { D \operatorname { M S E } ( \widehat { \boldsymbol { r } } , \boldsymbol { r } ) }$

The margin uses the normalized [0, 1] scale, and the error must hold on the final graph used for the executable order. Average MSE across windows alone does not imply this condition or establish convergence of distillation.

## A.6. Conditional risk guarantee

Theorem A.3 (preservation of the capacity advantage). Consider the same distribution and output class as Theorem A.1. Suppose the actual attention graph satisfies Assumption A.1 and Eq. (A.19), with an exact Fiedler vector or numerical error controlled by Eq. (A.22). Let $\pi _ { T }$ follow Eq. (A.25), and either execute $\widehat { \pi } = \pi _ { T }$ directly or obtain �b by descending sorting of an MLP output satisfying Eq. (A.28). If the final predictor $\widehat { q } \in \mathcal { F } _ { C }$ satisfies

$$
R ( \widehat { q } , \widehat { \pi } ) \leq R _ { C } ^ { \star } ( P , \widehat { \pi } ) + \varepsilon _ { \mathrm { f i t } } ,\tag{A.30}
$$

then

$$
\begin{array} { c } { { \displaystyle \widehat \pi \in \{ \rho , \mathrm { r e v } ( \rho ) \} , } } \\ { { \displaystyle R ( \widehat q , \widehat \pi ) \leq \frac { H ( P ) } { D } + \varepsilon _ { \mathrm { f i t } } , } } \\ { { \displaystyle R _ { \mathsf { A O } , C } ^ { \star } ( P ) - R ( \widehat q , \widehat \pi ) \geq \Gamma _ { C } - \varepsilon _ { \mathrm { f i t } } . } } \end{array}\tag{A.31}
$$

Proof. Theorem A.2 recovers exactly the two chain directions, orientation selects one, and Proposition A.2 preserves it when the MLP is used. Theorem A.1 gives $R _ { C } ^ { \star } ( P , \widehat { \pi } ) = H ( P ) / D ;$ combining the fitting bound with its AO lower bound proves the result. Both directions are certified, so no positive orientation margin is required.

Strict improvement requires $\varepsilon _ { \mathrm { f i t } } < \Gamma _ { C }$ and compares the selected fixed-order risk with the uniform-order average in the same class. The theorem first certifies beneficial orders independently of attention, then assumes graph alignment and controls recovery, distillation, and fitting. It does not establish that native Transformer training meets these conditions, certify them for the reported datasets, or identify a uniquely optimal direction.

## B. Implementation Details

## B.1. Attention Alignment and Exposure Correction

Attention matrices from diferent reveal orders must refer to the same ordered variable pairs before they are averaged. Let � map original variable IDs to current IDs, so that $x _ { i } ^ { \mathrm { c u r } } = x _ { \sigma ^ { - 1 } ( i ) } ^ { \mathrm { o r i g } }$ . This relabeling is fixed across examples; variable-aligned conditions follow it, whereas global conditions are unchanged. For example $b ,$ let $\pi _ { b }$ be its reveal permutation and $r _ { b } ( i ) = \pi _ { b } ^ { - 1 } ( i )$ its one-based rank map. Original IDs and semantic names are reserved for post-hoc interpretation.

The decoder input is (start, $x _ { \pi _ { b , 1 } } , \ldots , x _ { \pi _ { b , D } } )$ : query position $t - 1$ predicts $x _ { \pi _ { b , t } } ,$ while key position $s \geq 1$ contains $x _ { \pi _ { b , s } }$ . Consequently, the target-to-observed view of raw $\left( D + 1 \right) \times \left( D + 1 \right)$ attention $R ^ { ( b , \ell , h ) }$ is $\widetilde { a } _ { t s } ^ { ( b , \ell , h ) } = R _ { t - 1 , s } ^ { ( b , \ell , h ) }$ for $1 \leq t , s \leq D$ . The zero-based slice R[:-1, 1:] removes the start-state key and the unused final query, giving the visibility condition $s < t .$ An unshifted row index would associate the query with the previous observed variable rather than its prediction target. We map both axes to current IDs before averaging:

$$
a _ { i j } ^ { ( b , \ell , h ) } = \widetilde { a } _ { r _ { b } ( i ) , r _ { b } ( j ) } ^ { ( b , \ell , h ) } , \qquad M _ { i j } ^ { ( b ) } = \mathbb { I } \{ r _ { b } ( j ) < r _ { b } ( i ) \} .\tag{B.1}
$$

The mask includes a pair only when the column variable was legally visible to the row target. Removing the start key does not redistribute its original softmax mass over the retained keys.

Exposure correction. For � examples, average all $N _ { L }$ decoder layers and $N _ { H }$ heads equally, then accumulate masked sums and legal-exposure counts:

$$
\begin{array} { l } { { \displaystyle \bar { a } _ { i j } ^ { ( b ) } = \frac { 1 } { N _ { L } N _ { H } } \sum _ { \ell , h } a _ { i j } ^ { ( b , \ell , h ) } , } } \\ { { \displaystyle U _ { i j } = \sum _ { b } { \cal M } _ { i j } ^ { ( b ) } \bar { a } _ { i j } ^ { ( b ) } , \qquad C _ { i j } = \sum _ { b } { \cal M } _ { i j } ^ { ( b ) } } , } \\ { { \displaystyle A _ { i j } = \frac { U _ { i j } } { \operatorname* { m a x } ( 1 , C _ { i j } ) } \quad ( i \ne j ) , \qquad A _ { i i } = 0 . } } \end{array}\tag{B.2}
$$

Zero-exposure entries are zero, and attention is detached before accumulation. Dividing by $C _ { i j }$ prevents frequently exposed pairs from receiving larger afinities merely because they occurred more often. The correction removes unequal exposure frequency, not dependence on the backbone or reveal-context distribution; annealing windows therefore reflect the current order mixture. These afinities remain model-induced relational signals rather than estimates of causal influence. The two $D \times D$ accumulators span the task-specific windows in Appendix C.3. Completed windows yield $W _ { i j } = \sqrt { A _ { i j } A _ { j i } }$ with zero diagonal and ${ \cal L } = \mathrm { d i a g } ( W { \bf 1 } ) - W ;$ spectral recovery and orientation follow Appendices $\mathsf { A } . 3$ and $\mathsf { A } . 4$

## B.2. Order-MLP Parameterization

The Spectral Teacher orients the Fiedler coordinates using Eq. (6) and normalizes them to [0, 1] using Eq. (A.26). The resulting target $\mathbf { q } ^ { * } = \mathbf { r }$ assigns larger priorities to variables generated earlier. The student receives directed $A ,$ with population z-normalization over all $D ^ { 2 }$ entries, including the

zero diagonal:

$$
\mu _ { A } = D ^ { - 2 } \sum _ { i , j } A _ { i j } , \quad \quad A ^ { \mathrm { s t d } } = ( A - \mu _ { A } ) / s _ { A } ,
$$

$$
s _ { A } = \operatorname* { m a x } \left( 1 0 ^ { - 6 } , \sqrt { D ^ { - 2 } \sum _ { i , j } ( A _ { i j } - \mu _ { A } ) ^ { 2 } } \right) .\tag{B.3}
$$

The flattened $A ^ { \mathsf { s t d } }$ passes through two fully connected GELU hidden layers of width 32 and a �-unit sigmoid output layer to produce priorities $\widehat { \mathbf { q } } _ { \phi }$ , giving the architecture $D ^ { 2 }  3 2  3 2  D$ for every task. Training and order extraction use

$$
\mathcal { L } _ { \mathrm { o r d e r } } = D ^ { - 1 } \sum _ { i } ( \widehat { q } _ { \phi , i } - q _ { i } ^ { * } ) ^ { 2 } , \qquad \widehat { \boldsymbol { \pi } } _ { \phi } = \mathrm { a r g s o r t } _ { \mathrm { d e s c e n d i n g } } ( \widehat { \mathbf { q } } _ { \phi } ) .\tag{B.4}
$$

The [0, 1] outputs are priorities, not calibrated probabilities, and continuous targets retain relative spectral spacing. Both � and $\mathbf { q } ^ { * }$ are detached, so MSE updates only the MLP; each window produces one task-level order shared across examples.

## B.3. Training Lifecycle

Table S2 reports three milestones: starting distillation, stopping distillation, and completing annealing. For the current learned order $\widehat { \pi } _ { s }$ available at update $s ,$

$$
q _ { s } ( \pi ) = ( 1 - p _ { s } ) \mathrm { U n i f } ( \mathfrak { S } _ { D } ) + p _ { s } \delta _ { \widehat { \pi } _ { s } } ( \pi ) ,\tag{B.5}
$$

where $p _ { s }$ gradually increases from zero to one during training. During distillation, the backbone trains with random reveal orders while the Spectral Teacher supervises the Order MLP using aggregated attention. After distillation, MLP parameters remain fixed, and learned-order use is gradually increased. Within the configured refresh window, the MLP proposes candidate orders from new attention estimates for subsequent backbone updates. After annealing completes, the final accepted permutation is reused for training and generation.

Generation with the final learned order requires neither MLP inference nor an eigensolve.

## C. Experimental Protocols

## C.1. Datasets and Representations

WikiText103 Seq80. The mechanism experiment uses raw WikiText103 (Merity et al., 2017), GPT-2 BPE, and an end-of-text token after each source record. Training and oficial validation streams use unsigned 16-bit IDs and 80-token windows; the test split is excluded. Each position is one variable, relabeled once with permutation seed 42. This experiment studies order recovery and distillation under a short context; it does not train the model used for the OpenWebText zero-shot benchmark.

OpenWebText. OpenWebText (Gokaslan et al., 2019) uses the same tokenizer and document-end convention, separate unsigned 16-bit streams, and a document-level validation fraction 0.0005 with split seed 2357. Training uses 1,024-token windows, a 50,257-symbol vocabulary padded to 50,304 outputs, and relabeling seed 42. The 400k budget counts repeated token presentations, not distinct corpus tokens.

Table S1 | Backbone settings: architecture denotes decoder layers / heads / hidden width, and batch counts examples per optimizer update.
<table><tr><td>Setting</td><td>Seq80</td><td>OpenWebText</td><td>H3WB</td><td>CENSUS</td></tr><tr><td>Ordered variables D</td><td>80</td><td>1,024</td><td>133</td><td>68</td></tr><tr><td>Decoder architecture</td><td>4/8/384</td><td>12/10/640</td><td> $4 / 8 / 2 5 6$ </td><td> $2 / 4 / 1 2 8$ </td></tr><tr><td>Global batch</td><td>512</td><td>480</td><td>160</td><td>512</td></tr><tr><td>Backbone updates</td><td>50,000</td><td>400,000</td><td> $^ { 1 0 0 , 0 0 0 }$ </td><td> $5 0 { , } 0 0 0$ </td></tr><tr><td>Peak learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Minimum learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $6 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warm-up updates</td><td>0</td><td>2,000</td><td>5,000</td><td>1,000</td></tr></table>

H3WB masked-I2D. Each of 133 three-dimensional joint targets is conditioned on two-dimensional keypoints and visibility indicators (Zhu et al., 2023). Targets are pelvis-relative; observed coordinates retain oficial image coordinates before normalization. Oficial training sequences are partitioned by group ID with seed 1337, targeting fractions 0.1 for validation and 0.1 for discovery; preserving groups permits small deviations from those fractions. Training-partition coordinate means and standard deviations are fixed for validation and the released 10,000-pose test set. The ordinary-training window protocol collects attention from training batches; the existence of a separate prepared discovery partition does not mean this protocol draws its attention windows from that partition. Corruption mixes independent masking (weight 0.4, per-keypoint probability 0.25) with whole-face, left-hand, and right-hand masks (weight 0.2 each).

CENSUS. The processed 1990 U.S. Census table (Meek et al., 2001) has 2,458,285 rows and 68 categorical variables after removing the case identifier. Each column receives a fixed sorted-unique vocabulary of dense zero-based IDs. The already randomized table is split contiguously into 2,212,456 training, 122,914 validation, and 122,915 test rows. Semantic names do not enter order learning. Representative permuted H3WB/CENSUS runs use relabeling seed $0 ;$ canonical baselines retain source indexing. The source-domain category vocabulary is fixed during preparation, and mapping predictions back to original IDs is an evaluation operation.

## C.2. Backbones and Optimization

Table S1 specifies backbone sizes and schedules. All use zero dropout, AdamW weight decay 0.1 on eligible parameters, gradient clipping 1.0, and cosine learning-rate decay after warm-up. Adam coefficients are (0.9, 0.99) for Seq80 and (0.9, 0.95) otherwise. Language decoders use learned variableposition embeddings and target-position conditioning. H3WB adds one bidirectional conditionencoder layer and diagonal-Gaussian joint outputs with log scale clamped to [−7, 5]; CENSUS uses column-specific categorical outputs.

The OpenWebText backbone has 130,336,512 parameters. The OpenWebText OrderGPT and randombaseline runs use six RTX 5880 Ada Generation GPUs. Both use a global batch of 480, totaling 196.608 billion token presentations over 400k updates. All protocols use BF16, with compiled execution for language and CENSUS.

All tasks use the Order-MLP architecture described in Appendix B.2. Order MLP uses AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay 0.01, gradient clipping 1.0, and zero dropout. Adam coeficients are (0.9, 0.99) for language and (0.9, 0.999) for H3WB/CENSUS. Its input / hidden / output widths are $6 , 4 0 0 \to 3 2 \to 3 2 \to 8 0 ( { \mathrm { S e q } } 8 0 ) , 1 , 0 4 8 , 5 7 6 \to 3 2 \to 3 2 \to 1 , 0 2 4$ (400k OpenWebText), 17,689 → 32 → 32 → 133 (H3WB), and 4,624 → 32 → 32 → 68 (CENSUS).

Table S2 | Attention windows and three order-learning milestones.
<table><tr><td>Setting</td><td>Seq80</td><td>OpenWebText</td><td>H3WB</td><td>CENSUS</td></tr><tr><td>Examples per attention window</td><td>1,024</td><td>2,400</td><td>1,280</td><td>1,024</td></tr><tr><td>Updates per window</td><td>2</td><td>5</td><td>8</td><td>2</td></tr><tr><td>Start distillation</td><td>10k</td><td>4k</td><td>15k</td><td>5k</td></tr><tr><td>Stop distillation</td><td>18k</td><td>40k</td><td>40k</td><td>10k</td></tr><tr><td>Annealing complete</td><td>32k</td><td>60k</td><td>60k</td><td>20k</td></tr></table>

## C.3. Attention Windows and Order-Learning Milestones

Table S2 lists the attention windows and the three order-learning milestones in backbone optimizer updates. Window sizes count examples across optimizer updates, rather than micro-batches. All Spectral Teacher and OrderGPT experiments use the attention-based orientation rule in Eq. (6).

## C.4. Baselines and Comparison Scope

Canonical AR uses L2R in language and source indexing in H3WB/CENSUS; Fixed-random AR (throughout) uses one presampled permutation for training and inference; AO-GPT samples a fresh permutation per training example. Random and canonical language inference use the same AO-GPT checkpoint. Spectral Teacher uses the attention-oriented spectral candidate, whereas OrderGPT sorts the MLP priorities. The two 400k OpenWebText fixed-random controls share their terminal permutation and coordinate mapping. Fixed-random AR + curriculum uses the same order-use probability $p _ { s }$ as OrderGPT in Eq. (B.5), with one presampled random permutation replacing the learned order. This control tests whether the same schedule with an arbitrary terminal order reproduces the gains of OrderGPT. Within each domain, internal comparisons match the backbone capacity and data processing as well as the backbone-update budget. Matched backbone updates and token presentations do not imply identical hardware, elapsed time, or total compute including order discovery and MLP training. Internal OpenWebText runs omit language-model EMA and prescribed L2R training mixtures; published AO-GPT rows retain those protocol diferences (Xue et al., 2025) and serve as external context.

## C.5. Evaluation and Statistical Units

Language. Perplexity is exp $\left. ( \sum _ { n } \ell _ { n } ) / N _ { \mathrm { t o k e n s } } \right.$ for natural-log token losses. The fixed-record evaluator maps 1,024-token benchmark records into checkpoint current IDs, scores all targets including the first, and accumulates losses in float64. Inference uses random, original-L2R, or frozen learned order as specified, without benchmark-side order discovery. For multiple random orders, losses are averaged before exponentiation, not perplexities afterward. LAMBADA reports token perplexity, not last-word accuracy; Seq80 validation is a separate short-context experiment.

H3WB. Teacher-forced NLL is averaged per three-dimensional joint; closed-loop evaluation generates all 133 joints by successive Gaussian means and reports MPJPE in millimeters. Whole/Body use pelvis alignment, Face-local uses nose alignment, and Hand-local uses the corresponding wrist. The full 10,000-pose test protocol uses BF16, batch 256, and evaluation-order seed 20260713. Checkpoint selection uses the configured validation criterion after the method’s selection boundary, which is 60k for OrderGPT. Teacher-forced NLL and free-running MPJPE measure diferent outcomes.

CENSUS. Test NLL is averaged per variable on the held-out slice. Cardinality evaluation uses 1,000 shared queries with 5–11 predicates (seed 1234), exact counts on all 2,458,285 rows, and 10,000 particles per query. Predicate IDs are remapped for permuted checkpoints without changing query meaning. Fixed-order models use their saved permutation; the random-order model draws one per particle and keeps it for all 68 steps. Estimated cardinality is the ceiling of mean particle weight times table size. For positive counts, Q-error is max $( \widehat { c } / c , c / \widehat { c } )$ ; P50, P99, and maximum are the workload median, linearly interpolated 0.99 quantile, and maximum. Published references retain their original protocols (Yang et al., 2019; Liang et al., 2020).

Replicates. Main-table H3WB/CENSUS entries report means and sample standard deviations across independently trained models, with replicate counts in the captions. Q-error percentiles and maxima are computed within each run before aggregation. Relabeling, reveal-order, and particle-sampling randomness do not constitute additional training replicates. Appendix curves and order visualizations show individual runs and provide no across-run uncertainty estimate or significance test.

Fixed-checkpoint eficiency. Seq80 eficiency is measured by replaying an 18k-update checkpoint on one RTX 4090 with BF16 and compiled execution. Both methods aggregate 1,024 examples and refresh the order every two backbone updates. Timing uses synchronized measurements over 200 updates after a 100-update warm-up, with paired random baselines. These measurements concern per-update execution rather than complete-training wall time.

## D. Additional Results and Analyses

## D.1. Training Dynamics

Figure S1 shows training dynamics as learned-order use increases. Order MLP reaches a lower final training loss than the random-order and fixed-random baselines on OpenWebText. Its validation trajectory separates from the random-order baseline on H3WB and CENSUS.

## D.2. Structure of the Learned H3WB Order

The order in Figure S2 places all 68 facial joints first, followed by 22 of the 23 body-and-foot joints and then the hands, with the remaining body joint interleaved in the left-hand span. This coarse face-to-body-to-hands traversal difers from the dataset’s body/feet-to-face-to-hands indexing, while the interleaving shows that the learned order is not a strict concatenation of anatomical groups.

## D.3. CENSUS Locality and Representative Neighborhoods

For rank $r _ { \pi } ( i )$ of variable � and a fixed nonnegative symmetric afinity matrix � with positive ofdiagonal weight, define

$$
d _ { W } ( \pi ) = \frac { \sum _ { i < j } W _ { i j } | r _ { \pi } ( i ) - r _ { \pi } ( j ) | } { \sum _ { i < j } W _ { i j } } .\tag{D.1}
$$

Under a uniform random permutation independent of $W ,$ the expected distance is $( D + 1 ) / 3 ,$ giving 23 for the 68 CENSUS variables.

The representative audit constructs � from a frozen random-order AO-GPT checkpoint on 2,048 training examples, averaging its two layers and four heads with exposure correction. After aligning variable identities, the same matrix is evaluated under canonical indexing and the saved, frozen Order-MLP permutation, without recomputing attention under that permutation. Weighted separations are 17.91 for the learned order and 23.17 for canonical indexing, compared with the random expectation of 23.00. The graph and saved permutation come from diferent checkpoints but use the same task and training split, so this comparison is a structural diagnostic rather than an independent test of generalization.

![](images/cb2a18d65ba4536bd59ab1b4128ac680415eb770c050ec7f0268c6514ea94610.jpg)  
(a)

![](images/fd01054759d52fb72542b863d940ecb29cab53ba17aea5057b4ab60ead44c86e.jpg)  
(b)

![](images/2041b2f48c3545072027785f879db17448ce7b94dce4cb18d380ff7b7ac4e193.jpg)  
(c)  
Guides: start of distillation | start of annealing | fixed learned-order use Order MLP — - Canonical AR - v Fixed-random AR + curriculum - AO-GPT (random)  - Fixed-random AR (throughout)  
Figure S1 | Loss trajectories: (a) OpenWebText training token NLL ; (b) H3WB validation NLL per joint; (c) CENSUS validation NLL per variable. Vertical guides mark curriculum transitions. The OpenWebText Fixed-random AR + curriculum curve follows the same schedule as OrderGPT. Internal curves represent individual runs and are averaged or smoothed for display.

Table S3 shows a service-and-military prefix and shorter family neighborhoods, alongside interleaved mobility and family fields in the occupation and language windows. The income pair remains 24 positions apart: improved global locality does not force every related pair to be adjacent.

(a) H3WB reveal-rank map  
![](images/c4334bbf2a0106c425226a93553058140af79302ab55ac63be8fa3491fc8a3d0.jpg)

![](images/d0381245653c780d66c3b74b71943a3ce6c2e60bafaaab07d476cde408f8d275.jpg)  
Figure S2 | Post-hoc interpretation of one learned H3WB order. (a) Reveal rank on a representative training-set 2D pose, with face and hand enlargements; darker points appear earlier. (b) Anatomical regions along canonical and learned orders. The pose and connecting lines provide display context, not an input-specific prediction, learned dependency edges, or the evaluation bone set. The pose, canonical identities, and anatomical labels do not enter order learning, direction selection, or model selection.

Table S3 | Representative CENSUS neighborhoods in the audited order; positions are one-based. Column names and semantic groups are post-hoc annotations unavailable to the order learner, and the windows need not form a disjoint partition.
<table><tr><td>Neighborhood or check</td><td>Observed fields and positions</td></tr><tr><td>Service/military, 1-10</td><td>i0thrserv, iMay75880, iKorean, iWWII, iRvetserv, dYrsserv, iSept80, iFeb55, iVietnam, iMilitary.</td></tr><tr><td>Family, 13–14, 21–22, 28-29</td><td>iRelat1-iFertil; iRownchld-iMarital; iRspouse-iRrelchld.</td></tr><tr><td></td><td>Occupation/labor, 31-39 d0ccup, dHour89, dIndustry, iMobility, iTmpabsnt, dDepart, dHours,</td></tr><tr><td>Language/origin, 57–62 </td><td>iYearwrk, iRemplpar. dAncstry2, iCitizen, dAncstry1, iSubfam1, iEnglish, dPOB.</td></tr><tr><td>Pairwise distances</td><td>iRvetserv-dYrsserv: 1; d0ccup-dIndustry: 2; dIncome1-dRearning: 24.</td></tr></table>

## E. Extended Related Work

Masked difusion and any-order autoregression. Deep NADE shares parameters across autoregressive decompositions with diferent variable orders (Uria et al., 2016). AO-GPT connects maskeddifusion and any-order objectives while separating their formulation from a decoder-only implementation (Xue et al., 2025). The connection matches distributions over context sizes, observed subsets, and prediction targets; it does not equate conditional parameterizations or sampling trajectories. A uniform permutation and uniformly sampled prefix length induce a uniform observed subset at each size and a uniform next target among unrevealed variables. Matching these weights explains the objective correspondence even when the decoder retains the prefix’s presentation order. OrderGPT further selects one reusable task-level factorization instead of continuing to average training over all orders.

Adaptive paths and learned policies. Kim et al. (2025) establish dificult infilling problems in constructed masked-difusion settings and show that adaptive decoding can avoid some of them. Their conditional-inference complexity result is distinct from our shared-output-feature capacity separation. Path Planning selects state-dependent denoising updates, including revisions to previously unmasked variables (Peng et al., 2025). LO-ARM jointly learns a state-dependent probabilistic autoregressive order policy through a variational likelihood objective (Wang et al., 2025). OrderGPT instead distills shared dependency structure into a fixed permutation, eliminating generation-time policy decisions while sacrificing per-example adaptation. Our experiments do not compare its performance against adaptive planners.

Attention and spectral structure. Afinity from Attention learns semantic afinities from Transformer attention using segmentation pseudo labels (Ru et al., 2022). OrderGPT instead aligns reveal contexts, corrects pair exposure, and combines reciprocal weights without semantic-afinity or canonicalorder supervision. The resulting graph remains conditioned on the trained backbone and the reveal distribution used for aggregation. Spectral seriation reconstructs linear arrangements from structured similarities (Atkins et al., 1998); its connection to 2-SUM motivates the locality objective (Fogel et al., 2015). These results do not establish that arbitrary Transformer attention meets the required structure or that spectral sorting minimizes NLL. We therefore treat attention-graph alignment as an explicit condition, separate from the capacity theorem, and interpret directed-attention orientation as an endpoint preference rather than causal identification.