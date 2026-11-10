# The balance between feature learning and collapse in dynamical systems-based generative models

Julian Brandon Bruno Loureiro N Alex Cayco-Gajic<sup>∗</sup> Arthur Pellegrino<sup>∗</sup>

Ecole Normale Supérieure, PSL

Correspondence: julian.brandon@ens.psl.eu

## Abstract

Over the past decade, a growing body of work has established that implicit biases in learning dynamics fundamentally shape the solutions found by neural networks, governing their learned features and generalisation performance. However, despite these strides in deep learning theory, far less is known about representation learning in generative models based on dynamical systems, from normalising flows to diffusion. Importantly, empirical evidence suggests that continuous-time generative models can be prone to mode collapse, yet a learning theory explaining such biases remains elusive. To address this, we use tools from dynamical systems and operator theory to express the gradient flow of a general class of stochastic differential equations as a composition of linear operators, allowing us to relate collapse to low-rank biases in generative models. In variational inference and flow-matching tasks, we show that this bias can manifest in two different mechanisms: i) where collapsed models are stable fixed points of the gradient flow, or ii) in sequential feature learning, where smaller modes take progressively longer to be learned, producing an effective collapse for a fixed training budget. Together these findings point to a trade-off between feature learning and collapse, suggesting an operator view as a step towards a unified framework for understanding the learning dynamics of dynamical systems-based generative models.

## 1 Introduction

Neural networks exhibit learning phenomena that are not predicted by classical optimisation theory developed for convex problems<sup>1,</sup> <sup>2</sup>. This observation has led to the development of new theoretical approaches aimed at studying the learning dynamics of neural networks<sup>3</sup>. This perspective makes it possible to apply tools from dynamical systems theory to analyse the rich behaviours observed during training<sup>4,</sup> <sup>5</sup>. For example, gradient flow analyses on idealised settings provide information about stability of fixed points<sup>6,</sup> <sup>7</sup>, and different forms of implicit regularisation<sup>8,</sup> <sup>9</sup>. Despite being idealised, several phenomena observed in large-scale training can be explained in this framework: staircase-like patterns in the loss curve can correspond to trajectories passing near saddle points, reflecting the sequential learning of data features<sup>4,</sup> <sup>10–12</sup>; and slow changes in parameters can occur even when the loss appears constant, corresponding to a “silent” alignment of the network weights with relevant directions in the data<sup>13</sup>. Together, these illustrate that dynamical systems provide a powerful framework to investigate the rich behaviours observed in neural network training.

Despite this success, learning theory is currently under-developed for generative models based on differential equations. Yet, such dynamical systems are at the centre of flow and diffusion models, which are the state-of-the-art for generative models in vision<sup>14,</sup> <sup>15</sup>, biology<sup>16,</sup> <sup>17</sup> and physics<sup>18,</sup> <sup>19</sup>.

Figure 1: Low-rank biases in learning can drive mode collapse in generative models. Recent empirical evidence has suggested that generative models can exhibit learning biases such as memorisation of data samples (left) or mode collapse (right). Here, we argue that the latter can be caused by low-rank gradients biasing the learning dynamics.

![](images/85010da946f86974c975edfcf8ce33ceabeda76bc15915ae53363680c3266556.jpg)

The need for these new mathematical tools is reinforced by the presence of implicit biases in the distributions learned by generative models. Implicit biases emerge through training, limiting which solutions are learned through optimisation. For example, even when a generative model is expressive enough to learn the full data distribution, training may favour solutions that capture only a subset of the distribution’s modes<sup>20–25</sup>, a phenomenon known as “mode collapse”. An immediate adverse consequence of mode collapse is a reduction in the diversity of generated samples, where regions of the data distribution may be ignored. More broadly, the loss of sample diversity could exacerbate more pernicious forms of degeneration if models are recursively trained on synthetic data<sup>26</sup>.

Implicit biases have been extensively characterised in the learning dynamics of deep networks. For example, an emerging line of work in learning theory has shown that deep neural networks tend to exhibit simplicity biases, where learning favours low-rank<sup>27</sup> or low-frequency components of the data<sup>28</sup>. More recently, key advances in dynamical systems learning theory stemming from functional analysis and statistical physics have enabled characterising similar biases in the learning dynamics of dynamical systems<sup>29–31</sup>

Despite these results, the relationship between implicit biases induced by training and mode collapse in generative models remains unclear. We hypothesise that while implicit biases drive deep networks to learn low-dimensional data features, which is a beneficial form of representation learning, they are the same mechanism underlying mode collapse in dynamical systems-based generative models.

To support this hypothesis, we employ mathematical tools stemming from dynamical systems and operator theory to gain insight into collapse in two different model classes. We show that low-rank biases can arise in generative models as a direct consequence of low-rank operators constraining the rank of the gradient, which can induce collapse.

To validate this generally applicable theoretical framework in a controlled setting, we focus on an analytical study of simplified linear generative dynamical systems, allowing us to isolate the mechanisms by which the loss shapes the learning dynamics. We show that distinct objectives induce low-rank solutions through different architectural and data-driven effects. For variational inference, we find that mode collapse can arise from a rank bottleneck in the operators defining the gradient, which yields a stable collapsed fixed point in the gradient flow. In contrast, flowmatching objectives avoid stable sub-optimal solutions, but instead exhibit an effective mode collapse: learning proceeds through saddle-to-saddle transitions, leading to sequential acquisition of modes with progressively longer learning times. Finally, we show similar phenomena in non-linear generative models, demonstrating that the same qualitative mechanisms persist in practical settings.

Overall, our results reveal low-rank bias as a driver of different types of collapse, highlighting an operator framework as a key tool in understanding the learning dynamics of dynamical systems-based generative models.

## Contributions

• We apply an operator decomposition to a general class of models (SDEs), allowing us to derive a mathematical bound on the rank ofthe gradientfor dynamical systems based generative models.

• We analytically characterise mode collapse in linear continuous normalising flows trained with the reverse Kullback-Leibler (KL) objective, showing that collapse arises as a stable fixed point ofthe gradient. Collapsed weights have lower effective rank than non-collapsed ones.

• We show that while flow-matching objectives avoid stable collapsedfixed points, linear models can exhibit an “effective collapse” due to saddle-to-saddle dynamics induced by data imbalance constraining the rank of the operators.

• We validate our results in non-linearflow-matching models with imbalanced modes, showing that sequential learning ofmodes driving effective collapse persists in more practical settings, and that operator ranks provide a diagnosticfor the learning time.

## Related works

Learning theory and implicit biases. Prior work has shown that gradient flow tends to implicitly favour low-complexity solutions in overparameterised networks, including minimum-norm solutions<sup>32–34</sup>, maximum-margin classifiers<sup>8</sup>, algorithmically simple functions<sup>35,</sup> <sup>36</sup>, low-frequency modes<sup>28,</sup> <sup>37</sup>, and low-rank embeddings<sup>27,</sup> <sup>32,</sup> <sup>38</sup>. Low-rank biases in particular have been studied extensively in deep linear networks, generally with small weight initialisations, where it has been shown that the data singular values are learned one by one<sup>3</sup>. This sequential feature learning produces plateaus in the loss referred to as “saddle-to-saddle” dynamics, where each saddle corresponds to an increase in the rank of the weights<sup>4,</sup> <sup>10,</sup> <sup>12,</sup> <sup>39,</sup> <sup>40</sup>. Here we argue that similar saddle-to-saddle learning dynamics unfold during flow matching, and that increased learning time between each saddle drives an effective collapse. While the learning dynamics of generative models are less studied overall, linear diffusion has been shown to produce distributional simplicity biases<sup>31,</sup> <sup>41,</sup> <sup>42</sup>. In particular, linear denoisers have been shown to sequentially learn eigenmodes of the input covariance<sup>31</sup>. Unlike these works however, here we focus on understanding these biases in models trained on multi-moda distributions, especially under data imbalance.

Operator view of learning dynamics. Recent work has suggested that low-rank biases also emerge in recurrent architectures by using adjoint dynamics to represent gradient flow as a composition of linear operators<sup>29,</sup> <sup>30</sup>. In particular, one study derived bounds on the rank of weight updates in linear recurrent networks using this operator view<sup>29</sup>. Koopman operator theory has also been applied to diffusion to interpret the sampling process<sup>43,</sup> <sup>44</sup>, but not training dynamics. Our work applies a similar operator perspective to the learning dynamics of continuous-time generative models, and in particular applies it to mode collapse.

Collapse in generative models. Generative models are known to exhibit mode collapse, in which the full diversity of modes in the data distribution are not captured<sup>45–49</sup>. Collapse can come from many sources: for example, distributional biases can cause mode collapse when there is class imbalance due to long-tailed distributions<sup>23,</sup> <sup>50,</sup> <sup>51</sup>. In more extreme cases, e.g., when generative models are used to train other generative models, this can lead to model collapse<sup>26</sup>. Architectural biases due to the network parametrisation of diffusion models can in some cases promote strong generalisation<sup>52</sup>, or strong collapse<sup>53</sup>. Mode collapse can also come from the objective, for example recently studied in variational inference<sup>6,</sup> <sup>54,</sup> <sup>55</sup>. Previous work has noted a balance between generalisation and memorisation in flow matching and diffusion<sup>56–60</sup>. Our results tie this trade-off to the rank of weight parameters and a balance between generalisation and collapse.

## 2 Theory

Problem setup. The results in this section relate to a general class of dynamical systems-based generative models:

$$
\mathbf { d x } ( t ) = \mathbf { f } _ { \theta } ( \mathbf { x } ( t ) , t ) d t + G _ { \theta } ( \mathbf { x } ( t ) , t ) \circ \mathbf { d B } , \quad \mathbf { x } ( 0 ) \sim p _ { 0 }\tag{1}
$$

where ${ \bf x } ( t )$ is the state of the system, dB increments of a Brownian motion, $\mathbf { f } _ { \pmb { \theta } } : \mathbb { R } ^ { d } \times \mathbb { R }  \mathbb { R } ^ { d }$ and $G _ { \pmb { \theta } } : \mathbb { R } ^ { d } \times \mathbb { R }  \mathbb { R } ^ { d \times d }$ the drift and diffusion components of the stochastic differential equation with parameters $\pmb \theta \in \Theta , ( \circ )$ denotes the Stratonovich integral, and $p _ { 0 }$ is a chosen base distribution. We assume a general loss function:

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { B } } \left[ \int _ { 0 } ^ { T } \ell ( \mathbf { x } ( t ) , t ) d t + \Phi ( \mathbf { x } ( T ) ) \right] ,\tag{2}
$$

where $\ell : \mathbb { R } ^ { d } \times [ 0 , T ] $ R is the instantaneous loss and $\Phi : \mathbb { R } ^ { d } $ R is the terminal loss which only depends on $\pmb \theta$ through $\mathbf { x } ( T )$ . Depending on the form of $\ell ( \cdot )$ and Φ(·), this setting includes variational, flow matching and score-matching objectives as special cases.

Here, we are interested in the case where $\pmb \theta$ contains weights $W \in \mathbb { R } ^ { k \times q }$ which act linearly on activations, parametrising the drift and diffusion terms $\mathbf { f } _ { \theta }$ and $G _ { \theta } .$ , or projecting to the instantaneous loss. In particular, the rank of $W$ can constrain the trajectories of the dynamical system and, therefore, the probability distribution learned by the generative model. As we argue below, implicit low-rank biases can arise in the gradient flow learning dynamics, which can in turn bias the model towards collapsing certain modes of the learned distribution. To analyse this phenomenon, we study gradient flow by decomposing it into two linear operators.

An operator view of gradient flow. We are interested in the learning dynamics of generative models under gradient flow. Using adjoint dynamics, we show that, similarly to the neural tangent kernel of deep networks, the gradient of weight parameters of dynamical systems can be written as a composition of linear operators. In appendix A we provide a thorough derivation of these results along with illustrative examples. In summary:

Theorem 2.1. Consider a model as defined in Eq. 1 with loss as defined in $E q . 2 .$ . For any parameter $W \in \mathbb { R } ^ { k \times q } \subseteq \pmb { \theta }$ which acts linearly on activations $\mathbf { h } _ { ( \cdot ) } ( t ) \in \mathbb { R } ^ { q } \nu i a \mathbf { z } _ { ( \cdot ) } ( t ) = \mathbf { \bar { W } } \mathbf { h } _ { ( \cdot ) } ( t ) \in \mathbb { R } ^ { k }$ , where (·) denotes a projection to either the instantaneous loss, drift or diffusion terms, the gradient of L with respect to W can be written as the composition oftwo linear operators:

$$
\nabla _ { W } \mathcal { L } = A _ { ( \cdot ) } \circ H _ { ( \cdot ) } , \qquad A _ { ( \cdot ) } : L ^ { 2 } ( \mathbb { R } ) \to \mathbb { R } ^ { k } , \qquad H _ { ( \cdot ) } : \mathbb { R } ^ { q } \to L ^ { 2 } ( \mathbb { R } )
$$

where the linear operators act via integration:

$$
A _ { ( \cdot ) } ( f ) = \mathbb { E } _ { \mathbf { x _ { 0 } } , \mathbf { B } } \left[ \int _ { 0 } ^ { T } \mathbf { a } _ { ( \cdot ) } ( t ) f ( t ) d t \right] , \qquad ( H _ { ( \cdot ) } ( \mathbf { v } ) ) ( t ) = \mathbf { h } _ { ( \cdot ) } ( t ) \cdot \mathbf { v }
$$

where $f \in L ^ { 2 } ( \mathbb { R } ) , \mathbf { v } \in \mathbb { R } ^ { q }$ , and $\mathbf { a } _ { ( \cdot ) } ( t ) \in \mathbb { R } ^ { k }$ . The specific form of the adjoint $\mathbf { a } _ { ( \cdot ) } ( t )$ depends on the projection $\left( \cdot \right) \left( A p p . A . l \right)$ [PROOF]

Thus the gradient with respect to $W$ can be expressed in terms of two operators, an adjoint operator A and a state operator H, which compose as an integral over rank-1 terms.

Low-rank biases. Key results from functional analysis<sup>61</sup> say that A and H are compact, and admit singular value decompositions:

$$
A = U ^ { a } \circ S ^ { a } \circ V ^ { a } , \qquad H = U ^ { h } \circ S ^ { h } \circ V ^ { h }
$$

Furthermore, because they are maps between or from a finite-dimensional vector space, they have finite rank (i.e. finitely many non-zero diagonal entries of $S ^ { a }$ and $S ^ { h } )$ . It immediately follows that the singular values of their composition (i.e. of the gradient) are directly determined by i) the overlap between the right singular vectors $V ^ { a }$ and the left singular vectors $\check { U } ^ { h }$ and ii) the singular values $S ^ { a } , S ^ { h }$

For example, usual bounds on the rank of matrix products also hold for linear operators, and the rank of the gradient of the weights can be bounded as: Rank $( \nabla _ { W } \mathcal { L } ) \le$ min{Rank(A), Rank(H)}. In practical settings, however, these operators will be full rank, so we may be more interested in a numerically stable definition of the effective rank.

Definition 2.2. The stable rank of a compact bounded linear operator A is defined as:

$$
\operatorname { S R a n k } ( A ) : = { \frac { \| A \| _ { * } } { \| A \| _ { 2 } } } = \sum _ { i = 1 } ^ { m } { \frac { s _ { i } ^ { a } } { s _ { 1 } ^ { a } } }
$$

where $\| \cdot \|$ <sub>∗</sub> denotes the nuclear norm and $\| \cdot \| _ { 2 }$ the spectral norm and $s _ { i } ^ { a }$ the singular values of A.

In particular, if all nonzero singular values are equal, the stable rank coincides with the rank. Furthermore, because it is only normalised by the top singular value, the stable rank is less sensitive to the tails of the singular value distribution than other smooth ranks, such as the participation ratio.

Theorem 2.3. The stable rank ofthe gradient can be bounded as:

$$
\mathrm { S R a n k } ( \nabla _ { W } \mathcal { L } ) \leq \frac { \| A \| _ { 2 } \| H \| _ { 2 } } { \| A \circ H \| _ { 2 } } \operatorname* { m i n } \left\{ \mathrm { S R a n k } ( A ) , \mathrm { S R a n k } ( H ) \right\} .
$$

where A and H are the operators defined above.

[PROOF]

The prefactor depends on the alignment between the right and left singular vectors of A and H respectively, equalling 1 when they are aligned and bounded above by the top singular vectors according to $| \langle \mathbf { \bar { v } } _ { 1 } ^ { A } , \mathbf { u } _ { 1 } ^ { H } \rangle | ^ { - 1 } \ ( \mathrm { A p p . \ A . } 2 )$

Thus, in general, the weights will be biased towards lower-rank solutions ifeither the A or H operator is low rank. Such low-rank structure can arise through various mechanisms, including properties of the loss function or constraints imposed by the model architecture. In subsequent sections, we show that this bias can cause dynamical systems-based generative models to collapse certain data modes, either by converging to stable but suboptimal gradientfixed points during variational inference, or by exhibiting slow learning due to saddle-to-saddle dynamics in flow matching.

## 3 Results

## 3.1 Mode collapse as an attractor in learning dynamics under a variational objective

To understand how learning biases drive collapse, we first consider a variational inference task, for which generative models are known to have a mode-seeking behaviour. Let $p _ { \theta } ( \mathbf { x } )$ denote a parametric family of distributions, whose parameters $\theta$ are optimized to fit a target distribution $p ^ { * } ( \mathbf { x } )$ by minimizing the ‘reverse’ Kullback Leibler divergence:

$$
D _ { \mathrm { K L } } ( p _ { \theta } ( \mathbf { x } ) \parallel p ^ { * } ( \mathbf { x } ) ) = \mathbb { E } _ { p _ { \theta } } \left[ \log p _ { \theta } ( \mathbf { x } ) \right] - \mathbb { E } _ { p _ { \theta } } \left[ \log p ^ { * } ( \mathbf { x } ) \right] .
$$

The reverse KL is commonly used to train neural samplers when only an unnormalised density or an energy function is available<sup>45,</sup> <sup>49</sup>. Due to its asymmetric form, the reverse KL is intrinsically ‘mode-seeking’, as regions in the data distribution are ignored where $p _ { \theta } ( \mathbf { x } ) = 0$ . This provides an ideal case study to analytically study how the interplay between the state and adjoint operators leads to a low-rank bottleneck in the weight updates. As we will demonstrate, this results in collapse becoming an attractor of the gradient flow.

Model. For analytical tractability, we consider a linear continuous normalisingflow (CNF):

$$
\begin{array} { r } { \frac { d } { d t } { \bf x } ( t ) = W { \bf x } ( t ) , \qquad { \bf x } ( 0 ) \sim p _ { 0 } } \end{array}
$$

where $p _ { 0 }$ is the base distribution which can be sampled from. The model is trained on a variational objective under gradient flow:

$$
{ \mathcal { L } } ( W ) = D _ { \mathrm { K L } } ( p _ { 1 } ( \mathbf { x } ) \parallel p ^ { * } ( \mathbf { x } ) ) , \qquad { \textstyle { \frac { d } { d \tau } } } W ( \tau ) = - \nabla _ { W } { \mathcal { L } } ( W ( \tau ) )
$$

where $\tau$ is the training iteration and $p _ { 1 } ( \mathbf { x } )$ is the distribution obtained by pushing forward samples from $p _ { 0 }$ to time $t = 1 ,$ . In the notation of Eq. 1 and Eq. 2, this model has $G = 0$ , and ℓ independent of $\mathbf { x } ,$ so the adjoint is driven only by the drift term $\mathbf { f } _ { \pmb { \theta } }$ . The operators can be found using the adjoint method<sup>62</sup>.

Definition 3.1. For a dynamical system $\dot { { \mathbf x } } = { \mathbf f } _ { \theta } ( { \mathbf x } ( t ) , t )$ defined on $t \in [ 0 , 1 ]$ , with loss functional $\mathcal { L } : \mathbb { R } ^ { d }  \mathbb { R }$ , the state adjoint is defined as $\begin{array} { r } { \mathbf { a } ( t ) = \frac { d \mathcal { L } ( \mathbf { x } ( 1 ) ) } { d \mathbf { x } ( t ) } } \end{array}$ . Its solution is given by the following dynamical system and terminal condition depending on J<sub>f</sub> the Jacobian of $\mathbf { f } _ { \theta }$

$$
\begin{array} { r } { \frac { d } { d t } \mathbf { a } = - J _ { \mathbf { f } } ^ { \top } \mathbf { a } , \qquad \mathbf { a } ( 1 ) = \frac { d \mathcal { L } } { d \mathbf { x } ( 1 ) } } \end{array}
$$

For the linear model and variational loss above, the adjoint dynamical system is therefore given by:

$$
\begin{array} { r } { \frac { d } { d t } { \bf a } = - W ^ { \top } { \bf a } , \qquad { \bf a } ( 1 ) = - \nabla _ { \bf x } \log p ^ { \star } ( { \bf x } ( 1 ) ) } \end{array}
$$

Since there is no hidden layer, the state operator H is simply the transported data x(t). We can then analytically derive the gradient as a Fréchet derivative of the matrix exponential of $\dot { W } ^ { \top }$

Theorem 3.2. The weight gradient is $\nabla _ { W } \mathcal { L } = A \circ H - I _ { d } ,$ , where the first term simplifies to:

$$
A \circ H = \mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { 0 } } \left[ \int _ { 0 } ^ { 1 } \exp { \left( W ^ { \top } ( 1 - t ) \right) } ( \mathbf { a } ( 1 ) \otimes \mathbf { x } ( 0 ) ) \exp { \left( W ^ { \top } t \right) } d t \right]\tag{[PROOF]}
$$

Task. In this linear model, it is not possible for the flow to split an initial mode $( \mathrm { i . e . }$ , from Gaussian p ) to multiple target modes. Instead, we constrain both the initial and data distributions to be equally weighted Gaussian mixtures, with $p _ { 0 }$ and $p ^ { \star }$ containing an identical number of modes. Because the linear flow has no bias term, it can only transport mass about the origin. We therefore also centre both mixtures, which leaves all inter-mode distances unchanged and is equivalent to an affine flow whose translation is learned trivially. The initial and target distributions are then defined as:

$$
p _ { 0 } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { m } \frac { 1 } { m } \mathcal { N } ( \mathbf { x } ; \mu _ { i } , \sigma ^ { 2 } I _ { d } ) , \quad p ^ { * } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { m } \frac { 1 } { m } \mathcal { N } ( \mathbf { x } ; \mu _ { i } ^ { * } , \sigma ^ { 2 } I _ { d } ) , \quad \mu _ { i } \cdot \mu _ { j } ^ { * } = 0 , \quad \bar { \mu } = \bar { \mu } ^ { * } = \mathbf { 0 }
$$

where $\begin{array} { r } { \bar { \pmb { \mu } } = \sum _ { i } \pmb { \mu } _ { i } , \bar { \pmb { \mu } } ^ { * } = \sum _ { i } \pmb { \mu } _ { i } ^ { * } } \end{array}$ . By constructing the initial and target modes symmetrically, we also ensure the bound prefactor in Theorem 2.3 is 1 at initialisation (App. B.4). Under these assumptions, there exists a linear transport between $p _ { 0 }$ and $p ^ { \star }$ by rotating the initial subspace to the data subspace (Fig. 2a), meaning that any collapse is due to implicit bias rather than expressivity. We found that training the linear CNF with different weight initialisation seeds made it converge to different values of $D _ { \mathrm { K L } }$ , corresponding to different numbers of learned data modes (Fig. 2b).

![](images/6a9ac2abb485aedccfe5202a2095c10659ca4b46aae15d6deac64ff21f81fa55.jpg)

b  
![](images/618c34e68b601f75701d9a902da2295e255ade13d0dd320807604a7f9a0aaecf.jpg)  
e  
C

![](images/f33317e4668d981f92be7a2bb54aba83e0f323217ae066f9f82cc252fb864ff4.jpg)  
d

![](images/1267a3bc2902b8240d4da6405a2f94364607e9e4acea3fcbd5882ded7f0c0ae4.jpg)

![](images/55422294a282f19e5ac187b37580bd3419e827e21ccc273a12b99e4eb37d593e.jpg)

![](images/ff0dc84aae20b4b314a723d84a523beec433439db91354693e1dc02a4e1801c0.jpg)

f  
![](images/56b2163f471fba4f72763af7a74b73e92c7965ce2ec364f85a9edea16a3d9cb6.jpg)

![](images/567e8fc9700fc541fe6f74ae4f07515f60bba49fbf6ad9e4fc28e998b60fd81c.jpg)

![](images/bfbbe37c9eff8bb4c05977b63b5169b16a7d2776eabb8d5843087a78246d4ff3.jpg)

g  
![](images/9231b19bb3be820fc381d68dc34cf159b7659a871f72cc357f1555cee63b2359.jpg)

h  
![](images/30eb73a75900a1dc10995ad741ab29c42e86c8679ad7e6ef49c6f3daaf910b9f.jpg)

![](images/7153d0ce7a964fe6aaec32819cdfbd56fc6dce48e66a96ef8ba3eebc24d52046.jpg)

![](images/5d30761ec54dc6cb72c939be445c2f73ad8977bda6995499e947b1bd9f0e0e0b.jpg)  
Figure 2: Stable low-rank solutions in continuous normalising flows drive mode collapse. a. The model is trained to transport a Gaussian mixture from orthogonal initial and data subspaces. b. Loss over iterations in models with different numbers of collapsed modes. c. The number of collapsed modes scales linearly with the stable rank of the weights W at convergence. d. The stable rank of the weights over training in models with different numbers of modes. e. The overlap between a pushed forward initial mode and a target mode is a summary statistic and gives us a low-dimensional description of training. This highlights two phases of learning: compression of the initial modes and learning the data modes. f. Top: Schur decomposition of the post-training weights in collapsed and non-collapsed models. Bottom: Non-collapsed models $( l e f t )$ learn a rotational vector field. Collapsed models (right) reveal attraction towards a stable slow manifold with rotational dynamics. g. The saddle weights and perturbations in the direction of collapsed or non-collapsed weights. Training from this point induces the respective solutions, represented by the summary statistic matrix at convergence. h-i. The stable rank of the state and adjoint operators over training.

## The rank of W directly determines collapse.

Empirically, we observe mode collapse to occur when the weights fail to learn a rotation mapping an initial mode to a target, instead contracting the difference between two modes and sending them to a single target. Each correctly transported mode requires a rank-2 rotational block in ${ \bf { \bar { \cal W } } } ,$ , so we expect that the rank deficiency of $\dot { W }$ should lead to more collapsed modes. To quantify this effect, we computed the stable rank of the weights and observed that lower-rank solutions exhibited a greater number of collapsed modes (Fig. 2c-d). Moreover, the learning dynamics leading to collapse were lower-dimensional than in non-collapsed solutions (Fig. B1b-c). To further characterise these solutions, we performed a real Schur decomposition $W = \boldsymbol { U } \bar { \boldsymbol { L } } \boldsymbol { U } ^ { \top }$ , where L is quasi-lower triangular. Non-collapsed models implemented the correct rotations between initial and target modes, reflected in anti-symmetric blocks of $L ,$ each $2 \times 2$ block representing a rotation in the subspace from one initial mode to one target mode (Fig. 2f). Instead, collapsed models did not learn all rotations, exhibiting compressive dynamics corresponding to negative real eigenvalues along the diagonal.

Summary statistics explain the low-rank collapse. To understand the learning dynamics underlying such collapse, it can be informative to examine summary statistics of the gradient at different points in training (App. B.3). In short, we defined summary statistics $s _ { i j }$ as the scalar overlaps between the transported initial means and target means at $t = 1$

$$
s \in \mathbb { R } ^ { m \times m } , \quad s _ { i j } = \frac { ( e ^ { W } \pmb { \mu } _ { i } ) \cdot \pmb { \mu } _ { j } ^ { * } } { R ^ { 2 } }
$$

Over training, these summary statistics underwent two phases. In an initial “compression phase”, the model reduces the loss by compressing the initial modes and reducing the effective covariance of the distribution under the flow. Here, the weights have no rotational component, so the overlaps $s _ { i j }$ remain close to zero. In the subsequent “mode alignment phase”, the model rapidly learns to map initial modes to target modes by a rotation, settling at either collapsed or non-collapsed solutions.

In non-collapsed runs, the summary statistics converge to a permutation matrix, whereas collapsed runs have columns with no large entries, corresponding to the target modes with no pushed forward initial mode aligned with them (Fig. 2e). The compression, collapsed, and non-collapsed weights can be analytically shown to be fixed points of the gradient flow (App. B.5). From the compression fixed point, we can induce either solution by a perturbation in the direction of the corresponding fixed point, indicating that the compression weights are a saddle with unstable directions towards either collapsed or non-collapsed solutions (Fig. 2g).

Adjoint and state operator ranks track collapse. To understand the interplay between collapse, summary statistics and the gradient operators, we looked at the stable rank of the state and adjoint operators over training (Fig. 2h-i). This revealed that in the mode alignment phase, both operator ranks corresponding to collapsed solutions are lower than in non-collapsed solutions, mirroring the decrease in rank we previously observed in the weights. The prefactor starts at 1 and then increases to 8.9 in the compression phase, before splitting across collapsed and non-collapsed models with collapsed models having a higher value than non-collapsing ones (Fig. B1d). The operator ranks exhibit non-linear responses to perturbations around the saddle, with perturbations in the direction of different solutions leading to distinct rank values (Fig. B2). This suggests that the operator rank could be useful as an early diagnostic of collapse in less tractable models, where it is not clear what are the relevant summary statistics.

In sum, variational objectives can make CNFs collapse to suboptimal solutions, even when the model is expressive enough to learn the optimal transport between the initial and target distributions.

## 3.2 Saddle-to-saddle dynamics drive effective collapse during linear flow matching

In the previous section, we showed that collapse is a stable fixed point in the gradient flow. This tendency towards collapse comes from the form of the reverse KL which is known to produce mode seeking behaviour. However, under different objectives, it is possible to effectively get mode collapse from saddle-to-saddle dynamics, which lead to sequential feature learning, similar to what has been observed in the learning dynamics of deep linear networks<sup>3</sup>. Although this type of collapse is structurally distinct from the attractor-based collapse in CNFs, we argue that low-rank implicit biases combined with imbalanced data lead to effective collapse under a finite training budget.

Model. Flow-matching models are trained by regressing the optimal velocity, $\mathbf { u } ^ { * } ( \mathbf { x } , t ) \equiv \mathbf { u } _ { t } ^ { * } ( \mathbf { x } )$ ， directly with a neural network model $\mathbf { u } ^ { \theta } ( \mathbf { x } , t ) \equiv \mathbf { u } _ { t } ^ { \theta } ( \mathbf { x } )$ :

$$
\mathcal { L } _ { \mathrm { F M } } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { \bar { \mathbf { x } } \sim p _ { t } } \left[ \left| \left| \mathbf { u } _ { t } ^ { * } ( \bar { \mathbf { x } } ) - \mathbf { u } _ { t } ^ { \theta } ( \bar { \mathbf { x } } ) \right| \right| ^ { 2 } \right] d t
$$

where $\hat { \mathbf { x } } \sim p _ { t }$ is sampled from the chosen interpolant trajectory which interpolates between base distribution $p _ { 0 }$ and target distribution $p ^ { * }$ over time $t \in \ [ 0 , 1 ]$ , and $\mathbf { u } _ { t } ^ { * } ( \mathbf { x } )$ is the vector field that generates it. This formulation contains no terminal loss, so $\Phi ( { \bf \bar { x } } ( T ) ) = 0$ in Eq. 2. Moreover, since the loss is integrated along the ground-truth trajectory ¯x, the drift term $\mathbf { f } _ { \pmb { \theta } }$ from Eq. 1 contains no learnable parameters. Thus, θ enters only through the instantaneous loss ℓ. In practice, the optimal field is intractable, and so an objective conditioned on individual data samples is used, which yields the same gradient as the unconditional objective above $( { \mathrm { A p p . ~ C } } )$

To study the training dynamics in a tractable setting analogous to the CNF case, we trained linear flows in two different parameterisations using a teacher-student setup with optimal velocity:

$$
\mathbf { u } _ { t , \mathrm { s h a l l o w } } ^ { \theta } ( \mathbf { x } ) = W \mathbf { x } , \qquad \mathbf { u } _ { t , \mathrm { d e e p } } ^ { \theta } ( \mathbf { x } ) = W _ { 2 } W _ { 1 } \mathbf { x } \qquad \mathbf { u } _ { t } ^ { * } ( \mathbf { x } ) = W ^ { * } \mathbf { x }
$$

where $W ^ { * }$ is constructed to transport initial modes to their respective targets. This ensures that the model is well-specified, i.e. the student is able to achieve the optimal solution, isolating the effects due to the training dynamics from the expressivity of the model. We use the shallow model as a direct comparison to the CNF model, and the deep model to study sequential learning dynamics. As before, our initial and target distributions are Gaussian mixtures, but now we take our modes to be orthogonal: $\pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } = \pmb { \mu } _ { i } ^ { * } \cdot \pmb { \mu } _ { j } ^ { * } = R ^ { 2 } \delta _ { i j }$ . This choice means the model can act independently on each mixture mode, allowing us to study the learning time of each mode separately in the context of sequential learning. By varying the weighting of the $i ^ { \mathrm { { t h } } }$ target mode, denoted by $\pi _ { i }$ , we will study how imbalanced data affects learning dynamics driven by low-rank implicit biases.

Linear flow matching is robust to stable collapse. As expected, when trained with equal mode weightings, the model was able to recover the optimal vector field, and did not exhibit mode collapse, even as the number of modes in the target distribution increased (Fig. 3a). This suggests that the gradient flow of the flow-matching objective does not induce collapse as a stable attractor, in contrast to the variational objective.

To better understand the mechanisms that could lead to collapse under the flow-matching objective, we next adopt the operator view. The gradient of the linear flow-matching model can be written as:

$$
\nabla _ { W } \mathcal { L } = A \circ H = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathbf { r } ( t ) \otimes { \bar { \mathbf { x } } } ( t ) d t \right]
$$

where the state operator H is given by the interpolant trajectory $\bar { { \bf x } } ( t )$ , and the adjoint operator A by the residual, $\mathbf { r } ( t ) { \bar { \mathbf { \rho } } } = W { \bar { \mathbf { x } } } ( t ) - \mathbf { u } _ { t } ^ { * } ( { \bar { \mathbf { x } } } ( t ) )$ ). Importantly, here the state operator is independent of the weights, depending only on the optimal trajectory; thus its rank is constant throughout training. In contrast, under the variational objective, the stable rank of the state operator dropped at the transition between the compression and mode-learning phases, driven by changes in the weights $W$ (Fig. 3b). This suggests that the training dynamics of flow-matching models may become biased if the interpolant trajectory spectrum is imbalanced, affecting the stable rank of the state operator.

a  
![](images/7fe112830cc9d6c935ee988baa696876885d50902795ff1284f8a3ca0159eff9.jpg)

![](images/71796928dc241f487bb50be5f1d1a53ff2da7f83da8f750cdc043899d7ec0fc7.jpg)

Figure 3: In flow matching, collapse is not an attractor in the gradient flow. a. Likelihood of collapse as a function of the number of modes in the data for variational inference and flow-matching objectives. b. Rank of the state operators in the variational objective and flow-matching models over training.

Separated learning times of imbalanced modes drive effective collapse. Prior work has shown that deep linear networks exhibit sequential feature learning<sup>3</sup>. We asked whether this could serve as a mechanism for effective collapse, causing individual modes to be learned sequentially at different time scales. Since the state operator directly depends on the data spectrum, we hypothesised that introducing non-uniform class weightings might drive the model towards biased solutions. We thus trained linear models on Gaussian mixtures with imbalanced mode weightings. In this setting, the singular value spectrum of the state operator can be explicitly derived as a function of these weights, and used to analytically calculate its rank, showing that larger mode imbalance decreases the operator rank (App. C.2, Fig 4b). For imbalanced target distributions, deep linear models exhibited saddle-to-saddle patterns in the loss (Fig. 4a).

Summary statistics characterise mode learning times. Computing the overlap summary statistics, $s _ { i j }$ , for each initial-target mode pair over training revealed that progressively smaller modes are learned sequentially, with each mode undergoing its own saddle-like phase. Following prior work<sup>3</sup>, the learning dynamics can be solved exactly, and we find that the learning time of mode i is inversely related to its corresponding mixture weight $\pi _ { i } .$ , scaling as $\mathcal { O } ( \pi _ { i } ^ { - 1 } )$ (App. C.3). Interestingly, learning

Figure 4: Flow matching can have saddle-tosaddle dynamics under imbalanced modes. a. Flow-matching loss and summary statistics of the model over training (with weights $\pi =$ [0.01, 0.09, 0.90]). Both display staircase-like pattern corresponding to the sequential matching of initial to target modes over learning. Plateaus sit at roughly $s _ { i i } \approx 0 . 7 9$ as predicted. b. Rank of the state operator as a function of the imbalance of the weights of the mixture for 7 modes with small mode $\pi _ { \mathrm { m i n } } ~ = ~ \alpha$ and other modes $\begin{array} { r } { \pi _ { i } = \frac { 1 - \alpha } { 6 } } \end{array}$ . The mode imbalance is $\textstyle 1 - { \frac { \pi _ { \operatorname* { m i n } } } { \pi _ { i } } }$

![](images/2dedce456c81f39b7257e6d13a83d074447931749c2ecfdde103fa48e151923a.jpg)

![](images/3706457e3d9ab9a1179ad00de46b5644ab83ee51a873f61fa7708b0ca3852c43.jpg)

b  
![](images/964f609c137e7d23ba0497d8a77d82c46b57f44bfaa7480a2eebd99416fa8122.jpg)

a data mode corresponds to learning two singular mode directions with different learning times, so each mode’s learning trajectory contains a plateau. The first rank-1 direction to be learned is a shear, mixing the initial and target directions to achieve partial overlap of modes. The second direction adds another rank-1 component which cancels the symmetric part of the shear and induces the rotation mapping from initial to target modes. The bound prefactor began at 1 at initialisation and increased sequentially with each mode learnt to 6 at convergence (Fig. C2).

Thus, although theflow-matching objective prevents stable collapsedfixed points ofthe gradientflow in linear generative models, it can induce effective collapse by having slow collapsed saddles.

## 3.3 Non-linear flow-matching models sequentially learn data modes

So far we have focused on linear models for analytical tractability. However, our theoretical framework is general and applies to non-linear models. Non-linear models, unlike linear flows, can create new modes. We therefore asked whether similar sequential learning dynamics emerge in non-linear models, and whether this could contribute to a form of effective collapse.

First, we assume a Gaussian mixture target distribution with equal, isotropic covariances $\sigma _ { * } ^ { 2 } I _ { d }$ and unequal mixture weights. However here, unlike in the linear settings, the initial distribution is a single-mode Gaussian.

Operators in a tractable non-linear model. We trained a one hidden-layer ReLU network parameterised to match the form of the optimal velocity (App. C.6):

$$
\mathbf { u } _ { \theta } ( \mathbf { x } , t ) = c ( t ) \mathbf { x } + ( 1 - t c ( t ) ) W _ { 2 } \sigma ( W _ { 1 } \mathbf { x } + \mathbf { b } ) , \qquad c ( t ) = { \frac { t \sigma _ { * } ^ { 2 } - ( 1 - t ) \sigma ^ { 2 } } { t ^ { 2 } \sigma _ { * } ^ { 2 } + ( 1 - t ) ^ { 2 } \sigma ^ { 2 } } }
$$

where $c ( t )$ is a scalar given analytically and $W _ { 2 } \in \mathbb { R } ^ { n \times m } , W _ { 1 } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m }$ are trained parameters. By using such a parametrisation, the time-dependency is handled purely by the scalar prefactors, and the model only needs to learn the spatial component of the field. For such a model trained under the MSE loss with conditional velocity field target (App. C.4), we can write the gradients with respect to the two weight matrices as a composition of two operators (App. C.7):

$$
\nabla _ { W _ { 1 } } \mathcal { L } = \underset { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } { \mathbb { E } } \left[ \int _ { 0 } ^ { 1 } \underset { A _ { 1 } } { \underbrace { J R ( \bar { \mathbf { x } } ( t ) ) } } \otimes \underset { H _ { 1 } } { \bar { \mathbf { x } } ( t ) } d t \right] , \quad \nabla _ { W _ { 2 } } \mathcal { L } = \underset { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } { \mathbb { E } } \left[ \int _ { 0 } ^ { 1 } \underset { A _ { 2 } } { \underbrace { R ( \bar { \mathbf { x } } ( t ) ) } } \otimes \underset { H _ { 2 } } { \underbrace { \mathbf { h } _ { 1 } ( \bar { \mathbf { x } } ( t ) ) } } d t \right]
$$

where $\bar { { \bf x } } ( t )$ is a point along the interpolant trajectory, $R ( { \bar { \bf x } } ( t ) )$ ) is the residual error per sample, ${ \bf z } =$ $W _ { 1 } { \bf x } + { \bf b }$ is the network pre-activations, h is the network post-activations and $J = \mathrm { d i a g } \left( \sigma ^ { \prime } ( \bar { \bf z } _ { 1 } ) \right) W _ { 2 } ^ { \top }$ is the network Jacobian. Similarly to the linear flow matching case, these are both of the form $A \circ H$ where the adjoint operator A propagates errors backwards and the operator H represents the state of the network. As before, the $\dot { W _ { 1 } }$ gradient has an operator which only depends on the data distribution.

Similar to the linear model, we observed sequential learning dynamics corresponding to each mode being learned one-by-one (Fig. 5a). The first inflection point in the loss corresponded to the probability density being transported to the dominant mode. After a few thousand training iterations a second inflection point occurred when the density expanded to cover the second strongest mode. By the point the loss saturated, the transported density covered all modes, with attracting dynamics surrounding each mode in the learned vector field (Fig. 5b).

As in the linear model, we aimed to explain these phenomena by studying the rank of the weights and operators. First, the ranks of the weights correlated with the learning of each modes, with an increase in the rank of $W _ { 1 }$ at each time a mode was learned (Fig. 5d). Moreover, the first layer adjoint rank and second layer state rank also reflect mode learning, with sharp changes at each inflection point (Fig. 5e-f). By systematically varying the imbalance between the strongest and weakest mode weightings, we observed the time to learn the smallest mode increasing according to a scaling law of τ<sub>learn</sub> $\propto \pi _ { \mathrm { m i n } } ^ { - 0 . 7 2 \pm 0 . 0 1 }$ (Fig. 5c). The linear theory predicts an exponent of 1, so the non-linear model learned faster than the linear prediction. Finally, In appendix C.4 we also show that less tractable models trained on MNIST data display similar learning time scaling.

Together, these results show that the effective collapse learning dynamics observed in linear flowmatching models persist in non-linear settings, with progressively delayed learning of modes.

![](images/9d3555a2f47f7489ceb684b4375debb2474a45e26aeece84211adebaa0598f11.jpg)  
a  
C  
Figure 5: Non-linear flow-matching models exhibit effective collapse due to low-rank biases. a. Flow-matching loss over training displaying progressively longer learning time for each mode. b. (Top) Generative log-density at three time points of training evaluated via the continuity equation (App. C.4). The modes are progressively learned (Bottom) Corresponding vector field, where additional fixed points are created with each learned mode. c. Learning time as a function of the inverse of the smallest mode, with $\pi _ { \operatorname* { m i n } } \in [ 0 . 0 4 , 0 . 1 6 ]$ . Scaling exponent computed as $0 . 7 2 \pm 0 . 0 1$ . d. Stable rank of the weights over training. The hidden layer weights $W _ { 1 }$ show progressive increase in their rank with each learned mode. e. Adjoint (residual) operator rank for the two weight matrices. f. State operator rank for the two weight matrices.

## 4 Discussion

Summary. Implicit biases are generally considered as creating beneficial biases that help representation learning in deep networks. Here, we examined how low-rank implicit biases can cause mode collapse in dynamical systems-based generative models. We derived the gradient of linear parameters as a composition of a state operator and an adjoint operator, and demonstrated how this view can explain collapse in various models. First, in linear continuous normalising flows under a variational objective, we used the adjoint view to show analytically that collapse corresponds to a stable attractor in the gradient flow. Interestingly, in this setting collapse can occur even when the target distribution is balanced, in contrast to prior studies where imbalance was necessary for collapse<sup>6,</sup> <sup>54</sup>. Next, we showed in non-linear flow-matching models that an effective mode collapse could occur due to sequential mode learning.

Future work. This work focused on describing the learning mechanisms underlying mode collapse in generative models, but did not consider how to mitigate it. Since operator stable ranks are lower in collapsing models, it might be possible to use stable rank as a diagnostic to abort runs exhibiting low-rank operators early in training. To further mitigate collapse, it may be possible to use optimiser methods which re-weight the gradient singular value spectrum, such as Muon<sup>63</sup>. Indeed, recent work has shown that Muon can remove saddle-to-saddle dynamics classically observed in gradient flow<sup>64</sup>, and can be beneficial in settings where there is a class imbalance<sup>65</sup>. While in our linear models the collapsed and optimal solutions could be understood in terms of the Schur decomposition, future works could use recent dynamical systems tools to study the geometry of collapsed representations in non-linear models<sup>66–68</sup>. Finally, while we mainly used summary statistics as a diagnostic tool to identify collapse, future work could give a full low-dimensional description of the gradient flow.

Limitations. This work relies on simplified analytical settings, including linear and controlled non-linear generative models with structured data distributions, which may not fully capture the complexity of large-scale modern architectures and real-world datasets. While the operator-based framework is generally applicable and reveals clear mechanisms for low-rank bias and collapse, the study of stable sub-optimal fixed points or the explicit derivation of saddles in the dynamics of learning of non-linear models remains a challenge of learning theory. Moreover, most of our results concern the population loss, and future works could study how finite sampling affects collapse.

Funding statement. This work was supported by the Agence Nationale de la Recherche (ANR-23-IACL-0008, ANR-26-CE23-2356-01, ANR-17-910 EURE-0017) and the European Research Council (Grant No. 101170541). The authors declare no competing interests.

## References

[1] A. Curth et al. “A u-turn on double descent: Rethinking parameter counting in statistical learning”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 55932– 55962.

[2] A. Power et al. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. Jan. 6, 2022. arXiv: 2201.02177[cs].

[3] A. M. Saxe et al. Exact solutions to the nonlinear dynamics oflearning in deep linear neural networks. Feb. 19, 2014. arXiv: 1312.6120[cs].

[4] D. Saad et al. “On-line learning in soft committee machines”. In: Physical Review E 52.4 (1995), pp. 4225–4243.

[5] L. Arnaboldi et al. “From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks”. In: Proceedings ofThirty Sixth Conference on Learning Theory. The Thirty Sixth Annual Conference on Learning Theory. PMLR, July 12, 2023, pp. 1199–1227.

[6] R. Soletskyi et al. A theoretical perspective on mode collapse in variational inference. Oct. 17, 2024. arXiv: 2410.13300[stat].

[7] H. Ninou et al. “Curl Descent: Non-Gradient Learning Dynamics with Sign-Diverse Plasticity”. In: Advances in Neural Information Processing Systems 38 (2026), pp. 165703–165741.

[8] D. Soudry et al. “The implicit bias of gradient descent on separable data”. In: Journal of Machine Learning Research 19.70 (2018), pp. 1–57.

[9] A. Ali et al. “The Implicit Regularization of Stochastic Gradient Flow for Least Squares”. In: Proceedings of the 37th International Conference on Machine Learning. International Conference on Machine Learning. PMLR, Nov. 21, 2020, pp. 233–244.

[10] S. Goldt et al. “Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup”. In: Journal ofStatistical Mechanics: Theory and Experiment 2020.12 (Dec. 1, 2020), p. 124010. arXiv: 1906.08632[stat].

[11] E. Abbe et al. “The staircase property: How hierarchical structure can guide deep learning”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 26989–27002.

[12] A. Jacot et al. Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity. ADS Bibcode: 2021arXiv210615933J. June 1, 2021.

[13] A. Atanasov et al. “Neural Networks as Kernel Learners: The Silent Alignment Effect”. In: International Conference on Learning Representations. Oct. 6, 2021.

[14] Y. Song et al. “Score-based generative modeling through stochastic differential equations”. In: arXiv preprint arXiv:2011.13456 (2020).

[15] R. Rombach et al. “High-resolution image synthesis with latent diffusion models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 10684–10695.

[16] T. Geffner et al. “Proteina: Scaling flow-based protein structure generative models”. In: arXiv preprint arXiv:2503.00710 (2025).

[17] E. Shang et al. “Predicting the unseen: a diffusion-based debiasing framework for transcriptional response prediction at single-cell resolution”. In: Proceedings of the National Academy ofSciences 122.52 (2025), e2525268122.

[18] J. Huang et al. “DiffusionPDE: Generative PDE-solving under partial observation”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 130291–130323.

[19] B. Holzschuh et al. “Solving inverse physics problems with score matching”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 61888–61922.

[20] T. Minka et al. “Divergence measures and message passing”. In: (2005).

[21] L. I. Midgley et al. “Flow Annealed Importance Sampling Bootstrap”. In: The Eleventh International Conference on Learning Representations. 2023.

[22] D. Blessing et al. “Beyond ELBOs: A large-scale evaluation of variational methods for sampling”. In: arXiv preprint arXiv:2406.07423 (2024).

[23] Y. Qin et al. Class-Balancing Diffusion Models. June 14, 2023. arXiv: 2305.00562[cs].

[24] J. Xiao et al. “On the algorithmic bias of aligning large language models with rlhf: Preference collapse and matching regularization”. In: Journal of the American Statistical Association 120.552 (2025), pp. 2154–2164.

[25] J. Zhang et al. “Verbalized sampling: How to mitigate mode collapse and unlock llm diversity”. In: arXiv preprint arXiv:2510.01171 (2025).

[26] I. Shumailov et al. “AI models collapse when trained on recursively generated data”. In: Nature 631.8022 (July 2024), pp. 755–759.

[27] M. Huh et al. The Low-Rank Simplicity Bias in Deep Networks. Mar. 23, 2023. arXiv: 2103. 10427[cs].

[28] A. Canatar et al. “Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks”. In: Nature Communications 12.1 (May 18, 2021), p. 2914.

[29] A. Pellegrino et al. “Low tensor rank learning of neural dynamics”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 11674–11702.

[30] J. Hazelden et al. “KPFlow: An Operator Perspective on Dynamic Collapse Under Gradient Descent Training of Recurrent Networks”. In: arXiv preprint arXiv:2507.06381 (2025).

[31] B. Wang et al. “An analytical theory of spectral bias in the learning dynamics of diffusion models”. In: Advances in Neural Information Processing Systems (2025).

[32] S. Gunasekar et al. “Implicit regularization in matrix factorization”. In: Advances in neural information processing systems 30 (2017).

[33] L. Chizat et al. “Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss”. In: Proceedings of Thirty Third Conference on Learning Theory. Conference on Learning Theory. PMLR, July 15, 2020, pp. 1305–1338.

[34] S. Pesme et al. “Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 29218– 29230.

[35] H. Shah et al. “The pitfalls of simplicity bias in neural networks”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 9573–9585.

[36] G. Valle-Perez et al. “Deep learning generalizes because the parameter-function map is biased towards simple functions”. In: arXiv preprint arXiv:1805.08522 (2018).

[37] N. Rahaman et al. “On the Spectral Bias of Neural Networks”. In: Proceedings ofthe 36th International Conference on Machine Learning. International Conference on Machine Learning. PMLR, May 24, 2019, pp. 5301–5310.

[38] S. Arora et al. “Implicit regularization in deep matrix factorization”. In: Advances in neural information processing systems 32 (2019).

[39] S. Pesme et al. “Saddle-to-Saddle Dynamics in Diagonal Linear Networks”. In: Thirty-seventh Conference on Neural Information Processing Systems. Nov. 2, 2023.

[40] I. Bantzis et al. “Saddle-to-saddle dynamics in deep relu networks: Low-rank bias in the first saddle escape”. In: arXiv preprint arXiv:2505.21722 (2025).

[41] C. Merger et al. “Generalization dynamics of linear diffusion models”. In: arXiv preprint arXiv:2505.24769 (2025).

[42] L. Bardone et al. “A theory of learning data statistics in diffusion models, from easy to hard”. In: arXiv preprint arXiv:2603.12901 (2026).

[43] N. Berman et al. “One-step offline distillation of diffusion-based models via Koopman modeling”. In: arXiv preprint arXiv:2505.13358 (2025).

[44] H. Bai et al. “Hierarchical Koopman Diffusion: Fast Generation with Interpretable Diffusion Trajectory”. In: arXiv preprint arXiv:2510.12220 (2025).

[45] K. A. Nicoli et al. “Detecting and mitigating mode-collapse for flow-based sampling of lattice field theories”. In: Physical Review D 108.11 (2023), p. 114501.

[46] Y. Abbahaddou et al. “A Geometry-Aware Metric for Mode Collapse in Time Series Generative Models”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2025.

[47] H. Cui et al. “Asymptotics of feature learning in two-layer networks after one gradient-step”. In: Proceedings of the 41st International Conference on Machine Learning. Vol. 235. ICML’24. Vienna, Austria: JMLR.org, July 21, 2024, pp. 9662–9695.

[48] R. Barceló et al. Avoiding mode collapse in diffusion models fine-tuned with reinforcement learning. Oct. 10, 2024. arXiv: 2410.08315[stat].

[49] J. He et al. “Training neural samplers with reverse diffusive kl divergence”. In: arXiv preprint arXiv:2410.12456 (2024).

[50] J. Deasy et al. Heavy-tailed denoising score matching. Apr. 12, 2022. arXiv: 2112.09788[cs].

[51] Anonymous. “Debiasing Diffusion Models via Score Guidance”. In: Transactions on Machine Learning Research (2026).

[52] Z. Kadkhodaie et al. “Generalization in diffusion models arises from geometry-adaptive harmonic representations”. In: arXiv preprint arXiv:2310.02557 (2023).

[53] H. Cui et al. A solvable model of learning generative diffusion: theory and insights. Nov. 10, 2025. arXiv: 2501.03937[cs].

[54] L. Fogliani et al. “Annealing in variational inference mitigates mode collapse: A theoretical study on Gaussian mixtures”. In: arXiv preprint arXiv:2602.12923 (2026).

[55] A. GX-Chen et al. “KL-Regularized Reinforcement Learning is Designed to Mode Collapse”. In: arXiv preprint arXiv:2510.20817 (2025).

[56] T. Yoon et al. “Diffusion probabilistic models generalize when they fail to memorize”. In: ICML 2023 workshop on structured probabilistic inference {\&} generative modeling. 2023.

[57] W. Gao et al. “How do flow matching models memorize and generalize in sample data subspaces?” In: arXiv preprint arXiv:2410.23594 (2024).

[58] Z. Li et al. “A Kinetic-Energy Perspective of Flow Matching”. In: arXiv preprint arXiv:2602.07928 (2026).

[59] P. Gao et al. “On simplicity and complexity in the brave new world of large-scale neuroscience”. In: Current Opinion in Neurobiology. Large-Scale Recording Technology (32) 32 (June 1, 2015), pp. 148–155.

[60] J. Garnier-Brun et al. “Biased Generalization in Diffusion Models”. In: arXiv preprint arXiv:2603.03469 (2026).

[61] J. B. Conway. A course infunctional analysis. Springer, 2019.

[62] L. S. Pontryagin. Mathematical theory ofoptimal processes. Routledge, 2018.

[63] K. Jordan et al. “Muon: An optimizer for hidden layers in neural networks, 2024”. In: URL https://kellerjordan. github. io/posts/muon 6.3 (2024), p. 4.

[64] S. Dragutinovic et al. “To use or not to use muon: How simplicity bias in optimizers matters”.´ In: arXiv preprint arXiv:2603.00742 (2026).

[65] B. Vasudeva et al. “How Muon’s spectral design benefits generalization: A study on imbalanced data”. In: International Conference on Learning Representations. Vol. 2026. 2026, pp. 142276– 142311.

[66] A. Pellegrino et al. “RNNs perform task computations by dynamically warping neural representations”. In: Advances in Neural Information Processing Systems 38 (2026), pp. 3579– 3612.

[67] N. A. Cayco-Gajic et al. “Geometry-aware similarity metrics for neural representations on Riemannian and statistical manifolds”. In: arXiv preprint arXiv:2603.28764 (2026).

[68] R. Karczewski et al. “The spacetime of diffusion models: An information geometry perspective”. In: International Conference on Learning Representations. Vol. 2026. 2026, pp. 106128– 106152.

## A Theory

## A.1 Proof of theorem 2.1

In this section, we derive the exact gradient of a generalised loss function in a non-linear flow model, with respect to a linear weight parameter W. We consider a system governed by a Stratonovich stochastic differential equation (SDE), where W affects the instantaneous loss, the drift, and the diffusion terms through potentially distinct feature inputs. This derivation encompasses the three types of losses commonly encountered in generative modelling with dynamical systems: variational objectives, flow and score matching.

Mathematical setting. Let $\mathbf { x } ( t ) \in \mathbb { R } ^ { d }$ denote the state of the system at time $t \in [ 0 , T ]$ , evolving according to the Stratonovich SDE:

$$
{ \bf d } { \bf x } ( t ) = { \bf f } _ { \theta } ( { \bf x } ( t ) , t ) d t + G _ { \theta } ( { \bf x } ( t ) , t ) \circ { \bf d } { \bf B } , \qquad { \bf x } ( 0 ) \sim p _ { 0 }\tag{3}
$$

where ${ \bf x } ( t )$ is the state of the system, dB increments of a Brownian motion, $\mathbf { f } _ { \pmb { \theta } } : \mathbb { R } ^ { d } \times \mathbb { R }  \mathbb { R } ^ { d }$ and $G _ { \pmb { \theta } } : \mathbb { R } ^ { d } \times \mathbb { R }  \mathbb { R } ^ { d \times d }$ the drift and diffusion components of the stochastic differential equation with parameters $\pmb \theta \in \Theta$ . We use the Stratonovich integral (◦) rather than the Itô integral, but we note that they can be interchanged.

We evaluate the system via a continuous-time expected loss functional:

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbb { B } } \left[ \int _ { 0 } ^ { T } \ell ( \mathbf { x } ( t ) , t ) d t + \Phi ( \mathbf { x } ( T ) ) \right]\tag{4}
$$

where $\ell : \mathbb { R } ^ { d } \times [ 0 , T ] $ R is an instantaneous loss and $\Phi : \mathbb { R } ^ { d } $ R a terminal loss which only depends on θ through x(T). The core architectural assumption is that θ contains the parameter matrix $\bar { W } \in \mathbb { R } ^ { k \times q }$ which acts linearly. For the sake of generality, we assume that the weight parameter could potentially be part of any of the instantaneous loss, drift and diffusion functions. In several cases the drift and diffusion models are parameterised by different networks, in which case any particular weight would only be part of one of them. In specific scenarios, such as physics-informed neural networks and some generative architectures, a network and its derivative can be parameterising the drift and diffusion terms at once, in which case W appears in both terms.

To account for this, we assume W may act on distinct post-activation feature vectors $\mathbf { h } _ { \ell } ( t ) , \mathbf { h } _ { \mathbf { f } } ( t ) , \mathbf { h } _ { G } ( t ) \in \mathbb { R } ^ { q }$ for the instantaneous loss $\ell ,$ drift, f and diffusion G terms respectively. The pre-activations feature vectors are thus given by:

$$
\mathbf { z } _ { \ell } ( t ) = W \mathbf { h } _ { \ell } ( t ) , \quad \mathbf { z } _ { \ell } ( t ) = W \mathbf { h } _ { \mathrm { f } } ( t ) , \quad \mathbf { z } _ { G } ( t ) = W \mathbf { h } _ { G } ( t )
$$

so that $\mathbf { z } _ { ( \cdot ) } ( t ) \in \mathbb { R } ^ { k }$ . While here we assume a shared parameter for the sake of generality, in many instances different neural network models will separately parameterise the loss, drift and diffusion terms, and therefore some of these partial derivatives will be zero. Our goal is to derive $\nabla _ { W } \mathcal { L }$ in terms of two operators, one which will be directly related to the post-activation vector, and one adjoint operator.

Adjoints. Because the state ${ \bf x } ( t )$ depends on W through the integration of the SDE, we cannot differentiate $\mathcal { L } ( W )$ by simply moving the gradient operator inside the expectation. Instead we must define the adjoint system.

Definition A.1. The adjoint $\mathbf { a } ( t ) \in \mathbb { R } ^ { d }$ , represents how small changes in the trajectory of the dynamical system relate to small changes in the loss function:

$$
{ \bf a } ( t ) = \frac { \partial \mathcal { L } } { \partial { \bf x } ( t ) }
$$

where ${ \bf x } ( t )$ is the state of the system at time $t .$

Lemma A.2. The adjoint is the solution to the following dynamical system:

$$
\mathbf d \mathbf a ( t ) = - \left( \nabla _ { \mathbf x }  { \boldsymbol { \ell } } ^ { \top } + ( \nabla _ { \mathbf x } \mathbf f ) ^ { \top } \mathbf a ( t ) \right) d t - ( \nabla _ { \mathbf x }  { \boldsymbol { G } } ) ^ { \top } \mathbf a ( t ) \circ \mathbf d \mathbf B ( t ) , \quad \mathbf a ( T ) = \mathbf a _ { T }
$$

where ${ \bf a } _ { T } = \nabla _ { \bf x } \Phi ( { \bf x } ( T ) )$ , or ${ \bf a } _ { T } = 0$ if there is no terminal loss.

Proof. For the case with a terminal loss and no instantaneous loss $( \mathrm { i . e . ~ } \ell = 0 )$ , the equation is the stochastic adjoint equation of Kidger et al. $( 2 0 2 1 , \mathrm { E q . ~ } 6 ) ^ { 1 }$ <sup>1</sup>. To include a running loss, we can augment the state with $\tilde { \mathbf { x } } = ( \mathbf { x } , y )$ where $\bar { d y } = \ell ( { \bf x } ( t ) , t ) d t$ . Then the loss is terminal:

$$
\int _ { 0 } ^ { T } \ell d t + \Phi ( \mathbf { x } ( T ) ) = y ( T ) + \Phi ( \mathbf { x } ( T ) )
$$

$\ell , \mathbf { f } ,$ and G do not depend on $y ,$ so the adjoint of $y$ satisfies $d a _ { y } = 0$ with $a _ { y } ( T ) = 1$ , which gives $a _ { y } = 1$ . The x-component of the augmented adjoint then contains the extra term $- ( \nabla _ { \mathbf { x } } \ell ) ^ { \top } a _ { y } d t =$ $- \nabla _ { \mathbf x } \ell ^ { \top }$ dt which gives the stated equation. □

Importantly, the gradient of the model parameters can be written as an expectation of this adjoint, over initial states and Brownian paths.

Lemma A.3. Dropping the dependence on ${ \bf x } ( t )$ for notational clarity, the gradient of model parameters can be written as:

$$
\nabla _ { \theta } \mathcal { L } = \mathbb { E } _ { \mathbf { x _ { 0 } } , \mathbf { B } } \left[ \int _ { 0 } ^ { T } \left( \nabla _ { \theta } \ell + ( \nabla _ { \theta } \mathbf { f } ) ^ { \top } \mathbf { a } ( t ) \right) d t + \int _ { 0 } ^ { T } ( \nabla _ { \theta } G ) ^ { \top } \mathbf { a } ( t ) \circ d \mathbf { B } ( t ) \right]
$$

Proof. Let $\begin{array} { r } { \eta = \frac { \partial \mathbf { x } } { \partial \theta } } \end{array}$ . Differentiating the SDE in Eq. 3 gives:

$$
\mathbf { d } \eta = ( \nabla _ { \mathbf { x } } \mathbf { f } \eta + \nabla _ { \theta } \mathbf { f } ) d t + ( \nabla _ { \mathbf { x } } G \eta + \nabla _ { \theta } G ) \circ \mathbf { d } \mathbf { B }
$$

By the Stratonovich product rule and Lemma A.2:

$$
\begin{array} { r l } & { \mathbf { d } ( \mathbf { a } ^ { \top } \pmb { \eta } ) = \mathbf { d } \mathbf { a } ^ { \top } \pmb { \eta } + \mathbf { a } ^ { \top } \mathbf { d } \pmb { \eta } } \\ & { \qquad = - \nabla _ { \mathbf { x } } \ell \pmb { \eta } d t + \mathbf { a } ^ { \top } \nabla _ { \theta } \mathbf { f } d t + \mathbf { a } ^ { \top } \nabla _ { \theta } G \circ \mathbf { d } \mathbf { B } } \end{array}
$$

Integrate from 0 to $T .$ . The left hand side is $\mathbf { a } ( T ) ^ { \top } \pmb { \eta } ( T ) - \mathbf { a } ( 0 ) ^ { \top } \pmb { \eta } ( 0 ) = \nabla _ { \mathbf { x } } \Phi \pmb { \eta } ( T )$ since $\pmb { \eta } ( 0 ) = 0$ So:

$$
\int _ { 0 } ^ { T } \nabla _ { \mathbf { x } } \ell \eta d t + \nabla _ { \mathbf { x } } \Phi \eta ( T ) = \int _ { 0 } ^ { T } \mathbf { a } ^ { \top } \nabla _ { \theta } \mathbf { f } d t + \int _ { 0 } ^ { T } \mathbf { a } ^ { \top } \nabla _ { \theta } G \circ \mathbf { d B }
$$

Now we recognise the pathwise gradient of the loss in Eq. 4 is $\begin{array} { r } { \int _ { 0 } ^ { T } ( \nabla _ { \mathbf { x } } \ell \pmb { \eta } + \nabla _ { \mathbf { \theta } } \ell ) d t + \nabla _ { \mathbf { x } } \Phi \pmb { \eta } ( T ) } \end{array}$ Substituting into the above, transposing, and taking expectations gives the result.

In our setting, the parameter of interest is the matrix $W .$ , which acts linearly, thus endowing the SDE of the gradient with a particular structure.

We evaluate the parameter Jacobians $\nabla _ { W } \ell , \nabla _ { W } \mathbf { f } ,$ , and $\nabla _ { W } G$ by applying the chain rule through the linear projections $\mathbf { z } _ { \ell } , \mathbf { z } _ { \mathbf { f } } ,$ , and $\mathbf { z } _ { G }$ . First, the terms of the loss which do not depend on x have gradients independent of the adjoint. Since $\mathbf { z } _ { \ell } ( t ) = W \mathbf { h } _ { \ell } ( t )$ , differentiating with respect to the matrix $\bar { W }$ gives an outer product:

$$
\nabla _ { W } \ell = \mathbf { a } _ { \ell } ( t ) \otimes \mathbf { h } _ { \ell } ( t )
$$

where $\mathbf { a } _ { \ell } ( t ) = \partial _ { \mathbf { z } _ { \ell } } \ell ( t )$ . The sensitivity of the drift and diffusion terms are mediated by the adjoint state ${ \bf a } ( t )$ . We define the drift error vector $\mathbf { a } _ { \mathbf { f } } ( t ) \in \mathbb { R } ^ { k }$ by pulling the adjoint back through the drift function:

$$
\mathbf { a } _ { \mathbf { f } } ( t ) = \left( \nabla _ { \mathbf { z } _ { \mathbf { f } } } \mathbf { f } \right) ^ { \top } \mathbf { a } ( t )
$$

Applying the chain rule through $\mathbf { z } _ { \mathbf { f } } ( t ) = W \mathbf { h } _ { \mathbf { f } } ( t )$ gives:

$$
( \nabla _ { W } \mathbf { f } ) ^ { \top } \mathbf { a } ( t ) = \mathbf { a } _ { \mathbf { f } } ( t ) \otimes \mathbf { h } _ { \mathbf { f } } ( t )
$$

Similarly:

$$
\mathbf { a } _ { G } ( t ) = ( \nabla _ { \mathbf { z } _ { G } } G ( \mathbf { x } ( t ) , \mathbf { z } _ { G } ( t ) , t ) ) ^ { \top } \mathbf { a } ( t ) \qquad ( \nabla _ { W } G ) ^ { \top } \mathbf { a } ( t ) = \mathbf { a } _ { G } ( t ) \otimes \mathbf { h } _ { G } ( t )
$$

Corollary A.4. For a weight W that enters only through $\mathbf { z } _ { ( \cdot ) } = W \mathbf { h } _ { ( \cdot ) }$

$$
\nabla _ { W } \mathcal { L } = \mathbb { E } \left[ \int _ { 0 } ^ { T } \Big ( \mathbf { a } _ { \ell } ( t ) \otimes \mathbf { h } _ { \ell } ( t ) + \mathbf { a } _ { \mathbf { f } } ( t ) \otimes \mathbf { h } _ { \mathbf { f } } ( t ) \Big ) d t + \int _ { 0 } ^ { T } \left( \mathbf { a } _ { G } ( t ) \otimes \mathbf { h } _ { G } ( t ) \right) \circ \mathrm { d } \mathbf { B } ( t ) \right]
$$

where $\mathbf { a } _ { \ell } = \nabla _ { \mathbf { z } _ { \ell } } \ell , \mathbf { a } _ { \mathbf { f } } = ( \nabla _ { \mathbf { z } _ { \mathbf { f } } } \mathbf { f } ) ^ { \top } \mathbf { a } a n d \mathbf { a } _ { G } = ( \nabla _ { \mathbf { z } _ { G } } G ) ^ { \top } \mathbf { a } .$

Proof. Since $\mathbf { z } _ { ( \cdot ) } = W \mathbf { h } _ { ( \cdot ) }$ , the chain rule gives:

$$
\frac { \partial } { \partial W _ { i j } } = \left( \frac { \partial } { \partial \mathbf { z } _ { ( \cdot ) } } \right) _ { i } \mathbf { h } _ { ( \cdot ) , j }
$$

Applying this to each term,

$$
\nabla _ { W } \ell = \mathbf { a } _ { \ell } \otimes \mathbf { h } _ { \ell } , \quad \left( \nabla _ { W } \mathbf { f } \right) ^ { \top } \mathbf { a } = \mathbf { a } _ { \mathbf { f } } \otimes \mathbf { h } _ { f } , \quad \left( \nabla _ { W } G \right) ^ { \top } \mathbf { a } = \mathbf { a } _ { G } \otimes \mathbf { h } _ { G }
$$

Then substituting these into Lemma ${ \mathrm { A } } . 3$ gives the result.

In particular, because the contribution of the loss, drift and diffusion terms to the gradient are additive, the per sample instantaneous gradient is at most rank 3. As mentioned earlier, in some instances the weight parameter will in fact appear in only one of these objects $( \mathrm { e . g . }$ , the drift and loss terms in the CNF and FM models in the main text), and therefore the instantaneous gradient may be of even lower rank.

Gradient as an operator. From corollary $_ { \mathrm { A . 4 , } }$ the gradient can be rewritten as in Theorem 2.1:

$$
\nabla _ { W } \mathcal { L } = A _ { \ell } \circ H _ { \ell } + A _ { \mathbf { f } } \circ H _ { \mathbf { f } } + A _ { G } \circ H _ { G }
$$

where $A _ { ( \cdot ) }$ and $H _ { ( \cdot ) }$ are linear operators:

$$
A _ { ( \cdot ) } : L ^ { 2 } ( \mathbb { R } ) \to \mathbb { R } ^ { k } , \qquad A _ { ( \cdot ) } ( f ) = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { B } } \left[ \int _ { 0 } ^ { T } \mathbf { a } _ { ( \cdot ) } ( t ) f ( t ) d t \right]
$$

$$
H _ { ( \cdot ) } : \mathbb { R } ^ { q } \to L ^ { 2 } ( \mathbb { R } ) , \qquad ( H _ { ( \cdot ) } ( \mathbf { v } ) ) ( t ) = \mathbf { h } _ { ( \cdot ) } ( t ) \cdot \mathbf { v }
$$

First, we briefly describe how more specific losses commonly used in generative modelling fall within this framework, before describing general spectral properties of the gradient in terms of these operators.

Examples. Now we briefly provide a more concrete example how two different types of losses affect the same generative model. We consider an SDE whose drift is parameterised by a one-hidden-layer neural network:

$$
\mathbf { d x } = W _ { 2 } \phi ( W _ { 1 } \mathbf { x } ) d t + \sigma \mathbf { d B }
$$

For the sake of illustration we consider two simplified losses:

Drift fitting:

$$
\mathcal { L } ^ { \mathrm { d . f . } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbb { B } } \int _ { 0 } ^ { T } \lVert W _ { 2 } \phi ( W _ { 1 } \bar { \mathbf { x } } ( t ) ) - \mathbf { u } ^ { * } ( \bar { \mathbf { x } } ( t ) ) \rVert ^ { 2 } d t
$$

State fitting:

$$
\mathcal { L } ^ { \mathrm { s . f . } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbb { B } } \int _ { 0 } ^ { T } \| \mathbf { x } ( t ) - \mathbf { x } ^ { * } ( t ) \| ^ { 2 } d t
$$

where $\mathbf { u } ^ { * } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ and $\mathbf { x } ^ { * } : \mathbb { R }  \mathbb { R } ^ { d }$ are the target vector fields and states respectively. These two cases are particularly illustrative as generative flow and score matching are special cases of the former, while variational inference and FFJORD on CNFs are closely related to the latter. Because the drift fitting case is completely independent of ${ \bf x } ( t )$ the solution of the SDE, this is essentially a regression problem. We focus on the first layer’s gradient, since it encodes the learned non-linear features. In that case:

$$
{ \bf h } _ { \mathrm { f } } = { \bf h } _ { G } = 0 \qquad { \bf h } _ { \ell } = \bar { \bf x } ( t ) \qquad { \bf a } _ { \ell } = \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \bar { \bf x } ( t ) ) ) W _ { 2 } ^ { \top } { \bf r } ( \bar { \bf x } ( t ) )
$$

where $\mathbf { r } ( \bar { \mathbf { x } } ( t ) ) = W _ { 2 } \phi ( W _ { 1 } \bar { \mathbf { x } } ( t ) ) - \mathbf { u } ^ { * } ( \bar { \mathbf { x } } ( t ) )$ is the residual, so that the gradient is given by:

$$
\nabla _ { W _ { 1 } } \mathcal { L } ^ { \mathrm { d . f . } } = \mathbb { E } \left[ \int _ { 0 } ^ { T } \left( \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \bar { \mathbf { x } } ( t ) ) ) W _ { 2 } ^ { \top } \mathbf { r } ( \bar { \mathbf { x } } ( t ) ) \right) \otimes \bar { \mathbf { x } } ( t ) d t \right] = A _ { \ell } \circ H _ { \ell }
$$

In the state fitting case:

$$
\mathbf { h } _ { \ell } = \mathbf { h } _ { G } = 0 , \qquad \mathbf { h } _ { \mathbf { f } } = \mathbf { x } ( t ) \qquad \mathbf { a } _ { \mathbf { f } } = \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \mathbf { x } ( t ) ) ) W _ { 2 } ^ { \top } \mathbf { a } ( t )
$$

where ${ \bf a } ( t )$ is the state adjoint, which evolves as:

$$
\mathbf { d a } = - \left( \mathbf { x } ( t ) - \mathbf { x } ^ { * } ( t ) + W _ { 1 } ^ { \top } \left( \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \mathbf { x } ( t ) ) ) \big ( W _ { 2 } ^ { \top } \mathbf { a } ( t ) \big ) \right) \right) d t
$$

The gradient is given by:

$$
\nabla _ { W _ { 1 } } \mathcal { L } ^ { \mathrm { s . f . } } = \mathbb { E } \left[ \int _ { 0 } ^ { T } \left( \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \mathbf { x } ( t ) ) ) W _ { 2 } ^ { \top } \mathbf { a } ( t ) \right) \otimes \mathbf { x } ( t ) d t \right] = A _ { \mathbf { f } } \circ H _ { \mathbf { f } }
$$

we will return to closely related examples in our variational inference and flow matching sections.

## A.2 Proof of theorem 2.3

We’ve demonstrated above that, even with highly non-linear models with general losses, the gradient of weight parameters can be written as a sum of compositions of linear operators.

$$
\nabla _ { W } \mathcal { L } = A _ { \ell } \circ H _ { \ell } + A _ { \mathbf { f } } \circ H _ { \mathbf { f } } + A _ { G } \circ H _ { G }
$$

where $A _ { ( \cdot ) }$ are adjoint operators and $H _ { ( \cdot ) }$ state operators. In case of linear models as used in the main text, the state operator is the state of the system itself. The adjoint operator is directly dependent on the residual. As these operators are linear, and because they are maps from or to a finite dimensional space (the state-space) they are compact. They therefore admit singular value decompositions.

Proposition A.5. The adjoint and state operators admit the singular value decompositions:

$$
{ \cal A } _ { ( \cdot ) } = U ^ { a } \circ S ^ { a } \circ V ^ { a } \qquad H _ { ( \cdot ) } = U ^ { h } \circ S ^ { h } \circ V ^ { h }
$$

where:

$$
U ^ { a } \in \mathbb { R } ^ { k \times k } , \quad S ^ { a } \in \mathbb { R } ^ { k } , \quad V ^ { a } : L ^ { 2 } ( \mathbb { R } ) \to \mathbb { R } ^ { k }
$$

and:

$$
U ^ { h } : \mathbb { R } ^ { q }  L ^ { 2 } ( \mathbb { R } ) , \quad S ^ { h } \in \mathbb { R } ^ { q } , \quad V ^ { h } \in \mathbb { R } ^ { q \times q }
$$

such that their composition is a linear map $\nabla _ { W } \mathcal { L } \in \mathbb { R } ^ { k \times q } ;$

$$
\nabla _ { W } { \mathcal { L } } : \mathbb { R } ^ { q } \xrightarrow { V ^ { h } } \mathbb { R } ^ { q } \xrightarrow { S ^ { h } } \mathbb { R } ^ { q } \xrightarrow { U ^ { h } } L ^ { 2 } ( \mathbb { R } ) \xrightarrow { V ^ { a } } \mathbb { R } ^ { k } \xrightarrow { S ^ { a } } \mathbb { R } ^ { k } \xrightarrow { U ^ { a } } \mathbb { R } ^ { k }
$$

Stable rank. A natural consequence of this structure in the gradient is that if either operator is low rank then the gradient will be low rank as it is generally true that:

$$
\operatorname { R a n k } ( X \circ Y ) \leq \operatorname* { m i n } ( \operatorname { R a n k } ( X ) , \operatorname { R a n k } ( Y ) )
$$

This can for example occur when the model weights are already low rank. In both examples above, we had that the adjoint of the $W _ { 1 }$ matrix had the form:

$$
\mathbf { a } _ { ( \cdot ) } ( t ) = \mathrm { d i a g } ( \phi ^ { \prime } ( W _ { 1 } \mathbf { x } ( t ) ) ) W _ { 2 } ^ { \top } ( \cdot )
$$

Thus the rank of the operator is determined by the $W _ { 2 }$ weights. For linear networks, as considered in the main text, a consequence of this is that the rank of the gradient of model weights bound each other:

$$
\mathrm { R a n k } ( A _ { ( \cdot ) } ) \leq \mathrm { R a n k } ( W _ { 2 } ) \quad \Longrightarrow \quad \mathrm { R a n k } ( \nabla _ { W _ { 1 } } \mathcal { L } ) \leq \mathrm { R a n k } ( W _ { 2 } )
$$

this can cause an entrainment of the weight learning dynamics towards lower-rank solutions.

In practice weights are never exactly low-rank. However, we can characterise their effective rank via the stable rank.

Definition A.6. The stable rank is given by:

$$
\operatorname { S R a n k } ( W ) = { \frac { \| W \| _ { * } } { \| W \| _ { 2 } } } \quad = \quad \sum _ { i = 1 } ^ { \operatorname* { m i n } ( m , k ) } { \frac { s _ { i } } { s _ { 1 } } }
$$

Other notions of effective rank include the participation ratio or the energy $\mathrm { r a n k } ^ { 2 }$ . In general, the stable rank of the product of two matrices can be expressed in terms of the stable rank of each individual matrices.

Definition A.7. Define the Schatten norm of a linear operator:

$$
\| X \| _ { \mathrm { S c } ( r ) } = \mathrm { T r } ( S ^ { r } ) ^ { 1 / r }
$$

where $S$ is that of the SVD of $X$

Proofoftheorem 2.3. For all $r ^ { - 1 } \leq p ^ { - 1 } + q ^ { - 1 }$ the Schatten norm of the composition of linear operators satisfies:

$$
\| X \circ Y \| _ { \mathrm { S c } ( r ) } \leq \| X \| _ { \mathrm { S c } ( q ) } \| Y \| _ { \mathrm { S c } ( p ) }
$$

In particular, taking $r = 1$ and $p = \infty \ \mathrm { ( r e s p . \ 1 ) }$ and $q = 1 \left( \mathrm { r e s p . } \infty \right)$ , yields the bound:

$$
\| X \circ Y \| _ { * } \leq \operatorname* { m i n } \{ \| Y \| _ { 2 } \| X \| _ { * } , \| X \| _ { 2 } \| Y \| _ { * } \}
$$

multiplying and dividing the first and second argument of the min by the spectral norm of $Y$ and X respectively gives:

$$
\| X \circ Y \| _ { * } \leq \| Y \| _ { 2 } \| X \| _ { 2 } \operatorname* { m i n } \{ { \mathrm { S R a n k } } ( X ) , { \mathrm { S R a n k } } ( Y ) \}
$$

or:

$$
\operatorname { S R a n k } ( X \circ Y ) \leq { \frac { \| X \| _ { 2 } \| Y \| _ { 2 } } { \| X \circ Y \| _ { 2 } } } \operatorname* { m i n } \{ \operatorname { S R a n k } ( X ) , \operatorname { S R a n k } ( Y ) \}
$$

Thus, writing for the weight gradient we arrive at Theorem 2.3:

$$
\mathrm { S R a n k } ( \nabla _ { W } \mathcal { L } ) \leq \frac { \| A \| _ { 2 } \| H \| _ { 2 } } { \| A \circ H \| _ { 2 } } \operatorname* { m i n } \left\{ \mathrm { S R a n k } ( A ) , \mathrm { S R a n k } ( H ) \right\} .
$$

Thus, in the examples above, the rank of the gradient of the hidden layer weights are bounded by the rank of the decoder:

$$
\operatorname { S R a n k } ( \nabla _ { W _ { 1 } } { \mathcal { L } } ) \leq \gamma \operatorname { S R a n k } ( W _ { 2 } )
$$

where $\gamma$ depends on the spectral norms of the operators.

The stable rank bound in Theorem 2.3 depends on the prefactor:

$$
\gamma ( A , H ) = { \frac { \| A \| _ { 2 } \| H \| _ { 2 } } { \| A \circ H \| _ { 2 } } }
$$

where $\lVert \cdot \rVert _ { 2 }$ denotes the spectral norm. Therefore, the bound is only tight when $\gamma ( A , H ) \approx 1$ . This occurs when the left singular vectors of H and the right singular vectors of A are aligned.

Using the fact that:

$$
\| A \circ H \| _ { 2 } \geq \sigma _ { 1 } ^ { A } \sigma _ { 1 } ^ { H } | \langle \mathbf { v } _ { 1 } ^ { A } , \mathbf { u } _ { 1 } ^ { H } \rangle |
$$

where $\sigma _ { 1 } ^ { ( \cdot ) }$ are the top singular values, and $\mathbf { v } _ { 1 } ^ { ( \cdot ) }$ and $\mathbf { u } _ { 1 } ^ { ( \cdot ) }$ are the right and left singular vectors of an operator (·) respectively, the prefactor is bounded by:

$$
\gamma ( A , H ) \leq | \langle \mathbf { v } _ { 1 } ^ { A } , \mathbf { u } _ { 1 } ^ { H } \rangle | ^ { - 1 }
$$

## B Collapse in continuous normalising flow

## B.1 Linear CNF adjoint

Proof of theorem 3.2. In the notation of App. $\mathbf { A } , \Phi = - \log p ^ { * }$ and the log-determinant gives the running loss $\ell = - \mathrm { T r } W$ , contributing $\nabla _ { W } \ell = - I$ which is constant and full rank. Here we consider just the drift operators. Following from theorem 2.1, the gradient flow can be written as

$$
\nabla _ { W } \mathcal { L } = A \circ H = \int _ { 0 } ^ { 1 } \mathbf { a } ( t ) \otimes \mathbf { x } ( t ) d t
$$

As previously shown, for a linear model the adjoint has dynamics

$$
\frac { d } { d t } \mathbf { a } ( t ) = - W ^ { \top } \mathbf { a } ( t )
$$

This admits the solution

$$
\mathbf { a } ( t ) = \exp ( \boldsymbol { W } ^ { \top } ( 1 - t ) ) \mathbf { a } ( 1 )
$$

where the terminal condition is given by $\mathbf { a } ( 1 ) = - \nabla _ { \mathbf { x } } \log p ^ { * } ( \mathbf { x } ( 1 ) )$ . The state of the system at time t for the linear model is

$$
\mathbf { x } ( t ) = \exp ( W t ) \mathbf { x } _ { 0 }
$$

Putting these together, we obtain

$$
\mathbf { } A \circ H = \int _ { 0 } ^ { 1 } \exp ( W ^ { \top } ( 1 - t ) ) ( \mathbf { a } ( 1 ) \otimes \mathbf { x } ( 0 ) ) \exp ( W ^ { \top } t ) d t .
$$

## B.2 Model details

Training. In section 3.1 we used a linear continuous normalising flow model. The model is trained to minimize the reverse-KL divergence:

$$
D _ { \mathrm { K L } } ( p _ { 1 } ( \mathbf { x } ) | | p ^ { * } ( \mathbf { x } ) ) = \mathbb { E } _ { \mathbf { x } \sim p _ { 1 } } \left[ \log p _ { 1 } ( \mathbf { x } ) - \log p ^ { * } ( \mathbf { x } ) \right]
$$

Following previous $\mathrm { w o r k } ^ { 3 }$ , we define a parameterisation of the velocity field, $\begin{array} { r } { \frac { d \mathbf { z } } { d t } = f ( \mathbf { z } ( t ) , t ) } \end{array}$ . We can rewrite this using the learned distribution at time t, integrated from $\dot { t } \in [ 0 , 1 \} ] ;$

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { z } ( 0 ) \sim p _ { 0 } } \left[ \log p _ { 0 } ( \mathbf { z } ( 0 ) ) + \int _ { 0 } ^ { 1 } \frac { d } { d t } \log p _ { t } ( \mathbf { z } ( t ) ) d t - \log p ^ { * } ( \mathbf { z } ( 1 ) ) \right] } \\ & { \overset { ( 1 ) } { = } \mathbb { E } _ { \mathbf { z } ( 0 ) \sim p _ { 0 } } \left[ \log p _ { 0 } ( \mathbf { z } ( 0 ) ) - \int _ { 0 } ^ { 1 } \nabla \cdot f ( \mathbf { z } ( t ) , t ) d t - \log p ^ { * } ( \mathbf { z } ( 1 ) ) \right] } \end{array}
$$

where $p _ { t }$ and ${ \bf z } ( t )$ are the learned distribution and pushed forward initial sample at time t, and in (1) we used the instantaneous change of variables formula<sup>4</sup>. To compute gradients with respect to the loss, we define an augmented ODE:

$$
\frac { d { \bf x } ( t ) } { d t } = \left[ \begin{array} { c c } { f ( { \bf z } ( t ) ) } \\ { - \nabla \cdot f ( { \bf z } ( t ) , t ) } \end{array} \right] \qquad { \bf x } ( 0 ) = \left[ \begin{array} { c c } { { \bf z } ( 0 ) } \\ { 0 } \end{array} \right]
$$

and integrate forward in time using a differentiable ODE solver to obtain ${ \bf z } ( 1 )$ and the accumulated log-density of the learned distribution. Backpropagating through the ODE solver obtains the gradients.

We initialised weights $w _ { i j } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { i n i t } } / \sqrt { n } )$ where the dimension $n = 2 m$ and m is the number of initial and target modes. For simulations we used a differentiable ODE solver with the Tsit5 method and adaptive step size. To optimize the system we used SGD in Optax. The hyperparameters are as follows:

<table><tr><td>m</td><td> $\sigma _ { \mathrm { i n i t } }$ </td><td>lr</td><td>Iterations</td><td>Batch size</td><td>Optimizer</td><td>ODE solver</td><td>PID; rtol, atol</td></tr><tr><td>3-10</td><td> $\overline { { 1 e ^ { - 4 } } }$ </td><td> $\overline { { { 1 e ^ { - 3 } } } }$ </td><td>30k</td><td>20k</td><td>SGD</td><td>Tsit5</td><td> $\overline { { 1 e ^ { - 3 } , 1 e ^ { - 6 } } }$ </td></tr></table>

Mode geometry. We construct our mode setup by centring m orthogonal modes of norm R. Let $\mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \ldots , \mathbf { e } _ { m }$ and $\mathbf { f } _ { 1 } , \mathbf { f } _ { 2 } , \ldots , \mathbf { f } _ { m }$ be orthonormal vectors. The initial and target modes are

$$
{ \pmb \mu } _ { i } = R ( { \bf e } _ { i } - \bar { \bf e } ) , \qquad { \pmb \mu } _ { i } ^ { * } = R ( { \bf f } _ { i } - \bar { \bf f } )
$$

where $\bar { \mathbf { e } } = \textstyle { \frac { 1 } { m } } \sum _ { i } \mathbf { e } _ { i }$ and $\begin{array} { r } { \bar { \bf f } = \frac { 1 } { m } \sum _ { i } { \bf f } _ { i } } \end{array}$ . They satisfy:

$$
{ \begin{array} { r l } { \pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } = R ^ { 2 } ( \mathbf { e } _ { i } - \bar { \mathbf { e } } ) \cdot ( \mathbf { e } _ { j } - \bar { \mathbf { e } } ) } & { } \\ { = R ^ { 2 } \left( \delta _ { i j } - { \frac { 1 } { m } } \right) } & { } \end{array} }
$$

and additionally

$$
\sum _ { i } \pmb { \mu } _ { i } = 0 , \qquad \| \pmb { \mu } _ { i } \| = R \sqrt { \frac { m - 1 } { m } } , \qquad \pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } ^ { * } = 0
$$

and likewise for $\pmb { \mu } _ { i } ^ { * }$

The initial means span an $( m - 1 )$ -dimensional subspace S and target means span an $( m - 1 )$ dimensional subspace $T \perp \dot { S }$ . The remaining $d - 2 ( m - 1 )$ directions ⊥ contain no modes.

## B.3 Analyses

Here, we briefly describe the numerical analyses performed on the linear CNF model.

Identifying collapsed modes. In order to see early in training which solution a model will express at convergence, it is useful to consider scalar quantities which appear in the loss and which can tell us about the state of the model. The loss contains

$$
\log p ^ { * } ( { \bf x } ( 1 ) ) = \log \sum _ { j } \mathcal { N } ( { \bf x } ( 1 ) ; \pmb { \mu } _ { j } ^ { * } , \Sigma )
$$

which gives rise to terms of the form

$$
\begin{array} { r } { \| \mathbf { x } ( 1 ) - \pmb { \mu } _ { j } ^ { * } \| ^ { 2 } = \| e ^ { W } ( \pmb { \mu } _ { i } + \mathbf { z } ) - \pmb { \mu } _ { j } ^ { * } \| ^ { 2 } \qquad } \\ { \qquad = \| e ^ { W } \pmb { \mu } _ { i } - \pmb { \mu } _ { j } ^ { * } \| ^ { 2 } + \mathrm { t e r m s ~ d e p e n d e n t ~ o n ~ } \mathbf { z } } \end{array}
$$

for a particular initial mode i, and for noise sample $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I )$ . Normalising this term gives us our summary statistic:

$$
D _ { i j } = \frac { \| e ^ { W } \pmb { \mu } _ { i } - \pmb { \mu } _ { j } ^ { * } \| ^ { 2 } } { R ^ { 2 } }
$$

This quantifies the distance between pushed forward initial modes and target modes, and provides a useful diagnostic for where initial modes are routed by the model. We can equivalently look at

$$
s _ { i j } = \frac { ( e ^ { W } \pmb { \mu } _ { i } ) \cdot \pmb { \mu } _ { j } ^ { * } } { R ^ { 2 } } = - \frac { 1 } { 2 } D _ { i j } + c _ { i }
$$

where $c _ { i }$ is constant across row $i .$ Since at convergence, each initial mode is transported to a target mode, the final state of the statistic is a matrix with one entry per row significantly smaller than the rest (indicating which target each initial mode is sent to). This allows us to compute the assignment of initial modes:

$$
j _ { 0 } ( i ) = \mathrm { a r g m a x } _ { j } s _ { i j }
$$

which gives an m-dimensional vector of integers corresponding to the index of the target mode that each initial mode is assigned. For the perfect solution, this is a permutation of $j _ { 0 } ( i ) = ( 1 , \ldots , m )$ whereas collapsed solutions will have repeated indices.

By computing ${ j _ { 0 } ( i ) }$ at different points across training, we can see which assignment pattern is expressed by the weights. If at train time $\tau _ { \mathrm { l o c k } }$ the $D$ matrix predicts the solution at convergence, then ${ j _ { 0 } ( i ) }$ should be constant from time $\tau _ { \mathrm { l o c k } }$

Visualising the gradient flow near fixed points. In figure B1e we plot the gradient flow around fixed points of the weights for a collapsing and non-collapsing model. We do this by perturbing the converged weights $W ^ { * }$ in directions $- { \pmb { \mu } } _ { i } \otimes { \pmb { \mu } } _ { j } ^ { * }$ and $\pmb { \mu } _ { j } ^ { * } \otimes \pmb { \mu } _ { i }$ , corresponding to directions away from and towards the optimal solution respectively, for a given mode pairing $i , j$ . In the collapsed model the mode indices i and $j$ are chosen such that they correspond to a collapsing initial mode and an unassigned-to target mode. In the non-collapsed model they correspond to an initial mode and its assigned target. The gradient is then computed via auto-differentiation in JAX on a grid of weights along these two basis vectors and then visualised as a flow field.

Eigenvalues of the Hessian. Over training, Hessians are computed with respect to the vectorised weights: $\frac { \partial ^ { 2 } \mathcal { L } } { \partial \mathrm { v e c } ( W _ { i j } ) \mathrm { v e c } ( W _ { k l } ) } \in \mathbb { R } ^ { n ^ { 2 } \times n ^ { 2 } }$ , using auto-differentiation through the ODE solver in JAX. This yields $n ^ { 2 }$ eigenvalues with $n ^ { 2 }$ corresponding eigenvectors of shape $n \times n$ . The histogram of eigenvalues at the two fixed points of a non-collapsed model is plotted in figure B1a.

Tensor decomposition of weights. In figure B1b-c we decomposed the weights over training into a sum of rank-1 tensor products $\begin{array} { r } { \mathbf { \bar { W } } = \sum _ { i = 1 } ^ { r ^ { - } } \mathbf { a } _ { i } \otimes \mathbf { b } _ { i } \otimes \mathbf { c } _ { i } + \boldsymbol { \chi } } \end{array}$ where $\mathbf { a } _ { i } \in \mathbb { R } ^ { k }$ tracks the evolution over training and $\mathbf { b } _ { i } , \mathbf { c } _ { i } \in \mathbb { R } ^ { n \times n }$ are spatial components. We performed the decomposition using tensorly parafac. We then choose two spatial component products of the non-collapsed weights, $\mathbf { b } _ { i } \otimes \mathbf { c } _ { i } ,$ , to project the temporal components onto for both models. The projection components are chosen such that one of them collapses in the collapsed model, giving a separation between the two trajectories. We then plot the projected components over training in the 2-d subspace of the full $n \times n$ weight space.

Computing the operator rank. In figure 2h-i we plot the stable rank of the state and adjoint operators over training. From theorem 2.1 and theorem 3.2, we can write the gradient as a composition of two linear operators, $\nabla _ { W } \mathcal { L } = A \circ H$ , where:

$$
A = - \mathbb { E } _ { \mathbf { x } _ { 0 } } \left[ \int _ { 0 } ^ { T } e ^ { ( 1 - t ) W ^ { \top } } \nabla _ { \mathbf { x } _ { 1 } } \log p ^ { * } ( \mathbf { x } _ { 1 } ) d t \right] , \qquad H = \mathbb { E } _ { \mathbf { x } _ { 0 } } \left[ \int _ { 0 } ^ { 1 } e ^ { t W } \mathbf { x } _ { 0 } d t \right]
$$

where ${ \bf x } _ { 1 } = e ^ { W } { \bf x } _ { 0 }$ . There is a fundamental theorem in functional analysis that says that these operators admit a singular value decomposition. Since $r a n k ( A ) \leq m i n ( d i m ( ( R ^ { n } ) ^ { * } ) = n$ and $\dot { d i m } ( L ^ { 2 } ) = \infty$ , it has at most n non-zero singular values. By discretising the integral with $q$ initial samples and T time points we can write $A \in ( R ^ { n } ) ^ { * } \otimes R ^ { T \times q } ,$ , on which we can perform singular value decomposition and compute the stable rank for each training time. From this, it is trivial to compute the prefactor (Fig. B1d).

## B.4 CNF rank bound prefactor at initialisation

Theorem B.1. In the CNF model defined in Section 3.1, with initial modes $\pmb { \mu } _ { i }$ and target modes $\pmb { \mu } _ { i } ^ { * }$ satisfying $\pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } ^ { * } = 0 , \pmb { \mu } _ { i } \cdot \pmb { \mu } _ { k } = \pmb { \mu } _ { i } ^ { * } \cdot \pmb { \mu } _ { k } ^ { * }$ and with weights initialised to be zero and with well separated modes such that $R \gg \sigma$ , the prefactor can be evaluated as exactly

$$
\gamma ( A , H ) = \frac { \| A \| _ { 2 } \| H \| _ { 2 } } { \| A \circ H \| _ { 2 } } = 1
$$

Proof of Theorem B.1. We consider the well separated regime where mode radius dominates over noise, $R \gg \sigma ,$ , and sampling noise is negligible. In this case, we have:

$$
\mathbf { a } ^ { ( i ) } ( 1 ) = { \pmb { \mu } } _ { i } - { \pmb { \mu } } _ { i } ^ { * } \equiv \Delta _ { i } , \qquad \mathbf { x } ^ { ( i ) } ( 0 ) = { \pmb { \mu } } _ { i }
$$

which yield the discretised operators:

$$
H = \frac { 1 } { \sqrt { m } } \left( \begin{array} { l } { \mu _ { 1 } ^ { \top } } \\ { \vdots } \\ { \mu _ { m } ^ { \top } } \end{array} \right) \in \mathbb { R } ^ { m \times d } , \qquad A = \frac { 1 } { \sqrt { m } } ( \Delta _ { 1 } , \ldots , \Delta _ { m } ) \in \mathbb { R } ^ { d \times m }
$$

To obtain the operator spectral norm we need to compute SVD on the covariance type matrices $H H ^ { \top }$ and $A ^ { \top } A$ . To compute this, we note the following:

C  
b  
a  
![](images/2049f266ccbea3b0921c44350362ec0f7a5e36ff2e85c22482b955f916ff12b0.jpg)

![](images/00ca6106c2129f85cb1b79e1cf5afc58414dbffd3a6440fa1f6215cfc7bd0cfa.jpg)  
d

![](images/5bb744f2b84f1210298d3c66c53e7a560310b2da2262a77db9baa15a38f1bd9e.jpg)

![](images/c3a7cf3e23c9075ed69b92d797f8150c8fd8ac61d69d3a0ea93c9801d96514c8.jpg)

![](images/7e96f70918eaa78f0cf0f69ada7938df6ba69c6ca4277bb9facb4d98c3438294.jpg)  
Figure B1: Low-rank learning dynamics of CNF model. a. Empirical distributions of the eigenvalues of the Hessian, evaluated at the compression slow-point (grey) and at the end of training (blue) for a non-collapsed model. b. MSE of a rank-k CP decomposition of the weight tensor<sup>5</sup> formed by stacking $W$ over training. c. Low-tensor-rank subspace of the weight space highlighting how the trajectory of the collapsed and non-collapsed models split during the mode alignment phase. d. Prefactor during learning for collapsed and non-collapsed models. e. Visualisation of gradient flow near collapsed and non-collapsed fixed points.

Lemma B.2. For any matrix B $, \| B \| _ { 2 } ^ { 2 } = \lambda _ { m a x } ( B B ^ { \top } ) = \lambda _ { m a x } ( B ^ { \top } B )$ because $B ^ { \top } B$ and $B B ^ { \top }$ have the same non-zero eigenvalues.

Now we note that

$$
( H H ^ { \top } ) _ { i j } = \frac { 1 } { m } \mu _ { i } \cdot \mu _ { j } \equiv \frac { 1 } { m } G _ { i j } , \qquad ( A ^ { \top } A ) _ { i j } = \frac { 1 } { m } \Delta _ { i } \cdot \Delta _ { j } = \frac { 2 } { m } G _ { i j }
$$

where both are proportional to $G$ due to the initial and target modes having the same geometry in orthogonal subspaces. This means that by lemma B.2:

$$
\| \boldsymbol { H } \| _ { 2 } ^ { 2 } = \lambda _ { \operatorname* { m a x } } ( \boldsymbol { H } ^ { \top } \boldsymbol { H } ) = \frac { \lambda _ { \operatorname* { m a x } } ( \boldsymbol { G } ) } { m } , \qquad \| \boldsymbol { A } \| _ { 2 } ^ { 2 } = \lambda _ { \operatorname* { m a x } } ( \boldsymbol { A } \boldsymbol { A } ^ { \top } ) = \frac { 2 \lambda _ { \operatorname* { m a x } } ( \boldsymbol { G } ) } { m }
$$

Now we need to find the eigenvalues of the composition of the operators, A ◦ $H = A H$ . The covariance-style matrix is $( A H ) ^ { \top } ( A H ) = H ^ { \top } A ^ { \top } A H = P ^ { \top } P$ where $P = \left( { \textstyle { \frac { 2 } { m } } } G \right) ^ { \frac { 1 } { 2 } } H$ . We then construct the matrix:

$$
P P ^ { \top } = \left( { \frac { 2 } { m } } G \right) ^ { \frac { 1 } { 2 } } H H ^ { \top } \left( { \frac { 2 } { m } } G \right) ^ { \frac { 1 } { 2 } } = { \frac { 2 } { m ^ { 2 } } } G ^ { \frac { 1 } { 2 } } G G ^ { \frac { 1 } { 2 } } = { \frac { 2 } { m ^ { 2 } } } G ^ { 2 }
$$

From lemma B.2 and the fact that for a symmetric positive semi-definite matrix m, $\lambda _ { \operatorname* { m a x } } ( m ^ { 2 } ) =$ $\lambda _ { \operatorname* { m a x } } ( m ) ^ { 2 }$ , we have that

$$
\| A \circ H \| _ { 2 } ^ { 2 } = { \frac { 2 \lambda _ { \operatorname* { m a x } } ( G ^ { 2 } ) } { m ^ { 2 } } } = { \frac { 2 \lambda _ { \operatorname* { m a x } } ( G ) ^ { 2 } } { m ^ { 2 } } }
$$

Now by defining $\lambda = \lambda _ { \mathrm { m a x } } ( G )$ , we can compute the prefactor:

$$
\gamma ( A , H ) = \frac { | | A | | _ { 2 } | | H | | _ { 2 } } { | | A \circ H | | _ { 2 } } = \frac { \sqrt { 2 \lambda / m } \sqrt { \lambda / m } } { \sqrt { 2 \lambda ^ { 2 } / m ^ { 2 } } } = 1
$$

## B.5 Fixed points of the gradient flow

In this section we show that three important points during the training of the linear CNF are fixed points of the gradient flow, namely the optimal solution, collapsed solutions and the compression saddle point. We use an ansatz for each point derived from empirically observed weight matrix structures, showing that they satisfy the fixed point equation.

## B.5.1 Setup

Form of the gradient. First, note that the gradient has the form:

$$
\nabla _ { W } { \mathcal { L } } = \nabla _ { W } D _ { \mathrm { K L } } ( p _ { 1 } , p ^ { * } ) = \nabla _ { W } \mathbb { E } \left[ \log p _ { 1 } ( \mathbf { x } ( 1 ) ) - \log p ^ { * } ( \mathbf { x } ( 1 ) ) \right]
$$

By the continuity equations, for the linear CNF:

$$
\log p ( \mathbf { x } ( 1 ) ) = \log p ( \mathbf { x } ( 0 ) ) - \int _ { 0 } ^ { 1 } \mathrm { d i v } ( W \mathbf { x } ( t ) ) d t = \log p ( \mathbf { x } ( 0 ) ) - \mathrm { T r } ( W )
$$

Hence, because ${ \bf x } ( 0 )$ is independent of $W :$

$$
\nabla _ { W } \mathcal { L } = - \nabla _ { W } \mathbb { E } \left[ \log p ^ { * } ( \mathbf { x } _ { 1 } ) \right] - I
$$

Using Theorem 3.2, the gradient is given by:

$$
\nabla _ { W } \mathcal { L } = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } \mathbf { a } ( 1 ) \otimes \mathbf { x } ( 0 ) e ^ { W ^ { \top } t } d t \right] - I
$$

where $\mathbf { a } ( 1 ) = - \nabla _ { \mathbf { x } } \log p ^ { * } ( \mathbf { x } ( 1 ) )$ ) is the negative score of the data distribution. We are interested in the fixed points $\nabla _ { W } \mathcal { L } = 0$ , that is:

$$
I = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } \mathbf { a } ( 1 ) \otimes \mathbf { x } ( 0 ) e ^ { W ^ { \top } t } d t \right]
$$

Data assumptions. Our initial samples are drawn from $\begin{array} { r } { p _ { 0 } = \sum _ { i } w _ { i } \mathcal { N } ( \pmb { \mu _ { i } } , \sigma ^ { 2 } I _ { d } ) } \end{array}$ , and so we can write $\mathbf { x } ( 0 ) = \pmb { \mu } _ { i } + \sigma \mathbf { z }$ where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ . We use the centred mode setup detailed in App. B.2, with

$$
\mu _ { i } \cdot \mu _ { j } = R ^ { 2 } \left( \delta _ { i j } - { \frac { 1 } { m } } \right) , \qquad \sum _ { i } \mu _ { i } = 0 , \qquad \| \mu _ { i } \| = R { \sqrt { \frac { m - 1 } { m } } }
$$

and likewise for $\pmb { \mu } _ { i } ^ { * }$ , with $\pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } ^ { * } = 0$ . These are modes of a Gaussian mixture with m modes, covariances $\Sigma _ { i }$ . The initial means span an $( m - 1 )$ )-dimensional subspace S and target means span an $( m - 1 )$ )-dimensional subspace $T \perp S$ . The remaining $d - 2 ( m { \stackrel { - } { - } } 1 )$ directions ⊥ contain no modes. Let $Q _ { S }$ and $Q _ { T }$ be $d \times ( m - 1 )$ ) matrices whose columns form an orthonormal basis on $S$ and $T$ respectively. $Q _ { - }$ is a $d \times d - 2 ( m - 1 )$ ) matrix that spans the remaining dimensions. We define modes in their subspace basis as:

$$
\alpha _ { i } = Q _ { S } ^ { \top } \pmb { \mu } _ { i } , \quad \beta _ { i } = Q _ { T } ^ { \top } \pmb { \mu } _ { i } ^ { * } , \quad \sum _ { i } \alpha _ { i } = 0 , \quad \sum _ { i } \alpha _ { i } \otimes \alpha _ { i } = R ^ { 2 } I _ { m - 1 }
$$

We will also deal with terms of the form $\textstyle \sum _ { i } \pmb { \mu } _ { i } \otimes \pmb { \mu } _ { i }$ . Expanding gives:

$$
\begin{array} { c } { { \displaystyle \sum _ { i } { \pmb { \mu } } _ { i } \otimes { \pmb \mu } _ { i } = R ^ { 2 } \sum _ { i } ( { \bf e } _ { i } - \bar { \bf e } ) \otimes ( { \bf e } _ { i } - \bar { \bf e } ) } } \\ { { = R ^ { 2 } \left( \sum _ { i } { \bf e } _ { i } \otimes { \bf e } _ { i } - m \bar { \bf e } \otimes \bar { \bf e } \right) } } \end{array}
$$

The first term is a projection onto the span of $\{ \mathbf { e } _ { 1 } , \ldots . . \mathbf { e } _ { m } \}$ and the second is a projection onto mean. Together they project onto the subspace orthogonal to the mean directions. This is exactly the subspace S, which has projector $Q _ { S } Q _ { S } ^ { \top }$ . Thus we have that:

$$
\sum _ { i } \pmb { \mu } _ { i } \otimes \pmb { \mu } _ { i } = R ^ { 2 } Q _ { S } Q _ { S } ^ { \top }
$$

For a Gaussian Mixture, the negative score can be derived explicitly:

$$
\mathbf { a } ( 1 ) = - \nabla _ { \mathbf { x } } \log p ^ { * } ( \mathbf { x } ( 1 ) ) = - \nabla _ { \mathbf { x } } \log \sum _ { i = 1 } ^ { m } w _ { i } \mathcal { N } ( \mathbf { x } ( 1 ) ; \pmb { \mu _ { i } } ^ { * } , \pmb { \Sigma } _ { i } )
$$

We make the additional assumptions that $w _ { i } = 1 / m$ and $\Sigma _ { i } = \sigma ^ { 2 } I$ . Then:

$$
\nabla _ { \mathbf { x } } \log p ^ { * } ( \mathbf { x } ( 1 ) ) = \sum _ { i = 1 } ^ { m } \gamma _ { i } ( \mathbf { x } ( 1 ) ) ( \pmb { \mu } _ { i } ^ { * } - \mathbf { x } ( 1 ) )
$$

where:

$$
\gamma _ { i } ( \mathbf { x } ( 1 ) ) = \frac { \mathcal { N } ( \mathbf { x } ( 1 ) ; \pmb { \mu } _ { i } ^ { * } , \sigma ^ { 2 } I ) } { \sum _ { j = 1 } ^ { m } \mathcal { N } ( \mathbf { x } ( 1 ) ; \pmb { \mu } _ { j } ^ { * } , \sigma ^ { 2 } I ) }
$$

Fixed point equation for converged weights. We assume that at convergence (either for the optimal or collapsed solutions), $\mathbf { x } ( 1 )$ should be near one of the data modes (otherwise the $D _ { \mathrm { K L } }$ will be large), and if $R \gg \sigma$ , the density of all other Gaussians will be vanishingly small. Thus the softmax weighting in the score becomes a hard assignment:

$$
\gamma _ { i } ( \mathbf { x } ( 1 ) ) \approx \delta _ { i , j _ { 0 } ( i ) }
$$

where ${ j _ { 0 } ( i ) }$ gives the index of the target mode that initial mode i is transported to by the final weights. Thus we focus on the case where:

$$
\nabla _ { \mathbf { x } } \log p ^ { * } ( \mathbf { x } ( 1 ) ) \approx \frac { 1 } { \sigma ^ { 2 } } ( \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } - \mathbf { x } ( 1 ) )
$$

for a sample transported from the $i ^ { \mathrm { { t h } } }$ initial mode. Thus, the fixed point equation can be written as:

$$
- \sigma ^ { 2 } I = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } ( { \pmb \mu } _ { j _ { 0 } ( i ) } ^ { \ast } - e ^ { W } { \bf x } ( 0 ) ) \otimes { \bf x } ( 0 ) e ^ { W ^ { \top } t } d t \right]
$$

We note the identity:

$$
I = \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } e ^ { - W ^ { \top } } e ^ { W ^ { \top } t } d t
$$

which allows us to write the equation as:

$$
\mathcal { F } \left( \mathbb { E } \left[ \left( \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } - e ^ { W } \mathbf { x } ( 0 ) \right) \otimes \mathbf { x } ( 0 ) \right] + \sigma ^ { 2 } e ^ { - W ^ { \top } } \right) = 0
$$

where $\begin{array} { r } { \mathcal { F } ( B ) ~ = ~ \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } B e ^ { W ^ { \top } t } d t } \end{array}$ is the Fréchet derivative of the matrix exponential in a direction B. The fixed point equation is then:

$$
\sigma ^ { 2 } e ^ { - W ^ { \top } } = e ^ { W } \mathbb { E } \left[ { \mathbf { x } } ( 0 ) \otimes { \mathbf { x } } ( 0 ) \right] - \mathbb { E } \left[ \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \otimes { \mathbf { x } } ( 0 ) \right]
$$

$$
\sigma ^ { 2 } e ^ { - W ^ { \top } } = \frac { e ^ { W } } { m } \sum _ { i } \mathbb { E } \left[ \left( \pmb { \mu } _ { i } + \sigma \mathbf { z } \right) \otimes \left( \pmb { \mu } _ { i } + \sigma \mathbf { z } \right) \right] - \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \pmb { \mu } _ { j _ { 0 } \left( i \right) } ^ { * } \otimes \left( \pmb { \mu } _ { i } + \sigma \mathbf { z } \right) \right]
$$

$$
\sigma ^ { 2 } e ^ { - W ^ { \top } } = e ^ { W } \left( \frac { 1 } { m } \sum _ { i } \pmb { \mu } _ { i } \otimes \pmb { \mu } _ { i } + \sigma ^ { 2 } \mathbb { E } [ \mathbf { z } \otimes \mathbf { z } ] \right) - \frac { 1 } { m } \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \otimes \pmb { \mu } _ { i }
$$

$$
\sigma ^ { 2 } e ^ { - W ^ { \top } } = e ^ { W } \left( \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } + \sigma ^ { 2 } I _ { d } \right) - \frac { 1 } { m } \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \otimes \pmb { \mu } _ { i }
$$

$$
\sigma ^ { 2 } e ^ { - W ^ { \top } } = e ^ { W } \Sigma _ { 0 } - \mathcal { T }
$$

where $\Sigma _ { 0 }$ is the second moment of the initial distribution and $\tau$ is a mapping which sends initial subspace directions to target subspace directions.

## B.5.2 Optimal solution

Due to the symmetry of the initial and target mode geometry, an easy guess as to a non-collapsing solution is the rotation of the initial to target means. Write $\mathbf { s } _ { k }$ and $\mathbf { t } _ { k }$ as the $k ^ { \mathrm { { t h } } }$ column of $Q _ { S }$ and $Q _ { T }$ respectively, then such a solution can be written as:

$$
W = U L U ^ { \top }
$$

where $U = [ \mathbf { s } _ { 1 } , \mathbf { t } _ { 1 } , \ldots , \mathbf { s } _ { m - 1 } , \mathbf { t } _ { m - 1 } , Q _ { \perp } ]$ (the columns of $Q _ { \perp }$ are arbitrary since they have the same zero eigenvalue), and L is given by:

$$
L = \mathrm { d i a g } \left( B _ { 1 } , \ldots , B _ { m - 1 } , \mathbf { 0 } \right) , \qquad B _ { i } = { \frac { \pi } { 2 } } { \left( \begin{array} { l l } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} \right) }
$$

Then,

$$
e ^ { B _ { i } } = { \binom { 0 } { 1 } } \quad { \binom { - 1 } { 0 } }
$$

and so $e ^ { W } \mathbf { s } _ { i } = \mathbf { t } _ { i }$ and $e ^ { W } \mathbf { t } _ { i } = - \mathbf { s } _ { i }$ . We now take the RHS of the fixed point equation from above, noting that $e ^ { W } \pmb { \mu } _ { i } = \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \colon$

$$
\begin{array} { r l } { \mathrm { R H S : } } & { { } e ^ { W } \left( \displaystyle \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } + \sigma ^ { 2 } I _ { d } \right) - \displaystyle \frac { 1 } { m } \sum _ { i } \mu _ { j o ( i ) } ^ { * } \otimes \mu _ { i } } \\ { } & { { } = e ^ { W } \left( \displaystyle \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } + \sigma ^ { 2 } I _ { d } \right) - \displaystyle \frac { e ^ { W } } { m } \sum _ { i } \mu _ { i } \otimes \mu _ { i } } \\ { } & { { } = e ^ { W } \left( \displaystyle \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } + \sigma ^ { 2 } I _ { d } \right) - e ^ { W } \displaystyle \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } } \\ { } & { { } = \sigma ^ { 2 } e ^ { W } } \\ { } & { { } = \sigma ^ { 2 } e ^ { - W ^ { \top } } = \mathrm { L H S } } \end{array}
$$

where the last equality is true since $L$ is antisymmetric. Thus, the fixed point equation is satisfied for the perfect solution.

## B.5.3 Collapsed solution

Empirically, we observe collapsed solutions to occur when n pairs of initial modes are each sent to one target mode, with one target mode ignored/unassigned per collapsing pair. The remaining initial modes are sent to one target per mode each, as in the perfect solution. For simplicity, we will consider the case of 1 collapsed pair, with $\pmb { \mu } _ { 1 }$ and $\pmb { \mu } _ { 2 }$ both being sent to $\pmb { \mu } _ { 1 } ^ { * }$ , and $\pmb { \mu } _ { 2 } ^ { * }$ being ignored. Without loss of generality we assume that the $i ^ { \mathrm { { t h } } }$ non-collapsing initial mode is sent to the $i ^ { \mathrm { t h } }$ target mode. The extension to $n > 1$ is discussed at the end.

We begin by evaluating the terms in the fixed point equation for this case:

$$
\Sigma _ { 0 } = \frac { R ^ { 2 } } { m } Q _ { S } Q _ { S } ^ { \top } + \sigma ^ { 2 } I _ { d } \qquad \mathcal { T } = \frac { 1 } { m } \left( \mu _ { 1 } ^ { * } \otimes ( \mu _ { 1 } + \mu _ { 2 } ) + \sum _ { i \geq 3 } \mu _ { i } ^ { * } \otimes \mu _ { i } \right)
$$

We notice that $\Sigma _ { 0 }$ acts on subspace $S$ as $\bigl ( \frac { R ^ { 2 } } { m } + \sigma ^ { 2 } \bigr ) I _ { S }$ and subspace $T$ as $\sigma ^ { 2 } I _ { T }$ , and thus does not mix the subspaces. On the other hand, $\tau$ mixes the subspaces by sending $\pmb { \mu } _ { 1 }$ and $\pmb { \mu } _ { 2 }$ to $\pmb { \mu } _ { 1 } ^ { * }$ , and all other initial modes go to their respective target modes. This structure informs our guess of basis vectors:

$$
\mathbf { p } = { \frac { \mu _ { 1 } + \mu _ { 2 } } { \lVert { \boldsymbol { \mu } } _ { 1 } + { \boldsymbol { \mu } } _ { 2 } \rVert } } , \quad \mathbf { q } = { \frac { \mu _ { 1 } - \mu _ { 2 } } { \lVert { \boldsymbol { \mu } } _ { 1 } - { \boldsymbol { \mu } } _ { 2 } \rVert } } , \quad \mathbf { s } _ { 1 } , \ldots \mathbf { s } _ { m - 3 } \perp \mathbf { p } , \mathbf { q }
$$

where $\begin{array} { r } { \| { \pmb \mu } _ { 1 } + { \pmb \mu } _ { 2 } \| ^ { 2 } = \frac { 2 R ^ { 2 } ( m - 2 ) } { m } , \| { \pmb \mu } _ { 1 } - { \pmb \mu } _ { 2 } \| ^ { 2 } = 2 R ^ { 2 } } \end{array}$ , and $\textbf { p } \perp \textbf { q } . \textbf { s } _ { k }$ are any orthonormal basis of the directions orthogonal to $\pmb { \mu } _ { 1 }$ and $\pmb { \mu } _ { 2 }$ . Together, these form an orthonormal basis on the initial subspace $S$ . We guess that the weights act on these directions independently. To solve for the form of the weights, we will make an ansatz and solve the fixed point equation in each of the basis directions separately, yielding scalar equations. Now we compute the image of each vector under the transformation $\tau$

Shared direction p under $\tau .$ . We use the fact that $\begin{array} { r } { { \pmb \mu } _ { i } \cdot ( { \pmb \mu } _ { 1 } + { \pmb \mu } _ { 2 } ) = R ^ { 2 } \frac { m - 2 } { m } \mathrm { f o r } i = 1 , 2 \mathrm { a n d } - \frac { 2 R ^ { 2 } } { m } } \end{array}$ for $i \geq 3$ to compute:

$$
\begin{array} { r l r } & { } & { \mathcal { T } ( \pmb { \mu } _ { 1 } + \pmb { \mu } _ { 2 } ) = \frac { 2 R ^ { 2 } } { m ^ { 2 } } \left[ ( m - 2 ) \pmb { \mu } _ { 1 } ^ { * } - \sum _ { i \ge 3 } \pmb { \mu } _ { i } ^ { * } \right] } \\ & { } & { = \frac { 2 R ^ { 2 } ( m - 1 ) } { m ^ { 2 } } \left[ \pmb { \mu } _ { 1 } ^ { * } + \frac { 1 } { m - 1 } \pmb { \mu } _ { 2 } ^ { * } \right] } \end{array}
$$

Normalising gives ${ \mathcal { T } } \mathbf { p } = { \boldsymbol { \tau } } \mathbf { u }$ where

$$
\mathbf { u } = { \frac { \mu _ { 1 } ^ { * } + { \frac { 1 } { m - 1 } } \mu _ { 2 } ^ { * } } { \| \mu _ { 1 } ^ { * } + { \frac { 1 } { m - 1 } } \mu _ { 2 } ^ { * } \| } } , \qquad \tau = { \frac { R ^ { 2 } } { m } } { \sqrt { \frac { 2 ( m - 1 ) } { m } } }
$$

and $\begin{array} { r } { \| \pmb { \mu } _ { 1 } ^ { * } + \frac { 1 } { m - 1 } \pmb { \mu } _ { 2 } ^ { * } \| ^ { 2 } = R ^ { 2 } \frac { m - 2 } { m - 1 } } \end{array}$ . Note that u corresponds to $\pmb { \mu } _ { 1 } ^ { * }$ with its component in the $\pmb { \mu } _ { 2 } ^ { * }$ direction removed since $\mathbf { u } \cdot \pmb { \mu } _ { 2 } ^ { * } = 0$

Difference direction q under T . Since $\mathbf { q }$ is orthogonal to p and all other $\pmb { \mu } _ { i \geq 3 }$ , we compute that:

$$
\mathscr { T } ( \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 2 } ) = 0
$$

and so no target pulls on the difference direction (i.e. it stays in subspace S).

Basis directions under T. Since we have chosen ${ \bf s } _ { k } \cdot ( { \pmb \mu } _ { 1 } + { \pmb \mu } _ { 2 } ) = 0$ , the transformation becomes:

$$
\mathcal { T } \mathbf { s } _ { k } = \frac { 1 } { m } \sum _ { i \geq 3 } ( \pmb { \mu } _ { i } ^ { * } \otimes \pmb { \mu } _ { i } ) \mathbf { s } _ { k }
$$

Since in these directions modes are sent to a single target, $\tau$ (in these directions) is identical to in the optimal solution case. We note that $e ^ { W _ { \mathrm { o p t } } } \pmb { \mu } _ { i } = \pmb { \mu } _ { i } ^ { * }$ , we can write:

$$
\mathcal { T } \mathbf { s } _ { k } = e ^ { W _ { \mathrm { o p t } } } \frac { 1 } { m } \left( \sum _ { i } \pmb { \mu } _ { i } \otimes \pmb { \mu } _ { i } \right) \mathbf { s } _ { k } = \frac { R ^ { 2 } } { m } e ^ { W _ { \mathrm { o p t } } } \mathbf { s } _ { k }
$$

and so $\begin{array} { r } { \mathcal { T } \mathbf { s } _ { k } = \frac { R ^ { 2 } } { m } \mathbf { t } _ { k } } \end{array}$ where $\mathbf { t } _ { k }$ form an orthogonal basis on $T .$

Ansatz. We showed that $\left\{ \mathbf { p } , \mathbf { q } , \mathbf { s } _ { 1 } , \dots \mathbf { s } _ { m - 3 } \right\}$ form an orthonormal basis on S. Under the transformation $\boldsymbol { \mathcal { T } } , \mathbf { p }$ is sent to u ∝ $\begin{array} { r } { { \pmb \mu } _ { 1 } ^ { * } + \frac { 1 } { m - 1 } { \pmb \mu } _ { 2 } ^ { * } } \end{array}$ and $\mathbf { s } _ { k }$ is sent to $\mathbf { t } _ { k }$ , both of which lie in the target subspace. To complete the basis in the target subspace, we define $\mathbf { v } = \pmb { \mu } _ { 2 } ^ { * } / \| \pmb { \mu } _ { 2 } ^ { * } \| \operatorname { g i v i n g } \left\{ \mathbf { u } , \mathbf { v } , \mathbf { t } _ { 1 } , \dots , \mathbf { t } _ { m - 3 } \right\}$

We choose the ansatz $e ^ { W } = U e ^ { L } U ^ { \top }$ where $U = [ \mathbf { s } _ { 1 } , \mathbf { t } _ { 1 } , \dots , \mathbf { s } _ { m - 3 } , \mathbf { t } _ { m - 3 } , \mathbf { p } , \mathbf { u } , \mathbf { q } , \mathbf { v } , Q _ { \perp } ]$ is constructed from our orthonormal bases with $Q _ { \perp }$ forming a basis on the directions perpendicular to $S$ and T. We also choose:

$$
e ^ { L } = \mathrm { d i a g } ( E _ { 1 } , \dots , E _ { m - 3 } , E _ { \mathrm { { s h } } } , b , 1 , \mathbf { 1 } ) , \quad E _ { i } = { \binom { 0 } { 1 } } , \quad E _ { \mathrm { { s h } } } = { \binom { 0 } { a } } \quad 0 \quad \mathrm { { } }
$$

where $a , b , \beta$ are unknowns to solve for. Intuitively, $E _ { i }$ rotate the non-collapsing initial modes to their assigned targets, and $E _ { \mathrm { s h } }$ performs some mapping from shared initial direction p to the collapsed-to target direction u, q is scaled by b (since it stays in the initial subspace), and the other directions are ignored (scaled by 1).

The ansatz in the directions mapping $\mathbf { s } _ { k } \to \mathbf { t } _ { k }$ automatically satisfies the fixed point equation since it is identical to the optimal solution. The directions scaled by 1 satisfy $\sigma ^ { 2 } I = \stackrel { \cdot } { \sigma } ^ { 2 } I$ since $\mathbf { v } , Q _ { \perp } \perp S$ For the other directions, we must solve the fixed point equation.

To compute $\beta$ and $^ { a , }$ we solve the fixed point equation in the p and u subspaces, representing the mapping of two initial modes to the same target. For generic vectors s and t in the initial and target subspaces we have $E _ { \mathrm { { s h } } } \mathbf { t } = - \beta \mathbf { s } , E _ { \mathrm { { s h } } } ^ { - \top } \mathbf { t } = - \beta ^ { - 1 } \mathbf { s } , E _ { \mathrm { { s h } } } \mathbf { s } = a \mathbf { t }$ , and $E _ { \mathrm { s h } } ^ { - \top } \mathbf { s } = a ^ { - 1 } \mathbf { t }$ . Acting on the u subspace, the equation becomes:

$$
\begin{array} { r l } & { \sigma ^ { 2 } E _ { \mathrm { s h } } ^ { - \top } \mathbf { t } = E _ { \mathrm { s h } } \Sigma _ { 0 } \mathbf { t } - \mathcal { T } \mathbf { t } } \\ & { - \sigma ^ { 2 } \beta ^ { - 1 } \mathbf { s } = - \sigma ^ { 2 } \beta \mathbf { s } - 0 } \\ & { \quad \implies \beta = 1 } \end{array}
$$

Now acting on the p subspace:

$$
{ \begin{array} { r l } & { \sigma ^ { 2 } E _ { \mathrm { s h } } ^ { - \mathsf { T } } \mathbf { s } = E _ { \mathrm { s h } } \Sigma _ { 0 } \mathbf { s } - { \mathcal { T } } \mathbf { s } } \\ & { \sigma ^ { 2 } a ^ { - 1 } \mathbf { t } = \left( { \frac { R ^ { 2 } } { m } } + \sigma ^ { 2 } \right) a \mathbf { t } - \tau \mathbf { t } } \\ & { ~ \Longrightarrow ~ a = { \frac { \tau + { \sqrt { \tau ^ { 2 } + 4 \sigma ^ { 2 } \left( \sigma ^ { 2 } + { \frac { R ^ { 2 } } { m } } \right) } } } { 2 \left( \sigma ^ { 2 } + { \frac { R ^ { 2 } } { m } } \right) } } } \end{array} }
$$

with τ defined above.

Finally, to compute b we solve the same equation for the difference direction q:

$$
\begin{array} { c } { { \displaystyle \sigma ^ { 2 } e ^ { - W ^ { \top } } { \bf q } = e ^ { W } \Sigma _ { 0 } { \bf q } - \mathcal { T } { \bf q } } } \\ { { \displaystyle \sigma ^ { 2 } b ^ { - 1 } = \left( \sigma ^ { 2 } + \frac { R ^ { 2 } } { m } \right) b } } \\ { { \displaystyle \implies b = \left( \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + \frac { R ^ { 2 } } { m } } \right) ^ { \frac { 1 } { 2 } } } } \end{array}
$$

In sum, we constructed an ansatz for a solution with 1 collapsing pair and solved the fixed point equation to find the specific form of the solution, thus proving it is a fixed point. Table 1 shows the values of $a , b , \beta$ obtained from the derived expressions and from trained weights, showing a close match. The derivation can be extended to $n > 1$ by constructing initial subspace bases as shared directions $\begin{array} { r } { \mathbf p _ { a b } = \frac { \pmb \mu _ { a } + \pmb \mu _ { b } } { \lVert \pmb \mu _ { a } + \pmb \mu _ { b } \rVert } } \end{array}$ and difference directions $\begin{array} { r } { \mathbf q _ { a b } = \frac { \pmb \mu _ { a } - \pmb \mu _ { b } } { \lVert \pmb \mu _ { a } - \pmb \mu _ { b } \rVert } } \end{array}$ for pairs of collapsing initials $( \pmb { \mu } _ { a } , \pmb { \mu } _ { b } )$ . Then the target subspace bases are $\begin{array} { r } { \mathbf { u } _ { \alpha , \beta } = \frac { \pmb { \mu } _ { \alpha } ^ { * } + \frac { 1 } { m - 1 } \pmb { \mu } _ { \beta } ^ { * } } { \lVert \pmb { \mu } _ { \alpha } ^ { * } + \frac { 1 } { m - 1 } \pmb { \mu } _ { \beta } ^ { * } \rVert } } \end{array}$ for collapsed-to target mode $\mu _ { \alpha } ^ { * }$ and ignored target mode $\mu _ { \beta } ^ { * }$ with scaling $\tau _ { \alpha \beta }$ under the transformation $\tau$

<table><tr><td>Analytic</td><td>Computed</td></tr><tr><td>a 1.2750</td><td>1.2748</td></tr><tr><td>b 0.2558</td><td>0.2556</td></tr><tr><td>β 1.0000</td><td>0.9846</td></tr></table>

Table 1: Analytically derived vs numerically computed constants used in collapsed weight ansatz.

## B.5.4 Saddle/slow compression point

During training, we observe the weight trajectories pass through an extended period where they act independently on the initial and target subspaces. We call this the "compression phase". From observations, we use the following ansatz for the weight structure:

$$
W = U L U ^ { \top }
$$

where $U = [ Q _ { S } , Q _ { T } , Q _ { \perp } ]$ ] with $Q _ { T }$ and $Q _ { S }$ the $d \times ( m - 1 )$ matrices whose orthonormal columns span $\{ \mu _ { j } ^ { * } \}$ and $\{ \mu _ { i } \}$ respectively, and $Q _ { \perp }$ an orthonormal basis of the ⊥ subspace. We also define:

$$
L = \mathrm { d i a g } ( \lambda _ { S } I _ { m - 1 } , \lambda _ { T } I _ { m - 1 } , { \bf 0 } )
$$

where $\lambda _ { T } > 0$ and $\lambda _ { S } < 0$ are the scaling factors in the target and initials subspace respectively. In the rest of the section we will solve the fixed point equation for the values of these scaling factors.

The fixed point equation. Since at the compression point, the initial modes are not assigned to their targets, we cannot use the hard assignment limit to simplify the fixed point equation as in the previous sections. We must use the full terminal adjoint which has the form $\begin{array} { r } { \bar { \bf a ( x ( 1 ) ) } = \frac { 1 } { \sigma ^ { 2 } } ( { \bf x ( 1 ) } - \bar { \bf m ( x ( 1 ) ) } ) } \end{array}$ where:

$$
\bar { \mathbf { m } } ( \mathbf { x } ) = \sum _ { j = 1 } ^ { m } \gamma _ { j } ( \mathbf { x } ) \pmb { \mu } _ { j } ^ { * } , \qquad \gamma _ { j } ( \mathbf { x } ) = \frac { \mathcal { N } ( \mathbf { x } ; \pmb { \mu } _ { j } ^ { * } , \sigma ^ { 2 } I _ { d } ) } { \sum _ { k } \mathcal { N } ( \mathbf { x } ; \pmb { \mu } _ { k } ^ { * } , \sigma ^ { 2 } I _ { d } ) }
$$

Similarly to before, we can write the full fixed point equation with the form of the score substituted in (defining ${ \bf x } _ { t } \equiv { \bf x } ( t ) )$ :

$$
\begin{array} { l } { { \displaystyle - \sigma ^ { 2 } I _ { d } = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { W ^ { \top } ( 1 - t ) } \left( \mathbf { m } ( \mathbf { x } _ { 1 } ) - e ^ { W } \mathbf { x } _ { 0 } \right) \otimes \mathbf { x } _ { 0 } e ^ { W ^ { \top } t } d t \right] } \ ~ } \\ { { \displaystyle ~ - \sigma ^ { 2 } I _ { d } = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } U e ^ { L ( 1 - t ) } U ^ { \top } \left( \mathbf { m } ( \mathbf { x } _ { 1 } ) - U e ^ { L } U ^ { \top } \mathbf { x } _ { 0 } \right) \otimes \mathbf { x } _ { 0 } U e ^ { L t } U ^ { \top } d t \right] } \ ~ } \\ { { \displaystyle - \sigma ^ { 2 } U ^ { \top } U = - \sigma ^ { 2 } I _ { d } = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { L ( 1 - t ) } U ^ { \top } \left( \mathbf { m } ( \mathbf { x } _ { 1 } ) - U e ^ { L } U ^ { \top } \mathbf { x } _ { 0 } \right) \otimes \mathbf { x } _ { 0 } U e ^ { L t } d t \right] } } \end{array}
$$

where in the second line we use the ansatz and in the third line we use the fact that $U$ is orthogonal. Now we can write $\mathbf { x } _ { 0 } = \pmb { \mu } _ { i } + \sigma \mathbf { z }$ where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ . Splitting the expectation into a sum over initial modes we get:

$$
- \sigma ^ { 2 } I _ { d } = \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { L ( 1 - t ) } U ^ { \top } \left( \mathbf { m } ( \mathbf { x } _ { 1 } ) - U e ^ { L } U ^ { \top } ( \mu _ { i } + \sigma \mathbf { z } ) \right) \otimes ( \mu _ { i } + \sigma \mathbf { z } ) U e ^ { L t } d t \right]
$$

We also introduce coordinates in the rotated basis, such that

$$
\begin{array} { r } { U ^ { \top } \pmb { \mu } _ { i } = \left( \begin{array} { c } { \alpha _ { i } } \\ { 0 } \\ { 0 } \end{array} \right) , \quad U ^ { \top } \pmb { \mu } _ { i } ^ { * } = \left( \begin{array} { c } { 0 } \\ { \beta _ { i } } \\ { 0 } \end{array} \right) , \quad U ^ { \top } \mathbf { z } = \tilde { \mathbf { z } } = \left( \begin{array} { c } { \mathbf { z } _ { S } } \\ { \mathbf { z } _ { T } } \\ { \mathbf { z } _ { \perp } } \end{array} \right) , \quad U ^ { \top } \mathbf { m } ( \mathbf { x } _ { 1 } ) = \left( \begin{array} { c } { 0 } \\ { \mathbf { m } _ { T } } \\ { 0 } \end{array} \right) } \end{array}
$$

where $\tilde { \mathbf { z } } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ since U is orthogonal and $\begin{array} { r } { { \bf m } _ { T } = \sum _ { i } \gamma _ { j } ( { \bf x } _ { 1 } ) \beta _ { j } } \end{array}$ . We also note that $e ^ { c L } =$ diag $( e ^ { c \lambda _ { S } } I _ { m - 1 } , e ^ { c \lambda _ { T } } I _ { m - 1 } , 1 )$ . Now the fixed point equation becomes:

$$
\begin{array} { c } { { \displaystyle - \sigma ^ { 2 } I _ { d } = \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { L ( 1 - t ) } \left( U ^ { \top } { \bf m } ( { \bf x } _ { 1 } ) - e ^ { L } U ^ { \top } ( \pmb { \mu } _ { i } + \sigma { \bf z } ) \right) \otimes \left( U ^ { \top } ( \pmb { \mu } _ { i } + \sigma { \bf z } ) \right) e ^ { L t } \right] } } \\ { { \displaystyle ~ = \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \left( e ^ { ( 1 - t ) \lambda _ { S } } e ^ { \lambda _ { S } } ( \alpha _ { i } + \sigma { \bf z } _ { S } ) \right) \otimes \left( e ^ { t \lambda _ { S } } ( \alpha _ { i } + \sigma { \bf z } _ { S } ) \right) d t \right] } } \\ { { \displaystyle ~ - \sigma { \bf z } _ { \bot } } } \end{array}
$$

which allows us to split the computation into blocks of $[ S , T , \bot ] \otimes [ S , T , \bot ]$ . We will use the identity:

$$
\int _ { 0 } ^ { 1 } e ^ { ( 1 - t ) \lambda _ { a } } e ^ { t \lambda _ { b } } d t = { \left\{ \begin{array} { l l } { \kappa _ { a b } \equiv { \frac { e ^ { \lambda _ { a } } - e ^ { \lambda _ { b } } } { \lambda _ { a } - \lambda _ { b } } } } & { \lambda _ { a } \not = \lambda _ { b } } \\ { e ^ { \lambda _ { a } } } & { \lambda _ { a } = \lambda _ { b } } \end{array} \right. }
$$

$S \times S$ block. This yields:

$$
\begin{array} { l } { \displaystyle - \sigma ^ { 2 } I _ { m - 1 } = - \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { ( 1 - t ) \lambda _ { S } } e ^ { t \lambda _ { S } } \big ( \alpha _ { i } + \sigma \mathbf { z } _ { S } \big ) \big ( \alpha _ { i } + \sigma \mathbf { z } _ { S } \big ) ^ { \top } d t \right] } \\ { \displaystyle = - e ^ { 2 \lambda _ { S } } \left( \frac { 1 } { m } \sum _ { i } \alpha _ { i } \alpha _ { i } ^ { \top } + \frac { \sigma } { m } \sum _ { i } \mathbb { E } \left[ \alpha _ { i } \mathbf { z } _ { S } ^ { \top } + \mathbf { z } _ { S } \alpha _ { i } ^ { \top } \right] + \sigma ^ { 2 } \mathbb { E } \left[ \mathbf { z } _ { S } \mathbf { z } _ { S } ^ { \top } \right] \right) } \\ { \displaystyle = - e ^ { 2 \lambda _ { S } } \left( \frac { R ^ { 2 } } { m } + \sigma ^ { 2 } \right) I _ { m - 1 } } \\ { \displaystyle \implies \lambda _ { S } = - \frac { 1 } { 2 } \ln \left( \frac { R ^ { 2 } } { \sigma ^ { 2 } } + \sigma ^ { 2 } \right) < 0 } \end{array}
$$

$T \times T$ block. The fixed point equation yields:

$$
\begin{array} { l } { { \displaystyle - \sigma ^ { 2 } I _ { m - 1 } = \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } e ^ { ( 1 - t ) \lambda _ { T } } e ^ { t \lambda _ { T } } \left( { \bf m } _ { T } - e ^ { \lambda _ { T } } \sigma { \bf z } _ { T } \right) \otimes \left( \sigma { \bf z } _ { T } \right) d t \right] } \ ~ } \\ { { \displaystyle ~ = \sigma e ^ { \lambda _ { T } } \mathbb { E } \left[ { \bf m } _ { T } \otimes { \bf z } _ { T } \right] - \sigma ^ { 2 } e ^ { 2 \lambda _ { T } } \mathbb { E } \left[ { \bf z } _ { T } \otimes { \bf z } _ { T } \right] } \ ~ } \\ { { \displaystyle ~ = \sigma e ^ { \lambda _ { T } } \mathbb { E } \left[ { \bf m } _ { T } \otimes { \bf z } _ { T } \right] - \sigma ^ { 2 } e ^ { 2 \lambda _ { T } } I _ { m - 1 } } } \end{array}
$$

Since m $\begin{array} { r } { \mathbf { \Phi } _ { T } = \sum _ { j } \gamma _ { j } ( \mathbf { x } _ { 1 } ) \mathbf { \beta } _ { j } } \end{array}$ depends on $\mathbf { z } _ { T }$ through $\mathbf { x } _ { 1 }$ , we cannot factorise the expectation. Instead, we must use Stein’s lemma which states that for $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I )$

$$
\mathbb { E } \left[ f ( \mathbf { z } ) \mathbf { z } ^ { \top } \right] = \mathbb { E } \left[ \frac { \partial f } { \partial \mathbf { z } } \right] ^ { \top }
$$

for some smooth, bounded function $f .$ . Using standard results for derivatives of the softmax function and the chain rule through the logits, we can compute:

$$
\frac { \partial { \bf m } _ { T } } { \partial { \bf z } _ { T } } = \frac { e ^ { \lambda _ { T } } } { \sigma } \left( \sum _ { j } \gamma _ { j } \beta _ { j } \beta _ { j } ^ { \top } - { \bf m } _ { T } { \bf m } _ { T } ^ { \top } \right)
$$

This expression has the form of a covariance over the targets, weighted by the probability distribution over the targets $\gamma$ . We do not compute it but simply note that since the means and covariances of the target modes are symmetric, and noise $\mathbf { z } _ { T }$ is isotropic, the expected weighted covariance has no preferred direction and so is proportional to the identity. We can then write $\begin{array} { r } { C = \mathbb { E } \left[ \sum _ { j } \gamma _ { j } \beta _ { j } \beta _ { j } - \mathbf { m } _ { T } \mathbf { m } _ { T } ^ { \top } \right] = c I _ { m - 1 } } \end{array}$ . Thus the fixed point equation becomes:

$$
\begin{array} { c } { { - \sigma ^ { 2 } I _ { m - 1 } = e ^ { 2 \lambda _ { T } } c I _ { m - 1 } - \sigma ^ { 2 } e ^ { 2 \lambda _ { T } } I _ { m - 1 } } } \\ { { e ^ { 2 \lambda _ { T } } \left( \sigma ^ { 2 } - c \right) = \sigma ^ { 2 } } } \end{array}
$$

We cannot directly solve this since c depends on $\lambda _ { T }$ through the softmax weightings $\gamma .$ Instead we use a numerical solver, and show at the end of the section that the obtained value closely matches that from training a model.

$S \times T$ block. We have:

$$
\frac 1 m \sum _ { i } \mathbb E \left[ \left( . . . \right) \otimes \mathbf { z } _ { T } \right] = \frac 1 m \sum _ { i } \mathbb E \left[ . . . \right] \otimes \mathbb E \left[ \mathbf { z } _ { T } \right] = 0
$$

because $\mathbf { z } _ { T }$ is noise independent from all $S$ components.

$T \times S$ block. We have:

$$
\frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \left( \ldots \right) \otimes \left( \pmb { \alpha } _ { i } + \sigma \mathbf { z } _ { S } \right) \right] = \mathbb { E } \left[ \ldots \right] \otimes \left( \frac { 1 } { m } \sum _ { i } \pmb { \alpha } _ { i } + \sigma \mathbb { E } \left[ \mathbf { z } _ { S } \right] \right) = 0
$$

since the modes are centred and $\mathbf { z } _ { S }$ is noise independent of all $T$ components.

$\perp \times \perp$ block. This block is simply:

$$
\sigma ^ { 2 } \mathbb { E } \left[ - \mathbf { z } _ { \perp } \otimes \mathbf { z } _ { \perp } \right] = - \sigma ^ { 2 } I _ { d - 2 ( m - 1 ) }
$$

which immediately satisfies the fixed point equation.

Cross blocks involving ⊥. Since $\mathbf { z } _ { \perp }$ is noise independent from all other blocks, all terms involving it evaluate to zero due to the expectation.

We have thus verified that our ansatz satisfies the fixed point equation in each block, and either analytically or numerically found a value for the diagonal scalings which closely match the values found from training the models (Table 2).

<table><tr><td>Analytic</td><td>Computed</td></tr><tr><td> $\lambda _ { S }$  -1.3635</td><td>-1.3629</td></tr><tr><td> $\lambda _ { T }$  0.9674</td><td>0.9669</td></tr><tr><td> $\lambda _ { \perp }$ </td><td>0 -0.0002</td></tr></table>

Table 2: Analytically derived vs numerically computed eigenvalues of the saddle compression point weights for $m = 7 , R = 1 0$

## B.6 Response of statistics to perturbations around compression saddle point

In order to study the behaviour of the training dynamics near the compression saddle, we study how the distance statistics defined in App. B.3 respond to perturbations around the saddle weights, in the direction of the optimal or collapsed solutions.

We consider the compression saddle setup detailed in App. B.5, with saddle weights:

$$
W _ { s } = U L U ^ { \top }
$$

where $U = [ Q _ { S } , Q _ { T } , Q _ { \perp } ]$ has a block structure where $Q _ { S } , Q _ { T }$ are $d \times ( m - 1 )$ dimensional with orthonormal columns spanning $\{ \mu _ { i } \}$ and $\{ \mu _ { i } ^ { * } \}$ respectively and $Q _ { \perp }$ forms an orthonormal basis on the remaining $d - 2 ( m - 1 )$ dimensions. We refer to $S$ and $T$ as the initial and target subspaces spanning $\{ \mu _ { i } \}$ and $\{ \bar { \mu } _ { i } ^ { * } \}$ respectively. L has the block structure:

$$
L = \mathrm { d i a g } ( \lambda _ { S } I _ { m - 1 } , \lambda _ { T } I _ { m - 1 } , { \bf 0 } )
$$

For initial and target modes of constant radius, we can write $D _ { i j } = - 2 s _ { i j } + c _ { i }$ where

$$
s _ { i j } = \frac { \langle e ^ { W } \pmb { \mu } _ { i } , \pmb { \mu } _ { j } ^ { * } \rangle } { R ^ { 2 } }
$$

and $c _ { i }$ is a constant across row i. Next, we consider perturbations of the form

$$
\delta W = \epsilon ( \tilde { B } - \tilde { B } ^ { \top } ) , \qquad \tilde { B } = Q _ { T } B Q _ { S } ^ { \top }
$$

where B is an $( m - 1 ) \times ( m - 1 )$ matrix. δW explicitly mixes the initial and target subspaces, and so depending on the choice of B, perturbs the saddle towards either collapsed or non-collapsed solutions. In practice, we construct the perturbation as:

$$
\tilde { B } = \frac { \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \pmb { \mu } _ { i } ^ { \top } } { \left\| \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \pmb { \mu } _ { i } ^ { \top } \right\| _ { F } }
$$

where ${ j _ { 0 } ( i ) }$ is the chosen assignment pattern. Fig. 2g shows that training a model from saddle weight perturbed by $\delta W$ constructed by different assignment patterns $j _ { 0 }$ can induce different solutions.

At the saddle compression weights $s _ { i j } ~ = ~ 0$ , since $W _ { s }$ acts in the initial and target subspaces independently. Thus, any change to the statistics will come from $\delta W$ mixing initial and target subspaces. We can compute the first order effect of the perturbation on the matrix exponential using the Fréchet derivative in the direction of $\delta W$ , at $W _ { s }$

$$
\begin{array} { l } { { \displaystyle \delta ( e ^ { W } ) = \int _ { 0 } ^ { 1 } e ^ { ( 1 - s ) W _ { s } } \delta W e ^ { s W _ { * } } d s } } \\ { { \displaystyle \quad = \epsilon \int _ { 0 } ^ { 1 } e ^ { ( 1 - s ) W _ { s } } \left( Q _ { T } B Q _ { S } ^ { \top } - Q _ { S } B ^ { \top } Q _ { T } ^ { \top } \right) e ^ { s W _ { s } } d s } } \\ { { \displaystyle \quad = \epsilon \left[ \int _ { 0 } ^ { 1 } e ^ { ( 1 - s ) \lambda _ { T } } e ^ { s \lambda _ { S } } d s Q _ { T } B Q _ { S } ^ { \top } - \int _ { 0 } ^ { 1 } e ^ { ( 1 - s ) \lambda _ { S } } e ^ { s \lambda _ { T } } d s Q _ { S } B ^ { \top } Q _ { T } ^ { \top } \right] } } \\ { { \displaystyle \quad = \epsilon \frac { e ^ { \lambda _ { T } } - e ^ { \lambda _ { S } } } { \lambda _ { T } - \lambda _ { S } } ( Q _ { T } B Q _ { S } ^ { \top } - Q _ { S } B ^ { \top } Q _ { T } ^ { \top } ) } } \\ { { \displaystyle \quad = \epsilon \kappa _ { B } ( \bar { B } - \bar { B } ^ { \top } ) } } \end{array}
$$

where $\begin{array} { r } { \kappa _ { B } = \frac { e ^ { \lambda _ { T } } - e ^ { \lambda _ { S } } } { \lambda _ { T } - \lambda _ { S } } } \end{array}$ is a constant determined by the saddle compression eigenvalues.

We now want to know how the initial modes respond to this perturbation. At the saddle:

$$
e ^ { W _ { s } } \pmb { \mu } _ { i } = e ^ { \lambda _ { S } } \pmb { \mu } _ { i }
$$

with no component in the target direction. The only target component comes from the perturbation which mixes initial into target subspace:

$$
\begin{array} { r } { \delta ( e ^ { W } ) \pmb { \mu } _ { i } = \epsilon \kappa _ { B } ( \tilde { B } - \tilde { B } ^ { \top } ) \pmb { \mu } _ { i } } \\ { = \epsilon \kappa _ { B } Q _ { T } B Q _ { S } ^ { \top } \pmb { \mu } _ { i } } \end{array}
$$

where the second term drops since $\tilde { B } ^ { \top } \pmb { \mu } _ { i } = Q _ { S } B ^ { \top } Q _ { T } ^ { \top } \pmb { \mu } _ { i } = 0$ because $\mu _ { i } \perp T$ . Then we can compute the overlaps to first order:

$$
s _ { i j } = \frac { \langle e ^ { W } \pmb { \mu } _ { i } , \pmb { \mu } _ { j } ^ { * } \rangle } { R ^ { 2 } } \approx \epsilon \kappa _ { B } \frac { \langle Q _ { T } B Q _ { S } ^ { \top } \pmb { \mu } _ { i } , \pmb { \mu } _ { j } ^ { * } \rangle } { R ^ { 2 } } = \epsilon \kappa _ { B } \frac { \langle B \pmb { \alpha } _ { i } , \pmb { \beta } _ { j } \rangle } { R ^ { 2 } }
$$

where the penultimate expression is in ambient coordinates and the final expression is in $T$ coordinates with $\pmb { \alpha } _ { i } = \pmb { Q } _ { S } ^ { \top } \pmb { \mu } _ { i }$ and $\bar { \beta _ { i } } = Q _ { T } ^ { \top } \mu _ { i } ^ { * }$ . Hence, to first order, the overlaps are bilinear in ϵ and B. We confirm that the overlaps respond linearly to perturbation strength ϵ for different B matrices in Fig. B2a by measuring the change in $\| s \| _ { F }$ for varying perturbation strength ϵ.

## B.7 Response of operator ranks to perturbations around saddle

We would like to measure how the rank of the state and adjoint operators change in response to perturbations around the saddle point. Similarly to the response of the summary statistics, we take the saddle weight structure as $W _ { s } \doteq U L U ^ { \top }$ with $U = [ Q _ { S } ^ { - } , Q _ { T } , Q _ { \perp } ]$ and $L \doteq \mathrm { d i a g } ( \lambda _ { S } I _ { m - 1 } , \lambda _ { T } I _ { m - 1 } , \mathbf { 0 } )$ The perturbations are of the form $\delta W = \epsilon ( \tilde { B } - \tilde { B } ^ { \top } )$ with

$$
\tilde { B } = \frac { \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \pmb { \mu } _ { i } ^ { \top } } { \left\| \sum _ { i } \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } \pmb { \mu } _ { i } ^ { \top } \right\| }
$$

where ${ j _ { 0 } ( i ) }$ is the chosen assignment pattern. In the previous section we compute that the matrix exponential of $W = W _ { s } + \delta \bar { W }$ to first order becomes:

$$
\begin{array} { r l } & { e ^ { W _ { s } + \delta W } = e ^ { W _ { s } } + \epsilon \kappa _ { B } ( \tilde { B } - \tilde { B } ^ { \top } ) } \\ & { \qquad = e ^ { W _ { s } } + \epsilon \kappa _ { B } ( Q _ { T } B Q _ { S } ^ { \top } - Q _ { S } B ^ { \top } Q _ { T } ^ { \top } ) } \end{array}
$$

where $\begin{array} { r } { \kappa _ { B } = \frac { e ^ { \lambda _ { T } } - e ^ { \lambda _ { S } } } { \lambda _ { T } - \lambda _ { S } } } \end{array}$ . Thus, we can compute ${ \bf x } _ { 1 } = e ^ { W } { \bf x } _ { 0 }$ to first order as:

$$
\begin{array} { c } { { { \bf x } _ { 1 } = e ^ { W _ { s } + \delta W } ( { \pmb \mu } _ { i } + { \bf z } ) + { \mathcal O } ( { \epsilon } ^ { 2 } ) } } \\ { { = e ^ { W _ { s } } ( { \pmb \mu } _ { i } + { \bf z } ) + \epsilon \kappa _ { B } ( Q _ { T } B Q _ { S } ^ { \top } - Q _ { S } B ^ { \top } Q _ { T } ^ { \top } ) ( { \pmb \mu } _ { i } + { \bf z } ) + { \mathcal O } ( { \epsilon } ^ { 2 } ) } } \\ { { = \left( \begin{array} { c } { { e ^ { \lambda s } ( { \pmb \alpha } _ { i } + { \bf z } _ { S } ) } } \\ { { e ^ { \lambda r } { \bf z } _ { T } } } \\ { { { \bf z } _ { \bot } } } \end{array} \right) + \epsilon \kappa _ { B } \left( \begin{array} { c } { { - B ^ { \top } { \bf z } _ { T } } } \\ { { B ( { \pmb \alpha } _ { i } + { \bf z } _ { S } ) } } \\ { { { \bf 0 } } } \end{array} \right) + { \mathcal O } ( { \epsilon } ^ { 2 } ) } } \\ { { = { \bf x } _ { 1 } ^ { ( 0 ) } + \epsilon { \bf x } ^ { ( 1 ) } + { \mathcal O } ( { \epsilon } ^ { 2 } ) } } \end{array}
$$

where in the penultimate line we split into $[ S , T , \bot ]$ subspaces.

## B.7.1 State operator response

For the linear CNF the state operator covariance is

$$
H ^ { \top } H = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } { \mathbf { x } } ( t ) \otimes { \mathbf { x } } ( t ) d t \right] = \int _ { 0 } ^ { 1 } { e ^ { t W } C _ { H } } { e ^ { t W ^ { \top } } } d t , \quad C _ { H } = \mathbb { E } \left[ { \mathbf { x } } _ { 0 } \otimes { \mathbf { x } } _ { 0 } \right]
$$

where ${ \bf x } ( t ) = e ^ { W t } { \bf x } _ { 0 } = e ^ { W t } ( \pmb { \mu } _ { i } + \bf { z } )$ is a transported initial sample and $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I )$ . We use the above form of the expansion of $\mathbf { x } _ { 1 }$ to first order to compute the covariance $C _ { H }$ . The stable rank depends on the eigenvalues of the state covariance, and so we must propagate the effect of the perturbation $W = \bar { W } _ { s } + \delta W$ to the eigenvalues. We first consider the effects on the terminal covariance $C _ { H }$ , arguing that due to the structure of $W _ { s } ,$ , the same qualitative result holds for the integrated covariance $\check { H } ^ { \top } H$

Zeroth order. $\pmb { \alpha } _ { i } + \mathbf { z } _ { S } , \mathbf { z } _ { T }$ and $\mathbf { z } _ { \perp }$ are independent and have zero mean (the initial modes are centred), so the $S \times T$ and $T \times S$ blocks are zero. The $S \times S$ block is:

$$
\begin{array} { c } { { C _ { H , S S } ^ { ( 0 ) } = \displaystyle \frac { e ^ { 2 \lambda s } } { m } \sum _ { i } \mathbb { E } \left[ ( \alpha _ { i } + { \bf z } _ { S } ) \otimes ( \alpha _ { i } + { \bf z } _ { S } ) \right] } } \\ { { { } } } \\ { { { } = e ^ { 2 \lambda s } \left( \displaystyle \frac { R ^ { 2 } } { m } I _ { m - 1 } + I _ { m - 1 } \right) } } \\ { { { } } } \\ { { { } = e ^ { 2 \lambda s } \left( \displaystyle \frac { R ^ { 2 } } { m } + 1 \right) I _ { m - 1 } } } \end{array}
$$

The $T \times T$ block is:

$$
\begin{array} { c } { C _ { H , T T } ^ { ( 0 ) } = e ^ { 2 \lambda _ { T } } \mathbb { E } \left[ { \bf z } _ { T } \otimes { \bf z } _ { T } \right] } \\ { = e ^ { 2 \lambda _ { T } } I _ { m - 1 } } \end{array}
$$

All cross blocks with ⊥ components are zero because $\mathbf { z } _ { \perp }$ is noise independent of other subspaces. The $\perp \times \perp$ block is $I _ { d - 2 ( m - 1 ) }$

First order. The first order contains the terms:

$$
C _ { H } ^ { ( 1 ) } = \mathbb { E } \left[ { \mathbf { x } ^ { ( 0 ) } \otimes \mathbf { x } ^ { ( 1 ) } + \mathbf { x } ^ { ( 1 ) } \otimes \mathbf { x } ^ { ( 0 ) } } \right]
$$

The $S \times S$ and $T \times T$ blocks are zero since $\mathbf { z } _ { T }$ is independent of $\alpha _ { i }$ and $\mathbf { z } _ { S }$ . The $S \times T$ block is:

$$
\begin{array} { r l } & { C _ { H , S T } ^ { ( 1 ) } = \mathbb { E } \left[ e ^ { \lambda s } ( \pmb { \alpha } _ { i } + \mathbf { z } _ { S } ) \otimes ( B ( \pmb { \alpha } _ { i } + \mathbf { z } _ { S } ) ) - e ^ { \lambda _ { T } } ( B ^ { \top } \mathbf { z } _ { T } ) \otimes \mathbf { z } _ { T } \right] } \\ & { \quad \quad \quad = e ^ { \lambda s } \left( \cfrac { R ^ { 2 } } { m } + 1 \right) I _ { m - 1 } B ^ { \top } - e ^ { \lambda _ { T } } B ^ { \top } I _ { m - 1 } } \\ & { \quad \quad = \left( e ^ { \lambda s } \left( \cfrac { R ^ { 2 } } { m } + 1 \right) - e ^ { \lambda _ { T } } \right) B ^ { \top } } \end{array}
$$

The $T \times S$ block is:

$$
\begin{array} { r l } & { C _ { H , T S } ^ { ( 1 ) } = \mathbb { E } \left[ - e ^ { \lambda _ { T } } \mathbf { z } _ { T } \otimes ( B ^ { \top } \mathbf { z } _ { T } ) + e ^ { \lambda _ { S } } B ( \alpha _ { i } + \mathbf { z } _ { S } ) \otimes ( \alpha _ { i } + \mathbf { z } _ { S } ) \right] } \\ & { \quad \quad \quad = B e ^ { \lambda _ { S } } \left( \frac { R ^ { 2 } } { m } + 1 \right) I _ { m - 1 } - e ^ { \lambda _ { T } } I _ { m - 1 } B } \\ & { \quad \quad \quad = \left( e ^ { \lambda _ { S } } \left( \frac { R ^ { 2 } } { m } + 1 \right) - e ^ { \lambda _ { T } } \right) B } \\ & { \quad \quad \quad = \left( C _ { H , S T } ^ { ( 1 ) } \right) ^ { \top } } \end{array}
$$

Putting together. Now we have the form of the covariance $C _ { H } = C _ { H } ^ { ( 0 ) } + \epsilon C _ { H } ^ { ( 1 ) } + \mathcal { O } ( \epsilon ^ { 2 } )$ where:

$$
C _ { H } ^ { ( 0 ) } = \left( \begin{array} { c c } { { C _ { H , S S } ^ { ( 0 ) } } } & { { { \bf 0 } } } \\ { { { \bf 0 } } } & { { C _ { H , T T } ^ { ( 0 ) } } } \end{array} \right) , \qquad C _ { H } ^ { ( 1 ) } = \left( \begin{array} { c c } { { { \bf 0 } } } & { { C _ { H , S T } ^ { ( 1 ) } } } \\ { { \left( C _ { H , S T } ^ { ( 1 ) } \right) ^ { \top } } } & { { { \bf 0 } } } \end{array} \right)
$$

Hence the block-diagonal part of $C _ { H }$ equals $C _ { H } ^ { ( 0 ) } + { \mathcal { O } } ( \epsilon ^ { 2 } )$ , and by Weyl’s inequality its eigenvalues differ from $C _ { H } ^ { ( 0 ) }$ eigenvalues by $\mathcal { O } ( \epsilon ^ { 2 } )$ . The off-diagonal part is $O ( \epsilon )$ , and since the two diagona blocks have disjoint spectra (eigenvalues ≈ 1 for S and $\approx 6 . 9 2$ for $\dot { T ) }$ , it shifts each eigenvalue by at most $2 \| \Delta \| ^ { 2 } / ( \eta + \sqrt { \eta ^ { 2 } + 4 \| \Delta \| ^ { 2 } } ) = O ( \epsilon ^ { 2 } )$ where η is the gap in the eigenvalues<sup>6</sup>. Every eigenvalue of $C _ { H }$ , and therefore its stable rank, thus changes only at $O ( \epsilon ^ { 2 } )$ and so there is no linear response. The same argument applies to $H ^ { \top } H .$ , since at the saddle $e ^ { s W _ { s } }$ <sup>s</sup> is block diagonal and the first-order change in $e ^ { s W }$ has only off-diagonal blocks, so $H ^ { \top } H ^ { ( 1 ) }$ also has vanishing diagonal blocks. The state rank response is shown in Fig. B2c.

## B.8 Response of adjoint to perturbations around saddle

Next, we would like to measure how the rank of the adjoint operator changes in response to the same perturbations. Analytically, the adjoint covariance is

$$
A A ^ { \mathsf { T } } = \int _ { 0 } ^ { 1 } e ^ { s W ^ { \top } } C _ { A } e ^ { s W } d s , \qquad C _ { A } = \mathbb { E } \left[ \mathbf { a } _ { 1 } \mathbf { a } _ { 1 } ^ { \mathsf { T } } \right] , \qquad \mathbf { a } _ { 1 } = \mathbf { x } _ { 1 } - \sum _ { j } S _ { j } ( \mathbf { x } _ { 1 } ) \boldsymbol { \mu } _ { j } ^ { * } = \mathbf { x } _ { 1 } - \bar { \mathbf { m } } ( \mathbf { x } _ { 1 } ) .
$$

where $\mathbf { a } _ { 1 } = - \nabla _ { \mathbf { x } _ { 1 } } \log p ^ { * } ( \mathbf { x } _ { 1 } )$ is the adjoint terminal state and $\mathbf { x } _ { 1 } = e ^ { W } \mathbf { x } _ { 0 } = e ^ { W } ( \pmb { \mu } _ { i } + \pmb { \mathrm { z } } )$ is a pushed forward initial sample from the $i ^ { \mathrm { { \tilde { t h } } } }$ initial mode. Similarly to before, the stable rank depends on the eigenvalues of the adjoint covariance, and so we must propagate the effect of the perturbation $W = W \overline { { s } } + \delta W$ to the eigenvalues through the adjoint.

Using the definition of the expansion of $\mathbf { x } _ { 1 }$ in the last section, we compute the adjoint terminal state, starting with $\mathbf { a } _ { 1 } = \mathbf { a } _ { 1 } ^ { ( 0 ) } + \mathbf { a } _ { 1 } ^ { ( 1 ) } + \mathcal { O } ( \epsilon ^ { 2 } )$ . From the chain rule we have that $\begin{array} { r } { \mathbf { a } _ { 1 } ^ { ( 1 ) } = \frac { \partial \mathbf { a } _ { 1 } } { \partial \mathbf { x } _ { 1 } } \mathbf { x } _ { 1 } ^ { ( 1 ) } = } \end{array}$ $( I - J ( \mathbf { x } _ { 1 , T } ) ) \mathbf { x } _ { 1 } ^ { ( 1 ) }$ where $\begin{array} { r } { J ( { \bf x } _ { 1 , T } ) = \frac { \partial \bar { \bf m } ( { \bf x } _ { 1 , T } ) } { \partial { \bf x } _ { 1 } } = \sum _ { j } S _ { j } \mu _ { j } ^ { * } \mu _ { j } ^ { * \top } - \bar { \bf m } ( { \bf x } _ { 1 , T } ) \bar { \bf m } ( { \bf x } _ { 1 , T } ) ^ { \top } } \end{array}$ which acts only in the T subspace. We can then write the components of the adjoint as:

$$
\mathbf { a } _ { 1 } ^ { ( 0 ) } = \left( e ^ { \lambda s } \mathbf { \cong } \frac { e ^ { \lambda s } \left( \alpha _ { i } + \mathbf { z } _ { S } \right) } { \mathbf { \cong } \pi _ { T } - \mathbf { \neq } \pi _ { 1 } } \right) , \qquad \mathbf { a } _ { 1 } ^ { ( 1 ) } = \epsilon \kappa _ { B } \left( ( I - J ( e ^ { \lambda T } \mathbf { z } _ { T } ) ) B ( \alpha _ { i } + \mathbf { z } _ { S } ) \right)
$$

Now, a similar expansion of the adjoint covariance yields $C = C ^ { ( 0 ) } + C ^ { ( 1 ) } + \mathcal { O } ( \epsilon ^ { 2 } )$ . First we consider the zeroth-order term $C ^ { ( 0 ) }$ , noting that the off-diagonal blocks are zero:

$$
C _ { S T } ^ { ( 0 ) } = \mathbb { E } \left[ \mathbf { a } _ { 1 , S } ^ { ( 0 ) } \mathbf { a } _ { 1 , T } ^ { ( 0 ) \top } \right] = \mathbb { E } \left[ e ^ { \lambda _ { S } } ( \alpha _ { i } + \mathbf { z } _ { S } ) \left( e ^ { \lambda _ { T } } \mathbf { z } _ { T } - \bar { \mathbf { m } } ( e ^ { \lambda _ { T } } \mathbf { z } _ { T } ) \right) ^ { \top } \right] = \mathbf { 0 }
$$

since $\mathbb { E } [ { \pmb { \alpha } } _ { i } + { \bf z } _ { S } ] = { \bf 0 }$ . We are interested in the first order response to the perturbation, so we consider the terms:

$$
C ^ { ( 1 ) } = \mathbb { E } \left[ { \bf a } ^ { ( 0 ) } { \bf a } ^ { ( 1 ) \top } + { \bf a } ^ { ( 1 ) } { \bf a } ^ { ( 0 ) \top } \right]
$$

Using the previous results, we can compute this quantity block by block.

The initial subspace block:

$$
\mathbb { E } \left[ { \bf a } _ { S } ^ { ( 0 ) } { \bf a } _ { S } ^ { ( 1 ) \top } \right] = - \frac { \epsilon \kappa _ { B } } { m } \sum _ { i } \mathbb { E } \left[ e ^ { \lambda _ { S } } ( { \bf a } _ { i } + { \bf z } _ { S } ) { \bf z } _ { T } ^ { \top } B \right] = { \bf 0 }
$$

due to the fact that $\mathbf { z } _ { T }$ is zero-centred.

The target subspace block:

$$
\mathbb { E } \left[ \mathbf { a } _ { T } ^ { ( 0 ) } \mathbf { a } _ { T } ^ { ( 1 ) \top } \right] = \frac { 1 } { m } \sum _ { i } \mathbb { E } \left[ \epsilon \kappa _ { B } \left( e ^ { \lambda _ { T } } \mathbf { z } _ { T } - \bar { \mathbf { m } } ( e ^ { \lambda _ { T } } \mathbf { z } _ { T } ) \right) ( \pmb { \alpha } _ { i } + \mathbf { z } _ { S } ) ^ { \top } B ^ { \top } \left( I - J ( e ^ { \lambda _ { T } } \mathbf { z } _ { T } ) \right) ^ { \top } \right] = \mathbf { 0 }
$$

due to the centring of $\alpha _ { i }$ and independence of noise in initial and target subspaces.

Blocks involving ⊥ dimensions: Trivially zero since $\mathbf { z } _ { \perp }$ is zero-mean and independent of all other components.

Initial-target cross blocks: In general non-zero due to terms like $\mathbb { E } \left[ ( { \pmb { \alpha } } _ { i } + { \bf z } _ { S } ) ( { \pmb { \alpha } } _ { i } + { \bf z } _ { S } ) ^ { \top } \right]$ , and proportional to ϵ. Numerically computed to be small.

The first order change in stable rank of the adjoint operator will depend on the first order change in the eigenvalues of $C ^ { ( 1 ) }$ . This has a block structure with zero blocks on the diagonal and small, non-zero off-diagonal blocks proportional to ϵ. Since we have shown that $C ^ { ( 0 ) }$ is block diagonal, we have the following structure:

$$
C ^ { ( 0 ) } = \left( \begin{array} { c c } { { C _ { S } } } & { { { \bf 0 } } } \\ { { { \bf 0 } } } & { { C _ { T } } } \end{array} \right) , \qquad C ^ { ( 1 ) } = \left( \begin{array} { c c } { { { \bf 0 } } } & { { \Delta } } \\ { { \Delta ^ { \top } } } & { { { \bf 0 } } } \end{array} \right) , \qquad \Delta \propto \epsilon
$$

By the same argument as the state operator, the stable rank responds at $\mathcal { O } ( \epsilon ^ { 2 } )$ . The non-linear response of the maximum eigenvalues of $\bar { C }$ and the rank of the adjoint operator can be seen empirically in Fig. B2b.

## C Effective collapse in flow matching models

The goal of flow matching is to learn some function $\mathbf { u } _ { t } ^ { \theta } ( \mathbf { x } ) \equiv \mathbf { u } ^ { \theta } ( \mathbf { x } , t )$ which transports samples $\mathbf { x } ( 0 ) \sim p _ { 0 }$ , where $p _ { 0 }$ is a simple distribution (typically Gaussian), to $\mathbf { x } ( 1 ) \sim p ^ { * }$ via integration:

$$
\mathbf { x } _ { 1 } = \mathbf { x } _ { 0 } + \int _ { 0 } ^ { 1 } \mathbf { u } _ { t } ^ { \theta } ( \bar { \mathbf { x } } ( t ) ) d t
$$

![](images/756001362e74b862803b66cbc2224c4935261a418fc15285bf7a01a50d7db5aa.jpg)

![](images/b1e0036460a6aa0eb4a0c5bf31d671a86de94cc931c0f0236606fbed9ab3cd3f.jpg)

![](images/65fc598dafcd2fb81018e7bb0d7acabeb1e1bde61b72325faf52df19df42ae22.jpg)  
Figure B2: Response to perturbations around the compression saddle weights. a. Magnitude of distance statistics matrix in response to perturbation strength around saddle point. Closely matches linear trend. b-c. Change in rank of the adjoint and state operators in response to different perturbation strengths.

where $\bar { { \bf x } } ( t )$ follows a chosen “interpolant trajectory”. To do this, we define the marginal probability path $p _ { t }$ , which smoothly interpolates between the initial and data distribution, satisfying $p _ { t = 0 } = p _ { 0 }$ and $p _ { t = 1 } = p ^ { * }$ . The flow matching objective can then be written generally as

$$
\mathcal { L } _ { \mathrm { F M } } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { \mathbf { x } \sim p _ { t } } \left[ | | \mathbf { u } _ { t } ^ { * } ( \mathbf { x } ) - \mathbf { u } _ { t } ^ { \theta } ( \mathbf { x } ) | | ^ { 2 } \right] d t
$$

where $\mathbf { u } _ { t } ^ { * } ( \mathbf { x } )$ is the optimal velocity field that generates $p _ { t }$ . In general, the optimal velocity is intractable, and so in practice the conditional flow matching objective is used:

$$
\mathcal { L } _ { \mathrm { C F M } } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { \mathbf { z } \sim p ^ { * } , \mathbf { x } \sim p _ { t } ( \cdot \vert \mathbf { z } ) } \left[ | | \mathbf { u } _ { t } ^ { * } ( \mathbf { x } \mid \mathbf { z } ) - \mathbf { u } _ { t } ^ { \theta } ( \mathbf { x } ) | | ^ { 2 } \right] d t
$$

where $p _ { t } ( \cdot \mid \mathbf { z } )$ is the conditional probability path which is related to the marginal by $\begin{array} { r } { p _ { t } ( \mathbf { x } ) = \int p _ { t } ( \mathbf { x } \mid } \end{array}$ ${ \bf z } ) p ^ { * } ( { \bf z } ) d { \bf z }$ , and similarly $\mathbf { u } _ { t } ^ { * } ( \mathbf { x } ) \bar { \mathbf { \xi } } = \mathbb { E } \left[ \mathbf { u } _ { t } ^ { * } ( \bar { \mathbf { x } } \mid \mathbf { z } ) \mid \bar { \mathbf { x } } ( t ) = \mathbf { x } \right]$ relates the marginal and conditional velocity fields. There is a fundamental theorem in flow matching which says that the conditional flow matching loss has the same gradients as the flow matching loss<sup>7</sup>. From the conditional probability path we can easily compute the target vector field by taking the time derivative, ${ \mathbf { u } } _ { t } ^ { * } ( \bar { \mathbf { x } } ( t ) \mid \mathbf { z } ) = \dot { \mathbf { x } } ( t )$ which we can plug into the conditional loss and train our model.

## C.1 Linear flow matching

Model. To compare the FM objective to the linear CNF model, we choose $\mathbf { u } _ { t } ^ { \theta } ( \mathbf { x } ) = W \mathbf { x }$ . As before, because linear flows cannot split or join modes, we define our base and target distributions to be mixtures of Gaussians:

$$
p _ { 0 } ( { \bf x } ) = \sum _ { i = 1 } ^ { m } w _ { i } \mathcal { N } ( { \bf x } ; { \pmb \mu } _ { i } , \sigma ^ { 2 } I _ { d } ) , \qquad p ^ { * } ( { \bf x } ) = \sum _ { j = 1 } ^ { m } \pi _ { j } \mathcal { N } ( { \bf x } ; { \pmb \mu } _ { j } ^ { * } , \sigma ^ { 2 } I _ { d } )
$$

We now choose all modes to be orthogonal:

$$
\pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } ^ { * } = 0 , \qquad \pmb { \mu } _ { i } \cdot \pmb { \mu } _ { j } = \pmb { \mu } _ { i } ^ { * } \cdot \pmb { \mu } _ { j } ^ { * } = R ^ { 2 } \delta _ { i j }
$$

In $\mathrm { A p p . C . 6 } $ we prove that for such a setup, the standard linear interpolation probability path yields an optimal solution which contains a non-linear weighting over modes, and thus our linear model cannot represent it. To account for this, we consider a teacher student setup where the optimal velocity field is $\mathbf { u } _ { t } ^ { * } ( \mathbf { x } ) = W ^ { * } \mathbf { x }$ and the probability path is $\bar { \mathbf { x } } ( t ) = e ^ { t W ^ { * } } \mathbf { x } ( 0 )$ (i.e. the solution of the dynamics). We define $\dot { W } ^ { * }$ as the generator matrix of a rotation which maps initial modes to their respective targets:

$$
\boldsymbol { W } ^ { * } = \frac { \pi } { 2 } ( \boldsymbol { B } - \boldsymbol { B } ^ { \top } ) , \qquad \boldsymbol { B } = \frac { 1 } { R ^ { 2 } } \sum _ { i } \boldsymbol { \mu } _ { j _ { 0 } ( i ) } ^ { * } \boldsymbol { \mu } _ { i } ^ { \top }
$$

where ${ j _ { 0 } ( i ) }$ is the correct assignment pattern, which we choose without loss of generality to be $j _ { 0 } ( i ) = i$ . Below, we briefly show that this defines a valid probability path.

Rotational probability path. We start by defining

$$
p _ { 0 } ^ { ( i ) } = \mathcal { N } ( \pmb { \mu } _ { i } , \sigma ^ { 2 } I _ { d } ) , \qquad p ^ { * ( j ) } ( \mathbf { x } ) = \mathcal { N } ( \pmb { \mu } _ { j } ^ { * } , \sigma ^ { 2 } I _ { d } )
$$

as the densities of Gaussian mixtures around each mode. At sampling time, we choose an $i , j$ pair according to weights $w _ { i } , \pi _ { j }$ and draw $\mathbf { x } ^ { ( i ) } ( 0 ) = \pmb { \mu } _ { i } + \sigma \mathbf { z } _ { 0 }$ and $\mathbf { x } ^ { ( j ) } ( 1 ) = \pmb { \mu } _ { j } ^ { * } + \sigma \mathbf { z } _ { 1 }$ where $\mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ . Next, we construct the conditional path sample as:

$$
\begin{array} { c } { \bar { \mathbf { x } } ( t ) = e ^ { t W ^ { * } } \mathbf { x } ^ { ( i ) } ( 0 ) } \\ { = e ^ { t W ^ { * } } ( \pmb { \mu } _ { i } + \sigma \mathbf { z } _ { 0 } ) } \\ { = e ^ { t W ^ { * } } \pmb { \mu } _ { i } + \sigma \tilde { \mathbf { z } } _ { 0 } } \end{array}
$$

where in the last line we used the fact that since $W ^ { * }$ is skew-symmetric $( W ^ { * \top } = - W ^ { * } ) , e ^ { t W ^ { * } }$ is orthogonal and so $\tilde { \mathbf { z } } _ { 0 } = e ^ { t W ^ { * } } \mathbf { z } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ . From this expression we have that $\bar { \mathbf { x } } ( 0 ) = \pmb { \mu } _ { i } + \sigma \tilde { \mathbf { z } } _ { 0 } \sim$ $p _ { 0 } ^ { ( i ) }$ , and $\bar { \mathbf { x } } ( 1 ) = \pmb { \mu } _ { j _ { 0 } ( i ) } ^ { * } + \sigma \tilde { \mathbf { z } } _ { 0 } \sim p ^ { * ( j ) }$ from the definition of $W ^ { * }$ . Thus, $\bar { { \bf x } } ( t )$ defines a valid probability path.

Training. From the rotational probability path, we can compute the optimal conditional velocity as ${ \dot { \mathbf { x } } } ( t ) = W ^ { * } e ^ { t W ^ { * } } \mathbf { x } ( 0 ) = W ^ { * } { \bar { \mathbf { x } } } ( t )$ , and so we can write the objective as a linear regression integrated over time:

$$
\mathcal { L } = \mathbb { E } _ { \mathbf { x } ( 0 ) \sim p _ { 0 } } \left[ \int _ { 0 } ^ { 1 } | | W ^ { * } \bar { \mathbf { x } } ( t ) - W \bar { \mathbf { x } } ( t ) | | ^ { 2 } d t \right]
$$

To train the model, we use full-batch gradient descent with a fixed set of initial samples and time points. We initialise weights as $W _ { i j } \sim \mathbf { \bar { \mathcal { N } } } ( 0 , \sigma _ { \mathrm { i n i t } } / \sqrt { n } )$ . The hyperparameters are:

<table><tr><td>n</td><td> $\sigma _ { \mathrm { i n i t } }$ </td><td>lr</td><td>Iterations</td><td>Batch size</td><td>Optimizer</td></tr><tr><td>3-10</td><td> $\overline { { 1 e ^ { - 4 } } }$ </td><td> $\overline { { 1 e ^ { - 3 } } }$ </td><td>3000</td><td>10k</td><td>SGD</td></tr></table>

In figure 4 we parameterise the linear flow as $\mathbf { u } _ { \theta } ( \mathbf { x } ) = W _ { 2 } W _ { 1 } \mathbf { x }$ . Classic work in learning theory<sup>8</sup> has shown that this parameterisation leads to non-linear training dynamics, inducing saddle-to-saddle dynamics.

## C.2 Spectrum of the state operator

Data assumptions. As described in the previous section, we consider orthogonal modes radius R with mode pairing $\pmb { \mu } _ { i }  \pmb { \mu } _ { i } ^ { * }$ . We define the optimal weights as:

$$
W ^ { * } = \frac { 1 } { R ^ { 2 } } \frac { \pi } { 2 } \sum _ { i } \left( \pmb { \mu } _ { i } ^ { * } \pmb { \mu } _ { i } ^ { \top } - \pmb { \mu } _ { i } \pmb { \mu } _ { i } ^ { * \top } \right)
$$

with interpolant path $\bar { \mathbf { x } } ( t ) = e ^ { t W ^ { * } } \mathbf { x } ( 0 )$ and target $\mathbf { u } _ { t } ^ { * } = W ^ { * } \bar { \mathbf { x } } ( t )$ . Also assume $w _ { i } = \pi _ { i }$ so that the path ends at $p ^ { * }$

Covariance operator of the linear flow matching objective. To decompose the gradient of the FM objective into two linear operators, we note that the gradient (up to a scalar constant) is:

$$
\nabla _ { W } \mathcal { L } = - \mathbb { E } _ { \mathbf { x } ( 0 ) } \left[ \int _ { 0 } ^ { 1 } ( W ^ { * } - W ) \bar { \mathbf { x } } ( t ) \bar { \mathbf { x } } ( t ) ^ { \top } d t \right]
$$

which we can rewrite as the composition of two linear operators:

$$
\nabla _ { W } \mathcal { L } = A \circ H = \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathbf { r } ( t ) \otimes { \bar { \mathbf { x } } } ( t ) d t \right]
$$

where the state operator H is given by the interpolant trajectory $\bar { { \bf x } } ( t )$ , and the adjoint operator A by the residual, $\mathbf { r } ( t ) { \bar { \mathbf { \rho } } } = W { \bar { \mathbf { x } } } ( t ) - \mathbf { u } _ { t } ^ { * } ( { \bar { \mathbf { x } } } ( t ) )$ . We can compute the rank of H analytically. For this, we will compute the eigenvalues of the matrix:

$$
C = \mathbf { a d } ( H ) \circ H = \mathbb { E } _ { \mathbf { x } ( 0 ) } \left[ \int _ { 0 } ^ { 1 } \bar { \mathbf { x } } ( t ) \otimes \bar { \mathbf { x } } ( t ) d t \right] \in R ^ { n \times n }
$$

Spectrum of the state operator. Substituting the form of $\bar { \mathbf { x } } _ { t }$ yields:

$$
\begin{array} { l } { { \displaystyle C = \mathbb { E } _ { \mathbf { x } ( 0 ) } \left[ \int _ { 0 } ^ { 1 } \left( e ^ { t W ^ { * } } \mathbf { x } ( 0 ) \right) \otimes \left( e ^ { t W ^ { * } } \mathbf { x } ( 0 ) \right) d t \right] } } \\ { { \displaystyle \quad = \sum _ { i } w _ { i } \mathbb { E } _ { z \sim N ( 0 , I _ { d } ) } \left[ \int _ { 0 } ^ { 1 } \left( e ^ { t W ^ { * } } ( \mu _ { i } + \sigma \mathbf { z } ) \right) \otimes \left( e ^ { t W ^ { * } } ( \mu _ { i } + \sigma \mathbf { z } ) \right) d t \right] } } \\ { { \displaystyle \quad = \sum _ { i } w _ { i } \int _ { 0 } ^ { 1 } \left( ( e ^ { t W ^ { * } } \mu _ { i } ) \otimes ( e ^ { t W ^ { * } } \mu _ { i } ) + \sigma ^ { 2 } e ^ { t W ^ { * } } \mathbb { E } \left[ \mathbf { z } \otimes \mathbf { z } \right] e ^ { t W ^ { * } \tau } \right) d t } } \\ { { \displaystyle \quad = \sigma ^ { 2 } I _ { d } + \sum _ { i } w _ { i } \int _ { 0 } ^ { 1 } ( e ^ { t W ^ { * } } \mu _ { i } ) \otimes ( e ^ { t W ^ { * } } \mu _ { i } ) d t } } \end{array}
$$

where we split the expectation into a sum over samples from each initial mode: $\mathbf { x } ^ { ( i ) } ( 0 ) = \pmb { \mu } _ { i } + \sigma \mathbf { z }$ where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ , and we used the fact that $\mathbb { E } _ { \mathbf { z } } [ \mathbf { z } ] = 0$ to cancel the cross terms, $\mathbb { E } [ \mathbf { z } \otimes \mathbf { z } ] = I _ { d }$ and $W ^ { * \top } = - W ^ { * }$ such that $e ^ { t W ^ { * } } e ^ { t W ^ { * } } = I _ { d }$

Since $W ^ { * }$ is skew-symmetric, $e ^ { t W ^ { * } }$ is orthogonal. The optimal rotation acts on plane i by rotating the initial mode to the target mode, so $e ^ { t W ^ { * } } \pmb { \mu } _ { i } = \cos ( \frac { \pi t } { 2 } ) \bar { \pmb { \mu } _ { i } } + \sin ( \frac { \pi t } { 2 } ) \pmb { \mu } _ { i } ^ { * }$ . Thus the $n \times n$ covariance matrix splits into $2 \times 2$ blocks, written in the basis of $\textstyle { \bigl ( } { \frac { \mu _ { i } } { R } } , { \frac { \mu _ { i } ^ { * } } { R } } { \bigr ) }$

$$
\begin{array} { l c r } { { C _ { i } = \sigma ^ { 2 } I _ { 2 } + w _ { i } R ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } \left( \cos ( \frac { \pi t } { 2 } ) \phantom { \frac { \sigma ^ { 2 } } { 2 } } \right. \nonumber } } & { { \left. \cos ( \frac { \pi t } { 2 } ) \sin ( \frac { \pi t } { 2 } ) \right) d t } } \\ { { \phantom { C _ { i } = } } } & { { \phantom { - } } } & { { \phantom { - } } } \\ { { \phantom { C _ { i } = \sigma ^ { 2 } I _ { 2 } + w _ { i } R ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } \left( \frac { \cos ^ { 2 } ( \frac { \pi t } { 2 } ) } { \pi } \phantom { \frac { \sigma ^ { 2 } } { 2 } } \right) } } } & { { \phantom { - } } } \end{array}
$$

Thus we can compute the eigenvalues and eigenvectors for each block as

$$
\lambda _ { i , \pm } = \sigma ^ { 2 } + w _ { i } R ^ { 2 } \left( \frac { 1 } { 2 } \pm \frac { 1 } { \pi } \right) , \qquad { \bf v } _ { i , \pm } = \frac { 1 } { \sqrt { 2 } } \left( \pm 1 \right)
$$

and $C = \sigma ^ { 2 }$ in directions with no modes.

Since we define $C = \mathop { \mathrm { a d } } ( H ) \circ H$ , the singular values of H are the square roots of the eigenvalues of $C \colon s _ { k } ( H ) = \sqrt { \lambda _ { k } }$ . We are interested in the stable rank of H, which is:

$$
\mathrm { S R a n k } ( H ) = \frac { \sum _ { k } \sqrt { \lambda _ { k } } } { \sqrt { \lambda _ { \operatorname* { m a x } } } } = \frac { \sum _ { k } \left( \sqrt { \lambda _ { k , + } } + \sqrt { \lambda _ { k , - } } \right) + ( d - 2 m ) \sigma } { \sqrt { \lambda _ { \operatorname* { m a x } , + } } }
$$

Hence, as $w _ { i }$ decreases, its contribution to the sum decreases and the state operator rank decreases.

## C.3 Sequential learning of modes

Shallow model $\mathbf { u } = W \mathbf { x } .$ The gradient flow is

$$
\dot { W } = - ( W - W ^ { * } ) C
$$

which is linear and can be solved as $W ( \tau ) - W ^ { * } = ( W ( 0 ) - W ^ { * } ) e ^ { - C \tau }$ . Thus, the residual in the plane $( \mu _ { i } , \mu _ { i } ^ { * } )$ decays according to the smallest eigenvalue for each block: $e ^ { - \lambda _ { i , - } \tau }$ . The learning time is:

$$
\tau _ { i } \propto \frac { 1 } { \lambda _ { i , - } } = \frac { 1 } { \sigma ^ { 2 } + w _ { i } R ^ { 2 } \left( \frac { 1 } { 2 } - \frac { 1 } { \pi } \right) }
$$

Deep model $\bf { u } \ = \nabla W _ { 2 } W _ { 1 } { \bf x }$ . We write the input-output correlation as $\Sigma _ { y x } ~ = ~ W ^ { * } C$ and input correlation as $\Sigma _ { x x } = C$ . Computing the SVD of the input correlation using its eigenbasis: $\Sigma _ { x x } =$

$\begin{array} { r } { \sum _ { k } \lambda _ { k } \mathbf { v } _ { k } \mathbf { v } _ { k } ^ { \top } } \end{array}$ , which leads to:

$$
\begin{array} { c } { { \displaystyle \Sigma _ { y x } = \sum _ { k } \lambda _ { k } ( W ^ { * } { \bf v } _ { k } ) { \bf v } _ { k } ^ { \top } } } \\ { { \displaystyle \quad = \sum _ { k } \left( \frac { \pi } { 2 } \lambda _ { k } \right) { \bf u } _ { k } { \bf v } _ { k } ^ { \top } } } \\ { { \displaystyle \quad = \sum _ { k } s _ { k } { \bf u } _ { k } { \bf v } _ { k } ^ { \top } } } \end{array}
$$

From the form of $W ^ { * }$ and the eigenvectors of $C ,$ we compute:

$$
W ^ { * } \mathbf { v } _ { i , + } = - \frac { \pi } { 2 } \mathbf { v } _ { i , - } \equiv \frac { \pi } { 2 } \mathbf { u } _ { i , + } , \quad W ^ { * } \mathbf { v } _ { i , - } = \frac { \pi } { 2 } \mathbf { v } _ { i , + } \equiv \frac { \pi } { 2 } \mathbf { u } _ { i , - }
$$

and absorbing the factor into the singular values gives $\begin{array} { r } { s _ { k } = \frac { \pi } { 2 } \lambda _ { k } } \end{array}$ . Thus, $\Sigma _ { x x }$ and $\Sigma _ { y x }$ share the right basis and the dynamics separate mode by mode<sup>8</sup>. The learning time of the $i ^ { \mathrm { t h } }$ mode goes like ${ \mathcal { O } } ( s _ { i } ^ { - 1 } )$ or equivalently $\mathcal { O } ( w _ { i } ^ { - 1 } )$ when $w _ { i } R ^ { 2 } \gg \sigma ^ { 2 }$ . Note that the model must learn 2 singular mode directions for each mode, $\lambda _ { i , + }$ and $\lambda _ { i , - } < \lambda _ { i , + }$ , and so each mode also has 2 distinct learning times. The learning time equations apply to these sub-modes directly, namely $\tau _ { i , \pm } \propto 1 / \lambda _ { i , \pm } = \mathcal { O } ( w _ { i } ^ { - 1 } )$ , and we consider a data mode learned when the model learns the both singular mode directions.

For each data mode, we can compute what is learned by the model at the plateau and end of learning. In the plane $\left( \frac { \mu _ { i } } { R } , \frac { \mu _ { i } ^ { * } } { R } \right)$ , the plateau corresponds to learning:

$$
W _ { i } ^ { \mathrm { p l a t e a u } } = { \frac { \pi } { 2 } } \mathbf { u } _ { i , + } \mathbf { v } _ { i , + } ^ { \top } = { \frac { \pi } { 4 } } \left( { { - 1 \atop 1 } } \right) ( 1 1 ) = { \frac { \pi } { 4 } } \left( { - 1 - 1 \atop 1 } \right)
$$

This velocity defines a shear which sends $\begin{array} { r } { \pmb { \mu } _ { i }  ( 1 - \frac { \pi } { 4 } ) \pmb { \mu } _ { i } + \frac { \pi } { 4 } \pmb { \mu } _ { i } ^ { * } } \end{array}$ , which yields overlap $\begin{array} { r } { s _ { i i } = \frac { \pi } { 4 } } \end{array}$ ≈ 0.79. At the end of training, the model learns the other direction:

$$
W _ { i } ^ { \mathrm { c o n v e r g e } } = W _ { i } ^ { \mathrm { p l a t e a u } } + \frac { \pi } { 2 } \mathbf { u } _ { i , - } \mathbf { v } _ { i , - } ^ { \top } = \frac { \pi } { 4 } \left( \begin{array} { l l } { - 1 } & { - 1 } \\ { 1 } & { 1 } \end{array} \right) + \frac { \pi } { 4 } \left( \begin{array} { l l } { 1 } & { - 1 } \\ { 1 } & { - 1 } \end{array} \right) = \frac { \pi } { 2 } \left( \begin{array} { l l } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} \right)
$$

which is a rotation generator mapping $\pmb { \mu } _ { i }  \pmb { \mu } _ { i } ^ { * }$ , and hence the mode is learned $( s _ { i i } = 1 )$

Numerical validation. To verify the scaling laws, we train different models whilst varying the weighting of the smallest mode, measuring how the time to learn each mode scales with weighting. To do this we track the per mode loss:

$$
\mathcal { L } _ { i } = \int _ { 0 } ^ { 1 } \| \mathbf { u } _ { t } ^ { * } ( \bar { \mathbf { x } } ^ { ( i ) } ( t ) ) - \mathbf { u } _ { t } ^ { \theta } ( \bar { \mathbf { x } } ^ { ( i ) } ( t ) ) \| ^ { 2 } d t
$$

where $\bar { \mathbf { x } } ^ { ( i ) } ( t ) = e ^ { t W ^ { * } } \pmb { \mu } _ { i }$ is the transported $i ^ { \mathrm { { t h } } }$ initial mode, defining the $i ^ { \mathrm { { t h } } }$ mode learned when $\mathcal { L } _ { i }$ drops below some threshold ε (Fig. C1a). Using the analytic form of $\lambda _ { \operatorname* { m i n } }$ , we also plot learning times against $1 / \lambda _ { \operatorname* { m i n } }$ for deep and shallow models, reporting the log-log exponent using np.polyfit which are both close to the expected value of 1 (Fig. C1c-d).

## C.4 Non-linear flow matching

In section 3.3, we consider a task of learning a mixture of Gaussians target distribution with isotropic covariances:

$$
\mathbf { z } \sim p ^ { * } ( \mathbf { z } ) = \sum _ { i = 1 } ^ { m } \pi _ { i } \mathcal { N } ( \mathbf { z } | \pmb { \mu } _ { i } , \sigma _ { * } ^ { 2 } I _ { d } ) \in \mathbb { R } ^ { d }
$$

from a unimodal initial distribution

$$
\mathbf { x } ( 0 ) \sim p _ { 0 } ( \mathbf { x } ) = \mathcal { N } ( \mathbf { x } | \mathbf { 0 } , \sigma ^ { 2 } I _ { d } ) \in \mathbb { R } ^ { d } .
$$

with $\sigma = 1 . 0$ and $\sigma _ { * } ~ = ~ 0 . 1$ . The mixture weights are distributed according to a power law: $\begin{array} { r } { \pi _ { i } = \frac { i ^ { - \alpha } } { \sum _ { i } i ^ { - \alpha } } , \alpha = 3 } \end{array}$ . We choose the linear interpolation probability path, $\bar { \mathbf { x } } ( t ) = t \mathbf { z } + ( 1 - t ) \mathbf { x } ( 0 )$ . Motivated by the form of the analytic field derived in section C.6, we parameterise our model as:

$$
\mathbf { u } _ { \theta } ( \mathbf { x } , t ) = c ( t ) \mathbf { x } + ( 1 - t c ( t ) ) W _ { 2 } \mathbf { R e L U } ( W _ { 1 } \mathbf { x } + \mathbf { b } )
$$

a  
![](images/4c54ed6cd0aa549e5c243b8f97c4c44a1416b79e459408e3dac4e69772e8ef4d.jpg)  
b

![](images/1a0eda58cb5b68ebb698c114793c756fefdaa2b1f7dde05db2e5454951447214.jpg)

C  
![](images/680c6103a26564395440732dd4806da329486db50e309d20f40870ebf83de7ca.jpg)

d  
![](images/6a88c7855309a487e6ad6b92f702fbe463fcfa41141dcec1a3a96921cbaa30a3.jpg)  
Figure C1: Sequential learning for imbalanced mixture weights. a. Loss per mode over training for a model with 10 modes and weightings $\pi _ { i } \propto i ^ { - 1 }$ . Progressively smaller modes are learned sequentially. Plateaus within a single mode curve correspond to learning the first and second singular direction corresponding to that mode. b. Mode overlaps $s _ { i i }$ for each initial-target mode pair over training. c-d. Plots of learning time of smallest mode against $1 / \lambda _ { \operatorname* { m i n } }$ for a shallow (c) and deep (d) linear model. Models trained with 3 modes with weightings $\begin{array} { r } { \left[ \pi _ { \mathrm { m i n } } , \frac { 1 - \pi _ { \mathrm { m i n } } } { 2 } , \frac { 1 - \pi _ { \mathrm { m i n } } } { 2 } \right] } \end{array}$ with $\pi _ { \mathrm { m i n } } \in [ 0 . 0 1 , 0 . 3 ]$

a  
![](images/768504107df33f3df6c54046b14c0d7a190cb1468aaceacd0590be5a747db532.jpg)

b  
![](images/65ad497ee28f3d8e0dfa4b2a118b7288cc2601e65bcd6ce378eedd2e68879026.jpg)  
Figure C2: Flow matching operators under imbalanced modes. a. State and adjoint operator stable ranks over training. As expected the state operator depends on the data distribution and is constant. Adjoint rank decreases as modes are progressively learned b. Rank bound prefactor over training. Increases with each learned mode.

where $c ( t )$ is given explicitly to the model and $W _ { 2 } \in \mathbb { R } ^ { n \times m } , W _ { 1 } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m }$ are trained parameters. We train using the conditional velocity objective:

$$
\mathcal { L } = \mathbb { E } _ { \mathbf { x } ( 0 ) , \mathbf { z } } \left[ \int _ { 0 } ^ { 1 } | | \mathbf { u } _ { \boldsymbol { \theta } } ( \bar { \mathbf { x } } ( t ) , t ) - ( \mathbf { z } - \mathbf { x } ( 0 ) ) | | ^ { 2 } d t \right]
$$

To train the model, we use full-batch gradient descent with a fixed set of initial samples and time points. We initialise weights as $W _ { i j } \sim \mathrm { \bar { \mathcal { N } } } ( 0 , \sigma _ { \mathrm { i n i t } } / \sqrt { n } )$ and $\mathbf b = \mathbf 0$ . The hyperparameters are:

<table><tr><td>n</td><td>m</td><td> $\sigma _ { \mathrm { i n i t } }$ </td><td>lr</td><td>Iterations</td><td>Batch size</td><td>Optimizer</td></tr><tr><td>5</td><td>64</td><td> $\overline { { { 1 e ^ { - 4 } } } }$ </td><td> $5 e ^ { - 4 }$ </td><td>10k</td><td>1000</td><td>Adam</td></tr></table>

Learned density To compute the log-density of the learned model, log $p _ { \theta , t } ( \mathbf { x } )$ , we use the instantaneous change of variables for probability flow, derived from the continuity equation<sup>4</sup>:

$$
\log p _ { \theta , t } ( \mathbf { x } ) = \log p _ { 0 } ( \mathbf { x } ) - \int _ { 0 } ^ { t } \nabla \cdot u _ { \theta } ( \mathbf { x } , t ) d t
$$

By solving the same augmented ODE as in appendix B.2, we can compute the accumulated log-density at time t at a grid of initial points.

Time to learning vs mode imbalance In figure 5c, we plot the time taken to learn the smallest mode as a function of the imbalance between the weightings of the largest and smallest modes. We used a symmetric 3-mode setup in 2-d, with weightings $\begin{array} { r } { \overline { { \pi } } = [ \alpha , \frac { 1 - \alpha } { 2 } , \frac { 1 - \alpha } { 2 } ] } \end{array}$ , and with $\alpha \in [ 0 . 0 4 , 0 . 1 6 ]$ We define the time to learn as the first training iteration (except initialisation) where any generated sample is within a threshold distance of the smallest weighted mode.

Validation on more realistic data. To test whether the scaling of learning time with mode imbalance extends to more realistic data, we trained a one-hidden-layer ReLU MLP with 512 hidden units on a subset of MNIST containing two digit classes. To continuously control the relative weight of the two classes during training, we assigned separate weights to their losses:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = w \mathcal { L } _ { 1 } ( \boldsymbol { \theta } ) + ( 1 - w ) \mathcal { L } _ { 2 } ( \boldsymbol { \theta } ) , } \end{array}
$$

where $w \in ( 0 , 0 . 5 ]$ is the class weight. This weighting is equivalent to varying the relative proportions of the two classes in the data distribution, since the expected population loss under a mixture distribution decomposes into a weighted sum of the conditional expected losses for each class:

$$
\mathcal { L } = \mathbb { E } _ { x \sim p _ { w } } [ \ell ( \theta ; x ) ] = w \mathbb { E } _ { x \sim p _ { 1 } } [ \ell ( \theta ; x ) ] + ( 1 - w ) \mathbb { E } _ { x \sim p _ { 2 } } [ \ell ( \theta ; x ) ]
$$

where $p _ { w } ( x ) = w p _ { 1 } ( x ) + ( 1 - w ) p _ { 2 } ( x )$ . To approximate the gradient-flow regime, we used full-batch gradient descent with a small learning rate, replacing the conditional expectations with empirical averages over all samples from each class.

We considered the minority mode to be learned when its relative loss change $\rho _ { k } = ( \mathcal { L } _ { 1 } ( \theta _ { k } ) -$ $\mathcal { L } _ { 1 } ( \theta _ { 0 } ) ) / \mathcal { L } _ { 1 } ( \theta _ { 0 } )$ fell below a threshold δ. However, fluctuations in the loss during training due to the approximation of gradient flow (e.g. step size) can cause repeated crossings of this threshold, making the first crossing an unreliable estimate of learning time. To obtain a more robust quantity, we instead defined a soft learning time by using a soft sum of the number of iterations this condition was not satisfied.

$$
T = \sum _ { k = 0 } ^ { n } \left( 1 - \sigma \left( \beta ( \delta - \rho _ { k } ) \right) \right) , \qquad \sigma ( u ) = \frac { 1 } { 1 + e ^ { - u } } ,
$$

where $\beta$ controls the sharpness of the threshold. In the gradient-flow limit, the learning time $T$ continuously depends on gradient flow w. We trained twelve models with $w \in \{ 0 . 1 , \ldots , 0 . 5 \}$ . As for the linear models, we observed a power-law scaling of the learning time with the weight, but with lower exponent (0.749), akin to the nonlinear tractable model trained on the Gaussian mixture data.

![](images/c1ecf0b8891356484607bf6523c3b75327e73497b1c57b7463962d22abf1f816.jpg)  
Figure C3: Learning time of the smallest data mode as a function of the mode weight.

## C.5 Marginal distribution for MoG targets

At time $t ,$ the linear interpolation path yields the conditional (on a data sample z) probability distribution

$$
p _ { t } ( \mathbf { x } | \mathbf { z } ) = \mathcal { N } ( \mathbf { x } | t \mathbf { z } , ( 1 - t ) ^ { 2 } \sigma ^ { 2 } I _ { d } ) .
$$

Marginalising over the data distribution:

$$
\begin{array} { l } { { \displaystyle p _ { t } ( { \bf x } ) = \int p _ { t } ( { \bf x } | { \bf z } ) p ^ { * } ( { \bf z } ) d { \bf z } } \ ~ } \\ { { \displaystyle ~ = \int \left( N ( { \bf x } | t { \bf z } , ( 1 - t ) ^ { 2 } \sigma ^ { 2 } I _ { d } ) \right) \left( \sum _ { i = 1 } \pi _ { i } \mathcal { N } ( { \bf z } | \mu _ { i } , \sigma _ { * } ^ { 2 } I _ { d } ) \right) d z } \ ~ } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { m } \pi _ { i } \int N ( { \bf x } | t { \bf z } , ( 1 - t ) ^ { 2 } \sigma ^ { 2 } I _ { d } ) \mathcal { N } ( { \bf z } | \mu _ { i } , \sigma _ { * } ^ { 2 } I _ { d } ) d { \bf z } } \ ~ } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { m } \pi _ { i } p _ { t } ^ { ( i ) } ( { \bf x } ) } \ ~ } \end{array}
$$

where $p _ { t } ^ { ( i ) } ( \mathbf { x } ) = \mathcal { N } ( \mathbf { x } | t \mu _ { i } , \Sigma _ { t } )$ is the probability density conditioned on target mode i and $\Sigma _ { t } =$ $( t ^ { 2 } \sigma _ { * } ^ { 2 } \dot { + } \dot { ( } 1 - t ) ^ { 2 } \sigma ^ { 2 } ) \dot { I } _ { d }$

## C.6 Marginal velocity field for MoG targets

We start by writing the conditional velocity field for the linear interpolation path:

$$
{ \bf u } _ { t } ( { \bf x } | { \bf z } ) = \dot { \bf x } ( t ) = { \bf z } - { \bf x } ( 0 )
$$

Rearranging the probability path equation we get

$$
\mathbf { x } ( 0 ) = \frac { \bar { \mathbf { x } } ( t ) - t \mathbf { z } } { 1 - t }
$$

which we can substitute into the conditional velocity to remove dependency on $\mathbf { x } _ { 0 } \colon$

$$
\mathbf { u } _ { t } ( \mathbf { x } | \mathbf { z } ) = \frac { 1 } { 1 - t } ( \mathbf { z } - \mathbf { x } )
$$

To obtain the marginal field (for the $i ^ { \mathrm { t h } }$ target mode) we marginalise over z by taking the expectation with respect to the posterior distribution $p _ { t } ( \mathbf { z } | \mathbf { x } )$ :

$$
\begin{array} { r l } { { } } & { { u } _ { t } ^ { ( i ) } ( \mathbf { x } ) = \mathbb { E } _ { \mathbf { z } \sim p _ { t } ( \mathbf { z } \mid \mathbf { x } ) } [ u _ { t } ( \mathbf { x } \mid \mathbf { z } ) ] } \\ { { } } & { { = \displaystyle \frac { 1 } { 1 - t } ( \mathbb { E } _ { \mathbf { z } \sim p _ { t } ( \mathbf { z } \mid \mathbf { x } ) } \left[ \mathbf { z } \right] - \mathbf { x } ) } } \end{array}
$$

To evaluate the expectation, we use the following result:

Theorem C.1. Let $\mathbf { u } \sim \mathcal { N } ( \mathbf { a } _ { \mathbf { u } } , B _ { \mathbf { u } } )$ and $\mathbf { v } \sim \mathcal { N } ( \mathbf { a } _ { \mathbf { v } } , B _ { \mathbf { v } } )$

$$
p ( \mathbf { u } | \mathbf { v } + \mathbf { u } ) = \mathcal { N } \left( \mathbf { a _ { u } } + B _ { \mathbf { u } } ( B _ { \mathbf { u } } + B _ { \mathbf { v } } ) ^ { - 1 } \left[ ( \mathbf { u } + \mathbf { v } ) - ( \mathbf { a _ { u } } + \mathbf { a _ { v } } ) \right] , \quad B _ { \mathbf { u } } - B _ { \mathbf { u } } ( B _ { \mathbf { u } } + B _ { \mathbf { v } } ) ^ { - 1 } B _ { \mathbf { u } } \right)
$$

I.e. the conditional distribution of the sum of two Gaussians is itself Gaussian.

Applying this, we obtain $\begin{array} { r } { \mathbb { E } [ { \mathbf { z } } | { \mathbf { x } } ] = \pmb { \mu _ { i } } + t \sigma _ { * } ^ { 2 } I _ { d } \Sigma _ { t } ^ { - 1 } ( { \mathbf { x } } - t { \pmb { \mu _ { i } } } ) } \end{array}$ where $\Sigma _ { t } = t ^ { 2 } \sigma _ { * } ^ { 2 } + ( 1 - t ) ^ { 2 } \sigma ^ { 2 }$ . Thus the field becomes:

$$
{ \bf u } _ { t } ^ { ( i ) } ( { \bf x } ) = \frac { 1 } { 1 - t } \left( \pmb { \mu } _ { i } + t \sigma _ { * } ^ { 2 } I _ { d } \Sigma _ { t } ^ { - 1 } ( { \bf x } - t { \pmb \mu } _ { i } ) \right) - \frac { 1 } { 1 - t } { \bf x }
$$

$$
\stackrel { \left( 1 \right) } { = } \frac { 1 } { 1 - t } \mu _ { i } - \frac { 1 } { 1 - t } ( { \bf x } - t { \pmb \mu } _ { i } ) - \frac { t } { 1 - t } \mu _ { i } + \frac { t } { 1 - t } \sigma _ { * } ^ { 2 } I \Sigma _ { t } ^ { - 1 } ( { \bf x } - t { \pmb \mu } _ { i } )
$$

$$
\stackrel { ( 2 ) } { = } \pmb { \mu } _ { i } + \left[ \frac { t } { 1 - t } \sigma _ { * } ^ { 2 } I - \frac { 1 } { 1 - t } \Sigma _ { t } \right] \Sigma _ { t } ^ { - 1 } ( \mathbf { x } - t \pmb { \mu } _ { i } )
$$

$$
\overset { ( 3 ) } { = } \pmb { \mu } _ { i } + \frac { t \sigma _ { * } ^ { 2 } - ( 1 - t ) \sigma ^ { 2 } } { t ^ { 2 } \sigma _ { * } ^ { 2 } + ( 1 - t ) ^ { 2 } \sigma ^ { 2 } } ( \mathbf { x } - t \pmb { \mu } _ { i } )
$$

where in (1) we have added and subtracted by $\frac { t } { 1 - t } \pmb { \mu } _ { i }$ , in (2) we took out a factor of $\begin{array} { r } { \sum _ { t } ^ { - 1 } ( \mathbf x - t \pmb { \mu } _ { i } ) } \end{array}$ on the right, and in (3) we have substituted in the expression for $\Sigma _ { t }$ and simplified. Defining $\begin{array} { r } { c ( t ) = \frac { t \sigma _ { * } ^ { 2 } - ( 1 - t ) \sigma ^ { 2 } } { t ^ { 2 } \sigma _ { * } ^ { 2 } + ( 1 - t ) ^ { 2 } \sigma ^ { 2 } } } \end{array}$ we can write the field as

$$
\mathbf { u } _ { t } ^ { ( i ) } ( \mathbf { x } ) = c ( t ) \mathbf { x } + ( 1 - t c ( t ) ) \pmb { \mu } _ { i }
$$

Note that this is still conditional on a specific target mode $\pmb { \mu } _ { i }$ . To get the full unconditional field, we use the continuity equation:

Theorem C.2 (Continuity Equation). For a flow model with vector field $u _ { t }$ and ${ \bf x } _ { 0 } \sim p _ { i n i t }$ , then $\bar { \mathbf { x } } _ { t } \sim p _ { t } f o r a l l 0 \le t \le 1 \dot { } i \dot { }$ fand only $i f$

$$
\partial _ { t } p _ { t } ( { \bf x } ) = - \nabla \cdot ( p _ { t } u _ { t } ) ( { \bf x } )
$$

for all $\mathbf { x } \in \mathbb { R } ^ { d } , 0 \leq t \leq 1$

Then, using the form of the marginal probability path in section C.5, we can write the LHS as

$$
\begin{array} { l } { \displaystyle \partial _ { t } p _ { t } ( { \mathbf x } ) = \partial _ { t } \left( \displaystyle \sum _ { i = 1 } ^ { m } \pi _ { i } p _ { t } ^ { ( i ) } ( { \mathbf x } ) \right) } \\ { \displaystyle \qquad \stackrel { ( 1 ) } { = } \displaystyle \sum _ { i = 1 } ^ { m } \pi _ { i } \partial _ { t } p _ { t } ^ { ( i ) } ( { \mathbf x } ) } \\ { \displaystyle \qquad \stackrel { ( 2 ) } { = } - \displaystyle \sum _ { i = 1 } ^ { m } \pi _ { i } \nabla \cdot ( p _ { t } ^ { ( i ) } u _ { t } ^ { ( i ) } ) ( { \mathbf x } ) } \\ { \displaystyle \qquad \stackrel { ( 3 ) } { = } - \nabla \cdot \left( \displaystyle \sum _ { i = 1 } ^ { m } \pi _ { i } p _ { t } ^ { ( i ) } ( { \mathbf x } ) u _ { t } ^ { ( i ) } ( { \mathbf x } ) \right) } \end{array}
$$

where in (1) we used the linearity of derivatives, in (2) we used the continuity equation for the $i ^ { \mathrm { { t h } } }$ target mode and in (3) we use the linearity of the divergence operator. Comparing this to the RHS of the continuity equation, we get that

$$
p _ { t } ( \mathbf { x } ) u _ { t } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { m } \pi _ { i } p _ { t } ^ { ( i ) } ( \mathbf { x } ) u _ { t } ^ { ( i ) } ( \mathbf { x } )
$$

which we can rearrange to get

$$
u _ { t } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { m } \gamma _ { t } ^ { ( i ) } ( \mathbf { x } ) u _ { t } ^ { ( i ) } ( \mathbf { x } )
$$

where $\begin{array} { r } { \gamma _ { t } ^ { ( i ) } ( \mathbf { x } ) = \frac { \pi _ { i } p _ { t } ^ { ( i ) } ( \mathbf { x } ) } { p _ { t } ( \mathbf { x } ) } } \end{array}$ acts as a weighting between different modes. Substituting the expression for the field conditioned on target mode i, we obtain the final form of the marginal field:

$$
\mathbf { u } _ { t } ( \mathbf { x } ) = c ( t ) \mathbf { x } + ( 1 - t c ( t ) ) \sum _ { i = 1 } ^ { m } \gamma _ { t } ^ { ( i ) } ( \mathbf { x } ) \pmb { \mu } _ { i }
$$

where we used the fact that $\begin{array} { r } { \sum _ { i } \gamma _ { t } ^ { ( i ) } ( \mathbf { x } ) = 1 } \end{array}$

## C.7 Operators of the non-linear model

Consider the loss functional:

$$
\mathcal { L } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \int _ { 0 } ^ { 1 } \ell ( \bar { \mathbf { x } } ( t ) ) d t \right]
$$

where $\ell ( \bar { \mathbf { x } } ( t ) ) = | | \Phi ( \bar { \mathbf { x } } ( t ) ) - \mathbf { u } ( \bar { \mathbf { x } } ( t ) ) | | ^ { 2 }$ is the MSE objective for model Φ and target velocity u, and $\bar { { \bf x } } ( t )$ is the interpolated trajectory. Also consider a deep neural network architecture:

$$
\Phi = W _ { k } \circ \phi \circ W _ { k - 1 } \circ \phi \circ \cdot \cdot \cdot \circ W _ { 1 }
$$

which has $i ^ { \mathrm { { t h } } }$ layer pre-activations ${ \bf z } _ { i } = { \cal W } _ { i } \circ \phi . . .$ . and post-activations $\mathbf { h } _ { i } = \phi ( \mathbf { z } _ { i } )$ for some non-linearity $\phi .$ . The gradients with respect to the $i ^ { \mathrm { t h } }$ layer weights are:

$$
\begin{array} { r l r } {  { \nabla _ { W _ { i } } \mathcal { L } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } [ \int _ { 0 } ^ { 1 } \partial _ { \mathbf { z } _ { i } } \ell ( \bar { \mathbf { x } } ( t ) ) \otimes \mathbf { h } _ { i - 1 } ( \bar { \mathbf { x } } ( t ) ) d t ] } } \\ & { } & { = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } [ \int _ { 0 } ^ { 1 } R ( \bar { \mathbf { x } } ( t ) ) \frac { \partial \Phi } { \partial \mathbf { z } i } \otimes \mathbf { h } _ { i - 1 } ( \bar { \mathbf { x } } ( t ) ) d t ] } \end{array}
$$

where $R ( \bar { \mathbf { x } } ( t ) ) = \Phi ( \bar { \mathbf { x } } ( t ) ) - \mathbf { u } ( \bar { \mathbf { x } } ( t ) )$ is the residual per sample.

Now we define our network as in section 3.3 (omitting the skip connection which contributes zero to the gradient, and the time-dependent pre-factor which just scales the gradient):

$$
\Phi ( \mathbf { x } ) = W _ { 2 } \mathrm { R e L U } ( W _ { 1 } \mathbf { x } + \mathbf { b } )
$$

We can then compute the gradients as:

$$
\begin{array} { l } { \nabla _ { W _ { 1 } } \mathcal { L } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \displaystyle \int _ { 0 } ^ { 1 } \underbrace { \mathrm { d i a g ~ ( R e L U ^ { \prime } ( \mathbf { z } _ { 1 } ) ) } W _ { 2 } ^ { \top } R ( \bar { \mathbf { x } } ( t ) ) } _ { A _ { 1 } } \otimes \underbrace { \bar { \mathbf { x } } ( t ) } _ { \bar { H } _ { 1 } } d t \right] } \\ { \nabla _ { W _ { 2 } } \mathcal { L } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \displaystyle \int _ { 0 } ^ { 1 } \underbrace { R ( \bar { \mathbf { x } } ( t ) ) } _ { A _ { 2 } } \otimes \underbrace { \mathbf { h } _ { 1 } ( \bar { \mathbf { x } } ( t ) ) } _ { \bar { H } _ { 2 } } d t \right] } \end{array}
$$

where we define $A _ { 1 } , A _ { 2 }$ as the adjoint (backward) operators and $\bar { H } _ { 1 } , \bar { H } _ { 2 }$ as the state (forward) operators for layer 1 and layer 2 respectively.

## D Code and compute requirements

The appendix contains all architectures and training parameters necessary to reproduce the analyses. A Git repository containing all training functions and analyses notebooks will be released upon publication.

Packages The code makes use of standard python packages for computations and model training: NumPy, JAX, Diffrax, Equinox.

Compute The linear CNF and non-linear FM models were trained in under an hour using an A100 GPU. The linear FM models were trained in a few minutes on a 16 core CPU.

## Supplementary References

[1] P. Kidger et al. “Efficient and accurate gradients for neural sdes”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 18747–18761.

[2] V. Thibeault et al. “The low-rank hypothesis of complex systems”. In: Nature Physics 20.2 (2024), pp. 294–302.

[3] W. Grathwohl et al. “Scalable Reversible Generative Models with Free-form Continuous Dynamics”. In: International Conference on Learning Representations. 2019.

[4] R. T. Q. Chen et al. “Neural Ordinary Differential Equations”. In: Advances in Neural Information Processing Systems. Vol. 31. Curran Associates, Inc., 2018.

[5] A. Pellegrino et al. “Low tensor rank learning of neural dynamics”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 11674–11702.

[6] C.-K. Li et al. “A note on eigenvalues of perturbed Hermitian matrices”. In: Linear algebra and its applications 395 (2005), pp. 183–190.

[7] Y. Lipman et al. “Flow Matching for Generative Modeling”. In: The Eleventh International Conference on Learning Representations. 2023.

[8] A. M. Saxe et al. Exact solutions to the nonlinear dynamics oflearning in deep linear neural networks. Feb. 19, 2014. arXiv: 1312.6120[cs].