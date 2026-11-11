# YOU DON’T NEED FEATURE LEARNING TO DO IN-CONTEXT LEARNING

Joseph Turnbull<sup>1</sup> Jamie Simon<sup>1,2</sup>

{joeyturnbull,jsi}@berkeley.edu

## ABSTRACT

How should we understand memorization vs. generalization in in-context learning (ICL) tasks, and when does one happen vs. the other? To investigate, we study how rotation-invariant kernel machines learn the ICL linear regression task proposed by Raventos et al.´ (2023). By decomposing the task in the kernel eigenbasis, we obtain a spectral picture of learning in this ICL setting. This picture reveals that as sample size grows, kernel machines learn successively-higher-degree components of the memorizing solution, which depend upon successively-higher-degree statistics of the training task set. The generalizing solution is approximately a low-degree truncation of the memorizing solution, and so the model first generalizes, then degrades into memorization as training proceeds.

The takeaways are threefold. First, if this case is indicative, learning a “generalizing” solution on an ICL task is really just underfitting the memorizing solution! Second, the fact that kernel machines can generalize on this task means that feature learning is inessential to ICL. Third, transformers empirically trained to do ICL are shown to memorize tasks through the higher-order statistics of the training tasks that the kernel-based theory predicts, suggesting it applies even to models capable of feature learning.

## 1 INTRODUCTION

Large language models (LLMs) exhibit the striking ability to infer an intended task from context. For example, an LLM told to write a new file in a software repository can match style and content to existing files present in its context. More generally, an LLM given a few examples of a desired task can often infer the task and apply it to new settings. This is called few-shot learning or in-context learning (ICL) in the literature. This useful behavior and the conditions under which it arises remain largely mysterious.<sup>1</sup>

How should we begin to describe ICL in mathematical terms? A general setting is as follows. Suppose that there exists a set B of all possible tasks, of which only a subset $\{ \beta ^ { ( t ) } \} _ { t = 1 } ^ { T } = : B _ { \mathrm { t r } } \subset \mathcal { \bar { B } }$ are present in the model’s training data. We might suppose that each task specifies a certain function $f _ { \beta } : \mathcal { X }  \mathcal { Y }$ between two spaces. A given context $z \in ( \mathcal { X } \times \mathcal { Y } ) ^ { L } \times \dot { \mathcal { X } }$ consists of a number of examples $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { L }$ , where $y _ { i } = f _ { \beta } ( x _ { i } ) + \eta _ { i }$ with noise $\eta _ { i } ,$ together with an unpaired input $x _ { L + 1 }$ The model is trained to map $z \mapsto f _ { \beta } ( x _ { L + 1 } )$ , which requires first inferring $\beta$ from the example pairs and then applying it to the unpaired input $x _ { L + 1 }$

The most interesting questions arise when the model is provided many examples for each of the T tasks at training time, but at test time is provided a new task $\beta ^ { * } \notin B _ { \mathrm { t r } } .$ A model performing ICL will be able to extract the task and apply $f _ { \beta ^ { * } }$ to the unpaired input. We would like to know when the model can perform well on unseen tasks — and when it can, to understand why, and how it learned to do so.

![](images/80aa6d755d1a14cc133e47d743c632c97ca3aaf2b57108be5656c885a9f995ec.jpg)

![](images/d1f719b3ef1f70d679bdde1901aa95defd8607bb35f0e50408d1da81100efa31.jpg)  
Figure 1: As sample size grows, kernels learn successively higher-degree target function com ponents in in-context linear regression. They first learn the mean training task (phase 1), then approximate the degree-three generalizing solution (phase $^ { 2 ) , }$ and finally memorize the training tasks (phase 3). (Left) A visualization of a kernel machine’s inferred task in task space $\boldsymbol { B } = \mathbb { R } ^ { \breve { d } }$ decomposed into vector components learned in each phase (arrows). Grey dots indicate training tasks ${ \cal B } _ { \mathrm { t r } } = \{ \beta ^ { ( t ) } \}$ and a grey star indicates an unseen task $\beta ^ { * } \notin B _ { \mathrm { t r } }$ . In phase 1, the kernel learns the mean training task $\hat { \mu } _ { \beta } : = \mathbb { E } _ { \beta \sim B _ { \mathrm { t r } } } [ \beta ]$ . In phase $^ { 2 , }$ the kernel learns approximately $\beta ^ { * } - \hat { \mu } _ { \beta }$ , nearly recovering the true task $\beta ^ { * }$ . In phase 3, learned higher-degree terms pull the prediction towards the nearest task in $B _ { \mathrm { t r } }$ . (Right) We plot learning curves for KR on the in-context linear regression problem with $d = 5 , L = 7 , T = 1 \bar { 6 }$ . (Top) Evaluated on the training tasks, the error falls monotonically, eventually reaching that of the memorizing solution (Equation (3)). We show both theoretical predictions (lines) and experimental values (dots) in the experimentally-accessible region $n \in [ 1 , 5 \times 1 0 ^ { 4 } ]$ ]. (Middle) Evaluated on novel tasks $( \beta \sim \mathcal { N } ( 0 , I _ { d } ) )$ , error is minimized at the end of phase $^ { 2 , }$ , approaching that of the generalizing solution (Equation (4)), before rising to that of the memorizing solution in phase 3. Background shading indicates the three phases of learning. (Bottom) We also plot the predicted overlap between the function $\hat { f } _ { \mathrm { k e r } }$ learned by KR and various polynomial functions $g _ { i }$ . The functions $g _ { i }$ correspond to the vector components in the cartoon on the left, and are given explicitly in Section B. The function $g _ { i }$ is determined by the i-th order statistics of $\boldsymbol { B } _ { \mathrm { t r } } .$ . The component $g _ { 1 }$ is responsible for generalization and is learned in phase 2. We show theoretical predictions (lines) for the kernel’s error out to $1 0 ^ { 1 2 }$ samples, finding excellent agreement between theory and experiment in the experimentally accessible region.

It is worth stressing that generalization in ICL is strange indeed, and is in fact rather paradoxical. The Bayes-optimal solution for the training objective is to always interpret any new context z as indicating one of the $T$ training tasks and to simply apply that task. (Or, if there are several plausible candidates, as a distribution over several training tasks.) Given this training objective, it would seem hopeless that the model would correctly infer the unseen task $\beta ^ { * } \notin B _ { \mathrm { t r } }$ . Nevertheless, it evidently does so in many practical and useful settings. Our work in this paper is to offer a resolution to this paradox and to explain how, in spite of this mismatch, generalization occurs in ICL.

The main idea is that generalization occurs mid-optimization, as a milestone on the road to memorization. It is well-appreciated that neural networks learn simple functions early in training and more complex functions later on $( \mathrm { e . g } $ . Rahaman et al. (2019); Nakkiran et al. (2019); Xu et al. (2019)). We suggest that the generalizing solution is in fact just a simpler version of the memorizing solution, stopped before it learns the higher-degree functional components required for exact memorization of the training set.

We arrive at these conclusions from a study of the ICL-for-linear-regression setting of Raventos´ et al. (2023). In this setting, $\mathcal { X } = \mathbb { R } ^ { d } , \mathcal { y } \overset { \cdot } { = } \mathbb { R } , \mathcal { B } = \mathbb { R } ^ { d }$ , and each task is simply a linear map $f _ { \beta } ( { \pmb x } ) = \beta ^ { \top } { \pmb x }$ . Task vectors are sampled $\beta \sim \mathcal { N } ( 0 , I _ { d } )$ . The beauty of this setting is that the Bayes-optimal train-time (memorizing) and test-time (generalizing) solutions can be written down in closed form (Equations (3) and (4)). Which solution a model learns is largely determined by the model’s inductive biases. This suggests studying kernel regression (KR) as the model, since there exists a well-developed spectral theory of its inductive bias (Bordelon et al., 2020; Jacot et al., 2020; Simon et al., 2021). The main technical hurdle is in computing the kernel eigendecomposition on this task so that this existing theory may be applied.

Surmounting this hurdle, we find a resolution to the aforementioned paradox: the generalizing solution is approximately a low-polynomial-degree truncation of the memorizing solution. As the dataset size grows, kernel machines first learn simple low-degree polynomial components of the memorizing function before resolving increasingly high-degree components. They approximate the generalizing solution after learning the degree-three components of the memorizing solution. Given enough samples, they will proceed to fully memorize. Our results put theory behind the empirics of Raventos et al. ´ (2023), giving the first end-to-end theory of the generalization-memorization transition in ICL. This theory explains the “transience” phenomenon in which a model first learns the generalizing solution before “forgetting it” in favor of the memorizing solution (Singh et al., 2023).

A surprising consequence of our study of fixed-feature kernel machines is that a model does not need feature learning to do ICL. (Of course, feature learning certainly allows a model to do ICL more efficiently.) This suggests that memorization and generalization in ICL can be understood in simple polynomial language, without reference to feature learning or circuit formation. Experiments on feature-learning transformer models (Section 8) show that our kernel-based theory can be used to substantially delay when the models memorize, by removing higher-order statistics of the training tasks at fixed task diversity. This refines the task-diversity picture of Raventos et al.´ (2023): what matters for generalization is how Gaussian the training tasks are, not how many there are.

## 2 RESEARCH CONTEXT AND RELATED WORKS

Our work sits at the intersection of four threads: the tension between memorizing and generalizing Bayes-optimal solutions (most developed for diffusion models), in-context learning dynamics given finite task diversity, simplicity biases in learning, and the eigenstructure theory of kernel regression.

Memorizing and generalizing Bayes-optimal solutions. Whenever a model is trained on a finite sample from some latent distribution, two Bayes-optimal predictors are available: one that treats the empirical distribution as the prior (memorizing), and one that uses the population prior (generalizing). The former generically takes the form of a softmax-weighted average over training points. In our setting, the memorizing solution is the dMMSE (Equation (3)), a posterior average over the T pretraining tasks in task space. In diffusion modeling, the memorizing solution is akin to a nearestneighbor match to the closest training image (Scarvelis et al., 2025; Biroli et al., 2024; Kamb & Ganguli, 2025). In dense associative memories, retrieval is likewise a softmax-weighted average over stored patterns (Krotov & Hopfield, 2016; Ramsauer et al., 2021). The same paradox arises in each case: fully optimizing the training objective gives a predictor that cannot produce anything novel, so any observed generalization must come from the learnerfailing to reach that optimum.

The closest analogue to our resolution of this paradox comes from the diffusion literature. Li et al. (2024) found that linear models capturing the same Gaussian structure as nonlinear diffusion models are nearly identical to the optimal denoiser for Gaussian data. Bonnaire et al. (2025) train a random Fourier features model to show that memorization can be avoided through early stopping, given a strong enough implicit bias. Kamb & Ganguli (2025); Lukoianov et al. (2025) similarly attribute the avoidance of memorization in diffusion models to implicit biases of the network (equivariance) and the data (locality). Together, these results parallel our finding that the generalizing solution is a truncation of the memorizing one at its Gaussian statistics (Proposition 2), implemented through the inductive biases of the learner.

In-context learning. Since its description by Brown et al. (2020), ICL has remained a somewhat enigmatic capability of language models. To understand it, researchers have studied model-side attributes, such as parameter count, depth and the resulting expressivity of the architecture (Brown et al., 2020; Wei et al., 2022b; Garg et al., 2023), as well as data-side attributes, such as task diversity (Chan et al., 2022; Kirsch et al., 2024; Raventos et al.´ , 2023; Park et al., 2025). Closest to our non-attention setting, Tong & Pehlevan (2025) show that MLPs perform ICL linear regression in a similar manner to transformers. On the theoretical side, most analyses of ICL have focused on linear attention (Lu et al., 2025; Letey et al., 2025; Cui et al., 2024; Li et al., 2025). These models are tractable and give an end-to-end description of learning, but they cannot express the memorizing solution, and so cannot capture the transition toward it: they are, by construction, unable to exhibit transience.

Transience. Singh et al. (2023) found that ICL can emerge and later vanish during training. Most explanations of this transience posit competing memorizing and generalizing circuits (Carroll et al., 2025; Singh et al., 2025; Nguyen & Nallamala, 2025). Others (Wurgaft et al., 2025; Park et al., 2025) claim that transience can be explained through a linear combination of the memorizing and generalizing solutions. In Section 5, we explain transience fromfirst principles usingfixed-feature models. To be clear, we give the first description of transience purely as a consequence of our theory—we make no curve fits on training trajectories, and we find transience even without models having the ability to learn circuits.

Kernel ridge regression as a model of learning dynamics. Kernel methods have proven useful as models of network dynamics and optimization (Chizat et al., 2019; Du et al., 2019; Bordelon & Pehlevan, 2021), thanks in large part to the neural tangent kernel (NTK) line of work (Neal, 1996; Lee et al., 2018; Jacot et al., 2018), in which suitably initialized wide networks train like kernel learners. Recent works show that the test error of kernel learners can be predicted from the kernel and task eigenstructure (Sollich, 2001; Bordelon et al., 2020; Jacot et al., 2020; Simon et al., 2021). This “KRR eigenframework” depends on the kernel’s eigenvalues and eigenfunctions with respect to the data distribution, and is reviewed in Section D.

Simplicity biases and statistics of increasing order. Neural networks are known to fit lowfrequency or low-complexity functions first (Nakkiran et al., 2019; Abbe et al., 2024). This suggests that the low-to-high ordering of learning task-space statistics we find for kernels may also hold for feature-learning networks.

Approximating the kernel eigenstructure. Our eigensystem-finding framework (Definition 4) follows the strategy of Karkada et al. (2026), by approximating the kernel as an inner-product kernel, creating subspaces of polynomials, and performing Gram-Schmidt against the data distribution. Our work trades understandability for expressivity: we cannot generally get closed form expressions for the eigenfunctions, but our framework is more widely applicable.

## 3 PRELIMINARIES

In-context linear regression. We study the in-context linear regression setting of Raventos et al.´ (2023), which is defined as follows. $\dot { \mathbf { A } } \mathbf { t }$ the outset, we construct a pool of task vectors $B _ { \mathrm { t r } } : =$ $\{ \beta ^ { ( t ) } \} _ { t = 1 } ^ { T }$ of $T$ tasks sampled $\beta ^ { ( t ) } \sim \mathcal { N } ( 0 , I _ { d } )$ to generate the training data. In this setting, one training sample is a context vector constructed as follows. First, a task $\bar { \boldsymbol \beta }$ is drawn uniformly from $B _ { \mathrm { t r } }$ . Second, $L + 1$ vectors $( { \pmb x } _ { i } ) _ { i = 1 } ^ { L + 1 }$ are sampled $\begin{array} { r } { \pmb { x } _ { i } \sim \mathcal { N } ( 0 , \frac { 1 } { d } \pmb { I } _ { d } ) . ^ { 2 } } \end{array}$ Third, $L + 1$ values $( y _ { i } ) _ { i = 1 } ^ { \bar { L } + 1 }$ are computed as $y _ { i } = \beta ^ { \top } \pmb { x } _ { i } + \eta _ { i }$ , with noise $\eta _ { i } \sim \mathcal { N } ( 0 , \epsilon ^ { 2 } )$ . Finally, we construct the context

$$
\boldsymbol { z } = \left[ \pmb { x } _ { 1 } ^ { \top } , y _ { 1 } , \ldots , \pmb { x } _ { L } ^ { \top } , y _ { L } , \pmb { x } _ { L + 1 } ^ { \top } \right] \in \mathbb { R } ^ { ( d + 1 ) L + d } ,\tag{1}
$$

with $y _ { L + 1 } = : y$ held out as the regression target. We will write $\mathcal { D } _ { \beta }$ for the joint distribution of $( z , y )$ conditioned on a given task vector $\beta$ . We refer to $T$ as the task diversity and L as the context length.

Kernel regression. We work in a standard supervised setting in which the model is trained on n contexts $\bar { \mathcal Z } = \left\{ z ^ { ( i ) } \right\} _ { i = 1 } ^ { n }$ and the corresponding scalar targets $\pmb { y } = \left( y ^ { ( i ) } \right) _ { i = 1 } ^ { n }$ . We take KR as the learning algorithm. KR is specified by a choice of positive-semidefinite kernel function $K ( z , z ^ { \prime } )$ and a choice of ridge parameter $\delta \geq 0$ . We consider various rotation-invariant kernel functions: $K ( z , z ^ { \prime } ) =$ $K ( z ^ { \top } z ^ { \prime } , \| z \| , \| z ^ { \prime } \| )$ . KR returns the predicted function $\hat { f } ( z ) = K _ { z \mathcal { Z } } \left( K _ { \mathcal { Z Z } } + \delta I _ { n } \right) ^ { - 1 } y \mathrm { , }$ , where $[ K _ { z \mathcal { Z } } ] _ { m } = K ( z , z ^ { ( m ) } )$ and $[ K _ { \mathcal { Z Z } } ] _ { k m } = K ( z ^ { ( k ) } , z ^ { ( m ) } )$

Evaluating the learned function. Once we have the learned function ${ \hat { f } } ,$ this will be evaluated two ways, on two distributions. First, we study its error on new contexts from the training task distribution: $\begin{array} { r } { \mathcal { D } _ { \mathrm { t r } } : = \frac { 1 } { T } \sum _ { \beta \in \mathcal { B } _ { \mathrm { t r } } } \mathcal { D } _ { \beta } } \end{array}$ . Second, we compute its error on the prior distribution, with contexts generated from fresh tasks: $\mathcal { D } _ { \mathrm { n o v e l } } : = \mathbb { E } _ { \beta \sim \mathcal { N } ( 0 , I _ { d } ) } [ \mathcal { D } _ { \beta } ]$ . We compute mean-squared error (MSE) over both distributions:

$$
\mathcal { E } _ { \mathrm { t r } } ( \hat { f } ) : = \underset { ( z , y ) \sim \mathcal { D } _ { \mathrm { t r } } } { \mathbb { E } } \left[ \left( \hat { f } ( z ) - y \right) ^ { 2 } \right] , \qquad \quad \quad \mathcal { E } _ { \mathrm { n o v e l } } ( \hat { f } ) : = \underset { ( z , y ) \sim \mathcal { D } _ { \mathrm { n o v e l } } } { \mathbb { E } } \left[ \left( \hat { f } ( z ) - y \right) ^ { 2 } \right] .\tag{2}
$$

Memorizing and generalizing solutions The memorizing solution is defined as that which minimizes error on the training distribution: $f _ { \mathrm { m e m } } : = \mathrm { a r g m i n } _ { f } \mathcal { E } _ { \mathrm { t r } } ( f )$ . As shown by Raventos et al.´ (2023), $f _ { \mathrm { m e m } }$ takes the discrete-minimum-mean-squared-error (dMMSE) form

$$
f _ { \mathrm { m e m } } ( z ) = \frac { \sum _ { t = 1 } ^ { T } w _ { t } ( z ) \cdot \boldsymbol { \beta } ^ { ( t ) } ^ { \top } \boldsymbol { x } _ { L + 1 } } { \sum _ { t = 1 } ^ { T } w _ { t } ( z ) } , \quad \mathrm { w h e r e } \quad w _ { t } = \exp \left( - \frac { 1 } { 2 \epsilon ^ { 2 } } \sum _ { i = 1 } ^ { L } \left( y _ { i } - \boldsymbol { \beta } ^ { ( t ) } ^ { \top } \boldsymbol { x } _ { i } \right) ^ { 2 } \right)\tag{3}
$$

The generalizing solution is the Bayes-optimal solution on the prior distribution: $f _ { \mathrm { g e n } } : =$ $\mathrm { a r g m i n } _ { f } \mathcal { E } _ { \mathrm { n o v e l } } ( \bar { f } )$ ). This is simply ridge regression on the context:

$$
f _ { \mathrm { g e n } } ( z ) =  { \boldsymbol { x } } _ { L + 1 } ^ { \top } \left(  { \boldsymbol { X } }  { \boldsymbol { X } } ^ { \top } + \epsilon ^ { 2 }  { \boldsymbol { I } } _ { d } \right) ^ { - 1 }  { \boldsymbol { X } }  { \boldsymbol { y } } _ { 1 : L } , \qquad \mathrm { w h e r e } ~  { \boldsymbol { X } } = [  { \boldsymbol { x } } _ { 1 } , \cdots ,  { \boldsymbol { x } } _ { L } ] .\tag{4}
$$

We will find that, as sample size n grows, KR first learns $f _ { \mathrm { g e n } }$ , then learns $f _ { \mathrm { m e m } } \mathrm { a s } n  \infty$

Moments and cumulants. We write $\begin{array} { r } { \hat { \mu } _ { \beta } : = \frac { 1 } { T } \sum _ { t } \beta ^ { ( t ) } } \end{array}$ for the mean of the training tasks $B _ { \mathrm { t r } }$ and $\hat { \Sigma } _ { \beta }$ for its covariance. We write $\hat { \kappa } _ { \ell }$ for the order-ℓ cumulant tensor of $\boldsymbol { B } _ { \mathrm { t r } }$

## 4 THE KERNEL EIGENSTRUCTURE

## 4.1 REVIEW OF THE KR EIGENFRAMEWORK

Prior work has obtained a predictive analytic framework which gives the generalization behavior of KR on essentially arbitrary distributions (see e.g. Simon et al. (2021)). This KR eigenframework takes as inputs the kernel eigensystem and the associated eigendecomposition of the target function. The eigensystem of a kernel function $K ( \cdot , \cdot )$ w.r.t. a distribution $\mathcal { D }$ is the set of scalar-function pairs $\lambda _ { i } , \phi _ { i }$ that satisfy $\begin{array} { r } { \mathbb { E } _ { z ^ { \prime } \sim \mathcal { D } } [ K ( z , z ^ { \prime } ) \phi _ { i } ] = \lambda _ { i } \phi _ { i } } \end{array}$ . A function $f ^ { * }$ is then decomposed into coefficients $v _ { i } = \mathbb { E } _ { z \sim \mathcal { D } } [ \phi _ { i } ( z ) f ^ { * } ( z ) ]$ . The KR eigenframework gives equations predicting quantities such as E $\begin{array} { r } { \left. \hat { f } \right. \approx \sum _ { i } v _ { i } \phi _ { i } } \end{array}$ and the test-time MSE. The main qualitative idea is that the target components $v _ { i } \phi _ { i }$ are learned in decreasing order of $\lambda _ { i } { \mathrm { : } }$ eigenfunctions with larger eigenvalue are learned first. This behavior is known as spectral bias.

## 4.2 THE KERNEL EIGENSYSTEM OF IN-CONTEXT LINEAR REGRESSION

In order to use the KR eigenframework in our ICL setting, we need to compute the task eigensystem and target eigencoefficients (or at least say enough to draw useful conclusions). The training data are sampled $z \sim \mathcal { D } _ { \mathrm { t r } }$ and have target function E $[ y | z ] = f _ { \mathrm { m e m } } ( z )$ . We thus need to eigendecompose a rotation-invariant kernel K w.r.t. distribution ${ \mathcal { D } } _ { \mathrm { t r } }$

This eigensystem cannot generally be found in closed form, so to make progress, we invoke a series of assumptions and approximations. We give a sketch here and a derivation in Section E. First, following Karkada et al. (2026), we approximate the kernel as an inner-product kernel $\tilde { K } ( z , z ^ { \prime } ) =$ $\begin{array} { r } { \sum _ { \ell > 0 } \frac { c _ { \ell } } { \ell ! } ( z ^ { \top } z ^ { \prime } ) ^ { \ell } } \end{array}$ , obtaining the constants $\{ c _ { \ell } \}$ by matching K and $\tilde { K }$ on the sphere $r \cdot \mathbb { S } ^ { \dim ( z ) - 1 }$ with $r = \operatorname { r m s } { [ | | z | | ] }$ ]. Then, as in that same paper, we assume $c _ { \ell } \gg c _ { \ell + 1 }$ , which implies that the kernel eigensystem can be found by:

1. taking the subspaces $( \mathcal { P } _ { \leq \ell } ) _ { \ell = 0 } ^ { \infty } ,$ , where $\mathcal { P } _ { \leq \ell }$ contains all polynomials of z up to order $\ell ;$

2. performing Gram-Schmidt on these subspaces in order of increasing ℓ to obtain orthogonal subspaces $( \tilde { \mathcal { P } } _ { \leq \ell } ) _ { \ell = 0 } ^ { \infty } \dot { : }$ ; and finally

3. diagonalizing each subspace $\tilde { \mathcal { P } } _ { < \ell }$ w.r.t. ${ \mathcal { D } } _ { \mathrm { t r } }$

This procedure may appear more difficult and obscure than simply diagonalizing the empirical kernel numerically: for example, the ℓ-th subspace has dimension $( ( d + 1 ) L + d ) ^ { \ell }$ , which quickly exceeds the dimension n of the empirical kernel’s eigenspace for even moderate values of $d , L ,$ , and ℓ. However, symmetries in the problem setup substantially reduce the complexity in every subspace. Specifically, because the in-context sample-label pairs are permutation invariant, the subspaces can be Kronecker-factorized (reducing the effective dimension to roughly $( d + 1 ) ^ { \ell } )$ and further decomposed into an L-scaling rank-d spike plus a bulk term. At large context lengths, the leading eigenfunctions can be found in closed form, while the remainder must be computed numerically (due to the unsolvability of the quintic).

Luckily, the most important sets of the eigenfunctions (those with non-zero target coefficients) are those that can be found in closed-form. Despite being analytical, these functions are highly complex (see Sections F and 7). Here, we give said eigenfunctions in a more comprehensible, reduced form:

• at order one, there are d “linear readout” modes $\phi _ { 1 } ( z ) = \pmb { x } _ { L + 1 }$ , which have eigenvalue $\textstyle \lambda _ { 1 } = { \frac { c _ { 1 } } { d } }$ ;

• at order three, there are again d eigenmodes (see Table 2), with the most significant for the kernel learner being a single “regression” component

$$
\phi _ { 3 } ( z ) \approx \frac { d } { L + d + 1 } \left( \sum _ { i = 1 } ^ { L } y _ { i } \pmb { x } _ { i } ^ { \top } - \frac { L } { d } \hat { \pmb { \mu } } _ { \beta } ^ { \top } \right) \pmb { x } _ { L + 1 } ,\tag{5}
$$

which has eigenvalue $\lambda _ { 3 } ~ \propto ~ L _ { d ^ { 3 } } ^ { c _ { 3 } } ~ ( c _ { 3 } / c _ { 1 } ~ \sim ~ L ^ { - 1 } )$ . Note that $\begin{array} { r } { \sum _ { i } y _ { i } \pmb { x } _ { i } ^ { \top } \approx \frac { L } { d } \pmb { \beta } ^ { \top } } \end{array}$ as $\begin{array} { r } { \sum _ { i } \pmb { x } _ { i } \pmb { x } _ { i } ^ { \top } \approx \frac { L } { d } \pmb { I } _ { d } } \end{array}$ for L ≫ d;

• at order $2 \ell + 1 ( \ell \geq 2 )$ , there are $d ^ { \ell + 1 } / \ell ! \cdot$ “memorizing” components taking the form

$$
\phi _ { 2 \ell + 1 } ( z ) \sim { \frac { 1 } { L ^ { 1 + \ell } } } \left( \sum _ { i } ^ { L } y _ { i } \pmb { x } _ { i } - . . . \right) ^ { \otimes \ell } \pmb { x } _ { L + 1 } ,\tag{6}
$$

where we have dropped many highly involved prefactors and denoted orthogonalization against lower orders with dots.

In general, the top eigenvalues at order ℓ scale like $c _ { \ell } \cdot L ^ { \left( \ell - 1 \right) / 2 } \cdot d ^ { - \ell } \left( c _ { \ell } \right.$ scaling as $L ^ { - \ell / 2 } )$ . With $c _ { \ell } \gg c _ { \ell + 1 }$ , the target-relevant eigenfunctions are learned in ascending order. We experimentally check our eigensystem against a true numerical eigensystem computed from a finite trainset for several common rotation-invariant kernels in Section E and Figure 8. We see very good numerical agreement, which validates the assumptions we have made up to this point. In particular, it confirms that these kernels will indeed learn target function components in order of increasing degree.

## 4.3 EIGENDECOMPOSING THE GENERALIZING AND MEMORIZING SOLUTIONS

We must now find the eigencoefficients of $f _ { \mathrm { g e n } }$ and $f _ { \mathrm { m e m } }$ . We again sketch our results here and defer a full treatment to Section F.

We begin with decomposing $f _ { \mathrm { m e m } }$ . While the function appears highly complicated to extract coefficients from, we can use the law of total expectation (see Lemma 1) to find the eigencoefficients without ever using $f _ { \mathrm { m e m } } { \mathrm { : } }$

$$
v _ { i , \mathrm { m e m } } \equiv \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { \mathrm { t r } } } \left[ f _ { \mathrm { m e m } } ( z ) \phi _ { i } ( z ) \right] = \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { \mathrm { t r } } } [ y \phi _ { i } ( z ) ] .\tag{7}
$$

Substituting $f _ { \mathrm { m e m } }$ for $y ~ = ~ y _ { L + 1 }$ makes finding the eigencoefficients relatively simple. Using $\mathbb { E } _ { { \pmb x } \sim \mathcal { N } ( 0 , I _ { d } / d ) } [ y _ { k } { \pmb x } _ { k } ^ { \top } ] \sim \mathbb { E } _ { { \pmb x } \sim \mathcal { N } ( 0 , I _ { d } / d ) } [ \beta ^ { \top } { \pmb x } _ { k } { \pmb x } _ { k } ^ { \top } ] \sim \beta ^ { \top } \bar { { \cal I } } _ { d } / d$ for any $k ,$ we expect the order $2 \ell +$ 1 eigenfunctions described earlier to carry eigencoefficients that are (to dropped prefactors) $\propto$ $\mathbb { E } _ { \mathcal { D } _ { \mathrm { t r } } } \big [ \beta ^ { \otimes \ell + 1 } \big ]$ . In other words, the $2 \ell + 1$ -th eigencoefficients are defined using up to the $\ell + 1 - \operatorname { t h }$ cumulants of $\mathcal { D } _ { \mathrm { t r } } ;$ cumulants as opposed to moments due to orthogonalization against lower order terms. By symmetry, all even-order (2ℓ-th) eigencoefficients of $f _ { \mathrm { m e m } }$ are zero.

$f _ { \mathrm { g e n } }$ can now be easily obtained. $f _ { \mathrm { g e n } }$ can be decomposed in exactly the same way as $f _ { \mathrm { m e m } } .$ , with the distribution ${ \mathcal { D } } _ { \mathrm { t r } }$ changed to $\mathcal { D } _ { \mathrm { n o v e l } } \breve { ( \mathcal { N } ( 0 , \pmb { I } _ { d } ) ) }$ ). The Gaussian distribution is the unique distribution with only a first and second cumulant. Therefore, $f _ { \mathrm { g e n } }$ is understood as being primarily composed of the degree 3 “regression” eigenfunction. The degree 1 “linear readout” is not included as we draw $\beta$ from a mean-0 distribution. Equivalently, one can observe that as $L / d \to \infty , X X ^ { \top } \to$ const · $\mathbf { \delta } _ { I _ { d } }$ such that $f _ { \mathrm { g e n } }$ approaches the third-order polynomial $\pmb { x } _ { L + 1 } ^ { \top } \pmb { X } \pmb { y } _ { 1 : L }$

![](images/8c09df182217689c19b536f1ab304e355954b217a337110887fbcd9a156ae676.jpg)  
Figure 2: Transience is explained by learning higher order task statistics. We train $l a z y$ random features networks (RFNs) (see Section B.2 for a description of RFNs) with online SGD on ICL linear regression data. Using RFNs, we empirically find that the error on novel tasks rises when the higher degree terms are learned, as was theoretically predicted in Figure 1. We observe the same three phases of predicting from the mean task alone, then generalizing to any task, followed by memorizing using higher-degree terms. (Top) Test MSE on novel tasks $\beta \sim \mathcal { D } _ { \mathrm { n o v e l } } .$ , with the memorizing (dMMSE) and generalizing (ridge) oracles for reference. Unlike the kernels trained in Figure 1, the random feature networks train with enough samples to empirically probe phase 3. (Bottom) Overlap of the learned predictor with the leading terms of the memorizing solution, $\mathbb { E } [ \hat { f } g _ { i } ] / \mathbb { E } [ g _ { i } ^ { 2 } ]$ . Similar to Figure 1, the functions $g _ { i }$ are determined by the i-th order statistics of $B _ { \mathrm { t r } }$ , and are explicitly given in Section B. Novel-task error drops below the memorizing oracle as $\phi _ { 3 }$ is learned $( n \approx \dot { 1 0 } ^ { 5 } – 1 \dot { 0 } ^ { 6 } )$ and is lowest once ϕ<sub>3</sub> $( g _ { 1 } )$ saturates. The error then rises back toward the memorizing oracle as the network learns higher order statistics through $g _ { 2 } ( n \gtrsim 1 0 ^ { 7 } )$ . Further experimental details are in Section B.

$f _ { \mathrm { m e m } }$ truncated at order 3 approximates the exact same third-degree polynomial as $f _ { \mathrm { g e n } }$ . Thus, $f _ { \mathrm { g e n } }$ is approximately a degree-3 polynomial truncation of $f _ { \mathrm { m e m } }$ . Past the degree-3 truncation, f<sub>mem</sub> contains higher-degree polynomials that do not exist in $f _ { \mathrm { g e n } }$ , driven by the higher order statistics of the random, finite draws of $\beta ^ { ( t ) }$

## 4.4 THE RESULTING PICTURE OF MEMORIZATION AND GENERALIZATION

As the training sample size n grows, KR will learn successively lower-eigenvalue (i.e. higher-degree) components of $f _ { \mathrm { m e m } }$ , on which it is trained. The kernel will learn the third-order components of $f _ { \mathrm { m e m } }$ with approximately $L ^ { 2 } d ^ { 2 } / 2$ samples (there are roughly $L ^ { 2 } d ^ { 2 } / 2$ quadratic eigenmodes that have larger eigenvalue). With these third-order components learned, the kernel’s function (given a 3 subscript to denote the order of truncation) becomes approximately

$$
\hat { f } _ { \mathrm { k e r } , 3 } \approx \hat { \pmb { \mu } } _ { \beta } ^ { \top } \pmb { x } _ { L + 1 } + \frac { d } { L + d + 1 } \left( \sum _ { i } y _ { i } \pmb { x } _ { i } ^ { \top } - \frac { L } { d } \hat { \pmb { \mu } } _ { \beta } ^ { \top } \right) \pmb { x } _ { L + 1 } ,\tag{8}
$$

which approximates $f _ { \mathrm { g e n } }$ , as is seen experimentally in Figure $1 . ^ { 3 }$ As n grows further, $\hat { f }$ picks up yet higher-order components of $f _ { \mathrm { m e m } }$ , eventually arriving at the memorizing solution. In Figure $2 .$

we validate how a model trained without feature learning approximates $f _ { \mathrm { g e n } }$ when it has learned the low-order eigenfunctions, and begins memorizing using higher-order eigenfunctions.

## 5 TRANSIENCE IS AN UNDERFITTING PHENOMENON

Section 4 showed that $f _ { \mathrm { g e n } }$ is approximately a low-degree truncation of $f _ { \mathrm { m e m } }$ (and as we will later see, Section 6 gives the same picture in the task space). This is enough to explain the “transience of ICL” phenomenon, in which models first generalize and then memorize.

With this simple picture, we are now in a position to resolve the paradox we initially posed: why can a model generalize even though it is optimized solely using the training objective with finite training tasks? There is a simple line of logic here. Models are always minimizing the training objective, which is achieved by implementing the memorizing solution. While minimizing this training objective, they implement a low-degree truncation of the memorizing solution, which is nothing but an approximation of the generalizing solution.

Putting the theory to work. While this is a satisfying conclusion, there are still several further questions that we now have the ability to resolve. In the following sections, we begin by discussing how to map the found eigenfunctions into the more problem-intuitive “task space” where the task vectors live. We then describe why even an optimally truncated model will typically not reach the generalizing solution. We will similarly analyze the effect of “task diversity,” and explain the main findings of Raventos et al.´ (2023). Finally, in Section 8, we show that even transformers, models that do feature learn, implement the same cumulant-truncated solution described by this non-feature-learning theory.

## 6 THE LEARNING DYNAMICS IN TASK SPACE

The kernel’s eigenfunctions, and therefore computation, are polynomials of the raw data space (i.e. of X and Y), but most questions of ICL are asked in the more intuitive task space B. Our theory offers a bridge between the two spaces. All eigenfunctions with non-zero eigencoefficient share one key property allowing us to move between the two spaces—they are linear in ${ \pmb x } _ { L + 1 }$ . The kernel prediction can thus be written as a linear weight applied to the unpaired input $\pmb { x } _ { L + 1 } \colon \hat { f } _ { \mathrm { k e r } } = \hat { \beta } ^ { \top } \pmb { x } _ { L + 1 }$ for some estimated task ${ \hat { \beta } } .$ . While this estimated task is a function of the random input z and not just the underlying task $\beta ^ { * }$ , we can nevertheless predict what the expected task a kernel predicts is in terms of only the pretraining task distribution $\mathcal { D } _ { \mathrm { t r } } .$ , as well as the true task $\beta ^ { * }$ . We can similarly decompose both $f _ { \mathrm { g e n } }$ and $f _ { \mathrm { m e m } }$ into a predicted task applied to the unpaired input.<sup>4</sup> For further information on the task/data-space duality, refer to Section G.

## 6.1 EVALUATING PREDICTORS IN TASK SPACE

Let $\beta _ { ( \cdot ) } ^ { \top } \pmb { x } _ { L + 1 } \equiv \mathbb { E } _ { \pmb { x } , \eta } \big [ f _ { ( \cdot ) } ( z ) | \beta ^ { \star } \big ]$ for (·) being any of mem, gen, or ker and $\beta _ { ( \cdot ) }$ being a function of $\beta ^ { * }$ . Similar to Section 4, we can describe $\beta _ { \mathrm { m e m } }$ and $\beta _ { \mathrm { g e n } }$ as a sum of terms, thanks to the linearity of the expectation. Each factor of $y _ { k } \pmb { x } _ { k } ^ { \top }$ evaluates to ${ \scriptstyle { \frac { L } { d } } } \beta ^ { * }$ under the expectation, causing the degree $2 \ell + 1$ eigenfunctions previously discussed to yield degree ℓ functions of $\beta *$ (the +1 is dropped, being the ${ \pmb x } _ { L + 1 }$ term).

Before proceeding, we caution that the $2 \ell + 1$ eigenfunctions $\to \ell$ functions of $\beta ^ { * }$ mapping does not always hold. In particular, we note there exists a degree 5 eigenfunction with leading term $\begin{array} { r } { \frac { 1 } { L ^ { 3 } } \big ( \sum _ { i , j } \dot { y } _ { i } \pmb { x } _ { i } ^ { \top } ( \pmb { x } _ { j } \pmb { x } _ { j } ^ { \top } - \pmb { I } _ { d } ) - \dots \big ) \pmb { x } _ { L + 1 } } \end{array}$ , which does indeed appear in the kernel, memorizing, and generalizing solutions. Many higher order terms appear, although this is learned first and with the greatest coefficient. As this eigenfunction contains a single $y _ { i }$ , it becomes a linear function of $\beta ^ { * }$ in the task space picture. (These functions were not previously mentioned as they make only $\mathcal { O } ( L ^ { - 1 } )$ contributions to all functions, and don’t greatly impact the overarching narrative.)

Evaluating the per-term generalizing and memorizing predicted tasks, we get,

$$
\begin{array} { r } { \beta _ { \mathrm { g e n } } ( \beta ^ { * } ) \approx { \frac { L } { L + d + 1 } } \beta ^ { * } + { \frac { ( d + 1 ) L } { ( L + d + 1 ) ( L + d + 3 ) } } \beta ^ { * } + \dots , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { \beta _ { \mathrm { m e m } } ( \beta ^ { * } ) \approx \hat { \mu } _ { \beta } + \frac { L } { L + d + 1 } \left( \beta ^ { * } - \hat { \mu } _ { \beta } \right) + \cdots + \frac { ( d + 1 ) L } { ( L + d + 1 ) ( L + d + 3 ) } \beta ^ { * } + \dots . } \end{array}\tag{10}
$$

![](images/5e472bbd6d9f765e221283f80a07921993902d07301bfbdc182f08b1fc2434d8.jpg)  
Figure 3: Random feature models can perform ICL even with only $T = d + 1$ tasks. We train a random feature network using SGD with exactly $T = 4 = d + 1$ tasks $( d = 3 )$ , with context length $L = 9 6$ . We manually construct a training task pool such that $\hat { \Sigma } = \pmb { I } _ { d }$ and $\hat { \mu } = \mathbf { 0 }$ . As is predicted in Section 7, this training task distribution allows the model to near-perfectly generalize once the optimal degree-3 term $\textstyle \sum _ { i } y _ { i } { \bar { \pmb { x } } } _ { i } ^ { \top } { \pmb { x } } _ { L + 1 }$ is learned. We evaluate the random features model’s test error on novel tasks $\beta ^ { \star } \sim \mathcal { D } _ { \mathrm { n o v e l } }$ , normalized by $\mathbb { E } [ y ^ { 2 } ]$ . When the model has not seen many samples in phase one, the model uses the average training task, which we manually set to 0. At approximately ${ 1 0 } ^ { \dot { 6 } }$ samples, the model enters phase two and begins fitting the generalizing degree-3 term. Even though there are only 4 training tasks, the model does not memorize the tasks with up to $1 0 ^ { 9 }$ samples! Given infinite training samples, the model will eventually memorize. Further details in Section B.

To leading order in $L ,$ the low-degree truncation of the memorizing solution’s predicted task is still nothing but an approximation of the true underlying task. A kernel will learn each term of the memorizing weight sequentially: first using just the global mean task $\hat { \mu } _ { \beta }$ , then partially forgetting it in favor of approximating the true weight $\beta ^ { * }$ and generalizing, finally proceeding to compute higher order functions of $\beta ^ { * }$ to exactly memorize the training tasks. The dynamical picture in the task space is given in Figure 1 (left).

## 7 WHAT CAUSES EVEN OPTIMALLY-STOPPED MODELS TO (NOT) REACH $f _ { \mathrm { g e n } } ?$

While a model truncated to implement only its degree 3 eigenfunctions approximates the generalizing solution, there are a number of different effects that can improve the quality of the approximation. We begin with a discussion of the one present throughout every previous section, the finite context length, before delving into how non-Gaussianity drives memorization.

## 7.1 FINITE CONTEXT LENGTH L

Visible in Equations (8) and (10), the degree 3 (data) eigenfunction carries a prefactor $\frac { L } { L + d + 1 }$ , after taking into account the summation across context. That is, models truncated to polynomial degree 3 capture a $\frac { L } { L + d + 1 }$ fraction of the generalizing solution, which converges rather slowly to 1 as L grows. Despite this slow convergence, models that implement this polynomial degree 3 truncation of $f _ { \mathrm { m e m } }$ are often achieving their lowest possible loss on novel tasks; implementing higher degree terms will generally raise this loss. However, in cases where the training tasks are skew-less and do not express the memorizing-only degree 5 eigenfunction (see Equation (6)), the model is able to capture a further $\frac { L ( d + 1 ) } { ( L + d + 1 ) ( L + d + 3 ) }$ fraction of the generalizing solution by learning the (lower-eigenvalue) degree 5 eigenfunction discussed in Section 6.1. We conjecture that feature-learning models often learn this generalization-useful term earlier than the naive kernel eigenvalue ordering, when the tasks have a low skewness.

## 7.2 ANISOTROPY IN THE DRAWN TASKS

While we have been using $\begin{array} { r } { \phi _ { 3 } \sim \frac { d } { L + d + 1 } ( \sum _ { i } y _ { i } \pmb { x } _ { i } ^ { \top } - \frac { L } { d } \hat { \pmb { \mu } } _ { \beta } ) \pmb { x } _ { L + 1 } } \end{array}$ throughout to denote $t h e ^ { \textbf { * } }$ “generalizing” eigenfunction, this is only a true eigenfunction when the training tasks have an isotropic covariance. As a very simple demonstration of this effect, we train a network on a $d = 3$ dimension setup with only $T = 4$ tasks carefully selected to have isotropic covariance. With only 4 tasks, the lazy model is able to generalize, as is shown in Figure 3. The accurate “generalizing” eigenfunctions therefore are defined using the training task’s covariance (specifically its singular value decomposition), and any anisotropies will induce error in the novel task prediction. For a complete description of the eigenfunctions, refer to Table 2.

![](images/1f8a56543ed7587d2c037543fcc87b328d109b2ae24615aef1dc075e71a04b3d.jpg)  
Task Diversity $T$  
Figure 4: Task diversity trades pretraining-task accuracy for novel-task accuracy. Following Raventos et al. ´ (2023), we sweep the number of training tasks $T$ and report test MSE on the pretraining tasks ${ \mathcal { D } } _ { \mathrm { t r } }$ (left) and on novel tasks $\mathcal { D } _ { \mathrm { n o v e l } }$ (right). Our data is drawn with $d = 8 , L = 1 5 , \bar { \epsilon } ^ { 2 } = 0 . 2 5$ In blue, we show a fit random features network trained with SGD (see Section B for random feature network details), with the memorizing (dMMSE) and generalizing (ridge) oracles shown. $\mathrm { A s } T$ grows, the training task distribution’s cumulants approach that of the prior’s (Gaussian) and all three curves converge. Evaluated on novel tasks, the random features model initially tracks the memorizing solution up to a task diversity $T$ of 4, continuing to approach the generalizing solution until $T \approx 2 ^ { 6 } = 6 4$ tasks. All models are trained with enough samples such that they can fully realize the optimal degree-3 term, indicated by the green background. We note two exceptions: a blue background for one training task is used to signify that the model uses the mean task alone, and the seafoam background for two tasks signifies the mean task still being heavily utilized. The trained models do not exactly reach the generalizing solution, in part due to the finite context length L we use for training (see Section 7). Further experimental details are in Section B.

## 7.2.1 ANISOTROPIC INPUTS

While anisotropy in the sampled tasks affects downstream performance, we can also ask how sampling anisotropic inputs x affect the learned solution. We point to Section F for a complete description, but can note that $\beta ^ { \top } { \pmb x } _ { i } = \left( \beta ^ { \top } \Sigma _ { { \pmb x } } ^ { 1 / 2 } \right) \left( \Sigma _ { { \pmb x } } ^ { 1 / 2 } { \pmb x } _ { i } \right)$ , meaning anisotropic inputs can be viewed as whitened inputs with the tasks themselves being colored.

## 7.3 TASK DIVERSITY/NON-GAUSSIANITY OF TASKS

If models learn the generalizing solution by learning only low-degree functions of the memorizing solution—those defined by the Gaussian statistics of the training tasks—how should we understand the task diversity picture of (Raventos et al.´ , 2023)? In their work, providing a transformer more task diversity caused generalization, and transformers given fewer tasks would memorize. The story of task diversity is actually independent of the transformer architecture (as was hinted in Tong & Pehlevan (2025)), not even needing feature learning to obtain the headline results (see Figure 4).

As task diversity increases, models begin generalizing in large part due to the training task statistics becoming both more Gaussian, and more isotropic. As the tasks are drawn from an isotropic distribution, more tasks causes the sampled covariance to become increasingly white.<sup>5</sup> Furthermore, drawing more tasks causes the higher order cumulants to decrease, decreasing the scale of the memorization-only eigencoefficients. Both of these effects are further discussed in Section H.4.

## 8 TRANSFORMERS LEARN $f _ { \mathrm { g e n } }$ USING GAUSSIAN STATISTICS

We now arrive at the often uncomfortable question: having done theory on models unable to do feature learning, can we say anything about those that have adaptive features? We find the answer here is yes: even transformer models trained with the Adam optimizer learn to memorize by exploiting the non-Gaussian statistics of their training tasks.<sup>6</sup>

![](images/1e946d3bcd87140dd396f43e808ab8d6b06fb83a83fef4792fe4b96ebf2bd267.jpg)

(b) Training tasks  
![](images/7b9b319522800aab23c08ccf1540fe19ab9239db377f2fb79ca8b3a4dd89c33a.jpg)  
Figure 5: Transformers memorize through non-Gaussian statistics of their training tasks. We train 12-layer GPT-2-style transformers with Adam on $T = 2 5 6$ training tasks $( d = 8 , L = 1 6 , \epsilon ^ { 2 } = 0 . 2 5 )$ and evaluate the test MSE on (a) fresh tasks and (b) the training tasks. All pools are centered and whitened (see Section 7.2 for a discussion of the implications) and differ principally in their fourth cumulant. Whiten consists of i.i.d. Gaussian draws, partial kurtosis is made of a symmetric pool whose $\hat { \kappa } _ { 4 }$ is rescaled to 0.3× that of its initial Gaussian pool, no kurtosis uses a symmetric pool with $\hat { \kappa } _ { 4 } = 0 .$ Dashed and dotted lines mark the generalizing (ridge) and memorizing (dMMSE). In (a), each pool’s dMMSE is shown to highlight that the slight difference in oracle does not explain the onset of task memorization. All models first approach the generalizing solution, with the scale of the kurtosis setting when memorization begins. Further experimental details are in Section B.

Shown in Figure 5, we fix a total number of tasks, and train transformers on datasets containing various amounts of training task kurtoses (having been drawn symmetrically such that there is no skewness). When the model is given no kurtosis, it remains generalizing throughout the entire training procedure. When the model is given only part of the full kurtosis, it will stay generalizing longer than a model given a typical kurtosis. This is the ordering our theory predicts: with all odd cumulants removed by the symmetric pool, the first memorizing term is the degree-7 mode carrying $\hat { \kappa } _ { 4 }$ (Theorem 1). Removing $\hat { \kappa } _ { 4 }$ leaves only $\hat { \kappa } _ { \geq 5 }$ to memorize, which occurs much later in training.

## 9 FURTHER DISCUSSION

Our analysis is restricted to the in-context linear regression task with Gaussian inputs. While we believe the generalizing solution is contained within low-degree functions of the memorizing solution for ICL settings, and is thus the cause of generalization in ICL, we do not attempt to show this in full generality. We leave future work to characterize the effects of learning nonlinear ICL solutions or how natural-language affects the learned function. Furthermore, the memorization phase is particularly difficult to reach for models unable to perform feature learning; despite this, Figure 5 shows that our account of memorization—sequentially learning higher order statistics of the training tasks—holds even in transformer models.

In closing, we remind that our results were derived from simple, solvable models, and yet have provided insight into an aspect of transformers performing a seemingly complex task. We therefore hope this work shows the usefulness of solvable models, even ones that might initially seem ineffective for the problem at hand, to advance our understanding of nebulous phenomena.

## ACKNOWLEDGMENTS

We are grateful to Dhruva Karkada, Blake Bordelon, Michael DeWeese, Sultan Daniels, Divit Rawal, and Berkan Ottlik for useful discussion on key results. We further thank Dhruva Karkada, Berkan

Ottlik, Raymond Fan, Samyak Jain, and Mark Rhee for providing feedback on the manuscript. In addition to the above, JT thanks Amy Flather and Soren Bear for listening to his early rants about why ICL is important and should be studied using kernels, as well as Josh Albrecht and Kanjun Qiu for continuing to support his fundamental AI research efforts.

This work was funded by Imbue under the Feature Lab (FLAB) initiative.

STATEMENT ON THE USE OF LARGE LANGUAGE MODELS

We used large language models (LLMs) for writing code, giving a precise expansion of the Ridge and dMMSE functions in terms of cumulants and drafting their appendices, refining the mathematics of high order subkernel diagonalization, helping clarify the position of related works, and constructing proofs. As LLMs often made various implicit assumptions about the structure of the empirical cumulants, we independently verified all outputs of LLMs.

## REFERENCES

Emmanuel Abbe, Enric Boix-Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks, 2024. URL .

Francis Bach. High-dimensional analysis of double descent for linear regression with random projections. arXiv preprint arXiv:2303.01372, 2023.

Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc Mezard. Dynamical regimes of diffusion´ models. Nature Communications, 15(1), November 2024. ISSN 2041-1723. doi: 10.1038/ s41467-024-54281-3. URL .

Tony Bonnaire, Raphael Urfin, Giulio Biroli, and Marc M¨ ezard. Why diffusion models don’t´ memorize: The role of implicit dynamical regularization in training, 2025. URL .

Blake Bordelon and Cengiz Pehlevan. Learning curves for sgd on structured features. arXiv preprint arXiv:2106.02713, 2021.

Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In International Conference on Machine Learning, pp. 1024–1034. PMLR, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature communications, 12(1):1–12, 2021.

Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Out-of-distribution generalization in kernel regression, 2022. URL .

Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7(3):331–368, 2007.

Liam Carroll, Jesse Hoogland, Matthew Farrugia-Roberts, and Daniel Murfet. Dynamics of transient structure in in-context linear regression transformers, 2025. URL .

Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent incontext learning in transformers, 2022. URL .

Chen Cheng and Andrea Montanari. Dimension free ridge regression. arXiv preprint arXiv:2210.08571, 2022.

Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in Neural Information Processing Systems, 32, 2019.

Hugo Cui, Freya Behrens, Florent Krzakala, and Lenka Zdeborova. A phase transition between´ positional and semantic learning in a solvable model of dot-product attention. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pp. 36342–36389. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024. doi: 10.52202/079017-1146. URL .

Edgar Dobriban and Stefan Wager. High-dimensional asymptotics of predictions: Ridge regression and classification. The Annals ofStatistics, 46(1):247–279, 2018.

Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning, pp. 1675–1685. PMLR, 2019.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes, 2023. URL .

Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in highdimensional ridgeless least squares interpolation. The Annals ofStatistics, 50(2):949–986, 2022.

Arthur Jacot, Clement Hongler, and Franck Gabriel. Neural tangent kernel: Convergence and gener-´ alization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Arthur Jacot, Berfin S¸ims¸ek, Francesco Spadaro, Clement Hongler, and Franck Gabriel. Kernel´ alignment risk estimator: Risk prediction from training data. Advances in neural information processing systems, 33:15568–15578, 2020.

Mason Kamb and Surya Ganguli. An analytic theory of creativity in convolutional diffusion models, 2025. URL .

Dhruva Karkada, Joseph Turnbull, Yuxi Liu, and James B. Simon. Predicting kernel regression learning curves from only raw data statistics, 2026. URL .

Louis Kirsch, James Harrison, Jascha Sohl-Dickstein, and Luke Metz. General-purpose in-context learning by meta-learning transformers, 2024. URL .

Dmitry Krotov and John J Hopfield. Dense associative memory for pattern recognition, 2016. URL .

Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. Deep neural networks as gaussian processes. In International Conference on Learning Representations (ICLR), 2018.

Mary I. Letey, Jacob A. Zavatone-Veth, Yue M. Lu, and Cengiz Pehlevan. Pretrain-test task alignment governs generalization in in-context learning, 2025. URL .

Xiang Li, Yixiang Dai, and Qing Qu. Understanding generalizability of diffusion models requires rethinking the hidden gaussian structure, 2024. URL .

Yingcong Li, Davoud Ataee Tarzanagh, Ankit Singh Rawat, Maryam Fazel, and Samet Oymak. Gating is weighting: Understanding gated linear attention through in-context learning, 2025. URL

Bruno Loureiro, Cedric Gerbelot, Hugo Cui, Sebastian Goldt, Florent Krzakala, Marc Mezard, and Lenka Zdeborova. Learning curves of generic features maps for realistic datasets with a´ teacher-student model. Advances in Neural Information Processing Systems, 34:18137–18151, 2021.

Yue M. Lu, Mary Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan. Asymptotic theory of in-context learning by linear attention. Proceedings ofthe National Academy ofSciences, 122(28), 2025. ISSN 1091-6490. doi: 10.1073/pnas.2502599122. URL .

Artem Lukoianov, Chenyang Yuan, Justin Solomon, and Vincent Sitzmann. Locality in image diffusion models emerges from data statistics, 2025. URL .

Preetum Nakkiran, Gal Kaplun, Dimitris Kalimeris, Tristan Yang, Benjamin L. Edelman, Fred Zhang, and Boaz Barak. Sgd on neural networks learns functions of increasing complexity, 2019. URL .

Radford M Neal. Priors for infinite networks. In Bayesian Learning for Neural Networks, pp. 29–53. Springer, 1996.

Alex Nguyen and Gautam Reddy Nallamala. Differential learning kinetics govern the transition from memorization to generalization during in-context learning. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 8677–8697, 2025. URL .

Core Francisco Park, Ekdeep Singh Lubana, Itamar Pres, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning, 2025. URL .

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International Conference on Machine Learning, pp. 5301–5310. PMLR, 2019.

Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis (eds.), Advances in Neural Information Processing Systems, volume 20, pp. 1177–1184. Curran Associates, Inc., 2007.

Hubert Ramsauer, Bernhard Schafl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler,¨ Lukas Gruber, Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, Victor Greiff, David´ Kreil, Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. Hopfield¨ networks is all you need, 2021. URL .

Allan Raventos, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the´ emergence of non-bayesian in-context learning for regression, 2023. URL .

Dominic Richards, Jaouad Mourtada, and Lorenzo Rosasco. Asymptotics of ridge (less) regression under general source condition. In International Conference on Artificial Intelligence and Statistics, pp. 3889–3897. PMLR, 2021.

Christopher Scarvelis, Haitz Saez de Oc´ ariz Borde, and Justin Solomon. Closed-form diffusion´ models, 2025. URL .

James B Simon, Madeline Dickens, Dhruva Karkada, and Michael R DeWeese. The eigenlearning framework: A conservation law perspective on kernel ridge regression and wide neural networks. arXiv preprint arXiv:2110.03922, 2021.

Aaditya K. Singh, Stephanie C. Y. Chan, Ted Moskovitz, Erin Grant, Andrew M. Saxe, and Felix Hill. The transient nature of emergent in-context learning in transformers, 2023. URL .

Aaditya K. Singh, Ted Moskovitz, Sara Dragutinovic, Felix Hill, Stephanie C. Y. Chan, and Andrew M. Saxe. Strategy coopetition explains the emergence and transience of in-context learning, 2025. URL .

Peter Sollich. Gaussian process regression with mismatched models. In Advances in Neural Information Processing Systems, pp. 519–526. MIT Press, 2001.

William L. Tong and Cengiz Pehlevan. Mlps learn in-context on regression and classification tasks, 2025. URL .

Alexander Wei, Wei Hu, and Jacob Steinhardt. More than a toy: Random matrix models predict how real-world neural representations generalize. In International Conference on Machine Learning, Proceedings of Machine Learning Research, 2022a.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models, 2022b. URL .

Denny Wu and Ji Xu. On the optimal weighted ℓ<sub>2</sub> regularization in overparameterized linear regression. Advances in Neural Information Processing Systems, 33:10112–10123, 2020.

Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy, and Noah D. Goodman. In-context learning strategies emerge rationally, 2025. URL .

Zhi-Qin John Xu, Yaoyu Zhang, and Yanyang Xiao. Training behavior of deep neural network in frequency domain. In International Conference on Neural Information Processing, pp. 264–274. Springer, 2019.

## A REVIEW OF NOTATION

As notation in ICL literature evolves constantly, we provide a table to reference all important quantities.

<table><tr><td rowspan=1 colspan=2>Symbol</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Data and task distribution</td></tr><tr><td rowspan=1 colspan=2> $d$ </td><td rowspan=9 colspan=1>In-context data dimensionContext lengthNumber of pretraining tasksNumber of training samples (contexts) supplied to the kernelGeneral notation for tasksPrior distribution used to draw tasksTraining pool of tasksContext (samples) consisting of $( \pmb { x } _ { 1 } ^ { \top } , y _ { 1 } , \dotsc , \pmb { x } _ { L } ^ { \top } , y _ { L } , \pmb { x } _ { L + 1 } ^ { \top } )$ Covariance of the in-context data $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ Label noise variance, $\eta _ { i } \sim \mathcal { N } ( 0 , \epsilon ^ { 2 } )$ Ridge parameter in KRRMean and covariance of $\mathcal { D } _ { \mathrm { t r } } \left( \mathrm { i } . \mathrm { e } . \ \hat { \kappa } _ { 1 } , \hat { \kappa } _ { 2 } \right)$ Order-l cumulant of $\mathcal { D } _ { \mathrm { n o v e l } }$ and $\mathcal { D } _ { \mathrm { t r } } ,$ respectivelyOrder-m raw moment tensor of $\mathcal { D } _ { \mathrm { t r } }$ </td></tr><tr><td rowspan=1 colspan=2> $L$ </td></tr><tr><td rowspan=1 colspan=2> $T$ </td></tr><tr><td rowspan=1 colspan=2> $n$ </td></tr><tr><td rowspan=2 colspan=2> $\beta$ </td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2> $\boldsymbol { B } _ { \mathrm { p r i o r } } = \mathcal { N } ( \mu _ { \beta } , \Sigma _ { \beta } )$  $B _ { \mathrm { t r } } = \{ \beta ^ { ( t ) } \} _ { t = 1 } ^ { T }$ </td></tr><tr><td rowspan=1 colspan=1>Btr = {β</td></tr><tr><td rowspan=1 colspan=2> $_ { z }$  $\Sigma _ { x }$  $\epsilon ^ { 2 }$  $\delta$  $\hat { \mu } _ { \beta } , \hat { \Sigma } _ { \beta }$  $\kappa _ { \ell } , \hat { \kappa } _ { \ell }$  $M _ { m }$ </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Whitened quantities (Given $\overline { { \Sigma _ { x } \neq c I _ { d } ) } }$ </td></tr><tr><td rowspan=1 colspan=2> $\overline { { { \tilde { x } } = \Sigma _ { x } ^ { - 1 / 2 } x } }$  $\widetilde { \beta } = \Sigma _ { x } ^ { 1 / 2 } \beta$  $\tilde { \kappa } _ { \ell }$ </td><td rowspan=1 colspan=1>Whitened in-context dataColored task; $\tilde { B } = \Sigma _ { x } ^ { 1 / 2 } B$ Cumulants of B</td></tr><tr><td rowspan=1 colspan=3>Kernel and eigenstructure</td></tr><tr><td rowspan=1 colspan=2> $c _ { \ell }$  $K _ { \ell }$  $\lambda _ { i } , \phi _ { i }$  $v _ { i }$  $\mathcal { L } _ { i }$  $\zeta$  ${ \mathcal { E } } _ { 0 }$  $C _ { F F } , C _ { F L } , C _ { L L } , \tilde { C } _ { \ell }$  $S _ { i j }$   $\tilde { v } _ { i }$ </td><td rowspan=1 colspan=1>Level coefficients of an inner-product kernel (Definition 1)Degree-l subkernel (Definition 1)Kernel eigenvalues and (orthonormal) eigenfunctionsEigencoefficient, $\begin{array} { r } { f ^ { \star } = \sum _ { i } v _ { i } \phi _ { i } } \end{array}$ Learnability of mode i; $\begin{array} { r } { \hat { f } = \sum _ { i } \mathcal { L } _ { i } v _ { i } \phi _ { i } } \end{array}$ Eigenframework constant solving Eq. (23)Overfitting coefficientLevel covariance blocks and projected covariance of Definition 4Eigenfunction overlap and test eigencoefficients on $\mathcal { D } _ { \mathrm { n o v e l } }$ </td></tr></table>

Table 1: Quantities appearing throughout the main document and appendices.

## B EXPERIMENTAL DETAILS

In Figures 1 and 2, we introduced $g _ { i }$ to describe polynomial functions learned by the kernel. The functions $g _ { i }$ are chosen to be representative of the polynomials of degree $2 i + 1$ with large eigencoefficients: that is, $g _ { 0 }$ was randomly selected as one of the d eigenfunctions ${ \pmb x } _ { L + 1 } , g _ { 1 }$ was chosen to be eigenfunction most closely related to $\begin{array} { r } { \phi _ { 3 } ( z ) \sim \left( \sum _ { i } y _ { i } \pmb { x } _ { i } ^ { \top } - \frac { L } { d } \hat { \pmb { \mu } } _ { \beta } ^ { \top } \right) \pmb { x } _ { L + 1 } , g _ { 2 } } \end{array}$ was one of the d eigenfunctions most closely related to (at leading order) ϕ<sub>5</sub> $\begin{array} { r } { { \sim } \sum _ { i j } y _ { i } y _ { j } { \pmb x } _ { i } ^ { \top } { \pmb x } _ { j } { \pmb x } _ { L + 1 } } \end{array}$ , and finally $g _ { 3 }$ was the eigenfunction most closely related to (at leading order) $\begin{array} { r } { \phi _ { 7 } \sim \sum _ { i j k } y _ { i } y _ { j } y _ { k } \pmb { x } _ { i } ^ { \top } \pmb { x } _ { j } \pmb { x } _ { k } ^ { \top } \pmb { x } _ { L + 1 } } \end{array}$

## B.1 EXACT EXPERIMENTAL SETUPS

We ran three main types of experiments: exact kernel experiments, exact random feature solvers, and random feature (RF) model solutions found via SGD. In the main text, we detail only the kernel experiments and RF model experiments, as those were the most fruitful. Below, we describe the setup for each figure in the main text, along with our rationale for using random features models in place of kernels.

The ICL Kernel. A kernel method solving the ICL linear regression task requires an enormous number of samples to get to either of the Bayes-optimal predictors. We can, however, reduce this by very carefully choosing a kernel function constructed to remove as many terms (those with eigencoefficient 0) as possible. We use the subkernel expansion framework (Definition 4) and the eigencoefficients found in Section F, to make the “ICL Kernel”, a Gaussian kernel with the linear components of ${ \pmb x } _ { L + 1 }$ extracted, and the even polynomials removed:

$$
K _ { I C L } ( z , z ^ { \prime } ) = \Big ( \exp \big ( - | c - c ^ { \prime } | ^ { 2 } / 2 \sigma ^ { 2 } \big ) + \exp \big ( - | c + c ^ { \prime } | ^ { 2 } / 2 \sigma ^ { 2 } \big ) \Big ) x _ { L + 1 } ^ { \top } x _ { L + 1 } ^ { \prime } ,\tag{11}
$$

where c is nothing but z with the unpaired datapoint ${ \pmb x } _ { L + 1 }$ removed.

This “ICL Kernel” that is often used in the experiments is not truly rotation invariant due to the ${ \pmb x } _ { L + 1 }$ term; nevertheless, it contains the same eigenfunctions with non-zero eigencoefficients as the Gaussian kernel (shown empirically to match the theory in Figures 8 and 9). Therefore, we readily use this “ICL Kernel” in experiments and make the informed guess that rotation-invariant kernels would behave identically, given sufficient data.

Kernel model experiments. In the right half of Figure 1, we train an ICL kernel with kernel width $\sigma = 3$ , a ridge parameter $\delta = 1 \mathrm { \check { 0 } } ^ { - 2 }$ , and each individual sample partially whitened such that $\begin{array} { r } { \sum _ { i } \pmb { x } _ { i } \pmb { x } _ { i } ^ { \top } = I _ { d } ^ { \top } / \bar { d } } \end{array}$ . We take $d = 5 , L = 7 ,$ and $\epsilon ^ { 2 } = 0 . 1$ . The in-context data is sampled ${ \pmb x } _ { i } \sim \dot { \mathcal { N } } ( 0 , I _ { d } / ( d L ) )$ ). We draw $T = 1 6$ tasks to get $B _ { \mathrm { t r } }$ , taken from a prior of $\mathcal { N } ( 0 , I _ { d } )$ . For the trained kernel models, we use 20 trials, keeping $B _ { \mathrm { t r } }$ fixed.

Direct eigenstructure checks. In Figure 8, we check that the subkernel expansion framework (Definition 4) accurately predicts the eigendecomposition of a trained kernel. The two spectra are straightforward to compare: we sort both in decreasing order and overlay them against mode index on log–log axes. Agreement means the curves coincide. The eigenbases are more difficult, most notably due to the ordering. The spectrum is degenerate over many different sections, and so assigning a cosine similarity between a theoretical eigenvector and empirical one mode-by-mode is at best left to chance. Therefore, we must bin the eigenvectors and compare their overall overlap.

We instead place bin edges in the gaps of the spectrum between degenerate sectors at the geometric mean $\sqrt { \lambda _ { ( m ) } \lambda _ { ( m + 1 ) } }$ of each consecutive pair whose ratio exceeds a hyperparameter threshold r. We additionally consolidate many bins at smaller eigenvalues (defined by another hyperparameter) into 2 bins such that our images are not dominated by many bins that are empirically noisy.

Let $d _ { \ell } ^ { \mathrm { ( e m p ) } }$ be the number of empirical modes in bin ℓ. Let $\Phi _ { \ell } ^ { ( \mathrm { e m p } ) } \in \mathbb { R } ^ { n \times d _ { \ell } ^ { ( \mathrm { e m p } ) } }$ collect the empirical eigenvectors of bin ℓ as columns, and let $\Phi _ { k }$ do the same for the predicted eigenfunctions of bin k. The pairwise subspace overlap is

$$
\mathrm { O } ( k , l ) = \frac { 1 } { d _ { l } ^ { ( \mathrm { e m p } ) } } \left. \Phi _ { k } ^ { \top } \Phi _ { l } ^ { ( \mathrm { e m p } ) } \right. _ { F } ^ { 2 } .\tag{12}
$$

The bottom plots ofFigure 8 shows $\mathrm { O } ( k , l )$ as a heatmap with eigenvalue increasing towards the top left, so that the horizontal axis runs in the same direction as mode index in the panel above.

We use $n = 2 5 6 0 0$ samples. Our setup consistently uses $d = 4 , \epsilon ^ { 2 } = 0 . 1 , \Sigma _ { x } = I _ { d } / ( d L )$ , and $\Sigma _ { \beta } = I _ { d }$ . All setups use $L = 6$ , outside of the Laplace kernel which uses $L = 9$ such that the inner-product kernel expansion remains valid (concentration of norm of $\| Z \| )$ , as is described in more detail in B.3 of (Karkada et al., 2026). Outside of the Laplace kernel for norm concentration reasons, all kernel:task-diversity pairs were chosen at random. In order of their appearance:

• T=1, Inner-product kernel uses level coefficients $c _ { 0 } = 1 , c _ { 1 } = 1 / 9 , c _ { 2 } = 1 / 8 1 , c _ { 3 } = 1 / 7 2 9 ,$ with bin edges of $r = 1 . 0 1$ , and the coarse graining of the tail beginning at $\lambda = \mathrm { \dot { 1 } 0 ^ { - 5 } }$ ending at $\lambda = 1 0 ^ { - 7 }$

• T=4, ReLU NTK uses bias and weight variances of 1, bin edges of $r = 1 . 0 1 8$ , and the coarse graining of the tail beginning at $\lambda = 7 \cdot 1 0 ^ { - 5 }$ , ending at $\bar { \lambda } = 6 \cdot 1 0 ^ { - 6 }$

$_ { \mathrm { T = 1 6 , } }$ Gaussian kernel uses kernel width of 3, with bin edges of $r = 1 . 0 0 8$ , and the coarse graining of the tail beginning at $\lambda = 1 0 ^ { - 4 }$ , ending at $\lambda = \overline { { 1 } } 0 ^ { - 6 }$

$\mathrm { T } = 2 5 6 .$ , Laplace kernel uses kernel width of $3 ,$ with bin edges of $r = 1 . 0 1 $ , and the coarse graining of the tail beginning at $\lambda = 8 \cdot 1 0 ^ { - 5 }$ , ending at $\lambda \overset { - } { = } 1 0 ^ { - 5 }$

Learning curves. In Figure 9, we train a Gaussian kernel with kernel width parameter $\sigma = \sqrt { 3 }$ and ridge $\delta = 0 . 1$ , on isotropic ICL data with $d = 5$ intrinsic dimensions, context length $L = 7 .$ and noise variance $\epsilon ^ { 2 } = 0 . 1 \bar { . }$ . Both in-context data x and the tasks B are drawn isotropically as $\pmb { x } \sim \mathcal { N } ( 0 , I _ { d } / ( L d ) )$ and $\beta \sim \mathcal { N } ( 0 , I _ { d } )$ . All runs are performed across 20 seeds with the max n set by the memory constraints of a Google Colab A-100 GPU: roughly 25,600 samples. We sweep over 4 (roughly logspaced) values of task diversity T: 1, 4, 16, 256 to show our theoretical eigensystem is correct regardless of task diversity and regardless of what distribution we evaluate our theory on.

## B.2 RANDOM FEATURE ANALYTICAL SOLVER

![](images/01eb59d2e8431f81e03f6829de80e045d25cd5f60b99d62d8126feaa892294d7.jpg)

![](images/fdb57f24e799931898927b6541a2276695f8fd7c88a45f523da8603f065783e8.jpg)  
Figure 6: Analytically solved random feature models outperform the memorizing oracle on novel tasks. We show test MSE on novel tasks $( \mathcal { D } _ { \mathrm { n o v e l } } )$ ) and on the training tasks $\beta \in { \mathcal { D } } _ { \mathrm { t } }$ <sub>r</sub> of a random feature model whose readout layer weight has been analytically solved, as a function of the number of training samples n. The limiting kernel function is the ICL kernel of Equation (11), set with kernel width $\sigma = 3$ . We use $T = 8 , \bar { L } = 4 , d = 4 , \epsilon ^ { 2 } = 0 . 0 5$ , and tasks drawn from an isotropic prior. Shading shows the standard deviation over 5 trials. Past n ≈ $4 \times 1 0 ^ { 3 }$ , the model’s error drops below that of the memorizing oracle (dMMSE). It remains well above the generalizing oracle (ridge) throughout.

As exact kernel solvers scaled poorly with the number of samples (one needs to invert an $n \times n$ matrix), random features models were additionally considered. Kernel models with suitable kernels can be understood as an infinite-width two-layer MLP with the first layer frozen (i.e. a fixed feature map followed by a linear projection) (Rahimi & Recht, 2007). Random features (RF) models serve as their finite-width counterpart, so long as the random features defining the frozen layer give the correct NNGP in the limit that the number of features approaches $\infty$ . RF models have a computational advantage: with $p$ random features, fitting the readout by ridge regression only requires solving a $p \times p$ linear system, reducing the training cost from $\mathcal { O } ( n ^ { 3 } )$ time and $\mathcal { O } ( n ^ { 2 } )$ memory to $\mathcal { O } ( n p ^ { 2 } + \overline { { p } } ^ { 3 } )$ time and $\mathcal { O } ( n p )$ memory.

So long as the model is not limited by the number of features (a feature-limited regime), extra performance can be gained from using RF models. To see the models are in a data-limited regime, one can note that they are mostly noise-dominated: the non-zero eigencoefficients are sparsely populating our spectrum, and worse than that, the eigencoefficients are not particularly contained within the large eigenvalue region (the largest eigencoefficients appear at the degree-3 part of the spectrum corresponding to $\begin{array} { r } { \sum _ { i } y _ { i } \pmb { x } _ { i } ^ { \top } \pmb { x } _ { L + 1 } } \end{array}$ , for which we have to fit the noisy degree 1 and 2 eigenfunctions). Therefore, we use RF models to substantiate our claims, directly solving for the optimal readout layer. This culminates in Figure 6, where, although more samples are able to be fit, it nevertheless remains difficult to see our theory’s predicted characteristics (though the drop and rise still are present). We therefore need even more samples to be fit, suggesting that we move from analytical expressions to ones obtained using SGD.

## B.3 RANDOM FEATURE EXPERIMENTS

As previously mentioned, the RF model is nothing but a special type of two-layer MLP, and training said MLP with online SGD would allow us to trade memory constraints for runtime.

In Figure 2, we train an RF model using SGD using $3 . 3 \cdot 1 0 ^ { 6 }$ gradient steps with a batch size of 1024. The dataset uses $d = 4 , L = 5 , \epsilon ^ { 2 } = 0 . 0 5$ . Our RF models use random features drawn such that they converge to the ICL kernel of Equation (11). The kernel width parameter used for the converging kernel was chosen as $\sigma = 1 . 1 \| z \|$ . We draw $T = 8$ tasks $\sqrt { ( 0 , I _ { d } ) }$ . The tasks are then whitened such that the covariance is exactly $\pmb { I } _ { d } .$ , however we keep the finite non-zero task mean. Our data is drawn $\pmb { x } \sim \mathcal { N } ( 0 , I _ { d } / d )$ . We normalize our labels $y$ by $d \sigma _ { x } ^ { 2 } + \epsilon ^ { 2 }$ . We run our experiments and average across 3 trials, shading the standard deviation. We used only 8192 features for our model. We use a learning rate $\begin{array} { r } { \eta = 0 . 7 5 \cdot \frac { 2 } { \lambda _ { 1 } } , \lambda _ { 1 } } \end{array}$ being a numerically estimated maximum eigenvalue of the Hessian found via power iteration.

In Figure 3, we train a RF model on $d = 3$ dimensional data with only $T = 4$ tasks. We use the same ICL kernel of Equation (11) as our limiting kernel for which the features are drawn. The kernel width parameter used for the converging kernel was chosen as $\sigma = 1 . 5 \| z \|$ . These tasks are initially drawn from a white Gaussian prior, but are manually whitened and centered: Cov $\boldsymbol { B } _ { \mathrm { t r } } = \boldsymbol { I } _ { d }$ , and the mean training task is set to 0. We take a large context length of $L = 9 6$ and $\epsilon ^ { 2 } = 0 . 0 5$ . We draw data $\pmb { x } \sim \hat { \mathcal { N } } ( 0 , I _ { d } / d )$ . We perform only 1 trial. We normalize our labels y by $d \sigma _ { x } ^ { 2 } + \epsilon ^ { 2 } = 1 . 0 5$ to keep the scale invariant to choice of $\Sigma _ { x }$ . We use 65, 536 features for our model. We train for $3 . 2 \cdot 1 0 ^ { \bar { 6 } }$ gradient steps with a batch size of 256. We use a learning rate $\begin{array} { r } { \eta = 0 . 8 \cdot \frac { 2 } { \lambda _ { 1 } } , \lambda _ { 1 } } \end{array}$ being a numerically estimated maximum eigenvalue of the Hessian found via power iteration.

In Figure $^ { 4 , }$ we train an RF model using SGD using a maximum of $4 \cdot 1 0 ^ { 6 }$ gradient steps with a batch size of 1024. We exit our training loop prematurely once we are within a tolerance of 0.01 of the predicted optimal polynomial degree 3 solution’s novel test error (to save on compute time; models would need to train on at least two orders of magnitude more samples before the degree 5 polynomials begin to be learned). To reduce random noise, we weight average the random feature model weights of at most 20, 000 steps. The dataset uses $d = 8 , L = 1 5 , \epsilon ^ { 2 } = 0 . 2 5$ . Our RF models use random features drawn such that they converge to the ICL kernel of Equation (11). The kernel width parameter used for the converging kernel was chosen as $\sigma = 1 . 5 \| \mathsf { z } \|$ Our data is drawn $\pmb { x } \sim \tilde { \mathcal { N } } ( 0 , I _ { d } / ( d \cdot 4 ) )$ , the numerical factor of 4 added such that $d \sigma _ { x } ^ { 2 } = \epsilon ^ { 2 }$ (which we found best portrayed our theory). We sweep the task diversity $T$ from $2 ^ { 0 }$ to $2 ^ { 1 0 }$ , training a new model per task diversity and run. We normalize our labels y by $\dot { d \sigma _ { x } ^ { 2 } } + \epsilon ^ { 2 } = 1 / 2$ to keep the scale invariant to choice of $\Sigma _ { x }$ . We run our experiments and average across 3 trials, shading the standard deviation. We used 16, 384 features for our model. We use a learning rate $\begin{array} { r } { \eta = 0 . 8 \cdot \frac { 2 } { \lambda _ { 1 } } , \lambda _ { 1 } } \end{array}$ being a numerically estimated maximum eigenvalue of the Hessian found via power iteration.

## B.4 TRANSFORMER EXPERIMENTAL DETAILS

We follow the setup of Raventos et al.´ (2023). We train a GPT-2-style decoder model online with 12 transformer blocks of width 256, 8 attention heads, MLPs of width 1024, GELU activations, no dropout, a causal attention mask, and learned positional embeddings initialized from $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ We train using Adam $( \mathrm { P y } ^ { \prime }$ Torch defaults $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ with a constant learning rate of $1 0 ^ { - 4 }$ and batch size 64. There is no weight decay, warm-up or gradient clipping. Models are trained for $3 \cdot 1 0 ^ { 5 }$ steps. The dataset uses $d = \bar { 8 , } L = 1 \bar { 6 , } \epsilon ^ { 2 } = 0 . 2 \dot { 5 }$ . The in-context data is drawn $\pmb { x } _ { i } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$ During training, $\beta$ is drawn uniformly from a fixed pool $B _ { \mathrm { t r } }$ of $T = 2 5 6$ tasks.

Every pool satisfies $\hat { \kappa } _ { 1 } = 0$ and $\hat { \kappa } _ { 2 } = I _ { d }$ exactly, via ZCA whitening. For the individual trials,

• Whiten: T i.i.d. draws from $\mathcal { N } ( 0 , \pmb { I } _ { d } )$ , centered and ZCA-whitened. The higher cumulants are left at their random sampling: $\lVert \hat { \kappa } _ { 3 } \rVert _ { F } \approx 1 . 6$ and $\lVert \hat { \kappa } _ { 4 } \rVert _ { F } \in [ 5 . 2 , 5 . 7 ]$ across seeds

• Symmetric pools: we optimize $T / 2 = 1 2 8$ vectors ${ \pmb w } _ { t }$ and take the pool to be the ${ \mathrm { Z C A } } -$ whitened set $\{ { \pmb w } _ { t } \} \cup \{ - { \pmb w } _ { t } \}$ , so every odd cumulant vanishes. Starting from $\pmb { w } _ { t } \sim \mathcal { N } ( 0 , \pmb { I } _ { d } )$ we run L-BFGS (strong-Wolfe line search, at most 2000 iterations) on

$$
\mathcal { L } ^ { \prime } = \frac { 1 0 0 0 } { 2 4 } \mathbb { E } _ { u } \left[ \left( k _ { 4 } ( u ) - \alpha k _ { 4 } ^ { ( 0 ) } ( u ) \right) ^ { 2 } \right] + \frac { 1 } { 7 2 0 } \mathbb { E } _ { u } \left[ k _ { 6 } ( u ) ^ { 2 } \right] ,
$$

where u ranges over 6000 fixed random unit vectors and $k _ { \ell } ( { \boldsymbol { \mathbf { \mathit { u } } } } )$ are the cumulants of the projections $\overset { \smile } { \pmb { u } } ^ { \top } \beta$ over the pool: $k _ { 4 } = m _ { 4 } - 3$ and $k _ { 6 } = m _ { 6 } - 1 5 m _ { 4 } - 1 0 m _ { 3 } ^ { 2 } + 3 0$ (here $m _ { 3 } = 0 ) . \ k _ { 4 } ^ { ( 0 ) } ( u )$ is the directional kurtosis of the initial pool. No kurtosis uses $\alpha = 0$ which reaches $\lVert \hat { \kappa } _ { 4 } \rVert _ { F } \leq 0 . 0 0 2$ . Partial kurtosis uses $\alpha = 0 . 3$ , giving $\| \hat { \kappa } _ { 4 } \| _ { F } = 2 . 3 9$ against 7.95 for its starting pool.

## C CUMULANTS REVIEW

Our results are phrased in terms of the cumulants of the task pool. Here, we review cumulants and the few facts the main text uses.

Definition. Write $\begin{array} { r } { \mathbb { E } { \boldsymbol { { \mathcal { B } } } _ { \mathrm { t r } } } \left[ \boldsymbol { g } ( \beta ) \right] = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \boldsymbol { g } ( \beta ^ { ( t ) } ) } \end{array}$ for the average over the uniform measure on the pool. Its cumulant generating function is

$$
\begin{array} { r } { \hat { K } ( \mathbf { t } ) = \log \mathbb { E } _ { B _ { \mathrm { t r } } } \left[ e ^ { \mathbf { t } ^ { \top } \beta } \right] , } \end{array}\tag{13}
$$

and the order-ℓ cumulant $\hat { \kappa } _ { \ell }$ is the symmetric tensor of its ℓ-th derivatives at the origin,

$$
\big ( \hat { \kappa } _ { \ell } \big ) _ { a _ { 1 } \cdots a _ { \ell } } = \frac { \partial ^ { \ell } \hat { K } } { \partial t _ { a _ { 1 } } \cdot \cdot \cdot \partial t _ { a _ { \ell } } } \bigg | _ { { \bf t } = { 0 } } .\tag{14}
$$

We write $M _ { \ell } = \mathbb { E } _ { B _ { \mathrm { t r } } } [ \beta ^ { \otimes \ell } ]$ for the raw moment tensors.

Relation to moments. With $\delta \beta = \beta - \hat { \mu } _ { \beta }$ the centered task, the first four cumulants are

$$
\hat { \kappa } _ { 1 } = \hat { \mu } _ { \beta } ,\tag{15}
$$

$$
\begin{array} { r } { \hat { \kappa } _ { 2 } = \mathbb { E } _ {  { \mathcal { B } } _ { \mathrm { t r } } } \bigl [ \delta \beta \delta \beta ^ { \top } \bigr ] = \hat { \Sigma } _ { \beta } , } \end{array}\tag{16}
$$

$$
( { \hat { \kappa } } _ { 3 } ) _ { a b c } = \mathbb { E } _ { { B _ { \mathrm { t r } } } } [ \delta \beta _ { a } \delta \beta _ { b } \delta \beta _ { c } ] ,\tag{17}
$$

$$
\begin{array} { r } { \left( \hat { \kappa } _ { 4 } \right) _ { a b c d } = \mathbb { E } _ { \mathcal { B } _ { \mathrm { t r } } } [ \delta \beta _ { a } \delta \beta _ { b } \delta \beta _ { c } \delta \beta _ { d } ] - \left( \hat { \Sigma } _ { a b } \hat { \Sigma } _ { c d } + \hat { \Sigma } _ { a c } \hat { \Sigma } _ { b d } + \hat { \Sigma } _ { a d } \hat { \Sigma } _ { b c } \right) . } \end{array}\tag{18}
$$

In general, $\hat { \kappa } _ { \ell }$ is the order-ℓ central moment with all relevant lower-order information subtracted—it is the information in the order-ℓ moment not already carried by lower orders.

Key properties used throughout our findings. Cumulants beyond the first are invariant under shifts, and all cumulants are multilinear under linear maps: if ${ \tilde { \beta } } = A \beta$ , then

$$
\left( \tilde { \kappa } _ { \ell } \right) _ { a _ { 1 } \cdots a _ { \ell } } = \sum _ { b _ { 1 } , \ldots , b _ { \ell } } A _ { a _ { 1 } b _ { 1 } } \cdot \cdot \cdot A _ { a _ { \ell } b _ { \ell } } \left( \hat { \kappa } _ { \ell } \right) _ { b _ { 1 } \cdots b _ { \ell } } .\tag{19}
$$

Taking $A = \Sigma _ { x } ^ { 1 / 2 }$ gives the colored cumulants $\tilde { \kappa } _ { \ell }$ of Section F.

A Gaussian has all its cumulants beyond order two vanish. The prior $\mathcal { N } ( \mu _ { \beta } , \Sigma _ { \beta } )$ therefore carries only a mean and a covariance, but a finite pool of $T$ draws from it does not. Each $\hat { \kappa } _ { \ell }$ is a sample statistic that converges to the prior’s value as $T \to \infty$ with entrywise fluctuations of order $T ^ { - 1 / 2 }$ , so $\hat { \kappa } _ { 1 }  \mu _ { \beta } , \hat { \kappa } _ { 2 }  \Sigma _ { \beta }$ , and $\hat { \kappa } _ { \ell \geq 3 }  0$ . This rate is entrywise at fixed d: tensor norms carry additional factors of d, and when d is comparable to $T$ the spectrum of $\hat { \Sigma } _ { \beta }$ follows the Marchenko–Pastur law rather than concentrating at the prior’s eigenvalues (Section 7.3).

Several eigencoefficients depend on cumulants only through partial traces. We write $\mathrm { T r } _ { k }$ for the contraction of k index pairs,

$$
( \mathrm { T r } _ { k } \hat { \kappa } _ { \ell } ) _ { a _ { 2 k + 1 } \cdots a _ { \ell } } = \sum _ { b _ { 1 } , \ldots , b _ { k } } ( \hat { \kappa } _ { \ell } ) _ { b _ { 1 } b _ { 1 } \cdots b _ { k } b _ { k } a _ { 2 k + 1 } \cdots a _ { \ell } } ,\tag{20}
$$

which by symmetry of $\hat { \kappa } _ { \ell }$ does not depend on which pairs are contracted. For example, $\mathrm { T r } _ { 1 } \hat { \kappa } _ { 3 }$ with components $\sum _ { } ^ { } ( \hat { \kappa } _ { 3 } ) _ { a a b }$ , enters the degree-3 coefficients, and $\mathrm { T r } _ { 2 } \hat { \kappa } _ { 5 }$ , with components $\sum _ { a , b } \bigl ( \hat { \kappa } _ { 5 } \bigr ) \bar { _ { a a b b c } } .$ , enters the degree-5 coefficients. We call a pool traceless at order $\ell \ \mathrm { i f } \ \mathrm { T r } _ { 1 } \hat { \kappa } _ { \ell } = 0$ Contracted cumulants can therefore appear at lower polynomial degree than the full tensor, which first appears at degree $2 \omega - 1$ for $\hat { \kappa } _ { \omega }$ (Theorem 1).

## D REVIEW OF THE KRR EIGENFRAMEWORK FOR PREDICTING TESTPERFORMANCE FROM TASK EIGENSTRUCTURE

The central piece of existing theory which we use in this paper is a set of equations which give the average-case test MSE of kernel ridge regression (KRR) in terms of the task eigenstructure. In this appendix, we will review this KRR eigenframework.

This eigenframework has been derived many times by different means in both the statistical physics community and the classical statistics community (which usually phrases the result as applying to linear ridge regression). References studying KRR include Sollich (2001); Bordelon et al. (2020); Jacot et al. (2020); Simon et al. (2021); Loureiro et al. (2021); Wei et al. (2022a); references studying linear ridge regression include Caponnetto & De Vito (2007); Dobriban & Wager (2018); Wu & Xu (2020); Hastie et al. (2022); Richards et al. (2021); Cheng & Montanari (2022). The result is essentially the same in all cases. Here we will adopt the terminology and notation of Simon et al. (2021).

Recall that we are studying KRR with a kernel function K with data sampled $\begin{array} { r } { \mathbf { x } _ { i } \sim \mu , } \end{array}$ , targets generated as $y _ { i } = f _ { * } ( \pmb { x } _ { i } ) + \eta _ { \ast }$ , and noise $\eta \sim \mathcal { N } ( 0 , \epsilon ^ { 2 } )$ with variance $\epsilon ^ { 2 } \geq 0 .$ . The kernel admits an eigendecomposition $\begin{array} { r } { K ( { \pmb x } , { \pmb x } ^ { \prime } ) = \sum _ { i } \lambda _ { i } \phi _ { i } ( { \pmb x } ) \phi _ { i } ( { \pmb x } ^ { \prime } ) } \end{array}$ with orthonormal eigenfunctions $\left( \phi _ { i } \right)$ . Let us decompose the target function in the eigenbasis as $\begin{array} { r } { f _ { * } ( { \pmb x } ) = \sum _ { i } v _ { i } \phi _ { i } ( { \pmb x } ) } \end{array}$ . Suppose we run KRR with n samples and a ridge parameter $\delta \geq 0 .$ , and we compute the population (i.e. test) and train MSEs as

$$
\begin{array} { r } { \mathtt { M S E } _ { \mathrm { t e } , \mathrm { I D } } = \underset { \pmb { x } \sim \mathcal { D } } { \mathbb { E } } \Big [ ( f _ { * } ( \pmb { x } ) - \hat { f } ( \pmb { x } ) ) ^ { 2 } \Big ] + \epsilon ^ { 2 } , } \end{array}\tag{21}
$$

$$
\mathrm { { M S E } _ { t r } } = \frac { 1 } { n } \sum _ { i } ( y _ { i } - \hat { f } ( { \pmb x } _ { i } ) ) ^ { 2 } .\tag{22}
$$

## D.1 STATEMENT OF THE EIGENFRAMEWORK

We are now ready to state the eigenframework. Let $\zeta \geq 0$ be the unique nonnegative solution to

$$
\sum _ { i } { \frac { \lambda _ { i } } { \lambda _ { i } + \zeta } } + { \frac { \delta } { \zeta } } = n .\tag{23}
$$

Then test risk is given approximately by

$$
\begin{array} { r } { \mathtt { M S E _ { t e } } \approx \mathcal E _ { \mathrm { t e } } : = \mathcal E _ { 0 } \mathrm { B } , } \end{array}\tag{24}
$$

where the overfitting coefficient ${ \mathcal { E } } _ { 0 }$ is given by

$$
\ { \mathcal { E } } _ { 0 } : = { \frac { n } { n - \sum _ { i } \left( { \frac { \lambda _ { i } } { \lambda _ { i } + \zeta } } \right) ^ { 2 } } }\tag{25}
$$

and the bias is given by

$$
\mathrm { B } = \sum _ { i } \left( { \frac { \zeta } { \lambda _ { i } + \zeta } } \right) ^ { 2 } v _ { i } ^ { 2 } + \epsilon ^ { 2 } .\tag{26}
$$

Train risk is given by

$$
\mathtt { M S E } _ { \mathrm { t r } } \approx \mathcal { E } _ { \mathrm { t r } } \equiv \frac { \delta ^ { 2 } } { n ^ { 2 } \zeta ^ { 2 } } \mathcal { E } _ { \mathrm { t e } } ,\tag{27}
$$

What is meant by the $\ " \approx \ "$ in Equations (24) and (27)? This result only becomes exact in certain stringent cases; it is formally derived under an assumption that the eigenfunctions are independent Gaussian (or sub-Gaussian) variables when x is sampled from $\mu ,$ and it is exact only in an asymptotic limit in which n and the number of eigenmodes in a given eigenvalue range (or the number of duplicate copies of any given eigenmode) both grow large at a proportional rate (Hastie et al., 2022; Bach, 2023). These conditions emphatically do not apply to any realistic instance of KRR. Nonetheless, numerical experiments find that Equations (24) and (27) hold with small error even at modest n (Canatar et al., 2021; Simon et al., 2021; Wei et al., 2022a): though derived in an idealized setting and exact only in a limit, this eigenframework holds reliably in practical cases.

In this paper, we use this eigenframework as a tool to map predictions of task eigenstructure to predictions of learning curves. Since we are here using it in settings very similar to those tested by previous works (Bordelon et al., 2020; Jacot et al., 2020; Simon et al., 2021; Wei et al., 2022a), we expect it to work well. Its use introduces some small error (as it is not perfect at finite n), but this is usually not the dominant source of error.

We have found numerically that this framework fails on ICL problems with a single intrinsic dimension, d = 1, but works well otherwise. This failure mode is due to the Gaussian universality condition breaking; see Figure 7.

![](images/0721efb100ecb5404b977f10582c36ae32ab2151b9795531b31ac3d0fee93583.jpg)

![](images/180342ec4ece9df81394b009e9e9e780a72fa5971a171978b09a19392bf9480b.jpg)  
Figure 7: One-dimensional ICL data breaks Gaussian universality. We compare the eigenframework’s predicted test MSE (line) against kernel regression trained on true ICL contexts (circles) and on a Gaussian surrogate with matched moments (diamonds), for $T = 4 , d = 1 , L = 8$ . We use a Gaussian kernel with kernel width $\sigma _ { \mathrm { k e r } } = \sqrt { 3 }$ and in-context data with covariance $\Sigma _ { x } \sim I _ { d } / ( d L )$ . The left panel is ridgeless $( \delta = 0 )$ and the right uses a small ridge $( \delta = 5 \times 1 0 ^ { - 3 } )$ . The Gaussian surrogate tracks the theory at every n. The true contexts agree away from the interpolation peak but exceed the prediction near it, most visibly at small ridge. Both datasets share the same theoretical curve, so the mismatch comes from the non-Gaussianity of one-dimensional ICL data rather than from the predicted eigenstructure. We observe no such failure for $d > 1$

## D.2 THE OUT-OF-DISTRIBUTION EIGENFRAMEWORK

While the eigenframework as presented in Sollich (2001); Bordelon et al. (2020); Jacot et al. (2020); Simon et al. (2021); Loureiro et al. (2021); Wei et al. (2022a) specify only the learning dynamics when the distribution of the trainset is equivalent to that of the testset, our techniques require specifying the test error on a novel “ICL” distribution. Luckily, an out-of-distribution version of the eigenframework appears in Canatar et al. (2022), which we can use as a baseline, choosing to adapt the same notation as Simon et al. (2021). The main difference between the in-distribution and an out-of-distribution test set is that our eigenfunctions need no longer be eigen (orthonormal) on our new distribution. Therefore, the bulk of our work will be in describing this non-orthonormality. We begin by defining two useful quantities going forward, the eigenfunction overlap and the test eigencoefficients:

$$
S _ { i j } = \langle \phi _ { i } ( Z ) \phi _ { j } ( Z ) \rangle _ { \mathcal { D } _ { \mathrm { { 0 o D } } } }\tag{28}
$$

$$
\tilde { v } _ { i } = S _ { i j } ^ { - 1 } \left. y ^ { \star } \phi _ { j } ( Z ) \right. _ { \mathcal { D } _ { \mathrm { { O O D } } } }\tag{29}
$$

Writing $y ^ { \star }$ as a sum of a resolvable and unresolvable portion,

$$
y ^ { \star } ( Z ) = \sum _ { i } \tilde { v } _ { i } \phi _ { i } ( Z ) + f _ { \perp } ,\tag{30}
$$

we can expand $\mathrm { M S E } _ { \mathrm { t e . } }$ <sub>, OOD</sub> as,

$$
\begin{array} { r } { \mathtt { M S E } _ { \mathrm { t e } , \mathrm { 0 O D } } = \underset { \pmb { x } \sim \mathcal { D } _ { 0 0 0 } } { \mathbb { E } } \left[ ( f _ { * } ( \pmb { x } ) - \hat { f } ( \pmb { x } ) ) ^ { 2 } \right] + \epsilon ^ { 2 } } \end{array}\tag{31}
$$

$$
= \underset { \substack { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { 0 0 D } } } } { \mathbb { E } } \left[ ( \hat { \boldsymbol { v } } - \tilde { \boldsymbol { v } } ) ^ { \top } \boldsymbol { S } ( \hat { \boldsymbol { v } } - \tilde { \boldsymbol { v } } ) \right] + | | \boldsymbol { f } _ { \perp } | | _ { \mathrm { 0 0 D } } ^ { 2 } + \epsilon ^ { 2 }\tag{32}
$$

$$
= ( \mathbb { E } [ \hat { v } - \tilde { v } ] ) ^ { \top } S ( \mathbb { E } [ \hat { v } - \tilde { v } ] ) + \operatorname { t r } [ S \cdot C o v ( \hat { v } ) ] + | | f _ { \perp } | | _ { \mathrm { O O D } } ^ { 2 } + \epsilon ^ { 2 }\tag{33}
$$

$$
= \Delta ^ { \top } S \Delta + \frac { \mathcal { E } _ { 0 } } { n } \sum _ { i } S _ { i i } \mathcal { L } _ { i } ^ { 2 } \left( \sum _ { j } ( 1 - \mathcal { L } _ { j } ) ^ { 2 } v _ { j } ^ { 2 } + \epsilon ^ { 2 } \right) + | | f _ { \perp } | | _ { \mathrm { O O D } } ^ { 2 } + \epsilon ^ { 2 }\tag{34}
$$

where $\Delta = \tilde { v } - \mathcal { L } v$ . The first part of the equation $( \Delta ^ { \intercal } S \Delta )$ can be understood to be the “outof-distribution bias” term, the second part as the “out-of-distribution variance” term, and the last $( | | f _ { \perp } | | _ { \mathrm { O O D } } ^ { 2 } + \epsilon ^ { 2 } )$ as an unresolvable noise.

![](images/5aeb0f666689859850f89a68a24632d0516641b9e494fdc7021fb60be13ea22b.jpg)

![](images/69ee975f53e0fab1c81fa890d9cc6b390fdec26d9b07bfc330f7869c0ef9c221.jpg)

![](images/d279f29cbf03d056bf8c3580d6611b0465f97b25ad28f262e41ffe2e313dead2.jpg)

![](images/ad954548b76da20899a88ca98f2d4b1bc884aea50b60653579b8fd20ae6c265f.jpg)

![](images/d26e1e7a825e122d0abe8e9703a18fe8325dfe1edc47679610da9453855252dd.jpg)

![](images/f979c5689f5f595eb2e39c40174685867b07536fc1216984a8a023737fff1718.jpg)

![](images/19e85033788969cf5c4b6faf950b8082e2e33b5bca1acffbc1217c73f2ab066b.jpg)

![](images/fdc8bb5fd0cce0a2b0218a6c77958780cdcf07dde1ed43846618231a7349daea.jpg)  
Figure 8: The subkernel expansion framework (Definition 4) accurately predicts the eigenstructure of in-context learning data for various rotation-invariant kernels. Across various rotationinvariant kernels with varying task diversity (columns), we compute the empirical kernel eigensystem $\{ ( \lambda _ { i } ^ { ( \mathrm { e m p } ) } , \phi _ { i } ^ { ( \mathrm { e m p } ) } ) \}$ and compare to the theoretical eigensystem $\big \{ \big ( \lambda _ { i } ^ { \mathrm { ( t h ) } } , \phi _ { i } ^ { \mathrm { ( t h ) } } \big ) \big \}$ } obtained from Definition 4, indexed in order of decreasing $\lambda _ { i } ^ { ( \mathrm { t h } ) }$ . In the top row, we overlay the eigenvalue spectrum between $\lambda _ { i } ^ { ( \mathrm { e m p } ) }$ and $\lambda _ { i } ^ { ( \mathrm { t h } ) }$ . In the bottom row, we bin both the predicted and empirical eigenfunctions into shared spectral bins and visualize the pairwise subspace overlap (Equation (12)). Concentration along the diagonal indicates theory-experiment match. See Section B for further explanation and experimental details.

## E DETAILS ON THE KERNEL EIGENSTRUCTURE

The kernel’s prediction is governed by its (eigenvalue, eigenfunction) pairs $( \lambda _ { i } , \phi _ { i } )$ with respect to the training distribution ${ \mathcal { D } } _ { \mathrm { t r } }$

$$
\begin{array} { r } { \mathbb { E } _ { z ^ { \prime } \sim \mathcal { D } } \left[ K ( z , z ^ { \prime } ) \phi _ { i } ( z ^ { \prime } ) \right] = \lambda _ { i } \phi _ { i } ( z ) , \quad \quad \mathbb { E } _ { z \sim \mathcal { D } } \left[ \phi _ { i } \phi _ { j } \right] = \delta _ { i j } . } \end{array}\tag{35}
$$

This is analytically intractable for the ICL data and kernels that we analyze, so we need instead to find the eigendecomposition another way. Here, we walk through how to arrive at the subkernel expansion framework, our method of approximating the eigensystem, then give its full definition. The ICL regression task does not have closed-form eigenfunctions in full generality, but we can nonetheless approximate them well.

## E.1 THE SUBKERNEL EXPANSION FRAMEWORK

We first give a brief primer on what the subkernel expansion framework is at a high level before giving the technical definition, much in the same way integrals are first introduced using Riemann sums.

Because $\| z \|$ concentrates for ICL contexts, the rotation-invariant kernels are well approximated by inner-product kernels (Definition 1), which depend on their inputs only through $z ^ { \top } z ^ { \bar { \prime } }$

Definition 1 (Inner-product kernel and its subkernels). A kernel is an inner-product kernel if $\begin{array} { r } { K ( z , z ^ { \prime } ) = \sum _ { \ell > 0 } \frac { c _ { \ell } ^ { * } } { \ell ! } ( z ^ { \top } z ^ { \prime } ) ^ { \ell } } \end{array}$ with “level coefficients $v _ { \ell } \geq 0$ , which guarantees K is PSD. Its degree-ℓ subkernel is the single term $\begin{array} { r } { K _ { \ell } ( z , z ^ { \prime } ) = \frac { c _ { \ell } } { \ell ! } ( z ^ { \top } z ^ { \prime } ) ^ { \ell } } \end{array}$

We begin with an intuitive explanation for the eigenfunctions. We treat each subkernel of Definition 1 separately. The degree-1 subkernel is nothing but a linear kernel $( i . e .$ . performing linear regression), so its eigenfunctions are the principal components of the raw data. Since $( z ^ { \top } z ^ { \prime } ) ^ { \ell } = ( \breve { z } ) _ { \ell } ^ { \top } ( z ^ { \prime } ) _ { \ell } ^ { \top }$ where $( z ) _ { \ell }$ collects all degree-ℓ polynomials of $z ,$ the degree-ℓ subkernel is likewise a linear kernel on the degree-ℓ polynomials of the data, and we might expect its eigenfunctions to be their principal components. This fails because polynomials overlap across degrees $( e . g . \ x ^ { 2 }$ with the constant). To handle this, we impose the assumption that level coefficients decay quickly, $c _ { \ell + 1 } \mathbb { E } \| z \| ^ { 2 } \ll c _ { \ell }$ . Each shared component is then claimed by the lowest degree that contains it. The degree-ℓ eigenfunctions are therefore the principal components of the degree-ℓ polynomials after all lower degrees are projected out, with eigenvalues as $c _ { \ell } / \ell !$ times the variances of those components.

The terms of the subkernel expansion with ICL data In the walkthrough, we labeled the terms simply as “polynomials,” although it’s helpful to discuss which polynomials we are talking about, and which will be used for a model’s computation. We find that only a select subset will have non-0 coefficient when expanding functions in the kernel’s eigenbasis, which we label as subkernel expansion ICL polynomials. These serve as good approximations to the eigenfunctions, with the upside that they’re still interpretable.

Definition 2 (Subkernel expansion ICL polynomials). The degree $p$ Subkernel expansion ICL polynomials, $\Psi _ { p } ( z )$ , are a set of functions $\psi _ { p }$ that can be written as a tensor product of p components of the context z in the following manner:

$$
\psi _ { p } ( z ) \equiv \left( \bigotimes _ { i = 1 } ^ { m } \sum _ { \ell = 1 } ^ { L } y _ { \ell } ^ { a _ { i } } \pmb { x } _ { \ell } ^ { \otimes b _ { i } } \right) \otimes \pmb { x } _ { L + 1 }\tag{36}
$$

where $\begin{array} { r } { p = 1 + \sum _ { i = 1 } ^ { m } ( a _ { i } + b _ { i } ) . \Psi _ { p } } \end{array}$ contains multiple different polynomial functions, which we enumerate with $\psi _ { p } , \psi _ { p } ^ { \prime } , \psi _ { p } ^ { ( 2 ) } , . . . , \psi _ { p } ^ { ( k ) }$ . For instance, the degree 3 Subkernel expansion ICL polynomials contain both the functions $\begin{array} { r } { \psi = \sum _ { i } y _ { i } \pmb { x } _ { i } \otimes \pmb { x } _ { L + 1 } } \end{array}$ of size $d ^ { 2 }$ and $\begin{array} { r } { \psi _ { 3 } ^ { \prime } = \sum _ { i } y _ { i } ^ { 2 } { \pmb x } _ { L + 1 } } \end{array}$ of size d.

Although it would be mathematically simple to work with Subkernel expansion ICL polynomials, we cannot, as they couple across different orders. For instance, $\Psi _ { 5 }$ overlaps nontrivially with $\Psi _ { 3 }$ $\begin{array} { r } { { \tt e . g . } \psi _ { 5 } = \sum _ { i } y _ { i } ^ { 3 } x _ { i } { \bf x } _ { L + 1 } } \end{array}$ and $\begin{array} { r } { \psi _ { 3 } = \sum _ { i } y _ { i } \pmb { x } _ { i } \pmb { x } _ { L + 1 } } \end{array}$ are not orthogonal as $\dot { \mathbb { E } } _ { \mathcal { D } } [ y _ { i } ^ { 2 } ] > 0$ . To account for this overlap, we invoke that our kernels will learn the optimal low-degree function first $( i . e .$ . the lower-degree functions will have larger eigenvalues), and therefore project out the lower-degree information from the higher degree polynomial. We label these processed terms as the “Subkernel expansion ICL perpendicular polynomials.”

Definition 3 (Subkernel expansion ICL perpendicular polynomials). A degree p set of subkernel expansion ICL perpendicular polynomials, $\Psi _ { p } ^ { \perp } ( z )$ , is constructed as a corresponding subkernel expansion ICL polynomial (Definition 2) with all overlap between lower-degree subkernel expansion ICL polynomials projected out:

$$
\Psi _ { p } ^ { \perp } ( z ) \equiv \Psi _ { p } ( z ) - \mathbb { E } _ { \mathcal { D } } \left[ \Psi _ { p } \Psi _ { < p } ^ { \top } \right] \mathbb { E } _ { \mathcal { D } } \left[ \Psi _ { < p } \Psi _ { < p } ^ { \top } \right] ^ { + } \Psi _ { < p } ( z )\tag{37}
$$

where each function of $\Psi _ { p }$ is subtracted against every subkernel expansion ICL polynomial of a lower degree, and $( \cdot ) ^ { + }$ denotes a pseudoinverse. To note, we do not make any subtractions within a degree.

The subkernel expansion ICL perpendicular polynomials are now orthogonal across degrees, but not within a given degree: the polynomials share within-context fluctuations and become correlated through the task distribution’s statistics. For example, the overlap between $\begin{array} { r } { \psi _ { 3 } ^ { \perp } = \sum _ { i } y _ { i } \pmb { x } _ { i , 1 } \pmb { x } _ { L + 1 , 1 } \approx } \end{array}$ $\begin{array} { r } { \sum _ { i } \beta _ { 1 } ^ { \top } I _ { d } \pmb { x } _ { L + 1 , 1 } / d } \end{array}$ and $\begin{array} { r } { \psi _ { 3 } ^ { \bot } = \sum _ { i } y _ { i } \pmb { x } _ { i , 2 } \pmb { x } _ { L + 1 , 1 } \approx \sum _ { i } \beta _ { 2 } ^ { \top } I _ { d } \pmb { x } _ { L + 1 , 1 } / d } \end{array}$ is non-zero unless the covari ance of the training tasks is isotropic $( \hat { \Sigma } _ { \beta } \propto I _ { d } )$ . To account for this, we orthonormalize each degree as a single block using its Gram matrix to obtain the kernel’s predominately useful eigenfunctions, the ones with non-0 eigencoefficient.

The subkernel expansion framework, in full detail Here, we provide the subkernel expansion framework.

```latex
Definition 4 (Subkernel expansion). The subkernel expansion of a kernel K is obtained using the
following procedure:
1. Expand the chosen kernel function as an inner-product kernel (Definition 1) with level
coefficients $c _ { \ell } .$
2. Let $( z ) _ { \ell }$ be the vector of all entries of $z ^ { \otimes \ell } ,$ , i.e. all ordered products of ℓ coordinates of $z ,$
so that $( z ^ { \top } z ^ { \prime } ) ^ { \ell } = ( z ) _ { \ell } ^ { \top } ( z ^ { \prime } ) _ { \ell }$ exactly. Let $( z ) _ { < \ell }$ stack $( z ) _ { 0 } , \ldots , ( z ) _ { \ell - 1 }$ . For convenience,
denote all polynomials at the current level ℓ with the marker ${ } ^ { \mathrm { 6 6 } } \mathrm { F } _ { \mathrm { 9 } } ^ { \mathrm { 9 } }$ and polynomials at all
levels below the current level with the marker $^ { 6 6 } \mathrm { L } .$
3. Construct the second-moment matrix of the degree-ℓ features, $C _ { F F } \equiv \mathbb { E } \big [ ( \pmb { z } ) _ { \ell } ( \pmb { z } ) _ { \ell } ^ { \top } \big ]$
4. Similarly, construct $C _ { F L } \equiv \mathbb { E } \big [ ( \boldsymbol { z } ) _ { \ell } ( \boldsymbol { z } ) _ { < \ell } ^ { \top } \big ]$ between the degree-ℓ features and all lower
degree ones, and $C _ { L L } \equiv \mathbb { E } \big [ ( \boldsymbol { z } ) _ { < \ell } ( \boldsymbol { z } ) _ { < \ell } ^ { \top } \big ]$
5. Project out all lower-degree features using the Schur complement, ${ { \tilde { C } } _ { \ell } } \ = \ C _ { F F } \ -$
$\dot { C } _ { F L } \dot { C } _ { L L } ^ { + } \dot { C } _ { F L } ^ { \top }$ , where $^ +$ is the pseudoinverse $( C _ { L L }$ is singular because reordered products
repeat).
6. The subkernel $K _ { \ell } { \mathrm { : } } { \mathrm { s } }$ eigendecomposition is equivalent to a diagonalization of $\textstyle { \frac { c _ { \ell } } { \ell ! } } { \tilde { C } } _ { \ell }$
7. The eigendecomposition of the original kernel K is the set of eigendecompositions of
each subkernel:
$\operatorname { E i g e n d e c o m p } ( K ) \approx \operatorname { E i g e n d e c o m p } ( K _ { 0 } ) + \operatorname { E i g e n d e c o m p } ( K _ { 1 } ) + \dots$
```

Note on Hermite polynomials For Gaussian data, the level-ℓ eigenfunctions are Hermite polynomials of degree ℓ in the data’s principal components (Karkada et al., 2026). ICL data is exactly Gaussian only for a single task $( T = \dot { 1 } )$ , where each label is a linear function of its input plus Gaussian noise; there, the subkernel expansion reduces to their Hermite structure. For $T > \bar { 1 }$ , mixing over tasks makes the context non-Gaussian and correlates distinct positions through the shared task (the matrix C of Section E.3), so the eigenfunctions depart from Hermite polynomials. The exception is the query, which is Gaussian and independent of the context, so its factor remains Hermite (Proposition 1).

## E.2 POTENTIAL PROOF SKETCH FOR THE SUBKERNEL EXPANSION FRAMEWORK

We have not proven the subkernel expansion framework in general; it is validated experimentally in Figure 8. Nonetheless, we sketch a possible proof below.

Proof sketch main idea. Let $\mu$ be any data measure with $\| z \| ^ { 2 } \ \leq \ R ^ { 2 }$ , and let $K ( z , z ^ { \prime } ) =$ $\begin{array} { r } { \sum _ { \ell } \frac { c _ { \ell } } { \ell ! } ( z ^ { \top } z ^ { \prime } ) ^ { \ell } } \end{array}$ with $c _ { \ell + 1 } R ^ { 2 } \leq \varepsilon c _ { \ell }$ . For every fixed level ℓ, the true kernel eigenvalues match the level-ℓ subkernel expansion eigenvalues (with the same indices) up to an additive $O ( \varepsilon c _ { \ell } R ^ { 2 \ell } / \ell ! )$ . This is a relative $1 + O ( \varepsilon )$ for eigenvalues of that order. The true eigenspaces match the framework’s eigenspaces up to an overlap error of $O ( \varepsilon )$ divided by the relative spectral gap. For the Gaussian kernel, $c _ { \ell + 1 } / { \dot { c _ { \ell } } } = \sigma ^ { - 2 } , \ : \mathrm { s o } \ : \varepsilon \approx R ^ { 2 } / \sigma ^ { 2 }$

The argument bounds each eigenvalue from above and below; the eigenspace overlap then follows by standard perturbation theory.

Upper bound. Apply Courant–Fischer on the functions orthogonal to $\mathcal { P } _ { < \ell }$ and to the top $j - 1$ level-ℓ framework eigenfunctions. There, the lower subkernels vanish, level ℓ contributes exactly the framework’s quadratic form, and the higher levels contribute at most $\scriptstyle \sum _ { m > \ell } c _ { m } R ^ { 2 m } / m ! \ =$ $O ( \varepsilon c _ { \ell } R ^ { 2 \ell } / \ell ! )$

Lower bound. Assume the lower-level eigenvalues are of order $c _ { \ell - 1 } R ^ { 2 ( \ell - 1 ) } / ( \ell - 1 ) !$ , which by hypothesis is at least of order $c _ { \ell } R ^ { 2 \ell } / ( \varepsilon \ell ! )$ . The coupling between $\mathcal { P } _ { < \ell }$ and the level-ℓ functions comes through $K _ { \ell }$ and the higher levels, and is of order $c _ { \ell } R ^ { 2 \ell } / \ell !$ . Eliminating the lower block therefore shifts the eigenvalue by at most $O ( \varepsilon c _ { \ell } R ^ { 2 \ell } / \ell ! )$

## E.3 USING THE SUBKERNEL EXPANSION FRAMEWORK WITH ICL DATA

Above, we introduced the subkernel expansion. Now, we apply it to the ICL linear regression setting to find the eigensystem beyond heuristics.

We begin with $d = 1$ data for readability, so each $x _ { i }$ is a scalar with variance $\sigma _ { x } ^ { 2 }$ , and we write $\begin{array} { r } { m _ { k } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \beta _ { t } ^ { k } } \end{array}$ for the raw moments of the pool.

Linear subkernel. The level-1 subkernel $K _ { 1 } = c _ { 1 } z ^ { \top } z ^ { \prime }$ performs linear regression. Its eigenfunctions are the principal components of the data’s second-moment matrix, with eigenvalues $c _ { 1 }$ times those of the matrix. Every coordinate of z has mean zero, so no projection against the constant is needed, and the second-moment matrix is the covariance. Distinct positions are uncorrelated: $\operatorname { C o v } ( x _ { i } , x _ { j } ) = \operatorname { C o v } ( x _ { i } , y _ { j } ) = \operatorname { C o v } ( y _ { i } , y _ { j } ) = 0$ for $i \neq j ,$ , and the query is uncorrelated with every in-context coordinate. The covariance is therefore block diagonal. It consists of an isolated query block, with eigenfunction $x _ { L + 1 }$ and eigenvalue $c _ { 1 } \sigma _ { x } ^ { 2 }$ , and L identical $2 \times 2$ blocks, one per position:

$$
\mathrm { C o v } \left( { x _ { i } } \right) = \left( { \mathbb E } _ { \mathcal D } [ x _ { i } ^ { 2 } ] \atop { { \mathbb E } _ { \mathcal D } [ x _ { i } ( \beta x _ { i } + \eta _ { i } ) ] } \right. \left. { { \mathbb E } _ { \mathcal D } [ x _ { i } ( \beta x _ { i } + \eta _ { i } ) ] } \right) = \left( { \sigma _ { x } ^ { 2 } } \sum _ { m _ { 1 } \sigma _ { x } ^ { 2 } } { m _ { 1 } \sigma _ { x } ^ { 2 } } \right) ,
$$

where the expectation over $\mathcal { D }$ averages $\beta$ over the finite pool. Each block has eigenvalues and (unnormalized) eigenfunctions

$$
\lambda _ { \pm } = \frac { 1 } { 2 } \left( \sigma _ { x } ^ { 2 } ( 1 + m _ { 2 } ) + \epsilon ^ { 2 } \pm \sqrt { \left( \sigma _ { x } ^ { 2 } ( 1 - m _ { 2 } ) - \epsilon ^ { 2 } \right) ^ { 2 } + 4 m _ { 1 } ^ { 2 } \sigma _ { x } ^ { 4 } } \right) ,
$$

$$
\phi _ { \pm , i } \propto m _ { 1 } \sigma _ { x } ^ { 2 } x _ { i } + ( \lambda _ { \pm } - \sigma _ { x } ^ { 2 } ) y _ { i } ,
$$

so the subkernel eigenvalues $c _ { 1 } \lambda _ { \pm }$ are each L-fold degenerate. Of these linear eigenfunctions, only $x _ { L + 1 }$ carries a nonzero target coefficient (the mean $\tilde { \mu }$ of Table 2).

Quadratic subkernel. Degree-2 polynomials have nonzero means, so at level 2 we must project out the constant; in practice this means centering each monomial. Projection against level 1 is automatic, since every third moment vanishes under the symmetry $z \mapsto - z$ . The multiplicities matter here. Since $\begin{array} { r } { ( z ^ { \top } \dot { z } ^ { \prime } ) ^ { 2 } = \sum _ { \mu \nu } z _ { \mu } z _ { \nu } z _ { \mu } ^ { \prime } z _ { \nu } ^ { \prime } . } \end{array}$ a product of two distinct coordinates appears twice, so its feature carries a factor ${ \sqrt { 2 } } .$ . With that convention, the eigenvalues of $K _ { 2 }$ are $c _ { 2 } / 2 !$ times those of the feature covariance. The covariance splits into blocks:

1. $x _ { L + 1 } ^ { 2 } - \sigma _ { x } ^ { 2 }$ with itself.

2. ${ \sqrt { 2 } } x _ { L + 1 } x _ { i }$ with ${ \sqrt { 2 } } x _ { L + 1 } y _ { i }$ : the linear-subkernel block with a factor of $x _ { L + 1 }$ attached, giving eigenvalues $2 \sigma _ { x } ^ { 2 } \lambda _ { \pm }$ , each L-fold degenerate.

3. For each pair $i < j ,$ , the $4 \times 4$ block of ${ \sqrt { 2 } } x _ { i } x _ { j } , { \sqrt { 2 } } x _ { i } y _ { j } , { \sqrt { 2 } } y _ { i } x _ { j }$ and $\sqrt { 2 } y _ { i } y _ { j }$ , of which there are $\bar { L } ( L - \bar { 1 ) } / 2$ identical copies. Because both positions share the same task, this block is $2 \mathbb { E } _ { \beta } [ B ( \beta ) ^ { * } \otimes B ( \beta ) ]$ ], where $B ( \beta )$ is the conditional covariance of $( x _ { i } , y _ { i } )$ given $\beta ;$ it is not the Kronecker square of the pool-averaged linear block.

The remaining polynomials require more care. Define the centered single-position features

$$
a _ { i } = x _ { i } ^ { 2 } - \sigma _ { x } ^ { 2 } , \qquad b _ { i } = \sqrt { 2 } \left( x _ { i } y _ { i } - m _ { 1 } \sigma _ { x } ^ { 2 } \right) , \qquad c _ { i } = y _ { i } ^ { 2 } - \left( m _ { 2 } \sigma _ { x } ^ { 2 } + \epsilon ^ { 2 } \right) .
$$

Let $D$ be their $3 \times 3$ covariance at a single position, and $C$ their cross-covariance between two distinct positions $i \neq j$ . Neither depends on which positions are chosen. Given $\beta ,$ distinct positions are independent, so C is the pool covariance of the features’ conditional means,

$$
C = \operatorname { C o v } _ { \beta } \left( \operatorname { \mathbb { E } } [ ( a _ { i } , b _ { i } , c _ { i } ) | \beta ] \right) = \sigma _ { x } ^ { 4 } \left( 0 \begin{array} { c c c } { 0 } & { 0 } & { 0 } \\ { 0 } & { 2 \operatorname { V a r } \beta } & { \sqrt { 2 } \operatorname { C o v } ( \beta , \beta ^ { 2 } ) } \\ { 0 } & { \sqrt { 2 } \operatorname { C o v } ( \beta , \beta ^ { 2 } ) } & { \operatorname { V a r } \beta ^ { 2 } } \end{array} \right) ,
$$

with variances and covariances taken over the pool. In particular, $\mathbb { E } [ b _ { i } b _ { j } ] = 2 \sigma _ { x } ^ { 4 } ( m _ { 2 } - m _ { 1 } ^ { 2 } )$ vanishes only when every task in the pool is identical. Unlike the linear case, distinct positions are therefore correlated through the shared task, and the features span a single $3 L \times 3 L$ block, $I _ { L } \otimes ( D - C ) +$ $\mathbf { 1 1 } ^ { \top } \otimes C .$ . Its eigensystem is the union of the following:

1. $D + ( L - 1 ) C$ , for functions summed over the context, $\textstyle \sum _ { i } ( \cdot ) _ { i } ;$

2. $D - C ,$ , which is $( L - 1 )$ )-fold degenerate, for contrasts between positions, such as $b _ { i } - b _ { j }$

The first sector contains the two eigenvalues that grow with $L ;$ its eigenfunctions are spanned by $\sum _ { i } b _ { i }$ and $\sum _ { i } c _ { i }$ and diagonalize the pool covariance of $( \beta , \beta ^ { 2 } )$ . The sum $\textstyle \sum _ { i } a _ { i }$ does not grow, since the first row of $C$ vanishes. These are the $d = 1$ versions of the families ${ \bar { S } } , { \bar { A } }$ and $W$ of Section F.2, with $\beta$ and $\beta ^ { 2 }$ as the task objects of $S$ and A.

Cubic and higher subkernels. We do not track the higher subkernels in detail, but one property is key. The ICL-relevant function $\textstyle \sum _ { i } x _ { i } y _ { i } x _ { L + 1 }$ lies in the context-sum sector, whose eigenvalue grows linearly in L through $C _ { b b } = 2 \sigma _ { x } ^ { 4 } \operatorname { V a r } \beta$ , and therefore needs only to be orthonormalized using the subkernel expansion framework.

In higher dimensional data The same procedure outlined above applies in d dimensions. Within a position, the label $y _ { i } = \beta ^ { \top } \pmb { x } _ { i } + \eta _ { i }$ now couples to every coordinate of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ , so the per-position blocks become $( d + 1 ) \times ( d + 1 )$ . Across positions, the structure is as in one dimension: distinct positions are uncorrelated at level 1, and at level 2 and above they are correlated only through the shared task.

## F EIGENCOEFFICIENTS OF THE ICL LINEAR REGRESSION PROBLEM

## F.1 KERNELS ARE APPROXIMATIONS OF MEMORIZING SOLUTIONS

In the main text we used the fact that a kernel learner’s eigencoefficients are those of the memorizing solution. This holds in any supervised setting, not only ICL: it is the standard statement that kernel regression targets the conditional mean under the training distribution. We state it in the form we use.

Lemma 1 (Kernel eigencoefficients are those of the memorizing solution). Let $\{ \phi _ { i } ( z ) \}$ be the (z-measurable) eigenbasis of a PSD kernel K. As $f _ { \mathrm { m e m } } ( z ) \stackrel { - } { = } \mathbb { E } [ y | z ]$ is the projection of $y ^ { \star }$ onto z-measurable functions, the eigencoefficients of the target and a memorizing predictor are mode-identical:

$$
v _ { i } \equiv \mathbb { E } _ { ( z , y ) \sim { \mathcal D } _ { \mathrm { t r } } } [ f ^ { \star } ( z ) \phi _ { i } ( z ) ] = \mathbb { E } _ { ( z , y ) \sim { \mathcal D } _ { \mathrm { t r } } } [ y \phi _ { i } ( z ) ] = \mathbb { E } _ { ( z , y ) \sim { \mathcal D } _ { \mathrm { t r } } } [ f _ { m e m } ( z ) \phi _ { i } ( z ) ]\tag{38}
$$

for all i. Therefore, an infinite-sample kernel will implement the memorizing solution exactly, with a finite-sample kernel implementing an approximation based on the kernel’s largest eigenvalues.

Proof. Let $\mathcal { D } _ { \mathrm { t r } }$ be the training measure on contexts, let $y ^ { \star } \in L ^ { 2 }$ be a target whose randomness is not exhausted by z, and let $f _ { \mathrm { m e m } } ^ { - } ( z ) \equiv \mathbb { E } [ y ^ { \star } | z ]$ be the Bayes-optimal predictor under ${ \mathcal { D } } _ { \mathrm { t r } }$ . For any orthonormal set $\{ \phi _ { i } \}$ }, the law of total expectation yields

$$
\begin{array} { r } { \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { \mathrm { t r } } } [ y ^ { \star } , \phi _ { i } ] = \mathbb { E } \left[ \mathbb { E } \left[ y ^ { \star } \phi _ { i } ( z ) | z \right] \right] = \mathbb { E } \left[ \phi _ { i } ( z ) \mathbb { E } \left[ y ^ { \star } | z \right] \right] = \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { \mathrm { t r } } } [ f _ { \mathrm { m e m } } , \phi _ { i } ] . } \end{array}
$$

## F.2 WHITENED EIGENSYSTEM

The main text takes $\Sigma _ { x } \ \propto \ I _ { d }$ . General $\Sigma _ { x }$ reduces to this case through the rescaling $\beta ^ { \top } { \pmb x } =$ $( \Sigma _ { \pmb { x } } ^ { 1 / 2 } \beta ) ^ { \top } ( \Sigma _ { \pmb { x } } ^ { - 1 / 2 } \pmb { x } )$ This invertible linear map leaves the space of polynomials of each degree unchanged, so the eigencoefficients are those of the isotropic case with the tasks replaced by the colored tasks $\Sigma _ { x } ^ { 1 / 2 } \beta$ , and the cumulants that enter are those of the colored pool. The kernel itself is not whitened: its eigenvalues are still computed in raw coordinates and depend on $\Sigma _ { x }$

We denote whitened and colored quantities with a tilde,

$$
\tilde { \ b { x } } = \Sigma _ { \ b { x } } ^ { - 1 / 2 } \pmb { x } , \qquad \tilde { \ b { \beta } } = \Sigma _ { \ b { x } } ^ { 1 / 2 } \pmb { \beta } , \qquad \tilde { \mathcal { B } } _ { \mathrm { t r } } = \mathcal { U } \left( \{ \Sigma _ { \ b { x } } ^ { 1 / 2 } \pmb { \beta } ^ { ( t ) } \} _ { t = 1 } ^ { T } \right) ,\tag{39}
$$

and write $\tilde { \kappa } _ { \ell }$ for the cumulants of $\tilde { B } _ { \mathrm { t r } }$ ; by multilinearity (Equation (19)), $\tilde { \kappa } _ { \ell }$ is $\hat { \kappa } _ { \ell }$ with every index contracted against $\Sigma _ { x } ^ { 1 / 2 }$ . We write $\tilde { \mu } , \tilde { M _ { 2 } }$ and $\tilde { \Sigma }$ for the mean, second moment, and covariance of $\tilde { B } _ { \mathrm { t r } } , ( \tilde { s } _ { a } , { \bf \em u } _ { a } )$ for the eigenpairs of $ { \tilde { \Sigma } } .$ , and $\mathrm { T r } _ { k }$ for the partial trace of Equation (20). Whitening gives $\tilde { \mathbf { x } } _ { i } \sim \mathcal { N } ( 0 , I _ { d } )$ ), and the labels $y _ { i } = \tilde { \beta } ^ { \top } \tilde { \pmb { x } } _ { i } + \eta _ { i }$ are unchanged. We define the per-task label variance and its pool average,

$$
\tau ( \beta ) = \beta ^ { \top } \Sigma _ { \alpha } \beta + \epsilon ^ { 2 } = \| \tilde { \beta } \| ^ { 2 } + \epsilon ^ { 2 } , \qquad \bar { \tau } = \mathbb { E } _ { \mathcal { B } _ { \mathrm { t r } } } [ \tau ( \beta ) ] = \mathrm { T r } \Big ( \tilde { M } _ { 2 } \Big ) + \epsilon ^ { 2 } .\tag{40}
$$

We collect the L in-context pairs, but not the query, into the context matrix $z _ { \mathrm { c } } = [ \tilde { X } \ \pmb { y } ] \in \mathbb { R } ^ { L \times ( d + 1 ) }$ whose i-th row is $( \tilde { \pmb { x } } _ { i } ^ { \top } , y _ { i } )$ , and form its Gram matrix

$$
M \equiv z _ { \mathrm { c } } ^ { \top } z _ { \mathrm { c } } = \left( \begin{array} { c c } { { W } } & { { S } } \\ { { S ^ { \top } } } & { { A } } \end{array} \right) , \qquad W = \sum _ { i = 1 } ^ { L } \tilde { x } _ { i } \tilde { x } _ { i } ^ { \top } , \quad S = \sum _ { i = 1 } ^ { L } y _ { i } \tilde { x } _ { i } , \quad A = \sum _ { i = 1 } ^ { L } y _ { i } ^ { 2 } .\tag{41}
$$

Proposition 1 (The target lives on polynomials in M). Given $\beta ,$ , the rows of $z _ { \mathrm { c } }$ are i.i.d. $\mathcal { N } ( 0 , \Sigma _ { z | \beta } )$ with $\Sigma _ { z | \beta } = \bigl ( { { \ L } _ { \tilde { \beta } } ^ { I _ { d } } } _ { \tau ( \beta ) } ^ { \tilde { \beta } } \bigr )$ , and the query $\tilde { \mathbf { x } } _ { L + 1 } \sim \mathcal { N } ( 0 , I _ { d } )$ is independent of $z _ { \mathrm { c } } .$ . Hence, under the pool or under the prior, the density of $z _ { \mathrm { c } }$ depends on $z _ { \mathrm { c } }$ only through M, and it is invariant under $z _ { \mathrm { c } } \mapsto Q z _ { \mathrm { c } } f o r$ every $Q \in O ( L )$ acting on context positions. A rotation-invariant kernel is invariant under the same map applied to both arguments, since it leaves $\begin{array} { r } { z ^ { \top } z ^ { \prime } = \operatorname { T r } \left( z _ { \mathrm { c } } ^ { \top } z _ { \mathrm { c } } ^ { \prime } \right) + \pmb { x } _ { L + 1 } ^ { \top } \pmb { x } _ { L + 1 } ^ { \prime } } \end{array}$ <sub>1</sub> and $\| z \|$ unchanged. Itfollows that:

1. The kernel’s eigenfunctions can be chosen to be either $O ( L )$ -invariant or orthogonal to every $O ( L )$ )-invariant function. The target is invariant, so only invariant eigenfunctions carry nonzero eigencoefficients, under the pool or the prior. The invariant polynomials of $z _ { \mathrm { c } }$ are exactly the polynomials in the entries of M.

2. Within the subkernel expansion (Definition 4), the degree-p eigenfunctions separate by Hermite degree in the query, because the query is Gaussian and independent of the context. The target is linear in $\tilde { \pmb { x } } _ { L + 1 }$ , so it overlaps only eigenfunctions ofHermite degree one in the query.

Every eigenfunction with a nonzero eigencoefficient is therefore of the form $\phi ( M ) ^ { \top } \tilde { \pmb { x } } _ { L + 1 }$ , with ϕ a vector ofpolynomials in the entries ofM. At data degree $p = 2 k + 1$ , these polynomials have degree at most k in M.

Proof. For (i), the integral operator of K commutes with the unitary action of $O ( L )$ on $L ^ { 2 }$ , so each of its eigenspaces is $\bar { O } ( L )$ -invariant and splits orthogonally into its invariant functions and their complement. An invariant target has zero inner product with the complement. That the invariant polynomials are the polynomials in M is the first fundamental theorem of invariant theory for $O ( L )$ applied to the $d + 1$ columns of $z _ { \mathrm { c } }$ . For (ii), after the projections of Definition 4, a level-p feature of query degree $j$ retains only the Hermite-degree-j part of its query factor. Hermite polynomials of different degree in $\tilde { \pmb { x } } _ { L + 1 }$ are orthogonal, and they are independent of the context. □

In particular, sums that are invariant under permutations of the context but not under $O ( L )$ , such as $\sum _ { i } ^ { \cdot } y _ { i } ^ { 4 } \mathrm { o r } \sum _ { i } y _ { i } ^ { 3 } \tilde { { \pmb x } } _ { i }$ , carry no coefficient of their own; their invariant parts lie in the span of polynomials in M. We call a monomial type

$$
\psi _ { ( n _ { W } , n _ { S } , n _ { A } ) } = W ^ { \otimes n _ { W } } \otimes S ^ { \otimes n _ { S } } A ^ { n _ { A } } \otimes \tilde { x } _ { L + 1 } , \qquad n _ { W } + n _ { S } + n _ { A } = k ,\tag{42}
$$

a family of degree $p = 2 k + 1$ . For example, the degree-3 families are $S \otimes \tilde { \pmb { x } } _ { L + 1 } , A \tilde { \pmb { x } } _ { L + 1 }$ and $\dot { W } \otimes  { \tilde { \pmb { x } } } _ { L + 1 }$ , and there are six families at degree 5. Every family carries a single factor of the query. The query is independent of the context and has $\mathbb { E } [ \tilde { \mathbf { x } } _ { L + 1 } \tilde { \mathbf { x } } _ { L + 1 } ^ { \top } ] ^ { \top } = I _ { d }$ , so perpendicularization (as is done in Section E) acts on the context factor alone. We write $\tilde { \Psi } _ { p } = \tilde { \Psi } _ { p } ^ { \mathrm { c } } \otimes \tilde { \pmb { x } } _ { L + 1 }$ , where $\tilde { \Psi } _ { p } ^ { \mathrm { c } }$ stacks the context factors of all degree-p families, and write $\tilde { \Psi } _ { p } ^ { \mathrm { c } \perp }$ for its perpendicular version.

Definition 5 (Colored eigenfunctions, general $\Sigma _ { x } )$ . Let $\Gamma _ { p } \equiv \mathbb { E } \big [ \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp } \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp \top } \big ]$ be the Gram matrix of the degree-p context factors. It is singular because the tensor families contain repeated entries, so inverses are taken on its range. For any orthogonal matrix $R _ { p }$

$$
\Phi _ { p } \equiv R _ { p } \Gamma _ { p } ^ { - 1 / 2 } \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp } , \qquad \tilde { \phi } _ { p , m , b } = \Phi _ { p , m } \tilde { x } _ { L + 1 , b }\tag{43}
$$

are orthonormal under the training measure and span the degree-p eigenfunctions allowed by Proposi tion 1. Unless stated otherwise we take $R _ { p } = I ;$ the kernel’s own eigenbasis corresponds to the $R _ { p }$ given below.

The families of a single degree are not in general orthogonal to one another. At degree 3 they nearly are. The leading-order overlap between the S and A families is set by the pool covariance $\mathrm { C o v } _ { \tilde { B } _ { \mathrm { t r } } } ( \tilde { \beta } , \| \tilde { \beta } \| ^ { 2 } ) = \bar { \mathrm { T r } } _ { 1 } \tilde { \kappa } _ { 3 } + 2 \tilde { \Sigma } \tilde { \mu }$ . For a centered prior this is $O ( T ^ { - 1 / 2 } )$ , and it vanishes for a centered, symmetric pool. The overlaps of W with $S$ and A fall off as $L ^ { - 1 / 2 }$ . From degree 5 on, each family’s context factor concentrates on $L ^ { k }$ times its task object $\tilde { \beta } ^ { \otimes n _ { S } } \tau ^ { n _ { A } }$ (Section $\mathrm { G } )$ , and the pool covariances between these task objects do not vanish as $L \to \infty$ . No choice of families is then orthogonal, and the choice of $R _ { p }$ matters.

The kernel’s eigenbasis within a degree. Definition 5 fixes the degree-p span but not the basis within it; the kernel’s eigenfunctions correspond to one particular $R _ { p }$ . By Definition 4, the subkernel $\begin{array} { r } { K _ { p } = \frac { c _ { p } } { p ! } ( z ^ { \top } z ^ { \prime } ) ^ { \tau } } \end{array}$ acts through the degree-p monomials of the raw context $\{ ( z ) _ { p } \}$ , with lower degrees projected out; write $\widetilde { ( z ) } _ { p , \mu }$ for these projected monomials. By Proposition 1, the span of Equation (43) is invariant under $K _ { p } ,$ so step 6 of Definition 4 can be carried out inside it. In the basis $\tilde { \phi } _ { p , m , b }$ (with $R _ { p } = I ) , K _ { p }$ is the matrix

$$
[ \widetilde { K } _ { p } ] _ { ( m , b ) , ( m ^ { \prime } , b ^ { \prime } ) } = \frac { c _ { p } } { p ! } \sum _ { \mu } \mathbb { E } \big [ \widetilde { \phi } _ { p , m , b } \widetilde { ( z ) } _ { p , \mu } \big ] \mathbb { E } \big [ \widetilde { \phi } _ { p , m ^ { \prime } , b ^ { \prime } } \widetilde { ( z ) } _ { p , \mu } \big ] ,\tag{44}
$$

which measures how strongly two orthonormal polynomials load onto the same monomials. Its eigenvectors give the rows of $R _ { p } { \mathrm { . } }$ , and its eigenvalues are the kernel eigenvalues. Because the query is independent of the context, ${ \boldsymbol { \kappa } } _ { p }$ factorizes into a context part times $\Sigma _ { x } \colon$ the query index is rotated into the eigenbasis of $\Sigma _ { x }$ , and $\bar { \boldsymbol { R _ { p } } _ { p } }$ acts on the context index m alone. Since the monomials are in raw coordinates, $R _ { p }$ depends on $\Sigma _ { x }$ even though the span does not.

## F.3 THE EIGENCOEFFICIENTS

We start with a lemma for evaluating conditional expectations of the context factors.

Lemma 2 (Conditional moments of a context factor). For integers $a , b \geq 0 ,$ , the context factor $\begin{array} { r } { F _ { a , b } \equiv \sum _ { i = 1 } ^ { L } y _ { i } ^ { a } \tilde { \mathbf { x } } _ { i } ^ { \otimes b } } \end{array}$ is a sum of L i.i.d. terms given $\beta .$ Every joint cumulant of such factors given $\beta$ is therefore L times the corresponding joint cumulant at a single position; in particular

$$
\operatorname { \mathbb { E } } \big [ F _ { a , b } \big | \beta \big ] = L \operatorname { \mathbb { E } } \big [ y ^ { a } \tilde { x } ^ { \otimes b } \big | \beta \big ] , \qquad \operatorname { C o v } \big ( F _ { a , b } \big | \beta \big ) = L \operatorname { C o v } \big ( y ^ { a } \tilde { x } ^ { \otimes b } \big | \beta \big ) .\tag{45}
$$

The single-position cases used below are

$$
\mathbb { E } \big [ y \tilde { \pmb { x } } \big | \beta \big ] = \tilde { \beta } , \qquad \mathrm { C o v } \big ( y \tilde { \pmb { x } } \big | \beta \big ) = \tilde { \beta } \tilde { \pmb { \beta } } ^ { \top } + \tau I _ { d } ,\tag{46}
$$

$$
\begin{array} { r } { \mathbb { E } \left[ y ^ { 2 } \big | \beta \right] = \tau , \mathrm { V a r } \big ( y ^ { 2 } \big | \beta \big ) = 2 \tau ^ { 2 } , } \end{array}\tag{47}
$$

$$
\mathbb { E } \big [ \tilde { x } _ { a } \tilde { x } _ { c } \big | \beta \big ] = \delta _ { a c } , \qquad \mathrm { C o v } \big ( \tilde { x } _ { a } \tilde { x } _ { c } , \tilde { x } _ { a ^ { \prime } } \tilde { x } _ { c ^ { \prime } } \big | \beta \big ) = \delta _ { a a ^ { \prime } } \delta _ { c c ^ { \prime } } + \delta _ { a c ^ { \prime } } \delta _ { c a ^ { \prime } } ,\tag{48}
$$

$$
\begin{array} { r } { \mathrm { C o v } \big ( y \tilde { x } , y ^ { 2 } \big | \beta \big ) = 2 \tau \tilde { \beta } , \qquad \mathrm { C o v } \big ( y \tilde { x } _ { b } , \tilde { x } _ { a } \tilde { x } _ { c } \big | \beta \big ) = \delta _ { a b } \tilde { \beta } _ { c } + \delta _ { b c } \tilde { \beta } _ { a } , } \end{array}\tag{49}
$$

$$
\mathrm { C o v } \left( y ^ { 2 } , \tilde { x } _ { a } \tilde { x } _ { c } \vert \beta \right) = 2 \tilde { \beta } _ { a } \tilde { \beta } _ { c } ,
$$

with $\tau = \tau ( \beta )$ .

Proof. The positions are independent given $\beta ,$ , and cumulants add over independent summands. For $( 4 6 ) , y \tilde { \mathbfit { x } } = ( \tilde { \beta } ^ { \top } \tilde { \mathbfit { x } } ) \tilde { \mathbfit { x } } + \eta \tilde { \mathbfit { x } }$ has mean ${ \tilde { \beta } } .$ , and by Isserlis’ theorem $\mathbb { E } [ ( \tilde { \beta } ^ { \top } \tilde { \pmb { x } } ) ^ { 2 } \tilde { \pmb { x } } \tilde { \pmb { x } } ^ { \top } ] = 2 \tilde { \beta } \tilde { \beta } ^ { \top } + \| \tilde { \beta } \| ^ { 2 } I _ { d } .$ so the second moment is $2 \tilde { \beta } \tilde { \beta } ^ { \top } + \tau I _ { d }$ and the covariance follows. For $( 4 7 ) , y \mid \beta \sim \mathcal { N } ( 0 , \tau )$ gives $\mathrm { V a r } ( y ^ { 2 } \mid \beta ) = 2 \tau ^ { 2 }$ . The remaining entries follow in the same way, since $( { \tilde { x } } , y ) \mid \beta$ is jointly Gaussian with covariance $\Sigma _ { z | \beta }$ (Proposition 1). □

Task objects. By Lemma 2, each context sum concentrates on a function of the task: $W \approx L I _ { d }$ $S \approx L \tilde { \beta }$ and $A \approx L \tau$ . The context factor of a degree $\cdot p = 2 k + 1$ family therefore has conditional mean

$$
\mathbb { E } \big [ W ^ { \otimes n _ { W } } \otimes S ^ { \otimes n _ { S } } A ^ { n _ { A } } \big | \beta \big ] = L ^ { k } I _ { d } ^ { \otimes n _ { W } } \otimes \tilde { \beta } ^ { \otimes n _ { S } } \tau ^ { n _ { A } } + O ( L ^ { k - 1 } ) ,\tag{50}
$$

where the $O ( L ^ { k - 1 } )$ corrections come from coincident positions. We call $f ( \beta ) = I _ { d } ^ { \otimes n _ { W } } \otimes \tilde { \beta } ^ { \otimes n _ { S } } \tau ^ { n _ { A } }$ the family’s task object. We write $f _ { p }$ for the stacked task objects of all degree-p families, $f _ { p } ^ { \perp }$ for their residual after regression on all lower-degree task objects under the pool, and

$$
G _ { p } ^ { \perp } \equiv \mathrm { C o v } _ { \tilde { B } _ { \mathrm { t r } } } \big ( { f } _ { p } ^ { \perp } \big ) .\tag{51}
$$

Directions with $G _ { p } ^ { \perp } \neq 0$ are task modes. Directions with $G _ { p } ^ { \perp } = 0$ are noise modes: the W families, and trace components such as $\operatorname { T r } ( S \otimes S ) = \| S \| ^ { 2 }$ , whose task objects repeat lower-degree ones.

Theorem 1 (The eigencoefficient-cumulant correspondence). Let $\tilde { \phi } _ { p , m , b } = \Phi _ { p , m } \tilde { x } _ { L + 1 , b }$ be the orthonormal polynomials ofDefinition 5, and let $\begin{array} { r } { \pmb { q } _ { p } ( \beta ) \equiv \mathbb { E } \left[ \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp } \mid \beta \right] } \end{array}$

1. Exact form. The eigencoefficients are $v _ { 1 , b } = \tilde { \mu } _ { b }$ and, for $p \geq 3 ,$

$$
v _ { p , \cdot , b } = R _ { p } \Gamma _ { p } ^ { - 1 / 2 } V _ { p , b } , \qquad V _ { p , b } = \mathbb { E } \big [ \tilde { \beta } _ { b } \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp } \big ] = \mathrm { C o v } _ { \tilde { B } _ { \mathrm { t r } } } \big ( \tilde { \beta } _ { b } , ~ { q } _ { p } ( \beta ) \big ) .\tag{52}
$$

2. Scaling with L. Degree-3 task-mode coefficients are $O ( 1 )$ . For $p = 2 k + 1 \geq 5 , V _ { p , b } =$ $O ( L ^ { k - 1 } )$ while $\Gamma _ { p } = L ^ { 2 k } G _ { p } ^ { \perp } + { \cal O } ( L ^ { 2 k - 1 } )$ , so task-mode coefficients are $O ( L ^ { - 1 } )$ and noise-mode coefficients are $\overset { \cdot } { O } ( L ^ { - 1 / 2 } )$

3. Cumulant content. At degree $2 k + 1$ , the numerator ofafamily has at most $k + 1$ free task indices, and only $S ^ { \otimes k }$ attains this bound. The cumulant $\tilde { \kappa } _ { \omega }$ thereforefirst appears with all indices free at degree $2 \omega - 1$ , in the $S ^ { \otimes ( \omega - 1 ) }$ block; every lower degree contains it only through partial traces. The diagonal block of $\operatorname { T } _ { p } f o r S ^ { \otimes ( \omega - 1 ) }$ involves pool moments up to order $2 \omega - 2$ , and the full $\Gamma _ { p } ,$ through the A families, involves moments up to order 4ω − 4.

Proof. (i). By Lemma 1, $v _ { p , m , b } = \mathbb { E } [ y ^ { \star } \tilde { \phi } _ { p , m , b } ]$ . The query is independent of the context and $\mathbb { E } [ y _ { L + 1 } \tilde { \pmb { x } } _ { L + 1 } | \beta ] = \tilde { \beta } .$ , so $v _ { p , m , b } = \mathbb { E } [ \beta _ { b } \Phi _ { p , m } ]$ , and Equation (43) gives the first equality. Conditioning on $\beta$ gives $V _ { p , b } = \mathbb { E } _ { \tilde { B } _ { \mathrm { t r } } } [ \tilde { \beta } _ { b } \pmb { q } _ { p } ]$ . For $p \geq 3 , \tilde { \Psi } _ { p } ^ { \mathrm { c } \perp }$ is orthogonal to the constant (the degree-1 context factor), so $\mathbb { E } [ q _ { p } ] = 0$ and this expectation is a covariance. For $p = 1$ the context factor is 1, and $v _ { 1 , b } = \tilde { \mu } _ { b }$

(ii). By Lemma 2, the covariance between context factors of degrees $2 k + 1$ and $2 k ^ { \prime } + 1$ is $L ^ { k + k ^ { \prime } }$ times the pool covariance of their task objects, up to $O ( L ^ { k + k ^ { \prime } - 1 } )$ . At leading order, the perpendicularization of Definition 3 is therefore the regression of $f _ { p }$ on the lower-degree task objects under the pool, so $\pmb q _ { p } = L ^ { k } \pmb f _ { p } ^ { \perp } + O ( L ^ { k - 1 } )$ and $\Gamma _ { p } ^ { - } = L ^ { 2 k } G _ { p } ^ { \perp } + { \cal O } ( L ^ { 2 k - 1 } )$ . $\mathrm { A t } p = 3$ , the only lower-degree task object is the constant, so $V _ { 3 , b } = L \mathrm { C o v } ( \tilde { \beta } _ { b } , { \bf f } _ { 3 } ) + O ( 1 )$ and the task-mode coefficients are $O ( 1 )$ . For $p \geq 5$ , the lower degrees include the $S$ family, whose task object is $\tilde { \beta }$ itself. The residual $f _ { p } ^ { \perp }$ is then uncorrelated under the pool with every affine function of ${ \tilde { \beta } } ,$ so $\mathrm { C o v } ( \tilde { \beta } _ { b } , \pmb { f } _ { p } ^ { \perp } ) = 0$ and the $O ( L ^ { k } )$ part of $V _ { p , b }$ vanishes. What survives is $O ( L ^ { k - 1 } )$ ). It arises because the data-space regression is onto the context factors rather than onto their task objects. For instance, S has conditional covariance $L ( { \tilde { \beta } } { \tilde { \beta } } ^ { \top } + \tau I _ { d } )$ by (46), which attenuates its regression coefficient by a relative $O ( L ^ { - 1 } )$ and leaves part of the linear component of $f _ { p }$ in place. Dividing by $\Gamma _ { p } ^ { 1 / 2 }$ , which is $O ( L ^ { k } )$ ) on task modes and $O ( L ^ { k - 1 / 2 } )$ on noise modes, gives the stated orders.

(iii). A factor W contributes $I _ { d }$ and a factor A contributes the scalar τ, so the task object of $( n _ { W } , n _ { S } , n _ { A } )$ has $n _ { S }$ free task indices, and the target adds the index b. Contracting against $I _ { d } , \tau \mathrm { o r }$ the regression coefficients adds no free indices, so $V _ { p , b }$ has at most $n _ { S } + 1 \le k + 1$ 1 of them, with equality only for $S ^ { \otimes k }$ . A pool moment of order $\omega$ with all of its indices free, and hence $\tilde { \kappa } _ { \omega } ,$ , requires $k + 1 \ge \omega$ . It can therefore first appear at degree $2 \omega - 1$ , and only in the $S ^ { \otimes ( \omega - 1 ) }$ block. It does appear there: the attenuation term of (ii) contains a term proportional to $\mathrm { C o v } _ { \tilde { B } _ { \mathrm { t r } } } ( \tilde { \beta } ^ { \otimes ( \omega - 1 ) } , \tilde { \beta } )$ , as in the example below. The diagonal block of $\Gamma _ { p }$ for $S ^ { \otimes ( \omega - 1 ) }$ pairs two task objects of degree $\omega - 1$ giving moments up to order $2 \omega - 2$ . The A families have task objects of degree up to $2 \omega - 2$ in ${ \tilde { \beta } } ,$ so the full $\Gamma _ { p }$ involves moments up to order $4 \omega - 4$ □

The above eigencoefficient equation is highly opaque, and will remain so for full completeness.   
Nonetheless, we detail the first few eigencoefficients for lower-degree polynomials.

Example: the SS block at degree 5. Take a centered pool with $\mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } = 0$ . Then S is uncorrelated with A, W and the constant, and only the regression of $\bar { S } \otimes S$ on S contributes to $V _ { S S , b }$ . By Lemma 2,

$$
\begin{array} { r l } { \mathbb { E } [ S \otimes S | \beta ] = L ( L + 1 ) \tilde { \beta } \tilde { \beta } ^ { \top } + L \tau I _ { d } , } & { \quad \mathrm { C o v } ( S ) = L ( L + 1 ) \tilde { \Sigma } + L \bar { \tau } I _ { d } } \\ { \mathbb { E } [ S ^ { \otimes 3 } ] = ( L ^ { 3 } + 3 L ^ { 2 } ) \tilde { \kappa } _ { 3 } + O ( L ) . } \end{array}
$$

The regression coefficient of $S \otimes S$ on S is $\mathbb { E } [ S ^ { \otimes 3 } ] \operatorname { C o v } ( S ) ^ { - 1 }$ , and $\operatorname { C o v } ( \tilde { \beta } _ { b } , \mathbb { E } [ S \mid \beta ] ) = L \tilde { \Sigma } _ { b } .$ <sub>·</sub>, so

$$
V _ { S S , b } = L ( L + 1 ) \tilde { \kappa } _ { 3 , b . } - ( L ^ { 3 } + 3 L ^ { 2 } ) \big [ \big ( ( L + 1 ) \tilde { \Sigma } + \bar { \tau } I _ { d } \big ) ^ { - 1 } \tilde { \Sigma } \big ] _ { b b ^ { \prime } } \tilde { \kappa } _ { 3 , b ^ { \prime } - } = L \big [ \big ( \bar { \tau } \tilde { \Sigma } ^ { - 1 } - I _ { d } \big ) \tilde { \kappa } _ { 3 } \big ] _ { b . } + O ( 1 ) .
$$

![](images/333bba06b9b9ca29afa6f551ae3054da193f8adcf095c0d9757c62ef3caf87b7.jpg)

![](images/94cb713889459f5747021aef7b6e3f5d358bda7a2403fdb217dfbb6ed2d17bfd.jpg)  
Figure 9: Gaussian kernel learning curves on ICL Regression data. For four different settings (colors) of the task diversity T, we compute the expected test MSE as a function of the number of training samples n, and compare it to empirical errors obtained across 20 trials. Theoretical learning curve equations are provided in Section D, with validation of the eigensystem the equations are built off of provided in Sections E and F. Our data uses a Gaussian kernel with kernel width $\sigma _ { \mathrm { k e r } } = { \sqrt { 3 } } ,$ $d = \hat { 5 , L } = 7 , \epsilon ^ { 2 } = 0 .$ 1 where our data is drawn ${ \pmb x } \sim \mathcal { N } ( 0 , I _ { d } / ( d L ) )$ . Additionally, as $T \to \infty ,$ , we find that the errors on the pretraining distribution match those of the true distribution, as expected. Further experimental details are in Section B.

The $O ( L ^ { 2 } )$ terms cancel as in (ii). The surviving $\bar { \tau } \tilde { \Sigma } ^ { - 1 }$ is the per-position noise-to-signal ratio of S, which sets the attenuation. With $\Gamma _ { S S } = L ^ { 4 } G _ { 5 } ^ { \perp } \overline { { { + O ( L ^ { 3 } ) } } }$ , this gives the SS entry of Table 2.  
Equation (52) is exact, but its content is easiest to read family by family. Table 2 lists the leading-order coefficients through degree seven.
<table><tr><td>Degree p</td><td>family</td><td>task object</td><td>eigencoefficient (leading order)</td></tr><tr><td>1</td><td>1</td><td>1</td><td> $\tilde { \mu } _ { b }$ </td></tr><tr><td>3</td><td>S</td><td> $\tilde { \beta }$ </td><td> $\tilde { s } _ { a } / \sqrt { ( 1 + L ^ { - 1 } ) } \tilde { s } _ { a } + \bar { \tau } / L$ </td></tr><tr><td></td><td> $A$ </td><td> $\tau$ </td><td> $c _ { b } / \sqrt { \mathrm { V a r } \tau + 2 \mathbb { E } \tau ^ { 2 } / L } , c = \mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } + 2 \tilde { \Sigma } \tilde { \mu }$ </td></tr><tr><td></td><td>W</td><td> $I _ { d }$ </td><td>0 (noise mode)</td></tr><tr><td>5</td><td>SS</td><td> $\tilde { \beta } ^ { \otimes 2 }$ </td><td> $L ^ { - 1 } ( G _ { 5 } ^ { \perp } ) ^ { - 1 / 2 } ( \bar { \tau } \tilde { \Sigma } ^ { - 1 } - I _ { d } ) \tilde { \kappa } _ { 3 }$ </td></tr><tr><td></td><td>SA</td><td> $\tau \tilde { \beta }$ </td><td> ${ \cal L } ^ { - 1 } ( G _ { 5 } ^ { \bot } ) ^ { - 1 / 2 } \left( - ( 2 \mathrm { T r } \tilde { \Sigma } ^ { 2 } I _ { d } + 4 \tilde { \Sigma } ^ { 2 } ) + \dots \right)$ </td></tr><tr><td></td><td>AA</td><td> $\tau ^ { 2 }$ </td><td> $O ( L ^ { - 1 } ) ;$  odd moments</td></tr><tr><td></td><td>W families</td><td> $\cdot \cdot \cdot$ </td><td> $O ( L ^ { - 1 / 2 } )$  (noise modes)</td></tr><tr><td>7</td><td>SSS</td><td> $\tilde { \beta } ^ { \otimes 3 }$ </td><td> $L ^ { - 1 } ( G _ { 7 } ^ { \perp } ) ^ { - 1 / 2 } \frac { d } { d + 2 } \frac { \epsilon ^ { 2 } } { \tilde { s } } \tilde { \kappa } _ { 4 }$ </td></tr><tr><td></td><td>SAA</td><td> $\tau ^ { 2 } \tilde { \beta }$ </td><td> $\propto L ^ { - 1 } ~ \mathrm { T r } _ { 2 } ~ \tilde { \kappa } _ { 6 }$ </td></tr><tr><td></td><td>SSA, AAA</td><td> $\tau \tilde { \beta } ^ { \otimes 2 } , \tau ^ { 3 }$ </td><td>odd moments; zero for symmetric pools</td></tr><tr><td></td><td>W families</td><td>lower-degree objects</td><td> $O ( L ^ { - 1 / 2 } )$  (noise modes)</td></tr></table>

Table 2: Leading-order eigencoefficients through degree seven, in the symmetric basis $R _ { p } ~ = ~ I$ (Definition 5). $G _ { p } ^ { \perp }$ is the pool covariance of the degree-p task objects after regression on lower-degree ones, joint across families. Degree-3 rows treat the $S ,$ A and W blocks as uncoupled and are written in the eigenbasis of $\tilde { \Sigma } .$ . The S row is exact at every L when $\tilde { \mu } = 0$ and $\mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } = 0 .$ , and the A row is first order in c. Degree-5 rows are leading order in $\dot { 1 } / L$ . The SS row assumes $\tilde { \mu } = 0$ and $\mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } = 0 ;$ otherwise terms in µ˜ and $\mathrm { T r } _ { 1 } \ \tilde { \kappa } _ { 3 }$ are added. The SA row shows the Gaussian part for a centered pool. Degree-7 rows are leading order in $1 / L$ for an isotropic centered pool $( \tilde { \Sigma } = \tilde { s } I _ { d } )$ , at first order in the deviation from a Gaussian. Noise modes are the W families and trace components such as $\operatorname { T r } ( S \otimes S )$ (part of the SS family), whose task objects repeat lower-degree ones.

## F.4 VALIDATION OF THE EIGENCOEFFICIENTS

To validate our theoretically derived eigencoefficients as the correct set used by the kernel learner, we use the theoretically-derived learning curve (test error vs number of samples) plotted against empirically found errors. We do not directly compare the eigencoefficients that are found empirically to those theoretically derived, as we note that the “interesting region” of the eigencoefficients (the portion of the spectrum beyond the degree 1 terms) is partially affected by noise, with us not having enough memory to obtain clean empirical estimates. Therefore, we test our eigencoefficients in a different manner. Should our eigensystem, including the eigencoefficients, be correct, the learning curve (test error given number of samples) can be theoretically predicted, and will match well with empirical findings. In Figure 9, we do said experiment and find excellent agreement between our pure-theory curves and the empirically found kernel regression errors.

## G TASK SPACE LEARNING

In Sections E and F, we described the kernel’s eigensystem in the raw data space of $Z .$ The same results can be read in task space: averaging the predictor over the in-context inputs and noise, x and η, gives the task the kernel predicts on average for a given latent $\beta .$ Within the eigenframework this average is exact; the approximations below enter only through the large-L forms of the task images.

## G.1 EIGENFUNCTIONS IN TASK SPACE

Write each orthonormal polynomial of Definition $^ { 5 }$ as $\tilde { \phi } _ { p , m , b } ( Z ) = \Phi _ { p , m } \tilde { x } _ { L + 1 , b }$ , where $\Phi _ { p , m }$ depends only on the context, and define its task image

$$
\chi _ { p , m } ( \beta ) \equiv \mathbb { E } _ { \tilde { \mathbf { x } } , \eta } \big [ \Phi _ { p , m } \big | \beta \big ] , \qquad \mathrm { s o ~ t h a t } \qquad \chi _ { p } ( \beta ) = R _ { p } \Gamma _ { p } ^ { - 1 / 2 } \mathbf { q } _ { p } ( \beta ) ,\tag{53}
$$

with $\pmb { q } _ { p }$ as in Theorem 1. By Theorem 1(i), the eigencoefficients are task-space overlaps. By the law of total covariance, the task images are orthonormal up to the within-context noise:

$$
\begin{array} { r } { v _ { p , m , b } = \mathbb { E } _ { \tilde { B } _ { \mathrm { t r } } } \left[ \tilde { \beta } _ { b } \chi _ { p , m } \right] , \qquad \mathbb { E } _ { \tilde { B } _ { \mathrm { t r } } } \left[ \chi _ { p , m } \chi _ { p ^ { \prime } , m ^ { \prime } } \right] = \delta _ { p p ^ { \prime } } \delta _ { m m ^ { \prime } } - \mathbb { E } _ { \tilde { B } _ { \mathrm { t r } } } \left[ \mathrm { C o v } ( \Phi _ { p , m } , \Phi _ { p ^ { \prime } , m ^ { \prime } } | \beta ) \right] _ { : } } \end{array}\tag{54}
$$

At large $L ,$ the task images simplify. By Theorem 1(ii), $\begin{array} { r } { \pmb { q } _ { p } = L ^ { k } \pmb { f } _ { p } ^ { \perp } + O ( L ^ { k - 1 } ) } \end{array}$ and $\Gamma _ { p } =$ $L ^ { 2 k } G _ { v } ^ { \perp } + { \cal O } ( L ^ { 2 k - 1 } )$ , with the task objects $f _ { p }$ and their pool covariance $\dot { G } _ { p } ^ { \perp }$ as defined in Section F.3. On task modes, therefore

$$
\chi _ { p } ( \beta ) \approx R _ { p } ( G _ { p } ^ { \perp } ) ^ { - 1 / 2 } f _ { p } ^ { \perp } ( \beta ) ,\tag{55}
$$

while on noise modes $f _ { p } ^ { \perp } = 0$ and the task images vanish as $O ( L ^ { - 1 / 2 } )$ . At large L, the task images are thus orthonormal polynomials of $\tilde { \beta }$ under the pool measure. At degree $p = 2 k + 1$ , the new objects are $\tilde { \beta } ^ { \otimes n _ { S } } \tau ^ { n _ { A } }$ with $n _ { S } + n _ { A } = k _ {  }$ , of degree k to 2k in $\tilde { \beta }$ . As in data space, each degree is regressed on all lower degrees and then orthonormalized as a block through $G _ { p } ^ { \perp }$

## G.2 EIGENCOEFFICIENTS IN TASK SPACE

Write the predictor as $f ( Z ) = \hat { \tilde { \beta } } ^ { \top } \tilde { \mathbf { x } } _ { L + 1 }$ , where $\hat { \tilde { \beta } }$ is the predicted colored task. Take $\Sigma _ { x } \propto I _ { d } .$ as in all our experiments, so the learnability $\mathcal { L } _ { p , m }$ of a mode does not depend on the query index. Averaging over the context given $\beta$ gives

$$
\mathbb { E } _ { \tilde { \pmb { x } } , \eta } \big [ \hat { \tilde { \beta } } _ { b } \big | \beta \big ] = \sum _ { p , m } \mathcal { L } _ { p , m } v _ { p , m , b } \chi _ { p , m } ( \beta ) .\tag{56}
$$

Here v and $\chi$ must be taken in the kernel’s own basis (Equation (44)), since that is the basis in which learnabilities are assigned. For general $\Sigma _ { x } , { \mathcal { L } }$ also depends on the query index through the eigenvalues of $\Sigma _ { x }$ , with b taken in that eigenbasis. Once a degree is fully learned $( \mathcal { L } _ { p , m } = 1$ for all m), its contribution is basis-independent:

$$
\sum _ { m } v _ { p , m , b } \chi _ { p , m } ( \beta ) = V _ { p , b } ^ { \top } \Gamma _ { p } ^ { - 1 } q _ { p } ( \beta ) \approx L ^ { - k } V _ { p , b } ^ { \top } ( G _ { p } ^ { \bot } ) ^ { - 1 } f _ { p } ^ { \bot } ( \beta ) .\tag{57}
$$

The terms arrive in order of degree. At degree 1, the kernel learns the pool mean $\tilde { \mu } .$ . At degree 3, for $\tilde { \mu } = 0$ and $c = \mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } + 2 \tilde { \Sigma } \tilde { \mu } = 0 .$ , the S modes contribute

$$
\sum _ { a } \frac { L \tilde { s } _ { a } } { ( L + 1 ) \tilde { s } _ { a } + \bar { \tau } } { \pmb u } _ { a } { \pmb u } _ { a } ^ { \top } \tilde { \beta } ,
$$

a shrinkage estimate of the true task. It recovers $\tilde { \beta }$ in directions where $\tilde { s } _ { a } \gg \bar { \tau } / L$ , and not at all in directions the pool does not span $( \tilde { s } _ { a } = 0$ , which is unavoidable when $T \leq d )$ . A nonzero c adds the

A mode, a scalar function of $\tau ( \beta )$ along the fixed direction c. From degree 5 on, the learner adds nonlinear functions of $\tilde { \beta }$ of degree k to $2 k ,$ each with coefficient $O ( L ^ { - 1 } )$ . By Theorem 1(iii), these bring in the pool’s skewness at degree 5, its kurtosis at degree 7, and so on. Table 3 lists the task images and coefficients through degree seven.
<table><tr><td>Degree p</td><td>family</td><td> $\chi _ { p , m } ( \beta )$ </td><td> $v _ { p , m , b }$ </td></tr><tr><td>1</td><td>1</td><td></td><td> $\tilde { \mu } _ { b }$ </td></tr><tr><td>3</td><td>S</td><td> $\pmb { u } _ { a } ^ { \top } \tilde { \beta }$   $\overline { { \sqrt { ( 1 + L ^ { - 1 } ) \tilde { s } _ { a } + \bar { \tau } / L } } }$ </td><td> $\tilde { s } _ { a }$   $\overline { { \sqrt { ( 1 + L ^ { - 1 } ) \tilde { s } _ { a } + \bar { \tau } / L } } }$ </td></tr><tr><td>3</td><td>A</td><td>T-T  $\overline { { \sqrt { \operatorname { V a r } \tau + 2 \mathbb { E } \tau ^ { 2 } / L } } }$ </td><td>Cb  $\overline { { \sqrt { \operatorname { V a r } \tau + 2 \mathbb { E } \tau ^ { 2 } / L } } }$ </td></tr><tr><td>3</td><td>W</td><td>0</td><td> $0 ( { \mathrm { n o i s e ~ m o d e } } )$ </td></tr><tr><td>5</td><td>SS</td><td> $\left( G _ { 5 } ^ { \perp } \right) ^ { - 1 / 2 } \left( \tilde { \beta } ^ { \otimes 2 } \right) ^ { \perp }$ </td><td> $L ^ { - 1 } ( G _ { 5 } ^ { \perp } ) ^ { - 1 / 2 } ( \bar { \tau } \tilde { \Sigma } ^ { - 1 } - I _ { d } ) \tilde { \kappa } _ { 3 }$ </td></tr><tr><td>5</td><td>SA</td><td> $( G _ { 5 } ^ { \perp } ) ^ { - 1 / 2 } ( \tau \tilde { \beta } ) ^ { \perp }$ </td><td> ${ \cal L } ^ { - 1 } ( G _ { 5 } ^ { \bot } ) ^ { - 1 / 2 } \left( - ( 2 \mathrm { T r } \tilde { \Sigma } ^ { 2 } I _ { d } + 4 \tilde { \Sigma } ^ { 2 } ) + \dots \right)$ </td></tr><tr><td>5</td><td>AA</td><td> $\left( G _ { 5 } ^ { \perp } \right) ^ { - 1 / 2 } \left( \tau ^ { 2 } \right) ^ { \perp }$ </td><td> $O ( L ^ { - 1 } )$  odd moments</td></tr><tr><td>5</td><td>W families</td><td> $O ( L ^ { - 1 / 2 } )$ </td><td> $O ( L ^ { - 1 / 2 } )$  (noise modes)</td></tr><tr><td>7</td><td>SSS</td><td> $\left( G _ { 7 } ^ { \perp } \right) ^ { - 1 / 2 } \left( \tilde { \beta } ^ { \otimes 3 } \right) ^ { \perp }$ </td><td> $L ^ { - 1 } ( G _ { 7 } ^ { \perp } ) ^ { - 1 / 2 } \frac { d } { d + 2 } \frac { \epsilon ^ { 2 } } { \tilde { s } } \tilde { \kappa } _ { 4 }$ </td></tr><tr><td>7</td><td>SSA</td><td> $( G _ { 7 } ^ { \perp } ) ^ { - 1 / 2 } ( \tau \tilde { \beta } ^ { \otimes 2 } ) ^ { \perp }$ </td><td>odd moments</td></tr><tr><td>7</td><td>SAA</td><td> $( G _ { 7 } ^ { \perp } ) ^ { - 1 / 2 } \left( \tau ^ { 2 } \tilde { \beta } \right) ^ { \perp }$ </td><td> $\propto L ^ { - 1 } ~ \mathrm { T r } _ { 2 } ~ \tilde { \kappa } _ { 6 }$ </td></tr><tr><td>7</td><td>AAA</td><td> $\left( G _ { 7 } ^ { \perp } \right) ^ { - 1 / 2 } \left( \tau ^ { 3 } \right) ^ { \perp }$ </td><td>odd moments</td></tr><tr><td>7</td><td>W families</td><td> $O ( L ^ { - 1 / 2 } )$ </td><td> $O ( L ^ { - 1 / 2 } )$  (noise modes)</td></tr></table>

Table 3: Task images and eigencoefficients through degree seven, in the symmetric basis $R _ { p } = I $ $( \cdot ) ^ { \perp }$ is the regression on all lower-degree task objects under the pool measure, and $( G _ { p } ^ { \perp } ) ^ { - 1 / 2 }$ is the symmetric inverse square root of the joint degree-p pool covariance. The family column therefore labels blocks of $f _ { p } ^ { \perp }$ , which this rotation mixes unless $\bar { G } _ { p } ^ { \perp }$ is block diagonal. A family with $n _ { S }$ factors of S has a task image of tensor rank $^ { n _ { S } ; }$ its coefficient carries $n _ { S } + 1$ indices, of which $n _ { S }$ contract against the task image and one is the output index b. Degree-3 rows are exact at every L and are written in the eigenbasis of Σ<sup>˜</sup> : the $S$ row when $\tilde { \mu } = 0$ and $\mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } = 0$ , and the A row at first order in $c = \mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } + 2 \tilde { \Sigma } \tilde { \mu }$ . Degree-5 rows are leading order in $1 / L ;$ the SS row assumes $\tilde { \mu } = 0$ and Tr<sub>1</sub> ${ \tilde { \kappa } } _ { 3 } = 0 .$ , and the $_ { S A }$ row shows the Gaussian part for a centered pool. Degree-7 rows are leading order in $1 / L$ for an isotropic centered pool $( \tilde { \Sigma } = \tilde { s } I _ { d } )$ , at first order in the deviation from a Gaussian. χ is quoted at $L \to \infty$ except at degree 3.

## H ORACLE EXPANSIONS

Throughout this appendix we work in raw coordinates, with $\begin{array} { r } { W \equiv \sum _ { i = 1 } ^ { L } \pmb { x } _ { i } \pmb { x } _ { i } ^ { \top } , S \equiv \sum _ { i = 1 } ^ { L } y _ { i } \pmb { x } _ { i } } \end{array}$ $\textstyle A \equiv \sum _ { i = 1 } ^ { L } y _ { i } ^ { 2 }$ , and $\Delta W \equiv W - L \Sigma _ { x }$ . These are the unwhitened versions of the sums in Section F.2. We write $\Sigma _ { \beta }$ and $\mu _ { \beta }$ for the prior’s covariance and mean, and $\hat { \mu } _ { \beta } , \hat { \Sigma } _ { \beta } , \hat { \kappa } _ { \ell }$ for the cumulants of the pool $B _ { \mathrm { t r } }$

Both oracles are Bayes-optimal under the same likelihood, $y _ { i } \mid \beta \sim \mathcal { N } ( \beta ^ { \top } x _ { i } , \epsilon ^ { 2 } )$ ; they differ only in the prior they place on $\beta .$ . The ridge takes the population Gaussian $\mathcal { N } ( \mu _ { \beta } , \Sigma _ { \beta } )$ , and the dMMSE takes the uniform measure on the finite pool $B _ { \mathrm { t r } }$ . Everything in this appendix follows from that single difference.

## H.1 THE GENERALIZING SOLUTION

For $\beta \sim \mathcal { N } ( \mu _ { \beta } , \Sigma _ { \beta } )$ , the Bayes-optimal predictor is $\hat { f } _ { \mathrm { r i d g e } } = \beta _ { \mathrm { r i d g e } } ^ { \top } \pmb { x } _ { L + 1 }$ , where $\beta _ { \mathrm { r i d g e } }$ is ridge regression with matrix parameter $\epsilon ^ { 2 } \Sigma _ { \beta } ^ { - 1 }$ :

$$
\beta _ { \mathrm { r i d g e } } = \left( W + \epsilon ^ { 2 } \Sigma _ { \beta } ^ { - 1 } \right) ^ { - 1 } \left( S + \epsilon ^ { 2 } \Sigma _ { \beta } ^ { - 1 } \pmb { \mu } _ { \beta } \right) .\tag{58}
$$

The main text takes $\mu _ { \beta } = 0 ;$ we keep the general form for Section H.4.

## H.2 THE MEMORIZING SOLUTION

For $\beta$ uniform on $B _ { \mathrm { t r } }$ , the Bayes-optimal predictor is $\hat { f } _ { \mathrm { m e m } } = \hat { \beta } _ { \mathrm { m e m } } ^ { \top } \pmb { x } _ { L + 1 }$ , with the posterior mean over the pool

$$
\hat { \beta } _ { \mathrm { m e m } } = \sum _ { t = 1 } ^ { T } w _ { t } \beta ^ { ( t ) } , \qquad w _ { t } = \frac { e ^ { \ell _ { t } } } { \sum _ { s } e ^ { \ell _ { s } } } , \qquad \ell _ { t } = \frac { 1 } { \epsilon ^ { 2 } } \Big ( \beta ^ { ( t ) \top } S - { \textstyle \frac { 1 } { 2 } } \beta ^ { ( t ) \top } W \beta ^ { ( t ) } \Big ) .\tag{59}
$$

This is Equation $( 3 ) ,$ after expanding $\begin{array} { r } { \sum _ { i } ( y _ { i } - \beta ^ { \top } { \pmb x } _ { i } ) ^ { 2 } = A - 2 \beta ^ { \top } S + \beta ^ { \top } W \beta } \end{array}$ and cancelling $A .$ which is common to every task. Two consequences are used below. First, both oracles depend on the context only through $W$ and $S ;$ the sum $A$ enters the eigencoefficients only through its correlation with functions of $W$ and S. Second, $\begin{array} { r } { \hat { \beta } _ { \mathrm { m e m } } = \epsilon ^ { 2 } \nabla _ { S } \log \sum _ { t } e ^ { \ell _ { t } } } \end{array}$ . A Taylor series of the dMMSE in $S$ therefore generates cumulants of the pool reweighted by $e ^ { - \beta ^ { \top } W \beta / 2 \epsilon ^ { 2 } }$ , which at $W \approx L \Sigma _ { x }$ is strongly tilted, rather than cumulants of the pool itself. The pool’s own cumulants enter only after projection onto the orthonormal basis (Theorem 1).

## H.3 WHY NO FINITE TRUNCATION OF THE ORACLE EXPANSIONS YIELDS THE EIGENCOEFFICIENTS

It is tempting to skip choosing the eigenfunction basis and instead Taylor-expand the oracles directly. This does not work: reasonable expansions identify the modes an oracle occupies, but truncating at any finite order misstates their coefficients, by arbitrary factors and in some cases all the way to zero. We record two examples as a warning.

Ridge. With $r _ { j } \equiv y _ { j } - \mu _ { \beta } ^ { \top } x _ { j }$ and $P \equiv \left( L \Sigma _ { x } + \epsilon ^ { 2 } \Sigma _ { \beta } ^ { - 1 } \right) ^ { - 1 }$ , so that $W + \epsilon ^ { 2 } \Sigma _ { \beta } ^ { - 1 } = P ^ { - 1 } + \Delta W$ the Neumann series gives

$$
\begin{array} { r l } & { \hat { f } _ { \mathrm { r i d g e } } = \mu _ { \beta } ^ { \top } { \boldsymbol x } _ { L + 1 } + { \boldsymbol x } _ { L + 1 } ^ { \top } \left( { \boldsymbol W } + \epsilon ^ { 2 } \boldsymbol \Sigma _ { \beta } ^ { - 1 } \right) ^ { - 1 } \displaystyle \sum _ { j = 1 } ^ { L } r _ { j } { \boldsymbol x } _ { j } } \\ & { \quad \quad \quad = \mu _ { \beta } ^ { \top } { \boldsymbol x } _ { L + 1 } + \displaystyle \sum _ { m \geq 0 } ( - 1 ) ^ { m } { \boldsymbol x } _ { L + 1 } ^ { \top } P ( \Delta { \boldsymbol W } P ) ^ { m } \displaystyle \sum _ { j = 1 } ^ { L } r _ { j } { \boldsymbol x } _ { j } } \\ & { \quad \quad = \mu _ { \beta } ^ { \top } { \boldsymbol x } _ { L + 1 } + \displaystyle \sum _ { j } r _ { j } { \boldsymbol x } _ { j } ^ { \top } P { \boldsymbol x } _ { L + 1 } } \\ & { \quad \quad \quad - \left( \displaystyle \sum _ { j , j } r _ { j } ( { \boldsymbol x } _ { j } ^ { \top } P { \boldsymbol x } _ { i } ) ( { \boldsymbol x } _ { i } ^ { \top } P { \boldsymbol x } _ { L + 1 } ) - L \displaystyle \sum _ { j } r _ { j } { \boldsymbol x } _ { j } ^ { \top } P \boldsymbol \Sigma _ { \alpha } P { \boldsymbol x } _ { L + 1 } \right) + \ldots , } \end{array}\tag{60}
$$

which converges since $\| P \Delta W \| = O ( L ^ { - 1 / 2 } )$ . The first line uses $\begin{array} { r } { S - W \mu _ { \beta } = \sum _ { j } r _ { j } \pmb { x } _ { j } } \end{array}$

Take $\Sigma _ { x } \propto I _ { d }$ and $\mu _ { \beta } = 0$ , and evaluate under the prior, so that $( \tilde { s } _ { a } , { \boldsymbol { u } } _ { a } )$ are the eigenpairs of $\Sigma _ { x } ^ { 1 / 2 } \Sigma _ { \beta } \Sigma _ { x } ^ { 1 / 2 }$ and $\bar { \tau } = \mathrm { T r } ( \Sigma _ { \pmb { x } } \Sigma _ { \beta } ) + \epsilon ^ { 2 }$ . Truncating (60) at data degree three keeps $\begin{array} { r } { \sum _ { j } r _ { j } \pmb { x } _ { j } ^ { \top } P \pmb { x } _ { L + 1 } } \end{array}$ The coefficient it assigns to the normalized degree-3 mode is the raw coefficient times the mode’s norm,

$$
v _ { a } ^ { \mathrm { t r u n c } } = \frac { \widetilde s _ { a } } { L \widetilde s _ { a } + \epsilon ^ { 2 } } \sqrt { L ( L + 1 ) \widetilde s _ { a } + L \widetilde \tau } ,\tag{61}
$$

whereas the projection gives the overlap divided by the norm (Table 2),

$$
v _ { a } ^ { \mathrm { p r o j } } = \frac { L \tilde { s } _ { a } } { \sqrt { L ( L + 1 ) \tilde { s } _ { a } + L \bar { \tau } } } = \frac { \tilde { s } _ { a } } { \sqrt { ( 1 + L ^ { - 1 } ) \tilde { s } _ { a } + \bar { \tau } / L } } .\tag{62}
$$

Their ratio is

$$
\frac { v _ { a } ^ { \mathrm { t r u n c } } } { v _ { a } ^ { \mathrm { p r o j } } } = \frac { ( L + 1 ) \widetilde { s } _ { a } + \bar { \tau } } { L \widetilde { s } _ { a } + \epsilon ^ { 2 } } = 1 + \frac { \widetilde { s } _ { a } + \bar { \tau } - \epsilon ^ { 2 } } { L \widetilde { s } _ { a } + \epsilon ^ { 2 } } > 1 ,\tag{63}
$$

since $\bar { \tau } \geq \epsilon ^ { 2 }$ . The truncation therefore always overestimates, for every mode and every parameter setting, and approaches the projection only as $O ( L ^ { - 1 } )$ . For example, at $d = 2 , L = \mathsf { \bar { 6 } } , \mathsf { \bar { \epsilon } } ^ { 2 } = 0 . 5$ $\Sigma _ { x } = \bar { I _ { d } } / 2$ and $\Sigma _ { \beta } = I _ { d }$ , one has $\tilde { s } _ { a } = 0 . 5$ and $\bar { \tau } = 1 . 5$ , giving a ratio of $5 / 3 . 5 \approx 1 . 4 3$

The two operations differ because the eigencoefficients are projections of the oracle onto an orthonormal basis, $v _ { i } = \mathbb { E } _ { Z } [ y ^ { \star } \phi _ { i } ]$ ], while an expansion truncates the oracle itself. They agree only when the discarded terms are orthogonal to the retained modes, and regular polynomials of different degree are not. Here, every higher-order term of (60) generally has a nonzero projection onto the degree-3 mode, so the correct coefficient requires resumming the whole series.

dMMSE. The $A : = \textstyle \sum _ { i } y _ { i } ^ { 2 }$ modes give a more extreme case. By Section H.2, the dMMSE contains no quadratic functions of $y _ { i } ^ { 2 }$ at all (equivalent factors appear in both the numerator and denominator, cancelling), so any expansion of it assigns $A { \bf { \sigma } } x _ { L + 1 }$ a coefficient of zero. The projection is instead $c _ { b } / \sqrt { \mathrm { V a r } \tau + 2 \mathbb { E } \tau ^ { 2 } / L }$ with $c = \mathrm { T r } _ { 1 } \tilde { \kappa } _ { 3 } + 2 \tilde { \Sigma } \tilde { \mu }$ (Table 2), which is nonzero whenever $c \neq 0$ , since A correlates with functions of $W$ and S. For a centered Gaussian prior $c = 0 .$ , so this mode is specific to the memorizing solution.

## H.4 GENERALIZING VS MEMORIZING SOLUTION

The dMMSE is a functional of the pool measure, and each of its eigencoefficients depends on finitely many pool moments, which are polynomials in the cumulants. Setting $\hat { \kappa } _ { \ell \geq 3 } = 0$ throughout is therefore the same as replacing the pool by the Gaussian with the same mean and covariance.

Proposition 2 (The generalizing solution is a cumulant truncation of the memorizing one). Replacing the pool $B _ { \mathrm { t r } }$ in Equation (59) by $\mathcal { N } ( \hat { \mu } _ { \beta } , \hat { \Sigma } _ { \beta } )$ gives

$$
\left( { \cal W } + \epsilon ^ { 2 } \hat { \Sigma } _ { \beta } ^ { - 1 } \right) ^ { - 1 } \left( S + \epsilon ^ { 2 } \hat { \Sigma } _ { \beta } ^ { - 1 } \hat { \mu } _ { \beta } \right) ,\tag{64}
$$

which is the ridge solution Equation (58) with the prior’s mean and covariance replaced by the pool’s. When $\hat { \Sigma } _ { \beta }$ is singular $( T \leq \bar { d } ) , \hat { \Sigma } _ { \beta } ^ { - 1 }$ is infinite off its range, and the predicted task is pinned to $\hat { \mu } _ { \beta }$ in directions the pool does not span.

Proof. Under a Gaussian prior and the linear-Gaussian likelihood, the posterior over $\beta$ is Gaussian, and its mean is the displayed expression. □

The two oracles therefore differ in exactly two ways. The higher cumulants $\hat { \kappa } \ell { > } 3$ appear only in the dMMSE. The first two cumulants appear in both, as $( \hat { \mu } _ { \beta } , \hat { \Sigma } _ { \beta } )$ versus $( \mu _ { \beta } , \Sigma _ { \beta } )$ . Their difference shrinks as $T ^ { - 1 / 2 }$ entrywise, but is severe for $T \leq d ,$ where the pool does not span task space.