# Noise-regularized training of quantum circuit Born machines under a finite shot budget

Hoang M. Ngo1, Quan Nguyen1, Jung Taek Seo2, Wanli Xing3, My T. Thail\*

1\*Department of Computer and Information Science and Engineering, University of Florida, 1889 Museum Road, Gainesville, 32611, Florida, USA.

2Department of Smart Security, Gachon University, 1342 Seongnam-daero, Sujeong-gu, Seongnam-si, 13120, Gyeonggi-do, Republic of Korea.

3Department of Teaching and Learning, School of Education and Human Development, University of Miami, 5202 University Drive, Coral Gables, 33124, Florida, USA.

\*Corresponding author(s). E-mail(s): mythai@cise.ufl.edu; Contributing authors: hoang.ngo@ufl.edu; quan.nguyen1@ufl.edu; seojt@gachon.ac.kr; wxx151@miami.edu;

## Abstract

Quantum circuit Born machines (QCBMs) can learn distributions that are hard to sample from classically, but their value depends on generalizing from finite training data. Here, we study how hardware noise affects this generalization. We show that, for any noise process, the generalization gap of a QCBM trained with the maximum mean discrepancy is bounded by the complexity of the distributions the noisy model can reach. Scrambled local Pauli noise contracts this complexity but biases the model towards the uniform distribution, so residual noise acts as a regularizer whose optimal strength can be intermediate. We propose GENIE, a framework that realizes a chosen noise level with capped error mitigation and selects it during training from held-out data within a fixed execution budget, with provably bounded regret. In simulations on synthetic and census data, GENIE stays close to the best of four noise-control strategies, whereas every fixed noise level falls short in many settings.

Keywords: quantum computing, generative model, error mitigation

## 1 Introduction

Generative modelling, the task of learning an unknown probability distribution from examples and producing new samples from it, is one of the most natural tasks for a quantum computer. The reason is measuring a quantum state already yields a sample from a probability distribution [1]. Parameterized quantum circuits can prepare output distributions that are believed to be hard to sample from classically [2-4], and quantum generative models admit provable expressivity and learnability separations from classical models [5-7]. The most direct realization of this idea is the quantum circuit Born machine (QCBM) [8, 9], whose model distribution is given by the Born rule, $q _ { \pmb { \theta } } ( \pmb { x } ) = | \langle \pmb { x } | U ( \pmb { \theta } ) | \mathbf { 0 } \rangle | ^ { 2 }$ , for a parameterized circuit $U ( \pmb \theta )$ . Every circuit execution thus returns an exact sample, and because only samples rather than likelihoods are accessible, QCBMs are trained with sample-based losses such as the maximum mean discrepancy (MMD) [9-11]. QCBMs have been demonstrated on trapped-ion hardware [12], high-energy physics [13], combinatorial optimization [14] and drug discovery [15].

To be practically useful, a Born machine must generalize beyond its training set rather than reproduce it. For a generative model, generalization is defined as the closeness of its output distribution to the unknown target distribution $p ,$ rather than to the empirical distribution of the m training samples. Operationally, it is the ability to produce valid samples that never appeared in training [16]. For a sufficiently expressive model, driving the training loss to zero instead reproduces the empirical distribution, which places all of its mass on the observed bitstrings and none on valid but unseen ones [11]. This is the generative form of overfitting, and Born machines are particularly exposed to it. Specifically, their output space contains $2 ^ { n }$ bitstrings while practical training sets are only polynomially large, so the empirical distribution leaves almost every valid bitstring unobserved. Generalization bounds and regularizers developed for supervised quantum learning [17-21| do not transfer directly, because they act on a predictor that maps inputs to labels, whereas a Born machine has neither and outputs only the distribution it samples. Recent studies have begun to examine generalization in Born machines by bounding the excess MMD risk of an idealized empirical minimizer [22], benchmarking the validity, novelty and coverage of generated samples [16, 23], and showing that a converged MMD is an unreliable indicator of generalization [24]. However, these works evaluate generalization rather than control it, and they offer no mechanism for regularizing the finite-shot, noisy training that Born machines undergo on real hardware.

In classical learning, injecting stochasticity into training is a common way to improve generalization, whether through input noise [25], dropout [26] or stochastic gradient descent [27]. Quantum hardware provides such stochasticity intrinsically from gate errors and decoherence. In supervised quantum learning, analyses [20, 28, 29] show that it can act as an implicit regularizer [30], with a finite optimal noise level that can be targeted by tuning the noise [31-33], although excessive noise destroys trainability [34]. For Born machines, exploiting this noise is more challenging. The noisy output distribution is itself the model, so the noise that suppresses overfitting also distorts what is learned. Moreover, the noise level is fixed by the hardware, and lowering it requires error mitigation whose sampling cost grows exponentially with the amount of noise removed [35-38]. To build a noise-based regularizer for QCBMs, we must therefore answer two questions: (i) how does residual hardware noise affect the generalization of an MMD-trained Born machine? (ii) how can the amount of residual noise be selected and realized during training within a finite budget of circuit executions?

In this work, we answer these questions and develop GENIE (generalization with inherent noise), a framework that treats residual hardware noise as a tunable regularizer for MMD-trained Born machines (Fig. 1). Our contributions are as follows.

• Generalization theory. We derive a bound on the empirical-to-population MMD gap of QCBMs that holds for any noise process and training algorithm. In addition, we specialize it to scrambled local Pauli noise, where residual noise contracts the gap by the retained-signal factor 1 -γ. The bound makes the regularization effect of residual noise explicit, while showing that the same noise biases the model towards the uniform distribution, so the amount of noise to retain must be chosen.

• Adaptive noise regularization. We propose GENIE, a framework that adaptively selects the residual noise for MMD-trained QCBMs under a fixed circuitexecution budget. In general, first it realizes each level with capped probabilistic error reduction, which removes just enough hardware noise while keeping the remaining noise close to white noise. Then, it periodically spends a small portion of the budget to probe candidate noise levels on held-out data, and then selects the level for the next training block based on validation performance and the budget remaining for training. We establish a cumulative selection regret bound that quantifies how limited validation data, measurement precision, and the frequency of noise adjustments affect the controller's decisions along the training trajectory.

• Empirical validation. We compare GENIE with training at three fixed noise levels, namely no, half and full mitigation, under matched budgets of circuit executions. In simulations on three synthetic datasets with 12 qubits, it is within 1% of the best of the four methods in 64% of 144 conditions and has a median regret of 0.3%, whereas the fixed levels have median regrets between 8.4% and 22.7%. On the Adult census dataset with 10 qubits at strong noise, it gives the most accurate population shares, income rates within groups and income predictions of the four methods.

## 2 Results

## 2.1 Preliminaries

This subsection introduces the model, the noise and the loss, and the two quantities in which all of our results are stated, which are, the witness function and the Rademacher complexity.

## Born machines under hardware noise.

Let $\mathcal { X } = \{ 0 , 1 \} ^ { n }$ be the set of n-bit strings. A Born machine applies a parameterized circuit $U ( \theta ) , \theta \in \Theta$ , to a fixed input state ρo, typically $\left| 0 \right. ^ { \otimes n }$ , and measures all qubits

![](images/c1e3a8bdd1d8fceb60c57233b6f1e85312a769ed18d7051fa2b616489a4f4c17.jpg)  
Fig. 1: Overview of GENIE. The circuit-execution budget $B$ is divided between training, $( 1 - \alpha ) B ,$ and probing, $\alpha B .$ Circuit parameters $\theta$ are updated using the MMD loss estimated from training data and outcomes weighted by probabilistic error reduction (PER) $( y , w )$ , where $y$ is a measured bitstring and $w$ is its signed PER weight. Every K steps, θ is held fixed while candidate residual-noise profiles from capped PER allocation are evaluated on held-out validation data. The next profile is selected using validation fit, estimation uncertainty, and sampling overhead. The selected profile specifies target local Pauli-noise rates $r _ { k }$ throughout the circuit, with $\gamma ( \mathbf { r } )$ summarizing the effective global noise under the assumed model. GENIE returns the trained parameters $\hat { \theta }$ and the selected residual-noise profile $\hat { \mathbf { r } } .$

in the computational basis. By the Born rule, its ideal output distribution is

$$
\begin{array} { r } { q _ { \theta } ^ { 0 } ( x ) = \left. x \right| U ( \theta ) \rho _ { 0 } U ( \theta ) ^ { \dagger } \left| x \right. , \qquad x \in \mathcal { X } . } \end{array}\tag{1}
$$

On hardware, the circuit runs under a noise process ${ \mathcal { N } } .$ which may include gate errors, correlated errors and readout errors. If $\rho _ { \theta } ^ { N }$ denotes the resulting output state, each

circuit execution returns one sample from the noisy distribution

$$
q _ { \theta } ^ { N } ( x ) = \langle x \vert \rho _ { \theta } ^ { N } \vert x \rangle .\tag{2}
$$

The probabilities $q _ { \theta } ^ { N } ( x )$ are never observed, only samples from them. Therefore, the Born machine is an implicit generative model, and the distribution that is trained and evaluated is the noisy $q _ { \theta } ^ { \mathcal { N } }$ , not the ideal $q _ { \theta } ^ { 0 }$

To model the noise, we use the standard model of local gate noise. Each of the s two-qubit gates in the circuit is followed by an independent stochastic Pauli channel on each of the two qubits it acts on. Indexing these $L = 2 s$ noisy locations by $k ,$ the channel at location k is

$$
\mathcal { E } _ { k } ( \rho ) = ( 1 - \epsilon _ { k } ) \rho + \epsilon _ { k } \sum _ { P \in \{ X , Y , Z \} } \pi _ { k } ( P ) P \rho P , \qquad \sum _ { P } \pi _ { k } ( P ) = 1 ,\tag{3}
$$

with probability $\epsilon _ { k }$ an error occurs, and the error is a Pauli $P$ drawn from the distribution $\pi _ { k }$ . Randomized compiling brings general gate noise into this form, after which the error rates can be measured [39, 40]. We call $\epsilon = ( \epsilon _ { 1 } , \dots , \epsilon _ { L } )$ the hardware noise profile. It is fixed by the device, and later we reduce it selectively to a residual profile r with $0 \le r _ { k } \le \epsilon _ { k }$

## Learning problem and MMD loss.

The target distribution p on $\mathcal { X }$ is unknown and is accessible only through a training sample $\boldsymbol { S } = ( x _ { 1 } , \dots , x _ { m } )$ of m independent draws from p. The goal is a model whose distribution is close to p itself, not merely to the m observed strings. Because only samples of the model are available, we measure closeness with a kernel $\kappa : \mathcal { X } \times \mathcal { X } $ [0, 1] between bit strings that is positive semidefinite and satisfies $\kappa ( x , x ) = 1$ . A standard choice for Born machines is the Gaussian kernel on the Hamming distance $\begin{array} { r } { d _ { H } , \kappa ( x , y ) = \frac { 1 } { c } \sum _ { i = 1 } ^ { c } \exp [ - d _ { H } ( x , y ) / ( 2 \tau _ { i } ^ { 2 } ) ] } \end{array}$ , with bandwidths $\tau _ { 1 } , \ldots , \tau _ { c } \ [ 9 , 1 1 ]$ . The kernel defines a reproducing kernel Hilbert space (RKHS) H of functions on $\mathcal { X }$ in which $\kappa ( \boldsymbol { x } , \cdot )$ serves as a feature vector of the string x, with $\langle \kappa ( \boldsymbol { x } , \cdot ) , \kappa ( \boldsymbol { y } , \cdot ) \rangle _ { \mathcal { H } } = \kappa ( \boldsymbol { x } , \boldsymbol { y } )$ Averaging these features over a distribution ν gives its kernel mean embedding, and the distance between two embeddings is the maximum mean discrepancy (MMD) [10]:

$$
\mu _ { \nu } : = \mathbb { E } _ { x \sim \nu } \left[ \kappa ( x , \cdot ) \right] \in \mathcal { H } , \qquad \mathrm { M M D } _ { \kappa } ( \nu , \nu ^ { \prime } ) : = \| \mu _ { \nu } - \mu _ { \nu ^ { \prime } } \| _ { \mathcal { H } } .\tag{4}
$$

The population risk of the noisy model is its squared MMD to the target,

$$
\begin{array} { r } { R _ { N } ( \theta ) : = \mathrm { M M D } _ { \kappa } ^ { 2 } \big ( q _ { \theta } ^ { N } , p \big ) = \underbrace { \mathbb { E } _ { y , y ^ { \prime } \sim q _ { \theta } ^ { N } } \big [ \kappa ( y , y ^ { \prime } ) \big ] } _ { \mathrm { m o d e l - m o d e l } } + \underbrace { \mathbb { E } _ { x , x ^ { \prime } \sim p } \big [ \kappa ( x , x ^ { \prime } ) \big ] } _ { \mathrm { t a r g e t - t a r g e t } } - 2 \underbrace { \mathbb { E } _ { x \sim p , y \sim q _ { \theta } ^ { N } } \big [ \kappa ( x , y ) \big ] } _ { \mathrm { c o u p l i n g } } , } \end{array}\tag{5}
$$

where all draws are independent. The coupling term rewards model samples that resemble the data, the model-model term penalizes a model whose samples all resemble each other, and the target-target term is a constant.

## Training, evaluation and the generalization gap.

A learning algorithm $\mathcal { A } _ { \mathcal { N } }$ AN maps the training sample to parameters $\widehat { \theta } _ { \mathcal { N } } = \mathcal { A } _ { \mathcal { N } } ( \mathcal { S } )$ . It may be randomized, for example through the initialization, stochastic optimization and the finite number of shots used in training. Because $p$ is unknown, the risk of the learned model can only be estimated. To do so, we draw N fresh samples $\mathcal { V } = ( y _ { 1 } , \dotsc , y _ { N } )$ from the learned circuit, independent of S and of the training randomness given ${ \hat { \theta } } _ { \mathcal { N } }$ and replace each expectation in equation (5) by an average over distinct pairs:

$$
\widehat { R } _ { \mathcal { S } , \mathcal { S } } ^ { N } ( \theta ) : = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq i ^ { \prime } } \kappa ( x _ { i } , x _ { i ^ { \prime } } ) + \frac { 1 } { N ( N - 1 ) } \sum _ { j \neq j ^ { \prime } } \kappa ( y _ { j } , y _ { j ^ { \prime } } ) - \frac { 2 } { m N } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { N } \kappa ( x _ { i } , y _ { j } ) ,\tag{6}
$$

$$
m , N \geq 2 .
$$

For a fixed $\theta ,$ this U-statistic is an unbiased estimate of $R _ { \mathcal { N } } ( \theta )$ . For the learned parameters it is not, because $\hat { \theta } _ { \mathcal { N } }$ was chosen using the same training strings that appear in the estimate. The generalization gap measures the resulting optimism,

$$
\Delta _ { \mathcal { N } } ( m ) : = \mathbb { E } _ { \mathcal { S } , A _ { \mathcal { N } } , \mathcal { V } } \Big [ R _ { \mathcal { N } } ( \widehat { \theta } _ { \mathcal { N } } ) - \widehat { R } _ { \mathcal { S } , \mathcal { V } } ^ { \mathcal { N } } ( \widehat { \theta } _ { \mathcal { N } } ) \Big ] .\tag{7}
$$

Therefore, a low empirical risk certifies a low population risk only if the gap is small. For example, a model that simply reproduces the empirical distribution $\hat { p } _ { m }$ of the training strings has an empirical risk of at most zero on average over $\mathcal { V }$ , although its population risk $\mathrm { M M D } _ { \kappa } ^ { 2 } ( \hat { p } _ { m } , p )$ is positive whenever $\hat { p } _ { m } \neq p .$

## Witness functions.

Only the coupling term in equation (5) involves both the model and the data. It is described by the witness function of a generator,

$$
\begin{array} { r } { h _ { \theta } ^ { \mathcal { N } } ( x ) : = \Bigl \langle \mu _ { q _ { \theta } ^ { N } } , \kappa ( x , \cdot ) \Bigr \rangle _ { \mathcal { H } } = \mathbb { E } _ { y \sim q _ { \theta } ^ { N } } \bigl [ \kappa ( x , y ) \bigr ] \in [ 0 , 1 ] , } \end{array}\tag{8}
$$

which is large at strings x that resemble what the model generates and small elsewhere. Averaging the witness over the target recovers the coupling term. For a function $f$ on $x ,$ we write

$$
P _ { m } f : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } f ( x _ { i } ) , \qquad P f : = \mathbb { E } _ { x \sim p } \big [ f ( x ) \big ]\tag{9}
$$

for its average over the training sample and over the target. In this language, a learned model overfits when its witness is larger on the training strings than on typical target strings, $( P _ { m } - P ) h _ { \hat { \theta } _ { N } } ^ { \mathcal { N } } > 0$ . Adding a constant to a function does not change this difference,

$$
( P _ { m } - P ) ( f + c ) = ( P _ { m } - P ) f , \qquad c \in \mathbb { R } ,\tag{10}
$$

so constant parts of a witness never contribute to overfitting. The witness class

$$
\begin{array} { r } { \mathcal { W } ^ { \mathcal { N } } : = \big \{ h _ { \theta } ^ { \mathcal { N } } : \theta \in \Theta \big \} , \qquad \mathcal { W } _ { \pm } ^ { \mathcal { N } } : = \mathcal { W } ^ { \mathcal { N } } \cup \big ( - \mathcal { W } ^ { \mathcal { N } } \big ) , } \end{array}\tag{11}
$$

contains one witness for each distribution that the noisy device can reach. It depends on the noise and on which distributions are reachable, but not on how many parameters are used to reach them.

## Rademacher complexity.

How much a learned witness can overfit depends on how rich the witness class is. We measure richness with the Rademacher complexity [41],

$$
\Re ( \mathcal { F } ) : = \mathbb { E } _ { \mathcal { S } \sim p ^ { \otimes m } } \mathbb { E } _ { \sigma } \biggl [ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sigma _ { i } f ( x _ { i } ) \biggr ] ,\tag{12}
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { m }$ are independent uniform random signs. Each draw of the signs splits the training strings at random into two groups, and the supremum asks how well some function in $\mathcal { F }$ can be large on one group and small on the other. A rich class can follow any such random split, and hence the accidental features of a particular sample while a poor class cannot. Every witness has RKHS norm at most one, which implies $\Re ( \mathcal { W } _ { \pm } ^ { \mathcal { N } } ) \leq 1 / \sqrt { m }$ for any noise process. Our results show how noise lowers the constant in front of $1 / \sqrt { m }$

## 2.2 Residual noise contracts the generalization gap

In this subsection, we bound the generalization gap, first for an arbitrary noise process and then for local Pauli noise, where the effect of noise becomes explicit. Throughout we make one assumption on the kernel, which exposes the structure on which noise acts.

Assumption 1 (Translation-invariant kernel). The positive-semidefinite kernel $\kappa :$ ${ \mathcal { X } } \times { \mathcal { X } }  [ 0 , 1 ]$ satisfies

$$
\kappa ( x , y ) = \phi ( x \oplus y ) , \qquad \phi ( \mathbf { 0 } ) = 1 ,\tag{13}
$$

for some $\phi : \mathcal { X } \to [ 0 , 1 ]$

This assumption covers the Hamming-distance kernels commonly used for MMD training of Born machines, including the Gaussian kernels above [4, 9]. To bound the gap, note that the target-target term in equation (5) does not depend on $\theta$ and the model-model term is estimated from samples drawn after training, so only the coupling term contributes $( \mathrm { i . e . , ~ } \Delta _ { \cal N } ( m ) = 2 \mathbb { E } [ ( \tilde { P } _ { m } - P ) h _ { \hat { \theta } _ { \cal N } } ^ { \cal N } ] )$ , which does not depend on N (Lemma 1, Methods). Because the learned witness lies in $\mathcal { W } ^ { N }$ , its deviation is at most the largest deviation over the class, which symmetrization bounds by the Rademacher complexity. This gives our first result.

Theorem 1 (Noise-agnostic generalization bound). Let $m \ \geq \ 2$ and $\mathcal { A } _ { \mathcal { N } }$ be any possibly randomized learning algorithm with the noise model ${ \mathcal { N } } ,$ we have

$$
| \Delta _ { \mathcal { N } } ( m ) | \leq 4 \Re \left( \mathcal { W } _ { \pm } ^ { \mathcal { N } } \right) .\tag{14}
$$

Theorem 1 separates generalization from the details of the training algorithm and of the noise model. It holds for any noise process, including correlated and readout errors, and all dependence on the noise is captured by the reachable witness class. Therefore, noise improves the bound only when it reduces the complexity of that class. In the rest of this subsection, we quantify how much local Pauli noise reduces it.

## Scrambled local Pauli noise.

We now make this reduction explicit for local Pauli noise, a standard model of gate errors into which randomized compiling converts general gate noise [39]. Because noise acts on the output distribution, we first describe how a witness depends on that distribution. Bit strings form a group under bitwise addition, and the corresponding Fourier basis consists of the parities $\chi _ { A } ( x ) : = ( - 1 ) ^ { \sum _ { j \in A } x _ { j } }$ of subsets $A \subseteq [ n ]$ of qubits. We write

$$
\widehat { \phi } ( A ) : = 2 ^ { - n } \sum _ { t \in \mathcal { X } } \phi ( t ) \chi _ { A } ( t ) , \qquad \widehat { q } ( A ) : = \mathbb { E } _ { x \sim q } \bigl [ \chi _ { A } ( x ) \bigr ]\tag{15}
$$

for the Fourier coefficients of the kernel and of a distribution $q .$ The coefficient ${ \widehat { q } } ( A )$ is the parity moment of q on the qubits in $A ,$ that is, the expectation value of the Pauli string $Z _ { A } = \otimes _ { j \in A } Z _ { j } ,$ and $\widehat { \phi } ( A ) \geq 0$ is the weight that the kernel places on it. Under Assumption 1, a noiseless witness is then a kernel-weighted sum of these moments,

$$
h _ { \theta } ^ { 0 } ( x ) = \sum _ { A \subseteq [ n ] } { \widehat { \phi } } ( A ) { \widehat { q _ { \theta } ^ { 0 } } } ( A ) \chi _ { A } ( x ) .\tag{16}
$$

The term with $A = \emptyset$ equals the same constant $\widehat { \phi } ( \emptyset )$ for every generator, so by equation (10) it cannot cause overfitting. Therefore, overfitting can arise only from the nonconstant moments, and these are exactly what noise erodes.

Consider a circuit of s two-qubit gates in which every gate is followed by local stochastic Pauli errors of rate € on the qubits it acts on. Although each error is local, a sufficiently scrambling circuit spreads it over the entire output distribution. Dalzell et al. [42] showed that, for random circuits under weak local noise, the output distribution is close to a mixture of the ideal and uniform distributions, with effective noise rate $\gamma = 1 - \exp ( - 2 s \epsilon \pm \mathcal { O } ( s \epsilon ^ { 2 } ) )$ and an approximation error of order $( 1 - \gamma ) \epsilon \sqrt { s }$ Such white-noise models are also commonly used to describe the accumulated noise of hardware [43, 44]. We adopt this description for the circuits we train and measure its error in the geometry of the loss.

Assumption 2 (Scrambled local Pauli noise). Suppose that each circuit contains s two-qubit gates, each followed by local Pauli errors of rate € on both of its qubits. The circuit family scrambles these errors sufficiently that the resulting output distribution is uniformly approximated by an effective white-noise model. Specifically, there exist an effective noise rate $\gamma \in [ 0 , 1 ]$ and a constant $c _ { \mathrm { s c r } } > 0$ such that

$$
q _ { \theta } ^ { \gamma } ( x ) : = ( 1 - \gamma ) q _ { \theta } ^ { 0 } ( x ) + \gamma u ( x ) , \qquad u ( x ) = 2 ^ { - n } ,\tag{17}
$$

$$
\operatorname* { s u p } _ { \theta \in \Theta } \mathrm { M M D } _ { \kappa } \left( q _ { \theta } ^ { N } , q _ { \theta } ^ { \gamma } \right) \leq c _ { \mathrm { s c r } } ( 1 - \gamma ) \epsilon \sqrt { s } = : \varepsilon _ { \mathrm { P a u l i } } .\tag{18}
$$

Since trained circuits are not random, this is a modelling assumption about the circuit family. Because the uniform distribution has no nonconstant parity moments, white noise scales every nonconstant moment, and hence the part of every witness that can overfit, by $1 - \gamma$ . The part of the physical noise that is not white changes each witness by at most $\varepsilon _ { \mathrm { P a u l i } }$ in RKHS norm (Methods). Combining these two facts with Theorem 1 gives our main generalization result.

Theorem 2 (Generalization under scrambled local Pauli noise). Suppose that Assumptions 1 and 2 hold and that, for some $C \geq 1$

$$
\operatorname* { s u p } _ { \theta \in \Theta } 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } \leq C .\tag{19}
$$

Let m, $N \geq 2$ . Then the expected generalization gap satisfies

$$
| \Delta _ { \mathcal { N } } ( m ) | \leq 4 ( 1 - \gamma ) \frac { \Lambda _ { C } } { \sqrt { m } } + \frac { 2 \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } ,\tag{20}
$$

where $\begin{array} { r } { \Lambda _ { C } : = \operatorname* { m i n } \bigr \{ \sqrt { ( C - 1 ) \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) ^ { 2 } } , 1 - \widehat { \phi } ( \emptyset ) \bigr \} . } \end{array}$

Theorem 2 makes the effect of noise explicit. The constant $\Lambda _ { C }$ is the complexity of the noiseless witness class. In its first branch, $C - 1$ bounds the total squared magnitude of the nonconstant parity moments that the circuits can reach, which vanishes for the uniform distribution and grows as the outputs concentrate, and $\textstyle \sum _ { A \neq \emptyset } { \widehat { \phi } } ( A ) ^ { 2 }$ measures how strongly the loss weighs those moments. Because $C$ can be as large as $2 ^ { n }$ , for example when the circuit can output a single bit string, the second branch caps the complexity of every circuit family at a constant below one. Neither branch depends on the number of parameters, unlike supervised quantum learning bounds that grow with the number of trainable gates [19]. White noise contracts this complexity by the retained-signal factor $1 - \gamma$ . Specifically, uniform samples carry no information about which strings appeared in training, so retained noise dilutes the part of the model that can memorize them, as if a noiseless model were trained on $m / ( \bar { 1 } - \gamma ) ^ { 2 }$ samples. Noise that is not white enters only through $\varepsilon _ { \mathrm { P a u l i } }$ , which also decays as $\dot { m } ^ { - 1 / 2 }$ . Therefore, noise helps generalization to the extent that it acts as white noise, and this holds not only on average but for a single training run.

Corollary 1 (High-probability generalization bound). Under the conditions $o f$ Theorem 2, for every $\beta \in ( 0 , 1 )$ , with probability at least $1 - \beta$ over $s , A _ { \mathcal { N } } ,$ and ,

$$
\begin{array} { r l } { \displaystyle \left. R _ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) - \widehat { R } _ { \mathcal { S } , \mathcal { Y } } ^ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) \right. \leq 4 ( 1 - \gamma ) \frac { \Lambda _ { C } } { \sqrt { m } } + 2 ( 1 - \gamma ) \sqrt { \frac { 2 \log ( 4 / \beta ) } { m } } } & { } \\ { + \displaystyle \frac { 2 \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } \left( 1 + \sqrt { 2 \log ( 4 / \beta ) } \right) + \sqrt { \frac { \log ( 8 / \beta ) } { 2 \lfloor m / 2 \rfloor } } + \sqrt { \frac { 8 \log ( 8 / \beta ) } { N } } . } & { } \end{array}\tag{21}
$$

Corollary 1 adds two observations to Theorem 2. First, the first three terms all carry the factor $1 - \gamma$ , the third through $\varepsilon _ { \mathrm { P a u l i } }$ , so noise reduces not only the expected gap but also its fluctuation across draws of the training sample, and a single-run guarantee costs only a logarithmic factor in $1 / \beta .$ Second, the last two terms do not depend on γ. They come from estimating the target-target term with m samples and the model-dependent terms with N evaluation shots, and they form a floor of order $\sqrt { \log ( 1 / \beta ) / m } + \sqrt { \log ( 1 / \beta ) / N }$ that no amount of noise can lower. The shot term can be made negligible by measuring more, so the floor that remains is set by the training data. Because the contracted terms and this floor both decay as $m ^ { - 1 / 2 }$ , their ratio does not depend on m. In terms of the bound, noise therefore yields a substantial reduction only while $( 1 - \gamma ) \Lambda _ { C }$ exceeds $\sqrt { \log ( 1 / \beta ) }$ up to constants, which favours expressive circuit families with concentrated outputs, precisely the models most prone to overfitting. Beyond that point, additional noise barely shrinks the gap but still pulls the model towards the uniform distribution, a cost we quantify in the next subsection.

## 2.3 Realizing a target noise level by capped error reduction

The previous subsection shows that retained noise contracts the bound on the generalization gap. Noise, however, also changes the model itself. Under the white-noise model, $\mu _ { q _ { \theta } ^ { \gamma } } = ( 1 - \gamma ) \mu _ { q _ { \theta } ^ { 0 } } + \gamma \mu _ { u }$ , so the population loss becomes

$$
\begin{array} { r } { \mathrm { M M D } _ { \kappa } ^ { 2 } \big ( q _ { \theta } ^ { \gamma } , p \big ) = \Big \| \big ( \mu _ { q _ { \theta } ^ { 0 } } - \mu _ { p } \big ) - \gamma \big ( \mu _ { q _ { \theta } ^ { 0 } } - \mu _ { u } \big ) \Big \| _ { \mathcal { H } } ^ { 2 } . } \end{array}\tag{22}
$$

Noise thus pulls the kernel mean embedding of the model towards that of the uniform distribution, and even a perfect fit, $q _ { \theta } ^ { 0 } = p$ , incurs the loss $\gamma ^ { 2 } \mathrm { M M D } _ { \kappa } ^ { 2 } ( p , u )$ . The effective noise level therefore acts as a regularization parameter that trades a smaller gap against the loss of the nonuniform structure the target requires, and its best value can be intermediate.

Therefore, using noise as a regularizer raises two practical questions. The first is how to realize a chosen noise level. The hardware supplies a fixed amount of noise, so a lower level can only be reached by removing part of it with error mitigation. The second is which level to choose, because the best value is not known before training. In this work, we propose GENIE, a noise-based regularization framework that answers these two questions with two components (see Fig. 1 for an overview). In this subsection, we present the first component, capped probabilistic error reduction (capped PER), which realizes a target noise level by removing just enough hardware noise while keeping the remaining noise close to white noise.

## Residual noise and its cost.

We start from the characterized hardware profile € of equation (3). To lower the noise, we choose a residual profile $\mathbf { r } = \left( r _ { 1 } , \ldots , r _ { L } \right)$ with $0 \le r _ { k } \le \epsilon _ { k }$ , which keeps each $\pi _ { k }$ but reduces its rate to $r _ { k }$ . We implement it with probabilistic error reduction (PER) [36], the partial variant of probabilistic error cancellation [35, 40]. PER inserts randomly sampled Pauli corrections and attaches to each measurement outcome a signed weight of magnitude Γ(r). Weighted averages are then unbiased for expectation values under the residual noise, with a standard error that grows as $\Gamma ( \mathbf { r } ) / \sqrt { N }$ . In the weak-noise regime, this cost takes a simple form (Lemma 3, Methods),

$$
\Gamma ( \mathbf { r } ) \simeq \exp ( 2 \left\| \epsilon - \mathbf { r } \right\| _ { 1 } ) .\tag{23}
$$

The cost depends only on the total amount of noise removed, and it grows exponentially with it. The MMD is more expensive still, because its model-model term multiplies two weights, so a fixed precision can require about $\Gamma ( \mathbf { r } ) ^ { 4 }$ times more circuit executions than without mitigation. Every unit of noise that we keep therefore saves an exponential number of executions.

## Effective noise of a residual profile.

To connect a residual profile to Theorem 2, we assume Assumption 2 in a profiledependent form. Writing $q _ { \theta } ^ { \mathbf { r } }$ for the output distribution under $\mathbf { r } ,$ its effective noise level and its distance from white noise are

$$
\gamma ( \mathbf { r } ) : = 1 - \exp \bigl ( - \left\| \mathbf { r } \right\| _ { 1 } \bigr ) ,\tag{24}
$$

$$
\underset { \theta \in \Theta } { \operatorname* { s u p } } \mathrm { M M D } _ { \kappa } \bigl ( q _ { \theta } ^ { \mathbf { r } } , q _ { \theta } ^ { \gamma ( \mathbf { r } ) } \bigr ) \leq c _ { \mathrm { s c r } } \exp \bigl ( - \left\| \mathbf { r } \right\| _ { 1 } \bigr ) \left\| \mathbf { r } \right\| _ { 2 } ,\tag{25}
$$

which recovers the scaling of Assumption 2 for homogeneous rates up to a constant factor. Like the cost, the effective level depends only on the total retained noise $\left\| \mathbf { r } \right\| _ { 1 } .$ The distance from white noise, in contrast, also depends on how the retained noise is spread across the circuit, through $\left\| \mathbf { r } \right\| _ { 2 }$ . Where the noise is kept therefore matters only for how white it is.

## Capped allocation problem.

We can now state the design problem that capped PER solves. A target $\gamma _ { \star }$ specifies the largest effective noise level we are willing to keep. Many residual profiles stay within this level, and they differ in two respects that matter, namely how much mitigation they require and how close their remaining noise is to white noise. We therefore seek the residual profile that meets the target at the lowest cost while keeping its distance from white noise below a tolerance $\varepsilon _ { \mathrm { t o l } } .$

$$
\begin{array} { r l } { \underset { \mathbf { r } \in \mathbb { R } ^ { L } } { \mathrm { m i n } } } & { \exp \left( 2 \left. \epsilon - \mathbf { r } \right. _ { 1 } \right) } \\ { \mathrm { s u b j e c t ~ t o } } & { 1 - \exp \left( - \left. \mathbf { r } \right. _ { 1 } \right) \le \gamma _ { \star } , } \\ & { c _ { \mathrm { s c r } } \exp \left( - \left. \mathbf { r } \right. _ { 1 } \right) \left. \mathbf { r } \right. _ { 2 } \le \varepsilon _ { \mathrm { t o l } } , } \\ & { 0 \le r _ { k } \le \epsilon _ { k } , \qquad k = 1 , \ldots , L . } \end{array}\tag{26}
$$

Because the cost and the level depend only on $R : = \| \mathbf { r } \| _ { 1 }$ , the problem separates into deciding how much noise to keep and deciding where to keep it. For a fixed total $R ,$ the profile closest to white noise is the capped allocation (Lemma 4, Methods)

$$
r _ { k } ^ { \mathrm { c a p } } ( R ) = \mathrm { m i n } \{ \epsilon _ { k } , \nu ( R ) \} , \qquad \sum _ { k = 1 } ^ { L } \mathrm { m i n } \{ \epsilon _ { k } , \nu ( R ) \} = R .\tag{27}
$$

It lowers the noisiest locations to a common level $\nu ( R )$ and leaves the quieter ones untouched, so mitigation is spent only where the noise is concentrated. Capped PER then keeps the largest total noise that meets both requirements,

$$
R _ { \operatorname* { m a x } } : = \operatorname* { m i n } \Bigl \{ \| \pmb { \epsilon } \| _ { 1 } , \log \frac { 1 } { 1 - \gamma _ { \star } } \Bigr \} ,\tag{28}
$$

$$
R ^ { \star } : = \operatorname* { m a x } \Bigl \{ R \in [ 0 , R _ { \mathrm { m a x } } ] ~ \Big | ~ c _ { \mathrm { s c r } } e ^ { - R } \| \mathbf { r } ^ { \mathrm { c a p } } ( R ) \| _ { 2 } \le \varepsilon _ { \mathrm { t o l } } \Bigr \} ,\tag{29}
$$

and implements the profile $\mathbf { r } ^ { \star } = \mathbf { r } ^ { \mathrm { c a p } } ( R ^ { \star } )$ with PER (Algorithm 1, Methods). The result has effective level $\gamma _ { \mathrm { o u t } } = 1 - e ^ { - R ^ { \star } } \leq \gamma _ { \star }$ and cost $\Gamma ^ { \star } = \exp [ 2 ( \| \epsilon \| _ { 1 } - R ^ { \star } ) ]$ . When the profile that reaches $\gamma _ { \star }$ is too far from white noise, capped PER mitigates beyond the target, giving the validity of the white-noise description priority over cost.

Below the hardware noise level, the PER weights make the Born machine a quasiprobability model that is accessed through weighted expectation values rather than samples [40, 45]. The MMD estimates and gradients used in training are such expectations, which we obtain by executing many randomly sampled PER circuits and combining their outcomes with the known weights. At the hardware level no mitigation is applied, and the model can be sampled directly. Capped PER thus turns any target noise level into an implementable residual profile with a known cost, which leaves open the question of which level to target.

## 2.4 Budget-aware noise-control regularization framework

Capped PER realizes any target noise level, which leaves the second question of which level to target. The best level balances the contraction of the gap in Theorem 2 against the bias in equation (22) and the mitigation cost in equation (23). This balance depends on the target distribution and on the model being trained, neither of which is known in advance, so the level cannot be fixed before training. The second component of GENIE therefore selects it during training, by comparing candidate levels on held-out data within a fixed budget of circuit executions. In this subsection we formulate this selection problem, describe the controller that solves it, and bound the regret of its decisions.

## Noise allocation problem.

For a residual profile r, let $R _ { \mathbf { r } } ( \theta ) : = \mathrm { M M D } _ { \kappa } ^ { 2 } ( p , q _ { \theta } ^ { \bf r } )$ denote the population loss under that profile. As discussed in the first subsection, this loss equals the training loss plus the generalization gap, so minimizing it accounts for both fit and generalization. Training performs $T$ parameter updates within a total budget of B circuit executions. A policy π chooses, at each step $t ,$ a residual profile $\mathbf { r } _ { t } ^ { \pi }$ from the set $\mathcal { R } _ { \mathrm { t o l } }$ of profiles produced by capped PER and a number $N _ { t } ^ { \pi }$ of shots per training circuit. At the comparison steps $t \in { \mathcal { Z } } ,$ it also chooses a number $M _ { t j } ^ { \pi }$ of probe shots for each candidate profile j. Its objective is

$$
\begin{array} { r l } { \underset { \pi } { \mathrm { m i n i m i z e } } } & { \mathbb { E } \Big [ R _ { \hat { \mathbf { r } } ^ { \pi } } \big ( \hat { \theta } ^ { \pi } \big ) \Big ] } \\ { \mathrm { s u b j e c t ~ t o } } & { ~ \displaystyle \sum _ { t = 0 } ^ { T - 1 } P _ { \mathrm { c i r c } } N _ { t } ^ { \pi } + \sum _ { t \in \mathcal { T } } \sum _ { j \in \mathcal { T } _ { t } } M _ { t j } ^ { \pi } \leq B , } \\ & { ~ \mathbf { r } _ { t } ^ { \pi } \in \mathcal { R } _ { \mathrm { t o l } } , ~ t = 0 , \dots , T - 1 . } \end{array}\tag{30}
$$

where $P _ { \mathrm { c i r c } } = 1 + 2 \dim ( \theta )$ is the number of circuits per parameter-shift update. The constraint counts circuit executions, whereas the PER normalization determines how precise each execution is. A policy must therefore divide a single budget among the noise level, the precision of the training updates, and the comparisons between candidates.

## Noise selection from held-out data.

Because the population loss is unknown and depends on the whole training trajectory, we address equation (30) with a practical policy that periodically compares candidate noise levels on held-out data (Algorithm 2 in Methods). Applying capped PER to a grid of target levels gives J candidate profiles $\mathbf { r } ^ { ( j ) }$ with effective levels $\gamma _ { j } = 1 - \exp ( - \left. \mathbf { r } ^ { ( j ) } \right. _ { 1 } )$ and normalizations $\Gamma _ { j } = \Gamma ( \mathbf { r } ^ { ( j ) } )$ . A fraction α of the budget is reserved for comparisons and the rest for training, which is spread evenly so that each remaining training circuit receives $N _ { \mathrm { f u t u r e } , t }$ shots. Every $K$ steps, the controller holds the parameters $\theta _ { t }$ fixed, probes every candidate with $M _ { t }$ fresh PER executions, and estimates its MMD to a held-out validation sample of size v with a weighted U-statistic $\widetilde { L } _ { \mathrm { v a l } , t } ^ { U } ( j )$ (Methods). It then selects

$$
j _ { t } \in \arg \operatorname* { m i n } _ { j \in \mathcal { T } } \Bigl \{ \underbrace { \widetilde { L } _ { \mathrm { v a l } , t } ^ { U } ( j ) } _ { \mathrm { v a l i d a t i o n ~ f i t } } + \underbrace { b _ { t j } } _ { \mathrm { m e a s u r e m e n t ~ r a d i u s } } + \underbrace { \lambda \frac { \Gamma _ { j } ^ { 2 } } { N _ { \mathrm { f u t u r e } , t } } } _ { \mathrm { b u d g e t ~ l o o k a h e a d } } \Bigr \}\tag{31}
$$

and keeps the profile $\mathbf { r } ^ { ( j _ { t } ) }$ until the next comparison. The validation fit is an empirical proxy for the population loss at the current parameters. The measurement radius $b _ { t j } : =$ $z \widehat { \mathrm { S E } } ( \widetilde { L } _ { \mathrm { v a l } , t } ^ { U } ( j ) )$ , with $z \geq 0$ , keeps the controller from choosing a strongly mitigated candidate on the strength of a low but imprecise estimate. The budget lookahead penalizes candidates whose mitigation cost would leave too few shots for later training, and it relaxes as the remaining allowance grows. Training starts from the candidate with the largest $\gamma _ { j }$ , which is the cheapest, so noise is removed only when the validation evidence justifies its cost. Because the validation sample is reused across comparisons, final performance is assessed on an independent test set.

## Cumulative selection regret.

To assess these decisions, we compare the controller with the best candidate at every training step, with all candidates evaluated at the same current parameters,

$$
\Psi _ { t } ( j ) : = R _ { \mathbf { r } ^ { ( j ) } } ( \theta _ { t } ) + \lambda \frac { \Gamma _ { j } ^ { 2 } } { N _ { \mathrm { f u t u r e } , t } } , \qquad \mathrm { R e g } _ { T } : = \sum _ { t = 0 } ^ { T - 1 } \Bigl [ \Psi _ { t } ( j _ { t } ) - \operatorname* { m i n } _ { j \in \mathcal { I } } \Psi _ { t } ( j ) \Bigr ] .\tag{32}
$$

The regret reflects both the accuracy of each comparison and the cost of keeping a profile while the parameters change. Its analysis rests on three facts. First, capped PER keeps every candidate within $\varepsilon _ { \mathrm { t o l } }$ of its white-noise model by equation (25), so candidates differ mainly through their effective levels. Second, if each of the $d _ { \theta }$ parameters enters once through a Pauli rotation, the loss changes with the parameters at a rate of at most $L _ { R } : = 2 \sqrt { 2 d _ { \theta } }$ (Lemma 5, Methods). Third, the parameters themselves change slowly.

Assumption 3 (Controlled updates). For a constant $h > 0$ independent of $T _ { i }$

$$
\| \theta _ { t + 1 } - \theta _ { t } \| _ { 2 } \leq { \frac { h } { t + 1 } } , \qquad t = 0 , \ldots , T - 1 .\tag{33}
$$

Assumption 3 holds for stochastic gradient descent with step size $\eta _ { 0 } / ( t + 1 )$ and gradient estimates bounded by $G ,$ with $h = \eta _ { 0 } G$ [46]. It also holds approximately for Adam with the same decay, whose per-coordinate steps are bounded by a constant multiple of the learning rate [47]. The bound is stated in terms of a few quantities of the run. Comparisons occur at $\mathcal { T } = \{ 0 , K , 2 K , . . . \} \cap \{ 0 , . . . , T - 1 \}$ , which gives $H = \lceil T / K \rceil$ rounds, and the comparison at step s governs the next $w _ { s } : = \operatorname* { m i n } \{ K , T - s \}$ steps. We write $\overline { { \Gamma } } : = \operatorname* { m a x } _ { j } \Gamma _ { j }$ for the largest normalization, $\delta _ { \gamma } : = \mathrm { m a x } _ { j } \gamma _ { j } - \mathrm { m i n } _ { j } \gamma _ { j }$ for the range of candidate levels, and $\begin{array} { r } { D _ { s } : = \mathrm { M M D } _ { \kappa } ( q _ { \theta _ { \circ } } ^ { 0 } , u ) } \end{array}$ for how far the noiseless generator is from uniform at step s. Finally, $a _ { v } ( \beta ) : = v ^ { - 1 / 2 } + \sqrt { 2 \log ( 2 / \beta ) / v }$ bounds the error of the validation kernel mean embedding with probability at least $1 - \beta / 2$

Theorem 3 (Cumulative selection regret). Under Assumptions 2 and 3, let $T \geq 2$ and $1 \leq K \leq T$ . Algorithm 2 satisfies, with probability at least $1 - \beta$

$$
\mathrm { R e g } _ { T } \leq ( 1 5 + 6 z ) \overline { { \Gamma } } ^ { 2 } T \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } + 2 a _ { v } ( \beta ) \left( \delta _ { \gamma } \sum _ { s \in \cal Z } w _ { s } D _ { s } + 2 T \varepsilon _ { \mathrm { t o l } } \right)
$$

$$
+ 2 ( K - 1 ) \left[ L _ { R } h ( 1 + \log T ) + \frac { \lambda \overline { { \Gamma } } ^ { 2 } } { N _ { \mathrm { f u t u r e , 0 } } } \right] .\tag{34}
$$

Theorem 3 separates three sources of regret. The first term comes from comparing candidates with finite probe shots and from the measurement radius. It shrinks with the probe budget αB and grows with the number of candidates and comparison rounds, and its factor $\overline { { \Gamma } } ^ { 2 }$ shows that strongly mitigated candidates are expensive to compare, not only to train. The second term comes from the finite validation sample. It is scaled by how different the candidates actually are, so comparisons are robust when the generator is close to uniform or the candidate levels are similar. Because reusing the validation sample supplies no new target data, this term grows linearly in T however many probe shots are spent. The last term is the price of holding a profile for K steps while the loss and the training allowance change, and it vanishes for $K = 1$ . A larger probe fraction α sharpens comparisons but leaves fewer executions for training, which is the budget tradeoff of equation (30). Balancing the first and last terms through the comparison interval yields a sublinear rate.

Corollary 2 (Rate under a linear execution budget). Suppose $B = \Theta ( T )$ and choose $K = \lceil T ^ { 2 / 3 } \rceil$ . For xed circuit size, candidate grid, $h , \alpha , \lambda , z _ { \ i }$ and $\varepsilon _ { \mathrm { t o l } }$ , with probability at least $1 - \beta _ { i }$

$$
\mathrm { R e g } _ { T } = \widetilde { \mathcal { O } } \left( T ^ { 2 / 3 } + \frac { T } { \sqrt { v } } \right) ,\tag{35}
$$

where $\widetilde { \mathcal { O } }$ suppresses logarithmic factors in $T$ and $1 / \beta$

The interval $K = \lceil T ^ { 2 / 3 } \rceil$ balances the precision gained from less frequent comparisons against the cost of acting on stale ones. At fixed confidence, the average regret $\mathrm { R e g } _ { T } / T$ then decays as $\mathcal { O } ( T ^ { - 1 / 3 } \log T + v ^ { - 1 / 2 } )$ . The errors from finite probes and delayed comparisons therefore vanish as training proceeds, whereas a fixed validation set leaves a floor of order $v ^ { - 1 / 2 }$ . Ultimately, it is the amount of held-out data, not the number of shots, that limits how well the noise level can be selected.

## 2.5 Numerical Demonstrations

## 2.5.1 Experiments on Synthetic Datasets

## Loss across training-set sizes and execution budgets

Figure 2 shows the final population loss $L _ { P }$ of each method as a function of the training-set size $m ,$ with one panel per execution budget and the three error rates averaged. The vertical axis is shared across the four panels of each target, so reading a row from left to right shows how each method converts additional executions into a better model. Overall, GENIE gives the lowest loss of the four methods. On blob and wfc, it is the best method, or tied with the best within the seed range, in every panel. On Ising, it is among the best methods in every panel.

The clearest pattern is that the ranking of the fixed noise levels reverses as the budget grows. At the smallest budget, full mitigation is the worst on all three targets,

Binary blob

![](images/0e736074c888bdae804a42f0bfeb6ef74c9b963a8c2b3b99f240bdede0caeb64.jpg)

![](images/7647de248d0c02665e647721dff95717b975a940fe6afd72155c8ebae02317a7.jpg)

![](images/425450afbd92dc9526532d29201da18600dea0e675c904bcabd379e4621196c3.jpg)  
Fig. 2: Final population loss across training-set sizes and execution budgets. Rows show the blob (top), wfc (middle) and Ising (bottom) datasets, and panels within a row show the budgets $B \in \{ 1 , 5 , 2 5 , 1 2 5 \} \times U$ . Each curve shows the final population loss $L _ { P }$ against the training-set size m for training without mitigation, with half mitigation and with full mitigation, and for GENIE. Values are averaged over the three error rates and the three seeds, and bars show the range across seeds. The vertical axis is shared within each row.

because its PER estimates are $\Gamma ^ { 2 }$ times noisier and few executions cannot average this out. With more executions, the mitigated methods keep improving, while training without mitigation stalls on blob and $\mathrm { w f c , }$ because its noise-induced bias (equation (22)) does not shrink with more shots. Beyond the smallest budget, keeping all hardware noise is therefore the worst choice on these targets. No fixed level works well at every budget.

GENIE avoids this problem because it chooses the noise level from data. When executions are scarce, it keeps most of the hardware noise, as training without mitigation does. When executions are plentiful, it removes most of the noise, as full mitigation does. At the intermediate budgets, it does better than all three fixed levels, possibly because it can use levels between them or change the level during training. On Ising, the four methods are close and their seed ranges overlap, so the differences there are small.

Finally, the training-set size matters less than the budget. For most methods, the loss drops between m = 64 and m = 256 and changes little afterwards. On blob and wfc, the ranking of the methods also stays the same across m. In this range of m, the best noise level is therefore decided mainly by the budget and the error rate rather than by the amount of training data. Ising depends more strongly on m at large budgets.

## Regret across all conditions

To compare the methods over the whole grid, we compute in each of the 144 conditions (i.e., combinations of target, error rate, training-set size and budget). The regret is the percentage by which a method's loss exceeds that of the best of the four methods in that condition (Methods). It also differs from the cumulative selection regret RegT of Theorem 3, which sums, over the steps of a single training run, how much worse the level chosen by the controller is than the best candidate level. Figure 3 summarizes the regrets. GENIE has a median regret of 0.3% and a mean regret of 3.8%, far below every fixed level. Full and half mitigation have median regrets of about 8.5% and mean regrets of 11.7% and 14.8%, and training without mitigation is the worst, with a median of 22.7% and a mean of 33.0%. For every method the mean exceeds the median, because regret cannot be negative and a few large regrets pull the mean up.

Two summary numbers can hide how often a method fails, so Fig. 4 shows the full picture. Panel a plots, for each regret threshold x, the percentage of conditions in which a method is within x% of the best method. The curve of GENIE lies above those of all fixed levels at every threshold, so its advantage does not rest on a few favourable conditions. GENIE is within 1% of the best method in 64% of conditions and within 5% in 78%, whereas no fixed level exceeds 29% and 42% at these thresholds.

Panel b counts how often each method is the best and how often it is the worst in its condition. GENIE is the best method in 43% of conditions and the worst in only 2%. The fixed levels are risky in different ways. Full mitigation has a moderate mean regret but is the worst method in 38% of conditions, and training without mitigation is the worst in 57%. Half mitigation is rarely the worst (3%) but also rarely the best (15%).

Therefore, no fixed noise level is safe to use in practice. Each one performs badly in a sizeable fraction of conditions, and which conditions those are depends on the error rate and the budget at hand. In contrast, GENIE is not always the best method, but it is almost never far from the best.

## 2.5.2 Experiments on a Real Census Dataset

The synthetic experiments evaluate distributional fit through population MMD loss. To assess whether these improvements translate into more accurate population statistics and better predictions, we repeat the comparison on the Adult census dataset [48]. Each 10-bit string encodes a person's record through seven attributes, including whether their income exceeds \$50K (Methods). We evaluate the learned distributions against held-out real records using three practical criteria: the shares of people satisfying individual attributes, income rates within demographic subgroups, and the accuracy of income predictions for unseen individuals. These evaluations test whether lower MMD loss corresponds to better preservation of interpretable features of the data distribution.

![](images/62a1d451899a9c85336bb077ff612736faacfb824815cb6d1fe8c2cfe6dbfb6d.jpg)  
Fig. 3: Regret of each method over all conditions. Median (circles) and mean (diamonds) regret of each method relative to the best of the four methods, over the 144 conditions of the grid, with losses averaged over three seeds before the regret is computed. Labels give the median and the mean.

![](images/27371698d8a919b965264b15c94664f40e484c89030ea28ddc38ec12efc689f3.jpg)

b  
![](images/1e2afb82a3802e1cb9484d031c27b4f537643358299d21b1cf403024651507b9.jpg)  
Fig. 4: How often each method is near the best. a, Percentage of the 144 conditions in which each method is within a given regret threshold of the best method. A point (x, y) means that the method is within $x \%$ of the best method in $y \%$ of the conditions. The dotted lines mark thresholds of 1% and 5%. b, Percentage of conditions in which each method is the worst (left) and the best (right) of the four methods.

## Population shares.

We first evaluate whether generated samples reproduce the marginal probabilities of seven demographic and socioeconomic events in the Adult dataset. For each event, such as earning above \$50K, we compare the fraction of generated records satisfying that event with the corresponding fraction in the held-out real data. Figure 5 shows these fractions. In details, the gap between a method's bar and the dotted reference line is its absolute error for that event.

GENIE achieves the smallest error, or ties for the smallest, on six of the seven events. The exception is the share of married people, where GENIE is two points below the real share and the other methods are within one point. For example, without mitigation, 43% of generated records have income above \$50K, compared with 24% in the real data. GENIE reduces this share to 32%. Similarly, it reduces the generated share of managers or professionals from 43% to 31%, closer to the real share of 25%. Averaged over the seven events, the absolute discrepancy is approximately 4 percentage points for GENIE, compared with 8-10 percentage points for the fixed mitigation levels. All methods closely match the shares of married people and people with a bachelor's degree or higher, so these two events provide little separation between methods.

![](images/d11f9e32d4298f1db48bcb627399af705f1f47f7acc42d22f3b542140fd4ba88.jpg)  
Fig. 5: Generated population shares on Adult. Each group compares the percentage of records satisfying the indicated event in held-out real data (gray) and samples generated by four methods (colored). Dotted lines mark the real-data percentages. Smaller absolute gaps from these lines indicate more accurate marginal probabilities. Labels above bars show rounded percentages. GENIE is closest or tied for closest on six of the seven events, with an average absolute discrepancy of approximately 4 percentage points, versus 8-10 for the fixed mitigation levels. Values are averaged over combinations of $B \in \{ 5 , 2 5 , 1 2 5 \} \times U$ and training-set sizes m ∈ {64, 128, 256, 512}.

## Income rates within groups.

Matching marginal population shares does not ensure that a model preserves relationships between attributes. Therefore, we evaluate conditional income rates. Specifically, the rate is the fraction of records earning above \$50K within a specified subgroup, such as married men or people aged 28 or younger without a high-school diploma. We consider 125 subgroups, each defined by a combination of values of two non-income attributes and containing at least 2% of the real test records. For each subgroup g, we compare the generated income rate $\widehat { q } _ { g }$ with the real test rate $\widehat { p } _ { g }$ . We summarize these discrepancies using the group-size-weighted mean absolute error,

![](images/dd8946bacf535e46ce80cf79479fc2ad09f69e9aeaba2dec895c3a4c68a569b3.jpg)

![](images/3bee57f56764b609b90f6e38b14b5255ac92a4148a8fdaf7189955fa8ecebb59.jpg)

![](images/faa83a7cd67d88dbb41fbe6b518d4d84629fb82d96474e3e2a00dcefec58084f.jpg)

![](images/5ab284912024f9315a14242a90fda8174b44a2c4d33f03ea1bd35fd8d7a0167f.jpg)  
Fig. 6: Preservation of subgroup income rates on Adult. Each marker represents one of 125 subgroups defined by values of two non-income attributes and covering at least 2% of the real test records. Its horizontal coordinate is the fraction of real subgroup members earning above \$50K, while its vertical coordinate is the corresponding fraction in generated data. The dashed diagonal denotes exact agreement such that points above it overestimate the subgroup's income rate, and points below it underestimate it. Marker size indicates real subgroup size. Vertical bars show how much one group's generated answer varies across runs. Panel annotations report the group-sizeweighted mean absolute error in percentage points. GENIE reduces this error to 16 points, compared with 19–22 for the fixed mitigation levels.

$$
E _ { \mathrm { r a t e } } = { \frac { \sum _ { g \in { \mathcal { G } } } n _ { g } \left| { \widehat { q } } _ { g } - { \widehat { p } } _ { g } \right| } { \sum _ { g \in { \mathcal { G } } } n _ { g } } } ,
$$

where $n _ { g }$ is the number of real test records in subgroup $g .$ The subgroups may overlap.   
The weighting gives greater importance to errors on larger groups.

Figure 6 plots the generated rate against the real rate for each subgroup, with the diagonal indicating exact agreement. The real rates range from approximately 0% to 70%. Without mitigation, many generated rates cluster near 50%, including for groups whose real rates are much lower. The fixed mitigation levels also overestimate income rates for many low-income groups, while full mitigation exhibits substantial variation across seeds. GENIE produces lower rates for many of these groups and achieves the lowest weighted mean absolute error with 16 percentage points, compared with 21 without mitigation, 19 with 50% mitigation, and 22 with full mitigation.

Nevertheless, GENIE does not fully recover the conditional income structure. Many low-income groups remain overestimated, and some high-income groups are underestimated. For context, the same metric is approximately 28 percentage points for a uniform distribution over encoded records and less than 1 percentage point for the empirical training distribution. Thus, GENIE improves conditional fidelity under the available measurement budget, but a gap remains relative to the relationships present in the training data.

![](images/e3de633d1bad37ed9ff73c26050d9a795b3b3f9f2bac96050ffafe498cb1c814.jpg)  
Fig. 7: Income prediction on held-out records. Predictions are obtained by conditioning each learned distribution on a test record's nine non-income bits and applying the decision rule in Methods. Each stacked bar represents 100 real test records: blue denotes correct predictions, orange denotes false positives (income at most \$50K predicted as above \$50K), and gray denotes false negatives (income above \$50K predicted as at most \$50K).

## Income prediction for real people.

We next evaluate whether the learned distribution supports income prediction on unseen real records. For each test record, we condition on its nine non-income bits and use the model's conditional probability of the income bit to predict whether income exceeds \$50K, following the decision rule in Methods. This predictor is read directly from the learned distribution.

Figure 7 partitions the real test records into three outcomes including correct predictions, false positives (income at most \$50K predicted as above \$50K), and false negatives (income above \$50K predicted as at most \$50K). Each bar reports these outcomes per 100 test records. GENIE correctly classifies approximately 68 records, compared with 61 without mitigation, 57 with 50% mitigation, and 58 with full mitigation. This improvement comes primarily from fewer false positives, which are, 19 per 100 records for GENIE, versus 26–30 for the fixed mitigation levels. All four methods produce approximately 13 false negatives per 100 records at the displayed precision.

## 3 Discussion

In this work, we have shown that the residual noise of a quantum circuit Born machine acts as a regularizer whose strength can be quantified, realized on hardware and chosen during training. The analysis rests on an identity that reduces the generalization gap of an MMD-trained Born machine to the overfitting of its witness function, so that for any noise process the gap is controlled by the complexity of the witness class (Theorem 1). For scrambled local Pauli noise, this complexity contracts by the retained-signal factor 1 - γ (Theorem 2). The best noise level can be intermediate and unknown before training. GENIE turns this trade-off into a procedure. Capped probabilistic error reduction realizes any target level at the lowest mitigation cost while keeping the remaining noise close to white noise, and a budget-aware controller selects the level from held-out data with an average regret per step that decays as $T ^ { - 1 / 3 }$ up to a floor of order $v ^ { - 1 / 2 }$ set by the held-out sample size (Theorem 3 and Corollary 2).

These results add a second consideration to how error mitigation is used. Mitigation is usually judged by how much bias it removes at a given sampling cost [45]. When a generative model is trained under a fixed budget of circuit executions, however, mitigation buys noisier shots rather than fewer of them, and the noise it removes no longer regularizes the model. How much noise to remove then becomes a modelselection problem that depends on the target, the training data and the budget. Our simulations show what is at stake. Across 144 synthetic conditions, the fixed noise levels had median regrets of 8.4% to 22.7% relative to the best of the four methods, and each was the worst method in some conditions, whereas GENIE had a median regret of 0.3%. On the Adult census data at strong noise, it gave the most accurate population shares, income rates and income predictions of the four methods. This view parallels the use of injected noise as a regularizer in classical learning [25, 26] and the finite optimal noise levels found for supervised quantum models [28, 30, 31|. For a Born machine, however, the noisy output distribution is itself the model, so the regularizer carries a bias that we quantify and that the controller weighs against the reduction of the gap.

Our study has three remaining limitations. First, the effective depolarizing approximation underlying Theorem 2 and the controller requires sufficient circuit scrambling to keep the approximation error small. Thus, its precision on estimating the amount of residual noise depends on the circuit architecture and depth. Second, the controller compares candidate noise levels at its current parameters rather than planning a schedule for the whole run, and the reliability of its choices depends on the held-out sample size. In the census experiment, held-out samples of 13 to 102 people were already sufficient for GENIE to select useful noise levels. Finally, all experiments use simulated circuits of 10 and 12 qubits.

These limitations suggest several natural extensions. Testing GENIE on larger circuits and hardware would help assess the effective depolarizing approximation and the benefits of noise control under realistic conditions. Randomized compiling and sparse noise learning provide tools for tailoring and characterizing the Pauli channels used by capped probabilistic error reduction [39, 40], allowing the mismatch $\varepsilon _ { \mathrm { P a u l i } }$ to be assessed along training trajectories. On the theory side, Theorem 1 covers arbitrary noise processes, while the explicit contraction in Theorem 2 relies on scrambled Pauli noise. Deriving contraction bounds for other channels would clarify which correlations they suppress and when retaining noise is beneficial. More broadly, our results suggest that residual noise, when appropriately characterized and controlled, can serve as a resource for generalization on near-term hardware.

## 4 Methods

This section collects the technical ingredients behind the Results. We first give the lemmas and proof sketches behind the generalization bounds, then the construction of capped probabilistic error reduction, then the implementation and analysis of the

budget-aware controller, and finally the setup of our numerical experiments. Complete proofs are given in Supplementary Note [1].

## 4.1 Generalization bounds

## Reduction to the learned witness.

Both generalization theorems start from an exact identity, which shows that only the coupling term of the MMD can make the empirical risk optimistic.

Lemma 1 (Reduction to the learned witness). Let $m , N \ge 2$ , and let $\mathcal { A } _ { \mathcal { N } }$ be any possibly randomized learning algorithm. If the evaluation samples $\mathcal { V }$ are conditionally independent of S and the training randomness given ${ \hat { \theta } } _ { \mathcal { N } }$ , then

$$
\Delta _ { \mathcal { N } } ( m ) = 2 \mathbb { E } _ { \mathcal { S } , \mathcal { A } _ { \mathcal { N } } } \left[ ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } \right] .\tag{36}
$$

Proof sketch. The target-target U-statistic is unbiased and independent of θ. The model-model U-statistic uses samples drawn after training and is therefore unbiased conditionally on $\hat { \theta } _ { \mathcal { N } }$ . In the coupling term, averaging over $\mathcal { V }$ given S and $\hat { \theta } _ { \mathcal { N } }$ replaces $\begin{array} { r } { \frac { 1 } { N } \sum _ { j } \kappa ( x _ { i } , y _ { j } ) } \end{array}$ by $h _ { \hat { \theta } _ { \wedge r } } ^ { \mathcal { N } } ( x _ { i } )$ , so the sample coupling term becomes $P _ { m } h _ { \hat { \theta } _ { N } } ^ { N }$ . Its population counterpart is $\dot { P h } _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } }$ , which equals the coupling term of the risk by the definition of the witness, equation (8). Only the deviation in equation (36) remains, and it does not involve $N$

Proof sketch of Theorem 1. The learned witness belongs to $\mathcal { W } ^ { N }$ , so Lemma 1 gives $| \Delta _ { \mathcal { N } } ( m ) | \leq 2 \mathbb { E } _ { \mathcal { S } } \operatorname* { s u p } _ { h \in \mathcal { W } ^ { \mathcal { N } } } | ( P _ { m } - P ) h |$ . The standard symmetrization argument [41] bounds this uniform deviation by the Rademacher complexity,

$$
\mathbb { E } s \Big [ \operatorname* { s u p } _ { h \in \mathcal { W } ^ { \mathcal { N } } } \big | ( P _ { m } - P ) h \big | \Big ] \leq 2 \Re ( \mathcal { W } _ { \pm } ^ { \mathcal { N } } ) ,\tag{37}
$$

which gives $| \Delta _ { \mathcal { N } } ( m ) | \leq 4 \Re ( \mathcal { W } _ { \pm } ^ { \mathcal { N } } )$

## Complexity of the noiseless witness class.

Theorem 2 reduces the noisy complexity to that of the centered noiseless witnesses $\bar { h } _ { \theta } : = h _ { \theta } ^ { 0 } - \widehat { \phi } ( \emptyset )$ and their class $\bar { \mathcal { W } } _ { \pm } ^ { 0 } : = \{ \pm \bar { h } _ { \theta } : \theta \in \Theta \}$ , which the parity basis makes explicit.

Lemma 2 (Complexity of the noiseless witness class). Under Assumption 1, and for a given $C \geq 1$ such that $\begin{array} { r } { \operatorname* { s u p } _ { \theta \in \Theta } 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } \leq C _ { \mathrm { . } } } \end{array}$ , we have:

$$
\Re { \left( \bar { \mathcal { W } } _ { \pm } ^ { 0 } \right) } \leq \frac { \Lambda _ { C } } { \sqrt { m } } ,\tag{38}
$$

where $\Lambda _ { C }$ is the constant defined in Theorem ${ \it 2 . }$

Proof sketch. By equation (16), the Rademacher average of a centered witness is a sum over nonconstant parities,

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sigma _ { i } \bar { h } _ { \theta } ( x _ { i } ) = \sum _ { A \ne \emptyset } \widehat { \phi } ( A ) \widehat { q } _ { \theta } ^ { 0 } ( A ) g _ { A } , \qquad g _ { A } : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sigma _ { i } \chi _ { A } ( x _ { i } ) .
$$

The Cauchy-Schwarz inequality over A separates the generator from the kernel and the sample. Parseval's identity gives $\begin{array} { r } { \sum _ { A \neq \emptyset } \widehat { q _ { \theta } ^ { 0 } } ( A ) ^ { 2 } \ = \ 2 ^ { n } \sum _ { x } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } \ - \ 1 \ \leq \ } \end{array}$ $C - 1$ , and Jensen's inequality with $\mathbb { E } _ { \sigma } g _ { A } ^ { 2 } = 1 \dot { / } m$ bounds the remaining factor by $\textstyle ( \sum _ { A \neq \emptyset } { \widehat { \phi } } ( A ) ^ { 2 } / m ) ^ { 1 / 2 }$ . Weighting the Cauchy-Schwarz inequality by $\widehat { \phi } ( A )$ instead, and using $| \widehat { q _ { \theta } ^ { 0 } } ( A ) | \leq 1$ and $\textstyle \sum _ { A } { \widehat { \phi } } ( A ) = \phi ( \mathbf { 0 } ) = 1$ , gives the second branch $1 - { \widehat { \phi } } ( \emptyset )$ of $\Lambda _ { C }$

Proof sketch of Theorem 2. The proof combines Lemma 1 with Lemma 2. Because ${ \widehat { u } } ( A ) = 0$ for every $A \neq \emptyset$ , the white-noise witness is $h _ { \theta } ^ { \gamma } = { \widehat { \phi } } ( \emptyset ) + ( 1 - \gamma ) { \bar { h } } _ { \theta }$ . Writing $e _ { \theta } : = h _ { \theta } ^ { \mathcal { N } } - h _ { \theta } ^ { \gamma } = \mu _ { q _ { \theta } ^ { \mathcal { N } } } - \mu _ { q _ { \theta } ^ { \gamma } }$ for the remainder, every physical witness decomposes as

$$
h _ { \theta } ^ { \mathcal { N } } = \widehat { \phi } ( \varpi ) + \left( 1 - \gamma \right) \bar { h } _ { \theta } + e _ { \theta } ,\tag{39}
$$

$$
\begin{array} { r } { \| e _ { \theta } \| _ { \mathcal { H } } = \mathrm { M M D } _ { \kappa } \big ( q _ { \theta } ^ { N } , q _ { \theta } ^ { \gamma } \big ) \le \varepsilon _ { \mathrm { P a u l i } } , } \end{array}\tag{40}
$$

where the inequality is Assumption 2. Inserting this decomposition into equation (36), the constant vanishes by equation (10), and the contracted component contributes at most $4 ( 1 ~ - ~ \gamma ) \Re ( \hat { \mathcal { W } } _ { \pm } ^ { 0 } )$ . The remainder satisfies $\begin{array} { r l } { ( P _ { m } ~ - ~ P ) e _ { \theta } } & { { } = } \end{array}$ $\begin{array} { r } { \big \langle e _ { \theta } , \frac { 1 } { m } \sum _ { i } \kappa ( x _ { i } , \cdot ) - \mu _ { p } \big \rangle _ { \mathcal { H } } \le \varepsilon _ { \mathrm { P a u l } } } \end{array}$ i $\begin{array} { r } { \big \| \overline { { \frac { 1 } { m } } } \sum _ { i } \kappa ( x _ { i } , \cdot ) - \mu _ { p } \big \| _ { \mathcal { H } } . } \end{array}$ and the expected error of the empirical kernel mean is at most $m ^ { - 1 / 2 }$ , which gives the term $2 \varepsilon _ { \mathrm { P a u l i } } / \sqrt { m }$

Proof sketch of Corollary 1. McDiarmid's inequality controls how far the supremum over the contracted witness class and the error of the empirical kernel mean deviate from their expectations, and both deviations inherit the factor $1 - \gamma$ . Hoeffding's inequality for U-statistics controls the target-target estimate and the terms that involve the evaluation samples Y. A union bound over these events gives the stated probability.

## 4.2 Capped probabilistic error reduction

This subsection gives the quasi-probability construction behind the cost in equation (23) and the procedure that computes the capped profile.

## Quasi-probability implementation.

At location k, the residual channel with rate $r _ { k } \le \epsilon _ { k }$ keeps the Pauli distribution of the characterized channel (3),

$$
\mathcal { E } _ { k , r _ { k } } ( \rho ) = ( 1 - r _ { k } ) \rho + r _ { k } \sum _ { P \in \{ X , Y , Z \} } \pi _ { k } ( P ) P \rho P .\tag{41}
$$

PER implements the correction $\mathcal { M } _ { k , r _ { k } } : = \mathcal { E } _ { k , r _ { k } } \circ \mathcal { E } _ { k } ^ { - 1 }$ . This map is generally not a physical channel, but it can be written as a signed combination of Pauli conjugations, $\begin{array} { r } { \mathcal { M } _ { k , r _ { k } } = \sum _ { P } \eta _ { k , r _ { k } } ( P ) \mathcal { P } } \end{array}$ with $\mathcal { P } ( \rho ) = P \rho P$ . PER samples $\mathcal { P }$ with probability proportional to $| \eta _ { k , r _ { k } } ( P ) |$ and assigns the corresponding outcome the sign of $\eta _ { k , r _ { k } } ( P )$ Because the corrections are sampled independently across locations, the circuit-level normalization is

$$
\Gamma ( \mathbf { r } ) : = \prod _ { k = 1 } ^ { L } \Gamma _ { k } ( r _ { k } ) , \qquad \Gamma _ { k } ( r _ { k } ) : = \sum _ { P } \bigl | \eta _ { k , r _ { k } } ( P ) \bigr | ,\tag{42}
$$

and every outcome carries a weight of magnitude $\Gamma ( \mathbf { r } )$ . The following lemma gives the cost used in the Results.

Lemma 3 (Cost of a mitigation profile). Suppose that $\epsilon _ { k } < 1 / 2$ and $0 \le r _ { k } \le \epsilon _ { k }$ $f o r$ every k. The circuit-level quasiprobability normalization required to reduce ∈ to r satisfies

$$
\log \Gamma ( \mathbf { r } ) = 2 \left\| \boldsymbol { \epsilon } - \mathbf { r } \right\| _ { 1 } + \mathcal { O } \left( \sum _ { k = 1 } ^ { L } \epsilon _ { k } ( \epsilon _ { k } - r _ { k } ) \right) .\tag{43}
$$

Consequently, in the weak-noise regime,

$$
\Gamma ( \mathbf { r } ) \simeq \exp \left( 2 \left\| \epsilon - \mathbf { r } \right\| _ { 1 } \right) .\tag{44}
$$

Proof sketch. Each $\mathcal { M } _ { k , r _ { k } }$ is diagonal in the Pauli transfer basis. To first order in the rates, $\mathcal { M } _ { k , r _ { k } }$ ≈ id $\begin{array} { r } { - \left( \epsilon _ { k } - r _ { k } \right) \sum _ { P } \pi _ { k } ( P ) ( \mathcal { P } - \mathrm { i d } ) } \end{array}$ , SO $\Gamma _ { k } ( r _ { k } ) = 1 + 2 ( \epsilon _ { k } - r _ { k } ) +$ $\mathcal { O } ( \epsilon _ { k } ( \epsilon _ { k } - r _ { k } ) )$ . Taking logarithms and summing over the independent locations gives equation (43). The model-model term of the MMD multiplies two such weights, so its variance has a leading term of order $\Gamma ( \mathbf { r } ) ^ { 2 } / N$ from fluctuations of one weight at a time and a term of order $\Gamma ( \mathbf { r } ) ^ { 4 } / N ^ { 2 }$ from both weights together

## Capped allocation and the retained budget.

The capped profile in equation (27) solves the allocation step of problem (26) at a fixed total residual noise.

Lemma 4 (Capped allocation at a fixed budget). Fix $R \in \left[ 0 , \left. \epsilon \right. _ { 1 } \right]$ . The unique solution of

$$
\begin{array} { r l } { \underset { \mathbf { r } } { \mathrm { m i n } } } & { { } \left\| \mathbf { r } \right\| _ { 2 } } \\ { s u b j e c t ~ t o } & { { } \left\| \mathbf { r } \right\| _ { 1 } = R , \qquad 0 \leq r _ { k } \leq \epsilon _ { k } } \end{array}\tag{45}
$$

is

$$
\begin{array} { r } { r _ { k } ^ { \mathrm { c a p } } ( R ) = \operatorname* { m i n } \{ \epsilon _ { k } , \nu ( R ) \} , } \end{array}\tag{46}
$$

where $\nu ( R ) \geq 0$ is chosen such that

$$
\sum _ { k = 1 } ^ { L } \operatorname* { m i n } \{ \epsilon _ { k } , \nu ( R ) \} = R .\tag{47}
$$

```latex
Algorithm 1 Capped probabilistic error reduction
Require: characterized rates €, target $\gamma _ { \star }$ , tolerance $\varepsilon _ { \mathrm { t o l } }$ , and scrambling constant $c _ { \mathrm { s c r } }$
Ensure: residual profile $\mathbf { r } ^ { \star }$ , realized level $\gamma _ { \mathrm { o u t } }$ , and normalization $\Gamma ^ { \star }$
1: Compute $R _ { \mathrm { m a x } }$ using Equation (28)
2: Sort € to evaluate the capped profiles efficiently
3: Find $R ^ { \star }$ using Equation (29)
4: Find $\nu ( R ^ { \star } )$ satisfying Equation (47)
5: Set $r _ { k } ^ { \star } \gets$ min $\{ \epsilon _ { k } , \nu ( R ^ { \star } ) \}$ for every k
6: for $k = 1 , \dots , L$ do
7: Configure $\mathcal { M } _ { k , r _ { k } ^ { \star } } \gets \mathcal { E } _ { k , r _ { k } ^ { \star } } \circ \mathcal { E } _ { k } ^ { - 1 }$
8: end for
9: $\gamma _ { \mathrm { o u t } }  1 - e ^ { - R ^ { \star } }$
10: $\Gamma ^ { \star } \gets \exp \left[ 2 ( \| \epsilon \| _ { 1 } - R ^ { \star } ) \right]$
11: return $\mathbf { r } ^ { \star } , \gamma _ { \mathrm { o u t } } , \Gamma ^ { \star }$
```

Proof sketch. The problem minimizes a strictly convex function over a polytope. Its Karush-Kuhn-Tucker conditions give $r _ { k } = \operatorname* { m i n } \{ \epsilon _ { k }$ , max $\{ 0 , \nu \} \}$ for a multiplier ν. Because $\begin{array} { r } { \sum _ { k } \operatorname* { m i n } \{ \epsilon _ { k } , \nu \} } \end{array}$ is continuous and nondecreasing in ν, a value $\nu ( R )$ satisfying equation (47) exists, and strict convexity gives uniqueness.

It remains to choose the retained budget $R ^ { \star }$ in equation (29). Let $g ( R ) \ : =$ $e ^ { - R } \| \mathbf { r } ^ { \mathrm { { c a p } } } ( R ) \| _ { 2 }$ denote the mismatch bound of the capped profile. Increasing R raises only the capped entries, so $\begin{array} { r } { \frac { d } { d R } \left\| \mathbf { r } ^ { \mathrm { c a p } } ( R ) \right\| _ { 2 } ^ { 2 } = 2 \nu ( R ) } \end{array}$ and $g ^ { \prime } ( R )$ has the sign of $\nu ( R ) - \left\| \mathbf { r } ^ { \mathrm { c a p } } ( R ) \right\| _ { 2 } ^ { 2 }$ . Since every entry is at most $\nu ( R )$ , this quantity is at least $\nu ( R ) ( 1 - R )$ , which is positive in the weak-noise regime R < $R < 1$ . The mismatch bound therefore increases with R, the feasible budgets form an interval, and $R ^ { \star }$ is found by bisection. Algorithm 1 summarizes the procedure.

## 4.3 Budget-aware noise control

This subsection specifies the shot allocation and the validation estimator used by the GENIE controller, and then sketches the analysis behind Theorem 3 and Corollary 2.

## Shot allocation and validation estimates.

Let $B _ { \mathrm { t r } , t }$ and $B _ { \mathrm { p r } , t }$ denote the unspent training and probe budgets at step t, initialized to $( 1 - \alpha ) B$ and αB. Each training step receives an equal share of the remaining training budget, and each comparison an equal share of the remaining probe budget,

$$
N _ { \mathrm { f u t u r e } , t } : = \frac { B _ { \mathrm { t r } , t } } { P _ { \mathrm { c i r c } } ( T - t ) } , \qquad N _ { t } : = \big \lfloor N _ { \mathrm { f u t u r e } , t } \big \rfloor , \qquad M _ { t } : = \Big \lfloor \frac { B _ { \mathrm { p r } , t } } { J u _ { t } } \Big \rfloor ,\tag{48}
$$

where $u _ { t }$ is the number of comparisons remaining, including the current one. Subtracting the executions spent at each step keeps the policy within the budget of equation (30). At a comparison, the PER probes of candidate j return independent outcome-weight pairs $( Y _ { t j , a } , W _ { t j , a } )$ with $| W _ { t j , a } | \leq \Gamma _ { j }$ and $\mathbb { E } [ W _ { t j , a } \kappa ( Y _ { t j , a } , \cdot ) ] = \mu _ { q _ { \theta _ { \mathrm { + } } } ^ { \bf r } ( j ) }$

Algorithm 2 Budget-aware noise-controlled regularization   
Require: Target data $s ,$ fixed kernel $\kappa ,$ characterized hardware, budget $B ,$ steps $T ,$   
update interval $1 \leq K < T _ { \mathit { 3 } }$ target noise grid, mismatch tolerance $\varepsilon _ { \mathrm { t o l } } .$ and $\alpha , z , \lambda$   
Ensure: Trained parameters $\hat { \theta }$ and residual profile ê   
1: Split $s$ into $S _ { \mathrm { t r } }$ and $ { S _ { \mathrm { v a l } } }$   
2: Construct $J$ distinct admissible profiles $\mathbf { r } ^ { ( 1 ) } , \ldots , \mathbf { r } ^ { ( J ) }$ using Algorithm $1 ;$ record   
$\gamma _ { j } , \Gamma _ { j }$   
3: $\breve { \mathcal { I } } \gets \{ 1 , \dots , J \} ; \mathcal { T } \gets \{ 0 , K , 2 K , \dots \} \cap \{ 0 , \dots , T - 1 \}$   
4: Initialize $\theta _ { 0 } ;$ set r to the candidate with largest $\gamma _ { j }$   
5: $P _ { \mathrm { c i r c } }  1 + 2 \mathrm { d i m } ( \theta ) ; B _ { \mathrm { t r , 0 } }  ( 1 - \alpha ) B ; B _ { \mathrm { p r , 0 } }  \tilde { \alpha } B $   
6: for $t = 0 , \ldots , T - 1$ do   
7: Compute $N _ { \mathrm { f u t u r e } , t }$ and $N _ { t }$ using Equation (48); set $M _ { t } \gets 0$   
8: if $t \in \mathcal { Z }$ then   
9: $u _ { t } \gets | \{ s \in \mathbb { Z } : s \geq t \} |$ ; compute $M _ { t }$ using Equation (48)   
10: for $j \in \mathcal I$ do   
11: $\mathrm { A t }$ fixed $\theta _ { t } ,$ collect $M _ { t j } = M _ { t }$ fresh PER probe shots under profile $\mathbf { r } ^ { ( j ) }$   
12: Compute $\widetilde { L } _ { \mathrm { v a l } , t } ^ { U } ( j )$ and $b _ { t j }$ using Equation (49)   
13: end for   
14: Select $j _ { t }$ using Equation (31); set $\mathbf { r } \gets \mathbf { r } ^ { ( j _ { t } ) }$   
15: end if   
16: $\mathbf r _ { t } \gets \mathbf r$   
17: Estimate the training MMD gradient using $S _ { \mathrm { t r } } ,$ profile $\mathbf { r } _ { t } ,$ and $N _ { t }$ shots per   
circuit   
18: Update $\theta _ { t }$ to $\theta _ { t + 1 }$   
19: $B _ { \mathrm { t r } , t + 1 }  B _ { \mathrm { t r } , t } - P _ { \mathrm { c i r c } } N _ { t }$   
20: $B _ { \mathrm { p r } , t + 1 }  B _ { \mathrm { p r } , t } - J M _ { t }$   
21: end for   
22: return $\hat { \theta } \gets \theta _ { T } , \hat { \mathbf { r } } \gets \mathbf { r } ,$ and $\hat { \gamma } = 1 - \exp ( - \| \hat { \mathbf { r } } \| _ { 1 } )$

On the validation sample $ { \mathcal { S } } _ { \mathrm { v a l } } ~ = ~ \{ x _ { 1 } , \ldots , x _ { v } \}$ , the controller uses the weighted U-statistic

$$
\begin{array} { l } { { \displaystyle \widetilde { \cal L } _ { \mathrm { v a l } , t } ^ { U } ( j ) = \frac { 1 } { v ( v - 1 ) } \sum _ { i \neq i ^ { \prime } } \kappa ( x _ { i } , x _ { i ^ { \prime } } ) + \frac { 1 } { M _ { t } ( M _ { t } - 1 ) } \sum _ { a \neq b } W _ { t j , a } W _ { t j , b } \kappa ( Y _ { t j , a } , Y _ { t j , b } ) } } \\ { { \displaystyle ~ - \frac { 2 } { v M _ { t } } \sum _ { i = 1 } ^ { v } \sum _ { a = 1 } ^ { M _ { t } } W _ { t j , a } \kappa ( x _ { i } , Y _ { t j , a } ) , } } \end{array}\tag{49}
$$

which, averaged over the fresh probes, recovers the validation U-statistic with exact generator expectations. Excluding the diagonal pairs removes the self-interaction bias, and the estimates are compared without clipping because a U-statistic can be negative. The standard error in the measurement radius is estimated conditionally on the validation data and accounts for the dependence among the pairwise terms. Algorithm 2 combines these rules with the selection rule (31)

## Regret analysis.

The regret bound needs control over how fast candidate scores change between comparisons. On the circuit side, this is provided by the following sensitivity bound.

Lemma 5 (Risk sensitivity). Suppose each of the $d _ { \theta } = \dim ( \theta )$ parameters occurs once, through a Pauli rotation $e ^ { - i \theta _ { k } \hat { P _ { k } } / 2 }$ , with fxed, parameter-independent $C P T P$ noise channels and fxed measurement. Then, uniformly over admissible residual profiles,

$$
| R _ { \mathbf { r } } ( \theta ) - R _ { \mathbf { r } } ( \theta ^ { \prime } ) | \leq L _ { R } \| \theta - \theta ^ { \prime } \| _ { 2 } , \qquad L _ { R } : = 2 \sqrt { 2 d _ { \theta } } .\tag{50}
$$

Proof sketch. The parameter-shift rule writes each partial derivative of the kernel mean embedding $\mu _ { q _ { \theta } ^ { \mathbf { r } } }$ as half the difference of the embeddings of two valid output distributions. Because every MMD is at most ${ \sqrt { 2 } } .$ each partial derivative of $R _ { \mathbf { r } }$ is bounded by a constant, and summing over the $d _ { \theta }$ coordinates gives the bound.

Proof sketch of Theorem 3. Group the training steps by the comparison that governs them, and split the regret of each step into the selection error at that comparison and the change in scores since then. The selection error has three parts. The first is the probe error, which is controlled by the concentration of the weighted U-statistics with weights bounded by Γ and a union bound over the JH estimates, and it gives the first term. The second is the measurement radius, which is at most $6 z \overline { { \Gamma } } ^ { 2 } / \sqrt { M _ { s } }$ because the standard error is bounded by the range of its per-shot terms, and it adds 6z to the constant of the first term. The third is the validation error, which is controlled by $a _ { v } ( \beta )$ and by the distance $\delta _ { \gamma } D _ { s } + 2 \varepsilon _ { \mathrm { t o l } }$ between candidate mean embeddings implied by equation (25), and it gives the second term. The change in scores within each block is bounded with Lemma 5 and Assumption 3 by summing the step sizes $h / ( t + 1 )$ , and together with the change of the budget penalty it gives the last term.

Proof sketch of Corollary 2. With $B = \Theta ( T )$ and $K = { \bar { \lceil } } T ^ { 2 { \bar { / } } 3 } { \rceil }$ , there are $H =$ $\mathcal { O } ( T ^ { 1 / 3 } )$ comparisons, so the first term of equation (34) is $\widetilde { \mathcal { O } } ( T ^ { 2 / 3 } )$ . The second term is $\widetilde { \mathcal { O } } ( T / \sqrt { v } )$ because $D _ { s } \leq \sqrt { 2 }$ The last term is $\mathcal { O } ( T ^ { 2 / 3 } \log T )$ because $N _ { \mathrm { f u t u r e } , 0 } =$ $( 1 - \alpha ) B / ( P _ { \mathrm { c i r c } } T ) = \Theta ( 1 )$

## 4.4 Experimental Setup

We evaluate GENIE in simulation on three synthetic targets at $n = 1 2$ qubits and one real dataset (i.e., UCI Adult Census [48]), varying the training-set size, the hardware error rate and the execution budget. We first describe the setup for accessment on synthetic datasets first and spend the last paragraph on the real census experiment.

## Synthetic Datasets.

The first is a binary blob distribution generated by eight prototype strings whose bits are flipped independently with probability 0.2. The second is a weighted fixedcardinality distribution on the ${ \binom { 1 2 } { 6 } } = 9 2 4$ strings of Hamming weight six, with weight parameter $\omega = 6 .$ The third is the Boltzmann distribution of a ferromagnetic Ising ring at temperature 2.4. Their effective numbers of outcomes, $e ^ { H ( p ) }$ with H the Shannon entropy, are 1626, 560 and 1324. In each run we draw $m \in \{ 6 4 , 2 5 6 , 1 0 2 4 , 4 0 9 6 \}$ samples from the target and hold out 20% of them as the validation sample of the controller.

## Model, loss and noise.

The Born machine is a hardware-efficient circuit with 10 layers of single-qubit Pauli rotations and CNOT, giving 360 parameters that each enter once through a Pauli rotation. Gradients are computed with the parameter-shift rule, so each update evaluates $P _ { \mathrm { c i r c } } = 7 2 1$ circuits. The loss is the MMD U-statistic with the Gaussian mixture kernel on the Hamming distance, with bandwidths $( 0 . 5 , 1 , 2 ) \times \sqrt { n / 5 }$ . Hardware noise consists of stochastic Pauli channels at the $L = 2 4 0$ noisy locations of the circuit, with per-location rates of mean $\bar { \epsilon } \in \lbrace 0 . 0 0 2 , 0 . 0 0 4 , 0 . 0 0 8 \rbrace$ shared by all runs. Simulations are implemented in JAX and run on CPUs.

## Training, budgets and compared methods.

Each run performs $T = 5 0 0$ updates with the Adam optimizer at the starting learning rate of 0.1 which decays over time. Budgets are counted in circuit executions without any rescaling by the PER cost. The unit budget $U = T P _ { \mathrm { c i r c } } \times 2 0 0 0 \approx 7 . 2 \times 1 0 ^ { 8 }$ corresponds to 2000 shots per circuit at every step, and we consider $B \in \{ 1 , 5 , 2 5 , 1 2 5 \} \times { U } .$ with each step receiving an even share of the remaining budget as in equation (48). We compare four methods at the same budget. Three keep a fixed noise level throughout training, namely no mitigation $( \gamma ~ = ~ \bar { \gamma } )$ , half mitigation $( \gamma = \bar { \gamma } / 2 )$ and full mitigation $( \gamma ~ = ~ 0 )$ . The fourth is GENIE, which compares the candidate levels $\gamma / \bar { \gamma } \in \lbrace 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \rbrace$ every $K = 2 5$ steps, starting at the first step, for 20 comparisons per run. It uses z = 1, λ = 1 and a probe fraction $\alpha = 0 . 0 2$ , and its probe executions are charged to its own budget. These controller settings were tuned once at $n = 1 0$ and transferred unchanged.

## Evaluation and statistics.

The grid comprises 3 targets, 3 error rates, 4 sample sizes and 4 budgets, giving 144 conditions, each run with three seeds for a total of 1728 runs. The figure of merit is the exact population loss $L _ { P } = \mathrm { M M D } _ { \kappa } ^ { 2 } ( p , q )$ of the output distribution $q$ of the final model. Within each condition, we average $L _ { P }$ over the seeds and report the relative regret 100 $( L _ { P } - L ^ { \star } ) / L ^ { \star }$ , where $L ^ { \star }$ is the smallest seed-averaged loss among the four methods. Because this regret is bounded below and skewed, we report both its mean and its median.

## Census experiment.

For the real-data experiment, we use the Adult census data [48]. Its training split of 32,561 people defines the target distribution $p ,$ and its test split of 16,281 people is used only to score the income questions. Each person is encoded in $n = 1 0$ bits, two Gray-coded bits each for age, education and weekly working hours, and one bit each for marital status, sex, a managerial or professional occupation, and income above 50K. The encoding was fixed before any training, and 845 of its 1024 strings occur in the data. The Born machine has 6 strongly entangling layers with 180 parameters, so $P _ { \mathrm { c i r c } } = 3 6 1$ , and its $L = 1 2 0$ noisy locations have the mean error rate of 1.6%. The kernel rule, the white-noise simulation, the T = 500 Adam updates and the controller settings follow the synthetic study, with a learning rate of max $\{ 0 . 0 0 5 , 0 . 1 \times 0 . 5 ^ { t / 1 5 0 } \}$ at step t. We draw $m \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ people and hold out 20% of them. The unit budget is $U = T P _ { \mathrm { c i r c } } \times 2 0 0 0 \approx 3 . 6 \times 1 0 ^ { 8 }$ executions, and $B \in \{ 1 , 5 , 2 5 , 1 2 5 , 6 2 5 \} \times U$ All scores are computed from the exact output distribution q of the final model at the noise level at which it is deployed. The census shares compare the probability that q assigns to an attribute with its share in $p .$

## Use of AI tools

AI tools were used to assist with language polishing and surveying related work.

## Data availability

Data generated and analyzed during the current study are available from the corresponding author upon reasonable request.

## Code availability

Code used for the current study is available from the corresponding author upon reasonable request.

## Author contributions

H.M.N. performed the theoretical analysis, developed the methods, and designed the experiments. Q.N. implemented and carried out the experiments. J.T.S. and W.X. contributed to the conceptual development of the work. M.T.T. supervised the project. All authors discussed the results and contributed to the manuscript.

## Competing interests

The authors declare no competing interests.

## References

[1] Perdomo-Ortiz, A., Benedetti, M., Realpe-Gómez, J., Biswas, R.: Opportunities and challenges for quantum-assisted machine learning in near-term quantum computers. Quantum Sci. Technol. 3, 030502 (2018)

[2] Bremner, M.J., Montanaro, A., Shepherd, D.J.: Average-case complexity versus approximate simulation of commuting quantum computations. Phys. Rev. Lett. 117, 080501 (2016) https://doi.org/10.1103/PhysRevLett.117.080501

[3] Arute, F., et al.: Quantum supremacy using a programmable superconducting processor. Nature 574, 505–510 (2019)

[4] Coyle, B., Mills, D., Danos, V., Kashefi, E.: The Born supremacy: quantum advantage and training of an Ising Born machine. npj Quantum Inf. 6, 60 (2020)

[5] Gao, X., Zhang, Z.-Y., Duan, L.-M.: A quantum machine learning algorithm based on generative models. Science Advances 4(12), 9004 (2018) https://doi.org/10. 1126/sciadv.aat9004 https://www.science.org/doi/pdf/10.1126/sciadv.aat9004

[6] Gao, X., Anschuetz, E.R., Wang, S.-T., Cirac, J.I., Lukin, M.D.: Enhancing generative models via quantum correlations. Phys. Rev. X 12, 021037 (2022) https://doi.org/10.1103/PhysRevX.12.021037

[7] Sweke, R., Seifert, J.-P., Hangleiter, D., Eisert, J.: On the quantum versus classical learnability of discrete distributions. Quantum 5, 417 (2021)

[8] Benedetti, M., Garcia-Pintos, D., Perdomo, O., Leyton-Ortega, V., Nam, Y., Perdomo-Ortiz, A.: A generative modeling approach for benchmarking and training shallow quantum circuits. npj Quantum Inf. 5, 45 (2019)

[9] Liu, J.-G., Wang, L.: Differentiable learning of quantum circuit Born machines. Phys. Rev. A 98, 062324 (2018)

[10] Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A.: A kernel two-sample test. J. Mach. Learn. Res. 13, 723–773 (2012)

[11] Rudolph, M.S., Lerch, S., Thanasilp, S., et al.: Trainability barriers and opportunities in quantum generative modeling. npj Quantum Inf. 10, 116 (2024) https://doi.org/10.1038/s41534-024-00902-0

[12] Rudolph, M.S., et al.: Generation of high-resolution handwritten digits with an ion-trap quantum computer. Phys. Rev. X 12, 031010 (2022)

[13] Kiss, O., Grossi, M., Kajomovitz, E., Vallecorsa, S.: Conditional Born machine for Monte Carlo event generation. Phys. Rev. A 106, 022612 (2022)

[14] Alcazar, J., Ghazi Vakili, M., Kalayci, C.B., Perdomo-Ortiz, A.: Enhancing combinatorial optimization with classical and quantum generative models. Nat. Commun. 15, 2761 (2024)

[15] Ghazi Vakili, M., Gorgulla, C., Snider, J., Nigam, A., Bezrukov, D., Varoli, D., Aliper, A., Polykovsky, D., Padmanabha Das, K.M., Cox III, H., Lyakisheva, A., Hosseini Mansob, A., Yao, Z., Bitar, L., Tahoulas, D., Čerina, D., Radchenko, E., Ding, X., Liu, J., Meng, F., Ren, F., Cao, Y., Stagljar, I., Aspuru-Guzik, A., Zhavoronkov, A.: Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors. Nat. Biotechnol. 43, 1954–1959 (2025) https: //doi.org/10.1038/s41587-024-02526-3

[16] Gili, K., Mauri, M., Perdomo-Ortiz, A.: Generalization metrics for practical

quantum advantage in generative models. Phys. Rev. Appl. 21, 044032 (2024)

[17] Abbas, A., Sutter, D., Zoufal, C., Lucchi, A., Figalli, A., Woerner, S.: The power of quantum neural networks. Nat. Comput. Sci. 1, 403–409 (2021)

[18] Banchi, L., Pereira, J., Pirandola, S.: Generalization in quantum machine learning: a quantum information standpoint. PRX Quantum 2, 040321 (2021)

[19] Caro, M.C., Huang, H.-Y., Cerezo, M., Sharma, K., Sornborger, A., Cincio, L., Coles, P.J.: Generalization in quantum machine learning from few training data. Nat. Commun. 13, 4919 (2022)

[20] Du, Y., Tu, Z., Yuan, X., Tao, D.: Efficient measure for the expressivity of variational quantum algorithms. Phys. Rev. Lett. 128, 080506 (2022)

[21] Kobayashi, M., Nakaji, K., Yamamoto, N.: Overfitting in quantum machine learning and entangling dropout. Quantum Mach. Intell. 4, 30 (2022) https: //doi.org/10.1007/s42484-022-00087-9

[22] Du, Y., Tu, Z., Wu, B., Yuan, X., Tao, D.: Power of quantum generative learning. Preprint at https://arxiv.org/abs/2205.04730 (2022)

[23] Gili, K., Hibat-Allah, M., Mauri, M., Ballance, C., Perdomo-Ortiz, A.: Do quantum circuit Born machines generalize? Quantum Sci. Technol. 8, 035021 (2023)

[24] Raj, S., Mathur, N., Perdomo-Ortiz, A.: “Train classical, deploy quantum" requires rethinking generalization. Preprint at https://arxiv.org/abs/2608.31117 (2026)

[25] Bishop, C.M.: Training with noise is equivalent to Tikhonov regularization. Neural Comput. 7(1), 108–116 (1995)

[26] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: a simple way to prevent neural networks from overfitting. J. Mach. Learn. Res. 15(56), 1929–1958 (2014)

[27] Hardt, M., Recht, B., Singer, Y.: Train faster, generalize better: stability of stochastic gradient descent. In: Proc. 33rd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 48, pp. 1225–1234 (2016)

[28] Bu, K., Koh, D.E., Li, L., Luo, Q., Zhang, Y.: Effects of quantum resources and noise on the statistical complexity of quantum circuits. Quantum Sci. Technol. 8, 025013 (2023)

[29] Khanal, B., Rivas, P.: Data-dependent generalization bounds for parameterized quantum models under noise. J. Supercomput. 81, 611 (2025) https://doi.org/

[30] Heyraud, V., Li, Z., Denis, Z., Le Boité, A., Ciuti, C.: Noisy quantum kernel machines. Phys. Rev. A 106, 052421 (2022) https://doi.org/10.1103/PhysRevA. 106.052421

[31] Zhang, Z., Jia, Z., Li, X., Dong, Y.: A theory of finite-noise optima and generalization in quantum machine learning. Preprint at https://arxiv.org/abs/2608.24229 (2026)

[32] Kuzmin, V., Somogyi, W., Pankovets, E., Melnikov, A.: Method for noise-induced regularization in quantum neural networks. Adv. Quantum Technol. 8(12), 00603 (2025) https://doi.org/10.1002/qute.202400603

[33] Scala, F., Guarnieri, G., Lucchi, A.: Noise-induced equalization in quantum learning models. Quantum Sci. Technol. 11, 045015 (2026) https://doi.org/10.1088/ 2058-9565/ae9265

[34] Wang, S., Fontana, E., Cerezo, M., Sharma, K., Sone, A., Cincio, L., Coles, P.J.: Noise-induced barren plateaus in variational quantum algorithms. Nat. Commun. 12, 6961 (2021)

[35] Temme, K., Bravyi, S., Gambetta, J.M.: Error mitigation for short-depth quantum circuits. Phys. Rev. Lett. 119, 180509 (2017) https://doi.org/10.1103/ PhysRevLett.119.180509

[36] Mari, A., Shammah, N., Zeng, W.J.: Extending quantum probabilistic error cancellation by noise scaling. Phys. Rev. A 104, 052607 (2021)

[37] Takagi, R., Endo, S., Minagawa, S., Gu, M.: Fundamental limits of quantum error mitigation. npj Quantum Inf. 8, 114 (2022) https://doi.org/10.1038/ s41534-022-00618-z

[38] Quek, Y., Stilck França, D., Khatri, S., Meyer, J.J., Eisert, J.: Exponentially tighter bounds on limitations of quantum error mitigation. Nat. Phys. 20, 1648– 1658 (2024) https://doi.org/10.1038/s41567-024-02536-7

[39] Wallman, J.J., Emerson, J.: Noise tailoring for scalable quantum computation via randomized compiling. Phys. Rev. A 94, 052325 (2016) https://doi.org/10.1103/ PhysRevA.94.052325

[40] Berg, E., Minev, Z.K., Kandala, A., Temme, K.: Probabilistic error cancellation with sparse pauli-lindblad models on noisy quantum processors. Nature Physics 19(8), 1116–1121 (2023) https://doi.org/10.1038/s41567-023-02042-2

[41] Bartlett, P.L., Mendelson, S.: Rademacher and Gaussian complexities: risk bounds and structural results. J. Mach. Learn. Res. 3, 463–482 (2002)

[42] Dalzell, A.M., Hunter-Jones, N., Brandão, F.G.S.L.: Random quantum circuits transform local noise into global white noise. Commun. Math. Phys. 405, 78 (2024) https://doi.org/10.1007/s00220-024-04958-z

[43] Vovrosh, J., Khosla, K.E., Greenaway, S., Self, C., Kim, M.S., Knolle, J.: Simple mitigation of global depolarizing errors in quantum simulations. Phys. Rev. E 104, 035309 (2021) https://doi.org/10.1103/PhysRevE.104.035309

[44] Urbanek, M., Nachman, B., Pascuzzi, V.R., He, A., Bauer, C.W., Jong, W.A.: Mitigating depolarizing noise on quantum computers with noise-estimation circuits. Phys. Rev. Lett. 127, 270502 (2021) https://doi.org/10.1103/PhysRevLett. 127.270502

[45] Cai, Z., Babbush, R., Benjamin, S.C., Endo, S., Huggins, W.J., Li, Y., McClean, J.R., O'Brien, T.E.: Quantum error mitigation. Rev. Mod. Phys. 95, 045005 (2023) https://doi.org/10.1103/RevModPhys.95.045005

[46] Bottou, L., Curtis, F.E., Nocedal, J.: Optimization methods for large-scale machine learning. SIAM Review 60(2), 223–311 (2018) https://doi.org/10.1137/ 16M1080173

[47] Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: 3rd International Conference on Learning Representations (ICLR) (2015)

[48] Becker, B., Kohavi, R.: Adult. UCI Machine Learning Repository (1996). https: //doi.org/10.24432/C5XW20

## Supplementary Information

These Supplementary Notes give complete proofs of the results stated in the main text. Supplementary Note 1 collects the Walsh-Fourier facts that underlie the spectral analysis. Supplementary Note 2 proves the generalization bounds, Supplementary Note 3 proves the results on capped error reduction, and Supplementary Note 4 proves the results on budget-aware noise control. Each result is restated with its number from the main text before its proof.

## Supplementary Note 1. Walsh-Fourier coordinates

The coordinates in which quantum noise acts most simply are the characters of $\mathbb { Z } _ { 2 } ^ { n }$ For $A \subseteq [ n ]$ define the Walsh character and, for a distribution q and a function $\phi$ on $x ,$ the associated coefficients

$$
\chi _ { A } ( \boldsymbol { x } ) : = ( - 1 ) ^ { \sum _ { i \in A } x _ { i } } , \qquad \widehat { q } ( A ) : = \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { q } } \big [ \chi _ { A } ( \boldsymbol { x } ) \big ] , \qquad \widehat { \phi } ( A ) : = 2 ^ { - n } \sum _ { t \in \mathcal { X } } \phi ( t ) \chi _ { A } ( t ) .\tag{S1}
$$

Characters are multiplicative, $\chi _ { A } ( x \oplus y ) = \chi _ { A } ( x ) \chi _ { A } ( y )$ , and orthonormal under the uniform measure. Three facts will be used.

## (i) Coefficients are Pauli observables.

Since $Z _ { A } \left| x \right. = \chi _ { A } ( x ) \left| x \right.$ for $Z _ { A } : = \otimes _ { i \in A } Z _ { i }$

$$
{ \widehat { q _ { \theta } ^ { \mathcal { N } } } } ( A ) = \operatorname { T r } \left[ \rho _ { \theta } ^ { \mathcal { N } } Z _ { A } \right] .\tag{S2}
$$

The Walsh coefficients of the output distribution are exactly the computational-basis Pauli correlators of the output state, so $| A |$ is the order of the correlation described by the mode.

## (ii) Parseval links the spectrum to the collision probability.

Writing coll $\begin{array} { r } { \iota ( q ) : = \sum _ { x } q ( x ) ^ { 2 } } \end{array}$ for the probability that two independent draws from $q$ agree,

$$
\sum _ { A \subseteq [ n ] } { \widehat { q } } ( A ) ^ { 2 } = 2 ^ { n } \operatorname { c o l l } ( q ) , \qquad \sum _ { A \subseteq [ n ] } { \widehat { \phi } } ( A ) ^ { 2 } = \mathbb { E } _ { x , y \sim u } \left[ \phi ( x \oplus y ) ^ { 2 } \right] .\tag{S3}
$$

The uniform distribution attains col $\operatorname { l } ( u ) = 2 ^ { - n }$ , so $2 ^ { n } \mathrm { c o l l } ( q )$ measures how far $q$ is from flat and, by equation (S3), simultaneously budgets the total spectral mass available to the generator. This is the quantity through which the ansatz enters our complexity bound.

## (iii) Translation-invariant kernels are diagonal.

We call κ translation invariant if

$$
\kappa ( x , y ) = \phi ( x \oplus y ) \qquad { \mathrm { f o r ~ s o m e ~ } } \phi : \mathcal { X } \to \mathbb { R } , \quad \phi ( \mathbf { 0 } ) = 1 ,\tag{S4}
$$

the normalization being without loss of generality. The Gram matrix $G$ with entries $G _ { x y } = \kappa ( x , y )$ is then diagonalized by the characters, with $G \chi _ { A } = 2 ^ { n } { \widehat { \phi } } ( A ) \chi _ { A } ,$ SO κ is positive semidefinite if and only if $\widehat { \phi } ( A ) \geq 0$ for all $A ,$ and in that case $\textstyle \sum _ { A } { \widehat { \phi } } ( A ) =$ $\phi ( \mathbf { 0 } ) = 1$ . The Gaussian kernel on the Hamming distance used in the main text, $\begin{array} { r } { \kappa ( x , y ) = \frac { 1 } { c } \sum _ { i = 1 } ^ { c } \exp [ - d _ { H } ( x , y ) / ( 2 \tau _ { i } ^ { 2 } ) ] } \end{array}$ , is of this form with $\begin{array} { r } { \phi ( t ) = \frac { 1 } { c } \sum _ { i = 1 } ^ { c } a _ { i } ^ { | t | } } \end{array}$ and $a _ { i } = e ^ { - 1 / ( 2 \tau _ { i } ^ { 2 } ) }$

Under equation (S4), multiplicativity of the characters gives the witness its spectral form,

$$
h _ { \theta } ^ { \mathcal { N } } ( x ) = \sum _ { A \subseteq [ n ] } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { \mathcal { N } } } ( A ) \chi _ { A } ( x ) ,\tag{S5}
$$

so the witness is a kernel-weighted superposition of the generator's Pauli correlators. In particular the uniform distribution has $\widehat { \boldsymbol { u } } ( \boldsymbol { A } ) = 0$ for $A \neq \emptyset$ , so its witness $h _ { u } \equiv \dot { \phi ( \cal { O } ) }$ is constant. By equation (10), the class may therefore be recentered at no cost,

$$
( P _ { m } - P ) h _ { \theta } ^ { N } = ( P _ { m } - P ) \bar { h } _ { \theta } ^ { N } , \qquad \bar { h } _ { \theta } ^ { N } : = h _ { \theta } ^ { N } - h _ { u } = \sum _ { A \ne \infty } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { N } } ( A ) \chi _ { A } .\tag{S6}
$$

We write $\bar { \mathcal { H } } ^ { \mathcal { N } }$ and $\bar { \mathcal { H } } _ { + } ^ { \mathcal { N } }$ for the corresponding centered and symmetrized classes, and $\bar { h } _ { \theta } : = \bar { h } _ { \theta } ^ { 0 }$ in the noiseless case.

## Supplementary Note 2. Proofs of the generalization bounds

We first prove the reduction to the learned witness, which underlies both generalization theorems, and then the noise-agnostic bound, the noiseless complexity bound, and the bounds under scrambled local Pauli noise.

Lemma 1 (Reduction to the learned witness). Let $m , N \ge 2 _ { \mathrm { { \ell } } }$ and let $\mathcal { A } _ { \mathcal { N } }$ be any possibly randomized learning algorithm. If the evaluation samples $\mathcal { V }$ are conditionally independent of S and the training randomness given $\hat { \theta } _ { \mathcal { N } }$ , then

$$
\Delta _ { \mathcal { N } } ( m ) = 2 \mathbb { E } _ { \mathcal { S } , A _ { \mathcal { N } } } \left[ ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } \right] .\tag{36}
$$

Proof of Lemma 1 For brevity, we condition on $s$ and $\hat { \theta } _ { \mathcal { N } }$ and write

$$
q : = q _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } , \qquad h : = h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } .\tag{S7}
$$

Under this conditioning, the learned distribution q and the witness h are fixed, while

$$
\mathcal { V } = ( y _ { 1 } , \dots , y _ { N } ) \sim q ^ { \otimes N } .\tag{S8}
$$

We set the population target-only and generator-only terms by

$$
A _ { p } : = \mathbb { E } _ { x , x ^ { \prime } \sim p } \left[ \kappa ( x , x ^ { \prime } ) \right] , \qquad A _ { q } : = \mathbb { E } _ { y , y ^ { \prime } \sim q } \left[ \kappa ( y , y ^ { \prime } ) \right] .\tag{S9}
$$

Since $P h = \mathbb { E } _ { x \sim p , y \sim q } [ \kappa ( x , y ) ]$ by the definition of the witness, equation (8), the population risk can then be written as

$$
R _ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) = A _ { q } + A _ { p } - 2 P h .\tag{S10}
$$

Similarly, we write the empirical risk as

$$
\widehat { R } _ { S , y } ^ { N } ( \widehat { \theta } _ { N } ) = U _ { y } + U _ { S } - 2 \widehat { C } ,\tag{S11}
$$

where

$$
U _ { S } : = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq i ^ { \prime } } \kappa ( x _ { i } , x _ { i ^ { \prime } } ) ,\tag{S12}
$$

$$
U _ { \mathcal { Y } } : = \frac { 1 } { N ( N - 1 ) } \sum _ { j \neq j ^ { \prime } } \kappa ( y _ { j } , y _ { j ^ { \prime } } ) ,\tag{S13}
$$

$$
\widehat { C } : = \frac { 1 } { m N } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { N } \kappa ( x _ { i } , y _ { j } ) .\tag{S14}
$$

Because the evaluation samples are conditionally i.i.d. from $q ,$ every pair $( y _ { j } , y _ { j ^ { \prime } } )$ with $j \neq j ^ { \prime }$ has joint distribution $q \otimes q .$ Therefore,

$$
\mathbb { E } _ { \mathcal { Y } } \left[ U _ { \mathcal { Y } } \Big | \mathcal { S } , \hat { \theta } _ { \mathcal { N } } \right] = \frac { 1 } { N ( N - 1 ) } \sum _ { j \neq j ^ { \prime } } \mathbb { E } _ { y _ { j } , y _ { j ^ { \prime } } \sim q } \left[ \kappa ( y _ { j } , y _ { j ^ { \prime } } ) \right]\tag{S15}
$$

$$
= { \frac { 1 } { N ( N - 1 ) } } \sum _ { j \neq j ^ { \prime } } A _ { q } = A _ { q } .\tag{S16}
$$

For the empirical coupling term, each $x _ { i }$ is fixed under the conditioning, while every $y _ { j }$ is distributed according to $q .$ Hence,

$$
\mathbb { E } _ { \mathcal { Y } } \left[ \widehat { C } \Big | { S } , \widehat { \theta } _ { \mathcal { N } } \right] = \frac { 1 } { m N } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { N } \mathbb { E } _ { y _ { j } \sim q } \left[ \kappa ( x _ { i } , y _ { j } ) \right]\tag{S17}
$$

$$
= \frac { 1 } { m N } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { N } h ( x _ { i } )\tag{S18}
$$

$$
= { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } h ( x _ { i } ) = P _ { m } h .\tag{S19}
$$

Substituting equations (S16) and (S19) into equations (S10) and (S11) yields

$$
\mathbb { E } _ { \mathcal { V } } \left[ R _ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) - \widehat { R } _ { \mathcal { S } , \mathcal { V } } ^ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) \Big | \mathcal { S } , \hat { \theta } _ { \mathcal { N } } \right]
$$

$$
= A _ { q } + A _ { p } - 2 P h - \left( A _ { q } + U _ { S } - 2 P _ { m } h \right)\tag{S20}
$$

$$
= A _ { p } - U _ { S } + 2 ( P _ { m } - P ) h .\tag{S21}
$$

It remains to average over the target sample and the learning algorithm. Since $x _ { i }$ and $x _ { i ^ { \prime } }$ are independent draws from $p$ whenever $i \neq i ^ { \prime }$

$$
\mathbb { E } _ { S } [ U _ { S } ] = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq i ^ { \prime } } \mathbb { E } _ { x _ { i } , x _ { i ^ { \prime } } \sim p } \left[ \kappa ( x _ { i } , x _ { i ^ { \prime } } ) \right]\tag{S22}
$$

$$
= { \frac { 1 } { m ( m - 1 ) } } \sum _ { i \neq i ^ { \prime } } A _ { p } = A _ { p } .\tag{S23}
$$

Thus, ${ \mathbb E } _ { S } [ A _ { p } - U _ { S } ] = 0$ . Taking the remaining expectations in equation (S21) gives

$$
\Delta _ { \mathcal { N } } ( m ) = 2 \mathbb { E } _ { \mathcal { S } , \mathcal { A } _ { \mathcal { N } } } \left[ ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } \right] ,\tag{S24}
$$

which proves the claim.

Theorem 1 (Noise-agnostic generalization bound). Let m $\geq ~ 2$ and $\mathcal { A } _ { \mathcal { N } }$ be any possibly randomized learning algorithm with the noise model $\mathcal { N } _ { i }$ ,we have

$$
| \Delta _ { \mathcal { N } } ( m ) | \leq 4 \Re \left( \mathcal { W } _ { \pm } ^ { \mathcal { N } } \right) .\tag{14}
$$

Proof of Theorem 1 Using Lemma 1,

$$
\left| \Delta _ { \mathcal { N } } ( m ) \right| \leq 2 \mathbb { E } _ { \mathcal { S , A _ { N } } } \left[ \left| ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } \right| \right]\tag{S25}
$$

$$
\leq 2 \mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { h \in \mathcal { W } ^ { N } } | ( P _ { m } - P ) h | \right]\tag{S26}
$$

$$
\leq 4 \Re \left( \mathcal { W } _ { \pm } ^ { N } \right) ,\tag{S27}
$$

where the last step is the symmetrization bound in equation (37).

Lemma 2 (Complexity of the noiseless witness class). Under Assumption 1, and $f o r$ a given $C \geq 1$ such that sup $\begin{array} { r } { \mathfrak { p } _ { \theta \in \Theta } 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } \le C } \end{array}$ , we have:

$$
\Re ( \hat { \mathcal { W } } _ { \pm } ^ { 0 } ) \le \frac { \Lambda _ { C } } { \sqrt { m } } ,\tag{38}
$$

where $\Lambda _ { C }$ is the constant deined in Theorem ${ \it 2 . }$

Proof of Lemma 2 We first translate the concentration condition into a bound on the nonconstant Fourier moments of the generator. Under the convention in equation (15), Parseval's identity gives

$$
\sum _ { A \subseteq [ n ] } \widehat { q _ { \theta } ^ { 0 } } ( A ) ^ { 2 } = 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } .\tag{S28}
$$

The empty coefficient satisfies

$$
\widehat { q _ { \theta } ^ { 0 } } ( \emptyset ) = \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { q } _ { \theta } ^ { 0 } } [ \chi _ { \mathcal { O } } ( \boldsymbol { x } ) ] = 1 .\tag{S29}
$$

Therefore, removing the empty mode from equation (S28), we obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { A \neq \emptyset } \widehat { q _ { \theta } ^ { 0 } } ( A ) ^ { 2 } = 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } - 1 } \\ { \leq { C } - 1 \qquad \mathrm { ~ f o r ~ e v e r y ~ } \theta \in \Theta . } \end{array}\tag{S30}
$$

Thus, centering removes the constant Fourier coefficient and changes the available spectral budget from $C$ to $C - 1$

Now fix an arbitrary training sample $\mathcal { S } = ( x _ { 1 } , \dots , x _ { m } )$ and let $\sigma _ { 1 } , \ldots , \sigma _ { m }$ be independent Rademacher variables. For a class ${ \mathcal { F } } _ { : }$ we write

$$
{ \widehat { \mathfrak { R } } } _ { S } ( { \mathcal { F } } ) : = \mathbb { E } _ { \sigma } { \Big [ } \operatorname* { s u p } _ { f \in { \mathcal { F } } } { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \sigma _ { i } f ( x _ { i } ) { \Big ] }\tag{S31}
$$

for its empirical Rademacher complexity, so that $\Re ( \mathcal { F } ) = \mathbb { E } _ { S } [ \widehat { \Re } _ { S } ( \mathcal { F } ) ]$ by equation (12). For each nonempty $A \subseteq [ n ]$ , we define

$$
g _ { A } : = \frac 1 m \sum _ { i = 1 } ^ { m } \sigma _ { i } \chi _ { A } ( x _ { i } ) .\tag{S32}
$$

Using the spectral representation of the centered witness in equation (16), we obtain

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \sigma _ { i } \bar { h } _ { \theta } ( x _ { i } ) = \sum _ { A \ne \emptyset } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { 0 } } ( A ) g _ { A } .\tag{S33}
$$

Because $\bar { \mathcal { W } } _ { \pm } ^ { 0 } = \bar { \mathcal { W } } ^ { 0 } \cup ( - \bar { \mathcal { W } } ^ { 0 } )$ , its empirical Rademacher complexity is

$$
\widehat { \mathfrak { R } } _ { S } \left( \bar { \mathcal { W } } _ { \pm } ^ { 0 } \right) = \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { \theta \in \Theta } \left[ \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) q _ { \theta } ^ { \widehat { 0 } } ( A ) g _ { A } \right] \right] .\tag{S34}
$$

For every $\theta ,$ based on Cauchy-Schwarz and equation (S30), we obtain

$$
\left| \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { 0 } } ( A ) g _ { A } \right| \leq \left( \sum _ { A \neq \emptyset } \widehat { q _ { \theta } ^ { 0 } } ( A ) ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) ^ { 2 } g _ { A } ^ { 2 } \right) ^ { 1 / 2 }
$$

$$
\leq \sqrt { C - 1 } \left( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) ^ { 2 } g _ { A } ^ { 2 } \right) ^ { 1 / 2 } .\tag{S35}
$$

The final expression is independent of θ. Hence, by Jensen's inequality,

$$
\widehat { \mathfrak { R } } _ { \mathcal { S } } \bigl ( \bar { \mathcal { W } } _ { \pm } ^ { 0 } \bigr ) \leq \sqrt { C - 1 } \left( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) ^ { 2 } \mathbb { E } _ { \sigma } [ g _ { A } ^ { 2 } ] \right) ^ { 1 / 2 } .\tag{S36}
$$

It remains to evaluate $\mathbb { E } _ { \sigma } [ g _ { A } ^ { 2 } ]$ . Independence and centering of the Rademacher variables eliminate all cross terms, while $\chi _ { A } ( x _ { i } ) ^ { 2 } = 1$ . Consequently,

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \sigma } [ g _ { A } ^ { 2 } ] = \frac { 1 } { m ^ { 2 } } \sum _ { i , i ^ { \prime } = 1 } ^ { m } \mathbb { E } _ { \sigma } [ \sigma _ { i } \sigma _ { i ^ { \prime } } ] \chi _ { A } ( x _ { i } ) \chi _ { A } ( x _ { i ^ { \prime } } ) } \\ { \displaystyle = \frac { 1 } { m ^ { 2 } } \sum _ { i = 1 } ^ { m } \chi _ { A } ( x _ { i } ) ^ { 2 } = \frac { 1 } { m } . } \end{array}\tag{S37}
$$

Substituting this identity into equation (S36) gives, for every fixed sample $s$

$$
\widehat { \mathfrak { R } } _ { S } ( \bar { \mathcal { W } } _ { \pm } ^ { 0 } ) \leq \sqrt { \frac { ( C - 1 ) \sum _ { A \neq \infty } \widehat { \phi } ( A ) ^ { 2 } } { m } } .\tag{S38}
$$

For the second branch of $\Lambda _ { C }$ , we apply the Cauchy-Schwarz inequality with the weights $\widehat { \phi } ( A ) \geq 0$ instead such that:

$$
\left| \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { 0 } } ( A ) g _ { A } \right| \leq \Big ( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) \widehat { q _ { \theta } ^ { 0 } } ( A ) ^ { 2 } \Big ) ^ { 1 / 2 } \Big ( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) g _ { A } ^ { 2 } \Big ) ^ { 1 / 2 }\tag{S39}
$$

$$
\leq \big ( 1 - \widehat { \phi } ( \emptyset ) \big ) ^ { 1 / 2 } \Big ( \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) g _ { A } ^ { 2 } \Big ) ^ { 1 / 2 } ,\tag{S40}
$$

where the last step uses $| \widehat { q _ { \theta } ^ { 0 } } ( A ) | \leq 1$ and $\textstyle \sum _ { A } { \widehat { \phi } } ( A ) = \phi ( \mathbf { 0 } ) = 1$ . The right-hand side does not depend on $\theta ,$ so Jensen's inequality and equation (S37) give:

$$
\widehat { \mathfrak { R } } _ { \mathcal { S } } \bigl ( \bar { \mathcal { W } } _ { \pm } ^ { 0 } \bigr ) \leq \bigl ( 1 - \widehat { \phi } ( \varpi ) \bigr ) ^ { 1 / 2 } \Bigl ( \frac { 1 } { m } \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) \Bigr ) ^ { 1 / 2 } = \frac { 1 - \widehat { \phi } ( \emptyset ) } { \sqrt { m } } .\tag{S41}
$$

Both bounds hold for every fixed sample, and so does their minimum $\Lambda _ { C } / \sqrt { m }$ .Taking expectation over $s$ proves equation (38). □

Theorem 2 (Generalization under scrambled local Pauli noise). Suppose that Assumptions 1 and 2 hold and that, for some $C \geq 1$ 2

$$
\operatorname* { s u p } _ { \theta \in \Theta } 2 ^ { n } \sum _ { x \in \mathcal { X } } q _ { \theta } ^ { 0 } ( x ) ^ { 2 } \leq C .\tag{19}
$$

Let $m , N \ge 2$ . Then the expected generalization gap satisfies

$$
| \Delta _ { \mathcal { N } } ( m ) | \leq 4 ( 1 - \gamma ) \frac { \Lambda _ { C } } { \sqrt { m } } + \frac { 2 \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } ,\tag{20}
$$

where $\begin{array} { r } { \Lambda _ { C } : = \operatorname* { m i n } \bigr \{ \sqrt { ( C - 1 ) \sum _ { A \neq \emptyset } \widehat { \phi } ( A ) ^ { 2 } } , 1 - \widehat { \phi } ( \emptyset ) \bigr \} } \end{array}$

Corollary 1 (High-probability generalization bound). Under the conditions of Theorem 2, for every $\beta \in ( 0 , 1 )$ , with probability at least $1 - \beta$ over $s , A _ { \mathcal { N } } ,$ and $\mathcal { V } _ { i }$

$$
\begin{array} { r l } { \displaystyle \left. R _ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) - \widehat { R } _ { \mathcal { S } , \mathcal { Y } } ^ { \mathcal { N } } ( \hat { \theta } _ { \mathcal { N } } ) \right. \leq 4 ( 1 - \gamma ) \displaystyle \frac { \Lambda _ { C } } { \sqrt { m } } + 2 ( 1 - \gamma ) \sqrt { \displaystyle \frac { 2 \log ( 4 / \beta ) } { m } } } & { } \\ { + \displaystyle \frac { 2 \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } \left( 1 + \sqrt { 2 \log ( 4 / \beta ) } \right) + \sqrt { \displaystyle \frac { \log ( 8 / \beta ) } { 2 \lfloor m / 2 \rfloor } } + \sqrt { \displaystyle \frac { 8 \log ( 8 / \beta ) } { N } } . } & { } \end{array}\tag{21}
$$

Proof of Theorem 2 and Corollary 1 Let

$$
\mu _ { P _ { m } } : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \kappa ( x _ { i } , \cdot ) ,
$$

$$
\mu _ { \mathcal { P } } : = \mathbb { E } _ { x \sim p } \left[ \kappa ( x , \cdot ) \right]\tag{S42}
$$

denote the empirical and population kernel mean embeddings of the target distribution. By the reproducing property, for every $\theta \in \Theta$ ，

$$
( P _ { m } - P ) e _ { \theta } = \langle e _ { \theta } , \mu _ { P _ { m } } - \mu _ { P } \rangle _ { \mathcal { H } } .\tag{S43}
$$

Cauchy-Schwarz and equation (40) therefore give

$$
\operatorname* { s u p } _ { \theta \in \Theta } | ( P _ { m } - P ) e _ { \theta } | \leq \varepsilon _ { \mathrm { P a u l i } } \| \mu _ { P _ { m } } - \mu _ { p } \| _ { \mathcal { H } } .\tag{S44}
$$

Because $\kappa ( x , x ) = 1$ , independence of the target samples gives

$$
\begin{array} { r l } & { \mathbb { E } _ { S } \left[ \| \mu _ { P _ { m } } - \mu _ { p } \| _ { \mathcal { H } } ^ { 2 } \right] = \frac { 1 } { m } \mathbb { E } _ { x \sim p } \left[ \| \kappa ( x , \cdot ) - \mu _ { p } \| _ { \mathcal { H } } ^ { 2 } \right] } \\ & { \qquad = \frac { 1 } { m } \left( \mathbb { E } _ { x \sim p } [ \kappa ( x , x ) ] - \| \mu _ { p } \| _ { \mathcal { H } } ^ { 2 } \right) \leq \frac { 1 } { m } . } \end{array}\tag{S45}
$$

Jensen's inequality then implies

$$
\mathbb { E } _ { S } \left[ \lVert \mu _ { P _ { m } } - \mu _ { p } \rVert _ { \mathcal { H } } \right] \leq \frac { 1 } { \sqrt { m } } .\tag{S46}
$$

Expected bound (Theorem 2).

By Lemma 1,

$$
\Delta _ { \mathcal { N } } ( m ) = 2 \mathbb { E } _ { \boldsymbol { S } , A _ { N } } \left[ ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } \right] .\tag{S47}
$$

Using equation (39) and the fact that $P _ { m } - P$ annihilates constants, we obtain

$$
\begin{array} { r l } & { | \Delta _ { \mathcal { N } } ( m ) | \leq 2 ( 1 - \gamma ) \mathbb { E } _ { S } \left[ \underset { \theta \in \Theta } { \operatorname* { s u p } } \left| ( P _ { m } - P ) \bar { h } _ { \theta } \right| \right] } \\ & { ~ + \ 2 \mathbb { E } _ { S } \left[ \underset { \theta \in \Theta } { \operatorname* { s u p } } \left| ( P _ { m } - P ) e _ { \theta } \right| \right] . } \end{array}\tag{S48}
$$

Standard symmetrization $\mathrm { g i }$ ves

$$
\mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { \theta \in \Theta } \big | ( P _ { m } - P ) \bar { h } _ { \theta } \big | \right] \leq 2 \Re \big ( \bar { \mathcal { W } } _ { \pm } ^ { 0 } \big ) .\tag{S49}
$$

equations (S44) and (S46) give

$$
\mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { \theta \in \Theta } | ( P _ { m } - P ) e _ { \theta } | \right] \leq \frac { \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } .\tag{S50}
$$

Applying Lemma 2 proves equation (20).

## High-probability bound (Corollary 1).

Define the target-only U-statistic

$$
U _ { S } : = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq i ^ { \prime } } \kappa ( x _ { i } , x _ { i ^ { \prime } } )\tag{S51}
$$

and, for fixed $s$ and $\theta ,$ collect the generator-dependent terms in

$$
\begin{array} { r } { G _ { \mathcal { S } , \theta } ( \mathcal { V } ) : = \frac { 1 } { N ( N - 1 ) } \displaystyle \sum _ { j \neq j ^ { \prime } } \kappa ( y _ { j } , y _ { j ^ { \prime } } ) } \\ { - \frac { 2 } { m N } \displaystyle \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { N } \kappa ( x _ { i } , y _ { j } ) . } \end{array}\tag{S52}
$$

Conditioning on $s$ and ${ \hat { \theta } } _ { \mathcal { N } }$ , the fresh-sample assumption gives

$$
\begin{array} { r l } & { \quad R _ { \mathcal { N } } ( \widehat { \theta } _ { \mathcal { N } } ) - \widehat { R } _ { S , \mathcal { V } } ^ { \mathcal { N } } ( \widehat { \theta } _ { \mathcal { N } } ) } \\ & { = \underbrace { \mathbb { E } _ { x , x ^ { \prime } \sim p } [ \kappa ( x , x ^ { \prime } ) ] - U _ { S } } _ { \mathrm { t a r g e t ~ U - s t a t i s t i c ~ f l u c t u a t i o n } } + 2 ( P _ { m } - P ) h _ { \widehat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } } } \end{array}
$$

$$
+ \underbrace { \mathbb { E } _ { \mathcal { Y } } [ G _ { S , \hat { \theta } _ { \mathcal { N } } } ( \mathcal { Y } ) | \mathcal { S } , \hat { \theta } _ { \mathcal { N } } ] - G _ { S , \hat { \theta } _ { \mathcal { N } } } ( \mathcal { Y } ) } _ { \mathrm { g e n e r a t o r - s a m p l e ~ f l u c t u a t i o n } } .\tag{S53}
$$

We first control the noisy witness term. From equation (39),

$$
\begin{array} { r l r } {  { 2  ( P _ { m } - P ) h _ { \hat { \theta } _ { \mathcal { N } } } ^ { \mathcal { N } }  \leq 2 ( 1 - \gamma ) \displaystyle \operatorname* { s u p } _ { \theta \in \Theta }  ( P _ { m } - P ) \bar { h } _ { \theta }  } } \\ & { } & { + \ 2 \varepsilon _ { \mathrm { P a u l i } }  \mu _ { P _ { m } } - \mu _ { p }  _ { \mathcal { H } } . } \end{array}\tag{S54}
$$

Since $0 \leq h _ { \theta } ^ { 0 } \leq 1$ and ${ \widehat { \phi } } ( \emptyset ) \in [ 0 , 1 ]$ , a standard Rademacher deviation inequality gives, with probability at least $1 - \beta / 4$

$$
\operatorname* { s u p } _ { \theta \in \Theta } \left| ( P _ { m } - P ) \bar { h } _ { \theta } \right| \leq 2 \Re ( \bar { \mathcal { W } } _ { \pm } ^ { 0 } ) + \sqrt { \frac { 2 \log ( 4 / \beta ) } { m } } .\tag{S55}
$$

Changing one target sample changes $\| \mu _ { P _ { m } } \ - \ \mu _ { p } \| _ { \mathcal { H } }$ by at most $2 / m$ . Therefore, McDiarmid's inequality and equation (S46) imply that, with probability at least $1 - \beta / 4 ,$

$$
\| \mu _ { P _ { m } } - \mu _ { P } \| _ { \mathcal { H } } \leq \frac { 1 } { \sqrt { m } } + \sqrt { \frac { 2 \log ( 4 / \beta ) } { m } } .\tag{S56}
$$

Combining these two inequalities with Lemma 2 bounds the noisy witness term by

$$
\begin{array} { l } { { \displaystyle 4 ( 1 - \gamma ) \frac { \Lambda _ { C } } { \sqrt { m } } + 2 ( 1 - \gamma ) \sqrt { \frac { 2 \log ( 4 / \beta ) } { m } } } } \\ { { \displaystyle ~ + \frac { 2 \varepsilon _ { \mathrm { P a u l i } } } { \sqrt { m } } \left( 1 + \sqrt { 2 \log ( 4 / \beta ) } \right) . } } \end{array}\tag{S57}
$$

Next, Hoeffding's inequality for order-two U-statistics gives, with probability at least $1 - \beta / 4$

$$
\left| \mathbb { E } _ { x , x ^ { \prime } \sim p } [ \kappa ( x , x ^ { \prime } ) ] - U _ { S } \right| \leq \sqrt { \frac { \log ( 8 / \beta ) } { 2 \lfloor m / 2 \rfloor } } .\tag{S58}
$$

Finally, changing one generator sample changes $G _ { S , \theta } ( \mathcal { V } )$ by at most $4 / N$ . Conditional McDiarmid's inequality therefore gives, with conditional probability at least $1 - \beta / 4$ 2

$$
\left| \mathbb { E } _ { \mathcal { V } } \left[ G _ { S , \hat { \theta } _ { \mathcal { N } } } ( \mathcal { V } ) \left| \mathcal { S } , \hat { \theta } _ { \mathcal { N } } \right. \right] - G _ { S , \hat { \theta } _ { \mathcal { N } } } ( \mathcal { V } ) \right| \leq \sqrt { \frac { 8 \log ( 8 / \beta ) } { N } } .\tag{S59}
$$

Combining these four events and applying a union bound proves equation (21).

## Supplementary Note 3. Proofs for capped error reduction

This note proves the cost of a mitigation profile and the optimality of the capped allocation.

Lemma 3 (Cost of a mitigation profile). Suppose that $\epsilon _ { k } < 1 / 2$ and $0 \le r _ { k } \le \epsilon _ { k }$ for every k. The circuit-level quasiprobability normalization required to reduce ∈ to r satisfies

$$
\log \Gamma ( \mathbf { r } ) = 2 \left\| \boldsymbol { \epsilon } - \mathbf { r } \right\| _ { 1 } + \mathcal { O } \left( \sum _ { k = 1 } ^ { L } \epsilon _ { k } ( \epsilon _ { k } - r _ { k } ) \right) .\tag{43}
$$

Consequently, in the weak-noise regime,

$$
\Gamma ( \mathbf { r } ) \simeq \exp { ( 2 \left\| \epsilon - \mathbf { r } \right\| _ { 1 } ) } .\tag{44}
$$

Proof of Lemma 3 Let

$$
\delta _ { k } : = \epsilon _ { k } - r _ { k }\tag{S60}
$$

denote the amount of noise removed at location $k .$ For a Pauli operator $Q ,$ let $a _ { k } ( Q )$ be the probability that an error drawn from $\pi _ { k }$ anticommutes with $Q .$ The Pauli-transfer eigenvalues of the physical and residual channels $\operatorname { a r e } ,$ respectively,

$$
1 - 2 \epsilon _ { k } a _ { k } ( Q ) \qquad \mathrm { a n d } \qquad 1 - 2 r _ { k } a _ { k } ( Q ) .\tag{S61}
$$

Since $\epsilon _ { k } \ < \ 1 / 2$ , the physical channel is invertible. Hence, the correction map $\mathcal { M } _ { k , r _ { k } } ~ =$ $\mathcal { E } _ { k , r _ { k } } \circ \mathcal { E } _ { k } ^ { - 1 }$ has eigenvalues

$$
\frac { 1 - 2 r _ { k } a _ { k } ( Q ) } { 1 - 2 \epsilon _ { k } a _ { k } ( Q ) } = 1 + 2 \delta _ { k } a _ { k } ( Q ) + \mathcal { O } ( \epsilon _ { k } \delta _ { k } ) .\tag{S62}
$$

Transforming these eigenvalues back to the Pauli quasiprobability representation gives

$$
\eta _ { k } ( I ) = 1 + \delta _ { k } + \mathcal { O } ( \epsilon _ { k } \delta _ { k } ) ,\tag{S63}
$$

$$
\eta _ { k } ( P ) = - \delta _ { k } \pi _ { k } ( P ) + \mathcal { O } ( \epsilon _ { k } \delta _ { k } ) , \qquad P \neq I .\tag{S64}
$$

Using $\begin{array} { r } { \sum _ { P \neq I } \pi _ { k } ( P ) = 1 } \end{array}$ , the local normalization therefore satisfies

$$
\Gamma _ { k } ( r _ { k } ) : = \sum _ { P } | \eta _ { k } ( P ) | = 1 + 2 \delta _ { k } + \mathcal { O } ( \epsilon _ { k } \delta _ { k } ) .\tag{S65}
$$

The quasiprobability normalizations multiply across circuit locations, so

$$
\begin{array} { l } { { \displaystyle \log \Gamma ( { \bf r } ) = \sum _ { k = 1 } ^ { L } \log \Gamma _ { k } ( r _ { k } ) } \ ~ } \\ { { \displaystyle ~ = \sum _ { k = 1 } ^ { L } \log \big [ 1 + 2 \delta _ { k } + \mathcal { O } ( \epsilon _ { k } \delta _ { k } ) \big ] } \ ~ } \\ { { \displaystyle ~ = 2 \sum _ { k = 1 } ^ { L } \delta _ { k } + \mathcal { O } \left( \sum _ { k = 1 } ^ { L } \epsilon _ { k } \delta _ { k } \right) } , } \end{array}\tag{S66}
$$

where we used $\log ( 1 + x ) = x + \mathcal { O } ( x ^ { 2 } )$ and $\delta _ { k } ^ { 2 } \le \epsilon _ { k } \delta _ { k }$ . Finally,

$$
\sum _ { k = 1 } ^ { L } \delta _ { k } = \sum _ { k = 1 } ^ { L } ( \epsilon _ { k } - r _ { k } ) = \left\| \epsilon - \mathbf { r } \right\| _ { 1 } ,\tag{S67}
$$

which proves equation (43). Neglecting the accumulated higher-order term in the weak-noise regime and exponentiating yields equation (44). □

Lemma $^ 4$ (Capped allocation at a fixed budget). Fix $R \in [ 0 , \| \epsilon \| _ { 1 } ]$ . The unique solution $o f$

$$
\begin{array} { r l } { \underset { \mathbf { r } } { \mathrm { m i n } } } & { { } \left\| \mathbf { r } \right\| _ { 2 } } \\ { s u b j e c t ~ t o } & { { } \left\| \mathbf { r } \right\| _ { 1 } = R , \qquad 0 \leq r _ { k } \leq \epsilon _ { k } } \end{array}\tag{45}
$$

is

$$
\begin{array} { r } { r _ { k } ^ { \mathrm { c a p } } ( R ) = \operatorname* { m i n } \{ \epsilon _ { k } , \nu ( R ) \} , } \end{array}\tag{46}
$$

where $\nu ( R ) \geq 0$ is chosen such that

$$
\sum _ { k = 1 } ^ { L } \operatorname* { m i n } \{ \epsilon _ { k } , \nu ( R ) \} = R .\tag{47}
$$

Proof of Lemma 4 It is equivalent to minimize $\textstyle \sum _ { k } r _ { k } ^ { 2 }$ . Suppose that two residual rates satisfy $r _ { i } < r _ { j }$ and $r _ { i } < \epsilon _ { i }$ . Moving a sufficiently small amount $\delta > 0$ from $r _ { j }$ to $r _ { i }$ preserves the total residual noise and all feasibility constraints, while changing the objective by

$$
\begin{array} { r } { \left( r _ { i } + \delta \right) ^ { 2 } + \left( r _ { j } - \delta \right) ^ { 2 } - r _ { i } ^ { 2 } - r _ { j } ^ { 2 } } \\ { = 2 \delta ( r _ { i } - r _ { j } ) + 2 \delta ^ { 2 } < 0 . } \end{array}\tag{S68}
$$

Thus, all residual rates not limited by their physical upper bounds must be equal. Locations whose physical rates are below this common level remain unchanged. Hence,

$$
\begin{array} { r } { r _ { k } ^ { \mathrm { c a p } } ( R ) = \operatorname* { m i n } \{ \epsilon _ { k } , \nu ( R ) \} . } \end{array}\tag{S69}
$$

The budget constraint determines $\nu ( R )$ , and uniqueness follows from the strict convexity of $\textstyle \sum _ { k } r _ { k } ^ { 2 }$ □

## Supplementary Note 4. Proofs for budget-aware noise control

This note proves the sensitivity of the risk to parameter changes, the cumulative regret bound and its rate.

Lemma 5 (Risk sensitivity). Suppose each of the $d _ { \theta } = \dim ( \theta )$ parameters occurs once, through a Pauli rotation $e ^ { - i \theta _ { k } \hat { P _ { k } } / 2 }$ , with fxed, parameter-independent CPTP noise channels and fxed measurement. Then, uniformly over admissible residual profles,

$$
| R _ { \mathbf { r } } ( \theta ) - R _ { \mathbf { r } } ( \theta ^ { \prime } ) | \leq L _ { R } \| \theta - \theta ^ { \prime } \| _ { 2 } , \qquad L _ { R } : = 2 \sqrt { 2 d _ { \theta } } .\tag{50}
$$

Proof of Lemma 5 Fix a residual profile r and write $q { \boldsymbol { \theta } } : = { \boldsymbol { q } } _ { \boldsymbol { \theta } } ^ { \mathbf { r } }$ Since $R \mathbf { r } ( \theta ) = \| \mu _ { q _ { \theta } } - \mu _ { p } \| _ { \mathcal { H } } ^ { 2 }$ its partial derivatives are

$$
\partial _ { k } R _ { \mathbf { r } } ( \theta ) = 2 \langle \partial _ { k } \mu _ { q _ { \theta } } , \mu _ { q _ { \theta } } - \mu _ { p } \rangle _ { \mathcal { H } } .\tag{S70}
$$

The parameter $\theta _ { k }$ enters only through the rotation $e ^ { - i \theta _ { k } P _ { k } / 2 }$ with $P _ { k } ^ { 2 } = I$ , and every other operation in the circuit, including the noise channels and the measurement, is fixed and linear. For any state $\rho$ entering the rotation,

$$
e ^ { - i \theta _ { k } P _ { k } / 2 } \rho e ^ { i \theta _ { k } P _ { k } / 2 } = \frac { \rho + P _ { k } \rho P _ { k } } { 2 } + \cos \theta _ { k } \frac { \rho - P _ { k } \rho P _ { k } } { 2 } - \frac { i \sin \theta _ { k } } { 2 } [ P _ { k } , \rho ] ,\tag{S71}
$$

so each probability $q _ { \theta } ( x )$ is a trigonometric polynomial of degree one in $\theta _ { k }$ . The parametershift rule therefore holds exactly, and by linearity of the mean embedding,

$$
\partial _ { k } \mu _ { q _ { \theta } } = \frac { 1 } { 2 } \Big ( \mu _ { q _ { \theta + \frac { \pi } { 2 } e _ { k } } } - \mu _ { q _ { \theta - \frac { \pi } { 2 } e _ { k } } } \Big ) ,\tag{S72}
$$

where $e _ { k }$ is the kth unit vector. Both shifted circuits produce valid probability distributions, and $\| \mu _ { q } \| _ { \mathcal { H } } ^ { 2 } = \mathbb { E } _ { y , y ^ { \prime } \sim q } [ \kappa ( y , y ^ { \prime } ) ] \le 1$ for every distribution q, so $\| \partial _ { k } \mu _ { q _ { \theta } } \| _ { \mathcal { H } } \leq 1$ . Moreover, $\| \mu _ { q _ { \theta } } - \mu _ { p } \| _ { \mathcal { H } } ^ { 2 } = \mathrm { M M D } _ { \kappa } ^ { 2 } ( q _ { \theta } , p ) \leq 2$ because $\kappa \in [ 0 , 1 ]$ and $\kappa ( x , x ) = 1$ . The Cauchy-Schwarz inequality in equation (S70) then gives $\lvert \partial _ { k } R _ { \bf r } ( \theta ) \rvert \le 2 \sqrt { 2 }$ for every $k ,$ hence $\Vert \nabla R _ { \mathbf { r } } ( \theta ) \Vert _ { 2 } \ \leq$ $2 \sqrt { 2 d _ { \theta } }$ . Since $R _ { \mathbf { r } }$ is smooth in $\theta ,$ the mean value theorem along the segment from $\theta ^ { \prime } \ \mathrm { t o } \ \theta$ yields equation (50). None of these bounds depends on $\mathbf { r } ,$ so the constant is uniform over residual profiles. □

Theorem 3 (Cumulative selection regret). Under Assumptions 2 and ${ \mathcal { B } } ,$ let $T \geq 2$ and $1 \leq K \leq T$ . Algorithm 2 satisfies, with probability at least $1 - \beta$

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } \leq ( 1 5 + 6 z ) \overline { { \Gamma } } ^ { 2 } T \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } + 2 a _ { v } ( \beta ) \left( \delta _ { \gamma } \sum _ { s \in \cal Z } w _ { s } D _ { s } + 2 T \varepsilon _ { \mathrm { t o l } } \right) } \\ & { \quad \quad \quad + 2 ( K - 1 ) \left[ L _ { R } h ( 1 + \log T ) + \frac { { \lambda \overline { { \Gamma } } ^ { 2 } } } { { \cal N } _ { \mathrm { f u t u r e , 0 } } } \right] . } \end{array}\tag{34}
$$

Proof of Theorem 3 At each comparison step $s \in \mathcal { Z } ,$ let $j _ { s } ^ { \star } \in \arg \operatorname* { m i n } _ { j \in \mathcal { I } } \Psi _ { s } ( j )$ . The selected profile $j _ { s }$ is retained for $w _ { s } = \mathrm { m i n } \{ K , T - s \}$ steps. Since comparisons start at $s \ = \ 0$ these blocks partition the training trajectory. Regrouping the terms in $\mathrm { R e g } _ { T }$ gives the exact decomposition

$$
\begin{array} { r l } { \mathrm { R e g } _ { T } = } & { \underbrace { \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } \left[ \Psi _ { s } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ^ { \star } ) \right] } _ { \mathrm { s e l e c t i o n ~ e r r o r ~ a t ~ c o m p a r i s o n ~ s t e p s } } } \\ & { + \underbrace { \displaystyle \sum _ { s \in \mathbb { Z } } \sum _ { t = s } ^ { s + { w _ { s } } - 1 } \left[ \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) + \Psi _ { s } ( j _ { s } ^ { \star } ) - \operatorname* { m i n } _ { j } \Psi _ { t } ( j ) \right] } _ { \mathrm { s c o r e ~ c h a n g e s ~ b e t w e e n ~ c o m p a r i s o n s } } . } \end{array}\tag{S73}
$$

To decompose the selection error, recall that $\widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j )$ is the validation loss computed from the fresh PER probes using equation (49). Let $\mathcal { F } _ { s }$ contain the target data and all information available before these probes, including $\theta _ { s }$ and the probe counts. Define the probe-averaged validation loss by

$$
\begin{array} { r } { \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j ) : = \mathbb { E } \left[ \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) \Big | \mathcal { F } _ { s } \right] . } \end{array}\tag{S74}
$$

This average removes the fresh probe randomness while retaining the same validation sample.

We first bound the selection error in the first sum of equation (S73). At comparison s, the controller's selection rule (31) gives

$$
\widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) + \lambda \frac { \Gamma _ { j _ { s } } ^ { 2 } - \Gamma _ { j _ { s } ^ { \star } } ^ { 2 } } { N _ { \mathrm { f u t u r e } , s } } \leq b _ { s j _ { s } ^ { \star } } - b _ { s j _ { s } } \leq \bar { b } _ { s } .\tag{S75}
$$

where $\bar { b } _ { s } : = \operatorname* { m a x } _ { j \in \mathcal { I } } b _ { s j }$ and the last step uses $b _ { s j } \geq 0$ . Adding and subtracting the measured and probe-averaged losses therefore separates the selection error into three contributions,

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in \mathcal { Z } } w _ { s } \left[ \Psi _ { s } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ^ { \star } ) \right] } \\ & { \quad \leq \displaystyle \sum _ { s \in \mathcal { Z } } w _ { s } \left[ R _ { { \bf r } ^ { ( j _ { s } ) } } ( \theta _ { s } ) - R _ { { \bf r } ^ { ( j _ { s } ^ { \star } ) } } \left( \theta _ { s } \right) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) + \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) \right] } \\ & { \quad \quad \quad + \displaystyle \sum _ { s \in \mathcal { Z } } w _ { s } \left[ \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) + \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) \right] } \\ & { \quad \quad \quad + \displaystyle \sum _ { s \in \mathcal { Z } } w _ { s } \left[ \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) + \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) \right] } \end{array}
$$

$$
\begin{array} { r l } { + } & { { } \underbrace { \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } \bar { b } _ { s } } _ { \mathrm { r a d i u s ~ c o n t r i b u t i o n } } . } \end{array}\tag{S76}
$$

The next three paragraphs bound these validation, probe and radius contributions. The final paragraph bounds the second sum in equation (S73), which accounts for score changes between comparisons.

## Validation uncertainty.

We first bound the validation-error contribution to RegT,

$$
\sum _ { s \in \overline { { \cal Z } } } w _ { s } \Big [ R _ { { \bf r } ^ { \left( j _ { s } \right) } } \left( \theta _ { s } \right) - R _ { { \bf r } ^ { \left( j _ { s } ^ { \star } \right) } } \left( \theta _ { s } \right) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) + \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) \Big ] .\tag{S77}
$$

The expression inside the brackets measures how much the validation sample distorts the comparison of population losses between two candidates. We first express this difference using the mean-embedding representation of MMD, then bound the resulting sampling error.

Using the kernel mean embeddings

$$
\mu _ { q } : = \mathbb { E } _ { X \sim q } [ \kappa ( X , \cdot ) ] , \qquad \widehat { \mu } _ { \mathrm { v a l } } : = \frac { 1 } { v } \sum _ { i = 1 } ^ { v } \kappa ( x _ { i } , \cdot ) ,\tag{S78}
$$

we express the loss difference directly as

$$
R _ { { \bf r } ^ { ( j ) } } ( \theta _ { s } ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j ) = \underbrace { \| \mu _ { p } \| _ { \mathcal { H } } ^ { 2 } - U _ { \mathrm { v a l } } } _ { \mathrm { i n d e p e n d e n t ~ o f ~ t h e ~ c a n d i d a t e } } + 2 \left. \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } , \mu _ { q _ { \theta _ { s } } ^ { \mathrm { r } ( j ) } } \right. _ { \mathcal { H } } ,\tag{S79}
$$

where $\begin{array} { r } { U _ { \mathrm { v a l } } = [ v ( v - 1 ) ] ^ { - 1 } \sum _ { i \neq i ^ { \prime } } \kappa ( x _ { i } , x _ { i ^ { \prime } } ) } \end{array}$ . This follows by expanding population MMD and averaging the validation U-statistic over the independent, unbiased PER probes.

The common term cancels when comparing any candidate $j _ { s }$ with $j _ { s } ^ { \star }$ , giving

$$
\begin{array} { r l } & { \left. { R } _ { { \bf { r } } ^ { \left( j _ { s } \right) } } \left( \theta _ { s } \right) - { R } _ { { \bf { r } } ^ { \left( j _ { s } ^ { \star } \right) } } \left( \theta _ { s } \right) - \overline { { L } } _ { \mathrm { { v a l } } , s } ^ { U } \left( j _ { s } \right) + \overline { { L } } _ { \mathrm { { v a l } } , s } ^ { U } \left( j _ { s } ^ { \star } \right) \right. } \\ & { \quad = 2 \left. \left. \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } , \mu _ { q _ { \theta _ { s } } ^ { \bf { r } \left( j _ { s } \right) } } - \mu _ { q _ { \theta _ { s } } ^ { \bf { r } \left( j _ { s } ^ { \star } \right) } } \right. _ { \mathcal { H } } \right. } \\ & { \quad \le 2 \| \widehat { \mu } _ { \mathrm { { v a l } } } - \mu _ { p } \| _ { \mathcal { H } } \mathrm { M M D } _ { \kappa } \left( q _ { \theta _ { s } } ^ { \bf { r } ^ { \left( j _ { s } \right) } } , q _ { \theta _ { s } } ^ { \bf { r } ^ { \left( j _ { s } ^ { \star } \right) } } \right) \qquad \mathrm { ( C a u c h y - S c h w a r z ) } . } \end{array}\tag{S80}
$$

(S81)

For the validation sampling error,

$$
\begin{array} { r l r l } & { \mathbb { E } \| \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } \| _ { \mathcal H } \leq \sqrt { \mathbb { E } \| \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } \| _ { \mathcal H } ^ { 2 } } } & & { ( \mathrm { J e n s e n } ) } \\ & { \qquad = \sqrt { \frac { \mathbb { E } _ { X \sim p } \kappa ( X , X ) - \| \mu _ { p } \| _ { \mathcal H } ^ { 2 } } { v } } \leq \frac { 1 } { \sqrt { v } } } & & { ( \mathrm { i . i . d . } , \kappa ( x , x ) \leq 1 ) . } \end{array}\tag{S82}
$$

Changing one observation changes the norm by at most $2 / v$ . Hence, with probability at least $1 - \beta / 2$

$$
\begin{array} { r l } { \displaystyle \| \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } \| _ { \mathcal H } \leq \mathbb E \| \widehat { \mu } _ { \mathrm { v a l } } - \mu _ { p } \| _ { \mathcal H } + \sqrt { \frac { 2 \log ( 2 / \beta ) } { v } } \qquad } & { \mathrm { ( M c D i a r m i d ) } } \\ { \displaystyle \leq \frac { 1 } { \sqrt { v } } + \sqrt { \frac { 2 \log ( 2 / \beta ) } { v } } = a _ { v } ( \beta ) . } \end{array}\tag{S83}
$$

For the distance between candidates, recall $D _ { s } = \mathrm { M M D } _ { \kappa } ( q _ { \theta _ { s } } ^ { 0 } , u )$ . The mismatch guarantee and effective depolarization yield

$$
\begin{array} { r l r l } { \mathrm { M M D } _ { \kappa } \left( q _ { \theta _ { s } } ^ { \mathrm { r } ^ { ( j _ { s } ) } } , q _ { \theta _ { s } } ^ { \mathrm { r } ^ { ( j _ { s } ^ { * } ) } } \right) \le \mathrm { M M D } _ { \kappa } \left( q _ { \theta _ { s } } ^ { \gamma _ { j _ { s } } } , q _ { \theta _ { s } } ^ { \gamma _ { j _ { s } ^ { * } } } \right) + 2 \varepsilon _ { \mathrm { t o l } } } & & { \mathrm { ( t r i a n g l e ~ i n e q u a l i t y ) } } \\ { = | \gamma _ { j _ { s } } - \gamma _ { j _ { s } ^ { * } } | D _ { s } + 2 \varepsilon _ { \mathrm { t o l } } } & & { \mathrm { ( e f f e c t i v e ~ d e p o l a r i z a t i o n ) } } \\ { \le \delta _ { \gamma } D _ { s } + 2 \varepsilon _ { \mathrm { t o l } } . } \end{array}\tag{S84}
$$

Combining these bounds in equation (S81) controls the comparison of every candidate $j _ { s }$ with $j _ { s } ^ { \star }$ on the same validation event, including at adaptively chosen parameters. Summing over the comparison blocks gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } \Big [ R _ { { \mathbf r } ^ { ( j _ { s } ) } } \big ( \theta _ { s } \big ) - R _ { { \mathbf r } ^ { ( j _ { s } ^ { \star } ) } } \big ( \theta _ { s } \big ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } \big ( j _ { s } \big ) + \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } \big ( j _ { s } ^ { \star } \big ) \Big ] } \\ & { \quad \le 2 a _ { v } ( \beta ) \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } \big ( \delta _ { \gamma } D _ { s } + 2 \varepsilon _ { \mathrm { t o l } } \big ) } \\ & { \quad = 2 a _ { v } ( \beta ) \left( \delta _ { \gamma } \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } D _ { s } + 2 T \varepsilon _ { \mathrm { t o l } } \right) , \qquad \displaystyle \sum _ { s \in \mathbb { Z } } w _ { s } = T . } \end{array}\tag{S85}
$$

This bounds the validation contribution (S77) to cumulative regret.

Probe uncertainty and its budget dependence.

We next bound the probe-error contribution in equation (S76). The triangle inequality gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in \mathcal { T } } w _ { s } \Big [ \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) + \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j _ { s } ^ { \star } ) \Big ] } \\ & { \quad \leq 2 \displaystyle \sum _ { s \in \mathcal { T } } w _ { s } \operatorname* { m a x } _ { j \in \mathcal { T } } \Big | \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j ) \Big | . } \end{array}\tag{S86}
$$

Because $j _ { s }$ is selected using these same probes, we control the errors simultaneously over all candidates before applying the bound to $j _ { s }$ and $j _ { s } ^ { \star }$

Conditional on $\mathcal { F } _ { s }$ , separate the measured loss into its generator-generator and targetgenerator terms,

$$
\begin{array} { r l r } {  { \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) = U _ { \mathrm { v a l } } + \widehat { A } _ { s j } - 2 \widehat { C } _ { s j } , } } \\ & { } & { \widehat { A } _ { s j } : = \frac { 1 } { M _ { s } ( M _ { s } - 1 ) } \sum _ { a \neq b } W _ { s j , a } W _ { s j , b } \kappa ( Y _ { s j , a } , Y _ { s j , b } ) , } \\ & { } & { \widehat { C } _ { s j } : = \displaystyle \frac { 1 } { M _ { s } } \sum _ { a = 1 } ^ { M _ { s } } W _ { s j , a } ( \frac { 1 } { v } \sum _ { i = 1 } ^ { v } \kappa ( x _ { i } , Y _ { s j , a } ) ) . } \end{array}\tag{S87}
$$

The kernel in $\widehat { A } _ { s j }$ lies in $[ - \Gamma _ { j } ^ { 2 } , \Gamma _ { j } ^ { 2 } ]$ , and each summand averaged in $\widehat { C } _ { s j }$ lies in $[ - \Gamma _ { j } , \Gamma _ { j } ]$ . Thus,

$$
\begin{array} { r l r l } & { \left| \widehat { A } _ { s j } - \mathbb { E } [ \widehat { A } _ { s j } \mid \mathcal { F } _ { s } ] \right| \leq \Gamma _ { j } ^ { 2 } \sqrt { \frac { 2 \log ( 8 J H / \beta ) } { \lfloor M _ { s } / 2 \rfloor } } } & & { \qquad ( \mathrm { U \mathrm { - s t a t i s t i c ~ H o e f f d i n g } } ) , } \\ & { \left| \widehat { C } _ { s j } - \mathbb { E } [ \widehat { C } _ { s j } \mid \mathcal { F } _ { s } ] \right| \leq \Gamma _ { j } \sqrt { \frac { 2 \log ( 8 J H / \beta ) } { M _ { s } } } } & & { \qquad ( \mathrm { H o e f f d i n g } ) , } \end{array}\tag{S88}
$$

each with conditional failure probability at most $\beta / ( 4 J H )$ . Averaging over the history and taking a union bound over the 2JH errors gives simultaneous coverage with probability at least $1 - \beta / 2$

On this event, since $U _ { \mathrm { v a l } }$ is fixed under the conditioning,

$$
\begin{array} { r l } & { \left| \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j ) \right| } \\ & { \quad \le \left| \widehat { A } _ { s j } - \mathbb { E } [ \widehat { A } _ { s j } \mid { \mathcal F } _ { s } ] \right| + 2 \left| \widehat { C } _ { s j } - \mathbb { E } [ \widehat { C } _ { s j } \mid { \mathcal F } _ { s } ] \right| } \\ & { \quad \le \Gamma _ { j } ^ { 2 } \sqrt { \displaystyle \frac { 2 \log ( 8 J H / \beta ) } { \lfloor M _ { s } / 2 \rfloor } } + 2 \Gamma _ { j } \sqrt { \displaystyle \frac { 2 \log ( 8 J H / \beta ) } { M _ { s } } } } \\ & { \quad \le ( \sqrt { 6 } + 2 \sqrt { 2 } ) \overline { { \Gamma } } ^ { 2 } \sqrt { \displaystyle \frac { \log ( 8 J H / \beta ) } { M _ { s } } } , } \end{array}\tag{S89}
$$

where the last inequality uses $\lfloor M _ { s } / 2 \rfloor \ge M _ { s } / 3$ and $1 \leq \Gamma _ { j } \leq \overline { { \Gamma } }$

The allocation rule makes the remaining average probe allowance nondecreasing. Feasibility therefore gives

$$
M _ { s } \ge M _ { 0 } = \left\lfloor \frac { \alpha B } { J H } \right\rfloor \ge \frac { \alpha B } { 2 J H } , \qquad M _ { 0 } \ge 2 .\tag{S90}
$$

Substituting this into equation (S89) and summing over blocks yields

$$
\begin{array} { r l } {  { 2 \sum _ { s \in \mathbb { Z } } w _ { s } \operatorname* { m a x } _ { j \in \mathcal { T } }  \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) - \overline { { L } } _ { \mathrm { v a l } , s } ^ { U } ( j )  } } \\ & { \le ( 8 + 4 \sqrt { 3 } ) \overline { { \Gamma } } ^ { 2 } \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } \sum _ { s \in \mathcal { T } } w _ { s } } \\ & { \le 1 5 \overline { { \Gamma } } ^ { 2 } T \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } , \qquad \sum _ { s \in \mathcal { T } } w _ { s } = T . } \end{array}\tag{S91}
$$

This bounds the cumulative probe contribution (S86).

## Measurement radius.

The standard error in the radius is computed from the per-shot projection terms

$$
\psi _ { s j , a } : = 2 W _ { s j , a } \Big ( \frac { 1 } { M _ { s } - 1 } \sum _ { b \neq a } W _ { s j , b } \kappa ( Y _ { s j , a } , Y _ { s j , b } ) - \frac { 1 } { v } \sum _ { i = 1 } ^ { v } \kappa ( x _ { i } , Y _ { s j , a } ) \Big )\tag{S92}
$$

as

$$
\widehat { \mathrm { S E } } \big ( \widetilde { L } _ { \mathrm { v a l } , s } ^ { U } ( j ) \big ) : = \Big [ \frac { 1 } { M _ { s } ( M _ { s } - 1 ) } \sum _ { a = 1 } ^ { M _ { s } } ( \psi _ { s j , a } - \bar { \psi } _ { s j } ) ^ { 2 } \Big ] ^ { 1 / 2 } ,\tag{S93}
$$

where $\bar { \psi } _ { s j }$ is the mean of the $\psi _ { s j , a } .$ Since $| W _ { s j , a } | \le \Gamma _ { j }$ and $\kappa \in [ 0 , 1 ]$ , every term satisfies $| \psi _ { s j , a } | \leq \mathrm { 2 r } _ { j } ( \Gamma _ { j } + 1 ) \leq 4 \Gamma _ { j } ^ { 2 }$ . The sample variance of such terms is at most $1 6 \Gamma _ { j } ^ { 4 } M _ { s } / ( M _ { s } - 1 )$ 2 SO $\widehat { \mathrm { S E } } \le 4 \Gamma _ { i } ^ { 2 } / \sqrt { M _ { s } - 1 } \le 6 \Gamma _ { i } ^ { 2 } / \sqrt { M _ { s } }$ for $M _ { s } \ge 2$ , and thus ${ \bar { b } _ { s } } \le 6 z { \overline { { \Gamma } } ^ { 2 } } / { \sqrt { M _ { s } } }$ .Equation (S90) gives $M _ { s } \ \overset { ^ { \vee } } { \geq } \ \alpha B / ( 2 J H )$ , and log $\because ( 8 J H / \beta ) \ge$ log $8 > 2$ because $J , H \ge 1$ and $\beta < 1$ . As a result, we have:

$$
\sum _ { s \in \mathbb { Z } } w _ { s } \bar { b } _ { s } \leq 6 z \bar { \Gamma } ^ { 2 } T \sqrt { \frac { 2 J H } { \alpha B } } \leq 6 z \bar { \Gamma } ^ { 2 } T \sqrt { \frac { J H \log \left( 8 J H / \beta \right) } { \alpha B } } , \qquad \sum _ { s \in \mathbb { Z } } w _ { s } = T .\tag{S94}
$$

Together with equation (S91), the probe and radius contributions are at most (15 + $6 z ) \overline { { \Gamma } } ^ { 2 } T \sqrt { J H \log ( 8 J H / \beta ) / ( \alpha B ) }$

Changes between comparisons.

It remains to bound the score-change contribution in equation (S73),

$$
\sum _ { s \in \mathbb { Z } } \sum _ { t = s } ^ { s + w _ { s } - 1 } \left[ \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) + \Psi _ { s } ( j _ { s } ^ { \star } ) - \operatorname* { m i n } _ { j } \Psi _ { t } ( j ) \right] .\tag{S95}
$$

This term tracks changes in both the retained candidate's score and the best available score. The minimum score changes by at most the largest change among individual candidate scores. Hence, within a block,

$$
\begin{array} { r l } & { \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) + \Psi _ { s } ( j _ { s } ^ { \star } ) - \underset { j } { \operatorname* { m i n } } \Psi _ { t } ( j ) } \\ & { \quad = \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) + \underset { j } { \operatorname* { m i n } } \Psi _ { s } ( j ) - \underset { j } { \operatorname* { m i n } } \Psi _ { t } ( j ) } \\ & { \quad \le | \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) | + \bigg | \underset { j } { \operatorname* { m i n } } \Psi _ { s } ( j ) - \underset { j } { \operatorname* { m i n } } \Psi _ { t } ( j ) \bigg | } \\ & { \quad \le 2 \underset { \tau = s } { \overset { t - 1 } { \sum } } \underset { j } { \operatorname* { m a x } } | \Psi _ { \tau + 1 } ( j ) - \Psi _ { \tau } ( j ) | \quad \quad \mathrm { ( t e l e s c o p i n g ) } . } \end{array}\tag{S96}
$$

Summing over blocks gives

$$
\begin{array} { r l } {  { \sum _ { s \in \mathcal { T } } \sum _ { t = s } ^ { s + w _ { s } - 1 } \bigg [ \Psi _ { t } ( j _ { s } ) - \Psi _ { s } ( j _ { s } ) + \Psi _ { s } ( j _ { s } ^ { \star } ) - \operatorname* { m i n } _ { j } \Psi _ { t } ( j ) \bigg ] } } \\ & { \leq 2 \sum _ { s \in \mathbb { Z } } \sum _ { \tau = s } ^ { s + w _ { s } - 2 } ( s + w _ { s } - 1 - \tau ) \operatorname* { m a x } _ { j } \vert \Psi _ { \tau + 1 } ( j ) - \Psi _ { \tau } ( j ) \vert } \\ & { \leq 2 ( K - 1 ) \sum _ { t = 0 } ^ { T - 2 } \operatorname* { m a x } _ { j } \vert \Psi _ { t + 1 } ( j ) - \Psi _ { t } ( j ) \vert . } \end{array}\tag{S97}
$$

The first inequality interchanges the finite sums. The second uses $w _ { s } \leq K$ and the disjoint block interiors, and empty sums are zero.

We bound the remaining score variation through parameter movement and changes in the affordable shot allowance. By Lemma 5 and Assumption 3,

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 2 } \operatorname* { m a x } _ { j } \vert \Psi _ { t + 1 } ( j ) - \Psi _ { t } ( j ) \vert } } \\ & { } & { \leq L _ { R } \sum _ { t = 0 } ^ { T - 2 } \vert \vert \theta _ { t + 1 } - \theta _ { t } \vert \vert _ { 2 } + \lambda \bar { \Gamma } ^ { 2 } \sum _ { t = 0 } ^ { T - 2 }  N _ { \mathrm { f u t u r e } , t + 1 } ^ { - 1 } - N _ { \mathrm { f u t u r e } , t } ^ { - 1 }  } \\ & { } & { \leq L _ { R } h \sum _ { t = 0 } ^ { T - 2 } \frac { 1 } { t + 1 } + \lambda \bar { \Gamma } ^ { 2 } \sum _ { t = 0 } ^ { T - 2 }  N _ { \mathrm { f u t u r e } , t + 1 } ^ { - 1 } - N _ { \mathrm { f u t u r e } , t } ^ { - 1 }  . } \end{array}\tag{S98}
$$

The training allocation rule further implies

$$
\begin{array} { r l } & { N _ { \mathrm { f u t u r e } , t + 1 } = \frac { ( T - t ) N _ { \mathrm { f u t u r e } , t } - \lfloor N _ { \mathrm { f u t u r e } , t } \rfloor } { T - t - 1 } } \\ & { \qquad = N _ { \mathrm { f u t u r e } , t } + \frac { N _ { \mathrm { f u t u r e } , t } - \lfloor N _ { \mathrm { f u t u r e } , t } \rfloor } { T - t - 1 } \ge N _ { \mathrm { f u t u r e } , t } . } \end{array}\tag{S99}
$$

Thus,

$$
\sum _ { t = 0 } ^ { T - 2 } \left| N _ { \mathrm { f u t u r e } , t + 1 } ^ { - 1 } - N _ { \mathrm { f u t u r e } , t } ^ { - 1 } \right| = { \frac { 1 } { N _ { \mathrm { f u t u r e } , 0 } } } - { \frac { 1 } { N _ { \mathrm { f u t u r e } , T - 1 } } } \quad ( \mathrm { m o n o t o n i c i t y ~ a n d ~ t e l e s c o p i n g } )
$$

$$
\leq { \frac { 1 } { N _ { \mathrm { f u t u r e , 0 } } } } .\tag{S100}
$$

Together with $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 2 } ( t + 1 ) ^ { - 1 } \leq 1 + \log T , } \end{array}$ this gives

$$
\sum _ { t = 0 } ^ { T - 2 } \operatorname* { m a x } _ { j } | \Psi _ { t + 1 } ( j ) - \Psi _ { t } ( j ) | \leq L _ { R } h ( 1 + \log T ) + \frac { \lambda { \overline { { \Gamma } } } ^ { 2 } } { N _ { \mathrm { f u t u r e } , 0 } } .\tag{S101}
$$

Substitution into equation (S97) therefore bounds the cumulative score-change contribution by

$$
2 ( K - 1 ) \left[ L _ { R } h ( 1 + \log T ) + \frac { \lambda \overline { { \Gamma } } ^ { 2 } } { N _ { \mathrm { f u t u r e , 0 } } } \right] .\tag{S102}
$$

## Combining the bounds.

The validation and probe events hold jointly with probability at least $1 - \beta$ by a union bound. On this event, substituting the bounds on the validation, probe, radius and score-change into equation (S73) yields

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } \leq ( 1 5 + 6 z ) \overline { { \Gamma } } ^ { 2 } T \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } } \\ & { \qquad + 2 a _ { v } ( \beta ) \left( \delta _ { \gamma } \displaystyle \sum _ { s \in \mathcal { T } } w _ { s } D _ { s } + 2 T \varepsilon _ { \mathrm { t o l } } \right) } \\ & { \qquad + 2 ( K - 1 ) \left[ L _ { R } h ( 1 + \log T ) + \displaystyle \frac { \lambda \bar { \Gamma } ^ { 2 } } { N _ { \mathrm { f u t u r e , 0 } } } \right] , } \end{array}\tag{S103}
$$

which proves the theorem.

Corollary 2 (Rate under a linear execution budget). Suppose $B = \Theta ( T )$ and choose $K = \lceil T ^ { 2 / 3 } \rceil$ . For fxed circuit size, candidate grid, $h , \alpha , \lambda , z$ , and $\varepsilon _ { \mathrm { t o l } }$ , with probability at least $1 - \beta$

$$
\mathrm { R e g } _ { T } = \widetilde { \mathcal { O } } \left( T ^ { 2 / 3 } + \frac { T } { \sqrt { v } } \right) ,\tag{35}
$$

where $\widetilde { \mathcal { O } }$ suppresses logarithmic factors in T and $1 / \beta$

$P r o o f$ of Corollary 2 Use the event of probability at least $1 - \beta$ in Theorem 3. With $K =$ $\lceil T ^ { 2 / 3 } \rceil$ and $T \geq 2$

$$
H = \left\lceil { \frac { T } { K } } \right\rceil \leq { \frac { T } { K } } + 1 \leq { \frac { 2 T } { K } } \leq 2 T ^ { 1 / 3 } .\tag{S104}
$$

Since $B = \Theta ( T )$ , we write $B = \varrho T$ for a constant $\varrho > 0$ , and the probe term satisfies

$$
( 1 5 + 6 z ) \overline { { \Gamma } } ^ { 2 } T \sqrt { \frac { J H \log ( 8 J H / \beta ) } { \alpha B } } = \mathcal { O } \Big ( T ^ { 2 / 3 } \sqrt { \log ( T / \beta ) } \Big ) ,\tag{S105}
$$

with the remaining parameters fixed.

For the validation term, $\kappa \in [ 0 , 1 ]$ implies

$$
D _ { s } ^ { 2 } = \mathbb { E } _ { X , X ^ { \prime } \sim q _ { \theta _ { * } } ^ { 0 } } \kappa ( X , X ^ { \prime } ) + \mathbb { E } _ { Y , Y ^ { \prime } \sim u } \kappa ( Y , Y ^ { \prime } )
$$

$$
- 2 \mathbb { E } _ { X \sim q _ { \theta _ { s } } ^ { 0 } , Y \sim u ^ { K } } ( X , Y ) \le 2 .\tag{S106}
$$

Thus $\begin{array} { r } { \sum _ { s } w _ { s } D _ { s } \le \sqrt { 2 } T } \end{array}$ , and

$$
\begin{array} { l } { \displaystyle 2 a _ { v } ( \beta ) \left( \delta _ { \gamma } \sum _ { s } w _ { s } D _ { s } + 2 T \varepsilon _ { \mathrm { t o l } } \right) } \\ { \displaystyle \qquad \leq 2 T a _ { v } ( \beta ) \left( \sqrt { 2 } \delta _ { \gamma } + 2 \varepsilon _ { \mathrm { t o l } } \right) = \mathcal { O } \left( T \sqrt \frac { \log \left( 2 / \beta \right) } { v } \right) . } \end{array}\tag{S107}
$$

Finally,

$$
N _ { \mathrm { f u t u r e , 0 } } = \frac { ( 1 - \alpha ) B } { P _ { \mathrm { c i r c } } T } = \frac { ( 1 - \alpha ) \varrho } { P _ { \mathrm { c i r c } } } ,\tag{S108}
$$

which is independent of T. Since $K - 1 < T ^ { 2 / 3 }$ , the last term satisfies

$$
\begin{array} { r l } & { 2 ( K - 1 ) \left[ L _ { R } h ( 1 + \log T ) + \frac { \lambda \overline { { \Gamma } } ^ { 2 } } { N _ { \mathrm { f u t u r e , 0 } } } \right] } \\ & { ~ = \mathcal { O } \Big ( T ^ { 2 / 3 } ( 1 + \log T ) \Big ) . } \end{array}\tag{S109}
$$

Adding these three estimates yields

$$
\mathrm { R e g } _ { T } = \mathcal { O } \left( T ^ { 2 / 3 } \big [ 1 + \log T + \sqrt { \log ( T / \beta ) } \big ] + T \sqrt { \frac { \log ( 2 / \beta ) } { v } } \right) ,\tag{S110}
$$

which is $\widetilde { \mathcal { O } } ( T ^ { 2 / 3 } + T / \sqrt { v } )$