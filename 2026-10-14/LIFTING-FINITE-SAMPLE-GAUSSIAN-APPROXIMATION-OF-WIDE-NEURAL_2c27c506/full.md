# LIFTING FINITE-SAMPLE GAUSSIAN APPROXIMATION OF WIDE NEURAL NETWORKS TO FUNCTIONAL CONVERGENCE

Luis Wiederhold<sup>∗</sup> School of Computation, Information and Technology Technical University of Munich wiederhold.luis@gmx.de

Nil Ayday<sup>∗</sup> School of Computation, Information and Technology Technical University of Munich nil.ayday@tum.de

Debarghya Ghoshdastidar School of Computation, Information and Technology Technical University of Munich

October 14, 2026

## ABSTRACT

Randomly initialised neural networks of width m converge to Gaussian processes as m → ∞, with two types of quantitative bounds characterising this convergence. Finite-sample bounds compare the network with its Gaussian limit at a fixed finite sample set, on an arbitrary domain, in the quadratic Wasserstein distance W , achieving a convergence rate of m<sup>−1/2</sup> with explicit constants. However, these constants grow with the number of samples, so the bound becomes uninformative as the sample set is refined. Functional bounds remove that dependence by comparing the two as random functions, at the price of spherical input, metrics weaker than W<sub>2</sub>, or slower rates. In this paper, we therefore lift a finite-sample bound to a functional one. The lifting refines the sample set until it is dense in the domain, so the bound must first be made independent of the number of samples. We achieve this by measuring the error per sample, which makes the constant uniform. For a polynomial activation and any compact input set, this gives a rate of $m ^ { - 1 / 2 }$ in the supremum norm, with an explicit constant.

Keywords neural network Gaussian process, functional convergence rates, Wasserstein distance

## 1 Introduction

The connection between wide neural networks and Gaussian processes (GPs) was first established in Neal [1996], which showed that a randomly initialised single-hidden-layer network converges in distribution to a GP as its width grows. The limiting neural network Gaussian Process (NNGP) is fully specified by its covariance kernel, which depends explicitly on the network’s activation function and initialisation. For many standard activations, this kernel can be computed in closed form [Williams, 1996]. This convergence reduces the network to its kernel, enabling exact Bayesian inference for infinite-width networks [Lee et al., 2018], to distill large datasets into a handful of synthetic examples by optimizing them directly through the kernel [Nguyen et al., 2021], and to rank neural architectures without training them [Chen et al., 2021]. Beyond single-layer neural networks, this equivalence extends to architectures such as deep fully connected networks [Lee et al., 2018, Matthews et al., 2018]. Convergence to GPs has also been established for wide feedforward and recurrent architectures [Yang, 2019], with analogous kernels derived for transformers [Hron et al., 2020], graph convolutional networks [Niu et al., 2023], and graph transformers [Ayday et al., 2026]. The same framework has since led to more refined and practical approximations, such as tensor programs and the maximal update parametrisation (µP) [Yang and Hu, 2021, Yang et al., 2022].

Although the qualitative picture is now well understood, the convergence rate to NNGP in terms of width m remains less developed. Quantitative convergence has been studied from two distinct perspectives: comparing the finite-dimensional distributions of the network at a fixed finite sample set, or treating the network and its limit as random functions to establish convergence on a function space. Table 1 collects the results of both perspective.

<table><tr><td>Result</td><td></td><td>Depth</td><td>Inputs</td><td>Metric</td><td>Activation</td><td>Rate</td><td>with n</td></tr><tr><td rowspan="6">nmpe apie</td><td>[Basteri and Trevisan, 2024, Thm. 3.1]</td><td>any</td><td>finite</td><td> $\mathcal { W } _ { 2 }$ </td><td> $\operatorname { L i p s c h i t z }$ </td><td> $m ^ { - 1 / 2 }$ </td><td> $\sqrt { n } ~ \ S 2 . 3$ </td></tr><tr><td>[Bordino et al., 2024, Thm. 6]</td><td>1</td><td>finite</td><td> $\mathcal { W } _ { 1 }$ </td><td> ${ \mathcal { C } } ^ { 2 } , { \mathrm { p o l y . g r . } }$ </td><td> $m ^ { - 1 / 2 }$ </td><td> $\sqrt { n }$ </td></tr><tr><td>[Balasubramanian and Ross, 2025, Thm.1.1]</td><td>any</td><td>finite</td><td> $\mathcal { W } _ { 1 }$ </td><td>Lipschitz</td><td> $m ^ { - 1 / 6 }$ </td><td></td></tr><tr><td>[Favaro et al., 2025, Thm. 3.6(1)]</td><td>any</td><td> $\mathrm { f i n i t e } ^ { * }$ </td><td> $d _ { \mathrm { c v x } }$ </td><td> $\mathsf { p o l y . g r . }$ </td><td> $m ^ { - 1 / 2 }$ </td><td>ippiciıt</td></tr><tr><td>[Trevisan, 2023, Thm. 1.1(a)]</td><td>any</td><td> $\mathrm { f i n i t e } ^ { * }$ </td><td> $\mathcal { W } _ { p }$ </td><td>Lipschitz</td><td> $m ^ { - 1 }$ </td><td></td></tr><tr><td>[Celli and Peccati, 2025, Thm. 5]</td><td>any</td><td> $\mathrm { f i n i t e } ^ { * }$ </td><td> $\mathcal { W } _ { 2 }$ </td><td> ${ \mathcal { C } } ^ { 1 } \mathrm { o r } \mathrm { P L }$ </td><td> $m ^ { - 1 }$ </td><td></td></tr><tr><td rowspan="6">Functonal</td><td>[Eldan et al., 2021, Thm. 2]</td><td>1</td><td>sphere</td><td> $\mathcal { W } _ { 1 } ( \mathcal { C } ^ { 0 } )$ </td><td>polynomial</td><td> $m ^ { - 1 / 6 }$ </td><td></td></tr><tr><td>[Eldan et al., 2021, Thm. 5]</td><td>1</td><td>sphere</td><td> $\mathcal { W } _ { 1 } ( \mathcal { C } ^ { 0 } )$ </td><td>monomial</td><td> $m ^ { - 1 / 2 }$ </td><td></td></tr><tr><td>[Klukowski, 2022, Thm. 1]</td><td>1</td><td>sphere</td><td> $\mathcal { W } _ { 2 } ( L ^ { 2 } )$ </td><td>polynomial</td><td> $m ^ { - 1 / 2 }$ </td><td>none</td></tr><tr><td>[Balasubramanian et al., 2023, Thm. 1.2]</td><td>any</td><td>sphere</td><td> $\mathcal { W } _ { 1 } ( \mathcal { C } ^ { 0 } )$ </td><td>Lipschitz</td><td> ${ m ^ { - q _ { d } } }$ </td><td></td></tr><tr><td>[Favaro et al., 2025, Thm. 3.16]</td><td>any</td><td>ball</td><td> $\mathcal { W } _ { 2 } ( \mathcal { C } ^ { k } )$ </td><td>smooth</td><td> $m ^ { - 1 / 8 }$ </td><td></td></tr><tr><td>This paper (Thm. 2)</td><td>1</td><td>compact</td><td> $\mathcal { W } _ { 2 } ( \mathcal { C } ^ { 0 } )$ </td><td>polynomial</td><td> $m ^ { - 1 / 2 }$ </td><td></td></tr></table>

Table 1: Quantitative Gaussian approximation for wide networks; rates are in the width $m ,$ and n is the number of inputs. Finite-sample bounds hold at a fixed sample set with a constant growing in $n ;$ “implicit” means the bound depends on the input set, but this dependence is not quantified. Functional bounds avoid this at the cost of the input set, metric or rate. finite<sup>∗</sup>: finite input, non-degenerate kernel; $d _ { \mathrm { c v x } } \colon$ convex distance; poly. gr.: polynomial growth; PL: piecewise linear. Rates for Balasubramanian and Ross [2025] and Balasubramanian et al. [2023] are one-hidden-layer specialisations. For the latter, inputs lie on the unit sphere of $\mathbb { R } ^ { d }$ and $\begin{array} { r } { q _ { d } = \frac { 1 } { 8 d - 2 } - \varepsilon } \end{array}$ for arbitrary $\varepsilon > 0 .$ By Corollary 1 the last row holds on every compact subset of $\mathbb { R } ^ { d }$ , with the constant of an enclosing hyperrectangle.

Finite-Sample Convergence. The first approach fixes a finite sample set $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { n } \}$ and compares the random vector $f [ \mathcal { X } ]$ of n network evaluations with the corresponding Gaussian vector $G [ \mathcal { X } ]$ . The strength of the finite-input route is its broad applicability. Once the comparison is reduced to a pair of random vectors in $\mathbb { R } ^ { n }$ , the tools of finite-dimensional Gaussian approximation apply: the distance between two centred Gaussian laws is controlled by their covariances, and couplings can be constructed explicitly [Basteri and Trevisan, 2024]. This yields bounds in the quadratic Wasserstein distance $\mathcal { W } _ { 2 }$ , a stronger metric than the first-order $\mathcal { W } _ { 1 }$ , with explicit constants, and it asks nothing of the input points themselves: they may form an arbitrary finite collection in any dimension, rather than requiring restrictive structural assumptions such as lying on a unit sphere. It also covers the standard Gaussian initialisation with bias terms and arbitrary depth [Basteri and Trevisan, 2024]. Working with a finite sample set, however, introduces a dependence on its size.

A bound at X is a statement about a random vector in $\mathbb { R } ^ { n }$ , and with n fixed, quantities that vary with n are absorbed into the constants. For example in Basteri and Trevisan [2024] these include the Frobenius norm of a $n \times$ n covariance fluctuation and the smallest strictly positive eigenvalue of the limiting covariance matrix $K [ \mathcal { X } ]$ . Whether the bound survives refinement of the sample set depends on how these scale. In a learning-theoretic application the sample set is the training sample, so n needs to be large. Arora et al. [2019] bound the population error of kernel regression with the limiting kernel by a term of order $n ^ { - 1 / \bar { 2 } }$ , and carrying that guarantee over to a network of width m means comparing the two on that same sample. An n-dependent constant that grows with the sample size can offset the statistical decay. Moreover, a finite-input bound describes the network only at the chosen inputs. Convergence of the finite-dimensiona distributions is a statement about each fixed finite sample set, so it does not imply convergence of the laws on a function space. The second perspective treats the network directly as a random function.

Functional Convergence. Bracale et al. [2021] and Hanin [2021] establish functional convergence without rates. The lower half of Table 1 collects the quantative results. These avoid the dependence on a prescribed finite sample set, but each restricts the setting elsewhere, in the domain or in the metric. The inputs lie on a sphere in Eldan et al. [2021], Klukowski [2022], Balasubramanian et al. [2023], and Klukowski [2022] draws the weights from a sphere as well. A sphere has no interior, so such a bound says nothing about a ball, a hyperrectangle, or any set with positive volume. The metric is the weaker $\mathcal { W } _ { 1 }$ distance in Eldan et al. [2021], Balasubramanian et al. [2023]. Since $\bar { \mathcal { W } _ { 1 } } \leq \mathcal { W } _ { 2 }$ a bound in $\mathcal { W } _ { 2 }$ implies one in $\mathcal { W } _ { 1 }$ but not conversely. Closest to our setting is Favaro et al. [2025] with an open ball in any dimension, arbitrary depth, the same Gaussian initialisation with biases, and a $\mathcal { C } ^ { k }$ norm stronger than ours, since it controls derivatives as well. Unlike a sphere, their ball is full-dimensional but they pay for the larger domain and the stronger metric with the rate $m ^ { - 1 / 8 }$ , against the $m ^ { - 1 / 2 }$ available on the sphere. Apart from the last row, no functional row of Table 1 combines a general compact set of inputs with the rate $m ^ { - 1 / 2 }$

The two perspectives are therefore complementary: finite-input bounds are general in the domain, the initialisation and the metric but carry a constant that depends on the input set. In contrast, functional bounds remove that dependence at the price of restricted domains and metrics. To our knowledge, no functional bound in the quadratic Wasserstein distance with respect to the supremum norm attains the rate $m ^ { - 1 / 2 }$ on a general compact domain under the standard Gaussian initialisation. A functional bound with the generality of the finite-input analysis would give guarantees that do not weaken as the sample grows.

Contribution. Our result combines the two perspectives. We start from the finite-sample bound of Basteri and Trevisan [2024] and prove that under an effective-rank condition, which we verify for polynomial activations, the constant of Basteri and Trevisan [2024] is at most of order ${ \sqrt { n } } .$ . The bound therefore reads $\sqrt { n / m }$ , so the width must grow at least as fast as the number of inputs for it to be informative. Measuring the error per input removes this growth entirely. Then we provide a chain of results that converts this finite-sample bound into a functional bound in the supremum norm. For one hidden layer, any polynomial activation, and inputs in a hyperrectangle, we obtain $\mathcal { W } _ { 2 } ^ { C ^ { 0 } ( I ) } ( f , G ) = O ( m ^ { - 1 / 2 } )$ with explicit constants, and by restriction (Corollary 1) the bound holds on every compact subset of $\mathbb { R } ^ { d }$ . Among the functional bounds of Table 1, the rate $m ^ { - 1 / 2 }$ appears only on the sphere. Theorem 2 reaches it on an arbitrary compact set, so the spherical setting of Eldan et al. [2021], Klukowski [2022], Balasubramanian et al. [2023] is a special case. This is a first step towards merging the two formalisations of quantitative convergence, carrying the generality of the finite-sample bounds into a statement about the random function.

Section 2 fixes notation, introduces the network and its Gaussian process limit, and states the finite-sample bound we start from, whose constant grows at most like $\sqrt { n }$ and becomes independent of the input set once the error is measured per input. Section 3 turns that uniformity into a functional bound by refining the input set until it is dense in the domain, a limit that is available only because the constant no longer grows with the number of inputs. Section 4 discusses the open questions.

## 2 Preliminaries and notes on finite-sample bound

We fix here the objects and notation used throughout: the network and its Gaussian process limit (Section 2.1), the distance in which the two are compared (Section 2.3), and the finite-sample bound we start from, whose constant we track as the sample set grows (Section 2.3).

Notation. For $v \in \mathbb { R } ^ { n }$ we write $\textstyle \| v \| = \bigl ( \sum _ { i = 1 } ^ { n } v _ { i } ^ { 2 } \bigr ) ^ { 1 / 2 }$ for the Euclidean norm, and for a matrix $A = \left( A _ { i j } \right)$ we write $\begin{array} { r } { \| A \| = \big ( \sum _ { i , j } A _ { i j } ^ { 2 } \big ) ^ { 1 / 2 } } \end{array}$ for the Frobenius norm, which coincides with the Euclidean norm when A is a vector. For a symmetric positive semi-definite $A \in \mathbb { R } ^ { n \times n }$ with eigenvalues $\lambda _ { 1 } , . . . , \lambda _ { n } \geq 0$ we write $\lambda _ { \operatorname* { m i n } } ( A ) = \operatorname* { m i n } _ { i } \lambda _ { i }$ $\lambda ^ { + } ( A ) = \operatorname* { m i n } \{ \lambda _ { i } : \lambda _ { i } > 0 \}$ , and $\sqrt { A }$ for the symmetric positive semi-definite square root, obtained by replacing each $\lambda _ { i }$ by $\sqrt { \lambda _ { i } }$ in the spectral decomposition. $\lambda _ { \mathrm { m i n } }$ and $\lambda ^ { + }$ differ exactly when $A$ is singular, which is the regime of Section $2 . 3$ . For a function h and a finite set $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ in its domain, fixed together with an enumeration, we write $h [ \mathcal { T } ] = ( h ( t _ { 1 } ) , \ldots , h ( t _ { n } ) )$ for the vector of evaluations; if h is $\mathbb { R } ^ { m }$ -valued, $h [ \mathcal { T } ]$ is the m × n matrix whose i-th column is $h ( t _ { i } )$ . For a multi-index $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { d } ) \in  { \mathbb { N } } _ { 0 } ^ { d }$ we write $\begin{array} { r } { | \alpha | = \sum _ { j = 1 } ^ { d } \alpha _ { j } } \end{array}$ and $x ^ { \alpha } = x _ { 1 } ^ { \alpha _ { 1 } } \cdot \cdot \cdot x _ { d } ^ { \alpha _ { d } }$ for $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . The polynomials of total degree at most r in d variables form a vector space of dimension $\textstyle { \binom { r + d } { d } }$ , spanned by $\{ x ^ { \alpha } : | \alpha | \leq r \}$

Gaussian vectors. For a random vector $X \in \mathbb { R } ^ { n }$ with finite second moment, its covariance matrix is defined as $\Sigma ( X ) = \mathbb { E } \bigl [ ( X - \mathbb { E } [ X ] ) ( X - \mathbb { E } [ X ] ) ^ { \top } \bigr ]$ . If X is centred then $\Sigma ( X ) = \mathbb { E } [ X X ^ { \top } ]$ . We write $\mathcal { N } ( 0 , \Sigma )$ for the centred Gaussian law on $\mathbb { R } ^ { n }$ with covariance Σ.

## 2.1 Random neural networks and their Gaussian process (GP) limit

We now define the two objects the paper compares. The first is a network at finite width $m ;$ the second is the GP it approaches as the width grows $( m  \infty )$ , whose kernel is determined by the activation and the initialisation variances.

Definition 1 (randomly initialised shallow network). Let $\phi : \mathbb { R }  \mathbb { R }$ be an activation function. A shallow neural network of width m is a function $f : \mathbb { R } ^ { d } $ R oftheform

$$
g ( x ) = W x + b \in \mathbb { R } ^ { m } , \qquad f ( x ) = a ^ { \top } \phi { \big ( } g ( x ) { \big ) } + c \in \mathbb { R } ,
$$

with parameters $W \in \mathbb { R } ^ { m \times d } , b , a \in \mathbb { R } ^ { m } , c \in \mathbb { R }$ , where ϕ acts componentwise and g is the pre-activation. At random initialisation the parameters are independent, with $\begin{array} { r } { W _ { r j } \sim { \mathcal N } \Big ( 0 , \frac { \sigma _ { u } ^ { 2 } } { d } \Big ) , \quad b _ { r } \sim { \mathcal N } \big ( 0 , \sigma _ { b } ^ { 2 } \big ) , \quad a _ { r } \sim { \mathcal N } \Big ( 0 , \frac { \tau _ { a } ^ { 2 } } { m } \Big ) } \end{array}$ c ∼ $\mathcal { N } ( 0 , \tau _ { c } ^ { 2 } )$ , with $\sigma _ { w } ^ { 2 } , \tau _ { a } ^ { 2 } > 0$ and $\sigma _ { b } ^ { 2 } , \tau _ { c } ^ { 2 } \geq 0$

The network in Definition 1 maps d-dimensional inputs to scalar outputs. Multivariate outputs can be handled analogously and are omitted for notational simplicity. Throughout, $I \subset  { \mathbb { R } } ^ { d }$ denotes a fixed compact set, and input sets ${ \mathcal { X } } = { \bar { \{ } x _ { 1 } , \ . . . , x _ { n } \} }$ are finite subsets of I, with corresponding evaluations $f [ \mathcal { X } ] \in \mathbb { R } ^ { n }$ and $\boldsymbol { g } [ \mathcal { X } ] \in \mathbb { R } ^ { m \times n }$ . Since W and b are Gaussian, the pre-activation is a GP for every width. Applying ϕ destroys Gaussianity, but the output is a sum of m independent terms, so by the central limit theorem it is asymptotically Gaussian as $m  \infty$ . The covariance of that limit is computable, and the two kernels are the following.

Definition 2 (Gaussian process and NNGP kernels). A Gaussian process on I is afamily $( G ( x ) ) _ { x \in I }$ ofreal random variables on a common probability space, every finite subfamily of which is a Gaussian vector; it is centred if $\mathbb { E } [ G ( x ) ] = 0$ for every $x \in I .$ , and its covariance kernel is $( x , \dot { y } ) \stackrel { \cdot } { \mapsto } \mathbb { E } [ G ( x ) G ( y ) ]$ . A centred Gaussian process is determined in law by its covariance kernel, and every symmetric positive semi-definite function on $I \times I$ is the covariance kernel ofone [Kallenberg, 2002, Theorem 6.16]. For $x , y \in I$ define

$$
\begin{array} { r } { \kappa ( x , y ) = \frac { \sigma _ { w } ^ { 2 } } { d } \left. x , y \right. + \sigma _ { b } ^ { 2 } , } \end{array}\tag{1}
$$

$$
K ( x , y ) = \tau _ { a } ^ { 2 } \mathbb { E } \big [ \phi ( Z _ { x } ) \phi ( Z _ { y } ) \big ] , \qquad ( Z _ { x } , Z _ { y } ) \sim \mathcal { N } \bigg ( 0 , \binom { \kappa ( x , x ) \ \kappa ( x , y ) } { \kappa ( x , y ) \ \kappa ( y , y ) } \bigg ) ,\tag{2}
$$

$$
K _ { c } ( x , y ) = K ( x , y ) + \tau _ { c } ^ { 2 } .\tag{3}
$$

Here κ is the covariance of the pre-activation coordinates of Definition 1, the same for every coordinate, and for a finite ${ \mathcal { X } } = \{ x _ { 1 } , \ldots , x _ { n } \} \subseteq I$ we write $\kappa [ \mathcal { X } ] , K [ \mathcal { X } ] , K _ { c } [ \mathcal { X } ] \in \mathbb { R } ^ { n \times n }$ for the matrices with entries $\kappa ( x _ { i } , x _ { j } ) , K ( x _ { i } , x _ { j } )$ $K _ { c } ( x _ { i } , x _ { j } ) . \ \mathrm { { } } A l l$ three kernels are symmetric and positive semi-definite, so the neural network Gaussian process $( N N G P )$ associated with Definition 1 is well defined as the centred Gaussian process G on I with covariance kernel $K _ { c }$

The pre-activation $g [ \mathcal { X } ]$ consists of m independent rows with law $\mathcal { N } ( 0 , \kappa [ \mathcal { X } ] )$ and is exactly Gaussian at every width. The output $f [ \mathcal { X } ]$ is not. A finite-input Gaussian approximation bound such as Theorem 1 measures, at finite width, how far the law of $f [ \mathcal { X } ]$ is from $\mathcal { N } ( 0 , \dot { K } _ { c } [ \mathcal { X } ] )$ ). It is stated in the quadratic Wasserstein distance, defined next.

## 2.2 The quadratic Wasserstein distance

Definition 3 (quadratic Wasserstein distance). Let $( E , \| \cdot \| _ { E } )$ be a separable Banach space and let p, q be probability measures on E withfinite second moment. Then

$$
\begin{array} { r } { \mathcal { W } _ { 2 } ^ { E } ( p , q ) = \operatorname* { i n f } \Big \{ \sqrt { \mathbb { E } \big [ \| X - Y \| _ { E } ^ { 2 } \big ] } ~ : ~ X \sim p , ~ Y \sim q \Big \} , } \end{array}
$$

the infimum running over all couplings of p and $q ,$ i.e. all pairs (X, Y ) of E-valued random variables on a common probability space with the given marginals. We write $\mathcal { W } _ { 2 } ^ { E } ( X , Y ) f o r \ \bar { \mathcal { W } } _ { 2 } ^ { E } ( \mathbb { P } _ { X } , \mathbb { P } _ { Y } )$ . Four instances occur below, differing only in the state space E or its norm:

<table><tr><td> $E$ </td><td>||·|E</td><td>notation</td><td>compares</td><td>used in</td></tr><tr><td> $\mathbb { R } ^ { n }$ </td><td> $\begin{array} { r } { \| v \| = \left( \sum _ { i = 1 } ^ { n } v _ { i } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ </td><td> $\mathcal { W } _ { 2 } = \mathcal { W } _ { 2 } ^ { \mathbb { R } ^ { n } }$ </td><td>finite sample set, total error</td><td>Thm. 1</td></tr><tr><td> $\mathbb { R } ^ { n }$ </td><td> $\begin{array} { r } { \| v \| _ { n } = \big ( n ^ { - 1 } \sum _ { i = 1 } ^ { n } v _ { i } ^ { 2 } \big ) ^ { 1 / 2 } } \end{array}$ </td><td> $\mathcal { W } _ { 2 , n }$ </td><td>finite sample set, error per sample</td><td>§2.3</td></tr><tr><td> $L ^ { 2 } ( I )$ </td><td> $\textstyle \left( \int _ { I } h ( x ) ^ { 2 } d x \right) ^ { 1 / 2 }$ </td><td> $\mathcal { W } _ { 2 } ^ { L ^ { 2 } ( I ) }$ </td><td>random functions, mean-square error §3</td><td></td></tr><tr><td> ${ \mathcal { C } } ^ { 0 } ( I )$ </td><td> $\operatorname { s u p } _ { x \in I } \left| h ( x ) \right|$ </td><td> $\mathcal { W } _ { 2 } ^ { c ^ { 0 } ( I ) }$ </td><td>random functions, supremum error</td><td>§3</td></tr></table>

All four are separable Banach spaces. In $\mathbb { R } ^ { n }$ the norm $\| \cdot \| _ { n }$ is the root mean square of the coordinates, so $\mathcal { W } _ { 2 , n } =$ $n ^ { - 1 / 2 } { \mathcal { W } } _ { 2 }$ measures the transport cost per input, where $\ddot { \mathcal { W } } _ { 2 } = \mathcal { W } _ { 2 } ^ { \mathbb { R } ^ { n } }$ is the Euclidean case; it is again a metric, being a positive multiple of one. Since $\| h \| _ { L ^ { 2 } ( I ) } \leq \sqrt { \mathrm { V o l } ( I ) } \| h \| _ { \mathcal { C } ^ { 0 } ( I ) }$ , we have ${ \mathcal W } _ { 2 } ^ { L ^ { 2 } ( I ) } \le \sqrt { \mathrm { V o l } ( I ) } { \mathcal W } _ { 2 } ^ { \mathcal C ^ { 0 } ( I ) }$ , so the supremum-norm statement is the strongest ofthe twofunctional bounds.

Remark 1 (coupling convention). Since $\mathcal { W } _ { 2 } ^ { E }$ is an infimum over couplings, a single explicit coupling bounds $\mathcal { W } _ { 2 } ^ { E }$ from above (Proposition $\bar { 2 } ( l ) )$ . Throughout, $( f , { \bar { G } } )$ therefore denotes a pair defined on a common probability space with the laws ofDefinitions 1 and 2: all bounds on $\| { \dot { f } } - G \| _ { E }$ are provedfor almost every realisation, and the expectation and the infimum are taken only at the end. Further properties of $\mathcal { W } _ { 2 }$ are collected in Appendix A.

## 2.3 The finite-sample bound and its data dependence

Quadratic Wasserstein distance is well suited for Gaussian approximation since the distance between two centred Gaussian laws is bounded by the Frobenius distance of the square roots of their covariances, so covariance perturbations can be tracked explicitly. This is the mechanism behind the bound of Basteri and Trevisan [2024], revisited in Theorem 1. We state their result under the following standard assumptions about the activation function.

Assumption 1 (activation function). The activation $\phi : \mathbb { R }  \mathbb { R }$ of Definition 1 satisfies:

(1) Polynomial growth. There are $A , p < \infty$ with $| \phi ( z ) | \leq A ( 1 + | z | ) ^ { p } f o r a l l \ : z \in \mathbb { R }$

(2) Locally Lipschitz with derivative of polynomial growth. ϕ is absolutely continuous and there are $A ^ { \prime } , p ^ { \prime } < \infty$ with $| \phi ^ { \prime } ( z ) | \leq A ^ { \prime } ( 1 + | z | ) ^ { p ^ { \prime } }$ for almost every $z \in \mathbb { R }$

(3) Non-degeneracy. For every $x \in I$ the variance $\kappa ( x , x )$ is strictly positive and ϕ does not vanish $\mathcal { N } ( 0 , \kappa ( x , x ) )$ almost everywhere.

Assumption 1 is satisfied by tanh, the logistic sigmoid, erf, ReLU and its leaky and smooth variants (GELU, SiLU, softplus), by every polynomial, and by products such as $z \operatorname { t a n h } ( z )$ . It excludes exponential growth, which violates Assumption 1(1); unbounded derivatives at the origin such as $| z | ^ { \dot { \alpha } }$ with $\alpha \in ( 0 , 1 )$ , which violate Assumption 1(2); and jump discontinuities, which are not absolutely continuous. Assumption 1(3) also involves the bias and the domain: since $\begin{array} { r } { \kappa ( x , x ) = \frac { \sigma _ { w } ^ { 2 } } { d } \| x \| ^ { 2 } + \sigma _ { b } ^ { 2 } } \end{array}$ , it holds for all the above activations if and only if $\sigma _ { b } > 0$ or $0 \not \in I .$

Theorem 1 (adapted from [Basteri and Trevisan, 2024, Theorem 3.1]). Let $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset I$ be afinite sample set, let f be the network ofDefinition 1 and let G be the Gaussian process ofDefinition 2. Under Assumption $I , f o r$ every width m $\in \mathbb { N }$

$$
\mathcal { W } _ { 2 } \big ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \big ) \ \leq \ \frac { C ( \phi , \mathcal { X } ) } { \sqrt { m } } ,\tag{4}
$$

with the explicit constant

$$
C ( \phi , \mathcal { X } ) = \frac { \sqrt { \mathbb { E } \Big [ \big \| \tau _ { a } ^ { 2 } \phi ( Z ) \phi ( Z ) ^ { \top } - K [ \mathcal { X } ] \big \| ^ { 2 } \Big ] } } { \sqrt { \lambda ^ { + } \big ( K [ \mathcal { X } ] \big ) } } , \qquad Z \sim \mathcal { N } \big ( 0 , \kappa [ \mathcal { X } ] \big ) \in \mathbb { R } ^ { n } .\tag{5}
$$

The output bias contributes the same independent translation to $f [ \mathcal { X } ]$ and to $G [ \mathcal { X } ]$ , so by Proposition $2 \left( 3 \right)$ it cancels and the constant involves $K [ \mathcal { X } ]$ rather than $K _ { c } [ \mathcal { X } ] . C ( \phi , \mathcal { X } )$ depends only on the activation ϕ, the input set X and the variances $\sigma _ { w } ^ { 2 } , \sigma _ { b } ^ { 2 } , \tau _ { a } ^ { 2 } ,$ , but not on the width m.

Theorem 3.1 of Basteri and Trevisan [2024] makes the stronger assumption that ϕ is globally Lipschitz. The proof under Assumption 1 is given in Appendix B. The bound of Theorem 1 holds at any fixed finite sample set, but its constant $C ( \phi , \bar { \chi } )$ depends on that set. Section 3 evaluates $f$ and G on grids that become dense in I, and hence arbitrarily large, so the growth of $C ( \phi , \mathcal { X } )$ decides whether a functional comparison is possible. This remainder of this section determines how $C ( \phi , \mathcal { X } )$ behaves: the numerator of (5) grows at most linearly in the number of inputs, while the denominator admits no lower bound in general, so that enlarging the input set can render (4) vacuous. A lower bound on the denominator must therefore be assumed. Linear growth in $\sqrt { n }$ is the most that can be demanded, since $\lambda ^ { + } ( K [ \mathcal { X } ] )$ never exceeds tr $K [ \mathcal { X } ]$ , which grows linearly under Assumption 1.

Assumption 2 (effective-rank condition). A collection $\mathcal { F }$ of finite sample sets $\mathcal { X } \subset I$ satisfies the effective-rank condition with constant $\nu > 0 i f ,$ writing $n = | \mathcal { X } |$

$$
\lambda ^ { + } \bigl ( K [ \mathcal { X } ] \bigr ) \ \geq \ \nu n \qquad f o r e \nu e r y \ \mathcal { X } \in \mathcal { F } .\tag{6}
$$

Assumption 2 excludes every activation for which the rank $( K [ \mathcal { X } ] )$ is unbounded (see proof of Proposition 1 for details). In particular every activation whose kernel K is strictly positive definite on I is excluded, since then rank $\mathbf { \partial } ( K [ \mathcal { X } ] ) = | \mathcal { X } |$ <sup>1</sup> The condition is therefore a finite-rank condition, and Lemma 1 verifies it for polynomial activations on the grids of

$$
\begin{array} { r l r } { { 1 } | f - G | | _ { \mathcal { C } ^ { 0 } ( I ) } ~ } & { \stackrel { \mathrm { L e m m a ~ 2 } } { \lesssim } ~ } & { \frac { \mathrm { D i s c r e t i z a t i o n } } { \mathrm { L e m m a ~ } ^ { 3 } } ~ } & { \stackrel { \mathrm { C o u p l i n g ~ + U n i f o r m ~ W a s e c r s t e i n ~ D i s t a n c e } } { \lesssim } ~ } \\ { | | f - G | | _ { \mathcal { C } ^ { 0 } ( I ) } ~ } & { \stackrel { \mathrm { L e m m a ~ 2 } } { \lesssim } ~ } & { \| f - G \| _ { L ^ { 2 } ( I ) } ~ } & { \stackrel { \mathrm { L e m m a ~ 3 } } { \lesssim } ~ } & { \| ( f - G ) [ \mathcal { X } _ { k } ] \| _ { n } ~ } \end{array}
$$

Figure 1: Roadmap of Section 3. Each inequality bounds the quantity on its left by the one on its right, and the reference names the result that does it. A norm equivalence bounds the supremum norm by the $L ^ { 2 } ( I )$ norm, a discretization bounds that by the grid values, and a coupling bounds those by the width. Composing the three steps proves Theorem 2.

Section 3. Under Assumption 1 the numerator of the constant in Equation (5) grows at most linearly in $n ,$ while under Assumption 2 the denominator grows like ${ \sqrt { n } } .$ , leaving the bound of Theorem 1 with a factor $\sqrt { n }$ and a constant that depends neither on the sample set nor on the width. The following proposition makes these statements precise, and is proved in Appendix B.

Proposition 1 (the constant of Theorem 1). Let $\mathcal { X } \subset I$ be afinite sample set with $n = | \mathcal { X } |$

1. Under Assumption 1 with growth constants $A , p ,$ the numerator of (5),

$$
\begin{array} { r } { N : = \sqrt { \mathbb { E } \Big [ \big \| \tau _ { a } ^ { 2 } \phi ( Z ) \phi ( Z ) ^ { \top } - K [ \mathcal { X } ] \big \| ^ { 2 } \Big ] } , \qquad Z \sim \mathcal { N } \big ( 0 , \kappa [ \mathcal { X } ] \big ) \in \mathbb { R } ^ { n } , } \end{array}
$$

satisfies $N \leq \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } } n ,$ , where

$$
M _ { s } : = \mathbb { E } \big [ ( 1 + | \xi | ) ^ { s } \big ] < \infty \quad f o r \ : s > 0 \ : a n d \xi \sim \mathcal { N } ( 0 , \kappa _ { I } ^ { 2 } ) \ : w i t h \ : \kappa _ { I } ^ { 2 } : = \operatorname* { m a x } _ { x \in I } \kappa ( x , x ) < \infty .
$$

2. Ifin addition $K [ \mathcal { X } ]$ satisfies Equation (6) with constant $\nu > 0 _ { : }$ , thenfor every width $m \in \mathbb { N } ,$

$$
W _ { 2 } \big ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \big ) \ \leq \ D \sqrt { \frac { n } { m } } ,
$$

$$
w i t h D : = \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } / \nu } .
$$

The constants depend only on ϕ through $A , p ,$ , on $I ,$ on the variances $\sigma _ { w } ^ { 2 } , \sigma _ { b } ^ { 2 } , \tau _ { a } ^ { 2 }$ and, in part $( 2 ) _ { : }$ , on $\nu ,$ and neither on X nor on m.

Proposition 1 shows that the bound of Theorem 1 grows with the number of inputs: the width m must grow faster than n for it to be informative. This is the main limitation of finite-sample bounds. Section 3 removes that growth and compares f and G as random functions.

## 3 From a Finite-Input Bound to Functional Convergence

Measured per input rather than in total, the bound of Proposition 1 gives,

$$
{ \mathcal W } _ { 2 , n } \bigl ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \bigr ) \ \le \ \frac { D } { \sqrt { m } } , \qquad D : = \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } / \nu } .\tag{7}
$$

The right-hand side of 7 does not depend on the sample set. In the unrescaled form, the bound $D _ { \sqrt { n / m } }$ is informative only when $n \ll m$ , so the width must grow with the number of inputs for this bound to be useful, as discussed in the introduction. In the rescaled distance, the width alone governs the per-sample approximation error. But rescaling presupposes n, and (7) is still a statement about a random vector in $\mathbb { R } ^ { n }$ . This section uses (7) to get a bound on $f$ and $\breve { G }$ as random functions.

Converting this bound into a statement about the function is a chain of three steps, shown in Figure 1, where each arrow replaces the quantity on its left by a larger one. (i) A norm equivalence bounds the supremum norm by the $L ^ { 2 } ( I )$ norm on the finite-dimensional space when ϕ is a polynomial. (ii) A discretization estimate bounds the $L ^ { \dot { 2 } } ( I )$ norm by the values on a finite grid, up to a mesh error that shrinks as the grid is refined. Refining the grid adds inputs, so with the unrescaled constant the mesh error and the constant cannot be made small at once. This is the obstruction identified in Basteri and Trevisan [2024, Section 5], which is why the functional bound is only sketched there. Since the right-hand side of (7) does not grow with the number of inputs, the grid may be refined freely and the mesh error sent to zero. (iii) An explicit coupling then identifies the remaining grid term with the left-hand side of (7). The three bounds are deterministic and are applied pathwise to $h = f - G$ , for the coupled pair of Remark 1, with the expectation taken only afterwards.

We take I to be a hyperrectangle, so that the grids are products of d one-dimensional grids. The product structure gives the covering property that the discretization step needs (Remark 2 in Apendix A). For a polynomial activation these grids also satisfy Assumption 2 with one constant ν for all $k \geq k _ { 1 }$ (Lemma 1). The hyperrectangle restriction is removed afterwards by Corollary 1.

Definition 4 (midpoint grid). Assume $I = I _ { 1 } \times \cdots \times I _ { d }$ be a d-dimensional hyperrectangle, with $I _ { j } = [ a _ { j } , b _ { j } ]$ $| I _ { j } | : = b _ { j } - a _ { j } > 0$ and Vo $\begin{array} { r } { ( I ) : = \prod _ { j = 1 } ^ { d } | I _ { j } | } \end{array}$ . For $k \in \mathbb N$ let

$$
\begin{array} { r } { \mathcal { X } _ { k } ^ { ( j ) } = \left\{ a _ { j } + \left( l - \frac { 1 } { 2 } \right) \frac { | I _ { j } | } { k } ~ : ~ l = 1 , \ldots , k \right\} \subset I _ { j } } \end{array}
$$

be the one-dimensional midpoint grid of order k on $I _ { j } ,$ , the superscript indexing the coordinate. The midpoint grid of order k on I is

$$
\mathcal { X } _ { k } = \mathcal { X } _ { k } ^ { ( 1 ) } \times \cdots \times \mathcal { X } _ { k } ^ { ( d ) } \subset I .\tag{8}
$$

In particular $n = | { \mathcal { X } } _ { k } | = k ^ { d } .$ : the subscript k counts points per axis, not the total number ofpoints. Further properties of $\mathcal { X } _ { k }$ are collected in Appendix A.

The grids of Definition 4 form the collection to which Assumption 2 is applied. For a polynomial activation the following lemma establishes it with a single constant ν, so (7) holds with the same D for every $k ,$ which is what the limit taken in Theorem 2 requires.

Lemma 1 (polynomial activations). Let ϕ be a nonzero polynomial ofdegree r satisfying Assumption 1, and let $\mathcal { X } _ { k }$ be the midpoint grids ofDefinition 4. Then rank $( K [ \mathcal { X } _ { k } ] ) \le \binom { r + d } { d }$ for every $k ,$ and there are $\nu > 0$ and $k _ { 1 } \in \mathbb { N }$ such that $\mathcal { X } _ { k }$ satisfies the effective-rank condition (Assumption 2) with constant ν for every $k \geq k _ { 1 }$

The proof of Lemma 1 factorises the kernel matrix as $K [ \mathcal { X } _ { k } ] = V _ { k } M V _ { k } ^ { \top }$ , where the rows of $V _ { k } \in \mathbb { R } ^ { k ^ { d } \times R }$ are the monomials of total degree at most r evaluated at the grid points, $R = { \binom { r + d } { d } }$ , and $M \in \mathbb { R } ^ { R \times R }$ is positive semi-definite and independent of k. Hence rank $( K [ { \mathcal { X } } _ { k } ] ) \leq \mathrm { r a n k } ( M ) \leq R .$ however fine the grid is. The nonzero eigenvalues of $K [ \mathcal { X } _ { k } ]$ coincide with those of the $R \times R$ matrix $k ^ { d } M ^ { \mathrm { { 1 / 2 } } } \big ( \frac { 1 } { k ^ { d } } V _ { k } ^ { \top } V _ { k } \big ) M ^ { \mathrm { { 1 / 2 } } }$ , whose middle factor ${ \frac { 1 } { k ^ { d } } } V _ { k } ^ { \top } V _ { k }$ is a midpoint Riemann sum. Its limit is positive definite, which yields $\dot { \lambda ^ { + } } ( K [ \mathcal { X } _ { k } ] ) \geq \nu k ^ { d }$ for all $k \geq k _ { 1 }$

Our main result, Theorem $^ { 2 , }$ provides a functional Wasserstein bound in the supremum norm $( \mathcal { W } _ { 2 } ^ { \mathcal { C } ^ { 0 } ( I ) } ( f , G ) )$ . This bound compares the neural network and its NNGP as random continuous functions on a compact interval for polynomial activation functions.

Theorem 2 (main result: functional bound with $m ^ { - 1 / 2 }$ rate). Let ϕ be a nonzero polynomial of degree r satisfying Assumption 1, let $I = I _ { 1 } \times \cdots \times I _ { d }$ be a hyperrectangle. Then for every $m \in \mathbb { N }$

$$
\mathcal { W } _ { 2 } ^ { \mathcal { C } ^ { 0 } ( I ) } ( f , G ) \leq \frac { C ^ { * } } { \sqrt { m } } , \qquad C ^ { * } : = ( r + 1 ) ^ { d } \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { \frac { 2 M _ { 4 p } } { \nu } } ,\tag{9}
$$

where $A , p$ are the growth constants of Assumption $I , M _ { 4 p }$ is the moment of Theorem 2, and $\nu > 0$ is the effective-rank constant supplied by Lemma 1.

The supremum norm only decreases when the domain is made smaller, so Theorem 2 carries over from the hyperrectangle to every compact set it contains, with the same constant (Corollary 1).

Corollary 1. Let $S \subset \mathbb { R } ^ { d }$ be compact and let I be a hyperrectangle containing S. Under the hypotheses ofTheorem 2 on I,

$$
{ \mathcal W } _ { 2 } ^ { \mathcal C ^ { 0 } ( S ) } ( f , G ) \le { \mathcal W } _ { 2 } ^ { \mathcal C ^ { 0 } ( I ) } ( f , G ) = O \big ( m ^ { - 1 / 2 } \big ) .
$$

Since every compact set is contained in a hyperrectangle, the rate $m ^ { - 1 / 2 }$ holds on any compact set of inputs; only the constant depends on the enclosing hyperrectangle.

## 3.1 Proof Sketch

Each step of Figure 1 is one lemma, and the final paragraph combines them. All proofs of this section are given in Appendix C.

Supremum norm to $L ^ { 2 }$ norm. For a polynomial activation the sample paths of f and of G lie in a space of polynomials, on which the supremum norm and the $L ^ { 2 }$ norm are comparable.

Lemma 2 (norm equivalence for polynomial activations). Let $I = I _ { 1 } \times \cdots \times I _ { d }$ be a hyperrectangle, let $r \in  { \mathbb { N } } _ { 0 }$ , and let ${ \mathcal P _ { r } }$ denote the space ofreal polynomials on I oftotal degree at most r. For every $p \in \mathcal { P } _ { r }$

$$
\| p \| _ { \mathcal { C } ^ { 0 } ( I ) } \leq \frac { ( r + 1 ) ^ { d } } { \sqrt { \mathrm { V o l } ( I ) } } \| p \| _ { L ^ { 2 } ( I ) } .\tag{10}
$$

Ifϕ is a polynomial ofdegree $^ { r , }$ then f has sample paths in $\mathcal { P } _ { r }$ almost surely, and G admits a version with the same property.

Reduction to the grid (Discretization). The $L ^ { 2 } ( I )$ norm is controlled on the midpoint grid of Definition 4. The grid constrains h at finitely many points only, so it says nothing about h in between unless h is prevented from moving sharply. The Hölder seminorm measures that movement.

Definition 5 (Hölder spaces). Let $S \subset \mathbb { R } ^ { d }$ be compact and let $\gamma \in ( 0 , 1 ]$ . For $h : S  \mathbb { R }$ set

$$
\| h \| _ { \mathcal { C } ^ { 0 } ( S ) } = \operatorname* { s u p } _ { x \in S } | h ( x ) | , \qquad [ h ] _ { \mathcal { C } ^ { \gamma } ( S ) } = \operatorname* { s u p } _ { x , y \in S } \frac { | h ( x ) - h ( y ) | } { \| x - y \| ^ { \gamma } } , \qquad \| h \| _ { \mathcal { C } ^ { \gamma } ( S ) } = \| h \| _ { \mathcal { C } ^ { 0 } ( S ) } + [ h ] _ { \mathcal { C } ^ { \gamma } ( S ) } .
$$

The space $\mathcal { C } ^ { \gamma } ( S )$ consists of all $h : S  \mathbb { R }$ with $\| h \| _ { C ^ { \gamma } ( S ) } < \infty$

A finite $[ h ] _ { C ^ { \gamma } ( S ) }$ means that h changes by at most $[ h ] _ { \mathscr { C } ^ { \gamma } ( S ) } \| x - y \| ^ { \gamma }$ between any two points of S. The seminorms of f and of G have finite second moment at every fixed width, by the regularity of ϕ (Assumption 1) for f and by the Kolmogorov–Chentsov theorem [Revuz and Yor, 1999] for G. Both are verified in Appendix C, as part of the proof of Theorem 2.

The midpoint grid has the covering property: every point of I lies within a distance $\rho _ { k } = { \cal O } ( k ^ { - 1 } )$ of a grid point (Remark 2). A Hölder function therefore differs little from its value at the nearest grid point, and comparing h with that value bounds the $L ^ { 2 } ( I )$ norm by the grid values, up to an error governed by $\rho _ { k }$ and by the Hölder seminorm of h.

Lemma 3 (Discretization). Let $\gamma \in ( 0 , 1 ]$ , let $h \in \mathcal { C } ^ { \gamma } ( I _ { 1 } \times \cdots \times I _ { d } )$ and let $\mathcal { X } _ { k }$ be the midpoint grid of order k on $I _ { 1 } \times \cdots \times I _ { d } .$ . Using $h [ \mathcal { X } _ { k } ] = \left( h ( x _ { 1 } ) , \ldots , h ( x _ { k ^ { d } } ) \right)$ we get

$$
\| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } \le 2 { \frac { \mathrm { V o l } ( I ) } { k ^ { d } } } \| h [ { \mathcal { X } } _ { k } ] \| ^ { 2 } + 2 \mathrm { V o l } ( I ) \rho _ { k } ^ { 2 \gamma } [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } ^ { 2 } .\tag{11}
$$

The proof partitions I into the $k ^ { d }$ cells of the grid, each of volume $\mathrm { V o l } ( I ) / k ^ { d }$ and each containing its grid point as midpoint, and compares h with that value on each cell. Two properties of (11) matter. The mesh error (2 Vol(I) $\rho _ { k } ^ { 2 \gamma } [ h ] _ { C ^ { \gamma } ( I ) } ^ { 2 } )$ vanishes as the grid is refined, since $\rho _ { k } = { \cal O } ( k ^ { - 1 } )$ . The first term $2 \operatorname { V o l } ( I ) \| h [ \mathcal { X } _ { k } ] \| _ { n } ^ { 2 }$ carries the grid values in the rescaled norm of Definition 3, which averages over the grid instead of summing. This is the norm in which Equation (7) is uniform in the number of inputs, so refining the grid shrinks the mesh error without inflating the first term. The two limits of m and n therefore decouple, and this is the step for which Section 2.3 is needed.

Coupling The grid term of (11) is a second moment, not a Wasserstein distance. The two agree when the grid evaluations are optimally coupled, and such a coupling of the full processes exists.

Lemma 4 (Coupling the Processes). Let $\mathcal { X } \subset I$ be afinite sample set with $n = | \mathcal { X } |$ . There is a pair $( \tilde { f } , \tilde { G } )$ ofrandom elements $o f { \mathcal { C } } ^ { 0 } ( I )$ with $\tilde { f } \ \stackrel { d } { = } \ f$ and ${ \tilde { G } } { \stackrel { d } { = } } G$ , whose evaluations $\left( \tilde { f } [ \mathcal { X } ] , \tilde { G } [ \mathcal { X } ] \right)$ form an optimal coupling ofthe laws of $f [ \mathcal { X } ]$ and $G [ \mathcal { X } ]$ . Consequently

$$
\mathcal { W } _ { 2 , n } \big ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \big ) ^ { 2 } \ = \ \mathbb { E } \Big [ \big \lVert \tilde { f } [ \mathcal { X } ] - \tilde { G } [ \mathcal { X } ] \big \rVert _ { n } ^ { 2 } \Big ] .
$$

An optimal coupling of $f [ \mathcal { X } ]$ and $G [ \mathcal { X } ]$ exists by Villani [2008, Theorem 4.1], and the proof extends it from these n values to the whole functions by drawing f and G independently from their conditional laws given their values on X, which leaves the law of each unchanged.

Completing proof of Theorem 2. We can now put the pieces together. Fix a midpoint grid $\mathcal { X } _ { k }$ and let $( \tilde { f } , \tilde { G } )$ be the coupled pair of Lemma 4, so that $h = f - G$ is a single random function. Lemma 2 bounds $\| h \| _ { \mathcal { C } ^ { 0 } ( I ) } \operatorname { b y } \| h \| _ { L ^ { 2 } ( I ) }$ , and Lemma 3 bounds $\| h \| _ { L ^ { 2 } ( I ) }$ by the values on the grid plus a mesh error. Taking expectations, Lemma 4 turns the grid term into $\mathcal { W } _ { 2 , n } \big ( f [ \mathcal { X } _ { k } ] , G [ \mathcal { X } _ { k } ] \big ) ^ { 2 }$ , which Equation 7 bounds by $D ^ { 2 } / m ,$ the same bound for every k, its effective-rank hypothesis supplied by Lemma 1. The mesh error, by contrast, vanishes as the grid is refined, since $\rho _ { k } = { \cal O } ( k ^ { - 1 } )$ while the Hölder seminorms of f and of G have finite second moment at the fixed width m and do not depend on k. Letting $k \to \infty$ at fixed m therefore leaves only the width.

## 4 Discussion

Finite-sample bounds have two limitations: their constant depends on the sample set, so a guarantee transferred from the infinite-width limit weakens as the sample grows; and they compare two random vectors, not the two random functions. Section 2.3 shows a simple way to remove the first by measuring the error per input, and Section 3 shows how to remove the second by refining the sample set until it is dense in the domain. For polynomial activations, the result is a functional bound in the quadratic Wasserstein distance with respect to the supremum norm, at the finite-sample rate $m ^ { - 1 / 2 }$ , with an explicit constant, on every compact set of inputs.

Beyond Polynomial Activations. The effective-rank condition of Assumption 2 holds for polynomial activations but fails whenever $K [ \mathcal { X } ]$ has unbounded rank, in particular for strictly positive definite kernels. For such activations it must be replaced by a weaker spectral lower bound $\lambda ^ { + } ( K [ { \mathcal { X } } ] ) \geq { \dot { \nu } } { \dot { n } } ^ { 1 - \alpha }$ for some $\alpha \geq 0$ , with $\alpha = 0$ recovering Assumption 2. Every step of the roadmap in Figure 1 other than the norm equivalence of Lemma 2 requires only Assumption 1, and Lemma 2 can be replaced by an interpolation inequality.

Lemma 5 (interpolation). $L e t \gamma \in ( 0 , 1 ]$ and let $I = I _ { 1 } \times \cdots \times I _ { d }$ be a hyperrectangle. There are constants $C _ { 1 } , C _ { 2 } < \infty$ depending only on d and on $\operatorname* { m i n } _ { j } | { \dot { I } } _ { j } |$ such thatfor every $h \in \mathcal { C } ^ { \gamma } ( I )$

$$
\| h \| _ { \mathcal { C } ^ { 0 } ( I ) } \le C _ { 1 } \| h \| _ { L ^ { 2 } ( I ) } ^ { \frac { 2 \gamma } { 2 \gamma + d } } [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } ^ { \frac { d } { 2 \gamma + d } } + \frac { C _ { 2 } } { \sqrt { \mathrm { V o l } ( I ) } } \| h \| _ { L ^ { 2 } ( I ) } .
$$

The proof of Lemma 5 is in Appendix D. The $L ^ { 2 } ( I )$ norm carries the exponent $\frac { 2 \gamma } { 2 \gamma + d }$ , which is smaller than one and decreases with the dimension, and the Hölder seminorm carries the remainder. The seminorm now enters the rate, so it must be bounded uniformly in m.

With $\alpha > 0$ , the right-hand side of (7) is no longer uniform in the number of inputs, and the grid can no longer be refined freely after the discretization step. The grid term now grows with k while the mesh error decays, so instead of letting $k \to \infty$ one has to choose k as a function of m to balance the two. The resulting rate is slower than $m ^ { - 1 / 2 }$ , and it deteriorates with the dimension d through the exponent $2 \gamma / ( 2 \gamma + d )$ of the interpolation inequality.

Sharpness of the rate. Whether the rate in Theorem 2 is sharp remains open. A lower bound does not require a further lifting argument: for every $x \in I$

$$
\mathcal { W } _ { 2 } \big ( f ( x ) , G ( x ) \big ) \leq \mathcal { W } _ { 2 } ^ { \mathcal { C } ^ { 0 } ( I ) } ( f , G ) .
$$

Combining this inequality with single-input bound of Favaro et al. [2025, Theorem $3 . 3 ( 2 ) ]$ yields a lower bound of order $m ^ { - 1 }$ . Since this bound uses only one input, it does not establish whether the rate $m ^ { - 1 / 2 }$ is sharp for general n. The upper bound in Theorem 2, in contrast, inherits the rate of the corresponding finite-input bound. For finite input sets, a rate of $m ^ { - 1 }$ is available when $K [ \mathcal { X } ]$ is invertible [Trevisan, 2023]. When it is singular, the best known rate is $m ^ { - 1 / 2 }$ . Our setting falls into the latter case: the effective-rank condition imposes a rank bound on $K [ \mathcal { X } ]$ , precluding invertibility for sufficiently large input sets. Thus, the available finite-input bounds yield the rate $m ^ { - 1 / 2 }$

Deeper networks. Our analysis covers only one hidden layer. The steps in Section 3 that convert a finite-sample bound into a functional one carry over to deeper networks with polynomial activations. The finite-sample bound of Basteri and Trevisan [2024, Theorem 3.1] also holds at any depth, but its proof propagates the comparison through the layers and therefore requires a Lipschitz activation, while our functional argument requires a polynomial one. Extending Theorem 2 to deeper networks therefore requires a propagation step that does not rely on Lipschitz continuity.

The metric. Our bound controls the function values but not their derivatives. The only stronger metric in Table 1 is the $\mathcal { C } ^ { k }$ norm of Favaro et al. [2025], obtained at the slower rate $m ^ { - 1 / 8 }$ , and whether $m ^ { - 1 / 2 }$ is attainable in such a norm is open in general.

Implications for tensor programs. Tensor programs [Yang, 2019] extend the Gaussian process limit to essentially all standard architectures, and Agazzi et al. [2026] recently made this limit quantitative, recovering the $m ^ { - 1 / 2 }$ rate of Basteri and Trevisan [2024] for general programs. The bounds of Agazzi et al. [2026] are finite-sample statements: they hold on a fixed set of inputs, with a constant that depends implicitly on that set. Our approach suggests how to lift such bounds to functional statements for tensor programs, yielding convergence rates that do not deteriorate as the evaluation set grows, provided their dependence on the number of inputs can be made explicit.

## AI Disclosure

ChatGPT (OpenAI) and Claude (Anthropic) were used as auxiliary tools for language editing, literature discovery, and typesetting support. Each reference and bibliographic item identified through these tools was independently verified against primary sources, and we take full responsibility for the final content and accuracy of the document.

## References

Andrea Agazzi, Eloy Mósig García, and Dario Trevisan. Quantitative Gaussian-process limits of Tensor Programs. CoRR, 2026.

Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 322–332, 2019.

Nil Ayday, Lingchu Yang, and Debarghya Ghoshdastidar. Gaussian process limit reveals structural benefits of graph transformers. In Conference on Uncertainty in Artificial Intelligence, 2026.

Krishnakumar Balasubramanian and Nathan Ross. Finite-dimensional Gaussian approximation for deep neural networks: Universality in random weights. CoRR, 2025.

Krishnakumar Balasubramanian, Larry Goldstein, Nathan Ross, and Adil Salim. Gaussian random field approximation via Stein’s method with applications to wide random neural networks. CoRR, 2023.

Andrea Basteri and Dario Trevisan. Quantitative Gaussian approximation of randomly initialized deep neural networks. Machine Learning, 113(9):6373–6393, 2024.

Alberto Bordino, Stefano Favaro, and Sandra Fortini. Non-asymptotic approximations of Gaussian neural networks via second-order poincaré inequalities. In Symposium on Advances in Approximate Bayesian Inference, volume 253 of Proceedings ofMachine Learning Research, pages 45–78, 2024.

Daniele Bracale, Stefano Favaro, Sandra Fortini, and Stefano Peluchetti. Large-width functional asymptotics for deep Gaussian neural networks. In International Conference on Learning Representations, 2021.

Lucia Celli and Giovanni Peccati. Entropic bounds for conditionally Gaussian vectors and applications to neural networks. CoRR, 2025.

Wuyang Chen, Xinyu Gong, and Zhangyang Wang. Neural architecture search on imagenet in four GPU hours: A theoretically inspired perspective. In International Conference on Learning Representations, 2021.

Ronen Eldan, Dan Mikulincer, and Tselil Schramm. Non-asymptotic approximations of neural networks by Gaussian processes. In Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 1754–1775, 2021.

S. Favaro, B. Hanin, D. Marinucci, I. Nourdin, and G. Peccati. Quantitative CLTs in deep neural networks. Probability Theory and Related Fields, 191(3):933–977, 2025.

Boris Hanin. Random neural networks in the infinite width limit as Gaussian processes. CoRR, 2021.

Jiri Hron, Yasaman Bahri, Jascha Sohl-Dickstein, and Roman Novak. Infinite attention: NNGP and NTK for deep attention networks. In International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4376–4386, 2020.

Olav Kallenberg. Foundations ofmodern probability. Probability and its Applications. Springer-Verlag, New York, second edition, 2002.

Adam Klukowski. Rate of convergence of polynomial networks to Gaussian processes. In Conference on Learning Theory, Proceedings of Machine Learning Research, pages 701–722, 2022.

Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein Deep neural networks as Gaussian processes. In nternational Conference on Learning Representations, 2018.

Alexander G. de G. Matthews, Jiri Hron, Mark Rowland, Richard E. Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations, 2018.

Radford M. Neal. Priors for Infinite Networks, pages 29–53. Springer New York, 1996.

Timothy Nguyen, Roman Novak, Lechao Xiao, and Jaehoon Lee. Dataset distillation with infinitely wide convolutional networks. In Neural Information Processing Systems, pages 5186–5198, 2021.

L. Nirenberg. On Elliptic Partial Differential Equations. Springer, 2011.

Zehao Niu, Mihai Anitescu, and Jie Chen. Graph neural network-inspired kernels for Gaussian processes in semisupervised learning. In International Conference on Learning Representations, 2023.

Daniel Revuz and Marc Yor. Continuous Martingales and Brownian Motion, volume 293 of Grundlehren der mathematischen Wissenschaften. Springer, 3 edition, 1999.

Dario Trevisan. Wide deep neural networks with Gaussian weights are very close to Gaussian processes. 2023.

Cédric Villani. Optimal transport: Old and new. 2008.

Christopher K. I. Williams. Computing with infinite networks. In Advances in Neural Information Processing Systems, pages 295–301. MIT Press, 1996.

Greg Yang. Tensor programs I: wide feedforward or recurrent neural networks of any architecture are Gaussian processes. CoRR, 2019.

Greg Yang and Edward J. Hu. Tensor programs IV: feature learning in infinite-width neural networks. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11727–11737, 2021.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs V: tuning large neural networks via zero-shot hyperparameter transfer. CoRR, 2022.

## A Auxiliary results

Proposition 2 (properties of $\mathcal { W } _ { 2 } )$ . Let $X , Y , Z$ be E-valued random variables with finite second moments.

1. (coupling bound) For any pair $( X , Y )$ on a common probability space,

$$
\mathcal { W } _ { 2 } ^ { E } ( X , Y ) \leq \sqrt { \mathbb { E } [ \| X - Y \| _ { E } ^ { 2 } ] }
$$

2. (triangle inequality) $\mathcal { W } _ { 2 } ^ { E } ( X , Z ) \leq \mathcal { W } _ { 2 } ^ { E } ( X , Y ) + \mathcal { W } _ { 2 } ^ { E } ( Y , Z )$

3. (independent translations) $I f Z$ is independent of X and of Y , then

$$
\mathcal { W } _ { 2 } ^ { E } ( X + Z , Y + Z ) \leq \mathcal { W } _ { 2 } ^ { E } ( X , Y ) .
$$

4. (convexity) $I f Z$ takes values in some Euclidean space, then

$$
\left( \mathcal { W } _ { 2 } ^ { E } ( X , Y ) \right) ^ { 2 } \leq \int \left( \mathcal { W } _ { 2 } ^ { E } { \left( \mathbb { P } _ { X | Z = z } , \mathbb { P } _ { Y } \right) } \right) ^ { 2 } d \mathbb { P } _ { Z } ( z ) .
$$

5. (Gaussian comparison) $I f E = \mathbb { R } ^ { n }$ and $X , Y$ are centred Gaussian, then

$$
\mathcal { W } _ { 2 } ( X , Y ) \leq \| \sqrt { \Sigma ( X ) } - \sqrt { \Sigma ( Y ) } \| .
$$

Remark 2 (properties of the midpoint grids). $\begin{array} { r } { \pmb { I } , | \mathcal { X } _ { k } | = k ^ { d } . } \end{array}$ the subscript k counts points per axis, not the total number ofpoints.

2. For every $x \in I$ there is $x _ { i } \in \mathcal X _ { k }$ with

$$
\| x - x _ { i } \| \leq \rho _ { k } : = \frac { 1 } { 2 k } \Big ( \sum _ { j = 1 } ^ { d } | I _ { j } | ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{12}
$$

Choosing $x _ { i }$ coordinatewise gives $| x _ { j } - ( x _ { i } ) _ { j } | \leq | I _ { j } | / ( 2 k )$ for each j leading to (12).

Lemma 6 (square-root perturbation). Let $A , B \in \mathbb { R } ^ { n \times n }$ be positive semi-definite with range $( A ) \subseteq \mathrm { r a n g e } ( B )$ . Then

$$
\left\| { \sqrt { A } } - { \sqrt { B } } \right\| \leq { \frac { \| A - B \| } { \sqrt { \lambda ^ { + } ( B ) } } } .
$$

Proof. Put $W : = \mathrm { r a n g e } ( B )$ . Since A and B are symmetric with $\mathrm { r a n g e } ( A ) \subseteq W$ , both vanish on $W ^ { \perp }$ and map W into itself, and so do $\sqrt { A }$ and $\sqrt { B }$ . On W the matrix B is positive definite with smallest eigenvalue $\lambda ^ { + } ( B )$ , and the standard bound $\| \sqrt { A } - \sqrt { B } \| \leq \big ( \sqrt { \lambda _ { \operatorname* { m i n } } ( A ) } + \sqrt { \lambda _ { \operatorname* { m i n } } ( B ) } \big ) ^ { - 1 } \| A - B \|$ for positive definite matrices applies there. All four matrices vanish on $W ^ { \perp }$ , so the restriction changes no norm. □

Lemma 7 (range of a second-moment matrix). Let V be a random vector in $\mathbb { R } ^ { n }$ with $\mathbb { E } \Vert V \Vert ^ { 2 } < \infty$ and $\Sigma : = \mathbb { E } [ V V ^ { \top } ]$ Then $V \in { \mathrm { r a n g e } } ( \Sigma )$ almost surely.

Proof. $\mathrm { I f } \Sigma u = 0$ then $\mathbb { E } \big [ ( u ^ { \top } V ) ^ { 2 } \big ] = u ^ { \top } \Sigma u = 0 , \thinspace \mathrm { s o } \ u ^ { \top } V = 0$ almost surely. Applying this to a basis of ker Σ gives $V \perp \mathrm { k e r } \Sigma = \mathrm { r a n g e } ( \Sigma ) ^ { \perp }$ almost surely. □

## B Proofs of Section 2

Theorem 1. Write $\mathbf { 1 } \in \mathbb { R } ^ { n }$ for the all-ones vector and let $Z ^ { ( r ) } \in \mathbb { R } ^ { n }$ denote the r-th row of $\boldsymbol { g } [ \mathcal { X } ] \in \mathbb { R } ^ { m \times n }$ , that is $Z ^ { ( r ) } = \bigl ( g _ { r } ( x _ { 1 } ) , \ldots , g _ { r } ( x _ { n } ) \bigr ) ^ { \top }$

Step 1 (structure of the evaluation vector). Each $g _ { r } ( x ) = \langle W _ { r } , x \rangle + b _ { r }$ is an affine function of x with independent centred Gaussian coefficients, so $Z ^ { ( r ) }$ is a centred Gaussian vector with

$$
\mathrm { C o v } \big ( g _ { r } ( x _ { i } ) , g _ { r } ( x _ { j } ) \big ) = \frac { \sigma _ { w } ^ { 2 } } { d } \langle x _ { i } , x _ { j } \rangle + \sigma _ { b } ^ { 2 } = \kappa ( x _ { i } , x _ { j } )
$$

by (1); the pairs $\left( W _ { r } , b _ { r } \right)$ are independent across r, so $Z ^ { ( 1 ) } , \ldots , Z ^ { ( m ) }$ are i.i.d. $\mathcal { N } ( 0 , \kappa [ \mathcal { X } ] )$ . The input dimension enters only here, through κ. With ϕ acting componentwise,

$$
f [ { \mathcal X } ] = \sum _ { r = 1 } ^ { m } a _ { r } \phi \big ( Z ^ { ( r ) } \big ) + c { \bf 1 } .
$$

By Assumption 1 (1) the vector $\phi ( Z ^ { ( r ) } )$ has finite moments of every order, so all quantities below are finite.

Step 2 (removing the output bias). Since $K _ { c } = K + \tau _ { c } ^ { 2 }$ , we may realise $G [ \mathcal { X } ] = \Gamma + c \mathbf { 1 }$ with $\Gamma \sim \mathcal { N } ( 0 , K [ \mathcal { X } ] )$ independent of $c ,$ and c is independent of $( W , b , a )$ . Writing $\begin{array} { r } { \tilde { f } [ \mathcal { X } ] : = \sum _ { r } a _ { r } \phi ( Z ^ { ( r ) } ) } \end{array}$ , the independent translation c 1 cancels by Proposition 2 (3):

$$
\begin{array} { r } { \mathcal { W } _ { 2 } \big ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \big ) \ \leq \ \mathcal { W } _ { 2 } \big ( \tilde { f } [ \mathcal { X } ] , \Gamma \big ) . } \end{array}
$$

Step 3 (conditional Gaussianity). Condition on $( W , b )$ , equivalently on $Z ^ { ( 1 ) } , \dots , Z ^ { ( m ) }$ . The coefficients $a _ { r }$ are i.i.d. $\mathcal { N } ( 0 , \tau _ { a } ^ { 2 } / m )$ and independent of $( W , b )$ , so conditionally $\tilde { f } [ \mathcal { X } ]$ is a centred Gaussian vector with covariance

$$
{ \widehat K } _ { m } : = \displaystyle \frac { \tau _ { a } ^ { 2 } } { m } \sum _ { r = 1 } ^ { m } \phi \big ( Z ^ { ( r ) } \big ) \phi \big ( Z ^ { ( r ) } \big ) ^ { \top } , \qquad { \mathbb E } \big [ { \widehat K } _ { m } \big ] = \tau _ { a } ^ { 2 } { \mathbb E } \big [ \phi ( Z ) \phi ( Z ) ^ { \top } \big ] = K [ \mathcal { X } ] ,
$$

the last identity being (2). By Proposition 2 (4), applied with the Euclidean conditioning variable $( W , b )$

$$
\mathcal { W } _ { 2 } \big ( \tilde { f } [ \mathcal { X } ] , \Gamma \big ) ^ { 2 } \ \leq \ \mathbb { E } \Big [ \mathcal { W } _ { 2 } \big ( \mathcal { N } ( 0 , \widehat { K } _ { m } ) , \mathcal { N } ( 0 , K [ \mathcal { X } ] ) \big ) ^ { 2 } \Big ] .
$$

Step 4 (covariance perturbation). Apply Lemma 7 to $V = \tau _ { a } \phi ( Z )$ , whose second-moment matrix is $K [ \mathcal { X } ] \colon$ almost surely $\phi ( Z ) \in \mathrm { r a n g e } \big ( K [ { \mathcal { X } } ] \big )$ . Each $Z ^ { ( r ) }$ has the law of $Z ,$ , so almost surely $\mathrm { r a n g e } ( { \widehat { K } } _ { m } ) \subseteq \mathrm { r a n g e } ( K [ { \mathcal { X } } ] )$ . Assumption $1 \left( 3 \right)$ gives $K ( x , x ) > 0$ , so $K [ \mathcal { X } ] \neq 0$ and $\lambda ^ { + } ( K [ \mathcal { X } ] )$ is defined. Proposition $2 \left( 5 \right)$ followed by Lemma 6 therefore yields

$$
\mathcal { W } _ { 2 } \big ( \mathcal { N } ( 0 , \widehat { K } _ { m } ) , \mathcal { N } ( 0 , K [ \mathcal { X } ] ) \big ) \ \leq \ \big \| \sqrt { \widehat { K } _ { m } } - \sqrt { K [ \mathcal { X } ] } \big \| \ \leq \ \frac { \big \| \widehat { K } _ { m } - K [ \mathcal { X } ] \big \| } { \sqrt { \lambda ^ { + } \big ( K [ \mathcal { X } ] \big ) } } .
$$

Step 5 (thefluctuation of $\widehat { K } _ { m } ) .$ . The matrices $Y _ { r } : = \tau _ { a } ^ { 2 } \phi ( Z ^ { ( r ) } ) \phi ( Z ^ { ( r ) } ) ^ { \top }$ are i.i.d. with mean $K [ \mathcal { X } ]$ , and $\widehat { K } _ { m } - K [ \mathcal { X } ] =$ $\begin{array} { r } { \frac { 1 } { m } \bar { \sum _ { r } } ( Y _ { r } - K [ \mathcal { X } ] ) } \end{array}$ . Expanding the Frobenius norm and using that the summands are independent and centred, the cross terms vanish, so

$$
\mathbb { E } \Big \| \widehat { K } _ { m } - K [ \mathcal { X } ] \Big \| ^ { 2 } = \frac { 1 } { m } \mathbb { E } \Big [ \big \| \tau _ { a } ^ { 2 } \phi ( Z ) \phi ( Z ) ^ { \top } - K [ \mathcal { X } ] \big \| ^ { 2 } \Big ] .
$$

Combining Steps 2–5 gives ${ \mathcal W } _ { 2 } ( f [ \mathcal { X } ] , G [ \mathcal { X } ] ) ^ { 2 } \le C ( \phi , \mathcal { X } ) ^ { 2 } / m$ with $C ( \phi , \mathcal { X } )$ as in (5). No step above uses global Lipschitz continuity of $\phi .$ The only property of the activation that enters is the finiteness of the fourth moment in Step 5, which follows from Assumption 1 (1). □

Corollary 2 (at most linear growth). Let $\mathcal { X } \subset I$ be afinite sample set with $| { \mathcal { X } } | = n$ and let ϕ satisfy Assumption 1 with growth constants $A , p .$ . Then the numerator of (5),

$$
\begin{array} { r } { N : = \sqrt { \mathbb { E } \Big [ \big \| \tau _ { a } ^ { 2 } \phi ( Z ) \phi ( Z ) ^ { \top } - K [ \mathcal { X } ] \big \| ^ { 2 } \Big ] } , \qquad Z \sim \mathcal { N } \big ( 0 , \kappa [ \mathcal { X } ] \big ) \in \mathbb { R } ^ { n } , } \end{array}
$$

satisfies $N \le \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } } r$ , where

$$
M _ { s } : = \mathbb { E } \big [ ( 1 + | \xi | ) ^ { s } \big ] < \infty \quad f o r s > 0 a n d \xi \sim \mathcal { N } ( 0 , \kappa _ { I } ^ { 2 } ) w i t h \kappa _ { I } ^ { 2 } : = \operatorname* { m a x } _ { x \in I } \kappa ( x , x ) < \infty .
$$

The constant depends only on ϕ through $A , p ,$ on I, and on the variances, and not on n.

Corollary 2. Abbreviate $Z = ( Z _ { 1 } , \dots , Z _ { n } ) \sim { \mathcal { N } } ( 0 , \kappa [ { \mathcal { X } } ] )$ and consider the random matrix

$$
Y : = \tau _ { a } ^ { 2 } \phi ( Z ) \phi ( Z ) ^ { \top } \in \mathbb { R } ^ { n \times n } , \qquad Y _ { i j } = \tau _ { a } ^ { 2 } \phi ( Z _ { i } ) \phi ( Z _ { j } ) .
$$

For every pair $( i , j )$ the vector $( Z _ { i } , Z _ { j } )$ is centred Gaussian with covariance $\left( \begin{array} { l } { \kappa ( x _ { i } , x _ { i } ) ~ \kappa ( x _ { i } , x _ { j } ) } \\ { \kappa ( x _ { i } , x _ { j } ) ~ \kappa ( x _ { j } , x _ { j } ) } \end{array} \right)$ , so by (2) we have $\mathbb { E } [ Y _ { i j } ] = K ( x _ { i } , x _ { j } )$ , that is $\mathbb { E } [ Y ] = K [ \mathcal { X } ]$

Step 1 (centring). Entrywise, E $\left[ ( Y _ { i j } - \mathbb { E } [ Y _ { i j } ] ) ^ { 2 } \right] = \mathbb { E } [ Y _ { i j } ^ { 2 } ] - \left( \mathbb { E } [ Y _ { i j } ] \right) ^ { 2 } \leq \mathbb { E } [ Y _ { i j } ^ { 2 } ]$ . Summing over $i , j$ gives

$$
N ^ { 2 } = \mathbb { E } \Big [ \big \| Y - K [ \mathcal { X } ] \big \| ^ { 2 } \Big ] \ \leq \ \mathbb { E } \big [ \| Y \| ^ { 2 } \big ] .
$$

Step 2 (reduction to fourth moments). By the definition of the Frobenius norm,

$$
\| Y \| ^ { 2 } = { \bigl ( } \tau _ { a } ^ { 2 } { \bigr ) } ^ { 2 } \sum _ { i , j = 1 } ^ { n } \phi ( Z _ { i } ) ^ { 2 } \phi ( Z _ { j } ) ^ { 2 } , \qquad { \mathrm { h e n c e } } \qquad \mathbb { E } \bigl [ \| Y \| ^ { 2 } \bigr ] = \left( \tau _ { a } ^ { 2 } \right) ^ { 2 } \sum _ { i , j = 1 } ^ { n } \mathbb { E } \bigl [ \phi ( Z _ { i } ) ^ { 2 } \phi ( Z _ { j } ) ^ { 2 } \bigr ] .
$$

Step 3 (a uniform fourth-moment bound). By the Cauchy–Schwarz inequality, $\begin{array} { r l } { \mathbb { E } [ \phi ( Z _ { i } ) ^ { 2 } \phi ( Z _ { j } ) ^ { 2 } ] } & { { } \leq } \end{array}$ $\mathbb { E } [ \phi ( Z _ { i } ) ^ { 4 } ] ^ { 1 / 2 } \mathbb { E } [ \phi ( Z _ { j } ) ^ { 4 } ] ^ { 1 / 2 }$ , so it suffices to bound $\mathbb { E } [ \phi ( Z _ { i } ) ^ { 4 } ]$ uniformly in i. Polynomial growth (Assumption $1 ( 1 ) )$ gives $\dot { \phi ( z ) ^ { 4 } } \le \dot { A } ^ { 4 } ( \dot { 1 } + \overline { { { | z | } } } ) ^ { 4 p }$ . Each $Z _ { i }$ is centred Gaussian with variance $\kappa _ { i } ^ { 2 } = \kappa ( x _ { i } , x _ { i } ) \leq \kappa _ { I } ^ { 2 }$ , and for $\xi \sim \mathcal { N } ( 0 , 1 )$ the map $\sigma \mapsto \mathbb { E } \left[ ( 1 + \sigma | \xi | ) ^ { 4 p } \right]$ is non-decreasing on $[ 0 , \infty )$ , whence

$$
\begin{array} { r } { \mathbb { E } \big [ \phi ( Z _ { i } ) ^ { 4 } \big ] \le A ^ { 4 } \mathbb { E } \big [ ( 1 + | Z _ { i } | ) ^ { 4 p } \big ] \le A ^ { 4 } \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , \kappa _ { I } ^ { 2 } ) } \big [ ( 1 + | Z | ) ^ { 4 p } \big ] = A ^ { 4 } M _ { 4 p } . } \end{array}
$$

Combining the three steps,

$$
N ^ { 2 } \ \leq \ \left( \tau _ { a } ^ { 2 } \right) ^ { 2 } \sum _ { i , j = 1 } ^ { n } A ^ { 4 } M _ { 4 p } \ = \ \left( \tau _ { a } ^ { 2 } \right) ^ { 2 } A ^ { 4 } M _ { 4 p } n ^ { 2 } ,
$$

and taking square roots yields $N \leq \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } } n$ . The moment $M _ { 4 p }$ is finite because Gaussian random variables have moments of all orders, and $\kappa _ { I } ^ { 2 } < \infty$ because I is compact and κ is continuous; the constant depends only on $A , p ,$ on I through $\kappa _ { I } ^ { 2 } .$ , and on the variances $\sigma _ { w } ^ { 2 } , \sigma _ { b } ^ { 2 } , \tau _ { a } ^ { 2 }$ , as stated. □

Proposition 1. Part (1) is Corollary 2. The matrix inside the norm in (5) has $n ^ { 2 }$ entries, each bounded in expectation uniformly over the compact set $I ,$ so its Frobenius norm grows like ${ \sqrt { n ^ { 2 } } } = n$ . We bound the centred fluctuation by the second moment of $\tau _ { a } ^ { 2 } \phi \dot { ( } Z ) \phi ( Z ) ^ { \dagger }$ , reduce the Frobenius norm to its entries, and control those by Assumption 1 (1).

For part (2), let $\mathcal { X } \subset I$ satisfy (6) with constant ν and let $n = | \mathcal { X } |$ . Then Corollary 2 and (6) give

$$
C ( \phi , \mathscr { X } ) ^ { 2 } \ \leq \ \frac { \left( \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } } n \right) ^ { 2 } } { \nu n } \ = \ D ^ { 2 } n , \qquad D : = \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { \frac { M _ { 4 p } } { \nu } } ,\tag{13}
$$

so that $C ( \phi , { \mathcal { X } } ) \leq D { \sqrt { n } }$ , and Theorem 1 gives

$$
{ \mathcal W } _ { 2 } \big ( f [ { \mathcal X } ] , G [ { \mathcal X } ] \big ) \ \le \ \frac { C ( \phi , { \mathcal X } ) } { \sqrt m } \ \le \ D \sqrt { \frac n m } .
$$

The order $\sqrt { n }$ cannot be improved within (5). For any finite sample set $\mathcal { X } \subset I$ with $n = | \mathcal { X } |$ , write $Z \sim { \mathcal { N } } ( 0 , \kappa [ { \mathcal { X } } ] ) \in$ $\mathbb { R } ^ { n }$ as in Corollary 2. The trace of a positive semi-definite matrix dominates each of its eigenvalues, so by Assumption 1 (1) and $\kappa ( x _ { i } , x _ { i } ) \leq \kappa _ { I } ^ { 2 }$ , arguing as in the proof of Corollary 2,

$$
\lambda ^ { + } \bigl ( K [ \mathcal { X } ] \bigr ) \ \le \ \mathrm { t r } K [ \mathcal { X } ] \ = \ \tau _ { a } ^ { 2 } \sum _ { i = 1 } ^ { n } \mathbb { E } \bigl [ \phi ( Z _ { i } ) ^ { 2 } \bigr ] \ \le \ \bar { \nu } n , \qquad \bar { \nu } : = \tau _ { a } ^ { 2 } A ^ { 2 } M _ { 2 p } ,\tag{14}
$$

so linear growth is the largest order the denominator can have.

Combining (14) with (6) and $\mathrm { t r } ( K [ \mathcal { X } ] ) \ge \mathrm { r a n k } ( K [ \mathcal { X } ] ) \lambda ^ { + } ( K [ \mathcal { X } ] )$ gives rank $( K [ \mathcal { X } ] ) \leq \bar { \nu } / \nu$ so the condition forces the $K [ \mathcal { X } ]$ to have bounded rank. This excludes every activation for which this rank is unbounded, in particular every activation whose kernel K is strictly positive definite on I, since then rank $( K [ \mathcal { X } ] ) = | \mathcal { X } |$ . The condition is therefore a finite-rank condition, and Lemma 1 verifies it for the midpoint grids with polynomial activations. □

## C Proofs of Section 3

Lemma 1. Write $\begin{array} { r } { \phi ( z ) = \sum _ { h = 0 } ^ { r } \beta _ { h } z ^ { h } } \end{array}$ with $\beta _ { r } \neq 0$ , and abbreviate $R : = { \binom { r + d } { d } }$

Step 1 (finite-rank representation of the kernel). Let $\xi _ { 0 } \sim \mathcal { N } ( 0 , 1 )$ and $\xi _ { 1 } \sim \mathcal { N } ( 0 , \mathrm { I } _ { d \times d } )$ be independent, and set

$$
Z _ { x } : = \frac { \sigma _ { w } } { \sqrt { d } } \xi _ { 1 } ^ { \top } { x } + \sigma _ { b } \xi _ { 0 } , \qquad { x } \in \mathbb { R } ^ { d } .\tag{15}
$$

This realisation has the covariance required by (2): by independence of $\xi _ { 0 }$ and $\xi _ { 1 }$

$$
\mathrm { C o v } ( Z _ { x } , Z _ { y } ) \ = \ \frac { \sigma _ { w } ^ { 2 } } { d } \ \mathrm { C o v } \left( \xi _ { 1 } ^ { \top } x , \xi _ { 1 } ^ { \top } y \right) + \sigma _ { b } ^ { 2 } \ \mathrm { V a r } ( \xi _ { 0 } ) \ = \ \frac { \sigma _ { w } ^ { 2 } } { d } \ \langle x , y \rangle + \sigma _ { b } ^ { 2 } \ = \ \kappa ( x , y ) .
$$

By the binomial theorem,

$$
\phi ( Z _ { x } ) \ = \ \sum _ { h = 0 } ^ { r } \beta _ { h } Z _ { x } ^ { h } \ = \ \sum _ { i = 0 } ^ { r } ( \xi _ { 1 } ^ { \top } x ) ^ { i } P _ { i } , \qquad P _ { i } : = \ \Big ( \frac { \sigma _ { w } ^ { 2 } } { d } \Big ) ^ { i / 2 } \sum _ { h = i } ^ { r } { \binom { h } { i } } \beta _ { h } ( \sigma _ { h } \xi _ { 0 } ) ^ { h - i } \ \in \ \mathbb { R } .\tag{16}
$$

For $0 \leq i \leq r$ define $\varphi _ { i } : \mathbb { R } ^ { d }  \mathbb { R } ^ { \binom { i + d - 1 } { d - 1 } }$ by

$$
\varphi _ { i } ( x ) = \bigoplus _ { | \alpha | = i } { \sqrt { \binom { i } { \alpha _ { 1 } , \dots , \alpha _ { d } } } } x ^ { \alpha } ,\tag{17}
$$

the dimension $\binom { i + d - 1 } { d - 1 }$ being the number of multi-indices of total degree i. Fixing the order of these multi-indices gives a bijection $e _ { i } : \left\{ 1 , \dots , \binom { i + d - 1 } { d - 1 } \right\} \to \left\{ \alpha \in \mathbb { N } _ { 0 } ^ { d } : | \alpha | = i \right\}$ between the coordinates of $\varphi _ { i }$ and the multi-indices of degree i. In this notation the multinomial theorem reads $\left( \xi _ { 1 } ^ { \top } x \right) ^ { i } = \left. \varphi _ { i } ( \xi _ { 1 } ) , \varphi _ { i } ( x ) \right.$ , so (16) becomes

$$
\phi ( Z _ { x } ) = \sum _ { i = 0 } ^ { r } \bigl \langle P _ { i } \varphi _ { i } ( \xi _ { 1 } ) , \varphi _ { i } ( x ) \bigr \rangle = \Bigl \langle \bigoplus _ { i = 0 } ^ { r } P _ { i } \varphi _ { i } ( \xi _ { 1 } ) , \bigoplus _ { i = 0 } ^ { r } \varphi _ { i } ( x ) \Bigr \rangle .\tag{18}
$$

The vectors in the last inner product have dimension $\textstyle \sum _ { i = 0 } ^ { r } { \binom { i + d - 1 } { d - 1 } } = R$ . Setting

$$
\Psi ( x ) : = \bigoplus _ { i = 0 } ^ { r } \varphi _ { i } ( x ) \in \mathbb { R } ^ { R } , \qquad \Phi : = \bigoplus _ { i = 0 } ^ { r } P _ { i } \varphi _ { i } ( \xi _ { 1 } ) , \qquad M : = \tau _ { a } ^ { 2 } \mathbb { E } \big [ \Phi \Phi ^ { \top } \big ] ,
$$

we obtain from (2) and (18)

$$
K ( x , y ) \ = \ \tau _ { a } ^ { 2 } \mathbb { E } \big [ \phi ( Z _ { x } ) \phi ( Z _ { y } ) \big ] \ = \ \Psi ( x ) ^ { \top } M \Psi ( y ) .\tag{19}
$$

The matrix M is positive semi-definite, and $M \ne 0 ;$ since deg $\phi \ = \ r ,$ , the sum defining $P _ { r }$ in (16) has only the term $\textit { h } = \textit { r } ,$ so $P _ { r } ~ = ~ \beta _ { r } ( \sigma _ { w } ^ { 2 } / d ) ^ { r / 2 } ~ \ne ~ 0$ is deterministic, and the degree-r diagonal block of M equals $\tau _ { a } ^ { 2 } \beta _ { r } ^ { 2 } ( \sigma _ { w } ^ { 2 } / d ) ^ { r } \mathbb { E } \big [ \varphi _ { r } ( \xi _ { 1 } ) \varphi _ { r } ( \xi _ { 1 } ) ^ { \top } \big ] \ne 0$ . In particular $\lambda ^ { + } ( M ) > 0$

Let $V _ { k } \in \mathbb { R } ^ { k ^ { d } \times R }$ be the matrix with rows $\Psi ( x _ { 1 } ) ^ { \top } , \ldots , \Psi ( x _ { k ^ { d } } ) ^ { \top }$ , where $\mathcal { X } _ { k } = \{ x _ { 1 } , \dots , x _ { k ^ { d } } \}$ . Then (19) gives

$$
K [ \mathcal { X } _ { k } ] = V _ { k } M V _ { k } ^ { \top } ,\tag{20}
$$

whence rank $K [ \mathcal { X } _ { k } ] \leq \mathrm { r a n k } ( M ) \leq R$ , which is the first assertion.

Step 2 (passing to a matrix offixed dimension). Put $B : = V _ { k } M ^ { 1 / 2 }$ , so that $K [ \mathcal { X } _ { k } ] = B B ^ { \top }$ , and set

$$
T _ { k } : = B ^ { \top } B = M ^ { 1 / 2 } V _ { k } ^ { \top } V _ { k } M ^ { 1 / 2 } \in \mathbb { R } ^ { R \times R } .\tag{21}
$$

The matrices $B B ^ { \top }$ and $B ^ { \top }$ B have the same nonzero eigenvalues, so $\lambda ^ { + } \bigl ( K [ \mathcal { X } _ { k } ] \bigr ) = \lambda ^ { + } ( T _ { k } )$ whenever $T _ { k } \neq 0$ . Unlike $K [ \mathcal { X } _ { k } ]$ , the matrix $T _ { k }$ has dimension R, independent of k. We therefore analyse the normalised moment matrix

$$
\Theta _ { k } : = \frac { 1 } { k ^ { d } } V _ { k } ^ { \top } V _ { k } \in \mathbb { R } ^ { R \times R } , \qquad \mathrm { s o ~ t h a t } \qquad T _ { k } = k ^ { d } M ^ { 1 / 2 } \Theta _ { k } M ^ { 1 / 2 } .\tag{22}
$$

Step 3 (the Riemann-sum limit). Index the rows and columns of $\Theta _ { k }$ by pairs $( i , s )$ with $0 \leq i \leq r$ and $1 \leq s \leq ( { } _ { d - 1 } ^ { i + d - 1 } )$ where (i, s) refers to the s-th coordinate of the degree-i block of Ψ. By (17),

$$
\left( \Theta _ { k } \right) _ { ( i , s ) , ( j , t ) } \ = \ \frac { 1 } { k ^ { d } } \sum _ { l = 1 } ^ { k ^ { d } } \bigl ( \varphi _ { i } ( x _ { l } ) \bigr ) _ { s } \bigl ( \varphi _ { j } ( x _ { l } ) \bigr ) _ { t } \ = \ c _ { ( i , s ) , ( j , t ) } \cdot \frac { 1 } { k ^ { d } } \sum _ { l = 1 } ^ { k ^ { d } } x _ { l } ^ { e _ { i } ( s ) + e _ { j } ( t ) } ,\tag{23}
$$

where $c _ { ( i , s ) , ( j , t ) } : = \big [ \binom { i } { e _ { i } ( s ) } \binom { j } { e _ { j } ( t ) } \big ] ^ { 1 / 2 }$ does not depend on k. Since $\mathcal { X } _ { k }$ is the product grid (8), the points $x _ { l }$ are the midpoints of the $k ^ { d }$ cells of a uniform partition of $I ,$ each of volume $\mathrm { V o l } ( I ) / k ^ { d }$ . The sum in (23) is therefore the d-dimensional midpoint rule for the continuous function $x \mapsto x ^ { \alpha }$ with $\alpha : = e _ { i } ( s ) + e _ { j } ( t )$ , and

$$
{ \frac { 1 } { k ^ { d } } } \sum _ { l = 1 } ^ { k ^ { d } } x _ { l } ^ { \alpha } \ = \ { \frac { 1 } { \mathrm { V o l } ( I ) } } \sum _ { l = 1 } ^ { k ^ { d } } { \frac { \mathrm { V o l } ( I ) } { k ^ { d } } } x _ { l } ^ { \alpha } \ { \xrightarrow { } } \ { \frac { 1 } { \ { k {  } \infty } \ } } \ { \frac { 1 } { \mathrm { V o l } ( I ) } } \int _ { I } x ^ { \alpha } \ d x .\tag{24}
$$

Writing H for the matrix with entries $\begin{array} { r } { H _ { ( i , s ) , ( j , t ) } : = c _ { ( i , s ) , ( j , t ) } \operatorname { V o l } ( I ) ^ { - 1 } \int _ { I } x ^ { e _ { i } ( s ) + e _ { j } ( t ) } } \end{array}$ dx, the convergence (24) holds entrywise, and since the dimension R does not depend on k this gives $\lVert \dot { \Theta _ { k } } - H \rVert \to 0$

The limit H is positive definite. Indeed, for $u \in \mathbb { R } ^ { R }$

$$
u ^ { \top } H u = \frac { 1 } { \mathrm { V o l } ( I ) } \int _ { I } q _ { u } ( x ) ^ { 2 } d x , \qquad q _ { u } ( x ) : = \sum _ { i = 0 } ^ { r } \sum _ { s = 1 } ^ { \binom { i + d - 1 } { d - 1 } } u _ { ( i , s ) } \sqrt { \binom { i } { e _ { i } ( s ) } } x ^ { e _ { i } ( s ) } .\tag{25}
$$

The map $( i , s ) \mapsto e _ { i } ( s )$ is a bijection onto $\{ \alpha : | \alpha | \leq r \}$ , and the monomials $x ^ { \alpha }$ are linearly independent, so $u \ne 0$ implies $q _ { u } \not \equiv 0 . \ : \mathrm { A }$ nonzero polynomial vanishes on a Lebesgue null set, so the integral in (25) is strictly positive, and $\bar { \lambda _ { \operatorname* { m i n } } } ( H ) > 0$ . For every unit vector v,

$$
\begin{array} { r } { v ^ { \top } \Theta _ { k } v \ge v ^ { \top } H v - | v ^ { \top } ( \Theta _ { k } - H ) v | \ge \lambda _ { \operatorname* { m i n } } ( H ) - \| \Theta _ { k } - H \| , } \end{array}
$$

so there is $k _ { 1 } \in \mathbb { N }$ with

$$
\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( \Theta _ { k } ) \ \geq \ \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } ( H ) \ > \ 0 \qquad \mathrm { f o r ~ a l l } \ k \geq k _ { 1 } . } \end{array}\tag{26}
$$

Step 4 (conclusion). Fix $k \geq k _ { 1 }$ . By (26) the matrix $\Theta _ { k }$ is positive definite, so $T _ { k } \neq 0 :$ if $M ^ { 1 / 2 } \Theta _ { k } M ^ { 1 / 2 } = 0$ then $\| \Theta _ { k } ^ { 1 / 2 } M ^ { 1 / 2 } u \| ^ { 2 } = u ^ { \top } M ^ { 1 / 2 } \Theta _ { k } M ^ { 1 / 2 } u = 0$ for every $u ,$ forcing $M ^ { 1 / 2 } = 0$ and hence $M = 0$ , which Step 1

excludes. From (21) we have range $( T _ { k } ) \subseteq \mathrm { r a n g e } ( M ^ { 1 / 2 } ) = \mathrm { r a n g e } ( M )$ , and every unit vector $v \in { \mathrm { r a n g e } } ( M )$ satisfies $v ^ { \top } M v \geq \lambda ^ { + } ( M )$ . Hence for every unit vector $v \in { \mathrm { r a n g e } } ( T _ { k } )$

$$
\begin{array} { r l } & { \frac { 1 } { k ^ { d } } v ^ { \top } T _ { k } v = \left( M ^ { 1 / 2 } v \right) ^ { \top } \Theta _ { k } \left( M ^ { 1 / 2 } v \right) } \\ & { \qquad \quad \geq \lambda _ { \operatorname* { m i n } } ( \Theta _ { k } ) \left. M ^ { 1 / 2 } v \right. ^ { 2 } } \\ & { \qquad = \lambda _ { \operatorname* { m i n } } ( \Theta _ { k } ) v ^ { \top } M v } \\ & { \qquad \quad \geq \lambda _ { \operatorname* { m i n } } ( \Theta _ { k } ) \lambda ^ { + } ( M ) . } \end{array}\tag{27}
$$

Minimising over such v gives $\lambda ^ { + } \bigl ( K [ \mathcal { X } _ { k } ] \bigr ) = \lambda ^ { + } ( T _ { k } ) \ge k ^ { d } \lambda _ { \operatorname* { m i n } } ( \Theta _ { k } ) \lambda ^ { + } ( M )$ , and with (26),

$$
\begin{array} { r } { \lambda ^ { + } \big ( K [ { \mathcal X } _ { k } ] \big ) \ge \nu k ^ { d } \qquad \mathrm { f o r } \mathrm { e v e r y } k \ge k _ { 1 } , \qquad \nu : = \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } ( H ) \lambda ^ { + } ( M ) > 0 . } \end{array}
$$

For the finitely many $k < k _ { 1 }$ , Assumption 1(3) gives, for every $x \in \mathcal { X } _ { k }$ , that $\kappa ( x , x ) > 0$ and that ϕ does not vanish $\mathcal { N } ( 0 , \kappa ( x , x ) )$ )-almost everywhere, so $\bar { K } ( x , x ) = \bar { \tau } _ { a } ^ { 2 } \mathbb { E } [ \phi ( Z _ { x } ) ^ { 2 } ] > 0$ . Hence $K [ \mathcal { X } _ { k } ] \ne \mathrm { ~ 0 ~ }$ and $\lambda ^ { + } ( K [ \mathcal { X } _ { k } ] ) > 0 ;$ ; replacing ν by min{ν, $, \ \operatorname* { m i n } _ { k < k _ { 1 } } k ^ { - d } \lambda ^ { + } ( K [ \mathcal { X } _ { k } ] ) \} > 0$ gives the bound for every $k \in \mathbb N$ □

Lemma 2. Step 1 (the supremum bound). The affine change of variables $\begin{array} { r } { x _ { j } ~ = ~ \frac { a _ { j } + b _ { j } } { 2 } + \frac { | I _ { j } | } { 2 } t _ { j } } \end{array}$ maps $[ - 1 , 1 ] ^ { d }$ onto I and preserves total degree. Writing $\tilde { p } ( t ) : = p ( x ( t ) )$ ), we have $\| p \| _ { \mathcal { C } ^ { 0 } ( I ) } = \| \tilde { p } \| _ { \mathcal { C } ^ { 0 } ( [ - 1 , 1 ] ^ { d } ) }$ and $\| p \| _ { L ^ { 2 } ( I ) } ^ { 2 } =$ $2 ^ { - d } \mathrm { V o l } ( I ) \| \tilde { p } \| _ { L ^ { 2 } ( [ - 1 , 1 ] ^ { d } ) } ^ { 2 }$ , so it suffices to prove $\| \tilde { p } \| _ { \mathcal { C } ^ { 0 } } \leq 2 ^ { - d / 2 } ( r + 1 ) ^ { d } \| \tilde { p } \| _ { L ^ { 2 } }$ on the cube.

Let $P _ { 0 } , P _ { 1 } , . . . .$ . be the Legendre polynomials on $[ - 1 , 1 ] .$ , so that $\| P _ { n } \| _ { \infty } = 1$ and $\begin{array} { r } { \int _ { - 1 } ^ { 1 } P _ { n } ^ { 2 } = \frac { 2 } { 2 n + 1 } } \end{array}$ . Since $\tilde { p }$ has degree at most r in each coordinate, it expands as $\begin{array} { r } { \tilde { p } ( t ) = \sum _ { \alpha } c _ { \alpha } \prod _ { j = 1 } ^ { d } P _ { \alpha _ { j } } ( t _ { j } ) } \end{array}$ over multi-indices α with $\alpha _ { j } \leq r _ { \ast }$ , and these products are orthogonal on $[ - 1 , 1 ] ^ { d }$ . Hence

$$
\| \tilde { p } \| _ { C ^ { 0 } } \leq \sum _ { \alpha } \vert c _ { \alpha } \vert \leq \Big ( \sum _ { \alpha } c _ { \alpha } ^ { 2 } \prod _ { j } \frac { 2 } { 2 \alpha _ { j } + 1 } \Big ) ^ { 1 / 2 } \Big ( \sum _ { \alpha } \prod _ { j } \frac { 2 \alpha _ { j } + 1 } { 2 } \Big ) ^ { 1 / 2 } = \| \tilde { p } \| _ { L ^ { 2 } } \Big ( \prod _ { j = 1 } ^ { d } \sum _ { n = 0 } ^ { r } \frac { 2 n + 1 } { 2 } \Big ) ^ { 1 / 2 } ,
$$

by Cauchy–Schwarz and Parseval. As $\begin{array} { r } { \sum _ { n = 0 } ^ { r } \frac { 2 n + 1 } { 2 } = \frac { ( r + 1 ) ^ { 2 } } { 2 } } \end{array}$ , the last factor equals $2 ^ { - d / 2 } ( r + 1 ) ^ { d }$ , which proves the first inequality in (10).

Step 2 (sample paths). Each coordinate of $W x + b$ is affine in $x ,$ so $\phi$ applied to it is a polynomial of total degree at most r; hence $f ( x ) = a ^ { \top } \phi ( W x + b ) + c$ lies in $\mathcal { P } _ { r }$ for every realisation. For G, Step 1 of the proof of Lemma 1 gives $K ( x , y ) = \Psi ( x ) ^ { \top } M \Psi ( y )$ with Ψ a vector of polynomials of degree at most r and $M \succeq 0$ . Let $\zeta \sim \mathcal { N } ( 0 , M )$ and $c ^ { \prime } \sim \mathcal { N } ( 0 , \tau _ { c } ^ { 2 } )$ be independent, and set $\tilde { G } ( \boldsymbol { x } ) : = \Psi ( \boldsymbol { x } ) ^ { \top } \zeta + c ^ { \prime } ,$ . Then $\tilde { G }$ is a centred Gaussian process with Cov $( \tilde { G } ( x ) , \tilde { G } ( y ) ) = \Psi ( x ) ^ { \top } M \Psi ( y ) + \tau _ { c } ^ { 2 } = K _ { c } ( x , y )$ , so G<sup>˜</sup> has the law of $G ,$ and its sample paths lie in $\mathcal { P } _ { r }$ □

Lemma 3. Partition I into the $k ^ { d }$ cells $Q _ { 1 } , \ldots , Q _ { k ^ { d } }$ of the grid, each of volume $\mathrm { V o l } ( I ) / k ^ { d }$ and each containing its grid point $x _ { i }$ as midpoint. For $x \in Q _ { i }$ we have $\| x - x _ { i } \| _ { 2 } \leq \rho _ { k }$ by (12), so

$$
| h ( x ) | \leq | h ( x _ { i } ) | + | h ( x ) - h ( x _ { i } ) | \leq | h ( x _ { i } ) | + [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } \| x - x _ { i } \| _ { 2 } ^ { \gamma } \leq | h ( x _ { i } ) | + [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } \rho _ { k } ^ { \gamma } ,
$$

hence $h ( x ) ^ { 2 } \leq 2 h ( x _ { i } ) ^ { 2 } + 2 \rho _ { k } ^ { 2 \gamma } [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } ^ { 2 }$ . Integrating over $Q _ { i }$ and summing over $i ,$

$$
\begin{array} { l } { \displaystyle \| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } = \sum _ { i = 1 } ^ { k ^ { d } } \int _ { Q _ { i } } h ( x ) ^ { 2 } d x } \\ { \displaystyle \leq \frac { \mathrm { V o l } ( I ) } { k ^ { d } } \sum _ { i = 1 } ^ { k ^ { d } } \Big ( 2 h ( x _ { i } ) ^ { 2 } + 2 \rho _ { k } ^ { 2 \gamma } [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } ^ { 2 } \Big ) } \\ { \displaystyle = 2 \frac { \mathrm { V o l } ( I ) } { k ^ { d } } \| h [ \mathcal { X } _ { k } ] \| ^ { 2 } + 2 \mathrm { V o l } ( I ) \rho _ { k } ^ { 2 \gamma } [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) } ^ { 2 } . } \end{array}
$$

Lemma 4. We regard $f$ and G as random elements of the Polish space ${ \mathcal { C } } ^ { 0 } ( I )$ . The grid evaluations $f [ \mathcal { X } ]$ and $G [ \mathcal { X } ]$ are random vectors in $\mathbb { R } ^ { n }$ . Let $\pi ^ { * }$ be an optimal coupling of their laws for the quadratic Wasserstein distance, which exists by [Villani, 2008, Theorem 4.1]. Since $\mathcal { W } _ { 2 , n }$ is a positive multiple of $\mathcal { W } _ { 2 }$ , the coupling $\pi ^ { * }$ is optimal for the rescaled distance as well. Because ${ \mathcal { C } } ^ { 0 } ( I )$ is Polish, the regular conditional distributions

$$
\mu _ { u } ^ { f } : = \mathbb { P } _ { f | f [ \mathcal { X } ] = u } , \qquad \mu _ { v } ^ { G } : = \mathbb { P } _ { G | G [ \mathcal { X } ] = v } , \qquad u , v \in \mathbb { R } ^ { n } ,
$$

exist, and by disintegration $\begin{array} { r } { 1 = \mathbb { P } \big ( f [ \mathcal { X } ] = f [ \mathcal { X } ] \big ) = \int _ { \mathbb { R } ^ { n } } \mu _ { u } ^ { f } \big ( \{ \varphi : \varphi [ \mathcal { X } ] = u \} \big ) d \mathbb { P } _ { f [ \mathcal { X } ] } ( u ) } \end{array}$ , so $\mu _ { u } ^ { f }$ is concentrated on the fibre $\{ \varphi : \varphi [ { \mathcal { X } } ] = u \}$ for $\mathbb { P } _ { f [ \mathcal { X } ] }$ -almost every u; likewise for $\mu _ { v } ^ { G }$

Define the measure

$$
\pi ( A ) : = \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } \left( \mu _ { u } ^ { f } \otimes \mu _ { v } ^ { G } \right) ( A ) d \pi ^ { * } ( u , v ) , \qquad A \subseteq \mathcal { C } ^ { 0 } ( I ) \times \mathcal { C } ^ { 0 } ( I ) \mathrm { ~ B o r e l } ,
$$

which is a probability measure since $( u , v ) \mapsto \mu _ { u } ^ { f } \otimes \mu _ { v } ^ { G }$ is a probability kernel; on rectangles, $\pi ( A _ { 1 } \times A _ { 2 } ) =$ $\begin{array} { r l } { \int \mu _ { u } ^ { f } ( A _ { 1 } ) \mu _ { v } ^ { G } ( A _ { 2 } ) d \pi ^ { * } ( u , v ) } \end{array}$ . The two coordinate projections on $( \mathcal { C } ^ { 0 } ( I ) \times \mathcal { C } ^ { 0 } ( I ) , \pi )$ define random functions $\tilde { f }$ and $\tilde { G } .$ We check the marginals:

$$
\begin{array} { r l } & { \mathbb { P } \big ( \tilde { f } \in A _ { 1 } \big ) = \displaystyle \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } \mu _ { u } ^ { f } ( A _ { 1 } ) \underbrace { \mu _ { v } ^ { G } \big ( { \mathcal C } ^ { 0 } ( I ) \big ) } _ { = 1 } d \pi ^ { * } ( u , v ) } \\ & { \qquad = \displaystyle \int _ { \mathbb { R } ^ { n } } \mu _ { u } ^ { f } ( A _ { 1 } ) d \mathbb { P } _ { f [ { \mathcal X } ] } ( u ) } \\ & { \qquad = \displaystyle \int _ { \mathbb { R } ^ { n } } \mathbb { P } \big ( f \in A _ { 1 } \mid f [ { \mathcal X } ] = u \big ) d \mathbb { P } _ { f [ { \mathcal X } ] } ( u ) } \\ & { \qquad = \mathbb { P } ( f \in A _ { 1 } ) , } \end{array}
$$

where the second equality uses that the first marginal of $\pi ^ { * }$ is $\mathbb { P } _ { f [ \mathcal { X } ] }$ , and for G the argument works the same. Moreover, for every Borel set $\mathbf { \bar { A } } \subseteq \mathbf { \bar { R } } ^ { n } \times \mathbb { R } ^ { n }$ ,

$$
\begin{array} { r l } { \mathbb { P } \big ( \big ( \widetilde f [ \mathcal { X } ] , \widetilde G [ \mathcal { X } ] \big ) \in A \big ) = \displaystyle \int _ { \mathbb R ^ { n } \times \mathbb R ^ { n } } \big ( \mu _ { u } ^ { f } \otimes \mu _ { v } ^ { G } \big ) \big ( \{ ( \varphi , \psi ) : ( \varphi [ \mathcal { X } ] , \psi [ \mathcal { X } ] ) \in A \} \big ) d \pi ^ { * } ( u , v ) } & { } \\ { \quad \quad } & { = \displaystyle \int _ { \mathbb R ^ { n } \times \mathbb R ^ { n } } \big ( \delta _ { u } \otimes \delta _ { v } \big ) ( A ) d \pi ^ { * } ( u , v ) } \\ { \quad } & { = \displaystyle \int _ { \mathbb R ^ { n } \times \mathbb R ^ { n } } \mathbf { 1 } _ { A } ( u , v ) d \pi ^ { * } ( u , v ) } \\ { \quad } & { = \pi ^ { * } ( A ) , } \end{array}
$$

where the second equality uses the concentration on the fibres, whose exceptional sets are $\pi ^ { * }$ -null because they are null under the marginals of $\pi ^ { * }$ . Therefore $( \tilde { f } , \tilde { G } )$ is a coupling of $f$ and G whose evaluations $\left( \tilde { f } [ \mathcal { X } ] , \tilde { G } [ \mathcal { X } ] \right)$ have the optimal law $\pi ^ { * }$ , and consequently

$$
\mathbb { E } \Big [ \big \| \tilde { f } [ \mathcal { X } ] - \tilde { G } [ \mathcal { X } ] \big \| _ { n } ^ { 2 } \Big ] = \int _ { \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } } \| u - v \| _ { n } ^ { 2 } d \pi ^ { * } ( u , v ) = \mathcal { W } _ { 2 , n } \big ( f [ \mathcal { X } ] , G [ \mathcal { X } ] \big ) ^ { 2 } .
$$

Theorem 2. Fix $\begin{array} { r } { \gamma = \frac { 1 } { 2 } } \end{array}$ and $k \geq k _ { 1 }$ , with $k _ { 1 }$ as in Lemma 1, and let $( \tilde { f } , \tilde { G } )$ be the coupling of Lemma 4 for $\mathcal { X } _ { k } ;$ ; write $h : = { \tilde { f } } - { \tilde { G } }$ , so that $n = | { \boldsymbol { \chi } } _ { k } | = k ^ { d }$

Since ϕ is a polynomial of degree $r ,$ Lemma 2 gives that $f$ and G have sample paths in $\mathcal { P } _ { r }$ almost surely, hence so does $h ,$ and (10) may be applied pathwise. Combining it with (11) of Lemma 3 and taking expectations,

$$
\mathbb { E } \big [ \| h \| _ { C ^ { 0 } ( I ) } ^ { 2 } \big ] \leq \frac { ( r + 1 ) ^ { 2 d } } { \operatorname { V o l } ( I ) } \mathbb { E } \big [ \| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } \big ] \leq 2 ( r + 1 ) ^ { 2 d } \Big ( \mathbb { E } \big [ \| h [ \mathcal { X } _ { k } ] \| _ { k ^ { d } } ^ { 2 } \big ] + \rho _ { k } \mathbb { E } \big [ [ h ] _ { C ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] \Big ) ,\tag{28}
$$

the volume cancelling between the two steps.

By Lemma 4 the first expectation equal $\mathcal { W } _ { 2 , k ^ { d } } \big ( f [ \mathcal { X } _ { k } ] , G [ \mathcal { X } _ { k } ] \big ) ^ { 2 }$ . The grids $\mathcal { X } _ { k }$ satisfy the effective-rank condition of Assumption 2 with a constant ν that does not depend on k, by Lemma 1, so Equation 7 applies with $n = k ^ { d }$ and gives

$$
\mathcal { W } _ { 2 , k ^ { d } } \big ( f [ \mathcal { X } _ { k } ] , G [ \mathcal { X } _ { k } ] \big ) ^ { 2 } \ \leq \ \frac { D ^ { 2 } } { m } ,\tag{29}
$$

uniformly in k, with D as in (13).

For the second expectation, $[ \cdot ] _ { C ^ { 1 / 2 } ( I ) }$ is a seminorm, so $[ h ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \le 2 [ \tilde { f } ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } + 2 [ \tilde { G } ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 }$ pathwise. Since $\tilde { f } \ { \overset { d } { = } } \ f$ and $\tilde { G } \stackrel { d } { = } G$ , this gives

$$
\begin{array} { r } { \mathbb { E } \big [ [ h ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] \ \leq \ 2 \mathbb { E } \big [ [ f ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] + 2 \mathbb { E } \big [ [ G ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] \ = : \ \Lambda , } \end{array}\tag{30}
$$

and Λ does not depend on k, since it involves only the laws of f and G. Therefore we need to prove that Λ stays finite. To bound the first term, let W be the r-th row of W, so that $g _ { r } ( x ) = \langle W _ { r } , x \rangle + b _ { r }$ . Since I is convex and $\phi$ is absolutely continuous with $| \phi ^ { \prime } ( z ) | \leq A ^ { \prime } ( 1 + | z | ) ^ { p ^ { \prime } }$ (Assumption 1 (2)), each map $x \mapsto \phi ( g _ { r } ( x ) )$ is Lipschitz on I with constant

$$
L _ { r } : = A ^ { \prime } \| W _ { r } \| \Big ( 1 + \operatorname* { m a x } _ { x \in I } | g _ { r } ( x ) | \Big ) ^ { p ^ { \prime } } .
$$

$$
\begin{array} { r } { \mathrm { A s } \ f ( x ) - f ( y ) = \sum _ { r = 1 } ^ { m } a _ { r } \bigl ( \phi ( g _ { r } ( x ) ) - \phi ( g _ { r } ( y ) ) \bigr ) , \mathrm { f o r } x , y \in I } \end{array}
$$

$$
| f ( x ) - f ( y ) | \leq \sum _ { r = 1 } ^ { m } \lvert a _ { r } \rvert L _ { r } \lVert x - y \rVert \leq \mathrm { d i a m } ( I ) ^ { 1 / 2 } \sum _ { r = 1 } ^ { m } \lvert a _ { r } \rvert L _ { r } \lVert x - y \rVert ^ { 1 / 2 } ,
$$

hence

$$
\mathbb { E } \big [ [ f ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] \ \leq \ \mathrm { d i a m } ( I ) \mathbb { E } \Big [ \Big ( \sum _ { r = 1 } ^ { m } \lvert a _ { r } \rvert L _ { r } \Big ) ^ { 2 } \Big ] \ < \ \infty ,
$$

the parameters being Gaussian and hence having moments of every order.

The same Lipschitz bound controls the second term. By Step 1 of the proof of Theorem 1, the pair $\left( g _ { 1 } ( x ) , g _ { 1 } ( y ) \right)$ has the law of $( Z _ { x } , Z _ { y } )$ in (2), and the constant $\tau _ { c } ^ { 2 }$ in (3) cancels, so applying the bound to the first neuron gives, for $x , y \in I$

$$
\begin{array} { r l } & { \mathbb { E } \big [ \big ( G ( x ) - G ( y ) \big ) ^ { 2 } \big ] = K _ { c } ( x , x ) - 2 K _ { c } ( x , y ) + K _ { c } ( y , y ) = \tau _ { a } ^ { 2 } \mathbb { E } \big [ \big ( \phi ( g _ { 1 } ( x ) ) - \phi ( g _ { 1 } ( y ) ) \big ) ^ { 2 } \big ] } \\ & { \qquad \leq \tau _ { a } ^ { 2 } \mathbb { E } \big [ L _ { 1 } ^ { 2 } \big ] \| x - y \| ^ { 2 } = : L ^ { 2 } \| x - y \| ^ { 2 } . } \end{array}
$$

Since $G ( x ) - G ( y )$ is centred Gaussian, it follows that

$$
\begin{array} { r } { \mathbb { E } | G ( x ) - G ( y ) | ^ { 4 d } \ \leq \ \mu _ { 4 d } L ^ { 4 d } \| x - y \| ^ { d + 3 d } , \qquad \mu _ { 4 d } : = \mathbb { E } | \xi | ^ { 4 d } , \ \xi \sim \mathcal { N } ( 0 , 1 ) . } \end{array}
$$

The Kolmogorov–Chentsov theorem in the form of Revuz and Yor [1999, Theorem I.2.1] covers Hölder exponents below $3 d / 4 d = \textstyle { \frac { 3 } { 4 } }$ , in particular ${ \frac { 1 } { 2 } } ,$ , and yields ${ \mathbb E } \big [ [ G ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 4 d } \big ] < \infty$ for the continuous version of G from Lemma 2, with which the modification provided by the theorem is indistinguishable. By Jensen’s inequality, $\mathbb { E } \big [ [ G ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 2 } \big ] \leq$ $\mathbb { E } \left[ [ G ] _ { \mathcal { C } ^ { 1 / 2 } ( I ) } ^ { 4 d } \right] ^ { 1 / ( 2 d ) } < \infty$ , and hence $\Lambda < \infty$

The pair $( \tilde { f } , \tilde { G } )$ is a coupling of f and G as random elements of ${ \mathcal { C } } ^ { 0 } ( I )$ , so by Definition 3, (28) and (30),

$$
\mathcal { W } _ { 2 } ^ { \mathcal { C } ^ { 0 } ( I ) } ( f , G ) ^ { 2 } \ \leq \ \mathbb { E } \big [ \| h \| _ { \mathcal { C } ^ { 0 } ( I ) } ^ { 2 } \big ] \ \leq \ \frac { 2 ( r + 1 ) ^ { 2 d } D ^ { 2 } } { m } \ + \ 2 ( r + 1 ) ^ { 2 d } \Lambda \rho _ { k } .\tag{31}
$$

Neither the left hand side nor the finite constant Λ depends on $k ,$ and $\rho _ { k } = { \cal O } ( k ^ { - 1 } )$ by (12), so letting $k \to \infty$ in (31) removes the second term and leaves

$$
{ \mathcal W } _ { 2 } ^ { \mathcal C ^ { 0 } ( I ) } ( f , G ) ^ { 2 } \ \leq \ \frac { 2 ( r + 1 ) ^ { 2 d } D ^ { 2 } } { m } \ = \ \frac { ( C ^ { * } ) ^ { 2 } } m ,
$$

since $D = \tau _ { a } ^ { 2 } A ^ { 2 } \sqrt { M _ { 4 p } / \nu }$ by (13). Taking square roots gives (9).

Corollary 1. For $S \subseteq I$ and any $h \in { \mathcal { C } } ^ { 0 } ( I )$ we have $\| h \| _ { \mathcal { C } ^ { 0 } ( S ) } \leq \| h \| _ { \mathcal { C } ^ { 0 } ( I ) }$ . Applying this pathwise to $h = f - G$ under any coupling and taking the infimum gives the first inequality; Theorem 2 gives the second.

## D Beyond Polynomial Activations

For a general activation, an interpolation inequality of Gagliardo–Nirenberg type [Nirenberg, 2011] bounds the supremum norm by the $\dot { L } ^ { 2 } ( I )$ norm and the Hölder seminorm together.

Lemma 5. Write $\ell _ { j } : = | I _ { j } | > 0 , \ell _ { \mathrm { m i n } } : = \operatorname* { m i n } _ { j } \ell _ { j } , L : = [ h ] _ { \mathcal { C } ^ { \gamma } ( I ) }$ and $M : = \| h \| _ { { \mathcal { C } } ^ { 0 } ( I ) }$ . Since h is continuous and I is compact, M is attained at some $x _ { 0 } \in I ;$ ; replacing h by −h if necessary we may assume $h ( x _ { 0 } ) = M \geq 0 . { \mathrm { I f } } M = 0$ the inequality is trivial. If $L = 0$ , then $h \equiv M$ , $\mathrm { s o } \ \| h \| _ { L ^ { 2 } ( I ) } = M \sqrt { \mathrm { V o l } ( I ) } \geq M \ell _ { \mathrm { m i n } } ^ { d / 2 }$ and already the second term on the right-hand side is at least 2M. Assume from now on $\begin{array} { r } { \dot { M } , L > 0 . } \end{array}$

For $x \in I ,$ , Hölder continuity gives $h ( x ) \geq M - L \Vert x - x _ { 0 } \Vert ^ { \gamma }$ . Choose $\begin{array} { r } { \delta : = \left( \frac { M } { 2 L } \right) ^ { 1 / \gamma } } \end{array}$ , so that $\begin{array} { r } { L \delta ^ { \gamma } = \frac { M } { 2 } } \end{array}$ , and $\eta : = \delta / \sqrt { d }$ Let

$$
J : = \prod _ { j = 1 } ^ { d } J _ { j } , \qquad J _ { j } : = [ x _ { 0 , j } - \eta , x _ { 0 , j } + \eta ] \cap I _ { j } .
$$

Every $x \in J$ satisfies $\| x - x _ { 0 } \| \leq { \sqrt { d } } \eta = \delta$ , hence $\begin{array} { r } { h ( x ) \geq \frac { M } { 2 } } \end{array}$ on J. Since $x _ { 0 , j } \in I _ { j }$ , we get $| J _ { j } | \geq \operatorname* { m i n } ( \eta , \ell _ { j } ) \geq$ min $\mathrm { \Omega } ( \eta , \ell _ { \mathrm { m i n } } )$ , so $\operatorname { V o l } ( J ) \geq \operatorname* { m i n } ( \eta , \ell _ { \operatorname* { m i n } } ) ^ { d }$ and

$$
\| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } \geq \int _ { J } h ( x ) ^ { 2 } d x \geq \frac { M ^ { 2 } } { 4 } \operatorname* { m i n } ( \eta , \ell _ { \mathrm { m i n } } ) ^ { d } .
$$

Case $I \colon \eta \leq \ell _ { \mathrm { m i n } }$ . Substituting η,

$$
\| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } \geq \frac { M ^ { 2 } } { 4 } d ^ { - d / 2 } \Bigl ( \frac { M } { 2 L } \Bigr ) ^ { d / \gamma } = \frac { M ^ { \frac { 2 \gamma + d } { \gamma } } } { 2 ^ { \frac { 2 \gamma + d } { \gamma } } d ^ { d / 2 } L ^ { d / \gamma } } ,
$$

and raising both sides to the power $\frac { \gamma } { 2 \gamma + d }$ gives

$$
M \leq 2 d ^ { \frac { \gamma d } { 2 ( 2 \gamma + d ) } } L ^ { \frac { d } { 2 \gamma + d } } \| h \| _ { L ^ { 2 } ( I ) } ^ { \frac { 2 \gamma } { 2 \gamma + d } } .
$$

Case $2 \colon \eta > \ell _ { \mathrm { m i n } }$ . Then $\begin{array} { r } { \| h \| _ { L ^ { 2 } ( I ) } ^ { 2 } \geq \frac { M ^ { 2 } } { 4 } \ell _ { \mathrm { m i n } } ^ { d } , \mathrm { i . e . } M \leq 2 \ell _ { \mathrm { m i n } } ^ { - d / 2 } \| h \| _ { L ^ { 2 } ( I ) } . } \end{array}$

In either case M is bounded by one of the two non-negative terms on the right-hand side of the claim, hence by their sum. □