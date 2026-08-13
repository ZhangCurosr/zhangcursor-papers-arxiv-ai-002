# MOON: Multi-Objective OrthoNormalized Updates for Multitask Learning

Shiji Zhou<sup>1,2,3</sup> Kunlin Lyu<sup>4</sup> Lei Zhang<sup>1</sup> Ruodong Wang<sup>4</sup> Yifan Sun<sup>2,4,∗</sup>

<sup>1</sup>Institute of Artificial Intelligence, Beihang University

<sup>2</sup>Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing,

Beihang University

<sup>3</sup>Beijing Academy of Artificial Intelligence (BAAI)

<sup>4</sup>Center for Applied Statistics, School of Statistics, Renmin University of China <sup>∗</sup>Corresponding author

## Abstract

Multi-objective optimization (MOO) has demonstrated significant success in multi-task learning by mitigating task conflicts through gradient manipulation. However, most existing methods flatten model parameters into vectors and perform gradient manipulation under Eu clidean geometry, thereby overlooking the matrix structure prevalent in modern architectures such as Transformers. In this paper, we show that gradient manipulation in Euclidean space does not generally yield the steepest descent direction under matrix geometry, potentially limiting optimization eficiency. Drawing from the theory of steepest descent for matrix-valued parameters, we propose MOON (Multi-Objective OrthoNormalized Updates), which performs gradient manipulation under spectral–nuclear norm geometry and uses the orthonormalized manipulated gradient for parameter updates. Theoretically, for smooth non-convex objectives, we establish convergence of the averaged Pareto-stationarity measure at rates of $\mathcal { O } ( T ^ { - 1 / 2 } )$ in the deterministic setting and O(T<sup>−1/4</sup>) under stochastic gradients. Empirical results across various benchmarks show that MOON consistently improves both optimization eficiency and final multi-task performance. Our code is available at https://github.com/KunlinLyu/MOON.

## Introduction

A wide range of practical problems in machine learning are inherently multi-objective, where the objectives often conflict and must be traded of [Sener and Koltun, 2018, Peitz and Hotegni, 2025]. In deep learning, task conflicts often manifest as gradient conflicts, with task gradients exhibiting competing directions during optimization. Multi-objective optimization (MOO) methods [Zitzler and Thiele, 1999] mitigate task-gradient conflicts via gradient manipulation and have achieved great success in multi-task learning [Sener and Koltun, 2018, Yu et al., 2020, Navon et al., 2022, Liu et al., 2024].

Existing MOO methods such as MGDA [Sener and Koltun, 2018], PCGrad [Yu et al., 2020], FAMO [Liu et al., 2024], and other variants [Liu et al., 2021a,b, Navon et al., 2022] typically flatten model parameters into a single vector and then perform gradient manipulation in this Euclidean vector space. However, modern architectures, such as Transformers [Vaswani et al., 2017], are dominated by matrix-valued parameters. Blindly flattening all parameters into vectors ignores the linear-mapping structure of matrix-valued parameters and may fail to identify the steepest descent direction under matrix geometry. This motivates MOO methods tailored to network architectures with matrix-valued parameters.

One possible approach is to combine conventional MOO methods with matrix-aware optimizers such as Muon [Jordan et al., 2024], and MuonClip [Team et al., 2026], which exploit matrix structure through preconditioning or orthonormalized updates. However, conventional MOO manipulates gradients under Euclidean geometry, whereas matrix-aware optimizers apply structure-preserving transformations induced by matrix geometry. Naively combining them, therefore, lacks a consistent geometry and creates a theoretical gap between their underlying assumptions, creates a mismatch between their underlying optimization principles. Moreover, establishing MOO convergence under this matrix geometry is nontrivial. Since orthonormalization discards the singular-value magnitudes of the aggregated gradient, standard Euclidean descent analyses based on squared gradient norms are no longer directly applicable. The analysis must instead characterize Pareto stationarity under matrix geometry and exploit spectral–nuclear norm duality to connect the orthonormalized direction with optimization progress.

To address these challenges, we propose Multi-Objective OrthoNormalized Updates (MOON), a geometrically consistent framework for multi-objective optimization with matrix-valued parameters [Bernstein and Newhouse, 2024]. Starting from simultaneous matrix-smooth upper bounds of the objectives, we formulate the selection of a common descent direction as a spectral-normregularized minimax problem. Through spectral–nuclear norm duality, the corresponding dua problem determines the task weights by minimizing the nuclear norm of the weighted aggregate gradient, while the solution to the primal problem is given by its polar factor. This formulation provides a principled connection between multi-objective gradient manipulation and orthonormalized matrix updates, distinguishing MOON from conventional MOO methods based on Euclidean geometry. For smooth non-convex objectives, we establish convergence of the averaged Paretostationarity measure at rates of $\mathcal { O } ( T ^ { - 1 / 2 } )$ under deterministic gradients and $\mathcal { O } ( T ^ { - 1 / 4 } )$ under unbiased stochastic gradients. Extensive experiments further demonstrate that MOON accelerates optimization and achieves competitive or improved final performance across multiple benchmarks.

Our main contributions are: (1) We propose MOON, a structure-aware MOO method. MOON is derived by our analysis of multi-objective steepest descent for matrix values, providing a geometrically consistent update for modern architectures. (2) We establish convergence to Pareto stationarity for smooth non-convex objectives, where MOON attains an ${ \cal O } ( T ^ { - 1 / 2 } )$ rate under exact gradients and an ${ \cal O } ( T ^ { - 1 / 4 } )$ rate under unbiased stochastic gradients with bounded noise. (3) We conduct extensive experiments on diverse multi-task scenarios with various architectures, demonstrating faster optimization and competitive or improved final performance compared with representative MOO baselines.

## Background

Multi-Objective Optimization. Multi-objective optimization (MOO) aims to optimize multiple objectives simultaneously [Zitzler and Thiele, 1999]:

$$
\operatorname* { m i n } _ { \pmb { \theta } } L ( \pmb { \theta } ) = ( \ell ^ { 1 } ( \pmb { \theta } ) , \dots , \ell ^ { m } ( \pmb { \theta } ) ) ^ { \top } ,\tag{1}
$$

where $m \geq 2$ denotes the number of objectives, and $\ell ^ { i } : \mathbb { R } ^ { n }  \mathbb { R }$ is the i-th loss function. Denote $\Delta _ { m }$ as the $( m - 1 )$ -dimensional probability simplex. The concept of Pareto optimality/stationarity [Cust´odio et al., 2011] is introduced to determine whether a solution to MOO is optimal/critical.

Definition 1. Pareto Optimality: For any two solutions $\theta , \theta ^ { \prime }$ , we say that $\pmb \theta$ dominates $\pmb { \theta } ^ { \prime } .$ , denoted as $\mathbf { \nabla } \theta \mathbf { \nabla } \prec \theta ^ { \prime }$ or $\pmb { \theta } ^ { \prime } \succ \pmb { \theta }$ , if $\ell ^ { i } ( \pmb \theta ) \leq \ell ^ { i } ( \pmb \theta ^ { \prime } )$ for all i, and there exists an i such that $\ell ^ { i } ( \pmb \theta ) < \ell ^ { i } ( \pmb \theta ^ { \prime } )$ . A solution $\pmb { \theta } ^ { * }$ is called Pareto optimal if it is not dominated by any other solution. Pareto Stationary: $\mathrm { A }$ solution $\pmb \theta$ is called Pareto stationary if there exists some $\lambda \in \Delta _ { m }$ such that $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \lambda _ { i } \nabla _ { \pmb { \theta } } \ell ^ { i } ( \pmb { \theta } ) = 0 } \end{array}$

The typical Multiple Gradient Descent Algorithm (MGDA) method [Sener and Koltun, 2018] has been shown to be the steepest method for MOO Fliege and Svaiter [2000]. Specifically, under the assumption of L-smoothness with respect to the $\| \cdot \| _ { 2 }$ norm, in each iteration t, MGDA aims to find a direction $\mathbf { } d _ { t }$ to maximize the minimum decrease in losses by solving the following subproblem:

$$
\pmb { d } _ { t } = \underset { \pmb { d } \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \underset { i \in [ m ] } { \operatorname* { m a x } } \big \{ \nabla \ell ^ { i } ( \pmb { \theta } _ { t } ) ^ { \top } \pmb { d } + \frac { L } { 2 } \| \pmb { d } \| _ { 2 } ^ { 2 } \big \} .\tag{2}
$$

The right-hand side $\begin{array} { r } { \nabla \ell ^ { i } ( \pmb { \theta } _ { t } ) ^ { \top } \pmb { d } + \frac { L } { 2 } \| \pmb { d } \| _ { 2 } ^ { 2 } } \end{array}$ is actually the quadratic upper bound of the objective decrease $\ell ^ { i } ( \pmb { \theta } _ { t } + \pmb { d } ) - \ell ^ { i } ( \pmb { \theta } _ { t } )$ . MOO aims to optimize all the objectives simultaneously, and mathematically to maximize the minimum objective decrease, which formulates the subproblem above. Other MOO methods such as PCGrad Yu et al. [2020], CAGrad Liu et al. [2021a] and other variants Zhou et al. [2022], Navon et al. [2022], Fernando et al. [2023], He et al. [2024] also share a similar idea of seeking a mutual descent direction by optimizing towards some specific upper bounds.

Steepest Descent via Orthonormalized Updates. In modern neural network architectures, such as Transformer, matrix parameters are fundamental components.

Suppose the objective $\mathcal { L } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ is L-smooth with respect to the spectral norm $\| \cdot \| _ { S _ { \infty } }$ . Then $\mathcal { L }$ admits an upper bound that is quadratic in the spectral norm[Bernstein and Newhouse, 2024].

$$
\mathcal { L } ( \boldsymbol { \Theta } ) \leq \mathcal { L } ( \boldsymbol { \Theta } ^ { \prime } ) + \left. \nabla \mathcal { L } ( \boldsymbol { \Theta } ^ { \prime } ) , \boldsymbol { \Theta } - \boldsymbol { \Theta } ^ { \prime } \right. + \frac { L } { 2 } \| \boldsymbol { \Theta } - \boldsymbol { \Theta } ^ { \prime } \| _ { S _ { \infty } } ^ { 2 } .\tag{3}
$$

By minimizing this quadratic upper bound, we obtain the steepest descent under the spectral norm.

Proposition 1 (Steepest Descent for Matrix Parameters Bernstein and Newhouse [2024]). In each iteration $t ,$ steepest descent under the spectral norm aims to solve the following subproblem:

$$
\Delta \Theta _ { t } = \operatorname * { a r g m i n } _ { \Delta \Theta } \left[ \langle \nabla \mathcal { L } ( \Theta _ { t } ) , \Delta \Theta \rangle + \frac { L } { 2 } \| \Delta \Theta \| _ { S _ { \infty } } ^ { 2 } \right] ,\tag{4}
$$

where $\langle \cdot , \cdot \rangle$ denotes the Frobenius inner product.

From Bernstein and Newhouse [2024] we know that, if $\nabla \mathcal { L } ( \Theta _ { t } )$ has a reduced SVD given by $\begin{array} { r } { \nabla \mathcal { L } ( \Theta _ { t } ) = U _ { t } \Sigma _ { t } V _ { t } ^ { \top } } \end{array}$ , the subproblem (4) is solved with a step size $\alpha _ { t } = \mathrm { T r } \big ( \Sigma _ { t } \big ) = \| \nabla \mathcal { L } ( \Theta _ { t } ) \| _ { S _ { 1 } } / L$ and an update:

$$
\Delta \Theta _ { t } = - \alpha _ { t } \cdot U _ { t } V _ { t } ^ { \top } .\tag{5}
$$

This result highlights a fundamental diference between steepest-descent updates for matrixvalued parameters and those derived under Euclidean vector geometry. Whereas the Euclidean gradient preserves the original singular-value magnitudes, the spectral-norm steepest-descent direction is characterized by the polar factor $\boldsymbol { U } _ { t } \boldsymbol { V } _ { t } ^ { \intercal }$ , which preserves the singular subspaces while normalizing the nonzero singular values. Consequently, the resulting direction is less dominated by a few large singular components, while its overall update magnitude can be controlled separately by the learning rate.

Matrix-aware optimizers such as Shampoo [Gupta et al., 2018] and Muon [Jordan et al., 2024] have recently attracted increasing attention for exploiting matrix structure during optimization. Shampoo constructs structured preconditioners from gradient statistics, whereas Muon applies Newton–Schulz iterations to a momentum-smoothed gradient matrix to eficiently approximate its polar factor without an explicit singular value decomposition. Empirical studies have demonstrated the efectiveness and optimization eficiency of this strategy in large-scale Transformer training in large language models [Liu et al., 2025, Team et al., 2026].

## Multi-Objective Orthonormalized Updates

Although solving the Euclidean-norm minimax problem (Equation 2) yields the steepest descent direction in the Euclidean space, this direction may not be the steepest descent direction in the native matrix space of the matrix-valued parameters—such as the formulation shown in Equation 4. Consequently, existing methods struggle to adapt to the training of modern neural network architectures. Therefore, we extend the multi-objective optimization framework to matrix-valued parameters. For clarity, we present the derivation using a single matrix-valued parameter. For networks containing multiple matrix-valued parameter blocks, the same construction is applied blockwise.

Suppose that for each objective $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ that is L-smooth with respect to the spectral norm $\| \cdot \| _ { S _ { \infty } }$ , we assume that an update $\Theta _ { t + 1 } = \Theta _ { t } - \alpha W _ { t }$ is performed at step t and derive its quadratic upper bound from Equation (3)

$$
\ell ^ { i } ( \Theta _ { t + 1 } ) \leq \ell ^ { i } ( \Theta _ { t } ) - \alpha \langle \nabla \ell ^ { i } ( \Theta _ { t } ) , W _ { t } \rangle + \frac { L \alpha ^ { 2 } } { 2 } \| W _ { t } \| _ { S _ { \infty } } ^ { 2 } .\tag{6}
$$

Similar to Equation 2, we seek a mutual update direction in the matrix space that minimizes the worst-case upper bound. Thus, we aim to solve the following problem.

Proposition 2. Let $\alpha \leq 1 / L$ . Minimizing the common upper bound is equivalent to solving the following spectral-norm-regularized minimax problem:

$$
\operatorname* { m i n } _ { W _ { t } } \operatorname* { m a x } _ { i } - \langle \nabla \ell ^ { i } ( \Theta _ { t } ) , W _ { t } \rangle + \frac { 1 } { 2 } \| W _ { t } \| _ { S _ { \infty } } ^ { 2 } .\tag{7}
$$

Consequently, we derive the quadratic upper bound for MOO with matrix parameters, which distinguishes our approach from traditional MOO methods.

Since the primal problem (7) is dificult to solve, we usually consider the following dual problem.

Proposition 3. The dual problem of (7) is

$$
\operatorname* { m i n } _ { z \in \Delta _ { m } } \quad \frac { 1 } { 2 } \| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } ^ { 2 } ,\tag{8}
$$

where $\Delta _ { m }$ denotes the probability simplex. Assume the optimum of the dual (8) is $z ^ { * }$ and the manipulated gradient $\begin{array} { r } { G _ { t } = \sum _ { i = 1 } ^ { m } z _ { i , t } ^ { * } \nabla \ell ^ { i } ( \Theta _ { t } ) } \end{array}$ has full rank. The exact optimal solution of (7) is given by

$$
\widehat { \pmb { W } } _ { t } = \boldsymbol { C } \cdot \pmb { U } _ { t } \pmb { V } _ { t } ^ { \top } ,
$$

where $\begin{array} { r } { C = \| \sum _ { i = 1 } ^ { m } z _ { i , t } ^ { * } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } } \end{array}$ is the nuclear norm. $I f C = 0$ , then $\Theta _ { t }$ is Pareto stationary, and we set $\widehat { \pmb { W } } _ { t } = \pmb { W } _ { t } = \mathbf { 0 }$ ; otherwise, the following construction applies. $\mathbf { } U _ { t } , V _ { t }$ are the left and right (compact) singular matrices of the manipulated gradient $G _ { t }$

Proposition 3 derives the dual of problem (7) and shows how to recover the exact primal solution from the dual optimum. For $C > 0$ , the orthonormalized direction used in Algorithm 1 is $\begin{array} { r } { \pmb { W _ { t } } = \pmb { U _ { t } } \pmb { V } _ { t } ^ { \top } = \frac { 1 } { C } \widehat { \pmb { W } } _ { t } } \end{array}$ . Thus, $\mathbf { } W _ { t }$ retains the direction of the exact minimax solution, while the update magnitude is controlled separately by the learning rate α in Step 6.

Remark: Comparison with Euclidean MOO methods. For a fixed iterate $\mathbf { \Theta } _ { \Theta _ { t } , \mathrm { ~ \tiny ~ \Omega ~ } }$ define the weighted aggregate gradient as $\begin{array} { r } { \pmb { G } _ { t } ( z ) = \sum _ { i = 1 } ^ { m } z _ { i } \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) , z \in \Delta _ { m } } \end{array}$ . Convexity and spectral–nuclear norm duality can imply the optimization meaning of its nuclear norm (Appendix A.2):

$$
z ^ { \top } L ( \Theta _ { t } ) - \operatorname* { m i n } _ { \Theta \in \Omega } z ^ { \top } L ( \Theta ) \leq O \left( \left. G _ { t } ( z ) \right. _ { S _ { 1 } } \right) .\tag{9}
$$

Thus, $\| G _ { t } ( z ) \| _ { S _ { 1 } }$ provides a matrix-geometric certificate of the scalarized suboptimality at $\Theta _ { t }$ Let $z _ { t } ^ { \mathrm { M O O N } }$ be an exact minimizer of the nuclear-norm dual problem in Equation (8), and let $z _ { t } ^ { \mathrm { E } } \in \Delta _ { m }$ denote the weights obtained by a Euclidean min-norm method such as MGDA. Since $\boldsymbol { z } _ { t } ^ { \mathrm { E } }$ is feasible for the MOON dual problem, the optimality of $z _ { t } ^ { \mathrm { M O O N } }$ gives

$$
\left\| \boldsymbol { G } _ { t } \left( \boldsymbol { z } _ { t } ^ { \mathrm { M O O N } } \right) \right\| _ { S _ { 1 } } \leq \left\| \boldsymbol { G } _ { t } \left( \boldsymbol { z } _ { t } ^ { \mathrm { E } } \right) \right\| _ { S _ { 1 } } .\tag{10}
$$

Therefore, at the same iterate, the exact MOON weighting provides a no-looser nuclear-normdependent certificate than Euclidean min-norm weighting. Define the corresponding certificate gap as

$$
\begin{array} { r } { \mathrm { g a p } _ { t } : = \left. \boldsymbol { G } _ { t } \left( \boldsymbol { z } _ { t } ^ { \mathrm { E } } \right) \right. _ { S _ { 1 } } - \left. \boldsymbol { G } _ { t } \left( \boldsymbol { z } _ { t } ^ { \mathrm { M O O N } } \right) \right. _ { S _ { 1 } } \geq 0 . } \end{array}\tag{11}
$$

For a fixed sequence of iterates, the diference between the corresponding certificate terms is

$$
\Delta _ { \mathrm { c e r t } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { g a p } _ { t } \geq 0 ,\tag{12}
$$

with a strict inequality whenever the Euclidean weighting is not nuclear-norm optimal at least at one iterate<sup>1</sup>.

## Practical Implementation

Applying the exact MOON formulation to large-scale training presents three practical challenges: fluctuations in the aggregate gradient, the cost of exact polar-factor computation, and the innerloop optimization required to solve the dual problem (8). To address these challenges, practical MOON incorporates three components: (1) gradient momentum to stabilize the aggregate gradient across iterations, (2) Newton–Schulz iterations to eficiently approximate its polar factor, and (3) a single-step online update to track the dual task weights. Together, these components yield the practical MOON procedure summarized in Algorithm 1.

Gradient Momentum. At iteration t, MOON constructs the weighted aggregate gradient $\begin{array} { r } { \pmb { G } _ { t } = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) } \end{array}$ . Because both the stochastic task gradients and the task weights may change over time, directly constructing the matrix-aware update from $G _ { t }$ can lead to substantia iteration-to-iteration variation. We therefore maintain an exponential moving average

$$
M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu G _ { t } , \qquad 0 < \mu \leq 1 .\tag{13}
$$

The momentum matrix filters short-term variations in the aggregate gradient and provides a more stable matrix signal for constructing the subsequent orthonormalized direction.

Algorithm 1 Multi-Objective OrthoNormalized updates   
1: Input: Initial parameter $\Theta _ { 1 } , M _ { 0 } \gets 0 .$ , loss functions $\{ \ell ^ { i } \} _ { i = 1 } ^ { m }$ , learning rate $\alpha , \beta ,$ momentum   
$0 < \mu \leq 1$ , weight decay of logits update $\gamma , \pmb { \xi } _ { 1 } \gets \mathbf { 0 }$   
2: for $t = 1$ to $T$ do   
3: Normalize weight $z _ { t } \gets \mathrm { S o f t m a x } ( \pmb { \xi } _ { t } )$   
4: Compute composite gradient:   
$G _ { t } = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } )$   
5: Compute momentum: $M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu G _ { t }$   
6: Perform orthonormalization on momentum by:   
$\pmb { W } _ { t } = \pmb { U } _ { t } \pmb { V } _ { t } ^ { \top }$ (approximated by Newton-Schulz)   
7: Update model parameter $\Theta _ { t + 1 } = \Theta _ { t } - \alpha W _ { t }$   
8: Update weight logits by:   
$\pmb { \xi } _ { t + 1 } = \pmb { \xi } _ { t } - \beta ( \pmb { \delta } _ { t } + \gamma \pmb { \xi } _ { t } )$   
where $\pmb { \delta } _ { t } = [ \langle \pmb { W } _ { t } , \nabla \ell ^ { 1 } ( \pmb { \Theta } _ { t } ) \rangle , \dots , \langle \pmb { W } _ { t } , \nabla \ell ^ { m } ( \pmb { \Theta } _ { t } ) \rangle ] ^ { \top }$   
9: end for

Polar-Factor Approximation. Given the momentum matrix $M _ { t }$ , MOON uses its polar factor as the matrix-aware update direction:

$$
\boldsymbol { W } _ { t } = \operatorname { P o l a r } ( \boldsymbol { M } _ { t } ) .\tag{14}
$$

Since computing the exact polar factor through singular value decomposition can be expensive, we use a finite number of Newton–Schulz iterations to approximate $\mathrm { P o l a r } ( { M } _ { t } )$ similar to previous works Jordan et al. [2024], Bernstein and Newhouse [2024], Bj¨orck and Bowie [1971], Kovarik [1970]. By convention, we define $\mathrm { P o l a r } ( \mathbf { 0 } ) = \mathbf { 0 }$ , so that ${ \cal W } _ { t } = { \bf 0 }$ whenever ${ M } _ { t } = \mathbf { 0 }$

Online Approximation of the Dual Weights. Solving the nuclear-norm dual problem (8) would introduce an expensive inner optimization loop. We instead use a single online update to track its solution, following the approximation strategy of FAMO [Liu et al., 2024]. We use the following momentum-based, scale-normalized surrogate direction for tracking the dual weights.:

$$
\pmb { \delta } _ { t } = \left[ \left. \nabla \ell ^ { 1 } ( \Theta _ { t } ) , \pmb { W } _ { t } \right. , \ldots , \left. \nabla \ell ^ { m } ( \Theta _ { t } ) , \pmb { W } _ { t } \right. \right] ^ { \top } .\tag{15}
$$

To maintain the simplex constraint, we parameterize the task weights using logits and update

$$
\pmb { \xi } _ { t + 1 } = \pmb { \xi } _ { t } - \beta \left( \pmb { \delta } _ { t } + \gamma \pmb { \xi } _ { t } \right) , \pmb { z } _ { t + 1 } = \mathrm { S o f t m a x } \left( \pmb { \xi } _ { t + 1 } \right) ,\tag{16}
$$

where $\beta$ is the weight update stepsize, and $\gamma$ regularizes the logits. It provides an eficient momentumbased approximation to the exact dual solution while ensuring $z _ { t + 1 } \in \Delta _ { m }$

## Convergence Analysis

We analyze the convergence of MOON for smooth non-convex objectives. For any Θ, define the nuclear-norm Pareto-stationarity measure

$$
\mathcal { G } ( \Theta ) : = \operatorname* { m i n } _ { z \in \Delta _ { m } } \left. \sum _ { i = 1 } ^ { m } z _ { i } \nabla \ell ^ { i } ( \Theta ) \right. _ { S _ { 1 } } .\tag{17}
$$

In the unconstrained setting, $\mathcal { G } ( \Theta ) = 0$ if and only if Θ is Pareto stationary. We therefore study the averaged measure $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { \bar { G } } ( \Theta _ { t } ) } \end{array}$ . We first consider the deterministic setting.

Theorem 1 (Deterministic Convergence). Suppose that each objective $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ is L-smooth with respect to the spectral norm. Assume that there exist constants $H > 0$ and $B > 0$ , independent of both T and t, such that $\begin{array} { r } { \operatorname* { m a x } _ { t \in [ T + 1 ] } \operatorname* { m a x } _ { i \in [ m ] } \| \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } \leq H } \end{array}$ and ma $\mathbf { x } _ { t \in [ T + 1 ] } \operatorname* { m a x } _ { i \in [ m ] } | \ell ^ { i } ( \Theta _ { t } ) | \leq B$ Choose $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma < m H T$ . Then, for every fixed $\mu \in ( 0 , 1 ]$ , the iterates generated by Algorithm 1 satisfy

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \leq O ( T ^ { - 1 / 2 } ) .
$$

The detailed proof is provided in Appendix A.3. Consequently, we demonstrate that our algorithm achieves a convergence rate of $O ( 1 / \sqrt { T } )$ , showing that MOON converges to Pareto stationarity at a standard sublinear rate for smooth non-convex optimization. We next consider unbiased stochastic task gradients.

Theorem 2 (Stochastic Convergence). Under the same assumption as Theorem 1. We assume that the stochastic gradient estimator is unbiased and has variance bounded by σ. Choosing $\begin{array} { r } { \alpha = \sqrt { \frac { \mu B } { T L } } } \end{array}$ $\begin{array} { r } { \beta = \frac { 1 } { ( H + \sqrt { m } \rho \sigma ) m T } , 0 < \gamma < ( H + \sqrt { m } \rho \sigma ) m T } \end{array}$ and $\begin{array} { r } { \mu = \operatorname* { m i n } \{ \frac { \sqrt { L B } } { \sigma \rho \sqrt { T } } , 1 \} } \end{array}$ , where $\rho : = { \sqrt { \operatorname* { m i n } \{ p , q \} } }$ , then the iterations of Algorithm 1 satisfy

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \right] \leq \mathcal { O } ( 1 / T ^ { 1 / 4 } ) .\tag{18}
$$

The detailed proof is provided in Appendix A.4. The expected Pareto-stationarity measure of MOON satisfies $\mathcal { O } ( T ^ { - 1 / 4 } )$ bound. This slower rate reflects the additional error introduced by gradient noise and its interaction with momentum, and is also standard under stochastic gradients.

## Related Work

Multi-objective Optimization. Multi-objective optimization (MOO) optimizes multiple tasks simultaneously, with the central challenge of resolving conflicts among task gradients [Sener and Koltun, 2018, Yu et al., 2020, Liu et al., 2021a, Navon et al., 2022, Liu et al., 2024]. Methods most relevant to our work follow two main lines. Loss balancing approaches [Kendall et al., 2018, Liu et al., 2019, Lin et al., 2021] adjust objective weights using loss statistics, optimizing an explicit weighted objective whose update is a linear combination of task-specific updates. These methods are often motivated by empirical heuristics and have limited theoretical support. Gradient manipulation methods [Sener and Koltun, 2018, Yu et al., 2020, Liu et al., 2021a,b, Navon et al., 2022, Fernando et al., 2023, Liu et al., 2024] instead modify task gradients to obtain a common update direction that balances optimization across tasks. They generally provide stronger theoretical guarantees and often achieve better performance than loss balancing. However, existing methods typically flatten all parameters into a single vector, preventing them from attaining the steepest descent direction in the native non-Euclidean geometry of structured parameters. Our method preserves the matrix structure of the parameters and performs gradient manipulation directly in this geometry, improving both training eficiency and performance.

Optimization for matrix-valued parameters. Modern architectures such as Transformers [Vaswani et al., 2017] are dominated by matrix-valued parameters, motivating optimizers that exploit this structure. K-FAC approximates layerwise Fisher blocks using Kronecker factors [Martens and Grosse, 2015]; Adafactor factorizes row- and column-wise second moments to reduce memory usage [Shazeer and Stern, 2018]; and Shampoo uses tensor-structured preconditioners to improve conditioning and accelerate convergence [Gupta et al., 2018]. Muon instead approximately orthogonalizes momentum matrices for hidden-layer weights, producing eficient matrix-aware updates [Jordan et al., 2024]. Its theoretical properties have been studied from complementary perspectives: Li and Hong establish convergence guarantees [Li and Hong, 2025], while Kovalev interprets gradient orthogonalization as non-Euclidean trust-region optimization [Kovalev, 2025].

Subsequent work improves the eficiency, stability, and scalability of Muon. MuonBP reduces distributed communication through block-periodic orthogonalization [Khaled et al., 2025], while Polar Express develops GPU-friendly matrix-sign iterations for eficient polar-factor computation [Amsel et al., 2026]. MuonClip improves large-model training stability by controlling attention-logit growth through QK clipping [Team et al., 2026], and large-scale studies further demonstrate Muon’s practical eficiency and scalability in language-model pretraining [AI et al., 2025, Liu et al., 2025]. Despite these advances, such methods optimize a single aggregate objective and do not directly address conflicts among task objectives. MOON extends matrix-aware optimization to the multi objective setting by combining task gradients according to the geometry of matrix-valued parameters and applying orthonormalized updates. This preserves parameter structure while resolving task conflicts, improving training eficiency and final performance in our experiments.

## Experiments

In this section, we conduct experiments with MOON on several benchmarks to evaluate the performance and convergence. Our experiments cover simulation, multi-task scene understanding, classification, and regression.

Experimental Setup. We consider 5 benchmarks widely used in multi-task learning research: MultiMNIST (2 tasks) [Sabour et al., 2017], NYU-v2 (3 tasks) [Silberman et al., 2012], CityScapes (2 tasks) [Cordts et al., 2016], QM9 (11 tasks) [Blum and Reymond, 2009] and CelebA (40 tasks) [Liu et al., 2015]. We compare the proposed method with 12 multi-objective optimization baselines: Single-task learning (STL), Linear Scalarization (LS), a scale-invariant baseline (SI) that applies equal weighting to the log losses, Random Loss Weighting (RLW) [Lin et al., 2021], Dynamic Weight Average (DWA) [Liu et al., 2019], Uncertainty Weighting (UW) [Kendall et al., 2018], MGDA [Sener and Koltun, 2018], PCGrad [Yu et al., 2020], CAGrad [Liu et al., 2021a], IMTL-G [Liu et al., 2021b], Nash-MTL [Navon et al., 2022], FAMO [Liu et al., 2024]. We discuss each experiment in the following subsections.<sup>2</sup>

<table><tr><td rowspan="2">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3">∆m%↓</td></tr><tr><td></td><td>MIoU↑ PIx AcC↑</td><td>ABS ERR↓</td><td>REL ERR↓</td><td>ANGLE DIST↓</td><td></td><td colspan="3">WITHIN t° ↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>MEAN</td><td>MEDIAN</td><td>11.25</td><td>22.5</td><td>30</td></tr><tr><td>STL</td><td>38.30</td><td>63.76</td><td>0.6754</td><td>0.2780</td><td>25.01</td><td>19.21</td><td>30.14</td><td>57.20</td><td>69.15</td><td></td></tr><tr><td>LS</td><td>39.29</td><td>65.33</td><td>0.5493</td><td>0.2263</td><td>28.15</td><td>23.96</td><td>22.09</td><td>47.50</td><td>61.08</td><td>5.59</td></tr><tr><td>SI</td><td>38.45</td><td>64.27</td><td>0.5354</td><td>0.2201</td><td>27.60</td><td>23.37</td><td>22.53</td><td>48.57</td><td>62.32</td><td>4.39</td></tr><tr><td>RLW</td><td>37.17</td><td>63.77</td><td>0.5759</td><td>0.2410</td><td>28.27</td><td>24.18</td><td>22.26</td><td>47.05</td><td>60.62</td><td>7.78</td></tr><tr><td>DWA</td><td>39.11</td><td>65.31</td><td>0.5510</td><td>0.2285</td><td>27.61</td><td>23.18</td><td>24.17</td><td>50.18</td><td>62.39</td><td>3.57</td></tr><tr><td>UW</td><td>36.87</td><td>63.17</td><td>0.5446</td><td>0.2260</td><td>27.04</td><td>22.61</td><td>23.54</td><td>49.05</td><td>63.65</td><td>4.05</td></tr><tr><td>MGDA</td><td>28.40</td><td>58.24</td><td>0.5979</td><td>0.2335</td><td>25.00</td><td>19.49</td><td>29.56</td><td>56.84</td><td>69.11</td><td>1.23</td></tr><tr><td>PCGRAD</td><td>39.22</td><td>65.98</td><td>0.5373</td><td>0.2273</td><td>27.27</td><td>23.08</td><td>23.17</td><td>49.23</td><td>62.98</td><td>3.40</td></tr><tr><td>GRADDROP</td><td>39.39</td><td>65.12</td><td>0.5455</td><td>0.2279</td><td>27.48</td><td>22.96</td><td>23.38</td><td>49.44</td><td>62.87</td><td>3.58</td></tr><tr><td>CAGRAD</td><td>39.00</td><td>65.77</td><td>0.5561</td><td>0.2135</td><td>26.30</td><td>21.53</td><td>25.86</td><td>52.47</td><td>65.65</td><td>-0.12</td></tr><tr><td>IMTL-G</td><td>39.35</td><td>65.60</td><td>0.5426</td><td>0.2256</td><td>26.02</td><td>21.19</td><td>26.20</td><td>53.13</td><td>66.24</td><td>-0.76</td></tr><tr><td>NASHMTL</td><td>40.13</td><td>65.93</td><td>0.5261</td><td>0.2171</td><td>25.26</td><td>20.08 19.14</td><td>28.40 30.22</td><td>55.47</td><td>68.15</td><td>-4.04</td></tr><tr><td>FAMO</td><td>36.41</td><td>63.20</td><td>0.5406</td><td>0.2182</td><td>25.01</td><td></td><td></td><td>57.45</td><td>69.37</td><td>-4.11</td></tr><tr><td>MOON (OURS)</td><td>39.41</td><td>67.03</td><td>0.4891</td><td>0.2091</td><td>25.61</td><td>20.27</td><td>28.89</td><td>54.87</td><td>67.32</td><td>-4.63</td></tr></table>

Table 2: Results on NYU-v2 (3 tasks) dataset. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. The task specific metrics and the average performance drop $\Delta m \%$ are reported.

## A Toy Case

To better understand why we cannot directly apply structure-aware optimizers such as Muon [Jordan et al., 2024] to existing MOO methods, we design a toy experiment to provide an intuitive illustration.

In this experiment, we construct a 2-task optimization problem based on a multi-objective least-squares formulation. Given two 6-dimensional vectors x and y, we define two objectives $\begin{array} { r } { L _ { 1 } = \frac { 1 } { 6 } \| \Theta x - y \| _ { 2 } ^ { 2 } } \end{array}$ and $\begin{array} { r } { L _ { 2 } = \frac { 1 } { 6 } \| \Theta x + \pmb { y } \| _ { 2 } ^ { 2 } } \end{array}$ over a matrix parameter $\Theta \in \mathbb { R } ^ { 6 \times 6 }$ . We optimize Θ using three methods: (i) MGDA [Sener and Koltun, 2018], (ii) MGDA with Muon optimizer [Jordan et al., 2024] applied directly, and (iii) MOON. Then we plot the average of the two losses during the 1500-step optimization in Figure 1.

Figure 1 shows the average loss of the two tasks over training steps in this toy experiment. MOON converges faster and reaches a lower final average loss, indicating a more balanced optimization on matrix-valued parameters. In contrast, directly applying Muon to MGDA fails to achieve the same improvement.

## Multi-task Classification

We evaluate our method on MultiMNIST [Sabour et al., 2017] with Vision Transformer (ViT) [Dosovitskiy et al., 2021] backbone to assess convergence and performance under matrix-structured parameters. The ViT contains matrix-valued weights in the Transformer encoder (attention projections and MLP linear layers). ViT is composed of 76.88% matrix parameters. We treat classifying the left and right digits as two tasks, reporting per-task cross-entropy training loss for convergence and test accuracy for performance. An additional classification experiment is conducted on CelebA, adopting a CNN backbone with 99.46% learnable convolutional weights treated as matrix parameters. In our experiments, we follow the treatment of convolutional weights in Muon [Jordan et al., 2024], converting 4D convolutional parameters into matrices by flattening the last three dimensions. It evaluates training eficiency and performance with a large number of objectives (40 tasks).

![](images/c2b8576ee4e0eeb904337b3e8be208d4a0593c4574272732542c043bd605ef8a.jpg)  
Figure 1: Average loss curve on the toy case.

![](images/24afe0e9238eef9ecfa7e4e3527f522c9aec5702837a66d4acf08ad07a8a10db.jpg)  
(a) ViT left task

![](images/3f97b103c5399560be1b83646d0adc341144b5915864c6da2d4fab9fbfe2abce.jpg)  
(b) ViT right task  
Figure 2: Convergence of per-task training cross-entropy losses vs. training epochs on MultiMNIST.

In the MultiMNIST experiment, we observe from Figure 2 that our method reduces the training loss for both tasks more rapidly in the early stage and maintains a consistent advantage throughout training. Table 1 reports the test accuracies for the two classification tasks and their average. MOON achieves the best performance on both tasks and the highest average accuracy among all compared MOO methods. The right-hand side of Table 3 shows that MOON achieves performance comparable to state-of-the-art methods on the 40-task CelebA benchmark, further supporting the results.

## Multi-task Scene Understanding

We apply MOON to multi-task scene understanding and evaluate it on the NYU-v2 [Silberman et al., 2012] and CityScapes [Cordts et al., 2016] benchmarks. We adopt the SegNet-MTAN architecture Badrinarayanan et al. [2017], Liu et al. [2019], a standard backbone for multi-task dense prediction. In this model, matrix-structured parameters or convolutional filters arise in the shared SegNet encoder-decoder as well as the MTAN attention, which account for 99.57% of all parameters. We measure multi-objective optimization performance using the average performance drop $\Delta m \%$ relative to the single-task learning (STL) baseline [Navon et al., 2022]. For each method $m ,$ , the performance drop is calculated by $\begin{array} { r } { \Delta m \% = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } ( - 1 ) ^ { \delta _ { i } } ( M _ { i } ^ { m } - M _ { i } ^ { S \bar { T } L } ) \cdot 1 0 0 / M _ { i } ^ { S T L } } \end{array}$ , where $M _ { i } ^ { m }$ denotes the i-th metric achieved by method m, and $\delta _ { i } = 0$ if higher values are better for metric $i ;$ otherwise $\delta _ { i } = 1$ . Table 2 and Table 3 summarize the results on NYU-v2 and CityScapes, respectively. Figure 3 demonstrates how the test average performance drop $\Delta m \%$ decreases during training.

<table><tr><td>METHOD</td><td>LEFT AcC</td><td>RIGHT AcC</td><td>AVERAGE ACC</td></tr><tr><td>MGDA</td><td>95.58</td><td>94.10</td><td>94.84</td></tr><tr><td>PCGRAD</td><td>94.26</td><td>93.60</td><td>93.93</td></tr><tr><td>CAGRAD</td><td>95.65</td><td>94.72</td><td>95.19</td></tr><tr><td>IMTL-G</td><td>94.57</td><td>94.43</td><td>94.50</td></tr><tr><td>NASH-MTL</td><td>95.33</td><td>94.38</td><td>94.86</td></tr><tr><td>FAMO</td><td>95.89</td><td>94.88</td><td>95.39</td></tr><tr><td>MOON (OURS)</td><td>95.99</td><td>95.31</td><td>95.65</td></tr></table>

Table 1: Test accuracy (%) on MultiMNIST (2 tasks). Per-task accuracies $\left( \mathrm { l e f t / r i g h t } \right)$ and their average are reported.
<table><tr><td rowspan="3">Method</td><td colspan="5">CityScapes</td><td>CelebA</td></tr><tr><td colspan="2">SEGMENTATION</td><td colspan="2">DEPTH</td><td rowspan="2"> $\Delta m \% \downarrow$ </td><td rowspan="2">∆m%↓</td></tr><tr><td>MIoU↑</td><td>Pix Acc↑</td><td>ABS ERR↓</td><td>REL ERR↓</td></tr><tr><td>STL</td><td>74.01</td><td>93.16</td><td>0.0125</td><td>27.77</td><td></td><td></td></tr><tr><td>LS</td><td>70.95</td><td>91.73</td><td>0.0161</td><td>33.83</td><td>14.11</td><td>6.28</td></tr><tr><td>RLW</td><td>74.57</td><td>93.41</td><td>0.0158</td><td>47.79</td><td>24.38</td><td>5.22</td></tr><tr><td>DWA</td><td>75.24</td><td>93.52</td><td>0.0160</td><td>44.37</td><td>21.45</td><td>6.95</td></tr><tr><td>UW</td><td>72.02</td><td>92.85</td><td>0.0140</td><td>30.13</td><td>5.89</td><td>5.78</td></tr><tr><td>MGDA</td><td>69.60</td><td>91.87</td><td>0.0134</td><td>36.05</td><td>11.02</td><td>10.93</td></tr><tr><td>PCGRAD</td><td>75.13</td><td>93.48</td><td>0.0154</td><td>42.07</td><td>18.29</td><td>6.65</td></tr><tr><td>GRADDROP</td><td>75.27</td><td>93.53</td><td>0.0157</td><td>47.54</td><td>23.73</td><td>7.80</td></tr><tr><td>CAGRAD</td><td>75.16</td><td>93.48</td><td>0.0141</td><td>37.60</td><td>11.64</td><td>6.20</td></tr><tr><td>IMTL-G</td><td>75.33</td><td>93.49</td><td>0.0135</td><td>38.41</td><td>11.10</td><td>4.67</td></tr><tr><td>NASHMTL</td><td>75.41</td><td>93.66</td><td>0.0129</td><td>35.02</td><td>6.82</td><td>4.97</td></tr><tr><td>FAMO</td><td>72.48</td><td>92.56</td><td>0.0148</td><td>29.52</td><td>6.93</td><td>4.72</td></tr><tr><td>MOON (OURS)</td><td>78.61</td><td>94.36</td><td>0.0126</td><td>31.41</td><td>1.54</td><td>4.65</td></tr></table>

![](images/ffae1466aa1fd955de52d8164448ee9b79ca41e62ee7c4dab54c92fe00a92a65.jpg)  
(a) NYU-v2

Table 3: Results on CityScapes (2 tasks) and CelebA (40 tasks) dataset. Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. Task-specific metrics and the average performance drop $\Delta m \%$ are reported.  
![](images/637fa2e1a02c1e9bb2f2704f26e2d6f25a906d343af3bda5fcc3984cebec17f9.jpg)  
(b) CityScapes  
Figure 3: Performance drop $\Delta m \%$ relative to STL vs. training epochs on NYU-v2 and CityScapes benchmarks.

From Table 2 and Table 3, we observe that our method achieves state-of-the-art performance on both benchmarks. In particular, on CityScapes, it attains the best results on most task metrics. Figure 3 shows that our method exhibits faster descent on the performance drop $\Delta m \%$ than FAMO. This demonstrates that MOON not only converges faster but also achieves better results than traditional MOO methods, supporting our theoretical claims with practical evidence.

<table><tr><td rowspan="2">METHOD</td><td rowspan="2">μ</td><td rowspan="2">α</td><td rowspan="2">€HOMO</td><td rowspan="2">€LUMO</td><td> $\langle R ^ { 2 } \rangle$ </td><td>ZPVE</td><td> $U _ { 0 }$ </td><td>U</td><td></td><td>H</td><td>G  $c _ { v }$ </td><td rowspan="2"></td><td rowspan="2">∆m%↓</td></tr><tr><td></td><td></td><td>MAE↓</td><td></td><td></td><td></td><td></td></tr><tr><td>STL</td><td>0.07</td><td>0.18</td><td>60.6</td><td>53.9</td><td>0.50</td><td>4.53</td><td>58.8</td><td>64.2</td><td>63.8</td><td>66.2</td><td>0.07</td><td></td><td></td></tr><tr><td>LS</td><td>0.11</td><td>0.33</td><td>73.6</td><td>89.7</td><td>5.20</td><td>14.06</td><td></td><td>143.4</td><td>144.2</td><td>144.6</td><td>140.3</td><td>0.13</td><td>177.6</td></tr><tr><td>SI</td><td>0.31</td><td>0.35</td><td>149.8</td><td>135.7</td><td>1.00</td><td>4.51</td><td></td><td>55.3</td><td>55.8</td><td>55.8</td><td>55.3</td><td>0.11</td><td>77.8</td></tr><tr><td>RLW</td><td>0.11</td><td>0.34</td><td>76.9</td><td>92.8</td><td>5.87</td><td>15.47</td><td></td><td>156.3</td><td>157.1</td><td>157.6</td><td>153.0</td><td>0.14</td><td>203.8</td></tr><tr><td>DWA</td><td>0.11</td><td>0.33</td><td>74.1</td><td>90.6</td><td>5.09</td><td>13.99</td><td></td><td>142.3</td><td>143.0</td><td>143.4</td><td>139.3</td><td>0.13</td><td>175.3</td></tr><tr><td>UW</td><td>0.39</td><td>0.43</td><td>166.2</td><td>155.8</td><td>1.07</td><td>4.99</td><td></td><td>66.4</td><td>66.8</td><td>66.8</td><td>66.2</td><td>0.12</td><td>108.0</td></tr><tr><td>MGDA</td><td>0.22</td><td>0.37</td><td>126.8</td><td>104.6</td><td>3.23</td><td>5.69</td><td></td><td>88.4</td><td>89.4</td><td>89.3</td><td>88.0</td><td>0.12</td><td>120.5</td></tr><tr><td>PCGRAD</td><td>0.11</td><td>0.29</td><td>75.9</td><td>88.3</td><td>3.94</td><td>9.15</td><td></td><td>116.4</td><td>116.8</td><td>117.2</td><td>114.5</td><td>0.11</td><td>125.7</td></tr><tr><td>CAGRAD</td><td>0.12</td><td>0.32</td><td>83.5</td><td>94.8</td><td>3.22</td><td>6.93</td><td></td><td>114.0</td><td>114.3</td><td>114.5</td><td>112.3</td><td>0.12</td><td>112.8</td></tr><tr><td>IMTL-G</td><td>0.14</td><td>0.29</td><td>98.3</td><td>93.9</td><td>1.75</td><td>5.70</td><td></td><td>101.4</td><td>102.4</td><td>102.0</td><td>100.1</td><td>0.10</td><td>77.2</td></tr><tr><td>NASHMTL</td><td>0.10</td><td>0.25</td><td>82.9</td><td>81.9</td><td>2.43</td><td>5.38</td><td></td><td>74.5</td><td>75.0</td><td>75.1</td><td>74.2</td><td>0.09</td><td>62.0</td></tr><tr><td>FAMO</td><td>0.16</td><td>0.28</td><td>85.0</td><td>87.0</td><td>1.81</td><td>5.13</td><td></td><td>68.2</td><td>68.5</td><td>68.7</td><td>67.8</td><td>0.09</td><td>57.3</td></tr><tr><td>MOON (OURS)</td><td>0.07</td><td>0.24</td><td>52.9</td><td>71.0</td><td>3.35</td><td>5.67</td><td></td><td>46.4</td><td>46.7</td><td>46.9</td><td>46.3</td><td>0.08</td><td>49.9</td></tr></table>

Table 4: Results on QM9 dataset (11 tasks). Each experiment is repeated over 3 random seeds and the mean is reported. The best average result is marked in bold. The task specific metrics and the average performance drop $\Delta m \%$ are reported.

## Multi-task Regression

We further evaluate MOON on multi-task regression using the QM9 molecular property prediction benchmark [Blum and Reymond, 2009]. Following standard practice, we treat predicting each of the 11 quantum-chemical properties as a separate task. We adopt a graph-level multi-task regression network, including matrix-valued parameters like edge-conditioned linear transformations in NNConv and the Set2Set internal LSTM weights [Gilmer et al., 2017], which account for 98.91% of all parameters. Details of the parameterization are provided in Appendix F. In Table 4 we report the mean absolute error (MAE) for each property. To summarize multi-objective performance, we also report the average performance drop $\Delta m \%$ relative to the STL baseline.

Table 4 shows that MOON achieves the best overall performance, attaining the lowest performance drop $\Delta m \%$ and the lowest MAE on most individual tasks. The reason is that gradient manipulation conducted in non-Euclidean spaces is more suitable for addressing matrix gradient conflicts.

## Conclusion

This work addresses the geometric mismatch between conventional Euclidean MOO methods and the matrix-valued parameters prevalent in modern architectures such as Transformers. We proposed MOON (Multi-Objective OrthoNormalized Updates), which performs gradient manipulation under spectral–nuclear norm geometry and constructs matrix-aware orthonormalized updates. At any fixed iterate, the exact MOON weighting provides a nuclear-norm-based stationarity certificate no larger than that obtained by Euclidean min-norm weighting, thereby establishing its optimality for the matrix-geometric dual subproblem. For smooth non-convex objectives, we established convergence of the averaged Pareto-stationarity measure at rates of $\mathcal { O } ( T ^ { - 1 / 2 } )$ in the deterministic setting and $\mathcal { O } ( T ^ { - 1 / 4 } )$ in the stochastic setting. Experiments demonstrate that MOON improves optimization eficiency while achieving competitive or improved final performance.

## Acknowledgment

This work was supported in part by the Beijing Major Science and Technology Project under Contract no. Z251100008125031, National Science Foundation of China (62401327), and the MOE Project of Key Research Institute of Humanities and Social Sciences (22JJD110001). This work was supported by Beijing Academy of Artificial Intelligence (BAAI).

## References

Essential AI, Ishaan Shah, Anthony M. Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, Khoi Nguyen, Kurt Smith, Michael Callahan, Michael Pust, Mohit Parmar, Peter Rushton, Platon Mazarakis, Ritvik Kapila, Saurabh Srivastava, Somanshu Singla, Tim Romanski, Yash Vanjani, and Ashish Vaswani. Practical eficiency of muon for pretraining, 2025. URL https://arxiv.org/abs/2505.02222.

Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower. The polar express: Optimal matrix sign methods and their application to the muon algorithm, 2026. URL https: //arxiv.org/abs/2505.16932.

Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoderdecoder architecture for image segmentation. IEEE transactions on pattern analysis and machine intelligence, 39(12):2481–2495, 2017.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology, 2024. URL https://arxiv.org/abs/2409.20325.

<sup>˚</sup>Ake Bj¨orck and Clazett Bowie. An iterative algorithm for computing the best estimate of an orthogonal matrix. SIAM Journal on Numerical Analysis, 8(2):358–364, 1971.

Lorenz C Blum and Jean-Louis Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database gdb-13. Journal of the American Chemical Society, 131(25): 8732–8733, 2009.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Ana Lu´ısa Cust´odio, JF Aguilar Madeira, A Ismael F Vaz, and Lu´ıs Nunes Vicente. Direct multisearch for multiobjective optimization. SIAM Journal on Optimization, 21(3):1109–1140, 2011.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https://arxiv.org/abs/2010.11929.

Heshan Devaka Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen. Mitigating gradient bias in multi-objective learning: A provably convergent approach. In The Eleventh International Conference on Learning Representations, 2023.

J¨org Fliege and Benar Fux Svaiter. Steepest descent methods for multicriteria optimization. Mathematical methods of operations research, 51(3):479–494, 2000.

Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. Pmlr, 2017.

Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842–1850. PMLR, 2018.

Yifei He, Shiji Zhou, Guojun Zhang, Hyokun Yun, Yi Xu, Belinda Zeng, Trishul Chilimbi, and Han Zhao. Robust multi-task learning with excess risks. In International Conference on Machine Learning (ICML), 2024.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https:// kellerjordan.github.io/posts/muon/.

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.

Ahmed Khaled, Kaan Ozkara, Tao Yu, Mingyi Hong, and Youngsuk Park. Muonbp: Faster muon via block-periodic orthogonalization, 2025. URL https://arxiv.org/abs/2510.16981.

Dmitry Kovalev. Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization, 2025. URL https://arxiv.org/abs/2503.12645.

Zdislav Kovarik. Some iterative methods for improving orthonormality. SIAM Journal on Numerical Analysis, 7(3):386–389, 1970.

Jiaxiang Li and Mingyi Hong. A note on the convergence of muon, 2025. URL https://arxiv.org/ abs/2502.02900.

Baijiong Lin, Feiyang Ye, and Yu Zhang. A closer look at loss weighting in multi-task learning. arXiv preprint arXiv:2111.10603, 2021.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021a.

Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36, 2024.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025. URL https://arxiv.org/abs/2502.16982.

Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum?id=IMPnRXEWpvr.

Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.

Yining Lu and Meng Jiang. Uncovering cross-objective interference in multi-objective alignment, 2026. URL https://arxiv.org/abs/2602.06869.

James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, pages 2408–2417. PMLR, 2015.

Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game, 2022. URL https://arxiv.org/abs/2202. 01017.

Sebastian Peitz and Sedjro Salomon Hotegni. Multi-objective deep learning: Taxonomy and survey of the state of the art. Machine Learning with Applications, page 100700, 2025.

Sara Sabour, Nicholas Frosst, and Geofrey E Hinton. Dynamic routing between capsules. Advances in neural information processing systems, 30, 2017.

Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018.

Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International conference on machine learning, pages 4596–4604. PMLR, 2018.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.

Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, et al. Kimi k2: Open agentic intelligence, 2026. URL https://arxiv.org/abs/2507.20534.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33: 5824–5836, 2020.

Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu. On the convergence of stochastic multi-objective gradient manipulation and beyond. Advances in Neural Information Processing Systems, 35:38103–38115, 2022.

Eckart Zitzler and Lothar Thiele. Multiobjective evolutionary algorithms: a comparative case study and the strength pareto approach. IEEE Transactions on Evolutionary Computation, 3(4): 257–271, 1999.

## Appendix

## A Missing Proofs

Throughout this appendix, we consider the idealized setting in which $\mathbf { } W _ { t }$ is the exact polar factor of the momentum matrix $M _ { t }$ , i.e.,

$$
\boldsymbol { W } _ { t } = \operatorname { P o l a r } ( \boldsymbol { M } _ { t } ) .
$$

The efect of replacing the exact polar factor with a finite-step Newton–Schulz approximation is analyzed separately in Appendix D.

## A.1 Description for Assumption

In this paper, we have the following assumptions.

Assumption 1. For each $i \in [ m ]$ , the objective function $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ is diferentiable and L-smooth with respect to the spectral norm $\| \cdot \| _ { S _ { \infty } } .$ i.e., there exist a bounded constant $L > 0$ such that for any $\Theta , \Theta ^ { \prime } \in \mathbb { R } ^ { p \times q } , \| \nabla \ell ^ { i } ( \Theta ) - \nabla \ell ^ { i } ( \Theta ^ { \prime } ) \| _ { S _ { 1 } } \le L \| \Theta - \Theta ^ { \prime } \| _ { S _ { \infty } }$

Assumption 2. There exist a constant $H > 0$ such that for any $i \in [ m ]$ and $t = 1 , \dots , T + 1$ $\| \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } \leq H$

Assumption 3. There exist a constant $B > 0$ such that for any $i \in [ m ]$ and $t = 1 , \dots , T + 1$ ， $| \ell ^ { i } ( \Theta _ { t } ) | \le B$

## A.2 Proof of Remark 2

Suppose that every $\ell ^ { i } : \Omega \subseteq \mathbb { R } ^ { p \times q } \to \mathbb { R }$ is convex, where Ω is a nonempty compact convex set, and that the spectral-norm diameter of Ω is uniformly bounded

$$
D : = \operatorname* { s u p } _ { \boldsymbol { \Theta } , \boldsymbol { \Theta } ^ { \prime } \in \Omega } \left\| \boldsymbol { \Theta } - \boldsymbol { \Theta } ^ { \prime } \right\| _ { S _ { \infty } } .
$$

For any $z \in \Delta _ { m }$ , let

$$
\Theta _ { t } ^ { * } ( z ) \in \operatorname * { a r g m i n } _ { \Theta \in \Omega } z ^ { \top } L ( \Theta ) .
$$

Convexity and spectral–nuclear norm duality imply

$$
\begin{array} { r l } & { \mathrel { \phantom { = } } z ^ { \top } L ( \Theta _ { t } ) - \displaystyle \operatorname* { m i n } _ { \Theta \in \Omega } z ^ { \top } L ( \Theta ) } \\ & { \mathrel { \phantom { = } } \le \langle \pmb { G } _ { t } ( z ) , \Theta _ { t } - \Theta _ { t } ^ { * } ( z ) \rangle } \\ & { \mathrel { \phantom { = } } \le D \left\| \pmb { G } _ { t } ( z ) \right\| _ { S _ { 1 } } . } \end{array}\tag{19}
$$

## A.3 Convergence Analysis for Deterministic Gradient

Theorem 3 (Deterministic Convergence). Suppose that each objective $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { F }$ is L-smooth with respect to the spectral norm. Assume that there exist constants $H > 0$ and $B > 0$ , independent of both T and t, such that $\begin{array} { r } { \operatorname* { m a x } _ { t \in [ T + 1 ] } \operatorname* { m a x } _ { i \in [ m ] } \| \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } \leq H } \end{array}$ and $\operatorname* { m a x } _ { t \in [ T + 1 ] } \operatorname* { m a x } _ { i \in [ m ] } | \ell ^ { i } ( \Theta _ { t } ) | \leq B$ Choose $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma < m H T$ . Then, for every fixed $\mu \in ( 0 , 1 ]$ , the iterates generated by Algorithm 1 satisfy

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \leq O ( T ^ { - 1 / 2 } ) .
$$

Before the final proof, we first introduce the following Lemmas.

Lemma 1. Assume that Assumption 2 holds and that $0 < \beta \gamma < 1$ . Then for any $t = 1 , \dots , T$ , the weight updates in Algorithm 1 satisfy the following inequality

$$
\| z _ { t } - z _ { t + 1 } \| _ { \infty } \leq \beta H .\tag{20}
$$

Proof. The softmax mapping is $1 / 2 \ – \mathrm { I }$ ipschitz with respect to the $\ell _ { \infty }$ norm. Indeed, for any $\pmb { x } , \pmb { y } \in \mathbb { R } ^ { m }$

$$
\| \mathrm { s o f t m a x } ( { \pmb x } ) - \mathrm { s o f t m a x } ( { \pmb y } ) \| _ { \infty } \leq \frac { 1 } { 2 } \| { \pmb x } - { \pmb y } \| _ { \infty } .
$$

Consequently,

$$
\begin{array} { r } { \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \displaystyle \frac { 1 } { 2 } \| \pmb { \xi } _ { t + 1 } - \pmb { \xi } _ { t } \| _ { \infty } } \\ { = \displaystyle \frac { \beta } { 2 } \| \pmb { \delta } _ { t } + \gamma \pmb { \xi } _ { t } \| _ { \infty } . } \end{array}\tag{21}
$$

We first bound $\delta _ { t }$ . By the duality between the spectral norm and the nuclear norm,

$$
\begin{array} { r l r } {  { \| \pmb { \delta } _ { t } \| _ { \infty } = \operatorname* { m a x } _ { i \in [ m ] } \big | \big \langle \pmb { W } _ { t } , \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) \big \rangle \big | } } \\ & { } & { \leq \| \pmb { W } _ { t } \| _ { S _ { \infty } } \operatorname* { m a x } _ { i \in [ m ] } \| \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) \| _ { S _ { 1 } } \leq H . } \end{array}\tag{22}
$$

Here, $\| \boldsymbol { W } _ { t } \| _ { S _ { \infty } } \leq 1$ follows from the polar-factor construction in Algorithm 1. In particular, if

$$
\pmb { W } _ { t } = \pmb { U } _ { t } \pmb { V } _ { t } ^ { \top }
$$

is obtained from a compact singular value decomposition, then $\| \boldsymbol { W } _ { t } \| _ { S _ { \infty } } \leq 1$

Next, the update of $\xi _ { t }$ can be written as

$$
\pmb { \xi } _ { t + 1 } = ( 1 - \beta \gamma ) \pmb { \xi } _ { t } - \beta \pmb { \delta } _ { t } .
$$

Since $0 < \beta \gamma < 1$ , the coeficient $1 - \beta \gamma$ is nonnegative. Thus, using (22),

$$
\begin{array} { r } { \| \pmb { \xi } _ { t + 1 } \| _ { \infty } \leq ( 1 - \beta \gamma ) \| \pmb { \xi } _ { t } \| _ { \infty } + \beta \| \pmb { \delta } _ { t } \| _ { \infty } } \\ { \leq ( 1 - \beta \gamma ) \| \pmb { \xi } _ { t } \| _ { \infty } + \beta H . } \end{array}\tag{23}
$$

Unrolling (23) and using ${ \boldsymbol { \xi } } _ { 1 } = \mathbf { 0 }$ gives

$$
\begin{array} { c } { \displaystyle | | \xi _ { t } | | _ { \infty } \leq \beta H \sum _ { s = 0 } ^ { t - 2 } ( 1 - \beta \gamma ) ^ { s } } \\ { \displaystyle = \frac { H } { \gamma } \left[ 1 - ( 1 - \beta \gamma ) ^ { t - 1 } \right] \leq \frac { H } { \gamma } . } \end{array}\tag{24}
$$

Since $z _ { t } = \mathrm { S o f t m a x } ( \pmb { \xi } _ { t } )$ , the 1/2-Lipschitz continuity of softmax under the $\ell _ { \infty }$ norm bounds the change in $z _ { t }$ by the corresponding change in the logits. Applying the triangle inequality to the logit update and substituting $\| \delta _ { t } \| _ { \infty } \leq H$ from (22) and $\| \pmb { \xi } _ { t } \| _ { \infty } \leq H / \gamma$ from (24), we obtain

$$
\begin{array} { r l } {  { \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \frac { \beta } { 2 } ( \| \pmb { \delta } _ { t } \| _ { \infty } + \gamma \| \pmb { \xi } _ { t } \| _ { \infty } ) } } \\ & { \leq \displaystyle \frac { \beta } { 2 } ( H + H ) = \beta H . } \end{array}
$$

This completes the proof.

Lemma 2. Assume that Assumption 1 and 2 hold, and that $0 < \beta \gamma < 1$ . Then for any $t = 1 , \dots , T$ the weight updates in Algorithm 1 satisfy the following inequality

$$
\begin{array} { l } { \displaystyle \| M _ { t } - \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } } \\ { \displaystyle \le ( 1 - \mu ) ^ { t - 1 } \| \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell ^ { i } ( \Theta _ { 1 } ) \| _ { S _ { 1 } } + \frac { L \alpha } \mu + \frac { m H ^ { 2 } } \mu \beta . } \end{array}\tag{25}
$$

Proof. Let

$$
\pmb { G } _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \pmb { \Theta } _ { t } ) \quad \mathrm { a n d } \quad E _ { t } : = M _ { t } - \pmb { G } _ { t } .
$$

Using the momentum update $M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu G _ { t } ,$ , we obtain, for $t \geq 2$

$$
\mathbf { } E _ { t } = ( 1 - \mu ) \bigl ( \mathbf { } E _ { t - 1 } + \mathbf { } G _ { t - 1 } - \mathbf { } G _ { t } \bigr ) .
$$

Iterating this recursion and using $M _ { 0 } = 0$ gives

$$
\| \pmb { E } _ { t } \| _ { S _ { 1 } } \leq ( 1 - \mu ) ^ { t } \| \pmb { G } _ { 1 } \| _ { S _ { 1 } } + \sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } \| \pmb { G } _ { j - 1 } - \pmb { G } _ { j } \| _ { S _ { 1 } } .
$$

Moreover,

$$
\begin{array} { l } { { \displaystyle { \cal G } _ { j } - { \cal G } _ { j - 1 } = \sum _ { i = 1 } ^ { m } z _ { i , j } \big ( \nabla \ell _ { i } ( \Theta _ { j } ) - \nabla \ell _ { i } ( \Theta _ { j - 1 } ) \big ) } } \\ { { \displaystyle ~ + \sum _ { i = 1 } ^ { m } ( z _ { i , j } - z _ { i , j - 1 } ) \nabla \ell _ { i } ( \Theta _ { j - 1 } ) . } } \end{array}
$$

By Assumptions 1–2, Lemma 1, and $\| \Theta _ { j } - \Theta _ { j - 1 } \| _ { S _ { \infty } } \leq \alpha$ , we have

$$
\| \boldsymbol { G } _ { j } - \boldsymbol { G } _ { j - 1 } \| _ { S _ { 1 } } \leq L \alpha + m H ^ { 2 } \beta .
$$

Since

$$
\sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } \leq \frac { 1 } { \mu } ,
$$

it follows that

$$
\left. { M } _ { t } - \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \Theta _ { t } ) \right. _ { S _ { 1 } }
$$

$$
\leq ( 1 - \mu ) ^ { t - 1 } \left\| \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell _ { i } ( \Theta _ { 1 } ) \right\| _ { S _ { 1 } } + \frac { L \alpha } { \mu } + \frac { m H ^ { 2 } \beta } { \mu } .
$$

We are now ready to prove Theorem 3.

Theorem 4. Suppose Assumption 1–3 hold, let $\ell ^ { i } : \mathbb { R } ^ { p \times q } $ R be non-convex for each $i \in [ m ]$ Suppose that $0 < \beta \gamma < 1$ , then the iterations of Algorithm 1 satisfy the following inequality

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } } \\ & { \displaystyle \leq \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + \frac { 2 L \alpha } { \mu } + \frac { 2 m H ^ { 2 } } { \mu } \beta + m H B \frac { \beta } { \alpha } + \frac { L } { 2 } \alpha . } \end{array}\tag{26}
$$

Proof. Define

$$
G _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \Theta _ { t } ) .
$$

By L-smoothness and $\Theta _ { t + 1 } = \Theta _ { t } - \alpha W _ { t }$ ,

$$
\sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \Theta _ { t + 1 } ) \leq \sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \Theta _ { t } ) - \alpha \langle \pmb { G } _ { t } , \pmb { W } _ { t } \rangle + \frac { L \alpha ^ { 2 } } { 2 } .
$$

Since $\mathbf { } W _ { t }$ is the polar factor of $M _ { t } , \langle M _ { t } , W _ { t } \rangle = \lVert M _ { t } \rVert _ { S _ { 1 } }$ and $\| \boldsymbol { W } _ { t } \| _ { S _ { \infty } } \leq 1$ . Therefore,

$$
- \alpha \langle \boldsymbol { G } _ { t } , \boldsymbol { W } _ { t } \rangle \le - \alpha \| \boldsymbol { G } _ { t } \| _ { S _ { 1 } } + 2 \alpha \| \boldsymbol { G } _ { t } - \boldsymbol { M } _ { t } \| _ { S _ { 1 } } .
$$

After adding and subtracting $\begin{array} { r } { \sum _ { i } z _ { i , t + 1 } \ell _ { i } ( \Theta _ { t + 1 } ) } \end{array}$ and rearranging, we obtain

$$
\begin{array} { r l } { \displaystyle \| \pmb { G } _ { t } \| _ { S _ { 1 } } \leq \frac { 1 } { \alpha } \left( \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \pmb { \Theta } _ { t } ) - \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t + 1 } \ell _ { i } ( \pmb { \Theta } _ { t + 1 } ) \right) } & { } \\ { + \frac { 1 } { \alpha } \displaystyle \sum _ { i = 1 } ^ { m } ( z _ { i , t + 1 } - z _ { i , t } ) \ell _ { i } ( \pmb { \Theta } _ { t + 1 } ) + \frac { L \alpha } { 2 } } & { } \\ { + 2 \| \pmb { G } _ { t } - \boldsymbol { M } _ { t } \| _ { S _ { 1 } } . } \end{array}
$$

Lemma 1 imply

$$
\left. \sum _ { i = 1 } ^ { m } ( z _ { i , t + 1 } - z _ { i , t } ) \ell _ { i } ( \Theta _ { t + 1 } ) \right. \leq m B H \beta .
$$

Applying Lemma 2, summing over $t = 1 , \dots , T$ , and using $\| G _ { 1 } \| _ { S _ { 1 } } \leq H$ , we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \| \pmb { G } _ { t } \| _ { S _ { 1 } } \leq \frac { 2 B } { \alpha } + \frac { 2 H } { \mu } + \frac { 2 T L \alpha } { \mu } + \frac { 2 T m H ^ { 2 } \beta } { \mu } } \\ { \displaystyle + \frac { T m H B \beta } { \alpha } + \frac { T L \alpha } { 2 } . } \end{array}
$$

Finally, since $z _ { t } \in \Delta _ { m }$ , it is a feasible point of the minimization problem. Therefore,

$$
\operatorname* { m i n } _ { z \in \Delta _ { m } } \left. \sum _ { i = 1 } ^ { m } z _ { i } \nabla \ell _ { i } ( \Theta _ { t } ) \right. _ { S _ { 1 } } \leq \lVert \pmb { G } _ { t } \rVert _ { S _ { 1 } } .
$$

Dividing the preceding inequality by T yields the claimed result.

There is an immediate corollary of the above Theorem.

Corollary 1. Setting $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma < m H T$ in Theorem $^ { 4 , }$ then, for any fixed $\mu \in ( 0 , 1 ]$ , the following holds

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { \boldsymbol { z } _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) \boldsymbol { z } _ { t } ^ { * } \| _ { S _ { 1 } } } \\ & { \displaystyle \leq O ( \sqrt { \frac { L B } { T } } + \frac { 1 } { T } ) = O ( 1 / \sqrt { T } ) . } \end{array}\tag{27}
$$

Proof. Since $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma < m H T$ in Theorem $^ { 4 , }$ we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \displaystyle \operatorname* { m i n } _ { \boldsymbol { z } _ { t } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) \boldsymbol { z } _ { t } ^ { * } \| _ { s _ { 1 } } } \\ & { \le \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + \frac { 2 L \alpha } { \mu } + \frac { 2 m H ^ { 2 } } { \mu } \beta + m H B \frac { \beta } { \alpha } + \frac { L } { 2 } \alpha } \\ & { = ( \frac { 7 } { 2 } + \frac { 2 } { \mu } ) \sqrt { \frac { L B } { T } } + \frac { 4 H } { \mu } \cdot \frac { 1 } { T } } \\ & { = O ( \sqrt { \frac { L B } { T } } + \frac { 1 } { T } ) = O ( 1 / \sqrt { T } ) . } \end{array}
$$

## A.4 Convergence Analysis for Stochastic Gradient

Theorem 5 (Stochastic Convergence). Under the same assumption as Theorem 1. We assume that the stochastic gradient estimator is unbiased and has variance bounded by $\sigma .$ . Choosing $\begin{array} { r } { \alpha = \sqrt { \frac { \mu B } { T L } } } \end{array}$ 2 $\begin{array} { r } { \beta = \frac { 1 } { ( H + \sqrt { m } \rho \sigma ) m T } , 0 < \gamma < ( H + \sqrt { m } \rho \sigma ) m T } \end{array}$ and $\begin{array} { r } { \mu = \operatorname* { m i n } \{ \frac { \sqrt { L B } } { \sigma \rho \sqrt { T } } , 1 \} } \end{array}$ , where $\rho : = \sqrt { \operatorname* { m i n } \{ p , q \} }$ , then the iterations of Algorithm 1 satisfy

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \right] \leq \mathcal { O } ( 1 / T ^ { 1 / 4 } ) .\tag{28}
$$

We now establish the convergence of MOON in the stochastic setting. Let $\mathcal { F } _ { t }$ denote the sigmaalgebra containing all information available immediately before sampling $\zeta _ { t }$ . In particular, $\Theta _ { t }$ is $\mathcal { F } _ { t } .$ -measurable. For every $i \in [ m ]$ and $t \geq 1$ , assume that, almost surely,

$$
\mathbb { E } _ { t } \left[ g ^ { i } ( \Theta _ { t } ; \zeta _ { t } ) \right] = \nabla \ell ^ { i } ( \Theta _ { t } ) ,
$$

and

$$
\mathbb { E } _ { t } \left[ \left. g ^ { i } ( \Theta _ { t } ; \zeta _ { t } ) - \nabla \ell ^ { i } ( \Theta _ { t } ) \right. _ { F } ^ { 2 } \right] \leq \sigma ^ { 2 } ,
$$

where $\mathbb { E } _ { t } [ \cdot ] : = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ and $\sigma > 0$ is independent of t and $T .$

No independence is required among the stochastic-gradient errors associated with diferent objectives at the same iteration. Therefore, the stochastic gradients may be computed using the same sample $\zeta _ { t }$ . Independence across iterations is not required either, provided that the above conditional assumptions hold.

Lemma 3. Assume that Assumption 2 holds, and that $0 < \beta \gamma < 1$ . Then for any $t = 1 , \dots , T$ , the weight updates in Algorithm 1 satisfy the following inequality

$$
\begin{array} { r } { \mathbb { E } \left[ \| z _ { t } - z _ { t + 1 } \| _ { \infty } \right] \leq \beta ( H + \sqrt { m } \rho \sigma ) . } \end{array}\tag{29}
$$

Proof. Let

$$
\pmb { \eta } _ { i , t } : = g ^ { i } ( \pmb { \Theta } _ { t } ; \zeta _ { t } ) - \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } )
$$

denote the stochastic-gradient error.

The softmax mapping is 1/2-Lipschitz with respect to the $\ell _ { \infty } { \mathrm { - n o r m } }$ . Therefore,

$$
\begin{array} { r } { \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \frac { 1 } { 2 } \| \pmb { \xi } _ { t + 1 } - \pmb { \xi } _ { t } \| _ { \infty } } \\ { = \displaystyle \frac { \beta } { 2 } \| \pmb { \delta } _ { t } + \gamma \pmb { \xi } _ { t } \| _ { \infty } . } \end{array}\tag{30}
$$

We first bound $\mathbb { E } \| \delta _ { t } \| _ { \infty }$ . By the duality between the spectral and nuclear norms and the almostsure bound $\| \boldsymbol { W } _ { t } \| _ { S _ { \infty } } \leq 1$ 2

$$
\begin{array} { r l } & { \| \delta _ { t } \| _ { \infty } = \displaystyle \operatorname* { m a x } _ { i \in [ m ] } \left| \left. W _ { t } , g ^ { i } ( \Theta _ { t } ; \zeta _ { t } ) \right. \right| } \\ & { \qquad \leq \displaystyle \operatorname* { m a x } _ { i \in [ m ] } \left\| g ^ { i } ( \Theta _ { t } ; \zeta _ { t } ) \right\| _ { S _ { 1 } } } \\ & { \qquad \leq \displaystyle \operatorname* { m a x } _ { i \in [ m ] } \| \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } + \displaystyle \operatorname* { m a x } _ { i \in [ m ] } \| \eta _ { i , t } \| _ { S _ { 1 } } } \\ & { \qquad \leq H + \displaystyle \operatorname* { m a x } _ { i \in [ m ] } \| \eta _ { i , t } \| _ { S _ { 1 } } . } \end{array}\tag{31}
$$

For every matrix $A \in \mathbb { R } ^ { p \times q }$ 2

$$
\begin{array} { r } { \| \pmb { A } \| _ { S _ { 1 } } \le \sqrt { \operatorname* { m i n } \{ p , q \} } \| \pmb { A } \| _ { F } = \rho \| \pmb { A } \| _ { F } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } { \mathbb { E } _ { t } \left[ \underset { i \in [ m ] } { \operatorname* { m a x } } \left. \eta _ { i , t } \right. _ { S _ { 1 } } \right] \leq \rho \mathbb { E } _ { t } \left[ \underset { i \in [ m ] } { \operatorname* { m a x } } \left. \eta _ { i , t } \right. _ { F } \right] } & { } \\ & { \leq \rho \mathbb { E } _ { t } \left[ \left( \displaystyle \sum _ { i = 1 } ^ { m } \left. \eta _ { i , t } \right. _ { F } ^ { 2 } \right) ^ { 1 / 2 } \right] } \\ & { \leq \rho \left( \displaystyle \sum _ { i = 1 } ^ { m } \mathbb { E } _ { t } \left. \eta _ { i , t } \right. _ { F } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \leq \rho \sqrt { m } \sigma . } \end{array}\tag{32}
$$

Moreover, by L-smoothness and $\Theta _ { t + 1 } = \Theta _ { t } - \alpha W _ { t }$ , the loss-diference approximation satisfies

$$
\left| \frac { \ell ^ { i } ( \Theta _ { t } ) - \ell ^ { i } ( \Theta _ { t + 1 } ) } { \alpha } - \langle \nabla \ell ^ { i } ( \Theta _ { t } ) , W _ { t } \rangle \right| \leq \frac { L \alpha } { 2 } \| W _ { t } \| _ { S _ { \infty } } ^ { 2 } \leq \frac { L \alpha } { 2 } ,
$$

where the last inequality follows from $\| W _ { t } \| _ { S _ { \infty } } \leq 1$

The penultimate inequality follows from Jensen’s inequality. Combining (31) and (32), and then taking total expectation, gives

$$
\mathbb { E } \| \pmb { \delta } _ { t } \| _ { \infty } \le H + \rho \sqrt { m } \sigma .\tag{33}
$$

For brevity, define

$$
C _ { \sigma } : = H + \rho \sqrt { m } \sigma .
$$

The dual-logit update can be written as

$$
\pmb { \xi } _ { t + 1 } = ( 1 - \beta \gamma ) \pmb { \xi } _ { t } - \beta \pmb { \delta } _ { t } .
$$

Since $0 < \beta \gamma < 1$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \| \pmb { \xi } _ { t + 1 } \| \infty \leq ( 1 - \beta \gamma ) \mathbb { E } \| \pmb { \xi } _ { t } \| _ { \infty } + \beta \mathbb { E } \| \pmb { \delta } _ { t } \| _ { \infty } } \\ & { \qquad \leq ( 1 - \beta \gamma ) \mathbb { E } \| \pmb { \xi } _ { t } \| _ { \infty } + \beta C _ { \sigma } . } \end{array}\tag{34}
$$

Unrolling (34) and using ${ \boldsymbol { \xi } } _ { 1 } = \mathbf { 0 }$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| \pmb { \xi } _ { t } \| _ { \infty } \leq \beta C _ { \sigma } \displaystyle \sum _ { s = 0 } ^ { t - 2 } ( 1 - \beta \gamma ) ^ { s } } \\ { \displaystyle = \frac { C _ { \sigma } } \gamma \left[ 1 - ( 1 - \beta \gamma ) ^ { t - 1 } \right] \leq \frac { C _ { \sigma } } \gamma . } \end{array}\tag{35}
$$

Finally, taking expectations in (30) and applying the triangle inequality separate the contributions of $\delta _ { t }$ and $\xi _ { t } . \ \mathrm { B y } \ ( 3 3 )$ and (35), these two contributions are bounded by $C _ { \sigma }$ and $\gamma ( C _ { \sigma } / \gamma ) = C _ { \sigma }$ respectively. Therefore, we obtain

$$
\begin{array} { r l } & { \mathbb { E } \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \displaystyle \frac { \beta } { 2 } \left( \mathbb { E } \| \pmb { \delta } _ { t } \| _ { \infty } + \gamma \mathbb { E } \| \pmb { \xi } _ { t } \| _ { \infty } \right) } \\ & { \qquad \leq \displaystyle \frac { \beta } { 2 } ( C _ { \sigma } + C _ { \sigma } ) } \\ & { \qquad = \beta \left( H + \rho \sqrt { m } \sigma \right) . } \end{array}
$$

This completes the proof.

Lemma 4. Assume that Assumptions 1 and 2 hold. Suppose that $M _ { 0 } = \mathbf { 0 } , 0 < \mu \leq 1$ , and $0 < \beta \gamma < 1$ . Then, for any $t = 1 , \dots , T$ , the iterates generated by Algorithm 1 satisfy

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left\| M _ { t } - \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \right\| _ { S _ { 1 } } \right] } \\ & { \leq ( 1 - \mu ) ^ { t - 1 } \left\| \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell ^ { i } ( \Theta _ { 1 } ) \right\| _ { S _ { 1 } } + \frac { L \alpha } { \mu } } \\ & { + \sqrt { \mu } \rho \sigma + \displaystyle \frac { \left( H + \sqrt { m } \rho \sigma \right) m H } { \mu } \beta . } \end{array}\tag{36}
$$

Proof. Define the weighted true gradient and the aggregated stochastic-gradient noise as

$$
\begin{array} { l } { \displaystyle { G _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \boldsymbol { \Theta } _ { t } ) , } } \\ { \displaystyle { \ \epsilon _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \left( g ^ { i } ( \boldsymbol { \Theta } _ { t } ; \boldsymbol { \zeta } _ { t } ) - \nabla \ell ^ { i } ( \boldsymbol { \Theta } _ { t } ) \right) . } } \end{array}
$$

The momentum update can then be written as

$$
M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu ( G _ { t } + \epsilon _ { t } ) .
$$

Unrolling this recursion gives

$$
M _ { t } = ( 1 - \mu ) ^ { t } M _ { 0 } + \mu \sum _ { j = 1 } ^ { t } ( 1 - \mu ) ^ { t - j } \left( G _ { j } + \epsilon _ { j } \right) .
$$

Subtracting $G _ { t }$ and rearranging the deterministic terms yields

$$
\begin{array} { l } { { { \cal M } _ { t } - { \cal G } _ { t } = ( 1 - \mu ) ^ { t } ( { \cal M } _ { 0 } - { \cal G } _ { 1 } ) } } \\ { { \displaystyle ~ + \sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } ( { \cal G } _ { j - 1 } - { \cal G } _ { j } ) } } \\ { { \displaystyle ~ + \mu \sum _ { j = 1 } ^ { t } ( 1 - \mu ) ^ { t - j } \epsilon _ { j } . } } \end{array}
$$

We first bound the variation of the weighted true gradient. For every $j \geq 2$

$$
\begin{array} { r } { G _ { j } - \pmb { G } _ { j - 1 } = \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , j } \left( \nabla \ell ^ { i } ( \pmb { \Theta } _ { j } ) - \nabla \ell ^ { i } ( \pmb { \Theta } _ { j - 1 } ) \right) } \\ { + \displaystyle \sum _ { i = 1 } ^ { m } ( z _ { i , j } - z _ { i , j - 1 } ) \nabla \ell ^ { i } ( \pmb { \Theta } _ { j - 1 } ) . } \end{array}
$$

By Assumptions 1 and $2 , \| \pmb { W } _ { j - 1 } \| _ { S _ { \infty } } \le 1$ , and the stochastic weight-change bound

$$
\begin{array} { r } { \mathbb { E } \left[ \| z _ { j } - z _ { j - 1 } \| _ { \infty } \right] \leq \beta ( H + \sqrt { m } \rho \sigma ) , } \end{array}
$$

we obtain

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \| \pmb { G } _ { j } - \pmb { G } _ { j - 1 } \| _ { S _ { 1 } } \right] \leq L \mathbb { E } \left[ \| \pmb { \Theta } _ { j } - \pmb { \Theta } _ { j - 1 } \| _ { S _ { \infty } } \right] } & { } \\ { \displaystyle } & { + H \sum _ { i = 1 } ^ { m } \mathbb { E } \left[ | z _ { i , j } - z _ { i , j - 1 } | \right] } \\ { \displaystyle } & { \leq L \alpha + ( H + \sqrt { m } \rho \sigma ) m H \beta . } \end{array}
$$

We next bound the stochastic-noise term. Let ${ \mathcal { F } } _ { j }$ denote the history before sampling $\zeta _ { j }$ . Since $z _ { j }$ and $\Theta _ { j }$ are ${ \mathcal { F } } _ { j }$ -measurable, the conditional unbiasedness assumption gives

$$
\mathbb { E } [ \epsilon _ { j } \mid \mathcal { F } _ { j } ] = \mathbf { 0 } .
$$

Moreover, by the convexity of the squared Frobenius norm,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| \pmb { \epsilon } _ { j } \| _ { F } ^ { 2 } \mid \mathcal { F } _ { j } \right] } \\ & { = \mathbb { E } \left[ \left\| \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , j } \left( g ^ { i } ( \Theta _ { j } ; \zeta _ { j } ) - \nabla \ell ^ { i } ( \Theta _ { j } ) \right) \right\| _ { F } ^ { 2 } \Bigg | \mathcal { F } _ { j } \right] } \end{array}
$$

$$
\leq \sum _ { i = 1 } ^ { m } z _ { i , j } \mathbb { E } \left[ \left\| g ^ { i } ( \Theta _ { j } ; \zeta _ { j } ) - \nabla \ell ^ { i } ( \Theta _ { j } ) \right\| _ { F } ^ { 2 } \middle | \mathcal { F } _ { j } \right] \leq \sigma ^ { 2 } .
$$

Because $\{ \epsilon _ { j } \} _ { j \ge 1 }$ is a martingale-diference sequence, the cross terms vanish. Using $\| A \| _ { S _ { 1 } } \leq \rho \| A \| _ { F }$ and Jensen’s inequality, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \displaystyle \mu \sum _ { j = 1 } ^ { t } ( 1 - \mu ) ^ { t - j } \epsilon _ { j } \right. _ { S _ { 1 } } \right] } \\ & { \leq \rho \sqrt { \mathbb { E } \left[ \left. \displaystyle \mu \sum _ { j = 1 } ^ { t } ( 1 - \mu ) ^ { t - j } \epsilon _ { j } \right. _ { F } ^ { 2 } \right] } } \\ & { \leq \rho \mu \sigma \sqrt { \displaystyle \sum _ { j = 1 } ^ { t } ( 1 - \mu ) ^ { 2 ( t - j ) } } \leq \sqrt { \mu } \rho \sigma . } \end{array}
$$

Applying the triangle inequality to the preceding decomposition, using $M _ { 0 } = \mathbf { 0 }$ , and taking total expectation separate the momentum error into three parts: the initial error, the accumulated variation of the weighted true gradients, and the accumulated stochastic-gradient noise. The weighted-gradient variation at each step is bounded by $L \alpha + ( H + \sqrt { m } \rho \sigma ) m H \beta$ , while the expected nuclear norm of the accumulated noise is bounded by $\sqrt { \mu } \rho \sigma$ . Therefore, we obtain

$$
\begin{array} { r l } {  { \mathbb { E } [ \| M _ { t } - { \pmb { G } } _ { t } \| _ { S _ { 1 } } ] \le ( 1 - \mu ) ^ { t } \| { \pmb { G } } _ { 1 } \| _ { S _ { 1 } } + \sqrt { \mu } \rho \sigma } } \\ & { + \displaystyle \sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } ( L \alpha + ( H + \sqrt { m } \rho \sigma ) m H \beta ) . } \end{array}
$$

Finally, since

$$
( 1 - \mu ) ^ { t } \leq ( 1 - \mu ) ^ { t - 1 }
$$

and

$$
\sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } \leq \frac { 1 } { \mu } ,
$$

we conclude that

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \left[ \| M _ { t } - { \pmb { G } } _ { t } \| _ { S _ { 1 } } \right] \leq ( 1 - \mu ) ^ { t - 1 } \| { \pmb { G } } _ { 1 } \| _ { S _ { 1 } } + \frac { L \alpha } { \mu } } \\ & { \displaystyle + \sqrt { \mu } \rho \sigma + \frac { ( H + \sqrt { m } \rho \sigma ) m H } { \mu } \beta . } \end{array}
$$

Substituting the definition of $G _ { t }$ completes the proof.

Now we are ready to prove Theorem 5

Theorem 6. Let Assumption 1 to 3 hold, let $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ be non-convex for each $i \in [ m ]$ . Then the iterations of Algorithm 1 satisfy the following inequality

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \right] } \\ & { \displaystyle \leq \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + ( H + \sqrt { m } \rho \sigma ) m B \frac { \beta } { \alpha } + \frac { L } { 2 } \alpha } \\ & { \displaystyle + \frac { 2 L \alpha } { \mu } + 2 \sqrt { \mu } \rho \sigma + \frac { 2 m H ( H + \sqrt { m } \rho \sigma ) } { \mu } \beta . } \end{array}\tag{37}
$$

Proof. For brevity, define

$$
G _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) .
$$

From the L-smoothness of each objective function, we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t + 1 } ) \leq \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t } ) + \frac { L } { 2 } \| \Theta _ { t + 1 } - \Theta _ { t } \| _ { S _ { \infty } } ^ { 2 } } \\ & { \quad + \left. \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) , \Theta _ { t + 1 } - \Theta _ { t } \right. } \\ & { \displaystyle \leq \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t } ) + \frac { L } { 2 } \| \Theta _ { t + 1 } - \Theta _ { t } \| _ { S _ { \infty } } ^ { 2 } + \langle M _ { t } , \Theta _ { t + 1 } - \Theta _ { t } \rangle } \\ & { \quad + \left. \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) - M _ { t } , \Theta _ { t + 1 } - \Theta _ { t } \right. . } \end{array}
$$

Since $\Theta _ { t + 1 } = \Theta _ { t } - \alpha W _ { t }$ , we have

$$
\begin{array} { r l } & { \langle M _ { t } , \Theta _ { t + 1 } - \Theta _ { t } \rangle } \\ & { = \langle M _ { t } , - \alpha W _ { t } \rangle = - \alpha \| M _ { t } \| _ { S _ { 1 } } . } \end{array}
$$

The spectral-nuclear norm duality bounds the last inner product by

$$
\left. \boldsymbol { G } _ { t } - \boldsymbol { M } _ { t } \right. _ { S _ { 1 } } \left. \boldsymbol { \Theta } _ { t + 1 } - \boldsymbol { \Theta } _ { t } \right. _ { S _ { \infty } } .
$$

Moreover, $\| \Theta _ { t + 1 } - \Theta _ { t } \| _ { S _ { \infty } } = \alpha \| W _ { t } \| _ { S _ { \infty } } \leq \alpha$ , and the reverse triangle inequality gives

$$
\lVert M _ { t } \rVert _ { S _ { 1 } } \geq \lVert { G _ { t } } \rVert _ { S _ { 1 } } - \lVert { G _ { t } } - M _ { t } \rVert _ { S _ { 1 } } .
$$

Substituting these two bounds into the smoothness inequality yields

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t + 1 } ) \leq \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t } ) + \displaystyle \frac { L } { 2 } \alpha ^ { 2 } - \alpha \| M _ { t } \| _ { S _ { 1 } } } \\ { \displaystyle + \| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) - M _ { t } \| _ { S _ { 1 } } \| \Theta _ { t + 1 } - \Theta _ { t } \| _ { S _ { \infty } } } \\ { \displaystyle \leq \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell ^ { i } ( \Theta _ { t } ) + \displaystyle \frac { L } { 2 } \alpha ^ { 2 } - \alpha \left\| \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \right\| _ { S _ { 1 } } } \\ { \displaystyle + 2 \alpha \left\| \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) - M _ { t } \right\| _ { S _ { 1 } } . } \end{array}
$$

Rearranging the preceding inequality, adding and subtracting $\scriptstyle \sum _ { i = 1 } ^ { m } z _ { i , t + 1 } \ell ^ { i } ( \Theta _ { t + 1 } )$ to create a telescoping term, and taking total expectation gives the first inequality below. For the second inequality, we use

$$
\begin{array} { r l } {  { \mathbb { E } [  \sum _ { i = 1 } ^ { m } ( z _ { i , t + 1 } - z _ { i , t } ) \ell ^ { i } ( \Theta _ { t + 1 } )  ] } } \\ & { \leq m B \mathbb { E } [ \| z _ { t + 1 } - z _ { t } \| _ { \infty } ] \leq m B \beta ( H + \sqrt { m } \rho \sigma ) , } \end{array}
$$

together with Lemma 4. Therefore,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \displaystyle \sum _ { s = 2 } ^ { n } g _ { s } ( \theta _ { s } ) \right. _ { s , s } \right] } \\ & { \leq \alpha ^ { \frac { 1 } { 2 } } \operatorname* { l i m } _ { s \in \mathcal { S } _ { \varepsilon } } \left\{ \frac { \sum _ { s = 2 } ^ { n } g _ { s } ( \theta _ { s } ) } { \sin x } \right\} \Big | _ { s = 0 } ^ { s } } \\ & { \quad + \frac { 1 } { \alpha ^ { \frac { 1 } { 2 } } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } } \\ & { \quad + 2 \alpha ^ { \frac { 1 } { 2 } } \left. \displaystyle \sum _ { s = 2 } ^ { n } g _ { s } ( \theta _ { s } ) - \lambda \right. _ { s } ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } } \\ & { \quad + \frac { 1 } { \alpha ^ { \frac { 1 } { 2 } } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } } \\ &  \quad + \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } } \alpha ^ { \frac { 1 } { 2 } }  \end{array}
$$

Summing the preceding inequality from $t = 1$ to $T ,$ the weighted-loss terms telescope. Since $z _ { t } \in \Delta _ { m }$ and $| \ell ^ { i } ( \Theta _ { t } ) | \le B$ , the two endpoint terms are bounded by 2B. Moreover,

$$
\sum _ { t = 1 } ^ { T } ( 1 - \mu ) ^ { t - 1 } \leq \frac { 1 } { \mu } , \qquad \left\| \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell ^ { i } ( \Theta _ { 1 } ) \right\| _ { S _ { 1 } } \leq H .
$$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { T } \mathbb { E } \left[ \left\| \sum _ { s = 1 } ^ { \infty } z _ { k } \nabla ^ { \xi } ( \Theta _ { s } ) \right\| _ { s } \right] } \\ & { \le \frac { 1 } { \alpha } \mathbb { E } \left[ \sum _ { i = 1 } ^ { \infty } z _ { i + 1 } \ell ^ { \xi } ( \Theta _ { s } ) - \displaystyle \sum _ { i = 1 } ^ { \infty } z _ { i } z _ { 1 } + \ell ^ { \xi } ( \Theta _ { T + 1 } ) \right] } \\ & { \displaystyle + \frac { 2 H } { \mu } + ( I + \sqrt { m } \rho \sigma ) T m B _ { \alpha } ^ { \beta } + \frac { T L } { 2 } \alpha + \frac { 2 T L \alpha } { \mu } } \\ & { \displaystyle + 2 T \sqrt { \mu } \rho \sigma + \frac { 2 T m H ( I + \sqrt { m } \rho \sigma ) \beta } { \mu } } \\ & { \le \frac { 2 H } { \alpha } + \frac { 2 H } { \mu } + ( H + \sqrt { m } \rho ) T m B _ { \alpha } ^ { \beta } + \frac { T L } { 2 } \alpha } \\ & { \displaystyle + \frac { 2 T L \alpha } { \mu } + 2 T \sqrt { \mu } \rho \sigma + \frac { 2 T m H ( I + \sqrt { m } \rho \sigma ) } { \mu } \beta . } \end{array}
$$

Finally, since $z _ { t } \in \Delta _ { m }$ is feasible,

$$
\operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \left. \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \right. _ { S _ { 1 } } \leq \left. \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \right. _ { S _ { 1 } } .
$$

Dividing the preceding inequality by T and applying this pointwise bound proves the theorem.

There is an immediate corollary of the above Theorem 6.

Corollary 2. Setting $\begin{array} { r } { \alpha = \sqrt { \frac { \mu B } { T L } } , \beta = \frac { 1 } { ( H + \sqrt { m } \rho \sigma ) m T } , 0 < \gamma < ( H + \sqrt { m } \rho \sigma ) m T } \end{array}$ and $\begin{array} { r } { \mu = \operatorname* { m i n } \{ \frac { \sqrt { L B } } { \sigma \rho \sqrt { T } } , 1 \} } \end{array}$ in Theorem $\delta ,$ the following holds

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \right] } \\ & { \leq \mathcal { O } \left( \sqrt { \frac { L B } { T } } + \left( \frac { \rho ^ { 2 } \sigma ^ { 2 } L B } { T } \right) ^ { 1 / 4 } \right) = \mathcal { O } ( 1 / T ^ { 1 / 4 } ) . } \end{array}\tag{38}
$$

Proof. Since $\begin{array} { r } { \alpha = \sqrt { \frac { \mu B } { T L } } , \beta = \frac { 1 } { ( H + \sqrt { m } \rho \sigma ) m T } , 0 < \gamma < ( H + \sqrt { m } \rho \sigma ) m T } \end{array}$ and $\begin{array} { r } { \mu = \operatorname* { m i n } \{ \frac { \sqrt { L B } } { \sigma \rho \sqrt { T } } , 1 \} } \end{array}$ in Theorem $6 ,$ , we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \operatorname* { m i n } _ { \tau \neq \mathsf { A } _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \right] } \\ & { \displaystyle \leq \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + ( H + \sqrt { m } \rho \sigma ) m B \frac { \beta } { \alpha } + \frac { L } { 2 } \alpha } \\ & { \displaystyle + \frac { 2 L \alpha } { \mu } + 2 \sqrt { \mu } \rho \sigma + \frac { 2 m H ( H + \sqrt { m } \rho \sigma ) } { \mu } \beta } \\ & { \displaystyle \leq ( \frac { 4 \sigma \rho H } { L B } + \frac { 1 } { 2 } ) \sqrt { \frac { L B } { T } } + \tau \left( \frac { \rho ^ { 2 } \sigma ^ { 2 } L B } { T } \right) ^ { 1 / 4 } } \\ & { \displaystyle = \mathcal { O } \left( \sqrt { \frac { L B } { T } } + \left( \frac { \rho ^ { 2 } \sigma ^ { 2 } L B } { T } \right) ^ { 1 / 4 } \right) = \mathcal { O } ( 1 / T ^ { 1 / 4 } ) . } \end{array}
$$

## B Convergence of Convex Case

Theorem 7. Let $\Omega \subseteq \mathbb { R } ^ { p \times q }$ be a nonempty compact convex set satisfying

$$
\mathrm { d i a m } _ { S _ { \infty } } ( \Omega ) : = \operatorname* { s u p } _ { \Theta , \Theta ^ { \prime } \in \Omega } \| \Theta - \Theta ^ { \prime } \| _ { S _ { \infty } } \leq D .
$$

Suppose that each objective $\ell ^ { i } : \Omega  \mathbb { R }$ is convex and diferentiable. Assume that the iterates generated by Algorithm 1 satisfy

$$
\begin{array} { r } { \Theta _ { t } \in \Omega , \qquad z _ { t } \in \Delta _ { m } , \qquad t = 1 , \dots , T . } \end{array}
$$

Then

$$
\begin{array} { r l } & { \cfrac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { \boldsymbol { \Theta } ^ { * } \in \Omega } \operatorname* { m i n } _ { \boldsymbol { z } ^ { * } \in \Delta _ { m } } { \boldsymbol { z } ^ { * } } ^ { \top } \left( \boldsymbol { L } ( \boldsymbol { \Theta } _ { t } ) - \boldsymbol { L } ( \boldsymbol { \Theta } ^ { * } ) \right) } \\ & { \qquad \leq \cfrac { D } { T } \displaystyle \sum _ { t = 1 } ^ { T } \left\| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \boldsymbol { \Theta } _ { t } ) \right\| _ { S _ { 1 } } . } \end{array}\tag{39}
$$

Proof. Fix any $t \in \{ 1 , \ldots , T \}$ . Since $z _ { t } \in \Delta _ { m }$ , for every $\Theta ^ { * } \in \Omega$ , we have

$$
\begin{array} { r l } & { \underset { z ^ { * } \in \Delta _ { m } } { \operatorname* { m i n } } z ^ { * ^ { \top } } \left( L ( \Theta _ { t } ) - L ( \Theta ^ { * } ) \right) } \\ & { \quad \quad \leq z _ { t } ^ { \top } \left( L ( \Theta _ { t } ) - L ( \Theta ^ { * } ) \right) . } \end{array}\tag{40}
$$

Taking the maximum over $\Theta ^ { * } \in \Omega$ on both sides gives

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { \boldsymbol { \Theta } ^ { * } \in \Omega } \operatorname* { m i n } _ { \boldsymbol { z } ^ { * } \in \Delta _ { m } } { \boldsymbol { z } ^ { * } } ^ { \top } \left( \boldsymbol { L } ( \boldsymbol { \Theta } _ { t } ) - \boldsymbol { L } ( \boldsymbol { \Theta } ^ { * } ) \right) } \\ & { \quad \quad \quad \leq \displaystyle \operatorname* { m a x } _ { \boldsymbol { \Theta } ^ { * } \in \Omega } z _ { t } ^ { \top } \left( \boldsymbol { L } ( \boldsymbol { \Theta } _ { t } ) - \boldsymbol { L } ( \boldsymbol { \Theta } ^ { * } ) \right) } \\ & { \quad \quad \quad = z _ { t } ^ { \top } \boldsymbol { L } ( \boldsymbol { \Theta } _ { t } ) - \displaystyle \operatorname* { m i n } _ { \boldsymbol { \Theta } ^ { * } \in \Omega } z _ { t } ^ { \top } \boldsymbol { L } ( \boldsymbol { \Theta } ^ { * } ) . } \end{array}\tag{41}
$$

Because $\Omega$ is compact and each $\ell ^ { i }$ is continuous, there exists

$$
\Theta _ { t } ^ { * } \in \arg \operatorname* { m i n } _ { \Theta \in \Omega } z _ { t } ^ { \top } { \cal L } ( \Theta ) .
$$

Consequently, the right-hand side of (41) is

$$
\sum _ { i = 1 } ^ { m } z _ { i , t } \left( \ell ^ { i } ( \Theta _ { t } ) - \ell ^ { i } ( \Theta _ { t } ^ { * } ) \right) .
$$

By the convexity of each $\ell ^ { i }$ , we have

$$
\ell ^ { i } ( \Theta _ { t } ) - \ell ^ { i } ( \Theta _ { t } ^ { * } ) \leq \left. \nabla \ell ^ { i } ( \Theta _ { t } ) , \Theta _ { t } - \Theta _ { t } ^ { * } \right. .
$$

Since $z _ { i , t } \geq 0$ , multiplying by $z _ { i , t }$ and summing over i yields

$$
\begin{array} { r l } & { z _ { t } ^ { \top } L ( \boldsymbol { \Theta } _ { t } ) - z _ { t } ^ { \top } L ( \boldsymbol { \Theta } _ { t } ^ { * } ) } \\ & { \qquad \leq \left. \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \boldsymbol { \Theta } _ { t } ) , \boldsymbol { \Theta } _ { t } - \boldsymbol { \Theta } _ { t } ^ { * } \right. . } \end{array}\tag{42}
$$

Using the duality between the nuclear norm and the spectral norm,

$$
| \langle A , B \rangle | \leq \| A \| _ { S _ { 1 } } \| B \| _ { S _ { \infty } } ,
$$

we obtain

$$
\begin{array} { r l r } {  {  \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) , \Theta _ { t } - \Theta _ { t } ^ { * }  } } \\ & { } & { \leq \| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } \| \Theta _ { t } - \Theta _ { t } ^ { * } \| _ { S _ { \infty } } } \\ & { } & { \leq D \| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } , } \end{array}\tag{43}
$$

where the last inequality follows from $\Theta _ { t } , \Theta _ { t } ^ { * } \in \Omega$ and diam $s _ { \infty } ( \Omega ) \leq D$

Combining (41)–(43), we obtain, for every $t ,$

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { { \mathbf { \theta } } ^ { * } \in \Omega } \operatorname* { m i n } _ { { \mathbf { z } } ^ { * } \in \Delta _ { m } } { \mathbf { \boldsymbol { z } } } ^ { * } { \top } ( { \pmb { L } } ( { \pmb { \Theta } } _ { t } ) - { \pmb { L } } ( { \pmb { \Theta } } ^ { * } ) ) } } \\ & { } & { \leq D \| \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( { \pmb { \Theta } } _ { t } ) \| _ { S _ { 1 } } . } \end{array}\tag{44}
$$

Summing this inequality over $t = 1 , \dots , T$ and dividing by $T$ completes the proof.

□

## C Approximation to Avoid Calculating Multiple Gradients

In this section, we present an approximation of the weight update rule to avoid calculating multiple gradients in Algorithm 1.

In Algorithm 1, we update z using a single gradient-descent step as follows

$$
z _ { t + 1 } = z _ { t } - \beta \nabla _ { z } \left( \frac { 1 } { 2 } \| z _ { t } ^ { \top } \nabla \mathcal { L } ( \Theta _ { t } ) \| _ { S _ { 1 } } ^ { 2 } \right) ,
$$

where $\mathcal { L } ( \boldsymbol { \Theta } ) = ( \ell ^ { 1 } ( \boldsymbol { \Theta } ) , \dots , \ell ^ { m } ( \boldsymbol { \Theta } ) ) ^ { \top }$ . Let $[ \mathbf { \pmb { a } } ] _ { i }$ denote the ith component of a vector. However, the update rule still necessitates information about the gradients of all the objectives. Therefore, we consider approximating the gradient $\begin{array} { r } { \nabla _ { z } \big ( \frac { 1 } { 2 } \| z _ { t } ^ { \top } \nabla \mathcal { L } ( \Theta _ { t } ) \| _ { S _ { 1 } } ^ { 2 } \big ) } \end{array}$ using first-order Taylor’s approximation:

$$
\begin{array} { r l } & { \left[ \nabla _ { z } \frac { 1 } { 2 } \| z _ { t } ^ { \top } \nabla \mathcal { L } ( \Theta _ { t } ) \| _ { s _ { 1 } } ^ { 2 } \right] _ { i } } \\ & { = \| G _ { t } \| _ { S _ { 1 } } \left. \hat { \nabla } \| G _ { t } \| _ { s _ { 1 } } , \nabla \ell ^ { i } ( \Theta _ { t } ) \right. } \\ & { = \| G _ { t } \| _ { S _ { 1 } } \left. U _ { t } V _ { t } ^ { \top } , \nabla \ell ^ { i } ( \Theta _ { t } ) \right. } \\ & { = \| G _ { t } \| _ { S _ { 1 } } \left. W _ { t } , \nabla \ell ^ { i } ( \Theta _ { t } ) \right. } \\ & { \approx \displaystyle \frac { \| G _ { t } \| _ { S _ { 1 } } } { \alpha } \left[ \ell ^ { i } ( \Theta _ { t } ) - \ell ^ { i } ( \Theta _ { t + 1 } ) \right] , } \\ & { \quad i \in [ m ] . } \end{array}
$$

where $\hat { \nabla } \| G _ { t } \| _ { S _ { 1 } }$ represents an element of the subdiferential of the nuclear norm at $G _ { t }$ . This identity is exact in the no-momentum case $\mu = 1$ , where $M _ { t } = G _ { t }$ and $W _ { t } = { \mathrm { P o l a r } } ( G _ { t } )$

The factor $\| G _ { t } \| _ { S _ { 1 } }$ is a nonnegative scalar that does not change the update direction. Consistent with the practical algorithm, we retain the scale-normalized direction and control the update magnitude through the step size. Then consequently, we have an approximate method for solving $z _ { t }$ without backpropagation:

$$
z _ { t + 1 } = z _ { t } - \beta \tilde { \delta } _ { t } ,\tag{45}
$$

$$
[ \tilde { \pmb { \delta } } _ { t } ] _ { i } = \frac { 1 } { \alpha } \left[ \ell ^ { i } ( \Theta _ { t } ) - \ell ^ { i } ( \Theta _ { t + 1 } ) \right] , \quad i \in [ m ] .\tag{46}
$$

In practical MOON, we use the resulting scale-normalized dual direction in the logit-space update of Algorithm 1, which corresponds to an exponentiated-gradient-type online update on the simplex.

## D Efect of Finite-Step Newton–Schulz Approximation Error

The convergence analysis before is stated in terms of the exact polar factor of the momentum matrix. In practice, however, Algorithm 1 computes the update direction using only a finite number of Newton–Schulz iterations. Consequently, the resulting update direction may difer from an exact polar factor.

In this section, we quantify the efect of this approximation error on the deterministic convergence guarantee. We focus on the deterministic setting because it isolates the efect of the finite-step Newton–Schulz approximation from the error caused by stochastic-gradient noise. The stochastic case follows by the same argument after conditioning on the history up to iteration t and taking total expectation. We therefore present only the deterministic proof to avoid repeating the same argument. We show that, under a uniform relative approximation condition, the finite-step Newton– Schulz implementation preserves the $\mathcal { O } ( T ^ { - 1 / 2 } )$ Pareto-stationarity rate, up to a multiplicative factor depending on the approximation accuracy.

For each iteration t, define the weighted gradient matrix

$$
\pmb { G } _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) , \qquad z _ { t } \in \Delta _ { m } ,
$$

where

$$
\Delta _ { m } : = \left\{ z \in \mathbb { R } _ { + } ^ { m } : \sum _ { i = 1 } ^ { m } z _ { i } = 1 \right\} .
$$

The momentum matrix is updated according to

$$
M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu G _ { t } , \qquad 0 < \mu \leq 1 .
$$

Let

$$
P _ { t } \in { \mathrm { P o l a r } } ( M _ { t } )
$$

denote a polar factor of $M _ { t }$ . In particular, $P _ { t }$ satisfies

$$
\| P _ { t } \| _ { S _ { \infty } } \le 1 , \qquad \langle M _ { t } , P _ { t } \rangle = \| M _ { t } \| _ { S _ { 1 } } .\tag{47}
$$

These properties remain valid when $M _ { t }$ is rank deficient; in this case, $P _ { t }$ should be understood as any polar factor satisfying (47).

Let $\widetilde { W } _ { t }$ denote the direction returned by the finite-step Newton–Schulz iteration. The actual parameter update is

$$
\begin{array} { r } { \Theta _ { t + 1 } = \Theta _ { t } - \alpha \widetilde { W } _ { t } . } \end{array}\tag{48}
$$

We impose the following approximation condition.

Assumption 4. For every $t \geq 1$ , there exists a polar factor $P _ { t } \in { \mathrm { P o l a r } } ( M _ { t } )$ such that

$$
\varepsilon _ { t , q } : = \Bigl | \Bigl | \widetilde { W } _ { t } - P _ { t } \Bigr | \Bigr | _ { S _ { \infty } } \le \varepsilon _ { q } < 1 ,\tag{49}
$$

where $q$ is the number of Newton–Schulz iterations and $\varepsilon _ { q }$ is independent of t.

Assumption 4 is a uniform operator-norm approximation condition. Through nuclear–spectral norm duality, it induces a relative loss of alignment proportional to $\varepsilon _ { q } \lVert \boldsymbol { M } _ { t } \rVert _ { S _ { 1 } }$ . Unlike an additive descent-error condition, it does not introduce a nonvanishing residual term into the final stationarity bound. Therefore, any fixed number $q$ of Newton–Schulz iterations preserves convergence to Pareto stationarity, provided that the uniform bound

$$
\operatorname* { s u p } _ { t \geq 1 } \left\| \widetilde W _ { t , q } - { P } _ { t } \right\| _ { S _ { \infty } } \leq \varepsilon _ { q } < 1
$$

holds along the optimization trajectory. In this case, the finite-step approximation afects only the constants in the convergence bound.

Moreover, under the standard convergence regime of the employed Newton–Schulz iteration, the approximation error is nonincreasing with respect to the number of Newton–Schulz steps and satisfies

$$
\varepsilon _ { q + 1 } \leq \varepsilon _ { q } , \qquad \operatorname* { l i m } _ { q \to \infty } \varepsilon _ { q } = 0 .
$$

Consequently, the efect of the finite-step approximation becomes weaker as $q$ increases, and the resulting convergence bound approaches the corresponding exact-polar bound. These properties also explain why Assumption 4 is reasonable under the standard Newton–Schulz convergence conditions.

The following immediate consequence will be used repeatedly.

Lemma 5. Under Assumption $^ { 4 , }$

$$
\Bigl \| \widetilde { W } _ { t } \Bigr \| _ { S _ { \infty } } \le 1 + \varepsilon _ { t , q } \le 1 + \varepsilon _ { q } .\tag{50}
$$

Proof. By the triangle inequality and (47),

$$
\begin{array} { r l } & { \left\| \widetilde { W } _ { t } \right\| _ { S _ { \infty } } \leq \| { \cal P } _ { t } \| _ { S _ { \infty } } + \left\| \widetilde { W } _ { t } - { \cal P } _ { t } \right\| _ { S _ { \infty } } } \\ & { \qquad \leq 1 + \varepsilon _ { t , q } \leq 1 + \varepsilon _ { q } . } \end{array}
$$

Lemma 6. Assume that Assumption $\mathcal { Q }$ and $\it 4$ hold and that $0 < \beta \gamma < 1$ . Then for any $t = 1 , \dots , T$ the weight updates in Algorithm 1 satisfy the following inequality

$$
\| z _ { t } - z _ { t + 1 } \| _ { \infty } \leq \beta H ( 1 + \varepsilon _ { q } ) .\tag{51}
$$

Proof. The softmax mapping is $1 / 2 \ – \mathrm { I }$ ipschitz with respect to the $\ell _ { \infty }$ norm. Indeed, for any $\pmb { x } , \pmb { y } \in \mathbb { R } ^ { m }$ ，

$$
\| \mathrm { s o f t m a x } ( { \pmb x } ) - \mathrm { s o f t m a x } ( { \pmb y } ) \| _ { \infty } \leq \frac { 1 } { 2 } \| { \pmb x } - { \pmb y } \| _ { \infty } .
$$

Consequently,

$$
\begin{array} { r } { \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \frac { 1 } { 2 } \| \pmb { \xi } _ { t + 1 } - \pmb { \xi } _ { t } \| _ { \infty } } \\ { = \displaystyle \frac { \beta } { 2 } \| \pmb { \delta } _ { t } + \gamma \pmb { \xi } _ { t } \| _ { \infty } . } \end{array}\tag{52}
$$

We first bound $\delta _ { t }$ . By the duality between the spectral norm and the nuclear norm,

$$
\begin{array} { r l } & { \| \pmb { \delta } _ { t } \| _ { \infty } = \underset { i \in [ m ] } { \operatorname* { m a x } } \left| \left. \widetilde { W } _ { t } , \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) \right. \right| } \\ & { \qquad \leq \| \widetilde { W } _ { t } \| _ { S _ { \infty } } \underset { i \in [ m ] } { \operatorname* { m a x } } \| \nabla \ell ^ { i } ( \pmb { \Theta } _ { t } ) \| _ { S _ { 1 } } } \\ & { \qquad \leq ( 1 + \varepsilon _ { t , q } ) H \leq ( 1 + \varepsilon _ { q } ) H . } \end{array}
$$

Next, the update of $\xi _ { t }$ can be written as

$$
\pmb { \xi } _ { t + 1 } = ( 1 - \beta \gamma ) \pmb { \xi } _ { t } - \beta \pmb { \delta } _ { t } .
$$

Since $0 < \beta \gamma < 1$ , the coeficient $1 - \beta \gamma$ is nonnegative. Thus, using (53),

$$
\begin{array} { r l } & { \| \pmb { \xi } _ { t + 1 } \| _ { \infty } \leq ( 1 - \beta \gamma ) \| \pmb { \xi } _ { t } \| _ { \infty } + \beta \| \pmb { \delta } _ { t } \| _ { \infty } } \\ & { \qquad \leq ( 1 - \beta \gamma ) \| \pmb { \xi } _ { t } \| _ { \infty } + \beta ( 1 + \varepsilon _ { q } ) H . } \end{array}\tag{53}
$$

Unrolling (53) and using ${ \boldsymbol { \xi } } _ { 1 } = \mathbf { 0 }$ gives

$$
\| \pmb { \xi } _ { t } \| _ { \infty } \leq \beta ( 1 + \varepsilon _ { q } ) H \sum _ { s = 0 } ^ { t - 2 } ( 1 - \beta \gamma ) ^ { s }
$$

$$
= \frac { ( 1 + \varepsilon _ { q } ) H } { \gamma } \left[ 1 - ( 1 - \beta \gamma ) ^ { t - 1 } \right] \leq \frac { ( 1 + \varepsilon _ { q } ) H } { \gamma } .
$$

Since $z _ { t } = \mathrm { S o f t m a x } ( \pmb { \xi } _ { t } )$ , the 1/2-Lipschitz continuity of softmax under the $\ell _ { \infty }$ norm bounds the change in $z _ { t }$ by the corresponding change in the logits. Applying the triangle inequality to the logit update and substituting $\| \pmb { \delta } _ { t } \| _ { \infty } \le ( 1 + \varepsilon _ { q } ) H$ and $\| \pmb { \xi } _ { t } \| _ { \infty } \le ( 1 + \varepsilon _ { q } ) H / \gamma$ , we obtain

$$
\begin{array} { r l } & { \| z _ { t + 1 } - z _ { t } \| _ { \infty } \leq \displaystyle \frac { \beta } { 2 } \left( \| \delta _ { t } \| _ { \infty } + \gamma \| \pmb { \xi } _ { t } \| _ { \infty } \right) } \\ & { \leq \displaystyle \frac { \beta } { 2 } ( ( 1 + \varepsilon _ { q } ) H + ( 1 + \varepsilon _ { q } ) H ) = ( 1 + \varepsilon _ { q } ) \beta H . } \end{array}
$$

This completes the proof.

Lemma 7. Assume that Assumption 1, 2 and 4 hold, and that $0 < \beta \gamma < 1$ . Then for any $t = 1 , \dots , T$ the weight updates in Algorithm 1 satisfy the following inequality

$$
\begin{array} { r l } & { \displaystyle \| M _ { t } - \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell ^ { i } ( \Theta _ { t } ) \| _ { S _ { 1 } } } \\ & { \displaystyle \leq ( 1 - \mu ) ^ { t - 1 } \| \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell ^ { i } ( \Theta _ { 1 } ) \| _ { S _ { 1 } } } \\ & { \displaystyle + \frac { L \alpha ( 1 + \varepsilon _ { q } ) } { \mu } + \frac { m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) } { \mu } . } \end{array}\tag{54}
$$

Proof. Let

$$
\pmb { G } _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \pmb { \Theta } _ { t } ) \quad \mathrm { a n d } \quad E _ { t } : = M _ { t } - \pmb { G } _ { t } .
$$

Using the momentum update $M _ { t } = ( 1 - \mu ) M _ { t - 1 } + \mu G _ { t }$ , we obtain, for $t \geq 2 .$

$$
\mathbf { } E _ { t } = ( 1 - \mu ) \bigl ( \mathbf { } E _ { t - 1 } + \mathbf { } G _ { t - 1 } - \mathbf { } G _ { t } \bigr ) .
$$

Iterating this recursion and using $M _ { 0 } = 0$ gives

$$
\begin{array} { r l } {  { \| E _ { t } \| _ { S _ { 1 } } \le ( 1 - \mu ) ^ { t } \| G _ { 1 } \| _ { S _ { 1 } } } } \\ & { + \displaystyle \sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } \| G _ { j - 1 } - G _ { j } \| _ { S _ { 1 } } . } \end{array}
$$

Moreover,

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle { G _ { j } - G _ { j - 1 } = \sum _ { i = 1 } ^ { m } z _ { i , j } \big ( \nabla \ell _ { i } ( \Theta _ { j } ) - \nabla \ell _ { i } ( \Theta _ { j - 1 } ) \big ) } } \\ { \displaystyle } & { + \sum _ { i = 1 } ^ { m } ( z _ { i , j } - z _ { i , j - 1 } ) \nabla \ell _ { i } ( \Theta _ { j - 1 } ) . } \end{array}
$$

By Assumptions 1–2, Lemma 6, and $\| \Theta _ { j } - \Theta _ { j - 1 } \| _ { S _ { \infty } } = \alpha \| \widetilde { W } _ { j - 1 } \| _ { S _ { \infty } }$ , we have

$$
\| G _ { j } - G _ { j - 1 } \| _ { S _ { 1 } } \leq L \alpha ( 1 + \varepsilon _ { q } ) + m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) .
$$

Since

$$
\sum _ { j = 2 } ^ { t } ( 1 - \mu ) ^ { t - j + 1 } \leq \frac { 1 } { \mu } ,
$$

it follows that

$$
\begin{array} { l } { \displaystyle \left\| M _ { t } - \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \Theta _ { t } ) \right\| _ { S _ { 1 } } } \\ { \displaystyle \leq ( 1 - \mu ) ^ { t - 1 } \left\| \sum _ { i = 1 } ^ { m } z _ { i , 1 } \nabla \ell _ { i } ( \Theta _ { 1 } ) \right\| _ { S _ { 1 } } } \\ { \displaystyle + \frac { L \alpha ( 1 + \varepsilon _ { q } ) } { \mu } + \frac { m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) } { \mu } . } \end{array}
$$

Theorem 8. Suppose Assumption $1 \mathrm { - } \mathit { 4 }$ hold, let $\ell ^ { i } : \mathbb { R } ^ { p \times q }  \mathbb { R }$ be non-convex for each $i \in [ m ]$ Suppose that $0 < \beta \gamma < 1$ , then the iterations of Algorithm 1 satisfy the following inequality

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \underset { \varepsilon _ { t } ^ { \prime } \in \Delta _ { m } } { \mathrm { m i n } } \ \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } } \\ & { \displaystyle \leq \frac { 1 } { 1 - \varepsilon _ { q } } \Big ( \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + \frac { 2 L \alpha ( 1 + \varepsilon _ { q } ) } { \mu } } \\ & { \displaystyle + \frac { 2 m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) } { \mu } + \frac { m H B \beta ( 1 + \varepsilon _ { q } ) } { \alpha } } \\ & { \displaystyle + \frac { L \alpha ( 1 + \varepsilon _ { q } ) ^ { 2 } } { 2 } \Big ) . } \end{array}\tag{55}
$$

Proof. Define

$$
G _ { t } : = \sum _ { i = 1 } ^ { m } z _ { i , t } \nabla \ell _ { i } ( \Theta _ { t } ) .
$$

By L-smoothness and $\Theta _ { t + 1 } = \Theta _ { t } - \alpha \widetilde { W } _ { t }$ ,

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \Theta _ { t + 1 } ) \leq \sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \Theta _ { t } ) - \alpha \langle { \cal G } _ { t } , \widetilde { { \cal W } } _ { t } \rangle } } \\ { { \displaystyle \qquad + \frac { L \alpha ^ { 2 } ( 1 + \varepsilon _ { q } ) ^ { 2 } } { 2 } } . } \end{array}
$$

Since $\widetilde { W } _ { t }$ is the direction returned by the finite-step Newton–Schulz iteration, $\langle \boldsymbol { M } _ { t } , \widetilde { \boldsymbol { W } } _ { t } \rangle =$ $\langle M _ { t } , P _ { t } \rangle + \langle M _ { t } , \widetilde { W } _ { t } - P _ { t } \rangle \geq ( 1 - \varepsilon _ { q } ) \Vert M _ { t } \Vert _ { S _ { 1 } }$ . Therefore,

$$
\begin{array} { r l } & { - \alpha \langle \pmb { G } _ { t } , \widetilde { \pmb { W } } _ { t } \rangle \leq - \alpha \langle \pmb { M } _ { t } , \widetilde { \pmb { W } } _ { t } \rangle + \alpha \langle \pmb { M } _ { t } - \pmb { G } _ { t } , \widetilde { \pmb { W } } _ { t } \rangle } \\ & { \leq - ( 1 - \varepsilon _ { q } ) \alpha \Vert \pmb { M } _ { t } \Vert _ { S _ { 1 } } + ( 1 + \varepsilon _ { q } ) \alpha \Vert \pmb { M } _ { t } - \pmb { G } _ { t } \Vert _ { S _ { 1 } } } \\ & { \leq - ( 1 - \varepsilon _ { q } ) \alpha ( \Vert \pmb { G } _ { t } \Vert _ { S _ { 1 } } - \Vert \pmb { M } _ { t } - \pmb { G } _ { t } \Vert _ { S _ { 1 } } ) } \\ & { ~ + ( 1 + \varepsilon _ { q } ) \alpha \Vert \pmb { M } _ { t } - \pmb { G } _ { t } \Vert _ { S _ { 1 } } } \\ & { \leq - ( 1 - \varepsilon _ { q } ) \alpha \Vert \pmb { G } _ { t } \Vert _ { S _ { 1 } } + 2 \alpha \Vert \pmb { M } _ { t } - \pmb { G } _ { t } \Vert _ { S _ { 1 } } . } \end{array}
$$

After adding and subtracting $\begin{array} { r } { \sum _ { i } z _ { i , t + 1 } \ell _ { i } ( \Theta _ { t + 1 } ) } \end{array}$ and rearranging, we obtain

$$
\begin{array} { r l } & { ( 1 - \varepsilon _ { q } ) \| G _ { t } \| _ { S _ { 1 } } } \\ & { \le \displaystyle \frac { 1 } { \alpha } \left( \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t } \ell _ { i } ( \Theta _ { t } ) - \displaystyle \sum _ { i = 1 } ^ { m } z _ { i , t + 1 } \ell _ { i } ( \Theta _ { t + 1 } ) \right) } \\ & { + \displaystyle \frac { 1 } { \alpha } \displaystyle \sum _ { i = 1 } ^ { m } ( z _ { i , t + 1 } - z _ { i , t } ) \ell _ { i } ( \Theta _ { t + 1 } ) } \\ & { + \displaystyle \frac { L \alpha ( 1 + \varepsilon _ { q } ) ^ { 2 } } { 2 } + 2 \| M _ { t } - G _ { t } \| _ { S _ { 1 } } . } \end{array}
$$

Lemma 6 imply

$$
\left. \sum _ { i = 1 } ^ { m } ( z _ { i , t + 1 } - z _ { i , t } ) \ell _ { i } ( \Theta _ { t + 1 } ) \right. \leq m B H \beta ( 1 + \varepsilon _ { q } ) .
$$

Applying Lemma $^ { 7 , }$ summing over $t = 1 , \dots , T$ , and using $\| G _ { 1 } \| _ { S _ { 1 } } \leq H$ , we obtain

$$
\begin{array} { l } { \displaystyle ( 1 - \varepsilon _ { q } ) \sum _ { t = 1 } ^ { T } \| \pmb { G } _ { t } \| _ { S _ { 1 } } \leq \frac { 2 B } { \alpha } + \frac { 2 H } { \mu } + \frac { 2 T L \alpha ( 1 + \varepsilon _ { q } ) } { \mu } } \\ { + \frac { 2 T m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) } { \mu } + \frac { T m H B \beta ( 1 + \varepsilon _ { q } ) } { \alpha } } \\ { + \frac { T L \alpha ( 1 + \varepsilon _ { q } ) ^ { 2 } } { 2 } . } \end{array}
$$

Finally,

$$
\operatorname* { m i n } _ { z \in \Delta _ { m } } \left. \sum _ { i = 1 } ^ { m } z _ { i } \nabla \ell _ { i } ( \Theta _ { t } ) \right. _ { S _ { 1 } } \leq \lVert \pmb { G } _ { t } \rVert _ { S _ { 1 } } .
$$

Dividing the preceding inequality by $( 1 - \varepsilon _ { q } ) T$ yields the claimed result.

There is an immediate corollary of the above Theorem.

Corollary 3. Setting $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma <$ mHT in Theorem $\delta ,$ then, for any fixed $\mu \in ( 0 , 1 ]$ , the following holds

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } _ { z _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \Theta _ { t } ) z _ { t } ^ { * } \| _ { S _ { 1 } } \leq \frac { 1 } { 1 - \varepsilon _ { q } } O ( 1 / \sqrt { T } ) .\tag{56}
$$

Proof. Since $\begin{array} { r } { \alpha = \sqrt { \frac { B } { L T } } , \beta = \frac { 1 } { m H T } } \end{array}$ and $0 < \gamma < m H T$ in Theorem 8, we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \displaystyle \operatorname* { m i n } _ { \boldsymbol { z } _ { t } ^ { * } \in \Delta _ { m } } \| \nabla L ( \boldsymbol { \Theta } _ { t } ) \boldsymbol { z } _ { t } ^ { * } \| _ { S _ { 1 } } } \\ & { \displaystyle \leq \frac { 1 } { 1 - \varepsilon _ { q } } \bigg ( \frac { 2 B } { \alpha T } + \frac { 2 H } { \mu T } + \frac { 2 L \alpha ( 1 + \varepsilon _ { q } ) } { \mu } } \\ & { \quad \displaystyle + \frac { 2 m H ^ { 2 } \beta ( 1 + \varepsilon _ { q } ) } { \mu } + \frac { m H B \beta ( 1 + \varepsilon _ { q } ) } { \alpha } } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle { \vphantom { \frac { \sum \alpha \bigl ( 1 + \varepsilon _ { q } ) ^ { 2 } } } + \frac { L \alpha \bigl ( 1 + \varepsilon _ { q } \bigr ) ^ { 2 } } { 2 } \biggr ) } } } \\  { \displaystyle { = \frac { 1 } { 1 - \varepsilon _ { q } } \biggl [ \biggl ( 3 + \frac { 2 ( 1 + \varepsilon _ { q } ) } { \mu } + ( 1 + \varepsilon _ { q } ) } } \\ { { \displaystyle { \vphantom { \frac { 1 } { 1 - \varepsilon _ { q } } \biggl ( 1 + \varepsilon _ { q } \bigr ) ^ { 2 } } \biggr ) \sqrt { \frac { L B } { T } } + \frac { ( 4 + 2 \varepsilon _ { q } ) H } { \mu T } \biggr ] } } } \\ { { \displaystyle { = \frac { 1 } { 1 - \varepsilon _ { q } } \mathcal { O } ( 1 / \sqrt { T } ) } . } } \end{array}
$$

Increasing q improves all three approximation-dependent factors in Theorem 8:

$$
\frac { 1 } { 1 - \varepsilon _ { q } } , \qquad 1 + \varepsilon _ { q } , \qquad ( 1 + \varepsilon _ { q } ) ^ { 2 } .
$$

Hence, the efect of the finite-step approximation becomes weaker as more Newton–Schulz steps are performed, and the bound continuously approaches the exact-polar convergence bound as $q \to \infty$ For any fixed q satisfying the uniform condition ${ \varepsilon } _ { q } < 1$ , the approximation changes only the constants and does not alter the $\mathcal { O } ( T ^ { - 1 / 2 } )$ deterministic convergence rate.

## E A Two-Task Matrix Example

We present a two-task example to illustrate the diference between Euclidean gradient manipulation and the matrix-aware update derived in Proposition 3. Let the parameter be $\Theta \in \mathbb { R } ^ { 2 \times 2 }$ and define

$$
\begin{array} { c } { { \ell _ { 1 } ( \Theta ) = \displaystyle \frac { 1 } { 2 } \| \Theta \| _ { F } ^ { 2 } + \langle A + B , \Theta \rangle , } } \\ { { \ell _ { 2 } ( \Theta ) = \displaystyle \frac { 1 } { 2 } \| \Theta \| _ { F } ^ { 2 } + \langle A - B , \Theta \rangle , } } \end{array}\tag{57}
$$

where

$$
\begin{array}{c} A = { \binom { 2 } { 0 } } \ 0  \end{array} \qquad B = { \left( \begin{array} { l l } { 0 } & { 1 } \\ { - 1 } & { 0 } \end{array} \right) } .\tag{58}
$$

At $\Theta _ { 0 } = 0$ , the two losses have the same value, $\ell _ { 1 } ( \Theta _ { 0 } ) = \ell _ { 2 } ( \Theta _ { 0 } ) = 0$ , and their gradients are

$$
\begin{array} { r l } & { g _ { 1 } = \nabla \ell _ { 1 } ( \Theta _ { 0 } ) = A + B = \left( \begin{array} { l l } { 2 } & { 1 } \\ { - 1 } & { 1 } \end{array} \right) , } \\ & { g _ { 2 } = \nabla \ell _ { 2 } ( \Theta _ { 0 } ) = A - B = \left( \begin{array} { l l } { 2 } & { - 1 } \\ { 1 } & { 1 } \end{array} \right) . } \end{array}\tag{59}
$$

For a task weight $z \in [ 0 , 1 ]$ , let $s = 2 z - 1$ . The weighted gradient is

$$
\begin{array} { r l } { G ( z ) = z g _ { 1 } + ( 1 - z ) g _ { 2 } } & { { } } \\ { = A + s B = \binom { 2 } { - s } s \Big ) . } \end{array}\tag{60}
$$

MGDA direction. MGDA selects the task weight by minimizing the squared Frobenius norm:

$$
\frac 1 2 \| G ( z ) \| _ { F } ^ { 2 } = \frac 1 2 \left( 5 + 2 s ^ { 2 } \right) .\tag{61}
$$

The unique minimizer is $s ^ { * } = 0$ , or equivalently $z ^ { * } = 1 / 2$ , and hence $G _ { \mathrm { M G D A } } ^ { * } = A$ . To compare the two methods under the same spectral-norm trust-region budget, we normalize the MGDA direction to unit spectral norm:

$$
\begin{array} { r l } & { P _ { \mathrm { M G D A } } = \frac { G _ { \mathrm { M G D A } } ^ { * } } { \| G _ { \mathrm { M G D A } } ^ { * } \| _ { S _ { \infty } } } = \left( \begin{array} { l l } { 1 } & { 0 } \\ { 0 } & { \frac { 1 } { 2 } } \end{array} \right) , } \\ & { \| P _ { \mathrm { M G D A } } \| _ { S _ { \infty } } = 1 . } \end{array}\tag{62}
$$

MOON direction. MOON instead minimizes the squared nuclear norm. For any $2 \times 2$ matrix G,

$$
\| G \| _ { S _ { 1 } } ^ { 2 } = \| G \| _ { F } ^ { 2 } + 2 | \operatorname* { d e t } ( G ) | .\tag{63}
$$

Since det $( G ( z ) ) = 2 + s ^ { 2 } > 0$ , the MOON objective becomes

$$
\begin{array} { c } { \displaystyle \frac { 1 } { 2 } \| \boldsymbol { G } ( \boldsymbol { z } ) \| _ { S _ { 1 } } ^ { 2 } = \frac { 1 } { 2 } \left( 5 + 2 s ^ { 2 } + 2 ( 2 + s ^ { 2 } ) \right) } \\ { \displaystyle = \frac { 1 } { 2 } \left( 9 + 4 s ^ { 2 } \right) . } \end{array}\tag{64}
$$

Its unique minimizer is also $s ^ { * } = 0$ , so the optimal weighted gradient is

$$
G _ { \mathrm { M O O N } } ^ { * } = A = { \binom { 2 } { 0 } } \ 1 \biggr ) .\tag{65}
$$

The singular values of $G _ { \mathrm { M O O N } } ^ { * }$ are 2 and 1. Therefore, its polar factor is

$$
\begin{array} { r l } & { P _ { \mathrm { M O O N } } = \mathrm { P o l a r } ( G _ { \mathrm { M O O N } } ^ { * } ) = U V ^ { \top } = I , } \\ & { \| P _ { \mathrm { M O O N } } \| _ { S _ { \infty } } = 1 . } \end{array}\tag{66}
$$

Because the singular values of $G _ { \mathrm { M O O N } } ^ { * }$ are unequal, $P _ { \mathrm { M G D A } }$ and $P _ { \mathrm { M O O N } }$ are not scalar multiples of each other. Thus, their diference cannot be attributed solely to update scaling.

Common descent comparison. We compare the two directions under the same unit spectral norm budget. For MGDA,

$$
\begin{array} { c } { { \langle g _ { 1 } , P _ { \mathrm { M G D A } } \rangle = \langle g _ { 2 } , P _ { \mathrm { M G D A } } \rangle } } \\ { { { } } } \\ { { = 2 + \frac { 1 } { 2 } = \frac { 5 } { 2 } . } } \end{array}\tag{67}
$$

For MOON,

$$
\begin{array} { c l c r } { { \langle g _ { 1 } , P _ { \mathrm { M O O N } } \rangle = \langle g _ { 2 } , P _ { \mathrm { M O O N } } \rangle } } \\ { { } } & { { } } \\ { { } } & { { = 2 + 1 = 3 . } } \end{array}\tag{68}
$$

Consequently,

$$
\operatorname* { m i n } _ { i \in \{ 1 , 2 \} } \langle g _ { i } , P _ { \mathrm { M O O N } } \rangle > \operatorname* { m i n } _ { i \in \{ 1 , 2 \} } \langle g _ { i } , P _ { \mathrm { M G D A } } \rangle .\tag{69}
$$

Hence, for the parameter update $\Theta ^ { + } = \Theta _ { 0 } - \eta P$ , MOON provides a strictly larger first-order decrease for both objectives while using the same spectral-norm update budget:

$$
\| \eta P _ { \mathrm { M G D A } } \| _ { S _ { \infty } } = \| \eta P _ { \mathrm { M O O N } } \| _ { S _ { \infty } } = \eta .\tag{70}
$$

Quadratic upper-bound comparison. For either objective,

$$
\begin{array} { r l } & { \| \nabla \ell _ { i } ( \Theta ) - \nabla \ell _ { i } ( \Theta ^ { \prime } ) \| _ { S _ { 1 } } = \| \Theta - \Theta ^ { \prime } \| _ { S _ { 1 } } } \\ & { \qquad \leq 2 \| \Theta - \Theta ^ { \prime } \| _ { S _ { \infty } } . } \end{array}\tag{71}
$$

Thus, both losses are 2-smooth with respect to the spectral norm. For either direction satisfying $\| \cal P \| _ { S _ { \infty } } = 1$

$$
\ell _ { i } ( \Theta _ { 0 } - \eta P ) \leq \ell _ { i } ( \Theta _ { 0 } ) - \eta \langle g _ { i } , P \rangle + \eta ^ { 2 } .\tag{72}
$$

The corresponding upper bounds are

$$
\begin{array} { l } { { \ell _ { i } ( \Theta _ { 0 } - \eta P _ { \mathrm { M G D A } } ) \le - \frac { 5 } { 2 } \eta + \eta ^ { 2 } , } } \\ { { \ell _ { i } ( \Theta _ { 0 } - \eta P _ { \mathrm { M O O N } } ) \le - 3 \eta + \eta ^ { 2 } . } } \end{array}\tag{73}
$$

Therefore, the MOON direction gives a strictly smaller worst-case quadratic upper bound for every $\eta > 0$

The exact objective values are

$$
\begin{array} { l c l } { { \ell _ { i } ( \Theta _ { 0 } - \eta P _ { \mathrm { M G D A } } ) = - \frac { 5 } { 2 } \eta + \frac { 5 } { 8 } \eta ^ { 2 } , } } \\ { { \ell _ { i } ( \Theta _ { 0 } - \eta P _ { \mathrm { M O O N } } ) = - 3 \eta + \eta ^ { 2 } . } } \end{array}\tag{74}
$$

It follows that

$$
\begin{array} { r l r } & { } & { \ell _ { i } \big ( \Theta _ { 0 } - \eta P _ { \mathrm { M O O N } } \big ) < \ell _ { i } \big ( \Theta _ { 0 } - \eta P _ { \mathrm { M G D A } } \big ) , \quad } \\ & { } & { i \in \{ 1 , 2 \} , \quad 0 < \eta < \frac { 4 } { 3 } . } \end{array}\tag{75}
$$

This example demonstrates that the advantage of MOON is not merely caused by a larger update magnitude. Under the same spectral-norm budget, its matrix-aware direction provides a strictly stronger common descent projection, a tighter worst-case quadratic upper bound, and a larger exact decrease for a nontrivial range of step sizes.

## F Experimental Details & Further Experiments

## F.1 Benchmarks and Baselines

We consider five benchmarks widely used in multi-task learning research: MultiMNIST [Sabour et al., 2017], NYU-v2 [Silberman et al., 2012], CityScapes [Cordts et al., 2016], QM9 [Blum and Reymond, 2009] and CelebA [Liu et al., 2015]. Specifically, NYU-v2 is an indoor RGB-D dataset, comprising 1449 aligned RGB and depth images with pixel-level semantic annotations over 13 classes. In NYU-v2 benchmark, we consider image segmentation, depth prediction and surface normal prediction as three tasks. CityScapes is an urban street-scene benchmark with 5000 finely annotated stereo RGB image pairs providing dense per-pixel semantic labels for scene understanding. MultiMNIST is an MTL variant of MNIST dataset, where each image contains two randomly sampled MNIST digits. In MultiMNIST, we treat classifying each digit as a separate task. QM9 is a quantum-chemistry benchmark of over 130k small organic molecules with DFT-computed 3D geometries, widely used for molecular property prediction and multi-task learning. The objective is to predict 11 molecular properties. CelebA is a large-scale face attribute dataset containing over 200k facial images annotated with 40 binary attributes. We treat the prediction of each attribute as an individual task, resulting in a 40-task multi-task classification problem.

We compare the proposed method with 13 multi-objective optimization baselines: (1) Singletask learning (STL), training each task separately; (2) Linear Scalarization (LS), assigning equal weights to all tasks and minimizing the resulting weighted loss; (3) a scale-invariant baseline (SI) that applies equal weighting to the log losses; (4) Random Loss Weighting (RLW) [Lin et al., 2021], sampling task weights from a normal distribution; (5) Dynamic Weight Average (DWA) [Liu et al., 2019], adjusting task weights according to the rates of loss change; (6) Uncertainty Weighting (UW) [Kendall et al., 2018], incorporating task uncertainty to determine loss weights; (7) MGDA [Sener and Koltun, 2018], finding a common descent direction that yields balanced improvement across tasks at each step; (8) GradNorm [Chen et al., 2018], dynamically balancing task losses by adjusting their weights according to relative training rates and gradient magnitudes;(9) PCGrad [Yu et al., 2020], projecting the gradient of each task to that of other tasks before aggregation; (10) CAGrad [Liu et al., 2021a], minimizing the average loss while controlling the minimum decrease across tasks; (11) IMTL-G [Liu et al., 2021b], learning task weights such that the aggregated gradient has equal projection onto each task gradient; (12) Nash-MTL [Navon et al., 2022], formulating multi-objective optimization as a bargaining game and finding a solution that benefits all tasks; (13) FAMO [Liu et al., 2024], maximizing the worst-case improvement rate and using log losses to update task weights.

## F.2 Comparison with Naive Combinations with Muon

To examine whether MOON can be replaced by Muon itself or by directly combining existing multi-objective optimization methods with Muon, we additionally evaluate Muon by using it as the optimizer under standard linear scalarization, as Muon itself can already achieve competitive MTL performance without an explicit MOO mechanism. We further construct two baselines, MGDA+Muon and FAMO+Muon, by retaining the original multi-objective gradient aggregation of MGDA and FAMO while replacing Adam with Muon for parameter updates. We evaluate all methods on MultiMNIST with the ViT backbone under the same setting as the main experiments.

<table><tr><td>Method</td><td>Left acc</td><td>Right acc</td><td>Average acc</td></tr><tr><td>MGDA</td><td>95.58</td><td>94.10</td><td>94.84</td></tr><tr><td>MGDA+MUON</td><td>94.61</td><td>93.65</td><td>94.13</td></tr><tr><td>FAMO</td><td>95.89</td><td>94.88</td><td>95.39</td></tr><tr><td>FAMO+MUON</td><td>95.61</td><td>94.33</td><td>94.97</td></tr><tr><td>MUON</td><td>95.39</td><td>94.82</td><td>95.11</td></tr><tr><td>MOON (OURS)</td><td>95.99</td><td>95.31</td><td>95.65</td></tr></table>

Table 5: Performance comparison on MultiMNIST. Per-task test accuracies (%) and their average are reported. Each experiment is repeated over 3 random seeds and the mean is reported.

As shown in Table 5, Muon itself also achieves competitive performance under linear scalarization, but still falls short of MOON. Replacing Adam with Muon degrades the performance of both MGDA and FAMO, while MOON achieves the best results. This shows that simply combining traditional multi-objective methods with Muon is insuficient. MOON instead formulates multi-objective optimization directly over matrix-valued parameters, jointly accounting for objective trade-ofs and matrix update geometry.

## F.3 Transformer Experiment on CelebA

To further evaluate MOON on a larger dataset with a Transformer architecture, we conduct experiments on CelebA using a ViT backbone. CelebA contains 40 binary attribute prediction tasks, providing a substantially larger multi-objective setting than MultiMNIST. We compare MOON with representative multi-objective optimization methods under the same experimental setting.

<table><tr><td>METHOD</td><td> $\Delta m \% \downarrow$ </td></tr><tr><td>MGDA</td><td>19.20</td></tr><tr><td>PCGRAD</td><td>10.48</td></tr><tr><td>CAGRAD</td><td>12.61</td></tr><tr><td>IMTL-G</td><td>7.96</td></tr><tr><td>NASH-MTL FAMO</td><td>8.06 7.75</td></tr><tr><td>MOON (OURS)</td><td>6.09</td></tr></table>

Table 6: Performance comparison on CelebA with a ViT backbone. Lower $\Delta m \%$ indicates better overall multi-task performance. Each experiment is repeated over 3 random seeds and the mean is reported.

As shown in Table 6, MOON achieves the lowest $\Delta m \%$ among all compared methods. This demonstrates that MOON remains efective on a larger multi-task dataset with a Transformer architecture and scales well to a setting with 40 objectives.

## F.4 Multi-Objective Reinforcement Fine-Tuning on Qwen3-1.7B-Base

To evaluate MOON beyond conventional supervised multi-task learning, we apply it to multi-objective reinforcement fine-tuning of Qwen3-1.7B-Base on MATH500, following the setup of Figure 1(c) in Lu and Jiang [2026]. The 500 problems are split into 300 training, 100 validation, and 100 test examples. The training split is used exclusively for policy optimization, while the validation split is used for monitoring, hyperparameter tuning, and checkpoint selection under a prespecified selection rule. The test split is withheld during training and model selection and is used only to evaluate the validation-selected checkpoint. Accordingly, Figure 5 reports validation learning curves, whereas final performance is evaluated on the held-out test split.

REINFORCE is used to jointly optimize three alignment objectives: accuracy, conciseness, and clarity. All three objectives are implemented as verifiable binary rewards during training. Accuracy receives a reward of one when the final answer extracted from the last valid \boxed{} expression matches the ground-truth answer after standard MATH normalization, and zero otherwise. Conciseness receives a reward of one when the generated response is shorter than the running global average response length, and zero otherwise. Clarity receives a reward of one when the response contains a nonempty boxed answer and a step-by-step structure indicated by at least two enumerated steps or ordinal transitions, and zero otherwise. During evaluation, accuracy and clarity are reported as mean binary reward scores, while conciseness is reported as the mean response length in tokens, with lower values preferred.

We compare MOON with Linear, GradNorm, MGDA, Nash-MTL, and FAMO. For MOON, the objective-specific policy-gradient signals are aggregated through the proposed matrix-aware multi-objective update to optimize the shared LLM parameters.

As shown by the validation trajectories in Figure 5, MOON rapidly reaches a favorable tradeof across the three objectives. Its clarity score rises faster than those of most baselines, while accuracy remains competitive and response length decreases steadily. On the held-out test set, the validation-selected MOON checkpoint maintains high clarity and competitive accuracy while producing substantially shorter responses. These results demonstrate that MOON remains efective for multi-objective reinforcement fine-tuning of a modern LLM and reaches a strong multi-objective solution with relatively few optimization steps.

## F.5 Ablation on Key Hyperparameters

We study the sensitivity of MOON to two key hyperparameters, the task-weight update stepsize $\beta$ and the weight decay coeficient $\gamma ,$ on MultiMNIST with the ViT backbone. We vary one hyperparameter at a time while keeping the remaining settings unchanged. Each experiment is repeated over 3 random seeds and the mean is reported.

<table><tr><td>γ</td><td> $1 0 ^ { - 2 }$ </td><td> $5 \times 1 0 ^ { - 2 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>AVERAGE ACC 95.14</td><td></td><td>95.33</td><td>95.65</td><td>95.62</td><td>95.28</td></tr></table>

Table 7: Average accuracy (%) on MultiMNIST with diferent values of $\gamma .$

<table><tr><td>β</td><td> $1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>AVERAGE ACC</td><td>95.11</td><td>95.53</td><td>95.65</td><td>95.48</td><td>94.93</td></tr></table>

Table 8: Average accuracy (%) on MultiMNIST with diferent values of $\beta .$

As shown in Tables 7 and 8, MOON maintains stable performance across a range of $\beta$ and $\gamma _ { : }$ indicating that it is not sensitive to moderate variations of these hyperparameters.

## F.6 Ablation on Key Components

Orthonormalized Updates. We conduct an ablation study on the MultiMNIST experiment to evaluate the impact of the orthonormalized updates. We compare MOON with that using the Euclidean-direction updates while retaining the matrix-norm-oriented task weighting.

As shown in Table 9, without the orthonormalized update, using only matrix-norm-oriented task weighting leads to inferior performance. This indicates that the orthonormalized updates play an important role in the efectiveness of MOON.

Momentum. We further conduct an ablation study on the momentum mechanism by directly constructing the update from the current aggregated gradient without momentum smoothing.

As shown in Table 9, removing momentum results in a clear performance degradation. This indicates that the momentum is practically necessary.

## F.7 30-task Synthetic Data Regression Experiment

To better understand steepest descent for matrix-valued parameters, we construct a multi-objective regression problem on synthetic data. We adopt an MLP with parameters including weight matrices of its hidden linear layers. We randomly generate 20-dimensional inputs and 30-dimensional targets, treating the regression of each target dimension as a separate task. The average square loss $\begin{array} { r } { { \cal L } = \frac { 1 } { n d } \sum _ { i = 1 } ^ { n } \| { \pmb y } _ { i } - { \pmb \hat { y } } _ { i } \| _ { 2 } ^ { 2 } } \end{array}$ across these tasks is used to demonstrate training eficiency, where n represents the sample size and d represents the output dimension. We compare the proposed MOON method with traditional Euclidian-based gradient manipulation methods like MGDA and FAMO.

<table><tr><td>Method</td><td>Left</td><td>Right</td><td>Average</td></tr><tr><td>MOON</td><td>95.99</td><td>95.31</td><td>95.65</td></tr><tr><td>(with Euclidean update)</td><td>95.18</td><td>94.42</td><td>94.80</td></tr><tr><td>(without momentum)</td><td>93.96</td><td>93.44</td><td>93.70</td></tr></table>

Table 9: Ablation on the orthonormalized updates and momentum on MultiMNIST. Test accuracies (%) for the left and right digit classification tasks and their average are reported.

![](images/d1a33230ad186fc71fd63cd3ee04dfb19d68fcd2576067782b472d8964c43532.jpg)  
Figure 4: Average squared loss after 1000 training steps on the synthetic data multi-objective regression problem.

As shown in Figure 4, MOON converges faster and attains a lower average training loss than Euclidean-based gradient manipulation baselines. We observe that in the early stage, MOON quickly opens a clear gap over MGDA and FAMO, and this advantage persists throughout training. After 1k steps, MOON reduces the final average loss by about 20% compared to FAMO and about 15% to MGDA.

## F.8 Training Eficiency Comparison

We evaluate the training eficiency of MOON on MultiMNIST by tracking the average training cross-entropy (CE) loss against wall-clock time, with MGDA and FAMO as baselines. As reported in Table 10, MOON attains a lower training loss under the same time budget throughout training. The gap widens at later stages: after 1,000 seconds, the average CE loss reaches 0.144 with MOON, compared with 0.215 for MGDA and 0.179 for FAMO.

![](images/f3ff2e8f7c98acf86660753c9882c79930ba233a2042294b068fb402ed761450.jpg)  
(a) Accuracy

![](images/aa04693623418d5ba692ad51258ba5fdac30718be22684a0b1387bd8fe4b7006.jpg)  
(b) Clarity

![](images/2ef122556bde8e8ebd07a82744587fe3aa6548df2df3721450ed77786483ae89.jpg)  
(c) Response length

Figure 5: Multi-objective reinforcement fine-tuning of Qwen3-1.7B-Base on MATH500 following the setting of Figure 1(c) in Lu and Jiang [2026]. We report accuracy, clarity, and response length throughout training, where response length serves as the measure of conciseness. Higher accuracy and clarity are better, while shorter responses indicate better conciseness.
<table><tr><td>TIME (s) MGDA</td><td>FAMO</td><td>MOON</td></tr><tr><td>200 0.500</td><td>0.368</td><td>0.362</td></tr><tr><td>400 0.339</td><td>0.277</td><td>0.259</td></tr><tr><td>600 0.273</td><td>0.222</td><td>0.199</td></tr><tr><td>800 0.229</td><td>0.188</td><td>0.164</td></tr><tr><td>1000 0.215</td><td>0.179</td><td>0.144</td></tr></table>

Table 10: Average training CE loss vs. training time.

<table><tr><td>METHOD</td><td>TIME (s)</td></tr><tr><td>MGDA</td><td>1576</td></tr><tr><td>FAMO</td><td>1105</td></tr><tr><td>MOON</td><td>949</td></tr></table>

Table 11: Training time required to reach an average CE loss of 0.15.

We also measure the wall-clock time required to reduce the average training CE loss below a common threshold. As shown in Table 11, MOON reaches an average CE loss of 0.15 approximately 39.8% faster than MGDA and 14.1% faster than FAMO. Thus, the additional computation introduced by MOON is ofset by faster convergence, resulting in lower end-to-end training time.

We further compare GPU memory usage on MultiMNIST. MGDA, FAMO, and MOON consume 2,076 MB, 2,076 MB, and 2,078 MB, respectively. This indicates that MOON exhibits no significant diference in memory overhead compared with the other MOO methods.

Overall, MOON improves wall-clock convergence while retaining essentially the same memory footprint as MGDA and FAMO.

## F.9 Experimental Results with Error Bars

In the following tables 12 to 15, we report the results of MOON with error bars.

<table><tr><td>Method</td><td></td><td></td><td>Left Acc ↑ Right Acc ↑ Average Acc ↑</td></tr><tr><td>MOON (mean)</td><td>95.99</td><td>95.31</td><td>95.65</td></tr><tr><td>MOON (std)</td><td>±0.07</td><td>±0.10</td><td>±0.07</td></tr></table>

Table 12: Test accuracy (%) on MultiMNIST (2 tasks). Each experiment is repeated over 3 random seeds. The mean and the standard deviation is reported. Per-task accuracies $\left( \mathrm { l e f t / r i g h t } \right)$ and their average are reported. The best result is marked in bold.

<table><tr><td rowspan="3">Method</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal</td><td rowspan="3"> $\Delta m \% \downarrow$ </td></tr><tr><td>mIoU ↑ Pix Acc ↑</td><td></td><td></td><td>Abs Err ↓ Rel Err ↓</td><td>Mean ↓ Median ↓ 11.25° ↑ 22.5° ↑</td><td></td><td></td><td></td><td> $3 0 ^ { \circ } \uparrow$ </td></tr><tr><td>MOON (mean)</td><td>39.41</td><td>67.03</td><td>0.4891</td><td>0.2091</td><td>25.61</td><td>20.27</td><td>28.89</td><td>54.87</td><td>67.32</td><td>-4.63</td></tr><tr><td>MOON (std)</td><td>±0.48</td><td>±0.24</td><td>±0.0024</td><td>±0.0030</td><td>±0.11</td><td>±0.14</td><td>±0.19</td><td>±0.23</td><td>±0.22</td><td>±0.22</td></tr></table>

Table 13: Results on NYU-v2 (3 tasks) dataset. Each experiment is repeated over 3 random seeds. The mean and the standard deviation is reported. The best average result is marked in bold. The task specific metrics and the average performance drop $\Delta m \%$ are reported.

<table><tr><td></td><td colspan="5">CityScapes</td><td>CelebA</td></tr><tr><td></td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td rowspan="2"> $\Delta m \% \downarrow$ </td><td rowspan="2"> $\Delta m \% \downarrow$ </td></tr><tr><td>Method</td><td>mIoU↑</td><td>Pix Acc ↑</td><td>Abs Err ↓</td><td>Rel Err ↓</td></tr><tr><td>MOON (mean)</td><td>78.61</td><td>94.36</td><td>0.0126</td><td>31.41</td><td>1.54</td><td>4.65</td></tr><tr><td>MOON (std)</td><td>±0.24</td><td>±0.09</td><td>±0.0008</td><td>±1.00</td><td>±0.60</td><td>±0.08</td></tr></table>

Table 14: Results on CityScapes (2 tasks) and CelebA (40 tasks) dataset. Each experiment is repeated over 3 random seeds. The mean and the standard deviation is reported. The best average result is marked in bold. Task-specific metrics and the average performance drop $\Delta m \%$ are reported.

<table><tr><td>Method</td><td>μ</td><td>α</td><td>€HOMO €LUMO</td><td></td><td> $\langle R ^ { 2 } \rangle$ </td><td>ZPVE</td><td></td><td> $U _ { 0 }$ </td><td>U</td><td>H G</td><td> $c _ { v }$ </td><td> $\Delta m \% \downarrow$ </td></tr><tr><td colspan="10">MAE↓</td><td></td><td></td></tr><tr><td>MOON (mean)</td><td>0.07</td><td>0.24</td><td>52.9</td><td>71.0</td><td>3.35</td><td>5.67</td><td>46.4 46.7 46.9 46.3</td><td></td><td></td><td></td><td>0.08</td><td>49.9</td></tr><tr><td>MOON (std)</td><td> $\pm 0 . 0 0 6 \pm 0 . 0 0 9$ </td><td></td><td>±2.8</td><td>±2.3</td><td> $\pm 0 . 0 3 5 \pm 0 . 1 1 0 \pm 1 . 9 \pm 2 . 0 \pm 1 . 9 \pm 2 . 0 \pm 0 . 0 0 4$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>±1.60</td></tr></table>

Table 15: Results on QM9 dataset (11 tasks). Each experiment is repeated over 3 random seeds. The mean and the standard deviation is reported. The best average result is marked in bold. The task specific metrics and the average performance drop $\Delta m \%$ are reported.

## G Limitations

Our theoretical analysis assumes the exact computation of the polar factor, whereas our implementation uses a Newton–Schulz approximation. In practice, the approximation error is small, we therefore omit this error from the current analysis and leave its precise theoretical characterization to future work.