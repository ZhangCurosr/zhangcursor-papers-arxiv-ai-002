# OUTPERFORMANCE INVERSE OPTIMIZATION: LEARNING OBJECTIVE FUNCTIONS THAT OUTPERFORM AGENT DECISIONS

AKIRA KITAOKA

Abstract. Inverse optimization estimates the weights of an objective function that explain observed decisions as optimal solutions, and is used in a variety of fields. For mixed-integer linear programs (MILPs), existing methods aim to reproduce the observations as optimal solutions, and thus learn compromise weights when the observations are suboptimal. We propose outperformance inverse optimization, which instead seeks weights that induce, at each state, an optimal solution outperforming the observed action in every component. We give a loss function that can be evaluated with forward-problem oracles alone and is thus applicable to MILPs, together with gradient-based and DC optimization algorithms for minimizing it. For weights inducing a unique outperforming optimal solution at all observations, we prove that the probability of failing to induce such a solution at a new state (the generalization error) is bounded by a quantity inversely proportional to the number of observations, and that this bound is tight in the number of observations up to logarithmic factors. In experiments on synthetic and real data, the proposed methods improve the prediction of solutions outperforming the actions over existing methods.

## 1. Introduction

Many processes and systems, from human decision making to natural phenomena, can often be described by mathematical optimization. In practice, however, the true objective function of such a model is in many cases unknown in advance. Therefore, the inverse optimization problem (Ahuja and Orlin, 2001; Heuberger, 2004; Chan et al., 2019, 2025), which estimates the objective function from observed data, is of great practical importance.

Inverse optimization problems have been studied extensively (Ahuja and Orlin, 2001; Heuberger, 2004; Chan et al., 2019, 2025) and applied in a variety of fields, including geophysics (Tarantola, 2005; Burton and Toint, 1992), transportation (Bertsimas et al., 2015), power systems (Birge et al., 2017), scheduling of television advertisements (Suzuki et al., 2019), and healthcare (Chan et al., 2022). They have also been developed as a foundation of machine learning methods such as inverse reinforcement learning (Ng and Russell, 2000) and contrastive learning (Shi et al., 2023).

A challenge in inverse optimization is that the observed actions are not necessarily optimal solutions of the mathematical optimization under consideration (due to, e.g., measurement noise and the bounded rationality of the decision maker; Aswani et al., 2018). For such observations, tractable methods that relax the KKT conditions, the variational inequality, or the strong duality constraint (Keshavarz et al., 2011; Bertsimas et al., 2015; Chan et al., 2014, 2019) have been proposed, but all of them presuppose the convexity of the forward problem and cannot be extended to mixed-integer linear programs (MILPs) (Bärmann et al., 2018). On the other hand, methods (Bärmann et al., 2017, 2018; Kitaoka, 2024; Lin et al., 2024; Sakaue et al., 2025a,b; Kitaoka, 2026) that use losses evaluable only with an optimization oracle for the forward problem, such as the suboptimality loss (Mohajerin Esfahani et al., 2018), are applicable also to MILPs, but since they search for weights that explain the actions as optimal solutions, for suboptimal actions they learn compromise weights. Moreover, in designing mathematical models, an objective function that reproduces the observed actions may be required, but there are also situations that require one inducing solutions that outperform the observed actions in every indicator.

In this paper, we therefore propose a framework that learns weights inducing solutions that outperform the agent’s actions instead of reproducing them. The contributions of this paper are the following four.

Formulation and the OS loss We formulate the problem of finding weights that give, at each state, an optimal solution outperforming the agent’s action in every component as outperformance inverse optimization, and as its surrogate loss we introduce the outperformance suboptimality (OS) loss, which can be evaluated using only optimization oracles for the forward problem and its dominance-constrained version (§3 and §4).

Minimization methods As methods for minimizing the OS loss, we propose gradient-based optimization methods, including the projected subgradient method, and a DC (diference of convex) optimization method that terminates after finitely many iterations if the inner convex subproblems are solved exactly (§5). Moreover, when the feasible region of each state is an M<sup>♮</sup>-convex set (including M-convex sets) and the weight space is the probability simplex, we show that the uniform weight makes the OS loss 0 at all observations and thus attains its global minimum (Proposition F.24, Corollary F.32).

Generalization error For weights inducing a unique outperforming optimal solution at all observations, we show that the probability of failing to induce such a solution at a new state (the generalization error) is bounded by a quantity inversely proportional to the number of observations N, and via a lower bound that this upper bound is tight in N up to log factors (§6).

Experiments In five experiments with synthetic and real data, we show that, with the optimization method fixed, replacing the suboptimality loss of existing inverse optimization with the OS loss improves the prediction of solutions outperforming the actions (smaller componentwise shortfalls): indeed, the OS-loss variant wins in 14 of the 15 (optimization method, experiment) pairs (§7).

## 2. Related work

Details, including classical inverse optimization, are given in §A.

Inverse optimization for noisy data. As data-driven inverse optimization that handles the case where the observed actions are not optimal solutions, methods have been proposed that use as the loss the distance between the observations and the optimal solutions (Aswani et al., 2018) (NP-hard even when the forward problem is convex), the residual of the KKT conditions (Keshavarz et al., 2011), the violation of the variational inequality (Bertsimas et al., 2015), the relaxation of the strong duality constraint (Chan et al., 2014, 2019), and distributionally robust optimization of the suboptimality loss (Mohajerin Esfahani et al., 2018). However, since these tractable reformulations use the KKT conditions or duality, they presuppose the convexity of the forward problem and cannot be extended to the integer case (Bertsimas et al., 2015; Bärmann et al., 2018).

Inverse optimization for MILPs. Methods that minimize the suboptimality loss using only the forward-problem oracle do not require the convexity of the feasible region and are therefore applicable to MILPs (Bärmann et al., 2017, 2018; Kitaoka, 2024; Lin et al., 2024; Sakaue et al., 2025a,b; Kitaoka, 2026). However, since these seek weights that explain the agent’s actions as optimal solutions, when the actions are suboptimal the suboptimality loss is positive even at the true weight (Sakaue et al., $2 0 2 5 \mathrm { a } , \mathrm { b } )$

Generalization error analysis of inverse optimization. Kitaoka (2025) bounded the expected suboptimality loss by $O ( \sqrt { d / N } )$ for inverse optimization of MILPs that learns both the objective function and the constraints (d is the dimension of the weights and N is the number of observations). Fatemi et al. (2026) bounded, for noiseless inverse optimization in which the action is the unique optimal solution under the true weight, the probability that the true optimal solution ceases to be optimal under the learned weight by $O ( d / N )$ , and showed that this order is tight. Both presuppose that the agent’s actions are optimal solutions.

Position of this paper. Rather than explaining the agent’s actions as optimal solutions, this paper seeks weights that induce optimal solutions outperforming the actions in every component. Since the OS loss, like the suboptimality loss, can be evaluated using only the forward-problem oracle, it is applicable to nonconvex forward problems including ${ \mathrm { M I L P s } } ,$ and it can learn weights inducing solutions that outperform the actions even when the actions are suboptimal. In addition, the generalization error analysis (§6) evaluates, in a setting where the actions may be suboptimal, the probability that a unique optimal solution outperforming the action is not obtained at a new state.

## 3. Problem setting

We consider an ofline learning setting with two players, a learner and an $a g e n t . ^ { 1 }$ Let d be a positive integer, and let $\mathbb { R } ^ { d }$ be the space on which the forward optimization is defined. We call a nonempty set $s$ the state set, and for each state $s \in { \mathcal { S } }$ we denote the feasible region by $\mathcal { X } ( s ) ( \subseteq \mathbb { R } ^ { d } )$ . For a weight $\theta \in \mathbb { R } ^ { d }$ and a state $s \in S$ we write the forward problem (linear optimization) and its optimal solution as

$$
x ^ { * } ( \theta , s ) : \in \arg \operatorname* { m a x } _ { x \in \mathcal { X } ( s ) } \langle \theta , x \rangle .\tag{3.1}
$$

Let $\Theta \subseteq \mathbb { R } _ { > 0 } ^ { d } ( : = [ 0 , \infty ) ^ { d } )$ be the weight space.

For $n = 1 , \ldots , N .$ , given a state $s ^ { n } \in S$ , the agent chooses $x ^ { n } \in { \mathcal { X } } ( s ^ { n } )$ as its action. From the observations $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$ , we wish to find a weight $\theta ^ { * }$ satisfying, at each state $s ^ { n }$

$$
x ^ { * } ( \theta ^ { * } , s ^ { n } ) \geq x ^ { n } ,\tag{3.2}
$$

where $\geq$ means that $\geq$ holds componentwise. Hereafter, when $x \geq x ^ { \prime }$ , we say that x dominates (outperforms) $x ^ { \prime }$ . We call solving Equation (3.2) outperformance inverse optimization (outperformance IO). Each component of x is oriented so that larger is better (e.g., times enter with reversed sign), and observation noise is assumed to keep $x ^ { n }$ feasible.

Note that the set $\mathcal { X } ( s )$ is not necessarily convex. If $\mathcal { X } ( s )$ is the feasible region of a MILP, the solution of any MILP solver, e.g., Gurobi (Gurobi Optimization, LLC, 2026) or SCIP via OR-Tools (Perron and Furnon, 2026), can serve as the oracle $x ^ { * } ( \theta , s )$

## 4. Performance metrics

For $s \in S$ , we define an optimal solution that dominates $x ^ { \prime } \in \mathcal { X } ( s )$ as follows (we call this problem the dominance-constrained forward problem):

$$
x ^ { * } ( \theta ; x ^ { \prime } , s ) : \in \mathop { \operatorname { a r g m a x } } _ { x \in \mathcal { X } _ { > x ^ { \prime } } ( s ) } \langle \theta , x \rangle , \quad \mathcal { X } _ { \geq x ^ { \prime } } ( s ) : = \{ x \in \mathcal { X } ( s ) | x \geq x ^ { \prime } \} .\tag{4.1}
$$

As a metric useful for outperformance inverse optimization, we define the outperformance suboptimality loss (OS loss) $\ell ^ { \mathrm { O S } }$ for any $s \in { \mathcal { S } }$ and data $x ^ { \prime } \in \mathcal { X } ( s )$ by

$$
\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) : = \operatorname* { m a x } _ { x \in \mathcal { X } ( s ) } \langle \theta , x \rangle - \operatorname* { m a x } _ { x \in \mathcal { X } _ { \geq x ^ { \prime } } ( s ) } \langle \theta , x \rangle .\tag{4.2}
$$

By the inclusion $\mathcal { X } ( s ) \supseteq \mathcal { X } _ { \geq x ^ { \prime } } ( s )$ , the OS loss $\ell ^ { \mathrm { O S } }$ is nonnegative. The OS loss $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s )$ is 0 if and only if there exists an optimal solution $x \in \arg \operatorname* { m a x } _ { x \in { \mathcal { X } } ( s ) } \langle \theta , x \rangle$ satisfying $x \geq x ^ { \prime }$ . In particular, if no point of $\chi _ { \geq x ^ { \prime } } ( s )$ is optimal under any $\dot { \theta } \in \Theta$ $\left( \mathrm { e . g . } \right.$ ., an eficient but unsupported action; §B), no weight attains Equation (3.2). However, when the optimal solution is not unique, the optimal solution $x ^ { * } ( \theta , s )$ returned by the oracle does not necessarily dominate $x ^ { \prime }$ . To solve Equation (3.2), we introduce the empirical mean of the OS loss

$$
\bar { \ell } ^ { \mathrm { { O S } } } ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell ^ { \mathrm { { O S } } } ( \theta ; x ^ { n } , s ^ { n } ) .\tag{4.3}
$$

The empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } } ( \theta )$ is 0 if and only if, for each $n ,$ there exists an optimal solution $x \in \arg \operatorname* { m a x } _ { x \in { \mathcal { X } } ( s ^ { n } ) } \langle \theta , x \rangle$ that dominates $x ^ { n }$ . In particular, if θ satisfies Equation (3.2), then $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) = 0$ since $x ^ { \ast } ( \theta , s ^ { n } ) \in \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } )$ . Conversely, even if $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) = 0$ , when the optimal solution is not unique, $x ^ { * } ( \theta , s ^ { n } )$ returned by the oracle does not necessarily dominate $x ^ { n }$ , and attaining Equation (3.2) is not guaranteed. Accordingly, since a weight satisfying Equation (3.2) attains the minimum value 0 of the empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } }$ , we treat Equation (3.2) as the following problem using the empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } }$

$$
{ \mathrm { m i n i m i z e } } { \bar { \ell } } ^ { \mathrm { O S } } ( \theta ) { \mathrm { ~ s u c h ~ t h a t ~ } } \theta \in \Theta .\tag{4.4}
$$

Remark 4.1. The suboptimality loss is defined by $\textstyle \ell ^ { \mathrm { s u b } } ( \theta ; x ^ { \prime } , s ) : = \operatorname* { m a x } _ { x \in { \mathcal { X } } ( s ) } \langle \theta , x \rangle -$ $\langle \theta , x ^ { \prime } \rangle$ (Mohajerin Esfahani et al., 2018).

To solve outperformance inverse optimization, we introduce the following assumption.

Assumption 4.2. (1): The weight space $\Theta \subset \mathbb { R } _ { > 0 } ^ { d }$ is a nonempty bounded polyhedron $\begin{array} { r } { \Theta = \{ \theta \in \mathbb { R } _ { > 0 } ^ { d } \mid A _ { \Theta } \theta \leq b _ { \Theta } \} } \end{array}$ with $\boldsymbol { A } _ { \Theta } ^ { - } \in \mathbb { R } ^ { m _ { \Theta } \times d }$ and $b _ { \Theta } \in \mathbb { R } ^ { m _ { \Theta } }$ ， and $\mathbf { 0 } \not \in \Theta$ (in the implementation, we take the probability simplex $\Delta ^ { d - 1 } : =$ $\{ \theta \in \mathbb { R } _ { > 0 } ^ { d } \mid \sum _ { i = 1 } ^ { d } \theta _ { i } = 1 \} )$

(2): For each state $s \in \mathcal { S } , \mathcal { X } ( s ) \subset \mathbb { R } ^ { d }$ is a finite union of nonempty bounded polyhedra (we call each bounded polyhedron constituting the union a piece). Moreover, $\mathcal { X } ( s )$ is uniformly bounded in s, that is, $\begin{array} { r } { \operatorname* { s u p } _ { s \in \mathcal { S } } \operatorname* { s u p } _ { x \in \mathcal { X } ( s ) } \| x \| < } \end{array}$ ∞.

(3): (Extreme-point oracle) For any $\theta \in \mathbb { R } ^ { d }$ , the forward-problem oracle $x ^ { * } ( \cdot , s )$ and the dominance-constrained forward-problem oracle $x ^ { * } ( \cdot ; x ^ { \prime } , s )$ return a vertex of a bounded polyhedral piece attaining the maximum.

We denote the sets of all values that the forward-problem oracle $x ^ { * } ( \cdot , s )$ and the dominance-constrained forward-problem oracle $x ^ { * } ( \cdot ; x ^ { \prime } , s )$ can return by $\mathcal { V } ( s )$ and $\mathcal { V } _ { \geq x ^ { \prime } } ( s )$ , respectively.

Remark 4.3 (On Assumption 4.2). By $( 2 ) , \mathcal { X } ( s )$ is nonempty and compact, and for $x ^ { \prime } \in \mathcal { X } ( s )$ the dominance-constrained feasible region $\mathcal { X } _ { \geq x ^ { \prime } } ( s ) : = \{ x \in \mathcal { X } ( s ) \mid x \geq x ^ { \prime } \}$ (Equation (4.1)) is also a finite union of nonempty bounded polyhedra.

(3) is a natural assumption as the standard behavior of LP and MILP solvers based on the simplex method and branch-and-bound. Moreover, $\mathcal { V } ( s )$ and $\mathcal { V } _ { \geq x ^ { \prime } } ( s )$ are finite subsets of $\mathcal { X } ( s )$ , since they consist of vertices of the bounded polyhedral pieces.

## 5. Proposed methods

## 5.1. Gradient-based optimization methods.

Proposition 5.1. Assume Assumption $4 . 2 ( 2 )$ . Let $s \in S$ and $x ^ { \prime } \in \mathcal { X } ( s )$ . Then $g ( \theta ; x ^ { \prime } , s ) : = x ^ { * } ( \theta , s ) - x ^ { * } ( \theta ; x ^ { \prime } , s )$ is a gradient of $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s )$ with respect to θ in the sense of weak derivatives (derivatives in the sense of Schwartz distributions; cf. Kunzinger, 2019).<sup>2</sup>

For the proof, see $\ S { \mathrm { C } } .$ By Proposition 5.1, a gradient of the empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } }$ in the sense of weak derivatives is given by $\begin{array} { r l } { { } } & { { } { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } g ( \theta ; x ^ { n } , s ^ { n } ) } \end{array}$ . Therefore, gradient-based optimization methods can be applied to solve Equation (4.4), although they may stall at a local minimum or a stationary point without reaching the minimum value. In particular, in its simplest form, the projected subgradient method for the OS loss is the update $\begin{array} { r } { \theta ^ { k + 1 }  \Pi _ { \Theta } \big ( \theta ^ { k } - \frac { \alpha _ { k } } { N } \sum _ { n = 1 } ^ { N } \bigl ( x ^ { * } ( \theta ^ { k } , s ^ { n } ) - x ^ { * } ( \theta ^ { k } ; x ^ { n } , s ^ { n } ) \bigr ) \big ) } \end{array}$ $( \Pi _ { \Theta }$ is the Euclidean projection onto Θ and $\alpha _ { k } > 0$ is the learning rate at iteration $k )$ , which difers from that for the suboptimality loss (Kitaoka, 2024) only in that the observed action $x ^ { n }$ is replaced by $x ^ { * } ( \theta ^ { k } ; x ^ { n } , s ^ { n } )$ (Algorithm 8). In this paper, as gradient-based optimization methods, we adopt the projected subgradient method (PSG) (cf. Boyd et al., 2003), the Online Newton Step (ONS) (Hazan et al., 2007), and MetaGrad (van Erven and Koolen, 2016). For the Online Newton Step and MetaGrad, we adapt the implementation that Sakaue et al. (2025b) used for online inverse linear optimization. Although these are online methods, we update them with the average of the gradients over all states at each iteration (full-batch). Implementation details are given in §H.

5.2. DC optimization. We call a tuple of dominating feasible solutions $y ^ { \prime } =$ $( y ^ { \prime n } ) _ { n = 1 } ^ { N } \ ( y ^ { \prime n } \in \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) )$ ) a branch (not to be confused with branching in branchand-bound). Fixing a branch replaces the second term of Equation (4.2) (concave in θ) with the linear function $\left. { \theta , y ^ { \prime n } } \right.$ , which yields the convex upper bound on the empirical mean of the OS loss

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Bigl [ \operatorname* { m a x } _ { x \in \mathcal { X } ( s ^ { n } ) } \langle \theta , x \rangle - \langle \theta , y ^ { \prime n } \rangle \Bigr ] \ \geq \ \bar { \ell } ^ { \mathrm { O S } } ( \theta ) .\tag{5.1}
$$

We call $\mathrm { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } )$ the convex subproblem of the branch $y ^ { \prime } .$ . We denote the set of all branches by $\begin{array} { r } { \mathcal { X } _ { \geq } ^ { \mathrm { a l l } } : = \prod _ { n = 1 } ^ { N } \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) } \end{array}$ and the set of those consisting of values that the dominance-constrained forward-problem oracle can return by $\begin{array} { r } { \mathcal { V } _ { \ge } ^ { \mathrm { a l l } } : = \prod _ { n = 1 } ^ { N } \mathcal { V } _ { \ge x ^ { n } } ( s ^ { n } ) } \end{array}$ , and we let $m _ { \mathrm { s u b } } ( y ^ { \prime } ) : = \mathrm { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } )$ be the minimum value of the convex subproblem of a branch $y ^ { \prime } \in \mathcal { X } _ { \geq } ^ { \mathrm { a l l } }$ (the minimum is attained since Θ is compact and $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ is continuous).

Proposition 5.2 (Decomposition into branches). Under Assumption 4.2, for any $\theta \in \mathbb { R } ^ { d }$ , we have

$$
\bar { \ell } ^ { \mathrm { { O S } } } ( \theta ) = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { X } _ { \geq } ^ { \mathrm { a l } } } \bar { \ell } ^ { \mathrm { { s u b } } } ( \theta ; y ^ { \prime } ) = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { Y } _ { \geq } ^ { \mathrm { a l } } } \bar { \ell } ^ { \mathrm { { s u b } } } ( \theta ; y ^ { \prime } ) .\tag{5.2}
$$

Therefore, for the finite set $y _ { \ge } ^ { \mathrm { a l l } }$ , we have

$$
\operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { O S } } ( \theta ) = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { Y } _ { > } ^ { \mathrm { a l l } } } m _ { \mathrm { s u b } } ( y ^ { \prime } ) .\tag{5.3}
$$

The proof is given in §E.2.

Proposition 5.2 specializes to our problem the result of Sections 4.1–4.2 of Pham Dinh and Le Thi (1997) that a global solution of a polyhedral DC program reduces to finitely many convex programs (the DC structure of the empirical mean of the OS loss is described in Remark E.1, and the correspondence in §E.5). Our DC optimization uses this structure directly. By Equation (5.2), Equation (4.4) is equivalent to the joint minimization of $\bar { \ell } ^ { \mathrm { s u b } } ( \boldsymbol { \theta } ; \boldsymbol { y } ^ { \prime } )$ over $( \theta , y ^ { \prime } ) \in \Theta \times \mathcal { V } _ { > } ^ { \mathrm { a l l } }$ , and Algorithm 6 in the appendix (§E) performs this joint minimization alternately in $y ^ { \prime }$ and θ. The minimization in $y ^ { \prime }$ amounts to updating the branch by calling the dominance-constrained forward-problem oracle at each state, and the minimization in θ amounts to solving the convex subproblem of the branch. This alternating minimization is nothing but the simplified DCA (a simplified version of the DC Algorithm) of Pham Dinh and Le Thi (1997) (Remark E.1 and §E.5). Moreover, if we take the agent’s actions themselves as the branch, $\bar { \ell } ^ { \mathrm { s u b } } (  { \theta } ; ( x ^ { n } ) _ { n = 1 } ^ { N } )$ is the empirical mean of the suboptimality loss (Mohajerin Esfahani et al., 2018). Hence the first equality of Equation (5.2) means that the empirical mean of the OS loss is the pointwise minimum of the empirical means of the suboptimality loss obtained by replacing the agent’s actions with feasible solutions dominating them.

In this paper, we solve Equation (4.4) by a double-loop method with this simplified DCA as the outer loop and a solver such as the cutting-plane method (Kelley, 1960) (the inner solver) as the inner loop. Details such as the pseudocode are given in §E.

If the inner convex subproblems are solved exactly, this double-loop method terminates after finitely many iterations (Proposition E.4). If the optimal solutions of the dominance-constrained forward problems are unique at a point where Algorithm 6 stops because the branch is no longer updated (a fixed point of DCA), that point is a local minimizer of the empirical mean of the OS loss (Proposition E.5; the set of weights at which the optimal solution is not unique has measure 0, Proposition E.8). However, reaching the global minimum is not guaranteed. Since the OS loss is nonnegative, if the loss value becomes 0, it can be confirmed that the point is a global minimizer (Proposition E.9). In particular, when the feasible region of each state is an M<sup>♮</sup>-convex set (including M-convex sets) and $\Theta = \Delta ^ { d - 1 }$ , the uniform weight makes the empirical mean of the OS loss 0 and attains the global minimum (Proposition F.24, Corollary F.32).

## 6. Generalization error analysis

In this section, we evaluate whether weights learned from the observations induce a unique optimal solution outperforming the agent’s action also at states that were not observed, that is, the generalization of Equation (3.2). We show that the generalization error (Definition 6.2) of weights that give such optimal solutions at all observations is uniformly bounded by order $d / N$ up to log factors for fixed κ (Assumption 6.1) (in general, the bound contains a term d log $\kappa / N )$ , and that this upper bound is tight up to log factors. The proofs are given in $\ S \mathrm { G }$

Suppose that the pair $( s , x )$ of a state s and the agent’s action x at that state follows a distribution ${ \mathcal { P } } _ { : }$ , that the observations $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$ are i.i.d. samples from $\mathcal { P }$ , and we denote the probability with respect to the distribution of the observations by P (see §G.1 for details of the setting). We make the following assumption.

Assumption 6.1.

$$
\begin{array} { r } { ( \mathbf { 1 } ) \colon \kappa : = \mathrm { e s s } \operatorname* { s u p } _ { s } | \mathcal { V } ( s ) | \mathrm { ~ s a t i s f i e s ~ } 2 \leq \kappa < \infty . } \end{array}
$$

(2): (Realizability) There exists $\theta ^ { \ast } \in \Theta$ such that, a.s. for $( s , x ) \sim \mathcal { P } _ { \mathrm { ~ } }$ , the optimal solution of the forward problem Equation (3.1) at the state s is unique and that optimal solution $x ^ { * } ( \theta ^ { * } , s )$ dominates the agent’s action x $( x ^ { \ast } ( \theta ^ { \ast } , s ) \geq x )$

We define the generalization error as follows.

Definition 6.2. For a weight $\theta \in \Theta$ , a state $s \in S$ , and data $\boldsymbol { x } ^ { \prime } \in \mathbb { R } ^ { d }$ , we define the outperformance error by

$$
\ell ^ { 0 1 } ( \theta ; x ^ { \prime } , s ) : = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } \operatorname { a r g m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle = \{ v \} \mathrm { ~ f o r ~ s o m e ~ } v \geq x ^ { \prime } , } \\ { 1 } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

and define the generalization error as the probability $\operatorname { G E } ( \theta ) : = { \mathcal { P } } { \big ( } \ell ^ { 0 1 } ( \theta ; x , s ) = 1 { \big ) }$ with respect to $( s , x ) \sim \mathcal { P }$ . We say that θ is consistent with the observations if $\ell ^ { 0 1 } ( \theta ; x ^ { n } , s ^ { n } ) = 0$ for all $n .$

The upper bound is as follows.

Theorem 6.3. Assume Assumption $4 . 2 ( 2 ) ( 3 )$ and Assumption 6.1(1), and let $N \geq d .$ For any $\beta \in ( 0 , 1 )$ , with probability at least $1 - \beta _ { \mathrm { { ; } } }$ , any weight $\hat { \theta } ^ { N } \in \Theta$ consistent with the observations (it may depend on the observations) satisfies

$$
\mathrm { G E } ( \hat { \theta } ^ { N } ) \leq \frac { 2 } { N } \left( d \log _ { 2 } \frac { 2 \mathrm { e } N \kappa ^ { 2 } } { d } + \log _ { 2 } \frac { 2 } { \beta } \right) ,
$$

where e is Napier’s constant.

Example 6.4 (Integer programming). In addition to the assumptions of Theorem $6 . 3 ,$ suppose that there exist positive integers $M _ { 1 } , \dots , M _ { d }$ such that $\mathcal { X } ( s ) \subseteq$

$\textstyle \prod _ { i = 1 } ^ { d } \{ 0 , 1 , \dots , M _ { i } \}$ holds a.s. Since $\begin{array} { r } { \kappa \leq \prod _ { i = 1 } ^ { d } ( M _ { i } + 1 ) } \end{array}$ by $\mathcal { Y } ( s ) \subseteq \mathcal { X } ( s )$ (Remark 4.3), Theorem 6.3 implies that, for any $\beta \in ( 0 , 1 )$ , with probability at least $1 - \beta _ { \mathrm { { ; } } }$ any weight $\hat { \theta } ^ { N } \in \Theta$ consistent with the observations satisfies

$$
\operatorname { G E } ( \hat { \theta } ^ { N } ) \leq \frac { 2 } { N } \left( d \log _ { 2 } \frac { 2 \mathrm { e } N } { d } + 2 d \sum _ { i = 1 } ^ { d } \log _ { 2 } ( M _ { i } + 1 ) + \log _ { 2 } \frac { 2 } { \beta } \right) .
$$

In particular, for 0-1 integer programming $( M _ { i } \ = \ 1 )$ , the second term in the parentheses on the right-hand side is $2 d ^ { 2 }$

As in Example 6.4, κ can depend exponentially on $d ,$ in which case the upper bound of Theorem 6.3 contains a term of order $d ^ { 2 } / N$

Since Theorem 6.3 does not depend on a specific estimator, it applies directly to the weight reached by the methods of $\ S 5$ whenever that weight has OS loss 0 and a unique optimal solution at all observations.

The lower bound is as follows. It is obtained by embedding the instance that Fatemi et al. (2026) constructed to show that their upper bound cannot be improved (Theorem 3.1 of that paper; the case where the agent’s action is the unique optimal solution under the true weight) into our setting, in which the weights are restricted to the probability simplex.

Theorem 6.5. For any integer $d _ { 0 } \geq 2 .$ , with $d = 2 d _ { 0 } + 1$ , there exists a tuple of a state set $s ,$ a distribution ${ \mathcal { P } } _ { : }$ , feasible regions $\mathcal { X } ( s )$ , and the weight space $\Theta \bar { = } \Delta ^ { d - 1 }$ that satisfies Assumption 4.2 and Assumption 6.1 with $\kappa = 3$ such that, for any $\beta \in ( 0 , 1 / 2 ]$ and $N \ge \operatorname* { m a x } \{ d _ { 0 } , \ln ( 1 / \beta ) \}$ , there exists a weight $\hat { \theta } ^ { N } \in \Theta$ consistent with the observations (depending on the observations) that satisfies, with probability at least $\beta _ { ; }$

$$
\mathrm { G E } ( \hat { \theta } ^ { N } ) > \frac { d _ { 0 } + \ln ( 1 / \beta ) } { 4 N } = \frac { d - 1 + 2 \ln ( 1 / \beta ) } { 8 N } .
$$

Since Theorem 6.3 is an upper bound holding with probability at least $1 - \beta$ and Theorem 6.5 is a lower bound holding with probability at least $\beta _ { ; }$ we compare them by applying Theorem 6.3 with $\beta / 2$ . Then, for $\beta \in ( 0 , 1 / 2 ]$ and $N \geq \operatorname* { m a x } \{ d , \ln ( 1 / \beta ) \}$ both events hold simultaneously with probability at least $\beta / 2$ , so the right-hand side of Theorem 6.3 with $\beta$ replaced by $\beta / 2 .$ , for $\kappa = 3$ and $d = 2 d _ { 0 } + 1$ , is larger than that of Theorem 6.5. This replacement only changes the constant inside the logarithm, so tightness up to log factors is preserved. That is, Theorem 6.5 guarantees the tightness of Theorem 6.3.

## 7. Numerical experiments

In this section, we compare outperformance inverse optimization (minimization of the empirical mean of the OS loss Equation (4.3)) with existing inverse optimization (minimization of the suboptimality loss) in terms of the degree to which Equation (3.2) is attained. The five experiments are the knapsack problem and the single-machine scheduling problem with release times using synthetic data, the movie selection problem with viewing costs using MovieLens100K (Harper and Konstan, 2015), and the weighted multi-cover problem (selection of reading lists and reference lists) using Cora (Sen et al., 2008) and ogbn-arxiv (Hu et al., 2020). The results of all methods, implementation details, hyperparameters, and the experimental environment are given in §H.

## 7.1. Experimental setup.

Compared methods. In the main text, we compare seven methods. Sub- and OSindicate minimizing the empirical mean of the suboptimality loss $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; ( x ^ { n } ) _ { n = 1 } ^ { N } )$ (Equation (5.1)) and the empirical mean of the OS loss, respectively, and the following name indicates the optimization method (PSG: projected subgradient method (cf. Boyd et al., 2003), ONS (Hazan et al., 2007), MetaGrad (van Erven and Koolen, 2016), (FB): the full-batch version using the average of the subgradients (weak derivatives for the OS loss) over all states at each iteration). The comparison of the three pairs consisting of the existing methods Sub-PSG (Kitaoka, 2024), Sub-ONS(FB), and Sub-MetaGrad(FB) (Sakaue et al., 2025b) and the proposed methods OS-PSG, OS-ONS(FB), and OS-MetaGrad(FB), each obtained by replacing only the loss function, represents the efect of the diference in the loss function. We call the OS- method of each pair the OS-loss variant and the Sub- method the suboptimality-loss variant. The remaining method, OS-DC, is the proposed method based on the DC optimization of §5.2. As the inner solver we use the proximal bundle method with cutting-plane finishing (§D.4), and after DC optimization stops, we apply the projected subgradient method with a small step size to the empirical mean of the OS loss, starting from the obtained weight (subgradient finishing; $\mathrm { O S - D C [ P B + C P F ] + S G F }$ in Table 3). For all methods, the initial point is the uniform weight $\theta = { \bf 1 } / d$ and the weight space is the probability simplex $\Theta = \Delta ^ { d - 1 }$ . The list of all 13 methods including these (Table 3), and the detailed settings and hyperparameters, are given in §H.

Evaluation metrics. The main evaluation metric is the outperformance prediction loss $\begin{array} { r } { \ell ^ { \mathrm { O P } } ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { d } \operatorname* { m a x } \{ x _ { i } ^ { n } - x _ { i } ^ { * } ( \theta , s ^ { n } ) , 0 \} } \end{array}$ (Equation (E.6)). That is, $\ell ^ { \mathrm { O P } }$ is the average over states of the sum of the componentwise shortfalls of the oracle’s solution relative to the agent’s action. We have $\ell ^ { \mathrm { O P } } ( \theta ) = 0$ if and only if $\theta$ is a solution of outperformance inverse optimization Equation (3.2). In addition, we record the attainment rate of $\ell ^ { \mathrm { O P } } = 0 ^ { \circ }$ (the fraction of episodes—the unit in which one set of observations $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$ is generated or extracted and all methods are run on it—that reached $\ell ^ { \mathrm { O P } } = 0 )$ and the empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } }$

Measure of computational cost. We measure the computational cost by the number of calls to the forward-problem oracles. Calls to $x ^ { * } ( \cdot , s )$ and to $x ^ { * } ( \cdot ; x ^ { \prime } , s )$ are both counted as one, and calls to $x ^ { * } ( \cdot ; x ^ { \prime } , s )$ , which the existing methods do not solve, are counted only on the side of the proposed methods. Letting $T$ be the number of iterations of OS-PSG (Sub-PSG, which makes half as many oracle calls per iteration, runs 2T iterations) and N the number of states, we impose a common oracle-call budget $B : = 2 T N$ on all methods, evaluate each method by the weight with the smallest $\ell ^ { \mathrm { O P } }$ among the weights visited so far (best-so-far), and compare the values at the point where the number of calls reaches B. Details of how the calls are counted and of the evaluation procedure are given in §H.5.

7.2. Overview of the experiments. In the main text, among the five experiments, we present the results of the knapsack problem with synthetic data and the weighted multi-cover problem with real data (Cora (Sen et al., 2008), ogbn-arxiv (Hu et al., 2020)). The results of the single-machine scheduling problem and the movie selection problem with viewing costs are given in §H.8, and the mathematical models and instances of all experiments in §H.1. The knapsack problem uses the bounded integer knapsack problem as the forward problem, and the agent’s action is obtained by decreasing each component of the optimal solution under the true weight $\theta ^ { * }$ by 1 with probability 0.2. Hence a weight $\theta ^ { * }$ satisfying Equation (3.2) exists. The weighted multi-cover problem models, as a weighted multi-cover of words, the situation in which a researcher selects a limited number of papers, and the agent’s action is the depth to which the set of papers actually in citation relations covers each word (the depth to which a word is covered is the number of selected papers containing that word, capped at an upper bound r). Except for the single-machine scheduling problem (N = 10, d ∈ {4, 6, 8}, B = 2000), we set $N = 2 0 , d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ and $T = 1 0 0 \ ( B = 4 0 0 0 )$ ; in each episode, all methods use the same instance.

Table 1. Comparison of the three pairs difering only in the loss function (OS-PSG vs. Sub-PSG, OS-ONS(FB) vs. Sub-ONS(FB), OS-MetaGrad(FB) vs. Sub-MetaGrad(FB)). The numbers of episodes on the same instances in which $\ell ^ { \mathrm { { \scriptsize { O P } } } }$ at the point where the number of calls reaches B is smaller/equal/larger for the OS-loss variant (wins/ties/losses). The larger of wins and losses is shown in bold.
<table><tr><td>Problem</td><td>PSG ONS(FB)</td><td>MetaGrad(FB)</td><td>Episodes</td></tr><tr><td>Knapsack</td><td>241/9/0 75/27/148</td><td>230/4/16</td><td>250</td></tr><tr><td>Single machine</td><td>24/51/0 30/42/3</td><td>28/47/0</td><td>75</td></tr><tr><td>Movie (MovieLens)</td><td> $\mathbf { 9 7 } / 2 / 1$  45/23/32</td><td>90/9/1</td><td>100</td></tr><tr><td>Multi-cover (Cora)</td><td>92/2/6 59/19/22</td><td>72/11/17</td><td>100</td></tr><tr><td>Multi-cover (ogbn-arxiv)</td><td>81/15/4 74/22/4</td><td>63/30/7</td><td>100</td></tr></table>

7.3. Experimental results. Table 1 shows, for all five experiments, the wins and losses of the three pairs difering only in the loss function; Table 2 shows the attainment rates for the knapsack problem; and Figure 1 shows the evolution of $\ell ^ { \mathrm { O P } }$ for the weighted multi-cover problem (that for the knapsack problem is shown in Figure 2 in the appendix). Overall, when the optimization method is fixed and only the suboptimality loss is replaced with the OS loss, the prediction of solutions outperforming the agent’s actions improves. Indeed, the OS-loss variant won in 14 of the 15 pairs. A similar tendency is observed in the comparison replacing only the loss function also for the single-machine scheduling problem and the movie selection problem with viewing costs in the appendix (§H.8).

Knapsack problem. In this problem, where the agent’s actions are degraded versions of optimal solutions, the improvement is the most pronounced. OS-PSG and OS-DC attained Equation (3.2) in at least 76% of the episodes in all dimensions, whereas the attainment rate of the methods with the suboptimality loss is at most 16% (Table 2; the attainment rates of Sub-ONS and Sub-MetaGrad, which update state by state (Table 3), are also at most 9% when aggregated over all dimensions; Table 8). Existing inverse optimization searches for weights that make the degraded actions optimal and therefore moves toward compromise weights that fit the degradation, whereas the OS loss measures the gap to the best feasible solution dominating the action and is thus presumably less afected by it. However, only in the ONS(FB) pair did the suboptimality-loss variant win (75 wins and 148 losses; Table 1). In this pair, the attainment rate is 0% for both losses for $d \geq 6 0$ (Table 2), presumably because the full-batch ONS does not converge within the oracle-call budget B.

Table 2. Knapsack problem: the fraction [%] of episodes that attained $\ell ^ { \mathrm { O P } } = 0$ at the point where the number of calls reaches $B = 4 0 0 0$ . 50 episodes for each d. For each optimization method, the OS loss (OS) and the suboptimality loss (Sub) are shown side by side, and the higher one in each pair is shown in bold. The DC optimization uses only the OS loss. The median of $\ell ^ { \mathrm { O P } }$ for each dimension is shown in Table 4.

<table><tr><td>d</td><td colspan="2">PSG</td><td>ONS(FB) OS Sub</td><td colspan="2">MetaGrad(FB) OS Sub</td><td>OS-DC</td></tr><tr><td>20</td><td>OS 98</td><td>Sub 16</td><td>34 0</td><td>86</td><td>6</td><td>90</td></tr><tr><td>40</td><td>96</td><td>0</td><td>2 0</td><td>52</td><td>0</td><td>88</td></tr><tr><td>60</td><td>92</td><td>0</td><td>0 0</td><td>46</td><td>0</td><td>86</td></tr><tr><td>80</td><td>90</td><td>0</td><td>0 0</td><td>8</td><td>0</td><td>82</td></tr><tr><td>100</td><td>76</td><td>0</td><td>0 0</td><td>12</td><td>0</td><td>84</td></tr></table>

OS-PSG OS-MetaGrad(FB) OS-ONS(FB) OS-DCSub-PSG Sub-MetaGrad(FB) Sub-ONS(FB)

![](images/40d257cecd1610f27885ce80584a68f361cb1ec7b218f403529241768bafe0ea.jpg)  
(a) Weighted multi-cover problem (Cora)  
OS-PSG OS-MetaGrad(FB) OS-ONS(FB) OS-DCSub-PSG Sub-MetaGrad(FB) Sub-ONS(FB)

![](images/62e5794bc3b139718c282c8346bd9f5748590ae604ac3cd46f7443872b3bf4f3.jpg)  
(b) Weighted multi-cover problem (ogbn-arxiv)

Figure 1. Relation between the worst 10% (the 90th percentile across episodes) of $\ell ^ { \mathrm { O P } }$ (best-so-far) and the number of oracle calls. The vertical axis is $\ell ^ { \mathrm { O P } } + 0 . 0 1$ on a logarithmic scale, and 0.01 represents $\ell ^ { \mathrm { O P } } = 0 .$ . Since the actions are integer-valued, $\ell ^ { \mathrm { O P } }$ takes discrete values in steps of $1 / N$ , and the percentiles are chosen from the observed values without interpolation. The colors indicate the optimization methods (PSG, MetaGrad(FB), ONS(FB), DC optimization); solid lines indicate the OS loss (proposed) and dashed lines the suboptimality loss (existing).

Weighted multi-cover problem. In this problem, which uses the actions of real data, on both Cora and ogbn-arxiv the OS-loss variant won in all three pairs (Table 1). In particular, in the PSG pair, the OS-loss variant won in 92 of 100 episodes on Cora and 81 of 100 on ogbn-arxiv.

## 8. Conclusion

In this paper, we proposed outperformance inverse optimization. Instead of explaining the actions as optimal solutions, it seeks weights that induce, at each state, an optimal solution outperforming the action in every component. Its surrogate, the OS loss, requires only oracles for the forward problem and its dominance-constrained version, and thus applies to nonconvex forward problems including MILPs. Its empirical mean vanishes if and only if optimal solutions dominating the actions exist. To minimize the OS loss, we gave gradient-based optimization methods and a DC optimization that terminates finitely if the inner convex subproblems are solved exactly. When each feasible region is M<sup>♮</sup>-convex (including M-convex) and $\Theta = \Delta ^ { d - 1 }$ , the uniform weight attains the global minimum 0 of the OS loss. We also showed that the upper bound on the generalization error is tight up to log factors for fixed κ. In the experiments, replacing only the loss function made the OS-loss variant win in 14 of 15 pairs, with particularly large gains when the actions were degraded optimal solutions.

We list two directions for future work. The first is to clarify when the DC optimization for the OS loss reaches a global minimizer and how it can be made to reach one. The second is to clarify whether the dependence of the upper bound on the generalization error on κ (which yields a term of order $d ^ { 2 } / N$ for integer programming; Example 6.4) is essential, and whether an upper bound without log factors can be obtained for a specific estimator.

## References

Ahuja, R. K. and Orlin, J. B. (2001). Inverse optimization. Operations Research, 49(5):771–783.

Aswani, A., Shen, Z.-J., and Siddiq, A. (2018). Inverse optimization with noisy data. Operations Research, 66(3):870–892.

Bärmann, A., Martin, A., Pokutta, S., and Schneider, O. (2018). An online-learning approach to inverse optimization. Available at arXiv:1810.12997v2.

Bärmann, A., Pokutta, S., and Schneider, O. (2017). Emulating the expert: Inverse optimization through online learning. In The 34th International Conference on Machine Learning, volume 70, pages 400–410. PMLR.

Bertsimas, D., Gupta, V., and Paschalidis, I. C. (2015). Data-driven estimation in equilibrium using inverse optimization. Mathematical Programming, 153(2):595– 633.

Besbes, O., Fonseca, Y., and Lobel, I. (2025). Contextual inverse optimization: Ofline and online learning. Operations Research, 73(1):424–443.

Birge, J. R., Hortaçsu, A., and Pavlin, J. M. (2017). Inverse optimization for the recovery of market structure from market outcomes: An application to the MISO electricity market. Operations Research, 65(4):837–855.

Boyd, S., Xiao, L., and Mutapcic, A. (2003). Subgradient methods. Notes for EE392o, Stanford University, Autumn, 2003.

Bulut, A. and Ralphs, T. K. (2021). On the complexity of inverse mixed integer linear optimization. SIAM Journal on Optimization, 31(4):3014–3043.

Burton, D. and Toint, P. L. (1992). On an instance of the inverse shortest paths problem. Mathematical Programming, 53:45–61.

Campi, M. C. and Garatti, S. (2008). The exact feasibility of randomized solutions of uncertain convex programs. SIAM Journal on Optimization, 19(3):1211–1230.

Chan, T. C. Y., Craig, T., Lee, T., and Sharpe, M. B. (2014). Generalized inverse multiobjective optimization with application to cancer therapy. Operations Research, 62(3):680–695.

Chan, T. C. Y., Eberg, M., Forster, K., Holloway, C., Ieraci, L., Shalaby, Y., and Yousefi, N. (2022). An inverse optimization approach to measuring clinical pathway concordance. Management Science, 68(3):1882–1903.

Chan, T. C. Y. and Kaw, N. (2020). Inverse optimization for the recovery of constraint parameters. European Journal of Operational Research, 282(2):415– 427.

Chan, T. C. Y. and Lee, T. (2018). Trade-of preservation in inverse multi-objective convex optimization. European Journal of Operational Research, 270(1):25–39.

Chan, T. C. Y., Lee, T., and Terekhov, D. (2019). Inverse optimization: Closed-form solutions, geometry, and goodness of fit. Management Science, 65(3):1115–1135.

Chan, T. C. Y., Mahmood, R., and Zhu, I. Y. (2025). Inverse optimization: Theory and applications. Operations Research, 73(2):1046–1074.

De Loera, J. A., Hemmecke, R., and Köppe, M. (2012). Algebraic and Geometric Ideas in the Theory of Discrete Optimization, volume 14 of MOS-SIAM Series on Optimization. SIAM.

de Oliveira, W. (2019). Proximal bundle methods for nonsmooth DC programming. Journal of Global Optimization, 75:523–563.

Fatemi, P., Maskan, H., Sra, S., and Mohajerin Esfahani, P. (2026). Tight generalization bounds for noiseless inverse optimization. Available at arXiv:2605.08866v1.

Fey, M. and Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds.

Gaudioso, M., Taheri, S., Bagirov, A. M., and Karmitsa, N. (2023). Bundle enrichment method for nonsmooth diference of convex programming problems. Algorithms, 16(8):394.

Ghobadi, K. and Mahmoudzadeh, H. (2021). Inferring linear feasible regions using inverse optimization. European Journal of Operational Research, 290(3):829–843.

Gurobi Optimization, LLC (2026). Gurobi Optimizer Reference Manual.

Harper, F. M. and Konstan, J. A. (2015). The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4):1–19.

Hazan, E., Agarwal, A., and Kale, S. (2007). Logarithmic regret algorithms for online convex optimization. Machine Learning, 69:169–192.

Heuberger, C. (2004). Inverse combinatorial optimization: A survey on problems, methods, and results. Journal of Combinatorial Optimization, 8(3):329–361.

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. (2020). Open Graph Benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, volume 33, pages 22118– 22133.

Joki, K., Bagirov, A. M., Karmitsa, N., Mäkelä, M. M., and Taheri, S. (2018). Double bundle method for finding Clarke stationary points in nonsmooth DC programming. SIAM Journal on Optimization, 28(2):1892–1919.

Kelley, Jr., J. E. (1960). The cutting-plane method for solving convex programs. Journal of the Society for Industrial and Applied Mathematics, 8(4):703–712.

Keshavarz, A., Wang, Y., and Boyd, S. (2011). Imputing a convex objective function. In 2011 IEEE International Symposium on Intelligent Control, pages 613–619. IEEE.

Kitaoka, A. (2024). Exact solution to data-driven inverse optimization of MILPs in finite time via gradient-based methods. Available at arXiv:2405.14273v8.

Kitaoka, A. (2025). Inverse mixed-integer programming: Learning constraints then objective functions. Available at arXiv:2510.04455v3.

Kitaoka, A. (2026). Explicit Iteration Complexity of Exact Data-Driven Inverse Optimization for Integer Linear Programs. Available at arXiv:2607.22263v1.

Kiwiel, K. C. (1990). Proximity control in bundle methods for convex nondiferentiable minimization. Mathematical Programming, 46:105–122.

Kunzinger, M. (2019). Theory of distributions. Lecture notes, Universität Wien.

Le Thi, H. A. and Pham Dinh, T. (2018). DC programming and DCA: thirty years of developments. Mathematical Programming, 169:5–68.

Lemaréchal, C., Nemirovskii, A., and Nesterov, Y. (1995). New variants of bundle methods. Mathematical Programming, 69:111–147.

Lin, B., Delage, E., and Chan, T. C. Y. (2024). Conformal inverse optimization. In Advances in Neural Information Processing Systems, volume 37, pages 63534– 63564.

Mohajerin Esfahani, P., Shafieezadeh-Abadeh, S., Hanasusanto, G. A., and Kuhn, D. (2018). Data-driven inverse optimization with imperfect information. Mathematical Programming, 167:191–234.

Murota, K. (2003). Discrete Convex Analysis. Society for Industrial and Applied Mathematics, Philadelphia.

Ng, A. Y. and Russell, S. J. (2000). Algorithms for inverse reinforcement learning. In The 17th International Conference on Machine Learning, pages 663–670.

Onn, S. (2010). Nonlinear Discrete Optimization: An Algorithmic Theory. Zurich Lectures in Advanced Mathematics. European Mathematical Society.

Perron, L. and Furnon, V. (2026). OR-Tools. Version 9.15, Google, https:// developers.google.com/optimization/.

Pham Dinh, T. and Le Thi, H. A. (1997). Convex analysis approach to d.c. programming: Theory, Algorithm and Applications. Acta Mathematica Vietnamica, 22(1):289–355.

Ren, K., Mohajerin Esfahani, P., and Georghiou, A. (2025). Inverse optimization via learning feasible regions. In The 42nd International Conference on Machine Learning, volume 267, pages 51471–51488. PMLR.

Sakaue, S., Bao, H., and Tsuchiya, T. (2025a). Revisiting online learning approach to inverse linear optimization: A Fenchel–Young loss perspective and gap-dependent regret analysis. In The 28th International Conference on Artificial Intelligence and Statistics, volume 258, pages 46–54. PMLR.

Sakaue, S., Tsuchiya, T., Bao, H., and Oki, T. (2025b). Online inverse linear optimization: Eficient logarithmic-regret algorithm, robustness to suboptimality, and lower bound. In Advances in Neural Information Processing Systems, volume 38,

pages 94876–94904.

Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B., and Eliassi-Rad, T. (2008). Collective classification in network data. AI Magazine, 29(3):93–106.

Shi, L., Zhang, G., Zhen, H., Fan, J., and Yan, J. (2023). Understanding and generalizing contrastive learning from the inverse optimal transport perspective. In The 40th International Conference on Machine Learning, volume 202, pages 31408–31421. PMLR.

Suzuki, Y., Wee, W. M., and Nishioka, I. (2019). TV advertisement scheduling by learning expert intentions. In The 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3071–3081.

Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation. Society for Industrial and Applied Mathematics.

van Erven, T. and Koolen, W. M. (2016). MetaGrad: Multiple learning rates in online learning. In Advances in Neural Information Processing Systems, volume 29, pages 3666–3674.

Van Rossum, G. and Drake, F. L. (2009). Python 3 reference manual. CreateSpace, Scotts Valley, CA.

Villani, C. (2009). Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin.

Wang, K., Shen, Z., Huang, C., Wu, C.-H., Dong, Y., and Kanakia, A. (2020). Microsoft Academic Graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413.

Wang, L. (2009). Cutting plane algorithms for the inverse mixed integer linear programming problem. Operations Research Letters, 37(2):114–116.

Yang, Z., Cohen, W. W., and Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings. In The 33rd International Conference on Machine Learning, volume 48, pages 40–48. PMLR.

Zattoni Scroccaro, P., Atasoy, B., and Mohajerin Esfahani, P. (2025). Learning in inverse optimization: Incenter cost, augmented suboptimality loss, and algorithms. Operations Research, 73(5):2661–2679.

## Appendix A. Related work

Classical inverse optimization. Classical inverse optimization is the problem of finding the objective weight closest to a prior estimate such that a given solution becomes optimal; when the forward problem is an LP, it can be solved via LP duality (Ahuja and Orlin, 2001). As extensions to MILP, Wang (2009) proposed a cutting-plane method that uses the forward problem as an oracle, and Bulut and Ralphs (2021) studied the computational complexity of inverse MILP. All of these assume that the given solution is optimal under some weight. When this assumption fails, classical inverse optimization returns trivial solutions such as the zero vector in the linear case (Chan and Lee, 2018). For an overview of inverse optimization, see the survey by Chan et al. (2025).

Details of inverse optimization for noisy data. The method that minimizes the distance between the observations and the optimal solutions (the prediction loss) (Aswani et al., 2018) is statistically consistent, but it leads to a bilevel program that is NP-hard even when the forward problem is convex, and it has to rely on enumerating the parameters. The tractable reformulations based on the residual of the KKT conditions (Keshavarz et al., 2011), the violation of variational inequalities (Bertsimas et al., 2015), the relaxation of the strong duality constraint (Chan et al., 2014, 2019), and distributionally robust optimization of the suboptimality loss (Mohajerin Esfahani et al., 2018) all assume convexity (conic representability) of the forward problem. Methods that estimate the constraints of the feasible region (Chan and Kaw, 2020; Ghobadi and Mahmoudzadeh, 2021; Ren et al., 2025) also mainly target LPs.

Details of inverse optimization for MILP. Among the methods that use only the forward-problem oracle, in the ofline setting, Lin et al. (2024) proposed point estimation by a cutting-plane method, and Kitaoka (2024, 2026) obtained exact solutions in finitely many iterations by gradient methods. The augmented suboptimality loss (Zattoni Scroccaro et al., 2025) is evaluated by solving an augmented forward problem with a distance term, and its convex reformulation over mixed-integer sets enumerates the integer points; except for binary decision variables, for which the augmented forward problem reduces to the forward problem with modified weights, it cannot be evaluated with the forward-problem oracle alone. Moreover, like the suboptimality loss, it aims to reproduce the observations as optimal solutions. We therefore do not compare with it in our experiments. As a method that learns the constraints in addition to the objective, Kitaoka (2025) proposed two-stage learning of the constraints and the objective. In the online setting, Bärmann et al. (2017, 2018) and Sakaue et al. (2025a,b) give regret upper bounds, and also give guarantees in the ofline setting via online-to-batch conversion (a conversion that outputs, $\mathrm { e . g . }$ , the average of the iterates of online learning and derives an upper bound on the ofline loss from a regret upper bound). Among these, Sakaue et al. (2025b) give a regret upper bound that adapts to the cumulative suboptimality of the agent’s actions. Moreover, although it does not minimize the suboptimality loss, Besbes et al. (2025) proposed a method that handles ofline and online learning in a setting that does not assume the structure of the forward problem. The theoretical guarantees of Kitaoka (2024, 2026, 2025) are limited to the setting where the agent’s actions are optimal.

Details of generalization error analysis for inverse optimization. Kitaoka (2025) observed that the “distance” between parameters that naturally arises in the generalization error analysis of inverse optimization is in general not a metric but a pseudometric, and extended the Dudley-type bound via covering numbers of sub-Gaussian processes to pseudometric spaces for losses that are Lipschitz with respect to a pseudometric. Applying this to inverse optimization for MILP that learns the objective and the constraints, they bounded the expectation of the suboptimality loss at the learned weight by $O ( \sqrt { d / N } )$ , where d is the dimension of the features. Fatemi et al. (2026) treated noiseless inverse optimization, in which the agent’s action is the unique optimal solution under the true weight, regarded the consistency constraint that each observation imposes on the weight as a constraint of a scenario program, and applied the result of Campi and Garatti (2008). For the estimator that minimizes a convex function under the consistency constraints, they bounded the probability that the true optimal solution is not contained in the set of optimal solutions under the learned weight (the set-level mismatch) by $2 ( d + \log ( 1 / \beta ) ) / N$ with probability at least $1 - \beta$ (Fatemi et al., 2026, Proposition 2.1), and, under an assumption of covariance diversity or for the incenter estimator, bounded the probability that the action selected under the learned weight by a fixed measurable rule that chooses one optimal solution when there are several does not coincide with the true optimal solution (the action-level mismatch) at the same order (Fatemi et al., 2026, Theorems 2.2 and 2.3). Furthermore, for the class of estimators that minimize a linear objective, they constructed an instance showing that these orders are tight (Fatemi et al., 2026, Theorem 3.1). Fatemi et al. (2026, Appendix A.4) also point out that, for an arbitrary consistent estimator, only a bound with a log factor based on the VC dimension is available. Theorem 6.3 of this paper gives a uniform bound of this type for outperformance inverse optimization, in which the agent’s actions may be suboptimal, and Theorem 6.5 is obtained by embedding the instance of Theorem 3.1 of Fatemi et al. (2026) into the setting of this paper.

## Appendix B. An action that cannot be outperformed

We give a concrete example in which no weight makes the OS loss $0 . \mathrm { ~ A ~ }$ point $x ^ { \prime } \in \mathcal { X } ( s )$ is called eficient if there is no $x \in \mathcal { X } ( s )$ with $x \geq x ^ { \prime }$ and $x \neq x ^ { \prime }$ . An eficient point is called supported if it is an optimal solution of ma $\mathfrak { c } _ { x \in \mathcal { X } ( s ) } \langle \theta , x \rangle$ for some $\theta \in \mathbb { R } _ { > 0 } ^ { d } \backslash \{ 0 \}$ , and unsupported otherwise. If $x ^ { \prime }$ is eficient, then $\mathcal { X } _ { \geq x ^ { \prime } } ( s ) = \{ x ^ { \prime } \}$ 2 so $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) = 0$ if and only if $x ^ { \prime }$ is an optimal solution under $\theta .$ . Hence, if $x ^ { \prime }$ is unsupported and $0 \not \in \Theta$ , we have $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) > 0$ for every $\theta \in \Theta$

Let $d = 2 , \Theta = \Delta ^ { 1 } , \mathcal { X } ( s ) = \{ ( 3 , 0 ) , ( 0 , 3 ) , ( 1 , 1 ) \}$ , and $x ^ { \prime } = ( 1 , 1 )$ . Since $\mathcal { X } ( s )$ is a union of three bounded polyhedra each consisting of a single point, it satisfies Assumption $4 . 2 ( 2 )$ . Neither $( 3 , 0 )$ nor $( 0 , 3 )$ dominates $x ^ { \prime }$ , so $x ^ { \prime }$ is eficient and $\mathcal { X } _ { \geq x ^ { \prime } } ( s ) = \{ x ^ { \prime } \}$ . For $\theta = ( \theta _ { 1 } , \theta _ { 2 } ) \in \Delta ^ { 1 }$ , since max $\{ \theta _ { 1 } , \theta _ { 2 } \} \ge 1 / 2$ and $\theta _ { 1 } + \theta _ { 2 } = 1$ , we have

$$
\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) = \operatorname* { m a x } \{ 3 \theta _ { 1 } , 3 \theta _ { 2 } , \theta _ { 1 } + \theta _ { 2 } \} - ( \theta _ { 1 } + \theta _ { 2 } ) = 3 \operatorname* { m a x } \{ \theta _ { 1 } , \theta _ { 2 } \} - 1 \geq \frac { 1 } { 2 } ,
$$

with equality if and only if $\theta = ( 1 / 2 , 1 / 2 )$ . Hence $x ^ { \prime }$ is unsupported, and the OS loss at this state is at least $1 / 2$ for every $\theta \in \Theta$ . In particular, if $( s , x ^ { \prime } )$ is one of N observations, the empirical mean of the OS loss is at least $1 / ( 2 N )$ for every $\theta \in \Theta$ and Equation (3.2) cannot be attained. Thus, even an action that is not dominated by any feasible solution can be impossible to outperform by an optimal solution under a single linear objective.

## Appendix C. Weak derivative of the OS loss

Proposition C.1 (Weak derivative in the sense of Schwartz distributions). Let $W \subset \mathbb { R } ^ { d }$ be an open set and let $\mathcal { Z } \subset \mathbb { R } ^ { d }$ be nonempty and compact. Define

$$
F ( \theta ) : = \operatorname* { m a x } _ { x \in { \mathcal { Z } } } \langle \theta , x \rangle .
$$

Moreover, for each $\theta \in W$ , take

$$
x ^ { * } ( \theta ) \in \mathop { \mathrm { a r g } } _ { x \in \mathcal { Z } } \operatorname* { m a x } \langle \theta , x \rangle .
$$

Then, for each $i = 1 , \ldots , d .$ , in the sense of Schwartz distributions $\mathcal { D } ^ { \prime } ( W )$ (cf. Kunzinger, 2019), we have

$$
\partial _ { i } F = x _ { i } ^ { * } \quad \mathrm { i n } \ D ^ { \prime } ( W ) .
$$

That ${ \mathrm { i s } } ,$ for any test function $\psi \in C _ { c } ^ { \infty } ( W )$ , we have

$$
\langle \partial _ { i } F , \psi \rangle : = - \int _ { W } F ( \theta ) \partial _ { i } \psi ( \theta ) d \theta = \int _ { W } x _ { i } ^ { * } ( \theta ) \psi ( \theta ) d \theta .
$$

Proof. In what follows, let $L _ { F } : = \operatorname* { m a x } _ { x \in { \mathcal { Z } } } \| x \|$ (since $\mathcal { Z }$ is nonempty and compact, $0 \leq L _ { F } < \infty )$

Step 0. $F$ is a finite-valued convex function on $\mathbb { R } ^ { d }$ with Lipschitz constant $L _ { F }$ . Since $\mathcal { Z }$ is nonempty and compact and $x \mapsto \langle \theta , x \rangle$ is continuous, the maximum is attained at each $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ , and $F$ is well defined as a finite-valued function on $\mathbb { R } ^ { d }$ . Since $F$ is the supremum of the family of linear functions $\{ \langle \cdot , x \rangle \} _ { x \in { \mathcal { Z } } }$ , it is convex. Moreover, for any $\theta , \theta ^ { \prime } \in \mathbb { R } ^ { d }$ and $x \in { \mathcal { Z } }$ , we have

$$
\langle \theta ^ { \prime } , x \rangle = \langle \theta , x \rangle + \langle \theta ^ { \prime } - \theta , x \rangle \leq F ( \theta ) + L _ { F } \| \theta ^ { \prime } - \theta \| ,
$$

so taking the maximum over $x \in { \mathcal { Z } }$ yields $F ( \theta ^ { \prime } ) \leq F ( \theta ) + L _ { F } \Vert \theta ^ { \prime } - \theta \Vert$ . Exchanging the roles of θ and $\theta ^ { \prime }$

$$
| F ( { \boldsymbol { \theta } } ^ { \prime } ) - F ( { \boldsymbol { \theta } } ) | \leq L _ { F } \| { \boldsymbol { \theta } } ^ { \prime } - { \boldsymbol { \theta } } \| \qquad ( \forall { \boldsymbol { \theta } } , { \boldsymbol { \theta } } ^ { \prime } \in \mathbb { R } ^ { d } )\tag{C.1}
$$

follows. In particular, $F$ is continuous and $F \in L _ { \mathrm { l o c } } ^ { 1 } ( W )$

Step 1. $F$ is diferentiable $\mathbf { a . e . } ,$ , and $\nabla F ( \theta ) = x ^ { * } ( \theta )$ at such points. By Equation (C.1) and Rademacher’s theorem (cf. Villani, 2009, Theorem 10.8 (ii)), $F$ is diferentiable in the classical sense almost everywhere on W. Take a point of diferentiability $\theta \in W$ . Since $x ^ { \ast } ( \theta ) \in$ arg $\operatorname* { m a x } _ { x \in { \mathcal { Z } } } \langle \theta , x \rangle$ , for any $\theta ^ { \prime } \in \mathbb { R } ^ { d }$

$$
F ( \theta ^ { \prime } ) \geq \langle \theta ^ { \prime } , x ^ { * } ( \theta ) \rangle = \langle \theta , x ^ { * } ( \theta ) \rangle + \langle \theta ^ { \prime } - \theta , x ^ { * } ( \theta ) \rangle = F ( \theta ) + \langle \theta ^ { \prime } - \theta , x ^ { * } ( \theta ) \rangle ,
$$

that is, $x ^ { * } ( \theta )$ is a subgradient of $F$ at $\theta \colon$

$$
F ( \theta ^ { \prime } ) - F ( \theta ) \geq \langle \theta ^ { \prime } - \theta , x ^ { * } ( \theta ) \rangle \qquad ( \forall \theta ^ { \prime } \in \mathbb { R } ^ { d } ) .\tag{C.2}
$$

Take any $h \in \mathbb { R } ^ { d }$ . Since $W$ is open, $\theta + t h \in W$ for suficiently small $t > 0 ;$ ; setting $\theta ^ { \prime } = \theta + t h$ in Equation (C.2) and dividing by $t > 0$ gives

$$
{ \frac { F ( \theta + t h ) - F ( \theta ) } { t } } \geq \langle h , x ^ { * } ( \theta ) \rangle .
$$

By diferentiability at θ, letting $t \downarrow 0$ yields

$$
\langle h , \nabla F ( \theta ) \rangle \geq \langle h , x ^ { * } ( \theta ) \rangle .
$$

Replacing h with −h gives the reverse inequality, so

$$
\left. h , \nabla F ( \theta ) \right. = \left. h , x ^ { * } ( \theta ) \right. \qquad ( \forall h \in \mathbb { R } ^ { d } ) ,
$$

that is, $\nabla F ( \theta ) = x ^ { * } ( \theta )$ . Hence we have

$$
\nabla F ( \theta ) = x ^ { * } ( \theta ) \quad { \mathrm { a . e . ~ } } \theta \in W .\tag{C.3}
$$

Note that $\theta \mapsto \nabla F ( \theta )$ is Lebesgue measurable (as the pointwise limit of diference quotients on the set of points of diferentiability), and by the completeness of the Lebesgue measure and Equation $( \mathrm { C . 3 } ) , x ^ { \ast }$ is also Lebesgue measurable. Moreover, since $\lVert x ^ { * } ( \theta ) \rVert \leq L _ { F } .$ , we have $x ^ { * } \in L ^ { \infty } ( W ; \mathbb { R } ^ { d } ) \subset L _ { \mathrm { l o c } } ^ { 1 } ( \bar { W } ; \mathbb { R } ^ { d } )$ , and the integral on the right-hand side of the claim converges absolutely.

Step 2. Identification of the distributional derivative via convergence of diference quotients. Take any $i \in \{ 1 , \ldots , d \}$ and $\psi \in C _ { c } ^ { \infty } ( W )$ , and let $\mathcal { K } : = \mathrm { s u p p } \psi$ (a compact subset of W). Since W is open,

$$
\delta : = \mathrm { d i s t } ( \mathcal { K } , \mathbb { R } ^ { d } \setminus W ) > 0
$$

(when $W = \mathbb { R } ^ { d }$ , read $\delta : = + \infty )$ . Let $\mathcal { K } _ { \delta } : = \{ \theta \in \mathbb { R } ^ { d } \vert$ dist $( \theta , { \cal K } ) \le \delta / 2 \}$ . Then $\displaystyle \mathcal { K } _ { \delta }$ is a compact subset of $W$ , and if $0 < | t | < \delta / 2$ , then $\mathcal { K } + t e _ { i } \subset \mathcal { K } _ { \delta } \subset W$ (where $e _ { i }$ is the ith standard basis vector).

First, for $0 < | t | < \delta / 2$ , since the support of $\psi$ is contained in $\kappa .$ , the change of variables $\theta \mapsto \theta - t e _ { i }$ gives

$$
\int _ { W } F ( \theta + t e _ { i } ) \psi ( \theta ) d \theta = \int _ { K + t e _ { i } } F ( \theta ) \psi ( \theta - t e _ { i } ) d \theta = \int _ { W } F ( \theta ) \psi ( \theta - t e _ { i } ) d \theta ,
$$

and therefore we have

$$
\int _ { W } { \frac { F ( \theta + t e _ { i } ) - F ( \theta ) } { t } } \psi ( \theta ) d \theta = \int _ { W } F ( \theta ) { \frac { \psi ( \theta - t e _ { i } ) - \psi ( \theta ) } { t } } d \theta .\tag{C.4}
$$

For the right-hand side of Equation $\left( \mathrm { C . 4 } \right)$ , since $\psi \in C _ { c } ^ { \infty } ( W )$ , the convergence $\left( \psi ( \cdot - t e _ { i } ) - \psi ( \cdot ) \right) / t \to - \partial _ { i } \psi \ ( t \to 0 )$ is uniform, and for $0 < | t | < \delta / 2$ the support of the integrand is contained in $\displaystyle \mathcal { K } _ { \delta }$ . Since $F$ is continuous (Step 0), it is bounded on the compact set $\kappa _ { \delta } .$ , and therefore

$$
\operatorname* { l i m } _ { t \to 0 } \int _ { W } F ( \theta ) \frac { \psi ( \theta - t e _ { i } ) - \psi ( \theta ) } { t } d \theta = - \int _ { W } F ( \theta ) \partial _ { i } \psi ( \theta ) d \theta .
$$

For the left-hand side of Equation (C.4), by Step 1, for almost every $\theta \in W$

$$
\operatorname* { l i m } _ { t \to 0 } \frac { F ( \theta + t e _ { i } ) - F ( \theta ) } { t } = \partial _ { i } F ( \theta ) = x _ { i } ^ { * } ( \theta ) ,
$$

and by Equation (C.1),

$$
\left. { \frac { F ( \theta + t e _ { i } ) - F ( \theta ) } { t } } \psi ( \theta ) \right. \leq L _ { F } \left. \psi ( \theta ) \right. \in L ^ { 1 } ( W ) ,
$$

so the integrand is dominated by an integrable function independent of t. Hence, by Lebesgue’s dominated convergence theorem,

$$
\operatorname* { l i m } _ { t \to 0 } \int _ { W } { \frac { F ( \theta + t e _ { i } ) - F ( \theta ) } { t } } \psi ( \theta ) d \theta = \int _ { W } x _ { i } ^ { * } ( \theta ) \psi ( \theta ) d \theta .
$$

Combining these by letting $t  0$ in Equation $\left( \mathrm { C . 4 } \right)$ , for any $\psi \in C _ { c } ^ { \infty } ( W )$ , we have

$$
\langle \partial _ { i } F , \psi \rangle = - \int _ { W } F ( \theta ) \partial _ { i } \psi ( \theta ) d \theta = \int _ { W } x _ { i } ^ { * } ( \theta ) \psi ( \theta ) d \theta .
$$

That is, $\partial _ { i } F = x _ { i } ^ { * }$ in $\mathcal { D } ^ { \prime } ( W )$

Proof of Proposition 5.1. Apply Proposition C.1 with $W = \mathbb { R } ^ { d }$ and ${ \mathcal { Z } } = \mathcal { X } ( s )$ , and with $W = \mathbb { R } ^ { d }$ and $\mathcal { Z } = \mathcal { X } _ { \geq x ^ { \prime } } ( s )$ (both are nonempty and compact), and take the diference of the two terms in Equation (4.2). □

## Appendix D. Pseudocode of the rules for selecting evaluation points in the inner solver: Step 12 of Algorithm 5

In this section, as concrete forms of Step 12 of Algorithm 5, we give the pseudocode of the cutting-plane method (Kelley, 1960), the proximal bundle method (Kiwiel, 1990), and the level method (Lemaréchal et al., 1995). Furthermore, in §D.4, we give the pseudocode of the cutting-plane finishing of Theorem F.3, which starts with one of these and switches to the cutting-plane method for finishing. Below, UB denotes the minimum of the values of the convex subproblem evaluated so far in Algorithm 5 (an upper bound), LB denotes the minimum value of the model of cuts (a lower bound), and $\epsilon _ { \mathrm { i n } }$ denotes the inner tolerance. The cut $g _ { j }$ , the cut set ${ \mathcal { G } } ,$ the solution pool V, the maximum number of iterations J, the point $\hat { \theta }$ with the smallest evaluated value, and converged are defined in Algorithm 5 (§E.3), and inherited cuts and cut inheritance are defined in §E.4. The selection step refers to Step 12 of Algorithm 5. Each is written as a procedure that is called at iteration j of Algorithm 5 immediately after the cut $g _ { j }$ is added to $\mathcal { G }$ and UB, LB are updated (after passing the stopping test), and that returns $\theta ^ { j + 1 }$ (only the proximal bundle method also returns the quantities carried over to the next call).

We fix the notation. Let $\mathcal { G } _ { j }$ be the cut set at the selection step of iteration j (including the inherited cuts and $g _ { 0 } , \ldots , g _ { j } )$ , and define the cutting-plane model

$$
\hat { \ell } _ { j } ( \theta ) : = \operatorname* { m a x } _ { g \in \mathcal { G } _ { j } } \langle \theta , g \rangle .\tag{D.1}
$$

By Proposition E.3, $\hat { \ell } _ { j } \le \bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ , and at an evaluated point $\theta ^ { i } \ ( i \leq j )$ , since $g _ { i } \in { \mathcal { G } } _ { j } .$ , we have ${ \hat { \ell } } _ { j } ( \theta ^ { i } ) = { \bar { \ell } } ^ { \mathrm { s u b } } ( \theta ^ { i } ; y ^ { \prime } ) = \langle \theta ^ { i } , g _ { i } \rangle$ . The LB of Algorithm 5 equals min<sub>θ∈Θ</sub> $\hat { \ell } _ { j } ( \theta )$ . Let $\| \cdot \|$ denote the Euclidean norm. The subproblems of the three rules difer only in the objective over the common polyhedron

$$
P _ { j } : = \left\{ ( \theta , \tau ) \in \mathbb { R } ^ { d } \times \mathbb { R } ~ \middle | ~ \theta \in \Theta , ~ \langle \theta , g \rangle \leq \tau ~ ( \forall g \in \mathcal { G } _ { j } ) \right\}\tag{D.2}
$$

(the epigraph of the model $\hat { \ell } _ { j }$ over $\Theta )$ ; since Θ is a polyhedron by Assumption 4.2(1), they are an LP for the cutting-plane method and a convex quadratic program (QP) for the proximal bundle method and the level method.

D.1. Cutting-plane method. The cutting-plane method (Kelley, 1960) takes a minimizer of the model $\boldsymbol { \hat { \ell } } _ { j }$ over Θ as the next evaluation point (Algorithm 1). Since this minimization is exactly the LP that Algorithm 5 solves to compute LB, no additional solve is needed, and the θ component of the optimal solution of that LP can be reused. When the LP has multiple optimal solutions, any of them may be chosen (the proof of Lemma E.10 uses only optimality). Finite exactness and an upper bound on the number of iterations under this rule are given in Lemma E.10, and its use as the rule after the switch in cutting-plane finishing (§D.4) is given in Theorem F.3.

Algorithm 1 Rule for selecting evaluation points: cutting-plane method   
Input: Cut set $\mathcal { G } _ { j }$   
1: $( \theta ^ { \mathrm { L P } } , \tau ^ { \mathrm { L P } } )$ ← optimal solution and optimal value of the LP min $\{ \tau \ : | \ : ( \theta , \tau ) \in P _ { j } \}$   
2: $\theta ^ { j + 1 }  \theta ^ { \mathrm { { L P } } }$   
3: return $\theta ^ { j + 1 }$

D.2. Proximal bundle method. The proximal bundle method (Kiwiel, 1990) maintains a stability center $\theta ^ { \mathrm { c } } \in \Theta$ and takes as the candidate point the optimal solution of

$$
\operatorname* { m i n } \Bigl \{ \tau + \frac { u } { 2 } \| \theta - \theta ^ { \mathrm { c } } \| ^ { 2 } \ \Bigl | \ ( \theta , \tau ) \in P _ { j } \ \Bigr \} ,\tag{D.3}
$$

which adds the proximal term $\begin{array} { r } { \frac { u } { 2 } \| \theta - \theta ^ { \mathrm { c } } \| ^ { 2 } \ ( u > 0 } \end{array}$ is the proximal weight) to the model (Algorithm 2). Since the value $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j + 1 } ; y ^ { \prime } )$ of the candidate point $\theta ^ { j + 1 }$ is evaluated at the next iteration of Algorithm $5 ,$ the test of whether to update the center (the descent test) is performed at the beginning of the selection step of the next iteration. The predicted decrease $\nu : = f ^ { \mathrm { c } } - \hat { \ell } _ { j } ( \theta ^ { j + 1 } )$ (where $f ^ { \mathrm { c } } : = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { \mathrm { c } } ; y ^ { \prime } )$ is the value at the center) is carried over to the next call together with the candidate point; there, if the actual decrease $f ^ { \mathrm { c } } - \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j + 1 } ; y ^ { \prime } )$ is at least $\rho \nu \left( \rho \in \left( 0 , 1 \right) \right.$ is the descent parameter), the center is moved to the candidate point (serious step). Otherwise, the center is kept and only the cut of the candidate point is added (null step). Since $( \theta ^ { \mathrm { c } } , \tau ) = ( \theta ^ { \mathrm { c } } , f ^ { \mathrm { c } } ) \in P _ { j }$ is a feasible solution of Equation (D.3) with objective value $f ^ { \mathrm { c } }$ , we have $\hat { \ell } _ { j } ( \theta ^ { j + 1 } ) \leq f ^ { \mathrm { c } }$ , that is, always $\nu \geq 0$

We make two remarks specific to this framework. First, the $\hat { \theta }$ of Algorithm 5 (the point with the smallest evaluated value) and the center $\theta ^ { \mathrm { c } }$ difer in general (a null-step point may update UB). Always ${ \mathrm { U B } } \leq f ^ { \mathrm { c } }$ . Second, the stopping test specific to the proximal bundle method $( \nu \leq \epsilon ,$ , where $\epsilon > 0$ is a tolerance) is replaced by the stopping test $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { i n } }$ of Algorithm 5. Indeed, when the selection step is reached, $\mathrm { U B } - \mathrm { L B } > \epsilon _ { \mathrm { i n } } \geq 0 .$ , and in particular $\mathrm { L B } < \mathrm { U B } \leq f ^ { \mathrm { c } } = \hat { \ell } _ { j } ( \theta ^ { \mathrm { c } } )$ , so $\theta ^ { \mathrm { c } }$ is not a minimizer of the model $\boldsymbol { \hat { \ell } } _ { j }$ over Θ. Since $\theta ^ { \mathrm { c } }$ being a minimizer of the model with the proximal term Equation (D.3) is equivalent to its being a minimizer of $\hat { \ell } _ { j }$ (because the gradient of the proximal term vanishes at $\theta ^ { \mathrm { c } } )$ , we have $\theta ^ { j + 1 } \neq \theta ^ { \mathrm { c } }$ 2 and since the optimal value of Equation (D.3) is at most $f ^ { \mathrm { c } } , \hat { \ell } _ { j } ( \theta ^ { j + 1 } ) < f ^ { \mathrm { c } }$ , that is, $\nu > 0$ holds. Conversely, in the situation where $\nu = 0 ~ ( \theta ^ { \mathrm { c } }$ is a minimizer of the model), $\mathrm { L B } = f ^ { \mathrm { c } } \geq \mathrm { U B }$ , so Algorithm 5 has already stopped at the stopping test of the same iteration. Moreover, since cuts are never discarded owing to cut inheritance, the bundle compression (aggregate subgradient) of Kiwiel (1990) is not used. The proximal weight u is updated according to the proximity control rule of Kiwiel (1990) (u is not decreased at null steps, and the lower bound $u _ { \mathrm { m i n } } > 0$ is maintained; keeping u constant is also allowed).

D.3. Level method. The level method (Lemaréchal et al., 1995) sets a level between the lower bound LB and the upper bound UB,

$$
\begin{array} { r } { \mathrm { l e v } _ { j } : = \mathrm { L B } + \lambda \left( \mathrm { U B - L B } \right) \qquad \left( \lambda \in ( 0 , 1 ) { \mathrm { ~ i s ~ t h e ~ l e v e l ~ p a r a m e t e r } } \right) , } \end{array}\tag{D.4}
$$

and takes the projection of the current evaluation point $\theta ^ { j }$ onto the level set $\{ \theta \in \Theta \mid \hat { \ell } _ { j } ( \theta ) \leq \mathrm { l e v } _ { j } \}$ of the model as the next evaluation point (Algorithm 3).

Algorithm 2 Rule for selecting evaluation points: proximal bundle method   
Input: Cut set ${ \mathcal { G } } _ { j } ,$ current evaluation point $\theta ^ { j }$ and its value $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = \langle \theta ^ { j } , g _ { j } \rangle$   
descent parameter $\rho \in ( 0 , 1 )$ , initial proximal weight u<sub>0</sub> and its lower bound   
$u _ { \mathrm { m i n } } \ ( u _ { 0 } \ge u _ { \mathrm { m i n } } > 0 )$   
Input: Quantities carried over from the call at iteration $j - 1$ (undefined for $j = 0 )$   
stability center $\theta ^ { \mathrm { c } } \in \Theta$ , its value $f ^ { \mathrm { c } } = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { \mathrm { c } } ; y ^ { \prime } )$ , proximal weight $u \geq u _ { \mathrm { m i n } } ,$   
predicted decrease $\nu = f ^ { \mathrm { c } } - \hat { \ell } _ { j - 1 } ( \theta ^ { j } ) > 0$   
1: $\mathbf { i f } \ j = 0$ then   
2: $\dot { \theta ^ { \mathrm { c } } }  \theta ^ { 0 } , f ^ { \mathrm { c } }  \langle \theta ^ { 0 } , g _ { 0 } \rangle$ , u ← u<sub>0</sub> (initialization of the center)   
3: else if $\langle \theta ^ { j } , g _ { j } \rangle \leq f ^ { \mathrm { c } } - \rho \nu$ then   
4: $\theta ^ { \mathrm { c } }  \theta ^ { j } , \dot { f ^ { \mathrm { c } } }  \langle \theta ^ { j } , g _ { j } \rangle$ (serious step: update of the center)   
5: u ← update by the proximity control rule (Kiwiel, 1990)   
6: else   
7: u ← update by the proximity control rule (without decreasing u) (null step:   
the center is kept)   
8: end if   
9: $( \theta ^ { j + 1 } , \tau _ { j + 1 } )$ ← optimal solution of the QP Equation (D.3) (minimization of the   
model with the proximal term; $\boldsymbol { \tau } _ { j + 1 } = \hat { \ell } _ { j } ( \theta ^ { j + 1 } ) )$   
10: $\nu  f ^ { \mathrm { c } } - \tau _ { j + 1 }$ (predicted decrease; $\nu > 0 )$   
11: return $\theta ^ { j + 1 }$ , and $\theta ^ { \mathrm { c } } , f ^ { \mathrm { c } } , u , \nu$ carried over to the call at iteration $j + 1$

$$
\theta ^ { \mathrm { L P } }
$$

$$
\mathrm { L P }
$$

$$
\hat { \ell } _ { j } ( \theta ^ { \mathrm { L P } } ) = \mathrm { L B } \leq \mathrm { l e v } _ { j }
$$

$$
\left\{ \frac 1 2 \lVert \theta - \theta ^ { j } \rVert ^ { 2 } \ \left| \begin{array} { l } { \theta \in \Theta , \ \langle \theta , g \rangle \leq \mathrm { l e v } _ { j } \ ( \forall g \in \mathcal G _ { j } ) } \end{array} \right. \right\}
$$

$$
5 ,
$$

$$
P _ { j }
$$

$$
\tau = \mathrm { l e v } _ { j } )\tag{D.5}
$$

$$
\mathrm { U B } - \mathrm { L B } \leq \epsilon
$$

$$
\epsilon > 0
$$

Algorithm 3 Rule for selecting evaluation points: level method   
Input: Cut set $\mathcal { G } _ { j }$ , current evaluation point $\theta ^ { j } ,$ , upper bound UB, lower bound LB,   
level parameter $\lambda \in ( 0 , 1 )$   
1: $\mathrm { l e v } _ { j } \gets \mathrm { L B } + \lambda \left( \mathrm { U B } - \mathrm { L B } \right)$ (setting the level; Equation $\left( \mathrm { { D . 4 } } \right) )$   
2: $\theta ^ { j + 1 } $ optimal solution of the QP Equation (D.5) (projection of $\theta ^ { j }$ onto the   
level set $\{ \theta \in \Theta \mid \hat { \ell } _ { j } ( \theta ) \leq \mathrm { l e v } _ { j } \} ;$ ; nonempty)   
3: return $\bar { \theta } ^ { j + 1 }$

D.4. Cutting-plane finishing. The cutting-plane finishing of Theorem F.3 can also be written as a selection rule of Step 12 (Algorithm 4). Under the guarantee of Theorem F.3, Algorithm 5 is run with tolerance 0 and $J = \infty$ , and the switching threshold $\epsilon _ { \mathrm { s w } } > 0$ is held by the rule (since $\epsilon _ { \mathrm { s w } }$ is a diferent quantity from the tolerance $\epsilon _ { \mathrm { i n } }$ of Algorithm 5, we use a separate symbol). The rule distinguishes cases by comparing UB−LB with $\epsilon _ { \mathrm { s w } } \colon \mathrm { i f ~ U B - L B } > \epsilon _ { \mathrm { s w } } .$ it applies a selection rule R (e.g., Algorithm $1 , 2 , 3 ) ;$ ; otherwise, it applies the cutting-plane method (Algorithm 1).

We call the iterations at which the former condition holds Phase 1, and those at which the latter holds Phase 2 (they correspond to the two phases of Theorem F.3).

This case distinction gives a two-phase procedure because UB − LB is nonincreasing in the iteration $j .$ Indeed, by Steps 5–7 of Algorithm 5, UB is nonincreasing as the minimum of the evaluated values, and since the cut set G only grows in Step $4 , \mathrm { L B } = \mathrm { m i n } _ { \theta \in \Theta } \hat { \ell } _ { j } ( \theta )$ in Step 8 is nondecreasing. Therefore, once $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { s w } }$ holds, it continues to hold thereafter, and the switch of phases occurs only once and never goes back.

The guarantee is as stated in Theorem F.3. If we take $\epsilon _ { \mathrm { s w } } < \delta ( y ^ { \prime } ) ~ ( \delta ( y ^ { \prime } )$ is the submodel gap, Equation (F.3)), then at the time of the switch the lower bound is determined exactly as $\mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } ) ( m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is the minimum value of the convex subproblem of the branch $y ^ { \prime } { \mathrm { . } }$ , Equation (E.4)) (Lemma F.2), and since the cuts only increase thereafter, this value is maintained. In Phase 2, the evaluation point is a minimizer of the model, so $\mathrm { U B } = \mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is reached in finitely many iterations, the stopping test with tolerance 0 is satisfied, and an exact minimizer is returned. Note that since the cut set $\mathcal { G }$ and the solution pool V are carried over unchanged within Algorithm 5 across the phases, the switch requires no additional calls to the forward-problem oracle. When the proximal bundle method is used as the rule R, the quantities carried over by Algorithm 2 may be discarded at the switch.

Algorithm 4 Rule for selecting evaluation points: cutting-plane finishing (Theo  
rem F.3)   
Input: Cut set $\mathcal { G } _ { j }$ , current evaluation point $\theta ^ { j }$ , upper bound UB, lower bound LB,   
switching threshold $\epsilon _ { \mathrm { s w } } > 0$ , termination tolerance $\epsilon _ { \mathrm { t e r m } } \geq 0$ , selection rule R   
$( \mathrm { e . g . , A }$ lgorithm 1, 2, 3)   
1: if $\mathrm { U B } - \mathrm { L B } > \epsilon _ { \mathrm { s w } }$ then   
2: $\theta ^ { j + 1 } $ the point obtained by applying the rule R   
3: else   
4: (finishing by the cutting-plane method) $\theta ^ { j + 1 } $ the point obtained by applying   
Algorithm 1   
5: if $\lVert \theta ^ { j + 1 } - \theta ^ { i } \rVert _ { 2 } \leq \epsilon _ { \mathrm { t e r m } }$ for some evaluated point $\theta ^ { i } \ ( i \leq j )$ then   
6: stop Algorithm 5 with converged = TRUE without evaluating $\theta ^ { j + 1 }$   
7: end if   
8: end if   
9: return $\theta ^ { j + 1 }$

In Phase 2, if the minimizer of the LP coincides with an evaluated point within $\epsilon _ { \mathrm { t e r m } } ,$ the inner loop is stopped with converged = TRUE without evaluating that point (in this case $\mathrm { U B } - \mathrm { L B } = 0$ does not necessarily hold). This is because re-evaluating the same point yields no new cut, and what remains in UB − LB is a diference of the order of the rounding error of the loss values, so we avoid continuing the iterations due to rounding errors. When $\epsilon _ { \mathrm { t e r m } } = 0$ and the LPs are solved exactly, this test is never triggered: if $\theta ^ { j + 1 } = \theta ^ { i }$ for an evaluated point $\theta ^ { i }$ $( i \leq j )$ , then the cut $g _ { i }$ obtained at $\theta ^ { i }$ belongs to ${ \mathcal { G } } ,$ so $\operatorname { L B } = \operatorname* { m a x } _ { g \in { \mathcal { G } } } \langle \theta ^ { j + 1 } , g \rangle \geq$ $\langle \theta ^ { i } , g _ { i } \rangle = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { i } ; y ^ { \prime } ) \geq \mathrm { U B }$ , and Algorithm 5 has already stopped by $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { i n } }$ Theorem F.3 guarantees an exact minimizer under $\epsilon _ { \mathrm { t e r m } } = 0$ (the positive $\epsilon _ { \mathrm { t e r m } }$ used in the experiments is a safeguard against rounding errors; §H.3).

When Algorithm 6 is run with Algorithm 4 as the rule of Step 12, as long as we take $\begin{array} { r } { \epsilon _ { \mathrm { s w } } < \delta _ { \mathrm { i n } } : = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } } } \delta ( y ^ { \prime } ) } \end{array}$ and $\epsilon _ { \mathrm { t e r m } } = 0$ , and as long as Phase 1 ends after finitely many iterations under the rule R (which always holds if R is the cutting-plane method), the inner loop is solved exactly at every outer iteration, so the assumption of Proposition E.4 is satisfied (Remark F.8).

## Appendix E. DC optimization

This section describes the DC optimization method that solves Equation (4.4) on the basis of the decomposition into branches in Proposition 5.2 and the DC structure in Remark E.1. Its skeleton is a double loop: the outer loop is simplified DCA (Pham Dinh and Le Thi, 1997), which linearizes the concave part and reduces the problem to convex subproblems, and the inner loop solves the convex subproblems by a cutting-plane-type solver, typified by the cutting-plane method (Kelley, 1960). In addition, the method has a mechanism that accumulates the feasible solutions returned by the forward-problem oracle as global lower-bounding cuts and reuses them across iterations and runs (cut inheritance). Cut inheritance instantiates the idea of the proximal bundle method for DC programming of de Oliveira (2019) in accordance with the structure of our problem. Below, §E.1 gives the notation and assumptions, §E.2 the relation between branches and DCA, §E.3 the pseudocode, §E.4 cut inheritance, §E.5 the underlying literature, §E.6 finite convergence and local optimality, and §E.7 the theoretical analysis.

E.1. Notation and assumptions. Throughout this section, we assume Assumption 4.2. In the appendices below, a point u of a convex set C is an extreme point of $C { \mathrm { ~ i f ~ } } u = \lambda v + ( 1 - \lambda ) w$ with $v , w \in C$ and $\lambda \in ( 0 , 1 )$ implies $v = w = u$ . For a state $s \in S$ and $x ^ { \prime } \in \mathcal { X } ( s )$ , we set

$$
F _ { 1 } ( \theta ; s ) : = \operatorname* { m a x } _ { x \in \mathcal { X } ( s ) } \langle \theta , x \rangle , \qquad F _ { 2 } ( \theta ; x ^ { \prime } , s ) : = \operatorname* { m a x } _ { x \in \mathcal { X } _ { \geq x ^ { \prime } } ( s ) } \langle \theta , x \rangle .\tag{E.1}
$$

Since $x ^ { \prime } \in \mathcal { X } _ { \geq x ^ { \prime } } ( s )$ , the set $\mathcal { X } _ { \geq x ^ { \prime } } ( s )$ is nonempty, and it is compact as the intersection of the compact set $\mathcal { X } ( s )$ and the closed set $\{ x \in \mathbb { R } ^ { d } \mid x \geq x ^ { \prime } \}$ ; hence both maxima in Equation (E.1) are attained. By definition (Equation (4.2)),

$$
\ell ^ { \mathrm { { O S } } } ( \theta ; x ^ { \prime } , s ) = F _ { 1 } ( \theta ; s ) - F _ { 2 } ( \theta ; x ^ { \prime } , s ) .
$$

By the extreme-point oracle assumption (Assumption 4.2(3)), $\mathcal { V } ( s ^ { n } ) \subseteq \mathcal { X } ( s ^ { n } )$ and $\mathcal { V } _ { \geq x ^ { n } } ( s ^ { n } ) \subseteq \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } )$ are finite sets. Furthermore, since the value returned by the oracle attains the maximum for every $\theta ,$ we have

$$
F _ { 1 } ( \theta ; s ^ { n } ) = \operatorname* { m a x } _ { y \in \mathcal { Y } ( s ^ { n } ) } \langle \theta , y \rangle , \qquad F _ { 2 } ( \theta ; x ^ { n } , s ^ { n } ) = \operatorname* { m a x } _ { y \in \mathcal { Y } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , y \rangle .\tag{E.2}
$$

That is, $F _ { 1 }$ and $F _ { 2 }$ are maxima of finitely many linear functions (convex, piecewise linear, and positively homogeneous of degree 1).

Among the branches introduced in the main text (§5.2), we deal with the tuples $\begin{array} { r } { y ^ { \prime } = ( y ^ { \prime n } ) _ { n = 1 } ^ { N } \in \mathcal { Y } _ { > } ^ { \mathrm { a l l } } = \prod _ { n = 1 } ^ { N } \mathcal { Y } _ { \geq x ^ { n } } ( s ^ { n } ) } \end{array}$ consisting of values that the oracle can return. For $\boldsymbol { y } \in \mathbb { R } ^ { d }$ and a branch $y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } }$ , we set

$$
\ell ^ { \mathrm { s u b } } ( \theta ; y , s ) : = F _ { 1 } ( \theta ; s ) - \langle \theta , y \rangle ,\tag{E.3}
$$

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime n } , s ^ { n } ) , \qquad m _ { \mathrm { s u b } } ( y ^ { \prime } ) : = \operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) .\tag{E.4}
$$

We call $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ the branch minimum of the branch $y ^ { \prime } .$ . The function $\ell ^ { \mathrm { s u b } }$ has the same form as the suboptimality loss (Mohajerin Esfahani et al., 2018). By Equation (E.2), each $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; \bar { y } ^ { \prime } )$ is convex and piecewise linear, and by the compactness of Θ, the minimum in Equation (E.4) is attained.

## E.2. Relation between branches and DCA.

Remark E.1 (Relation to the DC structure). The empirical mean of the OS loss $\bar { \ell } ^ { \mathrm { O S } }$ is a DC function of $\theta \in \Theta$ , that is, a function that can be written in the form convex function − convex function. Indeed, by definition,

$$
\bar { \ell } ^ { \mathrm { { O S } } } ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m a x } _ { x \in \mathcal { X } ( s ^ { n } ) } \langle \theta , x \rangle - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m a x } _ { x \in \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle ,
$$

and both terms on the right-hand side are convex functions of $\theta .$ By this observation, Equation (4.4) can be regarded as a DC optimization problem and connected to the literature on DC programming. The operation of linearizing the concave part $\begin{array} { r } { - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \operatorname* { m a x } _ { x \in \mathcal { X } _ { \geq x ^ { n } } \left( s ^ { n } \right) } \langle \theta , x \rangle } \end{array}$ at the dominating vertex (the solution returned by the dominance-constrained forward-problem oracle) $x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ is nothing but the minimization with respect to $y ^ { \prime }$ in Proposition 5.2.

Proof of Proposition 5.2. By Assumption $4 . 2 ( 2 ) , \chi _ { \geq x ^ { n } } ( s ^ { n } )$ is nonempty and compact for each n, so the maximum in the second term of Equation (4.2) is attained, and $\begin{array} { r } { - \operatorname* { m a x } _ { x \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle = \operatorname* { m i n } _ { y \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } ) } \left( - \langle \theta , y \rangle \right) } \end{array}$ . Since the minimizations for diferent n can be carried out independently,

$$
\bar { \ell } ^ { \mathrm { { o s } } } ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Bigl [ \operatorname* { m a x } _ { x \in \mathcal { X } ( s ^ { n } ) } \langle \theta , x \rangle + \operatorname* { m i n } _ { y \in \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) } \bigl ( - \langle \theta , y \rangle \bigr ) \Bigr ] = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { X } _ { \geq } ^ { \mathrm { 2 l } } } \bar { \ell } ^ { \mathrm { { s u b } } } ( \theta ; y ^ { \prime } ) ,
$$

which gives the first equality in Equation (5.2). By Assumption 4.2(3), for every $\theta$ the value returned by the oracle $x ^ { \ast } ( \theta ; x ^ { n } , s ^ { n } ) \in \mathcal { V } _ { \geq x ^ { n } } \big ( s ^ { n } \big ) \subseteq \mathcal { X } _ { \geq x ^ { n } } \big ( s ^ { n } \big )$ attains the maximum, so $\scriptstyle { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ may be replaced by $\scriptstyle \left. \mathcal { P } _ { \geq x ^ { n } } ( s ^ { n } ) \right.$ in the argument above, which gives the second equality. Since $y _ { \ge } ^ { \mathrm { a l l } }$ is a finite set (Remark 4.3) and each $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is attained, Equation (5.3) follows by exchanging the min over θ and over $y ^ { \prime } . \sqsupset$

The decomposition into branches was given in Proposition 5.2 of the main text. Here we supplement the relation between DCA and branches.

DCA can be regarded as an iterative method that selects branches adaptively without enumerating them explicitly. If the concave part $\begin{array} { r } { - { \frac { 1 } { N } } \sum _ { n = 1 } ^ { N } F _ { 2 } \big ( \cdot ; x ^ { n } , s ^ { \bar { n } } \big ) } \end{array}$ is linearized at the current point $\theta ^ { k }$ by a subgradient, namely the dominating vertices $y ^ { ( k ) , n } : = x ^ { * } ( \theta ^ { k } ; x ^ { n } , s ^ { \bar { n } } )$ , then the resulting convex subproblem is exactly min ${ \mathfrak { l } } \in \Theta { \bar { \ell } } ^ { \mathrm { s u b } } ( \theta ; y ^ { ( k ) } )$ , that is, the minimization of the empirical mean of the suboptimality loss with the branch $y ^ { ( k ) } = ( y ^ { ( k ) , n } ) \mathrm { , }$ <sub>n</sub> fixed. A subgradient oracle for the convex subproblem is obtained immediately from the forward-problem oracle as

$$
\partial _ { \theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { ( k ) } ) \ni \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ x ^ { * } ( \theta , s ^ { n } ) - y ^ { ( k ) , n } \bigr ]
$$

(as in Bärmann et al., 2017). Evaluated at $\theta = \theta ^ { k }$ , it coincides with the gradient in the sense of weak derivatives $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } g ( \theta ^ { k } ; x ^ { n } , s ^ { n } ) } \end{array}$ of Proposition 5.1. That is, whereas the gradient-based optimization methods of §5.1 update the branch at every

step, DCA fixes the branch, solves the convex subproblem to completion, and then updates the branch.

## E.3. Algorithm. The pseudocode is given in Algorithms 5 and 6.

Algorithm 5 Inner solver based on a cutting-plane model for the convex subproblem   
(with cut inheritance)   
Input: Branch $y ^ { \prime } = ( y ^ { \prime n } ) _ { n = 1 } ^ { N } .$ , initial point $\theta ^ { 0 } \in \Theta ,$ , solution pool $\begin{array} { r } { \mathcal { V } _ { 0 } . } \end{array}$ , tolerance   
$\epsilon _ { \mathrm { i n } } \geq 0 ,$ maximum number of iterations J   
1: $\begin{array} { r } { \mathcal { G }  \{ \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { \prime n } ) \ | \ v \in \mathcal { V } _ { 0 } \} } \end{array}$ , UB $ + \infty$ (initialization with inherited   
cuts)   
2: for $j = 0 , 1 , \ldots , J - 1$ do   
3: Compute $v _ { j } ^ { n }  x ^ { * } ( \theta ^ { j } , s ^ { n } )$ for each $n ,$ and set $\mathcal { V } _ { j + 1 }  \mathcal { V } _ { j } \cup \{ ( v _ { j } ^ { n } ) _ { n } \}$ (update   
of the pool)   
4: $\begin{array} { r } { g _ { j } \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v _ { j } ^ { n } - y ^ { \prime n } ) , \mathcal { G } \gets \mathcal { G } \cup \{ g _ { j } \} \ : ( \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = \langle \theta ^ { j } , g _ { j } \rangle ) } \end{array}$   
5: if $\langle \theta ^ { j } , g _ { j } \rangle < \mathrm { U B }$ then   
6: $\mathrm { U B } \gets \langle \theta ^ { j } , g _ { j } \rangle , \hat { \theta } \gets \theta ^ { j }$   
7: end if   
8: $\scriptstyle \operatorname { L B } \gets \operatorname* { m i n } _ { \theta \in \Theta } \operatorname* { m a x } _ { g \in \mathcal { G } } \langle \theta , g \rangle$   
9: if $\mathrm { U B } - \mathrm { L B } \le \epsilon _ { \mathrm { i n } }$ then   
10: return $( { \hat { \theta } } ,$ converged = TRUE, $\nu _ { j + 1 } )$   
11: end if   
12: Choose the next evaluation point $\theta ^ { j + 1 } \in \Theta$ according to the selection rule of   
the inner solver (see §D for details).   
13: end for   
14: return $( { \hat { \theta } } ,$ converged $= \mathrm { F A L S E } , ~ \mathcal { V } _ { J } )$

Remark E.2 (Reasons for adopting the inner solver and cut inheritance). There are two reasons for using the cutting-plane method in the inner loop. First, the unit of computational cost in our problem is a call to the forward-problem oracle (solving a MILP for each state), and the value and a subgradient of the objective function of the convex subproblem are obtained simultaneously from a single oracle evaluation (without additional solves). The cutting-plane method keeps all the information obtained in this way as the cut set ${ \mathcal { G } } ,$ and thus makes full use of the information per oracle evaluation. Moreover, since the convex subproblem is convex and piecewise linear (Equation (5.1)), minimizing the model requires only an LP, and an exact minimizer is reached in finitely many iterations (Lemma E.10). Furthermore, with cutting-plane finishing (§D.4), which switches the selection rule for evaluation points only once during the run, the number of iterations until reaching it can be bounded in terms of the number $T _ { 1 }$ of iterations of Phase 1 per call (Proposition F.6), M $\begin{array} { r } { ( M : = \prod _ { n = 1 } ^ { N } | \mathcal { Y } ( s ^ { n } ) | } \end{array}$ ; Lemma E.10), and $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ (Theorem F.3, Proposition ${ \mathrm { F } } . 6 ,$ and Remark F.8). Moreover, in the integer case with the level method as the rule R for selecting evaluation points before the switch in cutting-plane finishing (Algorithm 4), this number is given as an explicit function of the problem size, M, and $| \mathcal { V } _ { > } ^ { \mathrm { a l i } } |$ whenever the iteration bound of Remark F.4 holds (Corollary F.12). Second, the cutting-plane method returns the gap between the upper and lower bounds, UB−LB, as a certificate of convergence. Tie stopping in Algorithm 6 may stop erroneously at an accidental tie (a spurious fixed point) when the inner loop returns an unconverged approximate solution, and it is therefore adopted only when the convergence of the inner loop has been confirmed. This decision requires a certificate. When a gradient-based method without a lower-bounding model is used as the inner solver, convergence is decided from the value of the convex subproblem and the subgradient at the weight (e.g., for ONS and MetaGrad, when the value is at most $\epsilon _ { \mathrm { i n } }$ or the subgradient is 0; Algorithm 11). The reason for accumulating and reusing cuts (cut inheritance) is as follows. Cuts are obtained for free as by-products of oracle evaluations, and their validity depends only on the feasibility of the solutions, not on the choice of the branch (Proposition E.3). Hence they can be safely reused across branch updates and outer iterations. The efect can be quantified: the total number of inner iterations executed in K outer iterations decreases from $K ( M + 1 )$ without sharing the pool to $M + 2 K - 1$ (Lemma E.10; M is an upper bound on the total number of tuples of solutions that the oracle can return). Note that Algorithm 5 does not fix the selection rule for the next evaluation point $\theta ^ { j + 1 }$ . Since UB is the minimum of the values actually evaluated and LB is a lower bound obtained from the model of cuts, the validity of the stopping criterion $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { i n } }$ (the ϵ<sub>in</sub>-optimality of $\hat { \theta } )$ is preserved however $\theta ^ { j + 1 }$ is chosen. The pseudocode of each rule is given in §D.

Algorithm 6 DC optimization of the empirical mean of the OS loss (simplified   
DCA + cut inheritance)   
Input: Data $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N } ,$ oracles $x ^ { * } ( \cdot , s ^ { n } )$ and $x ^ { * } ( \cdot ; x ^ { n } , s ^ { n } )$ , initial point $\theta ^ { 0 } \in \Theta$   
thresholds $\epsilon _ { \mathrm { g a p } } , \epsilon _ { \mathrm { t i e } } \ge 0$ , maximum number of iterations K   
1: $\mathcal { V } ^ { 0 }  \emptyset ,$ and $\bar { y ^ { \prime n } } \gets x ^ { * } ( \theta ^ { 0 } ; x ^ { n } , s ^ { n } )$ for each n (initialization of the branch)   
2: for $k = 0 , 1 , \ldots , K - 1$ do   
3: Apply Algorithm 5 to $( y ^ { \prime } , \theta ^ { k } , \mathcal { V } ^ { k } )$ and obtain $( \theta ^ { k + 1 }$ , converged, $\widetilde { \nu } ^ { k } )$   
4: For each $n , \ v ^ { n }  x ^ { * } ( \theta ^ { k + 1 } , s ^ { n } )$ and $y ^ { \prime \prime } { } ^ { n }  x ^ { * } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } ) ; \mathcal { V } ^ { k + 1 }  \widetilde { \mathcal { V } } ^ { k } \cup$   
$\left\{ ( v ^ { n } ) _ { n } \right\}$ (update of the pool)   
5: $\begin{array} { r } { \ell _ { k } \gets \frac { \mathrm { i } } { N } \sum _ { n = 1 } ^ { N } \langle \theta ^ { k + 1 } , v ^ { n } - y ^ { \prime \prime } { } ^ { n } \rangle \ ( = \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k + 1 } ) ) } \end{array}$   
6: $\begin{array} { r } { \Delta _ { k } \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \langle \theta ^ { k + 1 } , y ^ { \prime \prime { n } } - y ^ { \prime { n } } \rangle } \end{array}$   
7: if $\ell _ { k } \le \epsilon _ { \mathrm { g a p } }$ then   
8: break (certificate stopping; certificate of global optimality; Proposition E.9)   
9: end if   
10: if $\Delta _ { k } \le \epsilon _ { \mathrm { t i e } }$ and converged = TRUE then   
11: break (tie stopping; fixed point of DCA)   
12: end if   
13: $y ^ { \prime n }  y ^ { \prime \prime n }$ (∀n) (update of the branch)   
14: end for   
15: return In the case of target-attainment stopping (see Equation (E.6)), that   
point; otherwise, among the points $\theta ^ { k + 1 }$ of the outer iterations so far, the one   
with the smallest $\ell _ { k }$ (if several points attain the minimum, the one from the   
earliest iteration)

The outer loop (Algorithm 6) is simplified DCA, which repeats fixing the branch by the dominating vertices (linearization), solving the convex subproblem in the inner loop, and updating the branch. The inner loop (Algorithm 5) is a method based on a cutting-plane model for the convex subproblem min $\boldsymbol { \cdot } \theta \in \Theta \ \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } )$ (if the selection rule for evaluation points is taken to be the minimizer of the model, it becomes the cutting-plane method (Kelley, 1960)). The linearization of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ at a point $\theta ^ { j }$ is, with $v _ { j } ^ { n } : = x ^ { * } ( \theta ^ { j } , s ^ { n } )$ and $\begin{array} { r } { g _ { j } : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v _ { j } ^ { n } - y ^ { \prime n } ) } \end{array}$ 2

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) ~ \geq ~ \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) + \langle g _ { j } , \theta - \theta ^ { j } \rangle ~ = ~ \langle \theta , g _ { j } \rangle ,\tag{E.5}
$$

which is a homogeneous linear lower bound (cut) with intercept 0 (by $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) =$ $\left. \theta ^ { j } , g _ { j } \right. )$ . The problem of minimizing the model of cuts (the master problem) is the minimization over Θ of the maximum of the cuts, that is, a linear program (LP).

There are three stopping criteria. First, certificate stopping (stopping by $\ell _ { k } \le \epsilon _ { \mathrm { g a p } }$ in Algorithm 6): since the OS loss is nonnegative (§4), its value itself is an upper bound on the gap to the global optimal value, and if $\ell _ { k } \ \leq \ \epsilon _ { \mathrm { g a p } }$ , then $\theta ^ { k + \bar { 1 } }$ is $\epsilon _ { \mathrm { g a p } } \mathrm { { - g l o b a l l y } }$ optimal (Proposition E.9). This stopping does not depend on the convergence of the inner solver and is always valid. Second, tie stopping (stopping by $\Delta _ { k } \le \epsilon _ { \mathrm { t i e } } ) \colon \Delta _ { k } = 0$ means that the branch $y ^ { \prime }$ used for the linearization is also optimal as a linearization of the concave part at $\theta ^ { k + 1 }$ (a fixed point of DCA has been reached). However, a $\theta ^ { k + 1 }$ returned while the inner loop has not converged is an approximate solution of the convex subproblem, and accidental ties (spurious fixed points) can occur for combinatorial forward problems; hence, in the implementation, tie stopping is adopted only when the inner solver reports convergence. Third, target-attainment stopping: the outperformance prediction loss

$$
\ell ^ { \mathrm { O P } } ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { d } \operatorname* { m a x } \{ x _ { i } ^ { n } - x _ { i } ^ { * } ( \theta , s ^ { n } ) , 0 \}\tag{E.6}
$$

is nonnegative, and $\ell ^ { \mathrm { O P } } ( \theta ) = 0$ is equivalent to $x ^ { * } ( \theta , s ^ { n } ) \geq x ^ { n } \ ( \forall n )$ , that ${ \mathrm { i s } } ,$ to the attainment of Equation (3.2). The implementation computes $\ell ^ { \mathrm { O P } }$ every time it evaluates $x ^ { * } ( \cdot , s ^ { n } )$ , including at the inner evaluation points, and as soon as it falls below a threshold (default 0), it outputs that θ and immediately stops the whole procedure.

We describe the main points of the implementation. Besides the cutting-plane method (the default), the inner solver can be replaced by the projected subgradient method, the proximal bundle method (Kiwiel, 1990), the level method (Lemaréchal et al., 1995), the ellipsoid method, the analytic center cutting-plane method, the center of gravity method, the randomized center of gravity method, ONS, or MetaGrad (Algorithm 11), and every solver that has a lower-bounding model (all except the projected subgradient method, ONS, and MetaGrad) can use inherited cuts (corresponding to the initialization of G in Algorithm 5). The initial point of the inner loop is $\theta ^ { 0 } = \theta ^ { k }$ (warm start) (when ONS or MetaGrad is used as the inner solver, it resumes from the state of the learner at the end of the previous outer iteration), and the return value is projected onto the probability simplex to clean up numerical errors. The default values of the implementation are $J = 2 0$ (the inner loop is an inexact solve that allows truncation), $K = 1 0 0 , \epsilon _ { \mathrm { t i e } } = 1 0 ^ { - 8 }$ , and $\epsilon _ { \mathrm { g a p } } = 1 0 ^ { - 4 }$ (the values used in the experiments are given in §H.3). Computational cost is accounted for by the number of calls to the forward-problem oracle rather than by the number of iterations (calls to the forward-problem oracle occur only in computing $v _ { j }$ in Algorithm 5 and $v , y ^ { \prime \prime }$ and the initialization of the branch in Algorithm $6 ;$ the additional calls when ONS or MetaGrad is used as the inner solver are shown in Algorithm 11).

E.4. Cut inheritance. In DC programming, the idea of keeping the cuts (cuttingplane model) of the convex component across iterations was proposed by de Oliveira (2019) as a proximal bundle method for DC programming. That paper takes as its issue that DCA and the proximal linearized method (PLM) must solve a nonsmooth convex subproblem exactly at each iteration, and replaces the convex component $f _ { 1 }$ in the subproblem by a cutting-plane model ${ \breve { f } } _ { 1 } ^ { k }$ constructed from an information bundle $B _ { 1 } ^ { k }$ that accumulates the linearizations obtained in past iterations. Whereas the linearization of the concave component $- f _ { 2 }$ is updated only when the stability center is updated, the model of $f _ { 1 }$ is kept across iterations, so the convex subproblem need not be solved exactly. Cut inheritance, described in this subsection, instantiates this idea in accordance with the structure of our problem that the convex component $F _ { 1 }$ is a support function.

In the initialization of $\mathcal { G }$ in Algorithm 5, tuples of solutions of the forward-problem oracle obtained in the past, $v = ( v ^ { n } ) _ { \mathrm { { } } }$ <sub>n</sub> (the pool V), are converted into cuts for the current branch $y ^ { \prime }$ and used as initial cuts. We call the cuts made in this way from the tuples in the pool V inherited cuts, and this mechanism cut inheritance. Their validity follows from the next proposition.

Proposition E.3 (Validity of cuts). Let $v = ( v ^ { n } ) _ { n = 1 } ^ { N }$ be any tuple satisfying $v ^ { n } \in \mathcal { X } ( s ^ { n } ) ~ ( n = 1 , \ldots , N )$ . Then, for any branch $y ^ { \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } }$ , we have

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) \geq \left. \theta , \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { \prime n } ) \right. \qquad ( \forall \theta \in \mathbb { R } ^ { d } ) .
$$

Furthermore, if each $v ^ { n }$ attains $F _ { 1 } ( \theta ^ { 0 } ; s ^ { n } )$ at some $\theta ^ { 0 }$ , then equality holds at $\theta = \theta ^ { 0 }$

Proof. Since $v ^ { n } \in \mathcal { X } ( s ^ { n } )$ , we have $F _ { 1 } ( \theta ; s ^ { n } ) \ge \langle \theta , v ^ { n } \rangle$ for every θ. Hence

$$
\begin{array} { l } { \displaystyle \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ F _ { 1 } ( \theta ; s ^ { n } ) - \langle \theta , y ^ { \prime n } \rangle \bigr ] \geq \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \langle \theta , v ^ { n } - y ^ { \prime n } \rangle } \\ { \displaystyle \qquad = \left. \theta , \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { \prime n } ) \right. . } \end{array}
$$

The equality condition follows by substituting $F _ { 1 } ( \theta ^ { 0 } ; s ^ { n } ) = \langle \theta ^ { 0 } , v ^ { n } \rangle \ ( \forall n )$

As Proposition E.3 shows, these cuts have the following properties. First, the validity of a cut depends only on the feasibility of $v \ ( v ^ { n } \in { \mathcal { X } } ( s ^ { n } ) )$ , and not on at which θ, under which branch, or by which solver v was obtained. Hence the same pool V can be shared across branch updates, DC iterations, types of inner solvers, and diferent runs of DC optimization (as long as the sequence of states $\{ s ^ { n } \}$ is the same). Second, since v need not even be optimal (it is valid as a lower bound as long as it is feasible), the cuts are robust to truncation and approximate solution of the forward-problem solver (although the tightness of the cuts depends on optimality). Third, v has already been obtained as a by-product of evaluating $F _ { 1 }$ , and generating the cuts requires no additional forward-problem solves. These are consequences of $F _ { 1 }$ being a support function (positively homogeneous of degree 1), in contrast to cuts of a general convex function, which have nonzero intercepts depending on the linearization point.

E.5. Underlying literature. Every component of the skeleton of our method is a known method. We make the correspondence explicit. The outer loop is, among the variants of DCA formulated by Pham Dinh and Le Thi (1997) for DC programming, simplified DCA, which linearizes the concave part by a subgradient (here, the dominating vertices). Our setting fits directly into the polyhedral DC programs of Section 4 of that paper. Indeed, in

$$
\begin{array} { r l } & { \quad ( \tilde { \mathrm { P } } ) \colon \ \operatorname* { i n f } \{ \phi ( \theta ) - \tilde { h } ( \theta ) \ | \ \theta \in \mathbb { R } ^ { d } \} , } \\ & { \quad \tilde { h } ( \theta ) = \operatorname* { m a x } \{ \langle a ^ { i } , \theta \rangle - \alpha ^ { i } \ | \ i \in I \} \quad ( I \ \mathrm { i s ~ a ~ f i n i t e ~ s e t } ) } \end{array}
$$

of that paper, it sufices to set

$$
\phi ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } F _ { 1 } ( \theta ; s ^ { n } ) + \chi _ { \Theta } ( \theta ) , \qquad \widetilde { h } ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } F _ { 2 } ( \theta ; x ^ { n } , s ^ { n } )
$$

(where $\chi _ { \Theta }$ is the indicator function of Θ). By Equation (E.2),

$$
\tilde { h } ( \theta ) = \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } } \left. \theta , \frac { 1 } { N } \sum _ { n = 1 } ^ { N } y ^ { \prime n } \right. ,
$$

so $\tilde { h }$ is a polyhedral convex function with $\begin{array} { r } { a ^ { y ^ { \prime } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } y ^ { \prime n } , \alpha ^ { y ^ { \prime } } = 0 } \end{array}$ , and $I = \mathcal { V } _ { > } ^ { \mathrm { a l l } }$ (and ϕ is polyhedral convex as well), and the convex subproblem $( { \mathrm { P } } _ { i } ) ~ ( \operatorname* { i n f } _ { \theta } \{ \phi ( \theta ) { \bf \bar { \theta } } -$ $\left( \langle a ^ { i } , \theta \rangle - \alpha ^ { i } ) \right\} )$ of that paper coincides exactly with the convex subproblem of the branch $\begin{array} { r } { y ^ { \prime } \ m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) } \end{array}$ of this section. Hence Equation (5.3) of Proposition 5.2 is an instantiation, for our problem, of the equality (where α is the optimal value of the DC program (P<sup>˜</sup>): inf<sub>θ</sub> $\{ \phi ( \theta ) - \tilde { h } ( \theta ) \}$ of that paper) $\begin{array} { r } { \alpha = \operatorname* { i n f } _ { i \in I } \operatorname* { i n f } _ { \theta } \{ \phi ( \theta ) - ( \langle a ^ { i } , \theta \rangle - \alpha ^ { i } ) \} } \end{array}$ in Section 4.1 of that paper and of the remark at the beginning of Section 4.2 that “solving the polyhedral DC program $( \tilde { \mathrm { P } } )$ globally reduces to solving |I| convex programs $( \mathrm { P } _ { i } ) . ^ { \mathfrak { n } }$ The monotone decrease of the outer loop (Proposition E.11) corresponds to Theorem $5 ( \mathrm { i } )$ of that paper (although Proposition E.11 difers in that it allows inexact solution of the inner problem). The relation between finite convergence (Proposition E.4) and Theorem 6 of that paper was discussed in Remark E.6. For later developments of DCA, see Le Thi and Pham Dinh (2018). The default inner loop is the cutting-plane method (Kelley, 1960); the facts that the convex subproblem reduces to the minimization of the suboptimality loss (Mohajerin Esfahani et al., 2018) and that its subgradient is obtained from the forward-problem oracle share the same structure as the subgradient computation of Bärmann et al. (2017). Among the interchangeable inner solvers, the proximal bundle method is due to Kiwiel (1990) and the level method to Lemaréchal et al. (1995). The idea of keeping the cuts (model) of the convex part across updates of the linearization of the concave part was proposed by de Oliveira (2019), as described in §E.4. Related prior work includes the double bundle method (Joki et al., 2018), which has a bundle for each of the two DC components, and BEM-DC (Gaudioso et al., 2023), which assumes polyhedral DC and distributes cuts to both components; the latter is the closest to our method. The diferences from these are that (i) the cuts are made not from a subgradient oracle but from the feasible solutions themselves returned by the forward-problem oracle, and, by the support-function structure, they are global lower bounds with intercept 0 that do not depend on the branch (Proposition E.3); (ii) since their validity depends only on feasibility, they are robust to truncation; (iii) they are kept not only within a single run but across types of inner solvers and runs (in the experiments of this paper, the pool was shared within a single run of DC optimization); and (iv) the method is applied to outperformance inverse optimization (OS loss minimization).

E.6. Finite convergence of DCA and local optimality. This subsection describes the finite convergence of the DC optimization method of §5.2 and local optimality. We use the notation of §5.2 (in particular, the finite set of branches $\bar { \mathcal { I } } _ { \geq } ^ { \mathrm { a l l } }$ and the minimum value of the convex subproblem $m _ { \mathrm { s u b } } ( y ^ { \prime } ) )$ ). We also write the branch at iteration k of Algorithm 6 as $y ^ { ( k ) } : = ( x ^ { * } ( \theta ^ { k } ; x ^ { n } , s ^ { n } ) ) _ { n = 1 } ^ { N }$ . Then, if the inner convex subproblems are solved exactly, DC optimization stops in finitely many iterations.

Proposition E.4 (Finite convergence of exact DCA). Under Assumption 4.2, suppose that in line 3 of Algorithm 6 (the application of Algorithm 5), the inner convex subproblem is solved exactly, that is, $\begin{array} { r } { \theta ^ { k + 1 } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } \big ( \theta ; y ^ { ( k ) } \big ) } \end{array}$ (by Lemma E.10, this is possible with finitely many oracle calls, in which case we may set converged = TRUE). With $\epsilon _ { \mathrm { t i e } } = 0$ and $K = \infty$ , Algorithm 6 stops within at most $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ outer iterations (by certificate stopping or tie stopping). In the case of tie stopping, the point $\theta ^ { k + 1 }$ of the iteration at which it stopped satisfies

$$
\bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k + 1 } ) = m _ { \mathrm { s u b } } ( y ^ { ( k ) } ) = \operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { ( k ) } ) .\tag{E.7}
$$

That is, $\theta ^ { k + 1 }$ is an exact minimizer of the convex subproblem for its own branch (a fixed point of DCA). Since the output of Algorithm 6 is the point with the smallest $\ell _ { k }$ , its empirical mean of the OS loss is at most $m _ { \mathrm { s u b } } ( y ^ { ( k ) } )$ .

The proof is given in §E.7.

Proposition E.5 (Local optimality of tie-stopping points). In the setting of Proposition E.4, suppose that Algorithm 6 stops by tie stopping at iteration $k .$ If arg $\begin{array} { r } { \operatorname* { m a x } _ { x \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta ^ { k + 1 } , x \rangle } \end{array}$ is a singleton for every $n ,$ then $\theta ^ { \bar { k } + 1 }$ is a local minimizer of $\bar { \ell } ^ { \mathrm { O S } }$ over Θ.

Proof. Let $y ^ { \prime } : = y ^ { ( k ) }$ . Since $y ^ { \prime \prime { n } } : = x ^ { * } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } )$ is an optimal solution of $\begin{array} { r } { \operatorname* { m a x } _ { x \in \mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta ^ { k + 1 } , x \rangle } \end{array}$ and $y ^ { \prime n } \in { \mathcal { X } } _ { \geq x ^ { n } } \left( s ^ { n } \right)$ , each term $\langle { \theta ^ { k + 1 } , y ^ { \prime \prime n } - y ^ { \prime n } } \rangle$ ⟩ of $\Delta _ { k }$ in Algorithm 6 is nonnegative. Under tie stopping, $\Delta _ { k } \le \epsilon _ { \mathrm { t i e } } = 0$ , so each term is $0 ,$ and $y ^ { \prime n }$ is also an optimal solution at $\theta ^ { k + 1 }$ . By assumption, this optimal solution is unique, so arg $\begin{array} { r } { \operatorname { n a x } _ { x \in \mathcal { X } _ { > x ^ { n } } ( s ^ { n } ) } \langle \theta ^ { k + 1 } , x \rangle = \{ y ^ { \prime n } \} } \end{array}$ . Let $V _ { n }$ be the set of extreme points of the pieces of $\scriptstyle { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ (the nonempty intersections $P \cap \{ x \in \mathbb { R } ^ { d } \mid x \geq x ^ { n } \}$ with the pieces P of $ { \mathcal { X } } ( s ^ { n } ) )$ . Then $V _ { n }$ is finite, $y ^ { \prime n } \in V _ { n }$ by Assumption $4 . 2 ( 3 )$ , and $\begin{array} { r } { \operatorname* { m a x } _ { x \in \mathcal { K } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle = \operatorname* { m a x } _ { v \in V _ { n } } \langle \theta , v \rangle } \end{array}$ for every $\theta \in \mathbb { R } ^ { d }$ . By the uniqueness of the optimal solution, $\langle \theta ^ { k + 1 } , y ^ { \prime n } \rangle > \langle \theta ^ { k + 1 } , v \rangle$ for every $v \in V _ { n } \setminus \{ y ^ { \prime n } \}$ . Since $V _ { n }$ and the number of n are finite, there exists a neighborhood U of $\theta ^ { k + 1 }$ on which $\left. \theta , y ^ { \prime n } \right. >$ $\langle \theta , v \rangle$ for every n and every $v \in V _ { n } \setminus \{ y ^ { \prime n } \}$ . Hence max ${ } _ { x \in { \mathcal { X } } _ { > x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle = \langle \theta , y ^ { \prime n } \rangle$ for $\theta \in U$ , and by Equations (4.2) and (5.1), $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) = \bar { \ell } ^ { \mathrm { s u } \bar { \mathrm { b } } } ( \theta ; y ^ { \prime } )$ on $U \cap \Theta$ . By Proposition $\operatorname { E . 4 } , \ \dot { \theta } ^ { k + 1 }$ is a minimizer of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ over Θ, and by Equation (E.7), $\bar { \ell } ^ { \mathrm { O S } } \bar { ( } \theta ^ { k + 1 } ) = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k + 1 } ; y ^ { \prime } )$ . Therefore, for every $\theta \in U \cap \Theta$ ， $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) \ge$ $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k + 1 } ; y ^ { \prime } ) = \bar { \ell } ^ { \mathrm { { O S } } } ( \theta ^ { k + 1 } )$ □

Remark E.6 (Relation to a known finite convergence theorem). Pham Dinh and Le Thi (1997) proved the finite convergence of simplified DCA (Theorem 6 of that paper) for DC programs in $\mathrm { f } _ { \theta } \{ \phi ( \theta ) - \tilde { h } ( \theta ) \}$ (ϕ is a convex function) whose concave part is explicitly written as a polyhedral convex function

$$
\tilde { h } ( \theta ) = \operatorname* { m a x } _ { i \in I } \bigl ( \langle a ^ { i } , \theta \rangle - \alpha ^ { i } \bigr ) \qquad ( I \mathrm { ~ i s ~ a ~ f i n i t e ~ i n d e x ~ s e t } , a ^ { i } \in \mathbb { R } ^ { d } , \ \alpha ^ { i } \in \mathbb { R } ) .
$$

The empirical mean of the OS loss can also be written in this form $\phi - \tilde { h }$ , and the index i of the linear pieces corresponds to the branch $y ^ { \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } } \ ( \ S \mathrm { E . 5 } )$ . However, that theorem assumes the natural choice of subgradients, that is, a choice in the relative interior of $\partial \tilde { h } ( \theta )$ that puts positive weights on all active linear pieces (Lemma 3 of that paper), and implementing it requires enumerating all linear pieces active at each point θ, that is, all dominating vertices that are optimal solutions of $\operatorname* { m a x } _ { x \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle$ for each state $s ^ { n }$ . Since the forward-problem oracle returns only one optimal solution, this enumeration is dificult in general, and applying that theorem directly to our setting is dificult. Therefore, in this paper, we prove finiteness directly using the tie stopping rule in the setting where the oracle returns only one optimal solution (Proposition E.4).

Remark E.7 (Guarantee of global optimality). The point obtained by tie stopping is a minimizer for the branch $y ^ { ( k ) }$ . Global optimality is guaranteed by certificate stopping $( \ell _ { k } \le \epsilon _ { \mathrm { g a p } } ;$ Proposition E.9); in particular, in settings where Equation (3.2) is realizable (the global minimum value is 0), reaching the global minimum can be confirmed by a loss value of 0. Note that, when $\Theta \bar { = } \Delta ^ { d - \bar { 1 } }$ , the set of weights at which the optimal solution is not unique has measure 0 on $\Delta ^ { d - 1 }$ (Proposition E.8).

Proposition E.8 (Kitaoka, 2024, Lemma 3.3). Assume Assumption 4.2(2) and let $\bar { \Theta } = \Delta ^ { d - 1 }$ . Then, for any states $s ^ { 1 } , \ldots , s ^ { N } \in \mathcal { S }$ and $x ^ { n } \in { \mathcal { X } } ( s ^ { n } ) ~ ( n = 1 , . . . , N )$ for almost every $\theta \in \Delta ^ { d - \bar { 1 } }$ (with respect to the measure on $\Delta ^ { d - 1 }$ induced by the Lebesgue measure), both arg ${ \mathrm { m a x } } _ { x \in \mathcal { X } ( s ^ { n } ) } \langle \theta , x \rangle$ and arg $\operatorname* { m a x } _ { x \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } ) } \langle \theta , x \rangle$ are singletons for every n.

Kitaoka (2024, Lemma 3.3) shows that, when the feasible region of each state is a finite union of bounded closed convex polytopes, for almost every $\theta \in \Delta ^ { d - 1 }$ the feature of the optimal solution (in that paper, the image $f ( x , s )$ of a feasible solution x under the feature map $f$ is called the feature; our setting corresponds to $f ( x , s ) = x )$ is uniquely determined for every n. Its proof shows that the maximizer of ⟨θ, ·⟩ over the finite set of extreme points of the pieces is unique. If the maximizer among the extreme points is unique, the optimal face of each piece is a single point; hence, applying this with the identity feature map to $\mathcal X ( s ^ { n } )$ and $\scriptstyle { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ , both of which are finite unions of nonempty bounded polyhedra by Remark 4.3, yields Proposition E.8. Although that lemma is stated under the assumption that the observations are optimal solutions for a true weight, its proof does not use this assumption.

E.7. Theoretical analysis. Complete proofs are given for all propositions in this subsection. We first state the validity of certificate stopping.

Proposition E.9 (Certificate of global optimality). For any $\theta \in \Theta$ and $\epsilon \geq 0$ , if $\bar { \ell } ^ { \mathrm { O S } } ( \bar { \boldsymbol { \theta } } ) \leq \epsilon .$ , then we have

$$
\bar { \ell } ^ { \mathrm { O S } } ( \theta ) \leq \operatorname* { m i n } _ { \theta ^ { \prime } \in \Theta } \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { \prime } ) + \epsilon .
$$

In particular, if $\epsilon = 0$ , then $\theta$ is a global minimizer of Equation (4.4).

Proof. By the inclusion $\mathcal { X } _ { \geq x ^ { n } } ( s ^ { n } ) \subseteq \mathcal { X } ( s ^ { n } )$ , each term satisfies $\ell ^ { \mathrm { O S } } ( \theta ^ { \prime } ; x ^ { n } , s ^ { n } ) \geq 0$ and hence $\mathrm { m i n } _ { \theta ^ { \prime } \in \Theta } \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { \prime } ) \bar { \geq } 0$ . Therefore $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) \leq \epsilon \leq$ min<sub>θ</sub>′ $\bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { \prime } ) + \epsilon$ □

Next, we show that the inner cutting-plane method reaches an exact minimizer in finitely many iterations in the ideal setting (no truncation and tolerance 0).

Lemma E.10 (Finite exactness of the inner loop). Under Assumption 4.2, let $\epsilon _ { \mathrm { i n } } = 0$ and $J = \infty$ , suppose that the LP in Algorithm 5 is solved exactly, and let the selection rule for evaluation points be the cutting-plane method $( \theta ^ { j + 1 }$ is a minimizer attaining LB). Set $\begin{array} { r } { M : = \prod _ { n = 1 } ^ { N } \left| \mathcal { V } ( s ^ { n } ) \right| } \end{array}$ (an upper bound on the total number of values that a tuple of oracle solutions $( v ^ { n } ) _ { n }$ can take).

(1): When Algorithm 6 calls Algorithm 5 once, the call stops within at most $M + 1$ inner iterations (iterations of the loop over $j$ in Algorithm 5), and the returned $\hat { \theta }$ is an exact minimizer of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ over Θ. This bound does not depend on the contents of the pool $\nu$ at the time of the call.

(2): When Algorithm 6 is run for K outer iterations (iterations of the loop over $k )$ , the total number of inner iterations executed during the run is at most $M + 2 K - 1$

Proof. Let $\mathcal { G } _ { j }$ be the cut set at the end of iteration $j ,$ and let $\begin{array} { r l } { \mathrm { L B } _ { j + 1 } } & { { } = } \end{array}$ min<sub>θ∈Θ</sub> $\operatorname* { m a x } _ { g \in { \mathcal { G } } _ { j } } \langle \theta , g \rangle$ be the optimal value of the LP. By Proposition E.3, all cuts in $\mathcal { G } _ { j }$ (including the inherited cuts) are linear lower bounds of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ , so $\mathrm { L B } _ { j + 1 } \le m _ { \mathrm { s u b } } ( y ^ { \prime } )$ , and since $\mathcal { G } _ { j }$ is monotonically increasing, $\mathrm { L B } _ { j + 1 }$ is nondecreasing in j. Moreover, since UB is the minimum of the values actually evaluated, $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { i } ; y ^ { \prime } ) = \langle \theta ^ { i } , g _ { i } \rangle$ , we always have $\mathrm { U B } \geq m _ { \mathrm { { s u b } } } ( y ^ { \prime } )$ . Therefore, at stopping $( \mathrm { U B } \leq \mathrm { L B } ) , \mathrm { U B } = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ , and $\hat { \theta }$ is a minimizer.

We show finiteness. Let $\mathcal { V } _ { 0 }$ be the pool at the time of the call. By the initialization of $\mathcal { G }$ in Algorithm 5, the cut of each tuple in $\mathcal { V } _ { 0 }$ (for the current branch $y ^ { \prime } )$ is already contained in $\mathcal { G }$ at the first inner iteration. The tuple of oracle solutions $v _ { j }$ takes values in the finite set $\textstyle \prod _ { n } { \mathcal { V } } ( s ^ { n } )$ . Suppose that at some inner iteration $j \geq 1$ ， $v _ { j }$ coincides with an element of $\mathcal { V } _ { 0 } \cup \{ v _ { 0 } , \dotsc , v _ { j - 1 } \}$ . Then its cut equals $g _ { j }$ and belongs to $\mathcal { G } _ { j - 1 }$ . Since $\theta ^ { j } \ ( j \geq 1 )$ is an optimal solution of the LP over $\mathcal { G } _ { j - 1 }$

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = \langle \theta ^ { j } , g _ { j } \rangle \leq \operatorname* { m a x } _ { g \in \mathcal { G } _ { j - 1 } } \langle \theta ^ { j } , g \rangle = \mathrm { L B } _ { j } .
$$

Hence, after the update at inner iteration $j , \mathrm { U B } \le \mathrm { L B } _ { j } \le \mathrm { L B } _ { j + 1 }$ , and the call stops at this iteration. By contraposition, at an inner iteration $j \geq 1$ that does not stop, $v _ { j }$ is always a tuple newly added to V. $\mathrm { A t } ~ j = 0$ , the initial point $\theta ^ { 0 } = \theta ^ { k }$ is a warm start and is not necessarily an optimal solution of the LP, so $v _ { 0 }$ may coincide with an element of $\mathcal { V } _ { 0 }$ without stopping; this can happen only if $| \nu _ { 0 } | \geq 1$ . Since $| \nu | \leq M$ the inner iterations that do not stop occur at most $( M - | \mathcal { V } _ { 0 } | ) + \operatorname* { m i n } \{ 1 , | \mathcal { V } _ { 0 } | \} \leq M$ times, and adding the stopping iteration, the number of inner iterations is at most $M + 1$ . This bound does not depend on $\mathcal { V } _ { 0 }$ . This proves (1).

We show (2). In Algorithm 6, the pool V increases monotonically across outer iterations and is never reinitialized. By the argument of (1), every inner iteration that does not stop either adds one new tuple to $\nu$ or is an iteration $j = 0$ whose tuple v already belongs to the pool. Throughout the whole run, the former occur at most M times. The latter occur at most once per call, and not in the first call, since $\nu ^ { 0 } = \varnothing ;$ hence they occur at most $K - 1$ times. In addition, each call has one stopping inner iteration, and the number of calls equals the number K of outer iterations; hence the total number of inner iterations is at most $M + ( K - 1 ) + K = M + 2 K - 1$ □

Lemma E.10(2) expresses the quantitative efect of cut inheritance (Proposition E.3). Without sharing the pool, the bound $M + 1$ of (1) is needed independently for each call, and the bound on the total is only $K ( M + 1 )$ . With a shared pool, a tuple obtained once is reused as a cut in subsequent calls, and an inner iteration $j \geq 1$ that returns a duplicate tuple causes stopping on the spot, so the total drops to $M + 2 K - 1$

For the outer loop, we first show, as a property that holds even if the inner loop is inexact, the monotone decrease of the OS loss.

Proposition E.11 (Descent property). Write the branch at iteration k of Algorithm 6 (the $y ^ { \prime }$ when Algorithm 5 is applied) as $y ^ { ( k ) } : = ( x ^ { * } ( \theta ^ { k } ; x ^ { n } , s ^ { n } ) ) _ { n = 1 } ^ { N }$ . If the inner solver returns $\theta ^ { k + 1 }$ satisfying

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k + 1 } ; y ^ { ( k ) } ) \leq \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k } ; y ^ { ( k ) } )\tag{E.8}
$$

(since Algorithm 5 first evaluates the initial point $\theta ^ { 0 } = \theta ^ { k }$ and returns the best of the evaluated points, it satisfies Equation (E.8) even when truncated), then we have

$$
0 \leq \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k + 1 } ) \leq \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k } ) .
$$

Proof. Since $y ^ { ( k ) , n }$ attains $F _ { 2 } ( \theta ^ { k } ; x ^ { n } , s ^ { n } )$ , we have $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k } ; y ^ { ( k ) } ) = \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k } )$ . On the other hand, since $y ^ { ( k ) , n } \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ , we have $F _ { 2 } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } ) \ge \langle \theta ^ { k + 1 } , y ^ { ( k ) , n } \rangle$ ⟩, so

$$
\bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k + 1 } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ F _ { 1 } ( \theta ^ { k + 1 } ; s ^ { n } ) - F _ { 2 } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } ) \bigr ] \leq \bar { \ell } ^ { \mathrm { u b } } ( \theta ^ { k + 1 } ; y ^ { ( k ) } ) .
$$

Combined with Equation (E.8), this gives the claim (nonnegativity holds as in the proof of Proposition E.9). □

Finally, we prove the finite convergence of the outer loop when the inner convex subproblems are solved exactly (Proposition E.4).

Proof of Proposition $E . 4$ . At iteration k, $y ^ { \prime \prime { n } } \ = \ y ^ { ( k + 1 ) , { n } }$ . First, since $y ^ { ( k ) , n } \in$ $\scriptstyle { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ and $y ^ { ( k + 1 ) , n }$ attains $F _ { 2 } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } )$ , each term of $\Delta _ { k }$ satisfies

$$
\langle \theta ^ { k + 1 } , y ^ { ( k + 1 ) , n } - y ^ { ( k ) , n } \rangle = F _ { 2 } ( \theta ^ { k + 1 } ; x ^ { n } , s ^ { n } ) - \langle \theta ^ { k + 1 } , y ^ { ( k ) , n } \rangle \ge 0 ;
$$

hence $\Delta _ { k } \geq 0$ , and moreover we have

$$
\begin{array} { r } { \bar { \ell } ^ { \mathrm { O S } } ( \theta ^ { k + 1 } ) = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k + 1 } ; y ^ { ( k ) } ) - \Delta _ { k } = m _ { \mathrm { s u b } } ( y ^ { ( k ) } ) - \Delta _ { k } . } \end{array}\tag{E.9}
$$

Here, the second equality follows from the exactness of the inner loop. Also, since $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ^ { k + 1 } ; y ^ { ( k + 1 ) } ) \stackrel { \bullet } { = } \bar { \ell } ^ { \mathrm { O S } } \bar { ( } \theta ^ { k + 1 } )$ ，

$$
m _ { \mathrm { s u b } } ( y ^ { ( k + 1 ) } ) \leq \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { k + 1 } ; y ^ { ( k + 1 ) } ) = m _ { \mathrm { s u b } } ( y ^ { ( k ) } ) - \Delta _ { k } .\tag{E.10}
$$

If neither stopping occurs at iteration $k ,$ then $\Delta _ { k } > 0$ , and by Equation (E.10), $m _ { \mathrm { s u b } } ( y ^ { ( k + 1 ) } ) < m _ { \mathrm { s u b } } ( y ^ { ( k ) } )$ , i.e., the branch minimum strictly decreases. Since $m _ { \mathrm { s u b } }$ takes values in the finite set $\{ m _ { \mathrm { { s u b } } } ( y ^ { \prime } ) ~ | ~ y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } \}$ , strict decreases can occur at most $| \mathcal { V } _ { > } ^ { \mathrm { a l l } } | - 1$ times, and one of the stoppings occurs within at most $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ iterations. At tie stopping, $\Delta _ { k } = 0$ , so Equation (E.9) gives Equation (E.7). □

## Appendix F. Exact solution of the inner convex subproblems based on gaps

In this section, we extend the finite exactness of Lemma E.10 (the guarantee when the rule for selecting evaluation points is the cutting-plane method) to stabilized selection rules such as the level method and the proximal bundle method. The key is to transplant to the inner convex subproblems the idea used in Kitaoka (2024, 2026): strengthen an approximation guarantee to exact attainment through a positive gap possessed by a piecewise-linear structure, and make the number of iterations explicit. Throughout this section, we assume Assumption 4.2 and fix a branch $y ^ { \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } }$

F.1. Submodel gap. Let $\begin{array} { r } { \mathcal { Y } ^ { \mathrm { a l l } } : = \prod _ { n = 1 } ^ { N } \mathcal { Y } ( s ^ { n } ) } \end{array}$ be the set of all tuples of solutions that the oracle can return (a finite set; $\bar { | } \mathcal { V } ^ { \mathrm { a l l } } | = M$ is the M of Lemma E.10), and write the cut of a tuple $v \in \mathcal { V } ^ { \mathrm { a l l } }$ as $\begin{array} { r } { g _ { v } : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { \prime n } ) } \end{array}$ (g<sub>v</sub> depends on the fixed branch $y ^ { \prime }$ , which we suppress in the notation). We call $g _ { v }$ a vertex cut. By Equation (E.2), we have

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) = \operatorname* { m a x } _ { v \in \mathcal { V } ^ { \mathrm { a l l } } } \langle \theta , g _ { v } \rangle \qquad ( \forall \theta \in \mathbb { R } ^ { d } ) .\tag{F.1}
$$

For a nonempty $W \subseteq \mathcal { V } ^ { \mathrm { a l l } }$ , we define the minimum value of the submodel by

$$
m _ { W } ( y ^ { \prime } ) : = \operatorname* { m i n } _ { \theta \in \Theta } \operatorname* { m a x } _ { v \in W } \langle \theta , g _ { v } \rangle .\tag{F.2}
$$

If $W \subseteq W ^ { \prime }$ , then $m _ { W } ( y ^ { \prime } ) \leq m _ { W ^ { \prime } } ( y ^ { \prime } )$ , and by Equation (F.1), $m _ { \mathcal { y } ^ { \mathrm { a l l } } } ( y ^ { \prime } ) = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ We define the submodel gap by

$$
\delta ( y ^ { \prime } ) : = m _ { \mathrm { s u b } } ( y ^ { \prime } ) - \operatorname* { m a x } \{ m _ { W } ( y ^ { \prime } ) \ : \ : \left| \ : \ : \emptyset \neq W \subseteq \mathcal { y } ^ { \mathrm { a l l } } , \ : m _ { W } ( y ^ { \prime } ) < m _ { \mathrm { s u b } } ( y ^ { \prime } ) \ : \right. \}\tag{F.3}
$$

(when the set over which the max on the right-hand side is taken is empty, we read $\delta ( y ^ { \prime } ) : = + \infty ) . \delta ( y ^ { \prime } )$ is not an assumption but a complexity constant that is automatically positive by finiteness.

Lemma F.1 (Positivity of the gap and discreteness of the lower bound). (i) $\delta ( y ^ { \prime } ) > 0$

(ii) When Algorithm 5 is run with the branch $y ^ { \prime }$ and a pool $\mathcal { V } _ { 0 }$ consisting only of tuples of solutions returned by the oracle, the value LB computed at each iteration equals $m _ { W } ( y ^ { \prime } )$ for some nonempty $W \subseteq \mathcal { V } ^ { \mathrm { a l l } }$ . In particular, the values that LB can take belong to the finite set $\{ m _ { W } ( y ^ { \prime } ) \mid \emptyset \neq W \subseteq \mathcal { V } ^ { \mathrm { a l l } } \}$ , and we always have $\mathrm { L B } \leq m _ { \mathrm { s u b } } ( y ^ { \prime } )$

Proof. (i) Since $y ^ { \mathrm { a l l } }$ is finite, it has finitely many subsets, so the largest of the submodel minimum values with $m _ { W } ( y ^ { \prime } ) < m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is attained and is strictly smaller than $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ . (ii) The cut set G of Algorithm 5 is always the set of all cuts $\{ g _ { v } \ | \ v \in W \}$ of the set W consisting of “the tuples of $\mathcal { V } _ { 0 } { } ^ { \mathrm { ~ , ~ } }$ together with “the tuples evaluated in that call”. Since every tuple added to the pool in Algorithms 5 and 6 is a return value of the oracle, $W \subseteq \mathcal { V } ^ { \mathrm { a l l } }$ , and $\begin{array} { r } { \mathrm { L B } = \operatorname* { m i n } _ { \theta \in \Theta } \operatorname* { m a x } _ { v \in W } \langle \theta , g _ { v } \rangle = } \end{array}$ m<sub>W</sub> $( y ^ { \prime } )$ . The inequality $\mathrm { L B } \leq m _ { \mathrm { s u b } } ( y ^ { \prime } )$ follows from monotonicity with respect to $W \subseteq \mathcal { \hat { V } } ^ { \mathrm { a l l } }$ □

## F.2. Exact agreement of the lower bound and finite exactness of cuttingplane finishing.

Lemma F.2 (Exact agreement of the lower bound). In the setting of Lemma F.1(ii), let $0 \le \epsilon _ { \mathrm { i n } } < \delta ( y ^ { \prime } )$ . When Algorithm 5 stops with $\mathrm { U B } - \mathrm { L B } \le \epsilon _ { \mathrm { i n } }$ (converged = TRUE), regardless of the rule for selecting evaluation points, we have

$$
\mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } ) , \qquad \bar { \ell } ^ { \mathrm { s u b } } ( \hat { \theta } ; y ^ { \prime } ) = \mathrm { U B } \leq m _ { \mathrm { s u b } } ( y ^ { \prime } ) + \epsilon _ { \mathrm { i n } } .
$$

That is, the minimum value of the convex subproblem itself is identified exactly.

Proof. Since UB is the minimum of the true values of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ at the evaluation points, $\mathrm { U B } \geq m _ { \mathrm { { s u b } } } ( y ^ { \prime } )$ . At stopping, LB $: \ge \mathrm { U B } - \epsilon _ { \mathrm { i n } } \ge m _ { \mathrm { s u b } } ( y ^ { \prime } ) - \epsilon _ { \mathrm { i n } } > m _ { \mathrm { s u b } } ( y ^ { \prime } ) -$ $\delta ( \boldsymbol { y } ^ { \prime } )$ . On the other hand, by Lemma F.1(ii), LB = m<sub>W</sub> $( y ^ { \prime } ) \leq m _ { \mathrm { { s u b } } } ( y ^ { \prime } )$ , and by Equation (F.3), every value with $m _ { W } ( y ^ { \prime } ) < m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is at most $m _ { \mathrm { s u b } } ( y ^ { \prime } ) - \delta ( y ^ { \prime } )$ ; hence $\mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ □

What becomes exact in Lemma F.2 is the lower bound (the identification of the minimum value), and the returned point $\hat { \theta }$ is $\epsilon _ { \mathrm { i n } } \mathrm { - o p t i m a l } .$ . To obtain an exact minimizer, we use the following cutting-plane finishing: in Phase 1, run with an arbitrary rule for selecting evaluation points (e.g., the level method) until $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { s w } }$ (where $\epsilon _ { \mathrm { s w } } > 0$ is the switching threshold); in the subsequent Phase $^ { 2 , }$ switch the selection rule to the cutting-plane method (the minimizer of the model) and continue with tolerance 0.

Theorem F.3 (Finite exactness of cutting-plane finishing). In the setting of Lemma F.1(ii), let the tolerance of Algorithm 5 be $0 , 0 < \epsilon _ { \mathrm { s w } } < \delta ( y ^ { \prime } ) , \epsilon _ { \mathrm { t e r m } } = 0$ in Algorithm 4, and $J = \infty ,$ , and assume that the LPs are solved exactly. If Phase 1 reaches $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { s w } }$ in finitely many iterations (which holds for the cutting-plane method by Lemma E.10 and, for the level method, whenever the iteration bound of Remark F.4 holds, $\mathrm { e . g . }$ , for the standard level method without inherited cuts), then cutting-plane finishing stops in finitely many iterations, and the returned $\hat { \theta }$ is an exact minimizer satisfying

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \hat { \theta } ; y ^ { \prime } ) = m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \operatorname* { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) .
$$

Proof. At the end of Phase 1, since $\mathrm { U B } - \mathrm { L B } \le \epsilon _ { \mathrm { s w } } < \delta ( y ^ { \prime } )$ , applying the proof of Lemma F.2 with $\epsilon _ { \mathrm { i n } }$ replaced by $\epsilon _ { \mathrm { s w } }$ gives $\mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ . In Phase 2, cuts are only added, so LB is nondecreasing, and by Lemma F.1(ii) it is always at most $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ ; hence $\mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is maintained. Each iteration of Phase 2 evaluates at the model minimizer $\theta ^ { j }$ (a point attaining LB). If the tuple $v _ { j }$ obtained by the evaluation already belongs to the set W corresponding to ${ \mathcal { G } } .$ , then

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = \langle \theta ^ { j } , g _ { v _ { j } } \rangle \leq \operatorname* { m a x } _ { g \in \mathcal { G } } \langle \theta ^ { j } , g \rangle = \mathrm { L B } = m _ { \mathrm { s u b } } ( y ^ { \prime } ) ,
$$

and since $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } ) \ge m _ { \mathrm { s u b } } ( y ^ { \prime } )$ always holds (on Θ), $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ . At this point the algorithm stops with $\mathrm { U B } = m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \mathrm { L B }$ , and $\hat { \theta } = \theta ^ { j }$ is an exact minimizer (it likewise stops if $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = m _ { \mathrm { s u b } } ( y ^ { \prime } )$ occurs with a new tuple). In every iteration that does not stop, a new tuple is added to the pool, and since $y ^ { \mathrm { a l l } }$ is finite, Phase 2 stops in finitely many iterations. □

Remark F.4 (Explicit number of iterations with the level method). For a convex Lipschitz function on a compact convex set, the number of iterations the level method needs to reach a certified gap $\mathrm { U B } - \mathrm { L B } \le \epsilon$ is bounded by $c ( \lambda ) \left( L _ { g } ( y ^ { \prime } ) D _ { \Theta } / \epsilon \right) ^ { 2 }$ $( \lambda \in \mathsf { \Gamma } ( 0 , 1 )$ is the level parameter and $c ( \lambda )$ is a constant depending only on it; Lemaréchal et al., 1995). Here $L _ { g } ( y ^ { \prime } ) : = \operatorname* { m a x } _ { v \in \mathcal { V } ^ { \mathrm { a l l } } } \| g _ { v } \|$ is an upper bound on the Lipschitz constant of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ , and $D _ { \Theta } : = \dim ( \Theta )$ . Therefore, choosing $\epsilon _ { \mathrm { s w } } < \delta ( y ^ { \prime } )$ , the number of iterations of Phase 1 is of order $c ( \lambda ) \left( L _ { g } ( y ^ { \prime } ) D _ { \Theta } / \delta ( y ^ { \prime } ) \right) ^ { 2 }$ and is bounded explicitly, independently of the bound $M + 1$ on the number of tuples in Lemma E.10. Note that this bound is a guarantee for the standard level method whose model is built only from tangents at the evaluation points. Using inherited cuts does not afect the validity of the stopping test (Lemma F.2). Evaluating the number of iterations with inherited cuts is future work.

Lemma F.5 (Total number of finishing iterations). Under Assumption 4.2, call Algorithm 5 at line 3 of Algorithm 6 with tolerance 0 and $J = \infty ,$ , and let the rule of its Step 12 be Algorithm 4. Assume that the LPs of Algorithm 5 are solved exactly, and let $\begin{array} { r } { M : = \prod _ { n = 1 } ^ { \bar { N } } | \mathcal { V } ( s ^ { n } ) | } \end{array}$ . An inner iteration $j ~ ( \geq 1 )$ is called a finishing iteration if its evaluation point $\theta ^ { j }$ is chosen in the cutting-plane branch of Algorithm 4 (the side $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { s w } } )$ . Then the total number of finishing iterations executed while Algorithm 6 runs up to K outer iterations is at most $M + K$ . This bound does not depend on the value of the switching threshold $\epsilon _ { \mathrm { s w } }$

Proof. We use the notation of the proof of Lemma E.10: let $\mathcal { G } _ { j }$ be the cut set at the end of inner iteration $j ,$ and $\begin{array} { r } { \mathrm { L B } _ { j + 1 } = \operatorname* { m i n } _ { \theta \in \Theta } \operatorname* { m a x } _ { g \in \mathcal { G } _ { j } } \langle \theta , g \rangle } \end{array}$ . In a finishing iteration $j , \theta ^ { j }$ is the output of Algorithm 1, that is, a minimizer on Θ of the model of $\mathcal { G } _ { j - 1 }$ so $\operatorname* { m a x } _ { g \in { \mathcal G } _ { j - 1 } } \langle \theta ^ { j } , g \rangle = \mathrm { L B } _ { j }$ . If the tuple $v _ { j }$ returned by the oracle already belongs to the pool V at that time, then its cut (for the current branch $y ^ { \prime } )$ equals $g _ { j }$ and belongs to $\mathcal { G } _ { j - 1 }$ , so

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) = \langle \theta ^ { j } , g _ { j } \rangle \leq \operatorname* { m a x } _ { g \in \mathcal { G } _ { j - 1 } } \langle \theta ^ { j } , g \rangle = \mathrm { L B } _ { j } \leq \mathrm { L B } _ { j + 1 } .
$$

After UB is updated, $\mathrm { U B } \le \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { j } ; y ^ { \prime } ) \le \mathrm { L B } _ { j + 1 }$ , so the stopping test with tolerance 0 is satisfied and the call stops at this iteration. By contraposition, in a finishing iteration that does not stop, $v _ { j }$ is necessarily a tuple newly added to the pool. In Algorithm 6, the pool V increases monotonically across outer iterations and is never reinitialized (as in the proof of Lemma E.10(2)). Therefore, throughout the whole run, the number of finishing iterations that do not stop is at most $| \nu | \leq M$ (iterations of Phase 1 also consume the pool, but this only acts to decrease the number of finishing iterations). Each call has at most one stopping iteration, and the number of calls equals the number K of outer iterations, so the total number of finishing iterations is at most $M + K$ . The threshold $\epsilon _ { \mathrm { s w } }$ does not appear in the above argument. □

Proposition F.6 (Total number of inner iterations over the whole run). Under Assumption 4.2, let the rule of Step 12 of Algorithm 5 be Algorithm 4 (with switching threshold $\epsilon _ { \mathrm { s w } }$ , termination tolerance $\epsilon _ { \mathrm { t e r m } } = 0$ , and Phase-1 rule R), and call Algorithm 5 with tolerance 0 and $J = \infty$ . Assume that the LPs are solved exactly, and let

$$
\delta _ { \mathrm { i n } } : = \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } } \delta ( y ^ { \prime } ) > 0 , \qquad 0 < \epsilon _ { \mathrm { s w } } < \delta _ { \mathrm { i n } }
$$

(since $\mathcal { V } _ { > } ^ { \mathrm { a l l } }$ is finite, $\delta _ { \mathrm { i n } } > 0 )$ . Assume further that for every branch, the rule R reaches $\mathrm { \bar { U } B } - \mathrm { L B } \le \epsilon _ { \mathrm { s w } }$ in at most $T _ { 1 }$ inner iterations. Then the following hold.

(1): Algorithm 6 stops in at most $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ outer iterations, and each call returns an exact minimizer of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$

(2): The total number of inner iterations executed over the whole run is at most

$$
\underbrace { \vert y _ { \ge } ^ { \mathrm { a l l } } \vert T _ { 1 } } _ { \mathrm { P h a s e ~ 1 } } + \underbrace { M + \vert y _ { \ge } ^ { \mathrm { a l l } } \vert } _ { \mathrm { f n i s h i n g } }
$$

$$
\begin{array} { r } { ( M : = \prod _ { n = 1 } ^ { N } | \mathcal { Y } ( s ^ { n } ) | ) . } \end{array}
$$

(3): The number of calls to the forward-problem oracle is at most $N \big ( | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | T _ { 1 } +$ $M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | ) + 2 N | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | + N .$

When the level method (Algorithm 3) is taken as the rule R, $T _ { 1 } = c ( \lambda ) \big ( L _ { g } D _ { \Theta } / \epsilon _ { \mathrm { s w } } \big ) ^ { 2 }$ $( L _ { g } : = \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { y } _ { > } ^ { \mathrm { a l l } } } L _ { g } ( y ^ { \prime } )$ , D<sub>Θ</sub> := diam(Θ); Remark F.4. This $T _ { 1 }$ is the value when the bound of Remark F.4 holds).

Proof. We show (1). Since $\epsilon _ { \mathrm { s w } } < \delta _ { \mathrm { i n } } \leq \delta ( y ^ { \prime } )$ holds for every branch $y ^ { \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } }$ , each call satisfies the assumptions of Theorem F.3, returns an exact minimizer, and gives converged = TRUE. This is the premise of Proposition E.4 (exact solution of the inner problems), and by that proposition, Algorithm 6 stops in at most $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ outer iterations.

We show (2). Split the inner iterations of one call into those up to the iteration at which $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { s w } }$ first holds and those after it. Since UB − LB is nonincreasing in the iterations (§D.4), this split yields two intervals, a first half and a second half, and all evaluation points of the second half are chosen in the cutting-plane branch of Algorithm 4. That is, all iterations of the second half are finishing iterations in the sense of Lemma F.5. By assumption, the number of iterations of the first half is at most $T _ { 1 }$ in each call, and by (1) the number of calls is at most $| \mathcal { V } _ { > } ^ { \mathrm { a l l } } |$ , so the total of the first halves is at most $| \mathcal { Y } _ { \ge } ^ { \mathrm { a l l } } | T _ { 1 }$ . Substituting $K \leq | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ into Lemma F.5, the total of the second halves is at most $M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ . Together, we obtain (2).

We show (3). Calls to the forward-problem oracle occur N times per inner iteration at Step 3 of Algorithm 5 and 2N times per outer iteration at Step 4 of Algorithm 6, and N times at the initialization of the branch in Step 1 of Algorithm 6, and nowhere else. By (1) and (2), the first two are bounded by $N ( | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | T _ { 1 } + M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | )$ and $2 N | \mathrm { | \mathcal { V } _ { > } ^ { a l l } | }$ , respectively, and the last is N. □

Remark F.7 (Roles of the two terms). The first term of Proposition F.6(2) depends on the numerical fineness of the problem through the gap $\delta _ { \mathrm { i n } } .$ , whereas the second term $M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ does not depend on the gap and is determined only by two combinatorial quantities: the number $M = | \mathcal { V } ^ { \mathrm { a l l } } |$ of tuples of solutions that the oracle can return and the number $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ of branches $\begin{array} { r } { ( \mathcal { Y } ^ { \mathrm { a l l } } = \prod _ { n } \mathcal { Y } ( s ^ { n } ) } \end{array}$ and $\begin{array} { r } { \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } = \prod _ { n } \mathcal { V } _ { \geq x ^ { n } } ( s ^ { n } ) } \end{array}$ are diferent sets). Without cut inheritance, the second term would be $| \mathcal { V } _ { > } ^ { \mathrm { a l l } } | ( M + 1 )$ (applying Lemma E.10(1) to each call), which difers by a factor of $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$

Remark F.8 (Connection to the outer loop and future work). First, the number of iterations and the number of oracle calls over the whole run are summarized in Proposition F.6. Second, a lower bound on $\delta _ { \mathrm { i n } }$ is given explicitly for the integer case in §F.3. Improvements using the structure of integer programs (total unimodularity, Graver bases, M-convexity) are left for future work, as described in Remark F.14. Third, the finite exact attainment of Kitaoka (2024) uses the structure in which the set of minimizers has a relative interior point, but in the inner problem of a general branch, the arg min can be a low-dimensional face; hence, in this paper, we combine the discreteness of the lower bound (Lemma F.2) with finishing by the cutting-plane method (Theorem F.3) rather than relying on the convergence of the values at the evaluation points.

F.3. Explicit lower bound on the submodel gap in the integer case. With the same motivation as Kitaoka (2026), who gave an explicit lower bound on the constant $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ (a positive constant determined by the geometry of the set of minimizers of the suboptimality loss, which bounds by $O ( 1 / \gamma ( \ell ^ { \mathrm { s u b } } ) ^ { 2 } )$ the number of iterations a gradient method needs to reach $\ell ^ { \mathrm { { s u b } } } = 0 ;$ ; Kitaoka, 2024), we give an explicit lower bound on $\delta ( \boldsymbol { y } ^ { \prime } )$ . The key is that all submodel minimum values $m _ { W } ( y ^ { \prime } )$ are rational numbers with a common upper bound on their denominators; although there are exponentially many subsets $W ,$ , enumerating them is unnecessary (only the upper bound on the denominators matters).

Proposition F.9 (Explicit lower bound on δ in the integer case). In addition to $\mathrm { A s } -$ sumption 4.2, let $N = 1 , \Theta = \Delta ^ { d - 1 }$ , and $\mathcal { X } ( s ^ { 1 } ) \subseteq \mathbb { Z } ^ { d }$ , and let $R : = \mathrm { m a x } _ { x \in \mathcal { X } ( s ^ { 1 } ) } \| x \| _ { \infty }$ Then for every branch $y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } }$

$$
\delta ( y ^ { \prime } ) \ \geq \ { \frac { 1 } { Q ^ { 2 } } } , \qquad Q : = \big ( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( 2 R , 1 ) \big ) ^ { d + 1 } .
$$

Proof. First, we show that for every nonempty $W \subseteq \mathcal { V } ^ { \mathrm { a l l } }$ , m<sub>W</sub> $( y ^ { \prime } )$ is a rational number with denominator at most Q. Since $N = 1$ , each cut $g _ { v } = v - y ^ { \prime } \ ( v , y ^ { \prime } \in$ $\mathcal { X } ( s ^ { 1 } ) \subseteq \mathbb { Z } ^ { d } )$ is an integer vector with $\| g _ { v } \| _ { \infty } \leq 2 R$ . m<sub>W</sub> $( y ^ { \prime } )$ is the optimal value of the linear program

$$
\begin{array} { r } { \operatorname* { m i n } \bigl \{ \tau \ \mid \ ( \theta , \tau ) \in P _ { W } \ \bigr \} , \quad P _ { W } : = \Bigl \{ ( \theta , \tau ) \in \mathbb { R } ^ { d } \times \mathbb { R } \ \big | \begin{array} { l } { \theta \geq 0 , \sum _ { i = 1 } ^ { d } \theta _ { i } = 1 , } \\ { \langle g _ { v } , \theta \rangle \leq \tau \ ( \forall v \in W ) } \end{array} \Bigr \} . } \end{array}
$$

$P _ { W }$ is nonempty and contains no line (θ is bounded, and the direction $( 0 , - 1 )$ is incompatible with the constraints $\langle g _ { v } , \theta \rangle \leq \tau )$ , and the optimal value is finite by the compactness of $\Theta ,$ so it is attained at some extreme point $( \theta , \tau )$ . An extreme point is the unique solution of a nonsingular system of linear equations consisting of $( d + 1 )$ linearly independent tight constraints, and each row of its coeficient matrix A is an integer vector of type $( g _ { v } , - 1 ) , ( { \bf 1 } , 0 ) , \mathrm { o r } ( - e _ { i } , 0 )$ . The $\ell _ { 2 }$ norms of the rows are bounded by $\sqrt { d ( 2 R ) ^ { 2 } + 1 } \leq ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( 2 R , 1 ) , \sqrt { d }$ , and 1, respectively, so by Hadamard’s inequality,

$$
1 \leq \vert \operatorname* { d e t } A \vert \leq \left( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( 2 R , 1 ) \right) ^ { d + 1 } = Q
$$

(the lower bound holds because A is a nonsingular integer matrix). By Cramer’s rule, τ is a rational number with denominator | det $A | \leq Q$ , and we can write m<sub>W</sub> $\left( y ^ { \prime } \right) = p / q \ \left( p \in \mathbb { Z } , 1 \leq q \leq Q \right)$

If $\delta ( y ^ { \prime } ) = + \infty ,$ , the claim holds. Otherwise, there exists $W$ attaining the max in Equation (F.3), and $\delta ( y ^ { \prime } ) = m _ { \mathrm { s u b } } ( y ^ { \prime } ) - m _ { W } ( y ^ { \prime } ) > 0$ . Since $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = m _ { y ^ { \mathrm { a l l } } } ( y ^ { \prime } )$ is also of the above form, we can write $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = p _ { 1 } / q _ { 1 }$ and m<sub>W</sub> $( y ^ { \prime } ) = p _ { 2 } / q _ { 2 }$ $( q _ { 1 } , q _ { 2 } \leq Q )$ ; since the two are distinct,

$$
\delta ( y ^ { \prime } ) = \frac { \left| p _ { 1 } q _ { 2 } - p _ { 2 } q _ { 1 } \right| } { q _ { 1 } q _ { 2 } } \geq \frac { 1 } { q _ { 1 } q _ { 2 } } \geq \frac { 1 } { Q ^ { 2 } } .
$$

Corollary F.10 (Extension to $N \geq 2 )$ . In addition to Assumption 4.2, let $N \geq 2$ $\Theta = \Delta ^ { d - 1 }$ , and $\mathcal { X } ( s ^ { n } ) \subseteq \mathbb { Z } ^ { d }$ for each $n ,$ and let $R : = \operatorname* { m a x } _ { n } \operatorname* { m a x } _ { x \in { \mathcal { X } } ( s ^ { n } ) } \| x \| _ { \infty }$ . Then for every branch $y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } }$ , we have

$$
\delta ( y ^ { \prime } ) \ \geq \ \frac { 1 } { N Q _ { N } ^ { 2 } } , \qquad Q _ { N } : = \big ( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( 2 N R , 1 ) \big ) ^ { d + 1 } .
$$

As in the lower bound on $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ (Kitaoka, 2026), the dependence on N appears through $Q _ { N }$ and the factor $1 / N$

Proof. For the cut $\begin{array} { r } { g _ { v } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { \prime n } ) } \end{array}$ , we have $N g _ { v } \in \mathbb Z ^ { d }$ and $\| N g _ { v } \| _ { \infty } \leq 2 N R .$ Applying the first step of the proof of Proposition F.9 to the family of integer cuts $\{ N g _ { v } \} _ { v }$ , for every nonempty $W \subseteq \mathcal { V } ^ { \mathrm { a l l } }$ 2

$$
N m _ { W } ( y ^ { \prime } ) = \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } \operatorname* { m a x } _ { v \in W } \langle \theta , N g _ { v } \rangle
$$

is a rational number with denominator at most $Q _ { N }$ . Hence the diference of two distinct values of $N m _ { W } ( y ^ { \prime } )$ is at least $1 / Q _ { N } ^ { 2 }$ , and the diference of two distinct values of $m _ { W } ( y ^ { \prime } )$ is at least $1 / ( N Q _ { N } ^ { 2 } )$ . The rest follows as in the second step of the proof of Proposition F.9. □

Remark F.11 (Fully explicit number of iterations of Phase 1). In the setting of Proposition F.9, choosing $\epsilon _ { \mathrm { s w } } : = 1 / ( 2 Q ^ { 2 } ) < \delta ( y ^ { \prime } )$ and combining with Remark F.4, the number of iterations of Phase 1 (level method) is bounded by

$$
c ( \lambda ) \left( L _ { g } ( y ^ { \prime } ) D _ { \Theta } \cdot 2 Q ^ { 2 } \right) ^ { 2 } \leq c ( \lambda ) \left( 2 R \sqrt { d } \cdot \sqrt { 2 } \cdot 2 Q ^ { 2 } \right) ^ { 2 } ,
$$

a fully explicit function of the problem size $( d , \ R )$ (using $L _ { g } ( y ^ { \prime } ) \leq 2 R \sqrt { d }$ and diam $\left( \Delta ^ { d - 1 } \right) = \sqrt { 2 } )$ . Since $Q ^ { 4 } = \left( ( d + 1 ) \operatorname* { m a x } ( 2 R , 1 ) ^ { 2 } \right) ^ { 2 ( d + 1 ) }$ , this bound is exponential in d. This is consistent with the fact that the lower bound on $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ for general ILPs can be exponentially small in d (and that this is essential) (Kitaoka, 2026).

Corollary F.12 (Explicit number of iterations over the whole run in the integer case). In the setting of Proposition F.9 for $N = 1$ and of Corollary F.10 for $N \geq 2 \ ( \Theta = \Delta ^ { d - 1 } , \mathcal { X } ( s ^ { n } ) \subseteq \mathbb { Z } ^ { d }$ for each n, $\begin{array} { r } { R : = \operatorname* { m a x } _ { n } \operatorname* { m a x } _ { x \in \mathcal { X } ( s ^ { n } ) } \left\| x \right\| _ { \infty } ) } \end{array}$ , use the framework of Proposition F.6 with the rule R being the level method and the switching threshold

$$
\epsilon _ { \mathrm { s w } } : = \frac { 1 } { 2 N Q _ { N } ^ { 2 } } , \qquad Q _ { N } = \big ( ( d + 1 ) ^ { 1 / 2 } \operatorname * { m a x } ( 2 N R , 1 ) \big ) ^ { d + 1 } .
$$

Assume that the iteration bound of Remark F.4 holds for the level method in Phase 1. Then $\epsilon _ { \mathrm { s w } } < \delta _ { \mathrm { i n } }$ , and the total number of inner iterations over the whole run is at most

$$
\begin{array} { r l r } & { } & { | \mathcal { Y } _ { \ge } ^ { \mathrm { a l l } } | c ( \lambda ) \left( 2 R \sqrt { d } \cdot \sqrt { 2 } \cdot 2 N Q _ { N } ^ { 2 } \right) ^ { 2 } + M + | \mathcal { Y } _ { \ge } ^ { \mathrm { a l l } } | } \\ & { } & { = 3 2 c ( \lambda ) | \mathcal { Y } _ { \ge } ^ { \mathrm { a l l } } | R ^ { 2 } d N ^ { 2 } Q _ { N } ^ { 4 } + M + | \mathcal { Y } _ { \ge } ^ { \mathrm { a l l } } | , } \end{array}
$$

which is a fully explicit function of the problem size $( d , R , N ) , | \mathcal { V } _ { > } ^ { \mathrm { a l l } } |$ , and M.

Proof. By Corollary F.10, $\delta ( y ^ { \prime } ) \geq 1 / ( N Q _ { N } ^ { 2 } )$ for every branch, so $\delta _ { \mathrm { i n } } \geq 1 / ( N Q _ { N } ^ { 2 } ) >$ $\epsilon _ { \mathrm { s w } }$ . For the Lipschitz constant, $\begin{array} { r } { \| g _ { v } \| = \left\| \frac { 1 } { N } \sum _ { n } ( v ^ { n } - y ^ { \prime n } ) \right\| \leq \operatorname* { m a x } _ { n } \| v ^ { n } - y ^ { \prime n } \| \leq } \end{array}$ $2 R \sqrt { d }$ gives $L _ { g } \leq 2 R \sqrt { d } .$ , and $D _ { \Theta } = \mathrm { d i a m } ( \Delta ^ { d - 1 } ) = \sqrt { 2 }$ . By Remark F.4, $T _ { 1 } =$ $c ( \lambda ) ( L _ { g } D _ { \Theta } / \epsilon _ { \mathrm { s w } } ) ^ { 2 } \leq c ( \lambda ) \bigl ( 2 R \sqrt { d } \cdot \sqrt { 2 } \cdot 2 N Q _ { N } ^ { 2 } \bigr ) ^ { 2 } = 3 2 c ( \lambda ) R ^ { 2 } d N ^ { 2 } Q _ { N } ^ { 4 }$ , and it sufices to substitute this into Proposition F.6(2). □

Remark F.13 (Which term dominates). Since $Q _ { N } ^ { 4 } = \left( ( d + 1 ) \operatorname* { m a x } ( 2 N R , 1 ) ^ { 2 } \right) ^ { 2 ( d + 1 ) }$ the first term of Corollary F.12 is exponential in d. On the other hand, the second term $M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ depends neither on $\delta _ { \mathrm { i n } }$ nor on any exponent of $R ,$ and is determined only by the number $M = | \mathcal { V } ^ { \mathrm { a l l } } |$ of tuples of solutions that the oracle can return and the number $| \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } |$ of branches. Which term dominates depends on the ratio $\begin{array} { r } { M / | \mathcal { V } _ { > } ^ { \mathrm { a l l } } | = \prod _ { n = 1 } ^ { N } | \mathcal { V } ( s ^ { n } ) | / | \mathcal { V } _ { \geq x ^ { n } } ( s ^ { n } ) | } \end{array}$ . If this ratio is at most $3 2 c ( \lambda ) R ^ { 2 } d N ^ { 2 } Q _ { N } ^ { 4 }$ , the first term dominates, and the improvement on the finishing side by Lemma F.5 $( | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | ( M + 1 ) \to M + | \mathcal { V } _ { \geq } ^ { \mathrm { a l l } } | )$ does not change the overall order; in this case, the only way to lower the overall order is to improve the lower bound on $\delta _ { \mathrm { i n } }$ . On the other hand, for fixed $d , 3 2 c ( \lambda ) R ^ { 2 } d N ^ { 2 } Q _ { N } ^ { 4 }$ is polynomial in $N$ , whereas the ratio can grow exponentially in $N ;$ hence, when $N$ is large compared with $d ,$ the second term can dominate, and then the improvement by Lemma F.5 reduces the bound by a factor of up to about $| \mathcal { V } _ { > } ^ { \mathrm { a l l } } |$ . Given that the lower bound on $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ for general ILPs can be exponentially small in d (and that this is essential) (Kitaoka, 2026), improvements using the structure of the forward problem are given for the test-direction-cut inner solver in §F.4 and $\ S \mathrm { F . 5 ; }$ improving the lower bound on $\delta _ { \mathrm { i n } }$ for Algorithm 5 itself is future work (Remark F.27).

Remark F.14 (Improvement by structure is left for future work). Kitaoka (2026) replaced the dependence of the lower bound on $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ on the range of the features (which correspond to the feasible solutions in this paper) by structural constants, using structures of the forward problem such as total unimodularity, Graver bases, and M-convexity. A similar improvement is expected for $\delta ( \boldsymbol { y } ^ { \prime } )$ , but the minors in Proposition F.9 are those of the cut matrix $( g _ { v } ) _ { v } ,$ , and properties such as the total unimodularity of the constraint matrix of the forward problem are not immediately inherited by the cut matrix. This improvement is future work. Restricting attention to the value gap of the branch minimum value $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ at 0 (a positive lower bound on $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ when $m _ { \mathrm { s u b } } ( y ^ { \prime } ) > 0 )$ , an explicit lower bound via Graver bases (test sets) is obtained in §F.4 and $\ S \mathrm { F . 5 } ,$ and for a variant of the inner solver restricted to test-direction cuts, gap lower bounds independent of $R$ (polynomial in the dimension in the M<sup>♮</sup>-convex case) are obtained in Propositions F.20 and F.22.

F.4. Explicit lower bounds via test sets (single state). The lower bound of §F.3 is of order $R ^ { - 2 ( d + 1 ) }$ in the range R of the features and is exponentially small in the dimension d. In this subsection, using the integer structure of the forward problem (Graver bases), we bound from below explicitly, only in terms of the norm of the test set, the value gap of the branch minimum value $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ at 0 instead of the submodel gap $\delta ( \boldsymbol { y } ^ { \prime } )$ . In the second half, we define a variant of the inner solver whose cut generation is restricted to test directions, and give a lower bound on its gap independent of R (a lower bound polynomial in the dimension in the $\mathrm { M } ^ { \natural } .$ -convex case). In this subsection, let $N = 1$ (the extension to $N \geq 2$ is given in §F.5) and $\Theta = \Delta ^ { d - 1 }$ , and suppose that the feasible region of the forward problem is given in the form

$$
\mathcal { X } ( \mathfrak { s } ^ { 1 } ) = \left\{ x \in \mathbb { Z } ^ { d } \ \left| \ A x \leq b , \ 0 \leq x \leq u \right. \right\} \qquad ( A \in \mathbb { Z } ^ { n _ { A } \times d } , \ b \in \mathbb { Z } ^ { n _ { A } } , \ u \in \mathbb { Z } _ { \geq 0 } ^ { d } )
$$

(a special case of Assumption 4.2(2)). As in Proposition F.9, let $R : =$ $\mathrm { m a x } _ { x \in \mathcal { X } ( s ^ { 1 } ) } \| x \| _ { \infty }$ . Introducing slack variables and setting ${ \cal \tilde { A } } : = [ A | { \cal \cal I } _ { n _ { A } } ] \in$ $\mathbb { Z } ^ { n _ { A } \times ( d + n _ { A } ) }$ and $z ( x ) : = ( x , b - A x ) \in \mathbb { Z } ^ { d + n _ { A } } , x \in \mathcal { X } ( s ^ { 1 } )$ is equivalent to $\tilde { A } z = b$ and $0 \leq z \leq \tilde { u }$ (u˜ is a finite upper bound determined by the box $0 \leq x \leq u )$ . Let $\mathcal { T } : = \mathrm { G r } ( \tilde { A } )$ be the Graver basis of $\tilde { A }$ (the set of all nonzero elements of $\ker \mathbb { Z } \tilde { A }$ that are minimal with respect to the conformal order (for $t , w \in \mathbb { Z } ^ { d + n _ { A } } , t \subseteq w { \mathrm { ~ i f ~ } } t _ { l } w _ { l } \geq 0$ and $| t _ { l } | \le | w _ { l } |$ for every component $l ) ;$ ; Onn, 2010), and let $C : = \operatorname* { m a x } _ { t \in \mathcal { T } } \| t \| _ { \infty } .$ We write the x component of $t \in \tau$ as $t ^ { x } \in \mathbb { Z } ^ { d } \ ( \| t ^ { x } \| _ { \infty } \leq C )$ . We further set $C _ { R } : = \operatorname* { m i n } ( C , R )$ . Since $\mathcal { X } ( s ^ { 1 } ) \subseteq [ 0 , R ] ^ { d }$ , every test direction t such that both $y ^ { \prime }$ and $y ^ { \prime } + t ^ { x }$ belong to $\mathcal { X } ( s ^ { 1 } )$ satisfies $\| t ^ { x } \| _ { \infty } \leq C _ { R }$ . The Graver basis is a test set for linear optimization over $\dot { \mathcal { X } } ( s ^ { 1 } )$ ; that is, for any $\theta \in \mathbb { R } ^ { d }$ and $y \in \mathcal { X } ( s ^ { 1 } )$ , y is a maximizer of $\langle \theta , \cdot \rangle$ if and only if there exists no $t \in \mathcal T$ with $0 \leq z ( y ) + t \leq \tilde { u }$ and $\langle \theta , t ^ { x } \rangle > 0$ (Onn, 2010). For a branch $y ^ { \prime } \in \mathcal { V } _ { \geq } ^ { \mathrm { a l l } }$ (since $N = 1 , y ^ { \prime } \in \mathcal { V } _ { \geq x ^ { 1 } } ( s ^ { 1 } ) \subseteq \mathcal { X } ( s ^ { 1 } ) )$ , we define the test-set model by

$$
h _ { \mathcal { T } } ( \theta ) : = \operatorname* { m a x } \Bigl ( 0 , \ \operatorname* { m a x } \bigl \{ \langle \theta , t ^ { x } \rangle \ \mid \ t \in \mathcal { T } , \ z ( y ^ { \prime } ) + t \ \mathrm { i s ~ f e a s i b l e } \bigr \} \Bigr )\tag{F.4}
$$

(feasible means satisfying the equality constraints of $\tilde { A }$ and the box constraints; if there is no such t, we read the inner max as −∞, so that $h \tau = 0 )$

For the sandwich bound, we use the following property of Graver bases. Here, let $D : = d + n _ { A }$ , and for $t , w \in \mathbb { Z } ^ { D } , t \subseteq$ w means that $t _ { l } w _ { l } \ge 0$ and $| t _ { l } | \le | w _ { l } |$ | for every component l (conformality, i.e., sign compatibility).

Proposition F.15 (Conformal decomposition by the Graver basis; Onn, 2010). For an integer matrix A<sup>˜</sup> and $w \in \ker _ { \mathbb { Z } } \tilde { A } \setminus \{ 0 \}$ , there exist positive integers $\alpha _ { 1 } , \ldots , \alpha _ { r }$ and $t _ { 1 } , \ldots , t _ { r } \in \operatorname { G r } ( \tilde { A } )$ such that $\textstyle w = \sum _ { i = 1 } ^ { r } \alpha _ { i } t _ { i }$ and $t _ { i } \subseteq w$ for each i.

Lemma F.16 (Sandwich by the test-set model). Let

$$
D _ { z } : = \operatorname* { m a x } \bigr ( 1 , \operatorname* { m a x } _ { \substack { x , x ^ { \prime } \in \mathcal { X } ( s ^ { 1 } ) } } \| z ( x ^ { \prime } ) - z ( x ) \| _ { 1 } \bigr ) .
$$

For every $\theta \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { h _ { \mathcal { T } } ( \boldsymbol { \theta } ) \ \leq \ \bar { \ell } ^ { \mathrm { s u b } } ( \boldsymbol { \theta } ; \boldsymbol { y } ^ { \prime } ) \ \leq \ D _ { z } \ : h _ { \mathcal { T } } ( \boldsymbol { \theta } ) . } \end{array}
$$

Proof. Lower side: $\mathrm { i f } \ z ( y ^ { \prime } ) + t$ is feasible, then its x component $y ^ { \prime } + t ^ { x }$ belongs to $\chi ( s ^ { 1 } )$ , so $\begin{array} { r } { \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) = \operatorname* { m a x } _ { v \in \mathcal { X } ( s ^ { 1 } ) } \langle \theta , v - y ^ { \prime } \rangle \geq \langle \theta , t ^ { x } \rangle } \end{array}$ . Moreover, $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) \ge 0$ since $y ^ { \prime } \in \mathcal { X } ( s ^ { 1 } )$ . Together, $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) \ge h _ { T } ( \theta )$

Upper side: let $v \in \mathcal { X } ( s ^ { 1 } )$ be a maximizer of $\langle \theta , \cdot - y ^ { \prime } \rangle$ . Since $z ( v ) - z ( y ^ { \prime } ) \in \ker _ { \mathbb { Z } } \tilde { A }$ if $z ( v ) \neq z ( y ^ { \prime } )$ ), then by Proposition F.15 we can write

$$
z ( v ) - z ( y ^ { \prime } ) = \sum _ { i } \alpha _ { i } t _ { i } , \qquad \alpha _ { i } \in \mathbb { Z } _ { > 0 } , \ t _ { i } \in \mathcal { T } , \ t _ { i } \sqsubseteq z ( v ) - z ( y ^ { \prime } )
$$

(⊑ denotes componentwise conformality). By conformality, $z ( y ^ { \prime } ) + t _ { i }$ lies componentwise between $z ( y ^ { \prime } )$ and $z ( v )$ and satisfies the box constraints, and since $\tilde { A } ( z ( y ^ { \prime } ) + t _ { i } ) = b ,$ it is feasible. Hence $\langle \theta , t _ { i } ^ { x } \rangle \leq h _ { T } ( \theta )$ for each $i ;$ moreover, by conformality, $\begin{array} { r } { \sum _ { i } \alpha _ { i } \| t _ { i } \| _ { 1 } = \| z ( \boldsymbol { v } ) - z ( \boldsymbol { y } ^ { \prime } ) \| _ { 1 } \le D _ { z } , } \end{array}$ , and since $\left\| t _ { i } \right\| _ { 1 } \geq 1$ for each $i ,$ $\begin{array} { r } { \sum _ { i } \alpha _ { i } \le D _ { z } } \end{array}$ . Therefore,

$$
\bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) = \left. \theta , \sum _ { i } \alpha _ { i } t _ { i } ^ { x } \right. = \sum _ { i } \alpha _ { i } \langle \theta , t _ { i } ^ { x } \rangle \leq \left( \sum _ { i } \alpha _ { i } \right) h _ { \mathcal { T } } ( \theta ) \leq D _ { z } h _ { \mathcal { T } } ( \theta ) .
$$

Proposition F.17 (Graver lower bound on the branch value $\mathrm { g a p } )$ . In the above setting, we have

$$
m _ { \mathrm { s u b } } ( y ^ { \prime } ) \in \{ 0 \} \cup \Big [ \frac { 1 } { Q _ { \mathscr { T } } } , \infty \Big ) , \qquad Q _ { \mathscr { T } } : = \big ( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( C _ { R } , 1 ) \big ) ^ { d + 1 } .
$$

In particular, since $C _ { R } \ \leq \ C$ , we have $Q \tau \leq ( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( C , 1 ) ) ^ { d + 1 }$ , so this lower bound is bounded independently of R by the norm $C$ of the Graver basis. Moreover, since $C _ { R } \leq R , Q _ { T }$ is at most Q of Proposition F.9. For example, if A (and hence ${ \tilde { A } } = [ A \mid I _ { n _ { A } } ] )$ is totally unimodular, then $C = 1$ (Onn, 2010) and $Q \tau \leq ( d + 1 ) ^ { ( d + 1 ) / 2 }$

Proof. If $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ , there is nothing to show. Suppose $m _ { \mathrm { s u b } } ( y ^ { \prime } ) ~ > ~ 0$ . By the upper side of Lemma F.16, for every $\theta \in \Delta ^ { d - 1 } , h _ { \mathcal { T } } ( \theta ) \geq \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) / D _ { z } \geq$ $m _ { \mathrm { s u b } } ( y ^ { \prime } ) / D _ { z } > 0$ , so $\mu : = \mathrm { m i n } _ { \theta \in \Delta ^ { d - 1 } } h _ { \mathcal { T } } ( \theta ) > 0 ,$ . µ is the optimal value of the linear program

$$
\operatorname* { m i n } \Bigl \{ \tau \ \mid \ { \begin{array} { l } { \theta \geq 0 , \ \sum _ { i } \theta _ { i } = 1 , \ \tau \geq 0 , } \\ { \langle t ^ { x } , \theta \rangle \leq \tau \ ( t \in { \mathcal { T } } \ \mathrm { s u c h ~ t h a t ~ } z ( y ^ { \prime } ) + t \ \mathrm { i s ~ f e a s i b l e } ) } \end{array} } \Bigr \} , 
$$

and the rows of its coeficient matrix are integer vectors of type $( t ^ { x } , - 1 ) \ ( \ell _ { 2 }$ norm $\leq \sqrt { d C _ { R } ^ { 2 } + 1 } \leq ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( C _ { R } , 1 ) ) , ( \mathbf { 0 } , - 1 ) , ( \mathbf { 1 } , 0 )$ , or $( - e _ { i } , 0 )$ . By the same extreme-point, Cramer, and Hadamard argument as in the proof of Proposition F.9, $\mu$ is a rational number with denominator at most $Q \tau$ . Since $\mu > 0 , \mu \geq 1 / Q _ { T }$ , and by the lower side of Lemma F.16,

$$
m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) ~ \ge ~ \mu ~ \ge ~ \frac { 1 } { Q \tau } .
$$

Note that $D _ { z }$ is used only to derive $\mu > 0$ and does not appear in the value of the lower bound itself. □

The decomposition into test directions in Lemma F.16 and the estimates via extreme points, Cramer’s rule, and Hadamard’s inequality in Propositions F.9 and F.17 are similar to the arguments used for the lower bound on $\gamma ( \ell ^ { \mathrm { { s u b } } } )$ in Kitaoka (2026, Lemma 6.9, Theorem 6.24); in this section, we apply them to $\delta ( \boldsymbol { y } ^ { \prime } )$ $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ , and $\mu ( \boldsymbol { y } ^ { \prime } )$ (defined after Equation (F.5)).

We call a branch y<sup>′</sup> with $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ a zero branch.

Corollary F.18 (Exact decision of zero branches). In the above setting, if $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) ~ < ~ 1 / Q _ { \tau }$ is observed for some $\bar { \theta } \in \Delta ^ { \dot { d } - 1 }$ , then $m _ { \mathrm { s u b } } ( y ^ { \prime } ) ~ = ~ 0$ In particular, $m _ { \mathrm { s u b } } ( y ^ { \prime } ) ~ = ~ 0$ is established as soon as $\mathrm { U B } ~ < ~ 1 / Q \tau$ during the execution of Algorithm 5. Furthermore, for $N \ = \ 1$ , by Equation (5.3), min $\begin{array} { r } { . \varrho \in \Theta ^ { \bar { \ell } ^ { \mathrm { O S } } } ( \theta ) = \operatorname* { m i n } _ { y ^ { \prime \prime } \in \mathcal { V } _ { > } ^ { \mathrm { a l l } } } m _ { \mathrm { s u b } } ( y ^ { \prime \prime } ) } \end{array}$ , so this gives a decision of the realizable case (global minimum value 0) with the explicit tolerance $1 / Q _ { T }$ (for $N \geq 2$ , see Corollary F.31).

Proof. This follows from $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \leq \bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) < 1 / Q \tau$ and Proposition F.17. The same holds for UB, since it is the minimum of the values of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ at the evaluation points. □

Next, we define a variant of the inner solver whose cut generation is restricted to test directions (Definition F.19), and give a lower bound on its gap independent of R (Proposition F.20), a lower bound polynomial in the dimension in the M-convex and M<sup>♮</sup>-convex cases (Proposition F.22), and the connection to the decision of zero branches (Corollary F.26).

Definition F.19 (Test-direction-cut inner solver). Let $\mathcal T _ { y ^ { \prime } } : = \{ t ^ { x } ~ | ~ t \in \mathcal T , ~ z ( y ^ { \prime } ) +$ t is feasible}. A test-direction oracle is an oracle that, given $\theta \in \mathbb { R } ^ { d }$ , returns the value $\operatorname* { m a x } _ { g \in T _ { y ^ { \prime } } } \langle \theta , g \rangle$ and a cut $g \in \mathcal { T } _ { y ^ { \prime } }$ attaining the maximum (a maximizing cut)

(its implementability depends on the structure: in the M-convex case it reduces to feasibility checks of at most $d ( d - 1 )$ directions, and of at most $d ( d + 1 )$ directions in the $\mathrm { M } ^ { \natural . }$ -convex case, whereas for a general Graver basis it can be implemented by computing the Graver basis (Onn, 2010; De Loera et al., 2012) and checking the feasibility of each of its elements). The test-direction-cut inner solver is Algorithm 5 with the objective function replaced by $h \tau$ (Equation (F.4)), cut generation replaced from the forward-problem oracle by the test-direction oracle, and the zero cut $g = \mathbf { 0 }$ (corresponding to $h \tau \geq 0 )$ always added as an initial cut.

$h \tau$ is the maximum of the cuts of the finite family $\mathcal { T } _ { y ^ { \prime } } \cup \{ \mathbf { 0 } \}$ , and the test-direction oracle returns the value and a maximizing cut at each evaluation point, so the proofs of Lemmas F.1 and F.2 and Theorem F.3 go through verbatim with the family of cuts $\{ g _ { v } \ | \ v \in \mathcal { V } ^ { \mathrm { a l l } } \}$ replaced by $\mathcal { T } _ { y ^ { \prime } } \cup \{ \mathbf { 0 } \}$ . We define the corresponding submodel gap by

$$
\delta _ { \mathcal { T } } ( y ^ { \prime } ) : = \mu ( y ^ { \prime } ) - \operatorname* { m a x } \{ m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } ) ~ | ~ \emptyset \neq G \subseteq \mathcal { T } _ { y ^ { \prime } } \cup \{ \mathbf { 0 } \} , ~ m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } ) < \mu ( y ^ { \prime } ) ~ \} .\tag{F.5}
$$

Here $\begin{array} { r } { m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } ) : = \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } \operatorname* { m a x } _ { g \in G } \langle \theta , g \rangle } \end{array}$ and $\begin{array} { r } { \mu ( y ^ { \prime } ) : = \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } h _ { \mathcal { T } } ( \theta ) } \end{array}$ , and as in Equation (F.3), if there is no such $G ,$ we read $\delta \tau ( y ^ { \prime } ) : = + \infty$

Proposition F.20 (Lower bound on the test-direction gap independent of $R )$ . In the setting of Proposition F.17, we have

$$
\delta _ { T } ( y ^ { \prime } ) \ \geq \ \frac { 1 } { Q _ { T } ^ { 2 } } .
$$

In particular, since $Q \tau \leq ( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( C , 1 ) ) ^ { d + 1 }$ , this lower bound is bounded by a quantity independent of R. Furthermore, when the test-direction-cut inner solver is run with $0 \leq \epsilon _ { \mathrm { i n } } < \delta _ { T } ( y ^ { \prime } )$ , analogues of Lemma F.2 and Theorem F.3 hold: at stopping with $\mathrm { U B } - \mathrm { L B } \leq \epsilon _ { \mathrm { i n } } , \mathrm { L B } = \mu ( y ^ { \prime } )$ is identified exactly, and finishing with the selection rule of the cutting-plane method reaches an exact minimizer of $h \tau$ in finitely many iterations.

Proof. Each $m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } )$ is the optimal value of an LP consisting of integer cuts whose components have absolute values at most $C _ { R } ,$ , so by the same extreme-point, Cramer, and Hadamard argument as in the proof of Proposition F.17, it is a rational number with denominator at most $Q \tau$ . The diference of two distinct values is at least $1 / Q _ { T } ^ { 2 }$ which gives the first half. The second half holds because the proofs of Lemmas F.1 and F.2 and Theorem F.3 use only “the family of cuts is finite” and “the oracle returns a maximizing cut at each evaluation point”. □

Next, we consider the case where the feasible region of the forward problem is an M-convex set. A set $\mathcal { M } \subseteq \mathbb { Z } ^ { d }$ is an M-convex set if M is nonempty and, for any $x , y \in { \mathcal { M } }$ and any i with $x _ { i } > y _ { i }$ , there exists $j$ with $x _ { j } < y _ { j }$ such that $x - e _ { i } + e _ { j } \in { \mathcal { M } }$ and $y + e _ { i } - e _ { j } \in { \mathcal { M } }$ (exchange axiom). A set $\mathcal { M } \subseteq \mathbb { Z } ^ { d }$ is an $M ^ { \natural } .$ -convex set if M is nonempty and, for any $x , y \in { \mathcal { M } }$ and any i with $x _ { i } > y _ { i }$ , either $x - e _ { i } \in \mathcal { M }$ and $y + e _ { i } \in { \mathcal { M } }$ , or there exists j with $x _ { j } < y _ { j }$ such that $x - e _ { i } + e _ { j } \in { \mathcal { M } }$ and $y + e _ { i } - e _ { j } \in \mathcal { M }$ (Murota, 2003). Every M-convex set is $\mathrm { M } ^ { \natural } .$ -convex. For linear optimization over M-convex sets, the following optimality criterion holds.

Proposition F.21 (Optimality criterion for linear optimization over M-convex sets; Murota, 2003). For a bounded M-convex set $\mathcal { M } \subset \mathbb { Z } ^ { d } , \theta \in \mathbb { R } ^ { d }$ , and $y \in \mathcal M$ y is a maximizer of $\langle \theta , \cdot \rangle$ over M if and only if $\langle \theta , e _ { j } - e _ { i } \rangle \leq 0$ for all $i \neq j$ with $y + e _ { j } - e _ { i } \in { \mathcal { M } }$

Proposition F.22 (Lower bound polynomial in the dimension for the M-convex and M<sup>♮</sup>-convex cases). Consider the case where $\mathcal { X } ( s ^ { 1 } ) \subset \mathbb { Z } ^ { d }$ is a bounded M-convex set, let $\mathcal { T } _ { \mathrm { M } } : = \{ e _ { i } - e _ { j } | i \neq j \}$ and $\mathcal { T } _ { \mathrm { M } , y ^ { \prime } } : = \{ g \in \mathcal { T } _ { \mathrm { M } } \ | \ y ^ { \prime } + g \in \mathcal { X } ( s ^ { 1 } ) \}$ , and replace $\mathcal { T } _ { y ^ { \prime } }$ in Equation (F.4) and Equation (F.5) by $\mathcal { T } _ { \mathrm { M } , y ^ { \prime } }$ . Then:

$\mathrm { ( a ) } ~ \mathcal { T } _ { \mathrm { M } }$ is a test set for linear optimization over $\mathcal { X } ( s ^ { 1 } )$ . That is, for every $\theta \in \mathbb { R } ^ { d }$ and $y \in \mathcal { X } ( s ^ { 1 } )$ , y is a maximizer of $\langle \theta , \cdot \rangle$ if and only if there is no $g \in \mathcal { T } _ { \mathrm { M } }$ with $y + g \in \mathcal { X } ( s ^ { 1 } )$ and $\langle \theta , g \rangle > 0$

(b) For every nonempty $G \subseteq { \mathcal { T } } _ { \mathrm { M } , y ^ { \prime } } \cup \{ \mathbf { 0 } \} , \ m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } )$ is a rational number with denominator at most $d ^ { 2 } \colon$ ; hence we have

$$
\delta _ { T _ { \mathrm { M } } } ( y ^ { \prime } ) \geq \frac { 1 } { d ^ { 4 } } ,
$$

and we have

$$
\mu ( y ^ { \prime } ) > 0 \implies \mu ( y ^ { \prime } ) \ \geq \ { \frac { 1 } { d ^ { 2 } } } .
$$

For $\mathrm { M } ^ { \natural } .$ -convex sets, the same lower bounds hold with $\mathcal { T } = \{ e _ { i } - e _ { j } \} \cup \{ \pm e _ { i } \}$ (that $\tau$ is a test set follows from the optimality criterion for linear optimization over M<sup>♮</sup>-convex sets; Murota, 2003).

Proof. (a) is Proposition F.21 applied to $\mathcal { M } = \mathcal { X } ( s ^ { 1 } )$ We show (b). As in the proof of Proposition $\mathrm { F . 9 , }$ an extreme point of the LP of $m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } )$ is determined by a nonsingular integer matrix A consisting of $( d + 1 )$ linearly independent tight constraints, and each row of A is of type $( e _ { i } - e _ { j } , - 1 ) , ( \mathbf { 0 } , - 1 ) , ( \mathbf { 1 } , 0 ) , \mathrm { o r } ( - e _ { k } , 0 )$ Since the equality constraint $\textstyle \sum _ { i } \theta _ { i } = 1$ is tight at every feasible point, we can choose these $d + 1$ tight constraints so that they include the row $( \mathbf { 1 } , 0 )$ (extend $\{ ( { \bf 1 } , 0 ) \}$ to a basis of the row space of the tight constraints). By cofactor expansion along the τ column, | det $A |$ is bounded by the sum, over “the rows having a nonzero entry in the τ column (at most d rows, since the τ entry of $( \mathbf { 1 } , 0 )$ is $0 ) ^ { \dag }$ , of the absolute values of the determinants of the $d \times d$ submatrices $M ^ { \prime }$ formed by the θ parts of the remaining d rows. Each $M ^ { \prime }$ contains exactly one all-ones row, and its other rows are of type $e _ { i } - e _ { j }$ , of $\mathrm { t y p e } - e _ { k }$ , or zero rows. If it contains a zero row, det $M ^ { \prime } = 0$ Otherwise, cofactor expansion along the all-ones row bounds | det $M ^ { \prime } |$ by the sum of d determinants of $( d - 1 ) \times ( d - 1 )$ matrices in which each row has at most one +1 and at most one −1; their transposes are submatrices of the incidence matrix of a directed graph and are totally unimodular, so | det $M ^ { \prime } | \leq d .$ Hence $1 \leq$ | det $A | \leq d ^ { 2 }$ and $m _ { G } ^ { \mathcal { T } } ( y ^ { \prime } )$ is a rational number with denominator at most $d ^ { 2 }$ . The diference of two distinct values is at least $1 / d ^ { 4 }$ , and positive values are at least $1 / d ^ { 2 }$ . Adding rows of type ±e does not change the structure “each row has at most one +1 and at most $\mathrm { o n e \ - } 1 ^ { \mathfrak { s } }$ , so the same bound holds in the M<sup>♮</sup>-convex case. □

Remark F.23 (Branch minima in the M-convex case). Let $\Theta = \Delta ^ { d - 1 }$ and let $\mathcal { X } ( s ^ { 1 } )$ be an M-convex set. Then $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \mu ( y ^ { \prime } ) = 0$ for every branch $y ^ { \prime } .$ Indeed, by the exchange axiom, an M-convex set lies on a hyperplane $\{ x \mid \sum _ { i } x _ { i } = \mathrm { c o n s t } \}$ , so for the uniform weight $\bar { \theta } : = ( 1 / d , \dotsc , 1 / d ) \in \Delta ^ { d - 1 }$ , we have $\langle \bar { \theta } , v - y ^ { \prime } \rangle = 0$ for every $v \in \mathcal { X } ( s ^ { 1 } )$ and $\langle \bar { \theta } , e _ { i } - e _ { j } \rangle = 0$ for every $i \neq j$ . Hence $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) = h _ { T } ( \bar { \theta } ) = 0$ , and together with nonnegativity, $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = \mu ( y ^ { \prime } ) = 0$ . Therefore, in the M-convex case, no branch satisfies the hypothesis $\mu ( y ^ { \prime } ) > 0$ of the latter half of Proposition F.22(b), and Corollary F.26 decides that every branch is a zero branch. The latter half of Proposition F.22(b) and Corollary F.26 are used in the case of $\mathrm { M } ^ { \natural } .$ -convex sets that are not M-convex to identify branches with a positive branch minimum.

Proposition F.24 (Exact minimization of the OS loss in the $\mathrm { M } ^ { \natural } .$ -convex case). Let $\Theta = \Delta ^ { d - 1 }$ , and for each $n ,$ let $\mathcal { X } ( s ^ { n } ) \subset \mathbb { Z } ^ { d }$ be a bounded $\mathrm { M } ^ { \natural } .$ -convex set (including Mconvex sets) and $x ^ { n } \in { \mathcal { X } } ( s ^ { n } )$ . Then the uniform weight $\bar { \theta } : = ( 1 / d , \dotsc , 1 / d ) \in \bar { \Delta ^ { d - 1 } }$ satisfies $\ell ^ { \mathrm { O S } } ( \bar { \theta } ; x ^ { n } , s ^ { n } ) = 0$ for every n. Hence $\bar { \theta }$ is a global minimizer of $\bar { \ell } ^ { \mathrm { O S } }$ over $\Theta .$

Proof. Fix n and let $\mathcal { X } : = \mathcal { X } ( s ^ { n } )$ . The set $\mathcal T = \{ e _ { i } - e _ { j } \ | \ i \neq j \} \cup \{ \pm e _ { i } \}$ is a test set for linear optimization over M<sup>♮</sup>-convex sets (Proposition F.22; Murota, 2003). Since $\langle \bar { \theta } , e _ { i } - e _ { j } \rangle = 0$ and $\langle { \bar { \theta } } , - e _ { i } \rangle < 0 , { \mathrm { i f ~ } } x \in \mathcal { X }$ is not a maximizer of $\langle { \bar { \theta } } , \cdot \rangle$ then $x + e _ { i } \in { \mathcal { X } }$ for some i. Let $z ^ { 0 } : = x ^ { n }$ , and as long as $z ^ { k }$ is not a maximizer, update $z ^ { k + 1 } : = z ^ { k } + e _ { i } \in \mathcal { X }$ . Since $\langle { \bar { \theta } } , z ^ { k } \rangle$ strictly increases and X is finite, this update reaches a maximizer $z \in$ arg max $_ { \cdot x \in \mathcal { X } } \langle \bar { \theta } , x \rangle$ after finitely many steps. Each update only increases a component, so $z \geq x ^ { n } ,$ , that is, $z \in { \mathcal { X } } _ { \geq x ^ { n } } ( s ^ { n } )$ . Hence $\begin{array} { r } { \operatorname* { m a x } _ { x \in \mathcal { K } _ { > x ^ { n } } ( s ^ { n } ) } \langle \bar { \theta } , x \rangle \ = \ \operatorname* { m a x } _ { x \in \mathcal { X } } \langle \bar { \theta } , x \rangle } \end{array}$ , and $\ell ^ { \mathrm { O S } } ( \bar { \theta } ; x ^ { n } , s ^ { n } ) = 0$ by Equation (4.2). Since $\bar { \ell } ^ { \mathrm { O \bar { S } } } \geq 0 , \bar { \theta }$ is a global minimizer. □

Proposition F.24 gives a global minimizer of the OS loss. Attaining Equation (3.2) additionally requires that the optimal solution be unique or that the oracle return an optimal solution dominating the action (§4).

Proposition F.25 (Number of iterations of one call of the test-direction-cut inner solver). In the setting of Proposition F.17 or Proposition F.22, run the test-directioncut inner solver (Definition F.19) for a branch $y ^ { \prime }$ with tolerance 0 and $J = \infty$ , and assume that the LPs are solved exactly.

(i) If the rule for selecting evaluation points is the cutting-plane method, the solver stops within at most $| \mathcal { T } _ { y ^ { \prime } } | + 2$ iterations, and $\mathrm { L B } = \mathrm { U B } = \mu ( y ^ { \prime } )$ at stopping; the returned point is an exact minimizer over $\Delta ^ { d - 1 }$ of the test-set model $h \tau$ (Equation $\left( \mathrm { F . 4 } \right)$ , with minimum value $\mu ( \boldsymbol { y } ^ { \prime } )$ . In particular, in the $\mathrm { M } ^ { \natural } .$ -convex case, $| \mathcal { T } _ { y ^ { \prime } } | \leq d ( d + 1 )$ , so the number is at most $d ( d + 1 ) + 2$

(ii) In the setting of Proposition F.17, let the rule be cutting-plane finishing (Algorithm 4) with the level method as the rule $\mathsf { R } , \epsilon _ { \mathrm { s w } } : = 1 / ( 2 Q _ { T } ^ { 2 } )$ , and $\epsilon _ { \mathrm { t e r m } } = 0$ If the iteration bound of Remark F.4 holds for the level method in Phase 1, then the number of iterations of Phase 1 is at most $c ( \lambda ) \big ( C _ { R } \sqrt { d } \cdot \sqrt { 2 } \cdot 2 Q _ { T } ^ { 2 } \big ) ^ { 2 }$ , that of Phase 2 is at most $| \mathcal { T } _ { y ^ { \prime } } | + 1$ , and the returned point is an exact minimizer of $h \tau$ over $\Delta ^ { d - 1 }$ with minimum value $\mu ( \boldsymbol { y } ^ { \prime } )$

Proof. Let $\mathcal { G } _ { j }$ be the cut set at the end of iteration $j ;$ the zero cut 0 always belongs to $\mathcal { G } _ { j }$ . Let $g _ { j }$ be the maximizing cut returned by the test-direction oracle at the evaluation point $\theta ^ { j } \colon$ ; then $h _ { \mathcal { T } } ( \theta ^ { j } ) = \operatorname* { m a x } \{ \langle \theta ^ { j } , g _ { j } \rangle , 0 \}$

We first show the following: if the evaluation point $\theta ^ { j }$ of iteration $j \geq 1$ is a minimizer over $\Delta ^ { d - 1 }$ of the model m $\operatorname { a x } _ { g \in { \mathcal { G } } _ { j - 1 } } \langle \theta , g \rangle$ and $g _ { j } \in \mathcal { G } _ { j - 1 }$ , then the solver stops at iteration $j .$ . Indeed, then $\mathcal { G } _ { j } = \mathcal { G } _ { j - 1 }$ , so LB = min<sub>θ∈∆</sub>d−1 $\scriptstyle \operatorname* { m a x } _ { g \in { \mathcal { G } } _ { j - 1 } } \langle \theta , g \rangle =$ ma $\tau _ { g \in \mathcal { G } _ { i - 1 } } \langle \theta ^ { j } , g \rangle \geq h _ { T } ( \theta ^ { j } ) \geq \mathrm { U B }$ , and the stopping test $\mathrm { U B } - \mathrm { L B } \leq 0$ holds.

(i) With the cutting-plane method, the evaluation point of each iteration $j \geq 1$ is a minimizer of the model, so by the claim above, if the solver does not stop at iteration $j \geq 1$ , then $g _ { j } \in \mathcal { T } _ { y ^ { \prime } } \backslash \mathcal { G } _ { j - 1 }$ , and a new element of $\mathcal { T } _ { y ^ { \prime } }$ is added to ${ \mathcal { G } } .$ Since $\mathcal { T } _ { y ^ { \prime } }$ is finite, there are at most $| \mathcal { T } _ { y ^ { \prime } } |$ non-stopping iterations $j \geq 1$ . Adding iteration $j = 0$ and the stopping iteration, the number of iterations is at most $| \mathcal { T } _ { y ^ { \prime } } | + 2$ . Since the model is at most $h \tau$ $\mathrm { L B } \le \mu ( y ^ { \prime } )$ , and since UB is the minimum of the values of $h \tau$ at the evaluation points, $\mu ( y ^ { \prime } ) \leq \mathrm { U B }$ . At stopping, $\mathrm { U B } \le \mathrm { L B }$ , so $\mathrm { L B } = \mathrm { U B } = \mu ( y ^ { \prime } )$

and the returned point $\hat { \theta }$ satisfies $h _ { \mathcal { T } } ( \hat { \theta } ) = \mathrm { U B } = \mu ( y ^ { \prime } )$ . In the $\mathrm { M } ^ { \natural } .$ -convex case, $\mathcal { T } _ { y ^ { \prime } } \subseteq \{ e _ { i } - e _ { j } \mid i \neq j \} \cup \{ \pm e _ { i } \}$ , whose cardinality is $d ( d - 1 ) + 2 d = d ( d + 1 )$

(ii) Each element g of $\mathcal { T } _ { y ^ { \prime } } \cup \{ \mathbf { 0 } \}$ satisfies $\| g \| _ { 2 } \leq C _ { R } { \sqrt { d } } .$ , so h<sub>T</sub> is $C _ { R } \sqrt { d } \mathrm { - L i p s c h i t z }$ Since diam $( \Delta ^ { d - 1 } ) = \sqrt { 2 }$ , the assumed bound of Remark F.4 implies that the number of iterations of Phase 1 is at most $c ( \lambda ) \big ( C _ { R } \sqrt { d } \cdot \sqrt { 2 } / \epsilon _ { \mathrm { s w } } \big ) ^ { 2 } = c ( \lambda ) \big ( C _ { R } \sqrt { d } \cdot \sqrt { 2 } \cdot 2 Q _ { T } ^ { 2 } \big ) ^ { 2 }$ By Proposition F.20, $\epsilon _ { \mathrm { s w } } < 1 / Q _ { T } ^ { 2 } \leq \delta \dot { { \tau } } ( y ^ { \prime } )$ ; hence, applying the proof of Lemma F.2 with $\epsilon _ { \mathrm { i n } } , \delta ( y ^ { \prime } )$ , and $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ replaced by $\epsilon _ { \mathrm { s w } } , \delta \tau ( y ^ { \prime } )$ , and $\mu ( \boldsymbol { y } ^ { \prime } )$ , we have $\mathrm { L B } = \mu ( y ^ { \prime } )$ at the end of Phase 1. The evaluation point of each iteration of Phase 2 is a minimizer of the model, so by the claim above, each non-stopping iteration of Phase 2 adds a new element of $\mathcal { T } _ { y ^ { \prime } }$ to ${ \mathcal { G } } .$ . Hence there are at most $| \mathcal { T } _ { y ^ { \prime } } |$ non-stopping iterations in Phase 2, and adding the stopping iteration, at most $| \mathcal { T } _ { y ^ { \prime } } | + 1$ . At stopping, $\mathrm { U B } = \mu ( y ^ { \prime } )$ as in (i), so the returned point is an exact minimizer of $h \tau$ □

Corollary F.26 (Connection to the decision of zero branches). In the setting of Proposition F.17 or Proposition F.22, we have

$$
\mu ( y ^ { \prime } ) = 0 \quad \Longleftrightarrow \quad m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0 .
$$

Therefore, the test-direction-cut inner solver decides exactly whether a branch is a zero branch through the exact computation of $\mu ( \boldsymbol { y } ^ { \prime } )$ . In particular, in the $\mathrm { M } ^ { \natural _ { - } }$ convex case, the size of the family of cuts is at most $d ( d + 1 ) + 1$ , so the decision is completed with at most $d ( d + 1 ) + 2$ calls to the test-direction oracle using only the selection rule of the cutting-plane method (Proposition $\mathrm { F . 2 5 ( i ) } )$ . Moreover, when $\mu ( y ^ { \prime } ) > 0$ $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \geq \mu ( y ^ { \prime } )$ , and in the Graver-basis case, $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \leq D _ { z } \mu ( y ^ { \prime } )$ (Lemma F.16) also holds.

Proof. Suppose $\mu ( y ^ { \prime } ) = 0$ Then $h _ { \mathcal { T } } ( \bar { \theta } ) = 0$ for some $\bar { \theta } \in \Delta ^ { d - 1 }$ , that is, no feasible test direction from $y ^ { \prime }$ has $\langle { \bar { \theta } } , \cdot \rangle > 0$ . In the Graver-basis case, by the upper side of Lemma F.16, $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) \leq D _ { z } h _ { T } ( \bar { \theta } ) = 0 ;$ in the M-convex case, by Proposition $\mathrm { F . 2 2 ( a ) }$ , and in the M<sup>♮</sup>-convex case, by the test-set property for $\mathrm { M } ^ { \natural _ { - } }$ convex sets stated at the end of that proposition, $y ^ { \prime }$ is a maximizer with respect to ${ \bar { \theta } } ,$ so $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) = 0$ . Hence $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ (together with nonnegativity). Conversely, suppose $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ . Then $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { * } ; y ^ { \prime } ) = 0$ for some $\theta ^ { * }$ , and for every feasible test direction $t , y ^ { \prime } + t ^ { x } \in \mathcal { X } ( s ^ { 1 } )$ , so $\begin{array} { r } { \langle \theta ^ { * } , t ^ { x } \rangle \leq \bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { * } ; y ^ { \prime } ) = 0 , } \end{array}$ ; hence $h \tau ( \theta ^ { * } ) = 0$ and $\mu ( y ^ { \prime } ) = 0$ . The inequality $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \geq \mu ( y ^ { \prime } )$ follows from $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } ) \geq h \tau$ (the lower side of Lemma F.16; it uses only the feasibility of single steps, so it also holds in the M-convex case). The bound on the size of the family of cuts and the finite stopping follow from Proposition F.25(i). □

Remark F.27 (Scope). First, what Propositions F.20 and F.22 improve is $\delta _ { T } ( y ^ { \prime } )$ of the test-direction-cut inner solver, not $\delta ( \boldsymbol { y } ^ { \prime } )$ of Algorithm 5 itself, which uses vertex cuts. The denominators of $\delta ( \boldsymbol { y } ^ { \prime } )$ are determined by the cuts $\boldsymbol { v } - \boldsymbol { y } ^ { \prime }$ actually generated (with components of magnitude up to $2 R )$ , and the tie values of two cuts with slopes of order R can approach each other to within order $1 / R ^ { 2 }$ ; we therefore conjecture that $\delta ( \boldsymbol { y } ^ { \prime } )$ itself admits no lower bound independent of $R .$ . Its proof is future work. Second, what the test-direction-cut inner solver minimizes is $h \tau$ , not $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ , so it cannot be used directly for the premise of Proposition E.4 (exact solution of the convex subproblems). Its uses are the exact decision of zero branches (Corollary F.26; for $N \geq 2$ , Corollaries F.31 and F.32) and the $D _ { z }$ -approximation of $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ in the Graver-basis case.

Remark F.28 (What can be said in the M-convex and Graver-basis cases). We discuss whether Proposition F.6 can be specialized to the structured cases, in contrast with Corollary F.12. The conclusion is that the lower bounds of this subsection give a bound on the number of iterations of one call (per branch) of the test-direction-cut inner solver (Proposition F.25).

The reason why the lower bounds of this subsection are for the test-direction-cut inner solver is as described in Remark F.27. Proposition F.6 requires a lower bound on $\delta _ { \mathrm { i n } } = \mathrm { m i n } _ { y ^ { \prime } } \delta ( y ^ { \prime } )$ , whereas Propositions F.20 and F.22 give lower bounds on $\delta \tau ( y ^ { \prime } )$ of the test-direction-cut inner solver. The former is determined by the vertex cuts $\boldsymbol { v } - \boldsymbol { y } ^ { \prime }$ (with components of magnitude up to 2R), and the latter by the test directions $t ^ { x }$ (with components of magnitude up to $C _ { R } .$ , and 1 in the $\mathrm { M } ^ { \natural } .$ -convex case). If the conjecture of Remark F.27 is true, $\delta ( \boldsymbol { y } ^ { \prime } )$ admits no lower bound independent of $R ,$ and the dependence on R in Corollary F.12 is essential.

The number of iterations of one call (per branch) of the test-direction-cut inner solver is given in Proposition F.25. In the Graver-basis case, the number of iterations of Phase 1 is bounded by $c ( \lambda ) \big ( C _ { R } \sqrt { d } \cdot \sqrt { 2 } \cdot 2 Q _ { \mathcal { T } } ^ { 2 } \big ) ^ { 2 }$ (part (ii) of that proposition). Since $Q \tau \leq \left( ( d + 1 ) ^ { 1 / 2 } \operatorname* { m a x } ( C , 1 ) \right) ^ { d + 1 }$ , this number has an upper bound independent of R. It is exponential in d. The M<sup>♮</sup>-convex case is even stronger: the size of the family of cuts is at most $d ( d + 1 ) + 1$ , so the solver stops with at most $d ( d + 1 ) + 2$ calls to the test-direction oracle using only the selection rule of the cutting-plane method, without going through the level method (Proposition F.25(i)). That is, $O ( d ^ { 2 } )$ calls per branch, polynomial in the dimension and independent of both R and δ.

Therefore, the question of whether the number over the whole run can be made polynomial in the dimension reduces to whether the exact computation of $\mu ( \boldsymbol { y } ^ { \prime } )$ in Corollary F.26 can be lifted to the exact computation of $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ (the premise of Proposition E.4). In the M-convex case, the uniform weight <sup>¯</sup>θ gives a minimizer of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ (with minimum value 0) for every branch (Remark F.23), so $m _ { \mathrm { s u b } } ( y ^ { \prime } )$ is computed directly. In the $\mathrm { M } ^ { \natural } .$ -convex case as well, the global minimization of the OS loss itself is attained by the uniform weight (Proposition F.24). What remains is, as a subproblem of DC optimization, recovering the minimization of $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ at each branch from the minimization of $h \tau$ in the $\mathrm { M } ^ { \natural }$ -convex case, which is, as in Remark F.14, future work.

F.5. Explicit lower bounds via test sets (multiple states). In this subsection, we extend the results of §F.4 to $N \geq 1$ . Let $\Theta = \Delta ^ { d - 1 }$ , and suppose that the constraint matrix $A \in \mathbb { Z } ^ { n _ { A } \times d }$ is common to all states, while the right-hand side and the upper bounds of the box may depend on the state:

$$
\begin{array} { r } { \mathcal { X } ( s ^ { n } ) = \left\{ x \in \mathbb { Z } ^ { d } \ \middle \vert \ A x \leq b ^ { n } , 0 \leq x \leq u ^ { n } \right\} \qquad ( b ^ { n } \in \mathbb { Z } ^ { n _ { A } } , u ^ { n } \in \mathbb { Z } _ { \geq 0 } ^ { d } , n = 1 , \ldots , N ) . } \end{array}
$$

Setting $z _ { n } ( x ) : = ( x , b ^ { n } - A x ) \in \mathbb { Z } ^ { d + n _ { A } } , x \in \mathcal { X } ( s ^ { n } )$ is equivalent to $\tilde { A } z _ { n } ( x ) = b ^ { n }$ and $0 \leq z _ { n } ( x ) \leq \tilde { u } ^ { n } \ ( \tilde { u } ^ { n }$ is the finite upper bound determined by the box $0 \leq x \leq u ^ { n } )$ The Graver basis $\mathcal { T } = \mathrm { G r } ( \tilde { \cal { A } } )$ and $C = \operatorname* { m a x } _ { t \in \mathcal { T } } \| t \| _ { \infty }$ are determined by A<sup>˜</sup> alone and are therefore common to all states. Setting $R : = \operatorname* { m a x } _ { n } \operatorname* { m a x } _ { x \in { \mathcal { X } } ( s ^ { n } ) } \| x \| _ { \infty }$ and $C _ { R } : = \operatorname* { m i n } ( C , R )$ , every test direction t feasible at a state satisfies $\| t ^ { x } \| _ { \infty } \leq C _ { R }$ . For a branch $y ^ { \prime } = ( y ^ { \prime n } ) _ { n = 1 } ^ { N } \ ( y ^ { \prime n } \in \mathcal { X } ( s ^ { n } ) )$ ), let

$$
\mathcal { T } _ { y ^ { \prime } , n } : = \{ t ^ { x } ~ | ~ t \in \mathcal { T } , ~ 0 \leq z _ { n } ( y ^ { \prime n } ) + t \leq \tilde { u } ^ { n } \} , \qquad h _ { \mathcal { T } , n } ( \theta ) : = \operatorname* { m a x } \Bigl ( 0 , ~ \operatorname* { m a x } _ { g \in \mathcal { T } _ { y ^ { \prime } , n } } ~ \langle \theta , g \rangle \Bigr ) ,
$$

$$
h _ { \mathcal { T } } ^ { N } ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } h _ { \mathcal { T } , n } ( \theta ) , \qquad \mu _ { N } ( y ^ { \prime } ) : = \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } h _ { \mathcal { T } } ^ { N } ( \theta )
$$

(since $t \in \ker _ { \mathbb { Z } } \tilde { A } .$ , whether $z _ { n } ( y ^ { \prime n } ) + t$ is feasible at state n is determined by the box constraints). We also set $\begin{array} { r } { D _ { z , n } : = \operatorname* { m a x } \bigr ( 1 , \ \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } ( s ^ { n } ) } \| z _ { n } ( x ^ { \prime } ) - z _ { n } ( x ) \| _ { 1 } \bigr ) } \end{array}$ and $D _ { z } ^ { N } : = \operatorname* { m a x } _ { n } D _ { z , n }$

Lemma F.29 (Sandwich by the test-set model with multiple states). In the above setting, for any $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$

$$
h _ { \mathcal { T } } ^ { N } ( \theta ) \ \leq \ \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) \ \leq \ D _ { z } ^ { N } h _ { \mathcal { T } } ^ { N } ( \theta ) .
$$

Proof. We have $\begin{array} { r } { \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime n } , s ^ { n } ) } \end{array}$ . The proof of Lemma F.16 uses only the feasible region $\{ x \in \mathbb { Z } ^ { d } \mid A x \leq b , 0 \leq x \leq u \}$ of a single state and $y ^ { \prime } \in \mathcal { X } ( s ^ { 1 } ) ;$ applying it to each n with $( b , u )$ replaced by $( b ^ { n } , u ^ { n } )$ gives $h \tau _ { , n } ( \theta ) \leq$ $\ell ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime n } , s ^ { n } ) \leq D _ { z , n } h _ { T , n } ( \theta )$ . Averaging over n with $D _ { z , n } \leq \dot { D } _ { z } ^ { N }$ and $h \tau _ { , n } \geq 0$ yields the claim. □

Proposition F.30 (Graver lower bound on the branch value gap with multiple states). In the above setting, let

$$
Q _ { { \cal T } , N } : = \Bigl ( 1 + \frac { d } { N } \Bigr ) ^ { N } \bigl ( d ^ { 1 / 2 } \operatorname* { m a x } ( C _ { R } , 1 ) \bigr ) ^ { d } \ \leq \ \bigl ( \mathrm { e } d ^ { 1 / 2 } \operatorname* { m a x } ( C , 1 ) \bigr ) ^ { d } .
$$

Then

$$
N \mu _ { N } ( y ^ { \prime } ) \in \{ 0 \} \cup \Big [ \frac { 1 } { Q _ { \tau , N } } , \infty \Big ) , \qquad m _ { \mathrm { s u b } } ( y ^ { \prime } ) \in \{ 0 \} \cup \Big [ \frac { 1 } { N Q _ { \tau , N } } , \infty \Big ) .
$$

Moreover, $\mu _ { N } ( y ^ { \prime } ) = 0$ if and only if $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ . In particular, setting $\bar { Q } _ { \mathcal { T } } : =$ $( \mathrm { e } d ^ { 1 / 2 } \operatorname* { m a x } ( C , 1 ) ) ^ { d } , \mathrm { i f } \ m _ { \mathrm { s u b } } ( y ^ { \prime } ) > 0$ , then $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \geq 1 / ( N \bar { Q } _ { T } ) $ ; this lower bound depends neither on the right-hand sides $b ^ { n }$ nor on the upper bounds $u ^ { n }$ of the boxes, and depends on N only through the factor $1 / N$

Proof. $N \mu _ { N } ( y ^ { \prime } )$ is the optimal value of the linear program in the variables $( \theta , \tau ) \in$ $\mathbb { R } ^ { d } \times \mathbb { R } ^ { N }$

$$
\begin{array} { r } { \operatorname* { m i n } \Bigl \{ \sum _ { n = 1 } ^ { N } \tau _ { n } ~ \Big | ~ \theta \geq 0 , ~ \sum _ { i } \theta _ { i } = 1 , ~ \tau _ { n } \geq 0 , ~ \tau _ { n } \geq \langle g , \theta \rangle ~ ( g \in \mathcal { T } _ { y ^ { \prime } , n } , ~ n = 1 , \dots , N ) \Bigr \} . } \end{array}
$$

The feasible region is nonempty and contains no line (θ is bounded and $\tau \geq 0 )$ , and the objective is nonnegative, so the optimal value is attained at an extreme point. An extreme point is determined by a nonsingular integer matrix $A ^ { \prime }$ consisting of $d + N$ linearly independent tight constraints and an integer right-hand side, so by Cramer’s rule each component of the extreme point, and hence the objective value $\sum _ { n } \tau _ { n }$ , is a rational number with denominator | det $A ^ { \prime } |$

We bound | det $A ^ { \prime } |$ . Each row of $A ^ { \prime }$ is of type $( g , - e _ { n } ) ~ ( g \in \mathcal { T } _ { y ^ { \prime } , n } ) , ~ ( \mathbf { 0 } , - e _ { n } )$ $( \mathbf { 1 } , \mathbf { 0 } )$ , or $( - e _ { i } , \mathbf { 0 } )$ , and in the N columns corresponding to τ, each row has at most one nonzero entry (−1). Let $k _ { n }$ be the number of rows with a nonzero entry in the $\tau _ { n }$ column; since $A ^ { \prime }$ is nonsingular, $k _ { n } \ \geq \ 1$ , and $\textstyle \sum _ { n } k _ { n } \leq d + N$ In the Laplace expansion along the N columns of $\tau ,$ the choices of $N$ rows for which the $\tau$ part is nonsingular are exactly those choosing one row from each $\tau _ { n }$ column, and the absolute value of the determinant of that part is 1. There are $\prod _ { n } k _ { n }$ such choices, and by the inequality of arithmetic and geometric means, $\begin{array} { r } { \prod _ { n } k _ { n } \le \left( ( d + N ) / N \right) ^ { N } = ( 1 + d / N ) ^ { N } } \end{array}$ . The θ part of the remaining d rows has rows of norm at most $d ^ { 1 / 2 } \operatorname* { m a x } ( C _ { R } , 1 )$ , since $\| g \| _ { 2 } \le d ^ { 1 / 2 } C _ { R } , \| \mathbf { 1 } \| _ { 2 } = d ^ { 1 / 2 }$ , and $\| e _ { i } \| _ { 2 } = 1 ;$

hence by Hadamard’s inequality, the absolute value of its determinant is at most $( d ^ { 1 / 2 } \operatorname* { m a x } ( C _ { R } , 1 ) ) ^ { d }$ . Therefore $1 \leq | \operatorname* { d e t } A ^ { \prime } | \leq Q _ { T , N }$ , and $N \mu _ { N } ( y ^ { \prime } )$ is a nonnegative rational number with denominator at most $Q _ { T , N }$ . Hence, if $N \mu _ { N } ( y ^ { \prime } ) > 0$ , then $N \mu _ { N } ( y ^ { \prime } ) \geq 1 / Q _ { \mathcal { T } , N } . \mathrm { ~ A l s o , ~ } ( 1 + d / N ) ^ { N } \leq \mathrm { e } ^ { d } .$

By Lemma $\mathrm { F . 2 9 } , \mu _ { N } ( y ^ { \prime } ) \leq m _ { \mathrm { s u b } } ( y ^ { \prime } ) \leq D _ { z } ^ { N } \mu _ { N } ( y ^ { \prime } )$ . Hence $\mu _ { N } ( y ^ { \prime } ) = 0$ if and only if $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ , and if $m _ { \mathrm { s u b } } ( y ^ { \prime } ) > 0 .$ , then $m _ { \mathrm { s u b } } ( y ^ { \prime } ) \geq \mu _ { N } ( y ^ { \prime } ) \geq 1 / ( N Q _ { \mathcal { T } , N } )$ □

Corollary F.31 (Decision of zero branches and realizability with multiple states). In the above setting, if $\bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } ) < 1 / ( N Q _ { T , N } )$ for some $\check { \bar { \theta } } \in \Delta ^ { d - 1 }$ and a branch $y ^ { \prime } \in \mathcal { X } _ { > } ^ { \mathrm { a l l } }$ , then $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ . In particular, if $\bar { \ell } ^ { \mathrm { O S } } ( \bar { \theta } ) < 1 / ( N Q _ { T , N } )$ for some $\bar { \theta } \in \Delta ^ { d - 1 }$ , then min<sub>θ∈Θ</sub> $\bar { \ell } ^ { \mathrm { O S } } ( \theta ) = 0$

Proof. The first half follows from Proposition F.30. For the second half, $\bar { \ell } ^ { \mathrm { O S } } ( \bar { \theta } ) =$ min $\mathfrak { l } _ { y ^ { \prime } \in \mathcal { X } _ { > } ^ { \mathrm { a l l } } } \bar { \ell } ^ { \mathrm { s u b } } ( \bar { \theta } ; y ^ { \prime } )$ by Equation (5.2); applying the first half to a branch y<sup>′</sup> attaining this minimum gives $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0 .$ , and by Equation (5.3), min $\begin{array} { r } { \imath _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { O S } } ( \theta ) \leq } \end{array}$ $m _ { \mathrm { s u b } } ( y ^ { \prime } ) = 0$ . Equality holds since $\mathrm { \bar { \ell } ^ { O S } \geq 0 }$ □

The test-direction-cut inner solver (Definition F.19) also extends to $N \geq 1$ . At each evaluation point θ, let the test-direction oracle of each state return a maximizing cut $g _ { n }$ of ma $\mathrm { X } _ { g \in { \mathcal { T } } _ { y ^ { \prime } , n } \cup \{ \mathbf { 0 } \} } \left. \theta , g \right.$ , and use ${ \frac { 1 } { N } } \sum _ { n } g _ { n }$ , which we call the aggregated cut, as the cut. For the family of aggregated cuts

$$
\mathcal { F } _ { y ^ { \prime } } : = \Big \{ \frac { 1 } { N } \sum _ { n = 1 } ^ { N } g _ { n } ~ \Big | ~ g _ { n } \in \mathcal { T } _ { y ^ { \prime } , n } \cup \{ \mathbf { 0 } \} \Big \} ,
$$

we have $h _ { \mathcal { T } } ^ { N } ( \theta ) = \operatorname* { m a x } _ { f \in \mathcal { F } _ { y ^ { \prime } } } \langle \theta , f \rangle$ , and the aggregated cut is a maximizer of this maximum.

Corollary F.32 (Number of iterations of one call of the test-direction-cut inner solver with multiple states). In the above setting, run the test-direction-cut inner solver with aggregated cuts, with the zero cut 0 as the initial cut, the cutting-plane method as the rule for selecting evaluation points, tolerance $0 ,$ and $J = \infty ,$ and assume that the LPs are solved exactly. Then the solver stops within at most $\begin{array} { r } { | \mathcal { F } _ { y ^ { \prime } } | + 1 \leq \prod _ { n = 1 } ^ { N } \bigl ( | \mathcal { T } _ { y ^ { \prime } , n } | + 1 \bigr ) + 1 } \end{array}$ iterations, and the returned point is an exact minimizer over $\Delta ^ { d - 1 }$ of $h _ { \mathcal { T } } ^ { N }$ , with minimum value $\mu _ { N } ( y ^ { \prime } )$ . If each $\mathcal { X } ( s ^ { n } )$ is an M<sup>♮</sup>-convex set, then with the test set $\{ e _ { i } - e _ { j } \mid i \neq j \} \cup \{ \pm e _ { i } \} , | T _ { y ^ { \prime } , n } | \leq d ( d + 1 )$ , so the number is at most $( d ( d + 1 ) + 1 ) ^ { N } + 1$

Proof. The proof of Proposition F.25(i) uses only that the family of cuts is finite, that the oracle returns a maximizing cut of the model function at the evaluation point, and that the zero cut always belongs to the cut set. Replacing the family $\mathcal { T } _ { y ^ { \prime } } \cup \{ \mathbf { 0 } \}$ by $\mathcal { F } _ { y ^ { \prime } }$ and $h \tau$ by $h _ { \mathcal { T } } ^ { N }$ , the same argument holds. Since the zero cut is an element of $\mathcal { F } _ { y ^ { \prime } }$ and belongs to the cut set from the beginning, each non-stopping iteration $j \geq 1$ adds a new element of ${ \mathcal { F } } _ { y ^ { \prime } } \setminus \{ { \bf 0 } \}$ , so there are at most $| \mathcal { F } _ { y ^ { \prime } } | - 1$ of them. Adding iteration $j = 0$ and the stopping iteration, the number of iterations is at most $| \mathcal { F } _ { y ^ { \prime } } | + 1$ . Moreover, $\begin{array} { r } { | \mathcal { F } _ { y ^ { \prime } } | \leq \prod _ { n } ( | \mathcal { T } _ { y ^ { \prime } , n } | + 1 ) } \end{array}$ , and the bound in the M<sup>♮</sup>-convex case follows as in Proposition $\mathrm { F . 2 5 ( i ) }$ □

## Appendix G. Proofs for the generalization analysis

In this section, we prove the results of §6. We follow the notation of §6. Following the standard treatment in empirical process theory, we assume that the events appearing below (such as unions over all weights consistent with the observations) are measurable.

$\ell ^ { 0 1 } ( \theta ; x ^ { \prime } , s ) = 0$ holds if and only if the OS loss $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s )$ is 0 and the optimal solution is unique (Lemma $\mathrm { G . 2 } )$ . In this case, the oracle returns the unique optimal solution, which dominates $x ^ { \prime } ;$ hence $\operatorname { G E } ( \theta )$ is an upper bound on the probability that, at a new state, the optimal solution returned by the oracle does not dominate the agent’s action, regardless of which optimal solution the oracle returns when there are several. We do not use the OS loss itself because, for a weight whose optimal solution is not unique, the oracle may return an optimal solution that does not dominate the agent’s action even when the OS loss is 0 (§4). Under Assumption $6 . 1 ( 2 ) , \theta ^ { * }$ is a.s. consistent with the observations.

In this section, we use the following notation. For a set $Z \subseteq \mathbb { R } ^ { d }$ , we write Conv Z for its convex hull, and for a convex set $C ,$ we write ext $C$ for the set of its extreme points (§E.1). For a bounded polyhedron $P ,$ ext $P$ is the set of vertices of P. A point u of a convex set C is called an exposed point of C if, for some $\theta \in \mathbb { R } ^ { d }$ , u is the unique maximizer of $\langle \theta , \cdot \rangle$ over $C .$ . We write int $Z$ for the interior of a set Z. A face of a bounded polyhedron $P$ is a subset that can be written as arg ma $\mathfrak { x } _ { y \in P } \langle \theta , y \rangle$ for some $\theta \in \mathbb { R } ^ { d }$ . A finite family A of distinct afine hyperplanes $H = \{ x \mid g _ { H } ( x ) = 0 \}$ in $\mathbb { R } ^ { k }$ (where $g _ { H }$ is a nonconstant afine function) is called a hyperplane arrangement. For a sign pattern $\tau \in \{ - , + \} ^ { \mathcal { A } }$ , if $\{ x \in \mathbb { R } ^ { k } \mid \operatorname { s i g n } g _ { H } ( x ) = \tau _ { H } ( \forall H \in \mathcal { A } ) \}$ is nonemp $\mathrm { { , y , } }$ it is called a region of A. The regions form a partition of $\mathbb { R } ^ { k } \setminus \bigcup _ { H \in \mathcal { A } } H$ , and their number does not depend on the choice of signs of the $g _ { H }$

G.1. Preliminaries. We supplement the setting of $\ S 6$ . We assume that $x \in \mathcal { X } ( s )$ holds almost surely (a.s.) for $( s , x ) \sim \mathcal { P }$ . The ess sup in Assumption $6 . 1 ( 1 )$ is the essential supremum with respect to the marginal distribution of $\mathcal { P }$ in s. When $s$ is finite, $\kappa < \infty$ in Assumption $6 . 1 ( 1 )$ holds automatically, since each $\mathcal { V } ( s )$ is finite (Remark 4.3).

Lemma G.1. Assume Assumption $4 . 2 ( 2 ) ( 3 )$ , and fix $s \in S$ and $\theta \in \mathbb { R } ^ { d }$

(i): ext Conv ${ \mathcal { X } } ( s ) \subseteq { \mathcal { Y } } ( s ) \subseteq { \mathcal { X } } ( s )$ , Y(s) is a finite set, and we have max ${ \mathrm { \hat { \rho } } } _ { x \in { \mathcal { X } } ( s ) } \langle \theta , x \rangle = \operatorname* { m a x } _ { u \in { \mathcal { Y } } ( s ) } \langle \theta , u \rangle$

(ii): arg $\operatorname* { m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle$ is a singleton if and only if there is exactly one point u in $\mathcal { V } ( s )$ that maximizes $\langle \theta , \cdot \rangle$ , and in this case arg ma $\mathfrak { c } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle = \{ u \}$ In particular, if arg ma $\mathfrak { i } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle = \{ v \}$ , then $v \in \mathcal { V } ( s )$

Proof. (i) By Assumption 4.2(2), we can write ${ \mathcal { X } } ( s ) \ = \ U _ { i } Q _ { j }$ (finitely many nonempty bounded polyhedra $Q _ { j } )$ . Conv $\begin{array} { r } { \mathcal { X } ( s ) = \mathrm { C o n v } \bigl ( \bigcup _ { i } \operatorname { e x t } Q _ { j } \bigr ) } \end{array}$ is a bounded polyhedron, and its extreme points are contained in the finite set $\cup _ { j }$ ext $Q _ { j } \subseteq { \mathcal { X } } ( s )$ Let $u \in$ ext Conv $\mathcal { X } ( s )$ . Since the extreme points of a polyhedron are exposed points, there exists $\theta ^ { \prime }$ such that u is the unique maximizer over Conv $\mathcal { X } ( s )$ , and arg ma $\mathrm { x } _ { y \in \mathcal { X } ( s ) } \langle \theta ^ { \prime } , y \rangle = \{ u \}$ . Since the oracle returns an optimal solution, $x ^ { * } ( \theta ^ { \prime } , s ) =$ $u ,$ and thus $u \in \mathcal { Y } ( s ) . \mathcal { Y } ( s ) \subseteq \mathcal { X } ( s )$ and the finiteness of $\mathcal { V } ( s )$ are as stated in the remark after Assumption $4 . 2 ( 3 )$ . Since the maximum of a linear function over a compact convex set is attained at an extreme point, we have max $\chi _ { \left( s \right) } \langle \theta , \cdot \rangle \geq$ max $\begin{array} { r } { \langle y _ { ( s ) } \langle \theta , \cdot \rangle \geq \operatorname* { m a x } _ { \mathrm { e x t } \mathrm { C o n v } } \chi _ { ( s ) } \langle \theta , \cdot \rangle = \operatorname* { m a x } _ { \mathrm { C o n v } } \chi _ { ( s ) } \langle \theta , \cdot \rangle \geq \operatorname* { m a x } _ { \mathcal { X } ( s ) } \langle \theta , \cdot \rangle } \end{array}$

(ii) Let $c ^ { * }$ be the maximum, and set $U : = \{ u \in$ ext Conv ${ \mathcal { X } } ( s ) \mid \langle \theta , u \rangle = c ^ { * } \}$ and $U _ { \mathcal { V } } : = \{ u \in \mathcal { V } ( s ) \mid \langle \theta , u \rangle = c ^ { * } \}$ . By ${ \mathrm { ( i ) } } , \emptyset \neq U \subseteq U _ { \mathcal { V } } \subseteq \arg \operatorname* { m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle$ . The set $F ^ { * }$ of optimal solutions over Conv $\mathcal { X } ( s )$ is a face of the bounded polyhedron

Conv $\mathcal { X } ( s )$ , and since a face of a bounded polyhedron is the convex hull of the vertices it contains, ${ \cal F } ^ { * } = \mathrm { C o n v } { \cal U } ;$ by (i), arg ma ${ \mathfrak { c } } _ { y \in { \mathcal { X } } ( s ) } \langle \theta , y \rangle = { \mathcal { X } } ( s ) \cap F ^ { * }$ . If $| U _ { y } | = 1$ , then $| U | = 1$ , so $F ^ { * } = U \subseteq { \mathcal { X } } ( s )$ gives arg $\begin{array} { r } { \operatorname* { m a x } _ { y \in { \mathcal K } ( s ) } \langle \theta , y \rangle = U = U _ { y } } \end{array}$ . If $| U _ { \mathcal { Y } } | \ge 2$ , then $U _ { \mathcal { Y } } \subseteq \arg \operatorname* { m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle$ shows that arg $\operatorname* { m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle$ is not a singleton. □

Lemma G.2. Let $\theta \in \Theta , s \in { \mathcal { S } } .$ , and $x ^ { \prime } \in \mathcal { X } ( s )$

(i): We have $\ell ^ { 0 1 } ( \theta ; x ^ { \prime } , s ) = 0$ if and only if $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) = 0$ and the set arg $\operatorname* { m a x } _ { y \in { \mathcal { X } } ( s ) } \langle \theta , y \rangle$ is a singleton.

(ii): If $\ell ^ { 0 1 } ( \theta ; x ^ { \prime } , s ) = 0$ , then the optimal solution returned by the oracle satisfies $x ^ { * } ( \theta , s ) \geq x ^ { \prime }$

(iii): Under Assumption $6 . 1 ( 2 ) , \theta ^ { * }$ is a.s. consistent with the observations.

Proof. (i) As stated in $ \ S 4 , \ \ell ^ { \mathrm { O S } } ( \theta ; x ^ { \prime } , s ) = 0$ is equivalent to arg $\operatorname* { m a x } _ { y \in \mathcal { X } ( s ) } \langle \theta , y \rangle \cap$ $\chi _ { \geq x ^ { \prime } } ( s ) \neq \emptyset$ . When arg $\operatorname* { m a x } _ { y \in { \mathcal { X } } ( s ) } \langle \theta , y \rangle = \{ v \}$ , this is equivalent to $v \geq x ^ { \prime }$ . (ii) Since v is the only optimal solu $\operatorname { t i o n } , x ^ { * } ( \theta , s ) = v \geq x ^ { \prime } . { \mathrm { ~ ( i i ~ } }$ i) This follows from the fact that $\ell ^ { 0 1 } ( \theta ^ { \ast } ; x , s ) = 0$ holds a.s. □

G.2. Proof of Theorem 6.3. Fix finitely many points $z _ { i } = ( s _ { i } , x _ { i } ) ~ ( i = 1 , \dots , m$ $| \mathcal { y } ( s _ { i } ) | \leq \kappa )$ . For each i and distinct $u , u ^ { \prime } \in \mathcal { V } ( s _ { i } )$ , consider the hyperplane through the origin $\{ \theta \ | \ \langle \theta , u - u ^ { \prime } \rangle = 0 \}$ ; let A be the family of all such hyperplanes (identical hyperplanes are counted once), and let h be their number. Then $h \leq m \kappa ( \kappa - 1 ) / 2 \leq$ $m \kappa ^ { 2 } / 2$ . Fix a normal vector $a _ { H }$ of each hyperplane $H \in { \mathcal { A } }$ . For a sign vector $\sigma \in \{ - , 0 , + \} ^ { \mathcal { A } } , \mathrm { i f } \left\{ \theta \in \mathbb { R } ^ { d } \mid \mathrm { s i g n } \langle \theta , a _ { H } \rangle = \sigma _ { H } \left( \forall H \in \mathcal { A } \right) \right\}$ is nonempty, it is called a cell of A (we use this name to distinguish it from a face of a bounded polyhedron). $\mathbb { R } ^ { d }$ is partitioned into the cells of ${ \mathcal { A } } ,$ and a cell whose signs are all nonzero is a region of A.

Lemma G.3. Assume Assumption 4.2(2)(3). The map $\theta \mapsto ( \ell ^ { 0 1 } ( \theta ; x _ { i } , s _ { i } ) ) _ { i = } ^ { m }$ 1 $( \theta \in \Theta )$ is constant on the intersection of each cell of A with Θ.

Proof. On a cell, the sign of $\langle \theta , u \rangle - \langle \theta , u ^ { \prime } \rangle$ is constant for all i and $u , u ^ { \prime } \in \mathcal { V } ( s _ { i } )$ so the set of points in $\mathcal { \ V } ( s _ { i } )$ that maximize $\langle \theta , \cdot \rangle$ is constant. By Lemma $\mathrm { G . 1 } ( \mathrm { i i } )$ whether arg $\operatorname* { m a x } _ { y \in { \mathcal { X } } ( s _ { i } ) } \langle \theta , y \rangle$ is a singleton, and its element, are determined by this set alone. Hence $\ell ^ { 0 1 } ( \theta ; x _ { i } , s _ { i } )$ is constant. □

The following two lemmas bound the number of cells. In what follows, let $\begin{array} { r } { \Phi _ { k } ( h ) : = \sum _ { i = 0 } ^ { k } \bar { \binom { h } { i } } } \end{array}$ (with $\binom { h } { i } \ = \ 0$ for $i > h ) . \Phi _ { k } ( h )$ is nondecreasing in h and satisfies $\Phi _ { k } ( 0 ) = \bar { \Phi } _ { 0 } ( h ) = 1$ and $\Phi _ { k } ( h ) = \Phi _ { k } ( h - 1 ) + \Phi _ { k - 1 } ( h - 1 ) \ ( k , h \geq 1 )$

Lemma G.4. The number of regions of an arrangement of at most h hyperplanes in $\mathbb { R } ^ { k }$ is at most $\Phi _ { k } ( h )$ .

Proof. Let $r _ { k } ( h )$ be the maximum number of regions of an arrangement of at most h hyperplanes in $\mathbb { R } ^ { k }$ ; we show $r _ { k } ( h ) \leq \Phi _ { k } ( h )$ by induction on $k + h$ . If $h = 0$ , the only region is $\mathbb { R } ^ { k }$ , and if $k = 0 , \mathbb { R } ^ { 0 }$ is a single point and has no hyperplanes; hence $r _ { k } ( 0 ) = r _ { 0 } ( h ) = 1$ . Let $k , h \geq 1$ , let A be an arrangement of h hyperplanes in $\mathbb { R } ^ { k }$ let $H \in A .$ and set $\mathcal { A } ^ { \prime } : = \mathcal { A } \setminus \{ H \}$ . For a region $R ^ { \prime }$ of $\mathcal { A } ^ { \prime }$ with sign pattern $\tau ^ { \prime }$ , if $R ^ { \prime } \cap H = \emptyset$ , then g has no zeros on the convex set $R ^ { \prime }$ , so its sign is constant, and $R ^ { \prime }$ is a region of A. If $R ^ { \prime } \cap H \neq \emptyset$ , then, since $R ^ { \prime }$ is an open convex $\operatorname { s e t } .$ , it contains both points with $g _ { H } > 0$ and points with $g _ { H } < 0$ , and $R ^ { \prime } \setminus H$ splits into the two regions ${ \cal R } ^ { \prime } \cap \{ g _ { \cal H } > 0 \}$ and $R ^ { \prime } \cap \{ g _ { H } < 0 \}$ of A. Since each region of $\mathcal { A }$ is contained in exactly one $R ^ { \prime }$ , the number of regions of $\mathcal { A }$ equals the number of regions of $\mathbf { \mathcal { A } ^ { \prime } }$ plus the number of regions of $\mathcal { A } ^ { \prime }$ that intersect $H$

We bound the number of regions of $\mathcal { A } ^ { \prime }$ that intersect H. Identify H with $\mathbb { R } ^ { k - 1 }$ via an afine isomorphism. If $H ^ { \prime } \in \mathcal { A } ^ { \prime }$ is parallel to $H \ ( H \cap H ^ { \prime } \ = \ \varnothing )$ then $g _ { H ^ { \prime } }$ has constant sign on $H ;$ otherwise, $H \cap H ^ { \prime }$ is a hyperplane of H. Let $\boldsymbol { \mathcal { A } } _ { H }$ be the family of all the latter $H \cap H ^ { \prime }$ (duplicates counted once); then $\boldsymbol { \mathcal { A } } _ { H }$ is an arrangement of at most $h - 1$ hyperplanes in $H$ . If $R ^ { \prime } \cap H \neq \emptyset$ , then ${ \cal R } ^ { \prime } \cap { \cal H } = \{ x \in { \cal H } \ | \ \mathrm { s i g n } g _ { { \cal H } ^ { \prime } } ( x ) = \tau _ { { \cal H } ^ { \prime } } ^ { \prime } \ ( \forall { \cal H } ^ { \prime } \in { \cal A } ^ { \prime } ) \}$ is a region of $\boldsymbol { \mathcal { A } } _ { H }$ (the conditions for parallel $H ^ { \prime }$ hold identically on $H )$ . Since distinct $R ^ { \prime }$ are disjoint, the map $R ^ { \prime } \mapsto R ^ { \prime } \cap$ H is injective. Hence the number of regions of $\mathcal { A } ^ { \prime }$ that intersect H is at most $r _ { k - 1 } ( h - 1 )$ . Therefore $r _ { k } ( h ) \leq r _ { k } ( h - 1 ) + r _ { k - 1 } ( h - 1 )$ , and the induction hypothesis together with $\Phi _ { k } ( h ) = \Phi _ { k } ( h - 1 ) + \Phi _ { k - 1 } ( h - 1 )$ yields $r _ { k } ( h ) \leq \Phi _ { k } ( h )$ □

Lemma G.5. The number of cells of an arrangement of h hyperplanes through the origin in $\mathbb { R } ^ { d }$ is at most $\Phi _ { d } ( 2 h )$ .

Proof. Let $H \in { \mathcal { A } }$ be the hyperplanes of the arrangement and $a _ { H }$ their normal vectors, and let $\Sigma \subseteq \{ - , 0 , + \} ^ { A }$ be the set of sign vectors that can be written as $( \mathrm { s i g n } \langle \theta , a _ { H } \rangle ) _ { H \in \mathcal { A } }$ for some $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ (that is, the sign vectors corresponding to cells); Σ is a finite set. For each $\sigma \in \Sigma .$ , choose a point $\theta ^ { \sigma }$ realizing the sign vector $\sigma .$ Choose $\delta > 0$ such that $\left| \langle \theta ^ { \sigma } , a _ { H } \rangle \right| > \delta$ for all $( \sigma , H )$ with $\langle \theta ^ { \sigma } , a _ { H } \rangle \neq 0$ . Let $\mathcal { A } _ { \delta }$ be the arrangement of the 2h afine hyperplanes $\{ \theta \mid \langle \theta , a _ { H } \rangle = \delta \}$ and $\left\{ \theta \mid \langle \theta , a _ { H } \rangle = - \delta \right\}$ $( H \in { \mathcal { A } } )$ (these are distinct, since the normal vectors of distinct H are not parallel). By the choice of $\delta ,$ no $\theta ^ { \sigma }$ lies on any hyperplane of $\mathcal { A } _ { \delta }$ , so each $\theta ^ { \sigma }$ belongs to some region $R _ { \sigma }$ of $A _ { \delta } . ~ R _ { \sigma }$ determines, for each $H$ , the signs of $\langle \theta , a _ { H } \rangle - \delta$ and $\langle \theta , a _ { H } \rangle + \delta$ and $\sigma _ { H }$ is determined from them: $- \ \mathrm { i f }$ both are negative, 0 if they have opposite signs, and + if both are positive. Hence $\sigma \mapsto R _ { \sigma }$ is injective, and by Lemma G.4, $| \Sigma | \leq \Phi _ { d } ( 2 h )$ □

Let $G ( m )$ denote the maximum, over m points $( s _ { i } , x _ { i } ) \ ( | \mathcal { V } ( s _ { i } ) | \ \leq \ \kappa )$ , of the number of error value patterns $( \ell ^ { 0 1 } ( \theta ; x _ { i } , s _ { i } ) ) _ { i = 1 } ^ { m } \in \{ 0 , 1 \} ^ { m }$ realized by $\theta \in \Theta$ (the growth function). From $2 h \le m \kappa ( \kappa - 1 ) \le m \kappa ^ { 2 }$ , the monotonicity of $\Phi _ { d } .$ , and Lemmas G.3 and G.5, we obtain

$$
G ( m ) \leq \Phi _ { d } ( m \kappa ^ { 2 } )\tag{G.1}
$$

Moreover, if $n \geq d ,$ then we have

$$
\Phi _ { d } ( n ) \leq \left( { \frac { n } { d } } \right) ^ { d } \sum _ { i = 0 } ^ { d } { \binom { n } { i } } \left( { \frac { d } { n } } \right) ^ { i } \leq \left( { \frac { n } { d } } \right) ^ { d } \left( 1 + { \frac { d } { n } } \right) ^ { n } \leq \left( { \frac { \mathrm { e } n } { d } } \right) ^ { d } .\tag{G.2}
$$

Here, the first inequality holds because $n / d \ge 1$ implies, for each $i = 0 , \ldots , d$ $\textstyle { \binom { n } { i } } \leq { \binom { n } { i } } \left( { \frac { n } { d } } \right) ^ { d - i } = \left( { \frac { n } { d } } \right) ^ { d } { \binom { n } { i } } \left( { \frac { d } { n } } \right) ^ { i }$ . The second inequality follows by adding the nonnegative terms for $i = d + 1 , \ldots , n$ (recall $d \leq n )$ and applying the binomial theorem $\begin{array} { r } { \sum _ { i = 0 } ^ { n } \binom { n } { i } \left( \frac { d } { n } \right) ^ { i } = \left( 1 + \frac { d } { n } \right) ^ { n } } \end{array}$ . The last inequality follows from $1 + t \leq \mathrm { e } ^ { t }$ $( t \in \mathbb { R } )$ with $t = d / n$ , which gives $\left( 1 + \frac { d } { n } \right) ^ { n } \leq \mathrm { e } ^ { d }$

Proof of Theorem 6.3. Write $z ^ { n } : = ( s ^ { n } , x ^ { n } )$ and $E _ { \theta } : = \{ ( s , x ) \mid \ell ^ { 0 1 } ( \theta ; x , s ) = 1 \}$ 2 and set

$$
\varepsilon : = \frac { 2 } { N } \left( d \log _ { 2 } \frac { 2 \mathrm { e } N \kappa ^ { 2 } } { d } + \log _ { 2 } \frac { 2 } { \beta } \right) .
$$

Define the event $\mathcal { E }$ as “there exists $\theta \in \Theta$ that is consistent with the observations and satisfies $\mathrm { G E } ( \theta ) > \varepsilon . ^ { \prime \prime }$ If $\mathcal { E }$ does not occur, the generalization error of every weight consistent with the observations is at most $\varepsilon ,$ so it sufices to show $\mathbb { P } ( \mathcal { E } ) \le \beta$ Below, we show $\mathbb { P } ( \mathcal { E } ) \leq 2 \mathbb { P } ( \mathcal { E } ^ { \prime } )$ in Step 2 (where $\mathcal { E } ^ { \prime }$ is the event defined in Step 2), $\mathbb { P } ( \mathcal { E } ^ { \prime } ) \le G ( 2 N ) 2 ^ { - \varepsilon N / 2 }$ in Step $^ { 3 , }$ and $2 G ( 2 N ) 2 ^ { - \varepsilon N / 2 } \leq \beta$ in Step 4.

Step 1 (checking $N \varepsilon \geq 8 )$ : Since $N \geq d$ and $\kappa \geq 2$ , we have $N \kappa ^ { 2 } / d \geq 4$ , so $N \varepsilon \geq 2 d \log _ { 2 } ( 8 \mathrm { e } ) > 8$

Step 2 (symmetrization): Independently of the observations, prepare an i.i.d. sample $z ^ { \prime 1 } , \ldots , z ^ { \prime N } \sim \mathcal { P }$ . Define the event $\mathcal { E } ^ { \prime }$ as “there exists $\theta \in \Theta$ that is consistent with the observations and for which the number of n with $z ^ { \prime n } \in E _ { \theta }$ is at least $\varepsilon N / 2 . ^ { \cdot }$ Write $\mathcal { E } _ { \omega } ^ { \prime }$ for the section of $\mathcal { E } ^ { \prime }$ obtained by fixing the observations $\boldsymbol { \upsilon } : = ( z ^ { 1 } , \dots , \dot { z } ^ { N } )$ Since the observations and $z ^ { \prime } : = ( z ^ { \prime 1 } , \dots , z ^ { \prime N } )$ are independent, when integrating the indicator of $\mathcal { E } ^ { \prime }$ with respect to the product measure in $( \omega , z ^ { \prime } )$ , we may first integrate in $z ^ { \prime }$ and then in ω (the Fubini–Tonelli theorem), so $\mathbb { P } ( \mathcal { E } ^ { \prime } ) = \mathbb { E } _ { \omega } [ \mathbb { P } _ { z ^ { \prime } } ( \mathcal { E } _ { \omega } ^ { \prime } ) ]$ Let $\omega \in { \mathcal { E } }$ , and take one $\theta ^ { \omega }$ that is consistent with the observations and satisfies $p : = \mathrm { G E } ( \theta ^ { \omega } ) > \varepsilon$ The number $Y$ of n with $z ^ { \prime n } \in E _ { \theta ^ { \circ } }$ ω follows the binomial distribution Bin $( N , p )$ , so by Chebyshev’s inequality and Step 1,

$$
\mathbb { P } _ { z ^ { \prime } } \Big ( Y < \frac { N p } { 2 } \Big ) \le \mathbb { P } _ { z ^ { \prime } } \Big ( | Y - N p | > \frac { N p } { 2 } \Big ) \le \frac { 4 N p ( 1 - p ) } { ( N p ) ^ { 2 } } \le \frac { 4 } { N \varepsilon } \le \frac { 1 } { 2 } .
$$

If $Y \geq N p / 2$ , then $Y \ge \varepsilon N / 2$ , and $\theta ^ { \omega }$ satisfies the condition of $\mathcal { E } _ { \omega } ^ { \prime }$ ; hence $\mathbb { P } _ { z ^ { \prime } } ( \mathcal { E } _ { \omega } ^ { \prime } ) \geq$ $1 / 2$ . Therefore $\mathbb { P } ( \mathcal { E } ^ { \prime } ) \geq \mathbb { E } _ { \omega } [ \mathbf { 1 } _ { \mathcal { E } } ( \omega ) / 2 ] = \mathbb { P } ( \mathcal { E } ) / 2$

Step 3 (swapping): Let w $\mathbf { \Psi } : = ( z ^ { 1 } , \dots , z ^ { \hat { N } } , z ^ { \prime 1 } , \dots , z ^ { \prime N } )$ . For $\tau \in \{ 0 , 1 \} ^ { N }$ , write $\operatorname { S w } _ { \tau } w$ for the tuple obtained by swapping $z ^ { n }$ and $z ^ { \prime n }$ for each n with $\tau _ { n } = 1$ . Since the 2N components of w are i.i.d., $\operatorname { S w } _ { \tau } u$ and w have the same distribution for each $\tau ,$ and $\mathbb { P } ( \operatorname { S w } _ { \tau } \boldsymbol { w } \in \mathcal { E } ^ { \prime } ) = \mathbb { P } ( \boldsymbol { w } \in \mathcal { E } ^ { \prime } ) = \mathbb { P } ( \mathcal { E } ^ { \prime } )$ . Averaging this over the $2 ^ { N }$ choices of $\tau \in \{ 0 , 1 \} ^ { N }$ and interchanging the finite sum and the expectation, we obtain

$$
\mathbb { P } ( \mathcal { E } ^ { \prime } ) = \frac { 1 } { 2 ^ { N } } \sum _ { \tau \in \{ 0 , 1 \} ^ { N } } \mathbb { E } _ { w } \big [ \mathbf { 1 } \{ \mathrm { S w } _ { \tau } w \in \mathcal { E } ^ { \prime } \} \big ] = \mathbb { E } _ { w } \big [ \mathbb { P } _ { \tau } ( \mathrm { S w } _ { \tau } w \in \mathcal { E } ^ { \prime } ) \big ] .
$$

Here, for fixed $w , ~ \mathbb { P } _ { \tau } ( \mathrm { S w } _ { \tau } w \in \mathcal { E } ^ { \prime } ) : = 2 ^ { - N } | \{ \tau \ \in \ \{ 0 , 1 \} ^ { N } \ | \ \ \mathrm { S w } _ { \tau } w \in \mathcal { E } ^ { \prime } \} |$ is the probability when τ is drawn uniformly. Fix $w ,$ and let $\Xi ( w ) \subseteq \{ 0 , 1 \} ^ { 2 N }$ be the set of error value patterns realized by $\theta \in \Theta$ on the 2N points of w. Since each component of w satisfies $| \mathcal { V } ( s ) | \le \kappa \mathrm { ~ a . s . }$ ., we have $| \Xi ( w ) | \le G ( 2 N )$ a.s. Since the error value pattern of θ on $\operatorname { S w } _ { \tau }$ w is obtained by swapping the components of its pattern $\xi$ on w according to $\tau ,$ if $\operatorname { S w } _ { \tau } w \in { \mathcal { E } } ^ { \prime }$ , then for some $\xi \in \Xi ( w )$ the following event $\mathcal { E } _ { \xi }$ occurs: “after swapping, the first N components are all 0, and at least $\varepsilon N / 2$ of the last N components are $1 . { } ^ { \mathfrak { n } }$ Let r be the number of components of $\xi$ equal to 1. If $r < \varepsilon N / 2$ , then $\mathcal { E } _ { \xi }$ does not occur. If, for some $n ,$ both the nth and the $( N + n ) \mathrm { t h }$ components of $\xi$ are 1, then a 1 remains in the first half regardless of the swap, so $\mathcal { E } _ { \xi }$ does not occur. Otherwise, the r components equal to 1 belong to distinct $n _ { \colon }$ , and $\mathcal { E } _ { \xi }$ is equivalent to all of them being placed in the second half, so $\mathbb { P } _ { \tau } ( \mathcal { E } _ { \xi } ) =$ $2 ^ { - r } \leq 2 ^ { - \varepsilon N / 2 }$ . Therefore $\begin{array} { r } { \mathbb { P } _ { \tau } ( \mathrm { S w } _ { \tau } w \in \mathcal { E } ^ { \prime } ) \le \sum _ { \xi \in \Xi ( w ) } \mathbb { P } _ { \tau } ( \mathcal { E } _ { \xi } ) \le G ( 2 N ) 2 ^ { - \varepsilon N / 2 } } \end{array}$ , and we obtain $\mathbb { P } ( \mathcal { E } ^ { \prime } ) \le G ( 2 N ) 2 ^ { - \varepsilon N / 2 }$

Step 4 : By Steps 2 and $3 , \ \mathbb { P } ( \mathcal { E } ) \ \le \ 2 G ( 2 N ) 2 ^ { - \varepsilon N / 2 }$ . Since $2 N \kappa ^ { 2 } \geq d ,$ Equations (G.1) and (G.2) give $G ( 2 N ) \leq ( 2 \mathrm { e } N \kappa ^ { 2 } / d ) ^ { d }$ , and by the definition of $\varepsilon$

$$
2 G ( 2 N ) 2 ^ { - \varepsilon N / 2 } \leq 2 \left( \frac { 2 \mathrm { e } N \kappa ^ { 2 } } { d } \right) ^ { d } 2 ^ { - d \log _ { 2 } ( 2 \mathrm { e } N \kappa ^ { 2 } / d ) - \log _ { 2 } ( 2 / \beta ) } = \beta .
$$

Hence $\mathbb { P } ( \mathcal { E } ) \le \beta$ . That is, with probability at least $1 - \beta , \mathcal { E }$ does not occur, and every weight $\hat { \theta } ^ { N }$ consistent with the observations satisfies ${ \mathrm { G E } } ( { \hat { \theta } } ^ { N } ) \leq \varepsilon$ . This is the statement of Theorem 6.3. □

G.3. Proof of Theorem 6.5. Consider the following instance. Let $d _ { 0 } \geq 2$ and $d : = 2 d _ { 0 } + 1$ , and write elements of $\mathbb { R } ^ { d }$ as $( y _ { 0 } , y _ { + } , y _ { - } ) \in \mathbb { R } \times \mathbb { R } ^ { d _ { 0 } } \times \mathbb { R } ^ { d _ { 0 } }$ . Let the state set be the unit sphere $\mathcal { S } : = \mathbb { S } ^ { d _ { 0 } - 1 } \subset \mathbb { R } ^ { \dot { d _ { 0 } } }$ , and let the state distribution $\mathcal { P } _ { \mathcal { S } }$ be absolutely continuous with respect to the uniform surface measure on $\mathbb { S } ^ { d _ { 0 } - 1 }$ . Let the feasible region be

$$
\mathcal { X } ( s ) : = \{ 0 , \ ( - 1 , s , - s ) , \ ( - 3 , - s , s ) \}
$$

and let the agent’s action be $x = 0$ . That is, the distribution $\mathcal { P }$ of state–action pairs is given by $s \sim \mathcal { P } _ { s }$ and $x = 0$ . Let the weight space be $\Theta : = \Delta ^ { d - 1 }$ and $\theta ^ { * } : = ( 1 , 0 , 0 ) \in \Theta$ . For $\theta = ( \theta _ { 0 } , p , q )$ , setting $u : = p - q \in \mathbb { R } ^ { d _ { 0 } }$ , we have

$$
\langle \theta , 0 \rangle = 0 , \quad \langle \theta , ( - 1 , s , - s ) \rangle = - \theta _ { 0 } + \langle s , u \rangle , \quad \langle \theta , ( - 3 , - s , s ) \rangle = - 3 \theta _ { 0 } - \langle s , u \rangle .\tag{G.3}
$$

This is obtained from the instance in the proof of Theorem 3.1 of Fatemi et al. (2026) $( \psi ( s , a ) = ( - 2 | a | + a , a s ) , a \in [ - 1 , 1 ] , \theta = ( 1 , u ) )$ by lifting the features corresponding to $a \in \{ - 1 , 0 , 1 \}$ so that the positive and negative parts of u are placed in separate coordinates.

Lemma G.6. This instance satisfies Assumption 4.2 and Assumption 6.1 with $\kappa = 3$ . Moreover, for $\theta = ( \theta _ { 0 } , p , q ) \in \Theta$ and $u = p - q , \ell ^ { 0 1 } ( \theta ; 0 , s ) = 0$ is equivalent to ${ ^ { 6 4 } \langle s , u \rangle } < \theta _ { 0 }$ and $- \langle s , u \rangle < 3 \theta _ { 0 } .$

Proof. Since $\mathcal { X } ( s )$ consists of three points (a union of bounded polyhedra each consisting of a single point), it satisfies Assumption $4 . 2 ( 2 )$ , and every oracle satisfies Assumption 4.2(3). The three points are distinct and are all extreme points of the convex hull, so by Lemma $\mathrm { G . 1 } ( \mathrm { i } ) , \mathcal { V } ( s ) = \mathcal { X } ( s )$ and $\kappa = 3$ . By Equation (G.3), the values under $\theta ^ { * }$ are $0 , - 1 , - 3$ , so arg $\operatorname* { m a x } _ { y \in { \mathcal { X } } ( s ) } \langle \theta ^ { * } , y \rangle = \{ 0 \}$ and $x ^ { * } ( \theta ^ { * } , s ) =$ $0 \geq x$ . Since the other two points have a negative first component, $\mathcal { X } _ { \geq 0 } ( s ) = \{ 0 \}$ arg $\operatorname* { m a x } _ { y \in { \mathcal { X } } ( s ) } \langle \theta , y \rangle = \{ v \}$ with $v \geq 0$ holds only when $v = 0$ is the unique optimal solution, so Equation (G.3) gives the latter equivalence. □

The proof uses the following two propositions. The first is Theorem 1 of Campi and Garatti (2008) on the violation probability of scenario programs. Here we state it in the form used by Fatemi et al. (2026) in Theorem A.1 and in the proof of Theorem 3.1.

Proposition $\mathbf { G . 7 }$ (Theorem 1 of Campi and Garatti (2008)). Let k be a positive integer and $c \in \mathbb { R } ^ { k }$ , and suppose that each point ξ of a probability space $( \Xi , P _ { \Xi } )$ is associated with a closed convex set $C _ { \xi } \subseteq \mathbb { R } ^ { k }$ . Let $\xi ^ { 1 } , \dots , \xi ^ { N }$ be an i.i.d. sample from $P _ { \Xi }$ , and suppose that the scenario program

$$
\operatorname* { m i n } _ { u \in \mathbb { R } ^ { k } } \langle c , u \rangle \quad \mathrm { s . t . } \quad u \in C _ { \xi ^ { n } } \quad ( n = 1 , \dots , N )
$$

has a.s. a unique solution $\hat { u } _ { N }$ . Set $V ( u ) : = P _ { \Xi } ( u \notin C _ { \xi } )$ . Then, for any $\varepsilon \in ( 0 , 1 ]$ , we have

$$
\mathbb { P } \big ( V ( \hat { u } _ { N } ) > \varepsilon \big ) \le \sum _ { i = 0 } ^ { k - 1 } \binom { N } { i } \varepsilon ^ { i } ( 1 - \varepsilon ) ^ { N - i } .
$$

For each $n ,$ the constraint $u \in C _ { \xi ^ { n } }$ is called a support constraint if the scenario program with only this constraint removed has an optimal solution diferent from

$\hat { u } _ { N }$ or has no optimal solution. If the number of support constraints is a.s. exactly k (that is, if the problem is fully supported in the sense of Definition A.5 of Fatemi et al. (2026)), then the above inequality holds with equality.

The second is the following property of an LP, shown in the proof of Theorem 3.1 of Fatemi et al. (2026). Fix $\bar { c ^ { \prime } } \in \bar { \mathbb { R } ^ { d _ { 0 } } } \setminus \left\{ 0 \right\}$ , and for $s ^ { 1 } , \ldots , s ^ { N } \sim \mathcal { P } _ { S } ~ ( \mathrm { i . i . d . } )$ consider

$$
\operatorname* { m i n } _ { u \in \mathbb { R } ^ { d _ { 0 } } } \langle c ^ { \prime } , u \rangle \quad \mathrm { s . t . } \quad \langle s ^ { n } , u \rangle \leq 1 , \quad - \langle s ^ { n } , u \rangle \leq 3 \quad ( n = 1 , \ldots , N ) .\tag{G.4}
$$

Proposition G.8 (Proof of Theorem 3.1 of Fatemi et al. (2026)). Let $N \geq d _ { 0 }$ Then Equation (G.4) has a.s. a unique solution $\hat { u } _ { N }$ . Moreover, the number of n such that the LP with both inequalities $\langle s ^ { n } , u \rangle \leq 1$ and $- \langle s ^ { n } , u \rangle \leq 3$ corresponding to observation n removed has an optimal solution diferent from $\hat { u } _ { N }$ or has no optimal solution is a.s. exactly $d _ { 0 }$

Equation (G.4) is the scenario program in Proposition G.7 with $\begin{array} { r } { k = d _ { 0 } , \Xi = { \cal S } , } \end{array}$ $P _ { \Xi } = \mathcal { P } _ { S } , c = c ^ { \prime } ,$ and $C _ { s } : = \{ u \mid \langle s , u \rangle \leq 1 , - \langle s , u \rangle \leq 3 \}$ . Hence, when $N \geq d _ { 0 }$ Propositions G.7 and G.8 imply that, for $V ( u ) : = \mathcal { P } _ { S } ( \langle s , u \rangle > 1 \mathrm { ~ o r ~ } - \langle s , u \rangle > 3 )$ and any $\varepsilon \in ( 0 , 1 ]$ 2

$$
\mathbb { P } \big ( V ( \hat { u } _ { N } ) > \varepsilon \big ) = \sum _ { i = 0 } ^ { d _ { 0 } - 1 } \binom { N } { i } \varepsilon ^ { i } ( 1 - \varepsilon ) ^ { N - i } .\tag{G.5}
$$

We assume $d _ { 0 } \geq 2$ because, when $d _ { 0 } = 1$ , the distribution on $\mathbb { S } ^ { 0 } = \{ \pm 1 \}$ has atoms, identical constraints are duplicated, and the property in Proposition G.8 fails. Note that Equation $\left( \mathrm { G . 4 } \right)$ is Eq. (6) of Fatemi et al. (2026) written out for the normalization $\Theta _ { \mathrm { F } } : = \{ ( 1 , u ) \ | \ u \in \mathbb { R } ^ { d _ { 0 } } \}$ and $\Psi ( \theta ) : = \langle ( 0 , c ^ { \prime } ) , \theta \rangle$ . This Ψ belongs to the class of objective functions $\mathcal { T } : = \{ \theta \mapsto f ( \langle c , \theta \rangle + b ) \ | \ f$ is strictly monotone, $b \in$ $\mathbb { R } , \ \langle c , \theta \rangle$ is not constant on $\Theta _ { \mathrm { F } } \}$ targeted by the lower bound in that paper. Moreover, $V ( \hat { u } _ { N } )$ is the set-level mismatch of $\hat { \theta } = ( 1 , \hat { u } _ { N } )$ , that is, the probability that the true optimal solution 0 is not contained in the set of optimal solutions of $\hat { \theta }$ (d and T in that paper correspond to $d _ { 0 }$ and N in this section).

Proof of Theorem 6.5. Step 1 : For $\eta ~ \in ~ ( 0 , 1 )$ , let $u _ { \eta } : = ( 1 - \eta ) \hat { u } _ { N }$ and $\theta ^ { \eta } : =$ $( 1 , u _ { \eta } ^ { + } , u _ { \eta } ^ { - } ) / ( 1 + \| u _ { \eta } \| _ { 1 } ) \in \Theta$ (where $u ^ { \pm }$ are the componentwise positive and negative parts). Setting $Z _ { \eta } : = 1 + \| u _ { \eta } \| _ { 1 } > 0$ , the first component of $\theta ^ { \eta }$ is $1 / Z _ { \eta }$ and $p - q = u _ { \eta } / Z _ { \eta } ,$ so by Lemma $\mathrm { G . 6 , ~ } \ell ^ { 0 1 } ( \theta ^ { \eta } ; 0 , s ) = 0$ is equivalent to ${ ^ { \ast } \langle s , u _ { \eta } \rangle } < \mathrm { \dot { 1 } }$ and $- \langle s , u _ { \eta } \rangle < 3 . ^ { \mathfrak { n } }$ For each n, the constraints of Equation (G.4) give $\langle s ^ { n } , u _ { \eta } \rangle \leq 1 - \eta < 1$ and $- \langle s ^ { n } , u _ { \eta } \rangle \leq 3 ( 1 - \eta ) < 3$ , so $\theta ^ { \eta }$ is consistent with the observations.

Step 2 : By Lemma G.6,

$$
\operatorname { G E } ( \theta ^ { \eta } ) = \mathcal { P } _ { S } ( \Lambda _ { \eta } ) , \quad \Lambda _ { \eta } : = \Big \{ \langle s , \hat { u } _ { N } \rangle \geq \frac { 1 } { 1 - \eta } \mathrm { ~ o r ~ } - \langle s , \hat { u } _ { N } \rangle \geq \frac { 3 } { 1 - \eta } \Big \} .
$$

As η decreases, $\Lambda _ { \eta }$ increases monotonically, and $\begin{array} { r } { \bigcup _ { \eta \in ( 0 , 1 ) } \Lambda _ { \eta } = \{ \langle s , \hat { u } _ { N } \rangle > 1 } \end{array}$ or − $\left. s , { \hat { u } } _ { N } \right. > 3 \}$ . By continuity of measure from below, li $\mathrm { n } _ { \eta \downarrow 0 } ^ { \cdot } \mathrm { G E } ( \theta ^ { \eta } ) = V ( \hat { u } _ { N } )$ . Fix $\varepsilon \in \mathsf { \Gamma } ( 0 , 1 ]$ ; if there exists an integer $k \geq 2$ with $\operatorname { G E } ( \theta ^ { 1 / k } ) \ > \ \varepsilon ,$ let $k _ { N }$ be the smallest such integer, and otherwise let $k _ { N } : = 2 ;$ set $\hat { \theta } ^ { N } : = \theta ^ { 1 / k _ { N } } . \hat { \theta } ^ { N }$ is a function of the observations and, by Step 1, is consistent with the observations. Since $\operatorname { G E } ( \theta ^ { 1 / k } ) \to V ( { \hat { u } } _ { N } )$ also along $\eta = 1 / k \ ( k \to \infty ) , V ( \hat { u } _ { N } ) > \varepsilon$ implies $\mathrm { G E } ( { \hat { \theta } } ^ { N } ) > \varepsilon$

Hence, by Equation (G.5),

$$
\mathbb { P } \big ( \mathrm { G E } ( \hat { \theta } ^ { N } ) > \varepsilon \big ) \ge \sum _ { i = 0 } ^ { d _ { 0 } - 1 } \binom { N } { i } \varepsilon ^ { i } ( 1 - \varepsilon ) ^ { N - i } .\tag{G.6}
$$

Step 3: Let $X \sim \mathrm { B i n } ( N , \varepsilon )$ ; then the right-hand side of Equation (G.6) equals $\mathbb { P } ( X \leq d _ { 0 } - 1 )$ . For $\varepsilon _ { 1 } : = d _ { 0 } / ( 2 N )$ , Markov’s inequality gives $\mathbb { P } ( X \geq d _ { 0 } ) \leq N \varepsilon _ { 1 } / d _ { 0 } =$ $1 / 2 ,$ , so the right-hand side is at least $1 / 2 \geq \beta .$ For $\varepsilon _ { 2 } : = \ln ( 1 / \beta ) / ( 2 N ) \le 1 / 2 .$ since $1 - x \geq \mathrm { e } ^ { - 2 x }$ for $x \in [ 0 , 1 / 2 ]$ , the right-hand side is at least $( 1 - \varepsilon _ { 2 } ) ^ { N } \geq \mathrm { e } ^ { - 2 N \varepsilon _ { 2 } } = \beta$ Taking $\hat { \theta } ^ { N }$ of Step 2 for $\varepsilon : = \operatorname* { m a x } \{ \varepsilon _ { 1 } , \varepsilon _ { 2 } \}$ , Equation (G.6) shows that, with probability at least $\beta , \operatorname { G E } ( { \hat { \theta } } ^ { N } ) > \varepsilon \geq ( \varepsilon _ { 1 } + \varepsilon _ { 2 } ) / 2 = ( d _ { 0 } + \ln ( 1 / \beta ) ) / ( 4 N )$ □

Remark G.9. Remove the nonnegativity constraint on the weights, and let the weight space be $\Theta _ { \mathrm { F } } = \{ ( 1 , u ) \ | \ u \in \mathbb { R } ^ { d _ { 0 } } \} \subset \mathbb { R } ^ { d _ { 0 } + 1 }$ . In this remark only, we define $\ell ^ { 0 1 }$ and GE of Definition 6.2 for elements of $\Theta _ { \mathrm { F } }$ by the same formulas. In the instance of Theorem 3.1 of Fatemi et al. (2026), $\mathcal { X } ( s ) : = \{ 0 , ( - 1 , s ) , ( - 3 , - s ) \}$ and $x = 0$ , we have $\mathcal { X } _ { > 0 } ( s ) = \{ 0 \}$ , so the OS loss coincides with the suboptimality loss. Hence the estimator $\hat { \theta } : = \arg \operatorname* { m i n } _ { \theta \in \Theta _ { \mathrm { F } } } \Psi ( \theta )$ s.t. $\ell ^ { \mathrm { O S } } ( \theta ; x ^ { n } , s ^ { n } ) \le 0 \ ( \forall n )$ , which is Eq. (6) of that paper written with the OS loss, coincides with the estimator of that paper written with the suboptimality loss (Eq. (6) of that paper, denoted there by ${ \hat { \theta } } _ { T } ^ { \mathrm { s u b } } )$ ). Furthermore, since the set-level mismatch $0 \not \in$ arg $\mathrm { m a x } _ { y \in \mathcal { X } ( s ) } \langle \hat { \theta } , y \rangle$ implies $\ell ^ { 0 1 } ( \hat { \theta } ; 0 , s ) = 1$ (by the same argument as in Remark 3.2 of that paper), the lower bound of Theorem 3.1 of that paper, $\begin{array} { r } { \operatorname* { s u p } _ { \theta ^ { * } , \mathcal { P } _ { S } } \operatorname* { i n f } _ { \Psi \in \mathcal { I } } \mathbb { P } ( \operatorname { G E } ( \hat { \theta } ) > \varepsilon ) \ge } \end{array}$ $\begin{array} { r } { \sum _ { i = 0 } ^ { d _ { 0 } - 1 } \binom { N } { i } \varepsilon ^ { i } ( 1 - \varepsilon ) ^ { N - i } } \end{array}$ , holds as it is. However, since $\hat { \theta }$ is a vertex of the LP, the optimal solution is not unique at observations with active constraints, so $\hat { \theta }$ is not consistent in the sense of this paper. Theorem 6.5 converts this into a lower bound for consistent weights by shrinking $\hat { \theta }$ toward the interior.

## Appendix H. Details of the numerical experiments

This section gives the details of the numerical experiments in $\ S 7 .$

H.1. Problem settings of the experiments. This subsection gives the mathematical models of the forward problems of the five experiments and how their instances are constructed. In all four experiments other than the single-machine scheduling problem, the number of states is $N = 2 0$ , the dimension is $d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ and $T = 1 0 0$ , that is, the oracle-call budget is $B = 4 0 0 0$ . In every experiment, all the methods use the same instance within the same episode.

Knapsack problem. We consider decisions on how many units of each item to pack under a limited capacity $( \mathrm { e . g . }$ , loading or inventory planning). The forward problem is the bounded integer knapsack problem

$$
\operatorname* { m a x } _ { x \in \mathbb { Z } ^ { d } } \langle \theta , x \rangle \quad \mathrm { s . t . } \quad \langle w _ { s } , x \rangle \leq c _ { s } , 0 \leq x \leq u _ { s } .
$$

A state $\boldsymbol { s } ~ = ~ ( w _ { s } , c _ { s } , u _ { s } )$ is generated by drawing the weights $w _ { s } \in \{ 1 , \ldots , 9 \} ^ { d }$ the capacity $c _ { s } \in \{ 1 , \dots , 1 9 \}$ , and the upper bounds on the numbers of items $u _ { s } \in \{ 1 , \ldots , 4 \} ^ { d }$ uniformly and independently for each component. The true weight $\theta ^ { * }$ is drawn from the Dirichlet distribution Dir(1), and the action $x ^ { n }$ of the agent is generated by decreasing each component of $x ^ { * } ( \theta ^ { * } , s ^ { n } )$ by 1 independently with probability 0.2 (without going below 0). Hence $x ^ { * } ( \theta ^ { * } , s ^ { n } ) \geq x ^ { n }$ , and $\theta ^ { * }$ satisfies Equation (3.2).

Solving outperformance inverse optimization in this problem means finding weights of the item values under which an optimal packing contains every item at least as many times as the agent’s packing.

The forward problem is solved exactly by dynamic programming. We run 50 episodes for each d $( \theta ^ { * }$ and the states are redrawn for each pair of an episode and $d )$

Single-machine scheduling problem. We consider deciding the order in which jobs with diferent release times are processed on a single machine according to their importance (e.g., sequencing tasks on a production line). The forward problem is single-machine scheduling with release times that minimizes the weighted sum of completion times $\textstyle \sum _ { j = 1 } ^ { d } \theta _ { j } C _ { j }$ . A state s consists of the release times $r _ { j } \in \{ 0 , \ldots , 1 4 \}$ and the processing times $p _ { j } \in \{ 1 , \ldots , 4 \}$ of the jobs $j = 1 , \dotsc , d$ (each drawn independently and uniformly), and, to match the maximization form, the action is the sign-reversed completion times $x = - ( C _ { 1 } , \ldots , C _ { d } )$ . Hence $x \geq x ^ { n }$ means that every job is completed no later than by the agent, that is, a schedule earlier than that of the agent. From the optimal schedule under the true weight $\theta ^ { * } \sim \mathrm { D i r } ( { \bf 1 } )$ , we generate the action $x ^ { n }$ of the agent by using thresholds $\tau _ { 1 } ^ { n } \in \{ 8 , \ldots , 1 2 \}$ and $\tau _ { 2 } ^ { n } \in \{ 1 8 , \dots , 2 2 \}$ drawn for each state, delaying by 1 every job whose completion time is at least $\tau _ { 1 } ^ { n }$ , and further delaying by another 1 every job whose completion time after this delay is at least $\tau _ { 2 } ^ { n }$ (the generation procedure is given in Algorithm 14). Hence $x ^ { * } ( \theta ^ { * } , s ^ { n } ) \geq x ^ { n }$ , and $\theta ^ { * }$ satisfies Equation (3.2).

Solving outperformance inverse optimization in this problem means finding weights of the job importance under which an optimal schedule completes every job no later than the agent’s schedule.

We set the number of states to $N = 1 0 , d \in \{ 4 , 6 , 8 \}$ , 25 episodes for each $d ,$ and $T = 1 0 0$ , that is, the oracle-call budget is $B = 2 0 0 0$ , and the forward problem is solved by SCIP.

Movie selection problem with viewing costs. We consider a user selecting movies to watch under a budget on the viewing cost and per-genre upper limits. We use the rating data (u1.base) of MovieLens100K (Harper and Konstan, 2015). The ground set consists of the top d movies in the number of ratings, and $\Gamma \in \{ 0 , 1 \} ^ { 1 8 \times d }$ is the genre indicator matrix of the movies (the rows correspond to the genres $g = 1 , \ldots , 1 8 )$ . A state s corresponds to one user, and $W _ { s }$ denotes the set of movies in the ground set rated by the user s. The forward problem is the movie selection under per-genre upper limits and a budget on the viewing cost,

$$
\operatorname* { m a x } _ { x \in \{ 0 , 1 \} ^ { d } } ~ \langle \theta , x \rangle \quad { \mathrm { s . t . } } \quad \langle w , x \rangle \leq b _ { s } , \ \Gamma x \leq { \bar { c } } _ { s } .
$$

The cost $w _ { i }$ is the number of years elapsed from the release of the movie i at the time of rating (the median over users, rounded to years and set to at least 1), and the budget and the per-genre upper limits are determined from the viewing history as $b _ { s } = \left. w , { \bf 1 } _ { W _ { s } } \right. + \operatorname* { m i n } _ { i } w _ { i }$ (where $\mathbf { 1 } _ { W _ { s } }$ is the indicator vector of $W _ { s } )$ and $\bar { c } _ { s , g } = ( \Gamma { \bf 1 } _ { W _ { s } } ) _ { g } + 1$ . The action of the agent is the indicator vector $x ^ { n } = \mathbf { 1 } _ { R _ { s } }$ of the set $R _ { s } \subseteq W _ { s }$ of movies to which the user s gave a rating of at least 4. The users with $| R _ { s } | \geq 5$ are divided into groups of 20 in the order of user IDs, each group forming one episode, and we run 20 episodes for each $d .$ No randomness is used.

Since $R _ { s } \subseteq W _ { s } , x ^ { n }$ is feasible. In this experiment, we do not specify a true weight and do not assume the existence of a weight satisfying Equation (3.2).

Solving outperformance inverse optimization in this problem means finding preference weights on the movies under which an optimal selection contains all the movies to which the user gave a rating of at least 4.

The forward problem is solved by SCIP.

Weighted multi-cover problem (Cora, ogbn-arxiv). We formulate the situation in which a researcher selects (reads or cites) a limited number of papers as a weighted multi-cover of words. For candidate papers $i \in P _ { s }$ and words $j = 1 , \ldots , d$ (the set $P _ { s }$ is described below), let $A _ { i j } \in \{ 0 , 1 \}$ indicate whether the word j appears in the title or the text of the paper $i ,$ and let the forward problem be

$$
\operatorname* { m a x } _ { \substack { z \in \{ 0 , 1 \} ^ { | P _ { s } | } , x \in \{ 0 , \ldots , r \} ^ { d } } } \langle \theta , x \rangle \quad \mathrm { s . t . } \quad x _ { j } \leq \sum _ { i \in P _ { s } } A _ { i j } z _ { i } , \quad \sum _ { i \in P _ { s } } z _ { i } \leq k _ { s }
$$

$( x \in \{ 0 , \ldots , r \} ^ { d } ;$ ; the oracle returns, as the action, the componentwise min $( r , A ^ { \top } z )$ computed from the selected papers $z ;$ that $\operatorname { i s } , \mathcal { X } ( s )$ is the set of all $x$ satisfying the above constraints for some $z )$ . The action $x _ { j }$ is “the number of papers covering the word $j ~ ( \mathrm { u p } ~ \mathrm { t o } ~ r ) ^ { \mathfrak { N } }$ , and $\theta$ is the weight of interest in the words. A state s corresponds to a paper $v ,$ the action of the agent is the depth $x _ { j } ^ { n } = \operatorname* { m i n } ( r , | \{ i \in E _ { s } : A _ { i j } = 1 \} | )$ of the set $E _ { s } \subseteq P _ { s }$ of papers in a citation relation with v (described below), and the upper bound on the number of papers is $k _ { s } = | E _ { s } |$ . Hence $x \geq x ^ { n }$ means “covering every word at a depth no smaller than that of the agent with the same number of papers as the agent”, and $E _ { s }$ itself is a feasible solution.

Solving outperformance inverse optimization in this problem means finding weights of interest in the words under which an optimal selection of papers covers every word at a depth no smaller than that of the agent with the same number of papers.

The words are the first d words obtained by taking, alternately in decreasing order of the field size, the words ranked by their per-field scores, so that the words of several fields compete for the limited number of papers even when the number of words is small. We impose the upper bound on the depth because, with $r = 1$ (maximum coverage), the OS loss tends to be 0 by construction under the uniform weight, and without the upper bound $( r \geq k _ { s } )$ the problem reduces to selecting the top $k _ { s }$ papers (§H.4). Note, however, that this upper bound is not efective in the states with $k _ { s } \leq r \ ( 3 8 \%$ for Cora and 35% for ogbn-arxiv). For Cora (Sen et al., 2008), the words are the 1433-word Bag-of-Words, the scores are the occurrence rates in each of the $7$ fields, the states are the papers v of degree $3 - 1 0 , \ E _ { s }$ is the set of adjacent papers in the undirected citation graph (the papers cited by v and the papers citing v), $P _ { s }$ is the 2-hop neighborhood of v excluding v itself, and $r = 3$ . For ogbn-arxiv (Hu et al., 2020), we use the direction of citations and the publication years; the states are the papers v of 2019 that cite 5–8 papers (published in 2019 or earlier), $E _ { s }$ is the set of papers actually cited by $v ,$ and $P _ { s }$ consists of $E _ { s }$ and the papers cited by them (all published in 2019 or earlier, excluding v itself). The words are constructed from the titles and abstracts (Wang et al., 2020), the scores are the distinctiveness in each of the 40 fields (the occurrence rate within the field minus the overall occurrence rate), and $r = 5$ . For ogbn-arxiv, the states of the episode e are selected from the field with the (e + 1)-th largest number of papers in 2019 among the fields with at least N papers satisfying the conditions $( e = 0 , \ldots , 1 9 )$ .

In both cases, we run 20 episodes for each d, and the forward problem is solved by SCIP. The generation procedures are given in Algorithms 18 and 19.

We describe the computing environment in §H.2, the implementation and the hyperparameters of each method in §H.3, the details of the instances of the weighted multi-cover problem in §H.4, the pseudocode of the evaluation protocol, of each method, and of the instance generation in §H.5 and §H.6, and the results of all the methods in the five experiments in §H.9.

H.2. Computing environment. All the experiments were run on a machine with Intel Xeon Gold 6444Y CPUs (16 cores, 2 sockets, 64 threads in total), 1 TB of memory, and Ubuntu 24.04.4 LTS. The implementation is in Python 3.11.15 (Van Rossum and Drake, 2009) with NumPy 2.4.3 and SciPy 1.17.1. As forward problems, the knapsack problem was solved by dynamic programming, and the MovieLens, Cora, ogbn-arxiv, and single-machine scheduling problems were solved by SCIP 10.0.0 (with SoPlex 8.0.0 as the LP solver) via OR-Tools 9.15 (Perron and Furnon, 2026). The time limit of SCIP was 60 seconds per solve for the MovieLens, Cora, and ogbn-arxiv problems, and 300 seconds for the single-machine scheduling problem. For the MovieLens, Cora, ogbn-arxiv, and single-machine scheduling problems, optimality was proved in every solve (the implementation stops with an error when optimality is not proved, and solutions without a proof of optimality are not used). For the single-machine scheduling problem, we observed the following two issues with SCIP, and therefore we first solved with presolve turned of and, when optimality was not proved, rebuilt the model and solved again with presolve turned on. First, with presolve turned on, the process terminates abnormally within presolve for certain inputs. Second, with presolve turned of, SCIP sometimes erroneously declares a feasible problem infeasible when the objective coeficients include extremely small values (of the order of 10<sup>−9</sup>). In either path, the output of the oracle is a solution whose optimality for the original weight is proved. Among the master problems of the inner solvers, the LPs were solved by HiGHS 1.12.0 in SciPy and the QPs by Clarabel 0.11.1. The metric projections of ONS and MetaGrad were also solved by Clarabel. MovieLens100K and Cora were loaded with PyTorch Geometric 2.8.0.post1 (Fey and Lenssen, 2019) (for Cora, the Planetoid version (Yang et al., 2016)). MovieLens100K was used in accordance with the terms of use of the GroupLens Research Project (use for research purposes only, no redistribution of the data, and citation of Harper and Konstan (2015) in publications). For Cora (the Planetoid version), we used the distributed files at https://github.com/kimiyoung/planetoid (MIT license). For ogbn-arxiv, we directly loaded the files distributed by OGB (Open Graph Benchmark) (Hu et al., 2020) (citations, publication years, fields, and the correspondence to the paper IDs of the Microsoft Academic Graph) and the raw titles and abstracts (Wang et al., 2020) also distributed by OGB. We used ogbn-arxiv as distributed under the ODC-BY (Open Data Commons Attribution License) license. Three CPU cores were allocated to the computation, and at most three processes were run simultaneously. The movie selection problem with viewing costs was run in three simultaneous processes by dividing the episodes into 0–7, 8–15, and 16–19, and the knapsack problem was run in a single process following the end of one of these processes. The single-machine scheduling problem was run in three simultaneous processes by dividing the episodes into 0–8, 9–16, and 17–24. The two experiments on the weighted multi-cover problem were run in three simultaneous processes by dividing the episodes (while three processes run simultaneously, each process uses one core). Each method was run sequentially within a single process, and the computation time was measured in wall-clock time. As the random number generator we used numpy.random.default\_rng; for the knapsack problem and the single-machine scheduling problem, a generator with the seed $( 2 0 2 6 0 9 2 8 , e , d )$ was created for each pair of an episode e and a dimension d to generate the instances. For the weighted multi-cover problem, the seed (20260928, e) was used to sample the states of the episode e. Since in both cases the seed is determined only by the episode and the dimension, the same instances are obtained even when a run is resumed from the middle.

H.3. Implementation and hyperparameters of the methods. All 13 methods are listed in Table 3. In addition to the 7 methods compared in the main text (§7.1), they include Sub-ONS and Sub-MetaGrad, which are online learning methods updating state by state, and variants of OS-DC with diferent inner solvers (OS-DC[CP]+SGF, OS-DC[LV+CPF]+SGF, OS-DC[ONS], and OS-DC[MetaGrad]). SGF (subgradient finishing) denotes applying the projected subgradient method with a small step size after DC optimization stops, starting from the obtained weight. It is added because, even at a weight at which the OS loss is $0 ,$ the optimal solution of the forward problem is in general not unique, and the solution of the oracle does not necessarily dominate the action. All hyperparameters were fixed in advance and were not tuned on the evaluation instances. Within each pair that difers only in the loss function, the OS side and the Sub side use the same values (the Sub side is given twice as many iterations so that the oracle-call budgets coincide).

Implementation of DC optimization. The implementation of DC optimization (Algorithms 5 and 6 in the appendix) is a DCA that truncates the inner loop at $J = 2 0$ iterations by default (inexact DCA). Its validity under truncation is ensured by the following design: (i) the descent property that holds even under truncation (Proposition E.11); (ii) tie stopping only when the inner solver reports convergence (converged = TRUE) (to avoid stopping at a tie that occurs by chance while the inner problem has not converged); (iii) cut inheritance, which improves the accuracy of the inner solver without additional oracle cost (Proposition E.3); and (iv) certificate stopping (stopping when the empirical mean of the OS loss falls below the threshold $\epsilon _ { \mathrm { g a p } } ;$ Proposition E.9) and target-attainment stopping (stopping when $\ell ^ { \mathrm { O P } }$ falls below the threshold), both of which are valid regardless of the convergence of the inner loop.

Projected subgradient method (Sub-PSG, OS-PSG). The update at iteration $j$ is

$$
\theta ^ { j + 1 } = \Pi _ { \Delta ^ { d - 1 } } \left( \theta ^ { j } - \eta ( j + 1 ) ^ { - 1 / 2 } \frac { g _ { j } } { \| g _ { j } \| _ { 2 } } \right) , \qquad g _ { j } = \sum _ { n = 1 } ^ { N } \bigl ( x ^ { * } ( \theta ^ { j } , s ^ { n } ) - y _ { j } ^ { n } \bigr ) .
$$

Here $\Pi _ { \Delta ^ { d - 1 } }$ is the Euclidean projection onto the probability simplex (an exact sorting-based method), and $y _ { j } ^ { n }$ is $x ^ { n }$ for Sub-PSG and $x ^ { * } ( \theta ^ { j } ; x ^ { n } , s ^ { n } )$ for OS-PSG $( g _ { j } / N$ of OS-PSG is the gradient of the empirical mean of the OS loss in the sense of weak derivatives; Proposition 5.1). The step-size coeficient is $\eta = 0 . 1$ . The method stops when $g _ { j } = 0$ . The same coeficient, normalization, and projection are used in all five experiments, including the single-machine scheduling problem. When $\langle \theta ^ { j } , x ^ { * } ( \theta ^ { j } , s ^ { n } ) - y _ { j } ^ { n } \rangle < 0$ holds because of solver errors, $x ^ { * } ( \theta ^ { j } , s ^ { n } )$ is replaced by $y _ { j } ^ { n }$ before computing $\ell ^ { \mathrm { O P } }$ (the solution before the replacement is used to compute $g _ { j } )$ The number of iterations is $T$ for OS-PSG and $2 T$ for Sub-PSG.

Table 3. The 13 compared methods. PSG is the projected subgradient method, ONS is the Online Newton Step (Hazan et al., 2007), and MetaGrad is van Erven and Koolen (2016); we also call these optimization methods learners. FB (full-batch) denotes the update that gives the learner the average of the subgradients over all states only once per iteration and does not use the averaged iterate (the average of the iterates so far). The online-to-batch conversion is the conversion that outputs the averaged iterate. The inner solvers CP, LV+CPF, and PB+CPF are the cutting-plane method, the level method with cutting-plane finishing, and the proximal bundle method with cutting-plane finishing, respectively (§D). In the main text, OS-DC[PB+CPF]+SGF is simply written as OS-DC.
<table><tr><td>Method</td><td>Optimization method</td><td>Pseudocode</td><td>Remarks</td></tr><tr><td colspan="4">Empirical mean of the suboptimality loss (existing)</td></tr><tr><td>Sub-PSG Sub-ONS</td><td>PSG ONS</td><td>Algorithm 8</td><td>state-by-state</td></tr><tr><td></td><td></td><td>Algorithm 9</td><td>update, online-to- batch conversion (Sakaue et al.,</td></tr><tr><td>Sub-MetaGrad</td><td>MetaGrad</td><td>Algorithm 9</td><td>2025b) state-by-state update, online-to- batch conversion (Sakaue et al., 2025b)</td></tr><tr><td>Sub-ONS(FB) Sub-MetaGrad(FB)</td><td>ONS</td><td>Algorithm 10</td><td>FB</td></tr><tr><td>Empirical mean of the OS loss (proposed)</td><td>MetaGrad</td><td>Algorithm 10</td><td>FB</td></tr><tr><td colspan="4"></td></tr><tr><td>OS-PSG</td><td>PSG</td><td>Algorithm 8</td><td></td></tr><tr><td>OS-ONS(FB)</td><td>ONS</td><td>Algorithm 10</td><td>FB</td></tr><tr><td>OS-MetaGrad(FB)</td><td>MetaGrad</td><td>Algorithm 10</td><td>FB</td></tr><tr><td>OS-DC[CP]+SGF</td><td>DC optimization Algorithm 12</td><td></td><td>inner: CP, with SGF</td></tr><tr><td>OS-DC[LV+CPF]+SGF</td><td>DC optimization Algorithm 12</td><td></td><td>inner: LV+CPF, with SGF</td></tr><tr><td>OS-DC[PB+CPF]+SGF DC optimization Algorithm 12</td><td></td><td></td><td>inner: PB+CPF, with SGF (OS-DC</td></tr><tr><td>OS-DC[ONS]</td><td>DC optimization Algorithm 12</td><td></td><td>in the main text) inner: ONS (FB; Algorithm 11)</td></tr><tr><td>OS-DC[MetaGrad]</td><td>DC optimization Algorithm 12</td><td></td><td>inner: Meta- Grad (FB; Algo-</td></tr></table>

Sub-ONS. One pass over the states $s ^ { 1 } , \ldots , s ^ { N }$ in order constitutes one epoch, and at each round we update by the Online Newton Step (Hazan et al., 2007)

$$
H _ { t } = H _ { t - 1 } + g _ { t } g _ { t } ^ { \top } , \qquad \theta ^ { t + 1 } = \Pi _ { \Delta ^ { d - 1 } } ^ { H _ { t } } \left( \theta ^ { t } - \gamma ^ { - 1 } H _ { t } ^ { - 1 } g _ { t } \right)
$$

using the subgradient $g _ { t } = x ^ { * } ( \theta ^ { t } , s ^ { n } ) - x ^ { n }$ of $\ell ^ { \mathrm { s u b } } ( \cdot ; x ^ { n } , s ^ { n } )$ . Here $\Pi _ { \Delta ^ { d - } } ^ { H }$ is the projection onto the probability simplex in the H-norm, $H _ { 0 } = \epsilon _ { \mathrm { r e g } } I$ , and $\gamma = 1$ $\epsilon _ { \mathrm { r e g } } = 1$ . At the end of each epoch, the average of the $\theta ^ { t }$ visited so far (online-tobatch conversion) is evaluated on all states and recorded. The number of oracle calls per epoch is 2N in total, N for the updates and N for the evaluation, so that T epochs use up the oracle-call budget $B = 2 T N$

Sub-MetaGrad. This method replaces the learner of Sub-ONS by MetaGrad (van Erven and Koolen, 2016). It runs in parallel an ONS for each learning rate $\eta _ { i } =$ $2 ^ { - i } / ( 5 D G ) \ ( i = 0 , \ldots , \lceil \log _ { 2 } ( T N + 1 ) / 2 \rceil )$ and aggregates them by tilted exponential weights. We use the diameter $D = { \sqrt { 2 } }$ of the probability simplex, and for the upper bound G on the norm of the subgradients we use the norm of the first nonzero (per-state) subgradient. The online-to-batch conversion and the counting of oracle calls are the same as for Sub-ONS.

Full-batch ONS and MetaGrad (Sub-ONS(FB), Sub-MetaGrad(FB), OS-ONS(FB), OS-MetaGrad(FB)). As in the projected subgradient method, at iteration $j$ all states are solved with the same weight $\theta ^ { j }$ , and the full-batch subgradient

$$
g _ { j } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl ( x ^ { * } ( \theta ^ { j } , s ^ { n } ) - y _ { j } ^ { n } \bigr )
$$

is given to the learner only once. Here $y _ { j } ^ { n }$ is $x ^ { n }$ on the Sub side and $x ^ { * } ( \theta ^ { j } ; x ^ { n } , s ^ { n } )$ on the OS side. ONS updates as

$$
H _ { j } = H _ { j - 1 } + g _ { j } g _ { j } ^ { \top } , \qquad \theta ^ { j + 1 } = \Pi _ { \Delta ^ { d - 1 } } ^ { H _ { j } } \left( \theta ^ { j } - \gamma ^ { - 1 } H _ { j } ^ { - 1 } g _ { j } \right)
$$

(with the initial metric $\epsilon _ { \mathrm { r e g } } I , \gamma = 1 , \epsilon _ { \mathrm { r e g } } = 1 )$ , and MetaGrad applies the same update as Sub-MetaGrad to $g _ { j } \ ( D = { \sqrt { 2 } } , G = \| g _ { 0 } \| _ { 2 } )$ . The number of learning rates of MetaGrad is set to $\lceil \log _ { 2 } ( n _ { \mathrm { i t } } + 1 ) / 2 \rceil + 1$ from the number of iterations $n _ { \mathrm { i t } }$ . The number of iterations is $n _ { \mathrm { { i t } } } = T$ on the OS side (2N oracle calls per iteration) and $n _ { \mathrm { i t } } = 2 T$ on the Sub side (N per iteration), and both use up exactly the oracle-call budget $B = 2 T N$ . The averaged iterate is not used; at each iteration, $\ell ^ { \mathrm { O P } }$ at $\theta ^ { j }$ and the empirical mean of the OS loss are computed directly from the solutions used for the update and recorded $( x ^ { * } ( \theta ^ { j } ; x ^ { n } , s ^ { n } )$ used for recording the OS loss on the Sub side is a solve used only for logging and is not counted as an oracle call). Stopping when $g _ { j } = 0$ is the same as in the projected subgradient method. The replacement against solver errors is used not only for $\ell ^ { \mathrm { O P } }$ but also for computing $g _ { j }$ These methods difer from Sub-ONS and Sub-MetaGrad in that each update uses all states and in that the averaged iterate is not used.

DC optimization of the OS loss (OS-DC). The maximum number of outer iterations is $K = 2 T / 2 0$ , the maximum number of inner iterations is $J = 2 0$ , the threshold of tie stopping is $\epsilon _ { \mathrm { t i e } } = 1 0 ^ { - 8 }$ , and the threshold of certificate stopping is $\epsilon _ { \mathrm { g a p } } = 1 0 ^ { - 6 }$ . As the rules for selecting the evaluation points of the inner solver (§D), we compare the cutting-plane method (CP), the level method with cutting-plane finishing (LV+CPF), and the proximal bundle method with cutting-plane finishing (PB+CPF), and subgradient finishing is added to all of them. In addition, we also compare OS-DC with ONS or MetaGrad as the inner solver (without subgradient finishing) (next paragraph). For CP, the inner tolerance is $\epsilon _ { \mathrm { i n } } = 1 0 ^ { - 6 }$ ; for LV+CPF and PB+CPF, $\epsilon _ { \mathrm { i n } } = 0$ , the threshold for switching phases is $1 0 ^ { - 6 }$ , and the termination tolerance is $\epsilon _ { \mathrm { t e r m } } = 1 0 ^ { - 9 }$ (Algorithm 4). The exactness of Theorem F.3 is guaranteed for $\epsilon _ { \mathrm { t e r m } } =$ 0; the positive value is a safeguard against rounding errors. In subgradient finishing (SGF), OS-PSG is run for $T$ iterations from the output of DC optimization with the step-size coeficient $\eta = 1 0 ^ { - 3 }$ (1/100 of that of the usual projected subgradient method) in all five experiments. The step size is made small because a large move makes the OS loss positive again. At every evaluation of the forward-problem oracle (including the inner evaluation points), $\ell ^ { \mathrm { { \scriptsize { O P } } } }$ is computed, and the whole procedure stops when it becomes 0. The proximal weight of the proximal bundle method is fixed at $u = 1$ , the descent parameter is $\rho = 0 . 1$ , and the level parameter of the level method is $\lambda = 1 / ( 2 + \sqrt { 2 } )$ . The output of DC optimization is the attaining point in the case of target-attainment stopping and, otherwise, the point with the smallest $\ell _ { k }$ among all outer iterations (Algorithm 6). The whole procedure is given in Algorithm 12. The evaluation points computed by the inner solver along the way (the solutions of the LPs and QPs of the master problems) are passed to the oracle without being projected onto the probability simplex, and only the output of the inner solver is projected onto the probability simplex. Hence the intermediate evaluation points may lie outside the probability simplex within the tolerance of the LP and QP solvers. The weights reported as best-so-far may also lie outside the probability simplex within the same tolerance. OS-DC in the main text is OS-DC[PB+CPF]+SGF.

OS-DC with ONS or MetaGrad as the inner solver (OS-DC[ONS], OS-DC[MetaGrad]). This is OS-DC that solves the convex subproblem $\mathrm { m i n } _ { \theta \in \Theta } \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } )$ (Equation (5.1)) of a branch $y ^ { \prime }$ by ONS or MetaGrad. The convex subproblem $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ is exactly the empirical mean of the suboptimality loss in which the actions of the agent are replaced by the dominating vertices $y ^ { \prime n }$ . The outer loop (Algorithm 6) and $K , \epsilon _ { \mathrm { t i e } } .$ , and $\epsilon _ { \mathrm { g a p } }$ are the same as for OS-DC, and only the inner loop is replaced as follows. At each inner iteration $t , \ x ^ { * } ( \theta ^ { t } , s ^ { n } )$ is computed for all states (N oracle calls), and, using the full-batch subgradient $\begin{array} { r } { g _ { t } \ = \ \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl ( x ^ { * } ( \theta ^ { t } , s ^ { n } ) \ - \ y ^ { \prime n } \bigr ) } \end{array}$ , OS-DC[ONS] performs the same update as Sub-ONS,

$$
H _ { t } = H _ { t - 1 } + g _ { t } g _ { t } ^ { \top } , \qquad \theta ^ { t + 1 } = \Pi _ { \Delta ^ { d - 1 } } ^ { H _ { t } } \left( \theta ^ { t } - \gamma ^ { - 1 } H _ { t } ^ { - 1 } g _ { t } \right)
$$

$( H _ { 0 } = \epsilon _ { \mathrm { r e g } } I , \gamma = 1 , \epsilon _ { \mathrm { r e g } } = 1 )$ . OS-DC[MetaGrad] replaces this update by the same MetaGrad as Sub-MetaGrad $( D = { \sqrt { 2 } } .$ and G is the norm of the first subgradient). The number of learning rates is set to $\lceil \log _ { 2 } ( K J + 1 ) / 2 \rceil + 1$ from the total number of rounds KJ of the whole DC optimization. Within each outer iteration, the inner loop outputs the point with the smallest $\bar { \ell } ^ { \mathrm { s u b } } ( \cdot ; y ^ { \prime } )$ among the visited $\theta ^ { t }$ and their average (online-to-batch conversion). The internal state of the learner is carried over across outer iterations. That is, the first outer iteration starts from $\theta ^ { 0 } = { \bf 1 } / d \cdot$ and each subsequent one resumes from the state at the end of the previous outer iteration. What is carried over is the metric $H _ { t }$ and the last iterate for ONS, and, for MetaGrad, the point and the metric of each learning rate, the aggregation weights, $G ,$ and the learning rates. On the other hand, the cumulative sum for the average is reset at each outer iteration, because the guarantee of the online-to-batch conversion holds only for a fixed loss, and the convex subproblem changes when the branch changes. The maximum number of inner iterations is $J = 2 0$ and the tolerance is $\epsilon _ { \mathrm { i n } } = 1 0 ^ { - 6 } \ ;$ ; the inner loop stops when $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { t } ; y ^ { \prime } ) \le \epsilon _ { \mathrm { i n } }$ . Since ONS and MetaGrad have no cutting-plane model, no lower bound LB is available, and cut inheritance is not used. The inner loop reports converged = TRUE only when $\bar { \ell } ^ { \mathrm { s u b } } ( \theta ^ { t } ; y ^ { \prime } ) \le \epsilon _ { \mathrm { i n } }$ or when the subgradient becomes 0. Neither subgradient finishing nor cutting-plane finishing is used. Even when the branch changes, the metric built from the subgradients of the previous branch is carried over without correction. Hence the whole procedure takes the form of “running a single ONS (MetaGrad) while updating the linearization point along the way”.

Remark H.1 (Replacement against solver errors). In the projected subgradient method, subgradient finishing, and the full-batch ONS and MetaGrad, when $\langle \theta , x ^ { * } ( \theta , s ^ { n } ) - y ^ { n } \rangle \ < \ 0 , \ x ^ { * } ( \theta , s ^ { n } )$ is replaced by $y ^ { n } \ \left( x ^ { n } \right.$ on the Sub side and $x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ on the OS side) before computing $\mathcal { \bar { l } } ^ { \mathrm { O P } }$ (in the projected subgradient method and subgradient finishing, the solution before the replacement is used for the subgradient $g _ { j } ,$ , whereas in the full-batch ONS and MetaGrad the solution after the replacement is used). On the other hand, in Sub-ONS, Sub-MetaGrad, and the main body of DC optimization, $\ell ^ { \mathrm { O P } }$ is computed from the solution returned by the oracle as it is. However, since every $y ^ { n }$ is a feasible solution in $\mathcal X ( s ^ { n } )$ $\langle \theta , x ^ { * } ( \theta , s ^ { n } ) \rangle \geq \langle \theta , y ^ { n } \rangle$ always holds if $x ^ { * } ( \theta , s ^ { n } )$ is an exact optimal solution, and this replacement does not occur. In our experiments, the knapsack problem is solved exactly by dynamic programming, and the MovieLens, Cora, ogbn-arxiv, and single-machine scheduling problems stop with an error when optimality is not proved (§H.2); hence this replacement does not occur. Therefore, the diference in the treatment among the methods does not afect the results.

Reuse of the models of the forward problems. For the MovieLens, Cora, and ogbnarxiv problems (for the single-machine scheduling problem the model is rebuilt at every solve), the model of the forward problem for the same state and the same dominance constraint is kept, and only the objective coeficients are replaced for re-solving. Since which optimal solution is returned when the optimal solution is not unique may depend on the preceding solve history, the kept models are reset before running each method, so that the trajectory of a method does not depend on the order in which the other methods are run.

## H.4. Details of the instances of the weighted multi-cover problem.

Words. For Cora, the 1433-word Bag-of-Words (presence or absence) is used as it is. For ogbn-arxiv, the titles and abstracts are lowercased, and we use the words consisting only of alphabetic characters with at least three letters, excluding common English stop words (no stemming). Words that appear in fewer than 20 papers are discarded. For each field c, the score of a word j is the fraction of the papers of the field c containing the word j (occurrence rate) for Cora, and the occurrence rate minus the occurrence rate over all papers (distinctiveness) for ogbn-arxiv; the ranking of the words is constructed by taking one word at a time alternately from the per-field rankings of the scores, starting from the field with the largest number of papers (skipping words already taken). The words of dimension d are the first d words of this ranking, so that the sets of words are nested as d increases. We use the distinctiveness rather than the occurrence rate for ogbn-arxiv because with the occurrence rate, general words independent of the fields, such as based, paper, show, and model, occupy the top ranks.

States. The selection of the states does not depend on the dimension, and 20 papers are sampled at random for each episode (since the episode number is included in the seed, the same states are used in the same episode even when the dimension is changed). For Cora, the number of candidates has median 25 (3–210), and the number of papers selected by the agent has median 4 (3–10). For ogbn-arxiv, an edge $( u , w )$ represents that u cites w (98.2% of the edges have a citing paper whose publication year is no earlier than that of the cited paper). By the condition on the publication year (2019 or earlier), $E _ { s }$ and $P _ { s }$ consist only of papers that existed at the time v was written. The number of candidates has median 38 (6–211), and the number of papers selected by the agent has median 6 (5–8). The fields of the episodes $0 , 1 , \ldots , 1 9$ are, in decreasing order of the number of papers in 2019, cs.LG, cs.CV, cs.CL, cs.IT, cs.RO, . . . .

Degeneracy at the uniform weight. In a state in which the OS loss is 0 at the uniform weight $\theta ^ { 0 } = { \bf 1 } / d$ , the barycenter of the probability simplex is itself a global minimizer. The upper bound $k _ { s }$ on the number of papers and the upper bound r on the depth were chosen so that this degeneracy is reduced. The fraction of the states at which the OS loss is positive at the uniform weight (60 states of the episodes 0–2) is 57% for $d = 2 0$ and 88% for $d = 4 0$ for Cora $( r = 3 )$ , and 33% for $d = 2 0$ and 82% for $d = 4 0$ for ogbn-arxiv $( r = 5 ) ;$ for $d = 1 0 0 .$ , it was positive at all 20 states of the episode 0 in both experiments. Setting the upper bound on the number of papers to $k _ { s } = | E _ { s } | + 1$ increases the degeneracy (because the extra paper can recover the words of the agent), so we set $k _ { s } = | E _ { s } |$ . In the measured range, the states at which the OS loss is positive under the uniform weight account for at least 80% for $d = 4 0$ and for all states for $d = 1 0 0$ (episode 0).

H.5. Pseudocode of the evaluation protocol and of each method. This subsection gives the evaluation protocol as implemented and the pseudocode of each method. In what follows, “evaluating” a weight θ means solving $v ^ { n } \gets$ $x ^ { * } ( \theta , s ^ { n } )$ for all states (N oracle calls) and computing and recording $\ell ^ { \mathrm { O P } } ( \theta ) =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { d } } \end{array}$ max $\{ x _ { i } ^ { n } - v _ { i } ^ { n } , 0 \}$ (Equation (E.6)). In every method, if $\ell ^ { \mathrm { O P } } \leq 0$ at an evaluation, the whole method stops at that point and returns the weight evaluated then (target-attainment stopping). The solves $\tilde { y } ^ { n } \gets x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ used only for logging (for recording $\bar { \ell } ^ { \mathrm { O S } } )$ are not counted as oracle calls. Each method is evaluated at the weight with the smallest $\ell ^ { \mathrm { O P } }$ among the weights visited so far (for Sub-ONS and Sub-MetaGrad, the averaged iterates at the end of each epoch (one pass over the states); for OS-DC, all evaluation points of the inner loop, the outer loop, and subgradient finishing) (best-so-far; among weights with the same $\ell ^ { \mathrm { O P } }$ , the one with the smaller empirical mean of the OS loss), and the empirical mean of the OS loss is also measured at the same weight. Letting $T$ be the number of iterations of the projected subgradient method and N the number of states, we impose the oracle-call budget $B : = 2 T N$ common to all methods and compare the values at the time when the number of oracle calls reaches B. OS-PSG, OS-ONS(FB), and OS-MetaGrad(FB) make 2N oracle calls per iteration and are run for T iterations, and Sub-PSG, Sub-ONS(FB), and Sub-MetaGrad(FB) make N oracle calls per iteration and are run for 2T iterations. Sub-ONS and Sub-MetaGrad make 2N oracle calls per epoch (one pass over the states), N for the updates and N for the evaluation of the averaged iterate, and are run for T epochs. The number of oracle calls of OS-DC difers from one outer iteration to another, and it may be run beyond the oracle-call budget B, but the values at the time when the number of oracle calls reaches B are used for the comparison. All methods stop as target attainment when $\ell ^ { \mathrm { O P } } = 0$ is reached.

Algorithm 7 Common evaluation protocol (oracle-call budget B and best-so-far)   
Input: method, oracle-call budget $B = 2 T N$ , attainment tolerance $\epsilon _ { \mathrm { a t t } } = 1 0 ^ { - 9 }$   
1: Run the method and obtain the sequence of records $( c _ { i } , \ell _ { i } ^ { \mathrm { O P } } , \bar { \ell } _ { i } ^ { \mathrm { O S } } ) _ { i = 0 , 1 , \dots } ~ ( c _ { i }$ is   
the cumulative number of oracle calls at the time of recording: the number of   
calls to $x ^ { * } ( \cdot , s )$ for the 5 existing methods, and the sum of the numbers of calls   
to $x ^ { * } ( \cdot , s )$ and $x ^ { * } ( \cdot ; x ^ { \prime } , s )$ for the 8 proposed methods)   
2: for $i = 0 , 1 , \ldots$ do   
3: $( L _ { i } , O _ { i } ) $ the lexicographic minimum of $( \ell _ { i ^ { \prime } } ^ { \mathrm { O P } } , \bar { \ell } _ { i ^ { \prime } } ^ { \mathrm { O S } } ) _ { i ^ { \prime } \leq i }$ (an undefined $\bar { \ell } ^ { \mathrm { O S } }$ is   
regarded as $+ \infty ,$ and ties keep the earlier record)   
4: end for   
5: i ← max $\{ i \mid c _ { i } \leq B \}$   
6: return $L _ { i _ { B } } , O _ { i _ { B } }$ , attainment $[ L _ { i _ { B } } \leq \epsilon _ { \mathrm { a t t } } ]$ , and computation time (the maximum   
wall-clock time from the start of the method)

Algorithm 8 Projected subgradient method (Sub-PSG, OS-PSG, subgradient   
finishing)   
Input: initial point $\theta ^ { 0 }$ , step-size coeficient $\eta ,$ number of iterations $n _ { \mathrm { i t } }$ , type $\in$   
{Sub, OS}   
1: for $j = 0 , 1 , \ldots , n _ { \mathrm { i t } } - 1$ do   
2: For each $n , v ^ { n }  x ^ { * } ( \theta ^ { j } , s ^ { n } )$ and $\tilde { y } ^ { n } \gets x ^ { * } ( \theta ^ { j } ; x ^ { n } , s ^ { n } )$ (counted for OS; used   
only for logging for Sub)   
3: For each n, y<sup>n</sup> ← y˜<sup>n</sup> (OS) or $y ^ { n } \gets x ^ { n }$ (Sub)   
4: $\begin{array} { r } { \bar { \ell } _ { j } ^ { \mathrm { O S } }  \frac { 1 } { N } \sum _ { n = 1 } ^ { N } } \end{array}$ max $\{ \langle { \theta ^ { j } , v ^ { n } - \tilde { y } ^ { n } } \rangle , 0 \}$   
5: For each $n , \hat { v } ^ { n }  y ^ { n } \mathrm { ~ i f ~ } \langle \theta ^ { j } , v ^ { n } - y ^ { n } \rangle < 0 ,$ and $\hat { v } ^ { n } \gets v ^ { n }$ otherwise   
6: Record $\begin{array} { r } { \ell _ { j } ^ { \mathrm { O P } } \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { i = 1 } ^ { d } } \end{array}$ max $\{ x _ { i } ^ { n } - \hat { v } _ { i } ^ { n } , 0 \}$   
7: if $\ell _ { j } ^ { \mathrm { { O P } } } \leq 0$ then   
8: return $\theta ^ { j }$   
9: end if   
10: $\begin{array} { r } { g _ { j }  \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { n } ) } \end{array}$ (using $v ^ { n }$ before the replacement)   
11: if $g _ { j } = 0$ then   
12: return $\theta ^ { j }$   
13: end if   
14: $\theta ^ { j + 1 }  \Pi _ { \Delta ^ { d - 1 } } \big ( \theta ^ { j } - \eta ( j + 1 ) ^ { - 1 / 2 } g _ { j } / \| g _ { j } \| _ { 2 } \big )$   
15: end for   
16: return $\theta ^ { n _ { \mathrm { i t } } - 1 }$ (the last evaluated point)

In Algorithm $8 , ( \theta ^ { 0 } , \eta , n _ { \mathrm { i t } } )$ is $( 1 / d , 0 . 1 , 2 T )$ for Sub-PSG, $( \mathbf { 1 } / d , 0 . 1 , T )$ for OS-PSG, and (the output of DC optimization, $1 0 ^ { - 3 } , T )$ for subgradient finishing (type OS). In subgradient finishing, no stopping based on $\bar { \ell } ^ { \mathrm { O S } }$ is performed, and the decay $( j + 1 ) ^ { - 1 / 2 }$ of the step size is counted again from $j = 0$ . These parameters are common to all five experiments, including the single-machine scheduling problem.

In what follows, let $\mathcal { H } _ { d }$ be the set of $d \times$ d symmetric positive definite matrices, and for $H \in \mathcal { H } _ { d }$ and $z \in \mathbb { R } ^ { d }$ , let $\begin{array} { r } { \Pi _ { \Delta ^ { d - 1 } } ^ { H } ( z ) : = \arg \operatorname* { m i n } _ { \theta \in \Delta ^ { d - 1 } } ( \theta - z ) ^ { \top } H ( \theta - z ) } \end{array}$ (the minimizer is unique since the objective function is strictly convex; it is solved as a quadratic program by Clarabel). An online learner is given by a set L of internal states together with an initialization I : $\Delta ^ { d - 1 } \to \mathbb { L } ,$ a prediction $\mathsf { P } \colon \mathbb { L } \to \Delta ^ { d - 1 }$ , and an update $\mathsf { U } \colon  { \mathbb { L } } \times (  { \mathbb { R } } ^ { d } \setminus \{ 0 \} ) \to  { \mathbb { L } }$

Definition H.2 (ONS). Let $\gamma = 1$ and $\epsilon _ { \mathrm { r e g } } = 1$ . The set of internal states is $\mathbb { L } _ { \mathrm { O N S } } : = \Delta ^ { d - 1 } \times \mathcal { H } _ { d }$ , and for $\mathcal { L } = ( \theta , H ) \in \mathbb { L } _ { \mathrm { O N S } }$ we define

$$
\begin{array} { r l } & { \mathsf { I } ( \theta ^ { \mathrm { i n i t } } ) : = ( \theta ^ { \mathrm { i n i t } } , \epsilon _ { \mathrm { r e g } } I ) , \qquad \mathsf { P } ( \mathcal { L } ) : = \theta , } \\ & { \mathsf { U } ( \mathcal { L } , g ) : = \big ( \Pi _ { \Delta ^ { d - 1 } } ^ { H ^ { \prime } } \big ( \theta - \gamma ^ { - 1 } ( H ^ { \prime } ) ^ { - 1 } g \big ) , H ^ { \prime } \big ) , \qquad H ^ { \prime } : = H + g g ^ { \top } . } \end{array}
$$

Definition H.3 (MetaGrad). Let the diameter be $D = { \sqrt { 2 } }$ and the total number of rounds be $n _ { \mathrm { r } } .$ , and put $n _ { \eta } : = \lceil \log _ { 2 } ( n _ { \mathrm { r } } + 1 ) / 2 \rceil + 1$ . The set of internal states is

$$
\mathbb { L } _ { \mathrm { M G } } : = \left( \mathbb { R } _ { > 0 } ^ { n _ { \eta } } \cup \{ \perp \} \right) \times \Delta ^ { n _ { \eta } - 1 } \times ( \Delta ^ { d - 1 } ) ^ { n _ { \eta } } \times ( \mathcal { H } _ { d } ) ^ { n _ { \eta } } ,
$$

whose elements we write as $\mathcal { L } = ( \eta , \pi , ( w ^ { i } ) _ { i = 0 } ^ { n _ { \eta } - 1 } , ( H ^ { i } ) _ { i = 0 } ^ { n _ { \eta } - 1 } )$ . Here $\perp$ is a symbol indicating that the learning rates η are not yet set; they are determined from the norm of the first subgradient at the first update. We define the initialization and the prediction by

$$
1 ( \theta ^ { \mathrm { i n i t } } ) : = \big ( \perp , \pi ^ { \mathrm { i n i t } } , ( \theta ^ { \mathrm { i n i t } } ) _ { i } , ( I / D ^ { 2 } ) _ { i } \big ) , \qquad \pi _ { i } ^ { \mathrm { i n i t } } : = \frac { n _ { \eta } + 1 } { n _ { \eta } ( i + 1 ) ( i + 2 ) } ,
$$

$$
\mathsf { P } ( \mathcal { L } ) : = \left\{ \begin{array} { l l } { w ^ { 0 } } & { ( \eta = \perp ) , } \\ { \displaystyle \frac { \sum _ { i } \pi _ { i } \eta _ { i } w ^ { i } } { \sum _ { i } \pi _ { i } \eta _ { i } } } & { ( \mathrm { o t h e r w i s e } ) . } \end{array} \right.
$$

The update $\mathsf { U } ( \mathcal { L } , g ) = ( \eta ^ { \prime } , \pi ^ { \prime } , ( w ^ { i \prime } ) _ { i } , ( H ^ { i \prime } ) _ { i } )$ is defined, with $\theta : = \mathsf { P } ( \mathcal { L } )$ , by

$$
\eta ^ { \prime } : = \left\{ \begin{array} { l l } { \left( 2 ^ { - i } / ( 5 D \| g \| _ { 2 } ) \right) _ { i = 0 } ^ { n _ { \eta } - 1 } } & { ( \eta = \bot ) , } \\ { \eta } & { ( \mathrm { o t h e r w i s e } ) , } \end{array} \right.
$$

$$
\begin{array} { r } { \varsigma _ { i } : = \langle w ^ { i } - \theta , g \rangle , \qquad H ^ { i \prime } : = H ^ { i } + 2 ( \eta _ { i } ^ { \prime } ) ^ { 2 } g g ^ { \top } , } \end{array}
$$

$$
w ^ { i \prime } : = \Pi _ { \Delta ^ { d - 1 } } ^ { H ^ { i \prime } } \big ( w ^ { i } - \eta _ { i } ^ { \prime } ( 1 + 2 \eta _ { i } ^ { \prime } \varsigma _ { i } ) ( H ^ { i \prime } ) ^ { - 1 } g \big ) ,
$$

$$
\pi _ { i } ^ { \prime } : = \frac { \pi _ { i } \exp ( - \eta _ { i } ^ { \prime } \varsigma _ { i } - ( \eta _ { i } ^ { \prime } \varsigma _ { i } ) ^ { 2 } ) } { \sum _ { j } \pi _ { j } \exp ( - \eta _ { j } ^ { \prime } \varsigma _ { j } - ( \eta _ { j } ^ { \prime } \varsigma _ { j } ) ^ { 2 } ) } .
$$

Since $w ^ { i } = \theta ^ { \mathrm { i n i t } }$ for all i in $1 ( \theta ^ { \mathrm { i n i t } } )$ , the prediction right after $\eta$ is set equals $w ^ { 0 }$ also by the general formula, so the case distinction in P causes no discontinuous change of the prediction.

In OS-DC, $\bar { \ell } ^ { \mathrm { O S } }$ is recorded at all evaluation points of the inner and outer loops by solves used only for logging. The pool V of inherited cuts is shared within a single run of DC optimization.

H.6. Pseudocode for instance generation. For each experiment, an instance is generated once for each pair of an episode and a dimension $d ,$ and the same $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$ is given to all methods. All random numbers in the following pseudocode are drawn in order from a single generator of numpy.random.default\_rng.

Algorithm 9 Sub-ONS and Sub-MetaGrad (state-by-state update and online-to  
batch conversion)   
Input: learner (I, P, U) (ONS: Definition H.2, MetaGrad: Definition H.3; $n _ { \mathrm { r } } = T N$   
for MetaGrad), $\theta ^ { \mathrm { i n i t } } = { \bf 1 } / d ,$ number of epochs $T$   
1: $\mathcal { L } \gets \mathsf { I } ( \theta ^ { \mathrm { i n i t } } ) , \mathcal { S } \gets 0 , t \gets 0$   
2: for $e = 0 , 1 , \ldots , T - 1$ do   
3: for $n = 1 , \ldots , N$ do   
4: $\theta ^ { t } \gets \mathsf { P } ( \mathcal { L } ) , S \gets S + \theta ^ { t } , t \gets t + 1$   
5: $g _ { t } \gets x ^ { * } ( \theta ^ { t } , s ^ { n } ) - x ^ { n }$ (one oracle call)   
6: if $g _ { t } \neq 0$ then   
7: $\mathcal { L } \gets \mathsf { U } ( \mathcal { L } , g _ { t } )$   
8: end if   
9: end for   
10: Evaluate $\bar { \theta } ^ { e }  S / t$ (N oracle calls), and record $\bar { \ell } ^ { \mathrm { O S } }$ from $\tilde { y } ^ { n } \gets x ^ { * } ( \bar { \theta } ^ { e } ; x ^ { n } , s ^ { n } )$   
(used only for logging)   
11: end for   
12: return $\bar { \theta } ^ { T - 1 }$ (the last evaluated point)

Algorithm 10 Full-batch ONS and MetaGrad (Sub-ONS(FB), Sub-MetaGrad(FB),   
OS-ONS(FB), OS-MetaGrad(FB))   
Input: learner (I, P, U) (ONS: Definition H.2, MetaGrad: Definition H.3; $n _ { \mathrm { r } } = n _ { \mathrm { i t } }$   
for MetaGrad), ${ \theta } ^ { \mathrm { i n i t } } = { \bf 1 } / d ,$ number of iterations $n _ { \mathrm { i t } }$ (OS: T, Sub: 2T)   
1: ${ \mathcal { L } } \gets | ( \theta ^ { \mathrm { { i n i t } } } )$   
2: for $j = 0 , 1 , \ldots , n _ { \mathrm { i t } } - 1$ do   
3: $\theta ^ { j } \gets \mathsf { P } ( \mathcal { L } )$   
4: For each $n , v ^ { n }  x ^ { * } ( \theta ^ { j } , s ^ { n } )$ and $\tilde { y } ^ { n } \gets x ^ { * } ( \theta ^ { j } ; x ^ { n } , s ^ { n } )$ (counted for OS; used   
only for logging for Sub)   
5: For each $n ,$ let $y ^ { n }  \tilde { y } ^ { n }$ (OS) or $y ^ { n }  x ^ { n }$ (Sub), and set $v ^ { n }  y ^ { n }$ if   
$\langle \theta ^ { j } , v ^ { n } - y ^ { n } \rangle < 0$   
6: Record $\ell _ { j } ^ { \mathrm { { O P } } }$ and $\bar { \ell } _ { j } ^ { \mathrm { O S } }$ with $v ^ { n }$ after the replacement   
7: if $\ell _ { j } ^ { \mathrm { { O P } } } \leq 0$ then   
8: return $\theta ^ { j }$   
9: end if   
10: $\begin{array} { r } { g _ { j }  \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( v ^ { n } - y ^ { n } ) } \end{array}$   
11: if $g _ { j } = 0$ then   
12: return $\theta ^ { j }$   
13: end if   
14: $\mathcal { L } \gets \mathsf { U } ( \mathcal { L } , g _ { j } )$   
15: end for   
16: return $\theta ^ { n _ { \mathrm { i t } } - 1 }$ (the last evaluated point)   
Its seed is (20260928, e, d) for the knapsack problem and the single-machine schedul  
ing problem (Algorithms 13 and 14) and (20260928, e) for the weighted multi-cover   
problem (Algorithms 18 and 19), where e is the episode index (§H.2).   
Knapsack problem. For a state $s = ( w , c , u )$ , we have $x ^ { * } ( \theta , s ) \in$ arg max $\{ \langle \theta , x \rangle \mid x \in$   
$\mathbb { Z } ^ { d } , \ \langle w , x \rangle \leq c , \ 0 \leq x \leq u \}$ and $x ^ { \ast } ( \theta ; x ^ { n } , s ) \in$ arg max $\{ \langle \theta , x \rangle \mid x \in \mathbb { Z } ^ { d } , \langle w , x \rangle \leq$

Algorithm 11 Inner loop of OS-DC[ONS] and OS-DC[MetaGrad] (replacing line 3   
of Algorithm 6)   
Input: branch $y ^ { \prime } .$ , outer point $\theta ^ { k }$ , learner (I, P, U) (ONS: Definition H.2, MetaGrad:   
Definition H.3; $D = { \sqrt { 2 } }$ and $n _ { \mathrm { r } } = K J$ for MetaGrad), internal state of the   
learner $\mathcal { L }$ (undefined for $k = 0 )$ $\epsilon _ { \mathrm { i n } } = 1 0 ^ { - 6 }$ $J = 2 0$ , feasibility tolerance   
2   
$\epsilon _ { \mathrm { f e a s } } = 1 0 ^ { - 9 }$   
1: $\begin{array} { r } { f ( \theta ) : = \bar { \ell } ^ { \mathrm { s u b } } ( \theta ; y ^ { \prime } ) , g ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( x ^ { * } ( \theta , s ^ { n } ) - y ^ { \prime n } ) } \end{array}$ (one evaluation is shared   
for the same $\theta )$   
2: $f ^ { \star } \gets + \infty$ , converged $ \mathrm { F A L S E }$   
3: if L is undefined then   
4: if the learner is ONS then   
5: ${ \mathcal { L } } \gets | ( \theta ^ { k } )$   
6: else if the learner is MetaGrad then   
7: Evaluate $\theta ^ { k }$ (N oracle calls), and set $f ^ { \star } \gets f ( \theta ^ { k } ) , \hat { \theta } \gets \theta ^ { k }$ , and $G  \| g ( \theta ^ { k } ) \| _ { 2 }$   
8: if $G = 0$ then   
9: return (<sup>ˆ</sup>θ, TRUE, L) (L remains undefined)   
10: end if   
11: Set ${ \mathcal { L } } \gets | ( \theta ^ { k } )$ and replace the η component of L by $\left( 2 ^ { - i } / ( 5 D G ) \right) _ { i = 0 } ^ { n _ { \eta } - 1 }$   
12: end if   
13: else   
14: Use the input $\mathcal { L }$ as it is $( \theta ^ { k }$ is not used)   
15: end if   
16: $\Sigma  0$   
17: for $t = 1 , \dots , J$ do   
18: Evaluate $\theta ^ { t } \gets \mathsf { P } ( \mathcal { L } )$ (N oracle calls; at $t = 1$ of the first outer iteration of   
MetaGrad, the prediction is mathematically equal to $\theta ^ { k }$ , but it is evaluated   
again if the bit patterns do not coincide because of floating-point rounding)   
19: If $f ( \theta ^ { t } ) < f ^ { \star }$ , then $f ^ { \star } \gets f ( \theta ^ { t } )$ and $\hat { \theta } \gets \theta ^ { t }$ $\Sigma  \Sigma + \theta ^ { t }$   
20: if $f ^ { \star } \leq \epsilon _ { \mathrm { i n } }$ or $g ( \theta ^ { t } ) = 0$ then   
21: converged ← TRUE, break   
22: end if   
23: $\mathcal { L } \gets \mathsf { U } ( \mathcal { L } , g ( \theta ^ { t } ) )$   
24: end for   
25: if converged = FALSE and ${ \bar { \theta } } : = \Sigma / t \in \Theta$ (tolerance $\epsilon _ { \mathrm { f e a s } } )$ then   
26: Evaluate <sup>¯</sup>θ (N oracle calls), and set $\hat { \theta }  \bar { \theta }$ if $f ( \bar { \theta } ) < f ^ { \star }$   
27: end if   
28: return $( { \hat { \theta } } ,$ converged, L)

c, $x ^ { n } \leq x \leq u \}$ , both of which are solved exactly by dynamic programming for the bounded knapsack problem (for the latter, we solve the problem in $x - x ^ { n }$ with capacity $c - \langle w , x ^ { n } \rangle$ and upper bound $u - x ^ { n } )$

Since $0 \leq x ^ { n } \leq x ^ { * } ( \theta ^ { * } , s ^ { n } )$ and $w \geq 0 , x ^ { n }$ is feasible, and $x ^ { * } ( \theta ^ { * } , s ^ { n } )$ is a feasible solution of the dominance-constrained problem.

![](images/779c3f75ce9b474d476a44e20a491cf8f085929e5668fcddd46579cd3eb89c2e.jpg)  
Figure 2. Relation between the worst 10% of $\ell ^ { \mathrm { O P } }$ (best-so-far) and the number of oracle calls on the knapsack problem. The vertical axis and the lines are drawn as in Figure 1.

Single-machine scheduling problem. For a state $\boldsymbol { s } = ( r , p )$ and an upper bound $\bar { C } \in \mathbb { Z } ^ { d }$ on the completion times, we solve

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { \substack { S , C \in \mathbb { Z } ^ { d } , \ z \in \{ 0 , 1 \} \{ ( j , k ) : j < k \} } } \ \sum _ { j = 1 } ^ { d } \theta _ { j } C _ { j } } } \\ & { } & { \ \mathrm { s . t . } \ C _ { j } = S _ { j } + p _ { j } , \quad r _ { j } \leq S _ { j } , \quad C _ { j } \leq \bar { C } _ { j } \quad ( j = 1 , \ldots , d ) , } \\ & { } & { \quad S _ { k } \geq S _ { j } + p _ { j } - M _ { \mathrm { b i g } } ( 1 - z _ { j k } ) \quad ( 1 \leq j < k \leq d ) , } \\ & { } & { \quad S _ { j } \geq S _ { k } + p _ { k } - M _ { \mathrm { b i g } } z _ { j k } \quad ( 1 \leq j < k \leq d ) } \end{array}
$$

with SCIP and return the action $x = - C$ . For $x ^ { * } ( \theta , s )$ we set $\begin{array} { r } { \bar { C } _ { j } = \operatorname* { m a x } _ { k } r _ { k } + \sum _ { k } p _ { k } } \end{array}$ for $x ^ { * } ( \theta ; x ^ { n } , s )$ we set $\bar { C } _ { j } = - x _ { j } ^ { n }$ , and $M _ { \mathrm { b i g } } = 2 \operatorname* { m a x } _ { j } ( \bar { C } _ { j } + p _ { j } )$

Let $C _ { j } ^ { * }$ be the completion times of $x ^ { * } ( \theta ^ { * } , s ^ { n } )$ . Then the completion times of the agent are $f ( C _ { j } ^ { \ast } ) : = C _ { j } ^ { \ast } + [ C _ { j } ^ { \ast } \geq \tau _ { 1 } ^ { n } ] + [ C _ { j } ^ { \ast } \geq \tau _ { 2 } ^ { n } - 1 ]$ . Since $f ( c + 1 ) - f ( c ) \geq 1 , f$ preserves the processing order and the release-time constraints, so $x ^ { n }$ is feasible and $x ^ { * } ( \theta ^ { * } , s ^ { n } )$ is a feasible solution of the dominance-constrained problem.

Weighted multi-cover problem. The word ranking common to both experiments is given in Algorithm 15, the word scores for Cora and ogbn-arxiv in Algorithms 16 and 17, and the instance generation in Algorithms 18 and 19. Here, argsort<sup>↓</sup> denotes a stable sort in descending order in which ties are ordered by increasing index. The stop words for ogbn-arxiv are 164 words based on the lists of NLTK and scikit-learn. The forward problem is solved with SCIP with $x \in \{ 0 , \ldots , r \} ^ { d } \ ( x ^ { n } \leq x \leq r$ in the dominance-constrained version), and returns the componentwise min $( r , A ^ { \top } z )$ of the selected papers z.

H.7. Results for each dimension on the knapsack problem and the weighted multi-cover problem. For the knapsack problem and the weighted multi-cover problem, Tables 4 and 5 show, for each dimension, the median of $\ell ^ { \mathrm { O P } }$ of the seven methods at B oracle calls.

Figure 2 shows the evolution of the worst 10% of $\ell ^ { \mathrm { O P } }$ (best-so-far) on the knapsack problem.

H.8. Results on the single-machine scheduling problem and the movie selection problem with viewing costs. Tables 6 and 7 show, for each dimension, $\ell ^ { \mathrm { O P } }$ at $B$ oracle calls for the three pairs that difer only in the loss function and for ${ \mathrm { O S - D C } } .$ , and Figure 3 shows its trajectory on the worst 10% of the episodes.

Table 4. Knapsack problem: median of $\ell ^ { \mathrm { O P } }$ at $B = 4 0 0 0$ oracle calls (the attainment rate [%] of $\ell ^ { \mathrm { O P } } = 0$ in parentheses). 50 episodes for each d. For each optimization method, the pair of the OS loss (proposed) and the suboptimality loss (existing) is shown, and the one with the smaller median in each pair (with the higher attainment rate in case of a tie) is shown in bold.
<table><tr><td>Method</td><td>d = 20  $d = 4 0$ </td><td> $d = 6 0$  d = 80 d = 100</td></tr><tr><td>OS-PSG Sub-PSG</td><td>0.00 (98) 0.00 (96) 0.00 (92) 0.15 (16) 0.20 (0) 0.23 (0)</td><td>0.00 (90) 0.00 (76) 0.28 (0) 0.35 (0)</td></tr><tr><td>OS-MetaGrad(FB) Sub-MetaGrad(FB)</td><td>0.00 (86) 0.00 (52) 0.08 (46) 0.15 (6) 0.40 (0) 0.65 (0)</td><td>0.35 (8) 0.38 (12) 0.80 (0) 0.95 (0)</td></tr><tr><td>OS-ONS(FB) Sub-ONS(FB)</td><td>0.10 (34) 0.40 (2) 0.70 (0) 0.15 (0) 0.40 (0) 0.62 (0)</td><td>0.95 (0) 1.15 (0) 0.70 (0) 0.88 (0)</td></tr><tr><td>OS-DC</td><td>0.00 (90) 0.00 (88) 0.00 (86)</td><td>0.00 (82) 0.00 (84)</td></tr></table>

Table 5. Weighted multi-cover problem (left: Cora, right: ogbnarxiv): median of $\ell ^ { \mathrm { O P } }$ at B = 4000 oracle calls. 20 episodes for each d. For each optimization method, the pair of the OS loss (proposed) and the suboptimality loss (existing) is shown, and the one with the smaller median in each pair is shown in bold.

<table><tr><td></td><td>Cora (d)</td><td>ogbn-arxiv (d)</td></tr><tr><td>Method</td><td>20 40 60 80</td><td>100 20 40 60 80 100</td></tr><tr><td>OS-PSG</td><td>0.78 2.58 4.40 5.72 6.82</td><td>0.57 2.40 4.30 6.50 9.30</td></tr><tr><td>Sub-PSG</td><td>1.25 3.00 4.95 6.25 7.47</td><td>1.12 2.95 4.85 7.22 9.40</td></tr><tr><td>OS-MetaGrad(FB)</td><td>0.972.92 5.25 6.757.80</td><td>0.78 2.73 4.65 7.18 9.55</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>1.27 3.23 5.18 7.05 8.15</td><td>1.15 2.97 4.85 7.20 9.47</td></tr><tr><td>OS-ONS(FB)</td><td>1.00 3.05 5.22 6.88 8.00</td><td>0.822.674.686.92 9.43</td></tr><tr><td>Sub-ONS(FB)</td><td>1.33 3.23 5.20 6.82 8.38</td><td>1.15 2.97 4.85 7.00 9.43</td></tr><tr><td>OS-DC</td><td>0.82 2.60 4.40 5.83 7.30</td><td>0.65 2.52 4.53 7.00 9.35</td></tr></table>

Table 6. Single-machine scheduling problem: median of $\ell ^ { \mathrm { O P } }$ at $B \ : = \ : 2 0 0 0$ oracle calls (the attainment rate [%] of $\ell ^ { \mathrm { O P } } = 0$ in parentheses). 25 episodes for each d. For each optimization method, the pair of the OS loss (proposed) and the suboptimality loss (existing) is shown, and the one with the smaller median in each pair (with the higher attainment rate in case of a tie) is shown in bold.

<table><tr><td>Method</td><td> $d = 4$ </td><td> $d = 8$ </td></tr><tr><td>OS-PSG Sub-PSG</td><td>0.00 (100) 0.00 (100) 0.00 (88) 0.00 (72)</td><td>0.00 (96) 0.10 (44)</td></tr><tr><td>OS-MetaGrad(FB) Sub-MetaGrad(FB)</td><td>0.00 (100) 0.00 (100) 0.00 (84) 0.00 (76)</td><td>0.00 (96) 0.30 (28)</td></tr><tr><td>OS-ONS(FB) Sub-ONS(FB)</td><td>0.00 (100) 0.00 (100) 0.00 (84) 0.00 (52)</td><td>0.00 (84) 0.30 (40)</td></tr><tr><td>OS-DC</td><td>0.00 (100) 0.00 (96)</td><td>0.00 ) (96)</td></tr></table>

Table 7. Movie selection problem with viewing costs: median of $\ell ^ { \mathrm { O P } }$ at $B = 4 0 0 0$ oracle calls. 20 episodes for each d. For each optimization method, the pair of the OS loss (proposed) and the suboptimality loss (existing) is shown, and the one with the smaller median in each pair is shown in bold.
<table><tr><td>Method</td><td>d = 20</td><td>d = 40</td><td>d = 60</td><td>d = 80</td><td>d = 100</td></tr><tr><td>OS-PSG</td><td>0.72 0.95</td><td>2.52 2.85</td><td>4.05 4.60</td><td>5.18 5.80</td><td>6.50 [0.59,1.01] [2.02,2.90] [3.65,4.22] [4.57,5.71] [5.65,7.09] 7.57</td></tr><tr><td>Sub-PSG OS-MetaGrad(FB)</td><td>[0.70,1.11] 0.95</td><td>[2.44,3.17] 2.70</td><td>[4.40,5.11] 4.50</td><td>[5.39,6.53] 5.65</td><td>[6.84,7.98] 7.10</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>[0.79,1.10] [2.60,3.10] [4.09,4.79] [5.42,6.19] [6.50,7.75] 1.30</td><td>3.25</td><td>4.88</td><td>6.18</td><td>7.90</td></tr><tr><td>OS-ONS(FB)</td><td>[1.12,1.52] 1.10 [0.85,1.30] [2.91,3.31] [4.49,5.16]</td><td>[2.91,3.42] 3.15</td><td>[4.67,5.20] 4.95</td><td>[5.85,6.53] 6.45 [6.01,6.76]</td><td>[6.66,8.49] 8.00 [7.08,8.49]</td></tr><tr><td>Sub-ONS(FB)</td><td>1.35 [1.19,1.51]</td><td>3.27 [2.95,3.40]</td><td>4.95 [4.67,5.15]</td><td>6.17 [5.85,6.65] [6.96,8.26]</td><td>7.72</td></tr><tr><td>OS-DC</td><td>0.62 [0.50,0.85]</td><td>2.30 [2.05,2.96]</td><td>4.20 [3.89,4.36]</td><td>5.28 [4.99,5.81] [6.12,7.15]</td><td>6.83</td></tr></table>

![](images/2b61bf69880cd10434913663930b4f35d361ae952178071012a86dae055ae09c.jpg)  
(a) Single-machine scheduling problem

![](images/b6f1060e6f8b8efc8c5d7b7d1af6dbb5e0f42327ce3e143ddf7ab1d9dc6bbdd1.jpg)  
(b) Movie selection problem with viewing costs

Figure 3. Relation between the worst 10% of $\ell ^ { \mathrm { O P } }$ (best-so-far) and the number of oracle calls on the single-machine scheduling problem and the movie selection problem with viewing costs. The vertical axis and the lines are drawn as in Figure 1.

Single-machine scheduling problem. In all three pairs and in all dimensions, the OS-loss variant achieved a higher attainment rate (Table 6). The attainment rates of the four methods with the OS loss are 84–100%, whereas those of the suboptimality-loss variant decrease with the dimension; at $d = 8 ,$ , Sub-PSG attains 44%, Sub-MetaGrad(FB) 28%, and Sub-ONS(FB) 40% (the OS-loss variants of the same pairs attain 96%, 96%, and 84%, respectively). Aggregated over all dimensions, the attainment rates of the eight methods with the OS loss are 91–99%, which exceed those of the five existing methods including the online versions (23–68%) (Table 9). Comparing on the same instances, among the 75 episodes, the numbers of episodes in which $\mathsf { \Pi } _ { \ell } \mathrm { O P }$ of OS-PSG is smaller than that of Sub-PSG, Sub-ONS, and Sub-MetaGrad are 24, 58, and 36, respectively, and the numbers of episodes in which it is larger are 0, 0, and 0 (for OS-DC, the numbers of episodes with smaller $\ell ^ { \mathrm { O P } }$ are 24, 58, and 36, and those with larger $\ell ^ { \mathrm { O P } }$ are 2, 0, and 0). In the three pairs that difer only in the loss function, the OS-loss variant loses in only 3 episodes in total (Table 1). The remaining episodes are ties. As in the knapsack problem, in this setting where the actions of the agent are degraded optimal solutions, the advantage of the proposed method becomes more pronounced in higher dimensions. Movie selection problem with viewing costs. In this problem, the actions of the agent are real data, and since we do not assume the existence of a weight satisfying Equation (3.2), we compare the values of $\ell ^ { \mathrm { O P } }$ (Table 10). In the comparison that replaces only the loss function, the OS-loss variant won in all three pairs: 97 wins and 1 loss in the PSG pair, 90 wins and 1 loss in the MetaGrad(FB) pair, and 45 wins and 32 losses in the ONS(FB) pair (Table 1). Looking at each dimension, in the PSG pair and the MetaGrad(FB) pair, the median of the OS-loss variant is smaller in all dimensions (Table 7). In particular, in the PSG pair the diference widens with the dimension; at d = 100, OS-PSG attains 6.50 whereas Sub-PSG attains 7.57. In the ONS(FB) pair, the median of OS-ONS(FB) is smaller for $d \leq 4 0$ , the two are equal at d = 60, and the median of Sub-ONS(FB) is smaller for $d \geq 8 0$ . Among the seven methods in the table, the smallest median is attained by OS-DC (0.62, 2.30) for d = 20, 40 and by OS-PSG (4.05, 5.18, 6.50) for d = 60, 80, 100; that is, it is attained by a method with the OS loss in all dimensions. In other words, under the same optimization method, merely replacing the suboptimality loss with the OS loss improves the prediction of solutions that outperform the agent. Note that the online versions Sub-ONS and Sub-MetaGrad, which update state by state, have medians of 3.550 and 3.425 over all dimensions, respectively, which are smaller than that of OS-PSG (3.975) (Table 10). Since the medians of the full-batch versions of the same learners, Sub-ONS(FB) and Sub-MetaGrad(FB), are 4.950 and 4.875, respectively, this diference is presumably due not to the use of the suboptimality loss but to the optimization method, namely online learning that updates state by state. Among the proposed methods, the median of OS-DC[ONS] over all dimensions (3.550) equals that of Sub-ONS (3.550), and its median computation time is 31 seconds, about one third of that of Sub-ONS (91 seconds). Combining the OS loss with methods that minimize it state by state via online learning is left for future work.

## H.9. Results of all methods on the five experiments.

Results of the methods not included in the main text. On the knapsack problem, comparing on the same instances, among the 250 episodes, the numbers of episodes in which $\mathsf { \Omega } _ { \ell } \mathrm { O P }$ of OS-PSG is smaller than that of Sub-PSG, Sub-ONS, and Sub-MetaGrad are 241, 227, and 243, respectively, and there is no episode in which it is larger (for OS-DC, the numbers of episodes with smaller $\ell ^ { \mathrm { O P } }$ are 240, 222, and 242, and those with larger $\ell ^ { \mathrm { O P } }$ are 1, 3, and 0). Here, Sub-ONS and Sub-MetaGrad are the online versions that update state by state, and their attainment rates are also 9% and 3%, respectively, over all dimensions (Table 8). The empirical mean of the OS loss measured at the same weights is also 0 in median for OS-PSG and OS-DC, whereas it remains at $3 . 9 \times 1 0 ^ { - 4 } – 2 . 8 \times 1 0 ^ { - 3 }$ for Sub-PSG, Sub-ONS, and Sub-MetaGrad (Table 8). On the weighted multi-cover problem, unlike the movie selection problem with viewing costs (§H.8), the medians of $\ell ^ { \mathrm { O P } }$ of the online versions Sub-ONS and Sub-MetaGrad also fall short of the best method with the OS loss. Among all 13 methods, the smallest median of $\ell ^ { \mathrm { O P } }$ over all dimensions is attained by OS-DC[LV+CPF]+SGF (4.000) on Cora and by OS-DC[ONS] (3.575) on ogbn-arxiv, which are smaller than the best of the existing methods (4.500 of Sub-ONS on Cora and 3.950 of Sub-MetaGrad on ogbn-arxiv) (Tables 11 and 12). Comparing on the same instances the pairs of the proposed method and the existing method with the smallest mean of $\ell ^ { \mathrm { { { O P } } } }$ over all episodes (OS-DC[LV+CPF]+SGF vs. Sub-MetaGrad on Cora, and OS-DC[MetaGrad] vs. Sub-MetaGrad on ogbn-arxiv), the proposed method wins 65 times and loses 30 times on Cora, and wins 85 times and loses 11 times on ogbn-arxiv. The median computation time of OS-PSG is 8.7 seconds on Cora (second to 8.2 seconds of OS-ONS(FB)) and 7.4 seconds on ogbn-arxiv (the shortest among all methods), about one quarter of that of Sub-MetaGrad (35.4 seconds and 31.1 seconds); note that the time of Sub-MetaGrad includes the solves used only for recording, which OS-PSG does not need (see the definition of “Time” below). Among the three pairs difering only in the loss function, in the two full-batch pairs, although the OS-loss variant won overall, it lost in 17 episodes (MetaGrad(FB)) and 22 episodes (ONS(FB)) on Cora, so the margin is smaller than in the PSG pair (Table 1).

Tables 8 to 12 show the results of all 13 methods on the five experiments. In each table, the upper block shows existing inverse optimization (the suboptimality loss), and the lower block shows outperformance inverse optimization (the OS loss). In the method names, the bracket of OS-DC[·] indicates the inner solver, and +SGF indicates that subgradient finishing is added. Each value is the median, over all dimensions and episodes, of the value at B oracle calls; for $\ell ^ { \mathrm { O P } }$ , the interquartile range (the first and third quartiles) is also shown. $\bar { \ell } ^ { \mathrm { O S } }$ is measured at the weight that gives the best-so-far $\ell ^ { \mathrm { { \scriptsize { \dot { O P } } } } } \left( \ S 7 . 1 \right)$ . “Time” is the computation time until the run stops, that is, until the number of oracle calls reaches B or $\ell ^ { \mathrm { O P } } = 0$ is attained, whichever comes first (the part after that is not included), and its mean and median are shown. It also includes the time for the solves used only for recording and not counted as oracle calls, that is, the solves of $x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ for recording $\mathrm { \check { \bar { \ell } } o s }$ in the Sub methods and OS-DC. Hence, in comparisons with the OS methods other than OS-DC, which do not need these solves, the times of the Sub methods are overestimated.

On the knapsack problem, OS-PSG (attainment rate 90%), OS-DC[PB+CPF]+SGF (86%), and OS-DC[LV+CPF]+SGF (85%) attained a median of $\ell ^ { \mathrm { O P } }$ equal to 0 and outperform the five existing methods (attainment rates 0–9%). The attainment rates of OS-DC[CP]+SGF and OS-ONS(FB) are 12% and 7%, respectively. On the single-machine scheduling problem, the attainment rates of the eight methods with the OS loss are 91–99%, which exceed those of the five existing methods (23–68%). On the movie selection problem with viewing costs, the attainment rates of all methods are $0 \% ;$ the median of $\ell ^ { \mathrm { O P } }$ is the smallest for Sub-MetaGrad (3.425), followed by Sub-ONS and OS-DC[ONS] (both 3.550). On this problem, OS-DC[LV+CPF]+SGF, which has the smallest empirical mean of the OS loss (median $3 . 4 \times 1 0 ^ { - 3 } )$ ), has a median of $\ell ^ { \mathrm { O P } }$ equal to

Table 8. Knapsack problem (oracle-call budget $B = 4 0 0 0 )$ . Each value is the median, over the 250 runs $( d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ ， 50 episodes for each $d )$ , of the value at B oracle calls, and the attainment rate is the fraction of these 250 runs that attained $\ell ^ { \mathrm { O P } } = 0$ . For the time (in seconds), the mean is also shown.
<table><tr><td>Method</td><td> $\overline { { \ell ^ { \mathrm { O P } } } }$  median</td><td>interquartile attainment</td><td></td><td> $\overline { { \ell ^ { \mathrm { O S } } } }$ </td><td>time</td><td>time</td></tr><tr><td>Sub-PSG</td><td>0.250</td><td>range [0.150,0.300]</td><td>rate [%] 3</td><td>median  $\overline { { 4 . 9 \times 1 0 ^ { - 4 } } }$ </td><td>1.3</td><td>mean median 1.3</td></tr><tr><td>Sub-ONS</td><td>0.150</td><td>[0.100,0.250]</td><td>9</td><td> $3 . 9 \times 1 0 ^ { - 4 }$ </td><td>2.2</td><td>2.2</td></tr><tr><td>Sub-MetaGrad</td><td>0.350</td><td>[0.200,0.550]</td><td>3</td><td> $2 . 8 \times 1 0 ^ { - 3 }$ </td><td>8.4</td><td>8.3</td></tr><tr><td>Sub-ONS(FB)</td><td>0.600</td><td>[0.300,0.800]</td><td>0</td><td> $6 . 2 \times 1 0 ^ { - 3 }$ </td><td>1.4</td><td>1.4</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>0.550</td><td>[0.300,0.850]</td><td>1</td><td> $5 . 5 \times 1 0 ^ { - 3 }$ </td><td>2.4</td><td>2.4</td></tr><tr><td>OS-PSG</td><td>0.000</td><td>[0.000,0.000]</td><td>90</td><td>0</td><td>0.3</td><td>0.2</td></tr><tr><td>OS-ONS(FB)</td><td>0.700</td><td>[0.300,1.000]</td><td>7</td><td> $8 . 9 \times 1 0 ^ { - 3 }$ </td><td>0.7</td><td></td></tr><tr><td></td><td>0.100</td><td>[0.000,0.450]</td><td>41</td><td> $4 . 8 \times 1 0 ^ { - 4 }$ </td><td></td><td>0.7</td></tr><tr><td>OS-MetaGrad(FB)</td><td>0.100</td><td>[0.000,0.250]</td><td>26</td><td> $5 . 2 \times 1 0 ^ { - 4 }$ </td><td>0.9</td><td>0.8</td></tr><tr><td>OS-DC[ONS]</td><td>0.050</td><td>[0.000,0.150]</td><td>44</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td>1.3</td><td>1.3</td></tr><tr><td>OS-DC[MetaGrad]</td><td>0.650</td><td>[0.213,1.050]</td><td>12</td><td> $7 . 2 \times 1 0 ^ { - 3 }$ </td><td>1.8 1.7</td><td>1.5</td></tr><tr><td>OS-DC[CP]+SGF  $\mathrm { O S - D C [ L V + C P F ] + S G F }$ </td><td>0.000</td><td>[0.000,0.000]</td><td>85</td><td>0</td><td>0.9</td><td>1.7</td></tr><tr><td></td><td>0.000</td><td></td><td></td><td></td><td></td><td>0.7</td></tr><tr><td>OS-DC[PB+CPF]+SGF</td><td></td><td>[0.000,0.000]</td><td>86</td><td>0</td><td>0.7</td><td>0.6</td></tr></table>

Table 9. Single-machine scheduling problem (oracle-call budget B = 2000). Each value is the median, over the 75 runs $( d \in$ {4, 6, 8}, 25 episodes for each $d ) .$ , of the value at B oracle calls, and the attainment rate is the fraction of these 75 runs that attained $\ell ^ { \mathrm { O P } } = 0$ . For the time (in seconds), the mean is also shown.
<table><tr><td>Method</td><td> $\overline { { \ell ^ { \mathrm { O P } } } }$  median</td><td>interquartile attainment</td><td></td><td> $\overline { { \ell ^ { \mathrm { O S } } } }$ </td><td>time</td><td>time</td></tr><tr><td>Sub-PSG</td><td>0.000</td><td>range [0.000,0.150]</td><td>rate [%] 68</td><td>median 0</td><td>17.7</td><td>mean median 2.8</td></tr><tr><td>Sub-ONS</td><td>0.500</td><td>[0.100,1.350]</td><td>23</td><td> $1 . 5 \times 1 0 ^ { - 2 }$ </td><td>32.6</td><td>17.3</td></tr><tr><td>Sub-MetaGrad</td><td>0.000</td><td>[0.000,0.300]</td><td>52</td><td>0</td><td>24.9</td><td>11.8</td></tr><tr><td>Sub-ONS(FB)</td><td>0.000</td><td>[0.000,0.300]</td><td>59</td><td>0</td><td>24.6</td><td>13.7</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>0.000</td><td>[0.000,0.200]</td><td>63</td><td>0</td><td>19.7</td><td>1.2</td></tr><tr><td>OS-PSG</td><td>0.000</td><td>[0.000,0.000]</td><td>99</td><td>0</td><td>6.3</td><td>1.1</td></tr><tr><td>OS-ONS(FB)</td><td>0.000</td><td>[0.000,0.000]</td><td>95</td><td>0</td><td>10.3</td><td>1.2</td></tr><tr><td></td><td>0.000</td><td>[0.000,0.000]</td><td>99</td><td>0</td><td>6.3</td><td></td></tr><tr><td>OS-MetaGrad(FB)</td><td>0.000</td><td>[0.000,0.000]</td><td>91</td><td>0</td><td>19.6</td><td>1.5</td></tr><tr><td>OS-DC[ONS]</td><td>0.000</td><td>[0.000,0.000]</td><td>93</td><td>0</td><td>18.7</td><td>1.9</td></tr><tr><td>OS-DC[MetaGrad]</td><td>0.000</td><td>[0.000,0.000]</td><td>97</td><td>0</td><td>14.4</td><td>3.3</td></tr><tr><td>OS-DC[CP]+SGF OS-DC[LV+CPF]+SGF</td><td>0.000</td><td>[0.000,0.000]</td><td>95</td><td>0</td><td>9.9</td><td>3.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>1.7</td></tr><tr><td> $\mathrm { O S - D C [ P B + C P F ] + S G F }$ </td><td>0.000</td><td>[0.000,0.000]</td><td>97</td><td>0</td><td>10.9</td><td>2.2</td></tr></table>

Table 10. Movie selection problem with viewing costs (oracle-call budget $B = 4 0 0 0 )$ . Each value is the median, over the 100 runs $( d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ , 20 episodes for each $d )$ , of the value at B oracle calls, and the attainment rate is the fraction of these 100 runs that attained $\ell ^ { \mathrm { O P } } = 0$ . For the time (in seconds), the mean is also shown.
<table><tr><td rowspan="2">Method</td><td> $\overline { { \ell ^ { \mathrm { O P } } } }$ </td><td>interquartile attainment</td><td></td><td> $\overline { { \ell ^ { \mathrm { O S } } } }$ </td><td>time</td><td>time</td></tr><tr><td>median</td><td>range</td><td>rate [%]</td><td>median</td><td></td><td>mean median</td></tr><tr><td>Sub-PSG</td><td>4.600</td><td>[2.438,6.000]</td><td>0</td><td> $\overline { { 3 . 1 \times 1 0 ^ { - 2 } } }$ </td><td>106.7</td><td>95.3</td></tr><tr><td>Sub-ONS</td><td>3.550</td><td>[1.950,4.612]</td><td>0</td><td> $2 . 0 \times 1 0 ^ { - 2 }$ </td><td>98.9</td><td>91.2</td></tr><tr><td>Sub-MetaGrad</td><td>3.425</td><td>[1.900,4.600]</td><td>0</td><td> $2 . 4 \times 1 0 ^ { - 2 }$ </td><td>107.4</td><td>97.5</td></tr><tr><td>Sub-ONS(FB)</td><td>4.950</td><td>[2.950,6.312]</td><td>0</td><td> $5 . 0 \times 1 0 ^ { - 2 }$ </td><td>39.6</td><td>40.4</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>4.875</td><td>[2.913,6.300]</td><td>0</td><td> $5 . 4 \times 1 0 ^ { - 2 }$ </td><td>76.1</td><td>62.5</td></tr><tr><td>OS-PSG</td><td>3.975</td><td>[2.025,5.362]</td><td>0</td><td> $\overline { { 1 . 2 \times 1 0 ^ { - 2 } } }$ </td><td>75.7</td><td>81.1</td></tr><tr><td>OS-ONS(FB)</td><td>4.950</td><td>[2.913,6.487]</td><td>0</td><td> $4 . 9 \times 1 0 ^ { - 2 }$ </td><td>7.0</td><td>7.1</td></tr><tr><td>OS-MetaGrad(FB)</td><td>4.500</td><td>[2.600,5.850]</td><td>0</td><td> $7 . 5 \times 1 0 ^ { - 2 }$ </td><td>18.0</td><td>17.4</td></tr><tr><td>OS-DC[ONS]</td><td>3.550</td><td>[2.000,5.013]</td><td>0</td><td> $3 . 9 \times 1 0 ^ { - 2 }$ </td><td>21.6</td><td></td></tr><tr><td>OS-DC[MetaGrad]</td><td>3.900</td><td>[2.188,5.312]</td><td>0</td><td> $5 . 5 \times 1 0 ^ { - 2 }$ </td><td>23.2</td><td>23.0</td></tr><tr><td>OS-DC[CP]+SGF</td><td>4.750</td><td>[2.700,6.475]</td><td>0</td><td> $4 . 3 \times 1 0 ^ { - 2 }$ </td><td>16.2</td><td>23.6</td></tr><tr><td> $\mathrm { O S - D C [ L V + C P F ] + S G F }$ </td><td>4.150</td><td>[1.938,5.463]</td><td>0</td><td> $3 . 4 \times 1 0 ^ { - 3 } 1 1 5 . 0$ </td><td></td><td>16.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>110.3</td></tr><tr><td> $\mathrm { O S - D C [ P B + C P F ] + S G F }$ </td><td>4.200</td><td>[2.050,5.450]</td><td>0</td><td> $1 . 0 \times 1 0 ^ { - 2 } 1 0 5 . 2$ </td><td></td><td>102.9</td></tr></table>

Table 11. Weighted multi-cover problem (Cora) (oracle-call budget $B \ = \ 4 0 0 0 )$ Each value is the median, over the 100 runs $( d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ , 20 episodes for each $d ) _ { ; }$ , of the value at B oracle calls, and the attainment rate is the fraction of these 100 runs that attained $\ell ^ { \mathrm { O P } } = 0$ . For the time (in seconds), the mean is also shown.

<table><tr><td rowspan="2">Method</td><td> $\overline { { \ell ^ { \mathrm { O P } } } }$  range</td><td>interquartile attainment</td><td></td><td> $\overline { { \ell ^ { \mathrm { O S } } } }$ </td><td>time</td><td>time</td></tr><tr><td colspan="3">median</td><td colspan="2">rate [%] median</td><td>mean median</td><td></td></tr><tr><td>Sub-PSG</td><td>4.800</td><td>[2.787,6.338]</td><td>0</td><td> $7 . 0 \times 1 0 ^ { - 2 }$ </td><td>18.9</td><td>16.9</td><td></td></tr><tr><td>Sub-ONS</td><td>4.500</td><td>[2.688,6.100]</td><td>0</td><td> $7 . 9 \times 1 0 ^ { - 2 }$ </td><td>17.9</td><td></td><td>16.9</td></tr><tr><td>Sub-MetaGrad</td><td>4.600</td><td>[2.537,6.075]</td><td>0</td><td> $9 . 1 \times 1 0 ^ { - 2 }$ </td><td></td><td>40.8</td><td>35.4</td></tr><tr><td>Sub-ONS(FB)</td><td>5.150</td><td>[2.888,6.812]</td><td>0</td><td></td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td><td>17.7</td><td>16.2</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>5.150</td><td>[2.975,7.075]</td><td>0</td><td></td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td><td>21.8</td><td>20.3</td></tr><tr><td>OS-PSG</td><td>4.375</td><td>[2.362,5.900]</td><td></td><td>0</td><td> $\overline { { 6 . 3 \times 1 0 ^ { - 2 } } }$ </td><td>8.9</td><td>8.7</td></tr><tr><td>OS-ONS(FB)</td><td>5.150</td><td>[2.650,6.963]</td><td></td><td>0</td><td> $1 . 4 \times 1 0 ^ { - 1 }$ </td><td>8.8</td><td>8.2</td></tr><tr><td>OS-MetaGrad(FB)</td><td>5.125</td><td>[2.712,6.975]</td><td></td><td>0</td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td><td>10.5</td><td>10.1</td></tr><tr><td>OS-DC[ONS]</td><td>4.350</td><td>[2.337,5.663]</td><td></td><td>0</td><td> $7 . 9 \times 1 0 ^ { - 2 }$ </td><td>17.2</td><td></td></tr><tr><td>OS-DC[MetaGrad]</td><td>4.100</td><td>[2.375,5.713]</td><td></td><td>0</td><td> $7 . 1 \times 1 0 ^ { - 2 }$ </td><td>20.1</td><td>16.4</td></tr><tr><td>OS-DC[CP]+SGF</td><td>4.775</td><td>[2.300,7.075]</td><td></td><td>0</td><td> $8 . 3 \times 1 0 ^ { - 2 }$ </td><td>13.5</td><td>19.4</td></tr><tr><td> $\mathrm { O S - D C [ L V + C P F ] + S G F }$ </td><td>4.000</td><td></td><td>[1.550,6.100]</td><td>0</td><td> $4 . 6 \times 1 0 ^ { - 2 }$ </td><td>15.3</td><td>12.3</td></tr><tr><td> $\mathrm { O S - D C [ P B + C P F ] + S G F }$ </td><td>4.400</td><td></td><td>[2.287,6.150]</td><td>0</td><td> $5 . 1 \times 1 0 ^ { - 2 }$ </td><td>15.1</td><td>13.7 13.7</td></tr></table>

Table 12. Weighted multi-cover problem (ogbn-arxiv) (oracle-call budget $B = 4 0 0 0 )$ . Each value is the median, over the 100 runs $( d \in \{ 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ , 20 episodes for each $d ) _ { ; }$ , of the value at B oracle calls, and the attainment rate is the fraction of these 100 runs that attained $\ell ^ { \mathrm { O P } } = 0$ . For the time (in seconds), the mean is also shown.
<table><tr><td rowspan="2">Method</td><td rowspan="2"> $\overline { { \ell ^ { \mathrm { O P } } } }$  median</td><td colspan="2">interquartile attainment</td><td colspan="2"> $\overline { { \ell ^ { \mathrm { O S } } } }$ </td><td rowspan="2">time time</td></tr><tr><td>range</td><td>rate [%]</td><td>median</td><td>mean median</td></tr><tr><td>Sub-PSG</td><td>4.650</td><td>[2.612,7.375]</td><td>0</td><td> $\overline { { 8 . 4 \times 1 0 ^ { - 2 } } }$ </td><td>17.2</td><td>14.2</td></tr><tr><td>Sub-ONS</td><td>4.475</td><td>[2.663,6.650]</td><td>0</td><td> $3 . 4 \times 1 0 ^ { - 2 }$ </td><td>16.3</td><td>14.6</td></tr><tr><td>Sub-MetaGrad</td><td>3.950</td><td>[2.012,6.013]</td><td>0</td><td> $7 . 2 \times 1 0 ^ { - 2 }$ </td><td>36.5</td><td>31.1</td></tr><tr><td>Sub-ONS(FB)</td><td>4.650</td><td>[2.650,7.312]</td><td>0</td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td><td>18.0</td><td>14.1</td></tr><tr><td>Sub-MetaGrad(FB)</td><td>4.650</td><td>[2.650,7.537]</td><td>0</td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td><td>21.4</td><td>17.3</td></tr><tr><td>OS-PSG</td><td>4.200</td><td>[1.738,7.088]</td><td>0</td><td> $\overline { { 4 . 7 \times 1 0 ^ { - 2 } } }$ </td><td>8.5</td><td>7.4</td></tr><tr><td>OS-ONS(FB)</td><td>4.625</td><td>[2.300,7.188]</td><td></td><td> $1 . 2 \times 1 0 ^ { - 1 }$ </td><td></td><td></td></tr><tr><td></td><td>4.625</td><td></td><td>0</td><td></td><td>9.1</td><td>7.5</td></tr><tr><td>OS-MetaGrad(FB)</td><td>3.575</td><td>[2.325,7.263]</td><td>0</td><td> $1 . 2 \times 1 0 ^ { - 1 }$ </td><td>10.4</td><td>8.6</td></tr><tr><td>OS-DC[ONS]</td><td>3.600</td><td>[1.700,5.812]</td><td>0</td><td> $5 . 9 \times 1 0 ^ { - 2 }$ </td><td>17.5</td><td>14.4</td></tr><tr><td>OS-DC[MetaGrad]</td><td>4.550</td><td>[1.688,5.562] [2.438,7.500]</td><td>0</td><td> $6 . 9 \times 1 0 ^ { - 2 }$ </td><td>19.4</td><td>16.8</td></tr><tr><td>OS-DC[CP]+SGF</td><td>4.325</td><td>[1.887,7.050]</td><td>0</td><td> $8 . 2 \times 1 0 ^ { - 2 }$ </td><td>10.0</td><td>8.6</td></tr><tr><td> $\mathrm { O S - D C [ L V + C P F ] + S G F }$ </td><td></td><td></td><td>0</td><td> $1 . 2 \times 1 0 ^ { - 4 }$ </td><td>11.7</td><td>9.8</td></tr><tr><td> $\mathrm { O S - D C [ P B + C P F ] + S G F }$ </td><td>4.225</td><td>[1.938,7.125]</td><td>0</td><td> $1 . 4 \times 1 0 ^ { - 2 }$ </td><td>10.8</td><td>9.2</td></tr></table>

4.150, which is larger than that of Sub-ONS (median empirical mean of the OS loss $2 . 0 \times 1 0 ^ { - 2 } )$ . This is why we use $\ell ^ { \mathrm { O P } }$ as the main evaluation metric in addition to the empirical mean of the OS loss (§4). On the weighted multi-cover problem, the attainment rates of all methods are 0%, and the smallest median of $\ell ^ { \mathrm { O P } }$ is attained by $\mathrm { O S - D C [ L V + C P F ] + S G F }$ (4.000) on Cora and by OS-DC[ONS] (3.575) on ogbn-arxiv. On ogbn-arxiv, OS-DC[LV+CPF]+SGF has the smallest median empirical mean of the OS loss among all methods, $1 . 2 \times 1 0 ^ { - 4 }$ , but its median of $\ell ^ { \mathrm { O } \dot { \mathrm { P } } }$ is 4.325, which is larger than that of OS-DC[ONS].

NEC Corporation<sub>,</sub> 1753 Shimonumabe<sub>,</sub> Nakahara-ku<sub>,</sub> Kawasaki<sub>,</sub> Kanagawa<sub>,</sub> Japan Email address: akira-kitaoka@nec.com

Algorithm 12 The whole procedure of $\mathrm { O S - D C [ \cdot ] \ ( + S G F ) }$   
Input: $K = \lfloor 2 T / 2 0 \rfloor , J = 2 0 , \epsilon _ { \mathrm { t i e } } = 1 0 ^ { - 8 } , \epsilon _ { \mathrm { g a p } } = 1 0 ^ { - 6 }$ , inner solver, whether to   
perform subgradient finishing   
1: $\theta  \Pi _ { \Delta ^ { d - 1 } } ( 1 / d ) , y ^ { \prime n }  x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ for each n $( N$ oracle calls), $\nu  \emptyset .$ , and let   
the internal state of the learner $\mathcal { L }$ be undefined (ONS, MetaGrad)   
2: for $k = 0 , 1 , \ldots , K - 1$ do   
3: Run the inner solver $( \mathrm { C P } , \mathrm { L V + C P F } , \mathrm { P B + C P F } .$ : Algorithm 5 (using inherited   
cuts from $\nu ) ;$ ; ONS, MetaGrad: Algorithm 11 (inputs $\theta ^ { k } : = \theta$ and $\mathcal { L } ;$ the   
output $\mathcal { L }$ replaces $\mathcal { L } ) )$ from the initial point $\theta ,$ and obtain $( \theta ^ { \prime } .$ , converged)   
4: Evaluate $\theta \gets \Pi _ { \Delta ^ { d - 1 } } ( \theta ^ { \prime } )$ to obtain $v ^ { n }$ and add them to $\nu ,$ , and set $y ^ { \prime \prime n } \gets$   
$x ^ { * } ( \theta ; x ^ { n } , s ^ { n } )$ for each n (2N oracle calls in total)   
5: $\begin{array} { r } { \ell _ { k } \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \langle \theta , v ^ { n } - y ^ { \prime \prime n } \rangle , \Delta _ { k } \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \langle \theta , y ^ { \prime \prime n } - y ^ { \prime n } \rangle , \theta ^ { ( k ) } \gets \theta } \end{array}$   
6: if $\ell _ { k } \le \epsilon _ { \mathrm { g a p } } ,$ or $( \Delta _ { k } \le \epsilon _ { \mathrm { t i e } }$ and converged $= \mathrm { T R U E } )$ then   
7: break   
8: end if   
9: $y ^ { \prime }  y ^ { \prime \prime }$   
10: end for   
11: $\theta ^ { \mathrm { D C } }  \theta ^ { ( k ^ { \star } ) }$ , where $k ^ { \star }$ is the smallest index minimizing $\ell _ { k }$ (regardless of the   
reason for stopping)   
12: if subgradient finishing is performed then   
13: Run Algorithm $8$ with type OS and $( \theta ^ { 0 } , \eta , n _ { \mathrm { i t } } ) = ( \theta ^ { \mathrm { D C } } , 1 0 ^ { - 3 } , T )$ , and let $\theta ^ { \mathrm { S G F } }$   
be its output   
14: return $\dot { \theta } ^ { \mathrm { S G F } }$   
15: end if   
16: return $\theta ^ { \mathrm { D C } }$

Algorithm 13 Instance generation for the knapsack problem (one episode and   
dimension $d )$   
Input: episode index e, dimension $d ,$ number of states $N = 2 0$   
1: $\theta ^ { * } \sim \mathrm { D i r } ( { \bf 1 } )$   
2: for $n = 1 , \ldots , N$ do   
3: $w _ { i } ^ { n } \sim \mathrm { U } \{ 1 , \dots , 9 \} , u _ { i } ^ { n } \sim \mathrm { U } \{ 1 , \dots , 4 \} ( i = 1 , \dots , d ) , c ^ { n } \sim \mathrm { U } \{ 1 , \dots , 1 9 \}$ (all   
independent), $s ^ { n } \gets ( w ^ { n } , c ^ { n } , u ^ { n } )$   
4: end for   
5: for $n = 1 , \ldots , N$ do   
6: $x ^ { n }  x ^ { * } ( \theta ^ { * } , s ^ { n } )$   
7: for $i = 1 , \ldots , d$ do   
8: with probability 0.2, x<sup>n</sup> ← max{x<sup>n</sup> − 1, 0}   
9: end for   
10: end for   
11: return $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$

Algorithm 14 Instance generation for the single-machine scheduling problem (one   
episode and dimension d)   
Input: episode index $e ,$ number of jobs $d ,$ number of states $N = 1 0$   
1: $\theta ^ { * } \sim \mathrm { D i r } ( { \bf 1 } )$   
2: for $n = 1 , \ldots , N$ do   
3: $r _ { j } ^ { n } \sim \operatorname { U } \{ 0 , \ldots , 1 4 \} , p _ { j } ^ { n } \sim \operatorname { U } \{ 1 , \ldots , 4 \} \ ( j = 1 , \ldots , d ,$ all independent), $s ^ { n } \gets$   
$( \check { r } ^ { n } , p ^ { n } )$   
4: end for   
5: for $n = 1 , \ldots , N$ do   
6: $x ^ { n }  x ^ { * } ( \theta ^ { * } , s ^ { n } )$   
7: $\tau _ { 1 } ^ { n } \sim \mathrm { U } \{ 8 , \dots , 1 2 \} , \tau _ { 2 } ^ { n } \sim \mathrm { U } \{ 1 8 , \dots , 2 2 \}$   
8: for $j = 1 , \ldots , d$ do   
9: if $x _ { j } ^ { n } \leq - \tau _ { 1 } ^ { n }$ , then $x _ { j } ^ { n } \gets x _ { j } ^ { n } - 1$   
10: if $x _ { j } ^ { n } \leq - \tau _ { 2 } ^ { n }$ , then $x _ { j } ^ { n } \gets x _ { j } ^ { n } - 1$   
11: end for   
12: end for   
13: return $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$

Algorithm 15 Word ranking that alternates between fields   
Input: word scores $S _ { c } ( j )$ for each field $( j = 0 , \dots , n _ { \mathrm { w } } - 1 )$ , an order of fields   
$\left( c _ { 1 } , \ldots , c _ { n _ { \mathrm { f } } } \right)$ , upper limit on the number of words $L = 2 0 0$   
1: For each field $c , \pi _ { c } \gets \mathrm { a r g s o r t } _ { j } ^ { \dagger } S _ { c } ( j ) , p _ { c } \gets 0 ; \mathcal { R } \gets ( )$   
2: while $| \mathcal { R } | < L$ and a word was added to R in the previous round do   
3: for $l = 1 , \ldots , n _ { \mathrm { f } }$ do   
4: $c  c _ { l }$ , and $p _ { c } \gets p _ { c } + 1$ while $\pi _ { c } ( p _ { c } ) \in \mathcal { R }$   
5: if $p _ { c } < n _ { \mathrm { w } }$ and $| \mathcal { R } | < L$ then   
6: append $\pi _ { c } ( p _ { c } )$ to the end of ${ \mathcal { R } } ,$ and $p _ { c } \gets p _ { c } + 1$   
7: end if   
8: end for   
9: end while   
10: return R (the words for dimension d are the first d words)

```latex
Algorithm 16 Word scores for Cora
Input: Bag-of-Words $B ^ { \mathrm { B o W } } \in \{ 0 , 1 \} ^ { 2 7 0 8 \times 1 4 3 3 }$ , fields $c ( i ) \in \{ 0 , \ldots , 6 \}$
1: For each field $\begin{array} { r } { c , n _ { c } \gets | \{ i : \dot { c } ( i ) = c \} | , S _ { c } ( j ) \gets \frac { 1 } { n _ { c } } \sum _ { i : c ( i ) = c } B _ { i j } ^ { \mathrm { B o W } } } \end{array}$ (occurrence
rate)
2: Sort the fields in descending order of $n _ { c }$ (ties by increasing field index), apply
Algorithm 15, and let R be its output
3: return R
```

Algorithm 17 Word scores for ogbn-arxiv   
Input: titles and abstracts of all papers $i = 0 , \ldots , n _ { \mathrm { t o t } } - 1 \ ( n _ { \mathrm { t o t } } = 1 6 9 3 4 3 .$ , including   
papers from 2020), fields $c ( i ) \in \{ 0 , \ldots , 3 9 \}$ , a set of stop words $\mathcal { W } _ { \mathrm { s t o p } }$   
1: For each paper $i ,$ concatenate the title and the abstract and convert them to   
lowercase; let $W _ { i }$ be the set of words, extracted as maximal runs of alphabetic   
characters, that have length at least 3 and are not in $\mathcal { W } _ { \mathrm { s t o p } }$   
2: Sort the words that appear in at least 20 papers in lexicographic order, and   
denote them by $w _ { 0 } , \ldots , w _ { n _ { \mathrm { w } } - 1 }$   
3: For each field $c , n _ { c } \gets | \{ i : c ( i ) = c \} |$ (all years), $\begin{array} { r } { S _ { c } ( j ) \gets \frac { | \{ i : c ( i ) = c , ~ w _ { j } \in W _ { i } \} | } { n _ { c } } \gets } \end{array}$   
$\underline { { | \{ i : w _ { j } \in W _ { i } \} | } }$ (distinctiveness)   
n<sub>tot</sub>   
4: Sort the fields in descending order of $n _ { c }$ (ties by increasing field index), apply   
Algorithm $1 5 ,$ and let $\mathcal { R }$ be its output   
5: return $\mathcal { R }$

Algorithm 18 Instance generation for Cora (episode $e ,$ dimension $d )$   
Input: neighborhoods $\mathcal { N } ( \cdot )$ of the citation graph made undirected with self-loops   
removed, word ranking $\mathscr { R } , N = 2 0 , r = 3$   
1: Let $( C _ { 0 } , \dots , C _ { n _ { \mathrm { c a n d } } - 1 } )$ be the sequence of papers v with $3 \leq | \mathcal { N } ( v ) | \leq 1 0$ , sorted   
in increasing order of paper index   
2: Choose a subset from the uniform distribution over all N-element subsets of   
$\{ 0 , \ldots , n _ { \mathrm { c a n d } } - 1 \}$ , and denote its elements in increasing order by $I _ { 1 } < \cdots < I _ { N }$   
3: for $n = 1 , \ldots , N$ do   
4: $v  C _ { I _ { n } } , E _ { s ^ { n } }  \mathcal { N } ( v ) , P _ { s ^ { n } }  ( E _ { s ^ { n } } \cup \bigcup _ { u \in E _ { s ^ { n } } } \mathcal { N } ( u ) ) \setminus \{ v \}$   
5: $\begin{array} { r } { A _ { i j } \gets B _ { i , \mathcal { R } _ { j } } ^ { \mathrm { B o W } } \ ( i \in P _ { s ^ { n } } , j = 1 , \ldots , d ) , k _ { s ^ { n } } \gets | E _ { s ^ { n } } | , x _ { j } ^ { n } \gets \operatorname* { m i n } \bigl ( r , \sum _ { i \in E _ { s ^ { n } } } A _ { i j } \bigr ) } \end{array}$   
6: end for   
7: return $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$

Algorithm 19 Instance generation for ogbn-arxiv (episode $e ,$ dimension $d )$   
Input: set of directed edges $\mathcal { E } \ ( ( u , w ) \in \mathcal { E }$ means that u cites w), publication year   
$t ( \cdot )$ , field $c ( \cdot )$ , word ranking $\mathscr { R } , N = 2 0 , r = 5$   
1: $\operatorname { R e f } ( u ) : = \{ w : ( u , w ) \in \mathcal { E } , \ t ( w ) \leq 2 0 1 9 , \ w \neq u \}$   
2: For each field $^ { c , }$ let $G _ { c }$ be the sequence of papers v with $t ( v ) = 2 0 1 9 , c ( v ) = c ,$   
and $5 \ \leq \ | \mathrm { R e f } ( v ) | \ \leq \ 8 ,$ , sorted in increasing order of paper index, and let   
$m _ { c } \gets | \{ v : t ( v ) = 2 0 1 9 , \ c ( v ) = c \} |$   
3: Sort the fields in descending order of $m _ { c }$ (ties by increasing field index), keep   
only those with $\left| G _ { c } \right| \geq N .$ , let c be the e-th field of this sequence (0-indexed),   
and set $( G _ { 0 } , \dots , G _ { n _ { \mathrm { c a n d } } - 1 } ) \gets G _ { c }$   
4: Choose a subset from the uniform distribution over all N-element subsets of   
$\{ 0 , \ldots , n _ { \mathrm { c a n d } } - 1 \}$ , and denote its elements in increasing order by $I _ { 1 } < \cdots < I _ { N }$   
5: for $n = 1 , \ldots , N$ do   
6: $v  G _ { I _ { n } } , E _ { s ^ { n } }  \operatorname { R e f } ( v ) , P _ { s ^ { n } }  ( E _ { s ^ { n } } \cup \bigcup _ { w \in E _ { s ^ { n } } } \operatorname { R e f } ( w ) ) \setminus \{ v \}$   
7: $A _ { i j } $ [word $\mathcal { R } _ { j }$ appears in paper $i ] \ ( i \in \bar { P _ { s ^ { n } } } , j = 1 , \ldots , d ) , k _ { s ^ { n } } \gets | E _ { s ^ { n } } | ,$   
$x _ { j } ^ { n } \gets$ min $\textstyle \left( r , \sum _ { i \in E _ { s ^ { n } } } A _ { i j } \right)$   
8: end for   
9: return $\{ ( s ^ { n } , x ^ { n } ) \} _ { n = 1 } ^ { N }$