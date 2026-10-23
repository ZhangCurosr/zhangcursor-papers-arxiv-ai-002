# Semi-Linear Gaussian Bayesian Networks: Enhancing Flexibility in Node Dependencies via penalized B-Splines

Aitor Oviedo-Madrid<sup>1,2,\*</sup>, Mabel Morales-Otero<sup>1,2</sup>, Jose Gonzalez-Gomariz<sup>1,2</sup>, Rubén Armañanzas<sup>1,2</sup>

October 23, 2026

<sup>1</sup> Institute of Data Science and Artificial Intelligence (DATAI), University of Navarra, Pamplona, Spain <sup>2</sup> TECNUN School of Engineering, University of Navarra, San Sebastián, Spain Corresponding author: aoviedomadr@unav.es

## Abstract

Bayesian networks for continuous variables often rely on restrictive assumptions: linear Gaussian models lack flexibility to capture nonlinear dependencies, while non-parametric approaches provide greater flexibility at a high computational cost. This creates a gap for modeling scenarios where variables are approximately Gaussian but exhibit nonlinear relationships. We introduce Semi-Linear Gaussian Bayesian Networks (SLGBN), a novel class of continuous Bayesian net works designed to bridge this gap, preserving Gaussian conditional distributions while introducing nonlinear conditional mean functions. The proposed framework provides a unified probabilistic framework combining linear Gaussian and nonlinear Gaussian conditional models within the same Bayesian network Nonlinear conditional mean functions are estimated using generalized additive models (GAMs) with penalized B-splines. We present parameter estimation procedures for both types of nodes, as well as a greedy hill-climbing algorithm for learning the network structure under the SLGBN framework. To assess its performance, we compare SLGBN with other continuous Bayesian network models, including semiparametric Bayesian networks (SPBN) and Linear Gaussian Bayesian networks (LGBN). Experimental results evaluate both goodness of fit and the ability to recover the underlying network structure. The results show that, in the presence of nonlinear Gaussian relationships, SLGBN achieves better fit and improved structure recovery compared to alternative models, particularly when combined with the Extended Bayesian Information Criterion (EBIC\*) score. Furthermore, by retaining Gaussian conditional distributions, SLGBN ofers computational advantages over fully semiparametric approaches.

Keywords: Bayesian Networks; Structure Learning; Penalized B-splines; Greedy hill-climbing

## 1 Introduction

Bayesian networks are probabilistic graphical models that provide a compact representation of joint probability distributions. Their graphical structure encodes conditional independence relationships among variables, making them a powerful and interpretable tool for modeling complex multivariate systems [17]. Their ability to combine probabilistic reasoning with interpretable graphical representations has led to their widespread adoption in data mining and decision-support applications [13]. Despite these advantages, Bayesian networks also present important limitations, particularly when dealing with continuous variables.

Inference and learning in Bayesian networks with continuous variables is considerably more challenging than in the discrete case. Exact inference typically requires restrictive assumptions on the form of the conditional distributions, and structure learning becomes computationally demanding due to the need to estimate complex likelihood functions. As a consequence, continuous variables are commonly discretized to simplify learning and inference. While discretization can be efective in some settings, it may lead to information loss and can alter relevant features of the original data distribution [18].

To overcome these limitations, a growing body of research has focused on extending Bayesian networks to model continuous variables. Among continuous Bayesian networks, Linear Gaussian Bayesian Networks (LGBN) constitute one of the most widely used approaches. In this framework, each variable is modeled as a Gaussian distribution whose mean is a linear function of its parents. This assumption greatly simplifies inference and parameter estimation, making LGBN computationally attractive. Although linear Gaussian models often provide a reasonable approximation to real world data, they rely on strong assumptions of linearity and normality that are frequently violated in practice.

More flexible alternatives have been recently proposed. In particular, Semiparametric Bayesian Networks (SPBN) allow conditional distributions to vary between linear Gaussian forms and nonparametric estimates [2], commonly obtained via kernel density estimation (KDE). Since their introduction, SPBNs have motivated several extensions, including hybrid formulations [3] or transfer learning approaches [23], highlighting their current interest. These models have shown improved performance in capturing nonlinear dependencies and deviations from Gaussianity. However, this increased flexibility comes at the cost of significantly higher computational complexity and more challenging inference procedures.

Despite these advances, an important modeling gap remains. There are important scenarios where variables are approximately Gaussian but the relationships among them are nonlinear. In such scenarios, LGBNs are often too restrictive because of their linear conditional models, whereas fully semiparametric approaches may introduce unnecessary complexity. This motivates the need for models that preserve the computational advantages of Gaussian Bayesian networks while allowing nonlinear conditional relationships. Recent work has explored this direction by incorporating nonlinear regression models, such as spline-based regressors, into Bayesian network learning procedures [22]. However, these approaches primarily employ nonlinear models as components of structure learning algorithms rather than defining a unified probabilistic framework for continuous Bayesian networks.

In this paper, we introduce Semi-Linear Gaussian Bayesian Networks (SLGBN), a novel class of continuous Bayesian networks that bridges the gap between linear Gaussian and fully semiparametric models. SLGBN provides a unified probabilistic framework that combines linear Gaussian and nonlinear Gaussian conditional models within the same network while preserving Gaussian conditional distributions. Nonlinear conditional means are modeled using Generalized Additive Models (GAMs) with penalized B-splines, providing greater flexibility while retaining computational advantages over fully semiparametric models. To make the proposed framework applicable in practice, we derive parameter estimation procedures for both linear and nonlinear Gaussian nodes and develop a greedy hill-climbing algorithm for simultaneously learning the network structure and node types. Finally, we evaluate SLGBN on synthetic Bayesian networks and compare its performance with LGBN and SPBN models in terms of both goodness of fit and structure recovery. The main contributions of this work are summarized as follows:

• We introduce SLGBN, a novel class of continuous Bayesian networks that bridges the gap between linear Gaussian and fully semiparametric models.

• We define a unified probabilistic framework combining linear and nonlinear Gaussian conditional models while preserving Gaussian conditional distributions.

• We derive parameter estimation procedures and develop a greedy hill-climbing algorithm for jointly learning network structure and node types.

• We experimentally show that, for approximately Gaussian variables exhibiting nonlinear dependencies, SLGBN achieves improved goodness of fit and structure recovery while maintaining a favorable balance between modeling flexibility and computational complexity.

## 2 Background

This section introduces Bayesian networks, along with linear Gaussian Bayesian networks (LGBN) and semiparametric Bayesian networks (SPBN), two continuous Bayesian network models, which serve as reference models for comparison with SLGBN. It then presents generalized additive models (GAM) and penalized B-splines, which form the basis for estimating nonlinear nodes in the SLGBN framework.

## 2.1 Bayesian Networks

A Bayesian Network $B = \left( { \mathcal { G } } , { \boldsymbol { \theta } } \right)$ consists of a Directed Acyclic Graph (DAG) $G = ( V , E )$ with node set $V = \{ 1 , \cdots , p \}$ and edge set E. A set of parameters θ defines the conditional probability distributions between them. For a set of variables $\mathbf { X } = \{ X _ { 1 } , . . . , X _ { p } \}$ , a Bayesian Network encodes its joint probability P(X). The distribution is factorized as $P ( \mathbf { X } ) = \prod _ { j = 1 } ^ { p } P ( X _ { j } \mid \mathbf { Y } _ { j } )$ , where $\mathbf { Y } _ { j }$ denotes the parent variables of $X _ { j }$ in $\mathcal { G }$

## 2.2 Linear Gaussian Bayesian Networks

A Linear Gaussian Bayesian Network (LGBN) is a specific kind of continuous Bayesian network in which all the CPD are linear Gaussians. Let $X _ { j }$ be a linear Gaussian of its parents $\mathbf { Y } _ { j }$

$$
P ( X _ { j } | \mathbf { Y } _ { j } ) = \mathcal { N } \left( \beta _ { j , 0 } + \beta _ { j } ^ { \top } \mathbf { Y } _ { j } , \sigma _ { j } ^ { 2 } \right) .\tag{1}
$$

Assume that $\mathbf { Y } _ { j }$ is jointly Gaussian with distribution $\mathcal { N } \left( \mu _ { \mathbf { Y } _ { j } } , \pmb { \Sigma } _ { \mathbf { Y } _ { j } } \right)$ . Then the distribution of $X _ { j }$ is Gaussian with

$$
\mathbb { E } [ X _ { j } ] = \beta _ { j , 0 } + \beta _ { j } ^ { \top } \pmb { \mu } _ { Y _ { j } } ,
$$

$$
\operatorname { V a r } ( X _ { j } ) = \sigma _ { j } ^ { 2 } + \beta _ { j } ^ { \top } \Sigma _ { Y _ { j } } \beta _ { j } ,
$$

and the joint distribution of $( X _ { j } , \mathbf { Y } _ { j } )$ is a multivariate Gaussian. Therefore, if B is a linear Gaussian Bayesian network, then it defines a joint distribution that is jointly Gaussian. Multivariate Gaussian distributions are closed under marginalization and conditioning, which guarantees the existence of closed-form expressions for all conditional and marginal distributions. This property is particularly advantageous for exact inference and structure learning in Bayesian networks.

## 2.3 Semiparametric Bayesian Networks

Semiparametric Bayesian Networks (SPBN) [2] extend traditional Bayesian networks by allowing a mixture of parametric and non-parametric conditional probability distributions (CPD).

For parametric nodes, the conditional distribution follows a linear Gaussian model as in LGBN. For non-parametric nodes, the conditional density is estimated using Conditional Kernel Density Estimation (CKDE), defined as:

$$
{ \hat { f } } ( X _ { j } \mid \mathbf { Y } _ { j } ) = { \frac { { \hat { f } } ( X _ { j } , \mathbf { Y } _ { j } ) } { { \hat { f } } ( \mathbf { Y } _ { j } ) } } ,\tag{2}
$$

where ${ \hat { f } } ( X _ { j } , \mathbf { Y } _ { j } )$ is estimated using kernel density estimation. This flexible formulation allows SPBN to capture both linear Gaussian dependencies and complex nonlinear relationships. However, the use of non-parametric density estimation significantly increases computational complexity and limits scalability in structure learning tasks.

## 2.4 Generalized Additive Models

Generalized Linear Models (GLM) extend the linear regression framework allowing for response variables from the exponential family. For a response variable $X _ { j }$ with parent vector $\mathbf { Y } _ { j }$ , a GLM assumes a linear predictor

$$
\eta _ { j } = \beta _ { j , 0 } + \beta _ { j } ^ { \top } \mathbf { Y } _ { j } ,\tag{3}
$$

together with a link function $g ( \mu _ { j } ) = \eta _ { j }$ and a variance function Var $( X _ { j } ) = \phi V ( \mu _ { j } )$ . Generalized Additive Models (GAM) [12] are extensions of GLM where the relation between η and the predictors is not considered linear. Instead, a smooth function $f ( \cdot )$ is used in place of the linear functions

$$
\eta _ { j } = \beta _ { j , 0 } + \sum _ { k = 1 } ^ { d _ { j } } f _ { j k } ( Y _ { j k } ) ,\tag{4}
$$

where $d _ { j } = | \mathbf { Y } _ { j } |$ is the number of parents of $X _ { j }$ and each $f _ { j k } ( \cdot )$ is a smooth function. This framework allows us to convert the linear dependencies of LGBN to nonlinear dependencies conditionally Gaussian distributions.

## 2.5 B-Splines

Splines are functions constructed from polynomial pieces joined at certain values of $x ,$ the knots. Let ${ \pmb { \xi } } = ( \xi _ { 0 } , \xi _ { 1 } , \dots , \xi _ { t } )$ be a non-decreasing knot sequence. Any spline in the spline space of order m associated with ξ can be expressed as a linear combination of B-spline basis functions of degree $p = m - 1 \left[ 9 \right]$

$$
s ( x ) = \sum _ { \ell = 1 } ^ { L } \beta _ { \ell } B _ { \ell , p } ( x ) ,\tag{5}
$$

where $L = t - p - 1$ is the number of basis functions and $B _ { \ell , p } ( x )$ are the B-spline basis functions of degree p. B-splines are linear in their coeficients $\beta _ { \ell }$ . A B-spline can be defined as the smoothing function used by the GAM. For an $X _ { j }$ node

$$
\eta _ { j } = \beta _ { 0 } + \sum _ { k = 1 } ^ { d _ { j } } s _ { k } ( x _ { k } ) .\tag{6}
$$

Using B-splines is a powerful approach for approximating functions but they tend to overfit data, especially when the number of knots is large [10]. For this reason, spline coeficients are typically estimated using a penalized criterion that controls smoothness through a penalty on the coeficients.

## 3 Semi-Linear Gaussian Bayesian Networks

We now introduce the proposed Semi-Linear Gaussian Bayesian Networks (SLGBN). Learning in Bayesian networks generally involves two main tasks: estimating the local model parameters and recovering the graph structure that defines the dependency relationships among variables. In the following sections, we formally define the SLGBN framework and present the corresponding methodology for parameter estimation and structure learning, including a greedy hill-climbing algorithm adapted to jointly learn both graph structure and node-type assignments.

## 3.1 Definition

An SLGBN is a continuous Bayesian Network B defined as a DAG $\mathcal { G } = ( V , E )$ where nodes are divided into two diferent types $\mathbf { V } = \{ \mathbf { X } _ { L } , \mathbf { X } _ { N } \}$ : linear Gaussian nodes $\left( \mathbf { X } _ { L } \right)$ and nonlinear Gaussian nodes $\left( { \bf X } _ { N } \right)$ . While linear Gaussian nodes are modeled using the well-known linear Gaussian densities (see Equation 1), the conditional mean of the nonlinear Gaussian nodes is modeled using a GAM with penalized B-splines, assuming an additive structure over the parent variables. The linearity/nonlinearity is a local property, defined with respect to the conditional dependence on each node’s direct parents, rather than at the level of global dependencies in the DAG. The model reduces to a Linear Gaussian Bayesian network when all nodes V are modeled as linear Gaussians.

## 3.2 Parameter Estimation

Let G be a fixed structure. For each node in the network, the parameters $\theta _ { j }$ are estimated from data. One common approach is to estimate the parameters that maximize the probability of the observed dataset. Let $\mathcal { D } = \{ X _ { 1 } , \cdot \cdot \cdot , X _ { p } \}$ be a dataset with samples $X _ { j } = \{ x _ { 1 } ^ { ( j ) } , \cdot \cdot \cdot , x _ { n } ^ { ( j ) } \}$ . The probability of a set of observations from node $X _ { j }$ with a set of parents $\mathbf { Y } _ { j }$ and a set of parameters $\theta _ { j }$ is simply the product of the assigned density to each data point. This is the likelihood of the node.

$$
{ \mathcal { L } } \left( X _ { j } \mid \theta _ { j } , { \mathcal { G } } \right) = \prod _ { i = 1 } ^ { n } f ( x _ { i } \mid \theta _ { j } , \mathbf { Y } _ { j } )\tag{7}
$$

The likelihood of the complete network can be obtained by multiplying the individual likelihoods of the nodes.

$$
\mathcal { L } \left( \mathcal { D } \mid \theta _ { \mathcal { G } } , \mathcal { G } \right) = \prod _ { j = 1 } ^ { p } \mathcal { L } \left( X _ { j } \mid \theta _ { j } , \mathcal { G } \right) ,\tag{8}
$$

and given that $\theta _ { j } \cap \theta _ { k } = \emptyset$ for all $j \neq k$ , individually optimizing $\theta _ { j }$ for each $j \in p$ is equivalent to maximizing $\mathcal { L } \left( \mathcal { D } \mid \theta _ { \mathcal { G } } , \mathcal { G } \right)$ . Usually this equation is converted to log-likelihood, as this can be expressed as a summation of terms instead of a product.

$$
\mathcal { L L } \left( \mathcal { D } \mid \theta _ { \mathcal { G } } , \mathcal { G } \right) = \sum _ { j = 1 } ^ { p } \sum _ { i = 1 } ^ { n } \log f \left( X _ { j } ^ { ( i ) } \mid \theta _ { j } , \mathcal { G } \right)\tag{9}
$$

It is common to estimate the parameters by maximizing the log-likelihood of the dataset. In SLGBN, nodes can be either linear or nonlinear, and may or may not have parents.

• Nodes with no parents.

The probability density of every node with no parents is defined as a univariate Gaussian. For a $X _ { j }$ node with no parents:

$$
P ( X _ { j } ) \sim N ( \mu _ { j } , \sigma _ { j } ^ { 2 } )\tag{10}
$$

where parameters are given by sample mean and variance. Under the normality assumption, these are the maximum likelihood parameters [15]. Let $X _ { j }$ be a node with a set of parents $\mathbf { Y } _ { j }$ . In this case, the conditional mean of $X _ { j }$ is modeled as a function of its parents $\mu _ { j | \mathbf { Y } _ { j } } = \mu ( \mathbf { Y } _ { j } )$

$$
P ( X _ { j } \mid \mathbf { Y } _ { j } ) \sim N ( { \boldsymbol \mu } ( \mathbf { Y } _ { j } ) , { \boldsymbol \sigma } _ { j } ^ { 2 } )\tag{11}
$$

This function is determined from the type of dependency between the node and its parents, which may be linear or nonlinear.

## • Linear Gaussians.

For a linear node $X _ { j }$ with parent variables $\mathbf { Y } _ { j }$ , we assume a Gaussian conditional distribution with linear mean:

$$
P ( X _ { j } \mid \mathbf { Y } _ { j } ) \sim { \mathcal { N } } ( \mu _ { j } ( \mathbf { Y } _ { j } ) , \sigma _ { j } ^ { 2 } ) , \mu _ { j } ( \mathbf { Y } _ { j } ) = \beta _ { 0 , j } + \beta _ { j } ^ { \top } \mathbf { Y } _ { j } .\tag{12}
$$

In matrix notation, the model is:

$$
\mathbf { X } = \mathbf { Z } { \pmb \theta } + \varepsilon , \quad { \pmb \varepsilon } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { n } ) ,\tag{13}
$$

where Z is the design matrix including a column of ones for the intercept, and θ collects all coeficients. Under Gaussian assumption, the ordinary least squares (OLS) estimator of θ is equivalent to the MLE estimator [15].

$$
\hat { \pmb { \theta } } = ( { \bf Z } ^ { \top } { \bf Z } ) ^ { - 1 } { \bf Z } ^ { \top } { \bf X } , \quad \hat { \pmb { \varepsilon } } = { \bf X } - { \bf Z } \hat { \pmb { \theta } } ,\tag{14}
$$

The variance is estimated as

$$
\hat { \sigma } ^ { 2 } = \frac { \hat { \pmb { \varepsilon } } ^ { \top } \hat { \pmb { \varepsilon } } } { n } .\tag{15}
$$

## • Nonlinear Gaussians.

For a nonlinear node $X _ { j }$ with parents $\mathbf { Y } _ { j }$ , the conditional mean is modeled as an additive function of the parents using cubic B-splines:

$$
P ( X _ { j } \mid \mathbf { Y } _ { j } ) \sim { \mathcal { N } } ( \mu _ { j } ( \mathbf { Y } _ { j } ) , \sigma _ { j } ^ { 2 } ) , \quad \mu _ { j } ( \mathbf { Y } _ { j } ) = \beta _ { 0 , j } + \sum _ { k = 1 } ^ { d _ { j } } s _ { j k } ( y _ { j k } ) .\tag{16}
$$

Each smooth function $s ( \cdot )$ is expressed as a linear combination of cubic B-spline bases. The number of bases is fixed a priori and defines the number of knots and the maximal flexibility of the spline. In matrix form:

$$
\mathbf { X } = \mathbf { Z } { \pmb \theta } + \varepsilon , \quad { \pmb \varepsilon } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { n } ) ,\tag{17}
$$

where Z contains the evaluated spline basis (plus intercept) and θ the corresponding coeficients. As commented in Section 2.5, the spline can tend to overfit if its smoothness is not controlled. For this reason, a penalization over the smoothness of the splines is included through a penalization

parameter λ. This penalization shifts the parameter estimation problem from finding the MLE parameters to finding the parameters maximizing the penalized log-likelihood.

$$
\operatorname* { m a x } \mathcal { L } \mathcal { L } _ { p e n } \left( \mathcal { D } \mid \theta _ { \mathcal { G } } , \mathcal { G } \right) \ = \operatorname* { m a x } \{ \mathcal { L } \mathcal { L } \left( \mathcal { D } \mid \theta _ { \mathcal { G } } , \mathcal { G } \right) - \frac { 1 } { 2 } \lambda \theta ^ { \top } \mathbf { S } \theta \} ,\tag{18}
$$

where $\mathbf { S }$ is a penalty matrix (e.g., based on diferences of coeficients or derivatives of the spline) defined over all the coeficients of the node. The parameters optimizing this penalized likelihood are obtained through a penalized least squares (PLS) approach.

$$
\hat { \pmb { \theta } } = ( \mathbf { Z } ^ { \top } \mathbf { Z } + \lambda \mathbf { S } ) ^ { - 1 } \mathbf { Z } ^ { \top } \mathbf { X } , \quad \hat { \pmb { \varepsilon } } = \mathbf { X } - \mathbf { Z } \hat { \pmb { \theta } } .\tag{19}
$$

The parameter λ will control the smoothness of the spline, moving from a traditional OLS approach when $\lambda = 0$ , to a smoother function as the value of λ increases. The penalization only afects the estimation of the conditional mean function, and the variance is estimated using an MLE approach (see Equation 15), as in the linear case.

## 3.3 Structure Learning

The structure learning problem consists of identifying, from a dataset D, the DAG G that best represents the dependency structure among the variables. Existing approaches can be broadly categorized into constraint-based, score-based, and hybrid methods [16]. In this work, we focus on the score-based paradigm. In score-based structure learning, this task is formulated as an optimization problem in which each candidate structure $\mathcal { G }$ is assigned a score S measuring its goodness of fit to the data. The objective is to find the structure that maximizes this score.

$$
\operatorname* { m a x } _ { \mathcal { G } } \mathcal { S } \left( \mathcal { G } : \mathcal { D } \right)\tag{20}
$$

Since finding the globally optimal structure under standard scoring functions is a NP-hard problem [6], practical algorithms rely on heuristic search strategies. These methods explore the space of DAG through local modifications of an initial structure, selecting changes that improve the score. One common selection for the score function is the likelihood (see Equation 8). Usually, this is developed by finding, from candidate structures, the set $( \mathcal { G } , \pmb { \theta } )$ that maximizes the likelihood of data. This is obtained by finding first the MLE parameters for each candidate structure and then selecting the structure $\mathcal { G }$ with the maximum log-likelihood.

$$
\operatorname* { m a x } _ { \mathcal { G } , \theta } \mathcal { L } \mathcal { L } \left( \left\{ \mathcal { G } , \theta \right\} : \mathcal { D } \right) = \operatorname* { m a x } _ { \mathcal { G } } \mathcal { L } \mathcal { L } \left( \left\{ \mathcal { G } , \hat { \theta } _ { M L E } \right\} : \mathcal { D } \right)\tag{21}
$$

In SLGBN, the parameters of nonlinear nodes do not correspond exactly with MLE parameters but with those that maximize the penalized likelihood. Therefore, the problem shifts to finding the structure that maximizes the likelihood for the estimated parameters $\bar { \hat { \boldsymbol { \theta } } } .$

$$
\operatorname* { m a x } _ { \mathcal { G } } \mathcal { L } \mathcal { L } \left( \{ \mathcal { G } , \hat { \pmb { \theta } } _ { \mathcal { G } } \} : \mathcal { D } \right)\tag{22}
$$

However, likelihood-based learning tends to overfit and typically favors overly dense graphs. To control model complexity, regularized scores such as the Bayesian Information Criterion (BIC) are commonly used:

$$
\operatorname* { m a x } _ { \mathcal { G } } S _ { B I C } ( \mathcal { G } : \mathcal { D } ) = \operatorname* { m a x } _ { \mathcal { G } } \left( \mathcal { L } \mathcal { L } ( \{ \mathcal { G } , \hat { \pmb { \theta } } _ { \mathcal { G } } \} : \mathcal { D } ) - \frac { k } { 2 } \cdot \log n \right) ,\tag{23}
$$

where n denotes the sample size and k the total number of free parameters. The BIC penalty is local and decomposable across nodes, so for a network with $p$ nodes, the total value for k will correspond with the total sum of parameters of all the individual nodes in the network $k = \textstyle \sum _ { j } ^ { p } k _ { j }$ This property allows eficient evaluation and updates during structure learning, since each node contributes independently to the total score. The total number of free parameters of each node depends on its type and its parents. Linear nodes and nodes without parents will contribute $d _ { j } + 2$ parameters each, where $d _ { j }$ corresponds with the number of parents. For nonlinear nodes, the number of efective parameters is given by edf+ 1, where edf accounts for the complexity of the spline-based conditional mean. Following calculations in Equation 19, edf is defined as

$$
\mathbf { H } = \mathbf { Z } ( \mathbf { Z } ^ { \top } \mathbf { Z } + \lambda \mathbf { S } ) ^ { - 1 } \mathbf { Z } ^ { \top } , \quad \mathrm { e d f } = \operatorname { t r } ( \mathbf { H } ) .\tag{24}
$$

The additional parameter corresponds to the dispersion parameter $\sigma _ { j } ^ { 2 }$ of the conditional distribution. The extended BIC (EBIC) [4] extends the classical BIC criterion by introducing an additional penalty on model complexity. A related formulation has been proposed for Gaussian graphical model selection, where the global penalty is defined over graph density and explicitly accounts for the size of the edge space [11]. Such additional regularization can be particularly useful in Bayesian network structure learning, where the number of candidate DAG grows super-exponentially with the number of variables. In this work, we adopt a variant of this formulation (referred to here as EBIC\*), in which the local penalty is defined in terms of the number of free parameters $k ,$ as in the standard BIC formulation, while the global penalty depends on the total number of edges |E|.

$$
\operatorname* { m a x } _ { \mathcal { G } } S _ { E B I C ^ { * } } ( \mathcal { G } : \mathcal { D } ) = \operatorname* { m a x } _ { \mathcal { G } } \left( \mathcal { L } \mathcal { L } ( \{ \mathcal { G } , \hat { \theta } _ { \mathcal { G } } \} : \mathcal { D } ) - \frac { k } { 2 } \log n - 2 \gamma | \mathbf { E } | \log p \right) .\tag{25}
$$

This formulation allows a more flexible penalization of model complexity by assigning higher cost to nodes with richer parameterizations, which better reflects the heterogeneous nature of SLGBN. As with BIC, the likelihood and local penalty terms in $\mathrm { E B I C ^ { * } }$ remain decomposable across nodes. Only the additional graph-structure penalty is global, depending on the total number of edges, and can still be updated eficiently during structure learning.

Overall, these two criteria represent standard choices for structure learning in Bayesian networks. In general, other scoring functions that trade of data fit and model complexity can also be used within this framework.

## 3.3.1 Identifiability and Score Equivalence

A Markov equivalence class (MEC) consists of all DAG that encode the same set of conditional independence relations over a set of variables, and is uniquely characterized by a common skeleton and identical v-structures. Within this framework, two DAG $G _ { 1 }$ and $G _ { 2 }$ are equivalent, denoted $G _ { 1 } \sim G _ { 2 }$ , if they belong to the same MEC. A scoring function S is said to be score-equivalent if it assigns identical values to all DAG within the same equivalence class, that is, if $G _ { 1 } \sim G _ { 2 }$ implies $S ( G _ { 1 } ) = S ( G _ { 2 } )$ [8]. This property ensures that observationally indistinguishable DAG cannot be separated by the score.

In classical LGBN, homogeneous local parameterizations imply likelihood equivalence across Markov equivalent structures. As a result, scores such as BIC and $\mathrm { E B I C ^ { * } }$ are score-equivalent under standard assumptions for likelihood-based estimation. In particular, the additional graphcomplexity term in $\mathrm { E B I C ^ { * } }$ depends only on the number of edges, which remains invariant within a MEC. Consequently, the optimal score-based estimator is asymptotically consistent at the MEC level rather than at the level of a unique DAG [8, 5].

In contrast, both SPBN and SLGBN introduce heterogeneous local parameterizations. Since diferent DAG may induce diferent combinations of local functional forms, score equivalence may no longer hold across Markov equivalent structures. Consequently, observationally equivalent DAG can receive diferent scores, and recovery of the exact generating DAG cannot in general be guaranteed from observational data alone [14].

For this reason, evaluation of learned structures is primarily performed using MEC-level or skeleton-based metrics [7], which better reflect the intrinsic identifiability limitations of the problem.

## 3.3.2 Greedy Hill-Climbing for SLGBN.

We define the structure learning procedure for SLGBN as a variant of the tabu greedy hill-climbing algorithm proposed by Atienza et $a l .$ for SPBN [2]. The corresponding pseudocode is given by Algorithm 1. The objective is to find the best set of edges and node-type assignments for a given dataset by maximizing a user-defined scoring function $S ,$ which guides the structure learning process. At each iteration, candidate structures are generated through local modifications of the current graph $\mathcal { G } _ { \mathrm { c u r r e n t } }$ and evaluated according to $S ,$ selecting the highest-scoring admissible move subject to an acyclicity constraint. We also enforce a maximum parent constraint $( d _ { m a x } )$ during the search. This restriction is a common assumption in score-based structure search to reduce the search space and avoid overfitting [17]. Given the structure in each iteration $G _ { c u r r e n t } , \mathcal { O } _ { v a l i d } \subseteq \mathcal { O }$ will define the set of allowed operators.

Algorithm 1 Greedy Hill-Climbing Algorithm for SLGBN   
Require: Dataset D, initial graph ${ \overline { { \mathcal { G } _ { 0 } } } } .$ , operator set O, score $S ,$ patience $\eta ,$ size of tabu list $t ,$   
number of B-spline bases l, penalization parameter λ, max. number of parents $d _ { m a x }$   
1: $\mathcal { G } _ { b e s t }  \mathcal { G } _ { 0 }$   
2: $\mathcal { G } _ { c u r r e n t }  \mathcal { G } _ { 0 }$   
3: $\tau  \emptyset$   
4: $i \gets 0$   
5: while $i < \eta$ do   
6: $\begin{array} { r } { \mathcal { O } _ { v a l i d }  \{ o \in \mathcal { O } _ { \mathcal { G } _ { c u r r e n t } } : o \notin \mathcal { T } , \operatorname* { m a x } _ { i } | \mathrm { d } _ { \mathcal { G } _ { c u r r e n t } ^ { o } } ( X _ { i } ) | \leq d _ { \operatorname* { m a x } } , \mathcal { G } _ { c u r r e n t } ^ { o } \mathrm { a c y c l i c } \} ^ { * } } \end{array}$   
7: $o ^ { * }  \arg$ max $S ( \mathcal { D } , o ( \mathcal { G } _ { c u r r e n t } ) , l , \lambda )$   
$\mathsf { \Pi } _ { o \in \bigcup \sigma _ { v a l i d } }$   
8: $\mathcal { G } _ { c u r r e n t }  o ^ { * } ( \mathcal { G } _ { c u r r e n t } )$   
9: if $S ( \mathcal { D } , \mathcal { G } _ { c u r r e n t } , l , \lambda ) > S ( \mathcal { D } , \mathcal { G } _ { b e s t } , l , \lambda )$ then   
10: $\mathcal { G } _ { b e s t }  \mathcal { G } _ { c }$ urrent   
11: $i \gets 0$   
12: else   
13: $i  i + 1$   
14: end if   
15: ${ \mathcal { T } } \gets { \mathcal { T } } \cup \{ ( o ^ { * } ) ^ { \top } \}$   
16: if $\lvert \tau \rvert > t$ then   
17: remove oldest element from $\tau$   
18: end if   
19: end while   
20: return $\mathcal { G } _ { b e s t }$   
21: <sup>∗</sup> The operator o is valid only if $\mathcal { G } _ { c u r r e n t } ^ { o }$ is acyclic.

The search starts from an initial structure $\mathcal { G } _ { 0 }$ , typically defined as the empty graph with all nodes assigned to the linear type. A predefined set of operators O determines the local changes that can be applied to generate the candidate structures. In hill-climbing methods, these operators usually include actions like the addition, removal or reversal of edges. In the context of Semi-Linear Gaussian Bayesian Networks (SLGBN), this framework is extended by allowing the algorithm to switch between linear and nonlinear nodes. Following [2], we include a TYPE\_CHANGE operator alongside the classical ADD, REMOVE, and REVERSE operators.

For every candidate structure, model parameters are estimated, and the score is computed. Linear nodes are fitted using linear regression, and nonlinear nodes are estimated using GAM with penalized B-splines. The number of B-spline bases for each nonlinear node l, and the penalization parameter λ, are defined a priori and remain fixed during structure learning. Nodes without parents are automatically modeled using sample mean and variance, independently of their assigned type. Since their type will not afect their density, the type change operator is not available in these situations. Nodes will retain their node type until they get a new parent (see Figure 1).

![](images/315e662d16e9e09f99736bc0eaf0f2a76dc514ccbb7db08f4457f7e3c728fd1e.jpg)

![](images/36dcde61323f2e36dc010714a3a124c715e9c626e1576f9be4823cc3a680bb6b.jpg)

![](images/fe1f0c33b879931c47cedfb4c764c1db1ab1e9fb671e084202d50b6b5550d612.jpg)

![](images/93942c369a1ab7a0b2abd698c236b6ee64558da96ac84b10fc6b0184359c3464.jpg)  
Figure 1: Greedy Hill-Climbing: The figure shows four steps of the Greedy Hill-Climbing Algorithm and changes in the estimation of $X _ { 2 }$ . Yellow nodes represent nonlinear nodes and gray nodes represent linear nodes. Changes in arcs are represented with red arrows and changes in types are represented by red contours. Step 1 starts with a linear $X _ { 2 }$ node depending on $X _ { 1 }$ . In step 2 node type is changed from linear to nonlinear and $X _ { 2 }$ is estimated using B-splines. In step 3 arc $\left( X _ { 1 } , X _ { 2 } \right)$ is reversed and $X _ { 2 }$ loses its only parent. Its density is at this point estimated with sample mean but the node type is still nonlinear. Note that at this point type\_change operator is not available for $X _ { 2 }$ as it has no parents. In step 4 $X _ { 2 }$ gains a new parent after adding $( X _ { 3 } , X _ { 2 } )$ edge and is again nonlinearly estimated with B-splines, but now depending on $X _ { 3 }$

The algorithm will navigate over the candidate graph structures until the score can no longer be improved. In order to escape from local maxima, a patience parameter η is introduced. This parameter will determine the number of consecutive iterations during which the algorithm is allowed to accept lower-scoring structures before early-stopping. In this period, the best-scoring structure is saved in cache and returned when patience limit is exceeded. This mechanism may lead to cyclic behavior, in which an operator and its inverse are repeatedly applied. To prevent this, a tabu list T is maintained: the last t movements are stored and temporarily forbidden, discouraging immediate reversals and promoting exploration of the search space.

Despite these mechanisms, the proposed procedure remains a greedy local search strategy over the space of DAG. As a consequence, even if the underlying structure (or its equivalence class) is theoretically identifiable under a consistent scoring function, the algorithm is not guaranteed to recover it in practice, since it only explores a restricted neighborhood of the search space at each iteration. Therefore, the final solution may correspond to a local optimum of the scoring landscape rather than the globally optimal structure.

## 3.3.3 Asymptotic Time Complexity

In this section we analyze the computational complexity of the Hill Climbing algorithm for SLGBN and we compare it with SPBN and LGBN. The asymptotic time complexity of the algorithm depends basically on two steps: (i) the event generation and (ii) the event evaluation.

At each iteration, the algorithm considers candidate modifications to the network structure. In SLGBN, with all operators allowed and p nodes, up to $p ^ { 2 }$ operations for each edge operator and p node type changes are evaluated. This reduces the asymptotic complexity of event generation step to $O ( p ^ { 2 } )$

Each event requires the parameter estimation and the calculation of the score. The complexity of parameter estimation depends on the node type. In nodes with no parents, the parameters are simply estimated as sample mean and variance. With n samples in the training data, the complexity is defined as ${ \mathcal { O } } ( n )$ . For linear nodes, the parameters are estimated through an OLS multiple linear regression. In this case, the complexity increases quadratically depending on the number of parents $\mathcal { O } ( n d ^ { 2 } )$ [2]. The complexity of estimating nonlinear nodes depends on two steps, the spline construction and the model fitting. Being d the number of parents and l the number of spline bases, the former requires $\mathcal { O } ( n d l )$ , while the latter requires $\mathcal { O } ( n ( d l ) ^ { 2 } )$ , which dominates. Therefore, in the worst case where all nodes are estimated as nonlinear nodes, parameter estimation scales as $\mathcal { O } ( n ( d l ) ^ { 2 } )$ . With p nodes, this should be multiplied by $p , \mathcal { O } ( p n ( d l ) ^ { 2 } )$ . In practice, for decomposable scores such as BIC, in each event only afected nodes are recalculated, which means recomputing only 1 or 2 nodes, so the p term can be removed. The local term of EBIC\* corresponds to the BIC score and is therefore decomposable, with only the afected nodes being updated. The global term, which depends on $| E |$ , is updated in constant time without afecting the overall complexity. Therefore, the score evaluation per event can be performed in $O ( n )$ , without changing the overall complexity. Combining the number of events and the per-event cost, the asymptotic complexity order of each iteration is defined as $\mathcal { O } ( p ^ { 2 } n ( d l ) ^ { 2 } )$ . For K iterations, the total complexity is $\mathcal { O } ( K p ^ { 2 } n ( d l ) ^ { 2 } )$ . In practice, as d and l are bounded, this simplifies to:

$$
\mathbf { S L G B N } : { \mathcal { O } } ( K p ^ { 2 } n ) .
$$

In the linear Gaussian setting, node-type changes are not considered, and only $p ^ { 2 }$ structural events are evaluated. Parameter estimation is performed via OLS, with cost $\mathcal { O } ( n d ^ { 2 } )$ per node. As only afected nodes are re-estimated and d is bounded in practice, the per-event complexity reduces to $O ( n )$ . The score remains local and does not alter the asymptotic cost. Therefore, the total complexity is:

$$
\mathbf { L G B N } : { \mathcal { O } } ( K p ^ { 2 } n ) .
$$

For SPBN, nonlinear nodes are estimated using kernel density estimation (KDE). The event generation process is analogous to SLGBN (except for specific nodes with forbidden type change in SLGBN), yielding an asymptotic complexity of $\mathcal { O } ( p ^ { 2 } )$ in selecting candidate events per iteration. The cost of KDE-based parameter estimation is $\mathcal { O } ( n ^ { 2 } d ^ { 2 } )$ [2]. Assuming bounded $d ,$ this simplifies to $O ( n ^ { 2 } )$ . Thus, the total complexity is:

$$
\mathbf { S P B N } : \mathcal { O } ( K p ^ { 2 } n ^ { 2 } ) .
$$

Both SLGBN and LGBN scale linearly with the number of samples n, whereas SPBN exhibit quadratic scaling, making them more computationally expensive for large datasets. Additionally, the rate of convergence of nonparametric approaches like kernel density estimation is much slower than parametric settings and typically require suficiently large sample sizes to provide reliable estimates [19].

## 4 Methods

Our goal is to evaluate the proposed Semi-Linear Gaussian Bayesian Network (SLGBN) model and assess its ability to represent data accurately, identify nonlinear node relationships, and recover the underlying graph structure. To this end, we compare SLGBN against alternative Bayesian network models. In this section, we first introduce the synthetic datasets employed in the analysis, followed by the considered models. We then describe the structure learning procedure, including the optimization algorithm and its configurations, as well as the evaluation metrics used to assess both predictive performance and structural recovery. The complete implementation of all experiments is publicly available at https://github.com/digital-medicine-research-group-UNAV/SLGBN, ensuring full reproducibility of the results.

## 4.1 Data

To evaluate the ability of the SLGBN model to fit nonlinear Gaussian data, we implemented it on networks of varying structure and complexity. We considered three synthetic networks. While the structure of the first two networks (B5 & B8) were manually developed for this paper, the third network (B\_insu) is the insurance network extracted from the bnlearn network repository [20]. Table 1 shows the main characteristics of the three networks. The structure and parametrization of the first two networks can be found in Figure 2 and Figure 3. The third network can be found in Figure 6, in the appendix. While B\_insu is thought to be parametrized using discrete distributions, we developed an alternative continuous parametrization, in which linear Gaussian and nonlinear Gaussian relationships are combined additively. This parametrization can also be found in the appendix (Table 2). Obviously, the original semantic meaning of the variables is no longer preserved, as the parametrization was designed ad hoc for experimental purposes. Nevertheless, this is not relevant for our study, since our goal is to assess whether the model is able to correctly recover the underlying network structure, rather than to maintain the original interpretation of the variables.

We generated diferent samples from these networks and used the resulting data to recover the original network structure. The underlying hypothesis is that, if the SLGBN model works correctly, it should achieve a good fit to the observed data and, in favorable cases, recover a structure that is close to the true underlying graph. Four training sample sizes were considered for each network (200, 1000, 5000 and 10000), and 20 independent training datasets were generated for each sample size using diferent random seeds. Structure learning was then performed using diferent models and scoring functions, and the resulting networks were evaluated on an independent validation set $( \mathcal { D } _ { \mathrm { v a l } } )$ of 1000 samples.

## 4.2 Models

We tested three diferent models: Linear Gaussian Bayesian Networks (LGBN), Semiparametric Bayesian Networks (SPBN) and Semi-Linear Gaussian Bayesian Networks (SLGBN). In LGBN and

Table 1: Synthetic Networks.
<table><tr><td>Network</td><td>Number of nodes</td><td>Number of edges</td><td>Source</td></tr><tr><td>B5</td><td>5</td><td>5</td><td>Own elaboration</td></tr><tr><td>B8</td><td>8</td><td>12</td><td>Own elaboration</td></tr><tr><td>B_Insu</td><td>27</td><td>52</td><td>bnlearn</td></tr></table>

Table notes. Characteristics of synthetic networks.

![](images/61ec246c80d67592e408a5a84082a5360f10bfbcd50c1ea526fee94beb197497.jpg)

Figure 2: B5 network. First synthetic network structure and node probability distributions.  
![](images/08d4c93f54a2f9fef4693446c2093b8d9a5be25e148afcd0c420c27a081c4361.jpg)  
Figure 3: B8 network. Second synthetic network structure and node probability distribution.

SLGBN, structure learning was guided independently by the BIC and EBIC\* scores introduced in Section 3.3. For the EBIC\*, the sparsity parameter was defined at $\gamma = 0 . 7 5$ . Since SPBN relies on kernel-based conditional distributions, standard information criteria such as BIC or EBIC\* cannot be directly applied for structure learning; instead, the learning procedure was guided by crossvalidated log-likelihood. Overall, this resulted in the evaluation of five learning configurations: LGBN-BIC, LGBN-EBIC\*, SPBN-CVL, SLGBN-BIC and SLGBN-EBIC\*.

## 4.3 Algorithm

For SLGBN, we applied two variants of the Greedy Hill-Climbing (GHC) algorithm introduced in Section 3.3. In the first variant, we initialize the search from the true network structure and restrict the algorithm to the TYPE\_CHANGE operator, preventing edge modifications. This setup allowed us to evaluate whether nonlinear nodes can be correctly identified when the underlying network structure is known. In the second variant, we started from an empty network and executed GHC allowing all operators. This scenario evaluated the model’s ability to learn the full network structure from scratch. If SLGBN accurately represents the original relationships among variables, it is expected to achieve a better fit and a more faithful reconstruction of the true dependency structure than other models. LGBN was also executed using the SLGBN learning algorithm, forcing all nodes to be linear. In both cases, during the executions, the patience parameter (η) was defined at 5 and the size of the tabu list (t) was specified as 5. In SLGBN models, the probability distribution of nonlinear nodes was estimated using the pyGAM [21] package, with the number of bases set to 6 and a penalization parameter of $\lambda = 0 . 5$ . SPBN networks were implemented using the PyBNesian package [1], as described in the paper [2]. Every model started with all its nodes defined as linear.

## 4.4 Evaluation Metrics

For each model-sample size combination, we report the goodness of fit measured by the log-likelihood in the validation dataset (including its diference with respect to the ground truth) and the distance between the estimated structure and the true configuration using diferent metrics. We evaluated the learned structures using metrics based on the skeleton of the graph, comparing the skeleton of the learned DAG with that of the true DAG. Results are averaged across the 20 seeds for each sample size.

## • Log-Likelihood

The Greedy Hill-Climbing algorithm converges to a structure G and a set of parameters θ<sup>ˆ</sup>. The goodness of fit of this structure to data is measured through the log-likelihood value in the validation set: max<sub>G</sub> $\mathcal { L } \mathcal { L } \left( \{ \mathcal { G } , \hat { \pmb { \theta } } \} : \mathcal { D } _ { v a l } \right)$ . Although structure learning is guided by diferent score functions depending on the model, the resulting structures are all evaluated on the validation set using the pure log-likelihood, ensuring comparability across models.

## • Structural Distance

Structural diferences between the learned and true graphs are quantified using a skeleton-based Structural Hamming Distance (SHD) [24], computed on the undirected versions of the DAG. This metric measures the number of edge additions and deletions required to transform the learned skeleton into the true skeleton, ignoring edge orientation. This choice is consistent with the identifiability limitations discussed in Section 3.3.1, where multiple DAG may encode the same conditional independence structure. In SLGBN and SPBN models, not only the graph structure is learned, but also the functional type associated with each node. Accordingly, the Type Hamming Distance (THD) is also calculated, defined as the number of nodes whose type is incorrectly inferred. In SLGBN, computing the THD requires special consideration for nodes without parents, which are always modeled linearly. If a node loses its parents after being labeled as nonlinear (e.g., Figure 1-(3)), it becomes a node labeled as nonlinear but modeled linearly. Such nodes keep their nonlinear characteristics, which will be reflected if they later have parents. Therefore, when computing THD in SLGBN, nodes are evaluated based on their assigned type rather than their efective parametrization.

To further analyze diferent learning patterns, we compute the precision and recall of the learned structures for each network. Precision is defined as the proportion of edges in the learned network that are correct, whereas recall corresponds to the proportion of edges in the true underlying structure that are successfully recovered. Formally, precision is given by $P r = \mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } )$ and standard recall by $R e = \mathrm { T P / ( T P + F N ) }$ , where TP, FP, and FN denote true positives, false positives, and false negatives, respectively.

## 5 Results

This section presents the results of the learning procedures for each model in the selected evaluation metrics. All detailed numerical values can be found in the Appendix (Tables 3–8). Figure 4 and Figure 5 show the evolution of metrics in fixed and full approaches, respectively.

## 5.1 Fixed structure learning

For fixed-structure learning, we report the log-likelihood and the structural distance associated with node types (THD). SLGBN clearly achieves the best results, approaching the ground-truth likelihood as the sample size increases and obtaining the lowest THD values. For the two simpler networks, SLGBN correctly identifies the true types in all cases. For the more complex network, THD decreases steadily with increasing sample size, eventually converging to zero for the largest dataset. SPBN also substantially improves the LGBN model, showing similar trends but slightly worse performance than SLGBN. In LGBN, no type change is allowed, so the THD value is equal to the number of nonlinear nodes in the network. No noticeable diferences are observed between the BIC and EBIC\* score functions in any case, suggesting that both scores likely converge to the same structure in this setting.

![](images/89a92ac61a02d33198c8af6be56929838b217116575365a390129f42c8755cef.jpg)  
Figure 4: Fixed metrics. Evolution of the metrics in structure learning with fixed structure for diferent model-score configurations. Values represent the mean across all random seeds for each sample size. Rows correspond to metrics: a) Log-likelihood, b) THD (Lower is better). Columns correspond to networks: B5 (left), B8 (center), and B\_insu (right). X axis shows sample sizes. Y axis represents metric values, being common for THD.

## 5.2 Full structure learning

In the full structure learning setting, we evaluate model performance using log-likelihood and structural metrics, including THD, SHD, precision, and recall. Log-likelihood follows a pattern similar to that observed under fixed-structure learning. Its value increases with sample size, progressively approaching the ground truth values. The best overall performance is consistently achieved by the SLGBN, followed by the SPBN and LGBN. Small diferences emerge between the use of BIC and EBIC\*, although the resulting structures in both cases still produce a very similar data fit.

Converging to the original structure is more challenging than in fixed-structure learning, as models must recover both node types and graph structure from data. In general, SLGBN provides the most consistent performance across all metrics, particularly when combined with EBIC\*. A clear pattern emerges across all models: as sample size increases, recall tends to improve, while precision decreases, especially in more complex networks. This indicates that larger datasets allow models to recover more true edges, but at the cost of introducing additional false positives. This trade-of is reflected in the SHD, which does not necessarily decrease with sample size and can even increase in some settings. The EBIC\* score partially mitigates this efect, achieving higher precision while maintaining competitive recall, especially in larger sample sizes. As a result, SLGBN-EBIC\* consistently outperforms other models in terms of SHD.

## 5.3 Discussion on results

The observed trends show a clear divergence between log-likelihood (LL) and structural accuracy as measured by SHD. Across all settings, LL increases with sample size, which is expected since the learning procedure is explicitly driven by likelihood-based scoring functions such as cross-validated LL, BIC or EBIC\*. Structural recovery, measured through SHD, is not directly optimized. Consequently, SHD does not necessarily improve with increasing sample size and, in some cases, even deteriorates.

In the fixed structure setting, no diferences are observed between BIC and EBIC\* across all experiments. This behavior is expected because the graph structure is fixed and therefore the global complexity penalty introduced by EBIC\* remains constant for all candidate models. As a consequence, both score functions efectively optimize the same objective and tend to converge to identical solutions.

For full structure learning, exhaustive evaluation of the DAG space is computationally infeasible, so the increasing pattern of SHD with sample size may emerge. In practice, GHC provides only a tractable local approximation to the global optimization problem and may converge to a locally optimal solution due to its restricted search neighborhood. Even under an exhaustive search over all structures, score-equivalence and other identifiability limitations (as discussed in Section 3.3.1) prevent guarantees of recovering the true generating DAG from observational data alone. In addition, identifiability, when it holds, is asymptotic in nature and may not be applicable finite-sample regimes.

Within this framework, the increase in SHD can be explained by a trade-of between improved recovery of true edges and the inclusion of additional false positives as sample size increases. This efect becomes more pronounced in more complex networks, where the number of weak dependencies is higher. The global complexity penalty introduced by EBIC\* helps reduce the number of spurious edges, leading to a better balance between precision and recall and therefore improving SHD in many cases. Nevertheless, this efect does not fully reverse the overall trend.

Considering these limitations, SLGBN combined with EBIC\* provides the most robust performance across the considered settings, achieving the best compromise between likelihood maximization and structural accuracy. In this context, the learned models consistently achieve a strong fit to the observed data. Although recovering the true underlying DAG is not expected in general due to identifiability limitations, the estimated structures remain informative and practically useful. In particular, they capture a substantial portion of the true dependency structure and correctly identify many of the relevant conditional relationships between variables. As a result, the learned networks provide a meaningful and structured representation of the data-generating process.

## 6 Conclusion

In this work, we introduced Semi-Linear Gaussian Bayesian Networks (SLGBN), a flexible extension of Linear Gaussian Bayesian Networks (LGBN) that maintains conditional Gaussian distributions while allowing nonlinear relationships. We also introduced a structure learning algorithm for this model and evaluated its ability to fit nonlinear Gaussian data, comparing it with other approaches such as SPBN and LGBN. Finally, we presented and discussed the results of these experiments.

![](images/68aa2c101997d90b575bff59671cab0dea8d6ef51f7329abeaf9708c11a97b06.jpg)  
Figure 5: Full metrics. Evolution of metrics in full structure learning for diferent model-score configurations. Values represent the mean across all random seeds for each sample size. Rows correspond to metrics: a) Log-likelihood, b) THD (Lower is better), c) SHD (Lower is better), d) Precision, and e) Recall. Columns correspond to networks: B5 (left), B8 (center), and B\_insu (right). X axis shows sample sizes. Y axis represents metric values, with common axis by network for all metrics except log-likelihood.

The empirical results showed that SLGBN achieved strong goodness of fit and improved recovery of both node types and network structure, especially in combination with the EBIC\* score.

Performance was especially favorable when the underlying network structure was known, highlighting the efectiveness of the proposed estimation procedures under correct structural assumptions. However, the problem became increasingly challenging as network size grew, a limitation shared by all considered models.

From a practical perspective, the current implementation relied on hyperparameters specified a priori, such as the number of spline basis functions and the penalization parameter. While this choice ensured computational tractability, optimal performance would likely require node-specific tuning of these parameters. Such tuning could be achieved via grid search procedures but at the cost of a substantial increase in computational complexity. This approach may be feasible when the network structure is known or when dealing with small-scale networks, but it would quickly become prohibitive in larger or more complex settings. This highlights the importance of more eficient hyperparameter selection strategies to improve performance without incurring excessive computational cost.

Overall, in our experiments, SLGBN ofered a promising balance between flexibility and computational eficiency, combining the advantages of both fully parametric and semiparametric Bayesian network models. However, its performance was obtained under the assumption of conditional Gaussianity and additive dependencies between parent variables, which together define the scope of the proposed framework. Our results indicate that the model behaves as expected under these assumptions, although its robustness under non-Gaussian settings or non-additive dependency structures remains an open question. Future work should investigate the behavior of SLGBN beyond these assumptions, particularly in more general distributional scenarios and real-world datasets.

## 7 Funding information.

This work was supported by the Gobierno de Navarra through the ANDIA 2021 program (grant no. 0011-3947-2021-000023) and the ERA PerMed JTC2022 PORTRAIT project (grant no. 0011- 2750-2022-000000).

## 8 Conflicts of interest

The authors declare that they have no competing interests.

## References

[1] D Atienza, C Bielza, and P Larrañaga. Pybnesian: An extensible python package for bayesian networks. Neurocomputing, 504(C):204–209, September 2022.

[2] D Atienza, C Bielza, and P Larrañaga. Semiparametric bayesian networks. Information Sciences, 584:564–582, 2022.

[3] D Atienza, P Larrañaga, and C Bielza. Hybrid semiparametric bayesian networks. TEST, 31:299–327, 2022.

[4] Jiahua Chen and Zehua Chen. Extended bayesian information criteria for model selection with large model spaces. Biometrika, 95(3):759–771, 2008.

[5] David Maxwell Chickering. A transformational characterization of equivalent bayesian network structures. In Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence, page 87–98, 1995.

[6] David Maxwell Chickering. Learning Bayesian Networks is NP-Complete, pages 121–130. Springer, New York, 1996.

[7] David Maxwell Chickering. Learning equivalence classes of bayesian-network structures. J. Mach. Learn. Res., 2:445–498, 2002.

[8] David Maxwell Chickering. Optimal structure identification with greedy search. J. Mach. Learn. Res., 3:507–554, 2002.

[9] Carl De Boor. A Practical Guide to Splines. Springer Verlag, New York, 1978.

[10] Paul H. C. Eilers and Brian D. Marx. Flexible smoothing with b-splines and penalties. Statistical Science, 11(2):89 – 121, 1996.

[11] Rina Foygel and Mathias Drton. Extended bayesian information criteria for gaussian graphical models. In Advances in Neural Information Processing Systems, volume 23, 2010.

[12] T Hastie and R Tibshirani. Generalized additive models. Statistical Science, 1 (3):297–318, 1986.

[13] D. Heckerman. Bayesian networks for data mining. Data Mining and Knowledge Discovery, 1:79 – 119, 1997.

[14] David Heckerman, Dan Geiger, and David M. Chickering. Learning bayesian networks: The combination of knowledge and statistical data. Mach. Learn., 20(3):197–243, September 1995.

[15] McKean J.W Hogg R.V. and Craig A.T. Introduction to mathematical statistics. Pearson Prentice Hall, Upper Saddle River, NJ, 2005.

[16] Neville K Kitson, Anthony C Constantinou, Zhigao Guo, Yang Liu, and Kiattikun Chobtham. A survey of bayesian network structure learning. Artificial Intelligence Review, 56(Suppl 1), 2023.

[17] D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques. Adaptive Computation and Machine Learning. MIT Press, Cambridge, 2009.

[18] Farnaz Nojavan A., Song S. Qian, and Craig A. Stow. Comparative analysis of discretization methods in bayesian networks. Environmental Modelling & Software, 87:64–71, 2017.

[19] D. W. Scott. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons, New York, 1992.

[20] Marco Scutari. Learning bayesian networks with the bnlearn R package. Journal of Statistical Software, 35(3):1–22, 2010.

[21] D. Servén and C. Brummitt. pygam: Generalized additive models in python, 2018.

[22] Charupriya Sharma and Peter van Beek. Scalable bayesian network structure learning with splines. In Proceedings of the 11th International Conference on Probabilistic Graphical Models, pages 181–192, 2022.

[23] Rafael Sojo, Pedro Larrañaga, and Concha Bielza. Transfer learning for nonparametric bayesian networks. Knowledge-Based Systems, 344, 2026.

[24] Ioannis Tsamardinos, Laura E Brown, and Constantin F Aliferis. The max-min hill-climbing bayesian network structure learning algorithm. Machine learning, 65(1):31–78, 2006.

## Appendix A Additional Figures

![](images/cde1bb80feb9659d74b11773d8a45fac444a956053d6b1fdffaf0580999cda12.jpg)  
Figure 6: BN\_insu network. The structure of the insurance network from bnlean. Third synthetic network.

## Appendix B Additional Tables

Table 2: Parametrization of B\_insu.
<table><tr><td>Node</td><td>Probability distribution</td><td>Node Type</td></tr><tr><td>Age</td><td>N(0,1)</td><td>Linear</td></tr><tr><td>SocioEcon</td><td> ${ \mathcal { N } } ( 0 . 5 \cdot \sin \left( A g e \right) + 0 . 3 \cdot A g e ^ { 2 } , 1 )$ </td><td>Nonlinear</td></tr><tr><td>RiskAversion</td><td> $\mathcal { N } ( \operatorname { t a n h } { ( A g e ) } + 0 . 5 \cdot S o c i o E c o n , 1 )$ </td><td>Nonlinear</td></tr><tr><td>MakeModel</td><td> $\mathcal { N } ( 3 + 0 . 5 \cdot R i s k A v e r s i o n + 0 . 6 \cdot S o c i o E c o n , 1 )$ </td><td>Linear</td></tr><tr><td>Mileage</td><td>N(0,1)</td><td>Linear</td></tr><tr><td>HomeBase</td><td> $\mathcal { N } ( 0 . 3 \cdot R i s k A v e r s i o n + 0 . 5 \cdot S o c i o E c o n , 1 )$ </td><td>Linear</td></tr><tr><td>SeniorTrain</td><td> $\mathcal { N } ( 0 . 3 \cdot A g e + \sin { ( R i s k A v e r s i o n ) } + 0 . 6 \cdot S o c i o E c o n , 1 )$ </td><td>Nonlinear</td></tr><tr><td>VehicleYear</td><td> $\mathcal { N } ( 0 . 6 \cdot R i s k A v e r s i o n + 0 . 6 \cdot S o c i o E c o n , 1 )$ </td><td>Linear</td></tr><tr><td>Airbag</td><td> $\mathcal { N } ( \operatorname { t a n h } { ( M a k e M o d e l ) } + 0 . 5 \cdot V e h i c l e Y e a r , 1 )$ </td><td>Nonlinear</td></tr><tr><td>GoodStudent</td><td> $\mathcal { N } ( 0 . 6 \cdot A g e - 0 . 4 \cdot S o c i o E c o n , 1 )$ </td><td>Linear</td></tr><tr><td>DrivingSkill</td><td> $\mathcal { N } ( 1 + \operatorname { t a n h } { ( S e n i o r T r a i n ) } + 0 . 4 \cdot A g e , 1 )$ </td><td>Nonlinear</td></tr><tr><td>Cushioning</td><td> $\mathcal { N } ( 0 . 8 \cdot A i r b a g , 1 )$ </td><td>Linear</td></tr><tr><td>DrivQuality</td><td> $\mathcal { N } ( 0 . 7 \cdot R i s k A v e r s i o n + 0 . 6 \cdot D r i v i n g S k i l l , 1 )$ </td><td>Linear</td></tr><tr><td>AntiLock</td><td> $\mathcal { N } ( \sin { ( V e h i c l e Y e a r ) } + 0 . 5 \cdot M a k e M o d e l , 1 )$ </td><td>Nonlinear</td></tr><tr><td>CarValue</td><td> ${ \mathcal { N } } ( \operatorname { t a n h } \left( M a k e M o d e l \right) + 0 . 4 \cdot M i l e a g e + 0 . 2 \cdot V e h i c l e Y e a r , 1 )$ </td><td>Nonlinear</td></tr><tr><td>AntiTheft</td><td> $\mathcal { N } ( 1 / ( 1 + \exp ( - S o c i o E c o n ) ) + R i s k A v e r s i o n ^ { 2 } , 1 )$ </td><td>Nonlinear</td></tr><tr><td>Accident</td><td> $\mathcal { N } ( \sin { ( 0 . 5 \cdot A n t i L o c k + 0 . 4 \cdot M i l e a g e + 0 . 4 \cdot D r i v Q u a l i t y ) } , 1 )$ </td><td>Nonlinear</td></tr><tr><td>Theft</td><td> $\mathcal { N } ( 0 . 6 \cdot A n t i T h e f t + 0 . 5 \cdot H o m e B a s e + 0 . 4 \cdot C a r V a l u e , 1 )$ </td><td>Linear</td></tr><tr><td>RuggedAuto</td><td> $\mathcal { N } ( 0 . 8 \cdot V e h i c l e Y e a r + 0 . 5 \cdot M a k e M o d e l , 1 )$ </td><td>Linear</td></tr><tr><td>ThisCarDam</td><td> $\mathcal { N } ( 0 . 8 \cdot A c c i d e n t + 0 . 3 \cdot R u g g e d A u t o , 1 )$ </td><td>Linear</td></tr><tr><td>OtherCar</td><td> $\mathcal { N } ( 0 . 7 \cdot S o c i o E c o n , 1 )$ </td><td>Linear</td></tr><tr><td>MedCost</td><td> $\mathcal { N } ( 0 . 6 \cdot A c c i d e n t + 0 . 5 \cdot C u s h i o n i n g + \sin { ( A g e ) } , 1 )$ </td><td>Nonlinear</td></tr><tr><td>ThisCarCost</td><td> $\mathcal { N } ( 0 . 6 \cdot T h i s C a r D a m + 0 . 5 \cdot T h e f t + 0 . 4 \cdot C a r V a l u e , 1 )$ </td><td>Linear</td></tr><tr><td>OtherCarCost</td><td> $\mathcal { N } ( 0 . 8 \cdot A c c i d e n t + 0 . 5 \cdot R u g g e d A u t o , 1 )$ </td><td>Linear</td></tr><tr><td>ILiCost</td><td> $\mathcal { N } ( 0 . 7 \cdot A c c i d e n t , 1 )$ </td><td>Linear</td></tr><tr><td>DrivHist</td><td> $\mathcal { N } ( 1 + \operatorname { t a n h } { ( D r i v i n g S k i l l ) } , 1 )$ </td><td>Nonlinear</td></tr><tr><td>PropCost</td><td> $\mathcal { N } ( 0 . 8 \cdot T h i s C a r C o s t + 0 . 5 \cdot O t h e r C a r , 1 )$ </td><td>Linear</td></tr></table>

Table notes. Parametrization of B\_insu network. Linearity / Nonlinearity refers to their dependence to predecessors.

<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td> $\overline { { \mathbf { L L } } }$ </td><td>THD</td></tr><tr><td>LG-BIC</td><td>200</td><td>-3462.53</td><td> $- 4 3 3 9 . 3 7 { \pm } 6 1 . 9 9$ </td><td> $\overline { { 2 . 0 0 { \pm } 0 . 0 0 } }$ </td></tr><tr><td>LG-BIC</td><td>1000</td><td>-3462.53</td><td> $- 4 2 7 7 . 4 9 { \pm } 9 . 7 5$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td>LG-BIC</td><td>5000</td><td>-3462.53</td><td> $- 4 2 6 6 . 9 4 { \pm } 2 . 7 3$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td>LG-BIC</td><td>10000</td><td>-3462.53</td><td> $- 4 2 6 5 . 8 4 { \pm } 2 . 5 8$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td> $\mathrm { L G \mathrm { - E B I C ^ { \ast } } }$ </td><td>200</td><td>-3462.53</td><td> $- 4 3 3 9 . 3 7 { \pm } 6 1 . 9 9$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td> $\mathrm { L G \mathrm { - E B I C ^ { \ast } } }$ </td><td>1000</td><td>-3462.53</td><td> $- 4 2 7 7 . 4 9 { \pm } 9 . 7 5$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td> $\mathrm { L G \mathrm { - E B I C ^ { \ast } } }$ </td><td>5000</td><td>-3462.53</td><td> $- 4 2 6 6 . 9 4 { \pm } 2 . 7 3$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td> $\mathrm { L G \mathrm { - } E B I C ^ { \ast } }$ </td><td>10000</td><td>-3462.53</td><td> $- 4 2 6 5 . 8 4 { \pm } 2 . 5 8$ </td><td> $2 . 0 0 { \pm } 0 . 0 0 \ $ </td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-3462.53</td><td> $- 4 1 1 6 . 9 5 { \pm } 2 0 7 . 0 4$ </td><td> $\overline { { 0 . 6 0 { \pm } 0 . 6 8 } }$ </td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-3462.53</td><td> $- 3 7 0 7 . 4 6 { \pm } 4 5 . 3 1$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-3462.53</td><td> $- 3 5 8 0 . 3 3 { \pm } 1 2 . 2 0$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-3462.53</td><td> $- 3 5 5 1 . 2 3 { \pm } 1 1 . 6 4 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-3462.53</td><td> $- 3 7 8 9 . 7 4 { \pm } 7 1 . 0 9$ </td><td> $\overline { { 0 . 0 0 { \pm } 0 . 0 0 } }$ </td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-3462.53</td><td> $- 3 5 6 9 . 4 2 { \pm } 2 6 . 5 1$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-3462.53</td><td> $- 3 4 8 7 . 8 7 { \pm } 1 3 . 7 1 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-3462.53</td><td> $- 3 4 8 5 . 7 4 { \pm } 1 1 . 9 3 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>200</td><td>-3462.53</td><td> $- 3 7 8 9 . 7 4 { \pm } 7 1 . 0 9$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>1000</td><td>-3462.53</td><td> $- 3 5 6 9 . 4 2 { \pm } 2 6 . 5 1$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>5000</td><td>-3462.53</td><td> $- 3 4 8 7 . 8 7 { \pm } 1 3 . 7 1 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>10000</td><td>-3462.53</td><td> $- 3 4 8 5 . 7 4 { \pm } 1 1 . 9 3 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr></table>

Table 3: B5 Fixed.  
Table notes. Fixed structure learning results for B5 network. Values shown as ${ \mathrm { m e a n } } \pm { \mathrm { s t d } } .$

Table 4: B8 Fixed.
<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td> $\overline { { \mathbf { L L } } }$ </td><td>THD</td></tr><tr><td>LG-BIC</td><td>200</td><td>-5515.31</td><td> $- 7 1 8 8 . 7 3 { \pm } 1 7 6 . 4 5$ </td><td> $\overline { { 3 . 0 0 \pm 0 . 0 0 } }$ </td></tr><tr><td>LG-BIC</td><td>1000</td><td>-5515.31</td><td> $- 7 0 3 2 . 9 5 { \pm } 1 2 . 0 6$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>LG-BIC</td><td>5000</td><td>-5515.31</td><td> $- 7 0 1 8 . 3 9 { \pm } 8 . 1 4$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>LG-BIC</td><td>10000</td><td>-5515.31</td><td> $- 7 0 1 6 . 9 4 { \pm } 5 . 1 7$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { L G \mathrm { - E B I C ^ { \ast } } }$ </td><td>200</td><td>-5515.31</td><td> $- 7 1 8 8 . 7 3 { \pm } 1 7 6 . 4 5$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { L G \mathrm { - } E B I C ^ { \ast } }$ </td><td>1000</td><td>-5515.31</td><td> $- 7 0 3 2 . 9 5 { \pm } 1 2 . 0 6$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { L G \mathrm { - } E B I C ^ { \ast } }$ </td><td>5000</td><td>-5515.31</td><td> $- 7 0 1 8 . 3 9 { \pm } 8 . 1 4$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { L G \mathrm { - } E B I C ^ { \ast } }$ </td><td>10000</td><td>-5515.31</td><td> $- 7 0 1 6 . 9 4 { \pm } 5 . 1 7$ </td><td> $3 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-5515.31</td><td> $- 6 8 2 2 . 8 3 { \pm } 3 7 0 . 3 1$ </td><td> $\overline { { 0 . 8 5 { \pm } 0 . 6 7 } }$ </td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-5515.31</td><td> $- 6 1 5 8 . 2 6 { \pm } 1 5 6 . 3 7$ </td><td> $0 . 1 5 { \pm } 0 . 3 7$ </td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-5515.31</td><td> $- 5 8 8 7 . 3 0 { \pm } 4 2 . 9 0$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-5515.31</td><td> $- 5 8 5 9 . 5 9 { \pm } 9 8 . 2 1 $ </td><td> $0 . 1 0 { \pm } 0 . 3 1$ </td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-5515.31</td><td> $- 5 9 2 5 . 5 1 { \pm } 1 0 2 . 9 1 $ </td><td>一  $\overline { { 0 . 0 0 { \pm } 0 . 0 0 } }$ </td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-5515.31</td><td> $- 5 7 2 9 . 1 6 { \pm } 2 1 . 0 6$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-5515.31</td><td> $- 5 6 8 5 . 1 9 { \pm } 2 5 . 1 5$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-5515.31</td><td> $- 5 6 7 1 . 0 8 { \pm } 2 7 . 6 7$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>200</td><td>-5515.31</td><td> $- 5 9 2 5 . 5 1 { \pm } 1 0 2 . 9 1 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>1000</td><td>-5515.31</td><td> $- 5 7 2 9 . 1 6 { \pm } 2 1 . 0 6$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>5000</td><td>-5515.31</td><td> $- 5 6 8 5 . 1 9 { \pm } 2 5 . 1 5$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td> $\mathrm { S L G B N  – E B I C ^ { * } }$ </td><td>10000</td><td>-5515.31</td><td> $- 5 6 7 1 . 0 8 { \pm } 2 7 . 6 7$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr></table>

Table notes. Fixed structure learning results for B8 network. Values shown as ${ \mathrm { m e a n } } \pm { \mathrm { s t d } } .$

Table 5: B\_insu Fixed.
<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td>LL</td><td>THD</td></tr><tr><td>LG-BIC</td><td>200</td><td>-38246.71</td><td>-40247.69±58.36</td><td>11.00±0.00</td></tr><tr><td>LG-BIC</td><td>1000</td><td>-38246.71</td><td> $- 4 0 0 0 9 . 8 8 { \pm } 1 7 . 0 0$ </td><td>11.00±0.00</td></tr><tr><td>LG-BIC</td><td>5000</td><td>-38246.71</td><td> $- 3 9 9 6 5 . 9 1 { \pm } 4 . 1 1 $ </td><td>11.00±0.00</td></tr><tr><td>LG-BIC</td><td>10000</td><td>-38246.71</td><td> $- 3 9 9 6 1 . 1 4 { \pm } 3 . 6 2 $ </td><td>11.00±0.00</td></tr><tr><td>LG-EBIC*</td><td>200</td><td>-38246.71</td><td> $- 4 0 2 4 7 . 6 9 { \pm } 5 8 . 3 6$ </td><td>11.00±0.00</td></tr><tr><td>LG-EBIC*</td><td>1000</td><td>-38246.71</td><td> $- 4 0 0 0 9 . 8 8 { \pm } 1 7 . 0 0$ </td><td>11.00±0.00</td></tr><tr><td>LG-EBIC*</td><td>5000</td><td>-38246.71</td><td> $- 3 9 9 6 5 . 9 1 { \pm } 4 . 1 1 $ </td><td>11.00±0.00</td></tr><tr><td>LG-EBIC*</td><td>10000</td><td>-38246.71</td><td>-39961.14±3.62</td><td>11.00±0.00</td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-38246.71</td><td>-39660.00±73.64</td><td>8.65±0.99</td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-38246.71</td><td>-39095.53±32.31</td><td>7.05±0.60</td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-38246.71</td><td>-38860.37±10.71</td><td>5.70±0.57</td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-38246.71</td><td>-38785.79±12.30</td><td>4.50±0.69</td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-38246.71</td><td>-39096.07±64.29</td><td>6.40±0.75</td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-38246.71</td><td>-38697.99±22.45</td><td>2.50±0.89</td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-38246.71</td><td>-38606.37±9.84</td><td>0.15±0.37</td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-38246.71</td><td>-38590.98±7.39</td><td>0.00±0.00</td></tr><tr><td>SLGBN-EBIC*</td><td>200</td><td>-38246.71</td><td>-39096.07±64.29</td><td>6.40±0.75</td></tr><tr><td>SLGBN-EBIC*</td><td>1000</td><td>-38246.71</td><td>-38697.99±22.45</td><td>2.50±0.89</td></tr><tr><td>SLGBN-EBIC*</td><td>5000</td><td>-38246.71</td><td> $- 3 8 6 0 6 . 3 7 { \pm } 9 . 8 4$ </td><td>0.15±0.37</td></tr><tr><td>SLGBN-EBIC*</td><td>10000</td><td>-38246.71</td><td>-38590.98±7.39</td><td>0.00±0.00</td></tr></table>

Table notes. Fixed structure learning results for B\_insu network. Values shown as mean ± std.

Table 6: B5 Free.
<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td>LL</td><td>THD</td><td>SHD</td><td>Precision</td><td>Recall</td></tr><tr><td>LG-BIC</td><td>200</td><td>-3462.53</td><td>-4344.83±63.74</td><td>2.00±0.00</td><td>0.95±0.76</td><td>0.96±0.10</td><td>0.86±0.09</td></tr><tr><td>LG-BIC</td><td>1000</td><td>-3462.53</td><td>-4278.18±9.59</td><td>2.00±0.00</td><td>1.00±0.46</td><td>0.97±0.07</td><td>0.83±0.07</td></tr><tr><td>LG-BIC</td><td>5000</td><td>-3462.53</td><td>-4267.58±3.54</td><td>2.00±0.00</td><td>1.05±0.39</td><td>0.96±0.08</td><td>0.83±0.07</td></tr><tr><td>LG-BIC</td><td>10000</td><td>-3462.53</td><td>-4266.25±2.73</td><td>2.00±0.00</td><td>1.05±0.39</td><td>0.98±0.06</td><td>0.81±0.04</td></tr><tr><td>LG-EBIC*</td><td>200</td><td>-3462.53</td><td>-4337.12±54.29</td><td>2.00±0.00</td><td>0.85±0.49</td><td>0.99±0.04</td><td>0.84±0.08</td></tr><tr><td>LG-EBIC*</td><td>1000</td><td>-3462.53</td><td>-4277.08±9.43</td><td>2.00±0.00</td><td>0.90±0.31</td><td>1.00±0.00</td><td>0.82±0.06</td></tr><tr><td>LG-EBIC*</td><td>5000</td><td>-3462.53</td><td>-4266.87±2.97</td><td>2.00±0.00</td><td>0.95±0.39</td><td>0.99±0.04</td><td>0.82±0.06</td></tr><tr><td>LG-EBIC*</td><td>10000</td><td>-3462.53</td><td>-4266.14±2.56</td><td>2.00±0.00</td><td>1.00±0.32</td><td>0.99±0.04</td><td>0.81±0.04</td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-3462.53</td><td>-4227.83±294.20</td><td>1.65±0.75</td><td>0.60±0.94</td><td>0.94±0.10</td><td>0.95±0.14</td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-3462.53</td><td>-3738.51±61.03</td><td>1.45±1.00</td><td>1.10±0.79</td><td>0.84±0.10</td><td>0.99±0.04</td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-3462.53</td><td>-3581.58±15.46</td><td>1.50±0.76</td><td>0.95±0.39</td><td>0.84±0.06</td><td>1.00±0.00</td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-3462.53</td><td>-3550.00±16.37</td><td>1.85±0.37</td><td>1.40±0.68</td><td>0.79±0.07</td><td>1.00±0.00</td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-3462.53</td><td>-3847.44±125.88</td><td>0.45±0.69</td><td>0.55±0.83</td><td>0.92±0.11</td><td>0.99±0.04</td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-3462.53</td><td>-3607.08±98.39</td><td>0.95±0.94</td><td>0.25±0.55</td><td>0.97±0.07</td><td>0.99±0.04</td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-3462.53</td><td>-3487.87±13.71</td><td>0.45±0.60</td><td>0.00±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-3462.53</td><td>-3485.74±11.93</td><td>0.20±0.41</td><td>0.00±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td></tr><tr><td>SLGBN-EBIC*</td><td>200</td><td>-3462.53</td><td>-3803.64±82.79</td><td>0.35±0.59</td><td>0.10±0.45</td><td>0.99±0.04</td><td>0.99±0.04</td></tr><tr><td>SLGBN-EBIC*</td><td>1000</td><td>-3462.53</td><td>-3606.92±98.49</td><td>0.95±0.94</td><td>0.20±0.52</td><td>0.97±0.07</td><td>0.99±0.04</td></tr><tr><td>SLGBN-EBIC*</td><td>5000</td><td>-3462.53</td><td>-3487.87±13.71</td><td>0.45±0.60</td><td>0.00±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td></tr><tr><td>SLGBN-EBIC*</td><td>10000</td><td>-3462.53</td><td>-3485.74±11.93</td><td>0.20±0.41</td><td>0.00±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td></tr></table>

Table notes. Full structure learning results for B5 network. Values shown as mean ± std.

Table 7: B8 Free.
<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td>LL</td><td>THD</td><td>SHD</td><td>Precision</td><td>Recall</td></tr><tr><td>LG-BIC</td><td>200</td><td>-5515.31</td><td>-7206.62±158.62</td><td>3.00±0.00</td><td>5.50±1.82</td><td>0.82±0.12</td><td>0.71±0.09</td></tr><tr><td>LG-BIC</td><td>1000</td><td>-5515.31</td><td>-7032.67±13.20</td><td>3.00±0.00</td><td>6.15±2.03</td><td>0.75±0.10</td><td>0.77±0.06</td></tr><tr><td>LG-BIC</td><td>5000</td><td>-5515.31</td><td>-7007.75±10.85</td><td>3.00±0.00</td><td>7.85±1.79</td><td>0.64±0.06</td><td>0.82±0.06</td></tr><tr><td>LG-BIC</td><td>10000</td><td>-5515.31</td><td>-7003.69±7.45</td><td>3.00±0.00</td><td>8.00±1.84</td><td>0.62±0.06</td><td>0.87±0.05</td></tr><tr><td>LG-EBIC*</td><td>200</td><td>-5515.31</td><td>-7201.46±165.02</td><td>3.00±0.00</td><td>4.40±1.39</td><td>0.94±0.10</td><td>0.68±0.06</td></tr><tr><td>LG-EBIC*</td><td>1000</td><td>-5515.31</td><td>-7038.44±13.25</td><td>3.00±0.00</td><td>4.70±1.69</td><td>0.87±0.11</td><td>0.73±0.06</td></tr><tr><td>LG-EBIC*</td><td>5000</td><td>-5515.31</td><td>-7009.95±10.61</td><td>3.00±0.00</td><td>7.70±1.56</td><td>0.65±0.06</td><td>0.81±0.07</td></tr><tr><td>LG-EBIC*</td><td>10000</td><td>-5515.31</td><td>-7004.81±6.96</td><td>3.00±0.00</td><td>7.70±1.66</td><td>0.64±0.07</td><td>0.83±0.05</td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-5515.31</td><td>-7045.28±326.50</td><td>2.45±1.43</td><td>6.05±2.78</td><td>0.74±0.13</td><td>0.80±0.11</td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-5515.31</td><td>-6244.65±243.39</td><td>1.30±1.49</td><td>3.60±2.58</td><td>0.84±0.15</td><td>0.90±0.04</td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-5515.31</td><td>-5909.94±65.52</td><td>0.95±1.39</td><td>2.90±1.74</td><td>0.85±0.10</td><td>0.94±0.04</td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-5515.31</td><td>-5874.25±102.61</td><td>0.80±1.15</td><td>3.15±2.76</td><td>0.84±0.15</td><td>0.95±0.04</td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-5515.31</td><td>-5998.62±122.15</td><td>0.35±0.59</td><td>0.75±1.12</td><td>0.98±0.07</td><td>0.97±0.04</td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-5515.31</td><td>-5748.06±87.65</td><td>0.25±0.91</td><td>0.45±1.00</td><td>0.97±0.06</td><td>1.00±0.02</td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-5515.31</td><td>-5684.82±21.35</td><td>0.10±0.31</td><td>0.95±0.94</td><td>0.93±0.06</td><td>1.00±0.00</td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-5515.31</td><td>-5671.09±21.16</td><td>0.30±0.47</td><td>2.00±0.97</td><td>0.86±0.06</td><td>1.00±0.00</td></tr><tr><td>SLGBN-EBIC*</td><td>200</td><td>-5515.31</td><td>-6044.00±145.21</td><td>0.55±1.19</td><td>1.05±1.19</td><td>0.97±0.06</td><td>0.94±0.05</td></tr><tr><td>SLGBN-EBIC*</td><td>1000</td><td>-5515.31</td><td>-5747.14±87.72</td><td>0.25±0.91</td><td>0.25±0.91</td><td>0.99±0.05</td><td>1.00±0.02</td></tr><tr><td>SLGBN-EBIC*</td><td>5000</td><td>-5515.31</td><td>-5684.30±20.48</td><td>0.10±0.31</td><td>0.70±0.73</td><td>0.95±0.05</td><td>1.00±0.00</td></tr><tr><td>SLGBN-EBIC*</td><td>10000</td><td>-5515.31</td><td>-5670.77±21.36</td><td>0.30±0.47</td><td>1.65±1.14</td><td>0.88±0.07</td><td>1.00±0.00</td></tr></table>

Table notes. Full structure learning results for B8 network. Values shown as mean ± std.

Table 8: B\_insu Free.
<table><tr><td>Model-Score</td><td>Sample size</td><td>Ground LL</td><td>LL</td><td>THD</td><td>SHD</td><td>Precision</td><td>Recall</td></tr><tr><td>LG-BIC</td><td>200</td><td>-38246.71</td><td>-40433.97±130.31</td><td>11.00±0.00</td><td>34.55±7.79</td><td>0.65±0.08</td><td>0.74±0.04</td></tr><tr><td>LG-BIC</td><td>1000</td><td>-38246.71</td><td>-39909.71±29.50</td><td>11.00±0.00</td><td>37.35±7.11</td><td>0.61±0.07</td><td>0.81±0.02</td></tr><tr><td>LG-BIC</td><td>5000</td><td>-38246.71</td><td>-39817.52±30.12</td><td>11.00±0.00</td><td>46.50±11.14</td><td>0.54±0.08</td><td>0.86±0.02</td></tr><tr><td>LG-BIC</td><td>10000</td><td>-38246.71</td><td>-39792.27±27.43</td><td>11.00±0.00</td><td>44.55±12.21</td><td>0.55±0.08</td><td>0.88±0.02</td></tr><tr><td>LG-EBIC*</td><td>200</td><td>-38246.71</td><td>-40471.27±143.57</td><td>11.00±0.00</td><td>24.40±3.49</td><td>0.84±0.05</td><td>0.65±0.03</td></tr><tr><td>LG-EBIC*</td><td>1000</td><td>-38246.71</td><td>-39937.79±44.53</td><td>11.00±0.00</td><td>27.55±5.39</td><td>0.73±0.07</td><td>0.77±0.02</td></tr><tr><td>LG-EBIC*</td><td>5000</td><td>-38246.71</td><td>-39823.30±30.02</td><td>11.00±0.00</td><td>40.45±9.47</td><td>0.58±0.08</td><td>0.84±0.02</td></tr><tr><td>LG-EBIC*</td><td>10000</td><td>-38246.71</td><td>-39792.83±27.71</td><td>11.00±0.00</td><td>40.15±11.27</td><td>0.58±0.08</td><td>0.88±0.03</td></tr><tr><td>SPBN-CVL</td><td>200</td><td>-38246.71</td><td>-40237.64±312.16</td><td>10.10±1.07</td><td>29.90±4.94</td><td>0.74±0.08</td><td>0.67±0.05</td></tr><tr><td>SPBN-CVL</td><td>1000</td><td>-38246.71</td><td>-39091.94±50.34</td><td>8.40±0.68</td><td>30.50±4.08</td><td>0.67±0.05</td><td>0.82±0.03</td></tr><tr><td>SPBN-CVL</td><td>5000</td><td>-38246.71</td><td>-38816.69±40.35</td><td>8.20±1.01</td><td>38.25±8.88</td><td>0.60±0.07</td><td>0.88±0.03</td></tr><tr><td>SPBN-CVL</td><td>10000</td><td>-38246.71</td><td>-38737.89±43.36</td><td>8.05±0.94</td><td>38.05±10.55</td><td>0.60±0.07</td><td>0.89±0.01</td></tr><tr><td>SLGBN-BIC</td><td>200</td><td>-38246.71</td><td>-39519.32±215.21</td><td>8.80±1.01</td><td>33.40±7.50</td><td>0.66±0.08</td><td>0.76±0.04</td></tr><tr><td>SLGBN-BIC</td><td>1000</td><td>-38246.71</td><td>-38854.03±268.26</td><td>6.70±1.13</td><td>30.00±8.52</td><td>0.68±0.09</td><td>0.84±0.02</td></tr><tr><td>SLGBN-BIC</td><td>5000</td><td>-38246.71</td><td>-38637.79±75.69</td><td>6.55±2.14</td><td>40.15±12.44</td><td>0.59±0.10</td><td>0.90±0.02</td></tr><tr><td>SLGBN-BIC</td><td>10000</td><td>-38246.71</td><td>-38586.49±88.42</td><td>5.55±2.76</td><td>37.20±15.19</td><td>0.61±0.12</td><td>0.90±0.03</td></tr><tr><td>SLGBN-EBIC*</td><td>200</td><td>-38246.71</td><td>-39648.17±424.58</td><td>8.80±1.28</td><td>21.40±3.47</td><td>0.88±0.05</td><td>0.68±0.03</td></tr><tr><td>SLGBN-EBIC*</td><td>1000</td><td>-38246.71</td><td>-38828.77±213.72</td><td>6.55±1.15</td><td>21.90±8.16</td><td>0.79±0.11</td><td>0.81±0.03</td></tr><tr><td>SLGBN-EBIC*</td><td>5000</td><td>-38246.71</td><td>-38634.82±79.99</td><td>6.55±2.01</td><td>34.50±11.40</td><td>0.63±0.10</td><td>0.89±0.03</td></tr><tr><td>SLGBN-EBIC*</td><td>10000</td><td>-38246.71</td><td>-38577.58±98.36</td><td>5.25±2.61</td><td>35.00±15.68</td><td>0.63±0.13</td><td>0.90±0.04</td></tr></table>

Table notes. Full structure learning results for B\_insu network. Values shown as mean ± std.