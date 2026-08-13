# Proportional Analogies on Probability Distributions via Bayesian Updating

Pierre-Alexandre Murena

Hamburg University of Technology (TUHH) Research Group on Human-Centric Machine Learning pierre-alexandre.murena@tuhh.de

## Abstract

Analogies are quaternary relations of the form “A is to B as C is to D". Among the various formalizations of analogical reasoning, proportional analogies provide an important axiomatic framework by characterizing valid analogies through a set of postulates. While proportional analogies have been extensively studied over Boolean, symbolic, and real-valued domains, their extension to probability distributions remains largely unexplored. In this paper, we introduce a notion of proportional analogy for probability distributions based on Bayesian updating. Our approach builds upon the idea that two distributions are related whenever one can be transformed into the other through Bayesian updating induced by a suitable set of observations. We investigate this framework for several standard members of the exponential family and discuss how it naturally extends to arbitrary probability distributions through Gaussian mixture approximations.

Code — https://github.com/ppaamm/Analogies-on-Probability-Distributions-by-Bayes-Updating

## Introduction

Analogical reasoning is a fundamental aspect of human cognition (Mitchell 2021), enabling the comparison of objects through their structural similarities (Gentner 1983). An analogy is defined as a quaternary relation of the form “� is to � as � is to �”, usually denoted by � ∶ � ∶∶ � ∶ �. Although analogy has traditionally been studied within symbolic artificial intelligence, it has recently attracted increasing attention in numerical machine learning. Notable applications include the characterization of word embeddings (Mikolov et al. 2013) and Large Language Models (Webb, Holyoak, and Lu 2023), as well as data augmentation (Couceiro et al. 2017) and the study of inductive reasoning (Murena 2022; Bayoudh, Miclet, and Delhay 2007).

Despite these advances, the study of analogical reasoning remains largely confined to a limited range of data domains. Existing work has mainly focused on hierarchical structures (Gick and Holyoak 1980), Boolean data (Prade and Richard 2017), character strings (Lepage 1998), and vector spaces (Rumelhart and Abrahamson 1973). In contrast, analogies between probability distributions have received comparatively little attention, despite their potential importance for machine learning. To the best of our knowledge, only two approaches have been proposed. Murena,

Cornuéjols, and Dessalles (2018) define analogies through parallel transport in information geometry. Their approach is restricted to distributions belonging to the same parametric family, is computationally demanding, and does not satisfy accepted principles of analogical reasoning. More recently, Prade and Richard (2025) proposed an extension of classical proportional analogies to categorical distributions. However, their framework is limited to this specific family, has a restricted domain of definition, and does not explicitly exploit the statistical nature of probability distributions.

In this paper, we introduce a new notion of analogy between probability distributions based on Bayesian updating. The central idea is to characterize the similarity between two distributions through the existence of a dataset capable of transforming one distribution into the other by Bayesian inference with a given likelihood. This perspective naturally grounds analogical reasoning in the fundamental mechanisms of probabilistic inference while being applicable to arbitrary probability distributions. Moreover, our framework satisfies the classical properties expected from analogical reasoning and admits an elegant characterization for important statistical families. Our contributions are the following:

1. We introduce a novel definition of proportional analogy between probability distributions based on Bayesian updating, and show that it satisfies the fundamental properties of proportional analogies;

2. We prove that, for distributions belonging to the exponential family, the proposed notion admits a simple arithmetic characterization in the natural parameter space;

3. We develop a sampling-based algorithm for solving analogical equations between arbitrary probability distributions, making the proposed framework applicable beyond analytically tractable models.

## Background and Related Works

There exists various approaches to analogies, which all build upon the idea of correspondence between two domains, based on their inherent structure and of the transfer of properties (Hall 1989). Several computational approaches were proposed for analogies.

## Computational Approaches to Analogy

The model-based vision consists in using a model to describe the underlying structure of the domains and enforce the analogies. The Structure-Mapping Theory (Gentner 1983) defines analogy as a maximal alignment of latent structures. Antić (2022) characterizes analogies via universal algebra, where justifications are given as operations on shared primitives. Similarly, Murena (2022) introduces models as programs on a Turing machine and define the alignment of domains by the existence of minimal inputs producing a description of the analogy.

A special case of the model-based approach is the functional approach, that states that $a : b : : : c : ~ ( \cdot$ � if and only if there exists a function � such as $b = f ( a )$ and $d = f ( c )$ . A modern example of such an approach is given by the work of Lee et al. (2024) who build � as the composition of elementary operations learned with reinforcement learning.

The axiomatic vision describes analogies through a set of postulates that they must satisfy. The most common definition is the notion of proportional analogies (Lepage 2004), also called analogical proportions by some authors (Prade and Richard 2021).

## Proportional Analogies

The definition of proportional analogies is a formalization of principles introduced by Aristotle, in the form of the following three principles:

Definition 1. A proportional analogy is a quaternary relation  such that for all $a , b , c , d ,$ , we have:

1. Reflexivity: $\mathcal { A } ( a , b , a , b )$

2. Symmetry: A(a, b, c, d) iff A(c, d, a, b)

3. Central permutation: (�, �, �, �) if (�, �, �, �).

Some authors also include a fourth postulate, called determinism, and stating that $\mathcal { A } ( a , b , a , x )$ implies $x = b \mathrm { \ ( M i } .$ clet, Bayoudh, and Delhay 2008). An analogical proportion $\mathcal { A } ( a , b , c , d )$ is often denoted $a : b : : : c : d$

Analogical proportions have been extensively studied in the Boolean domain (Prade and Richard 2017; Antić 2022), on character strings (Lepage 1998; Stroppa and Yvon 2005), with applications e.g. in image processing (Duck et al. 2022), recommender systems (Hug et al. 2019) or natural language (Lepage 2004). We refer the reader to the position paper of Prade and Richard (2021) for a more extensive presentation of applications of proportional analogies.

An important relation satisfying the properties of Definition 1 is the arithmetic analogy:

Definition 2. Four vectors $a , b , c , d \in \mathbb { R } ^ { n }$ are said to be in arithmetic analogy is and only if $b - a = d - c$

This analogy was introduced before the formalization of proportional analogies (Rumelhart and Abrahamson 1973) and is sometimes designated as parallelogram rule.

## Proportional Analogies on Distributions

Despite their potential application to machine learning, analogies on probability distributions have not been extensively studied.

A first attempt was proposed by Murena, Cornuéjols, and Dessalles (2018). The authors propose to extend the parallelogram rule to Riemannian manifolds, by interpreting subtractions as a logarithmic map and the addition as a parallel transport along a geodesic. The authors show that this definition fails to satisfy the properties of proportional analogies when the manifold is not flat. However, they show that their definition can apply to probability distributions in the Fisher-Rao manifold (Nielsen 2020), and they propose an algorithm for analogies on multivariate normal distributions.

More recently, Prade and Richard (2025) propose a direct application of Definition 1 to categorical distributions by the introduction of arithmetico-geometric analogies, satisfying arithmetic analogy (Definition 2) to the probabilities themselves and to their logarithm. This approach has severe limitations. The double constraint strongly limits the sets of valid analogies, making it a too strong definition for practical applications. More importantly, their definition does not extend naturally to distributions beyond categorical distributions, since it does not rely on an understanding of the specific nature of probability distributions.

## Bayes-Based Analogies

In this section, we introduce the concept of Bayes-based proportional analogies on the set of probability distributions.

## Bayesian Inference

Let $( { \mathcal { X } } , A )$ be the observation space and (Θ, ) the parameter space, both endowed with their respective �-algebras. A statistical model is a family of probability measures $\mathcal { L } = \{ P _ { \theta } : \theta \in \Theta \}$ on $( { \mathcal { X } } , A )$ . When the measures $P _ { \theta }$ are dominated by a common measure, we denote their densities by $p ( x \mid \theta )$

In the Bayesian framework, the parameter � is modeled as a Θ-valued random variable. A prior distribution on � with density $p ( \theta )$ specifies the uncertainty about �. Thus, the prior turns the measurable space $( \Theta , \tau )$ into a probability space $( \Theta , \tau , \Pi )$ . The prior distribution is updated, after observation of $x \in \mathcal { X }$ , into the posterior distribution according to Bayes rule:

$$
p ( \boldsymbol { \theta } \mid \boldsymbol { x } ) = \frac { p ( \boldsymbol { x } \mid \boldsymbol { \theta } ) p ( \boldsymbol { \theta } ) } { p ( \boldsymbol { x } ) } ,\tag{1}
$$

where $\begin{array} { r } { p ( x ) = \int _ { \Theta } p ( x \mid \vartheta ) p ( \vartheta ) d \vartheta } \end{array}$ is the marginal likelihood.

To account for the absence of observation, we extend the observation space by introducing a distinguished symbol ∅ ∉  and define $\mathbf { \bar { \mathcal { X } } } = \mathcal { X } \cup \{ \mathbf { \bar { \alpha } } \}$ . The posterior associated with the null observation is defined by $p ( \theta \mid \emptyset ) = p ( \theta )$ Equivalently, Bayesian updating with ∅ acts as the identity on the space of prior distributions.

We use the notation $p \underset { \mathcal { L } } { \overset { x } {  } } q$ to denote that � is the posterior distribution obtained from the prior distribution � after observing � under the statistical model . Note that the relation $\underset { c } { \overset { \boldsymbol { x } } { } }$ is not symmetrical. We consider the symmetrized version by defining $p \underset { \mathcal { L } } { \overset { x } { \sim } } q$ if and only if $p \underset { \mathcal { L } } {  } q \mathrm { o r } q \underset { \mathcal { L } } {  } p$

## Bayes-Based Proportional Analogies

Consider a space  of probability distributions over Θ and a statistical model  with parameter in Θ. We define the quaternary relation $R _ { \mathcal { L } }$ over $\mathbf { \hat { \Pi } } _ { \mathcal { P } } ^ { * }$ as:

$$
R ( p _ { A } , p _ { B } , p _ { C } , p _ { D } ) \quad { \mathrm { i f f } } \quad { \mathrm { t h e r e ~ e x i s t ~ } } x _ { 1 } , x _ { 2 } \in { \bar { \mathcal { X } } } ; \{ \begin{array} { c } { x _ { 1 } } \\ { p _ { A } \underset { x _ { 1 } } {  } p _ { B } } \\ { p _ { C } \underset { C } {  } \varphi _ { D } } \\ { x _ { 2 } } \\ { p _ { A } \underset { x _ { 2 } } {  } p _ { C } } \\ { p _ { B } \underset { C } {  } \varphi _ { D } } \end{array} 
$$

Proposition 1. If P and L are such that for all $p , q \in \mathcal { P } _ { \cdot }$ , there exists $x \in \mathcal { X }$ such that $p \underset { \mathcal { L } } { \overset { x } { \sim } } q$ , then the relation � defined above is a proportional analogy, and we call it Bayes-based proportional analogyfor likelihood .

Proof. The symmetry and central permutation postulates follow directly from the definition. To prove that $p _ { A } : p _ { B } : : { \mathcal { L } }$ $p _ { A } : p _ { B }$ , we observe that, for any distribution $p \in { \mathcal { P } }$ , we have $p \underset { \mathcal { L } } { \overset { \mathcal { O } } { \sim } } p .$ . The reflexivity follows directly from this observation and from the premises of the proposition. □

A more restrictive definition requires that the four updates witnessing the analogy are shared across the corresponding pairs of distributions. That is, there exist $x _ { 1 } , x _ { 2 } , x _ { 3 } , x _ { 4 } \in \bar { \mathcal { X } }$ such that $x _ { 1 }$ induces both $p _ { A }  p _ { B }$ and $p _ { C }  p _ { D } , x _ { 2 }$ induces both $p _ { B }  p _ { A }$ and $p _ { D }  p _ { C } , x _ { 3 }$ induces both $p _ { A }  p _ { C }$ and $p _ { B }  p _ { D } .$ , and $x _ { 4 }$ induces both $p _ { C }  p _ { A }$ and $p _ { D }  p _ { B }$ . This amounts to defining a transformation between two distributions whenever one can be obtained from the other as either a prior-to-posterior or a posterior-to-prior update. This definition was not retained in this paper because it is overly restrictive. Indeed, Bayesian updates are generally not reversible. For example, in the family of normal distributions, every update decreases the posterior variance. Hence, if $\mathcal { N } _ { 1 } \not  \bar { \mathcal { N } _ { 2 } }$ the reverse transformation $\mathcal { N } _ { 2 } \  \ \mathcal { N } _ { 1 }$ cannot occur unless $\mathcal { N } _ { 1 } = \mathcal { N } _ { 2 }$ . As a result, this definition would preclude any non-trivial analogy between normal distributions.

Our proposed definition of analogy between distributions admits a natural interpretation in terms of statistical reachability. A similar perspective is already implicit in standard notions of analogy. For instance, an arithmetic analogy (Definition 2) can be characterized through a transformation $a  b$ induced by a translation.

This definition of analogy has an important consequence in terms of the distributions.

Proposition 2. If $p _ { A } : p _ { B } : : \cdot \ : { p _ { C } } : p _ { D }$ , then for all $\theta \in \Theta$ either (1) the distributions log $p ( \theta )$ are in arithmetic analogy; or (2) the reverse solution holds, i.e. log $p _ { A } ( \theta ) = \log p _ { D } ( \theta )$ and log $p _ { B } ( \theta ) = \log p _ { C } ( \theta )$

Proof. $p _ { A } \underset { \mathcal { L } } { \overset { x } {  } }$ is equivalent to:

$$
\log p _ { B } ( \theta ) = \log p _ { A } ( \theta ) + \log \mathcal { L } ( x \mid \theta ) - \log p ( x ) .
$$

Let $\theta \in \Theta$ . There exists $x , x ^ { \prime } \in { \mathcal { X } }$ and $\sigma _ { A B } , \sigma _ { C D } , \sigma _ { A C }$ and $\sigma _ { B D } \in \{ 0 , 1 \}$ such that:

$$
\begin{array} { r } { ( - 1 ) ^ { \sigma _ { A B } } ( \log p _ { B } ( \theta ) - \log p _ { A } ( \theta ) ) = \log \mathcal { L } ( x \mid \theta ) - \log p ( x ) } \\ { ( - 1 ) ^ { \sigma _ { C D } } ( \log p _ { D } ( \theta ) - \log p _ { C } ( \theta ) ) = \log \mathcal { L } ( x \mid \theta ) - \log p ( x ) } \\ { ( - 1 ) ^ { \sigma _ { A C } } ( \log p _ { C } ( \theta ) - \log p _ { A } ( \theta ) ) = \log \mathcal { L } ( x ^ { \prime } \mid \theta ) - \log p ( x ^ { \prime } ) } \\ { ( - 1 ) ^ { \sigma _ { B D } } ( \log p _ { D } ( \theta ) - \log p _ { B } ( \theta ) ) = \log \mathcal { L } ( x ^ { \prime } \mid \theta ) - \log p ( x ^ { \prime } ) } \end{array}
$$

which involves:

$$
\left\{ \begin{array} { c } { { \log p _ { B } - \log p _ { A } = ( - 1 ) ^ { \sigma _ { C D } - \sigma _ { A B } } \left( \log p _ { D } - \log p _ { C } \right) } } \\ { { \log p _ { C } - \log p _ { A } = ( - 1 ) ^ { \sigma _ { B D } - \sigma _ { A C } } \left( \log p _ { D } - \log p _ { B } \right) } } \end{array} \right.
$$

where we use the notation log $p _ { i }$ for log $p _ { i } ( \boldsymbol { \theta } )$

In the case where $\sigma _ { A B } = \sigma _ { C D }$ and $\sigma _ { A C } = \sigma _ { B D } ,$ we obtain that log $p _ { B } ( \theta ) - \log p _ { A } ( \theta ) = \log p _ { D } ( \theta ) - \log p _ { C } ( \theta ) .$

In the case where $\sigma _ { C D } = \sigma _ { A B }$ and $| \sigma _ { B D } - \sigma _ { A C } | = 1$ , the conditions imply that log $p _ { A } ( \theta ) = \log p _ { C } ( \theta )$ and log $p _ { B } ( \theta ) =$ log $p _ { D } ( \theta )$ . Similarly, when $| \sigma _ { C D } - \sigma _ { A B } | = 1$ and $\sigma _ { B D } = \sigma _ { A C } ,$ we have log $p _ { A } ( \theta ) = \log p _ { B } ( \theta )$ and log $p _ { C } ( \theta ) = \log p _ { D } ( \theta ) .$ Finally, when $| \sigma _ { B D } - \sigma _ { A C } | = | \sigma _ { C D } - \sigma _ { A B } | = 1$ , we observe log $p _ { A } ( \theta ) = \log p _ { D } ( \theta )$ and log $p _ { B } ( \theta ) = \log p _ { B } ( \theta )$ □

The reverse solution (log $\begin{array} { r c l } { p _ { A } ( \theta ) } & { = } & { \log p _ { D } ( \theta ) } \end{array}$ and log $p _ { B } ( \theta ) = \log p _ { C } ( \theta ) )$ is an important consequence of the choice of the ↭ relation. It corresponds to a case similar to the Klein model in analogies on Boolean (Klein 1981), judged as debatable by some authors (Prade and Richard 2021) while being naturally covered by some theoretical frameworks (Antić 2022).

## Analogical Equations

Given three distributions $p _ { A } , p _ { B }$ and $p _ { C } \in \mathcal { P }$ , solving the analogical equation ${ } ^ {  } p _ { A } : p _ { B } : : { } ^ { \mathcal { L } } p _ { C } : q ^ { , * }$ consists in finding the distributions $q \in \mathcal { P }$ such that the analogy holds. We define the support of the analogical equation associated with  the set Supp( ${ \mathcal { L } } ) \subseteq { \mathcal { P } } ^ { 3 }$ defined as the set of distributions $( p _ { A } , p _ { B } , p _ { C } )$ such that the analogical equation $p _ { A } : p _ { B } : : \mathcal { L } ~ p _ { C } : q$ admits at least one solution. Studying this set provides a natural way of evaluating the expressive power of a Bayes-based analogy: the larger the support, the larger the class of analogical problems that can be solved.

## Bayes-Based Analogies on the Exponential Family

A probability belongs to the exponential family (Brown 1986) if its probability density function or probability mass function takes the form:

$$
p ( \boldsymbol { \theta } \mid \boldsymbol { \xi } ) = h ( \boldsymbol { \theta } ) \exp \left( \eta ( \boldsymbol { \xi } ) ^ { T } T ( \boldsymbol { \theta } ) - A ( \boldsymbol { \xi } ) \right)\tag{2}
$$

where $\xi$ is the parameter of the distribution, $h ( \theta )$ the base measure, $T ( \theta )$ the suficient statistic, $\eta ( \xi )$ the natural parameter of the distribution and $A ( \xi )$ the log-partition.

## Exponential Family as Conjugate Prior

Consider the likelihood family $\mathcal { L } ^ { S }$ as the set of distributions of the form:

$$
p ( x \mid \theta ) \propto \exp ( S ( x ) ^ { T } T ( \theta ) )\tag{3}
$$

where $S ( x )$ corresponds to the suficient statistic of the observations with respect to the likelihood, that couples to the prior’s suficient statistic $T ( \theta )$

Proposition 3. The exponential family is a conjugate prior of $\mathcal { L } ^ { S }$ , and $p ( \theta \mid \xi ) \underset { \mathcal { L } ^ { S } } { \overset { x } { \underset { \mathcal { S } } {  } } } p ( \theta \mid \xi ^ { \prime } )$ if and only if:

$$
\eta ( \xi ^ { \prime } ) = \eta ( \xi ) + S ( x )\tag{4}
$$

Corollary 1. For all $x \in \mathcal { X }$ , we have $p ( \theta \mid \xi ) \overset { x } { \underset { \ell ^ { S } } {  } } p ( \theta \mid \xi ^ { \prime } )$ if and only if there exists $\sigma \in \{ 0 , 1 \}$ such that:

$$
( - 1 ) ^ { \sigma } \big ( \eta ( \xi ^ { \prime } ) - \eta ( \xi ) \big ) = S ( x )\tag{5}
$$

Proof. Using Bayes rule:

$$
\begin{array} { r l } & { p ( \theta \mid x , \xi ) \propto p ( x \mid \theta ) p ( \theta \mid \xi ) } \\ & { \qquad \propto \exp \left( S ( x ) ^ { T } T ( \theta ) \right) h ( x ) \exp \left( \eta ( \xi ) ^ { T } T ( \theta ) - A ( \xi ) \right) } \\ & { \qquad \propto h ( x ) \exp \left( \left( \eta ( \xi ) + S ( x ) \right) ^ { T } T ( \theta ) - A ( \xi ) \right) } \end{array}
$$

which proves the proposition.

In the corollary, the quantity � indicates the direction of the relation: $\sigma = 0$ corresponds to $p ( \theta \mid \xi ) \underset { \mathcal { L } ^ { S } } { \overset { x } {  } } p ( \theta \mid \xi ^ { \prime } )$ and $\sigma = 1 \ \mathrm { t o } \ p ( \theta \mid \xi ^ { \prime } ) \underset { \mathcal { L } ^ { S } } { \overset { x } {  } } \ p ( \theta \mid \xi ) .$

Note that, in this context, the exponential family is used as the prior, and not as the likelihood, as traditionally discussed in the literature (Diaconis and Ylvisaker 1979).

Characterization of Bayes-Based Analogies on the Exponential Family

The characterization of Bayes-based analogies on the exponential family follows directly from Proposition 1 and Corollary 1.

Proposition 4. For all hyperparameters $\xi _ { A } , \xi _ { B } , \xi _ { C }$ and $\xi _ { D } .$ , if

$$
p ( . \mid \xi _ { A } ) : p ( . \mid \xi _ { B } ) : : \overset { . . 5 } { p ( . \mid \xi _ { C } ) } : p ( . \mid \xi _ { D } )
$$

then either (1) the natural parameters $\eta ( \xi )$ satisfy the arithmetic analogy; or (2) the reverse solution holds, i.e. $\eta ( \xi _ { A } ) =$ $\eta ( \xi _ { D } )$ and $\eta ( \xi _ { B } ) = \eta ( \xi _ { C } )$

Proof. The proposition follows directly from Corollary 1 and Proposition 2. □

The necessary condition of the arithmetic proportion between natural parameters raises an important issue: the condition depends on the choice of the natural representation �. The choice of a natural representation is not unique for a same distribution. In general, we can define an equivalence relation ∼ between tuples $( T , \eta )$ and $( T ^ { \prime } , \eta ^ { \prime } )$ when there exists functions $\theta \mapsto u ( \theta )$ and $\xi \mapsto v ( \xi )$ such that:

$$
\eta ( \xi ) ^ { T } T ( \theta ) = \eta ^ { \prime } ( \xi ) ^ { T } T ^ { \prime } ( \theta ) + u ( \theta ) + v ( \xi ) .\tag{6}
$$

With such a definition of equivalence, it is possible to extend a given suficient statistic $T ( \theta )$ with some constants, making the dimension of the suficient statistics arbitrarily large. Brown (1986) defines the minimality of a suficient statistics $T ( . )$ if they are afinely independent.

An important result is the independence to the choice of the minimal natural representation:

Proposition 5. Let $( T , \eta )$ and $( T ^ { \prime } , \eta ^ { \prime } )$ such that $( T , \eta ) \sim$ $( T ^ { \prime } , \bar { \eta } ^ { \prime } )$ and $T ^ { \prime }$ is a minimal suficient statistic. If $\eta ( \xi _ { A } ) , \eta ( \xi _ { B } )$ $\eta ( \xi _ { C } )$ and $\eta ( \xi _ { D } )$ are in arithmetic analogy, then $\eta ^ { \prime } ( \xi _ { A } ) , \eta ^ { \prime } ( \xi _ { B } )$ $\eta ^ { \prime } ( \xi _ { C } )$ and $\eta ^ { \prime } ( \xi _ { D } )$ are also in arithmetic analogy.

Proof. Multiplying the arithmetic condition on $\eta ( \xi )$ ) by $T ( \theta )$ on the right, we obtain for all �:

$$
( \eta ( \xi _ { B } ) - \eta ( \xi _ { A } ) + \eta ( \xi _ { C } ) - \eta ( \xi _ { D } ) ) ^ { T } T ( \theta ) = 0
$$

Since $( T , \eta ) \sim ( T ^ { \prime } , \eta ^ { \prime } )$ , this rewrites as:

$$
\begin{array} { r } { ( \eta ^ { \prime } ( \xi _ { B } ) - \eta ^ { \prime } ( \xi _ { A } ) + \eta ^ { \prime } ( \xi _ { C } ) - \eta ^ { \prime } ( \xi _ { D } ) ) ^ { T } T ^ { \prime } ( \theta ) } \\ { + v ( \xi _ { B } ) - v ( \xi _ { A } ) + v ( \xi _ { C } ) - v ( \xi _ { D } ) = 0 } \end{array}
$$

We observe that the term on the second line is a constant w.r.t. �. Since $T ^ { \prime }$ is minimal, it is afinely independent, which guarantees $\eta ^ { \prime } ( \xi _ { B } ) - \eta ^ { \prime } ( \xi _ { A } ) + \eta ^ { \prime } ( \xi _ { C } ) - \bar { \eta } ^ { \prime } ( \xi _ { D } ) = 0 .$ □

Based on the last two propositions, we see that analogy is preserved by change of minimal suficient statistics, and, consequently, independent of the chosen representation of the distribution. Based on this result, we know assume that we work exclusively with minimal statistics.

## Defining Valid Analogies

The result of Proposition 4 provides a necessary condition for distributions to be in analogy. We saw that it does not depend on the choice of the function $S ( x )$ in the likelihood family. This condition is not necessarily suficient: it assumes the existence of observations $x , x ^ { \prime } \in { \mathcal { X } }$ that satisfy the conditions of Corollary 1, which is not always guaranteed. In order to introduce these, we consider the image of �, written Im(�), and defined as the set Im $( S ) = \{ S ( x ) : x \in \bar { \mathcal { X } } \}$ . Using the result of Proposition 4, we can express a full characterization of Bayes-based analogies in an exponential family using the natural parameters.

Theorem 1. For any $\xi _ { A } , \xi _ { B } , \xi _ { C } , \xi _ { D } \in \Xi$ , we have

$$
p ( . \mid \xi _ { A } ) : p ( . \mid \xi _ { B } ) : : \overset { . . 5 } { \dots } p ( . \mid \xi _ { C } ) : p ( . \mid \xi _ { D } )
$$

if and only if $\{ \eta ( \xi _ { A } ) - \eta ( \xi _ { B } ) , \eta ( \xi _ { B } ) - \eta ( \xi _ { A } ) \} \cap$ Im(�) and $\{ \eta ( \xi _ { C } ) - \eta ( \xi _ { A } ) , \eta ( \xi _ { A } ) - \eta ( \xi _ { C } ) \}$ ∩ Im(�) are not empty, and either of the two following conditions is satisfied:

$$
\begin{array} { r l } & { 1 . \ \eta ( \xi _ { A } ) = \eta ( \xi _ { D } ) , \eta ( \xi _ { B } ) = \eta ( \xi _ { C } ) ; \mathrm { o r } } \\ & { 2 . \ \eta ( \xi _ { B } ) - \eta ( \xi _ { A } ) = \eta ( \xi _ { D } ) - \eta ( \xi _ { C } ) . } \end{array}
$$

Proof. Considering that the analogy holds, we can deduce from Proposition 4 that either $\eta _ { A } = \eta _ { D }$ and $\eta _ { B } = \eta _ { C }$ , or that $\eta _ { B } - \eta _ { A } = \eta _ { D } - \eta _ { C }$ (using the simplified notation $\eta _ { i }$ for $\eta ( \xi _ { i } ) )$ . The conditions on the sets follows directly from the definition.

Assuming that $\{ \eta ( \xi _ { A } ) - \eta ( \xi _ { B } ) , \eta ( \xi _ { B } ) - \eta ( \xi _ { A } ) \}$ ∩ Im(�) and $\{ \eta ( \xi _ { C } ) - \eta ( \xi _ { A } ) , \eta ( \xi _ { A } ) - \eta ( \xi _ { C } ) \} \cap$ Im(�) are not empty, there exists $\sigma _ { A B } , \sigma _ { A C } \in \{ 0 , 1 \}$ and $x , x ^ { \prime } \in \bar { \mathcal { X } }$ such that $( - 1 ) ^ { \sigma _ { A B } } ( \eta _ { B } - \overline { { \eta _ { A } } } ) \overline { { = } } \overline { { S } } ( x )$ and $( - 1 ) ^ { \sigma _ { A C } } ( \eta _ { C } - \eta _ { A } ) = S ( x ^ { \prime } )$

If $\eta _ { D } ~ = ~ \eta _ { A }$ and $\eta _ { C } ~ = ~ \eta _ { B } ^ { } ,$ , we have $\eta _ { D } - \eta _ { C } = \eta _ { A } -$ $\eta _ { B } = ( - 1 ) ^ { 1 + \sigma _ { A B } } S ( x )$ . Additionally, $\eta _ { C } - \eta _ { A } = \eta _ { B } - \eta _ { A } =$ $( - 1 ) ^ { \sigma _ { A B } } S ( x )$ and $\eta _ { D } - \eta _ { B } = \eta _ { A } - \eta _ { B } = ( - 1 ) ^ { \sigma _ { A B } + 1 } S ( x )$ . By definition, this shows that the analogy holds.

If $\eta _ { B } - \eta _ { A } = \eta _ { D } - \eta _ { C }$ , we have $\eta _ { D } - \eta _ { C } = ( - 1 ) ^ { \sigma _ { A B } } S ( x )$ and $\eta _ { D } - \eta _ { B } = \eta _ { C } - \eta _ { A } = ( - 1 ) ^ { \sigma _ { A C } } S ( x ^ { \prime } )$ . By definition, the analogy holds. □

In this theorem, the suficient statistic � governs the existence of solutions through its image. Hence, the expressive power of the analogy relation is directly tied to the geometry of Im(�): richer images allow a larger set of valid analogies.

Proposition 6. Given two likelihoods $\mathcal { L } _ { S }$ and $\mathcal { L } _ { S ^ { \prime } }$ , if Im(�) ⊆ Im $( S ^ { \prime } )$ , then four distributions in analogy with respect $\mathrm { t o } \ : \mathcal { L } _ { S }$ are also in analogy with respect to $\mathcal { L } _ { S ^ { \prime } }$

As a direct consequence of this proposition, we can define a partial order between analogies, induced by the choices of �. This order corresponds to the inclusion of analogies seen as subsets of $\mathcal { P } ^ { 4 }$ . We call maximal an analogy associated to a likelihood $S _ { \mathrm { m a x } }$ such that $\operatorname { I m } ( S ) \subseteq \operatorname { I m } ( S _ { \operatorname* { m a x } } )$ for all �.

## Divergence-Based Characterization of Analogies

An alternative, coordinate-free characterization of Bayesbased analogies can be established using the Kullback-Leibler divergence. Because the KL divergence in a minimal exponential family equals the Bregman divergence generated by the log-partition function $A ( \eta )$ , the linear arithmetic structure maps directly to information geometry.

Proposition 7. Denote $M ( p , q ) \ = \ \mathrm { a r g } \mathrm { m i n } _ { r } ( D _ { \mathrm { K L } } ( p \| r ) \ +$ $D _ { \mathrm { K L } } ( q \Vert r ) )$ . If four distributions $p _ { A } , p _ { B } , p _ { C }$ and $p _ { D }$ of the exponential family are in analogy, then either $M ( p _ { A } , p _ { D } ) =$ $M ( p _ { B } , p _ { C } ) \mathrm { o r } M ( p _ { A } , p _ { C } ) = M ( p _ { B } , p _ { D } )$

Proof. For a minimal suficient statistic �, we can verify that $\eta ( M ( p , q ) ) = { \textstyle \frac { 1 } { 7 } } ( \eta ( p ) + \eta ( q ) )$ . We have the analogy if $\eta ( p _ { A } ) =$ 2 $\eta ( p _ { D } )$ and $\eta ( { \bar { p } } _ { B } ) = \eta ( p _ { C } )$ , in which case $M ( p _ { A } , p _ { C } ) =$ $M ( p _ { B } , p _ { D } ) $ , or if $\eta ( p _ { B } ) - \eta ( p _ { A } ) = \eta ( p _ { D } ) - \eta ( p _ { C } )$ , in which case $M ( p _ { A } , p _ { D } ) = M ( p _ { B } , p _ { C } )$ □

The converse implications do not hold in general: equality of the corresponding barycenters entails equality of the natural-parameter displacements, but does not ensure that their common value belongs to Im(�).

This result states that every valid analogy induces a common KL barycenter for one of these two pairings of distributions. This characterization by the barycenters is a wellknown property of arithmetic analogies in $\mathbb { R } ^ { d }$ , used for instance by Lepage and Couceiro (2024) as a definition of analogies.

## Analogical Equations

Theorem 1 characterizes the quadruples of distributions that are in analogy. The problem of defining the support Supp( $\mathcal { L } _ { S } )$ of analogical equations on an exponential family is closely related.

Theorem 2. The support $\operatorname { S u p p } ( \mathcal { L } _ { S } )$ is the set of triples of distributions $( p _ { A } , p _ { B } , p _ { C } )$ such that $\{ \eta ( \xi _ { A } ) - \eta ( \xi _ { B } ) , \eta ( \xi _ { B } ) -$ $\eta ( \xi _ { A } ) \}$ ∩ Im(�) and $\{ \eta ( \xi _ { C } ) - \eta ( \xi _ { A } ) , \eta ( \xi _ { A } ) - \eta ( \xi _ { C } ) \}$ ∩ Im(�)

are not empty, and either $\eta ( \xi _ { C } ) = \eta ( \xi _ { B } ) \mathrm { o r } \eta ( \xi _ { C } ) + \eta ( \xi _ { B } ) -$ $\eta ( \xi _ { A } ) \in \eta ( \Xi )$

Proof. Suppose $( p _ { A } , p _ { B } , p _ { C } )$ satisfies the conditions. If $\eta ( \xi _ { B } ) = \eta ( \xi _ { C } )$ , we define $\xi _ { D } = \xi _ { A }$ , and otherwise we take $\xi _ { D }$ such that $\eta ( \xi _ { D } ) = \eta ( \xi _ { C } ) + \eta ( \xi _ { B } ) - \eta ( \xi _ { A } )$ (which is possible by hypothesis). This defines an analogy by Theorem 1, and consequently $( p _ { A } , p _ { B } , p _ { C } ) \in \mathrm { S u p p } ( \mathcal { L } _ { S } )$ . Conversely, if $( p _ { A } , p _ { B } , p _ { C } ) \in \mathrm { S u p p } ( \mathcal { L } _ { S } )$ , there exists $p _ { D }$ such that the analogy holds. We have our result directly by Theorem 1. □

## Instantiation of Classical Exponential Families

The previous section established a complete characterization of Bayes-based analogies for exponential families. The remaining question is therefore not whether analogies exist, but how rich they are for a given statistical model. According to Theorem 1, this richness is entirely determined by the image of the suficient statistic of the likelihood. We now illustrate this framework on three classical exponential families. Besides demonstrating the generality of the approach, these examples highlight three qualitatively diferent situations regarding the expressive power of Bayes-based analogies.

## Bayes-Based Analogies on Bernoulli and Categorical Distributions

The categorical distribution is a probability distribution over a finite observation space $\mathcal { O } = \{ o _ { 1 } , \ldots , o _ { K } \}$ , parameterized by the probabilities $\pi _ { k }$ that every observation $o _ { k }$ can take. The space of categorical distributions corresponds to the simplex, i.e. the space of vectors $( \pi _ { 1 } , \ldots , \pi _ { K } )$ such that $\pi _ { k } \geq 0$ for all � and $\bar { \Sigma _ { k } } \pi _ { k } = 1$ . In this paper,we restrict our attention to strictly positive probability vectors, i.e., $\pi _ { k } > 0$ for every $k ,$ and denote the corresponding parameter space by $\Delta ^ { K }$ . The special case $K = 2$ corresponds to the Bernoulli distribution.

Natural parameters. A standard choice for the minimal parameters is the vector of log-ratios:

$$
\eta ( \xi ) = \left( \log \frac { \pi _ { 1 } } { \pi _ { K } } , \ldots , \log \frac { \pi _ { K - 1 } } { \pi _ { K } } \right) \in \mathbb { R } ^ { K - 1 }\tag{7}
$$

The corresponding natural statistics $T ( \theta )$ are then given by:

$$
T ( \theta ) = ( \mathbb { I } ( \theta = 1 ) , \dots , \mathbb { I } ( \theta = K - 1 ) )\tag{8}
$$

Compatible likelihood. Given � distributions $p _ { 1 } , \ldots , p _ { K }$ we define the likelihood $\mathcal { L } _ { M M } ^ { K }$ as a special case of $\mathcal { L } ^ { S }$ with:

$$
{ \cal S } ( x ) = \left( \log \frac { p _ { 1 } ( x ) } { p _ { K } ( x ) } , \dots , \log \frac { p _ { K - 1 } ( x ) } { p _ { K } ( x ) } \right) .\tag{9}
$$

We note that this distribution corresponds to a mixture model with distributions $p _ { 1 } , \ldots , p _ { K }$ , and that $\pi _ { k }$ is then the probability of the point to be generated by $p _ { k }$

Maximal analogy. The existence of a maximal analogy follows from the existence of parameters s.t. $\operatorname { I m } ( S ) = \mathbb { R } ^ { K - \tilde { 1 } }$

Proposition 8. There exists a maximal Bayes-based proportional analogy between categorical distributions induced by the likelihood $\mathcal { L } _ { \mathrm { M M } } ^ { K } .$

Proof. Let $p _ { k } = \mathcal { N } ( \mu _ { k } , \Sigma )$ , where Σ is positive definite and $\mu _ { 1 } - \mu _ { K } , \dots , \mu _ { K - 1 } - \mu _ { K }$ form a basis of ℝ $\textstyle K - 1$ . Then $S ( x ) =$ $A x + b$ for some invertible matrix � and vector $b ,$ so that $\operatorname { I m } ( S ) = \mathbb { R } ^ { K - 1 }$ □

Note that all the results above apply also to the case of Bernoulli distribution Ber(�), that is a categorical distribution with $K = 2$ . In that case, the natural parameter is simply the logit of �.

## Bayes-Based Analogies on Multivariate Normal Distributions

A multivariate normal distribution $\mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } )$ is a probability distribution over the parameter space $\dot { \Theta } = \mathbb { R } ^ { d }$ characterized by its mean vector $\pmb { \mu } \in \mathbb { R } ^ { d }$ and covariance matrix $\pmb { \Sigma } \in \mathbb { S } _ { + + } ^ { d } .$ where $\mathbb { S } _ { + + } ^ { d }$ denotes the set of symmetric positive definite matrices. Its density with respect to � is:

$$
p ( \pmb \theta \mid \pmb \mu , \pmb \Sigma ) = \frac { 1 } { \sqrt { ( 2 \pi ) ^ { d } | \pmb \Sigma | } } \exp \left( - \frac { 1 } { 2 } ( \pmb \theta - \pmb \mu ) ^ { T } \pmb \Sigma ^ { - 1 } ( \pmb \theta - \pmb \mu ) \right) .
$$

Natural parameters. Following our minimal exponential family formalism, a standard choice for the natural parameters is $\eta ( \pmb { \mu } , \pmb { \Sigma } ) = ( \pmb { \Sigma } ^ { - 1 } \pmb { \mu } , - \frac { 1 } { 2 } \mathrm { v e c } ( \pmb { \Sigma } ^ { - 1 } ) )$ . The corresponding minimal suficient statistics are $T ( \pmb \theta ) = ( \pmb \theta , \mathrm { v e c } ( \pmb \theta \pmb \theta ^ { T } ) )$

Compatible likelihood. We define the class $\mathcal { L } _ { A , B } ^ { \mathrm { q u a d } }$ of likelihoods over observations $x \in \mathcal { X }$ of the form:

$$
p ( x \mid \theta ) \propto \exp \left( A ( x ) ^ { \top } \theta - { \frac { 1 } { 2 } } \theta ^ { \top } B ( x ) \theta \right) ,\tag{10}
$$

where $A ( x ) \ \in \ \mathbb { R } ^ { d }$ and $B ( x ) ~ \in ~ \mathbb { S } _ { + + } ^ { d }$ for every observation �. Using the matrix trace identity $\theta ^ { T } B ( x ) \theta \ =$ $\operatorname { v e c } ( B ( x ) ) ^ { T } \operatorname { v e c } ( \theta \theta ^ { T } )$ , this family maps exactly to our general framework by setting the data suficient statistic to $\begin{array} { r } { S ( x ) = ( A ( x ) , - \frac { 1 } { 2 } \mathrm { v e c } ( B ( x ) ) ) ^ { T } } \end{array}$

Maximal analogy.

Proposition 9. There exists a unique maximal Bayes-based proportional analogy relation induced by likelihoods of the form $\mathcal { L } _ { A , B } ^ { \mathrm { q u a d } }$ , obtained when $\operatorname { I m } ( S ) = \mathbb { R } ^ { d } \times \operatorname { v e c } ( \mathbb { S } _ { + + } ^ { d } )$

Proof. We consider a standard Gaussian linear regression observation model $y = X \theta + \varepsilon ,$ , where $X \in \mathbb { R } ^ { d \times d }$ is an invertible design matrix and $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ . The completedata likelihood belongs to $\mathcal { L } _ { A , B } ^ { \mathrm { q u a d } }$ with:

$$
A ( X , y ) = { \frac { 1 } { \sigma ^ { 2 } } } X ^ { \top } y , \qquad B ( X ) = { \frac { 1 } { \sigma ^ { 2 } } } X ^ { \top } X .
$$

Let $M \in \mathbb { S } _ { + + } ^ { d }$ and $v \in \mathbb { R } ^ { d }$ be an arbitrary destination coordinate pair in the parameter space. We choose the matrix $X = \sigma \dot { M } ^ { 1 / 2 }$ , where $M ^ { 1 / 2 }$ denotes the unique symmetric positive definite square root of �. Substituting this yields $\begin{array} { r } { B ( X ) = \frac { 1 } { \sigma ^ { 2 } } X ^ { \top } X = M } \end{array}$ . Since $M \in \mathbb { S } _ { + + } ^ { d }$ , its square root $M ^ { 1 / 2 }$ is strictly invertible, making � invertible. Choosing the data vector $y = \sigma ^ { 2 } ( X ^ { \top } ) ^ { - 1 } v$ yields $\begin{array} { r } { A ( X , y ) = \frac { 1 } { \sigma ^ { 2 } } X ^ { \top } y = v . } \end{array}$ Therefore, every arbitrary parameter pair $( v , - \frac { 1 } { \gamma } \mathrm { v e c } ( M ) )$ ) is fully realizable by a valid observation tuple $( X , y ) ,$ proving that $\operatorname { I m } ( S ) = \mathbb { R } ^ { d } \times \operatorname { v e c } ( \mathbb { S } _ { + + } ^ { d } )$ □

## Bayes-Based Analogies on Dirichlet Distributions

The Dirichlet distribution is a probability distribution of the simplex $\Delta ,$ characterized by a vector $\alpha = ( \alpha _ { 1 } , \dots , \alpha _ { K } )$ . Its density with respect to � is:

$$
p ( \theta _ { 1 } , \dots , \theta _ { K } \mid \alpha ) = \frac { \Gamma \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } \right) } { \prod _ { k = 1 } ^ { K } \Gamma ( \alpha _ { k } ) } \prod _ { k = 1 } ^ { K } \theta _ { k } ^ { \alpha _ { k } - 1 }
$$

Natural parameters. A standard choice of natural parameters is $\bar { \eta ( \pmb { \alpha } ) } = ( \alpha _ { 1 } - 1 , \ldots , \alpha _ { K } - 1 ) \in ( 1 , + \infty ) ^ { K }$ . The corresponding suficient statistics are $T ( \pmb { \theta } ) = ( \log \theta _ { 1 } , \dots , \log \theta _ { K } )$

Compatible likelihood. With this choice of suficient statistics, the likelihood family $\mathcal { L } ^ { S }$ is the family of distributions of the form:

$$
p ( x \mid \theta ) = h ( x ) \prod _ { k = 1 } ^ { K } \theta _ { k } ^ { S _ { k } ( x ) }\tag{11}
$$

Proposition 10. Let $\mu$ be the reference measure on the observation space, and define the measure � by $\nu ( d x ) =$ $h ( x ) \mu ( d x )$ . Every coordinate of � is non-negative almost everywhere with respect to �.

Proof. Consider a compatible likelihood of the form of Equation 11, where � belongs to the interior of the probability simplex.

Fix a coordinate �. We assume, for contradiction, that the set of observations � for which $S _ { j } ( x )$ is negative has strictly positive �-measure.

We define a sequence $\theta ( t )$ such that $\theta _ { j } ( t ) ~ = ~ \mathrm { e x p } ( - t )$ and $\begin{array} { r } { \theta _ { k } ( t ) ~ = ~ \frac { 1 - \exp ( - t ) } { K - 1 } } \end{array}$ for $k \neq j$ . For any � such that $S _ { j } ( x )$ , we then have $\theta _ { j } ( t ) ^ { S _ { j } ( x ) } \  \ + \infty$ and $\theta _ { k } ( t ) ^ { S _ { h } ( x ) }$ tends to a positive constant for $k \neq j$ . Consequently, lim<sub>�</sub> $\begin{array} { r } { \prod _ { k = 1 } ^ { K } \theta _ { k } ( t ) ^ { S _ { k } ( x ) } = + \infty } \end{array}$ . By Fatou’s lemma, we then have lim<sub>�</sub> $\textstyle { \int } \prod _ { k = 1 } ^ { K } \theta _ { k } ( t ) ^ { S _ { k } ( x ) } \nu ( d x ) = + \infty$ , which is impossible for a probability density. This shows that $S _ { k } ( x ) \stackrel { } { \geq } 0$ �-almost everywhere for all �. □

Maximal analogy. Based on the following proposition, we can build a likelihood that makes the analogy maximal in the sense that Im $S ) = \mathbb { R } ^ { K }$

Proposition 11. There exists a maximal Bayes-based proportional analogy between Dirichlet distributions induced by likelihood ${ \mathcal { L } } ^ { S + }$ that satisfy Proposition 10.

Proof. Let $c \subset [ 0 , 1 ]$ be the Cantor set. We remind that  is an uncountable measurable set of Lebesgue measure zero. $c$ and $\mathbb { R } ^ { K }$ are isomorphic as uncountable standard Borel spaces, so there exists a measurable surjective map $T : { \mathcal { C } } $ $\mathring { \mathbb { R } } ^ { K }$ . We then define $S ( x ) = T ( x )$ if ${ \bar { x } } \in { \mathfrak { C } }$ and $S ( x ) = 0$ otherwise. By construction, we have Im $S ) = \mathbb { R } ^ { K }$ . Since  is a Lebesgue measure zero, we have $\textstyle \int p ( x \mid \theta ) d x = 1$ , so the defined likelihood is valid. Since � is surjective into ℝ $K _ { , }$ it defines a maximal analogy. □

Although this construction satisfies the present definition of Bayes-based analogies, it reveals a subtle measuretheoretic aspect of the framework. The existence of an analogy depends only on the existence of observations producing the corresponding Bayesian updates, irrespective of whether these observations belong to sets of positive measure. Consequently, maximality may be achieved through observations that are statistically negligible. Whether such exceptional observations should be regarded as genuine generators of analogies is ultimately a design choice. In the present work, we retain the existential definition, leaving the investigation of stronger notions of reachability for future work.

## Approximate Solver of Analogical Equations

The cases explored so far assume that the distributions of interest are parametric and belong to the exponential family. The question of generalizing to arbitrary distributions is important for potential applications in machine learning. In this section, we discuss a sampling-based approach to this problem inspired by Approximate Bayesian Computation (Sunnåker et al. 2013).

## Problem Specification

We consider the setting where the distributions $p _ { A } , p _ { B }$ and $p _ { C }$ are not known in closed form, but only through finite samples $\{ \theta _ { A } ^ { 1 } , \hdots , \theta _ { A } ^ { n _ { A } } \} , \{ \theta _ { B } ^ { 1 } , \hdots , \theta _ { B } ^ { n _ { B } } \}$ and $\{ \theta _ { C } ^ { 1 } , \hdots , \theta _ { C } ^ { n _ { C } } \}$ where $\theta _ { A } ^ { i } , \theta _ { B } ^ { i } , \theta _ { C } ^ { i } \ \in \Theta$ . Given a likelihood $\mathcal { L } ,$ we aim to generate a sample from an unknown distribution $p _ { D }$ such that $p _ { A } : p _ { B } : : \mathcal { L } p _ { C } : p _ { D }$

Since the latent datasets responsible for the Bayesian updates are unknown, we formulate their estimation as the search for two datasets $\boldsymbol { x } , \boldsymbol { x } ^ { \prime } \in \bar { \mathcal { X } }$ satisfying the following consistency conditions. Let $\{ \hat { \theta } _ { B } ^ { 1 } , \hdots , \hat { \theta } _ { B } ^ { m _ { B } } \}$ and $\{ \hat { \theta } _ { C } ^ { 1 } , \dots , \hat { \theta } _ { C } ^ { m _ { C } } \}$ denote samples drawn from the posteriors obtained by updating $p _ { A }$ with � and $x ^ { \prime }$ respectively. Likewise, let $\{ \hat { \theta } _ { D } ^ { ( B ) , 1 } , \dots , \hat { \hat { \theta } } _ { D } ^ { ( B ) , m _ { D } } \}$ and $\{ \hat { \theta } _ { D } ^ { ( C ) , 1 } , \dots , \hat { \theta } _ { D } ^ { ( C ) , \bar { m _ { D } } } \}$ denote samples drawn respectively from the update of $p _ { B }$ with $x ^ { \prime }$ and $p _ { C }$ with �. The datasets � and $x ^ { \prime }$ are then sought so to satisfy $\{ \theta _ { B } ^ { i } \} \approx \{ \hat { \theta } _ { B } ^ { i } \} , \{ \theta _ { C } ^ { i } \} \approx \{ \hat { \theta } _ { C } ^ { i } \}$ and $\{ \hat { \theta } _ { D } ^ { ( B ) , i } \} \approx \{ \hat { \theta } _ { D } ^ { ( C ) , i } \}$ , where ≈ denotes similarity between empirical distributions

Since the analogy is based on the symmetric relation ↭, the latent datasets � and $x ^ { \prime }$ are not restricted to updating $p _ { A }$ into $p _ { B }$ and $p _ { C }$ . They may equally correspond to updates in the reverse direction, i.e., from $p _ { B } \tan p _ { A }$ and/or from $p _ { C } \tan p _ { A } .$ Consequently, the algorithm evaluates all combinations of forward and reverse updates when searching for a consistent analogical completion.

## Algorithm

Posterior sampling. Following the standard importance sampling framework (Robert, Casella, and Casella 2004), posterior distributions are approximated through weighted sampling. Given a prior distribution $p ( \theta )$ and a candidate dataset �, one repeatedly samples $\theta ^ { * } \sim p ( \theta )$ and assigns to each sample an importance weight $w ( \theta ^ { * } )$ proportional to the likelihood $\mathcal { L } ( x \mid \bar { \theta } ^ { * } )$ . After normalization of the weights, the resulting weighted sample provides an empirical approximation of the posterior distribution $p ( \theta \mid x )$ , from which posterior samples can be obtained through resampling. This procedure is used to approximate each of the posterior distributions involved in the consistency conditions of the previous section.

Distribution comparison. Since the posterior distributions are represented only through finite samples, they cannot be compared analytically. Instead, similarity between two distributions is estimated through a discrepancy measure defined on empirical samples. Several choices are possible, including the Maximum Mean Discrepancy (MMD), the Wasserstein distance, the energy distance, or other metrics between empirical measures. The choice of discrepancy determines the notion of closeness used throughout the algorithm and replaces the exact equality required in the theoretical definition of analogy.

Joint optimization. The latent datasets � and $x ^ { \prime }$ are searched jointly so as to simultaneously satisfy the three consistency conditions introduced above. Let $d ( \cdot , \cdot )$ denote the chosen discrepancy between empirical distributions. The objective is to find $( x , \dot { x } ^ { \prime } )$ minimizing

$$
\begin{array} { r } { d \big ( \{ \theta _ { B } ^ { i } \} , \{ \hat { \theta } _ { B } ^ { i } \} \big ) + d \big ( \{ \theta _ { C } ^ { i } \} , \{ \hat { \theta } _ { C } ^ { i } \} \big ) + d \Big ( \{ \hat { \theta } _ { D } ^ { ( B ) , i } \} , \{ \hat { \theta } _ { D } ^ { ( C ) , i } \} \Big ) . } \end{array}
$$

A candidate pair $( x , x ^ { \prime } )$ is accepted only if each of the three discrepancies is smaller than a prescribed threshold �. Otherwise, the analogical equation is considered to have no solution for the chosen likelihood model and tolerance. Whenever several candidate pairs satisfy these conditions, they provide alternative approximations of the unknown distribution $p _ { D }$

Reverse updates. The above procedure assumes that the unknown distribution $p _ { D }$ is obtained by updating either $p _ { B }$ or $p _ { C }$ . However, because the analogical relation is defined through the symmetric reachability relation $ ,$ , it is also possible that $p _ { D }$ is instead the prior from which both $p _ { B }$ and $p _ { C }$ are obtained after a Bayesian update. In this case, posterior sampling cannot be performed directly. Let $p ( \theta )$ denote the known posterior obtained after updating an unknown prior $p _ { D } ( \theta )$ with a dataset �. By Bayes’ rule, $\begin{array} { r } { p _ { D } ( \theta ) \propto \frac { p ( \theta ) } { \mathcal { L } ( x | \theta ) } . \dot { - } } \end{array}$ Consequently, given samples $\{ \theta ^ { i } \} _ { i = 1 } ^ { N }$ from $p ,$ an empirical approximation of $p _ { D }$ can be obtained by importance sampling, assigning weights $w _ { i } \propto 1 / \mathcal { L } ( x \mid \theta ^ { i } )$ followed by a resampling step. This inverse update allows the proposed algorithm to handle all possible orientations of the analogy relation while remaining entirely sampling-based.

## Empirical Validation

We evaluate the proposed sampling-based solver on a synthetic benchmark for which the analytical solution of the analogical equation is known. This setting allows us to assess the extent to which the proposed algorithm recovers the exact solution from sampled distributions.

Protocol. A total of $N = 3 1 2 1$ independent analogical equations are generated. Each analogy is constructed by first randomly sampling the parameters of a distribution $p _ { A }$ together with two regression observations under the linear regression likelihood. The corresponding posterior distributions $p _ { B }$ and $p _ { C }$ are then obtained analytically through Bayesian updating, while the target distribution $p _ { D }$ is computed exactly using Theorem 1. Consequently, the analytical solution of every analogical equation is known. For each analogy, only samples drawn from the distributions $p _ { A } , p _ { B } .$ and $p _ { C }$ are provided to the sampling algorithm. Each distribution is represented by 100 particles.

We evaluate two likelihood models: the normal likelihood and the linear regression likelihood. Although both belong to the class $\mathcal { L } _ { S } ,$ only the latter induces maximal analogies. As discussed above, the normal likelihood generates only a limited support of analogical transformations.

To recover the latent observations, we compare two optimization strategies: an exhaustive grid search and Bayesian optimization (Frazier 2018). For the normal likelihood, the search space is discretized into a 100 × 100 grid over $\mathcal { X } ^ { 2 }$ For the regression likelihood, each observation is parameterized by a predictor-response pair (�, �). We discretize the predictor into 15 values and the response into 31 values, yielding $1 5 \times 3 1 = 4 6 5$ candidate observations and therefore $\mathsf { 4 6 } 5 ^ { 2 } \equiv 2 1 6 , 2 2 5$ candidate observation pairs for a single analogical equation.

A candidate solution is retained only after two successive filtering steps. First, every importance-sampling update must satisfy an efective sample size $\mathrm { E S S } = 1 / \stackrel { \cdot } { \sum } _ { i } \stackrel { \cdot } { w _ { i } ^ { 2 } }$ of at least 15, ensuring that the posterior approximation is not dominated by only a few particles. Second, the candidate must satisfy the reconstruction constraints by requiring the sum of the three Wasserstein distances to remain below 0.5. This ensures both that the reconstructed distributions accurately approximate $p _ { B }$ and $p _ { C }$ , and that the two independent reconstructions of $p _ { D }$ are mutually consistent.

Metrics. The quality of the importance-sampling approximation is assessed through the efective sample size (ESS). The accuracy of the recovered solution is primarily evaluated using the 2-Wasserstein distance between the analytical Gaussian solution and the Gaussian fitted to the recovered particles. We additionally report the absolute errors on the recovered mean and variance, $( \hat { \mu } _ { D } , \hat { \sigma } _ { D } ^ { 2 } )$ , with respect to the analytical solution. Since $p _ { D }$ is reconstructed through two independent analogy paths, we conservatively report the minimum ESS of the two reconstructions.

Implementation and execution. The source code is available on the Github repository https://github.com/ppaamm/Analogies-on-Probability-

Distributions-by-Bayes-Updating. All experiments were performed on a laptop equipped with a 13th Gen Intel(R) Core(TM) i7-13700H CPU and 32 GB of RAM. No GPU acceleration was used.

Empirical results. For the normal loss, the algorithm does not recover any solution. This result was expected, since the support of the analogy is of measure zero in this case. This observation indicates that our algorithm does not retrieve incorrect solutions when the analogy is not maximal.

The grid search with the regression loss was performed only on a few analogies, because of very high computation times. Whereas a typical grid search lasted between 5 and 20 seconds for the search with the normal likelihood (grid of size 10, 000), it lasted up to 10 minutes with the regression likelihood (grid of size 216, 225). This encourages the use of Bayesian optimization for a faster search. The results presented below were all obtained used Bayesian Optimization.

The proposed solver successfully recovers solutions for 1918 out of the 3121 equations, i.e. 61.5% of the generated analogies. A qualitative analysis indicates that this percentage depends mostly on the number of iterations in the BO, limited here to 40.

Among successful runs, the approximation error remains generally small, with a median Gaussian Wasserstein distance of 0.132 and a 90th percentile of 0.421 (Figure 1a). The error distribution is strongly right-skewed, indicating that most analogies are solved accurately while a relatively small number of dificult instances produce substantially larger errors. An analysis of the correlation between the predicted parameters and the parameters of the analytical solutions (Figures 1b and 1c) shows that the error is mostly due to the variance that tends to be under-estimated. We observe on contrary a strong correlation in the means, with a Pearson correlation of 0.997.

Figure 2 investigates the influence of the efective sample size on the approximation quality. A clear decreasing trend is observed: both the median approximation error and its upper quantiles decrease as ESS increases. In particular, large approximation errors occur almost exclusively for small ESS values, whereas high ESS values consistently lead to accurate reconstructions. Nevertheless, accurate solutions can occasionally be obtained even for relatively small ESS, suggesting that ESS should primarily be interpreted as a measure of numerical reliability rather than a deterministic predictor of approximation accuracy.

Overall, these experiments provide empirical evidence that the proposed sampling algorithm accurately approximates analogical solutions whenever the importance sampling step remains suficiently efective. They further indicate that our current implementation has two main practical limitations: insuficient search misses solutions, and particle degeneracy globally produces higher approximation errors.

## Conclusion

In this paper, we introduced a characterization ofanalogies on probability distributions that satisfies the predicates of proportional analogy while relying on the intrinsic properties of probability distributions. Our definition yields a simple characterization of analogies between members of an exponential family, expressed as an arithmetic proportion between their natural parameters, up to an inversion property on a set of problems of measure zero. Unlike existing approaches, our framework also naturally extends to non-parametric families. To this end, we proposed a sampling-based algorithm that approximately solves analogical equations from samples of the first three distributions. A proof-of-concept experimental evaluation demonstrated the feasibility of this approach, although the current implementation remains limited by the

![](images/a03f47a3b78c542c1db52e73125afc38fab412973d8ec0d68726f6631fc6ec04.jpg)

![](images/88f16da529b684c4feae7f63fad383e9f0e33e5dc47968c390844297e9fd4338.jpg)

![](images/eaddfef4c372b6b025a9cb9141dcc29898e832134fdef22c387cb4cd7475b0ef.jpg)  
(a) Distribution of the Wasserstein distance. (b) Correlation between $\mu _ { D }$ and estimated $\hat { \mu } _ { D } .$ . (c) Correlation between $\sigma _ { D } ^ { 2 }$ and estimated $\hat { \sigma } _ { D } ^ { 2 } .$

Figure 1: Empirical evaluation of the sampled solutions. (a) Distribution of the Wasserstein distance between the analytical and recovered solutions. (b) Distribution of the efective sample size. (c) Relationship between the minimum efective sample size and the reconstruction error.

![](images/137882a0fd02dbcca5018e3f1555a588fe08e5ea547431e7a95748b68a330ce4.jpg)  
Figure 2: Comparison of the approximation error and the effective sample size. It can be observed that the approximation error tends to decrease when the ESS increases, which indicates that particle degeneracy is one of the main limitations of the current implementation.

approximation quality of importance sampling. Future work will investigate more advanced inference techniques, such as posterior sampling and Approximate Bayesian Computation (Sunnåker et al. 2013), to improve both the robustness and scalability of the solver.

More broadly, we believe that this work provides a principled foundation for connecting analogical reasoning with machine learning. Beyond its theoretical interest, the proposed framework opens new perspectives for applications such as transfer learning, data augmentation, distribution reconstruction, case-based reasoning, and meta-learning, where reasoning directly over probability distributions may provide a natural and expressive mechanism for knowledge transfer.

## Acknowledgments

The author would like to warmly thank Jean Lieber for the supporting discussions and feedback about this work.

## References

Antić, C. 2022. Analogical proportions. Annals of Mathematics and Artificial Intelligence, 90(6): 595–644.

Bayoudh, S.; Miclet, L.; and Delhay, A. 2007. Learning by Analogy: A Classification Rule for Binary and Nominal Data. In IJCAI, 678–683.

Brown, L. D. 1986. Fundamentals ofstatistical exponential families: with applications in statistical decision theory.

Couceiro, M.; Hug, N.; Prade, H.; and Richard, G. 2017. Analogy-preserving functions: a way to extend Boolean samples. In 26th International Joint Conference on Artificial Intelligence (IJCAI 2017), 1–7. International Joint Conferences on Artifical Intelligence (IJCAI).

Diaconis, P.; and Ylvisaker, D. 1979. Conjugate priors for exponential families. The Annals ofstatistics, 269–281.

Duck, J.; Schaller, R.; Auber, F.; Chaussy, Y.; Henriet, J.; Lieber, J.; Nauer, E.; and Prade, H. 2022. Analogy-based post-treatment of CNN image segmentations. In International Conference on Case-Based Reasoning, 318–332. Springer.

Frazier, P. I. 2018. Bayesian optimization. In Recent advances in optimization and modeling of contemporary problems, 255–278. Informs.

Gentner, D. 1983. Structure-mapping: A theoretical framework for analogy. Cognitive science, 7(2): 155–170.

Gick, M. L.; and Holyoak, K. J. 1980. Analogical problem solving. Cognitive psychology, 12(3): 306–355.

Hall, R. P. 1989. Computational approaches to analogical reasoning: A comparative analysis. Artificial intelligence, 39(1): 39–120.

Hug, N.; Prade, H.; Richard, G.; and Serrurier, M. 2019. Analogical proportion-based methods for recommendation– first investigations. Fuzzy sets and systems, 366: 110–132.

Klein, S. 1981. Culture, mysticism and social structure and the calculation of behavior. Technical report, University of Wisconsin-Madison Department of Computer Sciences.

Lee, J.; Sim, W.; Kim, S.; and Kim, S. 2024. Enhancing analogical reasoning in the Abstraction and Reasoning Corpus

via Model-Based RL. In Proceedings of the Workshop on the Interactions between Analogical Reasoning and Machine Learning (IARML@IJCAI), 1–12.

Lepage, Y. 1998. Solving analogies on words: an algorithm. In COLING 1998 volume 1: the 17th international conference on computational linguistics.

Lepage, Y. 2004. Analogy and formal languages. Electronic notes in theoretical computer science, 53: 180–191.

Lepage, Y.; and Couceiro, M. 2024. Any four real numbers are on all fours with analogy. arXiv preprint arXiv:2407.18770.

Miclet, L.; Bayoudh, S.; and Delhay, A. 2008. Analogical dissimilarity: definition, algorithms and two experiments in machine learning. Journal of Artificial Intelligence Research, 32: 793–824.

Mikolov, T.; Chen, K.; Corrado, G.; and Dean, J. 2013. Eficient Estimation of Word Representations in Vector Space. In 1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings.

Mitchell, M. 2021. Abstraction and analogy-making in artificial intelligence. Annals of the New York Academy of Sciences, 1505(1): 79–101.

Murena, P.-A. 2022. Measuring the feasibility of analogical transfer using complexity. arXiv preprint arXiv:2206.11753.

Murena, P.-A.; Cornuéjols, A.; and Dessalles, J.-L. 2018. Opening the parallelogram: Considerations on non-Euclidean analogies. In International Conference on Case-Based Reasoning, 597–611. Springer.

Nielsen, F. 2020. An elementary introduction to information geometry. Entropy, 22(10): 1100.

Prade, H.; and Richard, G. 2017. Boolean analogical proportions-axiomatics and algorithmic complexity issues. In European Conference on Symbolic and Quantitative Approaches to Reasoning and Uncertainty, 10–21. Springer.

Prade, H.; and Richard, G. 2021. Analogical Proportions: Why They Are Useful in AI. In IJCAI, volume 2021, 4568– 4576.

Prade, H.; and Richard, G. 2025. Analogical Proportions Between Probabilities. In European Conference on Symbolic and Quantitative Approaches with Uncertainty, 148– 163. Springer.

Robert, C. P.; Casella, G.; and Casella, G. 2004. Monte Carlo statistical methods, volume 2. Springer.

Rumelhart, D. E.; and Abrahamson, A. A. 1973. A model for analogical reasoning. Cognitive Psychology, 5(1): 1–28.

Stroppa, N.; and Yvon, F. 2005. An analogical learner for morphological analysis. In Proceedings of the Ninth Conference on Computational Natural Language Learning (CoNLL-2005), 120–127.

Sunnåker, M.; Busetto, A. G.; Numminen, E.; Corander, J.; Foll, M.; and Dessimoz, C. 2013. Approximate bayesian computation. PLoS computational biology, 9(1): e1002803.

Webb, T.; Holyoak, K. J.; and Lu, H. 2023. Emergent analogical reasoning in large language models. Nature Human Behaviour, 7(9): 1526–1541.