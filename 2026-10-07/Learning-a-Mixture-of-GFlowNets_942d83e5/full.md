# Learning a Mixture of GFlowNets

Tiago da Silva MBZUAI, UAE tiago.dasilva@mbzuai.ac.ae

Amauri H. Souza Avra, Brazil & Federal Institute of Cear´a, Brazil amauri.holanda@avra.ai

Salem Lahlou MBZUAI, UAE salem.lahlou@mbzuai.ac.ae

October 7, 2026

## Abstract

Learning an ensemble of GFlowNets to sample from a discrete target distribution has become a common approach for achieving better state space exploration and convergence than that of a monolithic sam pler. However, these methods often add a substantial runtime overhead to the base model, and their con ceptual connection remains elusive. To address this, we first propose a general-purpose theoretical framework for describing a mixture of GFlowNets, which we specialize into continuously (CI) and discretely indexed (DI) collections. On the one hand, we show CI GFlowNets can be interpreted through the lens of a random features expansion, provably boosting the sampler’s expressivity in graph-structured tasks and reducing learning instability via spectral shifting. On the other hand, we demonstrate DI GFlowNets encompass prior approaches for GFlowNet training and provide the foundation for the newly proposed Stratum-Conditioned (SC) GFlowNets. This method, which is inspired by the Doob’s h-transform of Markov chains, decomposes the state space according to a prescribed modular function and restricts each component to sample from a distinct subset of it. Importantly, SC GFlowNets support centralized and component-wise embarrassingly parallel training, and we show both of them significantly speed up learning convergence and mode coverage without introducing any non-negligible extra computation.

## 1 Introduction

Suppose we are given an unnormalized distribution R over a discrete space . A common problem in both statistical inference and probabilistic machine learning is to draw samples from distributed according to R (Bishop, 2007; Neal et al., 2011; Blei & et al., 2017). When is intractably large—e.g., the space of graphs containing N nodes or of subsets of a set with N elements for largish N—addressing this challenge often requires the use of approximate algorithms such as Markov chain Monte Carlo (MCMC) or variational inference (VI). Notably, MCMC sufers from slow convergence and correlated samples, especially in discrete spaces, and its correctness cannot be tractably assessed in general (Owen, 2013). VI, in contrast, reframes the sampling problem as that of finding the best approximation to R within a tractable family, thereafter enabling the generation of independent samples from a measurably correct distribution (Malkin et al., 2023; Zimmermann et al., 2023).

As a major milestone of VI for discrete spaces, GFlowNets (Bengio et al., 2021; Lahlou et al., 2023) learn a sequential decision process that iteratively constructs each x   from an initial (empty) state in proportion to R(x). Since their conception, they have been successfully deployed into applications ranging from causal discovery (Deleu et al., 2022, 2023) to LLM fine-tuning (Hu et al., 2023; Venkatraman et al., 2024) and phylogenetic inference (Zhou et al., 2024). Despite their potential, they sufer from insuficient state space exploration (Kim et al., 2025b; Madan et al., 2025), slow training convergence (Liu et al., 2024), and learning instability (Figure 10), often failing to simulate even simple multi-modal distributions defined over relatively small  (Figure 20).

![](images/82d7774ed40c81e35b26406bff8035604ec6395e26be71f2231b0bf1a64078c7.jpg)  
Figure 1: An ensemble of GFlowNets that, when averaged, matches the target.

To mitigate this issue, recent works have proposed learning an ensemble of GFlowNets in which each component focuses on a nearly-unique subset of the state space (Yu et al., 2026a,b; Dall’Antonia et al., 2026b); in doing so, they incentivize the exploration of unvisited high-probability regions and accelerate learning convergence. However, they often incur a substantial computational overhead—requiring several neural networks to be separately trained—and, despite being similar, lack a shared conceptual foundation. In this scenario, our first contribution

is to introduce a unifying theory to describe a mixture of GFlowNets (Figure 1), shedding light on the applicability and limitations of such an approach. Drawing on this framework, we propose distinct algorithms that we show to both stabilize and accelerate learning convergence.

In this context, we separate our presentation into continuously (CI) and discretely indexed (DI) mixtures of GFlowNets. On the one hand, CI GFlowNets learn an uncountable collection of samplers, each of which is associated to a real-valued vector whose value serves as an extra input to the underlying neural network parameterizing the sampler. We demonstrate this formulation is connected to the theory of random features for graph learning (Sato et al., 2021; Abboud et al., 2021), providing a computationally simple approach to provably boost the sampler’s expressivity in graph-structured tasks (Sections C and 4). In addition, we show CI GFlowNets stabilize learning by shifting the spectrum of the loss function’s Hessian, reducing the dependence of the learned model on exogenous factors such as the initial value of the pseudorandom number generator (Proposition C.7).

On the other hand, DI GFlowNets consist of a finite (or countably infinite) mixture of samplers. Each sampler learns in proportion to a re-weighted target; for the kth component, this is $x \mapsto R ( x ) q ( k | x )$ for a posterior mixing distribution $q ( \cdot | x )$ (see Section 5). In this sense, DI GFlowNets support two learning modalities: centralized, with an embedding of k being used as extra input to a component-wise shared model, and embarrassingly parallel (Neiswanger et al., 2014), with distinct components being independently trained across non-communicating processes. We prove this framework encompasses two prior methods, Subgraph Asynchronous Learning (Silva et al., 2025b) and Boosted GFlowNets (Dall’Antonia et al., 2026b), unifying otherwise disparate algorithms.

As the simplest non-trivial instantiation of DI GFlowNets, we propose setting $q ( \cdot | x )$ as a sparse distribution assigning zero probability to all but a few components. This efectively corresponds to defining a covering of the state space. We show, however, that conditioning a sampler to exclusively explore an arbitrary subset of is NP-complete in general. We consequently introduce Stratum-Conditioning (SC) as a tractable approach for learning DI GFlowNets. Simply put, SC partitions the state space by binning the outputs of a modular and real-valued function defined on , with each bin being called a stratum (hence the name). This function can be set to the trajectory length required for reaching $x \in { \mathcal { X } } ,$ for instance, or as the distance between $x \in \mathcal { X }$ and a chosen reference point according to a prescribed metric (Section 5.1). We demonstrate the set of samplers learned in this way are connected via Doob’s h-transform of a base GFlowNet (Section 5.2).

Our experiments highlight SC GFlowNets substantially improve both learning convergence and state space exploration across diverse benchmark tasks. Importantly, contrarily to recent algorithmic advances in the GFlowNet literature, SC adds no computational overhead to training, being as fast to train as a singlecomponent sampler (see Figure 4 and Figures 11 to 13). As with most of machine learning research, which historically started from simple algorithms that have been incrementally enhanced over time, we believe SC GFlowNets set a minimal baseline for learning a mixture of GFlowNets. In summary, our contributions are as follows.

1. We introduce an abstract framework for learning an ensemble of GFlowNets, demonstrating its usefulness in providing an unifying view of prior techniques and in designing novel algorithms.

2. We define continuously (CI) and discretely indexed (DI) GFlowNets, outlining their empirical and theo-

retical advantages over a single-component sampler.

3. We propose Stratum-Conditioned (SC) GFlowNets as an efective form of learning DI GFlowNets, supporting both centralized and parallel training.

4. We show that SC GFlowNets speed up convergence and mode discovery across benchmark tasks.

## 2 Preliminaries

Definitions. We let and be finite spaces, with $s$ corresponding to partial realizations of $\mathcal { X } .$ . We also let $s _ { o } \in S$ be the initial state from which any $x \in \mathcal { X }$ can be constructed. An example the reader should keep in mind is when is the set of subsets of size K of a given finite set and is the set of subsets of size less than K, with $s _ { o } = \emptyset$ . We call a state graph a directed acyclic graph having $s _ { o }$ as the only state without incoming edges and $x \in \mathcal { X }$ as the only states without outgoing edges, which we call terminal, and such that there is a path from $s _ { o }$ to every x $\in \mathcal { X }$ containing only states from ${ \mathcal { S } } .$

Notationally, we let $[ N ] = \{ 1 , \dots , N \}$ for each positive integer $N \in  { \mathbb { N } }$ and $s  s ^ { \prime }$ be the set of trajectories $\tau = \{ ( s , s _ { 1 } ) , ( s _ { 1 } , s _ { 2 } ) , \ldots , ( s _ { d } , s ^ { \prime } ) \}$ from $s \in S$ to $s ^ { \prime } \in \mathcal { S } \cup \mathcal { X }$

GFlowNets. A GFlowNet learns to stochastically construct $x \in \mathcal { X }$ from $s _ { o }$ by following a policy $p _ { F } \colon S \times$ $( S \cup \mathcal { X } )  [ 0 , 1 ]$ such that $p _ { F } ( s , s ^ { \prime } ) > 0$ if and only if $s  s ^ { \prime }$ in the state graph. The objective is for the marginal distribution of $p _ { F }$ over to match the unnormalized distribution $R \colon \mathcal X \to \mathbb { R } _ { + }$ from which we aim to sample, i.e.,

$$
m _ { F } ( x ) : = \sum _ { \tau \in s _ { o }  x } \prod _ { ( s _ { i } , s _ { i + 1 } ) \in \tau } p _ { F } ( s _ { i } , s _ { i + 1 } ) : = \sum _ { \tau \in s _ { o }  x } p _ { F } ( s _ { o } , \tau ) \propto R ( x ) .\tag{1}
$$

As $m _ { F } ( x )$ is generally intractable, we introduce a backward policy $p _ { B } ( x , \tau )$ representing the probability of having followed τ given that x was sampled. We then search for $\left( p _ { F } , p _ { B } \right)$ satisfying a consistency condition, with a popular choice being the trajectory balance (Malkin et al., 2022, TB): $Z \cdot p _ { F } ( s _ { o } , \tau ) = R ( x ) \cdot p _ { B } ( x , \tau )$ and Z as a learned constant. In practice, $p _ { F } ( s , \cdot )$ is parameterized as a neural network and this search is carried out via stochastic gradient descent. Please refer to Sections A and I for a comprehensive background and recent advances on GFlowNets.

## 3 Learning a Mixture of GFlowNets

Simply put, the central challenges in achieving Equation (1) are the problem’s under-determination, owning to the existence (in general) of infinitely many pairs $\left( p _ { F } , p _ { B } \right)$ abiding by any given consistency condition, and sample ineficiency, requiring extensive state graph exploration for finding a resolutive $\left( p _ { F } , p _ { B } \right)$ . We will show a mixture formulation can naturally address both of them, focusing on the latter in the main text and elaborating on the former in Section C of the supplement.

Towards this objective, we develop an abstract framework for learning a mixture of GFlowNets. For starters, we let Γ be an index space and $\{ p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) } \} _ { \gamma \in \Gamma }$ be the corresponding set of policies. Our goal is for the marginal distribution of these policies on each $x \in \mathcal { X }$ , when weightedly averaged over $\gamma \in \Gamma$ , to match the target $R ( x )$ . Formally, we assume Γ to be embedded in a topological measurable space equipped with a measure ν over its Borel σ-algebra. This allows us to set aside the specific properties of Γ, while accounting for common use cases via a unified language.

Remark 3.1 (Γ can be either discrete or continuous). Common choices for Γ and ν are $\Gamma = \mathbb { R } ^ { d _ { \gamma } }$ with ν as the Lebesgue measure and $\Gamma = [ K ]$ with ν as the counting measure (Sections C and 5).

Designing a mixture of GFlowNets. With this setup, we let $p _ { M }$ be a probability measure on Γ absolutely continuous with respect to ν. The subscript M indicates $p _ { M }$ is a mixing $( p r i o r )$ distribution over Γ.

Our goal is for the marginal identity analogous to Equation (1)

$$
\int _ { \Gamma } p _ { M } ( { \mathrm { d } } \gamma ) \sum _ { \tau \in s _ { o }  x } p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) : = \int _ { \Gamma } p _ { M } ( { \mathrm { d } } \gamma ) m _ { F } ^ { ( \gamma ) } ( x ) \propto R ( x )\tag{2}
$$

to be satisfied. As in Equation (1), we call $m _ { F } ^ { ( \gamma ) } ( x )$ the marginal of $p _ { F } ^ { ( \gamma ) } ( s _ { o } , \cdot )$ over  . Under this condition, we can sample from R by first drawing $\gamma \sim p _ { M } ( \mathrm { d } \gamma )$ and then sampling $x \sim m _ { F } ^ { ( \gamma ) } ( x )$ . Nevertheless, Equation (2) is doubly intractable: neither $m _ { F } ^ { ( \gamma ) }$ nor its average under $\gamma \sim p _ { M }$ can be computed eficiently. We circumvent this issue by introducing a mixing (posterior) distribution $q _ { M } ( \mathrm { d } \gamma | x )$ and rewriting Equation (2) as<sup>1</sup>

$$
p _ { M } ( \mathrm { d } \gamma ) m _ { F } ^ { ( \gamma ) } ( x ) \propto R ( x ) q _ { M } ( \mathrm { d } \gamma | x ) .\tag{3}
$$

The role of $q _ { M }$ in the equation above is akin to that of responsibilities in expectation-maximization algorithms (Dempster et al., 1977), representing the probability a given $x \in \mathcal { X }$ was drawn from a γ-indexed sampler. As in Section 2, we turn Equation (3) into a learning objective—referred to as mixed trajectory balance (MTB)—by replacing the intractable marginal with tractable policies.

Definition 3.2 (MTB condition). Let ${ \mathfrak { g } } _ { \gamma } = ( p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) } )$ and $\{ { \mathfrak { g } } _ { \gamma } \} _ { \gamma \in \Gamma }$ be a Γ-indexed set of GFlowNets, and $p _ { M } ( \mathrm { d } \gamma )$ and $q _ { M } ( \mathrm { d } \gamma | x )$ be prior and posterior mixing distributions over Γ. Then,

$$
p _ { M } ( \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) = \frac { R ( x ) } { Z } q _ { M } ( \mathrm { d } \gamma | x ) p _ { B } ^ { ( \gamma ) } ( x , \tau )
$$

for each $x \in \mathcal { X }$ and $\tau \in s _ { o }  x$ is the mixed trajectory balance condition, with $\begin{array} { r } { Z = \sum _ { x \in \mathcal { X } } R ( x ) } \end{array}$

The suficiency of Definition 3.2 for Equation (2) is a direct consequence of (Malkin et al., 2022, Proposition 1). As it will soon become clear, it will be notationally convenient to define

$$
Z ( \mathrm { d } \gamma ) = Z \cdot p _ { M } ( \mathrm { d } \gamma ) { \mathrm { ~ a n d ~ } } R ( x , \mathrm { d } \gamma ) = R ( x ) q _ { M } ( \mathrm { d } \gamma | x )
$$

as unnormalized measures over Γ and $\mathcal { X } \times \Gamma$ . This allows us to rewrite MTB as

$$
Z ( { \mathrm { d } } \gamma ) \cdot p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) = R ( x , { \mathrm { d } } \gamma ) \cdot p _ { B } ^ { ( \gamma ) } ( x , \tau ) ,
$$

resembling the familiar TB condition. We also deduce from this that other balance conditions, such as Detailed Balance (DB) (Bengio et al., 2023) and SubTB (Madan et al., 2022), remain valid in our setting. Given a function $F \colon s \in { \mathcal { S } } \mapsto F ( s , \mathrm { d } \gamma )$ mapping s to a measure in Γ, for instance, the DB condition can be written as, for $s \in \mathcal { S } , s ^ { \prime } \in \mathcal { S } \cup \mathcal { X }$ , and $x \in \mathcal { X }$

$$
F ( s , \mathrm { d } \gamma ) \cdot p _ { F } ^ { ( \gamma ) } ( s , s ^ { \prime } ) = F ( s ^ { \prime } , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( s ^ { \prime } , s ) { \mathrm { ~ a n d ~ } } F ( x , \mathrm { d } \gamma ) = R ( x , \mathrm { d } \gamma ) .
$$

This raises the question of how to select both $Z ( \mathrm { d } \gamma )$ and $R ( x , \mathrm { d } \gamma )$ , which will be addressed in Section 5. Under our abstract setting, however, we show both measures are intimately related.

Proposition 3.3 (Relationship between $Z ( \mathrm { d } \gamma )$ and $R ( x , \mathrm { d } \gamma ) )$ . Let $\{ { \mathfrak { g } } _ { \gamma } \} _ { \gamma \in \Gamma }$ and $p _ { M }$ and $q _ { M }$ be as in $D e f i -$ nition 3.2, and $Z ( \mathrm { d } \gamma )$ and $R ( x , \mathrm { d } \gamma )$ be as above. Then, $Z ( \mathrm { d } \gamma )$ and $R ( x , \mathrm { d } \gamma )$ jointly satisfy

$$
Z ( \mathrm { d } \gamma ) = \sum _ { x \in \mathcal { X } } R ( x , \mathrm { d } \gamma ) \ a n d \ R ( x , \mathrm { d } \gamma ) = Z ( \mathrm { d } \gamma ) \cdot m _ { F } ^ { ( \gamma ) } ( x ) .
$$

Under Proposition 3.3, $Z ( \mathrm { d } \gamma )$ may be understood as the fraction of total probability mass $( Z )$ being assigned to the model indexed by γ, while $x \mapsto R ( x , \mathrm { d } \gamma )$ represents the (unnormalized) marginal distribution of such model over  . This supports the interpretation of Γ-indexing as a soft partition of  : each ${ \mathfrak { g } } _ { \gamma }$ primarily focuses on a (potentially) distinct subset of .

Learning a GFlowNet ensemble. We learn a MTB-abiding $\mathfrak { g } _ { \Gamma } = ( p _ { M } , q _ { M } , \{ \mathfrak { g } _ { \gamma } \} _ { \gamma \in \Gamma } )$ by optimizing

$$
\mathcal { L } _ { \mathrm { T B } } ( \tau , \mathrm { d } \gamma ; \mathfrak { g } _ { \Gamma } ) = \left( \log \frac { Z ( \mathrm { d } \gamma ) \cdot p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) } { R ( x , \mathrm { d } \gamma ) \cdot p _ { B } ^ { ( \gamma ) } ( x , \tau ) } \right) ^ { 2 }\tag{4}
$$

for all $\tau , \gamma ;$ we call $\mathcal { L } _ { \mathrm { T B } }$ the TB loss function (Malkin et al., 2022). As an aside, we notice in Definition B.2 that similar learning objectives from Bengio et al. (2023); Deleu et al. (2024), such as SubTB and DB, can also be seamlessly extended to our setting.

From this, learning continues business as usual: we parameterize $p _ { F } ^ { ( \gamma ) } ( s , \cdot )$ and (optionally) $p _ { B } ^ { ( \gamma ) } ( s , \cdot )$ as neural networks receiving both s and $\gamma$ as inputs and, defining exploratory (full support) distributions $\rho _ { M } ( \mathrm { d } \gamma )$ and $\rho ^ { ( \gamma ) } ( s _ { o } , \tau )$ , we solve the stochastic program

$$
\operatorname* { m i n } _ { \mathfrak { g } _ { \Gamma } } \mathbb { E } _ { B \sim \rho _ { M } ( \cdot ) , \tau \sim \rho ^ { ( \gamma ) } ( s _ { o } , \cdot ) } \left[ \mathcal { L } _ { \mathrm { T B } } \left( \tau , B ; \mathfrak { g } _ { \Gamma } \right) \right]\tag{5}
$$

via stochastic gradient descent (Jordan et al., 2024). In the following sections, we present both specific instantiations and an empirical validation of the proposed framework.

Remark 3.4 (Modularity). We also notice methods for learning both $p _ { F }$ and $\rho$ in Equation (5) can be directly applied to our setting (Kim et al., 2025b; Madan et al., 2025; Dall’Antonia et al., 2026a).

## 4 Continuously Indexed GFlowNets

The most straightforward form of Equation (5) assumes $q ( \mathrm { d } \gamma | x ) = p _ { M } ( \mathrm { d } \gamma )$ , i.e., each component is equally likely to generate $x \in \mathcal { X }$ . In this case, the ratio $Z ( \mathrm { d } \gamma ) \big / R ( x , \mathrm { d } \gamma ) = Z \big / R ( x )$ becomes γ-independent, and our prob lem reduces to minimizing the familiar TB loss averaged over $\gamma \in \Gamma$ (Malkin et al., 2022). Although deceptively simple, this approach corresponds to augmenting our model with random features (RFs) when $\Gamma = \mathbb { R } ^ { d _ { \gamma } }$ This is a well-established procedure for boosting the expressivity of graph neural networks (Abboud et al., 2021), often-used by GFlowNets in graph-structured spaces. Drawing on this, we show continuously indexed (CI) GFlowNets—as we call the RF-augmented sampler—can arbitrarily approximate any graph distribution.

Remark 4.1 (CI GFlowNets as Universal Approximators). CI GFlowNets are strictly more expressive, in terms of which distributions they can realize, than GFlowNets (see Example C.3). In particular, when considering graph-structured tasks based on edge-additive generative processes, CI GFlowNets are universal approximators (Proposition C.5).

We formalize this discussion in Section C, where we also demonstrate CI GFlowNets can have a smoothing efect on the loss surface (as in Bishop, 1995), significantly reducing training instability.

## 5 Discretely Indexed GFlowNets

To start our discussion on discrete GFlowNet mixtures, we first notice that the TB loss function in Equation (4) does not impose a preference of a non-trivial solution over a collapsed sampler in which $Z ( \mathrm { d } \gamma ) = Z$ and $R ( x , \mathrm { d } \gamma ) = R ( x )$ or, more generally, in which the ratio $R ( x , \mathrm { d } \gamma ) \big / Z ( \mathrm { d } \gamma )$ does not depend on γ. Although a γ-independent target can exhibit beneficial properties, such as greater expressivity and a smoother loss surface (as in Section 4), this should be a design decision—achieved by setting $q _ { M } ( \mathrm { d } \gamma | x ) = p _ { M } ( \mathrm { d } \gamma )$ —rather than a consequence of training. In practice, learning a non-collapsed mixture from Equation (5) requires constraining the search space for the mixing distributions $q _ { M }$ and $p _ { M }$ either via soft regularization or hard constraints.

We advocate for the latter in this section, suggesting to fix Γ as a finite set and $q _ { M } ( \mathrm { d } \gamma | x )$ as a sparse distribution assigning zero probability to most $\gamma \in \Gamma$ . We also briefly elaborate on using a countably infinite Γ when discussing Boosted GFlowNets (Dall’Antonia et al., 2026b) in Section F.

Discretely Indexed GFlowNets. We refer to the corresponding sampler as Discretely Indexed (DI) GFlowNets. As notational shortcuts, we let $\Gamma = [ K ]$ , and write the measures $Z$ and R and $q _ { M }$ as $Z ( \{ k \} ) = Z _ { k }$ and $R ( x , \{ k \} ) = R _ { k } ( x )$ and $q _ { M } ( \{ k \} | x ) = q _ { k } ( x )$ , respectively. In addition, we let $\mathbf { Z } = ( Z _ { k } ) _ { k \in [ K ] }$ and $\rho _ { k }$ be the exploration policy for the mixture’s kth component. Our objectives are to define $q _ { k }$ and $\rho _ { k } ;$ as with single-component GFlowNets, we set Z as a learnable parameter.

## 5.1 Stratum-Conditioned GFlowNets

In this context, we propose setting $q _ { k } ( x )$ according to a hand-crafted covering $\{ X _ { k } \} _ { k \in [ K ] }$ of $\mathcal { X } :$ under this condition, $q _ { k } ( x )$ corresponds to a uniform distribution over the indices k for which $\mathcal { X } _ { k }$ contains x. We call each $\mathcal { X } _ { k }$ a stratum due to its construction as the pre-image of an interval with respect to a real-valued function on $x ;$ see Remark 5.5. This section formalizes the ensuing algorithm, called Stratum-Conditioned (SC) GFlowNets, explains how it can be implemented, and describes suficient conditions for which sampling from $\mathcal { X } _ { k }$ is computationally tractable.

Stratum Conditioning. The first requirement for tractability is for us to know how many sets $N ( x )$ in $\{ \mathcal { X } _ { k } \} _ { k \in [ K ] }$ contain a given $x ,$ allowing for the eficient computation of $q _ { k } ( x )$ . As such, we define a tractable K-cover of , denoted as $\mathrm { T C } ( \mathcal { X } , K )$

Definition 5.1 (Tractable K-Cover). Let K be a positive integer and be a set. We say $\mathrm { T C } ( \mathcal { X } , K ) =$ $\{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ is a Tractable K-Cover of if the following conditions are satisfied.

1. $\mathrm { T C } ( \mathcal { X } , K )$ is an exact cover of  , i.e., each $\mathcal { X } _ { k } \subseteq \mathcal { X }$ and $\textstyle \bigcup _ { k = 1 } ^ { K } \mathcal { X } _ { k } = \mathcal { X }$

2. For each $x \in { \mathcal { X } } .$ the quantity $N ( x ) : = \# \{ k \in [ K ] \colon x \in \mathcal { X } _ { k } \}$ is $\mathrm { k n o w n ^ { 2 } }$

The second condition allows eficient querying of $q _ { k } ( x )$ , defined as $q _ { k } ( x ) = N ( x ) ^ { - 1 } { \mathrm { ~ i f ~ } } x \in \mathcal { X } _ { k }$ and $q _ { k } ( x ) = 0$ otherwise. In practice, we often let $\mathrm { T C } ( \mathcal { X } , K )$ be a disjoint cover—a partition—of ; in this case, $q _ { k } ( x ) = 1$ for the only k with $x \in \mathcal { X } _ { k }$ . A corollary of Proposition 3.3 cleanly characterizes both $Z _ { k }$ and $R _ { k } ( x )$ as a fraction of the probability masses assigned to $\mathcal { X } _ { k }$ and x under a TC.

Corollary 5.2 $( Z _ { k }$ as the mass within ${ \boldsymbol { \mathcal { X } } } _ { k } )$ . Let $\mathrm { T C } ( \mathcal { X } , K ) = \{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ and ${ \mathfrak { g } } _ { [ K ] }$ be a GFlowNet ensemble satisfying the MTB condition in Definition 3.2. Then, denoting $b y ~ \mathbb { 1 } [ \cdot ]$ the indicator function,

$$
Z _ { k } = \sum _ { x \in \mathcal { X } _ { k } } \frac { R ( x ) } { N ( x ) } \ a n d \ R _ { k } ( x ) = \frac { R ( x ) } { N ( x ) } \cdot \mathbb { 1 } [ x \in \mathcal { X } _ { k } ] .
$$

In particular, when $N ( x )$ is constant, the learned policy $p _ { F } ^ { ( k ) }$ generates samples in proportion to the restriction of $R ( x )$ to $\mathcal { X } _ { k }$ , a property that might be of independent interest when the covering has a meaningfu semantic value for the problem at hand.

Remark 5.3 $( R _ { k } ( x )$ for a disjoint cover). Under the conditions of Corollary 5.2, assume $N ( x ) = N \left( \mathrm { e . g . } \right.$ $N = 1$ for a disjoint cover). Then, $m _ { F } ^ { ( k ) } ( \bar { x } ) \propto R ( x ) \cdot \mathbb { 1 } [ x \in \mathcal { X } _ { k } ]$

We emphasize $\{ p _ { F } ^ { ( k ) } \} _ { k \in [ K ] }$ are parameterizable via a shared neural network architecture receiving that receives an embedding of k as input; they need not be completely separate samplers. To understand the benefits of partitioning, we consider the Lines domain (Silva et al., 2026), described next.

Example 5.4 (Lines domain). We define ${ \cal S } = \{ 1 , \ldots , N \}$ and $\mathcal { X } = \{ 1 , \ldots , N \} \times \{ \top \}$ with $N \geq 2$ and indicating the state is in and not in ${ \mathcal { S } } .$ . The initial state is $s _ { o } = 1$ and each transition corresponds to adding 1 to the current state if it is less than $N$ or attaching and terminating. Let

$$
\mathcal X _ { 1 } = \{ 1 , \dotsc , \lfloor { \cal N } / 3 \rfloor \} , \ \mathcal X _ { 2 } = \{ \lfloor { \cal N } / 3 \rfloor + 1 , \dotsc , \lfloor { \displaystyle 2 } \cdot { \cal N } / 3 \rfloor \} , \ \mathrm { a n d } \ \mathcal X _ { 3 } = \{ \lfloor { \displaystyle 2 } \cdot { \cal N } / 3 \rfloor + 1 , \dotsc , { \cal N } \}
$$

![](images/86eefd1e36a6ace851f39bcb891fd8b602334986d6fcf155f83c1de8227524e8.jpg)  
Figure 2: SC GFlowNets for the Lines environment $( N = 2 5 6 )$ . (Left) We show the TV distance between the learned and target distributions, emphasizing the faster convergence promoted by stratum-conditioning. (Middle) The marginal distribution of both models after 1000 training steps, and the target. $( \mathrm { R i g h t } )$ The marginal distribution of SC GFlowNets for each stratum $k \in \{ 1 , 2 , 3 \}$

be the partition. We design $p _ { F } ^ { ( k ) } ( s , \cdot )$ by masking out the terminating transition if $s \notin \mathcal { X } _ { k }$ and masking out the successor transition if $s + 1 \notin \mathcal { X } _ { k }$ , i.e.,

$$
p _ { F } ^ { ( k ) } ( s , s + 1 ) \propto \mathbb { 1 } [ s + 1 \in \mathcal { X } _ { k } ] \cdot e ^ { \phi ( s , k ) ^ { \top } w _ { 1 } } \mathrm { ~ a n d ~ } p _ { F } ^ { ( k ) } ( s , ( s , \top ) ) \propto \mathbb { 1 } [ s \in \mathcal { X } _ { k } ] \cdot e ^ { \phi ( s , k ) ^ { \top } w _ { 2 } } ,
$$

with $\phi \colon [ N ] \times [ K ] \to \mathbb { R } ^ { d }$ and $w _ { 1 } , w _ { 2 } \in \mathbb { R } ^ { d }$ being learnable parameters. Figure 2 shows the error curves measured in terms of total variation distance (TV) for (i) a GFlowNet and (ii) a SC GFlowNet for the target distribution $\begin{array} { r } { R ( x ) = \exp \{ - ( \beta N ) ^ { - 1 } \cdot \operatorname* { m i n } _ { y \in \{ 3 , N - 2 \} } | x - y | \} } \end{array}$ with $N = 2 5 6$ and $\beta = 0 . 1$ . Notably, while SC GFlowNet quickly converges to the target distribution, a GFlowNet struggles to achieve a comparable goodness-of-fit, requiring over three times more training steps.

How to design a tractable covering? A pivotal question for SC GFlowNets is how to define the mapping $\mathrm { T C } ( \mathcal { X } , K )$ for each  and K. Although distinct problems may benefit from idiosyncratic strategies, we provide next a blueprint for designing $\mathrm { T C } ( \mathcal { X } , K )$ in commonly encountered tasks.

Remark 5.5 (Modulo-based TC). Example 5.4 hints on how to construct TC: we let $\mathcal { X } _ { k }$ be the set of states whose distance to a reference point is within a prescribed interval. As most state spaces of interest lack an inherent geometric structure, however, we use a modular function as a general proxy for distance. To make sense of such definition, recall a compositional object can be described as a multisubset of a component set , i.e., each $s \in { \mathcal { S } } \cup { \mathcal { X } }$ can be identified with $s \subseteq { \mathcal { C } }$ . In Example 5.4, for instance, $\mathcal { C } = \{ 1 , \top \}$ and $s \equiv \{ \{ 1 , \ldots , 1 \} \}$ (s times). In this scenario, let ψ be a bounded modular function with $\psi ( \{ c \} ) \geq 0$ for all $c \in { \mathcal { C } }$ . We let

$$
\mathcal { X } _ { k } = \left\{ x \in \mathcal { X } \colon \frac { k - 1 } { K } ( \psi ( \mathcal { C } ) + 1 ) \le \psi ( x ) < \frac { k } { K } ( \psi ( \mathcal { C } ) + 1 ) \right\} .
$$

When $\psi ( s ) = \# s$ , we call $\mathrm { T C } ( \mathcal { X } , K )$ to be a length-based covering. In general, we let $\psi ( s ) = \# ( s \cap r )$ for a given reference set $r \subseteq { \mathcal { C } } ; \operatorname { i f } r = \emptyset$ , the SC GFlowNet reduces to a GFlowNet.

Importantly, while easily solved for some $\psi$ such as the length-based one in Example 5.4, we show next that the decision problem of inferring whether $\mathcal { X } _ { k }$ is reachable from a state in is NP-complete in general. This is required for masking out from $p _ { F } ^ { ( k ) }$ transitions leading to states for which $\mathcal { X } _ { k }$ becomes unreachable, ensuring the sampler does not become deadlocked in a zero-probability region.

Proposition 5.6 (Sampling from a general TC is NP-complete). Let $\mathrm { T C } ( \mathcal { X } , K )$ be a modulo-based TC built upon a modular function ψ and $\mathcal { X } _ { k } = \{ x \in \mathcal { X } \colon \psi ( x ) \in [ a _ { k } , b _ { k } ) \}$ for numbers $\{ ( a _ { k } , b _ { k } ) \} _ { k = 1 } ^ { K }$ . Let $s \in \mathcal { S } \subset 2 ^ { \mathcal { C } }$ Given $k \in [ K ]$ , we consider the following decision problem,

$$
\exists s ^ { \prime } \in 2 ^ { \mathcal { C } \setminus s } ~ s . t . ~ s \cup s ^ { \prime } \in \mathcal { X } ~ a n d ~ \psi ( s \cup s ^ { \prime } ) \in [ a _ { k } , b _ { k } ) ,\tag{6}
$$

i.e., whether there is a terminal $x \in \mathcal { X } _ { k }$ reachable from $s \in S$ . Then, if ψ is an arbitrary modular function evaluable in polynomial time on the size of its input, Equation (6) is NP-complete.

We demonstrate Proposition 5.6 by reducing Equation (6) to the SubsetSum problem, which is NPcomplete (Cormen et al., 2009). Rather than a strict limitation of SC GFlowNets, however, Proposition 5.6 should be read as a guiding design principle for $\mathrm { T C } ( \mathcal { X } , K )$ , suggesting there are partitions for which sampling cannot be eficiently implemented. In contrast, drawing on Example 5.4, we show there is a specific class of functions $\psi$ for which Equation (6) is decidable in polynomial time.

Proposition 5.7 (Eficient sampling from symmetric junta-based TC). We call a function $\psi \colon 2 ^ { \mathcal { C } }  ]$ R a symmetric junta if there are a reference set $r \subseteq 2 ^ { c }$ and a function $\xi \colon \mathbb { Z } \to \mathbb { R } ~ f o r$ which $\psi ( x ) = \xi ( \# ( x \cap r ) )$ for all $x \subseteq { \mathcal { C } }$ . We assume $\psi$ can be evaluated in polynomial time. Then, if

$$
\mathcal { X } _ { k } = \{ x \in \mathcal { X } \colon \psi ( x ) \in [ a _ { k } , b _ { k } ) \}
$$

for $k \in [ K ]$ is our $\mathrm { T C } ( \mathcal { X } , K )$ , the decision problem in Equation (6) can be resolved in polynomial time with respect to $\# \mathcal { C }$ . This remains true when we replace $2 ^ { \mathcal { C } }$ by the bounded multisubsets of .

As we will see in Sections 6 and G, symmetric junta-based TCs can be applied to a broad range of problems in the GFlowNet literature. Building on the solution to Equation (6), we can eficiently mask out undesirable transitions from each policy. In practice, this operation requires a negligible amount of arithmetic operations, adding an imperceptible extra cost to the sampling process. In Example 5.4, as an illustration, we solely verify whether $s + 1$ is within the boundaries of $\mathcal { X } _ { k }$

## 5.2 Complementary Characterization of SC GFlowNets

Connection to Doob’s h-transform.<sup>3</sup> A natural question raised by our exposition is whether there is a relationship between each $p _ { F } ^ { ( k ) }$ for $k \in [ K ]$ . We show in this section that, given a solution $p _ { F }$ to the original problem of sampling from $x \in \mathcal { X }$ in proportion to $R ( x )$ and any partition $\{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ , a solution $p _ { F } ^ { ( k ) }$ to the optimization problem in Equation (5) can be derived from a Doob’s h-transform of $p _ { F }$ when TC is a disjoint cover. To understand this, we review Doob’s h-transform of Markov chains.

Definition 5.8 (Doob’s h-transform). Let $\{ S _ { t } \} _ { t \ge 1 }$ be a Markov chain with state space ${ \bar { s } } : = s \cup \chi$ and transition kernel κ : $\bar { \mathcal { S } } \times \bar { \mathcal { S } }  [ 0 , 1 ]$ . Let also $h \colon \bar { S } $ R be a harmonic function, $\mathrm { i . e . , } h ( s ) = \mathbb { E } _ { s ^ { \prime } \sim \kappa ( s , \cdot ) } \left[ h ( s ^ { \prime } ) \right]$ We call $\kappa ^ { ( h ) } ( s , s ^ { \prime } ) = \kappa ( s , s ^ { \prime } ) \cdot \bar { h } ( s ^ { \prime } ) \bar { \big / } h ( s )$ the Doob’s h-transform of $\kappa .$

We refer the reader to (Rogers & Williams, 2000, Chapter III) for a comprehensive overview of Definition 5.8 and its applications to stochastic processes. This section shows that there is a harmonic function $h _ { k }$ for which $p _ { F } ^ { ( k ) }$ is the Doob’s $h _ { k } .$ -transformation of $p _ { F }$ for each $k \in [ K ]$ . To see this, we describe the generative process of a GFlowNet as a Markov chain having as the absorbing set.

Proposition 5.9 (GFlowNet as an absorbing Markov chain). Let $p _ { F }$ be a GFlowNet’s policy abiding by any balance condition and $\kappa _ { F }$ be a Markov kernel on <sup>¯</sup> defined as $\kappa _ { F } ( s , s ^ { \prime } ) = p _ { F } ( s , s ^ { \prime } ) \ i f \ s \in S$ and $\kappa _ { F } ( x , s ^ { \prime } ) =$ $\mathbb { 1 } [ s ^ { \prime } = x ] \ i f x \in \mathcal { X }$ for each $s ^ { \prime } \in \bar { S }$ . Let $\{ S _ { t } ^ { ( f ) } \} _ { t \ge 1 }$ be a Markov chain following $\kappa _ { F _ { \mathrm { ~ . ~ } } }$ , and let $T _ { \mathcal { X } } = \operatorname* { m i n } \{ t \colon S _ { t } ^ { ( f ) } \in$ be its hitting time on . When $p _ { F }$ satisfies the $T B ,$

$$
\mathbb { P } _ { \kappa _ { F } } \left[ \boldsymbol { T } \boldsymbol { \chi } < \infty | \boldsymbol { s } _ { o } \right] = 1 \ a n d \ \mathbb { P } _ { \kappa _ { F } } [ S _ { \boldsymbol { T } \boldsymbol { x } } ^ { ( f ) } = \boldsymbol { x } \in \mathcal { X } | \boldsymbol { s } _ { o } ] = \frac { R ( \boldsymbol { x } ) } { \sum _ { \boldsymbol { x } \in \mathcal { X } } R ( \boldsymbol { x } ) }
$$

with $\mathbb { P } _ { \kappa _ { F } }$ being the probability measure induced by the transition kernel $\kappa _ { F }$ . Equivalently, let $p _ { B }$ be a GFlowNet’s backward policy and $\kappa _ { B }$ a Markov kernel on $\bar { \boldsymbol { S } }$ such that $\kappa _ { B } ( s , s ^ { \prime } ) = p _ { B } ( s , s ^ { \prime } )$ for $s \in \bar { \mathcal { S } } \setminus \{ s _ { o } \}$ and $s ^ { \prime } \in \bar { S }$ and $\kappa _ { B } ( s _ { o } , s ) = \mathbb { 1 } [ s = s _ { o } ]$ . As such, let $\{ S _ { t } ^ { ( b ) } \} _ { t \ge 1 }$ follow κ<sub>B</sub>; if

$$
T _ { \{ s _ { o } \} } = \operatorname* { m i n } \{ t : S _ { t } ^ { ( b ) } = s _ { o } \} , \ t h e n \mathbb { P } _ { \kappa _ { B } } \left[ T _ { \{ s _ { o } \} } < \infty | x \right] = 1 \ f o r \ e v e r y \ x \in \mathcal { X } .
$$

Under Proposition 5.9, we show that there is a transition kernel $\kappa _ { F } ^ { ( k ) }$ for which $( \mathrm { i } ) \mathcal { X } _ { k }$ is the only absorbing set, (ii) each $x \in \mathcal { X } _ { k }$ is an absorbing state being arrived at with probability proportional to $R ( x )$ , (iii) and that ${ \bf \dot { \kappa } } _ { F } ^ { ( k ) }$ is a Doob’s transformation of $\kappa _ { F }$ . For this, let $\{ S _ { t } ^ { ( f ) } \} _ { t \ge 1 }$ be as above and

$$
T _ { k } = \operatorname* { m i n } \left\{ t \colon S _ { t } ^ { ( f ) } \in \mathcal { X } _ { k } \right\}
$$

be the hitting time on $\mathcal { X } _ { k }$ . We define $h _ { k } ( s ) = \mathbb { P } _ { \kappa _ { F } } [ T _ { k } = \operatorname* { m i n } _ { k ^ { \prime } } T _ { k ^ { \prime } } | s ]$ as the probability that the Markov chain $\{ S _ { t } ^ { ( f ) } \} _ { t \ge }$ <sub>1</sub> reaches $\mathcal { X } _ { k }$ before any other $\mathcal { X } _ { k ^ { \prime } }$ when starting at s. The next lemma shows $h _ { k }$ is a harmonic function in the sense of Definition 5.8.

Lemma 5.10. The functions $h _ { k } \colon \bar { \mathcal { S } }  \mathbb { R } , k \in [ K ] .$ , are harmonic in  with respect to $\kappa _ { F }$ . They also satisfy $h _ { k } ( x ) = \mathbb { 1 } [ x \in \mathcal { X } _ { k } ]$ for x   and, defining $Z _ { k }$ and Z as in Corollary 5.2, $h _ { k } ( s _ { o } ) = Z _ { k } / Z$

Lemma 5.10 ensures Doob’s $h _ { k }$ -transform of $\kappa _ { F }$ is well-defined. We show in the next proposition that the kernels resulting from such transform sample each $x \in \mathcal { X } _ { k }$ in proportion to $R ( x )$

Proposition 5.11 $( p _ { F } ^ { ( k ) }$ is a Doob’s h -transform of $p _ { F } )$ . Let κ be as in Proposition 5.9 and $h _ { k } \colon \bar { S } $ R be the harmonic function of Lemma 5.10 for $k \in [ K ]$ . We define $\underset { \mathcal { A } } { \kappa _ { F } ^ { ( k ) } } { ( s , s ^ { \prime } ) } = { \kappa _ { F } ( \bar { s } , s ^ { \prime } ) } { \cdot } h _ { k } ( s ^ { \prime } ) \big / { h _ { k } ( s ) }$ and $\{ S _ { t } ^ { ( k ) } \} _ { t \ge 1 }$ as a Markov chain following $\kappa _ { F } ^ { ( k ) }$ . Then, $i f T _ { m } ^ { ( k ) } =$ min $\{ t \colon S _ { t } ^ { ( k ) } \in \mathcal { X } _ { m } \}$

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ { T } _ { k } ^ { ( k ) } < \infty = T _ { m } ^ { ( k ) } \right] = 1 f o r m \neq k a n d \mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } ^ { ( k ) } } ^ { ( k ) } = x \right] = \frac { R ( x ) } { \sum _ { x \in \chi _ { k } } R ( x ) }
$$

for all $x \in \mathcal { X } _ { k }$ . In particular, the policy function $p _ { F } ^ { ( k ) }$ defined as $p _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) = \kappa _ { F } ^ { ( k ) } ( s , s ^ { \prime } )$ for $s \in S$ and $s ^ { \prime } \in \bar { S }$ satisfies Equation (2), i.e., the marginal of $p _ { F } ^ { ( k ) }$ on matches $R _ { k }$ up to a constant.

In other words, Proposition 5.11 states $p _ { F } ^ { ( k ) }$ solves SC GFlowNet’s learning problem.

## 5.3 Sampling Partitions & Parallel Learning for SC GFlowNets

Although Proposition 5.11 clearly connects each $p _ { F } ^ { ( k ) }$ to a base policy $p _ { F }$ , the quantity $h _ { k } ( x )$ cannot be eficiently computed in general even for known $\kappa _ { F } -$ let alone for the unknown solution to the GFlowNet’s balance equations. Besides, learning $h _ { k }$ from Lemma 5.10 can be impractical, as it would severely increase the per-transition computational cost. In practice, we parameterize our policies as in Example 5.4 and learn them by minimizing the objective function

$$
\begin{array} { r } { \mathbb { E } _ { k \sim \rho _ { M } , \tau \sim \rho _ { k } } \left[ \mathcal { L } _ { \mathrm { T B } } \left( \tau , \{ k \} ; \mathfrak { g } _ { [ K ] } \right) \right] . } \end{array}\tag{7}
$$

Active Partition Sampling. Equation (7) begs the question of how to sample k from $[ K ]$ during training. Intuitively, we should select most often the k for which the probability mass under $\mathcal { X } _ { k }$ is largest, while also serendipitously picking the k such that $\mathcal { X } _ { k }$ is thought to only include low-probability states. As $Z _ { k }$ serves as a surrogate for the probability mass under $\mathcal { X } _ { k }$ (Corollary 5.2), this intuition can be naturally operationalized by drawing k according to

$$
k \sim \rho _ { M } : = \mathrm { C a t e g o r i c a l } \left( ( 1 - \epsilon ) \cdot \mathbf { z } / \mathbf { { \mathbf { 1 } } ^ { \intercal } } \mathbf { z } + \epsilon \cdot K ^ { - 1 } \right)
$$

with $\epsilon \in [ 0 , 1 ]$ and 1 as a K-sized vector of ones. We emphasize more sophisticated approaches, such as curriculum-based sampling (Bengio et al., 2009; Laajil et al., 2025) focused on high-loss instead of high-mass components, could also be developed; however, we leave this for future research. We summarize both the training and inference procedures for SC GFlowNets in Algorithms 1 and 2.

Embarrassingly Parallel Learning. Besides being trainable as a monolithic model through active partition sampling, SC GFlowNets enable distributed learning of GFlowNets without inter-process communication. We call this embarrassingly parallel learning: we learn K independent models, each of which generates samples from $\mathcal { X } _ { k }$ according to $R _ { k } ( x )$ . Afterwards, during inference, we pick each model in proportion to its learned partition function to generate samples from . To separate SC GFlowNet from their asynchronous variants, we refer to the latter as ASC GFlowNets.

![](images/d2c7b934ad64d1308d1eae72fa4efdef93950a3e89e043fb9bbbcae55867d11e.jpg)

![](images/1eb2ed2c6064a1e52106c7e6203015f88a438e516338e5fda5e234ef7ff796b7.jpg)

![](images/bba7c0542d26e2772c651548e00fbacc1d16e94bbc13c01a478f8eb379cc8679.jpg)

![](images/3bf0e1735c343ef50dd349b5cf406e8a081a1fd2b8c49cca55ced3c165581e8c.jpg)  
Figure 4: Average wallclock times are similar for both samplers.

Figure 3: Length-based SC GFlowNets trains faster than conventional GFlowNets in the Set Generation domain, requiring less than half the number of iterations to achieve the same accuracy.  
![](images/4aa81ccb59eca52b6d7e3f67a18b5701873f571a2d90c72dd0ad64731c767108.jpg)

![](images/c1113f43dd13b3d93c3e079a9e8a67fb1bfa6d36084ee109d9452b5f16ef11e0.jpg)

![](images/303a772e7fa3417cbeb32017be62cf582502dad04a56cf16e98734d8951658d2.jpg)  
Figure 5: SC GFlowNets converge substantially faster than GFlowNets in the Hypergrid domain, with distinct partitioning strategies (Length & Octant) yielding similar benefits.

## 6 Experiments

Our experiments evaluate the efectiveness of SC GFlowNets. We consider the tasks of Set Generation, Hypergrid, and Ancestral Graphs, representing both classical and recent GFlowNet applications with tractable assessment metrics (Malkin et al., 2022, 2023; Kim et al., 2023, 2025b), and defer Lazy Random Walk and Variable Selection, along with an evaluation of ASC GFlowNets and a comparison against Boosted GFlowNets (Dall’Antonia et al., 2026b), to Section G.

Set generation. We let ${ \cal S } = \{ s \subseteq [ H ] \}$ for some $H \in \mathbb { N }$ and ${ \mathcal { X } } = { \mathcal { S } } \times \{ \top \}$ for the state space. We also define u: $[ H ] \to \mathbb { R } _ { + }$ and let $\begin{array} { r } { R ( x ) = \prod _ { e \in x } u ( e ) } \end{array}$ be the target distribution, which allows the probabil ity $\pi _ { \mathrm { i t e m } } ( e )$ of e being a member of x for $e \in [ H ]$ to be tractably computed. We evaluate the model via $\begin{array} { r } { \mathrm { E r r } _ { \mathrm { i t e m } } = \frac { 1 } { 2 } \sum _ { e \in [ H ] } \left| \hat { \pi } _ { \mathrm { i t e m } } ( e ) - \pi _ { \mathrm { i t e m } } ( e ) \right| } \end{array}$ , with $\hat { \pi } _ { \mathrm { i t e m } }$ as the Monte Carlo estimate of $\pi _ { \mathrm { i t e m } } .$ . For SC GFlowNets, we use a length-based partitioning with $K = 3$ . Results for $H \in \{ 2 4 , 4 8 , 6 4 \}$ in Figure 3 highlight the benefits of SC GFlowNet over its unpartitioned counterpart. In addition, we notice in Figure 4 the diference in runtime for both samplers is not significant. The main reason for this is that the cost of the extra steps required by the mixture model—active partition sampling and mask computation—is dwarfed by that of policy and reward evaluation. We observed similar runtime patterns across all our experiments.

Hypergrid. The state space for Hypergrid is $S = \{ 0 , \dots , H - 1 \} ^ { d }$ and ${ \mathcal { X } } = { \mathcal { S } } \times \{ \top \}$ . The initial state is $s _ { o } = \mathbf { 0 }$ and each transition from $s \in S$ either adds 1 to one of s’s coordinates, respecting the boundaries of $s$ or interrupts the generative process, producing $x = ( s , \top )$ . The target distribution, described in Section G, is shown in Figure 5. When designing SC GFlowNets, we consider both a length- $\left( K = 3 \right)$ and octant-based $( K = 4 )$ partitions, with the latter perfectly separating the four high-probability regions of $x ;$ see Figure 27 for illustrations. Results in Figure 5 confirm SC GFlowNets’s efectiveness even when using an imperfectly balanced partitioning.

Ancestral graphs. This task was studied in (da Silva et al., 2026) for addressing the problem of causal discovery under latent confounding. In a nutshell, contains all partially edge-labelled ancestral graphs (Richardson & Spirtes, 2002, AGs) with H nodes, while contains all fully edge-labelled AGs. The generative process starts at a graph with unlabelled edges, and each transition labels a node pair with either $\varnothing , \to ,  .$ or ; represents the absence of an edge. The objective is to generate high-scoring AGs x according to the

![](images/0f80f806a01ebc74add82e7258aa61e01b21bb2bd322957962e020190819120c.jpg)

![](images/43cf2327b7281ec28b6717fb4b9c71397e8762fb2d479b7a47919b6b9e00a1aa.jpg)  
(a) BIC (Equation (20)).

![](images/7ddaea7c30008f453205ecf502891f7ea6c84ec2a98850d998a32c73761daf24.jpg)

![](images/c2c24c732d4f8fac011cac999ff58740b7c3afabe7be1ddc787d5c99a7e5d1a0.jpg)  
(b) Erd˝os-Renyi model.  
Figure 6: SC GFlowNets enhance exploration (left) and speed up learning (right) for the Ancestral Graphs domain. We denote by H the number of variables (nodes).

Bayesian information criterion (BIC) based on a dataset modelled as a linear Gaussian structural equation model (Drton & Richardson, 2004); we set $R ( x ) = \exp \{ - \mathrm { B I C } ( x ) \}$ . As the distributional correctness of such method cannot be tractably measured for all but tiny H, we also consider an Erd˝os-Renyi model, for which edge marginals can be eficiently computed using Robinson’s formula (Robinson, 1973) and compared to that learned by our samplers (see Section G). In both cases, we partition according to the number of nonedges for SC GFlowNets. Figure 6 shows that our mixture model improves both exploration and convergence.

## 7 Conclusions

The central questions we addressed in this work are why and how to learn a mixture of GFlowNets. We first developed a general-purpose framework for answering them, outlining the capabilities and limitations of such approach. In particular, we showed SC GFlowNets, which learn an ensemble of samplers restricted to subsets of the state space, provide substantial empirical benefits at a negligible computational cost. Future research should investigate better ways of operationalizing our framework and, specifically, develop heuristics for automatic and tractable partitioning of the state space.

## References

Ralph Abboud, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The surprising power of graph neural networks with random node initialization. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), pp. 2112–2118, 2021.

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In NeurIPS (NeurIPS), 2021.

Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, 2009.

Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, and Emmanuel Bengio. Gflownet foundations. Journal of Machine Learning Research (JMLR), 2023.

Chris M Bishop. Training with noise is equivalent to tikhonov regularization. Neural computation, 7(1): 108–116, 1995.

Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2007.

David M. Blei and et al. Variational inference: A review for statisticians. Journal of the American Statistical Association, 2017.

David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan):993–1022, 2003.

Oussama Boussif, L´ena N´ehale Ezzine, Joseph D Viviano, Micha l Koziarski, Moksh Jain, Nikolay Malkin, Emmanuel Bengio, Rim Assouel, and Yoshua Bengio. Action abstractions for amortized sampling, 2024. URL https://arxiv.org/abs/2410.15184.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018.

Leo Brunswic, Yinchuan Li, Yushun Xu, Yijun Feng, Shangling Jui, and Lizhuang Ma. A theory of nonacyclic generative flow networks. Proceedings of the AAAI Conference on Artificial Intelligence, 38(10): 11124–11131, March 2024. ISSN 2159-5399. doi: 10.1609/aaai.v38i10.28989. URL http://dx.doi.org/ 10.1609/aaai.v38i10.28989.

Anthony Caterini, Rob Cornish, Dino Sejdinovic, and Arnaud Doucet. Variational inference with continuously-indexed normalizing flows, 2021. URL https://arxiv.org/abs/2007.05426.

Binqi Chen, Hongjun Ding, Ning Shen, Jinsheng Huang, Taian Guo, Luchen Liu, and Ming Zhang. Alphasage: Structure-aware alpha mining via gflownets for robust exploration, 2026. URL https: //arxiv.org/abs/2509.25055.

Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Cliford Stein. Introduction to Algorithms. MIT Press, 3rd edition, 2009. ISBN 9780262033848.

Rob Cornish, Anthony L. Caterini, George Deligiannidis, and Arnaud Doucet. Relaxing bijectivity constraints with continuously indexed normalising flows, 2021. URL https://arxiv.org/abs/1909.13833.

Tiago da Silva, Bruna Bazaluk, Eliezer de Souza da Silva, Ant´onio G´ois, Salem Lahlou, Dominik Heider, Samuel Kaski, Diego Mesquita, and Ad\`ele Helena Ribeiro. Expert-aided causal discovery of ancestral graphs. Information Sciences, 756:123816, 2026. ISSN 0020-0255. doi: https://doi.org/10.1016/j.ins.2026. 123816.

Pedro Dall’Antonia, Tiago da Silva, Daniel Csillag, Salem Lahlou, and Diego Mesquita. Avoid what you know: Divergent trajectory balance for gflownets, 2026a. URL https://arxiv.org/abs/2602.17827.

Pedro Dall’Antonia, Tiago da Silva, Daniel Augusto de Souza, C´esar Lincoln C. Mattos, and Diego Mesquita. Boosted gflownets: Improving exploration via sequential learning, 2026b.

Frederica Darema, David A. George, V. Alan Norton, and Gregory F. Pfister. A single-program-multipledata computational model for EPEX/FORTRAN. Parallel Computing, 7(1):11–24, 1988. doi: 10.1016/ 0167-8191(88)90095-4.

Tristan Deleu and Yoshua Bengio. Generative flow networks: a markov chain perspective, 2023. URL https://arxiv.org/abs/2307.01422.

Tristan Deleu, Ant´onio G´ois, Chris Chinenye Emezue, Mansi Rankawat, Simon Lacoste-Julien, Stefan Bauer, and Yoshua Bengio. Bayesian structure learning with generative flow networks. In UAI, 2022.

Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, and Yoshua Bengio. Joint Bayesian inference of graphical structure and parameters with a single generative flow network. In Advances in Neural Processing Systems (NeurIPS), 2023.

Tristan Deleu, Padideh Nouri, Nikolay Malkin, Doina Precup, and Yoshua Bengio. Discrete probabilistic inference as control in multi-path environments, 2024. URL https://arxiv.org/abs/2402.10309.

A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B, 39:1–38, 1977. URL http://web.mit.edu/ 6.435/www/Dempster77.pdf.

Mathias Drton and Thomas S. Richardson. Iterative conditional fitting for gaussian ancestral graph models. In Proceedings ofthe Twentieth Conference on Uncertainty in Artificial Intelligence (UAI-04), pp. 130–137. AUAI Press, 2004.

Andrew Gelman and Xiao-Li Meng. Simulating normalizing constants: From importance sampling to bridge sampling to path sampling. Statistical Science, 13(2):163–185, May 1998.

Edward I. George and Robert E. McCulloch. Variable selection via Gibbs sampling. Journal of the American Statistical Association, 88(423):881–889, 1993. doi: 10.2307/2290777.

Timofei Gritsaev, Nikita Morozov, Sergey Samsonov, and Daniil Tiapkin. Optimizing backward policies in gflownets via trajectory likelihood maximization, 2025. URL https://arxiv.org/abs/2410.15474.

Abhi Gupta, Polina Barabanshchikova, Vikas K Garg, Samuel Kaski, and Tommi Jaakkola. Divide-anddenoise: A game-theoretic method for fairly composing difusion models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=9voQUicsc2.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction, volume 2. Springer, 2nd edition, 2009

Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network library and ecosystem for JAX, 2026. URL http://github. com/google/flax.

Matthew D. Hofman, David M. Blei, and Francis Bach. Online learning for latent dirichlet allocation. In J. Laferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta (eds.), Advances in Neural Information Processing Systems, volume 23, pp. 856–864. Curran Associates, Inc., 2010. URL https://papers.nips. cc/paper/3902-online-learning-for-latent-dirichlet-allocation.

Edward J. Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, and et al. Amortizing intractable inference in large language models, 2023.

Rui Hu, Yifan Zhang, Zhuoran Li, and Longbo Huang. Beyond squared error: Exploring loss design for enhanced training of generative flow networks, 2026. URL https://arxiv.org/abs/2410.02596.

Moksh Jain, Emmanuel Bengio, Alex Hernandez-Garcia, Jarrid Rector-Brooks, Bonaventure F. P. Dossou, Chanakya Ajit Ekbote, Jie Fu, Tianyu Zhang, Michael Kilgour, Dinghuai Zhang, Lena Simine, Payel Das, and Yoshua Bengio. Biological sequence design with GFlowNets. In International Conference on Machine Learning (ICML), 2022.

E. T. Jaynes. Probability theory: The logic of science. Cambridge University Press, Cambridge, 2003.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github. io/posts/muon/.

Mohammad Emtiyaz Khan and Didrik Nielsen. Fast yet simple natural-gradient descent for variational inference in complex models, 2018. URL https://arxiv.org/abs/1807.04489.

Hohyun Kim, Seunggeun Lee, and Min hwan Oh. Symmetry-aware gflownets, 2025a. URL https://arxiv. org/abs/2506.02685.

Minsu Kim, Taeyoung Yun, Emmanuel Bengio, Dinghuai Zhang, Yoshua Bengio, Sungsoo Ahn, and Jinkyoo Park. Local search gflownets. arXiv preprint arXiv:2310.02710, 2023.

Minsu Kim, Joohwan Ko, Taeyoung Yun, Dinghuai Zhang, Ling Pan, Woochang Kim, Jinkyoo Park, Emmanuel Bengio, and Yoshua Bengio. Learning to scale logits for temperature-conditional gflownets, 2024. URL https://arxiv.org/abs/2310.02823.

Minsu Kim, Sanghyeok Choi, Taeyoung Yun, Emmanuel Bengio, Leo Feng, Jarrid Rector-Brooks, Sungsoo Ahn, Jinkyoo Park, Nikolay Malkin, and Yoshua Bengio. Adaptive teachers for amortized samplers. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview. net/forum?id=BdmVgLMvaf.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

Aya Laajil, Abduragim Shtanchaev, Sajan Muhammad, Eric Moulines, and Salem Lahlou. Curriculumaugmented gflownets for mrna sequence generation, 2025. URL https://arxiv.org/abs/2510.03811.

Salem Lahlou, Tristan Deleu, Pablo Lemos, Dinghuai Zhang, Alexandra Volokhova, Alex Hern´andez-Garc´ıa, L´ena N´ehale Ezzine, Yoshua Bengio, and Nikolay Malkin. A theory of continuous generative flow networks. In ICML, volume 202 of Proceedings of Machine Learning Research, pp. 18269–18300. PMLR, 2023.

Sida Li, Ioana Marinescu, and Sebastian Musslick. Gfn-sr: Symbolic regression with generative flow networks, 2023. URL https://arxiv.org/abs/2312.00396.

Jun S Liu and Jun S Liu. Monte Carlo strategies in scientific computing, volume 10. Springer, 2001.

Sulin Liu, Peter J. Ramadge, and Ryan P. Adams. Generative marginalization models, 2024. URL https: //arxiv.org/abs/2310.12920.

Francesco Locatello, Rajiv Khanna, Joydeep Ghosh, and Gunnar Ratsch. Boosting variational inference: an optimization perspective. In Amos Storkey and Fernando Perez-Cruz (eds.), Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pp. 464–472. PMLR, 09–11 Apr 2018.

Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio, and Nikolay Malkin. Learning gflownets from partial episodes for improved convergence and stability. In International Conference on Machine Learning, 2022.

Kanika Madan, Alex Lamb, Emmanuel Bengio, Glen Berseth, and Yoshua Bengio. Towards improving exploration through sibling augmented GFlownets. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=HH4KWP8RP5.

Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in GFlownets. In NeurIPS (NeurIPS), 2022.

Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, and Yoshua Bengio. GFlowNets and variational inference. International Conference on Learning Representations (ICLR), 2023.

Nikita Morozov, Ian Maksimov, Daniil Tiapkin, and Sergey Samsonov. Revisiting non-acyclic gflownets in discrete environments, 2025. URL https://arxiv.org/abs/2502.07735.

Radford M Neal et al. Mcmc using hamiltonian dynamics. Handbook of markov chain monte carlo, 2011.

W. Neiswanger, C. Wang, and E. P. Xing. Asymptotically exact, embarrassingly parallel MCMC. In UAI, 2014.

Art B. Owen. Monte Carlo theory, methods and examples. 2013.

Ling Pan, Nikolay Malkin, Dinghuai Zhang, and Yoshua Bengio. Better training of gflownets with local credit and incomplete trajectories. arXiv preprint arXiv:2302.01687, 2023a.

Ling Pan, Dinghuai Zhang, Aaron Courville, Longbo Huang, and Yoshua Bengio. Generative augmented flow networks. In International Conference on Learning Representations (ICLR), 2023b.

Mohit Pandey, Gopeshh Subbaraj, and Emmanuel Bengio. Gflownet pretraining with inexpensive rewards. arXiv preprint arXiv:2409.09702, 2024.

Fabio Pardo, Arash Tavakoli, Vitaly Levdik, and Petar Kormushev. Time limits in reinforcement learning, 2022. URL https://arxiv.org/abs/1712.00378.

Thomas Richardson and Peter Spirtes. Ancestral graph markov models. The Annals of Statistics, 30(4): 962–1030, 2002.

Robert W. Robinson. Counting labeled acyclic digraphs. In Frank Harary (ed.), New Directions in the Theory of Graphs, pp. 239–273. Academic Press, New York, 1973. ISBN 978-0-12-324255-6.

L. C. G. Rogers and David Williams. Difusions, Markov Processes, and Martingales: Volume 1, Foundations. Cambridge Mathematical Library. Cambridge University Press, 2nd edition, 2000. ISBN 9780521775946.

Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM), 2021.

J¨urgen Schmidhuber. Generative adversarial networks are special cases of artificial curiosity (1990) and also closely related to predictability minimization (1991). Neural Networks, 127:58–66, 2020.

Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, and Tommaso Biancalani. Towards understanding and improving gflownet training. In International Conference on Machine Learning, 2023.

Tiago Silva, Rodrigo Barreto Alves, Eliezer de Souza da Silva, Amauri H Souza, Vikas Garg, Samuel Kaski, and Diego Mesquita. When do GFlownets learn the right distribution? In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum?id=9GsgCUJtic.

Tiago Silva, Amauri H Souza, Omar Rivasplata, Vikas Garg, Samuel Kaski, and Diego Mesquita. Generalization and distributed learning of GFlownets. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=PJNhZoCjLh.

Tiago Silva, Esmeralda S. Whitammer, and Salem Lahlou. Path-dependent discrete amortized inference. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=viJvYPJHIj.

Daniil Tiapkin, Nikita Morozov, Alexey Naumov, and Dmitry Vetrov. Generative flow networks as entropyregularized rl, 2024.

Daniil Tiapkin, Artem Agarkov, Nikita Morozov, Ian Maksimov, Askar Tsyganov, Timofei Gritsaev, and Sergey Samsonov. gfnx: Fast and scalable library for generative flow networks in jax, 2025. URL https: //arxiv.org/abs/2511.16592.

Siddarth Venkatraman, Moksh Jain, Luca Scimeca, Minsu Kim, Marcin Sendera, Mohsin Hasan, Luke Rowe, Sarthak Mittal, Pablo Lemos, and Emmanuel Bengio others. Amortizing intractable inference in difusion models for vision, language, and control, 2024. URL https://arxiv.org/abs/2405.20971.

Joseph D. Viviano, Omar G. Younis, Sanghyeok Choi, Victor Schmidt, Yoshua Bengio, and Salem Lahlou. torchgfn: A pytorch gflownet library, 2026. URL https://arxiv.org/abs/2305.14594.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? International Conference on Learning Representations (ICLR), 2019.

Mingzhang Yin and Mingyuan Zhou. Semi-implicit variational inference. In International conference on machine learning. PMLR, 2018.

Xuan Yu, Xu Wang, Rui Zhu, Yudong Zhang, and Yang Wang. Exploring multiple high-scoring subspaces in generative flow networks, 2026a. URL https://arxiv.org/abs/2602.11491.

Xuan Yu, Xu Wang, Rui Zhu, Yudong Zhang, and Yang Wang. Partial gflownet: Accelerating convergence in large state spaces via strategic partitioning, 2026b. URL https://arxiv.org/abs/2602.11498.

David W Zhang, Corrado Rainone, Markus Peschl, and Roberto Bondesan. Robust scheduling with gflownets. In International Conference on Learning Representations (ICLR), 2023.

Ming Yang Zhou, Zichao Yan, Elliot Layne, Nikolay Malkin, Dinghuai Zhang, Moksh Jain, Mathieu Blanchette, and Yoshua Bengio. PhyloGFN: Phylogenetic inference with generative flow networks. In The Twelfth International Conference on Learning Representations, 2024.

Heiko Zimmermann, Fredrik Lindsten, Jan-Willem van de Meent, and Christian A Naesseth. A variational perspective on generative flow networks. Transactions on Machine Learning Research, 2023. ISSN 2835- 8856. URL https://openreview.net/forum?id=AZ4GobeSLq.

## A GFlowNets: Background & Challenges

We provide a detailed background on GFlowNets, and review current challenges in efectively training these samplers, explaining how they can be naturally addressed by our mixture framework.

## A.1 Background

This section reviews prior work on learning GFlowNets for discrete distributions, following Malkin et al.   
(2023); Lahlou et al. (2023), to which we refer for a comprehensive overview.

GFlowNets. Let be a finite set; we say $\mathcal { X }$ is a compositional space if there is a state graph $\left( \mathrm { S G } \right) \mathcal { G } = \left( S , \mathcal { E } \right)$ with vertices $s \supset \mathcal { X }$ and edges having the following properties.

1. $\mathcal { G }$ is a directed acyclic graph (DAG) with a single source, $s _ { o } .$ , and with $\mathcal { X }$ as the set of sink vertices. We call $s _ { o }$ the initial state.

2. There is a (possibly non-unique) path from from each $s _ { o }$ to each $\cal { S } \backslash \{ s _ { o } \}$

We further assume, for computational reasons, that the number of children of each $s \in S$ on $\mathcal { G }$ and the maximum trajectory length in $\mathcal { G }$ are exponentially smaller than $| \mathcal { X } |$ . (Otherwise, every finite set $\mathcal { X }$ would be compositional by considering $S = \{ s _ { o } \} \cup \mathcal { X }$ and ${ \mathcal { E } } = \{ s _ { o }  x \colon x \in \mathcal { X } \} \quad$ . Our objective is to generate $x \in \mathcal { X }$ in proportion to a positive function R: $\mathcal { X } \to \mathbb { R } _ { + }$

Sampling as learning. Towards this objective, we define $p _ { F } \colon S \times ( S \cup \mathcal { X } )  [ 0 , 1 ]$ (resp. $p _ { B } \colon ( S \cup \mathcal { X } ) \times S $ $[ 0 , 1 ] )$ as the forward (resp. backward) policy on $\mathcal { G }$ representing the probability $p _ { F } ( s , s ^ { \prime } )$ of transitioning from s to a child $s ^ { \prime }$ of s. The reason for $p _ { B } \mathrm { { ' s } }$ existence will become clearer shortly. We also define $s  \ s ^ { \prime }$ as the set of trajectories in $\mathcal { G }$ from s to $s ^ { \prime }$ . Our objective is for the marginal probability of $p _ { F }$ over $x ,$ , when starting at $s _ { o } ,$ to match $R ,$ i.e.,

$$
m _ { F } ( x ) : = \sum _ { \substack { \tau \in s _ { o } \ldots \ldots } } \prod _ { \substack { s , s ^ { \prime } ) \in \tau } } p _ { F } ( s , s ^ { \prime } ) : = \sum _ { \tau \in s _ { o } \ldots } p _ { F } ( s _ { o } , \tau ) \propto R ( x ) .\tag{8}
$$

The trajectory length K in the equation above need not be the same for each τ. Importantly, we emphasize this equation merely integrates out from $p _ { F }$ intermediate variables from the joint distribution $( s _ { o } , \ldots , x )$ over trajectories. Due to Equation (8)’s intractability, we estimate the summation above using $p _ { B }$ as a proposal for an importance-sampling-like scheme, that is, denoting $\begin{array} { r } { Z = \sum _ { x \in \mathcal { X } } R ( x ) } \end{array}$ ,

$$
\mu _ { F } ( s _ { o } , x ) = \mathbb { E } _ { \tau \sim p _ { B } ( x , \cdot ) } \left[ \frac { p _ { F } ( s _ { o } , \tau ) } { p _ { B } ( x , \tau ) } \right] = \frac { R ( x ) } { Z } ,
$$

with $\begin{array} { r } { p _ { B } ( x , \tau ) = \prod _ { ( s , s ^ { \prime } ) \in \tau } p _ { B } ( s ^ { \prime } , s ) } \end{array}$ . The condition $p _ { F } ( \tau | s _ { o } ) \big / p _ { B } ( \tau | x ) = R ( x ) \big / Z$ is known as trajectory balance (TB) Malkin et al. (2022). To search for policies satisfying it, we parameterize $p _ { F }$ and (optionally) $p _ { B }$ as neural networks and solve the stochastic program

$$
\operatorname* { m i n } _ { p _ { F } , p _ { B } } \mathcal { L } ( p _ { F } , p _ { B } | \rho ) : = \mathbb { E } _ { \tau \sim \rho } \left( \log \frac { Z \cdot p _ { F } ( \tau | s _ { o } ) } { p _ { B } ( \tau | x ) \cdot R ( x ) } \right) ^ { 2 }\tag{9}
$$

via stochastic gradient descent. As the reader may have noticed, $p _ { B }$ was introduced arbitrarily; in general, there are infinitely many pairs $\left( p _ { F } , p _ { B } \right)$ satisfying the TB condition, a fact that will come up later in Section A.2 when we discuss mixtures of amortized samplers. Additional learning objectives, with distinct parameterizations and properties, have been proposed and extensively studied (Madan et al., 2022; Malkin et al., 2023; Tiapkin et al., 2024; Pan et al., 2023a; Hu et al., 2026). In practice, we let θ be the (joint) parameters of $p _ { F } , p _ { B }$ , and log $Z , \{ \alpha _ { t } \} _ { t \ge 0 }$ be a sequence of step sizes, and iteratively minimize

$$
\small \theta ^ { ( t ) } = \operatorname* { m i n } _ { \theta } \theta ^ { \intercal } \nabla _ { \theta ^ { ( t - 1 ) } } \hat { \mathcal { L } } ( \theta | \theta ^ { ( t - 1 ) } ) + \frac { 1 } { 2 \alpha _ { t } } \| \theta - \theta ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 }
$$

(or a preconditioned variant of it (Khan & Nielsen, 2018)), with $\hat { \mathcal { L } } ( \theta | \theta ^ { ( t - 1 ) } )$ as an Monte Carlo estimate of $\mathcal { L } ( p _ { F } , p _ { B } | \rho ^ { ( t - 1 ) } )$ when both $p _ { F }$ and $p _ { B }$ are instantiated according to θ and $\rho ^ { ( t - 1 ) }$ as a function of $\theta ^ { ( t - 1 ) }$ (Tiapkin et al., 2024; Kim et al., 2025b; Madan et al., 2025; Gritsaev et al., 2025; Dall’Antonia et al., 2026a). For example, the popular ϵ-greedy exploration (Bengio et al., 2021; Malkin et al., 2022; Pan et al., 2023b; Zhou et al., 2024; Kim et al., 2025a; Dall’Antonia et al., 2026b) uses $\rho ^ { ( t - 1 ) } ( s , \cdot ) = ( 1 - \epsilon ) \cdot p _ { F } ^ { ( t - 1 ) } ( s , \cdot ) , + \epsilon \cdot p _ { U } ( s , \cdot )$ with $p _ { U } ( s , \cdot )$ as the policy function which selects the children of s uniformly at random and $p _ { F } ^ { ( t - 1 ) } ( s , \cdot )$ as the $p _ { F }$ corresponding to $\check { \theta } ^ { ( t - 1 ) }$

## A.2 Challenges in training GFlowNets

To further motivate our work, we briefly describe current challenges in learning GFlowNets, and later argue they can be partly addressed through a mixture formulation of the sampling problem.

![](images/1c8b59168204d293a009824884dc4b2a1e7c63a626e8c4b76775ffd21d3b8f32.jpg)  
Figure 7: Any $\alpha \in [ 1 , 2 ]$ solves the assignment problem in this state graph. The solution for uniform $p _ { B }$ is $\alpha = 1 . 5$

The first challenge is that the optimization problem described in Equation (9) is underdetermined, as there is no pair $\left( p _ { F } , p _ { B } \right)$ uniquely satisfying the TB condition in general (unless there is a single trajectory from $s _ { o }$ to each x, as in autoregressive sequence generation tasks Jain et al. (2022)). This is illustrated in Figure 7, which displays a family of $p _ { F }$ for which the marginal constraint in Equation (8) is satisfied, highlighting the existence of a continuum of solutions to the sampling problem.

Challenge A.1 (Underdetermination). There are infinitely many pairs $\left( p _ { F } , p _ { B } \right)$ for which TB is satisfied and the marginal of $p _ { F }$ on $\mathcal { X }$ matches R, and it is unclear how to choose between them.

A common choice, inspired by the maximum entropy principle (Jaynes, 2003), is to set $p _ { B } ( s , \cdot )$ as a uniform probability distribution over the parents of s in the SG (Shen et al., 2023). Alternatively, both $p _ { F }$ and p<sub>B</sub> can be jointly learned (Gritsaev et al., 2025). However, deriving a $p _ { B }$ for

which finding the corresponding $p _ { F }$ is easiest is not possible in general, and it is unclear which pair $\left( p _ { F } , p _ { B } \right)$ does gradient-based learning induce. We illustrate this in Figure 8 for the Hypergrid domain, showing the variability of $\mathrm { K L } ( p _ { B } ( s , \cdot ) | | p _ { U } ( s , \cdot ) )$ averaged over states $s \in \mathcal { S } \setminus \{ s _ { o } \}$ during training for 5 independent runs (the highlighted curve represents per-step averages). As we can observe, the learned $p _ { B }$ vary drastically solely by modifying the pseudorandom seed for sample generation—while maintaining both the parameter initialization and optimizer unchanged.

A mixture model, as we will show, naturally addresses Challenge A.1 by learning a family of backward policies—hence avoiding strict commitment to a single, possibly unknown $p _ { B }$ However, even when $p _ { B }$ is uniquely defined—e.g., in autoregressive sequence generation (Jain et al., 2022)—the underlying neural network might be incapable of finding an approximation to solution of the stochastic program in Equation (9). This problem, known as near state aliasing (Pardo et al., 2022), is particularly harmful for graph-structured domains (Silva et al., 2025a; Kim et al., 2025a).

![](images/314e72e9461145bec994007f8391c51490594b1e5a1c32642e9dc16df0c47409.jpg)  
Figure 8: KL between $p _ { B }$ and $p _ { U }$ for the to HYPERGRID domain Hypergrid domain.

Challenge A.2 ((Near) state aliasing). A solution $\left( p _ { F } , p _ { B } \right)$ Equation (9) might not be (accurately) realizable by the chosen family of parametric models.

In Section $\mathrm { C } ,$ we show how our method alleviates it by establishing a relationship between Random Network Initialization (RNI) in Graph Neural Networks (GNNs) and mixture modelling. Even an universal approximator, however, may struggle to search for a compatible pair $\left( p _ { F } , p _ { B } \right)$ simply due to the sheer size of the state space. The main reason for this is that, to accurately approximate the target distribution, the model has to reach its high-probability regions during training Kim et al. (2023, 2025b); Dall’Antonia et al. (2026a), an obstacle that typical ϵ-greedy strategies fail to overcome.

Challenge A.3 (Insuficient exploration). Accurately learning $\left( p _ { F } , p _ { B } \right)$ requires visiting the target R’s high-probability regions during training, which are often diverse and sparsely distributed.

To address Challenge A.3, recent works have proposed using boosting-style techniques (Dall’Antonia et al., 2026b) and artificial curiosity-inspired Schmidhuber (2020) exploratory models—which are discarded after training. These approaches, however, increase the per-step computational cost of conventional samplers by severalfold, requiring the learning and evaluation of one Kim et al. (2025b); Madan et al. (2025); Dall’Antonia et al. (2026a) or many additional models Silva et al. (2025b); Dall’Antonia et al. (2026b) throughout learning and inference. In contrast, in Section 6, we showed improved exploration is achievable by a mixture of GFlowNets—without relying on additional, independently parameterized costly models.

## B Mixture of GFlowNets

We outline additional results on the learning of a mixture of GFlowNets: an extension of Madan et al.   
(2022)’s SubTB condition and the associated family of loss functions. We start with the former.

Proposition B.1 (SubTB). For each $s \in { \mathcal { S } } \cup { \mathcal { X } }$ , let $F ( s , \mathrm { d } \gamma )$ be a measure on Γ such that $F ( s , \mathrm { d } \gamma ) = R ( x , \mathrm { d } \gamma )$ for $x \in \mathcal { X }$ . We call $\textit { F a }$ flow function. If, for $\tau \in s  s ^ { \prime }$ ，

$$
F ( s , \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s , \tau ) = F ( s ^ { \prime } , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( s ^ { \prime } , \tau ) ,
$$

then Equation (2) is satisfied. In addition, for each $s \in S$

$$
Z ( \mathrm { d } \gamma ) = F ( s _ { o } , \mathrm { d } \gamma ) \ a n d \ F ( s , \mathrm { d } \gamma ) = \sum _ { x \in \mathcal { X } } \sum _ { \tau \in s \to x } R ( x , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( x , \tau ) .
$$

We provide a demonstration for this and every other statement in the supplement in Section $\mathrm { H , }$ jointly with the statements in the main text. From Proposition B.1, the usual loss functions encountered in the GFlowNet literature can be readily derived.

Definition B.2 (SubTB & DB $\&$ TB losses). Let $Z ( \cdot )$ and $F ( s , \cdot )$ and $R ( x , \cdot )$ be measures on $\Gamma ,$ as in Proposition B.1, for $s \in \mathcal { S }$ and $x \in \mathcal { X }$ , and $p _ { F } ^ { ( \gamma ) }$ and $p _ { B } ^ { ( \gamma ) }$ be Γ-indexed policies. We concisely let $\mathfrak { g } _ { \Gamma } = \bar { ( Z , F , R , \{ p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) } \} } _ { \gamma \in \Gamma } )$ . Given $\tau = ( s _ { o } , \dots , s _ { N } )$ with $s _ { N } = x \in \mathcal { X }$ , we define $\tau _ { i : j } = ( s _ { i } , s _ { i + 1 } , \ldots , s _ { j } )$ and

$$
{ \mathcal { L } } _ { \mathrm { S u b T B } } ( \tau , \mathrm { d } \gamma ; \mathfrak { g } _ { \Gamma } ) = \sum _ { 0 \leq i < j \leq N } \alpha _ { i j } \left( \log { \frac { F ( s _ { i } , \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { i } , \tau _ { i : j } ) } { F ( s _ { j } , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( s _ { j } , \tau _ { j : i } ) } } \right) ^ { 2 }
$$

as the SubTB loss, with $( \alpha _ { i j } ) _ { i , j = 0 } ^ { N }$ as positive numbers with $\textstyle \sum _ { j > i } \alpha _ { i j } = 1$ . Similarly,

$$
\mathcal { L } _ { \mathrm { D B } } ( \tau , \mathrm { d } \gamma ; \mathfrak { g } _ { \Gamma } ) = \frac { 1 } { N } \sum _ { 0 \le i \le N - 1 } \left( \log \frac { F ( s _ { i } , \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { i } , s _ { i + 1 } ) } { F ( s _ { i + 1 } , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( s _ { i + 1 } , s _ { i } ) } \right) ^ { 2 }
$$

is the DB loss. In conclusion, we let the TB loss be

$$
\mathcal { L } _ { \mathrm { T B } } ( \tau , \mathrm { d } \gamma ; \mathfrak { g } _ { \Gamma } ) = \left( \frac { Z ( \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) } { R ( x , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( x , \tau ) } \right) ^ { 2 } .
$$

## C Continuously Indexed GFlowNets

In this section, we elaborate on the design of a continuously indexed mixture of GFlowNets. As we will show, this approach can drastically boost the expressivity of GNN-parameterized samplers (Kim et al., 2025a) without requiring expensive heuristics such as look-ahead search, as proposed in (Silva et al., 2025a).

In doing so, we designedly use a collapsed mixing posterior distribution— $- q ( \mathrm { d } \gamma | x ) = q ( \mathrm { d } \gamma )$ for every $x \in { \mathcal { X } } -$ an approach we refer to as random features (RFs) for GFlowNets. The reason for this is that either $Z ( \mathrm { d } \gamma )$ or $q ( \mathrm { d } \gamma | x )$ are generally intractable measures when Γ is uncountable, as we also demonstrate.

Continuous mixtures of GFlowNets. We henceforth consider $\Gamma = \mathbb { R } ^ { d }$ and $\nu$ as the Lebesgue measure, and we assume both $q _ { M } ( \mathrm { d } \gamma | x )$ (resp. $R ( x , \mathrm { d } \gamma ) , F ( s , \mathrm { d } \gamma ) )$ and $p _ { M } ( { \mathrm { d } } \gamma )$ (resp. $Z ( \mathrm { d } \gamma ) )$ are absolutely continuous with respect to it. As such, we write $R ( x , \gamma )$ and $F ( s , \gamma )$ and $Z ( \gamma )$ for their corresponding densities. It is an obvious consequence of Proposition B.1 that the balance conditions can be written as a function of these densities instead of the previously introduced measures. We highlight this in the following corollary.

Remark C.1 (SubTB). When $\Gamma = \mathbb { R } ^ { d }$ and $F ( s , \mathrm { d } \gamma )$ and $Z ( \mathrm { d } \gamma )$ are absolutely continuous with respect to the base Lebesgue measure $\nu ,$ the SubTB condition is

$$
F ( s , \gamma ) p _ { F } ^ { ( \gamma ) } ( s , \tau ) = F ( s ^ { \prime } , \gamma ) p _ { B } ^ { ( \gamma ) } ( s ^ { \prime } , \tau )
$$

for each $s , s ^ { \prime } \in \mathcal { S } \cup \mathcal { X }$ and $\tau \in s  s ^ { \prime }$ and ν-almost surely for $\gamma \in \Gamma$

As per Remark C.1, we learn $F ( s , \gamma )$ as a real-valued neural network-parameterized positive function receiving both $s \in S$ and $\gamma \in \Gamma$ as inputs; recall $F ( x , \gamma ) = R ( x , \gamma )$ for $x \in \mathcal { X }$ . The core issue in doing $\mathrm { s o . }$ however, is that we may not be able to sample from the prior mixing distribution during inference when the function $Z ( \gamma )$ is defined arbitrarily. Conversely, if we let $p _ { M } ( \gamma )$ have a simple form $( \mathrm { e . g . }$ , Gaussian) and $q _ { M } ( \gamma | x )$ be flexibly learned, the latter distribution may collapse into the former. This can be understood via the following corollary of Proposition B.1.

Corollary C.2 (Relationship between $p _ { M }$ and $q _ { M } )$ . Assume $p _ { M } , q _ { M } , p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) }$ satisfy the SubTB condition in Remark C.1. Then,

$$
p _ { M } ( \gamma ) = \sum _ { x \in \mathcal { X } } \frac { R ( x ) } { Z } \cdot q _ { M } ( \gamma | x ) \ a n d \ q _ { M } ( \gamma | x ) = \frac { Z \cdot p _ { M } ( \gamma ) \cdot m _ { F } ^ { ( \gamma ) } ( x ) } { R ( x ) } .
$$

The first equation in Corollary C.2 shows that, if $q _ { M } ( \gamma | x )$ has a tractable, non-trivial form, then the solution for $p _ { M } ( \gamma )$ is a combinatorial mixture of all $q _ { \gamma } ( \gamma | x ) \mathrm { { : } }$ for $x \in \mathcal { X }$ weighted by $R ( x )$ , which is the distribution we are aiming to sample from in the first place. The second equation, on the other hand, suggests that if $p _ { F } ^ { ( \gamma ) } ( s , \cdot )$ is too flexible a function of $\gamma$ and s—able to perfectly mimic the target $R ( x ) -$ —then the optimal form for $q _ { M } ( \gamma | x )$ reduces to $p _ { M } ( \gamma )$ . As we also discussed in Section $5 ,$ reduction is not necessarily undesirable, as we will show later; however, we believe a x-independent $q _ { M }$ should be an algorithmic design decision rather than an implicit consequence of training. Should the goal be to have meaningfully diferent $p _ { F } ^ { ( \gamma ) }$ for each $\gamma$ with the objective of, for instance, improving exploration, Corollary C.2 suggests that sensible structural constraints should be imposed on the relationship between $p _ { F } ^ { ( \gamma ) }$ and $\gamma$ , a venue we pursued in Section 5.

Random Features (RFs) for GFlowNets. In this scenario, we describe the potential benefits of setting $q _ { M } ( \gamma | x ) = p _ { M } ( \gamma )$ by connecting the ensuing framework with the theory of RFs for GNNs Abboud et al. (2021); Sato et al. (2021). We first recall the expressivity limitations of message passing GNN-parameterized GFlowNets in the following example.

Example C.3 (Limitations of GNN-based GFlowNets). We consider the following state graph (Figure 9) with initial state $s _ { o }$ being followed by $x _ { 1 }$ and $x _ { 2 }$ , each of them a 6-node graph. We notice that $x _ { 1 }$ and $x _ { 2 }$ correspond to $x _ { o }$ with additional edges $( a , b )$ and $( a , c )$ , respectively, and we write $x _ { 1 } = s _ { o } \cup \{ ( a , b ) \}$ and $x _ { 2 } = s _ { o } \cup \{ ( a , c ) \}$ . This is the simplest setting a GFlowNet can be applied to.

As in (Silva et al., 2025a), we let $h \colon ( \mathcal { G } , n ) \mapsto  { \mathbb { R } } ^ { d }$ be a 1-WL GNN Kipf & Welling (2016); Xu et al. (2019) yielding a representation for a node n in a graph $\mathcal { G }$ and $\sigma \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be a learnable real-valued function. We parameterize the policy function according to

$$
p _ { F } ( s , s \cup \{ ( n _ { 1 } , n _ { 2 } ) \} ) \propto \exp \left\{ \sigma ( h ( s , n _ { 1 } ) + h ( s , n _ { 2 } ) ) \right\}\tag{10}
$$

![](images/515887a06ca902b3ae084a6b845a0ccd0170d3556d3d20e253c8b64cd9704875.jpg)  
Figure 9: State graph for Example C.3.

for nodes $n _ { 1 } , n _ { 2 }$ such that there is no edge between $n _ { 1 } , n _ { 2 }$ in s, and 0 otherwise. Under these conditions, un less the target $R$ satisfies $R ( x _ { 1 } ) = R ( x _ { 2 } )$ , there is no instantiation of $p _ { F }$ for which the marginal distribution of $p _ { F }$ over $\{ x _ { 1 } , x _ { 2 } \}$ matches R—as h cannot distinguish node b from node $c .$

When augmenting h with random node-wise features $\gamma _ { ; }$ , however, the policy function in Equation (10) acquires the capability of approximating any target with high-probability over the draw of $\gamma .$ The intuition is that a RF-augmented 1-WL GNN will only be unable to separate nodes b and c in Figure 9 if their corresponding features match—an event of null probability since Γ is continuous.

Example C.4 (RFs boost the expressivity of GNN-based GFlowNets). We let $h \colon ( \mathcal { G } , n , \gamma _ { n } ) \mapsto  { \mathbb { R } } ^ { d }$ be a RF-augmented 1-WL GNN with $\gamma _ { n } \sim \mathcal { N } ( 0 , 1 )$ . Then, if

$$
p _ { F } ^ { ( \gamma ) } ( s , s \cup \{ ( n _ { 1 } , n _ { 2 } ) \} ) \propto \exp \left\{ \sigma ( h ( s , n _ { 1 } , \gamma _ { n _ { 1 } } ) + h ( s , n _ { 2 } , \gamma _ { n _ { 2 } } ) ) \right\}\tag{11}
$$

as in Equation (10), it follows from (Abboud et al., 2021, Theorem 1) that for each $\epsilon > 0$ and $\delta \in ( 0 , 1 )$ there is a GNN h such that $p _ { F }$ approximates a given target with accuracy ϵ and probability $1 - \delta$ over the drawing of $\gamma$

More broadly, we show in the next that a GFlowNet parameterized as in Equation (11) can arbitrarily approximate with high probability any given distribution defined on an edge-additive generative process for graph-structured objects. For this, we define $\mathrm { T V } ( p , q )$ as the total variation distance between (possibly unnormalized) probability distributions $p$ and $q .$

Proposition C.5 (RFs GNN-based GFlowNets are Universal Approximators). Let $\{ ( p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) } ) \} _ { \gamma \in \Gamma }$ be a GFlowNet augmented with RFs $\gamma \in \Gamma$ and defined over a generative process over graphs such that each transition corresponds to either adding an edge or stopping. We define the following hypothesis classes.

$\begin{array} { r } { \begin{array} { r } { 1 . } \end{array} \mathcal { H } _ { \mathrm { t } } = \{ h \colon ( \mathcal { G } , n , \gamma _ { n } ) \mapsto \mathbb { R } ^ { d } , \sigma \colon \mathbb { R } ^ { d } \to \mathbb { R } , d \in \mathbb { N } \} } \end{array}$ containing 1-WL GNNs h and a real-valued function σ.

2. $\mathcal { H } _ { \mathrm { s } } = \{ \xi \colon ( \mathcal { G } , \{ \mathbf { x } _ { n } \colon n \in \mathcal { G } \} ) \mapsto y \}$ containing readout functions mapping a graph and its node embeddings to $y \in \mathbb { R }$

We parameterize $p _ { F } ^ { ( \gamma ) }$ for edge addition, respectively, as in Equation (11), assigning also a probability proportional to exp $\{ \xi ( s , \{ h ( s , n ) \colon n \in s \} ) \}$ for interrupting the generation at $s \in { \mathcal { S } }$ , and fix $\bar { p _ { B } ^ { ( \gamma ) } }$ . Then, for any target distribution $R _ { : }$ , there is a $h \in \mathcal { H } _ { \mathrm { t } }$ and $a \xi \in { \mathcal { H } } _ { \mathrm { s } }$ such that the marginal distribution $m _ { F } ^ { ( \gamma ) }$ satisfies $\mathrm { T } \bar { \mathrm { V } } ( m _ { F } ^ { ( \bar { \gamma } ) } , R ) \leq \epsilon$ with probability larger than $1 - \delta$ over the draw $o f \gamma$

With this in mind, a thorough investigation of the benefits of RF GFlowNets is left for future research. An extensive analysis of how to usefully define $p _ { M }$ is of particular importance.

RFs & Stability. Besides improving the expressivity of GNN-based GFlowNets, we also demonstrate RFs impose an implicit Tikhonov regularization (Hastie et al., 2009) on the learning objective that mitigate the instability of $p _ { B }$ observed in Figure 8 in Section A.2. To understand this, we recall that, if $L ( \theta )$ is a loss function associated with parameters θ and $\theta ^ { \star }$ is a local minimum of $L$

$$
L ( \theta ) = L ( \theta ^ { \star } ) + \frac { 1 } { 2 } ( \theta - \theta ^ { \star } ) ^ { \top } H ( \theta ^ { \star } ) ( \theta - \theta ^ { \star } ) + \mathcal { O } ( \| \theta - \theta ^ { \star } \| ^ { 3 } )
$$

with $H ( \theta ^ { \star } ) = \nabla _ { \theta } ^ { 2 } L ( \theta ^ { \star } )$ representing $L \mathrm { { s } }$ Hessian at $\theta ^ { \star }$ . From this, we observe that $H ( \theta ^ { \star } )$ characterizes the local geometry of L around $\theta ^ { \star }$ ; when $H ( \theta ^ { \star } )$ is nearly singular, the loss surface around $\theta ^ { \star }$ is almost flat, a configuration that first-order gradient-based methods struggle to navigate. We next show that incorporating RFs into the policies of a GFlowNet has the efect of increasing the eigenvalues of the loss function’s Hessian, reducing the geometry’s flatness and—as we will empirically illustrate—reducing the functional variability of the learned $\left( p _ { F } , p _ { B } \right)$ originating solely from parameter initialization. We start with the following lemma.

Lemma C.6 (Local Approximation for RF-augmented Loss). We define $\mathfrak { g } _ { \Gamma } = \{ p _ { F } ^ { ( \gamma ) } , p _ { B } ^ { ( \gamma ) } \} _ { \gamma \in \Gamma }$ as a GFlowNet augmented with $R F s \gamma$ . We assume $\gamma$ is drawn from a zero-centered distribution $p _ { M }$ with covariance matrix $\sigma ^ { 2 } I$ for some $\sigma > 0$ . Then,

$$
\mathbb { E } _ { \gamma \sim p _ { M } } \left[ \mathcal { L } _ { \mathrm { T B } } ( \tau , \gamma ; \mathfrak { g } _ { \Gamma } ) \right] = \mathcal { L } _ { \mathrm { T B } } ( \tau , 0 ; \mathfrak { g } _ { \Gamma } ) + \frac { 1 } { 2 } \sigma ^ { 2 } \cdot \mathrm { t r } \left( H _ { \gamma } ( \tau , 0 ) \right) + \mathcal { O } ( \sigma ^ { 4 } ) ,\tag{12}
$$

with tr being the trace operator, $H _ { \gamma }$ representing the Hessian of $\mathcal { L } _ { \mathrm { T B } } ( \tau , \gamma ; \mathfrak { g } _ { \Gamma } )$ with respect to $\gamma _ { ; }$ , and $\mathcal { L } _ { \mathrm { T B } }$ as in Definition B.2.

Lemma C.6 follows from a Taylor approximation of the loss function around $\gamma = 0$ and the fact that the odd moments of $\gamma$ null out. Notably, the quantity $\mathcal { L } _ { \mathrm { T B } } ( \tau , 0 ; \mathfrak { g } _ { \Gamma } )$ corresponds to the loss of an unaugmented GFlowNet. By averaging τ out of Equation (12), we show that the trace term within it acts as a regularizer.

Proposition C.7 (RFs as Regularizers). Let $ { \mathfrak { g } } _ { \Gamma }$ be as in Lemma C.6 and ${ \mathfrak { g } } _ { o }$ be the GFlowNet corresponding to $\gamma = 0$ . We define

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { \Gamma } ) = \mathbb { E } _ { \tau \sim \rho } \mathbb { E } _ { \gamma \sim p _ { M } } \left[ \mathcal { L } _ { \mathrm { T B } } ( \tau , \gamma ; \mathfrak { g } _ { \Gamma } ) \right] , \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { o } ) = \mathbb { E } _ { \tau \sim \rho } \left[ \mathcal { L } _ { \mathrm { T B } } ( \tau , 0 ; \mathfrak { g } _ { \Gamma } ) \right] , } \end{array}
$$

and $\mathcal { L } _ { \mathrm { R e g } } ( \mathfrak { g } _ { \Gamma } ) = \mathbb { E } _ { \tau \sim \rho } \left[ \mathrm { t r } ( H _ { \gamma } ( \tau , 0 ) ) \right]$ for an exploratory policy $\rho .$ Also, let θ be the parameters of $ \mathfrak { g } _ { \Gamma }$ , and let $H _ { \mathrm { T B } } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { \Gamma } ) , H _ { o } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { o } )$ , and $H _ { \mathrm { R e g } } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { \mathrm { R e g } } ( \mathfrak { g } _ { \Gamma } )$ . In this setting,

$$
H _ { \mathrm { T B } } = H _ { o } + \frac { \sigma ^ { 2 } } { 2 } \cdot H _ { \mathrm { R e g } } + \mathcal { O } ( \sigma ^ { 4 } ) .\tag{13}
$$

Importantly, Equation (13) suggests $\sigma$ should be set to a small positive number to ensure the reminder term in $\mathcal { O } ( \sigma ^ { 4 } )$ is negligible. To measure the efects of our method in practice, we calculate the variance of $\begin{array} { r l } { \mathrm { a v g K L } ( p _ { B } | | p _ { U } ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { | \cal { S } | } \sum _ { s \in \cal { S } } \mathrm { K L } [ p _ { B } ( s , \cdot ) | | p _ { U } ( s , \cdot ) ] } \end{array}$ , with $p _ { U }$ being an uniform backward policy. We train a continuously indexed (CI) GFlowNet with $\Gamma \ = \ \mathbb { R } ^ { 8 }$ and $\gamma \sim$ $\mathcal { N } ( 0 , 0 . 1 5 ^ { 2 } \cdot I )$ and a conventional GFlowNet for the Hypergrid environment (see Section 6). As in Figure 8, we fix both the architecture and the parameter initialization of the neural networks underlying $p _ { F } ^ { ( \gamma ) }$ and $p _ { B } ^ { ( \gamma ) }$ , and compute the per-step variance of the KL divergence by solely modifying the initial pseu-

![](images/f441ef603f72048ca3b861c61da50f4a73e0b02f39df0d482cf6a3b04258bfcb.jpg)  
Figure 10: Random features act as a regularizer for GFlowNet training.

dorandom seed of the exploratory policy. We repeat this process across 5 independent runs. In conformity with Proposition C.7, we observe in Figure 10 that RFs have a stabilizing efect on the learning problem, smoothing the loss surface and increasing the model’s robustness to the randomness of sample generation—as emphasized by the reduced variance of $\mathrm { a v g K L } ( p _ { B } | | p _ { U } )$ .

## D Discretely Indexed GFlowNets

We complement Section 5 by introducing further examples of how to partition the state space, besides the abstract modulo-based technique described in Remark 5.5.

Example D.1 (Length-based TC). Let $d ( x )$ be the trajectory-size, or description-length, for generating a $x \in \mathcal { X }$ . Common GFlowNet applications such as sequence design (Laajil et al., 2025), variable selection, drug discovery (Bengio et al., 2021), and structure learning (Deleu et al., 2022) often have $d ( x ) \in [ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$ with both $d _ { \mathrm { m i n } }$ and $d _ { \mathrm { m a x } }$ known. Under this condition, we can define $\mathrm { T C } ( \mathcal { X } , K ) = \{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ for $K \leq D : =$ $d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } + 1$ as

$$
\mathcal { X } _ { k } = \left\{ x \in \mathcal { X } \colon d _ { \operatorname* { m i n } } ^ { ( k ) } : = d _ { \operatorname* { m i n } } + \frac { k - 1 } { K } \cdot D \leq d ( x ) < d _ { \operatorname* { m a x } } ^ { ( k ) } : = d _ { \operatorname* { m i n } } + \frac { k } { K } \cdot D \right\} .
$$

The policy $p _ { F . } ^ { ( k ) }$ can be simply implemented by masking out terminating transitions leading to x for which $d ( x ) \not \in [ d _ { \operatorname* { m i n } } ^ { ( k ) } , d _ { \operatorname* { m a x } } ^ { ( k ) } )$

Example D.2 (Hamming weight-based TC). Let $\mathcal { X } = \{ 1 , 0 \} ^ { d }$ , which is often the cases in applications such as bit sequences (Malkin et al., 2022) and Ising model simulation (Liu et al., 2024). We define $\begin{array} { r } { w ( x ) = \sum _ { 1 \leq i \leq d } x _ { i } } \end{array}$ as the Hamming weight of x and $\mathrm { T C } ( \mathcal { X } , K ) = \{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ as

$$
{ \mathcal { X } } _ { k } = \{ x \in { \mathcal { X } } \colon w ( x ) \in [ a _ { k } , b _ { k } ) \}
$$

with $a _ { 1 } = 0 , b _ { K } = d + 1$ , and $\textstyle \bigcup _ { k = 1 } ^ { K } [ a _ { k } , b _ { k } ) = [ 0 , d + 1 )$ . We implement $p _ { F } ^ { ( k ) }$ by pruning transitions resulting in states for which the condition $w ( x ) \in [ a _ { k } , b _ { k } )$ is unattainable.

The tasks above can be easily seen to fall into the symmetric-junta category we outlined in Proposition 5.7. Example D.1 straightforwardly follows from picking $\psi ( s )$ to be equal to the number of components defining s. Example D.2, on the other hand, follows from letting s be represented as a subset of $\{ 1 , 0 \} \times [ d ]$ and defining $\psi ( s ) = \# ( s \cap r )$ with the reference set $r = \{ ( 1 , 1 ) , \ldots , ( 1 , d ) \}$ corresponding to a sequence containing only 1s. In either case, a tractable implementation for SC GFlowNets is admissible.

Implementation of SC GFlowNets. In addition, to further clarify the implementation of SC GFlowNets, we provide a high-level description of its training and inference in the pseudocode in Algorithms 1 and 2, respectively, alongside detailed and eficient computer code for reproducibility.

Algorithm 1 Training of SC GFlowNets   
Require: K, $\mathbf { Z } = ( Z _ { k } ) _ { k \in [ K ] }$   
Require: trainingIterations, $\epsilon > 0 ,$ batchSize   
Require: SC GFlowNet ${ \mathfrak { g } } _ { [ K ] }$ Algorithm 2 Inference with SC GFlowNets   
Require: getPolicies : $k , \mathfrak { g } _ { [ K ] } \mapsto \{ p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } \}$ Require: $K , \mathbf { Z } = ( Z _ { k } ) _ { k \in [ K ] }$   
for in range(trainingIterations) do Require: SC GFlowNet ${ \mathfrak { g } } _ { [ K ] }$   
Samples $ \{ \}$ Require: getPolicies : k, $\mathfrak { g } _ { [ K ] } \mapsto \{ p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } \}$   
parfor in range(batchSize) do   
$\begin{array} { r } { \mathbf p _ { Z }  ( 1 - \epsilon ) \cdot \frac { \mathbf Z } { \mathbf 1 _ { K } ^ { T } \mathbf Z } + \epsilon \cdot \frac { \mathbf 1 _ { K } } { K } } \end{array}$ k Categorical $\left( \frac { \mathbf { Z } } { \mathbf { 1 } _ { K } ^ { T } \mathbf { Z } } \right)$   
k  Categorical (p<sub>Z</sub>) $p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } \gets \mathrm { g e t P o l i c i e s } ( k , \mathfrak { g } _ { [ K ] } )$   
$p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } \gets \mathrm { g e t P o l i c i e s } ( k , \mathfrak { g } _ { [ K ] } )$ $\#$ Sample trajectory   
$\#$ Sample trajectory from $p _ { F } ^ { ( k ) }$ $\tau \sim p _ { F } ^ { ( k ) } ( s _ { o } , \cdot )$   
τ ϵ-greedy of ${ p } _ { F } ^ { ( k ) } ( s _ { o } , \cdot )$ $\#$ Get terminal state from $\tau$   
Samples Samples $\cup \left\{ ( \tau , k ) \right\}$ x TerminalState(τ)   
return x   
end parfor   
loss  lossFunction(Samples, ${ \mathfrak { g } } _ { [ K ] } , { \mathbf Z } )$   
<sup>g</sup>[K]<sup>,</sup> <sup>Z</sup> ← <sup>GradientStep(loss,</sup> ${ \mathfrak { g } } _ { [ K ] } , { \mathbf Z } )$   
end for

## E Revisiting Subgraph Asynchronous Learning (SAL)

This section and the next highlight how our framework can be instantiated to accommodate prior algorithms for learning GFlowNets. We start with SAL (Silva et al., 2025b). Similarly to SC GFlowNets, SAL defines

a fixed-horizon covering $\mathrm { ( F H C ) ^ { 4 } }$ of $\mathcal { X }$ , referred to as $\mathrm { F H C } ( \mathcal { X } , K )$

Definition E.1 (Fixed-Horizon Covering). We call $\mathrm { F H C } ( \mathcal { X } , K ) = \mathcal { X } _ { o } \cup \{ \mathcal { X } _ { k } \} _ { k = 1 } ^ { K }$ a fixed-horizon covering (FHC) if the following conditions are satisfied.

1. There are disjoint subsets $\{ S _ { k } \} _ { k = 1 } ^ { K }$ of $s$ such that $\mathcal { X } _ { k }$ contains all states in $\mathcal { X }$ reachable from some $s \in S _ { k }$

2. Each $s \in S _ { k }$ is within a fixed-distance d to $s _ { o } ,$ and all states within distance d of $s _ { o }$ are contained in some $S _ { k }$ (fixed-horizon).

3. $\mathcal { X } _ { o }$ is the subset of whose elements’ to $s _ { o }$ to x is smaller than $d .$

We call distance from s to $s _ { o }$ the length of the shortest-path from $s _ { o }$ and $s .$

Drawing on Definition E.1, each partition enforces a so-called Amortized Trajectory Balance (ATB), defined as

$$
F ( s , k ) p _ { F } ^ { ( k ) } ( s , \tau ) = R ( x ) p _ { B } ( x , \tau ) \mathrm { ~ f o r ~ a l l ~ } k \in [ K ] , s \in S _ { k } , x \in \mathcal { X } _ { k } , \mathrm { ~ a n d ~ } \tau \in s \to x ,\tag{14}
$$

and then $Z \cdot p _ { F } ^ { ( o ) } ( s , \tau ^ { \prime } ) = F ( s , k ) p _ { B } ( s , \tau ^ { \prime } )$ and $Z \cdot p _ { F } ^ { ( o ) } ( s , \tau ^ { \prime } ) = R ( x ) p _ { B } ( s , \tau ^ { \prime } )$ for $k \in [ K ] , s \in \mathcal { S } _ { k } , x \in \mathcal { X } _ { o }$ , and $\tau \in s _ { o }  s$ and $\tau ^ { \prime } \in s _ { o }  x$ . In other words, each model first enforces a balance condition within its own partition according to a shared (uniform) backward policy $p _ { B } ,$ and their learned flow functions $F ( \cdot , k )$ are later used as the target distribution of the centralized model’s $( p _ { F } ^ { ( o ) } )$ own learning objective.

We next show that a solution to SAL’s amortized TB can be interpreted as a DI GFlowNet satisfying its own MTB in Definition 3.2. We do this by introducing a Doob h-transform (see Section 5.2) that maps $p _ { F } ^ { ( o ) }$ and $p _ { F } ^ { ( k ) }$ into a policy function that strictly goes through $S _ { k }$ when sampling from $\mathcal { X } _ { k }$ . With this in mind, we let

$$
T _ { k } ^ { ( f ) } = \operatorname* { m i n } \{ t \colon S _ { t } ^ { ( f ) } \in { \mathcal { S } } _ { k } \} { \mathrm { ~ a n d ~ } } T _ { k } ^ { ( b ) } = \operatorname* { m i n } \{ t \colon S _ { t } ^ { ( b ) } \in { \mathcal { S } } _ { k } \}
$$

be the hitting times on $S _ { k }$ of the Markov chains introduced in Proposition 5.9, and

$$
h _ { k } ( s ) = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } = \operatorname* { m i n } _ { k ^ { \prime } \in [ K ] } T _ { k ^ { \prime } } ^ { ( f ) } < \infty | s \right] \mathrm { ~ a n d ~ } g _ { k } ( s ) = \mathbb { P } _ { \kappa _ { B } } \left[ T _ { k } ^ { ( b ) } = \operatorname* { m i n } _ { k ^ { \prime } \in [ K ] } T _ { k ^ { \prime } } ^ { ( b ) } < \infty | s \right] .
$$

be the corresponding probabilities that a Markov chain starting at $s \in { \bar { S } }$ and following $\kappa _ { F } ~ ( \mathrm { r e s p . } ~ \kappa _ { B } )$ arrives at $\scriptstyle { S _ { k } }$ prior to arriving at any other $\boldsymbol { S } _ { \boldsymbol { k } ^ { \prime } }$ with $\boldsymbol { k } ^ { \prime } \ne k$ . We first show that both $h _ { k }$ and $g _ { k }$ are harmonic functions.

Lemma E.2. Both $h _ { k }$ and $g _ { k }$ are harmonic functions on $\bar { \cal S } \setminus { \cal S } _ { k }$ for $k \in [ K ]$ .

A core property required by the next Proposition E.4 is that both $h _ { k } ( s _ { o } )$ and $g _ { k } ( x )$ for $x \in \mathcal { X }$ can be interpreted as probability distributions over $[ K ]$ . As we emphasize next, this intuitively follows from the fact that $\{ S _ { k } \} _ { k = 1 } ^ { K }$ forms a boundary between $s _ { o }$ and the set of terminal states.

Lemma E.3. The functions $h _ { k }$ and $g _ { k }$ in Lemma E.2 satisfy

$$
\sum _ { 0 \leq k \leq K } h _ { k } ( s _ { o } ) = 1 \ a n d \ \sum _ { 0 \leq k \leq K } g _ { k } ( x ) = 1 \ f o r \ x \in \mathcal { X }
$$

with $h _ { o } ( s _ { o } ) = \mathbb { P } _ { \kappa _ { F } } \bigg [ T _ { k } ^ { ( f ) } = \infty \forall k \in [ K ] \bigg | s _ { o } \bigg ]$ and $g _ { o } ( x ) = \mathbb { P } _ { \kappa _ { B } } \bigg [ T _ { k } ^ { ( b ) } = \infty \forall k \in [ K ] \bigg | x \bigg ]$ referring to the case when the generative process terminates on $\mathcal { X } _ { o }$ (see Definition $\breve { E . 1 } )$

Building on Lemmas E.2 and E.3, we demonstrate in Proposition E.4 that $\mathrm { S A L }$ can be viewed as an instance of a DI GFlowNet by showing that a solution to Equation (14) can be uniquely represented as a solution to the mixture learning problem in Definition 3.2 with a specific choice of $Z _ { k }$ and $q ( k | x )$ . This suggests yet another form of designing a DI GFlowNet. It should nonetheless be noticed that, contrarily to SC GFlowNets, SAL is an inherently distributed algorithm—as neither $h _ { k }$ nor $g _ { k }$ in Lemma E.2 can be tractably computed in general.

Proposition E.4 (SAL as a DI GFlowNet). Let $\{ p _ { F } ^ { ( k ) } \} _ { k = 0 } ^ { K }$ and p<sub>B</sub> be the solutions to SAL’s balance equations having d as the distance from $s _ { o }$ to $\{ S _ { k } \} _ { k = 1 } ^ { K }$ . We define

$$
\tilde { p } _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) = \left\{ \begin{array} { l l } { p _ { F } ^ { ( o ) } ( s , s ^ { \prime } ) \cdot \frac { h _ { k } ( s ^ { \prime } ) } { h _ { k } ( s ) } \ i f \operatorname { d i s t } ( s _ { o } , s ^ { \prime } ) \le d , } \\ { p _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) \ o t h e r w i s e , } \end{array} \right.
$$

with dist representing the shortest-path distance in the underlying state graph, and

$$
\begin{array} { r } { \tilde { p } _ { B } ^ { ( k ) } ( s , s ^ { \prime } ) = \left\{ { p } _ { B } ( s , s ^ { \prime } ) \ i f \mathrm { d i s t } ( s _ { o } , s ^ { \prime } ) \leq d , \right. } \\ { \left. p _ { B } ( s , s ^ { \prime } ) \cdot \frac { g _ { k } ( s ^ { \prime } ) } { g _ { k } ( s ) } \ o t h e r w i s e , \right. } \end{array}
$$

for $k \geq 1$ and $\tilde { p } _ { B } ^ { ( o ) } ( s , s ^ { \prime } ) = p _ { B } ( s , s ^ { \prime } )$ for $k = 0$ . Then, for $k \in \{ 0 , \ldots , K \}$ and each trajectory τ from $s _ { o }$ to $x \in \mathcal { X } _ { k }$ going through $\boldsymbol { S } _ { k }$

$$
Z _ { k } \tilde { p } _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = R ( x ) q ( k | x ) \tilde { p } _ { B } ^ { ( k ) } ( x , \tau )
$$

with $\begin{array} { r } { Z = \sum _ { x \in \mathcal { X } } R ( x ) } \end{array}$ and $Z _ { k } = Z \cdot h _ { k } ( s _ { o } ) $ and $q ( k | x ) = g _ { k } ( x )$

In simpler terms, Proposition E.4 highlights SAL can be cast into our mixture learning framework by reframing $h _ { k } ( s _ { o } )$ and $g _ { k } ( x )$ as our prior and posterior mixing distributions, an interpretation supported by the normalization property established in Lemma E.3. Notably, the central statements on the correctness of SAL in (Silva et al., 2025b) can be immediately derived from Proposition E.4, highlighting the generality of DI GFlowNets as a theoretical tool for devising GFlowNet training algorithms with proven correctness. The next section emphasizes this point further.

## F Revisiting Boosted GFlowNets

This section is separated into two parts. The first one shows how Boosted GFlowNets can be understood through the lens of our abstract framework. The second one evaluates Boosted GFlowNets against our proposed SC GFlowNets, highlighting the computational amenability of the latter when compared to the former.

## F.1 Boosted GFlowNets satisfy the Mixed Trajectory Balance

As with SC GFlowNets and SAL, Boosted GFlowNets learn distinct samplers focusing on nearly-disjoint regions of the state space (Locatello et al., 2018; Dall’Antonia et al., 2026b). However, instead of relying on amortization (as in SC GFlowNets) or parallelization (SAL) over fixed structural primitives, Boosted GFlowNets adaptively learn a residual target distribution in a sequential fashion. Their connection to DI GFlowNets, which will be developed in this section, will reveal our framework’s applicability in the context of countably infinite mixtures.

To see this, we briefly recall the algorithmic design of Boosted GFlowNets. Let $\{ \mathfrak { g } _ { k } \} _ { k \ge 1 } : = \{ ( Z _ { k } , p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } ) \} _ { k \ge 1 }$ be GFlowNets and, for $\tau \in s _ { o }  x ,$ define

$$
R ^ { ( 0 ) } ( x , \tau ) = 0 \mathrm { ~ a n d ~ } R ^ { ( k ) } ( x , \tau ) = R ^ { ( k - 1 ) } ( x , \tau ) + \frac { Z ^ { ( k ) } p _ { F } ^ { ( k ) } ( s _ { o } , \tau ) } { p _ { B } ^ { ( k ) } ( x , \tau ) }
$$

for $k \geq 2$ . Intuitively, $R ^ { ( k ) } ( x , \tau )$ represents an estimate of the accumulated probability mass on x by the first k models. An implicit assumption of Boosted GFlowNets, which is compatible with the observed empirical

behavior of GFlowNets (Shen et al., 2023), is that each GFlowNet underestimates the total probability mass in any given region.

Assumption F.1 (Underallocation of probability mass). After trained, we assume assume the first k-th GFlowNets satisfy ${ \dot { R } } ^ { ( k ) } ( x , \tau ) \leq R ( x )$ for all x and $\tau \in s _ { o }  x$

Under Assumption F.1, each GFlowNet is trained by minimizing the boosted trajectory balance (BTB) loss function, which replaces $R ( x )$ by the residual target $R ( x ) - R ^ { ( K ) } ( x , \tau )$ with τ drawn from the corresponding backward policy $p \bar { (} \mathbf { \Psi } _ { B } ^ { K ) } ( x , \cdot )$

Definition F.2 (Boosted Trajectory Balance Condition & Loss). Let $\{ \mathfrak { g } _ { k } \} _ { k = 1 } ^ { K }$ be a sequence of GFlowNets with $\{ \mathfrak { g } _ { k } \} _ { k = 1 } ^ { K - 1 }$ satisfying Assumption F.1. By defining

$$
R ^ { ( k ) } ( x ) = \mathbb { E } _ { \tau \sim p _ { B } ^ { ( k ) } ( x , \cdot ) } \left[ R ^ { ( k ) } ( x , \tau ) \right] ,
$$

we call K-th Boosted Trajectory Balance (BTB) condition

$$
Z _ { K } \cdot p _ { F } ^ { ( K ) } ( s _ { o } , \tau ) = ( R ( x ) - R ^ { ( K - 1 ) } ( x ) ) \cdot p _ { B } ^ { ( K ) } ( x , \tau ) ,
$$

which is enforced through the loss function

$$
\mathcal { L } _ { \mathrm { B T B } } \left( \tau \middle | \left\{ \mathfrak { g } _ { k } \right\} _ { k = 1 } ^ { K } \right) = \left( \log \frac { Z _ { K } \cdot p _ { F } ^ { ( K ) } ( s _ { o } , \tau ) } { \left( R ( x ) - R ^ { ( K - 1 ) } ( x ) \right) \cdot p _ { B } ^ { ( K ) } ( x , \tau ) } \right) ^ { 2 } ,
$$

which we refer to as BTB loss.

The intuition for the BTB loss is that, when the the first K 1 GFlowNets collapse onto a subset of the state space, the K-th GFlowNet will—through the influence of the residual target—mostly ignore such region. In practice, (Dall’Antonia et al., 2026b) proposes several techniques for stabilizing training when the quantity $R ( { \bar { x } } ) - R ^ { ( K ) } ( x )$ becomes overly small, to which the interested reader is referred to for further details; however, none of them are relevant to our discussion. It should be clear from Definition F.2 and Assumption F.1 that Boosted GFlowNets only sample from the target distribution when the trajectory-wise residual target vanishes and that the BTB loss can only be non-biasedly evaluated if we can compute $R ^ { \check { ( } K - 1 ) } ( x )$ exactly. The latter is satisfied when $R ^ { ( K - 1 ) } ( x , \tau ) = R ( x )$ for every τ in the support of $p _ { F }$ , and 0 otherwise; equivalently, when the on-policy expectation of the BTB loss under $p _ { F } ^ { ( K - 1 ) }$ vanishes. This was the main condition for (Dall’Antonia et al., 2026b, Theorem 1). We explicitly state these behaviors in our next assumption.

Assumption F.3 (Convergence of residuals & Collapse). For all $x \in \mathcal { X }$ and $\tau \in s _ { o }  x$ , the series $\{ R ^ { ( k ) } \} _ { k \geq 1 }$ satisfies lim ${ } _ { ^ { 1 } k  \infty } R ^ { ( k ) } ( x , \tau ) = R ( x )$ . Also, there is a subset $\mathcal { X } _ { k }$ of such that $R ^ { ( k ) } ( x , \tau ) = R ^ { ( k ) } ( x )$ for each $\tau \in s _ { o }  x$ and $x \in \mathcal { X } _ { k }$ , and $R ^ { ( k ) } ( x , \tau ) = 0$ otherwise.

We now show that Assumptions F.1 and F.3 allow us to characterize Boosted GFlowNets as a solution to the MTB equations in Definition 3.2 based on a countably infinite mixing distribution. Importantly, as with classical topic modelling (Blei et al., 2003; Hofman et al., 2010) based on Dirichlet processes, we typically choose a $K$ and truncate the prior mixing distribution to be supported on [K]. Dall’Antonia et al. (2026b) sets K to 2 or 3 in their experiments.

Proposition F.4 (Boosted GFlowNets as DI GFlowNets). We define $\{ \mathfrak { g } _ { k } \} _ { k \ge 1 } : = \{ ( Z _ { k } , p _ { F } ^ { ( k ) } , p _ { B } ^ { ( k ) } ) \} _ { k \ge 1 }$ <sub>1</sub> as GFlowNets satisfying Assumptions F.1 and F.3. By letting

$$
R _ { k } ( x ) = R ^ { ( k ) } ( x ) - R ^ { ( k - 1 ) } ( x ) \ a n d \ q ( k | x ) = { \frac { R _ { k } ( x ) } { R ( x ) } } ,
$$

then the following properties are satisfied.

1. $q ( k | x )$ is a probability distribution, i.e., $\begin{array} { r } { \sum _ { k \geq 1 } q ( k | x ) = 1 } \end{array}$ for each $x \in \mathcal { X }$

![](images/bd8b62c43bd71ef7c5bc3ac210212d5c88b2738ecc3ad40f2c3292af3835bb90.jpg)

![](images/893f95feaa6f65f0d1e8dd6ad620e727dc70eeab14cdbe402c696d1965359f8b.jpg)  
(a) Runtimes per training step.

![](images/0a9c1e4305f3930914c2a7ea2a01b6bb01b69c13daacc9392c8163c9f7bc54cb.jpg)

![](images/cbc762a260cd81c3fd84d70f294e62797ed6486ca8f5c8121ff6d637ae6a20b7.jpg)

![](images/2972b94546157b4334712f1290c4848b36a01dd5cb6c6798cf268f759487432d.jpg)  
(b) FLOPs per training step.

Figure 11: SC GFlowNets run both faster (left) and more compute-eficiently (right) than Boosted GFlowNets for the Set Generation domain. Throughout this section, runtimes represent per-step averages over 3 independent runs, each with 600 iterations.  
![](images/eb095351792188c2a71441c594a2fdcdd5ed1b1ca4763655d7277c1881dcbd60.jpg)

![](images/f5603ec893c8fad28152db2a1314ad7c1e7a9c6468615c160df4113eb3794fc4.jpg)  
(a) Runtimes per training step.

![](images/47b7fa416ef306e4dd2c0a4340b6923a733ba682e3a0deb5b2bb094af8f9b86f.jpg)

![](images/7b996b8071e54bb8c3d49088124c96c6611c78cd511027568e6bf23df47185f4.jpg)  
(b) FLOPs per training step.

Figure 12: SC GFlowNets are more compute-eficient than Boosted GFlowNets in the Hypergrid domain, requiring substantially fewer FLOPs per gradient step during training.

2. The sequence $\{ { \mathfrak { g } } _ { k } \} _ { k \geq 1 }$ satisfies a MTB condition

$$
Z _ { k } \cdot p _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = R ( x ) q ( k | x ) p _ { B } ^ { ( k ) } ( x , \tau )
$$

$f o r \ k \geq 1$ , with $\begin{array} { r } { \sum _ { k \geq 1 } Z _ { k } = Z } \end{array}$

Once again, the correctness of Boosted GFlowNets can be directly inferred from Corollary 5.2 and Proposition F.4. However, contrarily to SC GFlowNets, which support both amortization and distributed training, Boosted GFlowNets can only be trained via sequential learning. Also, in contrast to both SC GFlowNets and SAL, Boosted GFlowNets cannot be trained with an near-embarrassingly parallel algorithm, primarily due to the unbounded support of the mixing distribution and the dependence of each model on its predecessors’ residuals.

In addition, we notice SC GFlowNets are the most foundational instantiation of DI GFlowNets when compared to both SAL and Boosted GFlowNets: stratum-conditioning could be used for training each booster in Boosted GFlowNets and each sampler in SAL, but neither of these algorithms could be used to learn SC GFlowNets without sacrificing its monolithicness, which we deem to be one of its fundamental properties—a single model can be used for learning the entire mixture.

## F.2 Experiments

We now present several experiments to support our claim that (i) Boosted GFlowNets introduce a substantial computational overhead over the base model, which is not observed for SC GFlowNets, and (ii) SC GFlowNets often converge faster than Boosted GFlowNets.

Computational profile. We start by showcasing the cost of Boosted GFlowNets is substantially larger than of SC GFlowNets in terms of both runtime and FLOPs per training step for the Set Generation, Hypergrid, and Lazy Random Walk domains. To evaluate the latter, we use JAX’s cost analysis method for compiled functions (Bradbury et al., 2018). Results are displayed in Figures 11 to 13. Notably, SC GFlowNets are up to roughly 1.6 faster than Boosted GFlowNets per step (for Set Generation), and execute up to 50% fewer FLOPs per training step. The reasons for this are clear: as discussed before, our method learns an amortized policy over the mixture’s components, while boosting requires training independent neural networks in a sequential fashion, having a monotonically increasing cost in the number of boosters (components). Importantly, during our experiments, we proceeded as suggested by Dall’Antonia et al. (2026b) and allocated a prescribed number of iterations equally among a predefined number of boosters for training; automatic sampler elicitation remains an unresolved issue for Boosted GFlowNets.

![](images/f4ec0c876559cc44d9a0904157cff2aa8be66bbdf362424b00484278e80acb7e.jpg)

![](images/30403cc7a86fbb95ff89cb5529429b25de8c47f6e52861efbe806d2822fe3546.jpg)  
(a) Runtimes per training step.

![](images/9ba689d51d2518f7ec7198a6370177cf8807fcb1d84d1446f45a8ee07484e31f.jpg)

![](images/0afde47ae23c83f5b541407df0e23a2eeef160a707e1ebd277181e89672b8a3c.jpg)  
(b) FLOPs per training step.

Figure 13: SC GFlowNets is far more eficient than Boosted GFlowNets in the Lazy Random Walk domain (Rings corresponds to the Rings target; GM, to Gaussian Mixture).  
![](images/6b60dc6345d260e7dabc31c57a16f7fac87ba2f19b89e8e11f48a54ec642c297.jpg)

![](images/7e232d1cac99fcea431edc6411e9854dcebc89a663595ae6e4fafa5620bf31a5.jpg)

![](images/ed933f284ebff0862a0d276edcd84a7e0194afe523cc1ac2fa809a89572c19e8.jpg)  
Figure 14: SC GFlowNets converge substantially faster than Boosted GFlowNets in the Set Generation domain, particularly for larger state spaces. See also Figure 3 for $\mathrm { E r r } _ { \mathrm { i t e m } } \mathrm { ^ { * } s }$ definition.

Learning convergence. In addition to being more compute-eficient, SC GFlowNets often lead to faster learning convergence in the considered tasks. This is highlighted in Figures 14 to 16. The intuition for this is that, while both Boosted and SC GFlowNets enhance state space exploration by design, hence improving the rate of discovery of high-probability regions, our method does so by stably partitioning the state space, ensuring consistent, non-random exploration of specific subsets of it. In contrast, boosting is built on a more delicate approach that is is reliant upon accurate training of several independent neural networks, and can be biased—unless Assumptions F.1 and F.3 are satisfied.

## G Experimental Details & Further Experiments

This section extends empirical analysis in Section 6 with additional domains, target distributions, and TC’s constructions. In addition, we provide further experimental details for reproducing our results. We also attach the computer code, written using JAX (Bradbury et al., 2018), we used for training and evaluating each of the described models on the considered tasks.

![](images/15c9ba27ba25c6a70775c3d435f840984195637a0e9ffd5da86e0837492337bd.jpg)

![](images/6bd1b872207145f4da5509be04778a1edadfa58ac3be4860ded64b67fe1665f1.jpg)  
Figure 15: SC GFlowNets exhibit faster learning convergence when compared to Boosted GFlowNets, while also being far more compute-eficient (Figure 12).

![](images/a11734c84a9338cad6bf0b0b004016bca96fd318d13a3016f9e12453d16c35a1.jpg)

![](images/28381ab2d7ccd61faeff317bda600333427a12536a4c49a0843562599fadc701.jpg)  
(a) Rings.

![](images/8c01fc1b5ef30e73dcec8b8cfb76e7e98704590d6a6c30a74990f143251d2785.jpg)

![](images/6fcb4ff009c5afb4e5bfc2411773c722bc78ae61eb989b2149290f8cd52f0e56.jpg)  
(b) Gaussian Mixture.

Figure 16: On top of being more computationally eficient (Figure 13), SC GFlowNets also learn a more accurate distributional approximation to the target relatively to Boosted GFlowNets in both the Rings and Gaussian Mixture variants of the Lazy Random Walk domain.  
![](images/5cd05e48ce0d621cb360e93a75ee98cbb8ed49d38067878f85059dc7e1817fff.jpg)

![](images/647bd8542efc404eb5d7d9012e37def0613267ba1e60573b7a935a0590cf3f5c.jpg)  
(a) L = 18.

![](images/d8171e7afdb1fe4b9917da227a0464633ac106d1cca86ebf1af5cdb45d31466c.jpg)

![](images/096c378bf98584ea672141eb02f1de86faa0635dbcfd4254abd586f07858f3cc.jpg)  
(b) L = 24.  
Figure 17: SC GFlowNets significantly accelerate learning convergence when approximating the generative process in Equation (15), with greater improvements being observed for larger state spaces. $\mathrm { T V } _ { \mathrm { i t e m } }$ and $\mathrm { T V } _ { \mathrm { l e n g t h } }$ are defined in Equation (17).

## G.1 More Experiments

Autoregressive sequence generation. We train a GFlowNet to reproduce the following generative process of token sequences $\left( t _ { 1 } , \ldots , t _ { l } \right)$ over a finite vocabulary,

$$
t _ { 1 } \sim p _ { \mathrm { i n i t } } , l \sim p _ { \mathrm { l e n g t h } } , t _ { i } | t _ { i - 1 } \sim p _ { \mathrm { k e r n e l } } ( t _ { i - 1 } , \cdot ) { \mathrm { f o r } } i \in \{ 2 , \ldots l \} .\tag{15}
$$

We let $[ V ] : = \{ 1 , \ldots , V \}$ be the support of $p _ { \mathrm { i n i t } }$ (vocabulary) and $p _ { \mathrm { k e r n e l } } ( t , \cdot )$ for $t \in [ V ]$ and $\{ 0 \} \cup [ L ]$ be that of $p _ { \mathrm { l e n g t h } }$ . The target distribution associated with $\mathbf { t } = ( t _ { 1 } , \ldots , t _ { l } )$ is

$$
R ( \mathbf { t } ) = p _ { \mathrm { i n i t } } ( t _ { 1 } ) p _ { \mathrm { l e n g t h } } ( l ) \prod _ { 2 \leq i \leq l } p _ { \mathrm { k e r n e l } } ( t _ { i - 1 } , t _ { i } ) .\tag{16}
$$

The GFlowNet is built upon a generative process starting at an empty sequence and iteratively deciding either to add a token from [V ] or to stop the generation, and the SC GFlowNet is designed according to Example D.1 with $K = 3$ partitions. Due to its simplicity, many quantities of interest from the model in Equation (16) can be tractably computed, including the marginal distribution over sequence lengths and the value of $t _ { i }$ for $i \in [ L ]$ . We denote these by $\pi ( l )$ and $\pi ( i , v )$ for $i \in [ L ]$ and $v \in [ V ]$ , respectively, and evaluate our model with

$$
\mathrm { T V } _ { \mathrm { l e n g t h } } = \frac { 1 } { 2 } \sum _ { 0 \leq l \leq L } | \hat { \pi } ( l ) - \pi ( l ) | \mathrm { ~ a n d ~ } \mathrm { T V } _ { \mathrm { i t e m } } = \frac { 1 } { 2 } \operatorname* { m a x } _ { v \in [ V ] } \sum _ { 1 \leq i \leq L } | \hat { \pi } ( i , v ) - \pi ( i , v ) | ,\tag{17}
$$

with ˆπ being a Monte Carlo estimate of π based on the learned sampler. We illustrate the evolution of these metrics throughout training in Figure 17. As we can observe, SC GFlowNets significantly improve learning convergence.

We also evaluate ASC GFlowNets on this domain, training $K = 3$ separate samplers in parallel on distinct subsets of the state space. We show the results in Figure 18. Again, we notice our partitioning technique significantly speeds up training.

![](images/dcbb3efebd1bb69f9b3be22a7fb37b5343a85859bece33abe85cf5a95528b189.jpg)

![](images/fc2239e4d137383ddf2ba9757cddd292a721f9d510d21e6ea2e13562fcd5f734.jpg)  
(a) L = 18.

![](images/1a60d318c6853893c1d200c0bf646968bdd7654a7767b5907a93611f123fdfcf.jpg)

![](images/ad903121bb15f6079d7dea8db753a4aa5c24d2d56d3774a928947c3a62608e16.jpg)  
(b) L = 24.

Figure 18: Embarrassingly parallel counterpart of Figure 17, comparing ASC GFlowNet against the best of K randomly initialized and independently trained GFlowNets with matched computational cost. ASC GFlowNet accelerates learning convergence.  
![](images/11a9098d022432eced9728da6eb5c9c4554ffb1bf3e8e298598c35bd32dd5327.jpg)

![](images/5ffbb2d3890bbbc4915a4f8958ab1c1ff2707da3a2b1fbb890f8b24a9ec434f5.jpg)  
Figure 19: Accuracy of SC GFlowNets as a function of the number of components in the mixture.

Another question that comes up when designing SC GFlowNets is how to select the number of partitions. Figure 19 shows that, for this domain, there is no significant diference between choosing $K \in \{ 2 , 3 , 4 , 5 , 1 0 \}$ This may not always be the case, however; see Figure 26. We discuss this further when describing the Lazy Random Walk domain.

Importantly, the question of whether one should use ASC or SC GFlowNets depends primarily on the available computational resources. ASC

GFlowNets, being massively parallelizable, have access to larger training sets and greater model capacity than any individual SC GFlowNet, being expected to both improve state space exploration and learning convergence, should the extra processing power be available. From an implementational perspective, as the samplers are architecturally identical, this parallelization can be directly attained by using tools designed to express single-program, multiple-data (SPMD) computational models (Darema et al., 1988), such as jax.pmap.

Lazy Random Walk. The state space for the Lazy Random Walk domain is $S ~ = ~ \{ ( \mathbf { x } , t ) \colon \mathbf { x } ~ \in$ $\{ - H , \dots , H \} ^ { d }$ and $0 \leq t < T \}$ and $\mathcal { X } = \{ ( \mathbf { x } , T ) \colon \mathbf { x } \in \{ - H , \ldots , H \} ^ { d } \}$ for given positive integers $H , \ d ,$ and T. The generative process starts at $\boldsymbol { s } _ { o } = ( \mathbf { 0 } , 0 )$ , and each iteration adds either $1 , \ - 1$ , or 0 to a chosen coordinate of x—as long as it remains within the boundary of —and increments t by one until $t = T$ As in (Dall’Antonia et al., 2026a; Silva et al., 2026), we consider $d = 2 , H = 1 6$ , and $T = 3 2$ . Our target distributions, Rings and Gaussian Mixture, are shown in Figure 20.

Notably, as the trajectory size is the same for each $( \mathbf { x } , T )$ , the length-based partitioning of Example D.1 cannot be used for Lazy Random Walk. Instead, we propose a uniform distance-based partitioning using the function $\psi ( \mathbf { x } , T ) = \| \mathbf { x } \| _ { 1 }$ in Remark 5.5 with $K = 3$ . We compute the TV distance between the learned and target distributions to measure a sampler’s goodness-of-fit.

Figure 20 shows both the model’s accuracy throughout training and a snapshot of the learned distribution for both SC and a traditional GFlowNet. In addition, we present in Figure 21 each component $p _ { F } ^ { ( k ) }$ for $k \in \{ 1 , 2 , 3 \}$ of the SC GFlowNet. We notice that our mixture modelling framework both accelerates learning convergence and improves fitness—even when the chosen partitioning does not separate the target distribution’s high probability regions, as in the Gaussian Mixture example. The intuition for this is that the sampler will naturally avoid being pulled back to and bogged in previously visited modes during training when we constraint the future value of $\lVert \mathbf { x } \rVert _ { 1 }$ at $t = T$ . We also compare ASC to SC GFlowNets in Figure 24 for both examples; as expected, ASC GFlowNets find a distinctly better fit to the target—particularly in the case of Rings, for which each component receives an easier target with connected high-probability regions.

Importantly, due to having a tractably enumerable state space, the $\mathrm { L A Z Y }$ Random Walk domain allows us to eficiently evaluate the log-partitions functions. This provides another road for evaluating our model. To see this, we denote by log $\hat { Z } _ { k }$ and log $\hat { Z }$ the learned partition functions, and by log $Z _ { k }$ and log $\hat { Z }$ their

k = 1  
![](images/8f78669f32ad723c5d468e0470729b29b8035e95cd069f75d13b93b36adb537e.jpg)

![](images/606c7003c51b159af132d8bd47812d9dde8d15b2afb8b5fd3e55e4f2f12efc3e.jpg)

![](images/aa49f9a77c3953787b6f02b66a74a8378d9eb34f07303e49ee809ba576d8ceb4.jpg)

![](images/38ca4c44ad63ecc2208c8be15b78d45929effb3e92ed60c0127d976c38de5b4e.jpg)

![](images/0c7fee267b32db9401e160afd9af9f7e7ba28de101a7026d908fbc3a8eba9bef.jpg)  
GFlowNet

![](images/36fb61dba8e58663555f1d53f7083ce76255ba15b1fa9adc8f9fcdadba934499.jpg)  
(a) Rings.

![](images/e16d8dfbe2054f9ca471dc4cd1af1cca8d3735880d1cef36e68ea7e88792d2f3.jpg)

![](images/2ee92b1c6e3f221cdc1bafc3ae37cfbb607597504c6f66a86dbbe4b071abb8d5.jpg)

![](images/81ffffd3f6171485e1360601e83681ed05b9a51efe3713400e061803a800475f.jpg)  
(b) Gaussian Mixture.

![](images/cc4a0017bc2f5b792c4a59eebe099b3cae94f00fde8376f8b9e2e2a04e88bb86.jpg)

Figure 20: SC GFlowNets improve goodness-of-fit for the Lazy Random Walk task, exhaustively covering the target distribution’s high probability regions even for an imperfect state space partitioning (as for the Gaussian Mixture; see Figure 21).  
![](images/c67e6ce9adfa553b758c46143da8e47bf693e1bbf67226b7ef8f9fdafef166e3.jpg)

![](images/43cc648eb3aedbeebf099bc56f1ed5ad0c03ed1bbf764b5fdef11375121d6909.jpg)  
(a) Rings.

![](images/7049cba4ffbaf45c02d300d1c52d61dae3234a3fedc90ae7fd0928f632e6f25a.jpg)

![](images/661e9dba02b8397d69fce8077029f72ddd2f53ca20dd8983dfdca83d136a8bc9.jpg)

![](images/eb611e3351de17e88c73bb4712547339a3098420333092d203725e0f166fbd35.jpg)  
k = 2  
(b) Gaussian Mixture.

![](images/aeb883cb8729dd22d513a75e8f695377542cf8adb9ba2a52aa48e8070ba52a37.jpg)  
k = 3

Figure 21: SC GFlowNet’s marginal $m _ { F } ^ { ( k ) } ( s _ { o } , x )$ (recall Equation (1)) in each $x \in \mathcal { X }$ for both the (a) Rings and (b) Gaussian Mixture target distributions of the Lazy Random Walk domain.  
![](images/f0b5a6c4864f4ffc774f27c11c48d2478611d156eb5a53c04bd328da4d2df9c9.jpg)  
(a) Rings.

![](images/9fe0ab54065bb0d4b63a3027933fbbdb575804b765273377b7ef809f546963d1.jpg)  
(b) Rings.

![](images/21028a0c0b5504e29b94800e6082445d12aecd7b535ed8ff029c4b34bd911c10.jpg)  
(c) Gaussian Mixture.

![](images/3d9f413a089016ebf3c61d3dad0442f651b070df8425aedd6f63ad3331968ca1.jpg)  
(d) Gaussian Mixture.

Figure 22: Learned log-partition functions by ASC GFlowNets in the Lazy Random Walk domain. We recall that, for Rings, $k = 3$ is the starved component, containing negligible probability mass; for Gaussian Mixture, it is $k = 1$ . As expected, ASC GFlowNets learn the correct values $( \mathrm { a } , \mathrm { c } ) ;$ however, the approximation of log $Z _ { k }$ for regions with near-zero probability is imperfect $( \mathrm { b } , \mathrm { d } )$ , as sizeable deviations in log $Z _ { k }$ in this case have little efect on distributional accuracy.

![](images/4c9b9d78852681fec918d14b585398de809dd7950cb141186035f365bd74411c.jpg)  
(a) Rings.

![](images/4c89a5e49b3873c0372ef1d9331be3b4ba0ce0ac4f7ab114c7d7cd2abfc00fc5.jpg)  
(b) Gaussian Mixture.

Figure 23: KL between prior mixing distributions.  
![](images/9d20e2a54f27c667c0046c9a536691adfd498a2b36b10a709e2161243fe562ef.jpg)  
(a) Rings.

![](images/27a0ee69d6b432d7877c7dc3f9dde72c4bb850b0221395a96d1309e242cf1192.jpg)  
(b) Gaussian Mixture.  
Figure 24: ASC GFlowNets converge faster than their synchronous counterparts in the Lazy Random Walk task, achieving a more accurate approximation in fewer training iterations.

ground-truth values for $k \in [ K ]$ . We consider the metrics

$$
\mathrm { K L } [ p _ { M } | | \hat { p } _ { M } ] : = \sum _ { 1 \leq k \leq K } \frac { Z _ { k } } { Z } \cdot \log \frac { Z _ { k } / Z } { \hat { Z } _ { k } / \hat { Z } } , \left( \frac { Z _ { k } } { Z } \cdot \left| \log \hat { Z } _ { k } - \log Z _ { k } \right| \right) , \mathrm { a n d } \left( \left| \log \hat { Z } _ { k } - \log Z _ { k } \right| \right) ,
$$

for $k \in [ K ]$ , which respectively evaluate the accuracy of the learned prior mixing distribution and the relative and absolute errors of the learned partition functions. We separate relative and absolute metrics as, in practice, if a $Z _ { k }$ is negligible when compared to other $Z _ { k ^ { \prime } }$ , small discrepancies in $Z _ { k }$ will result in large deviations for log $\hat { Z } _ { k }$ . As an example, the distributions with logits (1, 2, 10) and (1.05, 1.99, 15) are efectively the same, despite the large diference in the values of the third component. We display these values in Figures 22 and 23, with a comparison between SC and ASC GFlowNets according to them in the latter.

![](images/ca088266883e93b332903bc8d57625d69818c24d172110fc151cbb24357a4233.jpg)  
(a) Rings.

![](images/51f8643eafbda7b43a62ea5a93e2a30b1d0cc7b9e030ef165fea4e52ca1376c5.jpg)  
(b) Four Gaussians.  
Figure 26: SC GFlowNets’ distributional accuracy for distinct numbers of partitions.

We once again consider the problem of choosing K. Figure 26 highlights the dependence of SC GFlowNets’s accuracy on K for the Lazy Random Walk domain. Although SC GFlowNets improve upon a monolithic sampler for all evaluated choices of $K \geq 2$ , there is also a noticeable diference between them. In view of this, we provide two separate suggestions for selecting this hyperparameter. The first is to always let $K = 3$ components, should this choice be compatible with the chosen TC; despite its simplicity, we found setting $K = 3$ to consistently achieve an accurate approximation to the target dis-

tributions we considered. The second is to rely upon standard hyperparameter selection protocols, such as

![](images/f6c93b4daca4740fd2ba02f80c4984a23d2b8893c69d83a03ba9a0d30f57e335.jpg)

![](images/56ef8f37623789afe5b9ab4f012dca48644804adcbb1129d792097c5b515d9f5.jpg)

![](images/9a73bb4f2fbd21293cf20428d96763037263d2c24b61afa740be895924d8c1c7.jpg)  
(a) Log-predictive posterior density of the observed data (LPPD).

![](images/077ab9967f4a951aa72528268105df3fa9bdd71fb4c3c00751f8f0c592bcfef8.jpg)  
(b) Average unnormalized target density of the 10 most likely encountered states.

Figure 25: SC GFlowNets improve learning convergence (left) and accelerate the discovery of highprobabiltiy states (right) for the Bayesian variable selection task; H represents the number of variables (columns) in the dataset.

![](images/efb4ad27a7a8de5236a6283e7288f6db8f092d151d7aa54ef536656703faf03f.jpg)

![](images/d09d90df99633e547fc2e73bfaaac795f6931c1647d52998b92c3a19148c574d.jpg)  
(a) Length-based TC.

![](images/8118044a534cc6b467841e1fade614fc2a5a36cb85719df562189d2bd90303b6.jpg)

![](images/76247798cc0d641d618e4616897e5a60276fc2c248c242f6f37329d5e593d143.jpg)

![](images/155bcf552dc3923dec1db023ec4e9a0d512a3a919ad98e34b8dac2d253ab8b69.jpg)

![](images/9f14970a36d179d465a69641f8174f5facb1f4dadd057544a561ede375a80607.jpg)  
(b) Octant-based TC.

![](images/f501b27b4826e44d9305507c26c0a838cbd08c89be4bcf0760a597946d92c8b0.jpg)  
Figure 27: Component-wise learned distributions for both considered SC GFlowNets in Figure 5.

carrying out short training runs and measuring the sampler’s performance afterwards, to pick K. With this in mind, future research should investigate principled approaches for algorithmic picking of K. Nonetheless, a useful intuition to have is that the complexity of the function learned by our component-wise amortized policy increases with K, due to which we advise against setting too large a value for the number of components in SC GFlowNets.

The main takeaway from these experiments, however, is that designing an efective covering for SC GFlowNets does not require fine-grained prior knowledge about the shape of the target distribution.

Variable Selection. We consider the task of Bayesian variable selection (George & McCulloch, 1993). Given a data set $\mathcal { D } = \{ ( x ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ with $\boldsymbol { x } ^ { ( i ) } \in \mathbb { R } ^ { H }$ and $y ^ { ( i ) } \in \mathbb { R }$ , we let $p ( y | x , S , \beta ) = \mathcal { N } ( y | \beta _ { S } ^ { T } x _ { S } , \sigma ^ { 2 } )$ be defined according to a linear model with observational noise σ and coeficients $\boldsymbol { \beta } \in \mathbb { R } ^ { H }$ and features $S \subseteq [ H ]$ We here denote by $x _ { S }$ the vector $( x ^ { ( i ) } ) _ { i \in S }$ composed of the indices i in S, with a similar definitiong for $\beta _ { S }$ In this setting, we train a sampler to draw from the marginal posterior distribution over S, i.e.,

$$
R ( S ) : = p ( S | \mathcal { D } ) = \int _ { \mathbb { R } ^ { H } } p ( S , \beta | \mathcal { D } ) \mathrm { d } \beta ,
$$

with $p ( S , \beta | \mathcal { D } )$ being the usual Bayesian posterior for S and β. We notice $p ( S | \mathcal { D } )$ can be computed in closed form up to a normalizing constant independent of S for a linear model. After training, we evaluate the samplers according to the predictive posterior distribution of the observed data, as it is usually done in Bayesian modelling, and to the average unnormalized log-posterior of the 10 most likely variable sets, a common approach for assessing exploration efectiveness in the GFlowNet literature, e.g., (Kim et al., 2024, 2023, 2025b). We show both metrics in Figure 25, with the SC GFlowNet being designed as in the Set Generation domain. Notably, our mixture model improves both learning and the rate at which modes are encountered during training.

## G.2 Experimental Details

We provide further details for how each domain was implemented, including model architecture, mixtureencoding details, optimizer, training iterations, and batch size per gradient step. We ran all our experiments

in a MacBook Pro M4 with 16 GB of RAM, and we implemented our models using Jax’s Flax library (Bradbury et al., 2018; Heek et al., 2026). All reported results, including error bars, represent averages and standard deviations computed over three independent runs.

In every experiment, we used the Muon optimizer (Jordan et al., 2024) for the neural network’s parameters, and Adam (Kingma & Ba, 2014) for the log Z and $\{ \log Z _ { k } \} _ { k \in [ K ] }$ parameters. Unless stated otherwise, we parameterized the policies $p _ { F }$ and $p _ { B }$ of the GFlowNet using an MLP with two 64-dimensional layers,

$$
p _ { F } ( s , \cdot ) = \mathrm { S o f t m a x } \left( \mathrm { M L P } ( \iota ( s ) ) \odot \mathbf { m } ( s ) \right) ,
$$

with $\iota ( s ) \colon S \to \mathbb { R } ^ { 6 4 }$ being a learned vector embedding for the sta ${ \mathrm { { , e , } } }$ and $\mathbf { m } ( s ) \in \{ 1 , - \infty \} ^ { f }$ being masking function that ensures the sampler is constrained to the target region (either $\mathcal { X }$ or $\mathcal { X } _ { k }$ , in the case of SC GFlowNets); f is the maximum number of children a given state s has in the underlying state graph. We implement a similar representation for $p _ { B }$ , using a shared encoder and simply changing the output layer and the mask. In its centralized form, SC GFlowNets are implemented as

$$
p _ { F } ^ { ( \gamma ) } ( s , \cdot ) = \mathrm { S o f t m a x } ( \mathrm { M L P } ( \iota ( s ) \oplus \iota _ { \gamma } ( \gamma ) ) \odot \mathbf { m } ( s , \gamma ) ) .\tag{18}
$$

Again, $\mathbf { m } ( s , \gamma )$ is a masking function, which depends on γ in the case of SC GFlowNets but not in the case of RF GFlowNets;  represents concatenation. To ensure matching computational costs and similar parameter counts for both SC GFlowNets and GFlowNets, we let $\iota \colon S  \bar { \mathbb { R } } ^ { 6 4 - \bar { d } _ { \gamma } }$ and $\iota _ { \gamma } \colon \Gamma   { \mathbb { R } } ^ { d _ { \gamma } }$ , with $d _ { \gamma }$ being the dimension of the learned representation for $\gamma$ . For SC GFlowNets, $\iota _ { \gamma }$ is a learned dictionary from $[ K ]$ to $\mathbb { R } ^ { d _ { \gamma } }$ For RF GFlowNets, $\iota _ { \gamma }$ is a linear layer. We set $d _ { \gamma } = 8$ in all experiments, which we found to be consistently efective. ASC GFlowNets, on the other hand, was trained with K identical copies of Equation (18); as each model learns to sample from a distinct $\mathcal { X } _ { k }$ , information about k was not required to be explicitly encoded into the neural network.

Set generation. We considered sets of size 24, 48, 64 , and trained each sampler for 3000, 5000, 10000 iterations, respectively, using a batch size of 64 trajectories.

Hypergrid. We trained the models for $H \in \{ 1 6 , 3 2 \}$ , both of which over 16000 iterations and a batch size of 64 trajectories per step. The reward function is the same as in (Malkin et al., 2022), i.e.,

$$
\log { R ( \mathbf { x } ) } = \log \left( \frac { 1 } { 2 } \prod _ { i = 1 } ^ { d } \mathbb { 1 } ( a _ { i } > 0 . 5 ) + 2 \prod _ { i = 1 } ^ { d } \mathbb { I } ( 0 . 6 < a _ { i } < 0 . 8 ) + \rho \right) ,\tag{19}
$$

with $\rho = 1 0 ^ { - 3 }$ and $a _ { i } = 2 \cdot \mathbf { x } _ { i } \cdot ( H - 1 ) ^ { - 1 } - 1$

Ancestral Graphs. We considered two distinct tasks under the Ancestral Graphs domain. The first, considered also in (da Silva et al., 2026), considered graphs with $H \in \{ 1 5 , 2 0 \}$ nodes and used the BIC for scoring each $\mathrm { A G }$

$$
R ( x ) = - \mathrm { B I C } ( x ) = 2 \log \hat { p } ( \mathcal { D } | x ) - | x | _ { E } \log n - 2 | x | _ { E } \log | x | _ { N } ,\tag{20}
$$

in which $\hat { p }$ is the maximum likelihood of given $x ,$ computed via Drton & Richardson (2004)’s Iteratively Reweighted Least Squares (IRLS) algorithm, $| x | _ { N }$ is the number of nodes (H) in $x , \ | x | _ { E } ,$ , the number of edges, and was drawn from a randomly parameterized linear Gaussian structural equation model with H variables. We used the same data generative process as in (da Silva et al., 2026).

The second task consisted of simulating the Erd˝os Renyi model, whose unnormalized density is defined as

$$
R ( x ) = \alpha ^ { | x _ { E } | } .
$$

We restrict the GFlowNet’s generative process to directed acyclic graphs (DAGs) by masking out bidirected edges. Under this condition, both the partition function $Z _ { H } ( \alpha )$ of $R ( x )$ and the probability that there is an edge between each pair of nodes (the same for all of them, due to symmetry) can be computed in closed form and compared against that learned by our sampler. We notice that these quantities are actually related, and both of them can be computed eficiently by adapting Robinson’s formula (Robinson, 1973), which we quickly review. Fix any pair $( a , b )$ of nodes. Let $\mathcal { X } _ { H }$ be the space of all DAGs with H nodes. The probability that there is an edge from a to b in the graph is

$$
E _ { H } ( \alpha ) = \frac { \sum _ { x \in \mathcal { X } , ( a  b ) \mathrm { ~ i n ~ } x } \alpha ^ { | x _ { E } | } } { Z _ { H } ( \alpha ) } \mathrm { ~ w i t h ~ } Z _ { H } ( \alpha ) = \sum _ { x \in \mathcal { X } _ { H } } \alpha ^ { | x _ { E } | } .
$$

To compute the latter, pick any subset of size d of nodes, which will be deemed roots of the DAG, for $d \in \{ 1 , \ldots , H \}$ ; in a DAG, there is always at least one. There are $\textstyle { \binom { H } { d } }$ ways of choosing them. From them, we may pick each of the $H - d$ remaining nodes to decide whether there is an edge between them and the root nodes, totaling a maximum of $d ( H - d )$ edges. Given this choice, the remaining $H - d$ nodes may form any DAG. In our summation, the corresponding term is

$$
m ( d , \alpha ) = { \binom { H } { d } } \cdot Z _ { H - d } ( \alpha ) \cdot \sum _ { 0 \leq k \leq d ( H - d ) } { \binom { d ( H - d ) } { k } } \alpha ^ { k } = { \binom { H } { d } } \cdot Z _ { H - d } ( \alpha ) \cdot ( 1 + \alpha ) ^ { d ( H - d ) } .
$$

However, summing this expression over $d \in \{ 1 , \ldots , H \}$ over-counts certain graphs; $\mathrm { e . g . }$ , the edgeless graph is counted several times when $k = 0$ for each d. More comprehensively, let $R _ { H , d }$ be the set of all DAGs having exactly d nodes as roots, and let $\begin{array} { r } { c ( { d } , \alpha ) = \sum _ { x \in { R _ { H , d } } } \alpha ^ { | x | _ { E } } ; } \end{array}$ ; our objective is to compute $\begin{array} { r } { \sum _ { 1 \le d \le H } c ( d , \alpha ) } \end{array}$ However, the expression above evaluates

$$
m ( d , \alpha ) = \sum _ { d \leq j \leq H } { \binom { j } { d } } \cdot c ( j , \alpha ) ,
$$

since $R _ { H , j }$ appears $\textstyle { \binom { j } { d } }$ times when evaluating $m ( d , \alpha )$ ; to see this, fix a set $\mathbf { n } = \{ n _ { 1 } , \dots , n _ { j } \}$ of j nodes, and notice that our computation above considers the case in which every $n \in \mathrm { ~ n ~ }$ is root each time we pick any d-sized subset of n, i.e.,  <sup>j</sup> times, for $d \leq j$ , as our computation considers DAGs with at least d roots. We can circumvent this issue by instead computing

$$
\begin{array} { r l } { \displaystyle \sum _ { 1 \leq d \leq H } ( - 1 ) ^ { d + 1 } m ( d , \alpha ) = \displaystyle \sum _ { 1 \leq d \leq H } \displaystyle \sum _ { d \leq j \leq H } ( - 1 ) ^ { d + 1 } \cdot \left( \frac { j } { d } \right) \cdot c ( j , \alpha ) } & { } \\ { = \displaystyle \sum _ { 1 \leq j \leq H } c ( j , \alpha ) \cdot \displaystyle \sum _ { 1 \leq d \leq j } ( - 1 ) ^ { d + 1 } \cdot \left( \frac { j } { d } \right) } & { } \\ { = \displaystyle \sum _ { 1 \leq j \leq H } c ( j , \alpha ) \cdot \left( 1 - \displaystyle \sum _ { 0 \leq d \leq j } ( - 1 ) ^ { d } \cdot \left( \frac { j } { d } \right) \right) } & { } \\ { = \displaystyle \sum _ { 1 \leq j \leq H } c ( j , \alpha ) \left( 1 - ( 1 - 1 ) ^ { j } \right) = \displaystyle \sum _ { 1 \leq j \leq H } c ( j , \alpha ) . } \end{array}
$$

In conclusion,

$$
Z _ { H } ( \alpha ) = \sum _ { 1 \leq d \leq H } ( - 1 ) ^ { d + 1 } \cdot { \binom { H } { d } } \cdot Z _ { H - d } ( \alpha ) \cdot ( 1 + \alpha ) ^ { d ( H - d ) } .
$$

This can be computed recursively via the boundary condition $Z _ { 1 } ( \alpha ) = 1$ . To evaluate $E _ { H } ( \alpha )$ , we notice the probability of existing an edge from a to b is equal, by symmetry, to the expected number of edges divided by the total number of edges. On the other hand, the number of edges follows the exponential family $| x | _ { E } \mapsto \exp \left\{ | x | _ { E } \log \alpha - \log Z _ { H } ( \alpha ) \right\}$ with cumulant log $Z _ { H } ( \alpha )$ ; as a consequence, its expected value can be computed as the derivative of the cumulant,

$$
\mathbb { E } _ { x } [ | x | _ { E } ] = \frac { \mathrm { d } \log Z _ { H } ( \alpha ) } { \mathrm { d } \log \alpha } = \alpha \cdot \frac { \mathrm { d } \log Z _ { H } ( \alpha ) } { \mathrm { d } \alpha } ,
$$

which can be computed eficiently through our previously discussed recursive formula. The probability of there being an edge between any pair of nodes under Erd˝os-Renyi’s model is then

$$
E _ { H } ( \alpha ) = { \frac { 1 } { H ( H - 1 ) } } \cdot \alpha \cdot { \frac { \mathrm { d } \log Z _ { H } ( \alpha ) } { \mathrm { d } \alpha } } .
$$

During training, we compare $E _ { H } ( \alpha )$ with the edge-wise probabilities learned by our sampler, estimated via Monte Carlo with 2048 samples; the average $L _ { 1 }$ between both quantities is the value reported in Figure 6, called Marginal Edge Error and evaluated as

$$
\mathrm { M a r g i n a l E d g e E r r o r } = \frac { 2 } { H ( H - 1 ) } \sum _ { n _ { 1 } , n _ { 2 } \in [ H ] , n _ { 1 } \neq n _ { 2 } } | E _ { H } ( \alpha ) - \hat { \pi } ( n _ { 1 } , n _ { 2 } ) |
$$

with $\hat { \pi } ( n _ { 1 } , n _ { 2 } )$ being the referred Monte Carlo estimate.

Autoregressive sequence generation. We set $p _ { \mathrm { l e n g t h } }$ to an uniform distribution and both $p _ { \mathrm { i n i t } }$ and $p _ { \mathrm { k e r n e l } }$ to randomly sampled distribution with logits drawn according a standard multivariate normal mode of dimension $L \in \{ 1 8 , 2 4 \}$ . We train the samplers for 8000 iterations in both cases.

Lazy Random Walk. We consider the follow distributions for the Rings and Gaussian Mixture targets, respectively,

$$
\log R _ { \mathrm { R i n g s } } ( \mathbf { x } ) = \log \left( \sum _ { k = 1 } ^ { K } w _ { k } \exp \left( - \frac { 1 } { 2 } \left( \frac { \| \mathbf { x } \| _ { 2 } - r _ { k } } { s } \right) ^ { 2 } \right) \right) ,
$$

with $K = 2 , r _ { 1 } = 0 . 2 , r _ { 2 } = 0 . 8 , w _ { 1 } = 0 . 6 , w _ { 2 } = 0 . 4 _ { \cdot }$ , and $s = 0 . 0 5$ , and

$$
\log R _ { \mathrm { G A U s S L A N M I X T U R } } ( \mathbf { x } ) = \left[ \log \left( \sum _ { j = 1 } ^ { 4 } \exp \left( - \frac { 1 } { 2 } \left( \frac { \| \mathbf { x } - \boldsymbol { \mu } _ { j } \| _ { 2 } ^ { 2 } } { \sigma } \right) ^ { 2 } \right) \right) - \log \left( 4 \pi \sigma \sqrt { 2 \pi } \right) \right] ,
$$

with $\sigma = 0 . 1$ and $\mu _ { 1 } = ( - 0 . 8 , - 0 . 8 ) , \mu _ { 2 } = ( - 0 . 8 , 0 . 8 ) , \mu _ { 3 } = ( 0 . 8 , - 0 . 8 )$ , and $\mu _ { 4 } = ( 0 . 8 , 0 . 8 )$ . In both cases, we normalize x to $[ - 1 , 1 ]$ through the transformation $\mathbf { x } \mapsto H ^ { - 1 } \mathbf { x } .$

Variable Selection. We consider $H \in \{ 2 4 , 4 8 \}$ and train both models for 5000 and 10000 iterations, respectively, and we used a batch size of 64 trajectories per training step.

## H Proofs

We provide detailed proofs for each theoretical result in both the main text and the supplement.

## H.1 Proof of Proposition 3.3

As each ${ \mathfrak { g } } _ { \gamma }$ abides by the MTB condition, we observe that

$$
Z ( \mathrm { d } \gamma ) \cdot p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) = R ( x , \mathrm { d } \gamma ) \cdot p _ { B } ^ { ( \gamma ) } ( x , \tau )
$$

for each $\gamma \in \Gamma$ and trajectory τ from $s _ { o }$ to x. We then infer that, by marginalizing $\tau \in s _ { o }  x .$

$$
Z ( \mathrm { d } \gamma ) \sum _ { \tau \in s _ { o } \hookrightarrow x } p _ { F } ( s _ { o } , \tau ) = R ( x , \mathrm { d } \gamma ) \underbrace { \sum _ { \tau \in s _ { o } \hookrightarrow x } p _ { B } ^ { ( \gamma ) } ( x , \tau ) } _ { = 1 } = R ( x , \mathrm { d } \gamma ) .
$$

In addition, the left-hand side of the above equation is exactly $m _ { F } ^ { ( \gamma ) } ( x )$ (by definition). Hence,

$$
Z ( \mathrm { d } \gamma ) m _ { F } ^ { ( \gamma ) } ( x ) = R ( x , \mathrm { d } \gamma ) .
$$

This shows the second equality in Proposition 3.3. By summing the above equation over $x \in \mathcal { X }$ and noticing that $m _ { F } ^ { ( \gamma ) }$ is a probability measure on $x ,$ we obtain

$$
Z ( \mathrm { d } \gamma ) = Z ( \mathrm { d } \gamma ) \underbrace { \sum _ { x \in \mathcal { X } } m _ { F } ^ { ( \gamma ) } } _ { = 1 } ( x ) = \sum _ { x \in \mathcal { X } } R ( x , \mathrm { d } \gamma ) .
$$

This demonstrates the first equality in Proposition 3.3.

## H.2 Proof of Corollary 5.2

Under the notation of Proposition 3.3, we observe that

$$
R _ { k } ( x ) = R ( x , \{ k \} ) = R ( x ) q _ { k } ( x ) = R ( x ) N ( x ) ^ { - 1 }
$$

if $x \in \mathcal { X } _ { k }$ , and $R ( x , \{ k \} ) = 0$ otherwise. This proves the second equality in Corollary 5.2. The first equality can be derived by recalling that $\begin{array} { r } { Z _ { k } : = \sum _ { x \in \mathcal { X } _ { k } } R _ { k } ( x ) } \end{array}$ and hence

$$
Z _ { k } = \sum _ { x \in \mathcal { X } _ { k } } \frac { R ( x ) } { N ( x ) } .
$$

This is exactly the first equality in Corollary 5.2.

## H.3 Proof of Proposition 5.6

We demonstrate there are numbers $\{ a _ { k } , b _ { k } \} _ { k \in [ K ] }$ and a function $\psi$ for which the problem

$$
\exists s ^ { \prime } \in 2 ^ { { \mathcal { C } } \setminus s } { \mathrm { . ~ t . ~ } } s \cup s ^ { \prime } \in { \mathcal { X } } { \mathrm { ~ a n d ~ } } \psi ( s \cup s ^ { \prime } ) \in [ a _ { k } , b _ { k } )
$$

is equivalent to SubsetSum. We assume ${ \mathcal { C } } = [ C ]$ for some $C \in \mathbb { N }$ . We first notice that since $\psi$ is modular, i.e.,

$$
\psi ( s \cup \{ c \} ) = \psi ( s ) + \psi ( \{ c \} )
$$

for each $c \in \mathcal { C } \setminus s .$ , we may write

$$
\psi ( s ) = \psi ( \emptyset ) + \sum _ { c \in s } \psi ( \{ c \} ) .
$$

As such, we define $\eta _ { c } = \psi ( \{ c \} ) \geq 0$ for conciseness and assume $\psi ( \emptyset ) = 0$ without loss of generality. In this case, ψ(s) can be computed in linear time in the size of s. We also assume $\eta _ { c }$ are positive integers and fix $s = \emptyset$ . Under these conditions, our decision problem in Equation (6) becomes

$$
\exists s \in 2 ^ { c } \colon \sum _ { c \in s } \eta _ { c } \in [ a _ { k } , b _ { k } ) .
$$

Let there be a k for which $b _ { k } = a _ { k } + 1$ with both $a _ { k }$ and $b _ { k }$ positive integers; as $\eta _ { c }$ are also integers, the above sum can only be satisfied when

$$
\sum _ { c \in s } \eta _ { c } = a _ { k }
$$

for some subset s of . This is exactly the subset sum problem (with positive inputs), which is NP-complete (Cormen et al., 2009). In particular, deciding upon the reachability of $\mathcal { X } _ { k }$ from a given $s \in S$ is NP-complete in general.

## H.4 Proof of Proposition 5.7

We show the decision problem in Equation (6) can be eficiently solved when $\psi$ is a symmetric junta by relying upon the fact that, since is finite, there are only so many values realizable by $\psi ,$ all of which can be quickly verified in polynomial time.

Fix any $s \in S$ with $n = \# ( s \cap r )$ . We notice $n \leq \# ( \mathcal { C } \cap r )$ . Also, $\psi ( s ) = \xi ( n )$ by definition. Our decision problem then becomes

$$
\exists s ^ { \prime } \in 2 ^ { \mathcal { C } \setminus s } : \psi ( s \cup s ^ { \prime } ) \in [ a _ { k } , b _ { k } ) .
$$

$\operatorname { A s } s$ and $s ^ { \prime }$ are disjoint, $\# ( ( s \cup s ^ { \prime } ) \cap r ) = \# ( s \cap r ) + \# ( s ^ { \prime } \cap r )$ . Similarly, $\# ( s ^ { \prime } \cap r ) \leq \# ( \mathcal { C } \cap r ) - n$ . And not only this, but $\# ( s ^ { \prime } \cap r )$ can also assume any value within $\{ 0 , \ldots , \# ( \mathcal { C } \cap r ) - n \}$ as $s ^ { \prime } \in 2 ^ { \mathcal { C } \backslash s }$ . As such, deciding whether the above problem is solvable is equivalent to deciding whether there is a $m \in \{ 0 , \ldots , \# ( \mathcal { C } \cap r ) - n \}$ for which

$$
\xi ( n + m ) \in [ a _ { k } , b _ { k } )
$$

which can (by definition) be executed eficiently as $\psi$ is assumed to be eficiently evaluable in polynomial time.

To see that the result remains true if we replace $2 ^ { \mathcal { C } }$ by the set of bounded multisubsets of ${ \mathcal { C } } ,$ we define

$$
{ \mathcal { M } } ( B ) = \{ \{ \{ c _ { 1 } , \ldots , c _ { k } \} \} : c _ { 1 } , \ldots , c _ { k } \in { \mathcal { C } } { \mathrm { ~ a n d ~ } } k \leq B \}
$$

as the set of multisubsets of of size up to $B .$ . We emphasize the operator $s \mapsto \#$ s counts repeated entries as distinct in s. We assume B is polynomial on # . In this case, the above argument applies by replacing $2 ^ { \mathcal { C } }$ with $\mathcal { M } ( B )$ and noticing that it is enough to verify whether, given $s \in { \mathcal { C } } .$ , there is a $m \in \{ 0 , \ldots , B - \# ( s \cap r ) \}$ for which $\xi ( \# ( s \cap r ) + m ) \in [ a _ { k } , b _ { k } )$ , which can again be done eficiently.

## H.5 Proof of Proposition 5.9

Let $\{ S _ { t } ^ { ( f ) } \} _ { t \ge 1 }$ be the Markov chain defined by $\kappa _ { F } .$ , and let L be the maximum trajectory length in the GFlowNet’s state graph. For conciseness, we will denote by $\mathbb { P } _ { \kappa _ { F } [ \cdot ] }$ the probability measure corresponding to the case in which the forward Markov chain starts $s _ { o } , \mathrm { i . e . , } \mathbb { P } _ { \kappa _ { F } } [ \cdot | s _ { o } ]$ . It is clear that $T _ { \mathcal { X } } \leq L$ and, hence, $\mathbb { P } _ { \kappa _ { F } } [ T _ { \mathcal { X } } < \infty | s ] = 1$ for any s. We now proceed to show that $\bar { S } _ { T x } ^ { ( f ) }$ is distributed according to the target $R ( x )$ . We start by noticing by

$$
\mathbb { P } _ { \kappa _ { F } } [ S _ { T \mathcal { X } } ^ { ( f ) } = x ] = \sum _ { 1 \leq t \leq L } \mathbb { P } _ { \kappa _ { F } } [ T _ { \mathcal { X } } = t ] \mathbb { P } _ { \kappa _ { F } } \left[ S _ { T _ { \mathcal { X } } } ^ { ( f ) } = x | T _ { \mathcal { X } } = t \right] .
$$

and, by the definition of conditional probabilities,

$$
\mathbb { P } _ { \kappa _ { F } } \left[ S _ { T _ { \mathcal { X } } } ^ { ( f ) } = x | T _ { \mathcal { X } } = t \right] = \frac { \sum _ { ( s _ { 1 } , \ldots , s _ { t - 1 } , x ) \in \bar { S } ^ { t } } \kappa _ { F } \left( s _ { t - 1 } , x \right) \cdot \prod _ { 1 \leq i \leq t - 2 } \kappa _ { F } \left( s _ { i } , s _ { i + 1 } \right) } { \sum _ { x ^ { \prime } \in \mathcal { X } } \sum _ { ( s _ { 1 } , \ldots , s _ { t - 1 } , x ^ { \prime } ) \in \bar { S } ^ { t } } \kappa _ { F } \left( s _ { t - 1 } , x ^ { \prime } \right) \cdot \prod _ { 1 \leq i \leq t - 2 } \kappa _ { F } \left( s _ { i } , s _ { i + 1 } \right) } .
$$

This equation follows from the definition of $T _ { \mathcal { X } }$ and the Markov property. In addition, if there is a $\left( { { s _ { i } } , { s _ { i + 1 } } } \right)$ for which the transition $s _ { i } ~  ~ s _ { i + 1 }$ is not in the underlying GFlowNet’s state graph, the probability of $( s _ { o } , \ldots , s _ { t - 1 } , x )$ is zero by design (as $\kappa _ { F }$ matches $p _ { F } )$ ; hence, the above ratio can be written as

$$
\frac { \sum _ { \tau \in s _ { o } \sim x , | \tau | = t } p _ { F } ( s _ { o } , \tau ) } { \sum _ { x ^ { \prime } \in \mathcal { X } } \sum _ { \tau ^ { \prime } \in s _ { o } \sim x ^ { \prime } , | \tau ^ { \prime } | = t } p _ { F } ( s _ { o } , \tau ^ { \prime } ) } ,
$$

with $| \tau | = t$ stating that the length (number of states) of the trajectory τ is t. A similar reasoning shows that

$$
\mathbb { P } _ { \kappa _ { F } } [ T _ { \mathcal { X } } = t ] = \sum _ { x ^ { \prime } \in \mathcal { X } } \sum _ { \tau \in s _ { o } \sim x ^ { \prime } , | \tau | = t } p _ { F } ( s _ { o } , \tau ) .
$$

Thus,

$$
\mathbb { P } _ { \kappa _ { F } } [ S _ { T _ { \mathcal { X } } } ^ { ( f ) } = x ] = \sum _ { 1 \le t \le L } \sum _ { \tau \in s _ { o } \hookrightarrow x , | \tau | = t } p _ { F } ( s _ { o } , \tau ) = \sum _ { \tau \in s _ { o } \hookrightarrow x } p _ { F } ( s _ { o } , \tau ) = m _ { F } ( x ) .
$$

When the GFlowNet abides a balance condition, this implies

$$
\mathbb { P } _ { \kappa _ { F } } \left[ S _ { T _ { \mathcal { X } } } ^ { ( f ) } = x \right] = \frac { R ( x ) } { \sum _ { x \in \mathcal { X } } R ( x ) } ,
$$

as stated by Proposition 5.9.

Correspondingly, we observe that

$$
\mathbb { P } _ { \kappa _ { B } } \left[ T _ { \{ s _ { o } \} } < \infty | x \right] = 1
$$

since each trajectory starting at x must eventually reach $s _ { o } .$ , and the maximum distance between $s _ { o }$ and any $x \in \mathcal { X }$ is L. This shows the second part of Proposition 5.9.

## H.6 Proof of Lemma 5.10

We show the function $h _ { k }$ defined as

$$
h _ { k } ( s ) = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } = \operatorname* { m i n } _ { k ^ { \prime } } T _ { k ^ { \prime } } | s \right]
$$

for $s \in S$ is harmonic. For this, we observe that

$$
\begin{array} { l } { \mathbb { E } _ { s ^ { \prime } \sim \kappa _ { F } ( s , \cdot ) } \left[ h _ { k } ( s ^ { \prime } ) \right] = \displaystyle \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } } \kappa _ { F } ( s , s ^ { \prime } ) \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } = \operatorname* { m i n } _ { k ^ { \prime } } T _ { k ^ { \prime } } | s ^ { \prime } \right] } \\ { = \displaystyle \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } } \kappa _ { F } ( s , s ^ { \prime } ) \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } < \infty | s ^ { \prime } \right] } \\ { = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } < \infty | s \right] = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } = \displaystyle \operatorname* { m i n } _ { k ^ { \prime } } T _ { k ^ { \prime } } | s \right] = h _ { k } ( s ) . } \end{array}
$$

The first equation follows from definition. The second and fourth equations are a consequence of the fact that $\mathcal { X } _ { k }$ is an absorbing set of the Markov chain dictated by $\kappa _ { F }$ and hence that $\mathcal { X } _ { k }$ being arrived at before $\mathcal { X } _ { k ^ { \prime } }$ for $k \neq k ^ { \prime }$ implies that $T _ { k }$ is the only finite hitting time among $\{ T _ { k } \} _ { k \in [ K ] }$ . The third equation follows from the law of total probability, and the fifth equation follows from the definition of $h _ { k }$ . This demonstrates $h _ { k }$ is an harmonic function for $k \in [ K ]$

The fact that $h _ { k } ( x ) = 1$ for $x \in \mathcal { X } _ { k }$ and $h _ { k } ( x ) = 0$ for $x \in \mathcal { X } \backslash \mathcal { X } _ { k }$ follows from the definition of $h _ { k }$ as the probability of the Markov process reaching at $\mathcal { X } _ { k }$ before any $\mathcal { X } _ { k ^ { \prime } }$ for $\boldsymbol { k } ^ { \prime } \neq \boldsymbol { k }$

Similarly, $h _ { k } ( s _ { o } )$ is the probability that $\kappa _ { F }$ arrives at $\mathcal { X } _ { k }$ before arriving at any other $\mathcal { X } _ { k ^ { \prime } }$ for $k ^ { \prime } \neq k ;$ as the underlying GFlowNet is assumed to abide by any of its balance conditions, the probability of the former is proportional to $\begin{array} { r } { Z _ { k } = \sum _ { x \in \mathcal { X } _ { k } } R ( x ) } \end{array}$ , and of the latter, proportional to $\begin{array} { r } { Z = \sum _ { x \in \mathcal { X } } R ( x ) } \end{array}$ . This shows

$$
h _ { k } ( s _ { o } ) = { \frac { \sum _ { x \in \mathcal { X } _ { k } } R ( x ) } { \sum _ { x \in \mathcal { X } } R ( x ) } } = { \frac { Z _ { k } } { Z } } .
$$

We also provide a direct derivation of this fact in Section $\mathrm { H . 7 } .$

## H.7 Proof of Proposition 5.11

We show that the marginal distribution of the transformed kernel $\kappa _ { F } ^ { ( k ) }$ , as defined in Theorem 5.9, matches the restriction of $R ( x )$ to $\mathcal { X } _ { k }$ up to a normalizing constant.

As before, we denote by $\mathbb { P } _ { \kappa _ { F } } [ \cdot ]$ the probability measure $\mathbb { P } _ { \kappa _ { F } } [ \cdot | s _ { o } ]$ corresponding to the scenario in which the forward Markov chain starts at $s _ { o }$ . We first notice that, since $h _ { k }$ is harmonic,

$$
\sum _ { s ^ { \prime } \in \bar { \mathcal { S } } } \kappa _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) = \sum _ { s ^ { \prime } \in \bar { \mathcal { S } } } \kappa _ { F } ( s , s ^ { \prime } ) \cdot \frac { h _ { k } ( s ^ { \prime } ) } { h _ { k } ( s ) } = \frac { h _ { k } ( s ) } { h _ { k } ( s ) } = 1 .
$$

We then demonstrate that, under $\kappa _ { F } ^ { ( k ) }$ , the probability of any path that arrives at ${ \boldsymbol { \mathcal { X } } } _ { m }$ before $\mathcal { X } _ { k }$ for a $m \neq k$ is zero. To see this, let $( s _ { o } , s _ { 1 } \dots , s _ { t - 1 } , x )$ for $x \notin \mathcal { X } _ { k } ;$ then, $h _ { k } ( x ) = 0$ and there is a smallest index $i \geq 1$ for which $h _ { k } ( s _ { i } ) = 0$ . (By the definition of $\kappa _ { F } , h _ { k } ( s _ { o } ) > 0$ for each $k \in [ K ] )$ . The transition from $s _ { i - 1 }$ to $s _ { i }$ under $\kappa _ { F } ^ { ( k ) }$ thus has probability zero. This shows that, under $\kappa _ { F } ^ { ( k ) }$ , the Markov chain arrives at $\mathcal { X } _ { k }$ before any ${ \boldsymbol { \mathcal { X } } } _ { m }$ for m $\neq k$ with probability one. In particular, dropping the superscript (k) from $T _ { m }$ for clarity,

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ { T } _ { k } < \infty = T _ { m } \right] = 1 .
$$

This is Proposition 5.11’s first statement.

To show that

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } } ^ { ( k ) } = x \right] = \frac { R ( x ) } { \sum _ { x \in \mathcal { X } _ { k } } R ( x ) } ,
$$

we proceed as in the demonstration of Proposition 5.9. Since

$$
\begin{array} { r l } & { \mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } } ^ { ( k ) } = x \right] = \displaystyle \sum _ { 1 \leq t \leq L } \mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } [ T _ { k } = t ] \mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } } ^ { ( k ) } = x | T _ { k } = t \right] } \\ & { \quad \quad \quad = \displaystyle \sum _ { 1 \leq t \leq L } \mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ T _ { k } = t \right] \frac { \sum _ { \tau \in s _ { o } \sim x , | \tau | = t } \kappa _ { F } ^ { ( k ) } \left( s _ { o } , \tau \right) } { \sum _ { x ^ { \prime } \in \mathcal { X } _ { k } } \sum _ { \tau ^ { \prime } \in s _ { o } \sim x ^ { \prime } , | \tau ^ { \prime } | = t } \kappa _ { F } ^ { ( k ) } \left( s _ { o } , \tau ^ { \prime } \right) } , } \end{array}
$$

in which we define $\begin{array} { r } { \kappa _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = \prod _ { ( s , s ^ { \prime } ) \in \tau } \kappa _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) } \end{array}$ , and $h _ { k } ( x ) = 1$ and $h _ { k } ( s _ { o } ) > 0$ for each $k \in [ K ]$ and $x \in \mathcal { X } _ { k }$ , a telescoping product shows that

$$
\prod _ { ( s , s ^ { \prime } ) \in \tau } \kappa _ { F } ^ { ( k ) } ( s , s ^ { \prime } ) = \frac { 1 } { h _ { k } \left( s _ { o } \right) } \prod _ { ( s , s ^ { \prime } ) \in \tau } \kappa _ { F } ( s , s ^ { \prime } ) = \frac { 1 } { h _ { k } ( s _ { o } ) } \cdot \kappa _ { F } ( s _ { o } , \tau ) = \frac { 1 } { h _ { k } ( s _ { o } ) } \cdot p _ { F } ( s _ { o } , \tau )
$$

for any $\tau \in s _ { o }  x$ with $x \in \mathcal { X } _ { k }$ . As a consequence,

$$
\frac { \sum _ { \tau \in s _ { o } \sim x , | \tau | = t } \kappa _ { F } ^ { ( k ) } ( s _ { o } , \tau ) } { \sum _ { x ^ { \prime } \in \mathcal { X } _ { k } } \sum _ { \tau ^ { \prime } \in s _ { o } \sim x ^ { \prime } , | \tau ^ { \prime } | = t } \kappa _ { F } ^ { ( k ) } ( s _ { o } , \tau ^ { \prime } ) } = \frac { \sum _ { \tau \in s _ { o } \sim x , | \tau | = t } p _ { F } ( s _ { o } , \tau ) } { \sum _ { x ^ { \prime } \in \mathcal { X } _ { k } } \sum _ { \tau ^ { \prime } \in s _ { o } \sim x ^ { \prime } , | \tau ^ { \prime } | = t } p _ { F } ( s _ { o } , \tau ^ { \prime } ) } .
$$

The rest proceeds as in Proposition 5.9. In fact, as

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ T _ { k } = t \right] = \sum _ { x \in \mathcal { X } _ { k } } \sum _ { \tau \in s _ { o } \sim x } \kappa _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = \frac { 1 } { h _ { k } ( s _ { o } ) } \sum _ { x \in \mathcal { X } _ { k } } \sum _ { \tau \in s _ { o } \sim x } p _ { F } ( s _ { o } , \tau )
$$

and

$$
h _ { k } ( s _ { o } ) = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } < \infty | s _ { o } \right] = \sum _ { x \in \mathcal { X } _ { x } } \sum _ { \tau \in s _ { o } \sim x } p _ { F } ( s _ { o } , \tau ) ,
$$

we deduce that

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } } ^ { ( k ) } = x \right] = \frac { \sum _ { 1 \leq t \leq L } \sum _ { \tau \in s _ { o } \sim x , | \tau | = t } p _ { F } \left( s _ { o } , \tau \right) } { \sum _ { x \in \mathcal { X } _ { x } } \sum _ { \tau \in s _ { o } \sim x } p _ { F } \left( s _ { o } , \tau \right) } .
$$

When the GFlowNet abides by a balance condition, this expression can be rewritten as

$$
\mathbb { P } _ { \kappa _ { F } ^ { ( k ) } } \left[ S _ { T _ { k } } ^ { ( k ) } = x \right] = \frac { R ( x ) } { \sum _ { x \in \mathcal { X } _ { k } } R ( x ) } .
$$

This demonstrates the second statement in Proposition 5.11.

## H.8 Proof of Proposition B.1

We show that, if our mixture model satisfies the defined SubTB condition, then it samples from the target distribution. We then show write $F ( s , \mathrm { d } \gamma )$ in terms of the well-known quantities $Z ( \mathrm { d } \gamma )$ and $R ( x , \mathrm { d } \gamma )$

The first statement can be derived directly from the fact that, since the condition is satisfied for all trajectories τ between reachable pairs $s , s ^ { \prime } ,$ it is in particular satisfied by complete trajectories going from $s _ { o }$ to $x \in \mathcal { X }$ . This implies the trajectory balance is satisfied and, by (Malkin et al., 2022, Theorem 1), the marginal distribution of $p _ { M } ( \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) }$ over x matches $R ( x , \mathrm { d } \gamma )$ . By integrating γ out, we conclude that Equation (2) is satisfied.

To see that $\begin{array} { r } { Z ( \mathrm { d } \gamma ) : = \sum _ { x \in \mathcal { X } } R ( x , \mathrm { d } \gamma ) } \end{array}$ , we notice that

$$
F ( s _ { o } , \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) = R ( x , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( x , \tau )
$$

for all $x \in \mathcal { X }$ and all trajectories $\tau \in s _ { o }  x .$ . In particular, marginalizing out both τ and x (in sequence), we infer (as in Proposition 3.3) that

$$
F ( s _ { o } , \mathrm { d } \gamma ) = \sum _ { x \in \mathcal { X } } R ( x , \mathrm { d } \gamma ) : = Z ( \mathrm { d } \gamma ) .
$$

In a similar way, as $F ( s , \mathrm { d } \gamma ) p _ { F } ^ { ( \gamma ) } ( s , \tau ) = R ( x , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( x , \tau )$ , we directly infer that

$$
F ( s , \mathrm { d } \gamma ) = \sum _ { x \in \mathcal { X } } \sum _ { \tau \in s \sim x } R ( x , \mathrm { d } \gamma ) p _ { B } ^ { ( \gamma ) } ( x , \tau )
$$

with the convention that $\begin{array} { r } { \sum _ { \tau \in \varnothing } \eta _ { \tau } \ = \ 0 } \end{array}$ for any trajectory-indexed real sequence $\{ \eta _ { \tau } \} _ { \tau }$ . This is exactly Proposition B.1’s second statement.

## H.9 Proof of Corollary C.2

We derive the relationship between the prior and posterior mixing distributions when they both satisfy the SubTB condition. It is clear that, since $p _ { M } , q _ { M } , p _ { F } ^ { ( \gamma ) }$ , and $p _ { B } ^ { ( \gamma ) }$ abide by SubTB, they also abide by TB. In particular,

$$
p _ { M } ( \gamma ) p _ { F } ^ { ( \gamma ) } ( s _ { o } , \tau ) = \frac { R ( x ) } { Z } q _ { M } ( \gamma | x ) p _ { B } ^ { ( \gamma ) } ( x , \tau ) .
$$

As in Proposition 3.3, we marginalize over $\tau \in s _ { o }  x$ to infer that

$$
p _ { M } ( \gamma ) \cdot m _ { F } ^ { ( \gamma ) } ( x ) = \frac { R ( x ) } { Z } \cdot q ( \gamma | x ) ;
$$

we then derive the second equation from Corollary C.2 by isolating $q ( \gamma | x )$ . Similarly, the first equation can be obtained by marginalizing over $x \in \mathcal { X }$ and recalling $m _ { F } ^ { ( \gamma ) } ( x )$ is a probability distribution over  , i.e.,

$$
p _ { M } ( \gamma ) = p _ { M } ( \gamma ) \sum _ { x \in \mathcal { X } } m _ { F } ^ { ( \gamma ) } ( x ) = \sum _ { x \in \mathcal { X } } \frac { R ( x ) } { Z } q _ { M } ( \gamma | x ) .
$$

## H.10 Proof of Proposition C.5

Our proof has two steps. We first show that, if a model achieves an error of at most ϵ in terms of TV distance to the true policy (which is unique since we are fixing the backward policy for each γ), then the TV between the marginal distribution over is bounded as a monotonically decreasing and continuous function of ϵ. Then, using Abboud et al. (2021, Theorem 1), which characterizes the universality of GNNs with random features, and the intermediate value theorem for the prior bound, we infer that our sampler can approximate any distribution with high-probability.

To start with, we temporarily simplify our notation of $p _ { F } ^ { ( \gamma ) }$ as $p ,$ ommiting the dependence on $\gamma .$ We also refer to the true solution to the chosen balance equations as $p ^ { \star }$ . Our GNN approximates the distribution $p ( s , \cdot )$ ; however, we are interested in the diference

$$
 \sum _ { \tau \in s _ { o }  x } \prod _ { ( s , s ^ { \prime } ) \in \tau } p ( s , s ^ { \prime } ) - \prod _ { ( s , s ^ { \prime } ) \in \tau } p ^ { \star } ( s , s ^ { \prime } ) 
$$

for each $x \in \mathcal { X }$ . We start with the following lemma.

Lemma H.1. Let $a _ { 1 } , \ldots , a _ { n }$ and $b _ { 1 } , \ldots , b _ { n }$ be sequences in [0, 1] such that $| a _ { i } - b _ { i } | \leq \delta$ for some $\epsilon > 0$ Then,

$$
\left| \prod _ { 1 \leq i \leq n } a _ { i } - \prod _ { 1 \leq i \leq n } b _ { i } \right| \leq n \epsilon .
$$

Proof. We proceed by induction. This clearly holds for $n = 1$ . To build intuition, we also show for $N = 2$ Notice that

$$
\left| a _ { 1 } a _ { 2 } - b _ { 1 } b _ { 2 } \right| = \left| ( a _ { 1 } - b _ { 1 } ) a _ { 2 } + b _ { 1 } ( a _ { 2 } - b _ { 2 } ) \right| \leq a _ { 2 } \left| a _ { 1 } - b _ { 1 } \right| + b _ { 1 } \left| a _ { 2 } - b _ { 2 } \right| \leq 2 \epsilon
$$

since $a _ { 1 } , a _ { 2 } , b _ { 1 } , b _ { 2 } \in [ 0 , 1 ]$ . Assume the identity holds for $n .$ . Then,

$$
\left| \prod _ { 1 \leq i \leq n + 1 } a _ { i } - \prod _ { 1 \leq i \leq n + 1 } b _ { i } \right| = \left| ( a _ { n + 1 } - b _ { n + 1 } ) \prod _ { 1 \leq i \leq n } a _ { i } + b _ { n + 1 } \cdot \left( \prod _ { 1 \leq i \leq n } a _ { i } - \prod _ { 1 \leq i \leq n } b _ { i } \right) \right|
$$

by the induction hypothesis. This shows our desired bound.

As there is a realization of our GNN for which TV $( p ( s , \cdot ) , p ^ { \star } ( s , \cdot ) ) \le \epsilon$ for any $s \in { \mathcal { S } }$ , we infer that $| p ( s , s ^ { \prime } ) - p ^ { \star } ( s , s ^ { \prime } ) | \le 2 \epsilon$ . Under this condition, we let $T$ be the maximum number of trajectories from $s _ { o }$ to x for $x \in \mathcal { X }$ and L to be the largest trajectory length from $s _ { o }$ to $x \in \mathcal { X }$ . By Abboud et al. (2021, Theorem 1), there are $h \in \mathcal { H } _ { t }$ and $\xi \in \mathcal { H } _ { s }$ for which $p ( s , \cdot )$ approximates $p ^ { \star } ( s , \cdot )$ with error at most $\begin{array} { r } { \epsilon ^ { \prime } = \frac { \epsilon } { 2 | \mathcal { X } | T L } } \end{array}$ and with probability $\mathrm { ~ 1 ~ - ~ } \frac { \delta } { | \mathcal { X } | }$ over the realization of $\gamma$ . The above diference in marginals can then be bounded as, with probability $1 - \frac { \delta } { | \mathcal { X } | }$ over the realization of $\gamma _ { \mathrm { { i } } }$

$$
\begin{array} { r l } & { \left| \left( \displaystyle \sum _ { \tau \in s _ { o } \to x } \prod _ { ( s , s ^ { \prime } ) \in \tau } p ( s , s ^ { \prime } ) - \prod _ { ( s , s ^ { \prime } ) \in \tau } p ^ { \star } ( s , s ^ { \prime } ) \right) \right| } \\ & { \le \left( \displaystyle \sum _ { \tau \in s _ { o } \to x } \left| \prod _ { ( s , s ^ { \prime } ) \in \tau } p ( s , s ^ { \prime } ) - \prod _ { ( s , s ^ { \prime } ) \in \tau } p ^ { \star } ( s , s ^ { \prime } ) \right| \right) } \\ & { \le \left( 1 - \displaystyle \frac { \delta } { T } \right) ( 2 T L ) \frac { \epsilon } { 2 | \mathcal { X } | T L } = \frac { \epsilon } { | \mathcal { X } | } . } \end{array}
$$

In other words, denoting by $m _ { F } ^ { ( \gamma ) }$ and by $\pi ( x )$ the estimated and true marginals for the learned and target models,

$$
| m _ { F } ^ { ( \gamma ) } ( x ) - \pi ( x ) | \leq \frac { \epsilon } { | \mathcal { X } | }
$$

with probability $\begin{array} { r } { 1 - \frac { \delta } { | \mathcal { X } | } } \end{array}$ over $\gamma$ . Through an union bound<sup>5</sup> for $x \in \mathcal { X }$ , we infer that

$$
\mathrm { T V } \left( m _ { F } ^ { ( \gamma ) } , \pi \right) = \frac { 1 } { 2 } \sum _ { x \in \mathcal { X } } \left| m _ { F } ^ { ( \gamma ) } ( x ) - \pi ( x ) \right| \leq \frac { \epsilon } { 2 }
$$

with probability $1 - \delta$ over the sampling of $\gamma$ . This demonstrates that RF GFlowNets are universal approximators of distributions over graphs when using the kind of edge-additive process we described in Proposition C.5.

## H.11 Proof of Lemma C.6

We first expand $\mathcal { L } _ { \mathrm { T B } }$ around $\gamma = 0$ and then take the expectation of the corresponding expansion over $\gamma \sim p _ { M }$ . To start with, let $L ( \gamma ) : = \mathcal L ( \tau , \gamma ; { \mathfrak { g } } _ { \Gamma } )$ ; in this case,

$$
L ( \gamma ) = L ( 0 ) + \gamma ^ { T } \nabla _ { \gamma = 0 } L ( \gamma ) + \frac { 1 } { 2 } \gamma ^ { T } H _ { \gamma } ( \tau , 0 ) \gamma + \mathcal { O } ( \sigma ^ { 3 } ) .
$$

Since, for any matrix A,

$$
\mathbb { E } _ { \gamma \sim p _ { M } } \left[ \gamma \right] = 0 \mathrm { ~ a n d ~ } \mathbb { E } _ { \gamma \sim p _ { M } } \left[ \gamma ^ { T } A \gamma \right] = \sigma ^ { 2 } \cdot \operatorname { t r } ( A )
$$

as $\gamma ^ { T } A \gamma = \operatorname { t r } ( \gamma ^ { T } A \gamma ) = \operatorname { t r } ( A \gamma \gamma ^ { T } )$ for any matrix A and tr is a linear operator, and in general any odd moment of $\gamma$ has nulls out, we infer that

$$
\mathbb { E } _ { \gamma \sim p _ { M } } \left[ L ( \gamma ) \right] = L ( 0 ) + \frac { 1 } { 2 } \sigma ^ { 2 } \cdot \mathrm { t r } \left( H _ { \gamma } ( \tau , 0 ) \right) + \mathcal { O } ( \sigma ^ { 4 } ) ,
$$

which is the statement of Lemma C.6 after replacing L with $\mathcal { L } _ { \mathrm { T B } }$ . This characterizes the local behavior of the (population) loss in the learning objective of CI GFlowNets.

## H.12 Proof of Proposition C.7

We derive from Lemma C.6 that

$$
\mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { \Gamma } ) = \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { o } ) + \frac { 1 } { 2 } \cdot \sigma ^ { 2 } \cdot \mathbb { E } _ { \tau \sim \rho } \left[ \mathrm { t r } \left( H _ { \gamma } ( \tau , 0 ) \right) \right] + \mathcal { O } ( \sigma ^ { 4 } ) = \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { o } ) + \frac { 1 } { 2 } \cdot \sigma ^ { 2 } \cdot \mathcal { L } _ { \mathrm { r e g } } ( \mathfrak { g } _ { \Gamma } ) + \mathcal { O } ( \sigma ^ { 4 } ) .
$$

As taking the derivative is a linear operation, $f \in { \mathcal { O } } ( \sigma ^ { 4 } )$ implies $\nabla _ { \theta } ^ { 2 } f \in { \mathcal { O } } ( \sigma ^ { 4 } )$ for any twice diferentiable function $f \colon \Theta  \mathbb { R }$ on a diferentiable manifold Θ. In particular, we infer that, by evaluating the Hessian from both members of the above equation with respect to the parameters θ of the underlying neural network,

$$
H _ { \mathrm { T B } } = H _ { o } + \frac { \sigma ^ { 2 } } { 2 } \cdot H _ { \mathrm { r e g } } + \mathcal { O } ( \sigma ^ { 4 } ) ,
$$

with $H _ { \mathrm { T B } } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { \mathrm { T B } } ( \mathfrak { g } _ { \Gamma } ) , H _ { o } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } _ { o } ( \mathfrak { g } _ { \Gamma } )$ , and $H _ { \mathrm { r e g } } = \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { \mathrm { r e g } } ( \mathfrak { g } _ { \Gamma } )$

This in particular shows that the random features act as a form of spectral regularization, shifting the eigenvalues of $\mathrm { H } _ { o }$ by $\frac { \sigma ^ { 2 } } { 2 } \cdot \mathrm { e i g } ( H _ { \mathrm { r e g } } )$ , with $\mathrm { e i g } ( H _ { \mathrm { r e g } } )$ being the vector of eigenvalues of $H _ { \mathrm { r e g } }$

<sup>5</sup>We briefly recall that, if $P _ { 1 } , \ldots , P _ { n }$ are events with probability at least $1 - \delta ,$ , then $\cap _ { i } P _ { i }$ is an event with probability at least

$$
\mathbb { P } [ \cap _ { i } P _ { i } ] = 1 - \mathbb { P } [ \cup _ { i } P _ { i } ^ { c } ] \geq 1 - \sum _ { i } \mathbb { P } [ P _ { i } ^ { c } ] \geq 1 - n \delta .
$$

## H.13 Proof of Lemma E.2

We show both functions $h _ { k }$ and $g _ { k }$ are harmonic on $\bar { S } \setminus S _ { k }$ . Let $s \in \bar { \mathcal { S } } \setminus \mathcal { S } _ { k }$ . Then,

$$
\begin{array} { r l } & { \mathbb { E } _ { s ^ { \prime } \sim \kappa _ { F } ( s , \cdot ) } \left[ h _ { k } ( s ^ { \prime } ) \right] = \displaystyle \sum _ { s ^ { \prime } \in \bar { \mathcal { S } } } \kappa _ { F } \big ( s , s ^ { \prime } \big ) \cdot \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } = \operatorname* { m i n } _ { k ^ { \prime } } T _ { k ^ { \prime } } ^ { ( f ) } < \infty \vert s ^ { \prime } \right] } \\ & { \qquad = \displaystyle \sum _ { s ^ { \prime } \in \bar { \mathcal { S } } } \kappa _ { F } \big ( s , s ^ { \prime } \big ) \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } < \infty \vert s ^ { \prime } \right] } \\ & { \qquad = \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } < \infty \vert s \right] = h _ { k } ( s ) . } \end{array}
$$

The first equation follows from the definition of $h _ { k } ;$ the second and third equations follow from the fact that the distance from $s \in S _ { k }$ to $s _ { o }$ in the FHC in Definition E.1 is constant for $k \in [ K ]$ and, hence, if the Markov chains goes through $\scriptstyle { S _ { k } }$ it will not go through any $\boldsymbol { S } _ { k ^ { \prime } }$ for $k ^ { \prime } \neq k , . \mathrm { ~ A ~ }$ similar argument can be carried out for $g _ { k } ( s )$ and $T _ { k } ^ { ( b ) }$

## H.14 Proof of Lemma E.3

We show that $h _ { k } ( s _ { o } )$ and $g _ { k } ( x )$ can be interpreted as probability distributions over $k \in \{ 0 , \ldots , K \}$ . The intuition is that the Markov chain has to either hit one of $S _ { k }$ when starting at $s _ { o } ,$ or end up in ${ \mathcal { X } } _ { o } ,$ in which case none of $\scriptstyle { S _ { k } }$ are hit for $k \in [ K ] ;$ ; similarly, ${ \mathrm { i f ~ } } x \in { \mathcal { X } } _ { o }$ , the Markov chain following $\kappa _ { B }$ and starting at x is absorbed by $s _ { o }$ without going through any of the $\boldsymbol { S _ { k } }$ for $k \in [ K ]$

We start with $h _ { k } ( s _ { o } )$ . By Definition $\mathrm { E . 1 }$ , the events

$$
\left\{ T _ { 1 } ^ { ( f ) } < \infty , S _ { 1 } ^ { ( f ) } = s _ { o } \right\} , \ldots , \left\{ T _ { K } ^ { ( f ) } < \infty , S _ { 1 } ^ { ( f ) } = s _ { 1 } \right\} , \left\{ T _ { k } ^ { ( f ) } = \infty \mathrm { ~ f o r ~ } k \in [ K ] , S _ { 1 } ^ { ( f ) } = s _ { o } \right\}
$$

are mutually joint. In addition, $T _ { k } ^ { ( f ) } < \infty$ implies $T _ { k ^ { \prime } } ^ { ( f ) } = \infty$ almost surely for $\boldsymbol { k } ^ { \prime } \ne k$ . As a consequence, these events also cover the space of possibilities. Hence, by the law of total probability,

$$
\left( \sum _ { 1 \leq k \leq K } \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } < \infty | S _ { 1 } ^ { ( f ) } = s _ { o } \right] \right) + \mathbb { P } _ { \kappa _ { F } } \left[ T _ { k } ^ { ( f ) } = \infty \mathrm { ~ f o r ~ } k \in [ K ] | S _ { 1 } ^ { ( f ) } = s _ { o } \right] = 1 .
$$

This sum is exactly $\begin{array} { r } { h _ { o } ( s _ { o } ) + \sum _ { k \in \left[ K \right] } h _ { k } ( s _ { o } ) } \end{array}$ , which demonstrates the first statement of Lemma E.3.

We proceed similarly for $g _ { k }$ . As noticed, if $x \in \mathcal { X } _ { o }$ , then $T _ { k } ^ { ( b ) } = \infty$ for $k \in [ K ]$ . Otherwise, as $s _ { o }$ is the only absorbing state of a Markov chain dictated by $\kappa _ { B }$ in $\bar { \boldsymbol { S } }$ has to eventually go through some of the $\boldsymbol { S _ { k } }$ when starting at x. This means that the events

$$
\{ T _ { 1 } ^ { ( b ) } < \infty \} , \ldots , \{ T _ { K } ^ { ( b ) } < \infty \} , \{ T _ { k } = \infty \mathrm { f o r } k \in [ K ] \}
$$

are mutually disjoint when the Markov chain starts at $x \in \mathcal { X }$ and cover the space of possibilities. As with $h _ { k }$ , this implies

$$
g _ { o } ( x ) + \sum _ { 1 \leq k \leq K } g _ { k } ( x ) = 1 ,
$$

which is the second statement in Lemma E.3.

## H.15 Proof of Proposition E.4

We show that, if the policies $p _ { F } ^ { ( k ) }$ and $p _ { B } ^ { ( k ) }$ satisfy the ATB condition from Silva et al. (2025b), then the policies $\tilde { p } _ { F } ^ { ( k ) }$ and $\tilde { p } _ { B } ^ { ( k ) }$ satisfy the MTB condition in Definition 3.2.

We first notice that, for any trajectory τ from $s _ { o }$ to $x \in \mathcal { X } _ { o }$

$$
\tilde { p } _ { F } ^ { ( o ) } ( s _ { o } , \tau ) = \frac { h _ { o } ( x ) } { h _ { o } ( s _ { o } ) } p _ { F } ^ { ( o ) } ( s _ { o } , \tau ) = \frac { 1 } { h _ { o } ( s _ { o } ) } \cdot p _ { F } ^ { ( o ) } ( s _ { o } , \tau ) ,
$$

a fact that can be derived from a telescoping product and $h _ { o } ( x ) = 1$ since for a Markov chain starting at $x \in \mathcal { X } _ { o }$ and following $\kappa _ { F }$ it is true that $T _ { k } ^ { ( f ) } = \infty$ (as none of $\boldsymbol { S _ { k } }$ is reachable from $x \in \mathcal { X } _ { o }$ , which is an absorbing state). By the ATB,

$$
p _ { F } ^ { ( o ) } ( s _ { o } , \tau ) = \frac { R ( x ) } { Z } p _ { B } ^ { ( o ) } ( x , \tau ) = \frac { R ( x ) } { Z } \tilde { p } _ { B } ^ { ( o ) } ( x , \tau ) ,
$$

as $p _ { B } ^ { ( o ) } ( s , s ^ { \prime } ) = \tilde { p } _ { B } ^ { ( o ) } ( s , s ^ { \prime } )$ for states $s , s ^ { \prime }$ within a distance of d of $s _ { o } ;$ since $p _ { B }$ does not increase the distance to $s _ { o } ,$ and $x \in \mathcal { X } _ { o }$ is by definition within a distance d of ${ \mathit { s } } _ { o } ,$ it holds that $p _ { F } ^ { ( o ) } ( x , \tau ) = \tilde { p } _ { F } ^ { ( o ) } ( x , \tau )$ for any $\tau \in \mathfrak { s } _ { o } \ \ightsquigarrow \ x \in \mathcal { X } _ { o }$ . Similarly, for any trajectory going from $s _ { o } \mathrm { ~ t o ~ } x \in \mathcal { X } _ { k }$ through some $s \in S _ { k }$ (if the trajectory goes through $s \in S _ { k }$ , it must eventually end up in $\mathcal { X } _ { k }$ by definition),

$$
\tilde { p } _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = p _ { F } ^ { ( o ) } ( s _ { o } , \tau _ { 1 } ) \cdot \frac { h _ { k } ( s ) } { h _ { k } ( s _ { o } ) } \cdot p _ { F } ^ { ( k ) } ( s , \tau _ { 2 } ) = \frac { 1 } { h _ { k } ( s _ { o } ) } \cdot p _ { F } ^ { ( o ) } ( s _ { o } , \tau _ { 1 } ) \cdot p _ { F } ^ { ( k ) } ( s , \tau _ { 2 } ) ,
$$

with $\tau _ { 1 }$ and $\tau _ { 2 }$ being the portions of τ before and after $s \in S _ { k }$ , respectively, and $h _ { k } ( s ) = 1$ since $s \in S _ { k }$ and the Markov chain can only visit one of $\boldsymbol { S _ { k } }$ for $k \in [ K ]$ since there is no positive-probability trajector between them. Also, by the ATB,

$$
p _ { F } ^ { ( o ) } ( s _ { o } , \tau _ { 1 } ) p _ { F } ^ { ( k ) } ( s , \tau _ { 2 } ) = \frac { R ( x ) } { Z } p _ { B } ^ { ( k ) } ( s _ { o } , \tau _ { 2 } ) \cdot p _ { B } ^ { ( o ) } ( s _ { o } , \tau _ { 1 } ) .
$$

As before, for $\tau _ { 1 }$ , it holds that $p _ { B } ^ { ( o ) } ( x , \tau _ { 1 } ) = \tilde { p } _ { B } ^ { ( o ) } ( x , \tau _ { 1 } )$ . For $\tau _ { 2 }$ , on the other hand, a telescoping product shows that

$$
p _ { B } ^ { ( k ) } ( x , \tau ) = \tilde { p } _ { B } ^ { ( k ) } ( x , \tau ) \cdot \frac { g _ { k } ( x ) } { g _ { k } ( s ) } = g _ { k } ( x ) \cdot \tilde { p } _ { B } ^ { ( k ) } ( x , \tau ) .
$$

In particular,

$$
\tilde { p } _ { F } ( s _ { o } , \tau ) = \frac { R ( x ) } { Z } \cdot \frac { g _ { k } ( x ) } { h _ { k } ( s _ { o } ) } \cdot \tilde { p } _ { B } ^ { ( k ) } ( x , \tau ) ,
$$

i.e., defining $Z _ { k } = Z \cdot h _ { k } ( s _ { o } )$ and $q ( k | x ) = g _ { k } ( x )$ , which is a valid probability distribution by Lemma E.3, we infer that

$$
Z _ { k } \tilde { p } _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = R ( x ) q ( k | x ) \tilde { p } _ { B } ( x , \tau )
$$

for $k \in [ K ] , k = 0$ , and every $\tau$ from $s _ { o } \mathrm { ~ t o ~ } x \in \mathcal { X } _ { k }$ through $\boldsymbol { S _ { k } }$ . This shows the solution to SAL is also a solution to the MTB condition in Definition 3.2.

## H.16 Proof of Proposition F.4

We show that the solution $\{ { \mathfrak { g } } _ { k } \} _ { k \geq 1 }$ for the Boosted GFlowNets learning problem abides by the MTB under Assumptions F.1 and F.3.

We first demonstrate that $q ( k | x )$ is a probability distribution over $k \geq 1$ . To see this, we notice that

$$
\sum _ { 1 \leq k \leq K } q ( k | x ) = \sum _ { 1 \leq k \leq K } { \frac { R _ { k } ( x ) } { R ( x ) } } = \sum _ { 1 \leq k \leq K } { \frac { R ^ { ( k ) } ( x ) - R ^ { ( k - 1 ) } ( x ) } { R ( x ) } } = { \frac { R ^ { ( K ) } ( x ) } { R ( x ) } } ,
$$

as $R ^ { ( o ) } ( x ) = 0$ by design. By Assumption F.1, $q ( k | x ) \geq 0 ;$ ; by Assumption F.1, the limit of the above equation as $K  \infty$ is

$$
\operatorname* { l i m } _ { K  \infty } \frac { R ^ { ( K ) } } { R ( x ) } = \frac { R ( x ) } { R ( x ) } = 1 .
$$

This shows $q ( k | x )$ corresponds to a probability distribution over $k \geq 1$

We now prove that the learned policies abide by the MTB condition in Definition 3.2 having $q ( k | x )$ as the posterior mixing distribution. By Assumption F.3,

$$
R ^ { ( k ) } ( x ) = R ^ { ( k ) } ( x , \tau ) : = R ^ { ( k - 1 ) } ( x ) + \frac { Z _ { k } \cdot p _ { F } ^ { ( k ) } ( s _ { o } , \tau ) } { p _ { B } ^ { ( k ) } ( x , \tau ) } ,
$$

i.e.,

$$
p _ { B } ^ { ( k ) } ( x , \tau ) \cdot \Big ( R ^ { ( k ) } ( x ) - R ^ { ( k - 1 ) } ( x ) \Big ) = Z _ { k } \cdot p _ { F } ^ { ( k ) } ( s _ { o } , \tau ) .
$$

Since $R ^ { ( k ) } ( x ) - R ^ { ( k - 1 ) } ( x ) = R _ { k } ( x ) = R ( x ) \cdot q ( k | x )$ by definition,

$$
Z _ { k } \cdot p _ { F } ^ { ( k ) } ( s _ { o } , \tau ) = R ( x ) q ( k | x ) \cdot p _ { B } ^ { ( k ) } ( x , \tau ) .
$$

This is Proposition F.4’s second statement. It should also be clear that, since

$$
Z _ { k } = \sum _ { x \in \mathcal { X } } R _ { k } ( x ) = \sum _ { x \in \mathcal { X } } R ( x ) q ( k | x ) ,
$$

our prior verification shows that

$$
\sum _ { k \geq 1 } Z _ { k } = \sum _ { k \geq 1 } \sum _ { x \in \mathcal { X } } R ( x ) q ( k | x ) = \sum _ { x \in \mathcal { X } } R ( x ) \sum _ { k \geq 1 } q ( k | x ) = \sum _ { x \in \mathcal { X } } R ( x ) = Z .
$$

We can interchange the summations due to the fact that $\begin{array} { r } { \sum _ { k > 1 } q ( k | x ) = 1 } \end{array}$ and is finite. These derivations jointly highlight Boosted GFlowNets can be interpreted as DI GFlowNets having countably infinite components.

## I Related works

Generative Flow Networks have been introduced in (Bengio et al., 2021, 2023) as generative models for discrete, compositional spaces such as graphs and text. They have since been applied to diverse problems, including phylogenetic inference (Zhou et al., 2024), robust scheduling (Zhang et al., 2023), Bayesian structure learning (Deleu et al., 2022, 2023), LLM fine-tuning (Hu et al., 2023; Venkatraman et al., 2024), symbolic regression (Li et al., 2023), quantitative finance (Chen et al., 2026), and drug discovery (Bengio et al., 2021; Laajil et al., 2025; Pandey et al., 2024), and have been popularized by publicly released software libraries (Tiapkin et al., 2025; Viviano et al., 2026). From a theoretical perspective, their connection to variationa inference (Malkin et al., 2023) and reinforcement learning (Tiapkin et al., 2024) and Markov chains (Deleu & Bengio, 2023) has been well-established, and variants operating on cyclic graphs (Brunswic et al., 2024; Morozov et al., 2025) have also been recently considered. As with SC GFlowNets, recent works have proposed modifying the state space of the GFlowNet for improved learning, e.g., (Boussif et al., 2024; Yu et al., 2026b; Silva et al., 2025b), however, their focus lies either on improving the exploratory policy, an issue which is orthogonal to our investigations, or on developing strictly distributed algorithms requiring the training of independent models. Broadly, mixture models have been widely studied in variational and approximate inference, for instance, in semi-implicit variational inference methods (Yin & Zhou, 2018), path-sampling for Monte Carlo methods (Gelman & Meng, 1998; Liu & Liu, 2001), and in continuously indexed normalizing flows (Cornish et al., 2021; Caterini et al., 2021), the terminology of which we borrow for describing our framework. Correspondingly, recent works have proposed partitioning (binning) the continuous component of the state space of both GFlowNets (Zhou et al., 2024) and difusion probabilistic models (Gupta et al., 2026), the outcome of which is a coarse mixture approximation to a continuous density.