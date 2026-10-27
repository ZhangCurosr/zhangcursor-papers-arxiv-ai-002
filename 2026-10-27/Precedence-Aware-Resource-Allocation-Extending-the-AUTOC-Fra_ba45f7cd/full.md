# Precedence-Aware Resource Allocation: Extending the AUTOC Framework to Multi-Level Treatments

Olin E. Gilster

Work done while at Emory University, USA

Shengpu Tang

Emory University, USA

shengpu.tang@emory.edu

## Abstract

Many real-world decision problems require allocating limited resources across individuals to maximize total benefit often through interventions that vary in intensity rather than binary treat-or-not decisions. In these settings, higher levels treatments require first applying lower levels of treatment, inducing a precedence constraint that complicates allocation under a fixed budget. While the causal inference literature has largely emphasized accurate estimation of conditional average treatment efects (CATE), recent work has shown that estimation accuracy alone does not guarantee efective resource allocation, motivating ranking-based evaluation frameworks. The Area Under the Targeting Operating Characteristic (AUTOC) provides an efective way to evaluate prioritization rules across budget levels, but existing formulations are limited to binary treatments. We introduce Multi-Level AUTOC (ML-AUTOC), which is a generalization for AUTOC designed for multi-level treatment settings with precedence. We also propose precedence-aware policies for mapping true marginal treatment effects to feasible treatment assignments under fixed budget constraints. Our results show that in the setting of multi-level treatments, maximizing realized treatment efects may require re-allocation of treatments across individuals.

Data and Code Availability. All code used to generate the experiments and figures in this paper, along with supporting documentation, is publicly available at https://github.com/gilstero/ML-TOC. All datasets used in this study are synthetically generated and can be reproduced using the provided data generation scripts.

Institutional Review Board (IRB). This study does not require IRB approval.

## 1. Introduction

Across the domains of healthcare, education, and public policy, decision makers are often faced with the reality of resource constraints. For example, hospitals may find themselves confronting a surge of crit ically ill patients with limited number of beds or personnel. These constraints force hospitals to allocate their resources based on need, deciding who gets intensive care versus standard or minimal care (French et al., 2021). Beyond healthcare, other domains also face similar challenges. In education, overcrowded schools may limit the amount of attention each child gets. Thus, school systems need to make informed decisions on who to assign excess teacher hours to maximize school test scores and student well-being (Vakili et al., 2024). In the public-policy setting, limited stafing at social outreach programs can reduce the amount of care given to each person. Specifically, in the context of housing, people may be assigned to increasingly better programs from basic shelter, transitional housing, and long-term assistance fully based on the amount of resources available to an organization (Agonafer et al., 2021).

The stakes of such allocation decisions are particularly acute in the healthcare setting (Gabler et al., 2013). Recent papers covering the COVID-19 pan demic found that when hospitals experienced a capacity strain, they had higher rates of mortality. Specifically, they found that in a nationwide resource constraint where ICU capacity was at 75%, “12,000 excess deaths would occur in the following two weeks.” When ICU capacity was at 100%, it was predicted that 80,000 excess deaths would occur in the same time period (French et al., 2021). Together, these examples highlight the widespread challenge of allocating limited resources and the high cost of misallocation, emphasizing the need for systematic approaches to improve decision-making under resource constraints.

Researchers working at the intersection of causal inference and machine learning have largely focused on the problem of conditional average treatment effect (CATE) estimation, aiming to minimize the error between predicted and true individual treatment ef fects (Shalit et al., 2017; K¨unzel et al., 2019). Yet, accurate CATE estimation does not necessarily translate to optimal resource allocation. Thus, researchers have increasingly pivoted toward ranking-based opti mization and evaluation. For instance, Kamran et al. (2024) proposed algorithms that directly optimize rankings for expected benefit, while Caniglia et al. (2021) utilized dynamic marginal structural models (MSMs) to identify optimal strategies under resource usage constraints. To assess the quality of prioritization rules independently of estimation accuracy, Yadlowsky et al. (2025) popularized the area under the targeting operating characteristic (AUTOC) metric, previously described in Zhao et al. (2013). This metric quantifies how efectively a scoring function prioritizes individuals compared to a random allocation baseline across all possible budget levels.

Despite the utility of the AUTOC metric, existing works on ranking-based evaluation frameworks are limited to binary treatment decisions, where each individual is either treated or not treated. Current frameworks do not adequately address settings where interventions vary in intensity and follow a precedence structure, where higher-intensity treatments require first receiving lower-intensity ones. As our paper will illustrate, naively converting such problems into a binary treatment formulation obscures these nuances and can lead to suboptimal allocation decisions. Therefore, extending ranking-based allocation rules and evaluation beyond the binary treatment setting is an essential building block for applying them to a larger range of real-world decision problems.

In this work, we propose a generalization of the binary AUTOC metric to settings involving multilevel treatments. This extension introduces unique challenges, primarily due to the requirement of treatment level precedence. While a scoring function in a binary treatment setting maps to a unique ranking of individuals, our setting utilizes scoring functions that estimate marginal treatment efects for each individual at each treatment level. Every scoring function may correspond to multiple possible rankings of individual–level pairs, some respecting precedence, and some that do not. Furthermore, the random allocation baseline must also be redefined to respect precedence. Finally, as resource budget increases, the optimal allocation may require re-allocating previously assigned resources to diferent individual–level pairs. This leads to the phenomenon of “flipping” or nonmonotonic assignment, which does not exist for binary treatment settings by definition. Specifically, this paper makes the following contributions:

• We develop and formalize the multi-level AU-TOC (ML-AUTOC) metric to evaluate prioritization eficiency of resource allocation tasks with multiple treatment levels.

• We introduce and compare several resource allocation rules, which we refer to as policies, that can be derived from scoring functions, and evaluate their ability to maximize realized treatment efects on synthetic datasets.

• We show that in multi-level treatment settings, the re-allocating PAO policy fundamentally departs from ranking-based approaches, since maximizing realized treatment efects may require non-nested allocation policies that switch between partially treating one individual and initiating treatment for another.

## 1.1. Related Work: CATE

To understand the motivation for AUTOC and our extension to multi-level settings, it is important to examine the limitations of conditional average treatment efect (CATE) estimation in resource constrained problems. Consider the dataset $S \ =$ $( x _ { i } , t _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ , where each individual i has covariates $x _ { i } \in X \subseteq \mathbb { R } ^ { d }$ , assigned treatment $t _ { i } \in \{ 0 , 1 \}$ , and observed outcome $y _ { i } \in \mathbb { R } .$ . Under the potential outcomes framework (Rubin, 1974), each individual is given potential outcomes $Y _ { i } ( 0 )$ and $Y _ { i } ( 1 )$ corresponding to control and treatment. Given these, CATE is defined as:

$$
\tau _ { i } = \mathbb { E } [ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \mid x _ { i } ] .
$$

CATE allows researchers to go beyond a single global estimate of the average treatment efect (ATE) by estimating $\tau _ { i }$ conditioned on an individual’s covariates $x _ { i }$ (K¨unzel et al., 2019). Despite the success in CATE research, methods often focus on estimation accuracy rather than allocation eficiency (Kamran et al., 2024; Yadlowsky et al., 2025). While CATE-based models can estimate treatment efect heterogeneity with high precision, they may still lead to suboptimal allocation decisions when resources are limited. It was found in Kamran et al. (2024) that minimizing

CATE predictive error does not necessarily translate to maximizing total utility under a budget constraint. Moreover, most CATE frameworks implicitly assume that treatment can be assigned without accounting for constraints on total resource use (Caniglia et al., 2021). In practice, interventions often vary in cost and level, which require trade-ofs where only a subset of individuals can be treated under a given budget. These limitations have motivated the development of ranking-based frameworks, such as AUTOC, which explicitly evaluates how efectively a model prioritizes individuals for treatment as available resources vary (Kamran et al., 2024; Yadlowsky et al., 2025).

## 1.2. Related Work: AUTOC

AUTOC provides the basis for evaluating treatment prioritization when the treatment structure is binary, and ML-AUTOC closely mirrors this framework in the multi-level setting; for this reason, it is important to first understand AUTOC.

The goal of AUTOC is to evaluate how well a scoring function $f ( x _ { i } )$ , which can be mapped to a ranking over individuals, aligns with the true ranking based on CATE $\tau _ { i }$ . Given a treatment budget $u \in [ 0 , 1 ]$ $D _ { u } ^ { S } ( f )$ denotes the top u-fraction of individuals in $S$ ranked the highest by score $f \colon$

$$
D _ { u } ^ { S } ( f ) = \{ i \mid f ( x _ { i } ) \geq \psi ( f ( x _ { i } ) _ { i \in S } , u ) \}
$$

where $\psi ( \cdot , u )$ represents the $( 1 - u )$ quantile of model scores. The ATE at budget u induced by the scoring function $f$ is defined as the mean of individual treatment efects for those individuals selected to be treated:

$$
\mathrm { A T E } _ { u } ^ { S } ( f ) = \frac { 1 } { D _ { u } ^ { S } ( f ) } \sum _ { i \in D _ { u } ^ { S } ( f ) } \tau _ { i } .
$$

The targeting operating characteristic (TOC) at u quantifies how much better this ATE is than that of a random selection at the same budget. In the binary case, it is straightforward to obtain the random baseline since the expected ATE under random assignment is equal to the population ATE. Thus, the TOC is defined as:

$$
\mathrm { T O C } _ { u } ^ { S } ( f ) = \mathrm { A T E } _ { u } ^ { S } ( f ) - \frac { 1 } { | S | } \sum _ { i = 1 } ^ { | S | } \tau _ { i }
$$

Finally, the area under the TOC curve averages over all budgets:

$$
\operatorname { A U T O C } _ { S } ( f ) = \frac { 1 } { | S | } \sum _ { i = 1 } ^ { | S | } \operatorname { T O C } _ { \frac { i } { | S | } } ^ { S } ( f ) .
$$

Intuitively, AUTOC captures the average improvement in treatment benefit obtained by using the model to evaluate patients at every possible resource level. A larger AUTOC value indicates that a ranking is more accurately prioritizing individuals who would benefit most from treatment, while an $\mathrm { \ A U T O C = 0 }$ is equivalent to random allocation (Kamran et al., 2024; Yadlowsky et al., 2025).

## 2. Methods

## 2.1. Problem Statement

While AUTOC provides a strong evaluation metric for binary treatment allocation (French et al., 2021; Vakili et al., 2024; Agonafer et al., 2021), many de cision problems fall into multi-level intervention decisions where treatment intensity can be quantified as discrete levels $\ell \in \{ 0 , 1 , \ldots L \}$ . In our case, these treatment levels exhibit precedence; that is, higher treatment levels may only be assigned if previous levels have been assigned. Each higher level typically provides a positive marginal benefit, denoted:

$$
\delta _ { i } ^ { ( \ell ) } = \mathbb { E } [ Y _ { i } ( \ell ) - Y _ { i } ( \ell - 1 ) \mid x _ { i } ]\tag{1}
$$

representing the incremental gain from advancing an individual i from level ℓ − 1 to ℓ. Standard AUTOC (Yadlowsky et al., 2025; Kamran et al., 2024; Zhao et al., 2013) which assumes a binary decision space $t _ { i } \in \{ 0 , 1 \}$ cannot encapsulate these hierarchical relationships because it evaluates rankings on a single treatment basis.

Consider the dataset $S = ( x _ { i } , t _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ where each individual i has covariates $x _ { i } ~ \in ~ X$ , assigned treatment levels $t _ { i } \in \{ 0 , 1 , \ldots L \}$ , and observed outcome $y _ { i } .$ . Because of the treatment structure, each individual has a vector of potential outcomes:

$$
Y _ { i } = ( Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , \ldots , Y _ { i } ( L ) ) ,
$$

which can be derived from the marginal treatment efects in Equation (1). Given our goal is to evaluate rather than optimize a scoring function, in our experiments, we do not model or simulate covariates $X .$ . Instead, we directly simulate the outputs of the scoring functions in order to study how a given ranking should be translated into a treatment allocation policy under a fixed budget.

There remains a challenge in defining the budget for ML-AUTOC. In the standard AUTOC metric, the unit for each incremental movement in the budget is the choice to treat an individual. In our setting, individuals can be treated multiple times, which invalidates using them as a whole unit. To address this, we define the budget in terms of incremental treatment steps. Moving an individual from treatment level ℓ−1 to level ℓ consumes one unit of budget, providing a natural and consistent unit of resource expenditure in the multi-level setting. One may alternatively assume that higher treatment levels cost more (or less) than lower treatment levels, butu for simplicity, we focus our discussion within the unit-cost setting.

Our problem setting is closely related to multiperiod precedence-constrained knapsack problems, which study the allocation of resources across hierar chical or ordered choices under a fixed budget. Prior work has explored such problems in static or multiperiod settings (e.g., Moreno et al. (2010); Aslan et al. (2023); Samavati et al. (2017)), typically focusing on optimizing a single objective value at a fixed budget.

## 2.2. Solutions via Dynamic Programming

The allocation problem that underlies ML-AUTOC is closely related to precedence-constrained multiplechoice knapsack problem (MCKP). Each individual i represents a class of mutually exclusive cumulative treatments, corresponding to levels $\ell \in \{ 0 , \ldots , L \}$ , and each level produces cumulative benefit

$$
V _ { i } ( \ell ) = \sum _ { k = 1 } ^ { \ell } \delta _ { i } ^ { ( k ) } = \mathbb { E } [ Y _ { i } ( \ell ) - Y _ { i } ( 0 ) \mid x _ { i } ] .
$$

The planner must choose the level of treatment per individual such that the total budget does not exceed $\beta$ and the total benefit is maximized. Formally, the problem for a fixed budget $\beta$ is:

$$
\operatorname* { m a x } _ { \ell _ { i } \in \{ 0 , \ldots , L \} } \sum _ { i = 1 } ^ { n } V _ { i } ( \ell _ { i } ) \quad { \mathrm { s u b j e c t ~ t o } } \quad \sum _ { i = 1 } ^ { n } \ell _ { i } \leq \beta .
$$

This formulation is a special case of the MCKP, one of the most well-studied combinatorial optimization problems (Fr´eville, 2004; Kellerer et al., 2004). While integer linear programming (ILP) formulations are common, dynamic programming (DP) methods provide a natural and computationally eficient approach when budgets are discrete and relatively small (Martello and Toth, 1990).

Dynamic programming is particularly well suited to our setting for two reasons:

1. Evaluation Across All Budgets. ML-AUTOC requires evaluating optimal allocations for every $\beta \in \{ 0 , 1 , \ldots , n L \}$ . A single DP table computes optimal values for all budgets simultaneously, whereas solving separate ILPs for each $\beta$ would be redundant.

2. Ranking Recovery. The DP solution naturally stores dependencies between solutions to sub-problems of diferent $\beta$ values. By tracing which individual–level pair (marginal treatment) is selected at each increment of $\beta ,$ we recover an implied ordering of treatment increments.

More information on the exact dynamic programming implementation can be found in Appendix A.9.

## 2.3. Scoring Functions, Rankings, and Allocation Policies

Before describing our proposed approach, we need to distinguish a few related concepts that are equivalent under the binary treatment setting, but becomes tricky under multi-level treatments. In the binary treatment setting, a scoring function induces a unique ranking of individuals (up to tie-breaking), and treating the top ranked individuals at each bud get defines an allocation policy. As a result, scoring functions, rankings, and allocation policies are all effectively interchangeable. In the multi-level setting, this equivalence no longer holds. A multi-level scoring function associates each treatment level ℓ with a level-specific scoring function $f _ { \ell } ( x _ { i } )$ . These scores are intended to rank individuals according to their predicted marginal benefit at level $\ell , \delta _ { i } ^ { ( \ell ) }$ . Thus, the scoring function assigns values to individual–level pairs rather than individuals. Unlike in the binary case, these level-specific scores by themselves do not define how treatments should be allocated across levels and individuals as the budget increases. From the scoring function, we construct a policy π that selects a set of individual–level pairs (i, ℓ) subject to a budget constraint. Let

$$
\Delta = \{ ( i , \ell ) : i \in \{ 1 , \dots , n \} , \ell \in \{ 1 , \dots , L \} \}
$$

denote the set of all individual–level pairs. We de fine a policy as a mapping from a budget level $\beta \in$

$\{ 0 , 1 , \ldots , n L \}$ to a feasible subset of pairs:

$$
\pi : \{ 0 , 1 , \ldots , n L \}  2 ^ { \Delta } , \quad \beta \mapsto \pi ( \beta ) ,
$$

where $\pi ( \beta ) \subseteq \Delta$ contains the pairs selected by $\pi$ under budget $\beta _ { i }$ , and precedence constraints restrict what treatment allocations are feasible at a given budget. As the budget increases, the sequence of sets $\{ \pi ( \beta ) \} _ { \beta }$ (possibly) induces an ordering over treatment assignments. We interpret this sequence as a ranking over feasible individual–level pairs.

The best way to conceptualize the two objects, $f _ { \ell } ( x _ { i } )$ and $\pi ,$ is to consider the scoring function first. The scoring function produces a level-specific ranking of treatment assignments based on predicted marginal benefits, which can be interpreted as a priority list. A policy then takes these rankings and applies a set of allocation rules, such as bud get constraints, precedence requirements, or aggregation across levels, to determine which individual–level pairs are ultimately selected. Importantly, this mapping from $f _ { \ell } ( x _ { i } )$ to $\pi$ is not unique, unlike in the binary treatment setting. As we will demonstrate later in this section, it is possible that the same scoring function may give rise to multiple feasible policies, and some allocation policies can not be represented as a single global ranking.

Consider the following toy example in Table 1:

Table 1: True Marginal Benefit table
<table><tr><td rowspan=1 colspan=1> $\overline { { \delta _ { i } ^ { ( \ell ) } } }$ </td><td rowspan=1 colspan=1> $\ell = 1$ </td><td rowspan=1 colspan=1> $\ell = 2$ </td></tr><tr><td rowspan=1 colspan=1> $x _ { 1 }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1> $\overline { { 5 } }$ </td></tr><tr><td rowspan=1 colspan=1> $x _ { 2 }$ </td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr></table>

At level $\ell = 1$ , since $\delta _ { 2 } ^ { ( 1 ) } > \delta _ { 1 } ^ { ( 1 ) }$ , individual $x _ { 2 }$ should receive priority for the first unit of treatment. However, at level $\ell \ = \ 2 .$ , the ordering reverses, as $\delta _ { 2 } ^ { ( 2 ) } < \delta _ { 1 } ^ { ( 2 ) }$ . This highlights a key challenge in the multi-level setting: a single global score cannot, in general, correctly prioritize treatment assignments over individuals across all levels. For instance, a global scoring rule of the form $f ( x _ { i } ) \propto Y _ { i } ( L ) { - } Y _ { i } ( 0 )$ (or equivalently $\Sigma _ { \ell = 1 } ^ { L } \delta _ { i } ^ { ( \ell ) } )$ would rank $x _ { 1 } \succ x _ { 2 }$ overall in this example $\left( 6 > 5 \right)$ . Under tight budgets, such a ranking would incorrectly allocate the first unit of treatment to $x _ { 1 }$ , despite $x _ { 2 }$ having the larger marginal benefit at $\ell = 1$

## 2.4. Naive Policies

To map a scoring function to a policy in the multilevel treatment setting, one could naively convert the problem to a binary treatment problem. Below, we discuss three possibilities:

• Pooled Global Policy (PGP):

This policy creates a global ranking by pooling all marginal scores $\{ f _ { \ell } ( x _ { i } ) : \forall i , \ell \}$ across individuals i and levels $\ell ,$ and selecting the highest val ues according to the marginal score ranking (e.g. $5 , 3 , 2 , 1 )$ . While this identifies high-benefit individual–level pairs, it frequently violates precedence constraints, for example, assigning Level 2 to an individual before they receive Level 1.

## • Individual-First Policy (IFP)

This approach aggregates each individual’s total potential benefit, $\textstyle \sum _ { \ell } f _ { \ell } ( x _ { i } )$ , and ranks all individuals by this aggregate score $( \mathrm { e . g . , 1 + 5 = 6 }$ $2 + 3 = 5 )$ . Resources are allocated to the topranked individual until they are fully treated across all levels before moving to the next individual (e.g., 1, 5, 2, 3). Within each individual, treatment levels are assigned in increasing order of intensity, enforcing the precedenceinduced ranking of levels. While this respects precedence, it cannot spread lower-level treatments across many individuals, even when the marginal benefit of Level 1 is significantly higher than the marginal benefit of Level 2.

• Level-First Policy (LFP): This policy imposes a strict hierarchy based on treatment levels, similar to classical level-based scheduling algorithms $( \mathrm { H u } .$ , 1961). It requires that every individual in the population receive Level 1 before any individ ual can be considered for Level 2. Within each level, individuals are ranked by the marginal benefits of that level, e.g., allocating Level 1 in the order (2, 1) and allocating Level 2 in the order (5, 3) for the overall ranking of (2, 1, 5, 3).

## 2.5. Precedence-Aware Policies

Unfortunately, these rankings fall short as no specific one fully captures the structure of the problem: they ignore how marginal benefits vary by level or fail to respect the hierarchical dependencies across treatment levels. What we want is a ranking that i) accounts for level-specific marginal benefits and ii) preserves precedence constraints. Because the $\mathsf { P G P }$ IFP, and LFP policies fail to balance precedence with marginal utility, we propose two precedence-aware policies that dynamically navigate these trade-ofs to maximize total population benefit.

A feasible allocation must satisfy the precedence constraint that, for any individual i, treatment level ℓ can only be assigned if all lower levels have already been assigned. Formally, if $( i , \ell )$ is selected, then (i, k) must also be selected for all $k < \ell .$ . Once taking into account what possible treatments are available at a given time, we use a greedy strategy to select the one with the highest marginal benefit as the next one to treat. Two possible policies can be built of of these rules, which we call Precedence-Aware Greedy (PAG) and Precedence-Aware Optimal (PAO). The distinction between the two rankings lies in the reallocation of treatments at any distinct budget. The PAG policy constructs the ranking incrementally, fixing earlier treatment assignments and never revising past decisions as the budget expands. In contrast, the PAO policy allows re-optimization at each budget level, potentially revising earlier allocations to account for newly feasible treatment options. From an algorithmic perspective, PAO corresponds to solving a sequence of budget-constrained optimization problems via dynamic programming. Although we do not discuss it here, a detailed description of the dynamic programming formulation and its computational complexity is provided in Appendix A.9.

For the toy example in Table 1, each policy is evaluated compared to a random baseline allocation under the same budget (defined in Section 2.6). In this specific case, all feasible sets were calculated to determine a true average for the random baseline. We plot the TOC curves for each policy, showing budget levels versus average benefit across the cohort:

Figure 1: Toy example ML-TOC Graph  
![](images/570cd322eba7444787d9311cd5284da9d3d6cb948272523a8988426d339c77b0.jpg)

The TOC curves in Figure 1 illustrate how diferent policies behave under the same non-binary marginal benefit structure. The PAO policy achieves the greatest benefit while adhering to precedence constraints. The PAG policy matches PAO at the smallest budget but falls behind at intermediate budgets, since its nested allocation cannot revise earlier choices, while IFP matches PAO from the second budget onward.

The PGP appears to perform best, but its advantage is artificial, as it violates precedence twice. This illustrates why multi-level rankings cannot be evaluated correctly if these constraints are not enforced.

## 2.6. Formalizing ML-AUTOC

Because individuals can be treated incrementally across multiple levels, defining treatment thresholds in terms of the top u% of individuals is not a meaningful metric. Instead, we index allocations by a scalar budget parameter $\beta \in \{ 1 , \ldots , n L \} = { \cal B } _ { \mathrm { { f } } }$ which represents the total number of treatment units assigned across all individuals and levels. For a given scoring function $f = \{ f _ { \ell } : \ell = 1 \cdot \cdot \cdot L \}$ , we define the allocation set under budget $\beta$ as:

$$
D _ { \beta } ^ { S } ( f ) = \pi _ { f } ( \beta )\tag{2}
$$

where $\pi _ { f } ( \beta ) \subseteq \Delta$ is the set of individual–level pairs selected by the policy induced by f under budget $\beta .$ The average marginal benefit realized under budget $\beta$ is then:

$$
\mathrm { A T E } _ { \beta } ^ { S } ( f ) = \frac { 1 } { | D _ { \beta } ^ { S } ( f ) | } \sum _ { ( i , \ell ) \in D _ { \beta } ^ { S } ( f ) } \delta _ { i } ^ { ( \ell ) } .\tag{3}
$$

The targeting operating characteristic at budget $\beta$ quantifies how much better the selected subset is compared to a baseline equivalent to random selection and needs to respect the treatment level precedence structures. To calculate the random baseline, we define $\mathcal { F } _ { \beta }$ as the set of all feasible allocations of individual–level pairs under a budget $\beta$ that satisfy precedence. We consider a random allocation $\widetilde { A } _ { \beta }$ drawn uniformly from $\mathcal { F } _ { \beta }$ Therefore, for all $( i , \ell ) ~ \in ~ \widetilde { A } _ { \beta }$ we can define ${ \mathrm { R A N D } } _ { \beta }$ as the $\mathbb { E } _ { \widetilde { A } _ { \beta } } [ \sum _ { ( i , \ell ) \in \widetilde { A } _ { \beta } } \delta _ { i } ^ { ( \ell ) } ]$ . In practice, this can be calculated via enumeration (for small n and $L )$ or Monte Carlo sampling to approximate the expectation. Now for a scoring function that induces a feasible allocation, TOC can be defined as:

$$
\mathrm { T O C } _ { \beta } ^ { S } ( f ) = \mathrm { A T E } _ { \beta } ^ { S } ( f ) - \mathrm { R A N D } _ { \beta } .\tag{4}
$$

Let B denote the set of feasible budgets, with |B| its cardinality, so that $\frac { 1 } { | B | }$ averages over these budgets. Finally, the ML-AUTOC score aggregates realized average treatment efects across all budgets:

$$
\mathrm { M L - A U T O C } _ { S } ( f ) = { \frac { 1 } { | B | } } \sum _ { \beta \in { \mathcal { B } } } \mathrm { T O C } _ { \beta } ^ { S } ( f ) .\tag{5}
$$

ML-AUTOC and AUTOC are designed to measure the same conceptual object, which is how well a learned scoring rule prioritizes units for treatment as resources increase. The diference is in how the comparison is made. AUTOC is defined for binary treatment, where each additional unit of budget corresponds to treating one new individual and the ran dom baseline is a uniform draw over subsets of individuals of a given size. In ML-AUTOC, the budget expands over treatment increments (i, ℓ) which are subject to precedence, so the random baseline must be defined over the feasible allocation set. This change in the feasible set (and therefore in the baseline being subtracted) is the fundamental diference between AUTOC and ML-AUTOC.

A second diference is the unit for which the area is accumulated. In binary AUTOC, the horizontal axis is naturally interpreted as the treated fraction u. In the multi-level setting, the natural budget unit is the number of incremental level assignments, so the curve aggregates performance over $\beta = n L$ possible budget increments. One could equivalently normalize by the maximum budget nL, expressing performance as a function of the fraction of budget expended. This normalized representation is used in our plots, where the horizontal axis corresponds to the percentage of budget. One could alternatively normalize the horizontal axis by the number of unique individuals treated at each budget, yielding a per-patient notion of eficiency. This normalization is not efective because it is not aligned with the resource model in which costs increase per incremental level, so it would conflate diferent allocations that treat the same number of individuals but at diferent intensities.

Finally, ML-AUTOC has the same value interpretation as AUTOC for policies that induce a nested ranking over treatment actions, and can be extended as a cumulative performance metric for more general allocation policies. Naive policies such as LFP, IFP, and PGP as well as PAG produce an explicit ordering of (i, ℓ) actions, so the ML-AUTOC area can be interpreted as a cumulative comparison to the random precedence-feasible baseline as the budget expands. Although PAO does not generate a single nested ranking because it re-optimizes the allocation at each budget and may revise earlier decisions, its ML-AUTOC can still be computed as the cumulative average treatment efect relative to the random baseline across budgets. In this sense, ML-AUTOC provides a unified metric for comparing both nested and non-nested policies. In addition to ML-AUTOC, we also compare policies to PAO budget-wise by reporting the fraction of budgets for which one policy achieves a higher realized benefit than another (“win %”). This complementary metric highlights pointwise performance diferences that may not be fully captured by cumulative area comparisons.

## 2.7. Research Question

Does a precedence-aware greedy policy achieve higher total average treatment efect (ATE) than naive ranking policies in multi-level treatment settings with precedence constraints?

To investigate this, we examine the following empirical questions:

1. Policy Performance: On synthetic datasets with diferent per-level treatment efect structures, how do PAG and PAO compare to naive policies in terms of realized total average treatment efect across budget levels?

2. Structural Behavior: How do allocations produced by PAO which allows for re-allocations dif fer from those produced by nested greedy allocations (PAG), and when does relaxing the mono tonic allocation requirement lead to a higher re alized average treatment efect?

## 3. Experiments

## 3.1. Datasets

In order to evaluate ML-AUTOC under varying structural conditions, we generated 100 synthetic datasets of four types that were designed to isolate how diferent marginal benefit structures influence policy performance. Synthetic data was used for our experiment following Kamran et al. (2024) since true individual-level treatment efects are rarely observable in practice. Since only one realized out come was observed per individual, evaluating optimal treatment policies was infeasible using real data. Moreover, our framework assumes access to marginal treatment benefits, which further motivates the use of synthetic environments to simulate ground-truth performance under controlled assumptions.

Each dataset contains n individuals who may receive treatment levels $\ell \in \{ 1 , 2 , \ldots , L \}$ (with L = 4 in all experiments) with the choice to treat a single individual to the next level ℓ being a unit of expenditure. Thus, any diferences in model performance arise solely from the marginal benefit structure and from how rankings interact with precedence constraints. The datasets below were constructed to isolate how diferent marginal benefit structures interact with precedence constraints and to evaluate both the performance of ranking policies in terms of realized average treatment efect. In addition, their design facilitated the study of when naive, nested ranking heuristics fail and when precedence-aware, budgetadaptive policies can achieve higher realized benefit. In particular, they allowed us to:

• RQ1: evaluate how the proposed PAG and PAO policies compare to naive policies in terms of realized average treatment efect across budgets,

• RQ2: evaluate when relaxing nested (prefixbased) ranking requirements, by allowing reallocation across budgets in PAO, leads to a higher realized total average treatment efect compared to the nested greedy PAG policy.

The full data-generating process, including distributional assumptions and implementation details for each dataset type, is provided in Appendix A.10. Strictly Increasing Marginal Benefit Dataset. In this dataset, higher treatment levels are always substantially more beneficial. To generate this, we first draw a baseline benefit for each level and each individual from a normal distribution. We then enforce monotonicity by increasing each successive level’s marginal benefit by at least one unit if the raw draw does not satisfy this. As a result, the marginal benefits always satisfy: $\delta _ { i } ^ { ( \ell ) } \geq \delta _ { i } ^ { ( \ell - 1 ) } + 1$

Random Marginal Benefit Dataset. In this dataset, marginal benefits are nonnegative and have no ordering across treatment levels. For each individual i and level $\ell ,$ the marginal benefit $\delta _ { i } ^ { ( \ell ) }$ satisfies $\delta _ { i } ^ { ( \ell ) } \geq 0 .$ , but is otherwise unconstrained across levels. In particular, no monotonicity condition is imposed: higher treatment levels may yield greater, smaller, or comparable marginal benefit relative to earlier levels. As a result, individuals may benefit most from early treatment intensities, later intensities, or intermediate levels, and these patterns can vary irregularly across both individuals and levels.

Strictly Decreasing Marginal Benefit Dataset. In this dataset, marginal benefits are strictly decreasing across treatment levels, so earlier treatment intensities are always more beneficial than later ones. This dataset is constructed by taking the Strictly Increasing Marginal Benefit Dataset that appears as follows:

$$
\delta _ { i } ^ { ( 1 ) } < \delta _ { i } ^ { ( 2 ) } < \delta _ { i } ^ { ( 3 ) } < \delta _ { i } ^ { ( 4 ) }
$$

and applies a transformation so that:

$$
\tilde { \delta } _ { i } ^ { ( 1 ) } = \delta _ { i } ^ { ( 4 ) } , \quad \tilde { \delta } _ { i } ^ { ( 2 ) } = \delta _ { i } ^ { ( 3 ) } , \quad \tilde { \delta } _ { i } ^ { ( 3 ) } = \delta _ { i } ^ { ( 2 ) } , \quad \tilde { \delta } _ { i } ^ { ( 4 ) } = \delta _ { i } ^ { ( 1 ) } .
$$

where $\tilde { \delta } _ { i } ^ { ( \ell ) }$ denotes the marginal benefit at level ℓ in the strictly decreasing dataset. Formally, the marginal benefits are constrained to satisfy: $\delta _ { i } ^ { ( \ell ) } \ \leq$ $\delta _ { i } ^ { ( \ell - 1 ) } - 1$ for all individuals and levels.

Heterogeneous Strictly Increasing Marginal Benefit Dataset. In this dataset, marginal benefits remain strictly increasing across treatment levels for each individual, but the scale of these benefits varies across individuals. Unlike the homogeneous strictly increasing dataset, individuals are not drawn from a common marginal benefit distribution. Instead, each individual i has a scale parameter $s _ { i }$ drawn from a log-normal distribution, and marginal benefits are generated sequentially across levels subject to $\delta _ { i } ^ { ( \ell ) } \geq \delta _ { i } ^ { ( \ell - 1 ) } + 1$ Because individuals differ in scale, lower treatment levels of one individual may exceed higher treatment levels of another $( \mathrm { e . g . } , \delta _ { i } ^ { ( 2 ) } > \delta _ { j } ^ { ( 3 ) } )$ , so within-individual monotonicity is preserved while the relative ranking of marginal increments across individuals and levels is nontrivial. This setting is useful for studying the trade-of between completing treatment for high-potential individuals and allocating early treatment levels across many individuals.

## 3.2. RQ1: Strictly Increasing Marginal Benefit Dataset

As a result of the structure of the strictly increasing dataset, policies that prioritize allocating higher levels early achieve a substantially higher average treatment efect across budgets. Among policies that respect the precedence structure (i.e., all policies except PGP), both IFP and PAG achieve similar performance. To further illustrate how policies behave across the full budget range, we examine the average ML-TOC curves aggregated across all datasets. Figure 2 plots the mean targeting operating characteristic for each policy, with budgets on the horizontal axis and average treatment efect relative to a random baseline on the vertical axis. Consistent with the aggregate results in Table 2, the average curves reveal that IFP, PAO, and PAG exhibit nearly identical performance across all budget levels.

The LFP curve remains distinctly separated from the other policies, not only due to its lower overall performance, but also because of its characteristic shape. This reflects the fact that LFP fully allocates lower treatment levels across individuals before advancing to higher levels and thus spends early budgets on the lowest-level, smallest marginal benefits.

![](images/c15f3a7b71831071003b9749aafd620b6682494445c10ae6837db0e8796ee728.jpg)  
Figure 2: Average TOC curves across budgets for the strictly increasing marginal benefit dataset. Curves are averaged over 100 datasets and plotted relative to a random baseline. ± one standard deviation error bars are plotted but too small to be visually distinguishable.

Table 2: ML-AUTOC Values for Strictly Increasing Marginals
<table><tr><td>Policy</td><td> $\mathrm { M L \mathrm { - } A U T O C \ ( m e a n \pm s t d ) }$ </td></tr><tr><td>PGP</td><td> $1 . 6 6 6 3 5 2 \pm 0 . 0 0 3 7 4 4$ </td></tr><tr><td>IFP</td><td> $0 . 9 4 1 8 2 7 \pm 0 . 0 0 9 6 4 6$ </td></tr><tr><td>LFP</td><td> $- 0 . 1 5 4 4 8 8 \pm 0 . 0 0 3 2 0 9$ </td></tr><tr><td>PAG</td><td> $0 . 9 4 0 7 2 6 \pm 0 . 0 0 9 6 4 6$ </td></tr><tr><td>PAO</td><td> $0 . 9 4 1 8 2 8 \pm 0 . 0 0 9 6 4 6$ </td></tr></table>

Overall, in settings with strictly ordered marginal benefits, these findings indicate that allowing flexible, non-nested allocation (as in PAO) provides no additional benefit in strictly increasing settings, where the optimal allocation is inherently nested and can be naively achieved by IFP.

## 3.3. RQ1: Random Marginal Benefit Dataset

Unlike in the strictly increasing case, in the random marginal case clear separation emerges between policies as the budget increases. The flexible precedenceaware policy PAO consistently dominates all other valid policies.

![](images/f190d5b5d33daafe2e073f0fd1555dc0739aa62baecf62a0265861615a7bdd55.jpg)  
Figure 3: Average TOC curves across budgets for the random marginal benefit dataset. Curves are averaged over 100 datasets and plotted relative to a random baseline. ± one standard deviation error bars are plotted but too small to be visually distinguishable.

While the greedy policy PAG initially outperforms the global score policy IFP, the two curves intersect at approximately 25% of the total budget (β ≈ 1000/4000). Beyond this point, IFP achieves higher realized average treatment efect for the remainder of the budget range. This crossover highlights the cost of enforcing nested (prefix-based) rankings when marginal benefits vary across treatment levels, as early allocation decisions restrict the ability of PAG to adapt at later budgets. This suggests that in the case of smaller budgets and random marginal benefits, PAG may be a better policy to pursue to allocate resources.

In contrast, PAO avoids this tradeof entirely by allowing non-nested, dynamically optimized allocations, enabling it to select the highest marginal benefit action at each budget. As a result, PAO maintains superior performance across nearly all budget levels.

Between precedence-respecting policies, PAG and IFP achieve nearly identical ML-AUTOC values, despite exhibiting diferent behavior across the budget range in Figure 3. This highlights an important distinction between pointwise performance and cumulative performance: although IFP overtakes PAG at larger budgets, the early-budget advantage of PAG is suficient to equalize total realized benefit when aggregated across all budgets. However, PAO achieves a substantially higher ML-AUTOC than both PAG and IFP, indicating that flexible allocation improves not only pointwise performance but also cumulative performance when marginal benefits are heterogeneous.

Table 3: ML-AUTOC Values for Random Marginals
<table><tr><td>Policy</td><td>ML-AUTOC (mean ± std)</td></tr><tr><td>PGP</td><td> $0 . 2 8 5 9 4 3 \pm 0 . 0 0 5 9 2 6$ </td></tr><tr><td>IFP</td><td> $0 . 1 4 1 5 7 5 \pm 0 . 0 0 4 9 4 7$ </td></tr><tr><td>LFP</td><td> $0 . 0 9 6 4 4 5 \pm 0 . 0 0 3 3 4 1$ </td></tr><tr><td>PAG</td><td> $0 . 1 4 1 0 2 4 \pm 0 . 0 0 4 7 2 7$ </td></tr><tr><td>PAO</td><td> $0 . 2 0 9 5 9 9 \pm 0 . 0 0 4 1 8 3$ </td></tr></table>

Overall, in the random marginal benefit setting, relaxing nested ranking constraints enables higher pointwise performance across budgets, while cumulative $\mathrm { M L \mathrm { - } A U T O C }$ comparisons reveal tradeofs between early- and late-budget performance among nested policies.

that IFP, PAO, and PAG achieve nearly identical performance. The diferences across these policies are negligible both in mean performance and variance, indicating that the additional flexibility of PAO does not translate into meaningful gains.

Table 5: ML-AUTOC Values for Heterogeneous Strictly Increasing Marginals
<table><tr><td></td><td>Policy ML-AUTOC (mean ± std)</td></tr><tr><td>PGP</td><td> $1 . 9 0 1 6 8 1 \pm 0 . 0 1 9 5 8 3$ </td></tr><tr><td>IFP</td><td> $1 . 3 9 4 5 4 0 \pm 0 . 0 2 8 5 5 5$ </td></tr><tr><td>LFP</td><td> $0 . 0 0 3 4 7 6 \pm 0 . 0 0 8 9 1 9$ </td></tr><tr><td>PAG</td><td> $1 . 3 9 2 3 8 2 \pm 0 . 0 2 8 5 7 3$ </td></tr><tr><td>PAO</td><td> $1 . 3 9 4 5 4 8 \pm 0 . 0 2 8 5 5 4$ </td></tr></table>

## 3.4. RQ1: Strictly Decreasing Marginal Benefit Dataset

Table 4: ML-AUTOC Values for Strictly Decreasing Marginals
<table><tr><td>Policy</td><td> $\mathrm { M L \mathrm { - } A U T O C \ ( m e a n \pm s t d ) }$ </td></tr><tr><td>PGP</td><td> $0 . 3 5 0 0 6 9 \pm 0 . 0 0 3 8 2 0$ </td></tr><tr><td>IFP</td><td> $- 0 . 3 6 8 6 4 8 \pm 0 . 0 0 9 7 5 7$ </td></tr><tr><td>LFP</td><td> $0 . 3 4 8 2 3 3 \pm 0 . 0 0 3 3 4 8$ </td></tr><tr><td>PAG</td><td> $0 . 3 5 0 0 6 9 \pm 0 . 0 0 3 8 2 0$ </td></tr><tr><td>PAO</td><td> $0 . 3 5 0 0 6 9 \pm 0 . 0 0 3 8 2 0$ </td></tr></table>

In the strictly decreasing marginal benefit setting, PGP, PAG, PAO, and LFP exhibit nearly identical performance. This occurs because optimal allocations in this setting prioritize exhausting high early marginal benefits, leaving little room for gains from reallocating treatment across individuals at later budgets. IFP performs worse because it is forced to treat an entire individual and therefore is unable to capitalize on larger marginals earlier. As a result, flexibility provides no meaningful advantage when marginal benefits are strictly front-loaded. An average plot, similar to the one created for the strictly increasing and random marginal dataset is included in Appendix A.11.

## 3.5. RQ1: Heterogeneous Strictly Increasing Marginal Benefit Dataset

The heterogeneous strictly increasing marginal benefit setting introduces substantial overlap in the ranking of marginal treatment increments by PAO, PAG, and IFP. The ML-AUTOC values in Table 5 show

![](images/fd0ef0199b820728abae9df2633654018999825673c9dda058f5c953ba942b04.jpg)  
Figure 4: Average TOC curves across budgets for the heterogeneous strictly increasing marginal benefit dataset. Curves are averaged over 100 datasets and plotted relative to a random baseline. ± one standard deviation error bars are plotted but too small to be visually distinguishable.

Consistent with Table 5, the curves for IFP, PAO, and PAG overlap and are visually indistinguishable across all budget levels (Figure 4). This overlap indicates that, despite cross-individual heterogeneity, all three policies induce nearly identical rankings of marginal treatment increments. While heterogeneity introduces crossings in the global ordering, it does not fundamentally disrupt the prefix structure of the optimal allocation. As a result, policies that enforce nested allocations remain suficient to capture the majority of attainable gains.

## 3.6. RQ2

Table 6 quantifies the structural diferences between flexible and nested precedence-aware policies by measuring how often PAO strictly outperforms competing policies across budgets.

Table 6: Frequency with which PAO strictly outperforms baseline policies across budgets.
<table><tr><td>Dataset</td><td>% Budgets PAO &gt; IFP</td><td>% Budgets PAO &gt; PAG</td></tr><tr><td>Strictly Increasing</td><td>45.41 ± 31.32</td><td>87.65 ± 7.82</td></tr><tr><td>Random</td><td>96.70 ± 3.15</td><td>99.55 ± 0.26</td></tr><tr><td>Strictly Decreasing</td><td> $9 9 . 9 4 \pm 0 . 0 2$ </td><td>44.44 ± 29.59</td></tr><tr><td>Heterogeneous Increasing</td><td>47.92 ± 26.15</td><td>94.17 ± 3.83</td></tr></table>

In settings with strictly increasing marginal benefits, allowing reallocation across budgets yields frequent improvements over PAG, but does not translate into consistent gains over IFP. In contrast, when marginal benefits are random and heterogeneous across levels, relaxing the nested (prefix-based) constraint enables PAO to consistently dominate both IFP and PAG across nearly all budget levels. In this regime, the ability to reallocate across individuals and levels is critical, as the globally optimal sequence of marginal increments does not follow a simple prefix structure. When marginal benefits are strictly decreasing, PAO almost always outperforms IFP but ofers no systematic advantage over PAG, reflecting the fact that optimal allocations in this setting favor concentrating resources early rather than flexibly reallocating across individuals. In this case, the opti mal policy is naturally nested, so flexibility provides little additional benefit. Finally, in the heterogeneous strictly increasing setting, the conclusions are similar to the strictly increasing setting: allowing reallocation leads to consistent improvements over PAG at individual budgets but not over IFP.

## 4. Discussion and Conclusion

## 4.1. PAO Versus PAG

As mentioned in Section 2.5, PAO is uniquely diferent from the other policies in that it does not have a direct interpretable ranking that can be derived from the policy allocation. Except in the strictly decreasing setting, PAO outperforms PAG at the majority of budget levels, with particularly strong dominance in the random marginal benefit setting. This improvement comes at the cost of interpretability. Because allocations produced by PAO are not nested, following the policy at budget β does not guarantee that the treatment chosen at budget β+1 extends the previous allocation in a greedy manner. In particular, maximizing realized average treatment efect may require reallocating budget away from partially treated individuals toward initiating treatment for others, breaking the prefix structure that underlies ranking-based approaches such as PAG.

When individuals can receive multiple treatment intensities, optimal allocation may require switching between individuals and treatment levels as the bud get evolves, rather than committing to a fixed ordering. PAO captures this flexibility, demonstrating that in multi-level intervention problems, maximizing realized treatment efect can necessitate departing from nested ranking structures that are suficient in the binary AUTOC setting.

## 4.2. Time Analysis

To evaluate the computational cost of each ranking, we measure the runtime of each ranking on a strictly increasing marginal benefit dataset of depth $L = 4$ and varying population size n. Figure 5 and Table 9 report the time required to generate a full ranking for each policy at each size n.

Figure 5: Empirical runtime scaling of ranking policies as the number of individuals n increases, measured on strictly increasing marginal benefit datasets with L = 4.  
![](images/d3e95a89197e71e824618098c03e241c27c5393f2a0fc13457d6b45462b81691.jpg)

The naive ranking policies of LFP, IFP, and PGP exhibit negligible runtimes across the dataset sizes considered. While these policies difer in how rankings are constructed, each reduces the multi-level allocation problem to a small number of comparisonbased sorting operations, with the number of sorts bounded above by the treatment depth L. As a result, their computational cost is dominated by standard sorting steps whose total complexity scales nearlinearly in the number of individuals. For fixed treatment depth L, this leads to efective runtimes that grow approximately on the order of O(n log n), consistent with the empirical timing results shown in Figure 5. A detailed breakdown of the time complexity for each policy is provided in Appendix A.7.

Between PAG and PAO, PAG consistently achieves a lower runtime. This diference is substantial, with PAG running orders of magnitude faster than PAO in practice. This gap arises from the dynamic programming procedure used by PAO to compute a globally optimal allocation independently at each budget level. While our results show that this flexibility can improve solution quality, it introduces substantial computational overhead, limiting its usefulness.

These results highlight a clear tradeof between computational eficiency and allocation flexibility. While PAO provides the most reactive policy by reallocating every budget level, its runtime makes it impractical for large-scale cases. In contrast, PAG scales comparably to the naive ranking policies while still enforcing precedence constraints. As a result, PAG likely is the preferred policy in applications where computational constraints are binding, and rankings must be generated at scale.

## 4.3. When Does Flexibility Matter?

Our experiments suggest that the value of flexibility in allocation relies on the structure of marginal benefits. Specifically, to what extent the optimal decision at multiple budget levels β in a cohort requires switching treatments.

In the strictly increasing marginal benefit setting (heterogeneous and homogeneous), treating higher levels as early as possible is the most beneficial to overall score. This intuition is verified by our empirical observation that several policies exhibited the same behavior: treating an individual fully once started. This behavior reduced the importance of reallocation as treating a higher level provided more benefit than reallocating. Consequently, PAG and PAO do not achieve a meaningful improvement over strong naive baselines such as IFP.

In the random marginal benefit setting, early com mitments to treating marginals can become costly. For example, treating an individual to an intermediate level may be dominated by initiating a full treatment on a diferent individual during a subsequent budget level. This benefit structure is precisely the regime where the nested (prefix-based) constraint becomes binding. PAG can perform well at early budgets by selecting the best feasible next step greedily, but its nested allocation requirement limits its ability to revise earlier choices as the budget grows. By recomputing the allocation at each budget, PAO can reallocate away from partially treated individuals when a better frontier emerges, which explains its consistent pointwise dominance across budgets.

In the strictly decreasing marginal benefit setting, most of the valuable treatment marginals are able to be treated first, and therefore the best corresponding policy exhausts earlier treatment levels across many individuals. Because later levels are consistently less valuable than earlier levels, there is little benefit to reallocation. This explains why PAO ofers no systematic advantage over PAG in this regime, even though it can still outperform IFP due to IFP’s commitment to fully treating individuals.

Overall, these findings clarify when multi-level allocation meaningfully departs from the binary treatment setting. In binary allocation, policies naturally correspond to their ranking output, so a policy to allocate treatment efects further, such as a policy in our case, is unnecessary. In the multilevel setting, this equivalence can break down. When marginal benefits vary across treatment levels, maximizing realized treatment efect may require policies that change which individuals they prioritize as the budget evolves. In such cases, allocation cannot be represented by a single nested ranking, and flexible policies such as PAO provide a structural advantage.

## 4.4. Non-Nested Allocations In The Clinical Setting

A potential limitation of non-nested allocation poli cies such as PAO is their lack of interpretability and feasibility in real-world clinical settings. For example, in a setting where a hospital has already allocated treatment up to its current budget, the arrival of an additional unit of budget introduces ambiguity in how that resource should be deployed. Unlike nested policies, which extend naturally by following a fixed ranking, PAO may require re-solving the allocation problem entirely, potentially revising earlier decisions. This raises a real question of whether an incremental increase should trigger a reallocation or whether they should remain stable once allocated.

In practice, we propose that PAO should not be implemented as a full dynamic reallocation at each budget increment. Rather, allocation decisions can be made at regular intervals, such as shift changes in hospitals, when resources are known or fixed. This avoids the need to re-optimize allocations in response to small, incremental changes in budget, where solving a full allocation problem to assign a single additional treatment may be impractical.

We encourage this problem to be explored further in future work, particularly in designing policies that balance allocation flexibility with the need for stabil ity and interpretability in clinical environments.

## 4.5. On The Use Of Non-Synthetic Data

A natural question that arises at the end of this research is whether the findings in this paper can be extended to real-world data. While evaluating allo cation policies on observational data is highly desir able, the implementation of the multi-level treatment setting on non-synthetic data is dificult.

First, the evaluation of allocation policies in our framework requires access to individual-level marginal treatment efects for each treatment level. In real-world datasets, these quantities are inherently unobserved. Estimating these quantities requires strong assumptions of consistency, unconfoundedness, and overlap across all treatment levels (Rubin, 1974). In multi-level settings, the overlap assumption becomes particularly restrictive, as each individual must have a non-zero probability of receiving every treatment intensity.

Even when these assumptions hold, estimating marginal treatment efects at multiple levels introduces statistical noise. Because the multi-level setting requires estimating a sequence of incremental efects, the errors in the compounding of these estimates make it dificult to disentangle where policy performance comes from.

That said, extending this work to non-synthetic data is important future work. One promising approach could be the use of semi-synthetic data where real covariates and treatment assignments are combined with simulated outcome models. More broadly, applying the ML-AUTOC framework with high-quality causal estimators and explicitly accounting for estimation uncertainty will be critical for understanding how these allocation policies perform in practice.

## 4.6. Conclusion

This paper investigates whether precedence-aware greedy and reallocation policies outperform naive ranking-based policies in terms of achieving higher realized total average treatment efect in multi-level treatment settings. Our findings show that reallocation for optimal benefit (PAO) is not necessary to achieve strong performance. In settings with strictly ordered marginal benefits, PAG achieves performance comparable to naive ranking-based policies.

However, in settings where there are random marginal benefits for diferent treatment levels, the relaxation of the nesting constraint for PAO is shown to have higher pointwise performance across diferent budgets, indicating that in these settings, it is important to maximize the realized treatment efect by reallocating resources rather than relying on a naive approach.

## Acknowledgments

This work was completed as part of the Honors Thesis of Olin Gilster (https://etd.library.emory.edu/ concern/etds/5138jg33v). The authors would like to thank Ruoxuan Xiong and Michelangelo Grigni for their helpful feedback.

## AI Use Statement

We used generative AI tools to assist with copyediting of the final manuscript. The authors take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Efren P. Agonafer, Susan L. Carson, Victoria Nunez, Fiona Jones, Lori Carter-Edwards, Oluwasegun Adekeye, Jessica Resendez, Marsha R. Weintraub, J. Nwando Olayiwola, and Arleen F. Brown. Community-based organizations’ perspectives on improving health and social service integration. BMC Public Health, 2021. doi: 10. 1186/s12889-021-10449-w. URL https://doi.org/ 10.1186/s12889-021-10449-w.

Ayse Aslan, Evrim Ursavas, and Ward Romeijnders. A precedence constrained knapsack problem with uncertain item weights for personalized learning

systems. Omega, 115:102779, 2023. ISSN 0305- 0483. doi: https://doi.org/10.1016/j.omega.2022. 102779. URL https://www.sciencedirect.com/ science/article/pii/S0305048322001864.

Richard Bellman. Dynamic Programming. Dover Publications, 1957. ISBN 9780486428093.

Dimitris Bertsimas and John N. Tsitsiklis. Introduction to Linear Optimization. Athena Scientific, 1997.

Elizabeth C. Caniglia, Erica E. M. Moodie, Roger W. Logan, and Miguel A. Hern´an. Estimating optimal dynamic treatment strategies under resource constraints using dynamic marginal structural models. Statistics in Medicine, 2021. doi: 10.1002/sim. 9177. URL https://pmc.ncbi.nlm.nih.gov/articles/ PMC9017598/.

Michael F. Drummond, Mark J. Sculpher, Karl Claxton, Greg L. Stoddart, and George W. Torrance. Methods for the Economic Evaluation of Health Care Programmes. Oxford University Press, 2015.

G. French, M. Hulse, D. Nguyen, K. Sobotka, B. Webster, J. Corman, B. Aboagye-Nyame, M. McNulty, K. Klinker, G. Dumyati, and et al. Impact of hospital strain on excess deaths during the covid-19 pandemic — united states, july 2020–july 2021. MMWR. Morbidity and Mortality Weekly Report, 2021. URL https://www.cdc.gov/mmwr/volumes 70/wr/mm7046a5.htm.

Arnaud Fr´eville. The multidimensional 0–1 knapsack problem: An overview. European Journal of Operational Research, 155(1):1–21, 2004.

Nicole B. Gabler, Sarah J Ratclife, Jason Wagner, David A. Asch, Gordon D. Rubenfeld, Derek C. Angus, and Scott D. Halpern. Mortality among patients admitted to strained intensive care units. Am J Respir Crit Care Med, 2013. URL https: //pmc.ncbi.nlm.nih.gov/articles/PMC3826272/.

T. C. Hu. Parallel sequencing and assembly line problems. Operations Research, 9(6):841–848, 1961. ISSN 0030364X, 15265463. URL http://www.jstor. org/stable/167050.

Fahad Kamran, Maggie Makar, and Jenna Wiens. Learning to rank for optimal treatment allocation under resource constraints. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, pages 1–18. PMLR,

2024. URL https://proceedings.mlr.press/v238/ kamran24a/kamran24a.pdf.

Hans Kellerer, Ulrich Pferschy, and David Pisinger. Knapsack Problems. Springer Berlin Heidelberg, Berlin, Heidelberg, 2004. ISBN 3-540-40286-1, 978- 3-540-40286-2. doi: 10.1007/978-3-540-24777-7.

Donald Knuth. The Art Of Computer Programming, vol. 3: Sorting And Searching. Addison-Wesley, 1973.

S¨oren R. K¨unzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment efects using machine learning. Proceedings of the National Academy of Sciences, 116(10):4156–4165, 2019. doi: 10.1073/ pnas.1804597116. URL https://www.pnas.org doi/10.1073/pnas.1804597116.

Silvano Martello and Paolo Toth. Knapsack Problems: Algorithms and Computer Implementations. Wiley, 1990.

Eduardo Moreno, Daniel Espinoza, and Marcos Goycoolea. Large-scale multi-period precedence con strained knapsack problem: A mining application. Electronic Notes in Discrete Mathematics, 36:407– 414, 08 2010. doi: 10.1016/j.endm.2010.05.052.

David Pisinger. Where are the hard knapsack problems? Computers & Operations Research, 32(9): 2271–2284, 2005.

Donald B. Rubin. Estimating causal efects of treatments in randomized and nonrandomized stud ies. Journal of Educational Psychology, 66(5):688– 701, 1974. doi: 10.1037/h0037350. URL https: //doi.org/10.1037/h0037350.

Mehran Samavati, Daryl Essam, Micah Nehring, and Ruhul Sarker. A methodology for the largescale multi-period precedence-constrained knapsack problem: an application in the mining industry. International Journal of Production Economics, 193:12–20, 2017. ISSN 0925- 5273. doi: https://doi.org/10.1016/j.ijpe.2017.06. 025. URL https://www.sciencedirect.com/science/ article/pii/S0925527317301986.

Uri Shalit, Fredrik D Johansson, and David Sontag. Estimating individual treatment efect: generalization bounds and algorithms. In International conference on machine learning, pages 3076–3085. PMLR, 2017.

Rahim Vakili, Saba Vakili, Maryam Ajilian Abbasi, Shokofeh Masoudi, and Masumeh Saeidi. Overcrowded classrooms: Challenges, consequences, and collaborative solutions for educators: A literature review. Medical Education Bulletin, 2024. URL http://medicaleducation-bulletin.ir/ article 210691.html.

Steve Yadlowsky, Scott Fleming, Nigam Shah, Emma Brunskill, and Stefan Wager. Evaluating treatment prioritization rules via rank-weighted average treatment efects. Journal of the American Statistical Association, 120(549):38–51, 2025.

Lei Zhao, Lu Tian, Tianxi Cai, Brian Claggett, and Lee-Jen Wei. Efectively selecting a target population for a future comparative study. Journal of the American Statistical Association, 108(502): 527–539, 2013. URL https://doi.org/10.1080/ 01621459.2013.770703.

## Appendix A.

## A.1. Heterogeneous Treatment Costs

The primary analysis in this paper assumes that each marginal treatment level incurs a unit cost. While this simplifies the allocation problem and isolates the role of marginal benefit structure, it abstracts away an important feature of many real-world intervention systems: treatment intensity often varies in cost.

In practice, higher levels of intervention typically require greater resource expenditure. For example, escalating clinical treatment intensity may require additional staf time, specialized services, or longer duration of care (Drummond et al., 2015). Cost heterogeneity is therefore not merely a modeling convenience but reflects structural properties of real-world resource allocation problems. Allowing treatment levels to difer in cost aligns the framework more closely with standard welfare maximization and cost-efectiveness formulations in economics and operations research, where decision-makers face a fixed budget and must allocate heterogeneous interventions to maximize total benefit (Bertsimas and Tsitsiklis, 1997).

To start, let $c _ { \ell } > 0$ denote the marginal cost of advancing an individual from treatment level $\ell - 1$ to level ℓ. We define the cumulative cost of assigning individual i to level ℓ as

$$
C ( \ell ) = \sum _ { k = 1 } ^ { \ell } c _ { k } ,
$$

which is weakly increasing in ℓ. Given marginal benefits $\delta _ { i } ^ { ( \ell ) }$ as defined in Equation (1), the allocation problem under a fixed total budget B becomes:

$$
\operatorname* { m a x } _ { \{ z _ { i , \ell } \} } \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { L } \delta _ { i } ^ { ( \ell ) } z _ { i , \ell }
$$

subject to

$$
\sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { L } c _ { \ell } z _ { i , \ell } \leq B , \qquad z _ { i , \ell } \leq z _ { i , \ell - 1 } \forall i , \ell \geq 2 , \qquad z _ { i , \ell } \in \{ 0 , 1 \} .
$$

The second constraint enforces precedence: an individual may only receive level ℓ if all prior levels have been assigned. The problem now resembles a precedence-constrained knapsack problem, where each marginal treatment increment yields benefit $\delta _ { i } ^ { ( \ell ) }$ but consumes $c _ { \ell }$ units of budget (Pisinger, 2005). The individual optimizing must therefore trade of both benefit magnitude and cost-efectiveness.

Under unit costs, the budget parameter $\beta$ directly corresponds to the number of marginal treatment increments allocated. Each step in the ML-TOC curve therefore reflects one additional unit of treatment. With heterogeneous costs, this interpretation changes. The budget B now measures total expenditure rather than number of treatment increments. A single additional allocation may consume multiple units of budget, and diferent increments may advance the budget frontier by unequal amounts. Consequently:

1. The horizontal axis of ML-TOC curves would represent cumulative expenditure rather than count of treatment steps.

2. Budget increments become non-uniform.

3. The mapping between allocation rank and budget becomes nonlinear.

We believe that introducing heterogeneous costs should increase the importance of flexible, non-nested allocation policies. Under unit costs, the primary trade-of concerns only marginal benefit magnitude. However, when costs difer, optimal allocation depends on benefit-to-cost ratios $\delta _ { i } ^ { ( \ell ) } / c _ { \ell }$ . If higher treatment levels have disproportionately high costs, it may become optimal to partially treat multiple individuals rather than fully treat a small number. Conversely, if early levels are inexpensive but later levels are costly, the optimal policy may exhibit switching behavior across individuals to maximize aggregate benefit per dollar spent. We therefore hypothesize:

When treatment costs are heterogeneous, precedence-aware policies that allow reallocation (e.g., PAO) will outperform ranking-based or nested greedy policies by a larger margin than in the unitcost setting.

In particular, heterogeneous costs introduce an additional dimension along which early allocation decisions may become suboptimal at later budget levels, increasing the value of dynamic reoptimization.

## A.2. Naive Policy Allocation Matrices

Naive policies in the multi-level treatment framework benefit from collapsing the multi-level treatment structure into a one-time ordering problem.

We illustrate this flattening using the toy marginal benefit matrix in Table 7.

Table 7: True marginal benefit matrix (see also Table 1)
<table><tr><td rowspan=1 colspan=1> $\overline { { \delta _ { i } ^ { ( \ell ) } } }$ </td><td rowspan=1 colspan=1>l = 1</td><td rowspan=1 colspan=1>l = 2</td></tr><tr><td rowspan=1 colspan=1>x1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>x2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr></table>

Each policy maps this $2 \times 2$ matrix into a length-4 ranking over treatment actions (i, ℓ), efectively flattening the multi-level structure.

## 1. LFP

LFP sorts individuals independently within each treatment level and then concatenates the level-wise rankings. For ℓ = 1, we have $x _ { 2 } \succ x _ { 1 }$ since $2 > 1$ ; for $\ell = 2$ , we have $x _ { 1 } \succ x _ { 2 }$ since $5 > 3$ . The resulting flattened allocation matrix is:

$$
{ \left[ \begin{array} { l } { ( x _ { 2 } , 1 ) } \\ { ( x _ { 1 } , 1 ) } \\ { ( x _ { 1 } , 2 ) } \\ { ( x _ { 2 } , 2 ) } \end{array} \right] } \quad { \mathrm { w i t h ~ v a l u e s } } \quad [ 2 , \ 1 , \ 5 , \ 3 ] .
$$

## 2. IFP

IFP first aggregates marginal benefits across levels for each individual,

$$
\sum x _ { 1 } = 1 + 5 = 6 , \qquad \sum x _ { 2 } = 2 + 3 = 5 ,
$$

and ranks individuals by this total score. Treatment actions are then expanded in precedence order for each individual. The resulting flattened allocation matrix is:

$$
\begin{array} { r } { \left[ \begin{array} { l } { ( x _ { 1 } , 1 ) } \\ { ( x _ { 1 } , 2 ) } \\ { ( x _ { 2 } , 1 ) } \\ { ( x _ { 2 } , 2 ) } \end{array} \right] \quad \mathrm { w i t h ~ v a l u e s } \quad [ 1 , \ 5 , \ 2 , \ 3 ] . } \end{array}
$$

## 3. PGP

PGP ignores precedence at ranking time and globally sorts all (i, ℓ) pairs by their marginal benefit values. Sorting {1, 5, 2, 3} in descending order yields:

$$
\left[ { \begin{array} { l } { ( x _ { 1 } , 2 ) } \\ { ( x _ { 2 } , 2 ) } \\ { ( x _ { 2 } , 1 ) } \\ { ( x _ { 1 } , 1 ) } \end{array} } \right] \quad { \mathrm { w i t h ~ v a l u e s } } \quad [ 5 , ~ 3 , ~ 2 , ~ 1 ] .
$$

A.3. LFP Code   
# Level-First Policy (LFP)   
initialize empty list ranking   
for each level l from 1 to L:   
# sort individuals by marginal benefit at this level   
create list of individuals sorted by delta[i][l] in descending order   
for each individual i in this sorted list:   
append (i, l) to ranking   
return ranking   
A.4. IFP Code   
# Individual-First Policy (IFP)   
initialize empty list ranking   
# compute total value for each individual   
for each individual i:   
total[i] = sum over l of delta[i][l]   
# sort individuals by total value   
create list of individuals sorted by total[i] in descending order   
# assign all levels for each individual before moving on   
for each individual i in sorted list:   
for each level l from 1 to L:   
append (i, l) to ranking   
return ranking   
A.5. PGP Code   
# Pooled Global Policy (PGP)   
initialize empty list of pairs   
# treat every (i, l) as a separate item   
for each individual i:   
for each level l:   
add (i, l) to list   
# sort all pairs by marginal benefit   
sort list of pairs by delta[i][l] in descending order   
ranking = sorted list   
return ranking

A.6. PAG Code   
# Precedence-Aware Greedy (PAG)   
initialize empty ranking list   
# priority queue stores (i, l) with priority delta[i][l]   
initialize empty max-priority queue   
# initially, only level 1 is feasible for each individual   
for each individual i:   
push (i, 1) into priority queue with priority delta[i][1]   
# iterate over full budget   
for b = 1 to nL:   
# select best feasible marginal   
(i, l) = pop max from priority queue   
add (i, l) to ranking   
# unlock next level for that individual   
if l + 1 <= L:   
push (i, l+1) into priority queue   
with priority delta[i][l+1]   
return ranking

## A.7. Big-Oh Analysis

We analyze the computational complexity of the naive ranking policies as a function of the number of individuals n and treatment levels L. Our budget β represents our total treatment actions.

Our analysis relies on well-known results for comparison-based sorting that establish a worst-case lower bound of Ω(k log k) comparisons for sorting k keys, with matching O(k log k) upper bounds achievable by standard algorithms (Knuth, 1973).

## 1. LFP

The implementation of LFP in Appendix A.3 consists of an outer loop over levels ℓ = 1, . . . , L.

The policy computes a sort for each level which costs O(n log n) under comparison-based sorting. The subsequent inner loop iterates once over the sorted indices and writes n entries into the ranking array, costing O(n). Summing over all levels yields O(L n log n). Since the array writes are asymptotically dominated by the sorting step, the total running time of LFP is:

$$
O ( n L \log n )
$$

## 2. IFP

The implementation of IFP in Appendix A.4 first computes an aggregate score for each individual by summing marginal benefits across all levels. This requires iterating over the full (n × L) matrix once, costing O(nL).

The policy then sorts the n aggregate scores, which costs O(n log n) under comparison-based sorting. Finally, the nested loop expands each individual into its L treatment actions and writes nL entries into the ranking array, costing O(nL). Combining terms yields a total running time of

$$
O ( n L + n \log n ) .
$$

## 3. PGP

The implementation of PGP in Appendix A.5 first flattens the $( n \times L )$ marginal benefit matrix into a length nL vector, which costs $O ( n L )$

The policy then sorts these marginal benefit values, which costs $O ( n L \log ( n L ) )$ under comparison-based sorting. Constructing the ranking array from the sorted indices requires writing nL entries and therefore costs $O ( n L )$ . The sorting step dominates, so the total running time of PGP is $O ( n L \log ( n L ) )$ ).

## 4. PAG

The implementation of PAG in Appendix A.6 selects treatment actions sequentially over the full budget horizon of nL marginal increments. Rather than scanning all n individuals at each step, the algorithm maintains the current feasible frontier in a priority queue. Initially, the queue contains the first feasible marginal benefit for each of the n individuals.

At each budget step, the algorithm removes the maximum feasible marginal benefit from the priority queue, which costs O(log n). If the selected action corresponds to level ℓ for individual i, then level $\ell + 1$ for that same individual becomes newly feasible and is inserted into the priority queue, which also costs $O ( \log n )$ . Thus, each budget step requires at most one extract-max and one insert operation, for a total cost of O(log n) per step. Since there are nL total treatment actions, and initializing the priority queue costs $O ( n )$ (dominated by the main loop), the total running time is

$$
O ( n L \log n ) .
$$

## 5. PAO

The implementation of PAO first computes prefix sums across treatment levels for each individual, which costs $O ( n L )$ . The policy then applies a dynamic program over individuals and budgets nL, updating an array of length nL for each of L possible allocation sizes per individual. This yields a total dynamic programming cost of $O ( n L n L )$ . Therefore, the overall running time of PAO is

$$
O ( n ^ { 2 } L ^ { 2 } ) .
$$

## A.8. Big-Oh Table

Table 8: Asymptotic runtime of ranking and precedence-aware policies as a function of the number of individuals n and treatment levels L.
<table><tr><td>Policy</td><td>Runtime (Big-Oh)</td></tr><tr><td>PGP</td><td> $\overline { { O ( n L \log ( n L ) ) } }$ </td></tr><tr><td>IFP</td><td> $O ( n L + n \log n )$ </td></tr><tr><td>LFP</td><td> $O ( n L \log n )$ </td></tr><tr><td>PAG PAO</td><td> $O ( n L \log n )$   $O ( n ^ { 2 } L ^ { 2 } )$ </td></tr></table>

## A.9. Dynamic Programming

Dynamic programming was used to implement the PAO policy. The formulation is closely related to classical knapsack-style dynamic programming, in which a finite resource budget is allocated across multiple items to maximize total value (Bellman, 1957; Kellerer et al., 2004).

In this setting, each individual corresponds to a group of mutually exclusive choices indexed by $t _ { i } ~ \in$ $\{ 0 , 1 , \ldots , L \}$ , and the total budget is nL (Kellerer et al., 2004). Dynamic programming provides an exact solution by tracking the optimal achievable benefit for each intermediate budget level.

In order to use this approach, prefix sums of marginal benefits are computed, which allows for a constanttime computation of the assignment of treatment levels to a particular individual. The dynamic program iterates through individuals and prefix lengths, updating the value function on all budgets. The transition of the value function is shown in the following code:

\# PAO: Dynamic programming over individuals and budgets

```ini
n = number of individuals
L = number of treatment levels
B = n * L
# Step 1: compute prefix values
# prefix[i][m] = total value of assigning the first m levels to individual i
initialize prefix as an n x (L + 1) table of zeros
for each individual i:
prefix[i][0] = 0
for each level m from 1 to L:
prefix[i][m] = prefix[i][m - 1] + delta[i][m]
# Step 2: initialize dynamic program
# dp_prev[b] = best total value achievable using processed
# individuals and budget b
initialize dp_prev[0] = 0
initialize dp_prev[b] = -infinity for all b > 0
# choice[i][b] will store how many levels were assigned to individual i
# in the optimal solution at budget b during the i-th step
initialize choice as an n x (B + 1) table of zeros
# Step 3: process individuals one at a time
for each individual i:
# start with the option of assigning no treatment to this individual
best[b] = dp_prev[b] for every budget b
best_m[b] = 0 for every budget b
# consider assigning a prefix of length m to individual i
for each prefix length m from 1 to L:
# candidate value if m treatment units are assigned to individual i
for each budget b from m to B:
cand[b] = dp_prev[b - m] + prefix[i][m]
# if this improves the best known value at budget b, update it
if cand[b] > best[b]:
best[b] = cand[b]
best_m[b] = m
# store the chosen prefix length for this individual at each budget
for each budget b from 0 to B:
choice[i][b] = best_m[b]
```

```python
# move to the next individual
dp_prev = best
# Step 4: recover allocations by backward pass
# For each budget beta, trace backward through choice to determine
# how many levels were assigned to each individual
for each budget beta from 1 to B:
remaining_budget = beta
for individuals i in reverse order:
m = choice[i][remaining_budget]
assign levels 1 through m to individual i
remaining_budget = remaining_budget - m
```

To recover feasible allocations, the selected prefix length for each individual and budget is stored during the forward pass and used in a backward pass to reconstruct the allocation corresponding to each budget level.

## A.10. Dataset Generation

In the definitions below, $\mu$ is the mean of the per-level benefit draws in Datasets A and B, and $\sigma _ { s }$ is the scale parameter of the log-normal distribution over individual scales in Dataset D. The specific values used in our experiments are set in the data generation scripts available at https://github.com/gilstero/ML-TOC.

Dataset A: Strictly Increasing Marginal Benefits

$$
\begin{array} { c } { { z _ { i } \sim \mathcal { N } ( \mu { \bf 1 } _ { L } , 0 . 5 ^ { 2 } I _ { L } ) } } \\ { { \ } } \\ { { \delta _ { i } ^ { ( 1 ) } = \operatorname* { m a x } \{ 0 , z _ { i } ^ { ( 1 ) } \} } } \\ { { \ } } \\ { { \delta _ { i } ^ { ( \ell ) } = \operatorname* { m a x } \left\{ z _ { i } ^ { ( \ell ) } , \delta _ { i } ^ { ( \ell - 1 ) } + 1 \right\} ~ \mathrm { f o r } ~ \ell = 2 , \ldots , L } } \end{array}
$$

Dataset B: Random Marginal Benefits

$$
\begin{array} { c } { { z _ { i } \sim \mathcal { N } ( \mu { \bf 1 } _ { L } , 0 . 5 ^ { 2 } I _ { L } ) } } \\ { { { } } } \\ { { \delta _ { i } ^ { ( \ell ) } = \operatorname* { m a x } \{ 0 , \ z _ { i } ^ { ( \ell ) } \} ~ \mathrm { f o r } ~ \ell = 1 , \ldots , L } } \end{array}
$$

Dataset C: Strictly Decreasing Marginal Benefits

Dataset C is constructed as the mirror image of Dataset A. For each individual i, the strictly increasing marginal benefit sequence $\{ \delta _ { i } ^ { ( 1 ) } , \ldots , \delta _ { i } ^ { ( L ) } \}$ generated under Dataset A is reversed to obtain $\{ \delta _ { i } ^ { ( \bar { L } ) } , \dots , \delta _ { i } ^ { ( 1 ) } \}$ , yielding strictly decreasing marginal benefits across treatment levels.

Dataset D: Heterogeneous Strictly Increasing Marginal Benefits

$$
s _ { i } \sim \mathrm { L o g N o r m a l } ( 0 , \sigma _ { s } ^ { 2 } )
$$

$$
z _ { i } ^ { ( \ell ) } \sim \mathcal { N } ( s _ { i } , 0 . 5 ^ { 2 } )
$$

$$
\delta _ { i } ^ { ( 1 ) } = \operatorname* { m a x } \{ 0 , z _ { i } ^ { ( 1 ) } \}
$$

$$
\delta _ { i } ^ { ( \ell ) } = \operatorname* { m a x } \left\{ z _ { i } ^ { ( \ell ) } , \ : \delta _ { i } ^ { ( \ell - 1 ) } + 1 \right\} \quad \mathrm { f o r } \ell = 2 , \ldots , L
$$

Each individual i is assigned a scale $s _ { i } ,$ which determines how large their treatment efects are overall. The values $z _ { i } ^ { ( \ell ) }$ are noisy draws of the marginal benefit at each level, centered around $s _ { i } .$ The first level is truncated to be nonnegative, and each subsequent level is forced to be at least one unit larger than the previous level. This ensures that marginal benefits strictly increase for each individual, while allowing diferent individuals to have very diferent magnitudes of treatment efects.

## A.11. Strictly Decreasing Aggregate Plot

![](images/b299dbe19bf727f1b92df5175f49c29e4c2cd7b0b941a15c6c4c1a47fff866ae.jpg)  
Figure 6: Average TOC curves across budgets for the strictly decreasing marginal benefit dataset. Curves are averaged over 100 datasets and plotted relative to a random baseline. ± one standard deviation error bars are plotted but too small to be visually distinguishable.

## A.12. Run Time Data

Table 9: Runtime (seconds) of ranking policies as a function of dataset size n on strictly increasing marginal benefit datasets with L = 4.
<table><tr><td>n</td><td>PGP</td><td>IFP</td><td>LFP</td><td>PAG</td><td>PAO</td></tr><tr><td>500</td><td>0.0008</td><td>0.0009</td><td>0.0009</td><td>0.0041</td><td>0.6721</td></tr><tr><td>1,000</td><td>0.0017</td><td>0.0017</td><td>0.0017</td><td>0.0086</td><td>2.6607</td></tr><tr><td>2,000</td><td>0.0033</td><td>0.0033</td><td>0.0033</td><td>0.0178</td><td>10.7443</td></tr><tr><td>4,000</td><td>0.0069</td><td>0.0069</td><td>0.0069</td><td>0.0483</td><td>43.1270</td></tr><tr><td>8,000</td><td>0.0143</td><td>0.0135</td><td>0.0135</td><td>0.1529</td><td>176.8498</td></tr></table>

## A.13. Software

To enable the repeatability and eficiency in the experimental phase of this research, we have developed an accompanying software that allows for ML-AUTOC demonstrations available at https://mlautoc.com. As stated in the docs, the goal of the software was to facilitate the following:

1. Choose the number of patients in the dataset.

2. Select the number of marginal treatment levels each patient has.

3. Experiment with the cost structure of the experiment.

4. Configure the underlying structure of marginal treatment efects.

5. Optionally upload an individual dataset (CSV) and run an experiment.

6. View resource allocation graphs and ML-AUTOC scores.

7. Analyze runtime performance (time logged and displayed as a graph).

The platform is implemented using Next.js, and the backend runs locally on the experimenter’s computer. Experiments are executed using common Python libraries to ensure full reproducibility. Source code and documentation are available on the project website under https://mlautoc.com/experiments.

During research we developed several functions that allow us to generate, rank, and visualize controlled treatment allocation scenarios prior to web deployment. These utilities were used to test our theories on flip rankings, precedence enforcement, and cost heterogeneity before web implementation. These functions can be downloaded as files under https://mlautoc.com/downloads. If the website is inaccessible, all available code can be found on GitHub at https://github.com/gilstero/ML-TOC.