# Solving Few-Shot Multiobjective Multitask Optimization via Iterative Sequential Transfer

Tingyang Wei College of Computing and Data Science Nanyang Technological University Singapore tingyang001@e.ntu.edu.sg

Zhao Wei Centre for Frontier AI Research (CFAR) A\*STAR Singapore wei zhao@a-star.edu.sg

Haofeng Wu College of Computing and Data Science Nanyang Technological University Singapore haofeng.wu@ntu.edu.sg

Jiao Liu College of Computing and Data Science Nanyang Technological University Singapore jiao.liu@ntu.edu.sg

Ananda Phan Iman Department of AI Convergence Gwangju Inst. of Sci. & Tech. (GIST) South Korea anandaphan@gm.gist.ac.kr

Yew-Soon Ong College of Computing and Data Science Nanyang Technological University Singapore ASYSOng@ntu.edu.sg

Abstract—Applying knowledge transfer across multiple optimization tasks, multitask optimization (MTO) emerges as a promising approach to solving synergistic optimization tasks simultaneously. However, the development of effective knowledge transfer mechanisms in MTO fundamentally relies on aligning elite solution distributions across tasks. This dependency creates a critical bottleneck infew-shot optimization regimes, as restricted evaluation budgets impede the identification of elite solution distributions required for beneficial transfer. This challenge is exacerbated in multiobjective multitask problems, where each optimizer must approximate a continuous Pareto manifold rather than a single optimal point. This paper introduces Iterative Sequential Transfer (IST) to circumvent this bottleneck. We model MTO as a sequence of sequential transfer optimization problems, concentrating evaluations on a single target per iteration. We propose a likelihood-informed task prioritization mechanism to maximize transfer utility by identifying the task most likely ready for knowledge integration. Empirical results on benchmark and real-world problems verify the effectiveness of the proposed method under tight budgets.

Index Terms—Transfer optimization, evolutionary multitask, transfer evolutionary optimization, Gaussian process, multiobjective optimization.

## I. INTRODUCTION

Transfer optimization [1] has garnered significant attention in recent years as a novel approach to optimization problems by fully leveraging inter-task relationships. Considering that real-world optimization problems seldom exist in isolation [2], transfer optimization methods are designed to avoid optimizing given tasks from scratch and to alleviate the excessive computational burden. Several conceptual realizations of the transfer optimization paradigm, including sequential transfer optimization (STrO) [3], [4], multitask optimization [5]–[7], and multiform optimization [8], have spawned numerous studies in the context of knowledge transfer.

In particular, multitask optimization (MTO) [5], [7] emerges as a ubiquitous approach to solving multiple optimization tasks simultaneously by exploiting the inter-task synergies. MTO can be formulated as follows:

$$
\operatorname* { m i n } f _ { k } ( \mathbf { x } _ { k } ) , \mathrm { s . t . } \mathbf { x } _ { k } \in \Omega _ { k } , k \in \{ 1 , \ldots , K \} ,\tag{1}
$$

where $\Omega _ { k }$ is the decision space of the k-th optimization problem, and $f _ { k }$ is the objective function for the k-th task. MTO tackles distinct problems simultaneously, thereby aiming at generating outputs $( \mathbf { x } _ { 1 } ^ { * } , \mathbf { x } _ { 2 } ^ { * } , \ldots , \mathbf { x } _ { K } ^ { * } )$ that are the optimal solutions for each task. To achieve this, MTO seeks to develop proper knowledge transfer mechanisms among distinct optimization problems, either implicitly sharing the solution components [5], [6], [9] or explicitly building mapping among task pairs [10], [11]. Benefiting from these knowledge transfer mechanisms, recent years have witnessed significant advances of MTO across a plethora of applications [12] encompassing system-in-package design [13], grid shell design [14], and collision-avoidance control [15].

Albeit the surging advances, the efficacy of these transfer mechanisms fundamentally hinges on the alignment of elite solution distributions across tasks [16]. For knowledge transfer to be beneficial, the source task should provide a high-quality solution distribution [17] in its own search space to accurately guide the target task. Otherwise, stagnant source tasks are likely to trigger detrimental negative transfer [17]. This dependency creates a critical bottleneck in the few-shot optimization regime. Under stringent evaluation budgets, limited evaluations are typically dispersed across all tasks, failing to identify high-quality solutions within any single task and thereby giving rise to stagnated search or negative transfer [18]. This challenge is further intensified in the multiobjective case, where the optimizer must approximate a continuous Pareto manifold rather than a single optimal point [14]. Dispersing a restricted budget across multiple multiobjective search tasks more likely prevents any task from identifying an elite solution set, exacerbating the risks of negative transfer and inefficient resource utilization.

![](images/973d43a8ab8194b380aae2bc1d513f74cc745d93f21fa40497fdfb5df3c9373e.jpg)  
Fig. 1. Workflow of distinct multitask optimization frameworks. (a) Standard multitask optimization (tasks evaluated evenly) (b) The proposed multitask optimization with iterative sequential transfer optimization (tasks evaluated selectively).

To address this, we position sequential transfer optimizers as a practical approach for solving few-shot multi-objective multitask optimization, and provide empirical evidence through an Iterative Sequential Transfer (IST) framework. Sequential transfer optimization (STrO) can be formally denoted as:

$$
\operatorname* { m i n } _ { \mathbf { x } _ { T } \in \Omega _ { T } } ~ f _ { T } ( \mathbf { x } _ { T } ) \quad \mathrm { g i v e n } \quad \{ \mathcal { D } _ { S _ { k } } \} _ { k = 1 } ^ { K } ,\tag{2}
$$

where $\mathcal { D } _ { S _ { k } }$ is the optimization history for the source task $S _ { k }$ including evaluated solutions $\{ ( \mathbf { x } _ { S _ { k } } ^ { ( t ) } , f _ { S _ { k } } ( \mathbf { x } _ { S _ { k } } ^ { ( t ) } ) ) \} _ { t = 1 } ^ { N _ { t } }$ . In this regard, MTO can be explicitly converted into a sequence of STrO problems, where at each iteration we choose a target task $T$ and its source set $\{ S _ { k } \}$ by maximizing the universal transfer utility, so that knowledge transfer is applied only when it is most beneficial. This can potentially alleviate the fewshot challenge. Meanwhile, STrO has recently been developed into a fruitful line of inquiry with both theory-informed formulations [19]–[21] and principled empirical analyses [22], making it a solid foundation for our IST framework. To effectively convert the MTO into a sequence of STrO as per the search dynamics, we introduce a simple likelihood-informed task prioritization mechanism.

In this study, we focus on the intricate multiobjective multitask optimization (MOMTO) under a stringent computational budget. The inherent complexities of multiobjective optimization, coupled with the demands of few-shot optimization, exacerbate the difficulty of MTO. To verify our position, we test the IST framework based on a recent STrO method, forward-inverse transfer evolutionary multiobjective optimizer (F-invTrEMO) [21], in both few-shot MOMTO benchmark and real-world problems. Besides, we impose this IST framework on another potent STrO method, AMTEA [3], showcasing the generality of the proposed method. We position that the IST framework not only can highlight the potential for more effective MTO in few-shot multiobjective domains, but also offers a flexible foundation for blending advances in STrO and MTO.

The relevant source code can be found in the link: https://github.com/ambigeV/stro.

## II. RELATED WORKS

## A. Few-Shot Multiobjective Multitask Optimization

This work addresses few-shot multiobjective MTO (MOMTO) by extending the scalar functions $f _ { k } ( \cdot )$ in (1) to vector-valued functions $F _ { k } ( \cdot )$ with m objectives. Unlike standard MTO benchmarks that allow $O ( 1 0 ^ { 5 } )$ evaluations per task [23], we constrain the budget for each task to $O ( 1 0 ^ { 2 } )$ . This few-shot optimization setting better reflects real-world constraints and demands more efficiency of search and transfer mechanisms. To our best knowledge, the research efforts in this niche area remains limited. However, preliminary approaches include maintaining diverse types of surrogate models [24] or constructing inverse multitask models [21] to tackle multiple expensive multiobjective tasks. The latter method, F-invTrEMO [21], serves as a baseline method in this paper, due to its flexibility for extension.

## B. Sequential Transfer Optimization

STrO, as formalized in (2), leverages optimization experiences from source data or models to expedite the search in a target task. The pioneering method, AMTEA, applied transfer stacking to adaptively combine existing surrogates for target multiobjective problems [3]. Other related studies focus on bridging source and target domains via transfer Gaussian Processes [19], inverse modeling [20], or optimal transport [25]. Motivated by recent theoretical progress [7], [19], [26] and improvements in scalability [4], we propose to solve fewshot MOMTO through iterative sequential transfer, inspired by the structural similarities between MTO and STrO and the few-shot optimization challenge mentioned in Section I.

## III. MULTITASK OPTIMIZATION VIA ITERATIVE SEQUENTIAL TRANSFER

This section delineates the proposed IST framework. We first elucidate the workflow of the IST framework. Next, we introduce the base sequential transfer optimizer utilized in this study: F-invTrEMO. By integrating this optimizer within the IST framework, a likelihood-informed task prioritization mechanism is proposed. This approach effectively models multitask optimization as a sequence of iterative sequential transfer optimization problems.

Algorithm 1: General Framework of IST   
Data: Task size K, Initial budgets $N _ { i n i t }$ , Total budgets   
for each task $N _ { t o t }$ , Objective functions $F _ { k }$ for   
each task k, Task prioritization function $\phi ( \bullet )$   
Result: Optimal solutions for each optimization task.   
1 foreach task k do   
2 Evaluate the objective function $F _ { k }$ of task k for   
$N _ { i n i t }$ iterations   
3 $E v a l _ { k } \gets N _ { i n i t }$   
4 end   
5 while termination condition is not met do   
6 $k \gets \mathrm { a r g m a x } _ { t \in \{ 1 , . . . , K \} \wedge E v a l _ { t } < N _ { t o t } } \phi ( t )$   
7 $E v a l _ { k } \gets E v a l _ { k } + 1$   
8 Solve formulation (2) with task k as the target task   
9 end

## A. General Framework

The proposed IST framework pursues the same objective as the MTO paradigm described in formulation (1), that is, generating the optimal solution set $( \mathbf { x } _ { 1 } ^ { * } , \mathbf { x } _ { 2 } ^ { * } , \ldots , \mathbf { x } _ { K } ^ { * } )$ . However, unlike standard MTO that disperses evaluations across tasks evenly, IST can model the problem as a sequence of STrO tasks as in (2). Per iteration, a single target task T is selected to receive an evaluation based on a likelihood-informed task prioritization mechanism, formulated as:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { { \bf x } _ { T } \in \Omega _ { T } } \quad f _ { T } ( { \bf x } _ { T } ) \quad \mathrm { g i v e n } \quad \{ \mathcal { D } _ { S _ { k } } \} _ { k = 1 } ^ { K } } \\ { \displaystyle \mathrm { s . t . } \quad T = \quad \quad \mathrm { a r g m a x } \quad \quad \phi ( t ) } \\ { \displaystyle t \in \{ 1 , . . . , K \} \wedge E v a l _ { t } < N _ { t o t } \quad } \end{array}\tag{3}
$$

where ϕ(t) quantifies the utility of assigning task t as the target given the current optimization history $\{ \mathcal { D } _ { S _ { k } } \}$ of all tasks.

The primary distinctions of IST lie in its single-directional knowledge transfer and its selective evaluation mode. As illustrated in Fig. 1b, only the prioritized target is evaluated per iteration, reflecting an inherent budget designation process that prioritizes tasks most ready for knowledge integration. This formulation allows IST to leverage current STrO methods for a more meticulous controlled transfer process, to potentially circumvent the few-shot challenge. The complete workflow of IST is detailed in Algorithm 1:

• Initialization: Each task is evaluated for $N _ { i n i t }$ iterations, generating the source datasets $\mathcal { D } _ { S }$

• Task Prioritization: A target task is identified per iteration by the task prioritization function ϕ(·) to maximize transfer utility, while ensuring no task exceeds the total evaluation limit $N _ { t o t }$

• Sequential Transfer Optimization: The identified target task is optimized by transferring knowledge from the remaining source tasks using established STrO solvers.

## B. Base Sequential Transfer Optimizer

In this study, the IST framework is instantiated using the recently proposed F-invTrEMO [21]. This base optimizer leverages inter-task relationships through a hybrid forwardinverse mapping approach, particularly effective for few-shot multiobjective multitask optimization.

1) Scalarizing Multiobjective Optimization: To handle multiple objectives within each task, the vector-valued function $F _ { T } ( \cdot )$ is scalarized using augmented Tchebycheff scalarization. For a given weight vector w from a (m − 1) dimensional simplex W, the scalarized objective is:

$$
f _ { K } ^ { t c h } ( \mathbf { x } _ { K } | \mathbf { w } ) = \operatorname* { m a x } _ { 1 \leq i \leq m } \{ w _ { i } ( f _ { K , i } ( \mathbf { x } ) - ( z _ { K , i } ^ { * } - \epsilon ) ) \} +\tag{4}
$$

where $z _ { K , i } ^ { * }$ is the ideal point, $z _ { K , i } ^ { * } - \epsilon$ provides a utopia point, and $\rho$ is a small constant to maintain Pareto optimality.

2) Multitask Gaussian Process: To address few-shot multitask optimization problems, the Multitask Gaussian Process (MTGP) [27] is generally adopted to alleviate the evaluation cost and enable knowledge transfer. Given input spaces across tasks $\Omega _ { k } , k \in \{ 1 , \ldots , K \}$ , scalarized objective functions, $f _ { 1 } ^ { t c h } , \ldots , f _ { K } ^ { t c h }$ , are modelled by MTGP. During the modeling process, we have triplets $\{ ( i _ { s } , \mathbf { x } _ { s } ) , y _ { s } \} _ { s = 1 } ^ { N }$ with N evaluated solutions, with $i _ { s }$ , the task index of the s-th evaluated solution, $\mathbf { x } _ { s } \in \Omega _ { i _ { s } }$ , solutions, and $y _ { s } = f _ { i _ { s } } ^ { t c h } ( \mathbf { x } _ { s } ) + \epsilon _ { i _ { s } } ,$ noisy evaluations where the task-dependent noise $\epsilon _ { i _ { s } }$ is additive Gaussian noise with zero mean $( \mathrm { i . e . , ~ } \epsilon _ { i _ { s } } \sim \mathcal { N } ( 0 , \sigma _ { i _ { s } } ^ { 2 } ) )$ MTGP [27] is distinct for the formulation of the multitask kernel as below:

$$
\kappa ( ( i , { \bf x } ) , ( i ^ { \prime } , { \bf x } ^ { \prime } ) ) = \kappa \tau ( i , i ^ { \prime } ) \cdot \kappa _ { \Omega } ( { \bf x } , { \bf x } ^ { \prime } )\tag{5}
$$

where the pair (i, x) represents solution x for task i, κ measures the similarities among tasks, and $\kappa \Omega$ measures the similarities among solutions. Given the multitask kernel in formulation (5), one can estimate the posterior distribution, $\mathcal { N } ( \mu ( i , \mathbf { x } ) , \sigma ^ { 2 } ( i , \mathbf { x } ) )$ , of a query pair (i, x) as follows:

$$
\mu _ { t } ( i , \mathbf { x } ) = \kappa _ { t } ( i , \mathbf { x } ) ^ { \intercal } ( \mathbf { K } _ { t } + \Lambda ) ^ { - 1 } \mathbf { y } _ { 1 : t }\tag{6}
$$

$$
\begin{array} { c } { \sigma _ { t } ^ { 2 } ( i , { \bf x } ) = \kappa ( ( i , { \bf x } ) , ( i , { \bf x } ) ) - } \\ { \kappa _ { t } ( i , { \bf x } ) ^ { \top } ( { \bf K } _ { t } + { \bf \Delta } { \bf \Lambda } { \bf \Lambda } ) ^ { - 1 } \kappa _ { t } ( i , { \bf x } ) } \end{array}\tag{7}
$$

where $\boldsymbol { \kappa } _ { t } ( i , \boldsymbol { x } ) = \{ \kappa ( ( i , \mathbf { x } ) , ( i _ { s } , \mathbf { x } _ { s } ) ) \} _ { s = 1 } ^ { t } , \mathbf { y } _ { 1 : t } = \{ y _ { s } \} _ { s = 1 } ^ { t } ,$ $\mathbf { K } _ { t } ~ = ~ \{ \kappa ( ( i _ { s } , \mathbf { x } _ { s } ) , ( i _ { s ^ { \prime } } , \mathbf { x } _ { s ^ { \prime } } ) ) \} _ { s , s ^ { \prime } = 1 } ^ { t }$ , and Λ is the additive noise variance matrix of MTGP.

3) MTGP-based Forward-Inverse Transfer: Given the scalarized objective functions in formulation (4) and the MTGP model in formulation (6) and (7) for solving fewshot optimization, the knowledge transfer can be conducted in a hybrid forward-inverse approach. The forward mapping, $\Psi _ { f o r } .$ only approximates the scalarized objective functions, $f _ { T } ^ { t c h }$ , that is, $\Psi _ { f o r } : \Omega _ { T } \mapsto \mathbb { R }$ . The inverse mapping, $\Psi _ { i n v } .$ models the transformation from the $( m \textrm { -- } 1 )$ dimensional simplex, W, which includes the weight vector w to the solution space, $\Omega _ { T }$ , that is, $\Psi _ { i n v } : \mathcal { W } \mapsto \Omega _ { \mathcal { T } }$ . In this paper, we assume each task contains solutions in d dimensions, that is, $\Omega _ { T } \subset \mathbb { R } ^ { d }$ . To relieve the computational burden of multi-output GP modeling, we separated the inverse modeling into a series of single-output mapping, $\Psi _ { i n v , i } : \mathcal { W } \mapsto $ $\Omega _ { \mathcal { T } , i } , \Omega _ { \mathcal { T } , i } \subset \mathbb { R } , i \in \{ 1 , \ldots , d \}$ . After this separate singleoutput GP modeling, the predictions can then be aggregated to form the original d-dimensional predictions. Generally, prior to the inverse modeling process, the data $\{ \mathbf { w } _ { s } , ( i _ { s } , \mathbf { x } _ { s } ) \} _ { s = 1 } ^ { N }$ should be prepared in advance. In this paper, we assume that these data pairs have been constructed already, and one can refer to [20] for the details and rationale behind them. Considering this hybrid forward-inverse transfer mechanism,

Algorithm 2: Workflow of F-invTrEMO in One Pass   
Data: Target task $\mathcal { T } _ { K }$ , Source tasks $\overline { { \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { K - 1 } } } ,$   
Dimension size $d ,$ Predefined weight vector $\tilde { \mathbf { w } } .$   
Evaluation records $\begin{array} { r } { \{ ( i _ { s } , \mathbf { x } _ { s } ) , y _ { s } , \mathbf { w } _ { s } \} _ { s = 1 } ^ { N } , } \end{array}$   
Sample size $N _ { S } .$   
Result: Optimal solutions for the target task.   
1 /\*Forward MTGP Modeling\*/   
2 Build a forward MTGP model,   
$\mathcal { N } ( \mu _ { f m t } ( i , \mathbf { x } ) , \sigma _ { f m t } ^ { 2 } ( i , \mathbf { x } ) )$ , based on optimization   
history, $\{ ( i _ { s } , { \bf x } _ { s } ) , y _ { s } \} _ { s = 1 } ^ { N } .$   
3 /\*Inverse MTGP Modeling\*/   
4 foreach dimension $j$ do   
5 Build d inverse MTGP models,   
$\mathcal { N } ( \mu _ { i m t , j } ( i , \mathbf { w } ) , \sigma _ { i m t , j } ^ { 2 } ( i , \mathbf { w } ) ) .$ , based on the   
evaluation records, $\{ \mathbf { w } _ { s } , ( i _ { s } , \mathbf { x } _ { s , j } ) \} _ { s = 1 } ^ { N }$   
6 end   
7 /\*Conducting Sequential Transfer Optimization\*/   
8 Sample solution set $\mathcal { U } _ { T _ { K } }$ from $N _ { S }$ solutions from   
distribution, $\mathcal { N } ( \mu _ { i m t } ( \mathcal { \bar { T } } _ { K } , \tilde { \mathbf { w } } ) , \sigma _ { i m t } ^ { 2 } ( \mathcal { T } _ { K } , \tilde { \mathbf { w } } ) )$   
9 Select solution   
$\tilde { \mathbf { x } } = \mathop { \operatorname { a r g m a x } } _ { - \tau } - \mu _ { f m t } ( { \mathcal { T } } _ { K } , \mathbf { x } _ { { \mathcal { T } } _ { K } } ) + \boldsymbol { \beta } \cdot \sigma _ { f m t } ( { \mathcal { T } } _ { K } , \mathbf { x } _ { { \mathcal { T } } _ { K } } ) .$   
$\mathbf { x } \tau _ { K } ^ { - } \in \mathcal { U } \tau _ { K }$   
10 Evaluate solution $\tilde { \bf x }$ in the target task $\mathcal { T } _ { K }$

Algorithm 2 demonstrates the workflow of F-invTrEMO with a single pass. Given the datasets for both forward mapping $\Psi _ { f o r }$ and inverse mapping $\Psi _ { i n v }$ per iteration, both forward and inverse MTGP models are constructed and utilized for sequential transfer optimization directly, as depicted in line 8 to line 10 in Algorithm 2. Particularly, since this study focuses on minimization, the solution selection mechanism aims to maximize the lower confidence bound (LCB) [28] as shown in line 9.

4) Factorized MTGP-based Forward-Inverse Transfer: Our paper implements the MTGP-based forward-inverse transfer based on an efficient formulation in [29] so that the problem that, joint training MTGP can bias the source task when the volume of the source task is significantly more than that of the target task [29], can be alleviated. Specifically, instead of training the MTGP model for all the K tasks jointly, we replace the formulation (6) and (7) with the formulation as follows:

$$
\begin{array} { c } { { \displaystyle { \mu } _ { f m t } ( { \mathcal T } _ { K } , { \mathbf x } ) = \sigma _ { f m t } ^ { 2 } ( { \mathcal T } _ { K } , { \mathbf x } ) \{ ( \displaystyle \sum _ { j = 1 } ^ { K - 1 } \sigma _ { { \mathcal T } _ { j } } ^ { - 2 } ( { \mathcal T } _ { K } , { \mathbf x } ) \cdot \begin{array} { c } { { } } \\ { { } } \end{array} } } \\ { { \mu _ { { \mathcal T } _ { j } } ( { \mathcal T } _ { K } , { \mathbf x } ) + ( 2 - K ) \cdot \sigma _ { { \mathcal T } _ { K } } ^ { - 2 } ( { \mathbf x } ) \cdot \mu _ { { \mathcal T } _ { K } } ( { \mathbf x } ) \} } } \end{array}\tag{8}
$$

$$
\sigma _ { f m t } ^ { 2 } ( \mathcal T _ { K } , \mathbf x ) = 1 / \{ \sum _ { j = 1 } ^ { K - 1 } \sigma _ { \mathcal T _ { j } } ^ { - 2 } ( \mathcal T _ { K } , \mathbf x ) + ( 2 - K ) \cdot \sigma _ { \mathcal T _ { K } } ^ { - 2 } ( \mathbf x ) \}\tag{9}
$$

where $\mu _ { \mathcal { T } _ { j } } ( \mathcal { T } _ { K } , \mathbf { x } )$ and $\sigma _ { T _ { i } } ^ { 2 } ( \mathcal { T } _ { K } , \mathbf { x } )$ denote the posterior mean and variance of the solution x in task $\mathcal { T } _ { K }$ with MTGP using only the source task $\tau _ { j }$ and the target task $\mathcal { T } _ { K } ( \mathbf { x } ) , \mu _ { \mathcal { T } _ { K } } ^ { - 2 } ( \mathbf { x } )$ and $\sigma _ { T _ { K } } ^ { - \bar { 2 } } ( { \bf x } )$ are posterior mean and variance of the target task $\mathcal { T } _ { \mathcal { K } }$ using only the single-task GP. This formulation is practically useful inspired by recent works in STrO domains [19], [26].

## C. Likelihood-Informed Task Prioritization

The efficacy of the IST framework hinges on identifying the optimal target task $T$ to receive the next evaluation budget. We instantiate the task prioritization function $\phi ( \cdot )$ by leveraging the inter-task relationships captured during the MTGP modeling process.

1) Task Prioritization Function: The search synergy between any task pair $( i , i ^ { \prime } )$ is explicitly quantified through the task kernel parameter $\kappa \tau ( i , i ^ { \prime } )$ , where tasks i and $i ^ { \prime }$ serve as source and target, respectively. This kernel can be estimated online per iteration as outlined in Algorithm 2. As indicated in [26], for sequential transfer optimization with a single target task $\mathcal { T } _ { K }$ , a larger task search synergy parameter $\kappa \tau ( \mathcal { T } _ { S } , \mathcal { T } _ { K } )$ suggests that Gaussian process optimization with the LCB component can achieve a lower optimization regret bound compared to source tasks with lower search synergy parameters.

Because the search synergy parameter is learned through maximum likelihood estimation, it can be treated as the statistical likelihood that a given task pair should serve as source and target, respectively [27], [29]. We therefore formulate the task prioritization function as:

$$
\phi ( t ) = \operatorname* { m i n } _ { i \neq t } \kappa _ { \mathscr { T } } ( i , t ) .\tag{10}
$$

While high $\kappa \tau$ is generally preferred for beneficial transfer, the min operator in (10) ensures the selected target t is compatible with the entire source task pool. By maximizing this lower bound, the framework prioritizes targets that maintain a high degree of synergy across all potential source tasks, thereby ensuring reliable knowledge integration.

2) Stochastic Task Prioritization: Given the definition of the task prioritization function $\phi ( t )$ , according to Algorithm 1, we assign the target role to task k with the largest ϕ(t). However, the argmax selector may cause certain tasks to remain static for too many iterations, violating the assumption in the sequential transfer optimization that all the source tasks can achieve high-quality solution distributions. To mitigate this, we adopt the softmax function to relax this argmax selector as follows:

$$
\mathcal { T } _ { K } \sim M u l t i n o m i a l ( 1 ; \{ P _ { 1 } , \dots , P _ { K } \} )\tag{11}
$$

$$
P _ { t } = \frac { \exp ( S \cdot ( \operatorname* { m a x } \{ \phi ( t ) - \theta , 0 \} ) ) } { \sum _ { j = 1 \land E v a l _ { j } < N _ { t o t } } ^ { K } \exp ( S \cdot ( \operatorname* { m a x } \{ \phi ( j ) - \theta , 0 \} ) ) }\tag{12}
$$

where the target task $\mathcal { T } _ { K }$ is sampled from a multinomial distribution with probabilities $P _ { t }$ computed using this softmax function as defined in formula (12). In this formulation, S controls the selective pressure of the softmax function: a higher S makes it more similar to argmax function, while a lower S makes it resemble the uniform selection. Moreover, the parameter θ in formula (12) serves as a threshold value to maintain search vitality. Specifically, while tasks with $\phi ( t ) > \theta$ receive an exponential boost in selection priority, all tasks falling below this threshold share an identical baseline weight $( e ^ { 0 } = 1 )$ . This ensures that even tasks with extremely low source-target likelihood remain selectable with a non-zero probability, preventing any task from being perpetually frozen during the iterative process.

## IV. EXPERIMENTAL STUDIES

To verify the effectiveness of the proposed IST framework and the likelihood-informed task prioritization mechanism, we conduct comparative studies on MOMTO benchmark problems and multiobjective multitask hyperparameter optimization problems. Moreover, to further justify the generality of the framework, we instantiate another IST-based algorithm by extending a renowned sequential transfer optimizer, AMTEA [3], to AMTEA-IST in Section IV.D. We term the F-invTrEMO implementation under the IST framework as F-invTrEMO-IST. We compare F-invTrEMO-IST with the classical ParEGO, because many STrO methods [19]–[21] build upon this singletask few-shot multiobjective optimizer, and the few-shot multiobjective multitask optimizer, F-invTrEMO. One can refer to more detailed parameter settings in the supplementary materials <sup>1</sup>. Experiments upon benchmark problems are independently repeated 20 trials and upon hyperparameter optimization problems are repeated 10 independent trials.

## A. Test Problems

We conduct the comparative studies on multiobjective multitask benchmarks [23]. It includes nine MOMTO problems and each problem contains two tasks with certain relationships regarding search space similarity and optima intersection. The optima intersections includes complete intersection (CI), partial intersection (PI), and no intersection (NI), while the search space similarity includes high similarity (HS), medium similarity (MS), and low similarity (LS). Nine sets of problems can be constructed with these attributes, including CIHS, CIMS, CILS, PIHS, PIMS, PILS, NIHS, NIMS, and NILS. However, NIMS and NILS are not included in this paper, since our works are based on decomposition-based multiobjective optimizer, which cannot solve multitask problems with distinct objective sizes like NIMS and NILS. As for this limitation, we leave it as a future direction. One can refer to the supplementaries for more details on test problems.

We adopt the inverted generational distance (IGD+) [30] as the metric to quantify the performance of the algorithms, as recommended in [23]. One can refer to the details of IGD+ in the supplementary materials. The statistical significance test is conducted by the Wilcoxon signed-rank test.

## B. Results

In our comparative study upon the multitask optimization benchmark, we compare the proposed method F-invTrEMO-IST with the single-task method ParEGO and the plain method F-invTrEMO without IST settings. It can be indicated in TABLE I that both F-invTrEMO and F-invTrEMO-IST can significantly outperform the baseline method ParEGO due to the ability to exploit intertask relationship information through the forward-inverse transfer mechanism. The exceptions occur in both problem sets with the relationship LS (i.e., CILS and PILS), where the search space between tasks has the least similarity. Since both F-invTrEMO and F-invTrEMO-IST maintain the solution distribution by mapping the predefined weight vector w to the solution space, and the dataset used to train the mapping process is constrained according to the previous optimization process, the solution distribution can be spuriously biased towards the local Pareto front, hindering the subsequent optimization process. Moreover, the least similar search space between tasks makes it difficult to utilize the knowledge transfer to help escape the local optima in the target task from source tasks. In terms of the comparison between F-invTrEMO and F-invTrEMO-IST, it can be found in TABLE I that F-invTrEMO-IST that is implemented with IST settings can outperform the counterpart in 12 of 14 tasks, suggesting the superiority of the proposed IST framework. In most problem sets, IST can identify the more proper sourcetarget pair to conduct the knowledge transfer process using the proposed task prioritization mechanism, mitigating the potential negative transfer. Since the knowledge transfer process for source-task pairs with lower transfer utilities should be suppressed in certain stages, IST can also be viewed as an approach to implement adaptive resource allocation on the fly.

## C. A Case Study on Real-world Application: Multiobjective Multitask Hyperparameter Optimization

Hyperparameter optimization (HPO) is a standard topic in the field of machine learning. The configuration of hyperparameters for machine learning models can impact the model performance, computational resources, and interpretability to decision-makers. In this paper, we apply F-invTrEMO-IST to optimize the hyperparameters of these models for the aforementioned criterion across distinct tasks. We consider the following two scenarios:

• (HPO-1) The first scenario contains three HPO problems. The three problems tune the hyperparameters of the same model, Random Forest, but on three distinct classification tasks: credit approval, medical diagnosis, and speech recognition problems.

• (HPO-2) The second scenario contains two HPO problems. The two optimization problems tune the hyperparameters of distinct but related models on the same classification task, speech recognition problem [31]. One can refer to the definition of each objective function in supplementary materials.

As illustrated in TABLE II, with the IST framework, the proposed F-invTrEMO-IST can outperform F-invTrEMO across all the optimization tasks, generating better machine learning model sets trading off across model precision, model size, and interpretability. Importantly, in real-world cases, it is rare that the solution optima and search spaces share as high commonalities as benchmark problems such as CIHS or CIMS in TABLE I. This heterogeneity in both optimal distribution and search space similarity makes it important to consider the utilization of computing resources and knowledge transfer direction, where the proposed iterative sequential transfer framework matters.

![](images/32de77282aef18c83fa8941d7ce353484a1c71f3e521b9a3bf748b3e456a7507.jpg)  
(a)

![](images/e57cb01173fb372d655263625cd7f95cfa242982f37b3b9b99f6e335b2aa595d.jpg)

![](images/a101422d64193da8bf68dc3cde72104aa18735e3745f14d9b5672f4bad40652e.jpg)  
(b)

![](images/cc704f7460216c7520866c0ad24ad35bfe93557f7d0cc92805db24bb5ade96b0.jpg)  
Fig. 2. The IGD+ convergence trends of ParEGO, F-invTrEMO, and F-invTrEMO-IST upon multitask optimization benchmark (a) PIHS (b) NIHS.

TABLE I  
IGD+ COMPARISON RESULTS ON MULTITASK OPTIMIZATION BENCHMARK FOR PAREGO, F-INVTREMO, AND F-INVTREMO-IST
<table><tr><td rowspan=1 colspan=2>Problems</td><td rowspan=1 colspan=1>lems</td><td rowspan=1 colspan=1>Tasks</td><td rowspan=1 colspan=1>ParEGO</td><td rowspan=1 colspan=1>F-invTrEMO</td><td rowspan=1 colspan=1>F-invTrEMO-IST</td></tr><tr><td rowspan=2 colspan=3>CIHS</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>1.516E+02 (1.545E+01) +</td><td rowspan=1 colspan=1>8.061E+01 (1.534E+01) +</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>4.877E+00 (3.150E-01) +</td><td rowspan=1 colspan=1>3.557E+00 (2.329E-01) ≈</td><td rowspan=1 colspan=1>3.389E+00 (1.925E-01)</td></tr><tr><td rowspan=2 colspan=3>CIMS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>2.523E+02 (2.057E+01) +</td><td rowspan=1 colspan=1>2.086E+02 (2.658E+01) +</td><td rowspan=1 colspan=1>1.954E+02 (4.069E+01)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>4.174E+00 (2.983E-01) +</td><td rowspan=1 colspan=1>3.819E+00 (2.859E-01) +</td><td rowspan=1 colspan=1>3.606E+00 (1.799E-01)</td></tr><tr><td rowspan=2 colspan=3>CILS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>4.564E+01 (2.021E+00) +</td><td rowspan=1 colspan=1>4.435E+01 (1.592E+00) +</td><td rowspan=1 colspan=1>4.113E+01 (1.601E+00)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>3.050E-01 (1.040E-02) -</td><td rowspan=1 colspan=1>7.938E-01 (5.305E-02) +</td><td rowspan=1 colspan=1>6.856E-01 (4.870E-02)</td></tr><tr><td rowspan=2 colspan=3>PIHS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>8.728E+01 (9.112E+00) +</td><td rowspan=1 colspan=1>3.234E+01 (8.086E+00) +</td><td rowspan=1 colspan=1>1.399E+01 (1.978E+00)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>1.841E+02 (1.678E+01) +</td><td rowspan=1 colspan=1>9.913E+01 (1.033E+01) +</td><td rowspan=1 colspan=1>8.109E+01 (3.761E+00)</td></tr><tr><td rowspan=2 colspan=3>PIMS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>2.678E+00 (1.539E-01) +</td><td rowspan=1 colspan=1>4.324E-01 (3.790E-02) -</td><td rowspan=1 colspan=1>4.784E-01 (5.205E-02)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>3.736E+02 (1.269E+01) +</td><td rowspan=1 colspan=1>3.503E+02 (5.258E+00) ≈</td><td rowspan=1 colspan=1>3.407E+02 (4.165E+00)</td></tr><tr><td rowspan=2 colspan=3>PILS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>1.008E+00 (1.390E-02) ≈</td><td rowspan=1 colspan=1>1.029E+00 (1.165E-02) ≈</td><td rowspan=1 colspan=1>1.041E+00 (9.500E-03)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>1.423E+01 (6.948E-01) +</td><td rowspan=1 colspan=1>1.258E+01 (8.887E-01) +</td><td rowspan=1 colspan=1>1.053E+01 (6.600E-01)</td></tr><tr><td rowspan=2 colspan=3>NIHS</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>3.174E+03 (6.287E+02) +</td><td rowspan=1 colspan=1>1.651E+03 (5.881E+02) +</td><td rowspan=1 colspan=1>5.934E+02 (2.174E+02)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>5.820E+01 (4.852E+00) +</td><td rowspan=1 colspan=1>2.834E+01 (3.824E+00) +</td><td rowspan=1 colspan=1>1.903E+01 (4.056E+00)</td></tr></table>

TABLE II

IGD+ COMPARISON RESULTS ON MULTITASK HYPERPARAMETER TUNING PROBLEMS FOR F-INVTREMO AND F-INVTREMO-IST
<table><tr><td rowspan=1 colspan=1>Problems</td><td rowspan=1 colspan=1>Tasks</td><td rowspan=1 colspan=1>F-invTrEMO</td><td rowspan=1 colspan=1>F-invTrEMO-IST</td></tr><tr><td rowspan=2 colspan=1>HPO-1</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>0.0502 (0.0028) +</td><td rowspan=1 colspan=1>0.0455 (0.0021)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>0.0439 (0.0020) ≈</td><td rowspan=1 colspan=1>0.0423 (0.0014)</td></tr><tr><td rowspan=3 colspan=1>HPO-2</td><td rowspan=1 colspan=1>Task-1</td><td rowspan=1 colspan=1>0.0325 (0.0017) +</td><td rowspan=1 colspan=1>0.0303 (0.0031)</td></tr><tr><td rowspan=1 colspan=1>Task-2</td><td rowspan=1 colspan=1>0.0282 (0.0019) ≈</td><td rowspan=1 colspan=1>0.0279 (0.0014)</td></tr><tr><td rowspan=1 colspan=1>Task-3</td><td rowspan=1 colspan=1>0.0421 (0.0030) +</td><td rowspan=1 colspan=1>0.0392 (0.0023)</td></tr></table>

## D. A Case Study on Generality of the Framework: IST based on AMTEA

To verify the generality of the proposed IST on other sequential transfer optimizers, IST is integrated with another evolutionary sequential transfer optimizer, AMTEA [3]. We term AMTEA with IST settings as AMTEA-IST. AMTEA is originally designed for the canonical algorithms requiring thousands of function evaluations, while we integrate AMTEA into the few-shot multiobjective multitask optimization. To this end, we transform the multiobjective problem into singleobjective problems via (4) at each iteration. The singleobjective optimization is then evaluated using a Gaussian Process surrogate model, with solutions sampled from the solution distribution managed by AMTEA, similar to the approach used in ParEGO and Algorithm 2. One can refer to the supplementary materials for more details.

The results can be found in TABLE S-I in supplementary materials that both AMTEA and AMTEA-IST can outperform the ParEGO counterpart in 13 of 14 tasks, showcasing the effectiveness of the knowledge transfer in the context of fewshot MOMTO. Comparing AMTEA to AMTEA-IST, with the meticulous transfer direction configuration, AMTEA-IST can outperform AMTEA in 11 of 14 multiobjective tasks. Generally. Moreover, the IST framework can facilitate a better search process for the base optimizer in multi-task settings with lower search space similarity, as shown in TABLE S-I. The improvements upon AMTEA for CIHS and CIMS are not significant since high-quality solutions from both tasks are similar, so that the task prioritization components cannot precisely capture the inter-task relationship, thereby misleading the IST framework. In contrast, for those problem sets with lower similarity in TABLE S-I, AMTEA-IST generally can significantly outperform AMTEA in light of the ability to filter out the least likely task pair.

## V. CONCLUSION

In this work, we solve the few-shot MOMTO by introducing the Iterative Sequential Transfer (IST). Unlike conventional methods, our IST framework can model the multitask problem as a sequence of sequential transfer optimization problems. To facilitate this, a likelihood-informed task prioritization mechanism is devised to actively control the knowledge transfer direction. This novel framework alleviates the inherent limitations of simultaneous evaluation processes in traditional MTO, thereby mitigating negative transfer and enhancing efficiency in few-shot regimes. The IST framework not only advances traditional MTO but also accommodates existing sequential transfer optimization algorithms. By bridging the gap between multitask and sequential transfer studies, this approach paves the way for more efficient algorithmic designs. The comprehensive evaluation across benchmark and realworld hyperparameter testbeds demonstrates the effectiveness of the proposed framework under stringent evaluation budgets.

## ACKNOWLEDGMENTS

This research is partly supported by the National Research Foundation, Singapore and DSO National Laboratories under the AI Singapore Programme (AISG Award No.: AISG2- GC-2023-010, Design Beyond What You Know: Material-Informed Differential Generative AI (MIDGAI) for Light-Weight High-Entropy Alloys and Multi-functional Composites (Stage 1b), the A\*STAR Catalyst Project for Artificial Intelligence in Drug Discovery (AIDD) Programme (Grant No. H25A1N0004), the Centre for Frontier AI Research (CFAR) under Agency for Science, Technology and Research (A\*STAR), and the College of Computing and Data Science, Nanyang Technological University.

## REFERENCES

[1] A. Gupta, Y.-S. Ong, and L. Feng, “Insights on transfer optimization: Because experience is the best teacher,” IEEE Trans. on Emerg. Topics in Comput. Intell., vol. 2, no. 1, pp. 51–64, 2018.

[2] L. Feng, A. Gupta, K. C. Tan, and Y.-S. Ong, Evolutionary Multi-task Optimization: Foundations and Methodologies. Springer, 2023.

[3] B. Da, A. Gupta, and Y.-S. Ong, “Curbing negative influences online for seamless transfer evolutionary optimization,” IEEE Trans. on Cybern., vol. 49, no. 12, pp. 4365–4378, 2018.

[4] M. Shakeri, E. Miahi, A. Gupta, and Y.-S. Ong, “Scalable transfer evolutionary optimization: Coping with big task instances,” IEEE Trans. on Cybern., vol. 53, no. 10, pp. 6160–6172, 2023.

[5] A. Gupta, Y.-S. Ong, and L. Feng, “Multifactorial evolution: toward evolutionary multitasking,” IEEE Trans. on Evol. Comput., vol. 20, no. 3, pp. 343–357, 2016.

[6] K. K. Bali, Y.-S. Ong, A. Gupta, and P. S. Tan, “Multifactorial evolutionary algorithm with online transfer parameter estimation: Mfea-ii,” IEEE Trans. on Evol. Comput., vol. 24, no. 1, pp. 69–83, 2019.

[7] T. Wei, J. Liu, A. Gupta, P. S. Tan, and Y.-S. Ong, “(θ , θ )-parametric multi-task optimization: Joint search in solution and infinite task spaces,” IEEE Trans. on Evol. Comput., vol. 30, no. 3, pp. 1270–1283, 2026.

[8] B. Da, A. Gupta, Y.-S. Ong, and L. Feng, “Evolutionary multitasking across single and multi-objective formulations for improved problem solving,” in 2016 IEEE Congr. on Evol. Comput. (CEC), 2016, pp. 1695– 1701.

[9] K. K. Bali, A. Gupta, Y.-S. Ong, and P. S. Tan, “Cognizant multitasking in multiobjective multifactorial evolution: Mo-mfea-ii,” IEEE Trans. on Cybern., vol. 51, no. 4, pp. 1784–1796, 2021.

[10] L. Feng, L. Zhou, J. Zhong, A. Gupta, Y.-S. Ong, K.-C. Tan, and A. K. Qin, “Evolutionary multitasking via explicit autoencoding,” IEEE Trans. on Cybern., vol. 49, no. 9, pp. 3457–3470, 2018.

[11] Z. Chen, Y. Zhou, X. He, and J. Zhang, “Learning task relationships in evolutionary multitasking for multiobjective continuous optimization,” IEEE Trans. on Cybern., pp. 1–12, 2020.

[12] A. Gupta, L. Zhou, Y.-S. Ong, Z. Chen, and Y. Hou, “Half a dozen real-world applications of evolutionary multitasking, and more,” IEEE Comput. Intell. Mag., vol. 17, no. 2, pp. 49–66, 2022.

[13] W. Dai, Z. Wang, and K. Xue, “System-in-package design using multi-task memetic learning and optimization,” Memetic Comput., Sep 2021. [Online]. Available: https://doi.org/10.1007/s12293-021-00346-5

[14] T. Wei, J. Liu, A. Gupta, C. C. Ooi, P. S. Tan, and Y.-S. Ong, “Parametric expensive multi-objective optimization via generative solution modeling,” 2026.

[15] L. Luo, X. Wang, J. Ma, and Y.-S. Ong, “Grpavoid: Multigroup collisionavoidance control and optimization for uav swarm,” IEEE Trans. on Cybern., vol. 53, no. 3, pp. 1776–1789, 2023.

[16] T. Wei, S. Wang, J. Zhong, D. Liu, and J. Zhang, “A review on evolutionary multitask optimization: Trends and challenges,” IEEE Trans. on Evol. Comput., vol. 26, no. 5, pp. 941–960, 2022.

[17] M. Gong, Z. Tang, H. Li, and J. Zhang, “Evolutionary multitasking with dynamic resource allocating strategy,” IEEE Trans. on Evol. Comput., vol. 23, no. 5, pp. 858–869, 2019.

[18] T. Wei and J. Zhong, “Towards generalized resource allocation on evolutionary multitasking for multi-objective optimization,” IEEE Comput. Intell. Mag., vol. 16, no. 4, pp. 20–37, 2021.

[19] J. Liu, A. Gupta, C. Ooi, and Y.-S. Ong, “Extremo: Transfer evolutionary multiobjective optimization with proof of faster convergence,” IEEE Trans. on Evol. Comput., pp. 1–1, 2024.

[20] J. Liu, A. Gupta, and Y.-S. Ong, “Bayesian inverse transfer in evolutionary multiobjective optimization,” ACM Trans. Evol. Learn. Optim., vol. 4, no. 4, Nov. 2024.

[21] T. Wei, J. Liu, A. Gupta, P. S. Tan, and Y.-S. Ong, “Bayesian forwardinverse transfer for multiobjective optimization,” in Parallel Problem Solving from Nature – PPSN XVIII. Cham: Springer Nature Switzerland, 2024, pp. 135–152.

[22] X. Xue, C. Yang, L. Feng, K. Zhang, L. Song, and K. C. Tan, “Solution transfer in evolutionary optimization: An empirical study on sequential transfer,” IEEE Trans. on Evol. Comput., vol. 28, no. 6, pp. 1776–1793, 2024.

[23] Y. Yuan, Y.-S. Ong, L. Feng, A. K. Qin, A. Gupta, B. Da, Q. Zhang, K. C. Tan, Y. Jin, and H. Ishibuchi, “Evolutionary multitasking for multiobjective continuous optimization: Benchmark problems, performance metrics and baseline results,” arXiv preprint arXiv:1706.02766, 2017.

[24] X. Wu, S. Liu, Q. Lin, K. Chen Tan, and V. C. M. Leung, “Evolutionary multitasking with adaptive knowledge transfer for expensive multiobjective optimization,” IEEE Trans. on Evol. Comput., vol. 29, no. 6, pp. 2537–2551, 2025.

[25] J. Liu, W. Liu, J. T. W. En, C. Chen, P. S. Tan, and Y.-S. Ong, “Optimal transport-based distributional pairing in transfer multiobjective optimization,” IEEE Trans. on Evol. Comput., pp. 1–1, 2025.

[26] H. Wu, T. Wei, J. Liu, M. Xu, Y.-S. Ong, and Y. Jin, “Convergence of expensive multi-objective optimizers: From parego to extremo,” in 2025 IEEE Congress on Evol. Comput. (CEC), 2025, pp. 1–8.

[27] E. V. Bonilla, K. Chai, and C. Williams, “Multi-task gaussian process prediction,” in Advances in Neural Information Processing Systems, J. Platt, D. Koller, Y. Singer, and S. Roweis, Eds., vol. 20. Curran Associates, Inc., 2007.

[28] M. Seeger, “Gaussian processes for machine learning,” International journal of neural systems, vol. 14, no. 02, pp. 69–106, 2004.

[29] B. Da, Y.-S. Ong, A. Gupta, L. Feng, and H. Liu, “Fast transfer gaussian process regression with large-scale sources,” Knowledge-Based Systems, vol. 165, pp. 208–218, 2019.

[30] D. A. Van Veldhuizen and G. B. Lamont, “Multiobjective evolutionary algorithm research: A history and analysis,” Citeseer, Tech. Rep., 1998.

[31] F. Pfisterer, L. Schneider, J. Moosbauer, M. Binder, and B. Bischl, “Yahpo gym - an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization,” in Proceedings of the First International Conference on Automated Machine Learning, ser. Proceedings of Machine Learning Research, vol. 188. PMLR, 25–27 Jul 2022, pp. 3/1–39.