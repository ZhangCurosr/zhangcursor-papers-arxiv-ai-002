# Scalable Rao-Blackwellized Online Planning for High-Dimensional POMDPs

Jiho Lee<sup>1</sup>, Nisar Ahmed<sup>1</sup>, Kyle Hollins Wray<sup>2</sup>, Zachary Sunberg<sup>1</sup>

Abstract—Online planning under uncertainty remains a fundamental challenge for robotic systems operating in partially observable environments with high-dimensional state spaces. While sampling-based POMDP solvers enable approximate decisionmaking in large or continuous domains, their performance degrades as belief dimensionality increases due to the high variance inherent in Monte Carlo-based estimation. In this work, we extend the Rao-Blackwellized online POMDP (RB-POMDP) framework to improve its generalizability in high-dimensional settings through hybrid continuous–discrete belief representations. By analytically propagating uncertainty associated with marginalized state components during tree-based planning, the proposed approach reduces sampling-induced variance in value estimation. We demonstrate the effectiveness of this framework in a robotic search-and-rescue task by integrating it with Fast-SLAM 2.0. Experimental results show that the proposed planner achieves higher cumulative rewards using significantly fewer particles and planning simulations than purely sampling-based methods under equivalent computational budgets. These results suggest that structured high-dimensional robotic problems admitting tractable sufficient statistics can be effectively leveraged within the RB-POMDP framework for computationally feasible online decision-making.

Index Terms—POMDPs, FastSLAM, Rao-Blackwellization

## I. INTRODUCTION

R <sup>OBOTIC</sup> <sup>decision-making</sup> <sup>tasks</sup> <sup>such</sup> <sup>as</sup> <sup>search-and-</sup>rescue, autonomous navigation, and exploration often rescue, autonomous navigation, and exploration often require reasoning under significant uncertainty arising from limited sensing and incomplete knowledge of the environment [1] [2] [3] [4] [5]. Partially Observable Markov Decision Processes (POMDPs) provide a powerful mathematical framework for sequential decision-making under uncertainty by enabling agents to maintain beliefs over latent system states while planning actions that maximize expected long-term reward [6]. While POMDPs have been widely applied to various domains [7] [8] [9], their practical deployment still remains challenging due to the computational burden associated with belief representation and online planning in large-scale environments.

Sampling-based online POMDP solvers such as partially observable Monte Carlo planning (POMCP) [10] and its variant partially observable Monte Carlo planning with observation widening (POMCPOW) [11] maintain particle-based belief representations and rely on Monte Carlo simulation to approximate value estimates during online tree search. Recent theoretical results show that these planners can achieve convergence guarantees that depend only on the number of samples and are independent of state or observation space dimensionality, provided that the importance weight is bounded [12]. In practice, however, maintaining bounded importance weights become extremely challenging in high-dimensional settings, where probability mass concentrates in a small typical set of the state space and importance weights may vary significantly across samples [13]. As a result, particle filters often require a substantial increase in the number of particles to maintain reliable belief representation due to likelihood concentration and importance weight degeneracy [14] [15] [16]. In such settings, the effective sample size of the particles decreases, increasing the variance of estimators computed using the particles [17]. Since value estimates are computed via Monte Carlo runs under the particle belief, a low effective sample size increases the variance in value estimates in tree search. Consequently, consistent long-horizon planning requires a substantial increase in both the number of particles used for belief representation and the number of tree simulations needed for reliable planning. This leads to significant computational overhead and often renders these purely sampling-based approaches impractical for realistic robotic decision-making scenarios.

![](images/5a988b7102a9b10a0cee8c6dfd828d50e87e967214606b6299e340530b07bce4.jpg)  
Fig. 1. Motivating scenario for search-and-rescue under partial observability. A robot navigates an indoor environment consisting of multiple rooms while constructing a map and localizing itself. Victim presence behind each door is unknown and can only be inferred through limited sensing of both spatial and victim-related semantic cues. To accomplish the task, the agent maintains a belief over the joint robot and environment state and considers multiple possible future outcomes under different action-observation histories, represented by the planning tree on the right, to select actions that locate victims efficiently.

Many robotic systems, however, exhibit structured latent state spaces in which subsets of variables admit conditionally tractable posterior updates. For example, problems such as simultaneous localization and mapping (SLAM) involve latent geometric variables (e.g., landmark positions) whose uncertainty can be represented analytically given a sampled robot trajectory [18]. Such conditional structure implies that portions of the state admit sufficient statistics that can be propagated in closed form, rather than requiring full Monte Carlo approximation. Rao-Blackwellized Particle Filtering (RBPF) exploits this structure by analytically marginalizing subsets of the latent state conditioned on sampled variables, reducing the dimensionality that must be explored via particles [19] [20].

RBPF has been successfully applied to various estimation problems such as SLAM and target tracking [21] [22] [23]. Within the context of POMDPs, a recent work [24] introduced a Rao-Blackwellized POMDP (RB-POMDP) online planning framework that leverages analytical belief updates within the POMCPOW planner. While this approach demonstrated improved efficiency in continuous state and observation planning tasks, it remained limited to Kalman filter-based analytical components and relatively simple, low-dimensional experimental settings. Therefore, it is of significant interest to explore how this framework can provide a general mechanism to enable scalable decision-making in realistic robotic problems with much larger latent state spaces and non-Gaussian distributions.

This paper addresses this gap by extending the RB-POMDP framework to support arbitrary analytically tractable belief components and demonstrating the feasibility of online planning in high-dimensional problems. By analytically propagating uncertainty associated with marginalized state variables during tree-based planning, the new framework reduces sampling-induced variance in reward estimates, enabling consistent long-horizon reasoning with substantially fewer particles and tree iterations. In summary, this work makes the following three contributions:

1) We demonstrate that Rao-Blackwellized online planning is not merely beneficial but essential for computationally feasible decision-making in high-dimensional POMDPs with analytically tractable structure. Scalable planning becomes possible where purely sampling-based methods are infeasible under practical computational budgets.

2) We generalize the RB-POMDP framework beyond Gaussian assumptions by allowing the analytically maintained component to be instantiated with arbitrary closed-form belief update models. This enables planning over hybrid continuous-discrete latent state spaces in which marginalized variables may follow non-Gaussian distributions.

3) We validate the proposed framework in a realistic searchand-rescue scenario involving both geometric and semantic observations by integrating it with the FastSLAM 2.0 algorithm, as illustrated in Figure 1. This scenario induces a hybrid belief over robot pose, landmark geometry, and semantic victim presence, represented by particles, Gaussian, and Bernoulli distributions, respectively. This demonstrates that any structured high-dimensional estimation problems, if admitting tractable sufficient statistics, can be leveraged within the RB-POMDP framework.

The remainder of this article is organized as follows.

Section II reviews POMDPs, sampling-based POMDP planning methods, and RBPFs. Section III introduces the Rao-Balckwellized factorization of POMDPs, RBPF-based belief updates, and Rao-Blackwellized online planning. Section IV presents the search-and-rescue problem application, followed by Section V, which details the proposed Rao-Blackwellized belief representation and associated planning components. Experimental results are provided in Section VI. Finally, Section VII discusses the results and outlines directions for future work.

## II. BACKGROUND

## A. Partially Observable Markov Decision Process (POMDP)

In Markov Decision Processes (MDPs), agents operate with full knowledge of the current system state. At any given time, the agent fully understands where it is or what the situation is. In POMDPs, however, agents do not have perfect knowledge of the current state. They must make decisions under uncertainty about the current as well as future states; thus, POMDPs are more suitable for real-world robotics scenarios where the true state of the environment cannot be directly observed due to sensing limitations or environmental uncertainty [25].

A POMDP is formally defined by the tuple $( S , \mathcal { A } , \mathcal { T } , \mathcal { O } , \mathcal { R } , \mathcal { Z } , \gamma )$ where S is the state space, A is the action space, $\tau$ is the state transition function, O is the observation space, Z is the observation function, R is the reward function, and $\gamma \in ( 0 , 1 )$ is the discount factor. Because the agent cannot directly observe the system state, it instead maintains a probabilistic belief distribution over possible states. This belief is recursively updated using Bayes’ rule:

$$
b ^ { \prime } ( s ^ { \prime } ) \propto P ( o \mid s ^ { \prime } , a ) \sum _ { s \in { \cal S } } P ( s ^ { \prime } \mid s , a ) b ( s ) ,\tag{1}
$$

where $b ( s )$ and $b ^ { \prime } ( s ^ { \prime } )$ denote the prior and posterior beliefs before and after executing action a and receiving observation $^ { O , }$ respectively.

Solving POMDPs is therefore performed over belief states rather than fully observed states. The optimal value function, which guides the selection of the best action based on the current belief, is given by

$$
V ^ { * } ( b ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \left[ \mathcal { R } ( b , a ) + \gamma \sum _ { o \in \mathcal { O } } P ( o \mid b , a ) V ^ { * } ( b ^ { \prime } ) \right] ,\tag{2}
$$

where $\textstyle { \mathcal { R } } ( b , a )$ is the expected immediate reward for taking action a and $P ( o | b , a )$ is the probability of receiving observation o after taking action a. Consequently, the dimensionality and structure of the belief representation can directly influence the computational complexity of evaluating the value function during planning.

## B. Sampling-Based POMDP Online Planning

Evaluating the optimal value function in (2) is computationally intractable for most real-world problems due to the high-dimensional nature of belief spaces and observations. Consequently, a large body of work has focused on online planning methods that compute approximate action values during execution by simulating future outcomes from the current belief state.

The well-known sampling-based solver POMCP [10] employs particle-filter-based belief representations with Monte Carlo tree search (MCTS) and relies on a black-box generative model to sample future states, observations, and rewards during tree expansion. Extensions such as POMCPOW incorporate progressive widening techniques to regulate tree growth and improve planning performance in continuous observation spaces [11].

These sampling-based methods approximate value functions by averaging simulated rewards obtained from running N tree search simulations to a depth $D .$ . Specifically, the value at an observation-action history h at depth $\tau$ is estimated using the set of simulations consistent with that history, I(h), as

$$
V ( h ) \approx \frac { 1 } { \vert I ( h ) \vert } \sum _ { i \in I ( h ) } \sum _ { d = \tau } ^ { D } \gamma ^ { d - \tau } \mathcal { R } ( s _ { i , d } , a _ { i , d } ) .\tag{3}
$$

However, Monte Carlo sampling exhibits a convergence rate of $\mathcal { O } ( 1 / \sqrt { N } )$ , which can require a large number of simulations to achieve consistent estimates. Especially when operating over high-dimensional belief spaces, uncertainty in predicted observations and rewards during simulation can introduce substantial variance in estimated values within the planning tree. As a result, significant increase in the number of tree simulations is needed to achieve consistent, reliable longhorizon planning, inevitably leading to considerable computational overhead.

## C. Rao-Blackwellized Particle Filter (RBPF)

One approach to reducing the variance of purely samplingbased Monte Carlo estimators is to exploit analytically tractable structure in the latent state using the Rao-Blackwell theorem. This theorem states that conditioning an estimator on sufficient statistics cannot increase its variance [26], as expressed by

$$
\operatorname { v a r } [ \delta ( s ) ] \geq \operatorname { v a r } [ \delta ( s \mid \Theta ) ] ,\tag{4}
$$

where $\delta ( s )$ is any kind of estimator of the random variable $s ,$ and Θ denotes sufficient statistics for s.

In the context of sequential Monte Carlo estimation, $\mathrm { i . e . , }$ particle filtering via sequential importance sampling, RBPF leverages this principle by analytically marginalizing subsets of the latent state that admit closed-form conditional updates. By maintaining sufficient statistics for these tractable components within each particle, RBPF reduces the dimensionality of the belief representation that must be approximated through sampling. This enables more efficient belief updates by reducing the variance of the resulting particle importance weights [17] [20]. As a result, this leads to using fewer particles compared to purely sampling-based approaches which sample all variables whether they possess analytically tractable conditional posterior sufficient statistics or not.

## III. TECHNICAL APPROACH

Incorporating analytically tractable belief components within sampling-based approximate POMDP solvers requires not only modifying how belief is represented but also altering how its conditional uncertainty is propagated during online planning. A recent work [24] introduced such an approach, referred to as the RB-POMDP framework, in which each particle is associated with a conditional analytical distribution over marginalized state variables as shown in Fig. 2. This section describes the proposed method for addressing these challenges.

Algorithm 1 Belief Updates for Rao-Blackwellized Particle   
Filter adapted from [24]   
Require: $( s _ { k } ^ { \pi , i } , w _ { k } ^ { i } , \Theta _ { k } ^ { \alpha | \pi , i } ) _ { i = 1 } ^ { N _ { s } } , o _ { k + 1 } , a _ { k + 1 }$   
Ensure: $( s _ { k + 1 } ^ { \pi , i } , w _ { k + 1 } ^ { i } , \Theta _ { k + 1 } ^ { \alpha | \pi , i } ) _ { i = 1 } ^ { N _ { s } }$   
1: for $i = 1$ to $N _ { s }$ do   
2: Draw $s _ { k + 1 } ^ { \pi , i } \sim p ( s _ { k + 1 } ^ { \pi , i } | s _ { k } ^ { \pi , i } , a _ { k + 1 } )$   
3: $\Theta _ { k + 1 } ^ { \alpha | \pi , i } , \Lambda _ { k + 1 } ^ { \dot { \alpha } | \pi , i } \gets \mathrm { A n a l y t U p d } \left( \Theta _ { k } ^ { \alpha | \pi , i } , s _ { k + 1 } ^ { \pi , i } , o _ { k + 1 } , a _ { k + 1 } \right)$   
4: $\tilde { w } _ { k + 1 } ^ { i } \gets \mathrm { R e w e i g h t } \left( \Theta _ { k + 1 } ^ { \alpha | \pi , i } , \Lambda _ { k + 1 } ^ { \dot { \alpha } | \pi , i } , s _ { k + 1 } ^ { \pi , i } , o _ { k + 1 } \right)$   
5: end for   
6: $\begin{array} { r l } & { \frac { \mathsf { c m } \cdot \mathbf { r } \mathbf { u } ^ { s } } { \{ w _ { k + 1 } ^ { i } \} _ { i = 1 } ^ { N _ { s } } } = \mathrm { N o r m a l i z e } \left( \{ \tilde { w } _ { k + 1 } ^ { i } \} _ { i = 1 } ^ { N _ { s } } \right) } \end{array}$   
7: Compute $\begin{array} { r } { N _ { \mathrm { e s s } } = \frac { 1 } { \sum _ { s = 1 } ^ { N _ { s } } ( w _ { k + 1 } ^ { i } ) ^ { 2 } } } \end{array}$   
8: if $N _ { \mathrm { e s s } } < \tau$ then   
9: $\begin{array} { r l } & { \{ s _ { k + 1 } ^ { \pi , i } , w _ { k + 1 } ^ { i } \} _ { i = 1 } ^ { N _ { s } }  \mathrm { R e s a m p l e } ( \{ s _ { k + 1 } ^ { \pi , i } , w _ { k + 1 } ^ { i } \} _ { i = 1 } ^ { N _ { s } } ) } \\ & {   \sum _ { s } \ldots  } \end{array}$   
10: end if

## A. Rao-Blackwell factorization of POMDPs

The key idea underlying Rao-Blackwellization is to exploit conditional structure in the latent state by decomposing it into analytically tractable and non-tractable components. Let the state be partitioned as $s _ { t } = ( s _ { t } ^ { \pi } , s _ { t } ^ { \alpha | \pi } )$ , where $s _ { t } ^ { \pi }$ denotes the subset of variables that must be represented through randomly sampled particles and $s _ { t } ^ { \alpha | \pi }$ denotes variables whose conditional posterior admits a closed-form update. Using the chain rule, the joint posterior distribution over these components can be factorized as

$$
p ( s _ { t + 1 } ^ { \alpha | \pi } , s _ { t + 1 } ^ { \pi } \mid o _ { 1 : t } ) = p ( s _ { t + 1 } ^ { \alpha | \pi } \mid s _ { t + 1 } ^ { \pi } , o _ { 1 : t } ) p ( s _ { t + 1 } ^ { \pi } \mid o _ { 1 : t } ) .\tag{5}
$$

This factorization enables analytical marginalization of the tractable component $s _ { t + 1 } ^ { \alpha | \pi }$ conditioned on sampled realizations of $s _ { t + 1 } ^ { \pi }$ . In other words, the conditional belief $p ( s _ { t + 1 } ^ { \alpha | \pi }$ $s _ { t + 1 } ^ { \pi } , o _ { 1 : t } )$ can be propagated using closed-form updates (e.g., Kalman filtering). When such a decomposition exists, sampling is required only over the non-tractable component $s _ { t + 1 } ^ { \pi }$ effectively reducing the dimensionality of the particles.

## B. RBPF Belief Updates

In the most commonly used particle filtering variant, Sequential Importance Resampling particle filters (SIRPF) (also known as Bootstrap particle filters), a common choice for the proposal distribution is the transition function $p ( s ^ { \prime } \mid s , a )$ which leads to importance weights that are proportional to the observation likelihood [27]. In RBPFs, this likelihood is evaluated using the analytically updated belief over the tractable state components [28], as indicated by Λ in line 3 of Algorithm 1. When using Kalman filters, for instance, this corresponds to the filter innovation likelihood function for a received observation and involves the innovation covariance matrix computed from the measurement update step.

![](images/de38eb3f976a9334a1a75a30ad9d58c7575868384dc08208084ede2c14d5e999.jpg)  
Fig. 2. Tree structure comparison adapted from [24]. Each square and larger circle represents an action node and an observation node, respectively. In the POMCPOW tree, particles are shown as black dots while each particle in the RB-POMCPOW tree is associated with a Gaussian distribution. Due to the nature of POMCPOW, these particles form a weighted mixture of beliefs.

When alternative proposal distributions are employed, such as the measurement-informed proposal used in FastSLAM 2.0, the importance weights must be computed with additional care. The corresponding computation for this likelihood is described in detail in Section V for our application problem.

Unlike standard SIRPF approaches, where the weights are based solely on sampled observation likelihoods, the RBPF formulation leverages sufficient statistics (Θ) maintained within each particle. This conditioning mechanism reduces the variance of importance weights [17] and thereby enables lower variance belief updates in accordance with the Rao-Blackwell theorem. The overall Rao-Blackwellized belief update procedure is summarized in Algorithm 1.

## C. Online Planning with Rao-Blackwellization

The key modification required to incorporate Rao-Blackwellized belief representations into sampling-based online POMDP solvers arises during generative evaluation inside the planning tree. In standard POMCP and POMCPOW, rewards and observations are simulated by drawing a fully instantiated state realization from the particle-based belief at each tree expansion step. However, when analytically tractable components of the state are maintained as conditional distributions within each Rao-Blackwellized particle, randomly sampling these variables from their distributions during simulation introduces additional uncertainty into predicted observations and rewards.

Instead, planning can be performed by computing the expected values with respect to the analytically maintained conditional distribution. For states consistent with a given history h, the immediate reward can be approximated as

$$
\mathcal { R } ( h ) \approx \frac { 1 } { \vert I ( h ) \vert } \sum _ { i \in I ( h ) } \mathbb { E } _ { s _ { i } ^ { \alpha \vert \pi } } \left[ \mathcal { R } ( s _ { i } ^ { \pi } , s _ { i } ^ { \alpha \vert \pi } , a _ { i } ) \right] .\tag{6}
$$

A straightforward approach is to approximate this expectation via Monte Carlo sampling by drawing multiple realizations from the analytical distribution. However, drawing a large number of samples negates the computational benefits of Rao-Blackwellization, while again, using a single sample fails to adequately capture the underlying uncertainty and leads to high-variance value estimates. This increased variance degrades the accuracy of value estimates within the planning tree and typically requires substantially more tree simulations for convergence.

To address this issue, we instead approximate these expectations using deterministic quadrature methods. Quadrature techniques explicitly account for uncertainty in analytically tractable state components by replacing the expectation with a weighted sum over M quadrature points $s _ { i , d , k } ^ { \alpha \bar { | } \pi }$ and corresponding weights $w _ { i , d , k }$ . The resulting value estimate is given by the following:

$$
V ( h ) \approx \frac { 1 } { \vert I ( h ) \vert } \sum _ { i \in I ( h ) } \sum _ { d = \tau } ^ { D } \gamma ^ { d - \tau } \sum _ { k = 1 } ^ { M } w _ { i , d , k } \mathcal { R } ( s _ { i , d } ^ { \pi } , s _ { i , d , k } ^ { \alpha \vert \pi } , a _ { i , d } ) .\tag{7}
$$

This comparison between quadrature and Monte Carlo methods is illustrated in Fig. 3 which corresponds to lines 17 and 18 in ROLLOUT and lines 27 and 34 in SIMULATE functions in Algorithm 2.

When using quadrature techniques, the choice of method depends on the distribution used for the analytically marginalized states. For example, we can employ Gaussian-Hermite quadrature to interpolate and perform the necessary integration when using Gaussian beliefs for the linear states [29]. If different distributions were considered, other methods from the Askey family of orthogonal polynomials could be utilized [30] [31].

To mitigate the curse of dimensionality associated with high-order quadrature, we employ Smolyak sparse grids, which reduce the number of quadrature points required while maintaining approximation accuracy [32] [33]. The Smolyak formula for constructing the sparse grid is:

$$
\mathcal { A } ( \boldsymbol { q } , d ) = \sum _ { \substack { q - d + 1 \leq | \mathbf { i } | \leq q } } ( - 1 ) ^ { q - | \mathbf { i } | } { \binom { d - 1 } { q - | \mathbf { i } | } } \bigotimes _ { j = 1 } ^ { d } \mathcal { Q } _ { i _ { j } } ,\tag{8}
$$

where $\scriptstyle A ( q , d )$ is the set of Smolyak quadrature points and their corresponding weights for a given sparse grid level $q$ and dimensionality d, i is a multi-index with $i _ { j }$ representing the level of the univariate quadrature rules, and $\otimes _ { j = 1 } ^ { d } \mathcal { Q } _ { i _ { j } }$ denotes the tensor product of the univariate quadrature rules $\mathcal { Q } _ { i _ { j } }$ at levels $i _ { j }$ for each dimension $j .$

Higher sparse grid levels yield more accurate expectation estimates at the cost of increased computational time. Hence, by choosing different levels of sparse grid, we can fine-tune the trade off between computational cost and integration accuracy in the planner. Overall, this approach reduces the reliance on Monte Carlo sampling for all states by capturing the marginalized states using deterministic quadrature techniques and ultimately reduces the total number of tree iterations needed in POMCP and POMCPOW.

The RB-POMCPOW pseudocode, adapted from [24], is presented in Algorithm 2 with the modified code written in blue. Quadrature techniques are used in lines 17 and

![](images/5c909946689b74bbf0f5398bd276b3038978646c4fae404a103e543f4ec641c0.jpg)

![](images/d693986888962a3fc372c6734d55308dcc6b68361ff11f05d35e24db9ecb1b51.jpg)  
Fig. 3. Illustration of the Rao-Blackwellized particle generative evaluation and analytical update during online planning. Top: Quadrature-based integration A selected Rao-Blackwellized particle consists of sampled (particle-based) state components and analytically maintained conditional distributions. During generative evaluation, deterministic quadrature nodes are combined with the particle state to produce multiple fully instantiated and weighted next states, rewards, and observations. Although each realization yields a complete next state, only the sampled state components are used for the analytical belief update. The analytical state components are not averaged across these realizations; instead, they are updated using the expected observation and the expected next sampled component computed via their corresponding quadrature weights, yielding an updated analytical distribution. This results in a new Rao-Blackwellized particle stored at the corresponding tree node. Bottom: Monte Carlo evaluation. Analytically tractable components are randomly sampled to generate a single next-state realization. Integration is no longer needed, and the realized observation and next sampled state component is directly used for belief update. This introduces additional sampling-induced variance into predicted rewards, leading to slower convergence of value estimates and requiring substantially more tree simulations, as validated in Section VI-B.

27. Note that the sparse grid level or the integration methods can be adjusted independently for the ROLLOUT and SIMULATE functions to compute the expectations. While this study primarily focuses on the RB-POMCPOW algorithm, the underlying principles for handling uncertainty in the analytical distributions of tractable states remain the same for POMCP (refer to Appendix for the pseudocode).

While quadrature techniques are used to approximate expectations for continuous analytically maintained belief components, the RB-POMDP framework is not restricted to such distributions. In particular, when marginalized variables have finite discrete support, expectations can be computed exactly. For example, if X ∼ Bern(p), then

$$
\begin{array} { r } { \mathbb { E } [ R ( X ) ] = ( 1 - p ) R ( 0 ) + p R ( 1 ) . } \end{array}\tag{9}
$$

This again helps avoid randomly sampling stochastic realizations when evaluating expected rewards and observations during tree expansion. In general, any marginalized state component admitting a closed-form conditional update can be incorporated within the proposed RB-POMDP framework.

## IV. PROBLEM FORMULATION

We consider a robotic search-and-rescue task in an initially unknown indoor environment populated with a set of $N \in \mathbb N$ static landmarks, each of which may or may not be associated with a victim. The robot must operate under partial observability, simultaneously localizing itself, estimating landmark geometry, and reasoning about latent semantic information associated with each landmark in order to make sequential decisions that maximize a task-specific objective (i.e., locating all victims in the environment). An overview of the experimental setup is shown in Fig. 5.

## A. Problem Statement

At each discrete time step t, the robot executes an action $a _ { t } \in \ A .$ , transitions to a new state $s _ { t } ~ \in ~ S$ according to stochastic dynamics, and receives a noisy observation $o _ { t } \in \mathcal { O }$ Observations are composed of both geometric measurements $z _ { t }$ and semantic measurements $y _ { t }$ , such that $o _ { t } ~ = ~ ( z _ { t } , y _ { t } )$ Geometric observations provide information about landmark locations relative to the robot, while semantic observations provide probabilistic evidence regarding latent semantic attributes associated with each landmark. Therefore, the underlying state space contains both continuous geometric variables and discrete semantic variables that must be inferred from noisy observations over time. The robot’s objective is to select actions that maximizes the expected cumulative reward by efficiently visiting landmarks with high semantic likelihood.

Algorithm 2 RB-POMCPOW adapted from [24]   
1: procedure SEARCH(h) 26: a ← ACTIONPROGWIDEN(h)   
2: for $i \gets 1$ to N do 27: $( \hat { s ^ { \prime } } , \hat { o } , \hat { r } ) \gets \mathbb { E } [ G ( p , a ) ]$ ▷ Quadrature Techniques   
3: $\mathrm { S I M U L A T E } ( p \sim b ( h ) , h , d _ { m a x } )$ 28: if $| C ( h a ) | \leq \dot { k } _ { o } \bar { N } ( h \dot { a } ) ^ { \alpha _ { o } }$ then   
4: return arg max<sub>a</sub> Q(ha); 29: $o \gets \hat { o }$   
5: 30: $M ( h a o )  M ( h a o ) + 1$   
6: procedure ACTIONPROGWIDEN(h) 31: else   
7: if $| { \mathcal { C } } ( h ) | \leq k _ { a } N ( h ) ^ { \alpha _ { a } }$ then 32: o ← select $o \in C ( h a )$ w.p. $\frac { M ( h a o ) } { \sum _ { o } M ( h a o ) }$   
8: $a \gets \mathsf { N E X T A C T I O N } ( h )$ 33: end if   
9: ${ \mathcal { C } } ( h ) \gets { \mathcal { C } } ( h ) \cup \{ a \}$ 34: p<sup>′</sup> ←ANALYTICALUPDAT $\therefore ( p , \hat { s ^ { \prime } } , o , a )$   
10: return $\begin{array} { r } { \arg \operatorname* { m a x } _ { a \in \mathcal { C } ( h ) } \left[ Q ( h a ) + c \sqrt { \frac { \log N ( h ) } { N ( h a ) } } \right] } \end{array}$ 35: append $( p ^ { \prime } , \hat { r } )$ to B(hao)   
11: 36: append $\mathcal { Z } ( o \mid p , a , p ^ { \prime } )$ to W(hao)   
12: procedure ROLLOUT(p, h, d) 37: if $\bar { o } \notin C ( h \bar { a } )$ then ▷ new node   
13: $\mathbf { i f } \ \gamma ^ { \mathrm { d e p t h } } < \epsilon$ then 38: $C { \dot { ( } } h a { \dot { ) } } \gets C ( h a ) \cup \{ o \}$   
14: return 0 39: total $\gets \hat { r } + \mathrm { \hat { \gamma } } \ast$ ROLLOUT(p<sup>′</sup>, hao, d − 1)   
15: end if 40: else   
16: $\begin{array} { r } { a \sim \pi _ { \mathrm { r o l l o u t } } ( h , \cdot ) } \end{array}$ 41: (p<sup>′</sup>, r) ← select B(hao)[i] w.p. $\frac { W ( h a o ) [ i ] } { \sum _ { j = 1 } ^ { m } W ( h a o ) [ j ] }$   
17: $( \hat { s ^ { \prime } } , \hat { o } , \hat { r } ) \mathrel { \mathop \longleftarrow } \mathbb E [ G ( p , a ) ]$ ▷ Quadrature Techniques 42: total ← r + γSIMULATE(p<sup>′</sup>, hao, d − 1)   
18: $p ^ { \prime } \gets \mathrm { A N A L Y T I C A L U P D A T E } ( p , \hat { s ^ { \prime } } , \hat { o } , a )$ 43: end if   
19: return $\hat { r } + \gamma$ · ROLLOUT(p<sup>′</sup>, hao, dˆ − 1) 44: $N ( h )  N ( h ) + 1$   
20: end procedure 45: $N ( h a )  N ( h a ) + 1$   
21: 46: $\begin{array} { r } { Q ( h a ) \gets Q ( h a ) + \frac { t o t a l - Q ( h a ) } { N ( h a ) } } \end{array}$   
22: procedure SIMULATE(p, h, d) 47: return total   
23: if d = 0 then 48: end procedure   
24: return 0   
25: end if

This decision-making problem can be naturally formulated as a POMDP as detailed below. Importantly, because the state includes not only robot pose but also landmark geometry and latent semantic variable whose dimensionality grows with the number of landmarks, this results in a high-dimensional belief representation that poses significant challenges for samplingbased online POMDP solvers.

## B. State Space S

The agent’s state at time t is defined as

$$
s _ { t } = \left( x _ { t } , \{ \theta _ { n , t } \} _ { n = 1 } ^ { N } , \{ e _ { n , t } \} _ { n = 1 } ^ { N } , \{ v _ { n , t } \} _ { n = 1 } ^ { N } \right) ,\tag{10}
$$

where $x _ { t } ~ \in ~ \mathbb { R } ^ { d _ { x } }$ denotes the robot pose (i.e., position and orientation), $\theta _ { n , t } \in \mathbb { R } ^ { d _ { \theta } }$ denotes the geometric location of the n-th landmark, $e _ { n , t } \in \{ 0 , 1 \}$ denotes a latent semantic variable indicating the presence or absence of a victim associated with the n-th landmark, and $v _ { n , t } \in \{ 0 , 1 \}$ denotes a visited indicator used for bookkeeping and reward evaluation. This structured decomposition will later be exploited in section V-A to analytically marginalize subsets of the state within the proposed Rao-Blackwellized belief representation.

## C. Action Space A

At each time step, the robot selects an action $a _ { t } = ( v _ { t } , \omega _ { t } )$ where $v _ { t }$ denotes the linear velocity and $\omega _ { t }$ denotes the angular velocity. The action space is discretized to include forward, backward, and left and right turning motions. In addition to motion actions, the robot is equipped with a scanning action that does not induce motion but allows the robot to directly observe the latent semantic state of a landmark when the robot is within the sensing range. Successful identification of a victim through this action contributes to the detection reward defined in IV-F.

## D. Transition Model T

The agent dynamics are governed by the robot motion model and deterministic bookkeeping updates while landmark location and latent semantic variables are assumed to remain static. Given state $s _ { t }$ and action $a _ { t } ,$ , the robot pose evolves according to

$$
x _ { t + 1 } = h ( x _ { t } , a _ { t } ) + \delta _ { t } ,\tag{11}
$$

where h denotes a nonlinear unicycle motion model and $\delta _ { t } \sim$ $\mathcal { N } ( 0 , Q _ { t } )$ represents zero-mean Gaussian process noise. While additive Gaussian noise is assumed here for our experimental setup, both the transition and observation models are not required to follow additive noise assumptions. We adopt the notation convention used in [18] in accordancce to the SLAM literature.

The visited indicators $\{ v _ { n , t } \} _ { n = 1 } ^ { N }$ are updated deterministically when the robot executes a scan action. If the robot is within a predefined radius of landmark $n ,$ then the corresponding visited flag is set to one. Otherwise, it remains unchanged.

## E. Observations and Observation Model O, Z

At each time step, the robot receives observations $o _ { t } =$ $\begin{array} { r c l } { ( z _ { t } , y _ { t } ) } & { = } & { \{ ( z _ { n , t } , y _ { n , t } ) \} _ { n \in \mathcal { N } _ { t } } } \end{array}$ consisting of geomeric and semantic measurements associated with a set of visible landmarks $\mathcal { N } _ { t } \ \subseteq \ \{ 1 , \ldots , N \}$ . We assume that data association is known (i.e., each observation is associated with a unique landmark index). While FastSLAM has been extended to handle unknown data association [18], this assumption allows us to focus on belief representation and its integration within the POMDP framework.

Geometric observations provide noisy landmark positions relative to the robot and are received whenever landmarks are within the sensing range. Specifically, the geometric observation associated with landmark $n \in \mathcal N _ { t }$ is modeled as

$$
z _ { n , t } = g ( x _ { t } , \theta _ { n , t } ) + \epsilon _ { n , t } ,\tag{12}
$$

where $g$ denotes a nonlinear range-and-bearing measurement function and $\epsilon _ { n , t } \sim \mathcal { N } ( 0 , R _ { t } )$ is zero-mean Gaussian noise. Again, additive Gaussian observation noise assumption is not required.

Semantic observations provide information about the latent semantic variable $e _ { n , t }$ associated with each landmark. Such observations are also assumed to be available once the robot executes a sensing action within the sensing range. We assume conditional independence across landmarks and sensing modalities. Details of the semantic measurement model and its role in the importance weight computation are provided in Section V-D.

## F. Reward R

The reward function is designed to encourage efficient exploration toward potential victim locations while penalizing unnecessary motion and control effort:

$$
\begin{array} { r l } & { R ( s _ { t } , a _ { t } , s _ { t + 1 } ) = \underbrace { R _ { \mathrm { f o u n d } } ( s _ { t } , a _ { t } , s _ { t + 1 } ) } _ { \mathrm { d e t e c t i o n ~ r e w a r d } } + \underbrace { k _ { p } v _ { t } \cos ( e _ { \psi _ { t } } ) } _ { \mathrm { r a d i a l ~ m o t i o n ~ r e w a r d } } } \\ & { \qquad - \underbrace { ( k _ { v } v _ { t } ^ { 2 } + k _ { \omega } \omega _ { t } ^ { 2 } ) } _ { \mathrm { c o n t r o l ~ p e n a l t y } } - \underbrace { k _ { \mathrm { l a t } } | v _ { t } | \sin ^ { 2 } ( e _ { \psi _ { t } } ) } _ { \mathrm { l a t e r a l ~ m o t i o n ~ p e n a l t y } } - \underbrace { k _ { h } ( 1 - \cos ( e _ { \psi _ { t } } ) ) } _ { \mathrm { h e a d i n g ~ p e n a l t y } } , } \end{array}\tag{13}
$$

where $e _ { \psi _ { t } }$ denotes the heading error between the robot orientation and the direction toward the nearest, unvisited target landmark with high semantic probability, and $k _ { p } , \ k _ { v } , \ \breve { k } _ { \omega } ,$ $k _ { \mathrm { l a t } } ,$ and $k _ { h }$ are positive weighting coefficients. The detection reward is defined as

$$
\begin{array} { r l r } {  { R _ { \mathrm { f o u n d } } \bigl ( s _ { t } , a _ { t } , s _ { t + 1 } \bigr ) } } \\ & { } & { \qquad = \displaystyle \sum _ { n = 1 } ^ { N } \mathbb { I } [ a _ { t } = a _ { \mathrm { s c a n } } \wedge v _ { n , t + 1 } = 1 \wedge v _ { n , t } = 0 \wedge e _ { n , t } = 1 ] R _ { 0 } , } \\ & { } & \end{array}\tag{14}
$$

where $R _ { 0 }$ is a constant positive reward associated with successfully identifying a victim..

## V. METHODOLOGY

## A. Rao-Blackwellized Belief Representation

Conditioned on the robot trajectory, landmark states are assumed to be independent, as is standard in FastSLAMbased formulations. The posterior then admits the following factorization

$$
{ \begin{array} { r l } & { p \left( { \boldsymbol { x } } ^ { t } , \left\{ \theta _ { n , t } , e _ { n , t } , v _ { n , t } \right\} _ { n = 1 } ^ { N } \mid { \boldsymbol { o } } ^ { t } , { \boldsymbol { a } } ^ { t } \right) } \\ & { \qquad = p ( { \boldsymbol { x } } ^ { t } , \left\{ v _ { n , t } \right\} _ { n = 1 } ^ { N } \mid { \boldsymbol { o } } ^ { t } , { \boldsymbol { a } } ^ { t } ) \prod _ { n = 1 } ^ { N } p ( \theta _ { n , t } , e _ { n , t } \mid { \boldsymbol { x } } ^ { t } , { \boldsymbol { o } } ^ { t } , { \boldsymbol { a } } ^ { t } ) } \end{array} }\tag{15}
$$

where $\boldsymbol { x } ^ { t } , \boldsymbol { o } ^ { t }$ , and $a ^ { t }$ denote the histories of robot trajectory, observations, and actions up to time step t, respectively.

The belief is approximated using a set of M Rao-Blackwellized particles. Each particle represents a sampled robot trajectory and deterministic bookkeeping variables, combined with an associated map in which landmark geometry and semantic attributes are maintained analytically. In particular, each particle carries N EKFs for geometric landmark states and N Bernoulli distributions for semantic landmark states. Formally, the m-th particle at time t is given by

$$
S _ { t } ^ { [ m ] } = \Bigg ( x ^ { t , [ m ] } , \Big \{ \underbrace { \big ( \mu _ { n , t } ^ { [ m ] } , \Sigma _ { n , t } ^ { [ m ] } } _ { \mathrm { E K F } } , \underbrace { \pi _ { n , t } ^ { [ m ] } } _ { \mathrm { B e r n o u l l i } } , v _ { n , t } ^ { [ m ] } \big ) \Big \} _ { n = 1 } ^ { N } \Bigg ) ,\tag{16}
$$

where $( \mu _ { n , t } ^ { [ m ] } , \Sigma _ { n , t } ^ { [ m ] } )$ denote the sufficient statistics of the Gaussian estimate of landmark geometry maintained by each EKF and $\pi _ { n , t } ^ { [ m ] }$ denotes the Bernoulli parameter associated with the semantic variable $e _ { n , t }$

## B. Proposal Distribution

The efficiency of particle-based belief updates depends on the proposal distribution used for sampling. We adopt the measurement-informed proposal distribution from Fast-SLAM 2.0. The key idea is to sample robot poses by incorporating both the control input and the geometric landmakr measurements to reduce particle depletion compared to sampling from the motion model alone [21]. Specifically, the proposal distribution for the robot state at time t is given by

$$
\begin{array} { r l } & { q \left( x _ { t } \mid x _ { t - 1 } ^ { [ m ] } , a ^ { t } , z ^ { t } \right) \propto \underbrace { p \left( x _ { t } \mid x _ { t - 1 } ^ { [ m ] } , a _ { t } \right) } _ { \sim \mathcal { N } \left( x _ { t } ; h ( x _ { t - 1 } ^ { [ m ] } , a _ { t } ) , Q _ { t } \right) } } \\ & { \phantom { f _ { \theta \pi _ { t } \pi _ { t } } } \int \underbrace { p \left( z _ { n , t } \mid \theta _ { n , t } , x _ { t } \right) } _ { \sim \mathcal { N } \left( z _ { n , t } ; g ( \theta _ { n , t } , x _ { t } ) , R _ { t } \right) } \underbrace { p \left( \theta _ { n , t } \mid x ^ { t - 1 , [ m ] } , z ^ { t - 1 } \right) } _ { \sim \mathcal { N } \left( \theta _ { n , t } ; \mu _ { n , t - 1 } ^ { [ m ] } , \Sigma _ { n , t - 1 } ^ { [ m ] } \right) } d \theta _ { n , t } } \end{array}\tag{17}
$$

Although multiple landmark measurements may be available at time t, the proposal is constructed by incorporating landmark measurements sequentially. For notational clarity, this is illustrated here for a single measurement $z _ { n , t }$ following [21]. The above proposal does not admit a closed form solution in general. However, we can approximate the measurement function using a first-order Taylor expansion, yielding a Gaussian proposal distribution. Under this EKF-style approximation, the proposal distribution becomes

$$
\begin{array} { r } { \boldsymbol { q } ( \boldsymbol { x } _ { t } \mid \boldsymbol { x } _ { t - 1 } ^ { [ m ] } , \boldsymbol { a } ^ { t } , \boldsymbol { z } ^ { t } ) = \mathcal { N } \Big ( \boldsymbol { x } _ { t } ; \mu _ { \boldsymbol { x } _ { t } } ^ { [ m ] } , \boldsymbol { \Sigma } _ { \boldsymbol { x } _ { t } } ^ { [ m ] } \Big ) , } \end{array}\tag{18}
$$

where $\mu _ { x _ { t } } ^ { [ m ] }$ and $\Sigma _ { x _ { t } } ^ { [ m ] }$ are computed as described in [21].

## C. Importance Weight Computation

Note that semantic observations are not used in the proposal distribution. However, they still must be incorporated in the importance weights computation by marginalizing the continuous landmark geometry, robot’s pose, and the discrete semantic variables. Under the assumption of conditional independence across landmarks and sensing modalities, the overall observation likelihood factorizes as

$$
p ( o _ { t } \mid s _ { t } ) = \prod _ { n \in \mathcal { N } _ { t } } p ( z _ { n , t } \mid x _ { t } , \theta _ { n , t } , e _ { n , t } ) p ( y _ { n , t } \mid x _ { t } , \theta _ { n , t } , e _ { n , t } ) .\tag{19}
$$

Therefore, for an observation associated with landmark $n ,$ the weight update for particle m can be written as

$$
\begin{array} { r l } & { w _ { n , t } ^ { [ m ] } \propto p \Bigg ( z _ { n , t } , y _ { n , t } \mid x ^ { t - 1 , [ m ] } , a ^ { t } , o ^ { t - 1 } \Bigg ) } \\ & { \quad = \displaystyle \sum _ { e \in \{ 0 , 1 \} } p \left( e _ { n , t } = e \mid x ^ { t - 1 , [ m ] } , a ^ { t } , o ^ { t - 1 } \right) } \\ & { \qquad \displaystyle \iint \boldsymbol { p } ( y _ { n , t } \mid x _ { t } , \theta _ { n , t } , e ) \underbrace { p \left( z _ { n , t } \mid x _ { t } , \theta _ { n , t } \right) } _ { \displaystyle < \mathcal { N } \big ( z _ { n , t } \mid g ( n _ { t } , x , t ) , R _ { t } \big ) } } \\ & { \qquad \underbrace { p \Big ( \theta _ { n , t } \mid x ^ { t - 1 , [ m ] } , a ^ { t - 1 } , z ^ { t - 1 } \Big ) } _ { \displaystyle \sim \mathcal { N } \big ( \theta _ { n , t } ; \mu _ { n , t - 1 } ^ { [ m ] } , x _ { t - 1 } ^ { [ m ] } \big ) } d \theta _ { n , t } \underbrace { p \big ( x \mid x _ { t - 1 } ^ { [ m ] } , a t \big ) } _ { \displaystyle \sim \mathcal { N } \big ( z _ { t } ; \mu _ { n , t } ^ { [ m ] } , \sigma _ { t } \big ) , \mathcal { Q } _ { t } } d x _ { t } } \end{array}\tag{20}
$$

Following FastSLAM 2.0, the integrals over the robot pose and landmark geometry are again approximated in closed form by linearizing the measurement model $^ { g , }$ yielding a Gaussian likelihood evaluation. To preserve a fully closed-form importance weight computation, the semantic observation model must be carefully designed, as detailed in the next section V-D.

## D. Semantic Observation Modeling

One way to ensure that (20) remains analytically tractable is to model the semantic likelihood such that it admits a closed-form expression when marginalized under the Gaussian uncertainty induced by the robot pose and landmark geometry. Specifically, we model semantic detection as a Bernoulli random variable whose success probability is defined as a function of the range defined as d:

$$
p ( y _ { n , t } = 1 \mid \theta _ { n , t } , x _ { t } , e _ { n , t } = 1 ) = P _ { D } \big ( d ( x _ { t } , \theta _ { n , t } ) \big ) .\tag{21}
$$

To preserve analytical tractability, the detection function $P _ { D } ( d )$ must be chosen such that marginalization over a Gaussian-distributed range,

$$
\begin{array} { r } { d \sim \mathcal { N } \Big ( \hat { d } _ { n } ^ { [ m ] } , \sigma _ { n , d } ^ { 2 [ m ] } \Big ) , } \end{array}\tag{22}
$$

admits a closed-form solution. The mean and variance of the range distribution are again approximated via first-order linearization as

$$
\begin{array} { r } { \hat { d } _ { n } ^ { [ m ] } = f \left( \hat { x } _ { t } ^ { [ m ] } , \hat { \theta } _ { n , t } ^ { [ m ] } \right) , \quad \sigma _ { n , d } ^ { 2 [ m ] } = J _ { x } Q _ { t } J _ { x } ^ { \top } + J _ { \theta _ { n } } \Sigma _ { n , t - 1 } ^ { [ m ] } J _ { \theta _ { n } } ^ { \top } , } \end{array}\tag{23)(23}
$$

where $\hat { x } _ { t } ^ { [ m ] } \ = \ h ( x _ { t - 1 } ^ { [ m ] } , a _ { t } )$ and $\hat { \theta } _ { n , t } ^ { [ m ] } ~ = ~ \mu _ { n , t - 1 } ^ { [ m ] }$ denote the predicted robot pose and predicted landmark location, respectively, and $J _ { x }$ and $J _ { \theta _ { n } }$ are the Jacobians of f with respect to $x _ { t }$ and $\theta _ { n , t }$

While commonly used link functions such as the logistic function do not yield closed-form expressions when marginalized under Gaussian uncertainty, the probit link yields an exact analytical solution. Using the probit model, we therefore define the detection probabilities as

$$
\begin{array} { l } { { \displaystyle \bar { P } _ { D } ^ { [ m ] } = \mathbb { E } _ { d \sim \mathcal { N } ( \hat { d } _ { n } ^ { [ m ] } , \sigma _ { n , d } ^ { 2 [ m ] } ) } [ P _ { D } ( d ) ] } } \\ { { \displaystyle \quad = \int \Phi ( \alpha _ { 0 } + \alpha _ { 1 } d ) \mathcal { N } \Big ( d ; \hat { d } _ { n } ^ { [ m ] } , \sigma _ { n , d } ^ { 2 [ m ] } \Big ) d d = \Phi \left( \frac { \alpha _ { 0 } + \alpha _ { 1 } \hat { d } _ { n } ^ { [ m ] } } { \sqrt { 1 + \alpha _ { 1 } ^ { 2 } \sigma _ { n , d } ^ { 2 [ m ] } } } \right) } } \end{array}\tag{24}
$$

where $\Phi$ denotes the standard normal cumulative distribution function and $\alpha _ { 0 } , \alpha _ { 1 }$ are model parameters. A derivation of this closed-form expression is provided in Appendix VII. Under this formulation, the expected detection and similarly expected false-alarm probabilities (i.e. $\bar { P } _ { F A } ^ { [ m ] }$ with different model parameters $\beta _ { 0 }$ and $\beta _ { 1 } )$ can be computed in closed form. This enables the importance weight computation in (20) to be performed analytically:

$$
\begin{array} { r l } & { w _ { n , t } ^ { [ m ] } \approx \mathcal { N } \big ( z _ { n , t } ; ~ \hat { z } _ { n , t } ^ { [ m ] } , ~ \Lambda _ { n , t } ^ { [ m ] } \big ) \Big [ \pi _ { n , t } ^ { [ m ] } \big ( \bar { P } _ { D } ^ { [ m ] } \big ) ^ { y _ { n , t } } \big ( 1 - \bar { P } _ { D } ^ { [ m ] } \big ) ^ { 1 - y _ { n , t } } } \\ & { ~ + \big ( 1 - \pi _ { n , t } ^ { [ m ] } \big ) \big ( \bar { P } _ { F A } ^ { [ m ] } \big ) ^ { y _ { n , t } } \big ( 1 - \bar { P } _ { F A } ^ { [ m ] } \big ) ^ { 1 - y _ { n , t } } \Big ] , } \end{array}\tag{25}
$$

where $\hat { z } _ { n , t } ^ { [ m ] } = g \left( \hat { x } _ { t } ^ { [ m ] } , \hat { \theta } _ { n , t } ^ { [ m ] } \right)$ is the predicted measurement and the innovation covariance is

$$
\Lambda _ { n , t } ^ { [ m ] } = G _ { x } Q _ { t } G _ { x } ^ { \top } + G _ { \theta } \Sigma _ { n , t - 1 } ^ { [ m ] } G _ { \theta } ^ { \top } + R _ { t } .\tag{26}
$$

Here, the matrices $G _ { x }$ and $G _ { \theta }$ are the Jacobians of $g$ with respect to $x _ { t }$ and $\theta _ { n , t }$ , respectively [18].

## E. Belief update

The update step for geometric landmark states remains the same as in FastSLAM, which is equivalent to the standard EKF measurement update [18] [34]. We therefore focus on the update of the semantic belief associated with each landmark.

The latent semantic state is modeled as a Bernoulli random variable with particle-specific belief $\pi _ { n . t } ^ { [ m ] }$ for landmark n. Hence, given a binary semantic observation $y _ { n , t } ,$ , the semantic belief is updated analytically via Bayes’ rule as

$$
\pi _ { n , t } ^ { [ m ] } = \left[ 1 + \frac { 1 - \pi _ { n , t - 1 } ^ { [ m ] } } { \pi _ { n , t - 1 } ^ { [ m ] } } \left( \frac { \bar { P } _ { F A } ^ { [ m ] } } { \bar { P } _ { D } ^ { [ m ] } } \right) ^ { y _ { n , t } } \left( \frac { 1 - \bar { P } _ { F A } ^ { [ m ] } } { 1 - \bar { P } _ { D } ^ { [ m ] } } \right) ^ { 1 - y _ { n , t } } \right] ^ { - 1 } .\tag{27}
$$

## VI. EXPERIMENTAL RESULTS

## A. Belief Quality and Computation

We first compare the accuracy of RBPF and SIRPF by evaluating their root-mean-square error (RMSE). As shown in the top plot of Figure 4, RBPF achieves low RMSE with a relatively small number of particles, after which further increases in particle count yield only marginal improvements. In particular, approximately 50 particles are sufficient for RBPF to near-optimal accuracy. In contrast, SIRPF requires substantially more number of particles to approach comparable accuracy. Even with 10,000 particles, the RMSE of SIRPF remains higher than that of RBPF using as few as 10 particles. This behavior is consistent with prior results in FastSLAM works, where Rao-Blackwellization substantially improves estimation efficiency and accuracy by analytically marginalizing conditionally tractable state subcomponents [18] [21].

We next analyze the computational costs of belief updates shown in the bottom plot of Figure 4. For an equal number of particles, RBPF incurs higher per-update computation time than SIRPF due to the additional analytical updates required for each landmark belief. This increase is expected as each Rao-Blackwellized particle maintains an EKF for every landmark geometry as well as a Bernoulli belief for each latent semantic variable. Inevitably, updating these analytical belief components incurs additional per-particle computational overhead compared to SIRPF, which samples all state variables directly. However, because RBPF analytically marginalizes both continuous landmark geometry and discrete semantic variables, it decreases the dimensionality of the sampled state space. As a result, RBPF achieves improved estimation accuracy with significantly fewer particles, reducing the overall computational cost required to maintain a belief of similar quality. Thus, despite higher per-particle costs, RBPF still offers superior efficiency in practice by significantly reducing the number of particles required for accurate belief representation.

![](images/3a48feba761940855b291c6288a3b05e76249aea4f4988ba400233d2fffa1e39.jpg)

Computational Cost Comparison Between SIRPF and RBPF  
![](images/8d253d067a5d75e51fead76793b47498be9e1b1aea90643acc6bc73277920d4c.jpg)  
Fig. 4. Comparison of estimation accuracy and computational cost between SIRPF and RBPF. RMSE is computed by combining the squared estimation errors of both the robot pose and all observed landmark positions. Computational cost is computed per belief update at each time step.

## B. Planning Performance and Computation

We next evaluate how belief representation and generative evaluation influence planning performance under a fixed planning computational budget. Representative trajectories generated by each planner are shown in Fig. 5. In Fig. 5 (a), POM-CPOW with a standard SIRPF belief exhibits large trajectory deviations and fails to reach the high-probability victim in Room 4. In Fig. 5 (b), we replace the belief representation with RBPF while retaining Monte Carlo–based generative evaluation during tree expansion. In this setting, analytically maintained landmark geometry and semantic states are randomly sampled to produce a single realization when simulating predicted observations and rewards, as illustrated by the Monte Carlo scheme in Fig. 3. We denote this planner as RB-MC-POMCPOW. Although RB-MC-POMCPOW benefits from reduced belief estimation variance due to Rao-Blackwellization, the use of stochastic realizations of analytically tractable state components during planning introduces additional planning uncertainty. As a result, the planner eventually locates all victim rooms but exhibits unnecessary detours when approaching Room 5 from Room 7. While both sampling-based planners can occasionally generate near-optimal trajectories given sufficient simulations with increased planning computational budget, their performance remains inconsistent due to samplinginduced variance in predicted rewards and observations during tree expansion.

In contrast, RB-POMCPOW replaces these stochastic realizations with deterministic expectation over analytically tractable state components during generative evaluation. This corresponds to the quadrature scheme in Fig. 3. This reduces the variance of predicted outcomes within the planning tree and results in more consistent value estimation, yielding more direct navigation toward rooms with high victim belief, as shown in Fig. 5 (c).

To quantify this effect, Figure 6 shows cumulative reward as a function of per-step planning computation time for the baseline POMCPOW, RB-MC-POMCPOW, and RB-POMCPOW planners. To ensure comparable belief estimation accuracy, we use 50 and 10,000 particles for RBPF and SIRPF, respectively, as established in the previous section. For RB-POMCPOW, performance is evaluated across varying sparse grid levels with fixed tree iteration counts. For POMCPOW and RB-MC-POMCPOW, we only vary the tree iterations because both rely on Monte Carlo generative evaluation.

As expected, increasing either the sparse grid level or the number of tree iterations generally improves cumulative reward by producing more accurate value function estimates. However, computational cost grows proportionally with the sparse grid level for RB-POMCPOW, which limits the number of tree iterations that can be performed within a fixed computational budget. Despite this trade-off, RB-POMCPOW consistently outperforms both POMCPOW and RB-MC-POMCPOW under similar computational budgets. For example, even with 1500 tree iterations, POMCPOW achieves cumulative rewards comparable only to RB-POMCPOW using a sparse grid level of q = 1 with just 100 tree iterations. Similarly, RB-MC-POMCPOW with 500 tree iterations provides only marginal improvement over RB-POMCPOW with q = 1, indicating that improved belief representation alone is insufficient to achieve comparable planning performance.

These results can be explained by two complementary variance-reduction mechanisms. First, RBPF maintains conditionally independent landmark beliefs analytically within each particle, resulting in higher effective sample size compared to SIRPF. This aligns with the Rao-Blackwell theorem, ensuring that each selected Rao-Blackwellized particle provides a highquality representation of the belief state with reduced variance in belief representation. Second, RB-POMCPOW computes expected observations and rewards during generative evaluation by integrating over analytically tractable state components, further reducing uncertainty in predicted outcomes during tree expansion. Together, these effects reduce both belief and planning variance, allowing value function estimates to converge with much fewer tree iterations than purely samplingbased planners.

![](images/cd1a112c1b76c9627ffd0791a345886c721afa79ff1afe117f7b173e88d1e4ec.jpg)  
(a) POMCPOW (SIRPF)

![](images/8ed5f78d345a92f34844c9d378ce8b9ba4ffdf0f305579f2080eaa1b4927db7b.jpg)  
(b) RB-MC-POMCPOW (RBPF)

![](images/904aa09352fd57c62cbec293eeca0704535a66a0a56fa70f8e571a8a19d74c61.jpg)  
(c) RB-POMCPOW (RBPF)

Fig. 5. Effect of planner and belief representation on robot behavior. Representative robot trajectories generated using (a) POMCPOW with a standard SIRPF, (b) RB-MC-POMCPOW with RBPF but performs Monte Carlo–based generative evaluation by drawing a single realization (cf. Monte Carlo schem in Fig. 3), and (c) RB-POMCPOW which instead evaluates expected rewards and observations via analytical integration (cf. Quadrature scheme in Fig. 3) Victims are located at the centers of rooms, and the blue bars indicate the corresponding victim-evidence belief which converges toward the true value over time. Red and blue curves denote the estimated and ground-truth robot trajectories, respectively, while green crosses indicate particle or EKF-based estimates of victim locations. All planners are executed with the same maximum time horizon, computational budget, and hyperparameters. In the representative run shown, POMCPOW with SIRPF (a) fails to reach the high-probability victim in Room 4. RB-MC-POMCPOW (b) eventually reaches all of the victim rooms, but the trajectory exhibits unnecessary detours when approaching Room 5. Although these purely sampling-based planners can occasionally produc optimal trajectories, they exhibit higher variability, reflecting sampling-induced variance in value estimation. In contrast, RB-POMCPOW (c) better accounts for uncertainty with analytical integration at each tree node and results in more direct navigation toward rooms with high victim belief.  
![](images/e74ee071881ea65dd891b3105b66ead5363c0c6f14ad24a350f06b223b33edb5.jpg)  
Fig. 6. Planning performance versus computational cost. Cumulative reward as a function of per-step planning time for POMCPOW, RB-MC-POMCPOW and RB-POMCPOW. For RB-POMCPOW, results are shown for different sparse grid levels (denoted as q) with fixed tree iterations. The dashed red line indicates a theoretical upper bound on cumulative reward obtained by assuming perfect semantic information and optimal trajectory path; therefore, it represents a loose performance reference that cannot be achieved. RB-POMCPOW consistently achieves higher reward than both POMCPOW and RB-MC-POMCPOW under comparable computational budgets.

## VII. CONCLUSION AND FUTURE WORK

This work presented a Rao-Blackwellized online planning framework for high-dimensional POMDPs and validated its effectiveness in an indoor robotic search-and-rescue task by integrating it with the FastSLAM 2.0 algorithm. The results highlight an important relationship between belief quality and planning quality in POMDPs. While RBPFs have long been shown to improve estimation accuracy and efficiency in robotic systems, improving belief performance alone is not the only benefit of Rao-Blackwellization in online planning settings. By analytically marginalizing tractable state components and propagating their uncertainty through deterministic integration, the RB-POMDP framework reduces variance not only in belief estimation but also in value estimation during planning. This reduction in planning variance translates directly into improved computational efficiency, suggesting that Rao-Blackwellized online planning is not merely beneficial but becomes essential for computationally tractable decisionmaking in high-dimensional problems.

However, the effectiveness of Rao–Blackwellized planning depends fundamentally on whether expectations of rewards and observations can be computed tractably with respect to the marginalized state components during generative evaluation. In this work, landmark geometry and semantic attributes are represented using Gaussian and Bernoulli distributions, respectively, both of which admit closed-form belief updates and efficient expectation computation during planning.

In contrast, extending the RB-POMDP framework to occupancy grid-based SLAM such as GMapping [35] [36] presents additional challenges. Occupancy grid maps represent each cell using independent Bernoulli random variables, resulting in a very high-dimensional discrete belief space. Although these cell-wise beliefs are analytically tractable, the raycasting observation model depends jointly on multiple cells along each sensing beam. Therefore, computing the expected rewards and observations during generative evaluation requires marginalization over the joint occupancy configuration of all cells intersecting the ray, resulting in an exponential number of realizations.

While naive Monte Carlo sampling from independent cellwise occupancy distributions is computationally feasible, it fails to enforce spatial coherence in the instantiated map realizations and can produce spatially inconsistent checkerboardlike occupancy patterns. Planning over such sampled map can cause the agent to exhibit hallucinated behaviors during tree expansion.

Recent work on generative observation models for range sensing, including learned LiDAR simulators, offers a promising direction for addressing this limitation by producing realistic map realizations [37] [38] [39] [40]. Integrating such generative sensing models (e.g. [41]) into the Rao-Blackwellized online planning represents an important avenue for future work to enable integrating POMDPs in richer SLAM settings beyond feature-based representations.

## APPENDIX A. RB-POMCP

The following pseudocode outlines the RB-POMCP algorithm. The code is adapted from [10] with the modified code written in blue. The ROLLOUT function is the same as the one defined in Algorithm 2.

```latex
Algorithm 3 RB-POMCP
1: procedure SIMULATE(p,h, depth)
2: if $\gamma ^ { d e p t h } < \epsilon$ then
3: return 0
4: end if
5: if hao /∈ T then
6: for all $a \in A$ do
7: $T ( h a ) \gets ( N _ { \mathrm { i n i t } } ( h a ) , V _ { \mathrm { i n i t } } ( h a ) , \emptyset )$
8: end for
9: return ROLLOUT(p, h, depth)
10: end if
11: a ← arg max<sub>b</sub> $\begin{array} { r } { \left[ V ( h b ) + c \sqrt { \frac { \log N ( h ) } { N ( h b ) } } \right] } \end{array}$
12: $( \hat { s ^ { \prime } } , \hat { o } , \hat { r } ) \gets \mathbb { E } [ \tilde { G } ( p , a ) ]$ ▷ Quadrature Techniques
13: p<sup>′</sup> ← ANALYTICALUPDATE(p, s<sup>ˆ′</sup>, o, a ˆ )
14: $R \gets \hat { r } + \gamma \ ^ { * }$ SIMULATE(p<sup>′</sup>, haoˆ, depth + 1)
15: $N ( h )  N ( h ) + 1$
16: $N ( h a )  N ( h a ) + \underline { { 1 } }$
17: $\begin{array} { r } { V ( h a ) \gets V ( h a ) + \frac { R - V ( h a ) } { N ( h a ) } } \end{array}$
18: return R
19: end procedure
```

## APPENDIX B. CLOSED-FORM MARGINALIZATION OF PROBIT DETECTION MODELS UNDER GAUSSIAN UNCERTAINTY

This appendix derives the closed-form expression

$$
\int \Phi ( \alpha _ { 0 } + \alpha _ { 1 } d ) { \mathcal { N } } ( d ; \mu , \sigma ^ { 2 } ) d d = \Phi \left( { \frac { \alpha _ { 0 } + \alpha _ { 1 } \mu } { \sqrt { 1 + \alpha _ { 1 } ^ { 2 } \sigma ^ { 2 } } } } \right) ,\tag{28}
$$

which is used in the main text with $\mu = \hat { d } _ { n } ^ { [ m ] }$ and $\sigma ^ { 2 } = \sigma _ { n , d } ^ { 2 [ m ] }$ in (24). Let $d \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ and define the probit detection model

$$
P _ { D } ( d ) = \Phi ( \alpha _ { 0 } + \alpha _ { 1 } d ) .\tag{29}
$$

We seek the marginalized detection probability:

$$
\bar { P } _ { D } \triangleq \mathbb { E } [ \Phi ( \alpha _ { 0 } + \alpha _ { 1 } d ) ] = \int \Phi ( \alpha _ { 0 } + \alpha _ { 1 } d ) \mathcal { N } ( d ; \mu , \sigma ^ { 2 } ) d d .\tag{30}
$$

We introduce an independent standard normal random variable $u \sim \mathcal { N } ( 0 , 1 )$ with $u \perp d .$ Using the identity $\Phi ( x ) = \mathbb { P } ( u \leq x )$ we can rewrite

$$
\bar { P } _ { D } = \mathbb { E } _ { d } [ \mathbb { P } ( u \leq \alpha _ { 0 } + \alpha _ { 1 } d \bigm \vert d ) ] = \mathbb { P } ( u \leq \alpha _ { 0 } + \alpha _ { 1 } d ) .\tag{31}
$$

Equivalently,

$$
\bar { P } _ { D } = \mathbb { P } ( u - \alpha _ { 1 } d \leq \alpha _ { 0 } ) .\tag{32}
$$

Now define the random variable

$$
w \triangleq u - \alpha _ { 1 } d .\tag{33}
$$

Since u and d are Gaussian and independent, w is also Gaussian with mean and variance

$$
\begin{array} { r } { \mathbb { E } [ w ] = \mathbb { E } [ u ] - \alpha _ { 1 } \mathbb { E } [ d ] = 0 - \alpha _ { 1 } \mu = - \alpha _ { 1 } \mu , } \\ { \mathrm { V a r } ( w ) = \mathrm { V a r } ( u ) + \alpha _ { 1 } ^ { 2 } \mathrm { V a r } ( d ) = 1 + \alpha _ { 1 } ^ { 2 } \sigma ^ { 2 } . \qquad } \end{array}\tag{34}
$$

(35)

Therefore,

$$
w \sim { \mathcal { N } } { \left( - \alpha _ { 1 } \mu , 1 + \alpha _ { 1 } ^ { 2 } \sigma ^ { 2 } \right) } .\tag{36}
$$

Hence,

$$
\begin{array} { r l } & { \bar { P } _ { D } = \mathbb { P } ( w \leq \alpha _ { 0 } ) = \Phi \left( \frac { \alpha _ { 0 } - \mathbb { E } [ w ] } { \sqrt { \mathrm { V a r } ( w ) } } \right) } \\ & { \quad \quad = \Phi \left( \frac { \alpha _ { 0 } - ( - \alpha _ { 1 } \mu ) } { \sqrt { 1 + \alpha _ { 1 } ^ { 2 } \sigma ^ { 2 } } } \right) = \Phi \left( \frac { \alpha _ { 0 } + \alpha _ { 1 } \mu } { \sqrt { 1 + \alpha _ { 1 } ^ { 2 } \sigma ^ { 2 } } } \right) , } \end{array}\tag{37}
$$

(38)

which proves (28).

## REFERENCES

[1] A.-a. Agha-mohammadi, S. Agarwal, S.-K. Kim, S. Chakravorty, and N. M. Amato, “SLAP: Simultaneous Localization and Planning Under Uncertainty via Dynamic Replanning in Belief Space,” IEEE Transactions on Robotics, vol. 34, no. 5, pp. 1195–1214, Oct. 2018. [Online]. Available: https://ieeexplore.ieee.org/document/8479330/

[2] A. A. Meera, M. Popovic, A. Millane, and R. Siegwart, “Obstacle-aware Adaptive Informative Path Planning for UAV-based Target Search,” in 2019 International Conference on Robotics and Automation (ICRA). Montreal, QC, Canada: IEEE, May 2019, pp. 718–724. [Online]. Available: https://ieeexplore.ieee.org/document/8794345/

[3] G. Flaspohler, V. Preston, A. P. M. Michel, Y. Girdhar, and N. Roy, “Information-Guided Robotic Maximum Seek-and-Sample in Partially Observable Continuous Environments,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3782–3789, Oct. 2019. [Online]. Available: https://ieeexplore.ieee.org/document/8767964/

[4] J. A. Placed, J. Strader, H. Carrillo, N. Atanasov, V. Indelman, L. Carlone, and J. A. Castellanos, “A Survey on Active Simultaneous Localization and Mapping: State of the Art and New Frontiers,” IEEE Transactions on Robotics, vol. 39, no. 3, pp. 1686–1705, Jun. 2023. [Online]. Available: https://ieeexplore.ieee.org/document/10075065/

[5] A. Zhitnikov and V. Indelman, “Simplified Continuous High-Dimensional Belief Space Planning With Adaptive Probabilistic Belief-Dependent Constraints,” IEEE Transactions on Robotics, vol. 40, pp. 1684–1705, 2024. [Online]. Available: https://ieeexplore.ieee.org/ document/10354386/

[6] M. J. Kochenderfer, T. A. Wheeler, and K. H. Wray, Algorithms for decision making. Cambridge, Massachusetts: The MIT Press, 2022.

[7] Z. N. Sunberg, C. J. Ho, and M. J. Kochenderfer, “The value of inferring the internal state of traffic participants for autonomous freeway driving,” in 2017 American Control Conference (ACC). Seattle, WA, USA: IEEE, May 2017, pp. 3004–3010. [Online]. Available: https://ieeexplore.ieee.org/document/7963408/

[8] H. Durrant-Whyte, N. Roy, and P. Abbeel, “Unmanned Aircraft Collision Avoidance Using Continuous-State POMDPs,” in Robotics: Science and Systems VII, 2012, pp. 1–8.

[9] K. H. Wray, S. J. Witwicki, and S. Zilberstein, “Online Decision-Making for Scalable Autonomous Systems,” in Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence. Melbourne, Australia: International Joint Conferences on Artificial Intelligence Organization, Aug. 2017, pp. 4768–4774. [Online]. Available: https://www.ijcai.org/proceedings/2017/664

[10] D. Silver and J. Veness, “Monte-Carlo Planning in Large POMDPs,” in Advances in Neural Information Processing Systems, J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, Eds., vol. 23. Curran Associates, Inc., 2010. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/ 2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf

[11] Z. Sunberg and M. Kochenderfer, “Online Algorithms for POMDPs with Continuous State, Action, and Observation Spaces,” Proceedings of the International Conference on Automated Planning and Scheduling, vol. 28, pp. 259–263, Jun. 2018. [Online]. Available: https://ojs.aaai. org/index.php/ICAPS/article/view/13882

[12] M. H. Lim, T. J. Becker, M. J. Kochenderfer, C. J. Tomlin, and Z. N. Sunberg, “Optimality Guarantees for Particle Belief Approximation of POMDPs,” Journal of Artificial Intelligence Research, vol. 77, pp. 1591–1636, Aug. 2023. [Online]. Available: https://jair.org/index.php/jair/article/view/14525

[13] D. J. C. Mackay, “Introduction to Monte Carlo Methods,” in Learning in Graphical Models, M. I. Jordan, Ed. Dordrecht: Springer Netherlands, 1998, pp. 175–204. [Online]. Available: http: //link.springer.com/10.1007/978-94-011-5014-9\_7

[14] C. Snyder, T. Bengtsson, P. Bickel, and J. Anderson, “Obstacles to High-Dimensional Particle Filtering,” Monthly Weather Review, vol. 136, no. 12, pp. 4629–4640, Dec. 2008. [Online]. Available: http://journals.ametsoc.org/doi/10.1175/2008MWR2529.1

[15] S. C. Surace, A. Kutschireiter, and J.-P. Pfister, “How to Avoid the Curse of Dimensionality: Scalability of Particle Filters with and without Importance Weights,” SIAM Review, vol. 61, no. 1, pp. 79–91, Jan. 2019. [Online]. Available: https://epubs.siam.org/doi/10.1137/17M1125340

[16] P. Bickel, B. Li, and T. Bengtsson, “Sharp failure rates for the bootstrap particle filter in high dimensions,” in Institute of Mathematical Statistics Collections. Beachwood, Ohio, USA: Institute of Mathematical Statistics, 2008, pp. 318–329. [Online]. Available: http://projecteuclid.org/euclid.imsc/1209398477

[17] J. S. Liu, Monte Carlo Strategies in Scientific Computing, ser. Springer Series in Statistics Ser. New York, NY: Springer New York, 2008.

[18] M. Montemerlo, S. Thrun, D. Roller, and B. Wegbreit, “FastSLAM 2.0: an improved particle filtering algorithm for simultaneous localization and mapping that provably converges,” in Proceedings of the 18th International Joint Conference on Artificial Intelligence, ser. IJCAI’03. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 2003, pp. 1151–1156.

[19] A. Doucet, S. Godsill, and C. Andrieu, “On Sequential Monte Carlo Sampling Methods for Bayesian Filtering,” in Readings In Unobserved Components Models, A. C. Harvey and T. Proietti, Eds. Oxford University PressOxford, May 2005, pp. 418–441. [Online]. Available: https://academic.oup.com/book/52077/chapter/421060565

[20] K. Murphy and S. Russell, “Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks,” in Sequential Monte Carlo Methods in Practice, A. Doucet, N. Freitas, and N. Gordon, Eds. New York, NY: Springer New York, 2001, pp. 499–515. [Online]. Available: http://link.springer.com/10.1007/978-1-4757-3437-9\_24

[21] M. Montemerlo, S. Thrun, D. Koller, and B. Wegbreit, “FastSLAM: a factored solution to the simultaneous localization and mapping problem,” in Eighteenth National Conference on Artificial Intelligence. USA: American Association for Artificial Intelligence, 2002, pp. 593–598.

[22] S. Särkkä, A. Vehtari, and J. Lampinen, “Rao-Blackwellized particle filter for multiple target tracking,” Information Fusion, vol. 8, no. 1, pp. 2–15, 2007. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S1566253505000874

[23] C. Stachniss, G. Grisetti, and W. Burgard, “Information Gain-based Exploration Using Rao-Blackwellized Particle Filters,” in Robotics: Science and Systems I. Robotics: Science and Systems Foundation, Jun. 2005. [Online]. Available: http://www.roboticsproceedings.org/ rss01/p09.pdf

[24] J. Lee, N. Ahmed, K. H. Wray, and Z. Sunberg, “Rao-Blackwellized POMDP Planning,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). Atlanta, GA, USA: IEEE, May

2025, pp. 14 133–14 139. [Online]. Available: https://ieeexplore.ieee. org/document/11128426/

[25] S. Thrun, W. Burgard, and D. Fox, Probabilistic robotics, ser. Intelligent robotics and autonomous agents. Cambridge, Mass: MIT Press, 2005.

[26] B. Ristic, S. Arulampalam, and N. Gordon, Beyond the Kalman filter: particle filters for tracking applications, ser. Artech House radar library. Boston, MA: Artech House, 2004.

[27] A. Doucet, N. Freitas, and N. Gordon, “An Introduction to Sequential Monte Carlo Methods,” in Sequential Monte Carlo Methods in Practice, A. Doucet, N. Freitas, and N. Gordon, Eds. New York, NY: Springer New York, 2001, pp. 3–14. [Online]. Available: http://link.springer.com/10.1007/978-1-4757-3437-9\_1

[28] C. Andrieu, A. Doucet, and E. Punskaya, “Sequential Monte Carlo Methods for Optimal Filtering,” in Sequential Monte Carlo Methods in Practice, A. Doucet, N. Freitas, and N. Gordon, Eds. New York, NY: Springer New York, 2001, pp. 79–95. [Online]. Available: http://link.springer.com/10.1007/978-1-4757-3437-9\_4

[29] Q. Liu and D. A. Pierce, “A Note on Gauss-Hermite Quadrature,” Biometrika, vol. 81, no. 3, p. 624, Aug. 1994. [Online]. Available: https://www.jstor.org/stable/2337136?origin=crossref

[30] G. Szego,˝ Orthogonal polynomials, 4th ed., ser. Colloquium publications - American Mathematical Society. Providence: American Mathematical Society, 1939, no. v. 23.

[31] W. Schoutens, Stochastic processes and orthogonal polynomials, ser. Lecture notes in statistics. New York: Springer, 2000, no. 146.

[32] V. Barthelmann, E. Novak, and K. Ritter, “High dimensional polynomial interpolation on sparse grids,” Advances in Computational Mathematics, vol. 12, no. 4, pp. 273–288, Mar. 2000. [Online]. Available: https://link.springer.com/10.1023/A:1018977404843

[33] F. Heiss and V. Winschel, “Likelihood approximation by numerical integration on sparse grids,” Journal of Econometrics, vol. 144, no. 1, pp. 62–80, May 2008. [Online]. Available: https://linkinghub.elsevier. com/retrieve/pii/S0304407607002552

[34] P. S. Maybeck, Stochastic models, estimation and control, ser. Mathematics in science and engineering. New York: Academic Press, 1979, no. v. 141.

[35] G. Grisetti, C. Stachniss, and W. Burgard, “Improving Grid-based SLAM with Rao-Blackwellized Particle Filters by Adaptive Proposals and Selective Resampling,” in Proceedings of the 2005 IEEE International Conference on Robotics and Automation. Barcelona, Spain: IEEE, 2005, pp. 2432–2437. [Online]. Available: https: //ieeexplore.ieee.org/document/1570477/

[36] ——, “Improved Techniques for Grid Mapping With Rao-Blackwellized Particle Filters,” IEEE Transactions on Robotics, vol. 23, no. 1, pp. 34– 46, 2007.

[37] V. Zyrianov, X. Zhu, and S. Wang, “Learning to Generate Realistic LiDAR Point Clouds,” in Computer Vision – ECCV 2022, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds. Cham: Springer Nature Switzerland, 2022, vol. 13683, pp. 17–35, series Title: Lecture Notes in Computer Science. [Online]. Available: https://link.springer.com/10.1007/978-3-031-20050-2\_2

[38] V. Zyrianov, H. Che, Z. Liu, and S. Wang, “LidarDM: Generative LiDAR Simulation in a Generated World,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). Atlanta, GA, USA: IEEE, May 2025, pp. 6055–6062. [Online]. Available: https: //ieeexplore.ieee.org/document/11128001/

[39] L. Caccia, H. V. Hoof, A. Courville, and J. Pineau, “Deep Generative Modeling of LiDAR Data,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Macau, China: IEEE, Nov. 2019, pp. 5034–5040. [Online]. Available: https://ieeexplore.ieee.org/document/8968535/

[40] Y. Xiong, W.-C. Ma, J. Wang, and R. Urtasun, “Learning Compact Representations for LiDAR Completion and Generation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Vancouver, BC, Canada: IEEE, Jun. 2023, pp. 1074–1083. [Online]. Available: https://ieeexplore.ieee.org/document/10204732/

[41] S. Deglurkar, M. H. Lim, J. Tucker, Z. Sunberg, A. Faust, and C. J. Tomlin, “Compositional Learning-based Planning for Vision POMDPs,” in Conference on Learning for Dynamics & Control, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:254247358