# Safe Multi-Objective Reinforcement Learning: Zero Constraint Violation and Primal-Averaging Convergence

Chenxi Li ling112358@gmail.com   
Harbin Institute of Technology   
Harbin, Heilongjiang, China   
Yao Shu yaoshu@hkust-gz.edu.cn   
The Hong Kong University of Science and Technology (Guangzhou)   
Guangzhou, Guangdong, China   
Siyang Gao siyangao@cityu.edu.hk   
City University of Hong Kong   
Hong Kong, China   
Shuang Qiu<sup>†</sup> shuanqiu@cityu.edu.hk   
City University of Hong Kong   
Hong Kong, China

Guangxin Jiang<sup>†</sup> Harbin Institute of Technology Harbin, Heilongjiang, China

gxjiang@hit.edu.cn

## Abstract

Many applications of multi-objective reinforcement learning (MORL) require policies that balance conflicting objectives while satisfying safety constraints. We study safe MORL with zero constraint violation throughout learning, covering both cumulative and stepwise constraints. A key challenge is to explore suficiently under unknown dynamics and constraint costs without violating the constraints. For a given preference over objectives, we develop an algorithm to learn a safe Pareto optimal policy by combining optimistic reward estimates with pessimistic cost bounds. With high probability, every policy executed over T episodes incurs zero constraint violation. The algorithm returns a safe Markov policy induced by the averaged occupancy measure and achieves an $\widetilde { \mathcal { O } } ( 1 / \sqrt { T } )$ suboptimality gap under smooth Tchebychef scalarization. We further develop a preference-free algorithm that first explores safely without a prespecified preference and then computes a safe policy for any given preference without further interaction. Over T exploration episodes and K planning iterations, it retains the same safety guarantee and achieves an $\widetilde { \mathcal { O } } ( 1 / \sqrt { T } + 1 / \sqrt { K } )$ suboptimality gap. Both algorithms require knowledge of a policy that strictly satisfies the constraints and its costs. We then relax these requirements, incurring at most constant cumulative violation. To our knowledge, we provide the first sample-eficient MORL algorithms that guarantee safety for every executed policy.

Keywords: Multi-Objective Reinforcement Learning, Safe Reinforcement Learning, Constrained Markov Decision Processes, Smooth Tchebychef Scalarization

## 1 Introduction

Reinforcement learning (RL) studies sequential decision-making through interaction with an environment to maximize cumulative rewards (Sutton and Barto, 1998). In many realworld applications, however, the learner must simultaneously optimize multiple, potentially conflicting objectives such as eficiency, energy consumption, and fairness. This motivates multi-objective reinforcement learning (MORL) (Roijers et al., 2013; Liu et al., 2014), where the learner seeks policies that perform well under diferent trade-ofs among objectives. Since a single policy that maximizes all objectives rarely exists, MORL commonly targets Pareto optimal policies, i.e., policies for which no objective can be improved without degrading another (Van Mofaert and Now´e, 2014; Qiu et al., 2024). For a given preference over the objectives, the goal is to learn a Pareto optimal policy that reflects the desired trade-of. In preference-based learning, the preference is fixed before exploration. Learning under a different preference may require additional environment interaction. Preference-free learning removes this need by exploring the environment once without a prespecified preference and then computing a policy for any given preference from the collected data (Wu et al., 2021).

In safety-critical applications of MORL, the learner must balance multiple objectives while satisfying explicit safety requirements, such as avoiding collisions in autonomous driving, respecting resource and actuation budgets in robotics, or keeping operating temperatures within prescribed limits in industrial control (He and Lv, 2023; Tang et al., 2023). Existing safe MORL methods can be broadly classified into two categories. The first treats safety as an additional objective, or a set of objectives, augmenting the reward vector with costs and learning trade-ofs via scalarization or preference learning (Huang et al., 2022). This strategy does not in general guarantee safety, since whether the learned policy satisfies the safety requirements depends on the chosen preference or scalarization (Achiam et al., 2017). The second category imposes safety as explicit constraints while optimizing multiobjective performance, leading to constrained MORL frameworks (Kim et al., 2025; Park et al., 2026). Under cumulative safety, these constraints limit the expected total cost over an episode, whereas under stepwise safety, they limit the cost at each step. A safe Pareto optimal policy is Pareto optimal within the class of policies satisfying these constraints. The objective value vectors of such policies form the safe Pareto front. However, existing constrained MORL methods may still violate the constraints during learning (Gu et al., 2025). In constrained RL, prior work has established sublinear or constant bounds on cumulative constraint violation (Qiu et al., 2020; Liu et al., 2021; Stradi et al., 2025). Some methods further assume knowledge of a strictly safe policy, which satisfies every constraint with strict inequality. Under this assumption, they ensure that every executed policy satisfies the constraints with high probability, yielding zero constraint violation throughout learning (Liu et al., 2021; Bura et al., 2022; Stradi et al., 2025). However, the literature on constrained MORL still lacks a unified, sample-eficient framework that guarantees zero constraint violation throughout learning under either cumulative or stepwise safety constraints. This leads to the following question:

Can we design provably sample-eficient algorithms for safe MORL that traverse the safe Pareto front while guaranteeing zero constraint violation?

This problem involves three main challenges. First, safety requirements conflict with the need for exploration when costs and dynamics are unknown. The learner may not know whether an exploratory policy is safe before execution, yet every policy executed during learning must satisfy the safety constraints. If the learner executes only policies already known to be safe, the resulting data may be insuficient to identify safe Pareto optimal policies. Second, the set of safe Pareto optimal policies cannot generally be obtained by removing unsafe policies from the unconstrained Pareto optimal set. A policy can be Pareto optimal among safe policies even when it is dominated by an unsafe policy. To obtain a safe Pareto optimal policy for a given preference, we adopt the smooth Tchebychef scalarization (Lin et al., 2024; Qiu et al., 2024), which measures each objective relative to its optimal value over the safe policy class. These optimal values are unknown, and estimating them requires solving an auxiliary constrained RL problem for each objective while maintaining safety throughout learning. Finally, it is even more challenging to explore the environment only once and then compute a safe Pareto optimal policy for any subsequently specified preference. Since no preference is available during exploration, the collected data must sufice for every preference. At the same time, these data should be collected only by safe policies, which restrict the state-action pairs that the learner can visit.

To address the above challenges, we propose CSTRL (Constrained Smooth Tchebychef Reinforcement Learning) for preference-based learning. Its mirror-ascent updates use optimistic reward estimates to encourage exploration, while pessimistic cost bounds are used to maintain safety. We further develop PF-CSTRL (Preference-Free CSTRL), which uses reward and transition estimation uncertainty to guide safe exploration and collect data for planning under any subsequently specified preference. Both algorithms require prior knowledge of a strictly safe policy and its constraint costs. We also develop extensions that relax these requirements.

Contributions. Our main contributions are as follows. (i) We propose CSTRL and PF-CSTRL for preference-based and preference-free safe MORL, respectively. To our knowledge, these are the first sample-eficient MORL algorithms that guarantee zero constraint violation for every policy executed during learning. Both algorithms combine optimistic mirror-ascent updates of occupancy measures with pessimistic cost bounds under either cumulative or stepwise safety constraints. (ii) We establish that, over T learning episodes, CSTRL guarantees zero constraint violation with high probability under either safety formulation for every executed policy. It returns a safe Markov policy induced by the primal average of the occupancy measures of these policies and achieves a suboptimality gap of $\widetilde { \mathcal { O } } ( 1 / \sqrt { T } )$ under smooth Tchebychef scalarization. Using T exploration episodes followed by K planning iterations, PF-CSTRL maintains the same safety guarantee throughout both stages and achieves a suboptimality gap of $\widetilde { \mathcal { O } } ( 1 / \sqrt { T } + 1 / \sqrt { K } )$ . (iii) We extend both algorithms under weaker assumptions about prior knowledge. When a strictly safe policy is known but its costs are unknown, we estimate its costs conservatively while preserving zero constraint violation. When both the policy and its costs are unknown, we learn a strictly safe policy and a conservative estimate of its costs in an estimation phase and incur only constant cumulative violation.

Related Work. MORL studies sequential decision-making problems with multiple, potentially conflicting objectives and seeks policies that represent diferent trade-ofs among them (Roijers et al., 2013; Liu et al., 2014; Hayes et al., 2022). A substantial line of work characterizes the Pareto front using scalarization-based formulations, including linear and Tchebychef scalarizations, and develops methods for learning Pareto optimal policy sets (Roijers et al., 2013; Van Mofaert and Now´e, 2014; Hayes et al., 2022; Kubo et al., 2026). More recent work studies preference-based methods that learn policies for given preferences and preference-free methods that compute policies for any preference specified after exploration (Yang et al., 2019; Qiu et al., 2024; Hu and Yu, 2026). However, most existing MORL approaches are developed for unconstrained environments and therefore do not provide explicit safety guarantees (Hayes et al., 2022; Qiu et al., 2024).

Constrained reinforcement learning studies sequential decision-making under explicit safety or resource constraints, most commonly formalized as constrained Markov decision processes (Altman, 2021). Classical approaches enforce expected cumulative constraints over an episode via Lagrangian or primal-dual methods, providing asymptotic or averaged feasibility guarantees but allowing violations during learning (Achiam et al., 2017; Tessler et al., 2018; Wei et al., 2020). To obtain stricter guarantees, recent works adopt conservative or pessimistic designs that achieve zero or bounded cumulative constraint violation, often by tightening constraints or assuming access to a strictly safe policy (Liu et al., 2021; Vaswani et al., 2022; Bai et al., 2023; Ni and Kamgarpour, 2026). Other works incorporate optimism in the face of uncertainty to enable eficient exploration under safety constraints (Qiu et al., 2020; Liu et al., 2021; Ding and Lavaei, 2023; Wendl et al., 2026), and further investigate stepwise safety (Xiong et al., 2023; Zhao et al., 2024). However, existing methods are largely limited to single-objective settings and do not address the challenges of multi-objective or preference-free learning.

## 2 Preliminaries

We consider a finite-horizon constrained multi-objective Markov decision process. It is defined by the tuple $( \mathcal { S } , \mathcal { A } , H , P , r , c , \rho )$ , where S and A are the finite state and action spaces. The horizon H is the number of steps in each episode. The transition kernel is ${ P } : = \{ P _ { h } \} _ { h = 1 } ^ { H }$ with $P _ { h } : \mathcal { S } \times \mathcal { A } \times \mathcal { S } \mapsto [ 0 , 1 ]$ , and $P _ { h } ( s ^ { \prime } | s , a )$ denotes the probability of transitioning to state $s ^ { \prime } \in \mathcal { S }$ from state $s \in { \mathcal { S } }$ after taking action $a \in { \mathcal { A } }$ at step $h \in [ H ]$ The reward function is vector-valued with m objectives, denoted by $r : = ( r _ { 1 } , \ldots , r _ { m } )$ , where $r _ { i } : = \{ r _ { i , h } \} _ { h = 1 } ^ { H }$ for every $i \in [ m ]$ . Each $r _ { i , h } : S \times A \mapsto [ 0 , 1 ]$ maps a state-action pair to the reward for the i-th objective at step $h .$ . Moreover, the constraint cost function is vectorvalued with n constraints, denoted by $\pmb { c } : = ( c _ { 1 } , \ldots , c _ { n } )$ . We further define $c _ { j } : = \{ c _ { j , h } \} _ { h = 1 } ^ { H } ;$ where each $c _ { j , h } : S \times A \mapsto [ 0 , 1 ]$ maps a state-action pair to the cost for the j-th constraint at step h. The initial state distribution is denoted by $\rho .$ The learner follows a policy $\pi : = \{ \pi _ { h } \} _ { h = 1 } ^ { H } $ , where $\pi _ { h } ( a | s )$ represents the probability of taking action $a \in { \mathcal { A } }$ in state $s \in { \mathcal { S } }$ at step $h \in [ H ]$ . Let Π denote the class of all stochastic Markov policies. For any $\pi \in \Pi$ , we define the value of $\pi$ for the i-th objective as $\begin{array} { r } { V _ { i } ^ { r } ( \pi ) : = \mathbb { E } [ \sum _ { h = 1 } ^ { H } r _ { i , h } ( s _ { h } , a _ { h } ) | \rho , \pi , P ] } \end{array}$ and the value of $\pi$ for the j-th constraint as $\begin{array} { r } { V _ { j } ^ { c } ( \pi ) : = \mathbb { E } [ \sum _ { h = 1 } ^ { H } c _ { j , h } ( s _ { h } , a _ { h } ) | \rho , \pi , P ] } \end{array}$

In each episode $t \in [ T ]$ , the learner follows the policy $\pi ^ { t }$ and generates a trajectory $\{ ( s _ { h } ^ { t } , a _ { h } ^ { t } ) \} _ { h = 1 } ^ { H }$ . At each step $h \in [ H ]$ , the learner observes a stochastic reward $r _ { i , h } ^ { t } \in [ 0 , 1 ]$ for each objective $i \in [ m ]$ and a stochastic constraint cost $c _ { j , h } ^ { t } \in [ 0 , 1 ]$ for each constraint $j \in [ n ]$ These observations satisfy $\mathbb { E } [ r _ { i , h } ^ { t } | s _ { h } ^ { t } , a _ { h } ^ { t } ] = r _ { i , h } ( s _ { h } ^ { t } , a _ { h } ^ { t } )$ and $\mathbb { E } [ c _ { j , h } ^ { t } | s _ { h } ^ { t } , a _ { h } ^ { t } ] = c _ { j , h } ( s _ { h } ^ { t } , a _ { h } ^ { t } )$ . The reward function r, the constraint cost function c, and the transition kernel $P$ are unknown.

Under such a scenario, we aim to solve the following constrained optimization problem:

$$
\begin{array} { r l } { \underset { \pi \in \Pi } { \operatorname* { m a x } } } & { V _ { 1 } ^ { r } ( \pi ) , ~ V _ { 2 } ^ { r } ( \pi ) , \cdot \cdot \cdot , ~ V _ { m } ^ { r } ( \pi ) } \\ & { \mathrm { s . t . } ~ V _ { j } ^ { c } ( \pi ) \leq \tau _ { j } , ~ \forall j \in [ n ] , } \end{array}
$$

where $\tau _ { j }$ is the threshold for constraint $j ,$ , and $\tau : = ( \tau _ { 1 } , \dots , \tau _ { n } )$ collects these thresholds.

Occupancy Measure. In this work, we employ the occupancy measure, i.e., the joint distribution over the state, action, and next state, to formulate the optimization problem. Formally, an occupancy measure is defined as follows (Rosenberg and Mansour, 2019):

Definition 1 Letting $\Delta _ { | S | \times | A | \times | S | }$ denote the probability simplex over $s \times \mathcal { A } \times \mathcal { S }$ , we say that $\boldsymbol { \theta } : = \{ \boldsymbol { \theta } _ { h } \} _ { h = 1 } ^ { H }$ is an occupancy measure for an episodic MDP with transition kernel P and initial state distribution ρ if $\theta _ { h } \in \Delta _ { | S | \times | A | \times | S | }$ and

$$
\left\{ \begin{array} { l l } { \displaystyle \sum _ { a \in A } \displaystyle \sum _ { s ^ { \prime } \in S } \theta _ { 1 } ( s , a , s ^ { \prime } ) = \rho ( s ) , } & { \forall s \in \mathcal S , } \\ { \displaystyle \sum _ { a \in A } \displaystyle \sum _ { s ^ { \prime } \in S } \theta _ { h } ( s , a , s ^ { \prime } ) = \displaystyle \sum _ { s ^ { \prime } \in S } \displaystyle \sum _ { a \in A } \theta _ { h - 1 } ( s ^ { \prime } , a , s ) , } & { \forall h \in [ 2 , H ] , \ \forall s \in \mathcal S , } \\ { \theta _ { h } ( s , a , s ^ { \prime } ) = P _ { h } ( s ^ { \prime } | s , a ) \displaystyle \sum _ { s ^ { \prime \prime } \in S } \theta _ { h } ( s , a , s ^ { \prime \prime } ) , } & { \forall h \in [ H ] , \ \forall ( s , a , s ^ { \prime } ) \in \mathcal S \times A \times \mathcal S . } \end{array} \right.
$$

With a slight abuse of notation, we define $\begin{array} { r } { \theta _ { h } ( s , a ) : = \sum _ { s ^ { \prime } \in S } \theta _ { h } ( s , a , s ^ { \prime } ) } \end{array}$ . The policy induced by θ is given by $\textstyle \pi _ { h } ^ { \theta } ( a | s ) = \theta _ { h } ( s , a ) / \sum _ { a \in \mathcal { A } } \theta _ { h } ( s , a )$ . For a specific transition kernel P and policy π, we denote the corresponding occupancy measure by $\theta ^ { P , \pi } = \{ \theta _ { h } ^ { P , \pi } \} _ { h = 1 } ^ { H }$ . Then, we have $\theta _ { h } ^ { P , \pi } ( s , a , s ^ { \prime } ) : = \mathrm { P r } ( s _ { h } = s , a _ { h } = a , s _ { h + 1 } = s ^ { \prime } | \rho , \pi , P )$ and $\theta _ { h } ^ { P , \pi } ( s , a ) : = \operatorname* { P r } ( s _ { h } = s , a _ { h } =$ $a | \rho , \pi , P )$ . Moreover, by the definition of the occupancy measure, we can reconstruct both the transition kernel and the policy as $P _ { h } ( s ^ { \prime } | s , a ) \bar { } = \theta _ { h } ^ { \bar { P } , \pi } ( s , a , s ^ { \prime } ) / \theta _ { h } ^ { P , \pi } ( s , a )$ and $\pi _ { h } ( a | s ) =$ $\textstyle \theta _ { h } ^ { P , \pi } ( s , a ) / \sum _ { a \in \mathcal { A } } \theta _ { h } ^ { P , \pi } ( s , a )$

For any occupancy measure θ, overloading the notation V, the corresponding values for rewards and constraints are written as

$$
V _ { i } ^ { r } ( \theta ) : = \langle r _ { i } , \theta \rangle = \sum _ { h , s , a } r _ { i , h } ( s , a ) \theta _ { h } ( s , a ) , \quad V _ { j } ^ { c } ( \theta ) : = \langle c _ { j } , \theta \rangle = \sum _ { h , s , a } c _ { j , h } ( s , a ) \theta _ { h } ( s , a ) .
$$

Then, we have $V _ { i } ^ { r } ( \theta ^ { P , \pi } ) = V _ { i } ^ { r } ( \pi )$ and $V _ { j } ^ { c } ( \theta ^ { P , \pi } ) = V _ { j } ^ { c } ( \pi )$

Safety Constraints and Safe Policies. To formulate the safety constraints, we collect the constraint cost functions defined above into a matrix $C \in [ 0 , 1 ] ^ { H | S | | A | \times n }$ , whose j-th column is $( c _ { j , h } ( s , a ) ) _ { h \in [ H ] , ( s , a ) \in S \times \mathcal { A } } .$ We use $d \in \{ 1 , H \}$ to denote the number of time steps to which each safety constraint applies. For each constraint $j \in [ n ]$ , the learner knows the d time steps to which the constraint applies, and we set $c _ { j , h } \equiv 0$ at all other steps $h \in [ H ]$ When $d = 1$ , each constraint applies to a single time step, which gives stepwise safety. When $d = H$ , each constraint applies to all H steps, which gives cumulative safety. Accordingly, we have $\tau \in [ 0 , d ] ^ { n }$ . In both cases, a policy $\pi \in \Pi$ is safe if $C ^ { \top } \theta ^ { P , \pi } \leq \tau$ . The corresponding safe policy class and occupancy measure set are $\Pi _ { \mathrm { s a f e } } : = \{ \pi \in \Pi : C ^ { \top } \theta ^ { P , \pi } \leq \tau \}$ and $\Theta _ { \mathrm { s a f e } } : = \{ \theta ^ { P , \pi } : \pi \in \Pi _ { \mathrm { s a f e } } \}$ We adopt the following standard Slater condition, which requires that the constraints can be satisfied strictly.

Assumption 2 (Slater condition) There exists a strictly safe policy $\pi ^ { \dagger } \in \Pi$ such that ${ \cal C } ^ { \top } \theta ^ { P , \pi ^ { \dagger } } < \tau$

Although Assumption 2 ensures that such a policy exists, this fact alone does not guarantee safety during learning when the policy is unknown to the learner.

Assumption 3 The policy $\pi ^ { \dagger }$ in Assumption 2 is known to the learner. Moreover, the learner knows its constraint costs $\gamma : = \bar { C } ^ { \top } \theta ^ { P , \pi ^ { \dagger } }$ , which satisfy $0 \leq \gamma _ { j } < \tau _ { j } \leq d$ for all $j \in [ n ]$

Assumption 3 is standard in the constrained RL literature when zero constraint violation throughout learning is required (Liu et al., 2021; Stradi et al., 2025). In practice, a strictly safe policy is often available (Huang et al., 2023). For example, in autonomous driving, a conservative rule-based controller that prioritizes collision avoidance and enforces strict speed limits provides a natural candidate for $\pi ^ { \dagger }$ (Likmeta et al., 2020).

Pareto Optimality. In unconstrained MORL, the standard goal is to identify the Pareto optimal policy set, that is, the collection of policies for which no objective can be improved without degrading another (Qiu et al., 2024). When explicit safety constraints are imposed, policies that are Pareto optimal in the unconstrained setting may violate them. We therefore consider Pareto optimality relative to the safe policy class.

Definition 4 (Safe Pareto optimal policy) For any two policies $\pi \in \Pi _ { \mathrm { s a f e } }$ and $\pi ^ { \prime } \in$ $\Pi _ { \mathrm { s a f e } } ,$ , we say that $\pi ^ { \prime }$ dominates π if and only if $V _ { i } ^ { r } ( \pi ^ { \prime } ) \ge V _ { i } ^ { r } ( \pi )$ for all $i \in [ m ]$ and there exists at least one $i ^ { \prime } \in [ m ]$ such that $V _ { i ^ { \prime } } ^ { r } ( \pi ^ { \prime } ) > V _ { i ^ { \prime } } ^ { r } ( \pi )$ . A policy $\pi \in \Pi _ { \mathrm { s a f e } }$ is a safe Pareto optimal policy if and only if no other policy $\pi ^ { \prime } \in \Pi _ { \mathrm { s a f e } }$ dominates π. We denote the set of safe Pareto optimal policies by $\Pi _ { \mathrm { s a f e } } ^ { * }$

## 3 Scalarization of Multiple Objectives

To characterize the trade-ofs achieved by safe Pareto optimal policies, we seek a scalarization framework that covers the entire safe Pareto front and allows the learner to select a safe Pareto optimal policy according to a given preference. Since this front may contain infinitely many value vectors, direct enumeration is generally infeasible. We therefore consider a family of scalar optimization problems parameterized by a preference vector $\lambda \in \Delta _ { m }$ . Here, $\Delta _ { m }$ is the probability simplex in $\mathbb { R } ^ { m }$ , and $\Delta _ { m } ^ { o } : = \{ \lambda \in \Delta _ { m } : \lambda _ { i } > 0 , \forall i \in [ m ] \}$ denotes its relative interior. In unconstrained MORL, the smooth Tchebychef scalarization (Lin et al., 2024; Qiu et al., 2024) has been shown to provide both controllability with respect to the preference vector λ and $f u l l$ coverage of the Pareto front. In our safe MORL setting, however, the policy space is restricted to the safe class $\Pi _ { \mathrm { s a f e } ; }$ , and we adapt the scalarization to this class as follows. For notational simplicity, let $V _ { i } ( \cdot ) : = V _ { i } ^ { r } ( \cdot )$ and $V ( \theta ) : = ( V _ { 1 } ( \theta ) , \dots , V _ { m } ( \theta ) )$

Definition 5 (Safe smooth Tchebychef scalarization) Let $\lambda : = ( \lambda _ { 1 } , \ldots , \lambda _ { m } ) \in \Delta _ { m }$ be a preference vector, let $\mu > 0$ be a smoothing parameter, and let $\iota \in ( 0 , 1 ]$ be a prespecified regularization parameter. For each $i \in [ m ] .$ , we define $V _ { i } ^ { * } : = \operatorname* { m a x } _ { \theta \in \Theta _ { \mathrm { s a f e } } } V _ { i } ( \theta )$ . The safe smooth Tchebychef scalarization of a policy $\pi \in \Pi _ { \mathrm { s a f e } }$ is

$$
\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ) : = \mu \log \left[ \sum _ { i = 1 } ^ { m } \exp \left( \frac { \lambda _ { i } \bigl ( V _ { i } ^ { * } + \iota - V _ { i } ( \theta ^ { P , \pi } ) \bigr ) } { \mu } \right) \right] .
$$

We first establish that the safe value set is a nonempty convex polytope.

Proposition 6 Under Assumption ${ \mathcal { Q } } ,$ the safe value set $\mathcal { V } _ { \mathrm { s a f e } } : = \{ V ( \theta ) : \theta \in \Theta _ { \mathrm { s a f e } } \}$ is a nonempty convex polytope.

The safe smooth Tchebychef scalarization retains both controllability and full coverage on the safe policy class.

Proposition 7 Let $\mu > 0$ . Under Assumption ${ \mathcal { Q } } ,$ we have

$$
\bigcup _ { \lambda \in \Delta _ { m } ^ { o } } \arg \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ) = \Pi _ { \mathrm { s a f e } } ^ { * } .
$$

The proof of Proposition 7 uses the fact that $\nu _ { \mathrm { s a f e } }$ is a nonempty convex polytope, as established in Proposition 6. This convexity relies on Π containing all stochastic Markov policies. The analysis for deterministic policy classes difers substantially, and we leave this extension to future work. We next present a proposition that further characterizes the safe Pareto optimal policy set.

Proposition 8 Let $\mu > 0$ and $\lambda \in \Delta _ { m } ^ { o }$ . Under Assumption ${ \mathcal { Q } } ,$ all minimizers of $\mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi )$ over $\Pi _ { \mathrm { s a f e } }$ achieve the same values on all objectives.

Together with Proposition $^ { 7 , }$ this result implies that solving min $\cdot \pi { \in } \Pi _ { \mathrm { s a f e } } \operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi )$ once for each $\lambda \in \Delta _ { m } ^ { o }$ sufices to recover every value vector on the safe Pareto front. For the safe policy class, the corresponding scalarized optimization problem is

$$
\operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ) .\tag{1}
$$

Following Qiu et al. (2024), the safe smooth Tchebychef scalarization admits the equivalent reformulation

$$
\mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ) = \operatorname* { m a x } _ { w \in \Delta _ { m } } \operatorname* { m a x } _ { \{ \theta _ { i } \in \Theta _ { \mathrm { s a f e } } \} _ { i = 1 } ^ { m } } \sum _ { i = 1 } ^ { m } \left[ w _ { i } \lambda _ { i } \big ( V _ { i } ( \theta _ { i } ) + \iota - V _ { i } ( \theta ^ { P , \pi } ) \big ) - \mu w _ { i } \log w _ { i } \right] .\tag{2}
$$

Consequently, problem (1) admits an equivalent min-max-max formulation. This reformulation is well suited to the design of algorithms based on upper confidence bounds.

Performance Metric. For a given preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0$ , we define an optimal safe policy as $\pi _ { \mu , \lambda } ^ { * } \in \arg \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi )$ . The performance of a safe policy $\pi \in \Pi _ { \mathrm { s a f e } }$ is then measured by the suboptimality gap

$$
\mathrm { S u b O p t } ( \pi ) : = \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ) - \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi _ { \mu , \lambda } ^ { * } ) .
$$

Note that $\mathrm { S u b O p t } ( \pi ) \geq 0$ for all $\pi \in \Pi _ { \mathrm { s a f e } }$ , with equality if and only if π is optimal under the smooth Tchebychef scalarization.

We also assess the safety of any policy $\pi \in \Pi$ by its constraint violation

$$
\mathrm { V i o l a t i o n } ( \pi ) : = \operatorname* { m a x } _ { j \in [ n ] } \Big [ \big ( C ^ { \top } \theta ^ { P , \pi } \big ) _ { j } - \tau _ { j } \Big ] ^ { + } ,
$$

where $[ x ] ^ { + } : = \operatorname* { m a x } \{ 0 , x \}$ . For stepwise safety, $\left( C ^ { \top } \theta ^ { P , \pi } \right) _ { ? }$ gives the cost of constraint $j$ at the single step to which it applies. For cumulative safety, it aggregates the costs of constraint $j$ across all H steps. In both cases, $\pi \in \Pi _ { \mathrm { s a f e } }$ if and only if Violation $( \pi ) = 0 .$

Our goal is to design a sample-eficient algorithm that outputs a policy $\widehat { \pi }$ minimizing $\operatorname { S u b O p t } ( { \widehat { \pi } } )$ while guaranteeing Violation $( \widehat { \pi } ) = 0$ and $\mathrm { V i o l a t i o n } ( \pi ) = 0$ for every policy π executed during learning.

## 4 Safe Multi-Objective RL

We introduce CSTRL, our preference-based algorithm to solve (1) under a fixed preference vector $\lambda \in \Delta _ { m }$ . CSTRL combines optimistic exploration with pessimistic cost bounds to ensure safety with high probability. The full procedure is summarized in Algorithm 1. In episode t, Algorithm 1 collects one trajectory under the main policy $\pi ^ { t }$ and one under each auxiliary policy $\nu _ { i } ^ { t }$ for $i \in [ m ]$ , yielding $m + 1$ trajectories. The main trajectory is used to update the main empirical estimates, while the trajectory generated by $\nu _ { i } ^ { t }$ is used only to update the i-th auxiliary empirical estimates. The counts are

$$
N _ { h } ^ { t } ( s , a ) : = \sum _ { k = 1 } ^ { t } \mathbf { 1 } _ { \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} } , \quad N _ { h } ^ { t } ( s , a , s ^ { \prime } ) : = \sum _ { k = 1 } ^ { t } \mathbf { 1 } _ { \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } , s _ { h + 1 } ^ { k } ) = ( s , a , s ^ { \prime } ) \} } .\tag{3}
$$

Here, $\mathbf { 1 } _ { \{ \varepsilon \} }$ is the indicator of event $\mathcal { E } .$ For each objective $i \in [ m ]$ and each constraint $j \in [ n ]$ 2 the reward and cost estimators are

$$
\begin{array} { r } { \widehat { r } _ { i , h } ^ { t } ( s , a ) : = \frac { \displaystyle \sum _ { k = 1 } ^ { t } r _ { i , h } ^ { k } \mathbf { 1 } _ { \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} } } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } , \quad \widehat { c } _ { j , h } ^ { t } ( s , a ) : = \frac { \displaystyle \sum _ { k = 1 } ^ { t } c _ { j , h } ^ { k } \mathbf { 1 } _ { \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} } } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } . } \end{array}\tag{4}
$$

The empirical transition model is

$$
\widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) : = \frac { N _ { h } ^ { t } ( s , a , s ^ { \prime } ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } .\tag{5}
$$

To quantify estimation uncertainty, we fix $\delta \in ( 0 , 1 )$ and construct confidence sets for the rewards, costs, and transition kernel. For each reward function $r _ { i , h }$ , we define a bonus

$$
\sigma _ { h } ^ { t } ( s , a ) : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { 2 \ln ( 8 m \vert S \vert \vert A \vert H T / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \right\} .\tag{6}
$$

By Lemma 26, $| \widehat { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) | \leq \sigma _ { h } ^ { t } ( s , a )$ holds with high probability. For the cost functions, we define the bonus for every $j \in [ n ]$ , every time step h to which constraint $j$ applies, and every $( s , a ) \in \mathcal S \times \mathcal A$ as

$$
\xi _ { j , h } ^ { t } ( s , a ) : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { \ln ( 4 m n d T | S | | A | / \delta ) } { 2 \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \right\} .\tag{7}
$$

At all other steps, we set $\widehat { c } _ { j , h } ^ { t } ( s , a ) = \xi _ { j , h } ^ { t } ( s , a ) : = 0$ for every $( s , a ) \in \mathcal { S } \times \mathcal { A }$ . For each $j \in [ n ]$ and $h \in [ H ]$ , we collect the cost estimates and bonuses at step h into the vectors $\widehat { c } _ { j , h } ^ { t } : = ( \widehat { c } _ { j , h } ^ { t } ( s , a ) ) _ { ( s , a ) \in S \times \mathcal { A } }$ and $\xi _ { j , h } ^ { t } : = ( \xi _ { j , h } ^ { t } ( s , a ) ) _ { ( s , a ) \in S \times \mathcal { A } }$ , respectively. We then define $\widehat { C } ^ { t } , \Xi ^ { t } \in [ 0 , 1 ] ^ { H | S | | A | \times n }$ as the matrices whose j-th columns are $( \widehat { c } _ { j , h } ^ { t } ) _ { h \in [ H ] }$ and $( \xi _ { j , h } ^ { t } ) _ { h \in [ H ] }$ ， respectively. Lemma 25 ensures that $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t }$ holds with high probability. For the transition kernel, we define a confidence set $\mathcal { P } ^ { t }$ by imposing concentration bounds $\left| P _ { h } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \right| \le \epsilon _ { h } ^ { t } ( s , a )$ , where

$$
\epsilon _ { h } ^ { t } ( s , a ) : = \sqrt { \frac { 2 \ln ( 2 m T H | S | ^ { 2 } | A | / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } .\tag{8}
$$

By Lemma 23, with probability at least $1 - \delta .$ , the true kernel lies in $\mathcal { P } ^ { t }$ for all $t \in [ T ]$ . For the occupancy measure updates, we define the optimistic reward estimates by

$$
\bar { r } _ { i , h } ^ { t } ( s , a ) : = \operatorname* { m i n } \big \{ 1 , \hat { r } _ { i , h } ^ { t } ( s , a ) + \sigma _ { h } ^ { t } ( s , a ) \big \} , \quad \forall i \in [ m ] .\tag{9}
$$

Given the preference vector $\boldsymbol { \lambda }$ and the smooth Tchebychef dual variable $w ^ { t }$ , we define the scalarized optimistic reward

$$
\bar { R } _ { h } ^ { t } ( s , a ) : = \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \bar { r } _ { i , h } ^ { t } ( s , a ) .\tag{10}
$$

CSTRL learns directly on occupancy measures and maintains a dual variable $\boldsymbol { w } ^ { t } \in \Delta _ { m }$ for the smooth Tchebychef scalarization. At episode $t \in [ T - 1 ]$ , it applies exponential updates with step size $\eta > 0$ to the main and auxiliary occupancy measures using $\bar { R } ^ { t }$ and $\bar { r } _ { i } ^ { t } .$ , respectively, yielding $\widehat { \theta } ^ { t + 1 }$ and $\widehat { \theta } _ { i } ^ { t + 1 }$ for each $i \in [ m ]$ These updated occupancy measures may violate the safety constraints. We therefore project them onto an optimistic feasible occupancy measure set. We define the occupancy measure set associated with $\mathcal { P }$ by $\Theta ( \mathcal { P } ) \ : = \ \{ \theta ^ { P ^ { \prime } , \pi } \ : \ P ^ { \prime } \in \mathcal { P } , \ \pi \in \Pi \}$ . For a single transition kernel $P ,$ we define $\Theta ( P ) : = \Theta ( \{ P \} )$ . The projection minimizes the KL divergence to the exponentially updated iterate over $\Theta ( \mathcal { P } ^ { t } )$ , subject to an optimistic version of the safety constraints:

$$
\mathrm { P r o j } \left( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } \right) : = \left\{ \begin{array} { l l } { \displaystyle \mathrm { a r g } \operatorname* { m i n } _ { \theta \in \Theta \left( \mathcal { P } ^ { t } \right) } D _ { \mathrm { K L } } ( \theta \| \widehat { \theta } ^ { t + 1 } ) } \\ { \mathrm { s . t . } \quad \left( \widehat { C } ^ { t } - \Xi ^ { t } \right) ^ { \top } \theta \leq \tau , } \end{array} \right.\tag{11}
$$

where $D _ { \mathrm { K L } } ( \cdot | | \cdot )$ is the KL divergence. By Lemma $2 4 , \Theta ( \mathcal { P } ^ { t } )$ is a convex polytope. Since the safety constraints in (11) are linear in θ and $D _ { \mathrm { K L } } ( \theta \| \widehat \theta ^ { t + 1 } )$ is convex in $\theta ,$ problem (11) is a convex program. We denote the projected occupancy measures for the main and auxiliary updates in episode $t + 1$ by $\widetilde { \theta } ^ { t + 1 }$ and $\{ \widetilde { \theta } _ { i } ^ { t + 1 } \} _ { i = 1 } ^ { m }$ , respectively. We define their induced policies as $\widetilde { \pi } ^ { t + 1 } : = \pi ^ { \widetilde { \theta } ^ { t + 1 } }$ and $\widetilde \nu _ { i } ^ { t + 1 } : = \pi ^ { \widetilde { \theta } _ { i } ^ { t + 1 } }$ for each $i \in [ m ]$ . The projection and the choice of the executed policy are implemented by Subroutine 1 (SafeProj). When the projection in (11) is feasible, we use pessimistic cost bounds for $\widetilde { \pi } ^ { t + 1 }$ to define the probability of selecting $\pi ^ { \dagger }$ as

$$
\psi ^ { t } : = \operatorname* { m a x } _ { j \in [ n ] } \frac { \left[ \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \right\} - \tau _ { j } \right] ^ { + } } { \operatorname* { m a x } \left\{ \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \right\} - \gamma _ { j } , \ \tau _ { j } - \gamma _ { j } \right\} } .\tag{12}
$$

Algorithm 1 CSTRL: Constrained Smooth Tchebychef RL   
Require: $\{ \alpha _ { t } \} _ { t = 1 } ^ { T - 1 } , \{ \beta _ { t } \} _ { t = 1 } ^ { T - 1 } , \lambda , \mu$   
1: Initialize $\begin{array} { r } { \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) = \frac { 1 } { | S | ^ { 2 } | A | } ; } \end{array}$ set $\begin{array} { r } { \widetilde { \pi } ^ { 1 } = \operatorname { U n i f } ( \boldsymbol { A } ) , \psi ^ { 0 } : = \operatorname* { m a x } _ { j \in [ n ] } \{ \frac { d - \tau _ { j } } { d - \gamma _ { j } } \} } \end{array}$ , and $w _ { i } ^ { 1 } = 1 / m , \ \widetilde { \theta } _ { i } ^ { 1 } = \widetilde { \theta } ^ { 1 }$   
for all $i \in [ m ] ;$ select $\pi ^ { 1 }$ as $\dot { \pi } ^ { \dagger }$ w.p. $\psi ^ { 0 }$ and as $\widetilde { \pi } ^ { 1 } \ \mathrm { w . p . } \ 1 - \psi ^ { 0 } ;$ for each $i \in [ m ]$ , select $\nu _ { i } ^ { 1 } ~ \mathrm { a s } ~ \pi ^ { \dagger }$   
w.p. $\psi ^ { 0 }$ and as $\widetilde { \pi } ^ { 1 } \ \mathrm { w . p . } \ 1 - \psi ^ { 0 } ;$ roll out $\pi ^ { 1 }$ and $\{ \nu _ { i } ^ { 1 } \} _ { i = 1 } ^ { m }$   
2: for $t = 1 , 2 , \dots , T - 1$ do   
3: for $i = 1 , \ldots , m$ do   
4: Construct $( \bar { r } _ { i } ^ { t } , \widehat { C } ^ { t } , \mathcal { P } ^ { t } , \Xi ^ { t } )$ from the trajectories generated by $\{ \nu _ { i } ^ { k } \} _ { k = 1 } ^ { t }$ using (3)–(9)   
5: Set $\widehat { \theta } _ { i , h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { i , h } ^ { t } ( s , a , s ^ { \prime } ) \exp \{ \eta \bar { r } _ { i , h } ^ { t } ( s , a ) \}$ for all $h \in [ H ]$   
6: $( \widetilde { \theta } _ { i } ^ { t + 1 } , \nu _ { i } ^ { t + 1 } , \psi _ { i } ^ { t } ) \gets \mathtt { S a f e P r o j } ( \widehat { \theta } _ { i } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$   
7: Compute $\widetilde { V } _ { i } ^ { t + 1 } = \langle \bar { r } _ { i } ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \rangle$   
8: end for   
9: Construct $( \{ \bar { r } _ { i } ^ { t } \} _ { i = 1 } ^ { m } , \widehat { C } ^ { t } , \mathcal { P } ^ { t } , \Xi ^ { t } ) .$ from the trajectories generated by $\{ \pi ^ { k } \} _ { k = 1 } ^ { t }$ using $( 3 )  { - } ( 9 )$   
10: Compute $\bar { R } ^ { t }$ by (10) and set $\widehat { \theta } _ { h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { t } ( s , a , s ^ { \prime } )$ exp{η $\bar { R } _ { h } ^ { t } ( s , a ) \}$ for all $h \in [ H ]$   
11: $( \widetilde { \theta } ^ { t + 1 } , \pi ^ { t + 1 } , \psi ^ { t } ) \gets \mathtt { S a f e P r o j } ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$   
12: Compute $V _ { i } ^ { t + 1 } = \langle \bar { r } _ { i } ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle$ for all $i \in [ m ]$   
13: $\widetilde { w } _ { i } ^ { t } = \widetilde { ( 1 - \alpha _ { t } ) } w _ { i } ^ { t } + \alpha _ { t } / m , \ w _ { i } ^ { t + 1 } \propto ( \widetilde { w } _ { i } ^ { t } ) ^ { \widetilde { 1 } - \mu \beta _ { t } } \exp \{ \beta _ { t } \lambda _ { i } ( \widetilde { V } _ { i } ^ { t + 1 } + \iota - V _ { i } ^ { t + 1 } ) \}$ for all $i \in [ m ]$   
14: Roll out $\pi ^ { t + 1 }$ and $\{ \boldsymbol \nu _ { i } ^ { t + 1 } \} _ { i \in [ m ] }$   
15: end for   
16: Return: $\begin{array} { r } { \bar { \theta } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \theta ^ { P , \pi ^ { t } } } \end{array}$

Subroutine 1 SafeProj: Safe Projection   
Require: $\widehat { \theta } , \mathcal { P } , \widehat { C } , \Xi$   
1: if Pro $\mathfrak { j } ( \widehat { \theta } , \widehat { \mathcal { P } } , \widehat { C } , \Xi )$ is feasible then   
2: Set $\widetilde { \theta } = \mathtt { P r o j } ( \widehat { \theta } , \mathcal { P } , \widehat { C } , \Xi )$ and ${ \widetilde { \pi } } = \pi ^ { \widetilde { \theta } }$   
3: Compute ψ by (12)   
4: else   
5: Set $\widetilde { \theta } = \mathrm { P r o j } ( \widehat { \theta } , { \mathcal { P } } ) , \widetilde { \pi } = \pi ^ { \widetilde { \theta } } ;$ , and $\psi = 1$   
6: end if   
7: Select $\pi \ \mathrm { a s } \ \underset { \sim } { \pi ^ { \dagger } } \ \mathrm { w . p . } \ \psi$ and as $\widetilde { \pi } _ { \mathrm { ~ } \mathrm { ~ w . p . ~ } 1 - \psi }$   
8: Return: $( { \widetilde { \theta } } , \pi , \psi )$

When $d = 1$ , each constraint applies to a single step, while $\widehat { c } _ { j , h } ^ { t } = \xi _ { j , h } ^ { t } = 0$ at all other steps. The sum in (12) therefore reduces to a single term, and the truncated quantity is bounded by 1. When $d = H$ , each constraint applies to every step. In this case, the sum aggregates the contributions from all H steps, and the truncated quantity is bounded by H. If the problem (11) is infeasible, it reduces to $\begin{array} { r l } & { \mathrm { P r o j } ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } ) : = \arg \operatorname* { m i n } _ { \theta \in \Theta ( \mathcal { P } ^ { t } ) } D _ { \mathrm { K L } } ( \theta \| \widehat { \theta } ^ { t + 1 } ) } \end{array}$ . In this case, we set $\psi ^ { t } = 1$ and execute $\pi ^ { \dagger }$ . For each $i \in [ m ]$ , the probability $\psi _ { i } ^ { t }$ in the auxiliary update is defined analogously, with $\widetilde { \pi } ^ { t + 1 }$ replaced by $\widetilde \nu _ { i } ^ { t + 1 }$ . To state SafeProj in a generic form, we suppress iteration and objective indices. Given the inputs $\widehat { \theta } , \mathcal { P } , \widehat { C } .$ , and $\Xi , \mathsf { S a f e P r o j }$ constructs the projected occupancy measure $\widetilde { \theta }$ and its induced policy $\widetilde { \pi } = \pi ^ { \widetilde { \theta } }$ It then selects $\pi ^ { \dagger }$ with probability $\psi$ and $\widetilde { \pi }$ with probability $1 - \psi$ at the beginning of the episode, and follows the selected policy throughout the episode. Throughout the algorithms, $^ { 6 6 } \mathrm { W . p . } ^ { , \mathrm { p . } }$ stands for “with probability”. The subroutine returns $( { \widetilde { \theta } } , \pi , \psi )$ , where $\pi$ is the policy executed throughout the episode. After the occupancy measure updates, CSTRL updates the smooth Tchebychef dual variable. It first mixes $w ^ { t }$ with the uniform vector on $\Delta _ { m } .$ , which keeps the logarithmic terms bounded as in Wei et al. (2020) and Qiu et al. (2023). It then applies an entropic mirror-ascent update to obtain $w ^ { t + 1 }$ . This update uses the optimistic single-objective value $\widetilde { V } _ { i } ^ { t + 1 } = \langle \bar { r } _ { i } ^ { t } , \tilde { { \theta } } _ { i } ^ { t + 1 } \rangle$ from the i-th auxiliary update and the optimistic value $V _ { i } ^ { t + 1 } = \langle \bar { r } _ { i } ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle$ from the main update, where the two $\bar { r } _ { i } ^ { t }$ come from the auxiliary and main trajectories, respectively. The returned main policy $\scriptstyle { \dot { \pi } } ^ { t + 1 }$ and auxiliary policies $\{ \nu _ { i } ^ { t + 1 } \} _ { i = 1 } ^ { m }$ are then rolled out in episode $t + 1$ . Algorithm 1 returns $\begin{array} { r } { \bar { \theta } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \bar { \theta ^ { P , \pi ^ { t } } } } \end{array}$ , the primal average of the occupancy measures of the executed policies.

Theoretical Results. The following theorem quantifies the suboptimality of CSTRL.

Theorem 9 (Suboptimality gap of CSTRL) Suppose Assumptions 2 and 3 hold. For any preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0$ , set $\eta = T ^ { - 1 / 2 } , \alpha _ { t } = 1 / t ^ { 2 }$ and $\beta _ { t } = 1 / ( \mu t )$ for $t \in [ T - 1 ]$ . Then, with probability at least $1 - \delta$ , the suboptimality gap of CSTRL satisfies

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \Bigg ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } + \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } \Bigg ) ,
$$

where ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ is the Markov policy induced by the returned occupancy measure $\bar { \theta } , \phi _ { \mathrm { s a f e } } : =$ $\textstyle \operatorname* { m a x } _ { j \in [ n ] } \Bigl \{ \frac { d ^ { 2 } } { ( \tau _ { j } - \gamma _ { j } ) ^ { 2 } } \Bigr \}$ , and $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in $m , n , | { \cal S } | , | { \cal A } | , H , T$ , and $1 / \delta$

Theorem 9 guarantees that CSTRL achieves a suboptimality gap of order $\widetilde { \mathcal { O } } ( T ^ { - 1 / 2 } )$ with respect to the smooth Tchebychef scalarization objective. Compared with existing work, our result ofers several improvements. First, Qiu et al. (2024) study unconstrained MORL and output a policy mixture that randomly selects from $\{ \pi ^ { 1 } , \ldots , \pi ^ { T } \}$ , which does not yield a single Markov policy. In contrast, CSTRL constructs a single Markov policy $\widehat { \pi }$ from the primal average $\bar { \theta }$ of the occupancy measures, and the suboptimality guarantee applies to this policy. Second, Xiong et al. (2023) address stepwise safety constraints but are restricted to the single-objective setting. Our framework extends this line of work to the multi-objective setting, covers both cumulative and stepwise safety, and provably converges to the corresponding safe Pareto optimal policy set. Third, Gu et al. (2025) consider constrained MORL but only guarantee an $\mathcal { O } ( 1 / \sqrt { T } )$ expected constraint violation bound. In contrast, Theorem 10 shows that CSTRL achieves zero constraint violation with high probability throughout learning, providing a strictly stronger safety guarantee.

Theorem 10 (Safety of CSTRL) Suppose Assumptions 2 and 3 hold. Let $\{ \pi ^ { t } \} _ { t = 1 } ^ { T }$ and $\{ \nu _ { i } ^ { t } \} _ { t \in [ T ] , i \in [ m ] }$ be the policies executed by Algorithm 1, and let πb be the Markov policy induced by the returned occupancy measure <sup>¯</sup>θ. Then, with probability at least $1 - \delta$ , every executed policy is safe, i.e., Violation $( \pi ^ { t } ) = 0$ and Violation $( \nu _ { i } ^ { t } ) = 0$ for all $t \in [ T ]$ and $i \in [ m ]$ , and the output policy satisfies Violation $( \widehat { \pi } ) = 0$

Theorem 10 establishes stepwise safety when $d = 1$ and cumulative safety when $d = H$

## 5 Preference-Free Safe Multi-Objective RL

We note that CSTRL is preference-based, since its preference vector λ is fixed before exploration and a new exploration process is required whenever λ changes. To avoid repeated environment interaction across preferences while maintaining safety under either cumulative or stepwise constraints, we propose PF-CSTRL. The algorithm consists of a safe exploration stage followed by a preference-dependent planning stage. Its exploration stage collects a dataset using safe policies without a prespecified preference. The planning stage then computes a safe policy for any $\lambda \in \Delta _ { m }$ from this dataset without further environment interaction. The exploration stage collects one trajectory per episode, and the counts and estimators in $( 3 ) \AA { - } ( 5 )$ are computed from these trajectories. Compared with CSTRL, the main change is in Line 5 of Algorithm 2, where the preference-weighted reward is replaced by the exploration reward $\overline { { r } } _ { h } ^ { t } ( s , a )$ . This reward combines the reward bonus $\sigma _ { h } ^ { t } ( s , a )$ and the transition bonus $\zeta _ { h } ^ { t } ( s , a )$ to encourage broad state-action coverage for subsequent planning. We define the transition bonus

$$
\zeta _ { h } ^ { t } ( s , a ) : = \operatorname* { m i n } \biggr \{ H , H \sqrt { \frac { 8 | \mathcal { S } | \ln \bigl ( 2 m T H | \mathcal { S } | | A | / \delta \bigr ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \biggr \} .\tag{13}
$$

After the exploration stage, we omit the episode superscript and use $( \widehat { r } , \widehat { C } , \mathcal { P } , \Xi , \sigma , \zeta )$ to denote the corresponding quantities evaluated at the final counts $N ^ { T }$ . The fixed optimistic rewards $\{ \bar { r } _ { i } \} _ { i = 1 } ^ { m }$ are then given by (9). At planning iteration $k \in [ K ]$ , the scalarized optimistic reward $\bar { R } ^ { k }$ is given by (10) with the dual variable $\boldsymbol { w } ^ { k }$ . Both stages use SafeProj and exponential updates with $\eta > 0$ . We denote the policies induced by the projected occupancy measures in exploration episode t + 1 and planning iteration k by $\widetilde { \pi } ^ { t + 1 } : = \pi ^ { \widetilde { \theta } ^ { t + 1 } }$ and $\bar { \pi } ^ { k + 1 } : = \pi ^ { \widetilde { \theta } ^ { k + 1 } }$ , respectively.

Theoretical Results. The following theorem bounds the suboptimality of PF-CSTRL.

Theorem 11 (Suboptimality gap of PF-CSTRL) Suppose Assumptions 2 and 3 hold and PF-CSTRL is run for T exploration episodes and K planning iterations. Let πb denote the policy induced by the averaged occupancy measure output at the end of the planning stage. For any preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0 _ { : }$ , set $\alpha _ { k } = 1 / k ^ { 2 } , \beta _ { k } =$ $1 / ( \mu k ) , \eta = T ^ { - 1 / 2 }$ for exploration, and $\eta = K ^ { - 1 \bar { / } 2 }$ for planning. Then, with probability at least $1 - \delta _ { i }$ , we have

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { \frac { | A | } { T } } + \frac { H } { \sqrt { K } } + \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \right) ,
$$

where $\begin{array} { r } { \phi _ { \mathrm { s a f e } } = \operatorname* { m a x } _ { j \in [ n ] } \left\{ \frac { d ^ { 2 } } { ( \tau _ { j } - \gamma _ { j } ) ^ { 2 } } \right\} } \end{array}$ , and $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in $m , n , | { \cal S } | , | { \cal A } | , { \cal H }$ 9 T, K, and $1 / \delta$

The bound in Theorem 11 separates the statistical error of preference-free exploration from the optimization error of planning on the fixed dataset. The T-dependent term arises from the fixed reward and transition bonuses and the probabilities of selecting $\pi ^ { \dagger }$ during planning. The K-dependent terms arise from the main occupancy measure updates, the optimistic single-objective values ${ \widetilde { V } } _ { i } .$ , and the smooth Tchebychef dual updates. PF-CSTRL guarantees safety under either formulation throughout both stages.

Algorithm 2 Exploration Stage for PF-CSTRL Algorithm 3 Planning Stage for PF-CSTRL   
1: Initialize $\begin{array} { r } { \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) = \frac { 1 } { | S | ^ { 2 } | A | } ; } \end{array}$ set $\widetilde { \pi } ^ { 1 } = \mathrm { U n i f } ( \mathcal { A } )$ <sub>and</sub> Require: D from Algorithm $2 , \lambda , \{ \alpha _ { k } \} _ { k = 1 } ^ { K } , \{ \beta _ { k } \} _ { k = 1 } ^ { K } , \mu$   
1: Initialize $w _ { i } ^ { 1 } = 1 / m$ and $\begin{array} { r } { \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) = \frac { 1 } { | S | ^ { 2 } | A | } } \end{array}$   
$\textstyle \psi ^ { 0 } : = \operatorname* { m a x } _ { j \in [ n ] } \{ { \frac { d - \tau _ { j } } { d - \gamma _ { j } } } \}$ ; select $\pi ^ { 1 }$ as $\pi ^ { \dagger }$ w.p. $\psi ^ { 0 }$ and as   
2: Construct the fixed $\hat { C } , \mathcal { P } , \Xi ,$ and ${ \bar { r } } _ { i }$ from D by (4)–(9)   
$\widetilde { \pi } ^ { 1 }$ w.p. $1 - \psi ^ { 0 } ;$ roll out $\pi ^ { 1 }$ , initialize D, and construct 3: for $i = 1 , \ldots , m$ do   
$N ^ { 1 }$ by (3) 4: $\widehat { \theta } _ { i , h } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) \exp \{ \eta ^ { - 1 } \bar { r } _ { i , h } ( s , a ) \}$   
5: if Pro $\mathrm { j } ( \widehat { \theta } _ { i } , \mathcal { P } , \widehat { C } , \Xi )$ is feasible then   
2: for $t = 1 , 2 , \dots , T - 1$ do 6: Set $\widetilde { \theta } _ { i } = \mathtt { P r o j } ( \widehat { \theta } _ { i } , \mathcal { P } , \widehat { C } , \Xi )$   
3: Construct $( \widehat { C } ^ { t } , \mathcal { P } ^ { t } , \Xi ^ { t } )$ by (4), (5), (7), and (8) 7: else   
8: Set $\widetilde { \theta } _ { i } = \mathrm { P r o j } ( \widehat { \theta } _ { i } , \mathcal { P } )$   
4: Compute $\sigma _ { h } ^ { t }$ and $\zeta _ { h } ^ { t }$ by (6) and (13) 9: end if   
5: Uncertainty-guided exploration reward: 10: Set $\widetilde { V } _ { i } = \langle \bar { r } _ { i } , \widetilde { \theta } _ { i } \rangle$   
11: end for   
$\overline { { r } } _ { h } ^ { t } ( \cdot , \cdot ) = \operatorname* { m a x } \{ \zeta _ { h } ^ { t } ( \cdot , \cdot ) / H , \sigma _ { h } ^ { t } ( \cdot , \cdot ) \}$ 12: for $k = 1 , 2 , \ldots , K$ do   
6: $\widehat { \theta } _ { h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { t } ( s , a , s ^ { \prime } ) \exp \{ \eta \overline { { r } } _ { h } ^ { t } ( s , a ) \}$ 13: $\widehat { \theta } _ { h } ^ { k + 1 } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { k } ( s , a , s ^ { \prime } ) \exp \{ \eta \bar { R } _ { h } ^ { k } ( s , a ) \}$   
14: $( \widetilde { \theta } ^ { k + 1 } , \pi ^ { k + 1 } , \psi ^ { k } ) \gets \mathtt { S a f e P r o j } ( \widehat { \theta } ^ { k + 1 } , \mathcal { P } , \widehat { C } , \Xi )$   
7: $( \widetilde { \theta } ^ { t + 1 } , \pi ^ { t + 1 } , \psi ^ { t } ) \gets \mathtt { S a f e P r o j } ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$ 15: Set $V _ { i } ^ { k + 1 } = \langle \bar { r } _ { i } , \widetilde { \theta } ^ { k + 1 } \rangle , \forall i \in [ m ]$   
8: Roll out $\pi ^ { t + 1 }$ , update D, and compute $N ^ { t + 1 }$ by (3) 16: $\widetilde { w } _ { i } ^ { k } = ( 1 - \alpha _ { k } ) w _ { i } ^ { k } + \alpha _ { k } / m , \forall i \in [ m ]$   
17: $w _ { i } ^ { \dot { k } + 1 } \propto ( \widetilde { w } _ { i } ^ { k } ) ^ { 1 - \mu \dot { \beta } _ { k } } \exp \{ \beta _ { k } \lambda _ { i } \big ( \widetilde { V } _ { i } + \iota - V _ { i } ^ { k + 1 } \big ) \} , \forall i \in [ m ]$   
9: end for 18: end for   
10: Return: D 19: Return: $\begin{array} { r } { \bar { \theta } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \theta ^ { P , \pi ^ { k + 1 } } } \end{array}$

Theorem 12 (Safety of PF-CSTRL) Suppose Assumptions 2 and 3 hold. Let $\{ \pi ^ { t } \} _ { t = 1 } ^ { T }$ and $\{ \pi ^ { k + 1 } \} _ { k = 1 } ^ { K }$ denote the policies generated in the exploration and planning stages, respectively, by Algorithms $\mathcal { Q }$ and 3. Then, with probability at least $1 - \delta ,$ we have Violation $( \pi ^ { t } ) = 0 , \forall t \in$ [T], and Violation $( \pi ^ { k + 1 } ) = 0 , \forall k \in [ K ]$ . Consequently, the returned policy $\widehat { \pi }$ induced by $\bar { \theta }$ satisfies Violation(πb) = 0.

## 6 Extensions

Both CSTRL and PF-CSTRL assume that the learner knows a strictly safe policy $\pi ^ { \dagger }$ and its costs $\gamma$ . We relax these requirements in this section.

## 6.1 Known $\pi ^ { \dagger }$ and Unknown γ

We first remove the requirement that the learner knows $\gamma .$

Assumption 13 The policy $\pi ^ { \dagger }$ in Assumption 2 is known to the learner, but its costs $\gamma$ are unknown.

Since $\pi ^ { \dagger }$ is known, the learner can estimate γ by repeatedly executing $\pi ^ { \dagger }$ . We define the Slater margin of $\pi ^ { \dagger }$ as $\begin{array} { r } { \varrho ^ { \dagger } : = \operatorname* { m i n } _ { j \in [ n ] } ( \tau _ { j } - \gamma _ { j } ) } \end{array}$ . The empirical costs of $\pi ^ { \dagger }$ after t episodes are $\begin{array} { r } { \bar { \gamma } _ { j } ^ { t } : = \frac { 1 } { t } \sum _ { k = 1 } ^ { t } \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { k } } \end{array}$ . The stopping time $t _ { s }$ is the first episode $t \in [ T ]$ with $\tau _ { j } - \bar { \gamma } _ { j } ^ { t } \geq$ $3 d \sqrt { \ln ( 4 n T / \delta ) / ( 2 t ) }$ for all $j \in [ n ]$ , or $T$ if no such episode exists. The learner executes $\pi ^ { \dagger }$ for $t _ { s }$ episodes, which we call the estimation phase, and then constructs a conservative upper confidence bound $\widehat { \gamma }$ on $\gamma$ with $\widehat { \gamma } _ { j } : = \bar { \gamma } _ { j } ^ { t _ { s } } + d \sqrt { \ln ( 4 n T / \delta ) / ( 2 t _ { s } ) }$ . Our analysis shows that, with high probability, $t _ { s } \le 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 } = \widetilde { \mathcal { O } } ( d ^ { 2 } / ( \varrho ^ { \dagger } ) ^ { 2 } )$ . The resulting preferencebased extension is CSTRL-UC (CSTRL with Unknown Constraint Costs). It runs CSTRL for the remaining $T - t _ { s }$ episodes with $\widehat { \gamma }$ in place of $\gamma$ . To compute a safe policy for any subsequently specified preference, we also combine this estimation phase with PF-CSTRL. The resulting preference-free extension is PF-CSTRL-UC (PF-CSTRL with Unknown Constraint Costs). It runs PF-CSTRL with $\widehat { \gamma }$ in place of $\gamma$ for the remaining $T - t _ { s }$ exploration episodes, followed by K planning iterations.

Theoretical Results. We now present the guarantees for CSTRL-UC and PF-CSTRL-UC.

Theorem 14 (Suboptimality gap of CSTRL-UC) Suppose Assumptions $\mathcal { Q }$ and 13 hold. $L e t \widehat { \pi }$ be the policy returned by CSTRL-UC. Assume that $T > 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . For any preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0 _ { ; }$ , set $\eta ~ = ~ ( T - t _ { s } ) ^ { - 1 / 2 }$ $\alpha _ { t } = 1 / t ^ { 2 }$ , and $\beta _ { t } = 1 / ( \mu t )$ . Then, with probability at least $1 - \delta$ , we have $t _ { s } < T$ and

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \widehat { \phi } _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \frac { H ^ { 2 } } { \mu ( T - t _ { s } ) } + \frac { \mu } { T - t _ { s } } \right) ,
$$

where $\begin{array} { r } { \widehat { \phi } _ { \mathrm { s a f e } } : = \operatorname* { m a x } _ { j \in [ n ] } \frac { d ^ { 2 } } { ( \tau _ { j } - \widehat { \gamma } _ { j } ) ^ { 2 } } } \end{array}$ satisfies $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$ , and $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in $m , n , | { \cal S } | , | { \cal A } | , H , T$ , and $1 / \delta$

Theorem 15 (Safety of CSTRL-UC) Suppose Assumptions $\mathcal { Q }$ and 13 hold. Assume that $T > 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . Let πb be the policy returned by CSTRL-UC. With probability at least $1 - \delta _ { i }$ , the stopping time satisfies $t _ { s } < T$ , and every policy executed is safe. The returned policy satisfies $\mathrm { V i o l a t i o n } ( \widehat { \pi } ) = 0$

Theorem 16 (Suboptimality gap of PF-CSTRL-UC) Under Assumptions 2 and 13, let $\widehat { \pi }$ be the policy returned by PF-CSTRL-UC. Assume that $T > 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . For any preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0 ,$ , set $\alpha _ { k } = 1 / k ^ { 2 } , \beta _ { k } = 1 / ( \mu k )$ $\eta = ( T - t _ { s } ) ^ { - 1 / 2 }$ for exploration, and $\eta = K ^ { - 1 / 2 }$ for planning. Then, with probability at least $1 - \delta _ { i }$ , we have $t _ { s } < T$ and

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( \widehat { \phi } _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \frac { H } { \sqrt { K } } + \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \right) ,
$$

where $\widehat { \phi } _ { \mathrm { s a f e } }$ is defined in Theorem $1 \mathit { 4 }$ and satisfies $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$ . The notation $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in m, n, |S|, |A|, H, T, K, and $1 / \delta$

Theorem 17 (Safety of PF-CSTRL-UC) Suppose Assumptions $\mathcal { Q }$ and 13 hold. Assume that $T > 1 + 8 \dot { d } ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . With probability at least $1 - \delta ,$ , the stopping time satisfies $t _ { s } < T$ . On the same event, every policy executed during the estimation phase or the exploration stage is safe, as is every main policy generated during the planning stage. The returned policy $\widehat { \pi }$ satisfies Violation(πb) = 0.

## 6.2 Unknown $\pi ^ { \dagger }$ and Unknown γ

Both CSTRL-UC and PF-CSTRL-UC still require the learner to know $\pi ^ { \dagger }$ . We now remove this requirement and assume only Assumption 2, under which a strictly safe policy exists but the learner knows neither this policy nor its costs. CSTRL-US (CSTRL with an Unknown Safe Policy) first estimates a strictly safe policy and its costs and then runs CSTRL. Algorithm 4 in Appendix I gives the full procedure. We define the Slater parameter as $\begin{array} { r } { \varrho : = \operatorname* { m a x } _ { \pi \in \Pi } \operatorname* { m i n } _ { j \in [ n ] } \{ \tau _ { j } - ( C ^ { \top } \theta ^ { P , \pi } ) _ { j } \} } \end{array}$ . For the estimation phase, we define ${ \tt A l g } ^ { P }$ and $\mathsf { A l g } ^ { D }$ as the primal and dual regret minimizers, respectively.

• Primal algorithm $\mathsf { A l g } ^ { P }$ . For each episode t in the estimation phase, let $\phi ^ { t } \in \Delta _ { n }$ denote the probability distribution over the n constraints, where $\Delta _ { n }$ is the probability simplex in $\mathbb { R } ^ { n }$ . For each $\textit { h } \in \ [ H ]$ , we define the observed loss as $\ell _ { h } ^ { t } : =$ $\begin{array} { r } { \sum _ { j = 1 } ^ { n } \phi _ { j } ^ { t } c _ { j , h } ^ { t } - \frac { 1 } { H } \sum _ { j = 1 } ^ { n } \phi _ { j } ^ { t } \tau _ { j } } \end{array}$ . For any policy $\pi \in \Pi$ , we define its expected cumulative loss as $\begin{array} { r } { \bar { \ell } ^ { t } ( \pi ) : = \sum _ { j = 1 } ^ { \bar { n } } { \phi _ { j } ^ { t } [ ( C ^ { \top } \theta ^ { P , \pi } ) _ { j } - \tau _ { j } ] } } \end{array}$ . At the beginning of episode t, $\mathsf { A l g } ^ { P }$ takes $\{ ( s _ { h } ^ { k } , a _ { h } ^ { k } , \ell _ { h } ^ { k } ) \} _ { k < t , h \in [ H ] }$ as input and outputs a policy $\pi ^ { t } \in \Pi$ . After executing $\pi ^ { t }$ , the learner executes $\dot { \mathsf { A } } \dot { \mathsf { I } } \dot { \mathsf { g } } ^ { P }$ using $\{ ( s _ { h } ^ { t } , a _ { h } ^ { t } , \ell _ { h } ^ { t } ) \} _ { h = 1 } ^ { H } . \ \mathrm { A 1 g } ^ { P }$ can be implemented using the algorithm of Jin et al. (2020) together with a standard doubling trick (Lattimore and Szepesv´ari, 2020). We assume that, with probability at least $1 - \delta .$ , its regret satisfies $\begin{array} { r } { \sum _ { k = 1 } ^ { t } [ \bar { \ell } ^ { k } ( \pi ^ { k } ) - \bar { \ell } ^ { k } ( \pi ) ] \leq \mathcal { O } \big ( H ^ { 2 } | S | \sqrt { | A | t \ln ( 2 H | S | | A | t / \delta ) } \big ) } \end{array}$ simultaneously for every $t \geq 1$ and every fixed $\pi \in \Pi$

• Dual algorithm $\mathsf { A l g } ^ { D }$ . For each episode t in the estimation phase, we define $\varrho _ { j } ^ { t } : = \quad$ $\tau _ { j } - \textstyle \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { t }$ for every $j \in [ n ]$ and set $\varrho ^ { t } : = ( \varrho _ { 1 } ^ { t } , \ldots , \varrho _ { n } ^ { t } ) \in \mathbb { R } ^ { n }$ . At the beginning of episode t, $\mathsf { A l g } ^ { D }$ takes $\{ \varrho ^ { k } \} _ { k = 1 } ^ { t - 1 }$ as input and outputs the distribution $\phi ^ { t } \in \Delta _ { n }$ After observing the constraint costs in episode $t ,$ the learner executes $\mathsf { A l g } ^ { D }$ using $\varrho ^ { t } . \ \mathsf { A l g } ^ { D }$ can be implemented using online gradient descent (Orabona, 2019), whose regret satisfies $\begin{array} { r } { \sum _ { k = 1 } ^ { t } \langle { \phi } ^ { k } - { \phi } , { \varrho } ^ { k } \rangle \leq \mathcal { O } ( d \sqrt { n t } ) } \end{array}$ for every $t \geq 1$ and every fixed $\phi \in \Delta _ { n }$

We define the stopping time of the estimation phase as $\begin{array} { r } { t _ { s } : = \operatorname* { m i n } \{ t \in [ T ] : \operatorname* { m i n } _ { j \in [ n ] } \sum _ { k = 1 } ^ { t } \varrho _ { j } ^ { k } \geq } \end{array}$ $\mathcal { O } ( ( H ^ { 2 } | \mathcal { S } | \sqrt { | \boldsymbol { A } | } + d \sqrt { n } ) \sqrt { t \ln ( 2 n H | \mathcal { S } | | \boldsymbol { A } | t / \delta ) } ) \}$ , where the right-hand side of the inequality is the stopping threshold. At the end of episode $t _ { s } ,$ the algorithm constructs the policy $\widehat { \pi } ^ { \dagger }$ , which selects $\pi ^ { t }$ with probability $1 / t _ { s }$ for each $t \in [ t _ { s } ]$ . It then computes an estimate $\widehat { \varrho }$ of $\varrho$ and sets ${ \widehat { \gamma } } : = ( { \widehat { \gamma } } _ { 1 } , \ldots , { \widehat { \gamma } } _ { n } )$ , where $\widehat { \gamma } _ { j } : = \tau _ { j } - \widehat { \varrho }$ for every $j \in [ n ]$ The algorithm runs CSTRL for the remaining $T - t _ { s }$ episodes with $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$ in place of $\pi ^ { \dagger }$ and $\gamma .$ . To avoid repeated environment interaction across preferences, we also extend PF-CSTRL to this setting using the same estimation phase. The resulting extension is PF-CSTRL-US (PF-CSTRL with an Unknown Safe Policy). It runs PF-CSTRL with $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$ in place of $\pi ^ { \dagger }$ and $\gamma$ for the remaining $T - t _ { s }$ exploration episodes, followed by K planning iterations. For both CSTRL-US and PF-CSTRL-US, we assume that $T$ is suficiently large, and it sufices to take $T = \widetilde { \mathcal { O } } ( ( H ^ { 4 } | S | ^ { 2 } | A | + n d ^ { 2 } ) / \varrho ^ { 2 } )$ , where $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in n, $H , | S | , | A | , 1 / \delta$ 2 and $1 / \varrho$ . Under this condition, Lemma 43 gives $t _ { s } < T$ with high probability.

Theoretical Results. We now present the guarantees for CSTRL-US and PF-CSTRL-US.

Theorem 18 (Suboptimality gap of CSTRL-US) Suppose Assumption 2 holds. Let $\widehat { \pi }$ be the policy returned $b y$ Algorithm $\it 4 .$ For any preference vector $\lambda \in \Delta _ { m }$ and smoothing

(I)

parameter $\mu > 0$ , set $\eta = ( T - t _ { s } ) ^ { - 1 / 2 } , \alpha _ { t } = 1 / t ^ { 2 }$ , and $\beta _ { t } = 1 / ( \mu t )$ . Then, with probability at least $1 - \delta$ , we have $T > t _ { s }$ and

$$
\mathrm { S u b O p t } ( \widehat \pi ) \leq \widetilde O \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \widehat \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \frac { H ^ { 2 } } { \mu ( T - t _ { s } ) } + \frac { \mu } { T - t _ { s } } \right) ,
$$

where $\begin{array} { r } { \widehat { \phi } _ { \mathrm { s a f e } } : = \operatorname* { m a x } _ { j \in [ n ] } \frac { d ^ { 2 } } { ( \tau _ { j } - \widehat { \gamma } _ { j } ) ^ { 2 } } } \end{array}$ satisfies $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 d ^ { 2 } / \varrho ^ { 2 }$ , and $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in $m , n , | { \cal S } | , | { \cal A } | , H , T$ , and $1 / \delta$

Theorem 19 (Constant violation of CSTRL-US) Under Assumption 2, let $\{ \pi ^ { t } \} _ { t = 1 } ^ { T }$ be the main policies executed by Algorithm 4. Then, with probability at least $1 - \delta$ , we have $T > t _ { s }$ and $\Sigma _ { t = 1 } ^ { T }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$ . On the same event, every policy executed after the estimation phase is safe, and the output policy $\widehat { \pi }$ satisfies Violation $( \widehat { \pi } ) = 0$

Theorem 20 (Suboptimality gap of PF-CSTRL-US) Suppose Assumption 2 holds. Let πb be the policy returned by PF-CSTRL-US. For any preference vector $\lambda \in \Delta _ { m }$ and smoothing parameter $\mu > 0$ , set $\alpha _ { k } = 1 / k ^ { 2 } , \beta _ { k } = 1 / ( \mu k ) , \eta = ( T - t _ { s } ) ^ { - 1 / 2 }$ for exploration, and $\eta = K ^ { - 1 / 2 }$ for planning. Then, with probability at least $1 - \delta$ , we have $T > t _ { s }$ and

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( \widehat { \phi } _ { \mathrm { s a f e } } H ^ { 4 } | \mathcal { S } | ^ { 2 } \sqrt { \frac { | A | } { T - t _ { s } } } + \frac { H } { \sqrt { K } } + \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \right) ,
$$

where $\widehat { \phi } _ { \mathrm { s a f e } }$ is defined in Theorem 18 and satisfies $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 d ^ { 2 } / \varrho ^ { 2 }$ . The notation $\widetilde { \mathcal { O } } ( \cdot )$ hides logarithmic factors in $m , n , | S | , | A | , H , T , K$ , and $1 / \delta$

Theorem 21 (Constant violation of PF-CSTRL-US) Suppose Assumption 2 holds. Let $\{ \pi ^ { t } \} _ { t = 1 } ^ { T }$ be the policies executed during the estimation phase and the subsequent exploration stage of PF-CSTRL-US. Then, with probability at least $1 - \delta$ , we have $T > t _ { s }$ and $\Sigma _ { t = 1 } ^ { T }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$ . On the same event, every policy executed during preference-free exploration and every main policy generated during planning is safe, and the output policy πb satisfies $\mathrm { V i o l a t i o n } ( \widehat { \pi } ) = 0$

## 7 Theoretical Analysis

This section outlines the main arguments used to establish our theoretical results.

## 7.1 Proof Sketch of Theorem 9

The full proof of Theorem 9 is provided in Appendix D. For each objective $i \in [ m ]$ , let $\theta _ { i } ^ { * } \in \arg \operatorname* { m a x } _ { \theta \in \Theta _ { \mathrm { s a f e } } } V _ { i } ( \theta )$ be a safe single-objective optimal occupancy measure. Let $\theta _ { \lambda } ^ { * } \in$ $\Theta _ { \mathrm { s a f e } }$ be an optimal safe occupancy measure for the smooth Tchebychef objective under the preference λ. Since $\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta } )$ is convex in θ and changes by at most H between any two occupancy measures, the contribution of $\pi ^ { 1 }$ to SubOpt(πb) is at most $H / T$ . Combining this separation with the min-max reformulation of the smooth Tchebychef scalarization yields

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \frac { H } { T } + \operatorname* { m a x } _ { w \in \Delta _ { m } } \sum _ { i = 1 } ^ { m } w _ { i } \lambda _ { i } \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Bigl ( { V _ { i } ^ { * } } - { V _ { i } } ( \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ) \Bigr )
$$

$$
\begin{array} { r l } & { + \underbrace { \underset { w \in \Delta _ { m } } { \operatorname* { m a x } } \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \Big [ ( w _ { i } - w _ { i } ^ { t } ) \lambda _ { i } \left( V _ { i } ( \theta ^ { P , \tilde { v } _ { i } ^ { t + 1 } } ) + \iota - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \right) - \mu w _ { i } \log w _ { i } + \mu w _ { i } ^ { t } \log w _ { i } ^ { t } \Big ] } _ { \mathrm { ( I I ) } } } \\ & { + \underbrace { \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \left( V _ { i } ( \theta _ { \lambda } ^ { * } ) - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \right) } _ { \mathrm { ( I I I ) } } + \underbrace { \frac { H } { T } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \displaystyle \operatorname* { m a x } _ { w \in \Delta _ { m } } \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } \lambda _ { i } \psi _ { i } ^ { t } } _ { \mathrm { ( I V ) } } . } \end{array}
$$

Terms (I) and (III) are the auxiliary and main occupancy measure learning errors. Term (II) contains the dual and estimation errors, while Term (IV) arises from the probabilities of selecting $\pi ^ { \dagger }$ in the auxiliary updates.

Bounding Term (I). On the high-probability event, $\theta _ { i } ^ { * }$ is feasible for the auxiliary projection. Lemma 30 therefore gives $\begin{array} { r } { \frac { 1 } { T } { \sum _ { t = 1 } ^ { T - 1 } } ( V _ { i } ^ { * } - V _ { i } ( { \theta } ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ) ) \leq \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) } \end{array}$ $+ \textstyle { \frac { 4 H } { T } } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t }$ . The diference between the projected occupancy measure $\widetilde { \theta } _ { i } ^ { t + 1 } \in \Theta ( \mathcal { P } ^ { t } )$ and the occupancy measure of $\widetilde \nu _ { i } ^ { t + 1 }$ under P is controlled by Lemma 31. The remaining error is controlled by the probabilities $\psi _ { i } ^ { t }$ . For every $i \in [ m ]$ , we have $\textstyle \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } \leq$ $\tilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | T } )$ . Since $\pmb { w } , \pmb { \lambda } \in \Delta _ { m }$ , we obtain $( \mathrm { I } ) \le \widetilde { \mathcal { O } } ( ( 1 + \phi _ { \mathrm { s a f e } } H ) H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$

Bounding Term (II). We control Term (II) by comparing the true values in the smooth Tchebychef objective with the optimistic values used in the dual update. For each $t \in$ $[ T - 1 ]$ , we define $\widetilde { V } ^ { t + 1 } : = ( \widetilde { V } _ { 1 } ^ { t + 1 } , \cdot \cdot \cdot , \widetilde { V } _ { m } ^ { t + 1 } )$ and $V ^ { t + 1 } : = ( V _ { 1 } ^ { t + 1 } , \dots , \bar { V } _ { m } ^ { t + 1 } )$ . We also define the constant vector $\pmb { \iota } : = ( \iota , \ldots , \iota ) \in \mathbb { R } ^ { m }$ . Then we obtain

$$
\begin{array} { r l } & { \mathrm { ( I I ) } \le \underbrace { \displaystyle \operatorname* { m a x } _ { w \in \Delta _ { m } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left[ \left. w - w ^ { t } , \lambda \odot \left( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } \right) \right. - \mu \langle w , \log w \rangle + \mu \langle w ^ { t } , \log w ^ { t } \rangle \right] } _ { \mathrm { ( I I . 1 ) } } } \\ & { \quad + \underbrace { \displaystyle \frac { 2 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ( \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ) - \widetilde { V } _ { i } ^ { t + 1 } \right| } _ { \mathrm { ( I I . 2 ) } } + \underbrace { \frac { 2 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ^ { t + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \right| } _ { \mathrm { ( I I . 3 ) } } . } \end{array}
$$

Lemma 33 controls the dual mirror-ascent term from the entropic update of $\{ w ^ { t } \} _ { t = 1 } ^ { T - 1 }$ Therefore, we have $\begin{array} { r } { \mathrm { ( I I . 1 ) } \le \widetilde { \mathcal { O } } ( \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } ) } \end{array}$ . For every $i \in [ m ]$ and $t \in [ T - 1 ]$ , the reward confidence event gives $0 \leq \bar { r } _ { i } ^ { t } - r _ { i } \leq 2 \sigma ^ { t }$ . Thus, we obtain $| V _ { i } ( \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ) - \widetilde { V } _ { i } ^ { t + 1 } | \ \leq$ $2 \langle \sigma ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \rangle + \| \widetilde { \theta } _ { i } ^ { t + 1 } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \| _ { 1 }$ . The reward bonus term is controlled by summing the bonuses as in the proof of Lemma 30. The transition term is controlled by Lemma 31, together with the bound on $\scriptstyle \sum _ { t = 1 } ^ { { T - 1 } } \psi _ { i } ^ { t }$ above. Consequently, we have $( \mathrm { I I . 2 } ) \le \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } +$ $\phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$ . Using $0 \leq \bar { r } _ { i } ^ { t } - r _ { i } \leq 2 \sigma ^ { t }$ and $\theta ^ { P , \pi ^ { t + 1 } } = ( 1 - \psi ^ { t } ) \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } + \psi ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ we have $\vert V _ { i } ^ { t + \mathrm { i } } - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \vert \le 2 \langle \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle + \Vert \widetilde { \theta } ^ { t + 1 } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \Vert _ { 1 } + H \psi ^ { t }$ . The reward bonus term is bounded by the argument in the proof of Lemma 29, and the transition term is controlled by Lemma 28. The policy decomposition used in the proof of Theorem 10, together with Lemma 32, gives $\begin{array} { r } { \dot { \sum } _ { t = 1 } ^ { T - 1 } \dot { \psi } ^ { t } \leq \widetilde { \mathcal { O } } ( \dot { \phi } _ { \mathrm { s a f e } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | T } ) } \end{array}$ . Together, these bounds give $( \mathrm { I I . 3 } ) \le \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$ . Combining the bounds for (II.1), (II.2), and (II.3), we obtain $\begin{array} { r } { \mathrm { ( I I ) } \le \widetilde { \mathcal { O } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } + \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } ) } \end{array}$ Bounding Term (III). Since $V _ { i } ( \theta ) = \langle r _ { i } , \theta \rangle$ , Term (III) can be written as

$$
\mathrm { ( I I I ) } = \underbrace { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde \theta ^ { t + 1 } \right. } _ { \mathrm { ( I I I . 1 ) } } + \underbrace { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \widetilde \theta ^ { t + 1 } - \theta ^ { P , \pi ^ { t + 1 } } \right. } _ { \mathrm { ( I I I . 2 ) } } .
$$

Since $\theta _ { \lambda } ^ { \ast } \in \Theta _ { \mathrm { s a f e } }$ , Lemma 29 applies on the high-probability event and gives $( \mathrm { I I I . 1 } ) \ \leq$ $\begin{array} { r } { \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } } \end{array}$ . For the second term, the scalarized reward $\scriptstyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i }$ is bounded in [0, 1] since $w ^ { t } , \lambda \in \Delta _ { m }$ and $r _ { i } \in [ 0 , 1 ]$ . Using the decomposition of $\theta ^ { P , \pi ^ { t + 1 } }$ above and Lemma 28, we obtain (III.2 $\begin{array} { r } { \mathrm { \Lambda } : \tilde { \mathcal { O } } ( H ^ { \hat { 2 } } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } ) + \frac { 3 H } { T } \dot { \sum } _ { t = 1 } ^ { T - 1 } \psi ^ { t } } \end{array}$ . Thus, we obtain $\mathrm { ( I I I ) } \le \widetilde { \mathcal { O } } ( H ^ { 2 } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | A | / T } )$

Bounding Term (IV). Since $\begin{array} { r } { { \pmb w } , \pmb { \lambda } \in \Delta _ { m } , } \end{array}$ we have $\begin{array} { r } { \mathrm { ( I V ) } \le \frac { H } { T } \sum _ { i = 1 } ^ { m } \lambda _ { i } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } } \end{array}$ . Using the bound on $\scriptstyle \sum _ { t = 1 } ^ { - T - 1 } \psi _ { i } ^ { t }$ above, we have $\mathrm { ( I V ) } \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$

Final Bound. Combining the bounds for Terms (I), (II), (III), and (IV) with the $H / T$ contribution of the initial policy, we obtain, with probability at least $1 - \delta .$

$$
\mathrm { S u b O p t } ( \widehat \pi ) \leq \widetilde O \left( H ^ { 2 } | { \cal S } | ^ { 3 / 2 } \sqrt { | { \cal A } | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | { \cal S } | ^ { 3 / 2 } \sqrt { | { \cal A } | / T } + \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } \right) .
$$

Remark 22 In the unconstrained MORL analysis of Qiu et al. (2024), greedy updates and optimism bound the auxiliary and main learning errors using bonus sums. This argument does not directly apply under safety constraints, since a policy chosen greedily with respect to optimistic Q-functions may violate the safety constraints. We therefore apply mirror ascent followed by SafeProj to every auxiliary and main occupancy measure update. Accordingly, we bound the auxiliary and main learning errors in Terms (I) and (III) directly using the occupancy measure updates rather than the optimistic Bellman recursions used by Qiu et al. (2024). The probabilities of selecting $\pi ^ { \dagger }$ do not appear in their unconstrained analysis. The probabilities $\psi _ { i } ^ { t }$ in the auxiliary updates enter Terms (I), (II), and (IV), while the probabilities $\psi ^ { t }$ in the main updates enter Terms (II) and (III). Bounding their sums relies on Lemma 32 and yields the additional term involving $\phi _ { \mathrm { s a f e } }$ in the suboptimality bound.

## 7.2 Proof Sketch of Theorem 10

The full proof is given in Appendix E. We work on the high-probability event on which $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t }$ and $P \in \mathcal { P } ^ { t }$ for all $t \in [ T - 1 ]$ . In the first episode, the constraint costs of $\widetilde { \pi } ^ { 1 }$ are each at most d. The choice of $\psi ^ { 0 }$ therefore gives $( 1 - \psi ^ { 0 } ) d + \psi ^ { 0 } \gamma _ { j } \leq \tau _ { j }$ for every $j \in [ n ]$ . The same argument applies to the initial auxiliary policies. For each later episode, SafeProj uses $\pi ^ { \dagger }$ if the projection in (11) is infeasible. Otherwise, the cost confidence bound and $P \in \mathcal { P } ^ { t }$ provide the cost bound for $\widetilde { \pi } ^ { t + 1 }$ . The choice of $\psi ^ { t }$ in (12) then ensures $C ^ { \top } \theta ^ { P , \pi ^ { t + 1 } } \leq \tau$ . The same argument applies to each auxiliary policy $\nu _ { i } ^ { t + 1 }$ . Since $\theta ^ { P , \pi ^ { t } } \in \Theta _ { \mathrm { s a f e } }$ for every $t \in [ T ]$ ], convexity gives $\bar { \theta } \in \Theta _ { \mathrm { s a f e } }$ . Thus, the induced Markov policy ${ \widehat { \pi } } = \pi ^ { \widehat { \theta } }$ is safe.

## 7.3 Proof Sketch of Theorem 11

The complete proof of Theorem 11 is provided in Appendix F. The analysis is carried out on the joint event defined in Appendix F, which holds with probability at least $1 - \delta$ . Let $\bar { \theta }$ be the averaged occupancy measure returned by Algorithm 3. We use the optimal safe occupancy measure $\theta _ { \lambda } ^ { * }$ introduced in Section 7.1. The first step is to replace the true singleobjective optimal values $\{ V _ { i } ^ { * } \} _ { i = 1 } ^ { m }$ by the fixed optimistic single-objective values $\{ \widetilde { V } _ { i } \} _ { i = 1 } ^ { m }$ computed in the planning stage. By Lemma 39, on the joint event, for all $i \in \ \lceil m \rceil$ , we have $| \widetilde { V } _ { i } - V _ { i } ^ { * } | \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } + H / \sqrt { K } )$ . Since $\begin{array} { r } { V _ { i } ( \bar { \theta } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } V _ { i } ( \theta ^ { P , \pi ^ { k + 1 } } ) } \end{array}$ and $V _ { i } ^ { k + 1 } = \langle \bar { r } _ { i } , \widetilde { \theta } ^ { k + 1 } \rangle$ , we obtain

$$
\begin{array} { r l } & { \operatorname { S u b O p t } ( \widehat \pi ) \leq \underbrace { \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { i = 1 } ^ { m } \lambda _ { i } \left. V _ { i } ^ { k + 1 } - V _ { i } ( \theta ^ { P , \alpha ^ { k + 1 } } ) \right. } _ { ( 1 ) } } \\ & { \qquad + \underbrace { \operatorname* { m a x } _ { \omega \in \Delta _ { m } } \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { i = 1 } ^ { m } \left[ ( w _ { i } - w _ { i } ^ { k } ) \lambda _ { i } \left( \widetilde { V _ { i } } + \iota - V _ { i } ^ { k + 1 } \right) - \mu w _ { i } \log w _ { i } + \mu w _ { i } ^ { k } \log w _ { i } ^ { k } \right] } _ { ( 1 ) } } \\ & { \qquad + \underbrace { \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \left( V _ { i } ( \theta _ { \lambda } ^ { \star } ) - V _ { i } ^ { k + 1 } \right) } _ { ( \mathrm { u l l } ) } + \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a c } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } + H / \sqrt { K } \right) . } \end{array}\tag{14}
$$

Term (I) compares the optimistic main value $V _ { i } ^ { k + 1 }$ with the true value of the policy $\pi ^ { k + 1 }$ Terms (II) and (III) are the dual and primal planning errors, respectively.

Bounding Term (I). The reward confidence event gives $0 ~ \leq ~ \bar { r } _ { i } - r _ { i } ~ \leq ~ 2 \sigma$ . Since $\widetilde { \theta } ^ { k + 1 } \in \Theta ( \mathcal { P } )$ , inequality (60) bounds the transition error. By construction of $\pi ^ { k + 1 }$ , we have $\theta ^ { P , \pi ^ { k + 1 } } = ( 1 - \psi ^ { k } ) \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } + \psi ^ { k } \theta ^ { P , \pi ^ { \dagger } }$ . Combining this decomposition with the reward and transition bounds, we obtain $| V _ { i } ^ { k + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { k + 1 } } ) | \overset { - } { \underset { - } { \le } } 2 \langle \sigma , \widetilde { \theta } ^ { k + 1 } \rangle + 2 \sqrt { | S | } \langle \zeta , \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle + H \psi ^ { k }$ . After averaging over k, Lemma 35 bounds the bonus terms, and Lemma 38 bounds ${ \frac { 1 } { K } } \sum k = 1 ^ { \psi ^ { k } }$ Therefore, we have $( \mathrm { I } ) \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } )$

Bounding Term (II). For the planning stage, we define $\widetilde { V } : = ( \widetilde { V } _ { 1 } , \dots , \widetilde { V } _ { m } )$ . For each $k \in [ K ]$ , we define $\dot { V } ^ { k + 1 } : = ( V _ { 1 } ^ { k + 1 } , \dots , V _ { m } ^ { k + 1 } )$ . Applying Lemma 33 to the K dual updates with $\pmb { \lambda } \odot ( \widetilde { \pmb { V } } + \pmb { \iota } - \pmb { V } ^ { k + 1 } )$ gives $\begin{array} { r } { \mathrm { ( I I ) } \le \widetilde { \mathcal { O } } ( \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } ) } \end{array}$

Bounding Term (III). Since $\bar { r } _ { i } \geq r _ { i }$ and $\widetilde { \theta } ^ { k + 1 } \geq 0$ , we have $V _ { i } ( \theta _ { \lambda } ^ { * } ) - V _ { i } ^ { k + 1 } = \langle r _ { i } , \theta _ { \lambda } ^ { * } \rangle -$ $\langle \bar { r } _ { i } , \widetilde { \theta } ^ { k + 1 } \rangle \leq \langle r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde { \theta } ^ { k + 1 } \rangle$ ⟩. Therefore, we obtain $\begin{array} { r } { \mathrm { ( I I I ) } \le \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \langle \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde \theta ^ { k + 1 } \rangle } \end{array}$ Since $\theta _ { \lambda } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ , by Lemma 37 with $\theta = \theta _ { \lambda } ^ { * }$ , we have $\begin{array} { r } { \mathrm { ( I I I ) } \le \widetilde { \mathcal { O } } ( \frac { H } { \sqrt { K } } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | \boldsymbol { S } | ^ { 2 } \sqrt { \frac { | \boldsymbol { A } | } { T } } ) } \end{array}$ Final Bound. Combining (14) and the bounds for Terms (I), (II), and (III), we obtain

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \bigg ( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } + H / \sqrt { K } + \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \bigg ) .
$$

## 7.4 Proof Sketch of Theorem 12

The full proof is given in Appendix G. We work on a joint high-probability event that combines the cost and transition confidence bounds over the $T$ exploration episodes with the confidence bounds computed from the final exploration dataset. PF-CSTRL constructs its initial policy as in CSTRL and uses the same SafeProj subroutine. The proof of Theorem 10 therefore gives $C ^ { \top } \theta ^ { P , \pi ^ { t } } \leq \tau$ for every $t \in [ T ]$ . Since $( \widehat { C } , \Xi , \mathcal { P } )$ remain fixed during planning, the bounds $| \widehat { C } - C | \leq \Xi$ and $P \in { \mathcal { P } }$ continue to hold. The same SafeProj argument gives $C ^ { \top } \theta ^ { P , \pi ^ { k + 1 } } \leq \tau$ for every $k \in [ K ]$ . Since $\theta ^ { P , \pi ^ { k + 1 } } \in \Theta _ { \mathrm { s a f e } }$ for every $k \in [ K ]$ , convexity gives $\bar { \theta } \in \Theta _ { \mathrm { s a f e } }$ . Thus, the induced Markov policy ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ is safe.

## 7.5 Proof Sketch for the Unknown Constraint Cost Extensions

The complete proofs are provided in Appendix H. By Hoefding’s inequality and the stopping rule, we have $t _ { s } = \widetilde { \mathcal { O } } ( d ^ { 2 } / ( \varrho ^ { \dagger } ) ^ { 2 } )$ and $\gamma _ { j } \leq \widehat { \gamma } _ { j } \leq ( \gamma _ { j } + \tau _ { j } ) / 2$ for every $j \in [ n ]$ with high probability. The assumed lower bound on $T$ then gives $t _ { s } < T$ . The bound on $\widehat { \gamma }$ also gives $\dot { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$ . Every policy executed during the estimation phase is $\pi ^ { \dagger }$ and is therefore safe. Using $\widehat { \gamma }$ as a conservative upper bound, the analyses underlying Theorems 9 and 10 apply to the remaining $T - t _ { s }$ episodes, yielding Theorem 14 and Theorem 15. With the same conservative bound $\widehat { \gamma }$ , the PF-CSTRL analyses apply to the remaining $T - t _ { s }$ exploration episodes and K planning iterations, giving Theorems 16 and 17.

## 7.6 Proof Sketch for the Unknown Safe Policy Extensions

The complete proofs are provided in Appendix I. Lemma 43, together with the condition on $T$ stated above, gives $T > t _ { s }$ with high probability. On the same event, Lemma 44 gives $\widehat { \varrho } \geq \varrho / 2 , C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$ , and $\widehat { \phi } _ { \mathrm { s a f e } } \leq \bar { 4 } d ^ { 2 } / \varrho ^ { 2 }$ . The estimated $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$ are then fixed. The proof of Theorem 9 applies to the remaining $T - t _ { s }$ episodes, giving Theorem 18. Similarly, the proof of Theorem 11 applies to the remaining $T - t _ { s }$ exploration episodes and K planning iterations, giving Theorem 20. During the estimation phase, each executed policy has violation at most d. With the estimated $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$ , the proofs of Theorems 10 and 12 establish safety of all policies executed after estimation, the main policies generated during planning, and the returned policies. Consequently, both extensions satisfy $\sum _ { t = 1 } ^ { T }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$ which proves Theorems 19 and 21.

## 8 Conclusion

We develop a unified framework for safe multi-objective reinforcement learning that covers both cumulative and stepwise safety constraints. Pessimistic cost bounds ensure safety throughout learning, while averaging the resulting occupancy measures produces a single safe Markov policy. Separating exploration from planning also allows the learner to compute a policy for any given preference from the collected data, without further interaction with the environment. In both preference-based and preference-free settings, zero constraint violation can be guaranteed when a strictly safe policy is known but its costs are unknown. When the strictly safe policy itself is also unknown, the cumulative constraint violation remains bounded by a constant.

Appendix Contents   
A Proof of Proposition 6 22   
B Proof of Proposition 7 23   
C Proof of Proposition 8 24   
D Proof of Theorem 9 24   
D.1 Lemmas . 25   
D.2 Proof of Theorem 9 . 36   
E Proof of Theorem 10 40   
F Proof of Theorem 11 41   
F.1 Lemmas . 41   
F.2 Proof of Theorem 11 46   
G Proof of Theorem 12 48   
H Proofs for the Unknown Constraint Cost Extensions 48   
H.1 Proof of Theorem 14 . 49   
H.2 Proof of Theorem 15 . 49   
H.3 Proof of Theorem 16 . 50   
H.4 Proof of Theorem 17 . 50   
I Proofs for the Unknown Safe Policy Extensions 50   
I.1 Lemmas 51   
I.2 Proof of Theorem 18 53   
I.3 Proof of Theorem 19 54   
I.4 Proof of Theorem 20 54   
I.5 Proof of Theorem 21 55   
J Other Supporting Lemmas 55

## Appendix A. Proof of Proposition 6

Proof We work with the state-action marginals $\theta _ { h } ( s , a )$ of an occupancy measure. Since the transition kernel P is fixed, these marginals determine the corresponding occupancy measure through $\theta _ { h } ( s , a , s ^ { \prime } ) \ : = \ : \theta _ { h } ( s , a ) P _ { h } ( s ^ { \prime } | s , a )$ . Let $\theta \ = \ ( \theta _ { h } ( s , a ) ) _ { h \in [ H ] , s \in S , a \in \mathcal { A } }$ be a vector in $\mathbb { R } ^ { H | S | | A | }$ , and consider the following constraints:

$$
\theta _ { h } ( s , a ) \geq 0 ,
$$

$$
\forall h \in [ H ] , \ s \in { \mathcal { S } } , \ a \in { \mathcal { A } } ,\tag{15}
$$

$$
\sum _ { a \in \mathcal { A } } \theta _ { 1 } ( s , a ) = \rho ( s ) ,
$$

$$
\forall s \in S ,\tag{16}
$$

$$
\sum _ { s \in S } \sum _ { a \in \mathcal { A } } \theta _ { h } ( s , a ) P _ { h } ( s ^ { \prime } | s , a ) = \sum _ { a ^ { \prime } \in \mathcal { A } } \theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } ) , \qquad \forall h \in [ H - 1 ] , ~ s ^ { \prime } \in S .\tag{17}
$$

We show below that $\theta \in \Theta ( P )$ if and only if its marginals satisfy (15)–(17).

We next establish the correspondence between stochastic Markov policies and the elements of $\Theta ( P )$ . For any policy $\pi \in \Pi$ , its state-action occupancy measure $\theta ^ { P , \pi }$ satisfies (15) and (16). Moreover, since $\begin{array} { r } { \sum _ { s \in S } \sum _ { a \in \mathcal { A } } \theta _ { h } ^ { P , \pi } ( s , a ) P _ { h } ( s ^ { \prime } | s , a ) = \operatorname* { P r } ( s _ { h + 1 } = s ^ { \prime } | \rho , \pi , P ) = } \end{array}$ $\textstyle \sum _ { a ^ { \prime } \in { \mathcal { A } } } \theta _ { h + 1 } ^ { P , \pi } ( s ^ { \prime } , a ^ { \prime } )$ , (17) also holds. Since $\Theta ( P ) = \{ \theta ^ { P , \pi } : \pi \in \Pi \}$ , every $\theta \in \Theta ( P )$ satisfies (15)–(17).

Conversely, take any θ satisfying (15)–(17). We define its state marginal as $\theta _ { h } ( s ) : =$ $\textstyle \sum _ { a \in { \mathcal { A } } } \theta _ { h } ( s , a )$ . We construct a Markov policy $\pi ^ { \theta } = \{ \pi _ { h } ^ { \theta } \} _ { h = 1 } ^ { H }$ as follows. For each $h \in [ H ]$ and $s \in S .$ , if $\theta _ { h } ( s ) > 0$ , we set $\pi _ { h } ^ { \theta } ( a | s ) : = \theta _ { h } ( s , a ) / \theta _ { h } ( s )$ for every $a \in { \mathcal { A } }$ . If $\theta _ { h } ( s ) = 0$ , we set $\pi _ { h } ^ { \theta } ( \cdot | s )$ to the uniform distribution over A. We prove by induction that $\theta _ { h } ^ { P , \pi ^ { \theta } } ( s , a ) = \theta _ { h } ( s , a )$ for all $h \in [ H ] , s \in \mathcal { S }$ , and $a \in { \mathcal { A } }$ . For $h = 1$ , (16) gives $\theta _ { 1 } ( s ) = \rho ( s )$ . If $\theta _ { 1 } ( s ) > 0$ , then $\theta _ { 1 } ^ { P , \pi ^ { \theta } } ( s , a ) = \rho ( s ) \pi _ { 1 } ^ { \theta } ( a | s ) = \theta _ { 1 } ( s , a )$ . If $\theta _ { 1 } ( s ) = 0$ , nonnegativity implies $\theta _ { 1 } ( s , a ) = 0$ for every $a \in { \mathcal { A } }$ . Since $\rho ( s ) = 0$ , we also have $\theta _ { 1 } ^ { P , \pi ^ { \theta } } ( s , a ) = 0$ . Thus, the induction claim holds for $h = 1$

Suppose that it holds at some $h \in [ H - 1 ]$ . For every $s ^ { \prime } \in \mathcal { S }$ , we have

$$
\operatorname* { P r } ( s _ { h + 1 } = s ^ { \prime } | \rho , \pi ^ { \theta } , P ) = \sum _ { s \in \mathcal { S } } \sum _ { a \in \mathcal { A } } \theta _ { h } ( s , a ) P _ { h } ( s ^ { \prime } | s , a ) = \sum _ { a ^ { \prime } \in \mathcal { A } } \theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } ) = \theta _ { h + 1 } ( s ^ { \prime } ) ,
$$

where the second equality follows from (17). If $\theta _ { h + 1 } ( s ^ { \prime } ) > 0$ , then $\theta _ { h + 1 } ^ { P , \pi ^ { \theta } } ( s ^ { \prime } , a ^ { \prime } ) = \operatorname* { P r } ( s _ { h + 1 } =$ $s ^ { \prime } | \rho , \pi ^ { \theta } , P ) \pi _ { h + 1 } ^ { \theta } ( a ^ { \prime } | s ^ { \prime } ) = \theta _ { h + 1 } ( s ^ { \prime } ) \theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } ) / \theta _ { h + 1 } ( s ^ { \prime } ) = \theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } )$ . If $\theta _ { h + 1 } ( s ^ { \prime } ) = 0$ , both $\theta _ { h + 1 } ^ { P , \pi ^ { \theta } } ( s ^ { \prime } , a ^ { \prime } )$ and $\theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } )$ are zero. The induction is therefore complete. Since $\theta _ { h } ( s , a , s ^ { \prime } ) =$ $\theta _ { h } ( s , a ) P _ { h } ( s ^ { \prime } | s , a )$ , we obtain $\theta ^ { P , \pi ^ { \theta } } = \theta$ . Consequently, every $\theta ~ \in ~ \Theta ( P )$ is induced by a stochastic Markov policy, and $\Theta ( P ) \ : = \ : \{ \theta ^ { P , \pi } \ : : \ : \pi \ : \in \ : \Pi \}$ Since Definition 1 requires $\theta _ { h } \in \Delta _ { | S | \times | A | \times | S | }$ , the marginals satisfy $\begin{array} { r } { \sum _ { s , a } \theta _ { h } ( s , a ) = 1 } \end{array}$ for every $h \in [ H ]$ . Thus, $\Theta ( P )$ is defined by finitely many linear equalities and inequalities. Since it is bounded, $\Theta ( P )$ is a convex polytope.

We next impose the safety constraints. The safe occupancy measure set admits the following linear characterization:

$$
\Theta _ { \mathrm { s a f e } } = \left\{ \theta \in \Theta ( P ) : \sum _ { h = 1 } ^ { H } \sum _ { s \in \mathcal { S } } \sum _ { a \in A } \theta _ { h } ( s , a ) c _ { j , h } ( s , a ) \leq \tau _ { j } , ~ \forall j \in [ n ] \right\} .\tag{18}
$$

The occupancy measure of any $\pi \in \Pi _ { \mathrm { s a f e } }$ belongs to $\Theta ( P )$ and satisfies the inequalities in (18). Conversely, if θ belongs to the set on the right-hand side of (18), the policy $\pi ^ { \theta }$ constructed above satisfies $\theta ^ { P , \pi ^ { \theta } } = \theta$ . It therefore satisfies $C ^ { \top } \theta ^ { P , \pi ^ { \theta } } \leq \tau$ and belongs to $\Pi _ { \mathrm { s a f e } }$ This proves (18). Since (18) adds finitely many linear inequalities to the constraints defining $\Theta ( P ) , \Theta _ { \mathrm { s a f e } }$ is a convex polytope. Assumption 2 guarantees that it is nonempty. Moreover, the correspondence between policies and occupancy measures gives $\mathcal { V } _ { \mathrm { s a f e } } = \{ V ( \theta ^ { P , \pi } ) : \pi \in$ $\Pi _ { \mathrm { s a f e } } \} = \{ V ( \theta ) : \theta \in \Theta _ { \mathrm { s a f e } } \}$

Since $V ( \cdot )$ is linear, $\mathcal { V } _ { \mathrm { s a f e } }$ is the linear image of $\Theta _ { \mathrm { s a f e } }$ . The linear image of a polytope is a polytope, since a polytope is the convex hull of finitely many points and linear mappings preserve convex combinations. It is nonempty since $\Theta _ { \mathrm { s a f e } }$ is nonempty. Therefore, $\mathcal { V } _ { \mathrm { s a f e } }$ is a nonempty convex polytope. This completes the proof. ■

## Appendix B. Proof of Proposition 7

Proof Under Assumption 2, Proposition 6 ensures that $\mathcal { V } _ { \mathrm { s a f e } }$ is a nonempty convex polytope. Fix any $\mu > 0$

For any $\lambda \in \Delta _ { m } ^ { o }$ , consider a policy $\pi ^ { * } \in \arg \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi )$ . Suppose for con tradiction that $\pi ^ { \ast } \notin \Pi _ { \mathrm { s a f e } } ^ { \ast } .$ . Definition 4 then implies that there exists a policy $\pi ^ { \prime } \in \Pi _ { \mathrm { s a f e } }$ satisfying $V _ { i } ( \theta ^ { P , \pi ^ { \prime } } ) \ge V _ { i } ( \theta ^ { P , \pi ^ { * } } )$ for every $i \in [ m ]$ with strict inequality for at least one $i _ { 0 } \in [ m ]$ . Since $\lambda \in \Delta _ { m } ^ { o }$ , we have $\lambda _ { i } > 0$ for every $i \in [ m ]$ . Consequently, each exponential term in $\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \prime } )$ is no larger than its counterpart under $\pi ^ { * }$ , and the term corresponding to $i _ { 0 }$ is strictly smaller. We therefore obtain $\mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \prime } ) < \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { * } )$ . This contradicts the optimality of $\pi ^ { * }$ . Thus, every minimizer is safe Pareto optimal.

Conversely, consider any $\pi ^ { 0 } \in \Pi _ { \mathrm { s a f e } } ^ { * }$ and let $\pmb { v } ^ { 0 } : = \pmb { V } ( \theta ^ { P , \pi ^ { 0 } } )$ . Then $ { \boldsymbol { v } } ^ { 0 }$ is a Pareto optimal point of $\nu _ { \mathrm { s a f e } }$ . Since $\nu _ { \mathrm { s a f e } }$ is a convex polytope, a standard polyhedral supporting-hyperplane argument gives a vector $\pmb { a } = ( a _ { 1 } , \dots , a _ { m } ) \in \mathbb { R } _ { + + } ^ { m }$ and a constant $\kappa \in \mathbb { R }$ satisfying $\pmb { a } ^ { \top } \pmb { v } ^ { 0 } = \kappa$ and $\mathbf { \mathfrak { a } } ^ { \top } \pmb { v } \le \kappa$ for every $v \in \mathcal { V } _ { \mathrm { s a f e } }$ . We regard $\operatorname { S T C H } _ { \lambda } ^ { \mu }$ as a function on $\mathbb { R } ^ { m }$ by replacing $V _ { i } ( \theta ^ { P , \pi } )$ in Definition 5 with $v _ { i }$ for any $\pmb { v } = ( v _ { 1 } , \dots , v _ { m } ) \in \mathbb { R } ^ { m }$ . For every $i \in [ m ]$ , the definition of $V _ { i } ^ { * }$ gives $V _ { i } ^ { * } + \iota - v _ { i } ^ { 0 } \geq \iota > 0$

We next construct a preference vector for which $ { \boldsymbol { v } } ^ { 0 }$ minimizes the scalarization over the supporting hyperplane. For each $i \in [ m ]$ , the function $x \mapsto x \exp ( x ( V _ { i } ^ { * } + \iota - v _ { i } ^ { 0 } ) / \mu )$ is continuous and strictly increasing on $[ 0 , \infty )$ . It equals zero at $x = 0$ and tends to infinity as $x  \infty$ . Therefore, for every $w > 0$ , we define $\lambda _ { i } ( w ) > 0$ as the unique solution to $\begin{array} { r } { \lambda _ { i } ( w ) \exp ( \frac { \lambda _ { i } ( w ) ( V _ { i } ^ { * } + \iota - v _ { i } ^ { 0 } ) } { \mu } ) = \mu w a _ { i } } \end{array}$ . Since each of these functions is continuous and strictly increasing, its inverse is also continuous and strictly increasing. Moreover, we have $\lambda _ { i } ( w ) \to$ 0 as $w \to 0 ^ { + }$ and $\lambda _ { i } ( w ) \to \infty { \mathrm { ~ a s ~ } } w \to \infty$ . Consequently, the mapping $\begin{array} { r } { w \mapsto \sum _ { i = 1 } ^ { m } \lambda _ { i } ( w ) } \end{array}$ is continuous and strictly increasing from zero to infinity. The intermediate value theorem gives a unique $w ^ { * } > 0$ satisfying $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \lambda _ { i } ( w ^ { * } ) = 1 } \end{array}$ . We define $\lambda ^ { 0 } : = ( \lambda _ { 1 } ( w ^ { * } ) , \ldots , \lambda _ { m } ( w ^ { * } ) )$ . Then $\pmb { \lambda } ^ { 0 } \in \Delta _ { m } ^ { o }$ , and we have

$$
\lambda _ { i } ^ { 0 } \exp \left( \frac { \lambda _ { i } ^ { 0 } ( V _ { i } ^ { * } + \iota - v _ { i } ^ { 0 } ) } { \mu } \right) = \mu w ^ { * } a _ { i } , \quad \forall i \in [ m ] ,\tag{19}
$$

where $\lambda _ { i } ^ { 0 } : = \lambda _ { i } ( w ^ { * } )$

Since the logarithm is strictly increasing, minimizing $\mathrm { S T C H } _ { \lambda ^ { 0 } } ^ { \mu }$ is equivalent to minimizing its exponential sum. This exponential sum is strictly convex in the value vector. Equation (19) is precisely the first-order optimality condition at $\boldsymbol { v } ^ { 0 }$ for minimizing this sum subject to $\mathbf { \boldsymbol { a } } ^ { \top } \mathbf { \boldsymbol { v } } = \kappa$ . Since the exponential sum is strictly convex and ${ \pmb a } ^ { \top } { \pmb v } = \kappa$ is a linear equality constraint, this first-order condition is suficient for global optimality. Therefore, $\boldsymbol { v } ^ { 0 }$ minimizes $\mathrm { S T C H } _ { \lambda ^ { 0 } } ^ { \mu }$ over the supporting hyperplane.

It remains to extend this conclusion to $\mathcal { V } _ { \mathrm { s a f e } }$ . Consider any $v \in \mathcal { V } _ { \mathrm { s a f e } }$ . The supportinghyperplane inequality gives $\frac { \kappa - \pmb { a } ^ { \top } \pmb { v } } { \sum _ { i = 1 } ^ { m } a _ { i } } \ge 0$ . Let $\mathbf { 1 } \in \mathbb { R } ^ { m }$ denote the all-ones vector. We further define $\begin{array} { r } { \pmb { u } : = \pmb { v } + \frac { \kappa - \pmb { a } ^ { \top } \pmb { v } } { \sum _ { i = 1 } ^ { m } a _ { i } } \pmb { 1 } } \end{array}$ . Then $\mathbf { \delta } \mathbf { a } ^ { \top } \mathbf { \boldsymbol { u } } = \kappa$ and ${ \textbf { \em u } } \geq v$ componentwise. Since every component of $\lambda ^ { 0 }$ is positive, the scalarization is strictly decreasing in each coordinate. We therefore have $\mathrm { S T C H } _ { \lambda ^ { 0 } } ^ { \mu } ( \pmb { v } ^ { 0 } ) \leq \mathrm { S T C H } _ { \lambda ^ { 0 } } ^ { \mu } ( \pmb { u } ) \leq \mathrm { S T C H } _ { \lambda ^ { 0 } } ^ { \mu } ( \pmb { v } )$ . Thus, $ { \boldsymbol { v } } ^ { 0 }$ is a minimizer over $\mathcal { V } _ { \mathrm { s a f e } }$ . Equivalently, we obtain $\pi ^ { 0 } \in \arg \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C } \ddot { \mathrm { H } } _ { \lambda ^ { 0 } } ^ { \mu } ( \pi )$ . This completes the proof.

## Appendix C. Proof of Proposition 8

Proof Under Assumption 2, Proposition 6 ensures that $\nu _ { \mathrm { s a f e } }$ is a nonempty convex polytope. Since $\mathcal { V } _ { \mathrm { s a f e } }$ is compact and the scalarization is continuous in the value vector, its minimum over $\mathcal { V } _ { \mathrm { s a f e } }$ is attained. Consider any two policies $\pi ^ { 1 } , \pi ^ { 2 } \in \arg \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { s a f e } } } \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi )$ For each $k \in \{ 1 , 2 \}$ , we define $v ^ { k } : = V ( \theta ^ { P , \pi ^ { k } } )$ and $z ^ { k } \in \mathbb { R } ^ { m }$ by $z _ { i } ^ { k } : = \lambda _ { i } ( V _ { i } ^ { * } + \iota - v _ { i } ^ { k } ) / \mu$ for every $i \in [ m ]$ . Since $\mathcal { V } _ { \mathrm { s a f e } }$ is convex, we have $\bar { \pmb { v } } : = ( \pmb { v } ^ { 1 } + \pmb { v } ^ { 2 } ) / 2 \in \mathcal { V } _ { \mathrm { s a f e } }$ . Therefore, there exists a policy $\bar { \pi } \in \Pi _ { \mathrm { s a f e } }$ satisfying $V ( \theta ^ { P , \bar { \pi } } ) = \bar { v }$ . The argument of the log-sum-exp function at ¯π is $( z ^ { 1 } + z ^ { 2 } ) / 2$ . The mapping $z \mapsto \mu \log \bigl ( \sum _ { i = 1 } ^ { m } \exp ( z _ { i } ) \bigr )$ is convex. Its Hessian has null space spanned by 1. Therefore, the mapping is strictly convex along every segment whose direction is not a multiple of 1. If $z ^ { 1 } - z ^ { 2 }$ is not a multiple of 1, strict convexity gives

$$
\mathrm { S T C H } _ { \lambda } ^ { \mu } ( \bar { \pi } ) < \frac { \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { 1 } ) + \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { 2 } ) } { 2 } = \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { 1 } ) ,
$$

which contradicts the optimality of $\pi ^ { 1 }$ and $\pi ^ { 2 }$

It remains to consider the case $z ^ { 1 } - z ^ { 2 } = \alpha \dot { . }$ 1 for some $\alpha \in \mathbb { R }$ . The translation identity of the log-sum-exp function gives $\mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { 1 } ) = \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { 2 } ) + \mu \alpha$ . Since $\pi ^ { 1 }$ and $\pi ^ { 2 }$ are both minimizers, their scalarization values are equal. Therefore, we obtain $\alpha = 0$ and $z ^ { 1 } = z ^ { 2 }$ Since $\lambda \in \Delta _ { m } ^ { o } .$ we have $\lambda _ { i } > 0$ for every $i \in [ m ]$ . It follows that $v _ { i } ^ { 1 } = v _ { i } ^ { 2 }$ for every $i \in [ m ]$ Thus, all minimizers over $\Pi _ { \mathrm { s a f e } }$ achieve the same values on all objectives. This completes the proof.

## Appendix D. Proof of Theorem 9

We define the transition event $\mathcal { E } _ { T } ( \delta ) : = \{ P \in \mathcal { P } ^ { t }$ for all $t \in [ T - 1 ] \}$ , which holds with probability at least $1 - \delta$ by Lemma 23. We also define the cost event $\begin{array} { r } { \mathcal { \bar { E } } _ { C } ( \delta ) : = \{ | \widehat { C } ^ { t } - C | \le } \end{array}$ $\Xi ^ { t }$ for all $t \in [ T - 1 ] \}$ , which holds with probability at least $1 - \delta$ by Lemma 25. We further define the reward event $\mathcal { E } _ { R } ( \delta ) : = \{ | \hat { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) | \ \leq \ \sigma _ { h } ^ { t } ( s , a )$ , for all $i \in [ m ] , h \in$ $[ H ] , t \in [ T - 1 ] , ( s , a ) \in S \times \mathcal { A } \}$ . By Lemma $2 6 , \mathcal { E } _ { R } ( \delta )$ holds with probability at least $1 - \delta .$

Moreover, we define the joint confidence event $\mathcal { E } _ { R , C , T } ( \delta ) : = \mathcal { E } _ { R } ( \delta / 3 ) \cap \mathcal { E } _ { C } ( \delta / 3 ) \cap \mathcal { E } _ { T } ( \delta / 3 )$ . By a union bound, this event holds with probability at least $1 - \delta$ . The analysis also uses the martingale events introduced in the proofs of Lemmas 28 and 29. Throughout the proofs of Theorems 9 and 10, we work on the intersection of the events $\mathcal { E } _ { R , C , T }$ for the main trajectory sequence and each auxiliary trajectory sequence, together with the corresponding martingale events. There are $\mathcal { O } ( m )$ such events, and we divide the failure probability $\delta$ equally among them. By a union bound, this intersection holds with probability at least $1 - \delta$ , and this allocation afects only $\ln ( m / \delta )$ factors, which are already absorbed into the $\widetilde { \mathcal { O } } ( \cdot )$ notation.

## D.1 Lemmas

Lemma 23 We define the transition confidence set at episode t as

$$
\mathcal { P } ^ { t } : = \left\{ P ^ { \prime } : \left| P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \right| \leq \sqrt { \frac { 2 \ln ( 2 m T H | \mathcal { S } | ^ { 2 } | A | / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } , \forall h \in [ H ] , s , s ^ { \prime } \in \mathcal { S } , a \in \mathcal { A } \right\} .
$$

With probability at least $1 - \delta$ , the true transition kernel P belongs to $\mathcal { P } ^ { t }$ for all $t \in [ T ]$

Proof Fix any $( h , s , a , s ^ { \prime } ) \in [ H ] \times \mathcal { S } \times \mathcal { A } \times \mathcal { S }$ . Conditional on the history before each visit to $( s , a )$ at step h, the next state is $s ^ { \prime }$ with probability $P _ { h } ( s ^ { \prime } | s , a )$ . At a fixed step $h .$ , the pair $( s , a )$ can be visited at most once in each episode. Therefore, we have $N _ { h } ^ { t } ( s , a ) \le T$ Applying the Azuma-Hoefding inequality for each possible positive number of visits and taking a union bound over these possibilities and all $( h , s , a , s ^ { \prime } )$ yields, with probability at least $1 - \delta$

$$
\left| P _ { h } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \right| \le \sqrt { \frac { 2 \ln ( 2 m T H | \cal { S } } { N _ { h } ^ { t } ( s , a ) } | \cal { A } | / \delta ) }
$$

simultaneously for every $t \in [ T ]$ such that $N _ { h } ^ { t } ( s , a ) \ \geq \ 1$ . When $N _ { h } ^ { t } ( s , a ) = 0$ , the claim follows since both $P _ { h } ( s ^ { \prime } | s , a )$ and $\widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a )$ belong to $[ 0 , 1 ]$ and $\epsilon _ { h } ^ { t } ( s , a ) > 1$ . Therefore, we have $P \in \mathcal { P } ^ { t }$ for all $t \in [ T ]$ . This completes the proof. ■

Lemma 24 For every $t ~ \in ~ [ T - 1 ] , ~ \Theta ( \mathcal { P } ^ { t } )$ is exactly the set of $\theta \ : = \ : \{ \theta _ { h } \} _ { h = 1 } ^ { H }$ such that $\theta _ { h } \in \Delta _ { | S | \times | A | \times | S | }$ for every $h \in [ H ]$ and

$$
\begin{array} { r l r l } & { \displaystyle \sum _ { a \in \mathcal { A } } \displaystyle \sum _ { s ^ { \prime } \in \mathcal { S } } \theta _ { 1 } ( s , a , s ^ { \prime } ) = \rho ( s ) , } & & { \quad \forall s \in \mathcal { S } , } \\ & { \displaystyle \sum _ { a \in \mathcal { A } } \displaystyle \sum _ { s ^ { \prime } \in \mathcal { S } } \theta _ { h } ( s , a , s ^ { \prime } ) = \displaystyle \sum _ { s ^ { \prime } \in \mathcal { S } } \displaystyle \sum _ { a \in \mathcal { A } } \theta _ { h - 1 } ( s ^ { \prime } , a , s ) , } & & { \quad \forall h \in [ 2 , H ] , \ \forall s \in \mathcal { S } , } \end{array}
$$

and

$$
\begin{array} { r } { \left| \theta _ { h } ( s , a , s ^ { \prime } ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \theta _ { h } ( s , a ) \right| \le \epsilon _ { h } ^ { t } ( s , a ) \theta _ { h } ( s , a ) , \quad \forall h \in [ H ] , ( s , a , s ^ { \prime } ) \in \mathcal { S } \times A \times \mathcal { S } . } \end{array}
$$

In particular, $\Theta ( \mathcal { P } ^ { t } )$ is a convex polytope.

Proof Fix any $t \in [ T - 1 ]$ We first establish the stated characterization of $\Theta ( \mathcal { P } ^ { t } )$ Consider any $\theta \in \Theta ( \mathcal { P } ^ { t } )$ . By definition, there exist $P ^ { \prime } \in \mathcal { P } ^ { t }$ and $\pi \in \Pi$ such that $\theta =$ $\theta ^ { P ^ { \prime } , \pi }$ . Definition 1 gives nonnegative entries for each $\theta _ { h }$ whose sum is one, together with the two equalities displayed in the lemma. It also gives $\theta _ { h } ( s , a , s ^ { \prime } ) = P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) \theta _ { h } ( s , a )$ Since $P ^ { \prime } \in \mathcal { P } ^ { t }$ , we have $\left| P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \right| \le \epsilon _ { h } ^ { t } ( s , a )$ . Multiplying this inequality by $\theta _ { h } ( s , a ) \geq 0$ yields $\vert \theta _ { h } ( s , a , s ^ { \prime } ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } \vert s , a ) \theta _ { h } ( s , a ) \vert \le \epsilon _ { h } ^ { t } ( s , a ) \theta _ { h } ( s , a )$ . Thus, every element of $\Theta ( \mathcal { P } ^ { t } )$ satisfies the constraints in the lemma statement.

Conversely, consider any θ satisfying these constraints. First, $\mathcal { P } ^ { t }$ is nonempty. When $N _ { h } ^ { t } ( s , a ) \geq 1$ , the empirical estimate $\widehat { P } _ { h } ^ { t } ( \cdot | s , a )$ is a probability distribution. When $N _ { h } ^ { t } ( s , a ) =$ $0 _ { ; }$ we have $\widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) = 0$ for every $s ^ { \prime } \in \mathcal { S }$ and $\epsilon _ { h } ^ { t } ( s , a ) > 1$ , and therefore any probability distribution satisfies the corresponding inequalities. We may therefore fix some $\bar { P } ^ { t } \in \mathcal { P } ^ { t }$ . We define a transition kernel $P ^ { \prime }$ by setting $P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) : = \theta _ { h } ( s , a , s ^ { \prime } ) / \theta _ { h } ( s , a )$ whenever $\theta _ { h } ( s , a ) >$ 0. If $\theta _ { h } ( s , a ) = 0$ , we set $P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) : = \bar { P } _ { h } ^ { t } ( s ^ { \prime } | s , a )$ . When $\theta _ { h } ( s , a ) > 0$ , the nonnegativity of $\theta _ { h } ( s , a , s ^ { \prime } )$ and $\begin{array} { r } { \sum _ { s ^ { \prime } \in S } \theta _ { h } ( s , a , s ^ { \prime } ) = \theta _ { h } ( s , a ) } \end{array}$ imply that $P _ { h } ^ { \prime } ( \cdot | s , a )$ is a probability distribution. Dividing the inequality in the lemma statement by $\theta _ { h } ( s , a )$ gives $| P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) | \leq$ $\epsilon _ { h } ^ { t } ( s , a )$ . When $\theta _ { h } ( s , a ) = 0$ , the same inequality follows from $\bar { P } ^ { t } \in \mathcal { P } ^ { t }$ . Since the inequalities defining $\mathcal { P } ^ { t }$ are imposed separately for every $( h , s , a )$ , we conclude that $P ^ { \prime } \in \mathcal { P } ^ { t }$

By construction, we have $\theta _ { h } ( s , a , s ^ { \prime } ) = { \cal P } _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) \theta _ { h } ( s , a )$ whenever $\theta _ { h } ( s , a ) \ > \ 0$ . If $\theta _ { h } ( s , a ) \ = \ 0$ , the nonnegativity of $\theta _ { h }$ and the identity $\begin{array} { r } { \theta _ { h } ( s , a ) \ = \ \sum _ { s ^ { \prime } \in S } \theta _ { h } ( s , a , s ^ { \prime } ) } \end{array}$ imply that $\theta _ { h } ( s , a , s ^ { \prime } ) ~ = ~ 0$ for every $s ^ { \prime } \in \mathcal { S }$ The same equality therefore holds for all $( h , s , a , s ^ { \prime } )$ . Let $\pi ^ { \theta }$ be the policy induced by θ, with an arbitrary action distribution at any $( h , s )$ satisfying $\begin{array} { r } { \sum _ { a \in \mathcal { A } } \theta _ { h } ( s , a ) \ = \ 0 } \end{array}$ . For every $h \in [ H - 1 ]$ and $s ^ { \prime } \in \mathcal { S }$ , we have $\begin{array} { r } { \sum _ { s \in \cal S } \sum _ { a \in \cal A } \theta _ { h } ( s , a ) P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) = \sum _ { s \in \cal S } \sum _ { a \in \cal A } \theta _ { h } ( s , a , s ^ { \prime } ) = \sum _ { a ^ { \prime } \in \cal A } \theta _ { h + 1 } ( s ^ { \prime } , a ^ { \prime } ) } \end{array}$ , where the last equality is the second equality displayed in the lemma. Together with the first equality displayed in the lemma, these are exactly the equations used in the occupancy measure construction in $\mathrm { A p }$ pendix $\mathrm { A }$ , with $P ^ { \prime }$ in place of $P .$ The same induction therefore gives $\theta _ { h } ^ { P ^ { \prime } , \pi ^ { \theta } } ( s , a ) = \theta _ { h } ( s , a )$ for every $h \in [ H ]$ and $( s , a ) \in \mathcal { S } \times \mathcal { A }$ . Consequently, we have $\theta _ { h } ^ { P ^ { \prime } , \pi ^ { \theta } } ( s , a , s ^ { \prime } ) = P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) \theta _ { h } ^ { P ^ { \prime } , \pi ^ { \theta } } ( s , a ) = \theta _ { h } ( s , a , s ^ { \prime } )$ , which proves that $\theta = \theta ^ { P ^ { \prime } , \pi ^ { \theta } } \in \Theta ( \mathcal { P } ^ { t } )$ The stated set identity follows.

It remains to show that $\Theta ( \mathcal { P } ^ { t } )$ is a convex polytope. Since $\begin{array} { r } { \theta _ { h } ( s , a ) = \sum _ { s ^ { \prime } \in \mathcal { S } } \theta _ { h } ( s , a , s ^ { \prime } ) } \end{array}$ the inequality in the lemma statement is equivalent to two linear inequalities in θ. The simplex constraints and the two equalities in the lemma statement are linear. Since each $\theta _ { h }$ belongs to a probability simplex, all components are bounded. Therefore, $\Theta ( \mathcal { P } ^ { t } )$ is a convex polytope. This completes the proof. ■

Lemma 25 With probability at least $1 - \delta$ , we have $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t } , \forall t \in [ T ]$

Proof Fix any $j \in [ n ]$ , any $h \in [ H ]$ to which constraint $j$ applies, and any $( s , a ) \in \mathcal S \times \mathcal A$ Conditional on the history before each visit to $( s , a )$ at step $h ,$ the observed cost has mean $c _ { j , h } ( s , a )$ . At a fixed step h, the pair $( s , a )$ can be visited at most once in each episode. Therefore, we have $N _ { h } ^ { t } ( s , a ) ~ \leq ~ T$ . Applying the Azuma-Hoefding inequality for each possible positive number of visits and taking a union bound over these possibilities, all $j \in [ n ]$ , the d applicable time steps, and all $( s , a ) \in \mathcal { S } \times \mathcal { A }$ yields, with probability at least $\begin{array} { r } { 1 - \delta , \big | \widehat { c } _ { j , h } ^ { t } ( s , a ) - c _ { j , h } ( s , a ) \big | \leq \sqrt { \frac { \ln \left( 4 m n d T | S | | A | / \delta \right) } { 2 N _ { h } ^ { t } ( s , a ) } } } \end{array}$ simultaneously for every $t \in [ T ]$ such that $N _ { h } ^ { t } ( s , a ) \geq 1$ . Since both terms on the left belong to $[ 0 , 1 ]$ , clipping the right-hand side at one preserves the inequality. When $N _ { h } ^ { t } ( s , a ) = 0$ , the inequality follows from $\xi _ { j , h } ^ { t } ( s , a ) = 1$ Therefore, we have $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t }$ for all $t \in [ T ]$ . This completes the proof.

Lemma 26 With probability at least $1 - \delta _ { i }$ , for every $i \in [ m ] , \ h \in [ H ] , \ t \in [ T ]$ , and $( s , a ) \in \mathcal S \times \mathcal A$ , it holds that

$$
| \widehat { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) | \leq \sigma _ { h } ^ { t } ( s , a ) ,
$$

where $\begin{array} { r } { \sigma _ { h } ^ { t } ( s , a ) = \operatorname* { m i n } \lbrace 1 , \sqrt { \frac { 2 \ln ( 8 m \vert S \vert \vert A \vert H T / \delta ) } { \operatorname* { m a x } \lbrace 1 , N _ { h } ^ { t } ( s , a ) \rbrace } } \rbrace } \end{array}$

Proof Fix any $i \in [ m ] , h \in [ H ]$ , and $( s , a ) \in \mathcal { S } \times \mathcal { A }$ . Conditional on the history before each visit to $( s , a )$ at step $h ,$ the observed reward has mean $r _ { i , h } ( s , a )$ . At a fixed step $h ,$ the pair $( s , a )$ can be visited at most once in each episode. Therefore, we have $N _ { h } ^ { t } ( s , a ) \le T$ Applying the Azuma-Hoefding inequality for each possible positive number of visits and taking a union bound over these possibilities, all $i \in [ m ]$ , all $h \in [ H ]$ , and all $( s , a ) \in \mathcal { S } \times \mathcal { A }$ yields, with probability at least $1 - \delta$ 2

$$
\left| \widehat { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) \right| \leq \sqrt { \frac { 2 \ln \bigl ( 8 m | \mathscr { S } | | \boldsymbol { A } | \boldsymbol { H } T / \delta \bigr ) } { N _ { h } ^ { t } ( s , a ) } }
$$

simultaneously for every $t \in [ T ]$ such that $N _ { h } ^ { t } ( s , a ) \geq 1$ . Since both terms on the left belong to [0, 1], clipping the right-hand side at one preserves the inequality. When $N _ { h } ^ { t } ( s , a ) = 0$ , the inequality follows from $\sigma _ { h } ^ { t } ( s , a ) = 1$ . Therefore, the claimed bound holds for every $i \in [ m ]$ 2 $h \in [ H ] , t \in [ T ]$ , and $( s , a ) \in \mathcal S \times \mathcal A$ . This completes the proof.

Lemma 27 For any transition kernel $P ^ { \prime }$ and any policy $\pi \in \Pi$ , we have

$$
\big \| \theta ^ { P ^ { \prime } , \pi } - \theta ^ { P , \pi } \big \| _ { 1 } \leq H \sum _ { h = 1 } ^ { H } \sum _ { s , a } \theta _ { h } ^ { P , \pi } ( s , a ) \big \| P _ { h } ^ { \prime } ( \cdot | s , a ) - P _ { h } ( \cdot | s , a ) \big \| _ { 1 } .
$$

Proof Fix $h \in [ H ]$ . By Definition 1, we have $\theta _ { h } ^ { P ^ { \prime } , \pi } ( s , a , s ^ { \prime } ) = \theta _ { h } ^ { P ^ { \prime } , \pi } ( s , a ) P _ { h } ^ { \prime } ( s ^ { \prime } | s , a )$ and $\theta _ { h } ^ { P , \pi } ( s , a , s ^ { \prime } ) = \theta _ { h } ^ { P , \pi } ( s , a ) P _ { h } ( s ^ { \prime } | s , a )$ . The triangle inequality therefore gives

$$
\begin{array} { r l } { \displaystyle \| \theta _ { h } ^ { P ^ { \prime } , \pi } - \theta _ { h } ^ { P , \pi } \| _ { 1 } \leq \sum _ { s , a , s ^ { \prime } } | \theta _ { h } ^ { P ^ { \prime } , \pi } ( s , a ) - \theta _ { h } ^ { P , \pi } ( s , a ) \big | P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) } & { } \\ { + \displaystyle \sum _ { s , a , s ^ { \prime } } \theta _ { h } ^ { P , \pi } ( s , a ) \big | P _ { h } ^ { \prime } ( s ^ { \prime } | s , a ) - P _ { h } ( s ^ { \prime } | s , a ) } & { } \\ { = \displaystyle \sum _ { s , a } \big | \theta _ { h } ^ { P ^ { \prime } , \pi } ( s , a ) - \theta _ { h } ^ { P , \pi } ( s , a ) \big | } & { } \\ { + \displaystyle \sum _ { s , a } \theta _ { h } ^ { P , \pi } ( s , a ) \big | \big | P _ { h } ^ { \prime } ( \cdot | s , a ) - P _ { h } ( \cdot | s , a ) \big | \big | _ { 1 } , } \end{array}\tag{20}
$$

where the left-hand side is the $L _ { 1 }$ distance between the joint distributions of $( s _ { h } , a _ { h } , s _ { h + 1 } )$ For $h \leq H - 1$ , summing over the state and action at step h gives

$$
\theta _ { h + 1 } ^ { P ^ { \prime } , \pi } ( s ^ { \prime } , a ^ { \prime } ) = \pi _ { h + 1 } ( a ^ { \prime } | s ^ { \prime } ) \sum _ { s , a } \theta _ { h } ^ { P ^ { \prime } , \pi } ( s , a , s ^ { \prime } )
$$

for every $( s ^ { \prime } , a ^ { \prime } ) \in \mathcal { S } \times \mathcal { A }$ , and the same identity holds for P. Since $\begin{array} { r } { \sum _ { a ^ { \prime } \in \mathcal { A } } \pi _ { h + 1 } ( a ^ { \prime } | s ^ { \prime } ) = 1 } \end{array}$ , we obtain $\begin{array} { r } { \sum _ { s ^ { \prime } , a ^ { \prime } } | \theta _ { h + 1 } ^ { P ^ { \prime } , \pi } ( s ^ { \prime } , a ^ { \prime } ) - \theta _ { h + 1 } ^ { P , \pi } ( s ^ { \prime } , a ^ { \prime } ) | \leq \| \theta _ { h } ^ { P ^ { \prime } , \pi } - \theta _ { h } ^ { P , \pi } \| _ { 1 } } \end{array}$ . Moreover, we have $\theta _ { 1 } ^ { P ^ { \prime } , \pi } ( s , a ) =$ $\theta _ { 1 } ^ { P , \pi } ( s , a ) = \rho ( s ) \pi _ { 1 } ( a | s )$ . Combining these facts with (20), induction on h gives $\Vert \theta _ { h } ^ { P ^ { \prime } , \pi } -$ $\begin{array} { r } { \theta _ { h } ^ { P , \pi } \| _ { 1 } \leq \sum _ { k = 1 } ^ { h } \sum _ { s , a } \theta _ { k } ^ { P , \pi } ( s , a ) \| P _ { k } ^ { \prime } ( \cdot | s , a ) - P _ { k } ( \cdot | s , a ) \| _ { 1 } } \end{array}$ <sub>1</sub> for every $h \in [ H ]$ . Bounding the right-hand side by the corresponding sum over all $k \in [ H ]$ and summing over $h \in [ H ]$ proves the lemma.

Lemma 28 With probability at least $1 - \delta$ , for any sequence $P ^ { t + 1 } \in \mathcal { P } ^ { t }$ and the policies generated by Algorithm 1, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \big \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \big \| _ { 1 } \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) .
$$

Consequently, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \bigl \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \bigr \| _ { 1 } \leq \widetilde { \mathcal { O } } \Bigl ( H ^ { 2 } \vert \mathcal { S } \vert ^ { 3 / 2 } \sqrt { \vert \mathcal { A } \vert / T } \Bigr ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .
$$

Proof We work on the transition confidence event on which $P \in \mathcal { P } ^ { t }$ for all $t \in [ T - 1 ]$ Since both P and $P ^ { t + 1 }$ belong to $\mathcal { P } ^ { t }$ , by the definition of $\mathcal { P } ^ { t }$ in (8), for every $h \in [ H ]$ and $( s , a ) \in S \times \mathcal { A }$ , we have

$$
\big \| P _ { h } ^ { t + 1 } ( \cdot | s , a ) - P _ { h } ( \cdot | s , a ) \big \| _ { 1 } \leq \sum _ { s ^ { \prime } \in \mathcal { S } } \big | P _ { h } ^ { t + 1 } ( s ^ { \prime } | s , a ) - \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) \big | + \sum _ { s ^ { \prime } \in \mathcal { S } } \big | \widehat { P } _ { h } ^ { t } ( s ^ { \prime } | s , a ) - P _ { h } ( s ^ { \prime } | s , a ) \big |
$$

$$
\leq 2 | S | \epsilon _ { h } ^ { t } ( s , a ) \leq 2 | S | \sqrt { \frac { 2 \ln ( 2 m T H | S | ^ { 2 } | A | / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } .\tag{21}
$$

Lemma 27 with $P ^ { \prime } = P ^ { t + 1 }$ and $\pi = \widetilde { \pi } ^ { t + 1 }$ gives

$$
\big \lVert \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \big \rVert _ { 1 } \leq H \sum _ { h = 1 } ^ { H } \sum _ { s , a } \theta _ { h } ^ { P , \widetilde { \pi } ^ { t + 1 } } ( s , a ) \big \rVert P _ { h } ^ { t + 1 } ( \cdot | s , a ) - P _ { h } ( \cdot | s , a ) \big \rVert _ { 1 } .\tag{22}
$$

Applying (22) with weight $( 1 - \psi ^ { t } )$ and using $\theta ^ { P , \pi ^ { t + 1 } } = ( 1 - \psi ^ { t } ) \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } + \psi ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ , we have $( 1 - \psi ^ { t } ) \theta _ { h } ^ { P , \widetilde { \pi } ^ { t + 1 } } ( s , a ) \le \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a )$ . Therefore, by (21) and (22), we obtain

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \big \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \big \| _ { 1 } } \quad } & { } \\ & { \leq 2 H | S | \sqrt { 2 \ln ( 2 m T H | S | ^ { 2 } | A | / \delta ) } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { s , a } \frac { \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } . } \end{array}\tag{23}
$$

Let $\mathcal { F } _ { t }$ denote the history generated by all main and auxiliary trajectories up to the end of episode t. Since $\pi ^ { t + 1 }$ and $N _ { h } ^ { t }$ are $\mathcal { F } _ { t } .$ -measurable, the main trajectory satisfies $\mathbb { E } [ \mathbf { 1 } \{ ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) = ( s , a ) \} | \mathcal { F } _ { t } ] = \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a )$ . Therefore, $\begin{array} { r } { Y ^ { t } : = \sum _ { h = 1 } ^ { H } \sum _ { s , a } \frac { \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } - } \end{array}$ $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \frac { 1 } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \} } } } \end{array}$ is $\mathcal { F } _ { t + }$ <sub>1</sub>-measurable and satisfies $\mathbb { E } [ Y ^ { t } | \mathcal { F } _ { t } ] = 0$ . Moreover, we have $| \dot { Y } ^ { t } | \le H$ . By the Azuma-Hoefding inequality, with probability at least $1 - \delta .$ , we have

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { s , a } \frac { \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \leq \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \frac { 1 } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \} } } + H \sqrt { 2 T \ln ( 1 / \delta ) } .\tag{24}
$$

Fix $h \in [ H ]$ and $( s , a ) \in \mathcal { S } \times \mathcal { A }$ . Equation (3) gives $\begin{array} { r } { N _ { h } ^ { t } ( s , a ) = \sum _ { k = 1 } ^ { t } { \bf 1 } \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} } \end{array}$ Therefore, we have

$$
\sum _ { t = 1 } ^ { T - 1 } \frac { \mathbf 1 \{ ( \boldsymbol { s } _ { h } ^ { t + 1 } , \boldsymbol { a } _ { h } ^ { t + 1 } ) = ( \boldsymbol { s } , \boldsymbol { a } ) \} } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( \boldsymbol { s } , \boldsymbol { a } ) \} } } \leq 2 \sqrt { N _ { h } ^ { T } ( \boldsymbol { s } , \boldsymbol { a } ) } .
$$

Summing over $( s , a )$ and applying the Cauchy-Schwarz inequality gives

$$
\sum _ { s , a } \sum _ { t = 1 } ^ { T - 1 } \frac { \mathbf { 1 } \{ ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) = ( s , a ) \} } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \leq 2 \sqrt { | S | | A | \sum _ { s , a } N _ { h } ^ { T } ( s , a ) } = 2 \sqrt { | S | | A | T } .\tag{25}
$$

Applying (25) to the main trajectories in (24) and summing over $h \in [ H ]$ , we obtain

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { s , a } \frac { \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } \leq \mathcal { O } \Big ( H \sqrt { | \mathcal { S } | | \mathcal { A } | T } + H \sqrt { T \ln ( 1 / \delta ) } \Big ) .\tag{26}
$$

Using (26) in (23), we obtain $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \| _ { 1 } \leq \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } ) } \end{array}$ For the second claim, since both occupancy measures have total mass $H ,$ we have $\| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \| _ { 1 } \leq 2 H$ . Therefore, we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \lVert \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \rVert _ { 1 } \leq \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \lVert \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \rVert _ { 1 } + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } } \\ & { \qquad \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } . } \end{array}
$$

This completes the proof.

Lemma 29 Let $\theta \in \Theta _ { \mathrm { s a f e } }$ be any safe occupancy measure, and let $\widetilde { \theta } ^ { t + 1 }$ be the occupancy measure generated by SafeProj in Algorithm 1. On the event $\mathcal { E } _ { R , C , T } ( \delta )$ , with $\eta = T ^ { - 1 / 2 }$ 2 we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta - \widetilde { \theta } ^ { t + 1 } \right. \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | { \cal A } | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .
$$

Proof On the event $\mathcal { E } _ { R , C , T } ( \delta )$ , we have $P \in \mathcal { P } ^ { t }$ and ${ \widehat { C } } ^ { t } - \Xi ^ { t } \leq C$ . Since $\theta \in \Theta _ { \mathrm { s a f e } }$ , it follows that $\boldsymbol { \theta } \in \Theta ( \mathcal { P } ^ { t } )$ and $( \widehat { C } ^ { t } - \Xi ^ { t } ) ^ { \top } \theta \leq C ^ { \top } \theta \leq \tau$ . Thus, θ is feasible for the projection at every episode t. By Lemma 24, the feasible occupancy measure set is convex. Therefore, the exponential update $\widehat { \theta } _ { h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { t } ( s , a , s ^ { \prime } ) \mathrm { \tilde { e x p } } \{ \eta \bar { R } _ { h } ^ { t } ( s , a ) \}$ followed by the KL projection gives, for every feasible $\theta _ { ; }$

$$
\langle \bar { R } ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle \leq \frac { D _ { \mathrm { K L } } ( \theta \| \widetilde { \theta } ^ { t } ) - D _ { \mathrm { K L } } ( \theta \| \widetilde { \theta } ^ { t + 1 } ) } { \eta } .\tag{27}
$$

Summing (27) over $t = 1 , \dots , T - 1$ cancels the intermediate KL terms and gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \langle \bar { R } ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle \leq \frac { D _ { \mathrm { K L } } ( \theta \| \widetilde { \theta } ^ { 1 } ) } { \eta T } .\tag{28}
$$

Since the initial occupancy measure is set to $\begin{array} { r } { \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) = \frac { 1 } { | S | ^ { 2 } | A | } } \end{array}$ , and every feasible occupancy measure satisfies $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \sum _ { s , a , s ^ { \prime } } \theta _ { h } ( s , a , s ^ { \prime } ) = H } \end{array}$ , we have $D _ { \mathrm { K L } } ( \theta \| \widetilde { \theta } ^ { 1 } ) \leq H \log ( | S | ^ { 2 } | A | )$

We next express the preceding inequality using the true scalarized reward $R _ { h } ^ { t } ( s , a ) : =$ $\Sigma _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i , h } ( s , a )$ . By Lemma 26 and (9), on $\mathcal { E } _ { R , C , T } ( \delta )$ , for all $i , h , s , a , t .$ we have $0 \le$ $\bar { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) \leq 2 \sigma _ { h } ^ { t } ( s , a )$ . Consequently, for all $h , s , a , t$ , we have $0 \leq \bar { R } _ { h } ^ { t } ( s , a ) -$ $\begin{array} { r } { R _ { h } ^ { t } ( s , a ) \leq 2 \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s , a ) } \end{array}$ . Therefore, for every $t ,$ we have

$$
\begin{array} { r l r } {  { \langle R ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle = \langle \bar { R } ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle - \langle \bar { R } ^ { t } - R ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle } } \\ & { } & { \leq \langle \bar { R } ^ { t } , \theta - \widetilde { \theta } ^ { t + 1 } \rangle + 2  \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 }  . } \end{array}\tag{29}
$$

The last inequality uses $\bar { R } ^ { t } - R ^ { t } \geq 0$ and $\theta \ge 0$

We now bound the second term in (29). By construction, we have $\widetilde { \pi } ^ { t + 1 } = \pi ^ { \widetilde { \theta } ^ { t + 1 } }$ and $\widetilde { \theta } ^ { t + 1 } = \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } }$ for some $P ^ { t + 1 } \in \mathcal { P } ^ { t }$ . Moreover, at the beginning of episode $t + 1$ , SafeProj selects $\pi ^ { \dagger }$ with probability $\psi ^ { t }$ and $\widetilde { \pi } ^ { t + 1 }$ with probability $1 - \psi ^ { t }$ . Therefore, we have $\theta ^ { P , \pi ^ { t + 1 } } = \begin{array} { r l } \end{array}$ $( 1 - \psi ^ { t } ) \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \dot { + } \psi ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ . Since $\begin{array} { r } { 0 \leq \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s , a ) \leq 1 } \end{array}$ and $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \sum _ { s , a , s ^ { \prime } } \widetilde { \theta } _ { h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = } \end{array}$ $H _ { ; }$ , we have

$$
\begin{array} { r l } & { \Bigg \langle \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \Bigg \rangle } \\ & { \quad = ( 1 - \psi ^ { t } ) \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \right. + \psi ^ { t } \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \right. } \\ & { \quad \leq ( 1 - \psi ^ { t } ) \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \right. + ( 1 - \psi ^ { t } ) \left\| \theta ^ { P ^ { t + 1 } , \overline { { \pi } } ^ { t + 1 } - \theta ^ { P , \overline { { \pi } } ^ { t + 1 } } | } _ { 1 } + \psi ^ { t } H } \\ & \right\{ \quad \leq \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \right. + ( 1 - \psi ^ { t } ) \left\| \theta ^ { P ^ { t + 1 } , \overline { { \pi } } ^ { t + 1 } - \theta ^ { P , \overline { { \pi } } ^ { t + 1 } } } \right\| _ { 1 } + \psi ^ { t } H . } \end{array}\tag{30}
$$

The last inequality follows from $( 1 - \psi ^ { t } ) \theta _ { h } ^ { P , \widetilde { \pi } ^ { t + 1 } } ( s , a ) \leq \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) , \forall h , s , a$ . The first term in (30) is controlled by the visitation bound for the executed main trajectories. Since $\pi ^ { t + 1 }$ and $\{ \sigma _ { h } ^ { t } \} _ { h }$ are $\mathcal { F } _ { t }$ -measurable, we have $\mathbb { E } [ \mathbf { 1 } \{ ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) = ( s , a ) \} | \mathcal { F } _ { t } ] = \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a )$ . We define $\begin{array} { r } { Y ^ { t } : = \sum _ { h = 1 } ^ { H } \sum _ { s , a } ( \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s , a ) ) \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a ) - \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) } \end{array}$ Then we have $\mathbb { E } [ Y ^ { t } | \mathcal { F } _ { t } ] = 0$ . Moreover, since $w ^ { t } , \lambda \in \Delta _ { m }$ and $0 \leq \sigma _ { h } ^ { t } ( s , a ) \leq 1$ , we obtain $\begin{array} { r } { 0 \leq \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s , a ) \leq \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \leq 1 } \end{array}$ . Thus, we have $| Y ^ { t } | \leq H$ . By Hoefding’s inequality for bounded martingale diferences, with probability at least $1 - \delta$ , we obtain

$$
\sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \right. \leq \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) + H \sqrt { 2 T \ln ( 1 / \delta ) } .\tag{31}
$$

It remains to bound the realized sum of reward bonuses. By the definition of $\sigma _ { h } ^ { t }$ in (6), we have

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq \sqrt { 2 \ln ( 8 m | \mathcal { S } | | A | H T / \delta ) } \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \frac { 1 } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \} } } .\tag{32}
$$

Summing (25) over $h \in [ H ]$ for the main trajectories and using (32), we obtain

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq 2 H \sqrt { 2 | S | | A | T \ln ( 8 m | S | | A | H T / \delta ) } .\tag{33}
$$

Using (31) together with (33), we obtain $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \langle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \rangle \leq \widetilde { \mathcal { O } } ( H \sqrt { \frac { | \mathcal { S } | | A | } { T } } + } \end{array}$ $H \sqrt { \frac { 1 } { T } } )$

By Lemma 28, the second term is controlled as $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \| _ { 1 } \leq } \end{array}$ $\widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$ . Therefore, we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \right. \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } \Big ) + \frac { H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .
$$

Combining this bound with (29) and (28), and taking $\eta = T ^ { - 1 / 2 }$ , gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta - \widetilde { \theta } ^ { t + 1 } \right. \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | { \cal A } | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .
$$

This completes the proof.

Lemma 30 Let $\widetilde { \theta } _ { i } ^ { t + 1 }$ be the projected occupancy measure generated by Algorithm 1, and let $\theta _ { i } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ be a single-objective optimal occupancy measure for objective i. On the event $\mathcal { E } _ { R , C , T } ( \delta )$ , with $\eta = T ^ { - 1 / 2 }$ , for every $i \in [ m ]$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Big \langle r _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } ^ { t + 1 } \Big \rangle \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .
$$

Moreover, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Big \langle r _ { i } , \theta _ { i } ^ { * } - \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } \Big \rangle \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 4 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .
$$

Proof Fix $i \in [ m ]$ . For every $t \in [ T - 1 ]$ $\theta _ { i } ^ { * }$ is feasible for the auxiliary projection on $\mathcal { E } _ { R , C , T } ( \delta )$ . Since $\theta _ { i } ^ { * } \in \Theta _ { \mathrm { s a f e } } , P \in \mathcal { P } ^ { t }$ , and ${ \widehat { C } } ^ { t } - \Xi ^ { t } \leq C$ , we have $( \widehat { C } ^ { t } - \Xi ^ { t } ) ^ { \top } \theta _ { i } ^ { * } \le C ^ { \top } \theta _ { i } ^ { * } \le \tau$ Applying the KL projection argument in the proof of Lemma 29 with $\bar { R } ^ { t }$ replaced by $\bar { r } _ { i } ^ { t }$ , and using the uniform initialization, gives $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \langle \bar { r } _ { i } ^ { t } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } ^ { t + 1 } \rangle \leq \frac { D _ { \mathrm { K L } } ( \theta _ { i } ^ { * } \| \widetilde { \theta } _ { i } ^ { 1 } ) } { \eta T } \leq \frac { H \log ( | S | ^ { 2 } | A | ) } { \eta T } } \end{array}$ With $\eta = T ^ { - 1 / 2 }$ , the right-hand side is $\widetilde { \mathcal { O } } ( H / \sqrt { T } )$

We next express the preceding inequality using the true reward $r _ { i }$ . By Lemma 26 and (9), on $\mathcal { E } _ { R , C , T } ( \delta )$ , we have $0 \leq \bar { r } _ { i , h } ^ { t } ( s , a ) - r _ { i , h } ( s , a ) \leq 2 \sigma _ { h } ^ { t } ( s , a ) , \forall h \in [ H ] , s \in \mathcal { S } , a \in$ $\mathcal { A } , t \in [ T - 1 ]$ . Therefore, we have

$$
\left. r _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } ^ { t + 1 } \right. \leq \left. \bar { r } _ { i } ^ { t } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } ^ { t + 1 } \right. + 2 \left. \sigma ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \right. .\tag{34}
$$

It remains to control the reward bonus term. By construction, we have $\widetilde \nu _ { i } ^ { t + 1 } = \pi ^ { \widetilde { \theta } _ { i } ^ { t + 1 } }$ and $\widetilde { \theta } _ { i } ^ { t + 1 } = \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } }$ for some $P _ { i } ^ { t + 1 } \in \mathcal { P } ^ { t }$ . Moreover, in the auxiliary update, SafeProj selects $\pi ^ { \dagger }$ with probability $\psi _ { i } ^ { t }$ and $\widetilde \nu _ { i } ^ { t + 1 }$ with probability $1 - \psi _ { i } ^ { t }$ . Therefore, we have $\theta ^ { P , \nu _ { i } ^ { t + 1 } } =$ $( 1 - \psi _ { i } ^ { t } ) \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } + \psi _ { i } ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ . Since $0 \leq \sigma _ { h } ^ { t } ( s , a ) \leq 1$ and $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \sum _ { s , a , s ^ { \prime } } \widetilde { \theta } _ { i , h } ^ { t + 1 } ( s , a , s ^ { \prime } ) = H } \end{array}$ , the same decomposition as in (30) gives

$$
\left. \sigma ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \right. \leq \left. \sigma ^ { t } , \theta ^ { P , \nu _ { i } ^ { t + 1 } } \right. + \left( 1 - \psi _ { i } ^ { t } \right) \left. \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \right. _ { 1 } + H \psi _ { i } ^ { t } .\tag{35}
$$

The first term in (35) is bounded by the same visitation argument as in Lemma 29, applied to the trajectories generated by $\{ \nu _ { i } ^ { t } \} _ { t = 1 } ^ { T }$ . Thus, we have $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \langle \sigma ^ { t } , \theta ^ { P , \nu _ { i } ^ { t + 1 } } \rangle \leq \widetilde { \mathcal { O } } ( H \sqrt { \frac { | \mathcal { S } | | A | } { T } } + } \end{array}$ $H \sqrt { \frac { 1 } { T } } )$ . The second term in (35) is bounded by applying the argument of Lemma 28 to the same auxiliary trajectories. Specifically, we replace $( \widetilde { \pi } ^ { t + 1 } , \pi ^ { t + 1 } , \psi ^ { t } )$ by $( \widetilde \nu _ { i } ^ { t + 1 } , \nu _ { i } ^ { t + 1 } , \psi _ { i } ^ { t } )$ and obtain $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi _ { i } ^ { t } ) \| \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \| _ { 1 } \leq \widetilde { \mathcal { O } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } } \end{array}$ . Combining these bounds, we have $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \langle \sigma ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \rangle \leq \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) + \frac { H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } } \end{array}$ . Taking the average of (34) over $t = 1 , \dots , T - 1$ and using the preceding two bounds, we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Big \langle r _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } ^ { t + 1 } \Big \rangle \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .
$$

By construction, we have $\widetilde { \theta } _ { i } ^ { t + 1 } = \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } }$ for some $P _ { i } ^ { t + 1 } \in \mathcal { P } ^ { t }$ . Therefore, since $0 \leq$ $r _ { i , h } ( s , a ) \leq 1$ , we have $\langle r _ { i } , \widetilde { \theta } _ { i } ^ { t + 1 } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \rangle \leq \| \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \| _ { 1 }$ . Applying the unweighted form of Lemma 28 to the auxiliary policy sequence yields $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { \ddot { T } - 1 } \| \theta ^ { \dot { P } _ { i } ^ { t + 1 } , \tilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { \dot { P } , \tilde { \nu } _ { i } ^ { t + 1 } } \| _ { 1 } \leq } \end{array}$ $\begin{array} { r } { \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } } \end{array}$ . Therefore, we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Big \langle r _ { i } , \theta _ { i } ^ { * } - \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } \Big \rangle \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 4 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .
$$

This completes the proof.

Lemma 31 With probability at least $1 - \delta$ , for every objective $i \in [ m ]$ and any $P _ { i } ^ { t + 1 } \in \mathcal { P } ^ { t }$ associated with the auxiliary policies generated by Algorithm 1, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi _ { i } ^ { t } ) \big \| \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \big \| _ { 1 } \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) .
$$

Consequently, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \bigl \lVert \theta ^ { P _ { i } ^ { t + 1 } , \tilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } \bigr \rVert _ { 1 } \le \widetilde { \mathcal { O } } \Bigl ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } \Bigr ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .
$$

Proof The proof repeats the argument of Lemma 28 using the auxiliary trajectory sequence for objective i. We replace $( \widetilde { \pi } ^ { t + 1 } , \pi ^ { t + 1 } , \psi ^ { t } )$ by $( \widetilde { \nu } _ { i } ^ { t + 1 } , \nu _ { i } ^ { t + 1 } , \psi _ { i } ^ { t } )$

Lemma 32 Let $( \mathcal { P } , \widehat { C } , \Xi )$ be the confidence set, cost estimates, and cost bonuses used in a call of SafeProj. Let $\widetilde { \theta } \in \Theta ( \mathcal { P } )$ satisfy $( \widehat { C } - \Xi ) ^ { \top } \widetilde { \theta } \leq \tau$ , and let $\widetilde { \pi } : = \pi ^ { \widetilde { \theta } }$ . Let ψ be given by (12) with $( \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } , \widetilde { \pi } ^ { t + 1 } )$ replaced by $( \mathcal { P } , \boldsymbol { \widehat { C } } , \boldsymbol { \Xi } , \boldsymbol { \widetilde { \pi } } )$ . Then we have

$$
\psi \leq ( 1 - \psi ) \operatorname* { m a x } _ { j \in [ n ] } \frac { 1 } { \tau _ { j } - \gamma _ { j } } \bigg [ 3 \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \big | \big | \theta ^ { P ^ { \prime } , \widetilde { \pi } } - \theta ^ { P , \widetilde { \pi } } \big | \big | _ { 1 } + 2 \operatorname* { m a x } _ { j ^ { \prime } \in [ n ] } \sum _ { h = 1 } ^ { H } \xi _ { j ^ { \prime } , h } ^ { \top } \theta _ { h } ^ { P , \widetilde { \pi } } \bigg ] .
$$

Proof If $\psi = 0$ , the claim holds trivially. Otherwise, let $j \in [ n ]$ attain the maximum in (12). Since $\psi > 0$ , the minimum in the numerator of the corresponding ratio exceeds $\tau _ { j }$ and (12) gives

$$
\psi = \frac { \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } } , d \right\} - \tau _ { j } } { \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } } , d \right\} - \gamma _ { j } } .
$$

Rearranging this identity and bounding the minimum by its first argument, we obtain

$$
\psi ( \tau _ { j } - \gamma _ { j } ) \leq ( 1 - \psi ) \biggl ( \underset { P ^ { \prime } \in \mathcal { P } } { \operatorname* { m a x } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } } - \tau _ { j } \biggr ) .\tag{36}
$$

We next bound the right-hand side of (36). Since $\widehat { c } _ { j , h } + \xi _ { j , h } \in [ 0 , 2 ] ^ { | S | | A | }$ for every $h \in [ H ]$ 2 we have

$$
\operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \sum _ { h = 1 } ^ { H } ( \widehat c _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde \pi } \leq \sum _ { h = 1 } ^ { H } ( \widehat c _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P , \widetilde \pi } + 2 \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \left\| \theta ^ { P ^ { \prime } , \widetilde \pi } - \theta ^ { P , \widetilde \pi } \right\| _ { 1 } .
$$

As in the proof of Lemma 24, $\widetilde { \theta }$ is the occupancy measure of $\widetilde { \pi }$ under some transition kernel in $\mathcal { P }$ . Since $\widehat { c } _ { j , h } - \xi _ { j , h } \in [ - 1 , 1 ] ^ { | S | | A | }$ for every $h \in [ H ]$ , the constraint on $\widetilde { \theta }$ gives

$$
\tau _ { j } \geq \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } - \xi _ { j , h } ) ^ { \top } \widetilde { \theta } _ { h } \geq \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } - \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P , \widetilde { \pi } } - \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \big \| \theta ^ { P ^ { \prime } , \widetilde { \pi } } - \theta ^ { P , \widetilde { \pi } } \big \| _ { 1 } .
$$

Subtracting the last inequality from the preceding one, we obtain

$$
\operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } + \xi _ { j , h } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } } - \tau _ { j } \leq 3 \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \left\| \theta ^ { P ^ { \prime } , \widetilde { \pi } } - \theta ^ { P , \widetilde { \pi } } \right\| _ { 1 } + 2 \sum _ { h = 1 } ^ { H } \xi _ { j , h } ^ { \top } \theta _ { h } ^ { P , \widetilde { \pi } } .
$$

Substituting this bound into (36) and dividing both sides by $\tau _ { j } - \gamma _ { j } > 0$ proves the lemma.

Lemma 33 Set $\alpha _ { t } = 1 / t ^ { 2 }$ and $\beta _ { t } = 1 / ( \mu t ) ~ f o r ~ t \in [ T - 1 ]$ . Then the updating rule of w in Algorithm 1 guarantees that

$$
\begin{array} { r l } & { \displaystyle \underset { w \in \Delta _ { m } } { \operatorname* { m a x } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left[ \langle w - w ^ { t } , \lambda \odot ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } ) \rangle - \mu \langle w , \log w \rangle + \mu \langle w ^ { t } , \log w ^ { t } \rangle \right] } \\ & { \quad \le \widetilde { \mathcal { O } } \left( \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } \right) . } \end{array}
$$

Proof The mirror-ascent update that produces $w ^ { t + 1 }$ in Line 13 of Algorithm 1 is equivalent to solving the following minimization problem

$$
\begin{array} { r l } { \underset { \pmb { w } \in \Delta _ { m } } { \mathrm { m i n i m i z e } } } & { { } - \beta _ { t } \Big \langle \pmb { w } , \pmb { \lambda } \odot \big ( \widetilde { \pmb { V } } ^ { t + 1 } + \pmb { \iota } - \pmb { V } ^ { t + 1 } \big ) - \mu \log \widetilde { \pmb { w } } ^ { t } \Big \rangle + D _ { \mathrm { K L } } \big ( \pmb { w } \| \widetilde { \pmb { w } } ^ { t } \big ) . } \end{array}
$$

Let $w ^ { t + 1 }$ be the solution to the above optimization problem. Since $w _ { i } ^ { 1 } = 1 / m$ , the iterate $w ^ { t + 1 }$ remains in the relative interior of $\Delta _ { m }$ . Thus, applying Lemma 45 gives

$$
\begin{array} { r } { \beta _ { t } \langle w - w ^ { t + 1 } , \lambda \odot ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } ) - \mu \log { \tilde { w } ^ { t } } \rangle \le D _ { \mathrm { K L } } ( w \| \tilde { w } ^ { t } ) - D _ { \mathrm { K L } } ( w \| w ^ { t + 1 } ) - D _ { \mathrm { K L } } ( w ^ { t + 1 } \| \tilde { w } ^ { t } ) , } \end{array}
$$

where w is arbitrary. Rearranging the terms leads to

$$
\begin{array} { r l } & { \beta _ { t } \big \langle { \pmb w } - { \pmb w } ^ { t } , \lambda \odot \big ( \widetilde { \pmb V } ^ { t + 1 } + { \pmb \nu } - { \pmb V } ^ { t + 1 } \big ) \big \rangle - \beta _ { t } \mu \big \langle { \pmb w } , \log { \pmb w } \big \rangle + \beta _ { t } \mu \big \langle { \pmb w } ^ { t } , \log { \pmb w } ^ { t } \big \rangle } \\ & { \qquad \le D _ { \mathrm { K L } } \big ( { \pmb w } \| \widetilde { \pmb w } ^ { t } \big ) - D _ { \mathrm { K L } } \big ( { \pmb w } \| { \pmb w } ^ { t + 1 } \big ) - D _ { \mathrm { K L } } \big ( { \pmb w } ^ { t + 1 } \| \widetilde { \pmb w } ^ { t } \big ) } \\ & { \qquad + \beta _ { t } \big \langle { \pmb w } ^ { t + 1 } - { \pmb w } ^ { t } , \lambda \odot \big ( \widetilde { \pmb V } ^ { t + 1 } + { \pmb \nu } - { \pmb V } ^ { t + 1 } \big ) - \mu \log \widetilde { \pmb w } ^ { t } \big \rangle } \\ & { \qquad - \beta _ { t } \mu \big \langle { \pmb w } , \log { \pmb w } \big \rangle + \beta _ { t } \mu \big \langle { \pmb w } ^ { t } , \log { \pmb w } ^ { t } \big \rangle + \beta _ { t } \mu \big \langle { \pmb w } - { \pmb w } ^ { t } , \log \widetilde { \pmb w } ^ { t } \big \rangle . } \end{array}\tag{37}
$$

For the terms on the right-hand side of (37), by Lemma 46 and the definition of KL divergence, we have

$$
\begin{array} { r l } & { \beta _ { t } \langle { \pmb w } - { \pmb w } ^ { t } , \lambda \odot \big ( \widetilde { { \pmb V } } ^ { t + 1 } + \pmb \nu - { \pmb V } ^ { t + 1 } \big ) \rangle - \beta _ { t } \mu \langle { \pmb w } , \log { \pmb w } \rangle + \beta _ { t } \mu \langle { \pmb w } ^ { t } , \log { \pmb w } ^ { t } \rangle } \\ & { \qquad \le ( 1 - \beta _ { t } \mu ) D _ { \mathrm { K L } } \big ( { \pmb w } \| \widetilde { { \pmb w } } ^ { t } \big ) - D _ { \mathrm { K L } } \big ( { \pmb w } \| \widetilde { { \pmb w } } ^ { t + 1 } \big ) - D _ { \mathrm { K L } } \big ( { \pmb w } ^ { t + 1 } \| \widetilde { { \pmb w } } ^ { t } \big ) + 2 \alpha _ { t } \log m } \\ & { \qquad + \beta _ { t } \langle { \pmb w } ^ { t + 1 } - { \pmb w } ^ { t } , \lambda \odot \big ( \widetilde { { \pmb V } } ^ { t + 1 } + \pmb \nu - { \pmb V } ^ { t + 1 } \big ) - \mu \log \widetilde { { \pmb w } } ^ { t } \rangle , } \end{array}\tag{38}
$$

where $\mu \beta _ { t } = 1 / t \leq 1$ . In addition, by Pinsker’s inequality, we have $- D _ { \mathrm { K L } } \big ( \boldsymbol { w } ^ { t + 1 } \| \widetilde { \boldsymbol { w } } ^ { t } \big ) \ \leq$ $- \frac 1 2 \big \| \pmb { w } ^ { t + 1 } - \widetilde { \pmb { w } } ^ { t } \big \| _ { 1 } ^ { 2 }$ . By H¨older’s inequality, the last term in (38) is bounded as

$$
\begin{array} { r l } & { \beta _ { t } \big \langle w ^ { t + 1 } - w ^ { t } , \lambda \odot \big ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } \big ) - \mu \log \widetilde { w } ^ { t } \big \rangle } \\ & { \qquad \le \beta _ { t } \big ( \big \| w ^ { t + 1 } - \widetilde { w } ^ { t } \big \| _ { 1 } + \big \| \widetilde { w } ^ { t } - w ^ { t } \big \| _ { 1 } \big ) \big \| \boldsymbol { \lambda } \odot \big ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } \big ) - \mu \log \widetilde { w } ^ { t } \big \| _ { \infty } } \\ & { \qquad \le \frac 1 2 \big \| w ^ { t + 1 } - \widetilde { w } ^ { t } \big \| _ { 1 } ^ { 2 } + \big ( 3 H + \mu \log ( m / \alpha _ { t } ) \big ) ^ { 2 } \beta _ { t } ^ { 2 } + 2 \alpha _ { t } ^ { 2 } , } \end{array}
$$

where the last inequality uses Lemma 46 and $\| \lambda \odot ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } ) - \mu \log \widetilde { w } ^ { t } \| _ { \infty } \le$ $3 H + \mu \log ( m / \alpha _ { t } )$ with $\iota , \lambda _ { i } \leq 1$ . Substituting the above bounds into (38), we have

$$
\begin{array} { r l } & { \beta _ { t } \langle w - w ^ { t } , \lambda \odot \left( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } \right) \rangle - \beta _ { t } \mu \langle w , \log w \rangle + \beta _ { t } \mu \langle w ^ { t } , \log w ^ { t } \rangle } \\ & { \quad \le ( 1 - \beta _ { t } \mu ) D _ { \mathrm { K L } } ( w \| \widetilde { w } ^ { t } ) - D _ { \mathrm { K L } } ( w \| \widetilde { w } ^ { t + 1 } ) + 2 \alpha _ { t } \log m + ( 3 H + \mu \log ( m / \alpha _ { t } ) ) ^ { 2 } \beta _ { t } ^ { 2 } + 2 \alpha _ { t } ^ { 2 } . } \end{array}
$$

Now we set $\beta _ { t } = 1 / ( \mu t )$ and $\alpha _ { t } = 1 / t ^ { 2 }$ , and we divide both sides by $T \beta _ { t }$ . At $t = T - 1$ we retain $- D _ { \mathrm { K L } } ( \pmb { w } \Vert \pmb { w } ^ { T } )$ from (37) instead of applying Lemma 46 to it, since the last execution of Line 13 produces $\mathbf { \Delta } _ { w } \mathbf { \mathcal { T } }$ and performs no further mixing. The sum over $t = 1 , \dots , T - 1$ cancels all intermediate KL terms, leaving only − $\begin{array} { r } { \frac { \mu ( T - 1 ) } { T } \dot { D _ { \mathrm { K L } } } ( \pmb { w } \| \pmb { w } ^ { T } ) \leq 0 } \end{array}$ , which we discard. Therefore, we obtain

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \left[ \left. w - w ^ { t } , \lambda \odot ( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } ) \right. - \mu \langle w , \log w \rangle + \mu \langle w ^ { t } , \log w ^ { t } \rangle \right] } \\ & { \quad \le \displaystyle \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \left[ \frac { 2 \mu \log m } { t } + \frac { ( 3 H + 2 \mu \log ( m t ) ) ^ { 2 } } { \mu t } + \frac { 2 \mu } { t ^ { 3 } } \right] \le \frac { 1 8 H ^ { 2 } \log T } { \mu T } + \frac { 1 1 \mu \log ^ { 3 } ( m T ) } { T } . } \end{array}
$$

This completes the proof.

Lemma 34 Let $\sigma _ { h } ^ { t }$ and $\zeta _ { h } ^ { t }$ be defined as in (6) and (13), respectively. For any sequence of trajectories, let $\{ ( \boldsymbol { s } _ { h } ^ { \dot { t } } , \boldsymbol { a } _ { h } ^ { t } ) \} _ { h = 1 } ^ { \dot { H } }$ denote the trajectory collected in episode $t \in [ T ]$ , and construct the counts and bonuses from this sequence. Then we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq \mathcal { O } \bigg ( H \sqrt { T | S | | A | \ln \bigg ( \frac { m | S | | A | H T } { \delta } \bigg ) } \bigg ) , } \\ & { \displaystyle \sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \zeta _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq \mathcal { O } \bigg ( H ^ { 2 } \sqrt { T | S | ^ { 2 } | A | \ln \bigg ( \frac { m | S | | A | H T } { \delta } \bigg ) } \bigg ) . } \end{array}
$$

Proof The bound (25) applies at every step $\textit { h } \in \ [ H ]$ . By (6), we have $\sigma _ { h } ^ { t } ( s , a ) \ \leq$ $\sqrt { \frac { 2 \ln ( 8 m \vert S \vert \vert A \vert H T / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } }$ . Therefore, using (25) and summing over $h \in [ H ]$ , we obtain

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \sigma _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq \mathcal { O } \left( H \sqrt { T | \mathcal { S } | | \mathcal { A } | \ln \left( \frac { m | \mathcal { S } | | \mathcal { A } | H T } { \delta } \right) } \right) .
$$

By (13), we have $\begin{array} { r } { \zeta _ { h } ^ { t } ( s , a ) \le H \sqrt { \frac { 8 | S | \ln ( 2 m T H | S | | A | / \delta ) } { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } } } \end{array}$ . Using (25) and summing over $h \in [ H ]$ we have

$$
\sum _ { t = 1 } ^ { T - 1 } \sum _ { h = 1 } ^ { H } \zeta _ { h } ^ { t } ( s _ { h } ^ { t + 1 } , a _ { h } ^ { t + 1 } ) \leq \mathcal { O } \left( H ^ { 2 } \sqrt { T | S | ^ { 2 } | A | \ln \left( \frac { m | S | | A | H T } { \delta } \right) } \right) .
$$

This completes the proof.

## D.2 Proof of Theorem 9

Proof Let <sup>¯</sup>θ be the averaged occupancy measure returned by Algorithm 1. We use the optimal occupancy measures $\theta _ { i } ^ { * } , ~ i ~ \in ~ [ m ]$ , and $\theta _ { \lambda } ^ { * }$ introduced in Section 7.1. Since each $V _ { i }$ is linear in θ and the log-sum-exp function is convex, $\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta } )$ is convex in θ on $\Theta ( P )$ . Moreover, for any $\theta , \theta ^ { \prime } \in \Theta ( P )$ and every $i \in [ m ]$ , we have $\lambda _ { i } ( V _ { i } ^ { * } + \iota - V _ { i } ( { \boldsymbol { \theta } } ) ) ~ \leq$ $\begin{array} { r } { \lambda _ { i } ( V _ { i } ^ { * } + \iota - V _ { i } ( \theta ^ { \prime } ) ) + \operatorname* { m a x } _ { k \in [ m ] } \lambda _ { k } \big ( V _ { k } ( \theta ^ { \prime } ) - V _ { k } ( \theta ) \big ) } \end{array}$ . Taking the exponential, summing over $i \in [ m ]$ , and taking the logarithm, we obtain

$$
\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta } ) - \operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta ^ { \prime } } ) \leq \operatorname* { m a x } _ { i \in [ m ] } \lambda _ { i } \big ( V _ { i } ( \theta ^ { \prime } ) - V _ { i } ( \theta ) \big ) \leq H ,\tag{39}
$$

where the last inequality uses $\lambda _ { i } \leq 1$ and $V _ { i } ( \theta ) \in [ 0 , H ]$ . For $T = 1 , ( 3 9 )$ gives $\mathrm { S u b O p t } ( { \widehat { \pi } } ) \leq$ $H .$ , which is dominated by the stated bound. We assume $T \geq 2$ below and define $\bar { \theta } _ { 2 : T } : =$ $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \theta ^ { P , \pi ^ { t + 1 } } } \end{array}$ . Then $\begin{array} { r } { \bar { \theta } \ = \ \frac { 1 } { T } \theta ^ { P , \pi ^ { 1 } } + \frac { T - 1 } { T } \bar { \theta } _ { 2 : T } } \end{array}$ . By the convexity of $\operatorname { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta } )$ in $\theta$ and (39), we obtain

$$
\begin{array} { r l } { \mathrm { s u b O p t } ( \widehat { n } ) = \displaystyle \mathbb { S } ^ { \mathrm { T C } } \mathrm { C H } _ { 1 } ^ { n } ( \alpha ^ { \theta } ) - \mathrm { S T C H } _ { 1 } ^ { n } ( \widehat { \sigma } ^ { \theta } ) } \\ { \leq \displaystyle \frac { T } { T } + \frac { T - 1 } { T } ( \mathrm { S T C H } _ { 1 } ^ { n } ( \widehat { \sigma } ^ { \theta } \sigma ^ { \theta } ) - \mathrm { S T C H } _ { 1 } ^ { n } ( \sigma ^ { \theta } \lambda ) ) } \\ { = \displaystyle \frac { T } { T } + \frac { T - 1 } { T } \operatorname* { m a x } _ { i = \lambda _ { 0 } } \displaystyle \sum _ { i = 1 } ^ { m } \Big [ w _ { i } \lambda _ { i } ( V _ { i } ^ { \ast } + \iota - V _ { i } ( \theta _ { i } \sigma ) ) - \mu w _ { i } \log w _ { i } \Big ] } \\ { \displaystyle - \frac { 1 } { T } \sum _ { i = 1 } ^ { m } \sum _ { i = 1 } ^ { m } \Big [ w _ { i } ^ { \ast } \lambda _ { i } ( V _ { i } ( \theta ^ { D _ { i } ^ { \theta } \sigma ^ { \theta + 1 } } ) + \iota - V _ { i } ( \theta ^ { D _ { i } \sigma + 1 } ) ) - \mu w _ { i } ^ { \ast } \log w _ { i } \Big ] } \\ { + \displaystyle \frac { 1 } { T } \sum _ { i = 1 } ^ { \infty } \sum _ { i = 1 } ^ { m } \Big [ w _ { i } ^ { \ast } \lambda _ { i } ( V _ { i } ( \theta ^ { D _ { i } ^ { \theta } \sigma ^ { \theta + 1 } } ) + \iota - V _ { i } ( \theta ^ { D _ { i } \sigma + 1 } ) ) - \mu w _ { i } ^ { \ast } \log w _ { i } ^ { \prime } \Big ] } \\  - \displaystyle \frac { T - 1 } { T } \sum _ { i = 1 } ^ { m } \sum _ { i = 1 } ^ { \infty } \Big [ w _ { i } ^ { \ast } \lambda _ { i } ( V _ { i } ^ { \ast } + \iota - \end{array}\tag{40}
$$

where the last equality uses (2). We first bound the following diference.

$$
\begin{array} { r l } & { \frac { T - 1 } { T } \underset { w \in \Delta _ { m } } { \operatorname* { m a x } } \displaystyle \sum _ { i = 1 } ^ { m } \Big [ w _ { i } \lambda _ { i } \left( V _ { i } ^ { * } + \iota - V _ { i } ( \bar { \theta } _ { 2 : T } ) \right) - \mu w _ { i } \log w _ { i } \Big ] } \\ & { \quad - \frac { 1 } { T } \sum _ { k = 1 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { m } \Big [ w _ { i } ^ { * } \lambda _ { i } \left( V _ { i } ( \theta ^ { P ; P _ { i } ^ { * + 1 } } ) + \iota - V _ { i } ( \theta ^ { P ; \alpha ^ { * + 1 } } ) \right) - \mu w _ { i } ^ { * } \log w _ { i } ^ { * } \Big ] } \\ & { \le \underset { w \in \Delta _ { m } } { \operatorname* { m a x } } \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } \lambda _ { i } \left[ \frac { T - 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \Big ( V _ { i } ^ { * } - V _ { i } ( \theta ^ { P ; \alpha ^ { * + 1 } } ) \Big ) \right] } \\ & { \quad + \underset { w \in \Delta _ { m } } { \operatorname* { m a x } } \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { m } \Big ( ( w _ { i } - w _ { i } ^ { * } ) \lambda _ { i } \left( V _ { i } ( \theta ^ { P ; \alpha ^ { * + 1 } } ) + \iota - V _ { i } ( \theta ^ { P ; \alpha ^ { * + 1 } } ) \right) - \mu w _ { i } \log w _ { i } + \mu w _ { i } ^ { * } \log w _ { i } ^ { * } \Big ] . } \end{array}\tag{41}
$$

This inequality uses the definition of $\bar { \theta } _ { 2 : T }$ , the linearity of $V _ { i } ,$ and the fact that the maximum of a sum is at most the sum of the maxima. The first term is the auxiliary single-objective learning error, and the second term is the smooth Tchebychef dual optimization error. We next bound the diference between the last two terms in (40). On $\mathcal { E } _ { R , C , T } ( \delta )$ , Theorem 10 gives $\theta ^ { P , \nu _ { i } ^ { t + 1 } } \in \Theta _ { \mathrm { s a f e } }$ . Since $\theta _ { i } ^ { * } \in$ arg max<sub>θ∈</sub> $\Theta _ { \mathrm { s a f e } } V _ { i } ( \theta )$ , we have $V _ { i } ( \theta ^ { P , \nu _ { i } ^ { t + 1 } } ) \leq V _ { i } ( \theta _ { i } ^ { * } ) = V _ { i } ^ { * }$ Moreover, the construction of $\nu _ { i } ^ { t + 1 }$ gives $\theta ^ { P , \nu _ { i } ^ { t + 1 } } = ( 1 - \psi _ { i } ^ { t } ) \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } + \psi _ { i } ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ . Since $0 ~ \leq$ $V _ { i } ( \theta ) \leq H$ , it follows that $V _ { i } ( \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ) \le V _ { i } ^ { * } + H \psi _ { i } ^ { t }$ . Therefore, for any $w \in \Delta _ { m }$ , we obtain

$$
\begin{array} { r l } &  \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { j - 1 } \displaystyle \sum _ { u ^ { \mathrm { i n } } \leq u ^ { \mathrm { i n } } , 1 } [ w _ { \mathrm { i } } ^ { ( \lambda _ { \mathrm { i n } } ) } ( V _ { \mathrm { i } } ( \theta ^ { \mu ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha } } } } } ) + \lambda - V _ { \mathrm { i } } ( \theta ^ { \mu ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } } } } ) } ) - \mu w _ { \mathrm { i } } ^ { \lambda } \log w _ { \mathrm { i } } ^ { \lambda } ]  } \\ & { - \displaystyle \frac { 1 } { T } \displaystyle \sum _ { u ^ { \mathrm { i n } } \leq u ^ { \mathrm { i n } } , \alpha _ { u ^ { \mathrm { i n } } } } \displaystyle \sum _ { u ^ { \mathrm { i n } } \leq 1 } [ w _ { \mathrm { i } } \lambda _ { \mathrm { i n } } ( V _ { \mathrm { i } } ^ { \alpha } + \mu - V _ { \mathrm { i } } ( \theta ^ { \beta _ { \mathrm { i n } } ^ { \alpha } } ) ) - \mu w _ { \mathrm { i } } \log w _ { \mathrm { i } } ] } \\ &  \leq \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { j - 1 } \displaystyle \sum _ { u ^ { \mathrm { i n } } \leq u ^ { \mathrm { i n } } , 1 } [ w _ { \mathrm { i } } ^ { \lambda } ( V _ { \mathrm { i } } ( \theta ^ { \mu ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha } } } } } ) + \lambda - V _ { \mathrm { i } } ( \theta ^  \mu ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } ^ { \alpha _ { \mathrm { i n } } } } } ) ) - \mu w _  \mathrm { i }  \end{array}
$$

Combining the decomposition in (40) with (41) and (42), we obtain

$$
\frac { T - 1 } { T } \Big ( \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \bar { \theta } _ { 2 : T } } ) - \mathrm { S T C H } _ { \lambda } ^ { \mu } ( \pi ^ { \theta _ { \lambda } ^ { * } } ) \Big ) \leq ( \mathrm { I } ) + ( \mathrm { I I } ) + ( \mathrm { I I I } ) + ( \mathrm { I I I } ) + ( \mathrm { I V } ) .\tag{43}
$$

Term (I). Since $V _ { i } ( \theta ) = \langle r _ { i } , \theta \rangle$ , Lemma 30 implies that, for every $\begin{array} { r } { i \in [ m ] , \ \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } ( V _ { i } ^ { * } - } \end{array}$ $\begin{array} { r } { V _ { i } ( \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } ) ) \leq \widetilde { \mathcal { O } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) + \frac { 4 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } } \end{array}$ . Therefore, we have

$$
\mathrm { ( I ) } \le \widetilde { \mathcal { O } } \Bigl ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Bigr ) + \frac { 4 H } { T } \sum _ { i = 1 } ^ { m } \lambda _ { i } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } .\tag{44}
$$

The inequality uses $\begin{array} { r } { w _ { i } \geq 0 , \sum _ { i } w _ { i } = 1 } \end{array}$ , and $\lambda \in \Delta _ { m }$

It remains to bound the term involving $\psi _ { i } ^ { t }$ in (44). Fix an objective $i \in [ m ]$ and an episode $t \in [ T - 1 ]$ . On the confidence event, we have $P \in \mathcal { P } ^ { t }$ , and $\theta _ { i } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ is feasible for the auxiliary projection used to construct the policy for episode $t + 1$ . Recall that $\phi _ { \mathrm { s a f e } } =$ $\textstyle \operatorname* { m a x } _ { j \in [ n ] } d ^ { 2 } / ( \tau _ { j } - \gamma _ { j } ) ^ { 2 }$ . Since $0 < \tau _ { j } - \gamma _ { j } \leq d$ and $d \geq 1$ , we have $1 / ( \tau _ { j } - \gamma _ { j } ) \leq \sqrt { \phi _ { \mathrm { s a f e } } } \leq \phi _ { \mathrm { s a f e } }$ for every $j \in [ n ]$ . Moreover, for every $\bar { P } ^ { \prime } \in \mathcal { P } ^ { t }$ , the bound (21) also holds with $P ^ { t + 1 }$ replaced by $P ^ { \prime } { } _ { ; }$ and Lemma 27 gives $\begin{array} { r } { \| \theta ^ { P ^ { \prime } , \widetilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \| _ { 1 } \le 2 H | S | \sum _ { h = 1 } ^ { H } \sum _ { s , a } \theta _ { h } ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ( s , a ) \epsilon _ { h } ^ { t } ( s , a ) } \end{array}$ We apply Lemma 32 with $\widetilde { \theta } = \widetilde { \theta } _ { i } ^ { t + 1 }$ and use $( 1 - \psi _ { i } ^ { t } ) \theta _ { h } ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } ( s , a ) \leq \theta _ { h } ^ { P , \nu _ { i } ^ { t + 1 } } ( s , a )$ for all $h , s , a .$

which follows from the construction of $\nu _ { i } ^ { t + 1 }$ . Then we obtain

$$
\psi _ { i } ^ { t } \leq \phi _ { \mathrm { s a f e } } \left[ 6 H | \mathcal { S } | \sum _ { h = 1 } ^ { H } \sum _ { s , a } \theta _ { h } ^ { P , \nu _ { i } ^ { t + 1 } } ( s , a ) \epsilon _ { h } ^ { t } ( s , a ) + 2 \operatorname* { m a x } _ { j \in [ n ] } \sum _ { h = 1 } ^ { H } \sum _ { s , a } \theta _ { h } ^ { P , \nu _ { i } ^ { t + 1 } } ( s , a ) \xi _ { j , h } ^ { t } ( s , a ) \right] ,\tag{45}
$$

where $\epsilon _ { h } ^ { t }$ and $\xi _ { j , h } ^ { t }$ are computed from the auxiliary trajectories generated by $\{ \nu _ { i } ^ { k } \} _ { k = 1 } ^ { t }$ . By (8) and (7), both $\epsilon _ { h } ^ { t } ( s , a )$ and $\xi _ { j , h } ^ { t } ( s , a )$ are at most $\widetilde { \mathcal { O } } ( 1 / \sqrt { \operatorname* { m a x } \{ 1 , N _ { h } ^ { t } ( s , a ) \} } )$ . Summing (45) over $t = 1 , \dots , T - 1$ and applying (26) to the auxiliary trajectories, we obtain, for every $i \in [ m ]$ ，

$$
\sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } \le \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a f e } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | T } \right) .\tag{46}
$$

Since $\lambda \in \Delta _ { m } .$ , substituting (46) into (44) yields

$$
\mathrm { ( I ) } \le \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \right) .\tag{47}
$$

Term (II). Since $| w _ { i } - w _ { i } ^ { t } | \leq w _ { i } + w _ { i } ^ { t } \leq 2$ , we obtain

$$
\begin{array} { r l } & { \mathrm { ( I I ) } \le \displaystyle \operatorname* { m a x } _ { w \in \Delta _ { m } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left[ \left. w - w ^ { t } , \lambda \odot \left( \widetilde { V } ^ { t + 1 } + \iota - V ^ { t + 1 } \right) \right. - \mu \langle w , \log w \rangle + \mu \langle w ^ { t } , \log w ^ { t } \rangle \right] } \\ & { \quad \quad + \displaystyle \frac { 2 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } \big ( \theta ^ { P , \widetilde { \nu } _ { i } ^ { t + 1 } } \big ) - \widetilde { V } _ { i } ^ { t + 1 } \right| + \displaystyle \frac { 2 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ^ { t + 1 } - V _ { i } \big ( \theta ^ { P , \pi ^ { t + 1 } } \big ) \right| . } \end{array}\tag{8}
$$

Lemma 33 bounds the first term on the right-hand side of (48) by $\widetilde { \mathcal { O } } \left( H ^ { 2 } / ( \mu T ) + \mu / T \right)$

We next bound the auxiliary estimation term. For each $i \in [ m ]$ , using $\widetilde { \theta } _ { i } ^ { t + 1 } = \theta ^ { P _ { i } ^ { t + 1 } , \widetilde { \nu } _ { i } ^ { t + 1 } }$ for some $P _ { i } ^ { t + 1 } \in \mathcal { P } ^ { t }$ , we have

$$
\left. V _ { i } ( \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } ) - \widetilde { V } _ { i } ^ { t + 1 } \right. = \left. \left. r _ { i } , \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } \right. - \left. \tilde { r } _ { i } ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \right. \right. \leq \left. \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } - \theta ^ { P _ { i } ^ { t + 1 } , \tilde { \nu } _ { i } ^ { t + 1 } } \right. _ { 1 } + 2 \left. \sigma ^ { t } , \widetilde { \theta } _ { i } ^ { t + 1 } \right. .
$$

The unweighted form of Lemma 31 and the reward bonus bound used in Lemma 30 imply $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } | \check { V _ { i } } ( \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } ) - \widetilde { V } _ { i } ^ { t + 1 } | \leq \widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } ) + \frac { 4 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi _ { i } ^ { t } } \end{array}$ . Using $\lambda \in \Delta _ { m }$ together with (46), we obtain

$$
\frac 2 T \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ( \theta ^ { P , \tilde { \nu } _ { i } ^ { t + 1 } } ) - \widetilde { V } _ { i } ^ { t + 1 } \right| \leq \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \right) .
$$

It remains to bound the main estimation term. By construction, we have $\widetilde { \theta } ^ { t + 1 } =$ $\theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } }$ for some $P ^ { t + 1 } \in \mathcal { P } ^ { t }$ , and $\theta ^ { P , \pi ^ { t + 1 } } = ( 1 - \psi ^ { t } ) \dot { \theta ^ { P , \pi ^ { t + 1 } } } + \psi ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ . Thus, for every $i \in [ m ]$ , we have

$$
\vert V _ { i } ^ { t + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \vert = \vert \langle \bar { r } _ { i } ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle - \langle r _ { i } , \theta ^ { P , \pi ^ { t + 1 } } \rangle \vert \leq 2 \langle \sigma ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle + \Vert \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \Vert _ { 1 } + H \psi ^ { t } .
$$

The first term is controlled by the same reward confidence argument used in the proof of Lemma 29, using the bonus $\sigma ^ { t }$ . The second term is controlled by the unweighted form of Lemma 28. Hence, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ^ { t + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \right| \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 5 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .\tag{49}
$$

The sum $\scriptstyle \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t }$ is bounded by the same argument used in the proof of (46), with $( \widetilde \nu _ { i } ^ { t + 1 } , \nu _ { i } ^ { t + 1 } , \psi _ { i } ^ { t } )$ replaced by $( \widetilde { \pi } ^ { t + 1 } , \pi ^ { t + 1 } , \psi ^ { t } )$ . Therefore, we have

$$
\sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } \leq \widetilde { \mathcal { O } } \Bigl ( \phi _ { \mathrm { s a f e } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | T } \Bigr ) .\tag{50}
$$

Combining (49) and (50), we obtain

$$
\frac { 2 } { T } \sum _ { t = 1 } ^ { T - 1 } \sum _ { i = 1 } ^ { m } \lambda _ { i } \left| V _ { i } ^ { t + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { t + 1 } } ) \right| \leq \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \right) .
$$

Combining (48) with the bounds established above for its three terms, we obtain

$$
\mathrm { ( I I ) } \le \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } \right) .\tag{51}
$$

Term (III). Since $V _ { i } ( \theta ) = \langle r _ { i } , \theta \rangle$ , we have

$$
\mathrm { ( I I I ) } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde \theta ^ { t + 1 } \right. + \frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \widetilde \theta ^ { t + 1 } - \theta ^ { P , \pi ^ { t + 1 } } \right. .\tag{52}
$$

The first term in (52) is controlled by Lemma 29 with $\theta = \theta _ { \lambda } ^ { * }$ . Since $\theta _ { \lambda } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde { \theta } ^ { t + 1 } \right. \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } \Big ) + \frac { 2 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .\tag{53}
$$

We next bound the second term in (52). By construction, we have $\widetilde { \theta } ^ { t + 1 } = \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } }$ for some $P ^ { t + 1 } \in \mathcal { P } ^ { t }$ . At the beginning of episode t + 1, SafeProj selects $\pi ^ { \dagger }$ with probability $\psi ^ { t }$ and $\widetilde { \pi } ^ { t + 1 }$ with probability $1 - \psi ^ { t }$ . Thus, we have $\theta ^ { P , \pi ^ { t + 1 } } = ( 1 - \stackrel { \smile } { \psi } ^ { t } ) \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } + \psi ^ { t } \theta ^ { P , \pi ^ { \dagger } }$ . Moreover, since $r _ { i , h } ( s , a ) \in [ 0 , 1 ]$ and $w ^ { t } , \lambda \in \Delta _ { m }$ , we have $\begin{array} { r } { 0 \leq \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i , h } ( s , a ) \leq 1 , \forall h , s , a , t . } \end{array}$ Therefore, for every t, we obtain

$$
\begin{array} { r l } & { \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \widetilde { \theta } ^ { t + 1 } - \theta ^ { P , \pi ^ { t + 1 } } \right. } \\ & { \quad = \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \right. + \left. \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \pi ^ { t + 1 } } \right. } \\ & { \quad \le \Big \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \Big \| _ { 1 } + H \psi ^ { t } . } \end{array}\tag{54}
$$

The last inequality follows from $\theta ^ { P , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \pi ^ { t + 1 } } = \psi ^ { t } ( \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \pi ^ { \dagger } } )$ and the fact that any H-step occupancy measure has total mass H. Averaging (54) over $t = 1 , \ldots , T - 1$ and using the unweighted form of Lemma 28, we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T - 1 } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } r _ { i } , \widetilde { \theta } ^ { t + 1 } - \theta ^ { P , \pi ^ { t + 1 } } \right. \leq \widetilde { \mathcal { O } } \Big ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \Big ) + \frac { 3 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } .\tag{55}
$$

Combining (53) and (55), we obtain $\begin{array} { r } { \mathrm { ( I I I ) } \le \widetilde { \mathcal { O } } ( H ^ { 2 } | \mathcal { S } | ^ { 3 / 2 } \sqrt { | \mathcal { A } | / T } ) + \frac { 5 H } { T } \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } } \end{array}$ . Using the bound in (50), namely $\begin{array} { r } { \sum _ { t = 1 } ^ { T - 1 } \psi ^ { t } \leq \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | T } ) } \end{array}$ , we conclude that

$$
\mathrm { ( I I I ) } \le \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \right) .\tag{56}
$$

Term (IV). Using $w _ { i } \geq 0 , \sum _ { i = 1 } ^ { m } w _ { i } = 1$ , and $\lambda \in \Delta _ { m } ,$ , we have $\begin{array} { r } { \mathrm { ( I V ) } \le \frac { H } { T } \sum _ { i = 1 } ^ { m } \lambda _ { i } \sum _ { t = 1 } ^ { T - 1 } \psi } \end{array}$ <sup>t</sup><sub>i</sub> . By (46), we obtain

$$
\mathrm { ( I V ) } \le \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { | A | / T } \right) .\tag{57}
$$

Consequently, by combining (40), (43), (47), (51), (56), and (57), and noting that $H / T$ is dominated by $H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T }$ , we have, with probability at least $1 - \delta$

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } + \frac { H ^ { 2 } } { \mu T } + \frac { \mu } { T } \right) .
$$

This completes the proof.

## Appendix E. Proof of Theorem 10

Proof We work on the event $\mathcal { E } _ { R , C , T } ( \delta )$ , which holds with probability at least $1 - \delta .$ . On this event, we have $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t }$ and $P \in \mathcal { P } ^ { t }$ for every $t \in [ T - 1 ]$

We first verify the safety of the initial executed policy $\pi ^ { 1 }$ . Algorithm 1 sets $\psi ^ { 0 } = $ $\mathrm { m a x } _ { j \in [ n ] } ( d - \tau _ { j } ) / ( d - \gamma _ { j } )$ . Since $0 \leq \gamma _ { j } < \tau _ { j } \leq d .$ , we have $\psi ^ { 0 } \in [ 0 , 1 ]$ . At the beginning of the first episode, the algorithm selects $\pi ^ { \dagger }$ with probability $\psi ^ { 0 }$ and $\widetilde { \pi } ^ { 1 }$ with probability $1 - \psi ^ { 0 }$ . Thus, by the law of total probability, we have $\theta ^ { P , \pi ^ { 1 } } \stackrel {  } { = } ( 1 - \psi ^ { 0 } ) \theta ^ { P , \widetilde { \pi } ^ { 1 } } + \stackrel {  } { \psi ^ { 0 } } \theta ^ { P , \pi ^ { \dagger } }$ . For every $j \in [ n ]$ , the constraint cost of $\widetilde { \pi } ^ { 1 }$ is at most $d ,$ while that of $\pi ^ { \dagger }$ equals $\gamma _ { j }$ . Therefore, we have $( C ^ { \top } \theta ^ { P , \pi ^ { 1 } } ) _ { j } \leq ( 1 - \psi ^ { 0 } ) d + \psi ^ { 0 } \gamma _ { j } = d - \psi ^ { 0 } ( d - \gamma _ { j } ) \leq \tau _ { j }$ , where the last inequality follows from $\psi ^ { 0 } \geq ( d - \tau _ { j } ) / ( d - \gamma _ { j } )$ . Thus, we have $C ^ { \top } \theta ^ { P , \pi ^ { 1 } } \leq \tau$

It remains to show that, for every $t \in [ T - 1 ]$ , the policy π<sup>t+1</sup> constructed after episode t and executed in episode $t + 1$ is safe. Fix any $t \in [ T - 1 ]$ and $j \in [ n ]$ . We prove $( C ^ { \top } \theta ^ { P , \pi ^ { t + 1 } } ) _ { j } \le \tau _ { j }$ . When the projection is feasible, the cost confidence bound, $P \in \mathcal { P } ^ { t }$ , and the fact that every constraint cost is at most d give

$$
( C ^ { \top } \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } ) _ { j } \leq \operatorname* { m i n } \{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \} .\tag{58}
$$

We consider three cases.

Case $\boldsymbol { \mathit { 1 } } \cdot \textit { P r o j } ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$ is infeasible. In this case, SafeProj sets $\psi ^ { t } = 1$ , and hence $\pi ^ { t + 1 } = \pi ^ { \dagger } .$ Therefore, we have $( C ^ { \top } \theta ^ { P , \pi ^ { t + 1 } } ) _ { j } = \gamma _ { j } < \tau _ { j }$

Case $\mathcal { Q } \colon P r o j ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$ is feasible and $\psi ^ { t } = 0$ . In this case, we have $\pi ^ { t + 1 } = \widetilde { \pi } ^ { t + 1 }$ Since the denominator of every ratio in (12) is at least $\tau _ { j } - \gamma _ { j } > 0$ , the definition of $\psi ^ { t }$ implies that the right-hand side of (58) is at most $\tau _ { j }$ . It follows that $( C ^ { \top } \theta ^ { P , \pi ^ { t + 1 } } ) _ { j } \le \tau _ { j }$

Case 3: $P r o j ( \widehat { \theta } ^ { t + 1 } , \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \Xi ^ { t } )$ is feasible and $\psi ^ { t } \in ( 0 , 1 )$ . At the beginning of episode $t + 1 , \mathsf { S a f e P r o j }$ selects $\pi ^ { \dagger }$ with probability $\psi ^ { t }$ and $\widetilde { \pi } ^ { t + 1 }$ with probability $1 - \psi ^ { t }$ . By the law of total probability, we have $\theta ^ { \bar { P , \pi } ^ { t + 1 } } = ( 1 \bar { 1 } - \psi ^ { t } ) \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } + \psi ^ { t } \theta ^ { \bar { P , \pi } ^ { \dagger } }$ . Consequently, we have

$$
\big ( C ^ { \top } \theta ^ { P , \pi ^ { t + 1 } } \big ) _ { j } \leq ( 1 - \psi ^ { t } ) \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \right\} + \psi ^ { t } \gamma _ { j } .\tag{59}
$$

If the minimum in (59) is at most $\tau _ { j }$ , then the right-hand side of (59) is at most $( 1 - \psi ^ { t } ) \tau _ { j } +$ $\psi ^ { t } \gamma _ { j } \le \tau _ { j }$ . Otherwise, the definition of $\psi ^ { t }$ gives

$$
\psi ^ { t } \geq \frac { \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } \big ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } \big ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \right\} - \tau _ { j } } { \operatorname* { m i n } \left\{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } \big ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } \big ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \right\} - \gamma _ { j } } .
$$

Rearranging this inequality shows that the right-hand side of (59) is at most $\tau _ { j }$ . Since $t \in [ T - 1 ]$ and $j \in [ n ]$ were arbitrary, every generated policy $\pi ^ { t + 1 }$ is safe on $\mathcal { E } _ { R , C , T } ( \delta )$

The same analysis applies to each auxiliary policy $\nu _ { i } ^ { t + 1 }$ for $i \in [ m ]$ , with $\widetilde { \pi } ^ { t + 1 }$ and $\psi ^ { t }$ replaced by $\widetilde \nu _ { i } ^ { t + 1 }$ and $\psi _ { i } ^ { t }$ , respectively. The initial auxiliary policies $\nu _ { i } ^ { 1 }$ are safe by the argument used for $\pi ^ { 1 }$ . Therefore, every policy executed by Algorithm 1 is safe.

Finally, consider the averaged occupancy measure returned by the algorithm, $\bar { \theta } \ =$ $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \theta ^ { \dot { P } , \pi ^ { t } }$ . Since $\theta ^ { P , \pi ^ { t } } \in \bar { \Theta _ { \mathrm { s a f e } } }$ for every $t \in [ T ]$ and $\Theta _ { \mathrm { s a f e } }$ is convex, $\breve { \bar { \theta } }$ also belongs to $\Theta _ { \mathrm { s a f e } }$ . Let ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ be the Markov policy induced by ${ \bar { \theta } } ,$ so that $\theta ^ { P , \widehat { \pi } } = \bar { \theta }$ . Therefore, we have Violation $( \widehat { \pi } ) = 0$ . This completes the proof.

## Appendix F. Proof of Theorem 11

The exploration stage of Algorithm 2 executes one policy per episode and collects one trajectory $\{ ( s _ { h } ^ { t } , a _ { h } ^ { t } , s _ { h + 1 } ^ { t } ) \} _ { h = 1 } ^ { H }$ in each episode t. The counts and estimators in $( 3 ) \AA { - } ( 5 )$ are therefore constructed from these exploration trajectories. The confidence widths in (6), (7), and (8) remain unchanged, and the transition bonus in (13) uses the same counts. Lemmas 26, 25, and 23 apply directly to this trajectory sequence. Retaining these confidence widths afects the resulting bounds only through logarithmic factors. In the planning stage of Algorithm 3, the episode superscript is omitted and $( \widehat { r } , \widehat { C } , \mathcal { P } , \Xi , \sigma , \zeta )$ denote these quantities evaluated at the final counts $N ^ { T }$ of the exploration dataset D.

## F.1 Lemmas

Throughout this subsection, we use the fixed quantities described above. For the final exploration dataset, we define the confidence event $\mathcal { E } _ { \mathrm { f i x } } ( \delta ) : = \{ | \widehat { r } _ { i , h } ( s , a ) - r _ { i , h } ( s , a ) | \ \leq$ $\sigma _ { h } ( s , a ) , \forall i \in [ m ] , h \in [ H ] , ( s , a ) \in \mathcal { S } \times \mathcal { A } , | \widehat { C } - C | \leq \Xi$ , and $P \in { \mathcal { P } } \}$ . By Lemmas 26, 25, and 23, applied to this dataset with δ replaced by $\delta / 3 .$ each of the three requirements fails with probability at most $\delta / 3$ . Therefore, $\mathcal { E } _ { \mathrm { f i x } } ( \delta )$ holds with probability at least $1 - \delta .$ . The analysis in this appendix also uses the online cost and transition events $\mathcal { E } _ { C } ( \delta )$ and $\mathcal { E } _ { T } ( \delta )$ of Appendix D, instantiated for the single-rollout estimates of Algorithm 2, together with two martingale events. The first is the event in the proof of Lemma 28. Its martingale diferences depend only on the inverse square-root visit counts, and the same bound holds for the single exploration rollout. The second follows by repeating the martingale argument in the proof of Lemma 29 with the exploration reward $\overline { { r } } _ { h } ^ { t }$ in place of $\scriptstyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { t } \lambda _ { i } \sigma _ { h } ^ { t }$ . Since $\overline { { r } } ^ { t }$ is determined before episode $t + 1$ and satisfies $0 \leq \overline { { r } } _ { h } ^ { t } ( s , a ) \leq 1$ , the resulting sequence remains a martingale diference sequence with increments bounded by H. The same Azuma-Hoefding bound therefore applies. Throughout this appendix, we work on the intersection of $\mathcal { E } _ { C } ( \delta / 5 ) , \mathcal { E } _ { T } ( \delta / 5 ) , \mathcal { E } _ { \mathrm { f i x } } ( \delta / 5 )$ , and the two martingale events above, each taken with failure probability $\delta / 5$ . By a union bound, this intersection holds with probability at least $1 - \delta .$ , and this allocation afects only logarithmic factors, which are already absorbed into the $\widetilde { \mathcal { O } } ( \cdot )$ notation. We refer to this intersection as the joint event. The planning-stage argument in the proof of Theorem 12 uses only the fixed cost and transition confidence bounds. Therefore, on the joint event, every policy $\pi ^ { k + 1 }$ is $\mathrm { s a f e }$ , and Lemma 35 applies to each $\pi ^ { k + 1 }$

On the joint event, the transition estimate enters the analysis only through the following comparison. Let π be any policy and let $P ^ { \prime } \in \mathcal { P }$ Since $P ^ { \prime } \in \mathcal { P }$ and $P \in \mathcal { P }$ , both kernels satisfy the transition confidence bounds in (8). Therefore, for every $h \in [ H ]$ and $( s , a ) \in \mathcal { S } \times \mathcal { A }$ , we have $\big \| P _ { h } ^ { \prime } ( \cdot | s , a ) - P _ { h } ( \cdot | s , a ) \big \| _ { 1 } \leq \operatorname* { m i n } \{ 2 , 2 | S | \epsilon _ { h } ( s , a ) \}$ . The bound 2 is valid since both kernels are probability distributions. Comparing this quantity with the transition bonus in (13) and using l $1 ( 2 m T H | S | ^ { 2 } | A | / \delta ) \leq 2 \ln ( 2 m T H | S | | A | / \delta )$ , we obtain H min $\{ 2 , 2 | S | \epsilon _ { h } ( s , a ) \} \le 2 \sqrt { | S | } \zeta _ { h } ( s , a )$ . Therefore, Lemma 27 yields

$$
\sum _ { h = 1 } ^ { H } \bigl \lVert \theta _ { h } ^ { P ^ { \prime } , \pi } - \theta _ { h } ^ { P , \pi } \bigr \rVert _ { 1 } \leq 2 \sqrt { | \mathcal { S } | } \langle \zeta , \theta ^ { P , \pi } \rangle .\tag{60}
$$

Lemma 35 On the joint event, with $\eta = T ^ { - 1 / 2 }$ , for every policy π with $\theta ^ { P , \pi } \in \Theta _ { \mathrm { s a f e } } ,$ we have

$$
\langle \sigma , \theta ^ { P , \pi } \rangle \leq \widetilde { \mathcal O } \left( \sqrt { \phi _ { \mathrm { s a f e } } } H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } \right) , \quad \langle \zeta , \theta ^ { P , \pi } \rangle \leq \widetilde { \mathcal O } \left( \sqrt { \phi _ { \mathrm { s a f e } } } H ^ { 3 } | S | ^ { 3 / 2 } \sqrt { \frac { | A | } { T } } \right) .
$$

Proof We work on the joint event defined at the beginning of this appendix and fix any policy π with $\theta ^ { P , \pi } \in \Theta _ { \mathrm { s a f e } }$ . When $T = 1$ , both bounds are immediate, since $\sigma _ { h } \leq 1 , \zeta _ { h } \leq H$ and $\theta ^ { P , \pi }$ has total mass H. We therefore assume $T \geq 2 .$ , so that $1 / ( T - 1 ) \leq 2 / T$ and the bounds of Lemmas 28 and 29, which are normalized by $1 / T$ , can be applied directly.

As described at the beginning of this appendix, the fixed bonuses are the expressions in (6) and (13) evaluated at the final counts $N _ { h } ^ { T }$ , whereas the online bonuses are the same expressions evaluated at $N _ { h } ^ { t }$ . Since these expressions are nonincreasing in the count and $N _ { h } ^ { T } ( s , a ) \ge N _ { h } ^ { t } ( s , a )$ for every $t \leq T$ , we have $\sigma _ { h } ( s , a ) \leq \sigma _ { h } ^ { t } ( s , a )$ and $\zeta _ { h } ( s , a ) \leq \zeta _ { h } ^ { t } ( s , a )$ for all $h \in [ H ]$ and $( s , a ) \in S \times A$ . Moreover, Line 5 of Algorithm 2 gives $\sigma _ { h } ^ { t } ( s , a ) \leq \overline { { r } } _ { h } ^ { t } ( s , a )$ and $\zeta _ { h } ^ { t } ( s , a ) / H \leq \overline { { r } } _ { h } ^ { t } ( s , a )$ for all $h , s , a , t$ . It therefore sufices to bound $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \langle \overline { { r } } ^ { t } , \theta ^ { P , \pi } \rangle } \end{array}$ , since the bound for ζ follows after multiplication by H.

We first compare $\theta ^ { P , \pi }$ with the projected occupancy measures. On the joint event, we have $P \in \mathcal { P } ^ { t }$ and ${ \widehat { C } } ^ { t } - \Xi ^ { t } \leq C$ . Since $C ^ { \top } \theta ^ { P , \pi } \leq \tau$ , we have $( \widehat { C } ^ { t } - \Xi ^ { t } ) ^ { \top } \theta ^ { P , \pi } \leq C ^ { \top } \theta ^ { P , \pi } \leq \tau$ Thus, $\theta ^ { P , \pi }$ is feasible for the projection. In particular, the projection is feasible, and $\psi ^ { t }$ is computed by (12). Applying (27) to the exponential update in Line 6 of Algorithm 2 and summing over $t = 1 , \dots , T - 1$ , we obtain

$$
\frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \bigl < \bar { r } ^ { t } , \theta ^ { P , \pi } - \widetilde { \theta } ^ { t + 1 } \bigr > \leq \frac { D _ { \mathrm { K L } } ( \theta ^ { P , \pi } \| \widetilde { \theta } ^ { 1 } ) } { \eta ( T - 1 ) } \leq \widetilde { \mathcal { O } } \Bigl ( \frac { H } { \sqrt { T } } \Bigr ) ,\tag{61}
$$

where the last inequality uses $\widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) = 1 / ( | S | ^ { 2 } | A | )$ and $\eta = T ^ { - 1 / 2 }$

It remains to bound the average exploration reward under the projected occupancy measures. If $\psi ^ { t } > 0$ , let $j \in [ n ]$ attain the maximum in the definition of $\psi ^ { t }$ . Then we have

$$
\begin{array} { r } { 1 - \psi ^ { t } = \frac { \tau _ { j } - \gamma _ { j } } { \operatorname* { m i n } \Big \{ \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } ^ { t } } \sum _ { h = 1 } ^ { H } ( \widehat { c } _ { j , h } ^ { t } + \xi _ { j , h } ^ { t } ) ^ { \top } \theta _ { h } ^ { P ^ { \prime } , \widetilde { \pi } ^ { t + 1 } } , d \Big \} - \gamma _ { j } } \geq \frac { \tau _ { j } - \gamma _ { j } } { d } . } \end{array}
$$

If $\psi ^ { t } = 0$ , we directly have $1 - \psi ^ { t } = 1 \geq ( \tau _ { j } - \gamma _ { j } ) / d$ for every $j \in [ n ]$ , since $\tau _ { j } - \gamma _ { j } \leq d$ Thus, in both cases, we have $\begin{array} { r } { \frac { 1 } { 1 - \psi ^ { t } } \leq \mathrm { m a x } _ { j \in [ n ] } \frac { d } { \tau _ { i } - \gamma _ { i } } = \sqrt { \phi _ { \mathrm { s a f e } } } , \forall t \in [ T - 1 ] } \end{array}$ . By construction, we have $\widetilde { \theta } ^ { t + 1 } = \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } }$ for some $P ^ { t + 1 } \in \mathcal { P } ^ { t }$ , and $0 \leq \overline { { r } } _ { h } ^ { t } ( s , a ) \leq 1$ . Using $( 1 - \psi ^ { t } ) \theta _ { h } ^ { P , \widetilde { \pi } ^ { t + 1 } } ( s , a ) \le \theta _ { h } ^ { P , \pi ^ { t + 1 } } ( s , a )$ together with $\overline { r } ^ { t } \geq 0$ , we obtain $( 1 - \psi ^ { t } ) \langle \overline { { r } } ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle \leq$ $\langle \overline { { r } } ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \rangle + ( 1 - \psi ^ { t } ) \| \theta ^ { \hat { P } ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \| _ { 1 }$ . Dividing by $1 - \psi ^ { t }$ and using $\begin{array} { r } { \frac { 1 } { 1 - \psi ^ { t } } \leq \sqrt { \phi _ { \mathrm { s a f e } } } . } \end{array}$ we have $1 \leq \sqrt { \phi _ { \mathrm { s a f e } } } ( 1 - \psi ^ { t } )$ for the second term. Averaging over $t = 1 , \dots , T - 1$ , we obtain

$$
\begin{array} { r l } & { \frac { 1 } { T - 1 } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \langle \bar { r } ^ { t } , \widetilde { \theta } ^ { t + 1 } \rangle } \\ & { \quad \le \sqrt { \phi _ { \mathrm { s a f e } } } \left[ \frac { 1 } { T - 1 } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \langle \bar { r } ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \rangle + \frac { 1 } { T - 1 } \displaystyle \sum _ { t = 1 } ^ { T - 1 } ( 1 - \psi ^ { t } ) \| \theta ^ { P ^ { t + 1 } , \widetilde { \pi } ^ { t + 1 } } - \theta ^ { P , \widetilde { \pi } ^ { t + 1 } } \| _ { 1 } \right] . } \end{array}\tag{62}
$$

The second sum in (62) is controlled by the argument in the proof of Lemma 28, applied to the exploration trajectories of Algorithm 2, and is at most $\widetilde { \mathcal { O } } ( H ^ { 2 } | S | ^ { 3 / 2 } \sqrt { | A | / T } )$ . For the first sum, the exploration stage executes $\pi ^ { t + 1 }$ in episode $t + 1$ . Since both bonuses are nonnegative, Line 5 of Algorithm 2 gives $\textstyle { \overline { { r } } } _ { h } ^ { t } ( s , a ) \leq \sigma _ { h } ^ { t } ( s , a ) + \zeta _ { h } ^ { t } ( s , a ) / H$ . The martingale event for $\overline { { r } } ^ { t }$ and the two bounds in Lemma 34, applied to the same exploration trajectories, therefore give $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \langle \overline { { r } } ^ { t } , \theta ^ { P , \pi ^ { t + 1 } } \rangle \leq \widetilde { \mathcal O } ( H | \mathcal { S } | \sqrt { | A | / T } ) } \end{array}$

Combining these two bounds with (61) and (62) proves the bound for $\sigma ,$ and multiplying by H proves the bound for $\zeta .$ ■

Lemma 36 For each $i \in [ m ]$ , let $\widetilde { \theta } _ { i }$ and $\widetilde { V } _ { i } = \langle \bar { r } _ { i } , \widetilde { \theta } _ { i } \rangle$ be computed in Lines $\it 3 - 1 1$ of Algorithm 3. On the joint event, with $\eta = K ^ { - 1 / 2 }$ , we have $V _ { i } ^ { * } - \widetilde { V } _ { i } \le \widetilde { \mathcal { O } } ( H / \sqrt { K } )$ for every $i \in [ m ]$

Proof Consider any $i \in [ m ]$ , and let $\theta _ { i } ^ { * } \in \arg \operatorname* { m a x } _ { \theta \in \Theta _ { \mathrm { s a f e } } } V _ { i } ( \theta )$ . Since $\theta _ { i } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ and $P \in { \mathcal { P } }$ 2 we have $\theta _ { i } ^ { * } \in \Theta ( \mathcal { P } )$ . The cost confidence bound ${ \widehat { C } } - \Xi \leq C$ also gives $( \widehat { C } - \Xi ) ^ { \top } \theta _ { i } ^ { * } \leq$

$C ^ { \top } \theta _ { i } ^ { * } \leq \tau$ . Thus, on the joint event, the projection Pro $\mathrm { j } ( \widehat { \theta } _ { i } , \mathcal { P } , \widehat { C } , \Xi )$ is feasible. By the optimality condition of the KL projection with $\widehat { \theta } _ { i , h } ( s , a , s ^ { \prime } ) = \widetilde { \theta } _ { h } ^ { 1 } ( s , a , s ^ { \prime } ) \exp \{ K ^ { 1 / 2 } \bar { r } _ { i , h } ( s , a ) \}$ , we have $\begin{array} { r } { \langle \bar { r } _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } \rangle \leq \frac { D _ { \mathrm { K L } } ( \theta _ { i } ^ { * } \| \widetilde { \theta } ^ { 1 } ) - D _ { \mathrm { K L } } ( \widetilde { \theta } _ { i } \| \widetilde { \theta } ^ { 1 } ) } { \sqrt { K } } } \end{array}$ . Since $ { \widetilde { \theta } } ^ { 1 }$ is uniform, we obtain $D _ { \mathrm { K L } } ( \theta _ { i } ^ { * } \| \widetilde { \theta } ^ { 1 } ) \leq$ $H \log ( | S | ^ { 2 } | A | )$ . Therefore, we have $\langle \bar { r } _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } \rangle \le \widetilde { \mathcal { O } } ( H / \sqrt { K } )$ . Using $\bar { r } _ { i } ~ \geq ~ r _ { i }$ , we obtain $V _ { i } ^ { * } - \widetilde { V } _ { i } = \langle r _ { i } , \theta _ { i } ^ { * } \rangle - \langle \bar { r } _ { i } , \widetilde { \theta } _ { i } \rangle \leq \langle \bar { r } _ { i } , \theta _ { i } ^ { * } - \widetilde { \theta } _ { i } \rangle \leq \widetilde { \mathcal { O } } ( H / \sqrt { K } )$ . This completes the proof.

Lemma 37 Let $\{ \widetilde { \theta } ^ { k + 1 } \} _ { k = 1 } ^ { K }$ be generated by the planning-stage occupancy measure update in Algorithm 3. For any safe occupancy measure $\theta \in \Theta _ { \mathrm { s a f e } }$ , on the joint event, with $\eta = K ^ { - 1 / 2 }$ 2 we have

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left. \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } r _ { i } , \theta - \widetilde { \theta } ^ { k + 1 } \right. \leq \widetilde { \mathcal { O } } \left( H / \sqrt { K } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 2 } \sqrt { | A | / T } \right) .
$$

Proof For each planning iteration $k ,$ we define the true scalarized reward $R _ { h } ^ { k } ( s , a ) : =$ ${ \textstyle \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } r _ { i , h } ( s , a ) }$ . Since $\theta \in \Theta _ { \mathrm { s a f e } } , P \in \mathcal { P }$ , and ${ \widehat { C } } - \Xi \leq C$ , we have $( { \widehat { C } } - \Xi ) ^ { \top } \theta \leq C ^ { \top } \theta \leq \tau$ Thus, θ is feasible for the projection. Applying the KL projection argument in the proof of Lemma 29 to the K planning updates and summing over k gives $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \langle \bar { R } ^ { k } , \theta - \bar { \theta } ^ { k + 1 } \rangle \leq } \end{array}$ $\begin{array} { r } { \frac { H \log ( | S | ^ { 2 } | A | ) } { \sqrt { K } } = \widetilde { \mathcal { O } } ( H / \sqrt { K } ) } \end{array}$

We next express the preceding inequality using the true scalarized reward $R ^ { k }$ . On the event $\mathcal { E } _ { \mathrm { f i x } } ( \delta )$ , σ is the reward bonus computed from the final exploration data. Thus, by Lemma 26 and the definition of $\bar { r } _ { i } .$ we obtain $0 \leq \bar { r } _ { i , h } ( s , a ) - r _ { i , h } ( s , a ) \leq 2 \sigma _ { h } ( s , a ) , \forall i , h , s , a .$ Using this bound, we have $\begin{array} { r } { 0 \leq \bar { R } _ { h } ^ { k } ( s , a ) - R _ { h } ^ { k } ( s , a ) \leq 2 \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \sigma _ { h } ( s , a ) } \end{array}$ . Hence, we obtain $\begin{array} { r } { \langle R ^ { k } , \theta - \widetilde { \theta } ^ { k + 1 } \rangle \leq \langle \bar { R } ^ { k } , \theta - \widetilde { \theta } ^ { k + 1 } \rangle + 2 \langle \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \sigma , \widetilde { \theta } ^ { k + 1 } \rangle } \end{array}$ . It remains to control the last term. Since $w ^ { k } , \lambda \in \Delta _ { m }$ , we have $\begin{array} { r } { \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \le 1 } \end{array}$ and therefore $\textstyle \langle \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \sigma , { \widetilde { \theta } } ^ { k + 1 } \rangle \leq \langle \sigma , { \widetilde { \theta } } ^ { k + 1 } \rangle$ For each $k \in [ K ]$ , we have $\widetilde { \theta } ^ { k + 1 } = \dot { \theta } ^ { P ^ { k + 1 } , \widetilde { \pi } ^ { k + 1 } }$ for some $P ^ { k + 1 } \in \mathcal { P }$ . Since $0 \leq \sigma _ { h } ( s , a ) \leq 1$ inequality (60) gives $\langle \sigma , \widetilde { \theta } ^ { k + 1 } \rangle \leq \langle \sigma , \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle + 2 \sqrt { | { \cal S } | } \langle \zeta , \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle$ . By the construction of $\pi ^ { k + 1 }$ , we have $\begin{array} { r } { \theta _ { h } ^ { P , \widetilde { \pi } ^ { k + 1 } } ( s , a ) \le \frac { 1 } { 1 - \psi ^ { k } } \theta _ { h } ^ { P , \pi ^ { k + 1 } } ( s , a ) } \end{array}$ for all $h , s , a$ . The argument in the proof of Lemma 35 gives $1 / ( 1 - \psi ^ { k } ) \leq \sqrt { \phi _ { \mathrm { s a f e } } }$ . Since σ and $\zeta$ are nonnegative, we therefore obtain $\langle \sigma , \widetilde { \theta } ^ { k + 1 } \rangle \leq \sqrt { \phi _ { \mathrm { s a f e } } } [ \langle \sigma , \theta ^ { P , \pi ^ { k + 1 } } \rangle + 2 \sqrt { | \cal { S } | } \langle \zeta , \theta ^ { P , \pi ^ { k + 1 } } \rangle ]$ . Averaging over $k \in [ K ]$ and applying Lemma 35 gives $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \langle \sum _ { i = 1 } ^ { m } w _ { i } ^ { k } \lambda _ { i } \sigma , \widetilde { \theta } ^ { k + 1 } \rangle \leq \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 2 } \sqrt { \frac { | A | } { T } } ) } \end{array}$ . Combining this bound with the preceding KL projection bound proves the lemma.

Lemma 38 Let $\{ \psi ^ { k } \} _ { k = 1 } ^ { K }$ be the probabilities of selecting $\pi ^ { \dagger }$ during the planning stage of Algorithm 3. On the joint event, we have

$$
\frac 1 K \sum _ { k = 1 } ^ { K } \psi ^ { k } \le \widetilde O \left( \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 2 } \sqrt { | A | / T } \right) .
$$

Proof On the joint event, we have $P \in \mathcal { P }$ and ${ \widehat { C } } - \Xi \leq C$ . Therefore, we have $( \widehat { C } -$ $\Xi ) ^ { \top } \theta ^ { P , \pi ^ { \dagger } } \leq C ^ { \top } \theta ^ { P , \pi ^ { \dagger } } = \gamma < \tau$ . The occupancy measure $\theta ^ { P , \pi ^ { \dagger } }$ is thus feasible for the fixed

planning projection, and the projection is feasible at every planning iteration. Fix $k \in [ K ]$ Since $0 < \tau _ { j } - \gamma _ { j } \leq d$ and $d \geq 1$ , we have $1 / ( \tau _ { j } - \gamma _ { j } ) \leq \sqrt { \phi _ { \mathrm { s a f e } } }$ for every $j \in [ n ]$ . Lemma 32 with $\widetilde { \theta } = \widetilde { \theta } ^ { k + 1 }$ therefore gives

$$
\psi ^ { k } \leq \sqrt { \phi _ { \mathrm { s a f e } } } \left( 1 - \psi ^ { k } \right) \left[ 3 \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \left\| \theta ^ { P ^ { \prime } , \widetilde { \pi } ^ { k + 1 } } - \theta ^ { P \widetilde { \pi } ^ { k + 1 } } \right\| _ { 1 } + 2 \operatorname* { m a x } _ { j \in [ n ] } \sum _ { h = 1 } ^ { H } \xi _ { j , h } ^ { \top } \theta _ { h } ^ { P , \widetilde { \pi } ^ { k + 1 } } \right] .
$$

By (60), the transition term is at most $6 \sqrt { | { \cal S } | } \langle \zeta , \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle$ . Since the cost bonus in (7) and the transition bonus in (13) difer only in logarithmic factors and in the factor $H _ { \mathrm { \ell } } { \sqrt { | { \cal S } | } } .$ , we have $\xi _ { j , h } ( s , a ) \le \widetilde { \mathcal { O } } ( \zeta _ { h } ( s , a ) / H )$ . The cost bonus term is therefore at most $\widetilde { \mathcal { O } } ( \langle \zeta , \dot { \theta } ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle / H )$ Since $\theta ^ { P , \pi ^ { k + 1 } } = ( 1 - \psi ^ { k } ) \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } + \psi ^ { k } \theta ^ { P , \pi ^ { \dagger } }$ , we have $( 1 - \psi ^ { k } ) \theta _ { h } ^ { P , \widetilde { \pi } ^ { k + 1 } } ( s , a ) \leq \theta _ { h } ^ { P , \pi ^ { k + 1 } } ( s , a )$ for all $h , s , a$ . Combining these bounds, we obtain

$$
\psi ^ { k } \leq \widetilde { \mathcal { O } } \left( \sqrt { \phi _ { \mathrm { s a f e } } | \boldsymbol { S } | } \left. \zeta , \theta ^ { P , \pi ^ { k + 1 } } \right. \right) .\tag{63}
$$

Averaging (63) over $k = 1 , \ldots , K$ and applying Lemma 35 to each safe policy $\pi ^ { k + 1 }$ proves the lemma. ■

Lemma 39 Let $\widetilde { \theta } _ { i }$ and $\widetilde { V } _ { i } = \langle \bar { r } _ { i } , \widetilde { \theta } _ { i } \rangle$ be computed in Lines 3–11 of Algorithm 3. For any safe single-objective optimal occupancy measure $\theta _ { i } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ , on the joint event, we have

$$
- \widetilde { \cal O } ( H / \sqrt { K } ) \le \widetilde { V } _ { i } - V _ { i } ^ { * } \le \widetilde { \cal O } \left( \phi _ { \mathrm { s a f e } } H ^ { 4 } | { \cal S } | ^ { 2 } \sqrt { | { \cal A } | / T } \right) .
$$

Proof Fix $i \in [ m ]$ . The lower bound $\widetilde { V } _ { i } - V _ { i } ^ { * } \ge - \widetilde { \mathcal { O } } ( H / \sqrt { K } )$ follows from Lemma 36. It remains to prove the upper bound.

Let $\widetilde \nu _ { i } = \pi ^ { \widetilde \theta _ { i } }$ be the policy induced by the projected occupancy measure $\widetilde { \theta } _ { i }$ . We define $\psi _ { i }$ by (12) with $( \mathcal { P } ^ { t } , \widehat { C } ^ { t } , \bar { \Xi } ^ { t } , \tilde { \pi } ^ { t + 1 } )$ replaced by $( \mathcal { P } , \widehat { C } , \Xi , \widetilde { \nu } _ { i } )$ . Let $\nu _ { i }$ equal $\pi ^ { \dagger }$ with probability ψ<sub>i</sub> and $\widetilde { \nu _ { i } }$ with probability $1 - \psi _ { i }$ . Since $\theta _ { i } ^ { * }$ is feasible for the auxiliary projection on the joint event, the projection is feasible. The argument used in the proof of Lemma 35 then gives $\frac { 1 } { 1 - \psi _ { i } } \leq \sqrt { \phi _ { \mathrm { s a f e } } }$ . The argument that considers the three cases separately in the proof of Theorem 10, applied to $( \widetilde { \nu } _ { i } , \nu _ { i } , \psi _ { i } )$ , shows that $\nu _ { i }$ is safe.

Repeating the derivation of (63) for $( \widetilde { \theta } _ { i } , \widetilde { \nu } _ { i } , \nu _ { i } , \psi _ { i } )$ gives $\psi _ { i } \le \widetilde { \mathcal { O } } ( \sqrt { \phi _ { \mathrm { s a f e } } | \boldsymbol { S } | } \langle \zeta , \theta ^ { P , \nu _ { i } } \rangle )$ Applying Lemma 35 to the safe policy $\nu _ { i }$ , we obtain $\psi _ { i } \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 2 } \sqrt { | A | / T } )$

Since $\theta _ { i } ^ { * }$ is a safe single-objective optimal occupancy measure, we have $V _ { i } ( \theta ^ { P , \nu _ { i } } ) \leq V _ { i } ^ { * }$ Moreover, the definition of $\nu _ { i }$ gives $\theta ^ { P , \nu _ { i } } = ( 1 - \psi _ { i } ) \theta ^ { P , \widetilde { \nu } _ { i } } + \psi _ { i } \theta ^ { P , \pi ^ { \dagger } }$ . Since the cumulative reward over one episode is bounded by H, this implies

$$
V _ { i } ( \theta ^ { P , \widetilde { \nu } _ { i } } ) \leq V _ { i } ( \theta ^ { P , \nu _ { i } } ) + H \psi _ { i } \leq V _ { i } ^ { * } + H \psi _ { i } .
$$

We now compare the fixed optimistic single-objective value $\widetilde { V } _ { i }$ with the true value of ${ \widetilde \nu } _ { i }$ Using $0 \leq \bar { r } _ { i } - r _ { i } \leq 2 \sigma$ , we have $\langle \bar { r } _ { i } , \widetilde { \theta } _ { i } \rangle \leq \langle \bar { r _ { i } } , \widetilde { \theta } _ { i } \rangle + 2 \langle \sigma , \widetilde { \theta } _ { i } \rangle$ . Since $ { \widetilde { \theta } } _ { i } \in \Theta (  { \mathcal { P } } )$ and $r _ { i , h } \le 1$ ， inequality (60) applied to ${ \widetilde \nu } _ { i }$ gives $\langle r _ { i } , \widetilde { \theta } _ { i } \rangle \leq V _ { i } ( \theta ^ { P , \widetilde { \nu } _ { i } } ) + 2 \sqrt { | S | } \langle \zeta , \theta ^ { P , \widetilde { \nu } _ { i } } \rangle$ . Then we have

$$
\widetilde { V } _ { i } \leq V _ { i } ^ { * } + H \psi _ { i } + 2 \langle \sigma , \widetilde { \theta } _ { i } \rangle + 2 \sqrt { | { \cal S } | } \langle \zeta , \theta ^ { P , \widetilde { \nu } _ { i } } \rangle .
$$

Inequality (60) bounds the diference between $\widetilde { \theta } _ { i }$ and the occupancy measure $\theta ^ { P , \widetilde { \nu } _ { i } }$ of ${ \widetilde \nu } _ { i }$ under P. By the construction of $\nu _ { i } ,$ we have $\begin{array} { r } { \theta _ { h } ^ { P , \widetilde { \nu } _ { i } } ( s , a ) \le \frac { 1 } { 1 - \psi _ { i } } \bar { \theta } _ { h } ^ { P , \nu _ { i } } ( s , a ) } \end{array}$ for all $h , s , a .$ and $\begin{array} { r } { \frac { 1 } { 1 - \psi _ { i } } \le \sqrt { \phi _ { \mathrm { s a f e } } } } \end{array}$ by the argument used in the proof of Lemma 35. The resulting $\sigma \mathrm { - }$ and ζ-weighted occupancy measures are therefore bounded by $\sqrt { \phi _ { \mathrm { s a f e } } }$ times the corresponding terms under $\nu _ { i }$ , which are controlled by Lemma 35 applied to the safe policy $\nu _ { i }$ . Thus, we obtain

$$
2 \langle \sigma , \widetilde { \theta } _ { i } \rangle + 2 \sqrt { | { \cal S } | } \langle \zeta , \theta ^ { P , \widetilde { \nu } _ { i } } \rangle \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 3 } | { \cal S } | ^ { 2 } \sqrt { | { \cal A } | / T } ) .
$$

Together with the bound on $\psi _ { i }$ , we obtain $\widetilde { V } _ { i } - V _ { i } ^ { * } \le \widetilde { \mathcal { O } } ( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } )$ . This completes the proof. ■

## F.2 Proof of Theorem 11

Proof We work on the joint event defined at the beginning of this appendix, which holds with probability at least $1 - \delta$ . The bounds of Lemmas 35–39 hold simultaneously on this event. Let <sup>¯</sup>θ be the averaged occupancy measure returned by Algorithm 3. We use the optimal occupancy measures $\theta _ { i } ^ { * } , i \in [ m ]$ , and $\theta _ { \lambda } ^ { * }$ introduced in Section 7.1.

$$
\begin{array} { r l } { \mathrm { S u b o p t } ( \widehat { \pi } ) = \mathrm { S T C H } _ { \lambda } ^ { \theta } ( \sigma ^ { \theta } ) - \mathrm { S T C H } _ { \lambda } ^ { \theta } ( \sigma ^ { \theta } ) } & { } \\ { = \operatorname* { m a x } _ { i \in \Delta } \displaystyle \sum _ { i = 1 } ^ { m _ { \theta } } \Big [ | \pi _ { i } ^ { \lambda } + \iota - V _ { i } ( \widehat { \theta } _ { \lambda } ^ { i } ) \rangle - \mu w _ { i } | \mathrm { l o g } \pi _ { i } | i \Big ] } \\ & { \quad - \operatorname* { m a x } _ { i \in \Delta , \ i = 1 } ^ { m _ { \theta } } \displaystyle \sum _ { i = 1 } ^ { m _ { \theta } } [ w _ { i } \lambda _ { i } ( V _ { i } ^ { \star } + \iota - V _ { i } ( \theta _ { \lambda } ^ { i } ) ) - \mu w _ { i } \mathrm { l o g } \pi _ { i } ] } \\ { \leq \operatorname* { m a x } _ { w \in \Delta } \displaystyle \sum _ { i = 1 } ^ { m _ { \theta } } \Big [ w _ { i } \lambda _ { i } \Big ( \widehat { V } _ { i } + \iota - V _ { i } ( \widehat { \theta } _ { \lambda } ^ { i } ) \Big ) - \mu w _ { i } \mathrm { l o g } \pi _ { i } \Big ] } \\ & { \quad - \operatorname* { m a x } _ { i \in \Delta } \displaystyle \sum _ { i = 1 } ^ { m _ { \theta } } \Big [ w _ { i } \lambda _ { i } \Big ( \widehat { V } _ { i } + \iota - V _ { i } ( \theta _ { \lambda } ^ { i } ) \Big ) - \mu w _ { i } \mathrm { l o g } w _ { i } \Big ] } \\ & { \quad + \widehat { \mathcal { O } } \Big ( \omega _ { \omega \theta \epsilon } H ^ { 4 } \Big \lvert \delta ^ { 2 } \sqrt { i \mathrm { l o g } } + H \lvert \sqrt { \mathcal { H } } \Big \rvert ) . } \end{array}\tag{64}
$$

The first equality holds since ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ and $\pi ^ { \theta _ { \lambda } ^ { * } }$ is an optimal safe policy. The second equality follows from (2), since the inner maximum over $\{ \theta _ { i } \in \Theta _ { \mathrm { s a f e } } \} _ { i = 1 } ^ { m }$ is attained at $\theta _ { i } = \theta _ { i } ^ { * }$ For every w $\in \ \Delta _ { m } .$ , replacing $V _ { i } ^ { * }$ by $\widetilde { V } _ { i }$ changes the objective inside each maximum by $\begin{array} { r } { \sum _ { i = 1 } ^ { m } w _ { i } \lambda _ { i } ( \widetilde { V } _ { i } - V _ { i } ^ { * } ) } \end{array}$ , whose absolute value is at most $\mathrm { m a x } _ { i \in [ m ] } | \widetilde V _ { i } - V _ { i } ^ { * } |$ since $\lambda _ { i } \leq 1$ and $\textstyle \sum _ { i = 1 } ^ { m } w _ { i } = 1$ Thus, each of the two maxima changes by at most this amount, and the inequality follows from Lemma 39.

We now bound the remaining objective diference on the right-hand side of (64). By the definition of ${ \bar { \theta } } ,$ we have $\begin{array} { r } { V _ { i } ( \bar { \theta } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } V _ { i } ( \theta ^ { P , \pi ^ { k + 1 } } ) } \end{array}$ . Therefore, we obtain

$$
\begin{array} { r l } & { \frac { 1 } { \log { 2 \pi } } \displaystyle \sum _ { i = 1 } ^ { N } \left[ w _ { i } \delta \left( \hat { \bar { \xi } } _ { i } + \bar { x } - \bar { y } _ { i } ( \hat { \bar { y } } _ { i } ) \right) - \rho \operatorname* { m i n } \log { \pi } \kappa _ { i } \right] } \\ & { \quad = \lambda \eta \displaystyle \sum _ { i = 1 } ^ { N } \frac { \sum _ { \ell = 1 } ^ { N } \sum _ { \ell = 1 } ^ { N } \Big [ w _ { i } \delta \left( \hat { \bar { \xi } } _ { i } - \bar { x } - \bar { y } _ { i } ( \hat { \bar { y } } _ { i } ) \right) - \rho \operatorname* { m i n } \log { \pi } \kappa _ { i } \Big ] } { \log { \pi } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \end{array}\tag{65}
$$

We first bound Term (I). For each k, by construction, we have $V _ { i } ^ { k + 1 } = \langle \bar { r } _ { i } , \widetilde { \theta } ^ { k + 1 } \rangle$ . Using $0 \leq \bar { r } _ { i } - r _ { i } \leq 2 \sigma$ , inequality (60), and $\theta ^ { P , \pi ^ { k + 1 } } = ( 1 - \psi ^ { k } ) \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } + \psi ^ { \dot { k } } \theta ^ { P , \pi ^ { \dagger } }$ , we have

$$
\left. V _ { i } ^ { k + 1 } - V _ { i } ( \theta ^ { P , \pi ^ { k + 1 } } ) \right. \leq 2 \langle \sigma , \widetilde { \theta } ^ { k + 1 } \rangle + 2 \sqrt { \vert { \cal S } \vert } \langle \zeta , \theta ^ { P , \widetilde { \pi } ^ { k + 1 } } \rangle + { \cal H } \psi ^ { k } .\tag{66}
$$

Averaging (66) over $k = 1 , \ldots , K$ , we use (60) together with $( 1 - \psi ^ { k } ) \theta _ { h } ^ { P , \widetilde { \pi } ^ { k + 1 } } ( s , a ) \ \leq$ $\theta _ { h } ^ { P , \pi ^ { k + 1 } } ( s , a )$ to bound the first two terms by the corresponding fixed bonus terms evaluated under $\theta ^ { P , \pi ^ { k + 1 } }$ . This comparison incurs the factor $\begin{array} { r } { \frac { 1 } { 1 - \psi ^ { k } } \le \sqrt { \phi _ { \mathrm { s a f e } } } , } \end{array}$ as established in the proof of Lemma 35. The last term is controlled by Lemma 38 and dominates the first two. Thus, we have

$$
\mathrm { ( I ) } \le \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { | A | / T } \right) .\tag{67}
$$

We next bound Term (II). The update of $\boldsymbol { w } ^ { k }$ in the planning stage is the same entropic mirror-ascent update as in Lemma 33. Its proof applies with the $T - 1$ online updates replaced by the $K$ planning updates and the normalization $1 / T$ replaced by $1 / K$ . Therefore, we obtain

$$
\mathrm { ( I I ) } \leq \widetilde { \mathcal { O } } \biggl ( \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \biggr ) .\tag{68}
$$

It remains to bound Term (III). Since $\bar { r } _ { i } \geq r _ { i }$ and $ { \widetilde { \theta } } ^ { k + 1 } \geq 0$ , we have $V _ { i } ( \theta _ { \lambda } ^ { * } ) - V _ { i } ^ { k + 1 } =$ $\langle r _ { i } , \theta _ { \lambda } ^ { * } \rangle - \langle \bar { r } _ { i } , \widetilde { \theta } ^ { k + 1 } \rangle \leq \langle r _ { i } , \theta _ { \lambda } ^ { * } - \widetilde { \theta } ^ { k + 1 } \rangle$ . Therefore, we have $\begin{array} { r } { \left( \mathrm { I I I } \right) \le \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \langle \sum _ { i = 1 } ^ { m } \dot { w } _ { i } ^ { k } \lambda _ { i } r _ { i } , \theta _ { \lambda } ^ { * } } \end{array}$

$\widetilde { \theta } ^ { k + 1 } \rangle$ . Since $\theta _ { \lambda } ^ { * } \in \Theta _ { \mathrm { s a f e } }$ , applying Lemma 37 with $\theta = \theta _ { \lambda } ^ { * }$ gives

$$
( \mathrm { I I I } ) \leq \widetilde { \mathcal { O } } \Bigl ( H / \sqrt { K } + \phi _ { \mathrm { s a f e } } H ^ { 3 } | S | ^ { 2 } \sqrt { | A | / T } \Bigr ) .\tag{69}
$$

Combining (64), (65), (67), (68), and (69), we obtain

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \widetilde { \mathcal { O } } \left( \phi _ { \mathrm { s a f e } } H ^ { 4 } | S | ^ { 2 } \sqrt { \frac { | A | } { T } } + \frac { H } { \sqrt { K } } + \frac { H ^ { 2 } } { \mu K } + \frac { \mu } { K } \right) .
$$

This completes the proof.

## Appendix G. Proof of Theorem 12

Proof For the exploration stage, let $\mathcal { E } _ { C } ( \delta )$ and $\mathcal { E } _ { T } ( \delta )$ be the online cost and transition confidence events defined at the beginning of Appendix D, applied to the single-rollout estimates constructed by Algorithm 2. For the planning stage, let $\mathcal { E } _ { \mathrm { f i x } } ( \delta )$ be the confidence event defined at the beginning of Appendix F. We define the joint confidence event $\mathcal { E } _ { C , T , \mathrm { f i x } } ( \delta ) : = \mathcal { E } _ { C } ( \delta / 3 ) \cap \mathcal { E } _ { T } ( \delta / 3 ) \cap \mathcal { E } _ { \mathrm { f i x } } ( \delta / 3 )$ . By the confidence guarantees established in Appendices D and $\mathrm { F } , \mathcal { E } _ { C , T , \mathrm { f i x } } ( \delta )$ holds with probability at least $1 - \delta$ . Throughout the proof, we work on this event. In particular, we have $| \widehat { C } ^ { t } - C | \leq \Xi ^ { t } , P \in \mathcal { P } ^ { t } , \forall t \in [ T - 1 ]$ , and $| \widehat { C } - C | \leq \Xi , P \in \mathcal { P }$

We first consider the exploration stage. Algorithm 2 constructs its initial policy as in Algorithm 1. The initial policy argument in the proof of Theorem 10 therefore gives $C ^ { \top } \theta ^ { { \tilde { P , } } \pi ^ { 1 } } \leq \tau$ . For each $t \in [ T - 1 ]$ , the exploration reward in Algorithm 2 afects only the update of $\widehat { \theta } ^ { t + 1 }$ . Therefore, the argument in the proof of Theorem 10, applied with the online confidence bounds, gives $C ^ { \top } \theta ^ { P , \pi ^ { t } } \leq \tau , \forall t \in [ T ]$

We next consider the planning stage. For every $k \in [ K ]$ , the main planning update uses SafeProj with the fixed quantities $( \tilde { C } , \Xi , \mathcal { P } )$ . On $\mathcal { E } _ { C , T , \mathrm { f i x } } ( \dot { \delta } ) , | \widehat { C } - C | \leq \Xi$ and $P \in \mathcal { P }$ hold at every planning iteration. The argument in the proof of Theorem 10 therefore applies to each main planning policy. We obtain $C ^ { \top } \theta ^ { P , \pi ^ { k + 1 } } \leq \tau , \forall k \in [ K ]$

Finally, we show that the returned policy is safe. Since $\theta ^ { P , \pi ^ { k + 1 } } \in \Theta _ { \mathrm { s a f e } }$ for every $k \in [ K ]$ and $\Theta _ { \mathrm { s a f e } }$ is convex, the returned occupancy measure satisfies $\begin{array} { r } { \bar { \theta } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \theta ^ { P , \pi ^ { k + 1 } } \in \bar { \Theta _ { \mathrm { s a f e } } } } \end{array}$ Let ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ be the induced Markov policy. By the correspondence between policies and occupancy measures established in Appendix A, we have $\theta ^ { P , \widehat { \pi } } = \bar { \theta }$ , and therefore Violation(π) = 0. When $d = 1$ , these inequalities guarantee that each constraint is satisfied at the time step to which it applies. When $d = H$ , they guarantee that each cumulative constraint is satisfied over the entire episode. This completes the proof.

## Appendix H. Proofs for the Unknown Constraint Cost Extensions

Lemma 40 Suppose Assumptions 2 and 13 hold and $T > 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . Then, with probability at least $1 - \delta / 2$ , we have $t _ { s } \le 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 } < T , \ \gamma _ { j } \le \widehat { \gamma } _ { j } \le$ $( \gamma _ { j } + \tau _ { j } ) / 2 < \tau _ { j }$ for every $j \in [ n ]$ , and $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$

Proof Recall the empirical costs $\bar { \gamma } _ { j } ^ { t }$ , the stopping time $t _ { s } ,$ , and the estimate $\widehat { \gamma }$ defined in Section 6.1. By Assumption $2 ,$ we have $\varrho ^ { \dagger } > 0$ . For each $j \in [ n ]$ , the total cost observed in one episode belongs to $[ 0 , d ]$

By Hoefding’s inequality and a union bound over $j \in [ n ]$ and $t \in [ T ]$ , with probability at least $1 - \delta / 2$ , we have

$$
| \bar { \gamma } _ { j } ^ { t } - \gamma _ { j } | \leq d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 t } } \quad \mathrm { f o r ~ a l l ~ } j \in [ n ] \mathrm { ~ a n d ~ } t \in [ T ] .\tag{70}
$$

We work on this event throughout the proof.

To bound $t _ { s }$ , we may assume $t _ { s } \geq 2$ , since the stated bound is immediate otherwise. By the definition of $t _ { s } .$ , the stopping rule is not triggered at episode $t _ { s } - 1$ . Then, for some $j \in [ n ]$ , we have $\begin{array} { r } { \tau _ { j } - \bar { \gamma } _ { j } ^ { t _ { s } - 1 } < 3 d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 ( t _ { s } - 1 ) } } } \end{array}$ . Using (70) at $t = t _ { s } - 1$ , we obtain

$$
\varrho ^ { \dagger } \leq \tau _ { j } - \gamma _ { j } \leq \tau _ { j } - \bar { \gamma } _ { j } ^ { t _ { s } - 1 } + d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 ( t _ { s } - 1 ) } } < 4 d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 ( t _ { s } - 1 ) } } .
$$

Solving this inequality for $t _ { s } - 1$ yields $t _ { s } \le 1 + 8 d ^ { 2 } \ln ( 4 n T / \delta ) / ( \varrho ^ { \dagger } ) ^ { 2 }$ . By the assumed lower bound on $T$ , this bound implies $t _ { s } < T$ . The stopping condition therefore holds at $t _ { s }$

By the definition of $\widehat { \gamma } _ { j }$ , the stopping condition at $t _ { s } .$ , and (70) at $t = t _ { s } .$ , we have

$$
\gamma _ { j } \leq \widehat { \gamma } _ { j } = \frac { 1 } { 2 } \Bigg ( \bar { \gamma } _ { j } ^ { t s } - d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 t _ { s } } } \Bigg ) + \frac { 1 } { 2 } \Bigg ( \bar { \gamma } _ { j } ^ { t _ { s } } + 3 d \sqrt { \frac { \ln ( 4 n T / \delta ) } { 2 t _ { s } } } \Bigg ) \leq \frac { \gamma _ { j } + \tau _ { j } } { 2 } .
$$

This implies $\tau _ { j } - \widehat { \gamma } _ { j } \geq ( \tau _ { j } - \gamma _ { j } ) / 2$ , and therefore $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$

The proofs of Theorems 9-12 use $\gamma$ only as an upper bound on $C ^ { \top } \theta ^ { P , \pi ^ { \dagger } }$ that is strictly below τ. They therefore remain valid with $\widehat { \gamma }$ in place of $\gamma$ and $\widehat { \phi } _ { \mathrm { s a f e } }$ in place of $\phi _ { \mathrm { s a f e } }$ whenever $\gamma \le \widehat { \gamma } < \tau$

## H.1 Proof of Theorem 14

Proof By Lemma 40, with probability at least $1 - \delta / 2$ , we have $t _ { s } < T , \gamma \leq \widehat { \gamma } < \tau$ , and $\widehat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$ . Conditioned on any estimation-phase history in this event, $t _ { s }$ and $\widehat { \gamma }$ are fixed, and the proof of Theorem 9 applies to the remaining $T - t _ { s }$ episodes with $\widehat { \gamma }$ in place of $\gamma$ and $\delta$ replaced by $\delta / 2$ . This gives the bound in Theorem 14 with conditional probability at least $1 - \delta / 2$ , and a union bound completes the proof.

## H.2 Proof of Theorem 15

Proof By Lemma 40, with probability at least $1 - \delta / 2$ , we have $t _ { s } < T$ and $\gamma \le \widehat { \gamma } < \tau$ Every policy executed during the estimation phase is $\pi ^ { \dagger }$ and is therefore safe. Conditioned on any estimation-phase history in this event, $t _ { s }$ and $\widehat { \gamma }$ are fixed, and the proof of Theorem 10 applies to the remaining $T - t _ { s }$ episodes with $\widehat { \gamma }$ in place of $\gamma$ and $\delta$ replaced by $\delta / 2$

Thus, with conditional probability at least $1 - \delta / 2$ , every policy executed in CSTRL is safe and Violation $( { \widehat { \pi } } ) = 0 .$ , and a union bound completes the proof.

## H.3 Proof of Theorem 16

Proof By Lemma 40, with probability at least $1 - \delta / 2$ , we have $t _ { s } < T , \gamma \leq \widehat { \gamma } < \tau .$ and $\hat { \phi } _ { \mathrm { s a f e } } \leq 4 \phi _ { \mathrm { s a f e } }$ . Conditioned on any estimation-phase history in this event, $t _ { s }$ and $\widehat { \gamma }$ are fixed, and the proof of Theorem 11 applies to the remaining $T - t _ { s }$ exploration episodes and K planning iterations with $\widehat { \gamma }$ in place of $\gamma$ and $\delta$ replaced by $\delta / 2$ . This gives the bound in Theorem 16 with conditional probability at least $1 - \delta / 2$ , and a union bound completes the proof. 7

## H.4 Proof of Theorem 17

Proof By Lemma 40, with probability at least $1 - \delta / 2$ , we have $t _ { s } < T$ and $\gamma \le \widehat { \gamma } < \tau$ . Every policy executed during the estimation phase is $\pi ^ { \dagger }$ and is therefore safe. Conditioned on any estimation-phase history in this event, $t _ { s }$ and $\widehat { \gamma }$ are fixed, and the proof of Theorem 12 applies with $\widehat { \gamma }$ in place of $\gamma$ and δ replaced by $\delta / 2$ . Thus, with conditional probability at least $1 - \delta / 2$ , every policy executed during preference-free exploration and every main policy generated during planning is safe, and $\mathrm { V i o l a t i o n } ( \widehat { \pi } ) = 0$ . A union bound completes the proof.

## Appendix I. Proofs for the Unknown Safe Policy Extensions

We give the procedure for CSTRL-US in Algorithm 4.

Algorithm 4 CSTRL-US: CSTRL with an Unknown Safe Policy   
Require: $\lambda , \mu , \mathsf { A 1 g } ^ { P } , \mathsf { A 2 g } ^ { D }$   
1: Set $t = 0$   
2: while $t < t _ { s }$ do   
3: Set $t = t + 1$   
4: Obtain $\pi ^ { t }$ from $\mathsf { A l g } ^ { P }$ and $\phi ^ { t } \in \Delta _ { n }$ from $\mathsf { A l g } ^ { D }$   
5: Execute $\pi ^ { t }$ and observe $\{ ( s _ { h } ^ { t } , a _ { h } ^ { t } , \{ c _ { j , h } ^ { t } \} _ { j = 1 } ^ { n } ) \} _ { h = 1 } ^ { H }$   
6: Execute $\mathsf { A l g } ^ { P }$ with $\{ ( s _ { h } ^ { t } , a _ { h } ^ { t } , \ell _ { h } ^ { t } ) \} _ { h = 1 } ^ { H }$   
7: Execute $\mathsf { A l g } ^ { D }$ with $\varrho ^ { t }$   
8: end while   
9: Compute $\begin{array} { r } { \widehat { \varrho } : = \frac { 1 } { t _ { s } } \operatorname* { m i n } _ { j \in [ n ] } \sum _ { t = 1 } ^ { t _ { s } } \varrho _ { j } ^ { t } - \frac { 2 d } { t _ { s } } \sqrt { 2 t _ { s } \ln \frac { 2 4 n t _ { s } ^ { 2 } } { \delta } } } \end{array}$   
10: Let $\widehat { \pi } ^ { \dagger }$ select $\pi ^ { t } ~ \mathrm { w . p . } ~ 1 / t _ { s }$ for each $t \in [ t _ { s } ]$   
11: Set ${ \widehat { \gamma } } : = ( \widehat \gamma _ { 1 } , \ldots , \widehat \gamma _ { n } )$ , where $\widehat { \gamma } _ { j } : = \tau _ { j } - \widehat { \varrho }$ for all $j \in [ n ]$   
12: Run CSTRL for $T - t _ { s }$ episodes using $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$   
13: Return: <sup>¯</sup>θ

## I.1 Lemmas

Lemma 41 Let $\mathcal { E } _ { M } ( \delta )$ denote the event that the following inequalities hold simultaneously for every $t \geq 1$

$$
\left| \sum _ { k = 1 } ^ { t } \sum _ { j = 1 } ^ { n } \phi _ { j } ^ { k } \left[ \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { k } - \left( C ^ { \top } \theta ^ { P , \pi ^ { k } } \right) _ { j } \right] \right| \leq d { \sqrt { 2 t \ln { \frac { 8 n t ^ { 2 } } { \delta } } } } ,\tag{71}
$$

$$
\left| \sum _ { k = 1 } ^ { t } \left[ \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { k } - \left( C ^ { \top } \theta ^ { P , \pi ^ { k } } \right) _ { j } \right] \right| \leq d \sqrt { 2 t \ln \frac { 8 n t ^ { 2 } } { \delta } } , \quad \forall j \in [ n ] .\tag{72}
$$

Then $\Pr ( \mathcal { E } _ { M } ( \delta ) ) \geq 1 - \delta$

Proof Let $( \mathcal { F } _ { k } ) _ { k \ge 0 }$ be the filtration generated by the history up to the end of episode k. For each $k \geq 1$ , we define $\begin{array} { r } { X ^ { k } : = \sum _ { j = 1 } ^ { n } \phi _ { j } ^ { k } [ \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { k } - ( C ^ { \top } \theta ^ { P , \pi ^ { k } } ) _ { j } ] } \end{array}$ . Since $\phi ^ { k } \in \Delta _ { n }$ and $\pi ^ { k }$ are $\mathcal { F } _ { k - 1 }$ -measurable, the law of iterated expectations and the observation model give $\mathbb { E } [ \sum _ { h = 1 } ^ { H } \overset { \cdot } { c } _ { j , h } ^ { k } | \mathcal { F } _ { k - 1 } ] = ( C ^ { \top } \theta ^ { P , \pi ^ { k } } ) _ { j }$ . Moreover, since $c _ { j , h } ^ { k } \in [ 0 , 1 ]$ and $\phi ^ { k }$ lies in the simplex, we have $| X ^ { k } | \leq$ d almost surely. For each fixed $t \geq 1$ , the Azuma-Hoefding inequality gives $\begin{array} { r } { \operatorname* { P r } ( | \sum _ { k = 1 } ^ { t } X ^ { k } | > d \sqrt { 2 t \ln \frac { 8 n t ^ { 2 } } { \delta } } ) \leq \frac { \delta } { 4 n t ^ { 2 } } } \end{array}$ . Taking a union bound over all $t \geq 1$ , we conclude that (71) holds simultaneously for every $t \geq 1$ with probability at least $\begin{array} { r } { 1 - \sum _ { t = 1 } ^ { \infty } \frac { \delta } { 4 n t ^ { 2 } } \geq } \end{array}$ $\textstyle 1 - { \frac { \delta } { 2 n } }$ , where we use $\textstyle \sum _ { t = 1 } ^ { \infty } t ^ { - 2 } \leq 2$

Next, fix $j \in [ n ]$ and define $\begin{array} { r } { Y _ { j } ^ { k } : = \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { k } - ( C ^ { \top } \theta ^ { P , \pi ^ { k } } ) _ { j } } \end{array}$ . By the same argument, $( Y _ { j } ^ { k } ) _ { k \geq 1 }$ is a martingale diference sequence with respect to $( \mathcal { F } _ { k } ) _ { k \ge 0 }$ , and $| Y _ { i } ^ { k } | \leq d .$ For each fixed $t \geq 1$ , applying the Azuma-Hoefding inequality to $( Y _ { j } ^ { k } ) _ { k \geq 1 }$ gives $\begin{array} { r } { \operatorname* { P r } ( | \sum _ { k = 1 } ^ { t } Y _ { j } ^ { k } | > } \end{array}$ $\begin{array} { r } { d \sqrt { 2 t \ln \frac { 8 n t ^ { 2 } } { \delta } } ) \leq \frac { \delta } { 4 n t ^ { 2 } } } \end{array}$ . Taking a union bound over all $j \in [ n ]$ and $t \geq 1$ shows that (72) holds simultaneously for every $j \in [ n ]$ and $t \geq 1$ with probability at least $\begin{array} { r } { \mathrm { ~ . ~ - ~ } \sum _ { j = 1 } ^ { n } \sum _ { t = 1 } ^ { \infty } \frac { \delta } { 4 n t ^ { 2 } } \geq } \end{array}$ $\textstyle 1 - { \frac { \delta } { 2 } }$ , where we use $\textstyle \sum _ { t = 1 } ^ { \infty } t ^ { - 2 } \leq 2$

By the definition of $\mathcal { E } _ { M } ( \delta )$ , the preceding probability bounds and a union bound give $\Pr ( \mathcal E _ { M } ( \delta ) ) \geq 1 - \delta$ . This completes the proof.

Lemma 42 Let $\mathcal { E } _ { P } ( \delta )$ be the event on which the regret bound of $\mathsf { A l g } ^ { P }$ in Section 6.2 holds. On $\mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 )$ , for every $t \geq 1$ , we have

$$
\operatorname* { m i n } _ { j \in [ n ] } \sum _ { k = 1 } ^ { t } \varrho _ { j } ^ { k } \ge t \varrho - \mathcal { O } \Big ( H ^ { 2 } | \mathcal { S } | \sqrt { | \boldsymbol { A } | t \ln ( 2 H | \mathcal { S } | | \boldsymbol { A } | t / \delta ) } + d \sqrt { n t } \Big ) - d \sqrt { 2 t \ln \frac { 2 4 n t ^ { 2 } } { \delta } } ,\tag{73}
$$

where the $\mathcal { O } ( \cdot )$ term is the sum of the regret bounds of $\mathsf { A l g } ^ { P }$ and $\mathsf { A l g } ^ { D }$

Proof Fix any $t \geq 1$ and $\pi \in \Pi$ . For each episode k, we define $\bar { \varrho } ^ { k } \in \mathbb { R } ^ { n }$ by $\bar { \varrho } _ { j } ^ { k } : = \quad$ $\tau _ { j } - ( C ^ { \top } \theta ^ { P , \pi ^ { k } } ) _ { j }$ for every $j \in [ n ]$ . By the definition of $\bar { \ell } ^ { k }$ in Section 6.2 and $\phi ^ { k } \in \Delta _ { n }$ , we have $\langle \phi ^ { k } , \bar { \varrho } ^ { k } \rangle = - \bar { \ell } ^ { k } ( \pi ^ { k } )$ and − $\begin{array} { r } { \bar { \ell } ^ { k } ( \pi ) \ge \operatorname* { m i n } _ { j \in [ n ] } \{ \tau _ { j } - ( C ^ { \top } \theta ^ { P , \pi } ) _ { j } \} } \end{array}$ . The primal regret bound

on $\mathcal { E } _ { P } ( \delta / 3 )$ therefore gives

$$
\sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \bar { \varrho } ^ { k } \rangle \geq t \operatorname* { m i n } _ { j \in [ n ] } \{ \tau _ { j } - ( C ^ { \top } \theta ^ { P , \pi } ) _ { j } \} - \mathcal { O } \big ( H ^ { 2 } | S | \sqrt { | A | t \ln ( 2 H | S | | A | t / \delta ) } \big ) .
$$

Since $\pi \in \Pi$ is arbitrary, the definition of $\varrho \textbf { J }$ yields

$$
\sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \bar { \varrho } ^ { k } \rangle \geq t \varrho - \mathcal { O } \big ( H ^ { 2 } | S | \sqrt { | A | t \ln ( 2 H | S | | A | t / \delta ) } \big ) .\tag{74}
$$

Next, on $\mathcal { E } _ { M } ( \delta / 3 )$ , Lemma 41 gives

$$
\sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \varrho ^ { k } \rangle \geq \sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \bar { \varrho } ^ { k } \rangle - d \sqrt { 2 t \ln \frac { 2 4 n t ^ { 2 } } { \delta } } .\tag{75}
$$

Combining (74) and (75) gives

$$
\sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \varrho ^ { k } \rangle \geq t \varrho - \mathcal { O } \big ( H ^ { 2 } | \mathcal { S } | \sqrt { | { \cal A } | t \ln ( 2 H | \mathcal { S } | | { \cal A } | t / \delta ) } \big ) - d \sqrt { 2 t \ln \frac { 2 4 n t ^ { 2 } } { \delta } } .\tag{76}
$$

We next derive a lower bound on min $\scriptstyle \cdot j \in [ n ] \sum _ { k = 1 } ^ { t } \varrho _ { j } ^ { k }$ . By the dual regret bound and linearity over the simplex, we have

$$
\sum _ { k = 1 } ^ { t } \langle \phi ^ { k } , \varrho ^ { k } \rangle \leq \operatorname* { m i n } _ { \phi \in \Delta _ { n } } \sum _ { k = 1 } ^ { t } \langle \phi , \varrho ^ { k } \rangle + \mathcal { O } ( d \sqrt { n t } ) = \operatorname* { m i n } _ { j \in [ n ] } \sum _ { k = 1 } ^ { t } \varrho _ { j } ^ { k } + \mathcal { O } ( d \sqrt { n t } ) .
$$

Combining this inequality with (76) gives (73). This completes the proof.

Lemma 43 Suppose that T satisfies the condition in Section 6.2. On $\mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 )$ we have $t _ { s } < T$ and $t _ { s } = \widetilde { \mathcal { O } } ( ( H ^ { 4 } | S | ^ { 2 } | A | + n d ^ { 2 } ) / \varrho ^ { 2 } )$

Proof Let $t \geq 1$ be the smallest integer for which $t \varrho$ is at least twice the stopping threshold. The condition on T in Section 6.2 means that $T > t .$ . By Lemma 42, we have

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { j \in [ n ] } \sum _ { k = 1 } ^ { t } \varrho _ { j } ^ { k } \ge t \varrho - \mathcal { O } \Big ( H ^ { 2 } | \mathcal { S } | \sqrt { | A | t \ln ( 2 H | \mathcal { S } | | A | t / \delta ) } + d \sqrt { n t } \Big ) - d \sqrt { 2 t \ln \frac { 2 4 n t ^ { 2 } } { \delta } } } \\ { \displaystyle \qquad \ge t \varrho - \mathcal { O } \Big ( ( H ^ { 2 } | \mathcal { S } | \sqrt { | A | } + d \sqrt { n } ) \sqrt { t \ln ( 2 n H | \mathcal { S } | | A | t / \delta ) } \Big ) } \\ { \displaystyle \qquad \ge \mathcal { O } \Big ( ( H ^ { 2 } | \mathcal { S } | \sqrt { | A | } + d \sqrt { n } ) \sqrt { t \ln ( 2 n H | \mathcal { S } | | A | t / \delta ) } \Big ) , } \end{array}
$$

where the $\mathcal { O } ( \cdot )$ terms in the last two lines denote the stopping threshold at episode t. The second inequality holds since the constant hidden in the stopping threshold is chosen suficiently large, and the last inequality follows from the choice of t. Therefore, the inequality

in the definition of $t _ { s }$ holds at episode $t ,$ which gives $t _ { s } \leq t < T . { \mathrm { ~ I f ~ } } t \geq 2$ , the minimality of t gives

$$
\sqrt { t - 1 } \varrho \le \mathcal { O } \Big ( ( H ^ { 2 } | S | \sqrt { | A | } + d \sqrt { n } ) \sqrt { \ln ( 2 n H | S | | A | t / \delta ) } \Big ) .
$$

Solving this inequality for t yields $t = \widetilde { \mathcal { O } } ( ( H ^ { 4 } | S | ^ { 2 } | A | + n d ^ { 2 } ) / \varrho ^ { 2 } )$ , which also holds trivially when $t = 1$ . Since $t _ { s } \leq t$ , this completes the proof. ■

Lemma 44 Suppose that T satisfies the condition in Section 6.2. On $\mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 )$ , the estimate $\widehat { \varrho }$ satisfies

$$
\widehat { \varrho } \leq \operatorname* { m i n } _ { j \in [ n ] } \left\{ \tau _ { j } - \left( C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \right) _ { j } \right\} \leq \varrho .\tag{77}
$$

Moreover, on the same event, we have ${ \widehat { \varrho } } \geq \varrho / 2 > 0$ and $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$

Proof By definition, we have $\begin{array} { r } { \widehat { \varrho } = \frac { 1 } { t _ { s } } \operatorname* { m i n } _ { j \in [ n ] } \sum _ { t = 1 } ^ { t _ { s } } \varrho _ { j } ^ { t } - \frac { 2 d } { t _ { s } } \sqrt { 2 t _ { s } \ln \frac { 2 4 n t _ { s } ^ { 2 } } { \delta } } } \end{array}$ . Applying (72) on $\mathcal { E } _ { M } ( \delta / 3 )$ to each $j \in [ n ]$ , we obtain $\begin{array} { r } { \sum _ { t = 1 } ^ { t _ { s } } ( \sum _ { h = 1 } ^ { H } c _ { j , h } ^ { t } - \tau _ { j } ) \ge \sum _ { t = 1 } ^ { t _ { s } } [ ( C ^ { \top } \theta ^ { P , \pi ^ { t } } ) _ { j } - \tau _ { j } ] - } \end{array}$ $\begin{array} { r } { d \sqrt { 2 t _ { s } \ln { \frac { 2 4 n t _ { s } ^ { 2 } } { \delta } } } } \end{array}$ . Combining this inequality with the definition of $\widehat { \varrho }$ gives $\begin{array} { r } { \widehat { \varrho } \leq \operatorname* { m i n } _ { j \in [ n ] } \left\{ \tau _ { j } - \right. } \end{array}$ $\begin{array} { r } { \frac { 1 } { t _ { s } } \dot { \sum } _ { t = 1 } ^ { t _ { s } } ( C ^ { \top } \theta ^ { P , \pi ^ { t } } ) _ { j } \} } \end{array}$ . Since $\widehat { \pi } ^ { \dagger }$ selects $\pi ^ { t }$ with probability $1 / t _ { s }$ for each $t \in [ t _ { s } ]$ , its occupancy measure satisfies $\begin{array} { r } { \theta ^ { P , \widehat { \pi } ^ { \dagger } } = \frac { 1 } { t _ { c } } \sum _ { t = 1 } ^ { t _ { s } } \theta ^ { P , \pi ^ { t } } } \end{array}$ . Therefore, we obtain $\begin{array} { r } { \widehat { \varrho } \leq \operatorname* { m i n } _ { j \in [ n ] } \{ \tau _ { j } - ( C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } ) _ { j } \} } \end{array}$ The second inequality in (77) follows immediately from the definition of $\varrho .$ It remains to prove that $\widehat { \varrho } \geq \varrho / 2$ . By Lemma 43, we have $t _ { s } < T$ , and thus the inequality in the definition of $t _ { s }$ holds at episode $t _ { s }$ . Lemma 42 and the definition of $\widehat { \varrho }$ therefore give

$$
\begin{array} { r l r } {  { t _ { s } \varrho \le \operatorname* { m i n } _ { j \in [ n ] } \displaystyle \sum _ { k = 1 } ^ { t _ { s } } \varrho _ { j } ^ { k } + \mathcal { O } \Big ( H ^ { 2 } | \mathcal { S } | \sqrt { | \mathcal { A } | t _ { s } \ln ( 2 H | \mathcal { S } | | \mathcal { A } | t _ { s } / \delta ) } + d \sqrt { n t _ { s } } \Big ) + d \sqrt { 2 t _ { s } \ln \frac { 2 4 n t _ { s } ^ { 2 } } { \delta } } } } \\ & { } & { \le 2 \operatorname* { m i n } _ { j \in [ n ] } \displaystyle \sum _ { k = 1 } ^ { t _ { s } } \varrho _ { j } ^ { k } - 4 d \sqrt { 2 t _ { s } \ln \frac { 2 4 n t _ { s } ^ { 2 } } { \delta } } = 2 t _ { s } \widehat { \varrho } . } \end{array}
$$

The second inequality follows from the stopping condition at $t _ { s } ,$ since the constant hidden in the stopping threshold is chosen suficiently large. Dividing both sides by $2 t _ { s }$ yields $\widehat { \varrho } \geq \varrho / 2$ . Since $\widehat { \gamma } _ { j } = \tau _ { j } - \widehat { \varrho }$ for every $j \in [ n ]$ , the first inequality in (77) gives $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } .$ Moreover, $\widehat { \varrho } > 0$ implies $\widehat { \gamma } < \tau$ . This completes the proof.

## I.2 Proof of Theorem 18

Proof Consider the event $\mathcal { E } : = \mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 )$ . Since $\mathsf { A l g } ^ { P }$ is run with $\delta / 3$ in place of $\delta ,$ a union bound gives $\operatorname* { P r } ( \mathcal { E } ) \geq 1 - 2 \delta / 3$ . By the condition on $T$ in Section 6.2 and Lemma 43, we have $T > t _ { s }$ on $\mathcal { E } .$ . Thus, $T - t _ { s } \ge 1$ . On the same event, Lemma 44 yields $\widehat { \varrho } \geq \varrho / 2 > 0$ and $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$ . Moreover, since $\widehat { \gamma } _ { j } = \tau _ { j } - \widehat { \varrho }$ for every $j \in [ n ]$ , we have $\begin{array} { r } { \widehat { \phi } _ { \mathrm { s a f e } } = \operatorname* { m a x } _ { j \in [ n ] } \frac { d ^ { 2 } } { ( \tau _ { j } - \widehat \gamma _ { j } ) ^ { 2 } } = \frac { d ^ { 2 } } { \widehat { \varrho } ^ { 2 } } \leq \frac { 4 d ^ { 2 } } { \varrho ^ { 2 } } } \end{array}$ . Fix any estimation-phase history on which E holds. Then the pair $( \widehat { \pi } ^ { \dagger } , \widehat { \gamma } )$ is fixed. After episode $t _ { s } .$ , Algorithm 4 runs CSTRL for $T - t _ { s }$ episodes using this pair. The proof of Theorem 9 therefore applies with $\delta$ replaced by $\delta / 3$ $T$ by $T - t _ { s }$ , and $\phi _ { \mathrm { s a f e } }$ by $\phi _ { \mathrm { s a f e } }$ . This gives the bound in Theorem 18 with conditional probability at least $1 - \delta / 3$ . Combining this conditional guarantee with the estimation event gives an overall probability of at least $1 - \delta .$ . This proves Theorem 18.

## I.3 Proof of Theorem 19

Proof Let $\mathscr { E } : = \mathscr { E } _ { P } ( \delta / 3 ) \cap \mathscr { E } _ { M } ( \delta / 3 ) \cap \mathscr { E } _ { R , C , T } ( \delta / 3 )$ . Since $\mathsf { A l g } ^ { P }$ is run with $\delta / 3$ in place of $\delta ,$ the first two events each hold with probability at least $1 - \delta / 3$ . For any fixed history of the estimation phase, the subsequent CSTRL run uses $\delta / 3$ in place of $\delta ,$ and thus $\mathcal { E } _ { R , C , T } ( \delta / 3 )$ holds with conditional probability at least $1 - \delta / 3$ . A union bound therefore gives $\Pr ( \xi ) \geq$ $1 - \delta$ . It thus sufices to prove the result on $\mathcal { E } .$

By the condition on $T$ in Section 6.2 and Lemma $4 3 .$ we have $T > t _ { s }$ on $\mathcal { E }$ . On the same event, Lemma 44 gives $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$ . We now bound the cumulative violation over all episodes. During the estimation phase, the policy executed in each episode may be unsafe. Since every constraint value is at most $d$ and $\tau _ { j } \geq 0$ for every $j \in [ n ]$ , we have Violation $( \pi ^ { t } ) \leq d .$ . Therefore, we have $\textstyle \sum _ { t = 1 } ^ { t _ { s } }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$

After episode $t _ { s } ,$ the algorithm runs CSTRL for the remaining $T - t _ { s }$ episodes with $\widehat { \pi } ^ { \dagger }$ and $\widehat { \gamma }$ . Given the estimation-phase history, the pair $( \widehat { \pi } ^ { \dagger } , \widehat { \gamma } )$ is fixed. On $\mathcal { E } .$ , the argument in the proof of Theorem 10 therefore applies with $\widehat { \gamma }$ as a conservative upper bound on the costs of $\widehat { \pi } ^ { \dagger }$ . Consequently, every policy executed in CSTRL is safe, which gives $\textstyle \sum _ { t = t _ { s } + 1 } ^ { T }$ Violation $( \pi ^ { t } ) = 0$ . Combining the preceding bounds gives $\scriptstyle \sum _ { t = 1 } ^ { T }$ Violation $( \pi ^ { t } ) =$ $\textstyle \sum _ { t = 1 } ^ { t _ { s } }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$ . Moreover, Lemma 43 shows that $t _ { s } = \widetilde { \mathcal { O } } ( ( H ^ { 4 } | S | ^ { 2 } | A | + n d ^ { 2 } ) / \varrho ^ { 2 } )$ on $\mathcal { E }$ . This upper bound is independent of $T$

Finally, we have $\theta ^ { P , \pi ^ { t } } \in \Theta _ { \mathrm { s a f e } }$ for every $t \in [ t _ { s } + 1 , T ]$ . Since $\Theta _ { \mathrm { s a f e } }$ is convex, the <sup>¯</sup>θ returned by CSTRL also belongs to $\Theta _ { \mathrm { s a f e } }$ . Let ${ \widehat { \pi } } = \pi ^ { \bar { \theta } }$ . By the correspondence between policies and occupancy measures established in Appendix A, we have $\theta ^ { \bar { P } , \widehat { \pi } } = \bar { \theta }$ . Therefore, we obtain Violation $( \widehat { \pi } ) = 0$ . This completes the proof.

## I.4 Proof of Theorem 20

Proof Let $\mathcal { E } : = \mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 )$ . Since $\mathsf { A l g } ^ { P }$ is run with $\delta / 3$ in place of $\delta ,$ a union bound gives $\operatorname* { P r } ( \mathcal { E } ) \geq 1 - 2 \delta / 3$ . By the condition on $T$ in Section 6.2 and Lemma 43, we have $T > t _ { s }$ on E. On the same event, Lemma 44 gives $\widehat { \varrho } \geq \varrho / 2 > 0$ and $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$ Fix any estimation-phase history on which E holds. Then the pair $( \widehat { \pi } ^ { \dagger } , \widehat { \gamma } )$ is fixed. The algorithm runs Algorithms 2 and 3 for $T - t _ { s }$ exploration episodes and K planning iterations, respectively, using this pair. The proof of Theorem 11 therefore applies with $\delta$ replaced by $\delta / 3$ , T by $T - t _ { s }$ , and $\phi _ { \mathrm { s a f e } }$ by $\widehat { \phi } _ { \mathrm { s a f e } } = d ^ { 2 } / \widehat { \varrho } ^ { 2 } \leq 4 d ^ { 2 } / \varrho ^ { 2 }$ . This gives the bound in Theorem 20 with conditional probability at least $1 - \delta / 3$ . Combining this conditional guarantee with the estimation event gives an overall probability of at least 1−δ. This proves Theorem 20.

## I.5 Proof of Theorem 21

Proof Let $\mathcal { E } : = \mathcal { E } _ { P } ( \delta / 3 ) \cap \mathcal { E } _ { M } ( \delta / 3 ) \cap \mathcal { E } _ { C , T , \mathrm { f x } } ( \delta / 3 )$ . Since $\mathsf { A l g } ^ { P }$ is run with $\delta / 3$ in place of $\delta ,$ the first two events each hold with probability at least $1 - \delta / 3$ . For any fixed history of the estimation phase for which both events hold, PF-CSTRL is run with $\delta / 3$ in place of $\delta ,$ and thus $\mathcal { E } _ { C , T , \mathrm { f i x } } ( \delta / 3 )$ holds with conditional probability at least $1 - \delta / 3$ for the subsequent $T - t _ { s }$ exploration episodes and K planning iterations. A union bound therefore gives $\operatorname* { P r } ( { \mathcal { E } } ) \geq 1 - \delta$ Lemma 43 gives $t _ { s } = \widetilde { \mathcal { O } } ( ( H ^ { 4 } | S | ^ { 2 } | A | + n d ^ { 2 } ) / \varrho ^ { 2 } )$ on $\mathcal { E }$ . Each policy executed during the estimation phase has violation at most $d ,$ and therefore $\textstyle \sum _ { t = 1 } ^ { t _ { s } }$ Violation $( \pi ^ { t } ) \leq d t _ { s }$ . Lemma 44 further gives $C ^ { \top } \theta ^ { P , \widehat { \pi } ^ { \dagger } } \leq \widehat { \gamma } < \tau$ . Given the estimation-phase history, the pair $( \widehat { \pi } ^ { \dagger } , \widehat { \gamma } )$ is fixed. On $\mathcal { E } .$ , the proof of Theorem 12 applies with this pair. Consequently, every policy executed during preference-free exploration and every main policy generated during planning is safe. In particular, we have $\scriptstyle \sum _ { t = t _ { s } + 1 } ^ { \mathbf { \hat { T } } }$ Violation(π<sup>t</sup>) = 0. Combining the violation bounds for the estimation phase and the exploration stage gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathrm { V i o l a t i o n } ( \pi ^ { t } ) \le d t _ { s } } \end{array}$ . The returned policy $\widehat { \pi }$ also satisfies Violation $( \widehat { \pi } ) = 0$ . This proves Theorem 21.

## Appendix J. Other Supporting Lemmas

Lemma 45 (Tseng (2008); Wei et al. (2020)) Let $f : \Delta _ { m } \mapsto \mathbb { R }$ be a convex function, and let $D _ { \mathrm { K L } } ( \pmb { x } \| \pmb { y } ) : = \sum _ { i = 1 } ^ { m }$ x<sub>i</sub> log $\frac { x _ { i } } { y _ { i } }$ for $\pmb { x } \in \Delta _ { m }$ and $\pmb { y } \in \Delta _ { m } ^ { o }$ . For any $\beta > 0 , z \in$ $\Delta _ { m }$ , and $\begin{array} { r } { \pmb { y } \in \Delta _ { m } ^ { o } , \ i f \ \pmb { x } ^ { \mathrm { o p t } } \in \arg \operatorname* { m i n } _ { \pmb { x } \in \Delta _ { m } } \{ f ( \pmb { x } ) + \beta D _ { \mathrm { K L } } ( \pmb { x } \| \pmb { y } ) \} } \end{array}$ , then we have $f ( { \pmb x } ^ { \mathrm { o p t } } ) +$ $\beta D _ { \mathrm { K L } } ( { \pmb x } ^ { \mathrm { o p t } } \| { \pmb y } ) \le f ( { \pmb z } ) + \beta D _ { \mathrm { K L } } ( { \pmb z } \| { \pmb y } ) - \beta D _ { \mathrm { K L } } ( { \pmb z } \| { \pmb x } ^ { \mathrm { o p t } } )$

Lemma 46 (Wei et al. (2020); Qiu et al. (2023)) Suppose that $\pmb { x } \in \Delta _ { m }$ and $\widetilde { x } _ { i } = ( 1 -$ $\alpha ) x _ { i } + \alpha / m$ . Then for all $\pmb { y } \in \Delta _ { m }$ , we have $D _ { \mathrm { K L } } ( \pmb { y } \| \widetilde { \pmb { x } } ) - D _ { \mathrm { K L } } ( \pmb { y } \| \pmb { x } ) \leq \alpha$ log m, and $\parallel \widetilde { \pmb { x } } -$ $\pmb { x } \Vert _ { 1 } \leq 2 \alpha$

## References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pages 22–31. PMLR, 2017.

Eitan Altman. Constrained Markov decision processes. Routledge, 2021.

Qinbo Bai, Amrit Singh Bedi, and Vaneet Aggarwal. Achieving zero constraint violation for constrained reinforcement learning via conservative natural policy gradient primal-dual algorithm. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 6737–6744, 2023.

Archana Bura, Aria HasanzadeZonuzy, Dileep Kalathil, Srinivas Shakkottai, and Jean-Francois Chamberland. Dope: Doubly optimistic and pessimistic exploration for safe reinforcement learning. Advances in neural information processing systems, 35:1047– 1059, 2022.

Yuhao Ding and Javad Lavaei. Provably eficient primal-dual reinforcement learning for cmdps with non-stationary objectives and constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7396–7404, 2023.

Shangding Gu, Bilgehan Sel, Yuhao Ding, Lu Wang, Qingwei Lin, Alois Knoll, and Ming Jin. Safe and balanced: A framework for constrained multi-objective reinforcement learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Conor F Hayes, Roxana R˘adulescu, Eugenio Bargiacchi, Johan K¨allstr¨om, Matthew Macfarlane, Mathieu Reymond, Timothy Verstraeten, Luisa M Zintgraf, Richard Dazeley, Fredrik Heintz, et al. A practical guide to multi-objective reinforcement learning and planning. Autonomous Agents and Multi-Agent Systems, 36(1):26, 2022.

Xiangkun He and Chen Lv. Toward personalized decision making for autonomous vehicles: a constrained multi-objective reinforcement learning technique. Transportation research part C: emerging technologies, 156:104352, 2023.

Mingxi Hu and Meiling Yu. Near-minimax multi-objective rl under predictable adversarial preferences and preference-free exploration in linear mdps. In International Conference on Machine Learning, 2026.

Ruiquan Huang, Jing Yang, and Yingbin Liang. Safe exploration incurs nearly no additional sample complexity for reward-free rl. In International Conference on Learning Representations, 2023.

Sandy Huang, Abbas Abdolmaleki, Giulia Vezzani, Philemon Brakel, Daniel J Mankowitz, Michael Neunert, Steven Bohez, Yuval Tassa, Nicolas Heess, Martin Riedmiller, et al. A constrained multi-objective reinforcement learning framework. In Conference on Robot Learning, pages 883–893. PMLR, 2022.

Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, and Tiancheng Yu. Learning adversarial markov decision processes with bandit feedback and unknown transition. In International Conference on Machine Learning, pages 4860–4869. PMLR, 2020.

Dohyeong Kim, Mineui Hong, Jeongho Park, and Songhwai Oh. Conflict-averse gradient aggregation for constrained multi-objective reinforcement learning. In International Conference on Learning Representations, volume 2025, pages 36498–36527, 2025.

Akihiro Kubo, Kosuke Nakanishi, and Shin Ishii. A single deep preference-conditioned policy for learning pareto coverage sets. arXiv preprint arXiv:2605.08946, 2026.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

Amarildo Likmeta, Alberto Maria Metelli, Andrea Tirinzoni, Riccardo Giol, Marcello Restelli, and Danilo Romano. Combining reinforcement learning with rule-based controllers for transparent and general decision-making in autonomous driving. Robotics and Autonomous Systems, 131:103568, 2020.

Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Smooth tchebychef scalarization for multi-objective optimization. In International Conference on Machine Learning, pages 30479–30509. PMLR, 2024.

Chunming Liu, Xin Xu, and Dewen Hu. Multiobjective reinforcement learning: A comprehensive overview. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 45 (3):385–398, 2014.

Tao Liu, Ruida Zhou, Dileep Kalathil, Panganamala Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained mdps. Advances in Neural Information Processing Systems, 34:17183–17193, 2021.

Tingting Ni and Maryam Kamgarpour. Provably safe sim-to-real transfer. arXiv preprint arXiv:2609.01418, 2026.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.

Giseung Park, Hyunyoung Nam, Woohyeon Byeon, Amir Leshem, and Youngchul Sung. Constrained multi-objective reinforcement learning with max-min criterion. arXiv preprint arXiv:2605.31388, 2026.

Shuang Qiu, Xiaohan Wei, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. Upper confidence primal-dual reinforcement learning for cmdp with adversarial loss. Advances in Neural Information Processing Systems, 33:15277–15287, 2020.

Shuang Qiu, Xiaohan Wei, and Mladen Kolar. Gradient-variation bound for online convex optimization with constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9534–9542, 2023.

Shuang Qiu, Dake Zhang, Rui Yang, Boxiang Lyu, and Tong Zhang. Traversing pareto optimal policies: Provably eficient multi-objective reinforcement learning. arXiv preprint arXiv:2407.17466, 2024.

Diederik M Roijers, Peter Vamplew, Shimon Whiteson, and Richard Dazeley. A survey of multi-objective sequential decision-making. Journal of Artificial Intelligence Research, 48:67–113, 2013.

Aviv Rosenberg and Yishay Mansour. Online convex optimization in adversarial markov decision processes. In International Conference on Machine Learning, pages 5478–5486. PMLR, 2019.

Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Learning adversarial mdps with stochastic hard constraints. In International Conference on Machine Learning, pages 56920–56951. PMLR, 2025.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Fengrun Tang, Zhenxiang Feng, Yonggang Li, Chunhua Yang, and Bei Sun. A constrained multi-objective deep reinforcement learning approach for temperature field optimization of zinc oxide rotary volatile kiln. Advanced Engineering Informatics, 58:102197, 2023.

Chen Tessler, Daniel J Mankowitz, and Shie Mannor. Reward constrained policy optimization. arXiv preprint arXiv:1805.11074, 2018.

Paul Tseng. On accelerated proximal gradient methods for convex-concave optimization. Submitted to SIAM Journal on Optimization, 2008.

Kristof Van Mofaert and Ann Now´e. Multi-objective reinforcement learning using sets of pareto dominating policies. The Journal of Machine Learning Research, 15(1):3483–3512, 2014.

Sharan Vaswani, Lin Yang, and Csaba Szepesv´ari. Near-optimal sample complexity bounds for constrained mdps. Advances in Neural Information Processing Systems, 35:3110–3122, 2022.

Xiaohan Wei, Hao Yu, and Michael J Neely. Online primal-dual mirror descent under stochastic constraints. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 4(2):1–36, 2020.

Manuel Wendl, Yarden As, Manish Prajapat, Anton Pollak, Stelian Coros, and Andreas Krause. Safe exploration via policy priors. arXiv preprint arXiv:2601.19612, 2026.

Jingfeng Wu, Vladimir Braverman, and Lin Yang. Accommodating picky customers: Regret bound and exploration complexity for multi-objective reinforcement learning. Advances in Neural Information Processing Systems, 34:13112–13124, 2021.

Nuoya Xiong, Yihan Du, and Longbo Huang. Provably safe reinforcement learning with step-wise violation constraints. Advances in Neural Information Processing Systems, 36: 54341–54353, 2023.

Runzhe Yang, Xingyuan Sun, and Karthik Narasimhan. A generalized algorithm for multiobjective reinforcement learning and policy adaptation. Advances in neural information processing systems, 32, 2019.

Weiye Zhao, Feihan Li, Yifan Sun, Yujie Wang, Rui Chen, Tianhao Wei, and Changliu Liu. Absolute state-wise constrained policy optimization: High-probability state-wise constraints satisfaction. arXiv preprint arXiv:2410.01212, 2024.