# Micro Neural Policies for Safe Real-Time Robotic Control

Hongpeng Cao, Riccardo Curcio, Daniele Ottaviano, and Marco Caccamo

Abstract— In this paper, we investigate the synthesis of Micro Neural Policies (MNP) to enable safe and robust realtime robotic control on computationally constrained embedded devices. We demonstrate that integrating Evolution Strategy (ES) and Statistical Model Checking (SMC)-based verification for policy search can drastically reduce neural network size without compromising safety and robustness. We conduct a large-scale training and evaluation of MNP on Cartpole and Quadrotor control tasks, varying control frequencies and network architectures. After validating these policies in simulation, we evaluate their deployability through zero-shot transfer to physical systems. Our experiments show that MNP can successfully achieve safe sim-to-real transfer without sacrificing control performance. We then show that the policies’ memory footprint, ranging from 0.5 to 7.5 kB, allows deployment on microcontrollers, where they achieve real-time inference latency with under 25 ns of jitter while leaving the chip idle for over 97% of the time for additional workloads. This makes them a highly practical solution for severely resource-constrained robotic systems.

## I. INTRODUCTION

Autonomous robotic platforms increasingly rely on neural network policies to tackle complex, highly dynamic tasks. While these approaches demonstrate remarkable capabilities, their deployment often relies on powerful embedded computing platforms. For example, state-of-the-art autonomous drone racing systems execute learned policies on embedded GPUs such as the NVIDIA Jetson TX2 [1], a module with gigabytes of DRAM and a power envelope up to 15W.

In this paper, instead, we focus on bridging the gap between high-frequency neural control and severely resourceconstrained hardware. We target microcontroller-class platforms with only kilobytes of memory, limited compute capability and milliwatt-level power budgets. Operating under such constraints is challenging for conventional learning approaches (e.g., Deep Reinforcement Learning (DRL)), which often rely on highly parameterized policies (e.g., dense architectures with hidden layers of up to 512×512 units and hundreds of thousands of parameters) to facilitate optimization and maximize empirical performance in simulation [2].

By jointly investigating policy compactness, safety and real-time execution, our findings demonstrate that statistically verified neural controllers can successfully operate under these extreme hardware limits, bringing learning-based control to severely resource-constrained robotic platforms.

## A. Motivation

Synthesizing Micro Neural Policies (MNP) can substantially reduce memory and computational overhead, enabling low-latency, energy-efficient and predictable inference directly on resource-constrained embedded hardware [3]. However, discovering such policies is challenging. Conventional learning approaches (e.g., DRL) favor overparameterized policies to facilitate optimization and maximize empirical performance; therefore, aggressively reducing network size can compromise control quality, robustness and safety. This challenge is particularly critical for physical deployment, where compact policies must not only perform well but also satisfy safety requirements [4].

In this paper, we explore the combination of Evolution Strategy (ES) [5], [6] and Statistical Model Checking (SMC) [7], [8] to search within extremely compact policy regimes. Here, ES provides a derivative-free mechanism for searching policy parameters, while SMC verifies whether the resulting policy satisfies prescribed safety and performance requirements. We conduct a large-scale investigation to assess whether these algorithms can synthesize MNP that remain safe and robust, while characterizing the resulting tradeoffs among policy size, control performance, verifiability and real-world deployability.

## B. Contribution

We make the following contributions toward enabling safe and robust MNP for real-world deployment on severely resource-constrained robotic platforms:

• We investigate the synthesis of progressively smaller MNP for Cartpole and Quadrotor benchmarks across different control frequencies (50 Hz and 100 Hz). Using an integrated ES and SMC framework [7], we extensively evaluate both fully Neural Network (NN)- parameterized policies and residual architectures in simulation under probabilistic safety and performance guarantees. We compare these results with safe DRL baselines, demonstrating that the latter generally fail to learn effectively under such strict constraints.

• We validate the policies on physical Cartpole and Quadrotor platforms to assess their zero-shot sim-to-real transfer. Our real-world experiments demonstrate that these MNP can be directly deployed, achieving highly effective and robust control in the real world.

• We evaluate latency, memory footprint and energy consumption of our MNP on a modern Arm-based microcontroller (the Nordic nRF54L15-DK with a 128 MHz Cortex-M33 core), representative of current TinyML benchmark platforms [9]. Results show that the policies achieve real-time onboard execution. Across forty deployed policies, cost scales with network size: $0 . 5 -$ 7.5 kB of memory, $1 0 { - } 2 5 9 \mu \mathrm { s }$ per inference and $0 . 2 5 -$ $3 . 6 7 \mu \mathrm { J }$ , each policy running with under 25ns of jitter and consuming less than 2.6% of its control period.

We provide demonstration videos at the link<sup>1</sup>.

## II. RELATED WORK

Sim-to-Real Policy Transfer and Assessment. Simulation is widely used for policy synthesis [1], yet sim-toreal discrepancies remain a major obstacle to physical deployment. Common approaches improve robustness through domain randomization, disturbance injection and adversarial perturbations [10], [11], typically within gradient-based DRL pipelines. While these techniques can substantially improve transfer, robustness is often assessed under a limited set of scenarios and does not by itself provide explicit evidence of deployment readiness. As an alternative optimization paradigm, ES offers derivative-free policy search [5] and has recently been combined with SMC-based verification to provide probabilistic safety and performance guarantees under prescribed disturbance and operating-condition distributions [8], [7].

Micro Policy for Robotic Control. Recent literature [12], [6], has shown that simple policies can successfully solve complex continuous control tasks, suggesting that minimal architectures can achieve competitive generalization with significantly lower computational overhead. This perspective is further supported by theoretical and empirical analyses highlighting the efficiency and stability of policies inspired from classical control theory compared to learning-based approaches $( e . g .$ , DRL) [13]. A prominent example leveraging these insights is residual learning [14], [15], [16], which grounds the policy on a fixed nominal policy and trains the NN to compute a residual correction. Such compact architectural designs directly support the emerging paradigm of Tiny Machine Learning (TinyML), which focuses on executing neural inference directly on low power embedded devices [9], [3].

Post-Training Compression Methods. A common approach to deploy large NN-based policies onto embedded devices involves post-training model compression techniques, such as network pruning [17] and quantization [18]. While these methods successfully reduce the memory footprint and inference time of a NN, their direct application to Reinforcement learning (RL) presents unique pitfalls. Any post-hoc approximation of the network, including policy distillation [19], can unpredictably shift the policy’s action distribution. This fragility is especially critical when a policy comes with statistical guarantees [8], as post-hoc compression inherently invalidates the properties established for the uncompressed model.

## III. BACKGROUND

We use R to denote the set of real numbers. For $\mathbf { x } \in \mathbb { R } ^ { n }$ $x _ { i }$ denotes its i-th component, and vector inequalities are interpreted component-wise, $\mathrm { i . e . , ~ a ~ \leq ~ b ~ }$ means $a _ { i } \leq b _ { i }$ for all $i \in \{ 1 , \ldots , n \}$ . The zero vector in $\mathbb { R } ^ { n }$ is denoted by $\mathbf { 0 } _ { ( n ) }$ Let T = N denote the discrete time set. A signal $\mathbf { z } \in \mathcal { Z } ^ { \mathbb { T } }$ assigns each $t \in \mathbb { T }$ a value $z _ { t } \in \mathcal { Z }$ . For a random variable $X , x \sim X$ denotes a realization of X and $x \in X$ a member entailed by X. We write $\operatorname* { P r } ( E )$ for the probability of an event E and $\Delta ( X )$ for the set of probability distributions on $X .$

## A. MDP

We model the robotic control problem as a Markov Decision Process (MDP) defined by the tuple $( \mathcal { S } , \mathcal { A } , \mathrm { R } , P , \gamma _ { r } , \rho _ { 0 } )$ At each time step $t \in \mathbb { T }$ , the agent observes its state $\mathbf { s } _ { t } \in \mathcal { S } \subseteq \mathbb { R } ^ { n } \ ( n > 0 )$ , applies an action $\mathbf { a } _ { t } \in \mathcal { A } \subseteq \mathbb { R } ^ { m }$ $( m > 0 )$ determined by a policy π (deterministic, $\pi : { \mathcal { S } }  A .$ or stochastic, $\pi ~ : ~ { \mathcal { S } } ~  ~ \Delta ( { \mathcal { A } } ) )$ and receives a reward $r _ { t } = \mathrm { R } \big ( \mathbf { s } _ { t } , \mathbf { a } _ { t } , \mathbf { s } _ { t + 1 } \big ) \ \in \ \mathbb { R }$ as the system transitions to the next state $\mathbf { s } _ { t + 1 } \sim P ( \cdot \mid \mathbf { s } _ { t } , \mathbf { a } _ { t } )$

Following [7], we additionally define a set of hard safety constraint functions to satisfy $C = \{ c _ { i } : S \times \mathcal { A } \times \mathcal { S } \to \mathbb { R } \} _ { i = 1 } ^ { z }$ $( z > 0 )$ , where the i-th constraint is satisfied at time step $t \in$ T whenever $c _ { i } \Big ( \mathbf { s } _ { t } , \mathbf { a } _ { t } , \mathbf { s } _ { t + 1 } \Big ) \leq 0$ . Furthermore, we assume the stochastic transition kernel P is explicitly induced by a discrete-time dynamics function f subject to an unknown disturbance:

$$
\mathbf { s } _ { t + 1 } = \mathbf { f } ( \mathbf { s } _ { t } , \mathbf { a } _ { t } , \mathbf { w } _ { t } ) , \qquad \mathbf { w } _ { t } \in \mathcal { W } ,\tag{1}
$$

where $\mathcal { W } \subseteq \mathbb { R } ^ { q }$ denotes the set of admissible disturbances, including modeling errors and external perturbations.

Central to the framework [7] is the concept of an operational scenario $( \mathbf { s } _ { 0 } , \mathbf { w } )$ , which is defined as a joint realization of an initial state $\begin{array} { r } { \mathbf { s } _ { 0 } \ \sim \ \boldsymbol { \mathcal { U } } ( \Omega ) } \end{array}$ , sampled uniformly from a candidate region $\Omega \subseteq { \mathcal { S } }$ and a disturbance sequence w generated by a discrete-time stochastic process W taking values in W. Letting $\rho _ { c } = ( \mathcal { U } ( \Omega ) , \mathbf { W } )$ denote their joint distribution, we write $( \mathbf { s } _ { 0 } , \mathbf { w } ) \sim \rho _ { c }$ for a sampled scenario.

The agent’s objective is to synthesize a NN policy $\pi _ { \theta } ,$ parameterized by ${ \theta \in \mathbb { R } ^ { d } \mathrm { \Omega } ( d \mathrm { \Omega } > \mathrm { \Omega } 0 ) }$ , that maximizes the expected cumulative reward while strictly satisfying the safety constraints under any possible operational scenario.

## B. Policy synthesis with SMC-based verification

To synthesize statistically verified safe and robust policies, we adopt the simulation-based protocol from [7], which combines Evolution Strategy (ES) and SMC-based verification to identify verified NN controllers together with their statistically verified safe operating regions.

Within this framework, ES serves as the primary optimization engine, offering two key advantages over RL for robotic control. First, it operates on trajectory-level returns and therefore remains robust across control frequencies [5], unlike RL methods that struggle at high sampling rates when consecutive state transitions become nearly identical. Second, it can explicitly account for different scenarios, therefore enabling scenario-based optimization and verification [8].

To evaluate a given policy $\pi _ { \theta }$ over an execution horizon $H \in \mathbb { T }$ and within a scenario $\mathbf { \Psi } ( \mathbf { s } _ { 0 } , \mathbf { w } )$ , the framework computes the trajectory-level cumulative reward and constraint violations:

$$
\begin{array} { l } { \displaystyle \mathbf { J } ( \pi _ { \theta } , \mathbf { s } _ { 0 } , \mathbf { w } , H ) = \sum _ { t = 0 } ^ { H - 1 } \mathbf { R } \big ( \mathbf { s } _ { t } , \pi _ { \theta } ( \mathbf { s } _ { t } ) , \mathbf { s } _ { t + 1 } \big ) , } \\ { \displaystyle \mathbf { C } ( \pi _ { \theta } , \mathbf { s } _ { 0 } , \mathbf { w } , H ) = \left[ \sum _ { t = 0 } ^ { H - 1 } \mathrm { m a x } \big ( 0 , c _ { j } ( \mathbf { s } _ { t } , \pi _ { \theta } ( \mathbf { s } _ { t } ) , \mathbf { s } _ { t + 1 } \big ) \big ) \right] _ { j = 1 } ^ { z } , } \end{array}
$$

with state transitions governed by $\mathbf { s } _ { t + 1 } = \mathbf { f } \left( \mathbf { s } _ { t } , \pi _ { \boldsymbol { \theta } } ( \mathbf { s } _ { t } ) , \mathbf { w } _ { t } \right)$ as in Eq. (1).

By optimizing and verifying over these trajectory metrics, we can obtain probabilistic guarantees on both performance and safety for the synthesized controllers.

Property 1 (Probabilistic Guarantees, adapted from [7]). Consider a continuous control problem defined on a state space S, governed by transition dynamics f (Eq. (1)) over an execution horizon $H \in \mathbb { N } .$ Let the system be subject to a reward function R, hard safety constraints $C ,$ and an initial scenario distribution $\rho _ { c } = ( \mathcal { U } ( \Omega ) , \mathbf { W } )$ , comprising a discrete-time stochastic process W and a candidate safe operating region Ω.

Given a confidence level $\delta \in ( 0 , 1 ]$ and an allowable failure probability $\varepsilon \in ( 0 , 1 ] ,$ , the solution triplet $( \pi _ { \theta ^ { * } } , \beta ^ { * } , \Omega ^ { * } )$ synthesized by the framework in [7] guarantees that, with confidence at least $1 - \delta ,$ the probability of encountering a scenario in $\boldsymbol { \rho } _ { c } ^ { * } = ( \mathcal { U } ( \boldsymbol { \Omega } ^ { * } ) , \mathbf { W } )$ where $\pi _ { \theta ^ { \ast } }$ violates safety constraints or fails to achieve the performance lower bound $\beta ^ { * }$ is:

$$
\begin{array} { r } { \operatorname* { P r } _ { \mathbf { \Phi } } \operatorname* { P r } _ { \mathbf { \Phi } } \left( \begin{array} { c } { P e r f o r m a n c e \nu i o l a t i o n } \\ { \Im \big ( \pi _ { \theta ^ { * } } , \mathbf { s } _ { 0 } , \mathbf { w } , H \big ) < \beta ^ { * } } \\ { \big | \qquad \forall } \\ { \big ( \mathbf { s } _ { 0 } , \mathbf { w } \big ) \sim \rho _ { c } ^ { * } } \\ { \underbrace { \mathbf { C } \big ( \pi _ { \theta ^ { * } } , \mathbf { s } _ { 0 } , \mathbf { w } , H \big ) > \mathbf { 0 } _ { ( z ) } } _ { S a f e t y \nu i o l a t i o n } } \end{array} \right) \leq \varepsilon . } \end{array}
$$

## IV. MICRO NEURAL POLICY SEARCH

Learning safe and robust MNP requires satisfying safety constraints within severely limited network size. Since standard DRL struggles to optimize over highly restricted architectures and high control frequencies [2], we employ ES with scenario-based optimization and verification to search directly for compact and statistically verified policies.

## A. Policy Parameterization

We parameterize the policy $\pi _ { \theta }$ as a standard multilayer perceptron (MLP) that deterministically maps the observed state to an action:

$$
\mathbf { a } _ { t } = \pi _ { \theta } ( \mathbf { s } _ { t } ) .\tag{2}
$$

The compactness of $\pi _ { \theta }$ is determined by the number of hidden layers and the number of neurons in each hidden layer. We construct a set of compact-policy candidates with two hidden layers, varying the hidden-layer width from $N = 2$ to $N = 3 2$ neurons and with the nonlinear activation function in the hidden layers using tanh.

We additionally consider a residual policy architecture [14], denoted by $\pi _ { \theta } ^ { \mathrm { r e s } }$ , to study whether a baseline controller can facilitate policy learning. Specifically, the residual policy combines a learnable policy π with a fixed baseline policy π :

$$
\mathbf { a } _ { t } = \pi _ { \theta } ^ { \mathrm { r e s } } ( \mathbf { s } _ { t } ) = \pi _ { \theta } ( \mathbf { s } _ { t } ) + \pi _ { b } ( \mathbf { s } _ { t } ) .\tag{3}
$$

Here, the learnable policy $\pi _ { \theta }$ is parameterized as in (2), while $\pi _ { b }$ is a linear feedback controller obtained by solving a constrained convex optimization problem based on an approximate system model [20].

## B. Policy Search

To search for compact and safe policies, we employ a safety-aware Evolution Strategy (ES)-based optimization and verification strategy [7], [8], while varying the network size to assess how compact a policy can be while meeting these requirements.

Within the safety-aware ES framework, the optimizer trains the policy by jointly maximizing the objective function $\mathrm { J } ( \pi _ { \boldsymbol { \theta } } , \mathbf { s } _ { 0 } , \mathbf { w } , H )$ and satisfying a set of hard safety constraints $\mathrm { C } ( \pi _ { \boldsymbol { \theta } } , \mathbf { s } _ { 0 } , \mathbf { w } , H )$ over a finite temporal horizon $H \in \mathbb { N }$ and a fixed set of scenarios Σ initially selected via k-medoids method from a scenario distribution $\rho _ { c }$

The optimization proceeds iteratively. Specifically, at each iteration, given the current parameter vector $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ , we first evaluate the unperturbed policy $\pi _ { \theta }$ across all scenarios in the fixed set Σ. To accelerate policy search, we extract the Pareto-worst [8] subset of scenarios $\tilde { \Sigma } \subseteq \Sigma$ with respect to cumulative reward and constraints violations.

We then draw $\frac { t } { 2 }$ independent noise samples $\zeta _ { s } \sim$ $\mathcal { N } \big ( \mathbf { 0 } _ { ( d ) } , \mathbb { I } _ { ( d \times d ) } \big )$ . For each noise sample $s \in \{ 1 , \ldots , \frac { t } { 2 } \}$ we construct symmetric candidate parameter pairs scaled by $\eta ~ > ~ 0$ , indexing the positive and negative perturbed policies as $\pi _ { 2 s } = \pi _ { \theta + \eta \zeta _ { s } }$ and $\pi _ { 2 s - 1 } = \pi _ { \theta - \eta \zeta _ { s } }$ , respectively. All candidate policies $i \in \{ 1 , \ldots , t \}$ are then evaluated in parallel across the scenario set $\tilde { \Sigma } .$

For each candidate policy $\pi _ { i } ,$ , we compute the average undiscounted cumulative rewards and constraint violations as:

$$
\tilde { \mathrm { J } } _ { i } = \frac { 1 } { | \tilde { \Sigma } | } \sum _ { ( \mathbf { s } _ { 0 } , \mathbf { w } ) \in \tilde { \Sigma } } \mathrm { J } ( \pi _ { i } , \mathbf { s } _ { 0 } , \mathbf { w } , H ) ,\tag{4}
$$

$$
\tilde { \mathrm { C } } _ { i } = \frac { 1 } { | \tilde { \Sigma } | } \sum _ { ( \mathbf { s } _ { 0 } , \mathbf { w } ) \in \tilde { \Sigma } } \mathrm { C } ( \pi _ { i } , \mathbf { s } _ { 0 } , \mathbf { w } , H ) .\tag{5}
$$

These aggregated metrics are subsequently mapped into a scalar Augmented Lagrangian fitness value $h _ { i }$ derived as in [8]:

$$
h _ { i } = \tilde { \mathrm { J } } _ { i } - \sum _ { p = 1 } ^ { z } \left( \lambda _ { p } \hat { \mathrm { C } } _ { i , p } + \frac { \mu _ { p } } { 2 } \hat { \mathrm { C } } _ { i , p } ^ { 2 } \right) ,\tag{6}
$$

where $\hat { \mathrm { C } } _ { i , p } = \operatorname* { m a x } \{ \tilde { \mathrm { C } } _ { i , p } , - { \lambda _ { p } } / { \mu _ { p } } \}$ , and $\lambda \in \mathbb { R } ^ { z }$ and $\mu \in \mathbb { R } _ { + } ^ { z }$

are the Lagrange multipliers and positive penalty coefficients, respectively.

The policy parameters are then updated by weighting each noise perturbation $\zeta _ { i }$ by the differential fitness score of its corresponding symmetric pair:

$$
\theta  \theta + \frac { \alpha } { t } \sum _ { s = 1 } ^ { \frac { t } { 2 } } \bigl [ ( h _ { 2 s } - h _ { 2 s - 1 } ) \zeta _ { s } \bigr ] ,\tag{7}
$$

where $\alpha > 0$ is the learning rate.

Once the optimization phase terminates, it yields a candidate policy $\pi _ { \theta }$ along with its performance lower bound $\beta ^ { * }$ evaluated over Σ. If the candidate policy is valid $( i . e .$ , it satisfies all constraints over Σ), we apply Monte Carlo–based SMC [21], [22] to verify its safety and claimed performance across the full scenario distribution $\rho _ { c } = ( \mathcal { U } ( \Omega ) , { \bf W } )$ . If counterexamples are found, these are added to the scenario set Σ and optimization restarts. Otherwise, the policy is statistically verified to be safe within the considered scenario distribution $\rho _ { c }$ with a given confidence δ and error ε threshold (Property 1). The same optimization protocol is applied to the residual architecture, except that rewards and constraints are evaluated along trajectories generated by the composed policy $\pi _ { \theta } ^ { \mathrm { r e s } }$

## C. Training Curriculum

During policy training, we follow the training–verification curriculum introduced in [7]. The policy is first trained on scenarios sampled from an initial scenario distribution $\rho _ { c } =$ (U(Ω), W), where Ω is defined as an ellipsoidal set

$$
\Omega : = \{ \mathbf s \in \mathcal { S } \mid \mathbf s ^ { \top } \mathbf P \mathbf s \leq 1 \} , \qquad \mathbf P \succ 0 ,\tag{8}
$$

The positive-definite matrix P determines the geometry of the ellipsoid. It is obtained by solving a Linear Matrix Inequality (LMI)-based convex optimization problem that jointly enforces Lyapunov stability and the prescribed state and input constraints [23].

Once the policy passes SMC-based verification, the region is expanded $\Omega \subset \Omega ^ { \prime }$ and the training–verification cycle is repeated until no further verified expansion is achieved. As a result, this process yields a solution triplet $( \pi _ { \theta ^ { * } } , \beta ^ { * } , \Omega ^ { * } )$ guaranteeing, with confidence at least $1 - \delta ,$ , that the probability of encountering a scenario from $\rho _ { c } ^ { * } = ( \mathcal { U } ( \Omega ^ { * } ) , \mathbf { W } )$ in which $\pi _ { \theta ^ { \ast } }$ ∗ violates safety constraints or fails to achieve the performance lower bound $\beta ^ { * }$ is at most ε (Property 1).

To enhance sim-to-real transferability, we inject independent Gaussian perturbations $\mathbf { w } _ { t } \sim \mathcal { N } ( 0 , \nu ^ { 2 } )$ into the system dynamics at each time step $t ~ \in ~ \mathbb { T } ~ [ 1 0 ]$ . Sensor observations $( i . e . , \mathrm { N N }$ inputs) are perturbed multiplicatively by a factor of $\left( 1 + \mathbf { w } _ { t } \right)$ , while actuator commands (i.e., NN outputs) are perturbed additively, scaled by the corresponding actuation range and clipped to their admissible bounds. In our experiments, we set $\nu = 5 \%$ for sensor noise and $\nu = 1 \%$ for actuation noise, following previous studies [7].

## V. EXPERIMENTAL SETTING

In this section, we detail the experimental setup for evaluating our ES-based training approach and baselines.

TABLE I: State and action constraints for the Cartpole and Quadrotor systems.
<table><tr><td>System</td><td>Variable</td><td>Constraint</td><td>Variable</td><td>Constraint</td></tr><tr><td rowspan="3">Cartpole</td><td> $x t$ </td><td> $\left| x _ { t } \right| < 0 . 3 5 \mathrm { ~ m ~ }$ </td><td> $\theta _ { t }$ </td><td> $\lvert \theta _ { t } \rvert < 0 . 8 ~ \mathrm { r a d }$ </td></tr><tr><td> ${ \dot { x } } _ { t }$ </td><td> $| \dot { x } _ { t } | < 2 . 0 ~ \mathrm { m / s }$ </td><td> $\dot { \theta } _ { t }$ </td><td> $| \dot { \theta } _ { t } | < 4 . 0 \ \mathrm { r a d / s }$ </td></tr><tr><td> $a _ { t }$ </td><td> $\left| a _ { t } \right| < 1 0 \mathrm { ~ N ~ }$ </td><td></td><td></td></tr><tr><td rowspan="4">Quadrotor</td><td> $x _ { t } , y _ { t }$ </td><td> $| \cdot | < 4 . 0 \mathrm { ~ m ~ }$ </td><td> $\phi _ { t } , \theta _ { t }$ </td><td> $| \cdot | < 1 . 5 \ \mathrm { r a d }$ </td></tr><tr><td> $z _ { t }$ </td><td> $| z _ { t } | < 2 . 5 \ \mathrm { m }$ </td><td> $\psi _ { t }$ </td><td> $| \psi _ { t } | < \pi ~ \mathrm { r a d }$ </td></tr><tr><td> $\dot { x } _ { t } , \dot { y } _ { t } , \dot { z } _ { t }$ </td><td> $| \cdot | < 5 . 0 ~ \mathrm { m / s }$ </td><td> $p _ { t } , q _ { t } , r _ { t }$ </td><td> $| \cdot | < 2 . 0 \ \mathrm { r a d / s }$ </td></tr><tr><td> $a _ { t }$ </td><td> $| a _ { t } ^ { ( i ) } | < 1 5 N , \ i = 1 , \dots , 4$ </td><td></td><td></td></tr></table>

![](images/39b1489c1e4c3f64f29745bb2895bcf1b77551ae594cf9098b00e25a9e4ab031.jpg)

![](images/7e420525bd8daf77acbd6b6e042e912c85d2f09afa3ace752f3d8c99f11b0c49.jpg)  
(a) Cartpole  
(b) Quadrotor  
Fig. 1: Real-world experimental platforms: (a) Quanser Cartpole equipped with a Raspberry Pi 4B. (b) ANT-X quadrotor equipped with a Pixhawk flight controller (STM32F427).

## A. Case Studies

We choose two classical nonlinear control systems. The first is Cartpole, with a four-dimensional state and a scalar action. The task is to track a commanded cart position while stabilizing the pole upright, i.e., $x = { \hat { x } }$ and $\theta = 0$

The second, more challenging benchmark is a highdimensional quadrotor goal-reaching task with direct lowlevel thrust control. The state is twelve-dimensional, comprising position, attitude, linear velocity and angular velocity, while the control input consists of the four individual rotor thrusts. The quadrotor must reach a commanded position $[ \hat { x } , \hat { y } , \hat { z } ] ^ { \top }$ with all remaining state dimensions equal to zero, corresponding to a stationary hover at the goal.

For both systems, safety is enforced through state constraints as summarized in Table I. We evaluate both systems at control frequencies of 50 Hz and 100 Hz in both simulation and real-world experiments.

## B. Evaluation Methodology

Learning Methods. We evaluate four policy-learning paradigms, including our two ES-based approaches and two safe DRL-based baselines: i) ES: a policy directly optimized using ES as described in Section IV-A; and ii) ES-Res: a residual policy that augments a certified linear policy prior obtained as in [24], with the residual component optimized using ES [7]. iii) DRL: a model-free policy trained using soft actor-critic (SAC) [25] with a Lagrangian formulation to account for safety constraints [26]; and iv) DRL-Res: a residual policy [15] that augments the same certified linear policy prior with a learnable correction trained using Lagrangian SAC. Both DRL baselines are trained with domain randomization [11] to improve sim-to-real transfer.

Furthermore, for the SMC-based verification procedure, we set the confidence and error bounds to $\delta \ = \ 1 \%$ and $\varepsilon = 1 \% ,$ , respectively (see Section III). Because DRL baselines lack guarantees regarding their safe operating regions or performance lower bounds, we verify only their safety constraint satisfaction on nominal scenarios drawn from the initial unscaled region Ω, as defined in Eq. (8).

Real-world Evaluation. For real-world deployment, the experiments are conducted on a physical Cartpole platform (Quanser) equipped with a Raspberry Pi 4B and a quadrotor platform (ANT-X) equipped with an STM32F427. Neither is a compute-oriented platform. The Raspberry Pi 4B is a quadcore Cortex-A72 at 1.5 GHz with no neural accelerator, while the STM32F427 is a single-core Cortex-M4F at 180 MHz with 256 kB of SRAM. The experimental platforms are shown in Figures 1a and 1b.

Inference Benchmarking. To thoroughly assess inference performance on resource constrainted devices, we deploy policies on the Nordic nRF54L15-DK, a modern Arm-based microcontroller equipped with a 128 MHz Cortex-M33 core that is highly representative of the device class targeted by current TinyML benchmarking [9].

## VI. RESULTS

We evaluate the proposed approach in simulation and realworld experiments with respect to the following research questions:

• RQ1: Which training paradigm identifies the most Micro Neural Policies (MNP) while maintaining safety and robustness?

• RQ2: How well do the resulting MNP transfer from simulation to the real world?

• RQ3: What computational benefits do MNP provide on resource-constrained embedded platforms?

• RQ4: How efficiently do the training paradigms find safe and robust policies?

## A. Compact policy search (RQ1)

We first evaluate all algorithms in simulation across five neural network parameterizations, ranging from $2 \times 2$ to $3 2 \times 3 2 .$ , using the OpenAI Gym Cartpole environment adapted from [27] and the PyBullet Quadrotor environment adapted from [28]. We report three main metrics: i) Verifiability: the fraction of policy parameterizations that pass the SMC verification protocol; ii) Return Lower Bound: the minimum undiscounted cumulative reward encountered during SMC verification; and iii) Relative Enlarged Region: the relative enlargement of the verified operating region, normalized between the default and maximum constraintadmissible bounds. A larger value indicates broader safe operation, while zero denotes no enlargement. The results for all algorithms are summarized in Tables II and III.

Results show that the ES-based synthesis paradigms find verified policies much more reliably across all five parameterizations in both case studies and at both control frequencies. In contrast, DRL-based training paradigms find verified policies only for the simpler Cartpole task. This observation is consistent with [8], [7], where ES is shown to be more effective in learning constraint-respecting policies.

TABLE II: Safety and performance comparison in Cartpole simulation at the 50Hz and 100Hz control frequencies.
<table><tr><td>Freq.</td><td>Metric</td><td>DRL</td><td>DRL-Res</td><td>ES</td><td>ES-Res</td></tr><tr><td rowspan="3">50Hz</td><td>Verifiability</td><td>0/5</td><td>3/5</td><td>5/5</td><td>5/5</td></tr><tr><td>Return Lower Bound</td><td>7.46 ± 4.39</td><td>251.97 ± 223.71</td><td>461.57 ± 11.64</td><td>457.35 ± 5.26</td></tr><tr><td>Relative Enlarged Region (%)</td><td></td><td>0.00 ± 0.00</td><td>96.71 ± 3.63</td><td>96.62 ± 3.39</td></tr><tr><td rowspan="3">100 Hz</td><td>Verifiability</td><td>1/5</td><td>2/5</td><td>5/5</td><td>5/5</td></tr><tr><td>Return Lower Bound</td><td>85.52 ± 167.63</td><td>188.05 ± 207.56</td><td>442.40 ± 7.46</td><td>430.52 ± 8.33</td></tr><tr><td>Relative Enlarged Region (%)</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>95.50 ± 3.01</td><td>97.79 ± 2.57</td></tr></table>

TABLE III: Safety and performance comparison in Quadrotor simulation at the 50Hz and 100Hz control frequencies.
<table><tr><td>Freq.</td><td>Metric</td><td>DRL</td><td>DRL-Res</td><td>ES</td><td>ES-Res</td></tr><tr><td rowspan="3">50Hz</td><td>Verifiability</td><td>0/5</td><td>0/5</td><td>3/5</td><td>5/5</td></tr><tr><td>Return Lower Bound</td><td>68.71 ± 147.02</td><td>12.61 ± 15.24</td><td>319.60 ± 165.26</td><td>455.69 ± 14.33</td></tr><tr><td>Relative Enlarged Region (%)</td><td>一</td><td></td><td>81.64 ± 8.93</td><td>94.13 ± 2.87</td></tr><tr><td rowspan="3">100 Hz</td><td>Verifiability</td><td>0/5</td><td>0/5</td><td>4/5</td><td>5/5</td></tr><tr><td>Return Lower Bound</td><td>2.91 ± 5.23</td><td>12.60 ± 16.91</td><td>300.68 ± 137.79</td><td>424.09 ± 32.30</td></tr><tr><td>Relative Enlarged Region (%)</td><td></td><td></td><td>85.80 ± 15.76</td><td>87.46 ± 6.08</td></tr></table>

In these experiments, we also find that incorporating a baseline policy through the residual architecture is generally beneficial for both ES and DRL, improving verifiability in both case studies. Among the considered training paradigms, ES-Res achieves the highest verifiability and the largest verified operating region. We also note that the return lower bound can decrease as the verified region expands, since the likelihood of encountering more challenging scenarios increases.

Finally, we compare the policy learning capability through the detailed capacity-sweep results shown in Figures 2 and 3. For the simpler Cartpole task, ES and ES-Res are relatively insensitive to the network parameterization, yielding similar return lower bounds and enlarged regions across the considered network sizes. In contrast, for the higherdimensional quadrotor task, ES fails to find verified policies with the smallest network parameterizations, i.e., $2 \times 2$ and $4 \times 4$ at 50Hz, and 2 × 2 at 100Hz. Interestingly, among the verified policies, larger network parameterizations tend to yield smaller verified operating regions. This occurs because ES struggles to estimate accurate gradients in higherdimensional spaces, making it more difficult to adapt the policy as the operating region is iteratively expanded.

## B. Sim-to-real validation (RQ2)

In this section, we study the sim-to-real transfer capabilities of only the verified policies. To ensure a comprehensive comparison, we independently select the best-performing candidate from each of the four baseline paradigms. Specifically, within each baseline category, we compute an aggregated score based on the min–max normalized return lower bound $s _ { P }$ and region enlargement s , defined as Score $= 0 . 5 s _ { P } + 0 . 5 s _ { E }$ . The policy achieving the highest score within its respective category is then deployed for realworld testing. Baselines with no verified policy are depicted in gray within figure legends.

Since exhaustively resetting the physical systems to cover all possible scenarios is impractical, we focus primarily on challenging test cases at both control frequencies. For Cartpole, we evaluate recovery from the farthest physically realizable initial position of $x _ { 0 } = 0 . 2 8 \mathrm { n }$ m. For the quadrotor, we evaluate the policy in $1 4 \times 4 \times 2 . 5$ m indoor flight cage using a sequential waypoint-tracking task, where the setpoint is updated upon reaching each waypoint.

![](images/2a3e2e1370fbef6a63c07d8cc1b51a12e42e1c4fb9afbd4a7439e1b3f189f40d.jpg)

![](images/573acfc874725e397937ba659b24fc063a2945798096722b87abe0d1e85fd270.jpg)  
Fig. 2: Learning capability evaluation in Cartpole Simulation.

![](images/f60edd784c4d748083be8d5eda9ce348efcc602d040fe579cfdafe3e0ea658da.jpg)

![](images/5ab39e12d9248c0f3b53f98f9d99fd2b16252f810f6fbc8ff4880afda6b3e57f.jpg)  
Fig. 3: Learning capability evaluation in Quadrotor Simulation.

Results show that MNP trained directly with ES generalize better to the real world compared to both residual architectures and DRL baselines. In the real-world Cartpole evaluation (Figure 4), standard DRL fails to synthesize a verified policy at 50 Hz and exhibits severe instability at 100 Hz. Although DRL-Res deploys a verified policy at 50 Hz, it exhibits persistent oscillations, which become more pronounced at 100 Hz. Similarly, ES-Res displays slight oscillations at 50 Hz and sustained limit-cycle oscillations at 100 Hz. Standard ES remains the only approach capable of reliably stabilizing the system with minimal error across both control frequencies.

Quadrotor experiments (Figure 5) present a greater challenge where both DRL baselines fail to synthesize a verified policy. In this context, also the vulnerability of residual architectures becomes more evident: while the standard ES policy successfully completes the waypoint-tracking task, ES-Res fails and crashes at both 50 Hz and 100 Hz. The primary reason for this failure is the severe degradation of the model-based controller under model mismatch, as also observed in previous works [1]. To support this statement, high-fidelity Gazebo simulations using the manufacturerprovided Ant-X model (Section VI-B) show that ES-Res actually outperforms standard ES when model mismatch is small. This suggests that residual policies are more sensitive to sim-to-real gaps due to their reliance on an unadapted, suboptimal model-based baseline policy.

![](images/f737144563b0691242d946b5323b620a431e5a110a7adfd72035c4a0ba90e1cb.jpg)  
Fig. 4: Real-world Cartpole evaluation.

## C. Embedded inference cost (RQ3)

We assess MNP inference latency, memory usage and energy consumption on the Nordic nRF54L15-DK, an Armbased microcontroller equipped with a 128 MHz Cortex-M33 core, representative of the class of devices targeted by TinyML [29], [3]. The policies are first converted into pure C code with onnx2c tool [30] and then deployed as dedicated threads on top of the Zephyr real-time OS. Table IV reports the results for the Cartpole and Quadrotor tasks across forty policies in total: two tasks, two controller types (ES and ES-Res), five hidden-layer configurations $( 2 \times 2 \  t o \ 3 2 \times 3 2 )$ and two control frequencies (50 Hz and 100 Hz). Since latency, flash, RAM and energy per inference depend only on the network, we report them once, at 50 Hz; power, being frequency-dependent, is reported at both rates. Latency is given as the mean ± standard deviation over 1000 timed inferences.

Inference latency is measured using the Cortex-M33 DWT cycle counter, which has a resolution of 7.8 ns. The mean latency ranges from 10.10 µs to 258.98 µs, so all policies remain well within their control periods, the worst case consuming only 2.5% of the available deadline. The timing is effectively deterministic: the standard deviation stays below 24 ns, fewer than four CPU cycles, as the networks are dense, fixed-size and free of data-dependent branching.

![](images/2975556c7dfad195643dccdd41c7b85ebd1e20ae4493dfd9c0b696eae8238514.jpg)

![](images/77fa9d88ad2e21c1e38fbe4842c6152340aace5b4f27fa5ec900f1d27ca0f0c3.jpg)

![](images/cdf80d7f3adba95a9b2ed139a339d76996d3bc70ef45d74b2c675c55b7096294.jpg)

![](images/c17d5cd2f45c494d8d5927500250a3579daa60ed767344a568bc63ff56e7e3f1.jpg)

![](images/68a2c8eecce25ec94419a960991ab92ce021c2793e2cc2d84d284f15de50caab.jpg)

![](images/bed2ae470f1967d134db17180c525cdb147e9de9be5ca32770399f451209cb5d.jpg)  
(a) Cross domain validation in Gazebo simulation.

![](images/8e230809180e84816c96d35dd3c1f47543854f11e21f41593d217e046904a94c.jpg)

![](images/246ac96233e3713045d459b2706d0b63a0e4ae8163afe8a94b9a9753fd75ffd3.jpg)  
(b) Real-world evaluation.  
Fig. 5: Quadrotor trajectory tracking in simulation and real-world experiments.

Flash occupancy ranges from 536 B to 7.5 kB and is dominated by the network parameters, increasing with hiddenlayer width. RAM usage ranges from 40 B to 304 B and is required only for intermediate activations, while the network parameters remain in flash. The residual formulation adds only 156–428 B of flash and 16–48 B of RAM over the corresponding ES policy. These requirements are well within the resource budgets typically considered in the TinyML literature [9].

Power is measured with a Nordic Power Profiler Kit II, sampling the supply current at 100 kHz while the policy runs at frequency $f$ for $T = 3 \mathrm { s }$ , alternating between inference and idle. Let $I _ { \mathrm { r u n } }$ be the mean current over that window and $I _ { \mathrm { i d l e } } = 2 2 6 \mu \mathrm { A }$ at 3.0 V the idle baseline. Each control period $1 / f$ holds one inference of duration τ , so with duty cycle $d = f \tau , I _ { \mathrm { r u n } } = ( 1 - d ) I _ { \mathrm { i d l e } } + d I _ { \mathrm { i n f } } .$ , where $I _ { \mathrm { i n f } } > I _ { \mathrm { i d l e } }$ is the active inference current, hence $P = I _ { \mathrm { r u n } } V$ increases with $f .$ We also report the energy per inference as $E = I _ { \mathrm { i n f } } V \tau$ where $I _ { \mathrm { i n f } } = I _ { \mathrm { i d l e } } + ( I _ { \mathrm { r u n } } - I _ { \mathrm { i d l e } } ) / ( f \tau )$

The microcontroller draws 0.69 mW for the 2 × 2 ES Cartpole at 50 Hz and 1.04 mW for the 32×32 ES quadrotor at 100 Hz, with 0.25–3.67 µJ per inference. At the highest load the inference accounts only for ∼ 0.01% of the power budget of a nano-quadrotor [31].

## D. Training efficiency (RQ4)

Although ES yields safe and robust compact policies, its optimization process incurs a significant computational cost. We address this challenge through massive parallelization across an HPC cluster. To ensure efficient scaling, we use the shared random seed method [5], allowing workers to locally reconstruct the parameter perturbations required for gradient estimation without transmitting full weight matrices over the network. This distributed architecture is implemented using OpenMPI [32], scaling up to 1000 simultaneous CPU cores.

![](images/496e0ab05b7c9b3e2bf01ba5390f479ae4fe0b5c7c7cb4a5490af11853f24682.jpg)

![](images/c0e15756896096436007488748a2db5a56451a00b9dc156d2fd50a9506abc9bf.jpg)  
Fig. 6: Training time comparison

Each compute node provides 256 GB of RAM and two AMD EPYC 7301 CPUs, yielding 32 physical cores per node.

As shown in Figure 6, ES requires approximately 1–4 hours to find a verified policy for Cartpole, while training can take more than 10 hours for the Quadrotor. Incorporating the baseline policy reduces the training time for the more challenging cases, although the benefit is less pronounced for the simpler Cartpole task.

For reference, the DRL-based optimization paradigms are substantially more efficient during training. Training the DRL baselines for one million environment steps requires approximately 3 hours for both Cartpole and the Quadrotor on a single high-end CPU core (AMD Ryzen 9 9950X).

## VII. DISCUSSION AND LIMITATIONS

Although the proposed framework synthesizes compact policies with formal statistical guarantees, their real-world validity depends on simulation fidelity. While injecting zeromean noise during optimization significantly enhances robustness against unmodeled dynamics, structural biases such as the nominal baseline in residual architectures can prevent these safety guarantees from holding upon physical deployment.

TABLE IV: On-device inference latency, memory and energy cost for Cartpole and Quadrotor policies.
<table><tr><td rowspan="2">System</td><td rowspan="2">Architecture</td><td colspan="2">Latency</td><td colspan="2">Flash</td><td colspan="2">RAM</td><td colspan="2">Energy/inference</td><td colspan="2">Power @50 Hz</td><td colspan="2">Power @100 Hz</td></tr><tr><td>ES</td><td>ES-Res</td><td>ES</td><td>ES-Res</td><td>ES</td><td>ES-Res</td><td>ES</td><td>ES-Res</td><td>ES</td><td>ES-Res</td><td>ES</td><td>ES-Res</td></tr><tr><td rowspan="6">Cartpole</td><td>2×2</td><td> $\overline { { 1 0 . 2 6 \pm 0 . 0 0 \mu \mathrm { s } } }$ </td><td> $\overline { { 1 0 . 1 0 \pm 0 . 0 0 \mu \mathrm { s } } }$ </td><td>536B</td><td>712B</td><td>40B</td><td>56B</td><td> $\overline { { { 0 . 2 5 \pm 0 . 0 7 \mu \mathrm { { J } } } } }$ </td><td> $\overline { { { 0 . 2 9 \pm 0 . 0 7 \mu \mathrm { J } } } }$ </td><td>0.69 mW</td><td>0.69mW</td><td> $0 . 7 1 \mathrm { m W }$ </td><td> $0 . 7 0 \mathrm { m W }$ </td></tr><tr><td>4×4</td><td> $1 9 . 7 2 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td> $2 1 . 1 3 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td>696B</td><td>852 B</td><td>40B</td><td>56B</td><td> $0 . 4 5 \pm 0 . 0 5 \mathrm { | \vec { \mu } { \bf { J } } }$ </td><td> $0 . 4 0 \pm 0 . 0 8 \mu \mathrm { J }$ </td><td>0.70 mW</td><td>0.70mW</td><td> $0 . 7 2 \mathrm { m W }$ </td><td> $0 . 7 1 \mathrm { m W }$ </td></tr><tr><td>8×8</td><td> $3 9 . 5 0 \pm 0 . 0 2 \mu \mathrm { s }$ </td><td> $3 8 . 4 1 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td>1,016B</td><td>1,172B</td><td>64B</td><td>80B</td><td> $0 . 5 0 \pm 0 . 0 7 \mu \mathrm { J }$ </td><td> $0 . 5 5 \pm 0 . 0 8 \mathrm { \dot { \mu } J }$ </td><td>0.70 mW</td><td>0.70mW</td><td>0.73 mW</td><td> $0 . 7 4 \mathrm { m W }$ </td></tr><tr><td>16×16</td><td> $8 1 . 3 4 \pm 0 . 0 0 { \mathrm { } } { \mathrm { \mu { s } } }$ </td><td> $8 2 . 9 7 \pm 0 . 0 0 { \mathrm { \textmu s } }$ </td><td>2,040 B</td><td>2,200B</td><td>128 B</td><td>144B</td><td> $0 . 9 8 \pm 0 . 0 8 { \overset { \cdot } { \mu } } \mathbf { J }$ </td><td> $0 . 8 7 \pm 0 . 0 8 { \overset { \cdot } { \mu } } \mathbf { J }$ </td><td>0.72 mW</td><td>0.72 mW</td><td>0.78 mW</td><td> $0 . 7 7 \mathrm { m W }$ </td></tr><tr><td>32×32</td><td> $2 1 9 . 2 7 \pm 0 . 0 0 { \mathrm { \ddot { \mu s } } }$ </td><td> $2 0 6 . 6 4 \pm 0 . 0 1 \mathrm { \ddot { \mu s } }$ </td><td>5,624 B</td><td>5,780B</td><td>256B</td><td>272 B</td><td> $2 . 8 8 \pm 0 . 0 9 \mathrm { \dot { \mu } J }$ </td><td> $2 . 8 3 \pm 0 . 1 0 \mathrm { \dot { \mu } J }$ </td><td>0.81 mW</td><td>0.81 mW</td><td>0.95 mW</td><td>0.98 mW</td></tr><tr><td>2×2</td><td> $\overline { { 1 9 . 0 6 \pm 0 . 0 2 \mu \mathrm { s } } }$ </td><td> $2 3 . 9 3 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td>824 B</td><td>1,248 B</td><td>96B</td><td>144B</td><td> $\overline { { { 0 . 3 9 \pm 0 . 0 8 \mu \mathrm { J } } } }$ </td><td> $\overline { { 0 . 5 4 \pm 0 . 1 0 \mu \mathrm { J } } }$ </td><td>0.70 mW</td><td>0.70mW</td><td>0.70 mW</td><td>0.73 mW</td></tr><tr><td rowspan="4">Quadrotor</td><td>4x4</td><td> $2 8 . 8 0 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td> $3 4 . 1 4 \pm 0 . 0 1 \mu \mathrm { s }$ </td><td>1,028 B</td><td>1,452B</td><td>96B</td><td>144B 144B</td><td> $0 . 2 9 \pm 0 . 0 9 \mathrm { { \overset { . } { \mu } J } }$ </td><td> $0 . 3 4 \pm 0 . 0 9 \mathrm { { \overset { . } { \mu } J } }$ </td><td>0.69 mW</td><td>0.69mW</td><td>0.71 mW</td><td>0.72mW</td></tr><tr><td>8×8</td><td> $5 1 . 7 4 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td> $6 5 . 3 5 \pm 0 . 0 0 \mu \mathrm { s }$ </td><td>1,524 B</td><td>1,952 B</td><td>96B</td><td> $0 . 6 6 \pm 0 . 0 9 \mathrm { { \overset { . } { \mu } J } }$ </td><td></td><td> $0 . 6 9 \pm 0 . 0 7 \mu \mathrm { J }$ </td><td>0.71 mW</td><td>0.71 mW</td><td>0.73 mW</td><td>0.75 mW</td></tr><tr><td>16×16</td><td> $1 0 8 . 8 4 \pm 0 . 0 0 { \dot { \mu } } \mathrm { s }$ </td><td> $1 0 4 . 5 8 \pm 0 . 0 2 \mathrm { { \dot { \mu } s } }$ </td><td>2,868 B</td><td>3,296B</td><td>128B</td><td>176B</td><td> $1 . 3 5 \pm 0 . 0 8 \mathrm { \textmu J }$ </td><td> $1 . 3 7 \pm 0 . 0 7 \mathrm { \dot { \mu } J }$ </td><td>0.74 mW</td><td>0.74mW</td><td>0.80 mW</td><td>0.83 mW</td></tr><tr><td>32×32</td><td> $2 5 5 . 7 2 \pm 0 . 0 2 \mathrm { \mu s }$ </td><td> $2 5 8 . 9 8 \pm 0 . 0 1 \mathrm { \ddot { \mu s } }$ </td><td>7,096B</td><td>7,516B</td><td>256B</td><td>304B</td><td> $3 . 6 1 \pm 0 . 0 9 \mathrm { { \overset { . } { \mu } J } }$ </td><td> $3 . 6 7 \pm 0 . 0 7 \mathrm { \textmu J }$ </td><td>0.85 mW</td><td>0.85 mW</td><td>1.04 mW</td><td>1.01 mW</td></tr></table>

[2] M. Andrychowicz, A. Raichuk, P. Stanczyk, M. Orsini, S. Girgin,´ R. Marinier, L. Hussenot, M. Geist, O. Pietquin, M. Michalski et al., “What matters in on-policy reinforcement learning? a large-scale empirical study,” arXiv preprint arXiv:2006.05990, 2020.

[3] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, “Mcunet: Tiny deep learning on iot devices,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 11 711–11 722.

[4] L. Brunke, M. Greeff, A. W. Hall, Z. Yuan, S. Zhou, J. Panerati, and A. P. Schoellig, “Safe learning in robotics: From learning-based control to safe reinforcement learning,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 5, no. 1, pp. 411–444, 2022.

[5] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, “Evolution strategies as a scalable alternative to reinforcement learning,” arXiv preprint arXiv:1703.03864, 2017.

[1] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun, and¨ D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” Nature, vol. 620, no. 7976, pp. 982–987, 2023.

Future work will explore hardware-aware quantization within the Evolution Strategy (ES) loop to enable verified low-precision micro-policies on ultra-low-power devices, together with real-to-sim strategies to better ground probabilistic guarantees in physical systems.

## VIII. CONCLUSION

The low footprint also suggests opportunities for distributed control architectures. For example, in high-frequency multi-actuator systems, e.g., humanoid or quadruped robots, compact policies could be deployed on low-cost microcontrollers close to individual joints. This reduces reliance on centralized computation and enables greater flexibility in embedded controller design.

In this work, we demonstrate that neural policy size can be drastically reduced while retaining statistical safety and performance guarantees. Experiments on Cartpole and Quadrotor systems show that compact Micro Neural Policies (MNP) can achieve robust zero-shot sim-to-real transfer. On resource-constrained hardware, these policies require only 0.5–7.5 kB of memory and 0.25–3.67 µJ per inference, with less than 25 ns of jitter and over 97% of each control period remaining idle. These results indicate that safe continuous control can be achieved with substantially smaller neural architectures than commonly used in learning-based control.

## REFERENCES

[6] H. Mania, A. Guy, and B. Recht, “Simple random search of static linear policies is competitive for reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 31. Curran Associates, Inc., 2018.

[7] R. Curcio, H. Cao, and M. Caccamo, “Safe and robust neural policy learning with statistical verification for sim-to-real deployment in robotics,” 2026. [Online]. Available: https://arxiv.org/abs/2608.06481

[8] R. Curcio, T. Mancini, and E. Tronci, “Smc-es: Automated synthesis of formally verified control policies,” arXiv:2607.15003, 2026.

[9] R. David, J. Duke, A. Jain, V. Janapa Reddi, N. Jeffries, J. Li, N. Kreeger, I. Nappier, M. Natraj, T. Wang et al., “Tensorflow lite micro: Embedded machine learning for tinyml systems,” Proceedings of machine learning and systems, vol. 3, pp. 800–811, 2021.

[10] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-toreal transfer of robotic control with dynamics randomization,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3803–3810.

[11] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017, pp. 23–30.

[12] A. Rajeswaran, K. Lowrey, E. V. Todorov, and S. M. Kakade, “Towards generalization and simplicity in continuous control,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[13] B. Recht, “A tour of reinforcement learning: The view from continuous control,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 2, pp. 253–279, 2019.

[14] T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, “Residual policy learning,” arXiv preprint arXiv:1812.06298, 2018.

[15] T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, and S. Levine, “Residual reinforcement learning for robot control,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 6023–6029.

[16] H. Cao, Y. Mao, L. Sha, and M. Caccamo, “Physics-regulated deep reinforcement learning: Invariant embeddings,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 3712–3756.

[17] D. Livne and K. Cohen, “Pops: Policy pruning and shrinking for deep reinforcement learning,” IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 4, pp. 789–801, 2020.

[18] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” in Low-power computer vision. Chapman and Hall/CRC, 2022.

[19] A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” arXiv preprint arXiv:1511.06295, 2015.

[20] S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan, Linear Matrix Inequalities in System and Control Theory. Society for Industrial and Applied Mathematics, Jan. 1994.

[21] C. Jegourel, J. Sun, and J. S. Dong, “Sequential schemes for frequentist estimation of properties in statistical model checking,” ACM Transactions on Modeling and Computer Simulation (TOMACS), vol. 29, no. 4, pp. 1–22, 2019.

[22] M. Esposito, A. Leva, T. Mancini, L. Picchiami, and E. Tronci, “Simulation-based design of industry-size control systems with formal quality guarantees,” IEEE Transactions on Industrial Informatics, vol. 21, no. 5, pp. 3871–3879, 2025.

[23] S. P. Boyd and L. Vandenberghe, Convex optimization. Cambridge university press, 2004.

[24] Lui Sha, “Using simplicity to control complexity,” IEEE Software, vol. 18, no. 4, pp. 20–28, Jul. 2001.

[25] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy deep reinforcement learning with a stochastic actor,” in International Conference on Machine Learning (ICML), 2018.

[26] S. Ha, P. Xu, Z. Tan, S. Levine, and J. Tan, “Learning to walk in the real world with minimal human effort,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 16–18 Nov 2021, pp. 1110–1120.

[27] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” 2016.

[28] Z. Yuan, A. W. Hall, S. Zhou, L. Brunke, M. Greeff, J. Panerati, and A. P. Schoellig, “Safe-control-gym: A unified benchmark suite for safe learning-based control and reinforcement learning in robotics,” IEEE Robotics and Automation Letters, 2022.

[29] C. Banbury, V. J. Reddi, P. Torelli, J. Holleman, N. Jeffries, C. Kiraly, P. Montino, D. Kanter, S. Ahmed, D. Pau et al., “Mlperf tiny benchmark,” arXiv preprint arXiv:2106.07597, 2021.

[30] F. Kratz, “onnx2c: A portable onnx to c compiler,” https://github.com/ kraiskil/onnx2c, 2022.

[31] D. Palossi, A. Loquercio, F. Conti, E. Flamand, D. Scaramuzza, and L. Benini, “A 64mw dnn-based visual navigation engine for autonomous nano-drones,” arXiv preprint arXiv:1805.01831, 2018.

[32] E. Gabriel, G. E. Fagg, G. Bosilca, T. Angskun, J. J. Dongarra, J. M. Squyres, V. Sahay, P. Kambadur, B. Barrett, A. Lumsdaine et al., “Open mpi: Goals, concept, and design of a next generation mpi implementation,” in European Parallel Virtual Machine/Message Passing Interface Users’ Group Meeting. Springer, 2004, pp. 97–104.