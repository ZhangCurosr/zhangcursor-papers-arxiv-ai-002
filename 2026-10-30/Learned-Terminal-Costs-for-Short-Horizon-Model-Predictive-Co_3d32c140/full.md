# Learned Terminal Costs for Short-Horizon Model Predictive Control of Autonomous Trucks in Interactive Highway Trafic

Hooman Hamze<sup>∗</sup>   
Chalmers University of Technology   
Gothenburg, Sweden   
hoomanh@chalmers.se   
hooman.hamze230@gmail.com   
Erik Börve   
Chalmers University of Technology   
Gothenburg, Sweden   
Trucks Technology and Industrial   
Division, Volvo Group   
Gothenburg, Sweden   
borerik@chalmers.se

Benjamin Houshmand<sup>∗</sup> Chalmers University of Technology Gothenburg, Sweden benhou@chalmers.se

Morteza Haghir Chehreghani Chalmers University of Technology and University of Gothenburg Gothenburg, Sweden morteza.chehreghani@chalmers.se

Deepthi Pathare   
Chalmers University of Technology   
and University of Gothenburg   
Gothenburg, Sweden   
Trucks Technology and Industrial   
Division, Volvo Group   
Gothenburg, Sweden   
pathare@chalmers.se

Nikolce Murgovski Chalmers University of Technology Gothenburg, Sweden nikolce.murgovski@chalmers.se

## Abstract

Autonomous highway driving for heavy-duty trucks is a challenging control problem. The greater mass, dimensions and intricate dynamics increase the requirements for cooperation from surrounding trafic in critical maneuvers, requiring a careful balance between safety constraints and eficient long-horizon decision making. Model Predictive Control (MPC) provides rigorous constraint satisfaction but relies on terminal cost which can limit its efectiveness at short, computationally afordable horizons. Reinforcement Learning (RL) captures long-term driving behavior but ofers no constraint guarantees and tends toward overly conservative policies when trained under sparse collision penalties. In this paper, we propose a hybrid MPC and RL framework for highway driving in autonomous trucks, where a finite-horizon MPC, solved via dynamic programming, uses a pre-trained RL model for terminal cost estimation. We evaluate the framework in a simulation environment with reactive surrounding trafic and compare it against pure MPC with zero terminal cost, rollout with terminal cost estimation using a base policy, and pure RL. Results show that the hybrid approach achieves zero collisions with the lowest cost of any configuration.

## Keywords

Autonomous Driving, Reinforcement Learning, Model Predictive Control, Dynamic Programming, Heavy Duty Vehicle

## 1 Introduction

Autonomous driving promises three concrete benefits over humandriven vehicles: improve safety, since the large majority of road accidents are caused by human error rather than mechanical failure [28]; cost eficiency, since speed and fuel consumption profiles can be optimized directly rather than left to individual driving style [32]; and continuous utilization, since an automated vehicle is not bound by mandatory rest periods, fatigue, or shift limits and can in principle operate around the clock [30]. These benefits are amplified for heavy-duty commercial vehicles such as long-haul trucks, where driver wages are the largest single component of operating cost [14] and where even small eficiency gains scale directly with fleet size.

Achieving a reliable autonomous vehicle is, however, considerably harder for heavy vehicles than for passenger cars. A truck pulling a trailer has coupled dynamics that a single rigid-body vehicle does not. The trailer does not track the tractor’s path directly, which makes both motion planning and the underlying control problem more involved [6]. Its size and limited maneuverability mean that routine highway actions such as lane changes and merges are harder and require more cooperation from surrounding trafic [5]. Highway driving is, in addition, inherently interactive for any vehicle. An agent’s own actions influence what nearby drivers do next, so a controller that reacts only to the present state, without any model of how the situation might unfold, tends to be either too timid to be useful or too naive to be safe [5]. Meeting all of these requirements at once is the central dificulty this paper addresses.

Two methodological families dominate the literature on autonomous vehicle decision-making and control: Model Predictive Control (MPC) and Reinforcement Learning (RL) [27]. MPC formulates driving as a constrained optimization problem. Given a model of the vehicle and its surroundings, solve for the control sequence that minimizes a cost over a finite look-ahead horizon subject to constraints on speed, acceleration, steering, and collision avoidance, then apply only the first action and re-solve at the next step. Beyond the visible horizon, the controller must fall back on an approximate terminal cost standing in for everything that happens afterwards. A short horizon with a crude terminal cost is fast to solve but can behave poorly, while a longer horizon is more reliable but increasingly expensive online. RL, by contrast, learns a decision policy directly from interaction, with no requirement that the dynamics be known, which suits the strategic aspects of driving. However, a learned policy ofers no guarantee its output will respect a physical or safety constraint, and its behavior outside the situations it was trained on is unpredictable, which has motivated explicit out-of-distribution detection for learned driving policies [11].

Several works rely on MPC alone for interactive highway maneuvers. Börve et al. [6] couple an autonomous truck’s trajectory optimization with an iterative update against a trafic predictor to capture genuine interactivity during forced lane changes, but at the cost of solving several parallel nonlinear optimizations every control step, with no formal convergence guarantee and a terminal cost fixed by design. Brito et al. [5] use MPC with a hand-tuned, fixed velocity reference for merging in dense trafic and report a markedly higher collision rate than the same architecture using a learned reference instead.

RL alone has also been used directly for tactical driving decisions. Acquarone et al. [1] apply a TD3 agent to cooperative adaptive cruise control for heavy-duty battery electric vehicles, achieving substantial energy savings by exploiting air-drag reduction in a platoon, though the controller is longitudinal only and does not address lane-level decisions. Hoel et al. [10] trained a Double-DQN agent to choose discrete lane-change and speed actions for a truck-trailer combination, but like every pure-RL controller, without formal safety guarantees at all. Later in [11], they proposed a bayesian RL technique that estimates uncertainty of decisions to choose safe actions, measurably reducing collisions outside the training distribution at the cost of a hand-set threshold and roughly double the training time. Pathare et al. [23] separated high level and low decision making between RL and physics based controllers, which improved safety and driving eficiency relative to letting RL act directly. However, the fixed rule-based execution layer carries none of MPC’s constraint guarantees.

Motivated by these complementary shortcomings, a number of recent works combine MPC and RL. A first group targets MPC’s tunable parameters. Gupta et al. [9] add an RL-trained throttle correction in parallel with an MPC. Chen et al. [7] use RL to choose the prediction horizon for single-vehicle path tracking beating fixed horizon baselines, but without any surrounding trafic. Zarrouki et al. [33] pre-compute a catalog of cost-function weight sets that are each individually verified safe, and let an RL agent only ever choose among them for vehicle motion control. Wang et al. [31] let RL tune a single MPC weight conditioned on a risk classification for car-following. A second group interfaces a learned component with MPC at the level of a discrete decision or reference trajectory. Mallick et al. [17] propose a learned gear-shift schedule fed into a continuous MPC. Reiter et al. [26] let an RL policy reshape an MPC’s entire reference trajectory and cost weighting for autonomous racing, beating both a non-strategic MPC and an end-to-end RL policy while remaining safe. Albarella et al. [2] combine a Double-DQN decision layer with two NMPCs for highway driving, though the NMPC terminal cost stays fixed. In all of these, learning never touches the terminal cost itself.

A separate line of work uses RL to approximate the terminal cost directly. Bertsekas [4] provides the theoretical foundation, showing that one step of MPC look-ahead on top of an approximate terminal cost is equivalent to one Newton step on the Bellman equation, so even a mediocre approximation yields a near-optimal policy after suficient look-ahead. Building on this, Morita et al. [22] use a learned value function as the terminal cost of an MPPI controller for a robot; Lin et al. [15] and Moreno-Mora et al. [21] learn it via policy and value iteration respectively for general nonlinear systems; and Kang et al. [12] learn it online for a robotic manipulator. All of these are validated on continuous low-level control in single-agent, nonreactive settings, and none addresses the discrete tactical decisions or interactive multi-agent trafic that motivate the present work.

Two limitations cut across this literature. First, every learnedterminal-cost method operates on continuous low-level control; none addresses a discrete tactical decision such as a lane change, where the terminal cost must interact with a combinatorial DP optimization. Second, the entire terminal-cost-learning line is validated only in single-agent, non-reactive settings. It is not tested in an interactive environment where surrounding vehicles respond to the ego and thereby shift the very state distribution the value function was trained on. Conversely, the hybrid schemes that do operate in trafic [7, 9, 17, 31, 33] confine learning to horizon selection, a handful of cost weights, or an additive correction, and never learn the terminal cost that governs beyond-horizon foresight. No prior work integrates a learned or even rollout-based terminal cost into a MPC for a discrete tactical highway-driving problem, nor evaluates whether such a terminal cost remains efective when surrounding trafic is genuinely reactive.

We develop a highway controller for an autonomous truck that solves its finite-horizon MPC problem by dynamic programming over a discretized state-action space, and replaces the usual handdesigned terminal cost with a DQN-learned terminal cost, trained ofline and inserted directly into the DP backward pass. For the purpose of benchmarking, a rollout terminal cost approximations, obtained by simulating a working base policy forward from the end of the horizon is also developed. Both retain MPC’s exact constraint handling and receding-horizon replanning. Our contributions are as follows:

(1) We propose a hybrid MPC and RL framework for autonomous highway driving of a heavy-duty truck. The MPC performs speed control and lane-change decisions over a discrete state-action grid, while a pre-trained Deep Q-Network (DQN) provides the terminal cost to improve long-term decision making.

(2) We evaluate the efectiveness of the proposed framework in a highway trafic simulation by comparing it with conventional MPC and rollout-based terminal cost methods. The proposed DQN-based terminal cost consistently improves safety and decision consistency under both deterministic and noisy observation settings.

(3) We demonstrate that incorporating a learned terminal cost enables short-horizon MPC to achieve the best balance between safety, driving performance, and computational efficiency. The proposed three-step MPC eliminates the need for computationally expensive long-horizon optimization while remaining suitable for real-time implementation.

## 2 PROBLEM FORMULATION

We consider an autonomous heavy truck operating on a multi-lane highway in the presence of surrounding trafic. Surrounding vehicles are not controlled by the ego truck, but respond to its actions through their own car-following and lane-changing models [8, 13], so the environment is interactive and the optimal policy depends on multi-step predictions of how trafic will evolve.

## 2.1 System Model

The ego truck’s state at step $k$ is:

$$
\boldsymbol { x } _ { k } ^ { e } = [ s _ { k } ^ { e } , l _ { k } ^ { e } , v _ { k } ^ { e } , c _ { k } ^ { e } ] ^ { \top }\tag{1}
$$

where $s _ { k } ^ { e }$ is longitudinal position, $l _ { k } ^ { e } \in \{ 0 , \ldots , N _ { \mathrm { l a n e s } } - 1 \}$ is lane index and $N _ { \mathrm { l a n e s } }$ is the number of lanes, $v _ { k } ^ { e } \in [ v _ { \operatorname* { m i n } } ^ { e } , v _ { \operatorname* { m a x } } ^ { e } ]$ is velocity with $v _ { \mathrm { m i n } } ^ { e }$ and $v _ { \mathrm { m a x } } ^ { e }$ as its bounds, and $c _ { k } ^ { e }$ is a lane-change counter that enforces transition and cooldown durations. The control input:

$$
\boldsymbol { u } _ { k } ^ { e } = [ a _ { k } ^ { e } , \Delta l _ { k } ^ { e } ] ^ { \top }\tag{2}
$$

pairs a discretized longitudinal acceleration $a _ { k } ^ { e }$ with a discrete lanechange command $\Delta l _ { k } ^ { e } \in \{ - 1 , 0 , 1 \}$ . The ego dynamics

$$
x _ { k + 1 } ^ { e } = f ( x _ { k } ^ { e } , u _ { k } ^ { e } )\tag{3}
$$

evolve under a discrete-time kinematic model; the lane and counter update jointly, so a new lane-change command is accepted only once the counter has cleared its transition time and the ego truck reaches its target lane. Section 3.2 provides more details on the lane-change command and the counter.

Surrounding vehicles have states $x _ { k } ^ { T } = \{ x _ { j , k } ^ { T } \} _ { j \in N _ { k } }$ , where $N _ { k }$ is the set of vehicles within sensor range at step �. Their future states $x _ { i } ^ { T }$ at step $i > k$ are uncertain for two reasons. First, the initial observations $x _ { k } ^ { T }$ are subject to sensor measurement noise. Second, even with perfect initial observations, surrounding vehicles’ future behavior can only be approximated by a simplified prediction model. They respond to each other and to the ego in ways that are not fully predictable. Both sources of uncertainty accumulate over the prediction horizon and must be accounted for in the controller design. A detailed description of the prediction model is provided in Section 3.1.

## 2.2 The MPC Problem

The fundamental control objective is to find a policy � minimizing the expected cumulative cost over the full trafic state $x _ { k } = ( x _ { k } ^ { e } , x _ { k } ^ { T } ) ;$

$$
\operatorname* { i n f } _ { \pi } ~ \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \ell ( x _ { k } , u _ { k } ) \right]\tag{4}
$$

where the expectation is over the stochastic evolution of surrounding trafic. Solving Eq. (4) directly is intractable because it requires optimizing over an infinite sequence of future actions under physical constraints and uncertain trafic. MPC provides a tractable approximation by truncating the optimization horizon to � steps and re-optimizing at every step using the latest observations. Given the discretized state and action spaces, the resulting finite-horizon problem can be solved online using Dynamic Programming (DP). However, performing the DP backward pass over the full joint state space $( x _ { k } ^ { e } , x _ { k } ^ { T } )$ would be intractable, as the state grid grows exponentially with the number of surrounding vehicles [3]. The

DP is therefore performed over the ego vehicle’s state space $x ^ { e }$ while the predicted surrounding vehicle states are treated as time varying parameters during the optimization. This is reflected in the conditional notation $\ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { \bar { T } } ) ;$ : the ego trajectory is optimized while the surrounding states enter as given parameters at each step.

Because the surrounding vehicle states $x _ { i } ^ { T }$ are uncertain, we formulate the control task as a chance-constrained MPC [19]. At each step �, the controller solves:

$$
\operatorname* { m i n } _ { u _ { 0 } ^ { e } , \dots , u _ { N - 1 } ^ { e } } \quad \mathbb { E } \left[ \sum _ { i = 0 } ^ { N - 1 } \gamma ^ { i } \ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { T } ) + \gamma ^ { N } V _ { f } ( x _ { N } ^ { e } \mid x _ { N } ^ { T } ) \right]\tag{5a}
$$

$$
\quad \mathrm { s . t . } \quad x _ { i + 1 } ^ { e } = f ( x _ { i } ^ { e } , u _ { i } ^ { e } ) ,
$$

$$
\mathbb { P } \big [ x _ { i } ^ { e } \in \mathbb { X } ^ { T } ( x _ { i } ^ { T } ) \big ] \geq 1 - \epsilon ,\tag{5b}
$$

$$
\begin{array} { r } { x _ { i } ^ { e } \in \mathbb { X } ^ { e } , \quad u _ { i } ^ { e } \in \mathbb { U } ^ { e } , } \end{array}\tag{5c}
$$

(5d)

$$
x _ { 0 } ^ { e } = x _ { k } ^ { e } , \quad \forall i \in \{ 0 , \ldots , N - 1 \}\tag{5e}
$$

where $\ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { T } ) \ge 0$ is a stage cost penalizing undesired ego actions and states, conditioned on surrounding vehicle states; $V _ { f } ( \cdot )$ approximates the cost-to-go beyond the horizon; $\mathbb { X } ^ { T } ( x _ { i } ^ { T } )$ is the set of ego states that are safe with respect to the surrounding vehicles at step $i ; \mathbb { X } ^ { e }$ and $\mathbb { U } ^ { e }$ encode state and action limits; $\epsilon \in ( 0 , 1 )$ is the risk tolerance for collision avoidance; and $\gamma \in ( 0 , 1 ]$ is a discount factor. Unlike the undiscounted objective common in classical MPC, $\gamma ^ { i }$ down-weights stage costs at later horizon steps. Section 3.3 discusses the motivation for this choice.

Only the first optimal action $u _ { 0 } ^ { e * }$ is applied, and the state is then re-measured and the problem re-solved at $k + 1$ . This corrects automatically for prediction errors as new observations arrive.

The chance constraint Eq. (5c) treats $x _ { i } ^ { T }$ as a random variable incorporating both measurement noise and prediction model uncertainty, and requires the ego to remain safe with probability at least $1 - \epsilon .$ Under a Gaussian uncertainty model, this constraint admits a deterministic equivalent [19] in which the required safe distance is inflated by a margin proportional to the predicted position standard deviation. Suppressing that margin recovers a deterministic controller acting on the nominal prediction $\bar { x } _ { i } ^ { T }$ , which serves as a baseline to isolate the benefit of uncertainty handling.

The design burden of Eq. (5) is the terminal cost $V _ { f }$ . Classical stability theory [18] requires $V _ { f } ,$ , an associated terminal constraint set $\mathbb { X } _ { f } ^ { e } ,$ and a local controller to jointly satisfy a set of Lyapunovtype conditions, which is already dificult to construct for general nonlinear systems [25]. In the present setting, this dificulty is compounded by the presence of surrounding vehicles. The terminal set $\mathbb { X } _ { f } ^ { e }$ must be control-invariant under the terminal controller, but invariance cannot be guaranteed when surrounding vehicles are free to arbitrarily enter or leave the ego’s vicinity beyond the planning horizon.

Solving Eq. (5) by DP makes it straightforward to substitute alternative terminal cost approximations $V _ { f } ( x _ { N } ^ { e } \mid x _ { N } ^ { T } )$ into the same backward pass.

## 2.3 The Terminal Cost Approximation Problem

The quantity that $V _ { f }$ approximates can be characterized exactly. Viewing the same task as a Markov Decision Process (MDP) [29], the agent at state $x _ { k } ^ { e }$ selects $u _ { k } ^ { e }$ under a policy �, incurs cost $\ell ( x _ { k } ^ { e } , u _ { k } ^ { e }$ |

$x _ { k } ^ { T } )$ , and transitions to $x _ { k + 1 } ^ { e }$ . The value function of �,

$$
V ^ { \pi } ( x ^ { e } ) = \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } \ \ell ( x _ { k } ^ { e } , u _ { k } ^ { e } \mid x _ { k } ^ { T } ) \ \middle | \ x _ { 0 } ^ { e } = x ^ { e } , \ u _ { k } ^ { e } = \pi ( x _ { k } ^ { e } ) \right]\tag{6}
$$

satisfies the Bellman equation:

$$
V ^ { \pi } ( x ^ { e } ) = \ell ( x ^ { e } , \pi ( x ^ { e } ) \mid x ^ { T } ) + \mathbb { E } { \bigl [ } \gamma V ^ { \pi } ( f ( x ^ { e } , \pi ( x ^ { e } ) ) \mid x ^ { T } ) { \bigr ] }\tag{7}
$$

and the optimal value function minimizes over the action at every state:

$$
V ^ { * } ( x ^ { e } ) = \operatorname* { m i n } _ { u ^ { e } } \bigl \{ \ell ( x ^ { e } , u ^ { e } \mid x ^ { T } ) + \mathbb { E } \bigl [ \gamma V ^ { * } ( f ( x ^ { e } , u ^ { e } ) \mid x ^ { T } ) \bigr ] \bigr \}\tag{8}
$$

Setting $V _ { f } = V ^ { * }$ in $\operatorname { E q . } \left( 5 \right)$ makes the truncated and unconstrained (we cannot exactly compute the target cost if constraints are imposed beyond the horizon) �-step problem exact, and its solution coincides with the optimal infinite-horizon policy for any $N \geq 1$ since $V ^ { * }$ accounts precisely for all cost incurred beyond the horizon. This holds because Eq. (5) is the Bellman recursion unrolled � times, which requires the discount factor of the MPC objective, the discounting applied to the terminal cost, and the discount factor defining $V ^ { * }$ to be identical; a mismatch would make the truncated problem a diferent optimization rather than an exact reformulation. $V ^ { * }$ is therefore the ideal terminal cost, and it encodes exactly the information the MPC lacks beyond step �.

However, for the constrained system considered here, �<sup>∗</sup> cannot be obtained. The dynamics depend on surrounding vehicles whose behavior is not modeled in closed form, and the recursion Eq. (8) admits no analytical solution. The terminal cost must therefore be approximated. Hence, the design question becomes how much approximation error can be tolerated. Bertsekas [4] shows that solving Eq. (5) with $V _ { f } \approx V ^ { * }$ and applying only the first action is equivalent to one Newton step on the Bellman optimality equation $\operatorname { E q . }$ (8). The closed-loop policy is consequently more accurate than $V _ { f }$ itself since an imperfect terminal cost is substantially corrected by even a few steps of look-ahead, and the residual error decreases sharply as � increases. A short horizon is thus suficient provided $V _ { f }$ is a reasonable approximation of �<sup>∗</sup>, which is what makes the terminal cost, rather than the horizon length, the efective design variable.

Section 3 presents an RL-based model to obtain $V _ { f } \approx V ^ { * }$ , to be used directly in the DP backward pass as the terminal cost.

## 3 Proposed Method

Figure 1 illustrates the architecture of proposed hybrid method. At each step, the controller observes the current trafic state, constructs a discretized state-action grid, and evaluates the terminal cost, then runs the backward DP to recover the optimal action sequence. Only the first action is sent to simulator, and the loop repeats. For benchmarking, we also evaluate pure MPC, which sets $V _ { f } = 0 ;$ a rollout-based terminal cost, which simulates a base policy forward from the horizon boundary; and pure RL, which applies the learned policy directly without any MPC optimization.

## 3.1 Trajectory Prediction

Evaluating the safety terms of the cost in Section 3.2 over the prediction horizon requires an estimate of each surrounding vehicle’s future position, speed, and lane at every horizon step $i = 1 , \ldots , N _ { ; }$ since their true future behavior is not known to the controller. To model typical highway driving behavior, surrounding vehicles are extrapolated forward from their currently observed states under a constant-acceleration assumption, holding each vehicle’s lane fixed for the duration of the relatively short horizon. Among the prediction strategies evaluated, this approach achieves the lowest mean position error at all tested horizon lengths, attributed to the short timescales involved where the current acceleration is the dominant predictor of near-term position.

Ego vehicle interacting with other vehicles in SUMO environment  
![](images/63a09cc68a822733ab53718a120e3fe2b7ec49117a269b4572b509242e48d691.jpg)  
Figure 1: Architecture of controller

However, this prediction is afected by the uncertainty in initial measurements. Each surrounding vehicle’s initial position, speed, and acceleration are observed with additive Gaussian noise:

$$
\tilde { s } _ { j , 0 } = s _ { j , 0 } + \nu _ { s } , \quad \nu _ { s } \sim N ( 0 , \sigma _ { s } ^ { 2 } )\tag{9a}
$$

$$
\tilde { v } _ { j , 0 } = v _ { j , 0 } + \nu _ { v } , \quad \nu _ { v } \sim N ( 0 , \sigma _ { v } ^ { 2 } )\tag{9b}
$$

$$
\tilde { a } _ { j , 0 } = a _ { j , 0 } + \nu _ { a } , \quad \nu _ { a } \sim N ( 0 , \sigma _ { a } ^ { 2 } )\tag{9c}
$$

with $\nu _ { s } , \nu _ { v } , \nu _ { a }$ independent. Propagating these through the constantacceleration model yields a Gaussian distribution over the predicted position at horizon step �, with variance:

$$
\begin{array} { r } { \sigma _ { s , i } ^ { 2 } = \sigma _ { s } ^ { 2 } + ( i \Delta t ) ^ { 2 } \sigma _ { v } ^ { 2 } + \left( \frac { i ( i - 1 ) } { 2 } ( \Delta t ) ^ { 2 } \right) ^ { 2 } \sigma _ { a } ^ { 2 } } \end{array}\tag{10}
$$

so positional uncertainty grows with the horizon step, quadratically in the speed noise and quadratically in the acceleration noise.

A neighbor vehicle that is actively executing a lane change at the time of prediction is handled separately, since its true lane occupancy is ambiguous while it is physically transitioning between lanes. Rather than committing it to either lane, such a vehicle is represented by two copies in the predicted trajectory set, the original vehicle on its origin lane and a duplicate ghost vehicle, with identical position, speed, and acceleration, placed on its target lane. This ensures the safety and collision terms of ℓ account for the neighbor vehicle’s presence in both lanes during the transition, preventing the controller from planning a maneuver into a gap that is already being closed by an in-progress lane change. More details of the tested prediction models are provided in Appendix B.

## 3.2 Cost Function under Uncertainty

The stage cost $\ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { T } )$ combines seven terms into a single weighted sum:

$$
\ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { T } ) = \ell _ { \mathrm { s p e e d } } + \ell _ { \mathrm { a c c e l } } + \ell _ { \mathrm { L C } } + \ell _ { \mathrm { C D } } + \ell _ { \mathrm { s a f e t y } } + \ell _ { \mathrm { L L } } + \ell _ { \mathrm { c o l } }\tag{11}
$$

Of these, only the safety and collision terms depend on the predicted states of surrounding vehicles, and therefore only these are afected by the uncertainty. The remaining terms are functions of the ego state alone and are unafected.

The speed and acceleration terms shape the nominal driving behavior:

$$
\ell _ { \mathrm { s p e e d } } = w _ { v } ( v ^ { e } - v _ { \mathrm { r e f } } ) ^ { 2 } , \qquad \ell _ { \mathrm { a c c e l } } = w _ { a } ( a ^ { e } ) ^ { 2 }\tag{12}
$$

where $v _ { \mathrm { r e f } }$ is the reference speed and $w _ { v } , w _ { a } > 0$ are tunable weights penalizing speed deviation and control efort.

A lane change is penalized once per maneuver, at the step it is initiated, using the counter $c ^ { e }$ that governs the transition lock:

$$
\ell _ { \mathrm { L C } } = { \left\{ \begin{array} { l l } { P _ { \mathrm { L C } } } & { { \mathrm { i f ~ } } c ^ { e } \leq C _ { \mathrm { s t a r t ~ a n d ~ } } \Delta l ^ { e } \not = 0 } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{13}
$$

where $\mathrm { \mathit { P } _ { L C } } > 0$ is a fixed penalty and $C _ { \mathrm { s t a r t } }$ is the counter threshold below which a new lane change is permitted. A separate cooldown penalty $\ell _ { \mathrm { C D } } ,$ with a larger fixed value $\begin{array} { r } { P _ { \mathrm { C D } } > P _ { \mathrm { L C } } } \end{array}$ , discourages a second maneuver shortly after the first without making necessary back-to-back changes prohibitively costly. A progressive left-lane penalty $\ell _ { \mathrm { L L } } = w _ { \mathrm { L L } } \cdot l ^ { e }$ , zero in the rightmost lane, encourages the truck to return to the slow lane once an overtaking maneuver is no longer beneficial.

The safety and collision terms depend on the bumper-to-bumper gaps $g _ { \mathrm { f r o n t } }$ and $g _ { \mathrm { b a c k } }$ to the relevant neighbors, which are computed from the predicted surrounding vehicles’ states. Safety and collision avoidance requirement of Eq. (5c) is imposed on both front and back gaps. For front gap:

$$
\mathbb { P } \big ( \tilde { g } _ { \mathrm { f r o n t } , i } \ge d _ { \mathrm { s a f e } } \big ) \ge 1 - \epsilon\tag{14}
$$

where $d _ { \mathrm { s a f e } } = D _ { 0 } + T _ { \mathrm { h e a d } } v ^ { e }$ is the speed-dependent safe distance, $D _ { 0 }$ a minimum gap, and $T _ { \mathrm { h e a d } }$ a time headway. Since $\tilde { g } _ { \mathrm { f r o n t } , i }$ is Gaussian with standard deviation $\sigma _ { s , i }$ , this admits the deterministic equivalent:

$$
d _ { \mathrm { s a f e } } ^ { \mathrm { r o b u s t } } ( i ) = d _ { \mathrm { s a f e } } + \Phi ^ { - 1 } ( 1 - \epsilon ) \cdot \sigma _ { s , i }\tag{15}
$$

where $\Phi ^ { - 1 }$ is the inverse Gaussian CDF. Because $\sigma _ { s , i }$ grows with � by Eq. (10), the controller becomes progressively more conservative at later horizon steps. Similar logic applies to back gap.

Safety is enforced as a quadratic penalty when a gap falls below $d _ { \mathrm { s a f e } } ^ { \mathrm { r o b u s t } }$ , applied to the front vehicle at all times and to the rear vehicle on the target lane whenever a lane change is commanded:

$$
\begin{array} { r } { \ell _ { \mathrm { s a f e t y } } = w _ { \mathrm { s a f e } } \left( \operatorname* { m a x } \left( 0 , d _ { \mathrm { s a f e } } ^ { \mathrm { r o b u s t } } ( i ) - g _ { \mathrm { f r o n t } } \right) ^ { 2 } \right. } \\ { \left. + \delta _ { \mathrm { l c } } \operatorname* { m a x } \left( 0 , d _ { \mathrm { s a f e } } ^ { \mathrm { r o b u s t } } ( i ) - g _ { \mathrm { b a c k } } \right) ^ { 2 } \right) } \end{array}\tag{16}
$$

where $w _ { \mathrm { s a f e } } > 0$ and $\delta _ { \mathrm { l c } } = 1$ if $\Delta l ^ { e } \neq 0 .$

A large fixed penalty $\ell _ { \mathrm { c o l } } = P _ { \mathrm { c o l } }$ is charged whenever the front gap becomes non-positive, or whenever a commanded lane change would place the ego closer to a faster rear vehicle on the target lane than that vehicle could brake to match within its maximum deceleration. This braking-distance check is more conservative than a simple gap test, since a positive but rapidly closing rear gap can still result in a collision shortly after the maneuver completes. It is necessary because the DP assigns no back collision penalty to a vehicle already in the ego’s lane, and penalizes rear risk on the target lane only during a commanded change.

## 3.3 Discounted Objective

The discount factor $\gamma$ introduced in Eq. (5) departs from the undiscounted objective common in classical MPC, and is motivated by the structure of the uncertainty in this setting. As � grows, measurement noise accumulates through the prediction model and the constant-acceleration assumption of Section 3.1 degrades, so costs evaluated near $i \ = \ N$ rest on weaker evidence than those near $i = 1$ . The factor $\gamma ^ { i }$ down-weights them accordingly, preventing the controller from over-committing to uncertain predictions.

The same factor $\gamma ^ { N }$ is applied to the terminal cost, so that $V _ { f }$ is discounted consistently with the stage costs it summarizes. Using the same � in RL training additionally ensures that the learned $V _ { f }$ and the accumulated stage costs are expressed on a common scale, which is required for the exactness argument of Section 2.3.

## 3.4 RL-Based Terminal Cost Approximation

Following the MDP formulation of Section 2.3, the agent observes a 14-dimensional state comprising the ego speed and lane index, bumper-to-bumper gaps and relative velocities to the nearest neighbor vehicle in the current lane and each adjacent lane, and two indicators encoding the phase of an in-progress lane change or cooldown. The action space is the same acceleration and lanechange pair used by the MPC. The reward is the negative of the MPC stage cost, $r _ { i } = - r _ { \mathrm { s c a l e } } \ \ell ( x _ { i } ^ { e } , u _ { i } ^ { e } \mid x _ { i } ^ { T } )$ , where $r _ { \mathrm { s c a l e } }$ is for that the learned value function and the MPC’s own cost share a scale by construction. A Q-network is trained via standard DQN [20] to approximate �<sup>∗</sup>. DQN is used rather than an actor-critic method such as PPO because the quantity required at the horizon bound ary is an estimate of the optimal cost-to-go. The identity $V ^ { * } ( x ) =$ max $u Q ^ { * } ( x , u )$ makes an of-policy �-network an estimate of $V ^ { * }$ by construction, whereas an actor-critic critic estimates $V ^ { \pi }$ for the policy being learned, which would bias the DP toward that policy’s behavior rather than the optimum of the MPC objective.

Given the reward definition above, the optimal Q-function satisfies:

$$
Q ^ { * } ( x ^ { e } , u ^ { e } ) \approx - r _ { \mathrm { s c a l e } } \mathbb { E } \left[ \sum _ { i = 0 } ^ { \infty } \gamma ^ { i } \ell ( x _ { k + i } ^ { e } , u _ { k + i } ^ { e } \mid x _ { k + i } ^ { T } ) \right]\tag{17}
$$

so $V ^ { * } ( . )$ can be rescaled into the ${ \cal M P C } ^ { \prime } s$ cost units as a terminal cost:

$$
\hat { V } _ { f } ( x _ { N } ^ { e } \mid x _ { N } ^ { T } ) = - \frac { \operatorname * { m a x } _ { u } ~ Q ( x _ { N } ^ { e } , u ^ { e } ; \theta ) } { r _ { \mathrm { s c a l e } } }\tag{18}
$$

where � denotes the trained Q-network parameters.

Since the DP state $x _ { N } ~ = ~ [ s _ { N } ^ { e } , l _ { N } ^ { e } , v _ { N } ^ { e } , c _ { N } ^ { e } ] ^ { \top }$ <sup>⊤</sup> and the RL observation $z _ { N } \in \mathbb { R } ^ { 1 4 }$ use diferent representations, a mapping $z _ { N } =$ $\phi ( x _ { N } ^ { e } , \mathcal { T } _ { N } )$ converts one into the other before the Q-network is queried. The mapping constructs �<sub>�</sub> from the predicted trafic state $\bar { \mathcal { T } _ { N } } = \{ ( s _ { j , N } ^ { T } , v _ { j , N } ^ { T } , \bar { l } _ { j , N } ^ { T } ) \} _ { \mathrm { . } }$ <sub>�</sub> at horizon step � by computing the relevant gaps, relative velocities, and lane-change indicators that form the 14-dimensional RL observation.

## 3.5 Rollout-Based Terminal Cost Approximation

To provide a baseline for the RL-based approach, we additionally adapt the approach of [3] to our setting. Rather than tabulating a cost-to-go via DP over an extended horizon, the rollout terminal cost extends planning online. From each terminal state $x _ { N } ^ { e } , \varepsilon$ base policy is simulated forward for $N _ { r }$ further steps, and the resulting stage costs are accumulated into $\hat { V } _ { f _ { r } } ( x _ { N } ^ { e } \mid \bar { x } _ { N } ^ { T } )$ . The base policy re-implements the Krauss car-following model [13] used internally by simulator for longitudinal propagation of surrounding vehicles, holding their lanes fixed. The rollout therefore does not invoke the simulator at runtime, instead using this lightweight reimplementation directly for � steps. The ego vehicle additionally considers a lane change during the rollout when the gaps to the target lane’s leader and follower are safe and the resulting speed is closer to the reference.

Both the DP backward pass and the rollout apply the same discount factor $\gamma ,$ so a cost incurred at rollout step � beyond the horizon carries the same weight it would under a longer single DP horizon, keeping the combined $N + N _ { r }$ -step estimate consistent across the MPC/rollout boundary.

## 4 Results and Discussion

## 4.1 Experimental Setup

The proposed framework is evaluated in a simulation environment developed using Simulation of Urban MObility (SUMO) [16], an open-source microscopic trafic simulator. The controller interacts with the environment through TraCI, SUMO’s Trafic Control Interface, which provides surrounding vehicle states and accepts longitudinal and lateral commands for the ego vehicle at each step.

The scenario is a three-lane, one-way highway segment. The ego vehicle is a heavy-duty truck, and surrounding vehicles follow SUMO’s Krauss car-following model [13] and LC2013 lanechange model [8], with desired speeds sampled uniformly within a fixed range, maintaining an approximately constant trafic density around the ego. Each episode runs for a fixed maximum duration or terminates on a collision. The dynamics and cost parameters are listed in Appendix D.

## 4.2 RL Training

The DQN is trained using Stable-Baselines3 [24] in the deterministic simulation environment, under a range of hyperparameter settings summarized in Appendix D. Training with perfect state observations allows the agent to focus on learning long-term cost structure without the added variance of noisy measurements; the resulting value function is then used as a terminal cost within the stochastic MPC formulation, where measurement uncertainty is handled explicitly by the chance constraint.

![](images/4014311415c56f1ec001f5f0ff107d5a37135ed6c4da2f2f8cab5cebf4f20a9f.jpg)  
Figure 2: Holdout evaluation reward across five training runs, shown as mean ± one standard deviation.

In every run, the model is checkpointed every 100k steps, and each checkpoint is evaluated on 10 holdout seeds with deterministic actions. Since the evaluation reward is dominated by rare, high-cost collision events, holdout performance varies between checkpoints, so a two-stage selection is used: the ten highest-scoring checkpoints are re-evaluated on 500 independent seeds, and the final model is chosen on mean reward and mean speed, the latter excluding policies that avoid collisions only by driving well below the reference speed. To assess reproducibility, the selected configuration was trained five times with this procedure.

Figure 2 shows the holdout reward across the five runs. All runs improve rapidly within the first few million steps and fluctuate around a common level thereafter. Every run produced a collision-free policy at normal driving speed, with a mean reward $0 \mathrm { f } - 1 7 . 0 0 \pm 8 . 6 3$ and average speeds of 20.1–20.8 m/s across runs (Table 6, Appendix D). The model from the first run is used as the terminal cost throughout.

## 4.3 Controller Comparison

The proposed hybrid MPC-RL controller and three baselines are evaluated on 100 independent stochastic episodes. Deterministic results, obtained by suppressing measurement noise and setting $\epsilon  0 . 5$ which gives $\Phi ^ { - 1 } ( 1 - \epsilon ) = 0$ to recover the nominal predicted gaps, are reported in Appendix A as a baseline to isolate the contribution of uncertainty handling.

The proposed hybrid MPC-RL controller uses the learned value function to approximate the terminal cost as derived in Section 3.4, and is evaluated across horizons $N \in \{ 1 , . . . , 5 \}$ . Pure MPC uses the same DP formulation with zero terminal cost $( V _ { f } = 0 )$ , isolating the efect of lookahead length alone. Rollout replaces the terminal cost with a rollout estimate from Section 3.5, evaluated at two rollout horizons $N _ { r } \in \{ 1 0 , 2 0 \}$ } to study how rollout length afects terminal cost quality. Pure RL applies the trained DQN policy directly with no MPC optimization, providing a reference for what the learned value function achieves without constraint handling or recedinghorizon replanning. Deterministic variants of pure MPC, rollout $( N _ { r } = 1 0$ and 20), and hybrid MPC-RL apply noise-free conditions with full knowledge of the states of surrounding vehicles.

Table 1 reports collision rate, cost per step, cost standard deviation across all episodes (� ), cost standard deviation across collision-free episodes only (�<sub>0</sub>), mean speed, and computation time per step, together with the full cost decomposition. Table 2 compares selected configurations against their deterministic counterparts.

## 4.4 Discussion

4.4.1 Efect ofthe terminal cost. At a one-step horizon, pure MPC produces collisions in 86% of the episodes. With a single optimization step, the controller minimizes only the immediate stage cost with no visibility of future states. When the ego truck approaches a slow leader, the speed-tracking term encourages maintaining $v _ { \mathrm { r e f } } .$ , but because the front safety cost grows quadratically with gap shortfall, the DP evaluates deceleration as expensive once the gap is already small. The only apparently cheaper option is a lane change, but the target lane may have its own slow leader, a situation the one-step DP cannot see. The cost decomposition confirms this. At $N = 1$ the safety cost dominates at over 130,000 and the collision cost reaches over 100,000, while the speed cost is zero because the controller never deviates from $v _ { \mathrm { r e f } }$ before colliding. Adding one step of lookahead drops the collision rate to zero, since even 0.4 s of preview is suficient for the DP to evaluate the consequence of not decelerating.

The hybrid controller at � = 1 eliminates this failure mode en tirely, achieving zero collisions. The learned terminal cost provides a long-term value signal that recognizes proximity to a slow leader as a high-cost state even when the immediate stage cost appears moderate, compensating for the absence of explicit future trajectory optimization. The hybrid at $N = 3$ achieves the best overall result (329 cost, 0% collisions), outperforming pure MPC at � = 5 (754 cost) while using a substantially shorter horizon. The cost decompo sition shows that the safety cost at hybrid � = 3 (309) is 59% lower than pure MPC $N = 5 ( 7 4 7 )$ , meaning the hybrid maintains larger safety margins on average, not just fewer collisions. This confirms the central claim of this work, namely that a learned terminal cost approximation achieves better driving quality at shorter, cheaper horizons than a zero terminal cost at longer ones. Acceleration and cooldown costs are negligible across all controllers and do not drive any of the observed diferences.

The horizon also interacts with how efectively the controller handles prediction uncertainty. $\mathrm { A t } N = 2 ,$ , the wider safety distances derived from the growing prediction uncertainty add conservatism that increases cost relative to shorter-horizon pure MPC, because the RL terminal value does not internally account for the tightened safety bufers it was not exposed to during training. At $N = 3 ,$ the additional headroom from prediction-uncertainty-aware safety distances corrects a failure mode that occurs at this horizon without significantly penalizing the already-conservative terminal value signal, yielding the best overall result. This suggests that a moderate horizon and prediction uncertainty-aware safety distances are mutually reinforcing, where the horizon provides enough look ahead for the safety distances to be absorbed without sacrificing eficiency, while the safety distances correct edge cases the terminal cost alone misses.

4.4.2 Rollout horizon length and base policy quality. The rollout controller with $N _ { r } = 1 0$ partially improves short-horizon safety. The collision rate at $N = 1$ drops from 86% to 39%, confirming that the rollout estimate introduces genuine long-term awareness. However, it does not eliminate collisions at any horizon, and the total cost remains up to 44 times higher than the hybrid at the same �. The dominant cost terms are safety and collision, indicating that the rollout estimate is systematically overoptimistic. The base policy simulates a future in which the ego keeps a safe gap, but the actual closed-loop trajectory plays out diferently because surrounding trafic evolves in ways the Krauss-based, no-lane-change rollout does not capture. The uncertainty margin partially compensate for this at $N = 2$ by suppressing overoptimistic lane-change trajectories, which is why $N _ { r } = 1 0$ at $N = 2$ achieves zero collisions.

Increasing the rollout horizon to $N _ { r } = 2 0$ consistently degrades performance. Collision rates are several times higher than for $N _ { r } ~ = ~ 1 0$ at every horizon beyond $N \ = \ 1$ , and cost per step is substantially larger. The discount factor reduces the weight of distant rollout steps, but the accumulated prediction mismatch over 20 steps outweighs this correction. The DP is guided toward trajectories that appear favorable under a long rollout simulation but prove unsafe in practice. This demonstrates that the quality of a rollout terminal cost is bounded by the accuracy of the base policy, not by the length of the simulation, and that extending the rollout horizon is counterproductive when the base policy is imperfect.

4.4.3 Pure RL as a direct policy. Pure RL achieves zero collisions and a total cost of 551, competitive with pure MPC at $N = 3 .$ . The cost decomposition reveals a qualitatively diferent structure from the hybrid. The speed cost (18.0) is more than three times higher than for pure MPC or hybrid (≈ 5 and ≈ 15 respectively), and the acceleration cost (2.4) is the highest of any safe controller, reflecting a jerky longitudinal profile compared to the DP-optimized trajectories. These observations are consistent with a known tendency of value-based policies. The sparse, large collision penalty dominates training, producing a policy that avoids collisions conservatively through large gaps and low speed rather than by optimizing the full multi-objective cost. This conservatism makes pure RL a poor stand-alone controller for a truck that must maintain trafic flow, but it is precisely what makes the learned value function efective as a terminal cost. It provides a cautious, risk-averse signal that discourages the hybrid DP from ending the horizon in states with insuficient safety margins.

4.4.4 Comparison with deterministic baseline. Table 2 compares selected configurations against their deterministic counterparts. The efect is not uniform. For configurations that already produce collisions in the deterministic setting, accounting for prediction uncertainty consistently reduces or eliminates collisions while also reducing cost, because the collision has big penalty and the wider safety distances suppress the overoptimistic trajectories that caused the deterministic failures. For configurations that are already collision-free deterministically, accounting for uncertainty sometimes increases cost due to the added conservatism of the wider safety distances. For example, the cost for hybrid controller at $N = 2$ increases by 128% (257 → 585). Three configurations improve on both axes, becoming collision-free while also reducing cost: pure MPC at $N = 4 ,$ rollout $N _ { r } { = } 1 0$ at � = 2, and the hybrid at � = 3. Of these, the hybrid at $N = 3$ shows by far the largest improvement $( 1 , 0 1 5  3 2 9 )$ , making it the strongest configuration overall. This pattern confirms that the uncertainty-aware formulation is more conservative where predictions are uncertain and the deterministic controller would otherwise take risks.

Table 1: Stochastic controller performance evaluation and detailed cost decomposition.
<table><tr><td rowspan="2">Controller</td><td rowspan="2">N  $N _ { r }$ </td><td rowspan="2"></td><td colspan="6">Performance Results</td><td colspan="7">Detailed Cost Decomposition</td></tr><tr><td>Coll. %</td><td>Cost/step</td><td> $\sigma _ { \mathrm { a l l } }$ </td><td> $\sigma _ { 0 }$ </td><td>Mean Speed</td><td>Time (ms)</td><td> $\ell _ { \mathrm { s p e e d } }$ </td><td> $\ell _ { \mathrm { a c c e l } }$ </td><td> $\ell _ { \mathrm { s a f e t y } }$ </td><td> $\ell _ { \mathrm { c o l } }$ </td><td> $\ell _ { \mathrm { L C } }$ </td><td>lcD</td><td>fLL</td></tr><tr><td>Pure MPC</td><td>1</td><td></td><td>86%</td><td>234,519</td><td>233,924</td><td>17,447</td><td>25.0</td><td>2.5</td><td>0</td><td>0</td><td>130,826</td><td>103,692</td><td>0.0286</td><td>0.0067</td><td>0.9703</td></tr><tr><td>Pure MPC</td><td>2</td><td></td><td>0%</td><td>1,769</td><td>3,076</td><td>3,076</td><td>22.2</td><td>4.3</td><td>8.00</td><td>0.36</td><td>1,759</td><td>0</td><td>0.0149</td><td>0.0038</td><td>1.1031</td></tr><tr><td>Pure MPC</td><td>3</td><td>1</td><td>0%</td><td>830</td><td>2,022</td><td>2,022</td><td>22.6</td><td>6.3</td><td>5.82</td><td>0.51</td><td>823</td><td>0</td><td>0.0147</td><td>0.0030</td><td>1.0109</td></tr><tr><td>Pure MPC</td><td>4</td><td>一</td><td>0%</td><td>1,259</td><td>2,625</td><td>2,625</td><td>22.6</td><td>11.4</td><td>5.69</td><td>0.58</td><td>1,252</td><td>0</td><td>0.0136</td><td>0.0030</td><td>0.9717</td></tr><tr><td>Pure MPC</td><td>5</td><td>1</td><td>0%</td><td>754</td><td>2,019</td><td>2,019</td><td>22.6</td><td>17.5</td><td>5.68</td><td>0.58</td><td>747</td><td>0</td><td>0.0133</td><td>0.0030</td><td>0.9629</td></tr><tr><td>Rollout</td><td>1</td><td>10</td><td>39%</td><td>65,159</td><td>74,003</td><td>4,740</td><td>22.6</td><td>9.1</td><td>5.56</td><td>0.60</td><td>3,606</td><td>61,546</td><td>0.0364</td><td>0.0284</td><td>0.5256</td></tr><tr><td>Rollout</td><td>2</td><td>10</td><td>0%</td><td>1,072</td><td>3,058</td><td>3,058</td><td>22.5</td><td>15.1</td><td>6.17</td><td>0.65</td><td>1,065</td><td>0</td><td>0.0198</td><td>0.0052</td><td>0.4532</td></tr><tr><td>Rollout</td><td>3</td><td>10</td><td>1%</td><td>3,261</td><td>10,832</td><td>6,023</td><td>22.6</td><td>25.6</td><td>5.86</td><td>0.62</td><td>1,915</td><td>1,338</td><td>0.0208</td><td>0.0048</td><td>0.4511</td></tr><tr><td>Rollout</td><td>4</td><td>10</td><td>2%</td><td>3,978</td><td>13,137</td><td>5,021</td><td>22.6</td><td>41.5</td><td>5.85</td><td>0.61</td><td>1,286</td><td>2,686</td><td>0.0207</td><td>0.0070</td><td>0.4553</td></tr><tr><td>Rollout</td><td>5</td><td>10</td><td>1%</td><td>2,499</td><td>9,352</td><td>2,314</td><td>22.6</td><td>55.8</td><td>5.68</td><td>0.60</td><td>1,154</td><td>1,338</td><td>0.0216</td><td>0.0070</td><td>0.4478</td></tr><tr><td>Rollout</td><td>1</td><td>20</td><td>43%</td><td>65,150</td><td>73,027</td><td>4,265</td><td>22.7</td><td>12.8</td><td>5.31</td><td>0.62</td><td>4,701</td><td>60,443</td><td>0.0442</td><td>0.0396</td><td>0.6592</td></tr><tr><td>Rollout</td><td>2</td><td>20</td><td>3%</td><td>5,851</td><td>15,892</td><td>3,204</td><td>22.5</td><td>23.1</td><td>6.05</td><td>0.63</td><td>1,793</td><td>4,050</td><td>0.0242</td><td>0.0124</td><td>0.4769</td></tr><tr><td>Rollout</td><td>3</td><td>20</td><td>4%</td><td>6,395</td><td>18,937</td><td>2,447</td><td>22.7</td><td>37.1</td><td>5.51</td><td>0.60</td><td>1,652</td><td>4,736</td><td>0.0234</td><td>0.0091</td><td>0.4495</td></tr><tr><td>Rollout</td><td>4</td><td>20</td><td>4%</td><td>5,129</td><td>18,845</td><td>3,021</td><td>22.6</td><td>59.3</td><td>5.65</td><td>0.60</td><td>1,739</td><td>3,384</td><td>0.0224</td><td>0.0085</td><td>0.4636</td></tr><tr><td>Rollout</td><td>5</td><td>20</td><td>2%</td><td>2,390</td><td>16,475</td><td>1,666</td><td>22.7</td><td>85.8</td><td>5.53</td><td>0.58</td><td>1,035</td><td>1,348</td><td>0.0239</td><td>0.0097</td><td>0.4646</td></tr><tr><td>Pure RL</td><td>一</td><td>1</td><td>0%</td><td>551</td><td>1,029</td><td>1,029</td><td>20.8</td><td>0.3</td><td>18.0</td><td>2.4</td><td>529</td><td>0</td><td>0.017</td><td>0.02</td><td>2.17</td></tr><tr><td>Hybrid</td><td>1</td><td>1</td><td>0%</td><td>1,475</td><td>1,692</td><td>1,692</td><td>21.1</td><td>29.1</td><td>15.25</td><td>2.55</td><td>1,454.34</td><td>0</td><td>0.0229</td><td>0.0184</td><td>3.0524</td></tr><tr><td>Hybrid</td><td>2</td><td></td><td>0%</td><td>585</td><td>1,380</td><td>1,380</td><td>21.1</td><td>52.8</td><td>15.53</td><td>1.97</td><td>563.73</td><td>0</td><td>0.0205</td><td>0.0126</td><td>3.4646</td></tr><tr><td>Hybrid</td><td>3</td><td></td><td>0%</td><td>329</td><td>1,488</td><td>1,488</td><td>21.1</td><td>91.6</td><td>14.90</td><td>1.80</td><td>309.01</td><td>0</td><td>0.0194</td><td>0.0112</td><td>3.4633</td></tr><tr><td>Hybrid</td><td>4</td><td>一</td><td>0%</td><td>979</td><td>2,947</td><td>2,947</td><td>21.2</td><td>161.4</td><td>14.70</td><td>1.73</td><td>958.83</td><td>0</td><td>0.0218</td><td>0.0146</td><td>3.3872</td></tr><tr><td>Hybrid</td><td>5</td><td></td><td>1%</td><td>1,872</td><td>8,333</td><td>2,173</td><td>21.2</td><td>176.0</td><td>14.85</td><td>1.67</td><td>1,184.33</td><td>668.11</td><td>0.0213</td><td>0.0114</td><td>3.2741</td></tr></table>

Table 2: Deterministic vs. stochastic comparison for representative configurations.
<table><tr><td>Controller</td><td>N</td><td>Det. Coll.</td><td>S. Coll.</td><td>Det. Cost</td><td>S. Cost</td></tr><tr><td>Pure MPC</td><td>4</td><td>1%</td><td>0%</td><td>1,590</td><td>1,259</td></tr><tr><td>Pure MPC</td><td>5</td><td>0%</td><td>0%</td><td>448</td><td>754</td></tr><tr><td>Rollout  $N _ { r } = 1 0$ </td><td>2</td><td>1%</td><td>0%</td><td>2,078</td><td>1,072</td></tr><tr><td>Rollout  $N _ { r } = 1 0$ </td><td>5</td><td>3%</td><td>1%</td><td>5,072</td><td>2,499</td></tr><tr><td>Hybrid</td><td>2</td><td>0%</td><td>0%</td><td>257</td><td>585</td></tr><tr><td>Hybrid</td><td>3</td><td>1%</td><td>0%</td><td>1,015</td><td>329</td></tr></table>

4.4.5 Computational cost. All controllers remain within the 200 ms simulation-time budget, with the hybrid at � = 5 the most expensive at 176.0 ms. The hybrid controller’s cost grows faster with horizon than pure MPC because each reachable terminal state requires a forward pass through the Q-network, which has two hidden layers of 256 units, in addition to the DP state-space expansion. The best-performing configuration, hybrid $N = 3 ,$ uses 91.6 ms per step, under half the budget, so the recommended operating point carries a substantial timing margin. Batching terminal-state network evaluations or reducing the network size would lower this further and provide headroom for longer horizons on more constrained hardware. It should be noted that the reported times are influenced by the workload of the shared computational cluster at the time of evaluation.

## 5 Conclusion

This paper presented a hybrid MPC-RL framework for autonomous highway driving ofa heavy-duty truck, in which a chance-constrained finite-horizon MPC solved over a discretized state-action space uses a DQN-learned approximation of the true cost-to-go as its terminal cost. The framework was evaluated against pure MPC, rolloutbased, and pure RL baselines in a trafic simulator with reactive surrounding vehicles.

The results show that a learned terminal cost resolves the short horizon failure of pure MPC, achieves better driving quality at shorter horizons than a zero terminal cost at longer ones, and remains within real-time computational budget. Extending the rollout horizon degrades rather than improves performance, showing that rollout quality is bounded by base policy accuracy rather than simulation length.

Limitations include a straight homogeneous road, a kinematic ego model with no continuous lateral control, and a trajectory prediction model that assumes constant acceleration and fixed lanes for surrounding vehicles, which becomes less accurate at longer horizons and misses reactive lane-change behavior. Future work can address these by extending to richer road geometries, incorporating more accurate and reactive prediction models, and batching network evaluations to reduce computational cost at longer horizons.

## Acknowledgments

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

## References

[1] Matteo Acquarone, Federico Miretti, Daniela Misul, and Luca Sassara. 2023. Cooperative Adaptive Cruise Control Based on Reinforcement Learning for Heavy-Duty BEVs. IEEE Access 11 (2023), 127145–127156. doi:10.1109/ACCESS. 2023.3331827

[2] Nicola Albarella, Dario Giuseppe Lui, Alberto Petrillo, and Stefania Santini. 2023. A hybrid deep reinforcement learning and optimal control architecture for autonomous highway driving. Energies 16, 8 (2023), 3490. doi:10.3390/en16083490

[3] Dimitri P. Bertsekas. 2012. Dynamic Programming and Optimal Control, Vol. II: Approximate Dynamic Programming (4th ed.). Athena Scientific.

[4] Dimitri P Bertsekas. 2024. Model predictive control and reinforcement learning: A unified framework based on dynamic programming. IFAC-PapersOnLine 58, 18 (2024), 363–383. doi:10.1016/j.ifacol.2024.09.056

[5] Bruno Brito, Achin Agarwal, and Javier Alonso-Mora. 2022. Learning Interaction-Aware Guidance for Trajectory Optimization in Dense Trafic Scenarios. IEEE Transactions on Intelligent Transportation Systems 23, 10 (2022), 18808–18821. doi:10.1109/TITS.2022.316093

[6] Erik Börve, Nikolce Murgovski, and Leo Laine. 2023. Interaction-Aware Trajectory Prediction and Planning in Dense Highway Trafic using Distributed Model Predictive Control. In 2023 62nd IEEE Conference on Decision and Control (CDC). 6124–6129. doi:10.1109/CDC49753.2023.10384178

[7] Zhenbin Chen, Jiaqin Lai, Peixin Li, Omar I Awad, and Yubing Zhu. 2024. Prediction Horizon-Varying Model Predictive Control (MPC) for Autonomous Vehicle Control. Electronics 13, 8 (2024), 1442. doi:10.3390/electronics13081442

[8] Jakob Erdmann. 2015. SUMO’s lane-changing model. (2015), 105–123. doi:0. 1007/978-3-319-15024-6\_7

[9] Prakhar Gupta, Jonathon M. Smereka, and Yunyi Jia. 2024. Reinforcement Learning Compensated Model Predictive Control for Of-road Driving on Unknown Deformable Terrain. arXiv preprint arXiv:2408.09253 (2024). doi:10.48550/arXiv. 2408.09253

[10] Carl-Johan Hoel, Krister Wolf, and Leo Laine. 2018. Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC). 2148–2155. doi:10.1109/ITSC.2018.8569568

[11] Carl-Johan Hoel, Krister Wolf, and Leo Laine. 2020. Tactical Decision-Making in Autonomous Driving by Reinforcement Learning with Uncertainty Estimation. In 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 1563–1569. doi:10.1109/ IV47402.2020.9304614

[12] Erlong Kang, Hong Qiao, Ziyu Chen, and Jie Gao. 2022. Tracking of Uncertain Robotic Manipulators Using Event-Triggered Model Predictive Control With Learning Terminal Cost. IEEE Transactions on Automation Science and Engineering 19, 4 (2022), 2801–2815. doi:10.1109/TASE.2022.3152166

[13] Stefan Krauß. 1998. Microscopic modeling of trafic flow: Investigation of collision free vehicle dynamics. Ph. D. Dissertation.

[14] Alex Leslie and Dan Murray. 2025. An Analysis of the Operational Costs of Trucking: 2025 Update. Technical Report. American Transportation Research Institute (ATRI).

[15] Min Lin, Zhongqi Sun, Yuanqing Xia, and Jinhui Zhang. 2024. Reinforcement Learning-Based Model Predictive Control for Discrete-Time Systems. IEEE Transactions on Neural Networks and Learning Systems 35, 3 (2024), 3312–3324. doi:10.1109/TNNLS.2023.3273590

[16] Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Flötteröd, Robert Hilbrich, Leonhard Lücken, Johannes Rummel, Peter Wagner, and Evamarie Wiessner. 2018. Microscopic Trafic Simulation using

SUMO. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC). 2575–2582. doi:10.1109/ITSC.2018.8569938

[17] Samuel Mallick, Gianpietro Battocletti, Qizhang Dong, Azita Dabiri, and Bart De Schutter. 2025. Learning-Based MPC for Fuel Eficient Control ofAutonomous Vehicles With Discrete Gear Selection. IEEE Control Systems Letters 9 (2025), 1117–1122. doi:10.1109/LCSYS.2025.3575335

[18] David Q Mayne, James B Rawlings, Christopher V Rao, and Pierre OM Scokaert. 2000. Constrained model predictive control: Stability and optimality. Automatica 36, 6 (2000), 789–814. doi:10.1016/S0005-1098(99)00214-9

[19] Ali Mesbah. 2016. Stochastic Model Predictive Control: An Overview and Perspectives for Future Research. IEEE Control Systems Magazine 36, 6 (2016), 30–44. doi:10.1109/MCS.2016.2602087

[20] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. 2015. Human-level control through deep reinforcement learning. nature 518, 7540 (2015), 529–533. doi:10.1038/nature14236

[21] Francisco Moreno-Mora, Lukas Beckenbach, and Stefan Streif. 2023. Predictive Control with Learning-Based Terminal Costs Using Approximate Value Iteration. IFAC-PapersOnLine 56, 2 (2023), 3874–3879. 22nd IFAC World Congress. doi:10. 1016/j.ifacol.2023.10.1320

[22] Mitsuki Morita, Satoshi Yamamori, Satoshi Yagi, Norikazu Sugimoto, and Jun Morimoto. 2024. Goal-Conditioned Terminal Value Estimation for Real-time and Multi-task Model Predictive Control. arXiv preprint arXiv:2410.04929 (2024). doi:10.48550/arXiv.2410.04929

[23] Deepthi Pathare, Leo Laine, and Morteza Haghir Chehreghani. 2023. Improved Tactical Decision Making and Control Architecture for Autonomous Truck in SUMO Using Reinforcement Learning. In 2023 IEEE International Conference on Big Data (BigData). 5321–5329. doi:10.1109/BigData59044.2023.10386803

[24] Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernes tus, and Noah Dormann. 2021. Stable-Baselines3: Reliable Reinforcement Learn ing Implementations. Journal ofMachine Learning Research 22, 268 (2021), 1–8.

[25] James B. Rawlings, David Q. Mayne, and Moritz M. Diehl. 2024. Model predictive control : theory, computation, and design (2nd ed.). Nob Hill Pub.

[26] Rudolf Reiter, Jasper Hofmann, Joschka Boedecker, and Moritz Diehl. 2023. A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing. In 2023 European Control Conference (ECC). 1–8. doi:10.23919/ECC57647.2023. 10178143

[27] Rudolf Reiter, Jasper Hofmann, Dirk Reinhardt, Florian Messerer, Katrin Baumgärtner, Shambhuraj Sawant, Joschka Bödecker, Moritz Diehl, and Sebastien Gros. 2026. Synthesis of model predictive control and reinforcement learning: Survey and classification. Annual Reviews in Control 61 (2026), 101045. doi:10.1016/j.arcontrol.2026.101045

[28] Santokh Singh. 2015. Critical reasons for crashes investigated in the national motor vehicle crash causation survey. Technical Report DOT HS 812 115. National Highway Trafic Safety Administration (NHTSA), Washington, DC.

[29] Richard S Sutton, Andrew G Barto, and Andrew Barto. 1998. Reinforcement learning: An introduction. Vol. 17. MIT press Cambridge. doi:10.1017/ S0263574799271172

[30] Elizabeth Turnbull. 2017. Hours of service of drivers: Harnessing autonomous technology for safer operations. Jurimetrics 58, 1 (2017), 105–125.

[31] Liwen Wang, Shuo Yang, Kang Yuan, Yanjun Huang, and Hong Chen. 2023. A Combined Reinforcement Learning and Model Predictive Control for Car-Following Maneuver of Autonomous Vehicles. Chinese Journal ofMechanical Engineering 36, 1 (2023), 80. doi:10.1186/s10033-023-00904-7

[32] Nan Xu, Xiaohan Li, Qiao Liu, and Di Zhao. 2021. An Overview of Eco-Driving Theory, Capability Evaluation, and Training Applications. Sensors 21, 19 (2021). doi:10.3390/s21196547

[33] Baha Zarrouki, Marios Spanakakis, and Johannes Betz. 2024. A Safe Reinforcement Learning driven Weights-varying Model Predictive Control for Au tonomous Vehicle Motion Control. In 2024 IEEE Intelligent Vehicles Symposium (IV). 1401–1408. doi:10.1109/IV55156.2024.10588747

## A Deterministic Controller Results

The deterministic evaluation of the controllers is obtained by suppressing measurement noise and setting � → 0.5 so that $\Phi ^ { - 1 } ( 1 -$ $\epsilon ) = 0$ and the tightened safe distance of Eq. (15) reduces to the nominal $d _ { \mathrm { s a f e } }$ . The controller then acts on the nominal predicted gaps with full knowledge of the surrounding vehicle states. All other settings, including the 100 evaluation episodes and the checkpoint used for the terminal cost, are identical to the stochastic evaluation of Section 4.3, so the two difer only in the treatment of uncertainty. Table 3 reports the aggregate results and Table 4 the corresponding cost decomposition. The cooldown penalties are negligible throughout, therefore are omitted from Table 4.

Table 3: Deterministic controller results.
<table><tr><td>Controller</td><td>N</td><td>Nr</td><td>Coll. %</td><td>Cost/step</td><td>Mean Speed</td><td>Time/step (ms)</td></tr><tr><td>Pure MPC</td><td>1</td><td>=</td><td>88%</td><td>241,871</td><td>25.0</td><td>2.5</td></tr><tr><td>Pure MPC</td><td>2</td><td></td><td>0%</td><td>1,084</td><td>22.1</td><td>4.3</td></tr><tr><td>Pure MPC</td><td>3</td><td></td><td>0%</td><td>647</td><td>22.6</td><td>6.3</td></tr><tr><td>Pure MPC</td><td>4</td><td>=</td><td>1%</td><td>1,590</td><td>22.6</td><td>11.4</td></tr><tr><td>Pure MPC</td><td>5</td><td>=</td><td>0%</td><td>448</td><td>22.7</td><td>17.5</td></tr><tr><td>Rollout</td><td>1</td><td>10</td><td>34%</td><td>53,562</td><td>22.5</td><td>7.8</td></tr><tr><td>Rollout</td><td>2</td><td>10</td><td>1%</td><td>2,078</td><td>22.6</td><td>14.4</td></tr><tr><td>Rollout</td><td>3</td><td>10</td><td>2%</td><td>4,079</td><td>22.6</td><td>21.8</td></tr><tr><td>Rollout</td><td>4</td><td>10</td><td>2%</td><td>5,203</td><td>22.6</td><td>37.4</td></tr><tr><td>Rollout</td><td>5</td><td>10</td><td>3%</td><td>5,072</td><td>22.6</td><td>75.1</td></tr><tr><td>Rollout</td><td>1</td><td>20</td><td>32%</td><td>49,333</td><td>22.8</td><td>11.4</td></tr><tr><td>Rollout</td><td>2</td><td>20</td><td>13%</td><td>12,735</td><td>22.6</td><td>23.1</td></tr><tr><td>Rollout</td><td>3</td><td>20</td><td>10%</td><td>8,946</td><td>22.7</td><td>33.1</td></tr><tr><td>Rollout</td><td>4</td><td>20</td><td>15%</td><td>12,363</td><td>22.6</td><td>59.8</td></tr><tr><td>Rollout</td><td>5</td><td>20</td><td>7%</td><td>8,942</td><td>22.8</td><td>122.0</td></tr><tr><td>Hybrid</td><td>1</td><td>一</td><td>0%</td><td>883</td><td>21.1</td><td>17.1</td></tr><tr><td>Hybrid</td><td>2</td><td>–</td><td>0%</td><td>257</td><td>21.1</td><td>49.8</td></tr><tr><td>Hybrid</td><td>3</td><td></td><td>1%</td><td>1,015</td><td>21.2</td><td>113.7</td></tr><tr><td>Hybrid</td><td>4</td><td></td><td>0%</td><td>300</td><td>21.2</td><td>171.5</td></tr><tr><td>Hybrid</td><td>5</td><td>=</td><td>2%</td><td>2,178</td><td>21.1</td><td>242.3</td></tr></table>

Table 4: Cost decomposition for each controller type (deterministic).
<table><tr><td>Controller</td><td>N</td><td> $N _ { r }$ </td><td> $\ell _ { \mathrm { s p e e d } }$ </td><td> $\ell _ { \mathrm { a c c e l } }$ </td><td> $\ell _ { \mathrm { s a f e t y } }$ </td><td> $\ell _ { \mathrm { c o l } }$ </td><td> $\ell _ { \mathrm { L C } }$ </td><td> $\ell _ { \mathrm { L L } }$ </td></tr><tr><td>Pure MPC</td><td>1</td><td></td><td>0.0</td><td>0.0</td><td>131,885</td><td>109,984</td><td>0.027</td><td>1.00</td></tr><tr><td>Pure MPC</td><td>2</td><td>一</td><td>8.2</td><td>0.1</td><td>1,075</td><td>0</td><td>0.013</td><td>0.98</td></tr><tr><td>Pure MPC</td><td>3</td><td>1</td><td>5.8</td><td>0.2</td><td>640</td><td>0</td><td>0.013</td><td>1.00</td></tr><tr><td>Pure MPC</td><td>4</td><td>一</td><td>5.8</td><td>0.2</td><td>916</td><td>667</td><td>0.013</td><td>0.96</td></tr><tr><td>Pure MPC</td><td>5</td><td>一</td><td>5.3</td><td>0.2</td><td>442</td><td>0</td><td>0.013</td><td>1.01</td></tr><tr><td>Rollout</td><td>1</td><td>10</td><td>6.4</td><td>0.2</td><td>2,500</td><td>51,052</td><td>0.023</td><td>0.49</td></tr><tr><td>Rollout</td><td>2</td><td>10</td><td>6.0</td><td>0.2</td><td>1,403</td><td>667</td><td>0.018</td><td>0.36</td></tr><tr><td>Rollout</td><td>3</td><td>10</td><td>5.7</td><td>0.2</td><td>2,069</td><td>1,335</td><td>0.019</td><td>0.41</td></tr><tr><td>Rollout</td><td>4</td><td>10</td><td>5.8</td><td>0.2</td><td>3,851</td><td>2,012</td><td>0.018</td><td>0.46</td></tr><tr><td>Rollout</td><td>5</td><td>10</td><td>5.9</td><td>0.3</td><td>3,033</td><td>2,030</td><td>0.019</td><td>0.46</td></tr><tr><td>Rollout</td><td>1</td><td>20</td><td>5.0</td><td>0.2</td><td>3,436</td><td>45,892</td><td>0.030</td><td>0.52</td></tr><tr><td>Rollout</td><td>2</td><td>20</td><td>5.8</td><td>0.2</td><td>2,068</td><td>10,660</td><td>0.022</td><td>0.45</td></tr><tr><td>Rollout</td><td>3</td><td>20</td><td>5.3</td><td>0.2</td><td>1,309</td><td>7,632</td><td>0.022</td><td>0.42</td></tr><tr><td>Rollout</td><td>4</td><td>20</td><td>5.6</td><td>0.2</td><td>1,700</td><td>10,656</td><td>0.022</td><td>0.44</td></tr><tr><td>Rollout</td><td>5</td><td>20</td><td>5.0</td><td>0.2</td><td>3,441</td><td>5,496</td><td>0.021</td><td>0.49</td></tr><tr><td>Hybrid</td><td>1</td><td>一</td><td>15.4</td><td>1.4</td><td>863</td><td>0</td><td>0.024</td><td>3.13</td></tr><tr><td>Hybrid</td><td>2</td><td>–</td><td>15.3</td><td>1.3</td><td>237</td><td>0</td><td>0.017</td><td>3.61</td></tr><tr><td>Hybrid</td><td>3</td><td>=</td><td>14.6</td><td>1.3</td><td>325</td><td>671</td><td>0.018</td><td>3.53</td></tr><tr><td>Hybrid</td><td>4</td><td></td><td>14.9</td><td>1.3</td><td>280</td><td>0</td><td>0.016</td><td>3.46</td></tr><tr><td>Hybrid</td><td>5</td><td></td><td>14.9</td><td>1.2</td><td>821</td><td>1,338</td><td>0.023</td><td>3.32</td></tr></table>

## B Trajectory Prediction Methods

Two strategies were considered for predicting the future states of surrounding vehicles over the MPC horizon.

Reactive Multi-Agent Prediction. The first strategy propagates surrounding vehicles forward using a re-implementation of SUMO’s internal car-following logic. At each prediction step �, every neighbor vehicle adjusts its speed based on its gap to the vehicle immediately ahead ofit, following the Krauss safe-speed model [13]. The ego vehicle is projected at constant speed and fixed lane to serve as a stable reference for those reactions, avoiding a circular dependency between the controller’s own plan and the neighbors predicted responses.

While this approach captures realistic following behavior in principle, it introduces a systematic error at the boundary of the sensor range. The lead vehicle in each lane, the frontmost vehicle visible to the ego, has no visible leader within the sensor range, so the car-following model treats it as having a clear road ahead and accelerates it toward its reference (maximum) speed. This artificial acceleration propagates backward through the platoon, producing growing position errors over longer horizons. Freezing the boundary leader at constant speed mitigates this to some extent, but compounding model-mismatch errors still accumulate at every step as small inaccuracies in each vehicle’s response to the ego’s projected state propagate forward through the chain.

Constant-Acceleration Extrapolation. The second strategy extrapolates each surrounding vehicle forward from its currently observed position $s _ { j , 0 } ,$ speed $v _ { j , 0 } ,$ , and acceleration ${ { a } _ { j , 0 } } ,$ holding the lane assignment fixed:

$$
\begin{array} { r } { s _ { j , i } = s _ { j , 0 } + i v _ { j , 0 } \Delta t + \frac { i ( i - 1 ) } { 2 } a _ { j , 0 } ( \Delta t ) ^ { 2 } } \end{array}\tag{19}
$$

$$
v _ { j , i } = \mathrm { c l i p } \big ( v _ { j , 0 } + i a _ { j , 0 } \Delta t , 0 , v _ { j , \mathrm { m a x } } \big )\tag{20}
$$

This model requires no knowledge of the neighbor vehicles’ carfollowing parameters and introduces no inter-vehicle dependencies, so boundary efects do not arise.

Comparison. Both methods were evaluated by comparing predicted positions against ground-truth SUMO positions at each horizon step $i = 1 , \ldots , 5 ,$ , across 100 seeds using the pure MPC controller. The mean position error (bias) and its standard deviation are reported in Table 5 for the longest horizon $( i = 5 )$ , which has the most error.

Table 5: Mean (�) and standard deviation (�) of position prediction error (m) for � = 5.
<table><tr><td></td><td colspan="2">Reactive</td><td colspan="2">Const. acc.</td></tr><tr><td>Direction</td><td>μ</td><td>σ</td><td>μ</td><td>σ</td></tr><tr><td>Front</td><td>0.031</td><td>0.222</td><td>-0.002</td><td>0.089</td></tr><tr><td>Front-left</td><td>0.079</td><td>0.295</td><td>0</td><td>0.112</td></tr><tr><td>Front-right</td><td>0.059</td><td>0.262</td><td>0.001</td><td>0.096</td></tr><tr><td>Back-left</td><td>0.243</td><td>0.436</td><td>0.005</td><td>0.133</td></tr><tr><td>Back-right</td><td>0.167</td><td>0.393</td><td>0</td><td>0.131</td></tr></table>

Constant-acceleration extrapolation achieves lower mean absolute error and standard deviation than the reactive model at all horizon steps. At � = 5 (1 s ahead), the standard deviation remains by the DP solver. Constant-acceleration extrapolation is therefore adopted as the prediction model throughout this work.

Table 6: Selected model of each training run for the final hyperparameter setting, evaluated on 500 seeds.
<table><tr><td>Run</td><td>Checkpoint</td><td>Collision %</td><td>Mean reward</td><td> $ { v _ { \mathrm { a v g } } }$  (m/s)</td></tr><tr><td>1</td><td>16.1M</td><td>0</td><td> $- 3 1 . 4 6 \pm 4 2 . 4 4$ </td><td>20.8</td></tr><tr><td>2</td><td>12.6M</td><td>0</td><td> $- 1 6 . 9 0 \pm 2 8 . 4 5$ </td><td>20.1</td></tr><tr><td>3</td><td>6.8M</td><td>0</td><td> $- 1 3 . 7 2 \pm 2 2 . 0 3$ </td><td>20.5</td></tr><tr><td>4</td><td>13.6M</td><td>0</td><td> $- 8 . 6 0 \pm 2 0 . 3 9$ </td><td>20.8</td></tr><tr><td>5</td><td>17.5M</td><td>0</td><td> $- 1 4 . 3 0 \pm 2 3 . 8 5$ </td><td>20.6</td></tr><tr><td>Across runs</td><td>一</td><td>0</td><td> $- 1 7 . 0 0 \pm 8 . 6 3$ </td><td>20.6</td></tr></table>

Table 7: MPC/DP solver, DQN network, and training parameters. Bold denotes selected value; parentheses list other values tested.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>MPC / DP solver</td><td></td></tr><tr><td>Prediction horizon N</td><td>1-5</td></tr><tr><td>Rollout horizon Nr</td><td>10,20</td></tr><tr><td>Discount factor γ</td><td>0.98</td></tr><tr><td>Velocity grid step  $V _ { \mathrm { s t e p } }$ </td><td>0.2 m/s</td></tr><tr><td>Acceleration levels</td><td>5</td></tr><tr><td>DQN network</td><td></td></tr><tr><td>Hidden layers</td><td>[256, 256]</td></tr><tr><td>Replay buffer size</td><td>400,000</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Target network update</td><td>10,000 steps</td></tr><tr><td>DQN training (tuned)</td><td></td></tr><tr><td>Discount factor γ</td><td>0.98 (0.90, 0.95, 0.995)</td></tr><tr><td>Speed weight  $w _ { v }$ </td><td> $^ { 1 } \left( 3 , 5 , 1 0 \right)$ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 } ( 1 0 ^ { - 4 } )$ </td></tr><tr><td>Sensor range dsensor</td><td>150 m (100 m)</td></tr><tr><td>Expl. rate εend</td><td>0.01 (0.001)</td></tr><tr><td>Checkpoint</td><td>16.1 M steps</td></tr></table>

Table 8: Simulation, vehicle, and MPC cost parameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Simulation</td><td></td></tr><tr><td>Time step ∆t</td><td>0.2 s</td></tr><tr><td>Episode length</td><td>300 s</td></tr><tr><td>Lanes  $N _ { \mathrm { l a n e s } }$ </td><td>3</td></tr><tr><td>Traffic density</td><td>30 veh/km</td></tr><tr><td>Surrounding speed range</td><td>[20, 30] m/s</td></tr><tr><td>Sensor range  $d _ { \mathrm { s e n s o r } }$ </td><td>150 m</td></tr><tr><td>Ego vehicle</td><td></td></tr><tr><td>Speed limits  $[ v _ { \operatorname* { m i n } } ^ { e } , v _ { \operatorname* { m a x } } ^ { e } ]$ </td><td>[0, 33.33] m/s</td></tr><tr><td>Reference speed vref</td><td>25 m/s</td></tr><tr><td>Acceleration  $[ \boldsymbol { a } _ { \mathrm { m i n } } , \boldsymbol { a } _ { \mathrm { m a x } } ]$ </td><td>[−2.0, 2.0] m/s²</td></tr><tr><td>Vehicle length</td><td>12m</td></tr><tr><td>Lane-change duration  $T _ { \mathrm { L C } }$ </td><td>2.0 s</td></tr><tr><td>MPC cost function</td><td></td></tr><tr><td>Speed weight  $w _ { v }$ </td><td>1</td></tr><tr><td>Acceleration weight  $w _ { a }$ </td><td>1</td></tr><tr><td>Safety weight  $w _ { \mathrm { s a f e } }$ </td><td>1000</td></tr><tr><td>Left-lane weight  $w _ { \mathrm { L L } }$ </td><td>2</td></tr><tr><td>Min gap D0</td><td>5m</td></tr><tr><td>Time headway  $T _ { \mathrm { h e a d } }$ </td><td>2.0 s</td></tr><tr><td>Lane-change penalty  $P _ { \mathrm { L C } }$ </td><td>10</td></tr><tr><td>Cooldown penalty  $P _ { \mathrm { C D } }$ </td><td>30</td></tr><tr><td>Collision penalty  $P _ { \mathrm { c o l } }$ </td><td>108</td></tr></table>

below 0.15 m, confirming that the current acceleration is the dominant predictor of near-term position at the short timescales used

## C Rollout Base Policy

The rollout base policy propagates the ego vehicle and surrounding vehicles forward for $N _ { r }$ steps using lightweight analytical models, avoiding invocation of the SUMO simulator at runtime.

Surrounding vehicles. Each surrounding vehicle � is propagated longitudinally using the Krauss safe-speed model [13], holding its lane fixed throughout the rollout. The safe speed for vehicle � following its leader is:

$$
v _ { _ { \mathrm { { s a f e } } } } ^ { j } = - b \tau + { \sqrt { ( b \tau ) ^ { 2 } + ( v _ { _ { \mathrm { { l e a d } } } } ^ { j } ) ^ { 2 } + 2 b g ^ { j } } }\tag{21}
$$

where � is the maximum deceleration, � is the reaction time, $v _ { \mathrm { l e a d } } ^ { j }$ is the leader’s speed, and $g ^ { j }$ is the gap to the leader. The vehicle then accelerates toward its desired speed subject to $\boldsymbol { v } _ { \mathrm { s a f e } } ^ { j }$

Ego vehicle. For the ego vehicle, the safe speed is computed using the speed-dependent safe distance:

$$
d _ { \mathrm { s a f e } } = D _ { 0 } + T _ { \mathrm { h e a d } } v ^ { e }\tag{22}
$$

where $D _ { 0 }$ is a minimum gap and $T _ { \mathrm { h e a d } }$ is a time headway, applied to the front vehicle at all times and to the rear vehicle on the target lane whenever a lane change is commanded. At each rollout step, the ego considers a lane change if three conditions are met simultaneously: the gap to the target lane’s leader exceeds $d _ { \mathrm { s a f e } } ;$ the gap to the target lane’s follower exceeds $d _ { \mathrm { s a f e } } ;$ and the achievable speed in the target lane is closer to $v _ { \mathrm { r e f } }$ than in the current lane. If conditions are met, the lane change is executed; otherwise the ego remains in its current lane. The ego state is then updated as:

$$
v _ { r + 1 } ^ { e } = \mathrm { c l i p } ( v _ { r } ^ { e } + a _ { r } ^ { e } \Delta t , ~ v _ { \mathrm { m i n } } ^ { e } , ~ v _ { \mathrm { m a x } } ^ { e } )
$$

$$
s _ { r + 1 } ^ { e } = s _ { r } ^ { e } + v _ { r + 1 } ^ { e } \Delta t\tag{23}
$$

(24)

where � indexes rollout steps and $a _ { r } ^ { e }$ is computed to reach the target safe speed. The accumulated discounted stage cost over the rollout is

$$
\hat { V } _ { f _ { r } } ( x _ { N } ^ { e } ) = \sum _ { r = 0 } ^ { N _ { r } - 1 } \gamma ^ { r } \ell ( x _ { N + r } ^ { e } , \pi _ { r } ( x _ { N + r } ^ { e } ) )\tag{25}
$$

where $\pi _ { r }$ is the rollout policy described above. Surrounding vehicle lane changes during the rollout horizon are not modeled, which makes the rollout optimistic in scenarios where a neighboring vehicle cuts in during the rollout extension.

## D RL Model and Simulation Hyperparameter Tuning

Table 6 lists the model selected from each of the five training runs of Section 4.2, evaluated on 500 seeds. The configuration used in Section 4.2 was reached through the sequence of experiments summarized in Table 7, and Tables 8 list the simulation and MPC cost parameters. Each setting was trained and screened on the 10 holdout seeds; only the final configuration was carried through the full 500-seed evaluation, so the outcomes below are qualitative and are not directly comparable to Table 6.

Training initially used a discount factor of $\gamma = 0 . 9 9 5$ , which produced policies that avoided collisions by decelerating to a near standstill. Lowering the discount factor to 0.95 and 0.90 removed the degenerate behavior but left training unstable, with no checkpoint surviving the 500-seed evaluation; $\gamma = 0 . 9 8$ was retained.

The learning rate was reduced from $1 0 ^ { - 4 }$ to $1 0 ^ { - 5 }$ , which noticeably reduced the amplitude of the oscillation in the holdout curve without eliminating it. The sensor range was increased from 100 m to 150 m, and the final exploration rate was set to $\varepsilon _ { \mathrm { e n d } } = 0 . 0 1 ;$ a lower value of 0.001 produced comparable holdout results but a higher proportion of degenerate checkpoints.

The speed weight $w _ { v }$ deserves particular attention. Values of 3, $^ { 5 , }$ and 10 were tested, and while all trained stably, they degrade the hybrid controller rather than the RL policy itself. Because the reward is defined as the negative MPC stage cost (Section $3 . 4 ) ,$ any $w _ { v }$ difering from the value used by the ${ \mathrm { M P C } }$ rescales the learned �-values relative to the stage costs accumulated over the horizon, so the terminal cost is no longer expressed in the same units as the objective it is inserted into. The speed weight in training must therefore match the MPC cost function, i.e. $w _ { v } = 1$ , independently of which value trains best in isolation.