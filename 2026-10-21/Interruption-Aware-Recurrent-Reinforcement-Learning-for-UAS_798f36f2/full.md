# Interruption-Aware Recurrent Reinforcement Learning for UAS Navigation Under Temporary Overrides

Christopher Conrad , Student Member, IEEE, Deepak Panda , Member, IEEE, Weisi Guo , Member, IEEE, and Antonios Tsourdos , Member, IEEE

Abstract—Safe autonomous navigation of unmanned aircraft systems (UASs) requires tactical deconfliction to address conflicts that strategic coordination cannot fully prevent. This task is challenging under partial observability, when sensor or traffic data are missing, delayed, or degraded. Geometry-based methods can offer separation assurances but face limitations in dense encounters. Recurrent reinforcement learning (RL) has therefore been proposed to learn navigation policies that integrate observationaction histories to optimise long-term safety-efficiency tradeoffs. However, operational safeguards may require a remote pilot, onboard detect-and-avoid (DAA) system, or runtime assurance (RTA) mechanism to temporarily override the learned policy. A research gap remains in understanding how these interruptions affect recurrent policies when the agent regains authority. Benchmark evidence shows that interruptions make task variables less decodable from the recurrent hidden state and increase post-interruption collision risk. The findings suggest that altered recurrent context prolongs post-handback degradation through an action-observation feedback loop, in which changed actions alter the flight path and subsequent recurrent updates. To address this vulnerability, we propose three interruptionaware architectures that condition recurrent processing on interruption context through separate feature streams, dedicated interruption memory, or gated environmental features. Across simulations, all three architectures preserve more decodable task information than a standard recurrent policy. Moreover, the strongest architecture reduces post-interruption collisions by 74% and interruption-induced mission failures by 69% relative to the recurrent baseline, without degrading interruption-free performance. These results show that safe interruption does not guarantee safe handback, and support interruption-aware recurrent design for resilient UAS autonomy.

Index Terms—Air transportation, artificial intelligence, recurrent reinforcement learning, policy interruption, UTM.

## I. INTRODUCTION

U <sup>NMANNED</sup> <sup>aircraft</sup> <sup>system</sup> <sup>(UAS)</sup> <sup>traffic</sup> <sup>management</sup> (UTM) frameworks support flight planning, strategic deconfliction and information exchange for UAS navigation [1], [2]. Nonetheless, the UAS operator or remote pilot remains responsible for flight safety, and strategic coordination cannot eliminate every conflict that develops during a flight. Aircraft may deviate from their submitted intents, surveillance information may be delayed or degraded, and non-cooperative aircraft or other hazards may be absent from networked information. Tactical deconfliction is therefore required to address emerging conflicts and complement strategic UTM services [3]–[6]. Moreover, additional safeguards are required when the tactical navigation algorithm encounters unreliable information or conditions outside its validated operating envelope. An onboard detect-and-avoid (DAA) system, runtime-assurance (RTA) mechanism or remote pilot may temporarily override nominal autonomy to resolve an emerging hazard [3]–[5], [7]–[9]. Additionally, interruptions may arise from commandand-control (C2) link disruptions or malicious cyberphysical interference [10], [11]. The reliability of the nominal algorithm after it regains authority, hereby referred to as handback, is therefore part of the safety problem.

Within this operational framework, tactical deconfliction has traditionally relied on geometry-based methods, including velocity-obstacle and potential-field algorithms [12], [13]. These approaches often leverage rule-, search- or optimisationbased planning to provide predictable behaviour under explicit assumptions about aircraft dynamics, traffic behaviour and environmental conditions [14], [15]. Advanced air mobility (AAM), however, will increase the volume, heterogeneity, and autonomy of aircraft in the airspace, creating denser and more uncertain traffic interactions [1], [16]. These conditions make the behaviour of interacting aircraft harder to model and increase the computational cost of repeatedly solving multiaircraft deconfliction problems online [14], [15].

Reinforcement learning (RL) offers a complementary approach by learning navigation policies through simulated conflict encounters [17]. It can optimise safety-efficiency tradeoffs while shifting much of the computational burden to offline training [18]–[21]. However, finite sensor coverage, noise, occlusion and degraded traffic information make tactical deconfliction partially observable [1], [3]. Recurrent RL addresses this limitation by maintaining a hidden state that summarises task-relevant history to inform current decisions [22]–[24]. Notably, Lambrechts et al. [24] show that recurrent memory encodes information correlated with beliefs over latent task variables, with more informative representations resulting in better task performance. Recurrent memory can therefore be interpreted as a learned, task-dependent information state.

Temporary overrides introduce a distinct challenge for recurrent policies. During an interruption, the policy’s proposed actions are not executed, yet its recurrent state evolves with observations generated by externally selected manoeuvres. This altered context can make task-relevant information less accessible or distort how carried information influences subsequent decisions. Consequently, an interruption that resolves an immediate hazard does not necessarily ensure safe handback.

## A. Related Work

Table I summarises existing research relevant to interrupting recurrent navigation policies. Transportation research examines transitions between automated systems and human decision-makers, focusing on improving override timing and operator readiness [25]–[27]. In UTM and other safety-critical autonomous systems, DAA and RTA research instead addresses better hazard detection, safety monitoring, and recovery behaviour [3]–[5], [7]–[9]. Within RL, research primarily addresses the safety of learned behaviour and responsiveness to interruptions. Safe RL incorporates risk measures or safety constraints during training, whereas shielded RL masks or replaces actions that violate safety thresholds [29]– [31]. Corrigibility targets agents that accept or assist external corrections without manipulating the correcting authority [33]–[35]. Specifically, safely interruptible RL addresses how interruptions affect learning, developing algorithms that avoid seeking out or preventing future interruptions [32].

Action-modification research provides the closest formal abstraction of the interruptions considered in this paper. Langlois and Everitt [37] introduce the modified-action Markov decision process (MAMDP), which distinguishes the action selected by a policy from that ultimately executed following external modifications. Their analysis shows that some agents ignore modifications while others adapt their behaviour to avoid modifications that reduce reward. Human-in-the-loop RL also uses human overrides as demonstrations or feedback to improve policies during training [36].

Complementing these studies, recurrent RL has been applied to UAS navigation under partial observability [23], [28], while representation analysis examines which latent task variables are decodable from recurrent hidden states [24], [39]. Influence-aware memory restricts recurrent inputs to variables relevant to unobserved environmental states, reducing interference from irrelevant features [38]. Recurrent independent mechanisms similarly organise memory into sparsely interacting modules to improve robustness to environmental changes [40]. These approaches inform the analysis of recurrent representations and the design of structured memory architectures.

## B. Research Gap and Contributions

Existing literature highlights why interruptions are needed, when they should be initiated, and how they affect policy learning. However, a safe interruption does not guarantee safe handback. A research gap remains in understanding how interruptions affect recurrent representations, how they influence post-handback performance, and how they should be accommodated by recurrent policies. We address this gap through the following contributions:

• Interruption of recurrent UAS navigation is formulated as a handback reliability problem and modelled as a partially observable MAMDP (PO-MAMDP).

• A paired evaluation framework is developed to characterise the impact of interruptions across different trigger conditions, supervisor behaviours and override durations.

• Interruptions are shown to reduce task-variable decodability from the recurrent hidden state, degrade agent performance and increase post-handback collision risk.

• Three interruption-aware recurrent architectures are proposed that demonstrate increased robustness to interruptions without degrading interruption-free performance.

## II. PROBLEM FORMULATION

## A. Operational UTM Context

This study focuses on history-dependent tactical navigation under partial observability, as summarised in Table II. A single operator uses a nominal navigation policy to guide one UAS towards its destination while avoiding surrounding traffic and respecting operational constraints. The policy combines onboard sensor data with traffic and constraint information supplied through UTM services to generate navigation commands. It may run onboard, as illustrated in Fig. 1, or in a commandand-control centre (C2C), with commands transmitted over the C2 link. In either configuration, an onboard DAA system, RTA mechanism or remote pilot may temporarily override the policy to resolve emerging threats [3], [4], [6], [9]. Harmful overrides may also arise from malicious interference [11].

TABLE I  
SUMMARY OF EXISTING LITERATURE RELEVANT TO INTERRUPTING RECURRENT NAVIGATION POLICIES.
<table><tr><td>Research area</td><td>Primary focus</td><td>Key references</td><td>Research Gap</td></tr><tr><td>UAS supervision and transport handovers</td><td>Focuses on interruption timing, override conditions, operator readiness, and situational awareness.</td><td>[3]–[5], [7]–[9], [25]–[27]</td><td>Establishes the operational basis for temporary supervisor overrides, but is primarily concerned with system-level safety and human performance.</td></tr><tr><td>Recurrent RL for UAS navigation</td><td>Uses recurrent policies for navigation and collision avoidance under partial observability.</td><td>[23], [28]</td><td>Establishes the relevance of recurrent policies to UAS navigation, but assumes that the nominal policy selects all executed actions.</td></tr><tr><td>Safe RL and shielded RL</td><td>Incorporates safety constraints into learning and prevents unsafe actions from being executed.</td><td>[29]-[31]</td><td>Learns safe policies and prevents unsafe action execution. Although shielding may replace policy actions, recurrent handback is not considered.</td></tr><tr><td>Corrigibility and safe interruptibility</td><td>Ensures that agents accept external correction without seeking, resisting, or manipulating interruptions.</td><td>[32]-[35]</td><td>Examines behavioural incentives and the policy learned when interruptions occur during training, but does not consider deployment-time interruptions.</td></tr><tr><td>Modified-action and human-in-the-loop RL</td><td>Learns when executed actions differ from policy-selected actions, and uses interruptions as corrective feedback.</td><td>[36], [37]</td><td>Directly represents action substitution, but focuses on its effect on learning rather than recurrent state degradation and post-interruption performance.</td></tr><tr><td>Recurrent states and modular memory</td><td>Measures information encoded in recurrent states, and structures memory to improve robustness.</td><td>[24], [38]–[40]</td><td>Provides diagnostic and architectural foundations, but does not consider the effect of interruptions.</td></tr></table>

![](images/e0512600442844deb96a2b9b413245c4ffa0825e9c96dc29301b25c202ee256e.jpg)  
Fig. 1. Operational context and problem overview for interrupting a recurrent UAS navigation policy.

TABLE II  
IMPACT OF INTERRUPTIONS ACROSS UAS NAVIGATION FUNCTIONS
<table><tr><td>Function and temporal scope</td><td>Representative approaches</td><td>Impact of interruptions and handback</td></tr><tr><td>Stabilisation (milliseconds, low-level control) Strategic coordination (pre-flight)</td><td>PID, model-based control and physics- informed RL [41] Scheduling, optimisation and RL/MARL [42]</td><td>Not relevant. Commands can be recomputed from aircraft state without traffic history. Not relevant. Plans can be revised without interrupting an active navigation policy.</td></tr><tr><td>Short-horizon collision avoidance (seconds, local manoeuvres)</td><td>Velocity obstacles, potential fields and feedforward RL/MARL [15]</td><td>Not relevant. A new manoeuvre can be computed from traffic estimates without recurrent context.</td></tr><tr><td>Long-horizon tactical navigation (minutes-hours, mission execution)</td><td>Recurrent and history-dependent RL [43]</td><td>Primary scope. Safe handback requires retained traffic and mission context, which is impacted by interruptions.</td></tr></table>

## B. Simulated UAS Deconfliction Task

In this work, interruption of a nominal recurrent UAS navigation policy is studied using a kinematic simulation designed to isolate the interaction between partial observability, recurrent memory and interruptions. Table III summarises how this environment relates to practical UTM realities.

a) World and ego dynamics: A single UAS must navigate to its destination within a bounded airspace volume, while avoiding multiple moving intruders. The ego aircraft travels at constant ground speed in a horizontal plane. At each timestep, the navigation policy proposes a normalised yawrate command $a _ { t } ~ \in ~ [ - 1 , 1 ]$ , but during an interruption the action proposed by the supervisor is executed instead. The ego dynamics are hence given by (1).

$$
\begin{array} { r l } & { \psi _ { t + 1 } = \psi _ { t } + \dot { \psi } _ { \operatorname* { m a x } } \bar { a } _ { t } \Delta t , } \\ & { } \\ & { \mathbf { p } _ { t + 1 } = \mathbf { p } _ { t } + v _ { a } \left[ \cos \psi _ { t + 1 } \right. \left. \sin \psi _ { t + 1 } \right] ^ { \top } \Delta t , } \end{array}\tag{1}
$$

where ${ { \bar { a } } _ { t } }$ is the executed action; $\dot { \psi } _ { \mathrm { m a x } }$ is the maximum yaw rate; and $\mathbf p _ { t } ~ = ~ [ x _ { t } , y _ { t } ] ^ { \top }$ , ψ<sub>t</sub> and $v _ { a }$ are the ego position, heading and ground speed, respectively.

Start and destination positions are randomised in each episode. Moreover, surrounding traffic includes both intruders whose initial trajectories intersect the nominal start-destination path and intruders sampled elsewhere in the operating region, producing a range of direct and indirect encounters. Intruder headings are perturbed stochastically, preventing future encounter geometry from being inferred from a fixed trajectory model. An episode succeeds when the ego UAS reaches the destination and terminates unsuccessfully upon collision or when the mission horizon elapses. The ego position is also clamped at the operating-region boundary, analogous to an operational-volume or geozone limit [6]. Exact supervisor behaviours, trigger conditions, override durations, intruder speeds, traffic densities, and sampling distributions are reported in Section IV.

OPERATIONAL INTERPRETATION OF SIMULATION ABSTRACTIONS  
TABLE III
<table><tr><td>Operational element</td><td>Simulation abstraction</td></tr><tr><td>Tactical UAS navigation</td><td>One ego UAS follows a start-destination mission while avoiding traffic and respecting operational boundaries, representing a vehicle- or operator-side</td></tr><tr><td>Incomplete traffic information</td><td>navigation policy. Sensor range is limited and a restricted track set is refreshed intermittently, held between updates with an age indicator, and stochastically blanked,</td></tr><tr><td>Temporary overrides</td><td>representing incomplete or delayed onboard and networked information. An external supervisor replaces the nominal yaw-rate command for a bounded interval, representing a</td></tr><tr><td>Handback</td><td>DAA, RTA, remote-pilot or harmful interruption. The recurrent policy resumes from the aircraft and memory states reached during an interruption,</td></tr><tr><td>Operational constraints and objectives</td><td>exposing the post-handback reliability problem. Boundary clamping approximates an operational-volume or geozone limit, and the reward defines safety and efficiency objectives for a finite learning task.</td></tr></table>

b) Observations and partial observability: The environmental observation contains the information listed in (2).

$$
o _ { t } = \underbrace { \big [ \underbrace { \rho _ { 1 : K , t } , \beta _ { 1 : K , t } , \tau _ { 1 : K , t } } _ { \mathrm { t r a f f i c ~ t r a c k s } } ; \underbrace { \mathbf { d } _ { t } ^ { \mathrm { b n d } } } _ { \mathrm { b o u n d a r i e s } } ; \underbrace { d _ { t } ^ { g } , \beta _ { t } ^ { g } } _ { \mathrm { d e s t i n a t i o n } } ; \underbrace { \dot { \psi } _ { t } , \bar { a } _ { t - 2 } } _ { \mathrm { e g o ~ m o t i o n } } \big ] } _ { \mathrm { t r a f f i c ~ t r a c k s } } ,\tag{2}
$$

where $K$ is the maximum number of intruders the ego can observe simultaneously within a maximum sensing range, each reporting range $\rho _ { i , t } ,$ relative bearing $\beta _ { i , t } ,$ , and measurement age $\tau _ { i , t } .$ . Additionally, $\mathbf { d } _ { t } ^ { \mathrm { { b n d } } }$ contains distances to the operatingregion boundaries; $d _ { t } ^ { g }$ and $\beta _ { t . } ^ { g }$ are the observed destination range and bearing; and $\psi _ { t } = \psi _ { \mathrm { m a x } } \bar { a } _ { t - 1 }$ and $\bar { a } _ { t - 2 }$ reflect past actions of the UAS.

Partial observability is further imposed through temporally degraded information. At the beginning of each episode, a common traffic-track refresh period is sampled. All K slots are refreshed together at this period and held between refreshes with an increasing age indicator. A separate dropout process may stochastically blank the track information, and the destination channel is also periodically unavailable. Section IV reports the exact degradation schedules used.

c) Reward: The agent is rewarded for reaching its destination, separating from other traffic, and avoiding unnecessarily abrupt or prolonged manoeuvres, according to (3).

$$
\begin{array} { r l } & { r _ { t } = k _ { p } ( D _ { t - 1 } ^ { g } - D _ { t } ^ { g } ) - k _ { c } [ d _ { \mathrm { s a f e } } - d _ { t } ^ { \mathrm { m i n } } ] _ { + } } \\ & { ~ - k _ { \Delta a } | \bar { a } _ { t } - \bar { a } _ { t - 1 } | - c _ { \mathrm { t i m e } } } \\ & { ~ + B _ { g } \mathbb { I } \{ \mathrm { g o a l } _ { t } \} - P _ { c } \mathbb { I } \{ \mathrm { c o l l i s i o n } _ { t } \} - P _ { b } \mathbb { I } \{ \mathrm { b n d } _ { t } \} , } \end{array}\tag{3}
$$

where $D _ { t } ^ { g }$ is the distance to the goal; $d _ { t } ^ { \operatorname* { m i n } }$ is the distance to the nearest intruder; $d _ { \mathrm { s a f e } }$ is a prescribed separation distance; and $\mathrm { { c } _ { t i m e } }$ is a per-step time penalty. The operator $[ x ] _ { + } = \operatorname* { m a x } ( 0 , x )$ activates the separation penalty only when $d _ { t } ^ { \operatorname* { m i n } } \ < \ d _ { \mathrm { s a f e } }$ , and $\mathbb { I } \{ \cdot \}$ is an indicator function, equal to one when the specified event occurs and zero otherwise. Additionally, $k _ { p } , k _ { c } , k _ { \Delta a } , B _ { g } , P _ { c } ,$ and $P _ { b }$ are configurable weights defined in Section IV to reflect different reward tradeoffs. All terms are evaluated from the simulator state and executed actions, even when the corresponding information is degraded or unavailable in the policy observation. This ensures that observation degradation does not alter the task objective.

## C. UAS Navigation as a POMDP

When the state can be fully observed, UAS navigation is modelled as an MDP $\mathcal { M } ^ { \mathrm { F O } } \stackrel { \cdot } { = } ( S , \mathcal { A } , \mathcal { T } , \mathcal { R } , \rho _ { 0 } , \gamma )$ , where $s$ is the state space, A is the action space, $\mathcal { T } ( s ^ { \prime } \mid s , a )$ is the state-transition kernel, $\mathcal { R } ( s , a , s ^ { \prime } )$ is the reward function, $\rho _ { 0 }$ is the initial-state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. In practice, however, some information is unavailable, delayed or degraded [1], [3]. Consequently, UAS navigation is modelled as a partially observable MDP (POMDP) $\mathcal { M } = ( S , A , T , \mathcal { R } , \Omega , \mathcal { O } , \rho _ { 0 } , \gamma )$ , which extends the MDP with observation space Ω and an observation kernel $\mathcal { O } ( o \mid s ^ { \prime } , a )$ that gives the probability of observing o when action a leads to state $s ^ { \prime } \ [ 4 4 ]$ . Since the policy does not observe $s _ { t }$ directly, it must leverage the observation-action history, which under nominal execution is given by (4).

$$
H _ { t } = ( o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , \dots , a _ { t - 1 } , o _ { t } ) .\tag{4}
$$

An exact POMDP solution may condition on the belief state $b _ { t } ( s ) = \operatorname* { P r } ( s _ { t } = s \mid H _ { t } )$ . However, analytically computing this belief is generally intractable in high-dimensional, continuous or unknown environments [44], [45]. A recurrent policy therefore compresses its available observation history into a learned hidden state according to (5).

$$
\pi _ { \phi , t } ( \cdot ) = \pi _ { \phi } ( \cdot \mid z _ { t } ) , \qquad z _ { t } = F _ { \phi } ( z _ { t - 1 } , o _ { t } ) ,\tag{5}
$$

where $z _ { t }$ denotes the recurrent hidden state and $\phi$ denotes the policy parameters. The nominal policy action is sampled as $a _ { t } \sim \pi _ { \phi , t } .$ . Consequently, for an episode of horizon $N ,$ , the policy objective is given by (6).

$$
\begin{array} { r l r } {  { J _ { \mathcal { M } } ( \phi ) = \mathbb { E } _ { \tau _ { n o m } \sim p \phi , \mathcal { M } } [ \sum _ { t = 0 } ^ { N - 1 } \gamma ^ { t } r _ { t } ] , } } \\ & { } & { \quad r _ { t } = \mathcal { R } ( s _ { t } , a _ { t } , s _ { t + 1 } ) , } \end{array}\tag{6}
$$

where $\tau _ { n o m }$ is the resulting nominal trajectory and $p _ { \phi , { \mathcal M } }$ is the trajectory distribution induced jointly by the policy and POMDP dynamics.

## D. Interrupted UAS Navigation as a PO-MAMDP

A standard MDP assumes that the actions proposed by the nominal policy are always executed. Langlois and Everitt [37] relax this assumption through the modified-action MDP (MAMDP), which augments an MDP with an action-selection function $P _ { A } \left( a \mid \pi , s \right)$ , that maps a policy and state to a distribution over the actions actually executed. They term the unmodified policy the virtual policy and the resulting distribution over executed actions the empirical policy.

We extend this notion by defining a PO-MAMDP in (7),

$$
{ \widetilde { \mathcal { M } } } = ( { \mathcal { M } } , P _ { A } ) ,\tag{7}
$$

with the resulting empirical policy given by (8).

$$
\dot { \pi } _ { \phi , t } ( \bar { a } \mid H _ { t } ) : = P _ { A } ( \bar { a } \mid \pi _ { \phi , t } , H _ { t } ) ,\tag{8}
$$

where a¯ denotes an executed action and $\dot { \pi } _ { \phi , t }$ is the empirical policy induced by action modification. The ordinary POMDP can be recovered when the virtual policy is applied without modification, according to (9).

$$
P _ { A } ^ { \mathrm { P O M D P } } ( \bar { a } \mid \pi _ { \phi , t } , H _ { t } ) = \pi _ { \phi , t } ( \bar { a } ) .\tag{9}
$$

For simplicity, we consider $H _ { t } \in { \mathcal { H } }$ to denote the complete joint interaction history available at t for both the nominal policy and supervisors. Nonetheless, a policy or function conditioned on $H _ { t }$ is understood to only use the information available to it. For example, an onboard DAA function may receive local surveillance measurements of hazards that are absent from the networked information used by a ground-based navigation policy. Conversely, a remote pilot may have access to operational intents, airspace constraints, or supplementary traffic information that is unavailable onboard [1], [3].

## E. Interruptions in the UAS Navigation PO-MAMDP

Orseau and Armstrong [32] formally define an interruption scheme by the triplet $\langle I , \theta , \pi ^ { \mathrm { I N T } } \rangle$ . The initiation function $I ( H _ { t } ) \in [ 0 , 1 ]$ assesses whether an interruption should begin after the current history, ${ \theta _ { t } } ~ \in ~ [ 0 , 1 ]$ scales the probability that the interruption policy is applied when initiated, and $\pi ^ { \mathrm { I N T } }$ specifies the actions selected during the interruption. The authors further introduce an interruption operator $I N T ^ { \theta }$ $\Pi \ $ Π that turns a policy $\pi$ into an interruptible policy $I N T ^ { \theta } \left( \pi \right)$ given by (10), where the agent follows $\pi ^ { I N T }$ when it is interrupted and π otherwise.

$$
\begin{array} { r l } & { \mathrm { I N T } ^ { \theta } ( \pi ) ( a \mid H _ { t } ) = \theta _ { t } I ( H _ { t } ) \pi ^ { \mathrm { I N T } } ( a \mid H _ { t } ) } \\ & { ~ + \left[ 1 - \theta _ { t } I ( H _ { t } ) \right] \pi ( a \mid H _ { t } ) . } \end{array}\tag{10}
$$

We extend the definition of an interruption to include a termination function $\zeta _ { k } ( H _ { t } , \ell ) \in [ 0 , 1 ]$ , giving the probability that supervisor k hands decision authority back to the nominal policy after ℓ active steps. Moreover, we assume that an applied interruption will always trigger successfully in deployment, corresponding to $\theta _ { t } = 1 . \ \theta _ { t }$ is hence omitted from subsequent equations for simplicity. With this definition, overrides can be defined as interruptions that modify nominal policy actions within the PO-MAMDP. Let $m _ { t } \in \{ 0 , 1 , \ldots , K _ { \mathrm { s u p } } \}$ identify the source of the executed action of the empirical policy, where $m _ { t } = 0$ denotes nominal execution and $m _ { t } = k > 0$ denotes supervisor k. When no interruption is active, the probability that supervisor k initiates an override is given by (11).

$$
\begin{array} { r l } & { \operatorname* { P r } ( m _ { t } = k \mid m _ { t - 1 } = 0 , H _ { t } ) } \\ & { \quad = I ( H _ { t } ) q _ { t } ( k \mid H _ { t } ) , \qquad k = 1 , \ldots , K _ { \mathrm { s u p } } , } \end{array}\tag{11}
$$

where $q _ { t } ( k \mid H _ { t } )$ is the conditional distribution used to select the supervisor and $K _ { s u p }$ is the number of distinct supervisors considered. Similarly, the probability of a supervisor handing back control to the nominal policy is given by (12).

$$
\operatorname* { P r } ( m _ { t + 1 } = 0 \mid m _ { t } = k , H _ { t } , \ell ) = \zeta _ { k } ( H _ { t } , \ell ) .\tag{12}
$$

A fixed-duration interruption is obtained by making handback deterministic after a sampled duration $d _ { j } ,$ , and a scheduled interruption is obtained by concentrating I at an onset $t _ { j } ^ { \mathrm { o n } }$ . In general, an interruption $j$ is therefore the interval (13),

$$
\begin{array} { r } { \mathcal { T } _ { j } = \{ t _ { j } ^ { \mathrm { o n } } , \ldots , t _ { j } ^ { \mathrm { H B } } - 1 \} , \qquad t _ { j } ^ { \mathrm { H B } } = t _ { j } ^ { \mathrm { o n } } + d _ { j } , } \end{array}\tag{13}
$$

during which $m _ { t } = k _ { j } > 0$ . Handback occurs at $t _ { j } ^ { \mathrm { H B } }$ , when the nominal policy again determines the executed action.

The source $m _ { t }$ forms part of the joint history $H _ { t }$ , and defining $P _ { A , t } ( \bar { a } ) \equiv P _ { A } ( \bar { a } \mid \pi _ { \phi , t } , H _ { t } )$ gives (14).

$$
P _ { A , t } ( \bar { a } ) = \left\{ \begin{array} { l l } { { \pi _ { \phi , t } ( \bar { a } ) , } } & { { m _ { t } = 0 , } } \\ { { \pi _ { k } ^ { \mathrm { I N T } } ( \bar { a } \mid H _ { t } ) , } } & { { m _ { t } = k > 0 . } } \end{array} \right.\tag{14}
$$

Equivalently, if $\begin{array} { r } { { a } _ { t } \sim \ \pi _ { \phi , } } \end{array}$ <sub>t</sub> is the nominal proposal and supervisor k supplies an explicit command $a _ { t } ^ { ( k ) }$ , the executed action is given by (15).

$$
\begin{array} { r } { \bar { a } _ { t } = \left\{ \begin{array} { l l } { a _ { t } , } & { m _ { t } = 0 , } \\ { a _ { t } ^ { ( k ) } , } & { m _ { t } = k > 0 . } \end{array} \right. } \end{array}\tag{15}
$$

## III. INTERRUPTION-AWARE RECURRENT RL

## A. LSTM-based PPO as a Recurrent Baseline

Several RL algorithms have been applied to UAS navigation under partial observability [23], [28]. Among policy-gradient methods, proximal policy optimisation (PPO) is particularly suitable because it enables several minibatch updates from each collected rollout and discourages excessively large policy changes through a clipped surrogate objective [46]. Unlike value-based methods such as deep Q-learning, PPO can represent continuous actions without discretising the action space [47]. It also directly represents a stochastic policy, unlike deep deterministic policy gradient (DDPG) methods, and avoids the complex constrained optimisation required by trust-region policy optimisation (TRPO) [48], [49]. A recurrent PPO agent therefore serves as the baseline policy for this work, to optimise a stochastic distribution over the continuous yaw-rate commands of the ego UAS described in Section II.

For each rollout, let $\phi _ { \mathrm { o l d } }$ denote the actor parameters used to collect the data. PPO measures the change in probability assigned to a sampled action using the likelihood ratio in (16).

$$
\begin{array} { l } { \displaystyle \eta _ { t } ( \phi ) = \frac { \pi _ { \phi } \bigl ( a _ { t } \mid z _ { t } ^ { \phi } \bigr ) } { \pi _ { \phi _ { \mathrm { o l d } } } \bigl ( a _ { t } \mid z _ { t } ^ { \mathrm { o l d } } \bigr ) } , } \\ { \displaystyle \bar { \eta } _ { t } ( \phi ) = \mathrm { c l i p } \bigl ( \eta _ { t } ( \phi ) , 1 - \epsilon , 1 + \epsilon \bigr ) , } \end{array}\tag{16}
$$

where $\eta _ { t }$ is the likelihood ratio, $\bar { \eta } _ { t }$ is its clipped value, ϵ is the clipping threshold, and $z _ { t } ^ { \phi }$ and $z _ { t } ^ { \mathrm { o l d } }$ are the recurrent states produced by the updated and rollout policies, respectively. Given an advantage estimate $\widehat { A } _ { t } .$ , commonly calculated using generalised advantage estimation (GAE) [50], the actor maximises (17).

$$
L _ { \pi } ^ { \mathrm { c l i p } } ( \phi ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( \eta _ { t } ( \phi ) \widehat { A } _ { t } , \bar { \eta } _ { t } ( \phi ) \widehat { A } _ { t } \right) \right] .\tag{17}
$$

In the actor-critic formulation, the actor $\pi _ { \phi }$ defines the action distribution, while the critic $V _ { \psi }$ estimates the expected discounted return from the current state. The resulting advantage estimate indicates whether an action performed better or worse than the critic predicted and therefore guides the actor update. The complete PPO loss additionally contains a value-prediction loss for the critic and an entropy term that discourages premature policy convergence [46].

Recurrent policies address partial observability by compressing the observation-action history into a learned information state that can encode task-relevant belief information [24], [38]. In this work, long short-term memory (LSTM) networks are used because their gated cell structure supports information retention over longer temporal dependencies than an ungated recurrent neural network [51]. Although gated recurrent units (GRUs) offer a simpler gated alternative, LSTMs maintain a distinct cell state that provides an explicit pathway for longerterm information retention [52]. This separation is particularly useful for analysing how task-relevant memory changes during an interruption.

For recurrent input $x _ { t } ,$ an LSTM updates its cell state $c _ { t }$ and hidden state $z _ { t }$ according to (18).

$$
\begin{array} { r l } & { { g _ { t } ^ { \mathrm { f } } } = \sigma ( { W _ { \mathrm { f } } } { x _ { t } } + { U _ { \mathrm { f } } } { z _ { t - 1 } } + { b _ { \mathrm { f } } } ) , } \\ & { { g _ { t } ^ { \mathrm { i } } } = \sigma ( { W _ { \mathrm { i } } } { x _ { t } } + { U _ { \mathrm { i } } } { z _ { t - 1 } } + { b _ { \mathrm { i } } } ) , } \\ & { { g _ { t } ^ { \mathrm { o } } } = \sigma ( { W _ { \mathrm { o } } } { x _ { t } } + { U _ { \mathrm { o } } } { z _ { t - 1 } } + { b _ { \mathrm { o } } } ) , } \\ & { \smash { \widetilde { c } _ { t } } = \operatorname { t a n h } ( { W _ { \mathrm { c } } } { x _ { t } } + { U _ { \mathrm { c } } } { z _ { t - 1 } } + { b _ { \mathrm { c } } } ) , } \\ & { \smash { c _ { t } } = { g _ { t } ^ { \mathrm { f } } } \odot \smash { c _ { t - 1 } } + { g _ { t } ^ { \mathrm { i } } } \odot \smash { \widetilde { c } _ { t } } , } \\ & { \smash { z _ { t } } = { g _ { t } ^ { \mathrm { o } } } \odot \operatorname { t a n h } ( \smash { c _ { t } } ) , } \end{array}\tag{18}
$$

where $\sigma$ denotes the logistic sigmoid; ⊙ denotes elementwise multiplication; $g _ { t } ^ { \mathrm { f } } , \bar { g _ { t } ^ { \mathrm { i } } }$ , and $g _ { t } ^ { \mathrm { o } }$ denote the forget-, input-, and output-gate values, respectively; and W, U, and b denote learned weights and biases.

The recurrent baseline is therefore an LSTM-based PPO agent with separate actor and critic recurrent states. Nonetheless, the deployment-time failure mechanism should apply more generally to interrupted history-dependent policies. Similarly, the proposed interruption-aware architectures can be adapted to other recurrent networks, although the masked actor objective introduced later in this section is formulated specifically for PPO.

## B. Interruption-Aware Observations

The environmental observation in (2) contains past executed actions but does not identify whether those actions were selected by the nominal policy. We therefore augment this observation by the input $u _ { t }$ in (19).

$$
u _ { t } = \left[ a _ { t - 1 } , \lambda _ { t } \right] , \qquad \lambda _ { t } = \mathbb { I } \{ m _ { t - 1 } > 0 \} ,\tag{19}
$$

where $a _ { t - 1 }$ is the action proposed by the nominal policy and $\mathbb { I } ( \cdot )$ is the indicator function. $\lambda _ { t }$ is a binary flag that indicates whether the preceding action was replaced by a supervisor. This reflects the operational reality that navigation policy may not know which supervisor intervened or why, but the vehicle system can indicate that its nominal proposal was replaced by an externally selected action. All proposed architectures receive both $o _ { t }$ and $u _ { t }$ , but differ in how these input streams are encoded, combined, and incorporated into recurrent memory.

## C. Interruption-Aware Recurrent Architectures

An interruption-aware policy receives two qualitatively different input streams. The environmental observation $o _ { t }$ describes the aircraft, surrounding traffic, destination, and environment. Although its measurements may be intermittent or degraded, the underlying physical state generally evolves continuously over time. Conversely, the interruption input $u _ { t }$ changes abruptly at an interruption trigger and upon handback. The recurrent agent must therefore construct a task-relevant belief for the evolving physical environment while also representing sparse changes in decision authority.

Three interruption-aware recurrent architectures are proposed to explicitly encode this distinction, as summarised in Fig. 2. All three process the environmental and interruption inputs through separate learned encoders according to (20).

$$
e _ { o , t } = f _ { e } ( o _ { t } ) \in \mathbb { R } ^ { d _ { o } } , \qquad e _ { u , t } = f _ { u } ( u _ { t } ) \in \mathbb { R } ^ { d _ { u } } ,\tag{20}
$$

where $f _ { e }$ and $f _ { u }$ are the environmental and interruption branches of a feature extractor shared by the actor and critic. The resulting features $e _ { o , t }$ and $e _ { u , t }$ provide separate representations of the physical environment and interruption context. The architectures differ in where and how these representations are combined within the actor. To keep value estimation consistent, the critic always retains a single LSTM.

Algorithm 1 summarises one actor-critic forward pass for each architecture. Each recurrent state is written as $s _ { t } =$ $\left( { { c _ { t } } , { z _ { t } } } \right)$ , where $z _ { t }$ is its hidden-state component. The variable y<sub>t</sub> denotes the representation supplied to the policy head, while $x _ { t } ^ { V }$ denotes the input supplied to the critic. The critic hidden state $z _ { t } ^ { V }$ is processed by the post-recurrent layer $g _ { V }$ and the scalar value head $V _ { \psi }$ . Its output $\widehat { v } _ { t } .$ , estimates the expected discounted return from the current interaction history under the learned policy and interruption process.

1) Dual-feature architecture: The dual-feature architecture provides the least restrictive separation between environmental and interruption information. The two inputs are encoded independently, concatenated, and supplied to a single actor LSTM. Let $\mathbf { \bar { \mathbf { \Phi } } } _ { s _ { t } ^ { \mathrm { D F } } } ~ = ~ ( c _ { t } ^ { \mathrm { D F } } , z _ { t } ^ { \mathrm { D F } } )$ denote its cell and hidden states. The actor is then defined by (21).

$$
\begin{array} { r l } & { \quad x _ { t } ^ { \mathrm { D F } } = [ e _ { o , t } ; e _ { u , t } ] , } \\ & { \quad s _ { t } ^ { \mathrm { D F } } = \mathrm { L S T M } _ { \mathrm { D F } } ( x _ { t } ^ { \mathrm { D F } } , s _ { t - 1 } ^ { \mathrm { D F } } ) , } \\ & { \quad \pi _ { \phi , t } ^ { \mathrm { D F } } ( \cdot ) = \pi _ { \phi } ^ { \mathrm { D F } } \bigl ( \cdot \mid g _ { \mathrm { D F } } ( z _ { t } ^ { \mathrm { D F } } ) \bigr ) , } \end{array}\tag{21}
$$

where, $x _ { t } ^ { \mathrm { D F } }$ is the concatenated feature vector, $g _ { \mathrm { D F } }$ is the post-LSTM feed-forward layer, and $\pi _ { \phi , t } ^ { \mathrm { D F } }$ is the resulting action distribution. Separate feature extraction makes interruption status explicit without prescribing how it must be stored. Consequently, both feature streams update the same recurrent hidden state, allowing environmental and interruption information to interact, and potentially interfere, within the shared memory. The critic receives the same feature vector but processes it using its own LSTM and value head.

2) Dual-memory architecture: The dual-memory architecture imposes a stronger separation by assigning the environmental and interruption features to independent actor LSTMs. Let $s _ { o , t } ^ { \mathrm { D M } } = ( c _ { o , t } ^ { \mathrm { D M } } , \hat { z } _ { o , t } ^ { \mathrm { D M } } )$ and $s _ { u , t } ^ { \mathrm { D M } } = ( c _ { u , t } ^ { \mathrm { D M } } , z _ { u , t } ^ { \mathrm { D M } } )$ denote their respective recurrent states. The actor is defined by (22).

$$
\begin{array} { r l } & { s _ { o , t } ^ { \mathrm { D M } } = \mathrm { L S T M } _ { o } ( e _ { o , t } , s _ { o , t - 1 } ^ { \mathrm { D M } } ) , } \\ & { s _ { u , t } ^ { \mathrm { D M } } = \mathrm { L S T M } _ { u } ( e _ { u , t } , s _ { u , t - 1 } ^ { \mathrm { D M } } ) , } \\ & { y _ { t } ^ { \mathrm { D M } } = g _ { \mathrm { D M } } ( [ z _ { o , t } ^ { \mathrm { D M } } ; z _ { u , t } ^ { \mathrm { D M } } ] ) , } \\ & { \pi _ { \phi , t } ^ { \mathrm { D M } } ( \cdot ) = \pi _ { \phi } ^ { \mathrm { D M } } ( \cdot  { \vert } \ y _ { t } ^ { \mathrm { D M } } ) , } \end{array}\tag{22}
$$

where, the environmental LSTM maintains a historydependent representation of the physical encounter, while the interruption LSTM tracks the nominal proposal and interruption context. Their hidden states are combined only after recurrence by the learned fusion layer $g _ { \mathrm { D M } }$ , producing the policy feature $y _ { t } ^ { \mathrm { D M } }$ . This separation prevents the interruption features from directly updating the environmental recurrent state, although $o _ { t }$ may still contain observations generated by supervisor-selected manoeuvres. The critic receives the concatenated encoder output $[ e _ { o , t } ; e _ { u , t } ]$ but processes it using its own single LSTM and value head.

3) Gated architecture: The gated architecture combines the two input streams through feature-wise linear modulation (FiLM) [53]. Rather than supplying the interruption features directly to the recurrent layer, FiLM uses them to scale and shift each environmental feature according to (23).

$$
\begin{array} { r } { \begin{array} { c } { \alpha _ { t } = f _ { \alpha } ( e _ { u , t } ) , } \\ { \delta _ { t } = f _ { \delta } ( e _ { u , t } ) , } \\ { \widetilde { e } _ { o , t } = \alpha _ { t } \odot e _ { o , t } + \delta _ { t } , } \end{array} } \end{array}\tag{23}
$$

where $f _ { \alpha }$ and $f _ { \delta }$ are learned mappings from the interruption features to feature-wise scale and shift vectors, respectively; $\odot$ denotes element-wise multiplication; and $\widetilde { e } _ { o , t }$ is the modulated environmental representation. Let $s _ { t } ^ { \mathrm { G } } = ( c _ { t } ^ { \mathrm { G } } , z _ { t } ^ { \mathrm { G } } )$ denote the state of the actor LSTM. The actor is then defined by (24).

$$
\begin{array} { r } { s _ { t } ^ { \mathrm { G } } = \mathrm { L S T M } _ { \mathrm { G } } ( \widetilde { e } _ { o , t } , s _ { t - 1 } ^ { \mathrm { G } } ) , } \\ { \pi _ { \phi , t } ^ { \mathrm { G } } ( \cdot ) = \pi _ { \phi } ^ { \mathrm { G } } \left( \cdot \mid g _ { \mathrm { G } } ( z _ { t } ^ { \mathrm { G } } ) \right) . \qquad } \end{array}\tag{24}
$$

This architecture allows the interruption context to determine which environmental features are emphasised, attenuated, or shifted before they update recurrent memory. It therefore provides stronger interaction between the two input streams than dual memory, while avoiding their direct concatenation within a shared recurrent update. The critic receives the same modulated environmental representation $\widetilde { e } _ { o , t }$ but processes it using its own LSTM and value head.

## D. Interruption-Aware RL Training

The proposed architectures must experience interruptions during training to learn how the interruption input changes at takeover and handback, and how recurrent memory should evolve between these events. Training therefore introduces randomised interruptions and implements a masked PPO actor update, as summarised in Algorithm 2. For a fair comparison, the recurrent baseline is trained under the same conditions.

1) Randomised interruption training: The probability that an episode contains an interruption linearly increases throughout training, enabling the agent to first learn nominal deconfliction behaviour before progressively introducing interruptions. Each selected training episode contains a single bounded interruption whose onset, duration, and supervisor behaviour are sampled from predefined distributions. The specific training schedule and sampling distributions are specified in Section IV.

2) Masked PPO actor updates: During an interruption, the nominal actor continues to propose $a _ { t }$ and update its recurrent state, but the supervisor-selected action ${ { \bar { a } } _ { t } }$ is executed. The resulting transition and reward cannot be attributed directly to $a _ { t }$ and assigning its PPO advantage to that proposal would train the actor using an action it did not execute. The actor advantage is hence masked after GAE in (25).

$$
\widetilde { A } _ { t } = \mathbb { I } ( m _ { t } = 0 ) \widehat { A } _ { t } .\tag{25}
$$

The masked advantage $\widetilde { A } _ { t }$ is then used in the standard clipped PPO objective, so an interrupted step contributes no direct advantage-weighted policy gradient.

The critic and entropy terms remain unmasked. The critic is trained against returns from the executed-action trajectory, while the entropy term continues to regularise the policy at all visited histories. Actor and critic recurrent states are also unrolled through supervisor-executed steps. Masking therefore prevents direct credit assignment to unexecuted proposals without discarding the interrupted trajectories required to train recurrent memory and value estimation.

Algorithm 1 Forward pass of interruption-aware architectures   
Require: o<sub>t</sub>, a<sub>t−1</sub>, λ<sub>t</sub>, previous recurrent states and architecture A   
1: Form interruption input: $u _ { t } \gets [ a _ { t - 1 } , \lambda _ { t } ]$   
2: Encode context: ${ e _ { o , t } } ^ { \ i }  \{ { e _ { t } } ( { o _ { t } } ) , { e _ { u , t } }  \ i _ { u } ( u _ { t } )$   
3: if A = DUAL FEATURE then   
4: Concatenate features: $x _ { t  } ^ { \mathrm { D F } } \gets [ e _ { o , t } ; e _ { u , t } ]$   
5: Update shared memory $s _ { t } ^ { \mathrm { D F } }$ using $x _ { t } ^ { \mathrm { D F } }$   
6: Extract policy features: $\check { y _ { \underline { { t } } } } \gets g _ { \mathrm { D F } } ( z _ { t } ^ { \mathrm { D F } } )$   
7: Set critic input: $x _ { t } ^ { V }  x _ { t } ^ { \mathrm { { D F } } }$   
8: else if A = DUAL MEMORY then   
9: Update environmental memor $\cdot \ s _ { o , t } ^ { \mathrm { D M } }$ using e<sub>o,t</sub>   
10: Update interruption memory $s _ { u , t } ^ { \mathrm { D M } }$ using $e _ { u , t }$   
11: Concatenate hidden states: $\bar { z } _ { t } ^ { \mathrm { D M } } \gets [ z _ { o , t } ^ { \mathrm { D M } } ; z _ { u , t } ^ { \mathrm { D M } } ]$   
12: Fuse memories: $y _ { t } \gets g _ { \mathrm { D M } } ( \bar { z } _ { t } ^ { \mathrm { D M } } )$   
13: Set critic input: $x _ { t } ^ { V } \gets [ e _ { o , t } ; \bar { e } _ { u , t } ]$   
14: else if A = GATED then   
15: Compute scale and shift: $\alpha _ { t } \gets f _ { \alpha } ( e _ { u , t } ) , \delta _ { t } \gets f _ { \delta } ( e _ { u , t } )$   
16: Modulate environmental features: $\widetilde { e } _ { o , t }  \alpha _ { t } \odot e _ { o , t } + \delta _ { t }$   
17: Update gated memory $s _ { t } ^ { \mathrm { G } }$ using $\widetilde { e } _ { o , t }$   
18: Extract policy features: $\underline { y } _ { t }  g _ { \mathrm { G } } ( z _ { t } ^ { \mathrm { G } } )$   
19: Set critic input: $x _ { t } ^ { V }  \tilde { \tilde { e } } _ { o , t }$   
20: end if   
21: Evaluate nominal policy: $\pi _ { \phi , t } ( \cdot )  \pi _ { \phi } ( \cdot \mid y _ { t } )$   
22: Select nominal proposal: $\smash { a _ { t } \sim \pi _ { \phi _ { \frac { t } { r } } } ( \cdot ) }$   
23: Update critic memory $s _ { t } ^ { V }$ using $x _ { t . } ^ { V }$   
24: Estimate value: $V _ { \psi , t }  \ " { V _ { \psi } ( g _ { V } ( z _ { t } ^ { V } ) ) }$   
Ensure: $a _ { t } , \pi _ { \phi , t } ( \cdot ) , \dot { V } _ { \psi , t } ,$ and updated recurrent states

![](images/b005cd7e17490e2e1f960b7385ca97f34b2b453d09fb34221e4f1084e835e073.jpg)  
Fig. 2. Overview of proposed interruption-aware recurrent architectures: (a) Dual-feature architecture. (b) Dual-memory architecture. (c) Gated architecture.

Algorithm 2 Interruption-aware recurrent PPO training   
Require: Actor $\pi _ { \phi } ,$ critic $\overline { { V _ { \psi } , } }$ architecture A, interruption probability   
$p _ { \mathrm { i n t } } ( \kappa )$ , override-sampling distributions, and PPO/GAE hyperparameters   
1: Initialise network parameters and environment states   
2: for each fixed-length rollout do   
3: Clear the rollout buffer   
4: for each environment step t do   
5: if a new episode starts then   
6: Reset its recurrent states and action history   
7: if selected with probability $p _ { \mathrm { i n t } } ( \kappa )$ then   
8: Sample onset, duration, and supervisor to schedule override   
9: else   
10: Schedule nominal execution only   
11: end if   
12: end if   
13: Set previous-transition flag: $\lambda _ { t } \gets \mathbb { I } ( m _ { t - 1 } > 0 )$   
14: Run Algorithm 1   
15: Obtain nominal proposal $^ { a _ { t } , }$ value estimate $\widehat { v } _ { t } ,$ and new states   
16: Determine the current action source m from the episode schedule   
17: if $m _ { t } = 0$ then   
18: Execute the nominal proposal: $\bar { a } _ { t }  a _ { t }$   
19: else   
20: Obtain supervisor action $a _ { t } ^ { ( m _ { t } ) }$   
21: Execute the supervisor action: $\bar { \boldsymbol { a } } _ { t } \gets \boldsymbol { a } _ { t } ^ { ( m _ { t } ) }$   
22: end if   
23: Apply a¯<sub>t</sub> and observe $r _ { t } , o _ { t + 1 } .$ , and episode termination   
24: Set actor mask: w<sub>t</sub> $ \mathbb { I } ( m _ { t } = 0 )$   
25: Store the transition, nominal-action log-probability, ${ \widehat { v } } _ { t } ,$ and w   
26: Carry the updated recurrent states forward   
27: end for   
28: Compute GAE advantages $\widehat { A } _ { t }$ and return targets $\widehat { G } _ { t }$   
29: Mask actor advantages: $\widetilde { A } _ { t } \gets w _ { t } \widehat { A } _ { t }$   
30: for each PPO epoch and recurrent minibatch do   
31: Reconstruct the ordered recurrent sequences using the stored   
episode-start flags   
32: Re-evaluate the actor and critic along each sequence   
33: Form PPO ratios using the stored nominal-action log-probabilities   
34: Compute the masked clipped actor objective   
35: Compute the unmasked value and entropy terms   
36: Update the network parameters using the PPO loss equation   
37: end for   
38: end for   
Ensure: Trained actor and critic

## IV. EVALUATION METHODOLOGY

## A. Simulation and Training Configurations

Table IV reports the numerical settings used throughout training and experimentation, for the task described in Section II and the RL architectures introduced in Section III.

## B. Supervisor Behaviours

Distinct supervisor behaviours are implemented to reflect safety-oriented, neutral, and adverse overrides. Only the first four behaviours are sampled uniformly during training, allowing for evaluation with an unknown supervisor.

• Handcrafted: a heuristic supervisor that uses the same observation as the nominal policy. Intruder proximity is mapped to a set of turn commands that avoid collisions and track the observed goal bearing. This represents a simple on-board RTA system that uses the same observation as the nominal learning-based policy.

• Simple oracular: a privileged supervisor with access to the true ego and intruder kinematics. It samples a heading towards the goal and selects the closest yaw rate that remains outside a safety radius. This represents a groundbased RTA system with access to networked information.

• Max-separation: a safety-only oracular supervisor that turns away from the nearest intruder, with no goal term. This can represent a conservative remote pilot with access to additional networked information.

• Antagonistic: an inverted oracular planner that steers away from the goal and towards the nearest intruder, subject to a hard floor that avoids an immediate collision. It tests robustness to a strongly off-nominal supervisor and represents a malicious actor temporarily taking control.

• Hold-heading: a non-reactive supervisor that applies zero yaw rate and continues moving in straight line. This represents an interruption caused by temporary loss of the C2 link, during which the UAS continues moving in a straight line until the link is restored.

## C. Paired Experimental Design

Matched simulation pairs are used to isolate interruptions from the effects of initial-condition and stochastic-traffic variability. For each environment seed, a reference arm is executed entirely by the nominal policy, while an interrupted arm uses the same initial conditions and environmental seed but includes an interruption, as illustrated in Fig. 3. Any differences can therefore be attributed to the interruption and its consequences.

1) Scheduled interruptions: The core study considers four supervisors with interruption onset $t ^ { \mathrm { o n } } \in \{ 1 0 0 , 1 4 0 \}$ and duration $d \in \{ 1 0 , 2 0 \}$ , each with 500 matched pairs.

2) Hidden-state transplant: A hidden-state transplant study is used to separate the effect of a degraded recurrent representation from the physical displacement caused by an interruption, considering interruptions with $t ^ { \mathrm { o n } } \in \{ 6 0 , 1 0 0 \}$ and d = 20. A third arm is introduced that replays the reference arm’s physical trajectory while injecting the hidden state of the interrupted arm at handback, as illustrated in Fig. 3.

3) Multiple interruptions: The effect of repeated interruptions is evaluated using the patterns $4 \times 5 , 2 \times 1 0 , 1 \times 2 0$ and $3 \times 1 0$ , where the first value denotes the number of interruptions and the second their duration in steps.

4) Naive interruption-handling: Three naive solutions are applied by keeping the environmental observation untouched, freezing it at interruption onset, and masking it during the interruption. In each case, the effect of resetting the recurrent state after an interruption is also investigated.

5) State-triggered interruptions: State-triggered interruptions are evaluated using a max-separation supervisor that triggers when the nearest-intruder separation falls below 2 m and hands back after all intruders clear 3 m, subject to a maximum duration of 60 steps. A probabilistic trigger is first considered, whose per-step engagement probability increases as separation decreases, up to a maximum of 0.6. A deterministic trigger is also implemented to consistently avoid collisions.

6) Human-in-the-loop demonstration: As a qualitative complement to the scripted supervisors, a human-in-the-loop demonstration is conducted through an interactive map-based interface based on the physics-informed PyBullet framework [54]. The operator observes the aircraft in real time, and may temporarily interrupt the agent to issue steering commands. This provides a complementary test with less structured supervisory behaviour.

## D. Hidden-State Probing

To measure how much task-relevant information remains in recurrent memory, a linear ridge probe is fitted for each target variable using the actor’s LSTM hidden state $z _ { t } ,$ , whereby successful decoding indicates that a variable is linearly recoverable from $z _ { t } ~ [ 2 4 ]$ , [39]. Targets include ego kinematics, destination geometry, intruder ranges, the next nominal action, and past and future environmental states. Reference probes are trained on reference-arm states from 250 seeds and evaluated on 250 disjoint seeds, so scores reflect information that generalises across episodes. Since targets have different units and scales, each is standardised, and decodability is calculated separately per target using (26).

$$
\mathcal { D } = 1 - \mathrm { R M S E } ^ { 2 } ,\tag{26}
$$

where $\mathcal { D } = 1$ denotes perfect decoding. Scores are averaged across targets and evaluated over two windows. A sustained window covers steps 15-60 after handback and tests whether the degradation persists.

A probe may perform worse on interrupted-arm states because information is genuinely lost, corresponding to memory degradation. However, information may still be present but encoded differently after the interruption. To reduce sensitivity to such representational shifts, we leverage two additional metrics. The corrected gap directly subtracts the shift from the reference probe, measured as the additional error incurred by applying the reference probe to interrupted-arm states. The within-arm gap avoids the probe mismatch altogether by socring each arm using a probe fitted to that arm.

## E. Evaluation Metrics

Experiments are evaluated using the following metrics, and uncertainty is reported as 95% confidence intervals. In all cases, pairs in which an arm collides during the interruption or within 15 steps of handback are excluded to avoid reflecting collisions caused by poor supervisor performance, or a supervisor directly handing back authority before a collision.

• Post-handback collision rate: the proportion of pairs in which the interrupted arm collides after handback, but the reference arm does not.

• Lost-success rate: the proportion of pairs in which the interrupted arm fails, but the reference arm succeeds.

• Collision and success rates: the unconditional proportion of episodes that collide or reach the destination.

• Belief degradation: the reduction in variable decodability between the reference and interrupted recurrent states, using the corrected or within-arm gap defined above.

![](images/be83ef33e9f7b6e410ddf8f36df0b46291a874c8b3863c019b831008ede91967.jpg)  
Fig. 3. Reference, interrupted and transplanted arms used to evaluate the impact of interruptions and interruption-aware recurrent architectures.

TABLE IV  
NUMERICAL SETTINGS FOR SIMULATION AND TRAINING
<table><tr><td>Component</td><td>Parameter</td><td>Setting</td></tr><tr><td rowspan="4">Environment</td><td>Operating region</td><td>20 × 20 m planar region; start and destination randomised by episode.  $6 0 ^ { \circ } / \mathrm { s } .$ </td></tr><tr><td>Ego motion</td><td>Constant ground speed of 1.5 m/s; normalised yaw-rate action at  $\in [ - 1 , 1 ] ;$  maximum turn rate of</td></tr><tr><td>Temporal resolution</td><td> $\Delta t = 0 . 1 { \mathrm { s } } ;$  maximum training-episode horizon of 500 steps.</td></tr><tr><td>Goal and collision geometry</td><td>Goal radius of 0.5 m; ego and intruder radii of 0.15 m.</td></tr><tr><td rowspan="3">Traffic</td><td>Intruder count</td><td>Training curriculum of {3, 5, 7, 8, 9, 10} intruders; ten intruders during evaluation.</td></tr><tr><td>Intruder motion</td><td>Constant ground speed of 0.5 m/s with reflection at the operating-region boundary.</td></tr><tr><td>Heading perturbations</td><td>Applied independently with probability 0.04 per step; magnitude bounded by 0.7 rad.</td></tr><tr><td rowspan="4">Observation</td><td>Traffic tracks</td><td> $\overline { { K = 3 } }$  nearest available tracks within an 8 m sensing range.</td></tr><tr><td>Track refresh</td><td>Common refresh period sampled uniformly from  $\{ 2 , \ldots , 6 \}$  timesteps; all tracks are refreshed together</td></tr><tr><td></td><td>and held between updates with an increasing age indicator.</td></tr><tr><td>Track noise and blanking Destination availability</td><td>Position-noise standard deviation of 0.05 m; traffic-track blanking probability of 0.15 per step.</td></tr><tr><td rowspan="4">PPO</td><td></td><td>Destination-relative features are visible for five steps and blanked for the following 40 steps.</td></tr><tr><td>Training budget</td><td> $\overline { { 5 \times 1 0 ^ { 6 } } }$  environment steps; learning rate of  ${ \overline { { 3 \times 1 0 ^ { - 4 } } } } .$ </td></tr><tr><td>Rollout and optimisation</td><td>Rollout length of 2048; minibatch size of 256; four optimisation epochs per rollout.</td></tr><tr><td>Return estimation</td><td>Discount factor of  $\gamma = 0 . 9 9 ;$  GAE parameter  $\lambda _ { \mathrm { G A E } } = 0 . 9 5 .$ </td></tr><tr><td rowspan="3">Networks</td><td>PPO regularisation</td><td>Clipping threshold of 0.2; entropy coefficient of 0.01; observation and reward normalisation enabled.</td></tr><tr><td>Standard recurrent baseline</td><td>Separate 64-unit actor and critic LSTMs; 64-unit post-LSTM layer.</td></tr><tr><td>Proposed actors</td><td>One 64-unit LSTM for dual-feature and gated actors; two 64-unit LSTMs for dual memory.</td></tr><tr><td rowspan="3">Reward</td><td>Proposed critics</td><td>One separate 64-unit critic LSTM for every architecture.</td></tr><tr><td>Progress and completion</td><td> $\overline { { k _ { p } = 5 } }$  and terminal goal bonus  ${ \overline { { B _ { g } = 5 0 . } } }$ </td></tr><tr><td>Separation and collision</td><td> $k _ { c } = 2 . 5 , d _ { \mathrm { s a f e } } = 1 \mathrm { m } ,$  and collision penalty  $\ : P _ { c } = 3 0 . \ :$ </td></tr><tr><td rowspan="3">Curricula</td><td>Regularisation penalties</td><td>Action smoothness  $k _ { \Delta a } = 0 . 1$  , time  $c _ { \mathrm { t i m e } } = 0 . 0 5 ,$  and boundary contact  $P _ { b } = 0 . 3 .$ </td></tr><tr><td>Traffic progression</td><td>Advance after at least 75% success over 50 episodes and  $\overline { { 7 . 5 \times 1 0 ^ { 5 } } }$  steps at the current traffic level.  $2 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Interruption probability Training interruption</td><td>Linear increase from zero to 0.5 over the first training steps. One segment per selected episode; onset is uniform over steps 20–200; duration is uniform over 5–20</td></tr></table>

## V. RESULTS AND DISCUSSION

## A. Recurrent Memory is Required for the Simulated Task

Fig. 4 compares the training curve of a memoryless multilayer perceptron (MLP) PPO agent [46] against the LSTMbased recurrent PPO baseline. Under the same training curriculum, the memoryless policy plateaus at a success rate of approximately 10% with three intruders and never reaches the 75% success threshold needed to progress the intruder curriculum. Conversely, the recurrent policy exceeds this threshold at every curriculum stage and progresses to the final ten-intruder setting. Its episode length also decreases as training progresses, consistent with a policy that generates increasingly efficient UAS trajectories. This confirms that recurrent memory is essential for effective navigation in the simulated task.

## B. Recurrent Memory Learns Task-Relevant Information

Linear probes show that the actor’s LSTM hidden state contains information relevant to the navigation task, as summarised in Fig. 5. The current ego position $( \mathcal { D } = 0 . 8 0 \ – 0 . 8 6 )$ and nearest-intruder range (D = 0.72) are strongly decodable from recurrent memory. The next action achieves the highest score $( \mathcal { D } ~ = ~ 0 . 9 1 )$ , since the policy head selects its action from the same hidden representation. The recurrent state also retains information that is no longer directly available in the current observation. Ego position and nearest-intruder range from 10 steps earlier remain strongly decodable $( \mathcal { D } = 0 . 7 2 $ 0.84), with decodability decreasing to 0.55-0.64 at a 30- step lag as expected for a gradually fading memory trace.

![](images/9a2f4823ffaef72273d309bd4988127cad981fdd09b47fdc7148d134a18cc7df.jpg)  
Fig. 4. Training curves of memoryless (pink) and recurrent (gray) PPO agents for the simulated UAS navigation task.

Moreover, the future intruder range is decodable both 10 and 30 steps ahead (D = 0.70 and 0.69). This suggests that the hidden state contains information about the evolving encounter geometry from which near-term separation can be predicted. These results therefore support interpreting the hidden state as a learned, task-dependent information state containing current, historical, and short-horizon predictive information.

## C. Interruptions Reduce Task-Variable Decodability

Fig. 6 compares the post-handback decodability gap between matched reference and interrupted arms for different task-relevant variables, and for different interruption onsets and durations. A positive value indicates that less information remains linearly recoverable after an interruption.

![](images/9e10ae2ac8585fdc0782640e19f03f6bf1d84f2e21c2f2a623b7a6f46c678982.jpg)  
Fig. 5. Absolute decodability score of the reference arm hidden states for different task-relevant variables, averaged across a 45-step window.

![](images/bab37c475ae7185d8d69cdac172237815cb29f311a4c2a14e3d6c5232f23991a.jpg)  
Fig. 6. Memory degradation caused by interruptions, measured as a reduction in decodability of task-relevant variables from the recurrent hidden state averaged across a 45-step post-handback window.

The largest and most consistent corrected gaps occur for policy-readout and past-world variables, whereas currentworld variables generally exhibit smaller gaps. This pattern is consistent with new observations helping to reconstruct current quantities, while historical quantities remain dependent on retained recurrent context. Future-world gaps are smaller and more variable, particularly for interruptions with later onsets. At a later handback, the remaining mission horizon is short and targets often collapse toward the present, reducing the impact of an interruption. Overall, the predominantly positive gaps confirm that interruptions reduce the task-relevant information recoverable from recurrent memory after handback.

## D. Interruptions Degrade Post-Handback Performance

At the episode level, larger decoding gaps are associated with poorer post-handback performance. Fig. 7 groups episodes into quartiles according to their corrected decoding gap and the lost-success rate increases monotonically from 39% in the lowest quartile to 54% in the highest. Notably, this association remains after controlling for displacement, heading error, and nearest-intruder distance at handback, confirming that reduced decodability is associated with interruptioninduced mission failures beyond what can be explained by physical deviations at handback.

The transplant study further isolates the contribution of the recurrent state. At handback, the interrupted arm’s actor recurrent state, comprising both hidden and cell states, is inserted into the matched reference arm while its physical state and current observation remain unchanged. This increases the post-handback collision rate by 4.6% and suggests that the recurrent state produced during an interruption can independently increase post-handback collision risk even when the physical handback state is held constant. Nonetheless, recurrent memory is not the only reason why interruptions affect performance. In fact, both the modified aircraft trajectory and the altered recurrent state contribute to degraded posthandback performance.

![](images/7bd555afb538dc86d88c1905db939f8191e6dd42f7f493a7121e1593cd5fd9c2.jpg)  
Fig. 7. Lost-success rate for episodes with different corrected decoding gaps, confirming that greater memory degradation results in poorer post-handback task performance.

## E. Action-Observation Feedback Sustains the Degradation

When supplying reference and interrupted arms identical observations after handback, the resulting command difference is illustrated in Fig. 8, decomposed into contributions from the LSTM carried cell, candidate update, and gate activation signals. Carried cell content shows the largest contribution, confirming that accumulated context influences post-handback commands. Moreover, under identical observations, recurrentstate differences normalised by their initial value at handback continue to decrease, as illustrated in Fig. 9. The reduction is substantially weaker in actual closed-loop flights, where each arm follows its own trajectory and receives different observations. These results support a feedback hypothesis, whereby altered memory changes commands, while changed commands and interruption-induced displacement alter subsequent observations. Those observations in turn update recurrent memory, which shapes later commands in a closed feedback loop. This loop can continually generate new recurrent-state differences and prolong post-handback disruption, even as the original interruption-time differences diminish.

![](images/4be873612fcce551df629d7c976f013d79677ba5ffca294af7313927e7c7de85.jpg)  
Fig. 8. Contribution of different signals to command differences, with carried cell content showing the largest contribution.

![](images/4599aed0d98bfb6570a76f263d496da01bb6b737a0393aa54795c31fd7ed91ad.jpg)  
Fig. 9. Contraction of the reference and interrupted arm recurrent states after handback with shared inputs (blue) and closed-loop feedback (orange).

P(interrupted fails | reference succeeded)

## F. Longer Contiguous Interruptions Are More Harmful

The effect of an interruption depends on supervisor behaviour and the duration for which it remains active, as shown in Fig. 10. The antagonistic supervisor produces the highest lost-success rates, increasing from 25% for a 10-step interruption to 30% for a 20-step interruption. Hold-heading and handcrafted behaviours similarly increase from 18-20% to 23%. The potential-field and max-separation supervisors are less harmful, with rates of approximately 15% and 18%, respectively. The oracular planner produces the lowest lostsuccess rate at both durations. In general, longer and more off-nominal interruptions are more harmful.

Fig. 11 shows that the lost-success rate increases from approximately 30% for four five-step segments, to 37% for two ten-step segments, and to 48% for one continuous 20- step interruption. Conversely, three ten-step segments produce a lost-success rate of approximately 35% despite increasing total interruption exposure to 30 steps. Mission harm is therefore more closely associated with the longest contiguous interruption than with the total interrupted time. This pattern is expected due to recurrent-state degradation accumulating during a sustained period of supervisor-generated transitions. Intermittent periods of nominal execution may allow the policy to incorporate new experience generated by the nominal policy and partially realign its recurrent state before the degradation becomes too severe.

![](images/8bdd25e3bc1e73e4e1f1391b993377859303b1212b12ad1481b88a7d4b218fa9.jpg)

Fig. 10. Lost success rate for different supervisor behaviours and durations, showing that longer and less optimal supervisors are generally more harmful.  
![](images/39c9df1eaf23c877d1f05ded4be1b3036da56a12f8206be09076e6eaa91acc0f.jpg)  
Fig. 11. Lost success rate for episodes with multiple interruption segments, demonstrating that longer contiguous interruptions are more harmful.

## G. Naive Mitigations Are Insufficient

Two simple mitigation strategies are investigated as naive solutions to external interruptions. Fig. 12 shows that resetting the recurrent state at handback worsens the post-handback collision rate since it discards useful memory along with interruption-associated context. Similarly, freezing or masking the observations during an interruption does not reliably improve post-handback performance.

## H. Interruption-Aware Architectures Preserve Task-Variable Decodability and Improve Post-Handback Performance

Fig. 13 verifies that the proposed interruption-aware architectures learn the nominal navigation task without sacrificing interruption-free performance. All policies progress through the intruder curriculum, with the three interruption-aware architectures achieving comparable or higher success rates compared to the standard recurrent baseline. A capacity- and encoder-matched baseline is also included for reference.

Recurrent-memory degradation and interruption-induced mission harm for each architecture are summarised in Fig. 14. All three interruption-aware architectures mitigate the impact of interruptions, resulting in lower recurrent state degradation and improved post-handback performance. This is further confirmed in Fig. 15, where the post-handback collision rate decreases from 0.15 for the recurrent baseline to 0.04-0.06 for the proposed architectures, and the lost-success rate decreases from 0.23 to 0.07-0.14. In the strongest case, these changes correspond to relative reductions of approximately 74% in post-handback collisions and 69% in interruption-induced lost successes. Moreover, memory-induced command differences were also observed to decay faster under the interruptionaware architectures. This shorter persistence limits the action–observation feedback that may otherwise prolong posthandback disruption.

Fig. 16 provides a qualitative example of the resulting behaviour. Under the same environment seed and scheduled interruption configuration, the recurrent baseline enters an unstable trajectory after handback and fails to reach the destination, whereas all three interruption-aware policies recover and complete the mission.

![](images/0e2ae7864491caf2899801d2d7248558325b6b665c294459a02bc96985da41b5.jpg)  
Fig. 12. Naive mitigations fail to improve post-handback performance.

![](images/ace38f96240fdc368142b386e1e84da7f338bccf712b2144939164a16eeea8eb.jpg)

Fig. 13. Training curves showing convergence of interruption-aware and recurrent baseline policies.  
![](images/875c3ef36a2a4f62ea0cbf9a4440dd7a37083efff810308582433586c49eff68.jpg)  
Fig. 14. Belief degradation against behavioural harm for each architecture, confirming that the interruption-aware architectures perform better than the recurrent baseline.

![](images/7bd57b7a93597bda84fc0870a51783995306a9b67872d799f183fc66dbfe591c.jpg)  
Fig. 15. Post-handback performance improvements with interruption-aware architectures.

![](images/e24228c20861049a3ea402fea12321e36a1d6bff62a833d23bcd84f054f6f1f0.jpg)  
Fig. 16. Example trajectories generated by each architecture for the same environmental scenario.

## I. Results Under State-Triggered and Human Interruptions

Operational safety interruptions normally respond to a developing traffic conflict. Fig. 17 therefore evaluates the statetriggered supervisory scheme described in Section IV. Under probabilistic triggering, the lost-success rate is approximately 0.47 for the recurrent baseline, compared with 0.24-0.38 for the interruption-aware architectures. The gated architecture performs best in this regime. Under deterministic triggering, the dual feature achieves the lowest rate at approximately 0.24, followed by the gated architecture at 0.29. The dual memory architecture reaches approximately 0.47 and provides little improvement over the baseline. Consistent state-triggered interruptions therefore do not necessarily reduce failures. Although the supervisor is designed to address collision risk, the resulting manoeuvres and handback may still prevent an otherwise successful mission from reaching its destination.

The human-in-the-loop experiment provides a complementary test using less structured supervisory behaviour. Through the interface shown in Fig. 18, a human operator was instructed to intervene once per episode to avoid a collision or redirect the aircraft towards its destination. Forty human-interrupted episodes were collected for each architecture, with onset, duration, and steering commands selected by the operator. As shown in Fig. 19, the three interruption-aware architectures reduce the lost-success rate from approximately 0.65 under the recurrent baseline to 0.21-0.25. This demonstrates that the interruption-aware architectures also improve robustness to operator-generated manoeuvres.

![](images/0456f0ebd6185dd770e88c7936e2964a137d4dedba4b05fc661b740ab1aa2374.jpg)  
Fig. 17. Performance improvements of interruption-aware architectures with state-triggered interruptions.

![](images/aaa61b3cb938c918f950d370f1e3b51cc9217948b11924455bf1f76931ef0ac7.jpg)  
Fig. 18. Interface used for the human-in-the-loop demonstration, based on the physics-informed PyBullet simulation framework.

![](images/fcb1ebe45ebdeb4394fd7e4c301e8ce952e863b8a3a896f7e1b87679c2989499.jpg)  
Fig. 19. Performance improvements of interruption-aware architectures with human operator-based interruptions.

## J. Discussion and Operational Implications

The results identify handback reliability as an additional safety requirement. Although the interruption may resolve an immediate hazard, it can degrade the policy’s recurrent state and reduce task-variable decodability, especially for historical variables that cannot be reconstructed from new observations. This can increase collision risk and degrade post-handback performance. Notably, this vulnerability extends beyond the handback instant. Carried cell content continues to influence commands, while changed commands and supervisor-induced displacement alter subsequent observations. Those observations update the memory used for later decisions, sustaining a post-handback disruption feedback loop. Recovery therefore depends on how recurrent context is used and updated when the agent regains authority.

The risk of an interruption varies with supervisor behaviour and interruption duration. Safety- and goal-directed supervisors generally cause fewer induced failures, but even helpful interruptions can cause reference successes to fail. Longer contiguous overrides are particularly disruptive as they allow recurrent state degradation to accumulate. Resetting memory and masking or freezing observations during an interruption provide no consistent improvement. Conversely, all three proposed interruption-aware architectures preserve more task-relevant information and improve post-handback performance, including under unseen supervisor behaviours. Explicit override context enables the proposed architectures to recalibrate how carried recurrent memory influences commands after handback. This shorter persistence can limit the action-observation feedback loop that otherwise prolongs post-handback disruption. Among the proposed designs, dual feature and gated architectures achieve the best performance, while the dual memory architecture performs less strongly but still improves substantially over the baseline. Dual memory’s weaker results may reflect its integration of interruption context at action selection, whereas the dual feature and gated architectures allow that context to directly shape how environmental memory is formed and updated.

## K. Limitations

The principal experiments use simplified two-dimensional dynamics, sensing, traffic and supervisor behaviours. Moreover, state-triggered evaluations and the single-operator human study only provide preliminary evidence of broader applicability. Higher-fidelity evaluation should incorporate threedimensional dynamics, realistic surveillance and communication delays, standards-aligned safety functions, and more extensive human-in-the-loop trials. Additionally, the probes used in this study measure how readily task variables can be recovered from the recurrent hidden state. Lower decodability, however, does not necessarily imply that information has been irreversibly lost. Future work should combine nonlinear probes with carefully controlled interruptions to clarify what information remains available and better study how it influences post-handback decisions. Further training seeds can also be leveraged for more robust results.

## VI. CONCLUSION

This paper characterises temporary overrides as a handback reliability problem for recurrent UAS navigation under partial observability. The problem is formulated as a PO-MAMDP that distinguishes nominally proposed actions from externally modified actions. Controlled experiments show that interruptions can reduce task-variable decodability, cause otherwise successful missions to fail, and increase collision risk through changes in the recurrent state. Notably, harm depends on supervisor behaviour and the longest contiguous interruption. Moreover, degradation is likely sustained through a post-handback feedback loop in which carried recurrent context influences commands and subsequent experience renews recurrent-state differences after handback.

Three interruption-aware architectures are proposed to explicitly represent interruption context through separate feature streams, dedicated interruption memory, or gated environmental features. These reduce the persistence of interruptionassociated memory effects on commands, limiting actionobservation feedback that could prolong post-handback disruption. Under scheduled interruptions, the strongest evaluated design reduces post-handback collision harm by 74% and interruption-induced lost successes by 69% relative to the recurrent baseline, without degrading interruption-free performance. State-triggered and pilot-generated tests provide preliminary evidence that these benefits extend beyond scheduled supervisor behaviour.

These findings establish handback reliability as a distinct reliability concern for recurrent autonomy. Future work should assess robustness across training seeds and higher-fidelity environments, explore solutions that recover incorrect carried recurrent context, and explore the impact of interruptions on different learning algorithms and recurrent architectures. Moreover, future research should investigate the impact of interruptions in hierarchical and multi-agent navigation settings.

## ACKNOWLEDGMENTS

The authors would like to thank the European Commission and the SESAR Joint Undertaking for the initiation and funding of the SAFIR-Ready project. The SAFIR-Ready project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101114855.

## REFERENCES

[1] Federal Aviation Administration, “Unmanned aircraft system (UAS) traffic management (UTM) concept of operations, version 2.0,” Federal Aviation Administration, Tech. Rep., 2020. [Online]. Available: https://www.faa.gov/researchdevelopment/trafficmanagement/ utm-concept-operations-version-20-utm-conops-v20

[2] European Commission, “Commission implementing regulation (EU) 2021/664 of 22 april 2021 on a regulatory framework for U-space,” Official Journal of the European Union, L 139, pp. 161–183, Apr. 2021. [Online]. Available: https://eur-lex.europa.eu/eli/reg impl/2021/ 664/oj/eng

[3] Minimum Operational Performance Standards (MOPS) for Detect and Avoid (DAA) Systems, RTCA, Inc. Std. DO-365C, 2022.

[4] Standard Practice for Methods to Safely Bound Behavior of Aircraft Systems Containing Complex Functions Using Run-Time Assurance, ASTM International Std. ASTM F3269-21, 2021.

[5] Standard Specification for Detect and Avoid System Performance Requirements, ASTM International Std. ASTM F3442-25, 2025.

[6] European Commission, “Commission implementing regulation (EU) 2019/947 of 24 may 2019 on the rules and procedures for the operation of unmanned aircraft,” Official Journal of the European Union, L 152, pp. 45–71, Jun. 2019. [Online]. Available: https: //eur-lex.europa.eu/eli/reg impl/2019/947/oj/eng

[7] J. D. Schierman, M. D. DeVore, N. D. Richards, and M. A. Clark, “Runtime assurance for autonomous aerospace systems,” Journal of Guidance, Control, and Dynamics, vol. 43, no. 12, pp. 2205–2217, 2020.

[8] K. L. Hobbs, M. L. Mote, M. C. Abate, S. D. Coogan, and E. M. Feron, “Runtime assurance for safety-critical systems: An introduction to safety filtering approaches for complex control systems,” IEEE Control Systems Magazine, vol. 43, no. 2, pp. 28–65, 2023.

[9] J. T. Slagel, L. M. White, A. Dutle, C. A. Munoz, and N. Crespo,˜ “A verification framework for runtime assurance of autonomous uas,” in 2024 AIAA DATC/IEEE 43rd Digital Avionics Systems Conference (DASC), 2024, pp. 01–08.

[10] European Union Aviation Safety Agency, “Easy access rules for unmanned aircraft systems,” EASA, Tech. Rep., Jun. 2026, revision from June 2026. [Online]. Available: https://www.easa.europa.eu/en/document-library/easy-access-rules/ online-publications/easy-access-rules-unmanned-aircraft-systems

[11] Z. Yu, Z. Wang, J. Yu, D. Liu, H. H. Song, and Z. Li, “Cybersecurity of unmanned aerial vehicles: A survey,” IEEE Aerosp. Electron. Syst. Mag., vol. 39, no. 9, pp. 182–215, 2024.

[12] F. Candan, A. Beke, M. Mahfouf, and L. Mihaylova, “A real-time fuzzy interacting multiple-model velocity obstacle avoidance approach for unmanned aerial vehicles,” Journal of Intelligent & Robotic Systems, vol. 110, no. 2, p. 61, 2024.

[13] O. F. Archila, A. V. Wouwer, and J. Schiffer, “A multiple artificial potential functions approach for collision avoidance in uav systems,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 10, pp. 16 688–16 699, 2025.

[14] A. Hamissi, A. Dhraief, and L. Sliman, “A comprehensive survey on conflict detection and resolution in unmanned aircraft system traffic management,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 2, pp. 1395–1418, 2025.

[15] M. R. Rezaee, N. A. W. A. Hamid, M. Hussin, and Z. A. Zukarnain, “Comprehensive review of drones collision avoidance schemes: Challenges and open issues,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 7, pp. 6397–6426, 2024.

[16] Federal Aviation Administration, “Urban air mobility (UAM) version 2.0: Concept of operations,” Federal Aviation Administration, Tech. Rep., 2023. [Online]. Available: https://www.faa.gov/sites/faa.gov/files/Urban%20Air%20Mobility% 20%28UAM%29%20Concept%20of%20Operations%202.0 1.pdf

[17] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[18] M. Brittain and P. Wei, “Scalable autonomous separation assurance with heterogeneous multi-agent reinforcement learning,” IEEE Transactions on Automation Science and Engineering, vol. 19, no. 4, pp. 2837–2848, 2022.

[19] P. Zhao and Y. Liu, “Physics informed deep reinforcement learning for aircraft conflict resolution,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 7, pp. 8288–8301, 2022.

[20] Y. Li, Y. Zhang, T. Guo, Y. Liu, Y. Lv, and W. Du, “Graph reinforcement learning for multi-aircraft conflict resolution,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 3, pp. 4529–4540, 2024.

[21] D.-T. Pham, N. P. Tran, S. K. Goh, S. Alam, and V. Duong, “Reinforcement learning for two-aircraft conflict resolution in the presence of uncertainty,” in 2019 IEEE-RIVF International Conference on Computing and Communication Technologies (RIVF), 2019, pp. 1–6.

[22] F.-M. Luo, Z. Tu, Z. Huang, and Y. Yu, “Efficient recurrent off-policy RL requires a context-encoder-specific learning rate,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 48 484–48 518.

[23] A. Singla, S. Padakandla, and S. Bhatnagar, “Memory-based deep reinforcement learning for obstacle avoidance in uav with limited environment knowledge,” IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 1, pp. 107–118, 2021.

[24] G. Lambrechts, A. Bolland, and D. Ernst, “Recurrent networks, hidden states and beliefs in partially observable environments,” Transactions on Machine Learning Research, Aug. 2022. [Online]. Available: https://openreview.net/forum?id=dkHfV3wB2l

[25] B. Grindley, K. J. Parnell, T. Cherett, J. Scanlan, and K. L. Plant, “Understanding the human factors challenge of handover between levels of automation for uncrewed air systems: a systematic literature review,”

Transportation Planning and Technology, vol. 48, no. 6, pp. 1383–1408, 2025.

[26] B. Zhang, J. de Winter, S. Varotto, R. Happee, and M. Martens, “Determinants of take-over time from automated driving: A meta-analysis of 129 studies,” Transportation Research Part F: Traffic Psychology and Behaviour, vol. 64, pp. 285–307, 2019.

[27] F. Zhou, X. J. Yang, and J. C. F. de Winter, “Using eye-tracking data to predict situation awareness in real time during takeover transitions in conditionally automated driving,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 3, pp. 2284–2295, 2022.

[28] Z. Ye, K. Wang, Y. Chen, X. Jiang, and G. Song, “Multi-uav navigation for partially observable communication coverage by graph reinforcement learning,” IEEE Transactions on Mobile Computing, vol. 22, no. 7, pp. 4056–4069, 2023.

[29] S. Gu, L. Yang, Y. Du, G. Chen, F. Walter, J. Wang, and A. Knoll, “A review of safe reinforcement learning: Methods, theories, and applications,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 11 216–11 235, 2024.

[30] B. Konighofer, R. Bloem, N. Jansen, S. Junges, and S. Pranger, “Shields¨ for safe reinforcement learning,” Commun. ACM, vol. 68, no. 11, pp. 80–90, 2025.

[31] S. Mo, X. Pei, and C. Wu, “Safe reinforcement learning for autonomous vehicle using monte carlo tree search,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 7, pp. 6766–6773, 2022.

[32] L. Orseau and S. Armstrong, “Safely interruptible agents,” in Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence, 2016, pp. 557–566. [Online]. Available: https: //dl.acm.org/doi/10.5555/3020948.3021006

[33] N. Soares, B. Fallenstein, E. Yudkowsky, and S. Armstrong, “Corrigibility,” in AAAI Workshop on AI and Ethics, 2015. [Online]. Available: https://intelligence.org/2014/10/18/new-report-corrigibility/

[34] E. Firt, “Addressing corrigibility in near-future AI systems,” AI Ethics, vol. 5, pp. 1481–1490, 2025.

[35] R. Carey and T. Everitt, “Human control: Definitions and algorithms,” in Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, vol. 216, 2023, pp. 271–281. [Online]. Available: https://proceedings.mlr.press/v216/carey23a.html

[36] W. Huang, H. Liu, Z. Huang, and C. Lv, “Safety-aware human-in-theloop reinforcement learning with shared control for autonomous driving,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 16 181–16 192, 2024.

[37] E. Langlois and T. Everitt, “How RL agents behave when their actions are modified,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 13, pp. 11 586–11 594, 2021.

[38] M. Suau, J. He, E. Congeduti, R. A. N. Starre, A. Czechowski, and F. A. Oliehoek, “Influence-aware memory architectures for deep reinforcement learning in POMDPs,” Neural Comput & Applic, vol. 37, no. 19, pp. 13 145–13 161, 2025.

[39] Y. Belinkov, “Probing classifiers: Promises, shortcomings, and advances,” Computational Linguistics, vol. 48, no. 1, pp. 207–219, 2022.

[40] A. Goyal, A. Lamb, J. Hoffmann, S. Sodhani, S. Levine, Y. Bengio, and B. Scholkopf, “Recurrent independent mechanisms,” in¨ International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=mLcmdlEUxy-

[41] R. N. Hasanah, S. Wiyanto, M. A. Muslim, N. Sulistiyanto, and R. A. Setayawan, “From PID to reinforcement learning: Evolution of control strategies in quadcopter UAVs,” ITEGAM-JETIA, vol. 12, no. 59, p. 51–66, May 2026.

[42] J. Song, K. Zhao, and Y. Liu, “Survey on mission planning of multiple unmanned aerial vehicles,” Aerospace, vol. 10, no. 3, p. 208, 2023.

[43] F. AlMahamid and K. Grolinger, “Autonomous unmanned aerial vehicle navigation using reinforcement learning: A systematic review,” Engineering Applications ofArtificial Intelligence, vol. 115, p. 105321, 2022.

[44] M. Lauri, D. Hsu, and J. Pajarinen, “Partially observable markov decision processes in robotics: A survey,” IEEE Transactions on Robotics, vol. 39, no. 1, pp. 21–40, 2023.

[45] Y. Cai, X. Liu, A. Oikonomou, and K. Zhang, “Provable partially observable reinforcement learning with privileged information,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 63 790– 63 857.

[46] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv:1707.06347, 2017.

[47] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through

deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[48] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in International Conference on Learning Representations (ICLR), 2016. [Online]. Available: https://arxiv.org/abs/1509.02971

[49] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in Proceedings of the 32nd International Conference on Machine Learning, vol. 37, 2015, pp. 1889–1897. [Online]. Available: https://proceedings.mlr.press/v37/schulman15.html

[50] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” in International Conference on Learning Representations (ICLR), 2016. [Online]. Available: https://arxiv.org/abs/1506.02438

[51] A. Sherstinsky, “Fundamentals of recurrent neural network (RNN) and long short-term memory (LSTM) network,” Physica D: Nonlinear Phenomena, vol. 404, p. 132306, 2020.

[52] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1724–1734.

[53] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, pp. 3942– 3951, 2018.

[54] E. Coumans and Y. Bai, “PyBullet: A python module for physics simulation for games, robotics, and machine learning,” Software, 2016–2021. [Online]. Available: https://github.com/bulletphysics/bullet3