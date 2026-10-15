# Runtime Monitoring of Supermartingale Certificates

Mirco Giacobbe<sup>1</sup> Thomas A. Henzinger<sup>2</sup> Konstantin Kuefner<sup>5</sup> Emily Yu<sup>3</sup> Ðorđe Žikelić<sup>4</sup>

<sup>1</sup>University of Birmingham <sup>2</sup>Institute of Science and Technology Austria <sup>3</sup>Leiden University <sup>4</sup>Nanyang Technological University <sup>5</sup>University of Oxford

## Abstract

Supermartingale certificates formally guarantee temporal properties of discrete and continuous-time stochastic dynamical systems. However, these guarantees rely on a mathematical model of the world, leaving a sim-to-real gap that can invalidate them for the deployed system. We formulate the runtime monitoring problem for supermartingale certificates, in which a monitor observes a single system trajectory and raises an alarm when there is suficient statistical evidence that the supermartingale certificate has been violated during execution. We present an algorithm to automatically construct a monitor using betting-based sequential tests that bound the probability of ever raising a false alarm at a user-specified level. Within this betting framework, we formulate strategies that allow the monitor to exploit spatial regularity to improve detection power while preserving this statistical guarantee. We demonstrate the eficacy of our approach on supermartingale certificate benchmarks from the literature.

## 1 Introduction

Supermartingale certificates guarantee that probabilistic, potentially infinite-state systems satisfy temporal properties of their intended behaviour, making them a powerful tool for assuring sequential decision-making under uncertainty. Applications include the automated control of unmanned vehicles, manufacturing processes, and financial systems, where rigorous guarantees of safety and security are paramount. Supermartingale certificates are real-valued functions of system states whose expected values decrease along system executions in a manner determined by the specification of interest. They are the probabilistic analogues of Lyapunov and barrier functions in control systems (Kushner, 1965), and of inductive invariants and progress measures in software systems (Floyd, 1967; Chakarov and Sankaranarayanan, 2013). A supermartingale certificate reduces reasoning about a global temporal property of system behaviour—such as reachability, safety, reach-avoidance, stability, reactivity—to verifying a local expected-decrease condition at each individual step of the system.

The algorithmic construction of supermartingale certificates has traditionally relied on rigorous symbolic reasoning methods (Prajna and Jadbabaie, 2004; Chatterjee et al., 2016), with more recent work turning to neural representations and certified learning-based approaches (Abate et al., 2021; Lechner et al., 2022; Zikelic et al., 2023a; Neustroev et al., 2025). Yet, the formal guarantees provided by the existing methods are contingent on a given model of the world, fixed at design time. If the behaviour of the environment in which the controller is deployed deviates from this assumed model, the certificate may no longer satisfy the supermartingale conditions, invalidating its guarantees. Such deviations may arise, for example, from environmental changes, distribution shifts, or unmodelled physical conditions, and are commonly associated with what is known in control theory and robotics as the sim-to-real gap (Jakobi et al., 1995; Tobin et al., 2017; Zhao et al., 2020).

This raises a fundamental question: when the true environment is not fully known and can only be interacted with at runtime as a black box, how can we determine whether a certificate remains valid after deployment? This is the central question we address in this work: we propose for the first time a framework for the runtime monitoring of supermartingale certificates upon deployment. Given a supermartingale certificate computed at design time and a a black-box stochastic deployment environment, our method observes a single stream of system states, continually tests whether the supermartingale certificate satisfies its defining expected-decrease condition, and raises an alarm once it has gathered suficient statistical evidence that the supermartingale certificate is invalid.

Runtime monitoring of a supermartingale certificate requires detecting violations from a single stream of observations. Because the environment is stochastic, an increase (resp. decrease) in the certificate value at any individual step does not necessarily indicate a violation (resp. satisfaction) of the expected-decrease condition. To address this challenge, our monitor accumulates statistical evidence over successive state observations, seeking to determine when the observed changes in certificate value become suficiently inconsistent with the expected-decrease condition to signal a violation. This evidence is accumulated using a betting-based sequential testing method (Ramdas et al., 2023; Waudby-Smith and Ramdas, 2024). The monitor plays a betting game: it starts with an initial wealth and, for each pair of successive state observations, converts the change in certificate value into a normalised score and chooses a stake based on the observations seen so far. A positive score increases the monitor’s wealth, while a negative score decreases it, with the magnitude of either change determined by the chosen stake. Crucially, while the expecteddecrease condition holds, no admissible betting strategy can increase the monitor’s wealth in expectation. Sustained wealth growth therefore constitutes statistical evidence against this condition. Once the wealth exceeds a threshold calibrated to control the false-alarm probability, the monitor signals a violation.

We implemented our method and evaluated betting strategies that either weight past observations uniformly or favour recent observations and those from nearby states across a wide range of benchmarks. Our results show that the framework efectively detects certificate violations under changes in deployment dynamics, and that exploiting temporal and spatial regularity in the betting strategy improves the detection of local violations.

Our contribution is threefold. First, we present the first framework for runtime verification of supermartingale certificates. Our framework is agnostic to the temporal property being certified, applies to both discrete- and continuous-time stochastic systems, and encompasses the existing supermartingale certificates (see Section 2). Second, we introduce an algorithm for automatically constructing monitors based on betting games, with a user-specified bound on the false-alarm probability that holds uniformly over an unbounded monitoring horizon. This bound allows users to control the monitor’s conservativeness and avoid unnecessary intervention (see Section 3). Finally, we empirically demonstrate the efectiveness of our method in detecting certificate violations under changed deployment dynamics (see Section 4).

## 2 Runtime monitoring of supermartingale certificates

Markov Process We consider probabilistic systems modelled as Markov processes in either discrete or continuous time, over which supermartingale certificates are defined. We generically write T for the time domain, with $\mathbb { T } = \mathbb { N } _ { 0 }$ (the non-negative integers) in discrete time and $\mathbb { T } = \mathbb { R } _ { > 0 }$ (the non-negative reals) in continuous time, and take $t = 0$ as the initial time.

Let $\chi = \mathbb { R } ^ { d }$ be a Euclidean space, and $( \mathcal { X } , \mathcal { F } , \mathbb { P } )$ be a probability space over X . A property holds almost surely if it holds with probability one. Let $( \mathcal { G } _ { t } ) _ { t \in \mathbb { T } }$ be a filtration in this probability space. A stochastic process $\pmb { X } = ( X _ { t } ) _ { t \in \mathbb { T } }$ with respect to $( \mathcal { G } _ { t } ) _ { t \in \mathbb { T } }$ is an indexed family of random variables, with each $X _ { t }$ being $\mathcal { G } _ { t } { \mathrm { - m e a s u r a b l e } }$ . In the continuous-time setting, we also assume that X is right-continuous and that the filtration $( \mathcal { G } _ { t } ) _ { t \in \mathbb { T } }$ is complete and right-continuous. A (time-homogeneous) Markov process $\boldsymbol { X } \ = \ ( X _ { t } ) _ { t \in \mathbb { T } }$ is a stochastic process such that, for every measurable function $f : \mathcal { X }  \mathbb { R }$ and for every pair of time values $t , s \in \mathbb { T }$ , we have $\mathbb { E } [ f ( X _ { t + s } ) \mid { \mathcal G } _ { t } ] = \mathbb { E } [ f ( X _ { t + s } ) \mid X _ { t } ]$

Supermartingale Certificate A supermartingale certificate is a measurable function $f : \mathcal { X } $ R which satisfies a finite number of conditions of one of the following two types:

1. domain-bound condition. Given a domain $s \subseteq \mathcal { X }$ and a bound interval [l, u] where l, $u \in$ $\mathbb { R } \cup \{ \pm \infty \}$ , the $( S , [ l , u ] )$ -domain-bound condition specifies that, for all times $t \in \mathbb { T }$ , we have $f ( X _ { t } ) \in [ l , u ]$ . Intuitively, this condition requires the value of f to lie within the interval $[ l , u ]$ at all times, whenever the stochastic process value is in the domain set S.

2. expected-decrease condition. Given a domain $s \subseteq { \mathcal { X } }$ and a measurable drift bound $l : \mathcal { X }  \mathbb { R }$ zero outside $s ,$ the $( S , l )$ -expected drift condition requires, for $t \in \mathbb { T }$ and $X _ { t } \in S$ , a bound on

• (discrete-time) the one-step expected decrease: $\mathbb { E } [ f ( X _ { t + 1 } ) \mid X _ { t } ] - f ( X _ { t } ) \leq l ( X _ { t } )$ ;

• (continuous-time) the instantaneous drift: $\begin{array} { r } { A f ( X _ { t } ) : = \operatorname* { l i m } _ { h \downarrow 0 } \frac { \mathbb { E } [ f ( X _ { t + h } ) | X _ { t } ] - f ( X _ { t } ) } { h } \leq l ( X _ { t } ) } \end{array}$

Intuitively, this bounds the expected change per step, or the instantaneous drift rate, from above by l whenever the process is in S. The continuous-time setting requires additional regularity assumptions specified in Appendix A.

These defining conditions subsume all classical supermartingale certificates for proving temporal properties in stochastic systems, such as ranking supermartingales (Chakarov and Sankaranarayanan, 2013) for almost-sure reachability, stochastic barrier functions (Prajna et al., 2007) for probabilistic safety, reach-avoid supermartingales (Zikelic et al., 2023a) for probabilistic reach-avoidance, or Streett supermartingales (Abate et al., 2024, 2025) or limit-deterministic Büchi supermartingales (Henzinger et al., 2025d) for general omega-regular specifications. Thus, we follow this unified presentation to design a runtime monitoring framework that applies to all these supermartingale certificates.

Example 1 (A robot climbing a hill). A robot starts at distance $x _ { 0 } = 1 0 $ from its target. Each step moves it forwards by 0.01 or makes it slip backwards by the same distance. It stops upon reaching a distance in [0, 1] from the target, where it remains. A sloping section between positions 3 and 7 increases the probability of slipping backwards. As the robot moves from 7 towards 3, the slope becomes steeper until position 5, then gradually flattens. We describe this steepness profile by $b ( x ) : = ( 1 - | x - 5 | / 2 ) _ { + }$ . Thus, position 5 is the steepest part of the slope. Before reaching the target, the dynamics for $a \in [ 0 , 3 / 4 ]$ and $p _ { a } ( x ) : = 1 / 4 + a \cdot b ( x )$ are

$$
X _ { n + 1 } = X _ { n } - 0 . 0 1 + 0 . 0 2 \cdot Z _ { n } ,
$$

$$
Z _ { n } \mid { \mathcal { G } } _ { n } \sim { \mathrm { B e r n o u l l i } } \left( p _ { a } ( X _ { n } ) \right) .
$$

Each draw uses fresh randomness with slip probability determined by the current state. The range of a ensures that $p _ { a } ( x ) \in [ 0 , 1 ]$ . If we choose $f ( x ) = x$ and $l ( x ) = - 0 . 0 0 5 \cdot { \bf 1 } \{ x > 1 \}$ , the expected decrease before stopping is $d _ { a } ( x ) : = \mathbb { E } [ f ( X _ { n + 1 } ) - f ( X _ { n } ) \mid X _ { n } = x ] = - 0 . 0 0 5 + 0 . 0 2 \cdot a \cdot b ( x )$ If $a = 0 , f$ is a ranking supermartingale proving almost-sure target reachability. However, for every $a > 0$ , the expected-decrease condition is violated on (3, 7).

Trajectory Given a Markov process $\pmb { X } = ( X _ { t } ) _ { t \in \mathbb { T } }$ and a measurable function $f : \mathcal { X }  \mathbb { R }$ the goal of the supermartingale certificate monitoring problem is to construct a monitor which observes states of the Markov process at diferent time steps and, upon each newly observed state, issues a verdict on whether the function $f$ violates one of the supermartingale conditions or not. To that end, we define $\mathcal { V } : = \mathbb { T } \times \mathcal { X }$ to be the set of time–state pairs, and use ${ \mathcal { V } } ^ { * }$ to denote the set of all finite sequences of time-state pairs. We call elements of ${ \mathcal { V } } ^ { * }$ trajectories.

Monitor A monitor is a tuple ${ \mathfrak { M } } : = ( \pi , \xi )$ , where $\pi \colon \mathcal { V } ^ { * } \to \mathbb { T }$ is a sampling rule which for each sampled trajectory specifies the next observation time, and $\xi \colon \mathcal { V } ^ { * } \to \{ 0 , 1 \}$ is a verdict rule which for each sampled trajectory decides whether to raise an alarm or not. The verdict value 1 denotes an alarm, indicating statistical evidence of a certificate violation, while 0 means that no alarm has been raised. The semantics of a monitor are defined inductively as follows. At the beginning, the monitor starts with a single-state trajectory $Y _ { 0 : 0 } = ( T _ { 0 } , x _ { T _ { 0 } } )$ , with $T _ { 0 } = 0$ and $x _ { T _ { 0 } }$ sampled from $X _ { T _ { 0 } }$ . Then, for each n, if $Y _ { 0 : n } = ( Y _ { 0 } , \ldots , Y _ { n } )$ is the trajectory of time-state samples that have been sampled after n steps, the next time-state sample and the monitor verdict are computed via

$$
T _ { n + 1 } : = \pi ( Y _ { 0 : n } ) , \quad x _ { T _ { n + 1 } } \sim X _ { T _ { n + 1 } } , \quad Y _ { 0 : n + 1 } : = ( Y _ { 0 : n } , ( T _ { n + 1 } , x _ { T _ { n + 1 } } ) ) , \quad V _ { n + 1 } : = \xi ( Y _ { 0 : n + 1 } ) .
$$

Hence, the monitor determines the next sampling time $T _ { n + 1 }$ , samples the next state $x _ { T _ { n + 1 } }$ from $X _ { T _ { n + 1 } }$ , appends the new time-state pair to the trajectory to obtain trajectory ${ \cal Y } _ { 0 : n + 1 }$ , and finally computes the verdict $V _ { n + 1 }$ by applying the verdict rule to the trajectory $Y _ { 0 : n + 1 }$

Problem Statement Our goal is to design a monitor that raises an alarm as accurately as possible, while providing probabilistic guarantees that its false alarm rate is below some tolerable threshold δ over all sampling times, and not just for each individual sampled state. The accuracy of our monitor is an empirical claim that will be evaluated in our experiments. However, the false alarm rate is a property that we want to guarantee. We define it as follows. Given a candidate certificate f and an error tolerance $\delta \in ( 0 , 1 )$ , we want to construct a monitor ${ \mathfrak { M } } : = ( \pi , \xi )$ such that, whenever f satisfies the certificate conditions for the deployed process, we have $\mathbb { P } \left( \exists n \in \mathbb { N } _ { 0 } : V _ { n } = 1 \right) \leq \delta$

Challenge To determine whether f is a supermartingale certificate, it sufices to design one monitor for each domain-bound condition and each expected-decrease condition of the supermartingale. But designing a monitor for domain-bound conditions is trivial – if $S \subseteq { \mathcal { X } }$ is the domain and $[ l , u ]$ is the bound interval, the verdict rule is simply $V _ { n } = \xi ( Y _ { 0 : n } ) = 1$ if the last time-state pair $( T _ { n } , x _ { n } )$ satisfies $x _ { n } \in S$ but $f ( x _ { n } ) \not \in [ l , u ]$ . Such monitor design ensures that the false alarm rate is 0.

Hence, the challenge in supermartingale certificate monitoring is to compute a monitor for the expected-decrease condition, and in the rest of this work we focus on that. Specifically, our construction uses fixed observation times $t _ { n } = n h$ and bounds the probability of raising an alarm before any completed observation interval has positive conditional expected corrected drift. Formally, let $\begin{array} { r } { L _ { n } : = \sum _ { r = t _ { n } } ^ { t _ { n + 1 } - 1 } l ( X _ { r } ) } \end{array}$ in discrete time and $\begin{array} { r } { L _ { n } : = \int _ { t _ { n } } ^ { t _ { n + 1 } } l ( X _ { r } ) } \end{array}$ dr in continuous time denote the accumulated allowed drift. Writing $\tau _ { \delta }$ for the first alarm index and $E _ { n } : =$ $\mathbb { E } [ f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - L _ { n } \ | \ { \mathcal { F } } _ { n } ]$ for the conditional expected corrected drift, we prove

$$
\mathbb { P } \left( \tau _ { \delta } < \infty \mathrm { ~ a n d ~ } E _ { n } \leq 0 \mathrm { ~ f o r ~ e v e r y ~ } n < \tau _ { \delta } \right) \leq \delta .
$$

## 3 Runtime monitoring as betting-based sequential testing

We use a betting-based sequential testing framework (Ramdas et al., 2023) as the main underlying ingredient to construct a monitor for supermartingale certificates. In this section, we first show how a sequence of observed states can be transformed into scores that provide evidence of a certificate violation. We then define a general monitoring scheme, namely an abstract monitor, that guarantees false-alarm control for any admissible betting strategy. Lastly, we instantiate the abstract monitor with concrete strategies designed to improve detection power while preserving monitor validity.

A supermartingale $( M _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ is nonnegative if $M _ { n } \geq 0$ almost surely for every $n \geq 0$ . In the betting-based framework, we use $V i l i e ^ { \prime } s$ inequality to turn such a process into a monitor with a bound on the probability of ever raising a false alarm. Ville’s inequality states that, for any nonnegative supermartingale $( M _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ with $M _ { 0 } = 1$ and for every $\delta \in ( 0 , 1 )$ , the probability of the supermartingale value ever reaching or exceeding $1 / \delta$ is at most δ, i.e.,

$$
\mathbb { P } \left( \exists n \in \mathbb { N } _ { 0 } : M _ { n } \geq 1 / \delta \right) \leq \delta .\tag{1}
$$

We set the first alarm index $\tau _ { \delta }$ and the verdict $V _ { n }$ to

$$
\begin{array} { r } { \tau _ { \delta } : = \operatorname* { i n f } \{ n \in { \mathbb { N } } _ { 0 } : M _ { n } \geq 1 / \delta \} , \quad \mathrm { a n d } \quad V _ { n } : = { \bf 1 } \{ \tau _ { \delta } \leq n \} \quad \mathrm { w i t h ~ i n f } \ : \emptyset : = \infty . } \end{array}
$$

This defines a statistical test, such that whenever the null hypothesis ensures that $( M _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ is a nonnegative supermartingale, the probability of ever raising an alarm is at most δ.

## 3.1 Scoring schemes

To apply the betting framework, we assign a numerical score to each pair of consecutive observations. A positive score indicates that the observed certificate change exceeds the allowed drift upper bound. However, a positive score can also occur by chance. We construct the scores so that, whenever the expected-decrease condition holds, the expected next score, given the observations so far, is at most zero. This ensures that betting on these scores cannot increase wealth in conditional expectation while the condition holds. We also ensure that each score is at least −1 almost surely, so that admissible bets cannot make the wealth negative.

Assumptions The monitor observes the Markov process X at times $t _ { n } = n h$ , with fixed intervals $h \in \mathbb { R } _ { > 0 }$ in continuous time and $h = 1$ in discrete time. We write $\mathcal { F } _ { n } : = \sigma ( X _ { t _ { 0 } } , \ldots , X _ { t _ { n } } )$ for the information available to the monitor at $t _ { n }$ . Furthermore, in continuous time, the deployed process is absorbed at its first exit from $s ,$ retaining its terminal state. For example, for a reach–avoid certificate, the system terminates upon reaching the target or an unsafe state.

We assume that the monitor can compute two bounds from its observations. First, after observing a state at $t _ { n + 1 }$ , it can compute a finite upper bound ${ \overline { { L } } } _ { n } \geq L _ { n }$ almost surely on the accumulated allowed drift. Subtracting this bound from the observed certificate change gives the corrected certificate change $D _ { n } : = f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - { \overline { { L } } } _ { n }$ . Second, at $t _ { n }$ and before observing state $X _ { t _ { n + 1 } }$ , the monitor must already be able to compute a finite loss bound $B _ { n } \geq b _ { \mathrm { m i n } }$ , where $b _ { \operatorname* { m i n } } > 0$ is fixed, guaranteeing $D _ { n } \geq - B _ { n }$ almost surely. This ensures that the normalised score $D _ { n } / B _ { n }$ is at least −1. Formally, ${ \overline { { L } } } _ { n }$ is $\mathcal { F } _ { n + 1 } . . . \mathrm { m e a s u r a b l e }$ and $B _ { n }$ is $\mathcal { F } _ { n } { \mathrm { - m e a s u r a b l e } }$ , and these requirements hold for every $n \in  { \mathbb { N } } _ { 0 }$ . Moreover, we assume that the random variables $f ( X _ { t _ { n } } )$ ， $L _ { n }$ , and ${ \overline { { L } } } _ { n }$ are integrable.

Concrete conservative bounds can be obtained if we know a lower bound on the certificate value $f ( x ) \geq f _ { \mathrm { m i n } }$ and an upper bound on the allowed drift $l ( x ) \leq l _ { \mathrm { m a x } }$ for all reachable states. Then we get $\overline { { L } } _ { n } = h l _ { \mathrm { m a x } }$ and $B _ { n } = \operatorname* { m a x } \{ b _ { \operatorname* { m i n } } , f ( X _ { t _ { n } } ) + h l _ { \operatorname* { m a x } } - f _ { \operatorname* { m i n } } \}$

Scoring The monitor compares the observed certificate change with the upper bound on the allowed drift and normalises it by the loss bound. A positive score therefore indicates that the observed certificate change exceeds the drift upper bound. The score is bounded below by −1 and has nonpositive conditional mean whenever the certificate condition holds. Formally, upon observing a state at $t _ { n + 1 }$ , the monitor computes a raw score $D _ { n }$ and the normalised score $W _ { n } { \mathrm { : } }$

$$
D _ { n } : = f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - { \overline { { L } } } _ { n } , \qquad W _ { n } : = D _ { n } / B _ { n } .
$$

Let $\mu _ { n } : = \mathbb { E } [ W _ { n } \mid { \mathcal { F } } _ { n } ]$ denote the expected next score given the observations so far. The following lemma shows that $\mu _ { n } \leq 0$ whenever the supermartingale expected-decrease condition holds. Together with the lower bound $W _ { n } \geq - 1$ ensured by normalisation, this makes the scores valid for the betting construction, where admissible bets preserve nonnegative wealth and cannot increase it in conditional expectation while the certificate condition holds. The proof is deferred to Appendix B.

Lemma 1 (Score validity). Recall that $E _ { n } : = \mathbb { E } [ f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - L _ { n } \mid { \mathcal { F } } _ { n } ]$ is the conditional expected corrected drift over $[ t _ { n } , t _ { n + 1 } ]$ . For every $n \in  { \mathbb { N } } _ { 0 }$

$$
\mu _ { n } = \frac { E _ { n } - \mathbb { E } [ \overline { { L } } _ { n } - L _ { n } \mid \mathcal { F } _ { n } ] } { B _ { n } } \leq \frac { E _ { n } } { B _ { n } } \qquad a l m o s t \ s u r e l y .\tag{2}
$$

Thus, if the expected-decrease condition holds and $E _ { n } ~ \leq ~ 0$ , then $\mu _ { n } ~ \leq ~ 0$ almost surely on $\{ X _ { t _ { n } } \in S \}$

Example 2. Consider the hill robot in Example 1, with observation interval $h = 1$ . Since $f$ is nonnegative on reachable states and $l \leq 0$ , valid bounds are $\overline { { L } } _ { n } = 0$ and $B _ { n } = \operatorname* { m a x } \{ b _ { \mathrm { m i n } } , f ( X _ { t _ { n } } ) \}$ These bounds only assume that the certificate cannot decrease by more than its current value. However, we can use the known system dynamics to derive more informative bounds. The allowed drift is exactly $l ( X _ { t _ { n } } )$ , and a forward step decreases the certificate by only 0.01. Before stopping, the corrected increment is therefore at least $- 0 . 0 1 + 0 . 0 0 5 = - 0 . 0 0 5$ . We can thus choose $\overline { { L } } _ { n } = l ( X _ { t _ { n } } )$ and $B _ { n } = b _ { \mathrm { m i n } } = 0 . 0 2$ , giving

$$
W _ { n } = { \frac { f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) + 0 . 0 0 5 } { 0 . 0 2 } } = Z _ { n } - { \textstyle { \frac { 1 } { 4 } } } , \qquad \mu _ { n } = a \cdot b ( X _ { t _ { n } } ) .
$$

Hence, a forward step results in $W _ { n } = - 1 / 4$ , while a slip gives $W _ { n } = 3 / 4$ , so $W _ { n } \geq - 1$ for every amplitude, which validates the loss bound. After stopping, $W _ { n } = 0$

## 3.2 Runtime monitoring under general betting strategies

We describe how the monitor accumulates evidence from past scores and decides to raise an alarm. Our monitor combines multiple wealth accounts and alarms when its total wealth reaches $1 / \delta$ . The construction here is abstract because we do not yet specify the betting strategies in this subsection.

A single positive score may occur by chance, so the monitor accumulates evidence by betting on successive scores, following Ramdas et al. (2023). At each observation index $s \in  { \mathbb { N } } _ { 0 }$ , we may launch a set of betting strategies $A _ { s } ,$ called experts. For each expert $a \in \mathcal { A } _ { s }$ , let $G _ { s , n } ^ { a }$ denote its normalised wealth at index $n \geq s .$ , starting from $G _ { s , s } ^ { a } = 1$ . The capital allocated to these accounts is specified below. For every $n \geq s .$ , if $X _ { t _ { n } } \in S$ , expert a chooses an ${ \mathcal { F } } _ { n }$ -measurable stake $\lambda _ { s , n } ^ { a } \in [ 0 , 1 ]$ before observing $W _ { n } .$ , and otherwise sets $\lambda _ { s , n } ^ { a } = 0$ . Then the account wealth is updated via:

$$
G _ { s , n + 1 } ^ { a } : = G _ { s , n } ^ { a } ( 1 + \lambda _ { s , n } ^ { a } W _ { n } )
$$

Since $W _ { n } \geq - 1$ , wealth remains nonnegative. The stake controls the gains from positive scores and the losses from negative scores. A good betting strategy can therefore help wealth grow rapidly when the certificate condition fails, allowing the monitor to raise an alarm sooner.

Mixing experts Diferent betting strategies may detect diferent violations more quickly. Since the monitor does not know in advance which strategy will work best, it combines their wealth accounts. Specifically, for each observation index s, the monitor combines the accounts in $A _ { s }$ using fixed, nonnegative, $\mathcal { F } _ { s }$ -measurable weights $\eta _ { s , a }$ satisfying $\textstyle \sum _ { a \in { \mathcal { A } } _ { s } } \eta _ { s , a } = 1$ . The combined wealth is $\begin{array} { r } { Q _ { s , n } : = \sum _ { a \in \mathcal { A } _ { s } } \eta _ { s , a } G _ { s , n } ^ { a } } \end{array}$ with $Q _ { s , s } = 1$

A mixture may lose wealth before a violation begins. To launch fresh mixtures without earlier losses, the monitor reserves part of its initial unit capital. The monitor starts with one unit of capital. Let $c _ { s }$ denote the amount allocated to the mixture launched at observation $s ,$ with ${ c _ { s } = 0 }$ if no mixture is funded. Before launching the mixture at index $n ,$ the monitor has already allocated $\textstyle C _ { n } : = \sum _ { s = 0 } ^ { n - 1 } c _ { s }$ to earlier mixtures, leaving $1 - C _ { n }$ in reserve. It allocates an amount $c _ { n } \in [ 0 , 1 - C _ { n } ]$ from this reserve to the new mixture. This amount may depend on the observations so far $( c _ { n }$ is $\mathcal { F } _ { n } { \mathrm { - m e a s u r a b l e } } )$ . Launching a mixture therefore transfers existing capital rather than adding new capital. The total monitor wealth is the remaining reserve plus the current wealth of all funded mixtures:

$$
M _ { n } : = 1 - C _ { n } + \sum _ { s = 0 } ^ { n - 1 } c _ { s } \sum _ { a \in \mathcal { A } _ { s } } \eta _ { s , a } \prod _ { k = s } ^ { n - 1 } ( 1 + \lambda _ { s , k } ^ { a } W _ { k } ) , \qquad M _ { 0 } = 1 .
$$

Since $Q _ { n , n } = 1$ , funding a new mixture does not change the total wealth. For example, funding every $q \in \mathbb { N } _ { \geq 1 }$ observations can use $c _ { j q } = 6 / ( \pi ^ { 2 } ( j + 1 ) ^ { 2 } )$ for $j \in  { \mathbb { N } } _ { 0 }$ and $c _ { n } = 0$ otherwise. The allocations sum to one and leave reserve $6 / \pi ^ { 2 } \textstyle \sum _ { i = j + 2 } ^ { \infty } i ^ { - 2 } > 0$ so there always remains fresh capital.

Probabilistic guarantees If the certificate condition holds almost surely throughout execution, Lemma 1 gives $\lambda _ { s , n } ^ { a } \mu _ { n } = \lambda _ { s , n } ^ { a } \mathbb { E } [ W _ { n } \mid { \mathcal { F } } _ { n } ] \leq 0$ , and since the current wealth and the stake are determined by the observations available before $W _ { n }$ is observed, we obtain

$$
\mathbb { E } [ G _ { s , n + 1 } ^ { a } \mid \mathcal { F } _ { n } ] = G _ { s , n } ^ { a } ( 1 + \lambda _ { s , n } ^ { a } \mu _ { n } ) \leq G _ { s , n } ^ { a } .
$$

Thus, each expert account is a nonnegative supermartingale for any admissible stake strategy. Combining these accounts using fixed mixture weights preserves this property for $Q _ { s , n }$ . Funding new mixtures from the remaining reserve likewise preserves it for $M _ { n }$ . The threshold in (1) therefore applies to the total wealth regardless of the admissible stakes, mixture weights, and funding chosen.

Theorem 2. For every admissible choice of stakes, expert weights, and funding, the monitor satisfies

$$
\mathbb { P } \left( \tau _ { \delta } < \infty \ a n d \ E _ { k } \leq 0 \ f o r \ e v e r y \ k < \tau _ { \delta } \right) \leq \delta ,\tag{3}
$$

and, if the expected-decrease condition holds for the deployed process, then $\mathbb { P } ( \tau _ { \delta } < \infty ) \le \delta$

Inequality (3) bounds the probability of raising an alarm before the expected-decrease condition is violated, thus bounding the false-positive rate of the monitor, as desired. The proof is in Appendix B.

Detection power We also bound the probability that the monitor raises an alarm within a block of observations during which the supermartingale expected-decrease condition is persistently violated. Consider observation intervals indexed by $k = s , \ldots , s + N - 1$ , and suppose that there exists $\varepsilon \in ( 0 , 1 ]$ such that $\mathbb { E } [ W _ { k } \mid { \mathcal { F } } _ { k } ] \geq \varepsilon$ and $W _ { k } \le 1$ almost surely, for each k. Suppose that a betting strategy a, launched at index $r \leq s$ , has capital $\rho : = c _ { r } \eta _ { r , a } G _ { r , s } ^ { a } > 0$ at observation s and uses the constant stake $\varepsilon / 2$ over these intervals. Then, for any $\gamma \in ( 0 , 1 )$ ,

$$
\mathbb { P } \left( \tau _ { \delta } \le s + N \mid \mathcal { F } _ { s } \right) \ge 1 - \gamma \quad \mathrm { a l m o s t ~ s u r e l y ~ o n ~ } \left\{ N \ge \frac { 8 } { \varepsilon ^ { 2 } } \log \frac { 1 } { \delta \rho \gamma } \right\} .
$$

Thus, if the violation persists for suficiently many observation intervals, the monitor raises an alarm by observation $s + N$ with conditional probability at least $1 - \gamma$ . The parameter ε is a lower bound on the expected score at each observation. A larger value of ε reduces the number of observation intervals required to detect violation. A larger starting capital $\rho$ also reduces this number, because less wealth growth is needed to reach the alarm threshold $1 / \delta$

## 3.3 Weighted prediction strategies

We now give two concrete strategies for choosing the stake by predicting the next score from past scores. The better the monitor predicts the next score, the faster it can grow its capital. The prediction is based on the fact that many realistic systems exhibit regularity in space and time. Here, spatial regularity means that nearby states have similar conditional expected scores, whereas temporal regularity means that these expectations change slowly over time. We consider a mixture over several experts, where each expert guesses and weights past scores according to their spatial and temporal proximity to the current observation. If the guesses are accurate, this weighting can improve our estimate of the conditional mean of the next score.

Prediction weights For computational eficiency, we limit ourselves to a finite history, i.e., let $m \geq 1$ be the number of scores retained in the history, and define the retained indices $I _ { n } : = \{ i \in \mathbb { N } _ { 0 } : \operatorname* { m a x } \{ 0 , n - m \} \le i < n \}$ . Let $a \in \mathcal { A } _ { s }$ be an expert launched at observation index s. The expert chooses a spatial scale $r _ { \mathcal { X } , s } ^ { a }$ and a temporal scale $r _ { \mathbb { T } , s } ^ { a } ,$ each in $( 0 , \infty ]$ . We describe how the scales are obtained in Appendix C. The scales serve as input to a spatial kernel $k _ { \mathcal { X } , s } ^ { a }$ and a temporal kernel $k _ { \mathbb { T } , s } ^ { a } ,$ respectively. Both kernels are nonincreasing, take values in [0, 1], and equal one at zero. The kernels and scales are fixed when the expert is launched. At each observation index $n \geq s ,$ a assigns each retained score $W _ { i } , i \in I _ { n } .$ , a prediction weight based on its starting time $t _ { i }$ and starting state $X _ { t _ { i } }$

$$
\omega _ { s , n , i } ^ { a } : = k _ { \mathbb { T } , s } ^ { a } \left( \frac { t _ { n } - t _ { i } } { r _ { \mathbb { T } , s } ^ { a } } \right) k _ { \mathcal { X } , s } ^ { a } \left( \frac { \Vert X _ { t _ { n } } - X _ { t _ { i } } \Vert } { r _ { \mathcal { X } , s } ^ { a } } \right) , \qquad i \in I _ { n } .\tag{4}
$$

Intuitively, recent observations and observations starting near the current state therefore receive greater weight. An infinite scale removes the corresponding dependence on distance, using the convention $u / \infty = 0 , \mathrm { e . g . }$ , temporal-only experts use $r _ { \mathcal { X } , s } ^ { a } = \infty$ . Setting both scales to infinity gives uniform weights over the retained history.

Betting strategies Next, we transform past scores and their prediction weights into stakes, using two concrete strategies when $X _ { t _ { n } } \in S$ . A stake λ multiplies the wealth by $1 + \lambda W _ { n }$ , so its log return is log $\left( 1 + \lambda W _ { n } \right)$ . Log returns add across successive bets and therefore measure accumulated multiplicative wealth growth. The log-return strategy maximises the weighted historical log return:

$$
\lambda _ { s , n } ^ { a , \log } : = \operatorname* { m i n } \underset { 0 \leq \lambda \leq \overline { { \lambda } } } { \mathrm { m a x } } \sum _ { i \in I _ { n } } \omega _ { s , n , i } ^ { a } \log ( 1 + \lambda W _ { i } ) \quad \mathrm { f o r ~ } 1 / 2 \leq \overline { { \lambda } } < 1\tag{5}
$$

A second strategy is the moment strategy, which maximises a quadratic lower bound on the weighted historical log return. $\mathrm { S i n c e ~ l o g } ( 1 + \lambda w ) \ge \lambda w - \lambda ^ { 2 } w ^ { 2 }$ for w $\geq - 1$ and $\lambda \in [ 0 , 1 / 2 ]$ , this bound is $\lambda S _ { s , n } ^ { a } - \lambda ^ { 2 } U _ { s , n } ^ { a } ,$ where $\begin{array} { r } { S _ { s , n } ^ { a } : = \sum _ { i \in I _ { n } } \omega _ { s , n , i } ^ { a } W _ { i } } \end{array}$ and $\begin{array} { r } { U _ { s , n } ^ { a } : = \sum _ { i \in I _ { n } } \omega _ { s , n , i } ^ { a } W _ { i } ^ { 2 } } \end{array}$ . Its maximiser is

$$
\lambda _ { s , n } ^ { a , \mathrm { m o m } } : = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \frac { ( S _ { s , n } ^ { a } ) _ { + } } { 2 U _ { s , n } ^ { a } } \right\} , \qquad ( 0 / 0 : = 0 ) .\tag{6}
$$

Thus positive weighted scores favour betting, whereas large score magnitudes reduce the stake.

Both strategies choose zero when the weighted sum of the retained scores is nonpositive, including for an empty history. The moment strategy requires only two sums and bets no more than the log-return strategy for the same weights. At each launch, the monitor can include a range of experts A with diferent configurations of scales, kernels and strategies, to accommodate diferent violation patterns. Their accounts are then combined using the capital weights $\eta _ { s , a }$ from Section 3.2.

## 4 Experimental Evaluation

We evaluate on a wide range of benchmarks whether our monitoring framework can efectively detect violations of supermartingale certificate conditions. We aim to address the following research questions. Can the monitors detect certificate violations under changed system dynamics (RQ1)? How do temporal and spatial scaling afect the power of monitoring (RQ2)?

Benchmarks We consider three benchmark suites. The synthetic suite consists of six families that isolate persistent, delayed, temporary, and recurring local violations. The control suite comprises three systems which are variants of Building and Tanks from PRoTECT (Wooding et al., 2026), together with an LQR system. Their expected certificate increments can be computed exactly, providing ground truth for evaluating detection. The neural suite uses 16 learned controller–certificate pairs from LogRASM (Badings et al., 2025) to assess monitoring under modified deployment dynamics.

Monitors We use three prediction schemes: uniform (U), which weights retained scores equally; temporal (T), which favours recent observations; and spatial (S), which favours observations from nearby states. On the synthetic benchmarks, we compare moment (M) and log-return (L) strategies. The control and neural experiments use log-return and equal mixtures over five scales for T and S. Each monitor uses a false-alarm tolerance of δ = 0.05 and raises an alarm when its betting wealth reaches 1/δ = 20. Full configurations and implementation details are provided in the Appendix.

RQ1: Detection of certificate violations. Our results show that the monitors detect certificate violations under changed dynamics, with spatial prediction being particularly efective when violations are localised in the state space. Table 1 illustrates how violation magnitude and exposure afect detection. In Building, S-L detects every run, and its median delay falls from 3043 to 732 observations as violations become stronger and more frequent. The weaker, less frequent violations in Tanks remain harder to detect, even with spatial prediction. For LQR, we weaken the controller by reducing the applied control input. With a 5% reduction, violations occur throughout each trajectory, yet each monitor detects fewer than half the runs. With a 20% reduction, all monitors detect every run. Overall this shows that even persistent violations can remain undetected when their efect is too small to distinguish reliably from stochastic fluctuations within the observed trajectory.

Table 3 summarises the mean eligible first-alarm percentages across a number of disturbed settings. On the control benchmarks, S-L achieves 83.8%, compared with 29.7% for U-L and 28.2% for T-L. Specifically in Building and Tanks, the certificate condition fails in some regions even though it holds on average along the trajectory. This suggests spatial prediction helps detect these local failures by emphasising observations from nearby states. The neural benchmarks show a similar advantage under expected nonincrease (ϵ = 0), where S-L raises alarms on 43.1% of runs, compared with approximately 11% for either baseline. The higher alarm rates for positive ϵ concern stronger decrease requirements tested on the same trajectories.

RQ2: Temporal vs. spatial scaling impact Our results show that uniform scaling, while suficient for long persistent violations, struggles on traces with transient violations. Both temporal and spatial weighting are able to detect such behaviour. Spatial weighting, in particular, benefits from recurrent observations. The correct scaling depends on the system, i.e., no uniformly best scaling is observed. However, mixing over scales can retain some of the benefits of the respective scales, being often only slightly worse than the best for the respective benchmark. Table 2 summarises the results. We can observe that the uniform monitor has a median detection delay of 2126 on persistent violations, on par with large spatial and temporal scales. By contrast, for temporary and recurrent violations (Rec. and Ring) its median detection delay is either larger or fewer than 50% of the runs detect a fault. For example, on the spatial recurring ring benchmark, the medium spatial monitor has a delay of 640, while the uniform monitor has a delay of 2360; on the temporary fault benchmark only the medium temporal monitor alarmed among the individual scales in more than 50% of the cases.

Table 1: Violation strength and detection in selected control settings, with 50 runs each. For each run, magnitude is the average of $\mu _ { n }$ over violating observations, and exposure is the percentage of violating observations. Magnitude and exposure are reported as medians over the 50 runs. Detection entries give counts, with median delay in parentheses. Detection delay is measured in observations from violation onset or first entry into the violating region. U-L, T-L, and S-L use uniform, temporal, and spatial prediction, respectively. NR means fewer than 25 runs alarm.
<table><tr><td rowspan="3">Benchmark &amp; Setting</td><td rowspan="3">Magnitude</td><td rowspan="3">| Exposure (%)</td><td colspan="3">Detections (median delay)</td></tr><tr><td>U-L</td><td>T-L</td><td>S-L</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Building: 8% heating loss Building: 12% heating loss</td><td>0.0151 0.0309</td><td>24.9 37.6</td><td>0 (NR) 0 (NR)</td><td>0 (NR) 0 (NR)</td><td>50 (3043) 50 (732)</td></tr><tr><td>Tanks: burst probability 0.35</td><td>0.00561</td><td>5.5</td><td>0 (NR)</td><td>0 (NR)</td><td>0 (NR)</td></tr><tr><td>Tanks: burst probability 0.45</td><td>0.00615</td><td>11.0</td><td>0 (NR)</td><td>0 (NR)</td><td>34 (28507)</td></tr><tr><td>LQR: 5% feedback loss</td><td>0.00955</td><td>100.0</td><td>19 (NR)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>12 (NR)</td><td>16 (NR)</td></tr><tr><td>LQR: 20% feedback loss</td><td>0.0383</td><td>100.0</td><td>50 (2241)</td><td>50 (1349)</td><td>50 (2264)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Median detection delay in observations on the six synthetic benchmark families: persistent, delayed, temporary, direct recurrence, stochastic ring, and hill. XS–XL denote extrasmall through extra-large prediction scales (exact parameters given in Appendix). Temporal scales determine how far back in the history observations receive weight; spatial scales determine the neighbourhood radius around the current state. Equal divides capital equally among the five scale experts. Uniform weights all retained observations equally. NR means fewer than 25 runs alarm.
<table><tr><td colspan="7">Temporal prediction</td><td colspan="7">Spatial prediction</td></tr><tr><td>Scale</td><td>Pers.</td><td></td><td></td><td></td><td>Del. Temp. Rec. Ring</td><td>Hill</td><td>|Scale</td><td>Pers.</td><td></td><td>Del. Temp.</td><td>Rec. Ring</td><td></td><td>Hill</td></tr><tr><td>Uniform Equal</td><td>21263204</td><td></td><td>NR</td><td>NR</td><td>2360</td><td>588</td><td>Uniform Equal</td><td>21263204 3351 2141</td><td></td><td>NR</td><td>NR 2360</td><td></td><td>588</td></tr><tr><td>XS</td><td>2183</td><td>668</td><td>914</td><td>1102</td><td>664</td><td>492</td><td>|XS</td><td></td><td></td><td>NR</td><td>571</td><td>654</td><td>482</td></tr><tr><td>S</td><td></td><td>NR 1565</td><td>NR</td><td>570 2110</td><td></td><td>168 568 1127441</td><td>S</td><td>NR 2977</td><td>NR 3979</td><td>NR NR</td><td>NR</td><td>NR</td><td>670</td></tr><tr><td>M</td><td>6719 2442</td><td>756</td><td>NR</td><td></td><td></td><td></td><td>M</td><td>21051 3437</td><td></td><td></td><td>NR</td><td>655 594</td><td></td></tr><tr><td>L</td><td></td><td>575</td><td>766</td><td>NR</td><td>1713 476</td><td>NR 2338 560</td><td>L</td><td>74062275</td><td></td><td></td><td>NR 1578</td><td>640</td><td>552</td></tr><tr><td></td><td>21181174</td><td></td><td>NR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>NR 1040</td><td>1652</td><td>447</td></tr><tr><td>XL</td><td>2127 2439</td><td></td><td>NR</td><td></td><td>NR 2347 587</td><td></td><td>XL</td><td>21902379</td><td></td><td>NR 1576 2153 472</td><td></td><td></td><td></td></tr></table>

## 5 Related work

Supermartingale certificates have found broad applications in probabilistic program verification and stochastic (neural) control (Badings et al., 2025; Zikelic et al., 2023b; Chatterjee et al., 2017, 2022; Takisaka et al., 2021; Majumdar and Sathiyanarayana, 2025; Kura and Unno, 2026; Prajna et al., 2007; Ansaripour et al., 2023; Xue et al., 2024; Abate et al., 2024; Henzinger et al., 2025d; Abate et al., 2026). Our work is closest to recent approaches for runtime monitoring of neural Lyapunov certificates in deterministic systems (Yu et al., 2025; Henzinger et al., 2025c). Runtime monitoring is comparatively straightforward in the deterministic setting: a violation of the required decrease condition is directly observable from a single state transition. For supermartingale certificates, however, monitoring is fundamentally harder because the required decrease holds only in expectation and therefore cannot be assessed from a single transition. We overcome this challenge using betting-based sequential testing (Ramdas et al., 2023; Waudby-Smith and Ramdas, 2024; Timans et al., 2025), which we apply for the first time to supermartingale certificates for temporal properties. We further exploit temporal and spatial regularity in stochastic systems to make informed bets and improve detection power.

Table 3: Percentage of runs in which each monitor raises an alarm, averaged across settings (50 runs per setting). For neural, ϵ = 0 tests expected nonincrease; larger ϵ requires faster expected decrease.
<table><tr><td>Benchmark set</td><td>| No. Settings | U-L(%)</td><td></td><td> $\mathrm { { T } \mathrm { { - } } \mathrm { { L } } ( \% ) }$ </td><td>| S-L(%)</td></tr><tr><td>Synthetic</td><td>18</td><td>62.2</td><td>77.0</td><td>79.6</td></tr><tr><td>Control</td><td>18</td><td>29.7</td><td>28.2</td><td>83.8</td></tr><tr><td>Neural, € = 0</td><td>48</td><td>10.7</td><td>10.8</td><td>43.1</td></tr><tr><td>Neural, € = 0.0005</td><td>48</td><td>71.9</td><td>71.7</td><td>77.8</td></tr><tr><td>Neural,  $\epsilon = 0 . 0 0 1$ </td><td>48</td><td>76.1</td><td>76.3</td><td>81.0</td></tr></table>

Runtime monitoring is an established technique for ensuring correctness of cyber-physical systems (Bartocci et al., 2018; Bauer et al., 2011), providing a lightweight, black-box alternative to formal verification. Existing methods typically detect property violations directly from observed states, fundamentally restricting monitorable properties to safety (Alpern and Schneider, 1985; Pnueli and Zaks, 2006) and excluding properties such as reachability, stability, and general ω-regular properties. Predictive monitoring (Bortolussi et al., 2019; Lindemann et al., 2023; Henzinger et al., 2025a) extends this scope by forecasting future states, but remains limited to finite time horizons. In contrast, by monitoring supermartingale certificates, our approach can detect violations of certificate conditions supporting general temporal properties over unbounded horizons. Related work (Hindy et al., 2024; Henzinger et al., 2025b) uses martingale theory to detect distribution shifts or model misalignment in stochastic processes, but does not monitor the validity of certificates.

## 6 Conclusion

We introduced the first framework for runtime monitoring of supermartingale certificates for temporal properties that is applicable to both discrete- and continuous-time systems and to a broad range of supermartingale certificates. Our framework bounds the probability of ever raising a false alarm by a given threshold. Experimental evaluation demonstrates the practical efectiveness of our method for detecting supermartingale certificate violations at runtime in a range of stochastic control tasks. Interesting directions of future work include designing shielding and repair procedures that would use the alarms raised by our runtime monitoring framework.

## AI use statement

We used ChatGPT to assist with writing, literature search, theoretical development, and the design and implementation of experiments. We describe these uses below. For writing, we used these tools to draft and revise text, simplify explanations, shorten sections, and improve consistency in notation and terminology. We also used them to assist with LaTeX formatting, e.g., constructing tables. For literature search, we used these tools to identify relevant work and compare its assumptions and guarantees with ours. For theoretical development, we used these tools to suggest proof arguments, expand derivations, and identify possible gaps and counterexamples. They also helped clarify certain measure-theoretic arguments required in the continuous-time setting. For experiments, we used these tools to write basic code and generate plots. They supported the selection of system disturbances. They were used to draft the experimental setup in the appendix. The authors take full responsibility for the final content of the paper, including all AI-assisted text, mathematical claims and proofs, references, code, and experimental results.

## References

Alessandro Abate, Mirco Giacobbe, and Diptarko Roy. Learning probabilistic termination proofs. In Alexandra Silva and K. Rustan M. Leino, editors, Computer Aided Verification - 33rd International Conference, CAV 2021, Virtual Event, July 20-23, 2021, Proceedings, Part II, volume 12760 of Lecture Notes in Computer Science, pages 3–26. Springer, 2021. doi: 10.1007/978-3-030-81688-9\_1. URL https://doi.org/10.1007/978-3-030-81688-9\_1.

Alessandro Abate, Mirco Giacobbe, and Diptarko Roy. Stochastic omega-regular verification and control with supermartingales. In Arie Gurfinkel and Vijay Ganesh, editors, Computer Aided Verification - 36th International Conference, CAV 2024, Montreal, QC, Canada, July 24-27, 2024, Proceedings, Part III, volume 14683 of Lecture Notes in Computer Science, pages 395–419. Springer, 2024. doi: 10.1007/978-3-031-65633-0\_18. URL https://doi.org/10. 1007/978-3-031-65633-0\_18.

Alessandro Abate, Mirco Giacobbe, and Diptarko Roy. Quantitative supermartingale certificates. In Ruzica Piskac and Zvonimir Rakamaric, editors, Computer Aided Verification - 37th International Conference, CAV 2025, Zagreb, Croatia, July 23-25, 2025, Proceedings, Part II, volume 15932 of Lecture Notes in Computer Science, pages 3–28. Springer, 2025. doi: 10.1007/978-3-031-98679-6\_1. URL https://doi.org/10.1007/978-3-031-98679-6\_1.

Alessandro Abate, Mirco Giacobbe, Sergey Ichtchenko, and Diptarko Roy. Complete ω-regular supermartingale certificates. In Claudia Faggian and Joost-Pieter Katoen, editors, 41st Annual Symposium on Logic in Computer Science, LICS 2026, Lisbon, Portugal, July 20-23, 2026, volume 380 of LIPIcs, pages 3:1–3:28. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2026. doi: 10.4230/LIPICS.LICS.2026.3. URL https://doi.org/10.4230/LIPIcs.LICS.2026.3.

Bowen Alpern and Fred B. Schneider. Defining liveness. Inf. Process. Lett., 21(4):181–185, 1985. doi: 10.1016/0020-0190(85)90056-0. URL https://doi.org/10.1016/0020-0190(85) 90056-0.

Matin Ansaripour, Krishnendu Chatterjee, Thomas A. Henzinger, Mathias Lechner, and Dorde Zikelic. Learning provably stabilizing neural controllers for discrete-time stochastic systems. In Automated Technology for Verification and Analysis - 21st International Symposium, ATVA 2023, Singapore, October 24-27, 2023, Proceedings, Part I, pages 357–379, 2023. doi: 10.1007/978-3-031-45329-8\_17. URL https://doi.org/10.1007/978-3-031-45329-8\_17.

Thom Badings, Wietze Koops, Sebastian Junges, and Nils Jansen. Policy verification in stochastic dynamical systems using logarithmic neural certificates. In Ruzica Piskac and Zvonimir Rakamaric, editors, Computer Aided Verification - 37th International Conference, CAV 2025, Zagreb, Croatia, July 23-25, 2025, Proceedings, Part II, volume 15932 of Lecture Notes in Computer Science, pages 349–375. Springer, 2025. doi: 10.1007/978-3-031-98679-6\_16. URL https://doi.org/10.1007/978-3-031-98679-6\_16.

Ezio Bartocci, Jyotirmoy V. Deshmukh, Alexandre Donzé, Georgios Fainekos, Oded Maler, Dejan Nickovic, and Sriram Sankaranarayanan. Specification-based monitoring of cyber-physical systems: A survey on theory, tools and applications. In Ezio Bartocci and Yliès Falcone, editors, Lectures on Runtime Verification - Introductory and Advanced Topics, volume 10457 of Lecture Notes in Computer Science, pages 135–175. Springer, 2018. doi: 10.1007/978-3-319-75632-5\_5. URL https://doi.org/10.1007/978-3-319-75632-5\_5.

Andreas Bauer, Martin Leucker, and Christian Schallhart. Runtime verification for LTL and TLTL. ACM Trans. Softw. Eng. Methodol., 20(4):14:1–14:64, 2011. doi: 10.1145/2000799. 2000800. URL https://doi.org/10.1145/2000799.2000800.

Luca Bortolussi, Francesca Cairoli, Nicola Paoletti, Scott A. Smolka, and Scott D. Stoller. Neural predictive monitoring. In Bernd Finkbeiner and Leonardo Mariani, editors, Runtime Verification - 19th International Conference, RV 2019, Porto, Portugal, October 8-11, 2019, Proceedings, volume 11757 of Lecture Notes in Computer Science, pages 129–147. Springer, 2019. doi: 10.1007/978-3-030-32079-9\_8. URL https://doi.org/10.1007/978-3-030-32079-9\_ 8.

Aleksandar Chakarov and Sriram Sankaranarayanan. Probabilistic program analysis with martingales. In Natasha Sharygina and Helmut Veith, editors, Computer Aided Verification - 25th International Conference, CAV 2013, Saint Petersburg, Russia, July 13-19, 2013. Proceedings, volume 8044 of Lecture Notes in Computer Science, pages 511–526. Springer, 2013. doi: 10.1007/978-3-642-39799-8\_34. URL https://doi.org/10.1007/978-3-642-39799-8\_34.

Krishnendu Chatterjee, Hongfei Fu, and Amir Kafshdar Goharshady. Termination analysis of probabilistic programs through positivstellensatz’s. In CAV (1), volume 9779 of Lecture Notes in Computer Science, pages 3–22. Springer, 2016.

Krishnendu Chatterjee, Petr Novotný, and Dorde Zikelic. Stochastic invariants for probabilistic termination. In Giuseppe Castagna and Andrew D. Gordon, editors, Proceedings of the 44th ACM SIGPLAN Symposium on Principles of Programming Languages, POPL 2017, Paris, France, January 18-20, 2017, pages 145–160. ACM, 2017. doi: 10.1145/3009837.3009873. URL https://doi.org/10.1145/3009837.3009873.

Krishnendu Chatterjee, Amir Kafshdar Goharshady, Tobias Meggendorfer, and Dorde Zikelic. Sound and complete certificates for quantitative termination analysis of probabilistic programs. In Sharon Shoham and Yakir Vizel, editors, Computer Aided Verification - 34th International Conference, CAV 2022, Haifa, Israel, August 7-10, 2022, Proceedings, Part I, volume 13371 of Lecture Notes in Computer Science, pages 55–78. Springer, 2022. doi: 10.1007/978-3-031-13185-1\_4. URL https://doi.org/10.1007/978-3-031-13185-1\_4.

Robert W. Floyd. Assigning meanings to programs. Proceedings of Symposium on Applied Mathematics, 19:19–32, 1967. URL http://laser.cs.umass.edu/courses/cs521-621.Spr06/ papers/Floyd.pdf.

Thomas A. Henzinger, Fabian Kresse, Kaushik Mallik, Emily Yu, and Djordje Zikelic. Predictive monitoring of black-box dynamical systems. In Necmiye Ozay, Laura Balzano, Dimitra Panagou, and Alessandro Abate, editors, 7th Annual Learning for Dynamics & Control Conference, Ann

Arbor, MI, USA, 4-6 June 2025, volume 283 of Proceedings ofMachine Learning Research, pages 804–816. PMLR, 2025a. URL https://proceedings.mlr.press/v283/henzinger25a.html.

Thomas A. Henzinger, Konstantin Kuefner, Vasu Singh, and I. Sun. Alignment monitoring. In Runtime Verification: 25th International Conference, RV 2025, Graz, Austria, September 15–19, 2025, Proceedings, page 140–159, Berlin, Heidelberg, 2025b. Springer-Verlag. ISBN 978-3-032-05434-0. doi: 10.1007/978-3-032-05435-7\_9. URL https://doi.org/10.1007/ 978-3-032-05435-7\_9.

Thomas A. Henzinger, Konstantin Kuefner, and Emily Yu. Formal verification of neural certificates done dynamically. In Bettina Könighofer and Hazem Torfah, editors, Runtime Verification - 25th International Conference, RV 2025, Graz, Austria, September 15-19, 2025, Proceedings, volume 16087 of Lecture Notes in Computer Science, pages 54–72. Springer, 2025c. doi: 10.1007/978-3-032-05435-7\_4. URL https://doi.org/10.1007/978-3-032-05435-7\_ 4.

Thomas A. Henzinger, Kaushik Mallik, Pouya Sadeghi, and Dorde Zikelic. Supermartingale certificates for quantitative omega-regular verification and control. In Ruzica Piskac and Zvonimir Rakamaric, editors, Computer Aided Verification - 37th International Conference, CAV 2025, Zagreb, Croatia, July 23-25, 2025, Proceedings, Part II, volume 15932 of Lecture Notes in Computer Science, pages 29–55. Springer, 2025d. doi: 10.1007/978-3-031-98679-6\_2. URL https://doi.org/10.1007/978-3-031-98679-6\_2.

Ali Hindy, Rachel Luo, Somrita Banerjee, Jonathan Kuck, Edward Schmerling, and Marco Pavone. Diagnostic runtime monitoring with martingales. CoRR, abs/2407.21748, 2024. doi: 10.48550/ARXIV.2407.21748. URL https://doi.org/10.48550/arXiv.2407.21748.

Francis Hirsch and Marc Yor. On temporally completely monotone functions for Markov processes. Probability Surveys, 9(none):253 – 286, 2012. doi: 10.1214/11-PS179. URL https://doi.org/10.1214/11-PS179.

Nick Jakobi, Phil Husbands, and Inman Harvey. Noise and the reality gap: The use of simulation in evolutionary robotics. In Federico Morán, Alvaro Moreno, Juan Julián Merelo Guervós, and Pablo Chacón, editors, Advances in Artificial Life, Third European Conference on Artificial Life, Granada, Spain, June 4-6, 1995, Proceedings, volume 929 of Lecture Notes in Computer Science, pages 704–720. Springer, 1995. doi: 10.1007/3-540-59496-5\_337. URL https://doi.org/10.1007/3-540-59496-5\_337.

Satoshi Kura and Hiroshi Unno. A hierarchy of supermartingales for ω-regular verification. Proc. ACM Program. Lang., 10(PLDI):253–276, 2026. doi: 10.1145/3808257. URL https: //doi.org/10.1145/3808257.

Harold J Kushner. On the stability of stochastic dynamical systems. Proceedings of the National Academy of Sciences, 53(1):8–12, 1965.

Mathias Lechner, Dorde Zikelic, Krishnendu Chatterjee, and Thomas A. Henzinger. Stability verification in stochastic control systems via neural network supermartingales. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 7326–7336. AAAI Press, 2022. doi: 10.1609/AAAI.V36I7.20695. URL https://doi.org/10.1609/aaai.v36i7.20695.

Lars Lindemann, Xin Qin, Jyotirmoy V. Deshmukh, and George J. Pappas. Conformal prediction for STL runtime verification. In Sayan Mitra, Nalini Venkatasubramanian, Abhishek Dubey,

Lu Feng, Mahsa Ghasemi, and Jonathan Sprinkle, editors, Proceedings of the ACM/IEEE 14th International Conference on Cyber-Physical Systems, ICCPS 2023, (with CPS-IoT Week 2023), San Antonio, TX, USA, May 9-12, 2023, pages 142–153. ACM, 2023. doi: 10.1145/3576841.3585927. URL https://doi.org/10.1145/3576841.3585927.

Rupak Majumdar and V. R. Sathiyanarayana. Sound and complete proof rules for probabilistic termination. Proc. ACM Program. Lang., 9(POPL):1871–1902, 2025. doi: 10.1145/3704899. URL https://doi.org/10.1145/3704899.

Grigory Neustroev, Mirco Giacobbe, and Anna Lukina. Neural continuous-time supermartingale certificates. In Toby Walsh, Julie Shah, and Zico Kolter, editors, Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2025, Philadelphia, PA, USA, February 25 - March 4, 2025, pages 27538–27546. AAAI Press, 2025. doi: 10.1609/AAAI.V39I26.34966. URL https://doi.org/10.1609/aaai.v39i26. 34966.

Amir Pnueli and Aleksandr Zaks. PSL model checking and run-time verification via testers. In Jayadev Misra, Tobias Nipkow, and Emil Sekerinski, editors, FM 2006: Formal Methods, 14th International Symposium on Formal Methods, Hamilton, Canada, August 21-27, 2006, Proceedings, volume 4085 of Lecture Notes in Computer Science, pages 573–586. Springer, 2006. doi: 10.1007/11813040\_38. URL https://doi.org/10.1007/11813040\_38.

Stephen Prajna and Ali Jadbabaie. Safety verification of hybrid systems using barrier certificates. In Rajeev Alur and George J. Pappas, editors, Hybrid Systems: Computation and Control, 7th International Workshop, HSCC 2004, Philadelphia, PA, USA, March 25-27, 2004, Proceedings, volume 2993 of Lecture Notes in Computer Science, pages 477–492. Springer, 2004. doi: 10.1007/978-3-540-24743-2\_32. URL https://doi.org/10.1007/978-3-540-24743-2\_32.

Stephen Prajna, Ali Jadbabaie, and George J. Pappas. A framework for worst-case and stochastic safety verification using barrier certificates. IEEE Trans. Autom. Control., 52(8):1415–1428, 2007. doi: 10.1109/TAC.2007.902736. URL https://doi.org/10.1109/TAC.2007.902736.

Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576 –601, November 2023. ISSN 0883-4237. doi: 10.1214/23-STS894.

Toru Takisaka, Yuichiro Oyabu, Natsuki Urabe, and Ichiro Hasuo. Ranking and repulsing supermartingales for reachability in randomized programs. ACM Trans. Program. Lang. Syst., 43(2):5:1–5:46, 2021. doi: 10.1145/3450967. URL https://doi.org/10.1145/3450967.

Alexander Timans, Rajeev Verma, Eric T. Nalisnick, and Christian A. Naesseth. On continuous monitoring of risk violations under unknown shift. In Silvia Chiappa and Sara Magliacane, editors, Conference on Uncertainty in Artificial Intelligence, Rio Othon Palace, Rio de Janeiro, Brazil, 21-25 July 2025, volume 286 of Proceedings of Machine Learning Research, pages 4204–4226. PMLR, 2025. URL https://proceedings.mlr.press/v286/timans25a.html.

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2017, Vancouver, BC, Canada, September 24-28, 2017, pages 23–30. IEEE, 2017. doi: 10.1109/IROS.2017.8202133. URL https://doi.org/10.1109/IROS.2017.8202133.

Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1):1–27,

02 2024. ISSN 1369-7412. doi: 10.1093/jrsssb/qkad009. URL https://doi.org/10.1093/ jrsssb/qkad009.

Ben Wooding, Viacheslav Horbanov, and Abolfazl Lavaei. Protect: Parallelized construction of safety barrier certificates for nonlinear polynomial systems. In Zhiming Liu, Adnane Saoud, and Heike Wehrheim, editors, Theoretical Aspects of Computing – ICTAC 2025, pages 448–458, Cham, 2026. Springer Nature Switzerland. ISBN 978-3-032-11176-0.

Bai Xue, Naijun Zhan, and Martin Fränzle. Reach-avoid analysis for polynomial stochastic diferential equations. IEEE Trans. Autom. Control., 69(3):1882–1889, 2024. doi: 10.1109/ TAC.2023.3332570. URL https://doi.org/10.1109/TAC.2023.3332570.

Emily Yu, Dorde Zikelic, and Thomas A. Henzinger. Neural control and certificate repair via runtime monitoring. In Toby Walsh, Julie Shah, and Zico Kolter, editors, Thirty-Ninth AAAI Conference on Artificial Intelligence, Thirty-Seventh Conference on Innovative Applications of Artificial Intelligence, Fifteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2025, Philadelphia, PA, USA, February 25 - March 4, 2025, pages 26409–26417. AAAI Press, 2025. doi: 10.1609/AAAI.V39I25.34840. URL https://doi.org/10.1609/aaai. v39i25.34840.

Wenshuai Zhao, Jorge Peña Queralta, and Tomi Westerlund. Sim-to-real transfer in deep reinforcement learning for robotics: a survey. In 2020 IEEE Symposium Series on Computational Intelligence, SSCI 2020, Canberra, Australia, December 1-4, 2020, pages 737–744. IEEE, 2020. doi: 10.1109/SSCI47803.2020.9308468. URL https://doi.org/10.1109/SSCI47803.2020. 9308468.

Dorde Zikelic, Mathias Lechner, Thomas A. Henzinger, and Krishnendu Chatterjee. Learning control policies for stochastic systems with reach-avoid guarantees. In Brian Williams, Yiling Chen, and Jennifer Neville, editors, Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 11926–11935. AAAI Press, 2023a. doi: 10.1609/AAAI.V37I10.26407. URL https://doi.org/10.1609/aaai.v37i10.26407.

Dorde Zikelic, Mathias Lechner, Abhinav Verma, Krishnendu Chatterjee, and Thomas A. Henzinger. Compositional policy learning in stochastic control systems with formal guarantees. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023b. URL http://papers.nips.cc/paper\_files/paper/2023/ hash/95827e011b9e899f189a01fe2f4ef316-Abstract-Conference.html.

## A Continuous Time

Termination. For a drift condition $( S , l )$ , we assume that the deployed process terminates at its first exit from S. We retain the terminal state, so X denotes the absorbed process. More precisely,

$$
\sigma : = \operatorname* { i n f } \{ t \geq 0 : X _ { t } \notin S \} , \quad \quad \operatorname* { i n f } \varnothing : = \infty ,
$$

is an $\left( \mathcal { G } _ { t } \right)$ -stopping time, and, on $\{ \sigma < \infty \} , X _ { \sigma } \notin \mathcal { S }$ and $X _ { t } = X _ { \sigma }$ for every $t \geq \sigma$ . We do not require σ to be finite. For example, for a reach–avoid certificate, S is the continuation region before reaching the target or an unsafe state. This assumption concerns the actual first exit, including exits between observations. This stopping-time requirement ensures that the process does not leave S without our knowledge. Meaning, we are only concerned with the certificate and loss in ${ \mathcal { S } } .$ Hence, we extend the allowed drift by zero outside S:

$$
l _ { S } ( x ) : = \left\{ { l ( x ) , \atop D , \quad x \not \in S . } \right.\tag{7}
$$

Absorption ensures that an exit is detectable at the next observation because the terminal state is retained. The proof of Lemma 3 uses this to extend the regional drift inequality to all visited states.

Continuous-time assumptions. In continuous time, the expected drift condition bounds the instantaneous rate of change of the certificate. To obtain an expected bound on the certificate change between observations, we impose the following assumptions. As in the preliminaries, X is a right-continuous, time-homogeneous Markov process, and $\left( \mathcal { G } _ { t } \right)$ is complete and right-continuous. Right-continuity and adaptation ensure the joint measurability used in the Tonelli argument in the proof of Lemma 3.

• Generator: There is a finite measurable function $\mathbf { \boldsymbol { A f } }$ such that, for every fixed $t \geq 0$ , almost surely,

$$
A f ( X _ { t } ) = \operatorname* { l i m } _ { u \downarrow 0 } { \frac { \operatorname { \mathbb { E } } [ f ( X _ { t + u } ) \mid X _ { t } ] - f ( X _ { t } ) } { u } } .
$$

This is the generator drift of the absorbed process, with $A f = 0$ outside S. The identity is used in the proof of Lemma 3 to express the instantaneous certificate condition as $\mathcal { A } f ( X _ { r } ) \leq l ( X _ { r } )$ on $\{ X _ { r } \in S \}$

• Integrability: For every finite T, we assume

$$
\mathbb { E } \left[ | f ( X _ { 0 } ) | + \int _ { 0 } ^ { T } \bigl ( | A f ( X _ { r } ) | + | l _ { S } ( X _ { r } ) | \bigr ) \mathrm { d } r \right] < \infty .\tag{8}
$$

Together with integrability of $N _ { t } ^ { f }$ below, this establishes integrability of $f ( X _ { t } )$ and $L _ { n }$ in the opening argument of the proof of Lemma 3. It also ensures that the conditional expectation of the integrated drift diference in the final step is finite.

• Martingale: The process $( N _ { t } ^ { f } ) _ { t \geq 0 }$ defined by

$$
N _ { t } ^ { f } : = f ( X _ { t } ) - f ( X _ { 0 } ) - \int _ { 0 } ^ { t } { \mathcal { A } } f ( X _ { r } ) \mathrm { d } r\tag{9}
$$

is a right-continuous $\left( \mathcal { G } _ { t } \right)$ -martingale. Thus, subtracting the accumulated actual drift leaves a process with zero conditional expected change. The proof of Lemma 3 uses this property to obtain $\mathbb { E } [ N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } \ | \mathcal { G } _ { t _ { n } } ] = 0$ , leaving only the integrated drift diference in the expression for $E _ { n }$

For the martingale formulation of extended generators, see Hirsch and Yor (2012, Definition 4.1 and Proposition 4.1). We additionally require the displayed infinitesimal limit to exist. The certificate condition $\mathcal { A } f \leq l$ on $s$ is imposed separately. The usual filtration conditions and right-continuity of $N ^ { f }$ are retained as regularity assumptions; the fixed-time proof of Lemma 3 does not use them separately.

Observations. We observe the absorbed process at fixed times $t _ { n } = n h$ . The observation filtration satisfies $\mathcal { F } _ { n } \subseteq \mathcal { G } _ { t _ { n } } ,$ and $X _ { t _ { n } }$ is $\mathcal { F } _ { n } { \mathrm { - m e a s u r a b l e } }$ . The proof of Lemma 3 applies the martingale identity at these fixed times and uses the filtration inclusion to condition further on ${ \mathcal { F } } _ { n }$ . Measurability of the sampled state makes the loss bound below predictable, as required when taking $1 / B _ { n }$ outside the conditional expectation in the proof of Lemma 1.

The accumulated allowed drift is

$$
L _ { n } : = \int _ { t _ { n } } ^ { t _ { n + 1 } } l _ { S } ( X _ { r } ) \mathrm { d } r = \int _ { t _ { n } \wedge \sigma } ^ { t _ { n + 1 } \wedge \sigma } l _ { S } ( X _ { r } ) \mathrm { d } r .\tag{10}
$$

Thus, the allowed drift stops accumulating at termination. The interval containing termination is scored using the retained terminal state. On later intervals, we set $\overline { { L } } _ { n } = 0$ , giving $D _ { n } = W _ { n } = 0$ Termination itself does not raise a statistical alarm.

Bounds at termination. Suppose that $l ( x ) \leq c$ on S and $f ( x ) \geq f _ { \operatorname* { m i n } }$ on all reachable states, including terminal states. Write $c _ { + } : = \operatorname* { m a x } \{ c , 0 \}$ . If the termination time is available when it occurs, define

$$
\Delta _ { n } : = ( t _ { n + 1 } \wedge \sigma ) - ( t _ { n } \wedge \sigma ) , \qquad { \overline { { L } } } _ { n } : = c \Delta _ { n } .
$$

Then $L _ { n } \leq c \Delta _ { n }$ for either sign of c. If only sampled states are available, an upper bound is

$$
\begin{array} { r } { \overline { { L } } _ { n } : = \left\{ \begin{array} { l l } { c h , } & { X _ { t _ { n + 1 } } \in S , } \\ { c _ { + } h , } & { X _ { t _ { n } } \in S , \ X _ { t _ { n + 1 } } \notin S , } \\ { 0 , } & { X _ { t _ { n } } \notin S . } \end{array} \right. } \end{array}\tag{11}
$$

Indeed, survival until $t _ { n + 1 }$ ensures that the whole interval was spent in $s ,$ while a terminating interval has duration $0 \leq \Delta _ { n } \leq h$ and $L _ { n } \leq c \Delta _ { n } \leq c _ { + } h$ . If the process has already terminated, $L _ { n } = 0$ . In both cases, the predictable bound

$$
B _ { n } : = \operatorname* { m a x } \{ b _ { \mathrm { m i n } } , f ( X _ { t _ { n } } ) + c _ { + } h - f _ { \mathrm { m i n } } \}\tag{12}
$$

satisfies $D _ { n } \geq - B _ { n }$ , because $\overline { { L } } _ { n } \leq c _ { + } h$ . For a negative constant drift bound, using ch on the terminating interval would generally underestimate the accumulated allowance. The piecewise bound above avoids this without requiring the exact termination time.

Lemma 3 (Continuous-time expected increments). Under the continuous-time assumptions above, $f ( X _ { t } )$ is integrable for every finite t. If the $( S , l )$ -expected drift condition holds, then, for every $n \in  { \mathbb { N } } _ { 0 }$

$$
E _ { n } : = \mathbb { E } \left[ f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - L _ { n } \left| { \mathcal { F } } _ { n } \right. \right] \leq 0 \qquad a l m o s t \ s u r e l y .\tag{13}
$$

Proof. Fix an observation index $n \in  { \mathbb { N } } _ { 0 }$

Step 1: We check integrability. By the martingale decomposition in (9),

$$
f ( X _ { t } ) = f ( X _ { 0 } ) + N _ { t } ^ { f } + \int _ { 0 } ^ { t } { \mathcal { A } } f ( X _ { r } ) \mathrm { d } r .
$$

Taking absolute values and expectations gives

$$
\mathbb { E } [ | f ( X _ { t } ) | ] \leq \mathbb { E } [ | f ( X _ { 0 } ) | ] + \mathbb { E } [ | N _ { t } ^ { f } | ] + \mathbb { E } \left[ \int _ { 0 } ^ { t } | A f ( X _ { r } ) | \mathrm { d } r \right] < \infty .
$$

The first and third terms are finite by (8). The second is finite because $N ^ { f }$ is a martingale. The same integrability assumption gives

$$
\mathbb { E } [ | L _ { n } | ] \leq \mathbb { E } \left[ \int _ { t _ { n } } ^ { t _ { n + 1 } } | l _ { S } ( X _ { r } ) | \mathrm { d } r \right] < \infty .
$$

Thus, the certificate values and allowance defining $E _ { n }$ are integrable.

Step 2: We bound the drift diference at each fixed time. Fix $r \geq 0$ . If $X _ { r } \in S$ , the certificate condition gives

$$
A f ( X _ { r } ) \leq l ( X _ { r } ) = l s ( X _ { r } ) \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

If $X _ { r } \notin S$ , the process is absorbed. Its generator drift is therefore zero, and $l _ { S } ( X _ { r } ) = 0$ by definition. Combining the two cases yields

$$
A f ( X _ { r } ) - l s ( X _ { r } ) \leq 0 \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Step 3: We pass from fixed times to the whole interval. Write

$$
J _ { n } : = \int _ { t _ { n } } ^ { t _ { n + 1 } } ( A f ( X _ { r } ) - l _ { S } ( X _ { r } ) ) \mathrm { d } r .
$$

This integral is integrable because

$$
\mathbb { E } [ | J _ { n } | ] \leq \mathbb { E } \left[ \int _ { t _ { n } } ^ { t _ { n + 1 } } \bigl ( | A f ( X _ { r } ) | + | l _ { S } ( X _ { r } ) | \bigr ) \mathrm { d } r \right] < \infty .
$$

We next show that $J _ { n } \leq 0$ almost surely. Step 2 shows that, for each fixed time $r ,$ , the drift diference is nonpositive with probability one. We now apply Tonelli’s theorem to deduce that its integral over the observation interval is also nonpositive with probability one. For $( z ) _ { + } : = \operatorname* { m a x } \{ z , 0 \}$ , joint measurability and nonnegativity give

$$
\mathbb { E } \left[ \int _ { t _ { n } } ^ { t _ { n + 1 } } \bigl ( A f ( X _ { r } ) - l _ { \mathcal { S } } ( X _ { r } ) \bigr ) _ { + } \mathrm { d } r \right] = \int _ { t _ { n } } ^ { t _ { n + 1 } } \mathbb { E } \left[ \bigl ( A f ( X _ { r } ) - l _ { \mathcal { S } } ( X _ { r } ) \bigr ) _ { + } \right] \mathrm { d } r = 0 .
$$

The last equality follows from Step 2 at each fixed r. A nonnegative random variable with expectation zero is zero almost surely. Consequently,

$$
J _ { n } \leq \int _ { t _ { n } } ^ { t _ { n + 1 } } \bigl ( A f ( X _ { r } ) - l _ { S } ( X _ { r } ) \bigr ) _ { + } \mathrm { d } r = 0 \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

Step $\it 4 .$ We decompose the corrected certificate increment. Subtracting (9) at $t _ { n }$ from the same identity at $t _ { n + 1 }$ gives

$$
f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) = N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } + \int _ { t _ { n } } ^ { t _ { n + 1 } } \mathcal { A } f ( X _ { r } ) \mathrm { d } r .
$$

Subtracting $L _ { n }$ then yields

$$
f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - L _ { n } = N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } + J _ { n } .
$$

Thus, the corrected increment is the sum of a martingale increment and the nonpositive accumulated drift diference.

Step 5: We take conditional expectations. The martingale property gives

$$
\mathbb { E } [ N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } \mid \mathcal { F } _ { t _ { n } } ] = N _ { t _ { n } } ^ { f } - N _ { t _ { n } } ^ { f } = 0 .
$$

Since $\mathcal { F } _ { n } \subseteq \mathcal { F } _ { t _ { n } }$ , the tower property gives

$$
\begin{array} { r l } & { \mathbb { E } [ N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } \mid \mathcal { F } _ { n } ] = \mathbb { E } \left[ \mathbb { E } [ N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } \mid \mathcal { F } _ { t _ { n } } ] \Big \vert \mathcal { F } _ { n } \right] } \\ & { \quad \quad \quad = 0 . } \end{array}
$$

Applying conditional expectation to the decomposition in Step 4, we therefore obtain

$$
\begin{array} { r l } & { E _ { n } = \mathbb { E } [ f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) - L _ { n } | \mathcal { F } _ { n } ] } \\ & { \quad = \mathbb { E } [ N _ { t _ { n + 1 } } ^ { f } - N _ { t _ { n } } ^ { f } | \mathcal { F } _ { n } ] + \mathbb { E } [ J _ { n } | \mathcal { F } _ { n } ] } \\ & { \quad = \mathbb { E } [ J _ { n } | \mathcal { F } _ { n } | \le 0 . } \end{array}
$$

The final inequality follows from $J _ { n } \leq 0$ almost surely and monotonicity of conditional expectation.

## B Validity

Score decrease. We show the score decrease condition. Intuitively, the score compares the observed certificate change with an upper bound on the allowed drift. Subtracting this upper bound can only reduce the corrected increment. Normalising by a positive bound chosen before the next observation then preserves the sign of its conditional mean.

Proof of Lemma 1. The integrability assumptions imply that $D _ { n }$ is integrable. Since $B _ { n } \geq$ $b _ { \operatorname* { m i n } } > 0 , | W _ { n } | \leq | D _ { n } | / b _ { \operatorname* { m i n } }$ , so $W _ { n }$ is also integrable. The loss bound gives $W _ { n } \geq - 1$ . Adding and subtracting $L _ { n }$ and using the ${ \mathcal { F } } _ { n }$ -measurability of $B _ { n }$ gives

$$
\mu _ { n } = \frac { \mathbb { E } [ D _ { n } \mid \mathcal { F } _ { n } ] } { B _ { n } } = \frac { E _ { n } - \mathbb { E } [ \overline { { L } } _ { n } - L _ { n } \mid \mathcal { F } _ { n } ] } { B _ { n } } \leq \frac { E _ { n } } { B _ { n } } .
$$

The inequality follows from $\overline { { L } } _ { n } \geq L _ { n }$ . In continuous time, Lemma 3 gives $E _ { n } \leq 0$ under the certificate condition. In discrete time, $h = 1$ gives $t _ { n + 1 } = t _ { n } + 1$ . On $\{ X _ { t _ { n } } \in S \}$ , we have $L _ { n } = l ( X _ { t _ { n } } )$ . The Markov property and the one-step expected-drift condition therefore give

$$
\begin{array} { r l } & { E _ { n } = \mathbb { E } [ f ( X _ { t _ { n + 1 } } ) - f ( X _ { t _ { n } } ) \mid { \mathcal F } _ { n } ] - l ( X _ { t _ { n } } ) } \\ & { \quad \quad = \mathbb { E } [ f ( X _ { t _ { n } + 1 } ) - f ( X _ { t _ { n } } ) \mid X _ { t _ { n } } ] - l ( X _ { t _ { n } } ) \le 0 } \end{array}
$$

almost surely on $\{ X _ { t _ { n } } \in S \}$ . Consequently, the certificate condition implies $\mu _ { n } \leq 0$ for all continuous-time observations and, in discrete time, on $\{ X _ { t _ { n } } \in S \}$ □

Statistical guarantee. For the proof, we freeze the wealth just before the first observation interval with positive conditional expected corrected drift. Before freezing, each bet has nonpositive expected gain; afterwards, the wealth stays constant. The frozen wealth is therefore a supermartingale, so Ville’s inequality applies.

Proof of Theorem 2. We first construct an auxiliary monitor whose scores always have nonpositive conditional means. We then apply Ville’s inequality to its wealth and show that every alarm covered by the historical guarantee also crosses the auxiliary wealth threshold.

Step 1: We replace scores from the first violation onwards. We define the first index at which the conditional expected corrected increment is positive, i.e., κ := inf $\{ n \in \mathbb { N } _ { 0 } : E _ { n } > 0 \}$ , with inf $\mathcal { O } : = \infty$ and freeze the process at that time ${ \widetilde { W } } _ { n } : = \mathbf { 1 } \{ n < \kappa \} W _ { n }$ . The auxiliary scores agree with the actual scores before κ and are zero from κ onwards. For every $k \leq n$ , the random variable $E _ { k }$ is $\mathcal { F } _ { k }$ -measurable and hence ${ \mathcal { F } } _ { n }$ -measurable. Consequently,

$$
\{ n < \kappa \} = \bigcap _ { k = 0 } ^ { n } \{ E _ { k } \leq 0 \} \in \mathcal { F } _ { n } .
$$

The indicator multiplying $W _ { n }$ is therefore measurable before that score is observed. We may take it outside the conditional expectation:

$$
\mathbb { E } [ \widetilde { W } _ { n } \mid \mathcal { F } _ { n } ] = \mathbf { 1 } \{ n < \kappa \} \mathbb { E } [ W _ { n } \mid \mathcal { F } _ { n } ] = \mathbf { 1 } \{ n < \kappa \} \mu _ { n } .
$$

On $\{ n < \kappa \}$ , score validity gives $\mu _ { n } \leq E _ { n } / B _ { n } \leq 0$ . On its complement, the displayed expression is zero. Hence $\mathbb { E } [ \widetilde { W } _ { n } \mid \mathcal { F } _ { n } ] \leq 0$ almost surely. Moreover, $| \widetilde { W } _ { n } | \leq | W _ { n } |$ and $\widetilde { W } _ { n } \geq - 1$ , so the auxiliary scores remain integrable and satisfy the same lower bound.

Step 2: Each auxiliary account is a nonnegative supermartingale. For each expert a launched at s, define

$$
\widetilde { G } _ { s , s } ^ { a } = 1 , \qquad \widetilde { G } _ { s , n + 1 } ^ { a } = \widetilde { G } _ { s , n } ^ { a } ( 1 + \lambda _ { s , n } ^ { a } \widetilde { W } _ { n } ) .
$$

We use exactly the stakes chosen by the actual monitor. These stakes remain $\mathcal { F } _ { n } \mathrm { - m e a s u r a b l e } .$ because they are still computed from the actual observation history.

Since $0 \leq \lambda _ { s , n } ^ { a } \leq 1$ and $\widetilde { W } _ { n } \geq - 1$ , it follows that $1 + \lambda _ { s , n } ^ { a } \widetilde { W } _ { n } \geq 1 - \lambda _ { s , n } ^ { a } \geq 0$ . Every account therefore remains nonnegative. Both its current wealth and its next stake are ${ \mathcal { F } } _ { n } .$ -measurable, so Step 1 gives

$$
\mathbb { E } [ \widetilde { G } _ { s , n + 1 } ^ { a } \ | \ \mathcal { F } _ { n } ] = \widetilde { G } _ { s , n } ^ { a } \mathbb { E } [ 1 + \lambda _ { s , n } ^ { a } \widetilde { W } _ { n } \ | \ \mathcal { F } _ { n } ] = \widetilde { G } _ { s , n } ^ { a } \big ( 1 + \lambda _ { s , n } ^ { a } \mathbb { E } [ \widetilde { W } _ { n } \ | \ \mathcal { F } _ { n } ] \big ) \le \widetilde { G } _ { s , n } ^ { a } .
$$

This calculation also establishes integrability. Because the next account value is nonnegative, its conditional expectation is defined even before integrability is established, although it could initially be infinite. Starting from $\widetilde { G } _ { s , s } ^ { a } = 1$ , the displayed inequality and the tower property give inductively

$$
\mathbb { E } [ \widetilde { G } _ { s , n + 1 } ^ { a } ] \leq \mathbb { E } [ \widetilde { G } _ { s , n } ^ { a } ] \leq 1 .
$$

Thus every account value has finite expectation, and $( \widetilde { G } _ { s , n } ^ { a } ) _ { n \geq s }$ is a nonnegative supermartingale.

Step 3: The auxiliary mixtures are also supermartingales. Use the same mixture weights as the actual monitor and set

$$
\widetilde { Q } _ { s , n } : = \sum _ { a \in \mathcal { A } _ { s } } \eta _ { s , a } \widetilde { G } _ { s , n } ^ { a } .
$$

The weights are nonnegative and sum to one, so $\widetilde { Q } _ { s , n } \geq 0$ and $\widetilde { Q } _ { s , s } = 1$ . They are chosen at launch and are therefore $\mathcal { F } _ { s }$ -measurable, hence $\mathcal { F } _ { n }$ -measurable for $n \geq s$ . Using Step 2,

$$
\begin{array} { r l } & { \mathbb { E } [ \widetilde { Q } _ { s , n + 1 } \mid \mathcal { F } _ { n } ] = \displaystyle \sum _ { a \in \mathcal { A } _ { s } } \eta _ { s , a } \mathbb { E } [ \widetilde { G } _ { s , n + 1 } ^ { a } \mid \mathcal { F } _ { n } ] } \\ & { \quad \quad \le \displaystyle \sum _ { a \in \mathcal { A } _ { s } } \eta _ { s , a } \widetilde { G } _ { s , n } ^ { a } = \widetilde { Q } _ { s , n } . } \end{array}
$$

As above, nonnegativity and the initial value $\widetilde { Q } _ { s , s } = 1$ give integrability by induction. Thus each mixture is a nonnegative supermartingale. No independence between its accounts is needed.

Step 4: Funding preserves the supermartingale property. Use the actual funding amounts $c _ { s }$ and define

$$
C _ { n } : = \sum _ { s = 0 } ^ { n - 1 } c _ { s } , \qquad \widetilde M _ { n } : = 1 - C _ { n } + \sum _ { s = 0 } ^ { n - 1 } c _ { s } \widetilde Q _ { s , n } .
$$

The funding constraint $0 \leq c _ { n } \leq 1 - C _ { n }$ ensures $0 \leq C _ { n } \leq 1$ . Hence the reserve and every funded contribution are nonnegative, giving $\breve { M } _ { n } \geq 0$ . The sum contains finitely many terms at each n.

Since $c _ { s } \leq 1$ and each mixture is integrable, $\widetilde { M } _ { n }$ is integrable as well. Initially, no accounts have been funded, so $\widetilde { M } _ { 0 } = 1$ . At index $n ,$ funding removes $c _ { n }$ from the reserve and assigns it to a new mixture with unit initial wealth. Thus

$$
\widetilde { M } _ { n + 1 } = 1 - C _ { n } - c _ { n } + \sum _ { s = 0 } ^ { n - 1 } c _ { s } \widetilde { Q } _ { s , n + 1 } + c _ { n } \widetilde { Q } _ { n , n + 1 } .
$$

All funding amounts appearing here are ${ \mathcal { F } } _ { n } .$ -measurable. Taking conditional expectations and applying Step 3 gives

$$
\begin{array} { r l } & { \mathbb { E } [ \widetilde { M } _ { n + 1 } \mid \mathcal { F } _ { n } ] = 1 - C _ { n } - c _ { n } + \displaystyle \sum _ { s = 0 } ^ { n - 1 } c _ { s } \mathbb { E } [ \widetilde { Q } _ { s , n + 1 } \mid \mathcal { F } _ { n } ] + c _ { n } \mathbb { E } [ \widetilde { Q } _ { n , n + 1 } \mid \mathcal { F } _ { n } ] } \\ & { \quad \quad \quad \quad \quad \quad \leq 1 - C _ { n } - c _ { n } + \displaystyle \sum _ { s = 0 } ^ { n - 1 } c _ { s } \widetilde { Q } _ { s , n } + c _ { n } \widetilde { Q } _ { n , n } } \\ & { \quad \quad \quad \quad = 1 - C _ { n } + \displaystyle \sum _ { s = 0 } ^ { n - 1 } c _ { s } \widetilde { Q } _ { s , n } = \widetilde { M } _ { n } . } \end{array}
$$

The last equality uses ${ \tilde { Q } } _ { n , n } = 1 \colon$ the capital assigned to the new mixture exactly matches the amount removed from the reserve. Therefore $( \widetilde { M } _ { n } )$ is a nonnegative supermartingale.

Step 5: We bound threshold crossings of the auxiliary wealth. Since $\stackrel { \triangledown } { M } _ { 0 } = 1$ , Ville’s inequality gives

$$
\mathbb { P } \left( \exists n \in \mathbb { N } _ { 0 } : \widetilde { M } _ { n } \geq 1 / \delta \right) \leq \delta .
$$

Step 6: We transfer this bound to the actual monitor. Consider an outcome on which $\tau _ { \delta } < \infty$ and $E _ { k } \le 0$ for every $k < \tau _ { \delta } . \mathrm { ~ B y ~ }$ definition of $\kappa ,$ this implies $\kappa \geq \tau _ { \delta }$ . Consequently,

$$
\widetilde { W } _ { k } = W _ { k } \qquad \mathrm { f o r ~ e v e r y ~ } k < \tau _ { \delta } .
$$

These are precisely the scores used to compute the wealth at $\tau _ { \delta }$ . In particular, equality $\kappa = \tau _ { \delta }$ causes no problem: the score with index $\tau _ { \delta }$ has not yet contributed to that wealth.

The actual and auxiliary accounts have the same initial values and use the same stakes. Since their scores agree through index $\tau _ { \delta } - 1$ , their account values agree at $\tau _ { \delta }$ . Their mixture weights, funding amounts, and reserves also agree. It follows that

$$
\widetilde { M } _ { \tau _ { \delta } } = M _ { \tau _ { \delta } } \geq 1 / \delta .
$$

We have therefore established the event inclusion

$$
\left\{ \tau _ { \delta } < \infty , E _ { k } \leq 0 \mathrm { ~ f o r ~ e v e r y ~ } k < \tau _ { \delta } \right\} \subseteq \left\{ \exists n \in \mathbb { N } _ { 0 } : \widetilde { M } _ { n } \geq 1 / \delta \right\} .
$$

Applying the bound from Step 5 yields

$$
\mathbb { P } \left( \tau _ { \delta } < \infty \mathrm { ~ a n d ~ } E _ { k } \leq 0 \mathrm { ~ f o r ~ e v e r y ~ } k < \tau _ { \delta } \right) \leq \delta .
$$

Step 7: We obtain the unconditional false-alarm bound. Suppose that the certificate condition holds. Score validity gives $\mu _ { n } \leq 0$ almost surely on $\{ X _ { t _ { n } } \in S \}$ . Outside $s ,$ every expert chooses stake zero. Consequently, for every expert and every $n \geq s .$

$$
\lambda _ { s , n } ^ { a } \mu _ { n } \leq 0 \qquad \mathrm { a l m o s t ~ s u r e l y . }
$$

We can therefore apply the account argument directly to the actual scores:

$$
\begin{array} { r } { \mathbb { E } [ G _ { s , n + 1 } ^ { a } \ \vert \ \mathcal { F } _ { n } ] = G _ { s , n } ^ { a } \big ( 1 + \lambda _ { s , n } ^ { a } \mathbb { E } [ W _ { n } \ \vert \ \mathcal { F } _ { n } ] \big ) = G _ { s , n } ^ { a } ( 1 + \lambda _ { s , n } ^ { a } \mu _ { n } ) \le G _ { s , n } ^ { a } . } \end{array}
$$

The nonnegativity and integrability arguments from Step 2 apply unchanged. The mixture and funding arguments from Steps 3 and 4 then show that the actual wealth $\left( M _ { n } \right)$ is a nonnegative supermartingale with $M _ { 0 } = 1$ . Ville’s inequality therefore gives

$$
\begin{array} { r } { \mathbb { P } ( \tau _ { \delta } < \infty ) = \mathbb { P } \left( \exists n \in \mathbb { N } _ { 0 } : M _ { n } \geq 1 / \delta \right) \leq \delta . } \end{array}
$$

Monitoring several conditions. For a certificate with several drift conditions, each component monitor must satisfy the score assumptions for its own condition. The termination assumption applies separately to the region used in each continuous-time condition. In particular, one physical termination time sufices when the conditions share a continuation region. If their regions difer, stopping at the target or an unsafe state does not by itself justify every component: each component must have access to the process stopped at its own first exit, including its exit state, or use another valid interval construction.

Corollary 4 (Composition). Suppose that a certificate has J expected-drift conditions. For each condition j, run a monitor satisfying Theorem 2 at level $\delta _ { j } > 0$ , where $\textstyle \sum _ { j = 1 } ^ { J } \delta _ { j } \leq \delta$ . Let $\tau$ be the first index at which any component monitor or sampled domain-bound check raises an alarm. If all certificate conditions hold for the deployed process, then $\mathbb { P } ( \tau < \infty ) \le \delta$

Proof. Write $\tau _ { j }$ for the first alarm index of component j. A valid domain-bound condition holds at every fixed sampled time almost surely. Since there are countably many samples and finitely many such conditions, their checks raise no alarm almost surely. The component guarantees and the union bound therefore give

$$
\mathbb { P } ( \tau < \infty ) \le \sum _ { j = 1 } ^ { J } \mathbb { P } ( \tau _ { j } < \infty ) \le \sum _ { j = 1 } ^ { J } \delta _ { j } \le \delta .
$$

Independence is not required.

## C Power

Weighted stakes. The two stake strategies use the same weighted history to decide how strongly to bet. The log-return strategy directly maximises the historical log return. The moment strategy instead maximises a quadratic lower bound, which requires only the weighted sum of the scores and their squares. The following lemma justifies both constructions and explains how these quantities relate to the expected return of the next bet. For the following lemma we fix an expert $a \in \mathcal { A } _ { s }$ , an observation index $n \geq s$ , and a cap $1 / 2 \le \overline { { \lambda } } < 1$ . Moreover, for $0 \leq \lambda \leq \overline { { \lambda } }$ , define:

$$
\omega _ { i } : = \omega _ { s , n , i } ^ { a } , \quad S : = \sum _ { i \in I _ { n } } \omega _ { i } W _ { i } , \quad U : = \sum _ { i \in I _ { n } } \omega _ { i } W _ { i } ^ { 2 } , \quad H ( \lambda ) : = \sum _ { i \in I _ { n } } \omega _ { i } \log ( 1 + \lambda W _ { i } ) .
$$

Lemma 5 (Weighted stake strategies). The following statements hold on $\{ X _ { t _ { n } } \in S \}$

1. The log-return and moment stakes are $F _ { n } – m e a s u r a b l e .$ , both are zero exactly when $S \le 0$ and $0 \leq \lambda _ { s , n } ^ { a , \mathrm { m o m } } \leq \lambda _ { s , n } ^ { a , \mathrm { l o g } } \leq \overline { { \lambda } }$

2. For every $\lambda \in [ 0 , 1 / 2 ]$ , the moment stake maximises the lower bound $H ( \lambda ) \ge \lambda S - \lambda ^ { 2 } U$ over $[ 0 , 1 / 2 ]$

3. For any ${ \mathcal { F } } _ { n }$ -measurable stake $\lambda _ { n } \in [ 0 , 1 / 2 ] , \ i f \ q _ { n } : = \mathbb { E } [ W _ { n } ^ { 2 } \mid { \mathcal { F } } _ { n } ] < \infty \ a . s .$ then

$$
\mathbb { E } [ \log ( 1 + \lambda _ { n } W _ { n } ) \mid { \mathcal F } _ { n } ] \ge \lambda _ { n } \mu _ { n } - \lambda _ { n } ^ { 2 } q _ { n } \qquad a l m o s t s u r e l y .\tag{14}
$$

Proof. We first establish the quadratic lower bound. For $z \ge - 1 / 2$ , define $\psi ( z ) : = \log ( 1 + z ) -$ $z + z ^ { 2 }$ . We take its derivative $\begin{array} { r } { \psi ^ { \prime } ( z ) = \frac { z ( 1 + 2 z ) } { 1 + z } } \end{array}$ and observe that $\psi ^ { \prime } ( z ) \le 0 \mathrm { ~ o n ~ } [ - 1 / 2 , 0 ]$ , because the factor z is nonpositive, while $1 + 2 z$ is nonnegative and $1 + z$ is positive. Moreover, on $[ 0 , \infty )$ all factors are nonnegative, so $\psi ^ { \prime } ( z ) \geq 0$ . Thus, $\psi$ attains its minimum at zero. Since $\psi ( 0 ) = 0$ substituting $z = \lambda w$ implies that

$$
\begin{array} { r } { \log ( 1 + \lambda w ) \ge \lambda w - \lambda ^ { 2 } w ^ { 2 } \qquad \mathrm { f o r ~ } w \ge - 1 , \quad 0 \le \lambda \le \frac { 1 } { 2 } . } \end{array}\tag{15}
$$

Multiplying by the nonnegative weights and summing yields $H ( \lambda ) \geq \lambda S - \lambda ^ { 2 } U .$

The linear term rewards a positive weighted score sum, while the quadratic term penalises large stakes. If $U > 0$ , the derivative of this lower bound is $S - 2 \lambda U$ , so its maximiser over $[ 0 , 1 / 2 ]$ is

$$
\lambda _ { s , n } ^ { a , \mathrm { m o m } } = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \frac { S _ { + } } { 2 U } \right\} .
$$

If $U = 0 ,$ , every score with positive weight is zero. Hence, $S = 0$ , the lower bound is constant, and the strategy chooses zero.

We next consider the log-return strategy. Since $W _ { i } \geq - 1$ and $\bar { \lambda } < 1$ , the logarithms are well-defined throughout the optimisation interval. The objective is continuous on a compact interval, so a maximiser exists. Its derivatives are

$$
H ^ { \prime } ( \lambda ) = \sum _ { i \in I _ { n } } \frac { \omega _ { i } W _ { i } } { 1 + \lambda W _ { i } } , \qquad H ^ { \prime \prime } ( \lambda ) = - \sum _ { i \in I _ { n } } \frac { \omega _ { i } W _ { i } ^ { 2 } } { ( 1 + \lambda W _ { i } ) ^ { 2 } } .
$$

Consequently, H is concave, and strictly concave whenever $\textstyle \sum _ { i \in I _ { n } } \omega _ { i } W _ { i } ^ { 2 } > 0$ . The derivative $\begin{array} { r } { H ^ { \prime } ( 0 ) = \sum _ { i \in I _ { n } } \omega _ { i } W _ { i } } \end{array}$ describes whether increasing the stake from zero initially improves the historical return. If $S \le 0$ , concavity implies that zero is the smallest maximiser. $\mathrm { I f } \ S > 0$ , the maximiser is positive. It equals λ when $H ^ { \prime } ( { \overline { { \lambda } } } ) \geq 0 ;$ otherwise, it is the unique root of $H ^ { \prime }$ in $( 0 , { \bar { \lambda } } )$ . This gives the endpoint checks and bisection procedure used in the implementation. All inputs are ${ \mathcal { F } } _ { n } .$ -measurable, as are the bisection iterates and their limit. The explicit moment stake is likewise measurable.

To compare the stakes, suppose $S > 0$ . For $\lambda \in [ 0 , 1 / 2 ]$ , we have $1 + \lambda W _ { i } \ge 1 / 2$ , and therefore

$$
H ^ { \prime } ( \lambda ) = S - \lambda \sum _ { i \in I _ { n } } \frac { \omega _ { i } W _ { i } ^ { 2 } } { 1 + \lambda W _ { i } } \ge S - 2 \lambda U .
$$

At the moment stake, the final expression is nonnegative. Thus, the historical log return has not started decreasing at that stake, so its maximiser cannot lie below it. If $S \le 0$ , both stakes are zero.

Finally, the pointwise logarithmic bound also applies to the next score. Taking conditional expectations and using predictability gives

$$
\begin{array} { r } { \mathbb E [ \log ( 1 + \lambda _ { n } W _ { n } ) \mid \mathcal F _ { n } ] \ge \lambda _ { n } \mu _ { n } - \lambda _ { n } ^ { 2 } q _ { n } . } \end{array}
$$

The logarithm is integrable because

$$
| \mathrm { l o g } ( 1 + \lambda _ { n } W _ { n } ) | \leq \log 2 + \frac { 1 } { 2 } ( W _ { n } ) _ { + } ,
$$

and $W _ { n }$ is integrable under the standing assumptions.

The conditional bound explains why the moment strategy uses both the scores and their squares. A positive conditional mean favours betting, while a large second moment makes a large stake less attractive. In particular, the expected log return is positive whenever

$$
\lambda _ { n } > 0 \qquad \mathrm { a n d } \qquad \mu _ { n } > \lambda _ { n } q _ { n } .
$$

Thus, suficiently small positive stakes ensure positive expected log return when the conditional mean is positive and the conditional second moment is finite. The log-return strategy achieves at least as much weighted historical log return as the moment strategy, but neither is necessarily better for the next score, e.g., overcommitting to a wrong outcome may decrease wealth drastically.

Locality. We choose spatial and temporal scales using guesses about the regularity of the conditional expected scores. Fix $\theta \in ( 0 , 1 )$ and an expert $a \in \mathcal { A } _ { s }$ launched at observation index s, which aims to detect a conditional expected score of at least $\mu _ { \star , s } ^ { a } > 0$ . The expert guesses spatial and temporal Lipschitz constants $\alpha _ { \mathcal { X } , s } ^ { a } > 0$ and $\alpha _ { \mathbb { T } , s } ^ { a } > 0$ . For the spatial regularity argument, suppose that $\mu _ { n } = \mathbb { E } [ W _ { n } \mid { \mathcal { F } } _ { n } ] = g _ { h } ( X _ { t _ { n } } )$ almost surely for every n, where $g _ { h } \colon \mathcal { X }  \mathbb { R }$ gives the expected score over an observation interval of length h. The regularity guesses are

$$
\begin{array} { r l } { | g _ { h } ( y ) - g _ { h } ( x ) | \leq \alpha _ { \mathcal { X } , s } ^ { a } \| x - y \| , } & { \mathrm { f o r ~ a l l ~ } x , y \in \mathcal { X } , } \\ { | \mu _ { n } - \mu _ { i } | \leq \alpha _ { \mathbb { T } , s } ^ { a } | t _ { n } - t _ { i } | , } & { \mathrm { a l m o s t ~ s u r e l y , ~ f o r ~ a l l ~ } 0 \leq i < n . } \end{array}\tag{16}
$$

Using $\theta \mu _ { \star , s } ^ { a }$ as the tolerated diference between past and current conditional means, we define

$$
r _ { { \mathcal X } , s } ^ { a } : = \frac { \theta \mu _ { \star , s } ^ { a } } { \alpha _ { { \mathcal X } , s } ^ { a } } , \quad \quad r _ { \mathbb { T } , s } ^ { a } : = \frac { \theta \mu _ { \star , s } ^ { a } } { \alpha _ { \mathbb { T } , s } ^ { a } } .
$$

For a cutof or linear kernel with finite scale, if the corresponding guess is correct and $\mu _ { n } \geq \mu _ { \star , s } ^ { a } .$ every observation with positive weight has conditional mean $\mu _ { i } \geq ( 1 - \theta ) \mu _ { \star , s } ^ { a } > 0$ . Giving these observations greater weight can help the expert choose positive stakes during violations, without afecting false-alarm control.

Lipschitz Regularity. The regularity guesses guide which historical observations are useful for predicting the next score. Fix an expert $a \in \mathcal { A } _ { s }$ . Suppose that its spatial regularity guess is valid and its spatial kernel is a cutof or linear kernel with scale

$$
r _ { { \mathcal X } , s } ^ { a } = \frac { \theta \mu _ { \star , s } ^ { a } } { \alpha _ { { \mathcal X } , s } ^ { a } } .
$$

A positive prediction weight requires the spatial kernel to be positive. For a cutof or linear kernel, this is possible only when the distance from the historical state to the current state is at most the spatial scale. Hence,

$$
\omega _ { s , n , i } ^ { a } > 0 \quad \implies \quad \| X _ { t _ { i } } - X _ { t _ { n } } \| \le r _ { \mathscr { X } , s } ^ { a } .
$$

If the spatial regularity guess is valid, the conditional means at these states therefore satisfy

$$
\begin{array} { r l } & { | \mu _ { i } - \mu _ { n } | = | g _ { h } ( X _ { t _ { i } } ) - g _ { h } ( X _ { t _ { n } } ) | } \\ & { \qquad \leq \alpha _ { \mathscr { X } , s } ^ { a } \| X _ { t _ { i } } - X _ { t _ { n } } \| \leq \alpha _ { \mathscr { X } , s } ^ { a } r _ { \mathscr { X } , s } ^ { a } = \theta \mu _ { \star , s } ^ { a } . } \end{array}
$$

Thus, the chosen radius ensures that every observation with positive weight has a conditional mean within $\theta \mu _ { \star , s } ^ { a }$ of the current conditional mean. The same conclusion follows from a valid temporal guess and the corresponding temporal kernel. In particular, if $\mu _ { n } \geq \mu _ { \star , s } ^ { a } ,$ then

$$
\omega _ { s , n , i } ^ { a } > 0 \quad \implies \quad \mu _ { i } \geq ( 1 - \theta ) \mu _ { \star , s } ^ { a } > 0 .
$$

Hence, when the current conditional mean is suficiently positive, the selected observations also come from positive-mean steps. Their realised scores may nevertheless be negative, and their second moments may difer. Geometric kernels retain observations beyond the chosen scale, so they favour relevant observations without ensuring this implication. These regularity guesses guide prediction and need not hold for false-alarm control.

Accumulating evidence. A positive expected log return favours wealth growth, but an alarm requires the realised wealth to reach the prescribed threshold. The following lemma connects the evidence accumulated by one expert to the monitor’s verdict.

Lemma 6 (Suficient evidence for detection). Fix a starting index s, an expert $a \in \mathcal { A } _ { s }$ with $c _ { s } \eta _ { s , a } > 0$ , and an index $n > s$ . Suppose that this expert uses stakes strictly below one. If

$$
\sum _ { k = s } ^ { n - 1 } \log ( 1 + \lambda _ { s , k } ^ { a } W _ { k } ) \geq \log \frac { 1 } { \delta c _ { s } \eta _ { s , a } } ,\tag{17}
$$

then $\tau _ { \delta } \leq n$ . If its stakes are at most $1 / 2$ , it sufices that

$$
\sum _ { k = s } ^ { n - 1 } \left( \lambda _ { s , k } ^ { a } W _ { k } - ( \lambda _ { s , k } ^ { a } ) ^ { 2 } W _ { k } ^ { 2 } \right) \geq \log \frac { 1 } { \delta c _ { s } \eta _ { s , a } } .\tag{18}
$$

Proof. The mixture launched at index s receives $c _ { s }$ units of capital, of which the fraction $\eta _ { s , a }$ is assigned to expert a. The account $G _ { s , n } ^ { a }$ starts from one and records how this initial capital grows through betting. Hence, the expert’s contribution to the total wealth at index n is $c _ { s } \eta _ { s , a } G _ { s , n } ^ { a } .$

By definition, the total wealth is

$$
M _ { n } = 1 - C _ { n } + \sum _ { r = 0 } ^ { n - 1 } c _ { r } \sum _ { b \in \mathcal { A } _ { r } } \eta _ { r , b } G _ { r , n } ^ { b } .
$$

Since $n > s ,$ , this sum includes the contribution of expert a launched at s. The reserve $1 - C _ { n }$ and all other contributions are nonnegative. Keeping only the chosen expert’s contribution therefore gives the lower bound

$$
M _ { n } \geq c _ { s } \eta _ { s , a } G _ { s , n } ^ { a } .
$$

Thus, it sufices to show that this expert’s contribution alone reaches the alarm threshold $1 / \delta$ The expert’s account is the product of its successive betting factors:

$$
G _ { s , n } ^ { a } = \prod _ { k = s } ^ { n - 1 } ( 1 + \lambda _ { s , k } ^ { a } W _ { k } ) .
$$

Each factor is strictly positive because $W _ { k } \geq - 1$ and $\lambda _ { s , k } ^ { a } < 1$ . We can therefore express the product as the exponential of the sum of its logarithms. Under the first condition of the lemma,

$$
G _ { s , n } ^ { a } = \exp \left( \sum _ { k = s } ^ { n - 1 } \log ( 1 + \lambda _ { s , k } ^ { a } W _ { k } ) \right) \geq \exp \left( \log \frac { 1 } { \delta c _ { s } \eta _ { s , a } } \right) = \frac { 1 } { \delta c _ { s } \eta _ { s , a } }
$$

Multiplying by the expert’s initial capital gives

$$
M _ { n } \geq c _ { s } \eta _ { s , a } G _ { s , n } ^ { a } \geq c _ { s } \eta _ { s , a } \frac { 1 } { \delta c _ { s } \eta _ { s , a } } = \frac { 1 } { \delta } .
$$

Since $\tau _ { \delta }$ is the first index at which the total wealth reaches $1 / \delta ,$ it follows that $\tau _ { \delta } \leq n$

Finally, suppose that all stakes of this expert are at most $1 / 2$ . Applying (15) to each betting factor and summing yields

$$
\sum _ { k = s } ^ { n - 1 } \log ( 1 + \lambda _ { s , k } ^ { a } W _ { k } ) \geq \sum _ { k = s } ^ { n - 1 } \left( \lambda _ { s , k } ^ { a } W _ { k } - ( \lambda _ { s , k } ^ { a } ) ^ { 2 } W _ { k } ^ { 2 } \right) .
$$

If the second condition holds, the right-hand side is at least log $\left( 1 / ( \delta c _ { s } \eta _ { s , a } ) \right)$ . The sum of log returns is therefore also at least this threshold, which establishes the first condition. The preceding argument then gives $\tau _ { \delta } \leq n$ □

The lemma explains the cost of dividing capital among experts. An expert with less initial capital must accumulate more log return to trigger an alarm by itself. For example, dividing a mixture equally among K experts adds log K to this suficient threshold compared with assigning that mixture to one expert. This cost allows the monitor to support several strategies without selecting one in advance. Since other accounts also contribute wealth, the monitor may alarm before any individual expert satisfies the condition.

Monitoring several conditions. A certificate may contain several expected-decrease conditions. We can monitor them jointly by dividing the permitted false-alarm probability among their monitors. This union-bound approach allows the monitors to use the same observations, because independence is not required.

Corollary 7 (Composition). Suppose that a certificate has J expected-decrease conditions. For each condition $j ,$ run a monitor satisfying Theorem $\mathcal { Q }$ at a level $\delta _ { j } > 0$ , where $\textstyle \sum _ { j = 1 } ^ { J } \delta _ { j } \leq \delta$ . Let $\tau$ be the first index at which any component monitor or sampled domain-bound check raises an alarm. If all certificate conditions hold for the deployed process, then $\mathbb { P } ( \tau < \infty ) \le \delta$

Proof. A valid domain bound cannot be violated at an observed state. Writing $\tau _ { j }$ for the first alarm index of component monitor $j ,$ , Theorem 2 and the union bound give

$$
\mathbb { P } ( \tau < \infty ) \le \sum _ { j = 1 } ^ { J } \mathbb { P } ( \tau _ { j } < \infty ) \le \sum _ { j = 1 } ^ { J } \delta _ { j } \le \delta .
$$

Monitor power. To quantify detection, we consider a block on which the conditional expected corrected drift $E _ { k }$ exceeds the expected slack in the observable drift bound. The diference determines the signal available to the monitor. The following result uses an additional upper score bound and an expert whose stake is constant throughout the block. For the next lemma, fix a starting index $s ,$ an expert $a \in A _ { r }$ launched at $r \leq s .$ , a duration $N \in \mathbb { N } _ { \geq 1 }$ , and $\beta \in ( 0 , 1 )$ . Suppose that the expert’s capital at the start of the block is $\rho _ { r , s } : = c _ { r } \eta _ { r , a } G _ { r , s } ^ { a } > 0$ almost surely. Let $d _ { \star }$ be a lower bound on the conditional expected corrected drift $E _ { k } ,$ and let $s _ { \star } \geq 0$ be an upper bound on the conditional expected slack E $\sqrt [ 3 ] { L } _ { k } - L _ { k } \mid \mathcal { F } _ { k } ]$ throughout the block. We require $d _ { \star } > s _ { \star } .$ , so a positive excess remains after accounting for the slack. Let $B _ { \mathrm { m a x } } > 0$ be an upper bound on $B _ { k }$ , and assume $\varepsilon : = ( d _ { \star } - s _ { \star } ) / B _ { \mathrm { m a x } } \in ( 0 , 1 ]$ . Formally, for every $k = s , \ldots , s + N - 1$ we require almost surely

$$
\begin{array} { l } { E _ { k } \geq d _ { \star } , } \\ { B _ { k } \leq B _ { \operatorname* { m a x } } , } \end{array}
$$

$$
\begin{array} { r } { \mathbb { E } [ \overline { { L } } _ { k } - L _ { k } \mid \mathcal { F } _ { k } ] \leq s _ { \star } , } \\ { W _ { k } \leq 1 , } \end{array}
$$

$$
\lambda _ { r , k } ^ { a } = \varepsilon / 2 .
$$

The standing score assumptions already give $W _ { k } \geq - 1$ , so the scores lie in $[ - 1 , 1 ]$

Lemma 8 (Detection of a persistent certificate violation). On the $\mathcal { F } _ { s }$ -measurable event

$$
N \geq \frac { 8 B _ { \mathrm { m a x } } ^ { 2 } } { ( d _ { \star } - s _ { \star } ) ^ { 2 } } \log \frac { 1 } { \delta \rho _ { r , s } \beta } ,\tag{19}
$$

the monitor satisfies $\mathbb { P } ( \tau _ { \delta } \le s + N \mid \mathcal { F } _ { s } ) \ge 1 - \beta$ almost surely.

Proof of Lemma 8. First, we translate the certificate excess into a lower bound on the conditional score mean. Since $B _ { k }$ is positive and $\mathcal { F } _ { k }$ -measurable,

$$
\mu _ { k } = \frac { E _ { k } - \mathbb { E } [ \overline { { L } } _ { k } - L _ { k } \mid \mathcal { F } _ { k } ] } { B _ { k } } \geq \frac { d _ { \star } - s _ { \star } } { B _ { k } } \geq \frac { d _ { \star } - s _ { \star } } { B _ { \operatorname* { m a x } } } = \varepsilon .
$$

Thus, each score in the block has conditional mean at least $\varepsilon .$

Write $Z _ { k } = W _ { k } - \mu _ { k }$ . Each $Z _ { k }$ has conditional mean zero and conditional range of length at most 2, because $W _ { k } \in [ - 1 , 1 ]$ . The conditional Hoefding bound gives, for every $q \geq 0$ 2

$$
\begin{array} { r } { \mathbb { E } [ e ^ { - q Z _ { k } } \mid \mathcal { F } _ { k } ] \le e ^ { q ^ { 2 } / 2 } . } \end{array}
$$

For $j = 0 , \ldots , N$ , write $\begin{array} { r } { S _ { j } : = \sum _ { k = s } ^ { s + j - 1 } Z _ { k } } \end{array}$ , with $S _ { 0 } = 0$ . Since $S _ { j }$ is $\mathcal { F } _ { s + j }$ -measurable, the tower property and the conditional Hoefding bound give

$$
\begin{array} { r l } & { \mathbb { E } [ e ^ { - q S _ { j + 1 } } \mid \mathcal { F } _ { s } ] = \mathbb { E } \Big [ e ^ { - q S _ { j } } \mathbb { E } [ e ^ { - q Z _ { s + j } } \mid \mathcal { F } _ { s + j } ] \Big | \mathcal { F } _ { s } \Big ] } \\ & { \qquad \leq e ^ { q ^ { 2 } / 2 } \mathbb { E } [ e ^ { - q S _ { j } } \mid \mathcal { F } _ { s } ] . } \end{array}
$$

Applying this inequality successively for $j = 0 , \ldots , N - 1$ , starting from $e ^ { - q S _ { 0 } } = 1$ , yields

$$
\begin{array} { r } { \mathbb { E } [ e ^ { - q S _ { N } } \mid \mathcal { F } _ { s } ] \le e ^ { N q ^ { 2 } / 2 } . } \end{array}
$$

For $q > 0$ , the event $S _ { N } \leq - u$ is equivalent to $e ^ { - q S _ { N } } \geq e ^ { q u }$ . Conditional Markov’s inequality therefore gives

$$
\begin{array} { r l } & { \mathbb { P } ( S _ { N } \le - u \mid \mathcal { F } _ { s } ) = \mathbb { P } ( e ^ { - q S _ { N } } \ge e ^ { q u } \mid \mathcal { F } _ { s } ) } \\ & { \qquad \le e ^ { - q u } \mathbb { E } [ e ^ { - q S _ { N } } \mid \mathcal { F } _ { s } ] } \\ & { \qquad \le \exp ( - q u + N q ^ { 2 } / 2 ) . } \end{array}
$$

For $u > 0$ , the exponent is minimised at $q = u / N$ . Substituting this choice gives

$$
\mathbb { P } ( S _ { N } \le - u \mid \mathcal { F } _ { s } ) \le \exp \left( - \frac { u ^ { 2 } } { 2 N } \right) .
$$

For $u = 0 .$ , the same bound holds trivially.

Now choose $u = \sqrt { 2 N \log ( 1 / \beta ) }$ , so the right-hand side equals $\beta .$ . Thus, with conditional probability at least $1 - \beta , S _ { N } \ge - \sqrt { 2 N \log ( 1 / \beta ) }$ . Since $W _ { k } = \mu _ { k } + Z _ { k }$ and $\mu _ { k } \geq \varepsilon$ throughout the block, on this event we have

$$
\sum _ { k = s } ^ { s + N - 1 } W _ { k } = \sum _ { k = s } ^ { s + N - 1 } \mu _ { k } + S _ { N } \geq N \varepsilon - \sqrt { 2 N \log ( 1 / \beta ) } .
$$

We next translate the score bound into growth of the expert’s wealth. Since $\rho _ { r , s } > 0$ the account wealth $G _ { r , s } ^ { a }$ is positive. During the block, the expert uses the constant stake $\lambda = \varepsilon / 2 \le 1 / 2$ . Each betting factor satisfies $1 + \lambda W _ { k } \ge 1 / 2$ , so

$$
\frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } = \prod _ { k = s } ^ { s + N - 1 } ( 1 + \lambda W _ { k } ) > 0 .
$$

Taking logarithms converts this product into a sum. For each score, the inequality log $\left( 1 + \lambda w \right) \geq$ $\lambda w - \lambda ^ { 2 } w ^ { 2 }$ and the bound $W _ { k } ^ { 2 } \le 1$ give

$$
\begin{array} { r } { \log ( 1 + \lambda W _ { k } ) \ge \lambda W _ { k } - \lambda ^ { 2 } W _ { k } ^ { 2 } \ge \lambda W _ { k } - \lambda ^ { 2 } . } \end{array}
$$

Summing over the N scores therefore yields

$$
\log \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \lambda \sum _ { k = s } ^ { s + N - 1 } W _ { k } - N \lambda ^ { 2 } .
$$

On the score event established above, substituting $\lambda = \varepsilon / 2$ gives

$$
\log \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \frac { \varepsilon } { 2 } \left( N \varepsilon - \sqrt { 2 N \log ( 1 / \beta ) } \right) - \frac { N \varepsilon ^ { 2 } } { 4 } = \frac { N \varepsilon ^ { 2 } } { 4 } - \varepsilon \sqrt { \frac { N } { 2 } \log ( 1 / \beta ) } .
$$

To simplify the square-root term, apply $2 \sqrt { u v } \leq u + v$ with $u = N \varepsilon ^ { 2 } / 8$ and $v = \log ( 1 / \beta )$ :

$$
\varepsilon \sqrt { \frac { N } { 2 } \log ( 1 / \beta ) } = 2 \sqrt { \frac { N \varepsilon ^ { 2 } } { 8 } \log ( 1 / \beta ) } \leq \frac { N \varepsilon ^ { 2 } } { 8 } + \log ( 1 / \beta ) .
$$

Consequently,

$$
\log \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \frac { N \varepsilon ^ { 2 } } { 8 } - \log ( 1 / \beta ) .
$$

It remains to show that this growth sufices to reach the alarm threshold. Since $\varepsilon =$ $( d _ { \star } - s _ { \star } ) / B _ { \mathrm { m a x } }$ , the duration condition (19) implies

$$
\frac { N \varepsilon ^ { 2 } } { 8 } \geq \log \frac { 1 } { \delta \rho _ { r , s } \beta } = \log \frac { 1 } { \delta \rho _ { r , s } } + \log ( 1 / \beta ) .
$$

Combining the two bounds and exponentiating gives

$$
\log \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \log \frac { 1 } { \delta \rho _ { r , s } } \quad \Longrightarrow \quad \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \frac { 1 } { \delta \rho _ { r , s } } .
$$

The expert starts the block with capital $\rho _ { r , s } = c _ { r } \eta _ { r , a } G _ { r , s } ^ { a }$ . Multiplying this capital by its growth factor gives its contribution to the total wealth at $s + N$ . Since the reserve and all other accounts are nonnegative,

$$
M _ { s + N } \geq c _ { r } \eta _ { r , a } G _ { r , s + N } ^ { a } = \rho _ { r , s } \frac { G _ { r , s + N } ^ { a } } { G _ { r , s } ^ { a } } \geq \rho _ { r , s } \frac { 1 } { \delta \rho _ { r , s } } = \frac { 1 } { \delta } .
$$

Thus, the monitor has raised an alarm by index $s + N$

The score event has conditional probability at least $1 - \beta$ given ${ \mathcal { F } } _ { s } ,$ and the duration condition is $\mathcal { F } _ { s }$ -measurable. Hence, on that condition,

$$
\mathbb { P } ( \tau _ { \delta } \le s + N \mid \mathcal { F } _ { s } ) \ge 1 - \beta \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

The suficient duration increases quadratically as the observable excess $d _ { \star } - s _ { \star }$ decreases, and logarithmically as the expert’s capital at the start of the block decreases. For an expert launched at $s ,$ this capital is $\rho _ { s , s } = c _ { s } \eta _ { s , a }$ . For an expert launched earlier, $\rho _ { r , s }$ also accounts for its previous gains and losses. A conservative drift bound can conceal a violation if its expected slack is as large as the excess. With observation gap $h ,$ the N scores span physical time $h N$ The upper score bound is used only for this detection result; false-alarm control still requires only $W _ { k } \geq - 1$

## D Implementation

Algorithm 1 implements the monitor using a shared history of the most recent m completed scores and their starting states. Before each new observation, it chooses the funding amount, computes the loss bound, and uses the retained history to choose the experts’ stakes. Upon observing the next state, it computes the score, updates the accounts, and checks whether the total wealth reaches $1 / \delta$ . The completed score is then added to the history.

```latex
Algorithm 1 Monitor with combined expert accounts
Require: Certificate ${ \overline { { f } } } ,$ allowed drift l, spacing h, level ${ \overline { { \delta , } } }$ history size m;
valid score-bound routines; a fixed set A of K experts and weights $\eta _ { a } ;$
predictable funding rule $0 \leq c _ { n } \leq R _ { n }$
1: Observe $X _ { t _ { 0 } } ;$ initialise $R _ { 0 }  1 , A _ { 0 } ^ { a }  0$ for every $a \in A ,$ and empty history.
2: for $n = 0 , 1 , \ldots$ . while observations continue do
3: Choose $c _ { n }$ and compute $B _ { n }$ using information available at $t _ { n } .$
4: Compute prediction weights from the retained history using (4).
5: If $X _ { t _ { n } } \in S$ choose each expert’s stake $\lambda _ { n } ^ { a }$ using (6) , or (5), otherwise $\lambda _ { n } ^ { a } = 0 .$
6: Observe $X _ { t _ { n + 1 } }$ and compute ${ \overline { { L } } } _ { n }$ and $W _ { n } .$
7: Update the combined accounts and reserve using (20).
8: if $\begin{array} { r } { R _ { n + 1 } + \sum _ { a \in { \cal A } } A _ { n + 1 } ^ { a } \geq 1 / \delta } \end{array}$ then
9: return alarm at index $n + 1 .$
10: end if
11: Append $( t _ { n } , X _ { t _ { n } } , W _ { n } )$ to the history and retain its most recent m entries.
12: end for
```

Combining accounts. Repeated funding need not require storing a separate account for every starting index. Suppose that every mixture uses the same finite set of experts $\mathcal { A } _ { s } = \mathcal { A } .$ with $| { \mathcal { A } } | = K$ , and the same nonnegative weights $\eta _ { s , a } = \eta _ { a }$ satisfying $\textstyle \sum _ { a \in { \mathcal { A } } } \eta _ { a } = 1$ . Each expert keeps its kernels, scales, and stake strategy fixed and uses the shared retained history. For the moment and log-return strategies above, its stake therefore does not depend on the starting index:

$$
\lambda _ { s , n } ^ { a } = \lambda _ { n } ^ { a } \qquad \mathrm { f o r ~ e v e r y ~ } s \leq n .
$$

The accounts of expert a may hold diferent amounts of capital, but the next score multiplies each of them by the same factor $1 + \lambda _ { n } ^ { a } W _ { n }$ . We can therefore store their combined capital:

$$
A _ { n } ^ { a } : = \sum _ { s = 0 } ^ { n - 1 } c _ { s } \eta _ { a } G _ { s , n } ^ { a } , \qquad R _ { n } : = 1 - C _ { n } ,
$$

where $R _ { n }$ is the remaining reserve. Funding at index n adds $c _ { n } \eta _ { a }$ to the combined account of expert a and removes $c _ { n }$ from the reserve. Applying the common betting factor gives

$$
A _ { n + 1 } ^ { a } = \bigl ( A _ { n } ^ { a } + c _ { n } \eta _ { a } \bigr ) \bigl ( 1 + \lambda _ { n } ^ { a } W _ { n } \bigr ) , \qquad R _ { n + 1 } = R _ { n } - c _ { n } ,\tag{20}
$$

with $A _ { 0 } ^ { a } = 0 , R _ { 0 } = 1$ , and total wealth $\begin{array} { r } { M _ { n } = R _ { n } + \sum _ { a \in \mathcal { A } } A _ { n } ^ { a } } \end{array}$ . Multiplication distributes over the individual account balances, so this update gives exactly the same total wealth and alarm time as maintaining those accounts separately. The optimisation requires the combined accounts to use identical stakes. Accounts whose scales or prediction histories depend on their starting indices must generally remain separate.

The history stores each score with its starting state because spatial prediction compares the states from which the corresponding transitions began. Removing an old observation from the history afects future stake choices but leaves accumulated wealth unchanged. If observations end upon reaching a target or another observed termination condition, the monitor processes the final completed interval before stopping.

Computational cost. For states in $\mathbb { R } ^ { d } .$ , the retained history requires $O ( m d )$ storage and the combined accounts require $O ( K )$ . Processing prediction weights one expert at a time therefore gives total storage $O ( m d + K )$ for the betting implementation. At each observation, distances to the retained states can be computed once and shared across experts, costing O(md). Each moment strategy then computes two weighted sums in $O ( m )$ time, giving $O ( m d + K m )$ work per observation. For log-return stakes, each derivative evaluation scans the retained history. Using $J$ bisection iterations therefore gives $O ( m d + K m J )$ work. Finite bisection approximates the maximiser, but any returned stake in $[ 0 , { \overline { { \lambda } } } ]$ preserves false-alarm control. These bounds exclude the resources required by the certificate, score-bound, and funding routines.

Recursive updates for unlimited history. For uniform or purely temporal geometric weights, an expert using the moment strategy can use the entire history without storing individual observations. Since its temporal scale is fixed across launches, write $r _ { \mathbb { T } } ^ { a } : = r _ { \mathbb { T } , s } ^ { a } ,$ independently of s. For the geometric kernel $k ( u ) = 2 ^ { - u }$ , set $\gamma ^ { a } = 2 ^ { - h / r _ { \mathbb { T } } ^ { a } }$ ; for uniform weighting, set $\gamma ^ { a } = 1$ . The geometric weights are $2 ^ { - ( t _ { n } - t _ { i } ) / r _ { \mathbb { T } } ^ { a } } = ( \gamma ^ { a } ) ^ { n - i }$ . Rescaling them to $( \gamma ^ { a } ) ^ { n - 1 - i }$ leaves the moment stake unchanged, since the common factor cancels in the ratio of weighted sums. Starting from $S _ { 0 } ^ { a } = U _ { 0 } ^ { a } = H _ { 0 } ^ { a } = 0$ , we maintain the resulting sums through

$$
S _ { n + 1 } ^ { a } = \gamma ^ { a } S _ { n } ^ { a } + W _ { n } , \quad \quad \quad U _ { n + 1 } ^ { a } = \gamma ^ { a } U _ { n } ^ { a } + W _ { n } ^ { 2 } , \quad \quad \quad H _ { n + 1 } ^ { a } = \gamma ^ { a } H _ { n } ^ { a } + 1 .
$$

For $n \geq 1$ , the weighted mean and second moment are $S _ { n } ^ { a } / H _ { n } ^ { a }$ and $U _ { n } ^ { a } / H _ { n } ^ { a }$ . The moment stake uses $S _ { n } ^ { a }$ and $U _ { n } ^ { a }$ directly, so maintaining $H _ { n } ^ { a }$ is optional. When $\gamma ^ { a } = 1$ , these recursions give the ordinary sums and $H _ { n } ^ { a } = n$ ; when $\gamma ^ { a } < 1$ , each update discounts earlier observations. For $K$ such experts, maintaining these sums requires $O ( K )$ work per observation and $O ( K )$ storage, independently of the history length. This optimisation does not generally extend to spatial weights, which change with the current state, or to exact log-return optimisation, whose objective cannot be recovered from these sums alone.

## E Experimental setup

Prediction, stakes and funding. Uniform prediction (U) assigns equal weight to all retained scores. Temporal prediction (T) favours recent scores, while spatial prediction (S) favours scores whose starting states are close to the current state. Temporal and spatial prediction use the linear kernel $k ( u ) = ( 1 - u ) _ { - }$ <sub>+</sub> with fixed scales.

For the synthetic benchmarks, we retain 16384 observations. We use five scales, labelled XS–XL, calibrated to reference efective sample sizes 32, 128, 512, 2048, 8192, where

$$
\mathrm { E S S } ( w ) = \frac { ( \sum _ { i } w _ { i } ) ^ { 2 } } { \sum _ { i } w _ { i } ^ { 2 } } .
$$

The temporal cutofs are approximately 43.166, 171.167, 683.167, 2731.167 and 10923.167 observations. We evaluate the individual scales, an equal mixture, a centred mixture with weights (0.1, 0.2, 0.4, 0.2, 0.1), and five equal mixtures that each omit one scale. For the control and neural benchmarks, we use U-L, T-L and S-L with a history of 8192 observations. T-L equally mixes temporal cutofs 64, 128, 512, 2048, 8192, while S-L equally mixes spatial radii 0.00625, 0.025, 0.1, 0.4, 1.6. Log-return optimisation uses endpoint checks followed, when necessary, by 32 bisection steps. We launch accounts every 256 observations. Launch $j \geq 0$ receives $6 / [ \pi ^ { 2 } ( j + 1 ) ^ { 2 } ]$ of the initial capital in the synthetic benchmarks and $1 / [ ( j + 1 ) ( j + 2 ) ]$ in the control and neural benchmarks.

## E.1 Synthetic benchmarks

The synthetic benchmarks isolate diferent sources of detection dificulty (Figures 1–4). We consider six families of processes.

Temporal: The scalar process starts at $X _ { 0 } = 3 2 7 6 9$ and follows

$$
X _ { n + 1 } = X _ { n } - 0 . 0 0 1 + g _ { n } + \xi _ { n } , \qquad \xi _ { n } \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } [ - 0 . 6 , 0 . 6 ] .
$$

With certificate $f ( x ) = x$ and allowance −0.001, the conditional expected corrected increment is $\mathbb { E } [ D _ { n } \mid { \mathcal { F } } _ { n } ] = g _ { n }$ . Persistent violations use $g _ { n } \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 5 \}$ throughout. Delayed violations use $g _ { n } = - 0 . 0 2$ until onset 2048, 4096 or 8192, followed by $g _ { n } = 0 . 0 5$ . Temporary violations use $g _ { n } = 0 . 0 5$ for 256, 1024 or 4096 observations from onset 8192, and $g _ { n } = - 0 . 0 2$ otherwise. Each run lasts 32768 observations.

Spatial: The first cyclic system follows a deterministic circular path with directly prescribed scores. It does not use a controller–certificate pair, and its conditional score mean jumps at the sector boundaries, so it is not globally Lipschitz. Each period starts with 64 observations of mean 0.10 in a fixed sector, followed by observations of mean −0.02. Independent symmetric score noise has amplitude 0.3. We use periods $P \in \{ 5 1 2 , 2 0 4 8 , 4 0 9 6 \}$ and horizons 8P. Each run therefore contains exactly 512 positive-mean observations; increasing P lengthens the gaps between visits.

The second system is a stochastic ring with noise in both radial position and angular speed. The certificate is the squared radial error, and the allowance is its exact nominal expected increment. The radial contraction is 0.2, radial noise is uniform on $[ - 0 . 0 1 , 0 . 0 1 ]$ , and angular speed has independent uniform modulation of ±20%. A triangular angular envelope with half-width 0.2 reduces the radial feedback by up to 75%. We use the same periods and horizons, with a nominal sector visit of 256 observations. Stochastic return times make the number of observations under disturbance vary across runs.

Hill: We use the robot from the running example. It starts at $X _ { 0 } = 1 0$ and stops upon reaching [0, 1]. Before stopping,

$$
X _ { n + 1 } = X _ { n } - 0 . 0 1 + 0 . 0 2 Z _ { n } , \qquad Z _ { n } \mid \mathcal { G } _ { n } \sim \mathrm { B e r n o u l l i } ( 1 / 4 + a ( 1 - \vert X _ { n } - 5 \vert / 2 ) _ { + } ) .
$$

With certificate $f ( x ) = x$ , allowance −0.005 and loss bound 0.02, the conditional expected score is $a ( 1 - | X _ { n } - 5 | / 2 ) _ { + }$ . Amplitudes $a \in \{ 0 . 0 6 2 5 , 0 . 1 2 5 , 0 . 2 5 \}$ reduce expected progress at the hill centre by one quarter, one half or entirely. Stronger disturbances also keep the robot on the hill for longer.

For the cyclic systems, increasing the period changes the horizon, the observations available in the retained history and the number of funding opportunities, even though positive exposure is fixed for direct recurrence. Including the scalar, ring and hill references gives 21 settings. Direct recurrence has no separate saved reference. The six example trajectories use seed 970000003 and settings with weak violations, late onset, short duration or long gaps between visits. These figures illustrate individual runs; the heatmaps summarise all seeds.

## E.2 Control benchmarks

Building and Two Tanks adapt benchmarks from PRoTECT (Wooding et al., 2026). The stochastic double-integrator LQR uses a standard quadratic value function (?). Each system has one reference setting and two disturbances, each evaluated at three strengths, giving 21 settings in total. All disturbances act from the start of the run. The certificates and allowances remain fixed.

Building: We reduce heating efectiveness by 8%, 10% or 12%, or make the controller underestimate the temperature by 1, 2 or 3 degrees. In the latter case, the monitor observes the true temperature.

Tanks: We add inlet flow of amplitude 0.2, 0.3 or 0.4, which fades as the downstream level rises from 2.15 to 2.25. Alternatively, we add independent inlet bursts of size 0.5 with probability 0.35, 0.40 or 0.45 per step.

Building and Tanks use bounded triangular noise with the same means and variances as their source models. Their newly synthesised sum-of-squares certificates have degrees two and four, respectively. Independent checks using exact arithmetic validate the nominal allowances 0.04 and 0.07 under these noise distributions. Initial states are uniform on [19.5, 20] for Building and on $[ 4 . 4 5 , 4 . 5 5 ] \times [ 1 . 4 5 , 1 . 5 5 ]$ for Tanks.

LQR: The initial state lies in $[ 1 , 2 ] \times [ - 0 . 1 , 0 . 1 ]$ . The nominal acceleration difusion is 0.1. We sample every 0.1 time units using the exact Gaussian transition and use the exact nominal sampled certificate increment as the allowance. Disturbances either reduce feedback by 5%, 10% or 20%, or multiply the noise amplitude by 1.05, 1.10 or 1.20.

All control runs have a limit of 32768 observations. Building and Tanks also stop upon entering an unsafe set or leaving the domain; LQR stops upon leaving $[ - 1 0 , 1 0 ] ^ { 2 }$ . We compute conditional moments exactly along each recorded trajectory. These determine the fraction of runs encountering a violation and the expected-score summaries (Figures 12–14).

## E.3 Neural controller–certificate pairs

We retain 16 original controller–certificate pairs from LogRASM: four CollisionAvoidance pairs, six Linear layout-0 pairs, two Linear layout-1 pairs, two Pendulum pairs and two Triple-integrator pairs. The controller and certificate parameters remain fixed. We consider three disturbance magnitudes per setting. Initial states are sampled from fixed pools away from the target, unsafe sets and domain boundaries, with at least 20% headroom below the certificate threshold. Starts are shared across disturbance strengths and pairs within the same layout.

To obtain longer trajectories, we multiply the source drift increments by 1/20 and the direct plant-noise amplitudes by $1 / \sqrt { 2 0 }$ . The baseline controller efectiveness is 0.1 for Collision-Avoidance, 0.5 and 0.25 for the two Linear layouts, 1 and 0.5 for the ordinary and logarithmic Pendulum pairs, and 1 for Triple integrator.

For CollisionAvoidance, we add terrain-dependent slips with peak probabilities 0.25, 0.35 and 0.50. For the Linear systems, we add fluctuating sensor ofsets with amplitudes 0.30, 0.35 and 0.40. For Pendulum, random torque pushes the system away from the upright position, with amplitudes 0.075, 0.10 and 0.125. For Triple integrator, we add a positive actuator bias with amplitudes 0.06, 0.08 and 0.10. Paths stop upon reaching the target, entering an unsafe set, leaving the domain, reaching the certificate threshold or completing 20480 observations.

Let $C = 1 / ( 1 - \rho )$ . For an ordinary certificate output V, we use $f ( x ) = \operatorname* { m i n } \{ V ( x ) , C \}$ . For a logarithmic output $L ,$ we use $\begin{array} { r } { f ( x ) = \exp ( \operatorname* { m i n } \{ L ( x ) , \log C \} ) } \end{array}$ . We evaluate each recorded path under three decrease requirements, $\epsilon \in \lbrace 0 , 0 . 0 0 0 5 , 0 . 0 0 1 \rbrace$ . The allowance is −ϵ for ordinary certificates and $( e ^ { - \epsilon } - 1 ) f ( x )$ for exponentiated logarithmic certificates. Thus, $\epsilon = 0$ tests expected nonincrease; positive values require additive or multiplicative decrease, respectively. All three conditions use the same predictable loss bound and the same physical trajectories (Figures 15 and 16).

## F Synthetic

![](images/f8ec3d1419691389c5adc8099edffa01f4890f746f41fb99931fbca2f8f2f64f.jpg)  
All synthetic benchmarks — first-alarm percentages 21 settings · 50 seeds per setting · shared 0–100% colour scale

![](images/569c816251fac071618e76968faeab5f94990ed58c1398ac1dfe0f1923f6e331.jpg)  
Figure 1: Detection rates for the synthetic benchmarks, with 21 settings, 50 monitors and 50 paired runs per setting. Cells show the percentage of runs with an eligible first alarm; reference rows count any first alarm. The panels separate moment (M) and log-return (L) stakes. U uses uniform prediction, while T and S use temporal and spatial prediction. XS–XL denote individual scales; Eq is their equal mixture; Ctr uses weights (0.1, 0.2, 0.4, 0.2, 0.1). A minus label indicates that the corresponding scale is omitted from an otherwise equal mixture. All monitors use wealth threshold 20. The panels share a colour scale from 0 to 100%. NA denotes incomplete data.

![](images/1f3166a786f9bb4e8655d7062e58c7492aab3bb043e79e84fa7d9d9d391f2577.jpg)

All synthetic benchmarks — 50% all-run delay 21 settings · 50 seeds per setting · NR = quantile not reached; NA = incomplete; — = reference (delay not defined)  
![](images/548cf8efb80e050af4519cc3910a9c0f4ef8e1c68fca6ca9051b74946e37dc2c.jpg)  
Figure 2: The 50% detection delays for the synthetic monitors in Figure 1. Each value is the first delay by which 25 of all 50 runs have eligible first alarms, measured in observations since the declared onset or first entry into the region with positive conditional mean. Nondetections, short paths and early first alarms remain in the denominator. A later crossing does not replace an early first alarm. NR denotes a median that was not reached, NA incomplete data, and a dash an undefined reference delay. Both panels use the same colour scale.

Detection difference: log − moment  
![](images/9743f4c44f1c9a87f56b4f46fae40c948f2aecc0f6dd6dd99f716b3e7fec3943.jpg)  
Figure 3: Diferences in synthetic detection rates between log-return and moment stakes, computed as L−M in percentage points. Each comparison uses the same prediction rule and the same 50 paths. Positive green cells indicate more alarms with log-return stakes; negative cells indicate more alarms with moment stakes. For reference rows, more alarms mean a higher false-alarm rate. Column labels follow Figure 1. The comparison includes the diferent stake caps, 0.99 for L and 0.5 for M. These diferences are descriptive; no significance test is applied.

50% delay advantage: moment − log  
![](images/1e1a0605543d37c7e0d8b1424b12f5635cfcdddf64775e97168a30548978bd92.jpg)  
Figure 4: Diferences in synthetic median detection delays, computed as $Q _ { \mathrm { 0 . 5 } } ^ { \mathrm { M } } { - Q _ { \mathrm { 0 . 5 } } ^ { \mathrm { L } } }$ in observations. Positive green values indicate earlier detection with log-return stakes when both strategies reach 25 eligible first alarms among all 50 runs. L only and M only indicate that only the named strategy reaches this fraction; NR means neither does. Reference delays are undefined (Ref.), and NA denotes incomplete data. No artificial delays are assigned to non-numeric cells. The comparisons use the same prediction rules and paths, with the respective stake caps of each strategy.

![](images/aa938d7453df4a2e48b23b2dc3913534ad73ef07841d409853241a41b79636cc.jpg)  
Figure 5: Persistent violation with $g = 0 . 0 1$ and seed 970000003. From top to bottom: the scalar state; mean increase ${ g } ;$ the exact expected certificate increment $g - 0 . 0 0 1 = 0 . 0 0 9$ and allowance −0.001; and monitor wealth on a logarithmic scale. Uniform noise on $[ - 0 . 6 , 0 . 6 ]$ makes individual increments noisy despite the positive conditional mean of the corrected increment. U uses uniform prediction, while T-equal and S-equal mix the five XS–XL experts equally. M and $\mathrm { L }$ denote moment and log-return stakes. The dotted line marks wealth threshold 20. All 32768 observations are shown, including those after alarms.

![](images/1236c5ccd0bc3934b288554c98b1a80a7918dac709a0f36fb83af0bd040c0db9.jpg)  
Figure 6: Delayed violation with onset 8192 and seed 970000003. The conditional mean of the corrected increment changes from −0.02 to 0.05 and remains positive. From top to bottom: the state; mean increase 0.07; the expected certificate increment, changing from −0.021 to 0.049, and allowance −0.001; and logarithmic monitor wealth. Temporal prediction reduces the influence of observations collected before onset as new scores arrive. U, T-equal and S-equal are shown with both M and L stakes, as in Figure 5. The wealth threshold is 20. Neither the history nor wealth is reset at onset.

![](images/802ae0fa5848450708b407aad97b9cf9397ed0f27f3721a426de8848392854dc.jpg)  
Figure 7: Temporary violation lasting 256 observations from onset 8192, with seed 970000003. The conditional mean of the corrected increment is 0.05 during the disturbance and −0.02 before and after it. From top to bottom: the state; temporary mean increase 0.07; the expected certificate increment and allowance −0.001; and logarithmic wealth for U and the equal T/S mixtures with M and L stakes. None of the six displayed monitors reaches the dotted threshold 20. The path is shown through observation 32768, including recovery. The aggregate detection summaries also count eligible first alarms occurring after recovery.

![](images/e6721778438cecea27e2b1b760d378d146c0d99e406f74f2ff1acb2efa477530.jpg)  
Figure 8: Direct recurrence with period P = 4096 and seed 970000003. Each of eight visits to a fixed circular sector contains 64 observations with score mean 0.10. Between visits, the mean is −0.02. Independent symmetric score noise has amplitude 0.3. From top to bottom: the two state coordinates; local mean increase 0.12; the prescribed conditional score mean relative to zero; and logarithmic wealth for U and the equal T/S mixtures with M and L stakes. The 32768 observations include exactly 512 with positive conditional mean. Spatial prediction can reuse observations from earlier visits despite the long gaps between them. The dotted line marks wealth threshold 20.

![](images/d698f285e0b18d4f887a32e27a4d5dc6c4f3303b46b6d24261176523edb4ea41.jpg)  
Figure 9: Stochastic ring with nominal period $P = 4 0 9 6$ and seed 970000003. From top to bottom: radial error relative to the unit circle; the state-dependent reduction in restoring feedback, reaching 0.75 in the afected sector; the exact expected certificate increment minus its nominal allowance; and logarithmic monitor wealth. The certificate is the squared radial error. Radial and angular noise make return times and violation exposure vary across runs. U and the equal $\mathrm { T / S }$ mixtures are shown with M and L stakes. The dotted line marks wealth threshold 20. The expected-increment condition can be violated even while the process remains close to the circle. All observations are retained, including those after alarms.

![](images/5118a2180daeea04fa8a202b60a250cc0efb6fc219310ab36a5c892342c2d0b1.jpg)  
Figure 10: Hill disturbance with amplitude $a = 0 . 0 6 2 5$ and seed 970000003. From top to bottom: distance to the target [0, 1], with the hill region (3, 7) shaded; additional slip probability $a ( 1 - | x - 5 | / 2 ) _ { + }$ ; the exact expected certificate increment and required allowance −0.005; and logarithmic wealth for U and the equal $\mathrm { T / S }$ mixtures with M and L stakes. $\mathrm { A t } \ x = 5 .$ the expected increment is −0.00375, a 25% reduction in progress. The robot therefore still approaches the target in expectation while violating the required rate of decrease. The dotted line marks wealth threshold 20. The run ends when the robot reaches the target, not when a monitor raises an alarm.

## Synthetic processes, ring and hill Normalised expected-score summaries

<table><tr><td rowspan=1 colspan=6>Positive         Signed      Mean when   Mean positiveobservations       mean         positive          partScalar</td></tr><tr><td rowspan=1 colspan=2>Scalar nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Persistent: 0.01</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.00999</td><td rowspan=1 colspan=1>0.00999</td><td rowspan=1 colspan=1>0.00999</td></tr><tr><td rowspan=1 colspan=2>Persistent: 0.02</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=2>Persistent: 0.05</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=2>Delayed: 2048</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=2>Delayed: 4096</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=2>Delayed: 8192</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=2>Temporary: 256</td><td rowspan=1 colspan=1>1.0%</td><td rowspan=1 colspan=1>-0.0193</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00052</td></tr><tr><td rowspan=1 colspan=2>Temporary: 1024</td><td rowspan=1 colspan=1>4.2%</td><td rowspan=1 colspan=1>-0.0171</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00208</td></tr><tr><td></td><td rowspan=1 colspan=1>Temporary: 4096</td><td rowspan=1 colspan=1>16.7%</td><td rowspan=1 colspan=1>-0.00833</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00833</td></tr><tr><td></td><td rowspan=1 colspan=1>Direct recurrence</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1>Direct recurrence: 512</td><td rowspan=1 colspan=1>12.5%</td><td rowspan=1 colspan=1>-0.005</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.0125</td></tr><tr><td></td><td rowspan=1 colspan=1>Direct recurrence: 2048</td><td rowspan=1 colspan=1>3.1%</td><td rowspan=1 colspan=1>-0.0163</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.00313</td></tr><tr><td></td><td rowspan=1 colspan=1>Direct recurrence: 4096</td><td rowspan=1 colspan=1>1.6%</td><td rowspan=1 colspan=1>-0.0181</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.00156</td></tr><tr><td></td><td rowspan=1 colspan=1>Ring</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Ring nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stochastic ring: 512</td><td rowspan=1 colspan=1>49.9%</td><td rowspan=1 colspan=1>0.0552</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.0552</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stochastic ring: 2048</td><td rowspan=1 colspan=1>12.5%</td><td rowspan=1 colspan=1>0.0136</td><td rowspan=1 colspan=1>0.109</td><td rowspan=1 colspan=1>0.0136</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Stochastic ring: 4096</td><td rowspan=1 colspan=1>6.3%</td><td rowspan=1 colspan=1>0.00682</td><td rowspan=1 colspan=1>0.109</td><td rowspan=1 colspan=1>0.00682</td></tr><tr><td rowspan=1 colspan=2>Hill</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Hill nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Hill: 0.0625</td><td rowspan=1 colspan=1>69.5%</td><td rowspan=1 colspan=1>0.0229</td><td rowspan=1 colspan=1>0.0329</td><td rowspan=1 colspan=1>0.0229</td></tr><tr><td rowspan=1 colspan=2>Hill: 0.125</td><td rowspan=1 colspan=1>73.8%</td><td rowspan=1 colspan=1>0.051</td><td rowspan=1 colspan=1>0.0695</td><td rowspan=1 colspan=1>0.051</td></tr><tr><td rowspan=1 colspan=2>Hill: 0.25</td><td rowspan=1 colspan=1>86.7%</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.157</td></tr></table>

Cell = median of 50 per-run summaries. IQRs: expected\_scores\_summary.csv.  
Interval: onset/entry to stop; references: full path. Recovery is included.  
NA: incomplete setting. Hatched dash: no positive observations.  
Shared column scales across these maps; signed colours centre on zero.

Figure 11: Normalised expected-score summaries for the synthetic benchmarks. Each cell is the median of 50 quantities computed separately for each run: the fraction of observations with $\mu _ { n } > 0 ;$ , the signed mean of $\mu _ { n }$ , its mean over positive-mean observations, and the mean of $( \mu _ { n } ) _ { + }$ over all observations. Summaries start at onset or first entry and include recovery and later excursions. References use the full path. Each run receives equal weight. A hatched dash marks an undefined mean over positive observations; NA denotes incomplete data. Column colour scales match the control heatmap, with the signed scale centred at zero.

## G Control

All control benchmarks — first-alarm percentages 21 settings · 50 planned seeds per setting · NA = incomplete
<table><tr><td colspan="5">Validated nominal allowance</td></tr><tr><td></td><td>U-L</td><td>T-L</td><td>S-L</td><td>Viol.</td></tr><tr><td>Building — reference</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Building — actuator loss 0.08</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Building — actuator loss 0.1</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Building — actuator loss 0.12</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Building — sensor bias 1</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Building — sensor bias 2</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Building — sensor bias 3</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Tanks — reference</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Tanks — level inlet 0.2</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Tanks — level inlet 0.3</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Tanks — level inlet 0.4</td><td>0</td><td>0</td><td>100</td><td>100</td></tr><tr><td>Tanks — random inlet 0.35</td><td>0</td><td>0</td><td>0</td><td>100</td></tr><tr><td>Tanks — random inlet 0.4</td><td>0</td><td>0</td><td>10</td><td>100</td></tr><tr><td>Tanks — random inlet 0.45</td><td>0</td><td>0</td><td>68</td><td>100</td></tr><tr><td>LQR — reference</td><td>0</td><td>0</td><td>2</td><td>0</td></tr><tr><td>LQR — actuator loss 0.05</td><td>38</td><td>24</td><td>32</td><td>100</td></tr><tr><td>LQR — actuator loss 0.1</td><td>96</td><td>84</td><td>98</td><td>100</td></tr><tr><td>LQR — actuator loss 0.2</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>LQR — noise multiplier 1.05</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>LQR — noise multiplier 1.1</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>LQR — noise multiplier 1.2</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td colspan="4"></td></tr><tr><td rowspan="3">0</td><td></td><td></td><td colspan="2"></td></tr><tr><td>20</td><td>40 60</td><td>80</td><td>100</td></tr><tr><td colspan="2">Run fraction (%)</td><td></td><td></td></tr></table>

Figure 12: First-alarm percentages for Building, Two Tanks and LQR. Each system has a reference setting and two disturbances, each evaluated at three strengths. U-L, T-L and S-L use uniform, temporal and spatial prediction with log-return stakes and wealth threshold 20. Every cell uses all 50 paired runs. The Viol. column gives the percentage of trajectories encountering at least one positive conditional excess, computed from exact model moments with tolerance $1 0 ^ { - 1 2 }$ in raw units. This difers from both the alarm rate and the fraction of observations with positive conditional excess. Reference alarms measure false alarms under the validated nominal conditions.

All control benchmarks — 50% all-run delay 21 settings · 50 planned seeds per setting · NA = incomplete; NR = not reached  
![](images/e5d6075d03f94a19af9cf96f46bbcc3d8d1c4eecb26f318233b5dface2d1b693.jpg)  
Figure 13: The 50% detection delays for the control settings and log-return monitors in Figure 12. Disturbances act from the start, so each entry is the number of observations until 25 of all 50 runs have raised their first alarm. NR means this fraction was not reached, and NA denotes incomplete data. Nondetections remain in the denominator. Reference delays also start at the beginning of the run and concern false alarms. LQR observations are spaced by 0.1 physical time units; all entries are reported in observations.

Control systems Normalised expected-score summaries
<table><tr><td rowspan=1 colspan=6>Positive        Signed      Mean when   Mean positiveobservations       mean         positive          partBuilding</td></tr><tr><td rowspan=1 colspan=2>Nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>-0.051</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Loss d=0.08</td><td rowspan=1 colspan=1>24.9%</td><td rowspan=1 colspan=1>-0.0293</td><td rowspan=1 colspan=1>0.0151</td><td rowspan=1 colspan=1>0.00377</td></tr><tr><td rowspan=1 colspan=2>Loss d=0.1</td><td rowspan=1 colspan=1>33.6%</td><td rowspan=1 colspan=1>-0.0166</td><td rowspan=1 colspan=1>0.0285</td><td rowspan=1 colspan=1>0.00956</td></tr><tr><td></td><td rowspan=1 colspan=1>Loss d=0.12</td><td rowspan=1 colspan=1>37.6%</td><td rowspan=1 colspan=1>-0.00971</td><td rowspan=1 colspan=1>0.0309</td><td rowspan=1 colspan=1>0.0116</td></tr><tr><td></td><td rowspan=1 colspan=1>Bias β=1</td><td rowspan=1 colspan=1>23.2%</td><td rowspan=1 colspan=1>-0.0318</td><td rowspan=1 colspan=1>0.0132</td><td rowspan=1 colspan=1>0.00306</td></tr><tr><td></td><td rowspan=1 colspan=1>Bias β=2</td><td rowspan=1 colspan=1>32.7%</td><td rowspan=1 colspan=1>-0.0186</td><td rowspan=1 colspan=1>0.0288</td><td rowspan=1 colspan=1>0.00944</td></tr><tr><td></td><td rowspan=1 colspan=1>Bias β=3</td><td rowspan=1 colspan=1>37.0%</td><td rowspan=1 colspan=1>-0.0113</td><td rowspan=1 colspan=1>0.0327</td><td rowspan=1 colspan=1>0.0121</td></tr><tr><td></td><td rowspan=1 colspan=1>Tanks</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1>Nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>-0.0642</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td></td><td rowspan=1 colspan=1>Inlet a=0.2</td><td rowspan=1 colspan=1>16.1%</td><td rowspan=1 colspan=1>-0.0363</td><td rowspan=1 colspan=1>0.0141</td><td rowspan=1 colspan=1>0.00227</td></tr><tr><td></td><td rowspan=1 colspan=1>Inlet a=0.3</td><td rowspan=1 colspan=1>25.1%</td><td rowspan=1 colspan=1>-0.0351</td><td rowspan=1 colspan=1>0.0368</td><td rowspan=1 colspan=1>0.00923</td></tr><tr><td></td><td rowspan=1 colspan=1>Inlet a=0.4</td><td rowspan=1 colspan=1>24.4%</td><td rowspan=1 colspan=1>-0.0338</td><td rowspan=1 colspan=1>0.0629</td><td rowspan=1 colspan=1>0.0154</td></tr><tr><td></td><td rowspan=1 colspan=1>Burst p=0.35</td><td rowspan=1 colspan=1>5.5%</td><td rowspan=1 colspan=1>-0.0172</td><td rowspan=1 colspan=1>0.00561</td><td rowspan=1 colspan=1>0.00031</td></tr><tr><td></td><td rowspan=1 colspan=1>Burst p=0.4</td><td rowspan=1 colspan=1>8.2%</td><td rowspan=1 colspan=1>-0.0149</td><td rowspan=1 colspan=1>0.00595</td><td rowspan=1 colspan=1>0.000482</td></tr><tr><td></td><td rowspan=1 colspan=1>Burst p=0.45</td><td rowspan=1 colspan=1>11.0%</td><td rowspan=1 colspan=1>-0.013</td><td rowspan=1 colspan=1>0.00615</td><td rowspan=1 colspan=1>0.000677</td></tr><tr><td></td><td rowspan=1 colspan=1>LQR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1>Nominal</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td></tr><tr><td></td><td rowspan=1 colspan=1>Loss d=0.05</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.00955</td><td rowspan=1 colspan=1>0.00955</td><td rowspan=1 colspan=1>0.00955</td></tr><tr><td></td><td rowspan=1 colspan=1>Loss d=0.1</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.0191</td><td rowspan=1 colspan=1>0.0191</td><td rowspan=1 colspan=1>0.0191</td></tr><tr><td></td><td rowspan=1 colspan=1>Loss d=0.2</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.0383</td><td rowspan=1 colspan=1>0.0383</td><td rowspan=1 colspan=1>0.0383</td></tr><tr><td></td><td rowspan=1 colspan=1>Noise v=1.05</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.0298</td><td rowspan=1 colspan=1>0.0298</td><td rowspan=1 colspan=1>0.0298</td></tr><tr><td></td><td rowspan=1 colspan=1>Noise v=1.1</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.0581</td><td rowspan=1 colspan=1>0.0581</td><td rowspan=1 colspan=1>0.0581</td></tr><tr><td rowspan=1 colspan=2>Noise v=1.2</td><td rowspan=1 colspan=1>100.0%</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.111</td></tr></table>

Cell = median of 50 per-run summaries. IQRs: expected\_scores\_summary.csv.  
Interval: onset/entry to stop; references: full path. Recovery is included.  
NA: incomplete setting. Hatched dash: no positive observations.  
Shared column scales across these maps; signed colours centre on zero.

Figure 14: Normalised expected-score summaries for the control benchmarks, computed from exact conditional moments along each recorded path. Cells give medians of 50 equally weighted per-run summaries: the fraction of positive-mean observations, the signed mean, the mean over positive-mean observations, and the mean positive part. The last quantity includes all observations in its denominator. A negative signed mean can therefore coexist with local violations. Positivity requires a raw conditional excess above 10<sup>−12</sup>. Reference and disturbed runs use their full paths, including observations after alarms. A hatched dash marks an undefined mean over positive observations; NA denotes incomplete data. Column colour scales match Figure 11.

All neural benchmarks — first-alarm percentages 64 settings · 50 planned seeds per setting · NA = incomplete  
![](images/fba3ab95dab212892eb18c57f4d6a4a3a6771501eb433faee06dbbf3e09b26a4.jpg)  
Figure 15: First-alarm percentages for the neural benchmarks. Each of the 16 original controller– certificate pairs has a reference setting and three disturbance strengths. Each row uses 50 trajectories shared by U-L, T-L and S-L and by the conditions $\epsilon = 0 , 0 . 0 0 0 5 , 0 . 0 0 1$ . The allowance is −ϵ for ordinary certificates and $( e ^ { - \epsilon } - 1 ) f ( x )$ for exponentiated logarithmic certificates. All monitors use wealth threshold 20. Question marks in Viol. indicate that the conditional mean estimates at selected states do not determine the exact fraction of runs encountering a violation. All three conditions remain unverified under the adapted dynamics, including in the reference settings. Reference alarms therefore do not measure false-alarm rates under a verified null.

All neural benchmarks — 50% all-run delay  
64 settings · 50 planned seeds per setting · NA = incomplete; NR = not reached
<table><tr><td rowspan=6 colspan=14>Nonincrease           Decrease ε=0.0005         Decrease ε=0.001T-L                                  U-L     T-L     S-LCA: ordinary ρ=0.99 weighted — reference   NR    NR    NR            NR     NRCA: ordinary ρ=0.99 weighted — valley 0.25                               3241                         1590CA: ordinary ρ=0.99 weighted — valley 0.35         NR           3068CA: ordinary ρ=0.99 weighted — valley 0.5         NR</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>32563241</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2992</td><td rowspan=1 colspan=3>165616111590</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>30682938</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2887</td><td rowspan=1 colspan=3>162715721555</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>31072936</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2929</td><td rowspan=1 colspan=3>163615731567</td></tr><tr><td rowspan=4 colspan=3>CA: ordinary ρ=0.9 weighted — referenceCA: ordinary ρ=0.9 weighted — valley 0.25CA: ordinary ρ=0.9 weighted — valley 0.35CA: ordinary ρ=0.9 weighted — valley 0.5</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=3>NR     NR    NR</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=3>1565    1554   1581</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15063</td><td rowspan=1 colspan=2>2997    3018</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2929</td><td rowspan=1 colspan=3>1592   1573   1586</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>13334</td><td rowspan=1 colspan=2>3177    3216</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3090</td><td rowspan=1 colspan=3>1666    1654    1662</td></tr><tr><td rowspan=1 colspan=3>CA: log ρ=0.8 — reference</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=3>NR    NR    NR</td></tr><tr><td rowspan=3 colspan=3>CA: log ρ=0.8 — valley 0.25CA: log ρ=0.8 — valley 0.35CA: log ρ=0.8 — valley 0.5</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>9163</td><td rowspan=1 colspan=1>10967</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7825</td><td rowspan=1 colspan=3>3920    4066   3904</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>7314</td><td rowspan=1 colspan=1>7972</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7137</td><td rowspan=1 colspan=3>3735    3867    3761</td></tr><tr><td rowspan=3 colspan=3>CA: log ρ=0.999999 — referenceCA: log ρ=0.999999 — valley 0.25CA: log ρ=0.999999 — valley 0.35CA: log ρ=0.999999 — valley 0.5</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=3>NR    NR    NR</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=3>NR     NR     NR</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10927</td><td rowspan=1 colspan=1>9416</td><td rowspan=1 colspan=1>9953</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6786</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9898</td><td rowspan=1 colspan=1>8925</td><td rowspan=1 colspan=2>9739</td><td rowspan=1 colspan=1>6455</td><td rowspan=1 colspan=2>4365    4453</td><td rowspan=1 colspan=1>3984</td></tr><tr><td rowspan=1 colspan=3>Linear 0: ordinary ρ=0.99 — sensor 0.3</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>12080</td><td rowspan=1 colspan=1>14706</td><td rowspan=1 colspan=2>19750</td><td rowspan=1 colspan=1>8893</td><td rowspan=1 colspan=2>10134   10975</td><td rowspan=1 colspan=1>6847</td></tr><tr><td rowspan=2 colspan=3>Linear 0: ordinary ρ=0.99 — sensor 0.35Linear 0: ordinary ρ=0.99 — sensor 0.4</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>18263</td><td rowspan=1 colspan=2>NR     NR</td><td rowspan=1 colspan=1>14322</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR     NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR     NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.99 weighted — sensor 0.3</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15268</td><td rowspan=1 colspan=1>13275</td><td rowspan=1 colspan=2>10962</td><td rowspan=1 colspan=1>7631</td><td rowspan=1 colspan=2>6921    5408</td><td rowspan=1 colspan=1>4811</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.99 weighted — sensor 0.4</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10956</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>9848</td><td rowspan=1 colspan=2>NR     NR</td><td rowspan=1 colspan=1>8874</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.9 — reference</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR     NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.9 — sensor 0.3</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16434</td><td rowspan=1 colspan=1>9095</td><td rowspan=1 colspan=2>8287</td><td rowspan=1 colspan=1>6838</td><td rowspan=1 colspan=2>4672    4072</td><td rowspan=1 colspan=1>3908</td></tr><tr><td rowspan=3 colspan=3>Linear 0: ordinary ρ=0.9 — sensor 0.4</td><td rowspan=1 colspan=4>Linear 0: ordinary ρ=0.9 — sensor 0.35</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17997</td><td rowspan=1 colspan=1>12905</td><td rowspan=1 colspan=1>12648</td><td rowspan=1 colspan=1>9487</td></tr><tr><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19766</td><td rowspan=1 colspan=1>17318</td><td rowspan=1 colspan=2>17987</td><td rowspan=1 colspan=1>12186</td><td rowspan=1 colspan=2>8634    8648</td><td rowspan=1 colspan=1>7384</td></tr><tr><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.9 weighted — reference</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.9 weighted — sensor 0.3</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>7905</td><td rowspan=1 colspan=2>7307</td><td rowspan=1 colspan=1>6877</td><td rowspan=1 colspan=2>4073    3703</td><td rowspan=1 colspan=1>3671</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: ordinary ρ=0.9 weighted — sensor 0.35</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>9135</td><td rowspan=1 colspan=2>8853</td><td rowspan=1 colspan=1>8174</td><td rowspan=1 colspan=2>4680    4347</td><td rowspan=1 colspan=1>4285</td></tr><tr><td></td><td rowspan=1 colspan=2>Linear 0: log ρ=0.8 — reference</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR    NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 0: log ρ=0.8 — sensor 0.35</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 0: log ρ=0.8 — sensor 0.4</td><td rowspan=1 colspan=3>NR     NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>7819</td><td rowspan=1 colspan=2>7602</td><td rowspan=1 colspan=1>7271</td><td rowspan=1 colspan=1>3990</td><td rowspan=1 colspan=1>3800</td><td rowspan=1 colspan=1>3780</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 0: log ρ=0.999999 — reference</td><td rowspan=1 colspan=3>NR    NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 0: log ρ=0.999999 — sensor 0.3</td><td rowspan=1 colspan=3>NR    NR</td><td rowspan=1 colspan=1>16106</td><td rowspan=1 colspan=1>13721</td><td rowspan=1 colspan=2>9928</td><td rowspan=1 colspan=1>8281</td><td rowspan=1 colspan=1>6892</td><td rowspan=1 colspan=1>4801</td><td rowspan=1 colspan=1>4631</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 0: log ρ=0.999999 — sensor 0.4</td><td rowspan=1 colspan=3>NR    NR</td><td rowspan=1 colspan=1>13001</td><td rowspan=1 colspan=1>12067</td><td rowspan=1 colspan=2>10033</td><td rowspan=1 colspan=1>7537</td><td rowspan=1 colspan=1>6013</td><td rowspan=1 colspan=1>4714</td><td rowspan=1 colspan=1>4380</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.8 — reference</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.8 — sensor 0.35</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>9897</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.8 — sensor 0.4</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>9533</td><td rowspan=1 colspan=1>7151</td><td rowspan=1 colspan=2>8229</td><td rowspan=1 colspan=1>4821</td><td rowspan=1 colspan=1>3530</td><td rowspan=1 colspan=1>3530</td><td rowspan=1 colspan=1>2751</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.999999 — reference</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.999999 — sensor 0.3</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Linear 1: log ρ=0.999999 — sensor 0.35</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=3 colspan=1>Pendulum:</td><td rowspan=1 colspan=1>Linear 1: log ρ=0.999999 — sensor 0.4</td><td rowspan=1 colspan=1>6452</td><td rowspan=1 colspan=2>606</td><td rowspan=1 colspan=1>601</td><td rowspan=1 colspan=1>4332</td><td rowspan=1 colspan=2>605</td><td rowspan=1 colspan=1>596</td><td rowspan=1 colspan=1>2262</td><td rowspan=1 colspan=1>596</td><td rowspan=1 colspan=1>592</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td></td><td rowspan=1 colspan=1>ordinary ρ=0.9 weighted — gust 0.075</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>4255</td><td rowspan=1 colspan=1>4057</td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>Pendul</td><td rowspan=1 colspan=1>um: ordinary ρ=0.9 weighted — gust 0.1</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>Pendulum:</td><td rowspan=1 colspan=1>ordinary ρ=0.9 weighted — gust 0.125</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=2>NR</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>7079</td><td rowspan=1 colspan=2>5990</td><td rowspan=1 colspan=1>5535</td><td rowspan=1 colspan=1>3762</td><td rowspan=1 colspan=2>3064   2949</td></tr><tr><td rowspan=9 colspan=14>Pendulum: log ρ=0.999999 — gust 0.075 13353   13993   4807    9897    7552    4119   8066   4718   364544002847Triple integrator: log ρ=0.99 — bias 0.06  NR    NR    NR    14555   10297   9769    7757    5471    520411145                                44201686Triple integrator: log ρ=0.9 — bias 0.06  NR    NR    NR    12985   10591   9543    6514    5596   51022500   5000   7500  10000  12500  15000  17500  20000Delay (observations since launch)</td></tr><tr><td rowspan=1 colspan=1>Pendulum: log ρ=0.999999 — gust 0.1</td><td rowspan=1 colspan=3>1236712485</td><td rowspan=1 colspan=1>4400</td><td rowspan=1 colspan=1>9063</td><td rowspan=1 colspan=2>7042</td><td rowspan=1 colspan=1>3688</td><td rowspan=1 colspan=1>7017</td><td rowspan=1 colspan=1>4247</td><td rowspan=1 colspan=1>3144</td></tr><tr><td rowspan=1 colspan=1>Pendulum: log ρ=0.999999 — gust 0.125</td><td rowspan=1 colspan=3>1165612164</td><td rowspan=1 colspan=1>3691</td><td rowspan=1 colspan=1>8514</td><td rowspan=1 colspan=2>6766</td><td rowspan=1 colspan=1>3210</td><td rowspan=1 colspan=1>6565</td><td rowspan=1 colspan=1>4249</td><td rowspan=1 colspan=1>2847</td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.99 — reference</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.99 — bias 0.08</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11145</td><td rowspan=1 colspan=2>8808</td><td rowspan=1 colspan=1>7944</td><td rowspan=1 colspan=1>5615</td><td rowspan=1 colspan=1>4557</td><td rowspan=1 colspan=1>4420</td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.99 — bias 0.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7278</td><td rowspan=1 colspan=2>4313</td><td rowspan=1 colspan=1>3103</td><td rowspan=1 colspan=1>3706</td><td rowspan=1 colspan=1>2030</td><td rowspan=1 colspan=1>1686</td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.9 — reference</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.9 — bias 0.08</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7966</td><td rowspan=1 colspan=1>7293</td><td rowspan=1 colspan=1>4647</td><td rowspan=1 colspan=1>3918</td><td rowspan=1 colspan=1>3791</td></tr><tr><td rowspan=1 colspan=1>Triple integrator: log ρ=0.9 — bias 0.1</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2131</td><td rowspan=1 colspan=1>1770</td></tr></table>

Figure 16: The 50% detection delays for the neural pairs, disturbances and conditions in Figure 15, measured in observations from the start of each run. Each finite entry requires 25 first alarms among all 50 paired paths. Short paths and nondetections remain in the denominator. NR means this fraction was not reached, and NA denotes incomplete data. Paths end at the target, an unsafe set, domain exit, certificate threshold or 20480 observations. Detection therefore depends on both the evidence available and the length of the trajectory. All condition blocks share a colour scale. The diferent conditions change the allowance on the same physical trajectories.