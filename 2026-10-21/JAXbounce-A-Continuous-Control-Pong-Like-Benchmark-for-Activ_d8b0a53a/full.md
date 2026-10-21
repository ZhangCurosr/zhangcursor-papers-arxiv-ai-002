# JAXbounce: A Continuous-Control Pong-Like Benchmark for Active Inference Beyond Low Dimensional Environments

Chaitanya Kaul1(×)[0000-0003-4893-6222], Markus Klar1 0000-0003-2445-152X], Sebastian Stein1[0000–0003-1828-4008], Fraser C. Paterson1[0009-0003-4392-7092],

John H. Williamson1 [0000-0001-8085-7853], and Roderick Murray-Smith¹()[0000-0003-4228-7962]

University of Glasgow, Glasgow, Scotland, UK Chaitanya.Kaul@glasgow.ac.uk

Abstract. We introduce JAXbounce, a continuous control pong-like bench mark designed to evaluate active inference beyond discrete, low dimensional environments. JAXbounce combines continuous states and stochastic dynamics in discrete time with flexible and composable observation models that allow representing the environment as a state vector or a pixel grid, corruptible through the rich family of noise models we provide. It is fully differentiable and parallelisable in JAX. We run over 10, 000 agent simulations on our environment comparing a baseline deterministic solution, to a state-based active inference agent with Unscented Kalman Filter perception and expected free energy planning, and a pixel based agent whose measurement model is a pretrained neural network encoder. All agents are evaluated across three scenarios of increasing difficulty with eight different noise conditions. Our experiments characterise how performance of continuous state agents scales with planning, planning horizons, observation modalities and knowledge of the noise process, exhibiting preference-oriented behaviour even in the presence of increasing uncertainty. We release JAXbounce as a reproducible benchmark for continuous control active inference research.

Keywords: Benchmark Environment · Flexible Observations · Continuous Control · Expected Free Energy · Active Inference.

## 1 Introduction

Active inference combines perception, planning and action into a single unified objective of minimising the expected free energy of candidate sequences of future actions over a finite planning horizon [13]. The promise of active inference is a single normative objective from which perception, planning, and action follow jointly, and in which exploration is not explicitly added, but is a structural consequence of minimising the expected free energy. An agent's preferences enter the framework as priors over observations rather than an externally specified reward, and information-seeking behaviour arises directly from the epistemic component of the planning objective [3]. Empirically, these agents are implemented as partially observable Markov Decision Processes (POMDP) acting on low dimensional discrete environments with discrete observation and action spaces [2][16][5]. Continuous space formulations exist but are run on very low dimensional proof of concept environments [17][19]. With recent interest in continuous control research [11][10], there is a need for a robust, fully differentiable environment which provides a flexible benchmark to evaluate these agents.

We introduce JAXbounce, a pong-like benchmark environment, to address this gap. Previous research has demonstrated that cultural neurons grown in a dish learn intentional behaviour in the absence of rewards through playing pong [8]. Learning to play (Atari) pong has also previously been demonstrated by neural networks and deep reinforcement learning [12][15]. In the context of active inference, playing pong-like games by minimising the expected free energy has been recently studied by simulating in vitro cell cultures [4[[914]. The environments the agents were tested on however were discrete state and primarily low dimensional 5 × 6 and 8 × 4 grids that do not truly demonstrate their true performance. Existing continuous-control benchmarks [18] are well suited for evaluating reinforcement learning algorithms, but do not directly test several properties central to active inference, including uncertainty-aware planning, preference-based behaviour, epistemic exploration, flexible observation models, and scalable belief-state inference. There is currently a research gap in continuous state active inference primarily due to the lack of a unifying benchmark to assess its performance. Towards this, JAxbounce provides continuous states, stochastic dynamics and is fully differentiable and parallelisable in JAX. It contains a rich family of noise models to test continuous state agents while providing the flexibility of tuning the underlying dynamics of the system to evaluate an agent's planning under increasing uncertainty. We evaluate JAXbounce on a series of benchmarks of increasing difficulty. Our contributions are summarised below,

— JAXbounce. We provide a continuous control benchmark for active inference which includes a pong-like environment, as well as a continuous active inference formulation of the game capable of probabilistically inferring an agent's actions from a falling ball's trajectory.

Evaluation. We evaluate our benchmark on various noise conditions, plans, planning horizons and across different scenarios simulating different ball trajectories. We also analyse the ability of the agent to cope with known system dynamics through minimising its epistemic uncertainty of the environment. In total, we evaluate over 10,000 agents providing a truly robust baseline. Open Source. We release JAXbounce along with our benchmarks to the community. Code available at https://github.com/drckaul/JAXBounce.

## 2 JAXbounce

Most active inference benchmarks have low dimensional state spaces where the difference between raw pixels and observed/belief state is small, leading to little room to study how an inference algorithm degrades as the observation channel deteriorates. We propose an environment implemented in JAX for just-in-time compilation and fast evaluation where an agent must predict a fast-moving target several steps into the future, with the goal being a single brief contact event between two physical objects, and the observation channel can be either a clean low-dimensional state vector or a $R \times R$ grayscale rendering that can be corrupted with realistic image noise. Pong is the simplest arcade game with these properties. We therefore implemented a pong-like game in JAX in a form tailored for variational inference.

## 2.1 State and action

A single JAXbounce trial begins in time and evolves with $\varDelta t = 1 / 6 0 \mathrm { s }$ time steps. The world state at step t is

$$
\mathbf { x } _ { t } = \ [ \underset { \mathrm { 1 0 - D } } { \underbrace { \mathbf { x } _ { t } ^ { \mathrm { b a l l } } } } , \underset { \mathrm { 5 - D } } { \underbrace { \mathbf { x } _ { t } ^ { \mathrm { b a t } } } } ] \in \mathbb { R } ^ { 1 5 } .\tag{1}
$$

The ball. $\mathbf { x } _ { t } ^ { \mathrm { { b a l l } } }$ stores a fall coordinate $y _ { \mathrm { f } } ~ \in ~ \left[ - 1 , 1 \right]$ , a lateral coordinate $y _ { \ell } \in \left[ - 1 , 1 \right]$ , their velocities $v _ { \mathrm { f } } , v _ { \ell }$ , the ball radius $r ~ = ~ 0 . 0 6$ , and geometrical parameters describing the ball. The ball transition dynamics are defined as, $\mathbf { \bar { x } } _ { t + 1 } ^ { \mathrm { b a l l } } = A _ { \mathrm { b a l l } } \mathbf { x } _ { t } ^ { \mathrm { b a l l } }$ , where $A _ { \mathrm { b a l l } } \in \mathbb { R } ^ { 1 0 \times 1 0 }$ advances position by $\Delta t \cdot \mathbf { v } .$

The bat. $\mathbf { x } _ { t } ^ { \mathrm { { b a t } } }$ stores the lateral position $y _ { t } ^ { \mathrm { b a t } }$ , the bat's velocity, and its fixed geometry (length L and thickness $T = 0 . 0 3 5 )$ . The bat is anchored at the bottom row and only moves laterally. We empirically estimate and fix $L = 0 . 0 6 2 5$ for our experiments. The bat dynamics are defined as, $\mathbf { x } _ { t + 1 } ^ { \mathrm { b a t } } = A _ { \mathrm { b a t } } \mathbf { x } _ { t } ^ { \mathrm { b a t } } + B _ { \mathrm { b a t } } a _ { t } ,$ with a damping of 0.95 The control matrix $B _ { \mathrm { b a t } }$ is chosen so that the action acts as a normalised target velocity, displacing the bat by $\varDelta t a _ { t }$ within the step.

Action. An agent issues a scalar command $a _ { t } \in [ - 1 , 1 ]$ at each step which is the normalised target velocity of the bat. Within a timestep, the ball moves under its current velocity, and the bat translates by the commanded velocity based on their respective transition functions.

## 2.2 A differentiable collision signal

A trial is a success when the bat and ball collide. We represent a collision between the bat and the ball with a smooth scalar $c _ { t } \in [ 0 , 1 ]$ .

$$
\begin{array} { r } { u _ { t } ^ { \mathrm { o v } } = r + \frac { T } { 2 } - | y _ { \mathrm { f } , t } - \tilde { y } ^ { \mathrm { b a t } } | , } \end{array}\tag{2}
$$

$$
\begin{array} { r } { l _ { t } ^ { \mathrm { { o v } } } = r + \frac { L } { 2 } - | y _ { \ell , t } - y _ { t } ^ { \mathrm { { b a t } } } | , } \end{array}\tag{3}
$$

$$
c _ { t } = \sigma ( k u _ { t } ^ { \mathrm { o v } } ) \sigma ( k l _ { t } ^ { \mathrm { o v } } ) , \qquad k = 8 .\tag{4}
$$

Here, $T$ and L are the bat thickness and its length respectively, and $r$ is the ball radius. $\tilde { y } ^ { \mathrm { b a t } }$ is the fixed fall coordinate of the bat. The product is high only when both axes overlap (vertical denoted by $u _ { t } ^ { \mathrm { o v } }$ and lateral denoted by $l _ { t } ^ { \mathrm { o v } } )$ and varies smoothly. $k = 8$ was tuned so that $c _ { t }$ peaks near 0.5 at perfect contact for the bat length and falls below 0.1 at distances greater than the radius of the ball.

## 2.3 Observation channels

An agent does not see the world state $\mathbf { x } _ { t }$ directly, but through one of two observation functions. The state-based agents observe a 4-D vector

$$
\mathbf { o } _ { t } ^ { \mathrm { s } } = g _ { \mathrm { s } } ( \mathbf { x } _ { t } ) = [ y _ { \mathrm { f } , t } , y _ { \ell , t } , y _ { t } ^ { \mathrm { b a t } } , c _ { t } ] \in \mathbb { R } ^ { 4 } ,\tag{5}
$$

which are the ball's position, the bat's position, and the differentiable collision signal $c _ { t }$ (equation $( 4 ) )$ . Agents with pixel-based inputs observe a grayscale image $\mathbf { o } _ { t } ^ { \mathrm { ~ p ~ } } = g _ { \mathrm { p } } ( \mathbf { x } _ { t } ) \in [ 0 , 1 ] ^ { h \times R }$ with the ball rendered as a blob and the bat as a rectangle. We experiment with different values of R to evaluate the effect of different grid sizes on active inference agents.

## 2.4 Scenarios and catch criterion

Three different scenarios are designed in increasing order of difficulty. The ball always falls from the top of the grid $y _ { \mathrm { f , 0 } } ~ = ~ - 0 . 9$ at the start of each trial with a different location across the x-axis sampled from a uniform distribution $y _ { \ell , 0 } ~ \sim ~ \mathcal { U } [ - 0 . 7 , 0 . 7 ]$ . In scenario 1, S1 (easy), the ball falls vertically with a constant speed. In scenario 2, S2 (medium), The ball falls vertically with an unknown speed in each trial and the agents must infer the ball's velocity. In scenario 3, S3 (hard), the ball falls at a lateral angle of $\pm 3 5 ^ { \circ }$ with an unknown speed. A trial within each scenario counts as a success if ma $\mathrm { x } _ { t } c ( \mathbf { x } _ { t } ) > 0 . 4$ which equates to a physical overlap between the bat and the ball.

## 3 Active Inference for JAXbounce

Active inference treats perception and action as inference under a single objective. Perception is the agent's posterior belief over the latent states of the world given a stream of observations. Action is the choice of control that minimises the expected free energy of its predicted future observations. Our resulting active inference framework consists of the following components,

Generative process. The real world JAXbounce dynamics and one of the observation functions $g _ { \mathrm { s } }$ or $g _ { \mathrm { p } }$ . The process is the real world in which the agent acts, observed through its Markov blanket. The agent does not have access to $\mathbf { x } _ { t }$ directly, but only to its (noisy) projection $g ( \mathbf { x } _ { t } )$

\- Generative model. The agent's internal model of how observations are generated from latent states. Each agent type of agent has its own specific measurement model h. For our state-based agents, $\mathbf { h } = g _ { \mathrm { s } }$ and for our pixel-based agents, $\mathbf { h } = \phi _ { \theta } ( g _ { \mathrm { p } } )$ where a rendered JAXbounce grid is passed through a pretrained neural network to estimate the observed states. $\phi$ is the pretrained neural network encoder with θ parameters.

– Probabilistic beliefs. All agents we run carry two Gaussian beliefs at each simulation step. $Q _ { t } ^ { \mathrm { s } } : = \mathcal { N } ( \mu _ { t } ^ { \mathrm { s } } , \Sigma _ { t } ^ { \mathrm { s } } )$ over the 15-D latent state, and $Q _ { t } ^ { \mathrm { k } } : =$ $\mathrm { a x p } ( \mathcal { N } ( \pmb { \mu } _ { t } ^ { \mathrm { k } } , \pmb { \Sigma } _ { t } ^ { \mathrm { k } } ) )$ over the observations. Each belief is a mean and covariance matrix of a multivariate Gaussian distribution and they are log-Gaussian to keep their scale positive at all times.

\- Belief update. To update an agent's beliefs after performing an action, we used an Unscented Kalman Filter (UKF) [20][7]. The $\mathrm { U K F } \mathrm { s }$ prediction stage propagates the state belief forward through the system dynamics via the unscented transform (UT). In the correction stage we compare predicted belief against an observation using variational inference (VI). Given smooth system dynamics the unscented transform provides a cheap and exact belief estimate while the VI correction helps account for any non-nonlinearities arising from differentiable measurement models.

\- Action selection. Action selection is done by minimising the expected free energy (EFE) of simulated candidate future action sequences of horizon H. The EFE decomposes into a pragmatic and an epistemic term. We primarily experiment with the pragmatic term, minimising EFE to satisfy our set preference, and later show experiments where the epistemic term is useful in cases where unknown system dynamics require planning to reduce uncertainty about the latent states.

## 3.1 Beliefs and belief update

Beliefs. We parameterise the state belief at timestep $t , Q _ { t } ^ { \mathrm { s } }$ , by a mean $\pmb { \mu } _ { t } ^ { \mathrm { s } } \in \mathbb { R } ^ { 1 5 }$ and a lower-triangular Cholesky factor $\mathbf { L } _ { t }$ such that $\pmb { \Sigma } _ { t } ^ { \mathrm { s } } = \mathbf { L } _ { t } \mathbf { L } _ { t } ^ { \top }$ . The Cholesky form keeps the covariance matrix positive-definite by construction throughout VI optimisation.

Observation noise belief. The observation noise belief, $Q _ { t } ^ { \mathrm { k } }$ enters the VI based correction step as how much an agent trusts each value in its 4D observation vector. Our ablations show how mis-calibrating $Q ^ { \mathrm { k } }$ by having a strong belief that the agent's measurement model provides noise free observations can result in the agent to over trust the noisy observations.

UKF Predict step. Sigma points are placed at $\pmb { \mu } _ { t } ^ { \mathrm { s } }$ and at $\mu _ { t } ^ { \mathrm { s } } \pm [ \sqrt { ( d + \lambda ) \Sigma _ { t } ^ { \mathrm { s } } } ] _ { j }$ for $j = 1 , \ldots , d ,$ with $d = 1 5$ , scaling $\alpha = 1 0 ^ { - 3 } , \beta _ { \mathrm { u t } } = 2 , \kappa = 3 \times 1 0 ^ { 6 }$ . Each of the $2 d + 1 = 3 1$ sigma points is propagated through the system dynamics, $f ( \cdot , a _ { t } )$ 2 and the predict-prior $\bar { Q } _ { t + 1 } ^ { \mathrm { s } } = \mathcal { N } ( \bar { \pmb { \mu } } _ { t + 1 } ^ { \mathrm { s } } , \bar { \pmb { \Sigma } } _ { t + 1 } ^ { \mathrm { s } } )$ is recovered as the weighted mean and covariance of the propagated points.

UKF Correct step. The variational posterior is found by minimising the variational free energy

$$
\mathcal { F } ( Q ^ { \mathrm { s } } ) = D _ { \mathrm { K L } } ( Q ^ { \mathrm { s } } \parallel \bar { Q } _ { t + 1 } ^ { \mathrm { s } } ) + \mathbb { E } _ { Q ^ { \mathrm { s } } Q ^ { \mathrm { k } } } [ - \log p ( \mathbf { o } _ { t + 1 } \mid \mathbf { x } , \pmb { \sigma } _ { o } ) ] ,\tag{6}
$$

which trades off staying close to the predict-prior (first term) against fitting the new observation under the measurement model i.e. the log-likelihood of the observation under the measurement model (second term). $\mathcal { F }$ is minimised by gradient descent on $( \mu ^ { \mathrm { s } } , \mathbf { L } )$ , with the expectation in equation (6) estimated from samples drawn from $Q ^ { \mathrm { s } }$ and $Q ^ { \mathrm { k } }$

A small ridge term, €I $( \epsilon = 1 0 ^ { - 8 } )$ , is added to the state covariance before every Cholesky decomposition to keep the belief covariance positive-definite.

## 3.2 Action selection: expected free energy

The agent's preferences over observations are encoded as a prior preference distribution $P ^ { \mathrm { c } }$ . We encode this with a Gaussian preference on the collision channel of the observation,

$$
P ^ { \mathrm { c } } ( \mathbf { o } ) = \mathcal { N } ( o _ { c } , \sigma _ { C } ^ { 2 } ) ,\tag{7}
$$

where $o _ { c } = o _ { 4 }$ is the collision observation. The tightness $\sigma _ { C } ^ { 2 }$ controls how strongly the agent insists on the collision signal being close to its target value of 1.

Expected free energy. The agent scores a candidate plan $\tilde { \mathbf { a } } = ( a _ { t + 1 } , \dots , a _ { t + H } )$ by the expected free energy of the observations it expects the plan to produce. Without the epistemic term,

$$
\mathcal { G } _ { p } ( \tilde { \mathbf { a } } ) = - \mathbb { E } _ { Q ^ { \mathrm { o } } ( \tilde { \mathbf { a } } ) } [ \log P ^ { \mathrm { c } } ( \mathbf { o } ) ] ,\tag{8}
$$

where $Q ^ { \mathrm { o } } ( \tilde { \mathbf { a } } )$ is the agent's predictive distribution of observations under plan ã.

Hallucinated observations. In the absence of real observations, to evaluate equation (8) the agent simulates the observations the plans would produce. For each timestep in the planning horizon, $\tau \in \{ t + 1 , t + 2 , . . . t + H \}$ the agent runs a UKF prediction step under each action $a _ { \tau }$ in the timestep to obtain its predicted state belief $Q _ { \tau } ^ { \mathrm { s } } ( \tilde { a } )$ . We then sample $N _ { s }$ states from this predicted belief and $N _ { o }$ observations each to estimate their log-likelihood given the preference distribution. The plan with the lowest expected free energy is chosen as the optimal plan at that timestep by the agent. N plans are rolled out for a planning horizon τ in this step and the executed plan is the argmax of the per plan negative expected free energy,

$$
\mathrm { N E F E } ( \tilde { \mathbf { a } } ^ { ( i ) } ) = - \beta \mathcal { G } _ { p } ( \tilde { \mathbf { a } } ^ { ( i ) } ) ,\tag{9}
$$

where $\beta = 5$ is the pragmatic scaling term. The first action of the plan with the lowest NEFE is chosen and executed.

## 3.3 Agents

To evaluate our active inference framework, we propose three agents. The first is a deterministic baseline that executes actions based on predicting a closedform solution of the ball's landing point without active inference, the second is a state-based active inference agent which selects actions to minimise the expected free energy of its hallucinated future observations against its preference via a state based measurement model, and the third is a pixel-based active inference agent that takes a JAXbounce grid as an input and uses a pretrained encoder as its measurement model to predict its observations. All three agents share the JAXbounce dynamics, the UKF and the criteria that defines a successful catch. They differ in their measurement model h and in how they select actions $\mathbf { a } _ { \mathbf { t } }$ at each timestep.

Deterministic Baseline. The deterministic agent perceives the world through its observation states (Section 2.3). At every step, the deterministic baseline uses the UKF's posterior belief of $\pmb { \mu } _ { t } ^ { \mathrm { s } }$ to obtain a belief about the ball's position $( \hat { y } _ { \mathrm { f } } , \hat { y } _ { \ell } )$ , its velocity $( \hat { v _ { \mathrm { f } } } , \hat { v _ { \ell } } )$ , and the position $\hat { y } ^ { \mathrm { b a t } }$ , of the bat. The agent then computes the time for the ball to reach the bat, from which it then calculates its landing point as,

$$
t _ { \mathrm { g } } = \operatorname* { m a x } \Bigl ( 0 , \frac { \tilde { y } ^ { \mathrm { b a t } } - \hat { y } _ { \mathrm { f } } } { \hat { v } _ { \mathrm { f } } } \Bigr ) , \quad \hat { y } _ { \mathrm { l a n d } } = \hat { y } _ { \ell } + \hat { v } _ { \ell } t _ { \mathrm { g } } , \quad a _ { t } = \mathrm { c l i p } \bigl ( k _ { p } \bigl ( \hat { y } _ { \mathrm { l a n d } } - \hat { y } _ { t } ^ { \mathrm { b a t } } \bigr ) , \pm 1 \bigr ) ,\tag{10}
$$

Where $\tilde { y } ^ { \mathrm { b a t } }$ is the location of the bat on its fixed axis and $k _ { p } = 2$ is the gain that limits the bat action to within the grid. The gain is fixed to a value $> 1$ to allow the deterministic solution to reach the ball even for harder scenarios, making it a competitive baseline for the active inference agent.

State-Based Active Inference. The state-based AIF agent shares its observation model with the deterministic agent. They differ in their action selection methodology. The state-based AIF agent has a preference for observing a collision between bat and ball. It selects actions towards its preference by minimising the expected free energy of rolled out plans across fixed planning horizons with this preference and executes the first action in the selected plan with the least expected free energy. A comparison with the deterministic agent under this setting allows us to directly see the advantage of active inference based planning in our environment as this tests which agent can handle sensor uncertainty better.

Pixel-Based Active Inference. The pixel-based AIF agent differs from the state-based AIF agent in as its measurement model. Its action selection is similar to the state-based AIF agent. The measurement model is an MLP that takes a JAXbounce grid as an input and outputs the 4 observation channels from Section 2.3. It flattens the rendered frame to a 1D vector and applies two ReLU activated hidden layers of width 128 and 64 before a linear 4-dim output ${ \mathbf { o } } _ { t } ^ { \mathrm { s } } .$ We pre-train 4 such encoders, one for each image grid size we test in our experiments. The encoder's pretraining is described in appendix A.

## 4 Experiments

To evaluate the ability of active inference agents to play JAXBounce, we compare the deterministic baseline solution with the state-based AIF agent. We also run experiments with the pixel-based AIF agent for reference. Each agent in each experiment is run for 30 trials and a successful trial is one where the agent can align the bat with the ball to make contact with it. We run all agents across all 3 scenarios (Section 2.4) and 8 noise conditions. For the deterministic baseline and state-based AIF agent, the noise conditions are shown in Table 1, which increase the uncertainty about the ball's initial position, its speed which the agent needs to infer, and uncertainty for the measurement model to understand how the agents handle noisy environments. For the pixel-based AIF agent, we apply gaussian noise $( \sigma = 0 . 0 5 , 0 . 1 0 , 0 . 2 0 )$ , Hann-window blur (windows $3 \times 3 , 5 \times 5 )$ and salt and pepper noise (per-channel rate 0.02 and 0.05) to the pixel grids as the 8 noise conditions.

We evaluate two forms of agents for our experiments. In the first case, the agent has an initial belief about the noise conditions equivalent to Noise 1 in table 1. This simulates a scenario where the agent assumes the states of the environment are relatively noise free, and we observe how its performance degrades with increasing noise conditions. In the second case, for each simulation, the agent knows the noise in the environment, i.e. its initial belief of the noise is equivalent to the noise condition tested from table 1. This simulates a scenario where the agent knows the environment is noisy and we test its robustness to this noise through our simulations. Our experiments can be summarised as the following: For case 1, Experiment 1 (Bat Size): Bigger bat sizes make observing collision easier. We experiment with 3 different bat sizes, each making the scenarios harder as the agents now have a lesser margin for error. Experiment 2 (Plans): To observe the effect of planning horizons, we run all agents with 100, 200, 300 and 400 plans. Experiment 3 (Planning Horizons): We experiment with planning horizons of 10, 60 and 100 timesteps to see the effects of allowing agents to hallucinate observations over longer planning horizons to minimise its expected free energy. Experiment 4 (Pixel-based AIF grid-size): As the jaxBounce grid is a hyperparameter that defines the representation learned by the MLP encoder of the pixel-based AIF agent, we experiment with 3 grid sizes, $1 6 { \times } 1 6 , 3 2 { \times } 3 2 \mathrm { a n d } 6 4 { \times } 6 4$ For case 2, Experiment 5 (Agent knows the noise conditions): we evaluate the state AIF agent across 100, 200, 300 and 400 randomly sampled plans for horizons of 10, 60 and 100 timesteps.

## 5 Results

Our first set of experiments evaluated the active inference agent in cases where the agent does not have the true model of the observation noise. We first performed a preliminary experiment to assess the effect of bat size on task difficulty. Since the bat length L determines the margin of error tolerated at interception of the ball by the bat, varying L allowed us to calibrate the difficulty of the benchmark. Based on these experiments we fixed $L = 0 . 0 6 2 5$ , the value at which the task remained reliably solvable while still imposing a non-trivial control and estimation demand. Figure 6 summarises our results for the three different bat sizes assessed.

We next examined how performance scales with the breadth of the planning search. Holding the horizon fixed at $H = 1 0 0$ , we swept the number of sampled plans $N \in { 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 }$ . Since the agent selects actions by minimising expected free energy over N sampled plan rollouts, N determines how thoroughly the space of all possible future actions across different planning trajectories is explored at each step. Figure 7 summarises our results for the planning sweeps. As expected, increasing the number of plans provides the agent with the ability to search across a larger space of possible candidate actions to take, compensating for the increased uncertainty with growing noise. We observed $N = 3 0 0$ to be the optimal trade-off between computation time and successfully completed trials where the bat intercepted the ball across all evaluated scenarios.

![](images/3eb9e44c06d7c9ccc1cd39364c910400b3ca7168274e9bacd6f90d6a8a6108dc.jpg)  
Fig. 1: Experiment 5. State AIF performance heatmaps across all sampled plan variations $( N ~ = ~ \{ 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \} )$ and planning horizons $\begin{array} { l l } { ( H } & { = } \end{array}$ {20, 60, 100}). The agent here has an initial belief covariance of the noise similar to the actual noise conditions listed in table 1.

We varied the planning horizon H ∈ 20, 60, 100 while holding the number of randomly sampled plans fixed at $N = 3 0 0 .$ The horizon sets how many steps into the future each candidate policy is rolled out under the generative model before its expected free energy is evaluated. We observed that a longer horizon allowed the agent to anticipate the ball's trajectory earlier and position the bat in advance, whereas a shorter horizon yielded worse catch rates, especially in the harder scenario (scenario 3) where the agent needed to infer the ball velocity as well as its lateral trajectory. The catch rates are reported as a function of H across all three scenarios and eight observation-noise conditions in the figure 8 heatmaps. We provide the deterministic baseline as a reference for our results. As the deterministic baseline differs from the active inference agents in its planning, it allows us to assess the true performance of these agents under uncertainty. Particularly in cases of high uncertainty, as the scenarios get progressively harder, the catch rate for the bat intercepting the ball in deterministic baseline deteriorate, while the state AIF agent with access to longer planning horizons and sufficient planning reliably discards noisy observations to choose actions to minimise its uncertainty about the environment to intercept the ball. We also evaluated the effect of grid sizes on the pixel AIF agent. Our results are summarised in figure 9. We observed that an increase in the grid size resulted in a worse catch rate with increasing noise. This is due to the inability of the neural network encoder to handle out of distribution pixel perturbations. The neural network encoder (MLP) here serves are the measurement model of the agent compressing the pixel grid into its 4D observation vector. Increasing the noise in the pixel space results in increasingly accumulating errors leading to poor observation vector estimates which resulted in poorer performance. Due to this we used a $3 2 \times 3 2$ grid size for our experiments as it provided the best trade off between catch rates and the need to compute multiple forward passes of higher dimensional grid sizes through the measurement model to obtain the 4D observations.

N = 5 — N = 10  N = 50  N = 100  N = 200  N = 300  N = 400  
![](images/62d683b1939d7c3b895065b176cff1dbb8f019975a83b5ba69e5365122e9200e.jpg)

In our final experiment (figure 1), we evaluated how an agent whose noise model is calibrated to the true noise in the observations performs for the state AIF agent. Even in cases of higher uncertainty, the agent continues to act in accordance with its preference distribution, appropriately discarding unreliable observations rather than being driven by them.

## 6 Discussion

![](images/9eb5aa98074701b51403601fe61cf9308254048f9b99ecd11a9e22c0092a6931.jpg)

![](images/92428dc8ca42bff5d47f380654a90fb47f5bc4b5104d427cc868bdbbfaa56827.jpg)  
Fig. 2: Sensitivity analysis of the agent's chosen plan for four example noise conditions by minimizing the NEFE with respect to all actions in its planning horizon. The plot shows the partial derivative of the actions in the plan with respect to the NEFE. An increase in the number of sampled plans decreases the gradient such that a small perturbation in the actions does not change the agent's chosen plan's NEFE.

![](images/2f135227a4b19bb24b3171241162a80d7f2fe38f36c356b39492cc6b4a8e4cc9.jpg)

![](images/8c4635d9a2003a2ff644ea8f10814d33d425fe88abc6aa5b5cc339de4535f217.jpg)

![](images/a5ebe912e1059b6b5c8685baa41cb69ba70d462069b513a8827f23b8b896db3b.jpg)

![](images/dcd7def4622ed08b237b0af0037a581c064bfd68672e1cf9a06a19596ce54a50.jpg)  
Fig. 3: Sensitivity analysis of the agents planning for four example noise conditions across different number of plans. The plot shows the state AIF agent's sensitivity to the plan chosen by minimizing the NEFE to all the actions in its planning horizon for the case where the agent's belief of the noise matches the actual noise. An increase in the number of sampled plans decreases the gradient.

Plan Sensitivity. Figures 2 and 3 report the sensitivity of the state AIF agent's selected plan, measured as the average of the partial derivative of the negative expected free energy (NEFE) with respect to each action in the planning horizon (for four noise conditions from table1). A large gradient magnitude indicates that the chosen plan lies on a steep region of the NEFE landscape, such that small perturbations to the actions would substantially change the plan's score. A small gradient indicates that the plan sits near a flat optimum and is correspondingly robust to such perturbations. We sampled $\mathrm { N } \in \{ 5 , 1 0 , 5 0 , 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \}$ plans across a $H = 1 0 0$ planning horizon and averaged the gradient over five runs for each plan and each noise condition evaluated. We observe that as the number of sampled plans increases, the average gradient magnitude across the horizon decreases. With more candidate plans evaluated at each step, the agent is more likely to recover a plan close to the true NEFE minimum. Increasing the number of random plans drives the agent toward solutions that not only lead to better action selection but are also less sensitive to the precise choice of actions. This trend holds across all four noise conditions shown, and is preserved in the noise-aware case (Figure 3), where calibrating the agent's belief to the true observation noise yields the same flattening of sensitivity as the number of plans grows.

![](images/b0778cc39991485ddf27872d15c8ddb51bc0a49a728fdc7b3da3c52048a4be83.jpg)

![](images/fd5fef382fc9e4598b05ee23c92afad379b929f50a243728a2c280b7af688275.jpg)

![](images/3e07a15f92da67f4952e4df8b97d035d903a8959c2a91bb0ec52a72330a12138.jpg)  
Fig. 4: (Left) Evolution of the state AIF agent's belief about the friction field around the bat vs the true friction field. (Middle) The agent's surprise over its observation vector (Right) Agent's surprise over balls' x and y position and the bat's lateral position.

Environment Uncertainty. Our experiments do not explicitly evaluate the integration of the information gain term in the free energy computation. This is because the agents do not need an epistemic drive to explore the environment as the system dynamics are known and observable. To demonstrate the contribution of the epistemic term, we create a setting in which a relevant state component is not directly observable, but must be inferred. We do this by adding a friction field across the bat's lateral axis such that each action the agent takes is now damped by this field. We ran a state AIF agent to then assess whether the agent can infer this unknown field in the Noise 1 setting for scenario 3 (lateral falling ball with varying velocity). To run this agent, we scale the pragmatic value by a factor of $\beta = 5$ and clip the information term to have a maximum value of 1000. The final negative expected free energy is a sum of both terms. The agent updates its belief by computing the VFE for 100 SGD steps with a learning rate of $1 0 ^ { - 3 }$ after each new observation. The information-gain term in the expected free energy is approximated by sampling, $n _ { \mathrm { I G } } ^ { s } = 9$ latent-state samples and, for each, $n _ { \mathrm { I G } } ^ { o } = 1$ observation sample. Figure 4 shows how the agent's belief of the friction field changes over the simulation as it converges to the true value. We also show the agent's surprise of its observation for the entire 4D observation vector, as well as the individual components in the vector by computing the marginal probability of each component with respect to the agent's belief of the states. At the start of the simulation, the agent has a belief of 0.25 on the friction field (true field = 0.35). After the first few actions in the simulation we see a sharp rise in the agent's surprise over where the bat is as in a continuous state system, the surprise accumulates over a few time steps. Following this, the agent samples plans to minimise its uncertainty about the field as we see the friction field belief rise over the next few steps. This shows the agent isn't just minimising the EFE to satisfy a preference, but also has an epistemic drive to learn its environment. In future work, we aim to extend this scenario by introducing multiple friction-field bands across the bat's lateral axis to see how the agent can cope with varying regions of high and low friction affecting its actions.

## 7 Conclusion

In this paper, we presented JAXbounce, a continuous-control, Pong-like benchmark that moves active inference beyond the low-dimensional, discrete environments on which it is typically evaluated. The environment combines continuous bat dynamics and a continuous action interface with a calibrated family of observation-noise models, and is fully differentiable and parallelisable in JAX, making evaluating over an agent's planning hyperparameters cheap to run. We evaluate a state-based and a pixel-based active inference agent on JAXbounce across three increasingly difficult interception scenarios and eight observationnoise conditions, characterising how its performance scales with the number of plans, its planning horizon, and knowledge of the noise process. We show that agents evaluated in our environment exhibit goal-directed behaviour even under high perceptual uncertainty. JAXbounce and all experiment configurations are open-sourced to the community with hopes to accelerate continuous-control active inference research.

Acknowledgements. This work was funded by the Designing Interaction Freedom via Active Inference (DIFAI) ERC Advanced Grant (101097708), funded under the UK Horizon guarantee as EPSRC EP/Y029178/1. R.M-S. and J.H.W. acknowledge EPSRC EP/R018634/1. R.M-S. also acknowledges EP/T00097X/1 and EP/T021020/1. M.K. acknowledges the AI-Twin ERC Advanced Grant (101200584), funded by HORIZON.1.1–European Research Council (ERC).

## A Pixel-based Encoder Pretraining

The pixel-based AIF encoder is pretrained offline on 30000 random (state, image) pairs by minimising $\mathbb { E } _ { \mathbf { x } , \eta } \Vert \phi _ { \theta } ( \eta ( { \mathrm { r e n d e r } } ( \mathbf { x } ) ) ) - \mathbf { o } ^ { \mathrm { s } } ( \mathbf { x } ) \Vert _ { 2 } ^ { 2 }$ , where η is noise drawn uniformly from one of four categories, (1) clean, (2) additive Gaussian (std $\sim \mathcal { U } [ 0 , 0 . 0 2 ] )$ , (3) Hann-window blur (window $2 \times 2 )$ , and (4) salt-and-pepper (per-channel rate $\sim \mathcal { U } [ 0 , 0 . 0 0 5 ] )$ . We do this to test the effects of having a high noise in input measurements in our simulations. Neural network performance tends to degrade drastically in the presence of high noise and adversarial attacks (new noise). Due to this reason, we make sure that the encoder is familiar with each noise structure but has never seen the test-time noise distribution, creating a controlled out-of-distribution gap. All MLPs are trained using the Adam optimiser with a learning rate of $1 e ^ { - 2 }$ for 8000 steps with batch size 128.

## B Noise Conditions

We provide the different noise conditions applied to the deterministic agent and the state AIF agent in tabular form in table 1 below. Figure 5 visualizes the different noise conditions applied to the pixel grid for the pixel AIF agent.

![](images/ebf46464e38c02a83e076d645635d9917f02967018546d2a1621bd4bae70ff5f.jpg)  
Fig. 5: Visualisations of the JAXbounce grids used in pixel AIF simulations. The clean grid with increasing uncertainty on the agent's initial belief (table 1) of the position of the ball and the bat and increasing uncertainty on its measurement model is used for the deterministic and state AIF simulations.

## C Results Heatmaps

We visualise all catch rate heatmaps for experiment 1 (figure 6), experiment 2 (figure 7), experiment 3 (figure 8) and experiment 4 (figure 9) in this section.

![](images/9f5237a9cd1c637525e6c0b7390f4dca367304f71935f572b99e04081b28a171.jpg)  
Fig. 6: Experiment 1. Bat size comparison heatmaps between the deterministic and state AIF agents. All runs are averaged over 30 trials for Scenario 1 (constant velocity ball falling), Scenario 2 (variable ball velocity in each run) and Scenario 3 (variable ball velocity and lateral ball falling). State AIF agent runs randomly sample 300 plans over a finite horizon of 100 timesteps.

![](images/5c5d404dd5c04b5275e09d9bcd344804f2685ea0a43872502c6cca14aa87f833.jpg)  
Fig. 7: Experiment 2. Analysis of agent performance for varying number of plans across the three scenarios. All plans are run for a horizon of 100.

![](images/07b2a02a0d6d50f816379f1d9f4bf2b99e8aa81fc99ec49b07853101893fa171.jpg)  
Fig. 8: Experiment 3. Comparison of performance for varying planning horizons across the three scenarios. The agents randomly sample 300 plans.

<table><tr><td colspan="4">Scenario 1</td><td colspan="3">Scenario 2</td><td colspan="3">Scenario 3</td><td rowspan="2">1.00</td></tr><tr><td>Clean</td><td>0.93</td><td>0.90</td><td>0.83</td><td>1.00</td><td>0.90</td><td>0.67</td><td>0.83</td><td>0.67</td><td>0.33</td></tr><tr><td>Gauss σ=0.05</td><td>0.90</td><td>0.93</td><td>0.50</td><td>0.83</td><td>0.67</td><td>0.40</td><td>0.83</td><td>0.53</td><td>0.27</td><td>0.75</td></tr><tr><td>Gauss σ=0.10</td><td>0.90</td><td>0.61</td><td>0.47</td><td>0.67</td><td>0.53</td><td>0.27</td><td>0.77</td><td>0.40</td><td>0.33</td><td>Catc rate</td></tr><tr><td>Gauss σ=0.20</td><td>0.80</td><td>0.60</td><td>0.33</td><td>0.33</td><td>0.72</td><td>0.17</td><td>0.67</td><td>0.43</td><td>0.17</td><td></td></tr><tr><td>Blur w=5</td><td>0.67</td><td>0.58</td><td>0.37</td><td>0.77</td><td>0.63</td><td>0.43</td><td>0.67</td><td>0.33</td><td>0.17</td><td>0.50</td></tr><tr><td>Blur w=7</td><td>0.33</td><td>0.55</td><td>0.37</td><td>0.53</td><td>0.39</td><td>0.27</td><td>0.33</td><td>0.30</td><td>0.37</td><td>0.25</td></tr><tr><td>S&amp;P 2.5%</td><td>0.47</td><td>0.61</td><td>0.23</td><td>0.67</td><td>0.20</td><td>0.23</td><td>0.33</td><td>0.37</td><td>0.10</td><td></td></tr><tr><td>S&amp;P 5%</td><td>0.33</td><td>0.39</td><td>0.14</td><td>0.33</td><td>0.27</td><td>0.17</td><td>0.10</td><td>0.12</td><td>0.07</td><td></td></tr><tr><td></td><td>16×16</td><td>32×32</td><td>64×64</td><td>16×16</td><td>32×32</td><td>64×64</td><td>16×16</td><td>32×32</td><td>64×64</td><td>0.00</td></tr></table>

Fig. 9: Experiment 4. Varying grid sizes for the pixel AIF agent. Each simulation is run for 300 randomly sampled plans over a planning horizon of 100 time steps.

Table 1: Noise conditions in experiments for the deterministic baseline and statebased AIF agent. The eight noise tiers are designed to make inference increasingly harder for the agents.
<table><tr><td>Label</td><td> $| \sigma _ { \mathrm { p o s } } ^ { ( 0 ) }$ </td><td> $\sigma _ { \mathrm { v } } ^ { \mathrm { m u l t } }$   $\sigma _ { \mathrm { o } } ^ { \mathrm { i n j } }$ </td></tr><tr><td>NOIsE 1|</td><td> $1 0 ^ { - 3 }$  1.0</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>NOISE 2</td><td>0.02</td><td>1.5 0.05</td></tr><tr><td>NOISE 3</td><td>0.05</td><td>2.0 0.10</td></tr><tr><td>NOISE 4</td><td>0.10</td><td>2.5 0.20</td></tr><tr><td>NOIsE 5</td><td>0.12</td><td>3.0 0.25</td></tr><tr><td>NOISE 6</td><td>0.15</td><td>3.5 0.30</td></tr><tr><td>NOISE 7</td><td>0.18</td><td>4.0 0.35</td></tr><tr><td>NOISE 8</td><td>0.25</td><td>5.0 0.40</td></tr></table>

## D Limitations

A single agent for configuration (N = 300 policies, H = 100 horizon, $T =$ 150 steps) performs roughly 4.5 million forward passes through its generative model per trial. Our evaluations report over 10, 000 agent simulations. Across all agents, we have run over 20, 000 simulations with different seeds in our initial experiments. This requires considerable amount of compute. For the pixel AIF agent specifically, we have run approximately 2.9 billion forward passes of the MLP encoder parallelizing runs across 4 Nvidia RTX A6000 GPUs for 3 days. This is considerable amount of compute required to run these agents even with small MLP encoders which is a current limitation of our approach. We aim to mitigate this in future work by investigating how to sample plans in smarter and more efficient ways [1] as well as investigating ways to amortise [6] active inference for our pipeline.

## References

1. Apers, S., Gribling, S., Szilágyi, D.: Hamiltonian monte carlo for efficient gaussian sampling: long and random steps. Journal of Machine Learning Research 25(348), 1–30 (2024)

2. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., Friston, K.: Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology 99, 102447 (2020)

3. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., Pezzulo, G., et al.: Active inference and learning. Neuroscience & Biobehavioral Reviews 68, 862–879 (2016)

4. Friston, K.J., Salvatori, T., Isomura, T., Tschantz, A., Kiefer, A., Verbelen, T., Koudahl, M., Paul, A., Parr, T., Razi, A., et al.: Active inference and intentional behavior. Neural computation 37(4), 666–700 (2025)

5. Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I., Tschantz, A.: pymdp: A python library for active inference in discrete state spaces. arXiv preprint arXiv:2201.03904 (2022)

6. Huang, D., Wen, X., Bharti, A., Kaski, S., Acerbi, L.: ALINE: Joint amortization for Bayesian inference and active data acquisition. Advances in Neural Information Processing Systems 38, 54068–54102 (2026)

7. Julier, S.J., Uhlmann, J.K.: Unscented filtering and nonlinear estimation. Proceedings of the IEEE 92(3), 401–422 (2004)

8. Kagan, B.J., Kitchen, A.C., Tran, N.T., Habibollahi, F., Khajehnejad, M., Parker, B.J., Bhat, A., Rollo, B., Razi, A., Friston, K.J.: In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. Neuron 110(23), 3952–3969 (2022)

9. Khajehnejad, M., Habibollahi, F., Paul, A., Razi, A., Kagan, B.J.: Biological neurons compete with deep reinforcement learning in sample efficiency in a simulated gameworld. arXiv preprint arXiv:2405.16946 (2024)

10. Klar, M., Stein, S., Paterson, F., Williamson, J.H., Gollee, H., Murray-Smith, R.: Intermittent active inference. Entropy 28(3), 269 (2026)

11. Klar, M., Stein, S., Paterson, F., Williamson, J.H., Murray-Smith, R.: An active inference model of mouse point-and-click behaviour. In: International Workshop on Active Inference. pp. 235–254. Springer (2025)

12. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., et al.: Human-level control through deep reinforcement learning. nature 518(7540), 529–533 (2015)

13. Parr, T., Pezzulo, G., Friston, K.J.: Active inference: the free energy principle in mind, brain, and behavior. MIT Press (2022)

14. Paul, A., Khajehnejad, M., Habibollahi, F., Kagan, B.J., Razi, A.: Simulating biological intelligence: Active inference with experiment-informed generative model. arXiv preprint arXiv:2508.06980 (2025)

15. Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al.: Mastering Atari, go, chess and shogi by planning with a learned model. Nature 588(7839), 604–609 (2020)

16. Smith, R., Friston, K.J., Whyte, C.J.: A step-by-step tutorial on active inference and its application to empirical data. Journal of mathematical psychology 107, 102632 (2022)

17. Stein, S., Williamson, J.H., Murray-Smith, R.: Towards interaction design with active inference: A case study on noisy ordinal selection. In: International Workshop on Active Inference. pp. 3–15. Springer (2024)

18. Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D.d.L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq, A., et al.: Deepmind control suite. arXiv preprint arXiv:1801.00690 (2018)

19. Waade, P.T., Mikus, N., Mathys, C.: Inferring in circles: Active inference in continuous state space using hierarchical Gaussian filtering of sufficient statistics. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. pp. 810–818. Springer (2021)

20. Wan, E.A., van der Merwe, R.: The Unscented Kalman filter for nonlinear estimation. In: Proceedings of the IEEE 2000 adaptive systems for signal processing, communications, and control symposium (Cat. No. 00EX373). pp. 153–158. Ieee (2000)