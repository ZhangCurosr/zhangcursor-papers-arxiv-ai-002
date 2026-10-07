# Modeling Latent Disturbances for Robust Decision-Making in World Models

Junwon Seo and Andrea Bajcsy

https://junwon.me/LatentDisturbance/

![](images/7761a551558073365011d1f5718cf45ae8438bc460505cc10568475eb85c4d5e.jpg)  
Fig. 1. Overview of Robust Decision Making in World Models. Left: A latent world model learns a latent dynamics model $z ^ { \prime } \sim f _ { z } ( z , a )$ from observation– action transitions $( o , a , o ^ { \prime } )$ , predicting a distribution over next latent states $z ^ { \prime }$ given the current latent state z and the robot’s action a. Middle: Leveraging the learned predictive distribution, we construct an uncertainty set of latent dynamics, calibrated to include plausible transitions while excluding implausible out-of-distribution states $z _ { \mathrm { O O D } } ^ { \prime } .$ . Within this $\mathrm { s e t , }$ we model a latent disturbance as a perturbation to the learned dynamics that yields worst-case plausible next states $\hat { z } ^ { \prime } .$ . Right: This enables robust decision-making in the latent space of world models, including robust safety filtering that proactively safeguards the system without overestimating safety and robust policy steering that selects actions that remain effective under their worst-case plausible outcomes.

Abstract—In this paper, we study robust decision-making in the latent space of world models (WMs). Robust optimization is a mathematical framework where, given explicitly specified dynamics and physically meaningful disturbances, a robot can select actions that remain effective even under worst-case disturbances. However, applying this principle to the learned latent space of WMs introduces a fundamental challenge: because WMs have fully learned state spaces and dynamics inferred from high-dimensional observations, it is unclear how to define latent-space disturbances that faithfully represent uncertainty in the underlying physical system. Our key idea is to model a latent-space disturbance as a perturbation to the learned latent dynamics that induces pessimistic but plausible transitions. Specifically, we construct a set of plausible latent dynamics by combining a dynamics-aware similarity metric that captures physically plausible transitions with out-of-distribution detection that excludes implausible latent states. We calibrate this uncertainty set over latent dynamics using conformal prediction, ensuring that WM imaginations induced by the latent disturbance remain plausible without becoming overly pessimistic. We then jointly optimize robust robot actions and the worst-case latent disturbances within the calibrated uncertainty set through efficient game-theoretic optimization. We leverage this latentspace robust optimization framework to robustify policy steering under uncertainty, considering two paradigms: latent safety filtering and sample-and-verify style steering of a generative control policy. Our controlled simulation experiments show that our latent disturbance enables robust decision-making directly in WM latent spaces, and hardware experiments with a Franka manipulator show that modeling latent disturbances enables robust policy steering, reducing failures by 70% in safety filtering and 54% in sampling-based policy steering.

Index Terms—World Models, Decision-Making under Uncertainty, Robust Safe Control

## I. INTRODUCTION

Consider a robot manipulator serving a sunny-side-up egg by sliding it from a spatula onto a plate (Fig. 1). This task is deceptively challenging: it not only requires precise motor control from visual observations, such as carefully tilting the spatula, but the outcome of each robot action is uncertain: the same serving motion may successfully slide the egg onto the plate or accidentally flip it sunny-side-down depending on unobserved factors such as the egg’s mass or the amount of oil on the spatula. How can the robot choose actions that are robust to these uncertainties?

A formal way to model such robust decision-making under uncertainty is via robust optimization, which selects a robot action that remains effective under any possible realization of a disturbance, which represents system uncertainty [1]:

$$
\pi _ { \mathrm { r o b } } ( s ) = \arg \operatorname* { m i n } _ { a \in \boldsymbol { \mathcal { A } } } \operatorname* { m a x } _ { d \in D } J \left( \boldsymbol { f } ( s , a , d ) \right) .\tag{1}
$$

Here, f denotes discrete-time system dynamics conditioned on state s, action $^ { a , }$ and disturbance d representing uncertain factors beyond the robot’s control; D denotes the set of such disturbances, and the objective function J evaluates the resulting future evolution. This optimization can be viewed as a dynamic game [2], [3]: the outer minimization selects a robust action that remains effective despite the worst-case disturbance selected by the inner maximization [4], [5], [6].

Robust optimization typically requires a structured dynamical system model, where the state and disturbance are represented, often by an expert, to have clear physical meaning (e.g., position and velocity as states and external forces as disturbances [7], [2]). However, latent world models (WMs) [8], [9] have recently shown promise for representing hard-tomodel systems directly from high-dimensional sensor observations. By jointly learning compact latent representations and their associated dynamics from datasets of robot observations and actions, WMs enable complex systems to be modeled directly in a learned latent space, such as interactions with deformable objects [10] and complex rigid bodies [11].

However, extending robust optimization in (1) to the learned latent space of WMs raises a fundamental challenge: how to define a latent-space disturbance that faithfully represents uncertainty in the underlying physical system? Unlike disturbances in physically interpretable state spaces [12], arbitrary perturbations in the learned latent space need not correspond to physically plausible system evolutions. For example, in Fig. 1, factors such as friction or the amount of oil on the spatula are not directly observed or explicitly represented by the world model. This motivates the central question of our work:

## How can we represent disturbances in the latent

## space of world models for robust optimization?

Our key idea is to model a latent-space disturbance as a perturbation to the learned latent dynamics that induces pessimistic world-model imaginations, based on the observation that predictive uncertainty in the latent dynamics reflects disturbances in the underlying system. The main challenge is to construct an uncertainty set [6] of latent disturbances, a set of plausible latent dynamics that is broad enough to cover possible transitions while excluding implausible ones that would lead to overly conservative decisions [13], [14], [15]. This is particularly important in high-dimensional latent spaces, where nearby or high-likelihood states may still correspond to physically infeasible or hallucinated out-of-distribution (OOD) transitions [11], [16]. If such OOD transitions are not properly excluded from the uncertainty set, the disturbance can exploit these regions, producing implausibly pessimistic imaginations and leading to overly conservative decisions.

To address this, we propose Latent-space Uncertainty-Calibrated In-distribution Disturbance (LUCID), a model of latent-space disturbance, which combines a dynamics-aware similarity metric with out-of-distribution detection to construct an uncertainty set of plausible latent dynamics. We calibrate this uncertainty set using conformal prediction [17], [18], ensuring that world-model imaginations induced by the latent disturbance remain plausible without becoming overly pessimistic. We then jointly optimize the worst-case disturbance and robust action through game-theoretic optimization, using an efficient parameterization of the latent disturbance for tractable optimization. We instantiate LUCID for robust policy steering, including latent safety filtering [10], [11], [19] and sampling-based policy steering [20], [21], which can anticipate potential failures without overestimating safety and steer the policy with actions that remain safe under adverse uncertainty.

Through experiments, we show that: (i) in controlled settings with access to the true system dynamics, LUCID approximates the optimal solution to robust optimization in the world-model latent space; (ii) in high-fidelity simulated visionbased manipulation, where we can control system disturbances such as friction and mass, LUCID robustly prevents task failures despite unobservable disturbances; and (iii) in hardware experiments with a Franka robot, LUCID synthesizes a robust safety filter that proactively detects potential failures without becoming overly pessimistic, reducing the failure rate (44% → 13%), and enables robust policy steering through pessimistic yet plausible WM imaginations, improving the success rate of visuomotor policies (35% → 70%).

## II. RELATED WORK

Robust Decision Making Under Uncertainty. Robust optimization formalizes robust decisions that remain effective across a set of plausible system evolutions, including worst-case realizations [22], [6], by optimizing worst-case performance over an uncertainty set. The uncertainty set characterizes plausible variations in the system [1], often in physically interpretable terms such as external force [3], model errors [23], [24], or uncertainty estimated from data [25], [13]. Similarly, distributionally robust optimization defines the uncertainty set over probability distributions to account for distribution shifts in environments or model dynamics [26], [27]. The problem is commonly formulated as a game between the controller and an adversary that selects worst-case disturbances [26], [28], [27] with game-theoretic methods [12], [3] or Lagrangian approximations [29], [27]. A related perspective is risk-sensitive control [30], [31], [32], [33], which accounts for adverse outcomes using risk measures such as conditional value-at-risk (CVaR) [34]. Certain risk measures can also be expressed as worst-case expectations over corresponding sets of probability distributions, providing a connection between risk-sensitive and robust optimization [35]. In robust reinforcement learning, policies are optimized against adverse perturbations to observations [36], [37], [27], dynamics variations [5], [38], distributional shifts encountered at deployment, and outof-distribution state-action regions [39], [40], [41], [42].

A key design choice in robust optimization is the disturbance and its uncertainty set, often defined using physically meaningful bounds specified by experts [12], [3], [43] or estimated from data [13], [44], [45], [46], where an overly broad set may admit implausible system evolutions, leading to overly conservative decisions [47]. This issue is more pronounced in the learned latent space of world models, where latent-space similarity does not necessarily reflect the underlying physical dynamics [48] and may include implausible out-of-distribution states [11], [49], [16]. Our work addresses this challenge by constructing a conformalized uncertainty set over plausible latent dynamics, optimizing for worst-case within this set to produce pessimistic but plausible world-model imaginations.

World Models in Robotics. World models predict the outcomes of robot actions directly from high-dimensional observations such as RGB images, enabling vision-based policy learning [8], planning [50], policy evaluation [51], [52], and policy improvement [53], [54] through imaginations [55], [56], [57]. Compared to video world models that predict future observations directly in pixel space [58], [15], latent world models jointly learn compact latent representations and their associated dynamics [59], [9], enabling efficient decisionmaking directly in the learned latent space [10], [11], [60]. Because world models operate under partial observability, latent dynamics models are often probabilistic models such as recurrent state-space models [61] and diffusion models [62]. However, defining disturbances that meaningfully represent uncertainties for robust optimization is challenging [27], [63], while model errors and unseen inputs can further produce hallucinated states during imagination [11], [16]. Consequently, world models typically use finite samples from learned dynamics [64], [65], [66], which can miss rare but high-impact outcomes in high-dimensional spaces [15]. Our work introduces a latent-space disturbance that directly searches for worst-case latent dynamics with gradient-based optimization.

Latent Safety Filters. Safety filtering is a control-theoretic approach that safeguards dynamical systems from failures given known safety constraints and a system dynamics model [67]. Safety filters can be implemented through minimally restrictive switching with a safety fallback policy synthesized via Hamilton–Jacobi (HJ) reachability analysis [7], control barrier functions (CBFs) [28], or predictive rolloutbased methods such as model predictive safety filtering [68], [69]. While such filters can be synthesized using dynamic programming [70], self-supervised learning [71] and reinforcement learning (RL) [23] have enabled scalable approximations for high-dimensional systems [72]. Latent safety filters extend this idea to the latent space of learned world models, enabling safety filtering for hard-to-model tasks and safety constraints from high-dimensional observations [10], [11], [19]. By anticipating future failures within world-model imaginations, these methods can proactively steer robots away from failures such as spilling the contents of deformable bags [73] or toppling complex rigid-body structures [11]. However, existing latent safety filters evaluate safety under nominal latent dynamics and may therefore overestimate safety under optimistic imaginations. Our method enables the synthesis of robust safety filters using game-theoretic adversarial RL [12], [3], [43] in the learned latent space of world models.

## III. PRELIMINARIES: LATENT WORLD MODELS

System. The robot operates from high-dimensional observations $o _ { t } ~ = ~ \mathcal { H } ( s _ { t } ) ~ \in ~ \mathcal { O }$ (e.g., RGB images), where $\mathcal { H } : \mathcal { S }  \mathcal { O }$ denotes the sensor mapping from the privileged state to the observation. We assume that the underlying system $s _ { t + 1 } = f ( s _ { t } , a _ { t } , d _ { t } )$ is a discrete-time system with a privileged state and continuous control input $\overset { \cdot } { \underset { t } { \in \mathcal { A } } } \subset \mathbb { R } ^ { | \mathcal { A } | }$ , and the disturbance $d _ { t } \in D$ is not observable to the robot, inducing uncertainty in the system. In classical control, a disturbance set D is typically specified by human experts or identified from data and often carries a physical meaning (e.g., bounds on wind or actuation error) [23], [67], [72].

Latent World Model. To represent hard-to-model systems, we learn a latent world model from a dataset of observationaction trajectories, $\mathcal { D } _ { \operatorname { t r a i n } } : = \left\{ \left( o _ { t } , a _ { t } , o _ { t + 1 } \right) _ { t = 1 } ^ { T - 1 } \right\} _ { i = 1 } ^ { N } [ 8 ] ,$ [9]. It comprises: (i) an encoder $\mathcal { E }$ that maps an observation history to latent states $z \in { \mathcal { Z } }$ , (ii) a decoder $\mathcal { G }$ that reconstructs observations from latent states, and (iii) a latent dynamics model $f _ { z } : \mathcal { Z } \times \mathcal { A }  \Delta ( \mathcal { Z } )$ that predicts the next latent states, where $\Delta$ denotes the set of distributions:

$$
{ \mathrm { E n c o d e r } } ; z _ { t } \sim { \mathcal { E } } ( z _ { t } \mid o { \le } t )
$$

$$
\mathrm { D e c o d e r } ; o _ { t } \sim \mathcal { G } ( o _ { t } \mid z _ { t } )\tag{2}
$$

Latent Dynamics: $\boldsymbol { z } _ { t } \sim f _ { z } ( \boldsymbol { z } _ { t } \mid \boldsymbol { z } _ { t - 1 } , \boldsymbol { a } _ { t - 1 } )$

World Model Training. In this work, we focus on recurrent state-space models (RSSM) [61], [8], which learn probabilistic latent dynamics with tractable conditional distributions over future latent states, using Gaussian or categorical distributions:

$$
f _ { z } ( z ^ { \prime } \mid z , a ) = \mathcal { N } \big ( \mu ( z , a ) , \Sigma ( z , a ) \big ) \mathrm { ~ o r ~ } \mathrm { C a t } \big ( \phi ( z , a ) \big ) .\tag{3}
$$

The latent world model can be trained by learning both the latent representation through a reconstruction objective and the latent dynamics through a KL-divergence term:

$$
\sum _ { t = 1 } ^ { T } \Bigl [ \underbrace { \log \mathcal { G } ( o _ { t } | z _ { t } ) } _ { \mathrm { r e c o n s t r u c t i o n } } - \underbrace { D _ { \mathrm { K L } } ( \mathcal { E } ( z _ { t } | o _ { \le t } ) \| f _ { z } ( z _ { t } | z _ { t - 1 } , a _ { t - 1 } ) ) } _ { \mathrm { l a t e n t ~ d y n a m i c s } } \Bigr ] ,\tag{4}
$$

where the $\mathcal { E } ( z _ { t } \mid o _ { \leq t } )$ provides a learned posterior target for the latent dynamics predictive distribution $f _ { z } ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } )$

Uncertainty in Latent Dynamics Represents Disturbances. Recall the example from Fig. 1, where unobservable disturbances such as the oiliness of the spatula can lead to different next states under the same action. The probabilistic latent dynamics model therefore implicitly captures these disturbances through its predictive distribution over next latent states [25], [72] (Fig. 1, left). During WM training with (4), the latent state encoded from the next observation serves as the realized transition target for the dynamics model through the KL-divergence objective. In this way, uncertainty induced by unobserved system disturbances is captured in terms of the predictive uncertainty in the learned latent dynamics.

## IV. SETUP: ROBUST OPTIMIZATION IN LATENT SPACE

Using the latent world model to predict the outcomes of robot actions, our goal is to solve the robust optimization problem in (1) over the world model’s latent space. We specify the robot’s task through a cost function $J : \mathcal { Z } $ R that evaluates a latent state. Depending on the task, J may represent an immediate cost, a cumulative cost over a finite horizon [27], or an infinite-horizon value function [28], [67]. Throughout this section, we present the optimization in single-step form for notational simplicity; the formulation also extends to multistep $( \mathrm { e . g . , ~ } T \mathrm { - s t e p } )$ or infinite-horizon settings.

Latent Disturbances as Pessimistic Imaginations. Traditionally, solving (1) requires identifying disturbances $d \in D$ that induce worst-case next states, but the learned latent dynamics in (2) do not explicitly model such disturbances. Nevertheless, as described in Sec. III, the latent dynamics implicitly capture the effects of system disturbances through probabilistic predictions. Based on the intuition that predictive uncertainty in the learned dynamics reflects such disturbances, we model the worst-case latent-space disturbance as selecting perturbed latent dynamics $f _ { z } ^ { d }$ around the nominal learned dynamics $f _ { z }$ , inducing plausible but pessimistic imaginations:

$$
\pi _ { \mathrm { r o b } } ( z ) = \underset { a \in \mathcal { A } } { \mathrm { a r g m i n } } \ \underset { f _ { z } ^ { d } ( z , a ) \in \mathcal { F } ( z , a ) } { \mathrm { m a x } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { d } ( \cdot | z , a ) } \left[ J ( z ^ { \prime } ) \right] ,\tag{5}
$$

where the uncertainty set $\mathcal { F } ( z , a )$ defines a set of plausible latent transitions of the underlying system for a given $( z , a )$

Running Example: Naughty 3D Dubins’ Car. In Fig. 2, we consider a discrete-time 3D Dubins’ car with state $s = [ p _ { x } , p _ { y } , \theta ]$ , speed $v = 1 \mathrm { m } / \mathrm { s } ,$ and timestep $\Delta t = 0 . 0 5 \mathrm { s } .$ A circular failure set of radius 0.25 m centered at the origin defines the cost function $J ( s ) : = 0 . 2 5 ^ { 2 } - p _ { x } ^ { 2 } - p _ { y } ^ { 2 }$ . Each state is rendered as ${ \textrm { a } } 3 \times 1 2 8 \times 1 2 8$ RGB image, and the robot’s action is the angular velocity $a _ { t } \in [ - 1 . 2 5 , 1 . 2 5 ] \mathrm { r a d } / \mathrm { s } ,$ while its naughty behavior is a disturbance introducing uncertainty by randomly flipping the sign of the control input:

![](images/e8cf9d087d01f2a08e2a320c951279c54f31adda0b919291cbdbf949b87052a1.jpg)  
Fig. 2. Naughty 3D Dubins’ Car. (a) Environment with the vehicle and a failure set at the center. (b) For an action sequence that turns right, an adverse disturbance can instead drive the vehicle into failure, making the decision nonrobust. (c) Driving straight remains safe even in the worst case. However, an implausible imagination can make this robust action appear unsafe.

$$
s _ { t + 1 } = s _ { t } + \Delta t [ v \cos ( \theta _ { t } ) , v \sin ( \theta _ { t } ) , \delta _ { t } a _ { t } ] , \delta _ { t } \in \{ - 1 , 1 \} .\tag{6}
$$

This disturbance is unobservable to the robot, making the underlying dynamics uncertain: for the same state and action, two distinct plausible next states may be realized. A robust action remains safe under the worst-case plausible outcome, whereas a non-robust action may appear safe nominally but fail under an adverse disturbance.

Challenges: Overly Pessimistic Disturbances. While (5) formalizes robust optimization in the latent space of the world model, constructing an uncertainty set of latent dynamics is challenging because a latent-space distance metric can be misleading for two reasons. First, distance in a high-dimensional latent space does not necessarily reflect physical or dynamical plausibility, so nearby latent states may still correspond to infeasible outcomes. Second, learned latent dynamics may assign high likelihood to implausible out-of-distribution states, allowing the latent disturbance to exploit them and produce implausible imaginations. Consequently, a poorly constructed uncertainty set can admit overly pessimistic imaginations and make robust optimization overly conservative.

For example, an action sequence of turning right in Fig. 2(b) is a non-robust action because a control-sign flip can steer the vehicle into the central failure set, whereas driving straight in Fig. 2(c) is a robust action sequence because it remains safe under all possible control-sign flips. However, an implausible latent disturbance can imagine the vehicle turning into the failure set, making the robust action appear unsafe.

## V. CONFORMALIZED LATENT-SPACE DISTURBANCE

Our key idea is to model a latent-space disturbance as an alternative latent dynamics model within a conformalized uncertainty set over plausible latent dynamics. To prevent the latent-space disturbance from becoming overly pessimistic, we adopt a dynamics-aware similarity metric together with outof-distribution detection (Sec. V-A), and then calibrate this set using conformal prediction (Sec. V-B) (Fig. 1). Finally, we jointly compute the robust action and the worst-case latent disturbance through game-theoretic optimization (Sec. V-C).

## A. Uncertainty Set over Plausible Latent Dynamics

Dynamics-Aware Similarity. Recall that the latent dynamics $f _ { z } ( z _ { t + 1 } \mid z _ { t } , a _ { t } )$ is trained to match the distribution over next latent states induced by next observations $\mathcal { E } ( z _ { t + 1 } \mid o _ { \leq t + 1 } )$ using the KL divergence in (4). This motivates using the same KL divergence to define the uncertainty set over plausible latent transitions, yielding a dynamics-aware notion of similarity rather than relying on an arbitrary distance in latent space. Specifically, for each transition from a latent-action pair $( z , a )$ we define the dynamics-aware uncertainty set over plausible latent dynamics $\dot { f } _ { z } ^ { d }$ as a left-KL ball around the learned predictive distribution<sup>1</sup>: $D _ { \mathrm { K L } } \big ( f _ { z } ^ { d } ( z , a ) \big | \big | f _ { z } ( z , a ) \big ) \le \epsilon _ { \mathrm { K L } }$ , where $\epsilon _ { \mathrm { K L } }$ controls the size of the uncertainty set to ensure coverage. This dynamics-aware KL ball contains alternative latent transitions consistent with the learned distribution, whereas a Euclidean ball may include nearby but dynamically infeasible transitions.

In-Distribution Constraint. While the KL ball can provide coverage by setting its radius to include plausible transitions, in high-dimensional latent spaces, coverage alone does not ensure precision: latent states with high likelihood under the predictive distribution may still be implausible, or out-ofdistribution (OOD) [11], [16]. The disturbance may exploit such states, leading to overly pessimistic imaginations. We therefore augment the uncertainty set with an in-distribution constraint using an OOD score function s<sub>OOD</sub>(z):

$$
\begin{array} { r l r } & { } & { \mathrm { D y n a m i c s - a w a r e ~ S i m i l a r i t y } } \\ & { } & { \mathcal { F } ( f _ { z } ; z , a ) : = \biggr \{ f _ { z } ^ { d } \in \Delta ( \mathcal { Z } ) : D _ { \mathrm { K L } } \big ( f _ { z } ^ { d } ( z , a ) \big | \big | f _ { z } ( z , a ) \big ) \le \epsilon _ { \mathrm { K L } } , } \\ & { } & { \big \mathbb { P } _ { z ^ { \prime } \sim f _ { z } ^ { d } ( \cdot | z , a ) } \big ( s _ { \mathrm { O O D } } ( z ^ { \prime } ) > \epsilon _ { \mathrm { O O D } } \big ) \le \alpha _ { \mathrm { o o d } } \biggr \} , \ ( 7 ) } \\ & { } & { \mathrm { ~ \quad ~ \uparrow ~ I n - d i s t r i b u t i o n ~ C o n s t r a i n t } } \end{array}
$$

where $\epsilon _ { \mathrm { O O D } }$ is the OOD threshold and $\alpha _ { \mathrm { o o d } }$ is the error rate. In practice, we instantiate s<sub>OOD</sub> using a flow-matching-based density estimator (logpZO) [74] trained on training latent states, where smaller scores indicate in-distribution.

## B. Conformal Calibration of the Uncertainty Set

Because the uncertainty set depends on learned latent dynamics and OOD score functions, its thresholds must be calibrated to ensure that plausible transitions are captured without making the uncertainty set overly conservative. We therefore use conformal prediction (CP) [17], [18], [46], a distribution-free statistical method, to calibrate two quantities: the KL-ball radius $\epsilon _ { \mathrm { K L } }$ and the OOD threshold ϵ<sub>OOD</sub>. Since transitions within a trajectory are temporally dependent, we perform both calibrations at the trajectory level using a heldout in-distribution calibration dataset of observation-action trajectories: $\mathcal { D } _ { \mathrm { c a l i b } } : = \left\{ ( o _ { t } ^ { i } , a _ { t } ^ { i } , o _ { t + 1 } ^ { i } ) _ { t = 1 } ^ { T - 1 } \right\} _ { i = 1 } ^ { N _ { \mathrm { c a l i b } } }$ . Proofs and additional details for this section are provided in Appendix B.

Calibrating the Dynamics-Aware Similarity. Our goal is to calibrate the KL-ball radius $\epsilon _ { \mathrm { K L } }$ such that the KL ball around the learned latent dynamics prediction contains the ground-truth latent dynamics realized by the systems with high probability. Mirroring the training objective of the latent dynamics model (4), we adopt the nonconformity score $D _ { \mathrm { K L } } \left( \mathcal { E } ( z _ { t } \mid o _ { \leq t } ) \parallel f _ { z } ( z _ { t } \mid z _ { t - 1 } , a _ { t - 1 } ) \right)$ , and apply conformal prediction with a user-specified miscoverage level $\alpha _ { \mathrm { K L } } ~ \in ~ [ 0 , 1 ]$ . Under the exchangeability assumption, the calibrated uncertainty set guarantees inclusion of the test-time latent representation encoded from observations within the uncertainty set with probability at least $1 - \alpha _ { \mathrm { K L } }$

$$
\begin{array} { r } { \mathbb { P } \left( D _ { \mathrm { K L } } \big ( \mathcal { E } ( o _ { \le t } ^ { \mathrm { t e s t } } ) \| f _ { z } ( z _ { t - 1 } ^ { \mathrm { t e s t } } , a _ { t - 1 } ^ { \mathrm { t e s t } } ) \big ) \le \epsilon _ { \mathrm { K L } } \right) \ge 1 - \alpha _ { \mathrm { K L } } . } \end{array}\tag{8}
$$

Calibrating the In-Distribution Constraint. While we aim to classify a latent state as OOD when $s _ { \mathrm { O O D } } ( z ) ~ > ~ \epsilon _ { \mathrm { O O D } } ,$ OOD latent states are, by definition, not directly available [11]; we only have access to in-distribution states. Similar to [11], we therefore adopt class-conditional conformal prediction [75] to calibrate ϵ<sub>OOD</sub> using in-distribution states encoded from the calibration dataset, providing recall guarantees for detecting in-distribution states with a miscoverage level $\alpha _ { \mathrm { o o d } } \in [ 0 , 1 ] ;$

$$
\begin{array} { r } { \mathbb { P } \left( s _ { \mathrm { O O D } } ( z _ { \mathrm { I D } } ^ { \mathrm { t e s t } } ) \le \epsilon _ { \mathrm { O O D } } \right) \ge 1 - \alpha _ { \mathrm { o o d } } . } \end{array}\tag{9}
$$

Intuitively, an in-distribution latent state is accepted as indistribution with probability at least $1 - \alpha _ { \mathrm { o o d } }$ , while latent states whose OOD scores exceed ϵ<sub>OOD</sub> are less likely to be in-distribution and are therefore classified as OOD.

Trajectory-Level Calibration. (8) and (9) assume exchangeability of the calibration and test samples. However, individual transitions within a trajectory are temporally dependent and therefore cannot be treated as exchangeable. To address this, similar to [76], we perform calibration at the trajectory level, assuming that the calibration trajectories and the test trajectory are exchangeable, and causally reconstruct the state-level thresholds from the trajectory-level result.

Given a trajectory $\tau _ { i } : = \{ \big ( o _ { t } ^ { i } , a _ { t } ^ { i } , o _ { t + 1 } ^ { i } \big ) \} _ { t = 1 } ^ { T - 1 } \in \mathcal D _ { \mathrm { c a l i b } }$ with state-level nonconformity scores $s _ { t } ^ { i } ,$ we define the trajectorylevel nonconformity score as $S ( \tau _ { i } ) : = \operatorname* { m a x } _ { t } s _ { t } ^ { i }$ and compute the (1−α)-quantile of $\{ S ( \tau _ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { c a l i b } } }$ to obtain the calibrated threshold $\epsilon ^ { \mathrm { t r a j } }$ . This trajectory-level threshold can then be causally reconstructed as the state-level threshold:

$$
\mathbb { P } \left( \operatorname* { m a x } _ { t } s _ { t } ^ { \mathrm { t e s t } } \leq \epsilon ^ { \mathrm { t r a j } } \right) = \mathbb { P } \left( s _ { t } ^ { \mathrm { t e s t } } \leq \epsilon ^ { \mathrm { t r a j } } , \forall t \right) \geq 1 - \alpha ,\tag{10}
$$

where $\begin{array} { r l r } { ( \epsilon , \alpha ) } & { { } \in } & { \left\{ ( \epsilon _ { \mathrm { K L } } , \alpha _ { \mathrm { K L } } ) , ( \epsilon _ { \mathrm { O O D } } , \alpha _ { \mathrm { o o d } } ) \right\} } \end{array}$ . Thus, the trajectory-level threshold can be causally applied as the corresponding state-level threshold at each test-time transition.

## C. Solving Latent-Space Robust Optimization

With a calibrated model of latent disturbances, we now solve the latent-space robust optimization problem in (5) by jointly optimizing the latent disturbance within the conformalized uncertainty set and the action that minimizes the cost under the imagination induced by the worst-case latent disturbance.

Efficient Disturbance Parameterization. Directly searching over all admissible dynamics in the uncertainty set is intractable in high-dimensional latent spaces (e.g., 1K dims). We therefore parameterize the latent-space disturbance as a residual perturbation to the learned latent dynamics. Specifically, a latent-space disturbance network $\pi _ { d }$ outputs an additive residual to the next latent state distribution parameters:

$$
\pi _ { d } \left( z , a \right) = ( \Delta \mu \left( z , a \right) , \Delta \Sigma \left( z , a \right) ) \quad \mathrm { o r } \quad \Delta \phi \left( z , a \right) ,\tag{11}
$$

yielding the latent dynamics conditioned on $\pi _ { d }$ and $f _ { z } { \mathrm { : } }$

$$
f _ { z } ^ { \pi _ { d } } ( z ^ { \prime } \mid z , a ) = \mathcal { N } ( \mu + \Delta \mu , \Sigma + \Delta \Sigma ) \mathrm { ~ o r ~ } \mathrm { C a t } ( \phi + \Delta \phi ) .\tag{12}
$$

Game-Theoretic Optimization. We compute the worst-case latent disturbance $f _ { z } ^ { \pi _ { d } } ( \cdot \mid z , a )$ for each latent state and robot action pair $( z , a )$ by softly relaxing the uncertainty set into hinge penalties. Since both the latent dynamics and OOD score function are differentiable, we reformulate the inner maximization of (5) with coefficients $\lambda _ { \mathrm { O O D } } , \lambda _ { \mathrm { K L } } > 0 \colon$

$$
\begin{array} { r l } & { \underset { f _ { z } ^ { \pi } d } { \operatorname* { m a x } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( z , a ) } \Big [ J ( z ^ { \prime } ) - \lambda _ { \mathrm { O O D } } \big [ s _ { \mathrm { O O D } } ( z ^ { \prime } ) - \epsilon _ { \mathrm { O O D } } \big ] _ { + } ~ ( 1 3 ) } \\ & { ~ - \lambda _ { \mathrm { K L } } \Big [ D _ { \mathrm { K L } } \big ( f _ { z } ^ { \pi _ { d } } ( z , a ) \| f _ { z } ( z , a ) \big ) - \epsilon _ { \mathrm { K L } } \Big ] _ { + } \Big ] , } \end{array}
$$

where $[ x ] _ { + } : = \operatorname* { m a x } ( x , 0 )$ applies a penalty only when a constraint is violated. The $\pi _ { d }$ is optimized through differentiable latent dynamics with gradient ascent, and the robust action is then optimized under the resulting latent disturbance:

$$
\pi _ { \mathrm { r o b } } ( z ) = \arg \operatorname* { m i n } _ { a } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( \cdot | z , a ) } \left[ J ( z ^ { \prime } ) \right] ,\tag{14}
$$

where the robust action and latent-space disturbance form a two-player minimax game that can be solved iteratively [27]. Following [77], we update the disturbance faster than the action to promote convergence, providing a practical approximation to the latent-space robust optimization problem in (5).

## VI. APPLICATION: ROBUST POLICY STEERING

## A. Setup: Policy Steering with World Models

Modeling latent disturbances allows us to instantiate a suite of runtime steering techniques that guide a visuomotor policy robustly away from hard-to-model failures. We consider two steering paradigms: (i) latent safety filters, which safeguard a task policy using a safety monitor and safe controller computed from world-model imaginations [10], [11], [19], [78]; and (ii) sample-and-verify, which samples multiple action candidates from a task policy, evaluates their imagined outcomes using the world model, and selects the best action [20], [21]. Existing WM-based policy steering methods predominantly rely on nominal world-model imaginations both when synthesizing the safety controller and during runtime action selection. As a result, they can overestimate safety under optimistic predictions and produce non-robust safety actions that remain effective only under nominal system realizations.

Failure Specification. Hard-to-model constraints for policies are specified via a latent failure set $F : = \{ z : \ell _ { z } ( z ) \leq 0 \} \subset \mathcal { Z }$ defined as the zero-sublevel set of a latent failure margin function $\ell _ { z } : \mathcal { Z } \to \mathbb { R }$ . The margin function $\ell _ { z } \in [ - 1 , 1 ]$ is, in practice, instantiated as a binary classifier.

Steering Method #1: Latent Safety Filters. A latent safety filter synthesizes two components with WM: a safety value function $V ^ { \pmb { \sigma } } : \mathcal { Z }  \mathbb { R }$ , which measures how close the current latent state of the robot is to inevitable failures, and a safetypreserving policy $\pi ^ { \bullet } : \mathcal { Z }  \mathcal { A }$ , which steers the robot away from failure. The latent safety filter computes these functions using the learned latent dynamics $f _ { z }$ through Hamilton–Jacobi (HJ) reachability analysis [4], [67], satisfying the latent-space safety Bellman equation [10]:

$$
\mathrm { N o m i n a l \ L a t e n t \ D y n a m i c s }
$$

$$
V ^ { \pmb { \sigma } } ( z ) = \operatorname* { m i n } \Big \{ \ell _ { z } ( z ) , \mathop { \operatorname* { m a x } } _ { a \in \mathcal { A } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ( \cdot | z , a ) } \left[ V ^ { \pmb { \sigma } } ( z ^ { \prime } ) \right] \Big \} ,\tag{15a}
$$

$$
\pi ^ { \bullet } ( z ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ( \cdot | z , a ) } \left[ V ^ { \bullet } ( z ^ { \prime } ) \right] ,\tag{15b}
$$

where a discount factor $\gamma \in \ [ 0 , 1 )$ can be incorporated to induce a contraction mapping [23]. The zero-sublevel set of the safety value, $U : = \{ z : V ^ { \bullet } ( z ) \leq 0 \}$ , defines the unsafe set of latent states from which failure cannot be avoided.

At runtime, the safety filter safeguards an arbitrary task policy $\pi ^ { \mathrm { t a s k } }$ by evaluating its proposed action in a least-restrictive fashion [7]. Given the current latent state z, it estimates the expected safety value of next states $z ^ { \prime } \sim f _ { z } ( \cdot \mid z , \pi ^ { \mathrm { t a s k } } ( z ) )$ , and the task action is executed if the predicted safety value is positive; otherwise, it is overridden by the fallback policy:

$$
\pi ^ { \mathrm { s t e e r } } ( z ) = \mathbb { 1 } _ { \{ \mathbb { E } _ { z ^ { \prime } } [ V ^ { \bullet } ( z ^ { \prime } ) ] > 0 \} } \cdot \pi ^ { \mathrm { t a s k } } ( z ) + \mathbb { 1 } _ { \{ \mathbb { E } _ { z ^ { \prime } } [ V ^ { \bullet } ( z ^ { \prime } ) ] \leq 0 \} } \cdot \pi ^ { \bullet } ( z ) .\tag{16}
$$

Steering Method #2: Sample-and-Verify. Given K candidate action sequences $\{ \mathbf { a } ^ { k } \} _ { k = 1 } ^ { k = K } \sim \pi ^ { \mathrm { t a s k } } ( z )$ sampled from a task policy (e.g., diffusion policy $[ 7 9 ] )$ , we evaluate each candidate using WM imaginations and select the best action sequence:

$$
\pi ^ { \mathrm { s t e e r } } ( z ) = \arg \operatorname* { m i n } _ { \mathbf { a } \in \{ \mathbf { a } ^ { ( k ) } \} _ { k = 1 } ^ { K } } \mathbb { E } _ { \mathbf { z } ^ { \prime } \sim f _ { z } ( \cdot | z , \mathbf { a } ) } \left[ J ( \mathbf { z } ^ { \prime } ) \right] ,\tag{17}
$$

where $\mathbf { z } ^ { \prime } : = z _ { i : i + H }$ denotes the latent states autoregressively generated over the action chunk of length H using nominal imagination $z _ { i + 1 } ^ { k } \sim f _ { z } ( \cdot \mid z _ { i } ^ { k } , a _ { i } ^ { k } )$ starting from $z _ { i }$ , and J is a cost function that scores its outcome, such as a latent failure margin [60], safety value function [78], or a VLM-based verifier applied to decoded images [21].

Limitation: Nominal Outcome under Uncertainty. While policy steering methods with latent world models show promise in steering robots away from hard-to-model failures [10], [11], they can become non-robust in tasks with uncertain dynamics because both the synthesis in (15) and the runtime evaluation in (16) and (17) rely only on the nominal WM imaginations. When disturbances in the underlying system can induce qualitatively different outcomes for the same action, the robot may choose non-robust actions based on overly optimistic evaluations.

## B. Robustifying Policy Steering

Both policy-steering paradigms in Sec. VI-A require a latent disturbance that drives the system toward failure through a sequence of plausible future transitions, rather than a single worst-case transition. To learn such a worst-case latent disturbance, we instantiate LUCID using latent-space Hamilton– Jacobi–Isaacs (HJI) reachability analysis [4], which recursively evaluates future safety over successive latent transitions.

Latent-Space HJI Reachability Analysis. With the conformalized latent-space disturbance $f _ { z } ^ { d } \in \mathcal { F } ,$ , we formulate infinite-horizon robust decision making by replacing (15) with a latent-space Bellman–Isaacs fixed-point equation [4], [12], the discrete-time dynamic-programming counterpart of Hamilton–Jacobi–Isaacs (HJI) reachability analysis:

$$
V _ { \mathrm { r o b } } ( z ) \mathrm { = } \operatorname* { m i n } \left\{ \ell _ { z } ( z ) , \underset { a \in \mathcal { A } } { \operatorname* { m a x } } \underset { f _ { z } ^ { d } \in \mathcal { F } } { \operatorname* { m i n } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { d } ( \cdot | z , a ) } \left[ V _ { \mathrm { r o b } } ( z ^ { \prime } ) \right] \right\} .\tag{18}
$$

Similarly, the zero-sublevel set $U _ { \mathrm { r o b } } : = \{ z : V _ { \mathrm { r o b } } ( z ) \leq 0 \}$ defines a robust latent unsafe set, and the corresponding robust safety policy and worst-case latent-space disturbance $f _ { z } ^ { \pi _ { d } }$ are then formulated as a two-player game optimizing against $V _ { \mathrm { r o b } } { \mathrm { : } }$

$$
f _ { z } ^ { \pi _ { d } } ( \cdot | z , a ) = \underset { f _ { z } ^ { d } ( z , a ) \in \mathcal { F } ( z , a ) } { \mathrm { a r g m i n } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { d } ( \cdot | z , a ) } \left[ V _ { \mathrm { r o b } } ( z ^ { \prime } ) \right] ,\tag{19a}
$$

$$
\pi _ { \mathrm { r o b } } ( z ) = \underset { a \in \mathcal { A } } { \mathrm { a r g m a x } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( \cdot | z , a ) } \left[ V _ { \mathrm { r o b } } ( z ^ { \prime } ) \right] .\tag{19b}
$$

Computation via Adversarial RL. To compute the solution of the latent-space HJI reachability analysis, we approximate the dynamic game in (18)–(19) using off-policy actor–critic learning with Soft Actor-Critic (SAC) [80], using the WM as a simulator. We learn three networks: (i) a robust safety critic $Q _ { \mathrm { r o b } } ( z , a )$ , with $V _ { \mathrm { r o b } } ( z ) ~ = ~ Q _ { \mathrm { r o b } } ( z , \pi _ { \mathrm { r o b } } ( z ) )$ ; (ii) a latentspace disturbance network $\pi _ { d } ( z , a )$ that predicts residual distribution parameters for the nominal latent dynamics $f _ { z } ( z , a )$ yielding the pessimistic dynamics $f _ { z } ^ { \pi _ { d } } ( z , a )$ , as in (11); and (iii) a robust safety policy $\pi _ { \mathrm { r o b } } ( z )$

Starting from latent states encoded from $\mathcal { D } _ { \mathrm { t r a i n } }$ , we generate imaginations with $\pi _ { \mathrm { r o b } }$ and $f _ { z } ^ { \pi _ { d } }$ , and store the latent-action pairs $\{ ( z , a ) \}$ in a replay buffer B. At each iteration, we sample transitions from B and update the robust safety critic using the discounted safety Bellman equation, with $\gamma \in [ 0 , 1 )$ [23]:

$$
\begin{array} { r l } & { \mathcal { L } _ { Q _ { \mathrm { r o b } } } = \mathbb { E } _ { \ b { \mathcal { B } } } \Big [ \Big ( Q _ { \mathrm { r o b } } ( z , a ) - \Big ( ( 1 - \gamma ) \ell _ { z } ( z ) } \\ & { \qquad + \gamma \operatorname* { m i n } \Big \{ \ell _ { z } ( z ) , \hat { Q } _ { \mathrm { r o b } } ( z ^ { \prime } , a ^ { \prime } ) \Big \} \Big ) \Big ) ^ { 2 } \Big ] , } \end{array}\tag{20}
$$

where the next latent state $z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( \cdot \mid z , a )$ is sampled from the latent dynamics induced by the current disturbance policy, and $a ^ { \prime } \sim \pi _ { \mathrm { r o b } } ( \cdot \mid z ^ { \prime } )$ is sampled from the current robust safety policy. The latent-space disturbance $f _ { z } ^ { \pi _ { d } }$ is then updated to minimize the robust safety value while remaining within the calibrated uncertainty set, following (13):

$$
\begin{array} { r } { \mathcal { L } _ { \pi _ { d } } : = \mathbb { E } _ { \mathcal { B } } \Big [ \underbrace { Q _ { \mathrm { r o b } } ( z ^ { \prime } , \pi _ { \mathrm { r o b } } ( z ^ { \prime } ) ) } _ { \mathrm { M i n i m i z e ~ V a l u e } } + \underbrace { \lambda _ { \mathrm { O O D } } \big [ s _ { \mathrm { O O D } } ( z ^ { \prime } ) - \epsilon _ { \mathrm { O O D } } \big ] _ { \pm } } _ { \mathrm { I n - D i s t r i b u t i o n ~ C o n s t r a i n t } } } \end{array}
$$

$$
+ \underbrace { \lambda _ { \mathrm { K L } } \left[ D _ { \mathrm { K L } } ( f _ { z } ^ { \pi _ { d } } ( \cdot \vert z , a ) \Vert f _ { z } ( \cdot \vert z , a ) ) - \epsilon _ { \mathrm { K L } } \right] _ { + } } _ { \mathrm { D y n a m i c s - a w a r e ~ U n c e r t a i n t y ~ S e t ~ C o n s t r a i n t } } ] ,\tag{21a}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \pi _ { \mathrm { r o b } } } : = \mathbb { E } _ { \mathcal { B } } \Big [ \underbrace { - Q _ { \mathrm { r o b } } ( z , a ^ { \prime } ) } _ { \mathrm { M a x i m i z e ~ V a l u e } } + \underbrace { \alpha _ { \mathrm { e n t r o p y } } \log \pi _ { \mathrm { r o b } } ( a ^ { \prime } | z ) } _ { \mathrm { S A C ~ E n t r o p y ~ R e g u l a r i z a t i o n } } \Big ] . } \end{array}\tag{21b}
$$

where $\alpha _ { \mathrm { e n t r o p y } }$ is the SAC entropy-regularization coefficient.

Runtime #1: Robust Latent Safety Filters. Since the robust safety value is computed under worst-case latent dynamics, its zero-sublevel set $U _ { \mathrm { r o b } } : = \{ z : V _ { \mathrm { r o b } } ( z ) \leq 0 \}$ defines the robust unsafe set, with $V _ { \mathrm { r o b } } ( z ) = Q _ { \mathrm { r o b } } ( z , \pi _ { \mathrm { r o b } } ( z ) )$ : the latent states from which failure is unavoidable under adverse uncertainty. At deployment, we therefore evaluate the action proposed by the task policy using the learned robust safety value, and the safety policy overrides it when the value is non-positive:

$$
\pi _ { \mathrm { r o b } } ^ { \mathrm { s t e e r } } ( z ) = \left\{ { \begin{array} { l l } { \pi ^ { \mathrm { t a s k } } ( z ) , } & { { \mathrm { i f ~ } } Q _ { \mathrm { r o b } } \left( z , \pi ^ { \mathrm { t a s k } } ( z ) \right) > 0 , } \\ { \pi _ { \mathrm { r o b } } ( z ) , } & { { \mathrm { o t h e r w i s e . } } } \end{array} } \right.\tag{22}
$$

Runtime #2: Robust Sample-and-Verify. Replacing the nominal imaginations in (17) with pessimistic imaginations induced by the learned worst-case latent-space disturbance $f _ { z } ^ { \pi _ { d } }$ , we evaluate K candidate action sequences $\mathbf { a } ^ { ( k ) } \sim \pi ^ { \mathrm { t a s k } } ( z )$ with $k = 1 , \cdots , K$ , and select the best sequence:

$$
\pi _ { \mathrm { r o b } } ^ { \mathrm { s t e e r } } ( z ) = \arg \operatorname* { m i n } _ { \mathbf { a } \in \{ \mathbf { a } ^ { ( k ) } \} _ { k = 1 } ^ { K } } \mathbb { E } _ { \mathbf { z } ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( \cdot | z , \mathbf { a } ) } \left[ J ( \mathbf { z } ^ { \prime } ) \right] ,\tag{23}
$$

where $\mathbf { z } ^ { \prime }$ denotes the imagined latent trajectory autoregressively generated over the action chunk $z _ { i + 1 } ^ { k } \sim f _ { z } ^ { \pi _ { d } } \big ( \cdot \mid z _ { i } ^ { k } , a _ { i } ^ { k } \big )$ so that evaluation of sampled actions accounts for adverse outcomes under the calibrated system uncertainty.

## VII. CASE STUDY: DUBINS’ CAR WORLD MODEL WITH KNOWN DISTURBANCES

We first study our latent-space robust optimization in controlled settings where the underlying system dynamics and disturbances are known, allowing us to evaluate whether our latent-space robust optimization can recover the optimal solution directly in the latent space of world models. Throughout this section, we focus on the latent safety filter paradigm.

## A. Experimental Setup

Setup: 3D Dubins’ Car with Disturbances. Let a discretetime 3D Dubins’ car evolve under two different types of disturbances. First, we use the naughty dynamics introduced in Sec. IV, where the system randomly flips the sign of the action, inducing a discrete, multimodal transition distribution with two possible outcomes for each action. Second, we consider continuous positional disturbances, $d _ { x , t } , d _ { y , t } \in [ - 0 . 3 , 0 . 3 ]$

$$
s _ { t + 1 } = s _ { t } + \Delta t [ v \cos ( \theta _ { t } ) + d _ { x , t } , \ v \sin ( \theta _ { t } ) + d _ { y , t } , \ a _ { t } ] .\tag{24}
$$

For both settings, the ground-truth worst-case disturbance and corresponding robust action can be computed analytically, enabling evaluation of our latent-space robust optimization.

World Model. We adopt Dreamer [81] with a Recurrent State-Space Model (RSSM) [61] where latent dynamics are modeled as a Gaussian distribution [61]. The world model is trained offline on $N _ { \mathrm { t r a i n } } ~ = ~ 4 { , } 0 0 0$ observation–action trajectories, where actions and disturbances are sampled randomly for $T = 1 0 0$ timesteps. We annotate each observation with the binary ground-truth failure margin $\ell ( s ) = - 1 { \mathrm { ~ i f ~ } } s \in F$ and $\ell ( s ) = 1$ otherwise, and train a two-layer MLP on top of the latent state to predict the latent failure margin $\ell _ { z } ( z )$

Latent Safety Filter Setup. We follow Sec. VI-B to learn the latent safety filter using the pretrained world model. The latent disturbance takes the mean and covariance predicted by the nominal latent dynamics and outputs additive perturbations to both. The thresholds $\epsilon _ { \mathrm { K L } }$ and $\epsilon _ { \mathrm { O O D } }$ are calibrated on a heldout set of 1,000 trajectories with $\alpha _ { \mathrm { K L } } = \alpha _ { \mathrm { o o d } } = 0 . 0 1$

TABLE I  
3D DUBINS’ CAR WORLD MODEL: QUANTITATIVE RESULTS.
<table><tr><td rowspan="2">Method</td><td colspan="4">Naughty</td><td colspan="4">Positional Disturbances</td></tr><tr><td>B.Acc. ↑</td><td>FPR↓</td><td>FNR↓</td><td>Fail ↓</td><td>B.Acc. ↑</td><td>FPR↓</td><td>FNR↓</td><td>Fail ↓</td></tr><tr><td>Nominal</td><td>0.932</td><td>0.006</td><td>0.130</td><td>0.114</td><td>0.831</td><td>0.000</td><td>0.337</td><td>0.218</td></tr><tr><td>WoN</td><td>0.959</td><td>0.057</td><td>0.026</td><td>0.061</td><td>0.911</td><td>0.162</td><td>0.017</td><td>0.034</td></tr><tr><td>CVaR0.05</td><td>0.944</td><td>0.017</td><td>0.095</td><td>0.062</td><td>0.943</td><td>0.028</td><td>0.087</td><td>0.084</td></tr><tr><td>CVaR0.1</td><td>0.935</td><td>0.012</td><td>0.118</td><td>0.061</td><td>0.936</td><td>0.018</td><td>0.109</td><td>0.101</td></tr><tr><td> $\mathrm { C V a R } _ { 0 . 2 }$ </td><td>0.927</td><td>0.010</td><td>0.135</td><td>0.062</td><td>0.927</td><td>0.011</td><td>0.135</td><td>0.134</td></tr><tr><td>CVaR0.5</td><td>0.915</td><td>0.007</td><td>0.163</td><td>0.057</td><td>0.910</td><td>0.005</td><td>0.175</td><td>0.197</td></tr><tr><td>LUCID</td><td>0.958</td><td>0.037</td><td>0.046</td><td>0.038</td><td>0.956</td><td>0.023</td><td>0.066</td><td>0.021</td></tr></table>

![](images/9ed582988c622c15410e7423e981b93fa7a55495a55627b88f75dd4068e95eab.jpg)  
Fig. 3. Qualitative Results: 3D Dubins’ Car. Solid lines represent the ground-truth unsafe-set boundaries, and red regions denote the unsafe sets computed by each method. LUCID closely approximates the ground-truth unsafe sets of robust optimization under two different types of disturbances. In contrast, Nominal is overly optimistic, while the sampling-based baselines such as WoN and CVaR inaccurately approximate unsafe sets.

Evaluation & Metrics. With access to system dynamics, we compute the ground-truth safety value function and optimal robust action using grid-based methods [70]. We first evaluate the safety value $V ^ { \bullet }$ by its accuracy in classifying unsafe states across three state dimensions, discretized into 50 grids per dimension, reporting balanced accuracy (B.Acc.), false positive rate (FPR), and false negative rate (FNR). We then evaluate the learned safety policy with 1,000 random trajectories, comprising 500 safe and 500 that fail under the worst-case disturbance. We apply least-restrictive filtering (22) on those action sequences starting from the initial states in a closed loop, reporting the failure rate (Fail).

## B. Can LUCID Make Robust Decisions in World Models?

Baselines. We compare LUCID against the following latent safety-filter baselines, which differ in how they account for uncertainty in the learned latent dynamics, using the same WM. We first consider Nominal, which trains the safety filter following (15) that evaluates the expected safety value:

$$
V _ { \mathrm { n o m } } ^ { \pmb { \sigma } } ( z ) = \operatorname* { m i n } \left\{ \ell _ { z } ( z ) , \operatorname* { m a x } _ { a \in \mathcal { A } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ( \cdot | z , a ) } \left[ V _ { \mathrm { n o m } } ^ { \pmb { \sigma } } ( z ^ { \prime } ) \right] \right\}\tag{25}
$$

Worst-of-N (WoN) samples $N { = } 1 0$ states from the learned dynamics $z _ { i } ^ { \prime } \sim f _ { z } ( \cdot | z , a )$ and uses the least-safe sample:

$$
V _ { \mathrm { W o N } } ^ { \pmb { \sigma } } ( z ) = \operatorname* { m i n } \left\{ \ell _ { z } ( z ) , \operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { m i n } _ { i = 1 , \ldots , N } V _ { \mathrm { W o N } } ^ { \pmb { \sigma } } ( z _ { i } ^ { \prime } ) \right\} .\tag{26}
$$

Lastly, Risk-Sensitive (CVaR) learns a distributional value function [82], [83] from sampled next states of learned latent dynamics and evaluates each action using the conditional value-at-risk, $\mathrm { C V a R } _ { \beta } ,$ with $\beta \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 \}$

$$
V _ { \mathrm { C V a R } } ^ { \pmb { \sigma } } ( z ) = \operatorname* { m i n } \left\{ \ell _ { z } ( z ) , \operatorname* { m a x } _ { a \in \mathcal { A } } \mathrm { C V a R } _ { \beta } \left( V _ { \mathrm { C V a R } } ^ { \pmb { \sigma } } ( z ^ { \prime } ) \right) \right\}\tag{27}
$$

They are all trained using the discounted safety-value formulation in (20), with the safety policy optimized to maximize the learned safety value. However, unlike LUCID, these baselines account only for uncertainty captured by finite samples from the nominal learned dynamics and do not explicitly optimize a latent disturbance over the calibrated uncertainty set.

Result: LUCID Enables Robust Decision Making in Latent Space. Table I shows that LUCID learns a robust latent safety filter under both discrete and continuous system uncertainty, improving both the open-loop accuracy of the safety value function and closed-loop safety performance. As shown in Fig. 3, the learned safety value accurately identifies states from which the worst-case plausible disturbance can induce failure despite the best control action. In contrast, Nominal evaluates safety by averaging over nominal world-model futures and can therefore be overly optimistic: its high FNR indicates that many ground-truth unsafe states are predicted as safe, resulting in higher failure rates in closed-loop performance.

Result: Finite Sampling Limits Robust Decision Making. A simple way to account for uncertain latent dynamics is to use finite samples from the nominal next-state distribution, as in WoN or Risk-Sensitive with distributional critics. However, such zeroth-order optimization can miss adverse but plausible outcomes, especially as the space of possible transitions grows. Accordingly, WoN degrades more relative to LUCID under continuous disturbances than under the binary naughty disturbance. Risk-Sensitive also depends on the CVaR risk level: increasing it toward 0.5 emphasizes typical rather than adverse returns, degrading worst-case value estimation. These results motivate explicitly computing a worst-case latent disturbance that leads to plausibly pessimistic WM imaginations, using gradient ascent to search for adverse transitions within the calibrated uncertainty set rather than relying only on finite nominal sampling, consistent with the findings of [33].

## C. Does the Uncertainty Set Represent Plausible Dynamics?

To evaluate the conformalized uncertainty set in Sec. V-A and Sec. V-B, we ablate whether it effectively includes plausible transitions without admitting implausible ones.

Setup: Uncertainty Set Ablation. To test the conformalized uncertainty set, we ablate latent safety-filter training with the following setups: (i) Euclidean, which replaces the dynamicsaware similarity metric with the isotropic distance to the mean next-state prediction, with its threshold similarly calibrated; (ii) w/o $O O D ,$ which removes the in-distribution constraint;

TABLE II  
3D DUBINS’ CAR: UNCERTAINTY SET ABLATION. DYN., CONF., AND OOD REPRESENT DYNAMICS-AWARE SIMILARITY, CONFORMAL CALIBRATION, AND IN-DISTRIBUTION CONSTRAINTS, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="3">Uncertainty Set</td><td colspan="3">Naughty</td><td colspan="3">Positional Disturbances</td></tr><tr><td>Dyn.</td><td>Conf.</td><td>OOD</td><td>B.Acc. ↑</td><td>FPR↓</td><td>FNR↓</td><td>B.Acc. ↑</td><td>FPR↓</td><td>FNR↓</td></tr><tr><td>LUCID</td><td>√</td><td>√</td><td>√</td><td>0.958</td><td>0.037</td><td>0.046</td><td>0.956</td><td>0.023</td><td>0.066</td></tr><tr><td>w/o OOD</td><td>√</td><td>√</td><td>x</td><td>0.809</td><td>0.381</td><td>0.001</td><td>0.621</td><td>0.759</td><td>0.000</td></tr><tr><td> $0 . 5 \epsilon _ { \mathrm { K L } }$ </td><td>√</td><td>x</td><td>√</td><td>0.927</td><td>0.011</td><td>0.136</td><td>0.864</td><td>0.001</td><td>0.271</td></tr><tr><td> $2 . 0 \epsilon _ { \mathrm { K L } }$ </td><td>√</td><td>x</td><td>√</td><td>0.926</td><td>0.126</td><td>0.022</td><td>0.630</td><td>0.733</td><td>0.007</td></tr><tr><td>Euclidean</td><td>x</td><td>√</td><td>√</td><td>0.605</td><td>0.790</td><td>0.000</td><td>0.875</td><td>0.001</td><td>0.250</td></tr></table>

and (iii) Uncalibrated KL thresholds, which use either $0 . 5 \epsilon _ { \mathrm { K L } }$ or $2 \epsilon _ { \mathrm { K L } }$ instead of the conformally calibrated threshold.

Result: The Conformalized Uncertainty Set Prevents Over-Pessimism. Table II shows that robust optimization with uncertainty-unaware uncertainty sets can become overly pessimistic by admitting implausible latent transitions. Using a dynamics-unaware similarity metric such as Euclidean distance can include geometrically close but dynamically infeasible latent states or exclude plausible but distant ones, yielding high FPRs or FNRs in the safe set computation. Likewise, removing the in-distribution constraint (w/o OOD) allows the latent disturbance to exploit implausible OOD states and produce unrealistic imaginations, again resulting in high FPRs. In contrast, LUCID combines a dynamics-aware similarity metric with the in-distribution constraint using OOD detection to restrict the disturbance to plausible latent transitions.

Result: Conformal Calibration Balances Coverage and Pessimism. We further test whether conformal calibration selects an appropriate uncertainty-set size by ablating the calibrated KL threshold $\epsilon _ { \mathrm { K L } } .$ , which controls the radius around the nominal latent dynamics. Changing the threshold to $0 . 5 \times \epsilon _ { \mathrm { K L } }$ makes the uncertainty set too narrow and yields a high FNR, since worst-case transitions can fall outside the set and the disturbance fails to capture adverse outcomes that may occur under the true system. Conversely, increasing the threshold to $2 \times \epsilon _ { \mathrm { K L } }$ makes the uncertainty set too broad and substantially increases the FPR, as the disturbance can exploit overly pessimistic transitions that are not plausible. These results show that conformal calibration is important for balancing coverage of plausible latent dynamics against excessive conservatism.

## D. Ablation: Optimistic Imagination with Latent Disturbances

To further evaluate whether the latent disturbance can effectively perturb latent dynamics while preserving plausible world-model imaginations, we consider an optimistic imagination ablation in the naughty Dubins’ car setting.

Setup: Optimistic Latent Disturbance. By optimizing both the robot action and latent disturbance to maximize safety, we replace the minimax problem with a max–max objective, further assessing $L U C I D ' \mathrm { s }$ ability to identify alternative latent transitions within the uncertainty set while preserving plausibility. Specifically, we modify the disturbance objective in (21) to maximize the safety value while remaining within the calibrated uncertainty set. In the naughty Dubins’ car setting, this optimistic formulation corresponds to the disturbance-free system. We therefore use its ground-truth safety value without any disturbance as the evaluation reference.

TABLE III  
ABLATION: OPTIMISTIC LATENT DISTURBANCE.
<table><tr><td>Method</td><td>B.Acc. ↑</td><td>F1 ↑</td><td>FPR↓</td><td>FNR↓</td></tr><tr><td>Nominal</td><td>0.924</td><td>0.748</td><td>0.122</td><td>0.031</td></tr><tr><td>WoN</td><td>0.966</td><td>0.898</td><td>0.037</td><td>0.032</td></tr><tr><td>CVaR0.2</td><td>0.942</td><td>0.860</td><td>0.046</td><td>0.069</td></tr><tr><td>LUCID</td><td>0.966</td><td>0.947</td><td>0.009</td><td>0.059</td></tr><tr><td>w/o OOD</td><td>0.841</td><td>0.810</td><td>0.000</td><td>0.318</td></tr><tr><td>0.5€KL</td><td>0.956</td><td>0.861</td><td>0.055</td><td>0.034</td></tr><tr><td>2.0€KL</td><td>0.945</td><td>0.939</td><td>0.001</td><td>0.109</td></tr><tr><td>Euclidean</td><td>0.940</td><td>0.921</td><td>0.008</td><td>0.113</td></tr></table>

![](images/765c29ced645c68a3f8d18174d4b5d56cc5ee92e60d6b300b97558036a0d4d6e.jpg)  
Fig. 4. Qualitative: Naughty Dubins’ Car with Optimistic Disturbances. Solid lines show the ground-truth unsafe-set boundaries of the disturbancefree system, and dashed lines show the failure set. Without the in-distribution constraint, the disturbance exploits implausible latent states, trivially imagining trajectories that avoid failure from any non-failure state.

Result: Latent Disturbances Induce Effective and Plausible Imaginations. Table III shows that LUCID learns a safety value function close to that of the disturbance-free system through optimistic imagination, despite using a world model trained on trajectories collected under disturbances. The conformalized uncertainty set is essential for preserving plausibility: Fig. 4 shows that, without the in-distribution constraint, the disturbance can trivially generate implausible imaginations that avoid failure even from ground-truth unsafe states, leading to a high FNR. Dynamics-unaware uncertainty sets similarly yield high FNRs, while miscalibrated KL thresholds degrade the accuracy of value function computation.

## VIII. SIMULATION: SCALING LATENT-SPACE ROBUST OPTIMIZATION TO VISION-BASED MANIPULATION

We scale our latent-space robust optimization to a visionbased manipulation task, once again focusing on the latent safety filtering paradigm. We use IsaacLab [84], which allows us to control system parameters, such as friction coefficients, that induce uncertainty in the system dynamics.

## A. Experimental Setup: Vision-Based Block Pouring

Setup: Block Pouring. A Franka manipulator must pick up and tilt an orange block to slide a green block placed on top onto a blue block without falling off, as shown in Fig. 5. Observations consist of two tabletop 3×128×128 RGB camera images and 7-D joint-position proprioception. Actions are 6- DoF end-effector delta poses with a discrete gripper command.

Disturbance. To introduce uncertainty in the system dynamics, we randomize physics parameters including (i) the mass, (ii) friction coefficients, and (iii) restitution of the green block, without providing these parameter values to the robot.

World Model. We adopt DreamerV3 [8] with discrete latent states modeled by categorical latent dynamics, and train it on approximately 3,000 trajectories containing both successes and failures. The failure-margin function is trained on top of the latent states using state-level ground-truth failure labels.

TABLE IV  
QUANTITATIVE RESULTS: VISION-BASED BLOCK POURING.
<table><tr><td>πtask</td><td>Safety Filter</td><td>Success ↑</td><td>Failure ↓</td><td>Safety Gain ↑</td><td>Cond. Failure ↓</td></tr><tr><td rowspan="5">DreamerV3</td><td>No Filter</td><td>0.44</td><td>0.54</td><td>一</td><td></td></tr><tr><td>Nominal</td><td>0.45</td><td>0.54</td><td> $- 0 . 1 2 \pm 0 . 2 1$ </td><td>0.82 ± 0.38</td></tr><tr><td>WoN</td><td>0.46</td><td>0.50</td><td> $0 . 0 1 \pm 0 . 1 5$ </td><td>0.50 ± 0.50</td></tr><tr><td>CVaR0.1</td><td>0.39</td><td>0.48</td><td> $- 0 . 0 8 \pm 0 . 1 6$ </td><td> $0 . 4 8 \pm 0 . 5 0$ </td></tr><tr><td>UnConf.</td><td>0.00</td><td>0.43</td><td> $0 . 0 8 \pm 0 . 0 4$ </td><td> $0 . 4 3 \pm 0 . 5 0$ </td></tr><tr><td></td><td>LUCID</td><td>0.73</td><td>0.19</td><td> $0 . 2 2 \pm 0 . 2 1$ </td><td>0.20 ± 0.40</td></tr><tr><td rowspan="5">Diffusion Policy</td><td>No Filter</td><td>0.55</td><td>0.42</td><td>一</td><td></td></tr><tr><td>Nominal</td><td>0.56</td><td>0.40</td><td> $0 . 1 2 \pm 0 . 3 3$ </td><td>0.71 ± 0.46</td></tr><tr><td>WoN</td><td>0.46</td><td>0.48</td><td> $0 . 0 3 \pm 0 . 1 6$ </td><td>0.48 ± 0.50</td></tr><tr><td>CVaR0.1</td><td>0.13</td><td>0.19</td><td> $- 0 . 0 2 \pm 0 . 1 0$ </td><td> $0 . 1 9 \pm 0 . 3 9$ </td></tr><tr><td>UnConf.</td><td>0.00</td><td>0.11</td><td> $0 . 0 9 \pm 0 . 0 3$ </td><td> $0 . 1 1 \pm 0 . 3 2$ </td></tr><tr><td></td><td>LUCID</td><td>0.63</td><td>0.24</td><td> $0 . 1 8 \pm 0 . 1 9$ </td><td>0.27 ± 0.44</td></tr></table>

The task policy lifts Optimistically predicting a safe outcome, the nominal safety the bottom block. filter does not proactively intervene, leading to failuré.  
![](images/93a3455a321bfd13c2fdcdec0fadc4e8beeb449a2163c4aa1475ff9f121813c7.jpg)  
Fig. 5. Safety Value Function: Nominal vs. LUCID. Safeguarding the same $\pi ^ { \mathrm { { t a s k } } }$ from the same initial states, the robust safety filter proactively intervenes based on a plausibly pessimistic imagination $( \hat { o } _ { 2 8 } ^ { d } )$ in which $\pi ^ { \mathrm { t a s k } }$ can lead to failure, while the nominal imagination $\left( \hat { o } _ { 2 8 } \right)$ predicts a non-failure outcome. The resulting robust action prevents failure. In contrast, the nominal latent safety filter overestimates safety and intervenes too late based on an optimistic imagination (oˆ ), leading to failure under an adverse realization.

Latent Safety Filter Setup. In this section, we focus on latent safety filtering for policy steering using the least-restrictive filter in (22), which safeguards the task policy by intervening with the learned safety policy whenever the safety value of the task-policy action becomes non-positive. We follow [11] to learn the latent safety filter, and the latent disturbance takes the categorical logits predicted by the latent dynamics model and outputs additive perturbations. We calibrate the uncertainty set using 200 held-out calibration trajectories.

Task Policies. We evaluate the learned latent safety filter with three task policies $\pi ^ { \mathrm { t a s k . } }$ : (i) DreamerV3 [8], a visuomotor policy trained with dense task rewards independently from the world model used for the safety filter, (ii) Diffusion Policy [79], an imitation-learning policy trained on 500 successful trajectories, and (iii) replaying successful human teleoperation trajectories. For each policy, we roll out 2,000 trajectories with randomized initial states and physics parameters.

Baselines. Following Sec. VII, we compare against latent safety filters trained using the same world model and dataset but accounting for dynamics uncertainty in different ways: Nominal, Worst-of-N, and the risk-sensitive baseline (CVaR). We also include Unconformalized latent disturbance (Un-

TABLE V  
RESULTS: TRAJECTORY REPLAY ACROSS 20 DIFFERENT PHYSICS.
<table><tr><td>Safety Filter</td><td>Success ↑</td><td>Failure ↓</td><td>Safety Gain ↑</td><td>Cond. Failure ↓</td><td>Robust Rate ↑</td></tr><tr><td>No Filter</td><td>0.33</td><td>0.47</td><td>一</td><td></td><td>0.10</td></tr><tr><td>Nominal</td><td>0.31</td><td>0.40</td><td> $0 . 1 7 \pm 0 . 2$ </td><td> $0 . 4 6 \pm 0 . 5$ </td><td>0.11</td></tr><tr><td>WoN</td><td>0.35</td><td>0.31</td><td> $- 0 . 0 4 \pm 0 . 2$ </td><td> $0 . 3 2 \pm 0 . 4$ </td><td>0.07</td></tr><tr><td>CVaR0.1</td><td>0.30</td><td>0.31</td><td> $- 0 . 1 1 \pm 0 . 2$ </td><td> $0 . 3 2 \pm 0 . 4$ </td><td>0.10</td></tr><tr><td>UnConf.</td><td>0.05</td><td>0.06</td><td> $0 . 1 0 \pm 0 . 0 3$ </td><td> $0 . 0 6 \pm 0 . 2 3$ </td><td>0.90</td></tr><tr><td>LUCID</td><td>0.34</td><td>0.11</td><td> $0 . 2 1 \pm 0 . 1 5$ </td><td> $0 . 1 1 \pm 0 . 3 1$ </td><td>0.77</td></tr></table>

![](images/4308de604b9b1fe610d46ea5de92c157ea781abb6690d82ca52e988699974d7a.jpg)  
Fig. 6. Block Pouring: World Model Imaginations. (a) Initial state: the orange block is tilted while $\pi ^ { \mathrm { t a s k } }$ remains still. (b) The nominal imagination predicts a non-failure outcome. (c) With the conformalized uncertainty set, the latent disturbance imagines a plausible adverse transition in which the green block slides and falls. (d) Without the conformalized uncertainty set, the latent disturbance induces an implausible transition in which the green block falls in an infeasible manner, resulting in an overly pessimistic imagination.

Conf.), an ablation of LUCID that uses an uncertainty set without the in-distribution constraint.

Evaluation Metrics. Since the ground-truth solutions are unavailable, we focus on the closed-loop performance of the policy. Success is completing a task without a timeout or failure. To further assess whether the learned filter is robust, we additionally measure: (i) realized safety gain (Safety Gain): the increase in the learned safety value after each filter intervention, which measures whether the executed safety action robustly leads to a safer realized next state; and (ii) conditional failure rate (Cond. Failure): the fraction of trajectories that fail among those in which the filter intervenes at least once.

## B. Can LUCID Robustly Prevent Failures Under Uncertainty?

Result: LUCID Robustly Safeguards Task Policies. Table IV shows the closed-loop safety-filtering results under randomized physical parameters. LUCID achieves lower failure rates while maintaining higher success rates, demonstrating robustness to dynamics uncertainty without becoming overly conservative. In contrast, latent disturbances without the indistribution constraint (UnConf.) become overly pessimistic, resulting in low success rates as its imagination trivially drifts toward implausible failure states, as shown in Fig. 6.

Baselines that rely on sampling from the nominal latent dynamics (Nominal, WoN, and CVaR) are less effective, with a larger gap to LUCID than in the Dubins’ car experiments in Sec. VII. Compared to the Dubins’ car experiments, the visual manipulation task involves higher-dimensional system uncertainty and more diverse plausible next states, making zerothorder sampling less likely to capture meaningful worst-case outcomes within a limited sampling budget. Risk-sensitive distributional critics (CVaR) are likewise either ineffective or overly conservative depending on the CVaR level, since extreme tails in high-dimensional latent spaces can include implausibly pessimistic states even under random sampling. In contrast, LUCID explicitly optimizes against worst-case plausible latent dynamics within the calibrated uncertainty set.

Result: LUCID Does Not Overestimate Safety. Fig. 5 compares the nominal and robust safety values along an identical action sequence as a task policy. While the nominal safety value evaluates future safety under the average outcomes of actions and can therefore overestimate safety when adverse outcomes occur, the robust safety monitor evaluates safety under calibrated worst-case plausible outcomes and intervenes more preemptively when failure is possible, using robust actions whose plausible outcomes avoid failure. As shown in Table IV, LUCID achieves a lower conditional failure rate than the nominal filter when filtering is activated, while intervening more frequently but only when necessary, remaining less pessimistic than UnConf., which doesn’t use the in-distribution constraint for the uncertainty set construction. Moreover, LUCID yields larger positive realized safety gains, indicating that its safety actions move the system toward safer realized next states, whereas the nominal filter exhibits smaller or even negative gains under dynamics uncertainty.

## C. Ablation: Robustness Under Controlled Uncertainty

To further evaluate the robustness of safety filtering specifically to system uncertainty, independent of stochasticity in the task policy, we vary only the physics parameters of the system while replaying exactly the same task-policy action sequences.

Setup: Replaying Safe Trajectories. We evaluate the robustness to system disturbances by replaying 100 successful teleoperation trajectories from the same initial states and actions, each under 20 different physics parameter settings. Here, we also report robust rate, defined as the fraction of replay trajectories that remain non-failure across all 20 conditions.

Result: LUCID Robustly Prevents Failures Across Dynamics Variations. Table V shows that LUCID keeps replaying 77 of 100 teleoperation trajectories failure-free across all 20 physics settings, showing robustness to system disturbances. It also achieves the lowest overall and conditional failure rates among methods that retain nonzero task success; UnConf. attains lower failure rates only by being overly conservative, resulting in a near-zero success rate. This indicates that when the filter intervenes, LUCID proposes safety actions that remain effective across variations in system dynamics. Its positive realized safety gain further shows that these interventions consistently move the system toward safer realized states.

## D. Ablation: Robustness Under Partial Observability

While we have so far considered uncertainty in future transitions arising from explicit system disturbances, partial observability can also induce uncertainty even when the underlying dynamics are deterministic and disturbance-free. For example, small differences in the underlying angle of the orange block or position of the green block may be difficult to distinguish from pixel observations, yet can lead to different future outcomes under the same action. We therefore ask how our latent-space robust optimization performs when uncertainty arises from the learned world model itself, due to partial observability or model approximation.

Setup: Deterministic Block Pouring. We use the same block-pouring task, but fix the physical parameters to remove explicit disturbances from the underlying system dynamics. Thus, uncertainty arises from partial observability and approximation when mapping high-dimensional observations into the learned latent space. We collect approximately 3,000 trajectories and train the world model, latent safety filter, and task policies following the same procedure. This setting isolates uncertainty in future transitions arising from partial observability and model approximation, allowing us to evaluate whether robust latent optimization remains beneficial even when the underlying system is deterministic.

TABLE VI  
ABLATION: DETERMINISTIC BLOCK-POURING WITH FIXED PHYSICS
<table><tr><td> $\pi ^ { \mathrm { t a s k } }$ </td><td>Safety Filter</td><td>Success ↑</td><td>Failure ↓</td><td>Safety Gain ↑</td><td>Cond. Failure ↓</td></tr><tr><td rowspan="5">DreamerV3</td><td>No Filter</td><td>0.43</td><td>0.56</td><td>=</td><td>1</td></tr><tr><td>Nominal</td><td>0.60</td><td>0.39</td><td> $0 . 2 1 \pm 0 . 2 4$ </td><td> $0 . 4 0 \pm 0 . 4 8$ </td></tr><tr><td>WoN</td><td>0.34</td><td>0.54</td><td>0.10 ± 0.20</td><td> $0 . 5 4 \pm 0 . 4 9$ </td></tr><tr><td>CVaR0.1</td><td>0.00</td><td>0.34</td><td> $- 0 . 0 5 \pm 0 . 0 5$ </td><td> $0 . 3 4 \pm 0 . 4 7$ </td></tr><tr><td>UnConf.</td><td>0.00</td><td>0.00</td><td> $0 . 2 9 \pm 0 . 0 8$ </td><td> $0 . 0 0 \pm 0 . 0 6$ </td></tr><tr><td></td><td>LUCID</td><td>0.81</td><td>0.18</td><td> $0 . 2 7 \pm 0 . 2 2$ </td><td> $0 . 1 9 \pm 0 . 3 9$ </td></tr><tr><td rowspan="5">Diffusion Policy</td><td>No Filter</td><td>0.47</td><td>0.51</td><td>=</td><td>=</td></tr><tr><td>Nominal</td><td>0.57</td><td>0.35</td><td> $0 . 1 5 \pm 0 . 1 6$ </td><td> $0 . 2 5 \pm 0 . 4 8$ </td></tr><tr><td>WoN</td><td>0.28</td><td>0.66</td><td> $- 0 . 0 0 \pm 0 . 2 0$ </td><td> $0 . 6 6 \pm 0 . 4 7$ </td></tr><tr><td>CVaR0.1</td><td>0.00</td><td>0.19</td><td> $- 0 . 0 5 \pm 0 . 0 5$ </td><td> $0 . 1 9 \pm 0 . 3 9$ </td></tr><tr><td>UnConf.</td><td>0.00</td><td>0.00</td><td> $0 . 2 1 \pm 0 . 0 6$ </td><td> $0 . 0 0 \pm 0 . 0 6$ </td></tr><tr><td></td><td>LUCID</td><td>0.66</td><td>0.25</td><td> $0 . 1 6 \pm 0 . 1 6$ </td><td> $0 . 2 7 \pm 0 . 4 4$ </td></tr></table>

Result: Robust Optimization Improves Safety under Partial Observability. Table VI shows that robust latent-space optimization improves safety even when the underlying system dynamics are deterministic. In this setting, uncertainty in future transitions arises from partial observability and approximation error in the learned world model, rather than from explicit system disturbances. The nominal safety filter is more effective than in the stochastic environment, but LUCID further reduces failures and increases success by mitigating uncertainty. These results suggest that robust latent-space optimization can improve decision-making under different sources of transition uncertainty, provided that the realized test-time transition is contained within the calibrated uncertainty set.

## IX. HARDWARE: ROBUST STEERING DURING

## CONTACT-RICH VISUOMOTOR POLICY EXECUTION

Finally, we scale LUCID to a real-world visual manipulation task using a Franka Research 3. The robot must serve a sunnyside-up fried egg from a spatula onto a plate without dropping it on the table or flipping it sunny-side down. The egg’s dynamics are uncertain due to unobserved physical properties such as friction between the egg and spatula.

## A. Experimental Setup: Serving Sunny-Side-Up Fried Eggs

Setup. Observations consist of two tabletop $3 \times 1 9 2 \times 2 5 6$ RGB camera images and 7-D joint-position proprioception. Actions are 6-D end-effector delta poses with a discrete gripper command, executed at 15 Hz.

Disturbance. We introduce an explicit source of unobserved disturbance by varying the spatula surface to mimic changes in friction caused by oil and surface conditions in household kitchens. We use three surface conditions: (i) no tape, (ii) scotch tape, and (iii) heavy-duty tape (see Fig. 8).

World Model. We employ an RSSM with Gaussian latent dynamics [61] learned from frozen DINOv3 [85] embeddings, and construct the latent dynamics model using a Transformer with an 8-step history to predict the next latent-state distribution. We additionally train the model with a DINO token reconstruction objective alongside pixel reconstruction (see Appendix D for details). We collect 900 teleoperated trajectories with frame-level failure annotations, which are used to learn a latent failure margin over the learned latent space.

Actions without Filtering  
Nominal Safety Filter  
![](images/7b02ff29497bd52660bf33381aacffd149ddf7e350bb7b14b8c6ce2cc9b84910.jpg)

![](images/7a749082c07d08dd006dc493f4cf57e3d7dcfc9424b0860232ba43d757fd0c49.jpg)  
The teleoperator lowers the spatula, causing Nominal safety filter stops the teleoperator Robust safety filter stops the teleoperator earlier, the egg to slide off and fall outside the plate. too late, and the egg still flips on the plate. preventing the egg from flipping or falling.

![](images/39d688d390f0f51406b518b29dd649bc71a79a488cade6e10c4e28b28b74dc97.jpg)

![](images/16f8e6be18a95f463e1c8d2f946ca0918716a02f89c322eab692eeb076c5b706.jpg)

![](images/82dfe4e95f816382d6366b0e41b41fb5caee2a5881179a72ebbb7a10c7ce6706.jpg)

![](images/fe07684a92ec1cea7437d3a89fa2f60cd381502d9caef3d7359481f4d144ef53.jpg)  
The teleoperator tilts the spatula away from Nominal safety filter pushes the spatula toward Robust safety filter lifts and tilts the the plate, causing the egg to slide off. the plate, but the egg flips sunny-side down. spatula to prevent the egg from flipping.

Fig. 7. Qualitative Results: Preventing Sunny-Side-Down Egg. LUCID preemptively identifies teleoperator actions that may cause the egg to fall or flip and intervenes with robust actions that remain safe under worst-case dynamics. In contrast, the Nominal safety filter relies on overly optimistic safety values and non-robust actions, leading to sunny-side-down outcomes.  
![](images/eb7d0172725052162c60453030c2d90e31c00c5bc96f784c69ed588929ca7ea8.jpg)  
Fig. 8. Filtering the Same $\pi ^ { \mathrm { t a s k } }$ Across Different Surfaces. (Left) Failure rates under each spatula surface condition. (Right) Robust rate, measuring the fraction of trajectories that remain safe across all surface conditions. LUCID consistently minimizes failures and achieves a higher robust rate, whereas the Nominal safety filter prevents failures only under certain surface conditions.

Latent Safety Filter. We follow [11] to learn the latent safety filter. The latent disturbance predicts additive perturbations to the mean and covariance of the Gaussian latent dynamics, and the uncertainty set is calibrated using 50 held-out trajectories.

Task Policies. We consider three task policies that we seek to steer at runtime: (i) a Diffusion Policy [79] trained from scratch using 300 successful trajectories; (ii) a visionlanguage-action (VLA) policy fine-tuned from $\pi _ { 0 . 5 }$ [86] using the same trajectories; and (iii) a teleoperation from a human.

## B. Robust Latent Safety Filtering

We compare a Nominal latent safety filter, trained using the expected safety value (15), against LUCID trained with (18). Result: Robustness Across Surface Disturbances. We first evaluate robustness to disturbances induced by the spatula surface condition. We replay 20 success and 30 failure trajectories under all three surface conditions while safeguarding them. Fig. 8 shows that LUCID consistently achieves lower failure rates and a higher robust rate, defined as the fraction of replay trajectories that remain safe across all surface conditions. In contrast, the Nominal safety filter is effective only under certain surface conditions and achieves a lower robust rate.

![](images/a7a733ab0eee71ebee1108812335a4042770f679af8ebeff2b90eec84362ea85.jpg)  
Fig. 9. Quantitative Results: Success Rates. (Left) Safety filtering over 20 rollouts for Diffusion Policy and π<sub>0.5</sub>. (Right) Steering π<sub>0.5</sub> with sample-andverify action selection, using world-model imaginations to evaluate candidate action chunks. In both settings, LUCID robustly improves success rates.

Result: Robustly Safeguarding π<sup>task</sup>. We roll out Diffusion Policy [79] and $\pi _ { 0 . 5 }$ [86] for 20 trials each, safeguarded by either the Nominal safety filter or LUCID, with Scotch tape applied to the spatula. Fig. 9 shows that LUCID achieves a larger improvement in success rate than the Nominal filter, demonstrating more robustness against uncertain dynamics.

Fig. 7 shows qualitative results safeguarding the teleoperator. LUCID preemptively overrides unsafe actions, whereas the Nominal safety filter may fail to intervene in time due to overly optimistic imaginations. Moreover, LUCID intervenes with robust fallback actions whose plausible outcomes remain safe, while the Nominal filter may select actions that appear safe under nominal dynamics but fail under adverse realizations.

## C. Robust Sample-and-Verify with Latent Disturbances

We test the sampling-and-verify policy steering using WM imagination (23) on fine-tuned $\pi _ { 0 . 5 }$ [86], evaluating whether the learned latent disturbance enables robust action selection.

Setup. We sample 8 candidate action chunks from the policy, each containing 16 future actions. We evaluate the outcome of each candidate using WM imaginations and execute the action chunk with the best predicted outcome. We compare three imagination schemes: (i) Nominal, using the learned latent dynamics; (ii) LUCID, using the learned latent disturbance; and (iii) Unconformalized (UnConf.), using a latent disturbance without the in-distribution constraint, as illustrated in Fig. 10.

Objective Function (J). To evaluate action outcomes, we use a cost function that combines the learned failure margin $\ell _ { z } ( z )$ evaluated on predicted latent states, with VLM-based tasksuccess and progress scores from ROBOMETER [87], evaluated on decoded images. The VLM reward is averaged over each action chunk to provide a dense signal.

Result: LUCID Enables Robust Policy Steering. Fig. 9 reports sample-and-verify style policy steering results over 20 trials for each method, showing that LUCID improves policy success. Fig. 10 illustrates that LUCID selects actions that remain favorable even under plausible worst-case futures, enabling robust policy steering. In contrast, Nominal does not improve over the base policy without steering, imagining optimistic futures for risky actions. Latent disturbance using an unconformalized uncertainty set instead trivially imagines implausible failures even for safe actions, which weakens discrimination among candidate actions for policy steering.

![](images/6ace888f02205b9f3d9dd7778732cb3219a1f486d78fad22e9ec579620beaa1b.jpg)  
Fig. 10. Qualitative: Policy Steering with World Model. (a) Initial observation and sampled candidate actions. (b) For each candidate, Nominal imagines outcomes using $f _ { z } ,$ LUCID computes worst-case $f _ { z } ^ { \pi _ { d } }$ within the calibrated uncertainty set ${ \mathcal F } ,$ while Unconformalized $f _ { z } ^ { \hat { \pi _ { d } } }$ can exploit out-ofdistribution latent states. (c) Imagined outcomes for candidate actions. LUCID selects an action that remains successful under the plausible worst-case, whereas the other schemes lose discriminability among candidate actions.

## X. LIMITATIONS

Balancing Plausibility. We optimize our latent disturbance to generate adverse outcomes while regularizing it to stay close to the learned dynamics and remain in distribution. However, these constraints provide only proxies for plausibility, so the resulting uncertainty set can still become overly conservative or fail to capture plausible but critical outcomes. Moreover, because the constraints are enforced through soft penalties, the optimized disturbance may still produce implausible imaginations, particularly when the underlying world model is inaccurate [11]. Conformal calibration provides a marginal coverage guarantee over the calibration distribution rather than an inputconditional guarantee [18]. As disturbance optimization steers WM imaginations away from this distribution, the coverage guarantee may further weaken under distribution shift. More structured world models [48], adaptive or input-dependent calibration schemes, and hallucination detection for worldmodel imaginations [16] could help ensure that pessimistic imaginations remain physically meaningful.

World Model Fidelity. Our current implementation trains task-specific latent WMs from offline interaction data, requiring substantial data collection, and latent disturbance optimization cannot recover outcomes that the WM has not learned to represent. This underscores the need for diverse robot interaction data, including failures, rare outcomes, and variations in physical conditions [88]. In addition, our current formulation assumes a probabilistic latent WM with an explicit transition distribution, which is used to characterize predictive uncertainty and define the latent disturbance. It is therefore not directly applicable to deterministic WMs or diffusionbased video WMs, where uncertainty is represented through a different generative process; complementary approaches can instead steer diffusion-based WM imaginations toward adverse yet plausible outcomes [15]. Our framework also assumes a meaningful failure margin or objective function; inaccurate specifications can misdirect disturbance optimization and policy steering. More general task and safety specifications, for example those derived from vision-language models, could reduce this dependence on task-specific objectives [87].

Game-Theoretic Optimization. Solving the resulting dynamic game remains challenging in high-dimensional action (e.g., 7-D) and latent disturbance (e.g., 1K-D) spaces. Although our disturbance parameterization and adversarial optimization make the problem more tractable, the learned disturbance explores only a restricted class of latent dynamics, and the optimization does not guarantee convergence to a global saddle point [77]. An insufficiently optimized disturbance may miss adverse outcomes, while an insufficiently optimized robot policy may produce suboptimal robust actions. Developing more reliable game-theoretic optimization methods and characterizing the effects of function approximation and optimization error remain important future work.

## XI. CONCLUSION

In this work, we propose a framework for robust decisionmaking in the learned latent space of world models. We model latent-space disturbances as adverse yet plausible perturbations to learned latent dynamics, enabling robust optimization directly in latent spaces learned from high-dimensional observations without requiring disturbances to be explicitly represented in a physically interpretable state space. To characterize which latent dynamics constitute plausible disturbances, we construct an uncertainty set using a dynamicsaware similarity metric that captures predictive uncertainty in the learned dynamics, together with out-of-distribution detection to exclude implausible latent states. We calibrate this uncertainty set using conformal prediction, allowing the robot to reason about worst-case outcomes while preventing the resulting WM imaginations from becoming overly pessimistic. We instantiate this framework for two forms of robust policy steering: latent safety filtering and sample-and-verify steering of a generative control policy. Controlled experiments show that the proposed formulation closely approximates groundtruth robust safety values and reduces failures under unobserved system disturbances, while ablations demonstrate the importance of calibration and in-distribution constraints for preserving the plausibility of imaginations induced by latent disturbances. In simulated and real-robot vision-based manipulation, our framework improves policy success rates through robust runtime steering.

## ACKNOWLEDGMENT

This work was supported in part by Toyota Research Institute and the DARPA Young Faculty Award (YFA). This article solely reflects the opinions and conclusions of its authors, and not TRI (nor any other Toyota entity), nor DARPA, nor the U.S. Government.

## REFERENCES

[1] A. Ben-Tal, A. Nemirovski, and L. El Ghaoui, Robust optimization. Princeton university press, 2009.

[2] J. F. Fisac, A. K. Akametalu, M. N. Zeilinger, S. Kaynama, J. Gillula, and C. J. Tomlin, “A general safety framework for learning-based control in uncertain robotic systems,” IEEE Transactions on Automatic Control, vol. 64, no. 7, pp. 2737–2752, 2018.

[3] D. P. Nguyen, K.-C. Hsu, W. Yu, J. Tan, and J. F. Fisac, “Gameplay filters: Robust zero-shot safety through adversarial imagination,” in Conference on Robot Learning (CoRL), 2024.

[4] I. M. Mitchell, A. M. Bayen, and C. J. Tomlin, “A time-dependent hamilton-jacobi formulation of reachable sets for continuous dynamic games,” IEEE Transactions on Automatic Control, 2005.

[5] L. Pinto, J. Davidson, R. Sukthankar, and A. Gupta, “Robust adversarial reinforcement learning,” in International Conference on Machine Learning (ICML), 2017, pp. 2817–2826.

[6] J. Moos, K. Hansel, H. Abdulsamad, S. Stark, D. Clever, and J. Peters, “Robust reinforcement learning: A review of foundations and recent advances,” Machine Learning and Knowledge Extraction, vol. 4, no. 1, pp. 276–315, 2022.

[7] S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-jacobi reachability: A brief overview and recent advances,” in IEEE Conference on Decision and Control (CDC), 2017, pp. 2242–2253.

[8] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, no. 8059, pp. 647–653, Apr. 2025.

[9] L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.

[10] K. Nakamura, L. Peters, and A. Bajcsy, “Generalizing safety beyond collision-avoidance via latent-space reachability analysis,” Robotics: Science and Systems (RSS), 2025.

[11] J. Seo, K. Nakamura, and A. Bajcsy, “Uncertainty-aware latent safety filters for avoiding out-of-distribution failures,” Conference on Robot Learning (CoRL), 2025.

[12] K.-C. Hsu, D. P. Nguyen, and J. F. Fisac, “Isaacs: Iterative soft adversarial actor-critic for safety,” in Learningfor Dynamics and Control Conference (L4DC), 2023, pp. 90–103.

[13] C. Johnstone and B. Cox, “Conformal uncertainty sets for robust optimization,” in Conformal and Probabilistic Prediction and Applications. PMLR, 2021, pp. 72–90.

[14] H. Hu, Z. Zhang, K. Nakamura, A. Bajcsy, and J. F. Fisac, “Deception game: Closing the safety-learning loop in interactive robot autonomy,” in Conference on Robot Learning (CoRL), 2023.

[15] J. Seo, S. Veer, R. Tian, W. Ding, A. Sharma, K. Leung, E. Schmerling, M. Pavone, and A. Bajcsy, “Stressdream: Steering video world models for robust policy evaluation and improvement,” in Conference on Robot Learning (CoRL), 2026.

[16] N. Hansen and X. Wang, “Hallucination in world models is predictable and preventable,” arXiv preprint arXiv:2606.27326, 2026.

[17] G. Shafer and V. Vovk, “A tutorial on conformal prediction.” Journal of Machine Learning Research, vol. 9, no. 3, 2008.

[18] A. N. Angelopoulos, S. Bates et al., “Conformal prediction: A gentle introduction,” Foundations and Trends® in Machine Learning, 2023.

[19] S. Agrawal, J. Seo, K. Nakamura, R. Tian, and A. Bajcsy, “Anysafe: Adapting latent safety filters at runtime via safety constraint parameterization in the latent space,” in IEEE International Conference on Robotics and Automation (ICRA), 2026.

[20] Y. Wu, R. Tian, G. Swamy, and A. Bajcsy, “From foresight to forethought: Vlm-in-the-loop policy steering via latent alignment,” Robotics: Science and Systems (RSS), 2025.

[21] J. Yuan, Y. Wu, and A. Bajcsy, “When to act, ask, or learn: Uncertaintyaware policy steering,” Robotics: Science and Systems (RSS), 2026.

[22] D. Q. Mayne, M. M. Seron, and S. V. Rakovic, “Robust model predic-´ tive control of constrained linear systems with bounded disturbances,” Automatica, vol. 41, no. 2, pp. 219–224, 2005.

[23] J. F. Fisac, N. F. Lugovoy, V. Rubies-Royo, S. Ghosh, and C. J. Tomlin, “Bridging hamilton-jacobi safety analysis and reinforcement learning,” in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 8550–8556.

[24] R. K. Cosner, P. Culbertson, A. J. Taylor, and A. D. Ames, “Robust safety under stochastic uncertainty with discrete-time control barrier functions,” 2023.

[25] A. K. Akametalu, J. F. Fisac, J. H. Gillula, S. Kaynama, M. N. Zeilinger, and C. J. Tomlin, “Reachability-based safe learning with gaussian processes,” in IEEE Conference on Decision and Control (CDC), 2014.

[26] I. Yang, “A dynamic game approach to distributionally robust safety specifications for stochastic systems,” Automatica, vol. 94, pp. 94–101, 2018.

[27] A. Z. Ren and A. Majumdar, “Distributionally robust policy learning via adversarial environment generation,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1379–1386, 2022.

[28] J. J. Choi, D. Lee, K. Sreenath, C. J. Tomlin, and S. L. Herbert, “Robust control barrier–value functions for safety-critical control,” in IEEE Conference on Decision and Control (CDC), 2021, pp. 6814– 6821.

[29] A. Sinha, H. Namkoong, and J. Duchi, “Certifiable distributional robustness with principled adversarial training,” in International Conference on Learning Representations (ICLR), 2018.

[30] Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone, “Riskconstrained reinforcement learning with percentile risk criteria,” Journal of Machine Learning Research, vol. 18, no. 167, pp. 1–51, 2018.

[31] Y. Ma, D. Jayaraman, and O. Bastani, “Conservative offline distributional reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 19 235–19 247, 2021.

[32] M. P. Chapman, R. Bonalli, K. M. Smith, I. Yang, M. Pavone, and C. J. Tomlin, “Risk-sensitive safety analysis using conditional value-at-risk,” IEEE Transactions on Automatic Control, vol. 67, no. 12, pp. 6521– 6536, 2021.

[33] H. Nishimura, J. Mercat, B. Wulfe, R. T. McAllister, and A. Gaidon, “Rap: Risk-aware prediction for robust planning,” in Conference on Robot Learning (CoRL), 2023, pp. 381–392.

[34] A. Majumdar and M. Pavone, “How should a robot assess risk? towards an axiomatic theory of risk in robotics,” in International Symposium of Robotics Research (ISRR), 2017.

[35] Y. Chow, A. Tamar, S. Mannor, and M. Pavone, “Risk-sensitive and robust decision-making: a cvar optimization approach,” Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.

[36] Y. Liang, Y. Sun, R. Zheng, and F. Huang, “Efficient adversarial training without attacking: Worst-case-aware robust reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 22 547–22 561, 2022.

[37] H. Zhang, H. Chen, C. Xiao, B. Li, M. Liu, D. Boning, and C.-J. Hsieh, “Robust deep reinforcement learning against adversarial perturbations on state observations,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 21 024–21 037, 2020.

[38] A. Mandlekar, Y. Zhu, A. Garg, L. Fei-Fei, and S. Savarese, “Adversarially robust policy learning: Active construction of physically-plausible perturbations,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017, pp. 3932–3939.

[39] S. Curi, I. Bogunovic, and A. Krause, “Combining pessimism with optimism for robust and efficient model-based deep reinforcement learning,” in International Conference on Machine Learning (ICML). PMLR, 2021, pp. 2254–2264.

[40] M. Rigter, B. Lacerda, and N. Hawes, “Rambo-rl: Robust adversarial model-based offline reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 16 082–16 097, 2022.

[41] J. Blanchet, M. Lu, T. Zhang, and H. Zhong, “Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 66 845– 66 859, 2023.

[42] J. Chen, L. Xu, A. Venugopal, and J. Schneider, “Policy-driven world model adaptation for robust offline model-based reinforcement learning,” in International Conference on Machine Learning (ICML), 2026.

[43] D. D. Oh, D. P. Nguyen, H. Hu, and J. F. Fisac, “Synthesis and deployment of maximal robust control barrier functions through adversarial reinforcement learning,” arXiv preprint arXiv:2604.13192, 2026.

[44] M. Rigter, B. Lacerda, and N. Hawes, “One risk to rule them all: A risksensitive perspective on model-based offline reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 77 520–77 545, 2023.

[45] J. Sun, Y. Jiang, J. Qiu, P. Nobel, M. J. Kochenderfer, and M. Schwager, “Conformal prediction for uncertainty-aware planning with diffusion dynamics model,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 80 324–80 337, 2023.

[46] L. Marques and D. Berenson, “Quantifying aleatoric and epistemic dynamics uncertainty via local conformal calibration,” in International Workshop on the Algorithmic Foundations of Robotics. Springer, 2024, pp. 85–103.

[47] W. Zhou and S. Zhu, “Calibrating decision robustness via inverse conformal risk control,” in International Conference on Machine Learning (ICML), 2026.

[48] Y. Wang, O. Bounou, G. Zhou, R. Balestriero, T. G. J. Rudner, Y. LeCun, and M. Ren, “Temporal straightening for latent planning,” in International Conference on Machine Learning (ICML), 2026.

[49] P. Lutkus, K. Wang, L. Lindemann, and S. Tu, “Latent representations for control design with provable stability and safety guarantees,” in IEEE Conference on Decision and Control (CDC). IEEE, 2025, pp. 2937– 2944.

[50] N. Hansen, H. Su, and X. Wang, “Td-mpc2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations (ICLR), 2024.

[51] Y. Guo, L. Shi, J. Chen, and C. Finn, “Ctrl-world: A controllable generative world model for robot manipulation,” in International Conference on Learning Representations (ICLR), 2026.

[52] J. H. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang, “Worldgym: World model as an environment for policy evaluation,” in International Conference on Learning Representations (ICLR), 2026.

[53] Y. Guo, T. Lee, L. X. Shi, J. Chen, P. Liang, and C. Finn, “VLAW: Iterative co-improvement of vision-language-action policy and world model,” in International Conference on Machine Learning (ICML), 2026.

[54] A. K. Sharma, Y. Sun, N. Lu, Y. Zhang, J. Liu, and S. Yang, “Worldgymnast: Training robots with reinforcement learning in a world model,” arXiv preprint arXiv:2602.02454, 2026.

[55] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical ai,” arXiv preprint arXiv:2501.03575, 2025.

[56] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin et al., “Dreamdojo: A generalist robot world model from large-scale human videos,” arXiv preprint arXiv:2602.06949, 2026.

[57] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang et al., “World action models are zero-shot policies,” arXiv preprint arXiv:2602.15922, 2026.

[58] Z. Mei, T. Yin, O. Shorinwa, A. Badithela, Z. Zheng, J. Bruno, M. Bland, L. Zha, A. Hancock, J. F. Fisac et al., “Video generation models in robotics-applications, research challenges, future directions,” arXiv preprint arXiv:2601.07823, 2026.

[59] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “Dino-wm: World models on pre-trained visual features enable zero-shot planning,” International Conference on machine learning (ICML), 2025.

[60] A. K. Jain, Y. Wu, J. Farebrother, G. Swamy, and A. Bajcsy, “Weaver, better, faster, longer: An effective world model for robotic manipulation,” arXiv preprint arXiv:2606.13672, 2026.

[61] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International Conference on machine learning (ICML), 2019.

[62] T. Kerssies, G. Berton, J. He, Q. Yu, W. Ma, D. de Geus, G. Dubbelman, and L.-C. Chen, “A frame is worth one token: Efficient generative world modeling with delta tokens,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 27 978–27 988.

[63] D. Nath, A. Srinivasan, H. Yin, R. Jiang, J. Fang, and G. Chou, “Pixels to proofs: Probabilistically-safe latent world model control via parallel conformal robust mpc,” arXiv preprint arXiv:2606.15594, 2026.

[64] W. Zhang, G. Wang, J. Sun, Y. Yuan, and G. Huang, “Storm: Efficient stochastic transformer based world models for reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 27 147–27 166, 2023.

[65] D. Hafner, W. Yan, and T. Lillicrap, “Training agents inside of scalable world models,” arXiv preprint arXiv:2509.24527, 2025.

[66] S. Huang, P. Kaushik, M. Chen, H. Pan, K. Geng, O. Chehab, F. Moreno-Pino, and M. Simchowitz, “Nano world models: A minimalist implementation of future video prediction,” arXiv preprint arXiv:2605.23993, 2026.

[67] K.-C. Hsu, H. Hu, and J. F. Fisac, “The safety filter: A unified view of safety-critical control in autonomous systems,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 7, 2023.

[68] S. Li and O. Bastani, “Robust model predictive shielding for safe reinforcement learning with stochastic dynamics,” in IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 7166–7172.

[69] O. Bastani, “Safe reinforcement learning with nonlinear dynamics via model predictive shielding,” in American Control Conference (ACC), 2021, pp. 3488–3494.

[70] I. M. Mitchell et al., “A toolbox of level set methods,” UBC Department of Computer Science Technical Report TR-2007-11, 2007.

[71] S. Bansal and C. J. Tomlin, “Deepreach: A deep learning approach to high-dimensional reachability,” in IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 1817–1824.

[72] K. P. Wabersich, A. J. Taylor, J. J. Choi, K. Sreenath, C. J. Tomlin, A. D. Ames, and M. N. Zeilinger, “Data-driven safety filters: Hamiltonjacobi reachability, control barrier functions, and predictive methods for uncertain systems,” IEEE Control Systems Magazine, vol. 43, no. 5, pp. 137–177, 2023.

[73] K. Nakamura, A. L. Bishop, S. Man, A. M. Johnson, Z. Manchester, and A. Bajcsy, “How to train your latent control barrier function: Smooth safety filtering under hard-to-model constraints,” in Learning for Dynamics and Control Conference (L4DC), 2026.

[74] C. Xu, T. K. Nguyen, E. Dixon, C. Rodriguez, P. Miller, R. Lee, P. Shah, R. Ambrus, H. Nishimura, and M. Itkina, “Can we detect failures without failure data? uncertainty-aware runtime failure detection for imitation learning policies,” Robotics: Science and Systems (RSS), 2025.

[75] T. Ding, A. Angelopoulos, S. Bates, M. Jordan, and R. J. Tibshirani, “Class-conditional conformal prediction with many classes,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 64 555–64 576, 2023.

[76] A. Z. Ren, A. Dixit, A. Bodrova, S. Singh, S. Tu, N. Brown, P. Xu, L. Takayama, F. Xia, J. Varley, Z. Xu, D. Sadigh, A. Zeng, and A. Majumdar, “Robots that ask for help: Uncertainty alignment for large language model planners,” in Conference on Robot Learning (CoRL), 2023.

[77] J. Wang, H. Hu, D. P. Nguyen, and J. F. Fisac, “Magics: Adversarial rl with minimax actors guided by implicit critic stackelberg for convergent neural synthesis of robot safety,” arXiv preprint arXiv:2409.13867, 2024.

[78] M. Kim, K. Nakamura, and A. Bajcsy, “How well do latent world models understand partially observable safety constraints?” Conference on Robot Learning (CoRL), 2026.

[79] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems (RSS), 2023.

[80] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in International Conference on Machine Learning (ICML), 2018, pp. 1861–1870.

[81] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations (ICLR), 2020.

[82] M. G. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” in International Conference on Machine Learning (ICML), 2017.

[83] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos, “Distributional reinforcement learning with quantile regression,” in Proceedings of the AAAI Conference on Artificial Intelligence., 2018.

[84] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbr ˜ ugg, N. Rudin ¨ et al., “Isaac lab: A gpuaccelerated simulation framework for multi-modal robot learning,” arXiv preprint arXiv:2511.04831, 2025.

[85] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[86] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai et al., “π : A visionlanguage-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[87] A. Liang, Y. Korkmaz, J. Zhang, M. Hwang, A. Anwar, S. Kaushik, A. Shah, A. S. Huang, L. Zettlemoyer, D. Fox, Y. Xiang, A. Li, A. Bobu, A. Gupta, S. Tu, E. Biyik, and J. Zhang, “Robometer: Scaling generalpurpose robotic reward models via trajectory comparisons,” in Robotics: Science and Systems (RSS), 2026.

[88] A. Balaji, A. Bahety, S. Ambatipudi, D. Lam, J. Xu, and R. Mart´ın-Mart´ın, “Oopsieverse: A safety benchmark with damage-aware simulation for robot manipulation,” in Robotics: Science and Systems (RSS), 2026.

## APPENDIX A

## THEORETICAL ANALYSIS: ROBUST SAFETY FILTER

We provide a theoretical analysis of LUCID for safety filtering with learned latent dynamics, focusing on uncertainty in the latent dynamics and assuming no additional uncertainty from the learned representation. Assuming that the conformalized uncertainty set contains the latent dynamics realized at test time, we show that (i) the robust safety value does not overestimate the safety value under any plausible dynamics, whereas the nominal safety value can, and (ii) the perfect robust safety policy achieves zero worst-case suboptimality.

Assumption 1 (Markov Latent State). The latent state $z _ { t }$ is sufficient for predicting the system’s evolution. Consequently, the induced latent process is Markov and admits a conditional transition distribution of the form $f _ { z } ( z _ { t + 1 } \mid z _ { t } , a _ { t } )$

Zero Safety Overestimation. Safety values computed from nominal imaginations can be overly optimistic when the realized dynamics are adverse. In contrast, the robust safety value evaluates each action under the most adverse plausible transition in the uncertainty set, thereby avoiding overestimation. For a latent dynamics model $f _ { z }$ , given a policy π, define the discounted Bellman operator of a value function V based on its imaginations:

$$
\begin{array} { r l } & { ( \mathcal { T } _ { f _ { z } } ^ { \pi } V ) ( z ) : = ( 1 - \gamma ) \ell _ { z } ( z ) } \\ & { \qquad + \gamma \operatorname* { m i n } \bigl \{ \ell _ { z } ( z ) , \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ( \cdot \vert z , \pi ( z ) ) } [ V ( z ^ { \prime } ) ] \bigr \} , } \end{array}\tag{28}
$$

where its robust counterpart using $f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } )$ is

$$
\left. \begin{array} { l } { { ( T _ { \mathrm { r o b } } ^ { \pi } V ) ( z ) : = ( 1 - \gamma ) \ell _ { z } ( z ) \nonumber } } \\ { { + \left. \gamma \operatorname* { m i n } \Bigl \{ \ell _ { z } ( z ) , \underset { f _ { z } ^ { d } \in \mathcal { F } ( z , \pi ( z ) ) } { \operatorname* { m i n } } \mathbb { E } _ { z ^ { \prime } \sim f _ { z } ^ { d } ( z , \pi ( z ) ) } [ V ( z ^ { \prime } ) ] \Bigr \} . \right. } } \end{array} \right.\tag{29}
$$

Let $V _ { f _ { z } } ^ { \pi }$ and $V _ { \mathrm { r o b } } ^ { \pi }$ denote their unique fixed points, and denote the rectangular global uncertainty set is

$$
\mathcal { F } ( f _ { z } ) : = \left\{ f _ { z } ^ { d } : f _ { z } ^ { d } ( \cdot  { | } z , a ) \in \mathcal { F } ( z , a ) , \forall ( z , a ) \right\} .\tag{30}
$$

Theorem 1 (Robust Value Does Not Overestimate). For any policy π and any plausible latent dynamics $f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } )$

$$
V _ { \mathrm { r o b } } ^ { \pi } ( z ) \leq V _ { f _ { z } ^ { d } } ^ { \pi } ( z ) , \qquad \forall z \in \mathcal { Z } .\tag{31}
$$

Proof of Theorem 1. Fix π and $f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } )$ . By (30), $f _ { z } ^ { d } ( \cdot \mid z , \pi ( z ) )$ belongs to the local ambiguity set at every z. The minimum in the robust backup therefore cannot exceed the expectation under this particular transition. Hence, for every bounded V,

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { r o b } } ^ { \pi } V \leq \mathcal { T } _ { f _ { z } ^ { d } } ^ { \pi } V \qquad \mathrm { p o i n t w i s e } . } \end{array}
$$

Applying this inequality to the fixed point gives

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { r o b } } ^ { \pi } V _ { f _ { z } ^ { d } } ^ { \pi } \leq \mathcal { T } _ { f _ { z } ^ { d } } ^ { \pi } V _ { f _ { z } ^ { d } } ^ { \pi } = V _ { f _ { z } ^ { d } } ^ { \pi } . } \end{array}
$$

Monotonicity gives $( T _ { \mathrm { r o b } } ^ { \pi } ) ^ { m } V _ { f _ { \times } ^ { d } } ^ { \pi } \ \leq \ V _ { f _ { \times } ^ { d } } ^ { \pi }$ for every m. By contraction, $( T _ { \mathrm { r o b } } ^ { \pi } ) ^ { m } V _ { f _ { z } ^ { d } } ^ { \pi }$ converges to $\dot { V _ { \mathrm { r o b } } ^ { \tilde { \pi } } }$ , proving (31).

Theorem 1 states that, for any plausible dynamics in the uncertainty set, the safety value computed by robust optimization remains a lower bound on the realized safety value, enabling robust safety filtering. In contrast, the nominal value may overestimate safety when the realized dynamics lead to more pessimistic next states than the nominal model.

Remark 1. The above guarantee applies under the perfect coverage guarantee where the true latent dynamics are contained in the uncertainty set, while conformal calibration only provides a marginal probabilistic guarantee.

Corollary 1 (Nominal Safety Value Can Overestimate). For a policy π, there exists an admitted latent dynamics $f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } )$ such that the nominal safety value computed with $f _ { z }$ overestimates the realized safety value:

$$
V _ { f _ { z } } ^ { \pi } ( z ) \geq V _ { f _ { z } ^ { d } } ^ { \pi } ( z ) .
$$

Proof of Corollary 1. Let $f _ { z } ^ { \pi _ { d } }$ attain the worst-case admitted value:

$$
f _ { z } ^ { \pi _ { d } } \in \underset { f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } ) } { \operatorname { a r g m i n } } V _ { f _ { z } ^ { d } } ^ { \pi } ( z ) .
$$

Then, $V _ {  { f ^ { \pi } } _ { \gamma } d } ^ { \pi } ( z ) = V _ { \mathrm { r o b } } ^ { \pi } ( z )$ . Since the nominal dynamics $f _ { z }$ is also admitted, Theorem 1 gives that the safety value in $f _ { z } ^ { \pi _ { d } }$ can be lower than the nominal value computed with $f _ { z }$

$$
V _ { f _ { z } } ^ { \pi } ( z ) \geq V _ { \mathrm { r o b } } ^ { \pi } ( z ) = V _ { f _ { z } ^ { \pi _ { d } } } ^ { \pi } ( z ) .
$$

Thus, an adverse but plausible dynamics can make the nominal safety value overestimate the realized value. □

Sub-optimality of Safety Policy. Robust optimization aims to select actions that remain effective across plausible outcomes, while a nominal policy optimized only under nominal imaginations may select actions that become suboptimal when adverse dynamics are realized. To formalize this, for a policy π, define its state-wise worst-case safety value:

$$
\mathcal { T } _ { \mathrm { w c } } ^ { \pi } ( z ) : = \operatorname* { m i n } _ { f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } ) } V _ { f _ { z } ^ { d } } ^ { \pi } ( z ) ,\tag{32}
$$

representing the safety value achieved by $\pi$ under the worstcase dynamics. We then define the worst-case suboptimality of $\pi$ as the gap between its worst-case safety value and the maximum safety value achievable for the worst-case:

$$
\mathrm { S u b O p t } _ { \mathrm { w c } } ( \pi , z ) : = \operatorname* { m a x } _ { \pi ^ { \prime } } \mathcal { J } _ { \mathrm { w c } } ^ { \pi ^ { \prime } } ( z ) , - \mathcal { J } _ { \mathrm { w c } } ^ { \pi } ( z ) .\tag{33}
$$

Thus, zero suboptimality means that a policy achieves the best possible safety value despite the most adverse dynamics. In this analysis, we write $\pi ^ { \mathrm { { r o b } } }$ for the robust safety policy $\pi _ { \mathrm { r o b } }$ Let $\pi ^ { \mathrm { { r o b } } }$ solve the robust Bellman equation (18), and let π<sup>nom</sup> solve the nominal Bellman equation (15).

Theorem 2 (Worst-Case Suboptimality of the Robust Policy). For the global uncertainty set in (30) and every $z \in { \mathcal { Z } } _ { : }$

$$
\mathrm { S u b O p t } _ { \mathrm { w c } } ( \pi ^ { \mathrm { r o b } } , z ) = 0 .\tag{34}
$$

Proof of Theorem 2. Under the rectangular ambiguity set (30), for any $\pi ,$

$$
\mathcal { T } _ { \mathrm { w c } } ^ { \pi } ( z ) = \operatorname* { m i n } _ { f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } ) } V _ { f _ { z } ^ { d } } ^ { \pi } ( z ) = V _ { \mathrm { r o b } } ^ { \pi } ( z ) .\tag{35}
$$

Since $\pi ^ { \mathrm { { r o b } } }$ is optimal for the robust Bellman equation,

$$
\mathcal { I } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) = V _ { \mathrm { r o b } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) = \operatorname* { m a x } _ { \pi } V _ { \mathrm { r o b } } ^ { \pi } ( z ) = \operatorname* { m a x } _ { \pi } \mathcal { I } _ { \mathrm { w c } } ^ { \pi } ( z ) ,
$$

which proves SubOp ${ \bf \Pi } _ { \mathrm { w c } } ( \pi ^ { \mathrm { r o b } } , z ) = 0 .$

Corollary 2 (Nominal Safety Policy Can Be Suboptimal). Following Theorem 2, for every $z \in { \mathcal { Z } }$

$$
0 \leq \mathrm { S u b O p t } _ { \mathrm { w c } } ( \pi ^ { \mathrm { n o m } } , z ) \leq 2 \operatorname* { m i n } \left( 1 , \frac { \sqrt { 2 } \gamma } { 1 - \gamma } \sqrt { \epsilon _ { \mathrm { K L } } } \right)\tag{36}
$$

Also, the worst-case suboptimality is positive whenever

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { n o m } } } ( z ) < \mathcal { T } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) . } \end{array}
$$

Proof of Corollary 2. By Theorem 2, $\pi ^ { \mathrm { n o m } }$ is suboptimal:

$$
\mathrm { S u b O p t } _ { \mathrm { w c } } ( \pi ^ { \mathrm { n o m } } , z ) = \mathcal { I } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) - \mathcal { I } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { n o m } } } ( z ) \geq 0 .
$$

Consider any two latent dynamics models $f , g ,$ , and the fixedpoint property and triangle inequality give

$$
\| V _ { f } ^ { \pi } - V _ { g } ^ { \pi } \| _ { \infty } \leq \| T _ { f } ^ { \pi } V _ { f } ^ { \pi } - \mathcal T _ { f } ^ { \pi } V _ { g } ^ { \pi } \| _ { \infty } + \| T _ { f } ^ { \pi } V _ { g } ^ { \pi } - \mathcal T _ { g } ^ { \pi } V _ { g } ^ { \pi } \| _ { \infty } .
$$

The first term is at most $\gamma \| V _ { f } ^ { \pi } - V _ { g } ^ { \pi } \| _ { \infty }$ . Since the minimum is non-expansive, the second term is at most

$$
\gamma \operatorname* { m a x } _ { z } \left| \mathbb { E } _ { f } [ V _ { g } ^ { \pi } ( z ^ { \prime } ) ] - \mathbb { E } _ { g } [ V _ { g } ^ { \pi } ( z ^ { \prime } ) ] \right| ,
$$

where both expectations condition on $( z , \pi ( z ) )$ . Thus,

$$
\| V _ { f } ^ { \pi } - V _ { g } ^ { \pi } \| _ { \infty } \leq \frac { \gamma } { 1 - \gamma } \operatorname* { m a x } _ { z } \left| \mathbb { E } _ { f } [ V _ { g } ^ { \pi } ( z ^ { \prime } ) ] - \mathbb { E } _ { g } [ V _ { g } ^ { \pi } ( z ^ { \prime } ) ] \right| .\tag{37}
$$

For every $f _ { z } ^ { d } \in \mathcal { F } ( f _ { z } )$ , the KL condition in (7) and Pinsker’s inequality imply, for any $V : \mathcal { Z } \to [ - 1 , 1 ]$

$$
\begin{array} { r } { \big | \mathbb { E } _ { f _ { z } ^ { d } } [ V ] - \mathbb { E } _ { f _ { z } } [ V ] \big | \le \sqrt { 2 \epsilon _ { \mathrm { K L } } } . } \end{array}
$$

Substituting this into (37) gives

$$
\| V _ { f _ { z } ^ { d } } ^ { \pi } - V _ { f _ { z } } ^ { \pi } \| _ { \infty } \leq \kappa = \frac { \sqrt { 2 } \gamma } { 1 - \gamma } \sqrt { \epsilon _ { \mathrm { K L } } } .\tag{38}
$$

Therefore, for every π,

$$
V _ { f _ { z } } ^ { \pi } ( z ) - \kappa \le \mathcal { I } _ { \mathrm { w c } } ^ { \pi } ( z ) \le V _ { f _ { z } } ^ { \pi } ( z ) + \kappa .
$$

Nominal optimality gives $V _ { f _ { z } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) \leq V _ { f _ { z } } ^ { \pi ^ { \mathrm { n o m } } } ( z )$ . Hence,

$$
\begin{array} { r l } & { \mathrm { S u b O p t } _ { \mathrm { w c } } ( \pi ^ { \mathrm { n o m } } , z ) = \mathcal { I } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) - \mathcal { I } _ { \mathrm { w c } } ^ { \pi ^ { \mathrm { n o m } } } ( z ) } \\ & { \qquad \leq V _ { f _ { z } } ^ { \pi ^ { \mathrm { r o b } } } ( z ) + \kappa - V _ { f _ { z } } ^ { \pi ^ { \mathrm { n o m } } } ( z ) + \kappa } \\ & { \qquad \leq 2 \kappa . } \end{array}
$$

Theorem 2 states that the exact robust policy achieves zero worst-case suboptimality, while Corollary 2 shows that the nominal policy has no such zero-suboptimality guarantee and can be suboptimal if the worst case is realized.

Remark 2. The learned disturbance and policy only approximate this game, so the exact guarantees hold to the extent that ambiguity-set feasibility and optimization are achieved.

## APPENDIX B

## THEORETICAL ANALYSIS: CONFORMAL PREDICTION

In this section, we briefly review conformal prediction for the calibration procedure in Sec. V-B. Conformal prediction constructs prediction sets with finite-sample, distribution-free coverage under exchangeability [17], [18], which can be used as data-calibrated uncertainty sets for robust optimization [13].

Conformal Prediction. Conformal prediction (CP) provides distribution-free coverage guarantees by calibrating a nonconformity threshold on held-out data [17], [18]. In Sec. V-B, let $s _ { t } ^ { i }$ denote the nonconformity score at time t for calibration sample i, and write $s ^ { i }$ when t is fixed. Given a user-specified miscoverage level $\alpha , \mathrm { C P }$ sets the threshold ϵ to the appropriate (1−α) quantile of the calibration scores. Under exchangeability between the calibration and test samples, this yields

$$
\begin{array} { r } { \epsilon : = s ^ { ( k ) } , \quad k = \lceil ( n + 1 ) ( 1 - \alpha ) \rceil , \mathbb { P } \big ( s ^ { \mathrm { t e s t } } \leq \epsilon \big ) \geq 1 - \alpha , } \end{array}\tag{39}
$$

where $s ^ { ( k ) }$ is the kth smallest of the n calibration scores. The probability is marginal over both the calibration data and the test sample. We apply this procedure to calibrate the KL-ball radius with $( \epsilon , \alpha ) = ( \epsilon _ { \mathrm { K L } } , \alpha _ { \mathrm { K L } } )$ and the OOD threshold with $( \epsilon , \alpha ) = ( \epsilon _ { \mathrm { O O D } } , \alpha _ { \mathrm { o o d } } )$

Coverage of the Dynamics-Aware Similarity. Our goal is to calibrate the KL-ball radius $\epsilon _ { \mathrm { K L } }$ so that the uncertainty set contains the test-time latent distribution encoded from observations with high probability. In Sec. V-B, we measure the discrepancy between this encoded distribution and the learned latent dynamics using the KL nonconformity score:

$$
s _ { \mathrm { K L } } ^ { i } : = D _ { \mathrm { K L } } \big ( \mathcal { E } ( o _ { \leq t } ^ { i } ) \big | \big | f _ { z } ( z _ { t - 1 } ^ { i } , a _ { t - 1 } ^ { i } ) \big ) .\tag{40}
$$

Here, t is fixed as above. We apply (39) with miscoverage level α to obtain $\epsilon _ { \mathrm { K L } } = s _ { \mathrm { K L } } ^ { ( k ) }$ , where $k = \lceil ( n + 1 ) ( 1 - \alpha _ { \mathrm { K L } } ) \rceil$ The following theorem gives the coverage guarantee in (8).

Theorem 3 (Marginal KL Coverage). Suppose that the calibration and test KL scores in (40) are exchangeable and $k \leq n$ . Then, the calibrated KL ball contains the test-time encoded latent distribution with probability at least $1 - \alpha _ { \mathrm { K L } } .$

$$
\mathbb { P } ( s _ { \mathrm { K L } } ^ { \mathrm { t e s t } } \le \epsilon _ { \mathrm { K L } } ) \ge 1 - \alpha _ { \mathrm { K L } } .\tag{41}
$$

Proof. The encoded test distribution belongs to the calibrated KL ball whenever its score satisfies $s _ { \mathrm { K L } } ^ { \mathrm { t e s t } } \leq \epsilon _ { \mathrm { K L } }$ . To bound this probability, consider the rank J of the test score among the n calibration scores and the test score, with ties broken uniformly at random. Under exchangeability, the test score is equally likely to occupy any of the $n + 1$ ranks. If $J \leq k ,$ , at most $k { - } 1$ calibration scores precede the test score, so it cannot exceed the kth smallest calibration score $\epsilon _ { \mathrm { K L } } .$ Therefore,

$$
\mathbb { P } ( s _ { \mathrm { K L } } ^ { \mathrm { t e s t } } \le \epsilon _ { \mathrm { K L } } ) \ge \mathbb { P } ( J \le k ) = \frac { k } { n + 1 } \ge 1 - \alpha _ { \mathrm { K L } } .\tag{□}
$$

Dataset-Conditional KL Coverage. In practice, we calibrate ϵ<sub>KL</sub> once using a held-out dataset and keep it fixed during deployment. We therefore consider the probability that the resulting KL ball contains the encoded distribution of a new test sample, conditional on the calibration dataset:

$$
p _ { \mathrm { K L } } ( \mathcal { D } _ { \mathrm { c a l i b } } ) : = \mathbb { P } ( s _ { \mathrm { K L } } ^ { \mathrm { t e s t } } \leq \epsilon _ { \mathrm { K L } } \mid \mathcal { D } _ { \mathrm { c a l i b } } ) .\tag{42}
$$

Because different calibration datasets produce different radii, this coverage varies with $\mathcal { D } _ { \mathrm { c a l i b } }$ . The following theorem provides a lower bound on the coverage attained by the calibrated radius with high probability over the calibration data [18].

Theorem 4 (Dataset-Conditional KL Coverage). Suppose that the calibration and test samples are i.i.d., and use $\epsilon _ { \mathrm { K L } } = s _ { \mathrm { K L } } ^ { ( k ) }$ with $k = \lceil ( n + 1 ) ( 1 - \alpha _ { \mathrm { K L } } ) \rceil$ . The conditional coverage varies over calibration datasets according to

$$
p _ { \mathrm { K L } } ( \mathcal { D } _ { c a l i b } ) \sim \mathrm { B e t a } ( k , n + 1 - k ) .\tag{43}
$$

For a user-specified confidence level $1 - \eta ,$ let $b _ { \eta }$ be the $\eta -$ quantile of this Beta distribution. Then, with probability at

least $1 - \eta$ over the calibration data, the calibrated KL ball has test coverage at least $b _ { \eta } \colon$

$$
\begin{array} { r } { \mathbb { P } _ { \mathcal { D } _ { c a l i b } } ( p _ { \mathrm { K L } } ( \mathcal { D } _ { c a l i b } ) \geq b _ { \eta } ) \geq 1 - \eta . } \end{array}\tag{44}
$$

Proof. Let $F$ denote the KL score cumulative distribution function. Since the test sample is independent of the calibration data,

$$
p _ { \mathrm { K L } } ( \mathcal { D } _ { \mathrm { c a l i b } } ) = F ( \epsilon _ { \mathrm { K L } } ) = F ( s _ { \mathrm { K L } } ^ { ( k ) } ) .
$$

For continuous $F ,$ the transformed calibration scores $U _ { i } : = F ( s _ { \mathrm { K L } } ^ { i } )$ are i.i.d. uniform on [0, 1], and the coverage equals their kth order statistic $U _ { ( k ) }$ . The event $U _ { ( k ) } \leq$ u occurs when at least k of the n scores lie below u. Thus,

$$
\mathbb { P } ( U _ { ( k ) } \le u ) = \sum _ { j = k } ^ { n } \binom { n } { j } u ^ { j } ( 1 - u ) ^ { n - j } ,
$$

which is the cdf of Beta $\mathfrak { i } ( k , n { + } 1 { - } k )$ . Since $b _ { \eta }$ is its η-quantile, $\begin{array} { r } { \mathbb { P } _ { \mathcal { D } _ { \mathrm { c a l i b } } } ( p _ { \mathrm { K L } } ( \mathcal { D } _ { \mathrm { c a l i b } } ) \geq b _ { \eta } ) = 1 - \eta , } \end{array}$ , establishing the bound. With ties, the generalized-inverse representation $s _ { \mathrm { K L } } ^ { i } ~ = ~ F ^ { - 1 } ( U _ { i } )$ gives $F ( s _ { \mathrm { K L } } ^ { ( k ) } ) \ge U _ { ( k ) }$ , so the lower bound remains valid.

Class-Conditional In-Distribution Detection. Our goal is to calibrate the OOD threshold $\boldsymbol { \epsilon } _ { \mathrm { O O D } }$ so that in-distribution latent states are accepted with high probability. Since OOD states are not directly available, we follow Sec. V-B and use classconditional conformal prediction to calibrate the threshold using only held-out ID states [75], [11]. Given $n _ { \mathrm { I D } }$ calibration states with scores $s _ { \mathrm { O O D } } ^ { i } : = s _ { \mathrm { O O D } } ( z _ { t } ^ { i } )$ , we apply (39) with miscoverage level $\alpha _ { \mathrm { o o d } } \mathrm { : }$

$$
k _ { \mathrm { I D } } : = \lceil \big ( n _ { \mathrm { I D } } + 1 \big ) ( 1 - \alpha _ { \mathrm { o o d } } ) \rceil , \qquad \epsilon _ { \mathrm { O O D } } : = s _ { \mathrm { O O D } } ^ { ( k _ { \mathrm { I D } } ) } ,
$$

where we assume k<sub>ID</sub> $\leq n _ { \mathrm { I D } }$ . We classify a latent state as OOD when its score exceeds ϵ<sub>OOD</sub>:

$$
\widehat { Y } ( z ) : = \left\{ \begin{array} { l l } { \mathrm { I D } , } & { s _ { \mathrm { O O D } } ( z ) \leq \epsilon _ { \mathrm { O O D } } , } \\ { \mathrm { O O D } , } & { s _ { \mathrm { O O D } } ( z ) > \epsilon _ { \mathrm { O O D } } . } \end{array} \right.\tag{45}
$$

To quantify how reliably the detector retains ID states, we define its ID recall as the probability that an actual ID state is classified as ID:

$$
\begin{array} { r l } & { \mathrm { R e c a l l } _ { \mathrm { I D } } : = \mathbb { P } ( \widehat { Y } ( z _ { \mathrm { t e s t } } ) = \mathrm { I D } \mid Y _ { \mathrm { t e s t } } = \mathrm { I D } ) } \\ & { \qquad = 1 - \mathbb { P } ( \widehat { Y } ( z _ { \mathrm { t e s t } } ) = \mathrm { O O D } \mid Y _ { \mathrm { t e s t } } = \mathrm { I D } ) . } \end{array}\tag{46}
$$

Here, $Y _ { \mathrm { t e s t } }$ denotes the true class, and the probability is taken over the ID calibration data and an ID test state. Thus, guaranteeing recall of at least $1 - \alpha _ { \mathrm { o o d } }$ limits the probability of incorrectly rejecting an ID state to $\alpha _ { \mathrm { o o d } }$

Theorem 5 (In-Distribution Recall). Suppose that the OOD score function is fixed before calibration and, conditional on $Y _ { \mathrm { t e s t } } = \mathrm { I D }$ , the $n _ { \mathrm { I D } }$ ID calibration states and test state are exchangeable. The calibrated detector in (45) satisfies

$$
\mathrm { R e c a l l } _ { \mathrm { I D } } = \mathbb { P } ( s _ { \mathrm { O O D } } ( z _ { \mathrm { t e s t } } ) \leq \epsilon _ { \mathrm { O O D } } \ | \ Y _ { \mathrm { t e s t } } = \mathrm { I D } ) \geq 1 - \alpha _ { \mathrm { o o d } } .
$$

This establishes the ID recall guarantee in (9).

Proof. Conditional on the test state being ID, the calibration and test scores are exchangeable. Let $J _ { \mathrm { { I D } } }$ be the rank of $s _ { \mathrm { O O D } } ^ { \mathrm { t e s t } } : = s _ { \mathrm { O O D } } ( z _ { \mathrm { t e s t } } )$ among these $n _ { \mathrm { I D } } + 1$ scores. Exchangeability makes this rank uniform. If $J _ { \mathrm { I D } } \leq k _ { \mathrm { I D } }$ , the test score cannot exceed the $k _ { \mathrm { I D } }$ -th calibration score ϵ<sub>OOD</sub>, so the detector accepts the test state as ID:

$$
\mathrm { R e c a l l } _ { \mathrm { I D } } \ge \mathbb { P } ( J _ { \mathrm { I D } } \le k _ { \mathrm { I D } } \ | \ Y _ { \mathrm { t e s t } } = \mathrm { I D } ) = \frac { k _ { \mathrm { I D } } } { n _ { \mathrm { I D } } + 1 } \ge 1 - \alpha _ { \mathrm { o o d } } .\tag{48}
$$

The final inequality follows from the calibration rank.

Thus, calibration controls how often ID states are incorrectly excluded from the uncertainty set, averaged over calibration and test data. The ability to reject OOD states depends on the learned score and is not guaranteed by ID calibration alone.

Causal Reconstruction from Trajectory Calibration. Individual transitions within a trajectory are temporally dependent, so the state-level exchangeability assumption need not hold. In Sec. V-B, we therefore calibrate the maximum nonconformity score over each trajectory. Given a calibration trajectory $\tau _ { i } \stackrel { \cdot } { = } \{ ( o _ { t } ^ { i } , a _ { t } ^ { i } , o _ { t + 1 } ^ { i } ) \} _ { t = 1 } ^ { T - 1 }$ , we compute

$$
S ( \tau _ { i } ) : = \operatorname* { m a x } _ { 1 \leq t < T } s _ { t } ^ { i } , \quad \epsilon ^ { \mathrm { t r a j } } : = \mathrm { Q u a n t i l e } _ { 1 - \alpha } \Bigl ( \{ S ( \tau _ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { c a l i b } } } \Bigr ) .\tag{49}
$$

We then use the calibrated threshold at each transition. Since the maximum score is below the threshold exactly when every state-level score is below it, trajectory-level calibration provides simultaneous coverage across time. The following theorem formalizes this causal reconstruction, following [76].

Theorem 6 (Causal Reconstruction of Trajectory Coverage). Suppose that the calibration trajectories and test trajectory are exchangeable, the score functions are fixed, and each $s _ { t }$ depends only on observations and actions through the scored transition. Let $\epsilon ^ { \mathrm { t r a j } }$ be calibrated by (49), with $k \ = \ \lceil ( N _ { \mathrm { c a l i b } } + 1 ) ( 1 - \alpha ) \rceil \ \leq \ N _ { \mathrm { c a l i b } } .$ . Applying the same threshold at every transition gives

$$
\begin{array} { r } { \mathbb { P } \big ( s _ { t } ^ { \mathrm { t e s t } } \leq \epsilon ^ { \mathrm { t r a j } } , \forall 1 \leq t < T \big ) \geq 1 - \alpha . } \end{array}\tag{50}
$$

Proof. Let $\mathcal { C } ^ { \mathrm { t r a j } }$ denote the trajectories accepted by the $\epsilon ^ { \mathrm { t r a j } }$ and $\mathcal { C } ^ { \mathrm { c a u s a l } }$ those accepted at every transition:

$$
\begin{array} { r l } & { \mathcal { C } ^ { \mathrm { t r a j } } : = \{ \tau : S ( \tau ) \leq \epsilon ^ { \mathrm { t r a j } } \} , } \\ & { \mathcal { C } ^ { \mathrm { c a u s a l } } : = \{ \tau : s _ { t } ( \tau ) \leq \epsilon ^ { \mathrm { t r a j } } , \forall 1 \leq t < T \} . } \end{array}
$$

For any trajectory $\tau ,$ the definition of its maximum score gives

$$
\begin{array} { r l } { \tau \in \mathcal { C } ^ { \mathrm { t r a j } } \iff \underset { 1 \leq t < T } { \operatorname* { m a x } } s _ { t } ( \tau ) \leq \epsilon ^ { \mathrm { t r a j } } } \\ { \iff s _ { t } ( \tau ) \leq \epsilon ^ { \mathrm { t r a j } } , \quad \forall 1 \leq t < T } \\ { \iff \tau \in \mathcal { C } ^ { \mathrm { c a u s a l } } . } \end{array}
$$

The two sets therefore describe the same acceptance event. Since the trajectories are exchangeable and the score function is fixed, their maximum scores are also exchangeable:

$$
\mathbb { P } ( \tau _ { \mathrm { t e s t } } \in \mathcal { C } ^ { \mathrm { t r a j } } ) \geq \frac { k } { N _ { \mathrm { c a l i b } } + 1 } \geq 1 - \alpha .
$$

Substituting $\mathcal { C } ^ { \mathrm { c a u s a l } } = \mathcal { C } ^ { \mathrm { t r a j } }$ proves (50).

Thus, trajectory-level calibration guarantees coverage at all transitions over the calibrated horizon with a single miscoverage level $\alpha ,$ while allowing temporal dependence within each trajectory. For i.i.d. trajectories, the dataset-conditional bound in Theorem 4 also applies to the maximum scores with $n = N _ { \mathrm { c a l i b } }$ . These guarantees rely on exchangeability between calibration and deployment trajectories.

## APPENDIX C

## BACKGROUND: HJ REACHABILITY ANALYSIS

A safety filter is a policy-agnostic control mechanism that safeguards any task policy by evaluating its proposed actions and intervening when necessary to prevent failure under admissible disturbances [67]. We consider the discrete-time system $s _ { t + 1 } ~ = ~ f ( s _ { t } , a _ { t } , d _ { t } )$ , where $s _ { t } \ \in \ S , \ a _ { t } \ \in \ A ,$ , and $d _ { t } ~ \in ~ D$ denotes an unknown bounded disturbance. Safety is specified by a margin function $\ell ,$ where negative values indicate failure. Following [43], we define the failure set as $F = \{ s : \ell ( s ) < 0 \}$ , so states with zero margin lie on the safety boundary rather than in the failure set.

Maximal Robust Safe Set. Our goal is to identify the states from which the robot can avoid failure for all time despite admissible disturbances. Let $\pi : { \mathcal { S } }  A$ and $\pi _ { d } : \mathcal { S }  D$ be feedback policies, and let $s _ { s } ^ { \pi , \pi _ { d } } ( t )$ denote the trajectory starting from s under these policies. We define the maximal robust safe set as

$$
\begin{array} { r } { \Omega ^ { \star } : = \big \{ s \in { \mathcal S } : \exists \pi , \forall \pi _ { d } , \forall t \in \mathbb N _ { 0 } , s _ { s } ^ { \pi , \pi _ { d } } ( t ) \notin F \big \} . } \end{array}
$$

For every state in $\Omega ^ { \star }$ , there exists a control policy that keeps the system outside F under every admissible disturbance.

Dynamic Programming Isaacs Equation. HJ reachability characterizes this set through a safety value function [7]. Since a trajectory enters failure whenever its margin becomes negative, we evaluate the smallest margin attained over time. The controller maximizes this margin against the worst-case disturbance, giving the zero-sum safety game:

$$
V ( s ) : = \operatorname* { s u p } _ { \pi } \operatorname* { i n f } _ { \pi _ { d } } \operatorname* { i n f } _ { t \in \mathbb { N } _ { 0 } } \ell ( s _ { s } ^ { \pi , \pi _ { d } } ( t ) ) .\tag{51}
$$

The value $V ( s )$ therefore represents the minimum future safety margin that an optimal controller can maintain despite disturbances. A positive value indicates that the controller can keep the system safely outside the failure set, while a negative value indicates unavoidable failure under an adverse disturbance [12], [67], [43]. To compute this value, we apply the discrete-time dynamic programming Isaacs equation:

$$
V ( s ) = \operatorname* { m i n } \biggl \{ \ell ( s ) , \operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { m i n } _ { d \in D } V ( f ( s , a , d ) ) \biggr \} .\tag{52}
$$

The infinite-horizon value defines the maximal robust safe set:

$$
\Omega ^ { \star } = \{ s : V ( s ) \geq 0 \} ,
$$

A corresponding robust safety policy is

$$
\pi ^ { \mathrm { r o b } } ( s ) \in \underset { a \in \cal { A } } { \mathrm { a r g m a x } } \underset { d \in \cal { D } } { \mathrm { m i n } } V \left( f ( s , a , d ) \right) .
$$

This value function provides a criterion for safety filtering: starting within $\Omega ^ { \star }$ , the filter allows $\pi ^ { \mathrm { t a s k } } ( s )$ if its worst-case next-state value is nonnegative; otherwise, it intervenes with $\pi ^ { \mathrm { r o b } } ( s )$ to keep the system within the robust safe set.

Discounted Safety Update. The undiscounted backup in (52) is not generally a contraction. Consequently, a discount factor is introduced $\gamma \in [ 0 , 1 )$ for a contraction mapping [23]:

$$
V ( s ) \gets ( 1 - \gamma ) \ell ( s ) + \gamma \operatorname* { m i n } \Biggl \{ \ell ( s ) , \operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { m i n } _ { d \in D } V ( f ( s , a , d ) ) \Biggr \} .\tag{53}
$$

Game-Theoretic Adversarial RL. While (53) defines a contraction mapping for the robust safety value function, solving it directly becomes intractable in high-dimensional spaces. To approximate its solution, adversarial actor–critic methods learn a safety critic $Q _ { \omega } ( s , a , d )$ together with a control policy and a disturbance policy [12], [3]. The critic evaluates the nextstate safety value, $Q ( s , a , d ) : = V \left( f ( s , a , d ) \right)$ . The control actor maximizes the critic value, while the disturbance actor minimizes it. Repeated adversarial gameplay jointly improves the safety-value approximation and the competing policies. However, such game-theoretic learning can oscillate as the two policies continually adapt to one another. To stabilize training, ISAACS updates the disturbance and critic more frequently than the controller and maintains a leaderboard of past opponents, reducing overfitting to a single adversary [12]. Similarly, separating the learning rates allows the disturbance to more closely track a best response while the controller evolves more slowly; together with suitable critic updates, this supports local convergence under the assumptions in [77].

Operational Design Domain. These methods assume access to a system dynamics model $f ( s , a , d )$ , such as a simulator, together with a physically meaningful disturbance space D specified by the designer to capture uncertainty within the operational design domain (ODD), such as bounded external forces or model errors [12]. Thus, while the disturbance policy is learned, the admissible uncertainty itself is prescribed. In LUCID, the game instead operates over learned latent dynamics: the world model provides imagined transitions, while the disturbance perturbs the latent predictive distribution within an uncertainty set calibrated from data. This extends adversarial safety learning to systems for which physical dynamics and disturbances are difficult to specify explicitly.

## APPENDIX D IMPLEMENTATION DETAILS

## A. Adversarial Reinforcement Learning

Algorithm 1 summarizes the game-theoretic adversarial reinforcement learning procedure used to compute the robust latent safety filter in Sec. VI-B. Following Soft Actor-Critic [80], we implement $Q _ { \mathrm { r o b } }$ with twin safety critics and use their minimum in both the control and disturbance updates to mitigate value overestimation; the target critic $\hat { Q } _ { \mathrm { r o b } }$ is updated as an exponential moving average of the critic parameters. The replay buffer B stores each imagined transition together with the parameters of the nominal latent predictive distribution, so that the current disturbance policy can efficiently resample perturbed next latent states at every update without re-querying the world model. Rollouts are collected in parallel across independent WM imaginations, and transitions from all imaginations are aggregated into $\boldsymbol { B }$ and sampled in minibatches for replay updates.

Each of the $N _ { \mathrm { i t e r } }$ training iterations starts from a newly encoded trajectory and imagines up to $H _ { \mathrm { i m a g } }$ steps. We truncate an imagination once the next latent state satisfies $s _ { \mathrm { O O D } } ( z _ { t + 1 } ) > \epsilon _ { \mathrm { O O D } }$ , preventing rollouts from extending into regions unsupported by the training data. Each imagination step is followed by $G$ replay updates, where $j$ counts replay updates across iterations, $K _ { \pi }$ and $K _ { d }$ are the policy and disturbance update intervals, $\eta _ { Q } , \eta _ { \pi } , \eta _ { d }$ are the corresponding learning rates, and $\rho$ is the target-update coefficient. The world model and the uncertainty models remain fixed throughout safety-filter training.

Algorithm 1 Adversarial RL for Robust Latent Safety Filter   
Require: $\begin{array} { r } { \overline { { \mathcal { D } _ { \mathrm { t r a i n } } , \mathcal { E } , f _ { z } , \ell _ { z } } } , } \end{array}$ s<sub>OOD</sub>, ϵ<sub>KL</sub>, ϵ<sub>OOD</sub>;   
$N _ { \mathrm { i t e r } } , H _ { \mathrm { i m a g } } , G , K _ { \pi } , K _ { d }$   
Ensure: $Q _ { \mathrm { r o b } } , ~ \pi _ { \mathrm { r o b } } , ~ \pi _ { d }$   
1: Initialize $\theta _ { Q } , \theta _ { \pi } , \psi$ (zero residual output); ${ \bar { \theta } } _ { Q }  \theta _ { Q } , B  \emptyset ,$   
$j  0$   
2: for $n = 1 , \ldots , N _ { \mathrm { i t e r } }$ do   
3: $\tau \sim \mathcal { D } _ { \mathrm { t r a i n } } , \quad z _ { 0 } \sim \mathcal { E } ( \cdot \mid o { \le } t _ { 0 } ) ,$ with $\scriptstyle O \leq t _ { 0 }$ from τ   
4: for $t = 0 , \ldots , H _ { \mathrm { i m a g } } - 1$ do   
5: $a _ { t } \sim \pi _ { \mathrm { r o b } } ( \cdot \mid z _ { t } )$   
6: $z _ { t + 1 } \sim f _ { z } ^ { \pi _ { d } } ( \cdot \mid z _ { t } , a _ { t } )$   
7: $\mathcal { B }  \mathcal { B } \cup \{ ( z _ { t } , a _ { t } , \ell _ { z } ( z _ { t } ) , f _ { z } ( \cdot \mid z _ { t } , a _ { t } ) ) \}$   
8: for $g = 1 , \ldots , G$ do   
9: $\begin{array} { r } { \mathcal { M } \sim \mathcal { B } , \quad j \gets j + 1 } \end{array}$   
10: $z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } ^ { \prime } } ( \cdot \stackrel { \sim } { | } z , a ) , \quad a ^ { \prime } \sim \pi _ { \mathrm { r o b } } ( \cdot \mid z ^ { \prime } ) , \quad ( z , a ) \in \mathcal { M }$   
11: $\theta _ { Q }  \theta _ { Q } - \eta _ { Q } \nabla _ { \theta _ { Q } } \mathcal { L } _ { Q _ { \mathrm { r o b } } }$ (20)   
12: if j mod $K _ { \pi } = 0$ then   
13: $\hat { a } \sim \pi _ { \mathrm { r o b } } ( \cdot \mid z )$   
14: $\theta _ { \pi }  \theta _ { \pi } - \eta _ { \pi } \nabla _ { \theta _ { \pi } } \mathcal { L } _ { \pi _ { \mathrm { r o b } } }$ (21b)   
15: end if   
16: if j mod $K _ { d } = 0$ then   
17: $z ^ { \prime } \sim f _ { z } ^ { \pi _ { d } } ( \cdot \mid z , a )$ (reparameterized)   
18: $a ^ { \prime } \gets :$ stopgrad $\phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x }$   
19: $\psi  \psi - \eta _ { d } \nabla _ { \psi } \mathcal { L } _ { \pi _ { d } }$ (21a)   
20: end if   
21: $\bar { \theta } _ { Q }  ( 1 - \rho ) \bar { \theta } _ { Q } + \rho \theta _ { Q }$   
22: end for   
23: $\textbf { i f } s _ { \mathrm { O O D } } ( z _ { t + 1 } ) > \epsilon _ { \mathrm { O O D } }$ then   
24: break ▷ truncate OOD imagination   
25: end if   
26: end for   
27: end for

## B. Latent Safety Filter Training

Because the world model is trained offline, actions proposed during safety filter training may be poorly represented in the training data for the current latent state. Such out-ofdistribution (OOD) state–action pairs can induce hallucinated transitions, allowing the safety policy to exploit model errors and overestimate safety. Following UNISafe [11], we incorporate epistemic uncertainty into the safety margin to discourage these unreliable imaginations. Augmenting the task failure margin with this uncertainty criterion encourages the learned safety policy to select actions whose predicted outcomes are both safe and supported by the offline data.

## C. Disturbance Parameterization

In Sec. V-C, we parameterize the latent disturbance $\pi _ { d } ( z , a )$ as an additive residual to the nominal distribution parameters, enabling optimization over plausible next-state distributions through (11)–(12). We implement this network as an MLP that takes features of the nominal next-state prediction: the deterministic features h and prior parameters $( \mu , \sigma )$ for Gaussian dynamics, or h alone for categorical dynamics. Thus, the current latent state and action condition the disturbance through the nominal prediction. The residual changes only the stochastic prior, preserving the deterministic features.

TABLE VII  
RESIDUAL PARAMETERIZATION OF $\pi _ { d }$ IN (11).
<table><tr><td>Component</td><td>Gaussian</td><td>Categorical</td></tr><tr><td>Input</td><td> $[ h , \mu , \sigma ]$ </td><td>h</td></tr><tr><td>Input dimension</td><td> $5 1 2 + 3 2 + 3 2$ </td><td>512</td></tr><tr><td>Hidden layers</td><td>4 layers, 512 units each</td><td></td></tr><tr><td>Normalization / activation</td><td>LayerNorm / ReLU</td><td></td></tr><tr><td>Output residual</td><td> $( \Delta \mu , \Delta \sigma )$ </td><td> $\Delta \phi$ </td></tr><tr><td>Output dimension</td><td>32 + 32</td><td>32 × 32</td></tr><tr><td>Output transform</td><td>(tanh, 0.5 tanh)  $\mu + \Delta \mu$ </td><td>15 tanh</td></tr><tr><td>Prior update</td><td> $\sigma + \Delta \sigma$ </td><td> $\phi + \Delta \phi$ </td></tr><tr><td>Output initialization</td><td colspan="2">Zero weights and bias</td></tr></table>

Table VII summarizes the parameterization. For Gaussian dynamics, we perturb the standard deviation σ, and the covariance in (12) is obtained by squaring these standard deviations. For categorical dynamics, we add the residual to the prior logits ϕ. Zero initialization of the output layer makes the initial residual zero, and the KL and OOD penalties constrain the learned perturbation during optimization.

## D. 3D Dubins’ Car with Disturbances

Optimal Solution of Naughty Dubins’ Car. In this section, we derive the optimal robust control and worst-case disturbance for the naughty Dubins’ car in (6):

$$
\dot { p } _ { x } = v \cos \theta , \qquad \dot { p } _ { y } = v \sin \theta , \qquad \dot { \theta } = \delta a ,
$$

Let $V _ { \mathrm { g t } } ( s )$ be the optimal robust safety value for these physical dynamics. Define its gradient components as

$$
g _ { x } : = \frac { \partial V _ { \mathrm { g t } } } { \partial p _ { x } } , \qquad g _ { y } : = \frac { \partial V _ { \mathrm { g t } } } { \partial p _ { y } } , \qquad g _ { \theta } : = \frac { \partial V _ { \mathrm { g t } } } { \partial \theta } .
$$

By the chain rule, the Hamiltonian gives its instantaneous change along the dynamics:

$$
H ( s , a , \delta ) : = \nabla _ { s } V _ { \mathrm { g t } } ( s ) ^ { \top } \dot { s } = v ( g _ { x } \cos \theta + g _ { y } \sin \theta ) + g _ { \theta } \delta a .
$$

The action maximizes this, while the disturbance minimizes it after observing the action. The optimization reduces to

$$
\operatorname* { m a x } _ { | a | \le 1 . 2 5 } \operatorname* { m i n } _ { \delta \in \{ - 1 , 1 \} } { g _ { \theta } \delta a } = \operatorname* { m a x } _ { | a | \le 1 . 2 5 } - | g _ { \theta } a | = 0 .
$$

Thus, a Hamiltonian-optimal robust action is $\mathbf { a } ^ { \star } = \mathbf { 0 }$ , and a worst-case response to any proposed action is

$$
\delta ^ { \star } = \left\{ { \begin{array} { l l } { - 1 , } & { g _ { \theta } a > 0 , } \\ { + 1 , } & { g _ { \theta } a \le 0 . } \end{array} } \right.
$$

When $g _ { \boldsymbol { \theta } } \neq 0 ,$ , every nonzero turn lets the disturbance decrease safety through the heading contribution. Driving straight, as in Fig. 2, eliminates this contribution.

Latent Safety Filter Setup. World model hyperparameters are listed in Table VIII and safety filter training settings are listed in Table IX. We add the predicted residuals to the mean and standard deviation of the Gaussian prior, with the standard deviation bounded below by 0.1. We train a flowmatching density model on frozen latents and use its logpZO score [74] to penalize implausible perturbations. For safety learning, we train a Gaussian safety actor and twin critics using the discounted safety update in (20). The actor and critic networks each use four hidden layers of width 512, and the critic outputs are bounded by 1.5 tanh(·). We update the critics, safety policy, and disturbance every 1, 4, and 2 replay updates, respectively.

TABLE VIII  
3D DUBINS’ CAR LATENT WORLD MODEL SETTINGS.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Image Dimension</td><td> $1 2 8 \times 1 2 8 \times 3$ </td></tr><tr><td>Action Dimension</td><td>1 (continuous)</td></tr><tr><td>Stochastic Latent</td><td>Gaussian</td></tr><tr><td>Deterministic Dimension</td><td>512</td></tr><tr><td>Stochastic Dimension</td><td>32</td></tr><tr><td>Activation Function</td><td>SiLU</td></tr><tr><td>Encoder CNN Depth</td><td>32</td></tr><tr><td>Failure Classifier Layers</td><td>2</td></tr><tr><td>Batch Size</td><td>16</td></tr><tr><td>Batch Length</td><td>32</td></tr><tr><td>Optimizer</td><td></td></tr><tr><td>Learning Rate</td><td>Adam 10⁻4</td></tr><tr><td>Training Iterations</td><td>50,000</td></tr></table>

TABLE IX

3D DUBINS’ CAR LATENT SAFETY FILTER SETTINGS.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Actor Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Critic Learning Rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Disturbance Learning Rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Discount Factor</td><td>0.85 → 0.9999</td></tr><tr><td>Training Iterations</td><td>100,000</td></tr><tr><td>Replay Buffer Size</td><td>100,000</td></tr><tr><td>Batch Size</td><td>512</td></tr><tr><td>Parallel Imagination States</td><td>512</td></tr><tr><td>Imagination Steps</td><td>50</td></tr></table>

## E. Simulation: Block Pouring

System Disturbance. We randomize the environment physics at reset, with all randomized parameters sampled uniformly from the ranges listed in Table X. The poses of all three cubes are perturbed within the specified bounds, while the physical properties of the green cube are randomized independently. These parameters remain fixed throughout each episode. For paired evaluation, we pre-sample 20 physics configurations with a fixed seed and reuse them across source trajectories and filtering conditions, ensuring identical physics for corresponding rollouts.

Latent Safety Filter Setup. World model hyperparameters are listed in Table XI and safety filter training settings are listed in Table XII. Following UNISafe [11], a five-member probabilistic ensemble provides the log-JRD uncertainty score used in the safety margin to prevent OOD action exploitation of WM imaginations [11]. For safety filter synthesis, the actor and critic networks each use four hidden layers of width 512, and update the critics, safety policy, and disturbance every 1, 8, and 2 replay updates, respectively.

TABLE X  
PHYSICAL RANDOMIZATION OF BLOCK POURING IN ISAACLAB.
<table><tr><td>Parameter</td><td>Randomization Range</td></tr><tr><td>Cube Position x</td><td>[−0.02, 0.02] m</td></tr><tr><td>Cube Position y</td><td>[−0.02, 0.02] m</td></tr><tr><td>Mass</td><td>[0.8, 1.2] kg</td></tr><tr><td>Static Friction</td><td>[0.2, 0.8]</td></tr><tr><td>Dynamic Friction</td><td>[0.1, 0.5]</td></tr><tr><td>Restitution</td><td>[0.05, 0.3]</td></tr></table>

TABLE XI  
BLOCK POURING WORLD MODEL TRAINING SETTINGS.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Image Dimension</td><td> $2 \times 1 2 8 \times 1 2 8 \times 3$ </td></tr><tr><td>Proprioception Dimension</td><td>7 (eef position/quaternion)</td></tr><tr><td>Action Dimension</td><td>7 (continuous)</td></tr><tr><td>Stochastic Latent</td><td>Categorical</td></tr><tr><td>Deterministic Dimension</td><td> $5 1 2$ </td></tr><tr><td>Stochastic Dimension</td><td> $3 2 \times 3 2$ </td></tr><tr><td>Activation Function</td><td>SiLU</td></tr><tr><td>Encoder CNN Depth</td><td>32</td></tr><tr><td>Encoder MLP Layers</td><td>5</td></tr><tr><td>Failure Classifier Layers</td><td>3</td></tr><tr><td>Batch Size</td><td>16</td></tr><tr><td>Batch Length</td><td>64</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning Rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Training Iterations</td><td>1,000,000</td></tr></table>

TABLE XII

SIMULATED MANIPULATION SAFETY-LEARNING SETTINGS.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Actor Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Critic Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Disturbance Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Discount Factor</td><td>0.95</td></tr><tr><td>Training Iterations</td><td>400,000</td></tr><tr><td>Replay Buffer Size</td><td>1,000,000</td></tr><tr><td>Batch Size</td><td>512</td></tr><tr><td>Parallel Imagination States</td><td>1,024</td></tr><tr><td>Imagination Steps</td><td>50</td></tr></table>

Task Policy. We train the DreamerV3 as a task policy [8] independently of the safety filter. The actor optimizes imagined λ-returns with $\lambda = 0 . 9 5$ and continuation-weighted discount, backpropagating through the dynamics with an entropy coefficient of $3 \times 1 0 ^ { - 4 }$ . We normalize advantages using moving 5th and 95th return percentiles (update rate $1 0 ^ { - 2 }$ , minimum scale 1), and the critic uses a twohot log-likelihood objective with a slow-target regularizer. We also use Diffusion Policy [79] by behavior cloning as a task policy. A vision-conditioned MLP or conditional U-Net predicts 8 future actions of 7D using 100 denoising steps. Both are trained for 1,000 epochs at learning rate $1 0 ^ { - 4 }$ ; the MLP uses ViT image encoders and batch size 256, while the U-Net uses ResNet-18 encoders and batch size 64. During deployment, the task policy executes four actions before replanning.

## F. Real-World: Sunny-Side-Up Egg Serving

TABLE XIII  
TRANSFORMER-BASED LATENT WORLD MODEL
<table><tr><td></td><td>RSSM</td><td>Transformer RSSM</td></tr><tr><td>Visual encoder</td><td>Trainable CNN</td><td>Frozen DINOv3</td></tr><tr><td>Temporal model</td><td>GRU</td><td>Causal transformer</td></tr><tr><td>History</td><td>Recurrent</td><td>8-step attention window</td></tr><tr><td>Recon. Target</td><td>Pixels</td><td>Pixels and DINO features</td></tr></table>

TABLE XIV  
REAL WORLD: WORLD MODEL HYPERPARAMETERS.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Image Dimension</td><td> $2 \times 1 9 2 \times 2 5 6 \times 3$ </td></tr><tr><td>Proprioception Dimension</td><td>8</td></tr><tr><td>Action Dimension</td><td>7 (continuous)</td></tr><tr><td>Stochastic Latent</td><td>Gaussian</td></tr><tr><td>Deterministic Dimension</td><td>512</td></tr><tr><td>Stochastic Dimension</td><td>32</td></tr><tr><td>Activation Function</td><td>SiLU</td></tr><tr><td>Encoder MLP Layers</td><td>2 (proprioception)</td></tr><tr><td>Failure Classifier Layers</td><td>3 per failure head</td></tr><tr><td>Batch Size</td><td>20</td></tr><tr><td>Batch Length</td><td>64</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning Rate</td><td>10⁻4</td></tr><tr><td>Training Iterations</td><td>200,000</td></tr></table>

TABLE XV  
REAL WORLD LATENT SAFETY FILTER SETUPS.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Actor Learning Rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Critic Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Disturbance Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Target Smoothing</td><td>0.005</td></tr><tr><td>Discount Factor</td><td>0.85 → 0.9999</td></tr><tr><td>Training Iterations</td><td>400,000</td></tr><tr><td>Replay Buffer Size</td><td>500,000</td></tr><tr><td>Batch Size</td><td>512</td></tr><tr><td>Parallel Imagination States</td><td>256</td></tr><tr><td>Imagination Steps</td><td>30</td></tr></table>

Latent World Model. Each observation contains RGB images from two tabletop cameras, seven robot joint positions, and the gripper width, while actions specify an end-effector delta pose and gripper command. Failures are manually labeled for each frame based on falling and flipping conditions. For latent world mode, we adapt the RSSM [61] with pretrained visual features and transformer-based latent dynamics, with the main architectural differences summarized in Table XIII. A frozen DINOv3 ViT-S/16+ [85] extracts 192 patch tokens of dimension 384 from each image; learned view embeddings distinguish the two cameras, and a two-layer transformer fuses their tokens. Eight learned queries pool the fused tokens into 3,072 visual features, while a separate MLP maps the eight proprioceptive inputs to 512 features. Their concatenation forms the 3,584-dimensional observation embedding.

For latent dynamics, a causal transformer predicts 512 deterministic features from an eight-step history of stochastic states and actions, providing the same temporal context during training and imagination. Actions are embedded and injected into each transformer block through AdaLN-zero conditioning. The prior predicts a 32-dimensional diagonal Gaussian from the deterministic features, and concatenating a stochastic sample with the deterministic features yields the 544-dimensional latent state z used by the safety filter. We jointly train pixel and frozen-feature reconstruction together with continuation, falling, and flipping prediction. Table XIV lists the architecture and training hyperparameters.

Latent Safety Filter Setup. Table XV summarizes the safetyfilter training settings. Following Algorithm 1, we update the critics, safety policy, and disturbance every 1, 4, and 2 replay updates, respectively.