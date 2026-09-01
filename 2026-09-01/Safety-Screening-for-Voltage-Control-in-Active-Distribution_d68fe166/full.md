# Safety Screening for Voltage Control in Active Distribution Grids via Distributionally Robust Conformal Screening

Sarra Bouchkati<sup>\*,†</sup>, Petros Ellinas<sup>\*,†</sup>, Adriana Geisler<sup>\*,†</sup>, Steffen Kortmann, Johanna Vorwerk, Spyros Chatzivasiliadis, and Andreas Ulbig

Abstract—Deploying a new control policy for voltage control in active distribution grids requires evidence that physical limits will be satisfied before the policy is tested on the physical grid. This assessment is difficult for two reasons. First, simulations cannot capture every disturbance, modeling error, and device interaction present in the real grid. Second, historical measurements reflect operation under existing control policies, whereas a new policy may drive the grid into different operating conditions. To address these challenges, we propose Distributionally Robust Conformal Safety Screening (DR-CSS), a policy-agnostic framework for predeployment, scenario-by-scenario screening of a new control policy using historical data and a nominal simulator. For each new scenario, the simulator predicts a future voltage trajectory for the whole grid; DR-CSS then constructs a conformal safety interval around this prediction using historical simulation-toreality errors. The interval is further enlarged to account for closed-loop changes induced by the deployment of the new policy and its interactions with the remaining controllers. To the best of our knowledge, DR-CSS is the first framework in power systems to combine historical data from an existing control policy with an imperfect simulator for pre-deployment safety screening of a new policy. Experiments on the IEEE 33-bus and IEEE 141- bus systems evaluate the deployment of learning-based voltage control policies and show that DR-CSS identifies all unsafe test scenarios. To reduce unnecessary warnings on safe scenarios, we adapt the safety intervals to different operating conditions and gradually introduce new policies with recalibration after each stage. These extensions increase the informational value of the safety screening and support safer deployment decisions in active distribution grids.

Index Terms—Active distribution grids, conformal prediction, distribution shift, learning-based control, voltage control, voltage safety.

## I. INTRODUCTION

One of the growing challenges in operating active distribution grids is maintaining bus voltages within permissible limits. Numerous inverter-interfaced distributed energy resources (DERs), flexible loads, and heterogeneous control devices operate as independent decision-making components, forming a complex multi-agent environment that shapes realtime grid operation. While the intermittency of renewable DERs increases the risk of voltage-limit violations [1], smart inverters can help regulate voltages through their fast and continuously adjustable reactive-power capabilities [2]. Coordinating reactive-power injections to maintain acceptable voltage profiles, often while limiting control effort or associated costs, is commonly referred to as the Volt-Var control problem. Various control policies, or automated decision-making rules, have been proposed to address this problem, ranging from simple rule-based local droop schemes to advanced learningbased methods such as reinforcement learning (RL) [3]–[7]. However, because these policies often act solely based on only partial grid information, they may fail to guarantee gridwide safe voltage control due to the lack of coordination [8]. Several approaches aimed at addressing safety at the control policy-design stage. Safe RL techniques embed safety layers or constraint-satisfaction mechanisms directly into the training process to minimize operational risk, typically at the cost of more conservative exploration, or degraded nominal performance [4], [9], [10]. Crucially, these methods aim to enforce safety during policy training, but their guarantees are typically tied to the assumed training environment. They therefore do not guarantee grid-wide safety after deployment, where interactions with the physical grid and other controllers may lead to unanticipated voltage violations.

Deploying a new control policy at one or more inverters changes its interactions with the grid and with the controllers that remain in operation. Therefore, a key open problem is how to assess, before deployment, whether a new control policy introduced at one or more inverter buses will maintain all bus voltages within their admissible limits, using historical data and an imperfect simulator and without testing unsafe conditions on the physical system.

Ensuring the safe deployment of new control policies involves two complementary procedures: validation, which evaluates performance through simulation and empirical testing, and analytical verification, which provides analytical safety guarantees. Both are challenging in distribution grids because grid operators often lack a sufficiently accurate model of the physical system. In practice, distribution-grid models are limited by incomplete observability, data-quality issues, and imperfect topology knowledge [11], [12], as well as uncertainties in renewable injections, load behaviour, and measurement errors [13]. Traditional validation tools, such as scenario simulation and hardware-in-the-loop testing [14]–[16], therefore remain conditional on model fidelity and scenario representativeness. Direct testing of potentially unsafe conditions on the physical grid is also unacceptable, since power grids constitute a safety-critical infrastructure. Furthermore, historical operational data provides only limited support for evaluating new control policies, since it is generated under the closedloop behavior of previously deployed policies. Consequently, it may not adequately represent the states, interactions, and control actions that would arise after an untested control policy is introduced.

Analytical verification workflows share a common modeling bottleneck: their guarantees are only as reliable as the models on which they are built. Formal verification, reachability analysis, and certificate-based methods can provide strong deterministic guarantees under suitable assumptions, but they typically require accurate system models, tractable mathematical abstractions, and substantial computational resources [17]–[20]. When the underlying model deviates from the true system, these guarantees no longer transfer to physical deployment.

Statistical uncertainty quantification offers a complementary way to assess new control policies before deployment. Existing approaches include probabilistic voltage-risk assessment and conformal prediction for power-system forecasts and voltage trajectories [21]–[24]. Conformal prediction is attractive because it uses past differences between predictions and observations to construct uncertainty intervals without assuming a specific probability distribution. However, these intervals remain reliable only when past observations are representative of the setting being assessed. This condition may fail after a new control policy is deployed at one or more inverters because this new policy can produce different closed-loop grid behaviour through its interactions with the grid and the remaining controllers. Consequently, uncertainty intervals calibrated under the pre-existing control policy may not remain valid after deploying a new policy.

This is an off-policy conformal prediction setting, in which data collected under the existing control policies are used to assess a new one before deployment. Although off-policy conformal methods have been studied in reinforcement-learning and multi-agent applications [25], [26], to the best of our knowledge, this work is the first in the power-systems literature to formulate control policy replacement as an off-policy conformal problem for grid-wide voltage-safety screening.

We propose Distributionally Robust Conformal Safety Screening (DR-CSS), a policy-agnostic statistical framework for pre-deployment, scenario-by-scenario screening of a new control policy using historical data and an imperfect nominal simulator. DR-CSS complements existing validation and verification tools and can be applied to predictions produced by power-flow simulators, reduced-order models, learned surrogates, or verification workflows. For each operating scenario, the nominal simulator produces a future voltage trajectory, around which DR-CSS constructs an uncertainty interval using historical simulation-to-reality errors. The interval is then enlarged to account for bounded changes in prediction-error behaviour after a new control policy is deployed at one or several inverters.

We further extend DR-CSS in two practical ways. First, context-aware calibration adapts the uncertainty intervals to different operating conditions, so that one overly conservative interval is not used for every scenario. Second, gradual policy change introduces the new control policy in stages and recalibrates DR-CSS after each stage, reducing the change that must be handled if several inverters get a new control policy at once.

In summary, the main contributions are:

• We introduce, to the best of our knowledge, the first framework in power systems for pre-deployment, scenario-by-scenario screening of a new control policy. Distributionally Robust Conformal Safety Screening (DR-CSS) is model-agnostic and thus can be applied to any black-box control policy.

• We extend DR-CSS with context-aware calibration, which adapts the uncertainty intervals to different operating conditions, and gradual control policyreplacement, which updates the screening method after each deployment stage.

• Employing the IEEE 33-bus and 141-bus cases, we demonstrate how DR-CSS reaches the target statistical coverage with finite and informative uncertainty intervals, reduces the intervals’ average width by up to 69%, and flags every unsafe voltage trajectory in the gradualdeployment experiment.

The remainder of the paper is organized as follows. Section II formulates the voltage-control and deploymentassessment problem. Section III presents DR-CSS. Section IV reports the numerical results. Finally, Section V concludes the paper and outlines future work directions.

## II. SAFETY-SCREENING PROBLEM FORMULATION

Before introducing DR-CSS, this section defines the closedloop distribution-grid model, the joint control policy configuration, resulting from the combination of pre-existing control policies and new policies to be deployed, and the scenariolevel voltage-safety event assessed by the proposed method.

## A. Distributed Voltage-Control Setting

Let $B = \{ 1 , \ldots , B \}$ denote the set of buses in a distribution grid, and let $\mathcal { T } _ { \mathrm { i n v } } \subseteq B$ denote the subset of buses equipped with controllable converter-based reactive-power support and $\mathcal { L } \subseteq B$ denote the set of buses equipped with loads. At time t let $q _ { t } ~ : = ~ ( Q _ { i , t } ) _ { i \in  { \mathbb { Z } } _ { \mathrm { i n v } } }$ collect the inverter reactivepower commands, and let $d _ { t }$ denote uncontrollable operating conditions such as loads and active-power injections. The busvoltage magnitudes are given by $v _ { t } : = ( V _ { i , t } ) _ { i \in B }$ , and $V _ { \mathrm { m i n } }$ and $V _ { \mathrm { m a x } }$ denote the minimum and maximum admissible voltage magnitudes, respectively.

We observe the grid state up to time H and assess its future behavior over the look-ahead horizon $\begin{array} { r } { \mathcal { T } _ { \mathrm { f } } : = \{ H + \} } \end{array}$ $1 , . . . , T \} , T > H$ . The main control target, defined by the Volt-Var problem, is to minimize reactive-power use while maintaining admissible bus voltages. If the future operating conditions were known and the inverter commands could be chosen directly at each time step, the problem could be formulated as follows:

$$
\operatorname* { m i n } _ { \{ q _ { t } \} _ { t \in \mathcal { T } _ { \mathrm { f } } } } \quad \sum _ { t \in \mathcal { T } _ { \mathrm { f } } } \sum _ { i \in \mathcal { T } _ { \mathrm { i n v } } } Q _ { i , t } ^ { 2 }\tag{1a}
$$

$$
\begin{array} { r } { \mathrm { s . t . } \quad v _ { t } = h _ { \mathrm { p f } } ( q _ { t } , d _ { t } ) , \quad \forall t \in \mathcal { T } _ { \mathrm { f } } , } \end{array}\tag{1b}
$$

$$
V _ { \operatorname* { m i n } } \leq V _ { i , t } \leq V _ { \operatorname* { m a x } } , \quad \forall i \in \mathcal { B } , t \in \mathcal { T } _ { \mathrm { f } } ,\tag{1c}
$$

$$
| Q _ { i , t } | \leq Q _ { i , t } ^ { \operatorname* { m a x } } ( P _ { i , t } ) , \quad \forall i \in \mathcal { T } _ { \mathrm { i n v } } , t \in \mathcal { T } _ { \mathrm { f } } .\tag{1d}
$$

Here, $h _ { \mathrm { p f } }$ is the AC power-flow mapping, $P _ { i , t }$ is the activepower injection at inverter bus i, and the limit in (1d) is determined by the inverter apparent-power capability curve and based on the current output power of each inverter. The objective penalizes reactive-power usage, while constraints require all bus voltages to remain inside the admissible voltage band.

Please note that in this work, we do not discuss the implementation of the problem formulated in (1) nor do we develop a new control policy that employs it. Instead, we assume that a control policy is already specified and we study whether it can be safely deployed using historical operating data and a nominal simulator $\Phi _ { \mathcal { M } }$ , such as an AC power-flow model, a reduced-order system, or a learned surrogate.

In reality, each inverter observes only local or neighboring measurements. Let $X _ { t }$ denote the physical system state, which may include voltages, nodal injections, forecasts, and other variables needed to describe the evolution of the closed-loop system, consisting of the distribution grid and the local control policies. The control policy need not observe $X _ { t }$ directly, but can rather have access to only a local observation $o _ { t } .$ . For inverter i, let $\mathcal { N } _ { i } \subseteq B$ denote the set of buses whose voltage magnitudes are available to it, and define

$$
o _ { i , t } : = h _ { i } ^ { \mathrm { m e a s } } ( X _ { t } ) = \left( V _ { j , t } \right) _ { j \in \mathcal { N } _ { i } } .\tag{2}
$$

The case $\mathcal { N } _ { i } = \{ i \}$ corresponds to a local droop-type control policy, while larger neighborhoods allow the use of nearby measurements. Other inputs, such as active-power injections or forecasts, can be included through $h _ { i } ^ { \mathrm { m e a s } }$ . Each inverter applies a control policy that outputs reactive power setpoint $Q _ { i , t } =$ $\pi _ { i } ( o _ { i , t } )$ , where $\pi _ { i }$ may be rule-based, optimization-based, or learning-based. The joint control policy, comprising all local policies in the grid, is denoted by $\pi : = ( \pi _ { i } ) _ { i \in \mathbb { Z } _ { \mathrm { i n v } } }$ . The closedloop system evolves as

$$
X _ { t + 1 } = F ( X _ { t } , q _ { t } , W _ { t } ) ,\tag{3}
$$

where $F$ denotes the one-step physical state-transition mapping of the grid, including the grid, device, and controller interactions, and $W _ { t }$ collects exogenous effects such as load changes, PV generation, forecast errors, and modeling disturbances. Thus, future voltages depend jointly on operating conditions, control policyactions, and the physical grid response.

Since voltage safety is a horizon-wide property, we consider the full future trajectory

$$
v _ { H + 1 : T } : = ( v _ { H + 1 } , . . . , v _ { T } ) .\tag{4}
$$

The worst voltage-limit violation over the look-ahead horizon is

$$
\begin{array} { r } { g ( v _ { H + 1 : T } ) : = \underset { t \in \mathcal { T } _ { \mathrm { f } } } { \operatorname* { m a x } } \underset { i \in \mathcal { B } } { \operatorname* { m a x } } \Big \{ \mathrm { m a x } \{ V _ { i , t } - V _ { \operatorname* { m a x } } , 0 \} , } \\ { \mathrm { m a x } \{ V _ { \mathrm { m i n } } - V _ { i , t } , 0 \} \Big \} . } \end{array}\tag{5}
$$

The trajectory $v _ { H + 1 : T }$ is safe if and only if $g ( v _ { H + 1 : T } ) = 0$

## B. Safety Screening Goal

Consider a grid historically operated under an existing baseline control policy $\pi ^ { \mathrm { b a s e } }$ , from which historical operating

records are available. A new candidate policy $\pi ^ { \mathrm { n e w } }$ is proposed for deployment across a subset of inverter buses $\mathcal { E } \subseteq \mathbb { Z } _ { \mathrm { i n v } }$ Starting at time H, the joint deployed policy $\pi ^ { \mathrm { d e p } }$ satisfies:

$$
Q _ { i , t } = \left\{ \begin{array} { l l } { \pi _ { i } ^ { \mathrm { n e w } } \big ( o _ { i , t } \big ) , } & { i \in \mathcal { E } , } \\ { \pi _ { i } ^ { \mathrm { b a s e } } \big ( o _ { i , t } \big ) , } & { i \in \mathcal { T } _ { \mathrm { i n v } } \setminus \mathcal { E } . } \end{array} \right.\tag{6}
$$

Although DR-CSS can also accommodate added invertercontrol units, this paper focuses on replacing control policies at existing inverters. Given the observed grid history $X _ { 1 : H }$ and forecasts of future uncontrollable operating conditions $\ddot { d } _ { H + 1 : T }$ , we define the tested operating scenario as the tuple $x : = ( X _ { 1 : H } , \hat { d } _ { H + 1 : T } , \pi ^ { \mathrm { d e p } } )$ . A nominal closed-loop simulator $\Phi _ { \mathcal { M } }$ produces a predicted future reference voltage trajectory:

$$
\hat { v } _ { H + 1 : T } = \Phi _ { \mathcal { M } } ( x ) .\tag{7}
$$

This trajectory is considered nominal because it reflects idealized model and forecast assumptions rather than exact physical conditions. However, a discrepancy between simulation and reality is almost inevitable, which results in a distribution shift in the data. In our setting, two such shifts arise, and they are distinct rather than reducible to one another. Simulation-toreality shift reflects the fidelity of the simulator to the true grid model, that is, the discrepancy between simulated $\hat { v } _ { H + 1 : T }$ and physically realized voltages $v _ { H + 1 : T }$ due to unmodeled interactions, forecast errors, and measurement noise. Policyinduced shift, in contrast, reflects the mismatch between the existing policy $\pi ^ { \mathrm { b a s e } }$ , under which historical calibration data was collected, and the deployment policy $\pi ^ { \mathrm { d e p } }$ , whose closedloop interactions with the grid alter the operating trajectory even under identical underlying conditions. This second shift would persist even under a perfect simulator, since it originates from the feedback control loop itself rather than from model inaccuracy. For the remainder of this work, we use distribution shift to refer jointly to both sources unless otherwise specified.

The central problem addressed in this paper is therefore:

Given historical records under a baseline control policy, an observed system history, forecasts of future conditions, a candidate control policy, and a nominal closed-loop simulator, our goal is to construct a datadriven rule that assesses the safety of test scenarios under the candidate policy and flags the scenarios that are uncertain or unsafe.

## III. DISTRIBUTIONALLY ROBUST CONFORMAL SAFETY SCREENING

This section presents the workflow of DR-CSS <sup>1</sup>. Given a nominal simulator $\Phi _ { \mathcal { M } }$ , DR-CSS compares simulated and realized trajectories under baseline operation to quantify the simulation-to-reality error. Furthermore, it compares the baseline and candidate closed-loop simulations to account for the shift induced by changing the control policy at one or several inverters. It then enlarges the nominal voltage prediction band to account for both effects. Figure 1 summarizes the workflow. The left panel defines the scenario information and nominal closed-loop simulation, the middle panel converts historical simulation-to-reality errors into robust conformal scores, and the right panel applies the final voltage band to return a scenario-level accept/warning decision.

![](images/a9dfa0ca180b7fc65fdc0de8671a6bf12b0ba34cea5f917dc6c2b3f9f1dc1031.jpg)  
Fig. 1. Overview of the proposed DR-CSS workflow for safety screening of learning-based control policies under policy change. Historical data are collected under a baseline policy, while deployment may use a new candidate control policy at one or more inverter buses, shifting closed-loop voltage behavior and simulation-to-reality error statistics.

DR-CSS is an offline pre-deployment screening procedure rather than a voltage control policy. It applies the same decision rule independently to each scenario and does not modify the control policy actions. A warning identifies a scenario that requires further testing, policy revision, or exclusion from the intended operating envelope.

## A. Scenario-Level Nominal Simulation

This subsection corresponds to the left panel of Fig. 1. DR-CSS evaluates the control policy separately for each operating scenario. The observed grid history, forecasts of future uncontrollable injections, and control policy to be evaluated are provided to the nominal simulator in (7), which produces one future voltage trajectory for that scenario.

If this trajectory is used directly for safety assessment, the simulator is implicitly treated as exact. Equivalently, the predictive distribution is the point mass

$$
\mathbb { P } _ { x } ^ { \mathrm { n o m } } = \delta _ { \hat { v } _ { H + 1 : T } ( x ) } .\tag{8}
$$

where $\delta _ { z }$ denotes the Dirac probability measure concentrated at $z . \ \mathbb { P } _ { x } ^ { \mathrm { n o m } }$ assigns all probability to the single simulated trajectory. However, this is too optimistic when the simulator is affected by forecast errors, model mismatch, measurement noise, controller delays, or unmodeled dynamics. DR-CSS therefore replaces this point-mass view by a data-calibrated statistical abstraction: an uncertainty band around the nominal voltage trajectory.

## B. Voltage Error Scores and Conformal Calibration

The next step is to learn this band from data. To do so, we compare historical nominal simulations with their realized voltage trajectories and summarize each discrepancy by a scalar conformal score. The following subsection describes the calibration block in Fig. 1, which maps a historical dataset of grid trajectories down to a single, rigorous safety threshold using split conformal prediction [27]. First, we pair past historical predictions from the nominal simulator with the true, realized voltage profiles measured from the physical grid. Second, for each historical scenario, we compare these profiles to calculate a single scalar nonconformity score that captures the worst-case prediction error across all buses and time steps. We then collect these scores and sort them in ascending order. Finally, based on the target miscoverage level α, we extract a benchmark safety threshold from the high end of this sorted list. If future simulation errors match these historical error patterns, wrapping this single threshold around the nominal simulation solution provides a coverage guarantee for the entire true physical voltage trajectory, across all parts of the grid and all future hours simultaneously. Under exchangeability/calibration validity, this trajectory remains bounded within the calculated safety bands with finite-sample marginal coverage of at least 1 − α. Let

$$
\mathcal { D } _ { \mathrm { c a l } } = \Big \{ \Big ( x ^ { j } , v _ { H + 1 : T } ^ { j } \Big ) \Big \} _ { j = 1 } ^ { n _ { \mathrm { c a l } } } .\tag{9}
$$

denote historical cases used for the calibration of the split conformal prediction method, where $v _ { H + 1 : T } ^ { j }$ is the realized voltage trajectory, and $n _ { \mathrm { c a l } }$ denotes the number of calibration samples. For each case, the simulator gives

$\hat { v } _ { H + 1 : T } ^ { j } \ = \ \Phi _ { \mathcal M } ( x ^ { j } )$ . DR-CSS measures the simulationto-reality discrepancy using the trajectory-level standardized voltage-error score

$$
s _ { j } : = \operatorname* { m a x } _ { t \in \mathcal { T } _ { \mathrm { f } } } \operatorname* { m a x } _ { i \in \mathcal { B } } \gamma _ { t } \frac { \left| V _ { i , t } ^ { j } - \widehat { V } _ { i , t } ^ { j } \right| } { \widehat { \sigma } _ { i } ^ { \mathrm { e r r } } } .\tag{10}
$$

where $V _ { i , t } ^ { j }$ is the realized voltage magnitude at bus i and time t for calibration scenario $j ,$ and $\widehat { V } _ { i , t } ^ { j }$ is the corresponding nominal simulated voltage magnitude, obtained from $\widehat { v } _ { H + 1 : T } ^ { j } \ = \ \Phi _ { \mathcal { M } } ( x ^ { j } )$ . Furthermore, $\widehat { \sigma } _ { i } ^ { \mathrm { e r r } } > 0$ is a fixed busspecific residual scale estimated before conformal calibration, and $\gamma _ { t } > 0$ is a fixed weight for future time step t. The same values of ${ \widehat { \sigma } } _ { i } ^ { \mathrm { e r r } }$ and $\gamma _ { t }$ are used for the calibration, tuning, and test scores. In the experiments, we use uniform time weights, $\gamma _ { t } = 1$ . Thus, $s _ { j }$ is the largest standardized voltage-prediction error across all buses and future time steps for scenario $j$ and is dimensionless.

Let $s _ { ( 1 ) } \ \leq \ s _ { ( 2 ) } \ \leq \ \cdot \cdot \cdot \leq \ s _ { ( n _ { \mathrm { c a l } } ) }$ be the sorted calibration scores. For a target miscoverage level $\alpha .$ , define

$$
\begin{array} { r } { k _ { \alpha } : = \operatorname* { m i n } \left\{ n _ { \mathrm { c a l } } , \left\lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \right\rceil \right\} , } \\ { \hat { \tau } _ { 1 - \alpha } : = s _ { ( k _ { \alpha } ) } } \end{array}\tag{11}
$$

Here, ⌈a⌉ denotes the smallest integer greater than or equal to $a .$ For instance, $\alpha = 0 . 0 5$ corresponds to a nominal 95% trajectory-level coverage target. The conformal voltage interval is then

$$
\begin{array} { r } { \left| V _ { i , t } - \widehat { V } _ { i , t } \right| \leq \frac { \widehat \tau _ { 1 - \alpha } \widehat \sigma _ { i } ^ { \mathrm { e r r } } } { \gamma _ { t } } , \qquad i \in \mathcal { B } , \ t \in \mathcal { T } _ { \mathrm { f } } . } \end{array}\tag{12}
$$

Thus, the simulated trajectory is surrounded by a voltage interval learned from past simulation errors. Split conformal prediction guarantees the target coverage even with a finite calibration dataset, provided that the calibration and future errors are exchangeable, meaning that they follow the same underlying distribution. Here, the calibration errors are collected under the baseline control policy; replacing it with the candidate policy may change the closed-loop operating conditions and, consequently, the distribution of future simulation errors.

## C. Robust Enlargement for Policy-Induced Error Changes

The calibration scores defined in the previous section quantify simulation-to-reality error under baseline operation. Thus, the conformal interval defined above is reliable when future simulation errors behave like the calibration errors. However, changing the control policy at one or several inverters can additionally induce a policy-induced shift in the score distribution. As a result, the distribution of the score s may shift, and the historical simulation-error data may no longer be fully representative.

DR-CSS uses the calibration scores as historical examples of nominal-simulator error under baseline operation. Each score is a scalar: the largest weighted voltage prediction error over all buses and future time steps. Thus, DR-CSS does not model the full uncertain grid trajectory. Instead, it calibrates the distribution of this trajectory-level error score. After control policy replacement, these scores may shift or develop a heavier tail because the new control policy can move the grid into different operating regimes. DR-CSS protects against this effect by replacing the standard conformal threshold with a higher score quantile. Here, distributionally robust means robust to a bounded change in the scalar score distribution, not to a known full distribution of loads, PV generation, voltages, and control policy actions. The robust step keeps the same split-conformal calibration scores, but uses a higher quantile than standard split conformal prediction. The parameter $\rho$ controls this quantile inflation. Larger values of $\rho$ move the threshold to a higher calibration score and therefore produce a wider, more conservative voltage interval.

For the Total Variation (TV)-robust version, the inflated quantile level is

$$
\beta _ { \alpha , \rho } ^ { \mathrm { T V } } = \operatorname* { m i n } \left\{ 1 , 1 - \alpha + \frac { \rho } { 2 } \right\} .\tag{13}
$$

For a given robustness budget $\rho ,$ define

$$
k _ { \alpha , \rho } ^ { \mathrm { T V } } = \lceil ( n _ { \mathrm { c a l } } + 1 ) \beta _ { \alpha , \rho } ^ { \mathrm { T V } } \rceil .\tag{14}
$$

If $k _ { \alpha , \rho } ^ { \mathrm { T V } } > n _ { \mathrm { c a l } }$ , the calibration set does not support a finite robust threshold at the requested confidence level, and DR-CSS returns a warning. Otherwise, the robust threshold is

$$
\tau _ { \alpha } ^ { \mathrm { r o b } } ( \rho ) = s _ { ( k _ { \alpha , \rho } ^ { \mathrm { T V } } ) } .\tag{15}
$$

Here, $s _ { ( k ) }$ denotes the k-th order statistic, i.e., the k-th smallest score among the calibration scores $\{ s _ { j } \} _ { j = 1 } ^ { n _ { \mathrm { c a l } } }$ . The resulting robust voltage interval is

$$
\begin{array} { r } { \left| V _ { i , t } - \widehat { V } _ { i , t } \right| \leq \frac { \tau _ { \alpha } ^ { \mathrm { r o b } } ( \rho ) \widehat \sigma _ { i } ^ { \mathrm { e r r } } } { \gamma _ { t } } , \qquad i \in \mathcal { B } , t \in \mathcal { T } _ { \mathrm { f } } . } \end{array}\tag{16}
$$

Thus, $\rho$ does not directly modify the nominal simulated trajectory. Instead, it increases the score quantile used to construct the uncertainty band around that trajectory.

A small value of $\rho$ produces a relatively narrow interval but may provide insufficient protection against control policyinduced changes in simulation error. Conversely, a large value provides greater protection but may lead to unnecessarily conservative warning decisions. We therefore select $\rho$ before final testing using an independent tuning set generated under the candidate control policy $\pi ^ { \mathrm { d e p } }$

For a given robustness budget $\rho ,$ the empirical tuning coverage is

$$
\hat { c } _ { \mathrm { t u n } } ( \rho ) = \frac { 1 } { n _ { \mathrm { t u n } } } \sum _ { m = 1 } ^ { n _ { \mathrm { t u n } } } \mathbf { 1 } \{ s _ { m } ^ { \mathrm { n e w } } \leq \tau _ { \alpha } ^ { \mathrm { r o b } } ( \rho ) \} ,\tag{17}
$$

where $s _ { m } ^ { \mathrm { n e w } }$ is the simulation-error score for tuning case m under the new control policy. It is computed using (10), with the same fixed ${ \widehat { \sigma } } _ { i } ^ { \mathrm { e r r } }$ and $\gamma _ { t }$ used for calibration.

Because $\widehat { c } _ { \mathrm { t u n } } ( \rho )$ is estimated from finitely many tuning scenarios, it may overestimate the underlying coverage. We use the one-sided Wilson lower confidence bound

$$
L _ { \delta _ { \mathrm { t u n } } } ( \rho ) : = \mathrm { W L B } \left( \widehat { c } _ { \mathrm { t u n } } ( \rho ) , n _ { \mathrm { t u n } } , \delta _ { \mathrm { t u n } } \right) ,\tag{18}
$$

$$
\mathrm { W L B } ( \widehat { p } , n , \delta ) : = \operatorname* { m a x } \left\{ 0 , \frac { \widehat { p } + \frac { a } { 2 } - z \sqrt { \frac { \widehat { p } ( 1 - \widehat { p } ) + \frac { a } { 4 } } { n } } } { 1 + a } \right\} ,\tag{19}
$$

where $\begin{array} { r } { z = \Phi ^ { - 1 } ( 1 - \delta ) , a = \frac { z ^ { 2 } } { n } } \end{array}$ . and $\Phi$ is the standard normal cumulative distribution function.

The parameter $\delta _ { \mathrm { t u n } }$ specifies the error probability used to construct the tuning-coverage confidence bound; the corresponding confidence level is $1 - \delta _ { \mathrm { t u n } }$ . The lower confidence bound prevents the robustness budget from being selected solely on the basis of a potentially optimistic empirical coverage estimate. The selected robustness budget is the smallest value whose lower confidence bound reaches the target coverage:

![](images/025e2b2fd8de04610566ec39660c08cbc2b743715d739786c51833f2f065848f.jpg)

![](images/fa96bc376a0a355428fe55c00c3e2ca7c9a7598b1796675984169e5e00f0495a.jpg)  
Fig. 2. Robust enlargement of the voltage uncertainty interval. Panel (a) shows the historical voltage error scores. Panel (b) shows the effect on voltages: the robust interval is wider than the standard one under control policy replacement.

$$
\rho ^ { \star } = \operatorname* { i n f } \left\{ \rho \geq 0 : L _ { \delta _ { \mathrm { t u n } } } ( \rho ) \geq 1 - \alpha \right\} .\tag{20}
$$

Because the robust threshold is non-decreasing in $\rho ,$ the empirical tuning coverage and its corresponding Wilson lower bound are also non-decreasing. This monotonicity enables an efficient bisection search for the smallest robustness budget satisfying the tuning criterion. Selecting the smallest feasible value limits unnecessary interval enlargement and reduces overly conservative warning decisions.

Once selected, $\rho ^ { \star }$ is fixed before evaluation on the independent test set. It is then used to construct the final robust voltage intervals shown in Fig. 2. DR-CSS accepts the candidate deployment for an operating scenario only if the complete voltage band remains within the admissible limits over all buses and future time steps.

## D. Scenario-Level Accept/Warning Decision

After calibration, the final robust interval is checked against the voltage limits for every bus and every time step in the horizon. After selecting $\rho ^ { \star }$ , define $\tau ^ { \star } : = \tau _ { \alpha } ^ { \mathrm { r o b } } ( \rho ^ { \star } )$ , as shown in Fig. 1. For a tested scenario $x _ { \mathrm { n e w } }$ , DR-CSS accepts the candidate deployment only if the full robust interval lies inside the voltage limits:

$$
V _ { \operatorname* { m i n } } \leq \widehat { V } _ { i , t } - \frac { \tau ^ { \star } \widehat { \sigma } _ { i } ^ { \mathrm { e r r } } } { \gamma _ { t } } \quad \mathrm { a n d } \quad \widehat { V } _ { i , t } + \frac { \tau ^ { \star } \widehat { \sigma } _ { i } ^ { \mathrm { e r r } } } { \gamma _ { t } } \leq V _ { \operatorname* { m a x } } ,\tag{21}
$$

for every $i \in \boldsymbol { B }$ and $t \in \mathcal { T } _ { \mathrm { f } }$ . Under the stated assumptions, the robust interval provides the prescribed statistical coverage for the future voltage trajectory of the distribution grid after deployment of the candidate control policy, jointly over all buses and future time steps. Otherwise, DR-CSS issues a warning. A warning does not mean that a voltage violation will occur; it means that the nominal simulation, together with the calibration data and robustness budget, does not provide enough evidence to accept the deployment for that horizon.

## E. Context-Aware Calibration

The robust calibration introduced above uses a single threshold for all tested scenarios. However, the control policyinduced distribution shift depends on the operating condition, since changes in load demand and PV generation affect the inverter actions and the resulting voltage response. A global threshold may therefore be overly conservative in some conditions and insufficient in others. To address this, we extend our method with a context-aware calibration. Specifically, we divide the scenarios into operating groups and apply (10)-(20) separately within each group. Each scenario is assigned to a group using the grid net active power at the end of the observed history, $\begin{array} { r } { P _ { \mathrm { n e t } } = \sum _ { i \in \mathcal { L } } P _ { i } - \sum _ { j \in \mathbb { Z } _ { \mathrm { i n v } } } P _ { j } } \end{array}$ . Negative values indicate PV-dominated operation, whereas larger values indicate higher net demand. The scenarios are divided into bins using empirical quantiles of $P _ { \mathrm { n e t } }$ , and a separate robust threshold is calibrated for each bin. At test time, the threshold corresponding to the observed operating condition is used, while the accept/warning rule remains unchanged. Net active power provides a simple measure of the balance between load demand and PV generation. Other variables, such as recent voltages, forecasts, or control policy-action differences, could also be used to group the scenarios; identifying the most informative context variable is left for future work.

![](images/c7125c2ddd9f65831702d62cbc6d3c2b44433e16bc2da48a7db6875328a397bf.jpg)  
Fig. 3. IEEE 33-bus feeder and the observation zones used by the IDDPG actors.

## IV. EXPERIMENTAL RESULTS

This section evaluates DR-CSS on the IEEE 33-bus and IEEE 141-bus systems using droop control as the baseline and learning-based policies as the new candidate policies, within single-control policy, context-aware, and gradual replacement experiments.

## A. Experimental Setup

We conduct our experiments using the open-source Multi-Agent Power Distribution Network (MAPDN) environment [7]. MAPDN provides PV generation and residential load profiles, obtained from publicly available real-world datasets. The profiles are resampled at a uniform temporal resolution of three minutes, which is represented in the environment by the normalized time step $\Delta t = 1$ . Each daily PV and load profile defines an operating scenario. Collectively, these scenarios cover a broad range of generation and demand conditions. These profiles are assigned to two radial distribution-grid benchmarks: the IEEE 33-bus and IEEE 141-bus grids. The IEEE 33-bus and IEEE 141-bus radial grids contain 6 and 38 PV-equipped inverter buses, respectively. Fig. 3 shows the IEEE 33-bus grid. In both systems, high PV penetration can cause voltage-limit violations, motivating active voltage control.

For both grids, the admissible voltage range is set to $[ V _ { \mathrm { m i n } } , V _ { \mathrm { m a x } } ] \stackrel { - } { = } [ 0 . 9 5 , 1 . 0 5 ] \mathrm { p . } \imath$ u. The reactive-power capability of inverter i is constrained by its apparent-power rating, as defined in (1d). Consequently, the feasible reactive-power range changes dynamically with the local active-power injection $P _ { i , t } .$ In this work we only consider reactive power control and assume that active power is an exogenous variable determined by the PV generation.

Trajectory windows are extracted from the daily PV and load profiles, resulting in distinct voltage trajectories. The resulting trajectory windows are divided into disjoint training, validation, calibration, and test sets 2166, 334, 1500, and 1000 samples, respectively. These sets are used for predictor training, hyperparameter tuning, conformal quantile estimation, and final evaluation, respectively. Each trajectory window consists of a 21-step observation history and a 10-step prediction horizon, corresponding to 63 minutes of history and a 30-minute look-ahead, at a resolution of 3 minutes. The look-ahead horizon is chosen based on the timescale of PV generation variability.

1) Control Policies used in Experiments: We consider two inverter-control configurations, a baseline droop control policy and a target RL control policy provided by the MAPDN benchmark [7].

• Droop control policy: This policy implements a standard piecewise-linear Volt-Var droop characteristic with a deadband. Reactive power is saturated outside the outer voltage limits, varies linearly in the droop regions, and is zero inside the deadband. Let $V _ { \mathrm { d b } } ^ { - } ~ = ~ 0 . 9 7$ p.u. and $V _ { \mathrm { d b } } ^ { + } = 1 . 0 3$ p.u. denote the lower and upper deadband thresholds, the control rule can be defined as:

$$
\begin{array} { r } { Q _ { i , t } = \left\{ \begin{array} { l l } { Q _ { i } ^ { \mathrm { m i n } } , } & { V _ { i , t } \le V _ { \mathrm { m i n } } , } \\ { Q _ { i } ^ { \mathrm { m i n } } \frac { V _ { \mathrm { d b } } ^ { - } - V _ { i , t } } { V _ { \mathrm { d b } } ^ { - } - V _ { \mathrm { m i n } } } , } & { V _ { \mathrm { m i n } } < V _ { i , t } < V _ { \mathrm { d b } } ^ { - } , } \\ { 0 , } & { V _ { \mathrm { d b } } ^ { - } \le V _ { i , t } \le V _ { \mathrm { d b } } ^ { + } , } \\ { Q _ { i } ^ { \mathrm { m a x } } \frac { V _ { i , t } - V _ { \mathrm { d b } } ^ { + } } { V _ { \mathrm { m a x } } - V _ { \mathrm { d b } } ^ { + } } , } & { V _ { \mathrm { d b } } ^ { + } < V _ { i , t } < V _ { \mathrm { m a x } } , } \\ { Q _ { i } ^ { \mathrm { m a x } } , } & { V _ { i , t } \ge V _ { \mathrm { m a x } } . } \end{array} \right. } \end{array}\tag{22}
$$

Because the droop control depends only on local measurements, its neighborhood is $\mathcal { N } _ { i } = \{ i \}$ , and no communication with neighboring buses is required.

• MARL control policy: The second configuration employs the Independent Deep Deterministic Policy Gradient (IDDPG) algorithm implemented in the MAPDN framework [7]. In this configuration, a subset of inverters $\mathcal { E } \subseteq \mathbb { Z } _ { \mathrm { i n v } }$ is controlled by individual trained IDDPG actors. Each actor observes the voltage magnitudes and active-power injections of the inverter buses in its assigned zone $\mathcal { N } _ { i } ^ { * }$ . The zones for the IEEE 33-bus case are shown in Fig. 3. We adopt the same zone definition as in the MAPDN framework. Each actor outputs a continuous reactive-power setpoint $Q _ { i ^ { * } } ( t ) = \pi _ { i ^ { * } } ^ { \mathrm { R L } } ( o _ { i ^ { * } } ( t ) ) \in$ $[ Q _ { i ^ { * } } ^ { \operatorname* { m i n } } , Q _ { i ^ { * } } ^ { \operatorname* { m a x } } ]$ . All other inverters $i \in \mathcal { T } _ { \mathrm { i n v } } \setminus \mathcal { E }$ continue to follow the droop control in (22). The IDDPG agents are trained using the L1-shaped voltage-barrier reward introduced in [7], and their actor and critic networks are trained in the PandaPower-based MAPDN environment using the training scenarios, while the droop-controlled buses remain fixed. After training, the RL actors, droop control policies, and simulator are fixed; only the conformal calibration and robustness budget are estimated from the calibration and tuning data.

TABLE I  
AVERAGE PERFORMANCE ON THE IEEE 33-BUS AND IEEE 141-BUS CASES.
<table><tr><td>System</td><td>Method</td><td>Cov. 95%</td><td>Avg. width [p.u.]</td><td>Max. width [p.u.]</td><td>Unbd. rate</td></tr><tr><td rowspan="3">IEEE 33</td><td>CP</td><td>0.946</td><td>0.0237</td><td>0.0498</td><td></td></tr><tr><td>Weighted</td><td>0.999</td><td>0.0791</td><td>0.159</td><td>1.000</td></tr><tr><td>DR-CSS</td><td>0.955</td><td>0.0249</td><td>0.0511</td><td>0.000</td></tr><tr><td rowspan="3">IEEE 141</td><td>CP</td><td>0.945</td><td>0.040</td><td>0.054</td><td></td></tr><tr><td>Weighted</td><td>1.000</td><td>0.114</td><td>0.157</td><td>1.000</td></tr><tr><td>DR-CSS</td><td>0.965</td><td>0.042</td><td>0.058</td><td>0.000</td></tr></table>

2) Benchmark Methods: To demonstrate the efficiency of the proposed approach, we compare it to two benchmark methods. The compared methods use the same simulator $\Phi _ { \mathcal { M } }$ We set the conformal miscoverage level to $\alpha _ { \mathrm { C P } } = 0 . 0 5$ , giving 95% nominal coverage.

• Standard conformal prediction (Standard CP): We apply split conformal prediction using the all-droop calibration set $\mathcal { D } _ { \mathrm { c a l } }$ . The threshold $\hat { \tau } _ { 0 . 9 5 }$ is obtained from the finitesample-adjusted empirical quantile of the calibration scores $\{ s _ { j } \}$

• Weighted MA-COPP: This state-of-the-art baseline applies weighted conformal prediction [28] in the -agent - agent off-policy setting proposed in [26]. Each calibration score is assigned the importance weight $\begin{array} { r } { w _ { j } = \frac { p _ { \mathrm { t e s t } } ( x _ { j } ) } { p _ { \mathrm { c a l } } ( x _ { j } ) } } \end{array}$ The density ratio is estimated by logistic regression using the terminal-history voltage and active-power features $( V _ { i } ( H ) , P _ { i } ( H ) ) _ { i \in { \cal B } }$ . Under poor distributional overlap, large importance weights may cause the weighted critical value to be $+ \infty ,$ yielding uninformative and unbounded intervals. As a fallback in such cases, we replace +∞ with the largest admissible scalar score when constructing the voltage intervals. In this case, the original MA-COPP coverage guarantee no longer applies.

## B. Numerical Results

For both IEEE systems, the following results consider a single-control policy replacement: one inverter uses the RL policy, while all others employ the droop control policy.

1) Overall Performance: Table I compares the empirical joint trajectory coverage, average and maximum voltageinterval widths, and unbounded-interval rate of the three methods. Standard CP attains an empirical coverage of 0.946 and 0.945 on the IEEE 33 and IEEE 141, respectively, slightly below the nominal level $1 - \alpha \ = \ 0 . 9 5$ . This undercoverage is consistent with a control policy-induced shift in the nonconformity-score distribution: calibration errors are generated under the all-droop control policy, whereas test errors arise after one inverter is replaced by the RL control policy. Weighted MA-COPP achieves near-perfect empirical coverage on both systems, but all critical values are infinite, yielding unbounded intervals rate of 100%. We therefore report voltage interval widths obtained by capping the critical values at a finite scalar threshold. Thus, the MA-COPP coverage guarantee does not hold for the reported regions. In contrast, DR-CSS recovers the nominal coverage level with no unbounded intervals, while keeping the average interval width only 3.0% and 5.0% larger than Standard CP on the IEEE 33-bus and IEEE 141-bus systems, respectively. Compared with Weighted MA-COPP, the proposed robust approach reduces the average interval width by approximately 69% and 63% on the two systems. Overall, DR-CSS compensates for the policy-induced shift with a modest efficiency cost, while the standard CP does not achieve the required coverage and MA-COPP does not provide the aimed guarantees. Fig. 4 illustrates this effect for a representative voltage trajectory, where the conformal interval surrounds the nominal prediction over the forecast horizon and contains the realized future voltage.

![](images/a256a539764069485c99b38abfbeaa995a2d1de87ccfe52bc70a0920ef709743.jpg)

Fig. 4. Voltage prediction with conformal intervals for one agent in the IEEE141 grid.  
![](images/9bdb584550b9d38b0babae71dd0dcc822db43639b454fc42bcfec52a4e8ddd0d.jpg)  
Fig. 5. Mean interval width for each bus in the IEEE33 case.

To examine the spatial effect, Fig. 5 shows the mean prediction-interval width for each bus in the IEEE 33-bus system. A clear spatial pattern emerges: prediction intervals become wider as the distance from the substation increases. Buses located near the beginning of a feeder have limited influence on the local voltage because the substation has a dominant effect. Consequently, the baseline and new control policies induce similar voltage distributions, resulting in only a small distribution shift. Toward the end of the feeders, however, control policy actions have a greater influence on voltage. The replacement of the control policy, therefore, produces a more pronounced distribution shift, requiring wider prediction intervals. This observation highlights the importance of context: as discussed in Section III-E, uncertainty depends on the spatial locations and the operating regime.

![](images/e3d441f2db07c0afb01ea1097b72f2f39afe3e46d7c3c72a516552306d04471c.jpg)  
Fig. 6. Prediction-interval width in each load bin for uniform and loadwise robust conformal prediction.

2) Context-Aware Calibration: This section evaluates the context-aware calibration described in Section III-E. We consider the IEEE 33-bus system and replace the droop control policy at bus 25 with the RL control policy. This corresponds to a control policy change at one inverter. The scenarios are divided into five bins using empirical quantiles of $P _ { \mathrm { n e t } }$ in the shifted simulation set, yielding 400 simulated trajectories per bin.

Table II reports the empirical coverage of standard CP, uniform DR-CSS with a single context-independent threshold, and loadwise DR-CSS with loadwise calibration. The results reveal high variation across operating regimes. For standard CP and uniform DR-CSS, the two most PV-dominated bins (bins 1 and 2) attain the target coverage of $1 - \alpha = 0 . 9 5 .$ whereas the other bins fall below the target, with the largest undercoverage occurring in bin 4. The robust correction primarily accounts for the distribution shift induced by the change in control policy. Because control policy actions have a stronger effect on voltage during PV-dominated operating conditions, this correction may be more conservative in those regimes. In load-dominated conditions, however, the remaining uncertainty may be governed by operating-regime effects not fully captured by the control policy-shift correction, leading to larger conditional nonconformity scores and undercoverage. More generally, neither standard CP nor uniform DR-CSS guarantees coverage within each load bin: attaining the target marginally over the full dataset does not imply valid coverage within individual subgroups of the data. The smaller number of observations within each bin may also contribute to the observed variability.

Loadwise calibration, on the other hand, accounts for this regime dependence and attains the coverage target in every bin. This improvement comes at the cost of moderately wider intervals, as shown in Fig. 6. The increase is most pronounced in bin 4, where the uniform methods exhibit the greatest undercoverage, indicating that this regime requires a larger conditional critical value.

3) Abrupt and Gradual Control Policy Replacement: The previous experiments considered the replacement of the control policy at a single inverter bus. In practice, however, the control policy at multiple inverters may be updated during deployment. When several control policies are replaced at once, the shift in the score distribution can become larger. This requires a stronger robustness correction, which makes the screening more conservative and can increase the number of warning decisions. We therefore propose and evaluate a gradual deployment strategy in which control policies are introduced one at a time. After each step, new data under the new joint policy are collected and the calibration and tuning procedure is repeated. This reduces the shift handled at each stage and can lower unnecessary warnings while preserving the same accept/warning rule.

TABLE II  
COMPARISON OF THE EMPIRICAL COVERAGE OF STANDARD CP, UNIFORM DR-CSS AND LOADWISE DR-CSS ACROSS OPERATING-POINT BINS DEFINED BY $P _ { \mathrm { n e t } }$
<table><tr><td>Bin Index</td><td> $P _ { \mathrm { n e t } }$  range [p.u.]</td><td>Standard CP Cov.</td><td>Uniform DR-CSS Cov.</td><td>Loadwise DR-CSS Cov.</td></tr><tr><td>1</td><td> $[ - 5 . 8 6 , - 3 . 3 2 )$ </td><td>0.980</td><td>0.985</td><td>0.995</td></tr><tr><td>2</td><td> $\bar { [ - 3 . 3 2 , - 2 . 4 8 ) }$ </td><td>0.962</td><td>0.962</td><td>0.967</td></tr><tr><td>3</td><td> $- 2 . 4 8 , - 1 . 6 8 { \dot { ) } }$ </td><td>0.936</td><td>0.948</td><td>0.953</td></tr><tr><td>4</td><td> $\dot { \cdot } - 1 . 6 8 , - 0 . 6 2 )$ </td><td>0.921</td><td>0.942</td><td>0.995</td></tr><tr><td>5</td><td> $[ - 0 . 6 2 , 1 . 9 6 ]$ </td><td>0.926</td><td>0.944</td><td>0.974</td></tr></table>

![](images/ca7e2137797a15bf2fde16286c712abed3eee850607bfb6ee382c88edc6d5af9.jpg)  
(a) Threshold

![](images/bd5ea284580818679a970cd6e6c7d3aca03b1905e70914eb92846930606d4d05.jpg)  
(b) Coverage–width  
Fig. 7. Comparison of abrupt and gradual distribution shifts. Open markers in panel (a) denote the standard CP threshold and filled markers the robust threshold; panel (b) reports the resulting empirical coverage and mean interval width.

The previous experiments considered a single policy change at bus 25. Now, we consider the control policy replacement at buses 25, 29, and 13. In the abrupt protocol, all three inverters switch to RL control simultaneously, so calibration starts from the all-droop control and targets the three-RL-control configuration (3RL). In the gradual strategy, inverters are replaced sequentially $2 5 \to 2 9 \to 1 3 ;$ after each replacement, new trajectories are collected and the conformal method is recalibrated. For comparability, both strategies are evaluated on the same real 3RL deployment dataset.

Figures 7 (a)-(b) show that the proposed gradual deployment strategy reduces conservativeness while maintaining empirical coverage above the nominal target under the same final 3RL deployment. In Fig. 7(a), intermediate recalibration reduces the required robustness correction by approximately 59% compared with abrupt replacement, indicating a smaller effective distribution shift. In Fig. 7(b), both strategies exceed the nominal 95% coverage level, but gradual deployment achieves this with approximately 36% lower conservativeness. Overall, the results show that replacing control policies incrementally can preserve coverage while avoiding the over-conservatism caused by protecting against an abrupt one-step control policy change.

![](images/fdacc165edbede49d1054cc2fe0e909faa943a30defea74e5847a1683b0d6f39.jpg)  
(a) Abrupt shift: droop to 3RL

![](images/a3d6e42954ed13d94cceb962a78ab26a14806628bad87a90ed87d18fd7713144.jpg)  
(b) Gradual shift: 2RL to 3RL  
Fig. 8. Source-calibration and 3RL-target score distributions. Dotted and solid lines mark τ<sub>CP</sub> and τ<sup>∗</sup>, respectively. Target miss rates are 7.36% (abrupt) and 5.62% (gradual).

TABLE III  
WARNING PERFORMANCE UNDER GRADUAL CONTROL POLICY SHIFT. POINT-LEVEL METRICS ARE COMPUTED OVER INDIVIDUAL TIME POINTS WITHIN THE PREDICTION HORIZON. TRAJECTORY-LEVEL METRICS INDICATE WHETHER ANY TIME POINT WITHIN THE HORIZON IS VIOLATING OR UNCERTAIN.
<table><tr><td>Metric</td><td>Point level</td><td>Trajectory level</td></tr><tr><td>Violation rate</td><td>0.0156</td><td>0.4200</td></tr><tr><td>Alarm rate</td><td>0.0965</td><td>0.7710</td></tr><tr><td>Recall</td><td>0.9965</td><td>1.0000</td></tr><tr><td>Precision of alarms</td><td>0.1610</td><td>0.5447</td></tr></table>

Fig. 8 compares the calibration and 3RL-target score distributions. The abrupt transition produces a more pronounced discrepancy in the upper tail: 7.36% of the abrupt-transition test scores exceed the standard conformal threshold, compared with 5.62% under gradual replacement. The latter is much closer to the nominal 5% miscoverage level. Overall, the results show that sequential data collection and recalibration reduce the effective shift, resulting in a smaller robust correction and tighter prediction intervals while maintaining empirical coverage above the nominal target

Table III reports the warning performance under gradual control policy shift. A point-level alarm occurs when the prediction interval at a given bus and time intersects or exceeds a voltage limit. A trajectory-level alarm occurs when at least one point-level alarm is raised within the horizon. Recall is the fraction of true violations that are alarmed, while precision is the fraction of alarms that correspond to true violations. Voltage violations are rare at the point level, with a violation rate of 0.0156, but the method detects almost all of them, achieving recall 0.9965. This comes with conservative alarms: 9.65% of point-level checks are flagged, and the corresponding precision is 0.1610. At the trajectory level, the method flags 77.10% of rollouts and achieves recall 1.0000, meaning that every rollout with at least one violation is detected. The trajectory-level precision is 0.5447, so more than half of the alarmed rollouts contain a true violation. Overall, the screening prioritizes avoiding missed voltage violations under control policy shift, at the cost of some conservative alarms.

## V. CONCLUSION

This paper addresses a critical question: can we develop an efficient safety screening framework for new voltage control policies under imperfect simulations and historical data collected under another control policy? We solve this by proposing DR-CSS, which integrates nominal simulations with distributionally robust conformal screening to systematically bound both simulation-to-reality and policy-induced distribution shifts. Validated across multiple operational scenarios and distribution grid topologies, this framework demonstrates great potential, consistently maintaining tight, tight safety bounds where traditional importance-weighted baselines result in unbounded intervals. Ultimately, by uniting this rigorous methodological backbone with practical deployment policies, specifically a staged gradual replacement strategy, DR-CSS guarantees that safety screening of advanced data-driven control remains efficient and informative.

The results suggest several promising directions for future refinement, spanning both methodological enhancements and broader power system applications. One possible direction is to advance the underlying framework to continuous conformal risk control, shifting from a binary accept-or-reject threshold to the mathematical optimization and minimization of the expected physical severity or duration of minor overvoltages. In terms of application, DR-CSS can be expanded to evaluate multi-layered grid architectures where fast device controls and slow market scheduling interact across different time scales, as well as partially observable grids where operators must rely on imperfect state estimations.

During the preparation of this work, the authors used ChatGPT and Claude to improve readability and assist with grammar and phrasing. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## REFERENCES

[1] W. Murray, M. Adonis, and A. Raji, “Voltage control in future electrical distribution networks,” Renewable and Sustainable Energy Reviews, vol. 146, p. 111100, 2021.

[2] W. Cui, J. Li, and B. Zhang, “Decentralized safe reinforcement learning for inverter-based voltage control,” Electric Power Systems Research, vol. 211, p. 108609, 2022. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S037877962200685X

[3] Y. Zhang, X. Wang, J. Wang, and Y. Zhang, “Deep reinforcement learning based volt-var optimization in smart distribution systems,” IEEE Trans. Smart Grid, vol. 12, no. 1, pp. 361–371, 2021.

[4] W. Wang, N. Yu, Y. Gao, and J. Shi, “Safe off-policy deep reinforcement learning algorithm for volt-var control in power distribution systems,” IEEE Transactions on Smart Grid, vol. 11, no. 4, pp. 3008–3018, 2020.

[5] Y. Gao, W. Wang, and N. Yu, “Consensus multi-agent reinforcement learning for volt-var control in power distribution networks,” IEEE Transactions on Smart Grid, vol. 12, no. 4, pp. 3594–3604, 2021.

[6] F. Kabir, N. Yu, and Y. Gao, “Deep reinforcement learning-based twotimescale volt-var control with degradation-aware smart inverters in power distribution systems,” Applied Energy, vol. 335, p. 120629, 2023.

[7] J. Wang, W. Xu, Y. Gu, W. Song, and T. C. Green, “Multi-agent reinforcement learning for active voltage control on power distribution networks,” ser. NIPS ’21. Red Hook, NY, USA: Curran Associates Inc., 2021.

[8] S. Bolognani, R. Carli, G. Cavraro, and S. Zampieri, “On the need for communication for voltage regulation of power distribution grids,” IEEE Control Netw. Syst., vol. 6, no. 3, pp. 1111–1123, 2019.

[9] Y. Gao and N. Yu, “Model-augmented safe reinforcement learning for volt-var control in power distribution networks,” Applied Energy, vol. 313, p. 118762, 2022.

[10] H. T. Nguyen and D.-H. Choi, “Three-stage inverter-based peak shaving and volt-var control in active distribution networks using online safe deep reinforcement learning,” IEEE Trans. Smart Grid, vol. 13, no. 4, pp. 3266–3277, 2022.

[11] F. Geth, M. Vanin, W. V. Westering, T. Milford, and A. Pandey, “Making distribution state estimation practical: Challenges and opportunities,” arXiv preprint arXiv:2311.07021, 2023.

[12] K. Parlaktuna, E. Dursun, and M. Gol, “Wlav state estimation based¨ topology error detection and identification in distribution networks with limited number of measurements,” Electric Power Systems Research, vol. 232, p. 110406, 2024.

[13] Z. Wang, S. Q. Bu, J. Wen, and C. Huang, “A comprehensive review on uncertainty modeling methods in modern power systems,” International Journal of Electrical Power & Energy Systems, vol. 166, p. 110534, 2025.

[14] Y. Wang, M. H. Syed, E. Guillo-Sansano, Y. Xu, and G. M. Burt, “Inverter-based voltage control of distribution networks: A three-level coordinated method and power hardware-in-the-loop validation,” IEEE Trans. Sustain. Energy, vol. 11, no. 4, pp. 2380–2391, 2020.

[15] M. Otte, F. Leimgruber, R. Brundlinger et al., “Hardware-in-the-loop co-simulation based validation of power system control applications,” in 2018 IEEE 27th International Symposium on Industrial Electronics (ISIE), 2018, pp. 1229–1234.

[16] A. Paspatis, A. Kontou, P. Kotsampopoulos et al., “Advanced hardwarein-the-loop testing chain for investigating interactions between smart grid components during transients,” Electric Power Systems Research, vol. 228, p. 109990, 2024.

[17] H.-D. Tran, W. Xiang, and T. T. Johnson, “Verification approaches for learning-enabled autonomous cyber-physical systems,” IEEE Design & Test, vol. 39, no. 1, pp. 24–34, 2022.

[18] M. Everett, “Neural network verification in control,” in 2021 60th IEEE Conference on Decision and Control (CDC), 2021, pp. 6326–6340.

[19] N. Kochdumper, C. Schilling, M. Althoff, and S. Bak, “Open- and closed-loop neural network verification using polynomial zonotopes,” in NASA Formal Methods, ser. Lecture Notes in Computer Science, vol. 13903. Springer, 2023, pp. 16–36.

[20] C. Dawson, S. Gao, and C. Fan, “Safe control with learned certificates: A survey of neural lyapunov, barrier, and contraction methods for robotics and control,” IEEE Trans. Robot., vol. 39, no. 3, pp. 1749–1767, 2023.

[21] P. Pareek, S. Misra, and D. Deka, “Learning power flow with confidence: A probabilistic guarantee framework for voltage risk,” 2025. [Online]. Available: https://arxiv.org/abs/2308.07867

[22] A. Mollaali, G. Zufferey, G. Constante-Flores, C. Moya, C. Li, M. Yue, and G. Lin, “Conformalized prediction of post-fault voltage trajectories using pre-trained and finetuned attention-driven neural operators,” Neural Networks, vol. 192, p. 107809, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0893608025006896

[23] Y. Renkema, N. Brinkel, and T. Alskaif, “Conformal prediction for stochastic decision-making of pv power in electricity markets,” Electric Power Systems Research, vol. 234, p. 110750, 2024.

[24] Y. Renkema, L. Visser, and T. AlSkaif, “Enhancing the reliability of probabilistic pv power forecasts using conformal prediction,” Solar Energy Advances, vol. 4, p. 100059, 2024.

[25] M. F. Taufiq, J.-F. Ton, R. Cornish, Y. W. Teh, and A. Doucet, “Conformal off-policy prediction in contextual bandits,” arXiv preprint arXiv:2206.04405, 2022.

[26] T. Kuipers, R. Tumu, S. Yang, M. Kazemi Mehrabadi, R. Mangharam, and N. Paoletti, “Conformal off-policy prediction for multi-agent systems,” in 2024 IEEE 63rd Conference on Decision and Control (CDC), 2024, pp. 1067–1074.

[27] A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal prediction and distribution-free uncertainty quantification,” arXiv preprint arXiv:2107.07511, 2022.

[28] R. J. Tibshirani, R. F. Barber, E. J. Candes, and A. Ramdas, “Conformal prediction under covariate shift,” arXiv preprint arXiv:1904.06019, 2020.