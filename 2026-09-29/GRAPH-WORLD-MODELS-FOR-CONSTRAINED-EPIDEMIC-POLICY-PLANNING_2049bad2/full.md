# GRAPH WORLD MODELS FOR CONSTRAINED EPIDEMIC POLICY PLANNING

Yiqi Su<sup>1</sup> Rashed Shelim<sup>1,2</sup> Lingyi Wang<sup>2</sup> Walid Saad<sup>2</sup> Naren Ramakrishnan<sup>1</sup>

<sup>1</sup>Department of Computer Science, Virginia Tech, Alexandria, VA 22305, USA

<sup>2</sup>Department of Electrical and Computer Engineering, Virginia Tech, Alexandria, VA 22305, USA

## ABSTRACT

Epidemic policy planning often requires coordination between geographical regions, taking into account mobility-driven spillovers and how to make use of limited resources. Existing methods either lack action-conditioned models of coupled dynamics or cannot guarantee per-period feasibility. We present EpiMind, a graph world model framework for constrained epidemic policy planning across regions. A graph-factored recurrent state-space model generates joint policy-conditioned rollouts from regional latent beliefs, while graph-temporal ADMM optimizes regional interventions, enforces shared-resource feasibility through projection, and evaluates temporal specifications under the learned model. EpiMind reduces admission RMSE by 29% relative to graph-free dynamics modeling, plans within 1–5% of the best feasible constant policy with guaranteed shared-budget feasibility, and outperforms all deployable baselines across three resource budgets in real-context evaluation. These results demonstrate that graph-structured policy imagination with explicit constrained coordination supports effective epidemic interventions from learned dynamics.

## 1 INTRODUCTION

The effective control of epidemics requires coordinating non-pharmaceutical, pharmaceutical, and surveillance interventions across multiple regions Kraemer et al. (2020); Ferretti et al. (2020); Hsiang et al. (2020). The COVID-19 epidemic demonstrated how a patchwork of responses created uncertainty, chaos, and ultimately lack of trust in public health authorities Birkland et al. (2021); SteelFisher et al. (2023). Jurisdictions act on observable local conditions but remain coupled through mobility and competition for finite vaccines, hospital capacity, and budgets Emanuel et al. (2020).

Coordinating policy making across regions is difficult, especially under uncertainty and shared resource constraints Kermack & McKendrick (1927); Balcan et al. (2010). A given surveillance trend may reflect transmission changes, altered testing, or voluntary behavioral adaptation across regions. A useful planning model must therefore infer latent epidemic conditions from noisy, delayed, policydependent observations and predict their graph-coupled evolution under alternative actions.

Existing methods address only a fragment of this problem. Compartmental models Kermack & McKendrick (1927) and their metapopulation extensions Arino & van den Driessche (2003); Balcan et al. (2010) simulate forward trajectories by modifying mechanistic parameters such as transmission rates or contact matrices. They treat policies as exogenous inputs rather than decision vari ables. Further, they assume reported cases are direct measurements of true incidence rather than policy-dependent surveillance signals. Graph neural network (GNN)-based spatiotemporal forecasters such as Cola-GNN Deng et al. (2020) and county-level COVID-19 predictors Kapoor et al. (2020) learn flexible dynamics from time series but have no action space; thus, they cannot distinguish whether a forecasted decline reflects an intervention’s effect or a confounder such as reduced testing. Reinforcement learning (RL) for epidemic control Kompella et al. (2020); Ohi et al. (2020); Bushaj et al. (2023) reframes the problem as sequential decision-making within a calibrated simulator but typically optimizes a single composite agent over a concatenated multi-region state, with resource limits absorbed into shaped rewards that provide no feasibility guarantee for hard joint constraints such as total vaccine supply summed across regions. Multi-agent RL approaches that extend MAPPO Yu et al. (2022) to regional epidemic control Nayak et al. (2023) train per-region policies under a shared critic, but cross-agent sum constraints are typically incorporated as reward penalties or adaptive Lagrange multipliers, which guarantee only expected feasibility at convergence rather than per-timestep feasibility for a planner facing fixed inventory, and require a high-fidelity simulator that is unavailable for active outbreak response. Mathematical programming methods such as mixed-integer optimization for vaccination facility location Bertsimas et al. (2022) and vaccine supply-chain optimization Duijzer et al. (2018) enforce hard constraints exactly through branchand-cut, but require the transmission dynamics (case trajectory, susceptible fraction, reproduction number) to be supplied as a pre-fit input from a separately calibrated SEIR model, so the optimizer cannot adapt as decisions, behavioral response, or variant emergence shifts the trajectory.

World models Hafner et al. (2023); Wang et al. (2025); Memon et al. (2026) address several of these limitations by learning latent transition and observation models that can be rolled forward under candidate action sequences without further interaction with the environment. Conditioning the dynamics on actions allows the model to represent policy-dependent evolution and surveillance. It does not, however, identify causal intervention effects under endogenous historical policies, where regions may adopt stronger interventions precisely when outbreaks worsen. Graph world models (GWMs) Feng et al. (2025) extend recurrent state-space models (RSSMs) Hafner et al. (2023) by representing interacting entities as nodes that exchange information through message passing, providing an appropriate inductive bias for mobility-coupled epidemics. Existing world-model methods nevertheless provide limited machinery for coordinating distinct regional actions under per-period shared-resource constraints, and this is a gap we propose to address. We introduce EpiMind, a graph world-model planning framework whose key contributions are:

• EpiMind helps formulate multi-region epidemic planning on a dynamic policy graph, coupling regional dynamics through mobility and finite shared resources.

• EpiMind couples a parameter-shared GWM with a constrained multi-region planner for joint policy rollout, resource-feasible allocation, and temporal-logic evaluation.

• EpiMind achieves near-oracle performance in real-context evaluation, reduces admissions by 53.4% relative to no intervention while satisfying all shared-resource constraints.

## 2 METHOD

## 2.1 PROBLEM FORMULATION

Epidemics on graphs. We formalize the regional epidemic by a time-varying policy graph $G _ { t } =$ $( V _ { t } , E _ { t } )$ (see details in Section A), where nodes $V _ { t }$ represent regions and edges $E _ { t }$ capture mobility and coordination constraints at time t. This formulation makes the planning problem tractable by aligning each computational ingredient with the epidemic’s physical structure.

Graph-structured latent dynamics. Let $\boldsymbol { x } _ { t } ^ { i }$ denote region i’s unobserved epidemic state, and $\mathbf { x } _ { t } = \overline { { \left( x _ { t } ^ { 1 } , \ldots , x _ { t } ^ { N } \right) } }$ ) denote the joint state. Each region selects a D-dimensional intervention vector:

$$
a _ { t } ^ { i } \in \mathcal { A } ^ { i } \subseteq [ 0 , 1 ] ^ { D } , \qquad \mathbf { a } _ { t } = ( a _ { t } ^ { 1 } , \dots , a _ { t } ^ { N } ) ,\tag{1}
$$

The action comprises local non-pharmaceutical intervention (NPI) intensity and regional allocations of shared vaccine, hospitalization capacity, and fiscal-resource budget. The joint transition is policyconditioned and graph-coupled:

$$
p _ { \theta } \bigl ( \mathbf { x } _ { t + 1 } \mid \mathbf { x } _ { t } , \mathbf { a } _ { t } , G _ { t } \bigr ) = \prod _ { i = 1 } ^ { N } p _ { \theta } \Bigl ( x _ { t + 1 } ^ { i } \Big | x _ { t } ^ { i } , a _ { t } ^ { i } , \mathrm { A g g } _ { j \in \mathcal { N } _ { t } ( i ) } \Bigl ( E _ { t } ^ { i j } , x _ { t } ^ { j } , a _ { t } ^ { j } \Bigr ) \Bigr ) .\tag{2}
$$

Here $\mathcal { N } _ { t } ( i )$ is region $i \mathrm { \ ' } _ { \mathrm { s } }$ mobility neighborhood, and $\theta$ is shared across regions. Regional states, actions, observations, and neighborhoods remain distinct. The latent state is not observed directly. Instead, region i receives a surveillance observation, e.g., infections, hospital admissions, or deaths.

$$
o _ { t } ^ { i } \sim \Omega _ { \theta } ^ { i } \big ( \cdot \mid x _ { t } ^ { i } , \mathbf { a } _ { t - 1 } \big ) ,\tag{3}
$$

Candidate policies are evaluated over a multi-step horizon rather than through one-step prediction alone to response to the delayed surveillance and intervention effects.

![](images/af1badb43eb8c8dc00e89d6959383d6f77e76fab4876c00d9c8bcabffd9a8184.jpg)  
Figure 1: EpiMind framework. A parameter-shared GF-RSSM updates regional beliefs and generates graph-coupled policy rollouts. GT-ADMM coordinates and projects regional actions, rerolls the feasible allocation for model-relative STL evaluation, and executes its first action.

Constrained policy planning. At decision epoch t, the planner evaluates a candidate sequence of joint regional actions $\mathbf { a } _ { t : t + H - 1 }$ over horizon $\bar { H }$ . Starting from the current regional beliefs, a learned dynamics model $M _ { \theta }$ generates the joint policy-conditioned rollout:

$$
\begin{array} { r } { \hat { \tau } _ { t + 1 : t + H } ^ { 1 : N } = M _ { \theta } \left( b _ { t } ^ { 1 : N } , z _ { t } ^ { 1 : N } , \mathbf { a } _ { t : t + H - 1 } , G _ { t } \right) . } \end{array}\tag{4}
$$

These rollouts support model-dependent comparisons among candidate policies and, in simulation, are validated against known counterfactual outcomes. The planner minimizes predicted health and intervention costs subject to shared resource budgets:

$$
\begin{array} { r l } { \underset { \mathbf { a } _ { t : t + H - 1 } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { \tau = t } ^ { t + H - 1 } \ell \big ( \hat { \tau } _ { \tau } ^ { 1 : N } , \mathbf { a } _ { \tau } \big ) - \beta \widetilde { \rho } _ { \varphi } \big ( \hat { \tau } _ { t + 1 : t + H } ^ { 1 : N } \big ) } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { i = 1 } ^ { N } a _ { \tau } ^ { i , d } \leq B _ { \tau } ^ { d } , \qquad d \in \mathcal { D } _ { \mathrm { s h a r e d } } , } \\ & { 0 \leq a _ { \tau } ^ { i , d } \leq 1 , \qquad i = 1 , \dots , N , \quad \tau = t , \dots , t + H - 1 , } \end{array}\tag{5}
$$

where ℓ balances predicted epidemic burden and intervention cost, $\widetilde { \rho } _ { \varphi }$ is a differentiable robustness score for temporal specification $\varphi ,$ and $\mathcal { D } _ { \mathrm { s h a r e d } }$ indexes the resource-constrained action dimensions. Projection guarantees that the executed action satisfies the specified resource budgets, and temporal constraints are evaluated relative to the learned model by rerolling the projected action.

## 2.2 EPIMIND FRAMEWORK

EpiMind couples a learned GWM with constrained receding-horizon planning as shown in Figure 1. The world model predicts joint epidemic trajectories under candidate regional interventions, and the planner coordinates the interventions subject to the shared resource constraints.

## 2.2.1 GRAPH WORLD-MODEL ROLLOUTS

Graph-factored recurrent state-space model (GF-RSSM). We instantiate the dynamics in Section 2.1 with a graph-factored recurrent state-space model (GF-RSSM). Neural parameters are shared across regions, while each region i maintains its own recurrent belief $b _ { t } ^ { i }$ and stochastic state $z _ { t } ^ { i } .$ . Neighboring states and actions are aggregated through graph attention:

$$
c _ { t } ^ { i } = \mathrm { G A T } _ { \boldsymbol { \theta } } \left( \{ z _ { t } ^ { j } , a _ { t } ^ { j } , E _ { t } ^ { i j } \} _ { j \in \mathcal { N } ( i ) } \right) ,\tag{6}
$$

$$
b _ { t } ^ { i } = f _ { \theta } \big ( b _ { t - 1 } ^ { i } , z _ { t - 1 } ^ { i } , a _ { t - 1 } ^ { i } , c _ { t - 1 } ^ { i } \big ) ,\tag{7}
$$

$$
z _ { t } ^ { i } \sim q _ { \theta } \left( z _ { t } ^ { i } \mid b _ { t } ^ { i } , e _ { \theta } \left( o _ { t } ^ { i } \right) \right) , \qquad \hat { o } _ { t } ^ { i } \sim p _ { \theta } \left( o _ { t } ^ { i } \mid b _ { t } ^ { i } , z _ { t } ^ { i } \right) .\tag{8}
$$

Joint policy-conditioned rollout. The graph context $c _ { t } ^ { i }$ transmits mobility-weighted information from neighboring regions. Parameter sharing provides a common transition model without imposing identical regional trajectories: beliefs, latent states, observations, actions, and neighborhoods remain region specific. The posterior in (8) assimilates the current observation. Future observations are unavailable during planning, so imagined trajectories use the learned prior recursively. At every rollout step, all regional states advance jointly under the complete action vector and mobility graph:

$$
\hat { \mathbf { x } } _ { t + 1 : t + H } = \mathcal { M } _ { \theta } ( \mathbf { b } _ { t } , \mathbf { z } _ { t } , \mathbf { a } _ { t : t + H - 1 } ; G _ { t : t + H - 1 } ) .\tag{9}
$$

Thus, each trajectory depends on both local and neighboring interventions. Dedicated admission and occupancy heads decode the health quantities used by the planner.

World-model training. The GF-RSSM is trained offline for predictive and policy-effect fidelity:

$$
{ \mathcal { L } } _ { \mathrm { W M } } = { \mathcal { L } } _ { \mathrm { o b s } } + { \mathcal { L } } _ { \mathrm { r e w a r d } } + { \mathcal { L } } _ { \mathrm { K L } } + { \mathcal { L } } _ { \mathrm { r o l l o u t } } + { \mathcal { L } } _ { \mathrm { e f f e c t } } + { \mathcal { L } } _ { \mathrm { c a l } } ,\tag{10}
$$

where the first three terms form the recurrent state-space objective; the rollout, effect, and calibration terms supervise delayed policy responses. The trained world model is frozen during planning.

## 2.2.2 GRAPH-TEMPORAL CONSTRAINED PLANNING

Graph-temporal alternating direction method of multipliers (GT-ADMM). We implement the constrained planner as graph-temporal alternating direction method of multipliers (GT-ADMM), which alternates regional action optimization, shared-resource projection, and coordination updates. At decision epoch t, GT-ADMM optimizes region-specific action sequences over horizon H:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \mathbf { a } } } & { ~ J _ { \theta } ( \mathbf { a } ) = \displaystyle \sum _ { i = 1 } ^ { N } f ^ { i } \big ( \hat { \mathbf { x } } _ { t + 1 : t + H } , \mathbf { a } ^ { i } \big ) - \lambda _ { \mathrm { S T L } } \sum _ { i = 1 } ^ { N } \rho _ { \mathrm { s m } } \big ( \Phi ^ { i } , \hat { \mathbf { x } } _ { t + 1 : t + H } ^ { i } \big ) , } \\ & { ~ \hat { \mathbf { x } } _ { t + 1 : t + H } = \mathcal { M } _ { \theta } ( \mathbf { a } ; G _ { t } ) , } \end{array}\tag{11}
$$

subject to local action bounds, edge relations, and the shared resource set $\mathcal { Z } _ { t }$ . Here $f ^ { i }$ is the perregion health–intervention cost:

$$
f ^ { i } = \sum _ { h = 1 } ^ { H } \left[ \ell _ { \mathrm { h e a l t h } , t + h } ^ { i } + \lambda _ { \mathrm { N P I } } \ell _ { \mathrm { N P I } , t + h } ^ { i } \right] ,\tag{12}
$$

where $\lambda _ { \mathrm { { N P I } } }$ controls the trade-off between predicted health burden and NPI burden. The coefficient λ weights the smooth robustness $\rho _ { \mathrm { s m } }$ of temporal specification $\Phi ^ { i }$ . GT-ADMM alternates regional proposal updates, shared-resource projection, and coordination updates.

Regional proposal update. Each region updates its action block while holding the other regions at their latest reference actions:

$$
\begin{array} { l } { \displaystyle a ^ { i , k + 1 } = \underset { a ^ { i } \in \mathcal { X } ^ { i } } { \arg \operatorname* { m i n } } \left[ \mathcal { T } _ { \mathrm { r o l l } } ^ { i } + \frac { \rho _ { e } } { 2 } \sum _ { j \in \mathcal { N } ( i ) } \| a ^ { i } - a ^ { j , k } \| ^ { 2 } + ( \gamma ^ { i , k } ) ^ { \top } a ^ { i } \right. } \\ { \displaystyle \qquad \left. + \frac { \rho _ { g } } { 2 } \| a ^ { i } - y ^ { i , k } + u ^ { i , k } \| ^ { 2 } + ( \eta ^ { i , k } ) ^ { \top } s ^ { i } + \frac { \sigma } { 2 } \| a ^ { i } - a ^ { i , k } \| ^ { 2 } \right] , } \end{array}\tag{13}
$$

where

$$
\mathcal { I } _ { \mathrm { r o l l } } ^ { i } = f ^ { i } \left( a ^ { i } , \mathcal { M } _ { \theta } ( a ^ { i } , \mathbf { a } ^ { - i , k } ; G _ { t } ) \right) - \beta \rho _ { \mathrm { s m } } \left( \Phi ^ { i } , \mathcal { M } _ { \theta } ( a ^ { i } , \mathbf { a } ^ { - i , k } ; G _ { t } ) \right) .\tag{14}
$$

The rollout remains joint so that changing $a ^ { i }$ can alter the predicted outcomes of every connected region. The remaining terms encourage neighboring-policy agreement, consistency with the feasible allocation, spillover awareness, and stable successive updates. Equation (13) is solved by gradient descent through the frozen world model. The spillover signal is

$$
s ^ { i } = \sum _ { j \in \mathcal { N } ( i ) } E _ { t } ^ { i j } \nabla _ { a ^ { i } } g _ { \mathrm { s p i l l } } ^ { j } ( x _ { t } , a ^ { i } ) ,\tag{15}
$$

where $g _ { \mathrm { s p i l l } }$ predicts neighboring outcome changes. This signal is predictive, not causally identified.

Resource projection. Regional proposals need not be jointly feasible. GT-ADMM therefore computes the nearest allocation in the shared resource set:

$$
y ^ { k + 1 } = \underset { z \in \mathcal { Z } _ { t } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } \frac { \rho _ { g } } { 2 } \left\| a ^ { i , k + 1 } - z ^ { i } + u ^ { i , k } \right\| ^ { 2 } .\tag{16}
$$

Because $\mathcal { Z } _ { t }$ contains linear box and sum constraints, the z-step decomposes by resource into cappedsimplex projections. The planner executes $y ^ { k + 1 }$ , guaranteeing satisfaction of these budgets.

Coordination updates. After projection, the coordination variables are updated as

$$
\gamma ^ { i , k + 1 } = \gamma ^ { i , k } + \rho _ { e } \sum _ { j \in \mathcal { N } ( i ) } \left( a ^ { i , k + 1 } - a ^ { j , k + 1 } \right) ,\tag{17}
$$

$$
\begin{array} { r } { u ^ { i , k + 1 } = u ^ { i , k } + a ^ { i , k + 1 } - y ^ { i , k + 1 } , } \end{array}\tag{18}
$$

$$
\eta ^ { i , k + 1 } = \eta ^ { i , k } + \alpha s ^ { i } ,\tag{19}
$$

where $u ^ { i }$ reflects pressure from shared-resource scarcity, $\gamma ^ { i }$ tracks disagreement with mobilityconnected regions, and $\eta ^ { i }$ tracks predicted cross-region spillover sensitivity. The node-level $\dot { \gamma } ^ { i }$ and $\eta ^ { i }$ are approximate accumulators for neighbor disagreement and predicted spillover sensitivity, respectively. These approximations provide interpretable coordination signals.

## 2.2.3 PROJECTION, VERIFICATION, AND EXECUTION

After the final iteration, EpiMind retains the unprojected proposal $a ^ { K }$ and executes the projected allocation $y ^ { K }$ . Both are rerolled through the same joint world-model interface. Smooth STL robustness is used to obtain gradients during optimization, whereas exact nonsmooth robustness is evaluated on the projected trajectory:

$$
\rho _ { \mathrm { e x e c } } = \rho _ { \mathrm { e x a c t } } ( \Phi , \mathcal { M } _ { \theta } ( y ^ { K } ; G _ { t } ) ) .\tag{20}
$$

Resource feasibility is guaranteed for the constraints represented in $\mathcal { Z } _ { t }$ . In contrast, STL satisfaction is model relative, i.e., a positive $\rho _ { \mathrm { e x e c } }$ certifies the learned rollout, not the unknown true environment. The first action of $y ^ { K }$ is executed, and the resulting joint observation updates the regional posterior at decision epoch $t + 1$ . This produces the closed-loop sequence: beliefupdate → joint rollout → constrained planning → projection and verification → execution.

## 3 EXPERIMENTS

We evaluate EpiMind through five research questions spanning predictive fidelity, planning utility, constraint handling, coordination, and real-context transfer:

RQ1 Predictive fidelity. Does the graph world model accurately predict held-out trajectories and policy responses? (Section 3.2.1, Appendix D.1)

RQ2 Planning effectiveness. Does planning through learned joint rollouts improve the matched health–intervention objective? (Section 3.2.2, Appendix D.2)

RQ3 Feasibility and verification. Does projection enforce shared budgets, and do projected rollouts satisfy model-relative temporal specifications? (Section 3.2.3, Appendix D.3)

RQ4 Graph coordination. Does mobility-aware coordination improve allocation under heterogeneous conditions and shared resources (Section 3.2.4, Appendix D.4)

RQ5 Real-context transfer. Does EpiMind support forecasting and allocation in realistic multiregion settings? (Section 3.2.5, Appendices D.5–D.6)

## 3.1 EXPERIMENTAL PROTOCOL

Evaluation tracks. We evaluate EpiMind in three complementary settings: (1) a mobility-coupled multi-region simulator provides known dynamics and ground-truth outcomes for evaluating policyconditioned prediction and planning; (2) a retrospective U.S. state-level panel evaluates forecasting and allocation behavior under observed surveillance, intervention, capacity, and mobility data; and (3) a real-context semi-simulated setting which initializes the simulator from real data while retaining known dynamics for realized evaluation of alternative policies.

Table 1: World-model fidelity in the synthetic environment. Values are mean ± s.d. Admission errors are measured per 100K; cumulative error is over a five-step rollout. Lower is better.
<table><tr><td>Model</td><td>Adm. MAE↓</td><td>Adm. RMSE↓</td><td>Cum. err. @5 ↓</td><td>Params</td></tr><tr><td colspan="5">Statistical baselines</td></tr><tr><td>Persistence</td><td> $1 . 5 6 3 \pm 0 . 0 5 3$ </td><td> $1 . 9 2 7 \pm 0 . 0 5 6$ </td><td> $0 . 0 8 9 \pm 0 . 0 0 3$ </td><td>0</td></tr><tr><td>Climatology</td><td> $1 . 1 2 9 \pm 0 . 0 2 1$ </td><td> $1 . 2 8 3 \pm 0 . 0 2 7$ </td><td> $0 . 5 2 8 \pm 0 . 0 1 1$ </td><td>0</td></tr><tr><td>Ridge + action</td><td> $0 . 3 4 1 \pm 0 . 0 0 8$ </td><td> $0 . 4 3 7 \pm 0 . 0 0 3$ </td><td> $0 . 0 6 8 \pm 0 . 0 0 6$ </td><td>84</td></tr><tr><td> $\operatorname { V A R X } ( 1 ) + \arctan$ </td><td> $0 . 3 2 9 \pm 0 . 0 0 8$ </td><td> $0 . 4 2 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 6 \pm 0 . 0 0 7$ </td><td>1,960</td></tr><tr><td colspan="5">Learned dynamics models</td></tr><tr><td>Action-LSTM</td><td> $0 . 2 1 2 \pm 0 . 0 4 4$ </td><td> $0 . 3 1 0 \pm 0 . 0 6 2$ </td><td> $0 . 0 5 6 \pm 0 . 0 0 7$ </td><td>25,507</td></tr><tr><td>GF-RSSM (no graph)</td><td> $0 . 2 1 1 \pm 0 . 0 3 3$ </td><td> $0 . 3 1 1 \pm 0 . 0 4 6$ </td><td> $0 . 0 7 0 \pm 0 . 0 1 2$ </td><td>92,972</td></tr><tr><td>GF-RSSM (ours)</td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 4 9 }$ </td><td> $\mathbf { 0 . 2 2 1 \pm 0 . 0 7 0 }$ </td><td> ${ \bf 0 . 0 3 6 \pm 0 . 0 0 3 }$ </td><td>92,972</td></tr></table>

Datasets. The synthetic benchmark contains N=5 mobility-coupled regions over T=26 weekly decision epochs. Regional actions control NPI intensity and allocations of vaccine, hospitalizationcapacity, and fiscal resources under shared budgets. The planner observes delayed, noisy surveillance signals rather than latent SEIR states; Table A2 reports the complete simulator configuration.

For the real-context evaluation, we construct a weekly U.S. state-level panel combining reported cases, deaths, hospital admissions and capacity, vaccination, policy interventions, population, and directed interstate mobility. Track A retrospectively evaluates forecasting and model-relative policy projections at held-out decision origins; outcomes under unexecuted policies are unavailable. Track B initializes a semi-synthetic simulator from the same regional conditions and mobility graph, permitting realized evaluation of alternative policies under known dynamics. Data sources and preprocessing appear in Appendix C.1.

Benchmarking methods. World-model comparisons include statistical predictors, actionconditioned sequence models, and graph ablations. Planning comparisons include constant and heuristic policies, MPC, ADMM, RL-based controllers, graph-free and independent variants, and an oracle-dynamics reference. All policies are evaluated under the same action bounds, resource budgets, projection, and health–intervention objective. Implementation details appear in Appendix C.2.

Metrics. For RQ1, we report held-out admission MAE, RMSE, cumulative rollout error, and policy-conditioned dose response. For RQ2, we report admissions per 100K, NPI burden, matched objective J, and regret relative to the best feasible constant policy. For RQ3, we report budget feasibility, maximum excess, projection displacement, and exact model-relative STL robustness. For RQ4, we compare realized objective values and paired outcomes under shared budgets, supplemented by a stepwise matched-burden ablation. For RQ5, we report retrospective model-relative comparisons and realized outcomes in the real-context benchmark.

## 3.2 EXPERIMENTAL RESULTS

## 3.2.1 GF-RSSM SUPPORTS ACCURATE POLICY-CONDITIONED PREDICTION.

Table 1 evaluates deterministic prior rollouts on held-out synthetic episodes. GF-RSSM achieves the lowest error on all three admission metrics. Relative to the no-graph ablation, the full model reduces admission MAE by 26% (0.211 to 0.156), RMSE by 29% (0.311 to 0.221), and five-step cumulative error by 49% (0.070 to 0.036). It similarly improves over Action-LSTM by 26%, 29%, and 36%, respectively. Although uncertainty over three checkpoints limits strong statistical conclusions for MAE and RMSE, the cumulative-error improvement is consistent across checkpoints.

Figure 2 examines whether this predictive accuracy extends to policy-conditioned responses. In simulation, GF-RSSM preserves the monotonic NPI dose ordering but overpredicts admissions at low NPI and underpredicts them at high NPI. On retrospective data, predicted admissions decrease with NPI across all four held-out decision origins. These real-data curves demonstrate stable model sensitivity, not causal effects, because counterfactual outcomes are unavailable.

![](images/4a85ef52ff78b30d9500024b43eabb0e8856385b3239edf45d0b72d5305e5d8e.jpg)

![](images/75dfecb2fd83e9a42bff34212b1a535244801b4539d555141ed2091d64914ff3.jpg)  
Figure 2: Policy-conditioned admission response. (a) Peak weekly admissions under alternative NPI intensities applied from a common synthetic state. (b) Predicted cumulative admissions under the same sweep at four held-out U.S. decision origins.

## 3.2.2 LEARNED ROLLOUTS YIELDS EFFECTIVE RESOURCES-FEASIBLE INTERVENTIONS.

We evaluate all methods under the same health–intervention objective, action bounds, shared budgets, and final resource projection. Table 2 reports simulator-realized objective values across three intervention-cost regimes. The best constant-NPI policy saturates the shared budget, providing a strong non-adaptive comparator. EpiMind remains within 1.1%, 1.5%, and 4.9% of this comparator at $\lambda _ { \mathrm { N P I } } \in \{ 3 , \bar { 1 0 } , 3 0 \}$ , respectively, with paired regret $0 . 9 3 { \pm } 1 . 1 0 \mathrm { a t } \lambda _ { \mathrm { N P I } } = 1 0$ . It also consistently outperforms PPO, MPC-SEIR, independent MPC, and the remaining planning baselines. All executed allocations have zero post-projection budget excess. These results show that planning through learned joint rollouts produces effective resource-feasible interventions. The learned-versus-oracle decomposition in Appendix D.3 suggests a model contribution to the remaining regret, but the effect is not statistically resolved with four paired cells.

Table 2: Matched-objective planning performance. Values are mean ± s.d. Lower is better.
<table><tr><td></td><td colspan="3">Objective J↓</td><td>Regret at  $\lambda = 1 0 \downarrow$ </td></tr><tr><td>Method</td><td> $\lambda = 3$ </td><td> $\lambda = 1 0$ </td><td> $\lambda = 3 0$ </td><td>mean ± s.d.</td></tr><tr><td>Best feasible constant†</td><td>54.7</td><td>58.9</td><td>70.9</td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>PPO</td><td>60.4</td><td>64.6</td><td>76.6</td><td> $5 . 6 9 \pm 3 . 7 1$ </td></tr><tr><td>MPC-SEIR</td><td>108.6</td><td>112.8</td><td>124.6</td><td> $5 3 . 9 \pm 3 0 . 9$ </td></tr><tr><td>Independent MPC</td><td>366.3</td><td>431.1</td><td>394.1</td><td> $3 7 2 \pm 3 0 7$ </td></tr><tr><td>Greedy</td><td>7,186</td><td>7,189</td><td>7,197</td><td> $7 , 1 3 0 \pm 1 , 2 2 1$ </td></tr><tr><td>D-ADMM</td><td>9,300</td><td>9,263</td><td>9,224</td><td> $9 { , } 2 0 4 \pm 5 8 3$ </td></tr><tr><td>HRL</td><td>9,429</td><td>9,429</td><td>9,429</td><td> $9 , 3 7 0 \pm 4 7 8$ </td></tr><tr><td>No intervention</td><td>13,853</td><td>13,853</td><td>13,853</td><td> $1 3 , 7 9 5 \pm 2 9$ </td></tr><tr><td>EpiMind</td><td>55.3</td><td>59.8</td><td>74.4</td><td> ${ \bf 0 . 9 3 \pm 1 . 1 0 }$ </td></tr></table>

<sup>†</sup>Constant NPI at the shared-budget cap; no adaptive planning.

## 3.2.3 RESOURCE PROJECTION GUARANTEES FEASIBLE EXECUTION.

Table 3 evaluates the projected actions that are executed. All allocations satisfy the encoded linear resource constraints, with zero maximum budget excess. Projection modifies synthetic proposals more than real-context proposals, as indicated by their mean displacement (0.476 versus 0.0039). After projection, all evaluated world-model rollouts satisfy the STL specification with positive exact robustness. Resource feasibility is guaranteed for the encoded linear constraints, whereas STL satisfaction is model relative and does not certify the unknown environment. Robustness to operational perturbations and epidemiological model mismatch is reported in Appendix D.2 (Figure A3).

Table 3: Constraint handling and model-relative verification.
<table><tr><td>Setting</td><td>Feasible (%)</td><td>Max excess</td><td>Projection displacement STL satisfaction (%)</td><td></td><td>STL robustness</td></tr><tr><td>Synthetic</td><td>100.0</td><td></td><td> $0 . 4 7 6 \pm 0 . 2 0 1$ </td><td>100.00</td><td> $0 . 0 0 2 1 \pm 0 . 0 0 0 4$ </td></tr><tr><td>Real context</td><td>100.0</td><td></td><td> $0 . 0 0 3 9 \pm 0 . 0 4 0 0$ </td><td>100.00</td><td> $0 . 0 0 1 6 \pm 0 . 0 0 0 3$ </td></tr></table>

Table 4: Realized policy performance in the real-context semi-synthetic evaluation.
<table><tr><td>Method</td><td>Adm./100K↓</td><td>NPI burden</td><td> $J ^ { \dagger } \downarrow$ </td><td>∆J (%)</td><td>EpiMind wins (p)</td></tr><tr><td>Standard ADMM</td><td>30.24</td><td>0.423</td><td>31.51</td><td>+1.14</td><td>23/36 (0.132)</td></tr><tr><td>Centralized MPC</td><td>30.65</td><td>0.424</td><td>31.92</td><td>+2.45</td><td>32/36 (&lt; 0.001)</td></tr><tr><td>Independent MPC</td><td>30.94</td><td>0.424</td><td>32.21</td><td>+3.40</td><td>26/36 (0.011)</td></tr><tr><td>Uniform allocation</td><td>31.23</td><td>0.479</td><td>32.67</td><td>+4.87</td><td>30/36 (&lt; 0.001)</td></tr><tr><td>EpidRLearn</td><td>35.64</td><td>0.500</td><td>37.14</td><td>+19.23</td><td>36/36 (&lt; 0.001)</td></tr><tr><td>PPO</td><td>35.69</td><td>0.499</td><td>37.19</td><td>+19.37</td><td>36/36 (&lt; 0.001)</td></tr><tr><td>Incidence-proportional</td><td>36.61</td><td>0.352</td><td>37.67</td><td>+20.92</td><td>30/36 (&lt; 0.001)</td></tr><tr><td>No intervention</td><td>64.14</td><td>0.000</td><td>64.14</td><td>+105.90</td><td>36/36 (&lt; 0.001)</td></tr><tr><td>EpiMind</td><td>29.89</td><td>0.423</td><td>31.15</td><td></td><td></td></tr></table>

<sup>†</sup>Policies are evaluated by $J = \mathrm { A d m . } / 1 0 0 \mathrm { K } + \lambda _ { \mathrm { N P I } } \mathrm { N P I } ;$ ∆J is the percentage change relative to EpiMind.

## 3.2.4 COORDINATION BENEFITS ARE MODEST UNDER MATCHED INTERVENTION BURDEN.

Table 4 evaluates realized outcomes under a common objective and shared resource constraints. Among the directly comparable planning methods, which incur nearly identical NPI burden (0.423– 0.424), EpiMind reduces J by 1.14% relative to global-only ADMM, 2.13% relative to graph-free planning, 2.21% relative to time-shuffled planning, and 3.40% relative to independent MPC. These results suggest benefits from graph structure and temporal allocation, although their magnitude is small. A stricter matched-burden ablation in Appendix Table A5 holds the NPI trajectory fixed step by step, isolating where interventions are allocated from how much is spent. Under this control, the coordination gains fall below 1%, indicating that much of the uncontrolled difference arises from intervention burden rather than allocation alone. Region shuffling is reported only as a sensitivity diagnostic because shuffling after projection breaks the population-weighted budget constraints. A matched-burden component ablation further isolates the coordination mechanism (Appendix Table A5). The results indicate that the coordination signals provide consistent but incremental gains once intervention burden is controlled.

## 3.2.5 EPIMIND TRANSFERS TO REAL-CONTEXT MULTI-REGION PLANNING.

![](images/f4cb3013743dd62fc23fae8a8d97eb3461b14ba6d9be7567dfb36d290b47b699.jpg)  
Figure 3: Real-context policy comparisons. Each point reports the mean paired admission difference $\Delta \mathrm { A d m } = \mathrm { A d m } _ { \mathrm { c o m p a r a t o r } } - \mathrm { A d m } _ { E p i M i n d } ;$ positive values (blue circles) favor EpiMind; horizontal bars denote 95% confidence intervals. (a) Retrospective Track A reports model-relative projections at held-out U.S. state-level decision origins. (b) Semi-synthetic Track B reports realized outcomes under known simulation dynamics. Diamonds denote the track-specific reference policy.

Figure 3 summarizes paired comparisons across both real-context tracks. In retrospective Track A, EpiMind projects fewer admissions than most comparators, although Standard ADMM is marginally better on average. These comparisons are model relative because the candidate policies were not executed. In semi-synthetic Track B, where counterfactual outcomes are known, EpiMind outperforms every deployable comparator. Its largest gains are over no intervention, EpiPolicy-RL, incidenceweighted allocation, and EpidRLearn. Smaller differences from Standard ADMM, graph-free planning, and shuffled controls indicate that constrained optimization provides most of the improvement, with graph and temporal coordination contributing incrementally.

![](images/82c4f31f618c1b20d466ac612fc2ff170b97dbafb4a99bb7c0264a41509a9d08.jpg)

![](images/7f46a68cdbf553953ae5d7aebff28b443dc3b6fb529a4861399232ae32e41505.jpg)

(c) Policy allocation: Texas and neighbors  
![](images/cca7ae927980b3ff3bc5fefae06013a097e99b081bb90120967b878a17dc989f.jpg)  
Figure 4: Texas case study. (a) Admission forecasts under the observed policy; the inset enlarges weeks 50–60. (b) Model-relative counterfactual trajectories under matched resource constraints; the observed trajectory provides context but is not an outcome of the unexecuted policies. (c) EpiMind’s first-step allocation compared with historical actions in Texas and neighboring states. The dashed vertical line marks the training cutoff. MPC and RL baselines appear only in panel (b).

We next use Texas as a representative decision origin to illustrate the retrospective forecasting and planning workflow (Figure 4, Figure A5). Compared with the forecasting baselines, GF-RSSM more closely tracks the principal admission peak, while EpiMind and the planning baselines produce distinct model-relative trajectories under matched constraints. EpiMind assigns distinct actions to Texas and its mobility-connected neighbors under shared constraints (Figure 4c)). The empirical mobility graph and additional state-level rollouts are shown in Appendix Figures A4 and A6.

## 4 CONCLUSION AND LIMITATIONS

We presented EpiMind, a graph world-model framework for constrained epidemic planning across regions. Its parameter-shared GF-RSSM maintains region-specific beliefs and predicts joint trajectories under candidate policies. GT-ADMM uses these rollouts to coordinate regional interventions and project shared allocations onto the feasible set. The experiments demonstrate accurate policyconditioned prediction, effective planning through learned dynamics, and exact enforcement of the specified linear resource budgets. Under matched intervention burden, however, graph coordination provides a measurable but incremental benefit.

Several limitations remain. Projection guarantees only the constraints encoded in the feasible set, and STL satisfaction applies to learned trajectories rather than the unknown environment. Planning quality depends on world-model calibration, and the nonconvex GT-ADMM procedure has no global convergence guarantee. Because real-world outcomes under alternative policies are unobserved, the predicted trajectories and spillover effects represent model-based sensitivities rather than causally identified counterfactuals Hernan & Robins (2020). Finally, intervention costs and allocation pri-´ orities must reflect local economic, ethical, and public-health considerations. EpiMind is therefore intended to support policy comparison and resource allocation, not to make decisions autonomously.

## 5 SOFTWARE AND DATA

We release the full implementation at https://anonymous.4open.science/r/ epimind-9706/README.md.

## AI USE DISCLOSURE

Generative AI tools were used to assist with literature retrieval and discovery and to improve the clarity and readability of the manuscript. All AI-assisted text was reviewed and revised by the authors, and all citations and literature-derived statements were verified against their original sources. The authors take full responsibility for the final content of this work.

## REFERENCES

Julien Arino and Pauline van den Driessche. A multi-city epidemic model. Mathematical Population Studies, 10(3):175–193, 2003.

Duygu Balcan, Bruno Gonc¸alves, Hao Hu, Jose J Ramasco, Vittoria Colizza, and Alessandro´ Vespignani. Modeling the spatial spread of infectious diseases: The GLobal epidemic and mobility computational model. Journal ofComputational Science, 1(3):132–145, 2010.

Dimitris Bertsimas, Vassilis Digalakis Jr, Alexandre Jacquillat, Michael Lingzhi Li, and Alessandro Previero. Where to locate COVID-19 mass vaccination facilities? Naval Research Logistics, 69 (2):179–200, 2022.

Alyssa Bilinski, Joshua A. Salomon, John Giardina, Andrea Ciaranello, and Meagan C. Fitzpatrick. Passing the test: A model-based analysis of safe school-reopening strategies. Annals of Internal Medicine, 174(8):1090–1100, 2021. doi: 10.7326/M21-0600. URL https://doi.org/10. 7326/M21-0600. PMID: 34097433.

Thomas A. Birkland, Kristin Taylor, Deserai A. Crow, and Rob A. DeLeo. Governing in a polarized era: Federalism and the response of U.S. state and federal governments to the COVID-19 pandemic. Publius: The Journal of Federalism, 51(4):650–672, 2021. doi: 10.1093/publius/pjab024.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011a.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011b. doi: 10.1561/2200000016.

Jan M. Brauner, Soren Mindermann, Mrinank Sharma, David Johnston, et al. Inferring the effec-¨ tiveness of government interventions against covid-19. Science, 371(6531):eabd9338, 2021. doi: 10.1126/science.abd9338.

Sabah Bushaj, Xuecheng Yin, Arjeta Beqiri, Donald Andrews, and <sup>˙</sup>I. Esra Buy¨ uktahtakın. A¨ simulation-deep reinforcement learning (DRL) approach for epidemic control optimization. Annals ofOperations Research, 328:245–277, 2023.

Songgaojun Deng, Shusen Wang, Huzefa Rangwala, Lijing Wang, and Yue Ning. Cola-GNN: Crosslocation attention based graph neural networks for long-term ILI prediction. In CIKM, pp. 245– 254, 2020.

Lotty Evertje Duijzer, Willem van Jaarsveld, and Rommert Dekker. Literature review: The vaccine supply chain. European Journal ofOperational Research, 268(1):174–192, 2018.

Ezekiel J. Emanuel, Govind Persad, Ross Upshur, Beatriz Thome, Michael Parker, Aaron Glickman, Cathy Zhang, Connor Boyle, Maxwell Smith, and James P. Phillips. Fair allocation of scarce medical resources in the time of COVID-19. New England Journal ofMedicine, 382(21):2049– 2055, 2020. doi: 10.1056/NEJMsb2005114.

Tao Feng, Yexin Wu, Guanyu Lin, and Jiaxuan You. Graph world model. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=xjTrTlBbrc.

Luca Ferretti, Chris Wymant, Michelle Kendall, Lele Zhao, Anel Nurtay, Lucie Abeler-Dorner,¨ Michael Parker, David Bonsall, and Christophe Fraser. Quantifying SARS-CoV-2 transmission suggests epidemic control with digital contact tracing. Science, 368(6491):eabb6936, 2020. doi: 10.1126/science.abb6936.

Seth Flaxman, Swapnil Mishra, Axel Gandy, et al. Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe. Nature, 584(7820):257–261, 2020.

Sebastian Funk, Marcel Salathe, and Vincent A A Jansen. Modelling the influence of human be-´ haviour on the spread of infectious diseases: a review. Journal of the Royal Society Interface, 7 (50):1247–1256, 2010. doi: 10.1098/rsif.2010.0142.

Shikha Garg, Lindsay Kim, Michael Whitaker, Alissa O’Halloran, et al. Hospitalization rates and characteristics of patients hospitalized with laboratory-confirmed coronavirus disease 2019 — covid-net, 14 states, march 1–30, 2020. MMWR. Morbidity and Mortality Weekly Report, 69(15): 458–464, 2020. doi: 10.15585/mmwr.mm6915e3.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Thomas Hale, Noam Angrist, Rafael Goldszmidt, Beatriz Kira, et al. A global panel database of pandemic policies (oxford covid-19 government response tracker). Nature Human Behaviour, 5 (4):529–538, 2021. doi: 10.1038/s41562-021-01079-8.

Xi He, Eric H. Y. Lau, Peng Wu, Xilong Deng, et al. Temporal dynamics in viral shedding and transmissibility of covid-19. Nature Medicine, 26(5):672–675, 2020. doi: 10.1038/ s41591-020-0869-5.

Miguel A. Hernan and James M. Robins.´ Causal Inference: What If. Chapman & Hall/CRC, Boca Raton, FL, 2020.

Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural Computation, 9(8): 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

Emily Howerton, Lucie Contamin, Luke C. Mullany, Michelle Qin, et al. Evaluation of the us covid-19 scenario modeling hub for informing pandemic response under uncertainty. Nature Communications, 14(1):7260, 2023. doi: 10.1038/s41467-023-42680-x.

Solomon Hsiang, Daniel Allen, Sebastien Annan-Phan, Kendon Bell, Ian Bolliger, Trinetta Chong,´ Hannah Druckenmiller, Luna Yue Huang, Andrew Hultgren, Emma Krasovich, et al. The effect of large-scale anti-contagion policies on the COVID-19 pandemic. Nature, 584:262–267, 2020. doi: 10.1038/s41586-020-2404-8.

Amol Kapoor, Xue Ben, Luyang Liu, Bryan Perozzi, Matt Barnes, Martin Blais, and Shawn O’Banion. Examining COVID-19 forecasting using spatio-temporal graph neural networks. In NeurIPS Workshop on Machine Learning in Public Health, 2020.

William Ogilvy Kermack and Anderson G McKendrick. A contribution to the mathematical theory of epidemics. Proceedings ofthe Royal Society ofLondon A, 115(772):700–721, 1927.

Varun Kompella, Roberto Capobianco, Stacy Jong, Jonathan Browne, Spencer Fox, Lauren Meyers, Peter Wurman, and Peter Stone. Reinforcement learning for optimization of COVID-19 mitigation policies. In AAAI Fall Symposium on AI for Social Good, 2020.

Moritz U. G. Kraemer, Chia-Hung Yang, Bernardo Gutierrez, Chieh-Hsi Wu, Brennan Klein, David M. Pigott, Louis du Plessis, Nuno R. Faria, Ruoran Li, William P. Hanage, et al. The effect of human mobility and control measures on the COVID-19 epidemic in china. Science, 368 (6490):493–497, 2020. doi: 10.1126/science.abb4218.

Stephen A. Lauer, Kyra H. Grantz, Qifang Bi, Forrest K. Jones, et al. The incubation period of coronavirus disease 2019 (covid-19) from publicly reported confirmed cases. Annals of Internal Medicine, 172(9):577–582, 2020. doi: 10.7326/M20-0504.

Qun Li, Xuhua Guan, Peng Wu, Xiaoye Wang, et al. Early transmission dynamics in wuhan, china, of novel coronavirus–infected pneumonia. New England Journal of Medicine, 382(13):1199– 1207, 2020. doi: 10.1056/NEJMoa2001316.

Zeeshan Memon, Yiqi Su, Christo Kurisummoottil Thomas, Walid Saad, Liang Zhao, and Naren Ramakrishnan. Toward world models for epidemiology. In ICLR 2026 the 2nd Workshop on World Models: Understanding, Modelling and Scaling, 2026. URL https://openreview. net/forum?id=T5ACq6FQqh.

Gideon Meyerowitz-Katz and Lea Merone. A systematic review and meta-analysis of published research data on covid-19 infection fatality rates. International Journal of Infectious Diseases, 101:138–148, 2020. doi: 10.1016/j.ijid.2020.09.1464.

Siddharth Nayak et al. Multi-agent reinforcement learning for decentralized epidemic control. arXiv preprint arXiv:2301.11367, 2023.

Abu Quwsar Ohi, M F Mridha, Muhammad Mostafa Monowar, and Md Abdul Hamid. Exploring optimal control of epidemic spread using reinforcement learning. Scientific Reports, 10(1):22106, 2020.

Minah Park, Alex R. Cook, Jue Tao Lim, Yinxiaohe Sun, and Borame L. Dickens. Reproduction numbers of covid-19: a systematic review. Journal of Clinical Medicine, 9(4):967, 2020. doi: 10.3390/jcm9040967.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707. 06347.

Gillian K. SteelFisher, Mary G. Findling, Hannah L. Caporello, Keri M. Lubell, Kathleen G. Vidoloff Melville, Lindsay Lane, Alyssa A. Boyea, Thomas J. Schafer, and Eran N. Ben-Porath. Trust in US federal, state, and local public health agencies during COVID-19: Responses and policy implications. Health Affairs, 42(3), 2023. doi: 10.1377/hlthaff.2022.01204.

Yiqi Su, Ray Lee, Jiaming Cui, and Naren Ramakrishnan. How (not) to hybridize neural and mechanistic models for epidemiological forecasting. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=otBTuq5AKB.

Huan Wang and Arindam Banerjee. Randomized smoothing variance reduction for stochastic admm. In Proceedings of the SIAM International Conference on Data Mining, 2019.

Yifei Wang et al. DMWM: Dual-mind world model with long-term imagination. arXiv preprint, 2025.

Joseph T. Wu, Kathy Leung, Mary Bushman, Nishant Kishore, et al. Estimating clinical severity of covid-19 from the transmission dynamics in wuhan, china. Nature Medicine, 26(4):506–510, 2020. doi: 10.1038/s41591-020-0822-7.

Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Baez, Siddharth Bhatt, Pieter Abbeel, and Thanasis Karagiannis. The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS, 2022.

## A EPIDEMIOLOGICAL FOUNDATIONS

This section introduces the epidemiological structure underlying EpiMind, including its latent dynamics, observations, and planning constraints.

![](images/3be58208ed2a6cfd59e41469092b9ae91ca5210666be4af73626c765b803c8b3.jpg)  
Figure A1: Multi-region policy graph under shared resource constraints. Regions are represented as nodes $v _ { i }$ in a dynamic policy graph $G _ { t } = ( V _ { t } , E _ { t } )$ , with edges encoding interregional mobility coupling. Each region receives allocations of vaccine, hospitalization-capacity, and fiscal resources. The joint regional allocations must satisfy graph-level resource budgets, $\begin{array} { r } { \dot { \sum _ { i = 1 } ^ { N } } a _ { t , k } ^ { i } \ \le \ B _ { t , k } } \end{array}$ , for $k \in \{ v , h , f \}$

## A.1 EPIDEMICS AS GRAPHS.

The spatiotemporal policy graph enables reasoning at three levels:

• Node level. Each node represents a region and encodes its latent state, healthcare capacity, population characteristics, and interventions through policy actions.

• Edge level. Each edge encodes inter-regional coupling, including mobility flows, spatial proximity, or shared infrastructure that mediates epidemic spillovers and policy coordination between connected regions.

• Graph level. Global resource constraints (vaccine supply, hospitalization capacity, fiscalresource budget) and equity requirements operate over the entire graph, coupling all regions’ feasible action sets.

Latent states. As the true compartmental state is never directly observed, surveillance systems report cases (a function of testing), hospitalizations (a function of severity and care-seeking), and deaths (a lagged indicator). Each of these is a noisy, delayed, and policy-dependent projection of the underlying state. For instance, expanding testing increases reported cases without changing true cases, while reducing testing has the opposite effect. This means that an observed decline in cases may reflect either genuine transmission reduction or reduced testing coverage, a fundamental ambiguity that forecasting models cannot resolve without separating latent state from observation. This motivates the use of a latent state representation that captures the true epidemic reality, including infection burden, immunity, behavioral compliance, and variant fitness behind the noisy observations Memon et al. (2026). By separating latent dynamics from a policy-dependent observation model, this representation disentangles genuine transmission changes from surveillance artifacts, enabling policy reasoning based on inferred reality rather than distorted measurements.

Interventions, behavioral mediation, and delayed effects. Epidemic interventions include NPIs, PIs, and surveillance interventions. A critical feature of these interventions is behavioral mediation Funk et al. (2010). Mandates do not directly reduce transmission. Instead, they alter human behavior, which in turn changes contact patterns and infection risk. Compliance with interventions is partial, heterogeneous, and time-varying, depending on perceived risk, fatigue, trust, and economic pressure. In addition, all interventions operate with temporal delays. An NPI enacted today cannot reduce transmission until behavioral changes propagate through the population (typically 1–2 weeks, corresponding to the generation interval $\tau _ { g } )$ . Vaccination requires weeks to build immunity. These delays mean that the effect of an action taken at time t does not appear in surveillance data until $t + \tau _ { g }$ or later, so a planner cannot validate or course-correct an active intervention against current observations. By the time an intervention’s effect becomes visible, the epidemic has already evolved into a different state, which is precisely why planning under epidemic dynamics must rely on forward simulation of latent state under candidate interventions rather than on real-time feedback from surveillance alone.

Resource constraints. Since the resources available for intervention are shared across regions and physically finite, epidemic policy can be formulated as a constrained optimization, with two properties that distinguish it from standard constrained programs. First, feasibility must hold at every decision step, since vaccine doses cannot be administered beyond the available stock at time t, hospital and ICU occupancy cannot exceed bed capacity, and per-period public-health expenditures cannot exceed the appropriated budget. Formulations that incorporate constraints as additive cost terms drive expected violation to zero only asymptotically. They are insufficient here, because an over-allocation at any single step cannot be implemented in practice. Second, each constraint is a sum across regions rather than a per-region limit, as the underlying resources are pooled at the national or system-wide level. Therefore, each region’s feasible action set depends on what the others request, and the optimization cannot be decomposed into independent single-region problems.

## A.2 FROM POLICY-CONDITIONED PREDICTION TO CONSTRAINED COORDINATION

The epidemiological properties described in Section A create the following unique computational challenges that current methods only partially address:

(A) Policy-conditioned prediction under uncertainty. Epidemic dynamics are nonlinear, partially observed, and shaped by latent behavioral responses Su et al. (2026). Planning therefore requires inferring regional latent states from noisy surveillance and generating joint trajectories under candidate interventions.

(B) Coordination under shared constraints. Regional decisions are coupled by mobilitydriven spillovers, finite shared resources, and delayed intervention effects Flaxman et al. (2020). Effective planning must account for these interactions while enforcing resource limits and temporal requirements explicitly.

(C) Verification after projection. Resource projection can alter the planner’s unconstrained proposal. Specifications must therefore be evaluated on a new joint rollout under the projected action that will be executed. This separates exact resource feasibility from modelrelative temporal verification.

Table A1: Epidemiological challenges and their computational treatment in EpiMind.
<table><tr><td></td><td>Challenge Epidemiological property</td><td>Component</td><td>Computational response</td></tr><tr><td rowspan="4">(A)</td><td>Latent disease burden</td><td>GF-RSSM</td><td>Latent belief inference</td></tr><tr><td>Policy-dependent surveillance</td><td>GF-RSSM</td><td>Action-conditioned observation model</td></tr><tr><td>Delayed intervention effects</td><td>GF-RSSM</td><td>Multi-horizon policy-conditioned roll- out</td></tr><tr><td>Cross-region spillovers</td><td>GF-RSSM</td><td>Graph attention over neighbors</td></tr><tr><td rowspan="4">(B)</td><td>Shared resource limits</td><td>GT-ADMM</td><td>Capped-simplex projection</td></tr><tr><td>Cross-border coordination</td><td>GT-ADMM</td><td>Neighbor-consensus signal</td></tr><tr><td>Spillover externalities</td><td>GT-ADMM</td><td>Spillover-sensitivity accumulator</td></tr><tr><td>Temporal specifications</td><td>GT-ADMM/STL</td><td>Differentiable robustness objective</td></tr><tr><td>(C)</td><td>Proposal-execution mismatch</td><td>Verification</td><td>Joint reroll under projected action</td></tr></table>

These challenges motivate the pipeline in Figure 1. The learned graph world model maps the current posterior beliefs and a candidate joint action sequence to a differentiable joint trajectory. GT-ADMM uses these trajectories as its predictive objective, updates region-specific action sequences, and projects the joint allocation onto the shared-resource polytope. Finally, the framework rerolls the projected action and evaluates exact STL robustness under the learned model. Dynamics learning remains predictive: constraint satisfaction is not inserted into the world-model training loss. The interaction between learning and planning occurs operationally, because executed actions determine the observations used for the next posterior update.

Table A1 summarizes the division of computational responsibilities. The GF-RSSM answers how the coupled regional epidemic is predicted to evolve under a candidate joint intervention. GT-ADMM answers which region-specific intervention sequence minimizes the predicted objective while respecting the specified shared budgets. Post-projection reroll distinguishes the proposed trajectory from the model-predicted trajectory under the action that will actually be executed. This modular separation permits the dynamics model to be trained for predictive fidelity and the planner to enforce operational constraints explicitly, without claiming that optimization constraints reshape the learned dynamics.

## B EPIMIND FRAMEWORK

Algorithm 1 EpiMind constrained receding-horizon planning   
Require: Frozen world model $\mathcal { M } _ { \theta } .$ , graph $G _ { t } ,$ , observation $\mathbf { o } _ { t }$ , previous plan $\mathbf { a } _ { t - 1 } ^ { \star }$ , feasible set $\mathcal { Z } _ { t } .$   
temporal specifications Φ, horizon H   
Ensure: Executed joint action $\mathbf { a } _ { t } ^ { \star }$   
1: Update regional beliefs: $( \mathbf { b } _ { t } , \mathbf { \tilde { z } } _ { t } ) \gets \mathrm { P o s t e r i o r } _ { \theta } ( \mathbf { o } _ { t } , \mathbf { a } _ { t - 1 } , G _ { t } )$   
2: Warm-start action sequences $\mathbf { a } ^ { 0 ^ { \prime } } \gets \mathrm { S h i f t } ( \mathbf { a } _ { t - 1 } ^ { \star } ) ;$ ; set $\mathbf { y } ^ { 0 }  \mathbf { a } ^ { 0 }$ and $\gamma ^ { 0 } , \mathbf { u } ^ { 0 } , \eta ^ { 0 } \gets 0$   
3: for $k = 0 , \ldots , K _ { \mathrm { m a x } } - 1$ do   
4: for all regions i in parallel do   
5: Update $a ^ { i , k + 1 }$ by minimizing (13) through a joint prior rollout   
$\hat { \mathbf { x } } _ { t + 1 : t + H } = \mathcal { M } _ { \theta } \left( \mathbf { b } _ { t } , \mathbf { z } _ { t } , a ^ { i } , \mathbf { a } ^ { - i , k } ; G _ { t } \right)$   
6: end for   
7: Project shared allocations:   
$\mathbf { y } ^ { k + 1 } \gets \mathrm { P r o j } _ { \mathcal { Z } _ { t } } \left( \mathbf { a } ^ { k + 1 } + \mathbf { u } ^ { k } \right)$   
8: Update $\gamma ^ { k + 1 } , \mathbf u ^ { k + 1 } , \pmb { \eta } ^ { k + 1 }$ using $( 1 7 ) \ – ( 1 9 )$   
9: if primal residuals satisfy their tolerances then   
10: break   
11: end if   
12: end for   
13: Reroll the proposal $\mathbf { a } ^ { K }$ and projected allocation $\mathbf { y } ^ { K }$ through the same joint world model   
14: Evaluate exact model-relative robustness:   
$\rho _ { \mathrm { e x e c } } = \rho _ { \mathrm { e x a c t } } \big ( \Phi , \mathcal { M } _ { \theta } ( \mathbf { y } ^ { K } ; G _ { t } ) \big )$   
15: Execute the first projected action: $\mathbf { a } _ { t } ^ { \star } = \mathbf { y } _ { t } ^ { K }$   
16: return $\mathbf { a } _ { t } ^ { \star } , \rho _ { \mathrm { e x e c } }$

## B.1 SIGNAL TEMPORAL LOGIC FOR EPIDEMIOLOGICAL RULES

We use differentiable STL robustness during optimization and exact robustness for model-relative verification of the projected rollout. In the experiments, the specification requires predicted hospital occupancy to remain below regional capacity over the planning horizon:

$$
\begin{array} { r } { \Phi ^ { i } = \bigsqcup _ { [ 0 , H ] } \left( \widehat { H } _ { t + h } ^ { i } \leq H _ { \mathrm { c a p } } ^ { i } \right) . } \end{array}\tag{21}
$$

A positive exact robustness score certifies satisfaction only on the learned rollout; it is not a guarantee for the unknown environment.

## C EXPERIMENTAL SETTINGS

## C.1 REAL-WORLD DATA SOURCES AND PREPROCESSING

We construct a weekly U.S. state-level panel from the following sources: (1) Epidemic burden and hospital resources are obtained from the HHS COVID-19 Reported Patient Impact and Hospital Capacity by State Timeseries, which provides new COVID-19 admissions, inpatient and ICU occupancy, staffed beds, and ICU capacity. (2) Reported infections and deaths are obtained from the CDC COVID-19 state surveillance datasets; these measure reported cases rather than latent infections and are treated as noisy observations. (3) Vaccination is drawn from the CDC COVID-19 Vaccinations in the United States, Jurisdiction dataset, including doses delivered and administered, primary-series completion, and booster coverage. (4) Policy interventions are taken from the COVID-19 U.S. State Policy Database (CUSP) (GitHub), which records state-level mask requirements, gathering limits, stay-at-home orders, school and business closures, emergency declarations, and economic-support policies. We aggregate active mitigation policies into a normalized NPI-intensity index and use the economic-policy fields as fiscal-support indicators. (5) Interregional mobility is obtained from Advan Patterns+ (data dictionary), whose visitor origins are aggregated into directed state-to-state flows defining the time-varying graph $G _ { t }$ . (6) State populations are obtained from the U.S. Census Bureau’s 2021 Population Estimates Program and are used to calculate per-100K outcomes and population-weighted allocations.

## C.2 BASELINE METHODS

We organize the baselines by the capability being evaluated.

World-model baselines. Persistence repeats the latest observation, whereas climatology predicts the training-set mean. Ridge+action and VARX(1)+action are deterministic linear predictors conditioned on regional interventions. Action-LSTM is a recurrent sequence model conditioned on the joint action vector Hochreiter & Schmidhuber (1997). GF-RSSM (no graph) retains the recurrent latent-state architecture but removes graph attention, isolating the contribution of mobility-based message passing. All forecasting models use the same train/test split and are evaluated in the same observation units and rollout protocol.

Planning baselines. No intervention applies zero intervention throughout the horizon. Bestfeasible constant selects a time-invariant NPI level by grid search under the same budget and evaluation objective. Uniform divides each shared resource equally across regions, while incidence-weighted allocates resources in proportion to current reported incidence. Greedy assigns resources according to the immediate predicted health benefit without multi-step optimization. Centralized MPC jointly optimizes all regional actions through a common finite-horizon objective, whereas independent MPC optimizes each region without cross-region coordination. MPC-SEIR plans with access to the simulator’s compartmental state and fixed epidemiological parameters and is therefore an informed simulator-based comparator rather than a deployable real-data method. Standard ADMM retains only the global resource-consensus variable and projection of conventional distributed ADMM Boyd et al. (2011a). D-ADMM provides a distributed optimization baseline without EpiMind’s learned graph coordination. The graph-free ablation removes mobility coupling from EpiMind, while independent planning removes both graph coupling and cross-region coordination. Region- and timeshuffled controls preserve intervention burden while disrupting spatial or temporal allocation.

Policy-learning baselines. We compare with PPO Schulman et al. (2017), MAPPO Yu et al. (2022), hierarchical reinforcement learning (HRL), and adapted implementations of EpidRLearn and EpiPolicy-RL. These methods are trained under the same action bounds and resource budgets; their outputs are passed through the same final projection used for EpiMind. Because the adapted epidemic-policy baselines do not use the authors’ original implementations, we treat them as representative algorithmic adaptations rather than exact reproductions of published results.

Oracle reference. Oracle dynamics uses the same planning interface as EpiMind but replaces the learned rollout with the hidden simulator dynamics. It measures the effect of dynamics-model error and is reported as an idealized reference, not as a deployable baseline.

## C.3 PARAMETER TABLE

Table A2 summarizes the epidemiological, intervention, observation, world-model, planner, budget, objective, training, and evaluation parameters used in the experiments, together with their values and supporting references or implementation rationale.

Table A2: Parameter inventory.
<table><tr><td>Symbol</td><td>Description</td><td>Value</td><td>Reference / Justification</td></tr><tr><td colspan="4">Epidemiological</td></tr><tr><td> $\beta$ </td><td>Transmission rate</td><td>0.25/day</td><td>Park et al. (2020);  $R _ { 0 } = \beta / \gamma = 2 . 5$ </td></tr><tr><td>σ</td><td>E→I rate</td><td>0.25/day (4 d)</td><td>Li et al. (2020); Lauer et al. (2020)</td></tr><tr><td>γ</td><td>I→R rate</td><td>0.1/day (10 d)</td><td>He et al. (2020)</td></tr><tr><td> $\delta _ { H }$ </td><td>Hospitalization rate</td><td>0.15</td><td>Garg et ai. (2020) (pre-Omicron)</td></tr><tr><td> $\delta _ { D }$ </td><td>Base death rate</td><td>0.01</td><td>Meyerowitz-Katz &amp; Merone (2020)</td></tr><tr><td colspan="4">Intervention effects</td></tr><tr><td>NPI β-red.</td><td>Max local NPI effect</td><td>60%</td><td>Brauner et al. (2021): 13–77%</td></tr><tr><td>NPI import red.</td><td>Max effect on imported force</td><td>40%</td><td>Travel restriction is partial</td></tr><tr><td>FOI split</td><td>Local : imported weight</td><td>0.7:0.3</td><td>Mobility coupling strength</td></tr><tr><td>Vacc. uptake</td><td>S → R rate at full allocation</td><td>0.01/day of S</td><td>Vaccinated S move directly to R</td></tr><tr><td>Fiscal scale</td><td>Funding→compliance</td><td>0.3</td><td>Hale et al. (2021)</td></tr><tr><td colspan="4">Capacity and overflow</td></tr><tr><td>LOS</td><td>Hospital length of stay</td><td>7 days</td><td>Converts incidence to occupancy</td></tr><tr><td>Mmax</td><td>Max overflow mortality mult.</td><td>5.0</td><td> $\delta _ { D } ^ { \mathrm { e f f } } = \operatorname* { m i n } ( \delta _ { D } m , 1 )$ </td></tr><tr><td>Cbase , Csurge</td><td>Capacity fractions of pop.</td><td>0.001, 0.002</td><td>Baseline plus surge at full allocation</td></tr><tr><td colspan="4">Observation model</td></tr><tr><td>σobs</td><td>Measurement noise std</td><td>0.005</td><td>Additive Gaussian on all channels</td></tr><tr><td>Pdetect</td><td>Base detection prob</td><td>0.7</td><td>Wu et al. (2020): 14–86%</td></tr><tr><td>NPI det. boost</td><td>Testing scales with NPI</td><td>+0.2, [0.3, 1]</td><td>Bilinski et al. (2021)</td></tr><tr><td>Tdelay</td><td>Reporting delay</td><td>1 week</td><td>Infection-to-confirmation lag</td></tr><tr><td colspan="4">Graph world model (GF-RSSM)</td></tr><tr><td>db, dz, dc</td><td>Belief / state / context</td><td>64 / 16 / 32</td><td>Hafner et al. (2025); 5-region problem</td></tr><tr><td>Nheads</td><td>Graph attention heads</td><td>4</td><td></td></tr><tr><td> $d _ { o }$ </td><td>Observation channels</td><td>7</td><td></td></tr><tr><td>σmin</td><td>Min prior/posterior std</td><td>0.01</td><td></td></tr><tr><td colspan="4">GT-ADMM coordinator</td></tr><tr><td>ρe, ρg</td><td>Edge / global penalties</td><td>1.0</td><td>Boyd et al. (2011b) (ADMM default)</td></tr><tr><td> $\sigma _ { \mathrm { p r o x } }$ </td><td>Proximal weight</td><td>0.1</td><td>Wang &amp; Banerjee (2019)</td></tr><tr><td>Kmax</td><td>Max ADMM iterations</td><td>15</td><td></td></tr><tr><td>inner steps</td><td>Gradient steps per x-step</td><td>3</td><td></td></tr><tr><td>ηx</td><td>x-step learning rate</td><td>0.02</td><td></td></tr><tr><td>βSTL</td><td>Smooth-STL temperature</td><td>0.5</td><td>Smooth for gradients</td></tr><tr><td colspan="4">Shared budgets</td></tr><tr><td> $B _ { \mathrm { v a c } } , B _ { \mathrm { h o s p } } , B _ { \mathrm { f i s c } }$ </td><td>Resource budgets</td><td>3.0, 2.5, 3.5</td><td>0.6 / 0.5 / 0.7 per region</td></tr><tr><td>BNPI</td><td>Shared NPI budget</td><td>3.0</td><td>0.6 per region</td></tr><tr><td colspan="4">Objective</td></tr><tr><td>λNPI</td><td>NPI cost weight</td><td>0.1/3/{3,10,30}</td><td>Track A / Track B / synthetic sweep</td></tr><tr><td> $\lambda _ { \mathrm { { r e s } } }$ </td><td>Resource cost weight</td><td>0.02</td><td>Prices the budget channels</td></tr><tr><td colspan="4">Training and evaluation</td></tr><tr><td>Hrollout</td><td>Planning rollout horizon</td><td>4 steps</td><td>CDC 4–6 week (Howerton et al., 2023)</td></tr><tr><td>T</td><td>Episode length Train / val / test</td><td>26 weeks 128 / 32 / 32</td><td>Chronological split</td></tr><tr><td>Episodes Epochs</td><td>Max, with early stopping</td><td>1000</td><td>Validation-based selection</td></tr><tr><td>Seeds</td><td>Training / evaluation</td><td>3/5</td><td></td></tr></table>

## D RESULTS

## D.1 WORLD MODEL EVALUATION

Figure A2 separates one-step admission accuracy from cumulative open-loop error. The full GF-RSSM achieves the lowest admission MAE and the lowest cumulative error at $H = 5$ . Its advantage narrows by $H = 1 0$ , and the action-conditioned LSTM performs better at $H \ : = \ : 2 0$ , indicating greater long-horizon drift in the GF-RSSM. Removing graph attention consistently increases rollout error, while all learned models outperform the persistence and climatology references.

![](images/b2b8f52f33fadddee6c30b886075c24ae29db20dbf8de0083eeddf58c8370a88.jpg)

![](images/cbb024179badf93a418f09043d2289c2a61370141061165820a6767906d3964a.jpg)  
Figure A2: Extended world-model evaluation on the synthetic benchmark. (a) Held-out admission MAE for learned architectures and statistical baselines under the same evaluation protocol. Error bars for learned models show $\mathrm { m e a n } \pm \mathrm { s . d }$ . over three training seeds. (b) Relative cumulative admission error over open-loop rollout horizons $H \in \{ 5 , 1 0 , 2 0 \}$ , shown on a logarithmic scale.

## D.2 ROBUSTNESS ANALYSIS

![](images/fe8f18ff7c798bed14b39158a585c1096eda1f7689d1cab40074dc4f94f4ddd8.jpg)  
Figure A3: Sensitivity to operational and model perturbations. Changes in cumulative hospitalizations relative to each seed-matched baseline, reported as mean $\pm \ \mathrm { s . d . }$ The shaded band shows the largest within-condition standard deviation among the operational perturbations. Graph noise, masked regional observations, reporting delays, a mid-horizon budget cut, and regional noncompliance remain within this descriptive variability band. Epidemiological model mismatch pro duces the only substantially larger mean degradation.

Figure A3 evaluates sensitivity to graph noise, missing or delayed observations, a mid-horizon budget cut, regional non-compliance, and epidemiological model mismatch. Across five paired seeds, the operational perturbations change cumulative hospitalizations by less than 1% on average and remain within seed-level variability. Model mismatch produces the largest degradation $( + 6 . 4 7 \% )$ identifying misspecified epidemic dynamics as the dominant tested failure mode. Given the limited number of seeds, these results are descriptive rather than evidence of statistical equivalence.

## D.3 PLANNING AND CALIBRATION ANALYSIS

Table A3 expands the RQ2 regret analysis across intervention costs λ and planning horizons H. Learned- and oracle-dynamics planners use the same optimizer and per-cell best-constant comparator. The model effect is therefore the paired regret difference $R _ { \mathrm { l e a r n e d } } - R _ { \mathrm { o r a c l e } } .$ , isolating the change produced by replacing the learned rollout with the simulator dynamics.

Table A3: Learned- versus oracle-dynamics planning regret. Values are mean $\pm \ : \mathrm { s . d . }$
<table><tr><td rowspan="2">λ H</td><td rowspan="2"></td><td colspan="2">Regret↓</td><td rowspan="2">Model effect ↓</td><td rowspan="2"> $p ^ { \dagger }$ </td></tr><tr><td>Learned</td><td>Oracle</td></tr><tr><td>3</td><td>4</td><td> $3 2 . 5 \pm 3 2 . 8$ </td><td> $2 7 . 8 \pm 3 . 2$ </td><td> $4 . 7 \pm 3 4 . 6$ </td><td>1.000</td></tr><tr><td>3</td><td>8</td><td> $3 8 . 2 \pm 5 0 . 0$ </td><td> $1 7 . 6 \pm 0 . 7$ </td><td> $2 0 . 5 \pm 5 0 . 3$ </td><td>1.000</td></tr><tr><td>3</td><td>12</td><td> $3 2 . 5 \pm 3 4 . 2$ </td><td> $2 0 . 6 \pm 0 . 2$ </td><td> $1 1 . 9 \pm 3 4 . 2$ </td><td>1.000</td></tr><tr><td>10</td><td>4</td><td> $2 1 7 . 0 \pm 2 2 9 . 3$ </td><td> $1 0 3 . 2 \pm { 3 . 8 }$ </td><td> $1 1 3 . 9 \pm 2 3 1 . 3$ </td><td>1.000</td></tr><tr><td>10</td><td>8</td><td> $1 8 4 . 9 \pm 2 0 1 . 1$ </td><td> $7 9 . 3 \pm 0 . 4$ </td><td> $1 0 5 . 6 \pm 2 0 0 . 9$ </td><td>1.000</td></tr><tr><td>10</td><td>12</td><td> $1 8 6 . 4 \pm 1 9 8 . 0$ </td><td> $7 8 . 2 \pm 3 . 1$ </td><td> $1 0 8 . 2 \pm 1 9 9 . 7$ </td><td>0.625</td></tr><tr><td>30</td><td>4</td><td> $5 9 3 . 9 \pm 5 1 7 . 6$ </td><td> $3 2 8 . 6 \pm 6 . 8$ </td><td> $2 6 5 . 4 \pm 5 1 3 . 3$ </td><td>0.625</td></tr><tr><td>30</td><td>8</td><td> $4 5 7 . 9 \pm 4 0 1 . 8$ </td><td> $2 3 2 . 7 \pm 1 5 . 0$ </td><td> $2 2 5 . 2 \pm 3 9 2 . 6$ </td><td>0.625</td></tr><tr><td>30</td><td>12</td><td> $4 5 9 . 1 \pm 4 4 0 . 5$ </td><td> $2 3 5 . 9 \pm 2 7 . 9$ </td><td> $2 2 3 . 2 \pm 4 2 4 . 9$ </td><td>0.625</td></tr></table>

† p is an exact two-sided sign test.

The learned planner has higher mean regret in every configuration, but its variation across cells is large: the model-effect standard deviation exceeds its mean in every row, and no sign test is significant. The grid therefore suggests a rollout-model contribution to regret but does not establish its magnitude at this sample size.

Table A4 tests whether the dedicated hospitalization head and its calibration terms improve planning while holding the planner and comparator fixed. Because all variants are scored against the same per-cell best-constant policy, comparisons are paired.

Table A4: Calibration ablation. Values are mean ± s.d. over n=9 cells.
<table><tr><td rowspan="2">λ</td><td colspan="3">Regret↓</td><td colspan="2">vs. decoder</td><td colspan="2">vs. uncalibrated</td></tr><tr><td>Calibrated</td><td>Shared dec.</td><td>Uncalib.</td><td>cells</td><td>p</td><td>cells</td><td>p</td></tr><tr><td>3</td><td> $5 6 . 5 \pm 5 0 . 5$ </td><td> $7 2 . 8 \pm 8 5 . 4$ </td><td> $6 8 . 8 \pm 4 2 . 2$ </td><td>5/9</td><td>1.000</td><td>7/9</td><td>0.180</td></tr><tr><td>10</td><td> $2 5 2 . 8 \pm 2 4 2 . 3$ </td><td> $7 0 5 . 3 \pm 5 8 6 . 5$ </td><td> $3 6 4 . 9 \pm 3 0 9 . 3$ </td><td>8/9</td><td>0.039</td><td>9/9</td><td>0.004</td></tr><tr><td>30</td><td> $6 4 1 . 4 \pm 4 7 4 . 1$ </td><td> $2 4 0 1 . 9 \pm 9 0 6 . 1$ </td><td> $8 8 2 . 1 \pm 6 3 4 . 7$ </td><td>9/9</td><td>0.004</td><td>7/9</td><td>0.180</td></tr></table>

At λ=10, the calibrated head reduces mean regret by 64.2% relative to the shared decoder and by 30.7% relative to the uncalibrated head, with both paired tests significant. $\mathrm { A t } \lambda { = } 3 0$ , it reduces mean regret by 73.3% relative to the shared decoder, but the additional benefit over the uncalibrated head is not significant. Neither contrast is established at λ=3. These results support the dedicated head and calibration terms at intermediate intervention costs, while showing that aggregate mean ratios should not be interpreted as uniform per-cell gains.

## D.4 COORDINATION ABLATION

We isolate the contributions of neighbor consensus γ, predicted spillover sensitivity η, and global resource consensus $\mu$ under a matched-burden protocol. Every variant follows the full model’s stepwise NPI-burden trajectory, so differences reflect where and when interventions are allocated rather than total NPI use.

Full EpiMind achieves the lowest cumulative admissions (656.81/100K), but the matched-burden gains are limited. Removing global resource consensus (µ) produces the largest individual degradation $( + 0 . 9 4 \% )$ , followed by removing spillover sensitivity $( \eta , + 0 . 5 4 \% )$ and neighbor consensus (γ, $+ 0 . 4 7 \% )$ . Disabling all three signals increases admissions by 0.88%. Thus, the coordination components provide complementary but incremental improvements once intervention burden is controlled;

Table A5: Matched-burden coordination ablation. Simulator-realized outcomes on the synthetic benchmark. All variants follow the reference configuration’s stepwise NPI-burden trajectory. Here, γ denotes neighbor consensus, η spillover sensitivity, and µ global resource consensus. ∆ is the percentage change relative to the full model; lower is better.
<table><tr><td>Variant</td><td>γ</td><td>η</td><td>µ</td><td>Cum. adm./100K ↓</td><td>∆(%)</td></tr><tr><td>Graph-free</td><td>一</td><td></td><td>一</td><td>662.57</td><td>+0.88</td></tr><tr><td>No edge (γ off)</td><td></td><td>√</td><td>√</td><td>659.87</td><td>+0.47</td></tr><tr><td>No spillover (η off)</td><td>√</td><td>一</td><td>√</td><td>660.35</td><td>+0.54</td></tr><tr><td>No global (µ off)</td><td>」</td><td>√</td><td>一</td><td>662.99</td><td>+0.94</td></tr><tr><td>EpiMind</td><td>√</td><td>√</td><td>√</td><td>656.81</td><td></td></tr></table>

the substantially larger differences observed without burden matching partly reflect variation in total intervention effort rather than coordination alone.

## D.5 REAL MOBILITY GRAPH

Figure A4 shows a January 2021 snapshot of the row-normalized Advan mobility graph for ten selected high-flow U.S. states. Rows denote origins and columns denote destinations. The matrix exhibits directed, heterogeneous connectivity, with several dominant interstate links and long-range flows involving California, Florida, and Texas. EpiMind uses these flows as graph-edge weights, allowing the GF-RSSM to learn nonuniform neighbor contributions rather than assuming homogeneous regional mixing.

![](images/dd32e327fe3a05fa58b49562101a470f6d90cd3954c6c8f34e2cb24b72382b64.jpg)  
Figure A4: Real interstate mobility graph. Row-normalized Advan device-mobility flows among ten selected high-flow U.S. states in January 2021. Rows denote origins, columns denote destinations, and color indicates each destination’s share of an origin’s outgoing travel among the displayed states. The asymmetric, nonuniform matrix provides mobility-edge weights to the graph world model.

## D.6 REAL-CONTEXT PREDICTION AND PROJECTION

Figure A5 examines whether forecast-capable models provide both accurate predictions and useful planning signals. EpiMind attains the lowest forecast RMSE (4.1) and projects 9.9 weekly admissions per 100K, compared with 11.8 under the historical-policy reference. Several baselines also project admissions below this reference, but with substantially larger forecast errors. These results distinguish predictive fidelity from projected policy quality; because the proposed policies were not executed, the vertical axis represents model-relative outcomes rather than realized policy effects.

Figure A6 illustrates EpiMind’s shared-model behavior across heterogeneous regional trajectories. The historical-policy forecast tracks the timing of major admission waves, although peak magnitude is imperfectly calibrated in several states. The planner rollouts produce state-specific trajectories

Observed (ground truth)

FL  
(a) Policy-conditioned rollout  
![](images/19190f0f3b0dfd48dc967bb81b792cdfd306c380bb7f3d3e4712fffd30b2011e.jpg)

(b) Two-gate planner evaluation  
![](images/9b92ea10482559b39a45471af0c25916463e6f46574a1951bb2b22f21f5aafa1.jpg)

Figure A5: Texas real-context evaluation. (a) Observed admissions, the held-out forecast under historical actions, and the model-relative rollout under EpiMind’s proposed policy. The inset enlarges weeks 50–62; shading denotes the predicted difference between the historical- and proposed-policy rollouts. (b) Forecast RMSE versus model-relative admissions projected under each model’s optimized policy. Dashed lines mark the observed-policy admission mean and the selected forecast-error reference. Lower values are preferable on both axes.  
![](images/fd5e5988c3de64d1053a4eba56b67a2f05edbdf578f5c9976a637dd57143c7b7.jpg)

![](images/1c142fdfc666e0a67c13e318e78bc7fd1be831fd5d1a388dd9b2a015296152db.jpg)

![](images/8e20450b1d6183d8d39f00debfce6a76d7466bfd86627bb6c7f0b8aa0f30969b.jpg)

![](images/31f0165f2d148c3f1c5bdd24e4741fbc3e4097509b35bc197a36c1703c555b64.jpg)

![](images/c5509496c658481ce68d927c0060fe8700e939b959ff3a73a7277badd642fd0e.jpg)

![](images/6ef8c278c08e5841045401625a17c9dbc46b3530de4704e4762a903efba10224.jpg)

![](images/ea7c0d9be948bd294de508a122c7b05c3b08737bcd9c74c57122759f55191bb3.jpg)

![](images/fa4f90076028a3a4265f0f075ead8ca5bf2761dfecd6d7f9835f36b3338f5583.jpg)

![](images/9175dcd4afced8685f3c417c005fbf7b1c711c3329094963715874c7e432921a.jpg)

![](images/5cdff924c0164df76a9a92aab49b628abba885bf440dcd4df275996739c41775.jpg)  
---. Projected (planner policy)

Figure A6: Multi-state forecasting and planner projections. Observed weekly admissions (gray), forecasts under the recorded policy (blue), and model-relative rollouts under EpiMind’s proposed policy (red) for ten mobility-connected U.S. states. The vertical dashed line marks the training cutoff. Planner projections represent unexecuted counterfactuals, not observed outcomes.

from the same mobility-coupled model. Because these policies were not executed, the projected reductions are interpreted as model-relative policy comparisons rather than causal effects.