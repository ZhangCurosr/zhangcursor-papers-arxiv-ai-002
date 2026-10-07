# PARALLEL PREDICTIVE WORLD MODELS FOR ACCURATE AND EFFICIENT LONG-HORIZON PLANNING

Wanjin Feng<sup>1</sup> Baobin Zhang<sup>2,\*</sup> Ao Yu<sup>3</sup> Shibo Feng<sup>4</sup> Xi Wang<sup>2</sup> Xingyu Gao<sup>2,\*</sup>

<sup>1</sup>Tsinghua University

<sup>2</sup>Institute of Microelectronics, Chinese Academy of Sciences <sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>4</sup>Nanyang Technological University, Singapore <sup>\*</sup>Corresponding authors

## ABSTRACT

Long-horizon world-model planning typically relies on autoregressive rollouts, where predicted states are repeatedly fed back into the model. This preserves temporal structure but creates a horizon-length sequential path and exposes later predictions to recursive decoded-state feedback. We introduce Parallel Predictive World Models (PPWM), which predict a finite-horizon trajectory in parallel while retaining causal interaction among future representations. Each horizon is conditioned on its causal action prefix, and future representations interact before decoding, separating temporal causality from state-by-state output recursion. We formalize this distinction by viewing autoregressive rollout as a causal trajectory map and identifying the decoded-state feedback pathway removed by PPWM. Across four visual-control tasks, PPWM achieves the lowest long-horizon prediction error and the highest Cross-Entropy Method (CEM) simulator success among the evaluated predictive interfaces. Meanwhile, PPWM achieves more than a 3× average CEM planning speedup over the autoregressive LeWM baseline. These results suggest that accurate and efficient long-horizon world-model planning does not require state-by-state autoregression, but can instead be achieved through parallel causal trajectory prediction.

## 1 INTRODUCTION

World models let agents evaluate candidate behaviors through imagined futures before acting. Effective planning often requires sufficiently long futures, because the consequences of early actions may emerge only many steps later. Learned dynamics have therefore been paired with trajectory optimization and model-predictive control (Chua et al., 2018; Hansen et al., 2022; 2024), while recent sequence and visual world models extend this paradigm to high-dimensional observations and offline planning (Micheli et al., 2022; Zhou et al., 2024; Sobal et al., 2026). As horizons grow, world-model planning must preserve predictive accuracy while avoiding the increasing sequential cost of state-by-state rollout.

Most world models expose future dynamics through recursive local prediction. Distant states require a horizon-dependent chain of model evaluations, and each predicted state becomes part of the next model input, allowing prediction error or model misspecification to propagate through the rollout (Hafner et al., 2019; Janner et al., 2019; Gao and Xu, 2026; Somalwar et al., 2025). Direct multi-step prediction provides an alternative factorization that can mitigate recursive bias under partial observability or model misspecification (Somalwar et al., 2025; 2026). Crucially, downstream planners inherit the same temporal factorization when evaluating long trajectories or propagating the influence of early actions to distant outcomes.

Recent work reduces this dependency from two sides. Prediction-side methods directly parameterize longer futures (Mishra et al., 2017). Most closely, Fast-LeWM accumulates action effects with causal action prefixes and predicts multiple horizons in parallel (Gao and Xu, 2026), but does not explicitly couple future-state representations before decoding. Planning-side methods instead lift intermediate states to expose trajectory-level parallelism (Rybkin et al., 2021; Ziakas et al., 2026; Wu et al., 2026). GRASP performs parallel iterative optimization with virtual states while avoiding long chains of state-input Jacobians through an action-gradient formulation (Psenka et al., 2026). This leaves a concrete design problem: preserve trajectory causality without forcing prediction and planning through a state-by-state temporal chain.

![](images/4bef221e2383608bd15bb80018c6f26a55d2ee15afae05b29ea02bbb6b043358.jpg)  
Figure 1: Comparison of temporal interfaces for world models. (a) Autoregressive (AR) world models predict one step at a time and feed the predicted state back as input, forming a long temporal chain. (b) Fast-LeWM uses a causal action prefix to predict multiple future states in parallel, without explicit interaction among future-state representations. (c) Our Parallel Predictive World Model (PPWM) predicts a causal finite-horizon trajectory in parallel, allowing interaction among future representations while removing output-state recursion within a span.

We introduce Parallel Predictive World Models (PPWM) for long-horizon planning. Given recent context and an action span, PPWM predicts a causally structured finite-horizon trajectory directly, without output-state recursion inside the span and with causal interaction among future representations. This design induces a parallel trajectory interface that can be queried by population-based search and differentiated jointly for lightweight trajectory-wide refinement. In closed-loop control, the refined plan is repeatedly re-anchored by fresh observations rather than optimized once and executed open loop. By removing recursive decoded-state feedback within each span, PPWM targets both long-horizon predictive accuracy and planning efficiency: it removes one recursive error-propagation <sup>⋯</sup> path and shortens the sequential dependency exposed to planning. We evaluate these effects through prediction accuracy, simulator planning, closed-loop refinement, architectural generality, and wallclock scaling.

Our contributions are:

• Parallel Predictive World Models. We introduce PPWM, which replaces state-by-state decoded prediction with parallel causal trajectory prediction while preserving interaction among future representations.

• Causal trajectory factorization for planning. We characterize autoregressive rollout as a causal finite-horizon map and remove its decoded-state feedback pathway, reducing sequential model evaluation while exposing the full trajectory to planning.

• Accurate and efficient long-horizon planning. Across four visual-control tasks, PPWM achieves the lowest long-horizon prediction error and strongest CEM simulator performance, with over 3× lower average planning latency than autoregressive LeWM.

## 2 RELATED WORK

World models for planning. World models support control by coupling learned dynamics with test-time optimization. PlaNet plans in latent space from pixels (Hafner et al., 2019). TD-MPC and TD-MPC2 combine learned dynamics with local trajectory optimization (Hansen et al., 2022; 2024), while IRIS, DINO-WM, PLDM, and LeWorldModel extend planning to expressive sequence models, offline visual data, and predictive representations (Micheli et al., 2022; Zhou et al., 2024; Sobal et al., 2026; Maes et al., 2026). However, most of these models still expose long-horizon futures through recursively composed local transitions, making both prediction and planning increasingly dependent on rollout depth.

Multi-step and parallel predictive dynamics. Long-horizon prediction can be improved either through multi-step objectives or by changing the predictive interface. Latent overshooting improves consistency while retaining recursive dynamics (Hafner et al., 2019), whereas Temporal Segment Models directly parameterize future trajectory segments (Mishra et al., 2017). Direct multi-step prediction can reduce recursive bias under partial observability or misspecification, while wellspecified one-step models may remain statistically preferable (Somalwar et al., 2025; 2026). Most closely related, FAST-LEWM accumulates action effects with causal action prefixes and predicts multiple horizons in parallel (Gao and Xu, 2026). Existing approaches therefore either retain recursive dynamics or remove state recursion without explicitly modeling causal interaction among future representations.

Planning and trajectory optimization with learned world models. Sampling-based methods such as CEM, MPPI, and PETS optimize actions through repeated model evaluation (De Boer et al., 2005; Williams et al., 2017; Chua et al., 2018), while differentiable planners exploit learned-model geometry or trajectory-level objectives for local refinement (Georgiev et al., 2025; Pham and Bera, 2026; Huang et al., 2026). Lifted methods instead optimize intermediate states explicitly: LatCo applies latent-space collocation (Rybkin et al., 2021), GRASP exposes parallel optimization with virtual states (Psenka et al., 2026), and GVP-WM jointly optimizes latent states and actions (Ziakas et al., 2026). Related control work uses direct multi-step prediction in structured state–control optimization (Wu et al., 2026). These planners improve trajectory optimization, but their computation and optimization structure remain constrained by the temporal factorization exposed by the underlying world model.

## 3 PROBLEM FORMULATION

We consider goal-conditioned control from visual observations. Let $\mathbf { } _ { o _ { t } }$ denote the observation, $\mathbf { \boldsymbol { a } } _ { t } \in \mathbb { A }$ the action, and $\pmb { z } _ { t } = \pmb { E } ( \pmb { o } _ { t } ) \in \mathbb { R } ^ { D }$ the learned representation used for prediction and planning. We denote by $\mathcal { C } _ { t }$ the context available to the world model at time $t ,$ such as recent latent states and executed actions. Given a goal latent $z _ { g } = E ( o _ { g } )$ and a planning horizon H, let

$$
A = ( a _ { t } , \ldots , a _ { t + H - 1 } ) , \qquad Z = ( z _ { t + 1 } , \ldots , z _ { t + H } )\tag{1}
$$

denote a candidate action sequence and its corresponding future trajectory. A world model predicts

$$
\widehat { \pmb { Z } } = \mathcal { P } _ { \theta } ( \mathcal { C } _ { t } , \pmb { A } ) ,\tag{2}
$$

and the planner seeks

$$
\pmb { A } ^ { \star } = \arg \operatorname* { m i n } _ { \pmb { A } \in \mathbb { A } ^ { H } } J _ { \mathrm { t a s k } } \left( \mathcal { P } _ { \theta } ( \mathcal { C } _ { t } , \pmb { A } ) , \pmb { A } ; z _ { g } \right) .\tag{3}
$$

A conventional autoregressive world model predicts future states recursively. Longer-horizon predictions are obtained by repeatedly feeding predicted states back into the model,

$$
\widehat { \sf z } _ { t + j + 1 } = f _ { \theta } ( \widehat { \sf z } _ { t + j } , { \pmb a } _ { t + j } ) , \qquad \widehat { \sf z } _ { t } = \boldsymbol { z } _ { t } , \qquad j = 0 , \ldots , H - 1 .\tag{4}
$$

This recursive rollout has two important consequences for long-horizon planning. First, predicted states are repeatedly reused as inputs, so local prediction errors can propagate over the rollout. Second, each future state depends on the preceding prediction, forming a chain of H sequentially dependent transition steps and limiting parallelism within each trajectory.

## 4 PARALLEL PREDICTIVE WORLD MODELS FOR LONG-HORIZON PLANNING

PPWM is designed for long-horizon prediction and planning by exposing a causally structured finite-horizon trajectory in a single forward pass, avoiding recursive decoded-state feedback and the horizon-length sequential dependency of autoregressive rollout. Figure 1 compares this design with autoregressive rollout and Fast-LeWM-style prefix-conditioned direct prediction. Each prediction horizon is represented by a query conditioned on its causal action prefix, and the future queries interact through a shared causal trajectory predictor. The same finite-horizon interface exposes all future variables jointly, which also enables lightweight trajectory-wide refinement.

## 4.1 PARALLEL FUTURE-STATE PREDICTION

Given the current context $\mathcal { C } _ { t }$ and an action span of length K, PPWM predicts

$$
\Phi _ { \theta } ( \mathcal { C } _ { t } , A _ { t : t + K - 1 } ) = \left( \widehat { z } _ { t + 1 } , \dots , \widehat { z } _ { t + K } \right) .\tag{5}
$$

All K future states are produced in a single forward pass. Unlike autoregressive rollout, $\widehat { z } _ { t + k }$ is not fed back as the input used to generate $\widehat { z } _ { t + k + 1 }$ . Instead, each horizon is conditioned on the actions that causally precede it, while temporal dependence among future states is retained through hidden representations.

For each future horizon, a shared causal prefix encoder produces an action-prefix representation

$$
( { \pmb p } _ { t + 1 } , \dots , { \pmb p } _ { t + K } ) = E _ { \psi } ( \mathcal { C } _ { t } , { \pmb a } _ { t : t + K - 1 } ; M _ { \mathrm { c a u s a l } } ) ,\tag{6}
$$

where $\pmb { p } _ { t + k }$ depends only on the current context and actions up to $\mathbf { \pmb { a } } _ { t + k - 1 }$ . We then construct

$$
\begin{array} { r } { q _ { 1 } = z _ { t } , \qquad q _ { k } = z _ { t } + e _ { k } + W _ { p } p _ { t + k } , \qquad k = 2 , \ldots , K , } \end{array}\tag{7}
$$

where $e _ { k }$ identifies the prediction horizon and $W _ { p } p _ { t + k }$ represents the accumulated effect of the causal action prefix.

## 4.2 CAUSAL TRAJECTORY CONSTRUCTION

The future queries are arranged in temporal order and jointly processed by a shared causal trajectory predictor. Let ${ \pmb u } _ { t + k } ^ { ( 0 ) } = { \pmb q } _ { k }$ denote the input representation at horizon k. The predictor contains L causal Transformer layers, and each layer updates the representation at horizon k using only the current and preceding future positions.

$$
\pmb { u } _ { t + k } ^ { ( \ell ) } = \pmb { \mathcal { B } } _ { \theta } ^ { ( \ell ) } \left( \mathcal { C } _ { t } , \pmb { u } _ { t + 1 : t + k } ^ { ( \ell - 1 ) } , \pmb { A } _ { t : t + k - 1 } \right) , \qquad k = 1 , \dots , K ,\tag{8}
$$

where $B _ { \theta } ^ { ( \ell ) }$ denotes the ℓ-th causal Transformer block. The causal structure allows information from earlier future horizons to be progressively propagated and refined through the Transformer layers, while preventing access to later future positions or actions.

After the final layer, each future state is decoded from its corresponding representation.

$$
\widehat { \pmb z } _ { t + k } = F _ { \boldsymbol \theta } \left( \pmb u _ { t + k } ^ { ( L ) } \right) , \qquad k = 1 , \dots , K .\tag{9}
$$

The decoded state $\widehat { z } _ { t + k }$ is never fed back into the prediction of later horizons, so temporal dependence is carried through the causal hidden representations rather than through recursive state prediction.

This factorization distinguishes PPWM from Fast-LeWM-style prefix-conditioned direct prediction. As illustrated in Figure 1, both approaches remove recursive feedback of predicted states, but PPWM additionally preserves explicit interaction among future-state representations before decoding. PPWM therefore replaces autoregressive state feedback with causal interaction in latent space, retaining temporal structure across the predicted trajectory without restoring serial state recursion.

## 4.3 SELF-CONDITIONED LONG-HORIZON COMPOSITION

As our focus is the parallel predictive dynamics, we formulate the training objective directly in latent space and treat representation learning as orthogonal. All outputs within a span are directly

supervised by their corresponding future latent targets,

$$
\mathcal { L } _ { \mathrm { d i r e c t } } = \frac { 1 } { K D } \sum _ { k = 1 } ^ { K } \left\| \widehat { \boldsymbol { z } } _ { t + k } - \boldsymbol { z } _ { t + k } ^ { \mathrm { G T } } \right\| _ { 2 } ^ { 2 } .\tag{10}
$$

For horizons longer than $K$ , multiple PPWM spans are composed, so later spans receive modelgenerated rather than observed contexts. Training only from observed contexts would therefore create a mismatch between single-span training and long-horizon prediction. We address this with multi-anchor self-conditioned rollout (MAR) training. For a sampled restart anchor s, let $\widehat { \mathcal { C } } _ { t + s }$ denote the detached predicted context and let $R _ { s }$ be the remaining supervised horizon. We define

$$
\mathcal { L } _ { \mathrm { M A R } } = \mathbb { E } _ { s } \left[ \frac { 1 } { R _ { s } D } \sum _ { j = 1 } ^ { R _ { s } } \left. \Phi _ { \theta , j } \left( \mathrm { s g } [ \widehat { \mathcal { C } } _ { t + s } ] , A _ { t + s : t + s + R _ { s } - 1 } \right) - z _ { t + s + j } ^ { \mathrm { G T } } \right. _ { 2 } ^ { 2 } \right] ,\tag{11}
$$

where $\Phi _ { \theta , j }$ denotes the j-th prediction of the restarted span and $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operator. The final training objective is

$$
\mathcal { L } _ { \mathrm { P P W M } } = \mathcal { L } _ { \mathrm { d i r e c t } } + \lambda _ { \mathrm { M A R } } \mathcal { L } _ { \mathrm { M A R } } .\tag{12}
$$

The same finite-span predictor is therefore trained on the model-generated contexts encountered when spans are composed at test time. For a horizon $H > K$ , prediction remains parallel within each span and becomes sequential only across span boundaries.

## 4.4 PARALLEL TRAJECTORY REFINEMENT

The same finite-horizon interface can also be differentiated at trajectory level. Because PPWM exposes the complete predicted trajectory $\widehat { Z } ( A ) = \Phi _ { \theta } ( { \mathcal { C } } _ { t } , A )$ in one computational graph, the planner can optimize all future action and latent variables jointly instead of propagating updates through a decoded-state rollout. We use this property as a lightweight refinement of a plan produced by the Cross-Entropy Method (CEM).

Starting from the CEM plan $\pmb { A } _ { 0 } ,$ , we introduce an auxiliary latent trajectory Z and initialize

$$
{ \cal A } ^ { ( 0 ) } = { \cal A } _ { 0 } , \qquad { \cal Z } ^ { ( 0 ) } = \widehat { { \cal Z } } ( A _ { 0 } ) .\tag{13}
$$

The refinement couples the free latent trajectory to the trajectory induced by the current actions through

$$
\mathcal { L } _ { \mathrm { r e f } } ( A , Z ) = J _ { \mathrm { t a s k } } ( Z ) + \alpha J _ { \mathrm { t a s k } } ( \widehat { Z } ( A ) ) + \lambda _ { \mathrm { d y n } } \Vert Z - \widehat { Z } ( A ) \Vert _ { F } ^ { 2 } + \lambda _ { A } \Vert A - A _ { 0 } \Vert _ { F } ^ { 2 } .\tag{14}
$$

The first term is the task objective evaluated on the latent trajectory $z ,$ providing a direct optimization path toward the task goal in trajectory space. The executable-trajectory and consistency terms keep the update tied to model-predicted dynamics, while the final term regularizes actions around the CEM solution.

At refinement step m, gradients with respect to the complete action and latent trajectories are computed from the same objective and updated jointly,

$$
\begin{array} { r } { \boldsymbol A ^ { ( m + 1 ) } = \Pi _ { \boldsymbol B } \Big ( \boldsymbol A ^ { ( m ) } - \eta _ { \boldsymbol A } \nabla _ { \boldsymbol A } \mathcal L _ { \mathrm { r e f } } \Big ( \boldsymbol A ^ { ( m ) } , \boldsymbol Z ^ { ( m ) } \Big ) \Big ) , \boldsymbol Z ^ { ( m + 1 ) } = \boldsymbol Z ^ { ( m ) } - \eta _ { \boldsymbol Z } \nabla _ { \boldsymbol Z } \mathcal L _ { \mathrm { r e f } } \Big ( \boldsymbol A ^ { ( m ) } , \boldsymbol Z ^ { ( m ) } \Big ) } \end{array}\tag{15}
$$

where $\Pi _ { B }$ enforces a local feasible action region B around the CEM plan. Since PPWM produces all future positions within the span jointly, each refinement step updates the full action and latent trajectories without restoring a decoded-state rollout. After a small number of refinement steps, the action candidates are re-evaluated through PPWM and selected using the executable model trajectory,

$$
A ^ { \star } = \arg \operatorname* { m i n } _ { A \in \{ A _ { 0 } , A ^ { ( 1 ) } , \ldots , A ^ { ( R ) } \} } J _ { \mathrm { t a s k } } \Big ( \widehat { Z } ( A ) \Big ) .\tag{16}
$$

Keeping $A _ { 0 }$ in the candidate set provides a fallback under the fixed model score.

## 5 ANALYSIS OF TEMPORAL FACTORIZATION

PPWM preserves temporal dependence while changing its computational factorization. We first show that a finite autoregressive rollout is itself a causal trajectory map, so decoded-state recursion is a computational factorization rather than a requirement for temporal causality. We then characterize the decoded-state feedback pathway that is present in autoregressive rollout but absent from PPWM, while causal interaction among future hidden representations is retained.

## 5.1 AUTOREGRESSIVE ROLLOUT AS A CAUSAL TRAJECTORY MAP

Consider a deterministic autoregressive predictor over a finite horizon $K$

$$
\widehat { \boldsymbol { z } } _ { t + 1 } = g _ { 1 } ( \boldsymbol { \mathcal { C } } _ { t } , \boldsymbol { a } _ { t } ) , \qquad \widehat { \boldsymbol { z } } _ { t + k } = g _ { k } ( \boldsymbol { \mathcal { C } } _ { t } , \widehat { \boldsymbol { z } } _ { t + 1 : t + k - 1 } , \boldsymbol { a } _ { t : t + k - 1 } ) , \quad k = 2 , \ldots , K .\tag{17}
$$

Although this computation is expressed recursively, its finite-horizon input–output map need not be written in recursive form.

Theorem 1 (Equivalent causal trajectory factorization). For every finite deterministic autoregressive predictor of the form in Eq. (17), there exist causal functions $G _ { 1 } , \ldots , G _ { K }$ such that

$$
\widehat { z } _ { t + k } = G _ { k } ( \mathcal { C } _ { t } , \pmb { a } _ { t : t + k - 1 } ) , \qquad k = 1 , \dots , K .\tag{18}
$$

Hence the complete autoregressive rollout is exactly a causal finite-horizon map from the current context and action sequence to the predicted trajectory.

Theorem 1 separates temporal causality from a particular computational factorization. It shows that decoded-state recursion is not required by the causal input–output structure itself. PPWM realizes this structure while carrying cross-horizon dependence through causal hidden representations.

## 5.2 DECODED-STATE FEEDBACK PROPAGATION

The equivalent input–output form in Theorem 1 does not imply an equivalent computational graph. For the canonical autoregressive transition

$$
\widehat { \pmb { z } } _ { t + m + 1 } = f _ { \theta , m } ( \widehat { \pmb { z } } _ { t + m } , \pmb { a } _ { t + m } ) ,\tag{19}
$$

every decoded prediction is reused as the input to the next transition. PPWM instead performs causal interaction among future hidden representations and decodes each future latent only after this interaction.

Theorem 2 (Elimination of recursive decoded-state feedback). Assume the autoregressive transitions in Eq. (19) are differentiable, and fix the action sequence. If the decoded state at horizon j is perturbed as $\widehat { \pmb { z } } _ { t + j } \dot {  } \widehat { \pmb { z } } _ { t + j } + \delta ,$ , thenfor any $k > j$ the induced autoregressive perturbation satisfies

$$
\Delta \widehat { z } _ { t + k } = J _ { k - 1 } \cdot \cdot \cdot J _ { j } \delta + o ( \| \delta \| ) , \qquad J _ { m } = \frac { \partial f _ { \theta , m } } { \partial z } \Big | _ { ( \widehat { z } _ { t + m } , a _ { t + m } ) } .\tag{20}
$$

Within a PPWM prediction span, if an additive perturbation is applied only to the decoded output node $\widehat { z } _ { t + j }$ after all hidden future representations have been computed, then every later decoded output $\widehat { z } _ { t + k } , k > j$ , is unchanged.

Theorem 2 isolates one structural error-propagation path: autoregressive rollout contains a directed decoded-state feedback chain whose local sensitivities multiply with horizon, whereas PPWM has no such path within a span. Hidden-state perturbations may still propagate through Eq. (8). This retained causal interaction is precisely what distinguishes PPWM from independent horizon-wise prediction.

## 6 EXPERIMENTS

We evaluate PPWM from predictive modeling to downstream planning through five questions:

• Q1: Predictive fidelity. Does parallel causal prediction improve long-horizon accuracy?

• Q2: Planning utility. Does the PPWM interface improve simulator planning?

Table 1: Controlled prediction in the shared frozen LeWM latent space. Mean MSE over the first 5, 10, or 15 prediction steps (lower is better).
<table><tr><td>Model</td><td colspan="3">OGBench-Cube</td><td colspan="3">Push-T</td><td colspan="3">Two-Room</td><td colspan="3">Reacher</td></tr><tr><td></td><td> $\bar { E } _ { 1 : 5 }$ </td><td> $\bar { E } _ { 1 : 1 0 }$ </td><td> $\bar { E } _ { 1 : 1 5 }$ </td><td> $\bar { E } _ { 1 : 5 }$ </td><td> $\bar { E } _ { 1 : 1 0 }$ </td><td> $\bar { E } _ { 1 : 1 5 }$ </td><td> $\bar { E } _ { 1 : 5 }$ </td><td> $\bar { E } _ { 1 : 1 0 }$ </td><td> $\bar { E } _ { 1 : 1 5 }$ </td><td> $\bar { E } _ { 1 : 5 }$ </td><td> $\scriptstyle { \bar { E } } _ { 1 : 1 0 }$ </td><td> $\bar { E } _ { 1 : 1 5 }$ </td></tr><tr><td>TD-MPC2</td><td>0.0310</td><td>0.0452</td><td>0.0525</td><td>0.0511</td><td>0.0888</td><td>0.1454</td><td>0.2126</td><td>0.2601</td><td>0.2951</td><td>0.0628</td><td>0.0670</td><td>0.0703</td></tr><tr><td>PLDM</td><td>0.0200</td><td>0.0325</td><td>0.0374</td><td>0.0371</td><td>0.0740</td><td>0.1319</td><td>0.0749</td><td>0.1153</td><td>0.1480</td><td>0.0557</td><td>0.0586</td><td>0.0605</td></tr><tr><td>LeWM</td><td>0.0173</td><td>0.0343</td><td>0.0456</td><td>0.0312</td><td>0.0967</td><td>0.1852</td><td>0.1139</td><td>0.1867</td><td>0.2707</td><td>0.0133</td><td>0.0199</td><td>0.0266</td></tr><tr><td>Fast-LeWM</td><td>0.0144</td><td>0.0268</td><td>0.0319</td><td>0.0189</td><td>0.0526</td><td>0.1047</td><td>0.0536</td><td>0.1106</td><td>0.1610</td><td>0.0520</td><td>0.0555</td><td>0.0728</td></tr><tr><td>PP-LeWM</td><td>0.0130</td><td>0.0222</td><td>0.0270</td><td>0.0187</td><td>0.0485</td><td>0.0949</td><td>0.0395</td><td>0.0780</td><td>0.1159</td><td>0.0007</td><td>0.0009</td><td>0.0011</td></tr></table>

![](images/23a94aebeac005d9c302b05d02dd8c6214def237d56b0f656afdd7428bb6fd57.jpg)

![](images/b6cd71f79395015f729ba0c9d465a65a721a9d32e06aaa2cf246c4ed6f3f8928.jpg)

![](images/df51e07460fa07e8c95b7b7cd8646c5e383f46844357b5e1cfc20c376bc509d7.jpg)

![](images/0ec3cce4d7bb88695c625bbdd937acc94ac5af37ca509020a5e24c7b50e7e904.jpg)  
Figure 2: Cumulative latent prediction MSE as the open-loop prediction horizon increases. All methods use the same frozen LeWM representation (lower is better).

• Q3: Planning efficiency. How do prediction and planning costs scale with horizon?

• Q4: Local refinement. Can the parallel trajectory interface support effective refinement with feedback?

• Q5: Architectural generality. Does the same parallel prediction mechanism remain effective on different world-model architectures?

Experiments use Two-Room, Reacher, Push-T, and OGBench-Cube (Maes et al., 2026; Gao and Xu, 2026). LeWM-based models provide the main controlled comparison, while DINO-WM is used to evaluate architectural transfer. Planning success is measured from simulator execution. Detailed protocols, hyperparameter settings, and additional ablations are provided in Appendices A–D.

## 6.1 LONG-HORIZON PREDICTION

We first test whether PPWM improves long-horizon predictive accuracy. All methods use the same frozen LeWM encoder and projector, placing predictions and targets in a shared D = 192 latent space and making raw latent MSE directly comparable. We compare the native autoregressive LeWM predictor, recurrent and feed-forward one-step adaptations inspired by PLDM and TD-MPC2 (Sobal et al., 2026; Hansen et al., 2024), Fast-LeWM (Gao and Xu, 2026), and PP-LeWM. PP-LeWM instantiates PPWM on the LeWM architecture by retaining its representation and conditional Transformer backbone while replacing autoregressive future-state prediction with the proposed parallel causal trajectory interface.

Table 1 shows that PP-LeWM achieves the lowest MSE at all reported horizons on all four tasks. Fast-LeWM is the closest direct parallel baseline. It removes decoded-state recursion through actionprefix-conditioned prediction, whereas PP-LeWM additionally retains causal interaction among future representations before decoding. Because Fast-LeWM already removes decoded-state recursion, the consistent gap points to cross-horizon interaction as a useful design choice inside a parallel predictor. Reacher shows the largest separation. Its smooth, low-dimensional dynamics provide a regime in which cross-horizon structure can be exploited consistently across the prediction span. The cumulative-error curves in Figure 2 retain the same ordering beyond the tabulated horizons. PP-LeWM exhibits the slowest cumulative error growth across all four tasks, while the gap to autoregressive LeWM and Fast-LeWM generally widens at longer horizons. The separation grows most clearly on Reacher and Two-Room, where PP-LeWM accumulates substantially less error as prediction moves farther from the observed context.

Table 2: Open-loop simulator success across predictive interfaces (100 episodes × 3 planner seeds; higher is better). CEM uses each model’s frozen scoring rule; see Appendix C.3.
<table><tr><td></td><td></td><td colspan="2">OGBench-Cube</td><td colspan="2">Push-T</td><td colspan="2">Two-Room</td><td colspan="2">Reacher</td></tr><tr><td>Planner</td><td>Model</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td></tr><tr><td rowspan="3">LatCo</td><td>LeWM</td><td>28.67</td><td>20.67</td><td>0.67</td><td>0.33</td><td>3.33</td><td>1.67</td><td>35.67</td><td>18.67</td></tr><tr><td>Fast-LeWM</td><td>31.00</td><td>22.33</td><td>1.67</td><td>0.67</td><td>7.67</td><td>5.67</td><td>15.67</td><td>2.33</td></tr><tr><td>PP-LeWM</td><td>31.00</td><td>26.00</td><td>16.67</td><td>12.67</td><td>8.67</td><td>5.33</td><td>79.33</td><td>78.67</td></tr><tr><td rowspan="3">GRASP</td><td>LeWM</td><td>31.33</td><td>21.67</td><td>9.00</td><td>0.33</td><td>9.33</td><td>5.00</td><td>81.00</td><td>11.33</td></tr><tr><td>Fast-LeWM</td><td>29.00</td><td>18.67</td><td>15.00</td><td>2.00</td><td>24.00</td><td>20.00</td><td>62.67</td><td>5.67</td></tr><tr><td>PP-LeWM</td><td>34.00</td><td>26.00</td><td>17.00</td><td>3.67</td><td>10.33</td><td>4.67</td><td>38.67</td><td>0.67</td></tr><tr><td rowspan="3">CEM</td><td>LeWM</td><td>35.00</td><td>28.67</td><td>35.33</td><td>27.67</td><td>49.67</td><td>49.67</td><td>76.67</td><td>64.33</td></tr><tr><td>Fast-LeWM</td><td>38.00</td><td>35.33</td><td>35.33</td><td>28.67</td><td>44.67</td><td>44.33</td><td>51.33</td><td>35.00</td></tr><tr><td>PP-LeWM</td><td>45.67</td><td>39.33</td><td>37.67</td><td>29.33</td><td>55.33</td><td>55.00</td><td>97.00</td><td>96.67</td></tr></table>

## 6.2 PLANNING ACROSS PREDICTIVE INTERFACES

We evaluate whether the predictive differences persist after coupling each model to downstream planners. We compare LeWM, Fast-LeWM, and PP-LeWM under trajectory-wide LatCo and GRASP optimization as well as population-based CEM planning. Each completed trial plans over a horizon of 50 primitive simulator actions, and executes the full plan open loop. Anytime and Terminal success are computed only from the resulting simulator trajectory. Table 2 summarizes the resulting open-loop success rates across planners and predictive interfaces.

Under CEM, PP-LeWM achieves the highest anytime and terminal success on all four tasks. The lifted planners expose a different model–planner interaction: Fast-LeWM remains competitive on Cube and Two-Room but degrades sharply with LatCo on Push-T and Reacher, while GRASP exhibits high anytime but low terminal success on Reacher. CEM evaluates Fast-LeWM through its native direct finite-horizon interface, whereas the LatCo and GRASP adaptations require a chained one-step dynamics factor over optimized latent states. Appendix D.1 shows large virtual-to-rollout gaps when Fast-LeWM is forced into this chained local factor. The mismatch is therefore an interface issue: lifted optimization can improve the local surrogate without improving the native finite-horizon prediction used by the direct model.

## 6.3 PLANNING EFFICIENCY AND HORIZON SCALING

Planning efficiency is evaluated by measuring how dynamics and end-to-end CEM latency scale. Table 3 summarizes CEM planning latency at a ten-step model horizon, while Figure 3 reports average CEM planning and dynamics latency for planning horizons of 1, 5, 10, 15, 20, 25, and 30 steps. PP-LeWM achieves the lowest CEM planning latency on all four tasks, averaging 374.5 ms compared with 1235.2 ms for LeWM and 475.8 ms for Fast-LeWM. This corresponds to a 3.30× average speedup over LeWM. Figure 3 further shows that LeWM latency grows rapidly with rollout length, whereas PP-LeWM exhibits span-wise growth and remains substantially more efficient at medium and long horizons. Fast-LeWM has the lowest raw dynamics latency due to its lighter direct predictor, but its re-anchored self-consistency score adds planner-side model evaluations. Score-level details are provided in Appendix D.3. PP-LeWM therefore delivers the lowest end-to-end CEM latency while retaining causal interaction among future representations.

## 6.4 CLOSED-LOOP TRAJECTORY REFINEMENT

To evaluate trajectory-level refinement under feedback, we run PP-LeWM in closed loop on Push-T and Two-Room. The planner replans every five primitive simulator steps. At each decision, the agent receives a fresh observation, runs CEM over the remaining episode horizon, applies the local refinement, executes the first five primitive actions, and replans. We compare the unrefined CEM plan with action-only Adam and trajectory-wide refinement, which jointly updates the action sequence and an auxiliary latent trajectory. All Anytime and Terminal success rates are computed exclusively from the resulting closed-loop simulator trajectories, as shown in Table 4.

Trajectory refinement provides the clearest closed-loop gain on Two-Room, improving both Anytime and Terminal success by 2.67 percentage points over CEM and consistently outperforming actiononly Adam. The additional gain over action-only Adam on Two-Room comes with the ability to move the latent trajectory and actions together across the full horizon. Moreover, the parallel trajectory interface reduces the sequential prediction depth of each refinement iteration from O(H) autoregressive transitions to $\mathcal { O } ( \lceil H \bar { / } K \rceil )$ PPWM span evaluations, and to O(1) when $H \leq K$ . Closedloop replanning therefore combines trajectory-wide updates with a short sequential optimization path while repeatedly re-anchoring the model to observed states.

Table 3: CEM planning latency at H=10 (ms); speedup relative to LeWM is shown below.
<table><tr><td>Dataset</td><td>LeWM</td><td>Fast-LeWM</td><td>PP-LeWM</td></tr><tr><td rowspan="3">OGBench-Cube</td><td>1245.4</td><td>474.3</td><td>386.7</td></tr><tr><td>1.00×</td><td>2.63×</td><td>3.22×</td></tr><tr><td>1252.9</td><td>457.7</td><td>357.0</td></tr><tr><td rowspan="2">Push-T</td><td>1.00×</td><td>2.74×</td><td>3.51×</td></tr><tr><td>1243.8</td><td>505.9</td><td>385.6</td></tr><tr><td rowspan="2">Two-Room</td><td>1.00×</td><td>2.46×</td><td>3.23×</td></tr><tr><td>1198.6</td><td>465.3</td><td>368.9</td></tr><tr><td>Reacher</td><td>1.00×</td><td>2.58×</td><td>3.25×</td></tr><tr><td rowspan="2">Average</td><td>1235.2 1.00×</td><td>475.8</td><td>374.5</td></tr><tr><td></td><td>2.60×</td><td>3.30×</td></tr></table>

![](images/57491be2d0402ec7c4e771893af23077d52044685fa06d3e51900d97b840e776.jpg)  
Figure 3: Average CEM planning and dynamics latency across four tasks.

Table 4: Closed-loop refinement with PP-LeWM under a shared CEM planner (higher is better).
<table><tr><td></td><td colspan="2">Push-T</td><td colspan="2">Two-Room</td></tr><tr><td>Method</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td></tr><tr><td>CEM</td><td>56.33</td><td>49.00</td><td>55.33</td><td>55.00</td></tr><tr><td>+ Adam</td><td>57.00</td><td>50.33</td><td>56.00</td><td>56.00</td></tr><tr><td>+ Traj. Refine</td><td>58.33</td><td>50.00</td><td>58.00</td><td>57.67</td></tr></table>

Table 5: PP-DINO-WM transfer on Push-T. DI-NOv2 visual-patch MSE (lower is better).
<table><tr><td>Prediction Steps</td><td>DINO-WM</td><td>PP-DINO-WM</td><td>Reduction</td></tr><tr><td>5</td><td>0.098855</td><td>0.080229</td><td>18.84%</td></tr><tr><td>10</td><td>0.154497</td><td>0.108464</td><td>29.80%</td></tr><tr><td>15</td><td>0.219716</td><td>0.147109</td><td>33.05%</td></tr><tr><td>20</td><td>0.284458</td><td>0.180760</td><td>36.45%</td></tr></table>

## 6.5 ARCHITECTURAL GENERALITY ON DINO-WM

To test architectural transfer, we instantiate PPWM on DINO-WM (Zhou et al., 2024). PP-DINO-WM retains the frozen DINOv2 representation (Oquab et al., 2023) while replacing autoregressive future prediction with the proposed parallel causal trajectory interface. The autoregressive DINO-WM and PP-DINO-WM use the same test windows, histories, and future actions and are rolled out open loop. Table 5 reports prediction error across the evaluated horizons. PP-DINO-WM reduces visual latent error at every reported horizon, with the relative reduction increasing from 18.84% over 5 steps to 36.45% over 20 steps. The same temporal factorization therefore remains effective after replacing the LeWM backbone with DINO-WM.

## 6.6 PPWM COMPONENT ANALYSIS

Table 6 isolates the two main components of PP-LeWM on Push-T. Fast-LeWM provides a parallel reference without causal interaction among future representations, while PP-LeWM w/o MAR removes multi-anchor rollout training. Full PP-LeWM achieves the lowest error at every horizon. These results show that causal trajectory construction and MAR are both important for maintaining accurate long-horizon prediction.

Table 6: Push-T component analysis using mean latent MSE; lower is better.
<table><tr><td>Model</td><td> $\bar { E } _ { 1 : 1 0 }$  →</td><td> $\bar { E } _ { 1 : 1 5 }$  →</td><td> $\bar { E } _ { 1 : 2 0 }$  →</td><td> $\bar { E } _ { 1 : 3 0 \downarrow }$ </td></tr><tr><td>Fast-LeWM</td><td>0.052634</td><td>0.104680</td><td>0.146183</td><td>0.208614</td></tr><tr><td>PP-LeWM w/o MAR</td><td>0.051014</td><td>0.102552</td><td>0.150012</td><td>0.239441</td></tr><tr><td>PP-LeWM</td><td>0.048525</td><td>0.094867</td><td>0.133276</td><td>0.191456</td></tr></table>

## 7 CONCLUSION

We introduced Parallel Predictive World Models (PPWM) for accurate and efficient long-horizon planning. PPWM replaces recursive decoded-state prediction with parallel causal trajectory prediction, preserving temporal interaction among future representations while exposing the finite-horizon trajectory jointly to planning. Across four visual-control tasks, PPWM consistently improves longhorizon prediction and CEM planning performance while achieving over a 3× average planning speedup relative to autoregressive LeWM. The same factorization transfers to DINO-WM and naturally supports trajectory-wide refinement, demonstrating its compatibility with different worldmodel backbones and planning regimes. More broadly, PPWM shows that long-horizon world models can preserve causal temporal structure without relying on state-by-state autoregressive rollout, providing a direct and efficient alternative for trajectory prediction and planning.

## AI USE STATEMENT

Generative AI tools were used for language editing, manuscript organization, literature search, reference verification, and notation and consistency checking. All AI-assisted suggestions and bibliographic information were manually verified by the authors. The authors take full responsibility for the final manuscript.

## REPRODUCIBILITY STATEMENT

Sections 3–6 define PPWM, its predictive and planning formulations, and the evaluation questions. Appendix A provides implementation details, Appendix B states assumptions and derivations, Appendix C specifies the experimental protocols and metric definitions, and Appendix D collects additional ablations and diagnostics. The code will be made publicly available.

## REFERENCES

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. Advances in neural information processing systems, 31, 2018.

Pieter-Tjerk De Boer, Dirk P Kroese, Shie Mannor, and Reuven Y Rubinstein. A tutorial on the cross-entropy method. Annals ofoperations research, 134(1):19–67, 2005.

Yuntian Gao and Xiangyu Xu. Fast leworldmodel. arXiv preprint arXiv:2606.26217, 2026.

Ignat Georgiev, Varun Giridhar, Nick Hansen, and Animesh Garg. Pwm: Policy learning with multi-task world models. In International Conference on Learning Representations, volume 2025, pages 13737–13757, 2025.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 2555–2565. PMLR, 2019.

Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pages 47376– 47405, 2024.

Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022.

Xiaodi Huang, Ziyi Ding, Jingtian Wan, Yuchen Liu, Yuan Zhang, Xiao-Ping Zhang, Jiayu Chen, Zhang Zhang, and Tao Huang. Traj-lewm: Path-aware world-model planning via latent trajectory cost. arXiv preprint arXiv:2608.14125, 2026.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in neural information processing systems, 32, 2019.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Transformers are sample-efficient world models. arXiv preprint arXiv:2209.00588, 2022.

Nikhil Mishra, Pieter Abbeel, and Igor Mordatch. Prediction and control with temporal segment models. In International conference on machine learning, pages 2459–2468. PMLR, 2017.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Phu Pham and Aniket Bera. Latent energy action planning with world models. arXiv preprint arXiv:2609.03294, 2026.

Michael Psenka, Michael Rabbat, Aditi Krishnapriyan, Yann LeCun, and Amir Bar. Parallel stochastic gradient-based planning for world models. arXiv preprint arXiv:2602.00475, 2026.

Oleh Rybkin, Chuning Zhu, Anusha Nagabandi, Kostas Daniilidis, Igor Mordatch, and Sergey Levine. Model-based reinforcement learning via latent-space collocation. In International Conference on Machine Learning, pages 9190–9201. PMLR, 2021.

Uladzislau Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim GJ Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. Advances in Neural Information Processing Systems, 38:43905–43941, 2026.

Anne Somalwar, Bruce D Lee, George J Pappas, and Nikolai Matni. Learning with imperfect models: When multi-step prediction mitigates compounding error. In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 82–89. IEEE, 2025.

Anne Somalwar, Bruce D Lee, George J Pappas, and Nikolai Matni. Statistical efficiency of single-and multi-step models for forecasting and control. arXiv preprint arXiv:2603.23465, 2026.

Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M Rehg, Byron Boots, and Evangelos A Theodorou. Information theoretic mpc for model-based reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA), pages 1714–1721. IEEE, 2017.

Liang Wu, Wallace Gian Yion Tan, Richard D Braatz, and Jan Drgo ´ na. Koopman-boxqp: Solving ˇ large-scale nmpc at khz rates. arXiv preprint arXiv:2602.18331, 2026.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

Christos Ziakas, Amir Bar, and Alessandra Russo. Grounding generated videos in feasible plans via world models. arXiv preprint arXiv:2602.01960, 2026.

## A IMPLEMENTATION DETAILS

## A.1 PP-LEWM INSTANTIATION

Representation, temporal scale, and context. The visual encoder and projector are frozen through out PP-LeWM training. The latent dimension is $D = 1 9 2$ . For Push-T and Reacher, one model-step action is the same five-primitive-action block used by LeWM. With two-dimensional primitive controls, the raw block dimension is 10. The main configuration uses K = 10 future model steps.

History context encoder. The action-prefix branch is conditioned on a compact motion context rather than only the latest latent. We compute

$$
u _ { t } = [ z _ { t } , \ z _ { t } - z _ { t - 1 } , \ z _ { t - 1 } - z _ { t - 2 } ] \in \mathbb { R } ^ { 5 7 6 }\tag{21}
$$

and map it with a lightweight MLP

$$
5 7 6 \to 3 8 4 \to 1 9 2\tag{22}
$$

using normalization and GELU between linear layers. Its output $h _ { t } \in \mathbb { R } ^ { 1 9 2 }$ is used only by the action-prefix encoder. The original three-state LeWM trajectory input is retained separately.

History-conditioned action-prefix encoder. Each future action block is first projected to 32 dimensions and then mapped through a 32 → 768 → 192 MLP. The history context $h _ { t }$ is separately mapped through a 192 → 768 → 192 state-token MLP. We prepend this state token to the sequence of future action tokens, add fixed sinusoidal positional encodings, and process the sequence with a three-layer causal Transformer with model dimension 192, six heads, and FFN dimension 768. Dropping the context-token output yields $p _ { t + 1 : t + K } \in \mathbb { R } ^ { K \times 1 9 2 }$ . A LayerNorm followed by a linear 192 → 192 prefix projection $W _ { p }$ maps these features into the future-query space. The projection is initialized near zero so that the new prefix branch initially perturbs the pretrained trajectory dynamics only weakly.

Horizon-aware future queries. Fixed horizon codes $e _ { k } \in \mathbb { R } ^ { 1 9 2 }$ are added to the future-query tokens. We preserve the one-step LeWM prior by setting

$$
q _ { 1 } = z _ { t } , \qquad q _ { k } = z _ { t } + e _ { k } + W _ { p } p _ { t + k } , \quad k = 2 , \ldots , K .\tag{23}
$$

The full trajectory sequence is therefore

$$
[ z _ { t - 2 } , z _ { t - 1 } , q _ { 1 } , . . . , q _ { K } ] .\tag{24}
$$

The causal mask permits hidden trajectory interaction from earlier to later query tokens, but no predicted latent $\hat { z } _ { t + k }$ is fed back as the input used to generate $\hat { z } _ { t + k + 1 }$ within the span.

Conditional dynamics backbone and prediction head. The base LeWM predictor contains six causal conditional Transformer blocks with hidden dimension 192, AdaLN-style local action conditioning, and a 2048-dimensional feed-forward sublayer. The main model fine-tunes all six predictor blocks, the final normalization, and the pretrained prediction projection. The encoder, projector, and LeWM action encoder remain frozen. Future hidden states are decoded directly by the fine-tuned pretrained prediction projection.

Training curriculum. New history/prefix modules are first warmed up with the dynamics backbone frozen and $\mathcal { L } _ { \mathrm { d i r e c t } }$ only. We then jointly fine-tune the full dynamics backbone using a smaller learning rate for pretrained dynamics parameters than for the new history/prefix modules. After direct prediction stabilizes, MAR is enabled. During development, grouped learning rates are selected from $\mathrm { { 1 0 ^ { - 4 } { - 3 } \times 1 0 ^ { - 4 } } }$ for new modules and $1 0 ^ { - 5 } – \mathsf { \bar { 3 } } \times 1 0 ^ { - 5 }$ for pretrained dynamics parameters.

Direct and multi-anchor rollout losses. All K direct outputs receive equal-weight latent MSE supervision. For MAR, after the first span we choose two random anchors $s _ { 1 } , s _ { 2 } \sim U ( 3 , K - 1 )$ and the boundary anchor K. At every anchor, the three predicted states are detached before the next model call. The history encoder and prefix encoder are then recomputed from this detached predicted history, so the rollout objective trains the new action representation under self-generated dynamics context without backpropagating through the first-span predicted history. The final objective is

$$
\mathcal { L } _ { \mathrm { P P W M } } = \mathcal { L } _ { \mathrm { d i r e c t } } + 0 . 2 5 \mathcal { L } _ { \mathrm { M A R } } .\tag{25}
$$

The base configuration uses no auxiliary task-specific losses. Nominal and action-perturbed trajectories share the same objective.

## A.2 CONTROLLED DYNAMICS BASELINES

The controlled PLDM and TD-MPC2 baselines use the same cached LeWM latents and frozen action embedding as the other dynamics models, so the comparison isolates the temporal predictor. We retain their characteristic recurrent or feed-forward transition structures while omitting agent-specific components outside the controlled dynamics comparison.

PLDM. PLDM uses a recurrent predictor in its Two-Room instantiation. The original paper reports a two-layer GRU with 512 hidden dimensions (Sobal et al., 2026). Our controlled PLDM-GRU baseline projects the current 192-dimensional LeWM latent into the GRU hidden state, consumes the frozen LeWM action embeddings recurrently, and maps each recurrent output back to the 192- dimensional LeWM latent space. It is rolled out autoregressively at evaluation time. This baseline tests whether recurrent memory alters the long-horizon behavior of one-step autoregressive prediction while keeping the visual representation fixed.

TD-MPC2. TD-MPC2 uses a feed-forward one-step latent transition of the form $\boldsymbol { z } _ { t + 1 } = d _ { \theta } ( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } )$ implemented as an MLP (Hansen et al., 2024). Our controlled adaptation concatenates the frozen LeWM latent and action embedding, predicts the next LeWM latent with an MLP transition, and recursively feeds predictions back at evaluation time.

Training control. The two adapted baselines are trained only on the cached LeWM latent/action trajectories used for the PPWM comparison. Their objectives supervise predicted LeWM latents and do not include the representation-learning losses of their source methods. Parameter counts, optimizer schedules, and exact hidden widths follow the controlled experiment configuration.

## A.3 PP-DINO-WM INSTANTIATION

Representation and temporal scale. We instantiate PPWM on the Push-T DINO-WM backbone while keeping the DINOv2 ViT-S/14 visual encoder frozen. RGB observations are normalized to [−1, 1] and resized to 196 × 196 before DINOv2 encoding. Each observation is represented by 196 spatial patch tokens of dimension 384. The four-dimensional proprioceptive state is linearly encoded to 10 dimensions and concatenated to every patch, giving a 394-dimensional state token. Histories contain three frames separated by five primitive environment steps. Five primitive twodimensional actions are grouped into one 10-dimensional model-step action, and the native parallel span is K = 10, corresponding to 50 primitive environment steps.

Parallel dynamics. PP-DINO-WM uses the same PPWM factorization as Section 4: a causal action-prefix branch constructs horizon-specific future queries, which are jointly processed by the DINO-WM causal dynamics backbone. The prefix encoder has three Transformer layers with width 192, six attention heads, and FFN dimension 768. The DINO-WM dynamics backbone has six layers, 16 attention heads, and FFN dimension 2048. Attention is causal at the frame level, so each future frame can interact with preceding future-frame representations while remaining independent of later frames. All K future states are produced in one backbone call without feeding decoded states back within the span.

Initialization and training. The dynamics predictor, action encoder, and proprioceptive encoder are initialized from the official Push-T DINO-WM checkpoint, while the history/prefix branch is newly initialized. The episode-separated split contains 301,279 training windows, 512 development windows, and 1,008 test windows. The first five epochs train only the new branch. From epoch 6 onward, the dynamics predictor and action encoder are jointly fine-tuned. The results in Section 6.5 use the fixed epoch-10 checkpoint and therefore reflect direct 10-step supervision without MAR. Training uses four-GPU data parallelism and AdamW with learning rates of 10<sup>−4</sup> for newly initialized parameters and $1 0 ^ { - 5 }$ for pretrained dynamics parameters, weight decay 10<sup>−4</sup>, gradient-norm clipping at 1, FP32 arithmetic, and an effective batch size of 128.

## A.4 PARALLEL TRAJECTORY REFINEMENT DETAILS

CEM initialization and trajectory variables. At each closed-loop decision, refinement starts from the CEM action sequence $\pmb { A } _ { 0 }$ and its PPWM rollout $\widehat { Z } ( A _ { 0 } )$ . The variable Z is auxiliary, while executable trajectories are always generated by the frozen PPWM.

Tail task cost. CEM and the trajectory refinement use the same normalized tail objective. For the remaining horizon H, let $q = \operatorname* { m i n } ( 4 , H )$ and define

$$
J _ { \mathrm { t a s k } } = J _ { \rho } ( \mathbf { Z } , z _ { g } ) = \frac { \sum _ { j = 0 } ^ { q - 1 } \rho ^ { j } \operatorname { M S E } ( \mathbf { z } _ { H - j } , z _ { g } ) } { \sum _ { j = 0 } ^ { q - 1 } \rho ^ { j } } .\tag{26}
$$

The terminal state receives the largest weight, and at most the final four model states contribute to the score. The refinement objective in Eq. (14) uses this score for both the free trajectory and the executable PPWM rollout, together with a soft dynamics-consistency penalty and a quadratic action penalty around the CEM reference plan.

## B PROOFS AND ADDITIONAL ANALYSIS

## B.1 PROOF OF THEOREM 1

We prove the claim by recursive substitution. For the first horizon, define

$$
G _ { 1 } ( \mathcal { C } _ { t } , \pmb { a } _ { t } ) = g _ { 1 } ( \mathcal { C } _ { t } , \pmb { a } _ { t } ) .\tag{27}
$$

Then $\widehat { \pmb { z } } _ { t + 1 } = G _ { 1 } ( \mathcal { C } _ { t } , \pmb { a } _ { t } )$ . Assume that for every $j < k$ there exists a function $G _ { j }$ satisfying

$$
\widehat { \mathfrak { z } } _ { t + j } = G _ { j } ( \mathcal { C } _ { t } , \mathbf { a } _ { t : t + j - 1 } ) .\tag{28}
$$

Substituting these expressions into the k-th autoregressive prediction gives

$$
G _ { k } ( \mathcal { C } _ { t } , \pmb { a } _ { t : t + k - 1 } ) = g _ { k } ( \mathcal { C } _ { t } , G _ { 1 } ( \mathcal { C } _ { t } , \pmb { a } _ { t } ) , \dots , G _ { k - 1 } ( \mathcal { C } _ { t } , \pmb { a } _ { t : t + k - 2 } ) , \pmb { a } _ { t : t + k - 1 } ) .\tag{29}
$$

Therefore $\widehat { \pmb { z } } _ { t + k } = G _ { k } ( \mathcal { C } _ { t } , \pmb { a } _ { t : t + k - 1 } )$ . By induction, this construction holds for every $k = 1 , \ldots , K$ Each $G _ { k }$ depends only on the current context and the action prefix ending at $t + k - 1$ , so the resulting finite-horizon map is causal.

## B.2 PROOF OF THEOREM 2

Perturb $\widehat { z } _ { t + j } \mathrm { ~ t o ~ } \widehat { z } _ { t + j } + \delta$ while keeping the action sequence fixed, and let $\Delta _ { m }$ denote the induced perturbation of $\widehat { z } _ { t + m }$ . Differentiability of Eq. (19) gives

$$
\Delta _ { m + 1 } = J _ { m } \Delta _ { m } + o ( \| \Delta _ { m } \| ) ,\tag{30}
$$

where $J _ { m }$ is defined in Eq. (20). Starting from $\Delta _ { j } = \delta$ and recursively applying Eq. (30) yields

$$
\Delta _ { k } = J _ { k - 1 } \cdot \cdot \cdot J _ { j } \delta + o ( \| \delta \| ) ,\tag{31}
$$

which proves the autoregressive statement. Hence the autoregressive computational graph contains a directed path from each decoded future state to every later decoded state.

For PPWM, the future hidden representations are computed by the causal trajectory blocks before decoding, as in Eq. (8), and each output is obtained by

$$
\widehat { \sf z } _ { t + k } = F _ { \theta } ( { \pmb u } _ { t + k } ^ { ( L ) } ) .\tag{32}
$$

No decoded output $\widehat { \boldsymbol { z } } _ { t + j }$ is an input to any later hidden representation $\pmb { u } _ { t + k } ^ { ( \ell ) }$ . Therefore, after the hidden representations have been fixed, intervening on $\widehat { \boldsymbol { z } } _ { t + j }$ after $F _ { \theta }$ creates no directed computational path to any later output and induces exactly zero change in $\widehat { z } _ { t + k }$ for $k > j$ . At the same time, causal hidden interaction remains because $\pmb { u } _ { t + k } ^ { ( \ell ) }$ can depend on earlier future representations $\mathbf { } u _ { t + 1 : t + k } ^ { ( \ell - 1 ) }$ . The result therefore removes recursive decoded-state feedback without asserting independence among PPWM predictions.

## C EXPERIMENTAL PROTOCOLS

This appendix specifies the common evaluation protocol and the experiment-specific deviations used in the main text.

## C.1 EVALUATION SETUP

One model step corresponds to five primitive simulator actions. Unless stated otherwise, open-loop planning uses a one-shot horizon of 10 model steps (50 primitive actions), and each configuration is evaluated on 100 episodes with three planner seeds. CEM uses 300 candidates, 30 elites, and 30 iterations in standardized action coordinates. Anytime success records whether the native task success condition is reached at any point during simulator execution, while Terminal success evaluates the final primitive step. Planner hyperparameters are selected on development data and frozen before test evaluation.

## C.2 CONTROLLED PREDICTION PROTOCOL

The shared representation, baseline constructions, and training controls are described in Appendix A.2. All methods receive identical initial histories, future actions, and latent targets. For horizon j, we measure

$$
E _ { j } = \frac { 1 } { D } \left. \hat { z } _ { t + j } - z _ { t + j } ^ { \mathrm { G T } } \right. _ { 2 } ^ { 2 } .\tag{33}
$$

The mean error over the first h steps and the cumulative error are

$$
\bar { E } _ { 1 : h } = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } E _ { j } , \qquad C _ { h } = \sum _ { j = 1 } ^ { h } E _ { j } = h \bar { E } _ { 1 : h } .\tag{34}
$$

Table 1 reports $\bar { E } _ { 1 : h }$ and Figure 2 reports $C _ { h }$

## C.3 OPEN-LOOP CEM PROTOCOL

The CEM rows in Table 2 follow the common setup above: each trial plans once from the observed three-frame history and goal, then executes the complete 50-step action sequence without replanning or early stopping. Push-T success requires the joint agent/block positional criterion and block-angle error below $\pi / 9 .$ . Reacher requires both absolute joint-position errors below 0.05 radians. Two-Room uses final agent-to-goal distance below 16 pixels, and OGBench-Cube uses final cube-to-goal distance at most 0.04 meters.

## C.4 OPEN-LOOP TRAJECTORY OPTIMIZATION PROTOCOL

All planners use the same standardized action coordinates and environment bounds. LatCo uses its stochastic action initialization with start–goal state interpolation, while GRASP uses its lifted-state initialization with latent perturbations and periodic rollout synchronization. These initializations are kept separate from the CEM-initialized local refinement in Section 6.4.

## C.5 CLOSED-LOOP REFINEMENT PROTOCOL

The closed-loop study uses the common CEM search budget, but executes only one five-action block before receiving a new simulator observation and replanning. The remaining model horizon decreases from 10 to one over the 50-step episode, and both tasks use $\rho = 0 . 2 5$ . All methods start each decision from the same CEM solution and preserve the unrefined plan as a fallback candidate. Adam refines only the standardized action sequence, whereas trajectory refinement jointly updates the action sequence and a free virtual latent trajectory. The frozen trajectory-refinement settings are

<table><tr><td>Parameter</td><td>Push-T</td><td>Two-Room</td></tr><tr><td>Rounds R</td><td>2</td><td>3</td></tr><tr><td> $\eta _ { A }$ </td><td>1</td><td>2</td></tr><tr><td> $\eta _ { Z }$ </td><td>24</td><td>48</td></tr><tr><td>α</td><td>1</td><td>1</td></tr><tr><td> $\lambda _ { \mathrm { d y n } }$ </td><td>1</td><td>1</td></tr><tr><td> $\lambda _ { A }$ </td><td>0.01</td><td>0.01</td></tr><tr><td>Trust-region radius δ</td><td>1</td><td>0.25</td></tr><tr><td>Tail coefficient  $\rho$ </td><td>0.25</td><td>0.25</td></tr></table>

## C.6 SINGLE-GPU TIMING PROTOCOL

Timing uses horizons $H \in \{ 1 , 5 , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ on a single NVIDIA A100-SXM4-80GB GPU in FP32 evaluation mode. All methods receive the same three-frame latent history, past actions, and future action sequence. Data loading, image encoding, checkpoint loading, and simulator execution are excluded. Dynamics latency measures one evaluation of 300 candidate trajectories, and end-to-end CEM timing uses the common search budget from Appendix C.1. LeWM is rolled out autoregressively, while Fast-LeWM and PP-LeWM are composed in spans of at most 10 model steps. A complete CEM solve includes candidate sampling, action processing, dynamics prediction, objective evaluation, sorting, elite updates, and final plan scoring. CUDA synchronization is applied around timed regions, and the horizon-scaling curves report median wall-clock time.

## C.7 DINO-WM ARCHITECTURAL GENERALITY PROTOCOL

The DINO-WM experiment evaluates both models on the same 1,008 held-out Push-T windows and known future action sequences. DINO-WM rolls out autoregressively, whereas PP-DINO-WM predicts 10 steps per span and composes two spans for the 20-step horizon. Prediction error is computed on the 196-by-384 DINOv2 visual patch features, averaging the stepwise MSE over all valid windows at each reported horizon.

## D ADDITIONAL ABLATIONS AND DIAGNOSTICS

This appendix contains three focused analyses that directly support the main results: a plannerinterface diagnostic for the lifted Fast-LeWM comparisons, ablations of the PPWM parallel prediction span and self-conditioned long-horizon training, and a controlled analysis of the CEM scoring function.

## D.1 PLANNER-INTERFACE DIAGNOSTIC

Table 7: Planner-interface diagnostic for the Fast-LeWM lifted-planner adaptations. “Virtual” is the optimized terminal latent state, “chained rollout” re-evaluates the one-step planning factor from the true initial context, and “simulator” is the realized terminal latent MSE after executing the optimized actions. All entries use the completed 300-trial evaluations. Lower is better.
<table><tr><td>Task</td><td></td><td></td><td>Planner Virtual MSE Chained MSE Simulator MSE</td></tr><tr><td>Push-T</td><td>LatCo</td><td>0.0036</td><td>1.5570 1.5449</td></tr><tr><td>Push-T</td><td>GRASP</td><td>1.0202</td><td>1.0155 1.2472</td></tr><tr><td>Reacher LatCo</td><td></td><td>0.0002</td><td>0.9086 1.0736</td></tr><tr><td>Reacher GRASP</td><td></td><td>0.1736</td><td>0.1752 0.6278</td></tr></table>

The large virtual-to-rollout gaps show that the chained one-step factor used by lifted Fast-LeWM can become misaligned with its native direct predictor, especially for LatCo. Fast-LeWM consequently performs well when CEM queries its native finite-horizon predictor, but degrades when lifted optimization replaces that interface with a chained local factor.

## D.2 PPWM COMPONENT ABLATIONS

We examine three design choices tied to the proposed parallel predictor: the native parallel prediction span K, causal trajectory construction, and multi-anchor self-conditioned rollout training (MAR).

Parallel prediction span. For the span study, $K \in \{ 5 , 1 0 , 1 5 \}$ while all other predictor and rollout settings are held fixed.

Table 8: Effect of the native parallel prediction span K. Entries are mean latent MSE over the first h prediction steps, $\bar { E } _ { 1 : h }$ (lower is better).
<table><tr><td rowspan=1 colspan=1>Dataset K   $\bar { E } _ { 1 : 5 } \downarrow$     $\bar { E } _ { 1 : 1 0 } \downarrow$    $\bar { E } _ { 1 : 1 5 } \downarrow$    $\bar { E } _ { 1 : 2 0 } \downarrow$ </td></tr><tr><td rowspan=1 colspan=1>Push-T 50.0175440.0503090.0946250.138354Push-T100.0186560.0485250.0948670.133276Push-T150.0214140.0534200.0934730.143585</td></tr><tr><td rowspan=1 colspan=1>Reacher 5 0.000739 0.000926 0.001113 0.001296Reacher 10 0.000735 0.000900 0.001075 0.001261Reacher150.0008180.0009590.001144 0.001345</td></tr></table>

K = 10 is best on all reported Reacher horizons and gives the lowest Push-T error at $h = 1 0$ and h = 20, so we use it as the default span.

## D.3 CEM SCORE CONTROLS AND TAIL ABLATION

Table 2 reports end-to-end CEM performance under each model’s frozen scoring rule. The controls below complement it with goal-only scoring and a PP-LeWM tail ablation.

CEM candidate scores. For candidate m, the ordinary terminal-goal score and Fast-LeWM selfconsistency (SC) score (Gao and Xu, 2026) are

$$
\begin{array} { r l } & { J _ { \mathrm { g o a l } } ^ { ( m ) } = \left. \hat { z } _ { t + H } ^ { ( m ) } - z _ { g } \right. _ { 2 } ^ { 2 } , } \\ & { J _ { \mathrm { F a s t - S C } } ^ { ( m ) } = J _ { \mathrm { g o a l } } ^ { ( m ) } + \beta \left. \hat { z } _ { t + H } ^ { ( m ) } - \widetilde { z } _ { t + H } ^ { ( m ) } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{35}
$$

Here $\widetilde { z } _ { t + H } ^ { ( m ) }$ is Fast-LeWM’s re-anchored terminal estimate, while PP-LeWM uses the tail objective $J _ { \rho }$ defined once in Eq. (26). Setting $\beta = 0$ recovers goal-only CEM, and $\rho = 0$ gives the same terminal-goal ranking for PP-LeWM.

Controlled open-loop comparison. Table 9 combines the CEM score control with the PP-LeWM tail sweep and reports simulator Anytime and Terminal success.

Table 9: Open-loop CEM score control and PP-LeWM tail ablation $( H { = } 1 0 )$ . Entries report Anytime and Terminal simulator success (%). Fast-LeWM Goal removes SC, while PP-LeWM $\rho = 0$ is ordinary terminal-goal CEM. Each configuration uses 100 episodes and three planner seeds.
<table><tr><td></td><td></td><td colspan="2">Push-T</td><td colspan="2">Reacher</td><td colspan="2">Two-Room</td><td colspan="2">OGBench-Cube</td></tr><tr><td>Model</td><td>CEM score</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td><td>Anytime↑</td><td>Terminal↑</td></tr><tr><td>LeWM</td><td>Goal</td><td>35.33</td><td>27.67</td><td>76.67</td><td>64.33</td><td>49.67</td><td>49.67</td><td>35.00</td><td>28.67</td></tr><tr><td>Fast-LeWM</td><td>SC</td><td>35.33</td><td>28.67</td><td>51.33</td><td>35.00</td><td>44.67</td><td>44.33</td><td>38.00</td><td>35.33</td></tr><tr><td>Fast-LeWM</td><td>Goal</td><td>30.00</td><td>22.33</td><td>51.67</td><td>34.33</td><td>50.67</td><td>50.67</td><td>36.33</td><td>30.00</td></tr><tr><td>PP-LeWM</td><td>J0 (Goal)</td><td>37.67</td><td>29.33</td><td>97.00</td><td>96.67</td><td>54.67</td><td>54.33</td><td>38.67</td><td>32.33</td></tr><tr><td>PP-LeWM</td><td> $J _ { 0 . 1 }$ </td><td>35.67</td><td>25.00</td><td>97.00</td><td>96.67</td><td>55.33</td><td>55.00</td><td>45.67</td><td>39.33</td></tr><tr><td>PP-LeWM</td><td> $J _ { 0 . 2 5 }$ </td><td>38.67</td><td>27.33</td><td>96.67</td><td>95.00</td><td>55.00</td><td>55.00</td><td>41.67</td><td>37.00</td></tr><tr><td>PP-LeWM</td><td> $J _ { 0 . 5 }$ </td><td>41.33</td><td>26.67</td><td>96.33</td><td>93.67</td><td>52.33</td><td>52.33</td><td>43.67</td><td>38.33</td></tr></table>

Under ordinary goal-only CEM $( J _ { 0 } ) .$ , PP-LeWM already outperforms the goal-only LeWM and Fast-LeWM baselines on all four tasks, so the main planning gain does not depend on tail weighting. Tail weighting shifts the operating point by task: it raises success on OGBench-Cube and Two-Room and increases Push-T Anytime success, whereas Reacher already saturates under $J _ { 0 } .$ . Fast-LeWM’s SC term is similarly task dependent.