# WHEN RECURSIVE MODELS FINISH COMPUTING

Hare Krishna<sup>1</sup>, Shubham Singh<sup>2</sup>, Stephen Ebert<sup>3</sup>, Hao-Yu Sun<sup>4,5</sup>

<sup>1</sup>Weinberg Institute, Department of Physics, University of Texas at Austin

<sup>2</sup>Department of Urban and Environmental Policy Planning, Tufts University

<sup>3</sup>Zyphra Technologies, San Francisco, CA, USA

<sup>4</sup>Mathematics Department, Austin Community College

<sup>5</sup>SpaceXAI, USA

hkrishna.phy@gmail.com, shubhams12101@gmail.com, stephenebert@gmail.com, hkdavidsun@utexas.edu

## ABSTRACT

Recursive models can continue updating their latent states beyond their nominal inference budget, so an incorrect output at that budget does not show whether computation is unfinished or has entered a persistently unsuccessful regime. We study the dynamics of completion in attention- and MLP-based Tiny Recursive Models (TRMs) on 1,000 hard Sudoku puzzles. Extending recurrence from the nominal 16 steps to 512 steps increases cumulative exact-solve accuracy from 59.2% to 87.5% for the attention model and from 74.4% to 91.9% for the MLP model, solving more than two-thirds of the puzzles unsolved in the nominal budget. Across both architectures, latent-state motion drops sharply after the first exact solution. Completed states are typically locally contractive along the trajectory direction, even though the same local Jacobian retains strongly expanding directions. We characterize this phenomenon as trajectory-conditioned anisotropic stability. Perturbation experiments confirm this directional stability across both models. The multi-step fate of the maximally expanding direction differs: it is absorbed within 16 steps in the attention model but persists longer in the MLP model. The anisotropic-stability pattern also holds for a second attention checkpoint. Together, these results distinguish nominal-budget failure from completed computation and identify a common dynamical signature of completion across two recurrent architectures.

## 1 INTRODUCTION

Recursive models compute by iteratively updating a latent state, applying the same learned transition function at each step. This provides additional computational depth through weight sharing (Dehghani et al., 2019; Giannou et al., 2023) and enables computation to continue across multiple iterations (Graves, 2016; Schwarzschild et al., 2021; Bansal et al., 2022; Geiping et al., 2025). Hierarchical Reasoning Models (HRMs) and Tiny Recursive Models (TRMs) use this principle to build compact reasoning systems (Wang et al., 2025; Jolicoeur-Martineau, 2025). In such models, the number of recurrent steps determines how long the model can compute.

This creates a fundamental ambiguity. An incorrect answer at the end of the inference budget can mean two different things. The model may have failed, or it may simply need more computation (Anil et al., 2022). Conversely, a correct output does not tell us whether the underlying state has settled. The model may already have reached the answer, while its internal state continues to change (Knutson et al., 2026). Endpoint accuracy alone does not distinguish failure, completion, or stability.

Recurrent computation has long been studied via its state-space dynamics. Fixed points, attractors, and local Jacobians have been used to understand how recurrent models represent and transform information (Sussillo & Barak, 2013; Maheswaranathan et al., 2019). Equilibrium models make the connection between computation and fixed points explicit (Bai et al., 2019; Huang et al., 2026). Recent mechanistic studies of HRMs and TRMs have identified attractor structure, failure modes, spurious fixed points, and abrupt changes in solution quality (Efstathiou & Balwani, 2026; Ren & Liu, 2026). Complementary work on recurrent-depth reasoners has studied settling and the conditions under which additional test-time depth remains useful and stable (Viakhirev et al., 2026), while recent dynamical-systems analyses connect prolonged reasoning to transient chaos and fractal basin structure (Lai et al., 2026).

We ask a complementary question: what changes in the recurrent state when a particular computation finishes? Rather than comparing all trajectories at the same recurrent step, we align each problem to the step at which it is first solved. We then study completion, latent motion, and local stability separately. Completion is the first step at which the exact solution is produced. Latent motion measures how much the recurrent latent state changes from one step to the next. Stability describes how perturbations of that state grow or shrink. We measure it using the local Jacobian and controlled finite perturbations. This lets us distinguish directional from uniform stability. The model may become stable along the direction its own computation follows, even though the local Jacobian contains an expanding direction.

We study this question in attention and MLP-based TRMs on the same 1,000 hard Sudoku puzzles. Both models have a nominal budget of 16 recurrent steps. We continue the unchanged recurrence to 512 steps. Cumulative exact solution rises from 59.2% to 87.5% for the attention model and from 74.4% to 91.9% for the MLP model. Among the puzzles that are unresolved at step 16, 69.4% and 68.4% are solved later, respectively. In both models, more than two-thirds of the apparent failures in the nominal budget are unfinished computations. This separates the puzzles into three completion groups. EARLY solvers are solved by step 16. LATE solvers are solved between steps 17 and 512. PERSISTENT puzzles remain unsolved through step 512. Once aligned with their own solution time, EARLY and LATE solvers show a similar completion pattern. Their latent motion decreases, and they enter a directionally stable regime. The PERSISTENT trajectories remain much more active and show a different stability profile.

Completion has a directional stability signature. Latent-state motion drops sharply after completion. Completed states are locally contractive along the direction in which the model is moving. However, the same local Jacobian retains strongly expanding directions. The recurrence, therefore, does not become uniformly contractive when the task is solved. Perturbation experiments confirm this directional contrast. Perturbations along the most expanding direction grow much more than perturbations of the same size along the model’s natural direction of motion. We call this coexistence trajectory-conditioned anisotropic stability. The computation becomes stable along the direction of motion while the recurrent map retains locally expanding directions.

We repeat the full analysis in the attention and MLP models, using the same puzzles, recurrent schedule, probe protocol, and statistical procedures. The qualitative features replicate: unsolved trajectories remain active, completed trajectories settle, the natural direction is locally contractive after completion, and expanding directions remain in the Jacobian. But after completion, perturbations along the most expanding direction are absorbed within 16 steps in the attention model (§3.4) and persist in the MLP model (§3.5).

We also present supporting evidence beyond the two primary hard Sudoku runs. A second attention checkpoint, Attention-B, shows a similar anisotropic-stability pattern (Appendix G.7, Table 5). In Easy Sudoku, most puzzles are solved within the first few recurrent steps and fall into the EARLYsolver regime. Their latent dynamics closely resemble those of the EARLY solvers in Hard Sudoku (Appendix J). In Maze-Hard, a separate 30 × 30 grid task, most mazes are also solved within the first few recurrent steps and therefore behave mainly like EARLY solvers. Maze-Hard also shows that exact-match accuracy can be misleading. Some predictions differ from the ground-truth reference path, but are still valid solutions (Appendix K). The full extended-recurrence, Jacobian, and perturbation analyses are therefore centered on Hard Sudoku, where both early and late completion are well represented.

## 2 EXPERIMENTAL FRAMEWORK

## 2.1 MODELS, TASK, AND EXTENDED RECURRENCE

We use the TRM architecture of Jolicoeur-Martineau (2025) (see its Figures 1 and 3) and evaluate two trained checkpoints on the 1,000 hard Sudoku puzzles. Attention-A is a 6.83M-parameter recurrent transformer checkpoint (Gao, 2025a; Samsung SAIL Montreal, 2025). The MLP checkpoint replaces attention with sequence-axis MLP mixing and removes positional encoding (Gao, 2025c). Both use the same recurrent schedule, probe protocol, and statistics. The configuration details are shown in Appendices C and G.1.

The 1,000 puzzles come from the Sudoku-Extreme test split (Wang et al., 2025). All belong to its puzzles4 forum hardest 1905 source, the forum hardest 1905 list collected on the Enjoy Sudoku players’ forum and distributed with the tdoku benchmark suite (Dillon, 2019). Both checkpoints have a nominal 16-step outer budget. We reproduce the 16-step evaluation and continue the learned recurrence to a diagnostic horizon of 512 steps with the same weights, inputs, and pre-processing.

## 2.2 RECURRENT STATE, MOTION, AND COMPLETION GROUPS

In the recurrent step t, the model carries two latent states, $Z _ { H } ^ { ( t ) }$ and $Z _ { L } ^ { ( t ) }$ . We combine them into a single state,

$$
s _ { t } = ( Z _ { H } ^ { ( t ) } , Z _ { L } ^ { ( t ) } ) , \qquad s _ { t + 1 } = F _ { \theta } ( s _ { t } ; x ) .\tag{1}
$$

All state-space analyses use this joint state, with dimension $2 \times 9 7 \times 5 1 2$ . We measure how much each latent state component $Z _ { H } , Z _ { L }$ changes between consecutive steps using the relative update

$$
\delta _ { B } ( t ) = \frac { \| B ^ { ( t ) } - B ^ { ( t - 1 ) } \| _ { 2 } } { \| B ^ { ( t - 1 ) } \| _ { 2 } } , \qquad B \in \{ Z _ { H } , Z _ { L } \} .\tag{2}
$$

A small value of $\delta _ { B } ( t )$ means that the latent state changes little from one step to the next. For each puzzle $i ,$ let $\tau _ { i }$ denote the first recurrent step at which the decoded Sudoku is exactly correct. We divide the puzzles into three groups: EARLY solvers have $\tau _ { i } \leq 1 6$ , LATE solvers have $1 6 < \tau _ { i } \le 5 1 2$ and PERSISTENT puzzles remain unsolved through step $^ { 5 1 2 }$ . When comparing trajectories around completion, we measure time relative to the first solve: $r = t - \tau _ { i }$ . Here, $r = 0$ is the solving step; negative values precede completion, and positive values follow.

## 2.3 LOCAL STABILITY ALONG THE TRAJECTORY

To characterize the dynamics along the model’s trajectory, we linearize the recurrent map at each visited state,

$$
J _ { t } = \left. \frac { \partial F _ { \theta } } { \partial s } \right| _ { s _ { t } } .\tag{3}
$$

We use it to distinguish stability along the direction followed by the model’s trajectory from the largest amplification available in the surrounding state space. We define the unit trajectory direction

$$
d _ { t } ^ { \mathrm { o u t } } = \frac { s _ { t + 1 } - s _ { t } } { \| s _ { t + 1 } - s _ { t } \| _ { 2 } } ,\tag{4}
$$

whenever $\lVert s _ { t + 1 } - s _ { t } \rVert _ { 2 }$ is numerically nonzero. Its one-step Jacobian gain is

$$
\gamma _ { t } ^ { \mathrm { n a t } } = \| J _ { t } d _ { t } ^ { \mathrm { o u t } } \| _ { 2 } .\tag{5}
$$

Perturbations along the trajectory direction therefore contract to first order when $\gamma _ { t } ^ { \mathrm { n a t } } < 1$ and grow when $\gamma _ { t } ^ { \mathrm { n a t } } > 1$ . Since $\dot { \Delta _ { t + 1 } } \approx J _ { t } \Delta _ { i }$ <sub>t</sub> for $\Delta _ { t } ~ = ~ s _ { t + 1 } - s _ { t } , \gamma _ { t } ^ { \mathrm { n a t } }$ approximates the ratio $\| \Delta _ { t + 1 } \| _ { 2 } / \| \Delta _ { t } \| _ { 2 }$ <sub>2</sub> of successive updates wherever the linearization is accurate. Thus, $\gamma _ { t } ^ { \mathrm { n a t } }$ provides a local dynamical interpretation of shrinking or growing latent updates. Its additional value comes from comparing the gain along the direction actually followed by the trajectory with the amplification available along other directions at the same state. In particular, $\gamma _ { t } ^ { \mathrm { n a t } } < 1$ indicates contraction only along the local trajectory direction and does not imply contraction of the full surrounding state space.

To measure the strongest local amplification available at the same state, we compute the spectral norm of the Jacobian,

$$
\sigma _ { \operatorname* { m a x } } ( J _ { t } ) = \operatorname* { m a x } _ { \| v \| _ { 2 } = 1 } \| J _ { t } v \| _ { 2 } = \sigma _ { 1 } ( t ) .\tag{6}
$$

Let $v _ { 1 } ( t )$ denote a corresponding leading right singular vector,

$$
v _ { 1 } ( t ) \in \arg \operatorname* { m a x } _ { \| v \| _ { 2 } = 1 } \| J _ { t } v \| _ { 2 } .\tag{7}
$$

(a)  
![](images/1483622765d20c3294a813e9536086c1f379b26419476da8b2715e64ecba0181.jpg)

![](images/15f8f35f104c777dd8c186f36559bdcbfcd830f760944466f96daabaec015e3c.jpg)

![](images/f40484d1a0f940ce22575ade48264542b296cb66627efab4afa6d514addc7349.jpg)  
Figure 1: The nominal horizon truncates active computation. (a) Cumulative exact solve rate over $^ { 5 1 2 }$ outer steps with Wilson bands; the dashed line is the nominal 16-step budget. (b,c) Median relative latent update with interquartile bands for the three completion groups. LATE solvers are as mobile as PERSISTENT ones at step 16 and settle only when they solve. PERSISTENT trajectories never settle.

Under one linearized recurrent step,

$$
J _ { t } v _ { 1 } ( t ) = \sigma _ { 1 } ( t ) u _ { 1 } ( t ) ,\tag{8}
$$

where $u _ { 1 } ( t )$ is the corresponding leading left singular vector. Here $v _ { 1 } ( t )$ is the most expanding local perturbation direction, $\sigma _ { 1 } ( t )$ is its one-step amplification factor, and $u _ { 1 } ( t )$ is its output direction.

The comparison between $\gamma _ { t } ^ { \mathrm { n a t } }$ and $\sigma _ { 1 } ( t )$ distinguishes stability along the trajectory from worst-case local stability. In particular,

$$
\gamma _ { t } ^ { \mathrm { n a t } } < 1 \qquad \mathrm { a n d } \qquad \sigma _ { 1 } ( t ) > 1\tag{9}
$$

indicates anisotropic local dynamics: to first order, the trajectory-aligned direction is contracting even though expanding directions remain available in the surrounding state space.

We compute the leading singular triplet $( \underline { { \sigma } } _ { 1 } , v _ { 1 } , u _ { 1 } )$ ) without explicitly constructing the Jacobian. We apply matrix-free power iteration to $J _ { t } ^ { \top } J _ { t }$ using forward-mode Jacobian-vector products and reverse-mode vector-Jacobian products. Implementation details, convergence checks, and numerical validation appear in Appendix E. We also compute the same directional gains for probes restricted to either the $Z _ { H }$ or $Z _ { L }$ block.

The squared alignment $A _ { 1 } ( t ) = \vert \langle d _ { t } ^ { \mathrm { o u t } } , v _ { 1 } ( t ) \rangle \vert ^ { 2 }$ measures how closely the model’s own trajectory aligns with the most expanding direction. We compare this $A _ { 1 } ( t )$ with an isotropic random-direction baseline evaluated at the same state.

Finally, we measure whether the amplified direction persists across recurrent steps. We define

$$
T _ { 1 } ( t ) = \lvert \langle u _ { 1 } ( t ) , v _ { 1 } ( t + 1 ) \rangle \rvert ^ { 2 } .\tag{10}
$$

A large $T _ { 1 }$ means that the direction produced by maximal amplification at step t is aligned with the most expanding input direction at the next step. A small value means that this direction is largely rotated away. To probe stability beyond a single step, we measure finite-horizon growth by perturbing $s _ { t }$ along a unit direction v, with $\tilde { s } _ { t } = s _ { t } + \varepsilon v$ . After evolving both trajectories for k recurrent steps, let

$$
\delta s _ { t + k } ^ { ( v ) } = \tilde { s } _ { t + k } - s _ { t + k }
$$

denote their separation. We define

$$
\lambda _ { k } ( v ) = \frac { 1 } { k } \log \left( \frac { \| \delta s _ { t + k } ^ { ( v ) } \| _ { 2 } } { \varepsilon } \right) .\tag{11}
$$

Negative values indicate decay of the perturbation over k steps, while positive values indicate growth.   
This is a finite-horizon, trajectory-dependent measure of perturbation growth.

![](images/2126e7813ad183fe0d22059d10c518021fd398d3f8a94900ebcac9ce86952bea.jpg)

![](images/e1b52a5ad64162abacbf961040119053edd10a41633a3bf7dfe6baeea374eb63.jpg)

$$
r = t - \tau _ { i }
$$

![](images/5affd619b49cd8846da7345f1db09922468e8a1128a342f274e144c51c26ada9.jpg)  
Figure 2: Settling is aligned to each puzzle’s own completion. (a,b) Median relative update on the solvealigned axis $r = t - \tau _ { i } ,$ with interquartile bands. Motion is roughly constant before completion, spikes at $r = 0 .$ and then falls by more than an order of magnitude. The EARLY and LATE curves coincide once aligned, although their absolute solve times differ by an order of magnitude. (c) Distribution of $\tau _ { i }$ for LATE solvers.

## 3 RESULTS

## 3.1 EXTENDED RECURRENCE REVEALS UNFINISHED COMPUTATION

Continuing the same recurrence beyond its nominal 16-step horizon changes the picture substantially. The exact solve rate rises from 59.2% at step 16 to 87.5% by step 512 (Figure 1a). Of the 408 puzzles that are still unsolved at step 16, 283 are solved later. About two-thirds of the apparent failures at the nominal horizon are not failures at all. They are computations that have not yet finished. LATE solvers typically need many more steps: their median first solve occurs at step 62 (interquartile 33-130), although some solve much later.

The latent dynamics of these late solvers support the same interpretation. At step 16 (Figure 1b,c), late solvers are still moving strongly and look much more like persistently unsolved trajectories than early solvers. For example, their median $\delta _ { L }$ is 0.763, compared with only 0.025 for early solvers. By step 512, however, late solvers have settled to the same low-motion regime as early solvers, while persistent trajectories remain highly active (Figure 1b,c). This yields three populations to analyze: EARLY solvers, LATE solvers, and PERSISTENT puzzles.

Puzzle difficulty does not explain this separation. Persistent puzzles are not systematically harder according to the dataset ratings. If anything, their median rating is slightly lower. Full difficulty statistics are reported in Appendix D.

## 3.2 LATENT SETTLING ALIGNS WITH COMPUTATIONAL COMPLETION

When trajectories are aligned to each puzzle’s first solve time $\tau _ { i } ,$ the EARLY and LATE solvers follow a similar pattern (Figure 2). For LATE solvers, latent motion stays roughly constant before completion and then drops sharply after the solution is reached. In $Z _ { L }$ , the median motion after completion is less than half of the pre-solve level. After completion, LATE solvers settle to the same low-motion regime as EARLY solvers.

The solving step is a large reorganization. The strongest change occurs exactly at the solving step, $r = 0 ( \mathrm { F i g u r e } 2 \mathrm { a } , \mathrm { b } )$ . For LATE solvers, the median $\delta _ { H }$ rises from 0.238 at $r = - 1$ to 0.878 at $r = 0$ . For 99.3% of LATE solvers, the $Z _ { H }$ update at the solving step is larger than at every earlier step in the trajectory. The effect is much weaker in $Z _ { L }$ , where the solving step is the largest update for only 35% of puzzles. This spike is not explained simply by a change in the decoded grid. Many earlier steps also change the prediction without solving the puzzle, but their $Z _ { H }$ updates are much smaller. The solving step, therefore, marks an unusually large reorganization of the high-level latent state. It is followed immediately by a sharp reduction in motion.

Solutions are usually stable. Solved puzzles stay in the recurrence, and almost all of them remain solved. Of the 875 puzzles that solve at least once, 859 never lose the solution again. The remaining

Table 1: Local stability at the nominal horizon and at step 512, by completion group. $\gamma ^ { \mathrm { n a t } }$ is the Jacobian gain along the model’s own direction of travel. $\sigma _ { \mathrm { m a x } }$ is the worst-case gain over all directions in the same state. $\mathrm { P r } [ \gamma ^ { \mathrm { n a t } } < 1 ]$ represents the fraction of states in that group having $\gamma ^ { \mathrm { n a t } } < 1$
<table><tr><td>Step</td><td>Group</td><td> $\gamma ^ { \mathrm { n a t } }$ </td><td> $\mathrm { P r } [ \gamma ^ { \mathrm { n a t } } < 1 ]$ </td><td> $\sigma _ { \mathrm { m a x } }$ </td></tr><tr><td rowspan="3">16</td><td>EARLY</td><td>0.83</td><td>0.74</td><td>25.7</td></tr><tr><td>LATE</td><td>2.28</td><td>0.24</td><td>392.2</td></tr><tr><td>PERSISTENT</td><td>3.44</td><td>0.12</td><td>529.4</td></tr><tr><td rowspan="3">512</td><td>EARLY</td><td>0.25</td><td>0.96</td><td>26.8</td></tr><tr><td>LATE</td><td>0.31</td><td>0.92</td><td>26.9</td></tr><tr><td>PERSISTENT</td><td>2.09</td><td>0.30</td><td>316.9</td></tr></table>

![](images/20c876854dd77f13f5563adbe586a7ccdf08b3d7c7ded2f6ae0a60d0bc262394.jpg)

(b)  
![](images/16be6f056595f7ee8e5fc6c5f8d4bc8a27a57809a4e9003caa29e68b50dacd47.jpg)

(c)  
![](images/99922356e6be426952a9de89f46ca86afc835c40efc6178594d806dee01acbcb.jpg)  
Figure 3: Stability is conditioned on the direction of travel. (a) Median natural-direction gain $\gamma _ { t } ^ { \mathrm { n a t } }$ by group and input-state step, with interquartile bands. The dotted line $\mathrm { i s } \gamma ^ { \mathrm { n a t } } = 1$ . (b) Fraction of states with $\gamma _ { t } ^ { \mathrm { n a t } } < \mathrm { 1 }$ 1, with Wilson bands. (c) Every probed state: $\gamma _ { t } ^ { \mathrm { n a t } }$ against $\sigma _ { \operatorname* { m a x } } ( J _ { t } )$ . No state has $\sigma _ { \operatorname* { m a x } } < 1$ , yet most completed states sit below $\gamma ^ { \mathrm { n a t } } = 1$

16 temporarily revert, but all recover by step 512. We therefore use the first exact solution as the completion time.

## 3.3 COMPLETED COMPUTATION IS DIRECTIONALLY STABLE, NOT UNIFORMLY CONTRACTIVE

Unlike our earlier analysis of latent settling, Jacobian analysis asks whether the perturbations along different state-space directions are amplified or contracted. The completed trajectories are typically locally contracting along their own direction of motion, while the same local Jacobian has strongly expanding directions. We evaluate the Jacobian at 750 visited states, corresponding to 150 puzzles at five fixed recurrent steps each. The recovered singular vectors have unit norm up to numerical precision. Additional numerical checks and implementation details are in Appendix E.

Table 1 and Figure 3 show the main result. At step 16, the three completion groups have different natural-direction gains. EARLY solvers have a median $\gamma ^ { \mathrm { n a t } }$ of 0.83, whereas LATE and PERSISTENT trajectories remain locally expansive, with medians of 2.28 and 3.44.

LATE solvers change once they finish computing. By step 512 (Table 1), their median $\gamma ^ { \mathrm { n a t } }$ has fallen to 0.31, close to the EARLY value of 0.25 (Figure 3a). Most completed states then contract locally in the direction the model is moving. 96% of EARLY states and 92% of LATE states have $\gamma ^ { \mathrm { n a t } } < 1$ At that point, every LATE solver has completed, so the two groups occupy the same directionally stable regime. Because Figure 3 compares fixed recurrent steps, it shows where the groups end up rather than when the change happens. The solve-aligned transition is in latent motion: aligned to their own solve step, LATE and EARLY solvers show the same drop in latent updates (Figure 2a,b). PERSISTENT trajectories remain different, with median $\gamma ^ { \mathrm { n a t } } = 2 . 0 \dot { 9 }$

This stability is not a property of the full recurrent map. The worst-case Jacobian gain remains larger than one at every probed state. At step 512, $\sigma _ { \mathrm { m a x } }$ is about 27 for both completed groups and above 300 for PERSISTENT trajectories. Contraction along the natural direction and expansion along some other direction coexist at 62.1% of probed states (Figure 3c).

![](images/6372d25da35d71725e057d81fc00600b91a7a0a21ee4b871c07486622c69a231.jpg)

(b)  
![](images/d500bddef12a1b6c0ed0366fce7431be49e1abea6a5cc466d5c29eae3513f2cc.jpg)

(c)  
![](images/fd424759d958ecc548c203b84ad4410de83fde455ee277719911a5dd0d993be9.jpg)  
leading singular v<sub>1</sub>; natural trajectory; random in $Z _ { H ; } \quad \cdot \circ ^ { }$ random isotropic  
Figure 4: An expanding direction is present, but the trajectory is not orthogonal to it. Norm-matched perturbations $( 1 0 ^ { - 4 }$ of the state norm) along four directions, followed for k further steps, at (a) states whose computation has completed and (b) states still computing. In (b) every direction, including a random one, reaches the scale of the state itself. These curves measure trajectory divergence. (c) Squared overlaps: the trajectory’s overlap with $v _ { 1 }$ is tiny but distinctly above the isotropic baseline. The transport of the leading direction across one step is of the same order. The “random isotropic” is the overlap of $v _ { 1 }$ with a random unit direction drawn uniformly from the full hidden-state space, and “random $Z _ { H } { ^ { , , } }$ uses a random unit direction drawn from the $Z _ { H }$ subspace only.

Probes restricted to $Z _ { H }$ or $Z _ { L }$ show the same pattern. At step 512, the median $Z _ { L }$ -probe gain is 0.023 for EARLY solvers and 0.021 for LATE solvers, compared with 0.309 for PERSISTENT trajectories. Finite-horizon growth over the 16 steps after step 256 is also negative for both completed groups and positive for PERSISTENT trajectories. Full results are reported in Appendix F.

## 3.4 FATE AND GEOMETRY OF THE EXPANDING DIRECTION

Although $\sigma _ { \operatorname* { m a x } } > 1$ at every probed state, this strong local expansion does not destabilize completed trajectories. We measure how the trajectory aligns with the most-expanding direction $v _ { 1 }$ and track perturbations injected along that direction.

The trajectory does not avoid $v _ { 1 }$ . The alignment $A _ { 1 }$ is small, with median $4 . 1 5 \times 1 0 ^ { - 5 }$ over 750 states. However, this is about four times larger than the isotropic random-direction baseline of $9 . 6 3 \times 1 0 ^ { - 6 }$ . The trajectory is therefore not orthogonal to the direction of expansion. The overlap is small because the state space has $\sim 1 0 ^ { 5 }$ dimensions.

A $v _ { 1 }$ perturbation and its later fate depend on computation completeness. A perturbation of size $1 0 ^ { - 4 }$ of the state norm along $v _ { 1 }$ grows by 44.4× after one step, against 0.75× along the natural direction (medians over the 180 fixed-step probe states). Paired state by state over all 264 probes, the median v<sub>1</sub>-to-natural ratio is about $8 3 \times$ . The median ratio of one-step $v _ { 1 }$ amplification to the estimated $\sigma _ { \mathrm { m a x } }$ is 1.00. Over the 147 completed probe states (99 fixed-step and 48 solve-aligned), the median $v _ { 1 }$ perturbation grows by about $2 8 \times$ initially but falls to 0.11 of its injected size by $k = 1 6$ . At states that are still computing, perturbations in all tested directions eventually produce large trajectory separation (Figure 4a,b).

The expanding direction is not carried forward. The amplified output direction $u _ { 1 } ( t )$ has very little overlap with the next step’s most expanding input direction. The transport defined earlier is $T _ { 1 } ( t ) \approx 4 . 9 \times 1 0 ^ { - 5 }$ . Thus, a direction that expands strongly in one step is largely rotated away before the next step. Strong one-step expansion, therefore, need not produce sustained multi-step growth. Non-normal recurrent dynamics can amplify a perturbation transiently without sustained growth Kerg et al. (2019), although we do not test whether this mechanism operates here. The leading input direction $v _ { 1 }$ lies almost entirely in $Z _ { H }$ , while its amplified output lies mostly in $Z _ { L }$ . The strongest local amplification, therefore, acts mainly from $Z _ { H }$ into $Z _ { L }$ . This is also consistent with the latent supported probes in Appendix F.

(a)  
![](images/377105e9e5a7f96962fa15cb82ba4d5d766b484c7b91301fc6a13582739e20d1.jpg)

(b)  
![](images/ca74be966bd7e80854bdf0c894b7651cc1891400b269330f1b2d19d480de46b5.jpg)

(c)  
![](images/041d4fa21028a2a3962ce7a8e8006cdae84f0cefceafa31ddf79be8121835c07.jpg)

(d)  
![](images/ead936715c3e1d2210fd47031f5c04fd96b0260bab7d7237f9ef8b9459233c33.jpg)  
Figure 5: The completion geometry reproduces across architectures. Attention-A in blue, MLP in red. (a) Cumulative exact solve with extended recurrence, with the nominal horizon marked. (b) LATE-solver $\delta _ { L }$ on the solve-aligned axis, each model normalized by its own pre-solve level. (c) Natural-direction gain at step 512 for completed (colored) and PERSISTENT (grey) states, with the red line at $\gamma ^ { \mathrm { n a t } } = 1$ and stars marking median $\sigma _ { \mathrm { m a x } }$ at the same states $( \sigma _ { \operatorname* { m a x } } > 1$ at every probed state in both models). (d) Norm-matched perturbations at completed fixed-step states along v<sub>1</sub> (solid markers) and along the natural direction (dotted). Both models expand $v _ { 1 }$ and contract the natural direction, but only Attention-A absorbs the $v _ { 1 }$ perturbation within 16 steps.

## 3.5 CROSS-ARCHITECTURE REPLICATION IN AN MLP RECURRENT MODEL

We repeat the full analysis on the MLP model with the same task, data, completion definitions, diagnostic horizon, and statistics (§2.1). Only the model and checkpoints change.

The completion dynamics replicate. The MLP model is a stronger solver at the nominal horizon, with 74.4% exact solve at step 16 compared with 59.2% for Attention-A. Yet many of its remaining failures are also unfinished computations. By step 512, exact solve reaches 91.9%, and 175 of the 256 puzzles unresolved at step 16 are solved later (Figure 5a). The same solve-aligned settling pattern also appears. LATE solvers remain highly mobile at the nominal horizon. Their latent motion drops sharply around their own solution time, and they then enter the same low-motion regime as EARLY solvers (Figure 5b). The solving step is again the largest $Z _ { H }$ update of the preceding trajectory for almost every LATE solver. Full statistics are given in Appendix G.

Trajectory-conditioned stability also replicates. At step 512, the median natural-direction gain is about 0.47 for both completed MLP groups, but 2.82 for PERSISTENT trajectories. At the same time, $\sigma _ { \operatorname* { m a x } } > 1$ at every probed MLP state, just as in Attention-A. Completed computations are stable along their own direction of travel, while the recurrent map still contains expanding directions (Figure 5c). This coexistence occurs at 60.4% of MLP states, close to the 62.1% observed in Attention-A.

The scale of the worst-case expansion does differ. At completed states, $\sigma _ { \mathrm { m a x } }$ is about 6.4-6.7 in the MLP model, compared with about 27 in Attention-A. Persistent trajectories, by contrast, have similar values in the two models. The qualitative geometry therefore replicates, while its magnitude does not.

The crucial difference lies in the expanding direction. In both models, a perturbation along $v _ { 1 }$ expands much more strongly than one along the natural direction. The difference appears in later recurrent steps. For the 99 completed fixed-step Attention-A states, the $v _ { 1 }$ perturbation is strongly amplified initially but falls to 0.10 of its injected magnitude by $k = 1 6$ . In the MLP model, it remains at 1.91 times its injected size (Figure 5d). To determine whether this amplification ultimately decays or persists, the perturbation must be followed beyond the current horizon of $k = 1 6$ . Transport shows similar differences: the overlap between the amplified output at one step and the most expanding input at the next is $T _ { 1 } = 1 . 8 3 \times \mathrm { { \dot { 1 } 0 ^ { - 3 } } }$ in the MLP model, compared with $4 . 9 3 \times 1 0 ^ { - 5 }$ in Attention-A. Attention-A rotates the expanding direction away after completion, whereas the MLP model carries more of it into the next step. Detailed singular-vector statistics and numerical checks are reported in Appendix G.

## 3.6 REPLICATION ACROSS TASKS AND CHECKPOINTS

Our main claims use only Attention-A and MLP, which were analyzed in full detail. As a checkpoint replication, we also run the complete 16-to-512 extended recurrence analysis on the Attention-B (PreetiMLresearcher, 2026) checkpoint. Attention-B shows the same trajectory-conditioned anisotropic-stability pattern: its exact-solve rate rises from 0.521 at step 16 to 0.726 at step 512, natural-direction gain is below one in the completed groups, and expanding directions remain in the state space. All three extended-recurrence analyses are compared in Appendix G.7, Table 5.

We also tested Easy Sudoku and Maze-Hard (Appendices J and K). On Easy Sudoku, computation typically finishes within the first few steps, and the trajectories resemble EARLY solvers. The same pattern appears in Maze-Hard with 30 × 30 grids, where most trajectories settle within the first few steps and again resemble EARLY solvers. However, in Maze-Hard, many exact-match failures are valid alternative paths.

Finally, a constraint-aware decoder recovers many puzzles missed by simple argmax decoding (Appendix I). The TRM decoder independently fills each blank with the highest-scoring digit. To test whether useful alternatives remain in the logits, we evaluate the constraint-aware decoder on Attention-B. For each blank, it keeps the top-k predicted digits and uses backtracking to search only assignments that satisfy the Sudoku row, column, box, and clue constraints. The number of solved puzzles barely changes at k = 1, 2, 3, but rises by 22.3 percentage points at k = 4.

## 4 DISCUSSION

Our results show that failure at a fixed inference depth is not equivalent to computational failure. In both models, many puzzles that remain unsolved at step 16 are completed when the same recurrent computation is continued. The latent dynamics also change around completion: state updates decrease sharply, exact solutions typically persist, and the natural trajectory direction becomes locally contracting even though the full Jacobian has strongly expansive directions. Completion, therefore, does not correspond to uniform contraction of the recurrent map. Instead, contraction emerges along the direction followed by the ongoing computation, while unstable directions remain available in the state space. This qualitative picture is shared by the attention and MLP-based models. Unfinished computations remain dynamically active, whereas completed computations settle, and trajectoryaligned contraction coexists with strong off-trajectory expansion. The clearest difference between the two architectures lies in the fate of perturbations initialized along the locally most expansive direction. In the attention model, this perturbation is rapidly rotated away from the expanding direction and subsequently attenuated, whereas in the MLP model, a larger component persists over later recurrent steps. Thus, a similar trajectory-conditioned stability can coexist with different local perturbation geometries.

## 5 LIMITATIONS

Our results have three main limitations. First, step 512 is still finite, so we cannot determine the asymptotic fate of the PERSISTENT trajectories. Second, the full Hard-Sudoku analysis covers two attention checkpoints and one MLP checkpoint. The Maze-Hard provides a separate nominal-horizon task control, but broader claims require more checkpoints, architectures, and tasks. Finally, the Jacobian analysis is local to visited states and uses a subset of the full evaluation set. The most expensive perturbation measurements use smaller subsets.

## 6 CONCLUSION

A fixed inference budget can obscure the distinction between failed and unfinished computation. In the recursive models studied here, many nominal-horizon failures are solved when the same computation is allowed to continue, and completion is accompanied by a transition to a low-motion, trajectory-stable regime.

This suggests that completion is not the same as uniform contraction of the recurrent map. Instead, stability is tied to the particular path followed by the computation. Looking at these dynamics may help us understand when a recurrent model is still computing, when it has effectively settled, and how it uses its latent state during inference.

## REPRODUCIBILITY STATEMENT

We provide the information needed to reproduce the main experimental results in Appendix A and in the supplementary code. These materials document the datasets, checkpoints, model, sampling, perturbation protocols, control points, and numerical validation used for the Attention-A, Attention-B, and MLP experiments. Statistical procedures are documented in Appendix B. All reported statistics and figures can be traced back to puzzle-level outputs. The supplementary code contains an analysis pipeline that regenerates them from those output archives without requiring model inference. The GPU procedure for regenerating the extended-recurrence results from the original checkpoints is also discussed in Appendix A.

## AI USE STATEMENT

Generative AI tools, including OpenAI Codex and Anthropic Claude, were used during several stages of this work. They assisted in implementing experimental and data-analysis code, generating plotting code, refining aspects of the experimental and statistical methodology (including the use and presentation of Wilson intervals, interquartile ranges, and confidence intervals), interpreting numerical results, and improving the clarity and presentation of the manuscript. AI tools also suggested additional diagnostic visualizations, including the analyses reported in Fig. 3c and Fig. 2c and the correlation between Sudoku rating and solve steps in Appendix D. The research direction, scientific questions, and final methodological and interpretive decisions were determined by the author. All AI-generated or AI-assisted code used for the reported experiments was reviewed and tested, and all reported numerical results and figures were obtained from executed computational experiments. All AI-assisted analyses were reviewed by the authors, who take full responsibility for the correctness of the results, claims, code, and final content of the paper.

## REFERENCES

Cem Anil, Ashwini Pokle, Kaiqu Liang, Johannes Treutlein, Yuhuai Wu, Shaojie Bai, J. Zico Kolter, and Roger B. Grosse. Path independent equilibrium models can better exploit testtime computation. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models, 2019. URL https: //arxiv.org/abs/1909.01377.

Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. In Advances in Neural Information Processing Systems, volume 35, 2022. doi: 10.52202/068431-1471. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html.

Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: A survey. Journal ofMachine Learning Research, 18(153):1–43, 2018. doi: 10.48550/arXiv.1502.05767.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=HyzdRiR9Y7.

Tom Dillon. tdoku: A fast Sudoku solver and benchmark suite. https://github.com/ t-dillon/tdoku, 2019. Includes the forum hardest 1905 list collected from the Enjoy Sudoku players’ forum.

B. Efron. Bootstrap methods: Another look at the jackknife. The Annals ofStatistics, 7(1), 1979. doi: 10.1214/aos/1176344552.

Andreas Efstathiou and Aishwarya Balwani. Recursive reasoning as attractor landscape search: Mechanistic dynamics of the tiny recursive model. In ICLR 2026 Workshop on Latent & Implicit Thinking – Going Beyond CoT Reasoning, 2026. URL https://openreview.net/forum? id=kKps9W1K7n.

Rainer Engelken, Fred Wolf, and L. F. Abbott. Lyapunov spectra of chaotic recurrent neural networks. Physical Review Research, 5(4):043044, 2023. doi: 10.1103/PhysRevResearch.5.043044.

Xin Gao. TinyRecursiveModels-Sudoku-Extreme-att. Hugging Face model repository, 2025a. Accessed 2026-08-02.

Xin Gao. TinyRecursiveModel-Maze-Hard. Hugging Face model repository, 2025b. Maze-Hard checkpoint step 9765; accessed 2026-08-23.

Xin Gao. TinyRecursiveModels-Sudoku-Extreme-mlp. Hugging Face model repository, 2025c. MLP Sudoku checkpoint step 16275; accessed 2026-08-23.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up testtime compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-1380. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 11398–11442. PMLR, 2023. URL https://proceedings.mlr.press/ v202/giannou23a.html.

Alex Graves. Adaptive computation time for recurrent neural networks, 2016. URL https: //arxiv.org/abs/1603.08983.

Benhao Huang, Zhengyang Geng, and Zico Kolter. Equilibrium reasoners: Learning attractors enables scalable reasoning, 2026. URL https://arxiv.org/abs/2605.21488.

Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025. URL https://arxiv.org/abs/2510.04871.

Giancarlo Kerg, Kyle Goyette, Maximilian Puelma Touzel, Gauthier Gidel, Eugene Vorontsov, Yoshua Bengio, and Guillaume Lajoie. Non-normal recurrent neural network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ hash/9d7099d87947faa8d07a272dd6954b80-Abstract.html.

Brandon Knutson, Amandin Chyba Rabeendran, Michael Ivanitskiy, Jordan Pettyjohn, Cecilia Diniz Behn, Samy Wu Fung, and Daniel McKenzie. On logical extrapolation for mazes with recurrent and implicit networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 22635–22643, 2026. doi: 10.1609/aaai.v40i27.39424. URL https://ojs.aaai.org/ index.php/AAAI/article/view/39424.

Jeffrey Lai, Anthony Bao, John Quinn, and William Gilpin. Fractal basins trap latent reasoning, 2026. URL https://arxiv.org/abs/2609.04963.

Niru Maheswaranathan, Alex Williams, Matthew D. Golub, Surya Ganguli, and David Sussillo. Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https:// arxiv.org/abs/1906.10720.

Rasmus Berg Palm, Ulrich Paquet, and Ole Winther. Recurrent relational networks. In Advances in Neural Information Processing Systems, 2018. URL https://arxiv.org/abs/1711. 08028.

PreetiMLresearcher. TinyRecursiveModels-Sudoku. Hugging Face model repository, 2026. Twentyfive attention-model snapshots; accessed 2026-08-12.

Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, and Tolga Birdal. Grokking at the edge of numerical stability. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=TvfkSyHZRA. arXiv:2501.04697.

Zirui Ren and Ziming Liu. Are your reasoning models reasoning or guessing? a mechanistic analysis of hierarchical reasoning models, 2026. URL https://arxiv.org/abs/2601.10679.

Samsung SAIL Montreal. Tinyrecursivemodels. https://github.com/ SamsungSAILMontreal/TinyRecursiveModels, 2025. Reference implementation; accessed 2026-08-02.

Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, and Tom Goldstein. Can you learn an algorithm? Generalizing from easy to hard problems with recurrent networks. In Advances in Neural Information Processing Systems, volume 34, pp. 6695–6706, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 3501672ebc68a5524629080e3ef60aef-Abstract.html.

Jeffrey Seely, Yuki Imajuku, Tianyu Zhao, Edoardo Cetin, and Llion Jones. Sudoku-Bench: Evaluating creative reasoning with Sudoku variants. arXiv preprint arXiv:2505.16135, 2025. URL https://arxiv.org/abs/2505.16135.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021. URL https://arxiv.org/abs/2104.09864.

David Sussillo and Omri Barak. Opening the black box: Low-dimensional dynamics in highdimensional recurrent neural networks. Neural Computation, 25(3):626–649, 2013. doi: 10.1162/ NECO a 00409.

Ivan Viakhirev, Kirill Borodin, Amirah Almutairi, Serguei Barannikov, Maxim Abramov, and Grach Mkrtchian. Think shallow, solve deep: Controlling recurrent dynamics for reliable test-time depth, 2026. URL https://arxiv.org/abs/2608.18222.

Ryan Vogt, Maximilian Puelma Touzel, Eli Shlizerman, and Guillaume Lajoie. On Lyapunov exponents for RNNs: Understanding information propagation using dynamical systems tools. Frontiers in Applied Mathematics and Statistics, 8:818799, 2022. doi: 10.3389/fams.2022.818799.

Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025. URL https://arxiv.org/ abs/2506.21734.

Po-Wei Wang, Priya L. Donti, Bryan Wilder, and Zico Kolter. SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver. In International Conference on Machine Learning, 2019. URL https://arxiv.org/abs/1905.12149.

Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal ofthe American Statistical Association, 22(158):209–212, 1927. doi: 10.1080/01621459.1927.10502953.

## A REPRODUCIBILITY AND IMPLEMENTATION DETAILS

The extended-recurrence results use three complete analyses. Attention-A uses step 21700 and took 91.5 minutes; MLP uses step 16275 and took 82.1 minutes; and Attention-B uses step 65100 and took 88.3 minutes. All three use the same hard sudoku.csv data file and trm.py model source, PyTorch 2.11.0+cu128 on one CUDA device. The run uses global seed

0, the joint continuous carry $Z _ { H } \oplus Z _ { L }$ , a nominal horizon of 16, and a diagnostic horizon of 512. Attention-A and Attention-B use PyTorch’s mathematical scaled-dot-product attention backend, explicitly pinned as SDPBackend.MATH to prevent hardware-dependent backend selection. The forward recurrence uses bfloat16, while saved latent states and Jacobian, JVP, and VJP calculations use float32. MLP contains no attention operation. The analysis configurations are identical apart from the model and checkpoint paths. For each analysis, we save the full experimental setup in its own run config.json and run manifest.json. This includes the group definitions, the 150-puzzle probe subset (50 per group), the 36-puzzle expensive subset (12 per group), and all perturbation settings. Jacobian probes use a relative perturbation of $\varepsilon = 1 0 ^ { - 4 }$ , while noise experiments use $\sigma ~ \in ~ \{ 0 . 0 2 , 0 . 1 \}$ with seeds {0, 1, 2}. We also record the use of 20,000 bootstrap replicates and five-fold cross-validation. Numerical checks, including the 16-step horizon and the singular-triplet $( \sigma _ { 1 } , v _ { 1 } , u _ { 1 } )$ identities, are in manifests/numerical checks.json and singular geometry/table10b numerical checks.csv within each archive. Every number in this paper can be traced back to a puzzle-level CSV, with the mapping given in ICLR RESULT INVENTORY.md for Attention-A and MLP RESULT INVENTORY.md for MLP. A single CPU, python analysis/run all.py, recomputes every reported quantity from those puzzle-level files and regenerates all figures in about a minute, with no model inference. REPRODUCE.md gives the environment, the pinned package versions, and the expected outputs, and requirements.txt pins the analysis environment. Regenerating the results from the checkpoints needs a separate GPU path, which is also documented there. Attention-B follows a similar path. Its entries in Table 5 (Appendix G.7) come from the extended run with the math SDPA backend, which solves 521 puzzles at step 16. Appendices H and I use an earlier nominal-horizon evaluation of the same checkpoint, which solves 531. Easy Sudoku (Appendix J), Maze-Hard (Appendix K), and the constraint decoder (Appendix I) come from separate 16-step evaluations.

## B STATISTICAL PROCEDURES

Statistical procedures. We report binary proportions with Wilson 95% intervals (Wilson, 1927). Continuous per-puzzle quantities use a percentile bootstrap (Efron, 1979) over puzzles with 20,000 replicates. Every puzzle is an independent sampling unit. Repeated recurrent steps, probe directions, perturbation horizons, and noise seeds are treated as repeated measurements of the same puzzle rather than as independent samples. For solve-aligned analyses, we preserve the pairing between the preand post-solve windows by resampling whole puzzles. Jacobian analyses use a fixed subset of 150 puzzles, with 50 puzzles from each completion group. The more expensive direct-perturbation and transport analysis uses 12 puzzles per group. Perturbations injected at step 512 are followed for up to 16 further steps, beyond the 512-step run. Excluding these states moves the Attention-A median $v _ { 1 }$ separation at k = 16 from 0.111 to 0.115 over all completed states and from 0.102 to 0.108 over fixed-step states, and leaves the MLP value of 1.91 unchanged.

Numerical precision. The standard forward pass uses bfloat16, matching the released model checkpoint. For the Jacobian analysis, we upcast the recurrent computation to float32, disable TF32, and use the mathematical SDPA attention backend. In a matched float32 control, the differentiable recurrence exactly reproduces the reference float32 forward pass, with a maximum absolute difference of 0.

Terminology. We use several dynamical terms in a limited sense. “Settling” means that the observed relative state updates become small. It does not imply that the map is contractive. “Directional gain” is the one-step Jacobian gain along a specified direction. “Finite-horizon growth” describes amplification over a fixed number of future recurrent steps. “Completion” means the first exact decode. A small fraction of trajectories later revert, as quantified in §3.2. A “block” probe is restricted to $Z _ { L }$ or to $Z _ { H }$

## C TRM SETUP

The attention-based TRM configuration is summarized in Table 2. Jolicoeur-Martineau (2025) gives the complete architecture and pseudocode (Figures 1 and 3 of that paper).

Table 2: Configuration of the attention-based TRM checkpoints.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>H_cycles</td><td>3</td></tr><tr><td>L_cycles</td><td>6</td></tr><tr><td>L_layers</td><td>2</td></tr><tr><td>H_layers</td><td>0</td></tr><tr><td>Hidden size</td><td>512</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>MLP expansion factor</td><td>4</td></tr><tr><td>Position encoding</td><td>Rotary</td></tr><tr><td>puzzle_emb_len</td><td>16</td></tr><tr><td>puzzle_emb_ndim</td><td>512</td></tr><tr><td>halt_max_steps</td><td>16</td></tr><tr><td>no_ACT_continue</td><td>true</td></tr><tr><td>Forward dtype</td><td>bfloat16</td></tr><tr><td>Loss</td><td>StableMax cross-entropy via ACTLossHead</td></tr><tr><td>Number of parameters</td><td>6,829,570</td></tr></table>

The loss in the above table is StableMax cross entropy loss (Prieto et al., 2025). The input sequence contains 81 grid-cell tokens prefixed by 16 puzzle-embedding tokens, giving 97 token positions. The recurrent carry contains two latent states, z<sub>H</sub> and $z _ { L }$ , each of shape $9 7 \times 5 1 2$ . Hence the joint continuous carry has dimension

$$
\mathrm { d i m } ( z _ { H } , z _ { L } ) = 2 \times 9 7 \times 5 1 2 = 9 9 , 3 2 8 .
$$

In evaluation mode, ACT halts at the final permitted step and resets the carry on the subsequent call. Consequently, obtaining an uninterrupted N-step trajectory requires setting the halt cap to N. For the long-horizon evaluations in this work, we therefore set halt max steps=512; see §2.1.

## D EXTENDED-RECURRENCE RESULTS

## SOLVE RATE

The cumulative solve rate is reported in §3.1. We distinguish it from the instantaneous solve rate. A puzzle enters the cumulative count as soon as it is solved exactly for the first time. The instantaneous count instead asks whether it is solved at that particular step. These two counts are identical at al reported checkpoints except step 32. By then, 660 puzzles had been solved at least once, while 659 were correct at step 32. One puzzle had briefly reverted after being solved. Since our completion groups are defined by the first exact solve, all group assignments use the cumulative count.

## PUZZLE DIFFICULTY BY COMPLETION GROUP

Datasets. Hard Sudoku: 1,000 puzzles from the test split of Sudoku-Extreme (Wang et al., 2025) (https://huggingface.co/datasets/sapientinc/sudoku-extreme), all from its puzzles4 forum hardest 1905 source, which is the forum hardest 1905 list of the tdoku benchmark suite (Dillon, 2019). Sudoku-Extreme permutes every puzzle by row, column, box, and digit and keeps its train and test splits mathematically inequivalent. The dataset carries a per-puzzle rating defined by Sudoku-Extreme as the number of backtracks the tdoku solver (Dillon, 2019) needs to solve the puzzle. Table 3 reports the ratings by completion group.

The puzzles the model never solves are rated easier than the ones it solves inside the nominal budget. Two rank statistics say the same thing. Among the 875 eventual solvers, Spearman correlation between rating and first solve time is $\rho = - 0 . 0 9 7 \left( p = 0 . 0 0 4 \right)$ : higher-rated puzzles are solved marginally earlier. Across all 1,000 puzzles, the correlation between rating and ever solving is $\rho = + 0 . 0 9 5 \ : ( p =$ 0.003). Binned by rating quintile, the PERSISTENT fraction runs 0.112, 0.200, 0.167, 0.066, 0.075 from the lowest to the highest quintile, with no monotone trend. Difficulty therefore does not explain the completion groups.

Table 3: Puzzle difficulty by completion group on the dataset’s own rating scale. Difference rows report unpaired median differences relative to EARLY solvers, with 20,000 bootstrap replicates over puzzles in each group.
<table><tr><td>Group</td><td>n</td><td>min</td><td>Q1</td><td>median</td><td>Q3</td><td>max</td><td>mean</td><td>median 95% CI</td></tr><tr><td>EARLY solver</td><td>592</td><td>2</td><td>14.0</td><td>27</td><td>42</td><td>140</td><td>31.23</td><td>[25.0,28.5]</td></tr><tr><td>LATE solver</td><td>283</td><td>2</td><td>11.5</td><td>24</td><td>39</td><td>110</td><td>27.63</td><td>[20.0,27.0]</td></tr><tr><td>PERSISTENT unsolved</td><td>125</td><td>3</td><td>13.0</td><td>20</td><td>28</td><td>127</td><td>24.05</td><td>[18.0,23.0]</td></tr><tr><td>All puzzles</td><td>1000</td><td>2</td><td>13.0</td><td>25</td><td>40</td><td>140</td><td>29.31</td><td>[23.0, 26.0]</td></tr><tr><td colspan="9">Unpaired median difference against EARLY</td></tr><tr><td>LATE – EARLY</td><td>283</td><td></td><td></td><td>-3</td><td></td><td></td><td></td><td>[-7.0,1.0]</td></tr><tr><td>PERSISTENT – EARLY</td><td>125</td><td></td><td></td><td>-7</td><td></td><td></td><td></td><td>[-9.5,-3.0]</td></tr></table>

## E JACOBIAN IMPLEMENTATION

Our Jacobian analysis requires differentiating through one outer recurrence of the model. We therefore use a differentiable version of the recurrence in which the detach and no grad boundaries are removed. We verified that this change does not alter the forward computation. After converting the relevant operations to float32, the differentiable and original implementations agree exactly at the level of the model state.

We estimate the leading singular value of the Jacobian, $\sigma _ { 1 }$ , using matrix-free power iteration on $J ^ { \top } J .$ This avoids constructing the full Jacobian, which would be prohibitively large for our state space. Jacobian-vector and vector-Jacobian products are computed with jvp and $\forall \dot { \exists } \mathsf { p } ,$ respectively (Baydin et al., 2018). We use 10 power iterations with a convergence tolerance of $1 0 ^ { - 3 }$ . Once the leading right singular vector $v _ { 1 }$ is obtained, the corresponding left singular vector is defined as

$$
u _ { 1 } = \frac { J v _ { 1 } } { \sigma _ { 1 } } .\tag{12}
$$

Because the quantities of interest are sensitive to small numerical errors, we perform these calculations in float32 and disable TF32. We also use the mathematical SDPA backend for attention rather than the fused Flash, memory-efficient, or cuDNN SDPA backends.

We carried out numerical checks on the estimated singular triplets. Across 1,100 probe entries, the singular vectors remain normalized to numerical precision, and 98.6% of the triplets satisfy the scalar convergence criterion based on relative changes in $\sigma _ { 1 }$ . The relation $J v _ { 1 } = \sigma _ { 1 } u _ { 1 }$ holds by construction. The adjoint norm ratio $\lVert J ^ { \top } u _ { 1 } \rVert _ { 2 } / \sigma _ { 1 }$ remains within 1.3% of one across these entries.

Taken together, these checks suggest that the matrix-free procedure is sufficiently stable for the Jacobian analyses reported in the paper.

## F LATENT BLOCK PERTURBATIONS AND FINITE-HORIZON GROWTH

To probe stability beyond one step, we measure the finite-horizon separation of a clean and a perturbed trajectory. For each probed state $s _ { t } ,$ , we draw a isotropic unit direction $v _ { L }$ supported entirely in $Z _ { L }$ and set

$$
v = ( 0 , v _ { L } ) , \qquad \widetilde { s } _ { t } = s _ { t } + \varepsilon _ { t } v , \qquad \varepsilon _ { t } = 1 0 ^ { - 4 } \| s _ { t } \| _ { 2 } .
$$

After evolving both states under the same recurrent map for k steps, let

$$
\Delta s _ { t + k } ^ { ( v ) } = \widetilde s _ { t + k } - s _ { t + k } .
$$

We define

$$
\lambda _ { k } ( v ) = \frac { 1 } { k } \log \left( \frac { \| \Delta s _ { t + k } ^ { ( v ) } \| _ { 2 } } { \varepsilon _ { t } } \right) .
$$

Negative values mean that the separation at horizon k is smaller than the injected perturbation, while positive values mean that it is larger. This is a finite-horizon, trajectory-conditioned measure of perturbation growth, not an asymptotic Lyapunov exponent.

(a)  
![](images/8adc4295ea354e67839f7b050f941a40caaadd25a0b1ca53914fc8a21487c9b3.jpg)

![](images/3324786b69e04b694f2590127501b4f6b04f9017494d348eb652f1a28d86b747.jpg)

![](images/beb540adb4e37af19c6c2a91897779f2ee6247406fe8eed555e04b8431b0d8f4.jpg)  
Figure 6: (a,b) Median one-step directional gain for probe directions supported on $Z _ { L }$ and $Z _ { H } .$ , shown by group and input-state step. (c) Finite-horizon growth $\lambda _ { 1 6 }$ following a perturbation to $Z _ { L }$ . Shaded regions show interquartile ranges.

Latent block-supported probes show how the response differs between the two latent components (Figure 6 a,b). At step 16, the median gain for EARLY trajectories is below one in both blocks. The separation from PERSISTENT trajectories is more pronounced in $Z _ { H }$ , where the median gains are 0.190 for EARLY and 1.851 for PERSISTENT trajectories.

By step 512, the EARLY and LATE groups have similar responses in both latent blocks. PERSISTENT trajectories retain larger gains in $Z _ { H }$ with a median greater than one.

Finite-horizon growth shows the same distinction (Figure 6 c). For $Z _ { L }$ perturbations introduced at step 256, the median growth over the next 16 steps is $\lambda _ { 1 6 } = - 0 . 2 8 2$ in both the EARLY and LATE groups and 0.526 in the PERSISTENT group.

These quantities describe growth along the observed recurrent trajectory over a fixed number of future steps. They should therefore be interpreted as finite-horizon, trajectory-conditioned rates rather than asymptotic Lyapunov exponents (Vogt et al., 2022; Engelken et al., 2023).

## G FULL MLP MODEL CROSS-ARCHITECTURE REPLICATION

In this appendix, we study the MLP experiments in full detail. Relevant quantities are defined in §2.1 and computed on the same task and data.

## G.1 CHECKPOINT AND CONFIGURATION FOR MLP

The configuration of the MLP model used in our experiments is summarized in Table 4. Except for the token-mixing mechanism and positional encoding, its architectural and evaluation settings are identical to those of Attention-A. Both runs use the same puzzles file and the same model source.

The MLP replaces attention-based token mixing with a sequence-axis MLP and omits rotary positional encoding. The remaining architectural settings match Attention-A as shown in Table 4.

The analysis uses global seed 0 on a CUDA device. Over all 512,000 recorded step observations, no non-finite values occur, and both latent-state norms remain bounded. After conversion to float32, the differentiable recurrence also matches the original forward computation exactly.

## G.2 EXTENDED RECURRENCE AND COMPLETION GROUPS

Extending the recurrence from 16 to 512 steps raises the number of puzzles solved at least once from 744 to 919 (Figure 7a). Moreover, blank-cell accuracy improves while StableMax loss decreases.

Of the 1,000 puzzles, 744 belong to the EARLY group [71.6%, 77.0%], 175 to the LATE group [15.3%, 20.0%], and 81 remain PERSISTENT[6.6%, 10.0%]. Among the puzzles that eventually solve, the median first solve time is 3 steps for EARLY trajectories (IQR 2-6) and 43 steps for LATE trajectories (IQR 28-100).

Table 4: Configuration of the MLP model used in the experiments.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Checkpoint Number of parameters Attention-A parameters</td><td>step-16275 5,030,402 6,829,570</td></tr><tr><td>Token mixing</td><td>Sequence-axis MLP</td></tr><tr><td>mlp-t Position encoding</td><td>true None</td></tr><tr><td>H_cycles</td><td>3</td></tr><tr><td>L_cycles</td><td>6</td></tr><tr><td></td><td></td></tr><tr><td>L_layers</td><td>2</td></tr><tr><td>Hidden size</td><td>512</td></tr><tr><td>MLP expansion factor</td><td>4</td></tr><tr><td>Forward dtype</td><td>bfloat16</td></tr><tr><td>Loss</td><td></td></tr><tr><td>Nominal recurrence length</td><td>StableMax cross-entropy 16 steps</td></tr></table>

(a)  
![](images/2bbcc64f805e2a7650e47bba69f87dee995e090f4d91c6ea32a0003c1b880bfa.jpg)

![](images/733569095cf1dded958848cb48f6242a11bddae214cea99e43ae048296bd544b.jpg)

(c)  
![](images/b0b194f08914e6139216baabf3f3ecd8e12e5715343c05ce0162503ca83e1b30.jpg)  
Figure 7: MLP extended recurrence. (a) Cumulative exact solve with Wilson bands. (b,c) Median relative latent update by completion group with interquartile bands.

All 919 MLP puzzles that ever reach an exact solution keep it: P(stays solved) = 1.000 [0.9958, 1.0], with zero reversions, against $8 5 9 / 8 7 5 = 0 . 9 8 2$ in Attention-A.

The latent dynamics yield the same ordering as in Attention-A (Figure 7b,c). EARLY trajectories already have small updates at step 16 while LATE and PERSISTENT trajectories remain more active. At step 512, the LATE group reaches a similar low-motion regime to the EARLY group, whereas PERSISTENT trajectories continue to change substantially.

The difficulty rating is only weakly associated with these outcomes. Among the 919 eventual solvers, higher ratings are associated with slightly earlier first solutions $( \rho , p ) \stackrel { - } { = } ( - 0 . 1 0 5 , 0 . 0 0 1 4 )$ . The association with whether a puzzle ever solves is small $( \rho = 0 . 0 2 5 , p = 0 . 4 3 )$ . These results do not support a simple interpretation in which later or unsuccessful trajectories are harder puzzles on this rating scale.

## G.3 SOLVE-ALIGNED SETTLING AND THE SOLVING STEP

The MLP shows a sharp reduction in latent motion around completion (Figure 8a,b). Among the 175 LATE solvers, the median paired change in window-mean $\delta _ { L }$ between the pre-solve window [−8, −1] and post-solve window $[ 0 , 7 ] ~ { \mathrm { i s - 0 . 3 9 9 } } \ [ - 0 . 4 0 8 , - 0 . 3 9 1 ]$ . The corresponding change in $\delta _ { H }$ is −0.138 [−0.147, −0.132] compared with −0.029 in Attention-A.

(a)  
![](images/9aaa17be648186e69caf5e342e14bc3d2359234d17289d01559f07cfdfb65776.jpg)

![](images/ee435aff70c1a3b3b108aa361c94eab6a3ce2486095a1917e20758d4dbbae843.jpg)

![](images/20618e7dac79a077247d57a15c746e4d8ec3ee06102f19ed0a1792915ec53063.jpg)  
Figure 8: MLP solve-aligned dynamics. (a,b) Relative updates along the solve-aligned axis. (c) Distribution of first solve times for LATE solvers.

The earlier comparison between [−16, −9] and [−8, −1] shows little change in $\delta _ { L } \mathbf { : }$ : 0.006 [−0.007, 0.014]. The reduction is concentrated around completion rather than spread across the preceding windows. After solving, LATE trajectories approach a low-motion regime similar to EARLY trajectories.

The solving step itself contains a pronounced state update. The paired increase in $\delta _ { H }$ from $r = - 1$ to r = 0 is 0.508 [0.498, 0.523]. For 99.4% of LATE solvers, the $Z _ { H }$ update at the solving step is larger than at every earlier step. The solving step $Z _ { L }$ update is the largest observed up to that step for 74.3% of LATE MLP trajectories, compared with 35.3% in Attention-A.

## G.4 FIXED-TIME JACOBIAN, NATURAL-DIRECTION GAIN, AND $\sigma _ { \mathrm { m a x } }$

![](images/39c6d65a024f61c083ad609533a10f95e2d896cfd78a49b55ec1b2e891ea40a1.jpg)

(b)  
![](images/5bd014e4cc404df5f4722eb6e470342e0748a726a7b80b655aeac5ef49532c0f.jpg)

(c)  
![](images/0f999a8e4a896779cdf1bd7d26bcc8f8382161ebfe8f8ba00dd4e96b3ad8fe7c.jpg)  
Figure 9: MLP local stability at fixed recurrent steps. (a) Median natural-direction gain by group. (b) Probe gains supported on $Z _ { L }$ (solid) and $Z _ { H }$ (dashed). (c) Natural-direction gain $\gamma _ { t } ^ { \mathrm { n a t } }$ versus $\sigma _ { \mathrm { m a x } } ( \bar { J } _ { t } )$ for all probed states.

The natural-direction gain already separates the three groups at the nominal 16-step horizon (Figure 9). Median $\gamma ^ { \mathrm { n a t } }$ is 0.538 [0.494, 0.607] for EARLY, 2.271 [1.347, 2.858] for LATE, and 3.505 [2.450, 4.635] for PERSISTENT trajectories. Accordingly, 92% of EARLY states have gain below one, compared with 32% of LATE states and 12% of PERSISTENT states.

By step 512, median natural-direction gains are similar in the EARLY and LATE groups (0.470 and 0.473) while remaining above one in the PERSISTENT group (2.816). Nevertheless, $\sigma _ { \operatorname* { m a x } } > 1$ at all 750 probed states. Small gains along the natural direction coexist with expanding directions in the surrounding state space (Figure 9c).

Latent block-supported probes show the same group ordering. The EARLY and LATE groups have smaller median gains than the PERSISTENT group in both latent components (Figure 9b).

The finite-horizon measurements extend this comparison beyond one step. For $Z _ { L }$ perturbations introduced at step 256, growth over the following 16 steps is negative in 98% of EARLY states, 90% of LATE states and 12% of PERSISTENT states.

## G.5 LEADING SINGULAR-VECTOR GEOMETRY

![](images/d6a0ff9c1ff325dfd8473ae62f0f27ec5f440295d248222b8c08ec1eaa93ef2b.jpg)

(b)  
![](images/90a5f4a7a6b24f67454eadd6340a9f0f8b21187a7a37214b9abc482d125e6bf8.jpg)

![](images/a07439673c6ba29256d85ca6dea5d799cba2e8e498d913858bd52ba1e8936024.jpg)  
Figure 10: MLP leading singular-vector geometry. (a) Squared overlaps and one-step transport for the fixed-time states, together with random baselines. The “random isotropic” is the overlap of $v _ { 1 }$ with a random unit direction drawn uniformly from the full hidden-state space, and “random $Z _ { H } { ^ { \mathbf { \curlyeq } } }$ uses a random unit direction drawn from the $Z _ { H }$ subspace only. Transport is evaluated on the 180-state subset. (b) Fraction of the leading right and left singular vectors supported on $Z _ { L }$ and $Z _ { H } .$ . (c) Distribution of $\log _ { 1 0 } T _ { 1 }$ across completion groups.

The natural update has a small overlap with $v _ { 1 }$ , but it exceeds the random isotropic baseline (Figure 10a). The median squared overlap is $2 . 5 5 \times 1 0 ^ { - 5 }$ compared with $9 . 4 6 \times 1 0 ^ { - 6 }$ for the random baseline. The paired difference is $\bar { 1 . 6 1 } \times 1 0 ^ { - 5 } \ [ 1 . 3 3 \times \bar { 1 } 0 ^ { - 5 } , 2 . 1 9 \times 1 0 ^ { - 5 } ]$ . The $Z _ { H }$ -restricted comparison shows the same pattern.

In both architectures, the leading input direction $v _ { 1 }$ lies almost entirely in $Z _ { H }$ , while its amplified output $u _ { 1 }$ lies mainly in $Z _ { L }$ (Figure 10b). The $Z _ { L }$ component contains 65% of the output-direction energy in the MLP compared with 85.3% in Attention-A. The leading response is less concentrated in $Z _ { L }$ for the MLP.

The leading direction is transported more strongly between successive steps in the MLP (Figure 10c). Over the 180-state subset, median $T _ { 1 }$ is $1 . { \overset { \sim } { 8 3 } } \times 1 0 ^ { - 3 } \ [ 5 . 7 \times 1 0 ^ { - 4 } , 4 . { \overset { \cdot } { 6 } } \times 1 0 ^ { - 3 } ]$ , compared with $4 . 9 3 \times 1 0 ^ { - 5 }$ in Attention-A.

## G.6 DIRECT v<sub>1</sub> VERSUS NATURAL-DIRECTION PERTURBATIONS $v _ { 1 }$

![](images/5cba09f83a4867009dbc237ec74269ce796a6de5ee6d16c53e5765733ba4e36e.jpg)

![](images/07913284473384b1e21649edfb1696bf9ec12c6da183c7ec63e1097b26985c85.jpg)  
Figure 11: MLP response to norm-matched perturbations with initial size $1 0 ^ { - 4 }$ of the state norm. Results are separated according to whether the unperturbed computation has already completed. Panel (a) pools the 100 fixed-step and 48 solve-aligned completed states (the text quotes fixed-step medians), and panel (b) shows the 116 still-active states. Shaded regions show interquartile ranges.

At the 100 completed fixed-step states, perturbations along $v _ { 1 }$ initially expand, with a median one-step amplification of 6.58 [6.43, 6.96] (Figure 11a). After 16 steps, the median separation is 1.91 [0.87, 2.74] times the injected size. Although the point estimate remains greater than one, its confidence interval spans one, and 42% of states have separations less than the injected size. Recovery also varies between completion groups: the corresponding medians are 0.91 for EARLY states and 3.10 for LATE states.

Natural-direction perturbations contract more consistently. After 16 steps, their median separation is 0.181 [0.113, 0.234] times the injected size, with 93% of states below one. Both random controls contract more strongly still.

At the 116 still-active probe states, all four perturbation directions produce separations of approximately $4 \times 1 0 ^ { 3 }$ times the injected size by step 16. Given the initial scale of $1 0 ^ { ^ { \bullet } - 4 }$ of the state norm, these separations are of the order of the state norm (Figure 11b).

These results distinguish initial amplification from subsequent recovery. The less consistent recovery of $v _ { 1 }$ perturbations in the MLP is compatible with its larger measured direction transport.

## G.7 CROSS-CHECKPOINT AND ARCHITECTURE SUMMARY

Table 5: Cross-checkpoint and cross-architecture comparison. Completion-group quantities use step 512. γ<sup>nat</sup> and $\sigma _ { \mathrm { m a x } }$ are medians over 50 probed states per group. Perturbation separations are medians over completed fixed-step probed states. A range in a completed entry gives the Early-Late group medians.
<table><tr><td>Quantity</td><td>Attention-A</td><td>MLP</td><td>Attention-B</td></tr><tr><td>Step-16 / step-512 exact solve</td><td>0.592 / 0.875</td><td>0.744 / 0.919</td><td>0.521 / 0.726</td></tr><tr><td>Step-16 unresolved with any later exact solve</td><td> $2 8 3 / 4 0 8 = 0 . 6 9 4$ </td><td> $1 7 5 / 2 5 6 = 0 . 6 8 4$ </td><td> $2 0 6 / 4 7 9 = 0 . 4 3 0$ </td></tr><tr><td>Solutions never lost after first solve</td><td> $8 5 9 / 8 7 5 = 0 . 9 8 2$ </td><td> $9 1 9 / 9 1 9 = 1 . 0 0 0$ </td><td> $7 1 9 / 7 2 7 = 0 . 9 8 9$ </td></tr><tr><td> $\gamma ^ { \mathrm { n a t } } .$  , completed / PERSISTENT</td><td>0.25-0.31 / 2.09</td><td>0.47 / 2.82</td><td>0.20-0.21 / 2.16</td></tr><tr><td> $\sigma _ { \mathrm { m a x } } .$  completed / PERSISTENT</td><td>26.8-26.9 / 316.9</td><td>6.4-6.7 / 312.4</td><td>17.6-19.0 / 283.9</td></tr><tr><td> $\mathrm { P r } [ \gamma ^ { \mathrm { n a t } } < 1 < \sigma _ { \mathrm { m a x } } ]$ </td><td>0.621</td><td>0.604</td><td>0.624</td></tr><tr><td>v1 perturbation,  $k { = } 1 / k { = } 1 6$ </td><td>24.4 / 0.10</td><td>6.6 / 1.91</td><td>21.1 / 0.096</td></tr><tr><td>Natural perturbation,  $k { = } 1 / k { = } 1 6$ </td><td>0.38 / 0.012</td><td> $0 . 5 0 / 0 . 1 8$ </td><td> $0 . 2 3 / 0 . 0 0 7 0$ </td></tr><tr><td>Transport T1</td><td> $4 . 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table 5 summarizes the shared findings and differences between checkpoints. Extended recurrence resolves many nominal-budget failures. In Attention-A and the MLP, LATE trajectories settle after solving and approach the low-motion regime of EARLY trajectories. Completed groups typically have the natural-direction gains below one even though expanding directions remain in the local Jacobian.

The checkpoints differ in solve rates, the size of the completion transition, and the response to perturbations. In particular, the MLP has greater transport of the leading expanding direction and less consistent recovery of perturbations along that direction over 16 steps.

## H ATTENTION-B TRAINING SNAPSHOTS

This section reports the separate nominal-horizon sweep over all 25 released Attention-B training snapshots (PreetiMLresearcher, 2026). The full extended analysis of the terminal checkpoint is reported in Appendix G.7, Table 5. The snapshots span training steps 2,604 to 65,100. Most are spaced by 2,604 training steps, except that the third snapshot is at step 6,510.

Nominal-horizon exact-solve performance improves overall across training, reaching 53.1% at the final checkpoint, with several temporary declines (Figure 12 a). Conflict severity among unsolved puzzles initially decreases and later rises slightly (Figure 12 b), although the set of unsolved puzzles changes between snapshots.

(c) Constraint-aware completion  
![](images/21a601a1ad5b305770421e367c399aa7cc831676d117e7c42f8faae453945476.jpg)

![](images/9b13f3b846d8131fdad7cbe67fad9ba6cb3d314ac002379dd4fdfd2d62a28f07.jpg)  
Training step (×10,000)

![](images/aefe5b0bd0e565bd342ffd517122d71e95abbdc8551814421161aba0937bff4e.jpg)  
Model candidates retained per blank (k)  
Figure 12: Attention-B training and decoder diagnostics. (a) Exact-solve and blank-cell accuracy across 25 correlated training snapshots. (b) Conflict severity. (c) Exact completion under model-guided Sudoku constraints; the dotted line is native argmax and k = 9 is a symbolic-only control.

## H.1 COMPARISON BETWEEN CHECKPOINTS AT THE NOMINAL HORIZON

The 25-snapshot sweep provides a training-time view of the outcome-associated trends seen in Attention-A. The terminal Attention-B checkpoint also provides an independent full-run replication of the same attention architecture, summarized in Appendix G.7.

The three Sudoku checkpoints are as follows. Attention-A (step 21700) is the checkpoint studied in the main text (Gao, 2025a). Attention-B (step 65100) comes from a separate public training run of the same architecture (PreetiMLresearcher, 2026). MLP (step 16275) replaces attention-based token mixing with an MLP along the sequence axis and uses no positional encoding (Gao, 2025c).

All three models use hidden width 512, three H-cycles, six L-cycles, two layers in the repeated L-level module, and bfloat16 forward computation. The two attention models use eight attention heads and rotary positional encodings (Su et al., 2021).

Table 6: Nominal 16-step evaluation on 1,000 hard Sudoku puzzles. Final $Z _ { L }$ updates are reported as medians within the solved and failed groups. “Factor” is the ratio of the failed-group median to the solved-group median.
<table><tr><td>Checkpoint</td><td>Exact solve</td><td>Final  $Z _ { L }$  update, solved / failed</td><td>Factor</td></tr><tr><td>Attention-A</td><td>0.592</td><td>0.025 / 0.761</td><td>~30</td></tr><tr><td>Attention-B</td><td>0.531</td><td>0.069 / 0.708</td><td>10.2</td></tr><tr><td>MLP</td><td>0.744</td><td>0.018 / 0.536</td><td>~30</td></tr></table>

The MLP achieves the highest nominal solve rate followed by Attention-A and Attention-B. All three checkpoints have larger median final $Z _ { L }$ updates on failed puzzles than on solved puzzles (Table 6).

## I CONSTRAINT-AWARE DECODING

The TRM decoder fills each blank with the highest-scoring digit independently. Related neural Sudoku methods incorporate constraint structure through recurrent message passing (Palm et al., 2018) or differentiable satisfiability layers (Wang et al., 2019). To test whether useful alternatives remain in the logits, we also evaluate a constrained decoder on Attention-B. For each blank, it keeps the top-k predicted digits and uses backtracking to search only assignments that satisfy the Sudoku row, column, box, and clue constraints. The search is capped at 200,000 nodes per puzzle (Table 7).

All 469 failures of the native decoder are complete boards that violate Sudoku constraints. Allowing two or three candidate digits per blank changes the solve rate slightly. With four candidates, the constrained decoder finds 754 exact solutions, recovering 223 cases missed by native argmax decoding (Table 7).

These recoveries show that useful alternatives remain in the logits. A correct completion can be found within the top-four candidate sets when Sudoku constraints are enforced by backtracking. The choice of k = 4 was made using the same 1,000 evaluation puzzles.

Table 7: Constraint decoding for Attention-B on 1,000 puzzles. “Supported” means that the correct digit for every blank appears in that blank’s top-k predictions. The $k = 9$ setting removes model-based pruning and serves only as a symbolic-solver control.
<table><tr><td>k</td><td>Supported</td><td>Exact</td><td>Median nodes</td><td>95th pct. nodes</td></tr><tr><td>1</td><td>0.531</td><td>0.531</td><td>56</td><td>59</td></tr><tr><td>2</td><td>0.533</td><td>0.533</td><td>56</td><td>59</td></tr><tr><td>3</td><td>0.537</td><td>0.537</td><td>59</td><td>659</td></tr><tr><td>4</td><td>0.754</td><td>0.754</td><td>59</td><td>9,304</td></tr><tr><td>9</td><td>1.000</td><td>1.000</td><td>59</td><td>16,049</td></tr></table>

## J EASY SUDOKU

The Easy Sudoku control uses the 100 puzzles in the nikoli 100 set of Sudoku-Bench (Seely et al., 2025) (https://huggingface.co/datasets/SakanaAI/sudoku-bench-nikoli). These puzzles contain 46-58 blanks compared with 55-60 in the hard split.

Both Attention-A and the MLP solve all 100 puzzles within the nominal 16 steps. At the first step, Attention-A solves 21 puzzles and the MLP solves 66. The final latent updates are small in both models, although their magnitudes differ (Table 8). These updates are comparable to those of Hard Sudoku puzzles after they are solved, for both EARLY and LATE solvers.

Table 8: Easy Sudoku control, n = 100, at the nominal 16-step horizon.
<table><tr><td>Quantity</td><td>Attention-A</td><td>MLP</td></tr><tr><td>Exact-solve rate</td><td>1.000</td><td>1.000</td></tr><tr><td>Solved at step 1</td><td>21</td><td>66</td></tr><tr><td>Steps to 99% of final mean blank-cell accuracy</td><td>6</td><td>2</td></tr><tr><td> $Z _ { L }$  final update</td><td>0.017</td><td>0.024</td></tr><tr><td> $Z _ { H }$  final update</td><td>0.008</td><td>0.034</td></tr></table>

## K MAZE-HARD

Maze-Hard provides a cross-task control using 1,000 30 × 30 mazes from the benchmark of Wang et al. (2025) (https://huggingface.co/datasets/sapientinc/ maze-30x30-hard-1k). Each cell holds one of six symbols. We evaluate an attention checkpoint (step 9765; Gao, 2025b) with three H-cycles and four L-cycles.

![](images/1d2bf011e873616f77b08821b7a48b0cf9dd455562b0b59d34eeccaa6dacfd76.jpg)

![](images/3b1bfaec8b7c73c47fa1519059d8e01d79553ffd791893877f90a3aa607e21c6.jpg)

(c)  
![](images/32bad1d40bc1e6dcb2b4b0a4e328499463a8892d776cc5c58aef8955e7737e0b.jpg)  
Figure 13: Maze-Hard outcomes and latent settling. (a) Cumulative exact solves from bfloat16 run, with Wilson bands. Figures (b,c) Median relative latent updates in float64. Shaded bands show interquartile ranges.

Most solutions that match the ground truth appear early, with 800 mazes matching the ground truth at step 3 (790 at step 2). Over the entire 16-step evaluation, 806 mazes match the ground truth at least once, but only 788 do so at the final step because 18 earlier solutions are subsequently lost. This 16-step evaluation does not determine whether the unresolved cases would solve if recurrence were extended beyond step 16.

Exact match with the ground truth also understates path validity. Of the 212 final non-matches, 191 are valid start-to-goal paths, 75 of them as short as the ground truth, and only 21 are structurally invalid.

Table 9: Maze-Hard results, n = 1,000, at the nominal 16-step horizon. Intervals are Wilson 95%.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Exact-match rate</td><td>0.788 [0.762, 0.812]</td></tr><tr><td>Valid-path rate</td><td>0.979 [0.968, 0.986]</td></tr><tr><td>Valid but not exact</td><td>191</td></tr><tr><td>equal path length to reference</td><td>75</td></tr><tr><td>longer (mean +2.1 cells)</td><td>116</td></tr><tr><td>Structurally invalid paths</td><td>21</td></tr><tr><td>Exactly correct earlier, lost by step 16</td><td>18</td></tr></table>

## LATENT SETTLING

We categorize the puzzle in three groups: MATCHED, UNMATCHED, and UNSOLVED. MATCHED means the model solved the puzzle and it matches the ground truth $( n = 7 8 8 )$ . UNMATCHED means the solution provided by the model is a valid path but doesn’t match the ground truth $( n = 1 9 1 )$ ). UNSOLVED means the model failed to provide the structurally valid solution to the puzzle $( n = 2 1 )$ ).

At this scale, the measurements cannot reliably resolve differences in settling. We therefore replay the recurrence from the checkpoint’s initial state in float64. We keep the groups and solve times fixed, so that the control does not also change the outcome definition.

Accuracy: Acc. is all-cell accuracy: the fraction of the 900 maze cells whose predicted label matches the reference. It includes walls, open cells, endpoints, and path cells. Many unchanged background cells can make this high even when the route is wrong. Path F1 measures overlap between predicted and reference path-marked cells:

$$
\operatorname { P a t h } \operatorname { F 1 } = \frac { 2 | P \cap R | } { | P | + | R | } ,\tag{13}
$$

where P and R are the predicted and reference path-cell sets. Start and goal cells have separate labels and are excluded from these sets. The table (10) reports the mean per-puzzle score within each group at step 16, using the BF16 predictions. Importantly, Path F1 is not path validity. A valid alternative route can score below 1 because it differs from the reference. An invalid route can score highly because most of its cells overlap the reference. That explains why the Unsolved group can have higher accuracy and Path F1 than the Unmatched group. The change in latents $Z _ { L }$ and $Z _ { H }$ for these categories are also reported in table (10) and plotted in fig. (13 b, c).

Local stability: Small latent updates alone do not establish local stability. We investigate whether a small displacement along the trajectory is contracted by the next recurrent step. Let $s _ { t } = ( Z _ { H , t } , Z _ { L , t } )$ be the joint latent state, with the puzzle held fixed

$$
J _ { t } = D F _ { \theta } ( s _ { t } ) , \qquad d _ { t } ^ { \mathrm { o u t } } = \frac { s _ { t + 1 } - s _ { t } } { \| s _ { t + 1 } - s _ { t } \| _ { 2 } } , \qquad \gamma _ { t } ^ { \mathrm { n a t } } = \| J _ { t } d _ { t } ^ { \mathrm { o u t } } \| _ { 2 } .\tag{14}
$$

A gain $\gamma _ { t } ^ { \mathrm { n a t } } < 1$ indicates local contraction along that direction. We compare it with $\sigma _ { \operatorname* { m a x } } ( J _ { t } )$ , the largest gain over all directions. The latter is estimated by power iteration. For contractions, see table 11 and fig. 14 a, b for more information.

Table 10: Maze outcomes and latent motion at step 16. Groups, mean all-cell accuracy, and mean path F1 come from the bfloat16 run. Median relative latent updates $\delta _ { L }$ and $\delta _ { H }$ come from the float64 evaluation.
<table><tr><td>Group</td><td>n</td><td> $\operatorname { A c c } .$ </td><td>Path F1</td><td> $\delta _ { L }$ </td><td> $\delta _ { H }$ </td></tr><tr><td>MATCHED</td><td>788</td><td>1.000</td><td>1.000</td><td> $1 . 4 \times 1 0 ^ { - 6 }$ </td><td> $6 . 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>UNMATCHED</td><td>191</td><td>0.968</td><td>0.871</td><td> $4 . 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>UNSOLVED</td><td>21</td><td>0.975</td><td>0.901</td><td> $1 . 9 \times 1 0 ^ { - 3 }$ </td><td> $9 . 0 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table 11: Local stability by input-state step. $\gamma ^ { \mathrm { n a t } }$ measures gain along the trajectory. $\sigma _ { \mathrm { m a x } }$ estimates the largest gain over all directions. Entries are medians with IQR ranges. $n _ { \gamma }$ counts valid directions. Each measured $\sigma _ { \mathrm { m a x } }$ entry uses all 50, 50, or 21 states.
<table><tr><td>Step</td><td>Group</td><td> $n _ { \gamma }$ </td><td> $\gamma ^ { \mathrm { n a t } }$ </td><td> $\mathrm { P r } [ \gamma ^ { \mathrm { n a t } } < 1 ]$ </td><td> $\sigma _ { \mathrm { m a x } }$ </td></tr><tr><td rowspan="3">2</td><td>MATCHED</td><td>50</td><td>0.15 [0.14,0.21]</td><td>1.00</td><td> $1 2 . 5 \ [ 9 . 2 , 1 9 . 9 ]$ </td></tr><tr><td>UNMATCHED</td><td>50</td><td>0.18 [0.14,0.25]</td><td>0.96</td><td>21.5 [12.7,28.9]</td></tr><tr><td>UNSOLVED</td><td>21</td><td>0.51 [0.32, 3.12]</td><td>0.62</td><td>66.3 [38.5, 223.0]</td></tr><tr><td rowspan="3">7</td><td>MATCHED</td><td>50</td><td>0.53 [0.30, 0.76]</td><td>0.92</td><td>11.3 [8.8, 17.7]</td></tr><tr><td>UNMATCHED</td><td>50</td><td>0.54 [0.36, 0.77]</td><td>0.82</td><td>16.1 [11.7,25.1]</td></tr><tr><td>UNSOLVED</td><td>21</td><td>0.79 [0.51, 1.18]</td><td>0.67</td><td>74.3 [27.0, 114.9]</td></tr><tr><td rowspan="3">15</td><td>MATCHED</td><td>49</td><td>0.61 [0.38, 0.77]</td><td>0.96</td><td>11.7 [8.9, 17.5]</td></tr><tr><td>UNMATCHED</td><td>47</td><td>0.59 [0.43, 0.76]</td><td>0.91</td><td>15.5 [12.1,23.1]</td></tr><tr><td>UNSOLVED</td><td>21</td><td>0.75 [0.48, 1.27]</td><td>0.71</td><td>68.8 [27.0,99.7]</td></tr></table>

![](images/3be6a272712adfdd0cf0c6566c27a52602386bc356ca6a5da1bb406a0b0a8a90.jpg)

(b)  
![](images/d445383bdb5fc4b7ac2583f7e1826315a69514c840195bf96641ea9b8d61b5fe.jpg)  
Figure 14: Directional contraction and worst-case expansion. (a) Fraction of valid directions with $\gamma _ { t } ^ { \mathrm { n a t } } < 1$ with Wilson bands. (b) States with both a valid directional gain and a singular-value estimate, at $t = 2 , 7 , 1 5$ The inequalities $\gamma _ { t } ^ { \mathrm { n a t } } < 1 < \sigma _ { \operatorname* { m a x } } ( J _ { t } )$ hold together at 317/359 states (88.3%).

A larger gain can accompany smaller motion. For MATCHED puzzles, median gain rises from 0.15 at $t = 2$ to 0.61 at $t = 1 5$ (Table 11). Over the same steps, the full-group median $\delta _ { L }$ falls from $4 . 4 \times 1 0 ^ { - 1 } \mathrm { t o } 2 . 5 \times 1 0 ^ { - 6 }$ . There is no contradiction. Gain is a local amplification factor, whereas the update measures how far the state moves. These two also uses different populations. The update is a blockwise (either $Z _ { L }$ or $Z _ { H } )$ quantity over all 788 MATCHED puzzles. The gain uses selected joint-state probes.

Stability is directional, not global. The contraction along the trajectory coexists with expansion in another direction at $3 1 7 / 3 5 9$ states with both measurements (88.3%; Figure 14). The Wilson 95% interval is 84.6%–91.2%.