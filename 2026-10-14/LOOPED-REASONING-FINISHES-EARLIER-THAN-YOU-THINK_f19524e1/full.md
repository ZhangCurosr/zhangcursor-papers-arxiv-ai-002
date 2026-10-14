# LOOPED REASONING FINISHES EARLIER THAN YOU THINK

Sumin Park<sup>1,2,3</sup>, T. Konstantin Rusch<sup>1,2,3,4∗</sup>

<sup>1</sup>ELLIS Institute Tubingen, ¨ <sup>2</sup>Max Planck Institute for Intelligent Systems,

<sup>3</sup>Tubingen AI Center, ¨ <sup>4</sup>Liquid AI

## ABSTRACT

Looped reasoning with dynamic depth can be viewed as relaxation toward solution attractors in a learned dynamical landscape. In practice, however, latent trajectories often oscillate, cycle, or drift rather than cleanly converge to stable fixed points. How these non-stationary dynamics contribute to reasoning and constrain convergence remains unclear. We show that slowly decaying oscillatory modes can impose their convergence rate on coupled latent dynamics, creating a bottleneck. Yet damping these modes indiscriminately can disrupt the reasoning computation they support. This tension motivates an architectural approach: introducing rotational degrees of freedom along which symmetry-generated rotational motion leaves the reasoning-relevant quotient dynamics unchanged by construction. We propose QRR (Quotient Recurrent Reasoner), a gauge-equivariant recurrent architecture with an exact per-token phase-rotation symmetry. Motion along these symmetry directions leaves the quotient representation unchanged and is dynamically decoupled from reasoning-relevant dynamics. Empirically, QRR achieves stable and faster convergence to solution roots in quotient space, without forming a dominant slow oscillatory bottleneck, while matching or surpassing recent reasoning models on Sudoku, Maze, and Mini-ARC. Code is available at here.

## 1 INTRODUCTION

Looped reasoners trade architectural depth for recurrent computation (Dehghani et al., 2019; Saunshi et al., 2025; Fan et al., 2025). A shared neural block is applied repeatedly to refine an internal state, allowing additional computation without increasing parameter count (Bansal et al., 2022; Geiping et al., 2025; Wang et al., 2025; Jolicoeur-Martineau, 2025). Recent works make the dynamical interpretation explicit, viewing reasoning as the evolution of a latent state converging toward solution aligned equilibria (Blayney et al., 2026; Kim, 2026). Fixed-Point Reasoning Model (FPRM) uses convergence of the recurrent state to a fixed point as an intrinsic stopping criterion, Attractor Models define the prediction itself as the equilibrium of a learned refinement process, and Equilibrium Reasoners (EqR) explicitly learns task-conditioned attractors whose stable fixed points correspond to valid solutions, using multiple initializations to explore their attraction basins (Movahedi et al., 2026; Fein-Ashley & Rashidinejad, 2026; Huang et al., 2026). In this view, reasoning is not just a sequence of refinements but a recurrent trajectory through the learned dynamical landscape.

However, viewing recurrent computation as convergence to a stable attractor does not capture the full range of dynamics it can exhibit. Trajectories can instead exhibit orbits, drifts, cyclic limiting behavior, and other structured forms of non-stationarity, rather than simply relaxing toward a single fixed point (Geiping et al., 2025; Pappone et al., 2025; Blayney et al., 2026; Kim, 2026; Yang et al., 2026). These studies reveal a wide range of latent geometries in recurrent models, yet how these different forms of oscillatory motion affect the underlying latent computation remains unexplored.

The key distinction is therefore not the geometry of the motion itself, but whether and how it influences the recurrent computation. Around a solution root z<sup>⋆</sup>, where the recurrence admits a fixed linearization with the Jacobian $J = { \cal D } f ( z ^ { \star } )$ , we establish a local coupling mechanism by separating an oscillatory spectral coordinate from the complementary latent dynamics and quantifying how the former drives the latter. When the oscillatory mode decays more slowly, this coupling transfers its rate to the complementary dynamics and delays settlement. We then test this mechanism causally in the original nonlinear recurrences of FPRM and HRM (Wang et al., 2025) in Section 3.2. Across distinct oscillatory geometries, an initial perturbation of the identified mode produces latent responses and settling-time shifts that closely match the predictions of the local coupling model. This agreement directly links the convergence delay near solution roots to the coupled slow oscillatory mode.

Oscillatory appearance alone, however, does not reveal which motion can be safely removed. In FPRM, suppressing the alternating mode before semantic stabilization degrades task performance, showing that slow oscillatory motion can still support reasoning (Appendix A.4.4). Thus, rather than identifying and damping particular oscillations, we instead redesign what counts as convergence by introducing a controlled representational redundancy: motion along rotational symmetry directions is irrelevant to the underlying quotient state by construction. We propose QRR, the Quotient Recurrent Reasoner, which imposes an exact per-token rotational symmetry on the recurrent state, inducing a quotient representation invariant to this motion. Gauge-equivariant updates pre-

![](images/7a538c26299856c9fde56b5165b497adec845df8efddccfb1d02a639409810da.jpg)  
Figure 1: Oscillatory geometry and quotient convergence. FPRM and HRM exhibit distinct alternating and spiral dynamics near solution roots. In QRR, raw motion persists while quotient trajectory contracts.

serve this equivalence, allowing symmetry-generated motion to persist in the raw trajectory without driving or limiting the convergence of quotient dynamics.

As in Figure 1, FPRM and HRM exhibit distinct oscillatory geometries whereas QRR retains raw phase motion while its quotient trajectory converges. Oscillatory modes outside the gauge subspace remain free to participate in reasoning. Empirically, QRR also retains oscillatory modes in quotient space but does not develop the isolated dominant slow mode as in FPRM and HRM (Section 4.4).

## 2 BACKGROUND

## 2.1 LOOPED REASONING MODELS

Looped reasoning models increase effective computational depth by repeatedly applying shared neural blocks to a latent state, allowing additional test-time computation without increasing parameter count. Recent approaches can be broadly grouped by how they organize this recurrent computation. General recurrent-depth models iterate a shared Transformer block and scale reasoning by increasing the number of latent recurrences at inference (Geiping et al., 2025). Puzzle-oriented recursive reasoners instead structure the recurrence explicitly. HRM couples high- and low-level recurrent modules operating at different frequencies (Wang et al., 2025), whereas Tiny Reasoning Model (TRM) simplifies this hierarchy to a single small network that repeatedly refines a latent reasoning state and the current answer (Jolicoeur-Martineau, 2025). A complementary line organizes recurrent computation around equilibria or attractors: FPRM uses convergence to a fixed point as an adaptive halting criterion (Movahedi et al., 2026), EqR learns task-conditioned attractors whose stable equilibria correspond to solutions (Huang et al., 2026), and Attractor Models explicitly solve a learned fixed-point refinement with implicit differentiation (Fein-Ashley & Rashidinejad, 2026).

## 2.2 GAUGE SYMMETRY AND QUOTIENT DYNAMICS

Gauge symmetry formalizes a local representational redundancy where changing the local reference frame changes the coordinates used to represent a state, but not the underlying object itself (Cohen et al., 2019a; Bronstein et al., 2021; Gerken et al., 2023). If a map $T$ is equivariant under the action of a group $G ,$ , gauge-related states $z \sim g \cdot z$ evolve consistently and can be identified as a single equivalence class $[ \breve { z } ]$ in the quotient space $\mathcal { Z } / G$ (Kondor & Trivedi, 2018; Cohen et al., 2019b). The quotient collapses symmetry-related representations into a single state, so motion along the symmetry orbit is ignored while all remaining state changes are preserved. This distinction also changes the notion of convergence. An equivariant system may continue moving along a symmetry orbit without changing its quotient state. Such motion is captured by a relative equilibrium, $T ( z ^ { \star } ) = g ^ { \star } \cdot z ^ { \star }$ which is non-stationary in the original coordinates but becomes a fixed point after quotienting, $[ T ( z ^ { \star } ) ] = [ z ^ { \star } ]$ (Krupa, 1990; Marsden & Ratiu, 1999). In neural architectures, gauge equivariance provides a principled way to implement this invariance to local reference frames (Weiler et al., 2021; Wiersma et al., 2020). Gauge-equivariant convolutions extend conventional equivariance from global transformations to local gauge choices on manifolds (Cohen et al., 2019a), and Gauge Equivariant Mesh CNNs combine anisotropic convolution with parallel transport to communicate features in different local frames (Haan et al., 2021). The same principle has also been extended to attention, where the Gauge Equivariant Transformer constructs gauge-equivariant self-attention by comparing features across local coordinate systems after transporting them to a common frame (He et al., 2021).

## 2.3 OSCILLATORY AND NON-STATIONARY DYNAMICS IN RECURRENT REASONING

Recent depth-recurrent language models exhibit structured latent trajectories that are substantially richer than monotone convergence to a fixed point, including multi-dimensional orbits and approximately one-dimensional drifts, particularly on tokens requiring extended computation (Geiping et al., 2025; Pappone et al., 2025; Kim, 2026). Other mechanistic studies find cyclic fixed-point trajectories, where recurrent layers repeatedly trace a stable cycle in latent space, as well as persistent non-stationarity even after successive state differences become small (Huang et al., 2026; Blayney et al., 2026). Oscillatory behavior also appears in recurrent reasoners based on solution equilibria. FPRM, for example, reports oscillation around solution fixed points and introduces damping to stabilize these trajectories (Movahedi et al., 2026). More broadly, looped Transformers with the same architecture, objective, and task performance can settle into different dynamical phases, including bounded non-stationary motion generated by complex-conjugate Jacobian modes (Kim, 2026).

What remains unclear is the dynamical role of this motion. An observed orbit or alternating trajectory may support recurrent computation, or instead reflect a slowly decaying mode that unnecessarily prolongs latent settlement after the remaining dynamics already converged.

## 3 RATE-LIMITING SLOW OSCILLATORY MODES IN RECURRENT REASONERS

Motivated by oscillatory structures observed in recurrent reasoners, we study whether an identified oscillatory mode can become rate-limiting for latent convergence. Around solution roots, the nonlinear recurrence admits a stationary first-order linearization, allowing us to isolate an oscillatory spectral component from its complementary latent dynamics.

We use a Sylvester decomposition as an analytic attribution tool to separate the latent state dynamics into its intrinsic evolution and the response induced by coupling with the identified spectral coordinate. This provides a direct way to quantify how the identified coordinate influences the remaining latent dynamics and their convergence behavior. The decomposition itself is general to any selected mode, particular oscillatory structure enters through local Jacobian block identified from the trained model.

## 3.1 LOCAL SPECTRAL DECOMPOSITION AND RATE DOMINANCE

We analyze how an oscillatory component of the local dynamics affects the remaining latent state near a solution root. Let $z _ { t + 1 } = f ( z _ { t } )$ denote the trained nonlinear recurrence, and let $z ^ { \star }$ be a solution root satisfying $f ( z ^ { \star } ) = z ^ { \star }$ . For a nearby trajectory state, write the root-centered displacement $\delta _ { t } = z _ { t } - z ^ { \star }$ . Since $z ^ { \star }$ is stationary, the same local Jacobian governs subsequent steps as

$$
\delta _ { t + 1 } = J \delta _ { t } + O ( \| \delta _ { t } \| ^ { 2 } ) , \qquad J = D f ( z ^ { \star } ) .\tag{1}
$$

This stationary linearization allows us to identify and track a fixed spectral component over recurrence. Consider an oscillatory spectral component of $^ { J , }$ and let $U \in \mathbb { R } ^ { \hat { D } \times m }$ be an orthonormal basis for the corresponding left-invariant subspace of $^ { J , }$ and let S denote the restriction of J to this subspace. Equivalently, $J ^ { \top } \bar { U } = U S ^ { \top }$ , with $S \dot { = } U ^ { \top } J U$ . Hence, defining a coordinate $u _ { t } : = U ^ { \top } \delta _ { t }$ , the local dynamics satisfy

$$
u _ { t + 1 } = U ^ { \top } J \delta _ { t } + O ( \Vert \delta _ { t } \Vert ^ { 2 } ) = S u _ { t } + O ( \Vert \delta _ { t } \Vert ^ { 2 } ) .\tag{2}
$$

Thus $u _ { t }$ is an autonomous spectral coordinate to first order, its next state update depends only on itself through the reduced Jacobian block S. We represent the remaining displacement by the orthogonal residual $e _ { t } ,$

$$
\begin{array} { r } { u _ { t } = U ^ { \top } \delta _ { t } , \qquad e _ { t } = Q \delta _ { t } , \qquad Q = I - U U ^ { \top } , } \end{array}\tag{3}
$$

so that $\delta _ { t } = U u _ { t } + e _ { t }$ and $U ^ { \top } e _ { t } = 0$ . Projecting the linearized dynamics onto these two coordinates, $u _ { t }$ and $e _ { t } ,$ gives (see full derivation in Appendix A.2)

$$
u _ { t + 1 } = S u _ { t } , \qquad e _ { t + 1 } = A e _ { t } + B u _ { t } , \qquad A : = Q J Q , \quad B : = Q J U .\tag{4}
$$

Here A governs the intrinsic evolution of the complementary state, whereas B measures how the isolated spectral coordinate drives that complement. The observed $e _ { t }$ therefore mixes two effects: its intrinsic transient under A and the response transmitted from $u _ { t }$ coupled by B. We separate these two distinct contributions to evolve $e _ { t }$ using the following Sylvester decomposition.

Lemma 3.1 (Exact Sylvester decomposition of a coupled spectral coordinate). Consider the linear dynamics in Eq. 4, with $A \in \mathbb { R } ^ { d \times d } , \overline { { S } } \in \mathbb { R } ^ { m \times m }$ , and $B \in \mathbb { R } ^ { d \times m }$ , and assume $\sigma ( A ) \cap \sigma ( S ) = \emptyset .$ Then there exists a unique matrix $P \in \mathbb { R } ^ { d \times m }$ satisfying $P S - A P = B$ . The complementary state admits the exact decomposition

$$
e _ { t } = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) + P S ^ { t } u _ { 0 } .\tag{5}
$$

Lemma 3.1 separates the complementary state $e _ { t }$ into two subsystems. The first term, $A ^ { t } ( e _ { 0 } { - } P u _ { 0 } )$ , is its intrinsic transient, whereas $P S ^ { t } u _ { 0 }$ is the response transmitted from the coupled spectral coordinate $u _ { t }$ . The lemma is not specific to oscillatory dynamics, it provides a general framework for attributing the effect of one autonomous spectral component on the rest of a coupled linear system.

Slow oscillatory specialization. We now specialize this decomposition to the reduced block $S = U ^ { \top } J U$ observed in the actual recurrent reasoners, where the isolated mode is oscillatory and decays more slowly than the complementary dynamics. When the measured S has a complexconjugate pair with modulus r and frequency $\omega ,$ , its real two-dimensional dynamics admit

$$
\sigma ( S ) = \{ r e ^ { \pm i \omega } \} , \qquad S = M ( r Q _ { \omega } ) M ^ { - 1 } , \qquad 0 < \omega < \pi ,\tag{6}
$$

where $Q _ { \omega }$ is the planar rotation through ω and M is real and invertible. When $\rho ( A ) < r$ , the intrinsic transient in Eq. 5 decays faster than the coupled response $P S ^ { t } u _ { 0 }$ . For a trajectory $e = ( e _ { t } ) _ { t \geq 0 }$ , define

$$
\tau _ { \delta } ( e ) = \operatorname* { i n f } \{ t \in \mathbb { N } _ { 0 } : \| e _ { k } \| _ { 2 } \leq \delta \mathrm { ~ f o r ~ a l l ~ } k \geq t \} , \qquad \mathcal { E } ( e ) = \operatorname* { i n f } _ { c \in \mathbb { R } ^ { d } } \operatorname* { l i m } _ { t \to \infty } \| e _ { t } - c \| _ { 2 } .\tag{7}
$$

Theorem 3.2 (Rate inheritance from a slow oscillatory mode). Under the setting of Lemma 3.1, suppose S satisfies Eq. 6, with $\rho ( A ) < r \leq 1 , B \neq 0 ,$ and $u _ { 0 } \neq 0 . \mathrm { ~ } H r < 1$ , then $e _ { t } \to 0 ;$

$$
\operatorname* { l i m } _ { t \to \infty } \| e _ { t } \| _ { 2 } ^ { 1 / t } = r > \rho ( A ) ,\tag{8}
$$

and

$$
\tau _ { \delta } ( e ) = \frac { \log ( 1 / \delta ) } { - \log r } + O ( 1 ) , \qquad \delta \downarrow 0 .\tag{9}
$$

$I f r = 1$ , then $e _ { t }$ remains bounded but does not converge to any constant, with

$$
\mathcal { E } ( e ) \geq \frac { \| u _ { 0 } \| _ { 2 } \| B \| _ { F } } { \sqrt { 2 } \kappa _ { 2 } ( M ) ( 1 + \| A \| _ { 2 } ) } > 0 .\tag{10}
$$

Theorem 3.2 shows that a slower coupled mode transfers its rate to the complementary dynamics, delaying settlement when $r \digamma < 1$ and preventing stationary convergence when $r = 1$ . It covers general two-dimensional spiral modes, while $\bar { M } = I$ recovers a scaled rotation and $S = - r$ gives the alternating flip case of Corollary A.1. All proofs are provided in Appendix A.1.

## 3.2 CAUSAL VERIFICATION IN NONLINEAR RECURRENT DYNAMICS

We now use this framework to ask whether the slow oscillatory modes identified in trained recurrent reasoners actually exert the predicted rate-limiting effect on their nonlinear trajectories. For each solution root, we identify a leading oscillatory eigenspace of the local Jacobian, construct its leftinvariant basis $U ,$ , and obtain $S , A ,$ , and B matching the decomposition in Section 3.1. We focus on cases satisfying the rate-dominance condition $\rho ( A ) < r \leq 1$ (Appendix A.4.3).

To test the contribution of this mode in the original nonlinear dynamics, we take a state from the native trajectory once it has entered the local linear regime around the solution root $z ^ { \star }$ . Writing

$$
\begin{array} { r } { \delta _ { 0 } : = z _ { t _ { 0 } } - z ^ { \star } = e _ { 0 } + U u _ { 0 } , \qquad e _ { 0 } = Q \delta _ { 0 } , \qquad u _ { 0 } = U ^ { \top } \delta _ { 0 } , } \end{array}\tag{11}
$$

separates the displacement into the identified oscillatory coordinate $u _ { 0 }$ and the remaining local state $e _ { 0 }$ . We then vary only the amplitude of $u _ { 0 }$ by adjusting $\alpha .$

$$
z _ { t _ { 0 } } ^ { ( \alpha ) } = z ^ { \star } + e _ { 0 } + \alpha U u _ { 0 } , \qquad \alpha \in \left\{ 1 , \frac { 1 } { 2 } , 0 \right\} ,\tag{12}
$$

while keeping $e _ { 0 }$ exactly fixed. Here $\alpha = 1$ is the native state, $\alpha = 0 . 5$ attenuates the oscillatory coordinate, and $\alpha = 0$ removes it at the intervention point. This is the only modification and all trajectories subsequently evolve using the original trained nonlinear recurrence $f .$ . Because all arms share the same complementary state $e _ { 0 } ,$ the linearized rollout gives $\widehat { e } _ { k } ^ { ( \alpha ) } = Q \delta _ { k } = Q J ^ { k } ( e _ { 0 } + \alpha U u _ { 0 } )$ (Eq. 3), so subtracting the $\alpha = 0$ arm isolates the response to the oscillatory coordinate. Using the decomposition in Eq. 5, this prediction can be further written equivalently as

$$
\widehat { \Delta e } _ { k } ^ { ( \alpha ) } = \widehat { e } _ { k } ^ { ( \alpha ) } - \widehat { e } _ { k } ^ { ( 0 ) } = \alpha Q J ^ { k } U u _ { 0 } = \alpha \left( P S ^ { k } - A ^ { k } P \right) u _ { 0 } .\tag{13}
$$

This response prediction is determined entirely by the local Jacobian and the native displacement at $t _ { 0 } .$ , with no quantities fit to the subsequent nonlinear rollout.

Alternating oscillatory mode in FPRM. We apply the intervention protocol to the rate-dominant oscillatory mode identified in FPRM. As illustrated in Figure 1, this mode appears as an approximately one-dimensional alternating flip, with $S = - r$ and $\bar { \rho ( A ) } < r \le 1$ . The corresponding response prediction is (derivations in Appendix A.1)

$$
\widehat { \Delta e } _ { k } ^ { ( \alpha ) } = \alpha \left[ P ( - r ) ^ { k } - A ^ { k } P \right] u _ { 0 } = \alpha P ( - r ) ^ { k } u _ { 0 } + o ( r ^ { k } ) .\tag{14}
$$

Thus, changing α scales the leading response while leaving its asymptotic decay rate unchanged: $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ halves its leading amplitude, whereas $\alpha = 0$ removes the $r ^ { \hat { k } }$ contribution. We compare this closed-form local prediction directly with the response measured from the original nonlinear FPRM recurrence. For $\alpha = 0$ , the linearized counterfactual reduces to $\hat { e } _ { k } ^ { ( 0 ) } = A ^ { k } e _ { 0 }$ , allowing us to additionally test whether removing the oscillatory coordinate accelerates root-relative settlement.

![](images/214fe06376fce82001d487f445efae30f262a9c5766085573632fd7c939c337d.jpg)

![](images/72b9b4fe1d91e683a184c7ef95bb9f20a5f99e7c61aef944025a8125f6372ae2.jpg)

![](images/d8f3fc6f6864ff25ddf80237f59addafdc36596340cf9eab20598849d489a683.jpg)  
Figure 2: Local oscillatory-coupling theory predicts the response and settling cost in the original FPRM recurrence. We control only the Jacobian-identified oscillatory coordinate once with α ∈ {0, 0.5, 1} and then resume unmodified recurrent updates. HRM results in Appendix A.4.1.

Figure 2 tests the oscillatory-coupling mechanism at three levels: (i) the response induced in the complementary state, (ii) its evolution over recurrent time, and (iii) the resulting cost in settling time.

The left panel measures how a perturbation of the oscillatory coordinate subsequently propagates into the complementary latent dynamics over k recurrent steps. For each intervention strength α and rollout step k, the horizontal axis shows the prediction from the root Jacobian, while the vertical axis shows the corresponding change measured from the original nonlinear rollout:

$$
\underbrace { \| \Delta e _ { k } ^ { ( \alpha ) } \| _ { 2 } = \left\| Q \Big ( z _ { t _ { 0 } + k } ^ { ( \alpha ) } - z _ { t _ { 0 } + k } ^ { ( 0 ) } \Big ) \right\| _ { 2 } } _ { \mathrm { a c t u a l ~ n o n l i n e a r ~ r o l l o u t } } \qquad \mathrm { v s . } \qquad \underbrace { \| \widehat { \Delta e } _ { k } ^ { ( \alpha ) } \| _ { 2 } = \| \alpha Q J ^ { k } U u _ { 0 } \| _ { 2 } } _ { \mathrm { l o c a l ~ l i n e a r ~ p r e d i c t i o n } } .\tag{15}
$$

Agreement with the identity line shows that the response induced in the complementary state is quantitatively predicted by the local Jacobian.

The middle panel tests whether this prediction remains accurate throughout the subsequent nonlinear rollout. Consistent with Eq. 14, the measured response follows the predicted alternating temporal envelope, including the proportional reduction under $\alpha = 0 . 5$

The right panel tests whether the same local mode produces the predicted delay in convergence. Using the root-relative settling criterion $\tau _ { \delta } ( e ) = \operatorname* { i n f } \{ k : \| e _ { j } \| \leq \operatorname { i n f }$ for all $j \geq k \bar  \}$ from Eq. 7, we define $\hat { \tau } ^ { ( \alpha ) } : = \tau _ { \delta } ( \hat { e } ^ { ( \alpha ) } )$ and $\tau ^ { ( \alpha ) } : = \tau _ { \delta } ( e ^ { ( \alpha ) } )$ ) using the same sustained-window implementation. The corresponding gains from removing the oscillatory coordinate are $\Delta \tau _ { \mathrm { p r e d } } = \hat { \tau } ^ { ( 1 ) } - \bar { \hat { \tau } } ^ { ( 0 ) }$ and $\Delta \tau _ { \mathrm { o b s } } =$ $\tau ^ { ( 1 ) } - \tau ^ { ( 0 ) }$ . The identity-line agreement therefore confirms that the local prediction quantitatively matches the actual settling-time gain from removing the identified oscillatory component.

Together, the three columns test the causal chain $u _ { 0 } ~ \to ~ \Delta e _ { k } \to \Delta \tau _ { \delta }$ in the original FPRM. The root Jacobian predicts how the oscillatory coordinate drives the complementary latent state, and removing that coordinate accelerates settlement while preserving the decoded output. Extended analysis for distinct two-dimensional spiral mode in HRM is provided in Appendix A.4.1, where we obtain the same prediction–measurement agreement.

## 4 QUOTIENT DYNAMICS CONVERGENCE

Our analysis in Section 3 shows that slowly decaying oscillatory modes can limit convergence speed in a locally linearized regime near a solution root. This does not, however, justify suppressing oscillatory motion throughout the entire reasoning trajectory. Rotational or alternating dynamics may still participate in the reasoning computation, and removing them indiscriminately can disrupt reasoning. Indeed, persistently suppressing an alternating trajectory component in FPRM before output stabilization degrades task performance (see Appendix A.4.4). Rather than attempting to directly correct the oscillatory bottlenecks identified above, QRR takes a complementary architectural approach by introducing an exact token-local gauge symmetry.

Importantly, QRR does not assume any correspondence between these gauge directions and the slow modes identified in other reasoners. QRR learns its own recurrent trajectory. The symmetry instead introduces token-local rotational degrees of freedom whose motion is separated from the task-relevant quotient dynamics. How this symmetry changes the learned dynamics is therefore an empirical question: QRR still retains oscillatory modes in quotient space, but does not develop the isolated dominant slow mode observed in FPRM or HRM.

## 4.1 TOKEN-LOCAL GAUGE SYMMETRY

Let the recurrent state consist of S token states $h = ( h _ { 1 } , \ldots , h _ { S } )$ , with $h _ { s } \in \mathbb { C } ^ { d } .$ , together with momentum states $m _ { s } \in \mathbb { C } ^ { d }$ . In our implementation, each real-valued token state $\bar { y _ { s } } ~ \in ~ \mathbb { R } ^ { 2 d }$ is reinterpreted as $h _ { s } \in \mathbb { C } ^ { d }$ through the fixed pairing $h _ { s , c } = y _ { s , 2 c - 1 } + i y _ { s , 2 c } , c = 1 , \ldots , d$ . The input representation x is gauge-neutral. We introduce the group $G = U ( 1 ) ^ { S }$ , with one phase $\phi _ { s }$ for each token. For $g _ { \phi } = ( e ^ { i \bar { \phi } _ { 1 } } , \bar { \phi } _ { 1 } \bar { \phi } { s } ) \in { \cal G }$ , the group action is

$$
g _ { \phi } \cdot ( h , m ) = \bigl ( ( e ^ { i \phi _ { s } } h _ { s } ) _ { s = 1 } ^ { S } , ( e ^ { i \phi _ { s } } m _ { s } ) _ { s = 1 } ^ { S } \bigr ) , \qquad g _ { \phi } \cdot x = x .\tag{16}
$$

Thus, all complex channels within a token share a common phase, while different tokens may rotate independently. Equivalently, in real coordinates, let J denote the block-diagonal infinitesimal generator of the phase rotation, so that the gauge tangent associated with token s is generated by $\mathcal { I } y _ { s }$ (or, equivalently, by $i h _ { s }$ in complex coordinates). We require the recurrent map $T$ to be G-equivariant while the prediction readout $\mathcal { R }$ is G-invariant,

$$
T ( g \cdot z ; x ) = g \cdot T ( z ; x ) , \qquad \mathcal R ( g \cdot z ; x ) = \mathcal R ( z ; x ) , \qquad z = ( h , m ) .\tag{17}
$$

Hence, states that differ only by token-local phase are identified as the same task-relevant state, and motion along the corresponding gauge directions is invisible to the prediction readout.

## 4.2 QUOTIENT DYNAMICS

The symmetry identifies all states on the same group orbit, $[ z ] = G \cdot z = \{ g \cdot z : g \in G \}$ , yielding the quotient state space $\mathcal { Q } = \mathcal { Z } / G$ . Because the recurrent map is G-equivariant, it induces a well-defined map on the quotient,

$$
{ \overline { { T } } } ( [ z ] ; x ) = [ T ( z ; x ) ] .\tag{18}
$$

The quotient removes only symmetry-generated degrees of freedom, while preserving all remaining latent variation. Accordingly, a state may satisfy $T ( z ^ { \star } ; x ) = g ^ { \star } \cdot z ^ { \star }$ for some $g ^ { \star } \in G ,$ , making $z ^ { \star }$ a relative equilibrium where the unreduced trajectory may continue along the group orbit, while $\overline { { T } } ( [ z ^ { \star } ] ; x ) \bar { = } [ z ^ { \star } ]$ is an ordinary fixed point in the quotient. Moreover, exact equivariance ensures that a perturbation along the gauge orbit remains gauge-tangent after the recurrent update, and therefore produces no change in the quotient state.

QRR

At a relative equilibrium, the quotient analogue of Eq. 4 takes the form $e _ { t + 1 } = A _ { Q } e _ { t } + B _ { G } u _ { t }$ with $B _ { G } = 0$ , so this symmetry-generated motion cannot transmit its timescale into the quotient dynamics. We formalize this non-forcing property in Proposition A.2.

Quotient-based halting. The quotient view changes what it means for recurrent reasoning to have converged. At a relative equilibrium, the state may continue moving along its gauge orbit while its quotient state remains fixed. Thus, requiring raw-state stationarity can mistake symmetry-generated phase motion for unfinished computation even after the quotient state has converged. Figure 3 illustrates this distinction between raw-state and quotient convergence. In FPRM, residual oscillatory motion prevents the raw latent state from becoming stationary until it is damped by the step-size controller, making its fixed-point halting criterion sensitive to this motion. In QRR, by contrast, the raw state may continue moving along its gauge orbit after the quotient trajectory has converged to numerical precision, allowing convergence to be detected substantially earlier. We therefore measure convergence modulo the gauge,

![](images/cfef7c1d8567ef8a6797a414e71d51356261355996a983aca109c28f7222cc43.jpg)

$$
r _ { G } ( z ) = \operatorname* { i n f } _ { g \in G } \frac { \| T ( z ; x ) - g \cdot z \| } { \| z \| + \epsilon } .\tag{19}
$$

![](images/2565b8d707288ed1412e1ffc6905d04f82a545a5906665552858631d2660bc99.jpg)  
Figure 3: Convergence and halting in quotient space. halting in quotient space.

This residual vanishes when the current state and its next update differ only by a gauge transformation, while remaining sensitive to motion in the quotient. At inference, QRR halts once $r _ { G }$ remains below a calibrated threshold for a prescribed persistence window.

## 4.3 GAUGE-EQUIVARIANT RECURRENT BLOCK

QRR adopts the recurrent Transformer block of FPRM as its backbone, while replacing its internal operations with gauge-equivariant counterparts. Under the token-local symmetry in Equations 16–17, the latent state and momentum transform covariantly with the local token phase, while attention weights and prediction readouts remain gauge-invariant. Figure 4 provides an overview of the overall QRR architectural workflow. Full architectural details are in Appendix A.3.

Relative transport. Because each token has its own local phase, features from different tokens must be aligned before they can be mixed. We associate each token with a covariant phase $\gamma _ { s } \in \dot { U } ( 1 )$ , satisfying $\gamma _ { s } \mapsto e ^ { i \phi _ { s } } \gamma _ { s }$ , and define the relative transport $U _ { s  t } = \dot { \gamma } _ { s } \gamma _ { t } ^ { * }$ . Under a pure gauge rotation, $U _ { s  t } \mapsto e ^ { i ( \phi _ { s } - \phi _ { t } ) } U _ { s  t } ,$ , providing the relative phase required to transport features from token t into the local frame of token s. We use this transport for both token mixing and self-attention.

![](images/69485c890ea4d7351709ddf603b1e053df7f993c82361e82eedbddbda3d5152d.jpg)  
Figure 4: Overview of the QRR recurrent block.

Gauge-equivariant attention. Queries, keys, and values are produced by complex-linear maps. Using the relative transport, we define the attention scores as

$$
a _ { s t } = \frac { 1 } { \sqrt { d } } \operatorname { R e } [ U _ { s  t } q _ { s } ^ { \dag } k _ { t } ] ,\tag{20}
$$

which are gauge-invariant. Values are likewise transported into the local frame of token $s$ before aggregation, $o _ { s } = W _ { O } \sum _ { t }$ softmax<sub>t</sub> $( a _ { s t } ) U _ { s  t } v _ { t }$ , so that $O _ { s }$ transforms covariantly with token s.

Gauge-covariant input injection and feed-forward layers. Because the input $x _ { s }$ is gauge-neutral, it cannot be added directly to the recurrent state. We construct a covariant phase carrier

$$
D _ { s } = \frac { \langle W _ { d } x _ { s } , h _ { s } \rangle } { | \langle W _ { d } x _ { s } , h _ { s } \rangle | + \epsilon } , \qquad h _ { s } \gets \alpha _ { 0 } h _ { s } + \beta _ { 0 } D _ { s } x _ { s } ,\tag{21}
$$

where $D _ { s } \mapsto e ^ { i \phi _ { s } } D _ { s }$ , so that the injected input transforms in the local frame of token s. For the feed-forward block, we use an invariant gate with an equivariant value branch,

$$
\mathrm { G \mathrm { - } S w i G L U } ( h ) = W _ { \mathrm { d o w n } } \left[ \mathrm { S i L U } ( | W _ { g } h | ) \odot W _ { v } h \right] .\tag{22}
$$

Momentum-carrying recurrent dynamics. We use a momentum-carrying second-order recurrence to evolve the latent state. At each recurrent step, the gauge-equivariant trunk produces a state proposal $z _ { t } = \Phi _ { \boldsymbol { \theta } } ( y _ { t } ; x , m _ { t } )$ , defining the residual $F _ { t } = z _ { t } - y _ { t }$ . We update the momentum as

$$
p _ { t } = \beta m _ { t } + ( 1 - \beta ) F _ { t } , \qquad \beta = \sigma ( b _ { \mathrm { H } } ) ,\tag{23}
$$

where $b _ { \mathrm { H } } \in$ R is a learned global scalar. The recurrent state is then updated by

$$
y _ { t + 1 } = y _ { t } + p _ { t } , \qquad m _ { t + 1 } = p _ { t } .\tag{24}
$$

The carried momentum is also fed back into the recurrent trunk at the next step.

## 4.4 SPECTRAL DOMINANCE AND CONVERGENCE IN REASONING DYNAMICS

We next examine how these dynamical constraints are reflected in the learned spectra of QRR.

We characterize the local spectral structure of FPRM and QRR around certified solution roots that decode to the ground truth. FPRM roots are ordinary fixed points of the latent dynamics, whereas QRR roots are relative equilibria, fixed points in quotient space. We linearize FPRM in latent space and QRR in quotient

![](images/515be380bc0916edd0bb333b2de794e0cbe40948fbc0d89044b2774160b8c5ae.jpg)

![](images/46108275429e1540361afe8397df6bbb6f1968eaf8ac6e99a90f95ed70ad2641.jpg)

![](images/4e15c607888206a7cd5b8e9bcd2108a818eb2f2ed21db439a93f1be5f0e1f309.jpg)  
Figure 5: Spectral analysis and settling near solution roots for FPRM and QRR. Results for HRM in Appendix A.4.2

space, and report spectral rates as $| \lambda | ^ { 1 / L }$ , where L is the number of recurrent layer-forwards (lfwd) per native step. Following Section 3, we decompose each system into its leading spectral mode and complementary dynamics. For FPRM, the leading mode is slow oscillatory block identified in Section 3.2. For QRR, both are derived from quotient space.

Figure 5 (a) shows QRR operates in a faster spectral regime with less clear separation between its leading and remaining dynamics. Figure 5 (b) quantifies this separation using the dominance score $D = \log r _ { \mathrm { c o m p } } / \log r _ { \mathrm { s l o w } } , D = 1$ indicates matched timescales, while larger D indicates a more isolated slow leading mode. FPRM frequently exhibits strong spectral dominance, whereas QRR remains much closer to $D = 1$ despite retaining some weaker oscillatory modes. Panel (c) shows the corresponding time settlement. QRR settles earlier than FPRM, while factoring out the leading mode accelerates FPRM (218→162) but the effect remains only modest in QRR (138→130). Thus, QRR operates in a faster convergence regime with weaker dependence on a single slow leading mode.

## 5 EXPERIMENTS

Datasets and baselines We evaluate on four reasoning benchmarks covering constraint satisfaction, path finding, and abstract transformation. Sudoku-Extreme consists of challenging partially filled 9 × 9 Sudoku puzzles with unique solutions; following the standard HRM protocol, we train on 1,000 base puzzles with 1,000× augmentation and evaluate using exact sequence accuracy on the full test set (Wang et al., 2025). Maze-Hard is a $3 0 \times 3 0$ shortest-path benchmark with 1,000 training and 1,000 test mazes. We follow the FPRM setup with no augmentation (Movahedi et al., 2026). For Mini-ARC, a compact $5 \times 5 \mathrm { A R C }$ variant for abstract rule induction (Kim et al., 2022; Huang et al., 2026), we use the released tasks with 100 data augmentations per task. We additionally evaluate on Maze-Unique, following the EqR construction. Unlike Maze-Hard, where multiple optimal paths may correspond to a single supervised target, Maze-Unique uses perfect tree-structured mazes with a unique start–goal solution (Kim et al., 2022). This provides a cleaner convergence analysis, avoiding the ambiguous solution geometry that can hinder formation of a single stable basin (Huang et al., 2026). We compare against HRM, TRM, EqR, and FPRM, spanning hierarchical recurrence, recursive refinement, learned attractors, and fixed-point adaptive halting.

## 5.1 RESULTS

Table 1 summarizes both reasoning performance and computational cost. In Table 1 (a), we prioritize matched evaluation from an official checkpoint when available, otherwise use the official result, and reproduce the baseline under its native setup. HRM, TRM, and EqR use their own fixed inference budgets, whereas FPRM and QRR halt adaptively according to fixed-point residuals evaluated in the latent and quotient space, respectively. For the latter, we report halt-or-cap performance with caps of 35k recurrent steps on Sudoku and 3k on the Maze tasks. QRR achieves the strongest performance on Sudoku, Maze-Unique, and Mini-ARC while remaining competitive on Maze-Hard. Table 1 (b) reports corresponding runtime cost in single A100 GPU on Sudoku-Extreme. QRR takes approximately 1.5× higher LFWD latency and 1.3× higher memory than FPRM, reflecting the additional cost of its complex-valued gauge-equivariant operations and explicit phase transport. Nevertheless, faster convergence outweighs this overhead. The median end-to-end rollout time decreases from 5.4 s to 1.9 s, and the p95 from 168.1 s to 60.0 s (2.8× faster). Thus, despite its higher per-LFWD cost, QRR completes full reasoning rollout faster than FPRM.

Table 1: Reasoning performance and runtime. (a) exact accuracy on Sudoku-Extreme, Maze-Hard, and Maze-Unique, and task-level Pass@1 on Mini-ARC. Models with adaptive halting (FPRM and QRR) report halt-or-cap results. Superscripts denote: <sup>1</sup> evaluation matched from an official checkpoint, <sup>2</sup> official report, and <sup>3</sup> full reproduction following the native setup. (b) Comparison of per-layer-forward (LFWD) computational cost and end-to-end rollout time (s).  
(a) Reasoning performance
<table><tr><td rowspan="2">Model</td><td rowspan="2">Params (M)</td><td rowspan="2">Sudoku-Extreme Exact ↑</td><td colspan="2">Maze</td><td rowspan="2">Mini-ARC Pass@1↑</td></tr><tr><td>Hard Exact ↑</td><td>Unique Exact ↑</td></tr><tr><td>HRM</td><td>27</td><td>55.4¹</td><td> $7 4 . 5 ^ { 2 }$ </td><td> $5 1 . 9 ^ { 3 }$ </td><td> $5 6 . 4 ^ { 3 }$ </td></tr><tr><td>TRM</td><td>7</td><td>75.71</td><td> $8 5 . 3 ^ { 2 }$ </td><td> $5 8 . 0 ^ { 3 }$ </td><td> $4 0 . 9 ^ { 3 }$ </td></tr><tr><td>EqR</td><td>7</td><td>89.91</td><td>=</td><td> $8 8 . 9 ^ { 2 }$ </td><td> $4 1 . 6 ^ { 3 }$ </td></tr><tr><td>FPRM</td><td>7</td><td>91.51</td><td> $8 6 . 4 ^ { 1 }$ </td><td></td><td></td></tr><tr><td>FPRM (Rep.)</td><td>7</td><td>83.73</td><td>85.33</td><td> $9 4 . 7 ^ { 3 }$ </td><td>57.1³</td></tr><tr><td>QRR</td><td>7</td><td>97.9</td><td>85.8</td><td>97.0</td><td>62.4</td></tr></table>

(b) Runtime comparison
<table><tr><td>Setting</td><td>QRR</td><td>FPRM</td></tr><tr><td>Per LFWD (B = 512)</td><td></td><td></td></tr><tr><td>Runtime (ms)</td><td>34.7</td><td>22.3</td></tr><tr><td>Peak memory (GB)</td><td>2.88</td><td>2.14</td></tr><tr><td>End-to-end rollout</td><td></td><td></td></tr><tr><td>Coverage (%)</td><td>98.0</td><td>100.0</td></tr><tr><td>Halt step,  $\mathrm { p } 5 0 / \mathrm { p } 9 5$ </td><td>27 / 865</td><td>122 / 3,769</td></tr><tr><td>Runtime (s), p50 / p95</td><td>1.9 / 60.0</td><td>5.4 / 168.1</td></tr></table>

Table 2: Convergence and adaptive halting. (a) Root convergence on Sudoku-Extreme. (b) Adaptive halting of FPRM and QRR across tasks. Coverage is the fraction halted within finite cap.

(a) Root convergence on Sudoku-Extreme  
(b) Adaptive halting (cap Sudoku 35k, Maze-Unique 3k)
<table><tr><td>Model</td><td> $\tau _ { \mathrm { r o o t } }$  median [IQR] ↓</td><td>Correct root found (%) ↑</td></tr><tr><td>HRM</td><td>1,032 [744, 1,272]</td><td>55.2</td></tr><tr><td>EqR</td><td>672 [546, 966]</td><td>86.1</td></tr><tr><td>FPRM</td><td>234 [152, 396]</td><td>90.4</td></tr><tr><td>QRR</td><td>122 [108, 148]</td><td>89.6</td></tr></table>

<table><tr><td colspan="3"></td><td rowspan="2"> $\tau _ { \mathrm { h a l t } }$  median [IQR] ↓</td><td rowspan="2">Acc.@halt (%) ↑</td><td rowspan="2">False halt (%)↓</td></tr><tr><td>Task</td><td>Model</td><td>Coverage (%) ↑</td></tr><tr><td rowspan="2">Sudoku-Extreme</td><td>FPRM</td><td>100.0</td><td>122 [39, 295]</td><td>91.1</td><td>8.9</td></tr><tr><td>QRR</td><td>98.0</td><td>27 [23, 63]</td><td>99.9</td><td>0.1</td></tr><tr><td rowspan="2">Maze-Unique</td><td>FPRM</td><td>96.4</td><td>36 [27, 76]</td><td>98.2</td><td>1.8</td></tr><tr><td>QRR</td><td>98.0</td><td>42 [32, 61]</td><td>98.5</td><td>1.5</td></tr></table>

Table 2 (a) evaluates root convergence on Sudoku-Extreme using Anderson acceleration applied to each trained recurrent map. Full solver, certification, and compute-normalization details are provided in Appendix A.5.2. Correct rootfound is the fraction of examples for which the solver finds a root decoding to the ground-truth solution, while $\tau _ { \mathrm { r o o t } }$ is the first sustained entry of the native trajectory into a neighborhood of that root, measured in recurrent layer-forward equivalents. For QRR, distance is measured in quotient space after removing token-local gauge motion. QRR reaches solution roots significantly earlier than the baselines while maintaining a root-finding rate comparable to FPRM. Table 2 (b) evaluates deployed adaptive halting under finite rollout caps. Coverage is the fraction halted within the cap, while ${ \tau } _ { \mathrm { h a l t } }$ reports the median halting step and IQR over halted examples. Among these, Acc.@halt is the fraction with an exactly correct prediction, and False halt the fraction halted incorrectly. QRR halts earlier with fewer false halts on Sudoku, while achieving comparable halting depth with higher coverage and accuracy on Maze-Unique.

## Ablations

We ablate the main structural components of QRR. Restricting the tokenlocal $U ( \bar { 1 } ) ^ { S }$ symmetry to a single global U(1) preserves moderate task accuracy but never triggers halting, indicating that a shared global phase is too restrictive to account for the

<table><tr><td>Variant</td><td>Exact (%) ↑</td><td>Coverage (%) ↑ False halt (%) ↓</td><td></td><td> $\tau _ { \mathrm { h a l t } } \left[ \mathrm { I Q R } \right] \downarrow$ </td></tr><tr><td>Training</td><td></td><td></td><td></td><td></td></tr><tr><td>QRR</td><td>97.9</td><td>98.0</td><td>0.1</td><td>27 [23, 63]</td></tr><tr><td>global U(1)</td><td>91.3</td><td>0.0</td><td></td><td>CAP (never fires)</td></tr><tr><td>first order w/o momentum</td><td>87.8</td><td>79.6</td><td>0.1</td><td>28 [22, 44]</td></tr><tr><td>FPRM trunk + momentum</td><td>77.0</td><td>100</td><td>23.0</td><td>12,650 [12,522, 12,871]</td></tr></table>

Table 3: QRR ablations on Sudoku-Extreme.

residual token-wise phase motion that persists after the decoded answer settles. Removing momentum and using first-order dynamics disrupts both exact accuracy and coverage, while leaving the faster halting time and quotient halting precision of trajectories that do settle largely intact. Finally, replacing the G-equivariant trunk with FPRM trunk while applying momentum sharply degrades accuracy and leads to unreliable convergence. All examples eventually halt, but only after longer trajectories and with a high false-halt rate.

## 6 CONCLUSION

We studied how non-stationary latent dynamics constrain convergence in recurrent reasoners. Our analysis shows that a slowly decaying oscillatory mode can transmit its rate to coupled latent dynamics and become the bottleneck to settlement, while direct suppression of such motion can also disrupt the reasoning computation. This motivates QRR, which does not eliminate oscillations based on their dynamical appearance, but instead introduces an exact token-local gauge symmetry under which symmetry-generated motion leaves the quotient state unchanged by construction. Exact gauge equivariance prevents this motion from forcing the quotient dynamics, allowing convergence and halting to be defined modulo the symmetry. Across various reasoning tasks, QRR achieves strong reasoning performance while reaching solution roots much earlier and supporting reliable adaptive halting. These results suggest that convergence in recurrent reasoning is better understood in terms of task-relevant quotient space than raw latent space.

## ACKNOWLEDGMENTS

This work was supported by the Hector Foundation.

## REFERENCES

Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 20232–20242. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1471. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/7f70331dbe58ad59d83941dfa7d975aa-Paper-Conference.pdf.

Hugh Blayney, Alvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville,<sup>´</sup> Michael M. Bronstein, and Xiaowen Dong. A mechanistic analysis of looped reasoning language models, 2026.

Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Velickovi ˇ c. Geometric deep learning:´ Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

Taco Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant convolutional networks and the icosahedral CNN. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 1321–1330. PMLR, 2019a. URL https://proceedings. mlr.press/v97/cohen19d.html.

Taco S. Cohen, Mario Geiger, and Maurice Weiler. A general theory of equivariant cnns on homogeneous spaces. arXiv preprint arXiv:1811.02017, 2019b.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped transformers for length generalization. In International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=2edigk8yoU.

Jacob Fein-Ashley and Paria Rashidinejad. Solve the loop: Attractor models for language and reasoning, 2026. URL https://arxiv.org/abs/2605.12466.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach, 2025. URL https://arxiv.org/abs/2502. 05171.

Jan E. Gerken, Jimmy Aronsson, Oscar Carlsson, Hampus Linander, Fredrik Ohlsson, Christoffer Petersson, and Daniel Persson. Geometric deep learning and equivariant neural networks. Artificial Intelligence Review, 56:14605–14662, 2023. doi: 10.1007/s10462-023-10502-7.

Pim De Haan, Maurice Weiler, Taco Cohen, and Max Welling. Gauge equivariant mesh cnns: Anisotropic convolutions on geometric graphs. In International Conference on Learning Represen tations, 2021. URL https://openreview.net/forum?id=Jnspzp-oIZE.

Lingshen He, Yiming Dong, Yisen Wang, Dacheng Tao, and Zhouchen Lin. Gauge equivariant transformer. In Advances in Neural Information Processing Systems, volume 34, pp. 27331–27343, 2021.

Benhao Huang, Zhengyang Geng, and Zico Kolter. Equilibrium reasoners: Learning attractors enables scalable reasoning, 2026. URL https://arxiv.org/abs/2605.21488.

Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025. URL https://arxiv.org/abs/2510.04871.

Gunn Kim. Dynamical phase selection controls compute scaling in looped transformers, 2026. URL https://arxiv.org/abs/2608.26556.

Subin Kim, Prin Phunyaphibarn, Donghyun Ahn, and Sundong Kim. Playgrounds for Abstraction and Reasoning. In NeurIPS Workshop on neuro Causal and Symbolic AI, 2022.

Risi Kondor and Shubhendu Trivedi. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2747–2755. PMLR, 2018.

Martin Krupa. Bifurcations of relative equilibria. SIAM Journal on Mathematical Analysis, 21(6): 1453–1486, 1990. doi: 10.1137/0521081.

Jerrold E. Marsden and Tudor S. Ratiu. Introduction to Mechanics and Symmetry: A Basic Exposition ofClassical Mechanical Systems, volume 17 of Texts in Applied Mathematics. Springer, 1999.

Sajad Movahedi, Vera Milovanovic, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann,´ Valentina Boeva, T. Konstantin Rusch, and Antonio Orvieto. Fixed-point reasoners: Stable and adaptive deep looped transformers, 2026. URL https://arxiv.org/abs/2606.18206.

Francesco Pappone, Donato Crisostomi, and Emanuele Rodola. Two-scale latent dynamics for\` recurrent-depth transformers, 2025. URL https://arxiv.org/abs/2509.23314.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=din0lGfZFd.

Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025. URL https://arxiv.org/ abs/2506.21734.

Maurice Weiler, Patrick Forre, Erik Verlinde, and Max Welling. Coordinate independent convolutional´ networks: Isometry and gauge equivariant convolutions on riemannian manifolds. arXiv preprint arXiv:2106.06020, 2021.

Ruben Wiersma, Elmar Eisemann, and Klaus Hildebrandt. Cnns on surfaces using rotation-equivariant features. ACM Transactions on Graphics, 39(4), 2020. doi: 10.1145/3386569.3392437.

Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, and Yu-Feng Li. Stabilizing recurrent dynamics for test-time scalable latent reasoning in looped language models, 2026. URL https://arxiv.org/abs/2605.26733.

## A APPENDIX

## A.1 THEORETICAL PROOFS

Lemma 3.1 (Exact Sylvester decomposition of a coupled spectral coordinate). Consider the linear dynamics in Eq. 4, with $A \in \mathbb { R } ^ { d \times d } , S \in \mathbb { R } ^ { m \times m }$ , and $B \in \mathbb { R } ^ { d \times m }$ , and assume $\sigma ( A ) \cap \sigma ( S ) = \emptyset$ Then there exists a unique matrix $P \in \mathbb { R } ^ { d \times m }$ satisfying $P S - A P = B$ . The complementary state admits the exact decomposition

$$
e _ { t } = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) + P S ^ { t } u _ { 0 } .\tag{5}
$$

Proof. Define the Sylvester operator

$$
\mathcal { L } ( X ) : = X S - A X , \qquad \mathcal { L } : \mathbb { R } ^ { d \times m }  \mathbb { R } ^ { d \times m } .
$$

Using vectorization,

$$
\operatorname { v e c } ( { \mathcal { L } } ( X ) ) = \left( S ^ { \top } \otimes I _ { d } - I _ { m } \otimes A \right) \operatorname { v e c } ( X ) .\tag{25}
$$

The eigenvalues of $S ^ { \top } \otimes I _ { d } - I _ { m } \otimes A$ are

$$
\mu - \lambda , \qquad \mu \in \sigma ( S ) , \quad \lambda \in \sigma ( A ) .
$$

Because $\sigma ( A ) \cap \sigma ( S ) = \emptyset$ , none of these eigenvalues is zero. Hence the matrix in Eq. 25 is invertible, so L is bijective. Therefore, for every B, there exists a unique $P \in \mathbb { R } ^ { d \times m }$ satisfying

$$
P S - A P = B .
$$

Now define the reparameterized auxiliary variable using P,

$$
w _ { t } : = e _ { t } - P u _ { t } .\tag{26}
$$

Using $e _ { t + 1 } = A e _ { t } + B u _ { t } , u _ { t + 1 } = S u _ { t } ,$ , and $P S - A P = B$ , we obtain

$$
w _ { t + 1 } = e _ { t + 1 } - P u _ { t + 1 }\tag{27}
$$

$$
\mathbf { \nabla } = A e _ { t } + B u _ { t } - P S u _ { t }\tag{28}
$$

$$
= A e _ { t } - A P u _ { t }\tag{29}
$$

$$
= A ( e _ { t } - P u _ { t } ) = A w _ { t } .\tag{30}
$$

Thus

$$
w _ { t } = A ^ { t } w _ { 0 } = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) .
$$

Since $u _ { t } = S ^ { t } u _ { 0 }$ , substituting back into $e _ { t } = w _ { t } + P u _ { t }$ gives

$$
e _ { t } = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) + P S ^ { t } u _ { 0 } ,
$$

which proves Eq. 5.

Theorem 3.2 (Rate inheritance from a slow oscillatory mode). Under the setting of Lemma 3.1, suppose S satisfies Eq. 6, with $\rho ( A ) < r \leq 1$ , B ̸= 0, and $u _ { 0 } \neq 0 . \mathrm { ~ } H r < 1$ , then $e _ { t } \to 0 ;$

$$
\operatorname* { l i m } _ { t \to \infty } \| e _ { t } \| _ { 2 } ^ { 1 / t } = r > \rho ( A ) ,\tag{8}
$$

and

$$
\tau _ { \delta } ( e ) = \frac { \log ( 1 / \delta ) } { - \log r } + O ( 1 ) , \qquad \delta \downarrow 0 .\tag{9}
$$

$I f r = 1$ , then $e _ { t }$ remains bounded but does not converge to any constant, with

$$
\mathcal { E } ( e ) \geq \frac { \| u _ { 0 } \| _ { 2 } \| B \| _ { F } } { \sqrt { 2 } \kappa _ { 2 } ( M ) ( 1 + \| A \| _ { 2 } ) } > 0 .\tag{10}
$$

Proof. Since

$$
S = M ( r Q _ { \omega } ) M ^ { - 1 }
$$

and every eigenvalue of $S$ has modulus $r > \rho ( A )$ , we have

$$
\sigma ( A ) \cap \sigma ( S ) = \emptyset .
$$

Hence Lemma 3.1 gives a unique solution $P$ of $P S - A P = B$ and the exact decomposition

$$
e _ { t } = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) + P S ^ { t } u _ { 0 } .\tag{31}
$$

Since $B \neq 0$ , the Sylvester equation implies $P \neq 0$ . Define

$$
q : = M ^ { - 1 } u _ { 0 } , \qquad C : = P M , \qquad a : = \| q \| _ { 2 } .
$$

Invertibility of M implies $a > 0$ and $C \neq 0$ . Moreover,

$$
S ^ { t } = r ^ { t } M Q _ { \omega } ^ { t } M ^ { - 1 } ,
$$

so

$$
P S ^ { t } u _ { 0 } = r ^ { t } C Q _ { \omega } ^ { t } q .
$$

This change of oscillator coordinates leaves the complementary state $e _ { t }$ and its Euclidean norm unchanged. Choose ν such that

$$
\rho ( A ) < \nu < r .
$$

There exists $C _ { \nu } < \infty$ satisfying

$$
\| A ^ { t } \| _ { 2 } \leq C _ { \nu } \nu ^ { t } , \qquad t \geq 0 .
$$

Writing

$$
\xi _ { t } : = A ^ { t } ( e _ { 0 } - P u _ { 0 } ) , \qquad p _ { t } : = C Q _ { \omega } ^ { t } q ,
$$

we obtain

$$
e _ { t } = r ^ { t } p _ { t } + \xi _ { t } , \qquad \| \xi _ { t } \| _ { 2 } = O ( \nu ^ { t } ) = o ( r ^ { t } ) ,\tag{32}
$$

where $\xi _ { t }$ takes the A-transient part and $r ^ { t } p _ { t }$ corresponds to the oscillatory response. We first establish a uniform two-step lower bound on the oscillatory response. This accounts for possible cancellation at individual steps. Let

$$
v _ { t } : = \frac { Q _ { \omega } ^ { t } q } { a } \in \mathbb { R } ^ { 2 } , \qquad V _ { t } : = ( v _ { t } \quad Q _ { \omega } v _ { t } ) \in \mathbb { R } ^ { 2 \times 2 } .
$$

Because $Q _ { \omega }$ is orthogonal,

$$
\| v _ { t } \| _ { 2 } = \| Q _ { \omega } v _ { t } \| _ { 2 } = 1 , \qquad \langle v _ { t } , Q _ { \omega } v _ { t } \rangle = \cos \omega .
$$

Thus

$$
V _ { t } ^ { \top } V _ { t } = \left( \begin{array} { c c } { 1 } & { \cos \omega } \\ { \cos \omega } & { 1 } \end{array} \right)
$$

has eigenvalues $1 \pm | \cos \omega |$ . Moreover, because $v _ { t }$ and $Q _ { \omega } v _ { t }$ are non-collinear for $0 < \omega < \pi$ , the Gram matrix $V _ { t } ^ { \top } V _ { t }$ is positive definite. The same eigenvalues occur in $V _ { t } V _ { t } ^ { \top }$ , giving

$$
( 1 - | \cos \omega | ) I _ { 2 } \preceq V _ { t } V _ { t } ^ { \top } \preceq ( 1 + | \cos \omega | ) I _ { 2 } .
$$

Since

$$
p _ { t } = a C v _ { t } , \qquad p _ { t + 1 } = a C Q _ { \omega } v _ { t } ,
$$

we have

$$
\| p _ { t } \| _ { 2 } ^ { 2 } + \| p _ { t + 1 } \| _ { 2 } ^ { 2 } = a ^ { 2 } \| C V _ { t } \| _ { F } ^ { 2 } = a ^ { 2 } \operatorname { t r } ( C V _ { t } { V _ { t } ^ { \top } } C ^ { \top } ) .
$$

Consequently,

$$
a ^ { 2 } ( 1 - | \cos \omega | ) \| C \| _ { F } ^ { 2 } \leq \| p _ { t } \| _ { 2 } ^ { 2 } + \| p _ { t + 1 } \| _ { 2 } ^ { 2 } \leq a ^ { 2 } ( 1 + | \cos \omega | ) \| C \| _ { F } ^ { 2 } .\tag{33}
$$

The lower bound is strictly positive because $0 < \omega < \pi , a > 0$ , and $C \neq 0$ . The results show that $p _ { t }$ can be temporarily collapsed close to 0 but $p _ { t }$ and $p _ { t + 1 }$ cannot become arbitrarily small at the same time.

Extending this result to $e _ { t } = r ^ { t } p _ { t } + \xi _ { t }$ , define the two-step envelope

$$
W _ { t } : = \left( \| e _ { t } \| _ { 2 } ^ { 2 } + r ^ { - 2 } \| e _ { t + 1 } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } .\tag{34}
$$

Eq. 32 gives

$$
\binom { e _ { t } } { r ^ { - 1 } e _ { t + 1 } } = r ^ { t } \binom { p _ { t } } { p _ { t + 1 } } + \binom { \xi _ { t } } { r ^ { - 1 } \xi _ { t + 1 } } ,
$$

where

$$
\left\| \left( { \xi } _ { t } \right) \right\| _ { 2 } = O ( \nu ^ { t } ) = o ( r ^ { t } ) .
$$

The triangle and reverse triangle inequalities therefore imply

$$
\left| W _ { t } - r ^ { t } \left( \| p _ { t } \| _ { 2 } ^ { 2 } + \| p _ { t + 1 } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } \right| = O ( \nu ^ { t } ) .
$$

Combining this with Eq. 33, there exist constants $c _ { - } , c _ { + } > 0$ and $t _ { 0 }$ such that

$$
c _ { - } r ^ { t } \leq W _ { t } \leq c _ { + } r ^ { t } , \qquad t \geq t _ { 0 } .\tag{35}
$$

Now suppose $r < 1$ . Since

$$
\| p _ { t } \| _ { 2 } \leq \| C \| _ { 2 } a ,
$$

Eq. 32 implies $e _ { t } \to 0$ and

$$
\operatorname* { l i m } _ { t \to \infty } \| e _ { t } \| _ { 2 } ^ { 1 / t } \leq r .
$$

If this limsup were strictly smaller than $r ,$ there would exist $0 < s < r$ such that

$$
\| e _ { t } \| _ { 2 } = O ( s ^ { t } ) .
$$

It would follow from Eq. 34 that

$$
W _ { t } = O ( s ^ { t } ) ,
$$

contradicting the lower bound $W _ { t } \geq c _ { - } r ^ { t }$ in Eq. 35. Hence

$$
\operatorname* { l i m } _ { t \to \infty } \| e _ { t } \| _ { 2 } ^ { 1 / t } = r ,
$$

which proves Eq. 8. We next derive the settling-time asymptotic. Since $e _ { t } \to 0 , \tau _ { \delta } ( e )$ is finite for every $\delta > 0$ . The positive lower bound in Eq. 35 rules out an eventually zero trajectory,

$$
\tau _ { \delta } ( e ) \longrightarrow \infty \qquad \mathrm { a s } \delta \downarrow 0 .
$$

Thus, for sufficiently small δ, the relevant settling times are at least $t _ { 0 } . \mathrm { I f } t \geq t _ { 0 }$ satisfies

$$
c _ { + } r ^ { t } \leq \delta ,
$$

then, for every $k \geq t ,$

$$
\| e _ { k } \| _ { 2 } \leq W _ { k } \leq c _ { + } r ^ { k } \leq c _ { + } r ^ { t } \leq \delta .
$$

Therefore, for sufficiently small δ,

$$
\tau _ { \delta } ( e ) \leq \left\lceil \frac { \log ( c _ { + } / \delta ) } { - \log r } \right\rceil \leq 1 + \frac { \log ( c _ { + } / \delta ) } { - \log r } = \frac { \log ( 1 / \delta ) } { - \log r } + O ( 1 ) .\tag{36}
$$

Conversely for the lower bound, i ${ \mathrm { : } } \| e _ { k } \| _ { 2 } \leq \delta$ for every $k \geq t ,$ then

$$
W _ { t } \leq \delta { \sqrt { 1 + r ^ { - 2 } } } .
$$

At $t = \tau _ { \delta } ( e ) \geq t _ { 0 } ,$ the lower bound in Eq. 35 therefore gives

$$
c _ { - } r ^ { \tau _ { \delta } ( e ) } \leq \delta \sqrt { 1 + r ^ { - 2 } } ,
$$

and hence

$$
\tau _ { \delta } ( e ) \geq { \frac { \log \left( c _ { - } / ( \delta { \sqrt { 1 + r ^ { - 2 } } } ) \right) } { - \log r } } = { \frac { \log ( 1 / \delta ) } { - \log r } } + O ( 1 ) .\tag{37}
$$

Combining Eq. 36 and Eq. 37 yields

$$
\tau _ { \delta } ( e ) = \frac { \log ( 1 / \delta ) } { - \log r } + O ( 1 ) , \qquad \delta \downarrow 0 ,
$$

which proves Eq. 9. This shows that $e _ { t }$ inherits the slow rate of the oscillatory mode with $r < 1$ due to the coupling.

Now it remains to consider $r = 1$ . Eq. 32 becomes

$$
e _ { t } = p _ { t } + \xi _ { t } , \qquad p _ { t } = C Q _ { \omega } ^ { t } q , \qquad \xi _ { t } \to 0 .\tag{38}
$$

The sequence $\left( e _ { t } \right)$ is bounded because $\left( { p _ { t } } \right)$ is bounded and $\xi _ { t }$ decays geometrically. Write

$$
\begin{array} { r } { \boldsymbol { q } = \boldsymbol { a } ( \cos \phi , \sin \phi ) ^ { \top } . } \end{array}
$$

Since $0 < \omega < \pi .$ , both $e ^ { i \omega }$ and $e ^ { 2 i \omega }$ differ from 1. For $\ell \in \{ 1 , 2 \}$ , the geometric-sum formula gives

$$
{ \frac { 1 } { T } } \sum _ { t = 0 } ^ { T - 1 } e ^ { i \ell t \omega } = { \frac { 1 - e ^ { i \ell T \omega } } { T ( 1 - e ^ { i \ell \omega } ) } } \longrightarrow 0 .
$$

Applying the $\ell = 1$ identity to

$$
Q _ { \omega } ^ { t } q = a \left( \begin{array} { c } { { \cos ( \phi + t \omega ) } } \\ { { \sin ( \phi + t \omega ) } } \end{array} \right)
$$

yields zero-mean for the rotating vector $Q _ { \omega } ^ { t } q$

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Q _ { \omega } ^ { t } q \longrightarrow 0 .\tag{39}
$$

For the second moment, the double-angle identities give

$$
\begin{array}{c} Q _ { \omega } ^ { t } q q ^ { \top } ( Q _ { \omega } ^ { t } ) ^ { \top } = \frac { a ^ { 2 } } { 2 } I _ { 2 } + \frac { a ^ { 2 } } { 2 } \left( \cos ( 2 \phi + 2 t \omega ) \right. \ ~ \sin ( 2 \phi + 2 t \omega )  \\ { \left. \sin ( 2 \phi + 2 t \omega ) \right. \ ~ \left. - \cos ( 2 \phi + 2 t \omega ) \right) . } \end{array}
$$

The $\ell = 2$ identity therefore implies

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Q _ { \omega } ^ { t } q q ^ { \top } ( Q _ { \omega } ^ { t } ) ^ { \top } \longrightarrow \frac { a ^ { 2 } } { 2 } I _ { 2 } .\tag{40}
$$

These limits require no irrationality assumption on $\omega / ( 2 \pi )$ . Since $p _ { t } = C Q _ { \omega } ^ { t } q \mathrm { . }$ , we obtain

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } p _ { t } = C \left( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Q _ { \omega } ^ { t } q \right) \longrightarrow 0 ,
$$

and

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| p _ { t } \| _ { 2 } ^ { 2 } = \mathrm { t r } \left[ C \left( \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Q _ { \omega } ^ { t } q q ^ { \top } ( Q _ { \omega } ^ { t } ) ^ { \top } \right) C ^ { \top } \right] \longrightarrow \frac { a ^ { 2 } } { 2 } \| C \| _ { F } ^ { 2 } .
$$

Thus

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } p _ { t } \longrightarrow 0 , \qquad \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| p _ { t } \| _ { 2 } ^ { 2 } \longrightarrow \frac { a ^ { 2 } } { 2 } \| C \| _ { F } ^ { 2 } .\tag{41}
$$

Fix any $c \in \mathbb { R } ^ { d }$ . The squared distance satisfies

$$
\| e _ { t } - c \| _ { 2 } ^ { 2 } = \| p _ { t } - c \| _ { 2 } ^ { 2 } + 2 \langle p _ { t } - c , \xi _ { t } \rangle + \| \xi _ { t } \| _ { 2 } ^ { 2 } .
$$

Because $\xi _ { t }$ decays geometrically, the last two terms vanish as

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \xi _ { t } \| _ { 2 } \longrightarrow 0 , \qquad \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \xi _ { t } \| _ { 2 } ^ { 2 } \longrightarrow 0 .
$$

Moreover,

$$
\left| \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \langle p _ { t } - c , \xi _ { t } \rangle \right| \leq \left( \| C \| _ { 2 } a + \| c \| _ { 2 } \right) \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \xi _ { t } \| _ { 2 } \longrightarrow 0 .
$$

Using

$$
\| p _ { t } - c \| _ { 2 } ^ { 2 } = \| p _ { t } \| _ { 2 } ^ { 2 } - 2 \langle p _ { t } , c \rangle + \| c \| _ { 2 } ^ { 2 } ,
$$

Eq. 41 therefore yields

$$
\operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| e _ { t } - c \| _ { 2 } ^ { 2 } = \frac { a ^ { 2 } } { 2 } \| C \| _ { F } ^ { 2 } + \| c \| _ { 2 } ^ { 2 } .\tag{42}
$$

For every c, the limsup is at least the limiting Cesaro average,\`

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { \| \boldsymbol { e } _ { t } - \boldsymbol { c } \| _ { 2 } ^ { 2 } } \geq \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \| \boldsymbol { e } _ { t } - \boldsymbol { c } \| _ { 2 } ^ { 2 } .
$$

Taking square roots and then the infimum over c gives

$$
\mathcal { E } ( e ) \geq \frac { a } { \sqrt { 2 } } \| C \| _ { F } = \frac { \| q \| _ { 2 } \| C \| _ { F } } { \sqrt { 2 } } = \frac { \| M ^ { - 1 } u _ { 0 } \| _ { 2 } \| P M \| _ { F } } { \sqrt { 2 } } .\tag{43}
$$

Finally, we express this bound in the original coordinates. Multiplying $P S - A P = B$ on the right by M, and using $S M = M Q _ { \omega }$ when $r = 1$ , gives

$$
B M = P S M - A P M = C Q _ { \omega } - A C .
$$

Since $Q _ { \omega }$ is orthogonal,

$$
\| C Q _ { \omega } \| _ { F } = \| C \| _ { F } .
$$

Consequently,

$$
\begin{array} { r l } & { \| B M \| _ { F } \leq \| C Q _ { \omega } \| _ { F } + \| A C \| _ { F } } \\ & { \qquad \leq \| C \| _ { F } + \| A \| _ { 2 } \| C \| _ { F } } \\ & { \qquad = ( 1 + \| A \| _ { 2 } ) \| C \| _ { F } . } \end{array}
$$

Substituting into Eq. 43 gives

$$
\mathcal { E } ( e ) \geq \frac { \| M ^ { - 1 } u _ { 0 } \| _ { 2 } \| B M \| _ { F } } { \sqrt { 2 } ( 1 + \| A \| _ { 2 } ) } .
$$

The identities $u _ { 0 } = M ( M ^ { - 1 } u _ { 0 } )$ and $B = ( B M ) M ^ { - 1 }$ imply

$$
\lVert u _ { 0 } \rVert _ { 2 } \leq \lVert M \rVert _ { 2 } \lVert M ^ { - 1 } u _ { 0 } \rVert _ { 2 }
$$

and

$$
\| B \| _ { F } \leq \| B M \| _ { F } \| M ^ { - 1 } \| _ { 2 } .
$$

Therefore,

$$
\| M ^ { - 1 } u _ { 0 } \| _ { 2 } \| B M \| _ { F } \geq \frac { \| u _ { 0 } \| _ { 2 } \| B \| _ { F } } { \| M \| _ { 2 } \| M ^ { - 1 } \| _ { 2 } } = \frac { \| u _ { 0 } \| _ { 2 } \| B \| _ { F } } { \kappa _ { 2 } ( M ) } .
$$

We conclude that

$$
\mathcal { E } ( e ) \geq \frac { \| u _ { 0 } \| _ { 2 } \| B \| _ { F } } { \sqrt { 2 } \kappa _ { 2 } ( M ) ( 1 + \| A \| _ { 2 } ) } > 0 ,
$$

which proves Eq. 10. Hence the complementary state remains bounded but cannot converge to any constant. □

Corollary A.1 (Alternating modes). Consider the alternating oscillatory mode

$$
e _ { t + 1 } = A e _ { t } + b u _ { t } , \qquad u _ { t + 1 } = - r u _ { t } ,\tag{44}
$$

where $e _ { t } , b \in \mathbb { R } ^ { d } , u _ { t } \in \mathbb { R } , A \in \mathbb { R } ^ { d \times d } , \rho ( A ) < r \leq 1$ , and $b u _ { 0 } \neq 0 . \mathrm { ~ } I f r < 1$ , the rate and settling-time conclusions from 3.2 hold. More precisely, we can further derive

$$
\operatorname* { l i m } _ { t \to \infty } \frac { \| e _ { t } \| _ { 2 } } { r ^ { t } } = \| ( - r I - A ) ^ { - 1 } b u _ { 0 } \| _ { 2 } > 0 .\tag{45}
$$

$I f r = 1$ , the trajectory approaches an alternating two-cycle, and

$$
\mathcal { E } ( e ) = \| ( I + A ) ^ { - 1 } b u _ { 0 } \| _ { 2 } > 0 .\tag{46}
$$

Proof. Since $\rho ( A ) < r$ , the matrix $- r I - A$ is invertible. Define

$$
v = ( - r I - A ) ^ { - 1 } b u _ { 0 } .\tag{47}
$$

Then $v \neq 0$ and $- r v - A v = b u _ { 0 }$ . Direct substitution into Eq. 44 gives

$$
e _ { t } = A ^ { t } ( e _ { 0 } - v ) + ( - r ) ^ { t } v .\tag{48}
$$

The matrix-power bound implies $A ^ { t } ( e _ { 0 } - v ) = o ( r ^ { t } )$ . Therefore,

$$
r ^ { - t } e _ { t } = ( - 1 ) ^ { t } v + o ( 1 ) ,\tag{49}
$$

which proves Eq. 45.

For $r < 1$ , the norm is consequently bounded above and below by positive multiples of $r ^ { t }$ for all sufficiently large t. The exponential-rate and settling-time conclusions follow as in Theorem 3.2.

For $r = 1 , \mathrm { E q } .$ . 48 gives

$$
e _ { 2 t } \longrightarrow v , \qquad e _ { 2 t + 1 } \longrightarrow - v .\tag{50}
$$

Thus

$$
\mathcal { E } ( e ) = \operatorname* { i n f } _ { c \in \mathbb { R } ^ { d } } \operatorname* { m a x } \{ \| v - c \| _ { 2 } , \| - v - c \| _ { 2 } \} = \| v \| _ { 2 } .\tag{51}
$$

The last equality follows because any ball containing v and −v has radius at least $\lVert \boldsymbol { v } \rVert _ { 2 }$ , attained at $c = 0$ . Since $v \stackrel { \cdot } { = } - ( I + A ) ^ { - 1 } b u _ { 0 }$ when $r = 1$ , this proves Eq. 46. □

Perturbation response In the alternating case $S = - r ,$ the Sylvester equation gives

$$
P ( - r ) - A P = b , \qquad P = ( - r I - A ) ^ { - 1 } b .
$$

Hence the linearized complementary trajectory is

$$
\widehat { e } _ { k } ^ { ( \alpha ) } = A ^ { k } e _ { 0 } + \alpha \big ( P ( - r ) ^ { k } - A ^ { k } P \big ) u _ { 0 } .
$$

Subtracting the $\alpha = 0$ counterfactual isolates the response induced by the oscillatory coordinate,

$$
\widehat { \Delta e } _ { k } ^ { ( \alpha ) } : = \widehat { e } _ { k } ^ { ( \alpha ) } - \widehat { e } _ { k } ^ { ( 0 ) } = \alpha \big ( P ( - r ) ^ { k } - A ^ { k } P \big ) u _ { 0 } .
$$

Since $\rho ( A ) < r$ , we have $A ^ { k } P u _ { 0 } = o ( r ^ { k } )$ , and therefore

$$
\widehat { \Delta e } _ { k } ^ { ( \alpha ) } = \alpha P ( - r ) ^ { k } u _ { 0 } + o ( r ^ { k } ) .
$$

## A.1.1 STRUCTURAL NON-FORCING OF GAUGE MOTION.

The quotient construction in Section 4.2 is stronger than readout invariance, exact equivariance prevents symmetry-generated motion from forcing dynamics transverse to the gauge orbit.

Proposition A.2 (Gauge-to-quotient non-forcing). Let T be differentiable and equivariant under the token-local $G = \check { U } ( 1 ) ^ { S }$ action. Define the gauge tangent space $\mathcal { V } _ { z } : = \mathrm { T a n } _ { z } ( G \cdot z )$ , and let $\Pi _ { z }$ denote the orthogonal projector onto $\mathcal { V } _ { z } ^ { \perp }$ . Then

$$
\Pi _ { T ( z ) } D T ( z ) \xi = 0 , \qquad \xi \in \mathcal { V } _ { z } .\tag{52}
$$

At a relative equilibrium $T ( z ^ { \star } ) = g ^ { \star } \cdot z ^ { \star }$ , define $\widehat { T } = ( g ^ { \star } ) ^ { - 1 }$ · T and $\widehat { J } = D \widehat { T } ( z ^ { \star } )$ . If U<sub>G</sub> has orthonormal columns spanning $\nu * z ^ { \star }$ , then

$$
\widehat J { \cal U } _ { G } = { \cal U } _ { G } , \qquad { \cal B } _ { G } : = \Pi * z ^ { \star } \widehat J { \cal U } _ { G } = 0 .\tag{53}
$$

Proof. Let $\xi \in \mathcal { V } _ { z }$ . By definition of the tangent space to the group orbit, there exists a differentiable curve $g ( \phi ) \in G$ with $g ( 0 ) = e$ such that

$$
\xi = \left. \frac { d } { d \phi } g ( \phi ) \cdot z \right| _ { \phi = 0 } .\tag{54}
$$

By equivariance,

$$
T ( g ( \phi ) \cdot z ) = g ( \phi ) \cdot T ( z ) .\tag{55}
$$

Differentiating at $\phi = 0$ gives

$$
D T ( z ) \xi = \left. \frac { d } { d \phi } g ( \phi ) \cdot T ( z ) \right| _ { \phi = 0 } \in \mathcal { V } _ { T ( z ) } .\tag{56}
$$

Hence

$$
D T ( z ) \mathcal { V } _ { z } \subseteq \mathcal { V } _ { T ( z ) } ,\tag{57}
$$

and therefore

$$
\Pi _ { T ( z ) } D T ( z ) \xi = 0 , \qquad \xi \in \mathcal { V } _ { z } .\tag{58}
$$

Now let $z ^ { \star }$ satisfy $T ( z ^ { \star } ) = g ^ { \star } \cdot z ^ { \star }$ , and define

$$
\widehat { T } = ( g ^ { \star } ) ^ { - 1 } \cdot T .\tag{59}
$$

Then $\widehat { T } ( z ^ { \star } ) = z ^ { \star }$ . Since $G = U ( 1 ) ^ { S }$ is Abelian, $g g ^ { \star } = g ^ { \star } \colon$ g for all g, $g ^ { \star } \in G$ , and hence

$$
\widehat { T } ( g \cdot z ) = ( g ^ { \star } ) ^ { - 1 } \cdot T ( g \cdot z )\tag{60}
$$

$$
= ( g ^ { \star } ) ^ { - 1 } g \cdot T ( z )\tag{61}
$$

$$
= g ( g ^ { \star } ) ^ { - 1 } \cdot T ( z ) = g \cdot { \widehat T } ( z ) .\tag{62}
$$

Thus $\widehat { T }$ is also G-equivariant.

Thus, for any $\xi \in \mathcal { V } _ { z }$ ⋆ and any curve $g ( \phi )$ generating ξ,

$$
\widehat { J } \xi = \left. \frac { d } { d \phi } \widehat { T } \big ( g ( \phi ) \cdot z ^ { \star } \big ) \right| _ { \phi = 0 }\tag{63}
$$

$$
= \left. { \frac { d } { d \phi } } g ( \phi ) \cdot { \widehat { T } } ( z ^ { \star } ) \right| _ { \phi = 0 }\tag{64}
$$

$$
= \left. { \frac { d } { d \phi } } g ( \phi ) \cdot z ^ { \star } \right| _ { \phi = 0 } = \xi .\tag{65}
$$

Therefore $\widehat { J }$ acts as the identity on $\mathcal { V } _ { z }$ ⋆ . If $U _ { G }$ spans this space,

$$
\widehat { J } \bar { U } _ { G } = U _ { G } .\tag{66}
$$

Thus, in the co-moving frame, while the raw state advances along the gauge orbit by $g ^ { \star }$ , removing this common rotation leaves infinitesimal gauge perturbations unchanged to first order.

In direct analogy with the coupling matrix $B = Q J U$ in Section 3.1, define the coupling from the gauge coordinate into the transverse quotient dynamics as

$$
B _ { G } : = \Pi _ { z ^ { \star } } \widehat { J } U _ { G } .\tag{67}
$$

Since $\Pi _ { z ^ { \star } } U _ { G } = 0$ and $\widehat { J } \bar { U } _ { G } = U _ { G }$

$$
B _ { G } = \Pi _ { z ^ { \star } } \widehat { J } U _ { G } = \Pi _ { z ^ { \star } } U _ { G } = 0 .\tag{68}
$$

Finally, let $W$ be an orthonormal basis of $\mathcal { V } _ { z ^ { \star } } ^ { \perp }$ and write $\delta = W e + U _ { G } u$ . Projecting the co-moving linearization onto the quotient coordinates gives

$$
e ^ { + } = W ^ { \top } \widehat { J } ( W e + U _ { G } u )\tag{69}
$$

$$
= W ^ { \top } { \widehat { J } } W e + W ^ { \top } { \widehat { J } } U _ { G } u\tag{70}
$$

$$
= A _ { Q } e ,\tag{71}
$$

where $\boldsymbol { A } _ { \boldsymbol { Q } } : = \boldsymbol { W } ^ { \top } \widehat { \boldsymbol { J } } \boldsymbol { W }$ . Thus the quotient component evolves independently of the gauge coordinate. □

To connect this result to the coupled-mode analysis in Section 3, let W be an orthonormal basis of $\mathcal { V } _ { z ^ { \star } } ^ { \perp }$ <sub>⋆</sub> and write $\delta = W e + U _ { G } u$ . Since $W ^ { \top } \widehat { J } U _ { G } = 0$ , the quotient component evolves locally as

$$
e _ { k + 1 } = A _ { Q } e _ { k } , \qquad A _ { Q } : = W ^ { \top } \widehat { J } W ,
$$

independently of the gauge coordinate $u _ { k }$ . Hence a symmetry-generated gauge rotation cannot transmit its timescale into the quotient dynamics. This statement applies only to the gauge subspace so other oscillatory modes transverse to the gauge orbit remain unconstrained and may continue to participate in the quotient dynamics.

## A.2 LOCAL THEORY AND EMPIRICAL PREDICTION

## Derivation of the local block dynamics.

Let $U \in \mathbb { R } ^ { D \times m }$ have orthonormal columns and satisfy

$$
J ^ { \top } U = U S ^ { \top } .\tag{72}
$$

Transposing gives

$$
U ^ { \top } J = S U ^ { \top } .\tag{73}
$$

Define

$$
Q = I - U U ^ { \top } .\tag{74}
$$

Since $U ^ { \top } U = I , Q$ is the orthogonal projector onto the complement of span(U), and

$$
Q U = 0 , \qquad U ^ { \top } Q = 0 .\tag{75}
$$

For any root-centered displacement $\delta ,$ define

$$
u = U ^ { \top } \delta , \qquad e = Q \delta .\tag{76}
$$

Then $\delta = e + U u$ , with $U ^ { \top } e = 0$ . Under the linearized dynamics

$$
\delta ^ { + } = J \delta ,\tag{77}
$$

the oscillatory coordinate evolves as

$$
u ^ { + } = U ^ { \top } \delta ^ { + }
$$

$$
= U ^ { \top } J ( e + U u )\tag{78}
$$

(79)

$$
= S U ^ { \top } e + S U ^ { \top } U u\tag{80}
$$

$$
\ d u = S u .\tag{81}
$$

Thus the complementary coordinate does not feed back into u to first order.

Similarly,

$$
e ^ { + } = Q \delta ^ { + }
$$

$$
= Q J ( e + U u )\tag{82}
$$

(83)

$$
= Q J Q e + Q J U u ,\tag{84}
$$

where we used $e = Q \epsilon$ . Defining

$$
A : = Q J Q , \qquad B : = Q J U ,\tag{85}
$$

gives

$$
e ^ { + } = A e + B u , \qquad u ^ { + } = S u .\tag{86}
$$

Hence the left-invariant oscillatory coordinate evolves autonomously, while it can still drive the complementary state through the coupling block $B$

## A.3 QRR ARCHITECTURE

QRR follows the recurrent-block organization of FPRM, but replaces the recurrent trunk with the gauge-equivariant construction described in Section 4.3 and augments recurrence with momentum.

Input and recurrent state. Let x denote the embedded task input, which is kept fixed and reinjected at every recurrent step. The recurrent state consists of a latent state $y _ { t }$ and a momentum state $m _ { t }$ , both transforming covariantly under the token-local $G = U ( 1 ) ^ { S }$ action. In real coordinates, $y _ { t } , m _ { t } \in \mathbb R ^ { S \times { 2 d } }$ , with the fixed identification $\mathbb { R } ^ { 2 d } \simeq \mathbb { C } ^ { d }$ used throughout the gauge-equivariant trunk. For each input, we initialize $m _ { 0 } = 0$ and draw $y _ { 0 }$ independently at random. Specifically, each four-dimensional real cell of $y _ { 0 }$ is sampled coordinate-wise from a truncated normal distribution and normalized to unit norm.

Gauge-equivariant recurrent trunk. At recurrent step t, the trunk receives the current state, carried momentum, and input,

$$
z _ { t } = \Phi _ { \boldsymbol \theta } \big ( y _ { t } , m _ { t } ; x \big ) ,\tag{87}
$$

where $\Phi _ { \theta }$ is G-equivariant,

$$
\Phi _ { \theta } ( g \cdot y , g \cdot m ; x ) = g \cdot \Phi _ { \theta } ( y , m ; x ) .\tag{88}
$$

The trunk retains the pre-norm Transformer structure of FPRM, while token mixing, input injection, attention, and feed-forward transformations are replaced by their gauge-equivariant counterparts. The input x is injected through a covariant carrier, while the carried momentum $m _ { t }$ is also fed back into the trunk. The resulting output $z _ { t }$ is a state proposal, rather than the next recurrent state directly.

Momentum-carrying update. We define the proposal residual

$$
F _ { t } = z _ { t } - y _ { t } .\tag{89}
$$

The deployed recurrence applies momentum directly to this residual,

$$
p _ { t } = \beta m _ { t } + ( 1 - \beta ) F _ { t } , \qquad \beta = \sigma ( b _ { \mathrm { H } } ) ,\tag{90}
$$

where $b _ { \mathrm { H } }$ is a learned global scalar. Ignoring an equivalent gauge rephasing described below, the update is

$$
\begin{array} { r } { \widetilde { y } _ { t + 1 } = y _ { t } + p _ { t } , \qquad \widetilde { m } _ { t + 1 } = p _ { t } . } \end{array}\tag{91}
$$

Gauge rephasing. The implementation additionally permits a state-dependent gauge rotation

$$
a _ { t } = \frac { \langle F _ { t } , \mathcal { T } y _ { t } \rangle } { \| y _ { t } \| ^ { 2 } + \epsilon } , \qquad R _ { t } = \exp ( a _ { t } \mathcal { T } ) ,\tag{92}
$$

with $a _ { t }$ detached from the computational graph. The stored recurrent state is then

$$
y _ { t + 1 } = R _ { t } \widetilde { y } _ { t + 1 } , \qquad m _ { t + 1 } = R _ { t } \widetilde { m } _ { t + 1 } .\tag{93}
$$

Since $R _ { t }$ is itself a gauge transformation, this operation doesn’t change quotient state.

Readout and adaptive halting. Predictions are produced by a gauge-invariant readout

$$
\begin{array} { r } { \widehat { y } _ { t } = \mathcal { R } ( y _ { t } ; x ) , \qquad \mathcal { R } ( g \cdot y _ { t } ; x ) = \mathcal { R } ( y _ { t } ; x ) . } \end{array}\tag{94}
$$

Consequently, prediction is insensitive to motion confined to the gauge orbit. Adaptive halting is based on the quotient residual $r _ { G }$ from Equation 19, computed from the joint state $( y _ { t } , m _ { t } )$ after phase alignment. A trajectory halts once this residual remains below the task-specific threshold for the prescribed persistence window; otherwise recurrence continues until the rollout cap.

Recurrent training. The pair $( y _ { t } , m _ { t } )$ is carried across recurrent steps and across successive deepsupervision segments for the same input, and is reset only when a new example is started. Within each segment, the model is unrolled for a task-specific number of recurrent steps, while gradients are propagated only through the final truncated-BPTT window. Earlier steps advance the recurrent state without storing gradients. After each deep-supervision segment, the prediction loss is applied and the carried state is detached before the next segment.

## A.4 ADDITIONAL EXPERIMENTAL RESULTS

## A.4.1 CAUSAL VERIFICATION ON HRM SPIRAL DYNAMICS

We additionally verify the local oscillatory-coupling prediction on HRM, whose solution-root dynamics exhibit a qualitatively different oscillatory geometry from the alternating mode analyzed for FPRM in Section 3.1. Here, the identified slow spectral component is a two-dimensional real invariant block

$$
S \in \mathbb { R } ^ { 2 \times 2 } , \qquad \sigma ( S ) = \{ r e ^ { \pm i \omega } \} , \qquad 0 < \omega < \pi ,\tag{95}
$$

with $\rho ( A ) < r \leq 1$ . Thus, rather than alternating along a single direction, the slow coordinate rotates while decaying toward the solution root. We use exactly the same intervention protocol used for FPRM in main paper. The local linearization from Eq. 13 predicts

$$
\widehat { \Delta e } _ { k } ^ { ( \alpha ) } = \alpha Q J ^ { k } U u _ { 0 } = \alpha \left( P S ^ { k } - A ^ { k } P \right) u _ { 0 } ,\tag{96}
$$

![](images/3586d34c0932aafd6f289e6874a3b6e034ea0d2c68ed847d5f14bb3d0b57a1a8.jpg)

![](images/173700ad9916efa54b5b3ecc16436dfc2d2b9f2f0f83b956ef58724027e85b76.jpg)

![](images/f31e6394a8c02bd56c736f00477b4a2141e5d015098476f8a4ce25fcb655163b.jpg)  
Figure 6: The local oscillatory-coupling prediction extends to the two-dimensional spiral dynam ics of HRM. We modify only the Jacobian-identified oscillatory coordinate once with $\alpha { \stackrel { - } { \in } } \{ 0 , 0 . 5 , 1 \}$ and subsequently resume the original nonlinear HRM recurrence.

where the full matrix power $S ^ { k }$ retains the rotation of the two-dimensional spiral mode.

Figure 6 mirrors the three tests reported for FPRM in Figure 2. First, the response measured from the nonlinear rollout agrees with the Jacobian-based prediction

$$
\begin{array} { r } { \left\| Q \left( z _ { t _ { 0 } + k } ^ { ( \alpha ) } - z _ { t _ { 0 } + k } ^ { ( 0 ) } \right) \right\| _ { 2 } \qquad \mathrm { v s . } \qquad \left\| \alpha Q J ^ { k } U u _ { 0 } \right\| _ { 2 } . } \end{array}\tag{97}
$$

Second, the temporal response follows the predicted rotating and decaying evolution, rather than the one-dimensional alternation observed in FPRM. Finally, removing the oscillatory component with $\alpha = 0$ reduces root-relative settling time by an amount consistent with the local prediction, using the same settling criterion and sustained-window implementation as in the main-text analysis. These results show that the intervention agreement in Figure 2 is not specific to the nearly onedimensional flip geometry of FPRM. The same local decomposition predicts the nonlinear response and convergence cost for a qualitatively distinct two-dimensional spiral mode.

## A.4.2 SPECTRAL ANALYSIS ON HRM VS. QRR

![](images/6824f7a6d1c419d2be770356eb446f1735547df62460fcca6f0e9f4e680ca5b2.jpg)

![](images/63eb23ed8d4a0db1e513470ad1b30617fb8eec776a9f7c614423fd0ca36bdd06.jpg)

![](images/e11abaf7c746728ca418a093631b17b967a1758113df077a3fb719878b5dff8c.jpg)  
Figure 7: Spectral organization and settling in HRM and QRR. (a) Leading and complementary spectral rates near certified solution equilibria. QRR rates are measured in quotient space. (b) Distribution of the dominance score $D = \log r _ { \mathrm { c o m p } } / \log r _ { \mathrm { l e a d } }$ . (c) Root-relative settling time before and after factoring out the leading-mode response.

We repeat the spectral analysis of Section 4.4 for HRM, whose local dynamics exhibit the spira geometry studied in Appendix A.4.1. For HRM, we linearize the recurrent H-cycle map around certified solution fixed points. For QRR, we analyze the corresponding linearization in quotient space around relative equilibria. As in the main analysis, we decompose each system into its leading spectral mode and complementary dynamics and normalize spectral rates to recurrent layer-forward.

Figure 7 shows the same distinction at a different absolute timescale. Although the leading– complementary spectral gaps appear similar in magnitude, HRM operates much closer to unit spectral rate, producing a larger separation in decay timescales and a stronger dominance score than QRR. This is reflected in finite-time settlement. Factoring out the leading-mode response reduces the median HRM settling time from 768 to 672 layer-forwards, whereas the corresponding change in the QRR quotient is only 138 to 130. Thus, QRR remains in a substantially faster regime and is less dependent on its leading mode for settlement, even relative to the qualitatively different spiral dynamics of HRM.

## A.4.3 SPECTRAL PROPERTIES OF LEADING OSCILLATORY MODES

For each certified solution root, we linearize the one-step recurrent map and identify the leading spectral mode of the local Jacobian J. We retain roots for which this mode is oscillatory, corresponding to either a complex-conjugate pair or a real negative flip mode. Let U span this leading block, with modulus r, and define its complement by $Q = \mathsf { \check { I } } - U U ^ { \dagger }$ and $A = Q J Q$ . Because the selected block contains the leading eigenmode of $J , r \geq \rho ( A )$ holds by construction. We therefore use $r - \rho ( A )$ to measure the spectral separation between the leading oscillatory mode and the remaining dynamics.

![](images/6084e992e75808103d35c7ee644b7abdd369bb3c15bbde0b57a4c462f3a5f3d8.jpg)

![](images/449e4b67eec2494753ce140f40787a48aaa4e4bdd9559824437313e9bec83efb.jpg)

![](images/5771059da04aeb5644c094b2209b471e0e1ed5c641772b86a97c20ba5ac523a2.jpg)  
Figure 8: Spectral separation of leading oscillatory modes. (a) Leading-mode modulus r and complementary spectral radius $\rho ( A )$ . (b) Spectral margin $r - \rho ( A )$ . (c) Oscillation angle |ω| for FPRM and HRM.

Figure 8 shows a strict spectral gap for all selected roots. The median margin $r - \rho ( A )$ is 0.0176 for FPRM and 0.035 for HRM, with minimum margins of $2 . 2 \times 1 0 ^ { - 4 }$ and $1 . 5 \times 1 0 ^ { - 4 }$ . The leading modes also exhibit distinct geometries. FPRM is concentrated near $| \omega | = \pi$ with median $1 7 9 . 7 ^ { \circ }$ whereas HRM exhibits small-angle spirals with median $3 . 9 ^ { \circ }$ . Thus, the selected oscillatory modes are cleanly separated from the complementary spectrum across both flip and spiral regimes.

## A.4.4 FPRM FLIP MODE SUPPRESSION

The local analysis in Section 3 shows that the slowly decaying flip mode can limit late-stage convergence in FPRM. This does not imply, however, that the same mode is dispensable throughout reasoning. We therefore test whether persistently suppressing the alternating component before the output has stabilized interferes with the computation itself.

For each trajectory, we estimate a one-dimensional alternating direction U from the preceding recurrent updates and, from an intervention time $t _ { 0 }$ onward, modify every subsequent update $d _ { t }$ as

$$
z _ { t + 1 } ^ { ( \alpha ) } = z _ { t } + \left[ I - ( 1 - \alpha ) U U ^ { \top } \right] d _ { t } .\tag{98}
$$

We use $\alpha = { \textstyle { \frac { 1 } { 2 } } }$ as the suppression intervention and compare against the unmodified rollout. Early interventions start before the native output-stabilization (decoded output stabilizes and remains equal for the remainder of the rollout), whereas late interventions start after stabilization. To distinguish suppression of the oscillatory mode from a generic reduction in update magnitude, we additionally suppress a non-oscillatory control direction while matching the removed energy. All comparisons are paired against the corresponding original rollout, and confidence intervals are obtained by bootstrapping over boards.

Early suppression decreases final exact accuracy by 5.9 percentage points, and delays output stabilization. In contrast, removing a comparable amount of update energy along a non-oscillatory direction produces no statistically distinguishable degradation, and suppressing the flip mode only after the output has stabilized has no measurable effect. These results indicate that the oscillatory motion of the same alternating form can still participate in the earlier reasoning computation. This motivates QRR to avoid suppressing oscillations based on their dynamical appearance alone, and instead quotient out only motion whose representational redundancy is guaranteed by symmetry.

Table 4: Persistent suppression of the FPRM flip mode. Suppressing the alternating component before output stabilization reduces final exact accuracy, whereas an energy-matched non-oscillatory intervention does not show a significant effect.
<table><tr><td>Intervention</td><td>∆ final accuracy</td><td>95% CI</td></tr><tr><td>Oscillatory suppression, early</td><td>-5.9 pp</td><td> $[ - 1 3 . 1 , - 0 . 6 ]$ </td></tr><tr><td>Energy-matched control, early</td><td>-1.1 pp</td><td>[−2.8,0.0]</td></tr><tr><td>Oscillatory suppression, late</td><td>0.0 pp</td><td>[0.0, 0.0]</td></tr></table>

## A.4.5 HALTING TIME SENSITIVITY TO TASK DIFFICULTY

![](images/f832aeab4de2d551daded2a3c7275f7969fd510a0bd68b755a508c95e685db33.jpg)  
Figure 9: Adaptive halting as a function of Sudoku difficulty. Difficulty as the number of blank cells (top) and the backtrack count (bottom). Left: mean halting time among certified examples. Middle: fraction of examples certified within the rollout budget. Right: exact-match accuracy.

On 42,023 held-out Sudoku-Extreme boards, we evaluate QRR and FPRM under the same halting rules and 35,000-step cap as in Table 2. As shown in Figure 9, halting latency increases with difficulty for both models, but more strongly for QRR. The Spearman correlation is $\rho = 0 . 3 5 6$ versus 0.201 for blank count and 0.365 versus 0.202 for solver backtracks. QRR nevertheless certifies significantly earlier across the difficulty range while maintaining 98.0% overall coverage and consistently higher exact-match accuracy, indicating stronger difficulty-adaptive compute allocation with lower cost.

## A.5 EXPERIMENTAL SETUP

## A.5.1 REASONING TASKS

We evaluate on four reasoning benchmarks. Sudoku-Extreme consists of partially filled 9×9 Sudoku boards with unique solutions and is evaluated by exact-match accuracy of the completed grid (Wang et al., 2025). Maze-Hard contains $3 0 \times 3 0$ shortest-path problems and is evaluated by exact match to the supervised solution path (Wang et al., 2025). Because multiple optimal paths may exist in Maze-Hard, we additionally use Maze-Unique, constructed from perfect tree-structured mazes so that each start–goal pair has a unique solution (Kim et al., 2022; Huang et al., 2026). Mini-ARC is a compact $5 \times 5$ abstract rule-induction benchmark in which the model infers a transformation from demonstration pairs and applies it to a held-out input (Kim et al., 2022; Huang et al., 2026). We follow the standard augmentation-based evaluation, map predictions from each augmented view back to the original frame, aggregate identical candidate grids by voting, and report task-level Pass@1 from the highest-ranked prediction.

## A.5.2 ROOT CONVERGENCE ANALYSIS

Here we provide detailed protocol for root convergence analysis used in this paper, which is used for the local Jacobian and spectral analyses in Sections 3, 4.4, and Table 2 (a). We firstly identify solution equilibria using Anderson acceleration with history $m = 6 ,$ Tikhonov regularization $1 0 ^ { - 8 }$ and $\beta = 1$ in fp64. The solver runs for a fixed budget and returns the iterate with the smallest residual. For FPRM, HRM, and EqR, the solved maps are one FPRM recurrent step, two HRM H-cycles, and three EqR cycles, respectively. For QRR, Anderson acceleration is applied to the gauge-aligned map.

For each certified solution root, we run the original model from its native initialization and measure when the trajectory reaches the corresponding root. We define $\tau _ { \mathrm { r o o t } }$ as the first step after which the root-relative distance remains below $\delta \doteq 1 0 ^ { - 3 }$ for a persistence window of approximately 96 recurrent layer-forwards. This corresponds to 48 native steps for FPRM and QRR and 4 trajectory steps for HRM. For EqR, whose native step contains 42 layer-forwards, we use two steps, corresponding to 84 layer-forwards. Correct rootfound denotes the fraction of evaluation examples for which the solver returns a certified solution equilibrium decoding to the ground truth.

To compare convergence across architectures with different computation per native recurrent step, we report convergence times in recurrent layer-forward equivalents (LFWD). An LFWD counts one forward evaluation of one recurrent network layer. If one native step of model m evaluates $\ell _ { m }$ recurrent layers, then

$$
\tau _ { \mathrm { l f w d } } ^ { ( m ) } = \ell _ { m } \tau _ { \mathrm { n a t i v e } } ^ { ( m ) } ,
$$

where $\tau _ { \mathrm { n a t i v e } } ^ { ( m ) }$ is the number of native recurrent steps required to reach the certified solution equilibrium.   
Table 5 summarizes the conversion used for each model.

Table 5: Conversion from native recurrent steps to recurrent layer-forward equivalents. An LFWD counts one forward evaluation of one recurrent network layer.
<table><tr><td>Model</td><td>Layer-forward computation per native step</td><td> $\ell _ { m } \ \mathrm { ( L F W D / s t e p ) }$ </td></tr><tr><td>QRR</td><td> $1 \times L _ { \mathrm { l a y e r s } } = 1 \times 2$ </td><td>2</td></tr><tr><td>FPRM</td><td> $1 \times L _ { \mathrm { l a y e r s } } = 1 \times 2$ </td><td>2</td></tr><tr><td>HRM</td><td> $H _ { \mathrm { c y c } } \left( L _ { \mathrm { c y c } } L _ { \mathrm { l a y e r s } } + H _ { \mathrm { l a y e r s } } \right) = 2 ( 2 \cdot 4 + 4 )$ </td><td>24</td></tr><tr><td>EqR</td><td> $H _ { \mathrm { c y c } } ( L _ { \mathrm { c y c } } + 1 ) L _ { \mathrm { l a y e r s } } = 3 ( 6 + 1 ) 2$ </td><td>42</td></tr></table>

## A.5.3 HYPERPARAMETERS

We use the same core QRR architecture across all tasks and tune only the task-specific training, recurrent-compute, and inference settings summarized in Table 6. Architectural details shared across tasks are given in Section A.3.

Table 6: Task-specific training and inference configurations for QRR. The core architecture is shared across tasks. The reported result is from the best in-run evaluation checkpoint.
<table><tr><td>Setting</td><td>Sudoku-Extreme</td><td>Maze-Hard</td><td>Maze-Unique</td><td>Mini-ARC</td></tr><tr><td>Data and optimization</td><td></td><td></td><td></td><td></td></tr><tr><td>Training data</td><td>1K × 1000 aug.</td><td>1K, no aug.</td><td>1K, no aug.</td><td>149 tasks, 100 aug.</td></tr><tr><td>Training epochs</td><td>60K</td><td>60K</td><td>6K</td><td>100K</td></tr><tr><td>Global batch size</td><td>768</td><td>768</td><td>384</td><td>768</td></tr><tr><td>Learning rate</td><td>3×10 -4</td><td>1×10 -4</td><td>2×10 -4</td><td>6×10 -4</td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 3 }$ </td><td>1.0</td><td>1.0</td><td>0.1</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Compute setup</td><td></td><td></td><td></td><td></td></tr><tr><td>Training GPUs</td><td>4× H100</td><td>4× B200</td><td>4× H100</td><td>1× A100</td></tr><tr><td>Parallelism</td><td>DDP ×4</td><td>DDP ×4</td><td>DDP ×4</td><td>single GPU</td></tr><tr><td>Per-GPU batch size</td><td>192</td><td>192</td><td>96</td><td>768</td></tr><tr><td>Recurrence and deep supervision</td><td></td><td></td><td></td><td></td></tr><tr><td>Max DS segment</td><td>12</td><td>24</td><td>24</td><td>8</td></tr><tr><td>Recurrent steps / DS segment</td><td>6</td><td>6</td><td>6</td><td>6</td></tr><tr><td>TBPTT depth</td><td>6</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Inference and halting</td><td></td><td></td><td></td><td></td></tr><tr><td>Halting criterion</td><td>rG &lt; 0.03</td><td>rG &lt; 0.075</td><td>rG &lt; 0.15</td><td>rG &lt; 0.08</td></tr><tr><td>Persistence window W</td><td>1</td><td>1</td><td>1</td><td>2</td></tr><tr><td>Rollout cap</td><td>35,000</td><td>3,000</td><td>3,000</td><td>100</td></tr><tr><td>Evaluation</td><td>halt-or-cap</td><td>halt-or-cap</td><td>halt-or-cap</td><td>halt-or-cap + vote</td></tr></table>