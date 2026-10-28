# REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal

Shengzhang Xu<sup>1</sup>, Yuning Guo<sup>1</sup>, Weiyi Hu<sup>1</sup>, Yifeng Xu<sup>1</sup>, Zhongyu Li<sup>2</sup>, and He Kong<sup>1,∗</sup>

![](images/918a4b48806efe1a2dbdee75edbe5e7d2d4a7b86b6f7d1b6c6de73d8640d8348.jpg)  
Fig. 1. We introduce REACH, a modular, adaptable, RL-based hierarchical framework for humanoid crawling, enabling the Unitree G1 to transition smoothly between bipedal walking and crawling while traversing confined spaces.

Abstract— Humanoid locomotion in confined spaces requires continuous height adaptation across multiple locomotion modes, yet most existing approaches remain limited to heightadjustable bipedal walking. We present REACH, an RL-based hierarchical framework that enables a Unitree G1 humanoid to traverse height-constrained environments through variableheight bipedal walking and crawling. The learned policies regulate body height over complementary command ranges of 1.27–0.50 m in bipedal locomotion and 0.50–0.20 m in crawling, extending traversal beyond the clearance limits of crouched walking. To address contact-rich crawling dynamics, we employ a history-conditioned latent representation that captures terrain and interaction context from proprioceptive observations. A bidirectional transition policy further connects crouched bipedal locomotion and crawling across large kinematic and contact discontinuities. These skills are integrated with a LiDAR-based planner that converts point clouds into collision-free three-dimensional paths and locomotion commands, including proactive height targets. Real-world

experiments demonstrate accurate height and velocity tracking, robust terrain traversal, reliable mode transitions and confinedspace traversal with the proposed pipeline. Code and videos are available at https://aislab-sustech.github.io/REACH/.

## I. INTRODUCTION

Confined-space traversal is essential to extending humanoid robots from structured settings to environments with narrow passages [1], [2]. However, the existing literature remains dominated by bipedal gaits [3], with torso height either fixed or adjusted only for crouched walking. This restriction prevents existing systems from accessing severely constrained spaces. In recent years, reinforcement learning (RL) has substantially enhanced humanoid mobility and robustness [4], [5], [6] or posture adaptation under complex terrain and overhead constraints [7], [8], [9], [10]. Nevertheless, most approaches remain bipedal and are limited by leg workspace: aggressive crouching reduces maneuverability and balance margins.

However, extending a conventional humanoid from bipedal walking to another motion modality, e.g., crawling, introduces a few challenges. First, the two modes differ sub stantially in configuration–directly switching between two independently trained policies requires the robot to pass through dynamically sensitive intermediate states and can produce discontinuous actions, imbalance, or falls. Second, crawling is intrinsically contact-rich: body segments such as the palms, wrists, knees, feet interact with the environment, generating impacts and disturbances that are difficult to model, leading to severe sim-to-real gaps.

Towards a practical solution for large-range height adaptation, robust multi-contact control, smooth transitions between locomotion modes, we present REACH (see Fig. 1), an RLbased hierarchical framework for adaptable confined-space traversal using an unmodified Unitree G1 humanoid. Inspired by latent terrain-inference approaches such as DreamWaQ [11], we design a history-conditioned latent adaptation module tailored to contact-rich humanoid crawling. The module extracts terrain and interaction context from proprioceptive histories, improving robustness to multi-contact disturbances and real-world deployment conditions. A bidirectional motion-tracking policy connects crouched walking and crawling, enabling stable transitions across large kinematic and contact discontinuities. At the upper level, an adapted EGO-Planner [12] converts onboard LiDAR observations into collision-free 3D paths and proactive height commands. This hierarchical decomposition separates perception and planning from locomotion, allowing integration with other navigation methods. Our main contributions are:

• An adaptable locomotion framework that extends locomotion beyond the clearance limits of crouched walking. More specifically, the proposed framework enables variable-height bipedal walking (from 1.27 to 0.50 m) and crawling with height commands from 0.50 to 0.20 m (in experiments, traversal of clearances as low as 0.40 m).

• A history-conditioned latent adaptation mechanism and a bidirectional transition policy for contact-rich crawling and bipedal–crawling switching.

• A practical planner–controller hierarchy that interfaces learned skills with LiDAR-based planning, validated successful real-world confined-space experiments.

## II. RELATED WORK

## A. Humanoid Locomotion under Complex Terrains

Recent learning-based humanoid locomotion methods increasingly incorporate exteroceptive perception to anticipate geometric constraints, using depth images [13], [14], LiDARderived elevation maps [15], [16], [17], or RGB observations [18]. These approaches improve mobility over rough, deformable, and discontinuous terrain, while model-informed objectives further enhance stability. Xie et al. [19] introduce a Zero-Moment Point-based reward for balance recovery, whereas Dai et al. [20] incorporate Control Lyapunov Function constraints for traversing sparse footholds. Attentionbased terrain encoders additionally support accurate foothold selection [16], [17]. Related systems achieve robust blind locomotion through large-scale reinforcement learning [7] and terrain-aware traversal of stairs and gaps by combining online reconstruction with adaptive gait control [8]. However, these methods primarily assume vertically unconstrained environments and remain centered on bipedal locomotion, limiting their applicability to severely narrow passages.

## B. Humanoid Postural Adaptation for Confined Spaces

Confined-space traversal requires substantial posture adaptation under low ceilings and overhanging obstacles.

Perception-driven methods adapt humanoid posture online: HumanoidPF [9] enables crouching and obstacle negotiation, while Gallant [10] uses voxel representations and reinforcement learning under overhead constraints. Both remain predominantly bipedal, however, and aggressive crouching is limited by leg workspace, reduced maneuverability, and degraded disturbance rejection. Specialized low-profile platforms and hardware adaptations further ease confined locomotion [21], [3], but their morphological assumptions do not transfer directly to human-sized humanoids. Bridging large-range height adaptation with contact-rich crawling on a standard humanoid therefore remains an open challenge.

## C. Motion Tracking and Multi-Contact Crawling

Motion tracking and imitation learning provide a complementary route to whole-body humanoid skills. BeyondMimic [22] demonstrates versatile control by tracking richly generated motions, while generative models such as Kimodo [23] synthesize controllable whole-body references. We adopt this tracking paradigm only for bipedal–crawling mode switching: a single bidirectional tracker follows generated crawlto-squat references (and their reverse), rather than building a general skill library. Separately, low-profile crawling has long been studied with model-based gait design [24], compliant or specialized hardware [3], [25], and morphologies that favor multi-contact support [21]. In contrast, our method learns command-conditioned crawling on an unmodified Unitree G1, uses history-conditioned latent adaptation for contact-rich disturbances, and exposes a modular interface to LiDAR-based planning.

## III. METHODOLOGY

## A. RL-based Variable-Height Locomotion

As shown in Fig. 2, we employ a shared architecture to train separate variable-height bipedal and crawling policies. The architecture combines an asymmetric actor– critic formulation with a history-conditioned context estimator, adopted from [11]. Both policies share the same deployable observation and network interface, enabling command execution across locomotion modes without privileged sensing on the physical robot. The actor maps the current proprioceptive observation, estimated base velocity, and latent context to a 29-dimensional joint action $\mathbf { a } _ { t } \sim \pi _ { \psi } \left( \mathbf { a } _ { t } \mid \mathbf { o } _ { t } , { \tilde { \mathbf { v } } } _ { t } , \mathbf { z } _ { t } \right)$ , where the 99-dimensional observation is $\mathbf { o } _ { t } ~ = ~ [ \omega _ { t } , \mathbf { g } _ { t } , \mathbf { c } _ { t } , \Phi _ { t } , \pmb { \theta } _ { t } , \dot { \pmb { \theta } } _ { t } , \mathbf { a } _ { t - 1 } ] ^ { T }$ . Here, $\begin{array} { r l } { \mathbf { c } _ { t } } & { { } = } \end{array}$ $[ v _ { x } ^ { \mathrm { c m d } } , v _ { y } ^ { \mathrm { c m d } } , \omega _ { z } ^ { \mathrm { c m d } } , h ^ { \mathrm { c m d } } ]$ specifies planar velocity, yaw rate, and target height, and $\Phi _ { t } ~ = ~ [ \sin ( 2 \pi \phi _ { t } ^ { \mathrm { L F } } ) , \cos ( 2 \pi \phi _ { t } ^ { \mathrm { L F } } ) ]$ encodes the gait clock of Eq. (4); it is set to zero for the phase-free crawling policy. A context estimator processes the five-step history $\mathbf { o } _ { t } ^ { H } ~ = ~ \left[ \mathbf { o } _ { t } , \ldots , \mathbf { o } _ { t - 4 } \right] ^ { T }$ . Its encoder produces the latent context $\mathbf { z } _ { t }$ and estimated base velocity $\tilde { \mathbf { v } } _ { t } .$ while a decoder reconstructs the subsequent observation. Joint velocity-estimation and future-observation reconstruction objectives encourage $\mathbf { z } _ { t }$ to encode terrain properties, contact events and dynamics that cannot be recovered from a single observation.

![](images/c7c40352f49f9a42f58ff7cd29716439d6bd4873cfaae92d63f06fa1de1250f3.jpg)  
Fig. 2. Overview of REACH. (A) A bidirectional motion-tracking policy connects crouched bipedal locomotion and crawling. (B) The history-conditioned asymmetric actor–critic encodes proprioceptive histories, predicts future observations and body velocity, and uses privileged information only for critic training. (C) A model-based planner converts LiDAR point clouds into locomotion and height commands. (D) Real-world trials demonstrate crawling through consecutive low-clearance obstacles.

1) Bipedal Walking: For training, the critic receives $\mathbf { s } _ { t } ^ { \mathrm { w a l k } } = \bar  [ \mathbf { o } _ { t } , \mathbf { v } _ { t } , \mathbf { h } _ { t } ^ { \mathrm { s c a n } } , \mathbf { p } _ { t } ^ { \mathrm { f e e t } } , \mathbf { v } _ { t } ^ { \mathrm { f e e t } } , \mathbf { f } _ { t } ^ { \mathrm { f e e t } } , h _ { t } ^ { \mathrm { r o o t } } ] ^ { T }$ , which augments proprioception with terrain and foot-contact information available only in simulation.

2) Crawling: The crawling policy retains the same actor and context estimator, while the critic explicitly represents the four hand-and-knee contacts $\begin{array} { r l r } { \mathbf { s } _ { t } ^ { \mathrm { c r a w l } } } & { { } = } & { \left[ \mathbf { o } _ { t } , \mathbf { v } _ { t } , \mathbf { h } _ { t } ^ { \mathrm { s c a n } } , \mathbf { p } _ { t } ^ { \mathrm { h } \& \mathrm { k } } , \mathbf { v } _ { t } ^ { \mathrm { h } \& \mathrm { k } } , \mathbf { f } _ { t } ^ { \mathrm { h } \& \mathrm { k } } , h _ { t } ^ { \mathrm { r o o t } } \right] ^ { T } } \end{array}$ This 308-dimensional input exposes the critic to contact-rich crawling dynamics, while the latent estimator helps the actor compensate for impacts, uncertain contacts, and modeling errors using onboard proprioception alone. Shared network interfaces therefore provide a common learning structure, whereas mode-specific privileged observations and rewards specialize training of the two locomotion policies.

3) Reward Design: Although the walking and crawling policies share the same command-conditioned learning architecture, they solve different contact-control problems. For mode m ∈ {walk, crawl},

$$
r _ { t } ^ { m } = r _ { \mathrm { c m d } , t } ^ { m } + r _ { h , t } ^ { m } + r _ { \mathrm { p o s t } , t } ^ { m } + r _ { \mathrm { g a i t } , t } ^ { m } + r _ { \mathrm { r e g } , t } ^ { m } ,\tag{1}
$$

command tracking and regularization are shared, while height reference, posture prior, and gait shaping differ (see Appendix Tables II and III for full details of reward design). We highlight the two distinctions that most affect confinedspace behavior. Let $\bar { h } ^ { \mathrm { s c a n } } \ = \ \mathrm { m e a n } ( \mathbf { h } ^ { \mathrm { s c a n } } )$ and ${ \bf c } _ { \mathrm { v e l } } ^ { \mathrm { c m d } } = { }$ $[ \bar { v } _ { x } ^ { \mathrm { c m d } } , v _ { y } ^ { \mathrm { c m d } } , \omega _ { z } ^ { \mathrm { c m d } } ] ^ { \top }$

a) Mode-specific height regulation.: Both policies interpret $h ^ { \mathrm { c m d } }$ relative to local terrain, but track different body points:

$$
r _ { h } ^ { \mathrm { w a l k } } = \mathrm { e x p } \left[ - 2 5 \left( h ^ { \mathrm { h e a d } } - \bar { h } ^ { \mathrm { s c a n } } - h ^ { \mathrm { c m d } } \right) ^ { 2 } \right] ,\tag{2}
$$

$$
r _ { h } ^ { \mathrm { c r a w l } } = \exp \left[ - \frac { \left( h ^ { \mathrm { b a s e } } - \bar { h } ^ { \mathrm { s c a n } } - h ^ { \mathrm { c m d } } \right) ^ { 2 } } { 0 . 1 5 ^ { 2 } } \right] .\tag{3}
$$

Walking uses head height as an overhead-clearance proxy, with torso-pitch constraints (see Table II) preventing trivial satisfaction by upper-body tilt. Crawling uses base height as

a more stable clearance measure when the torso is nearly horizontal, and the head moves with arm placement.

b) Phase-indexed walking versus phase-free crawling.: Bipedal walking uses an explicit periodic contact schedule. For each foot $l \in \mathcal { L } _ { \mathrm { f e e t } }$

$$
\begin{array} { l } { \displaystyle \phi _ { t } ^ { l } = \left( \frac { t \mathrm { m o d } T } { T } + o ^ { l } \right) \mathrm { m o d } 1 , } \\ { \displaystyle T = 0 . 8 \mathrm { s } , \qquad \left( o ^ { \mathrm { L F } } , o ^ { \mathrm { R F } } \right) = ( 0 , 0 . 5 ) , } \end{array}\tag{4}
$$

with desired stance $D _ { t } ^ { l } \ = \ 1 ( \phi _ { t } ^ { l } \ < \ 0 . 5 5 )$ and measured contact $C _ { t } ^ { l } = \mathbb { 1 } ( F _ { z , t } ^ { l } > \bar { 1 } )$ ). Matching these states yields

$$
r _ { \mathrm { g a i t } } ^ { \mathrm { w a l k } } = \mathbb { 1 } \big ( \lVert \mathbf { c } _ { \mathrm { v e l } } ^ { \mathrm { c m d } } \rVert > 0 . 1 \big ) \frac { 1 } { 2 } \sum _ { l \in \mathcal { L } _ { \mathrm { f e e t } } } \mathbb { 1 } ( C _ { t } ^ { l } = D _ { t } ^ { l } ) ,\tag{5}
$$

encouraging alternating support with a duty factor close to normal walking [26]. A fixed four-limb phase clock is less suitable for crawling, where timing varies with height, terrain, and hand–knee impacts. Instead, for $\mathcal { L } _ { \mathrm { h } \& \mathrm { k } } =$ {LH, RH, LK, RK} and $C _ { t } ^ { l } = \bar { \mathbb { 1 } } \bar { ( F _ { z , t } ^ { l } \geq 2 0 ) }$ , we penalize unsafe support combinations

$$
\begin{array} { r l r } & { } & { p _ { \mathrm { p a i r } } = \mathbb { 1 } ( C ^ { \mathrm { L H } } + C ^ { \mathrm { R H } } = 0 ) + \mathbb { 1 } ( C ^ { \mathrm { L K } } + C ^ { \mathrm { R K } } = 0 ) , } \\ & { } & { p _ { \mathrm { i p s i } } = \mathbb { 1 } ( C ^ { \mathrm { L H } } + C ^ { \mathrm { L K } } = 0 ) + \mathbb { 1 } ( C ^ { \mathrm { R H } } + C ^ { \mathrm { R K } } = 0 ) , } \end{array}\tag{6}
$$

(7)

which preserve longitudinal and lateral support and favor diagonal hand–knee coordination without prescribing an exact stance sequence. Mode-duration and left–right timing terms further suppress contact chatter and asymmetric cycles:

$$
r _ { \mathrm { m o d e } } = \Im \left( \| \mathbf { c } _ { \mathrm { v e l } } ^ { \mathrm { c m d } } \| > 0 . 1 \right) \operatorname* { m i n } \left( \operatorname* { m i n } _ { l \in \mathcal { L } _ { \mathrm { h \ell { c } } \mathrm { k } } } \Delta _ { t } ^ { l } , 0 . 4 \right) ,\tag{8}
$$

$$
p _ { \mathrm { s y m } } = \frac { \mathbb { 1 } \left( \left. \mathbf { c } _ { \mathrm { v e l } } ^ { \mathrm { c m d } } \right. > 0 . 1 \right) } { 4 } \sum _ { q \in \{ \mathrm { H } , \mathrm { K } \} } \sum _ { \tau \in \{ \mathrm { l c } , \mathrm { l a } \} } \left. t _ { \tau } ^ { \mathrm { L } q } - t _ { \tau } ^ { \mathrm { R } q } \right. .\tag{9}
$$

where $\Delta _ { t } ^ { l }$ is the elapsed duration of the active contact or aerial mode of limb l, and $t _ { \mathrm { l c } } ^ { l } , \ t _ { \mathrm { l a } } ^ { l }$ are the most recently completed contact and aerial durations. Together these terms define a phase-free crawling objective—stable support and balanced timing without a global gait clock—in contrast to the phase-indexed walking reward.

4) Training Settings: The two modes share the planarvelocity range but use different yaw-rate and height ranges:

$$
\begin{array} { r l } { \mathcal { C } _ { \mathrm { w a l k } } : } & { v _ { x } ^ { \mathrm { c m d } } \in [ - 0 . 6 , 0 . 8 ] , \quad v _ { y } ^ { \mathrm { c m d } } \in [ - 0 . 6 , 0 . 6 ] , } \\ & { \omega _ { z } ^ { \mathrm { c m d } } \in [ - 1 . 0 , 1 . 0 ] , \quad h ^ { \mathrm { c m d } } \in [ 0 . 5 0 , 1 . 2 7 ] , } \end{array}
$$

$$
\begin{array} { r l } { \mathcal { C } _ { \mathrm { c r a w l } } : } & { v _ { x } ^ { \mathrm { c m d } } \in [ - 0 . 6 , 0 . 8 ] , \quad v _ { y } ^ { \mathrm { c m d } } \in [ - 0 . 6 , 0 . 6 ] , } \\ & { \omega _ { z } ^ { \mathrm { c m d } } \in [ - 0 . 7 5 , 0 . 7 5 ] , \quad h ^ { \mathrm { c m d } } \in [ 0 . 2 0 , 0 . 5 0 ] . } \end{array}
$$

Linear velocity, yaw rate, and height are expressed in $\mathrm { m } \mathrm { s } ^ { - 1 }$ rad $\mathrm { s } ^ { - 1 }$ , and meters, respectively. The height commands $h ^ { \mathrm { c m d } } \in [ 0 . 5 0 , 1 . 2 7 ]$ m and $h ^ { \mathrm { c m d } } \in [ 0 . 2 0 , 0 . 5 0 ]$ m are used only during policy training. They are terrain-relative targets for head height in walking and base height in crawling, and do not represent the robot’s physical height. For both modes, velocity commands are resampled every 5–10 s and height commands every 2–10 s. Both policies use the same terrain curriculum over box arrays, random rough ground, undulating terrain, slopes, and platforms, with difficulty increasing gradually. The termination criteria remain mode-specific. Walking terminates upon torso–terrain contact $( F _ { z } ^ { \mathrm { t o r s o } } > 1 )$ , whereas crawling terminates when the torso crosses the rollover criterion $( g _ { x } ^ { \mathrm { t o r s o } } < 0 )$ . Combined with the termination penalty, these criteria encourage fall avoidance in walking and stable four-contact support in crawling.

## B. Bidirectional Bipedal–Crawling Transition

The bipedal and crawling policies are trained independently and therefore operate around different posture and contact distributions. Direct switching is unreliable because neither policy is trained to recover from the intermediate states between a deep squat and four-limb support. We connect the two skills using a generated reference motion and a single bidirectional tracking policy, as illustrated in Fig. 2(A). This decomposition makes the framework extensible: a new locomotion skill can be connected by generating a transition between compatible endpoint postures and training a tracker, without modifying the existing low-level policies.

1) Reference Motion Generation: We use Kimodo [23] to generate a coherent transition from quadrupedal crawling to a deep squat. Compared with collecting and retargeting motion-capture data, motion generation provides direct control over the terminal postures, allowing them to match the boundary states of the learned locomotion policies. In our implementation, the crawling endpoint corresponds to the stationary crawling policy at its maximum commanded height of 0.5 m, whereas the squat endpoint matches the stationary bipedal policy at its minimum height of 0.5 m.

The generated motion lasts 5 s at 30 Hz. Whole-body keyframes guide the sequence from the initial crawl, through a compact posture with the legs advanced beneath the body, to a single-foot–single-knee support state with both hands lifted, and finally to a stabilized deep squat, as shown in Fig. 2(A). The text prompt specifies the same progression: sequential leg advancement, arm retraction, and successive knee lift. These constraints preserve the intended contact order while leaving the generator free to synthesize the intervening whole-body motion. We generate the dynamically harder crawl-to-squat direction; the reverse sequence is used only as a tracking reference, while the physics-based policy learns a feasible closed-loop realization of both directions.

2) Unified Transition Motion Tracking: Confined-space traversal requires both crawl-to-squat and squat-to-crawl switching; because contact changes and balance constraints are direction-dependent, a one-way tracker is insufficient. We therefore train a single transition controller, extending BeyondMimic [22], to realize both directions; bidirectionality therefore refers to this shared policy, not to kinematic time reversal alone. In practice, half of the parallel environments track the fixed reference in crawl-to-squat order and the remainder track its reverse, with the playback direction resampled every 2–6 s so that one policy covers both transitions and their intermediate states. We also extend the observation history from one to three steps, providing sufficient temporal context to distinguish rising toward the squat from descending toward four-limb support (Fig. 2(A)).

To smooth switching transients, the supervisor first brings the robot to the shared 0.5 m endpoint under the active policy, then runs the closed-loop tracker along the reference, and finally transfers control to the target policy, rather than swapping policies through unsupported intermediate states. State and dynamics randomization, with permissive termination, further trains the tracker to absorb handoff mismatch and external disturbances. Under strong randomization, the standard tracking objective occasionally admits a kneeling local optimum near the squat endpoint. Strict termination can remove this behavior but substantially reduces useful training data. We instead introduce phase-gated shaping terms, summarized in Appendix Table IV. Let $t _ { c }$ denote the current reference-frame index. During the rising phase, headand knee-height rewards promote completion of the deep squat, foot-level and non-foot-contact penalties stabilize the final support configuration, and hand-level shaping maintains reliable palm support before lift-off. These terms target the ambiguous contact changes without constraining the entire trajectory, preserving the diversity and robustness of the original motion-tracking objective.

## IV. PERCEPTION AND PLANNING

Figure 2(C) summarizes the hierarchical navigation framework. The model-based front end converts LiDAR measurements into a collision-free three-dimensional body path, while a lightweight tracking interface converts this path into the velocity and height commands accepted by the learned locomotion policies. A locomotion supervisor then dispatches these commands to the bipedal or crawling policy and invokes the bidirectional transition policy when the required mode changes.

## A. Spatiotemporal Point-Cloud Representation

Each LiDAR scan is first processed by Elevation Mapping [27] to estimate the local ground surface and separate obstacle returns from traversable terrain. A single scan, however, is incomplete because of the sensor’s limited field of view and self-occlusion by the robot. We therefore transform the current obstacle points into the world frame using the pose estimated by Point-LIO [28] and fuse them with its accumulated static map. The resulting obstacle cloud combines current measurements with previously observed geometry, retaining overhanging structures that cannot be represented by a ground-height map alone. This temporally persistent representation is supplied to the front-end $\mathbf { A } ^ { * }$ search and the subsequent path optimizer.

![](images/2f61ca5da2dfba37e45636145d1afbb2ef8afd2619474920b652baac80e0ec01.jpg)  
Fig. 3. Real-world evaluation of the complementary locomotion capabilities enabled by REACH. (A) Variable-height bipedal walking continuously adjusts body height to traverse an 80 cm-clearance obstacle. (B) Quadrupedal crawling enables robust multi-contact traversal through passages as low as 40 cm. (C) Bidirectional transitions between bipedal standing and quadrupedal crawling enable the robot to enter and exit low-clearance obstacles under different locomotion modes.

## B. Humanoid-Constrained Spatial Planning

We adapt EGO-Planner [12], originally developed for aerial robots, to produce geometric paths compatible with variable-height humanoid locomotion. Let $\textbf { Q } = \{ \mathbf { q } _ { i } \} _ { i = 1 } ^ { N _ { c } }$ denote the path control points, where $\mathbf { q } _ { i } ~ = ~ [ x _ { i } , y _ { i } , z _ { i } ] ^ { \top }$ The original optimization $J _ { \mathrm { o r i g } } ( \mathbf { Q } ) = \lambda _ { s } J _ { s } + \lambda _ { c } J _ { c } + \lambda _ { d } J _ { d }$ balances smoothness, collision avoidance, and dynamic feasibility. Because timing, velocity tracking, and dynamic balance are handled by the low-level policies, our planner optimizes a static spatial path rather than a dynamically feasible aerial trajectory. We therefore remove $J _ { d }$ and introduce two humanoid-specific geometric terms: $J _ { \mathrm { o u r s } } ( \mathbf { Q } ) =$ $\lambda _ { s } J _ { s } + \lambda _ { c } J _ { c } + \lambda _ { h } J _ { h } + \lambda _ { o t } J _ { o t }$

a) Terrain-relative height regularization.: The first term regulates the body height along the path while allowing vertical adaptation when it is preferable to a long horizontal detour. Let $\hat { z } _ { g } ( x _ { i } , y _ { i } )$ be the estimated ground elevation beneath $\mathbf { q } _ { i } ,$ , and define the planned terrain-relative height as $h _ { i } \ = \ z _ { i } - \hat { z } _ { g } ( x _ { i } , y _ { i } )$ . We use $\begin{array} { r } { J _ { h } \ = \ \sum _ { i = 1 } ^ { N _ { c } } \left( h _ { i } - h ^ { \mathrm { n o m } } \right) ^ { 2 } } \end{array}$ where $h ^ { \mathrm { n o m } }$ is the nominal walking height. In open space, this term suppresses unnecessary vertical oscillations. Near an overhead obstacle, its finite weight allows the optimizer to lower the path when the resulting height change is less costly than a lateral bypass. Importantly, expressing the cost relative to the local ground makes the planned height consistent with the height commands used to train both low-level policies.

b) Overhead traversal suppression.: An aerial planner may pass above an obstacle whenever doing so shortens the path, even when the obstacle exceeds the humanoid’s stepping capability. To suppress such fly-over solutions, we penalize path points above the obstacle footprint via $\begin{array} { r } { J _ { o t } = \sum _ { i \in \Omega _ { \mathrm { t o n } } } \exp \left( \kappa \left[ z _ { i } - z _ { \mathrm { o b s } } ( x _ { i } , y _ { i } ) \right] \right) } \end{array}$ , where $\Omega _ { \mathrm { t o p } }$ contains control points whose horizontal coordinates lie within an obstacle footprint, $z _ { \mathrm { o b s } } ( x _ { i } , y _ { i } )$ is the obstacle’s top elevation, and $\kappa > 0$ controls the growth rate. The cost increases rapidly above the obstacle and thus acts as a soft vertical extension of its footprint. Together with $J _ { c } ,$ it directs the optimizer around solid obstacles while preserving collisionfree underpasses beneath elevated structures. The two added terms therefore distinguish feasible humanoid height adaptation from the unrestricted vertical motion assumed by the original aerial planner.

## C. 3D Path-to-Policy Interface

The optimized path P is converted into low-dimensional commands by a three-dimensional Pure Pursuit tracker running at 30 Hz. At each update, a lookahead sphere of radius $L _ { d } ,$ centered at the robot, selects a forward target on ${ \mathcal { P } } .$ After transformation into the robot base frame, this target is written as ${ } ^ { B } \mathbf { p } _ { L } = [ x _ { L } , y _ { L } , z _ { L } ] ^ { \top }$ . The horizontal components determine the commanded planar velocity, and the local pathheading error determines the yaw-rate command; velocitycoupled lateral feedback compensates for drift during path tracking. The vertical component is converted to a terrainrelative body-height command, providing advance notice of an approaching overhead constraint because the target lies ahead of the robot rather than at its current position. The resulting interface is $\mathbf { c } _ { t } = \left[ v _ { x , t } ^ { \mathrm { c m d } } \quad v _ { y , t } ^ { \mathrm { c m d } } \quad \omega _ { z , t } ^ { \mathrm { c m d } } \quad h _ { t } ^ { \mathrm { c m d } } \right] ^ { | }$ which exactly matches the command parameterization used during low-level policy training. The planned height also coordinates locomotion-mode selection. The bipedal and crawling policies meet at the shared switching height $h ^ { \mathrm { s w } } =$ 0.5 m: walking realizes commands above this boundary, whereas crawling realizes lower-clearance commands. When the requested height crosses the boundary, the supervisor first brings the active policy to the corresponding endpoint posture, executes the bidirectional transition policy, and then transfers control to the target locomotion policy. Commands are restricted to the training range of the active policy, preventing the planner from requesting heights or velocities outside its learned operating domain. Thus, a single geometric path jointly specifies where the robot should move, how its body height should change, and when a locomotion-mode transition is required.

TABLE I  
ABLATION VARIANTS FOR DIFFERENT LOCOMOTION MODES.
<table><tr><td>Variant</td><td>Hist. estimator Priv. scan</td><td></td><td>Idle penalty</td></tr><tr><td>FULL</td><td>√</td><td></td><td> $\checkmark$ </td></tr><tr><td>NO-IDLE</td><td>√</td><td>一</td><td></td></tr><tr><td>PROPRIO</td><td>一</td><td>一</td><td>√</td></tr><tr><td>PRIV-SCAN</td><td>一</td><td>V</td><td>√</td></tr></table>

## V. EXPERIMENTS AND DISCUSSIONS

We organize the evaluation around three research questions: RQ1, how do the low-level ablations affect curriculum progression, which is an indicator of training progress; RQ2, how accurately do the resulting policies jointly track velocity and height commands; and RQ3, which training components enable robust bidirectional transitions and sim-to-real transfer? The first two questions evaluate variable-height walking and crawling in simulation. The third separates transition-policy ablations in simulation from deployment of the full policy on hardware. Complementing these controlled studies, Fig. 3 further shows real-world composability: 80 cm-clearance walking, crawling as low as 40 cm, and composable skills in a single run that combines crouched walking and crawling via bidirectional transitions (Fig. 3(C)).

## A. Variable-Height Locomotion

a) Ablation design.: For each locomotion mode, we train four policy variants in Isaac Lab [29] using 8192 parallel environments and 4000 training iterations. All variants receive current proprioception; Table I summarizes their differences. FULL is the proposed deployable architecture, in which the history-conditioned estimator provides the actor with an estimated body velocity $\tilde { \mathbf { v } } _ { t }$ and latent context $\mathbf { z } _ { t }$ . NO-IDLE removes the linear- and angular-command idle penalties from FULL. PROPRIO removes both estimated quantities and uses only current proprioception. Finally, PRIV-SCAN augments current proprioception with privileged ground-truth terrain heights $\mathbf { h } ^ { \mathrm { s c a n . } }$ ; it is a simulation-only reference.

1) RQ1: Curriculum Progression: Terrain difficulty in creases only when the robot traverses a prescribed distance; its progression therefore measures how quickly a policy acquires command-responsive locomotion on increasingly complex terrain. Figure 4(a,b) reports the curriculum level throughout training. For walking, FULL is the first variant to exceed terrain level 4, doing so at approximately 1700 iterations. PROPRIO and PRIV-SCAN eventually reach similar levels but require substantially more training, whereas NO-IDLE remains below level 3 at the 4000-iteration budget. The crawling task provides a clearer separation: at iteration 4000, FULL reaches approximately level 3.6 and NO-IDLE reaches approximately level 3.3 after a pronounced delay, while PROPRIO and PRIV-SCAN do not progress beyond the initial level. These comparisons isolate two effects. The gap between FULL and NO-IDLE shows that idle penalties accelerate the acquisition of command-responsive motion. The failure of PROPRIO and PRIV-SCAN in crawling further indicates that explicit terrain heights alone do not replace the temporal context and body-velocity estimate required for contact-rich locomotion.

2) RQ2: Joint Command Tracking: We evaluate each trained policy in 2048 parallel environments. A command is sampled from the corresponding training range at the beginning of each rollout and held fixed. Terrain levels are set to 4 for walking and 2 for crawling. After a 1-s warm-up, we record 9 s of absolute tracking error, $e _ { q } ( t ) =$ $| q ( t ) - q ^ { \mathrm { c m d } } | , \qquad q ~ \in ~ \{ v _ { x } , v _ { y } , \omega _ { z } , h \}$ , where h denotes the mode-specific terrain-relative height used by the corresponding low-level policy. Figure 4 jointly reports curriculum progression and the resulting tracking-error distributions. In walking, FULL reduces all three velocity errors relative to NO-IDLE. Although PROPRIO and PRIV-SCAN exhibit smaller velocity errors, their height errors are several times larger. Rollout inspection shows that these variants often remain near an upright posture instead of realizing the commanded crouched gait; their apparently favorable velocity errors therefore do not represent successful variable-height walking. FULL provides the best joint tracking of velocity and height. In crawling, FULL has forward-velocity error comparable to NO-IDLE, while reducing lateral, yaw-rate, and height errors. PROPRIO and PRIV-SCAN show smaller yaw-rate errors, but this must be interpreted together with Fig. 4(b): neither variant advances beyond the initial crawling terrain, and both have much larger translational errors. Their low yaw-rate errors therefore do not compensate for the absence of robust translational crawling. Across both evaluations, FULL is the only deployable variant that combines rapid curriculum progression with balanced tracking of all four commands.

## B. RQ3: Bidirectional Transition Robustness

a) Ablations and protocol.: We train three transition variants in Isaac Lab using 8192 environments and 4000 iterations. FULL (SIM.) uses the complete motion-tracking objective, phase-gated shaping terms, dynamics randomization, and permissive termination described in the methodology. NO SHAPING removes the additional phase-gated terms and retains only the BeyondMimic objective. REDUCED ROBUSTNESS halves all randomization magnitudes except friction and halves all termination thresholds except the rollover threshold, thereby combining weaker randomization with stricter termination. The three policies are evaluated in MuJoCo [30] for 25 trials in each direction. Every trial begins at the shared 0.5-m boundary posture. A squatto-crawl trial is successful if the robot establishes stable hand–knee support without torso collision and subsequently moves under the crawling policy. Conversely, a crawl-tosquat trial must recover stable feet-only support without torso collision and subsequently walk under the bipedal policy. We additionally deploy the unchanged full policy on the physical Unitree G1 using the same initial states, criteria, and number of trials. This FULL (REAL) result is a hardware deployment of FULL (SIM.), not a fourth training configuration.

![](images/0d783e32f22462416a047088964fe0297f389a135c0df7bc3e08edeb1caf7c86.jpg)

Fig. 4. Low-level locomotion ablations. (a,b) Mean terrain-curriculum level over training iterations. (c,d) Absolute command-tracking errors (median, IQR, 10th–90th percentiles; descriptive, not confidence intervals). †: crawling variants with no curriculum progress—low yaw-rate error should not be read as successful locomotion.  
![](images/098fe9b92e5075627be66aaef9f1fce04ec4c92f1dab42929e52b05e150444f0.jpg)  
Fig. 5. Bidirectional transition success over 25 trials per direction. The first three configurations are evaluated in MuJoCo; the hatched FULL (REAL) bars denote deployment of the unchanged full policy on the Unitree G1.

b) Results.: As shown in Fig. 5, FULL (SIM.) succeeds in $2 5 / 2 5$ squat-to-crawl trials and 24/25 crawl-to-squat trials in MuJoCo, corresponding to success rates of 100% and 96%. NO SHAPING retains $2 3 / 2 5$ squat-to-crawl successes but yields $0 / 2 5$ crawl-to-squat successes. This asymmetric failure supports the intended role of the shaping terms: they are specifically necessary for escaping the kneeling optimum during crawl-to-squat recovery, rather than merely improving generic tracking accuracy. REDUCED ROBUSTNESS succeeds in only $1 5 / 2 5$ and 17/25 trials, showing that broad randomization and permissive termination are important for recovery from switching-state variation. On hardware, FULL (REAL) achieves 25/25 squat-to-crawl and 23/25 crawlto-squat successes, or 100% and 92%, respectively. The small decrease from simulation in the harder rising direction demonstrates that the learned bidirectional transition transfers to the physical robot while preserving reliable handover to both locomotion policies. Together with the qualitative real-world sequences in Fig. 3, these results indicate that the full REACH stack, including variable-height walking, contact-rich crawling, and mode switching, supports robust confined-space traversal on hardware.

## VI. CONCLUSIONS

We presented REACH, an RL-based hierarchical framework that enables an unmodified Unitree G1 to traverse confined spaces through variable-height bipedal walking, crawling, bidirectional mode transitions, and a modular LiDAR planner–controller interface. Ablation studies show that history-conditioned latent adaptation improves curriculum progression and joint command tracking, particularly for contact-rich crawling. The bidirectional transition policy transfers reliably from simulation to hardware, and real-world demos demonstrate complementary walking and crawling under low clearances with planner-issued velocity and height commands.

## APPENDIX

## (A) Reinforcement Learning Reward Summary

The signed coefficients in Tables II and III are applied directly to the listed nonnegative rewards or penalties. For compactness, define $\begin{array} { r } { \rho _ { \mathcal { T } } ( \pmb { \theta } ^ { \mathrm { r e f } } ) = \sum _ { i \in \mathcal { I } } | \theta _ { i } - \theta _ { i } ^ { \mathrm { r e f } } | } \end{array}$ and $\beta ( \theta ) =$ $\frac { 1 } { 2 } \left[ e ^ { - 1 0 0 | \operatorname* { m i n } ( \theta , 0 ) | } + e ^ { - 1 0 0 | \operatorname* { m a x } ( \theta - 1 . 2 , 0 ) | } \right]$ . Here, $\pmb { \theta } ^ { \mathrm { s t a n d } }$ and $\bar { \theta } ^ { \mathrm { c r a w l } }$ are the standing and crawling reference configurations, respectively; ${ \bf { g } } _ { x y }$ is the horizontal component of projected gravity; and $\tau _ { i } ^ { \mathrm { P D } }$ and $\tau _ { i }$ are the PD-requested and postsaturation torques.

TABLE II  
REWARD TERMS FOR VARIABLE-HEIGHT BIPEDAL WALKING.
<table><tr><td>Term</td><td>Definition</td><td>Weight</td></tr><tr><td>Head-height tracking Eq. (2)</td><td></td><td>2.0</td></tr><tr><td>Yaw-rate tracking</td><td> $\begin{array} { r l r } { \bar { \omega } _ { z , t } } & { { } = } & { \frac { 1 } { 5 } \sum _ { k = 0 } ^ { 4 } \omega _ { z , t - k } ; \quad r _ { \omega } \quad = \quad 2 . 0 } \end{array}$   $\exp [ - ( \bar { \omega } _ { z , t } - \dot { \omega } _ { z } ^ { \mathrm { c m i d } } ) ^ { \tilde { 2 } } / 0 . 3 ^ { 2 } ]$ </td><td></td></tr><tr><td>Linear idle</td><td> $\mathbb { 1 } ( \| \mathbf { v } _ { x y } ^ { \mathrm { c m d } } \| > 0 . 2 \land \| \mathbf { v } _ { x y } \| < 0 . 2 )$ </td><td>-3.0</td></tr><tr><td>Yaw idle</td><td> $\mathbb { 1 } [ ( \omega _ { z } ^ { \mathrm { c m d } } > 0 . 2 \land \omega _ { z } < 0 . 1 ) \lor ( \omega _ { z } ^ { \mathrm { c m d } } <$   $- 0 . 2 \wedge \omega _ { z } > - 0 . 1 ) ]$ </td><td>-2.0</td></tr><tr><td>Arm posture</td><td> $\rho _ { \mathcal { T } _ { \mathrm { a r m } } } ( \theta ^ { \mathrm { s t a n d } } )$ </td><td>-0.2</td></tr><tr><td>Waist posture</td><td> $\rho \mathcal { T } _ { \mathrm { w a i s t } } \left( \theta ^ { \mathrm { s t a n d } } \right)$ </td><td>-1.0</td></tr><tr><td>Hip-roll posture</td><td> $\mathbb { 1 } ( | v _ { y } ^ { \mathrm { c m d } } | \leq 0 . 2 ) | \theta _ { \mathrm { h i p r o l l } } - \theta _ { \mathrm { h i p r o l l } } ^ { \mathrm { s t a n d } } |$ </td><td>-0.3</td></tr><tr><td>Torso-pitch range</td><td> $\beta ( \theta _ { \mathrm { t o r s o } } ^ { \mathrm { p i t c h } } )$ </td><td>1.0</td></tr><tr><td>High-stance torso tilt</td><td> $\mathbb { 1 } ( h ^ { \operatorname { \scriptsize { c m d } } } > 1 . 0 ) \Vert \mathbf { g } _ { x y } \Vert ^ { 2 }$ </td><td>-3.0</td></tr><tr><td>Phase-matched gait</td><td> $\operatorname { E q . } \ ( 5 )$ </td><td>1.0</td></tr></table>

## (B) Transition-Policy Reward Terms

Table IV lists the additional terms applied on top of the BeyondMimic objective. Define the filtered contact indicator $\begin{array} { r } { \chi _ { l } ( t ) = \mathbb { 1 } \left( \operatorname* { m a x } _ { k \in \{ 0 , \ldots , 4 \} } \| \mathbf { f } _ { t - k , l } \| _ { 2 } > 1 . 0 \right) } \end{array}$ , where $\mathbf { f } _ { t , l }$ is the contact force on link l. We use L for all links except the ankles, wrists, and knees, and $\mathcal { L } _ { \mathrm { { n o n f e e t } } }$ for all non-foot links.

TABLE III  
REWARD TERMS FOR VARIABLE-HEIGHT HUMANOID CRAWLING.
<table><tr><td>Term</td><td>Definition</td><td>Weight</td></tr><tr><td>Base-height tracking</td><td> $\operatorname { E q . } \ ( 3 )$ </td><td>3.0</td></tr><tr><td>Waist-pitch posture</td><td> $| \theta _ { \mathrm { w a i s t p i t c h } } - \theta _ { \mathrm { w a i s t p i t c h } } ^ { \mathrm { c r a w l } } |$ </td><td>-1.2</td></tr><tr><td>Wrist posture</td><td> $\rho \mathcal { T } _ { \mathrm { w r i s t } } \mathopen { } \mathclose \bgroup \left( \theta ^ { \mathrm { c r a w l } } \aftergroup \egroup \right)$ </td><td>-0.2</td></tr><tr><td>Waist-roll posture</td><td> $\begin{array} { r l r } { \mathbb { 1 } ( | \stackrel {  \mathrm {  ~ } } { v } _ { x } ^ { \mathrm { c m d } } | < } & { { } } & { 0 . 3  \ | \omega _ { z } ^ { \mathrm { c m d } } \| } \end{array}$   $0 . 4 ) | \theta _ { \mathrm { w a i s t r o l l } }$   $\theta _ { \mathrm { w a i s t r o l l } } ^ { \mathrm { c r a w l } } |$ </td><td> $< \mathrm { ~ - 0 . 6 ~ }$ </td></tr><tr><td>Hand/knee support</td><td>Eq. (6)</td><td>-0.6</td></tr><tr><td>Ipsilateral support</td><td>Eq. (7)</td><td>-0.3</td></tr><tr><td>Mode duration</td><td>Eq. (8)</td><td>1.0</td></tr><tr><td>Timing symmetry</td><td>Eq. (9)</td><td>-0.75</td></tr><tr><td>Torque saturation</td><td> $\begin{array} { r } { \dot { \sum _ { i \in \mathcal { I } _ { \mathrm { a l l } } } } | \tau _ { i } - \tau _ { i } ^ { \mathrm { P D } } | } \end{array}$ </td><td>-0.1</td></tr></table>

TABLE IV

ADDITIONAL REWARD TERMS FOR BIDIRECTIONAL MOTION TRACKING.
<table><tr><td>Term</td><td>Definition</td><td>Weight</td></tr><tr><td>Head-height tracking</td><td> $\begin{array} { r c l } { { \Im } { \left( 1 1 0 \right. } } & { { \le } } & { { t _ { c } } } & { { \le } } & { { \left. 1 4 9 \right) } \exp [ - ( h ^ { \mathrm { h e a d } } \mathrm { ~ - } \quad } \end{array}$   $0 . { \dot { 5 } } ) ^ { 2 } / 0 . 3 ^ { 2 } ]$ </td><td>0.5</td></tr><tr><td>Knee-height tracking</td><td> $\begin{array} { r } { \Im ( 1 1 0 \ \le \ \dot { t } _ { c } \ \le \ 1 4 9 ) \exp [ - 2 5 ( h ^ { \mathrm { k n e e s } } \ - \ } \end{array}$  0.25)2]</td><td>1.0</td></tr><tr><td>Foot-level shaping</td><td> $1 1 ( 9 0 \leq t _ { c } \leq 1 4 9 ) \| \mathbf { g } _ { x y } ^ { \mathrm { f e e t } } \|$ </td><td>-0.4</td></tr><tr><td>Hand-level shaping</td><td> $\begin{array} { r l r l r } { \mathbb { 1 } ( 0 } & { { } \le } & { t _ { c } } & { { } \le } & { 9 5 ) ( \sqrt { 1 - g _ { y } ^ { \mathrm { R H } } } } \end{array}$ </td><td>十 -0.6</td></tr><tr><td></td><td> $\sqrt { 1 + g _ { y } ^ { \mathrm { L H } } } )$ </td><td></td></tr><tr><td>Undesired contacts Squat-phase contacts</td><td> $\dot { \sum } _ { l \in \mathcal { L } } \chi _ { l } ( t )$   $\begin{array} { r } { \mathbb { 1 } ( 9 0 \leq t _ { c } \leq 1 4 9 ) \sum _ { l \in \mathcal { L } _ { \mathrm { n o n f e e t } } } \chi _ { l } ( t ) } \end{array}$ </td><td>-0.3 -0.5</td></tr></table>

## REFERENCES

[1] Z. Gu, J. Li, W. Shen, W. Yu, Z. Xie, S. McCrory, X. Cheng, A. Shamsah, R. Griffin, C. K. Liu, A. Kheddar, X. B. Peng, Y. Zhu, G. Shi, Q. Nguyen, G. Cheng, H. Gao, and Y. Zhao, “Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning,” IEEE/ASME Transactions on Mechatronics, vol. 31, no. 2, pp. 2300–2330, 2026.

[2] T. Miki, J. Lee, L. Wellhausen, and M. Hutter, “Learning to walk in confined spaces using 3d representation,” in IEEE Int. Conference on Robotics and Automation (ICRA), 2024, pp. 8649–8656.

[3] Z. Huang, X. Jiang, H. Liu, X. Chen, T. Fukuda, and Q. Huang, “Design of crawling motion for a biped walking humanoid with 3-dof rigid-flexible waist,” in 2018 IEEE-RAS 18th International Conference on Humanoid Robots (Humanoids), 2018, pp. 974–979.

[4] Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control,” The International Journal of Robotics Research, vol. 44, no. 5, pp. 840–888, Apr. 2025.

[5] Z. Li, X. Cheng, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Reinforcement learning for robust parameterized locomotion control of bipedal robots,” in IEEE Int. Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 2811–2817.

[6] X. Ren, Y. Yang, Z. Weng, Z. Liu, and H. Kong, “Stubborn: A streamlined and unified reinforcement learning framework for robust motion tracking and fall recovery for humanoids,” arXiv:2606.12814, 2026.

[7] I. Radosavovic, S. Kamat, T. Darrell, and J. Malik, “Learning humanoid locomotion over challenging terrain,” arXiv preprint arXiv:2410.03654, 2024.

[8] H. Song, H. Zhu, T. Yu, Y. Liu, M. Yuan, W. Zhou, H. Chen, and H. Li, “Gait-adaptive perceptive humanoid locomotion with real-time under-base terrain reconstruction,” IEEE Robotics and Automation Letters, 2025. [Online]. Available: https://arxiv.org/abs/2512.07464

[9] H. Xue, S. Liang, Z. Zhang, Z. Zeng, Y. Liu, Y. Lian, J. Wang, Q. Liu, X. Shi, and L. Yi, “Collision-free humanoid traversal in cluttered indoor scenes,” 2026. [Online]. Available: https://arxiv.org/abs/2601.16035

[10] Q. Ben, B. Xu, K. Li, F. Jia, W. Zhang, J. Wang, J. Wang, D. Lin, and J. Pang, “Gallant: Voxel grid-based humanoid locomotion and localnavigation across 3d constrained terrains,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[11] I. M. A. Nahrendra, B. Yu, and H. Myung, “Dreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning,” in IEEE Int. Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 5078–5084.

[12] X. Zhou, Z. Wang, H. Ye, C. Xu, and F. Gao, “EGO-Planner: An ESDF-free gradient-based local planner for quadrotors,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 478–485, 2021.

[13] S. Zhu, Z. Zhuang, M. Zhao, K.-Y. Lee, and H. Zhao, “Hiking in the wild: A scalable perceptive parkour framework for humanoids,” 2026.

[14] Y. Zhang, Y. Seo, J. Chen, Y. Yuan, K. Sreenath, P. Abbeel, C. Sferrazza, K. Liu, R. Duan, and G. Shi, “Rpl: Learning robust humanoid perceptive locomotion on challenging terrains,” arXiv:2602.03002.

[15] J. Long, J. Ren, M. Shi, Z. Wang, T. Huang, P. Luo, and J. Pang, “Learning humanoid locomotion with perceptive internal model,” in IEEE Int. Conference on Robotics and Automation (ICRA), 2025.

[16] J. He, C. Zhang, F. Jenelten, R. Grandia, M. Bacher, and M. Hutter, ¨ “Attention-based map encoding for learning generalized legged locomotion,” Science Robotics, 2025.

[17] C. Zhang, V. Klemm, F. Yang, and M. Hutter, “AME-2: Agile and generalized legged locomotion via attention-based neural map encoding,” 2026. [Online]. Available: https://arxiv.org/abs/2601.08485

[18] Y. Liu, X. Chen, H. Pan, C. Shi, Y. Chen, K. Huang, Z. Zeng, and H. Lu, “Geoloco: Leveraging 3d geometric priors from visual foundation model for robust rgb-only humanoid locomotion,” arXiv:2603.07624, 2026.

[19] W. Xie, C. Bai, J. Shi, J. Yang, Y. Ge, and W. Zhang, “Humanoid whole-body locomotion on narrow terrain via dynamic balance and reinforcement learning,” in IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS). IEEE, 2025.

[20] M. Dai, W. D. Compton, J. Li, L. Yang, and A. D. Ames, “Walk the planc: Physics-guided rl for agile humanoid locomotion on constrained footholds,” in IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS). IEEE, 2026.

[21] A. Schperberg, Y. Tanaka, S. Di Cairano, and D. Hong, “Mobius: A multi-modal bipedal robot that can walk, crawl, climb, and roll,” in Robotics: Science and Systems (RSS), 2026.

[22] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu, “BeyondMimic: From motion tracking to versatile humanoid control via guided diffusion,” arXiv:2508.08241, 2025.

[23] D. Rempe, M. Petrovich, Y. Yuan, H. Zhang, X. B. Peng, Y. Jiang, T. Wang, U. Iqbal, D. Minor, M. de Ruyter, J. Li, C. Tessler, E. Lim, E. Jeong, S. Wu, E. Hassani, M. Huang, J.-B. Yu, C. Chung, L. Song, O. Dionne, J. Kautz, S. Yuen, and S. Fidler, “Kimodo: Scaling controllable human motion generation,” arXiv:2603.15546, 2026.

[24] G. Brooks, P. Krishnamurthy, and F. Khorrami, “Low-profile crawling for humanoid motion in tight spaces,” in IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS), 2015.

[25] J. Ma, Y. Dong, W. Li, and P. Lu, “MASH: MAstering safe and high-speed quadruped navigation with adaptive gait transitions,” IEEE Transactions on Field Robotics, June 2026.

[26] D. A. Winter, “Kinematic and kinetic patterns in human gait: Variability and compensating effects,” Human Movement Science, vol. 3, no. 1–2, pp. 51–76, 1984.

[27] P. Fankhauser, M. Bloesch, and M. Hutter, “Probabilistic terrain mapping for mobile robots with uncertain localization,” IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 3019–3026, 2018.

[28] C. Zheng et al., “Point-LIO: Robust high-bandwidth light detection and ranging inertial odometry,” IEEE Transactions on Robotics, 2022.

[29] M. Mittal, P. Roth, J. Tigue, A. Richard et al., “Isaac lab: A gpuaccelerated simulation framework for multi-modal robot learning,” arXiv preprint arXiv:2511.04831, 2025.

[30] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in IEEE/RSJ Int. Conference on Intelligent Robots and Systems (IROS), 2012, pp. 5026–5033.