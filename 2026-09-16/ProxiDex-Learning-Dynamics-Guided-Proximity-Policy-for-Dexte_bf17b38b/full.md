# ProxiDex: Learning Dynamics-Guided Proximity Policy for Dexterous Manipulation

Yushan Bai<sup>1,2,∗</sup> Boyu Zheng<sup>1,2,∗</sup> Zhiyang Mao<sup>4</sup> Hongzheng Sun<sup>1,2</sup> Yuchuang Tong<sup>1,2,†</sup> En Li<sup>1,2,3</sup> Zhengtao Zhang<sup>1,2,3,†</sup>

<sup>1</sup>CAS Engineering Laboratory for Intelligent Industrial Vision, Institute of Automation, Chinese Academy of Sciences <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences <sup>3</sup>Beijing Zhongke Huiling Robot Technology Co., LTD. <sup>4</sup>School of Intelligent Science and Technology, Xinjiang University <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

![](images/a78be7c341df9ddd577f89fca0980964b93da97d7ed88ea3dcd597160452901d.jpg)  
Figure 1: We propose ProxiDex, a framework that integrates teleoperation with proximity-policy learning. The left panel shows immersive teleoperation with proximity feedback, while the right panel demonstrates robust performance in both simulation and real-world dexterous manipulation.

Abstract: Multi-finger dexterous manipulation relies on stable hand-object interactions, yet these interactions are partially observable in practice. Visual observations are often occluded by the hand, tactile sensors introduce hardware-specific modalities and calibration burdens, and existing policies rarely model how these cues evolve under actions, making them brittle under contact uncertainty. To address these, we present ProxiDex, a dynamics-guided proximity policy framework that treats hand-object proximity as an interaction state for dexterous manipulation. ProxiDex reconstructs interaction point clouds and converts geometric distances into proximity cues, forming a hardware-agnostic contact representation that provides immersive feedback during VR teleoperation. Built on this representation, ProxiDex learns action-conditioned proximity dynamics with a coupled forward–inverse design: future observation latents are predicted from actions, while proximity variations are decoded from latent changes. Leveraging these dynamics, ProxiDex adaptively reweights proximity tokens across manipulation phases and uses dynamics-consistency supervision to guide policy inference, stabilizing action generation under unreliable visual feedback. Simulation and realworld experiments demonstrate improved success rates and robustness over representative baselines across standard, unseen objects, and perturbation scenarios. Additional visualizations are available at https://proxidex.github.io/.

Keywords: Dexterous Manipulation, Interaction Dynamics, Imitation Learning

## 1 Introduction

Dexterous manipulation with multi-fingered hands is a fundamental capability for human-like robotic systems [1]. Compared with parallel jaw grippers, dexterous hands require coordinated control of arm-level motion and fingertip-level interactions to accomplish complex tasks such as grasping, insertion, and tool use [2, 3]. A key requirement for learning such skills is the ability to represent local hand-object geometric interaction, which governs how contact is formed, transferred, and maintained during manipulation [4]. However, acquiring such interaction cues from demonstrations remains challenging. Vision-based teleoperation systems [5, 6] suffer from frequent occlusions that obscure critical contact regions, while tactile sensing provides limited scalability due to heterogeneous hardware designs and costly calibration across platforms [7, 8]. As a result, reliable local interaction information is often missing or inconsistent in collected data, limiting the ability of visuomotor policies to infer the underlying and evolving physical interaction structure.

This limitation exposes a broader gap in existing imitation learning approaches for dexterous manipulation. Most methods [9, 10, 11] directly learn end-to-end mappings from visual observations to actions, relying on implicit feature representations without explicitly modeling hand-object geometric proximity or its temporal evolution. Even when contact-related signals are incorporated, they are typically treated as auxiliary inputs without modeling how actions induce changes in interaction states or how interaction structure constrains future control [12, 13]. This leads to brittle behavior under occlusion, tracking noise, or contact uncertainty, where policies often generate physically inconsistent actions and fail to generalize beyond clean demonstrations.

To address these challenges, we propose ProxiDex, a dynamics-guided proximity policy learning framework that builds on hand-object geometric proximity as a coherent interaction representation and models its action-conditioned evolution for robust policy learning. As shown in Fig. 1, within a VR teleoperation setting [14], we reconstruct interaction point clouds and visualize them as immersive proximity feedback for operators. We further convert point-cloud distances into hardwareagnostic proximity feedback, enabling scalable inference of local interaction cues without tactile instrumentation. Based on this representation, ProxiDex learns forward-inverse latent dynamics that couple action-driven observation transitions with proximity variations, capturing the evolution of local hand-object interactions during manipulation. Leveraging these interaction dynamics, ProxiDex adaptively reweights proximity tokens across manipulation phases, balancing global observation guidance during motion with local proximity constraints during interaction. Dynamics consistency further guides online inference, stabilizing action generation under occlusion and tracking failures.

The main contributions of ProxiDex are summarized as follows: 1) We propose a data-construction pipeline and a VR-based teleoperation with proximity feedback, which provides scalable local interaction cues without tactile instrumentation. 2) We develop a forward-inverse latent dynamics model to capture action-conditioned evolution between observation transitions and proximity variations. 3) We introduce a trajectory-adaptive proximity policy with dynamics-consistency guided inference, improving robustness under occlusion and tracking failures.

## 2 Related Work

Demonstration data acquisition for dexterous manipulation. High-quality demonstration data are fundamental to learning dexterous manipulation. Existing data-collection systems often trade off sensing accuracy against deployment cost [9, 15]. Methods based on data gloves or tactile sensing devices [16, 17, 18] can provide fine-grained contact feedback, but they typically rely on expensive hardware and require calibration, maintenance, and cross-platform alignment. Visionbased teleoperation methods [6, 19, 20, 21] reduce deployment barriers, yet they struggle to capture local proximity relations reliably under complex hand-object occlusions. UMI-style devices [22, 23] and video-parsing methods [24, 25] offer promising scalability, but they are often constrained by specific end-effector designs or the quality of 3D reconstruction. In contrast, we adopt a VR-based arm-hand co-teleoperation pipeline and explicitly recover interaction structure through geometric proximity, enabling scalable and interaction-aware data collection.

Imitation learning for dexterous manipulation. Imitation learning has advanced dexterous manipulation by modeling action distributions with Transformers or diffusion models [9, 10]. Yet commonly used 2D observations struggle to capture the fine spatial relationships required for multifinger contact. Point-cloud-based policies improve geometric awareness [11, 26, 27], but these methods still fail to maintain a consistent representation of interaction states under occlusion. Reconstruction and pose-based methods recover more complete geometry [1], but mainly emphasize static geometric alignment rather than the temporal evolution of contact states. These limitations indicate that existing methods lack an explicit modeling of interaction as a structured and evolving state, motivating our explicit proximity-based interaction representation.

Dynamics-aware representation learning. Using dynamics models to enhance the physical awareness of policies has become an important direction in dexterous manipulation. Latent dynamic pretraining methods [28, 29] improve the dynamics sensitivity of observation encoders by predicting future states, but their latent representations often lack explicit physical meaning. Multimodal world models [30, 31, 32] assist action generation through future latent prediction, yet their complex model structures and inference pipelines can limit deployment efficiency. Geometry-evolution prediction method [33] can capture changes in point-cloud positions and velocities, but they remain limited in representing fine-grained contact logic in dexterous manipulation. Particle-based world models [34, 35] can reconstruct hand-object interaction processes, but they rarely model interaction explicitly as a bidirectional evolution of geometric proximity under actions. Motivated by these observations, we model interaction as a bidirectional evolution of geometric proximity under action influence, enabling explicit reasoning over contact dynamics.

## 3 Problem Formulation

Given an expert demonstration dataset $\boldsymbol { \mathcal { D } } = \{ \tau _ { i } \}$ , each trajectory $\tau = \{ ( o _ { t } , a _ { t } ^ { E } ) \} _ { t = 1 } ^ { T }$ consists of observations $o _ { t }$ and expert actions $a _ { t } ^ { E }$ . Imitation learning aims to train a parameterized policy $\pi _ { \theta }$ to match expert behavior. For 3D point-cloud observations, an encoder $f _ { \phi }$ is typically used to map $o _ { t }$ into a latent representation $z _ { t } = f _ { \phi } ( o _ { t } )$ , from which the policy generates an action distribution $a _ { t } \sim \pi _ { \theta } ( \cdot | z _ { t } )$

To improve the policy’s awareness of future physical evolution, prior methods often introduce a latent dynamics model $h _ { \psi }$ as auxiliary supervision. Given the current latent representation $z _ { t }$ and a future k-step sequence of expert actions $a _ { t : t + k - 1 } ^ { E }$ , the model predicts the future latent state and is optimized through a feature reconstruction loss:

$$
\hat { z } _ { t + k } = h _ { \psi } ( z _ { t } , a _ { t : t + k - 1 } ^ { E } ) , \quad \mathcal { L } _ { d y n } = \mathbb { E } _ { \tau \sim \mathcal { D } } [ \| \hat { z } _ { t + k } - f _ { \phi } ( o _ { t + k } ) \| _ { 2 } ^ { 2 } ]\tag{1}
$$

However, directly modeling the observation latent space is insufficient for contact-rich dexterous manipulation. First, frequent hand-object occlusions cause the raw observation $o _ { t }$ to miss fingertip contact regions, making interaction cues unobservable. Second, when only $\mathcal { L } _ { d y n }$ is used to enforce future feature consistency, the model tends to prioritize salient motion patterns such as object-pose changes while overlooking the fine-grained evolution of hand-object distance and contact state. Our central motivation is therefore to construct a more physically meaningful interaction representation. We first reconstruct an occlusion-free hand-object interaction point cloud $o _ { t } ^ { p }$ to mitigate geometric information loss. We then introduce latent interaction dynamics, consisting of a forward dynamics model and an inverse differential module, to help the policy anticipate proximity and contact-state evolution and generate reliable actions under perturbation.

## 4 Method

We propose ProxiDex to address visual occlusion and missing local interaction cues in dexterous manipulation, as illustrated in Fig. 2. The method proceeds in three stages: (1) interaction-aware data acquisition, which reconstructs hand–object interaction point clouds from manipulation images and robot states (Sec. 4.1); (2) latent interaction dynamics modeling, which extracts region-level proximity cues and learns action-conditioned proximity evolution (Sec. 4.2); and (3) trajectory-adaptive policy inference, which incorporates these cues into a diffusion policy to reweight proximity tokens across manipulation phases, while dynamics-consistency guidance supports closed-loop inference for robust action generation (Secs. 4.3 and 4.4; Fig. 3).

![](images/bb81b2ad95f9246de0d7875c5e3cb176bb39ac1067e8647c54da6eb05179d856.jpg)  
Figure 2: Overview of ProxiDex. It combines proximity-aware data acquisition, latent dynamics modeling, and trajectory-adaptive policy learning for robust dexterous manipulation.

## 4.1 Interaction-Aware Data Acquisition

To collect high quality demonstrations that reflect hand-object contact relations, we build an interaction-aware data collection pipeline that unifies visual observations, robot proprioception, and proximity information in the robot base frame, as illustrated in Appendix Fig. 10. Demonstrations are collected using a VR arm-hand co-teleoperation system based on Hand Tracking Streamer [14], which synchronously records robot states, target-object poses, and expert actions. Details of teleop eration control, action retargeting, and proximity feedback in VR are provided in Appendix A.1.

To represent hand-object interaction details, we first use Grounded-SAM [36] and SAM3D [37] to reconstruct the target object from a single initial image, followed by metric scale calibration to obtain the object model $\mathcal { M } _ { o } .$ , and then track its 6D pose during demonstrations using Foundation-Pose++ [38]. The full procedure is described in Appendix A.2. Inspired by interaction-centric grasp representations [39], we sample surface points from both the reconstructed object mesh and the hand-link at time t. As shown in Fig. 2(a), the sampled points are then transformed into the robot base frame to obtain $P _ { t } ^ { o }$ and $P _ { t } ^ { h }$ . The interaction point cloud observation at time t is then defined as $o _ { t } ^ { p } = P _ { t } ^ { o } \cup P _ { t } ^ { h }$ . Compared with raw scene point clouds, the interaction point cloud preserves hand-object geometric relations more reliably and reduces the influence of background noise.

From $o _ { t } ^ { p } .$ , we derive hand-object proximity observations by computing local distances between sampled hand surface points and object surface points. The resulting point-wise proximity strengths are then aggregated within each physical hand region $S _ { k }$ , yielding the region-level proximity vector $o _ { t } ^ { \mathrm { p r o x } } = [ \gamma _ { t } ^ { \bar { 1 } } , \ldots , \gamma _ { t } ^ { K } ]$ . The detailed computation is provided in Appendix A.3. In addition, we use the palm-region response $\gamma _ { t } ^ { \mathrm { p a l m } }$ to generate a trajectory-phase label $o _ { t } ^ { \mathrm { t r a j } } = \mathbb { I } ( \gamma _ { t } ^ { \mathrm { p a l m } } > \tau _ { \mathrm { p a l m } } )$ where $o _ { t } ^ { \mathrm { t r a j } } = 0$ denotes the approach phase and $o _ { t } ^ { \mathrm { t r a j } } = 1$ denotes the contact-interaction phase. This label provides a coarse motion-phase prior for subsequent models. Each demonstration frame is finally represented as $( o _ { t } ^ { p } , o _ { t } ^ { \mathrm { p r o x } } , \bar { o _ { t } ^ { \mathrm { t r a j } } } , s _ { t } , a _ { t } )$ , where $s _ { t }$ is the robot proprioceptive state and $a _ { t }$ is the expert action. This representation jointly contains the interaction point cloud, region-level proximity, and trajectory-phase label, providing a physically interpretable interaction observation for latent dynamics modeling and policy learning.

## 4.2 Latent Interaction Dynamics Modeling

After obtaining the interaction-aware demonstration representation, we model latent interaction dynamics with self-supervised learning, encouraging the observation encoder to capture action-driven changes in hand-object proximity. As illustrated in Fig. 2(b), we first pretrain a VAE on the proximity observation $o _ { t } ^ { \mathrm { p r o x } }$ , compressing region-level proximity responses into a proximity latent variable $z _ { t } ^ { \mathrm { { { p r o x } } } }$ and reconstructing $\bar { z } _ { t } ^ { \mathrm { p r o x } }$ to obtain a stable contact representation. This proximity latent variable then serves as a physical anchor for aligning visual and contact cues.

Next, the observation encoder $E _ { \theta }$ encodes the interaction point cloud $o _ { t } ^ { p }$ and robot state $s _ { t }$ into an observation latent variable $z _ { t } ^ { o b s }$ . A forward dynamics model $\mathrm { F D M } ( \cdot )$ based on the DiT architecture [40] takes $z _ { t } ^ { o b s }$ and the action sequence $\scriptstyle a _ { t : t + h - 1 }$ as input and predicts future observation latents. Meanwhile, the inverse differential module $\mathrm { I D M } ( \cdot )$ infers the proximity latent at the corresponding time step from predicted latent variable differences and aligns it with the VAE-derived $z _ { t } ^ { \mathrm { p r o x } }$ . Details are provided in Appendix B.1:

$$
\mathcal { L } _ { \mathrm { f o r } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \Vert \mathrm { F D M } ( z _ { t } ^ { o b s } , a _ { t : t + h - 1 } ) - z _ { t + h } ^ { o b s } \Vert _ { 2 } ^ { 2 } , \quad \mathcal { L } _ { \mathrm { i n v } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \Vert \mathrm { I D M } ( \hat { z } _ { t + h } ^ { o b s } - \hat { z } _ { t + h - 1 } ^ { o b s } ) - z _ { t + h } ^ { \mathrm { p r o x } } \Vert _ { 2 } ^ { 2 }\tag{2}
$$

FDM(·) constraint encourages $z _ { t } ^ { \mathrm { o b s } }$ to encode visual interaction and proprioceptive cues relevant to future state evolution, while IDM(·) alignment makes latent transitions sensitive to proximity variations. To distinguish trajectory phases, we introduce a trajectory head that predicts the interactionphase probability $p _ { \mathrm { i n t } } ^ { t }$ from $z _ { t } ^ { o \dot { b } s }$ , supervised by the trajectory label $o _ { t } ^ { \mathrm { t r a j } }$ defined in Sec. 4.1 with a cross-entropy loss $\mathcal { L } _ { \mathrm { t r a j } }$ . This stage is optimized with the following multi-task objective:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 1 } } = \lambda _ { v } { \mathcal { L } } _ { \mathrm { v a e } } + \lambda _ { f } { \mathcal { L } } _ { \mathrm { f o r } } + \lambda _ { c } { \mathcal { L } } _ { \mathrm { i n v } } + \lambda _ { t } { \mathcal { L } } _ { \mathrm { t r a j } } .\tag{3}
$$

After training, the observation encoder $E _ { \theta }$ maps hand-object geometric observations into latent representations that capture the coupling between action-driven state transitions and proximity changes. These representations provide the state basis for downstream policy generation.

## 4.3 Trajectory-Adaptive Proximity Policy

During policy learning, we freeze the observation encoder trained in the previous stage and use a Transformer-based denoising policy to generate actions. The core design is a trajectory-adaptive asymmetric causal encoder. Specifically, the global observation feature $z _ { t } ^ { o b s }$ is fully visible to all action steps and provides high-level guidance about the manipulation scene. In contrast, the proximitytoken sequence $z _ { t } ^ { \mathrm { p r o x } }$ is constrained by a causal memory mask $M _ { \mathrm { c a u s a l } }$ , so that the action at step s can access only the current and historical proximity states (Fig. 2(c)). This branch is therefore responsible for local action refinement over time. Architectural details are given in Appendix B.2. During Stage-2 training, $z _ { t } ^ { \mathrm { p r o x } }$ is obtained from the frozen VAE, whereas at inference it is replaced by the IDM estimate $\hat { z } _ { t } ^ { \mathrm { p r o x } }$ , which is trained to match the same VAE latent space through ${ \mathcal { L } } _ { \mathrm { i n v } }$

We further use the interaction-phase probability $p _ { t } ^ { \mathrm { i n t } }$ from the trajectory head in Sec. 4.2 to adaptively regulate the proximity branch: the local interaction branch is suppressed during the motion phase and strengthened during the interaction phase. Given an expert action sequence $A _ { 0 }$ , diffusion step k, noisy action $A _ { k }$ , and noise ϵ, the training objective of the second stage is:

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathbb { E } _ { A _ { 0 } , \epsilon , k } [ \| \epsilon - \epsilon _ { \theta } ( A _ { k } , k  z _ { t } ^ { o b s } , g ( p _ { t } ^ { \mathrm { i n t } } ) z _ { t } ^ { \mathrm { p r o x } } ) \| _ { 2 } ^ { 2 } ] + \lambda _ { \mathrm { g a t e } } \mathcal { L } _ { \mathrm { g a t e } }\tag{4}
$$

Here, $g ( p _ { t } ^ { \mathrm { i n t } } )$ is a gate conditioned on the interaction-phase probability, and $\mathcal { L } _ { \mathrm { g a t e } }$ regularizes the gate distribution. This design allows the policy to rely on global visual guidance during the motion phase and adaptively strengthen proximity feedback during the interaction phase, producing more stable contact-aware actions throughout dexterous manipulation.

## 4.4 Dynamics-Consistency Guided Policy Inference

During online inference, we combine the policy with the forward dynamics model FDM(·) to form a closed-loop controller with physical consistency checking, as shown in Fig. 3. At each time step, the observation encoder $E _ { \theta }$ extracts the latent variable $z _ { t } ^ { o b s }$ from the real-time interaction point cloud $o _ { t } ^ { p }$ and robot state $s _ { t } ,$ , and estimates the current proximity latent feature $\hat { z } _ { t } ^ { \mathrm { p r o x } }$ through the IDM(·). The trajectory head then outputs the interaction probability $p _ { t } ^ { \mathrm { i n t } }$ , which adaptively modulates the proximity branch. The policy generates an action sequence conditioned on $z _ { t } ^ { o b s } , \hat { z } _ { t } ^ { \mathrm { p r o x } }$ , and $p _ { t } ^ { \mathrm { i n t } }$ , and executes it in a receding-horizon manner.

![](images/253151ed9d1fd5deb1d8af49dc2d14c84baa9e1636296430d99957ea68efc89c.jpg)  
Figure 3: Overview of the Policy Inference Pipeline with Dynamics Consistency Guidance.

Table 1: Simulation results on dexterous manipulation benchmarks. We evaluate all methods on Adroit, DexArt, and three self-designed IsaacLab tasks.
<table><tr><td>Algorithm\Task</td><td>Obs.</td><td>Pretrain</td><td>Adroit 3 Tasks</td><td>DexArt 4 Tasks</td><td>Self-Designed 3 Tasks</td><td>Avg.</td></tr><tr><td>DP3 [11]</td><td>PC</td><td>X</td><td>77.8±2.4</td><td>60.6±0.7</td><td>80.8±2.3</td><td>71.8±1.7</td></tr><tr><td>ManiFlow [26]</td><td>PC</td><td>X</td><td>78.6±2.3</td><td>63.3±2.7</td><td> $8 4 . 4 \pm 1 . 9$ </td><td> $7 4 . 2 { \pm } 2 . 3 $ </td></tr><tr><td>AFRO [28]</td><td>PC</td><td>V</td><td>84.0±2.8</td><td> $6 4 . 7 { \pm } 2 . 5 $ </td><td> $8 6 . 0 { \pm } 1 . 5 $ </td><td> $7 6 . 9 { \pm } 2 . 3 $ </td></tr><tr><td>CordViP [1]</td><td>Int.PC</td><td>X</td><td>85.6±2.3</td><td> $7 1 . 8 { \pm } 1 . 8 $ </td><td> $8 8 . 8 { \pm } 1 . 6 $ </td><td> $8 1 . 0 { \pm } 1 . 9 $ </td></tr><tr><td>ProxiDex (Ours)</td><td>Int.PC</td><td></td><td>90.3±1.4</td><td>73.7±2.2</td><td> ${ \bf 9 1 . 1 { \pm } 1 . 7 }$ </td><td>83.9±1.8</td></tr></table>

To handle occlusion or short-term perception failure, the system uses FDM(·) to predict the next latent variable and computes the dynamics deviation between the prediction and the real observation:

$$
e _ { t + 1 } ^ { \mathrm { d y n } } = \left. z _ { t + 1 } ^ { o b s } - \mathrm { F D M } ( z _ { t } ^ { o b s } , a _ { t } ) \right. _ { 2 }\tag{5}
$$

When $e _ { t + 1 } ^ { \mathrm { d y n } } \leq \tau _ { \mathrm { d y n } }$ , ProxiDex regards real-time visual feedback as reliable and updates the policy state with the perceived latent observation. Otherwise, it switches to dynamics-dream mode, where action generation is maintained using latent variables autoregressively predicted by FDM(·). The system returns to regular closed-loop inference once the deviation $e _ { t + 1 } ^ { \mathrm { d y n } }$ falls below the threshold $\tau _ { \mathrm { d y n } }$ . In deployment, FoundationPose++ runs at around 10 Hz, the proximity policy at about 28 Hz, and FDM(·) at about 20 Hz, which is sufficient for receding-horizon inference; implementation details are given in Appendix B.3.

## 5 Experiments

We evaluate ProxiDex in both simulation and real-world settings, focusing on two aspects: (i) whether interaction point clouds and proximity cues can improve dexterous manipulation performance; (ii) whether the learned interaction dynamics can maintain stable control under disturbances.

## 5.1 Simulation Experiments

Setup. We evaluate ProxiDex on 10 multi-finger tasks from Adroit [41], DexArt [42], and our customized IsaacLab environments [43]. Demonstrations for Adroit and DexArt are generated by reinforcement-learning experts, while customized tasks are collected through VR teleoperation. For each task, we collect 30 high-quality demonstrations covering representative hand–object contact patterns. Since object poses are directly available in simulation, we can accurately generate interaction point clouds $o _ { t } ^ { p }$ and region-level proximity features $o _ { t } ^ { \mathrm { p r o x } }$ , as visualized in Appendix Fig. 21. Further details are provided in Appendices C.1 and C.2.

Baselines. We compare ProxiDex with four representative 3D point cloud policies. DP3 [11] and ManiFlow [26] use camera scene point clouds (PC) for end-to-end action prediction. AFRO [28] uses the same input with dynamics pretraining to evaluate the effect of latent dynamics. CordViP [1] adopts reconstructed interaction point clouds (Int.PC) and an arm-coordinated denoising policy.

Results. Table 1 summarizes the benchmark-level performance averaged over the evaluated tasks. ProxiDex achieves the best performance on all three task groups, reaching an overall success rate of 83.9% and outperforming DP3, ManiFlow, AFRO, and CordViP by 12.1, 9.7, 7.0, and 2.9 percentage points, respectively. On Adroit, DexArt, and our customized IsaacLab tasks, ProxiDex obtains 90.3%, 73.7%, and 91.1% success rates, exceeding CordViP by 4.7, 1.9, and 2.3 percentage points. Since CordViP also uses interaction point clouds, these gains mainly come from modeling fine-grained hand–object proximity and its temporal evolution. This is especially beneficial for contact-sensitive tasks such as Pen and Driller, where the policy must continuously adjust fingertip placement and wrist pose after contact under occlusion and object motion.

Ablation on Task Success Rate. To further analyze the contribution of each core component to task success, Fig. 4 presents task-wise ablations on six representative Adroit and DexArt tasks. ProxiDex performs best across all tasks, showing the joint benefit of interactioncentric representation and dynamics-aware policy learning. Removing proximity cues or replacing interaction point clouds with camera-observed scene point clouds causes clear drops, especially on contact-sensitive tasks such as Bucket and Pen, highlighting the importance of explicit hand–object interaction geometry under occlusion. Removing FDM, IDM, or closed-loop dynamics

![](images/8a37cbbfb64cdf64a0d5f0664904efd863b933c62e9da58d68ed0434c97baa9a.jpg)  
Figure 4: Ablation on Success Rate.

checking also reduces task success, indicating that latent dynamics modeling and online verification help improve execution stability during contact-rich manipulation.

Ablation on Occlusion Robustness. To evaluate component contributions to occlusion robustness, we introduce a cuboid occluder into the depth-camera view for 2, 4, and 6 s, and measure recovery success on the Adroit Pen task. As shown in Fig. 5, full ProxiDex achieves 80% and 72% recovery under 2-s and 4-s occlusions. Removing FDM causes the largest drop, as future-latent prediction and dynamics takeover are no longer available. Without IDM, hand–object proximity is harder to recover from latent transitions, reducing success to 71% and 59%. Removing proximity cues or

![](images/a4ffedbf2d42ade8b5f9323faf24d13ca8d7629e80b6417887cff4884c7571ac.jpg)  
Figure 5: Ablation on Occlusion.

trajectory gating also weakens local interaction modeling and phase-adaptive modulation. Under 6-s occlusion, accumulated prediction errors reduce the full model to 35% recovery.

## 5.2 Real-World Experiments

Setup. The real-robot platform consists of a Realman RM75B 7-DoF robotic arm, a RealSense L515 RGB-D camera, and two dexterous hands. Specifically, a CasBot-P1L hand with 6 active DoFs is used for the standard tabletop manipulation tasks, while a 20-DoF Wuji Hand V1 is used for the more contact-rich manipulation tasks. As shown in Fig. 6 and Fig. 22, we design three tabletop tasks. For each task, we collect 50 demonstrations and evaluate the policy under in-distribution objects, unseen objects, and disturbance conditions. More details are provided in Appendix E.

Results on Tabletop Tasks. Table 2 shows that ProxiDex achieves the highest real-robot success rate, with an overall average of 72.6%. It outperforms CordViP, AFRO, ManiFlow, and DP3 by 9.7, 19.1, 22.9, and 32.6 percentage points, respectively. ProxiDex also obtains the best or tied-best result in nine task settings, demonstrating that the proposed interaction-centric representation and proximity-aware policy consistently improve real-world dexterous manipulation performance.

Table 2: Real-robot task success counts under in-distribution, unseen-object, perturbation, and contact-rich settings. The results show that ProxiDex achieves accurate and robust performance.
<table><tr><td rowspan="2">Tasks Algorithm</td><td colspan="3">In Distribution</td><td colspan="3">Unseen Objects</td><td colspan="3">Perturbation</td><td colspan="2">Contact-Rich</td><td rowspan="2">Avg.</td></tr><tr><td>Pick</td><td>Pinch</td><td>Sweep</td><td>Pick</td><td>Pinch</td><td>Sweep</td><td>Pick</td><td>Pinch</td><td>Sweep</td><td>Twist</td><td>Flip</td></tr><tr><td>DP3 [11]</td><td>14/20</td><td>10/20</td><td>12/20</td><td>18/40</td><td>9/40</td><td>16/40</td><td>14/30</td><td>13/30</td><td>14/30</td><td>2/20</td><td>2/20</td><td>40.0%</td></tr><tr><td>ManiFlow [26]</td><td>16/20</td><td>11/20</td><td>15/20</td><td>24/40</td><td>12/40</td><td>19/40</td><td>17/30</td><td>18/30</td><td>17/30</td><td>2/20</td><td>3/20</td><td>49.7%</td></tr><tr><td>AFRO [28]</td><td>17/20</td><td>11/20</td><td>16/20</td><td>25/40</td><td>15/40</td><td>24/40</td><td>17/30</td><td>17/30</td><td>16/30</td><td>3/20</td><td>5/20</td><td>53.5%</td></tr><tr><td>CordViP [1]</td><td>18/20</td><td>13/20</td><td>17/20</td><td>29/40</td><td>21/40</td><td>27/40</td><td>19/30</td><td>19/30</td><td>18/30</td><td>6/20</td><td>8/20</td><td>62.9%</td></tr><tr><td>ProxiDex-1</td><td>18/20</td><td>14/20</td><td>17/20</td><td>30/40</td><td>22/40</td><td>28/40</td><td>20/30</td><td>20/30</td><td>20/30</td><td>7/20</td><td>6/20</td><td>65.2%</td></tr><tr><td>ProxiDex-2</td><td>18/20</td><td>14/20</td><td>18/20</td><td>31/40</td><td>23/40</td><td>28/40</td><td>21/30</td><td>22/30</td><td>21/30</td><td>8/20</td><td>7/20</td><td>68.1%</td></tr><tr><td>ProxiDex</td><td>19/20</td><td>15/20</td><td>18/20</td><td>32/40</td><td>24/40</td><td>29/40</td><td>22/30</td><td>23/30</td><td>22/30</td><td>10/20</td><td>11/20</td><td>72.6%</td></tr></table>

Unseen Objects  
Perturbation1 (Cluttered)  
Perturbation2 (Occluded)  
![](images/bc23f0569a7f158c673e569d06cabd4eaece2243e68b8bf4cf58821561d8f743.jpg)  
Figure 6: We introduce unseen objects, cluttered scenes, and visual occlusions into three tabletop manipulation tasks. The results show that ProxiDex can maintain stable hand-object interaction and complete the corresponding manipulation tasks in complex real-world scenarios.

Generalization and robustness. Beyond the overall performance, Table 2 further shows that ProxiDex remains effective under distribution shifts and external disturbances. For unseen objects, it succeeds in $8 5 / 1 2 0$ trials, compared with 77/120 for CordViP. Under perturbations, ProxiDex achieves 67/90 successful trials, while CordViP obtains 56/90.

These results indicate that explicit hand–object proximity modeling improves robustness to object variations and external disturbances. In contrast, visible-scene pointcloud methods such as DP3 and ManiFlow degrade more clearly in Sweep, where hand occlusion and object pose changes make the observed scene geometry unstable.

Backbone Ablation. To further examine whether the gains only come from the policy backbone, the additional rows in Table 2 compare different policy backbones within ProxiDex. ProxiDex-1 with a U-Net backbone and ProxiDex-2 with a standard Transformer achieve 65.2% and 68.1% average success rates. The full ProxiDex further improves them by 7.4 and 4.5 percentage points, showing that the proposed backbone can emphasize hand–object proximity at critical contact moments. As shown in Fig. 7, the interaction-phase probability is used to adaptively adjust the attention weights between proximity and point-cloud tokens, allowing the policy to emphasize hand-object proximity at critical interaction moments.

![](images/0b6401018fcad953faa5c13d76a8834e06462226db99aead5d8155a01e47af6b.jpg)  
Figure 7: Attention visualization. Averaged Transformer attention weights of point-cloud and proximity tokens across three tabletop tasks.

Contact-Rich Task Analysis. We further evaluate two contact-rich tasks involving sustained finger–object interaction and object reorientation. As shown in Fig. 8, ProxiDex maintains stable interaction representations and completes complex in-hand manipulation even under short-term visual occlusion. The interaction point cloud provides local hand–object geometry, while dynamics prediction preserves interaction-state continuity when visual feedback becomes temporarily unreliable. As shown in Table 2, ProxiDex achieves the best performance on both contact-rich tasks, indicating that their combination improves stability during sustained contact and object reorientation.

Comparison with Real Tactile Sensing. To compare proximity with real tactile feedback, each fingertip of the P1L hand is equipped with a 2 × 4 tactile array. As shown in Fig. 9, proximity achieves competitive task performance compared with tactile sensing. Although proximity cannot replace force feedback, it captures changes in hand–object distance and spatial distribution before and around contact, providing useful geometric cues for pre-grasp adjustment and subsequent contact establishment.

![](images/f2c84e396cb891b6e909c55538ae163d508f0fc535f7fecaaeb5038c9f90b8a7.jpg)

![](images/0c86df106b86910028e1d954e3f8f2cda2a656c36f4e7ee5e512e168b705e15e.jpg)

![](images/db4596a89f74ac9c414d84aae538a413b0a004420d03528511ec1d7cebe6a829.jpg)

![](images/947a30013337f6ba95aac6365f26ca463d520794a42ed95036859c0e9e1dd1f7.jpg)

![](images/472417a47a994145734241b354651659bd34b4424c76ac27767625303668550a.jpg)

![](images/94931e1821833cee7d5d3fbe6e50b207dd3debab37d90ff2a9c67b347c0efb38.jpg)  
Figure 8: ProxiDex performance and interaction point-cloud visualization on contact-rich tasks.

![](images/478502cca95073898253d63aaa2e5dbfa29deb7053e73b24d6430969d266605e.jpg)

(a) Dexterous Hand + Tactile sensors  
![](images/b06abcc3781a961138cbed4b8fc8ff8300e8d02f6c8887000dde2943e9fd1712.jpg)  
Figure 9: (a) P1L hand with tactile arrays. (b) Proximity–tactile comparison.

## 6 Conclusion

We presented ProxiDex, a proximity-policy learning framework for dexterous manipulation. ProxiDex constructs region-level proximity representations from reconstructed hand–object interaction point clouds and further learns action-conditioned latent interaction dynamics to guide trajectoryadaptive policy learning and closed-loop inference. By jointly modeling local hand–object geometry and its dynamic evolution, ProxiDex supports stable policy execution under short-term unreliable visual feedback, sustained contact, and complex object interactions. Simulation and real-world experiments show consistent performance across standard, unseen-object, disturbance, visual-occlusion, and contact-rich settings. Overall, these results demonstrate that ProxiDex provides a robust and physically meaningful interaction modeling framework for dexterous manipulation.

## 7 Limitations

Although ProxiDex shows strong stability, it still has several limitations. First, it depends on accurate object reconstruction and real-time pose tracking, which may be unreliable for small components, articulated objects, or high-precision manipulation. Second, dynamics takeover is mainly effective for short-term perception failures, while prolonged occlusion can accumulate autoregressive prediction errors and reduce control reliability. Finally, the current proximity representation remains geometry-driven and cannot measure real contact forces or precise fingertip force distributions. Future work will combine improved pose estimation, real tactile sensing, and large-scale human manipulation videos to enhance long-horizon recovery and fine-grained contact modeling.

## Acknowledgments

This work was supported by the Hebei Provincial Science and Technology Plan Project under Grant No. 25241802D, the National Natural Science Foundation of China under Grant No. 62303457, and the New Generation Artificial Intelligence-National Science and Technology Major Project under Grant No. 2025ZD0122900.

## References

[1] Y. Fu, Q. Feng, N. Chen, Z. Zhou, M. Liu, M. Wu, T. Chen, S. Rong, J. Liu, H. Dong, et al. Cordvip: Correspondence-based visuomotor policy for dexterous manipulation in real-world. arXiv preprint arXiv:2502.08449, 2025.

[2] Y. Bai, F. Chen, H. Sun, Y. Tong, E. Li, and Z. Zhang. Far-dex: Few-shot data augmentation and adaptive residual policy refinement for dexterous manipulation. arXiv preprint arXiv:2603.10451, 2026.

[3] Y. Qin, B. Huang, Z.-H. Yin, H. Su, and X. Wang. Dexpoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation. In Conference on Robot Learning, pages 594–605. PMLR, 2023.

[4] Z. Xu, Y. Wang, B. Abbatematteo, J. Preechayasomboon, S. Chan, N. Colonnese, and A. H. Memar. Contact-grounded policy: Dexterous visuotactile policy with generative contact grounding. arXiv preprint arXiv:2603.05687, 2026.

[5] A. Handa, K. Van Wyk, W. Yang, J. Liang, Y.-W. Chao, Q. Wan, S. Birchfield, N. Ratliff, and D. Fox. Dexpilot: Vision-based teleoperation of dexterous robotic hand-arm system. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 9164–9170. IEEE, 2020.

[6] A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto. Open teach: A versatile teleoperation system for robotic manipulation. In Conference on Robot Learning, pages 2372– 2395. PMLR, 2025.

[7] B. Huang, Y. Wang, X. Yang, Y. Luo, and Y. Li. 3d-vitac: Learning fine-grained manipulation with visuo-tactile sensing. In Conference on Robot Learning, pages 2557–2578. PMLR, 2025.

[8] J. Ren, J. Zou, and G. Gu. Mc-tac: Modular camera-based tactile sensor for robot gripper. In International Conference on Intelligent Robotics and Applications, pages 169–179. Springer, 2023.

[9] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[10] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[11] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954, 2024.

[12] J. Li, T. Wu, J. Zhang, Z. Chen, H. Jin, M. Wu, Y. Shen, Y. Yang, and H. Dong. Adaptive visuotactile fusion with predictive force attention for dexterous manipulation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3232–3239, 2025.

[13] T. Wu, J. Li, J. Zhang, M. Wu, and H. Dong. Canonical representation and force-based pretraining of 3d tactile for dexterous visuo-tactile policy learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 6786–6792, 2025.

[14] Z. K. Weng. Hand tracking streamer: Meta quest vr app for motion capture and teleoperation, 2026. URL https://githu.com/wengmister/hand-tracking-streamer.

[15] H. Fang, C. Wang, Y. Wang, J. Chen, S. Xia, J. Lv, Z. He, X. Yi, Y. Guo, X. Zhan, et al. Airexo-2: Scaling up generalizable robotic imitation learning with low-cost exoskeletons. In Conference on Robot Learning, pages 198–220. PMLR, 2025.

[16] H. Zhang, S. Hu, Z. Yuan, and H. Xu. Doglove: Dexterous manipulation with a low-cost open-source haptic force feedback glove. arXiv preprint arXiv:2502.07730, 2025.

[17] R. Wen, J. Zhang, G. Chen, Z. Cui, M. Du, Y. Gou, Z. Han, J. Hu, L. Huang, H. Niu, et al. Dexterous teleoperation of 20-dof bytedexter hand via human motion retargeting. arXiv preprint arXiv:2507.03227, 2025.

[18] K. Li, S. M. Wagh, N. Sharma, S. Bhadani, W. Chen, C. Liu, and P. Kormushev. Hapticact: Bridging human intuition with compliant robotic manipulation via immersive vr. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1084– 1091. IEEE, 2025.

[19] Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox. Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system. arXiv preprint arXiv:2307.04577, 2023.

[20] X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang. Open-television: Teleoperation with immersive active visual feedback. In Conference on Robot Learning, pages 2729–2749. PMLR, 2025.

[21] R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang. Bunny-visionpro: Realtime bimanual dexterous teleoperation for imitation learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12248–12255. IEEE, 2025.

[22] M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025.

[23] T. Tao, M. K. Srirama, J. J. Liu, K. Shaw, and D. Pathak. Dexwild: Dexterous human interactions for in-the-wild robot policies. arXiv preprint arXiv:2505.07813, 2025.

[24] H. Chen, T. Dong, T. Wu, L. Wang, Y. Jangir, Y. Niu, Y. Ye, H. Bharadhwaj, Z. Erickson, and J. Ichnowski. Dexterous manipulation policies from rgb human videos via 3d hand-object trajectory reconstruction. arXiv preprint arXiv:2602.09013, 2026.

[25] J. Mu, S. Yang, Y. Bao, H. Bae, T. Wei, L. Xu, B. Li, H. Xu, and J. Pang. Deximit: Learning bimanual dexterous manipulation from monocular human videos. arXiv preprint arXiv:2602.10105, 2026.

[26] G. Yan, J. Zhu, Y. Deng, S. Yang, R.-Z. Qiu, X. Cheng, M. Memmel, R. Krishna, A. Goyal, X. Wang, et al. Maniflow: A general robot manipulation policy via consistency flow training. In Conference on Robot Learning, pages 2268–2293. PMLR, 2025.

[27] Y. Fang, X. Zhang, H. Cheng, X. Zang, R. Song, and J. Zhao. Flow policy: Generalizable visuomotor policy learning via flow matching. IEEE/ASME Transactions on Mechatronics, 2025.

[28] Q. Liang, B. Cai, M. Lai, S. Zhuang, T. Lin, Y. Qin, Y. Ye, J. Liang, and R. Xu. Bootstrap dynamic-aware 3d visual representation for scalable robot learning. arXiv preprint arXiv:2512.00074, 2025.

[29] J. Lyu, K. Liu, X. Zhang, H. Liao, Y. Feng, W. Zhu, T. Shen, J. Chen, J. Zhang, Y. Dong, et al. Lda-1b: Scaling latent dynamics action model via universal embodied data ingestion. arXiv preprint arXiv:2602.12215, 2026.

[30] Y. Zheng, S. Gu, W. Li, Y. Zheng, Y. Zang, S. Tian, X. Li, C. Hao, C. Gao, S. Liu, et al. Omnivta: Visuo-tactile world modeling for contact-rich robotic manipulation. arXiv preprint arXiv:2603.19201, 2026.

[31] C. Higuera, S. Arnaud, B. Boots, M. Mukadam, F. R. Hogan, and F. Meier. Visuo-tactile world models. arXiv preprint arXiv:2602.06001, 2026.

[32] H. Yuan, W. Yi, Z. Zhang, W. Chen, Y. Mo, J. Yin, X. Li, X. Zeng, C. Wen, C. Lu, et al. Vtam: Video-tactile-action models for complex physical interaction beyond vlas. arXiv preprint arXiv:2603.23481, 2026.

[33] Y. Zheng, J. Lyu, Y. Zhang, J. Chen, M. Yan, Y. Deng, X. Shi, X. Zhao, Y. Wang, Z. Zhang, et al. Emerging extrinsic dexterity in cluttered scenes via dynamics-aware policy learning. arXiv preprint arXiv:2603.09882, 2026.

[34] Z. Hong, Y. Liu, H. Hou, B. Ai, J. Wang, T. Mu, Y. Qin, J. Gu, and H. Su. Learning particlebased world model from human for robot dexterous manipulation. In 3rd RSS Workshop on Dexterous Manipulation: Learning and Control with Diverse Data, 2025.

[35] Z. He, B. Ai, T. Mu, Y. Liu, W. Wan, J. Fu, Y. Du, H. I. Christensen, and H. Su. Scaling crossembodiment world models for dexterous manipulation. arXiv preprint arXiv:2511.01177, 2025.

[36] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

[37] X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, et al. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025.

[38] W. Yan and J. Chu. FoundationPose++, Mar. 2025. URL https://github.com/teal024/ FoundationPose-plus-plus.

[39] Z. Wei, Z. Xu, J. Guo, Y. Hou, C. Gao, Z. Cai, J. Luo, and L. Shao. D(R, O) Grasp: A Unified Representation of Robot and Object Interaction for Cross-Embodiment Dexterous Grasping. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 4982–4988. IEEE, 2025.

[40] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[41] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.

[42] C. Bao, H. Xu, Y. Qin, and X. Wang. Dexart: Benchmarking generalizable dexterous manipulation with articulated objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21190–21200, 2023.

[43] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbrugg, N. Rudin, et al. Isaac lab: A gpu-accelerated simulation framework for multi-¨ modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

## Supplementary Material for ProxiDex

This supplementary material provides additional details on data acquisition, latent dynamics model ing, policy learning, and experimental protocols. It is organized as follows:

• Appendix A Proximity-guided data acquisition and geometric representation.

• Appendix B Training and closed-loop inference details of ProxiDex.

• Appendix C Simulation setup and qualitative analysis.

• Appendix D Additional experiments and ablation studies.

• Appendix E Real-world experiment setup and evaluation protocol.

## A Proximity-Guided Data Acquisition and Geometric Representation

## A.1 VR Teleoperation and Action Retargeting

Arm teleoperation. We use incremental inverse kinematics (IK) to retarget VR wrist motion to the robot end-effector. The system [14] first records the initial VR wrist position ${ \bf { x } } _ { \mathrm { { i n i t } } }$ and the initial robot end-effector position $\mathbf { x } _ { \mathrm { h o m e } } .$ . At runtime, it computes the displacement of the current VR wrist position ${ \pmb x } _ { \mathrm { v r } }$ relative to the calibrated origin, and maps this displacement into the robot base frame using a scale vector S and a coordinate retargeting matrix $R _ { \mathrm { m a p } } { \mathrm { . } }$

$$
{ \pmb x } _ { \mathrm { t a r g e t } } = { \pmb x } _ { \mathrm { h o m e } } + { \bf R } _ { \mathrm { m a p } } \left( { \bf S } \circ ( { \pmb x } _ { \mathrm { v r } } - { \pmb x } _ { \mathrm { i n i t } } ) \right) ,\tag{6}
$$

where $\mathbf { S }$ is an anisotropic scaling vector, $\mathbf { R } _ { \mathrm { m a p } }$ aligns the VR coordinate frame with the robot base frame, and ◦ denotes element-wise multiplication. The target pose is solved by numerical inverse kinematics under joint-limit and redundancy constraints, and the resulting joint commands are sent to the Realman RM75B arm through the low-level controller.

For orientation control, the system uses quaternion increments to synchronize the VR wrist orientation and the robot end-effector orientation. Let $q _ { \mathrm { v r } }$ be the current VR wrist orientation, $q _ { \mathrm { i n i t } }$ the initial calibrated orientation, and $q _ { \mathrm { h o m e } }$ the initial robot end-effector orientation. The target orientation is

$$
\Delta q = q _ { \mathrm { v r } } \otimes q _ { \mathrm { i n i t } } ^ { - 1 } , \qquad q _ { \mathrm { t a r g e t } } = q _ { \mathrm { h o m e } } \otimes \Delta q .\tag{7}
$$

The Cartesian target is passed to a numerical IK solver in PyBullet, which computes joint commands under 7-DoF redundancy and joint-limit constraints. The resulting commands are synchronized to the RealMan RM75B arm through CAN-FD, enabling low-latency spatial following.

Dexterous-hand teleoperation. The VR detector provides 21 hand keypoints $\mathbf { { \pmb { u } } } _ { i } \ [ 1 4 ]$ . We normalize them by removing the wrist translation and applying a scale factor, i.e., $\tilde { \mathbf { u } } _ { i } = \eta ( \mathbf { { u } } _ { i } - \mathbf { { u } } _ { 0 } )$ followed by a handedness correction when needed. A local wrist frame is estimated from the wrist and finger-root keypoints, and the normalized keypoints are reprojected into a canonical hand coordinate system. This step reduces differences among operators and provides a consistent geometric input for retargeting.

The retargeting problem converts human-hand topology into robot joint commands. We construct target vector flows $\hat { v } _ { k }$ that describe the relative geometry of the human fingers and solve for the robot joint configuration $q _ { t } ^ { * }$ by minimizing

$$
q _ { t } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { q } } \sum _ { k } w _ { k } \rho _ { \delta } \left( \| r _ { k } ( \boldsymbol { q } ) - \hat { v } _ { k } \| _ { 2 } ^ { 2 } \right) + \lambda \| \boldsymbol { q } - \boldsymbol { q } _ { t - 1 } \| _ { 2 } ^ { 2 } .\tag{8}
$$

Here, $\rho _ { \delta }$ is the Huber loss for robustness to keypoint noise, and the second term regularizes temporal smoothness by penalizing abrupt joint changes. We further introduce a DexPilot-style grasp prior projection [5] and update the projection state with threshold hysteresis, improving fingertip closure while reducing hand jitter.

![](images/fe60d83db5368d4f40f8359c49785cf9d0e01a0b48e777fa73b445da21bd0b24.jpg)  
Figure 10: VR teleoperation pipeline in ProxiDex, where OpenXR hand keypoints and wrist poses are streamed to a workstation for hand–arm control, and the reconstructed hand–object interaction point cloud is sent back to the headset as optional proximity feedback for immersive teleoperation.

The teleoperation loop is sufficiently responsive for real-time operation. TCP hand-joint commands are transmitted at approximately 30 Hz, object-pose detection runs at around 10 Hz, and the interaction point-cloud stream is updated at about 23 Hz. These rates allow the operator to receive timely visual and proximity feedback during manipulation, as summarized in Fig. 10. In particular, when direct visual observation is degraded by hand-object occlusions, the proximity feedback provides an augmented-reality style cue that helps the operator perceive the hand-object distance and adjust the manipulation process more intuitively. With the teleoperation stream defined, the next step is to convert the reconstructed object geometry into a metric-scale representation for distance-based proximity computation.

## A.2 Object Reconstruction and Metric-Scale Calibration

We further detail how ProxiDex reconstructs occlusion-free hand-object interaction point clouds from demonstration data. We first use Grounded-SAM[36] to segment the target object from the initial RGB observation, and then reconstruct the target mesh $\mathcal { M } _ { o }$ using SAM3D[37]. Monocular reconstruction can complete invisible regions and provide a full geometric prior, but its output usually lacks reliable metric scale. Therefore, before using the mesh for interaction point-cloud synthesis and proximity computation, we perform scale calibration through the following procedure.

For each object, the system captures one aligned RealSense RGB-D observation and estimates a scale factor κ from the depth map. Given an RGB image, a metric depth map, and camera intrinsics, Grounded-SAM first produces the target object mask. Valid depth pixels inside the mask are then back-projected into the camera frame:

$$
p ( u , v ) = \left( \frac { ( u - c _ { x } ) d } { f _ { x } } , \frac { ( v - c _ { y } ) d } { f _ { y } } , d \right) ,\tag{9}
$$

where $( u , v )$ denotes pixel coordinates, d is the depth value, and $( c _ { x } , c _ { y } )$ and $( f _ { x } , f _ { y } )$ are the principal point and focal lengths. After removing statistical outliers from the back-projected object point cloud, we compute its oriented bounding-box size $e _ { \mathrm { r g b d } }$

We also load the vertices of the OBJ mesh reconstructed by SAM3D and compute its oriented bounding-box size $e _ { \mathrm { o b j } }$ . To avoid axis-order effects in scale estimation, both size vectors are sorted before comparison:

$$
r = { \frac { \mathrm { s o r t } ( e _ { \mathrm { r g b d } } ) } { \mathrm { s o r t } ( e _ { \mathrm { o b j } } ) } } .\tag{10}
$$

The final uniform scale coefficient is $\kappa = \mathrm { m e d i a n } ( r )$ , which is applied to the mesh vertices. This calibration preserves the shape-completion advantage of SAM3D while grounding subsequent distance calculations in metric geometry. Using the median of the sorted axis ratios also reduces sensi tivity to a single noisy bounding-box dimension, which can appear when the RGB-D mask contains small boundary artifacts. The calibrated mesh therefore provides a stable object surface for the region-level proximity computation described next.

Table 3: Input data specifications across simulation and real-world environments.
<table><tr><td></td><td>Environment Input Setting</td><td>Point-Cloud Input</td><td>Auxiliary Input</td><td>Action Dim.</td><td>Horizon / Notes</td></tr><tr><td colspan="6">Simulation Environments</td></tr><tr><td>Adroit</td><td>Scene Point Cloud</td><td>[512, 3] point_cloud</td><td>[24] 1ow_dim state</td><td>28</td><td> $n _ { \mathrm { o b s } }$ </td></tr><tr><td>Adroit</td><td>Interaction Point Cloud</td><td>[3 × 512, 3] point_cloud</td><td>[24] 1ow_dim state</td><td>28</td><td>N = 3, including hand and object</td></tr><tr><td>DexArt</td><td>Scene Point Cloud</td><td>[1024, 3] point_cloud</td><td>[32] 1ow_dim state</td><td>22</td><td>nobs</td></tr><tr><td>DexArt</td><td>Interaction Point Cloud</td><td>[2 × 1024, 3] point_cloud</td><td>[32] 1ow_dim state</td><td>22</td><td>N = 2, including hand and object</td></tr><tr><td>IsaacLab</td><td>Scene Point Cloud</td><td>[1024, 3] point_cloud</td><td>[18] 1ow_dim state</td><td>18</td><td>Nobs</td></tr><tr><td>IsaacLab</td><td>Interaction Point Cloud</td><td>[N × 1024, 3] point_cloud</td><td>[18] low_dim state</td><td>18</td><td>N = 2 for Box and N = 3 for other tasks</td></tr><tr><td colspan="6">Real-World Environment</td></tr><tr><td>Real World</td><td>Scene Point Cloud</td><td>[512, 3] point_cloud</td><td>[13] low_dim state</td><td>13</td><td>Nobs</td></tr><tr><td>Real World</td><td>Interaction Point Cloud</td><td>[N × 512, 3] point_cloud</td><td>[13] low_dim state</td><td>13</td><td>N = 3 for Sweep and N = 2 for other tasks</td></tr></table>

## A.3 Geometric Proximity Representation

Given the metrically calibrated object mesh, the system samples points from the surfaces of object mesh and hand URDF links, and transforms them into the robot base frame using the object 6D pose tracked by FoundationPose++ [38] and robot forward kinematics. This yields the object point cloud $P _ { t } ^ { o }$ and the hand point cloud $P _ { t } ^ { h }$ . The interaction point cloud at frame t is defined as $o _ { t } ^ { p } = P _ { t } ^ { o } \cup P _ { t } ^ { h }$ Compared with a raw scene point cloud, this representation explicitly preserves the relative geometry between the hand and object while reducing the influence of background points and self-occlusion.

Proximity is computed from geometric distances between the hand and object point clouds, without assuming any specific tactile sensor. For each hand sample $p _ { i } ^ { h } \in  { \mathcal { P } } _ { t } ^ { h }$ , we first compute its nearestneighbor distance to the object point cloud:

$$
d _ { i } ^ { \mathrm { n n } } = \operatorname* { m i n } _ { p _ { j } ^ { o } \in P _ { t } ^ { o } } \left. p _ { i } ^ { h } - p _ { j } ^ { o } \right. _ { 2 }\tag{11}
$$

In practice, small geometric interpenetrations may occur due to pose-tracking noise, calibration errors, or simplified hand/object meshes. Since the proximity representation is intended to describe near-field interaction rather than penetration depth, we treat such cases as saturated contact responses. Specifically, using the signed distance field $\phi _ { \mathcal { O } } ( \cdot )$ of the object mesh, where negative values indicate points inside the object, the effective distance is clipped as

$$
d _ { i } = \left\{ \begin{array} { l l } { d _ { i } ^ { \mathrm { n n } } , } & { \phi \sigma ( p _ { i } ^ { h } ) \geq 0 , } \\ { 0 , } & { \phi \sigma ( p _ { i } ^ { h } ) < 0 . } \end{array} \right.\tag{12}
$$

This prevents occasional interpenetration artifacts from producing unstable proximity values while preserving consistent contact cues for policy learning. A Gaussian kernel then maps the effective distance to a normalized proximity response:

$$
\eta _ { i } = \exp \left( - \frac { 1 } { 2 } \left( \frac { d _ { i } } { \sigma } \right) ^ { 2 } \right) , \qquad \eta _ { i } \in [ 0 , 1 ] ,\tag{13}
$$

where σ controls the decay sensitivity. For VR visualization, $\eta _ { i }$ can be mapped to the hand point color $c _ { i } = c _ { \mathrm { b a s e } } + \eta _ { i } ( c _ { \mathrm { h o t } } - c _ { \mathrm { b a s e } } )$ , providing proximity guidance related to object distance. For policy training, we record the full $o _ { t } ^ { p }$ and proximity observation to maintain input consistency.

To obtain structured learning inputs, we group hand samples according to physical hand regions $S _ { k }$ such as the palm, finger roots, and fingertips. Let $I _ { k } = \{ i \ | \ p _ { i } ^ { h } \in S _ { k } \}$ be the set of sample indices belonging to region $S _ { k }$ . The region-level proximity is

$$
\gamma _ { t } ^ { k } = \frac { 1 } { | I _ { k } | } \sum _ { i \in I _ { k } } \eta _ { i } .\tag{14}
$$

The proximity observation at frame t is therefore

$$
o _ { t } ^ { \mathrm { p r o x } } = [ \gamma _ { t } ^ { 1 } , \dots , \gamma _ { t } ^ { K } ] .\tag{15}
$$

We further derive the trajectory-stage label from the palm response:

$$
o _ { t } ^ { \mathrm { t r a j } } = \mathbb { I } ( \gamma _ { t } ^ { \mathrm { p a l m } } > \tau _ { \mathrm { p a l m } } ) ,\tag{16}
$$

where $o _ { t } ^ { \mathrm { t r a j } } = 0$ denotes the approach stage and $o _ { t } ^ { \mathrm { t r a j } } = 1$ denotes the contact-interaction stage. Each demonstration frame is finally stored as $\left( o _ { t } ^ { p } , o _ { t } ^ { \mathrm { p r o x } } , o _ { t } ^ { \mathrm { t r a j } } , s _ { t } , a _ { t } \right)$ , consistent with the notation in the main paper. The concrete input dimensions of scene point clouds, interaction point clouds, auxiliary states, and actions across simulation and real-world environments are summarized in Table 3. In this representation, $o _ { t } ^ { p }$ keeps the complete hand–object geometry, $o _ { t } ^ { \mathrm { p r o x } }$ compresses near-field distances into physically meaningful hand-region responses, and $o _ { t } ^ { \mathrm { t r a j } }$ provides a coarse phase indicator. This separation is used in the following training stage, where the latent observation is explicitly aligned with proximity dynamics rather than only with visible geometry.

## B Training and Inference Details of ProxiDex

ProxiDex is trained in two stages. The first stage learns a proximity-aware dynamics representation from interaction point clouds, robot states, and region-level proximity cues. The second stage freezes this representation and trains a Transformer-based diffusion policy, in which the predicted trajectory-stage probability adaptively modulates the proximity branch. This design allows the policy to emphasize global geometry during approach and exploit local proximity cues during dexterous manipulation. All training and inference are conducted on an Intel i9-14700K CPU and an NVIDIA RTX 4090D GPU.

## B.1 Latent Interaction Dynamics Architecture

Given the interaction point cloud $o _ { t } ^ { p }$ and robot state $s _ { t } ,$ , the observation encoder $E _ { \theta }$ extracts geometric features and fuses them with the state embedding to obtain the observation latent $z _ { t } ^ { o b s }$ . This latent is shared by the forward dynamics model (FDM), inverse differential module (IDM), and trajectory head in Stage 1, and is subsequently used as the global condition for policy generation in Stage 2.

Before learning the interaction dynamics, we independently pretrain a proximity variational autoencoder (VAE) for 100 epochs to compress the region-level proximity observation $o _ { t } ^ { p r o x }$ into a compact latent representation. The encoder parameterizes a Gaussian posterior, while the decoder reconstructs the original proximity observation. In our implementation, the proximity input and latent dimensions are 124 and 16, respectively. The VAE is optimized using a reconstruction loss and a KL-divergence regularizer:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v a e } } = \lambda _ { \mathrm { r e c } } \| \hat { o } _ { t } ^ { p r o x } - o _ { t } ^ { p r o x } \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } \left( q ( z _ { t } ^ { p r o x } \mid o _ { t } ^ { p r o x } ) \| \mathcal { N } ( 0 , I ) \right) , } \end{array}\tag{17}
$$

where $\lambda _ { \mathrm { r e c } } = 1 . 0$ and $\lambda _ { \mathrm { K L } } = 0 . 0 1$ . After pretraining, the VAE is frozen, and its posterior mean $\mu _ { \mathrm { p r o x } } ( o _ { t } ^ { p r o x } )$ is used as a deterministic proximity target for interaction dynamics learning.

The forward dynamics model FDM(·) is implemented as a DiT-style conditional denoising model that predicts future observation features after action execution. Specifically, the future observation latent is perturbed and then denoised conditioned on the current latent and action sequence:

$$
\begin{array} { r l } & { \tilde { z } _ { t + h } ^ { o b s , ( \ell ) } = \sqrt { \bar { \alpha } _ { \ell } } z _ { t + h } ^ { o b s } + \sqrt { 1 - \bar { \alpha } _ { \ell } } \epsilon , } \\ & { \quad \hat { z } _ { t + h } ^ { o b s } = f _ { \theta } \left( \tilde { z } _ { t + h } ^ { o b s , ( \ell ) } , \ell \mid z _ { t } ^ { o b s } , a _ { t : t + h - 1 } \right) , } \end{array}\tag{18}
$$

where $\ell$ denotes the diffusion step and ϵ is Gaussian noise. This forward-prediction objective encourages $z _ { t } ^ { o b s }$ to encode not only object geometry and hand configuration, but also short-horizon interaction dynamics associated with approach, grasping, and contact transfer.

To connect the predicted state transition with local geometric proximity, ProxiDex introduces the inverse differential module IDM(·), implemented as an MLP with layer normalization and nonlinear activations. For one-step prediction, IDM maps the change in observation latent to the next-step proximity latent, which is aligned with the deterministic target provided by the frozen VAE:

$$
\begin{array} { r l } & { \hat { z } _ { t + 1 } ^ { \mathrm { p r o x } } = \mathrm { I D M } \left( \hat { z } _ { t + 1 } ^ { \mathrm { o b s } } - z _ { t } ^ { \mathrm { o b s } } \right) , } \\ & { \hat { \mathcal { L } } _ { \mathrm { i n v } } = \left\| \hat { z } _ { t + 1 } ^ { \mathrm { p r o x } } - \mu _ { \mathrm { p r o x } } \left( o _ { t + 1 } ^ { \mathrm { p r o x } } \right) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{19}
$$

Table 4: Key training and model hyperparameters for the Interaction Dynamics Model.
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Policy and Diffusion Settings</td><td></td><td></td></tr><tr><td>Policy Horizon</td><td> $T$ </td><td>16</td></tr><tr><td>Observation Horizon</td><td> $T _ { o }$ </td><td>4</td></tr><tr><td>Action Horizon</td><td> $T _ { a }$ </td><td>8</td></tr><tr><td>Diffusion Training Timesteps Dynamics and Proximity Modeling</td><td> $K$ </td><td>100</td></tr><tr><td>Point-Cloud Feature Dimension</td><td> $d _ { p }$ </td><td>128</td></tr><tr><td>Future Prediction Steps Proximity Latent Dimension</td><td> $H _ { \mathrm { d y n } }$ </td><td>2</td></tr><tr><td>Action Embedding Dimension</td><td> $d _ { \mathrm { p r o x } }$   $d _ { a }$ </td><td>16</td></tr><tr><td>FDM Hidden Dimension</td><td></td><td>16</td></tr><tr><td>Proximity VAE Pretraining Epochs</td><td> $d _ { \mathrm { f d m } }$ </td><td>256</td></tr><tr><td>Action Dropout / Negative Sampling Probability</td><td></td><td>100</td></tr><tr><td></td><td></td><td>0.1 / 0.1</td></tr><tr><td>Stage-1 Loss Weights</td><td></td><td> $\lambda _ { \mathrm { v } } = \lambda _ { \mathrm { f } } = \lambda _ { \mathrm { c } } = \lambda _ { \mathrm { t } } = 1 . 0$ </td></tr><tr><td>Stage-2 Loss Weights</td><td></td><td> $\lambda _ { \mathrm { d i f f } } = 1 . 0 , \lambda _ { \mathrm { g a t e } } = 0 . 0 0 1$ </td></tr><tr><td>Reconstruction / KL Weights</td><td></td><td> $\lambda _ { \mathrm { { r e c } } } = 1 . 0 , \lambda _ { \mathrm { { K L } } } = 0 . 0 1$ </td></tr><tr><td>Optimization Settings</td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td></td></tr><tr><td>Training Epochs</td><td></td><td>AdamW</td></tr><tr><td>Batch Size</td><td></td><td>301</td></tr><tr><td></td><td></td><td>64</td></tr><tr><td>Learning Rate</td><td></td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight Decay</td><td></td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Learning Rate Schedule</td><td></td><td>Cosine</td></tr></table>

This alignment encourages the observation-latent transition to preserve action-induced changes in hand–object proximity and enables IDM to infer implicit proximity information from visual and robot-state dynamics.

In parallel, the trajectory-stage head predicts the interaction probability $p _ { t } ^ { \mathrm { i n t } }$ from $z _ { t } ^ { o b s }$ and is supervised by $o _ { t } ^ { \mathrm { t r a j } }$ , as defined in Appendix A.3. Together, forward prediction, proximity-latent alignment, and trajectory-stage classification produce a physically meaningful interaction representation for downstream control.

As summarized in Table 4, the point-cloud feature and proximity latent dimensions are 128 and 16, respectively. The dynamics branch uses a hidden dimension of 256 and a short prediction horizon $H _ { \mathrm { d y n } } = 2$ , allowing it to capture immediate action-induced interaction changes while limiting longhorizon prediction errors. Stage-1 training spans 301 epochs in total. We first pretrain the proximity VAE for 100 epochs. The VAE is then frozen, and the observation encoder, FDM, IDM, and trajectory head are trained for the remaining 201 epochs. The forward-dynamics, proximity-alignment, and trajectory-stage losses are assigned equal weights.

## B.2 Trajectory-Adaptive Proximity Policy Learning

The Stage-2 policy backbone is a Transformer-based diffusion model. The policy is trained using VAE-derived proximity latents, whereas IDM-estimated proximity latents are used during online deployment. It takes a noised action sequence as input and denoises it under two types of conditions: the observation latent $z _ { t } ^ { o b s }$ , which provides global information such as target geometry, object pose, and hand configuration; and the proximity latent $z _ { t } ^ { p r o x }$ , which provides local distance cues between hand regions and the object surface. To avoid overusing local feedback before contact, we gate the proximity branch according to the interaction probability $p _ { t } ^ { i n t }$ predicted by the trajectory-stage head in Stage 1:

$$
\begin{array} { r } { \bar { z } _ { t } ^ { p r o x } = g ( p _ { t } ^ { i n t } ) z _ { t } ^ { p r o x } , \qquad g ( p ) = 1 - \alpha + \alpha p , \qquad \alpha \in [ 0 , 1 ] . } \end{array}\tag{20}
$$

Table 5: Key training and model hyperparameters for the Proximity Policy.
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Policy and Diffusion Settings</td><td></td><td></td></tr><tr><td>Policy Horizon</td><td> $T$ </td><td>16</td></tr><tr><td>Observation Horizon</td><td> $T _ { o }$ </td><td>4</td></tr><tr><td>Action Horizon</td><td> $T _ { a }$ </td><td>8</td></tr><tr><td>Diffusion Training Timesteps</td><td> $K$ </td><td>100</td></tr><tr><td>Inference Denoising Steps</td><td></td><td>10</td></tr><tr><td>Proximity Policy Modeling Point-Cloud Feature Dimension</td><td> $d _ { p }$ </td><td>128</td></tr><tr><td>Proximity Latent Dimension</td><td> $d _ { \mathrm { p r o x } }$ </td><td>16</td></tr><tr><td>Interaction Logit Dimension</td><td> $C _ { \mathrm { i n t } }$ </td><td>1</td></tr><tr><td>Gate Warm-up Steps</td><td> $N _ { \mathrm { w a r m } }$ </td><td>2,000</td></tr><tr><td>Gate Temperature Range</td><td></td><td></td></tr><tr><td>Temperature Annealing Steps</td><td>T</td><td> $2 . 0  0 . 7$ </td></tr><tr><td></td><td> $N _ { \tau }$ </td><td>50,000</td></tr><tr><td>Gate Ramp-up Steps</td><td> $N _ { \alpha }$ </td><td>20,000</td></tr><tr><td>Stage-2 Loss Weights</td><td></td><td> $\lambda _ { \mathrm { d i f f } } = 1 . 0 , \lambda _ { \mathrm { g a t e } } = 1 0 ^ { - 3 }$ </td></tr><tr><td>Transformer Backbone</td><td></td><td></td></tr><tr><td>Transformer Layers</td><td>L</td><td>8</td></tr><tr><td>Attention Heads</td><td> $H$ </td><td>8</td></tr><tr><td>Embedding Dimension</td><td> $d _ { \mathrm { e m b } }$ </td><td>256</td></tr><tr><td>Optimization Settings</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td>AdamW</td></tr><tr><td>Training Epochs</td><td></td><td>301</td></tr><tr><td>Batch Size</td><td></td><td>128</td></tr><tr><td>Learning Rate</td><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight Decay</td><td></td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr></table>

During the first 2,000 training steps, we set $p _ { t } ^ { i n t } = 0 . 5$ to stabilize policy optimization. After this warm-up period, the interaction probability is computed as $p _ { t } ^ { i n t } = \mathrm { s i g m o i d } ( c _ { t } / \tau )$ , where $c _ { t }$ is the predicted interaction logit. The temperature τ is annealed from 2.0 to 0.7 over 50,000 steps, while α is linearly increased from 0 to 1 over 20,000 steps after warm-up. This schedule gradually introduces proximity modulation without abruptly changing the policy condition.

When $p _ { t } ^ { i n t }$ is low, the policy relies more on $z _ { t } ^ { o b s }$ for global approach. As the interaction probability increases, the proximity branch is strengthened, enabling the policy to refine fingertip closure, contact maintenance, and release timing. To prevent the gate probability from prematurely saturating, we introduce a negative binary-entropy regularizer:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g a t e } } = \mathbb { E } \left[ p _ { t } ^ { \mathrm { i n t } } \log \left( p _ { t } ^ { \mathrm { i n t } } + \epsilon \right) + \left( 1 - p _ { t } ^ { \mathrm { i n t } } \right) \log \left( 1 - p _ { t } ^ { \mathrm { i n t } } + \epsilon \right) \right] , } \\ { \mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { g a t e } } \mathcal { L } _ { \mathrm { g a t e } } . } \end{array}\tag{21}
$$

where ${ \mathcal { L } } _ { \mathrm { d i f f } }$ denotes the diffusion denoising loss and ϵ is a small constant for numerical stability.

For temporal policy generation, the proximity condition is projected into the policy embedding space and shared across the predicted action sequence, while the observation latent remains available as stable global context. This conditioning scheme preserves global planning ability while allowing the interaction probability to adaptively regulate the contribution of local proximity information.

The detailed hyperparameters for proximity-adaptive policy learning are summarized in Table 5. Overall, Stage 2 follows the same temporal window configuration as Stage 1 while using fewer denoising steps during inference for efficient online deployment. Since the Stage-1 representation is frozen, optimization mainly focuses on the diffusion policy and its trajectory-adaptive proximity conditioning.

Algorithm 1 Dynamics-Consistency Guided Policy Inference   
Input: Multimodal observations $\{ ( P _ { t } , S _ { t } ) \} _ { t = 0 } ^ { T + 1 }$ , encoder E, forward dynamics model FDM, diffusion policy $\pi _ { \theta } ,$ dynamics threshold   
$\tau _ { \mathrm { d y n } } ,$ and scaling factor α   
Output: Executed action sequence $\left\{ a _ { t } \right\} _ { t = 1 } ^ { T }$   
1: $z _ { 0 } ^ { \mathrm { o b s } } \gets E ( P _ { 0 } , S _ { 0 } )$   
2: $z _ { 0 } ^ { \mathrm { i n } } \gets z _ { 0 } ^ { \mathrm { o b s } }$   
3: Mode ← Normal   
4: for t = 1 to T do   
/\* Observation encoding $^ { * }$   
5: $\boldsymbol { z } _ { t } ^ { \mathrm { o b s } } \gets E ( P _ { t } , S _ { t } )$   
if Mode = Normal then   
$z _ { t } ^ { \mathrm { i n } } \gets z _ { t } ^ { \mathrm { o b s } }$   
else   
$\boldsymbol { z } _ { t } ^ { \mathrm { i n } } \gets \hat { \boldsymbol { z } } _ { t } ^ { \mathrm { o b s } }$   
10: end if   
11: $\Delta z _ { t } \gets z _ { t } ^ { \mathrm { i n } } - z _ { t - 1 } ^ { \mathrm { i n } }$   
/\* Current proximity inference and adaptive gating \*/   
12: $\hat { z } _ { t } ^ { \mathrm { p r o x } ^ { \star } } \gets \mathrm { I D M } ( \overline { { \Delta z _ { t } } } )$   
13: $p _ { t } ^ { \mathrm { i n t } }  h ( z _ { t } ^ { \mathrm { i n } } )$   
14: $\bar { z } _ { t } ^ { \mathrm { p r o x } } \gets \left( 1 - \alpha + \alpha p _ { t } ^ { \mathrm { i n t } } \right) \hat { z } _ { t } ^ { \mathrm { p r o x } }$   
/\* Action generation and latent dynamics prediction \*/   
15: $a _ { t } \gets \pi _ { \theta } \left( z _ { t } ^ { \mathrm { i n } } , \bar { z } _ { t } ^ { \mathrm { p r o x } } \right)$   
16: Execute a via the robot controller   
17: $\hat { z } _ { t + 1 } ^ { \mathrm { o b s } } \gets \mathrm { F D M } \left( z _ { t } ^ { \mathrm { i n } } , a _ { t } \right)$   
/\* Dynamics monitoring and mode switching \*/   
18: Receive the next observation $( P _ { t + 1 } , \stackrel { \sim } { S } _ { t + 1 } )$   
19: $z _ { t + 1 } ^ { \mathrm { o b s } } \gets E ( P _ { t + 1 } , S _ { t + 1 } )$   
20: $e _ { t + 1 } ^ { \mathrm { d y n } } \gets \left. z _ { t + 1 } ^ { \mathrm { o b s } } - \hat { z } _ { t + 1 } ^ { \mathrm { o b s } } \right. _ { 2 }$   
21: $\mathbf { i f } \ e _ { t + 1 } ^ { \mathrm { d y n } } \leq \tau _ { \mathrm { d y n } }$ then   
22: Mode ← Normal   
23: $z _ { t + 1 } ^ { \mathrm { i n } }  z _ { t + 1 } ^ { \mathrm { o b s } }$   
24: else   
25: Mode ← Dream   
26: $z _ { t + 1 } ^ { \mathrm { i n } } \gets \hat { z } _ { t + 1 } ^ { \mathrm { o b s } }$   
27: end if   
28: end for

## B.3 Dynamics-Consistency Guided Policy Inference

During online inference, ProxiDex places policy generation and FDM prediction in the same loop. The system encodes the real-time interaction point cloud $o _ { t } ^ { p }$ and robot state $s _ { t }$ into $z _ { t } ^ { o b s }$ , estimates $\hat { z } _ { t } ^ { \mathrm { p r o x } }$ through IDM, and obtains $p _ { t } ^ { \mathrm { i n t } }$ from the trajectory head. The diffusion policy then generates an action sequence and executes actions in a receding-horizon manner.

After action execution, FDM predicts the next latent state and compares it with the encoded real observation:

$$
e _ { t + 1 } ^ { \mathrm { d y n } } = \left. z _ { t + 1 } ^ { o b s } - \mathrm { F D M } ( z _ { t } ^ { o b s } , a _ { t } ) \right. _ { 2 } .\tag{22}
$$

If $e _ { t + 1 } ^ { \mathrm { d y n } } \leq \tau _ { \mathrm { d y n } } .$ , the system continues closed-loop control with real-time perception. If the deviation exceeds the threshold, the system treats the current visual feedback as potentially affected by occlusion, pose drift, or short-term tracking failure, and temporarily switches to dynamics takeover, using autoregressive FDM latents to maintain action generation. Once the deviation returns below the threshold, the system switches back to real observations and re-aligns the state.

This mechanism makes FDM both a training-time dynamics supervisor and an inference-time physical consistency checker. In deployment, FoundationPose++ provides object-pose updates at around 10 Hz, the Proximity Policy runs at about 28 Hz, and the learned world model runs at about 20 Hz. These runtime rates are sufficient for real-time closed-loop inference, enabling dynamics takeover to maintain action generation from predicted latents when visual pose tracking becomes temporarily unreliable. Algorithm 1 summarizes the procedure.

![](images/926759159c17dc97484189b3a2935a0f35a312f6df3400a3021060012048d00f.jpg)  
Figure 11: Visualization of the simulation tasks. We evaluate ProxiDex on ten dexterous manipulation tasks, including four DexArt tasks, three Adroit tasks, and three custom tasks built in IsaacLab.

## C Simulation Experiments

## C.1 Simulation Setup and Task Description

We evaluate ProxiDex on Adroit, DexArt, and three self-designed IsaacLab tasks, as illustrated in Fig. 11. Adroit covers classic dexterous manipulation tasks requiring fine finger coordination, such as Hammer, Door, and Pen, while DexArt provides articulated-object interactions closer to daily manipulation, including Laptop, Faucet, Toilet, and Bucket. Since the simulator directly provides object poses and robot joint states, we can reliably synthesize occlusion-free interaction point clouds o<sup>p</sup> and compute region-level proximity observations o<sup>prox</sup>. The interaction point clouds are visualized in Fig. 21, showing that they preserve active hand–object regions throughout manipulation.

The self-designed tasks are defined as follows:

• Box. The robot grasps a box-like object with multiple fingers and places it into a basket. This task evaluates grasp stability and collision-free transport around the basket rim.

• Driller. The robot side-grasps a drill and moves the drill tip above a target hole. This task requires stable tool orientation and accurate alignment after long-range motion.

• Banana. The robot pinches a banana-like object using the thumb and index finger and places it into a bowl. This task is sensitive to fingertip closure timing, contact location, and holding stability.

Together, these tasks cover rigid-object grasping, articulated-object manipulation, tool alignment, and fine fingertip pinching. The implementation details below keep the data split and evaluation protocol consistent across all methods, so that the comparisons isolate the effect of interaction point clouds, proximity features, and dynamics modeling.

## C.2 Simulation Implementation Details

For Adroit and DexArt, demonstrations are generated by reinforcement-learning experts, while the self-designed IsaacLab demonstrations are collected through VR teleoperation. All methods use the same demonstration split, action horizon, and evaluation protocol. We report the mean success rate and standard deviation across repeated evaluations. Fig. 11 visualizes the ten simulation tasks before the quantitative comparison, and Table 6 provides the corresponding task-level results.

The detailed task-level results in Table 6 further support the overall advantage of ProxiDex. ProxiDex achieves the best ten-task average of 83.9%, outperforming DP3, ManiFlow, AFRO, and Cord-ViP by 12.1, 9.7, 7.0, and 2.9 percentage points, respectively. The improvement is consistent across Adroit, DexArt, and the self-designed tasks, indicating that proximity-aware dynamics benefit both standard dexterous benchmarks and customized tabletop manipulation scenarios.

Table 6: Simulation results on dexterous manipulation benchmarks. We evaluate all methods on Adroit, DexArt, and three self-designed IsaacLab tasks.
<table><tr><td rowspan="2">Algorithm\Tasks</td><td colspan="3">Adroit</td><td colspan="4">DexArt</td><td colspan="3">Self-Designed</td><td rowspan="2">Avg.</td></tr><tr><td>Hammer</td><td>Door</td><td>Pen</td><td>Laptop</td><td>Faucet</td><td>Toilet</td><td>Bucket</td><td>Box</td><td>Driller</td><td>Banana</td></tr><tr><td>DP3</td><td>100.0±0.0</td><td>76.7±4.7</td><td>56.7±2.6</td><td>89.7±0.9</td><td>41.7±0.5</td><td>79.7±0.9</td><td>31.3±0.5</td><td>90.3±3.2</td><td>74.3±2.6</td><td>77.8±1.2</td><td>71.8±1.7</td></tr><tr><td>ManiFlow</td><td>100.0±0.0</td><td>80.3±1.2</td><td>55.5±5.8</td><td>93.0±1.6</td><td>45.0±3.6</td><td>79.9±3.3</td><td>35.3±2.1</td><td>94.7±2.1</td><td>78.6±1.4</td><td>79.8±2.3</td><td>74.2±2.3</td></tr><tr><td>AFRO</td><td>96.0±2.8</td><td>83.8±3.4 72.3±2.1</td><td></td><td>86.3±2.5</td><td>42.3±1.8</td><td>82.2±3.4</td><td>48.0±2.3</td><td>97.2±1.3</td><td>79.4±1.5</td><td>81.4±1.8</td><td>76.9±2.3</td></tr><tr><td>CordViP</td><td>96.0±2.1</td><td>84.7±1.7 76.2±3.2</td><td></td><td>92.6±2.1</td><td>56.5±1.3</td><td>81.0±2.2</td><td>57.0±1.6</td><td>96.3±2.0</td><td>85.3±1.1</td><td>84.7±1.6</td><td>81.0±1.9</td></tr><tr><td>ProxiDex (Ours)</td><td>100.0±0.0</td><td>86.5±1.9 84.3±2.4</td><td></td><td>93.0±1.2</td><td>53.4±2.1</td><td>84.3±2.3</td><td>64.0±3.2</td><td>96.7±1.6</td><td>89.2±2.0</td><td>87.3±1.5</td><td>83.9±1.8</td></tr></table>

![](images/e4a433206f13202e50cf435e66bf306edf4ab531fc18ebb225311a6b8b45903d.jpg)  
Figure 12: Visualization of hand–object interactions on the self-designed tasks, compared with typical failure cases from existing baselines.

A closer look at the task results shows that the gains are more evident in contact-rich tasks requiring sustained fingertip adjustment, such as Pen, Bucket, Driller, and Banana. These tasks involve continuous hand-object proximity regulation after initial contact, where the proposed proximity representation and adaptive policy modulation provide more informative feedback than geometry-only baselines. Although CordViP performs slightly better on Faucet, ProxiDex achieves the best aggregate performance by improving tasks with more complex contact-state transitions.

Fig. 12 provides qualitative evidence complementary to the quantitative results in Table 6, comparing representative executions on the self-designed tasks with typical failure cases of the baselines. Specifically, in the Box task, baseline methods often rely on a small distal contact region and can lose grasp stability during transport, whereas ProxiDex adjusts finger closure after approaching the target. In Driller, visible-point-cloud methods struggle to maintain the relative relation between the drill and the hole under occlusion; ProxiDex improves local adjustment near the hole through proximity-difference constraints. In Banana, ProxiDex forms more suitable fingertip contact re gions, showing the value of region-level proximity for fine pinching.

## D Additional Experiments and Analysis

## D.1 Teleoperation System Comparison

To evaluate the benefit of immersive proximity feedback, we invited 20 participants to experience the three teleoperation systems shown in Fig. 13(c). After 10 minutes of training, participants collected 30 demonstrations under each setting. As shown in Fig. 13(a)(b), most participants found proximity feedback helpful, and 75% preferred ProxiDex. ProxiDex also reduces the average task completion time to 26.6 ± 5 s, outperforming the other two systems. This advantage mainly comes from more direct interaction feedback: compared with Vanilla VR, which is susceptible to hand–object occlusion, ProxiDex visualizes hand–object distances and spatial relations through color-coded proximity point clouds, enabling timely adjustment of grasp position and finger closure and thereby improving teleoperation experience and demonstration-collection efficiency.

![](images/54cbc78705a7ebee34a664905862efab4a127a7394c34b5c02db34d5f81e9ef2.jpg)

![](images/b977b8992c76779c214d68b7e46c4c33d6c24207e570767bd271ea3645821512.jpg)  
Figure 13: User study and teleoperation-system comparison. (a) User experience and system preference. (b) Task completion time. (c) Overview of the three teleoperation systems.

Table 7: Ablation study of key components on representative dexterous manipulation tasks.
<table><tr><td>Components</td><td>Door</td><td>Pen</td><td>Laptop</td><td>Faucet</td><td>Toilet</td><td>Bucket</td><td>Avg.</td></tr><tr><td>w/o Prox</td><td> $7 6 . 7 \pm 2 . 3$ </td><td> $7 9 . 4 \pm 1 . 6$ </td><td> $8 8 . 3 { \pm } 1 . 5 $ </td><td> $4 7 . 9 { \pm } 1 . 5 $ </td><td> $\overline { { 8 0 . 7 \pm 1 . 6 } }$ </td><td> $5 0 . 6 { \pm } 1 . 3$ </td><td> $7 0 . 6 \pm 1 . 6$ </td></tr><tr><td>w/o Int. PC</td><td> $8 0 . 3 { \pm } 1 . 2 $ </td><td> $7 5 . 9 2 2 . 4 $ </td><td> $8 6 . 1 \pm 1 . 4$ </td><td> $4 8 . 7 { \pm } 1 . 4 $ </td><td> $8 0 . 4 \pm 1 . 5$ </td><td> $4 8 . 9 { \pm } 2 . 0 \ $ </td><td> $7 0 . 1 \pm 1 . 7$ </td></tr><tr><td>w/o FDM</td><td> $8 3 . 8 { \pm } 2 . 2 $ </td><td> $8 0 . 3 { \pm } 1 . 7 $ </td><td> $8 9 . 6 \pm 1 . 5$ </td><td> $5 0 . 5 { \pm } 1 . 7 $ </td><td> $8 1 . 1 { \pm } 1 . 7 $ </td><td> $5 6 . 4 \pm 1 . 8$ </td><td> $7 3 . 6 { \pm } 1 . 8 $ </td></tr><tr><td>w/o IDM</td><td> $8 4 . 7 \pm 1 . 7$ </td><td> $8 0 . 9 { \pm } 2 . 1 $ </td><td> $9 2 . 3 { \pm } 1 . 7 $ </td><td> $5 1 . 8 { \pm } 1 . 3 $ </td><td> $8 2 . 0 { \pm } 1 . 8 $ </td><td> $6 0 . 1 \pm 1 . 5$ </td><td> $7 5 . 3 { \pm } 1 . 7 $ </td></tr><tr><td>w/o Closed-Loop</td><td> $8 5 . 2 { \pm } 1 . 2 $ </td><td> $8 2 . 5 { \pm } 1 . 2 $ </td><td> $9 2 . 8 { \pm } 1 . 3 $ </td><td> $5 2 . 6 { \pm } 1 . 1$ </td><td> $8 3 . 4 \pm 2 . 0 $ </td><td> $6 2 . 3 { \pm } 1 . 6 $ </td><td> $7 6 . 5 { \pm } 1 . 4 $ </td></tr><tr><td>ProxiDex (Ours)</td><td> ${ \bf 8 6 . 5 \pm 1 . 9 }$  </td><td> ${ \bf 8 4 . 3 \pm 2 . 4 }$ </td><td> ${ \bf 9 3 . 0 { \pm 1 . 2 } }$ </td><td> ${ \bf 5 3 . 4 } \pm 2 . 1$ </td><td> ${ \bf 8 4 . 3 \pm 2 . 3 }$ </td><td> ${ \bf 6 4 . 0 \pm 3 . 2 }$ </td><td> $7 7 . 6 { \pm } 2 . 2 $ </td></tr></table>

## D.2 Ablation Study

Ablation on core components. Table 7 reports the task-level numerical results corresponding to the ablation summary in Fig. 4. The largest average drops occur when geometric interaction inputs are removed: removing interaction point clouds decreases the success rate from 77.6% to 70.1%, while removing proximity cues decreases it to 70.6%, with drops of 7.5 and 7.0 points, respectively. The degradation is especially clear on contact-sensitive tasks: Bucket drops by 15.1 and 13.4 points under these two removals, while Pen drops by 8.4 points without interaction point clouds. These results indicate that complete hand–object geometry and region-level proximity provide complementary cues for contact-rich manipulation.

From the dynamics perspective, removing FDM reduces the average success rate to 73.6%, corresponding to a 4.0-point drop, while removing IDM reduces it to 75.3%, a 2.3-point drop. FDM contributes more strongly to tasks that require predicting near-future contact consequences, whereas IDM improves sensitivity to local proximity changes caused by actions. Removing closed-loop feedback yields 76.5%, a smaller but consistent 1.1-point drop. This pattern suggests that most of the performance gain comes from representation and proximity-aware policy learning, while inferencetime dynamics checking mainly stabilizes execution when perception is temporarily unreliable.

The component-level ablation motivates a separate analysis of the Stage-2 policy architecture: after the interaction representation is fixed, the remaining question is whether the policy backbone can use proximity information at the correct manipulation phase.

Ablation on the policy backbone. Complementing the real-robot backbone ablation in Table 2, Fig. 14 further evaluates the Stage-2 policy backbone in simulation while keeping Stage-1 pretraining unchanged. Replacing the proposed causal Transformer with a U-Net reduces the average success rate by about 3.1 percentage points, since temporal convolutions are less flexible in fusing multimodal tokens across manipulation phases. Replacing it with a standard Transformer reduces performance by about 2.8 percentage points, showing that causal proximity masking and trajectoryadaptive weighting are important for stable contact-aware action generation. These drops are smaller than the $7 . 0 – 7 . 5$ point drops caused by removing proximity cues or interaction point clouds in Table 7, suggesting that the representation provides the primary contact information, while the backbone determines how effectively it is used over time.

![](images/d48ca51bdb3d6305cd46fdfd5c516d094b52a241ee94553db9ddd71301f94663.jpg)  
Figure 14: Policy-backbone ablation study on six Adroit and DexArt tasks. The proposed causal Transformer with adaptive weight converges faster and achieves higher final success rates.

Table 8: Ablation study on the latent dynamics prediction horizon $H _ { \mathrm { d y n } }$
<table><tr><td> $H _ { \mathrm { d y n } }$ </td><td>Door</td><td>Pen</td><td>Laptop</td><td>Faucet</td><td>Toilet</td><td>Bucket</td><td>Avg.</td></tr><tr><td>1</td><td> ${ \bf 8 6 . 8 \pm 1 . 7 }$ </td><td> $8 4 . 0 { \pm } 2 . 2 $ </td><td> $9 2 . 6 { \pm } 1 . 5 $ </td><td> $5 3 . 3 { \pm } 1 . 9 $ </td><td> $8 0 . 7 { \pm } 1 . 6 $ </td><td> $6 4 . 1 { \pm } 1 . 8 $ </td><td> $7 6 . 9 { \pm } 1 . 8 $ </td></tr><tr><td>2</td><td> $8 6 . 5 { \pm } 1 . 9 $ </td><td> ${ \bf 8 4 . 3 \pm 2 . 4 }$ </td><td> ${ \bf 9 3 . 0 { \pm 1 . 2 } }$ </td><td> $5 3 . 4 \pm 2 . 1$ </td><td> ${ \bf 8 4 . 3 \pm 2 . 3 }$ </td><td> $6 4 . 0 { \pm } 3 . 2 $ </td><td> $7 7 . 6 { \pm } 2 . 2 $ </td></tr><tr><td>4</td><td> $8 5 . 9 { \pm } 2 . 0 \ $ </td><td> $8 3 . 6 \pm 1 . 9$ </td><td> ${ \bf 9 3 . 0 { \pm } 1 . 5 }$ </td><td> ${ \bf 5 3 . 7 \pm 1 . 4 }$ </td><td> $8 1 . 1 { \pm } 1 . 7 $ </td><td> ${ \bf 6 4 . 3 \pm 1 . 9 }$ </td><td> $7 6 . 9 { \pm } 1 . 7 $ </td></tr><tr><td>8</td><td> $8 4 . 7 { \pm } 1 . 1 $ </td><td> $8 2 . 8 { \pm } 1 . 8 $ </td><td> $9 2 . 1 { \pm } 1 . 4 $ </td><td> $5 3 . 4 \pm 1 . 5$ </td><td> $8 2 . 0 { \pm } 1 . 8 $ </td><td> $6 3 . 9 2 1 . 7$ </td><td> $7 6 . 5 { \pm } 1 . 6 $ </td></tr><tr><td>10</td><td> $8 4 . 8 { \pm } 1 . 2 $ </td><td> $8 2 . 6 { \pm } 1 . 3 $ </td><td> $9 1 . 8 { \pm } 1 . 1 $ </td><td> $5 2 . 8 { \pm } 1 . 3 $ </td><td> $8 3 . 4 \pm 2 . 0 $ </td><td> $6 3 . 2 { \pm } 2 . 0 $ </td><td> $7 6 . 4 \pm 1 . 5$ </td></tr></table>

Ablation on the latent dynamics prediction horizon. To examine how the dynamics prediction horizon affects latent dynamics modeling, we vary $H _ { \mathrm { d y n } }$ and report the results in Table 8. $H _ { \mathrm { d y n } } = 2$ achieves the best average success rate of 77.6%, slightly outperforming the other settings. Although the differences are modest, shorter horizons provide limited supervision for contact evolution, while overly long horizons may accumulate prediction errors and shift the dynamics objective toward coarse trajectory fitting. We therefore use $H _ { \mathrm { d y n } } = 2$ by default, as it balances local contact sensitivity and prediction stability.

## D.3 Data Scalability Evaluation

The number of demonstrations directly affects how well an imitation-learning policy covers contact patterns. To evaluate learning efficiency under few-shot and medium-scale data regimes, we vary the number of expert trajectories on six Adroit and DexArt tasks and compare DP3, ManiFlow, AFRO, CordViP, and ProxiDex. The demonstration sizes are $N \in \{ 5 , 1 0 , 2 0 , 3 0 , 5 0 , 1 0 0 \}$ , and all other training settings remain unchanged.

As shown in Fig. 15, overall success rates improve as the number of demonstrations increases, but different methods exhibit different growth rates and saturation levels. DP3 and ManiFlow are unstable with limited data, suggesting that policies based only on visible scene point clouds require more samples to cover contact establishment and object-pose variation. AFRO improves sample efficiency through dynamics pretraining, but because it lacks explicit proximity supervision, its gain remain limited on strongly contact-driven tasks such as Bucket and Faucet. CordViP improves low data performance through interaction point clouds, showing that recovering occlusion-free hand object geometry is critical for few-shot learning.

Number of Expert Demonstrations  
![](images/7404abee7b16ac9386b78a56460db7a67eb736b2f23700bd442d331c78e60438.jpg)

![](images/1bf9f21f2b2137f291161be3b5f6a67f5c31e6da1c0535b167af61fd5beab152.jpg)

![](images/d20d64d1ba9aac5668739cc0826d362a79fc97b02fb5463b5823d1a6a7bea0fd.jpg)

![](images/4bb7ad7c0d04dc8ecd9eb97c1b88833cbdf7fa2fbe78d50dcd7354957c5164d3.jpg)

![](images/d535f5949b66b89bf661d566e4572a04b3d96ff8683744ad1fae9cebccafc13f.jpg)

![](images/82be5993a1fd47a63c2b4ec537ac25e3c464d1314962397c8e335e2331b18736.jpg)

Figure 15: Scalability analysis with varying numbers of expert demonstrations on Adroit and DexArt tasks. ProxiDex consistently outperforms the baselines from 5 to 100 demonstrations, demonstrating scalable and data-efficient learning from collected demonstrations.  
![](images/5c60d37ba68bfe1d91c9da53831c69a743a7f7dc058732b1a11d35a0939e3473.jpg)

![](images/dc052192018e1fb9052b58e9d8678445349efb786d3fc3651232cef8dad76f5a.jpg)

![](images/d1b723a503b1b7740e159378116500891e7651c249490489b3b9e563d57c28ad.jpg)

![](images/358b1e4263df8ed5154adc17fae340021a14c44c5f5f54996d024b78161aa52a.jpg)

![](images/d31ff5c3b9140d0bcc15e59d5b2f0f0746a8a8f5232517cee8c141d0cb792f87.jpg)

![](images/c62e80a3a76cf143d4125f2bab77ee99d287cb511be9244e661e5b35a8e1dbee.jpg)

![](images/d379c449182bbf81e4aaefb744dd7f05548f644acde96bdef3e8c1ff80b3779b.jpg)  
Figure 16: Attention weights visualization in simulation. We visualize the trajectory probability and averaged Transformer attention weights of point-cloud and proximity tokens across representative Adroit and DexArt tasks.

ProxiDex achieves the highest average success rate across data scales. With 5, 10, 20, 30, 50, and 100 demonstrations, its six-task average success rates are approximately 11.5%, 40.0%, 64.0%, 77.5%, 80.0%, and 81.8%, respectively. The largest marginal gains occur before 30 demonstrations: the average increases by about 28.5 points from 5 to 10 demonstrations, 24.0 points from 10 to 20, and 13.5 points from 20 to 30. After 30 demonstrations, the curve begins to saturate, gaining only 4.3 points when moving from 30 to 100 demonstrations. Thus, 30 demonstrations already recover about 94.7% of the 100-demonstration performance, indicating that proximity representation provides a strong structured inductive bias for learning finger closure, contact maintenance, and release timing from limited data.

## D.4 Attention Visualization in Simulation

We further visualize the attention weights in simulation environments, as shown in Fig. 16. The trajectory probability usually rises rapidly before contact or during the early contact-establishment stage, indicating that the policy can identify the transition from approach to manipulation. Pointcloud attention mainly reflects external geometric observations and contributes more to estimating the target pose, hand–object configuration, and reachability, so it is more active during early motion and pose adjustment. In contrast, proximity attention remains smoother and acts as a near-field interaction modulation signal, constraining hand–object distance, contact stability, and manipulation robustness after contact becomes likely.

This visualization is consistent with the quantitative ablations above: the point-cloud branch provides global scene context, while the proximity branch becomes most useful when the task requires local contact refinement. Along with the change of trajectory probability, the policy adaptively balances attention between point-cloud geometry and proximity cues, leading to more stable dexterous control during contact establishment and subsequent manipulation.

## D.5 Dream-Mode Robustness Analysis

To further characterize the stability range and error accumulation of Dream Mode under visual interruption, we measure end-effector pose deviations from the no-occlusion trajectory under controlled camera occlusions. As shown in Fig. 17, a 2-s occlusion causes only a transient deviation of about 1 cm, after which the system gradually realigns once visual tracking recovers. In contrast, under a 5-s occlusion, prediction error accumulates continuously: the deviation grows slowly at first, but increases rapidly after about 3.5 s to around 3 cm and further reaches about

![](images/bf756d099a65bb62424e4339fe5f92926bd355ade8582233d511ee6d987f69cd.jpg)  
Figure 17: Dream-Mode robustness under controlled visual occlusion.

5 cm, eventually causing arm drift and stalling. This indicates a reliable open-loop prediction horizon of approximately 3.5 s, which is sufficient for brief perception interruptions but not prolonged visual loss. This trend is also consistent with Fig. 5, where recovery performance decreases as occlusion duration increases. During inference, FDM evaluates latent-dynamics consistency at 20 Hz, and $\tau _ { \mathrm { d y n } }$ is set to the observation-feature discrepancy that first corresponds to a 1-cm end-effector pose deviation, enabling switching between normal visual feedback and dynamics takeover.

## D.6 Pose-Tracking Analysis under Motion Speed

The preceding Dream-Mode robustness analysis (Sec. D.5) shows that short-term hand–object occlusion can be partially handled through latent prediction. Beyond occlusion, object motion speed is another important factor affecting the quality of interaction-pointcloud estimation. To evaluate this effect, we replay training trajectories at different end-effector speeds and measure the object pose-tracking error (ADD) and point-cloud error (Chamfer distance), both in millimeters. As shown in Fig. 18, both errors remain low up to $0 . 8 \mathrm { m } / \mathrm { s }$ , where the pose and point-cloud errors are approximately 8 mm and 6 mm, respectively. Beyond $0 . 8 \mathrm { m } / \mathrm { s } ,$ , both errors increase rapidly and approach 50 mm at 1 $\mathrm { . . 2 m / s . }$ , indicating degraded rotation estimation and point-cloud alignment under fast motion. Since proximity estimation relies on accurate hand–object relative geometry, such tracking errors directly reduce the reliability of proximity cues. Therefore, the current system is better suited to moderate-speed contact-rich manipulation, while highly dynamic motions remain limited by realtime pose-estimation accuracy.

![](images/eeeade82123633597b7c4ddfed4a08ae921549fd230f02a220648c781d6ee8a5.jpg)  
Figure 18: Pose and point-cloud errors at different speeds.

![](images/80fe0be85fa846acec590acf71a5955b51bddddf47f25bdf47fd2009cd1151d9.jpg)  
Figure 19: Real-world platform and evaluation objects. We use a RealMan RM75B arm with a 6-active-DoF dexterous hand and a fixed Intel RealSense L515 camera, and evaluate on seen and unseen objects with diverse shapes and sizes.

## E Real-World Experiments

## E.1 Experimental Setup and Task Description

Real-robot experiments evaluate ProxiDex under real perception noise, cross-object variation, and manipulation disturbances. The platform consists of a Realman RM75B 7-DoF arm, an Intel RealSense L515 RGB-D camera, and two dexterous hands, as summarized in Fig. 19. A 6-DoF CasBot-P1L hand is used for standard tabletop tasks, while a 20-DoF Wuji Hand V1 is used for more contact-rich manipulation. The policy must control not only the global arm pose but also finger closure timing and local contact state. The external RGB-D camera provides object pose tracking, and the reconstructed object point cloud is combined with robot forward kinematics to generate the hand-object interaction point cloud.

The three real-world tabletop tasks are Pick, Pinch, and Sweep. Pick requires the robot to grasp a tabletop object with multiple fingers and place it into a basket, testing grasp stability, transport, and release timing. Pinch requires the robot to form a small-area thumb-index contact and transfer the target object to the desired location, emphasizing fingertip alignment and stable closure. Sweep requires the robot to push a target object into a dustpan, grasp the dustpan handle, and pour the object into a container, involving pushing, tool grasping, and secondary contact transfer. In addition, Wuji Hand V1 is used for two contact-rich tasks involving sustained finger–object interaction and object reorientation. Together, these tasks cover whole-hand grasping, precise fingertip contact, and sustained in-hand interaction.

Demonstrations are collected using the VR teleoperation system described in Appendix A.1, with the operator controlling the arm and hand through Quest 3S. For each task, we collect 50 highquality demonstrations while varying object initial positions, orientations, and some object types to improve adaptation to different geometric configurations. During point-cloud visualization, object pose estimates may exhibit mild jitter. This perturbation does not break action learning; instead, it increases observation variation in the training data and makes both the policy and the dynamics checker more robust to mild perception errors in deployment.

Fig. 22 visualizes real-world executions and their reconstructed interaction point clouds. The proximity responses remain concentrated near active hand–object contact regions, such as the grasping fingers in Pick, the thumb-index contact in Pinch, and the tool-object contact transfer in Sweep. This qualitative behavior links the real-world setup back to the geometric proximity representation in Appendix A.3 and motivates the evaluation protocol below.

## E.2 Evaluation Protocol

We use three test settings to evaluate generalization and robustness. In Distribution uses the same objects and similar scene configurations as data collection, measuring task completion under standard conditions. Unseen Objects uses objects that do not appear in training and differ in shape, size, or surface appearance, testing generalization to object geometry changes. Perturbation intro duces complex tabletop backgrounds, clutter, short-term visual occlusion, or mild pose perturbations during execution, evaluating robustness under unstable perception.

Success criteria consider not only the final object location, but also contact stability and pose consistency during execution. For Pick, the robot must stably grasp the target and place it into the basket. For Pinch, it must form a stable thumb-index pinch and transfer the target to the specified area. For Sweep, it must sweep the target into the dustpan and complete pouring into the target container by grasping the dustpan. We evaluate ProxiDex under three test settings: standard execution, cross-object generalization, and disturbance robustness.

## E.3 Failure Case Analysis

The results above collectively demonstrate the accuracy and robustness of ProxiDex, while Fig. 20 illustrates the remaining limitations of ProxiDex in fine-grained manipulation. Although ProxiDex achieves a practical balance between low-cost data collection and real-world deployment, its current proximity representation is still mainly geometry-driven and cannot fully capture precise fingertip force distribution. As shown in the zoomed failure case, the middle finger and thumb may occasionally fail to align symmetrically with the object during closure, resulting in unbalanced contact forces. In addition, during the cup-pinching task, the cup may exhibit slight shaking during detachment and transfer. This motion, together with partial hand-object occlusion, can reduce the accuracy and temporal stability of object pose estimation, thereby perturbing the interaction point cloud and affecting the stability of proximity-based feedback.

![](images/bbf1a38e85ef5802020280ad3ca5393d730b953c0ffaf228b22232b0dbaeda24.jpg)  
Figure 20: Failure cases under fine interaction. Finger alignment may occasionally be inaccurate, and object pose estimation may show mild jitter during contact-rich manipulation.

Future work can incorporate data gloves or lightweight tactile sensors to obtain higher-precision fingertip contact information, and use contrastive learning to align it with the geometric proximity representation proposed in this work. This would preserve the low-cost and cross-platform advantages of ProxiDex while improving force awareness and finger coordination in fine manipulation.

![](images/c1cc7f03225e893937846cb119e9f3fd2076484163d7b7e7730fd69664b1eafe.jpg)  
Figure 21: Visualization of interaction point clouds in simulation. Across Adroit, DexArt, and selfdesigned tasks, the processed hand–object point clouds preserve fine-grained interaction geometry, with red regions indicating stronger proximity responses along the task progress.

![](images/ad9b22d38654a545d94e80263ac56789ebb022e2cce48416248ca164e6fc5930.jpg)  
Figure 22: Visualization of real-world executions and the corresponding interaction point clouds. The reconstructed point-cloud sequences closely match the camera observations, while the colorcoded proximity responses highlight local hand–object interaction regions during manipulation.