# Underwater $\mathrm { C } ^ { 3 } { \mathrm { - } } \mathrm { J } \mathrm { E } \mathrm { P } \mathrm { A }$ : An Object-Centric Cross-View World Model for ROV Salvage

Yuncong Yang, Jinlong Li, Yulong Xue, Feng Wu, Chunwen Zhang, Lei Qiao, and Xuyang Wang

Abstract—We present Underwater C<sup>3</sup>-JEPA (cross-view, control-conditioned, context-extended), an object-centric multiview predictive world model for near-field heavy-load underwater ROV salvage. Without contact sensors, it predicts in latent space how the task-object state evolves through contact interaction and under the hydrodynamic lag of the vehicle, from synchronized multi-view RGB observations and vehicle control signals. C<sup>3</sup>- JEPA encodes multi-camera observations into task-object and context tokens, fuses cross-camera evidence through held-out-view attention, and directly predicts future states conditioned on control. Weak binding anchors the target and gripper at low annotation cost, while SIGReg sharpens the geometric representation. Experi ments show that the learned representation transfers substantially more task-relevant information to downstream probes than a reconstruction-free latent baseline, while keeping the predictor lightweight. The resulting predictive interface supports modelpredictive-control (MPC) candidate evaluation and imaginedrollout behavior-agent training. Validation on real underwater video shows the same architecture recovering a withheld camera’s object state and staying ahead of persistence, so the recipe transfers beyond simulation.

Note to Practitioners—Recovering a large sunken structure with a work-class ROV is decided in the final meters of approach, where the target is only partially visible and the vehicle answers each thrust command with a visible delay. This paper develops a predictive model for that setting. Rather than reconstructing the scene, it tracks the two task objects that matter, target and gripper, in a compact representation pooled across the vehicle’s cameras, learning directly from ordinary mission recordings how that representation evolves under the pilot’s commands. Because the model is small, it can run in real time and score candidate control sequences before they are executed, which is the capability a conventional autopilot lacks. Three practical consequences follow: no new sensors are required, since training uses the video and telemetry an ROV already logs; losing the target in one camera is tolerated because the other cameras’ evidence is combined; and the recipe transfers to real footage with the simulator out of the loop. The limitations are straightforward: the field trials were open-loop, underwater accuracy is expressed in relative units, and closed-loop success rates at sea have yet to be measured. The same interface should extend to other near-contact subsea tasks—inspection, connector mating, tool deployment—whenever a work-class ROV must act on a partially visible target.

Index Terms—underwater robotics, world models, object-centric learning, multi-view fusion, model predictive control

## I. INTRODUCTION

Deep-sea salvage of large sunken objects is time-critical: contamination grows while a wreck lies on the seabed [1]. Industrial practice rigs slings with a work-class ROV and lifts with a crane—slow, intervention-heavy, damaging to a fragile structure [2], [3], [4]; carrying the gripper on the vehicle removes that stage but forces the ROV to envelope a large, shape-uncertain object from a hydrodynamically lagged base. Our platform is atypical: not a dexterous arm but paired linkage driven underactuated grippers closing downward, trading degrees of freedom for load capacity [5], [6], [7], with coupled, contact-dependent, partly closed-loop kinematics and no contact sensing. Fig. 1 shows the platform and its gripper. The mission is therefore staged for this platform: acquisition, approach, alignment, enveloping contact, lift/hold, and verification, all specific to this vehicle–tool–target combination.

In the decisive near-field window the target can sit at any bearing across the cameras’ combined field of view, occluded by hull and gripper and only partially visible in any single view under turbidity and low contrast [1], [8]. Mapping first and perceiving in a point cloud fits poorly: visual SLAM drifts, loses tracking under backscatter, and needs loop closures a short approach never revisits [9], [10], [11], [12]; structured light is denser but demands short standoff, a calibrated projector, and dwell-and-scan motion [13], [14]. Both approaches reconstruct more than the decision requires.

Control is constrained differently. Real-system learning must succeed from few trials, at control rate, under large but unknown actuation delays [15], [16], and simulation handles this gripper poorly: closed kinematic loops stiffen the constrained dynamics into tiny steps, soft-constraint relaxation, or outright solver failure [17], [18]. Tactile instrumentation is fragile and costly, and needs pressure housing and cabling on a jaw closing on rough wreckage [19], [20]. Neither route internalizes this behavior affordably.

Underwater C<sup>3</sup>-JEPA changes what must be reconstructed, building an object-centric perception domain (OCPD): taskobject classes (target and gripper) are held at full fidelity and forcibly aligned across cameras, while remaining evidence stays in free context slots kept alongside without alignment. We call this routing guided object-centric cross-view fusion (GO-CVF). Because the decomposition fixes which entities receive tokens, fusion fits an ROV budget and new task classes are synthesized from the same weak-anchor route rather than exhaustively reconstructed; anchors are DINOv3-clustering proposals, with human review reserved for ambiguous frames.

A second element provides the physical dynamics in two roles: offline, the control-conditioned world model synthesizes imagined rollouts so a behavior agent improves without extra ship time [21], [22], [23]; online, it ranks candidate commands for multi-trajectory reasoning. One model serves both: driven by commands rather than by measured motion, the predictor learns the thrust-buildup and added-mass lag that short-horizon models leave unmodeled [24]; predicting fused tokens instead of pixels keeps it at 1.33 M parameters.

![](images/d4fcdf7023ecb390f39da0cf8d9e0a85f518cec55c6eb58377fd0ecf9eda3bbb.jpg)  
(a)

![](images/345f1344989139d4c4f85b2bc269729f3c820377221b3979a592d68abd7b1e74.jpg)  
(b)

![](images/a1f2a817fa76a44bd8b084df16a8b6dad0810facac5edd5b515395928af03ec2.jpg)  
(c)  
Fig. 1. The heavy-duty underwater salvage ROV used in this study: (a) isometric view; (b) grasping a target object (CAD rendering); (c) lifting a target object in a real-environment trial.

Sonar, navigation and force sensing all matter; this paper asks the narrower question: whether the on-board visual channels compress into a state that keeps the target and gripper identifiable across views and through contact and supports rollouts.

Our contributions are:

1) Object-centric perception domain (OCPD) and guided object-centric cross-view fusion (GO-CVF): weak anchors stabilize identity, held-out-view attention consolidates partial evidence, and free slots keep unaligned context alongside; because alignment is spent only where it pays, fusion fits an ROV budget and new task classes are synthesized cheaply.

2) A control-conditioned JEPA world model of the vehicle–object dynamics, with a model-predictivecontrol (MPC) interface: driving the predictor by commands makes the actuation lag between command and realized motion part of the learned state; a 1.33 M footprint lets one model serve imagined training on limited offline data and candidate-command ranking for model-predictive control; and each ingredient of the loop—predictor, latent geometry and ranking signal— is validated separately.

3) Validation on real underwater video: two field deployments on uncalibrated cameras confirm that the hiddenview recovery and the control-conditioned rollout carry beyond simulation.

## II. RELATED WORK

## A. Underwater perception: scene mapping versus task-object state

Underwater perception for intervention is conventionally organized around mapping. Visual, visual-inertial, and acoustic inertial SLAM systems estimate vehicle pose and build sparse or semi-dense structure, with underwater variants adding image enhancement, pressure-depth terms, and sonar or DVL factors to counter drift and tracking loss [9], [11], [12], [10]; where dense near-field geometry is required, active projection and structured-light scanners deliver higher per-frame accuracy at the cost of short standoff, sensitivity to scattering and turbidity, and usually a dwell-or-scan motion constraint [13], [14]. Both families reconstruct the scene first and interpret it afterwards. C<sup>3</sup>-JEPA takes the complementary route and never commits to a metric map: it maintains an object-centric perception domain (OCPD) in which only task objects are resolved and aligned. The two answers are not adversarial—mapping remains right for search and survey—but differ in cost and in what survives occlusion and contact.

## B. Object-centric visual representation

Slot Attention decomposes a scene into a small unordered set of entity-like vectors [25]. SAVi and SAVi++ extend objectcentric learning through time [26], [27]; DINOSAUR and VideoSAUR show that reconstructing self-supervised features can support object discovery in natural images and videos [28], [29]; and SlotFormer studies long-range dynamics in object slots [30]. A practical difficulty for robotics is that unconstrained slots may permute or absorb a small task object. We use weak binding only for the target and gripper (a single extra binding-only slot handles the fixed ROV frame), leaving the remaining slots to account for scene context. This is deliberately lighter than dense segmentation: DINOv3- feature clustering proposes masks from a few region prompts. An orthogonal upgrade path replaces instance slots with task-functional roles for planning [31]; our design keeps two instance-level roles (target and gripper), sufficient for enveloping salvage, with the mechanism validated on real frames.

## C. Multi-view fusion and underwater predictive models

Multi-view object-centric models such as ROOTS and MulMON aggregate partial observations and test unobserved viewpoints [32], [33], and the held-out-view principle is well established at the pixel level: cross-view completion predicts the masked content of one image from a second view of the same scene [34], and multi-view masked autoencoders reconstruct withheld viewpoints before a world model is trained on the resulting representations [35]. We adapt the idea to cameras rigidly mounted on one moving ROV: instead of forcing a shared token to decode identically in all cameras, the target view is withheld from cross-view keys and values, and a viewconditioned decoder reconstructs its features. This discourages direct copying while preserving legitimate view-dependent appearance. AquaJEPA combines RGB, forward sonar and proprioception for simulated robot dynamics [36], a setting outside the current C<sup>3</sup>-JEPA data interface. A recent AUV docking study combines hand-built hydrodynamic priors with a compact learned model under communication constraints [37]; our route is fully learned but shares the goal of onboard-scale predictive control.

## D. Latent prediction and planning

A world model is a functional role rather than a single architecture: a learned state, explicit or latent, whose evolution can be queried under candidate actions. It is used for several related purposes: compressing high-dimensional observations into a state for control, predicting short-horizon consequences for model-predictive control (MPC), evaluating counterfactual action sequences, training a behavior policy in imagined rollouts, and transferring a reusable state representation to downstream estimators. Designs differ along three overlapping axes: pixel-space generative models that reconstruct future observations; latent dynamics models that predict a compact embedding; and object-centric models whose state is partitioned into entities, relations, and context.

Early latent-dynamics systems such as PlaNet learn a compact transition model from pixels for planning, while Dreamer extends the idea to behavior learning through latent imagination [38], [21], [22]. Visual Foresight uses action-conditioned visual prediction to select robot motions, and TD-MPC2 repeatedly optimizes short candidate sequences in a learned latent space [39], [40], [41]. These approaches establish the broad worldmodel planning loop—observe, predict candidate consequences, score them, execute a short prefix, re-observe—but do not by themselves guarantee that the latent state preserves target and gripper identity across occlusion and contact. For underwater manipulation this matters: a representation can predict average motion while discarding the pose, joint, or object evidence a downstream controller needs.

JEPA methods shift the prediction target from pixels to representations. I-JEPA predicts masked image embeddings without reconstructing every pixel, and V-JEPA 2 extends this principle to video understanding, action conditioning, and planning [42], [43]. LeJEPA frames joint-embedding prediction as a geometrically regularized objective, while AdaJEPA studies adaptive latent prediction; both motivate treating the latent geometry as a first-class design object [44], [45]. The efficiency is attractive for an ROV decision interface—the predictor stays small while candidates are rolled out in state space—but a low latent prediction error does not imply object semantics, observable calibration, or task-relevant information transfer. The closest published analogue is SkyJEPA, which maps frozen JEPA latents to interpretable state with a physics-inspired prober and carries it to real quadrotor control without fine-tuning [46]; our frozen geometry head plays the analogous role for salvage.

Causal-JEPA makes the control question more explicit by studying object-level latent interventions, asking how changes in an object or intervention variable alter the predicted future [47]. It builds on object-centric encoders such as VideoSAUR or SAVi; we take the VideoSAUR branch because it yields slots in real-world video without the motion or depth conditioning SAVi-style pipelines assume [28], [29]. This work combines the three ingredients: weak binding preserves target and gripper identity, held-out-view fusion preserves complementary evidence, and the latent predictor exposes candidate consequences to MPC and imagined-rollout evaluation.

Reconstruction-free latent world models. LeWM-style architectures train a JEPA encoder–predictor from raw pixels with a next-embedding prediction loss and a SIGReg prior, without any reconstruction or object-centric structure [48]. We train such a model from scratch on exactly the same multi view recordings and control signals as $C ^ { 3 }$ -JEPA, and compare both representations under identical downstream probes. This head-to-head comparison, reported in Sec. V-F, isolates what reconstruction-guided, object-centric supervision adds over purely latent regularization at this scale. Training-time geometric priors (paired-depth alignment) similarly improve JEPA representations from real robot data while leaving inference RGB-only [49]. In our recipe, geometry enters through the frozen reconstruction decoder instead of an auxiliary loss.

## III. METHOD

## A. Architecture overview

Fig. 2 summarizes the pipeline. At time t, synchronized camera observations are $\hat { I } _ { t } ^ { 1 : \hat { V } }$ . A frozen DINOv3-L feature extractor [50] maps each view to patch features $X _ { t } ^ { v } \in \mathbb { R } ^ { P \times d _ { p } }$ and a temporal slot encoder returns K slots $S _ { t } ^ { v } \in \mathbb { R } ^ { K \times d _ { s } }$ . In simulation $V = 4 ;$ the architecture itself accepts the subset of views available on a platform. Three binding slots represent the target UUV, the gripper, and the fixed ROV frame. Because the frame occludes parts of every view at a fixed pose, it is kept in a binding-only slot (slot $2 )$ , excluded from cross-view fusion and from the predictive state; the free slots are not spent on it. Cross-view attention produces shared target and gripper tokens, and the predictive state concatenates these with the three context slots of a single reference view, $v _ { r }$ (the rear camera, which also serves as the free-slot prediction target),

$$
z _ { t } = [ \bar { s } _ { t } ^ { \mathrm { t a r g e t } } , \bar { s } _ { t } ^ { \mathrm { g r i p } } , s _ { t } ^ { v _ { r } , 3 } , s _ { t } ^ { v _ { r } , 4 } , s _ { t } ^ { v _ { r } , 5 } ] .\tag{1}
$$

Equation (1) is the object-centric perception domain (OCPD) introduced in Sec. I: the two task-object entries carry the target and gripper at full fidelity and are forcibly aligned across cameras, while the three context entries are free slots carried without alignment. The state is object-centric by construction rather than a uniformly compressed scene descriptor, which keeps cross-view fusion inside an ROV compute budget.

## B. Object-centric slots and geometric regularization

For bound object $^ { O , }$ a slot renderer produces a mask $\hat { M } _ { t , o } ^ { v }$ and a weak mask proposal $M _ { t , o } ^ { v }$ supplies an identity anchor. We use

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e p r } } = \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { b } \mathcal { L } _ { \mathrm { b i n d } } + \lambda _ { m } \mathcal { L } _ { \mathrm { m e r g e } } } \\ & { ~ + ~ \lambda _ { t } \mathcal { L } _ { \mathrm { t i m e s i m } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v i s } } + \lambda _ { s } \mathcal { L } _ { \mathrm { S I G R e g } } , } \end{array}\tag{2}
$$

![](images/3b8554e18dc73fe516ad40b5b21512464b6287b7d33dc324f7f56c8ef6d7d8e9.jpg)  
Fig. 2. C<sup>3</sup>-JEPA architecture overview. Stage I maps synchronized multi-view RGB to bound task-object slots and free context slots; Stage II injects the control latent through AdaLN-Zero and predicts the state autoregressively. Right: the predictive interface.

where $\mathcal { L } _ { \mathrm { f e a t } }$ reconstructs frozen DINOv3 patch features, free slots receive no object identity label, $\mathcal { L } _ { \mathrm { t i m e s i m } }$ matches consecutive slot states to keep tracks temporally consistent, and ${ \mathcal { L } } _ { \mathrm { v i s } }$ supervises the per-view visibility of the bound objects. The binding path is patch-to-slot rather than a detached DINO segmentation head: frozen DINOv3 patch evidence is scored against weak target/gripper anchors, and the selected support guides the corresponding VideoSAUR slot masks (Fig. 3). Identity is assigned by affinity argmax against the query vectors $q ^ { \mathrm { t a r g e t } }$ and $q ^ { \mathrm { g r i p } }$ . A PCA–clustering view can still serve as a rapid proposal for the patch support. This makes the supervision auditable and inexpensive, but does not convert the method into fully unsupervised segmentation.

SIGReg is the Sketched Isotropic Gaussian Regularizer introduced for JEPA representations and used in LeWM to prevent collapse by matching the latent marginal to an isotropic Gaussian [44], [48]. Let $Z \in \mathbb { R } ^ { N \times B \times d }$ collect token embeddings over a temporal window of length N, a batch of size B, and embedding dimension $d .$ SIGReg projects $Z$ onto M random unit directions $u ^ { ( m ) } \in \mathbb { S } ^ { d - 1 }$ , applies the univariate Epps–Pulley normality statistic $T ( \cdot )$ to each projection, and averages the result:

$$
\mathcal { L } _ { \mathrm { S I G R e g } } ( Z ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } T \Bigl ( h ^ { ( m ) } \Bigr ) , \qquad h ^ { ( m ) } = Z u ^ { ( m ) } .\tag{3}
$$

The Cramer–Wold construction makes one-dimensional pro-´ jections a tractable surrogate for the full distribution. SIGReg regularizes token geometry but does not assign target or gripper semantics; those come from weak binding and object-feature reconstruction.

## C. Held-out-view fusion

For each bound object, cross-attention receives visible-view tokens and returns a fused token $\bar { s } _ { t } ^ { o }$ . During training, target view $v ^ { * }$ is excluded from keys and values. A view-conditioned decoder predicts its held-out feature evidence,

$$
\mathcal { L } _ { \mathrm { m e r g e } } = \sum _ { o } { q _ { t , o } ^ { v ^ { * } } \left\| D _ { v ^ { * } } \left( \bar { s } _ { t } ^ { o } \right) - X _ { t , o } ^ { v ^ { * } } \right\| _ { 2 } ^ { 2 } } ,\tag{4}
$$

where $q _ { t , o } ^ { v ^ { * } }$ masks genuinely invisible objects. This test requires fused tokens to retain cross-view object evidence without requiring all camera appearances to collapse.

## D. Control-conditioned JEPA prediction

The conditioning vector is the vehicle’s normalized control signal,

$$
\begin{array} { r } { c _ { t } = [ \tilde { u } _ { 1 } , \dots , \tilde { u } _ { 1 2 } , \tilde { g } _ { 1 } , \dots , \tilde { g } _ { 6 } ] _ { t } , } \end{array}\tag{5}
$$

combining thruster command channels and gripper command channels. An MLP produces AdaLN-Zero parameters for the predictor [51]. Given a history of four states, the model predicts twelve future states and minimizes token MSE plus cosine distance. The complete predictor has only 1.33 M parameters: because it predicts fused object tokens rather than pixels, the decision interface stays light even as the learned dynamics grow complex. In simulation the model is driven by commands rather than by measured motion, so the hydrodynamic mapping from control to realized vehicle motion, including thrust buildup, added-mass inertia, and cross-couplings, is learned inside the predictor. The control interface is modality-agnostic: thruster and gripper commands in simulation and recorded control and DVL-derived motion in the field drive the same autoregressive rollout. This makes the predictive state directly usable as a decision variable: candidate command sequences can be rolled out, scored, and replanned against the learned dynamics. Sec. V-E validates each ingredient of that loop separately. Extended architectural and training details are given in Sections S2 and S3 of the supplementary material.

## IV. EXPERIMENTAL SETUP AND EVALUATION

Table I lists the training configuration of the visual encoder and the control-conditioned predictor, shared by all experiments below.

## A. Dataset taxonomy and split policy

Data are collected in the ROS–Gazebo UUV Simulator environment [52], which implements Fossen’s equations of motion for the vehicle dynamics [24]; each recording provides synchronized cameras, vehicle state, gripper signals, poses, and event annotations. The caveat of Sec. I about solving this gripper in simulation concerns closed-loop policy training; here the simulator instead supplies synchronized observations and privileged state, from which the model learns the commandconditioned mapping. Model inputs are the camera streams and command channels; privileged simulator state is used only to build training targets and evaluation references. The current camera configuration passes channel-wise attenuation values (0.3, 0.3, 0.09) and RGB noise with standard deviation 0.02, so the recordings embody a simplified attenuation/noise camera model rather than full radiative transfer. We package recordings by behavior rather than by storage folder: privileged automatic data comprise wandering near a target, blind touching, and two naive privileged auto-grasp batches; manual teleoperation data comprise wandering, gripper motion, interaction, grasping, and verification. The release used here contains 705 recordings across the two privileged auto-grasp batches, plus grasping (41), interaction (30) and gripper-motion (14) recordings; the remainder is unscripted collection. Each release manifest records source behavior, camera availability, calibration, annotation provenance, and splits. All results use recording-disjoint train/validation partitions, so no recording contributes frames to both sides of any reported comparison. Beyond the simulator, two field deployments of the same vehicle working a submerged real target supply synchronized camera streams and navigation telemetry; field evidence is reported in relative units in Sec. VI.

![](images/702abd655bc7f64f1e61e3b3b106a947c6b3aeca102800249b7f5a01e5fffb87.jpg)  
Fig. 3. Binding-guided patch-to-slot grounding in C<sup>3</sup>-JEPA. Left to right: synchronized multi-view RGB frames from one real recording; grouping slot maps for the target (slot 0) and gripper (slot 1); the binding overlay, where weak anchors select patch support; the VideoSAUR-decoded slot masks that feed temporal state formation; and a PCA view of the patch features.

TABLE I  
TRAINING CONFIGURATION OF THE VISUAL ENCODER AND THE CONTROL-CONDITIONED JEPA PREDICTOR.
<table><tr><td colspan="2">Visual encoder (frozen DINOv3-L backbone)</td></tr><tr><td>Slots</td><td>6 total: 3 binding (target UUV, gripper, and a binding-only slot for the fixed ROV frame), 3 free context</td></tr><tr><td>Slot dimension</td><td> $d _ { s } = 2 5 6$  (with  $1 2 5 6 {  } 1 2 8 {  } 2 5 6$  adapter)</td></tr><tr><td>Fusion</td><td>held-out-view cross-attention, 8 heads (CrossViewMerge)</td></tr><tr><td>Reconstruction target</td><td>frozen DINOv3-L fp16 patch features</td></tr><tr><td>Loss weights</td><td> $\lambda _ { b } \mathrm { = } 1 . 0 , \lambda _ { m } \mathrm { = } 1 . 0 , \bar { \lambda } _ { t } \mathrm { = } \bar { 0 . } 1 0 , \lambda _ { v } \mathrm { = } 0 . 2 0 , \lambda _ { s } \mathrm { = } 0 . 0 3$ </td></tr><tr><td>Training</td><td> $1 \times 1 0 ^ { - 4 } .$  ≈644 epochs, batch 16×8 GPUs</td></tr><tr><td colspan="2">Control-conditioned JEPA predictor (transformer encoder)</td></tr><tr><td>History / future frames</td><td>4 / 12 (recorded as  $1 . 0 \mathrm { ~ s ~ / ~ } 3 . 0 \mathrm { ~ s ~ o f ~ }$  context)</td></tr><tr><td>Anchored masked slots</td><td>1 (one masked slot identity per window, recovered from the anchor frame)</td></tr><tr><td>Transformer</td><td>depth 6, width 128, 8 heads, GELU, norm-first, AdaLN-Zero</td></tr><tr><td>Optimizer Epochs / batch / split</td><td>AdamW, lr  $5 \times 1 0 ^ { - 4 } .$  weight decay 0.05</td></tr><tr><td></td><td>30 / 256; 159 train, 18 validation</td></tr><tr><td colspan="2">Longer-context variants (stability probe)</td></tr><tr><td>Context frames</td><td>32 (8.0 s) and 16 (4.0 s), 25 epochs, initialized from the short-context predictor</td></tr></table>

## B. Metrics and evidence ladder

We evaluate five progressively stronger levels:

1) Representation: task-object mask Dice, held-out-view feature error, and slot stability.

2) Prediction: held-out token MSE and cosine distance under teacher forcing.

3) Recursive rollout: horizon-stratified comparison to persistence and controls shuffled within matched windows.

4) Observable calibration: position, relative-pose, or image-space error through a frozen readout with documented calibration limits.

5) Decision and transfer: executable-action ranking and conservative closed-loop simulation.

An improvement at one level is not evidence for a later level. The full evaluation protocols, including the held-out-view protocol and the negative controls, are specified in Section S4 of the supplementary material.

## V. RESULTS

The evidence below follows the evaluation ladder of Sec. IV.

## A. Representation evidence

The learned representation preserves task-object semantics while reconstructing frozen visual features (Table II). Across the recording-disjoint split, binding-guided training maintains high task-group Dice for the target UUV and gripper while held-out-view fusion lowers feature reconstruction error.

Binding guidance trades reconstruction for downstream utility. Compared with an unguided self-supervised slot baseline (identical architecture and equal budget, but no binding anchors), binding guidance makes the slot decomposition semantically explicit—the target and gripper occupy stable, identifiable slots (group Dice ≈ 0.90/0.88) whereas the unguided baseline’s slots carry essentially no task-object semantics (group Dice ≈ 0.09/0.12)—while feature reconstruction error rises only modestly, from 0.0157 to 0.0186 (Table II). The representation therefore pays a small reconstruction cost in exchange for taskrelevant structure, confirming that reconstruction fidelity and downstream utility are distinct objectives.

Held-out-view fusion, measured against matched arms. Block (b) of Table II isolates the fusion design on a recordingdisjoint split at equal budget, where the arms differ only in the flags named in the row. The two blocks are not comparable in absolute level, and all between-arm claims are drawn from (b). Replacing weighted-sum fusion with held-out-view crossattention improves all five axes: feature MSE falls from 0.0231 to 0.0200 (−13.4%), target decoder Dice rises from 0.713 to 0.820 (+10.7 pp), gripper decoder Dice from 0.814 to 0.891 (+7.7 pp), and the target and gripper group Dice by +2.7 and +3.8 pp. Two further contrasts sharpen what is doing the work. Against averaging the visible views with the target held out, cross-attention is better on four of five axes (target group $0 . 8 2 1  0 . 8 2 6$ , target decoder 0.811 → 0.820, gripper group $0 . 8 3 6 \to 0 . 8 4 3$ , feature MSE 0.0201 → 0.0200). And against the same cross-attention allowed to attend to the target view, holding that view out is again better on four of five axes (target group 0.796 → 0.826, gripper group 0.824 → 0.843, target decoder 0.805 → 0.820, feature MSE 0.0205 → 0.0200), consistent with the constraint acting as a regularizer. One axis runs the other way: gripper decoder Dice is 1.1–1.2 pp lower under held-out fusion (0.891 against 0.902/0.903). The rearview-only arm is not a competing fusion method: it is trained with the target view included and cannot answer Equation (4);

its comparable average Dice contributes nothing to the held-out claim. The block (b) margins come from a single training run per arm and are read as effect sizes rather than as variance estimates.

The fused token carries evidence for a view it never saw. The ladder above measures average quality; the direct test is whether the fused token can reconstruct a withheld view at all. On a purpose-built held-out evaluation of the same design (each of the four cameras held out in turn; the zero-information floor averages the three visible views’ local outputs), held-out target Dice rises from 0.2206 to 0.7604 for the UUV and from 0.2034 to 0.8695 for the gripper, held-out feature MSE falls from 2.3104 to 1.2183 and from 2.3552 to 1.0148, and spurious mask mass on frames where the target is absent in that view collapses from 0.1339 to 0.0063 and from 0.1257 to 0.0007. The shared token does not collapse: its effective rank [53] is 14.69 for the UUV and 8.71 for the gripper, out of 128 dimensions. Two limits belong with this result. The evaluation uses 512 fixed random windows rather than a recording-disjoint split, so it establishes the mechanism and its magnitude, not a generalization estimate; and four of its five acceptance criteria, fixed before the evaluation, pass, with the front camera (which meets the target at a grazing angle) landing at 0.456 UUV Dice where right, bottom and rear sit between 0.755 and 0.800. That spread is itself the argument for fusion over any single chosen view: which view will be the inadequate one is not knowable in advance.

SIGReg improves the task-object position head. Separately from the DINOv3 checkpoint in Table II, a representation sweep isolated the contribution of SIGReg by training the geometry renderer with and without the SIGReg term while keeping everything else fixed. Adding SIGReg at weight 0.03 reduced the estimated 3D target-center error from 0.625 to 0.496 (normalized center error), increased Dice from 0.887 to 0.895, and improved both IoU and feature reconstruction. The position head therefore benefits from SIGReg independently of the fusion change: a small amount of latent-geometry regularization sharpens the geometry head and compounds with, rather than substitutes for, the object-centric supervision.

## B. Conditional rollout evidence

On a 229-recording interaction-focused diagnostic set, the conditional predictor reduced a frozen-readout terminal relativemotion error from 39.1 cm for persistence (recording-level bootstrap 95% CI 36.7–41.5) to 25.7 cm (23.6–27.9). At five seconds, the same diagnostic reported 59.7 to 31.0 cm on the AutoGrasp recordings and 93.8 to 82.9 cm on the Interacting recordings. Zeroing the conditioning signal increased latent MSE, confirming that the learned tokens encode controlcorrelated change and outperform persistence in dynamic windows. Shuffling or zeroing the control input at a fixed state is an established intervention for verifying that a predictor conditions on actions rather than on visual habit [54]. The same diagnostic shows where persistence stays competitive (Fig. 4c): at short horizons the model’s latent MSE is comparable to persistence, the two curves nearly coinciding on AutoGrasp at 10 s. A persistence copy is a strong latent-space baseline because most tokens change slowly; the model’s advantage lies in the readout-relevant terminal error and in the controlconditioning margin, not in raw latent distance. Repeating the last observation as a latent-space baseline is standard practice rather than a strawman [55], and recent work normalizes rollout error by a temporal-persistence baseline to discount temporal smoothness [56]. Longer-context variants (4 s and 8 s) and an unroll-regularized variant were probed on the same diagnostic; neither improves on the deployed 1 s-context model overall (terminal error 32.2 cm for both longer-context variants, 28.6 cm for the unroll variant, versus 25.7 cm; Fig. 4), the single exception being the 10 s horizon on the Interacting recordings, where the 8 s variant is best. The compact configuration is retained.

TABLE II  
REPRESENTATION CHECKPOINTS. HIGHER DICE AND LOWER FEATURE MSE ARE BETTER. (A) DEPLOYED FULL-DATA CHECKPOINTS; THE UNGUIDED ROW IS THE EQUAL-BUDGET BASELINE WITHOUT BINDING ANCHORS. (B) EQUAL-BUDGET FUSION LADDER.
<table><tr><td colspan="6">(a) Deployed checkpoints</td></tr><tr><td>Model</td><td>Feat. MSE</td><td>Target group</td><td>Target dec.</td><td>Grip group</td><td>Grip dec.</td></tr><tr><td>Binding + SIGReg</td><td>0.0186</td><td>0.897</td><td>0.915</td><td>0.881</td><td>0.933</td></tr><tr><td>Held-out fusion Unguided self-supervised</td><td>0.0183</td><td>0.895</td><td>0.903</td><td>0.877</td><td>0.921</td></tr><tr><td>(no binding)</td><td>0.0157</td><td>0.085</td><td>0.018</td><td>0.116</td><td>0.118</td></tr><tr><td colspan="6">(b) Equal-budget fusion ladder</td></tr><tr><td>Binding, weighted-sum fusion (no cross-view)</td><td>0.0231</td><td>0.799</td><td>0.713</td><td>0.805</td><td>0.814</td></tr><tr><td>Binding, rear view only</td><td>0.0201</td><td>0.828</td><td>0.811</td><td>0.845</td><td>0.902</td></tr><tr><td>Binding, mean of visible</td><td>0.0201</td><td>0.821</td><td>0.811</td><td>0.836</td><td>0.902</td></tr><tr><td>views, target held out Binding, cross-attention,</td><td>0.0205</td><td>0.796</td><td>0.805</td><td>0.824</td><td>0.903</td></tr><tr><td>target view exposed Binding, cross-attention, target held out</td><td>0.0200</td><td>0.826</td><td>0.820</td><td>0.843</td><td>0.891</td></tr></table>

Synthetic-perturbation surprise. To test whether the predictor holds expectations about observation continuity rather than extrapolating means, we inject four perturbations into the history window and leave the target unchanged (12,000 windows over 250 held-out records). Overwriting the history with the training-set mean token raises future error by 536% (95% CI of the absolute increase 0.20–0.27, detected in 82% of windows); replacing it with a distant window of the same recording raises it by 260% (0.10–0.12, 75%). Reversing the three pre-anchor history frames or permuting them moves the prediction by 1.6% and 0.6%. The predictor is therefore sharply sensitive to violations of observation continuity, and nearly indifferent to frame ordering inside a one-second history.

Using the frozen geometry UUV renderer, we also decode the predicted future UUV token into a slot mask and compare its centroid track to the recorded one (Fig. 5). The world model reproduces the recorded UUV motion both when the target is far from the ROV (≈3.4 m, over a 3 s window the recorded mask center moves by about 1.1 normalized units and the predicted center tracks it with a mean error of about 0.09) and when the target is close (≈0.8 m), where a recursive 5 s rollout keeps the mean centroid-tracking error under 0.05 normalized units with no divergence at the endpoint. The multi-view design is what makes this tracking robust in near-field salvage: as a task object moves out of one camera’s field of view and into another, a single camera would lose the object exactly when its view is lost, while held-out-view cross-attention aggregates the evidence still visible in the other cameras. Consistent with this mechanism, the shared target token is decoded to the same 3D location across views (its mask-centroid rays stay close, including for the predicted token; see Fig. 5 and the cross-view results below), and the observed trajectory is preserved through an imagined rollout.

## C. Cross-view consistency

A shared task token must fuse evidence across views: the masks it decodes in different cameras should point at the same 3D location. We measure the mean pairwise angular spread of the four view-centroid rays for the UUV token on 15 interacting recordings, binned by prediction horizon, and compare it between the world model’s predicted token and the observed token. The two agree closely at every horizon (16– 20<sup>◦</sup> predicted versus 17–22<sup>◦</sup> observed), and the mean predicted spread is no larger than the observed spread within each bin. Prediction thus preserves the multi-view fusion structure: the fused token stays view-consistent through an imagined rollout.

## D. Causal consistency and behavior-agent training

Beyond representation, we test whether conditioning is real: the predicted slot motion should track the true motion only when the recorded control is supplied. On the same interacting set, the direction cosine between the predicted and true UUV token displacement rises to about 0.42 with control at 15 s versus 0.23 without control (recording-weighted means over the phase-split rows of Table S4 of the supplementary material; the simple average of the two 15 s rows differs because weighting follows per-phase counts), so the model uses the conditioning signal. Splitting by the gripper command, the grasp/contact phase shows a positive control gain at every horizon, growing from +0.07 at 1 s to about +0.24 at 15 s; the approach phase is positive only at the longer horizons (slightly negative at 1 s, +0.06 at 5 s, up to about +0.16 at 10 s). Table S4 of the supplementary material reports the recording-level means.

![](images/81e34769084a3cedd9a753edf8076c7ad90c81f0823c45c5e487788d4b028b20.jpg)

![](images/1916e603fe6546f47da0d73afed45bbf8fb29702b53e7b5667fbedfa3f1465e5.jpg)

![](images/986e29db2e170f204aa698d1ba2f6f416bec19e9f9b74b2b5967f1c2dc3874e2.jpg)

![](images/048d468b8170c9eb80846c3e999e1a264e64626f6a26252585c257e00418a3a5.jpg)  
Fig. 4. Conditional rollout diagnostic on the 229-recording interaction set. (a,b) Terminal relative-motion error versus free-rollout horizon for persistence C<sup>3</sup>-JEPA, the unroll-regularized variant, and the long-context variants (16/32 frames); (c) latent MSE against persistence; (d) 5 s control-conditioning ablation, recorded versus zeroed control, with bootstrap 95% CIs.

We further probe whether the world model can train a behavior agent. Using the conditional dynamics and a frozen potential critic, we train an action-generating agent with behavior-cloning regularization under imagined rollouts, separately in the search and grasp/contact phases. On held-out recordings the learned agent’s critic-predicted task progress exceeds the behavior-cloning baseline by +0.383 (search) and +0.209 (grasp/contact). The imagination interface guides agent improvement in both phases. Progress is measured by the same frozen critic that shapes the training signal, so this is a statement about the consistency of the imagined-rollout loop rather than an independent task-success measure.

![](images/1b7b47b0ca489e4dd39fef680453a07757429e3f99934de66f82598e7e8ffac6.jpg)

![](images/8e1e9bde00c05935e4daf4437e89e755e3155ce5bb8dc5028d80c59705b7fc6f.jpg)

![](images/6e21015d810e1f25140075a54c88396553fbc512f7d6232bd9a32e18b6e2f07d.jpg)

![](images/c3ced6958c88132cfa1c6b5b402c5075765cd75273f02a9a2d2dee0413143012.jpg)  
Fig. 5. UUV (slot 0) mask motion under the world model at two ranges: far (≈3.4 m, 3 s rollout, top) and close (≈0.8 m, recursive ${ 5 } \mathrm { s } ,$ bottom). Each panel pairs recorded (blue) and predicted (orange) mask-centroid tracks with the bottom-camera frame.

## E. Interface validation for model-predictive control

The matched-endpoint ranking diagnostic validates the planning interface: candidate trajectories share a common terminal pose and differ only in control timing, and the world model’s predicted latent endpoints rank them well above shuffled-control rollouts. Concretely, on 90 matched-endpoint candidate groups under recording-level out-of-fold evaluation, ranking by the world model’s predicted endpoint selects the true best candidate first in 53.3% of groups (bootstrap 95% CI 43.1–63.3%) versus 23.3% (15.8–33.1%) for shuffled-control rollouts and 25% for random choice, with mean regret (selected-candidate distance penalty relative to the group’s best) reduced from 0.0091 to 0.0058; each group contains four trajectories, so chance is 25%, and the model’s interval excludes both chance and the shuffledcontrol rate. Recent work audits whether latent distances rank candidate plans consistently with executed outcomes [57]; the diagnostic above is a positive instance. Together with the causal-consistency gains of Table S4 of the supplementary material (positive at every horizon in the grasp/contact phase), all three ingredients of the model-predictive loop are in place: a control-conditioned predictor, a consistent latent geometry, and a ranking signal that separates candidate actions. Closedloop operation composes these verified ingredients with a conventional automatic controller, as described in Sec. VI.

## F. Downstream transfer against a reconstruction-free world model

To quantify how much task-relevant information each representation retains, we train a reconstruction-free latent world model (LeWM [48]: ViT encoder, latent next-step prediction, SIGReg prior on the embedding; no reconstruction, no slots, no object supervision) from scratch on the same multi-view recordings and control signals as $\mathrm { C } ^ { 3 } { \mathrm { - } } \mathrm { J E P A } .$ , and compare frozen representations under an identical probe architecture, split, and optimization predicting gripper joint angles, target relative position, and target yaw. Table III reports the result: C<sup>3</sup>-JEPA roughly halves the joint-angle and yaw error of the reconstruction-free embedding and reduces relativeposition error by a quarter, transferring substantially more task information on every head.

A purely latent objective with SIGReg as its only regularizer discards a large share of the visual information manipulation requires, driving the embedding toward compressed dynamics statistics and away from pose- and joint-level evidence. The object-centric supervision in $\mathbf { C } ^ { 3 } \mathbf { - J } \mathbf { E P A } .$ —reconstruction-guided slots with weak binding and a small SIGReg weight (0.03)—is what preserves that evidence. SIGReg alone does not recover it, and at reconstruction-free training scales it actively hurts downstream heads; combined with reconstruction and binding, the same regularizer improves them.

TABLE III  
DOWNSTREAM PROBE TRANSFER ON THE GRASPING SCENE (RECORDING-DISJOINT SPLIT; IDENTICAL PROBE ARCHITECTURE AND TRAINING). LOWER IS BETTER. LEWM IS TRAINED FROM SCRATCH ON THE SAME RECORDINGS AND CONTROL SIGNALS AS C<sup>3</sup>-JEPA.
<table><tr><td>Representation</td><td colspan="3">Joint MAE (°) Rel. pos. MAE (m) Yaw MAE (°)</td></tr><tr><td>C³-JEPA (binding + slots + SIGReg)</td><td>4.90</td><td>0.60</td><td>6.50</td></tr><tr><td>LeWM (latent, SIGReg only)</td><td>9.81</td><td>0.79</td><td>13.82</td></tr></table>

## VI. FIELD VALIDATION ON REAL UNDERWATER VIDEO

All results above are trained and measured on simulated salvage scenes. The question here: does the same predictive state, with the same recipe, hold up on real underwater video? Everything here is trained from the raw field recordings alone; the simulator does not participate. Open-loop replay of recorded actions is a recognized evaluation protocol when closed-loop deployment is unavailable [58]. Both recordings were collected in a test pool measuring 60 m by 20 m with a water depth of 10 m, under two operation modes (Fig. 9): unscripted navigation 3–5 m above a steel frame installed on the pool floor, and active interaction with an aluminum pipe on the floor, bringing the gripper into contact with the object. The two deployments are reported separately and never pooled: Deployment A is the navigation recording, which keeps the steel frame in view; Deployment B is the contact recording. The platform is the heavy-duty salvage ROV of Fig. 1, with paired linkage-driven underactuated grippers, a synchronized camera pair, navigation telemetry, and no contact sensing. Both are recorded by two co-mounted cameras for which no intrinsics or extrinsics were measured, and the target carries no metric ground truth. A video walkthrough of the experiments accompanies this paper as supplementary multimedia.

Three reference points appear throughout: predicting the mean token of the training sessions, copying the previous token (persistence), and decoding the true tokens of the view being predicted. Token error is the per-dimension mean squared error against these references, and mask quality is reported as a floor/model/ceiling intersection-over-union triple.

Evidence for a hidden camera survives in the shared tokens. Predicting one camera’s tokens from the other camera alone lowers token error by 76% and 65% on the two deployments and recovers the hidden camera’s mask at 88% and 81% of the achievable range (Table IV; Fig. 6a–c and Fig. 7). The recovered tokens also keep their spread, close to the true spread against about half of it for the mean-token prediction, so the gain is not the trivial mean-token shrinkage effect. A linear readout of the same history fails: it is 5.7× and 3.5× worse than persistence on the two deployments, and once targets are expressed as frame-to-frame change it is indistinguishable from a shuffled control. The evidence is present but not linearly recoverable; a nonlinear head is therefore required. The two cameras are rigidly co-mounted and largely co-visible, so the field study tests hidden-view recovery (the capability fusion must deliver) rather than a dedicated fusion benchmark; the identity-preserving form of the fusion claim is established on the simulated camera rig of Sec. V. Fig. 8 shows both cameras of one recording decoded from their own tokens and from the merged cross-view tokens.

TABLE IV  
FIELD-VALIDATION SUMMARY. DEPLOYMENT A: NAVIGATION ABOVE A SUBMERGED STEEL FRAME; DEPLOYMENT B: CONTACT WITH A SUBMERGED ALUMINUM PIPE.
<table><tr><td></td><td>Deployment A (77 segments)</td><td>Deployment B (36 segments)</td></tr><tr><td>Hidden-view token error</td><td> $0 . 6 1 5  0 . 1 4 7$ </td><td> $0 . 9 6 2  0 . 3 3 2$ </td></tr><tr><td>Hidden-view mask IoU</td><td></td><td>0.21 / 0.48 / 0.520.20 / 0.50 / 0.58</td></tr><tr><td>Recovered share of range</td><td>88%</td><td>81%</td></tr><tr><td>5 s prediction vs. persistence</td><td>-23%</td><td>-31%</td></tr></table>

Prediction against recorded motion. A state-only closed-form ridge regression on the 16-step token history reduces 5 s-ahead token error by 23% and 31% on the two deployments relative to persistence (Table IV); re-running the reference implementation reproduces the recorded values to within $4 \times 1 0 ^ { - 6 }$ . Letting the field-trained C<sup>3</sup>-JEPA predictor, conditioned on the recorded control signals, feed itself for 36 steps is the harder test, and it holds: error falls 30% below persistence (95% CI 21–38%) and 89% below the mean-token floor (Fig. 6d–e).

Mask quality follows its own axis. The same rollout that cuts token error by 30% does not improve mask quality: 0.159 against 0.184 for persistence (CI −0.053 to +0.002, Fig. 6f). Squared error on tokens is minimized by a shrunken conditional mean, and the frozen token-to-mask decoder turns that shrinkage into a diffuse mask that a fixed threshold penalizes. Training the recovery with a mask term of weight 0.3 lifts held-out mask quality by 0.092 and 0.050 on the two deployments, at a token-error cost of 0.016 and 0.032. A variant trained on masks alone draws masks as good as any other (0.58) while its token error sits at twice the mean-token floor. Mask quality and predictive fidelity are separate axes; we treat predictive fidelity as the acceptance criterion and report mask quality as a floor/model/ceiling triple.

Scope of the field evidence. Field quantities are reported in relative units, and actuated behavior is exercised only in simulation. The withheld view is recovered close to its achievable limit, and the control-conditioned predictor stays ahead of persistence at planning horizons and over self-fed rollouts; per-camera breakdowns and the mask-term arms appear in Section S6 of the supplementary material.

![](images/fc3b33ebebcbf89c195cd5b0d61c58410894c5812e8843530375ae5f13b17d4f.jpg)

![](images/f588d6c59cf8eba106adf2f633893a1b9097e205db112584e4c1770838bc6740.jpg)

![](images/495b825d64abdf6e729ab17ab84b8608243591cd3667a75b7f9868af5ce3f186.jpg)

![](images/554f3ed667d9abb6a1bdf715a9b3a4e736b48c20db72de910926495845d769c4.jpg)

![](images/08ec7d6b7b512e965d8367191faad43d56332c556570b2a0cc9c5e7c31f5f99b.jpg)

![](images/8feac418d29c1234d79b7e26d0093a04e7e8437dd0aff49a73e844a363f7cdfd.jpg)

Fig. 6. Field validation. (a,b) Mask quality on the withheld camera for the mean-token, recovery and mask-term arms; dashed: that camera’s true tokens. (c) Token error of that recovery against the mean-token floor, over the seven slots (bars) and the target (rules). (d) Per-segment token error of a 36-step free rollout against persistence. (e) Error reduction over each reference. (f) Mask quality through the rollout.  
![](images/b3fe6ac14c8ed1f98cf950afeef18275ac6d7bb5f6042dc91c09b69ad7cff29e.jpg)  
(a)

![](images/909159918c144bb7cb4a0cc80484fbf5b187dc0817568607df66fc49ae508eb5.jpg)  
(b)

![](images/a8215ee6c3314d32eb029d7ab7702aad98667588c654f061a8e216c8b11e2efc.jpg)  
(c)

![](images/13f3b2a4f0486ff2ff42eb143f9bf54987de71aaa934db9526165b73df4a38f1.jpg)  
(d)  
Fig. 9. Field experimental conditions. (a) The heavy-duty underwater salvage ROV used in this study. (b) The test pool: 60 m by 20 m, 10 m deep. (c) The aluminum pipe on the pool floor. (d) The steel frame on the pool floor.

C<sup>3</sup>-JEPA is a predictive state model. Binding keeps the target and the gripper in stable slots, the fusion head gathers what the cameras see, the remaining slots hold unmodeled context, and the predictor hands state, predicted consequences and candidate scores to a decision module. In simulation, coupling the interface to the platform’s conventional controller completes the approach, alignment, contact and hold sequence, and the separation between predictive interface and control back end keeps the model portable across platforms. We report this coupling as an interface demonstration, not a powered success-rate study; a closed-loop field trial is the natural next step. At deployment scale the predictor remains small: 1.33 M parameters, 6.5 M FLOPs per window, and 1.97 ms per forward pass on a single A800 GPU (0.6% of the 0.31 s state step), with a 76 MB peak footprint; the latency is a datacenter-GPU reference, and an embedded port of the same predictor is left to future work. Up to Sec. V all evidence is produced in ROS–Gazebo with the simplified attenuation/noise camera model of Sec. IV; the field study extends the evaluation to real

![](images/5aa669c1c2c792626cd1ea681348e1dfb08f024caf8c14e90cea29cb280140fc.jpg)  
Fig. 7. Hidden-camera reconstruction on the two deployments (one row each): the field frame from the withheld camera, the mask decoded from its own tokens, and the mask recovered from the other camera.

underwater video.

## VII. CONCLUSION

We introduced C<sup>3</sup>-JEPA, which instantiates the object-centric perception domain (OCPD) of Sec. I through guided objectcentric cross-view fusion (GO-CVF) and predicts its evolution with a control-conditioned JEPA world model, for physicsbased near-field operation of a heavy-duty underwater salvage ROV with an underactuated, self-adaptive gripper. The model combines low-cost weak binding, held-out-view fusion, free context slots, and control-conditioned latent prediction, internalizing the command-to-motion hydrodynamics without contact sensing. The representation preserves task-object masks through dynamic approach and contact windows (group Dice ≈ 0.9 versus ≈ 0.1 for unguided slots) and predicts the evolution of the object–gripper interaction, cutting persistence’s terminal motion error by a third on the interaction diagnostic and by half at 5 s on the AutoGrasp recordings. Binding guidance improves downstream target-pose estimation even at a higher reconstruction cost, while a small SIGReg weight independently sharpens the geometry head. Against a reconstruction-free latent world model trained on identical data, C<sup>3</sup>-JEPA transfers substantially more task-relevant information to identical probes, showing that object-centric reconstruction guidance preserves information that latent regularization alone discards. The predictive interface supports MPC candidate rollouts, candidate ranking, and imagined-rollout behavior-agent training, and in simulation it is consumed by a conventional automatic controller. On real underwater video the same architecture reconstructs a withheld camera to 81–88% of its achievable range and beats persistence by 23–31% at a five-second horizon. More broadly, the method provides a compact recipe for constructing MPC-compatible world models for low-cost embodied agents with multiple sensing modalities, and the perception-domain view is what keeps that reference affordable: alignment is spent on task objects, and context is carried unaligned.

![](images/d200df376cc259b9b0fd72060f110f14e64ff2d5f256e2497961942c79506840.jpg)  
Fig. 8. Cross-view decoding on a field recording (CAM A top row, CAM B bottom row): the original frame, the mask from that camera’s own tokens, and the merge decoder mask from cross-view fusion.

## ACKNOWLEDGMENT

This work was supported by the National Key Research and Development Program of China under Grant 2023YFC2809701. During the preparation of this work the authors used an AI-assisted writing tool (Hermes) to improve language and consistency; the authors reviewed and edited the content as needed and take full responsibility for the publication. The data and code that support the findings of this study are available from the corresponding authors upon reasonable request. The authors declare no conflicts of interest.

## REFERENCES

[1] Y. Cong, C. Gu, T. Zhang, and Y. Gao, “Underwater robot sensing technology: A survey,” Fundamental Research, vol. 1, no. 3, pp. 337– 345, 2021.

[2] E. Simetti, “Autonomous underwater intervention,” Current Robotics Reports, vol. 2, pp. 119–130, 2021.

[3] L. Sun, Y. Wang, X. Hui, X. Ma, X. Bai, and M. Tan, “Underwater robots and key technologies for operation control,” Cyborg and Bionic Systems, vol. 5, p. 0089, 2024.

[4] E. Morgan, I. Carlucho, W. Ard, and C. Barbalata, “Autonomous underwater manipulation: Current trends in dynamics, control, planning, perception, and future directions,” Current Robotics Reports, vol. 3, no. 4, pp. 187–198, 2022.

[5] L. Birglen, T. Laliberte, and C. Gosselin,´ Underactuated Robotic Hands, ser. Springer Tracts in Advanced Robotics. Springer, 2008, vol. 40.

[6] A. Bicchi, “Hands for dexterous manipulation and robust grasping: A difficult road toward simplicity,” IEEE Transactions on Robotics and Automation, vol. 16, no. 6, pp. 652–662, 2000.

[7] Y. Yang, C. Zhang, F. Wu, J. Li, and X. Wang, “Rapid salvage of deep sea underactuated manipulators: Real-time solution of contact forces and impact assessment of disturbance forces,” Robot, 2025, (in Chinese).

[8] D. Q. Huy, N. Sadjoli, A. B. Azam, B. Elhadidi, Y. Cai, and G. Seet, “Object perception in underwater environments: A survey on sensors and sensing methodologies,” Ocean Engineering, vol. 266, p. 113202, 2023.

[9] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: An accurate open-source library for visual, visual–´ inertial, and multimap SLAM,” IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, 2021.

[10] N. Zhang, S. Zhao, D. An, J. Liu, H. Wang, Y. Feng, D. Li, and R. Zhao, “Visual SLAM for underwater vehicles: A survey,” Computer Science Review, vol. 46, p. 100510, 2022.

[11] B. Joshi, S. Rahman, M. Kalaitzakis, B. Cain, J. Johnson, M. Xanthidis, N. Karapetyan, A. Hernandez, A. Q. Li, N. Vitzilaios, and I. Rekleitis, “Experimental comparison of open source visual-inertial-based state estimation algorithms in the underwater domain,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019, pp. 7227–7233.

[12] S. Rahman, A. Q. Li, and I. Rekleitis, “SVIn2: A multi-sensor fusionbased underwater SLAM system,” The International Journal of Robotics Research, vol. 41, no. 11–12, pp. 1027–1047, 2022.

[13] N. Lyu, H. Yu, J. Han, and D. Zheng, “Structured light-based underwater 3-d reconstruction techniques: A comparative study,” Optics and Lasers in Engineering, vol. 161, p. 107344, 2023.

[14] Y. Ou, J. Fan, C. Zhou, L. Cheng, and M. Tan, “Water-MBSL: Underwater movable binocular structured light-based high-precision dense reconstruction framework,” IEEE Transactions on Industrial Informatics, 2023.

[15] G. Dulac-Arnold, D. Mankowitz, and T. Hester, “Challenges of real-world reinforcement learning,” arXiv preprint arXiv:1904.12901, 2019.

[16] K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep reinforcement learning in a handful of trials using probabilistic dynamics models,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2018.

[17] R. Featherstone, Rigid Body Dynamics Algorithms. Springer, 2008.

[18] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012, pp. 5026–5033.

[19] W. Yuan, S. Dong, and E. Adelson, “GelSight: High-resolution robot tactile sensors for estimating geometry and force,” Soft Robotics, vol. 4, no. 3, pp. 253–263, 2017.

[20] M. Lambeta, P.-W. Chou, S. Tian, B. H. Yang, B. Maloon, V. R. Most, D. Stroud, R. Santos, A. Byagowi, G. Kammerer, D. Jayaraman, and R. Calandra, “DIGIT: A novel design for a low-cost compact highresolution tactile sensor with application to in-hand manipulation,” IEEE Robotics and Automation Letters, vol. 5, no. 3, pp. 3838–3845, 2020.

[21] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations (ICLR), 2020.

[22] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” arXiv preprint arXiv:2301.04104, 2023.

[23] P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg, “Day-Dreamer: World models for physical robot learning,” in Proc. Conf. Robot Learning (CoRL), ser. Proceedings of Machine Learning Research, vol. 205, 2023, pp. 2226–2240.

[24] T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control, 2nd ed. Wiley, 2021.

[25] F. Locatello, D. Weissenborn, T. Unterthiner et al., “Object-centric learning with slot attention,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.

[26] T. Kipf, G. F. Elsayed, A. Mahendran, A. Stone, S. Sabour, G. Heigold, R. Jonschkowski, A. Dosovitskiy, and K. Greff, “Conditional objectcentric learning from video,” arXiv preprint arXiv:2111.12594, 2021.

[27] G. F. Elsayed, A. Mahendran, S. van Steenkiste, K. Greff, M. C. Mozer, and T. Kipf, “SAVi++: Towards end-to-end object-centric learning from real-world videos,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[28] M. Seitzer, M. Horn, A. Zadaianchuk, D. Zietlow, T. Xiao, C.-J. Simon-Gabriel, T. He, Z. Zhang, B. Scholkopf, T. Brox, and F. Locatello,¨ “Bridging the gap to real-world object-centric learning,” in International Conference on Learning Representations (ICLR), 2023, DINOSAUR; arXiv:2209.14860.

[29] A. Zadaianchuk, M. Seitzer, and G. Martius, “Object-centric learning for real-world videos by predicting temporal feature similarities,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[30] Z. Wu, N. Dvornik, K. Greff, T. Kipf, and A. Garg, “SlotFormer: Unsupervised visual dynamics simulation with object-centric models,” in International Conference on Learning Representations (ICLR), 2023.

[31] J. Cheng, J. Wang, Y. Shen, X. Chen, and Z. Yu, “Beyond instance slots: Semantically rich world models for physical interaction planning,” 2026, arXiv:2608.22294.

[32] C. Chen, F. Deng, and S. Ahn, “ROOTS: Object-centric representation and rendering of 3d scenes,” arXiv preprint arXiv:2006.06130, 2020.

[33] N. Li, C. Eastwood, and R. B. Fisher, “Learning object-centric representations of multi-object scenes from multiple views,” arXiv preprint arXiv:2111.07117, 2021.

[34] P. Weinzaepfel, V. Leroy, T. Lucas, R. Bregier, Y. Cabon, V. Arora,´ L. Antsfeld, B. Chidlovskii, G. Csurka, and J. Revaud, “CroCo: Selfsupervised pre-training for 3D vision tasks by cross-view completion,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[35] Y. Seo, J. Kim, S. James, K. Lee, J. Shin, and P. Abbeel, “Multi-view masked world models for visual robotic manipulation,” in Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.

[36] A.-B. Gazzaev, A. Gavrilov, and S. Muravyov, “AquaJEPA: Actionconditioned multimodal predictive representations for underwater robot dynamics,” arXiv preprint arXiv:2607.29393, 2026. [Online]. Available: https://arxiv.org/abs/2607.29393

[37] B. Zhang and C. Yang, “Physics-informed compact world models for predictive control in AUV terminal docking under communication constraints,” International Journal of Intelligent Robotics and Applications, 2026.

[38] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in International Conference on Machine Learning (ICML), 2019.

[39] C. Finn and S. Levine, “Deep visual foresight for planning robot motion,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2017.

[40] F. Ebert, C. Finn, S. Dasari, A. Xie, A. Lee, and S. Levine, “Visual foresight: Model-based deep reinforcement learning for vision-based robotic control,” arXiv preprint arXiv:1812.00568, 2018.

[41] N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations (ICLR), 2024.

[42] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[43] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[44] R. Balestriero and Y. LeCun, “LeJEPA: Provable and scalable self-supervised learning without the heuristics,” arXiv preprint arXiv:2511.08544, 2025.

[45] Y. Wang, O. Bounou, Y. LeCun, and M. Ren, “AdaJEPA: An adaptive latent world model,” arXiv preprint arXiv:2606.32026, 2026.

[46] P. Rao, W. Zhang, R. Balestriero, Y. LeCun, and G. Loianno, “SkyJEPA: Learning long-horizon world models for zero-shot sim-to-real control of quadrotors,” 2026, arXiv:2606.23444.

[47] H. Nam, Q. L. Lidec, L. Maes, Y. LeCun, and R. Balestriero, “Causal-JEPA: Learning world models through object-level latent interventions,” arXiv preprint arXiv:2602.11389, 2026.

[48] L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “LeWorld-Model: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.

[49] U. M. Khan, “Depth-regularized JEPA world models learn more transferable representations from real outdoor robot data,” 2026, arXiv:2607.16314.

[50] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab´ et al., “DINOv3,” arXiv preprint arXiv:2508.10104, 2025.

[51] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, best Paper Award.

[52] M. M. M. Manhaes, S. A. Scherer, M. Voss, L. R. Douat, and˜ T. Rauschenbach, “UUV simulator: A gazebo-based package for underwater intervention and multi-robot simulation,” in OCEANS 2016 MTS/IEEE Monterey, 2016.

[53] Q. Garrido, R. Balestriero, L. Najman, and Y. LeCun, “RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank,” in Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.

[54] J. Hang and Z. Cai, “Identifying habit, physics, and nuisance in robot world models,” 2026, arXiv:2609.09210.

[55] H. Zhang, Y. Li, S. He, T. Nagarajan, M. Chen, J. Lu, A. Li, and Y. Fu, “ThinkJEPA: Empowering latent world models with large vision-language reasoning model,” 2026, arXiv:2603.22281.

[56] K. Wen, Z. Li, S. Luo, and F. Shi, “JEPA-x: Cross-predictive physics grounding for forecastable latent dynamics,” 2026, arXiv:2608.24044.

[57] J. Wang, K. Rui, Y. Zuo, Y. Feng, and M. Li, “Decision-metric alignment in latent world models: Diagnostics and action-conditioned objectives for MPC planning,” 2026, arXiv:2608.18746.

[58] E. Aljalbout et al., “The reality gap in robotics: Challenges, solutions, and best practices,” 2026, arXiv:2510.20808.