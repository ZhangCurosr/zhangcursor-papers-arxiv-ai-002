# Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection

Li Guan,<sup>\*</sup>Rachael Granberry<sup>\*</sup>, Sahar Aseeri<sup>\*</sup>, Sahar Rostami<sup>\*</sup>, Taylor Niehues<sup>\*</sup>, Sai Dhanooshi Marri<sup>\*</sup>, Mia   
Huang<sup>\*</sup>, Dingtian Zhang<sup>\*</sup>, Ziyi Kou<sup>\*</sup>, Jason Pursel<sup>\*</sup>, Eric Whitmire<sup>\*</sup>, Vatsal Mehta<sup>\*</sup>, Ashish Deshpande<sup>\*†</sup>,   
Tristan Trutna<sup>\*</sup>, Reed Belcher<sup>†</sup>, Emelia Black, Julian Blake<sup>†</sup>, Roger Boldu, Sean Bonaventure, Einar   
Boyovich, Jordan Brennan<sup>†</sup>, Elizabeth Casillas<sup>†</sup>, David Christensen, Tyler Christensen, Nicholas Colonnese,   
Stuart Dealey, Gaige DeHaven, Katie Doran, Chris Doughty, Robert Drury<sup>†</sup>, Silvia Espino, Gabriel Geyne,   
Beth Goodrich<sup>†</sup>, Kyra Gould, Joe Grisso, Susana Guerrero<sup>†</sup>, Andrew Gwozdziowski, Monica Ha, Ehsan   
Hajiesmaili, Kate Healy, Rose Hope, Michael Hurst, Claire Ikemiya, Daisy Isibor, Rick James, Yashasvi   
Joginapally, Samuel Johnson<sup>†</sup>, Tyler Jorgensen<sup>†</sup>, Kristy Jost, Ben Kamath<sup>†</sup>, Slava Karulin<sup>†</sup>, Alexander   
Keener, Woo Kim<sup>†</sup>, Alison Krauskopf, Ankit Kumar, Joshua Lang, Lizzy Li, Cameron Lynch, Daniel Ma,   
Hector Martinon, Christina Matos, Keagan McCurdy, Lee Mincey, Stuart Nezlek<sup>†</sup>, Tiffany Nguyen, James   
O’Hara<sup>†</sup>, Jason Osborne, Jacky Pai, Roham Pardakhtim, David Perek, Alexandre Poulin, Nick Raines,   
Salwa Rasheed, Leah Resneck, Kevin Rhodes, Ergys Ristani, Douglas Roberts<sup>†</sup>, Anik Sarker, Harrison   
Scarborough, Jonas Schmidtler, Brian Shieh, Benjamin Shuman, Garrett Simard, Forrest Smith, Udara   
Somarathna, Eliot Sosin, Jonathan Stewart, Kristine Strom, Adam Swett, Hal Talbot-Mantooth, Julia Tanner,   
Sai Sharan Thirunagari, Curtis Torrey, Simge Uzun, Eileen Wachs<sup>†</sup>, Rylan Wasson, Henry Wei, Jesse White,   
Beau Woodbyrne, Antonio Woods, Noah Young, Chandler Yount, Joseph Zhang, Sean Keller

Meta Reality Labs Research

<sup>\*</sup>Equal first authors to this work.

<sup>†</sup>Work done while at Meta.

Abstract—Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning—yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-session calibration for subject-dependent correction. To enable objective comparison, we introduce standardized benchmarking protocols for tactile force regression and hand pose estimation, and apply them to Tactus and two commercially available tactile gloves — to our knowledge the first quantitative cross-platform evaluation of this kind. Tactus resolves a minimum detectable force of 0.59 N at 95% confidence, 1.6–2.7× lower than the commercial baselines, and with 0.87 cross-participant consistency, while our pose tracking sustains 11.0 mm fingertip keypoint relative error during manipulation, when occlusion is heaviest. We show that these combined signals support object property estimation, slip detection, and representative recording of hand-object interaction — yielding demonstration data with the fidelity required for contact-aware manipulation policies and

world models.

Index Terms—Perception for Grasping and Manipulation, Dexterous Manipulation, Sensor Fusion

## I. INTRODUCTION

Learning dexterous manipulation from human demonstrations is a promising path toward general-purpose robot skills, and the quality of that demonstration data has been shown to directly impact downstream policy performance [1]. There are three major options for capturing human manipulation sequences: egocentric-Computer-Vision (ego-CV) only; CV with an exoskeleton-based solution; and CV with a sensingglove-based solution.

The ego-CV-only approach is the least intrusive of the three and obtains the most natural manipulation data, but CV cannot recover high-fidelity hand pose under occlusion, which becomes more severe exactly when hand-object interaction happens. Paliwal et al. reported that in ego-CV recorded manipulation tasks, < 10% of the raw data eventually are usable for training purposes [2]. Besides limitation in measuring hand pose, vision-based methods cannot directly measure the contact forces that are important for fine dexterity tasks. Exoskeleton-based systems like MILE [3] provide high-fidelity kinesthetic data but constrain finger motion and alter grasp strategies; recent systems such as WHED [4] and DexEXO [5] achieved some progress, but still made great trade-off of naturalism with data quality. Sensing gloves promise both naturalism and direct force measurement, yet as we detail in the following section, no existing platform jointly provides calibrated force sensing, occlusion-resilient pose tracking, and a form factor validated to preserve natural grasp behavior.

![](images/4369833c4cacd73ab06a541b38aacddc5be6d2697c6e13841106e5bc2d2ea931.jpg)  
Fig. 1. Form-factor of Tactus digital sensing gloves capable of high-fidelity tactile and pose sensing for natural hand-object manipulation data collection.

We present Tactus, a textile-integrated tactile and pose sensing glove designed from the ground up for naturalistic manipulation data collection, as shown in Fig. 1. It is a 3Dknit textile glove, built in left- and right-hand variants, carrying 47 capacitive textile force sensing locations spanning fingertip to palm and 16 6-DoF IMUs; Sec. III-C details the sensing and electronics. Besides the glove design, our major contributions are the following:

• Minimal interference with natural grasp behavior, validated with human manipulation studies (Sec V-A). A thin, conformal 3D-knit base glove (1.5 mm fingertip stackup) with low-profile, stretchable embedded sensors minimizes encumbrance while preserving the fidelity of natural manipulation.

• Data-driven tactile sensor placement (Sec. III-A) and efficient calibrated force sensing at every contact location (Sec. IV-A). Placement is optimized by ink studies and a sensor-compliance threshold that maintains 65% fingertip coverage across a broad anthropometric range in a single glove size. Calibrated forces are recovered in two stages: a scalable pneumatic-pressure factory calibration and a per-session online procedure.

• Standardized benchmarks. We introduce Protocol-Tactile (normal contact force Sec. V-B) and Protocol-Pose (hand pose tracking Sec. V-C) and apply to both Tactus and commercially available gloves, yielding the first reproducible quantitative cross-platform comparison.

• Application validation (Sec. VI). Tactus tactile and pose data enables force-dependent robotic perception and manipulation tasks—object material classification, slip detection, and multi-sensory hand object manipulation.

## II. RELATED WORK

Vision-Only Capture with Indirectly Sensed Contact / Force. CV-only human hand object manipulation data capturing infers contact, and increasingly force, from imagery alone, trading traceable measurement for unencumbered, in-the-wild scale. Hand pressure has been regressed from a single RGB image using appearance cues such as skin blanching and softtissue spread [6], extended to uncontrolled imagery [7] and to the egocentric viewpoint [8], while contact-only capture matured through physical proxies such as thermal residue [9]. Recent work moves from planar pressure to volumetric hand– object interaction, jointly estimating dense per-vertex contact and 3D force on hand and object meshes with force supervision synthesized in simulation [10], annotating inthe-wild egocentric clips with dense 3D contact correspondences [11], and targeting full-hand grasp pressure for everyday objects [12]. While these approaches are least intrusive and capture the most natural manipulation, their limitations are also obvious: their signals are indirectly sensed—predictions supervised by simulation, by instrumented surfaces that constrain what can be manipulated, or by proxies that recover contact but not magnitude—so a calibrated, occlusion-invariant, worn measurement stream is precisely the supervision and evaluation signal they lack.

Sensing Wearables for Manipulation Data Collection. Wearable tactile sensing for manipulation data has evolved along two axes: spatial coverage and force measurement fidelity. The STAG glove [13] achieved unprecedented coverage with 548 pressure sensors across the full hand, but without per-sensor calibration its force estimates remain unreliable; the OSMO platform [14] captures finger–object interaction with 12 magnetic sensors, again with no calibration pipeline; and commercial gloves from PPS [15] and JQ [16] offer fingertip force sensing in production-ready form factors but their performance on real manipulation tasks is not welldocumented. The downstream value of calibrated tactile data is nonetheless established: Feel the Force [17] achieved zeroshot transfer of human tactile demonstrations to robot policies, and MILE [3] reduced both tracking and tactile error with an exoskeleton — at the cost of constraining the very finger motion and contact whose grasp strategies it seeks to capture.

Hand Pose Tracking for Manipulation. Hand tracking during manipulation faces an inherent tension: the moments of greatest scientific interest—hand–object interaction, bimanual coordination, power grasps—are precisely when vision-based tracking degrades due to occlusion. Monocular and multiview estimators [18], [19] have advanced rapidly, yet their evaluation is dominated by occlusion: hand–object benchmarks [20], [21] were built to expose exactly this regime, and the multi-method challenge report of Fan et al. [22] shows that no submitted method resolves hand–object interaction reliably under egocentric viewing. Marker-based mocap supplies the ground truth for these datasets and for corpora such as GRAB [23], but requires an instrumented volume and collapses when many markers are simultaneously invisible across all views, for which added modalities are the likely remedy [24]. Gloves instead measure articulation directly and are occlusion-invariant by construction—via resistive or capacitive strain [25], electromagnetic fields [26], or helical stretch yarns with IMUs [27]—but each modality carries a characteristic error source (strain hysteresis and per-wearer recalibration, field distortion, inertial drift and magnetic interference) [28], and validation studies find accuracy and repeatability differing markedly across technology and joint [29], which is largely why glove-derived pose is rarely used as a quantitative reference. Sparse-IMU fusion, well established for full-body pose [30], has been extended to hands with inertialonly estimators on instrumented gloves [31] and then with learned cross-sensor attention that re-weights egocentric visual features against on-glove IMU signals as occlusion varies [32]. Tactus adopts this fusion formulation and places it on the same garment as calibrated tactile sensing, so that pose and contact force are captured in a single, co-registered stream.

Tactile Sensing and Calibrations. Wearable system force calibration remains under-studied. Exoskeleton trades manipulation naturalism with signal consistency. Calibrating a soft, stretchable tactile glove is challenging because of its welldocumented strain artifact [33], especially in the worn state, where fit, grounding, and hand pose perturb the transfer function [34]. So existing work treats glove and wearer as an inextricable system and absorbs session variability through online calibration [14], [35], [36], or end-to-end learning without any force reference [13], [37]. To date no method characterizes full-glove tactile performance without confounding from the wearer. A different approach loads all elements simultaneously with a known, spatially uniform pressure. A pneumatically controlled bladder with a thin elastic membrane conforms to the Device Under Test (DUT), and sweeping the applied pressure yields per-sensor transfer functions. It shows promise on robot skins [38], insoles and other wearables [39], and commercial sensor arrays [40], but only on rigid or planar objects; uniform pressurization of a soft glove with inherently 3D, non-developable curvature is unaddressed, and is what our factory calibration method provides.

## III. SYSTEM DESIGN

Our design choice is driven by a single requirement: capturing naturalistic manipulation demonstrations with quantitatively reliable force and pose data. While all gloves alter the pose and contact forces used in object interaction [41], we isolated three properties that can be tuned to minimize such alteration to preserve naturalistic hand-object interactions: thickness, stiffness, and friction.

Thickness. Task completion time increases approximately linearly with glove thickness [41]. Our task coverage evaluation (Protocol Dexterous-Tasks, details in Sec. V-A) also confirmed this relationship: glove configurations exceeding 1.5 mm fingertip thickness produced measurable increases in task time and technique adaptations. Tactus (1.5 mm) reached similar barehand task completion time, best among all tested sensing glove candidates, thanks to our sensor design, which was knitted directly into the glove fabric—eliminating the added layer of discrete sensor laminated over or sandwiched between an existing textile stackup.

Stiffness. As shown in Fig. 2-B, low bending and tensile stiffness allow a glove to stretch around compound finger curvatures. Conformal fit preserves natural grasp behavior and ensures sensors remain coupled to the hand surface during contact—oversized gloves contact the environment before the hand does, and gloves that are either too big or too small increase task time by up to 10 % [42]. The Tactus sensor achieves 0.20 N/cm bending stiffness and 0.22 N/cm tensile stiffness per 25 mm sample width, conforming to the hand and tracking it through flexion. In contrast, PPS glove [15] necessitates an oversized fit in which the sensor stands off from the hand surface (Figure 2-A). This stiffness-fit coupling is reflected in encumbrance outcomes: Tactus achieved correspondingly lower NASA TLX load (smaller the better) [43] (2.97 vs 3.10) (full details in section V-A).

![](images/27be3cebbec68af9befba54f0b509c8ba684172df1d52d19ec570ff9faea6de9.jpg)

Fig. 2. Glove fit sets whether sensor contact means hand contact. A) An oversized glove lifts the sensor off the skin, so it touches first. A conformal glove tensions sensors against the hand. B) Compound curvature demands sensor multi-axial conformation.  
![](images/ec72f60c6b3e1f2968d818f0041da869139db35f916d58e4c6c81c75bfef2a7e.jpg)  
Fig. 3. Knitted tactile capacitive sensor mechanism. The conductor and insulation layers form a capacitor whose capacitivity changes when compressed by external force.

Friction. Surface friction must be high enough to maintain stable grasp without excessive grip force, yet low enough to permit the micro-slips essential for in-hand manipulation, such as pivoting and re-orientation [44], [45]. The human skin coefficient of friction (COF) varies widely (0.4-1.0) against different surfaces [45], and grip force is continuously modulated to maintain a safety margin above the slip threshold [44]. Tactus design achieves sufficient interface friction through thermoplastic polyurethane (TPU) surface treatment in the fingertip regions, ensuring 0.4 COF (static friction on polished steel), a moderate value where stable grasp and slip prevention are achievable for dexterous manipulation.

## A. Tactile Sensing Design

The Tactus tactile sensor is a capacitive pressure sensor knitted directly into the textile substrate as a double-layer weft knit, giving a low-profile, fully embedded sensor, as shown in Fig. 3. The electrodes are custom insulated conductive wires, machine-knittable on industrial equipment, with knit structure chosen to meet sensor performance requirements: each capacitive cell pairs a half-gauge-knit upper electrode with an interlock lower layer, and its capacitance varies with applied normal force.

We determine tactile sensor placement through three empirical studies: contact frequency mapping, hand geometry constraint analysis, and population coverage assessment.

Contact frequency. An ink transfer study (N=3, 13 manipulation tasks) mapped contact frequency across hand regions (Fig. 4 A). Contact favored the radial side for index-pinky fingers and ulnar side for the thumb rather than the centerline. These frequencies establish coverage targets that minimize device complexity without sacrificing tactile data collection.

Hand geometry constraints. Sensors cannot occupy all high-contact regions. Inking fully flexed hands (N=6) mapped fold lines across the finger and palmar surfaces; these compress during flexion and generate forces indistinguishable from object contact, so sensors must not span them (Fig. 4 B). Additionally, 3D scanning of gloved and un-gloved hands across multiple postures located palmar concavities (thenar and hypothenar valleys), where the glove bridges the hand surface rather than maintaining sensor–skin coupling, producing dead zones that cannot register force even under load (Fig. 4 C). Both define exclusion zones that refine the contact-driven coverage targets into specific sensor footprints.

Population coverage. A single glove size must accommodate a broad anthropometric range without sacrificing contact coverage. Population analysis established the threshold: sensors with an effective elastic modulus above 35 N/cm (thickness excluded from calculations) behave as effectively inextensible in a glove system (Fig. 5). The Tactus sensor (approximately 7 N/cm) remains well below this threshold, preserving glove elasticity through the sensing region (Fig. 5 A). Sensor footprints are further constrained to less than 50% of the finger circumference (Fig. 5 B), leaving the remainder free to stretch and recover with a surrounding base textile made of high-denier elastane yarns for maximum compliance and elastic recovery, preventing the glove from bagging over repeated use. Together these choices maintain 65% fingertip coverage across a broad population with one discrete glove size, whereas effectively inextensible sensors must limit coverage to approximately 40% or require additional sizes.

Based on the above studies, we finalize our tactile sensor placement as in Fig. 1, each distal phalange carries a 4×2 array (8 independent sensing elements), while other phalanges carry at most single-element (1×1) sensors. The total system comprises 47 sensor locations. In addition, a 12 pF reference capacitor improves ADC resolution, i.e. the number of discrete output codes an analog-to-digital converter can produce and reduces minimum detectable force (MDF).

## B. Pose Tracking Design

We aim to address the visual occlusion limitation of the ego-CV-only hand tracking solutions [18], [19] during hand-object manipulation demonstrations for robotic tasks. Similar to [31], we use distributed IMUs that are invariant to visual occlusion, fused with egocentric vision so that each modality covers the other’s failure mode: vision supplies absolute position and a drift-free global reference, while IMUs keep updating joint angle estimates during visual occlusions.

![](images/4bebcf4099e20f0d097ff86dabd87bb0e098e92a29a0f08e65b2cc078df398f7.jpg)  
Fig. 4. Sensor layout was informed by three datasets. A) A contact frequency map identifies where sensors are worth placing. B) Inking the flexed hand leaves skin folds un-inked, ruling out those regions because sensors there register self-contact. C) 3D scans of gloved- & bare-hands rule out glove tenting regions, e.g. the palm center, where sensors contact the world first.

![](images/c2cc0f984d4c8ddef4d30ffa9c9d4bf52a9d05faa08ba0cf769fb92ca032680e.jpg)  
Fig. 5. Two ways to widen population coverage. A) Increase the elasticity of the sensor. B) Reduce the sensor coverage area, which helps when the sensor is less stretchable than the base glove.

Three principles shape the layout. First, every phalange is instrumented so that each finger joint angle is directly observable. Second, all units are 6-axis (accelerometer and gyroscope) rather than 9-axis: omitting the magnetometer removes the failure mode when approaching ferrous objects or active magnetic fields. Third, the IMUs are co-located with the tactile array board, rather than independently placed. The resulting configuration lands three 6-axis IMUs on each digit and one in the dorsum enclosure as a root reference (16 total), each sampling at 200 Hz (as in Fig. 1).

## C. Supporting Electronics

Low encumbrance sensors require equally compliant supporting electronics—rigid wiring that cannot elongate with the finger restricts joint motion and negates the sensors performance. Electrical routing addresses this encumbrance challenge by coiling conductors into a spring geometry: a mylar-wrapped, kevlar-core construction with TPU overmold, achieved ≥70% elongation at 0.60 N, >25 N failure load, and 168K cycles to 70% elongation—providing the required stretch at a spring rate low enough to avoid perceptible resistance during natural finger motion.

![](images/9c6f3e62740f9641efb542f0ae0f9c699374a20742be45685370a48a56888905.jpg)  
Fig. 6. Factory calibration rig design. A closed pneumatic chamber in which the glove, donned on a prosthetic hand, is inflated against from above by a thermoformed elastomeric membrane with regulated air pressure.

All dorsum-mounted boards are housed in knit textile pockets rather than overmolded to the glove surface to preserve the surrounding textile’s elasticity and prevent localized stiffening that would increase encumbrance. Each finger carries three phalange-level island PCBAs, each integrating a PSoC 6 microcontroller, an IMU (ICM-47688), and analog sensor connections, communicating over a UART finger bus. A dorsum-mounted dual-board controller stack provides WiFi+BLE communication, micro SD storage (32 GB+), and USB-C (6.5 Mbps) for data download and charging. A 900 mAh battery provides 3.5+ hours of untethered operation. Time-sync uses BLE or a SubGHz transceiver (S2-LP).

## IV. SIGNAL MODELING

## A. Tactile Calibration Pipeline

Characterization and factory calibration Our calibration rig characterizes all 47 sensor locations of a complete glove simultaneously in under 23 min. As shown in Fig. 6, an elastomeric membrane thermo-formed to the hand shape presses against the assembled glove donned on a prosthetic hand, applying controlled, repeatable (≪0.01 N) pressure stimuli with ≪0.1 N location-wise variation; each sensor is driven through two [0, 15] PSIG step-cycling sequences. We then regress raw signal to pressure with a per-sensor cubic transfer function $Y = A ( X - B ) ^ { 3 } + C .$ , and extract full span, sensitivity, hysteresis, noise, baseline, and fitting error $( R ^ { 2 } )$

With over 15 hours of forceful use of three Tactus gloves with periodic characterization, we find no evidence of localized sensor degradation: a one-sided Wilcoxon signed-rank test is null across most KPIs $( p > 0 . 0 5 )$ , the sole exception being zero-load drift on two gloves $( p = 0 . 0 0 4 , 0 . 0 4 )$ , which zero-offset taring mitigates. The drift we do observe is global and not attributable to sensor wear. Full span falls ∼3% per don/doff session on test and reference gloves alike, consistent with viscoelastic fatigue from repeated stretching; hysteresis grows a further ∼3% per session on test gloves only, consistent with inter-layer friction from physiological fouling. To address the glove-level sensor drift, along with don/doff variations and fit & grounding differences, per-session calibration is needed with human-in-the-loop.

TABLE I  
FACTORY VS. PER-SESSION ONLINE CALIBRATION ACCURACY
<table><tr><td>Method</td><td>Median full-glove force RMSE (N)</td><td>Median Relative Error</td></tr><tr><td>Factory cal.</td><td>2.50–5.22</td><td>40.6%-98.1%</td></tr><tr><td>Online cal.</td><td>1.46-1.82</td><td>27.7%–40.9%</td></tr></table>

Online per-session calibration. Participants wear the glove and apply normal force from fingertip to a reference sensor. Force from each element is calculated by multiplying its area and pressure (using sensor output and factory calibration), and then aggregated over all fingertip elements to compare with the normal reference. To correct factory calibration, a scaling factor α is applied to the aggregated force, where α is derived from the best fit of predicted vs actual force $Y = \alpha X$

As shown in Table I, a single per-session online scaling factor improves median force RMSE by 1.7–2.9× and median relative error by 1.5–2.4× over factory calibration alone. A one-parameter correction is thus sufficient to make reported forces comparable across gloves and sessions, which indicates that the dominant residual after factory calibration is a wearerspecific gain term — fit, grounding, and hand pose — rather than per-sensor error.

## B. Hand Pose Tracking Pipeline

The glove design enables different hand pose tracking modes. In this section, we highlight a few representative options, spanning the operating points a data-collection platform requires from real-time tracking for tele-operation to occlusion-agnostic offline solver feeding imitation policy training. Table IV in Sec.V-C reports their tracking performance against marker-based mocap (OptiTrack) as the ground truth.

CV+IMU Extended Kalman Filter (EKF) fusion. We implement a ResNet-based ego-CV hand pose estimation network [19] that consumes ego-CV video streams and regresses a 26-DOF hand pose states, fine-tuned on both bare-hand and gloved-hand images. An EKF then fuses the on-glove IMUs, augmented with joint velocities, accelerations, and onlinecalibrated IMU-to-bone orientation parameters. Each update propagates state and covariance forward in time, then predicts expected sensor values from the current state and corrects the estimate against actual measurements (CV keypoints, IMU accelerometer and gyroscope). It consumes the CV and IMU observations asynchronously and runs in real-time.

Adaptive vision-IMU Deep-Learning Fusion (AVI-HT). A multi-modal deep learning network replacing the explicit EKF is implemented [32]. For each video frame, IMU samples are encoded into tokens, which are fused with a global visual token through a hierarchical cross-sensor attention modules to implicitly model kinematic prior and local sensor relationship under occlusion. This is also a real-time solution.

AVI-HT Offline Full-Sequence Solver. A full-sequence non-linear optimization solver refines the hand trajectory provided by AVI-HT by leveraging the complete temporal context. This method trades real-time capability for tracking accuracy, as a good candidate for dataset quality curation.

TABLE II  
WEARABLES TASK COVERAGE PERFORMANCE VS BARE-HAND BASELINE
<table><tr><td></td><td>福 Barehand (baseline)</td><td><img src="images/323bafc83cef6ed98a1a6091cbc2b2687541a810c0259ccbc48d3b99fb10c82e.jpg"/> PPS</td><td><img src="images/713f79c73edb1bd7125bc9c4d7d0ae36caedd89eea16d2746dccec74f4319f30.jpg"/> JQ</td><td><img src="images/a90ff64f92d8c91f1702bf1121d0a95d2e59dcb50dd764de0b9ae00c0fa039ab.jpg"/> Exo</td><td><img src="images/ee16043a7fd029b90a526f5161ecfff34b701404856371689d1f22e2310fce69.jpg"/> Tactus</td></tr><tr><td>Completion Time Ratio ↓</td><td>1.00</td><td>1.13</td><td>1.11</td><td>1.89</td><td>1.06</td></tr><tr><td>Task Success Rate ↑</td><td>100%</td><td>96.5%</td><td>100%</td><td>95.7%</td><td>100%</td></tr><tr><td>NASA TLX [1—7] ↓</td><td>1.87</td><td>3.10</td><td>3.43</td><td>5.23</td><td>2.97</td></tr></table>

## V. PERFORMANCE EVALUATION

To assess whether Tactus supports dexterous manipulation data collection, we introduce three protocols measuring (A) how much the device interferes with natural hand manipulation task performance, (B) tactile-force sensing quality, and (C) hand pose estimation quality.

## A. Protocol Dexterous-Tasks

To assess whether a sensing wearable preserves natural manipulation behavior, we conducted a within-participant encumbrance study (N=5) comparing barehand manipulation as baseline against our Tactus glove, a few state-of-theart human manipulation tactile gloves (PPS glove [15] and JQ glove [16]), and one customized exo-skeleton device, as shown in Tab. II. For a total of 23 household, personal, and standardized dexterity tasks, we recorded task completion time ratio compared with barehand time, success rate, and adapted the NASA TLX task load [43], which is a subjective, selfreported workload metric that combines six subscale ratings (mental, physical and temporal demand, performance, effort, and frustration) into one overall score on a 7-point Likert scale.

Tactus achieves the best performance among the tested wearables against all the three metrics, validating our assumption that our design achieves the closest resemblance to barehand object manipulation compared with other wearables. With almost the same task completion time and slight trade-off of subjective workload, what we gain is high quality perceptual information for hand-object manipulation data.

## B. Protocol Tactile

Participants (N=3) perform varying-pose fingertip pressing tasks as shown in Tab. III Left. We then train an MLP force regression model for each of glove candidates supervised by the 6DoF force-torque sensor (Nano-17), and report 5-fold validation about the sensor performance. The protocol captures ecological performance including don/doff variation and Pose-Related Artifacts or PRA (changes in tactile sensor signals caused by strain from hand poses, rather than by contact force) [46]. To measure that, we define PRA ratio as the maximum per-channel ratio between the relative signal during in-air free motion and 3N normal load after the base signal load compensation, or $m a x ( | S _ { i n _ { - } a i r \_ m o t i o n } - S _ { b a s e } | ) / m a x ( | S _ { 3 N \_ l o a d } - S _ { b a s e } | )$ For an ideal sensing wearable, its PRA ratio should be zero.

TABLE III  
TACTILE BENCHMARK RESULT. LEFT: TESTING RIG, AND PRESSINGLOCATIONS AT THE FINGERTIP. RIGHT: TACTILE GLOVES’ FORCEREGRESSION PERFORMANCE COMPARISON.  
![](images/f37bacdda487de16b324eab45d220a569da8144206d44104d6aeb66e6d50b218.jpg)

<table><tr><td>Metric</td><td>PPS</td><td>JQ</td><td>Tactus</td></tr><tr><td>MDF@95% (N)↓</td><td>1.58</td><td>0.94</td><td>0.59</td></tr><tr><td>Fingertip force error (N)↓</td><td>0.23</td><td>0.18</td><td>0.20</td></tr><tr><td>X-subj. consistency↑</td><td>0.77</td><td>0.86</td><td>0.87</td></tr><tr><td>PRA ratio ↓</td><td>2.13</td><td>0.43</td><td>0.17</td></tr></table>

Results in Tab. III Right show that comparing with PPS and JQ tactile sensing gloves, Tactus achieves the lowest Minimum Detectable Force (MDF), the highest cross-participant consistency, and the lowest PRA ratio, all thanks to its design choices and calibration process. The JQ glove achieves slightly better point accuracy (0.18 N vs. 0.20 N)—but neither JQ nor PPS provides a documented calibration pipeline to maintain performance across subjects or sessions.

Note that the results here are not directly comparable with the calibration performance in Sec. IV-A, where Tab. I gives per-sensor cubic transfer functions across all 47 locations over the full pressure range, while here Tab. III gives a learned multi-channel MLP for fingertip normal force at the lighter forces for glove-to-glove performance comparison purposes.

## C. Protocol Hand Pose

Three tracking modes as in Sec.IV-B are evaluated against an ego-CV bare-hand baseline UmeTrack [19]. The ground truth pose is obtained by attaching OptiTrack mocap markers to hand key points, and solving the inverse kinematic pose. Participants (N=5) each perform a fixed set of 16 object manipulation tasks spanning dexterous manipulation task space (e.g. YCB object manipulation [47]). As shown in Tab. IV, MKPE reports mean keypoint position error over the whole hand and F.MKPE restricts it to the fingertips; the MKPE.T and F.MKPE.T variants are computed after translation alignment at the wrist, isolating joint-angle accuracy from absolute positioning error. The transformed fingertip accuracy is the most relevant quantity for manipulation analysis, where relative finger geometry determines contact.

The fused modes are all more accurate than the CVbarehand [19] baseline. Adding on-glove IMUs to the causal vision estimate reduces root-aligned error by 11% and rootaligned fingertip error by 17%, while absolute error improves by under 3%—confirming that the IMUs supply joint-angle information rather than global position, which vision already provides. AVI-HT inverts this profile: its largest gains are in the absolute metrics (34% MKPE, 33% F.MKPE relative to the vision-only baseline) while retaining the relative improvements, indicating that attention over per-sensor inertial features recovers global hand configuration that the explicitly-modelled filter does not. Full-sequence optimization improves all four metrics further by exploiting complete temporal context, making it the preferred mode for post-hoc dataset curation where latency is irrelevant.

![](images/322293da05e8461c60b88d5fde1a87a10dfcf0d1ef12698c849a6f6fa72a5b14.jpg)  
Fig. 7. We design a rig to test controlled slip events with measured force and displacement. We can automatically label different stages of slip from the ground truth sensory signals.

## VI. APPLICATIONS: FORCE-DEPENDENT TASKS

The benchmarks above establish that Tactus captures calibrated fingertip force and occlusion-resilient hand pose. We now present some capabilities that enable downstream manipulation tasks that indirect sensing cannot support—the core value of Tactus for high-fidelity, natural data collection.

## A. Surface Material Estimation

Inferring object material properties provides critical context from hand-object manipulation, and is a direct test about the tactile signal fidelity. We evaluate a 9-category roughness classification on Tactus. As shown in Tab. V, given a set of 9 texture pads for children cognitive development, participants slide the index finger across each pad while the glove records the tactile signal, and we compare two models: a tactile-only network consisting of a stack of transformer modules followed by an MLP head, and a zero-shot vision–language baseline (Llama-3.2-11B-Vision-Instruct) that predicts the texture category from an egocentric image and a text prompt. The two models are expected to exhibit complementary strengths— the tactile network resolving fine surface structure that is not visually apparent, the VLM exploiting semantic priors about materials—which motivates tactile-augmented visual language modeling as a direction for object attribute inference.

## B. Slip Detection

Detecting the onset of slip, i.e. when an object starts to move relatively to hand, is fundamental to stable manipulation. It requires the capability to detect rapid directional force transients. As shown in Fig. 7, a rotary stage pulls a sled via a cable and spring; the DUT sits on a tower above it. The spring ramps pull force smoothly until static friction breaks, then decompresses so the sled re-sticks, producing repeated natural stick-slip cycles rather than a single commanded shear. Ground truth labels (NO-SLIP / PRE-SLIP / SLIP) come from encoder position plus tangential load.

TABLE IV  
HAND POSE BENCHMARKS PERFORMANCE
<table><tr><td>Tracking Mode</td><td>MKPE (mm) ↓</td><td>MKPE.T (mm) ↓</td><td>F.MKPE (mm) ↓</td><td>F.MKPE.T (mm) ↓</td></tr><tr><td>CV-barehand [19]</td><td>15.8</td><td>8.9</td><td>19.8</td><td>17.1</td></tr><tr><td>CV+IMU EKF</td><td>15.4</td><td>7.9</td><td>18.7</td><td>14.2</td></tr><tr><td>AVI-HT [32]</td><td>10.4</td><td>7.0</td><td>13.3</td><td>12.8</td></tr><tr><td>AVI-HT offline</td><td>9.7</td><td>6.4</td><td>12.1</td><td>11.0</td></tr></table>

TABLE V  
SURFACE ROUGHNESS 9-CLASS ACCURACY TABLE. THE TACTILE NETWORK VS VLM ZERO-SHOT PERFORMANCE.
<table><tr><td>Model Overall Mirror Bushy Scaly Bumpy Braided Ridged Fuzzy Fluffy Hairy</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tactile</td><td>0.75</td><td>0.78</td><td>0.83</td><td>0.66</td><td>0.81</td><td>0.67</td><td>0.61</td><td>0.67</td><td>0.97</td><td>0.84</td></tr><tr><td>VLM</td><td>0.56</td><td>0.84</td><td>0.00</td><td>0.96</td><td>0.97</td><td>0.08</td><td>1.00</td><td>0.86</td><td>0.45</td><td>0.00</td></tr></table>

Across 7 sessions (5 training, 2 testing; ∼34 slip events each), a compact causal temporal CNN consuming a 4 sec window of the 4 × 2 fingertip pressure channels detected slip with F1 = 1.00 on a single train/test split, firing within 0.01 sec of labelled onset. Although Tactus outputs pressure only and performs no force decomposition, the multi-channel signal profile is distinctive enough to resolve an event whose defining physics is the onset of tangential relative motion. We attribute this to the spatial distribution of the array: as the contact area shears, the pressure distribution deforms in a characteristic pattern, so shear is represented implicitly without being measured directly.

The account remains under-determined, however, since a normal-pressure distribution does not uniquely specify shear. We regard this only as a feasibility result; follow-up experiments across additional preloads, slip directions, and a second held-out group are planned.

## C. Object Manipulation Modeling

We demonstrate on-glove pose and tactile signals fusion yields higher quality manipulation trajectories than egocentric vision solution alone—both the object and hand—and therefore more suitable for robotic policy learning. For our setup: a participant manipulates objects spanning a range of size, geometry, and grasp affordance, each with a pre-scanned mesh. Three streams record concurrently: the glove’s IMU and tactile signals, an ego-CV RGBD camera. Streams are timealigned. For each object, the participant performs a natural fingertip grasp, 3D translation and rotation, and free-form manipulation. The trials deliberately induce hand–object and hand–hand occlusion as would happen in natural manipulation.

We compare against vision-only baselines—HaMeR [18] for hand pose and FoundationPose [48] for object pose qualitatively by overlaying the hand and object poses to the original video. Because contact constrains kinematics between the hand and object, we are able to achieve high-fidelity hand and object pose together, as in Fig. 8. The tracking quality can be greatly improved when applying Tactus pose and tactile information to the joint optimization.

## VII. CONCLUSION AND FUTURE WORK

Tactus provides calibrated fingertip force sensing and occlusion-resilient hand tracking in a textile-integrated form factor that preserves natural grasp behavior. Its contribution is a complete measurement system: sensor placement derived from empirical contact studies, a documented calibration pipeline with quantified drift, and standardized benchmarking protocols that make cross-platform comparison reproducible.

![](images/a3aee7f5a297f1e40a53472fa8b3433449af87d060c322d82cb147a3f75a9ca8.jpg)  
Fig. 8. Multi-modalities from Tactus (Tactile and IMU) enable qualitatively better hand-object interaction recording compared to CV-only HaMeR hand pose and RGBD CV-only FoundationPose object pose. $t _ { 1 } { : }$ FoundationPose has hard time to pin-point the cup handle orientation for such close-tosymmetric object. t & $t _ { 3 } { \mathrm { : } }$ HaMeR hand and FoundationPose object loses track completely due to occlusion. $t _ { 4 } { \mathrm { : } }$ although HaMeR recovers both hands, the hands scaling / depth are incorrectly estimated.

Applied to Tactus and to commercially available gloves, these protocols establish quantitative performance bars: 0.59 N minimum detectable force at 95% confidence, 11.0 mm rootaligned fingertip tracking error from offline trajectory optimization, at 106% of bare-hand task completion time. These capabilities position Tactus as a platform for the forcedependent natural manipulation tasks driving the next generation of robot learning. As importantly, the introduced benchmarking protocols are the transferable artifact: by specifying ground truth, metrics, and drift reporting, they offer the field a route from incomparable single-platform demonstrations toward reproducible, quantitative evaluation of sensing gloves.

Our major future work is large-scale user study spanning more participants, surfaces, slip detection, and glove platforms under Protocol-Tactile and Protocol-Pose — the protocols are specified precisely so that this extension is possible by others as well as by us. We would also like to reduce the persession calibration burden as a parallel thread. We will also systematically report Tactus’ performance degradation during the full life-cycle.

## REFERENCES

[1] F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data scaling laws in imitation learning for robotic manipulation,” in arXiv, 2024.

[2] B. Paliwal, H. Etukuru, W. Liang, P. Abbeel, N. M. M. Shafiullah, and J. Malik, “Do as I do: Dexterous manipulation data from everyday human videos,” in arXiv, 2026.

[3] J. Du, J. Ren, Q. Yu, N. Zhang, Y. Deng, X. Wei, Y. Liu, G. Gu, and X. Zhu, “MILE: A mechanically isomorphic hand exoskeleton and visuotactile robotic hand for data collection in dexterous manipulation,” in arXiv, 2026.

[4] M. Zhu, A. Zhu, J. V. S. H. Ramos, B. J. Kim, Y. Shi, Y. Wu, R. Hou, Q. Wang, E. Song, T. Fan, Y. Cui, and D. W. Hong, “WHED: A wearable hand exoskeleton for natural, high-quality demonstration collection,” in International Conference on Ubiquitous Robots (UR), 2026.

[5] A. Zhu, M. Zhu, B. J. Kim, J. V. S. H. Ramos, Y. Shi, Y. Wu, R. Dhar, F. Yang, R. Hou, H. Fang, Q. Wang, Y. Cui, and D. W. Hong, “Dexexo: A wearability-first dexterous exoskeleton for operator-agnostic demonstration and learning,” in arXiv, 2026.

[6] P. Grady, C. Tang, S. Brahmbhatt, C. D. Twigg, C. Wan, J. Hays, and C. C. Kemp, “Pressurevision: Estimating hand pressure from a single rgb image,” in European Conference on Computer Vision (ECCV), 2022.

[7] P. Grady, J. A. Collins, C. Tang, C. D. Twigg, K. Aneja, J. Hays, and C. C. Kemp, “Pressurevision++: Estimating fingertip pressure from diverse rgb images,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 8698–8708.

[8] Y. Zhao, T. Kwon, P. Streli, M. Pollefeys, and C. Holz, “Egopressure: A dataset for hand pressure and pose estimation in egocentric vision,” in CVPR, 2025.

[9] S. Brahmbhatt, C. Tang, C. D. Twigg, C. C. Kemp, and J. Hays, “Contactpose: A dataset of grasps with object contact and hand pose,” in ECCV, 2020.

[10] A. Ilic, R. Schuchert, Y. Jiang, and C. Holz, “Egophi: Estimating contact and force from egocentric vision,” in ECCV, 2026.

[11] S. Bansal, Z. Zhu, S. Tripathi, J. Zhao, M. J. Black, and D. Damen, “Towards in-the-wild egocentric 3d hand-object pose estimation,” in ECCV, 2026.

[12] Y. Zeng, Y. Shi, T. Tan, X. Li, Y. Qin, Z. Lu, W. Yang, J.-H. Xue, and Q. Liao, “Egotactile: Learning grasp pressure for everyday objects from egocentric video,” arXiv, 2026.

[13] S. Sundaram, P. Kellnhofer, Y. Li, J.-Y. Zhu, A. Torralba, and W. Matusik, “Learning the signatures of the human grasp using a scalable tactile glove,” Nature, 2019.

[14] J. Yin, H. Qi, Y. Wi, S. Kundu, M. Lambeta, W. Yang, C. Wang, T. Wu, J. Malik, and T. Hellebrekers, “OSMO: Open-source tactile glove for human-to-robot skill transfer,” in arXiv, 2025.

[15] Pressure Profile Systems, “Tactile glove,” https://pressureprofile.com/ body-pressure-mapping/tactile-glove, 2026.

[16] JQ Industries, “Precision tactile glove,” https://jq-industries.com/ precision/glove, 2026, accessed: 2026-08-13.

[17] A. Adeniji, Z. Chen, V. Liu, V. Pattabiraman, R. Bhirangi, S. Haldar, P. Abbeel, and L. Pinto, “Feel the force: Contact-driven learning from humans,” 2025.

[18] G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik, “Reconstructing hands in 3D with transformers,” in CVPR, 2024.

[19] S. Han, P. Wu, Y. Zhang, B. Liu, L. Zhang, Z. Wang, W. Si, P. Zhang, Y. Cai, T. Hodan, R. Cabezas, L. Tran, M. Akbay, T. Yu, C. Keskin, and R. Wang, “Umetrack: Unified multi-view end-to-end hand tracking for VR,” in SIGGRAPH Asia, 2022.

[20] Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges, “ARCTIC: A dataset for dexterous bimanual handobject manipulation,” in CVPR, 2023.

[21] P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol, R. Newcombe, R. Wang, J. J. Engel, and T. Hodan, “HOT3D: Hand and object tracking in 3D from egocentric multi-view videos,” in CVPR, 2025.

[22] Z. Fan, T. Ohkawa, L. Yang, N. Lin, Z. Zhou, S. Zhou, J. Liang, Z. Gao, X. Zhang, X. Zhang, F. Li, Z. Liu, F. Lu, K. Knaebel, B. Leibe, J. On, S. Baek, A. Prakash, S. Gupta, K. He, Y. Sato, O. Hilliges, H. J. Chang, and A. Yao, “Benchmarks and challenges in pose estimation for egocentric hand interactions with objects,” in ECCV, 2024.

[23] O. Taheri, N. Ghorbani, M. J. Black, and D. Tzionas, “GRAB: A dataset of whole-body human grasping of objects,” in ECCV, 2020.

[24] Y. Liang, S. Xu, Y. Zhang, B. Zhan, H. Zhang, and L. Liu, “DexterCap: An affordable and automated system for capturing dexterous hand-object manipulation,” arXiv, 2026.

[25] O. Glauser, S. Wu, D. Panozzo, O. Hilliges, and O. Sorkine-Hornung, “Interactive hand pose estimation using a stretch-sensing soft glove,” ACM Transactions on Graphics, 2019.

[26] G. Saggio, L. Pietrosanti, I.-J. Lee, and B.-S. Lin, “Quasi-static and dynamic measurement capabilities provided by an electromagnetic fieldbased sensory glove,” Biosensors, 2025.

[27] A. Tashakori et al., “Capturing complex hand movements and object interactions using machine learning-powered stretchable smart textile gloves,” Nature Machine Intelligence, 2024.

[28] Z. Yang, B.-J. F. van Beijnum, B. Li, S. Yan, and P. H. Veltink, “Estimation of relative hand-finger orientation using a small IMU configuration,” Sensors, 2020.

[29] J. Henderson, J. Condell, J. Connolly, D. Kelly, and K. Curran, “Reliability and validity of clinically accessible smart glove technologies to measure joint range of motion,” Sensors, 2021.

[30] X. Yi, Y. Zhou, and F. Xu, “TransPose: Real-time 3D human translation and pose estimation with six inertial sensors,” ACM Transactions on Graphics, 2021.

[31] A. Sarker, Z. Kou, E. Ristani, L. Guan, and T. Niehues, “Real-time hand pose tracking using 6-axis IMUs,” in IEEE HRI, 2026.

[32] Z. Kou, A. Kumar, M. Huang, T. Niehues, V. Mehta, E. Ristani, and L. Guan, “AVI-HT: Adaptive vision-IMU fusion for 3D hand tracking,” in arXiv, 2026.

[33] C. Jiang, W. Xu, Y. Li, Z. Yu, L. Wang, X. Hu, Z. Xie, Q. Liu, B. Yang, X. Wang, W. Du, T. Tang, D. Zheng, S. Yao, C. Lu, and J. Liu, “Capturing forceful interaction with deformable objects using a deep learning-powered stretchable tactile array,” Nature Communications, 2024.

[34] J. Park, P. Heo, J. Kim, and Y. Na, “A Finger Grip Force Sensor with an Open-Pad Structure for Glove-Type Assistive Devices,” Sensors, 2019.

[35] L. Zhang, G. Song, D. Yu, G. Zhou, X. Ma, and D. Chen, “A dual-stage adaptive normalization strategy to compensate resistance drift in flexible sensor smart gloves,” in ICHCI, 2025.

[36] Y. Luo, Y. Li, P. Sharma, W. Shou, K. Wu, M. Foshey, B. Li, T. Palacios, A. Torralba, and W. Matusik, “Learning human–environment interactions using conformal tactile textiles,” Nature Electronics, 2021.

[37] Y. Song, J. Li, R. Fu, D. Murphy, K. Zhou, R. Shiv, Y. Li, H. Xiong, C. Owens, Y. Du, Y. Luo, X. Cheng, A. Torralba, W. Matusik, and P. Liang, “OPENTOUCH: Bringing Full-Hand Touch to Real-World Interaction,” arXiv, 2025.

[38] J. Kangro, A. V. Sureshbabu, S. Traversaro, D. Pucci, and F. Nori, “A Plenum-Based Calibration Device for Tactile Sensor Arrays,” IEEE Robotics and Automation Letters, vol. 3, 2018.

[39] I. Sorrentino, F. J. Andrade Chavez, C. Latella, L. Fiorio, S. Traversaro, L. Rapetti, Y. Tirupachuri, N. Guedelha, M. Maggiali, S. Dussoni, G. Metta, and D. Pucci, “A Novel Sensorised Insole for Sensing Feet Pressure Distributions,” Sensors, 2020.

[40] “Calibrating And Verifying Tactile Pressure Sensors,” 2026. [Online]. Available: https://pressureprofile.com/calibration-white-paper

[41] C. K. Bensel, “The effects of various thicknesses of chemical protective gloves on manual dexterity,” Ergonomics, 1993.

[42] I.-G. Yoo, J. Lee, M.-Y. Jung, and J.-H. Lee, “Effects of wearing the wrong glove size on shoulder and forearm muscle activities during simulated assembly work,” Industrial Health, vol. 53, 2015.

[43] S. G. Hart, “NASA-Task Load Index (NASA-TLX); 20 years later,” in Proc. Human Factors and Ergonomics Society Annual Meeting, 2006.

[44] R. S. Johansson and G. Westling, “Roles of glabrous skin receptors and sensorimotor memory in automatic control of precision grip when lifting rougher or more slippery objects,” Experimental Brain Research, 1984.

[45] M. J. Adams, S. A. Johnson, P. Lefevre, V. L\` evesque, V. Hayward,´ T. Andre, and J.-L. Thonnard, “Finger pad friction and its role in grip´ and touch,” Journal of the Royal Society Interface, 2013.

[46] T. C. Yu, Z. Kou, M. Huang, T. Niehues, Y. Luo, L. Guan, and D. Zhang, “Pose-aware modeling to mitigate pose-related artifacts in tactile gloves,” in IROS, 2026.

[47] B. C¸ allı, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar, “Benchmarking in manipulation research: The YCB object and model set and benchmarking protocols,” in IEEE Robotics & Automation Magazine, 2015.

[48] B. Wen, W. Yang, J. Kautz, and S. Birchfield, “FoundationPose: Unified 6D pose estimation and tracking of novel objects,” in CVPR, 2024.