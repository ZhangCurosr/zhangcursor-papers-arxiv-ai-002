# Learning Tactile-Aware Hand Representations from Human Hand–Object Interaction Data

Yeonguk Yu, Jaehong Kim, and Byungok Han

Abstract— Tactile representation learning for dexterous manipulation often requires substantial robot-side data collection, limiting its scalability across different hands and sensors. We explore whether large-scale human hand-object interaction (HOI) data can provide a useful prior for robotic tactile learning. To this end, we propose HuTact, a framework that pretrains tactile-aware hand representations from human HOI data without requiring real tactile signals during pretraining. HuTact transfers the interaction structure learned from human hand motions to robotic tactile observations and can be further adapted with a small amount of unlabeled robot interaction data. We evaluate HuTact on tactile perception and manipulation tasks, including grasp prediction, slip detection, object classification, and tactile-conditioned policy learning. Across these tasks, human HOI pretraining improves data efficiency and provides a strong initialization for downstream learning. Notably, with HOI pretraining followed by midtraining on unlabeled robot play data, HuTact achieves 88.26% Macro-F1 on grasp prediction using only 25% of the labeled downstream data, comparable to training from scratch on the full labeled dataset (88.23%). These results suggest that human interaction data can serve as a scalable complementary source for robotic tactile representation learning.

## I. INTRODUCTION

Tactile sensing enables humans to perceive and interact with the physical world, providing direct information about contact, force, texture, and motion that is difficult to obtain through vision alone [1]. Inspired by this remarkable human capability, tactile sensing has emerged as a critical modality for robotic manipulation, enabling robots to perceive subtle contact interactions, infer local physical properties, and regulate contact forces during complex manipulation tasks that demand precise and adaptive physical interaction [2]–[4].

Accordingly, a wide range of tactile sensors have been developed to capture rich information arising from physical contact, including vision-based sensors (e.g., GelSight [2], DIGIT [5]) and magnetic-based sensors (e.g., AnySkin [4], ReSkin [6]). Beyond advances in tactile sensing hardware, the growing availability of large-scale tactile interaction data has further motivated learning-based approaches to extract generalizable representations from raw tactile observations [7]. In particular, recent works have explored selfsupervised pretraining using large-scale tactile interaction data collected with respective tactile sensors, learning representations that can be transferred to downstream manipulation tasks [8]–[11].

Learning generalizable tactile representations can reduce the amount of task-specific data required for downstream tasks. For vision-based tactile sensing, Feng et al. propose AnyTouch [12], which learns transferable representations from a sensor-aligned dataset collected across multiple visuo-tactile sensors. For non-vision-based tactile sensing, Sharma et al. [8] apply DINO-based self-distillation to masked tactile sequences collected from a dexterous hand, while Guzey et al. [9] pretrain tactile representations using task-agnostic robotic play data. However, existing approaches for non-vision-based tactile sensors still largely require real tactile interaction data collected directly with the target robotic hand and sensing hardware. As a result, adapting representation learning to a new robotic platform or tactile sensor requires additional data collection, introducing substantial time and cost for pretraining.

As an alternative, human hand-object interaction (HOI) data provide a promising source of pretraining supervision, where they contain large-scale and diverse interactions across objects, hand configurations, and manipulation behaviors [13]–[16]. These datasets are particularly attractive for learning transferable hand representations because they provide (1) interaction patterns that can be shared across different hand embodiments, (2) large-scale interaction data that would be costly to collect on robotic platforms, and (3) an opportunity to reduce embodiment-specific data collection when adapting to a new robotic hand. Many of these datasets also provide temporally aligned 3D reconstructions of human hands and manipulated objects, allowing fingertip configurations and their evolution during interaction to be recovered without collecting data on a robotic platform. Although HOI datasets do not contain real tactile signals, their hand–object trajectories encode spatial and temporal structure associated with physical interaction. We hypothesize that this structure can serve as a transferable prior for robotic tactile representation learning and reduce the amount of robot-side tactile data required for adaptation.

In this work, our research question is: Can human handobject interaction (HOI) data provide a transferable prior for robotic tactile representation learning? To answer this question, we propose HuTact, a framework for learning tactile-aware robot hand representations from human HOI data. Rather than attempting to reconstruct real tactile signals from human data, HuTact learns the spatial and temporal structure of hand interactions from wrist-local fingertip trajectories. To bridge human HOI pretraining and robotic tactile sensing, we separately tokenize fingertip coordinates and tactile observations, allowing the coordinate representation and shared Transformer backbone learned from HOI data to be retained while adapting the sensor-specific tactile input embedding. We further incorporate geometry-derived pseudo-contact signals as an auxiliary cue, encouraging the model to capture the relationship between fingertip coordinates and their corresponding tactile-like signals. This framework enables large-scale human interaction data to provide a transferable initialization for tactile-aware robot learning while reducing the amount of robot-side tactile data required for adaptation.

To validate the effectiveness and transferability of HuTact, we evaluate it on tactile-sensitive downstream tasks, including grasp prediction, slip detection, object classification, and policy learning on tactile-sensitive tasks, and analyze the factors that enable transfer from human HOI data to robotic tactile sensing. In summary, our contributions are as follows:

• We propose HuTact, a framework that uses large-scale human hand-object interaction data to pretrain tactileaware robot hand representations without requiring real tactile signals during the initial pretraining stage.

• We show that representations learned from HOI data transfer to robotic tactile perception and manipulation, improving grasp prediction, slip detection, object classification, and tactile-conditioned policy learning.

• We provide systematic analyses of the transfer mechanism through architectural ablations and data-efficiency experiments, studying the roles of temporal modeling, token separation, pseudo-contact cues, and robot-side midtraining. Our results show that HOI pretraining can substantially reduce the amount of task-specific robotic tactile data required for downstream adaptation.

## II. RELATED WORK

Tactile Sensing: Tactile sensors for robotic manipulation broadly include vision-based and non-optical sensing modalities. Vision-based sensors [2], [5] capture fine-grained contact geometry and have been applied to object identification [17], geometry sensing [3], [18], pose estimation [19], [20], and policy learning [21], [22]. However, their optical components can result in a relatively large form factor, making integration across multiple fingers challenging. Nonoptical sensors instead use compact magnetic [4], [23] or capacitive sensing elements [24] to enable distributed tactile sensing. In this work, we use the BrainCo Revo 2 hand [25], which is equipped with fingertip capacitive sensors that measure normal force, tangential force, tangential-force direction, and proximity.

Tactile Representation Learning: Recent work has explored transferable tactile representations through large-scale self-supervised pretraining. For vision-based tactile sensors, objectives developed for visual representation learning [26]– [28] have been adapted to tactile observations [10]–[12], [29]. For skin-like tactile sensors, Sharma et al. [8] apply self-distillation to masked tactile sequences from a dexterous hand, while Guzey et al. [9] pretrain tactile representations using task-agnostic robotic play data. While these approaches demonstrate the benefits of tactile pretraining, they generally require interaction data collected with the target robot and tactile sensor. In contrast, HuTact initializes tactileaware representations from human hand–object interaction data without real tactile signals. We pretrain on the spatial and temporal structure of wrist-local fingertip interactions, augmented with geometry-derived pseudo-contact cues, and subsequently adapt the representation to the target robotic tactile sensor.

Robot Learning from Human Data: Human interaction data have been increasingly used to reduce robot-specific supervision for manipulation learning [30]–[34], by extracting hand poses, object motions, and interaction trajectories from large-scale demonstrations. HOI data have also been applied directly to dexterous manipulation: Zhao et al. [35] retarget human hand-object interactions to dexterous robot hands and learn a residual policy for embodiment mismatch, while Zeng et al. [36] use HOI data for vision-languageaction learning. Large-scale datasets such as OakInk2 [16], TACO [37], ARCTIC [38], and HOT3D [39] provide scalable supervision for learning hand–object interaction structure. HuTact differs in objective: rather than transferring human motion directly to a robot policy, we use HOI data to pretrain a tactile-aware hand representation that is subsequently adapted to real robotic tactile sensing. To our knowledge, prior work has not systematically studied large-scale HOI data as a pretraining source for robotic tactile representation learning.

## III. METHODOLOGY

HuTact is designed to transfer interaction structure learned from human HOI data to robotic tactile sensing. The key challenge is that the source HOI datasets provide hand and object geometry but no real tactile signals, whereas the target robot provides sensor-specific tactile observations. We address this mismatch by factorizing the input representation into two streams: a shared wrist-local fingertip coordinate stream and a sensor-dependent contact stream. The shared Transformer backbone is pretrained on human interactions, while the sensor-specific tactile input embedding is adapted when transferring to real robotic tactile observations.

## A. Problem setup

We consider two dexterous robot hands (BrainCo Revo2 Hand) with N=10 fingertip tactile sensors. At each timestep t, sensor i captures a signal $\mathbf { s } _ { i } ^ { t } \in \mathbb { R } ^ { C }$ with C=4 channels—normal force, tangential force, tangential direction, and proximity—and forward kinematics provides the corresponding fingertip coordinate $\mathbf { c } _ { i } ^ { t } \in \mathbb { R } ^ { 3 }$ in the wrist-local frame as in [33]. The model operates on a temporal window

$$
\mathbf { X } = \{ ( \mathbf { c } _ { i } ^ { t } , \mathbf { s } _ { i } ^ { t } ) \} _ { i = 1 , \ldots , N } ^ { t = 1 , \ldots , T } ,\tag{1}
$$

with T=15 frames sampled at 30 Hz, spanning 0.5 s of contact. Then, we pretrain a tactile encoder $f _ { \theta }$ with humanobject interaction (HOI) data and subsequently transfer it to robotic downstream tasks. To maintain a consistent input structure across the human and robot domains, human hand samples are represented using fingertip coordinates paired with a one-dimensional geometry-derived pseudocontact signal (i.e., C = 1), whereas robot observations use the corresponding fingertip coordinates paired with fourdimensional tactile signals.

![](images/9ddb43185ecfd3a50e72b9c663cc9ca105013fedac392518c08b0358d566cfc8.jpg)  
Fig. 1. Overview of HuTact. Stage 1 pretrains the encoder on human HOI data using fingertip coordinates and geometry-derived pseudo-contact signals. Stage 2 optionally adapts the representation to real robotic tactile signals using unlabeled play data, and Stage 3 fine-tunes the encoder for downstream robotic tasks.

## B. HOI Data Processing

The pretraining corpus is assembled from HOI dataset. We obtain its 3D coordinate from the MANO hand model [40] and extract a one-dimensional pseudo-contact signal from the geometric relationship between the hand and object meshes. First, we convert the MANO hand representation [40] into a common 21-joint skeleton by augmenting the 16 MANO kinematic joints with five fingertip vertices and reordering them into a unified joint convention. From this representation, we extract wrist-local fingertip coordinates and pseudocontact signals for HOI pretraining.

Fingertip coordinates. For each hand and frame, we construct a wrist-local coordinate frame from the hand geometry. The wrist defines the origin, while the palm-normal and finger directions are derived from the pinky–thumb and wrist–middle-finger axes and orthogonalized to form a righthanded basis. We transform all joints into this frame and retain the five fingertips per hand, yielding N=10 coordinates $\mathbf { c } _ { i } ^ { t } \in \mathbb { R } ^ { 3 }$ . During the experiment, robot fingertip coordinates from forward kinematics use the same local convention.

Pseudo-contact signals. We define a surface patch $\mathcal { P } _ { i } ^ { t }$ over each fingertip pad rather than estimating contact from a single fingertip point. MANO vertices are assigned to joints using dominant skinning weights, and each fingertip is augmented with the 20 nearest rest-pose vertices around its fingertip vertex. At each frame, we retain only vertices on the palmar side using the local pad direction, preventing dorsalside vertices from producing spurious contact responses.

For each retained vertex $\boldsymbol { \nu } \in \mathcal { P } _ { i } ^ { t }$ , we compute the shortest point-to-triangle distance $d _ { \nu } ^ { t }$ to the object surface. Given a contact threshold of $d _ { \operatorname* { m a x } } = 1 0$ mm, we define

$$
u _ { \nu } ^ { t } = \frac { \displaystyle \operatorname* { m a x } \bigl ( d _ { \mathrm { m a x } } - d _ { \nu } ^ { t } , 0 \bigr ) } { d _ { \mathrm { m a x } } } , \qquad r _ { \nu } ^ { t } = ( u _ { \nu } ^ { t } ) ^ { 2 } ( 3 - 2 u _ { \nu } ^ { t } ) ,\tag{2}
$$

where $r _ { \nu } ^ { t } \in [ 0 , 1 ]$ provides a smooth proximity response.

We then aggregate the vertex responses using their associated surface areas:

$$
s _ { i } ^ { t } = \frac { \sum _ { \nu \in \mathcal { P } _ { i } ^ { t } } a _ { \nu } ^ { t } r _ { \nu } ^ { t } } { \sum _ { \nu \in \mathcal { P } _ { i } ^ { t } } a _ { \nu } ^ { t } } ,\tag{3}
$$

where $a _ { \nu } ^ { t }$ is the representative surface area of vertex v, computed from its incident triangles. This area-weighted formulation reduces sensitivity to mesh tessellation and yields a bounded proximity-like signal $s _ { i } ^ { t } \in [ 0 , 1 ]$ , which increases as a larger portion of the fingertip pad approaches the object surface. When multiple objects are present, we select the object yielding the largest $s _ { i } ^ { t }$ for each fingertip. For HOI pretraining, we use $s _ { i } ^ { t }$ as a one-dimensional contact observation (C = 1).

## C. Architecture

As shown in Fig. 2, HuTact consists of separate coordinate and tactile tokenizers, a frame encoder, and a temporal encoder.

Tokenizers. Within a frame t, fingertip coordinates $c ^ { t }$ and tactile signals $s ^ { t }$ are tokenized as two separate streams: $\{ \mathbf { c } _ { i } ^ { t } \}$ and $\{ \mathbf { s } _ { i } ^ { t } \}$ are projected by distinct input embeddings into N coordinate tokens and $N$ tactile tokens, each stream normalized with its own statistics. Specifically, the tokenizing process can be formalized as follows:

![](images/fed2b8d13ce2e92733484e7554dbb9d96795bd2fcb4fb0d2e6401204b19b596d.jpg)  
Fig. 2. Illustration of backbone architecture. Each frame is encoded from separate tactile and wrist-local fingertip coordinate token streams. The resulting frame-level CLS tokens are then aggregated across 15 frames by a temporal Transformer to produce a window-level representation for downstream tasks.

$$
z _ { c , i } ^ { t } = \mathrm { L N } _ { c } \left( \mathbf { W } _ { c } \mathbf { c } _ { i } ^ { t } + b _ { c } \right) , \qquad z _ { s , i } ^ { t } = \mathrm { L N } _ { s } \left( \mathbf { W } _ { s } \mathbf { s } _ { i } ^ { t } + b _ { s } \right) ,\tag{4}
$$

where the $z _ { c , i } ^ { t } \in \mathbb { R } ^ { D }$ and $z _ { s , i } ^ { t } \in \mathbb { R } ^ { D }$ denote the coordinate tokens and tactile tokens for the i-th finger, respectively. Also, LN, $\mathbf { W } \in \mathbb { R } ^ { D \times m }$ (m is the dimension of input), and $b \in \mathbb { R } ^ { D }$ refer to the Layer Norm [41], linear weight, and bias, respectively. Finally, we concatenate these tokens with [CLS] token as:

$$
\mathbf { Z } ^ { t } = \left[ \mathbf { z } _ { \mathrm { C L S } } ^ { t } , \{ \mathbf { z } _ { c , i } ^ { t } \} _ { i = 1 } ^ { N } , \{ \mathbf { z } _ { s , i } ^ { t } \} _ { i = 1 } ^ { N } \right] .\tag{5}
$$

As shown in this Eq 4 and Eq 5, we separate the token input for the tactile signals and fingertip coordinates unlike previous approaches [8]. This token separation is particularly important for transferring from HOI pretraining to real tactile sensing: while the contact signal used during pretraining and the real tactile signal have different dimensionalities (i.e., 1 dim vs. 4 dim), the coordinate tokenizer remains unchanged and can be directly reused after pretraining.

Frame encoder. Given the concatenated token sequence Z<sup>t</sup> at each timestep $t \in \{ 1 , 2 , . . . , T \}$ , we apply 2D multimodal RoPE [42] using hand and finger identities as the two positional dimensions. We then process the tokens with a lightweight 4-layer Transformer encoder, following the remaining architectural hyperparameters of ViT-Tiny architecture [43]. Through self-attention, the encoder captures interactions between tactile observations and fingertip coordinates, producing a unified representation of the hand-object contact state at each timestep. The frame-level representations are then passed to the temporal encoder to model their relationship over time. Specifically, we take the [CLS] token from each frame encoder output and concatenate them along the token dimension to construct a sequence of frame-level representations. Given $T = 1 5$ input frames, we obtain

$$
\mathbf { H } = \left[ \mathbf { h } _ { \mathrm { C L S } } , \hat { \mathbf { z } } _ { \mathrm { C L S } } ^ { 1 } , \hat { \mathbf { z } } _ { \mathrm { C L S } } ^ { 2 } , \dots , \hat { \mathbf { z } } _ { \mathrm { C L S } } ^ { 1 5 } \right] ,\tag{6}
$$

where $\mathbf { h } _ { \mathrm { C L S } }$ and $\hat { \mathbf { z } } _ { \mathrm { C L S } } ^ { i }$ are new CLS token for temporal encoder and output CLS token for i-th frame with frame encoder, respectively.

Temporal encoder. Given the frame-level sequence H, the temporal encoder applies RoPE based on the absolute frame index and models temporal dependencies across the $T = 1 5$ frame-level [CLS] tokens. It uses the same Transformer architecture as the frame encoder with independent parameters. The output temporal [CLS] token $\hat { \mathbf { h } } _ { \mathrm { C L S } }$ is used as the final window-level representation. This representation summarizes the spatiotemporal hand-object interaction over the entire $T = 1 5$ frame window and is subsequently used by the downstream prediction head.

## D. Training objectives and stages

Objective. We pretrain the model using temporal selfdistillation following the DINOv2 [27] as [8] in the pretraining and midtraining stages. A student network is trained to match an exponential-moving-average (EMA) teacher by combining a distillation loss on the temporal [CLS] embedding with a masked-token prediction loss on the frame-level tokens.

Stage 1: Pretraining on HOI data. In the first stage, the encoder is pretrained for 200 epochs using the objective described above on the HOI data. The data consists of contact signals and fingertip coordinates derived from public HOI data. Note that, no robot data are used at any point during this stage.

Stage 2 (Optional): Midtraining on robotic play data. In the second stage, we optionally continue self-supervised training using the same objective and optimization recipe on a small unlabeled robotic play data. Because the real robot provides $C = 4$ tactile sensor channels, whereas Stage 1 uses a different pseudo-tactile channel configuration, we reinitialize the tactile input tokenizer at the transition between the two stages while retaining all remaining pretrained parameters. Each epoch repeats the robot corpus 32 times, preserving approximately the same number of training samples per epoch as in Stage 1 (≈ 505k samples per epoch). We train for 100 epochs using the same warmup-cosine learning-rate schedule. Hand poses are min-max normalized to the unit box during this stage.

Stage 3: Downstream transfer. For downstream transfer, we fine-tune the full encoder jointly with an attentive-pooling task head. When Stage 2 midtraining is not conducted, we reinitialize the tactile input tokenizer before downstream finetuning; otherwise, the tokenizer adapted during Stage 2 is used. For the given downstream data, we use AdamW with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ for 100 epochs.

Optimization. We use AdamW with a peak learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 512. The learning rate is linearly warmed up for the first 10 epochs and subsequently decayed using a cosine schedule to $1 \times 1 0 ^ { - 6 }$ . The teacher temperature is increased from 0.04 to $0 . 0 7$ during the first 2 epochs. The EMA momentum of the teacher is increased from 0.994 to 1.0 throughout training, while the weight decay follows a cosine schedule ending at 0.04.

![](images/53d7ea22f284854fc715d1a915ad97fcdbde1d75a9a43f5cbd631be0d275bb39.jpg)  
Fig. 3. Hardware setup and downstream tasks. We use a Unitree G1 humanoid equipped with a BrainCo Revo2 dexterous hand. Downstream tasks includes grasp prediction, slip detection, object classification, cup destacking, and single-card sliding.

## IV. EXPERIMENTS

## A. Experimental setup

We use a Unitree G1 humanoid equipped with a BrainCo Revo2 dexterous hand [25], which has capacitive tactile sensors at each fingertip, as shown in Fig. 3.

Pretraining datasets. For HOI pretraining, we use HOT3D [39], OakInk-v2 [16], TACO [37], and ARC-TIC [38], extracting fingertip coordinates and pseudo-contact signals from 2.57M frames (∼24 hours). We additionally collect 80K frames (∼44 minutes) of Revo2 play data containing fingertip coordinates and tactile signals through Meta Quest 3 teleoperation [44] with six objects (e.g., plastic bottle, bubble wrap, driver, pot, sponge, and towel).

Downstream datasets. For grasp prediction, we collect 590 teleoperated grasp episodes over five objects, totaling 18K frames (∼10 minutes), with each episode labeled by grasp success. For slip detection, we hold the robot hand fixed and manually move the grasped object while tracking an attached marker; marker motion is labeled as slip, yielding 18 episodes over six object conditions and 51K frames (∼28 minutes). For object classification, we reuse successful grasp episodes, extract the result-phase frames, and label the grasped object identity, yielding 252 episodes and 43K frames (∼24 minutes).

For ID evaluation, we perform episode-level crossvalidation before constructing temporal windows, ensuring that all windows from the same episode remain in the same split. We use stratified 4-fold cross-validation for grasp prediction and object classification, with 75% of the episodes for training and 25% for evaluation in each fold, and 3-fold cross-validation for slip detection. Reported ID performance is averaged across the corresponding folds.

For policy learning, we collect two datasets using a scripted controller. The cup-grasping and card-sliding datasets contain 40 and 55 episodes, respectively, each totaling approximately 30K frames (∼17 minutes). For each policy configuration, success rates are computed over 40 evaluation trials per task.

## B. Performance on downstream tasks

In Table I, we evaluate HuTact on three downstream perception tasks: grasp prediction, slip detection, and object classification. Grasp prediction and slip detection are binary classification tasks that predict grasp success from observations collected 1 second after grasp execution and whether the current grasp is slipping or stable, respectively. Object classification is a five-way classification task that predicts the identity of the grasped object from observations collected during the post-grasp phase. Also, in the ID setting, all object categories are included in both the training and evaluation sets. In the OOD setting, we use a leave-one-object-out protocol, where one object is excluded from training and used only for evaluation, while the remaining objects are used exclusively for training (except object classification). We compare HuTact against two baselines: (i) From Scratch, where the model is randomly initialized, and (ii) Revo2 Pretraining, where the encoder is pretrained on Revo2 play data using the same self-supervised objective as Stage 1.

HOI pretraining (Stage 1 only) consistently improves grasp prediction over both training from scratch and Revo2- only pretraining, with particularly clear gains in the OOD setting. For slip detection, HOI pretraining provides smaller but consistent gains in ID Avg and OOD performance, while achieving comparable ID Last performance to training from scratch. Revo2-only pretraining performs worse on slip detection, indicating that a small amount of robot play data alone is insufficient to learn transferable features for this task. For object classification, however, Revo2 pretraining outperforms HOI pretraining, suggesting that real tactile observations contain additional object-specific information. Across all three tasks, HOI pretraining followed by Revo2 midtraining achieves the best overall performance, demonstrating that large-scale HOI data and robot-side tactile data provide complementary benefits for downstream tactile representation learning.

## C. Performance with action policy

In Table II, we evaluate the learned tactile representation on two contact-sensitive manipulation tasks using ACT [45] and Diffusion Policy (DP) [46]. Cup Destacking requires the robot to separate a cup from a stack while maintaining a stable grasp, whereas Single-Card Sliding requires the robot to slide a single card without displacing the remaining cards. We compare three tactile configurations: (i) no tactile input, (ii) a tactile encoder trained from scratch, and (iii) our HOI-pretrained encoder followed by Revo2 midtraining. We modify ResNet18-based ACT and DP with our tactile encoder. For ACT, the tactile representation is linearly projected to the Transformer embedding dimension and appended as a single token to the encoder input sequence. The tactile representation is used only as a conditioning input to the policy and is not provided to the VAE encoder. For Diffusion Policy, we concatenate the tactile representation directly with the visual observation features in the global conditioning vector.

TABLE I  
COMPARISON OF PRETRAINING STRATEGIES ON DOWNSTREAM PERCEPTION TASKS. MACRO-F1 SCORES ARE MEAN ± STANDARD DEVIATION OVER 9 RUNS. “LAST” DENOTES THE FINAL EPOCH AND “AVG” THE MEAN OVER EVALUATIONS EVERY 10 EPOCHS.
<table><tr><td>Task</td><td>Method</td><td>ID Last</td><td> $\mathbf { I D \ A v \mathbf { g } }$ </td><td> $\mathbf { O O D \ L a s t }$ </td><td> $\mathbf { O O D \ A v g }$ </td></tr><tr><td rowspan="4">Grasp Prediction</td><td>From Scratch</td><td> $8 8 . 2 3 \pm 0 . 7 8$ </td><td> $8 6 . 3 9 \pm 0 . 3 8$ </td><td> $7 4 . 9 2 \pm 9 . 7 8$ </td><td> $7 4 . 1 2 \pm 9 . 7 8$ </td></tr><tr><td>Revo2 Pretraining</td><td> $9 2 . 1 5 \pm 0 . 4 0$ </td><td> $9 2 . 0 9 \pm 0 . 3 5$ </td><td> $7 7 . 8 4 \pm 9 . 3 1$ </td><td> $7 7 . 5 1 \pm 9 . 1 5$ </td></tr><tr><td>HOI Pretraining (Ours)</td><td> $9 2 . 6 1 \pm 0 . 7 4$ </td><td> $9 2 . 1 8 \pm 0 . 7 5$ </td><td> $7 9 . 2 1 \pm 1 0 . 0 6$ </td><td> $7 8 . 4 6 \pm 1 0 . 0 5$ </td></tr><tr><td>HOI + Revo2 Midtraining (Ours)</td><td> $\overline { { { \bf 9 3 . 2 9 } } } \pm { \bf 0 . 5 3 }$ </td><td> ${ \bf 9 3 . 2 4 \ : \pm 0 . 5 4 }$ </td><td> ${ \bf 8 0 . 2 7 \pm 9 . 5 2 }$ </td><td> ${ \bf 8 0 . 0 2 \pm 9 . 4 9 }$ </td></tr><tr><td rowspan="4">Slip Detection</td><td>From Scratch</td><td> $7 5 . 6 8 \pm 1 . 7 3$ </td><td> $7 3 . 1 4 \pm 0 . 7 3$ </td><td> $6 9 . 2 8 \pm 4 . 5 5$ </td><td> $6 7 . 3 9 \pm 3 . 2 4$ </td></tr><tr><td>Revo2 Pretraining</td><td> $\overline { { 7 2 . 3 7 } } \pm 1 . 6 9$ </td><td> $7 1 . 5 5 \pm 1 . 9 6$ </td><td> $6 5 . 7 5 \pm 4 . 3 8$ </td><td> $6 5 . 8 0 \pm 3 . 8 7$ </td></tr><tr><td>HOI Pretraining (Ours)</td><td> $7 5 . 2 8 \pm 1 . 0 0$ </td><td> $7 3 . 7 8 \pm 0 . 8 0$ </td><td> $6 9 . 5 6 \pm 4 . 3 8$ </td><td> $6 8 . 8 6 \pm 3 . 1 4$ </td></tr><tr><td>HOI + Revo2 Midtraining (Ours)</td><td> ${ \bf 7 6 . 8 8 \pm 1 . 0 3 }$ </td><td> ${ \bf 7 6 . 4 1 \pm 0 . 8 5 }$ </td><td> $\overline { { { \bf 6 9 . 7 9 } } } \pm 4 . 2 4$ </td><td> $\overline { { { \bf 6 9 . 8 3 } } } \pm 3 . 7 5$ </td></tr><tr><td rowspan="4">Object Classification</td><td>From Scratch</td><td> $8 5 . 1 1 \pm 1 . 3 5$ </td><td> $8 2 . 9 6 \pm 0 . 7 1$ </td><td>N/A</td><td>N/A</td></tr><tr><td>Revo2 Pretraining</td><td> $\underline { { 8 9 . 0 7 } } \pm 1 . 3 1 $ </td><td> $\underline { { 8 8 . 8 5 } } \pm 1 . 2 3 $ </td><td>N/A</td><td>N/A</td></tr><tr><td>HOI Pretraining  $( \mathrm { O u r s } )$ </td><td> $\overline { { 8 7 . 6 4 } } \pm 0 . 8 2$ </td><td> $8 7 . 3 1 \pm 1 . 0 8$ </td><td>N/A</td><td>N/A</td></tr><tr><td>HOI + Revo2 Midtraining (Ours)</td><td> ${ \bf 8 9 . 8 9 \pm 1 . 5 2 }$ </td><td> ${ \bf 8 9 . 7 0 \pm 1 . 4 8 }$ </td><td>N/A</td><td>N/A</td></tr></table>

TABLE II

SUCCESS RATES (%) ON TACTILE-SENSITIVE MANIPULATION TASKS.
<table><tr><td>Model</td><td>Tactile Encoder</td><td>Cup Destacking</td><td>Single-Card Sliding</td><td>Avg.</td></tr><tr><td rowspan="3">ACT</td><td>None</td><td>30.0</td><td>37.5</td><td>33.8</td></tr><tr><td>From Scratch</td><td>5.0</td><td>60.0</td><td>32.5</td></tr><tr><td>HOI + Revo2 Midtraining (Ours)</td><td>37.5</td><td>62.5</td><td>50.0</td></tr><tr><td rowspan="3">Diffusion Policy</td><td>None</td><td>27.5</td><td>17.5</td><td>22.5</td></tr><tr><td>From Scratch</td><td>37.5</td><td>40.0</td><td>38.8</td></tr><tr><td> $\mathrm { H O I } + \mathrm { R e v o } 2$  Midtraining (Ours)</td><td>47.5</td><td>52.5</td><td>50.0</td></tr></table>

TABLE III  
CROSS-SENSOR TRANSFER PERFORMANCE ON DOWNSTREAM TASKS.
<table><tr><td>Encoder Architecture</td><td>Pretraining Source</td><td>Grasp Slip Pred.</td><td>Det.</td><td>Object Cls.</td></tr><tr><td>Sparsh-Skin [8] Allegro + Xela uSkin</td><td></td><td>87.8</td><td>70.6</td><td>81.0</td></tr><tr><td>HuTact (Ours)</td><td>HOI</td><td>92.2</td><td>73.8</td><td>87.3</td></tr></table>

Our tactile representation achieves the best overall performance for both policy architectures. With ACT, HOI + Revo2 midtraining improves the average success rate to 50.0%, compared with 33.8% without tactile input and 32.5% with a tactile encoder trained from scratch. With Diffusion Policy, the improvement is more consistent across both tasks: our method achieves 47.5% on Cup Destacking and 52.5% on Single-Card Sliding, yielding an average success rate of 50.0%, compared with 22.5% without tactile input and 38.8% from scratch. We observe that ACT with a tactile encoder trained from scratch often grasps multiple cups during Cup Destacking, which is counted as a failure and contributes to its low success rate of 5.0%. These results indicate that the pretrained tactile representation provides useful contactsensitive features for downstream policy learning beyond either omitting tactile input or learning the tactile encoder solely from task-specific demonstrations.

## D. Transfer performance comparison

Table III compares HuTact with Sparsh-Skin [8] to evaluate cross-sensor transfer across different pretraining sources and encoder architectures. We initialize Sparsh-Skin from the authors’ publicly released checkpoint, pretrained on an Allegro hand equipped with Xela uSkin sensors, and finetune it on the Revo2 downstream tasks. Because the two platforms differ in the number of sensors and tactile signal dimensions, we reinitialize the patch and positional embeddings while retaining the remaining pretrained parameters. HuTact consistently outperforms Sparsh-Skin across all three downstream tasks. This performance gap is consistent with the benefits of token separation for transfer across sensing configurations and temporal modeling for capturing contact dynamics over time. These results indicate that HuTact provides an effective initialization for robotic tactile representation learning.

TABLE IV  
ABLATION STUDIES ON IN-DISTRIBUTION DOWNSTREAM TASKS. MACRO-F1 (%) IS AVERAGED EVERY 10 EPOCHS FROM 10 TO 100.
<table><tr><td colspan="6">(a) Architecture ablation</td></tr><tr><td>Temp. Encoder</td><td>Token Sep.</td><td>HOI Pretrain.</td><td>Grasp Slip Object Pred.</td><td>Det.</td><td>Cls.</td></tr><tr><td>A</td><td></td><td>1</td><td>84.8</td><td>60.8</td><td>81.8</td></tr><tr><td>B</td><td></td><td>√</td><td>87.7</td><td>60.0</td><td>83.7</td></tr><tr><td>C</td><td></td><td>√ √</td><td>84.3</td><td>61.8</td><td>83.7</td></tr><tr><td>D</td><td></td><td>√ √</td><td>91.0</td><td>60.1</td><td>87.9</td></tr><tr><td>E</td><td>√</td><td>1</td><td>86.4</td><td>73.1</td><td>83.0</td></tr><tr><td>F</td><td>√</td><td>√</td><td>92.2</td><td>73.8</td><td>87.3</td></tr><tr><td colspan="6">(b) Pseudo-contact ablation</td></tr><tr><td></td><td>Pseudo-Contact Signals</td><td>Revo2 Midtrain.</td><td>Pred.</td><td>Grasp Slip Det.</td><td>Object Cls.</td></tr><tr><td>F′</td><td>一</td><td></td><td>92.0</td><td>72.6</td><td>87.3</td></tr><tr><td>F</td><td>√</td><td>一</td><td>92.2</td><td>73.8</td><td>87.3</td></tr><tr><td>G′</td><td>一</td><td>√</td><td>92.9</td><td>73.0</td><td>89.5</td></tr><tr><td>G</td><td>√</td><td>√</td><td>93.2</td><td>76.4</td><td>89.7</td></tr></table>

## E. Ablation Studies

In Table IV-(a), we analyze the effects of the temporal encoder, token separation, and HOI pretraining on the three downstream tasks. Configurations A and B use a nontemporal, unified-token architecture similar to prior tactile representation learning approaches [8], serving as a reference for our architectural modifications. (i) The temporal encoder provides the largest gain for slip detection: without HOI pretraining, adding temporal modeling improves Macro-F1 from 61.8% to 73.1% (C→E), highlighting the importance of modeling contact dynamics over time. (ii) Token separation is particularly effective when combined with HOI pretraining, improving grasp prediction from 87.7% to 91.0% and object classification from 83.7% to 87.9% (B→D). Separating fingertip coordinates from sensor-specific tactile inputs also allows the coordinate tokenizer learned from HOI data to be retained during transfer to the robot domain, and HOI pretraining yields larger gains when separate tokenizers are used. (iii) The complete configuration (F) achieves the best grasp prediction and slip detection performance among the architecture variants.

Table IV-(b) examines the contribution of pseudo-contact signals during Stage 1 pretraining. (i) Pseudo-contact pretraining provides its clearest benefit on slip detection, a task that relies strongly on tactile contact information, improving Macro-F1 from 72.6% to 73.8% without Revo2 midtraining $( \mathrm { F } ^ { \prime }  \mathrm { F } )$ , while the gains in grasp prediction and object classification are negligible. (ii) This benefit becomes more evident after Revo2 midtraining, where pseudo-contact improves slip detection from 73.0% to 76.4% (G<sup>′</sup> →G). These results suggest that pseudo-contact pretraining helps the model capture the relationship between fingertip-coordinate and tactile tokens, providing a more transferable initialization when adapting to real robotic tactile signals.

![](images/d556f7806113401e91b00e32c46abff6dee3c4dea555a5f23bd884000e32249b.jpg)  
Fig. 4. Downstream data efficiency on (a) grasp prediction, (b) slip detection, and (c) object classification. Macro-F1 (%) is reported as the downstream training data varies from 5% to 100%. Error bars denote standard deviation.

## F. Downstream Data Efficiency

We further evaluate the data efficiency of HuTact by varying the amount of downstream training data from 5% to 100%. As shown in Fig. 4, HOI + Revo2 midtraining consistently reduces the amount of task-specific data required across all three tasks. It matches or exceeds the full-data scratch baseline using only 25% of the data for grasp prediction (88.3% vs. 88.2%), 75% for slip detection (75.9% vs. 75.7%), and 50% for object classification (86.2% vs. 85.1%). These results demonstrate that HuTact substantially improves downstream data efficiency.

## V. LIMITATIONS

Our work has two main limitations. First, geometryderived pseudo-contact signals do not capture physical contact properties and provide only modest additional gains beyond fingertip coordinates for grasp prediction and object classification. They also require reconstructed hand and object meshes, limiting applicability to HOI datasets with 3D geometry. Second, our evaluation is limited to a Revo2- based bimanual setup, leaving transfer to other robotic hands and tactile sensing modalities unexplored. Future work will investigate richer contact cues from visual and hand-motion observations and extend HuTact to broader robotic embodiments and sensor layouts.

## VI. CONCLUSION

In this work, we presented HuTact, a framework for pretraining tactile-aware robot hand representations from large-scale human hand-object interaction (HOI) data without requiring real tactile signals during initial pretraining. HuTact leverages fingertip coordinates and pseudo-contact signals, with separate coordinate and tactile tokenization to support transfer to robotic tactile sensing. Across perception and policy-learning tasks, HOI pretraining provides a useful initialization that is further improved by Revo2 midtraining, while also substantially improving downstream data efficiency. These results show that human interaction data can serve as an effective pretraining prior for robotic tactile learning and reduce the need for robot-side tactile data collection.

[1] R. S. Dahiya, G. Metta, M. Valle, and G. Sandini, “Tactile sensing—from humans to humanoids,” IEEE transactions on robotics, vol. 26, no. 1, pp. 1–20, 2009.

[2] W. Yuan, S. Dong, and E. H. Adelson, “Gelsight: High-resolution robot tactile sensors for estimating geometry and force,” Sensors, vol. 17, no. 12, p. 2762, 2017.

[3] I. H. Taylor, S. Dong, and A. Rodriguez, “Gelslim 3.0: High-resolution measurement of shape, force and slip in a compact tactile-sensing finger,” in 2022 international conference on robotics and automation (ICRA). IEEE, 2022, pp. 10 781–10 787.

[4] R. Bhirangi et al., “Anyskin: Plug-and-play skin sensing for robotic touch,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 16 563–16 570.

[5] M. Lambeta et al., “Digit: A novel design for a low-cost compact highresolution tactile sensor with application to in-hand manipulation,” IEEE Robotics and Automation Letters, vol. 5, no. 3, pp. 3838–3845, 2020.

[6] R. Bhirangi, T. Hellebrekers, C. Majidi, and A. Gupta, “Reskin: versatile, replaceable, lasting tactile skins,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, D. Hsu, and G. Neumann, Eds., vol. 164. PMLR, 08–11 Nov 2022, pp. 587–597.

[7] F. Yang et al., “Touch and go: Learning from human-collected vision and touch,” in Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. [Online]. Available: https://openreview.net/forum?id=ZZ3FeSSPPblo

[8] A. Sharma et al., “Self-supervised perception for tactile skin covered dexterous hands,” in 9th Annual Conference on Robot Learning, 2025. [Online]. Available: https://openreview.net/forum?id=eLeCrM5PEO

[9] I. Guzey, B. Evans, S. Chintala, and L. Pinto, “Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play,” in 7th Annual Conference on Robot Learning, 2023. [Online]. Available: https://openreview.net/forum?id=EXQ0eXtX3OW

[10] Q. Liu et al., “Masked visual-tactile pre-training for robot manipulation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 13 859–13 875.

[11] R. Feng et al., “Anytouch 2: General optical tactile representation learning for dynamic tactile perception,” in The Fourteenth International Conference on Learning Representations, 2026.

[12] ——, “Anytouch: Learning unified static-dynamic representation across multiple visuo-tactile sensors,” in The Thirteenth International Conference on Learning Representations, 2025.

[13] Y.-W. Chao et al., “Dexycb: A benchmark for capturing hand grasping of objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 9044–9053.

[14] L. Yang et al., “Oakink: A large-scale knowledge repository for understanding hand-object interaction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 20 953–20 962.

[15] S. Hampali, M. Rad, M. Oberweger, and V. Lepetit, “Honnotate: A method for 3d annotation of hand and object poses,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[16] X. Zhan et al., “Oakink2: A dataset of bimanual hands-object manipulation in complex task completion,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 445–456.

[17] R. Patel, R. Ouyang, B. Romero, and E. Adelson, “Digger finger: Gelsight tactile sensor for object identification inside granular media,” in International Symposium on Experimental Robotics. Springer, 2020, pp. 105–115.

[18] S. Dong, W. Yuan, and E. H. Adelson, “Improved gelsight tactile sensor for measuring geometry and slip,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 137–144.

[19] M. B. Villalonga et al., “Tactile object pose estimation from the first touch with geometric contact rendering,” in Conference on robot learning. PMLR, 2021, pp. 1015–1029.

[20] M. Bauza, A. Bronars, and A. Rodriguez, “Tac2pose: Tactile object pose estimation from the first touch,” The International Journal of Robotics Research, vol. 42, no. 13, pp. 1185–1209, 2023.

[21] H. Xue et al., “Reactive diffusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation,” in Proceedings of Robotics: Science and Systems (RSS), 2025.

[22] W. Chen et al., “Implicitrdp: An end-to-end visual-force diffusion policy with structural slow-fast learning,” IEEE Robotics and Automation Letters, 2026.

[23] T. P. Tomo et al., “A new silicone structure for uskin—a soft, distributed, digital 3-axis skin sensor and its integration on the humanoid robot icub,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2584–2591, 2018.

[24] Y. Song et al., “Fingertip-scale six-axis tactile interface with highprecision force sensing and position localization for dexterous human– machine interactions,” Microsystems & Nanoengineering, vol. 12, no. 1, p. 193, 2026.

[25] BrainCo, “Revo 2 dexterous hand documentation,” https: //www.brainco-hz.com/docs/revolimb-hand/en/revo2/parameters.html, accessed: 2026-08-16.

[26] K. He et al., “Masked autoencoders are scalable vision learners,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 15 979–15 988.

[27] M. Oquab et al., “Dinov2: Learning robust visual features without supervision,” 2023.

[28] A. Radford et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[29] F. Yang et al., “Binding touch to everything: Learning unified multimodal tactile representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 340–26 353.

[30] R. Hoque et al., “Egodex: Learning dexterous manipulation from large-scale egocentric video,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 4218–4237.

[31] R. Punamiya et al., “Egoverse: An egocentric human dataset for robot learning from around the world,” arXiv preprint arXiv:2604.07607, 2026.

[32] H. Luo et al., “Being-h0: Vision-language-action pretraining from large-scale human videos,” in International Conference on Machine Learning. PMLR, 2026.

[33] H. Bi et al., “H-rdt: Human manipulation enhanced bimanual robotic manipulation,” 2025. [Online]. Available: https://embodiedfoundation. github.io/hrdt

[34] Y. Wi et al., “Tactalign: Human-to-robot policy transfer via tactile alignment,” 2026. [Online]. Available: https://arxiv.org/abs/ 2602.13579

[35] S. Zhao et al., “Dexh2r: Task-oriented dexterous manipulation from human to robots,” IEEE/ASME Transactions on Mechatronics, 2025.

[36] H. Zeng et al., “Flowhoi: Flow-based semantics-grounded generation of hand-object interactions for dexterous robot manipulation,” arXiv preprint arXiv:2602.13444, 2026.

[37] Y. Liu et al., “Taco: Benchmarking generalizable bimanual tool-actionobject understanding,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 21 740– 21 751.

[38] Z. Fan et al., “ARCTIC: A dataset for dexterous bimanual hand-object manipulation,” in Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[39] P. Banerjee et al., “HOT3D: Hand and object tracking in 3D from egocentric multi-view videos,” CVPR, 2025.

[40] J. Romero, D. Tzionas, and M. J. Black, “Embodied hands: Modeling and capturing hands and bodies together,” ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), vol. 36, no. 6, Nov. 2017.

[41] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv preprint arXiv:1607.06450, 2016.

[42] P. Wang et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[43] H. Touvron et al., “Training data-efficient image transformer & distillation through attention,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 10 347–10 357. [Online]. Available: https://proceedings.mlr.press/v139/touvron21a.html

[44] Unitree Robotics, “XR-Teleoperate: An open-source teleoperation framework and data collection toolkit for embodied intelligence,” https://github.com/unitreerobotics/xr teleoperate, 2024, accessed: 2026-02.

[45] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained

Bimanual Manipulation with Low-Cost Hardware,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[46] C. Chi et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.