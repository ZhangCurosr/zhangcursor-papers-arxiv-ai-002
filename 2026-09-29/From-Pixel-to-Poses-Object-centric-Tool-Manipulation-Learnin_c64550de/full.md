# From Pixel to Poses: Object-centric Tool Manipulation Learning from Human Demonstrations

Bangjun Wang<sup>1</sup>, Longyan Wu<sup>2</sup>, Yukun Wei<sup>1</sup>, Shenghe Shao<sup>1</sup>, Chaoyi Huang<sup>1</sup>, Wenze Cui<sup>1</sup>, Zetong Xu<sup>1</sup>, Hanlin Wu<sup>1</sup>, Long Chen<sup>3</sup>, Yi Ma<sup>1</sup>, Hongyang Li<sup>1,2</sup>

<sup>1</sup>The University of Hong Kong <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Xiaomi EV

https://github.com/OpenDriveLab/P2P-T

![](images/bc413751ffdc3640d32d67a542854c0c7a8fded6533a80421918ae3f7c6c678e.jpg)  
Fig. 1: Overview of the P2P-T framework. P2P-T leverages in-the-wild human demonstrations (both egocentric and allocentric) to efficiently teach robots complex tool manipulation. Raw video data is first processed through an automated pipeline utilizing foundation models (Grounded-SAM [1], SAM3D [2], and FoundationPose [3]). Stage 1 focuses on pose-aware world model pretraining to extract stable, object-centric pose priors from daily unstructured environments. Stage 2 integrates these learned priors into a low-level execution policy for robot fine-tuning and deployment, successfully bridging the human-robot morphological gap without requiring costly aligned data.

Abstract— Scaling up robotic manipulation is primarily bottlenecked by the scarcity of real-world robot data. While recent approaches leverage human video demonstrations to mitigate this shortage, they remain computationally expensive and still rely on paired human-robot data for domain alignment. Although current state-of-the-arts excel at long-horizon tasks, they struggle with the delicate and precise control required for complex tool manipulation. To overcome these limitations, we introduce P2P-T, from Pixel to Poses for Tool Manipulation, a dataefficient, object-centric framework that learns tool use directly from human demonstrations. P2P-T bridges the cognitive and physical execution gap through a two-stage approach. First, pretraining an object-centric world model to extract stable pose priors; Second, integrating these priors into an efficient, pose-aware low-level policy. By utilizing a robust automated data processing pipeline powered by modern foundation models, P2P-T completely bypasses the need for human-robot aligned data. This reduces overall training overhead drastically. With minimal per-task fine-tuning, our framework achieves a 73% improvement over the previous state of the art in execution performance on complex, real-world tool manipulation tasks that currently remain out of reach for standard large-scale pretrained models.

## I. INTRODUCTION

The scarcity of in-domain, real-world robot data has long been the primary bottleneck preventing the field of robotics from scaling up. Collecting paired teleoperation data is notoriously expensive and labor-intensive. To mitigate this shortage, recent approaches such as EgoScale [4] and EgoVLA [5] have begun to investigate whether models can be directly trained on massive datasets of human videos, or if such data can be integrated into the pretraining recipe. While leveraging human demonstrations presents a highly promising and scalable direction, existing methods are constrained by several practical limitations.

First, these approaches do not fundamentally eliminate the reliance on human-robot aligned data. They still require an additional mid-training or post-training stage using paired datasets to bridge the morphological gap between human hands and robotic end-effectors. Second, the computational cost of extracting motion priors directly from raw human video is exceptionally high. Executing a full-stage training pipeline often demands immense computing resources, sometimes on the scale of hundreds of GB200 GPUs. Finally, these methods suffer from a distinctly limited task scope. Models trained directly on human data are largely restricted to simple pick-and-place actions. Similarly, while current state-of-theart pre-trained models like $\pi _ { 0 }$ [6] and GEN-0 [7] are highly capable of executing long-horizon planning tasks, they still fall short when confronted with complex tool manipulation. Such tasks inherently demand far more delicate and precise physical control than current generalist models can reliably output.

The importance of mastering tool manipulation cannot be overstated. In dynamic deployments like household environments, it is entirely impractical for robots to frequently change their hardware end-effectors based on the specific task at hand. Therefore, the necessity for a real-world household agent to master the use of tools specifically designed for human hands is self-evident. Confining robots to simple tasks fails to fully exploit their inherent agility and artificially restricts their operational scope. Historically, previous works such as FUNCTO [8] and SimToolReal [9] have attempted to solve complex tool-use tasks primarily through rigid rulebased systems or reinforcement learning methods, which often struggle to generalize across various real-world scenarios.

To overcome these challenges, we delve into the possibility of simultaneously instilling both the high-level cognitive capabilities and the low-level executive precision required for tool-use tasks through a highly efficient framework. As shown in Figure. 1, we propose P2P-T, an efficient, objectcentric framework for learning tool manipulation directly from human demonstrations. P2P-T is architected across two distinct stages. In the first stage, rather than attempting to derive dense motion priors directly from unstructured human data, we focus on object-centric world model pretraining. We refine our training objectives to extract object-centric pose priors, which provide a more stable and abstract representation of the task environment. In the second stage, we conduct an efficient, pose-aware policy post-training. During this phase, we seamlessly integrate the pretrained object-centric world model with a low-level execution module. By leveraging the structured pose priors learned during the pretraining phase, we significantly enhance the robot’s physical execution capabilities.

In summary, our proposed methodology provides direct solutions to the traditional bottlenecks of robot learning through the following key contributions:

• Automated Data Pipeline: Capitalizing on modern foundation models (Grounded-SAM [1], SAM3D [2], and FoundationPose [3]), we construct a robust data processing pipeline that drastically streamlines complex curation procedures and generates clean, low-noise training data.

• Elimination of Aligned Data: Our approach successfully bypasses the need for human-robot paired data. By utilizing an object-centric world model design, we completely eliminate the intermediate training stages typically used to map pretrained representations to a robot’s physical action space, drastically reducing overall training overhead.

• Scalable Two-Stage Framework: We introduce a dataefficient object-centric world model alongside a corresponding pose-aware low-level execution policy. With minimal per-task fine-tuning, P2P-T achieves strong execution performance on complex tool manipulation tasks that currently remain beyond the capabilities of standard, large-scale pretrained models.

## II. RELATED WORK

a) Learning from Human.: Scaling robot manipulation requires supervision beyond the limited amount of real-world robot data that can be collected through teleoperation. Recent generalist robot policies and vision-language-action(VLA) models have shown that large-scale robot datasets can provide strong policy initializations for downstream manipulation tasks [6,10,11,12,13]. However, these methods still rely heavily on robot embodiment-specific action data, which remains expensive to collect and often limits the diversity of tasks, scenes, and physical interactions available during training. To overcome this bottleneck, a growing line of work studies how to leverage human videos as a scalable source of behavioral supervision. LAPA learns latent actions from videos without requiring ground-truth robot action labels [14], while Dreamitate fine-tunes a video generative model on human demonstrations and uses generated execution videos to guide real-world robot control [15]. EgoVLA [5] and EgoScale [4] further demonstrate the promise of largescale egocentric human videos for VLA pretraining and dexterous manipulation transfer. Despite their scalability, these approaches usually require either human-to-robot retargeting, paired alignment data, or additional robot fine-tuning to bridge the morphological gap between human hands and robot endeffectors. In contrast, P2P-T avoids directly modeling human hand motion. Instead, it extracts embodiment-independent object pose trajectories from human demonstrations and learns reusable tool-use priors in the object coordinate space.

b) Tool Manipulation.: Tool manipulation is substantially more challenging than standard pick-and-place manipulation because it requires reasoning about functional affordances, contact-rich interaction, precise tool orientation, and often forceful or long-horizon execution. Recent works have explored structured representations to improve generalization in tool-use tasks. FUNCTO introduces a functioncentric one-shot imitation learning framework that extracts 3D functional keypoints from a human demonstration and establishes correspondences between tools with different geometries [8]. SimToolReal studies zero-shot dexterous tool manipulation by procedurally generating diverse tool-like objects in simulation and training a single goal-conditioned RL policy to manipulate tools toward target poses [9]. SPOT further shows that SE(3) object pose trajectories can serve as an effective intermediate representation for objectcentric imitation learning, decoupling task constraints from robot-specific actions and enabling learning from actionless human videos [16]. Although FUNCTO and SPOT are closely related to our focus, neither method is open-sourced, preventing us from including them in our experimental comparisons under the same evaluation protocol. These methods nonetheless highlight the importance of structured object-level representations for tool use. Our work shares this object-centric motivation but differs in both its learning source and policy integration: P2P-T learns pose priors directly from in-the-wild human demonstrations through an automated visual processing pipeline and injects these priors into a lowlevel, pose-aware action policy for precise execution.

c) World Model.: World models aim to learn compact predictive representations of environment dynamics, which can support planning, policy learning, and sample-efficient control [17,18]. In robotic manipulation, object-centric world models are particularly attractive because manipulation tasks are often defined by how task-relevant objects move and interact. Recent video-based policy learning methods suggest that generative or predictive models can provide useful intermediate plans for downstream control [15]. However, many world-model-based approaches either operate in latent visual spaces or require additional mechanisms to translate predicted dynamics into precise robot actions. P2P-T instead builds a world model over explicit 6D tool poses. This design provides a structured and physically meaningful prediction target that is more directly transferable across embodiments. Moreover, recent foundation models for open-vocabulary segmentation, 3D reconstruction, and 6D pose estimation make it increasingly feasible to extract such structured supervision from raw videos at scale [1,2,3]. By combining automated pose extraction with object-centric prediction, P2P-T learns a compact pose prior from human demonstrations and uses it to guide a downstream diffusion-based robot policy.

## III. METHODS

In this section, we detail P2P-T, an efficient two-stage framework for learning complex tool manipulation directly from human demonstrations. Instead of forcing a model to learn morphology-dependent human hand motions from raw videos, our methodology relies on a core insight: while human and robot hands differ fundamentally, the spatial trajectory required to manipulate a specific tool remains consistent. Therefore, P2P-T isolates the task-relevant tool to model its spatial dynamics independently of the embodiment. We first describe an automated pipeline that extracts 6D pose trajectories from human videos. Next, we detail Stage 1, which pretrains an embodiment-independent world model to predict future tool states. Finally, Stage 2 integrates these learned pose priors into a low-level policy to guide precise real-world execution.

## A. Human Data Collection and Pre-process

To construct our Stage 1 pre-training corpus, we recruited human volunteers over a two-month period to capture inthe-wild video demonstrations of diverse tool-use tasks. This self-curated dataset, combined with the existing TACO [19] dataset, comprises our complete pre-training mixture. Detailed statistics are available in Section. III-A.1

While previous approaches are often bottlenecked by brittle, heuristic-based pre-processing methods, we leverage recent advances in foundation models to introduce a streamlined and highly robust pipeline. This approach not only yields highquality data labels but also significantly reduces annotation costs, paving the way for future scalability.

1) Data Distribution of Stage 1: The pretraining corpus for the Stage-1 object-centric world model consists of 3,094 video clips spanning 17 distinct tool categories. As illustrated in the data distribution (Figure 2), the dataset exhibits a natural long-tail variance. The most frequently represented tools are those requiring complex, dynamic, or contact-rich interactions, such as spoons (479 clips) and knives (464 clips). These are closely followed by tools that demand precise orientation and force application, including rollers (377 clips), spatulas (343 clips), hammers (264 clips), and brushes (248 clips).

This concentration of data on highly functional tools is intentional; it ensures the model receives rich supervision to learn robust 6D pose priors for the intricate SE(3) spatial maneuvers required during downstream task execution. Conversely, less frequently represented objects include simpler or less dynamically manipulated items like plates (16 clips), glue (9 clips), and soap (2 clips). Despite this class imbalance, combining our self-curated in-the-wild videos with the existing TACO dataset provides a highly diverse and comprehensive foundation. This variety enables the world model to effectively capture embodiment-independent spatial dynamics, preparing the low-level policy to generalize across a broad array of household tools.

![](images/ea7c73299c7c9c2564df0d45b83a45085b48bf7bde6da37cd293a955e2891073.jpg)

![](images/9749d767ef3d327418a32991766cda32189f37c30efeb6d3c1f3198475000845.jpg)  
Fig. 2: Stage 1 Pretrain Data Distribution. This figure illustrates the composition of the pretraining dataset used for the Stage-1 object-centric world model. The dataset contains a total of 3,094 video clips distributed across 17 different tool categories. The distribution highlights a strong focus on highly interactive and geometrically complex tools, with spoons (479) and knives (464) being the most prevalent. This diverse mixture provides the rich spatial supervision necessary for extracting stable, embodiment-independent pose priors.

A human verification step was applied during preprocessing to assess the quality of the object tracking results. Clips with inaccurate or unreliable trajectories were discarded, yielding an overall discard rate of 27%, which we consider acceptable given the in-the-wild nature of the collected videos. The discard rate varies across tool categories according to the complexity of their motion and geometry. For example, cups have a relatively low discard rate of 19% because their usage typically involves simple and easily trackable motions. In contrast, screwdrivers have a substantially higher discard rate of 37.875%, as their rotational motion is difficult to track reliably when the tool is approximately symmetric about its z-axis. This verification process improves the quality of the retained trajectories and provides more reliable supervision for learning the Stage-1 pose priors.

![](images/2ac5ae5d4bee6edab3f4dbd37d8fe71bc478e74654929eb8c5ce5ff658b63e3d.jpg)  
Fig. 3: Overview of the P2P-T pre-processing pipeline. We first extract the tool from the background environment to create a 2D segmentation mask. The mask and the raw RGB frames are subsequently fed into SAM 3D [2], which outputs a high-fidelity, instance-specific 3D tool mesh. Finally, FoundationPose [3] takes the reconstructed 3D mesh and the sequence of raw video frames as inputs to compute and track precise 6D tool poses across the entire demonstration sequence.

2) Pose Representation: To accurately capture the spatial dynamics of tool manipulation, we formulate the state of the tool as a 6-Degree-of-Freedom (6DoF) pose within 3D space. Formally, for a given video frame at timestamp t, the rigid transformation of the tool relative to the camera coordinate frame is represented by a homogeneous transformation matrix $P _ { t } \in S E ( 3 )$

$$
P _ { t } = { \left[ \begin{array} { l l } { R _ { t } } & { T _ { t } } \\ { \mathbf { 0 } } & { 1 } \end{array} \right] }\tag{1}
$$

where the translation vector $T _ { t } = [ x , y , z ] ^ { \top } \in \mathbb { R } ^ { 3 }$ denotes the 3D spatial location of the tool’s centroid, and the rotation matrix $R _ { t } \in S O ( 3 )$ describes its 3D orientation.

Although the transformation has six intrinsic degrees of freedom, we retain the matrix representation during training. Specifically, each pose matrix is flattened into a 16-dimensional vector:

$$
p _ { t } = \operatorname { v e c } ( P _ { t } ) \in \mathbb { R } ^ { 1 6 } ,\tag{2}
$$

where vec(·) denotes row-wise vectorization. The vector is reshaped back into a 4 × 4 matrix when the predicted pose is decoded or evaluated. This over-parameterized representation is intentional. Unlike quaternions, which have the sign ambiguity q and −q representing the same rotation, and Euler angles, which suffer from angle wrapping and representation singularities, the matrix entries vary continuously with the underlying rotation. Consequently, flattening the homogeneous transformation provides an unambiguous and continuous regression target, avoiding the discontinuous targets that can lead to oscillatory training behavior.

3) Pre-processing Pipeline: Extracting precise 6D tool poses from in-the-wild videos is challenging due to dynamic backgrounds, occlusions, and absent 3D annotations. To address this, we propose an automated pipeline leveraging foundation models for zero-shot 3D reconstruction and tracking (Figure 3). For an input sequence $V = \{ I _ { 1 } , \ldots , I _ { N } \}$ where $I _ { t }$ is an RGB frame, our pipeline proceeds in three stages:

1) 2D Mask Extraction: We employ Grounded-SAM [1] to isolate the target tool and generate a binary segmentation mask $S _ { t }$

2) 3D Mesh Reconstruction: Feeding the multi-view RGB frames and masks into SAM 3D [2], we lift the 2D visual boundaries into 3D space to output a high-fidelity tool mesh $M = ( \mathcal { V } , \mathcal { F } )$

3) 6D Pose Tracking: FoundationPose [3] aligns the geometric and textural priors of the 3D mesh M with the raw sequence $I _ { t }$ to reliably compute precise 6D tool poses $p _ { t }$ across the entire demonstration.

Ultimately, this automated pipeline transforms raw videos into a dataset of paired $\left( I _ { t } , p _ { t } \right)$ samples after human validation. By removing human annotators from the loop, we effectively bypass traditional 3D annotation bottlenecks to provide scalable, high-quality supervision.

## B. Stage 1: Object-centric World Model Pretraining

Stage 1 learns a reusable, object-centric world model from human demonstrations without introducing a robot-specific action space. Its central representation is a sequence of pose latents spanning future time steps t + 1 through $t + H - 1$ These latents capture task-conditioned tool dynamics and are learned through supervision on decoded tool poses, together with auxiliary visual predictions over the same horizon. Since the desired tool trajectory is largely shared across embodiments, this formulation reduces the need for paired human–robot demonstrations.

Each demonstration consists of a sequence of RGB observations $I _ { t } ,$ a language instruction c, and tool poses $p _ { t } \in \mathbb { R } ^ { 1 6 }$ obtained by flattening homogeneous transformation matrices $P _ { t } \in S E ( 3 )$ ). As illustrated in the left part of Figure 4, the world model autoregressively predicts pose and visual latents over a horizon of $H - 1$ future steps, conditioned on the current observation, instruction, and tool pose:

$$
\left\{ Z _ { t + k } ^ { \mathrm { p o s e } } , Z _ { t + k } ^ { \mathrm { v i s } } \right\} _ { k = 1 } ^ { H - 1 } = F _ { \theta } ( I _ { t } , c , p _ { t } ) .\tag{3}
$$

Separate prediction heads decode the latents at each future step into a tool pose and a visual observation:

$$
\hat { p } _ { t + k } = D _ { \mathrm { p o s e } } \left( Z _ { t + k } ^ { \mathrm { p o s e } } \right) , \qquad \hat { I } _ { t + k } = D _ { \mathrm { f r a m e } } \left( Z _ { t + k } ^ { \mathrm { v i s } } \right) ,\tag{4}
$$

$k = 1 , \ldots , H - 1$ , The decoded predictions provide supervi sion across the future trajectory, encouraging the pose latents to capture temporally coherent tool dynamics. The learned pose latents subsequently guide the pose-aware low-level policy in Stage 2.

The world model adopts a transformer-based multimodal fusion architecture. Visual observations are encoded using frozen DINOv2 [21] and SigLIP [20] encoders, while the language instruction is encoded using a frozen T5 [22] encoder. Freezing these pretrained backbones reduces computational cost and allows the trainable components to focus on taskconditioned spatial and temporal dynamics.

The visual features, language features, and current tool pose are projected into a shared latent space and augmented with learnable modality embeddings. These tokens are concatenated with learnable fusion tokens and processed by an autoregressive transformer to produce the future pose and visual latents. A pose head decodes the pose latents into tool poses, while an auxiliary frame-prediction head decodes the visual latents into the corresponding future observations.

![](images/1c550f2295c88e18915d1e990b2bce1562077ffb6077d0c2eea2c88dfb542115.jpg)  
Fig. 4: Overview of P2P-T’s two-stage framework. Stage 1: An autoregressive transformer learns an object-centric world model from human demonstrations by fusing observations encoded with SigLIP [20] and DINOv2 [21], instructions encoded with T5 [22], and projected tool poses. It learns future pose latents spanning t + 1 through $t + H - 1 .$ , supervised through decoded poses and auxiliary frame predictions. Stage 2: The frozen world model provides future pose latents, mapped through a trainable adaptor to condition an RDT-based diffusion action expert. Together with visual observations, language instructions, and robot proprioception, these latents guide action chunks $a _ { t : t + H - 1 }$ for receding-horizon execution, connecting human-derived object-centric dynamics with robot-specific control without paired human–robot demonstrations.

We jointly supervise the decoded pose and visual predictions across the entire prediction horizon. The pose regression objective is

$$
\mathcal { L } _ { \mathrm { p o s e } } = \frac { 1 } { H - 1 } \sum _ { k = 1 } ^ { H - 1 } \left\| \hat { p } _ { t + k } - p _ { t + k } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

The auxiliary frame-prediction objective provides additional supervision for learning scene dynamics:

$$
\mathcal { L } _ { \mathrm { f r a m e } } = \frac { 1 } { H - 1 } \sum _ { k = 1 } ^ { H - 1 } \left\| \hat { I } _ { t + k } - I _ { t + k } \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

The complete Stage 1 objective is

$$
\mathcal { L } _ { \mathrm { s t a g e 1 } } = \mathcal { L } _ { \mathrm { p o s e } } + \lambda \mathcal { L } _ { \mathrm { f r a m e } } ,\tag{7}
$$

where λ balances the two objectives. We set $\lambda = 0 . 0 1$ to regularize the learned dynamics through visual prediction while prioritizing pose learning. Through this joint supervision over multiple future steps, Stage 1 learns structured, objectcentric pose latents that encode future tool trajectories and support policy learning in Stage 2.

## C. Stage 2: Pose-aware Post-training

Stage 2 translates the object-centric pose latents learned in Stage 1 into executable robot actions by post-training a lowlevel action expert initialized from a pretrained RDT [12] backbone. The action expert conditions on a sequence of future pose latents, allowing it to use the predicted tool trajectory as a structured task-space prior.

Given the current RGB observation $I _ { t } ^ { c }$ from a designated camera view, the language instruction $c ,$ and the current tool pose $p _ { t }$ when available, the frozen Stage-1 model produces pose latents spanning time steps $t + 1$ through $t + H - 1 \colon$

$$
\left\{ Z _ { t + k } ^ { \mathrm { p o s e } } \right\} _ { k = 1 } ^ { H - 1 } = F _ { \phi } ^ { \mathrm { p o s e } } ( I _ { t } ^ { c } , c , p _ { t } ) ,\tag{8}
$$

where $F _ { \phi } ^ { \mathrm { p o s e } }$ denotes the pose-latent output of the pretrained world model. A shared trainable MLP projects each pose latent token into the RDT embedding space:

$$
U _ { t + k } ^ { \mathrm { p o s e } } = \mathrm { M L P } _ { p } \left( Z _ { t + k } ^ { \mathrm { p o s e } } \right) , \qquad k = 1 , \dots , H - 1 .\tag{9}
$$

The projected tokens are concatenated in temporal order to form

$$
U _ { t } ^ { \mathrm { p o s e } } = \mathrm { C o n c a t } \left( U _ { t + 1 } ^ { \mathrm { p o s e } } , \dots , U _ { t + H - 1 } ^ { \mathrm { p o s e } } \right) ,\tag{10}
$$

and inserted before the robot’s proprioceptive state token $s _ { t }$ in the RDT input sequence. The action expert thus operates directly on the future pose latents, without requiring decoded pose matrices as policy inputs.

The policy is trained as a conditional diffusion model to predict an action chunk $a _ { t : t + H - 1 }$ , conditioned on the pose latent sequence, the proprioceptive state $s _ { t } .$ , the multi-view RGB history $\mathcal { T } _ { t } .$ and the language instruction $c .$ The Stage-1 model and the policy’s T5 [22] and SigLIP [20] encoders remain frozen, while the RDT action expert and pose adaptor are optimized.

During training, Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ is added to the demonstrated action chunk using a DDPM scheduler [23]:

$$
\tilde { a } ^ { n } = \sqrt { \bar { \alpha } _ { n } } a _ { t : t + H - 1 } + \sqrt { 1 - \bar { \alpha } _ { n } } \epsilon ,\tag{11}
$$

where n denotes the diffusion step and $\hat { \alpha } _ { n }$ is the cumulative noise-schedule coefficient. We use sample prediction, training the action expert to recover the clean action chunk directly:

$$
\mathcal { L } _ { \mathrm { a c t } } = \mathbb { E } _ { \mathcal { D } , n , \epsilon } \left[ \left| \left| a _ { \theta } \left( \tilde { a } ^ { n } , U _ { t } ^ { \mathrm { p o s e } } , s _ { t } , \mathcal { T } _ { t } , c , n \right) - a _ { t : t + H - 1 } \right| \right| _ { 2 } ^ { 2 } \right] ,\tag{12}
$$

where D denotes the robot demonstration dataset. Conditioning on pose latents across the prediction horizon guides action generation with both the spatial structure and temporal progression of the intended tool trajectory.

During inference, the frozen Stage-1 model first generates the future pose latent sequence. The action expert then conditions on its projected tokens to sample an action chunk using a fast DPM-Solver [24]. Actions are executed in a receding-horizon manner, with the pose latents and action chunk updated as new observations become available. This architecture connects embodiment-independent tool dynamics with robot-specific control through a learned latent interface.

![](images/6956c7ded8456806d32809ab6c7675aa8691d2f9fb79ceb547fe6bbb15c74491.jpg)  
Fig. 5: Real-world evaluation tasks. We evaluate on six complex tool-use scenarios that demand delicate physical control and continuous spatial reasoning, moving far beyond standard pick-and-place actions.

## IV. EXPERIMENTS

The primary goal of our evaluation is to assess whether P2P-T can effectively bridge the morphological gap and enable precise tool manipulation without relying on paired human-robot data. Unlike conventional evaluations restricted to simple pick-and-place maneuvers, we benchmark our framework on a curated suite of complex tool-use tasks. These tasks require not only a high-level cognitive understanding of functional affordances but also delicate, long-horizon execution involving intricate SE(3) spatial movements.

Our experiments are structured to answer the following core questions:

1) How does P2P-T compare against state-of-the-art visionlanguage-action baselines in complex, real-world tool manipulation tasks? (See Sec. IV-C)

2) Does the object-centric world model learned in Stage 1 provide a robust and transferable pose prior for the downstream low-level execution policy? (See Sec. IV-D)

3) How does scaling the model size affect the performance of our two-stage framework on complex tool manipulation tasks? (See Sec. IV-E)

## A. Tasks

To systematically assess our framework’s capacity for precise, pose-aware physical execution, we designed a suite of six real-world tasks (Figure 5) on a UR10e robot with a Robotiq 2F-140 gripper and two D-435 RealSense cameras. Each task challenges a distinct facet of complex spatial manipulation: striking a target with a hammer using dynamic, high-velocity impacts; picking up a screwdriver and rotate it to match with the screw; pouring from a cup while maintaining orientation control; slicing with a knife using precise downward force; sweeping with a brush through sustained surface contact; and scooping with a spoon via coordinated pitch adjustments. During each trial, tools are randomly placed on the table with various initial poses. Models are required to re-orient them to their specific working poses and execute the operation.

By evaluating across these diverse tool categories, we explicitly test the framework’s capacity to translate humandesigned tool affordances into actionable, robust robot execution, moving far beyond the primitive constraints of standard pick-and-place benchmarks.

## B. Implementation Details

We pretrain the Stage-1 object-centric world model on parsed TACO [19] trajectories and human demonstrations. It processes RGB frames and FoundationPose-estimated [3] 6D tool poses via frozen DINOv2 [21], SigLIP [20], and T5-base [22] encoders, predicting next tool poses via a transformer. Stage 2 initializes the low-level execution policy from RDT [12], conditioned on Stage-1 pose priors. The policy fuses multi-view images, proprioception, and language to predict 64-step action chunks via diffusion.

Octo, OpenVLA-OFT, and $\pi _ { 0 . 5 }$ use official defaults with adapted interfaces and action horizons of 4/8/50; Octo and $\pi _ { 0 . 5 }$ use 10 denoising/integration steps. LingBot-VA finetunes its ∼5.1B transformer for 40k steps/task on 4 GPUs (BF16, FSDP, checkpointing, batch size 1/rank, AdamW lr $1 0 ^ { - 5 }$ , weight decay 0.1). Videos use 256×320 resolution at 10 fps (8 actions/latent frame); inference uses 5/10 video/action denoising steps, guidance 5/1, and 16-action chunks at 8 Hz. FastWAM fine-tunes Wan2.2-TI2V-5B video/action DiTs and proprioceptive encoder for 80k steps/task (frozen VAE/text encoders) using BF16, AdamW (batch size 16, weight decay 0.01, peak lr $1 0 ^ { - 4 }$ , decay $1 0 ^ { - 5 }  1 0 ^ { - 6 } )$ . Inputs use $2 2 4 \times$ 224 images and min–max-normalized states/actions. Inference runs 10 denoising steps for 32-action predictions, executing 5 before replanning at 10 Hz.

TABLE I: Real-world Policy Evaluation. We report the success rate for each task, together with the average success rate and inference rate. Each model is fine-tuned with 100 demonstrations and evaluated over 50 trials per task. We report the average results, with the best success rates highlighted in bold. Inference rate is measured in Hz, with higher values being better.
<table><tr><td>Method</td><td>Hammer</td><td>Cup</td><td>Brush</td><td>Screwdriver</td><td>Knife</td><td>Spoon</td><td>Average (↑)</td><td>Execution Rate (Hz) (↑)</td></tr><tr><td colspan="9">Vision-Language-Action Models</td></tr><tr><td>DP2-DINOv2 [25]</td><td>0.09</td><td>0.12</td><td>0.03</td><td>0.02</td><td>0.09</td><td>0.04</td><td>0.07</td><td>30</td></tr><tr><td>Octo [10]</td><td>0.21</td><td>0.27</td><td>0.27</td><td>0.17</td><td>0.16</td><td>0.26</td><td>0.22</td><td>15</td></tr><tr><td>OpenVLA-OFT [26]</td><td>0.09</td><td>0.27</td><td>0.13</td><td>0.17</td><td>0.24</td><td>0.12</td><td>0.17</td><td>25</td></tr><tr><td>π0.5 [13]</td><td>0.24</td><td>0.49</td><td>0.27</td><td>0.35</td><td>0.37</td><td>0.24</td><td>0.33</td><td>30</td></tr><tr><td colspan="9">World Action Models</td></tr><tr><td>FastWAM [27]</td><td>0.00</td><td>0.08</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.00</td><td>0.02</td><td>10</td></tr><tr><td>Lingbot-VA [17]</td><td>0.12</td><td>0.38</td><td>0.04</td><td>0.36</td><td>0.22</td><td>0.14</td><td>0.21</td><td>8</td></tr><tr><td>P2P-T (Ours)</td><td>0.59</td><td>0.60</td><td>0.55</td><td>0.54</td><td>0.57</td><td>0.58</td><td>0.57</td><td>30</td></tr></table>

TABLE II: Ablations on Framework Design.
<table><tr><td>Object-centric Pretraining</td><td>Pose-aware Post-training</td><td>Hammer</td><td>Cup</td><td>Average</td></tr><tr><td>√</td><td>x</td><td>0.12</td><td>0.14</td><td>0.13</td></tr><tr><td>x</td><td>√</td><td>0.32</td><td>0.36</td><td>0.34</td></tr><tr><td>√</td><td>√</td><td>0.59</td><td>0.60</td><td>0.60</td></tr></table>

## C. Result Analysis

To evaluate real-world tool manipulation, we compare P2P-T against state-of-the-art vision-language-action baselines (Table I). Our framework achieves an average success rate of 0.57, substantially outperforming the strongest baseline, $\pi _ { 0 . 5 }$ [13] (0.33), showing an average improvement of 73%.

Trajectory analysis reveals that while generalist visionlanguage-action models like $\pi _ { 0 . 5 }$ exhibit strong initial grasping, they lack explicit knowledge of functional geometry. Consequently, they consistently fail during the interaction phase, struggling with the intricate SE(3) spatial maneuvers required to correctly orient and translate tools to their functional working positions $( e . g .$ , aligning a hammer head).

Meanwhile, although world action models (WAMs), such as FastWAM [27] and Lingbot-VA [17], have demonstrated promising generalization on pick-and-place and assembly tasks, they remain inadequate for tool-use scenarios, which require substantially more complex and dexterous manipulation. Given the same number of demonstrations, these models are also more difficult to fine-tune for novel tool-use tasks, as they must learn both the task-specific interaction dynamics and the precise, dexterous motions required to operate the tools, thus resulting in suboptimal performance(0.02 and 0.21 respectively). FastWAM achieves a 0% success rate on certain tasks, primarily because 100 demonstrations are insufficient for full task adaptation, which is a limitation consistent with our prior experience.Their lower inference rates also reduce responsiveness: FastWAM and Lingbot-VA operate at 10 Hz and 8 Hz, respectively, compared with 30 Hz for P2P-T.

TABLE III: Scaling Verification. Performance comparison among the P2P-T-Small, P2P-T-Base, and P2P-T-Large variants. “#Param.” denotes the number of parameters.
<table><tr><td>Model Variant</td><td>#Param.</td><td>Hammer</td><td>Cup</td><td>Average</td></tr><tr><td>P2P-T-Small</td><td>1.189B</td><td>0.34</td><td>0.48</td><td>0.41</td></tr><tr><td>P2P-T-Base</td><td>1.921B</td><td>0.59</td><td>0.60</td><td>0.60</td></tr><tr><td>P2P-T-Large</td><td>2.832B</td><td>0.64</td><td>0.68</td><td>0.66</td></tr></table>

Conversely, while P2P-T occasionally shows minor grasping instability due to its smaller pretraining scale, this is overwhelmingly compensated for by the robust Stage-1 pose priors. These structured priors provide superior spatial awareness, enabling the policy to precisely reason over the tool’s 6D geometry, maintain continuous working orientations, and successfully execute complex tasks.

## D. Ablations

To evaluate the necessity and transferability of the Stage-1 pose priors, we ablate our framework’s components (Table II).

First, attaching a standard DP2-DINOv2 [25] policy to the Stage-1 world model results in poor success rates (0.12 on Hammer and 0.14 on Cup). This suggests that, without a tailored pose-aware architecture, standard policies cannot fully exploit the world model’s geometric priors for complex SE(3) maneuvers.

Second, a fine-tuned RDT [12] conditioned on the current pose estimated by FoundationPose outperforms the DP2 baseline, achieving 0.32 on Hammer and 0.36 on Cup, but still falls short of satisfactory performance. This suggests that implicit 2D visual features and current-pose information alone are insufficient for fine-grained tool alignment. Pose prior guidance is essential for models to learn those specific tool protocols.

In contrast, our two-stage framework synergizes both components, leaping to 0.59 on Hammer and 0.60 on Cup. This confirms that the Stage-1 world model provides a robust pose prior that, when ingested by the Stage-2 policy, effectively bridges the gap between high-level functional understanding and low-level execution.

## E. Scaling Potential

To evaluate the effect of model scaling on performance, we compare three variants of our framework: P2P-T-Small (1.19B parameters), P2P-T-Base (1.92B parameters), and P2P-T-Large (2.83B parameters).

As shown in Table III, performance improves consistently with model capacity: P2P-T-Small reaches an average success rate of 0.41, P2P-T-Base reaches 0.60, and P2P-T-Large reaches 0.66. Scaling allows the architecture to absorb more complex spatial distributions. Qualitatively, the larger variants exhibit enhanced generalization to geometric variations (e.g., differing handle lengths or weights) and dynamic disturbances. While the smaller models occasionally require behavioral corrections under extreme variance, the scaled policies yield more robust, smoother action trajectories. This trend indicates that our two-stage framework possesses significant potential to generalize across a broader array of un-modeled household tools as capacity increases.

## V. LIMITATIONS

We present P2P-T, an efficient two-stage framework for tool manipulation that bypasses the need for paired humanrobot data. By extracting 6D pose trajectories from human demonstrations, we pretrain an object-centric world model to capture transferable spatial dynamics. Integrating these priors into a pose-aware diffusion policy effectively bridges the human-robot morphological gap without costly domain alignment.

Limitation. Despite its strong performance, labor and compute constraints currently limit our evaluation to a single robot embodiment, precluding the use of multi-fingered dexterous hands. Furthermore, verifying the framework’s scaling behavior under massive parameter regimes remains unexplored. Expanding to diverse embodiments, dexterous hardware, and larger-scale training represent critical directions for future work.

## REFERENCES

[1] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan et al., “Grounded-SAM: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

[2] X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li et al., “SAM-3D: 3dfy anything in images,” arXiv preprint arXiv:2511.16624, 2025.

[3] B. Wen, W. Yang, J. Kautz, and S. Birchfield, “FoundationPose: Unified 6d pose estimation and tracking of novel objects,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 17 868–17 879.

[4] R. Zheng, D. Niu, Y. Xie, J. Wang, M. Xu, Y. Jiang, F. Castaneda, F. Hu,˜ Y. L. Tan, L. Fu, T. Darrell, F. Huang, Y. Zhu, D. Xu, and L. Fan, “EgoScale: Scaling dexterous manipulation with diverse egocentric human data,” arXiv preprint arXiv:2602.16710, 2026.

[5] R. Yang, Q. Yu, Y. Wu, R. Yan, B. Li, A.-C. Cheng, X. Zou, Y. Fang, X. Cheng, R.-Z. Qiu, H. Yin, S. Liu, S. Han, Y. Lu, and X. Wang, “EgoVLA: Learning vision-language-action models from egocentric human videos,” arXiv preprint arXiv:2507.12440, 2025.

[6] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky, “π<sub>0</sub>: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2026.

[7] Generalist AI Team, “GEN-0: Embodied foundation models that scale with physical interaction,” https://generalistai.com/blog preview-uqlxvb-bb.html, 2025.

[8] C. Tang, A. Xiao, Y. Deng, T. Hu, W. Dong, H. Zhang, D. Hsu, and H. Zhang, “FUNCTO: Function-centric one-shot imitation learning for tool manipulation,” arXiv preprint arXiv:2502.11744, 2025.

[9] K. Kedia, T. G. W. Lum, J. Bohg, and C. K. Liu, “SimToolReal: An object-centric policy for zero-shot dexterous tool manipulation,” arXiv preprint arXiv:2602.16863, 2026.

[10] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu et al., “Octo: An open-source generalist robot policy,” arXiv preprint arXiv:2405.12213, 2024.

[11] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “OpenVLA: An opensource vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[12] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “RDT-1B: A diffusion foundation model for bimanual manipulation,” in International Conference on Learning Representations, vol. 2025, 2025.

[13] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky, “π : A vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[14] S. Ye, J. Jang, B. Jeon, S. J. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, B. Y. Lin et al., “Latent action pretraining from videos,” in International Conference on Learning Representations, 2025.

[15] J. Liang, R. Liu, E. Ozguroglu, S. Sudhakar, A. Dave, P. Tokmakov, S. Song, and C. Vondrick, “Dreamitate: Real-world visuomotor policy learning via video generation,” arXiv preprint arXiv:2406.16862, 2024.

[16] C.-C. Hsu, B. Wen, J. Xu, Y. Narang, X. Wang, Y. Zhu, J. Biswas, and S. Birchfield, “SPOT: SE-(3) pose trajectory diffusion for object-centric manipulation,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 4853–4860.

[17] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu et al., “Causal world modeling for robot control,” arXiv preprint arXiv:2601.21998, 2026.

[18] J. Yang, K. Lin, J. Li, W. Zhang, T. Lin, L. Wu, Z. Su, H. Zhao, Y.-Q. Zhang, L. Chen et al., “RISE: Self-improving robot policy with compositional world model,” arXiv preprint arXiv:2602.11075, 2026.

[19] Y. Liu, H. Yang, X. Si, L. Liu, Z. Li, Y. Zhang, Y. Liu, and L. Yi, “TACO: Benchmarking generalizable bimanual tool-action-object understanding,” arXiv preprint arXiv:2401.08399, 2024.

[20] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[21] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[22] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[23] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[24] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “DPM-Solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” Advances in neural information processing systems, vol. 35, pp. 5775–5787, 2022.

[25] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[26] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-languageaction models: Optimizing speed and success,” arXiv preprint arXiv:2502.19645, 2025.

[27] T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-WAM: Do world action models need test-time future imagination?” arXiv preprint arXiv:2603.16666, 2026. [Online]. Available: https://arxiv.org/abs/ 2603.16666