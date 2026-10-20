# OV-PEAR: Observation-Verified Pose-Scale Estimation and Reconstruction for Unknown Industrial Manipulation Objects

Guoqin Tang, Qingxuan Jia, Yichen Tan, Zihang Lv, Ruiquan Wang, Zeyuan Huang, and Gang Chen

Abstract—Fine industrial robotic manipulation requires 6D pose, scale, and an operation-ready 3D model. Existing industrial pose estimation relies on CAD models, object-specific training data, or stable workcells, limiting use with unseen objects; zero-shot methods suffer from single-view occlusion, generative bias, and scale uncertainty. We present OV-PEAR, an observation-verified framework for 3D reconstruction and pose-scale estimation of unknown industrial objects. Its threestage object-centric pipeline: (1) segments RGB-D instances and generates object-level 3D hypotheses; (2) anchors them to the robot frame via appearance-aware view retrieval, explicit scale search, and visibility-sensitive registration; (3) back-projects multi-view RGB-D evidence into the object-canonical frame to verify voxel occupancy, suppress hallucinations, and produce reusable trusted object models. On BOP-Industrial, OV-PEAR achieves 0.657 average AR and improves pose and voxel-level geometric consistency in real scenes; real-robot connector assembly reaches 99.67% grasp and 84.00% assembly success. The model library and pose-scale log enable short-term object-level state maintenance at 0.02 ms/frame.

Index Terms—6D pose and scale estimation, zero-shot 3D reconstruction, industrial robotic manipulation.

## I. INTRODUCTION

NDUSTRIAL robotic applications typically include both mobile robotic tasks, such as inspection and environment exploration, and manipulation tasks, such as assembly, sorting, and palletizing. Compared with navigation tasks, which mainly rely on scene-level understanding and path planning, manipulation tasks require more precise object-centric perception. To generate executable grasping, insertion, or assembly motions, a robot must not only localize the target object, but also estimate its accurate 6D pose, physical scale, and contactrelevant local geometry, so that the end-effector pose and manipulator configuration can be planned accordingly [1]–[3]. This requirement is particularly critical in industrial workcells, where manipulated objects may vary across product models, be updated frequently, or lack readily available CAD models and training samples. In addition, inter-object occlusion, placement perturbations, and workcell reconfiguration are common in industrial environments, forcing robots to operate under continuously changing object categories, poses, and visibility conditions. Therefore, achieving operation-ready high-precision perception for unseen manipulation objects in variable industrial scenes is a key prerequisite for autonomous industrial robotic manipulation.

Most existing pose-perception methods for industrial robots are built on the assumption of known objects and relatively stable workcell configurations, making it difficult to simultaneously satisfy the requirements of unseen objects, scene variability, and fine manipulation accuracy. Traditional poseestimation methods typically rely on pre-built target models, controlled imaging conditions, and stable scene layouts, and localize objects through 2D feature extraction, matching, and geometric solving [4], [5]. Deep learning-based methods have substantially improved feature representation under complex conditions, but their performance often depends on large-scale annotated data collected for specific objects, workcells, and operating conditions [6], [7]. These methods can be effective in standardized production lines; however, their generalization and estimation accuracy often degrade when new objects are introduced, occlusion becomes stronger, backgrounds change, or workstations are reconfigured. For unseen manipulation objects without available templates, CAD models, or training samples, existing methods still struggle to provide highaccuracy pose estimates that can directly support fine manipulation without retraining. This motivates the need for a high-precision pose-perception method that can handle unseen objects and scene variations without scene-specific training.

Accurate pose estimation alone, however, is still insufficient for fine manipulation. For grasping, insertion, and assembly, robots also require geometric information about the target object in order to reason about contact regions, kinematic constraints, and feasible manipulation poses. This requirement is particularly important for unseen manipulation objects. Recently, zero-shot reconstruction methods that exploit priors from visual foundation models have provided a promising direction for estimating the structure of unknown objects. Representative methods such as SAM3D can leverage semantic and shape priors from pretrained models to rapidly generate complete object-level geometry from a single image or a small number of observations, showing strong generality in open environments [8]–[10]. Nevertheless, their structural predictions are largely driven by learned priors rather than continuously verified observations. Under occlusion, incomplete sensing, and cluttered industrial backgrounds, such predictions may contain geometric hallucinations or local structural deviations. Moreover, most existing methods rely on singleshot prediction or one-time fusion, and lack a mechanism for continuously incorporating multi-view evidence to revise the target model. Therefore, for unseen manipulation objects, a key challenge is how to exploit model priors for generalization while using incremental observations to suppress prior-induced errors and obtain a trusted 3D object model for industrial robotic manipulation.

The above challenges can be illustrated by a typical industrial assembly scenario. On a cluttered assembly workbench, multiple connector types, such as USB, RJ45, and HAN10 connectors, may appear simultaneously. These objects can have similar external shapes but different assembly constraints. Existing industrial model libraries or historical templates may cover only a subset of standard objects, while newly introduced parts remain unseen to the robot. At the same time, large occluders may prevent a single camera view from capturing complete pose and structural information. Under such conditions, a robot that relies only on pose estimators trained for known objects may fail to identify and localize unseen connectors accurately. Conversely, directly using a oneshot generated 3D model may introduce structural distortions due to partial occlusion and prior-induced bias, thereby reducing the reliability of downstream insertion. To execute such manipulation tasks, the robot must actively acquire multi-view RGB-D observations with an eye-in-hand camera and update both the pose estimate and the geometric model of the target object as new observations become available.

To address these issues, we propose OV-PEAR, an observation-verified zero-shot framework for pose-scale perception and 3D reconstruction of unseen industrial manipulation objects. The framework is designed to estimate both an object model and an accurate pose-scale state without requiring a target CAD model, object-specific training data, or scene-specific retraining, while also providing an interface for subsequent object-model reuse. Unlike methods that directly treat a single-view generated result as the final operation model, OV-PEAR uses the generative 3D object prior as an initial object hypothesis. It then anchors this hypothesis to the robot frame and verifies its geometry using robotacquired RGB-D observations, producing a trusted object-level perception representation for robotic manipulation.

Specifically, OV-PEAR first constructs an initial geometric hypothesis for an unseen object using a single-view generative 3D model. It then instantiates the object model in the robot coordinate frame through appearance-aware view retrieval, explicit scale search, and visibility-sensitive geometric registration. Furthermore, multi-view RGB-D evidence is used to verify and update the generated voxel occupancy, suppress hallucinated structures, and improve consistency between the reconstructed object model and real observations. Based on the verified models, OV-PEAR organizes object geometry into a reusable object model library. During subsequent continuous observations, the system updates only the pose, scale, and visibility state of each object instance, enabling compact object-level state records for short-term scene replay. We evaluate the proposed framework on public industrial benchmarks, real workbench scenes with unseen objects, fine insertion initialization tasks, and short-term continuous-frame scene replay experiments, demonstrating its effectiveness in zero-shot pose estimation, object-model refinement, robotic manipulation initialization, and reusable object-level scene representation.

The main contributions of this paper are summarized as follows:

1) We propose OV-PEAR, an observation-verified framework that converts generative 3D priors into robot-usable object models and pose-scale states for unseen industrial manipulation objects without target CAD models or object-specific training.

2) We develop a pose-scale anchoring method that integrates appearance-aware view retrieval, explicit scale search, and visibility-sensitive registration to instantiate generated object hypotheses in the robot frame.

3) We introduce an observation-verified reconstruction scheme that uses multi-view RGB-D evidence to update voxel occupancy, suppress hallucinated geometry, and improve the trustworthiness of reconstructed object models.

4) We validate OV-PEAR on public industrial benchmarks, real workbench scenes, connector insertion tasks, and continuous-frame dynamic scenes, demonstrating its effectiveness for unseen-object perception, manipulation initialization, and short-term object-level scene maintenance.

## II. RELATED WORK

## A. 6D Pose Estimation

6D pose estimation is a fundamental problem for fine industrial manipulation, aiming to recover the position and orientation of a target object under clutter, occlusion, and sensor noise. Existing methods can be broadly grouped into modelbased, data-driven, and zero-shot paradigms, each of which faces different limitations in flexible industrial deployment.

Classical model-based methods estimate object pose by matching pre-built templates or local geometric descriptors to online observations. Template-based approaches, such as LineMod [11], match the scene image against rendered views of a known 3D model, while feature-based methods, such as PFH [12] and FPFH [13], establish point-cloud correspondences and often refine the result using ICP. These methods can be accurate and efficient when the target model is available and the observation is sufficiently complete. However, their dependence on CAD models, reliable features, and limited occlusion makes them less suitable for cluttered industrial workcells containing unseen objects.

Deep learning-based methods have improved pose estimation by learning stronger visual and geometric representations. RGB methods such as PoseCNN [14], SSD-6D [15], and YOLO-6D [16] regress poses or keypoints from images, while RGB-D methods such as PVN3D [17] and G2L-Net [18] incorporate depth information to strengthen geometric constraints. Despite their strong performance on standard benchmarks, these methods typically require object-specific or scene-specific annotated data. Their accuracy can degrade when new parts are introduced, workstations are reconfigured, or background and occlusion patterns change, increasing the cost of deployment in flexible manufacturing.

Recent zero-shot pose-estimation methods, represented by SAM-6D [19], exploit priors from visual foundation models to infer poses for unseen objects without object-specific training. This direction is promising for rapid industrial changeover, but existing methods are still largely constrained by singleview observations and are often more reliable as coarse pose initializers than as high-precision estimators for fine manipulation. Under severe occlusion, scale uncertainty, and strict spatial-alignment requirements, it remains challenging to achieve both zero-shot generalization and operation-level pose accuracy. This motivates a pose-scale anchoring mechanism that can instantiate generated object hypotheses in the robot frame under partial observations.

## B. 3D Reconstruction

Fine manipulation also requires operation-ready object geometry, since grasping, insertion, and assembly depend on contact regions, local constraints, and collision geometry. Existing 3D reconstruction methods include classical multi-view reconstruction, learning-based reconstruction, and zero-shot generative reconstruction, but none of these directly provides a reliable and continuously verified object model for unseen industrial manipulation objects.

Classical SfM and MVS methods, such as global or incremental SfM [20], [21], COLMAP [22], and PMVS [23], recover 3D structure by enforcing multi-view geometric consistency. These methods can produce high-quality reconstructions when sufficient viewpoints, texture, and camera motion are available. However, their computational cost and dependence on favorable viewing conditions limit their applicability to rapid object modeling in cluttered industrial workcells, where the target may be partially occluded or observable only from constrained viewpoints.

Learning-based reconstruction methods, including MVS-Net [24], Pixel2Mesh [25], and Neuralangelo [26], improve reconstruction efficiency and surface quality by learning priors from data. Nevertheless, their performance is still tied to training distributions, and additional data collection or retraining may be required when object categories, workcell layouts, or sensing conditions change. More recently, zero-shot reconstruction methods such as VGGT [27] and SAM3D [9] have used foundation-model priors to generate 3D object geometry from limited observations, while inverse-neuralrendering work has explored generalizable and interpretable 3D tracking under distribution shifts [28]. These methods are attractive for unseen-object modeling, but their outputs are still driven by learned shape priors and may contain hallucinated structures, scale deviations, or local geometric errors under occlusion and incomplete sensing.

Overall, existing pose-estimation and reconstruction methods address complementary parts of the manipulation perception problem, but they do not fully solve the joint requirement of accurate pose-scale anchoring and trustworthy geometry construction for unseen industrial objects. What is still missing is a mechanism that uses generative 3D priors to bootstrap object modeling, anchors the generated hypothesis to the robot frame, and then verifies and corrects the geometry with multiview RGB-D evidence. OV-PEAR is designed to fill this gap through pose-scale anchoring and observation-verified voxel updating.

## III. METHODOLOGY

OV-PEAR is designed for perception of unknown industrial manipulation objects without a target CAD model, objectspecific training data, or scene-specific retraining. Given RGB-D observations, the goal is to obtain an observation-verified 3D object model, a pose-scale state in the robot coordinate frame, and an object-level representation that can be reused in subsequent manipulation and short-term state maintenance. The method consists of three stages. First, a pretrained singleview voxel generator produces initial geometric hypotheses for unknown object instances. Second, pose-scale anchoring instantiates each generated object model in the robot frame. Third, multi-view RGB-D evidence verifies and corrects voxel occupancy, yielding a trusted object model that is inserted into an object-model library. This section describes the objectcentric data flow, the pose-scale anchoring module, and the observation-verified multi-view reconstruction module.

## A. OV-PEAR Architecture and Object-Centric Data Flow

As shown in Fig. 1, OV-PEAR follows an object-centric perception pipeline.

OV-PEAR starts from scene-level RGB-D observations rather than from pre-cropped object images. Let the robot acquire RGB-D observations from N camera poses,

$$
X = \{ ( I _ { i } , D _ { i } , T _ { i } ^ { c } ) \} _ { i = 1 } ^ { N } ,
$$

where $I _ { i }$ and $D _ { i }$ denote the RGB image and depth map of the ith view, respectively, and $T _ { i } ^ { c }$ denotes the extrinsic transformation from the camera coordinate frame of that view to the robot base frame or the world frame. These observations can be collected by a wrist-mounted RGB-D camera moving along a planned trajectory, or by a calibrated multi-camera system.

For each view, an instance segmentation front end first detects candidate object masks in the scene:

$$
\begin{array} { r } { B _ { i } = \mathrm { S e g } ( I _ { i } ) = \left\{ B _ { i , j } \right\} _ { j = 1 } ^ { J _ { i } } , } \end{array}
$$

where $B _ { i , j }$ is the pixel-level mask of the jth candidate instance in view i, and $J _ { i }$ is the number of candidate instances in that view. In our implementation, Seg(·) can be instantiated by SAM3 or another open-vocabulary instance segmentation model derived from the Segment Anything family [29]. Its role is to convert the full scene image into object-level candidate observations, rather than to serve as the core reconstruction module.

The system then associates candidate masks across views or consecutive frames using 2D appearance features, depthprojected positions, 3D spatial consistency, and task-level constraints. This gives the multi-view observation set of the kth target object:

$$
O _ { k } = \mathrm { A s s o c } \Big ( \{ ( B _ { i } , D _ { i } , T _ { i } ^ { c } , I _ { i } ) \} _ { i = 1 } ^ { N } \Big ) .
$$

For a valid observation of object k in view i, let its mask be $B _ { k , i }$ . The local RGB observation is

$$
I _ { k , i } = I _ { i } \odot B _ { k , i } ,
$$

and the corresponding local 3D point cloud is

![](images/592ac4758fcfb6c801a5d183679266835c97180373a9c38397080bf75e47d33a.jpg)  
Fig. 1. Timeline-centered overview of the proposed OV-PEAR framework. After key-frame RGB-D observations are extracted, each key frame first support 3D hypothesis generation and two-stage pose-scale alignment. Evidence accumulated across multiple key frames is then integrated for observation-verified voxel modeling, trusted-voxel filtering, and mesh decoding.

$$
P _ { k , i } = T _ { i } ^ { c } \Pi ^ { - 1 } ( D _ { i } , B _ { k , i } ) ,
$$

where $\Pi ^ { - 1 } ( \cdot )$ denotes back-projection from the depth map and instance mask using the camera intrinsics. The original scene-level input is thereby converted into the object-centric observation sequence

$$
X _ { k } = \{ ( I _ { k , i } , B _ { k , i } , P _ { k , i } , T _ { i } ^ { c } ) \} _ { i \in \mathbb { Z } _ { k } } ,
$$

where $\mathcal { T } _ { k }$ is the set of view indices in which object k is validly observed.

OV-PEAR does not require a target object to remain static over the entire task sequence. It only assumes rigid shape consistency during one modeling or update window, and it requires object correspondences to be established across views or frames through instance segmentation, appearance matching, or state association. If an object translates or rotates, the estimated pose-scale state of each frame is used to compensate local RGB-D evidence into a common object-canonical coordinate frame. Multi-view geometric fusion therefore relies on object-level pose compensation, not on a globally static scene assumption. Non-rigid deformation and topological changes are outside the scope of this work.

For each valid view of object k, the single-view voxel generation module G(·) produces a voxel hypothesis from the local RGB/mask observation:

$$
V _ { k , i } = G ( I _ { k , i } , B _ { k , i } ) , \qquad i \in \mathcal { T } _ { k } .
$$

This voxel hypothesis is treated only as an initial geometric candidate for an unknown object. It is not used directly as the final operation model, since single-view generation can be affected by occlusion, segmentation errors, degenerate viewpoints, generative-prior bias, and scale ambiguity. OV-PEAR therefore instantiates and corrects the hypothesis through posescale anchoring and multi-view observation verification.

The system selects a reference view $r \in \mathcal { T } _ { k }$ from the valid observation set and decodes its voxel representation $V _ { k , \ast }$ <sub>r</sub> into a reference object model:

$$
M _ { k , r } = D ( V _ { k , r } ) .
$$

The reference model $M _ { k , r }$ is not the final optimized model. It is used to build candidate view templates, extract appearance and geometric features, and provide initial object geometry for pose-scale anchoring. Based on this reference model, the system constructs a candidate template or feature library:

$$
\mathcal { T } _ { k } = \mathrm { T e m p } ( M _ { k , r } ) ,
$$

where $\mathcal { T } _ { k }$ denotes the set of templates rendered or extracted from the reference model. For each observed view, template matching or appearance-feature retrieval gives an initial view hypothesis:

$$
q _ { k , i } ^ { 0 } = \mathrm { M a t c h } ( I _ { k , i } , B _ { k , i } , \mathcal { T } _ { k } ) .
$$

This initial view hypothesis reduces the canonical-view ambiguity of the generated object model and initializes the subsequent pose-scale anchoring step.

Given the local RGB-D observation, the reference object model, the initial view hypothesis, and the camera extrinsic parameters, the pose-scale anchoring module estimates a similarity transformation for object k in view i:

$$
\Theta _ { k , i } = ( R _ { k , i } , t _ { k , i } , s _ { k , i } ) = A ( I _ { k , i } , B _ { k , i } , P _ { k , i } , M _ { k , r } , q _ { k , i } ^ { 0 } , T _ { i } ^ { c } ) ,
$$

where $R _ { k , i } \in S O ( 3 )$ is the rotation matrix, $t _ { k , i } \in \mathbb { R } ^ { 3 }$ is the translation vector, and $s _ { k , i } \in \mathbb { R } _ { + }$ is the scale factor. To avoid coordinate ambiguity, $\Theta _ { k , i }$ is defined as the pose-scale transformation from the object-canonical coordinate frame of object k to the robot base frame or the world frame. The internal mechanism of $A ( \cdot )$ is described in the next subsection.

After obtaining $\Theta _ { k , i }$ , OV-PEAR maps generated voxel hypotheses and real depth evidence into the same objectcanonical coordinate frame. For the generated voxels, the aligned voxel hypothesis is

$$
\begin{array} { r } { \bar { V } _ { k , i } = W _ { \mathrm { v o x } } ( V _ { k , i } ; \Theta _ { k , i } ) , } \end{array}
$$

where $W _ { \mathrm { v o x } } ( \cdot )$ denotes the voxel-space transformation that aligns the generated voxel hypothesis of view i to the objectcanonical frame. For the real depth point cloud and camera center, the inverse transformation $\Theta _ { k , i } ^ { - 1 }$ compensates the evidence from the robot base frame or world frame to the objectcanonical frame:

$$
\bar { P } _ { k , i } = W _ { \mathrm { d e p } } ( P _ { k , i } ; \Theta _ { k , i } ^ { - 1 } ) , \qquad \bar { o } _ { k , i } = W _ { \mathrm { d e p } } ( o _ { i } ; \Theta _ { k , i } ^ { - 1 } ) ,
$$

where $o _ { i }$ is the camera center of view i. After this step, local geometric evidence collected from different camera poses, different observation times, and even rigid object motion is expressed in the object coordinate frame.

In the object-canonical coordinate frame, OV-PEAR first aggregates multi-view voxel hypotheses into a candidate set:

$$
V _ { k , \mathrm { c a n d } } = \bigcup _ { i \in \mathcal { T } _ { k } } \bar { V } _ { k , i } .
$$

This union-first candidate set preserves structural recall from multi-view generated hypotheses and prevents an early, irreversible deletion caused by a locally incomplete lowquality view. The system then verifies and carves the candidate voxels using real RGB-D depth evidence, camera centers, visibility relations, and multi-view support:

$$
\begin{array} { r } { V _ { k , \mathrm { t r u s t } } = U _ { \mathrm { c a r v e } } \Big ( V _ { k , \mathrm { c a n d } } , \big \{ ( \bar { P } _ { k , i } , \bar { o } _ { k , i } , B _ { k , i } , \Theta _ { k , i } ) \big \} _ { i \in \mathcal { T } _ { k } } \Big ) . } \end{array}
$$

Here, $U _ { \mathrm { c a r v e } } ( \cdot )$ denotes the trusted-voxel selection module that combines depth-surface hits, free-space ray carving, and multi-view support. The output $V _ { k , \mathrm { t r u s t } }$ is the set of voxels verified by real geometric evidence. The detailed design of this module is given in the observation-verification subsection.

Finally, the system generates an optimized object mesh by combining the trusted voxels, the reference RGB image, and the reference mask:

$$
M _ { k } ^ { * } = D _ { \mathrm { m e s h } } ( V _ { k , \mathrm { t r u s t } } , I _ { k , r } , B _ { k , r } ) .
$$

Here, $V _ { k , \mathrm { t r u s t } }$ provides trusted geometric support in the object-canonical coordinate frame, while $I _ { k , r }$ and $B _ { k , r }$ provide appearance, boundary, and color information. The optimized object model $M _ { k } ^ { * }$ is inserted into the object-model library:

$$
L \gets L \cup \{ M _ { k } ^ { * } \} .
$$

The final object model in OV-PEAR is therefore neither a direct reuse of a single-view generated result nor a simple extraction of a coarse voxel surface. It is constrained jointly by generative candidate geometry, pose-scale anchoring, real depth carving, and RGB/mask appearance information.

Once the model library has been established, subsequent frames do not require repeated full object-model construction for previously modeled objects. For the tth input frame, the system obtains current instance masks and uses librarytemplate matching and state association to determine the correspondence between each object instance and the model library:

$$
( \ell _ { k , t } , \Theta _ { k , t } ) = \mathrm { L i b M a t c h } ( I _ { t } , D _ { t } , B _ { k , t } ^ { \mathrm { o b s } } , L ) ,
$$

where $\ell _ { k , t }$ is the model-library index of object instance $k , \ \Theta _ { k , t }$ is the pose-scale state estimated for the current frame, and $B _ { k , t } ^ { \mathrm { o b s } }$ is the observed instance mask. In this paper, LibMatch(·) is used as the system interface for object-model reuse. Its effectiveness is evaluated through the short-term continuous-frame state maintenance experiments, while the core modeling and anchoring mechanisms are detailed below.

The system further maintains an object-level state record:

$$
S _ { t } = \{ ( \mathrm { i d } _ { k , t } , \ell _ { k , t } , T _ { k , t } , s _ { k , t } , c _ { k , t } , v _ { k , t } ) \} _ { k = 1 } ^ { N _ { t } } ,
$$

where $\mathrm { i d } _ { k , t }$ is the object-instance identifier, $\ell _ { k , t }$ is the model-library index, $T _ { k , t }$ and $s _ { k , t }$ are the pose and scale of the object in frame ${ t , \ c _ { k , t } }$ is the state confidence, and $v _ { k , t }$ is the visibility state. This record decouples object geometry from temporal state. Short-term state maintenance can therefore reuse the shared object model in the library and update only the pose, scale, identity, and visibility of each instance, without regenerating a full 3D model at every frame.

Overall, the object-centric data flow of OV-PEAR converts scene-level RGB-D observations into an object-level model library and state log. Instance segmentation and object association create object-level observation units. The singleview voxel generator provides initial geometric candidates for unknown objects. Template matching and pose-scale anchoring instantiate each candidate in the robot frame. Multi-view observation verification converts generative candidates into trusted 3D object models. The model library and state log provide a unified interface for grasp initialization, collision checking, model reuse, and short-term continuous-frame state maintenance.

## B. Pose-Scale Anchoring Without Target CAD Models

In the object-centric pipeline above, the single-view voxel generator provides only candidate geometry. Before this geometry can be used for multi-view fusion and robotic manipulation, its pose and scale relative to the real observed object must be estimated. This differs from conventional pose estimation with known CAD models. Here, the initial object model is generated by a zero-shot model and has three coupled uncertainties with respect to the real target: its canonical orientation may not match the observed orientation, its physical scale is unconstrained, and the RGB-D point cloud usually covers only the visible surface from the current view. OV-PEAR addresses these uncertainties with a two-stage pose-scale anchoring method. It first retrieves an appearanceaware initial rotation from rendered templates, and then jointly refines rotation, translation, and scale through explicit scale search and visibility-sensitive local registration.

For object k observed in view i with object-centric observation $( I _ { k , i } , B _ { k , i } , P _ { k , i } , T _ { i } ^ { c } )$ , the anchoring module outputs the similarity transformation

$$
\Theta _ { \boldsymbol { k } , i } = ( R _ { k , i } , t _ { k , i } , s _ { k , i } ) ,
$$

where $R _ { k , i } ~ \in ~ S O ( 3 ) , ~ t _ { k , i } ~ \in ~ \mathbb { R } ^ { 3 }$ , and $s _ { k , i } ~ \in ~ \mathbb { R } _ { + }$ . This transformation maps the object-canonical coordinate frame to the robot base frame or the world frame, and is used for objectcanonical alignment, trusted-voxel fusion, and model-library state maintenance.

1) Retrieval-Based Coarse Initialization: The coarseinitialization stage does not aim to output the final pose. Its purpose is to provide a rotation initialization from which local geometric registration can converge. Since the reference model $M _ { k , r }$ is generated from a single view, its canonical orientation may not be consistent with the orientation of the real object in the observation. Directly applying ICP from an arbitrary initial rotation can therefore lead to an erroneous local alignment. OV-PEAR instead builds a multi-view template feature library from the reference model and converts coarse pose search in continuous rotation space into appearance retrieval over a finite set of candidate views.

Fibonacci sphere sampling is first used to generate $K$ approximately uniform candidate viewpoints on the unit sphere:

$$
Q = \left\{ q _ { l } \right\} _ { l = 1 } ^ { K } .
$$

For each candidate viewpoint $q _ { l } ,$ the system renders the reference model $M _ { k , r }$ <sub>r</sub> to obtain a synthetic view image $I _ { l } ^ { \mathrm { s y n } }$ and extracts a global appearance feature using the visual encoder Φ(·):

$$
f _ { l } = \Phi ( I _ { l } ^ { \mathrm { s y n } } ) , \qquad l = 1 , \ldots , K .
$$

This produces the template feature library

$$
\mathcal { T } _ { k } = \{ ( f _ { l } , R _ { l } ) \} _ { l = 1 } ^ { K } ,
$$

where $R _ { l }$ is the initial rotation matrix corresponding to candidate viewpoint $q _ { l }$

For the local image $I _ { k , i }$ and mask $B _ { k , i }$ of object k in the real observation view $i ,$ the observed feature is

$$
f _ { k , i } = \Phi ( I _ { k , i } \odot B _ { k , i } ) .
$$

The cosine similarity between the observed feature and each template feature is then computed as

$$
\mathrm { S i m } ( f _ { k , i } , f _ { l } ) = { \frac { f _ { k , i } ^ { \top } f _ { l } } { \| f _ { k , i } \| _ { 2 } \| f _ { l } \| _ { 2 } } } .
$$

The template with the highest similarity determines the initial rotation:

$$
l ^ { * } = \arg \operatorname* { m a x } _ { l } \mathrm { S i m } ( f _ { k , i } , f _ { l } ) , \qquad R _ { k , i } ^ { 0 } = R _ { l ^ { * } } .
$$

This step turns the canonical-view ambiguity of the generated model into a finite template retrieval problem. The subsequent local registration therefore starts from a plausible rotation neighborhood, rather than from an unconstrained search over a large rotation space.

For translation initialization, OV-PEAR uses the target local point cloud associated with the instance mask, $P _ { k , i } =$ $\{ p _ { j } \} _ { j = 1 } ^ { N _ { i } }$ . Its geometric centroid is used as the initial translation:

$$
t _ { k , i } ^ { 0 } = \frac { 1 } { N _ { i } } \sum _ { j = 1 } ^ { N _ { i } } p _ { j } .
$$

This gives the initial pose $( R _ { k , i } ^ { 0 } , t _ { k , i } ^ { 0 } )$ for local registration.

2) Pose-Scale Anchoring With Template Retrieval and Visibility-Sensitive Registration: After obtaining the initial rotation and translation, OV-PEAR performs scale-adaptive local registration. This stage addresses two issues. First, the single-view generated model has no true metric scale, so the scale factor must be explicitly estimated. Second, the observed point cloud $P _ { k , i }$ contains only the visible surface under the current viewpoint, whereas the initial model $M _ { k , r }$ is a complete-object hypothesis. Direct standard ICP would incorrectly force invisible model regions to participate in nearest-neighbor matching.

The system first samples a source point cloud from the surface of the reference model $M _ { k , r } \mathrm { : }$

$$
P _ { s } = \{ ( x _ { m } , n _ { m } ) \} _ { m = 1 } ^ { M } ,
$$

where $x _ { m } \in \mathbb { R } ^ { 3 }$ is a sampled model point, $n _ { m }$ is its outward unit normal, and M is the number of sampled points. To

compensate for scale differences between the generated model and the real object, OV-PEAR constructs a candidate scale set over $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ]$ with step size $\Delta s \mathrm { : }$

$$
\begin{array} { r } { S = \left\{ s _ { q } ~ \vert ~ s _ { q } = s _ { \operatorname* { m i n } } + q \Delta s , ~ q = 0 , 1 , \ldots , K _ { s } , ~ s _ { q } \leq s _ { \operatorname* { m a x } } \right\} . } \end{array}
$$

The scale search range is determined by the typical scale error of the generative model on the development set and by the tolerance of the target operation. If the characteristic size of the target object is D and the allowable scale error is $\boldsymbol { e } _ { s } ,$ the scale step should satisfy

$$
\Delta s \cdot D \leq e _ { s } ,
$$

so that the geometric difference between neighboring scale candidates does not exceed the manipulation tolerance.

To reduce interference from invisible model surfaces, OV-PEAR recomputes the visibility weight during local registration instead of fixing it at the initial pose. The posescale state is estimated through joint pose-scale anchoring: scale is searched over a finite candidate set, and for each scale candidate, rotation and translation are refined by a finite visibility-weighted ICP process. For each $s _ { q } \in \mathcal S$ , visibilityweighted ICP is initialized with $( R _ { k , i } ^ { 0 } , t _ { k , i } ^ { 0 } )$ . At iteration $\ell ,$ each source point and normal are transformed as

$$
p _ { m } ^ { ( \ell ) } = s _ { q } R ^ { ( \ell ) } x _ { m } + t ^ { ( \ell ) } , \qquad n _ { m } ^ { ( \ell ) } = R ^ { ( \ell ) } n _ { m } .
$$

Let $o _ { i }$ be the camera center of view i. The viewing direction and visibility weight are

$$
v _ { m } ^ { ( \ell ) } = \frac { o _ { i } - p _ { m } ^ { ( \ell ) } } { \left\| o _ { i } - p _ { m } ^ { ( \ell ) } \right\| _ { 2 } } , \qquad w _ { m } ^ { ( \ell ) } = \operatorname* { m a x } \Bigl ( 0 , ( n _ { m } ^ { ( \ell ) } ) ^ { \top } v _ { m } ^ { ( \ell ) } \Bigr ) .
$$

For $p _ { m } ^ { ( \ell ) }$ , the nearest neighbor $y _ { m } ^ { ( \ell ) }$ is searched in $P _ { k , i } .$ , and pairs with $\| y _ { m } ^ { ( \ell ) } - p _ { m } ^ { ( \ell ) } \| _ { 2 } \geq \tau _ { \mathrm { c o r r } }$ are removed from the valid set $\mathcal { C } ^ { ( \ell ) }$ . The pose update for the current scale candidate minimizes

$$
E ^ { ( \ell ) } ( R , t \mid s _ { q } ) = \sum _ { m \in \mathcal { C } ^ { ( \ell ) } } w _ { m } ^ { ( \ell ) } \cdot \bigg \| y _ { m } ^ { ( \ell ) } - \big ( s _ { q } R x _ { m } + t \big ) \bigg \| _ { 2 } ^ { 2 } .
$$

The ICP refinement stops when the fixed convergence criteria of the ICP implementation are satisfied or when the maximum iteration number is reached. The locally refined pose returned for scale $s _ { q }$ is denoted as $( R _ { k , i } ^ { ( s _ { q } ) } , t _ { k , i } ^ { ( s _ { q } ) } )$ . Because this refinement is repeated for all $s _ { q } ,$ the final anchoring result depends jointly on scale, rotation, and translation rather than on three independent estimates.

After ICP convergence, the registration quality of each scale candidate is evaluated using a visible-region fitness score. Let $\mathcal { M } _ { v } ^ { ( s _ { q } ) }$ be the set of final model points that have positive visibility weights and valid correspondences after refinement, $N _ { v } ^ { ( s _ { q } ) } = | \mathcal { M } _ { v } ^ { ( \tilde { s } _ { q } ) } |$ , and $r _ { m } ^ { ( s _ { q } ) }$ be the final residual for point m. The fitness of $s _ { q }$ is defined as

$$
\mathrm { F i t } ( s _ { q } ) = \frac { 1 } { N _ { v } ^ { ( s _ { q } ) } } \sum _ { m \in \mathcal { M } _ { v } ^ { ( s _ { q } ) } } \mathbb { I } \Bigl ( r _ { m } ^ { ( s _ { q } ) } < \tau _ { \mathrm { f i t } } \Bigr ) ,
$$

where $\tau _ { \mathrm { { f i t } } }$ is used only for scale selection and is distinct from the correspondence gating threshold $\tau _ { \mathrm { c o r r } }$ and the ICP convergence thresholds.

The scale with the highest fitness is selected as the optimal scale estimate:

$$
s _ { k , i } ^ { * } = \arg \operatorname* { m a x } _ { { s _ { q } \in { \mathcal { S } } } } \mathrm { F i t } ( s _ { q } ) ,
$$

and the corresponding rotation and translation are used as the final anchoring result:

$$
R _ { k , i } = R _ { k , i } ^ { ( s _ { k , i } ^ { * } ) } , \qquad t _ { k , i } = t _ { k , i } ^ { ( s _ { k , i } ^ { * } ) } , \qquad s _ { k , i } = s _ { k , i } ^ { * } .
$$

The pose-scale state of object k in observation view i is therefore

$$
\Theta _ { \boldsymbol { k } , i } = ( R _ { k , i } , t _ { k , i } , s _ { k , i } ) .
$$

This two-stage design decomposes CAD-free pose-scale estimation into appearance retrieval initialization and scaleconditioned visibility-sensitive geometric registration. The first stage reduces the canonical-orientation ambiguity of the generated model and narrows the rotation search range. The second stage compensates for scale bias while suppressing the influence of invisible surfaces and hallucinated structures under partial observations. The key is not template retrieval or local ICP alone, but the integration of appearance discriminability, scale uncertainty, and partial visibility constraints into a single anchoring process. The resulting $\Theta _ { k , i }$ provides the basis for object-canonical alignment, trusted-voxel fusion, modellibrary matching, and continuous-frame state maintenance.

## C. Observation-Verified Multi-View Object Modeling

After pose-scale anchoring, OV-PEAR has obtained the transformation parameters of object k in all valid observation views:

$$
\Theta _ { k , i } = ( R _ { k , i } , t _ { k , i } , s _ { k , i } ) , \qquad i \in \mathcal { T } _ { k } .
$$

The goal of this subsection is therefore not to re-estimate object pose. Instead, OV-PEAR uses $\Theta _ { k , i }$ to express multiview generated voxel hypotheses and real RGB-D geometric evidence in the object-canonical coordinate frame, and then selects trusted object geometry. In contrast to directly using a single-view generated result, generated voxels are treated as high-recall but unverified candidate geometry. In contrast to intersection-based or purely subtractive fusion, OV-PEAR does not delete inconsistent regions at an early stage. It follows a union-first then carve strategy: it first preserves potentially correct structures from multi-view generated hypotheses, and then verifies and carves them using real depth hits, freespace ray carving [30], and multi-view support. This design suppresses hallucinated structures in the generated 3D model while avoiding irreversible geometric deletion caused by one low-quality view, occlusion, or mask incompleteness.

1) Union-First Candidate Voxel Aggregation: Let the voxel hypothesis generated for object k in valid view i be

$$
V _ { k , i } = \{ ( x _ { k , i , j } , \rho _ { k , i , j } ) \} _ { j = 1 } ^ { N _ { k , i } ^ { v } } ,
$$

where $x _ { k , i , j } \in \mathbb { R } ^ { 3 }$ is the voxel-center coordinate, $\rho _ { k , i , j } \in$ [0, 1] is the corresponding occupancy response, and $N _ { k , i } ^ { v }$ is the number of voxels in that view.

The generated voxels $V _ { k , \ast }$ <sub>i</sub> and the real depth point cloud $P _ { k , i }$ originate from different coordinate systems. We therefore use $W _ { \mathrm { v o x } }$ and $W _ { \mathrm { d e p } }$ to denote generated-voxel normalization and depth-evidence inverse transformation, respectively. Both are associated with the same pose-scale anchoring result $\Theta _ { k , i }$ , but their mapping directions are determined by their original coordinate systems. Specifically, $\Theta _ { k , i }$ maps the object-canonical coordinate frame to the robot base frame or world frame. Since the real depth point cloud $P _ { k , i }$ is already expressed in the robot base frame or world frame through the camera extrinsic parameters, it is mapped back to the object-canonical frame using $\Theta _ { k , i } ^ { - 1 }$ . The generated voxels $V _ { k , i }$ originate from the candidate voxel space of the singleview generator, and $W _ { \mathrm { v o x } }$ normalizes and aligns the voxel hypothesis of that view to the shared object-canonical frame:

$$
\begin{array} { r } { \bar { V } _ { k , i } = W _ { \mathrm { v o x } } ( V _ { k , i } ; \Theta _ { k , i } ) . } \end{array}
$$

At the same time, the real depth point cloud and camera center are compensated into the same object coordinate frame:

$$
\bar { P } _ { k , i } = W _ { \mathrm { d e p } } ( P _ { k , i } ; \Theta _ { k , i } ^ { - 1 } ) , \qquad \bar { o } _ { k , i } = W _ { \mathrm { d e p } } ( o _ { i } ; \Theta _ { k , i } ^ { - 1 } ) .
$$

Here, $o _ { i }$ is the camera center of view i. After this transformation, local geometric evidence collected from different camera positions, different observation times, or even rigid target motion is represented in the object’s own coordinate frame.

Let the regular voxel grid in the object-canonical coordinate frame be

$$
\Omega _ { k } = \left\{ v _ { n } \right\} _ { n = 1 } ^ { N _ { g } } ,
$$

where $v _ { n }$ denotes the nth voxel cell. The voxel resampling operator $\Gamma ( \cdot )$ projects the aligned generated voxel hypothesis onto this regular grid:

$$
\pi _ { k , i } ( v _ { n } ) = \Gamma ( v _ { n } , \bar { V } _ { k , i } ) , \qquad \pi _ { k , i } ( v _ { n } ) \in [ 0 , 1 ] .
$$

Here, $\pi _ { k , i } ( v _ { n } )$ denotes the occupancy support of the generated voxel hypothesis from view i for grid voxel $v _ { n }$

OV-PEAR first constructs a candidate voxel union:

$$
V _ { k , \mathrm { c a n d } } = \left\{ v _ { n } \in \Omega _ { k } \vert \exists i \in \mathcal { I } _ { k } , \pi _ { k , i } ( v _ { n } ) > 0 \right\} .
$$

This union-first design avoids the false-negative sensitivity of intersection-based fusion. If one view produces an undersized voxel hypothesis due to occlusion, mask fragmentation, scale bias, or a single-view generation failure, intersectionbased fusion would delete structures that are correctly present in other views at an early stage. Union-first fusion instead retains candidate geometry proposed by all valid views and leaves over-expanded regions to be carved by real RGB-D evidence, preserving high structural recall.

The candidate set $V _ { k , \mathrm { c a n d } }$ is not used directly as the final object geometry. Union-first aggregation increases structural recall, but it can also introduce more potential false-positive voxels. OV-PEAR therefore treats it only as a set of candidates to be verified. Subsequent confidence updating requires each candidate voxel to obtain depth hits, multi-view generative support, or at least no conflict with visible free space, thereby preventing single-view hallucinations from being directly written into the final model.

2) Initial Confidence Fusion Based on Visible Geometric Coverage: The candidate voxel set $V _ { k , \mathrm { c a n d } }$ captures structural recall from multi-view generated hypotheses, but it is not yet trusted geometry. Different views cover different visible parts of the object. If all views are accumulated with equal weight, a severely occluded view or a view containing only a small visible surface may exert excessive influence. OV-PEAR therefore assigns an initial fusion weight to each view according to the visible geometric coverage of its real observed point cloud. This weight is only an approximate measure of geometric coverage, not a complete evaluation of observation quality or sensor reliability. The actual occupancy confidence is further corrected by depth-hit and free-space evidence.

For the object-canonical point cloud of view i,

$$
{ { \bar { P } } _ { k , i } } = \left\{ { { \bar { p } } _ { k , i , m } } \right\} _ { m = 1 } ^ { { N _ { i } } } ,
$$

the point-cloud centroid is

$$
\mu _ { k , i } = \frac { 1 } { N _ { i } } \sum _ { m = 1 } ^ { N _ { i } } \bar { p } _ { k , i , m } .
$$

The covariance matrix is

$$
C _ { k , i } = \frac { 1 } { N _ { i } } \sum _ { m = 1 } ^ { N _ { i } } ( \bar { p } _ { k , i , m } - \mu _ { k , i } ) ( \bar { p } _ { k , i , m } - \mu _ { k , i } ) ^ { \top } .
$$

Eigenvalue decomposition of $C _ { k , i }$ gives three orthogonal principal directions $e _ { k , i , j }$ . Projecting the point cloud onto each principal direction gives the coverage range

$$
\begin{array} { r l } & { \Delta l _ { k , i , j } = \underset { m } { \operatorname* { m a x } } ( e _ { k , i , j } ^ { \top } \bar { p } _ { k , i , m } ) - \underset { m } { \operatorname* { m i n } } ( e _ { k , i , j } ^ { \top } \bar { p } _ { k , i , m } ) , } \\ & { \qquad j = 1 , 2 , 3 . } \end{array}
$$

The visible coverage of view i is then

$$
S _ { k , i } = \prod _ { j = 1 } ^ { 3 } \operatorname* { m a x } ( \Delta l _ { k , i , j } , \epsilon ) ,
$$

where ϵ is a lower-bound threshold that prevents degeneracy when the point cloud has a very small range along one direction. The corresponding view-fusion weight is

$$
\alpha _ { k , i } = \frac { S _ { k , i } } { \sum _ { q \in \mathcal { T } _ { k } } { S _ { k , q } } } .
$$

The initial confidence of a candidate voxel is defined as

$$
C _ { 0 } ( v _ { n } ) = \sum _ { i \in \mathcal { Z } _ { k } } \alpha _ { k , i } \pi _ { k , i } ( v _ { n } ) , \qquad v _ { n } \in V _ { k , \mathrm { c a n d } } .
$$

In this definition, multi-view support is not introduced as an additional discrete rule. It is accumulated in $C _ { 0 } ( v _ { n } )$ through the occupancy responses $\pi _ { k , i } ( v _ { n } )$ and the coverage weights $\alpha _ { k , i }$ of different views. Voxels jointly predicted by multiple valid views receive higher initial confidence, whereas responses accidentally produced by a single low-coverage view are more likely to be suppressed by later thresholding or freespace carving. Thus, $C _ { 0 } ( v _ { n } )$ represents multi-view support at the generative-prior level, but it still requires verification by real RGB-D geometric evidence.

3) Trusted Voxel Selection Using Depth Hits and Ray Carving: To improve the efficiency of nearest-neighbor queries between voxel centers and real depth surfaces, OV-PEAR builds a KD-tree for each object-canonical point cloud $\bar { P } _ { k , i }$ and uses it to determine whether a candidate voxel is supported by the visible depth surface. Free-space judgment, however, does not rely only on KD-tree nearest-neighbor distance. It is determined by depth ordering. For a candidate voxel $v _ { n } ,$ the system projects its center $c _ { n }$ into the depth map of view i through the current object transformation and camera extrinsic parameters, obtaining the corresponding pixel location and projected voxel depth. If the voxel lies in front of the observed depth surface and no local real-surface hit is found in its neighborhood, it is classified as free space. If it agrees with the observed depth surface within $\epsilon _ { h }$ , it is classified as a hit. If it lies behind the observed surface or the current view provides no valid depth, it is classified as occluded or unknown.

Formally, for $v _ { n } \in V _ { k , \mathrm { c a n d } }$ , let $c _ { n }$ be its center in the objectcanonical coordinate frame. In view $i ,$ it is transformed to the camera coordinate frame and projected as

$$
\begin{array} { r } { \left( u _ { k , i , n } , z _ { k , i , n } \right) = \Pi _ { i } \big ( ( T _ { i } ^ { c } ) ^ { - 1 } \Theta _ { k , i } \tilde { c } _ { n } \big ) , } \end{array}
$$

where ${ \tilde { c } } _ { n }$ is the homogeneous coordinate of $c _ { n } , u _ { k , i , n }$ is the projected pixel location, $z _ { k , i , n }$ is the projected depth of the voxel center in that view, and $\Pi _ { i } ( \cdot )$ is the camera projection model. Let the observed depth at that pixel be $D _ { i } ( u _ { k , i , n } )$ . The evidence state is defined as follows:

$\mathrm { i f ~ } | z _ { k , i , n } - D _ { i } ( u _ { k , i , n } ) | < \epsilon _ { h }$ , then $v _ { n }$ is classified as a depth hit;

if $z _ { k , i , n } < D _ { i } ( u _ { k , i , n } ) - \epsilon _ { h }$ and the KD-tree query does not find a local real surface, then $v _ { n }$ is classified as free space;

if $z _ { k , i , n } > D _ { i } ( u _ { k , i , n } ) + \epsilon _ { h }$ , or if the pixel depth is invalid, then $v _ { n }$ is treated as occluded or unknown.

The threshold $\epsilon _ { h }$ is set at a scale consistent with voxel side length, depth noise, and calibration error, so that isolated noise or boundary quantization does not cause erroneous hit decisions.

To avoid order dependence from sequential per-view updates, OV-PEAR first counts the geometric evidence obtained by each candidate voxel across all views:

$$
\begin{array} { l } { h _ { n } = \displaystyle \sum _ { i \in \mathcal { Z } _ { k } } \mathbb { I } ( v _ { n } \mathrm { ~ i s ~ h i t ~ i n ~ v i e w ~ } i ) , } \\ { f _ { n } = \displaystyle \sum _ { i \in \mathcal { Z } _ { k } } \mathbb { I } ( v _ { n } \mathrm { ~ i s ~ f r e e ~ s p a c e ~ i n ~ v i e w ~ } i ) , } \\ { g _ { n } = \displaystyle \sum _ { i \in \mathcal { Z } _ { k } } \mathbb { I } ( \pi _ { k , i } ( v _ { n } ) > 0 ) . } \end{array}
$$

Here, $h _ { n }$ and $f _ { n }$ denote the number of views that classify the voxel as a hit and as free space, respectively. The value $g _ { n }$ denotes the number of generated-view hypotheses that support the voxel.

Based on this evidence, the updated voxel confidence is defined as

$$
C ( v _ { n } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } ( C _ { 0 } ( v _ { n } ) , p _ { \mathrm { h i t } } ) , } & { h _ { n } > 0 , } \\ { \operatorname* { m i n } ( C _ { 0 } ( v _ { n } ) , p _ { \mathrm { f r e e } } ) , } & { h _ { n } = 0 \mathrm { ~ a n d ~ } f _ { n } > 0 , } \\ { C _ { 0 } ( v _ { n } ) , } & { h _ { n } = 0 \mathrm { ~ a n d ~ } f _ { n } = 0 . } \end{array} \right.\tag{1}
$$

Here, $p _ { \mathrm { h i t } } ~ = ~ 0 . 9 5$ is the high-confidence lower bound for depth-hit voxels, and $p _ { \mathrm { f r e e } } = 0 . 0 5$ is the low-confidence upper bound for free-space voxels. They can be interpreted as bounded probabilistic relaxations of hard 1/0 decisions. In controlled industrial RGB-D observations, hits and free-space evidence provide strong support and strong contradiction, respectively, while still preserving uncertainty caused by depth noise, segmentation error, and hand-eye calibration error.

When the same voxel receives both hit and free-space evidence, OV-PEAR gives priority to hit evidence. A depthsurface hit provides direct local occupancy support, whereas free-space classification is more sensitive to registration error, depth-boundary quantization, mask-boundary error, and hand-eye calibration residuals. This hit-prioritized update is a conservative strategy that reduces the risk of over-carving real thin structures, local boundaries, or small protrusions. For candidate voxels that lack hit support and are stably classified as conflicting with observed free space, p<sub>free</sub> suppresses their confidence.

The final trusted voxel set is

$$
\begin{array} { c } { { V _ { k , \mathrm { t r u s t } } = \left\{ v _ { n } \in V _ { k , \mathrm { c a n d } } \vert C ( v _ { n } ) \geq \tau _ { c } , \right. } } \\ { { \left. h _ { n } \geq \tau _ { h } \ \mathrm { o r } \ ( h _ { n } = 0 \ \mathrm { a n d } \ g _ { n } \geq \tau _ { s } ) \right\} . } } \end{array}
$$

Here, $\tau _ { c }$ is the global confidence threshold, $\tau _ { h }$ is the minimum number of real depth hits, and $\tau _ { s }$ is the minimum multi-view generative support required when no depth hit is available. A voxel predicted only accidentally by a single-view prior and lacking depth-hit or multi-view support is unlikely to pass the threshold. In contrast, real object structures usually receive depth hits or consistent cross-view support and are therefore retained.

A voxel that is not directly hit by the current view is not deleted immediately. If it receives multi-view generative support and does not conflict with visible free space, it can be retained as an occluded or incompletely observed structure. Conversely, voxels that clearly conflict with visible depth and lack hit or multi-view support are suppressed.

OV-PEAR therefore implements a union-first then carve observation-verification mechanism. Union-first aggregation maintains structural recall from generated candidate geometry. Depth hits provide positive surface evidence. Free-space ray carving provides negative spatial evidence. Occluded or unknown regions are not forcibly removed by a single view. This mechanism allows the final voxel set to use generative priors to complement unobserved structures while suppressing hallucinated regions and local errors using real RGB-D observations.

4) Trusted-Voxel and RGB/Mask-Guided Mesh Decoding: After obtaining $V _ { k , \mathrm { t r u s t } }$ , OV-PEAR combines the reference RGB/mask information to generate the final object mesh:

$$
M _ { k } ^ { * } = D _ { \mathrm { m e s h } } ( V _ { k , \mathrm { t r u s t } } , I _ { k , r } , B _ { k , r } ) .
$$

Here, $V _ { k , \mathrm { t r u s t } }$ provides trusted geometric support in the object-canonical coordinate frame, $I _ { k , r }$ and $B _ { k , r }$ provide appearance, boundary, and color information, and $D _ { \mathrm { m e s h } } ( \cdot )$ denotes a pretrained mesh decoder that takes voxel occupancy and RGB/mask appearance as input to produce a colored object mesh. The decoder itself is not claimed as the main methodological contribution. Instead, OV-PEAR replaces its geometric input from a single-view voxel prediction with $V _ { k , \mathrm { t r u s t } }$ , which has been verified by multi-view RGB-D evidence, thereby constraining the occupiable region, scale range, and structural boundary of the final model.

Compared with directly decoding any single-view generated voxel grid, mesh generation based on $V _ { k , \mathrm { t r u s t } }$ reduces local structural errors caused by single-view occlusion and generative-prior bias. Compared with extracting a mesh only from the voxel surface, incorporating RGB/mask information helps produce a more continuous object surface and a more stable appearance representation.

The final object model $M _ { k } ^ { * }$ is inserted into the object-model library:

$$
L \gets L \cup \{ M _ { k } ^ { * } \} .
$$

This model replaces the original single-view generated model as the object-level geometric template for subsequent pose estimation, model-library matching, grasp initialization, and collision checking. OV-PEAR thereby unifies single-view generative candidates, multi-view pose-scale anchoring, and real RGB-D observation verification, completing the conversion from a generative object hypothesis to an observationverified 3D object model. After insertion into the library, $M _ { k } ^ { * }$ also serves as the shared geometric template for continuousframe state maintenance and object-level scene replay; its reuse and compression effects are evaluated in the experiments.

## IV. EXPERIMENT

This section evaluates OV-PEAR along four axes: poseestimation accuracy, object-model quality, robotic manipulation initialization, and short-term object-state maintenance. The experiments are organized as public industrial benchmarking, real workbench perception, object-model reuse, connector assembly, and ablation studies. Implementation parameters are fixed across objects and scenes; the full parameter list, runtime accounting, USB tolerance calibration, and auxiliary visualizations are provided in Appendices A–C.

## A. Experimental Setup

The robotic platform consists of a Franka Panda manipulator, a Robotiq gripper, and a wrist-mounted Intel RealSense RGB-D camera in an eye-in-hand configuration. During object modeling, the robot follows pre-defined observation trajectories, captures multi-view RGB-D data, and uses calibrated camera extrinsics to express all observations in a common robot coordinate frame. Unless otherwise stated, the target object is static within one multi-view modeling window. The short-term state-maintenance experiment uses a different mode: after a verified object model is inserted into the model library, subsequent frames reuse the model and update only pose, scale, confidence, identity, and visibility states.

For public industrial benchmarking, we use the images, depth maps, annotations, and evaluation scripts provided by BOP-Industrial and the BOP evaluation ecosystem [31], [32]. For the real workbench and connector experiments, the robot actively acquires multi-view RGB-D observations. The real workbench evaluation set contains 15 scenes, four calibrated views per scene, 352 scene-camera records, 2264 per-view pose/visibility annotations, and 10 object models. All experiments are executed on a workstation equipped with an NVIDIA RTX 4090D GPU.

We compare OV-PEAR with FRTPose-WAPR, FoundationPose, and SAM-6D. FRTPose-WAPR and Foundation-Pose [33] represent competitive industrial and model-based novel-object pose-estimation pipelines, while SAM-6D [19] represents recent zero-shot 6D pose estimation based on visual foundation models. On BOP-Industrial, all methods are evaluated under the same official single-view AR protocol. In real workbench scenes, object detection and segmentation are evaluated by mAP@0.5:0.95, pose accuracy by object-center error and frame-aligned rotation error, and geometry by voxel IoU. For short-term state maintenance, we evaluate objectstate error, replay quality, and representation efficiency. For connector assembly, we report grasp success, insertion success, and assembly success.

a) Metric definitions: The reported accuracy metrics follow the protocol of each experiment and are not crosscompared across protocols. In the public BOP-Industrial benchmark, we follow the official single-view BOP evaluation setting and report AR for pose estimation. In the real workbench perception experiment, mAP@0.5:0.95 evaluates instance detection and segmentation quality, while object-center error, frame-aligned rotation error, and voxel IoU evaluate pose-scale estimation and geometric modeling. In the core perception ablation, AR is reported only for the corresponding 6D detection protocol, whereas voxel IoU measures the quality of the verified object geometry.

For the real workbench benchmark, dataset-provided object masks are used to define object crops and visible object regions for pose evaluation, so that the comparison focuses on pose-scale estimation and geometric modeling rather than mask quality. In subsequent online manipulation and statemaintenance experiments, object masks are generated by SAM3. Object-center error is computed as the Euclidean distance between transformed 3D bounding-box centers in the robot/world frame, rather than as the distance between arbitrary model origins.

![](images/753b3fbe1fb44a8c55d1523ef144ea3cd87cb1a9b1bcf4219f88795b8df2d373.jpg)  
Fig. 2. Qualitative object-instantiation and pose-scale estimation results on representative BOP-Industrial objects. For each object, the input observation is shown on the left, the middle visualization shows the alignment between the generated object model and the observed point cloud, and the right visualization overlays the reconstructed object model in the scene using the estimated 6D pose and scale.

TABLE I  
POSE-ESTIMATION RESULTS ON THREE BOP INDUSTRIAL DATASETS USING AR AS THE EVALUATION METRIC
<table><tr><td>Method</td><td>Views</td><td>IPD</td><td>XYZ-IBD</td><td>ITODD-MV</td><td>Average</td></tr><tr><td>FRTPose-WAPR</td><td>1</td><td>0.481</td><td>0.516</td><td>0.688</td><td>0.562</td></tr><tr><td>FoundationPose</td><td>1</td><td>0.408</td><td>0.564</td><td>0.512</td><td>0.495</td></tr><tr><td>SAM-6D</td><td>1</td><td>0.415</td><td>0.578</td><td>0.595</td><td>0.529</td></tr><tr><td>OV-PEAR</td><td>1</td><td>0.609</td><td>0.647</td><td>0.715</td><td>0.657</td></tr></table>

Note: FRTPose-WAPR, FoundationPose, and SAM-6D were reproduced using their official GitHub repositories or public implementations and evaluated under the same BOP-Industrial single-view input setting, data subsets, and BOP official AR scripts. The corresponding commits, configuration files, and execution scripts will be listed in the reproducibility package.

Because the reconstructed OV-PEAR mesh has an objectcanonical frame that is not guaranteed to match the coordinate frame of the reference object model, rotation error is evaluated after a fixed object-level frame correction. Let C denote the reconstructed mesh frame, M the reference model frame, and $A _ { M C }$ the fixed transform from C to M, estimated once per object from the final reconstructed mesh and the reference model. For each evaluation frame, the corrected ground-truth pose of the reconstructed mesh is

$$
T _ { W C } ^ { g t } = T _ { W M } ^ { g t } A _ { M C } .
$$

The reported rotation error is computed between $T _ { W C } ^ { p r e d }$ and $T _ { W C } ^ { g t }$ . This correction is fixed at the object level and is not fitted per frame. For symmetric or near-symmetric objects, equivalent rotations are handled by taking the minimum error over the valid symmetry set when such symmetry is geometrically supported.

## B. Comparative Experiments

1) Public Industrial Benchmark: BOP-Industrial contains IPD, XYZ-IBD, and ITODD-MV subsets collected in real industrial environments with textureless, reflective, symmetric, densely stacked, and occluded objects [34], [35]. To keep this comparison aligned with the official benchmark setting, OV-PEAR is evaluated in a single-view mode and only poseestimation AR is reported. Multi-view voxel optimization, robot manipulation, and continuous-frame replay are not mixed into this comparison.

As shown in Table I, OV-PEAR obtains the highest AR on all three subsets and reaches an average AR of 0.657, improving over FRTPose-WAPR, FoundationPose, and SAM-6D by 0.095, 0.162, and 0.128, respectively. Relative to the strongest baseline on each subset, OV-PEAR improves AR by 0.128 on IPD, 0.069 on XYZ-IBD, and 0.027 on ITODD-MV, with a mean best-baseline gain of 0.075 and a sample variance of 0.00257 across the three subsets. Relative to the threebaseline mean after excluding OV-PEAR, the gains are 0.174, 0.094, and 0.117 AR on IPD, XYZ-IBD, and ITODD-MV, respectively. Because this experiment uses only the official single-view benchmark inputs, the gain cannot be attributed to multi-view voxel fusion or downstream manipulation feedback. It mainly reflects the effect of pose-scale anchoring: the generated object hypothesis is not accepted as a freeform prior, but is explicitly scaled and registered against the observed RGB-D evidence.

The subset-wise pattern is also informative. The improvement is largest on IPD and XYZ-IBD, where clutter, weak texture, and industrial object ambiguity make purely template-based or frame-wise zero-shot estimation unstable. On ITODD-MV, where FRTPose-WAPR is already competitive, OV-PEAR still improves AR but by a smaller margin. This pattern is directionally consistent across the three subsets, but the number of benchmark subsets is too small for a highpower significance test. A one-sample test on the three bestbaseline gains gives a positive trend but should be interpreted descriptively rather than as definitive inferential evidence. The public benchmark therefore supports the front-end poseestimation component, but it does not by itself evaluate the main reconstruction and reuse claims, which are tested in the following real-workbench and state-maintenance experiments.

TABLE II  
PERCEPTION PERFORMANCE IN REAL WORKBENCH SCENES WITH UNSEEN OBJECTS.
<table><tr><td>Method</td><td>mAP</td><td>Center err. (cm)</td><td>Aligned rot. (deg)</td><td>Voxel IoU</td><td>Build (s)</td><td>Pose (s)</td></tr><tr><td>FRTPose-WAPR</td><td>N/A</td><td>8.23</td><td>19.21</td><td>N/A</td><td>N/A</td><td>0.72</td></tr><tr><td>FoundationPose</td><td>N/A</td><td>12.56</td><td>26.30</td><td>N/A</td><td>N/A</td><td>4.53</td></tr><tr><td>SAM-6D</td><td>0.428</td><td>6.95</td><td>21.87</td><td>N/A</td><td>N/A</td><td>0.28</td></tr><tr><td>OV-PEAR-SV</td><td>0.415</td><td>2.88</td><td>15.93</td><td>0.613</td><td>12.61</td><td>0.53</td></tr><tr><td>OV-PEAR-4V</td><td>0.415</td><td>1.28</td><td>13.17</td><td>0.689</td><td>33.45</td><td>0.53</td></tr></table>

Note: mAP is evaluated at IoU thresholds from 0.5 to 0.95. Center error measures the transformed 3D bounding-box center error. Aligned rotation error is computed after a fixed object-level frame correction. Build time denotes one-time object-model construction cost, while pose time denotes per-query pose estimation cost. N/A indicates that the method does not natively provide the corresponding output.

![](images/600066801857c1a8646ad4bf76c88c82ae030775f8d9dcfcc31db5f3a30528b9.jpg)  
Fig. 3. Qualitative results of multi-view observation verification in real workbench scenes with unseen objects. Spool, cafe bottle, remote control, and hose are shown as representative targets. Blue voxels denote hallucinated or weakly supported regions suppressed by real geometric evidence, while red voxels denote trusted regions supported by cross-view observations.

2) Unseen-Object Perception in Real Workbench Scenes: The real workbench experiment evaluates whether OV-PEAR generalizes beyond fixed benchmark categories and layouts. Empty bottles, snack packages, glasses, and other objects are randomly placed in a cluttered container. The wrist-mounted RGB-D camera observes the scene from multiple viewpoints, and the system estimates object category, pose, scale, and geometry. Since not all baselines support segmentation, multiview input, or object-level reconstruction, unsupported outputs are marked as N/A rather than treated as execution failures.

Table II shows that OV-PEAR provides stronger overall perception performance in cluttered unseen-object scenes. In detection and segmentation, OV-PEAR and SAM-6D achieve comparable mAP, which indicates that the subsequent pose and geometry improvements are not simply caused by a stronger segmentation front end. In pose estimation, OV-PEAR-SV already reduces center error to 2.88 cm, and OV-PEAR-4V further reduces it to 1.28 cm. The four-view setting also lowers frame-aligned rotation error from 15.93 deg to 13.17 deg and increases voxel IoU from 0.613 to 0.689. These changes isolate the value of observation verification: additional views do not only add more surface points, but also suppress unsupported generative geometry and improve the consistency between the object model and the robot-frame pose.

The remaining errors clarify the boundary of the method. Frame-aligned rotation error is reduced less strongly than center error because several workbench objects have weak texture, partial symmetry, or incomplete side observations. In such cases, multi-view verification can remove implausible occupied voxels and improve the usable object extent, but it cannot fully resolve all orientation ambiguity without more informative viewpoints or contact feedback. This is consistent with the design goal of OV-PEAR: the method is intended to produce a manipulation-ready pose-scale state and reusable object model for previously unseen objects, rather than a globally optimal dense reconstruction from arbitrary sparse views.

The initial model-building time of OV-PEAR is longer than that of pure pose-estimation methods, because this stage constructs a verified object model and writes it into the feature library. This cost is therefore paid when a new object model is introduced, not at every pose update. Once the library is available, subsequent pose-scale estimation reuses the verified object model and takes 0.53 s per instance. The update is slightly slower than SAM-6D because OV-PEAR additionally performs scale alignment and visibility-sensitive registration, but it yields a reusable object model and lower pose errors. This trade-off is appropriate for industrial workcells in which a previously unseen object may be observed once and then manipulated or localized repeatedly.

Qualitatively, Fig. 3 shows that OV-PEAR separates target objects in multi-object clutter and uses multi-view evidence to correct unsupported single-view geometry. The progressive removal of hallucinated voxels and the preservation of observation-supported regions explain why the multi-view model is useful for manipulation: it improves geometric plausibility, pose-scale consistency, and model reuse in the robot frame.

![](images/4608d9e27a63fc34c8646479a1e03a647e22ecd86c75b148cff20a7965bfddb5.jpg)  
(a) Object-model construction and world-frame replay.  
Same-frame panorama and local reconstruction comparison around a dynamic mobile cart

![](images/916c2ae2f625e8d7f870e0eaa7d866b5691e5ebf6a9e2571742ba298b2c68755.jpg)  
(b) Same-frame local reconstruction comparison.  
Fig. 4. Qualitative evidence for short-term object-state maintenance. (a) OV-PEAR constructs an observation-verified object model from RGB-D evidence and reuses it through compact object-level state logging. (b) Same-frame local reconstruction comparison around a dynamic mobile cart. OV-PEAR preserves a compact object-level instance, whereas map-oriented baselines mainly reconstruct visible surfaces without persistent object identity.

TABLE III  
AUDIT OF SCENE-REPRESENTATION CAPABILITY FOR SHORT-TERM OBJECT-STATE MAINTENANCE
<table><tr><td>Method</td><td>Obj. state err. T/S</td><td>Replay Sp./Sem.</td><td>Visible F/Chamfer-L1</td><td>Eff. Comp./Init/Upd.</td></tr><tr><td>OV-PEAR</td><td>0.170 / 4.36%</td><td>0.7800 / 0.9998</td><td>0.5638 / 0.1485</td><td>1.52× / 455.06 s / 0.02 ms</td></tr><tr><td>nvblox</td><td>N/A</td><td>0.1070 / N/A</td><td>0.6532 / 0.0972</td><td>17.43× / 12.31 s / 14.66 ms</td></tr><tr><td>SplaTAM</td><td>N/A</td><td>0.0230 / N/A</td><td>0.0423 / 0.2739</td><td> $0 . 6 4 \times \mid 1 3 4 . 3 5 \mathrm { ~ s ~ / ~ } 1 1 1 9 . 5 4 \mathrm { ~ m s }$ </td></tr><tr><td>GS-ICP</td><td>N/A</td><td>0.0074 / N/A</td><td>0.2687  / 0.1871</td><td> $1 7 0 . 8 0 \times / 1 7 . 7 2 \mathrm { ~ s ~ / ~ } 1 4 7 . 6 9$  ms</td></tr><tr><td>ConceptFusion</td><td>N/A</td><td>0.0930 / N/A</td><td>0.9593 / 0.0331</td><td> $0 . 0 1 \times / 1 5 1 . 7 6 \mathrm { ~ s ~ / ~ } 1 8 0 . 6 6$  ms</td></tr></table>

Note: This table audits usable outputs for short-term object-state maintenance rather than ranking general SLAM systems. Dynamic Spatial/Semantic is computed only for methods with stable object identity, object state, and replayable instance trajectories. N/A denotes an unsupported native output type.

3) Short-Term State Maintenance Through Object-Model Reuse: We further evaluate whether a verified rigid-object model can be reused in a short-horizon industrial workcell. OV-PEAR first constructs an object-model library from multiview RGB-D observations and then records only the pose, scale, confidence, visibility, and identity of each object instance in subsequent frames. The experiment does not aim to replace general dynamic SLAM or open-set scene mapping systems [36]–[38]; it tests whether object-level state can be preserved compactly after verification.

OV-PEAR obtains 0.7800 / 0.9998 for Dynamic Spatial/Semantic, with normalized object-state error of 0.170, scale error of 4.36%, and incremental update time of 0.02 ms/frame after library construction. The near-unity semantic score indicates stable object identity and model-library indexing, whereas the spatial score is still affected by meshsize mismatch, voxelization resolution, local occlusion, and residual state-estimation error. The result therefore supports reusable object-level state consistency rather than optimal point-level visible-surface fitting.

This distinction is important for interpreting the baseline comparison. As shown qualitatively in Fig. 4, nvblox and ConceptFusion provide strong visible-surface reconstructions, and Gaussian-splatting methods provide efficient differentiable scene representations [39]; however, their native outputs do not directly encode replayable object identity, scale, and perobject state. OV-PEAR instead compresses the dynamic part of the scene into a verified object model plus a lightweight state log. It is therefore better suited to short-horizon industrial object maintenance, while general SLAM systems remain more appropriate for dense static mapping. The stage-wise runtime interpretation is given in Appendix A, and additional layout and ablation visualizations are given in Appendix C.

TABLE IV  
POSE USABILITY OF DIFFERENT PERCEPTION INITIALIZATION METHODS IN THE USB TASK
<table><tr><td>Method</td><td>Prior condition</td><td>Trials</td><td>Within tolerance</td><td>Unusable</td><td>Gross outlier</td><td>Main failure cause</td></tr><tr><td>SAM-6D</td><td>No target CAD; zero-shot pose</td><td>50</td><td>3/50</td><td>47/50</td><td>Observed</td><td>Orientation ambiguity, center bias, and unstable frame-wise estimates</td></tr><tr><td>ICP-based</td><td>estimation Accurate object model required</td><td>50</td><td>43/50</td><td>7/50</td><td>Rare</td><td>Degenerate viewpoints and weak local geometric constraints</td></tr><tr><td>OV-PEAR single con- struction</td><td>No target CAD; generated and verified object model</td><td>50</td><td>42/50; 42/47</td><td>8/50</td><td>0/50</td><td>Three unusable near top-down meshes; five resid- ual errors outside tolerance</td></tr><tr><td>OV-PEAR verified-model reuse</td><td>No target CAD; verified model- library reuse</td><td>20</td><td>20/20</td><td>0/20</td><td>0/20</td><td>No evident failure</td></tr></table>

## C. Robotic Manipulation Experiments

The connector experiments test whether the estimated posescale state and verified geometry can initialize real grasping and compliant insertion. USB, HAN10, and RJ45 connectors differ in insertion geometry and tolerance. To keep the main text focused on perception value, the detailed USB tolerance calibration and extended failure breakdown are moved to Appendix B; the key operating threshold is that the current compliant controller becomes unreliable when the initial pose error exceeds approximately 5 mm or 3<sup>◦</sup>.

Table IV isolates the front-end perception contribution. We define a catastrophic pose outlier as an estimate with center error larger than 5 cm, a clear orientation flip, or an initialization that sends grasping or insertion away from the target region. SAM-6D produces only 3 usable poses in 50 trials because of large angular errors or center shifts. The ICP-based baseline reaches 43/50 usable trials but requires an accurate object model. OV-PEAR produces usable models in 47/50 trials, with 42 poses inside the compliant-control capture range and no catastrophic pose outlier. When the verified USB model is reused, all 20 random-initial-pose trials fall inside the tolerance range.

a) Failure Case Analysis: The failure cases show that OV-PEAR changes the failure profile rather than simply increasing the success count. SAM-6D mainly fails through gross pose outliers, orientation ambiguity, and center bias, which place the connector outside the compliant controller’s capture range. ICP is stronger when an accurate object model is supplied, but its failures arise from degenerate local geometry and weak convergence constraints. OV-PEAR failures are less catastrophic: three unusable models come from insufficient side-view geometry under near top-down observation, and five additional trials exceed the 5 mm or 3<sup>◦</sup> tolerance because of small residual pose errors. The end-to-end failures in Table V then mainly reflect post-grasp connector shift, socketside angular error, hand-eye residuals, robot repeatability, and contact jamming rather than gross target localization errors. A fuller breakdown is provided in Appendix B.

The end-to-end connector trials further show that OV-PEAR provides stable initialization for real manipulation. The average grasp success rate is 99.67%, and the average assembly success rate is 84.00%. The remaining insertion failures follow task tolerance: HAN10 is most sensitive because its multipin structure amplifies angular error, while RJ45 reaches 95% assembly success. These failures mainly originate from post-grasp connector shift, socket-side angular error, handeye residuals, robot repeatability, and contact jamming rather than gross target localization errors. Further gains for lowtolerance connectors require socket-side perception, in-gripper pose estimation, force feedback, or visual servoing.

TABLE V  
GRASPING AND COMPLIANT-INSERTION RESULTS FOR THREE CONNECTOR TYPES
<table><tr><td>Task type</td><td>Grasp success rate</td><td>Insertion success rate</td><td>Assembly success rate</td></tr><tr><td>USB assembly</td><td>99/100</td><td>86/99</td><td>86%</td></tr><tr><td>HAN10 assembly</td><td>100/100</td><td>71/100</td><td>71%</td></tr><tr><td>RJ45 assembly</td><td>100/100</td><td>95/100</td><td>95%</td></tr><tr><td>Average</td><td>99.67%</td><td>84.29%</td><td>84.00%</td></tr></table>

## D. Ablation Studies

We conduct two ablation studies. The first tests whether the core perception modules are necessary for object instantiation and observation-verified modeling. The second tests whether the reuse mechanism preserves object-level state efficiently and stably.

1) Core Perception Modules: Removing feature-librarybased coarse initialization decreases AR from 0.762 to 0.283 and voxel IoU from 0.851 to 0.407. Without a reasonable initial rotation, the subsequent ICP refinement often converges to an incorrect local optimum, making the generated object model difficult to anchor reliably. Removing parallel optimalscale search further reduces AR to 0.194 and voxel IoU to 0.374, showing that scale error affects not only objectsize consistency but also the registration residual and subsequent geometry modeling. These results support the two-stage pose-scale alignment strategy: coarse initialization provides a convergent pose prior, and scale search compensates for size mismatch between the generated object model and real observations.

For multi-view voxel-confidence updating, the single-view setting obtains AR of 0.715 and voxel IoU of 0.760. With two views, the scores increase to 0.733 and 0.825. With three views, they reach 0.762 and 0.851. Multi-view observations therefore both recover areas occluded in a single view and suppress hallucinated structures through cross-view consistency, producing a more reliable object template. Together, coarse initialization, scale refinement, and multi-view voxel updating support OV-PEAR through pose convergence, scale consistency, and geometric trustworthiness.

2) Object-Model Reuse and State Maintenance: Table VII separates accuracy, identity, and efficiency effects. Objectmodel reuse mainly affects representation efficiency rather than single-frame pose accuracy. Removing it leaves pose and replay scores nearly unchanged, but reduces the compression ratio from 1.52× to 0.006× and increases replaymaterialization cost to 5026.94 s. Removing the identity log keeps Dynamic Spatial at 0.779 but reduces Dynamic-only ID w/ coverage from 0.992 to 0.000, showing that frame-wise detections cannot preserve instance identity without an explicit identity log.

![](images/d533727a0cbb2b10b446cbbecf616e446db30ab387563473f3f3148df2b21aa8.jpg)  
Fig. 5. Fine assembly processes for USB, RJ45, and HAN10 connectors. The figure shows representative procedures including multi-view perception, target grasping, pose adjustment, and compliant insertion.

TABLE VI  
ABLATION RESULTS FOR CORE PERCEPTION MODULES USING AR AS THE 6D DETECTION METRIC
<table><tr><td>Module</td><td>Setting</td><td>Views</td><td>AR</td><td>Voxel IoU</td></tr><tr><td>Coarse initialization</td><td>Feature-library rotation initialization</td><td>3</td><td>0.762</td><td>0.851</td></tr><tr><td>Coarse initialization</td><td>Without coarse initialization</td><td>3</td><td>0.283</td><td>0.407</td></tr><tr><td>Optimal-scale search</td><td>With parallel optimal-scale search</td><td>3</td><td>0.762</td><td>0.851</td></tr><tr><td>Optimal-scale search</td><td>Without optimal-scale search</td><td>3</td><td>0.194</td><td>0.374</td></tr><tr><td>Multi-view voxel optimization</td><td>Single view</td><td>1</td><td>0.715</td><td>0.760</td></tr><tr><td>Multi-view voxel optimization</td><td>Multi-view</td><td>2</td><td>0.733</td><td>0.825</td></tr><tr><td>Multi-view voxel optimization</td><td>Multi-view</td><td>3</td><td>0.762</td><td>0.851</td></tr></table>

TABLE VII  
INTERNAL ABLATION OF THE OBJECT-MODEL REUSE MECHANISM
<table><tr><td>Variant</td><td>State err. T/Y/S</td><td>Replay Sp./ID</td><td>Coverage</td><td>Eff. Comp./Upd./Cost</td></tr><tr><td>Full</td><td>0.170 /  10.55°  / 4.36%</td><td>0.779 / 0.992</td><td>100.0%</td><td>1.52× / 0.017 ms / 455.06 s</td></tr><tr><td>w/o model reuse</td><td>0.170 /  10.55°  / 4.36%</td><td>0.779 / 0.992</td><td>100.0%</td><td>0.006 × / 0.017 ms / 5026.94 s</td></tr><tr><td>w/o identity log</td><td>0.170 / 10.55°  / 4.36%</td><td>0.779 / 0.000</td><td>100.0%</td><td>1.52× / 0.018 ms / 455.06 s</td></tr><tr><td>w/o visibility</td><td>0.336 / 49.70°  / 22.06%</td><td>0.600 / 0.769</td><td>77.0%</td><td>1.52× / 0.015 ms / 455.06 s</td></tr><tr><td>w/o heading stab.</td><td>0.170 / 66.52°  / 4.36%</td><td>0.778  / 0.992</td><td>100.0%</td><td>1.52× / 0.017 ms / 455.06 s</td></tr></table>

Note: State err. T/Y/S denotes Trans./BBox, Yaw MAE, and Scale error. Replay Sp./ID denotes Dynamic Spatial and Dynamic-only ID w/ coverage. Eff. Comp./Upd./Cost denotes compression ratio, state-log update time, and initial construction or replay-materialization cost.

Visibility-state maintenance is the most important robustness component under occlusion. Removing it reduces coverage from 100.0% to 77.0%, increases normalized translation error from 0.170 to 0.336, increases Yaw MAE from 10.55<sup>◦</sup> to 49.70<sup>◦</sup>, and decreases Dynamic Spatial from 0.779 to 0.600. By contrast, removing heading stabilization mainly affects yaw, increasing Yaw MAE to 66.52<sup>◦</sup>, while leaving Dynamic Spatial and identity scores nearly unchanged. This indicates that heading stabilization is an orientation-auditing safeguard for cart-like objects rather than the main driver of replay quality. A targeted visualization is provided in Appendix C.

## V. CONCLUSION

This paper presented OV-PEAR, an observation-verified framework for pose-scale estimation and 3D reconstruction of unseen industrial manipulation objects. Instead of assuming target CAD models or object-specific training data, OV-PEAR uses generative 3D priors as initial hypotheses and anchors them in the robot frame through appearance-aware view retrieval, explicit scale search, visibility-sensitive registration, and multi-view RGB-D voxel verification. The resulting representation provides an operation-oriented object interface that combines pose, physical scale, contact-relevant geometry, and reusable object identity.

Experiments on public benchmarks, real workbench scenes, connector assembly, and short-term state maintenance validate this design. OV-PEAR achieved an average AR of 0.657 on BOP-Industrial, reached 1.28 cm translation error, 13.17 deg rotation error, and 0.689 voxel IoU in cluttered real workbench scenes, and achieved 99.67% grasp success and 84.00% assembly success across USB, HAN10, and RJ45 tasks. The model-library extension further reduced framewise state maintenance to a lightweight update after initial object verification. These results show that OV-PEAR is most suitable for industrial workcells where new objects appear but can be repeatedly manipulated or observed after verification. The method is not intended to replace general SLAM, visible-surface mapping, or compliant control, and its current performance still depends on instance segmentation, RGB-D quality, and camera calibration. Future work will focus on active view planning, online model updating, and feedbackguided assembly for more complex dynamic scenes.

TABLE VIII  
KEY EXPERIMENTAL PARAMETERS AND FIXED RULES.
<table><tr><td>Symbol</td><td>Parameter</td><td>Fixed value/rule</td><td>Role</td></tr><tr><td>[smin, smax]</td><td>Scale-search interval</td><td>[0.05, 2.5]</td><td>Covers scale bias from single-view generation.</td></tr><tr><td> $\Delta s$ </td><td>Scale step</td><td>0.025</td><td>Discretizes scale search under the manipulation-tolerance constraint.</td></tr><tr><td> $\mathcal { M } _ { v }$ </td><td>Visible-point selection</td><td> $w _ { m } > 0$ </td><td>Keeps model points facing the camera.</td></tr><tr><td> $\tau _ { \mathrm { c o r r } }$ </td><td>ICP correspondence gate</td><td>0.015 m</td><td>Removes distant nearest-neighbor matches</td></tr><tr><td> $\tau _ { \mathrm { f i t } }$ </td><td>Scale-fitness threshold</td><td>0.015 m 1.5 times the voxel side length</td><td>Scores refined scale candidates.</td></tr><tr><td> $\epsilon _ { h }$ </td><td>Depth-hit threshold</td><td></td><td>Separates depth hits, free-space conflicts, and unknown states.</td></tr><tr><td> $\tau _ { c }$ </td><td>Global confidence threshold</td><td>0.55</td><td>Selects trusted occupied voxels.</td></tr><tr><td> $\tau _ { h }$ </td><td>Minimum depth support</td><td>1</td><td>Requires real depth evidence for trusted voxels.</td></tr><tr><td> $\tau _ { s }$ </td><td>Predicted support without depth hit</td><td>2</td><td>Requires cross-view generative support without a depth hit.</td></tr><tr><td> $N _ { \mathrm { m a x } }$ </td><td>Maximum ICP iterations</td><td>50</td><td>Bounds each visibility-weighted ICP refinement.</td></tr></table>

## APPENDIX A

## FIXED PARAMETERS AND RUNTIME ACCOUNTING

This appendix reports fixed experimental parameters and deployment-stage runtime accounting. These details support the main evaluation but are separated from the main text to keep the experimental narrative focused on method-level comparisons and task outcomes.

## A. Fixed Experimental Parameters

Table VIII lists the fixed experimental parameters used by OV-PEAR. They were set once on the development data according to sensor noise, voxel resolution, number of observation views, and manipulation-tolerance requirements, and were kept unchanged across object categories and test scenes. No test-object-specific or table-result-specific retuning is performed. Variables such as N, M, N<sub>i</sub>, and $N _ { g }$ denote input sizes or data-dependent counts rather than cross-experiment tuning parameters.

## B. Deployment Runtime Accounting

Table IX reports runtime according to deployment stage. The initial construction stage builds an observation-verified object model and writes it into the model library, excluding robot motion, manual annotation, and downstream manipulation execution. The pose-scale update reuses this model and mainly includes view retrieval, scale search, and visibility sensitive local registration. The continuous-frame state-log update only accounts for object-level bookkeeping after the library has been built, and excludes RGB-D instance segmentation, object modeling, and scene-level SLAM or Gaussian map updating.

![](images/e8da279eaf29203a162c278949464c98d9cd75cf971353a789f6257420526040.jpg)  
Fig. 6. End-effector tolerance calibration for USB compliant insertion. The green spherical region indicates the capture-tolerance region jointly induced by end-effector position and angular perturbations. The blue arrow indicates the nominal insertion direction, and the black samples indicate initial-error samples used for calibration.

## APPENDIX B

## USB TOLERANCE AND FAILURE ANALYSIS

To determine the accuracy required for USB insertion, we manually introduce translational and angular offsets and measure the success rate of the compliant insertion strategy. Fig. 6 illustrates the calibrated capture region, and Table X reports the quantitative tolerance range.

The robot experiments contain two different failure sources that should not be conflated. The first is perception-induced initialization error, which determines whether the target pose enters the capture range of the compliant controller. The second is execution- and contact-induced error, which can occur after a usable target pose has already been provided. These include post-grasp connector shift, hand-eye calibration residuals, robot repeatability, socket-side angular error, and contact-stage jamming.

In the USB initialization comparison, SAM-6D failures are dominated by gross pose outliers, orientation ambiguity, and center bias. Such errors frequently place the connector outside the 5 mm and $3 ^ { \circ }$ capture range and cannot be recovered by the compliant controller. The ICP-based baseline has fewer outliers when an accurate object model is manually supplied, but it remains vulnerable to degenerate viewpoints and weak local geometric constraints. These cases show that a correct object model alone is insufficient when local registration has a poor convergence basin.

TABLE IX  
DEPLOYMENT STAGES AND RUNTIME ACCOUNTING OF OV-PEAR.
<table><tr><td>Stage</td><td>Trigger</td><td>Output or maintained state</td><td>Runtime</td></tr><tr><td>Object-model construction</td><td>New object or new library entry</td><td>Observation-verified object mesh and model-library entry</td><td>33.45 s (real workbench) 455.06 s (continuous-frame library)</td></tr><tr><td>Pose-scale update</td><td>stance appears</td><td>Before manipulation or when a new in- Object pose, scale, and confidence in the robot frame</td><td>0.53 s/instance</td></tr><tr><td>State-log update</td><td>Each frame after library construction</td><td>Object ID, pose-scale state, confidence, and visibility log</td><td>0.02 ms/frame</td></tr></table>

TABLE X

TOLERANCE RANGE OF USB COMPLIANT INSERTION CONTROL WITH RESPECT TO INITIAL POSE ERROR.
<table><tr><td>Initial pose-error range</td><td>Success rate</td><td>Interpretation</td></tr><tr><td>Translation error  $\leq 3 \mathrm { { m m } }$  Angular error  $\leq$  1°</td><td>100%</td><td>Fully inside the stable capture region.</td></tr><tr><td>Translation error 3–5 mm Angular error  $\leq 1 ^ { \circ }$ </td><td>98%</td><td>Mostly recoverable with small directional error.</td></tr><tr><td>Translation error 3–5 mm Angular error  $1 ^ { \circ } - 3 ^ { \circ }$ </td><td>73%</td><td>Close to the tolerance bound- ary and sensitive to contact</td></tr><tr><td>Translation error &gt; 5 mm or angular error  $> 3 ^ { \circ }$ </td><td>0%</td><td>conditions. Outside the capture range of the compliant controller.</td></tr></table>

OV-PEAR failures have a different profile. In the singleconstruction setting, three trials fail because the generated and verified USB mesh lacks sufficient side-view geometry under near top-down observation. Among the remaining usable models, five trials still exceed the calibrated tolerance because small residual translation or angular errors remain outside the compliant-control capture basin. No catastrophic pose outlier is observed. When the verified USB model is reused, all 20 random-initial-pose trials enter the tolerance range, indicating that model verification and reuse remove most perception-side instability.

The end-to-end connector results further show that final assembly failure is not equivalent to front-end perception failure. For USB, HAN10, and RJ45, the object is usually localized and grasped successfully, but insertion can still fail when the connector shifts after grasping or when contact $\mathrm { { d y } \mathrm { { - } } }$ namics amplify small pose errors. HAN10 is the most sensitive case because its multi-pin structure strongly penalizes angular deviation. These observations indicate that OV-PEAR provides stable pose-scale initialization, while higher success rates for low-tolerance connectors will require socket-side perception, in-gripper pose estimation, force feedback, or closed-loop visual servoing.

## APPENDIX C

## STATE-MAINTENANCE VISUAL DETAILS

This appendix provides supplementary visual context for the short-term state-maintenance experiment and the statemaintenance component ablation. Fig. 7 shows the registered static assets, RGB-D camera locations, and trajectories of the four moving carts. Fig. 8 visualizes the component ablation summarized in Table VII; blue indicates the full OV-PEAR output, whereas orange indicates the corresponding ablation variant.

Top-down scene layout and raw mobile-cart trajectories  
![](images/1a364ccce8458e1f788b679fe66b5169a45fd5ff2f0288a173eef1efabe39f14.jpg)  
Fig. 7. Top-down layout of the short-term state-maintenance scene.

![](images/76d221eb3ff1419c214114656003c844ea1f1ff580be49730a02f907c19d5be9.jpg)  
Fig. 8. Diagnostic visualization of state-maintenance component ablations.

[1] C. Zhuang, S. Li, and H. Ding, “Instance segmentation based 6D pose estimation of industrial objects using point clouds for robotic binpicking,” Robotics and Computer-Integrated Manufacturing, vol. 82, p. 102541, 2023. [Online]. Available: https://doi.org/10.1016/j.rcim.2023. 102541

[2] J. Mahler, M. Matl, V. Satish, M. Danielczuk, B. DeRose, S. McKinley, and K. Goldberg, “Learning ambidextrous robot grasping policies,” Science Robotics, vol. 4, no. 26, p. eaau4984, 2019. [Online]. Available: https://doi.org/10.1126/scirobotics.aau4984

[3] Y. Chen, M. Wei, X. Wang, Y. Liu, J. Wang, H. Song, L. Ma, D. Di, C. Sun, K. Liu, L. Qi, J. Yu, X. Tian, S. Liang, C. Duan, Z. Hong, W. Zhang, and T. Liu, “Embodied AI: A survey on the evolution from perceptive to behavioral intelligence,” SmartBot, vol. 1, no. 3, p. e70003, 2025. [Online]. Available: https://doi.org/10.1002/smb2.70003

[4] G. Marullo, L. Tanzi, P. Piazzolla, and E. Vezzetti, “6D object position estimation from 2D images: A literature review,” Multimedia Tools and Applications, vol. 82, no. 16, pp. 24 605–24 643, 2022. [Online]. Available: https://doi.org/10.1007/s11042-022-14213-z

[5] C. Yuanwei, M. Hairi Mohd Zaman, and M. Faisal Ibrahim, “A review on six degrees of freedom (6D) pose estimation for robotic applications,” IEEE Access, vol. 12, pp. 161 002–161 017, 2024. [Online]. Available: https://doi.org/10.1109/ACCESS.2024.3487263

[6] I. D. Apostolopoulos and M. A. Tzani, “Industrial object and defect recognition utilizing multilevel feature extraction from industrial scenes with deep learning approach,” Journal of Ambient Intelligence and Humanized Computing, vol. 14, no. 8, pp. 10 263–10 276, 2022. [Online]. Available: https://doi.org/10.1007/s12652-021-03688-7

[7] C. A. Akar, J. Tekli, D. Jess, M. Khoury, M. Kamradt, and M. Guthe, “Synthetic object recognition dataset for industries,” in 2022 35th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI). IEEE, 2022, pp. 150–155. [Online]. Available: https://doi.org/10.1109/SIBGRAPI55357.2022.9991784

[8] L. Zhou, G. Wu, Y. Zuo, X. Chen, and H. Hu, “A comprehensive review of vision-based 3D reconstruction methods,” Sensors, vol. 24, no. 7, p. 2314, 2024. [Online]. Available: https://doi.org/10.3390/s24072314

[9] SAM 3D Team, X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, A. Lin, J. Liu, Z. Ma, A. Sagar, B. Song, X. Wang, J. Yang, B. Zhang, P. Dollar, G. Gkioxari,´ M. Feiszli, and J. Malik, “SAM 3D: 3Dfy anything in images,” 2025. [Online]. Available: https://arxiv.org/abs/2511.16624

[10] Z. Fan, J. Zhang, R. Li, J. Zhang, R. Chen, H. Hu, K. Wang, H. Qu, S. Zhou, D. Wang, Z. Yan, H. Xu, J. Theiss, T. Chen, J. Li, Z. Tu, Z. Wang, and R. Ranjan, “VLM-3R: Vision-language models augmented with instruction-aligned 3D reconstruction,” 2025. [Online]. Available: https://arxiv.org/abs/2505.20279

[11] S. Hinterstoisser, S. Holzer, C. Cagniart, S. Ilic, K. Konolige, N. Navab, and V. Lepetit, “Multimodal templates for real-time detection of texture-less objects in heavily cluttered scenes,” in 2011 International Conference on Computer Vision. IEEE, 2011, pp. 858–865. [Online]. Available: https://doi.org/10.1109/ICCV.2011.6126326

[12] R. B. Rusu, N. Blodow, Z. C. Marton, and M. Beetz, “Aligning point cloud views using persistent feature histograms,” in 2008 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2008, pp. 3384–3391. [Online]. Available: https://doi.org/10.1109/IROS. 2008.4650967

[13] R. B. Rusu, N. Blodow, and M. Beetz, “Fast point feature histograms (FPFH) for 3D registration,” in 2009 IEEE International Conference on Robotics and Automation. IEEE, 2009, pp. 3212–3217. [Online]. Available: https://doi.org/10.1109/ROBOT.2009.5152473

[14] Y. Xiang, T. Schmidt, V. Narayanan, and D. Fox, “PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes,” in Robotics: Science and Systems XIV. Robotics: Science and Systems Foundation, 2018. [Online]. Available: https://doi.org/10.15607/RSS.2018.XIV.019

[15] W. Kehl, F. Manhardt, F. Tombari, S. Ilic, and N. Navab, “SSD-6D: Making RGB-based 3D detection and 6D pose estimation great again,” in 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017, pp. 1530–1538. [Online]. Available: https://doi.org/10.1109/ICCV.2017.169

[16] B. Tekin, S. N. Sinha, and P. Fua, “Real-time seamless single shot 6D object pose prediction,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018, pp. 292–301. [Online]. Available: https://doi.org/10.1109/CVPR.2018.00038

[17] Y. He, W. Sun, H. Huang, J. Liu, H. Fan, and J. Sun, “PVN3D: A deep point-wise 3D keypoints voting network for 6DoF pose estimation,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 11 629–11 638. [Online]. Available: https://doi.org/10.1109/CVPR42600.2020.01165

[18] W. Chen, X. Jia, H. J. Chang, J. Duan, and A. Leonardis, “G2L-Net: Global to local network for real-time 6D pose estimation with embedding vector features,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 4232–4241. [Online]. Available: https://doi.org/10.1109/CVPR42600. 2020.00429

[19] J. Lin, L. Liu, D. Lu, and K. Jia, “SAM-6D: Segment anything model meets zero-shot 6D object pose estimation,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 27 906–27 916. [Online]. Available: https://doi.org/10.1109/CVPR52733.2024.02636

[20] P. Moulon, P. Monasse, and R. Marlet, “Global fusion of relative motions for robust, accurate and scalable structure from motion,” in 2013 IEEE International Conference on Computer Vision. IEEE, 2013, pp. 3248–3255. [Online]. Available: https://doi.org/10.1109/ICCV.2013.403

[21] H. Yin and H. Yu, “Incremental SFM 3D reconstruction based on monocular,” in 2020 13th International Symposium on Computational Intelligence and Design (ISCID). IEEE, 2020, pp. 17–21. [Online]. Available: https://doi.org/10.1109/ISCID51228.2020.00011

[22] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016, pp. 4104–4113. [Online]. Available: https: //doi.org/10.1109/CVPR.2016.445

[23] Y. Furukawa and J. Ponce, “Accurate, dense, and robust multiview stereopsis,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 8, pp. 1362–1376, 2010. [Online]. Available: https://doi.org/10.1109/TPAMI.2009.161

[24] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “MVSNet: Depth inference for unstructured multi-view stereo,” in Computer Vision– ECCV 2018. Springer International Publishing, 2018, pp. 785–801. [Online]. Available: https://doi.org/10.1007/978-3-030-01237-3 47

[25] N. Wang, Y. Zhang, Z. Li, Y. Fu, W. Liu, and Y.-G. Jiang, “Pixel2Mesh: Generating 3D mesh models from single RGB images,” in Computer Vision–ECCV 2018. Springer International Publishing, 2018, pp. 55–71. [Online]. Available: https://doi.org/10.1007/978-3-030-01252-6 4

[26] Z. Li, T. Muller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and¨ C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 8456–8465. [Online]. Available: https://doi.org/10.1109/CVPR52729.2023.00817

[27] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 5294–5306. [Online]. Available: https://doi.org/10.1109/CVPR52734.2025.00499

[28] J. Ost, T. Banerjee, M. Bijelic, and F. Heide, “Towards generalizable and interpretable three-dimensional tracking with inverse neural rendering,” Nature Machine Intelligence, vol. 7, no. 8, pp. 1322–1330, 2025. [Online]. Available: https://doi.org/10.1038/s42256-025-01083-x

[29] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 3992–4003. [Online]. Available: https://doi.org/10.1109/ICCV51070.2023.00371

[30] K. N. Kutulakos and S. M. Seitz, “A theory of shape by space carving,” International Journal of Computer Vision, vol. 38, no. 3, pp. 199–218, 2000. [Online]. Available: https://doi.org/10.1023/A:1008191222954

[31] T. Hodan, F. Michel, E. Brachmann, W. Kehl, A. G. Buch,ˇ D. Kraft, B. Drost, J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt, F. Tombari, T.-K. Kim, J. Matas, and C. Rother, “BOP: Benchmark for 6D object pose estimation,” in Computer Vision – ECCV 2018. Springer, 2018, pp. 19–35. [Online]. Available: https://doi.org/10.1007/978-3-030-01249-6 2

[32] T. Hodan, M. Sundermeyer, Y. Labbe, V. N. Nguyen, G. Wang,´ E. Brachmann, B. Drost, V. Lepetit, C. Rother, and J. Matas, “BOP challenge 2023 on detection, segmentation and pose estimation of seen and unseen rigid objects,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2024, pp. 5610–5619. [Online]. Available: https://doi.org/10.1109/CVPRW63382. 2024.00570

[33] B. Wen, W. Yang, J. Kautz, and S. Birchfield, “FoundationPose: Unified 6D pose estimation and tracking of novel objects,” in 2024

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 17 868–17 879. [Online]. Available: https://doi.org/10.1109/CVPR52733.2024.01692

[34] K. Kleeberger, C. Landgraf, and M. F. Huber, “Large-scale 6D object pose estimation dataset for industrial bin-picking,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 2573–2578. [Online]. Available: https: //doi.org/10.1109/IROS40897.2019.8967594

[35] B. Drost, M. Ulrich, P. Bergmann, P. Hartinger, and C. Steger,¨ “Introducing MVTec ITODD: A dataset for 3D object recognition in industry,” in 2017 IEEE International Conference on Computer Vision Workshops (ICCVW). IEEE, 2017, pp. 2200–2208. [Online]. Available: https://doi.org/10.1109/ICCVW.2017.257

[36] A. Millane, H. Oleynikova, E. Wirbel, R. Steiner, V. Ramasamy, D. Tingdahl, and R. Siegwart, “nvblox: GPU-accelerated incremental signed distance field mapping,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 2698–2705. [Online]. Available: https://doi.org/10.1109/ICRA57147.2024.10611532

[37] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, “SplaTAM: Splat, track and map 3D gaussians for dense RGB-D SLAM,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 21 357–21 366. [Online]. Available: https://doi.org/10.1109/ CVPR52733.2024.02018

[38] K. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, G. Iyer, S. Saryazdi, T. Chen, A. Maalouf, S. Li, G. Gan, C. M. de Melo, S. Tulsiani, F. Shkurti, L. Paull, A. Kapoor, A. Gaidon, S. Scherer, J. Luiten, J. Johnson, M. Engelcke, L. Guibas, and M. Savva, “ConceptFusion: Open-set multimodal 3D mapping,” in Robotics: Science and Systems XIX, 2023. [Online]. Available: https://doi.org/10.15607/RSS.2023.XIX.066

[39] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023. [Online]. Available: https://doi.org/10.1145/3592433