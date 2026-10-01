# HiWE: Hierarchical World Knowledge Model with Visual Keypoint Enhancement for Zero-Shot 3D Path Planning

Guoqing Ma 1,2,4, Mingqi Yuan 3, Chen Gao 5, Jiayu Chen 3, Shan Yu 1,4 †

<sup>∗</sup> <sup>1</sup> Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>2</sup> School of Future Technology, University of Chinese Academy of Sciences

<sup>3</sup> The University of Hong Kong, Hong Kong SAR, China

<sup>4</sup> State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology, CAS

<sup>5</sup> Department of Electronic Engineering, Tsinghua University, China

## Abstract

Robot demonstration generation requires a system to identify where an interaction should occur, plan a feasible motion, and execute the required contact. HiWE connects these decisions through a point-based interface between visual grounding and language-based planning. PointVLM is instruction-tuned to associate task-relevant objects with image coordinates using a mixture of point annotations, segmentation-derived samples, robot observations, and visual question answering data. Depth measurements lift these predictions into a semantic 3D representation. A language planner, 3DLLM, uses this representation to specify end-efector waypoints and gripper commands, while a hybrid grasping module resolves local grasp poses. The evaluation covers 14 simulated manipulation tasks and four physical-robot tasks, together with ablations of the visual training data, spatial inputs, and grasp selection. Here, zero-shot execution refers to deployment without taskspecific demonstration training; the visual model uses existing robot data during fine-tuning. This paper describes the original point-based formulation of the framework; its relationship to the subsequent GeneralVLA extension is detailed in the introduction.

## Introduction

Collecting a successful robot demonstration requires more than recognizing an object or describing an intended action. A system must locate an interaction region, relate that region to the geometry of the scene, and execute a motion that respects the object and its surroundings. Errors at any of these stages can invalidate the resulting demonstration. This paper studies how an explicit point representation can connect visual grounding to motion planning for this purpose.

Action-predicting vision-language models, such as Open-VLA (Kim et al. 2024), learn from observations paired with robot actions. Large datasets expand the range of available training experience (O’Neill et al. 2024; Khazatsky et al. 2024), but collecting deployment-specific demonstrations remains costly. An alternative is to divide the problem into components with diferent supervision requirements: imagespace grounding can use point or region annotations, a language model can reason over a spatial description, and a controller can handle execution. The usefulness of this division depends on what information passes between the components.

HiWE (Hierarchical World Knowledge Model with Visual Keypoint Enhancement) uses PointVLM to predict objectassociated image coordinates. Depth measurements convert these coordinates into 3D points for 3DLLM, which generates a sequence of waypoints and gripper commands. A hybrid grasping module (HGM) selects local grasp poses for execution. We use “world knowledge” to describe the pretrained models’ semantic and spatial priors; this terminology does not imply that HiWE learns an explicit transition model of the environment.

Relationship to GeneralVLA. HiWE is the original formulation ofthe framework, and GeneralVLA (Ma et al. 2026) is a subsequent extension that changes perception and adds experience-based planning. This relationship concerns the development of the methods, rather than their order of appearance on arXiv. HiWE uses PointVLM for direct coordinate prediction and 3DLLM for planning from the current instruction and scene. GeneralVLA introduces ASM, which combines language-conditioned segmentation with iterative refinement, and a KnowledgeBank that retrieves and accumulates experience across executions. The three-stage organization, the spatial planning interface, and HGM are common to the two manuscripts. We describe these common components here as part of the original system, while identifying segmentation refinement and persistent experience retrieval as features of the extension. Some experimental values and implementation descriptions also occur in both manuscripts, as noted alongside the relevant results.

The present evaluation examines point localization, the composition of PointVLM’s training data, and the spatial information supplied to the planner. It also measures task execution in simulation and on a physical robot. Throughout this paper, “zero-shot” concerns deployment without taskspecific demonstration training. It does not mean that every component is untrained or that no robot data are used:

![](images/89b0d22fd755d65745646b979bf6c671154f486dd514eafe0d9094a8f4ee90ec.jpg)  
Figure 1: Interfaces for robot manipulation. HiWE exposes task-relevant points and planned trajectories between perception and execution. This separation allows the visual component to be trained on annotations that do not require deployment-specific action trajectories.

Table 1: Scope of the original system and its subsequent extension. Shared components are not independent evidence of a new architecture.
<table><tr><td>Component</td><td>HiWE</td><td>GeneralVLA</td></tr><tr><td>Perception</td><td>PointVLM coordinate prediction</td><td>ASM segmentation and refinement</td></tr><tr><td>Planning</td><td>Current task and 3D scene points</td><td>3D scene planning with KnowledgeBank</td></tr><tr><td>Experience</td><td>No persistent retrieval in 3DLLM</td><td>Retrieval,construc- tion, consolidation</td></tr><tr><td>Execution</td><td>HGM grasp selection</td><td>HGM grasp selection</td></tr></table>

PointVLM is fine-tuned using existing datasets, including robot observations from outside the deployment environment.

This paper presents the original system and its evaluation through three contributions:

• A point-based interface that connects an instruction-tuned visual model to a language planner and a grasp execution module, separating visual supervision from deploymentspecific action demonstrations.

• A concrete implementation comprising PointVLM training, depth-based spatial grounding, 3DLLM waypoint generation, and HGM grasp selection.

• An evaluation on simulated and physical manipulation tasks, with ablations of visual training data, planner inputs, and grasp selection, and a study of demonstrationbased policy training.

The experiments characterize this original configuration. They do not isolate the incremental benefit of GeneralVLA’s segmentation or memory extensions; that would require a controlled comparison under matched conditions.

## Related Work

Language models as robot planners. Code-as-Policies (Liang et al. 2023) represents a plan as executable code calling control functions. VoxPoser (Huang et al. 2023) connects language to geometric planning through 3D value maps, while Scaling-up (Ha, Florence, and Song 2023) uses language-guided interaction to obtain data for policy learning. These systems provide diferent interfaces between semantic decisions and robot execution. HiWE uses an ordered sequence of spatial waypoints and gripper commands; its evaluation compares the resulting system against these baselines under the inputs specified in the experiments.

Learning actions from vision and language. Models including RT-1 and OpenVLA learn action predictions conditioned on observations and instructions (Brohan et al. 2023; Kim et al. 2024). LLARVA additionally predicts trajectories as part of its learning formulation (Niu et al. 2024). HiWE instead exposes an intermediate spatial description to a separate planner and execution module. This changes the interfaces and supervision used by the system; it does not guarantee that reasoning, visual accuracy, or execution speed will improve in every setting.

Points and trajectories as intermediate representations. Afordance prediction connects object semantics to possible interaction locations (Sundaresan et al. 2023; Nasiriany et al. 2024; Yuan et al. 2024b). RoboPoint is particularly relevant to PointVLM because it supplies point-based visual supervision. Trajectory specifications also provide a way to condition policies (Gu et al. 2024), and point tracking makes motion information available from image sequences (Doersch et al. 2023; Yuan et al. 2024a; Wen et al. 2024). GeneralVLA (Ma et al. 2026) is the closest architectural comparison because it combines semantic afordances, 3D planning, and grasp-pose selection. As the subsequent extension of HiWE, it retains the planning and execution backbone while developing the perception and experience-reuse mechanisms. The relationship is summarized in Table 1.

![](images/1dc340e17113c633d3ff97e6a5814bc37cfe8f86247c426c439c087c44b85a7b.jpg)  
Figure 2: HiWE data interfaces. PointVLM associates task-relevant objects with image points. Depth supplies their 3D coordinates for 3DLLM, which outputs waypoints and gripper commands. HGM provides local grasp poses for the execution stage. The overall decomposition and grasping component are also described in GeneralVLA (Ma et al. 2026).

## HiWE: Point-Based Grounding and Planning System interfaces

HiWE passes explicit spatial information between perception, planning, and execution (Figure 2). PointVLM receives an image and task instruction and returns object-associated image points. Depth supplies a 3D location for each point. The resulting semantic scene description is the input to 3DLLM, whose output consists of end-efector waypoints and gripper commands. HGM resolves local grasp poses before a motion planner executes the commands. GeneralVLA retains this overall organization while extending the perception and planning components (Ma et al. 2026).

## PointVLM: predicting interaction locations

PointVLM provides the geometric anchors used by the subsequent planner. Given a task instruction, it identifies relevant objects and represents each object by a set of image coordinates, {object : $( x _ { 0 } , y _ { 0 } ) , \ldots , ( x _ { n } , y _ { n } ) \}$ . Associating several points with an object gives the planner information about its spatial extent after depth projection. The output is a coordinate sequence; HiWE does not use the segmentation decoder or segmentation-feedback loop described for ASM in GeneralVLA.

Training. The visual backbone is QWen-VL-7B (Wang et al. 2024). Following the instruction-tuning formulation of (Liu et al. 2023), the manuscript’s implementation updates the MLP projector and vision-language transformer while keeping the image encoder and tokenizer fixed. Responses are generated autoregressively, with a boundary token delimiting the instruction and answer. Point supervision specifies where an interaction can occur, while language supervision supports interpretation of the instruction.

Supervision sources. The training mixture contains five sources (??). RoboPoint (Yuan et al. 2024b) contributes 347k point-prediction examples. For LVIS (Gupta, Dollár, and Girshick 2019), points are sampled inside segmentation masks and paired with object semantics. Robot observations come from Open X-Embodiment (O’Neill et al. 2024) and SIM-PLER (Li et al. 2024); these sources are outside the deployment environment. Finally, 667K visual question-answering conversations (Kafle and Kanan 2017) provide languagebased supervision. The contribution of each source is evaluated in Table 4. Additional implementation details appear in Appendix ??.

## 3DLLM: planning from a spatial description

Depth and camera geometry lift PointVLM’s image coordinates into 3D. Each coordinate remains associated with its object name, so 3DLLM receives a task instruction together with a compact, semantic description of the scene. The planner specifies an ordered sequence of spatial waypoints interleaved with commands to open or close the gripper. This interface can express several interaction stages within one plan.

Multiple points help describe an object’s orientation and extent; obstacle points supply context for choosing a route. These inputs are evaluated separately in Table 6. They inform the proposed motion but do not guarantee collision-free execution. Unlike the KnowledgeBank-equipped planner in GeneralVLA (Ma et al. 2026), the HiWE planner described here does not retrieve or consolidate a persistent collection of previous experiences. It plans from the current instruction and scene description. Training a separate behavior-cloning policy on successful demonstrations, described in the appendix, is distinct from such retrieval during planning.

## HGM: resolving local grasp geometry

A waypoint specifies where the end efector should move, but a successful grasp also requires an appropriate orientation and contact configuration. HGM uses the task-relevant 3D points to restrict the region of interest in the RGB-D reconstruction. The grasp predictor (Yuan et al. 2023) then generates candidate poses in the resulting object-centered point cloud.

HGM rejects candidates that fail its collision check and selects the remaining candidate whose grasp center is nearest to the object center. This is a selection heuristic rather than a guarantee of a globally optimal grasp. The selected pose is combined with the planned waypoints and passed to the motion planner for execution. HGM is part of the original HiWE pipeline and also appears in the GeneralVLA extension; the ablation in Table 7 evaluates its role within HiWE.

## Experiments

We evaluate the original HiWE configuration through task execution, point localization, component ablations, and behavior cloning from generated demonstrations. The configuration uses PointVLM and current-scene 3DLLM planning, without the ASM or KnowledgeBank extensions.

Relationship between the reported experiments. The baseline entries for VoxPoser, CAP, and Scaling-up in Table 2, and the CAP and RoboPoint entries in Table 5, also appear in GeneralVLA (Ma et al. 2026). They are reported here to contextualize HiWE’s results and should not be interpreted as evidence of an independent replication across the two manuscripts. The tables do not provide a controlled comparison of HiWE against the later extension.

Implementation details. PointVLM extracts the objectassociated coordinates used to construct the planner input. The implementation represents each object with at least three points to expose more spatial information than a single location. DeepSeek R1 receives the textual 3D scene description and generates the motion plan. The output is limited to 20 spatial waypoints to bound the length of the generated plan. Full prompts are included in the Appendix. PointVLM uses the front-camera view in these experiments. The image input has a resolution of 256 × 256.

## Zero-shot performance in simulation

The simulation evaluation covers 14 tasks, including grasping and non-prehensile interactions. We measure task success under the specified environment and baseline inputs.

Benchmark and execution. The 14 RLBench tasks (James et al. 2020) vary in object identity, placement, and the number of required interactions. The simulator is CoppeliaSim, accessed through PyRep, with a Franka Panda arm and parallel gripper. The environment provides four RGB-D cameras; the front view supplies the PointVLM input described above. A motion planner (Sucan, Moll, and Kavraki 2012) converts the requested waypoints into executable robot motion. Taskspecific success conditions are listed in the appendix.

Comparison methods and their inputs. We include Codeas-Policies (CAP) (Liang et al. 2023), Scaling-up-Distilling-Down (Ha, Florence, and Song 2023), and VoxPoser (Huang et al. 2023). Their control interfaces difer: CAP composes programs from supplied action primitives; Scaling-up performs language-guided exploration with 6-DoF primitives; VoxPoser obtains motion targets from spatial value maps. In this evaluation, CAP and Scaling-up receive simulator states and object models, and VoxPoser receives segmented object point clouds. These inputs difer from HiWE’s learned grounding, so the table compares the specified complete systems rather than isolating planner quality. This benchmark protocol and its baseline results are shared with the GeneralVLA report (Ma et al. 2026).

Coverage and limitations. Table 2 reports a nonzero success rate for HiWE on every tested task. The corresponding coverage is 10 tasks for Scaling-up, 9 for VoxPoser, and 7 for CAP. HiWE has the highest reported mean on 10 tasks; the remaining tasks show that its advantage is not uniform. In particular, non-prehensile interactions and fine manipulation expose the need for more accurate pose estimates or adjustments during execution. The illustrated rollouts (Figure 3) show examples with multiple objects and interaction stages, rather than establishing performance on every such task.

## Real-world experiments

Hardware and trial design. Physical tests use an Agilex-2.0 Piper arm with a parallel gripper and an Intel RealSense L515 LiDAR RGB-D camera viewing the workspace from above. Language instructions specify four tasks: move\_spray\_bottle, open\_drawer, open\_jar, and sort\_object. Each task has three trials of ten episodes, with object poses varied across episodes. Table 5 summarizes the resulting success rates.

Observed behavior. HiWE succeeds in at least some episodes of all four tasks and has a higher reported success rate than both listed baselines. The examples in ?? illustrate the role of spatial planning: bottle placement uses the inferred bottle height, while drawer opening requires a motion consistent with the drawer’s orientation. In the evaluated baseline configurations, RoboPoint supplies localization without the trajectory-planning stage, and CAP lacks a supplied draweropening primitive. These implementation choices constrain what can be concluded from the comparison.

## Ablation Study

## Point location accuracy of PointVLM

Point localization is measured by the fraction of predicted points that fall within the annotated target mask. Table 3 reports the mean and standard deviation across 3 runs. PointVLM obtains the highest value among the listed models; this metric evaluates localization rather than the feasibility of a complete robot motion.

Table 4 removes each supervision source in turn. Every removal lowers the reported RoboRefIt accuracy, with the largest decrease occurring when LVIS is omitted. The result supports using segmentation-derived semantic point annotations in the training mixture.

## Information required by 3DLLM

The planner ablations vary the dimensionality of the coordinates, the number of points describing each object, and the inclusion of obstacles (Table 6). The 2D variant plans in image coordinates before converting the path into 3D. The one-point variant removes the geometric extent supplied by multiple points. The no-obstacle variant retains the target object but omits surrounding obstacle information. These changes test whether the available scene description is suficient for interactions such as extracting an umbrella along the direction permitted by its stand. The full representation performs best or ties for best on the two reported tasks. Obstacle information has a clear efect on Take umbrella, whereas its removal leaves the reported Put block result unchanged. The results therefore support task-dependent benefits rather than universal necessity of every input.

Table 2: Task-averaged success rate % for zero-shot evaluation. HiWE outperformed other baselines in 10 out of 14 simulation tasks from RLBench (James et al. 2020). Each task was evaluated over 3 seeds to obtain the task-averaged success rate and standard deviations.
<table><tr><td>Method</td><td>Put_block</td><td>Play_jenga</td><td>Open_jar</td><td>Close_box</td><td>Open_box</td><td>Pickup_cup</td><td>Push_block</td></tr><tr><td>VoxPoser (Huang et al. 2023)</td><td>70.70±2.31</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>26.70±14.00</td><td>25.33±8.33</td></tr><tr><td>CAP (Liang et al. 2023)</td><td>84.00±16.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>14.67±4.62</td><td>8.00±4.00</td></tr><tr><td>Scaling-up (Ha, Florence, and Song 2023)</td><td>77.33±6.11</td><td>0.00±0.00</td><td> $7 8 . 6 7 { \scriptstyle \pm 1 1 . 5 5 }$ </td><td>0.00±0.00</td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>9.33±2.26</td><td>5.33±6.11</td></tr><tr><td>HiWE (Ours)</td><td>93.33±4.16</td><td>82.00±10.39</td><td>84.00±6.00</td><td> $\pm 8 . 6 7 { \pm } 1 2 . 0 6$ </td><td> $3 4 . 6 7 { \scriptstyle \pm 1 6 . 7 7 }$ </td><td>88.67±3.06</td><td> $2 3 . 3 3 { \pm } 1 0 . 0 7$ </td></tr><tr><td>HiWE w/o FT</td><td>75.33±7.57</td><td>60.67±9.45</td><td>71.33±10.07</td><td> $3 1 . 3 3 { \pm } 9 . 0 2 $ </td><td> $8 . 6 7 { \pm } 3 . 0 6 $ </td><td>74.67±6.11</td><td>14.67±11.02</td></tr><tr><td>Method</td><td>Take_umbrella</td><td>Sort_mustard</td><td>Open_wine</td><td>Lamp_on</td><td>Put_knife</td><td>Pick_&amp;_lift</td><td>Insert_block</td></tr><tr><td>VoxPoser (Huang et al. 2023)</td><td>33.33±8.33</td><td> $\mathbf { 9 6 . 0 0 { \pm } } 6 . 9 3 $ </td><td>8.00±4.00</td><td>57.30±12.22</td><td>92.00±4.00</td><td>96.00±0.00</td><td>0.00±0.00</td></tr><tr><td>CAP (Liang et al. 2023)</td><td>4.00±4.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>64.00±6.93</td><td>14.67±8.33</td><td>100.00±0.00</td><td>0.00±0.00</td></tr><tr><td>Scaling-up (Ha, Florence, and Song 2023)</td><td> $6 . 6 7 { \scriptstyle \pm 2 . 3 1 }$ </td><td> $4 1 . 3 3 { \pm } 1 2 . 8 6$ </td><td>33.33±20.13</td><td>60.00±8.00</td><td> $2 4 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $\mathbf { 1 0 0 . 0 0 { \pm } 0 . 0 0 }$ </td><td>0.00±0.00</td></tr><tr><td>HiWE (Ours)</td><td> ${ \bf 6 4 . 6 7 { \pm } 1 5 . 0 1 }$ </td><td> $7 3 . 6 7 { \pm } 1 5 . 5 0 $ </td><td>45.33±11.02</td><td> $7 2 . 6 7 { \pm } 1 4 . 0 5$ </td><td> $6 0 . 0 0 { \scriptstyle \pm 1 4 . 0 0 }$ </td><td> $8 7 . 3 3 { \pm } 1 0 . 2 6 $ </td><td>36.67±5.03</td></tr><tr><td>HiWE w/o FT</td><td> $5 0 . 6 7 { \scriptstyle \pm 1 1 . 0 2 }$ </td><td>59.33±8.33</td><td>26.67±8.08</td><td> $6 2 . 6 7 { \scriptstyle \pm 1 3 . 3 2 }$ </td><td>45.33±5.03</td><td> $6 9 . 3 3 { \pm } 1 5 . 1 4$ </td><td>14.67±7.02</td></tr></table>

Table 3: Quantitative comparisons on object reference (RoboRefIt). The metric is percentage of predicted points within the target mask.
<table><tr><td colspan="2">Qwen-VL LLaVA-NeXT SpaceLLaVA GPT-4o PointVLM</td></tr><tr><td>24.1±0.9 20.0±0.9</td><td>21.3±0.9 15.3±1.3 52.1±1.2</td></tr></table>

Table 4: Ablation on the data composition. Results on RoboRefIt show that best results are achieved when all of the data sources are combined during instruction-tuning.
<table><tr><td>No VQA No LVIS No Pixel No Sim No Robo All</td></tr><tr><td>42.5±3.7 25.8±2.1 32.6±5.3 46.6±3.2 47.8±3.3 52.1±1.2</td></tr></table>

## Ablation study of the HGM module

Table 7 tests the information and selection rules used by HGM on Play jenga and Take umbrella. Removing RGB leaves depth-only grasp estimation. Removing the semantic 3D points eliminates the target-object guidance. The filter-C ablation disables collision rejection; the filter-N ablation samples a remaining candidate randomly instead of using the nearest-center rule. The full HGM configuration has the highest reported mean on both tasks. Removing target-point guidance yields no successful trials in this evaluation, while the other ablations reduce success without eliminating it. These findings concern the tested tasks and do not imply that each modality is indispensable for every manipulation problem.

Table 5: Zero-shot success rates (%) in real-world manipulation. HiWE consistently outperforms Code as Policies (CAP) (Liang et al. 2023) and RoboPoint across four representative tasks.
<table><tr><td>Method</td><td>Move Spray Bottle</td><td>Open Drawer</td><td>Open Jar</td><td>Sort Object</td></tr><tr><td>CAP</td><td>6.67</td><td>0.00</td><td>36.67</td><td>70.00</td></tr><tr><td>RoboPoint</td><td>0.00</td><td>0.00</td><td>20.00</td><td>63.33</td></tr><tr><td>HiWE</td><td>60.00</td><td>33.33</td><td>56.67</td><td>80.00</td></tr></table>

Table 6: Ablation Study on 3DLLM for Trajectory Planning
<table><tr><td>Method</td><td>Take umbrella</td><td>Put block</td></tr><tr><td>3DLLM-2D</td><td>2.00±2.00</td><td>19.33±5.03</td></tr><tr><td>3DLLM-1point</td><td>26.67±3.06</td><td>82.00±4.00</td></tr><tr><td>3DLLM w/o obstacle</td><td>23.33±7.02</td><td>93.33±4.16</td></tr><tr><td>3DLLM</td><td> $6 4 . 6 7 { \scriptstyle \pm 1 5 . 0 1 }$ </td><td>93.33±4.16</td></tr></table>

## Data scaling

We examine how the performance of an RVT-2 policy changes as more HiWE demonstrations are supplied for training. The comparison uses demonstrations generated by RL-Bench as a reference. The reported linear fits have slopes of 0.543 for HiWE-generated demonstrations and 0.156 for RLBench-generated demonstrations. These coeficients summarize the evaluated range and are not a general scaling law.

Table 7: Ablation Study on the HGM Module
<table><tr><td>Method</td><td>Play jenga</td><td>Take umbrella</td></tr><tr><td>HGM w/o rgb</td><td>56.67±7.57</td><td>34.00±12.49</td></tr><tr><td>HGM w/o 3D point</td><td>0.00±0.00</td><td>0.00±0.00</td></tr><tr><td>HGM w/o filter-C</td><td>59.33±5.03</td><td>54.67±15.01</td></tr><tr><td>HGM w/o filter-N</td><td>78.67±7.57</td><td>52.00±14.00</td></tr><tr><td>HGM</td><td>82.00±10.39</td><td>64.67±15.01</td></tr></table>

![](images/068be610e139768b1f2c18347d7d071f797ab546783468b680681ea0c891e051.jpg)

![](images/0b88a566890687b98ae70bd114f4ff9d66cb3582d2a9106240ed5a741fb1a61d.jpg)

![](images/456cd58af305f8e1edcd5c1e68b6a2c3f506be6bd64b1a89508d3876ece073d8.jpg)

![](images/af6ccb5f38922c165e79024508391cf8dd2f311eb4cbde4df34619645b7c137a.jpg)  
Figure 3: Illustrative manipulation sequences. The displayed stages connect object localization, a spatial motion plan, and robot execution for scenes involving several interactions.

![](images/aa7147fd7ffb6975c439d0abdcd9fc4e35aa897afaeceb64fcb10b330499e366.jpg)

![](images/0b8e2fa5104936d5d82acb959891f7af17c0dfbffada672516290252691eb68b.jpg)  
Figure 4: The multi-view robustness of PointVLM. This assists 3DLLM in reasoning about the direction for pulling out the block.

## Conclusion

HiWE establishes a point-based pipeline for connecting visual grounding, spatial language planning, and grasp execution. The original system uses a fine-tuned PointVLM to produce geometric anchors, 3DLLM to plan from the current scene, and HGM to resolve grasp poses. Its evaluation examines manipulation success, the visual supervision mixture, and the spatial information needed for planning. Successful executions also provide demonstrations for policy training. GeneralVLA subsequently extends this framework through afordance segmentation and persistent experience retrieval (Ma et al. 2026); these mechanisms are outside the

HiWE configuration evaluated here. Remaining limitations include errors in point localization, incomplete geometric descriptions, and grasp execution failures. Zero-shot deployment should be understood in the stated task-specific sense, since existing robot observations are part of the visual training data.

## References

Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; Ibarz, J.; Ichter, B.; Irpan, A.; Jackson, T.; Jesmonth, S.; Joshi, N. J.; et al. 2023. RT-1: Robotics Transformer for Real-World Control at Scale. In Bekris, K. E.; Hauser, K.; Herbert, S. L.; and Yu, J., eds., Robotics: Science and Systems XIX, Daegu, Republic ofKorea, July 10-14, 2023.

Doersch, C.; Yang, Y.; Vecerík, M.; Gokay, D.; Gupta, A.; Aytar, Y.; Carreira, J.; and Zisserman, A. 2023. TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, 10027–10038. IEEE.

Gu, J.; Kirmani, S.; Wohlhart, P.; Lu, Y.; Arenas, M. G.; Rao, K.; Yu, W.; Fu, C.; Gopalakrishnan, K.; et al. 2024. RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Gupta, A.; Dollár, P.; and Girshick, R. B. 2019. LVIS: A

Dataset for Large Vocabulary Instance Segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, 5356–5364. Computer Vision Foundation / IEEE.

Ha, H.; Florence, P.; and Song, S. 2023. Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition. In Tan, J.; Toussaint, M.; and Darvish, K., eds., Conference on Robot Learning, CoRL 2023, 6-9 November 2023,Atlanta, GA, USA, volume 229 of Proceedings ofMachine Learning Research, 3766–3777. PMLR.

Huang, W.; Wang, C.; Zhang, R.; Li, Y.; Wu, J.; and Fei-Fei, L. 2023. VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models. In Tan, J.; Toussaint, M.; and Darvish, K., eds., Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, volume 229 of Proceedings of Machine Learning Research, 540– 562. PMLR.

James, S.; Ma, Z.; Arrojo, D. R.; and Davison, A. J. 2020. RLBench: The Robot Learning Benchmark & Learning Environment. IEEE Robotics Autom. Lett., 5(2): 3019–3026.

Kafle, K.; and Kanan, C. 2017. An Analysis of Visual Question Answering Algorithms. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, 1983–1991. IEEE Computer Society.

Khazatsky, A.; Pertsch, K.; Nair, S.; Balakrishna, A.; Dasari, S.; Karamcheti, S.; Nasiriany, S.; Srirama, M. K.; Chen, L. Y.; Ellis, K.; Fagan, P. D.; Hejna, J.; Itkina, M.; Lepert, M.; et al. 2024. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. In Kulic, D.; Venture, G.; Bekris, K. E.; and Coronado, E., eds., Robotics: Science and Systems XX, Delft, The Netherlands, July 15-19, 2024.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E. P.; Sanketi, P. R.; Vuong, Q.; Kollar, T.; Burchfiel, B.; Tedrake, R.; Sadigh, D.; Levine, S.; Liang, P.; and Finn, C. 2024. OpenVLA: An Open-Source Vision-Language-Action Model. In Agrawal, P.; Kroemer, O.; and Burgard, W., eds., Conference on Robot Learning, 6-9 November 2024, Munich, Germany, volume 270 of Proceedings of Machine Learning Research, 2679– 2713. PMLR.

Li, X.; Hsu, K.; Gu, J.; Mees, O.; Pertsch, K.; Walke, H. R.; Fu, C.; Lunawat, I.; Sieh, I.; Kirmani, S.; Levine, S.; Wu, J.; Finn, C.; Su, H.; Vuong, Q.; and Xiao, T. 2024. Evaluating Real-World Robot Manipulation Policies in Simulation. In Agrawal, P.; Kroemer, O.; and Burgard, W., eds., Conference on Robot Learning, 6-9 November 2024, Munich, Germany, volume 270 of Proceedings of Machine Learning Research, 3705–3728. PMLR.

Liang, J.; Huang, W.; Xia, F.; Xu, P.; Hausman, K.; Ichter, B.; Florence, P.; and Zeng, A. 2023. Code as Policies: Language Model Programs for Embodied Control. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, 9493–9500. IEEE.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems 36: Annual Conference on Neural

Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Ma, G.; Wang, S.; Zhang, Z.; Yu, S.; and Tang, H. 2026. GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning. Version 1, arXiv:2602.04315.

Nasiriany, S.; Xia, F.; Yu, W.; Xiao, T.; Liang, J.; Dasgupta, I.; Xie, A.; Driess, D.; Wahid, A.; et al. 2024. PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Niu, D.; Sharma, Y.; Biamby, G.; Quenum, J.; Bai, Y.; Shi, B.; Darrell, T.; and Herzig, R. 2024. LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning. In Agrawal, P.; Kroemer, O.; and Burgard, W., eds., Conference on Robot Learning, 6-9 November 2024, Munich, Germany, volume 270 of Proceedings of Machine Learning Research, 3333– 3355. PMLR.

O’Neill, A.; Rehman, A.; Maddukuri, A.; Gupta, A.; Padalkar, A.; Lee, A.; Pooley, A.; Gupta, A.; Mandlekar, A.; Jain, A.; Tung, A.; Bewley, A.; et al. 2024. Open X-Embodiment: Robotic Learning Datasets and RT-X Models : Open X-Embodiment Collaboration. In IEEE International Conference on Robotics and Automation, ICRA 2024, Yokohama, Japan, May 13-17, 2024, 6892–6903. IEEE.

Sucan, I. A.; Moll, M.; and Kavraki, L. E. 2012. The Open Motion Planning Library. IEEE Robotics Autom. Mag., 19(4): 72–82.

Sundaresan, P.; Belkhale, S.; Sadigh, D.; and Bohg, J. 2023. KITE: Keypoint-Conditioned Policies for Semantic Manipulation. In Tan, J.; Toussaint, M.; and Darvish, K., eds., Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, volume 229 of Proceedings ofMachine Learning Research, 1006–1021. PMLR.

Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Fan, Y.; Dang, K.; Du, M.; Ren, X.; Men, R.; Liu, D.; Zhou, C.; Zhou, J.; and Lin, J. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. CoRR, abs/2409.12191.

Wen, C.; Lin, X.; So, J. I. R.; Chen, K.; Dou, Q.; Gao, Y.; and Abbeel, P. 2024. Any-point Trajectory Modeling for Policy Learning. In Kulic, D.; Venture, G.; Bekris, K. E.; and Coronado, E., eds., Robotics: Science and Systems XX, Delft, The Netherlands, July 15-19, 2024.

Yuan, C.; Wen, C.; Zhang, T.; and Gao, Y. 2024a. General Flow as Foundation Afordance for Scalable Robot Learning. In Agrawal, P.; Kroemer, O.; and Burgard, W., eds., Conference on Robot Learning, 6-9 November 2024, Munich, Germany, volume 270 of Proceedings of Machine Learning Research, 1541–1566. PMLR.

Yuan, W.; Duan, J.; Blukis, V.; Pumacay, W.; Krishna, R.; Murali, A.; Mousavian, A.; and Fox, D. 2024b. RoboPoint: A Vision-Language Model for Spatial Afordance Prediction in Robotics. In Agrawal, P.; Kroemer, O.; and Burgard, W., eds., Conference on Robot Learning, 6-9 November 2024,

Munich, Germany, volume 270 of Proceedings of Machine Learning Research, 4005–4020. PMLR.

Yuan, W.; Murali, A.; Mousavian, A.; and Fox, D. 2023. M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place. In Tan, J.; Toussaint, M.; and Darvish, K., eds., Conference on Robot Learning, CoRL 2023, 6-9 November 2023, Atlanta, GA, USA, volume 229 of Proceedings of Machine Learning Research, 3619–3630. PMLR.