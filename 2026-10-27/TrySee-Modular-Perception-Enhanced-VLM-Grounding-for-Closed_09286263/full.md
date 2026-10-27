# TrySee: Modular Perception-Enhanced VLM Grounding for Closed-Loop Robot Task Planning in Specialized Domains

Elias Hofele<sup>1,2</sup>, Thomas Trapp<sup>1</sup>, David Timmermann<sup>1</sup>, Tristan Schnell<sup>1</sup>, Georg Heppner<sup>1</sup>, Arne Roennau<sup>1,2</sup>

Abstract— Vision Language Models (VLMs) have shown impressive capabilities in high-level reasoning and task planning for robots. However, specialized domains remain a challenge for VLMs’ image interpretation and grounding abilities. We introduce TrySee, a modular closed-loop robot task planning pipeline integrating general-purpose VLM reasoning with domain-specific perception and established robotics components. The system consists of an inherently reactive cyclic planning pipeline, augmented with domain-specific modular vision preprocessing, to provide online grounding. Finally, a dynamic preprocessing selection mechanism contributes to the scalability of the pipeline. The system requires neither VLM fine-tuning nor end-to-end demonstrations and adapts through replaceable perception modules and motion planning. The approach is evaluated on planning tasks across 5 widely different domains, including a real-world robot disassembly task. The field tests in a brownfield environment demonstrate the approach’s practicality in integrating VLM planning into specialized robotic systems with limited domain-specific training data.

## I. INTRODUCTION

The idea of controlling robots with natural language has long been a staple of science fiction. However, accomplishing even simple tasks, such as “Clean up the table.”, requires a complex understanding of the world and the robot’s capabilities, as well as drawing on many of the core challenges in robotics, such as perception, manipulation, and planning.

With the recent advances in Vision Language Models (VLMs) [1] [2], significant progress has been made in enabling robot reasoning and planning by leveraging vast amounts of real-world knowledge [3] [4]. While VLMbased systems have produced impressive results for tasks in everyday environments (such as offices or homes), they still struggle in more specialized domains, including industrial and scientific applications [5] [6]. This is attributable to these domains being underrepresented in the VLMs’ training data. The resulting performance gap is a barrier to the more widespread adoption of VLMs in these fields, in which data collection and annotation is often particularly expensive and time-consuming.

A fundamental requirement for VLMs to generate effective robot task plans is the ability to ground their language understanding in the robot’s perception and action capabilities. In other words, the objects and features comprising the robot’s environment must be linked to the semantics used in the given task, and these, in turn, must be linked to the concrete movements and poses necessary to interact with them. General-purpose VLMs often fail at the first step for recondite objects, and their image-level outputs alone do not provide sufficiently reliable geometric targets for robot execution. A practical planning system therefore requires explicit interfaces connecting domain-specific perception, VLM-level semantics, and the robot’s action capabilities.

![](images/dedaf08ff05b798a59e1da0a489d1eca3ecb8b2c503b244fc167ba3f16687ae2.jpg)

![](images/ae9d5b1a2436dd09a0452ba58dee6b4e48721717213b8db2c4f30432782d62d1.jpg)

![](images/9c9b3c9e5c535b011232cd56743301d2285c01db6bb5e72a72b813bc780a0a56.jpg)

![](images/097cd4e518c5a2605b80c3ee3d7c0110a2ed0cb7f12b891da1f310ff35471c8c.jpg)

![](images/622857481c2bb7e68426399187d077516560c7314bcafd33157496b8ec7c5d29.jpg)  
Fig. 1. Example images used to evaluate TrySee, preprocessed by the pipeline’s Vision Preprocessing Module. The images are labeled A through E, corresponding to the tasks used to evaluate the proposed pipeline (see Table II). A shows a computer assembly task, B an agricultural task, C a service task, D an aircraft cockpit interaction task, and E an e-waste disassembly task. Additional examples are collected in Fig. 3.

In this work, we tackle the challenge of VLM grounding in specialized domains by presenting a modular pipeline for VLM-based robot task planning. Rather than fine-tuning the model or replacing the entire existing robot software stack, TrySee compensates the general-purpose VLM’s limitations in object recognition and robot pose generation by connecting its reasoning abilities with replaceable domainspecific perception modules and established motion-planning components. This allows the pipeline to be adaptable to different domains and tasks, through targeted perception models without requiring end-to-end robot demonstrations.

In TrySee, the VLM incrementally generates and verifies short-horizon action sequences, while the perception modules provide semantic object labels, visual masks, and executable target poses. This provides online grounding that is continuously updated as the environment changes. For object-dense environments, the VLM can request preprocessing only for the information relevant to the current verification step.

Our approach enables the VLM to generate task plans for a variety of domains. We evaluate the pipeline using tasks across 5 vastly different domains (see Fig. 1), including ewaste recycling, a workshop environment, and an agricultural setting. The focus is on complex, multi-step tasks. In order to underline the real-world applicability of our approach, we also implement the pipeline on a real robot and evaluate it in an industrial recycling environment.

## II. RELATED WORK

In recent years, substantial progress has been made in the application of Large Language Models (LLMs) and Vision Language Models (VLMs) in high-level reasoning and task planning for robots.

LLM- and VLM-based task planning: In SAYCAN [3], planning is performed by sequentially selecting skills from a predefined library. Candidate skills are scored using both their linguistic relevance and learned affordance functions, grounding their selection in the robot’s capabilities. PROGPROMPT [7] leverages the code-generation capabilities of modern LLMs by expressing actions, assertions, and plans directly in Python syntax, making them easier to process downstream. SAYPLAN [4] grounds planning in large environments by traversing a 3D scene graph representation.

These approaches represent open-loop task planners and are easily prone to failure in dynamic environments. By contrast, our approach and the remainder of this section discuss closed-loop task planning approaches instead.

Reactive task planning using LLMs and VLMs: INNERMONOLOGUE [8] attempts to close the loop of SAYCAN by injecting execution feedback and scene descriptions from visual-question-answering models (or human feedback) into the LLM prompt at each step. ROSLLM [9] implements a generic ROS-based framework for integrating LLM agents with robotic systems and supports task correction based on feedback cues. REFLECT [10] maintains a hierarchical memory structure and semantic scene representation, extracted via object detection and CLIP to find relations, to support failure analysis and replanning. However, the LLM itself is only fed textual prompts. BETR-XP-LLM [11] and REPLAN [12] employ VLM-based verification and critique mechanisms to detect and correct planning failures.

AHA [13] further studies failure detection through taskspecific fine-tuning of VLMs. More recent work investigates VLM-based failure detection and recovery through optimized prompting [14], behavior-tree-based reactive failure handling [15], and constraint-aware visual monitoring [16]. Other approaches represent plans as behavior trees [17], allowing the resulting policies to be executed directly within established robotic control frameworks [18]–[22].

These works demonstrate the importance of reactive execution. However, the above approaches rely on external grounding mechanisms to execute the generated plans. Conversely, in our approach the grounding is performed online through the vision module itself.

Explicit visual grounding: ConceptGraphs [23] uses 3D scene graph representations to support languageconditioned planning and spatial reasoning. Other work explores multimodal interfaces between vision-language models and symbolic planners [24]. Recent segmentationoriented VLMs such as LISA [25] show that segmentation masks can be generated from language-based reasoning queries, highlighting the relevance of explicit visual grounding for downstream robotic tasks. Related visual-prompting approaches explicitly annotate candidate image regions: SoM (Set-of-Mark Prompting) uses interpretable marks to improve visual grounding in multimodal models [26]. Similarly, MOKA employs mark-based visual prompts and point-based affordances to connect VLM predictions to robot manipulation actions [27].

Our work follows a deployment-oriented philosophy by integrating modular perception components directly into the planning loop. This allows the planning system to continuously update its grounding during execution while remaining adaptable to different domains through the replacement of individual perception modules.

Vision Language Action (VLA) models: More recently, VLA models have been proposed to directly generate actions from visual inputs [28]–[31]. These models are trained on large datasets of teleoperation data, thus learning the relationship between visual features and actions.

However, VLA models generally require substantial amounts of training data and are often tied to specific robot embodiments, action spaces, or domains. Adapting them to highly specialized applications remains challenging when representative demonstration data is unavailable. By contrast, our approach operates at the task-planning level and combines VLM reasoning with modular perception and classical motion generation techniques. This enables the reuse of existing robotic software stacks while facilitating adaptation to new domains through targeted perception modules rather than large-scale retraining.

## III. TRYSEE

This work aims to improve the high-level task planning capabilities of robots for tasks from specialized domains based on instructions in natural language. In this context, a specialized domain is defined as comprising a specific set of objects, features, or components that are not found in general-purpose datasets and thus are unlikely to be recognized by the VLM out-of-the box. In particular, the focus is on improving the practical applicability of VLMbased robotic planning systems. An overview of the proposed approach is shown in Fig. 2.

![](images/a68e98b27794dc34ddfbc50001fda73cb3bab648c144ee40348f0401f7949573.jpg)  
Fig. 2. TrySee is a cyclic planning pipeline, with the numbers in the top-right corners indicating the flow of operations. In each iteration, the next step of the overall plan is generated, executed and verified. The Vision Preprocessing Module is central to the pipeline, and interacts with every one of the other components. The images shown were not used in the experiments, but purely serve to illustrate the pipeline’s operation.

## A. Components

In this section, we introduce the two main components TrySee is based on: a Vision Language Model (VLM) and a Vision Preprocessing Module. Since the focus of this work is on the integration of these components into a pipeline, we discuss the high-level functionality of the components and how they interact with each other. For more in-depth details on one possible implementation of these components, see the Evaluation Setup in Section IV. Additionally, the underlying robot execution stack is treated as existing infrastructure and exposed through the action toolbox in Table I.

TABLE I  
TOOLBOX OF ACTIONS AVAILABLE TO THE ROBOT.
<table><tr><td>Action</td><td>Parameters</td></tr><tr><td>open_gripper</td><td>N/A</td></tr><tr><td>close-gripper</td><td>N/A</td></tr><tr><td>move_to</td><td>location label</td></tr><tr><td>move_next_to</td><td>location label &amp; direction</td></tr><tr><td>move_towards</td><td>direction</td></tr><tr><td colspan="2">Additional task-specific actions...</td></tr></table>

Vision Language Model: For this work, the VLM is employed in two different modes, differing in how they are prompted. The alternating use of these two modes forms the basis of the cyclic planning pipeline. On the one hand, the VLM is used to generate the next planning step based on the task at hand and the current state of the environment. To this end, it is provided with the task description, the list of actions the robot can perform (each of which represents a basic motion primitive, as seen in Table I, which can be combined with others to form more complex behaviors), current images of the environment, and the list of interactable objects and points of interest found therein. In turn, the VLM generates a Thought (a natural language description of the next step to be executed), a sequence of actions from the toolbox, as well as the goal it aims to achieve with the generated sequence (in natural language). The Thought provides an explicit rationale alongside the structured action output. The division of the overall plan into smaller steps is entirely self-directed by the VLM, which is instructed to generate only one major interaction with its surroundings (e. g., picking up an object) at a time.

Additionally, the same VLM is also used to verify whether the generated plan was executed successfully. This is accomplished by comparing the goal generated in the planning step against current images of the environment as well as images from directly before the execution of the plan. The model first describes the images provided before generating an Observation stating whether or not the expected goal was achieved. The collection of past Observations comprises the memory of the VLM, which is also appended to its input in subsequent planning steps.

The division of the pipeline into explicit planning and verification steps is a key design choice. It allows the VLM to focus on each step individually, without being overwhelmed by having to perform both of these complex tasks simultaneously. Separate prompts are used for planning and verification so that each invocation has a constrained objective and output format.

For the purposes of the proposed pipeline, the VLM is treated as a black box, with the models not being finetuned or adapted to the specific task at hand. This separates domain adaptation from the choice of reasoning model and allows newer VLMs to be integrated without modifying the perception or execution interfaces. Additionally, while finetuning the VLM might improve the model’s innate scene understanding capabilities, external modules would still be required to provide the necessary action grounding.

Vision Preprocessing Module: The modular vision preprocessing is provided by a collection of dedicated computer vision systems, which together aid in grounding the VLM’s language understanding by providing three functions. Despite its name, the module serves both as a visual preprocessing interface for the VLM and as the action-grounding interface for the robot execution layer. Initially, a find objects function identifies any task-relevant interactable objects in the environment and returns their (expressive, natural language) labels. The VLM then uses these labels to reason about the objects in the environment and to parametrize the resulting action sequence. This creates a shared vocabulary between the VLM and the Vision Preprocessing Module. Secondly, a mask objects function is used to overlay a segmentation mask of the given object(s) on the input image. The colors of the masks are communicated to the VLM together with the object labels and masked image, allowing it to reason about the spatial relations of the object in the environment. This performs the task of grounding the VLM’s language understanding in the visual input. Finally, a generate pose function is used to generate a target pose for the given object label. Applying this function for each label used in the task plan enables the robot to actually interact with its surroundings, grounding its actions in the real world.

## B. Pipeline

Fig. 2 shows an overview of the pipeline. The implementation is based on the ReAct framework [32], with the actions from Table I being used as the tools of a VLM-based agent.

The pipeline starts by taking current images of the workspace and preprocessing them using the find objects and mask objects functions introduced above. The resulting images are added to the VLM’s planning mode input, which uses them to generate an action sequence. The generated actions are parsed into a Behavior Tree (BT), whose leaf nodes invoke the corresponding functions from the robot’s action toolbox. The BT provides an inspectable interface between the VLM-generated plan and the underlying robot-control framework. Any object labels parametrizing the plan are mapped to the corresponding robot pose using the Vision Preprocessing Module’s generate pose function. The mapping is performed using a fresh set of current images from the moment in which the relevant action starts being executed. This reduces sensitivity to small scene changes occurring during VLM inference or the execution of preceding actions, increasing the robustness of the pipeline.

Once the execution is completed, another new set of images is taken and preprocessed. These new images, along with the initial images that were used to generate the plan, are then passed to the VLM in verification mode. The preprocessing is especially important here, as the VLM must recognize any differences between the two sets of images to ascertain whether they match the plan’s expected goal. However, in dense environments, the sheer amount of information provided to the VLM can be overwhelming. This can lead the VLM to disregard key information by attending to the wrong masks. Instead, we introduce a dynamic preprocessing selection cycle. Initially, the VLM is only provided with the unpreprocessed images, as well as the list of object labels. Using these labels, the VLM can iteratively request additional preprocessed images until it has enough information to generate an answer. This limits the number of simultaneous visual annotations and helps the verifier focus on task-relevant changes in object-dense scenes. Analogously to the Thought step in the planning mode, the VLM first describes the images, focusing on their differences, giving the model space to reason. Then, it decides whether or not it requires more information based on these descriptions. Finally, once the verification process is finished, the model’s Observation is appended to the VLM’s input in planning mode for the next iteration of the pipeline.

![](images/08fa6605089beec9e7bce5cf64cf04a2fac51fd94ab82dbb3b7d84c5298dd2c2.jpg)  
Fig. 3. Example excerpts of input image sequences for the VLM, as used in the evaluation. The sequences are labeled A through E, corresponding to the tasks in Table II.

TABLE II  
OVERVIEW OF THE TASKS (WITH RESPECTIVE DOMAINS & REQUIRED OBJECT RECOGNITION CAPABILITIES) USED TO PERFORM THE EVALUATION.
<table><tr><td></td><td>Task</td><td>Objects</td><td>Domain</td><td>Evaluation</td></tr><tr><td>A</td><td>PC assembly</td><td>PC components</td><td>Workshop</td><td>Simulation</td></tr><tr><td>B</td><td>Crop care &amp; harvest</td><td>Plant species</td><td>Agriculture</td><td>Simulation</td></tr><tr><td>C</td><td>Order fulfillment</td><td>Facial recognition</td><td>Service</td><td>Simulation</td></tr><tr><td>D</td><td>Cockpit interaction</td><td>Button/dial status</td><td>Aerospace</td><td>Simulation</td></tr><tr><td>E</td><td>Heat cost allocator disassembly</td><td>Allocator parts</td><td>E-waste recycling</td><td>Hardware</td></tr></table>

## IV. EXPERIMENTAL SETUP

The evaluation addresses three questions: whether explicit domain-specific visual grounding improves VLMbased planning and verification, whether visual preprocessing provides benefits beyond textual domain descriptions, and whether VLM-directed preprocessing helps verification in object-dense scenes. To investigate these questions, we compare five configurations constituting ablations of the grounding interface of the cyclic planning pipeline. Initially, we evaluate the performance of the pipeline without the modular vision preprocessing (and consequently, without the dynamic preprocessing selection). This forms our baseline. The second configuration augments the basic VLM inputs with natural language descriptions (D) of the domainspecific objects that can appear in the environment. Next, we introduce the modular vision preprocessing (V ) to the pipeline, both with and without the natural language descriptions. Finally, we evaluate the full pipeline with the dynamic preprocessing selection (S), in order to assess its effect on the scalability of the preprocessing in object-dense environments.

The evaluation consists of a controlled image-sequence study and a real-robot case study. Both provide the VLM with the same forms of visual and textual input, allowing the planning and verification interface to be evaluated consistently. The controlled study isolates the effect of the ground ing information, whereas the hardware study demonstrates how the complete system can be instantiated using learned perception, conventional pose estimation, and an established robot execution stack.

## A. Evaluation Tasks

In order to evaluate the breadth of the pipeline’s applicability, we consider five different tasks from specialized domains, as summarized in Table II. The tasks were chosen to cover a wide range of scenarios, include different types of objects, and require multiple interaction steps to complete. To help visualize the tasks, example input images are shown in Fig. 3.

In task A, the robot is required to pick up a series of PC components (e. g., GPU, RAM, SSD, etc.) and install them into a computer case, according to the given specification. If a given slot is already occupied, the robot must first remove the existing component before proceeding. A core challenge of this task is recognizing which components are installed in the computer’s case in any given moment.

Task B requires the robot to recognize specific crops and handle them in accordance with the instructions, such as harvesting, watering or fertilizing them. Both greenhouse plants and field crops are considered. The main challenge of this task is the fine-grained recognition of different plant species, such as distinguishing between wheat and barley.

Task C is a service task, in which the robot must accept an order and serve it to the correct customer. The customers can be differentiated using facial recognition.

In task D, the robot must complete a simplified flying sequence by interacting with the cockpit of a small aircraft. The task investigates the scalability of the approach, given the large number of interactable objects.

Finally, for task E the robot must disassemble different variants of heat cost allocators, the main goal being the removal of the built-in battery. Destructive means of disassembly are allowed and provided through the use of a milling spindle, which can be interacted with through a dedicated action. Being executed on a real robot, the task explores the real-world applicability of the approach.

Each task is evaluated over 11 different sequences, with variations either in the initial setup or during execution (e. g., failures in the robot’s actions).

## B. Simulation

The evaluation of tasks A through D is performed with simulated perception modules. For each time step, the input images are manually annotated with the labels and segmentation masks that would be produced by a domainspecific perception system. Additional alternate inputs are also created, including failed actions and unexpected scene changes. This isolates the effect of the grounding interface on VLM planning and verification without requiring a complete perception and manipulation implementation for every domain. It does not, however, model recognition or segmentation errors within the perception modules.

## C. Real-World Robot

As seen in Fig. 4, the evaluation environment for task E consists of a table featuring a mounted Universal Robots UR10e with a Weiss Robotics CRG 200-085 gripper with custom fingers for object manipulation. Vision is provided by two StereoLabs ZE 2i stereo cameras with inherent stereo depth estimation capabilities. Movement trajectory planning and execution is done using the ROS 2 middleware [33] and the MoveIt 2 motion planning framework [34]. The BT generation and execution is handled by the ros2 ros bt py framework [35]. This implementation reuses the existing execution stack, confining application-specific adaptation primarily to the perception models and task-specific BT nodes.

![](images/b7d5a9ff15c531889fe9bd0e386a337a279bad5756530293476316788ab4fe8f.jpg)  
Fig. 4. The hardware setup used for the real-world evaluation of the pipeline. Note the milling spindle on the right side of the table (circled in green), as well as the cameras mounted on the frame on the top and on the left (circled in red), providing both a top-down and a lateral perspective of the workspace.

The object and part recognition (as provided by the find objects function) is performed using a YOLOv9 model [36], fine-tuned on a small dataset of 181 labeled instances. For mask objects, component segmentation was not learned but is calculated using a Segment Anything 2 [37] instance utilizing the bounding boxes generated by the YOLOv9 network as region proposals. Finally, localization of the components for generate pose is executed by projecting the detected object features into 3D space using a calibrated camera model.

In order to operate the mill, the VLM can select a part (identified by the vision module) to attempt to remove using a masked close-up image of the heat cost allocator. Based on this selection, the vision module generates a custom milling trajectory, which can be executed by the robot arm itself, which simply moves along it above the spindle while grasping the heat cost allocator.

## D. Choice of VLM

The VLM used for the bulk of the evaluation is the opensource model LLaVA-OneVision [38], consisting of a SigLIP vision encoder [39] and Qwen2 [40] text inference backbone. However, to examine whether the grounding interface transfers between reasoning models, we repeat a subset of the evaluation using the open-source model Gemma 3 [41]. With these additional experiments we investigate the transferability of the results across different VLMs, and the extent to which the choice of VLM affects the performance of the pipeline.

## E. Evaluation Metrics

We manually assess the pipeline at three levels corresponding to its planning and verification stages. Planning Accuracy denotes the proportion of sequences for which the generated plans are judged correct, independently of subsequent verification outputs. Verification Accuracy denotes the proportion of verification decisions matching the known execution outcome and is evaluated for correctly generated plans. Overall Accuracy denotes the proportion of sequences for which both planning and all associated verification decisions are correct.

## V. RESULTS

An overview of the results of the ablation study is shown in Table III, using the metrics defined in the previous section. D indicates the addition of natural language descriptions, V the inclusion of the Vision Preprocessing Module, and S the use of the dynamic preprocessing selection. The table reports results for each of the five tasks and their combined average.

The Vision Preprocessing Module produces a marked increase in Planning Accuracy across all tasks. This is primarily attributable to the improved grounding of the objects in the semantics used in the task instructions, without which the VLM would often use incorrect location labels to attempt to interact with the objects. As the dynamic preprocessing selection (S) is only used for verification, it does not affect the planning accuracy.

For Verification Accuracy, the baseline performance is approximately 50.0%, as the individual verification decisions are binary. As with planning, the Vision Preprocessing Module (V ) greatly improves the verification accuracy. As predicted, the dynamic preprocessing selection (S) further improves the verification accuracy for particularly objectdense environments, while not increasing the verification duration significantly. In fact, in most cases, the VLM only requested a single preprocessed image containing exactly the objects mentioned in the goal.

At the end-to-end level, the complete pipeline achieves an Overall Accuracy of 67.3%. However, we note that none of these planning errors concerned the semantic grounding of the objects but rather issues with instruction-following (such as skipping part of the task) or logical correctness (such as not opening the gripper before attempting to grasp an object). The primary failure mode therefore shifted from semantic grounding to high-level task reasoning.

The evaluation of task C is of particular interest, as it requires performing facial recognition to differentiate between customers. The baseline achieves no successful plans this task. However, once the recognition is provided by the vision module, the task is solved with perfect planning accuracy. Natural language descriptions initially also improve the planning accuracy (by 36.4%), such as for distinguishing between a blonde and a brunette. However, more minute details are difficult to convey using language alone.

Also noteworthy is the performance on task D, which involves a large number of interactable objects in the cockpit setting. The smaller improvement obtained from simultaneously masking all cockpit controls is consistent with the VLM being distracted by the large number of visual annotations. The dynamic preprocessing selection (S) increases verification accuracy by 19.5 percentage points by presenting masks only for objects relevant to the current goal.

TABLE III  
OVERVIEW OF THE RESULTS OF THE ABLATION STUDY, BY TASK.
<table><tr><td rowspan="2"></td><td rowspan="2">Components</td><td colspan="2">Accuracy [%]</td><td rowspan="2">Overall</td></tr><tr><td>Plan.</td><td>Verif.</td></tr><tr><td rowspan="5">A Workshop</td><td>Baseline</td><td>0.0</td><td>50.9</td><td>0.0</td></tr><tr><td>Descriptions D</td><td>0.0</td><td>52.8</td><td>0.0</td></tr><tr><td>Vision V</td><td>72.7</td><td>81.1</td><td>27.3</td></tr><tr><td> $D + V$ </td><td>72.7</td><td>81.1</td><td>27.3</td></tr><tr><td> $D + V +$  - Selection S</td><td>−Ⅱ−</td><td>90.6</td><td>54.6</td></tr><tr><td rowspan="5">B Agriculture</td><td>Baseline  $D$ </td><td>0.0</td><td>52.0</td><td>0.0</td></tr><tr><td></td><td>27.3</td><td>60.0</td><td>0.0</td></tr><tr><td> $V$ </td><td>90.9</td><td>84.0</td><td>45.5</td></tr><tr><td> $D + V$ </td><td>81.8</td><td>86.0</td><td>54.6</td></tr><tr><td> $D + V + S$ </td><td>−Ⅱ−</td><td>94.0</td><td>81.8</td></tr><tr><td rowspan="5"> $\mathbf { C }$  Service</td><td> $\overline { { \mathrm { B a s e l i n e } } }$ </td><td>0.0</td><td>47.2</td><td>0.0</td></tr><tr><td> $D$ </td><td>36.4</td><td>55.6</td><td>9.1</td></tr><tr><td> $V$ </td><td>100</td><td>94.4</td><td>81.8</td></tr><tr><td> $D + V$ </td><td>100</td><td>94.4</td><td>81.8</td></tr><tr><td> $D + V + S$ </td><td>−Ⅱ−</td><td>94.4</td><td>81.8</td></tr><tr><td rowspan="5">D Aerospace</td><td>Baseline</td><td>0.0</td><td>51.2</td><td>0.0</td></tr><tr><td> $D$ </td><td>0.0</td><td>48.8</td><td>0.0</td></tr><tr><td> $V$ </td><td>54.6</td><td>61.0</td><td>9.1</td></tr><tr><td> $D + V$ </td><td>54.6</td><td>61.0</td><td>18.2</td></tr><tr><td> $D + V + S$ </td><td>-1-</td><td>80.5</td><td>27.3</td></tr><tr><td rowspan="5">E Recycling</td><td> $\overline { { \mathrm { B a s e l i n e } } }$ </td><td>18.2</td><td>48.7</td><td>0.0</td></tr><tr><td> $D$ </td><td>36.4</td><td>54.1</td><td>0.0</td></tr><tr><td> $V$ </td><td>100</td><td>86.5</td><td>54.6</td></tr><tr><td> $D + V$ </td><td>100</td><td>89.2</td><td>63.6</td></tr><tr><td> $D + V + S$ </td><td>−Ⅱ−</td><td>97.3</td><td>90.9</td></tr><tr><td rowspan="5">Combined</td><td> $\overline { { \mathrm { B a s e l i n e } } }$ </td><td>3.6</td><td>50.2</td><td>0.0</td></tr><tr><td> $D$ </td><td>20.0</td><td>54.4</td><td>1.8</td></tr><tr><td> $V$ </td><td>83.6</td><td>81.1</td><td>43.6</td></tr><tr><td> $D + V$ </td><td>81.8</td><td>82.0</td><td>49.1</td></tr><tr><td> $D + V + S$ </td><td>−Ⅱ−</td><td>91.2</td><td>67.3</td></tr></table>

## Cross-Model Evaluation

As shown in Table IV, the results obtained with the Gemma 3 VLM are generally consistent with those obtained with LLaVA-OneVision, albeit slightly higher. Nonetheless, the same trends can be observed, with the Vision Preprocessing Module (V) providing a substantial boost in both planning and verification accuracies, and the dynamic preprocessing selection (S) further improving the verification accuracy. This experiment provides preliminary evidence that the benefits of the grounding interface are not specific to the primary VLM.

## Discussion

The use of vision modules for preprocessing the VLM’s visual input shifts the generalization requirements of the pipeline from the VLM to the dedicated vision model. While this may seem like a minor change, it is important to note that the vision modules can be trained on much smaller datasets than VLMs, and can be easily fine-tuned to new domains with only a few additional examples. Therefore, this shift is a key strength of the pipeline, and significantly increases its modularity and its practical applicability in specialized domains.

TABLE IV  
CROSS-MODEL EVALUATION ON TASK A USING GEMMA 3.
<table><tr><td colspan="2">Components</td><td colspan="3">Accuracy [%]</td></tr><tr><td colspan="2"></td><td>Plan.</td><td>Verif.</td><td>Overall</td></tr><tr><td rowspan="3">A</td><td>Baseline</td><td>18.2</td><td>60.4</td><td>0.0</td></tr><tr><td>Descriptions D</td><td>27.3</td><td>66.0</td><td>0.0</td></tr><tr><td>Vision V</td><td>81.8</td><td>81.1</td><td>45.5</td></tr><tr><td rowspan="2">Workshop (Gemma 3)</td><td> $D + V$ </td><td>81.8</td><td>84.9</td><td>45.5</td></tr><tr><td> $D + V +$  Selection S</td><td>− &quot; −</td><td>94.3</td><td>63.6</td></tr></table>

The controlled image-sequence evaluation uses manually prepared perception outputs and therefore isolates their effect on downstream planning and verification rather than measuring sensitivity to perception errors. The real-robot case study complements this evaluation by demonstrating one complete implementation using learned detection, segmentation, pose estimation, and physical execution. A natural extension would be to expose perception confidence through the same modular interface, allowing uncertain detections to trigger additional observations or controlled recovery actions.

After visual grounding is provided, the remaining planning failures primarily concern instruction following and logical consistency. This indicates that grounding alone cannot compensate for all limitations of the reasoning model. These errors could be addressed without changing the perception interface, for example through explicit action constraints, validation of generated Behavior Trees, or a structured memory for longer tasks. Within its current scope, TrySee is best suited to structured applications in which domain-specific perception and robot skills are available and the environment changes primarily between actions. For such applications, the proposed architecture provides a practical, flexible, and robust approach to combine evolving VLM capabilities with mature robot-control software and targeted domain knowl edge.

## VI. CONCLUSION

We presented TrySee, a modular system that connects general-purpose VLM reasoning with domain-specific perception and established robot execution components. The system incrementally generates and verifies short-horizon plans, while a Vision Preprocessing Module supplies semantic object labels, visual masks, and executable target poses. For object-dense scenes, dynamic preprocessing selection allows the VLM to request only the information relevant to the current verification step.

Across five specialized tasks, the complete system achieves 81.8% Planning Accuracy, 91.2% Verification Accuracy, and 67.3% Overall Accuracy, compared with 3.6%, 50.2%, and 0.0% for the ungrounded baseline. The brownfield industrial e-waste disassembly application further demonstrates how the architecture can reuse ROS 2, MoveIt 2, Behavior Trees, and targeted perception models without task-specific VLM fine-tuning. The results show that explicit, modular visual grounding provides a practical route for applying VLM-based task planning in specialized robotic applications while retaining existing robot software and domain-specific capabilities.

## VII. ACKNOWLEDGMENTS

We would like to thank our colleagues at the FZI Research Center for Information Technology for their valuable feedback and support. Special thanks to Vincenzo Di Pentima, for his tireless work on the robot hardware and software setup, as well as to Martin Schulze, for helpful discussions on the current state of the art, and David Oberacker and Niklas Spielbauer, for their insightful early reviews of this paper.

## REFERENCES

[1] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023. [Online]. Available: https://arxiv.org/abs/2304.08485

[2] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” 2023. [Online]. Available: https://arxiv.org/abs/ 2304.10592

[3] M. Ahn, et al., “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances,” Aug. 2022.

[4] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. Reid, and N. Suenderhauf, “Sayplan: Grounding large language models using 3d scene graphs for scalable task planning,” in 7th Annual Conference on Robot Learning, 2023. [Online]. Available: https://openreview.net/forum?id=wMpOMO0Ss7a

[5] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang, “Anomalygpt: Detecting industrial anomalies using large vision-language models,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, 2024, pp. 1932–1940.

[6] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

[7] I. Singh, et al., “ProgPrompt: Generating Situated Robot Task Plans using Large Language Models,” Sept. 2022, arXiv:2209.11302 [cs]. [Online]. Available: http://arxiv.org/abs/2209.11302

[8] W. Huang, et al., “Inner Monologue: Embodied Reasoning through Planning with Language Models,” July 2022, arXiv:2207.05608 [cs]. [Online]. Available: http://arxiv.org/abs/2207.05608

[9] C. E. Mower, et al., “ROS-LLM: A ROS framework for embodied AI with task feedback and structured reasoning,” July 2024.

[10] Z. Liu, A. Bahety, and S. Song, “REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction,” Oct. 2023.

[11] J. Styrud, M. Iovino, M. Norrlof, M. Bj¨ orkman, and C. Smith, “Auto-¨ matic Behavior Tree Expansion with LLMs for Robotic Manipulation,” Sept. 2024.

[12] A. Mei, G.-N. Zhu, H. Zhang, and Z. Gan, “ReplanVLM: Replanning Robotic Tasks with Visual Language Models,” July 2024.

[13] J. Duan, et al., “AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation,” Oct. 2024.

[14] H. Chen, Y. Yao, R. Liu, C. Liu, and J. Ichnowski, “Automating robot failure recovery using vision-language models with optimized prompts,” arXiv preprint arXiv:2409.03966, 2024.

[15] F. Ahmad, H. Ismail, J. Styrud, M. Stenmark, and V. Krueger, “A unified framework for real-time failure handling in robotics using visionlanguage models, reactive planner and behavior trees,” in 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE), 2025, pp. 887–894.

[16] E. Zhou, et al., “Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[17] M. Colledanchise and P. Ogren,<sup>¨</sup> Behavior Trees in Robotics and AI: An Introduction. CRC Press, Oct. 2022, arXiv:1709.00084 [cs]. [Online]. Available: http://arxiv.org/abs/1709.00084

[18] R. A. Izzo, G. Bardaro, and M. Matteucci, “BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Oct. 2024, pp. 9684–9690.

[19] H. Zhou, Y. Lin, L. Yan, J. Zhu, and H. Min, “LLM-BT: Performing Robotic Adaptive Tasks based on Large Language Models and Behavior Trees,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), May 2024, pp. 16 655–16 661.

[20] A. Lykov and D. Tsetserukou, “LLM-BRAIn: AI-driven Fast Generation of Robot Behaviour Tree based on Large Language Model,” May 2023.

[21] A. Lykov, et al., “LLM-MARS: Large Language Model for Behavior Tree Generation and NLP-enhanced Dialogue in Multi-Agent Robot Systems,” Dec. 2023.

[22] X. Chen, Y. Cai, Y. Mao, M. Li, W. Yang, W. Xu, and J. Wang, “Integrating Intent Understanding and Optimal Behavior Planning for Behavior Tree Generation from Human Instructions,” June 2024.

[23] Q. Gu, et al., “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021–5028.

[24] K. Shirai, et al., “Vision-language interpreter for robot task planning,” in IEEE International Conference on Robotics and Automation (ICRA), 2024.

[25] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language model,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 9579–9589.

[26] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v,” 2023. [Online]. Available: https://arxiv.org/abs/2310.11441

[27] K. Fang, F. Liu, P. Abbeel, and S. Levine, “Moka: Open-world robotic manipulation through mark-based visual prompting,” Robotics: Science and Systems (RSS), 2024.

[28] A. Brohan, et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” 2023. [Online]. Available: https://arxiv.org/abs/2307.15818

[29] M. J. Kim, et al., “Openvla: An open-source vision-language-action model,” 2024. [Online]. Available: https://arxiv.org/abs/2406.09246

[30] NVIDIA, et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” 2025. [Online]. Available: https: //arxiv.org/abs/2503.14734

[31] K. Black, et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” 2026. [Online]. Available: https: //arxiv.org/abs/2410.24164

[32] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing Reasoning and Acting in Language Models,” Mar. 2023, arXiv:2210.03629 [cs]. [Online]. Available: http://arxiv.org/abs/2210.03629

[33] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, “Robot operating system 2: Design, architecture, and uses in the wild,” Science Robotics, vol. 7, no. 66, p. eabm6074, 2022. [Online]. Available: https://www.science.org/doi/abs/10.1126/ scirobotics.abm6074

[34] D. Coleman, I. Sucan, S. Chitta, and N. Correll, “Reducing the barrier to entry of complex robotic software: a moveit! case study,” arXiv preprint arXiv:1404.3785, 2014.

[35] G. Heppner, N. Berg, D. Oberacker, N. Spielbauer, A. Roennau, and R. Dillmann, “Distributed behavior trees for heterogeneous robot teams,” in 2023 IEEE 19th International Conference on Automation Science and Engineering (CASE), 2023, pp. 1–8.

[36] C.-Y. Wang, I.-H. Yeh, and H.-Y. M. Liao, “Yolov9: Learning what you want to learn using programmable gradient information,” 2024. [Online]. Available: https://arxiv.org/abs/2402.13616

[37] N. Ravi, et al., “Sam 2: Segment anything in images and videos,” 2024. [Online]. Available: https://arxiv.org/abs/2408.00714

[38] B. Li, et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[39] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[40] A. Yang, et al., “Qwen2 technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2407.10671

[41] G. Team, et al., “Gemma 3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2503.19786