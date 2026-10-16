# LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation

Zhixu Li<sup>1,∗,†</sup>, Keqian Tang<sup>1,∗</sup>, Litian Gong<sup>1,∗</sup>, Jingyu Yao<sup>2</sup>, Wenqian Zhang<sup>1</sup>, Tianze Xu<sup>5</sup>, Zehao Wang<sup>3</sup>, Yuping Wang<sup>4</sup>, Junge Zhang<sup>3</sup>, and Jiachen Li<sup>3,1,‡</sup> <sup>1</sup>University of California, Riverside <sup>2</sup>Yale University <sup>3</sup>Georgia Institute of Technology

<sup>4</sup>University of Michigan, Ann Arbor <sup>5</sup>University of Toronto

<sup>∗</sup>Equal Contribution <sup>†</sup>Project Lead <sup>‡</sup>Corresponding Author: jiachen\_li@gatech.edu

## Abstract

Personalization is essential for general-purpose robots to transition into household environments and achieve true practicality. While general robotic manipulation has been extensively studied, personalized robotic manipulation (PeRM) lacks a controlled benchmark for separating task completion from preference satisfaction. We introduce LIBERO-PeRM, a benchmark for preference learning in embodied agents. LIBERO-PeRM studies five questions: 1) how policies follow preferences in zero-shot settings; 2) how they learn single preferences from demonstrations; 3) how efficiently they acquire preferences as demonstrations increase; 4) how learned preferences generalize across layouts and tasks; and 5) how policies compose preferences and behave under conflicts. We propose a controlled formulation and taxonomy of observable preferences in simulation, covering motion, skill, and userprofile choices. This formulation is restricted to language-expressible preferences that can be instantiated in LIBERO without changing the task goal, evaluated automatically, and reused across tasks. Building on LIBERO, our procedural pipeline generates paired demonstrations with and without preferences. We create five task suites (312 tasks) with high-quality demonstrations, rich metadata, and preference-satisfaction metrics. Experiments show that current VLA policies struggle with zero-shot personalization, while SFT makes preferences learnable. Mixed training achieves better no-preference retention than sequential adaptation and improves performance with contextual prompts compared with preference-only training, but can weaken motion-preference learning; preference generalization, decomposition, and conflict handling remain difficult. The code, datasets, and demos can be found on the project website: https://libero-perm.github.io/.

## 1 Introduction

Household manipulation has been a long-standing goal in robotics and embodied AI, with the promise of improving daily life by enabling robots to perform chores from natural language instructions (e.g., “put an apple, bread, and milk on my plate”). Recent progress in generalist robotic manipulation has been driven in large part by Vision-Language-Action (VLA) models [1, 2, 3, 4], which show strong generalization across diverse multimodal inputs [5, 6] and heterogeneous embodiments [7, 8]. However, real-world deployment demands more than completing the nominal task: embodied agents must also execute tasks in ways that align with the preferences, habits, and constraints ofindividual users, a requirement we refer to as personalization.

Preferences introduce diverse execution requirements for identical goals. For instance, placing a mug might require a specific handle orientation or a designated landing spot. Consequently, personalized manipulation must satisfy user-specific constraints on both what strategies are selected and how actions are performed. In this framework, preferences are treated as a learning dimension parallel to object-oriented task learning; the policy’s objective is to maximize preference adherence within a single trajectory without compromising fundamental task success.

![](images/25616bf4f5009342d3a51196cb35efa8b83c5345c3343a8b0812beacd1c3b70f.jpg)  
Figure 1: Top: LIBERO-PeRM contains five procedurally generated task suites with pipeline-generated, humanreviewed demonstrations. Bottom: we investigate five key research questions in personalized manipulation.

Nevertheless, progress on personalized robotic manipulation (PeRM) remains impeded by two major gaps. First, PeRM lacks a shared formulation and standardized evaluation protocol across prior work. Existing studies define personalization through diverse problem settings, including implicit preference alignment [9, 10], multi-user style adaptation [11], user-specific object-receptacle pairing [12], and helpful action prediction [13]; they also specify preferences through natural language [13, 14], few user examples [9], explicit preference functions [15], human speech [5], or numerical preference vectors [11]. Evaluation is similarly fragmented, spanning task success, user studies [9, 11, 13, 14], and preference-based scoring functions [16, 17], making methods difficult to compare and failure modes hard to diagnose. Second, existing benchmarks typically emphasize high-level navigation or semantic selection over fine-grained manipulation preferences and multi-preference conflicts. The closest preference-based RL benchmark [18] lacks robotic preference-learning datasets and multipreference evaluation. Other personalization benchmarks focus on object navigation or high-level receptacle selection [19, 20], while recent memory-based manipulation benchmarks [21, 22] do not distinguish task-instance memorization from genuine user-preference-driven personalization.

To address these gaps, we introduce a novel Personalized Robotic Manipulation Benchmark (LIBERO-PeRM), built using the popular LIBERO [23] framework. We elevate user preferences to a fundamental building block alongside tasks and layouts. Following the taxonomy of human preferences [24], we define a controlled taxonomy of observable preferences in simulation, covering Motion Preference (trajectory constraints), Skill Preference (task strategies), and User Profile (seman tic biases). The benchmark is restricted to preferences that can be expressed in language, instantiated without changing the original task goal, evaluated automatically, and reused across tasks. Within this scope, we provide 37 single-preference instances and 18 two-preference combinations.

LIBERO-PeRM features an extensible procedural generation pipeline that automatically instantiates tasks and their preference-conditioned variants within shared layouts. To support systematic training and evaluation, the pipeline simultaneously produces explicit preference satisfaction metrics and rich subtask-level metadata, including language instructions, goal images, and physical attributes.

For benchmarking, we construct five task suites with 312 preference-conditioned tasks and pipeline generated, human-reviewed demonstrations, each targeting a core capability in personalized manipu lation (Figure 1, top). LIBERO-SinglePref evaluates single-preference learning; LIBERO-PrefScale evaluates scaling across data sizes, tasks, and layouts; LIBERO-PrefConflict evaluates preferencepreference and task-preference conflicts; LIBERO-PrefCompose evaluates multi-preference composition; and LIBERO-ComprePref evaluates realistic preference-conditioned reasoning based on contextual language. In LIBERO-PrefScale, we further provide no-preference data, unseen layouts, and different tasks to support in-domain evaluation, task/layout generalization, and preference-level generalization over different tasks in unseen layouts.

Table 1: Comparison to existing manipulation benchmarks. Unlike prior benchmarks, LIBERO-PeRM targets personalized manipulation with large-scale preference-conditioned demonstrations and rich metadata, including subtask goals, subgoal images, trajectory statistics, and preference-satisfaction signals.
<table><tr><td>Dataset</td><td># Traj.</td><td># Tasks</td><td>Metadata</td><td>Pref. Data</td><td>Pref. Metric</td><td>Targeted Ability</td><td>Data Collection</td></tr><tr><td>ManiSkill2[25]</td><td>30k</td><td>20</td><td>x</td><td>x</td><td>x</td><td>Generalized, Sim-to-real Manipulation</td><td>TAMP &amp; MPC &amp; RL</td></tr><tr><td>RLBench[26]</td><td>2.7k</td><td>100</td><td>x</td><td>X</td><td>X</td><td>Multiple general Manipulation types</td><td>Waypoints &amp; Planners</td></tr><tr><td>LIBERO[23]</td><td>6.5k</td><td>130</td><td>X</td><td>x</td><td>x</td><td>Lifelong Robot Manipulation</td><td>Teleoperation</td></tr><tr><td>CALVIN[27]</td><td>37.5k</td><td>34</td><td></td><td>x</td><td>x</td><td>Long-horizon Manipulation</td><td>Teleoperation</td></tr><tr><td>VLABench[28]</td><td>1.6k</td><td>100</td><td></td><td>x</td><td>x</td><td>Long-Horizon Reasoning Manipulation</td><td>Autonomous</td></tr><tr><td>RoboCasa[29]</td><td>77k</td><td>100</td><td></td><td>x</td><td>x</td><td>Generalized, Sim-to-real Manipulation</td><td>Teleoperation &amp; Aug.</td></tr><tr><td>RoboTwin 2.0[30]</td><td>100k</td><td>50</td><td></td><td>X</td><td>X</td><td>Generalized, Sim-to-real Manipulation</td><td>Autonomous</td></tr><tr><td>InternVLA-M1[31]</td><td>244k</td><td>200</td><td></td><td>x</td><td></td><td>Long-horizon Manipulation</td><td>Autonomous</td></tr><tr><td>InternData-A1[32]</td><td>630k</td><td>70</td><td></td><td>X</td><td>X</td><td>Multiple general Manipulation types</td><td>Autonomous</td></tr><tr><td>RoboMIND[33]</td><td>107k</td><td>479</td><td></td><td>X</td><td>X</td><td>Multi-embodiment Manipulation</td><td>Teleoperation</td></tr><tr><td>RoboCerebra[34]</td><td>1k</td><td>100</td><td></td><td>x</td><td></td><td>Long-horizon Manipulation</td><td>Teleoperation</td></tr><tr><td>LIBERO-OOD[35]</td><td></td><td>20</td><td></td><td>X</td><td></td><td>Generalized Manipulation</td><td>Derived from LIBERO</td></tr><tr><td>LIBERO-Plus[36]</td><td>22.4k</td><td></td><td></td><td>x</td><td>X</td><td>Generalized Manipulation</td><td>Autonomous</td></tr><tr><td>LIBERO-Pro[37]</td><td></td><td></td><td></td><td>x</td><td>X</td><td>Generalized Manipulation</td><td></td></tr><tr><td>LIBERO-X[38]</td><td>2.5k</td><td>600</td><td></td><td>x</td><td>x</td><td>Generalized Manipulation</td><td>Teleoperation</td></tr><tr><td>LIBERO-Para[39]</td><td></td><td>10</td><td>x</td><td>x</td><td>X</td><td>Generalized Manipulation</td><td>Autonomous</td></tr><tr><td>LIBERO-PeRM(Ours)</td><td>40.8k</td><td>312</td><td></td><td>V</td><td></td><td>Personalized Manipulation</td><td>Autonomous</td></tr></table>

We use LIBERO-PeRM to conduct an initial study of five key research topics in personalized manipulation (see bottom row of Figure 1): 1) how policies follow preferences in zero-shot settings; 2) how policies learn single preferences from demonstrations; 3) how efficiently policies acquire preferences as demonstrations increase; 4) how learned preferences generalize across layouts and tasks; and 5) how policies compose multiple preferences and behave under conflicts. These topics organize our experiments and highlight the core capabilities that robots must develop for reliable personalization.

Our contributions and findings are as follows:

• We introduce a controlled formulation and taxonomy of observable preferences in simulation, together with an automated pipeline that produces preference-conditioned demonstrations, rich metadata, and preference-satisfaction metrics.

• We construct five task suites, LIBERO-SinglePref, LIBERO-PrefScale, LIBERO-PrefConflict, LIBERO-PrefCompose, LIBERO-ComprePref, comprising 312 preference-conditioned tasks with pipeline-generated, human-reviewed demonstrations and evaluation protocols for controlled personalization studies.

• Extensive experiments show that current VLA policies struggle with zero-shot personalization, while SFT makes preferences learnable. Mixed training improves no-preference task success relative to sequential adaptation and contextual-prompt task success and preference satisfaction relative to preference-only LoRA, but weakens motion-preference learning; broader layout diversity and matched no-preference data also help, while preference generalization, decomposition, and execution under conflicts remain difficult.

## 2 Related Work

Robotic Manipulation Benchmark. Many benchmarks have been developed to evaluate different aspects of robotic manipulation, including cross-embodiment generalization [7, 32, 33], generalized manipulation [26, 37, 39], long-horizon reasoning [27, 28, 31, 34] [40, 41], and sim-to-real transfer [25, 29, 30]. Among them, LIBERO [23] has emerged as a widely adopted benchmark due to its high-quality demonstrations and standardized data generation pipeline, primarily focusing on lifelong learning and task generalization. However, existing benchmarks largely focus on task completion, without explicitly modeling user-specific preferences. In contrast, our LIBERO-PeRM targets person alized manipulation with preference-aware task suites that disentangle task success from preference satisfaction. Table 1 compares commonly used robotic manipulation benchmarks.

Personalized Robotic Manipulation. Prior work uses personalization to cover heterogeneous settings, leaving the scope of personalized manipulation unclear. Some studies frame it as trajectory-level preference alignment via human feedback, reward signals [10, 42] [43, 44], RL post-training [45, 46, 47] [48], or implicit preference optimization [49, 50, 51]. Others study multiuser style adaptation [11] [52], object-receptacle pairing [12], or helpful action prediction [13]. In contrast, personalized manipulation requires an operational formulation for preference-conditioned physical action. We address this gap with LIBERO-PeRM, which provides a controlled taxonomy of observable preferences in simulation and five preference-aware task suites targeting key personalization aspects.

![](images/8376ca9c02f1e767d21cb2a36329bd4b84a7282e1c0ee81fc5a5604579cd034f.jpg)  
Figure 2: LIBERO-PeRM dataset statistics. (a) Distribution of tasks by task suite. (b) Distribution of tasks by preference type; No Preference denotes generated tasks from LIBERO-PeRM. (c) Distribution of preference instances by preference type. (d) Trajectory length distribution. (e) Distribution of skill numbers in a task.

Vision-Language-Action (VLA) Models. Vision-Language-Action (VLA) models [2, 4, 6, 53, 54, 55] [56] have shown strong reasoning and generalization abilities by leveraging VLM backbones [57, 58, 59], large-scale multimodal instruction datasets [60, 61, 62], and diverse robotic datasets [7, 32, 51, 63] [64], alongside advances in VLM-based spatial reasoning [65], policy guidance [66, 67], and tactile- and force-aware robot learning [68]. However, existing benchmarks [23, 27] mainly measure task success, leaving preference alignment underexplored despite its importance for home robots, where users may specify not only what task to perform but also how it should be executed. In this work, we introduce LIBERO-PeRM, which evaluates five key personalization topics through preference-injected trajectories, natural-language preference instructions, and ground-truth preference satisfaction matrices. For hierarchical VLAs [69, 70], LIBERO-PeRM further provides subtask decompositions and subgoal-image annotations, covering the key signals needed to evaluate personalized manipulation.

## 3 The LIBERO-PeRM Benchmark

This section introduces the four components of LIBERO-PeRM. Section 3.1 defines what preferences are; Section 3.2 describes the procedural preference data generation pipeline; Section 3.3 defines the five task suites used for benchmarking; and Section 3.4 defines the evaluation methodologies.

## 3.1 Preference Formulation

Preference describes the relative desirability among alternatives [71]. Personalized robotic manipulation, therefore, requires a policy to select, among feasible trajectories that complete a requested task, one that also satisfies the user’s preference. A preference is a variable independent of the task goal: it constrains how the same goal is achieved rather than replacing the goal. Natural language is our interface for both variables, but this independence distinguishes the setting from merely redefining a language-conditioned task. We organize preferences by the part of execution they constrain: motion, skill, or subtask-level choices. Specifically, they are:

• Motion Preference. This type of preference constrains the physical properties of the trajectory while preserving task completion. For the task “put the mug on the table”, the preference “avoid a particular area ofthe table” induces a trajectory-level distance constraint, requiring the robot to keep the end-effector and mug away from that region during execution. We denote this kind of preference as P<sub>motion</sub>.

• Skill Preference. This type of preference alters the execution behavior of a specific skill c while preserving task completion. For the task “put the mug on the table”, the preference “orient the handle outward” induces a pose-constrained placement strategy. When executing the placement skill, the robot must rotate the mug during placement so that the handle faces outward. We denote this kind of preference as $P _ { \mathrm { s k i l l } }$

![](images/50bacfc4dc8b20b6df9bfe8db6ade0818eb3679ac1d833ac01c8d7a14f31a49d.jpg)  
Figure 3: Data generation pipeline for personalized manipulation. Building on the LIBERO framework, the pipeline first generates preference-conditioned data, including preferences, conditions, subtasks, and trajectories, and then derives preference metrics and metadata.

• User Profile. This type of preference uses a user profile or context as background knowledge to guide subtask-level semantic choices, such as selecting a target object or goal region. For the task “put the mug on the table”, the user profile “prefers to protect table surfaces” induces a target-selection constraint, causing the policy to choose a coaster as the support target while preserving the original task goal. We denote this kind of preference as $P _ { \mathrm { p r o f i l e } } .$

Together, these three types provide a controlled taxonomy of observable preferences in simulation, from user-profile conditioning and skill-specific execution to motion-level control. The current benchmark does not model implicit inference from interaction history, non-linguistic preference signals, or open-ended preferences that lack an automatic execution metric.

Policy input and output. Let $\ell _ { \mathrm { t a s k } }$ and $\ell _ { \mathrm { p r e f } }$ denote the task and preference language. We concatenate them with text labels into the prompt “Task: {task\_language}. Preference: {preference\_language}.”, denoted as $\ell _ { \mathrm { t a s k } } \parallel \ell _ { \mathrm { p r e f } } .$ At each timestep, the policy receives AgentView and Eye-in-Hand images, $I _ { t } ^ { \mathrm { a g e n t } }$ and $I _ { t } ^ { \mathrm { h a n d } }$ , and outputs an action: $\pi _ { \boldsymbol { \theta } } ( a _ { t } \mid I _ { t } ^ { \mathrm { a g e n t } } , I _ { t } ^ { \mathrm { h a n d } } , \ell _ { \mathrm { t a s k } } \parallel \ell _ { \mathrm { p r e f } } )$

Preference Instances. With the three preference types defined above, we provide 37 single-preference instances, including 8 Motion Preference instances, 14 Skill Preference instances, and 15 User Profile instances. Appendix B, Table 6, provides the complete list. We further construct 18 multi-preference combinations from these instances. Figure 2(c) reports the resulting preference-instance distribution by type. Figure 2(b) shows the corresponding distribution in generated data tasks, including the no-preference tasks provided by LIBERO-PeRM for comparison. Detailed preference-selection criteria are provided in Appendix B.

## 3.2 Preference-Conditioned Data Generation Pipeline

Building upon the LIBERO framework, LIBERO-PeRM provides an extensible procedural datageneration pipeline for preference learning and evaluation, as illustrated in Figure 3; model-assisted generation and validation details are given in Appendix C. The pipeline consists of three stages: (1) Preference Selection and Layout Modification; (2) Preference-Conditioned Subtask and Trajectory Generation; and (3) Satisfaction Metric Generation.

Preference Selection and Layout Modification. The user first selects a preference instance and an initial LIBERO layout [23]. Given these inputs, the pipeline uses a VLM [72] to minimally modify the layout so the selected preference can be operationalized while limiting changes to the original scene. For Motion Preference and Skill Preference preferences, the layout and task language remain unchanged; the VLM only selects the target object or region, e.g., choosing the stove as the object to avoid for “avoid certain object”. For User Profile preferences, which may change the subtask plan, the pipeline makes minimal edits, such as adding a red object of the same category for a red-object preference, so preference-conditioned and no-preference trajectories can be generated in the same setting. The task language is also abstracted to cover both settings, ensuring that behavioral differences are driven by the preference rather than unrelated scene changes.

Preference-Conditioned Subtask and Trajectory Generation. After the layout and task language are fixed, the pipeline uses a VLM with the task language and preference language to generate both a no-preference subtask plan and a preference-conditioned subtask plan. It then calls a library of pre-recorded heuristic skills to generate trajectory data. To construct each heuristic skill, we first collect one human demonstration for the skill, extract key waypoints, and use an OMPL planner [73] with randomized initial states to generate many successful trajectories. The generated trajectories are then smoothed; heuristic-skill parameters are manually offset-adjusted when needed. Empirically, we find more than 72% of skills require no offset adjustment across any task, suggesting the VLM-based edit is already very effective. Preferences are applied at different stages according to their type: User Profile affects subtask-plan generation via a prompt; Skill Preference affects skill selection, such as using pick\_handle instead of a regular pick skill; and Motion Preference affects skill-level trajectory generation, such as reducing OMPL-sampled waypoints for the “speed up” preference.

Satisfaction Metric Generation. After preference-conditioned trajectories are generated, the pipeline constructs explicit preference satisfaction metrics for each selected preference. The form of each metric depends on the preference type. For Motion Preference, we directly compute satisfaction functions from trajectory quantities, such as the distance between the robot trajectory and the target object for the preference “avoid certain object”. For Skill Preference, we evaluate key waypoints and use VLM-based image checks to verify skill-specific execution, such as whether the object is grasped by its handle. For User Profile, we monitor the executed subtask plan, including which objects are manipulated, to verify that the plan adheres to the profile.

Because User Profile yields only discrete Satisfaction Scores, namely whether the profile-conditioned subtask is satisfied, whereas Motion Preference and Skill Preference can yield continuous Satisfaction Scores, we construct a mapping from Satisfaction Scores to Discrete Satisfaction Labels using trajectory statistics and human inspection; Appendix D details the review protocol. Specifically, each episode is mapped to a fixed four-level Discrete Satisfaction Label $d \in \{ 0 , 1 / 4 , 3 / 4 , \overset { \cdot } { 1 } \}$ , independent of the number of pipeline-generated subtasks. A continuous Satisfaction Score, such as speed, is binned by trajectory statistics and then incorporated into this label. Raw Satisfaction Scores provide denser feedback for training, while Discrete Satisfaction Labels provide a more interpretable evaluation signal at inference time.

Initial Human Validation. To assess the real-world plausibility of the selected preferences and the human-likeness of the generated demonstrations, we conducted a survey with 15 participants using a 10-point scale. Across all preference categories, the demonstrations received an average human-likeness score of 9.11, and the preferences received an average real-world plausibility score of 8.35. These results provide initial human validation of the benchmark design; Appendix D reports the protocol and category-level results.

## 3.3 Task Suites

Using the pipeline above, we generate five task suites: LIBERO-SinglePref, LIBERO-PrefScale, LIBERO-PrefConflict, LIBERO-PrefCompose, and LIBERO-ComprePref. We follow LIBERO terminology and call the inseparable combination of a task instruction, scene layout, and task-goal condition a mission. A LIBERO-PeRM task pairs one mission with one preference condition, which may be a specified preference or no preference; the 312 tasks include both preferenceconditioned and no-preference tasks (see Figure 2(b) for the proportions), with more than 100 trajectories per task on average and 40.8K rollout episodes in total. Randomized initial object positions, paraphrased instructions, and repeated executions increase rollout diversity but do not create additional tasks. The release prioritizes quality-controlled coverage over claiming the largest trajectory count; the procedural pipeline can extend it with more initial states and missions. Figure 2(ac) shows distributions over preference instances, task suites, and preference types, including nopreference tasks for comparison. Figure 2(d) shows that episode horizons and trajectory fluency remain comparable to LIBERO. Each suite isolates a core capability for personalized manipulation, as summarized in Figure 1.

LIBERO-SinglePref. This suite evaluates single-preference learning and execution. It contains 100 tasks and serves as the foundation for evaluating personalization. LIBERO-PrefScale. Thi suite evaluates how preference learning scales across demonstrations, tasks, and layouts. It supports controlled analysis of preference acquisition and generalization across tasks and layouts. It contains 100 tasks. LIBERO-PrefConflict. This suite evaluates policy behavior when preferences conflict with other preferences or with the task goal. Preference-preference conflicts pair the same layout and task with mutually conflicting single preferences, while task-preference conflicts pair no-preference task data with task-incompatible preference data. It contains 48 tasks. LIBERO-PrefCompose. This suite evaluates multi-preference learning and preference composition in a controlled two-preference setting. For each aligned layout-task pair, it provides four conditions: no preference, $P _ { A } , P _ { B }$ , and $P _ { A } + P _ { B }$ , enabling controlled comparisons for composition and multi-preference learning. It contains 64 tasks. LIBERO-ComprePref. This suite rewrites prompts over tasks from the other suites so the preference must be inferred from contextual language rather than read from a separate explicit field. It includes explicit task-preference mixtures, where key preference words are preserved, and implicit preference reasoning, where contextual cues imply the preference without naming it. For example, “Put both the alphabet soup and the cream cheese box in the basket. Preference: speed up” becomes “Put both the alphabet soup and the cream cheese box in the basket. Guests will use them immediately.” Similarly, “Pick the mug by its handle” is conveyed as “The mug just came out ofthe microwave—the body is too hot to touch.” ComprePref reuses existing tasks, so its contextual prompts do not add to the 312-task count or the base-suite counts in Figure 2(a). Appendix E gives a worked example; Appendix G describes the release and maintenance plan.

## 3.4 Evaluation

Task and Preference Metrics. We report two types of metrics: task success rate (SR), which measures task completion, and preference satisfaction (SA), which measures how well a preference is satisfied using the Discrete Satisfaction Label from Section 3.2. The Discrete Satisfaction Label for episode n is $d _ { n } \in \{ 0 , 1 / 4 , 3 / 4 , 1 \}$ , and $\mathbb { I } [ G ( \tau _ { n } ) ]$ is its task-success indicator. We compute $\begin{array} { r } { \mathrm { S A } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } [ G ( \tau _ { n } ) ] d _ { n } } \end{array}$ , so preference satisfaction contributes only when the task succeeds.

Conflict Policy. For a task-preference conflict, completing the original task takes priority. For preference-preference conflicts, SA is computed using the higher of the two preference satisfaction scores for each episode, with task-failure episodes contributing zero satisfaction. These scoring conventions probe execution under inconsistent instructions; they do not evaluate conflict detection, preference weighting, or when a policy should ask the user for guidance.

Generalization. We evaluate in-domain, task-or-layout, and preference-level generalization using SR and SA. In-domain evaluation keeps the training tasks, preferences, and layouts with small object-position perturbations. Task-or-layout generalization changes either the task or layout, while preference-level generalization changes tasks under unseen layouts while keeping the preference unchanged.

## 4 Experiments

We conduct experiments on the five task suites introduced in Section 3.3 to evaluate zero-shot personalization, preference learning ability, preference learning efficiency, preference generalization, composition, and behavior under conflicts. Our experiments focus on the following research questions:

Q1. Can off-the-shelf policies follow preferences without preference-specific training?

Q2. Can supervised fine-tuning teach policies to satisfy preferences on seen tasks and layouts?

Q3. How does performance change as the number of demonstrations increases?

Q4. Do learned preferences generalize across different generalization levels?

Q5. Can a policy compose multiple preferences, and how does it behave under incompatible preferences?

## 4.1 Experimental Setup

We select six representative VLA policies for evaluation: OpenVLA-OFT [53], $\pi _ { 0 } – \mathrm { f a s t } \ [ 1 ] , \pi _ { 0 . 5 } \ [ 6 ]$ GR00T-N1.6 [4], SmolVLA [74], and UniVLA [8]. For non-zero-shot experiments, we mainly use $\pi _ { 0 . 5 } ,$ , our strongest zero-shot baseline, as shown in Table 2. Training details are provided in

Table 2: Zero-shot preference on LIBERO-SinglePref. For each preference-conditioned column we report task success rate (SR, %) and preference satisfaction (SA, %); No-Pref reports SR only.
<table><tr><td rowspan="2">Method</td><td>No-Pref</td><td colspan="2">All-Pref</td><td colspan="2">Motion Preference</td><td colspan="2">Skill Preference</td><td colspan="2">User Profile</td></tr><tr><td>SR</td><td colspan="2">SR SA</td><td colspan="2">SR SA</td><td colspan="2">SR SA</td><td colspan="2">SR SA</td></tr><tr><td>OpenVLA-OFT [53]</td><td>91.2</td><td>40.00</td><td>4.53</td><td>70.00</td><td>7.30</td><td>64.10</td><td>3.20</td><td>3.92</td><td>3.92</td></tr><tr><td>π0-fast [1]</td><td>86.88</td><td>51.25</td><td>30.37</td><td>80.00</td><td>27.60</td><td>73.08</td><td>18.27</td><td>17.65</td><td>17.65</td></tr><tr><td>π0.5 [6]</td><td>97.6</td><td>66.67</td><td>37.92</td><td>97.50</td><td>24.38</td><td>100.00</td><td>25.00</td><td>19.12</td><td>19.12</td></tr><tr><td>GR00T-N1.6 [4]</td><td>88.3</td><td>51.67</td><td>27.45</td><td>90.00</td><td>20.70</td><td>76.92</td><td>11.54</td><td>9.80</td><td>9.80</td></tr><tr><td>SmolVLA [74]</td><td>60.0</td><td>4.17</td><td>2.03</td><td>16.67</td><td>4.17</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>UniVLA [8]</td><td>80.8</td><td>45.83</td><td>21.52</td><td>90.00</td><td>10.17</td><td>71.79</td><td>3.59</td><td>0.00</td><td>0.00</td></tr></table>

Table 3: Fine-tuned preference-learning performance on LIBERO-PeRM, reported as mean ± standard deviation over 10 evaluation runs with fixed checkpoints. SR is task success rate (%), and SA is preference satisfaction (%); No-Pref reports SR only. All variants start from the $\pi _ { 0 . 5 }$ base checkpoint. All-Pref evaluates one checkpoint jointly trained across Motion, Skill, and User-Profile tasks. Mix-LoRA and Seq-LoRA additionally use LIBERO data: Seq-LoRA first fine-tunes on LIBERO and then on LIBERO-SinglePref, while Mix-LoRA fine-tunes on their mixture. <sup>∗</sup> indicates the same task and preference under an unseen layout.
<table><tr><td rowspan="2">Method</td><td>No-Pref</td><td colspan="2">All-Pref</td><td colspan="2">Motion Preference</td><td colspan="2">Skill Preference</td><td colspan="2">User Profile</td><td colspan="2">LIBERO-ComprePref</td></tr><tr><td>SR</td><td>SR</td><td>SA</td><td>SR</td><td>SA</td><td>SR</td><td>SA</td><td>SR</td><td>SA</td><td>SR</td><td>SA</td></tr><tr><td>π0.5-LoRA</td><td></td><td>84.96 ±1.06</td><td>82.48 ±1.02</td><td>90.00 ±2.85</td><td>85.37 ±2.66</td><td>78.86 ±1.44</td><td>76.67 ±1.40</td><td>86.67 ±1.49</td><td>85.22 ±1.47</td><td>44.00 ±0.71</td><td>42.64 ±0.69</td></tr><tr><td>π0.5-Mix-LoRA</td><td>91.00 ±0.70 52.33</td><td>84.55 ±1.94 74.75</td><td>84.45 ±2.04 74.73</td><td>66.88 ±3.56 67.50</td><td>66.06 ±4.38</td><td>89.29 ±2.82</td><td>89.29 ±2.82</td><td>91.33 ±3.42</td><td>91.33 ±3.42</td><td>72.83 ±0.95</td><td>70.07 ±0.91</td></tr><tr><td>π0.5-Seq-LoRA</td><td>±0.37</td><td>±1.97 37.49</td><td>±2.12 28.36</td><td>±5.68 55.00</td><td>67.44 ±6.52 43.43</td><td>72.14 ±2.04 28.57</td><td>72.14 ±2.04 26.94</td><td>81.00 ±2.79</td><td>81.00 ±2.79 20.58</td><td>70.00 ±1.32</td><td>68.25 ±1.29</td></tr><tr><td> $\pi _ { 0 . 5 } – \mathrm { L o R A } ^ { * }$ </td><td></td><td>±3.27</td><td>±2.55</td><td>±7.53</td><td>±7.02</td><td>±3.79</td><td></td><td>±3.57</td><td>34.00 ±5.60 ±3.39</td><td></td><td></td></tr></table>

Appendix A.2. For evaluation, we adopt SR and SA as defined in Section 3.4. Table 3 additionally reports the mean and standard deviation over 10 independent evaluation runs with fixed checkpoints; these error bars capture evaluation randomness rather than training-seed variation.

## 4.2 Experimental Results

Q1: Zero-shot Personalization. Q1 asks whether off-the-shelf VLA policies can follow user preferences without preference-specific training. Table 2 evaluates six policies on the LIBERO-SinglePref task suite. The no-preference column uses the original LIBERO task language with minimally edited LIBERO-SinglePref layouts.

Finding. Table 2 shows that minimal layout edits from our pipeline preserve no preference zero-shot task performance, and existing SOTA VLA models fail to follow preferences in the preferenceconditioned zero-shot setting.

Most policies maintain high no-preference SR, with $\pi _ { 0 . 5 }$ reaching 97.6%, indicating that the edited layouts preserve the LIBERO task distribution. Thus, low SA mainly reflects preference-following failure rather than layout-editing artifacts. Notably, $\pi _ { 0 . 5 }$ reaches high SR on Motion and Skill Preference (97.50% and 100.00%), but low SA (24.38% and 25.00%), showing that it completes tasks while ignoring preference. Executing with a User Profile is harder since it changes the target object or semantic choice, reducing $\pi _ { 0 . 5 }$ SR to 19.12% and causing near-zero SR for weaker policies.

Q2: Learning Ability. Table 3 reports in-domain and unseen-layout evaluation results under LoRAbased supervised fine-tuning (SFT). The $\pi _ { 0 . 5 ^ { - } } \mathrm { L o R A }$ setting is used only on LIBERO-SinglePref to validate the effectiveness of our dataset.

Finding. Sequential training degrades no-preference and preference-conditioned performance. Mixed training instead yields a category-dependent trade-off: it better retains no-preference behavior and contextual-prompt robustness, but weakens motion preference relative to preference-only LoRA.

Table 3 compares the No-Pref and All-Pref settings. $\pi _ { 0 . 5 }$ -Seq-LoRA obtains 52.33±0.37% No-Pref SR and 74.75±1.97% All-Pref SR, whereas $\pi _ { 0 . 5 } \cdot$ -Mix-LoRA retains 91.00±0.70% No-Pref SR and reaches 84.55±1.94% All-Pref SR. The LIBERO-ComprePref columns further evaluate robustness to contextual prompt variation: Mix-LoRA reaches 72.83±0.95% SR and 70.07±0.91% SA, compared with 44.00±0.71% and 42.64±0.69% for preference-only LoRA. However, Mix-LoRA reduces

![](images/5e75726b788b72ecd68cdc3eb914bddc4c6beb2704afe08b46529ce0d21163b2.jpg)

![](images/9d92bdf074302127dbe85c1be31d94fc25f60291423f89af60859d09c4bd6ba1.jpg)  
Figure 4: Preference scaling behavior under increasing data scale. We report task success rate (SR) for in-domain (blue), unseen-layout (red), and unseen-layout-with-different-task (green) generalization. Solid lines show the main scaling trend, while dashed branches compare L-series alternatives at the same 2.25K scale: L1 (⋆) emphasizes demonstration density over fewer layouts, L2 is the solid-line intermediate setting, L3 (△) emphasizes broader layout coverage with fewer demonstrations per layout, and L4 (⋄) adds matched no-preference missions for preference-layout decoupling. As the data scale increases, the number of training layouts also increases to provide broader layout coverage.

Motion Preference SR from 90.00±2.85% to 66.88±3.56%. The results therefore compare strategies for jointly learning preference-conditioned and preference-agnostic behavior; they do not establish mixed training as uniformly superior.

## Q3: Learning Efficiency.

This question evaluates how data scale affects learning efficiency under the same task and preference distribution. In Figure 4, the blue curves show in-domain scaling as the number of training demonstrations increases. At the 2.25K scale, L1-L4 isolate data-composition effects under a comparable budget. L1, L2, and L3 use $K = 2 , K = 3 ,$ , and K = 5 layouts, respectively, trading off layout diversity against demonstrations per layout. L4 uses the same K = 3 layout setting as L2 but adds matched no-preference missions, testing whether no-preference data helps decouple preference-conditioned behavior from task/layout cues.

Finding. Across all data scales, in-domain preference-conditioned behavior is learned consistently well. For in-domain preference learning, increasing layout diversity is more effective than simply increasing the number of demonstrations. Notably, adding matched no-preference data further improves learning even without increasing layout diversity, suggesting that such data helps decouple preference-conditioned behavior from task/layout-specific cues.

Figure 4 provides detailed evidence for these observations. First, the in-domain SR and SA curves are closely aligned across data scales, indicating strong in-domain preference learning: successful executions usually also satisfy the specified preferences. Second, at the fixed 2.25K scale, L1-L3 use the same demonstration budget but increase the number of layouts, and the improvement from L1 to L3 shows that broader layout coverage is more effective than denser demonstrations within fewer layouts. Third, L4 uses the same layout setting as L2 but adds matched no-preference missions; its improvement over L2 suggests that no-preference data benefits learning beyond increasing layout diversity, likely by helping separate task execution from preference-conditioned behavior.

Q4: Generalization. This question evaluates two forms of generalization: task-and-layout generalization, where either the task or layout changes, and preference-level generalization, where both tasks and layouts change while the preference remains unchanged. In Figure 4, the red curves denote unseen-layout generalization, and the green curves denote unseen-layout-with-different-task generalization. Both settings use the same L1-L4 controlled comparisons at the matched 2.25K scale.

Finding. Preference-conditioned behavior is consistently learned across settings, but jointly achieving task success and preference satisfaction remains challenging under generalization. Unseen-layout generalization benefits from increased data scale, whereas generalization to both unseen layouts and different tasks remains limited. As shown in Figure 4, SR and SA are closely aligned within each setting, suggesting that successful executions usually also satisfy the specified preference. However, under generalization, especially unseen-layout-with-different-task, both metrics remain much lower than in-domain performance. This indicates that learning preference-conditioned behavior is easier than executing it successfully together with the task under new task-layout combinations.

Table 4: Preference composition tests forming $P _ { A } +$ $P _ { B }$ from $P _ { A }$ and $P _ { B } ;$ decomposition tests inferring $P _ { B }$ from $P _ { A } + P _ { B }$ and $P _ { A } .$
<table><tr><td rowspan="2">Preference Composition</td><td colspan="3">In-dom. In-dom. Eval Eval</td><td rowspan="2">SA</td></tr><tr><td>SR</td><td>SA</td><td>SR</td></tr><tr><td> $P _ { A } + P _ { B }$ </td><td>58.3</td><td>58.3</td><td>25.0</td><td>25.0</td></tr><tr><td>Preference Decomposition</td><td>In-dom. In-dom. Eval Eval</td><td></td><td></td><td></td></tr><tr><td>Eval</td><td>SR</td><td>SA</td><td>SR</td><td>SA</td></tr><tr><td> $P _ { B }$ </td><td>66.7</td><td>66.7</td><td>37.5</td><td>0.0</td></tr></table>

Table 5: Conflict performance. Preference-preference success requires completing the task while satisfying one of the two conflicting preferences; task-preference conflicts prioritize the task.
<table><tr><td colspan="3">Preference-Preference Conflict In-dom. In-dom.</td><td rowspan="2">Conf. SR</td><td rowspan="2">Conf. SA</td><td rowspan="2">Pref Dom.</td></tr><tr><td>Eval</td><td>SR</td><td>SA</td></tr><tr><td> $P _ { 1 } + P _ { 2 }$ </td><td>73.00</td><td>63.40</td><td>66.00</td><td>47.75</td><td>11.25</td></tr><tr><td>Task-Preference Conflict</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Eval</td><td>In-dom. In-dom. SR</td><td>SA</td><td>Follow task (SR)</td><td>Follow pref.</td><td>Incomplete</td></tr><tr><td> $T _ { A }$  十  $P _ { B }$ </td><td>50.00</td><td>50.00</td><td>33.3</td><td>29.2</td><td>37.5</td></tr></table>

Q5: Composition and Conflict. This question evaluates whether policies can compose preferences, decompose a target preference from joint training, and handle preference-preference or task-preference conflicts. Tables 4 and 5 report the results. For composition/decomposition, Eval SR/SA denote task success and target preference satisfaction on the evaluation condition. For conflicts, Pref-Dom. measures the gap between following either preference under a joint prompt; lower values indicate less consistent preference domination. In task-preference conflicts, Follow Task, Follow Preference, and Incomplete denote completing the task while violating the preference, following the preference while violating the task, and satisfying neither, respectively.

Finding. Preference composition is partially learnable, whereas decomposition fails. After learning $P _ { A }$ and $P _ { B }$ separately, the policy can compose them into $P _ { A } + P _ { B }$ with 25.0% Eval SR/SA. However, when asked to infer $P _ { B }$ from $\dot { P _ { A } } \dot { + } \dot { P _ { B } }$ and $P _ { A }$ , the policy obtains 37.5% Eval SR but 0.0% Eval SA, showing task success does not imply successful preference behavior execution.

Finding. Conflicting preferences reduce performance and have no dominant preference. Under preference-preference conflicts, conflict-time SR/SA drop to 66.00%/47.75%, and Pref-Dom. is only 11.25%, suggesting no clear preference dominates the other under our either-preference success rule. Under task-preference conflicts, Follow Task and Follow Preference are close, 33.3% vs. 29.2%, with 37.5% incomplete trials, indicating that conflicts often lead to failed execution rather than task- or preference-following behavior.

## 5 Conclusion

In this work, we introduce LIBERO-PeRM, a comprehensive benchmark for personalized robotic manipulation. By treating preferences as a first-class building block alongside tasks and layouts, LIBERO-PeRM provides five task suites that isolate and evaluate preference learning under singlepreference, scaling, conflict, composition, and comprehensive preference reasoning settings. Experiments show that current VLA policies struggle with zero-shot personalization, while SFT makes preferences learnable. Compared with sequential adaptation, mixed training better preserves nopreference behavior and improves contextual-prompt robustness, but it weakens motion-preference performance relative to preference-only LoRA. Preference learning benefits more from layout diversity and same-layout no-preference data than denser demonstrations, but generalization to unseen layouts and different tasks remains difficult. Composition is only partially observed, decomposition fails, and conflicts often degrade execution instead of producing resolution.

Limitations and Future Work. LIBERO-PeRM instantiates predefined, language-expressible, automatically measurable preferences in simulation. It therefore excludes implicit preference inference, interaction history, non-linguistic signals, and open-ended or evolving preferences. Its generated demonstrations may contain pipeline- and skill-library-specific biases despite randomized initial states, multiple missions per preference, and human review. The current composition suite combines only two preferences, and the conflict suite does not test conflict detection or clarification. Finally, the human survey provides only initial validation; no real-robot or longitudinal user study is included. Future work could extend LIBERO-PeRM in directions ranging from three-or-more-preference settings to non-linguistic preference signals, human-in-the-loop preference acquisition, preference weighting and when-to-ask decisions, and real-robot evaluation.

## References

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, Laura Smith, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. In Proceedings ofRobotics: Science and Systems, 2025.

[2] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. $\pi _ { 0 . 6 } ^ { * } \colon$ a VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[3] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In Proceedings ofThe 8th Conference on Robot Learning, 2025.

[4] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[5] Wei Zhao, Pengxiang Ding, Zhang Min, Zhefei Gong, Shuanghao Bai, Han Zhao, and Donglin Wang. Vlas: Vision-language-action model with speech instructions for customized robot manipulation. In International Conference on Learning Representations, volume 2025, pages 51676–51693, 2025.

[6] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, brian ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. $\pi _ { 0 . 5 } { : }$ a vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025.

[7] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Schölkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter Büchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, João Silvério, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi Jim Fan,

Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick Tree Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Martín-Martín, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, and Zipeng Lin. Open x-embodiment: Robotic learning datasets and rt-x models : Open x-embodiment collaboration0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903, 2024.

[8] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Learning to Act Anywhere with Task-centric Latent Actions. In Proceedings ofRobotics: Science and Systems, 2025.

[9] Rajat Kumar Jenamani, Tom Silver, Ben Dodson, Shiqin Tong, Anthony Song, Yuting Yang, Ziang Liu, Benjamin Howe, Aimee Whitneck, and Tapomayukh Bhattacharjee. FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization. In Proceedings of Robotics: Science and Systems, 2025.

[10] Sriyash Poddar, Yanming Wan, Hamish Ivison, Abhishek Gupta, and Natasha Jaques. Personalizing reinforcement learning from human feedback with variational preference learning. In Advances in Neural Information Processing Systems, volume 37, pages 52516–52544, 2024.

[11] Heramb Nemlekar, Robert Ramirez Sanchez, and Dylan P. Losey. Pecan: Personalizing robot behaviors through a learned canonical space. J. Hum.-Robot Interact., 14(4), 2025.

[12] Dongge Han, Trevor McInroe, Adam Jelley, Stefano V. Albrecht, Peter Bell, and Amos Storkey. LLM-personalize: Aligning LLM planners with human preferences via reinforced self-training for housekeeping robots. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 1465–1474, 2025.

[13] Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, and Dorsa Sadigh. Provox: Personalization and proactive planning for situated human-robot collaboration. IEEE Robotics and Automation Letters (RA-L), 2025.

[14] Zhaxizhuoma, Pengan Chen, Ziniu Wu, Jiawei Sun, Dong Wang, Peng Zhou, Nieqing Cao, Yan Ding, Bin Zhao, and Xuelong Li. Alignbot: Aligning vlm-powered customized task planning with user reminders through fine-tuning for household robots. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 12549–12556, 2025.

[15] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, Zhizheng Zhang, and He Wang. GraspVLA: a grasping foundation model pre-trained on billion-scale synthetic action data. In 9th Annual Conference on Robot Learning, 2025.

[16] Ruiqi Wang, Dezhong Zhao, Dayoon Suh, Ziqin Yuan, Guohua Chen, and Byung-Cheol Min. Personalization in human-robot interaction through preference-based action representation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 7377–7384, 2025.

[17] Renpu Liu, Peng Wang, Donghao Li, Cong Shen, and Jing Yang. A shared low-rank adaptation approach to personalized rlhf. In Proceedings ofThe 28th International Conference on Artificial Intelligence and Statistics, 2025.

[18] Kimin Lee, Laura Smith, Anca Dragan, and Pieter Abbeel. B-pref: Benchmarking preferencebased reinforcement learning. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[19] Filippo Ziliotto, Jelin Raphael Akkara, Alessandro Daniele, Lamberto Ballan, Luciano Serafini, and Tommaso Campari. Personal: Towards a comprehensive benchmark for personalized embodied agents. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 13193–13200, 2026.

[20] Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser. Tidybot: personalized robot assistance with large language models. Autonomous Robots, 47:1087–1102, 2023.

[21] Haoquan Fang, Markus Grotz, Wilbert Pumacay, Yi Ru Wang, Dieter Fox, Ranjay Krishna, and Jiafei Duan. SAM2Act: Integrating visual foundation model with a memory architecture for robotic manipulation. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[22] Taijing Chen, Sateesh Kumar, Junhong Xu, Georgios Pavlakos, Joydeep Biswas, and Roberto Martín-Martín. Searching in space and time: Unified memory-action loops for open-world object retrieval. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 6010–6017, 2026.

[23] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, volume 36, pages 44776–44791, 2023.

[24] Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, Nesreen K. Ahmed, and Yu Wang. Personalization of large language models: A survey. Transactions on Machine Learning Research, 2025.

[25] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In The Eleventh International Conference on Learning Representations, 2023.

[26] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2): 3019–3026, 2020.

[27] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

[28] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, and Xipeng Qiu. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[29] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024.

[30] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Forty-third International Conference on Machine Learning, 2026.

[31] Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, et al. Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy. arXiv preprint arXiv:2510.13778, 2025.

[32] Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, and Jiangmiao Pang. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 976–985, 2026.

[33] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, Shichao Fan, Xinhua Wang, Fei Liao, Zhen Zhao, Guangyu Li, Zhao Jin, Lecheng Wang, Jilei Mao, Ning Liu, Pei Ren, Qiang Zhang, Yaoxu Lyu, Mengzhen Liu, He Jingyang, Yulin Luo, Zeyu Gao, Chenxuan Li, Chenyang Gu, Yankai Fu, Di Wu, Xingyu Wang, Sixiang Chen, Zhenyu Wang, Pengju An, Siyuan Qian, Shanghang Zhang, and Jian Tang. RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation. In Proceedings ofRobotics: Science and Systems, 2025.

[34] Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, and Si Liu. Robocerebra: A large-scale benchmark for long-horizon robotic manipulation evaluation. In Advances in Neural Information Processing Systems, 2025.

[35] Quanyi Li. Vlas are confined yet capable of generalizing to novel tasks. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26, pages 7583–7591, 2026.

[36] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: A progressive robustness benchmark for visual-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 38574– 38583, 2026.

[37] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-languageaction models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[38] Guodong Wang, Chenkai Zhang, Qingjie Liu, Jinjin Zhang, Jiancheng Cai, Junjie Liu, and Xinmin Liu. LIBERO-X: Robustness Litmus for Vision-Language-Action Models. In Proceedings of Robotics: Science and Systems, 2026.

[39] Chanyoung Kim, Minwoo Kim, Minseok Kang, Hyunwoo Kim, and Dahuin Jung. Libero-para: A diagnostic benchmark and metrics for paraphrase robustness in vla models. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, 2026.

[40] Xiaopan Zhang, Hao Qin, Fuquan Wang, Yue Dong, and Jiachen Li. Lamma-p: Generalizable multi-agent long-horizon task allocation and planning with lm-driven pddl planner. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

[41] Piyush Gupta, Sangjae Bae, Jiachen Li, and David Isele. Scale-plan: Scalable language-enabled task planning for heterogeneous multi-robot teams. In 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

[42] Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Bean, Katerina Margatina, Juan Ciro, Rafael Mosquera, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott Hale. The prism alignment dataset: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. In Advances in Neural Information Processing Systems, volume 37, pages 105236–105344, 2024.

[43] Udita Ghosh, Dripta S. Raychaudhuri, Jiachen Li, Konstantinos Karydis, and Amit Roy-Chowdhury. Reducing oracle feedback with vision-language embeddings for preference-based rl. In IEEE International Conference on Robotics and Automation (ICRA), 2026.

[44] Udita Ghosh, Dripta S Raychaudhuri, Jiachen Li, Konstantinos Karydis, and Amit Roy-Chowdhury. Preference vlm: Leveraging vlms for scalable preference-based reinforcement learning. arXiv preprint arXiv:2502.01616, 2025.

[45] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. In 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

[46] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. In 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026.

[47] Chen Tang, Ben Abbatematteo, Jiaheng Hu, Rohan Chandra, Roberto Martín-Martín, and Peter Stone. Deep reinforcement learning for robotics: A survey of real-world successes. Annual Review ofControl, Robotics, and Autonomous Systems, 8(1):153–188, 2025.

[48] Hyeonjun Kim, Kanghoon Lee, Junho Park, Jiachen Li, and Jinkyoo Park. Human implicit preference-based policy fine-tuning for multi-agent reinforcement learning in usv swarm. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 18653–18659, 2025.

[49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741, 2023.

[50] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Proceedings of the 41st International Conference on Machine Learning, 2024.

[51] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. In Proceedings ofRobotics: Science and Systems, 2024.

[52] Zehao Wang, Huaide Jiang, Shuaiwu Dong, Yuping Wang, Hang Qiu, and Jiachen Li. Drive my way: Preference alignment of vision-language-action model for personalized driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 25204–25214, 2026.

[53] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. In Proceedings ofRobotics: Science and Systems, 2025.

[54] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Tai Wang, Ya-Qin Zhang, Jingjing Liu, and Xianyuan Zhan. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In The Fourteenth International Conference on Learning Representations, 2026.

[55] Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, et al. π : a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

[56] Xiangbo Gao, Xiukun Huang, Boyu Lu, Junge Zhang, Mengjie Mao, Jiachen Li, Wei Xiong, and Zhengzhong Tu. Neuro-symbolic drive: Rule-grounded faithful reasoning for driving vlas. arXiv preprint arXiv:2606.23938, 2026.

[57] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-e: An embodied multimodal language model. In Proceedings ofthe 40th International Conference on Machine Learning, pages 8469–8488, 2023.

[58] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R Sanketi, Grecia Salazar, Michael S Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J Joshi, Alex Irpan, brian ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: Vision-language-action models transfer web knowledge to robotic control. In 7th Annual Conference on Robot Learning, 2023.

[59] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikoł aj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan.´ Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, volume 35, pages 23716–23736, 2022.

[60] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916, 2023.

[61] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

[62] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In Advances in Neural Information Processing Systems, volume 37, pages 87310–87356, 2024.

[63] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3549–3556, 2025.

[64] Mengfei Zhao, Dihong Huang, Yikai Tang, Peihao Li, Mingxuan Yan, Ruiqi Zhuang, Yanjia Huang, Jie Wang, Hai Zhai, Tony Zhou, et al. Axis: A growable community-driven data engine for scalable robot manipulation. arXiv preprint arXiv:2607.21588, 2026.

[65] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Peihao Wang, Huaizhi Qu, Shijie Zhou, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 31054–31065, 2026.

[66] Minquan Gao, Xinyi Li, Qing Yan, Xiaojian Sun, Xiaopan Zhang, Chien-Ming Huang, and Jiachen Li. Guides: Guidance using instructor-distilled embeddings for pre-trained robot policy enhancement. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 14679–14686, 2026.

[67] Litian Gong, Fatemeh Bahrani, Yutai Zhou, Amin Banayeeanzade, Jiachen Li, and Erdem Bıyık. Autofocus-il: Vlm-based saliency maps for data-efficient visual imitation learning without extra human annotations. In International Conference on Robotics and Automation (ICRA), 2026.

[68] Shilin Shan, Chuhao Zhou, Ruize Wang, Xinyan Chen, Xiangyu Chen, Xinyu Zhou, Boyu Ma, Iris Yuxuan Hu, Jingliang Li, Celeste Yuxuan Hu, et al. Learning physical interaction: A survey of tactile-and force-aware robot learning. arXiv preprint arXiv:2608.07558, 2026.

[69] Lucy Xiaoyang Shi, Brian Ichter, Michael Robert Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, and Chelsea Finn. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[70] Mingxuan Yan, Yuping Wang, Zechun Liu, and Jiachen Li. Rdd: Retrieval-based demonstration decomposer for planner alignment in long-horizon tasks. In Advances in Neural Information Processing Systems, 2025.

[71] Klaus R. Scherer. What are emotions? and how can they be measured? Social Science Information, 44(4):695–729, 2005.

[72] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[73] Ioan A. Sucan, Mark Moll, and Lydia E. Kavraki. The open motion planning library. IEEE Robotics & Automation Magazine, 2012.

[74] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[75] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[76] Yiyou Sun, Xinyang Han, Weichen Zhang, Yuanbo Pang, Tianyu Wang, Yuhan Cao, Yixiao Huang, Chris Duroiu, Haoyun Zhang, Jeffrey Lin, et al. Agents’ last exam. arXiv preprint arXiv:2606.05405, 2026.

[77] Zehao Wang, Jianpeng Yao, Yuping Wang, Yiping Li, Jiaqi Li, Zhixu Li, Kanghoon Lee, Jinfeng Wang, Cong Liu, Zhaowei Tan, Jinkyoo Park, and Jiachen Li. Coopuq: Robust multi-robot social navigation via uncertainty-aware cooperative forecasting. In Conference on Robot Learning (CoRL), 2026.

[78] Xiaopan Zhang, Zejin Wang, Zhixu Li, Jianpeng Yao, and Jiachen Li. Commcp: Efficient multi-agent coordination via llm-based communication with conformal prediction. In 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026.

## A Additional Information

This appendix includes additional experimental details and results that support the main findings in Section 4.2.

## A.1 Ethics Statement

LIBERO-PeRM is built from procedurally generated preference-conditioned tasks, layouts, trajectories, and evaluation signals in simulation. Human feedback is used to review generated data and validate preference plausibility, but no personal data or real-world user logs are released or used as policy-training data. Its intended use is research on personalized robotic manipulation, including preference learning, generalization, composition, and conflict handling. Our dataset construction process is designed to reduce ethical and legal risk. We build on LIBERO [23] scenes and simulator assets, then generate preference-conditioned variants through minimal layout edits, procedural trajectory generation, and automatic preference metrics. Preference descriptions, satisfaction rules, and benchmark annotations are created for research use rather than copied from external products or commercial systems. Any public release should preserve attribution to upstream assets and follow the licenses and terms of use of LIBERO and other dependencies. The benchmark also has important limitations. Our preference metrics only capture observable execution outcomes in simulation and cannot represent open-ended, private, or highly subjective human preferences. As a result, LIBERO PeRM should not be treated as a substitute for human evaluation in safety-critical or real-world personalized deployment. We intend this benchmark to support controlled scientific evaluation, not direct real-world decision making.

We take responsibility for the released benchmark artifacts. The LIBERO-PeRM dataset and opensource code are released under the MIT license.

## A.2 Training Details

For non-zero-shot experiments, we further tune $\pi _ { 0 . 5 }$ with LoRA [75]. We scale the training budget with the number of rollouts: for every 4,000 rollout episodes, we use roughly 45,000 optimization steps on eight A6000 Ada GPUs with a learning rate of $\mathrm { \bar { 5 } } \times 1 0 ^ { - 5 }$ . For the main fine-tuning comparison, All-Pref denotes one checkpoint jointly trained on Motion, Skill, and User-Profile preference tasks; it is not an ensemble or an average of category-specific checkpoints. We evaluate each fixed checkpoint in 10 independent runs and report mean ± standard deviation in Table 3. The variation therefore reflects randomized evaluation rollouts, not training seeds.

## B Preference Selection Criteria

We select a preference only if it satisfies five criteria: 1) expressible in natural language; 2) can be grounded in existing LIBERO [23] scenes; 3) satisfiable without changing the task goal; 4) measurable by an automatic preference metric; and 5) reusable across multiple tasks or layouts. Examples include the Motion Preference preference “speed up”, measured by execution steps; the Skill Preference preference “pick up an object by its handle”, measured by waypoints and contact points; and the User Profile preference “prefer dairy product”, measured by the executed subgoal. We exclude preferences that require scene-external information, unsafe or task-infeasible behavior, or subjective judgments that cannot be mapped to observable execution outcomes.

Table 6 lists all 37 single-preference instances in LIBERO-SinglePref.

Table 6: Complete list of the 37 single-preference instances in LIBERO-SinglePref.
<table><tr><td>ID</td><td>Preference</td></tr><tr><td colspan="2">Motion (8 preferences)</td></tr><tr><td>1-1</td><td>Speed up</td></tr><tr><td>1-2</td><td>Slow down</td></tr><tr><td>1-3</td><td>Avoid sudden acceleration</td></tr><tr><td>1-4</td><td>Keep away from wine bottle / Keep away from stove / Keep away from the other bowl</td></tr><tr><td>1-5</td><td>Close to ground when moving</td></tr><tr><td>1-6</td><td>Move straight</td></tr><tr><td>1-7</td><td>Move higher</td></tr><tr><td>1-8</td><td>Rotate when moving</td></tr><tr><td>Skill (14 preferences)</td><td></td></tr><tr><td>2-1</td><td>Pick from right edge</td></tr><tr><td>2-2</td><td>Pick from back edge</td></tr><tr><td>2-3</td><td>Pick by handle</td></tr><tr><td>2-4</td><td>Pick from top at an angle</td></tr><tr><td>2-5</td><td>Pick from left side</td></tr><tr><td>2-6</td><td>Pick from right side</td></tr><tr><td>2-7</td><td>Pick from horizontal side</td></tr><tr><td>2-8</td><td>Push instead of pick-and-put</td></tr><tr><td>2-9</td><td>Drag instead of pick-and-put</td></tr><tr><td>2-10</td><td>Throw instead of placing</td></tr><tr><td>2-11</td><td>Pick from front</td></tr><tr><td>2-12</td><td>Rotate when placing</td></tr><tr><td>2-13</td><td>Pick from left edge</td></tr><tr><td>2-14</td><td>Lay object down when placing</td></tr><tr><td>User Profile (15 preferences)</td><td></td></tr><tr><td>3-1 Prefer green objects</td><td></td></tr><tr><td>3-2</td><td>Prefer red objects</td></tr><tr><td>3-3</td><td>Prefer yellow objects</td></tr><tr><td>3-4</td><td>Prefer blue objects</td></tr><tr><td>3-5</td><td>Prefer bigger objects</td></tr><tr><td>3-6</td><td>Upper drawer for food and drink</td></tr><tr><td>3-7</td><td>Middle drawer for kitchenware</td></tr><tr><td>3-8</td><td>Serve double portion</td></tr><tr><td>3-9</td><td>Prefer dairy products</td></tr><tr><td>3-10</td><td>Prefer sweet drinks</td></tr><tr><td>3-11</td><td>Prefer sour sauces</td></tr><tr><td>3-12</td><td>Prefer boxed objects</td></tr><tr><td>3-13</td><td>Prefer bottled objects</td></tr><tr><td>3-14</td><td>Prefer canned food</td></tr><tr><td>3-15</td><td>Prefer smaller objects</td></tr><tr><td>Total (37 preferences)</td><td></td></tr></table>

## C Model-Assisted Generation and Validation

We use Claude Code with an Opus-family model and Gemini as backends for three model-assisted stages. In layout editing, the model receives a BDDL scene specification, preference language, object geometry, and spatial constraints, and proposes a modified BDDL layout. In personalized planning, it receives the original task and preference instructions, baseline plan, available objects and skills, and a scene image, and returns a sequence of (skill, object) pairs plus required, prohibited, and ordering constraints. In image inspection, independent VLM judgments assess task completion and visually observable preference compliance from rollout frames.

Inspired by single- and multi-agent evaluations [76, 77, 78], the layout and plan candidates follow a generate-and-trial loop: each proposal is instantiated and tested in simulation, and any candidate violating the output format, physical constraints, task-success condition, or preference constraints is rejected. The observed failure is included when generating a revised candidate. The reported VLM failure rate in each stage was below 23%. For image inspection, inconsistent independent judgments are flagged for re-evaluation rather than accepted directly. Image judgments are restricted to visible properties such as contact location, approach direction, gripper orientation, and final placement. Motion speed, acceleration, trajectory smoothness, and clearance height are instead measured with trajectory-based geometric or kinematic metrics.

The following condensed prompts specify the inputs, output formats, and constraints used in the three stages. Complete prompts, model configurations, and validation logic will accompany the code.

[Layout Editing]   
Generate an initial position for {obj\_name} with size   
{obj\_dimensions} under constraint "{lan\_description}".   
Existing objects: {name}: pos=({x},{y}), size=({w},{h}), ...   
Workspace: x,y in [-0.25,0.25].   
Choose collision-free (x,y) satisfying the constraint,   
reachability, workspace margins, >=0.03m clearance,   
and layout diversity.   
Guidelines: accessible/center/reachable: within 0.10m of (0,0);   
clustered with X: 0.02-0.04m beyond collision;   
left/right: x<-0.05/x>0.05; front/back: y>0.05/y<-0.05.   
Return only: {"x":float,"y":float,"reasoning":str}   
[Personalized Planning]   
Task: {lan\_inst}   
Baseline: {task\_plan.plans}   
Preference: {high\_level\_pref}   
Existing objects: {layout.object\_set}   
Available objects: {available\_obj\_set}   
Skills: {high\_skill\_name\_and\_desc}   
Preserve the task goal and ground the preference into   
(skill,object) pairs. Add 1-5 objects only if needed and   
use geometry-compatible skills.   
Return: (1) personalized sequence;   
(2) "required", "prohibited", "order" constraints;   
(3) unchanged preference.   
Wildcards: ("skill","\_ALL")=any object;   
("\_ALL","object")=any skill; ("skill","object")=specific pair.   
Format:   
1.||(’skill’,’object’),...||   
2.||"required":[...],"prohibited":[...],"order":[...]||   
3.||{high\_level\_pref}||   
Generate {n\_task\_per\_response} plans.   
Retry feedback: {additional\_prompt}   
[Image Inspection]   
Evaluate robot-manipulation frames in reading order and   
independently judge task success and preference compliance.   
Goal: "{main\_goal}"   
Preference: "{pref\_lan}"   
Baseline: {naive\_plan\_text}

Preference plan: {task\_plan\_text}   
Subtasks/frames: {steps\_text}   
Skill semantics: {base\_skill\_description}   
Preference semantics: {pref\_variant\_description}   
Use only visible evidence: contact location, gripper orientation,   
approach direction, and final pose. Do not infer speed,   
acceleration, smoothness, or clearance height; return   
"unverifiable" when compliance is not visually observable.   
Judge completion for each subtask and compliance for each   
preference-relevant subtask.   
Output:   
Task\_Verdict:<yes|no>   
Task\_Reason:<brief justification>   
Preference\_Verdict:<yes|no|unverifiable>   
Preference\_Reason:<brief justification with frame numbers>   
Subtask\_Analysis:   
Subtask 1: completed=<yes|no>|   
preference=<yes|no|n/a|unverifiable>-<reason>

## D Human Review and Initial User Validation

## Construction and cross-review.

Human contributors participated in three stages: 1) providing one reference demonstration for each newly introduced skill; 2) inspecting sampled generated demonstrations together with their satisfaction labels and metrics; and 3) rating preference plausibility and demonstration human-likeness. Before beginning, contributors received common instructions and task-specific training. Contributors were assigned to different portions of the generated data, after which a second group cross-reviewed the demonstrations and labels. Disputed cases were jointly re-examined before inclusion; the initial reviewer disagreement rate was 11.8%. All participants provided informed consent, and contributors were credited according to their level of involvement.

## Survey protocol and results.

For initial user validation, 15 participants rated the selected preferences and their demonstrations on a 10-point scale along two dimensions: real-world preference plausibility, whether each preference was plausible and applicable in a household setting; and human-likeness of demonstrations, how closely the corresponding demonstration resembled how a human would naturally perform the task while satisfying the preference. Table 7 summarizes 37 preferences in three categories. These results provide initial evidence of plausibility and human-likeness, but they do not substitute for real-robot, longitudinal, or larger-scale user studies.

Before participating, survey respondents were informed of the study’s purpose and procedure and provided informed consent. They received compensation via WeChat in accordance with fair labor practices. We did not identify any risks beyond those ordinarily associated with completing a questionnaire.

Table 7: Initial human validation with 15 participants; ratings use a 10-point scale.
<table><tr><td>Preference type</td><td>Instances</td><td>Human-likeness ↑</td><td>Plausibility ↑</td></tr><tr><td>Motion</td><td>8</td><td>8.63</td><td>9.25</td></tr><tr><td>Skill</td><td>14</td><td>8.50</td><td>8.50</td></tr><tr><td>User Profile</td><td>15</td><td>9.93</td><td>7.73</td></tr><tr><td>Overall</td><td>37</td><td>9.11</td><td>8.35</td></tr></table>

## E A Worked Example of ComprePref

Underlying task: Put the white mug on the left plate.   
Explicit preference: Pick by the handle.   
Input prompt: Put the white mug on the left plate. The mug just came out of the microwave—the body   
is too hot to touch.   
Implied behavior: Grasp the handle while completing the same placement task.   
What stays fixed: The task goal, scene, and preference evaluator. SR evaluates the original placement   
goal; SA uses the original handle-grasp preference label with task-success gating.   
What changes: The language supplied to the policy conveys the preference through context instead of a   
separate explicit preference field.

A second example replaces the explicit preference “speed up” with “Guests will use them immediately” for the task of putting alphabet soup and a cream cheese box in a basket. The motion preference is still evaluated from the trajectory, not from a still image. These contextual prompts assess grounding of a specified benchmark preference; they do not establish recovery of an individual user’s unobserved preference history.

## F Adding a New Preference

Adding a standard preference requires one human reference demonstration and a natural-language description of its satisfaction metric for the preference-conditioned generation pipeline, followed by approximately three to five validation trials for each applicable task. For a preference applied to roughly five tasks, the observed manual effort is about 20 minutes on average; demonstration generation then proceeds automatically. This estimate covers preferences expressible through the current scene representation, skill library, and automatic programmable metrics. Preferences requiring a new skill or non-observable feedback incur additional work.

For example, to add a preference for grasping a mug by its handle, a user provides one reference demonstration and a natural-language description of the satisfaction metric: the robot should grasp the handle rather than the mug body. The original mug-placement goal remains unchanged. The user then runs approximately three to five validation trials for each applicable task to check whether the generated trajectories follow the preference and whether the metric correctly evaluates grasp location. After validation, the pipeline automatically generates additional demonstrations.

## G Benchmark Release and Maintenance

The benchmark has five evaluation suites. The 312 tasks include both preference-conditioned and no-preference tasks across SinglePref (100), PrefScale (100), PrefConflict (48), and PrefCompose (64); ComprePref reuses tasks with contextual prompts and does not add new mission-preference combinations to this count. Demonstrations are generated trajectories used as training/reference data; evaluation rollouts are executions of the evaluated policy and are not additional benchmark tasks.

The release documentation covers installation, dataset/configuration descriptions, and examples for generation, training, and inference. The configuration follows the five-suite organization described here, and the Croissant metadata includes the fields rai:personalSensitiveInformation and rai:hasSyntheticData. We will maintain the code, datasets, and leaderboard, improve the documentation based on community feedback, and expand rollout coverage and generation diversity.