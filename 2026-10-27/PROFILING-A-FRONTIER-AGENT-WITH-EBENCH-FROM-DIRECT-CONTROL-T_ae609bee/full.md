# PROFILING A FRONTIER AGENT WITH EBENCH: FROM DIRECT CONTROL TO HARNESS WITH POLICY

Yifei Yao<sup>1∗</sup> Yude Zou<sup>1,2∗</sup> Guanbo Jiao<sup>1,3∗</sup> Xing Gao<sup>1,†</sup> Ning Gao<sup>1,2</sup> Yukai Wang<sup>1</sup> Jiantong Chen<sup>1</sup> Zihou Zhu<sup>1</sup> Xinyu Li<sup>1</sup> Ningyu Yan<sup>1</sup> Shuai Wang<sup>1</sup> Jianming Ma<sup>1,2</sup> Haoyu Guo<sup>1</sup> Weinan Zhang<sup>1,2</sup> Chunhua Shen<sup>1,4</sup>

<sup>1</sup>Shanghai Artificial Intelligence Laboratory <sup>2</sup>Shanghai Jiao Tong University <sup>3</sup>Fudan University <sup>4</sup>Zhejiang University

## ABSTRACT

Frontier multimodal large language models (MLLMs) can control robots without robot-specific training, but how their capabilities compare and interact with post-trained vision-language-action (VLA) and world action model (WAM) policies across manipulation demands remains to be thoroughly evaluated. In this report, we evaluate GPT-6-Astra on EBench as the core of an agent that controls a robot through an agent harness under the same visual and proprioceptive observation protocol as post-trained policies. We study two agent configurations: direct control, in which the agent generates waypoint commands directly, and harness with policy, in which it can also invoke a post-trained VLA or WAM policy for action execution. Under direct control, the agent achieves competitive overall performance relative to post-trained policies. It leads on short-horizon tasks and exhibits the least performance variation across generalization dimensions, whereas it underperforms post-trained policies on long-horizon tasks, tabletop manipulation, and high-precision contact-rich tasks. In the harness with policy, invoking a post-trained VLA or WAM combines the complementary strengths of both components, raising overall task success to the level of the strongest compared post-trained policies while substantially reducing token cost. However, performance remains below the per-task upper bound of the standalone components due to suboptimal policy invocation, fixed-instruction execution without subgoals, and grasp disruption when the agent resumes control. Overall, our study characterizes the capability boundary of a frontier MLLM agent and suggests directions for improving future agent harness design. Project page: https: //internrobotics.github.io/EBench-Astra-Tech-Report/.

## 1 INTRODUCTION

Transferring knowledge from large pretrained models to robotics has become a primary approach to general-purpose robot control (Firoozi et al., 2025; Ma et al., 2026b; Shen et al., 2026). Visionlanguage-action (VLA) models adapt vision-language backbones for end-to-end visuomotor control (Kim et al., 2025; Black et al., 2025b;a; Ma et al., 2026a; GigaBrain Team et al., 2026; Yuan et al., 2026a), whereas world action models (WAMs) jointly model future visual dynamics and actions to utilize the physical and temporal priors of video generative models (Ye et al., 2026; Kim et al., 2026; Li et al., 2026a; Yuan et al., 2026b; Wang et al., 2026b; Cai et al., 2026a). Both approaches rely on post-training with robot manipulation trajectories: although effective within their training distributions, their generalization is bounded by the scale and diversity of collected robot data. By contrast, frontier multimodal large language models (MLLMs) acquire stronger emergent capabilities from massive-scale pretraining data, and this training scale remains costly to replicate with embodied data alone.

Without robot-specific post-training, frontier MLLMs provide an alternative approach through their spatial understanding and embodied reasoning capabilities (Zhang et al., 2026a; Rao, 2026). GPT-6-Astra (OpenAI, 2026), for example, can generate end-effector (EEF) commands directly from camera observations and language instructions (Zhang et al., 2026a; Liu et al., 2026; Jia et al., 2026;

Lou et al., 2026). Combined with an agent harness that supplies its prompt, observation interface, and tools, the model serves as the core of an agent that controls the robot in closed loop, as formalized in Section 3. Beyond direct control, recent studies explore a harness with policy in which the agent invokes a post-trained policy for action execution (Zhang et al., 2026b; Xu et al., 2026; Wang et al., 2026a). Such a harness combines the agent’s task planning, progress monitoring, and error recovery with the closed-loop visuomotor control that the policy learns from robot demonstrations.

Although prior studies have evaluated both direct control and a harness with policy, they have not comprehensively profiled these agent configurations against post-trained VLA and WAM policies across diverse manipulation demands. Specifically, two central questions remain to be addressed. First, under direct control, on which tasks does the agent outperform post-trained VLAs and WAMs, where does it fall short, and what limits its performance? Second, in a harness-with-policy setting, how does invoking a post-trained policy complement the agent’s direct control, which limitations of direct control can it alleviate, and what new failure modes arise when the agent invokes a policy for execution and subsequently resumes decision-making?

To further explore these questions, we evaluate GPT-6-Astra as the core of an agent on EBench (Gao et al., 2026). This diagnostic manipulation benchmark categorizes tasks by mobility, manipulation precision, and task horizon and evaluates robustness along the generalization dimensions of unseen objects, unseen backgrounds, paraphrased instructions, and their mixture. For a fair comparison, the agent operates under the same observation protocol as post-trained policies, receiving only RGB camera images and proprioceptive robot states without depth, camera calibration, or simulator ground truth. For each task, the agent receives a task description annotated by GPT-6-Astra from a single training episode as single-shot in-context learning (ICL). Across all 510 test episodes, we evaluate both direct control and a harness with policy invoking either the VLA $\pi _ { 0 . 5 }$ or the WAM InternW0-∆, compare each configuration against the post-trained policies on the EBench leaderboard, and analyze execution trajectories across paired episodes.

Under direct control, the GPT-6-Astra-based agent achieves a 46.7% overall success rate and a 0.654 stage-completion score, outperforming post-trained policies such as Qwen-RobotManip and $\pi _ { 0 . 5 }$ trained on 6,600 in-domain demonstrations. Among the compared post-trained policies, the agent ranks first on short-horizon tasks with a 73.2% success rate, shows the smallest success-rate variation across the four generalization dimensions, and completes a novel composite instruction zero-shot on which both compared post-trained policies fail. Conversely, its success rate drops to 27.3% on longhorizon tasks, 20.0% on tabletop manipulation, and 11.3% on high-precision contact-rich tasks. Trajectory analysis shows that while the agent performs well at visual grounding and in-episode error recovery, its physical execution is limited by long action chunks between observations and by contact geometry that it must infer explicitly from RGB views.

Invoking a post-trained policy through the agent harness increases the overall success rate to 53.0% with $\pi _ { 0 . 5 }$ and 56.0% with InternW0-∆ and reduces the agent’s token cost per success by more than half. With InternW0-∆, the harness surpasses all post-trained policies compared in Table 1. The harness with policy retains the agent’s high success rate on short-horizon tasks and improves performance on long-horizon and bimanual tasks where the post-trained policy is competent. At the same time, case analysis identifies two limitations of the current harness with policy. First, neither component resolves high-precision contact-rich manipulation. Second, agent and policy coordinate loosely in three respects: the agent invokes the policy for most execution steps even on tasks where that policy alone rarely succeeds, the policy follows the fixed task instruction instead of the agent’s subgoal, and the agent often leaves a held grasp unmaintained when it resumes control after a policy invocation, which makes object drops more frequent.

The main contributions of this report are as follows:

• Paired evaluation across agent configurations and post-trained policies. Under the same observation protocol as post-trained policies, we evaluate an GPT-6-Astra-based agent under direct control and in a harness with policy across all 510 EBench test episodes. We compare both agent configurations against standalone post-trained VLA and WAM policies across all task groups and generalization dimensions.

• Characterization of agent capabilities under direct control. We locate where the GPT-6-Astrabased agent leads or trails eight post-trained policies while using only a single annotated task demonstration and no parameter updates, trace its unsuccessful episodes to their failure mechanisms, and examine its in-episode adaptation, the safety of its execution, and its zero-shot composition of skills on a novel instruction. Together, these analyses show where the agent and the policies complement each other.

• Analysis of harness with policy and implications for agent harness design. We evaluate the harness with either policy against each of its components on every task, trace the episodes that newly fail under the harness and those that fail with both policies to their decisive incidents, and analyze the agent’s policy invocation and the resulting token cost. These analyses locate the remaining gap to the better component in policy invocation, subgoal passing, and grasp maintenance when the agent resumes control, which informs the design of future agent harnesses.

## 2 RELATED WORK

Robot manipulation with foundation models. Two lines of work apply foundation models to robot manipulation. Post-trained policies typically adapt foundation models on robot manipula tion trajectories, including VLA models that fine-tune vision-language backbones for action predic tion (Kim et al., 2025; Black et al., 2025b;a) and WAMs that jointly model future visual dynamics and actions (Ye et al., 2026; Kim et al., 2026; Yuan et al., 2026b; Miao et al., 2026). Direct control with frontier MLLMs, in contrast, uses frontier MLLMs as the cores of agents that con trol robots without parameter updates. Early approaches translate model predictions into robot motion through generated programs that call perception and control APIs (Liang et al., 2023) or through embodiment-specific semantic primitives (Chen et al., 2026), whereas more recent stud ies prompt MLLMs to output EEF or joint commands directly (Jia et al., 2026; Lou et al., 2026). On the RoboDojo benchmark, for example, GPT-6-Astra generates robot actions directly from observations (Zhang et al., 2026a), and providing in-context demonstration segments together with in-episode interaction history further improves task success (Liu et al., 2026; Cai et al., 2026b). Rather than evaluating direct control in isolation, our work systematically compares GPT-6-Astra under direct control against post-trained VLA and WAM policies on EBench (Gao et al., 2026) by task group and generalization dimension, linking dimension-level performance to step-by-step execution trajectories.

Harness with policy. Combining both lines, recent frameworks build hierarchical harnesses in which an agent with an MLLM core invokes post-trained policies for action execution (Qin et al., 2026; Cai et al., 2026b). For instance, Harness VLA invokes a frozen VLA alongside scripted or analytical primitives (Zhang et al., 2026b), BATON coordinates long-horizon VLA execution through a transition-aware memory that verifies subtask entry and exit conditions (Xu et al., 2026), and HarnessPAI synthesizes task-specific control programs that invoke post-trained policies without online deliberation (Wang et al., 2026a). Related frameworks also adapt harness coordination logic, policy interfaces, or recovery tools during deployment (Tang et al., 2026; Li et al., 2026c; Ding et al., 2026). However, existing studies primarily evaluate each proposed harness as a single integrated system. In contrast, we evaluate the agent both under direct control and in a harness with policy, and compare both agent configurations against standalone VLA and WAM policies across the same test scenes under a fixed in-context protocol. This paired evaluation quantifies how the agent’s policy invocation decisions align with the standalone competence of the post-trained policy, and traces episode-level gains and losses to the agent’s direct control, policy execution, or the transitions between them.

## 3 PROBLEM FORMULATION

To compare a frontier MLLM with post-trained policies on a common benchmark, we first define the agent and its harness, and then specify the two agent configurations evaluated in this study: direct control and harness with policy.

## 3.1 AGENT AND AGENT HARNESS

An agent $( F , H )$ consists of an MLLM core F and an agent harness $H = ( p _ { H } , \mathcal { O } _ { H } , \mathcal { T } _ { H } )$ through which it controls a robot. In this formulation, the prompt $p _ { H }$ specifies the robot, the rules of interaction, and the use of each tool. The observation interface ${ \mathcal { O } } _ { H }$ determines which sensor readings and robot states reach the core, and the tool set $\mathcal { T } _ { H }$ lists the calls available to the core together with how each is executed. Because the harness runs the decision loop, executing each call and returning its result to the core, the agent defines a closed-loop policy $\pi _ { A }$ without robot-specific post-training of its core. For a task τ on an episode scene s with language instruction $\ell _ { s }$ and task context $c _ { \tau } .$ at each decision step t the agent issues a tool call $a _ { t }$ conditioned on the episode interaction history $h _ { t } = ( o _ { 1 } , a _ { 1 } , \ldots , o _ { t } )$ , where $o _ { 1 }$ is the initial observation and each later $o _ { t }$ is the result returned for $a _ { t - 1 } { : }$

$$
a _ { t } \sim \pi _ { A } \big ( \cdot \mid h _ { t } , \ell _ { s } , c _ { \tau } \big ) , \qquad a _ { t } \in \mathcal { A } _ { H } \cup \{ \mathrm { s t o p } \} , \qquad \mathcal { A } _ { H } = \mathcal { A } _ { H } ^ { \mathrm { a c t } } \cup \mathcal { A } _ { H } ^ { \mathrm { i n f o } } ,\tag{1}
$$

where $\boldsymbol { \mathcal { A } } _ { H }$ contains the calls of the tools in $\mathcal { T } _ { H }$ with their admissible arguments. A call in $\boldsymbol { \mathcal { A } } _ { H } ^ { \mathrm { a c t } }$ advances the physics, either by commanding the robot through joint or EEF targets, gripper commands, motion primitives, or programs that compose them, or by invoking a controller such as a post-trained policy for a requested number of physics steps, and returns an updated observation through ${ \mathcal { O } } _ { H }$ . A call in $\mathcal { A } _ { H } ^ { \mathrm { i n f o } }$ returns information without advancing the physics, such as the current camera views and robot state or a stored demonstration. Agent control ends when the agent issues stop or reaches the physics-step limit. Unlike post-trained VLA and WAM policies that learn visuomotor control through parameter updates on robot trajectories, the agent adapts online solely through in-context conditioning, visual and proprioceptive observations, and tool calls.

## 3.2 DIRECT CONTROL AND HARNESS WITH POLICY

The two agent configurations evaluated in this study share the observation interface ${ \mathcal { O } } _ { H }$ and differ in the tool calls $\boldsymbol { \mathcal { A } } _ { H }$ exposed by the agent harness. Under direct control, $\mathcal { A } _ { H } ^ { \mathrm { a c t } } = \mathcal { A } _ { \mathrm { E E F } }$ , which consists of EEF, base, and gripper commands, and $\mathcal { A } _ { H } ^ { \mathrm { i n f o } }$ contains only a call that returns the current views and robot state, so the agent generates all motion commands itself. Under harness with policy, following Harness VLA (Zhang et al., 2026b), the harness expands the action set to $\mathcal { A } _ { H } ^ { \mathrm { a c t } } \bar { = } \mathcal { A } _ { \mathrm { E E F } } \cup \mathcal { A } _ { \mathrm { p o l } }$ where $\boldsymbol { A } _ { \mathrm { p o l } }$ denotes invocations of a post-trained policy, and adds to $\mathcal { A } _ { H } ^ { \mathrm { i n f o } }$ a call that returns the annotated task demonstration on request. In each policy invocation, the agent specifies a step limit within which the post-trained policy executes autonomously under the fixed task instruction before returning an updated observation through ${ \mathcal { O } } _ { H }$ . In this configuration, the agent alternates between issuing direct waypoint commands and invoking the post-trained policy for closed-loop trajectory execution, while retaining task planning, progress monitoring, and error recovery. Section 5 characterizes the capability boundary under direct control, and Section 6 evaluates how the harness with policy affects performance across capability dimensions.

## 4 EXPERIMENTAL SETUP

## 4.1 BENCHMARK

EBench (Gao et al., 2026) is a diagnostic simulation benchmark for generalist robot manipulation, built on GenManip (Gao et al., 2025) and Isaac Sim with the ARX Lift 2 dual-arm mobile robot. Its 26 tasks span 9 scene categories across household, retail, and industrial settings, testing functional contexts such as food preparation, household cleaning, retail checkout, and component assembly. Completing these tasks requires relating object functions and spatial layouts to the task goal and sequencing the necessary manipulation steps. EBench annotates each task along five capability dimensions: scene, atomic skill, operating mode, manipulation precision, and task horizon. Our analyses mainly group tasks by the last three, and Appendix C.2 reports the results of Section 6 by atomic skill. Operating mode, which we refer to as mobility, separates 19 mobile from 7 tabletop tasks, manipulation precision separates 14 low-, 8 medium-, and 4 high-precision tasks, and task horizon separates 11 short- from 15 long-horizon tasks. The 510 test episodes span four general ization dimensions: Background with unseen backgrounds, Object with unseen object instances, Instruction with paraphrased instructions, and Mix with their joint perturbation.

The evaluation covers all 26 test tasks across 510 episodes, with 20 episodes for each of 24 tasks and 15 each for Sandwich and Microwave, spanning 468 distinct scenes. Performance is evaluated using two official metrics: task success rate (SR) and stage-completion score, which assigns partial credit for intermediate stages completed by the end of an episode according to task-specific rules. Both metrics are first averaged across episodes within each task and then averaged equally across all

![](images/1f30e712c74b831e67280152b787badc9dde0328bd8808cbdda9d66d79928547.jpg)  
Figure 1: Overview of the GPT-6-Astra control loop on EBench and executor timelines on a Microwave test scene. The input prompt combines the task instruction with a detailed task description annotated from a single training episode. The timeline bars compare active executors across an episode under direct control and under the harness with π<sub>0.5</sub>, with blue segments executed by the agent and purple segments by $\pi _ { 0 . 5 } .$ , and with keyframes showing right-wrist camera views from that episode at the marked steps.

26 tasks. Each episode is bounded by a task-specific physics-step limit ranging from 1,500 to 5,000 steps at 30 Hz, which is not disclosed to the agent.

## 4.2 AGENT AND CONTROL INTERFACE

We use GPT-6-Astra (OpenAI, 2026) with high reasoning effort as the core of the agent and initiate an independent session thread for each episode. At each decision step, based on the current visual and proprioceptive observations, the agent outputs one or more EEF and base waypoints, including target gripper states and execution durations. Meanwhile, the agent provides a brief text explaining the rationale for the command, which we refer to as its stated reason. These waypoints are interpolated into an action chunk and executed in the simulator, after which the updated observations are returned. The shared task prompt includes empirical operational routines, such as grasp verification: closing the gripper while stationary, lifting by 1 cm, and confirming that the object moves with the gripper, as detailed in Appendix A. Token usage, tool calls, and inference time are unconstrained. If the agent terminates an episode before the physics-step limit, the robot holds its final pose until the evaluation server computes the official SR and score.

Observation protocol. Across both direct control and the harness with policy, the agent is evaluated under the same observation protocol as post-trained policies, receiving only RGB camera observations and proprioceptive robot states through pre-defined tools. Specifically, the interface supplies the views of three RGB cameras: the overview camera, termed overlook in the prompts, and the left and right wrist cameras. All three views are provided at the first step and can be requested selectively thereafter, together with the proprioceptive state comprising both EEF poses, gripper widths, the mobile base pose, and elapsed simulator steps. It provides no depth maps, camera calibration parameters, ground-truth object poses, segmentation masks, intermediate reward signals, subgoal indicators, or remaining step budgets. Official SR and score values are computed only after an episode terminates, and no state or experience is retained across episodes.

Annotated task demonstrations. Each task is accompanied by a detailed task description derived from a single demonstration in the EBench training set as single-shot ICL. Specifically, prior to evaluation, a separate GPT-6-Astra instance with high reasoning effort inspects one training trajectory, selects 10–18 keyframes, summarizes the operation sequence, and pairs the selected keyframes with the recorded robot states and actions, as shown in Appendix A.3. During evaluation, the agent is instructed to adapt the demonstrated procedure and spatial relations to current visual observations rather than replaying recorded coordinates.

Table 1: SR and score on the EBench test set, overall and by mobility group, of the eight post-trained policies with per-task results on the EBench leaderboard and of the GPT-6-Astra-based agent under direct control and in the harness with each policy. Best and second-best values in each column are in bold and underlined. MLLM denotes an agent whose motion commands all come from its MLLM core, and Hybrid an agent that also invokes a post-trained policy.
<table><tr><td></td><td></td><td colspan="2">Overall</td><td colspan="2">Mobile</td><td colspan="2">Tabletop</td></tr><tr><td>Policy</td><td>Type</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td></tr><tr><td colspan="8">Post-trained policies</td></tr><tr><td>OpenWAM-α</td><td>WAM</td><td>55.3</td><td>0.701</td><td>60.2</td><td>0.740</td><td>42.1</td><td>0.594</td></tr><tr><td>InternW0-∆</td><td>WAM</td><td>49.2</td><td>0.660</td><td>55.0</td><td>0.700</td><td>33.6</td><td>0.552</td></tr><tr><td>Qwen-RobotManip</td><td>VLA</td><td>45.6</td><td>0.608</td><td>43.9</td><td>0.576</td><td>50.0</td><td>0.696</td></tr><tr><td>π0.5</td><td>VLA</td><td>41.4</td><td>0.544</td><td>45.9</td><td>0.578</td><td>29.3</td><td>0.452</td></tr><tr><td>InternVLA-A1.5</td><td>VLA</td><td>34.2</td><td>0.468</td><td>42.3</td><td>0.562</td><td>12.1</td><td>0.215</td></tr><tr><td>π0</td><td>VLA</td><td>33.7</td><td>0.475</td><td>34.0</td><td>0.469</td><td>32.9</td><td>0.491</td></tr><tr><td>GigaBrain-0.7</td><td>VLA</td><td>33.3</td><td>0.460</td><td>31.6</td><td>0.413</td><td>37.9</td><td>0.588</td></tr><tr><td>Fast-WAM</td><td>WAM</td><td>25.6</td><td>0.371</td><td>34.0</td><td>0.460</td><td>2.9</td><td>0.129</td></tr><tr><td colspan="8">GPT-6-Astra-based agent configurations</td></tr><tr><td>Direct control</td><td>MLLM</td><td>46.7</td><td>0.654</td><td>56.6</td><td>0.730</td><td>20.0</td><td>0.447</td></tr><tr><td>Harness with  $\pi _ { 0 . 5 }$ </td><td>Hybrid</td><td>53.0</td><td>0.692</td><td>61.0</td><td>0.771</td><td>31.4</td><td>0.480</td></tr><tr><td>Harness with InternWO-∆</td><td>Hybrid</td><td>56.0</td><td>0.744</td><td>64.3</td><td>0.813</td><td>33.6</td><td>0.556</td></tr></table>

## 4.3 CONFIGURATIONS

We evaluate the GPT-6-Astra-based agent under two agent configurations, direct control and harness with policy instantiated with either $\pi _ { 0 . 5 }$ or InternW0-∆, as illustrated in Figure 1.

Direct control. The agent harness exposes only the EEF waypoint interface described above, and the agent invokes no external policy. This configuration isolates the standalone manipulation capability of the MLLM core.

Harness with policy. In this configuration, the agent retains the EEF waypoint interface and can additionally invoke one of two frozen post-trained policies as a tool: the official leaderboard checkpoint of the VLA $\pi _ { 0 . 5 }$ (Black et al., 2025a) or the WAM InternW0-∆ (Miao et al., 2026), both post-trained on EBench demonstrations. For each policy invocation, the agent specifies a step limit of 1–300 physics steps together with a stated reason. Within this step limit, the post-trained policy executes autonomously under the fixed task instruction, after which the updated visual and propri oceptive observations are returned to the agent. The harness also lets the agent query the annotated task demonstration again at any point. At each decision step, the agent chooses between issuing direct EEF waypoints and invoking the post-trained policy. In other words, the agent controls when and for how many steps the policy executes, but cannot pass agent-modified task prompts to the underlying policy, as discussed in Section 6.5.

## 4.4 COMPARED POLICIES

We compare the GPT-6-Astra-based agent under direct control and in the harness with $\pi _ { 0 . 5 }$ or InternW0-∆ against the post-trained VLA and WAM policies on the EBench leaderboard, as summarized in Table 1. Dimension-level and per-task analyses in Section 5 compare the agent against eight post-trained policies with per-task results on the 510 test episodes: OpenWAM-α (Wang et al., 2026b), InternW0-∆ (Miao et al., 2026), Qwen-RobotManip (Yuan et al., 2026a), $\pi _ { 0 . 5 }$ (Black et al., 2025a), InternVLA-A1.5 (Ma et al., 2026a), $\pi _ { 0 }$ (Black et al., 2025b), GigaBrain-0.7 (GigaBrain Team et al., 2026), and Fast-WAM (Yuan et al., 2026b), all post-trained on the public EBench training set of 6,600 episodes across the 26 tasks. All comparisons across configurations of the agent are paired by test episode, with statistical procedures detailed in Appendix B.3.

## 5 CAPABILITY BOUNDARY UNDER DIRECT CONTROL

This section evaluates the GPT-6-Astra-based agent under direct control, in which the harness exposes observations and EEF waypoint commands but no post-trained policy. Every motion of the robot is therefore commanded by GPT-6-Astra itself, so the results isolate the manipulation capabilities of the MLLM. We first compare overall performance, task groups, and generalization dimensions with those of the eight post-trained policies, and then trace the failure patterns of unsuccessful episodes, the agent’s adaptation within an episode, and the safety risks of its execution. Through case studies and a pilot experiment, we further examine the role of the in-context demonstration and zero-shot compositional generalization, and finally summarize the resulting capability boundary.

## 5.1 OVERALL PERFORMANCE

The GPT-6-Astra-based agent under direct control receives a single annotated demonstration per task and no parameter updates, yet it is competitive with VLA and WAM policies post-trained on 6,600 demonstrations of the same tasks. Compared with the eight post-trained policies in Table 1, the direct-control agent ranks third on both metrics with 46.7% SR and a 0.654 score. It trails only two WAMs, as OpenWAM-α reaches 55.3% SR and a 0.701 score and InternW0-∆ reaches 49.2% SR and a 0.660 score. It outperforms the remaining six policies, including the VLAs Qwen-RobotManip with 45.6% SR and a 0.608 score and $\pi _ { 0 . 5 }$ with 41.4% SR and a 0.544 score. Appendix Tables B.2 and B.3 report the per-task SR and score.

![](images/ca86c57bb6ee18d3f5c3e8a1a09222c612ddf9b6eac941b804d873292e53759d.jpg)  
(a) Capability profile by task group

![](images/51273a36001637510629fd8e16d27f338e3fe8bf6ef5516fb2ba8ba970370863.jpg)  
Figure 2: SR of the direct-control agent and of OpenWAM-α, InternW0-∆, Qwen-RobotManip, and $\pi _ { 0 . 5 } \left( \mathrm { a } \right)$ across mobility, horizon, and precision task groups and (b) across the four generalization dimensions of the EBench test set. The radial axis of each panel ends at its largest SR, rounded up to a multiple of 10%.

The gap between score and SR shows that the direct-control agent frequently earns stage credit in episodes that it does not finish. SR counts only complete successes, whereas the score also credits the stages completed before an unsuccessful episode ends. The SR of the direct-control agent trails that of InternW0-∆ by 2.5 percentage points and exceeds that of Qwen-RobotManip by only 1.1 points, whereas its score nearly matches that of InternW0-∆ and clearly exceeds those of Qwen-RobotManip and $\pi _ { 0 . 5 } .$ . More generally, its score exceeds its SR by a larger margin than that of any post-trained policy. Section 5.2 therefore first uses Figure 2 to locate where the agent leads and trails, and Section 5.3 then examines the stages at which its episodes fail.

## 5.2 CAPABILITY PROFILE ACROSS TASK GROUPS AND GENERALIZATION DIMENSIONS

Across task groups, the performance of the direct-control agent splits by the type of demand rather than declining uniformly with task difficulty, as shown in Figures 2a and 3a and Appendix Figure B.2. It ranks first on short-horizon tasks and second on mobile and low-precision tasks, where only OpenWAM-α leads it. It drops to seventh on long-horizon and tabletop tasks and to eighth on high-precision tasks. The following paragraphs profile the agent along the horizon, mobility, and precision of the tasks and along the generalization dimensions, and Section 5.3 traces the failures behind the trailing ranks to their mechanisms.

![](images/a88dc993d63f5bd401e5e9dffb032d37658633c83cc3d4f43633021bddd82d33.jpg)

![](images/cd35a314856d01cf3b11a429792d1af0a5fc1a5c700478e32b5ec654c6c669d0.jpg)  
Figure 3: The direct-control agent across the 26 EBench test tasks by task group. (a) SR of the direct-control agent, range across the eight post-trained policies, and best post-trained policy, with its rank among the agent and the eight post-trained policies in the column on the right of the panel. (b) Shares of its episodes ending in complete success, partial credit, or a score of zero, averaged equally across tasks within each group and rounded to whole percent so that each bar sums to 100.

Task horizon. The horizon of a task separates the agent’s strongest group from one of its weakest. On the 11 short-horizon tasks, it achieves the highest SR of 73.2% and score of 0.766, surpassing all eight post-trained policies including OpenWAM-α at 65.9% SR and a 0.700 score. Eight of these 11 tasks require only low precision, and most of them consist of locating an object and its target, repositioning the base, and carrying the object between them, which the agent plans from its current views and executes with base and EEF waypoints. When a task reduces to such coarse transport between located targets, a single annotated demonstration and the priors of the MLLM therefore suffice for the agent to exceed policies post-trained on the full training set across this group. On the 15 long-horizon tasks, by contrast, the agent reaches only 27.3% SR, whereas OpenWAM-α reaches 47.6%. On these tasks, the agent often completes the early stages but not the full task. On Detergent, for example, it reaches a mean score of 0.80 but completes the full task in only 55% of episodes.

Mobility. The mobility groups order task difficulty differently for the agent than for several posttrained policies. Because 10 of the 11 short-horizon tasks are mobile, the agent ranks second on the 19 mobile tasks with 56.6% SR. On the seven tabletop tasks, its SR falls to 20.0%, whereas Qwen-RobotManip reaches 50.0%. Qwen-RobotManip and GigaBrain-0.7, by contrast, achieve a higher SR on tabletop tasks than on mobile tasks. The tabletop gap extends beyond the four highprecision tabletop tasks to the other three: Frame, Flip cup & cookies, and Glass into box. On these three tasks, the agent reaches a mean SR of 31.7%, whereas Qwen-RobotManip reaches 91.7%. On the mobile tasks, in turn, the agent’s leading results rest on visual grounding. The Bookmark task requires locating a thin, flat bookmark on a desk crowded with visually similar stationery and everyday objects and placing it onto an open book. The agent succeeds in 90% of these episodes, whereas $\pi _ { 0 . 5 }$ reaches 55% and the remaining seven post-trained policies reach 25%–45%. In the Teacup & teapot task, the agent has to tell the two objects apart and place the teacup on the saucer and the teapot on the tray. It reaches 85% SR and a 0.93 score, whereas InternW0-∆ and Qwen-RobotManip lead the post-trained policies with 45% SR. In the Remote task, the agent must place the remote into a designated compartment of a divided holder. It succeeds in every episode, whereas InternW0-∆ succeeds in 85%. Perfume also targets a compartment of a divided organizer, and on this task the agent ties OpenWAM-α for the highest SR at 75% with a higher score of 0.88 against 0.78. Bookmark, Teacup & teapot, and Remote are the three tasks on which the agent leads all eight post-trained policies. These three tasks and Perfume all require locating the specified object among distractors or the specified compartment among several before placing the object there. These leads therefore indicate that the agent grounds objects and targets reliably, as Appendix Figure D.1 illustrates for Bookmark and Remote from the grasp approach to the placement.

Manipulation precision. Success declines most steeply with the precision that a task demands. The SR of the agent falls from 60.6% on low-precision tasks to 40.2% on medium-precision tasks and 11.3% on high-precision tasks. Its rank drops accordingly from second to fourth and eighth. On high-precision tasks, OpenWAM-α and $\pi _ { 0 . 5 }$ retain 32.5% and 17.5%, respectively. On Peg in hole and Tighten nut, the agent reaches SRs of only 20% and 10% yet mean scores of 0.60 and 0.55. Every unsuccessful episode on these tasks earns 0.50 for positioning the part at the target location and then fails at the final insertion or threading step, as shown in Appendix Figure D.3. The agent’s failures on these two tasks thus concentrate on the final contact step.

Generalization dimensions. Across the generalization dimensions, the agent’s strength holds even with minimal task context. The post-trained policies are trained on 6,600 episodes, whereas the direct-control agent receives for each task only one annotated demonstration derived from a single training episode. This demonstration cannot fully illustrate the shifts that the test episodes apply along the four dimensions of Object, Background, Instruction, and Mix. Even so, the SR of the direct-control agent stays between 44.2% and 50.8% across the four dimensions, as shown in Figure 2b and Appendix Table B.4. This 6.6-point range is the smallest among the agent and the eight post-trained policies. The four policies in Figure 2b vary more widely: 7.7 points for InternW0-∆, 11.5 points for $\pi _ { 0 . 5 }$ , 16.9 points for OpenWAM-α, and 21.5 points for Qwen-RobotManip. What the agent takes from the demonstration therefore transfers across these shifts. Moreover, the advan tage grows under the Mix dimension, where the object instance, background texture, and instruction phrasing change jointly. Every post-trained policy reaches its lowest SR under Mix. OpenWAM-$\alpha ,$ for example, falls from 61.5% under Background to 44.6% under Mix, and Qwen-RobotManip falls from 55.4% to 33.8%. The direct-control agent, by contrast, maintains 46.2% SR and a 0.643 score under Mix and ranks first on both metrics. In conclusion, the generalization shifts limit the post-trained policies more than the direct-control agent. The limits of the agent lie instead in the manipulation demands of the long-horizon, tabletop, and high-precision groups, which the next subsection examines.

## 5.3 FAILURE PATTERNS AND TOKEN COST

Most unsuccessful episodes of the direct-control agent end with partial progress rather than failing initially, which explains the score margin of Section 5.1. Of the 510 episodes, 237 end in complete success, and 188 of the 273 unsuccessful episodes achieve a nonzero score, so only 85 end at zero, as shown in Figure 3b. Zero scores concentrate on Gear, Utensils, and Glass into box, where the agent fails before any scored progress in 75%, 65%, and 50% of episodes, respectively. The trajectory analysis below traces the unsuccessful episodes to four mechanisms: imprecise contact geometry, delayed feedback over long horizons, uncoordinated bimanual motion, and unconfirmed completion claims. A final paragraph measures the token cost of the resulting failures.

Contact geometry at the final step. Precise contact fails where the agent has to infer geometry from images. The agent does not infer the surface height, part clearance, and contact geometry precisely enough for the final insertion and threading from RGB views and the base-relative EEF pose alone. Post-trained policies receive the same RGB views without depth but map them to contact motions learned from demonstration trajectories. The same limit appears on tasks with moderate nominal precision. On Glass into box, closing the case requires folding both temples inward, but the agent’s corrective pushes leave a temple protruding and prevent the lid from closing, as shown in Appendix Figure D.6b.

Delayed feedback and cascading disturbances over long horizons. The rollouts of long-horizon tasks point to two causes of failure. First, the agent detects disturbances late and recovers from them slowly. It receives an observation only after each commanded waypoint chunk finishes. Although 84% of its motion commands consist of a single waypoint, each command runs for a median of 48 physics steps, whereas $\pi _ { 0 . 5 }$ and InternW0-∆ replan every 30 and 10 physics steps, as Section 6.1 describes. A grasp loss or contact change within a chunk therefore persists until the chunk returns, and the retries that follow consume the remaining physics steps. As a result, the step limit ends 229 of all 273 unsuccessful episodes and is by far their most common ending. Second, the agent plans its approaches and grasps without anticipating how contact will move the object, so the scene becomes unstable over a long task. On Detergent, a bottle tips over or falls during the agent’s own approach, grasp, or transport in 19 of the 20 episodes, and on Bottle this happens in 11 of the 20. The agent must then grasp the fallen bottle again from the table. In one Detergent episode, for example, the agent places the first of three bottles within 11 commands, but the second bottle falls during the next command, which withdraws and approaches over 100 physics steps. The agent spends the remaining 92 commands grasping that bottle again and then the third bottle, which slips during transport. The episode reaches the step limit with two bottles placed. Such cascades accumulate most on Bottle, which is one of the two longest tasks. No evaluated policy or configuration completes Bottle, as analyzed in Section 6.6.

Uncoordinated bimanual motion. On bimanual tasks, the agent coordinates its two arms only between waypoint chunks. Each chunk executes open loop, so the agent cannot adjust one arm to the contact that the other arm makes during execution, and one hand can withdraw or move before the other has secured the shared object. On Flip cup & cookies, where the cup passes from one hand to the other, the agent fails in 90% of episodes. Appendix Figure D.3c shows one such episode, in which the supporting hand withdraws during the second handover and the cup falls off the table. On Frame, the agent takes the division of labor between the arms from the demonstration, as analyzed in Section 5.6. Its SR of 65% on this task exceeds the 50% of OpenWAM-α but stays below the 95% of Qwen-RobotManip.

Unconfirmed completion claims. Finally, the direct-control agent sometimes ends an episode early because it misjudges completion. The agent ends 43 unsuccessful episodes before the step limit, and 21 of them end with a claim of task success while a subgoal remains unmet. Seven of these claims come from Dishwasher, and the agent makes all seven with the door still open, as shown in Appendix Figure D.5. The instruction of this task asks to put every bowl into the dishwasher, and the last keyframe of the demonstration shows the door closed, yet the agent treats the placement of both bowls as completion. It states an intent to close the door after loading in only one of the 20 episodes, which is also its only success.

Token cost of failures and long horizons. These failures also dominate the token cost of direct control. The agent resends its growing episode context with every call, so the tokens of an episode grow with its number of calls, as detailed in Appendix C.3. Across the 510 episodes, the Pearson correlation between the two is r = 0.88. An unsuccessful episode keeps issuing corrections until it reaches the step limit or ends early. On average, it consumes 5.45M tokens over 55.9 calls, roughly three times the 1.84M tokens and twice the 27.4 calls of a successful episode. Long horizons compound this cost, as an episode of a long-horizon task consumes 5.40M tokens against 1.63M on a short-horizon task while succeeding less often. One success on a long-horizon task therefore costs 20.6M tokens when the tokens spent on unsuccessful episodes within each group are included. This cost is about nine times the 2.23M tokens per success on a short-horizon task.

## 5.4 ADAPTIVE CAPABILITIES

The direct-control agent observes the scene only between its own action chunks, yet the returned observations give it in-episode feedback that informs its next decision. After a failed action, the agent records a possible explanation, explores the workspace, and revises its approach. Such inepisode diagnosis is frequent across the 510 episodes: the agent’s stated reasons name a slip, a drop, or a displaced object in 302 of them, as detailed in Appendix B.2. The two examples in Figure 4 show two forms of this adaptation.

First, the agent diagnoses failures that occur while a chunk executes and changes its motion strategy accordingly. For example, in the Apple to bowl task of Figure 4a, the interface reports no failure when the apple slips, but the agent infers a possible slip from the closed finger gap and withdraws to obtain a wider view. After regrasping the apple, it avoids the base motion that preceded the slip, transports the apple with the arm alone, and completes the episode. Second, the agent adjusts the contact geometry with the physical guidance supplied in its prompt. In the Coffee beans task of Figure 4b, an initial sweep leaves the beans in place. The agent then applies a supplied caveat that the measured local EEF z-coordinate is not the height above the tabletop, repeats the sweep 25 mm lower, and tilts the fingertips downward to establish contact with the surface.

(a) Apple to bowl  
Success (Score 1.00)  
![](images/e1281fb4e9261da92277184ebd954a8c3ca5a7ef8c0bf575d4d75f13f060a654.jpg)  
Figure 4: Adaptation in two episodes of the direct-control agent, with the agent’s stated reasons below the frames. Orange boxes show enlarged views of the contact and manipulation details. (a) Apple to bowl: after a slip, the agent takes a wider view and carries the apple with the arm alone, reaching a 1.00 score. (b) Coffee beans: after a sweep leaves the beans in place, the agent lowers the edge and tilts the fingertips, reaching a 0.50 score.

The two episodes thus separate the agent’s diagnosis and adaptation from the precision of its physical execution. In both, the agent states a likely cause of the failure and revises its strategy, and the outcome then depends on how precisely the revised motion makes contact. In the Apple to bowl episode, the revised transport completes the task. In the Coffee beans episode, the lowered and tilted sweep follows the supplied caveat quoted in Appendix D.1, yet the episode ends at a 0.50 score with the collection incomplete. In-episode adaptation therefore reaches the level of diagnosis and strategy, while fine contact marks the boundary that Section 5.8 locates.

## 5.5 SAFETY IN EXECUTION

Aggregate task scores do not capture how safely the agent executes its commands, and here the direct-control agent shows clear weaknesses. They stem mainly from unconstrained execution, as the agent drives each command toward its target without accounting for the other arm, the workspace bounds, contact with the scene, or the speed of the motion. Grip losses are common, and more than half of them occur outside grasp tests. Across the 510 episodes under direct control, the gripperstate criteria of Appendix B.2 identify 811 grip losses in 333 episodes. Of these losses, 469 in 265 episodes occur in calls whose stated reasons describe no test of a grasp. Recurring patterns in the recorded rollouts illustrate four resulting hazards, and Appendix Figure D.4 shows the first three. First, independent single-arm commands can destabilize held objects during bimanual tasks. In the Dishwasher task, for example, a bowl slips from the left gripper while the right hand opens the door, as shown in Appendix Figure D.4a. Second, recovery attempts can drive the robot beyond safe workspace bounds. After an apple falls outside the workspace, for example, the agent keeps requesting EEF targets beneath the table and behind the robot base, as in Appendix Figure D.4b. Third, height and contact with the scene are hard to judge from RGB images without depth. On Coffee beans, a held jar and later the spoon strike the table edge in an awkward pose, while sweeping motions elsewhere pass above the surface, as in Appendix Figure D.4c.

The fourth hazard concerns the speed of wrist and joint motion. The agent commands target EEF poses directly and relies on the inverse-kinematics solver to compute joint trajectories, so its commands can turn the wrist through large angles within a few physics steps. In 31 of the 510 episodes, a wrist turns through $9 0 ^ { \circ }$ or more within 24 physics steps, and a joint exceeds 0.08 rad per physics step in 169 episodes. By comparison, the segments that $\pi _ { 0 . 5 }$ executes in the harness with policy of Section 6 contain no wrist turn this large within 24 physics steps, and they exceed the joint threshold in only 59 episodes, as detailed in Appendix B.2. On physical hardware, the forces and torques that these fast rotations require risk motor damage or collisions. Together, these cases highlight the need to preserve grasp stability, enforce workspace and motion constraints, and account for contact throughout manipulation before real-world deployment.

![](images/052e9e6f74dd2a936529d2a928120b5c5d7f0cc93bd7daca2ca8feb50b0ccc31.jpg)  
Figure 5: Rollouts of the direct-control agent evaluated zero-shot, π , and OpenWAM-α on a composite instruction outside the standard EBench tasks: placing a bookmark and then a pen onto the same open book. Keyframes are shown in chronological order with video time.

## 5.6 CASE STUDIES: IN-CONTEXT LEARNING, ADAPTATION, AND PRECISION

In-context learning. On Frame and Gear, the annotated demonstration conveys task-level structure to the direct-control agent but not the precise action of a final contact step. Appendix D.3 compares runs with and without the demonstration on all 20 test episodes of each task. On Frame, the demonstration raises the number of successful episodes from 1/20 to 13/20 and the mean score from 0.283 to 0.800, because it supplies the two-arm division of labor that zero-shot execution does not find. On Gear, by contrast, both conditions reach 3/20 successes and a mean score of 0.200, although the failures differ. Without the demonstration, the agent leaves the gear outside the assembly region. With it, the agent positions the gear between the mounted gears and lowers it, yet 15 of the 20 episodes still end without a scored stage, so the mean score does not rise above the zero-shot level.

ffAdaptation and precision. On two contrasting tasks, the direct-control agent adjusts its grasps and revisits disrupted goals, whereas the post-trained policies execute precise manipulation more accurately. Appendix D.2 sets the agent beside $\pi _ { 0 . 5 }$ and OpenWAM-α on Teacup & teapot, where it leads, and on Glass into box, where it trails. On Teacup & teapot, the agent targets the handle region and adjusts its wrist and gripper configuration across attempts. When later manipulation displaces the cup from the saucer, it revisits the cup and carries it back toward the saucer. In contrast, π<sub>0.5</sub> repeats approach and retraction motions without completing the cup transfer, and OpenWAM-α ffmoves the teapot onto the tray while the cup stays off the saucer. On Glass into box, however, both post-trained policies fold the temples and OpenWAM-α also closes the lid, whereas the agent’s corrective contacts leave a temple protruding as described in Section 5.3.

## 5.7 COMPOSITIONAL GENERALIZATION

To evaluate whether the direct-control agent and post-trained policies can compose skills into a new multi-step sequence, we conduct a pilot experiment with an instruction outside the standard EBench task set: placing a bookmark and then a pen onto the same open book, as shown in Figure 5. The EBench training set contains placing a bookmark onto a book and placing a pen into a holder or onto a ruler, but not placing a pen onto a book. This setting therefore tests more than the sequencing of two familiar tasks, as the combined objective includes a subtask absent from the robot-policy training data. Completing the first placement in the prescribed order awards a partial score of 0.50. The agent and both post-trained policies each run 10 rollouts, and the agent runs zero-shot without an in-context demonstration. It succeeds in all 10 episodes with a 1.00 mean score by placing the bookmark and then the pen onto the book in the required order. By contrast, $\pi _ { 0 . 5 }$ and OpenWAM-α achieve 0/10 successes with mean scores of 0.40 and 0.35, respectively. In their rollouts, both posttrained policies either place the bookmark and stop without initiating the second step or directly manipulate the pen out of order. In this pilot, generalization to individual atomic skills does not yield multi-step compositional execution, whereas the agent’s zero-shot instruction following completes the sequence.

## 5.8 THE CAPABILITY BOUNDARY

Taken together, the results of this section place the capability boundary of the direct-control agent at the transition from task-level reasoning to physical execution, and the way in which the agent acts explains why the boundary lies there. On the reasoning side, the semantic and spatial priors of multimodal pretraining supply the task-level knowledge, so that one annotated demonstration per task suffices to direct them. On the execution side, the agent acts through EEF waypoint chunks and observes the scene only after each chunk, so it lacks the continuous physical feedback that precise contact, coordinated two-arm motion, and long horizons require. The post-trained policies learn their control from demonstration trajectories and act on fresh observations throughout execution, so they receive this feedback by design.

The two sides are complementary: the direct-control agent is stronger in visual grounding, robustness across generalization dimensions, in-episode adaptation, and zero-shot skill composition, and the post-trained policies in closed-loop physical execution. Section 6 therefore examines whether this boundary and its token cost shift in a harness with policy, where the MLLM core retains task-level reasoning and invokes a post-trained policy for closed-loop execution.

## 6 HARNESS WITH POLICY

This section tests whether combining the agent with post-trained policies can overcome the execution bottlenecks of direct control. We build the harness-with-policy setting following the Harness VLA framework (Zhang et al., 2026b) and evaluate it across all 510 EBench test episodes with two representative policies: the VLA π<sub>0.5</sub> (Black et al., 2025a) and the WAM InternW0-∆ (Miao et al., 2026). We first describe the harness design and compare the two harness configurations with direct control and the standalone policies in overall performance, across task groups, and across generalization dimensions. We then diagnose how failure patterns shift once the agent invokes a policy, analyze its policy invocation behavior, and finally examine the failure modes shared across both policies to identify the sources of the remaining gap. The final subsection also compares the token cost of each configuration.

## 6.1 HARNESS DESIGN

Following Harness VLA (Zhang et al., 2026b), our harness with policy exposes a frozen post-trained policy as a callable tool alongside the direct EEF interface described in Section 4.3, together with a call that returns the annotated task demonstration on request. In each invocation, the agent assigns 1–300 physics steps. Within these steps, the policy acts without intervention from the agent and then returns new visual and proprioceptive observations to the agent. Both harness configurations keep the system prompt, annotated task demonstration, and test scenes fixed and differ in the policy and its execution and history settings. The two policies contrast in temporal conditioning and in standalone strength. For the $\mathrm { V L A } ~ \pi _ { 0 . 5 }$ (Black et al., 2025a), the harness uses the official EBench leaderboard checkpoint. This policy acts on the current observation alone, predicts 50 joint and gripper targets per inference, and replans every 30 physics steps. Evaluated alone, it trails the direct-control agent on both metrics in Table 1. The WAM InternW0-∆ (Miao et al., 2026) is likewise post-trained on EBench demonstrations but additionally conditions on two earlier frames of the episode. It predicts 32 joint, gripper, and base targets per inference and replans every 10 physics steps, which is three times as often as $\pi _ { 0 . 5 } ,$ as detailed in Appendix C. Alone, it exceeds direct control on both metrics. Pairing the same agent with both policies therefore asks whether the agent still gains from a policy weaker than itself and whether a policy stronger than the agent lifts the combination beyond both components.

Table 2: Comparison of the GPT-6-Astra-based agent under direct control, each post-trained policy evaluated alone, and the harness with $\pi _ { 0 . 5 }$ or InternW0-∆ across all 510 test episodes of EBench. Best and second-best values in each row are shown in bold and underlined.
<table><tr><td rowspan="2">Tasks</td><td rowspan="2">Direct control</td><td colspan="2">Policy alone</td><td colspan="2">Harness with</td></tr><tr><td>π0.5</td><td>InternW0-∆</td><td>π0.5</td><td>InternW0-∆</td></tr><tr><td colspan="7">All tasks (26)</td></tr><tr><td>SR (%)</td><td>46.7</td><td>41.4</td><td>49.2</td><td>53.0</td><td>56.0</td></tr><tr><td>Score</td><td>0.654</td><td>0.544</td><td>0.660</td><td>0.692</td><td>0.744</td></tr><tr><td colspan="6">Precision</td></tr><tr><td>Low (14)</td><td>60.6</td><td>49.6</td><td>58.7</td><td>68.9</td><td>68.2</td></tr><tr><td>Medium (8)</td><td>40.2</td><td>39.0</td><td>49.2</td><td>47.9</td><td>56.5</td></tr><tr><td>High (4)</td><td>11.3</td><td>17.5</td><td>16.3</td><td>7.5</td><td>12.5</td></tr><tr><td colspan="6">Mobility</td></tr><tr><td>Mobile (19)</td><td>56.6</td><td>45.9</td><td>55.0</td><td>61.0</td><td>64.3</td></tr><tr><td>Tabletop (7)</td><td>20.0</td><td>29.3</td><td>33.6</td><td>31.4</td><td>33.6</td></tr><tr><td colspan="6">Horizon</td></tr><tr><td>Short (11)</td><td>73.2</td><td>52.3</td><td>59.1</td><td>70.5</td><td>77.3</td></tr><tr><td>Long (15)</td><td>27.3</td><td>33.4</td><td>42.0</td><td>40.2</td><td>40.4</td></tr></table>

## 6.2 OVERALL PERFORMANCE

Each harness configuration outperforms both of its standalone components in Table 2, though the gains are asymmetric: pairing with the stronger policy yields the higher overall performance, whereas the weaker policy gains more from the agent. Relative to the agent under direct con trol, overall SR rises by 6.3 percentage points to 53.0% with $\pi _ { 0 . 5 }$ and by 9.3 points to 56.0% with InternW0-∆. The score rises from 0.654 to 0.692 and 0.744, respectively. OpenWAM-α is the strongest of the eight post-trained policies, and the harness with InternW0-∆ surpasses it on both metrics and ranks first in Table 1. The harness with $\pi _ { 0 . 5 }$ trails only OpenWAM-α among these eight. Relative to each policy alone, by contrast, SR rises by 11.6 points with $\pi _ { 0 . 5 }$ but only by 6.8 points with InternW0-∆. Appendix Table C.1 reports the per-task results.

These net improvements combine resolved failures with newly failed episodes. Across the 510 paired episodes, the harness with $\pi _ { 0 . 5 }$ succeeds on 91 episodes that fail under direct control and fails on 59 that succeed under it. The harness with InternW0-∆ gains 96 episodes and loses 49, as Appendix Table C.2 details. Figure 6 shows where this combination holds across task demands. On six of the seven task groups, each harness configuration comes within three percentage points of the stronger of its two components or exceeds it, whereas on high-precision tasks both configurations remain below the standalone policies. Section 6.3 traces these group results to individual tasks and generalization dimensions, and Section 6.4 then analyzes the episode-level failure patterns behind both.

![](images/02cd79578fd9b7207e201e2830da391cb961afb6c6de43cde014e79da87a8d95.jpg)

![](images/bff7a353b60c00787649eb33abca019c19cefa1ee587a04bd0b463ba94e1999d.jpg)

![](images/cd12514731e28b14c5e4edb92086fca7b5dd01841b57ee4e175ad4fcccf0ed54.jpg)

![](images/90104b0ee2511e4b584759b47a0cf7e706195f3fbe45044f0de3606379d2929f.jpg)  
Figure 6: SR of the GPT-6-Astra-based agent under direct control, of each policy alone, and of the harness with that policy, for $\pi _ { 0 . 5 }$ in (a, b) and InternW0-∆ in (c, d), across mobility, horizon, and precision task groups in $\left( \mathsf { a } , \mathsf { c } \right)$ and across the four generalization dimensions of the EBench test set in (b, d). Panels in the same column share a radial axis that ends at the largest SR of the column, rounded up to a multiple of 10%.

## 6.3 CAPABILITY PROFILE ACROSS TASK GROUPS AND GENERALIZATION DIMENSIONS

Task groups. The harness with policy improves task success mainly on long-horizon and tabletop tasks, where the post-trained policy outperforms the agent under direct control, and it keeps the agent’s lead on short-horizon tasks. With either policy, SR increases by roughly 13 percentage points on long-horizon tasks and by 11–14 percentage points on tabletop tasks in Table 2. These are two of the three groups on which the agent ranks lowest under direct control in Section 5.2. On these groups, invoking the policy brings the agent to about the level of the policy alone. Specifically, the harness with InternW0-∆ matches InternW0-∆ alone on tabletop tasks at 33.6% and stays close to it on long-horizon tasks at 40.4% versus 42.0%. The harness with $\pi _ { 0 . 5 }$ exceeds $\pi _ { 0 . 5 }$ alone on both groups, with 40.2% versus 33.4% on long-horizon tasks and 31.4% versus 29.3% on tabletop tasks. On short-horizon tasks, the agent already leads all compared post-trained policies under direct control. Both harness configurations preserve this strong performance and therefore outperform the corresponding policy alone by 18.2 percentage points. On Apple from shelf, for instance, InternW0- $\Delta$ alone achieves 15% SR, whereas the harness with InternW0-∆ reaches 75% SR. Appendix B.3 details the statistical significance of task-level improvements across all 26 tasks.

Task-level gains on bimanual and articulated stages. At the task level, the two largest gains over direct control arise with either policy on Glass into box and Dishwasher, as shown in Figure 7. On Glass into box, SR increases from 20% under direct control to 85% with $\pi _ { 0 . 5 }$ and 80% with

![](images/ccf74e1b3fbd03fbf18afe00725bf1cd7f864750f460f77538558ceb15e662a2.jpg)  
Figure 7: Per-task SR in percent across the 26 test tasks for the GPT-6-Astra-based agent under direct control, for each post-trained policy alone, and for the harness with $\pi _ { 0 . 5 }$ and with InternW0-∆, together with the share in percent of physics steps that each post-trained policy executes in its harness configuration. Tasks are grouped by precision and ordered within each group by the mean SR of the two post-trained policies alone. One blue-to-red scale shades SR and step share alike on an absolute 0–100% scale, values are rounded to whole percent, and bold marks the highest nonzero SR of each task. Table C.1 lists the full task names.

InternW0-∆. On Dishwasher, it increases from 5% to 60% and 75%, respectively. On these two tasks, the invoked policy executes the bimanual transport, temple folding, lid closure, and rack manipulation. These include the stages where the agent under direct control ran out of steps or ended early, as Appendix Figure E.2a,b illustrates for Glass into box.

Task-level shortfalls against either component. Two sets of tasks depart from these gains. First, both harness configurations fall below direct control on four of the 11 tasks where neither policy alone exceeds the SR of direct control: Peg in hole, Pen, Microwave, and Teacup & teapot. On Peg in hole, for example, SR drops from 20% to 5% with $\pi _ { 0 . 5 }$ and 0% with InternW0-∆, and on Pen from 15% to 5% and 0%. Second, the harness with InternW0-∆ falls furthest below InternW0-∆ alone on Shop and Detergent, where SR drops from 60% to 5% and from 95% to 70%. On both tasks, InternW0-∆ executes more than 80% of the physics steps in the harness, so the policy itself performs most of the motion. The shortfall arises from objects lost during the delegated segments or after the agent resumes control. Section 6.4 traces both sets to their failure patterns.

Generalization dimensions. The choice of policy also determines whether the harness keeps the balanced generalization profile of the agent under direct control, as shown in Figure 6b and d. With $\pi _ { 0 . 5 } ,$ the harness stays above $\pi _ { 0 . 5 }$ alone on all four dimensions. Over direct control, however, it improves only on Object, where SR rises from 44.2% to 59.2%, and on Instruction, where it rises from 44.6% to 56.2%. On Background and Mix, it remains at about the level of direct control. As a result, its range across dimensions widens from 6.6 to 13.0 points. The newly failed episodes explain this pattern. Resolved failures are distributed evenly at 21 to 25 episodes per dimension. Newly failed episodes, by contrast, concentrate on Background and Mix with 22 and 21 episodes, against only 7 on Object and 9 on Instruction. By failure cause, 25 of these 43 newly failed episodes begin when the agent’s own grasp slips. By task, 21 of the 43 come from just four tasks. Teacup & teapot leads them, as all five of its Background episodes newly fail. With InternW0-∆, newly failed episodes are fewer and spread more evenly at 9 to 17 per dimension, against 22 to 26 resolved failures on each. The harness therefore gains on every dimension and surpasses both of its components on each. Its Mix SR of 53.1% and score of 0.718 are the highest of all compared policies and configurations, and its range narrows to 4.6 points, the smallest in Appendix Table B.4.

## 6.4 FAILURES RESOLVED AND INTRODUCED BY THE POLICY

Table 3 follows how the 510 episodes end under direct control and in each harness configuration. Most unsuccessful episodes reach the physics-step limit in all three configurations, and nearly all of the rest end early with an unconfirmed completion claim or a reported failure of the agent. This section compares each ending between direct control and the harness to show which failures the policy resolves and which it introduces. Section 6.6 then examines the episodes that fail with both policies.

Table 3: Episode outcomes across the 510 test episodes under direct control and in the harness with $\pi _ { 0 . 5 }$ or InternW0-∆, termination modes of unsuccessful episodes, and policy invocation statistics. A reported failure denotes an early termination in which the agent explicitly reports the task as incomplete. Recovery requests denote policy invocations whose stated reason is coded lexically as asking the policy to recover from a slip, missed grasp, or displaced object, as detailed in Appendix C.
<table><tr><td rowspan="2">Direct control</td><td rowspan="2"></td><td colspan="2">Harness with</td></tr><tr><td>π0.5</td><td>InternW0-∆</td></tr><tr><td colspan="3">Outcome of the 510 episodes</td><td></td></tr><tr><td>Success</td><td>237</td><td>269</td><td>284</td></tr><tr><td>Partial credit</td><td>188</td><td>162</td><td>168</td></tr><tr><td>Score of zero</td><td>85</td><td>79</td><td>58</td></tr><tr><td colspan="3">Ending of the unsuccessful episodes</td><td></td></tr><tr><td>Physics steps exhausted</td><td>229</td><td>201</td><td>178</td></tr><tr><td>Unconfirmed completion claim</td><td>21</td><td>15</td><td>13</td></tr><tr><td>Reported failure</td><td>22</td><td>24</td><td>33</td></tr><tr><td>Other</td><td>1</td><td>1</td><td>2</td></tr><tr><td colspan="3">Policy invocation</td><td></td></tr><tr><td>Physics steps executed by the policy (%)</td><td></td><td>66.9</td><td>65.4</td></tr><tr><td>Policy invocations coded as recovery requests (%)</td><td></td><td></td><td></td></tr><tr><td>in successful episodes</td><td></td><td>6.3</td><td>5.8</td></tr><tr><td>in unsuccessful episodes</td><td></td><td>22.0</td><td>20.7</td></tr></table>

Step-limit failures. Invoking a policy mainly reduces failures at the step limit on long-horizon tasks but rarely resolves those on high-precision tasks, where both policies alone also remain weak. Relative to direct control, unsuccessful episodes that exhaust the physics-step limit decrease from 229 to 201 with $\pi _ { 0 . 5 }$ and to 178 with InternW0-∆, and zero-score episodes decrease from 85 to 79 and 58, respectively. Of the 229 step-limit failures under direct control, 72 succeed with $\pi _ { 0 . 5 }$ and 76 with InternW0-∆, and most of these conversions occur on long-horizon tasks. On high-precision tasks, only 2 and 8 of the 59 step-limit failures succeed, as detailed in Appendix Table C.2. In the newly successful episodes, the policy executes 78% of the physics steps and typically completes the final stage itself, as it does in every newly successful episode of Dishwasher.

Unconfirmed completion claims. Invoking a policy also reduces early terminations on an unconfirmed claim of completion from 21 under direct control to 15 with $\pi _ { 0 . 5 }$ and 13 with InternW0-∆. Of these 21 episodes, 10 and 11 become full successes in Appendix Table C.2. The policy brings this gain through execution. In 7 of the 10 and 8 of the 11 converted episodes, the last motion is a policy segment that completes the task, and every converted Dishwasher episode ends this way. The remaining claims concentrate on two tasks. Dishwasher accounts for 6 of the 15 claims with $\pi _ { 0 . 5 } ,$ and Gear accounts for 7 of the 13 with InternW0-∆, whereas direct control makes a single claim on Gear. In these Gear episodes, the agent claims completion while the gear rests unseated near the middle shaft.

Object losses in newly failed episodes. Object losses, by contrast, end more episodes in the harness configurations. In a grip loss, an object slips from a gripper that remains commanded closed and the fingers close fully, as detected by the criteria of Appendix B.2. Early terminations in which the agent reports an unrecoverable failure rise from 22 under direct control to 24 with $\pi _ { 0 . 5 }$ and 33 with InternW0-∆, and every one of them in either harness configuration names a lost object. Across the test set, 59 episodes that succeed under direct control newly fail in the harness with π<sub>0.5</sub>, and 49 newly fail with InternW0-∆. A call-by-call reading of these episodes shows that the two

![](images/0ea895754e95d4ef89da32ec3d921b50ad57d2a013a83be270ea21918e41c703.jpg)

![](images/279def6eabb96631e5439bd838e086b3ee4654f6a7ce49839ca564cecd434053.jpg)

(c) Policy step share  
![](images/fa0307573d27a402eeb6e5a5e6522f5b54046c12ea6a49137938ae797ec63ace.jpg)  
Progress through the episode (%) Policy alone − direct control (SR points) SR of the policy alone (%)

$$
\pi _ { 0 . 5 } ,
$$

Figure 8: Policy invocation in the two harness configurations. (a) Share of the physics steps executed by the policy in each tenth of an episode, averaged over the successful and over the unsuccessful episodes. (b) Per task, the change in SR from direct control to the harness with policy against the SR difference between the policy alone and direct control, with the identity line. (c) Per task, the share of physics steps executed by the policy against the SR of the policy alone. Each marker in (b) and (c) is one of the 26 tasks, and r is the Pearson correlation over tasks.

configurations lose objects in different ways, as reported in Appendix Table C.3. With $\pi _ { 0 . 5 } ,$ , 30 of the 59 begin with a slip of the agent’s own grasp during a lift test or transport. In the Teacup & teapot episode of Appendix Figure E.1c, for example, segments of $\pi _ { 0 . 5 }$ end with an empty hand before the agent’s own grasp of the teapot slips. With InternW0-∆, 30 of the 49 lose or misplace an object during a policy segment or just after control returns to the agent. Shop leaves little margin for such a loss, as InternW0-∆ alone needs 4,161 to 4,876 of the 5,000 physics steps in its successful episodes. In the harness, the agent delegates more than three quarters of the physics steps in 17 of the 19 unsuccessful episodes, and the delegated segments include failed bimanual handovers and their retries. All 19 episodes end at the step limit, and the harness completes a single Shop episode. Across the four high-precision tasks, none of their 6 and 7 newly failed episodes completes the final contact stage. In these episodes, pegs slip or are released before seating, the nut is released unseated, and gears are released off-axis or not fully seated.

Unmaintained grasps after policy segments. Policy segments hold objects more securely than the agent’s own commands. Within policy segments, grip losses occur 0.63 and 0.72 times per 1,000 held arm-steps with $\pi _ { 0 . 5 }$ and InternW0-∆. Under the agent’s commands, they occur 2.18 and 2.22 times overall and 0.87 and 1.00 times outside its grasp tests, as detailed in Appendix C.2. The losses after a segment arise once control returns to the agent. The harness then holds each gripper at its measured finger width, and the agent receives fresh images together with these widths. Its next command, however, often operates one arm and leaves the holding arm unspecified instead of commanding it to stay closed. Counted per holding arm, such unmaintained grasps occur 338 times after a segment of $\pi _ { 0 . 5 }$ and 316 times after one of InternW0-∆. A grip loss on that arm or a disruption reported in the agent’s next stated reasons follows within two calls in 17% and 19% of these cases, respectively, compared with 6.5% and 9.9% when the agent leaves a holding arm unspecified after one of its own commands in the same harness configuration. Stratified by task, the odds of such a disruption after a policy segment are 2.95 and 1.29 times those after the agent’s own commands, as detailed in Appendix C. All four Detergent episodes that newly fail in the harness with InternW0-∆ are lost this way, as illustrated in Appendix E.3. The agent thus does not yet use the returned proprioceptive state to confirm and maintain a grasp that the policy has established.

## 6.5 POLICY INVOCATION BEHAVIOR

Delegation over an episode. The agent delegates first and takes control back as the episode proceeds. Successful episodes then return control to the policy, whereas unsuccessful ones stay largely under the agent’s own commands. In all 510 episodes with either policy, its first motion command is a policy invocation, and the policy executes 92–97% of the physics steps in the first tenth of an episode, as shown in Figure 8a. This share declines over the first half as the agent steps in with grasp tests, corrections, and recoveries. In the second half of successful episodes, the policy’s share recovers to 58–64% in the final tenth. Within the same task, successful episodes also leave about 14 points more of their steps to $\pi _ { 0 . 5 }$ than unsuccessful ones, and about 13 points more to InternW0-∆. In unsuccessful episodes, by contrast, the policy’s share keeps falling to 35–36%, and 68% and 72% of these episodes end with a motion command of the agent. Unsuccessful episodes are also longer and contain more transitions between the agent and the policy. They average 7.2 transitions against 4.2 in successful episodes with $\pi _ { 0 . 5 }$ and 6.8 against 3.8 with InternW0-∆. Between transitions, however, the agent keeps control for longer stretches, with 0.26 and 0.24 transitions per motion command against 0.33 and 0.28 in successful episodes, as listed in Appendix Table C.5. The policy invocations of unsuccessful episodes also ask for recovery more often. With $\pi _ { 0 . 5 }$ and InternW0-∆, 22.0% and 20.7% of these invocations are coded as asking the policy to recover a slipped, missed, or displaced object. These shares are more than three times the 6.3% and 5.8% in successful episodes in Table 3. Recovery requests thus mark the least successful episodes. Among the episodes with a reported disruption, those with a recovery request fail in 79.3% and 79.0% of cases, against 58.3% and 53.8% of those without one. The agent delegates 35% and 34% of the motion commands whose stated reasons report a slip, a lost object, or another disruption, and it executes the rest itself.

Competence and delegation across tasks. Across tasks, the gain of the harness follows the competence of the policy, but the agent’s delegation does not. On the one hand, the change in SR from direct control tracks the advantage of the policy alone over the agent under direct control, with Pearson correlations of $r = 0 . 7 7$ for $\pi _ { 0 . 5 }$ and $r = 0$ .68 for InternW0-∆, as shown in Figure 8b. On the other hand, the policy executes on average 66.9% of the physics steps of a task with $\pi _ { 0 . 5 }$ and 65.4% with InternW0-∆. This share is not significantly correlated with the SR of the policy alone, with $r = 0 . 3 1$ and $r = 0 . 2 3$ in Figure 8c. For example, $\pi _ { 0 . 5 }$ alone succeeds in at most 10% of episodes on seven tasks, yet the agent still leaves it 60% of the physics steps on these tasks on average and 42–90% per task. The share is 71% on the 11 tasks on which $\pi _ { 0 . 5 }$ alone succeeds in at least half of the episodes. By the same thresholds, InternW0-∆ is weak on five tasks and strong on 15, and the agent leaves it 68% and 71% of the physics steps on them. The share differs instead by horizon. With $\pi _ { 0 . 5 }$ and InternW0-∆, it is 58% and 54% on short-horizon tasks against 73% and 74% on long-horizon tasks.

Subgoal delegation under a fixed task instruction. The agent’s stated reasons show that it divides the work with the policy by subgoal, whereas the policy pursues the whole task. In a successful Flip cup & cookies episode, for instance, the agent tests a grasp with its own waypoints under the stated reason “Test cup grasp by raising 1 cm without rotation”. It then leaves the handover, the reorientation of the cup, and the pour to InternW0-∆, as listed in Appendix E.1. In a Gear episode with $\pi _ { 0 . 5 }$ , the agent assigns each invocation a stage, from carrying the gear to the middle mounting position to its fine alignment and seating. At times it also assigns a verification such as “Verify the released blue gear remains seated while withdrawing the hand”. The policy, however, is condi tioned on the task instruction instead of that stage and reports nothing back. The agent then claims completion after checking only the overview image while the gear rests on the board unseated, as detailed in Appendix E.2. These episodes show that the agent’s subgoal does not reach the policy upon invocation.

## 6.6 FAILURE MODES SHARED ACROSS POLICIES

The two harness configurations fail largely on the same tasks. Of the 510 test episodes, 164 fail with both policies. This count is close to the 168 expected if the two configurations failed independently within each task. On the four high-precision tasks, 65 of the 80 episodes fail with both. Shared failures are therefore governed by task-level difficulty rather than by scene-specific factors.

![](images/5ff24ceaf41503d399269e12d47119004bea85da51fb9ab1d7031a0fbcd7a7de.jpg)  
Figure 9: Token use of the GPT-6-Astra-based agent under direct control and in the harness with each policy on the 510 EBench test episodes. (a) Mean tokens per episode in successful and unsuccessful episodes. (b) Tokens per success on short-horizon and long-horizon tasks, dividing all tokens of each group, including those of its unsuccessful episodes, by its number of successful episodes.

Within these shared failures, however, where an episode breaks along its chain of grasping, transfer, and contact depends on the policy. The two runs of an episode end with the same category of decisive incident in only 58 of the 164 episodes, as broken down in Appendix Table C.4. First, object losses decide 109 shared failures with $\pi _ { 0 . 5 }$ and 105 with InternW0-∆, and their location shifts with the policy. With $\pi _ { 0 . 5 } ,$ 58 of these losses occur during the agent’s own lift tests and transport and 37 inside policy segments. With InternW0-∆, the order reverses, as 53 occur inside policy segments and 42 during the agent’s own lift tests and transport. The remaining 14 and 10 follow a grasp left unmaintained after a policy segment, as analyzed in Section 6.4. Second, 19 and 24 shared failures stall at the final insertion, threading, or seating stage, and 16 and 15 spend policy segments emptyhanded or on the wrong object. Third, 19 and 20 shared failures end without a single decisive object loss. Of these, 16 and 11 reach the step limit as recovered slips, empty grasps, and displaced objects accumulate. Bottle accounts for 14 of these 27 step-limit runs, and in 11 of the 14, bottles slip, fall beside or behind the basket, or lie outside it. No run on Bottle places more than 9 of the 12 bottles. The other 3 and 9 end with a false completion claim.

## 6.7 SHIFTED BOUNDARY, REMAINING GAPS, AND TOKEN COST

The harness with policy moves the boundary of Section 5.8 where the policy supplies the closedloop execution that direct control lacks, chiefly on the long-horizon and tabletop tasks. Two limits remain, and a final paragraph examines how the shift changes the agent’s token cost.

Shared ceiling on high-precision contact. First, the agent and both policies share an execution ceiling on high-precision contact, so combining them does not lift it. A per-task choice in hindsight between the agent alone and the policy alone reaches only 21.3% SR on the high-precision tasks with either policy. The best of all three components reaches 30.0%, which remains below the 32.5% of OpenWAM-α.

Gap to the better component from loose coordination. Second, the combination loses part of what its components achieve alone. The per-task choice of the better component would attain 56.2% with $\pi _ { 0 . 5 }$ and 60.8% with InternW0-∆. It would thus exceed the respective harness configuration by 3.1 and 4.8 points over all tasks and by 13.8 and 8.8 points on the high-precision tasks. The harness with $\pi _ { 0 . 5 }$ exceeds both of its components on only ten tasks and falls below the better one on thirteen. The harness with InternW0-∆ exceeds both on eight tasks and falls below the better one on eleven. The preceding analyses trace this loss to three sources in how agent and policy coordinate. First, the agent allocates physics steps to the policy by task horizon rather than by the policy’s competence on the task, as shown in Section 6.5. Second, each policy segment conditions on the task instruction and the observations listed in Section 6.1 rather than on the agent’s subgoal or the stages already completed, as shown in Section 6.5. Third, once a policy segment ends, the agent does not command the holding arm to stay closed at the returned finger width, as shown in Section 6.4.

Token cost. Invoking a policy also moves computation from the agent into the policy. The agent consumes 1.83M tokens per episode with $\pi _ { 0 . 5 }$ and 2.22M with InternW0-∆, which is 52% and 41% less than the 3.77M under direct control. It also makes 23.7 and 22.7 tool calls per episode instead of 42.6. Per success, it needs 3.5M and 4.0M tokens, which is 57% and 51% fewer than the 8.1M under direct control. The savings concentrate where direct control is most expensive, as shown in Figure 9. On long-horizon tasks, one success costs 6.2M and 7.5M tokens in the two harness configurations, which is less than two fifths of its cost under direct control. On short-horizon tasks, the cost per success falls by only a quarter to a third. Within each configuration, unsuccessful episodes still consume 2.6 and 2.4 times the tokens of successful ones, because tokens grow with the number of calls in the harness as well. At the published tariff of GPT-6-Astra, the agent’s tokens cost USD 16.82 per success under direct control but only 4.99 and 5.49 in the two harness configurations, as detailed in Appendix C.3. In exchange, the policy adds 45 inferences of $\pi _ { 0 . 5 }$ and 135 of InternW0-∆ per episode.

With a post-trained policy, the agent reaches further on long-horizon and tabletop tasks at a lower token cost per success, but it remains bounded by high-precision contact, which none of its components masters, and by loose coordination in delegation, subgoals, and control transfer. Section 7 turns these limits into directions for future work.

## 7 CONCLUSIONS AND FUTURE DIRECTIONS

This report profiles GPT-6-Astra on EBench as the core of an agent that controls a robot through an agent harness. The agent observes the scene under the same visual and proprioceptive protocol as post-trained policies and acts first under direct control and then in a harness with a post-trained VLA or WAM. Under direct control, the capability boundary of the agent lies between flexible task interpretation and reliable physical execution. Without robot-specific post-training of the MLLM, the direct-control agent draws on multimodal pretraining and one annotated demonstration for each task to compete with post-trained policies overall, with particular strengths in visual grounding, robustness to generalization shifts, adaptation to failures during an episode, and zero-shot composition of skills. Physical execution, by contrast, exposes a tension in how the agent controls the robot. The agent chooses its corrections between waypoint chunks, whereas grasps, contacts, and the alignment of parts evolve within each chunk, so precise contact, bimanual coordination, and long horizons limit its success. Because each correction also resends the accumulated context, the unsuccessful and long-horizon episodes that this tension produces account for most of the agent’s token cost. The same unconstrained execution also drops held objects outside the agent’s deliberate grasp tests and at times produces abrupt wrist rotations that risk damaging physical hardware.

Invoking a post-trained policy moves part of this boundary. With InternW0-∆, the harness with policy reaches the highest SR and score in Table 1, and with either policy the agent needs far fewer tokens per success on long-horizon tasks, where direct control is costliest. However, two shortfalls of the combination indicate where further progress is needed. First, precise contact limits the agent and each policy alike, so combining them leaves this ceiling in place. Second, the combination falls short of its better component on many tasks, because the coordination between agent and policy remains loose. The agent delegates without regard to the policy’s competence, the policy pursues the task instruction rather than the agent’s subgoal, and when control returns from the policy, the agent often leaves the holding arm unmaintained, allowing grasped objects to slip. These limitations motivate us to explore several directions.

Designing subtask interfaces and training policies as learnable tools. In the evaluated harness, the agent passes the policy only a step count, so it decides when the policy acts but can neither assign a subgoal nor receive a report of its completion. A finer interface would not only expose learned skills as tools that a planner selects (RLinf, 2026) but also pass an explicit subgoal to the policy, verify the entry and exit conditions of each subtask (Xu et al., 2026), and return whether the subgoal was reached. Such an interface also requires a policy that can serve it. The evaluated policies are post-trained on complete EBench task trajectories under the task instruction, whereas the harness invokes them mid-episode from states that the agent has reached and for subgoals that the agent has chosen. The policy itself therefore has to be trained as a learnable tool of the harness. This training spans three levels and can build on the subtask instruction following of $\pi _ { 0 . 5 }$ (Black et al., 2025a)

and on the subtask descriptions and execution feedback that the authors of InternW0-∆ list as future extensions of its interface (Miao et al., 2026). First, the policy should follow the agent’s subgoals rather than only the task instruction. Second, it should be trained on the mid-episode states in which the agent transfers control, including disturbed and displaced ones. Third, it should end each segment on subgoal completion or failure, returning structured feedback to the agent and leaving any held object in a stable grasp. The agent, in turn, can be trained to delegate a subtask according to the policy’s competence on it and to use the proprioceptive state returned at every control transfer to keep a held object grasped.

Grounding exploratory experience into reusable memory and tools. Within an episode, the agent diagnoses slips and displaced objects and revises its strategy, yet every new episode starts again from the same static prompt. Recent work turns such experience into reusable assets. Languagemodel programs compose perception and control APIs into robot behavior (Liang et al., 2023), and recent harnesses synthesize task programs around post-trained policies (Wang et al., 2026a), expose embodiment-specific primitives (Chen et al., 2026), or extend their control interfaces during deployment (Li et al., 2026c). To admit only experience that generalizes, recent work also validates candidate skills on held-out episodes (Lu et al., 2026) and isolates the evaluation from the agent that revises its own prompt and tools (Elmaaroufi et al., 2026). For embodied agents, two questions follow: which recoveries generalize beyond a single trial, and whether experience accumulated with one MLLM serves other MLLMs as well. The stated reasons of the agent offer a natural starting point, as each names a diagnosed failure together with the revised action that follows it. Validated pairs of failure and revision can be retained across episodes as memory or compiled into tools that wrap closed-loop control.

Transferring the agent’s exploratory and corrective strategies to smaller policies. The recorded rollouts contain corrective strategies such as withdrawing for a wider view after a slip and lowering a sweep that missed the beans. The agent produces these strategies at the latency and token cost of a frontier MLLM. Distilling them into smaller models that run at deployment speed would pair failure states with the agent’s revised strategies and their outcomes, including corrections that did not succeed. Such distillation would thereby extend behavior cloning from trajectories toward the intentions behind them. Policies already improve from their own experience and from corrective interventions (Physical Intelligence et al., 2025), and an LLM agent can run the cycle of training and evaluating a VLA without a human in the loop (Yu, 2026). Because agent rollouts carry the reason for every correction, they extend such training data to corrections that the agent discovers on its own.

Coupling geometric perception with active safety constraints in execution. Precise contact and safe execution both depend on geometry that the agent infers from RGB views without depth. On insertion, threading, and seating, the agent has to judge the height of a surface, the clearance of a part, and the onset of contact from images, and its commands reach the robot without checks against the table, the workspace bounds, or the speed of the motion. The harness can therefore pair perception tools with a safety layer. Perception tools such as depth estimation, open-vocabulary detection and segmentation, and visual checks of the terminal state can estimate these quantities from the same camera views. The safety layer can check each command against workspace, reachability, and velocity limits and interrupt chunk execution on slip or collision. On real hardware, a queryable model of the robot’s body already lets an agent convert functional targets into control targets (Lou et al., 2026), and such a model can supply the kinematic limits that this safety layer enforces. Reporting safety events beside SR and score would let benchmarks measure this progress.

Limitations. This report mainly studies the capability profile of one frontier MLLM, GPT-6-Astra, as the core of an agent on one simulated benchmark with synchronous execution, both under direct control and in a harness with two post-trained policies. Natural extensions include measuring the agent’s execution time, asynchronous execution in which the scene keeps evolving while the model deliberates, self-improvement of the agent with or without training the post-trained policy, a full evaluation of other frontier MLLMs as agent cores under the same protocol, and deployment on a physical robot. Appendix B.2 lists the protocol of each run.

## REFERENCES

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, et al. π<sub>0.5</sub>: a visionlanguage-action model with open-world generalization. In Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 17–40. PMLR, 2025a.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π : A vision-language-action flow model for general robot control. In Proceedings of Robotics: Science and Systems, 2025b. doi: 10.15607/RSS.2025.XXI.010.

Jisong Cai, Yao Mu, Ganlin Yang, Zhe Cao, Zhangzheng Tu, Xing Gao, Kailin Li, Xinyu Zhan, Lixin Yang, Yangkun Zhu, et al. InternW0: A foundational physical world model for efficient real-world interactions. arXiv preprint arXiv:2609.27656, 2026a. URL https://arxiv. org/abs/2609.27656.

Zhongang Cai, Ziang Cao, Tailai Chen, Han Chong Chua, Linjun Dai, Hanming Deng, Hokin Deng, Haiwen Diao, Chenjing Ding, Weichen Fan, et al. Has GPT-6-Astra achieved embodied intelligence? arXiv preprint arXiv:2610.22471, 2026b. URL https://arxiv.org/abs/2610. 22471.

Yanzhe Chen, Zechen Bai, Zhijun Cao, Wenzheng Zeng, Kevin Qinghong Lin, Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, Qiming Huang, and Mike Zheng Shou. Show-Harness: Just a VLM agent can play robots. arXiv preprint arXiv:2609.10522, 2026. URL https://arxiv.org/ abs/2609.10522.

Xin Ding, Liang Mi, Mingzhe Huang, Zixuan Wang, Chao Zhang, Zixu Hao, Fu Chen, Xiangyu Li, Yikai Zheng, Yaoyu Guo, et al. Zetta ζ: An efficient closed-loop embodied harness for self-evolving physical intelligence. arXiv preprint arXiv:2608.16590, 2026. URL https:// arxiv.org/abs/2608.16590.

Karim Elmaaroufi, Justin Svegliato, Sarunas Kalade, Graham Schelle, Sanjit A. Seshia, and Matei Zaharia. RHO: Your coding agent is secretly a roboticist. arXiv preprint arXiv:2606.16458, 2026.

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, 44(5): 701–739, 2025. doi: 10.1177/02783649241281508.

Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, and Jiangmiao Pang. GenManip: LLM-driven simulation for generalizable instructionfollowing manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12187–12198, 2025. doi: 10.1109/CVPR52734.2025.01138. URL https://arxiv.org/abs/2506.10966.

Ning Gao, Jinliang Zheng, Xing Gao, Haoxiang Ma, Hanqing Wang, Yukai Wang, Jiantong Chen, Zanxin Chen, Shujie Zhang, Mingda Jia, et al. EBench: Elemental diagnosis of generalist mobile manipulation policies. arXiv preprint arXiv:2606.18239, 2026. URL https://arxiv.org/ abs/2606.18239.

GigaBrain Team, Angen Ye, Axiang Sun, Can Jin, Chenxi Cheng, Chong Shi, Dengke Shang, Dingqian Zhang, Guan Huang, Guangqiang Wang, et al. GigaBrain-0.7: Scaling embodied foundation models to emergent capabilities with a three-system architecture. arXiv preprint arXiv:2608.15875, 2026. URL https://arxiv.org/abs/2608.15875.

Mengzhao Jia, Yang Lin, Xixin Zhang, Zhihan Zhang, Xiaobai Liu, and Meng Jiang. Agent as policy for robotic manipulation. arXiv preprint arXiv:2609.12541, 2026. URL https://arxiv. org/abs/2609.12541.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. OpenVLA: An opensource vision-language-action model. In Proceedings of the 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pp. 2679–2713. PMLR, 2025. URL https://proceedings.mlr.press/v270/kim25c.html.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026. URL https:// arxiv.org/abs/2601.16163.

Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026a. URL https://arxiv.org/abs/2601.21998. LingBot-VA.

Yifan Li, Jiaxu Wang, Dongming Wu, Yicheng Jiang, Ryan Ji, Xiangyu Yue, and Yanwei Fu. Vela: Scaling vision-language-action models with adaptive action curve parametrization. arXiv preprint arXiv:2610.05230, 2026b. URL https://arxiv.org/abs/2610.05230.

Zexi Li, Yehang Zhang, Haojian Huang, Bohan Zhou, Wenqian Li, Chenxu Wang, Yifan Chang, Yangkai Wei, Tianyi Zhang, Ying-Cong Chen, et al. RACaP: Agentic reasoning, acting, and coding as policies for evolvable robot learning. arXiv preprint arXiv:2609.29394, 2026c. URL https://arxiv.org/abs/2609.29394.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as Policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation (ICRA), pp. 9493–9500, 2023. doi: 10. 1109/ICRA48891.2023.10160591.

Fangcheng Liu, Yeqing Shen, Anda Cheng, Weishi Mi, Chao Tang, Chenyuan Liu, Yushun Xiang, Tingguang Li, Yong-Lu Li, and Yehui Tang. RoboICL: Embodied in-context learning with GPT-6 Astra. arXiv preprint arXiv:2609.34261, 2026. URL https://arxiv.org/abs/2609. 34261.

Zeyu Lou, Yanhong Zeng, Yong Wang, and Chenyang Si. Know your body: A harness for direct and self-improving robot control with VLMs. arXiv preprint arXiv:2609.28530, 2026. URL https://arxiv.org/abs/2609.28530.

Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, Ken Goldberg, Ang Chen, et al. ASPIRE: Agentic skills discovery for robotics. arXiv preprint arXiv:2607.00272, 2026.

Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, et al. InternVLA-A1.5: Unifying understanding, latent foresight, and action for compositional generalization. arXiv preprint arXiv:2607.04988, 2026a. URL https://arxiv.org/abs/2607.04988.

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on visionlanguage-action models for embodied AI. IEEE Transactions on Neural Networks and Learning Systems, 37(7):3031–3051, 2026b. doi: 10.1109/TNNLS.2025.3650584.

Xingyu Miao, Zizun Li, Baole Fang, Kaiwen Song, Tenghui Wang, Hanxue Zhang, Yating Wang, Xudong Li, Yuping He, Xueyuan Wei, et al. InternW0-∆: A world action model bridging predictive dynamics and actions with 20K+ hours of open data. arXiv preprint arXiv:2609.31394, 2026. URL https://arxiv.org/abs/2609.31394.

NVIDIA. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026. URL https://arxiv.org/abs/2606.02800.

OpenAI. GPT-6 Astra system card. OpenAI Deployment Safety Hub, sep 2026. URL https: //deploymentsafety.openai.com/gpt-6-astra. Accessed October 21, 2026.

Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. π<sup>∗</sup> : a VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

Chonghan Qin, Dongming Wu, Kaituo Feng, Jiaxu Wang, Xinlei Yu, Peiwen Sun, Yijia Fan, Xilin He, Bingao Chen, Weitong Lian, et al. Towards agentic robots: Model, data, environment, and harness. arXiv preprint arXiv:2610.20442, 2026. URL https://arxiv.org/abs/2610. 20442.

Kanishka Rao. Gemini 4 Argon for robotics. Google Developers Blog, oct 2026. URL https: //developers.googleblog.com/gemini-4-argon-for-robotics/. Accessed October 21, 2026.

RLinf. RPent documentation: System description. Online documentation, 2026. URL https://rpent.readthedocs.io/en/latest/rst\_source/development/ architecture.html. Accessed 2026-09-26.

Qiuhong Shen, Shihua Zhang, Yue Liao, Qi Li, Zhenxiong Tan, Shizun Wang, Shuicheng Yan, and Xinchao Wang. World action models: A survey. arXiv preprint arXiv:2606.20781, 2026. URL https://arxiv.org/abs/2606.20781.

Junyi Tang, Jie Peng, Zezhen Ding, Yuan Shen, and Tianlong Chen. AdaHVLA: Adaptive harnesses for long-horizon vision-language-action execution. arXiv preprint arXiv:2609.29204, 2026. URL https://arxiv.org/abs/2609.29204.

Xin Wang, Wenhao Wu, Menghao Zhang, Zhi Wang, Kun Shao, Jian Luan, Yang Li, Qing Li, Shangding Gu, Huichi Zhou, et al. HarnessPAI: An evolving harness for physical AI. arXiv preprint arXiv:2609.29166, 2026a. URL https://arxiv.org/abs/2609.29166.

Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, et al. OpenWAM: An open, modular exploration towards systematic world-action model pretraining. arXiv preprint arXiv:2609.07398, 2026b. URL https://arxiv.org/abs/2609.07398.

Bingxin Xu, Yuzhang Shang, and Emilio Ferrara. Don’t drop the BATON: Long-horizon robot manipulation via agentic subtask exploration and transition-aware memory. arXiv preprint arXiv:2608.16889, 2026. URL https://arxiv.org/abs/2608.16889.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World Action Models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. URL https://arxiv.org/abs/2602. 15922.

Hang Yu. You don’t need to stay in the loop: An agentic robotics loop for robot-policy improvement. arXiv preprint arXiv:2608.07555, 2026.

Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026a. URL https://arxiv.org/abs/2606.17846.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026b. URL https: //arxiv.org/abs/2603.16666.

Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, Zhiyang Dou, et al. An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv preprint arXiv:2609.24170, 2026a. URL https://arxiv.org/abs/2609.24170.

Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Yi Nie, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, et al. Harness VLA: Steering frozen VLAs into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026b. URL https://arxiv.org/abs/2607.08448. Code: RLinf/RPent.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025. URL https://arxiv.org/abs/2510.10274.

Yude Zou, Yifei Yao, Guanbo Jiao, Jianming Ma, Guanjie Zheng, and Xing Gao. PICA-WAM: Visual perturbation invariance via cross-appearance self-alignment in world action models. arXiv preprint arXiv:2610.26813, 2026. URL https://arxiv.org/abs/2610.26813.

## APPENDIX

A Agent Interface and Prompts 28   
A.1 Agent instructions and task prompt . 28   
A.2 Tool interface 29   
A.3 Demonstration format . 30   
A.4 Tool responses . 31   
B Additional Benchmark Results 33   
B.1 Tasks, leaderboard, and per-task comparisons 33   
B.2 Evaluation protocol and event definitions . 38   
B.3 Statistics . 38   
C Harness Details 39   
C.1 Post-trained policies . 39   
C.2 Additional results . 39   
C.3 API-equivalent cost 43   
D Additional Qualitative Results 44   
D.1 Prompt-grounded cases . . 46   
D.2 Rollouts of the agent and post-trained policies 46   
D.3 In-context learning 47   
E Case Studies in the Harness with Policy 48   
E.1 Division of labor 49   
E.2 Verifying physical completion 49   
E.3 Unrecovered object losses 50   
E.4 Preserving completed stages 51   
E.5 Recovery after delegated handovers 51

## A AGENT INTERFACE AND PROMPTS

This appendix reproduces the inputs and outputs of the agent from the stored records of the three configurations, and the system instructions and tool declarations from the code that each run executed. Omissions are marked [. . . ], other bracketed text replaces images, identifiers, and file paths or labels an excerpt, non-ASCII characters are transliterated, and line breaks within a paragraph result from typesetting.

## A.1 AGENT INSTRUCTIONS AND TASK PROMPT

System instructions. Every agent session of the direct-control evaluation begins with the following system instructions, followed by instructions of the model host on returning images from tool calls. In both harness configurations, the list of prohibited actions ends with “or delegate to other agents”, and the query icl paragraph that closes the task prompt below is inserted between the two parts.

You control one simulated EBench robot episode. Use only the supplied   
RGB and measured robot state and the declared robot tools. Do not   
read external files, use shell, browse, or delegate. The server   
judges success independently. Stop when it reports a terminal result   
or a budget is exhausted. Repeated observe at the same timestep   
shows the same scene. A closed gripper or finger contact alone does   
not prove a secure grasp. Choose continuous poses, motion timing and   
observation cadence yourself.   
[... instructions of the model host on returning images from tool calls   
...]

Task prompt. The first user turn consists of the task prompt, the demonstration (Appendix A.3), and the current observation. The excerpt below is the first turn of an episode of the harness with π<sub>0.5</sub> on the bowl-to-plate test task. Under direct control, the agent received the same turn without the paragraph on execute pi05 and the paragraph on query icl.

Complete this single task yourself using the tools: Take out one of the   
three bowls and stack it on the plate.   
Robot: manip/lift2/R5a, two arms and a movable planar base. Left arm is   
left in the overlook view. EEF poses are local to each arm's planner   
base: +X forward, +Y robot-left, +Z up; quaternion order wxyz. [...]   
There is no supplied depth, object pose or camera calibration. Only   
RGB and measured robot EEF/gripper/base state are available. [...]   
Use execute\_eef sequentially, not parallel calls. Use observations   
to judge motion and object manipulation. The server determines task   
success independently. Do not start another episode. Resource   
limits: unlimited tool calls, physics deadline enforced by the   
server; no hard token limit; [...] When finished, state the observed   
outcome concisely.   
You may call execute\_pi05 to delegate a manipulation segment to a   
learned policy. Choose when to delegate using the current scene. It   
replans every 30 steps and automatically returns control after the   
requested segment (at most 300 steps). The pi0.5 prompt is fixed for   
this task; always pass instruction=null and never invent a per-call   
prompt. Inspect its returned RGB/state and take over with EEF   
waypoints or delegate again. [...] The learned policy uses the same   
current three RGB cameras plus joint/gripper/base state; it receives   
no object ground truth or demonstration images.

Manipulation guidance: after visually aligning, close at a stationary   
pose and allow settling. A nonzero finger gap on the table is   
contact evidence, not proof of a secure grasp. Test each grasp by   
lifting only 0.01 m first, using motion\_steps=16 or more, then hold   
8..16 steps and request the active wrist plus overlook views. [...]   
RGB inputs are downsampled: overlook 640x360, wrist 384x288; no crop or   
object annotation.   
The saved ICL demonstration remains available through query\_icl   
throughout this episode. [...] Ground actions in the current   
RGB/state.   
[... the demonstration: annotation text and ten captioned keyframe   
images ...]   
END OF HISTORICAL DEMONSTRATION. The following images and measured   
state are the CURRENT episode. [...]   
Measured robot state (EEF poses in each arm's planner base):   
{"state.gripper": [0.044, 0.044, 0.044, 0.044], "state.base": [0.0,   
0.0, -0.0], "state.ee\_pose": [[[0.1002, 0.0, 0.16349], [1.0, 0.0,   
2e-05, -0.0]], [[0.1002, 0.0, 0.16349], [1.0, -0.0, 2e-05, -0.0]]],   
"timestep": 0}   
CURRENT RGB: overlook, 640x360   
[image: overview camera]   
[... the left and right wrist views in the same format ...]

The harness with InternW0-∆ received the same turn with the following paragraph in place of the one on execute pi05.

You may call execute\_wam to delegate a manipulation segment to a   
learned policy. Choose when to delegate using the current scene. It   
replans every 10 steps and automatically returns control after the   
requested segment (at most 300 steps). The WAM prompt is fixed to   
this task's instruction; always pass instruction=null and never   
invent a per-call prompt. [...] The learned policy uses the same   
current three RGB cameras plus robot state, and its own frames from   
earlier in this episode; it receives no object ground truth or   
demonstration images.

## A.2 TOOL INTERFACE

Table A.1 lists the agent’s tools with their argument schemas and their descriptions as declared to the model. Under direct control, the agent has the first three tools. The harness with π adds execute pi05 and query icl, the harness with InternW0-∆ adds execute wam and query icl, and the task prompt of each harness configuration adds the paragraphs on its two tools shown above. The declarations of the EEF tools and of query icl are identical in every configuration that has them, and no other tool is declared.

Table A.1: Agent tools, with descriptions reproduced from the tool declarations (omissions marked [. . . ]). All tools except query icl also take cameras (one to three of overlook, left wrist, and right wrist), and execute eef, execute pi05, and execute wam a free-text reason. Availability: all denotes every configuration, VLA the harness with π<sub>0.5</sub>, WAM the harness with InternW0-∆, and demo both harness configurations, in which the demonstration of the first turn also remains available on request.
<table><tr><td>Tool</td><td>Arguments</td><td>Available</td><td>Description as shown to the model</td></tr><tr><td>execute_eef</td><td>waypoints: 1-6of {1eft,right (position, quaternion, and two finger targets, or null), base pose or null, steps 1–64, and optional</td><td>all</td><td>“Execute 1..6 native GenManip absolute EEF waypoints in order. Each arm target has xyz(m), unit quaternion(wxyz) in that arm&#x27;s planner-base frame, and two finger joint targets(m): [0.044,0.044] open, [0,0] close. [.. . ] The simulator performs IK; targets may remain unreachable, so inspect returned measured poses. No automatic grasp approval or task-stage rules are imposed. Returns final robot state and selected RGB cameras; sequences have no intermediate model observation. [.. . ]”</td></tr><tr><td>set_gripper</td><td>side, opening in [0,1], steps 1–64</td><td>all</td><td>“Send a normalized gripper opening command: opening=0 fully closed, opening=1 fully open. [... ] Use this for closing at rest; measured opening may remain nonzero due to contact. Returns selected RGB and measured robot state.&quot;</td></tr><tr><td>observe</td><td>none</td><td>all</td><td>“Return selected RGB views and measured robot state from the latest acknowledged simulator step. Does not advance physics. [. . . ]&quot;</td></tr><tr><td>execute-pi05</td><td>steps 1–300, instruction (string or null)</td><td>VLA</td><td>&quot;Delegate a bounded manipulation segment to the trained EBench pi0.5 policy. It uses the CURRENT RGB images and robot state, predicts absolute joint/gripper targets, and replans every 30 steps. Choose this when learned dexterity may help with grasping, alignment, insertion or other fine manipulation. [.. .] After the requested steps, control automatically returns to you with fresh RGB and state; [... ] pass instruction=null. [... ]”</td></tr><tr><td>execute_wam</td><td>steps 1–300, instruction (string or null)</td><td>WAM</td><td>“Delegate a bounded manipulation segment to the trained EBench InternW0-delta world-action-model policy. It uses the CURRENT RGB images and robot state together with its own frames from earlier in this episode, predicts absolute joint/gripper targets, and replans every 10 steps. [...] After the requested steps, control automatically returns to you with fresh RGB and state; [...] pass instruction=null. [... ]&quot;</td></tr><tr><td>query-icl</td><td>view (text, frames, or all), frame_indices (keyframe indices)</td><td>demo</td><td>&quot;The saved ICL demonstration remains available through query_icl throughout this episode. [.. .] Querying does not advance simulation. Ground actions in the current RGB/state.&quot;</td></tr></table>

## A.3 DEMONSTRATION FORMAT

The demonstration enters the first turn as one text block followed by alternating keyframe captions and images, and the three configurations received the same demonstration. Table A.2 lists the part of this text block for the bowl-to-plate task: the annotation of the source episode, a fixed paragraph on the interpretation of the records, and one numeric record per keyframe. Each record holds the measured EEF poses of the left arm ee0 and the right arm ee1 as position and wxyz quaternion, the summed finger gaps, and the base triplet, together with the exported action target.

Table A.2: Parts of the annotated bowl-to-plate demonstration in the first turn, with text reproduced from the prompt and omissions marked [. . . ]. The numeric record of one keyframe is shown with values rounded to three decimals.
<table><tr><td>Part</td><td>Content as shown to the model</td></tr><tr><td>Header</td><td>&quot;HISTORICAL DEMONSTRATION – reference from a separate recorded episode, not the current observation or the current episode&#x27;s action history.&quot;</td></tr><tr><td>Source and instruction</td><td>&quot;Historical demonstration of bowl_to_plate, source episode 600. Reference instruction: Take out one of the three bowls and stack on the plate.&quot;</td></tr><tr><td>Arm roles</td><td>“Right arm; base shifts from the sink toward the plate.&quot;</td></tr><tr><td>Observed sequence</td><td>&quot;Reach into the sink and grip the rim of the brown bowl. Lift it clear of the other bowls and sink rim, reposition the base toward the plate on the counter, place the bowl over the plate, lower it onto the plate, and open the fingers.&quot;</td></tr><tr><td>Observed outcome</td><td>&quot;The brown bowl is visibly supported on the white plate after the right fingers open; two patterned bowls remain in the sink.&quot;</td></tr><tr><td>Evidence limits</td><td>“Final contact is partly occluded by the gripper. No official success label or force measurement is supplied.&quot;</td></tr><tr><td>Fixed guidance</td><td>“This separate recorded episode is not the current observation or action history. [.. .] Transfer the sequence and relative geometry, ground commands in live RGB and robot state, and obey the live task instruction.&quot;</td></tr><tr><td>Numeric record, one per keyframe</td><td>frame 188, source_timestamp_s 12.533 measured: ee0_xyz_wxyz [0.100, 0.000, 0.163, 1.000, 0.000, 0.000, 0.000], ee1_xyz_wxyz [0.384, 0.015, 0.046, 0.701, 0.000, 0.713, 0.000], finger-gaps_m_ee0_ee1 [0.088, 0.015], recorded_base_triplet</td></tr><tr><td></td><td>[0.000, 0.152, 0.000] exported_action: ee0_xyz_wxyz as measured, ee1_xyz_wxyz [0.010, 0.040, 0.204, 0.735, 0.246, -0.259, 0.576], finger-gaps_m_ee0_ee1 [0.088, 0.000], recorded_base_triplet as measured</td></tr><tr><td>Keyframe caption and image, one per</td><td>&quot;Historical demonstration: Historical episode 600; frame 188; 12.533s; overlook camera. Grasp and lift the brown bowl.&quot; followed by the keyframe image, as shown for all ten keyframes in Figure A.1.</td></tr></table>

Bowl to plate: annotated demonstration from training episode 600  
![](images/0e62ae88ec73b7885ad7ea67e74343bce4183d8b55949201a2be2d1ee98e0ae6.jpg)  
Figure A.1: The ten keyframes of the annotated bowl-to-plate demonstration (training episode 600) with their captions, as they enter the first turn. Each frame shows the video time of the source recording and the camera when it is not the overview camera. The agent does not observe the top camera during execution.

## A.4 TOOL RESPONSES

Direct control. Every tool response consists of a JSON receipt, a text line with the measured robot state, and a caption and an image for each requested camera. The receipts under direct control have the same structure, and no receipt in the three configurations compares a command with the measured state. The exchange below is call 20 of the episode whose first turn is shown above. The right arm holds a bowl, and the commanded position lies 0.107 m above the position reached. The receipt reports no error, so the shortfall is visible only from the measured state. The reason stated with the next call is appended.

```jsonl
{"cameras": ["overlook", "right_wrist"], "reason": "Lift the securely
moving bowl slowly above sink.", "waypoints": [{"left": null,
"right": {"xyz_m": [0.48996, -0.12997, 0.3], "quaternion_wxyz":
[0.70722, 8e-05, 0.707, 0.00013], "gripper_m": [0, 0]},
"base_xy_yaw_deg": null, "motion_steps": 56, "steps": 64}]}
[Response]
{"executed_waypoints": [{"waypoint": 12, "executed_steps": 64,
"measured_after": {"state.gripper": [0.044, 0.044, 0.00963, 0.0097],
"state.base": [0.0003, 0.09162, -0.0], "state.ee_pose": [[[0.10034,
-0.00058, 0.16485], [0.99998, 0.00116, 0.0035, -0.00427]],
[[0.48621, -0.12846, 0.19264], [0.71538, -0.00048, 0.69874,
0.00032]]], "timestep": 1154}}], "remaining_tool_calls": null,
"remaining_physics_steps": null, "reported_token_usage": {[...]}}
Measured robot state (EEF poses in each arm's planner base): [... the
state of measured_after ...]
CURRENT RGB: overlook, 640x360
[image: overview camera]
CURRENT RGB: right_wrist, 384x288
[image: right wrist camera]
[Reason stated with the next call, call 21: execute_eef]
"Retract arm while raising to clear sink and reach a feasible carrying
pose."
```

Post-trained policies. A call to execute pi05 returns the same structure, with a controller record in place of the waypoint receipts. The excerpt is call 1 of the same episode.

```jsonl
[Request, call 1: execute_pi05]
{"cameras": ["overlook", "right_wrist"], "instruction": null, "reason":
"Approach and align right hand with a bowl rim in the sink.",
"steps": 120}
[Response]
{"executed_waypoints": [{"controller": "pi05", "requested_steps": 120,
"executed_steps": 120, "inference_calls": 4, "control_returned_to":
"agent", "episode_ended": false}], "remaining_tool_calls": null,
"remaining_physics_steps": null, "reported_token_usage": {[...]}}
Measured robot state (EEF poses in each arm's planner base):
{"state.gripper": [0.044, 0.044, 0.04398, 0.04398], "state.base":
[-7e-05, 0.09306, -0.0], "state.ee_pose": [[[0.10035, 8e-05,
0.16352], [1.0, -0.00018, 0.00103, 0.00099]], [[0.35261, -0.03683,
-0.00515], [0.70409, -0.00107, 0.71011, -0.00244]]], "timestep":
120}
[... captions and images of the overview and right wrist cameras ...]
```

A call to execute wam returns the same structure, and its controller record also states the fixed policy prompt and the number of inferences with a padded history frame. The excerpt is call 1 of the harness with InternW0-∆ on the same test scene.

[Request, call 1: execute\_wam]   
{"cameras": ["overlook", "right\_wrist"], "instruction": null, "reason":   
"Approach a bowl in the sink and align the right hand for a rim   
grasp.", "steps": 100}   
[Response]

```jsonl
{"executed_waypoints": [{"controller": "wam", "requested_steps": 100,
"prompt": {"prompt": "Take out one of the three bowls and stack it
on the plate.", "source": "episode_instruction", "fixed_for":
"episode", "prompt_sha256": "[hash]"}, "executed_steps": 100,
"inference_calls": 10, "padded_recent_frames": 4,
"control_returned_to": "agent", "episode_ended": false}],
"remaining_tool_calls": null, "remaining_physics_steps": null,
"reported_token_usage": {[...]}}
[... the measured state, and captions and images of the overview and
right wrist cameras ...]
```

Fields. Table A.3 explains the fields of a response, which are the same in the three configurations apart from the controller records of the two policies.

Table A.3: Fields of a tool response.
<table><tr><td>Field</td><td>Meaning</td></tr><tr><td>executed_waypoints (EEF)</td><td>One record per executed waypoint: its running index within the episode, the physics steps executed, and the measured state after it.</td></tr><tr><td>executed_waypoints (π0.5)</td><td>The controller, the requested and executed physics steps, the number of policy inferences (one per 30 steps), and whether control returned to the agent or the episode ended.</td></tr><tr><td>executed_waypoints (InternW0-∆)</td><td>As for πo.5, with one inference per 10 steps, together with the fixed policy prompt and its source, scope, and SHA-256 hash, and the number of inferences in the call whose frame from 32 steps earlier had not been observed and was padded.</td></tr><tr><td>state.gripper,state.base</td><td>The four finger-joint positions in meters, left arm first (0.044 per finger is fully open), and the planar base pose (x, y in meters, yaw in radians).</td></tr><tr><td>state.ee-pose,timestep</td><td>For each arm, the EEF position in meters and its wxyz quaternion in that arm&#x27;s planner-base frame, together with the physics steps executed since the start of the episode.</td></tr><tr><td>remaining-tool_calls, remaining-physics_steps</td><td>Always null, since no local limit applies and the benchmark&#x27;s step limit is not disclosed.</td></tr><tr><td>reported_token_usage</td><td>Cumulative token counts of the session, omitted above.</td></tr><tr><td>Measured robot state, CURRENT RGB</td><td>The measured state after the call as a separate text line, and for each requested camera its name and resolution followed by its image.</td></tr></table>

## B ADDITIONAL BENCHMARK RESULTS

## B.1 TASKS, LEADERBOARD, AND PER-TASK COMPARISONS

Tasks. Figure B.1 shows the initial overview frame of one test episode of each of the 26 tasks, grouped by mobility and horizon, with the precision label of each task.

Leaderboard. Table B.1 lists all entries of the EBench leaderboard by mobility group, including the entries not shown in Table 1: PICA-WAM (Zou et al., 2026), which fine-tunes OpenWAM-α with an action-alignment objective under visual perturbations, Vela, AMapbot, Cosmos3-Edge, and X-VLA, as well as InternW0-∆ with the evaluation of Appendix C. Among these 14 entries, the GPT-6-Astra-based agent under direct control ranks sixth in SR and fifth in score.

Per-task comparison. Tables B.2 and B.3 give the SR and score of the agent and the eight posttrained policies on every test task. Relative to the best post-trained policy on each task in Table B.2, the agent’s SR is higher on three tasks, equal on five, and lower on eighteen. The three higher tasks are all mobile: Bookmark on book, Remote to holder, and Teacup, saucer & teapot. The five equal tasks are Dish, Pen, Perfume to cosmetics rack, Salt to spice rack, and Bottle, which no policy solves. The largest deficits occur on Dishwasher, Flip cup & collect cookies, and Glass into glassbox, where the agent reaches 5%, 10%, and 20% against 90% for the best post-trained policy. These three tasks involve an articulated object, coordinated use of both arms, and two-arm transfer with folding, respectively. The outcome breakdown separates two kinds of shortfall. On Tighten nut, Peg in hole, and Shop, no episode ends with a score of zero, so unsuccessful episodes receive partial credit for completed stages. On Gear installation and Utensils to holder, 15 and 13 of the 20 episodes end with a score of zero, so in these episodes the agent fails before completing the first scored stage.

(a) Mobile, short horizon  
![](images/688e3a0e9137da1528201e0c52be17a573728ef1a0bee29c3339144a63f941d7.jpg)  
Apple from shelf Low precision

![](images/f23c3508277cd8b6ea774896f1046a8a83823b83633decfba2adeb5d18d75bdc.jpg)  
Apple to fruit bowl Low precision

![](images/053fe01953ef577eeda6a4832148141ae345e975b3971450c6985f7a469264c9.jpg)  
Bookmark on book Low precision

![](images/275ba5b8fd16f1d3b8e7fffa56e934df352658b79116af64659cef1c309006ec.jpg)  
Bowl to plate Low precision

![](images/35e59c3aae4481c244743361508460fb658aa88d6239bdf927c615837c4255d8.jpg)  
Perfume to cosmetics rack Low precision

![](images/15772d5e4b9d7b342cd95feceea0747bcdb35b343e15b69c6ea5175314019b3b.jpg)  
Remote to holder Low precision

![](images/ea54fb1010c331194ea5b261a59f69b90122e33b052b7cfac022e3ba03be63f7.jpg)  
Salt to spice rack Low precision

![](images/25b8d1ee0ab9bea80f15df21b0b1eeef1df738a3e717dff4dfe2332a5273b96e.jpg)  
Utensils to holder Low precision

![](images/315be1d1b656c3ae1308a1a0d21c92fa253b482201de70b7dfc658e78bb04d72.jpg)  
Soap to dish Medium precision

![](images/0576a7e8703645ec73f8081a306ae17f5ea2f718dfc734b6070471ccf12b8c0a.jpg)  
Teacup, saucer & teapot Medium precision

(b) Mobile, long horizon  
![](images/7d0cd7cf0afa8888f408c4d38af428418da0a5ce46cb707c6ed86eec44fcd242.jpg)  
Detergent Low precision

![](images/7f89fe58fcdb585ebc28a09ca6025eb3b47228d1ceab3858f6050498ad630d48.jpg)  
Dishwasher Low precision

![](images/2286f8c47f20a74fc985e2268de36f6b4a54ba8f3c5398ad90fc7c960d4fade9.jpg)  
Fruit Low precision

![](images/9e7464380f6c58d6231965e484297af9d3c6f9a26c968691569d86776e9a7b0b.jpg)  
Microwave Low precision

![](images/d372473758ca1a60143987b603bfa3386d9bfb33a46acf90fd4c2ba9f703344d.jpg)  
Shop Low precision

![](images/206187288fe310176897ac88213956384d7830b98a94b418000a4a6293464073.jpg)  
Bottle Medium precision

![](images/bb89ea93b9216a72e2b228950218f7977b13d573e55fbcfe497f17e979e7c2ce.jpg)  
Dish Medium precision

![](images/a48b9dbd3a581c962d9d7d0826cece08421ac637503ce9b76242fa5555a062ab.jpg)  
Make sandwich Medium precision

![](images/0cd29adaf6605561c21a66d53a8759bc6d487762002b900e231905682eaf4e7a.jpg)  
Pen Medium precision

(c) Tabletop, long horizon except gear installation (short horizon)  
![](images/dca2fbaecf87509a0aeeffa9659ba64522b620c775c5923c9f5731416ab3a797.jpg)  
Gear installation High precision

![](images/3a88cf7ba4786ed4090219219d301ef9a3cceb2c99199b273d4b1bfa0d1c235f.jpg)  
Frame placement Low precision

![](images/ad2ef4b2e64a8b3a3796c21367ed96327d0f60811aa78e6d6b7645142d888651.jpg)  
Flip cup & collect cookies Medium precision

![](images/20f764726e007caf28d4190489c10ee07016dd42b22d1d1272784fc9232a58f8.jpg)  
Glass into glassbox Medium precision

![](images/09836d29982cd45289858b09a2badc634f5c3ec3a423b81782a3d956785db0b9.jpg)  
Collect co ee beans High precision

![](images/4471ad55d9d3d854f2382be1e80afb01da1300061284b99d6d8eb7a1a117d9b9.jpg)  
Peg in hole High precision

![](images/ff754bf6c72477dbffce7db13195bcfd9678975932cc977ed7b7df56de810c43.jpg)  
Tighten nut High precision

Figure B.1: Initial overview frame of one direct-control test episode of each EBench task, by mobility and horizon, with the precision label of each task. Task names as in Table B.2. The tabletop frames show the upper central 80% of the view.

Generalization dimensions. Table B.4 gives SR and score in the four generalization dimensions. The Object dimension covers 24 tasks (120 episodes) and the other dimensions cover all 26 tasks (130 episodes), with five episodes per task and dimension.

Capability profile. Figure B.2 compares the three highest-ranked systems among the agent and the eight post-trained policies by task group, as discussed in Section 5.2.

Table B.1: SR and score of all entries of the EBench leaderboard, overall and by mobility (19 mobile and 7 tabletop tasks, each averaged equally over tasks), with the best and second-best values in each column shown in bold and underlined. The results of the post-trained policies are taken from the leaderboard.
<table><tr><td></td><td></td><td colspan="2">Overall</td><td colspan="2">Mobile</td><td colspan="2">Tabletop</td></tr><tr><td>Policy</td><td>Type</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td></tr><tr><td colspan="8">Post-trained policies</td></tr><tr><td>PICA-WAM (Zou et al., 2026)</td><td>WAM</td><td>61.4</td><td>0.749</td><td>67.5</td><td>0.783</td><td>45.0</td><td>0.663</td></tr><tr><td>OpenWAM-α (Wang et al., 2026b)</td><td>WAM</td><td>55.3</td><td>0.701</td><td>60.2</td><td>0.740</td><td>42.1</td><td>0.594</td></tr><tr><td>Vela (Li et al., 2026b)</td><td>VLA</td><td>49.7</td><td>0.655</td><td>51.7</td><td>0.661</td><td>44.3</td><td>0.641</td></tr><tr><td>InternW0-∆ (Miao et al., 2026)</td><td>WAM</td><td>49.2</td><td>0.660</td><td>55.0</td><td>0.700</td><td>33.6</td><td>0.552</td></tr><tr><td>AMapbot</td><td></td><td>48.9</td><td>0.639</td><td>49.3</td><td>0.642</td><td>47.9</td><td>0.630</td></tr><tr><td>Qwen-RobotManip (Yuan et al., 2026a)</td><td>VLA</td><td>45.6</td><td>0.608</td><td>43.9</td><td>0.576</td><td>50.0</td><td>0.696</td></tr><tr><td>π0.5 (Black et al., 2025a)</td><td>VLA</td><td>41.4</td><td>0.544</td><td>45.9</td><td>0.578</td><td>29.3</td><td>0.452</td></tr><tr><td>InternVLA-A1.5 (Ma et al., 2026a)</td><td>VLA</td><td>34.2</td><td>0.468</td><td>42.3</td><td>0.562</td><td>12.1</td><td>0.215</td></tr><tr><td>π0 (Black et al., 2025b)</td><td>VLA</td><td>33.7</td><td>0.475</td><td>34.0</td><td>0.469</td><td>32.9</td><td>0.491</td></tr><tr><td>GigaBrain-0.7 (GigaBrain Team et al., 2026)</td><td>VLA</td><td>33.3</td><td>0.460</td><td>31.6</td><td>0.413</td><td>37.9</td><td>0.588</td></tr><tr><td>Cosmos3-Edge (NVIDIA, 2026)</td><td>WAM</td><td>29.3</td><td>0.425</td><td>33.2</td><td>0.466</td><td>18.6</td><td>0.316</td></tr><tr><td>Fast-WAM (Yuan et al., 2026b)</td><td>WAM</td><td>25.6</td><td>0.371</td><td>34.0</td><td>0.460</td><td>2.9</td><td>0.129</td></tr><tr><td>X-VLA (Zheng et al., 2025)</td><td>VLA</td><td>23.7</td><td>0.358</td><td>29.3</td><td>0.402</td><td>8.6</td><td>0.241</td></tr><tr><td colspan="8">GPT-6-Astra-based agent, no robot training</td></tr><tr><td>Direct control</td><td>MLLM</td><td>46.7</td><td>0.654</td><td>56.6</td><td>0.730</td><td>20.0</td><td>0.447</td></tr></table>

![](images/e92195f5f8fd1df28e6264c842a03fd807055855d1dc6aa60fa2627784f027d8.jpg)  
Figure B.2: SR of OpenWAM-α, InternW0-∆, and the GPT-6-Astra-based agent (direct control), the three highest-ranked systems in the per-task comparison, by precision, mobility, horizon, and generalization dimension on the 26 EBench test tasks.

Table B.2: SR (%) per task on the EBench test set, with the best and second-best value of each task in bold and underlined. OWAM: OpenWAM-α, IW0-∆: InternW0-∆, IVLA: InternVLA-A1.5, Giga: GigaBrain-0.7, FWAM: Fast-WAM. The direct-control run evaluated Bottle and Shop with a step limit of 3,000 steps instead of 5,000.
<table><tr><td>Task</td><td></td><td>Prec. Mob.</td><td>Hor.</td><td>OWAM</td><td>IW0- Δ</td><td>Astra</td><td>Qwen</td><td> $\pi _ { 0 . 5 }$ </td><td>IVLA</td><td> $\pi _ { 0 }$ </td><td>Giga</td><td>FWAM</td></tr><tr><td>Apple from shelf</td><td>Low</td><td>Mob.</td><td>Short</td><td>70.0</td><td>15.0</td><td>70.0</td><td>65.0</td><td>75.0</td><td>35.0</td><td>40.0</td><td>55.0</td><td>40.0</td></tr><tr><td>Apple to fruit bowl</td><td>Low</td><td>Mob.</td><td>Short</td><td>95.0</td><td>95.0</td><td>90.0</td><td>50.0</td><td>25.0</td><td>20.0</td><td>10.0</td><td>20.0</td><td>70.0</td></tr><tr><td>Bookmark on book</td><td>Low</td><td>Mob.</td><td>Short</td><td>35.0</td><td>35.0</td><td>90.0</td><td>40.0</td><td>55.0</td><td>45.0</td><td>25.0</td><td>30.0</td><td>30.0</td></tr><tr><td>Bowl to plate</td><td>Low</td><td>Mob.</td><td>Short</td><td>85.0</td><td>80.0</td><td>75.0</td><td>75.0</td><td>85.0</td><td>60.0</td><td>60.0</td><td>65.0</td><td>75.0</td></tr><tr><td>Gear installation</td><td>High</td><td>Tab.</td><td>Short</td><td>40.0</td><td>50.0</td><td>15.0</td><td>50.0</td><td>10.0</td><td>15.0</td><td>30.0</td><td>20.0</td><td>15.0</td></tr><tr><td>Perfume to cosmetics rack Low</td><td></td><td>Mob.</td><td>Short</td><td>75.0</td><td>30.0</td><td>75.0</td><td>55.0</td><td>60.0</td><td>45.0</td><td>25.0</td><td>25.0</td><td>50.0</td></tr><tr><td>Remote to holder</td><td>Low</td><td>Mob.</td><td>Short</td><td>65.0</td><td>85.0</td><td>100.0</td><td>65.0</td><td>60.0</td><td>60.0</td><td>70.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Salt to spice rack</td><td>Low</td><td>Mob.</td><td>Short</td><td>100.0</td><td>100.0</td><td>100.0</td><td>70.0</td><td>70.0</td><td>90.0</td><td>80.0</td><td>75.0</td><td>80.0</td></tr><tr><td>Soap to dish</td><td>Med.</td><td>Mob.</td><td>Short</td><td>85.0</td><td>100.0</td><td>95.0</td><td>85.0</td><td>100.0</td><td>95.0</td><td>95.0</td><td>70.0</td><td>90.0</td></tr><tr><td>Teacup, saucer &amp; teapot</td><td>Med.</td><td>Mob.</td><td>Short</td><td>40.0</td><td>45.0</td><td>85.0</td><td>45.0</td><td>30.0</td><td>30.0</td><td>10.0</td><td>20.0</td><td>10.0</td></tr><tr><td>Utensils to holder</td><td>Low</td><td>Mob.</td><td>Short</td><td>35.0</td><td>15.0</td><td>10.0</td><td>15.0</td><td>5.0</td><td>20.0</td><td>0.0</td><td>10.0</td><td>0.0</td></tr><tr><td>Bottle</td><td></td><td>Med. Mob.</td><td>Long</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Collect coffee beans</td><td>High</td><td>Tab.</td><td>Long</td><td>20.0</td><td>5.0</td><td>0.0</td><td>5.0</td><td>30.0</td><td>0.0</td><td>20.0</td><td>50.0</td><td>0.0</td></tr><tr><td>Detergent</td><td>Low</td><td>Mob.</td><td>Long</td><td>95.0</td><td>95.0</td><td>55.0</td><td>85.0</td><td>100.0</td><td>95.0</td><td>75.0</td><td>95.0</td><td>35.0</td></tr><tr><td>Dish</td><td>Med.</td><td>Mob.</td><td>Long</td><td>40.0</td><td>50.0</td><td>50.0</td><td>20.0</td><td>25.0</td><td>20.0</td><td>0.0</td><td>5.0</td><td>0.0</td></tr><tr><td>Dishwasher</td><td>Low</td><td>Mob.</td><td>Long</td><td>90.0</td><td>85.0</td><td>5.0</td><td>15.0</td><td>65.0</td><td>90.0</td><td>90.0</td><td>0.0</td><td>70.0</td></tr><tr><td>Flip cup &amp; collect cookies Med.</td><td></td><td>Tab.</td><td>Long</td><td>40.0</td><td>50.0</td><td>10.0</td><td>90.0</td><td>25.0</td><td>30.0</td><td>30.0</td><td>40.0</td><td>0.0</td></tr><tr><td>Frame placement</td><td>Low</td><td>Tab.</td><td>Long</td><td>50.0</td><td>60.0</td><td>65.0</td><td>95.0</td><td>45.0</td><td>0.0</td><td>25.0</td><td>70.0</td><td>0.0</td></tr><tr><td>Fruit</td><td>Low</td><td>Mob.</td><td>Long</td><td>55.0</td><td>60.0</td><td>40.0</td><td>45.0</td><td>10.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Make sandwich</td><td>Med.</td><td>Mob.</td><td>Long</td><td>80.0</td><td>73.3</td><td>46.7</td><td>66.7</td><td>66.7</td><td>33.3</td><td>20.0</td><td>80.0</td><td>46.7</td></tr><tr><td>Microwave</td><td>Low</td><td>Mob.</td><td>Long</td><td>93.3</td><td>6.7</td><td>73.3</td><td>33.3</td><td>40.0</td><td>60.0</td><td>46.7</td><td>0.0</td><td>0.0</td></tr><tr><td>Peg in hole</td><td>High</td><td>Tab.</td><td>Long</td><td>40.0</td><td>10.0</td><td>20.0</td><td>10.0</td><td>10.0</td><td>0.0</td><td>15.0</td><td>15.0</td><td>0.0</td></tr><tr><td>Pen</td><td>Med.</td><td>Mob.</td><td>Long</td><td>5.0</td><td>15.0</td><td>15.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Glass into glassbox</td><td>Med.</td><td>Tab.</td><td>Long</td><td>75.0</td><td>60.0</td><td>20.0</td><td>90.0</td><td>65.0</td><td>0.0</td><td>85.0</td><td>55.0</td><td>5.0</td></tr><tr><td>Shop</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.0</td><td>60.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Tighten nut</td><td>High</td><td>Tab.</td><td>Long</td><td>30.0</td><td>0.0</td><td>10.0</td><td>10.0</td><td>20.0</td><td>40.0</td><td>25.0</td><td>15.0</td><td>0.0</td></tr></table>

Table B.3: Score per task on the EBench test set, with the best and second-best value of each task in bold and underlined. Abbreviations and step limits as in Table B.2.
<table><tr><td>Task</td><td>Prec.</td><td>Mob.</td><td>Hor.</td><td>OWAM</td><td>IW0- Δ</td><td>Astra</td><td>Qwen</td><td>π0.5</td><td>IVLA</td><td>π0</td><td>Giga</td><td>FWAM</td></tr><tr><td>Apple from shelf</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.70</td><td>0.15</td><td>0.70</td><td>0.65</td><td>0.75</td><td>0.35</td><td>0.40</td><td>0.55</td><td>0.40</td></tr><tr><td>Apple to fruit bowl</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.95</td><td>0.95</td><td>0.90</td><td>0.50</td><td>0.25</td><td>0.20</td><td>0.10</td><td>0.20</td><td>0.70</td></tr><tr><td>Bookmark on book</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.35</td><td>0.35</td><td>0.90</td><td>0.40</td><td>0.55</td><td>0.45</td><td>0.25</td><td>0.30</td><td>0.30</td></tr><tr><td>Bowl to plate</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.85</td><td>0.80</td><td>0.75</td><td>0.75</td><td>0.85</td><td>0.60</td><td>0.60</td><td>0.65</td><td>0.75</td></tr><tr><td>Gear installation</td><td>High</td><td>Tab.</td><td>Short</td><td>0.50</td><td>0.60</td><td>0.20</td><td>0.58</td><td>0.13</td><td>0.20</td><td>0.38</td><td>0.28</td><td>0.15</td></tr><tr><td>Perfume to cosmetics rack Low</td><td></td><td>Mob.</td><td>Short</td><td>0.78</td><td>0.30</td><td>0.88</td><td>0.60</td><td>0.68</td><td>0.60</td><td>0.30</td><td>0.33</td><td>0.50</td></tr><tr><td>Remote to holder</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.65</td><td>0.85</td><td>1.00</td><td>0.65</td><td>0.60</td><td>0.60</td><td>0.70</td><td>0.50</td><td>0.50</td></tr><tr><td>Salt to spice rack</td><td>Low</td><td>Mob.</td><td>Short</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.70</td><td>0.70</td><td>0.90</td><td>0.80</td><td>0.75</td><td>0.80</td></tr><tr><td>Soap to dish</td><td>Med.</td><td>Mob.</td><td>Short</td><td>0.85</td><td>1.00</td><td>0.95</td><td>0.85</td><td>1.00</td><td>0.95</td><td>0.95</td><td>0.70</td><td>0.90</td></tr><tr><td>Teacup, saucer &amp; teapot</td><td>Med.</td><td>Mob.</td><td>Short</td><td>0.53</td><td>0.58</td><td>0.93</td><td>0.50</td><td>0.40</td><td>0.33</td><td>0.15</td><td>0.28</td><td>0.23</td></tr><tr><td>Utensils to holder</td><td>Low</td><td>Mob.</td><td>Short</td><td>0.55</td><td>0.45</td><td>0.23</td><td>0.40</td><td>0.30</td><td>0.35</td><td>0.20</td><td>0.23</td><td>0.23</td></tr><tr><td>Bottle</td><td>Med. Mob.</td><td></td><td>Long</td><td>0.32</td><td>0.44</td><td>0.17</td><td>0.17</td><td>0.05</td><td>0.03</td><td>0.02</td><td>0.04</td><td>0.03</td></tr><tr><td>Collect coffee beans</td><td>High</td><td>Tab.</td><td>Long</td><td>0.40</td><td>0.11</td><td>0.24</td><td>0.51</td><td>0.55</td><td>0.16</td><td>0.35</td><td>0.71</td><td>0.00</td></tr><tr><td>Detergent</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.98</td><td>0.98</td><td>0.80</td><td>0.93</td><td>1.00</td><td>0.97</td><td>0.83</td><td>0.98</td><td>0.67</td></tr><tr><td>Dish</td><td>Med.</td><td>Mob.</td><td>Long</td><td>0.88</td><td>0.89</td><td>0.81</td><td>0.76</td><td>0.70</td><td>0.80</td><td>0.45</td><td>0.30</td><td>0.44</td></tr><tr><td>Dishwasher</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.95</td><td>0.93</td><td>0.53</td><td>0.27</td><td>0.73</td><td>0.90</td><td>0.95</td><td>0.50</td><td>0.87</td></tr><tr><td>Flip cup &amp; collect cookies Med.</td><td></td><td>Tab.</td><td>Long</td><td>0.59</td><td>0.91</td><td>0.44</td><td>0.97</td><td>0.44</td><td>0.45</td><td>0.35</td><td>0.61</td><td>0.08</td></tr><tr><td>Frame placement</td><td>Low</td><td>Tab.</td><td>Long</td><td>0.60</td><td>0.70</td><td>0.80</td><td>0.97</td><td>0.48</td><td>0.00</td><td>0.40</td><td>0.72</td><td>0.10</td></tr><tr><td>Fruit</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.76</td><td>0.84</td><td>0.59</td><td>0.68</td><td>0.38</td><td>0.60</td><td>0.10</td><td>0.27</td><td>0.05</td></tr><tr><td>Make sandwich</td><td>Med.</td><td>Mob.</td><td>Long</td><td>0.95</td><td>0.93</td><td>0.70</td><td>0.90</td><td>0.92</td><td>0.78</td><td>0.75</td><td>0.95</td><td>0.80</td></tr><tr><td>Microwave</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.98</td><td>0.64</td><td>0.90</td><td>0.38</td><td>0.50</td><td>0.82</td><td>0.81</td><td>0.00</td><td>0.45</td></tr><tr><td>Peg in hole</td><td>High</td><td>Tab.</td><td>Long</td><td>0.60</td><td>0.33</td><td>0.60</td><td>0.35</td><td>0.23</td><td>0.00</td><td>0.43</td><td>0.55</td><td>0.00</td></tr><tr><td>Pen</td><td>Med.</td><td>Mob.</td><td>Long</td><td>0.37</td><td>0.32</td><td>0.55</td><td>0.25</td><td>0.05</td><td>0.28</td><td>0.10</td><td>0.03</td><td>0.05</td></tr><tr><td>Glass into glassbox</td><td>Med.</td><td>Tab.</td><td>Long</td><td>0.85</td><td>0.72</td><td>0.30</td><td>0.95</td><td>0.77</td><td>0.00</td><td>0.92</td><td>0.68</td><td>0.10</td></tr><tr><td>Shop</td><td>Low</td><td>Mob.</td><td>Long</td><td>0.67</td><td>0.90</td><td>0.60</td><td>0.60 0.55</td><td>0.58</td><td>0.17</td><td>0.44</td><td>0.30</td><td>0.09</td></tr><tr><td>Tighten nut</td><td>High</td><td>Tab.</td><td>Long</td><td>0.63</td><td>0.50</td><td>0.55</td><td></td><td>0.58</td><td>0.70</td><td>0.63</td><td>0.58</td><td>0.48</td></tr></table>

Table B.4: SR (%) and score of the GPT-6-Astra-based agent under direct control and the eight post-trained policies in the Object, Background, Instruction, and Mix generalization dimensions of the EBench test set, with the best and second-best value of each column in bold and underlined. Rows below the midrule are excluded from this ranking and report the agent in the harness with each policy, as analyzed in Section 6.3.
<table><tr><td rowspan="2">Policy</td><td colspan="2">Object</td><td colspan="2">Background</td><td colspan="2">Instruction</td><td colspan="2">Mix</td></tr><tr><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td><td>SR (%)</td><td>Score</td></tr><tr><td>OpenWAM-α</td><td>52.5</td><td>0.657</td><td>61.5</td><td>0.767</td><td>60.0</td><td>0.751</td><td>44.6</td><td>0.604</td></tr><tr><td>InternW0-∆</td><td>49.2</td><td>0.645</td><td>51.5</td><td>0.672</td><td>52.3</td><td>0.694</td><td>44.6</td><td>0.619</td></tr><tr><td>GPT-6-Astra (direct control)</td><td>44.2</td><td>0.648</td><td>50.8</td><td>0.692</td><td>44.6</td><td>0.620</td><td>46.2</td><td>0.643</td></tr><tr><td>Qwen-RobotManip</td><td>40.8</td><td>0.565</td><td>55.4</td><td>0.722</td><td>51.5</td><td>0.655</td><td>33.8</td><td>0.485</td></tr><tr><td>π0.5</td><td>41.7</td><td>0.534</td><td>42.3</td><td>0.575</td><td>46.2</td><td>0.597</td><td>34.6</td><td>0.457</td></tr><tr><td>InternVLA-A1.5</td><td>33.3</td><td>0.433</td><td>38.5</td><td>0.515</td><td>33.8</td><td>0.463</td><td>30.0</td><td>0.434</td></tr><tr><td>π0</td><td>28.3</td><td>0.422</td><td>42.3</td><td>0.547</td><td>39.2</td><td>0.508</td><td>24.6</td><td>0.394</td></tr><tr><td>GigaBrain-0.7</td><td>27.5</td><td>0.405</td><td>42.3</td><td>0.541</td><td>36.9</td><td>0.502</td><td>25.4</td><td>0.389</td></tr><tr><td>Fast-WAM</td><td>21.7</td><td>0.295</td><td>32.3</td><td>0.440</td><td>26.9</td><td>0.394</td><td>21.5</td><td>0.330</td></tr><tr><td>Harness with π0.5</td><td>59.2</td><td>0.728</td><td>50.0</td><td>0.689</td><td>56.2</td><td>0.719</td><td>46.2</td><td>0.623</td></tr><tr><td>Harness with InternW0-∆</td><td>57.5</td><td>0.749</td><td>54.6</td><td>0.739</td><td>57.7</td><td>0.757</td><td>53.1</td><td>0.718</td></tr></table>

## B.2 EVALUATION PROTOCOL AND EVENT DEFINITIONS

Evaluation records. The horizon labels follow the corrected annotation of the benchmark design ers, which differs from the grouping displayed on the leaderboard. When the agent ends an episode, the robot is held in place until the evaluation server determines the outcome, and 6 of the 49 directcontrol episodes that the agent ended early succeeded during this hold. In the 56 direct-control episodes interrupted by an infrastructure failure, execution resumed from the unchanged simulator state in a new model thread initialized with the recorded action history and the current observa tions. The direct-control run evaluated Bottle and Shop with a step limit of 3,000 steps, whereas the benchmark’s current protocol, used by both harness configurations, sets a step limit of 5,000 steps for these two tasks. Within 3,000 steps, no episode of either task succeeds in any configuration. Under a step limit of 5,000 steps, the harness with policy solves 3 of 20 Shop episodes with $\pi _ { 0 . 5 }$ , 1 with InternW0-∆, and none of Bottle. These episodes raise the overall SR of the harness configurations from 52.4% to 53.0% and from 55.8% to 56.0%, respectively.

Comparison protocol. The results of the post-trained policies come from the public leaderboard, and those of InternW0-∆ alone come from the evaluation of Appendix C, which is its leaderboard entry. Comparisons between configurations of the agent, and between InternW0-∆ alone and its harness configuration, pair test episodes that were run at different times, and each configuration was run once per episode. The two harness configurations differ in the policy’s replanning interval and history as well as in the policy itself. The demonstrations include top-camera keyframes that the agent does not observe during execution, and for 31 of the 510 test episodes the demonstration is a recording of the test scene itself. The demonstration of Frame placement records the scene of four of its 20 test episodes. In the comparison with and without the demonstration (Appendix D.3), the two conditions were rendered with partly different textures. The recordings shown in the figures were selected to illustrate each behavior.

Grasp events. The frequencies of disruptions and grip losses in Section 5 are computed from the finger gaps and commanded finger targets of both arms, recorded after every executed chunk of at most 8 physics steps, and from the stated reasons of all calls in the 510 direct-control episodes, including the segments resumed after an interruption. The finger gap of a hand is the summed opening of its two fingers, which reaches 88 mm when the hand is fully open, or 0.044 m per finger joint in state.gripper. A grip counts as closed on an object when, after a close command, the fingers come to rest between 2 and 80 mm apart over two samples at different timesteps, so that fingertips closed for pushing do not count, and as lost when the gap then collapses while the gripper is still commanded closed. A stated reason counts as reporting a disruption when it names a slip, a drop, a fall, a displacement, or a loss outside a preventive phrase such as “avoid” or “in case”, and words of an intended release such as “released” do not count. A loss counts as occurring in a grasp test when the reason of its call contains test, verify, confirm, or check as a whole word, or names a one-centimeter lift. In a random sample of 20 of the other losses, read together with the arm that loses the grip, 16 occur while an object is lifted or carried, 3 during door manipulation, and 1 during a placement.

Wrist and joint speed. The speed measures of Section 5.5 use the same chunk records, from all 510 episodes under direct control and from the segments of $\pi _ { 0 . 5 }$ in all 510 episodes of the harness with $\pi _ { 0 . 5 } ,$ as listed in Table B.5. Between consecutive chunks of one command, the wrist rotation is the angle between the measured EEF orientations of an arm, and the joint speed is the largest change of any arm joint, both divided by the executed physics steps. The two controllers differ in the tail of the rotation rates, as a single command of the agent can complete a large reorientation within a few chunks. In 38 commands, spread over 31 episodes, a wrist turns through $9 0 ^ { \circ }$ or more within 24 physics steps, whereas no segment of $\pi _ { 0 . 5 }$ reaches this threshold. Most of these commands reorient or reposition the hand for a new approach. Fast joint motion shows the same contrast, as transitions above 0.08 rad per physics step occur more than five times as often under direct control.

## B.3 STATISTICS

Aggregate differences between configurations are tested with a bootstrap over episodes within each task (5,000 resamples), which keeps the task composition fixed, while single-task differences use exact McNemar tests with Holm correction over the 26 tasks. For the harness with $\pi _ { 0 . 5 }$ against direct control, the 95% SR interval is $[ + 2 . 0 , + 1 0 . 5 ]$ points when episodes are resampled within each task and $[ - 2 . 5 , + 1 5 . 6 ]$ when tasks are resampled, and the score interval is $[ + 0 . 0 0 \dot { 6 } , + 0 . 0 7 1 ]$ . By group, the SR intervals are $[ + 6 . 8 , + 1 9 . 0 ]$ for long-horizon, $[ + 3 . 6 , + 1 9 . 3 ]$ for tabletop, $[ - 8 . { \dot { 6 } } , + 3 . 2 ]$ for short-horizon, and $[ - 1 1 . 3 , + 3 . 8 ]$ for high-precision tasks, and the score interval for high-precision tasks is $[ - 0 . 1 6 3 , - \mathrm { \dot { 0 } } . 0 2 9 ]$ . The gain of the harness with InternW0- $\cdot \Delta$ over direct control is signifi cant under both resampling schemes, and on the short-horizon tasks this configuration also exceeds InternW0-∆ alone under both. After Holm correction over the 26 tasks, Glass into glassbox and Dishwasher rise significantly over direct control with either tool, Teacup, saucer & teapot falls significantly with $\pi _ { 0 . 5 } ,$ , and against InternW0-∆ alone, the corresponding harness configuration gains significantly on Apple from shelf and Perfume to cosmetics rack while losing significantly on Shop.

Table B.5: Wrist and joint speed between consecutive chunks of one command under direct control and in the segments of $\pi _ { 0 . 5 }$ in the harness with $\pi _ { 0 . 5 } ,$ over all 510 test episodes of each configuration. Parentheses give the number of episodes. A window spans three consecutive chunk transitions of one command.
<table><tr><td></td><td>Direct control</td><td> $\pi _ { 0 . 5 }$  in the harness</td></tr><tr><td>Chunk transitions measured</td><td>109,079</td><td>88,573</td></tr><tr><td>Median wrist rotation (° per step)</td><td>0.00</td><td>0.08</td></tr><tr><td>99th percentile  $\binom { \circ } { }$  per step)</td><td>2.81</td><td>2.58</td></tr><tr><td>99.9th percentile (° per step)</td><td>4.06</td><td>3.81</td></tr><tr><td>Transitions above  $3 ^ { \circ }$  per step (episodes)</td><td>643 (147)</td><td>196 (94)</td></tr><tr><td>Commands turning a wrist by  $9 0 ^ { \circ }$  within a window of 24 steps (episodes)</td><td>38 (31)</td><td>0 (0)</td></tr><tr><td>Transitions with a joint above 0.08 rad per step (episodes)</td><td>606 (169)</td><td>89 (59)</td></tr></table>

## C HARNESS DETAILS

## C.1 POST-TRAINED POLICIES

$\pi _ { 0 . 5 } .$ The VLA tool is the $\pi _ { 0 . 5 }$ checkpoint fine-tuned on the EBench training set that appears as $\pi _ { 0 . 5 }$ on the leaderboard. A call executes 1–300 physics steps. The policy predicts 50 absolute joint and gripper targets per inference and replans every 30 steps under the fixed task instruction, and control returns to the agent after the requested steps with the same observations as an EEF command. The agent’s other tools and its prompt are those of Appendix A.

InternW0-∆. The WAM tool is an InternW0-∆ checkpoint (Miao et al., 2026) post-trained on EBench data for 77,150 steps. It represents the robot by joint and EEF state and, in 10 denoising steps, predicts 32 absolute joint and gripper targets and base motions, without generating video at inference, of which the first 10 are executed before the next inference. Each inference is conditioned on the three camera views, the proprioceptive state, the first frame of the episode, and the frame 32 steps earlier, under the fixed task instruction. The harness opens one model session per episode with the episode’s initial observation and executes each 10-action chunk as 8 and 2 steps, so that the frame 32 steps back is observed within a call. During the first inferences after the agent’s own control, that frame is unavailable and is padded as at an episode start, which affects $^ { 4 , 2 7 6 }$ of the 69,054 inferences. The agent invokes the policy 8.1 times per episode, with 135 inferences, against 45 inferences of $\pi _ { 0 . 5 }$ in the configuration with $\pi _ { 0 . 5 }$ . Its standalone evaluation ran the same checkpoint with the same horizon, execution length, and history on all 510 test episodes with the original inference loop in which the earlier frame is always observed. This standalone evaluation is the entry of InternW0-∆ on the public leaderboard.

## C.2 ADDITIONAL RESULTS

Per-task results. Table C.1 lists every task, ordered by the change in SR with $\pi _ { 0 . 5 } .$ . The harness with $\pi _ { 0 . 5 }$ exceeds both the agent under direct control and $\pi _ { 0 . 5 }$ alone on ten tasks and falls below the better of the two on thirteen. The ten tasks are Apple from shelf, Apple to fruit bowl, Bowl to plate, Dish, Flip cup & collect cookies, Make sandwich, Perfume to cosmetics rack, Glass into glassbox, Utensils to holder, and Shop. On Shop, it solves 3 of 20 episodes at the 5,000-step limit, against none for $\pi _ { 0 . 5 }$ alone and none for the direct-control run at 3,000 steps. The harness with $\pi _ { 0 . 5 }$ loses 59 episodes that the agent solved under direct control. Of these, 11 come from Teacup, saucer & teapot, 6 from Fruit, and 5 each from Dish, Frame placement, and Microwave. Of the 91 episodes that it gains, 14 come from Glass into glassbox, 11 from Dishwasher, and 7 each from Detergent, Dish, and Flip cup & collect cookies. With InternW0-∆, the harness exceeds both the agent under direct control and InternW0-∆ alone on eight tasks and falls below the better of the two on eleven. These eight tasks are Apple from shelf, Apple to fruit bowl, Bowl to plate, Dish, Flip cup & collect cookies, Make sandwich, Perfume to cosmetics rack, and Glass into glassbox. Of its 49 lost episodes, 8 come from Frame placement and 7 from Teacup, saucer & teapot. Of its 96 gained episodes, 14 come from Dishwasher, 13 from Glass into glassbox, and 11 from Flip cup & collect cookies.

Policy invocation measures. The policy step share of an episode is the fraction of its executed physics steps that the policy executes, with EEF waypoints and gripper commands counted as the agent’s steps, and the share of a task is the mean over its episodes. For Figure $^ { 8 \mathrm { a } , }$ the executed steps of each episode are divided into tenths, and the policy’s share is computed within each tenth. A policy invocation counts as a recovery request in Table 3 when its stated reason contains a wholeword match for a term of recovery, retrying, slipping, dropping, missing, displacement, re-approach, regrasp, or failure, as in “Recover slipped soap and realign a deeper centered grasp”. In a random sample of 40 calls coded in this way, 33 clearly ask the policy to recover from a disruption, 6 are ambiguous, and 1 is not a recovery request. Both harness configurations enter these measures with their Bottle and Shop episodes at 5,000 physics steps, as in Table 2.

Table C.1: Per-task SR (%) of GPT-6-Astra under direct control and in the harness with each policy, of each policy alone, and the share (%) of physics steps executed by the policy in each harness configuration, with the best and second-best SR of each task in bold and underlined. The direct-control run evaluated Bottle and Shop with a step limit of 3,000 steps instead of 5,000.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Direct control</td><td colspan="2">Policy alone</td><td colspan="2">Harness with</td><td colspan="2">Policy step share</td></tr><tr><td>π0.5</td><td>InternW0-∆</td><td>π0.5</td><td>InternW0-∆</td><td>π0.5</td><td>InternW0-∆</td></tr><tr><td>Teacup, saucer &amp; teapot</td><td>85.0</td><td>30.0</td><td>45.0</td><td>30.0</td><td>55.0</td><td>51.9</td><td>48.9</td></tr><tr><td>Peg in hole</td><td>20.0</td><td>10.0</td><td>10.0</td><td>5.0</td><td>0.0</td><td>61.1</td><td>63.4</td></tr><tr><td>Microwave</td><td>73.3</td><td>40.0</td><td>6.7</td><td>60.0</td><td>60.0</td><td>75.0</td><td>71.2</td></tr><tr><td>Fruit</td><td>40.0</td><td>10.0</td><td>60.0</td><td>30.0</td><td>45.0</td><td>57.3</td><td>64.0</td></tr><tr><td>Pen</td><td>15.0</td><td>0.0</td><td>15.0</td><td>5.0</td><td>0.0</td><td>53.5</td><td>53.4</td></tr><tr><td>Remote to holder</td><td>100.0</td><td>60.0</td><td>85.0</td><td>90.0</td><td>100.0</td><td>67.4</td><td>54.0</td></tr><tr><td>Bookmark on book</td><td>90.0</td><td>55.0</td><td>35.0</td><td>85.0</td><td>90.0</td><td>47.0</td><td>46.0</td></tr><tr><td>Gear installation</td><td>15.0</td><td>10.0</td><td>50.0</td><td>10.0</td><td>40.0</td><td>59.4</td><td>72.3</td></tr><tr><td>Soap to dish</td><td>95.0</td><td>100.0</td><td>100.0</td><td>90.0</td><td>100.0</td><td>68.3</td><td>52.3</td></tr><tr><td>Bottle</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>41.8</td><td>88.1</td></tr><tr><td>Collect coffee beans</td><td>0.0</td><td>30.0</td><td>5.0</td><td>0.0</td><td>0.0</td><td>75.8</td><td>69.1</td></tr><tr><td>Frame placement</td><td>65.0</td><td>45.0</td><td>60.0</td><td>65.0</td><td>45.0</td><td>81.9</td><td>67.5</td></tr><tr><td>Salt to spice rack</td><td>100.0</td><td>70.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>79.0</td><td>74.0</td></tr><tr><td>Apple to fruit bowl</td><td>90.0</td><td>25.0</td><td>95.0</td><td>95.0</td><td>100.0</td><td>48.9</td><td>62.4</td></tr><tr><td>Tighten nut</td><td>10.0</td><td>20.0</td><td>0.0</td><td>15.0</td><td>10.0</td><td>53.6</td><td>47.0</td></tr><tr><td>Utensils to holder</td><td>10.0</td><td>5.0</td><td>15.0</td><td>15.0</td><td>15.0</td><td>55.7</td><td>54.4</td></tr><tr><td>Perfume to cosmetics rack</td><td>75.0</td><td>60.0</td><td>30.0</td><td>80.0</td><td>80.0</td><td>62.9</td><td>48.1</td></tr><tr><td>Dish</td><td>50.0</td><td>25.0</td><td>50.0</td><td>60.0</td><td>70.0</td><td>73.6</td><td>70.1</td></tr><tr><td>Shop</td><td>0.0</td><td>0.0</td><td>60.0</td><td>15.0</td><td>5.0</td><td>89.7</td><td>82.9</td></tr><tr><td>Apple from shelf</td><td>70.0</td><td>75.0</td><td>15.0</td><td>85.0</td><td>75.0</td><td>53.4</td><td>39.3</td></tr><tr><td>Bowl to plate</td><td>75.0</td><td>85.0</td><td>80.0</td><td>95.0</td><td>95.0</td><td>47.0</td><td>44.6</td></tr><tr><td>Make sandwich</td><td>46.7</td><td>66.7</td><td>73.3</td><td>73.3</td><td>86.7</td><td>82.3</td><td>86.1</td></tr><tr><td>Flip cup &amp; collect cookies</td><td>10.0</td><td>25.0</td><td>50.0</td><td>40.0</td><td>60.0</td><td>78.9</td><td>83.0</td></tr><tr><td>Detergent</td><td>55.0</td><td>100.0</td><td>95.0</td><td>90.0</td><td>70.0</td><td>88.4</td><td>83.2</td></tr><tr><td>Dishwasher</td><td>5.0</td><td>65.0</td><td>85.0</td><td>60.0</td><td>75.0</td><td>91.6</td><td>88.2</td></tr><tr><td>Glass into glassbox</td><td>20.0</td><td>65.0</td><td>60.0</td><td>85.0</td><td>80.0</td><td>93.8</td><td>85.4</td></tr></table>

Generalization dimensions. Of the direct-control successes, the harness with $\pi _ { 0 . 5 }$ loses 43 of 126 on the Background and Mix dimensions against 16 of 111 on Object and Instruction (Fisher’s exact test, $p = 0 . 0 0 0 5 )$ . With InternW0-∆, the corresponding counts are 31 of 126 and 18 of 111 $( p = 0 . 1 5 )$

Atomic skills. Figure C.1 groups the per-task SRs of Table C.1 by the atomic skills that EBench annotates for each task. With $\pi _ { 0 . 5 }$ and InternW0-∆ alike, mean SR rises most on pushing, pressing, and pulling, by 21 to 70 percentage points, and sweeping improves by 17 and 24 points, respectively. Several tasks require more than one of these skills, as Dishwasher counts toward pushing, pulling, and sweeping, and Glass into glassbox accounts for most of the increase on pressing. Insertion is the only skill whose mean SR drops in both harness configurations, from 36.7% under direct control to 30.0% with $\pi _ { 0 . 5 }$ and 26.7% with InternW0-∆, while the two policies alone reach 23.3% and 18.3%. On handover, the harness with InternW0-∆ reaches 45.0% against 65.0% for InternW0-∆ alone, it largest shortfall against the standalone policy among the skills.

![](images/52e3367fa1cd8c08a1188ce23d3a78a77e2d6a0a198cc41c6961a623c6b05566.jpg)  
Figure C.1: SR by atomic skill of the GPT-6-Astra-based agent under direct control, in the harness with $\pi _ { 0 . 5 }$ and InternW0-∆, and of each policy standalone. Each skill averages over the tasks that require it: placing (23 tasks), moving (17), handover (5), sweeping (4), inserting (3), pressing (3), flipping (2), pouring (2), pulling (2), and pushing (1). Grasping, which is required by all 26 tasks, is omitted. The radial axis ends at the largest SR, rounded up to a multiple of 10%.

Diagnosis. An episode is classified as agent-terminated when the agent stops before the step limit without succeeding. Phrase rules assign its final message to either an unconfirmed completion claim or a reported failure, and a manual check of ten episodes per run confirms this assignment. Table C.2 follows every episode of the agent under direct control to the same scene in each harness configuration. The two harness configurations newly fail 108 episodes that the agent solved under direct control: 59 with $\pi _ { 0 . 5 }$ and 49 with InternW0-∆, on 85 distinct test episodes. Each of these episodes was read call by call, with the executor, the finger widths before and after, the base motion, and the agent’s stated reason of every call. Each was then assigned the incident after which the unfinished stage was never completed, as Table C.3 lists. Each agent-terminated episode is assigned the decisive incident preceding termination: a slip of the agent’s grasp, a loss around a policy segment, or an unmet stage behind a completion claim. A grip counts as held when the fingers come to rest between 2 and 80 mm apart over two samples at different timesteps while commanded closed, and as lost when it collapses while still commanded closed, and a reported slip is a stated reason coded as in Appendix B.2. An unmaintained grasp is an agent command that follows a policy segment and leaves an arm holding an object unspecified rather than closing it, so that the arm holds its measured finger width. Unmaintained grasps are counted per arm, so the 338 and 316 of Section 6.4 fall in 337 and 313 agent commands. The comparison in Section 6.4 counts unspecified holding arms after the agent’s own commands within the same configuration, and its stratified odds ratios use the Mantel–Haenszel estimator over the tasks on which both kinds occur.

Table C.2: How the 510 episodes under direct control end and how the same scenes end in each harness configuration, by success, step limit, a completion claim, or a failure report of the agent, with the few episodes of other kinds omitted from the columns of the harness configurations.
<table><tr><td rowspan="2">Direct control</td><td rowspan="2">n</td><td colspan="4">Harness with π0.5</td><td colspan="4">Harness with InternW0-∆</td></tr><tr><td>Succ.</td><td>Limit</td><td>Claim</td><td>Fail</td><td>Succ.</td><td>Limit</td><td>Claim</td><td>Fail</td></tr><tr><td>Success</td><td>237</td><td>178</td><td>52</td><td>3</td><td>4</td><td>188</td><td>33</td><td>2</td><td>13</td></tr><tr><td>Step limit</td><td>229</td><td>72</td><td>140</td><td>6</td><td>10</td><td>76</td><td>135</td><td>8</td><td>9</td></tr><tr><td>Ended, claims completion</td><td>21</td><td>10</td><td>3</td><td>6</td><td>2</td><td>11</td><td>5</td><td>3</td><td>2</td></tr><tr><td>Ended, reports failure</td><td>22</td><td>8</td><td>6</td><td>0</td><td>8</td><td>9</td><td>4</td><td>0</td><td>9</td></tr><tr><td>Other</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr></table>

Table C.3: Decisive incident in the episodes that the agent solved under direct control and each harness configuration lost.
<table><tr><td>Decisive incident</td><td>With π0.5</td><td>With InternW0-∆</td></tr><tr><td>Object lost or misplaced during a policy segment or when the agent resumes control</td><td>16</td><td>30</td></tr><tr><td>Policy without the object, on the wrong object or subgoal, or undoing a stage</td><td>5</td><td>1</td></tr><tr><td>Agent&#x27;s own grasp slips</td><td>30</td><td>12</td></tr><tr><td>Completion claimed while a stage remains unmet</td><td>3</td><td>2</td></tr><tr><td>Other</td><td>5</td><td>4</td></tr><tr><td>Total</td><td>59</td><td>49</td></tr></table>

Invocation and outcome. Table C.5 compares successful and unsuccessful episodes of the two harness configurations. A motion command is a call of the EEF interface, the gripper command, or the policy, and a transition is a change of executor between two consecutive motion commands. Pooled over episodes, the number of transitions correlates with failure, with point-biserial correlations of $r = 0 . 4 2$ and $r = 0 . 4 4$ , and within the 23 and 18 tasks with both outcomes, unsuccessful episodes make more transitions on 22 and 17 tasks (Wilcoxon signed-rank $p = 2 . 1 \times 1 0 ^ { - 4 }$ and $\bar { 1 . 5 } \times 1 0 ^ { - 5 } )$ . They also issue more motion commands on every one of these tasks, with fewer transitions per motion command on 17 of the 23 tasks $( p = 0 . 0 1 5 )$ and 15 of the 18 $( p = 0 . 0 6 )$ . Episodes with a recovery request to the policy fail in 75.5% and 72.0% of cases, against 15.8% and 16.8% without one. Of the 251 and 239 episodes with a disruption report, 227 and 200 contain a recovery request.

Table C.4: Decisive incidents of the 164 test episodes that fail with both harness configurations, read call by call, per configuration, with the number of episodes whose two runs show the same incident and the tasks that contribute most.
<table><tr><td>Decisive incident</td><td> $\pi _ { 0 . 5 }$ </td><td></td><td>InternW0-∆ Same in both Main tasks</td><td></td></tr><tr><td>Agent&#x27;s own grasp slips (lift test or transport)</td><td>58</td><td>42</td><td>19</td><td>Utensils to holder, Peg in hole, Pen</td></tr><tr><td>Object lost or misplaced in a policy segment</td><td>37</td><td>53</td><td>10</td><td>Peg in hole, Pen, Shop</td></tr><tr><td>Contact stage never completed</td><td>19</td><td>24</td><td>12</td><td>Tighten nut, Gear installation, Peg in hole</td></tr><tr><td>Policy segment empty-handed or on the wrong object</td><td>16</td><td>15</td><td>4</td><td>Shop, Bottle</td></tr><tr><td>Step limit without a decisive loss</td><td>16</td><td>11</td><td>4</td><td>Bottle</td></tr><tr><td>Slip after resuming control without a maintained grasp</td><td>14</td><td>10</td><td>7</td><td>Collect coffee beans</td></tr><tr><td>False completion claim</td><td>3</td><td>9</td><td>2</td><td>Gear installation, Dishwasher</td></tr><tr><td>Other</td><td>1</td><td>0</td><td>0</td><td></td></tr><tr><td>Total</td><td>164</td><td>164</td><td>58</td><td></td></tr></table>

Table C.5: Policy invocation in the successful and unsuccessful episodes of the two harness configurations. Transitions are changes of executor between consecutive motion commands, and disruption reports are motion commands whose stated reasons report a slip, a drop, or a displaced object.
<table><tr><td rowspan="2"></td><td colspan="2">Harness with  $\pi _ { 0 . 5 }$ </td><td colspan="2">Harness with InternW0-∆</td></tr><tr><td>Successful</td><td>Unsuccessful</td><td>Successful</td><td>Unsuccessful</td></tr><tr><td>Episodes</td><td>269</td><td>241</td><td>284</td><td>226</td></tr><tr><td>Motion commands per episode</td><td>14.7</td><td>32.0</td><td>15.2</td><td>30.4</td></tr><tr><td>Transitions per episode</td><td>4.2</td><td>7.2</td><td>3.8</td><td>6.8</td></tr><tr><td>Transitions per motion command</td><td>0.33</td><td>0.26</td><td>0.28</td><td>0.24</td></tr><tr><td>Episodes ending with an agent command (%)</td><td>42.8</td><td>68.0</td><td>47.2</td><td>71.7</td></tr><tr><td>Episodes with a recovery request</td><td>24.5</td><td>84.2</td><td>25.0</td><td>81.0</td></tr><tr><td>to the policy (%) Disruption reports executed by the policy (%)</td><td>50.5</td><td>32.4</td><td>39.6</td><td>32.7</td></tr></table>

Grip losses by executor. Using the held state and the losses defined in Appendix B.2, every physics step between two samples of a held arm counts as a held arm-step of the executor of its call, and rates are given per 1,000 held arm-steps. The harness with $\pi _ { 0 . 5 }$ holds objects for 278,496 arm-steps under the policy and 146,922 under the agent, with 176 and 321 losses, corresponding to rates of 0.63 and 2.18 per 1,000 held arm-steps. With InternW0-∆, the corresponding values are 262,037 and 142,024 arm-steps, with 189 and 315 losses, or 0.72 and 2.22 per 1,000. Of the agent’s losses, 193 and 173 fall in grasp-test calls, and outside these calls the agent’s rates are 0.87 and 1.00. Under direct control, 811 losses over 447,352 held arm-steps give 1.81 per 1,000, and 1.05 outside grasp tests. Among the tasks with more than 2,000 held arm-steps under both executors, the agent’s rate exceeds the policy’s on 13 of 16 tasks with $\pi _ { 0 . 5 }$ and 15 of 18 with InternW0-∆.

## C.3 API-EQUIVALENT COST

Prompt caching dominates the API-equivalent cost of every configuration (Table C.6). On each request, the agent resends the growing episode context, including the instructions, the demonstration, earlier observations, and tool results, so most of the input repeats an earlier prefix from the same episode and is billed at the cached-input rate. At the Standard tariff of GPT-6-Astra (USD 10, 1, 12.5, and 50 per million uncached input, cached input, cache-write, and output tokens), cached input covers 91.5–97.0% of the input tokens and still accounts for 44.0–70.5% of the cost per episode despite its tenfold discount. The recorded tokens cost USD 7.82 per episode under direct control and 2.63 and 3.06 in the harness with $\pi _ { 0 . 5 }$ and with InternW0-∆, corresponding to 16.82, 4.99, and 5.49 per success. For 44 direct-control episodes the final token record is missing, and the direct-control values here, in Figure 9, and in Section 5.3 count the recorded tokens. In all three configurations, token use rises with the number of tool calls, with Pearson correlations of 0.88 under direct control and 0.94 and 0.87 in the harness with $\pi _ { 0 . 5 }$ and with InternW0-∆. No request exceeds the 272k input tokens above which the tariff raises input and cache rates twofold and output rates by half. Only the direct-control run records cache writes, because part of its episodes ran through the API rather than the subscription route, which reports none.

Table C.6: Mean token composition and API-equivalent cost per episode of GPT-6-Astra on the 510 EBench test episodes, with input split into cached input, cache writes, and other uncached input, and the cost per success dividing the total cost by the number of successes.
<table><tr><td>Configuration</td><td>Tokens (M)</td><td>Cached (%) Uncached (k)</td><td></td><td></td><td></td><td></td><td>Writes (k) Output (k) USD/episode USD/success</td></tr><tr><td>Direct control</td><td>3.77</td><td>91.5</td><td>75.6</td><td>244.8</td><td>11.2</td><td>7.82</td><td>16.82</td></tr><tr><td>Harness with  $\pi _ { 0 . 5 }$ </td><td>1.83</td><td>96.5</td><td>64.2</td><td>0</td><td>4.6</td><td>2.63</td><td>4.99</td></tr><tr><td>Harness with InternWO-∆</td><td>2.22</td><td>97.0</td><td>65.7</td><td>0</td><td>4.9</td><td>3.06</td><td>5.49</td></tr></table>

## D ADDITIONAL QUALITATIVE RESULTS

This appendix collects keyframes and prompt excerpts from the recordings of the direct-control evaluation. Appendix D.1 gives prompt-grounded cases, Appendix D.2 compares rollouts of the agent with those of two post-trained policies, and Appendix D.3 shows the episodes with and without the demonstration. Figure D.1 shows two successful episodes on tasks where the agent leads, as discussed in Section 5.2. Figure D.2 shows two unsuccessful episodes that end with partial credit, as discussed in Section 5.3. Figure D.3 shows three unsuccessful episodes of the kind analyzed in Section 5.3: two that stop at the final stage of a high-precision task and one that fails in a two-arm handover.

(a) Bookmark on book  
![](images/7ece2835186d28e054a2cadc17609adaf066547de32636427dca51b5579b9506.jpg)  
Initial state

![](images/a04b5697ee1d8db55c218d1b1b94173282b038717cb7f8cb66bd6296f7a646b3.jpg)  
Grasp approach

![](images/1d028f8136a3b619db08a3dddbb4c88c0212ab4a328184947176fd4d7df08459.jpg)  
Bookmark grasped

![](images/ffc38c3512190e0ea8f2c0e8972a905f791a297c9c3400726e1eb8930ab8ab93.jpg)  
Transfer toward book

Success (Score 1.00)  
![](images/dc46b087ead86189ab7bc62d031d8b9cade6cb406c91fa1d2d80de0ed27076f6.jpg)  
Placement on book

(b) Remote to holder  
![](images/61552464440d8a8423fa5637867d5fe6c736440b5f58a15b25bbe75635c64caf.jpg)  
Initial state

![](images/908e5fb15dcdcd42f20d80c888522513036c1f1b0b8d836ad5dca0c17ecde7e4.jpg)  
Grasp approach

![](images/08be6f6fdd8f54a75e6adc5041589090d277fd056c9c0ab841a4b4c59636afd1.jpg)  
Remote grasped

![](images/816619a5a406e336a26d641e980e78c80a7e0eaf9c02afb7989724057778dbca.jpg)  
Transfer toward holde  
Success (Score 1.00)

![](images/09faaa4009d6a342d4d99cbbc3d02d5f3b36d64836cb6f3d4b29679c69fd48bd.jpg)  
Placement in holder

Figure D.1: Two successful direct-control episodes of the GPT-6-Astra-based agent: Bookmark on book and Remote to holder. Overview frames with video time, with insets enlarging the outlined grasp regions.  
(a) Peg in hole  
![](images/85d8fe42684d82073b67d0d99a434ce3a3c87c6712b31312188283dc0c900325.jpg)

![](images/8129acbf91ae96ee2394bac9697c7fa201acc85e8194636c28dfc074db348443.jpg)

![](images/7fc54d33e20a6d8f540861bec2a08753b74d83bff019ca2e4b0f66dca014236c.jpg)  
Alignment attempt

![](images/922de3d42ae16902a4406b5b2616aeb64225b9f51c48201795013df14deb7afd.jpg)  
Further adjustment

Failure (Score 0.50)  
(b) Bottle  
![](images/2a21b7fe64d82f4fc4d1ad9a362f4156d6b76a2b9c58b8e2d43882cb7504d649.jpg)  
Insertion incomplete

![](images/04644de6276cf541e9f106dd6961fcfa3ede66e14a63efe644f34fe92b3ea0a6.jpg)  
Initial state

![](images/7460335d4490fd295660c4526460ed8ff2c012f56a1115fb57da96edf1b1947f.jpg)  
Grasp approach

![](images/6337b2401b1c257a79fa974bcaada8ad5c6c9d6ca6a340342ef56a0fa397c7a2.jpg)  
Transfer toward rack

![](images/7a0d24d8c74e9a498f1dc4f359aa0c42fb983c0692cb9c86eaaebe815a989233.jpg)  
Bottle placement

![](images/79c294189d062bf29daa8a536f63316b6979933ec9d3e9ef8c145a562c3724d2.jpg)  
Task incomplete

Figure D.2: Two unsuccessful direct-control episodes of the GPT-6-Astra-based agent, each ending with score 0.50: Peg in hole (right-wrist view) and Bottle (overview). Video time is shown on each frame.

Figure D.4 shows three episodes with the safety-relevant behavior discussed in Section 5.5, and Figure D.5 shows the two ways in which the agent ends an unsuccessful episode itself, as discussed in Section 5.3.

(a) Peg in hole  
![](images/17784e8fd522f1e038b91085441765229d835c9be0e98c755eea5543ddc020f1.jpg)  
“Correct lateral o set…”

![](images/90da595421f7e7b699e8282dcd5b143d01a37c17d82ab8e71a0d6f1ecb3894a6.jpg)  
Tip beside the hole

![](images/2cb29587650824f8e60534e6365be8b1eaf0f3c3c4f17e141db31cdfdd81e82e.jpg)  
“…to seat the peg.”

Failure (Score 0.50)  
![](images/09102241fc7b84a14701c221e62b88dcb620b429849c3d12c4925f673212f868.jpg)  
“The peg rests across…”

(b) Tighten nut  
![](images/216f74d760e187793042bd6e0eb5afc2d580173c83cf82605799c6e4c70e721b.jpg)  
“…above the bolt axis”

![](images/ad6010e9f938d0224fa6f1633703dd54173e8c397b0bd9c18b7c128dd2d3818e.jpg)

![](images/7aeaa52fbb9e2a3a4847c166ebe0a1938cfc624a05aff54aab4b360f7cf86f46.jpg)  
Nut lies beside the bolt  
“Reacquire the nut…”

Failure (Score 0.50)

![](images/fa476dd12a597056bc33d9bf0281ffc62db95024b6b7ad741bc7a8dccecbd571.jpg)  
Final frame (step limit)

(c) Flip cup & collect cookies  
![](images/67ecca617e959cf4c50a48cdb16275adba833beb833ce3f98dc349446cd00152.jpg)  
“Align right ngers…”

![](images/22a178a38999c77097cbeace7357e5ce5629c0ff8e3f0d83da025277719df441.jpg)  
“Withdraw support…”

![](images/d125daec29a95b54512622e62ea5a4ba21a782437b26acfa29c2b81c757f1d0e.jpg)  
Cup falls

Failure (Score 0.10)  
![](images/17e5e45161bd5f272e0711b0b68678dfb358eda97573495dd5dd3e43df57ecf6.jpg)  
“…locate slipped cup”

Figure D.3: Three unsuccessful direct-control episodes of the GPT-6-Astra-based agent, with the agent’s stated reasons below the frames. Orange boxes are enlarged in the insets. (a) Peg in hole: the peg tip lands beside the hole, and the peg ends lying across the receiver at the step limit (score 0.50). (b) Tighten nut: the nut falls beside the bolt, is re-acquired, and rests unthreaded on the bolt tip at the step limit (score 0.50). (c) Flip cup & collect cookies: during the second handover the supporting hand withdraws, and the cup falls off the table (score 0.10).  
(a) Dishwasher  
![](images/4adf12a2924660f45ac510326d3844059363fb604b0585555223e09b7111c3c3.jpg)  
Bowl held, door opening

![](images/fd105ca6230e1d04690d78ff8ec12b7b1e9d92feacdaafa8a456c5521934e6f9.jpg)

![](images/9fb99ffca264aa31a1efe59d0120a5949466143879426de67bf8a13093bb3c2b.jpg)  
(b) Apple to fruit bowl

Failure (Score 0.67)  
![](images/54f6f0d378193ab95bd57ba4b13c5716f5b742fa60cbf98ca78734462e2ad278.jpg)  
“Scan the open oor…”

![](images/ac33354f074d12f60ab5e43b57a0054e0e40d6495bd155db23b8df442456d66e.jpg)  
“The apple slipped…”

![](images/e69bc4f346f6d36b3216b4c6470ef007d358d320b181cfdf6434f5485568116f.jpg)

![](images/2ff89f0ba0379d082ad2fba59811e4d17f62ee435f35a8c32a5846f967fe91e1.jpg)  
“… oor behind the robot…”

Failure (Score 0.00)  
![](images/a3ef99dbfe71b52ce304604da1f264628e47ef3fe92117baa071496e46628948.jpg)  
“Inspect deeper beneath…”

(c) Collect co ee beans  
![](images/1d4287ae00eac5f70f55f793ce8eb6e574bbd417d783cb5b6d751fd13d12b589.jpg)  
“…jar from table edge…”

![](images/cdae5e4cdfe7b9a0a5da3d2e5ec6127cdf6493bd3f060dfc70ff33483a62b61c.jpg)  
“Rotate held jar upright…”

![](images/f41a4fbaec9833e30aa9e8438fef95d5013f58d06a8c3028e40943b4b37a5cf0.jpg)  
“Sweep beans toward…”

Failure (Score 0.07)  
![](images/305e9533373e27bc8f57031ae3ba8af78a213133d1bca12c688341966bc393d5.jpg)  
“…after edge contact”

Figure D.4: Three direct-control episodes of the GPT-6-Astra-based agent with safety-relevant behavior, with the agent’s stated reasons below the frames. Orange boxes are enlarged in the insets. (a) Dishwasher: the bowl slips from the left hand while the right hand opens the door, and the agent later searches the floor (score 0.67). (b) Apple to fruit bowl: after the apple falls, the agent searches below the table and behind the robot with targets outside the workspace (score 0.00). (c) Collect coffee beans: the held jar and later the spoon contact the table edge (score 0.07).

![](images/898f06f5ef3ed83c156aeb497fefd48ca45f01052fa244518dd8659e19a89d2c.jpg)  
Figure D.5: Two unsuccessful direct-control episodes that the GPT-6-Astra-based agent ended itself, with the video time and camera of each frame. (a) Dishwasher: the final message reports both bowls in the rack while the door is open. (b) Collect coffee beans: the held jar is lost at the table edge.

## D.1 PROMPT-GROUNDED CASES

The stored prompts and stated reasons of unsuccessful episodes show how supplied guidance and its execution diverge. The first box below belongs to panel b of Figure 4, and the second shows a Fruit episode in which the agent follows the live instruction over the demonstration. Both list the guidance and the agent’s response.

![](images/80eb368cf600333216b3bdb3b11160849cae2355e74e693e19cc597ea2aceae6.jpg)  
D.2 ROLLOUTS OF THE AGENT AND POST-TRAINED POLICIES

Recovery from subgoal disruptions. Figure D.6 compares rollouts of the GPT-6-Astra-based agent, $\pi _ { 0 . 5 } ,$ , and OpenWAM-α on the two tasks summarized in Section 5.6. In Teacup, saucer & teapot (Figure D.6a), the agent first approaches the cup handle and refines its wrist and gripper configuration over successive attempts before grasping and placing the cup onto the saucer. When subsequent manipulation of the teapot inadvertently displaces the cup from the saucer, the agent detects that the completed subgoal has been invalidated, re-grasps the cup, and restores it to the saucer rather than continuing the remaining action sequence. By contrast, $\pi _ { 0 . 5 }$ repeatedly approaches and retracts without completing the cup transfer, whereas OpenWAM-α fails to grasp the cup and proceeds to place the teapot onto the tray while leaving the cup off the saucer.

Precise contact in object manipulation. In Glass into glassbox (Figure D.6b), although the agent uses both arms to place the glasses into the open case, its subsequent adjustments fail to fold the temples fully inward, leaving a protruding temple that prevents the lid from closing despite repeated

## (a) Teacup, saucer & teapot

![](images/1252b9d1c187b0ce2846ef1abec13ca496e2535b5bbde20b3582b2e59894f1af.jpg)  
Figure D.6: The GPT-6-Astra-based agent (direct control), π<sub>0.5</sub>, and OpenWAM-α on two tasks: (a) Teacup, saucer & teapot and (b) Glass into glassbox. Keyframes are in chronological order with video time. The outcome and score of each rollout are given at the right, and insets show enlarged views of the outlined manipulation regions. The rollouts differ in object variants and viewpoints.

corrective pushes. By comparison, π<sub>0.5</sub> places the glasses into the case and folds both temples compactly (though leaving the lid open), while OpenWAM-α completes both temple folding and lid closure.

## D.3 IN-CONTEXT LEARNING

Figure D.7 shows keyframes from the paired evaluation with and without the annotated demonstration across all 20 test episodes of Frame placement and Gear installation (Section 5.6). On Frame placement (1/20 successes zero-shot vs. 13/20 with the demonstration, mean score 0.283 vs. 0.800), under zero-shot control the agent reaches and displaces the frame, but repeatedly adjusts its approach trajectory and wrist orientation without achieving stable placement. When conditioned

Failure (Score 0.67)

![](images/855a6f9dd59085945a15de5a01f3dc03b94c3d15cf89c46f0509ce445d963ddd.jpg)  
Figure D.7: The GPT-6-Astra-based agent (direct control) without (zero-shot) and with the demonstration on the same test episodes: (a) Frame placement and (b) Gear installation. Keyframes are in chronological order with video time, and columns do not correspond to matched elapsed time.

on the demonstration, the agent coordinates both grippers to hold and transfer the frame in 17 of 20 episodes (including all 13 successful episodes, compared with 1 of 20 under zero-shot control) and positions the frame upright against the pen holder (Figure D.7a). On Gear installation (3/20 successes under both conditions), without the demonstration the agent grasps and lifts the gear but leaves it outside the assembly region, whereas with the demonstration it brings the gear between the two mounted gears and lowers it, but it scores no stage in 15 of the 20 episodes (Figure D.7b).

## E CASE STUDIES IN THE HARNESS WITH POLICY

The direct-control episodes of Appendix D illustrate three failure categories: an unconfirmed completion claim, an object lost beyond the reachable workspace, and a long-horizon task that ends with partial credit. Figure E.1 presents three episodes in the harness with policy on scenes that the agent solved under direct control, covering an unconfirmed completion claim, an unrecovered object loss, and a completed stage undone by a subsequent policy segment. The call-by-call records report the executor, the physics steps, the finger opening as defined in Appendix B.2, and the agent’s stated reason.

Figure E.2 compares, on one scene each of Glass into glassbox and Teacup, saucer & teapot, the agent under direct control with the harness with $\pi _ { 0 . 5 }$ (Sections 6.3 and 6.4).

![](images/4bf39ee5d0e0b896f3ddc754b046029ba0b96ec618505871a5cf40a83241eb18.jpg)  
Figure E.1: Three unsuccessful episodes in the harness with policy on scenes that the agent solved under direct control: (a) Gear installation with $\pi _ { 0 . 5 } .$ , (b) Detergent with InternW0-∆, and (c) Teacup, saucer & teapot with $\pi _ { 0 . 5 } .$ . Each frame shows the video time, the camera when it is a wrist camera, and the executor, with EEF marking the agent’s own commands.

## E.1 DIVISION OF LABOR

The stated reasons below come from the successful Flip cup & collect cookies episode discussed in Section 6.5.

Division of labor: Flip cup & collect cookies, harness with InternW0-∆, score 1.00   
Agent, call 3 (EEF). “Test cup grasp by raising 1 cm without rotation.”   
Agent, call 4 (InternW0-∆). “Cup rose with the left gripper; continue controlled reorientation and   
handover.”   
Agent, call 12 (InternW0-∆). “Plate rises with the right hand; transport toward receiving bowl and begin   
controlled tilt.”   
Outcome. InternW0-∆ performs the handover, the reorientation of the cup, and the pour, and the episode   
succeeds after 14 calls and 2,301 physics steps. On the same scene, the agent under direct control reaches   
a score of 0.60 after 100 calls.

## E.2 VERIFYING PHYSICAL COMPLETION

Like the direct-control Dishwasher episode of Figure D.5a, the Gear installation episode of Figure E.1a with $\pi _ { 0 . 5 }$ ends with an unconfirmed completion claim. The agent closes the left hand on the gear hub and lifts the gear by 1 cm (calls 3 and 4, finger opening 29.9 and 30.0 mm), then invokes π<sub>0.5</sub> for transport and seating in three segments of 180, 180, and 120 physics steps. In the second segment (call 6, stated reason “Continue fine alignment and seating of the blue gear on the middle axis”), $\pi _ { 0 . 5 }$ opens the left fingers from 29.9 to 86.2 mm beside the large gear. The agent then withdraws the open left hand and closes it (calls 8 and 9), requesting only the overview image for both commands, in which the board occupies a small part of the frame. It ends the episode at 828 physics steps with the final message “The blue gear is in the middle position between the white gears and remained there after the hand withdrew”, and the episode scores 0.5. On the same scene, the agent flunder direct control releases the gear, inspects the assembly from a more vertical wrist view (call <sup>fl</sup>12), re-approaches the hub to correct a forward offset from the spindle (call 13), and withdraws only after the spindle becomes visible through the gear bore (call 16), solving the scene in 821 physics steps.

Failure (Score 0.33)

(a) Glass into glassbox: direct control  
![](images/6ec1139ed9e2d0a41abfc116612dc0ed67a9b5fa67016e5eeaca51752ca3619e.jpg)  
Figure E.2: Keyframes of the GPT-6-Astra-based agent under direct control and in the harness with $\pi _ { 0 . 5 }$ as a tool, with the video time, the executor of each frame, and the outcome of each episode. (a, b) Glass into glassbox on one test scene, under direct control and with π<sub>0.5</sub>. (c, d) Teacup, saucer & teapot on one test scene, under direct control and with $\pi _ { 0 . 5 }$

## E.3 UNRECOVERED OBJECT LOSSES

In the recorded episodes, an object that leaves the reachable workspace is not recovered for the rest of the episode, as when the apple of Figure D.4b falls and the agent under direct control searches outside the workspace. In the Detergent episode of Figure E.1b, the agent drops the first bottle to the floor when it resumes control during a bimanual handover. InternW0-∆ closes the right hand on the first bottle while the left hand still holds it (call 4, 250 physics steps), ending with a fully closed finger command and a measured finger opening of 32.9 mm. The agent then opens the left hand (call 5) without commanding the right hand to stay closed, so the right fingers remain at their measured opening of 32.9 mm (Section 6.4), and the bottle tilts in the right hand while the left hand opens. In the 1 cm lift test that follows (call 6, stated reason “Check whether the receiving hand retained the bottle using a small lift”), the right fingers close from 32.8 to 2.2 mm within the first 8 physics steps, and the bottle falls from the counter to the floor. InternW0-∆ places the two other bottles in the basket by physics step 2,348, but the next 20 calls (calls 19 to 38) and 2,020 physics steps fail to recover the bottle on the floor: seven EEF waypoints directed toward the floor end 11 to 20 cm from their targets, and three InternW0-∆ segments of 300 physics steps each end with the fingers closed at 0.0 to 0.1 mm. The agent ends the episode at 4,368 physics steps with the final message “The blue/red bottle fell to the floor, and recovery attempts failed”, and the episode scores 0.67. Under direct control, the agent solves the same scene in 1,681 physics steps, carrying each bottle in the left hand without a transfer between the hands.

## E.4 PRESERVING COMPLETED STAGES

Figure D.2b shows a Bottle episode under direct control that ends with partial credit. In the harness with policy, a policy segment can also undo a stage that is already complete, because the post-trained policy receives the fixed task instruction without a record of the completed stages, as analyzed in Section 6.7. In the Teacup, saucer & teapot episode of Figure E.1c, five segments of $\pi _ { 0 . 5 }$ at the cup (calls 2, 4, and 5) and at the teapot (calls 22 and 24) end with the right hand empty and take 540 of the first 1,602 physics steps. Between those segments, the agent completes the cup placement itself, grasping the cup, releasing it on the saucer at physics step 994 (call 20), and verifying the placement (call 21). Its own handle grasp of the teapot holds at a 9.6 mm finger opening in the lift test (call 31), but the fingers close to 0.0 mm as the teapot is lifted clear (call 32). The agent invokes $\pi _ { 0 . 5 }$ for recovery with the stated reason “reapproach shifted handle with learned alignment” (call 34, 180 physics steps). In this segment, the hand descends beside the teapot, the saucer tips up under it, and the cup falls and comes to rest upside down on the table, undoing the earlier placement. The agent then tests the new contact as a teapot grasp (call 36) and releases it (call 37), and the episode runs until the step limit of 2,000 physics steps inside a final segment of $\pi _ { 0 . 5 }$ with the stated reason “Recover displaced cup and saucer” (call 38). The episode scores 0. Under direct control, the agent solves the same scene in 1,631 physics steps.

## E.5 RECOVERY AFTER DELEGATED HANDOVERS

Figure E.3 shows a Shop episode in the harness with InternW0-∆ that fails although InternW0-∆ alone completes the same scene in 4,166 physics steps. The failure begins when the small carton drops in a handover between the hands that the agent delegates to the policy. The agent first lets the policy retry and then takes over, placing the carton on the destination belt with its own waypoints. After a can also slips in a later delegated handover, the agent again takes over, this time to secure the long box. Both recoveries succeed, but together with the failed handovers they use up the budget, so the episode reaches the 5,000-step limit while the agent carries a can across the scanner.

Shop: harness with InternW0-Δ  
![](images/2183644c57b3ab34a89e17005276c2d3bcd1c0e3249577c58b26186276f7ec18.jpg)  
Bottle carried across the scanner

![](images/ee08902f0f29a7ed19def6e349691948c2bf24d1e8e790fcac08ceaf7084d8d4.jpg)  
“Small carton remains on source belt after failed handover”

![](images/6b78a4b0e76032299c1607ceefcc416d89d6b65bb4069ade8c3f0b1c1482aeb9.jpg)  
“Align over the fallen carton using the wrist view…”

Failure (Score 0.67)  
![](images/5f39b2175b7e55cd07f55479f19dec90c832e86637a83d9011f1e8732506e7bb.jpg)  
“Place the scanned carton on the destination belt”

![](images/aa377d5de5721b8cd8e2132a641e77e4a3bbc810217525583a6daf2c585a0f04.jpg)  
Figure E.3: A Shop episode in the harness with InternW0-∆, with the video time, the executor of each frame, and the agent’s stated reasons in quotation marks. Orange boxes are enlarged in the insets.

![](images/6d34439c7222ac7f45e3cc787af67824d3674372419ceb5e9bebabed3e504ae6.jpg)

![](images/e659c2f0b2934ef9301c8f0bf8767c35c39bbc3a8e676be7c6e978ab8bc1295b.jpg)  
Long box carried to the destination belt

![](images/7c0f1cf313d163035a12d41cc5687ccc32c4e9a27af1809b1631db6e48d495e8.jpg)  
Step limit reached while carrying a can across the scanner