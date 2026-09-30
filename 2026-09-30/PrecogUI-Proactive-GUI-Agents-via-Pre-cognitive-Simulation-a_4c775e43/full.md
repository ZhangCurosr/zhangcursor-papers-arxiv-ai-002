# PrecogUI: Proactive GUI Agents via Pre-cognitive Simulation and Experience Retrieval

Bin Kang<sup>1,3,5</sup>, Jiarui Ouyang<sup>2</sup>, Li Jiang<sup>4</sup>,

Bin Chen<sup>3</sup>, Zhuotao Tian\*<sup>3,5</sup>

<sup>1</sup>University of Chinese Academy of Sciences

<sup>2</sup>The Hong Kong University of Science and Technology

<sup>3</sup>Harbin Institute of Technology

<sup>4</sup>The Chinese University of Hong Kong

<sup>5</sup>Shenzhen Loop Area Institute

## Abstract

Existing reactive Graphical User Interface (GUI) agents often fail in long-horizon, dynamic scenarios, where unexpected disturbances trigger attention-diverting and cascading failures. To address this, we propose PrecogUI, a pre-cognitive architecture that shifts the paradigm from reactive execution to proactive decision-making. Specifically, we design a Proactive Experience Pool (PEP), which caches recurring anomaly and success patterns as ”state-action-result” tuples in a dual-memory repository. Furthermore, we introduce a Proactive Simulation Executor (PSE) that learns to forecast the next symbolic UI layout given a candidate action, enabling early anomaly avoidance and ranking candidate actions by predicted reliability. Finally, a Pre-cognitive Execution Controller (PEC) fuses these priors and predictions, prioritizes handling of foreseen anomalies, and ensures execution robustness through a closed-loop error correction mechanism. For robust evaluation, we develop AutoTraj, an automatic data-generation engine, to construct InterfereBench, a benchmark for long-horizon tasks with strong disturbances. Experiments demonstrate that PrecogUI surpasses state-ofthe-art methods on InterfereBench while maintaining competitive performance on public benchmarks. The code will be publicly available.

Keywords: GUI Agent ; Long-horizon ; Proactive

## 1 Introduction

Graphical User Interface (GUI) agents (Cheng et al. 2024; Lin et al. 2025; Gou et al. 2025; Hong et al. 2024) are built on Multimodal Large Language Models (MLLMs) to comprehend user queries, interpret context, and perform actions like clicks and swipes for accomplishing GUI tasks. The advancement of MLLMs (Li et al. 2023; Alayrac et al. 2022; Dai et al. 2023) has notably enhanced agents’ interface perception and decision-making precision. Nevertheless, anomalies such as pop-ups and black screens in dynamic settings remain a significant hurdle, diverting attention and causing persistent cascading failures.

Prior research (Hong et al. 2024; Huang et al. 2025; Chen et al. 2025) has significantly advanced the perception-action loop. However, the prevailing approach remains reactive, relying on current observations for decision-making. While effective in short-horizon, disturbance-free settings (Rawles et al. 2025; Deng et al. 2023), these reactive methods may struggle in long-horizon tasks and dynamic environments. Recent efforts have attempted to address this challenge through online exploration (Sun et al. 2025; Fan et al. 2025) and app-specific memory or layout-aware retrieval (Wen et al. 2024; Kong et al. 2025). Nevertheless, the reactive nature still leaves agents vulnerable to distractions from nongoal cues such as pop-ups and delays.

Key Observations. To investigate robustness, we evaluate representative reactive agents (Liu et al. 2025; Qin et al. 2025; Zhang et al. 2025b) on AndroidControl (Li et al. 2024) under injected disturbances at both the overlay level (e.g., pop-ups, notifications) and environment level (e.g., black screens, freezing). Performance is assessed by success rate (SR), stratified by disturbance type and task horizon. Specifically, the bars in Figure 1(b) report absolute SR reductions by disturbance type. Overlay-level disturbances induce the most significant degradation, reducing SR by more than 20 percentage points on average, compared with an approximately 10-point drop under environment-level perturbations. Besides, the performance degradation scales monotonically with horizon length, as shown in Figure 1(c). On short-horizon tasks (< 5 steps), all models maintain high robustness (SR ≥ 91%). However, for medium-length tasks (6–15 steps), reactive agents exhibit increasingly pronounced SR degradation. In long-horizon tasks (>15 steps), reactive agents’ SR declines to approximately 50%. See Appendix A for further analysis.

These results show that reactive agents are easily distracted by non-goal stimuli, allowing errors to accumulate into cascading failures. This observation prompts a crucial question: how can we empower agents with pre-cognitive planning and explicit exception handling to ensure robustness in long-horizon, dynamic environments?

Our Solution. In this study, we propose PrecogUI, a framework that integrates experience retrieval with online look-ahead simulation to improve the robustness of GUI agents in long-horizon and disturbance-prone settings. The conceptual architecture is illustrated in Figure 1(a).

Specifically, PrecogUI introduces the Proactive Experience Pool (PEP), a dual-memory repository that stores and retrieves recurring interaction patterns from both successful and anomalous executions, enabling knowledge reuse via pattern matching. Then, the Proactive Simulation Executor (PSE) employs a conditional diffusion model (Rombach et al. 2022) to simulate the symbolic UI layout resulting from candidate actions, providing look-ahead forecasts for early anomaly detection and candidate-action reliability ranking. Finally, these are integrated by the Pre-cognitive Execution Controller (PEC), which prioritizes anomaly handling, selects high-utility actions, and ensures robustness through state monitoring and hierarchical rollback/retry.

![](images/9247b3e0b6f6650355152ff3677f5c0a9cc951a1649dee0c9fe8e97dfbf0f520.jpg)  
Figure 1: (a) PrecogUI combines experience retrieval with look-ahead simulation; AutoTraj generates perturbed trajectories. (b) Absolute SR drop by disturbance type on AndroidControl. (c) Disturbance impact grows with task horizon.

To the best of our knowledge, no existing benchmark systematically evaluates long-horizon robustness under diverse, sustained perturbations. We thus introduce InterfereBench, a new benchmark consisting of 1,160 task-level trajectory groups (∼27k annotated source screenshots) across 34 diverse applications. Each group provides a clean execution and two controlled perturbation replays, enabling paired evaluation under prolonged task horizons and dynamic interference. AutoTraj is the automated engine used to generate these perturbation-rich interaction trajectories at scale. Experiments on InterfereBench and public benchmarks such as AndroidControl (Li et al. 2024) and GUI-Odyssey (Lu et al. 2024) show that PrecogUI outperforms the strongest baseline by 22.4 percentage points in success rate under strong perturbations, improving robustness without sacrificing overall performance. To summarize, our contributions are as follows:

• We propose PrecogUI, a unified framework that combines offline experience reuse, proactive layout prediction, and exception-aware execution recovery to enhance robustness in long-horizon GUI interactions.

• We present InterfereBench, a new benchmark designed to evaluate robustness under strong, sustained perturbations in long-horizon tasks, along with AutoTraj, an automated pipeline for scalable, realistic trajectory generation.

• Extensive experiments on InterfereBench and the public benchmarks demonstrate that PrecogUI effectively improves long-horizon reliability and anomaly resilience while maintaining general GUI capabilities.

## 2 Method

## 2.1 Overview

Toward robust long-horizon execution under perturbations, we propose PrecogUI, which closes the loop between experience, foresight, and feedback via four modules: (i) AutoTraj builds InterfereBench, a long-horizon benchmark with controlled perturbations; (ii) PEP forms a memory of anomaly/success patterns by indexing experiences based on their UI layout structure, enabling efficient retrieval of similar past cases; (iii) PSE predicts the next symbolic UI layout, estimates anomaly risk, and ranks candidate actions across index, relative, and absolute-level variants; (iv) PEC fuses PEP and PSE with online monitoring and rollback/retry to deliver robust, closed-loop control. We discuss related work in Sec. 4, with an extended review in Appendix B.

## 2.2 Data Construction

The capabilities of GUI agents are fundamentally constrained by data scale, diversity, and quality. To address this, we present AutoTraj, an automated pipeline that generates high-quality GUI interaction trajectories with explicit disturbance awareness. AutoTraj comprises three core components as follows:

Autonomous Explorer. The Explorer efficiently discovers diverse, high-value interaction trajectories using a hybrid perception strategy: it prefers the UI view hierarchy to find actionable elements; when structured signals are missing or incomplete, it falls back to a vision pipeline that combines object detection and optical character recognition (OCR), producing a unified candidate set of controls.

![](images/a2aae753b13be14324092dc12794025467fb15bdbdbc8d2925c1b7c95e9e2356.jpg)  
Figure 2: Data Construction. Stage 1 discovers clickable elements via view hierarchy and vision, executes basic actions, and logs replayable UI trajectories. Stage 2 prunes redundant steps, ranks trajectories with an MLLM, and outputs structured annotations. Stage 3 injects realistic disturbances to create clean–perturbed pairs for robustness evaluation.

Exploration is driven by a pre-trained agent (Ye et al. 2025) that tries atomic actions (click, scroll) and logs preand post-screenshots, as well as action metadata, to produce replayable trajectories. To guide informative exploration, we define the exploration value at state $s _ { t }$ as:

$$
V ( s _ { t } ) = \alpha \cdot \frac { \left| E _ { t } \setminus ( \bigcup _ { i < t } E _ { i } ) \right| } { | E _ { t } | + \varepsilon } + ( 1 - \alpha ) \cdot \frac { 1 } { \sqrt { n ( s _ { t } ) + 1 } } ,\tag{1}
$$

where $E _ { t }$ denotes the control set at $s _ { t } , \bigcup _ { i < t } E _ { i }$ is the union of controls seen so far, and $n ( s _ { t } )$ counts visits to $s _ { t }$ . The first term promotes the discovery of unseen controls/layouts, while the second enforces novelty to favor coverage and rarely visited states. $\alpha \in [ 0 , 1 ]$ balances layout discovery and rare-state exploration; hyperparameter analysis appears in Appendix D.5.

Trajectory Parser. To ensure semantic and structural quality, the raw trajectories undergo two-stage filtering and parsing. Stage-1 removes excessively long or redundant trajectories using self-loop and no-op statistics derived from layout changes; their definitions, pruning criteria, and threshold analyses are provided in Appendix D.6.

In Stage-2, we filter trajectories using a high-capacity MLLM (Comanici et al. 2025) that evaluates topic consistency, causal soundness, and task complexity, selecting the top-K trajectories for detailed labeling. For each trajectory, the parser yields a high-level goal and stepwise descriptions, exporting structured JSON with goals, step descriptions, action types (normalized coordinates), UI boxes, screen deltas, and execution outcomes for training and evaluation.

Perturbation Injector. To study robustness, we develop a Perturbation Injector that creates paired samples for evaluation. For each clean trajectory, we randomly inject realworld perturbations covering: (1) overlay interference (simulating system notifications, pop-up dialogues, etc.); (2) environmental perturbations (black or repeated frames to simulate loading/lag, and spontaneous layout changes). All perturbations are screened by six experts to ensure correctness. This yields paired samples for each trajectory: a clean baseline and perturbed variants, enabling comparative evaluation in both “normal” and “perturbed” modes.

Following this pipeline, AutoTraj produces InterfereBench with 1,160 task-level trajectory groups across 34 applications. Each group is anchored by a clean source trajectory of 14–37 steps and includes two controlled perturbation replays of different types. The ∼27k figure counts annotated screenshots in the clean source trajectories; replayed perturbation frames are generated during evaluation and are not counted again. We use task group for the shared instruction and source trajectory, and execution variant for its clean or perturbed replay.

To faithfully capture and replay complex interactions beyond single-tap actions, we additionally employ PolyTouch (Appendix C), a multi-gesture and macro execution layer that synthesizes deterministic multi-pointer gestures (e.g., three-finger chords, pinch/zoom/rotation) and declarative macros with explicit timing, guards, retries, and rollback.

## 2.3 Proactive Experience Pool

We observe that failure-inducing anomaly patterns (e.g., permission pop-ups, network delays) and success-inducing patterns (e.g., app navigation) repeat widely across tasks and applications. Therefore, we propose the Proactive Experience Pool (PEP), which converts costly trial-and-error into efficient experience retrieval. By caching and indexing critical state–action–outcome patterns, the executor can leverage priors rather than plan in isolation. PEP maintains two parallel memories:

(1) Anomaly Memory $( M _ { a } ) \colon M _ { a }$ records two classes of failures: (i) state–action mappings $( s , a ) \mapsto \ell _ { \mathrm { a n o m } }$ when an action in a state yields a specific anomaly; $( \mathrm { i i } ) s \mapsto \ell _ { \mathrm { a n o m } }$ for states that inherently denote failure (e.g., network outage). Each anomaly entry additionally stores a human-designed or previously successful handling action $a _ { \mathrm { h a n d l e } }$ , which can be reused as a remedy.

(2) Success Memory $( M _ { s } ) \colon M _ { s }$ stores high-confidence successful transitions $( s , a ) \mapsto s ^ { \prime }$ , indicating that action a in state s reliably reaches a successful successor state $s ^ { \prime }$

State Representation and Retrieval Mechanism. To enable robust and efficient retrieval, we represent each UI state s by its layout signature: a structured set of interactive elements, with each abstracted as (type, bbox). Instead of relying on exact matches, retrieval is performed by finding the nearest neighbors in the experience pool. Specifically, for a query state $s _ { t } .$ , we compute the similarity between its layout $\dot { L } _ { t }$ and each memory layout $L _ { m }$ via a greedy matching algorithm based on element IoU:

![](images/0a1fc23412fe5dded2f181544f1a3f31c2b74b256241d558a70d029f4f8f79d5.jpg)  
Figure 3: Illustration of PrecogUI. (a) PEP builds a dual memory of anomaly/success patterns; (b) PSE forecasts the next symbolic UI layout and estimates anomaly risk for candidate actions; (c) PEC fuses priors and predictions to prioritize exception handling, select the highest-utility action, and enforce closed-loop monitoring with rollback/retry.

$$
S _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { m } ) = \frac { 2 | \mathcal { M } | } { | L _ { t } | + | L _ { m } | } ,\tag{2}
$$

where M is the set of matched pairs identified by the greedy algorithm, which pairs elements from $L _ { t }$ and $\dot { L } _ { m }$ with the highest Intersection over Union (IoU) score above a predefined threshold. The top-k most similar historical cases are then retrieved to inform the agent. Similarly, actions a are canonicalized based on their type and the target element’s normalized coordinates. Crucially, PEP is a dynamic, online-updated memory. Upon encountering new anomalies or discovering successful cases, the agent extracts the stateaction-result tuple and asynchronously appends it to the memory pool. We further address cold-start (pool initially empty) and entry staleness (app updates invalidate cached layouts) via a version-aware update policy with exponential staleness decay; empirical analysis is provided in Sec. D.3.

## 2.4 Proactive Simulation Executor

Mainstream GUI agents are fundamentally reactive and lack foresight into post-action effects. With abrupt transitions in dynamic UIs, reactive policies without anticipation of anomalies tend to fall into irrecoverable failures. Accordingly, we propose the Proactive Simulation Executor (PSE), which forecasts the next symbolic UI layout before acting and evaluates anomaly risk and candidate-action reliability, shifting the paradigm from “observe–act” to “observe– predict–act.”

## Conditional Future Layout Generation.

Following ViMo’s action-conditioned GUI worldmodeling paradigm (Luo et al. 2025), we model one-step UI transitions with a conditional latent diffusion model (Rombach et al. 2022), fine-tuned on InterfereBench and public GUI datasets (Li et al. 2024; Lu et al. 2024). Let $\begin{array} { r c l } { c _ { t , j } } & { = } & { { \mathrm { E n c o d e } } ( a _ { t } , r _ { j } ) } \end{array}$ denote the concrete command obtained by instantiating candidate action $a _ { t }$ with format $r _ { j }$ PSE predicts $P ( \hat { L } _ { t + 1 } ^ { c _ { t , j } } \mid L _ { t } , c _ { t , j } )$ . Unlike ViMo’s full-screen prediction, PSE generates only an ordered set of (type, bbox) elements for efficient geometric risk tests rather than photorealistic GUI simulation. Architecture, training, and sampling details are provided in Appendix D.4.

Reliability Forecasting. Subsequently, we apply a set of efficient rules to the predicted layout $\hat { L } _ { t + 1 }$ for anomaly recognition. For example, if a bounding box significantly occludes multiple interactive controls in $L _ { t } ,$ it is flagged as a pop-up anomaly; if interactive elements are nearly absent, it is flagged as a blank-screen anomaly; if $\hat { L } _ { t + 1 }$ remains largely unchanged from $L _ { t } ,$ the action is likely ineffective or causes freezing.

Motivated by the navigation-style tasks (Gou et al. 2025; Liu et al. 2025; Xu et al. 2025a) in our benchmarks, where successful interactions typically induce non-trivial layout changes while failed or frozen steps exhibit nearzero change, we define layout dissimilarity from Eq. (2) as ${ \mathcal { D } } _ { \mathrm { l a y o u t } } ( L , \mathbf { \bar { \Phi } } ^ { \prime } ) = 1 - S _ { \mathrm { l a y o u t } } ( L , L ^ { \prime } )$ and use it as a heuristic progress signal. We modulate this signal with the anomaly severity predicted on $\hat { L } _ { t + 1 } ^ { c _ { t , j } }$ . Specifically, let $w ( \hat { L } _ { t + 1 } ^ { c _ { t , j } } ) \in$ [0, 1] denote the anomaly-aware reliability weight, which is set to 1 for non-severe layouts and 0 for severe anomalies. For each candidate action $a _ { t }$ and command format $r _ { j }$

$$
s ( a _ { t } , r _ { j } ) = \mathcal { D } _ { \mathrm { l a y o u t } } ( L _ { t } , \hat { L } _ { t + 1 } ^ { c _ { t , j } } ) \cdot w ( \hat { L } _ { t + 1 } ^ { c _ { t , j } } ) .\tag{3}
$$

The rule-based estimator converts each PSE forecast into an anomaly label and a relative reliability score for ranking candidate action-format pairs; these heuristic scores are not interpreted as calibrated success probabilities. We independently evaluate PSE’s prediction quality on held-out test splits, achieving 89.4% element-type accuracy, 80.2% mean bbox IoU, and 83.2% element-level F1 (details in $\mathsf { A p - }$ pendix D.7). Note that Eq. (3) can misrank edge cases (e.g., confirm dialogs with minimal layout change); the anomaly filter mitigates risk before execution, while PEC verifies residual cases after execution (Sec. 2.5).

## 2.5 Pre-cognitive Execution Controller

While PSE offers look-ahead predictions, robustness remains uncertain in the absence of a decision framework that converts them into concrete actions. We designa closed-loop controller that fuses PEP priors, PSE predictions, and execution feedback, converting open-ended trial-and-error into guided, self-correcting policy control.

Deep Think & Decision. The cycle begins by generating a set of semantically grounded candidate actions A. We steer a base MLLM’s reasoning by prompting it to populate a structured JSON schema. This schema mandates a chain of thought that includes: (i) Historical Validation, verifying the outcome of the previous step; (ii) Content Grounding, ensuring that critical UI elements for the current instruction are present; (iii) Think, a step for rationale articulation and failure attribution analysis; and finally (iv) Action, which outputs a ranked set of candidate actions, each with index, relative, and absolute coordinate formats. Expanding each action over its applicable formats yields the candidate-pair set $\mathcal { C } _ { t } = \{ ( a , r ) : \overset { - } { a } \in \mathcal { A } , r \in \mathcal { R } ( a ) \}$ ; the schema and action formats are detailed in Appendix E.

Pre-cognitive Execution. Prior to execution, PEC performs an anomaly check on the current state $s _ { t }$ . It first queries the anomaly memory $M _ { a }$ with the current layout. If a sufficiently similar past case is retrieved, the controller triggers the associated remedy $a _ { \mathrm { h a n d l e } } ;$ otherwise, deterministic current-layout rules $\mathcal { R } _ { \mathrm { a n o m } }$ check for visible occlusion, blank-screen, and known anomalous-layout patterns. If these rules flag an anomaly, PEC uses a foundation model with a structured anomaly prompt to synthesize a handling action (Appendix E.3). In Algorithm 1, this observed-state check is denoted by CurrentState $\left( s _ { t } ; \mathcal { R } _ { \mathrm { a n o m } } \right)$ and is distinct from forecasting a candidate action’s next layout.

When the state is judged as normal, PEC requests reliability reports for the candidate pairs in $\mathcal { C } _ { t }$ and selects the highest-scoring action-format pair among candidates not flagged as high-risk by PSE.

State Monitoring & Adaptive Recovery After executing the action-format pair $( a ^ { * } , r ^ { * } )$ , PEC captures the new state $s _ { t + 1 }$ and verifies the outcome via both the layout change $\mathscr { D } _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { t + 1 } )$ and a semantic validation from its MLLM, as part of the subsequent step’s Historical Validation. If the execution is judged a failure, PEC triggers recovery according to the failure type:

• Stagnation: For minimal layout change $\begin{array} { r l r } { ( \mathcal { D } _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { t + 1 } ) } & { { } < } & { \tau _ { c } ) } \end{array}$ , PEC treats the command as invalid and retries the action using PSE’s next-best command format.

• Unexpected Transition: If the layout changes significantly $( { \mathcal { D } } _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { t + 1 } ) \geq \tau _ { c } )$ but the MLLM’s semantic validation deems the new state an incorrect outcome, PEC performs a rollback and adds the failed pair to a temporary taboo list $\mathcal { F } _ { t }$ to block immediate reuse.

Overall, PEC selects reliable actions in normal settings and adapts to execution failures. The complete pseudocode is given in Algorithm 1 (Appendix G), and a detailed analysis of rollback scope, usage statistics, and handling of irreversible operations is provided in Appendix H.2.

## 3 Experiments

## 3.1 Implementation Details

We implement PrecogUI on a smartphone UI-automation stack, using Gemini-2.5-Pro as the reasoning backend (Comanici et al. 2025). The system is training-free at the agent level: all modules are non-learned except the PSE’s nextstep layout predictor, which is lightly fine-tuned on InterfereBench and public datasets to model action-conditioned UI transitions. Implementation details are in Appendix D.1.

## 3.2 Benchmarks

To evaluate model performance, we use two types of datasets: (1) Self-constructed InterfereBench, designed to test agent robustness in long-horizon and disturbance tasks. It includes two settings: (i) normal, a clean environment; (ii) perturbed, with dynamic perturbations like overlays and layout changes. (2) Public benchmarks, split into (a) ScreenSpot (Cheng et al. 2024), which measures UI element grounding (text/icon) to assess localization capability, and (b) navigation-centric suites AndroidControl (Li et al. 2024) and GUI-Odyssey (Lu et al. 2024), which evaluate end-to-end task completion and generalization. Android-Control provides Low- and High-level instruction settings, whereas GUI-Odyssey uses a single cross-app navigation setting. Evaluation follows standard GUI agent metrics: success rate (SR) and type matching (TM). See Appendix D.2 for benchmark details.

## 3.3 Main Results

Perturbation Handling. We evaluate PrecogUI on InterfereBench to assess long-horizon and dynamic-UI robustness. We compare against: 1) base models, GPT-4o (OpenAI et al. 2024), Gemini-2.5-Pro (Comanici et al. 2025), and Qwen-2.5-VL (Bai et al. 2025b); and 2) specialized GUI agents, including OmniParser (Wan et al. 2024), InfiGUI-R1 (Liu et al. 2025), AgentCPM-GUI (Zhang et al. 2025b), OS-Atlas (Wu et al. 2025), and UI-TARS (Qin et al. 2025). As shown in Table 1, on the normal subset, PrecogUI reaches an SR of 79.2% (Low) and 52.7% (High), outperforming the best reactive GUI agent by 3.5 and 15.4 percentage points, respectively. On the perturbed subset, PrecogUI degrades by only 10.8 and 11.1 points (Low/High) and outperforms the strongest baseline by 22.4 and 22.1 points, respectively.

Table 1: Performance on InterfereBench under two instruction settings (Low/High), $T M _ { n }$ and $S R _ { n }$ denote type-matching and success rate on the normal (non-perturbed) subset; and $T M _ { a }$ and $S R _ { a }$ denote the corresponding metrics on the perturbed subset.
<table><tr><td rowspan="2">Method</td><td colspan="4">InterfereBench-Low</td><td colspan="4">InterfereBench-High</td><td colspan="2">Average</td></tr><tr><td> $T M _ { n }$ </td><td> $S R _ { n }$ </td><td> $T M _ { a }$ </td><td> $S R _ { a }$ </td><td> $T M _ { n }$ </td><td> $S R _ { n }$ </td><td> $T M _ { a }$ </td><td> $S R _ { a }$ </td><td> $T M _ { m }$ </td><td> $S R _ { m }$ </td></tr><tr><td>GPT-40</td><td>76.7</td><td>22.1</td><td>69.0</td><td>9.3</td><td>73.4</td><td>3.9</td><td>65.0</td><td>1.2</td><td>71.0</td><td>9.1</td></tr><tr><td>Gemini-2.5-Pro</td><td>87.1</td><td>28.4</td><td>80.0</td><td>12.5</td><td>83.8</td><td>17.7</td><td>76.0</td><td>7.5</td><td>81.7</td><td>16.5</td></tr><tr><td>Qwen-2.5-VL</td><td>90.6</td><td>33.7</td><td>82.5</td><td>18.9</td><td>71.7</td><td>36.5</td><td>64.0</td><td>18.0</td><td>77.2</td><td>26.8</td></tr><tr><td>OmniParser</td><td>84.1</td><td>70.9</td><td>76.0</td><td>41.5 (↓29.4)</td><td>70.6</td><td>27.3</td><td>63.0</td><td>12.8 (↓14.5)</td><td>73.4</td><td>38.1</td></tr><tr><td>InfiGUI-R1</td><td>88.9</td><td>73.6</td><td>81.5</td><td>45.8 (↓27.8)</td><td>77.4</td><td>37.3</td><td>68.0</td><td>19.5 (↓17.8)</td><td>79.0</td><td>44.0</td></tr><tr><td>OS-Atlas</td><td>88.4</td><td>72.1</td><td>80.8</td><td>43.3 (↓28.8)</td><td>72.8</td><td>30.4</td><td>63.5</td><td>14.7 (↓15.7)</td><td>76.4</td><td>40.1</td></tr><tr><td>AgentCPM-GUI</td><td>90.0</td><td>75.7</td><td>82.1</td><td>46.0 (↓29.7)</td><td>78.9</td><td>35.7</td><td>66.8</td><td>18.4 (↓17.3)</td><td>79.5</td><td>44.0</td></tr><tr><td>UI-TARS-1.5</td><td>90.8</td><td>74.5</td><td>82.6</td><td>45.0 (↓29.5)</td><td>76.9</td><td>36.6</td><td>66.0</td><td>19.2 (↓17.4)</td><td>79.1</td><td>43.8</td></tr><tr><td>PrecogUI (Qwen3-VL)</td><td>91.4</td><td>76.8</td><td>88.3</td><td>63.7 (↓13.1)</td><td>78.6</td><td>48.9</td><td>72.0</td><td> $3 7 . 2 \ ( \downarrow 1 1 . 7 )$ </td><td>82.6</td><td>56.7</td></tr><tr><td>PrecogUI</td><td>92.9</td><td>79.2</td><td>90.0</td><td> ${ \bf 6 8 . 4 } \ ( \downarrow 1 0 . 8 )$ </td><td>80.0</td><td>52.7</td><td>74.5</td><td> ${ \bf 4 1 . 6 } \left( { \downarrow 1 1 . 1 } \right)$ </td><td>84.4</td><td>60.5</td></tr></table>

Table 2: Results on AndroidControl and GUI-Odyssey. TM and SR denote type matching and success rate, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">AndroidControl-Low</td><td colspan="2">AndroidControl-High</td><td colspan="2">GUI-Odyssey</td><td colspan="2">Average</td></tr><tr><td>TM</td><td>SR</td><td>TM</td><td>SR</td><td>TM</td><td>SR</td><td> $T M _ { m }$ </td><td> $S R _ { m }$ </td></tr><tr><td>GPT-4o (OpenAI et al. 2024)</td><td>74.3</td><td>19.4</td><td>63.1</td><td>21.2</td><td>37.5</td><td>5.4</td><td>58.3</td><td>15.3</td></tr><tr><td>Qwen-2.5-VL (Bai et al. 2025b)</td><td>94.1</td><td>85.0</td><td>75.1</td><td>62.9</td><td>59.5</td><td>46.3</td><td>76.2</td><td>64.7</td></tr><tr><td>UI-TARS-7B (Qin et al. 2025)</td><td>98.0</td><td>90.8</td><td>83.7</td><td>72.5</td><td>94.6</td><td>87.0</td><td>92.1</td><td>83.4</td></tr><tr><td>SeeClick</td><td>93.0</td><td>75.0</td><td>82.9</td><td>59.1</td><td>71.0</td><td>53.9</td><td>82.3</td><td>62.7</td></tr><tr><td>OS-Atlas-4B (Wu et al. 2025)</td><td>91.9</td><td>80.6</td><td>84.7</td><td>67.5</td><td>83.5</td><td>56.4</td><td>86.7</td><td>68.2</td></tr><tr><td>InfiGUI-R1 (Liu et al. 2025)</td><td>96.0</td><td>92.1</td><td>82.7</td><td>71.1</td><td></td><td></td><td>89.4</td><td>81.6</td></tr><tr><td>AgentCPM-GUI (Zhang et al. 2025b)</td><td>94.4</td><td>90.2</td><td>77.7</td><td>69.2</td><td>90.9</td><td>75.0</td><td>87.7</td><td>78.1</td></tr><tr><td>PrecogUI</td><td>94.9</td><td>88.7</td><td>86.8</td><td>76.4</td><td>91.3</td><td>89.1</td><td>91.0</td><td>84.7</td></tr></table>

Table 3: Grounding performance on ScreenSpot. “–” denotes unavailable per-subset results.
<table><tr><td rowspan="2">Method</td><td colspan="2">Mobile</td><td colspan="2">Desktop</td><td colspan="2">Web</td><td rowspan="2">Avg</td></tr><tr><td>Text</td><td>Icon</td><td>Text</td><td>Icon</td><td>Text</td><td>Icon</td></tr><tr><td>GPT-40 Gemini-2.0 Qwen-2.5-VL</td><td>30.5 一</td><td>23.2 一</td><td>20.6 一</td><td>19.4 1</td><td>11.1 一</td><td>7.8 一</td><td>18.8 84.0 84.7</td></tr><tr><td>SeeClick</td><td>78.0</td><td>52.0</td><td>72.5</td><td>30.0</td><td>55.7</td><td>32.5</td><td>53.4</td></tr><tr><td>ShowUI</td><td>92.3</td><td>75.5</td><td>76.3</td><td>61.1</td><td>81.7</td><td>63.6</td><td>75.1</td></tr><tr><td>OmniParser</td><td>93.9</td><td>57.0</td><td>91.3</td><td>63.6</td><td>81.3</td><td>51.0</td><td>73.0</td></tr><tr><td>UI-TARS-7B</td><td>93.0</td><td>75.5</td><td>90.7</td><td>68.6</td><td>84.3</td><td>74.8</td><td>82.3</td></tr><tr><td>InfiGUI-R1</td><td>97.1</td><td>81.2</td><td>94.3</td><td>77.1</td><td>91.7</td><td>77.6</td><td>87.5</td></tr><tr><td>PrecogUI</td><td>96.5</td><td>87.8</td><td>97.5</td><td>82.2</td><td>94.6</td><td>91.7</td><td>91.2</td></tr></table>

Grounding Capability. Table 3 reports ScreenSpot grounding accuracy across mobile, desktop, and web interfaces. PrecogUI achieves the best sample-weighted average accuracy of 91.2% and leads five of the six displayed subsets; InfiGUI-R1 remains 0.6 points higher on Mobile Text. Overall, PrecogUI exceeds InfiGUI-R1’s weighted average by 3.7 points. Further discuss in Appendix H.1.

Navigation Capability. To validate generalization, PrecogUI is evaluated on AndroidControl and GUI-Odyssey. As shown in Table 2, PrecogUI achieves SRs of 88.7% (Low) and 76.4% (High) on AndroidControl and 89.1% on GUI-Odyssey. Relative to Qwen-2.5-VL, the gains on AndroidControl-Low and -High are 3.7% and 13.5%. Compared with UI-TARS-7B, PrecogUI is 2.1 points lower on AndroidControl-Low but 3.9 and 2.1 points higher on AndroidControl-High and GUI-Odyssey, yielding a 1.3- point gain in average SR over the three datasets. Moreover, Table 1 shows that replacing the reasoning backbone with Qwen3-VL (Bai et al. 2025a) yields slightly lower yet competitive results, consistently surpassing the specialized GUIagent baselines on InterfereBench and indicating that the robustness gains are not tied to a single reasoning backend.

## 3.4 Ablation Study

Effectiveness of the Experience Pool (PEP) and Incremental Components. To assess the contributions of PrecogUI components, we conduct cumulative ablations on the InterfereBench strong-perturbation subset. As shown in Figure 10(a), the reactive baseline achieves 41.5% SR; adding PEP, PSE, and PEC successively raises SR to 52.1%, 61.8%, and 68.4%, respectively. The corresponding incremental gains are 10.6, 9.7, and 6.6 percentage points. On the clean long-horizon subset, SR similarly increases from 70.9% to 74.8%, 77.1%, and 79.2%. These controlled results show the incremental contribution of next-layout forecasting within the complete pipeline. Further reliability and PEP analyses are provided in Appendix D.3.

![](images/6f4fa64d841fa165185766b77a5064b48f10d940a05d6e4376e9b6683acbb471.jpg)

Figure 4: PSE prediction quality. Left: current state $L _ { t } ;$ center: predicted symbolic layout $\hat { L } _ { t + 1 } ;$ right: actual next state.  
![](images/4486ec5c5dc33db0f6c34c7dcf1d6904b735abe509fa45ee608d202faebe8b35.jpg)  
Figure 5: Case visualization of PrecogUI under representative anomalous scenarios.

PSE Prediction Quality. We evaluate PSE’s predicted layouts against ground truth on held-out test splits. Figure 4 shows a representative case: given the current state $L _ { t }$ (left) and a tap action, PSE generates a symbolic layout $\hat { L } _ { t + 1 }$ (center), containing only (type, bbox) elements on a blank canvas. The rule-based estimator then flags the predicted occlusion as a pop-up anomaly, while PEC supplies the handling action, consistent with the actual next state (right).

Case Visualization. As illustrated in Figure 5, PrecogUI robustly handles anomalies across cross-application, strongly perturbed, and dynamic GUI tasks, including application and in-game pop-ups, black or white screens, and abrupt layout shifts caused by system-level interruptions. Beyond reacting to visible disturbances, it predicts potential layout changes and latent anomalies. During loading delays, for example, it waits rather than interacting with blank or unresponsive screens. This predictive avoidance enables reliable execution in complex workflows, distinguishing PrecogUI from conventional reactive agents. More cases are provided in Appendix F.

## 4 Related Work

Multimodal large language models (MLLMs) (Li et al. 2023; Liu et al. 2024; Chen et al. 2024) provide the visuallanguage representations used by many GUI agents, yet models trained primarily on static perception tasks do not by themselves provide persistent state tracking in dynamic interfaces. Building upon MLLMs, GUI agents (Gou et al. 2025; Liu et al. 2025; Lin et al. 2025; Kang et al. 2026a) learn sequential action policies by mapping instruction– screenshot pairs to grounding and interaction. Recent efforts further introduce structure and memory: AutoDroid (Wen et al. 2024) injects app-specific knowledge collected through exploration, while MapAgent (Kong et al. 2025) retrieves structured page memories during planning. ViMo (Luo et al. 2025) instead predicts full future GUI observations for candidate-action selection. PrecogUI follows this proactive world-modeling direction but targets perturbation-robust execution: its lightweight symbolic predictor supplies geometric risk cues, while PEP and PEC provide experience retrieval and closed-loop recovery. A comprehensive discussion is provided in Appendix B.

## 5 Concluding Remarks

In this work, we present PrecogUI for reliable and adaptive long-horizon GUI execution under frequent and diverse disturbances. Through an experience–foresight–feedback loop,

PEP retrieves prior success and anomaly patterns, PSE predicts future layouts and ranks candidate actions by estimated reliability, and PEC enables monitored execution with retry and rollback. Experiments on InterfereBench and public benchmarks demonstrate improved task success and robustness in complex, dynamic environments.

## References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; and et al., H. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. In Advances in Neural Information Processing Systems, volume 35, 23716–23736.

Bai, S.; Cai, Y.; Chen, R.; et al. 2025a. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.;Dang, K.; and et al. 2025b. Qwen2.5-VL Technical Report.arXiv:2502.13923.

Chen, W.; Cui, J.; Hu, J.; Qin, Y.; Fang, J.; Zhao, Y.; Wang, C.; Liu, J.; Chen, G.; Huo, Y.; Yao, Y.; Lin, Y.; Liu, Z.; and Sun, M. 2025. GUICourse: From General Vision Language Model to Versatile GUI Agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 21936–21959.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen; and et al. 2024. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 24185–24198.

Cheng, K.; Sun, Q.; Chu, Y.; Xu, F.; Li, Y.; Zhang, J.; and Wu, Z. 2024. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 9313–9332.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv preprint arXiv:2507.06261.

Dai, W.; Li, J.; LI, D.; Tiong, A.; Zhao, J.; Wang, W.; Li, B.; Fung, P. N.; and Hoi, S. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In Advances in Neural Information Processing Systems, volume 36, 49250–49267.

Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems, volume 36, 28091–28114. Curran Associates, Inc.

Fan, Y.; Zhao, H.; Zhang, R.; Shen, Y.; Wang, X. E.; and Wu, G. 2025. GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration. arXiv:2501.13896.

Gao, L.; Zhang, L.; Wang, S.; Wang, S.; Li, Y.; and Xu, M. 2024. MobileViews: A Large-Scale Mobile GUI Dataset. arXiv:2409.14337.

Gao, X.; Hu, C.; Chen, B.; and Li, T. 2025. Chainof-Memory: Enhancing GUI Agents for Cross-Application Navigation. arXiv preprint arXiv:2506.18158.

Gou, B.; Wang, R.; Zheng, B.; Xie, Y.; Chang, C.; Shu, Y.; Sun, H.; and Su, Y. 2025. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents. In The Thirteenth International Conference on Learning Representations.

Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang, Y.; Wang, Z.; Dong, Y.; Ding, M.; and Tang, J. 2024. CogAgent: A Visual Language Model for GUI Agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14281–14290.

Huang, Z.; Cheng, Z.; Pan, J.; Hou, Z.; and Zhan, M. 2025. SpiritSight Agent: Advanced GUI Agent with One Look. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 29490–29500.

Kang, B.; Chen, B.; Wang, J.; Li, Y.; Zhao, J.; Wang, J.; and Tian, Z. 2025. CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval. In Proceedings of the 33rd ACM International Conference on Multimedia, 5140–5149.

Kang, B.; Wen, S.; Bi, Y.; Wu, S.; Yuan, X.; Shao, R.; Wang, J.; and Tian, Z. 2026a. LongHorizonUI: A Unified Framework for Robust Long-Horizon Task Automation of GUI Agent. In The Fourteenth International Conference on Learning Representations.

Kang, B.; Wen, S.; Fan, Y.; Wu, S.; Wang, J.; Li, Y.; Zhao, J.; Wang, J.; and Tian, Z. 2026b. AgentSteerTTS: A Multi-Agent Closed-Loop Framework for Composite-Instruction Text-to-Speech. arXiv:2605.17583.

Kong, Y.; Shi, D.; Yang, G.; ke di, Z.; Huang, C.; Li, X.; and Jin, S. 2025. MapAgent: Trajectory-Constructed Memory-Augmented Planning for Mobile Task Automation. arXiv:2507.21953.

Lei, W.; Gao, D.; and Shou, M. Z. 2025. Grounding Multimodal Large Language Model in GUI World. In The Thirteenth International Conference on Learning Representations.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In Proceedings of the 40th International Conference on Machine Learning, volume 202, 19730–19742.

Li, W.; Bishop, W.; Li, A.; Rawles, C.; Campbell-Ajala, F.; Tyamagundlu, D.; and Riva, O. 2024. On the Effects of Data Scale on Computer Control Agents. arXiv:2406.03679.

Lin, K. Q.; Li, L.; Gao, D.; Yang, Z.; Wu, S.; Bai, Z.; Lei, S. W.; Wang, L.; and Shou, M. Z. 2025. ShowUI: One Vision-Language-Action Model for GUI Visual Agent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19498–19508.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 26296–26306.

Liu, Y.; Li, P.; Xie, C.; Hu, X.; Han, X.; Zhang, S.; Yang, H.; and Wu, F. 2025. InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners. arXiv:2504.14239.

Lu, Q.; Shao, W.; Liu, Z.; Meng, F.; Li, B.; Chen, B.; Huang, S.; Zhang, K.; Qiao, Y.; and Luo, P. 2024. GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. arXiv:2406.08451.

Lu, Y.; Yang, S.; Qian, C.; Chen, G.; Luo, Q.; and et al. 2025. Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance. In Proceedings of the International Conference on Learning Representations (ICLR), 1–27.

Luo, D.; Tang, B.; Li, K.; Papoudakis, G.; Song, J.; Gong, S.; Hao, J.; Wang, J.; and Shao, K. 2025. ViMo: A Generative Visual GUI World Model for App Agents. arXiv:2504.13936.

Ma, J.; Wang, P.; Kong, D.; Wang, Z.; Liu, J.; Pei, H.; and Zhao, J. 2024. Robust Visual Question Answering: Datasets, Methods, and Future Challenges. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8): 5575–5594.

OpenAI; Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; and et al. 2024. GPT-4o System Card. arXiv:2410.21276.

Qin, Y.; Ye, Y.; Fang, J.; Wang, H.; and et al., S. L. 2025. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. arXiv:2501.12326.

Rawles, C.; Clinckemaillie, S.; Chang, Y.; Waltz, J.; and et al., G. L. 2025. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents. arXiv:2405.14573.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-Resolution Image Synthesis With Latent Diffusion Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.

Sun, Y.; Zhao, S.; Yu, T.; Wen, H.; Va, S.; Xu, M.; Li, Y.; and Zhang, C. 2025. GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR), 19477–19486.

Wan, J.; Song, S.; Yu, W.; Liu, Y.; Cheng, W.; Huang, F.; Bai, X.; Yao, C.; and Yang, Z. 2024. OmniParser: A Unified Framework for Text Spotting Key Information Extraction and Table Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15641–15653.

Wang, J.; Xu, H.; Jia, H.; Zhang, X.; Yan, M.; Shen, W.; Zhang, J.; Huang, F.; and Sang, J. 2024. Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration. In Advances in Neural Information Processing Systems, volume 37, 2686–2710.

Wen, H.; Li, Y.; Liu, G.; Zhao, S.; Yu, T.; Li, T. J.-J.; Jiang, S.; Liu, Y.; Zhang, Y.; and Liu, Y. 2024. AutoDroid: LLMpowered Task Automation in Android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 543–557.

Wu, Z.; Wu, Z.; Xu, F.; Wang, Y.; Sun, Q.; Jia, C.; Cheng, K.; Ding, Z.; Chen, L.; Liang, P. P.; and Qiao, Y. 2025. OS-ATLAS: Foundation Action Model for Generalist GUI Agents. In International Conference on Learning Representations, 5090–5108.

Xie, T.; Zhang, D.; Chen, J.; Li, X.; Zhao, S.; Cao, R.; Hua, T. J.; Cheng, Z.; Shin, D.; Lei, F.; Liu, Y.; Xu, Y.; Zhou, S.; Savarese, S.; Xiong, C.; Zhong, V.; and Yu, T. 2024. OS-World: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. In Advances in Neural Information Processing Systems, volume 37, 52040– 52094.

Xu, Y.; Lu, D.; Shen, Z.; Wang, J.; Wang, Z.; Mao, Y.; Xiong, C.; and Yu, T. 2025a. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials. In The Thirteenth International Conference on Learning Representations.

Xu, Y.; Wang, Z.; Wang, J.; Lu, D.; Xie, T.; Saha, A.; Sahoo, D.; Yu, T.; and Xiong, C. 2025b. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction. In Fortysecond International Conference on Machine Learning.

Yang, J.; Tan, R.; Wu, Q.; Zheng, R.; Peng, B.; Liang, Y.; Gu, Y.; Cai, M.; Ye, S.; Jang, J.; Deng, Y.; and Gao, J. 2025. Magma: A Foundation Model for Multimodal AI Agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14203–14214.

Ye, J.; Zhang, X.; Xu, H.; Liu, H.; Wang, J.; Zhu, Z.; Zheng, Z.; Gao, F.; Cao, J.; Lu, Z.; Liao, J.; Zheng, Q.; Huang, F.; Zhou, J.; and Yan, M. 2025. Mobile-Agent-v3: Fundamental Agents for GUI Automation. arXiv:2508.15144.

Yue, X.; Ni, Y.; Zhang, K.; Zheng, T.; Liu, R.; and et al. 2024. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9556–9567.

Zhang, Z.; Dai, Q.; Bo, X.; Ma, C.; Li, R.; and et al. 2025a. A Survey on the Memory Mechanism of LLM-based Agents. ACM Transactions on Information Systems (TOIS), 43(6): 155:1–155:47.

Zhang, Z.; Lu, Y.; Fu, Y.; Huo, Y.; Yang, S.; and et al., Y. W. 2025b. AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning. arXiv:2506.01391.

## Appendix

This is the supplementary file for our submission titled PrecogUI: Proactive GUI agents via Pre-cognitive Simulation and Experience Retrieval. This material supplements the main paper with the following content:

• (A) Motivation of PrecogUI

• (B) Related work

• (C) PolyTouch: A Multi-Gesture and Macro Execution Layer

• (D) Additional Experiments

– (D.1) Implementation Details

– (D.2) Benchmarks

– (D.4) Details of Diffusion-based Future Layout Generation

– (D.5) Hyperparameter analysis of the exploration value

– (D.6) Hyperparameter analysis of parser thresholds

– (D.7) PSE prediction quality evaluation

## • (E) Prompts in Automated Pipeline

– (E.1) Output Format Structure Template

– (E.2) Action Selection Template

– (E.4) Role and Context Template

– (E.3) Anomaly Handling Template

– (E.5) OS-Specific Hints

– (E.6) General Instructions

• (F) Qualitative Analysis

• (G) PEC Algorithm Pseudocode

• (H) Additional Discussions

– (H.1) Cross-Device Grounding and Mobile Navigation Analysis

– (H.2) Rollback Scope and Empirical Statistics

## A Motivation of PrecogUI

Figure 6 reveals two critical patterns. First, the success rate (SR) declines sharply with an increasing number of injected disturbances. Reactive baselines plummet from nearly 100% SR to below 20% with zero to six injections, showing a performance gap of at least 10% by just two injections (left panel). This highlights the inherent brittleness of purely reactive policies under sustained interference. Second, disturbance timing significantly impacts performance (right panel). Shifting a single injection later in the trajectory yields greater SR losses across all baselines. For instance, UI-TARS exhibits an SR drop escalating from 3.0% (steps 0–5) to 16.4% (> 20 steps). In contrast, PRE-COGUI demonstrates consistent resilience, increasing only from 1.6% to 7.1%—approximately 2.3× less degradation than UI-TARS in the long-horizon tail—while maintaining higher nominal SR. These trends suggest that coupling experience priors with look-ahead simulation is crucial for mitigating late-stage error cascades.

## B Related Work

Multimodal Large Language Models. MLLMs (Li et al. 2023; Liu et al. 2024; Chen et al. 2024) have emerged as a central enabler for GUI automation, boosting both perceptual and reasoning capabilities of agents. By parsing complex screen structures and grounding natural-language instructions in UI elements, MLLMs serve as the perception backbone for mainstream agents (Wang et al. 2024; Yang et al. 2025). Benchmarks such as MMMU (Yue et al. 2024) measure broad multimodal understanding, while taskspecific studies evaluate capabilities such as visual question answering (Ma et al. 2024) and image captioning (Dai et al. 2023). These static or single-turn capabilities are useful foundations but do not alone provide persistent state tracking or anticipatory control in dynamic interfaces.

GUI agents. Research on GUI agents (Gou et al. 2025; Liu et al. 2025; Xu et al. 2025a) has explored diverse strategies for policy learning and grounding. A common paradigm (Lin et al. 2025; Kang et al. 2026a) is to fine-tune multimodal models to map instruction–screenshot inputs into sequential action predictions. For example, UGround (Gou et al. 2025) trains a purely visual grounding model on millions of UI elements, enabling click and operation solely through visual localisation. Recent efforts (Gao et al. 2025; Zhang et al. 2025a) have added structure and memory: Auto-Droid (Wen et al. 2024) injects app-specific knowledge collected through automated exploration, and MapAgent (Kong et al. 2025) retrieves structured page memories during planning. While effective on common GUI benchmarks (Gao et al. 2024), these methods (Lei, Gao, and Shou 2025; Xu et al. 2025b) remain largely reactive, leaving them vulnerable to unforeseen perturbations. An unexpected pop-up can hijack the agent’s attention, while a loading delay may be misinterpreted as a failed action.

Adjacent Multimodal and Proactive-Agent Research. Several studies outside direct GUI action prediction provide complementary perspectives. CalibCLIP (Kang et al. 2025) calibrates dominant visual and textual tokens for text-driven image retrieval, illustrating how representation-level calibration can reduce misleading visual semantics, although it does not model GUI transitions. Proactive Agent (Lu et al. 2025) studies when an LLM agent should initiate assistance from contextual signals rather than waiting for an explicit request; its notion of proactivity is conceptually related but differs from PrecogUI’s action-conditioned UI forecasting. AgentSteerTTS (Kang et al. 2026b) demonstrates a multiagent closed loop for composite-instruction speech synthesis. We cite it only as a cross-domain example of iterative feedback control, not as a GUI perception or execution baseline.

GUI world models. ViMo (Luo et al. 2025) predicts full future GUI observations for action selection. PrecogUI adopts the same action-conditioned foresight principle but predicts lightweight symbolic layouts tailored to geometric anomaly detection and closed-loop recovery.

![](images/8eb5e1f344f566174b9440485be8846bafb8fa8c411a915964c4ddc5a9fd3817.jpg)  
(a) Number of Injection Disturbances

![](images/8f4fbf804ccef54127475bf66450e66f5c508221aad9e2000cc2797c0938eafd.jpg)  
(b) Injection step window

Figure 6: Impact of disturbance count and timing on policy success rates. The left panel shows SR degradation with increasing disturbance count. The right panel illustrates the greater sensitivity of reactive policies to later disturbance injections, in contrast to PRECOGUI’s robustness.  
![](images/32a30a47ffb5e68978ff1bf3d000c812bebc3f203e4de3e0cca4fdd56277de88.jpg)  
Figure 7: The illustration of PolyTouch, a multi-gesture and macro execution layer for GUI agents. It depicts multi-finger gestures and macro-level commands, highlighting their role in robust, long-horizon task execution.

## C PolyTouch: A Multi-Gesture and Macro Execution Layer

Real-world mobile applications often require multi-pointer and multi-step interactions, such as three- or four-finger system shortcuts, pinch/zoom and rotation in media and map viewers, or coordinated sequences in creative tools. Existing GUI agents generally assume single-touch atomic operations and one-shot execution, which makes them fragile when facing complex gestures, long interaction flows, or OS-level controls that demand precise synchronization. To address this gap, we introduce PolyTouch, an execution layer that extends the action space to multi-finger gestures and macro-level commands with explicit timing, guards, and rollback mechanisms.

PolyTouch supports a wide range of interaction patterns rarely considered in prior work: (i) Multi-finger chords for dialogs, split-screen, or editing shortcuts; (ii) Continuous gestures such as pinch, zoom, and rotation; (iii) Multi-step flows with explicit waiting, retries, and overlay dismissal; (iv) Recovery sequences (e.g., back, home, or targeted close) that must be executed atomically to exit unexpected states. These abstractions allow agents to operate robustly in longhorizon tasks where traditional atomic actions fail.

PolyTouch builds on Appium’s W3C Actions API for deterministic multi-pointer synthesis and it falls back to ADB when accessibility channels are blocked. Its design centers on: (1) deterministic timing through tick-based scheduling; (2) unified coordinate formats (index, relative-in-box, absolute) with boundary-safe mapping; (3) a declarative macro interface that bundles taps, swipes, multi-swipes, key events, and waits into atomic, retryable units; (4) graceful degradation to equivalent ADB commands while preserving ordering and timing.

PolyTouch exposes two main capabilities: (a) Multigesture execution. Three- and four-finger gestures are represented as synchronized pointer streams (pointerDown → pointerMove → pointerUp), while pinch/zoom and rotation are parameterized around target boxes and derived from relative coordinates. (b) Macro execution. JSON-defined macros encapsulate an ordered list of primitives with explicit guard, retry, and rollback semantics, supporting flexible coordinate specifications.

PolyTouch integrates into the agent control loop by providing reliability-aware plans and structured execution reports (success flags, layout changes, anomaly tags). These outputs feed the Proactive Experience Pool to accumulate reusable patterns and guide the Pre-cognitive Execution Controller in anticipating failures and triggering recovery. In this way, PolyTouch transforms low-level taps into a closedloop, macro-level control primitive that is both expressive and robust.

## D Additional Experiments

## D.1 Implementation Details

Hardware & Devices. All experiments were conducted on a single training node with 8× NVIDIA H20 (96 GB) GPUs. For on-device evaluation, we used a pool of mainstream Android phones covering Huawei/Honor, Xiaomi/Redmi, and OPPO/realme, spanning Android 10–14 and common resolutions (720p–1440p). Devices were connected over USB with ADB (USB debugging enabled) for reliable screenshot capture and input dispatch; Wi-Fi ADB was used only for long-duration soak tests.

Data Collection & Real-World Tests. We employ Appium 2.x (Android driver: uiautomator2) together with ADB to (i) scrape view hierarchies and screenshots, (ii) execute action sequences in real apps, and (iii) log pre/post frames, timing, and outcomes for replayable trajectories. For latency-critical fallback (e.g., when Appium is blocked by transient overlays), we issue low-level commands via adb shell input (tap/swipe/keyevent) and re-sync with Appium on the next stable frame. Randomized perturbation placement, candidate ordering, and PSE initialization use fixed seeds; device capture settings are held constant. Screen coordinates are normalized to [0, 1] and mapped to device pixels at runtime.

Evaluation Isolation. Each trajectory corpus is partitioned before PSE training, and no transition from a held-out evaluation trajectory is included in the training split. For every reported test episode, PEP is initialized from the same training-split memory and is reset to that state before the episode begins. Updates collected within an episode may guide later steps of that episode but are discarded at termination; therefore, no memory entry created from one test episode can be retrieved in another. The same held-out task split and action budget are used for all methods within each benchmark.

## D.2 Benchmarks

Grounding-Centric Benchmark: ScreenSpot. Accurate element localization is the foundation of GUI automation. ScreenSpot (Cheng et al. 2024) is a cross-platform grounding benchmark with over 1,200 natural-language instructions spanning iOS, Android, macOS, Windows, and Web interfaces. Each instruction is paired with pixel-level bounding boxes and element-type labels (text, icon, or widget) and covers challenging scenarios such as icon-text composites and occluded controls.

Navigation-Centric Benchmarks: AndroidControl & GUI Odyssey. Once elements can be reliably located, agents must navigate within and across apps. AndroidControl (Li et al. 2024) contains 15,283 human demonstrations of everyday Android tasks. Each task is paired with a highlevel goal instruction and a low-level, step-by-step instruction; the Low/High labels in our tables refer to instruction granularity rather than separate single-app and cross-app difficulty partitions. We use the GUI-Odyssey v1 protocol (Lu et al. 2024), which contains 7,735 cross-app episodes collected on six mobile devices and spans 201 apps and approximately 1.4K app combinations. GUI-Odyssey is reported as a single cross-app navigation setting and is not assigned AndroidControl-style Low/High labels.

Disturbance-Aware Benchmark: InterfereBench. InterfereBench covers 34 applications—complex games, enterprise tools, and general apps—with bilingual (Zh/En) UIs recorded on diverse phone models. It contains 1,160 tasklevel trajectory groups whose clean source trajectories span 14–37 steps and contain 27,124 annotated screenshots. We captured 574 real abnormal screens and curated 217 synthetic disturbance assets (pop-ups, notifications, black screens, and layout shifts). Each task group comprises one clean execution and two controlled perturbation replays; replayed frames are generated online and are not added to the source-screenshot count. Annotations include high-level goals and step-level structures (action type, normalized coordinates, UI boxes, screen deltas, and outcomes).

## D.3 Further Analysis of Proactive Reliability Mechanisms

Analysis of Proactive Reliability Forecasting. PSE forecasts the next-step layout for each concrete index, relative, or absolute command and assigns a relative reliability score. Figure 10(b) illustrates how the resulting risk signal supports command-format correction. Thus, the hierarchy Index → Relative-in-Box → Absolute is a default preference rather than an immutable order: PSE may override it when another concrete command is predicted to be safer in the current context.

In-Depth Analysis of PEP. We study PEP’s cold-start behavior and entry staleness under app evolution. As shown in Figure 9(a), even with zero entries the system achieves 53.8% SR (thanks to PSE and PEC); adding 50 entries boosts SR to 60.4%, and SR saturates ∼500 entries, indicating that a compact pool of recurring patterns suffices. For staleness (Figure 9(b)), we compare three strategies across five simulated app-update rounds: No Expiry degrades SR to 49.6%, LRU Eviction retains 61.5%, while our Version-Aware policy with exponential staleness decay best preserves effectiveness at 65.4% (−3.0% vs. the fresh pool).

![](images/9c7bc9664b61288432a1d9e02419db8a3dbf314e220ebb2b22b064d8af6f7fd6.jpg)  
(a) Effect of exploration weight α on 50-step coverage and novelty.

![](images/178459daeae77560e168e9c935ba2cc09d9e3d581126646de2479599e191ec64.jpg)  
(b) Trade-off of Stage-1 parser acceptance under different $\tau _ { c }$

Figure 8: Ablation of exploration value and parsing thresholds. (a) Larger α accelerates unique-control coverage but reduces rare-state revisits. (b) Moderate pruning near $\tau _ { c } \in [ 0 . 2 5 , 0 . 3 5 ]$ gives the best acceptance–success–efficiency balance.  
![](images/3c20c1499fb91a9cafedb3808caa3448211cd1e1f64d56cbb7b10a20f8d77c8d.jpg)

![](images/1e8382e0bdd33a9200597fdf89643eec23265a4c98f0b6ab286bb979a1f09b05.jpg)  
Figure 9: In-depth analysis of PEP. (a) SR as a function of experience pool size, showing rapid gains up to 200 entries and saturation beyond 500. (b) Memory-management strategy comparison across app-update rounds; the version-aware policy best preserves effectiveness.

## D.4 Details of Diffusion-based Future Layout Generation

For completeness, we outline the training setup of the conditional latent diffusion model used in PSE. Each layout is linearized into a sequence of at most N UI elements $e = ( \mathrm { t y p e } , \mathrm { b b o x } )$ sorted in reading order. Element types are embedded with a learned lookup table, while bounding boxes $( x _ { \mathrm { m i n } } , y _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } )$ are normalized to [0, 1] and projected by a linear layer. A Transformer encoder $E _ { \phi }$ maps the ground-truth next layout $L _ { t + 1 }$ to a clean latent $\mathbf { z } _ { 0 } ~ \in \mathbb { R } ^ { d }$ , and a Transformer decoder $D _ { \psi }$ reconstructs the ordered element sequence. The encoder–decoder is trained with element-type cross-entropy and bounding-box regression losses. Following latent diffusion (Rombach et al. 2022), the denoiser $\epsilon _ { \theta }$ is conditioned on $\mathbf { C } = f _ { \omega } ( L _ { t } , c _ { t , j } )$ and optimized as

$$
\begin{array} { r l } & { \quad \mathbf { z } _ { 0 } = E _ { \phi } ( L _ { t + 1 } ) , } \\ & { \quad \quad \mathbf { z } _ { i } = \sqrt { \bar { \alpha } _ { i } } \mathbf { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { i } } \epsilon , } \\ &  \quad \quad \mathcal { L } _ { \mathrm { d i f f } } ( \theta ) = \mathbb { E } _ { \underset { i \sim \mathcal { M } \{ 1 , \ldots , N _ { \mathrm { t r a i n } } \} , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } { ( L _ { t } , c _ { t , j } , L _ { t + 1 } ) \sim \mathcal { D } , } \left[ \lVert \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { i } , i , \mathbf { C } ) \rVert _ { 2 } ^ { 2 } \right] . } \end{array}\tag{4}
$$

Here, D denotes the trajectory training distribution, and the conditioning vector is obtained by concatenating pooled embeddings of the current layout and concrete executable command, followed by a linear projection. The reconstruction and diffusion objectives are optimized jointly, with the reconstruction terms supervising element types and bounding boxes and ${ \mathcal { L } } _ { \mathrm { d i f f } }$ supervising the conditional latent transition.

![](images/d313f4f3b56faac5c125b85723a2609c659393efc53c402c1802bfb7730074a4.jpg)  
（a ) PrecogUI Component Ablation

![](images/970dea3ab92010829e3a809ca0b195f6794747179f481d0c0d73cc0e6a313c50.jpg)  
（b) Failure correction for actions command

Figure 10: Ablation and reliability analysis of PrecogUI. (a) Cumulative component contributions on clean and strongperturbation subsets; the legacy labels PME and PCE in the exported panel correspond to PSE and PEC, respectively. (b) Representative command-format failure correction using PSE reliability forecasts.
<table><tr><td colspan="2">Output Format Structure Template: Defines the Mandated JSON Structure for Agent Output.</td></tr><tr><td>{&quot; Historical_status&quot;: &quot;Success|Failed|Unknown - Evaluate if the previous action visually achieved its intended goal. Base this ONLY on the screen image. Ignore the execution result status provided in the input.&quot;, &quot;import contents&quot;: &quot;Output important contents closely related to user\&#x27;s instruction on the current page. If there is, please output the contents. If not, please output empty string &quot;.&quot;, “think”: “Provide a step-by-step thinking process. Analyze the current screen, relate it to the overall task and the visual outcome of the previous step (Historical_status&#x27;). Decide the next best *single* action. Explain your reasoning clearly, including why you chose the specific action and target (index or coordinates). If &#x27;evaluation_prev_goal&#x27; was &#x27;Failed&#x27;, reflect on why and how the next action addresses it.&quot;, &quot;progress&quot;: &quot;Assess the overall progress. List the sub-tasks from the main `task&#x27;or `task_steps&#x27;that have been verifiably completed up to the current step. This judgment must be based on the *actual visual evidence* on the screen, not merely on the sequence of actions planned or executed, as actions can fail or produce unexpected results.&quot;, &quot;next_goal&quot;: &quot;Briefly describe the specific, immediate goal of the *next action* you are proposing in the &#x27;action&#x27; field.&quot;, &quot;action&quot;: { &quot;action_name&quot;: { /* dictionary of parameters for the action */ } } }</td><td>{&quot; Historical status &quot;: &quot;Failed - The previous action (open settings) was blocked by a pop-up overlay.&quot;, &quot;import_contents&quot;: &quot;Popup dialog is visible with title &#x27;活动提醒&#x27;; two buttons: 立即前往&#x27; (index 15) and &#x27;稍后再说&#x27; (index 9). A close icon &#x27;X&#x27; at top-right (index 5). Underlying Honor of Kings lobby is dimmed and not interactable.&quot;, &quot;think&quot;: &quot;A blocking popup covers the lobby. The overall task is to continue interacting with the lobby. The safest way to resume is to dismiss the popup without navigating elsewhere. The close icon at index 5 is visible and should remove the overlay without side effects. I will use the &#x27;click&#x27; action with this index.&quot;, &quot;progress&quot;: [&quot;1. Reached the main lobby (visible in background).&quot;], &quot;next_goal&quot;: &quot;Dismiss the popup to restore interaction with the lobby.&quot;, &quot;action&quot;: {&quot;click&quot;: {&quot;position&quot;: 5} }}</td></tr></table>

Figure 11: Mandated JSON Schema for Agent Reasoning. The figure shows the output template (left) and an in-context example of handling a pop-up overlay (right).

We train the model on the training splits of InterfereBench, AndroidControl (Li et al. 2024), and GUI-Odyssey (Lu et al. 2024), using ground-truth next layouts as supervision; held-out evaluation trajectories are excluded as described in Sec. D.1. We use $\bar { N _ { \mathrm { t r a i n } } } ~ = ~ 1 0 0$ diffusion timesteps with a cosine noise schedule and optimize with

AdamW (learning rate $1 \times 1 0 ^ { - 4 }$ , weight decay 0.01, batch size 128) and gradient clipping at 1.0. At inference, we use a deterministic reverse sampler over $N _ { \mathrm { s a m p l e } } = 5 0$ selected timesteps. The initial latent is sampled from $\mathbf { z } _ { N _ { \mathrm { t r a i n } } }$ ∼ $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ using a fixed per-instance seed; conditional on this initial latent, the reverse trajectory introduces no additional stochastic noise. The resulting latent is decoded by $D _ { \psi }$ into element-type logits and bounding-box offsets, yielding the symbolic layout $\hat { L } _ { t + 1 }$ consumed by the anomaly and reliability estimators in Sec. 2.4.

## D.5 Hyperparameter Analysis of the Exploration Value

We study the single coefficient α that balances novel-control discovery against rare-state probing during exploration. As illustrated in Figure 8a, on a 50-step horizon, larger settings $( \mathbf { e . g . } , \alpha \ge 0 . 5 )$ consistently deliver higher cumulative coverage and higher moving-average novelty, indicating faster expansion of the actionable UI space. Very small α emphasizes repeatedly visiting under-explored screens; while this can stabilize early behavior, it sacrifices coverage and slows progress. We observe no significant increase in redundancy within 50 steps, suggesting that short-horizon exploration benefits most from prioritizing discovery. In practice, $\alpha \in [ 0 . 5 , 0 . 7 5 ]$ is a strong operating region that frontloads novel controls without noticeable revisit overhead. For longer horizons or highly volatile apps, an adaptive schedule is preferable: start near α ≈ 0.5 to stabilize initial navigation, then increase toward 0.75–1.0 as the uncovered-control ratio declines. Overall, α provides an interpretable knob for exploration granularity; tuning (or scheduling) it materially impacts coverage speed and downstream success rates.

## D.6 Parser-Threshold Analysis

Filtering Statistics. Stage-1 removes trajectories with excessive length and redundancy. We formalise the self-loop and no-op statistics as:

$$
\begin{array} { r l } & { \rho _ { \mathrm { l o o p } } = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { 1 } [ a _ { t } \in \mathcal { A } _ { \mathrm { l } } ] \ \frac { \mathrm { m a x } \left( 0 , \tau _ { \mathrm { c } } - \mathcal { D } _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { t + 1 } ) \right) } { \tau _ { \mathrm { c } } } , } \\ & { \rho _ { \mathrm { n o o p } } = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathrm { m a x } \left( 0 , \tau _ { \mathrm { c } } - \mathcal { D } _ { \mathrm { l a y o u t } } ( L _ { t } , L _ { t + 1 } ) \right) } { \tau _ { \mathrm { c } } } , } \end{array}\tag{5}
$$

where $T$ is the number of steps, $a _ { t }$ is the action at step $t ,$ and $L _ { t }$ is the UI layout at step t. Here, 1[·] denotes the indicator function, $\mathcal { D } _ { \mathrm { l a y o u t } }$ is the layout-difference measure derived from the Dice-style similarity in Eq. $( 2 ) , \tau _ { \mathrm { c } }$ is the noop threshold, and $\boldsymbol { \mathcal { A } } _ { \mathrm { l } }$ denotes the set of actions annotated as layout-preserving self-loops. A trajectory is pruned as structurally low-quality if either statistic exceeds its preset threshold.

We examine how the Stage-1 pruning thresholds (selfloop ratio and no-op ratio) interact with the no-op cutoff and impact downstream quality, as shown in Fig. 8b. As the cutoff increases, the acceptance rate drops monotonically across all regimes (e.g., from ∼0.60–0.65 at a low cutoff of 0.05 to ∼0.20 at 0.45), indicating that more micro-changes are filtered as no-ops. Strict pruning rapidly depresses acceptance (often < 0.25 once the cutoff exceeds ≈0.20), and downstream quality declines as data volume becomes the bottleneck. Lenient pruning maintains high acceptance (> 0.55 across most cutoffs) but retains many low-signal segments; the success proxy plateaus or degrades when the cutoff is high $( \mathrm { e . g . }$ , normalized success ${ \lesssim } \bar { 0 } . 5 5$ once the cutoff ≥0.35). By contrast, the Moderate regime achieves the best balance in a mid-range cutoff of 0.25–0.35: acceptance stays around 0.35–0.50 while the normalized success proxy peaks around 0.75–0.85, yielding the highest harmonic mean of acceptance and success.

## D.7 PSE Prediction Quality Evaluation

We evaluate PSE’s predicted future layouts on held-out trajectory splits from all three training corpora. Each predicted layout $\hat { L } _ { t + 1 }$ is compared with the ground-truth layout $L _ { t + 1 }$ using one-to-one greedy IoU matching at a threshold of 0.5.

Table 4 reports three metrics: (i) Type Accuracy—the fraction of matched element pairs whose predicted type label is correct; (ii) Bbox IoU—the mean IoU of matched bounding boxes; and (iii) Element-level F1—the harmonic mean of element-level precision and recall under IoU≥ 0.5 matching. The Overall row is computed by pooling all predictions and references from the three test splits before calculating each metric; it is therefore not the arithmetic mean of the three dataset-level percentages. Under this pooled evaluation, PSE obtains 89.4% type accuracy, 80.2% mean bbox IoU, and 83.2% element-level F1. Results on AndroidControl and GUI-Odyssey further show that the predictor transfers across held-out trajectories from multiple GUI corpora; we do not interpret these trajectory-level splits as evidence of app-disjoint generalization.

Table 4: PSE layout prediction quality on held-out trajectory splits. Overall metrics are computed after pooling predictions and references across the three splits.
<table><tr><td>Dataset</td><td>Type Acc. (%) Bbox IoU (%) Element F1 (%)</td><td></td><td></td></tr><tr><td>InterfereBench</td><td>91.3</td><td>82.6</td><td>85.4</td></tr><tr><td>AndroidControl</td><td>88.7</td><td>79.1</td><td>82.3</td></tr><tr><td>GUI-Odyssey</td><td>87.2</td><td>77.8</td><td>80.9</td></tr><tr><td>Overall</td><td>89.4</td><td>80.2</td><td>83.2</td></tr></table>

Boundary Case Analysis of $\mathbf { E q } .$ (3). The dissimilarity heuristic in Eq. (3) assumes that larger layout changes correlate with successful actions. Two edge-case categories violate this assumption: (A) Success with low ∆layout—actions like confirming a dialog or dismissing a toast produce correct outcomes yet minimal layout change (8.6% of test transitions); (B) Failure with high ∆layout—accidental taps that trigger page jumps induce large layout shifts despite being incorrect (5.2%). When relying solely on Eq. (3), the misjudge rates for these two categories are 72.4% and 68.1%, respectively. The anomaly-aware weight first filters predictions flagged as severe; PEC’s post-execution semantic verification then catches residual errors. With both safeguards, the residual system-level decision errors drop to 18.3% and 12.7%. Because the latter verification occurs after execution, these final rates characterize the complete controller rather than PSE prediction alone.

Desktop/Web Benchmark: OSWorld. OSWorld (Xie et al. 2024) contains 369 tasks involving real web and desktop applications, OS file I/O, and cross-application workflows. We follow the screenshot-only protocol and the 15- /50-step action budgets used by UI-TARS (Qin et al. 2025). Baseline scores in Table 5 are reported under this protocol; PrecogUI reaches 20.7% and 27.5% SR under the 15- and 50-step budgets, respectively. It exceeds UI-TARS-7B-DPO by 2.0 points at 15 steps and UI-TARS-72B-DPO by 2.9 points at 50 steps, while UI-TARS-72B-DPO remains 2.0 points higher under the 15-step budget.

![](images/e8f0d4415c8c271531b5e2b9d8da501967d5f4842e6e6f32ff2d8e0c1ac6b7e5.jpg)  
Figure 12: Action commands use three coordinate formats with a default preference of Index → Relative-in-Box → Absolute. PSE may override this preference when another action-format pair has higher predicted reliability and is not flagged as high risk.

![](images/522d8a29b4444807bd3f307c7c2094a5d7595f591bad014c6fc563cb1a7fe4b4.jpg)  
Figure 13: Anomaly Handling Template. The figure illustrates the structured anomaly-handling workflow, including detection rules for pop-ups, blank screens, and freezes (left), and an in-context example of dismissing a pop-up overlay (right).

![](images/37be955e136ff402c446073966103fc92164c0deb29e1e7ea2625d92783b0409.jpg)  
Figure 14: Role and context template. Specifies agent responsibilities and I/O context with indexed screenshots, prior execution results, and step numbers to guide evidence-based task completion.

![](images/8685b10278ed63dd40e8f9288f6459b6cc3524f5ac47c6f3c2e3d247d81c37d9.jpg)  
Figure 15: OS-specific action hints. Encodes Android conventions for app access, navigation keys, keyboard input, in-game flows, and app termination to ensure robust, context-aware execution.

Table 5: Success rate (%) on OSWorld under the UI-TARS screenshot-only protocol.
<table><tr><td>Method</td><td>15-step</td><td>50-step</td></tr><tr><td>GPT-4o (OpenAI et al. 2024)</td><td>5.0</td><td></td></tr><tr><td>CogAgent-9B (Hong et al. 2024)</td><td>8.1</td><td></td></tr><tr><td>OS-Atlas-7B (Wu et al. 2025)</td><td>14.6</td><td></td></tr><tr><td>Aguvis-72B (Xu et al. 2025b)</td><td>17.0</td><td></td></tr><tr><td>UI-TARS-7B-SFT (Qin et al. 2025)</td><td>17.7</td><td></td></tr><tr><td>UI-TARS-7B-DPO (Qin et al. 2025)</td><td>18.7</td><td></td></tr><tr><td>UI-TARS-72B-SFT (Qin et al. 2025)</td><td>18.8</td><td></td></tr><tr><td>UI-TARS-72B-DPO (Qin et al. 2025)</td><td>22.7</td><td>24.6</td></tr><tr><td>PrecogUI</td><td>20.7</td><td>27.5</td></tr></table>

## E Prompts in Automated Pipeline

## E.1 Output Format Structure Template

As illustrated in Figure 11, our Deep Think & Decision mechanism is governed by a mandated JSON schema that structures the agent’s output. This schema enforces a rigorous, multi-stage reasoning process through several key fields: Historical status for visual verification of the previous action’s outcome, severing reliance on potentially noisy execution logs; import contents for grounding the agent’s awareness in the current UI context; think for articulating a step-by-step causal rationale; progress and next goal for explicit task decomposition and forward planning; and finally action, which specifies the precise, parameterized command for environmental actuation (e.g., via index-based coordinates). Crucially, the schema’s emphasis on populating fields like

![](images/64b8e72331638cf0b138a9dab2f73c4dd3d6d17757a3416ed2b69742644b0015.jpg)  
Figure 16: General instruction template. Defines structured reasoning, precise targeting, verification, controlled waiting, and disciplined termination to ensure robust, evidence-driven task execution.

![](images/65f148ec2ddd3d13f3d00310f335e0e80198cd02b4c7b802daa6e916af1e7103.jpg)  
Figure 17: Apps pop-up handling. A type-aware policy combined with hierarchical position selection (Index → Relative-in-Box → Absolute); the figure presents concrete dismissal commands for diverse pop-up cases.

Historical status based solely on visual evidence establishes a tight closed-loop verification system. This structured output thereby functions as a transparent and auditable interface between the agent’s cognitive deliberation and its concrete actions within the GUI environment.

## E.2 Action Selection Template

To ensure robust action grounding, we define three coordinate formats. (1) Highlight Index targets an element through a semantic identifier and is used as the default when the index is stable. (2) Relative-in-Box specifies a sub-point within an indexed element and is useful when the desired target is only part of a larger control. (3) Absolute Coordinates provide a normalized fallback when semantic indexing is unavailable. The order Index → Relative-in-Box → Absolute is a default preference, not a hard constraint. At each step, PEC evaluates the available action-format pairs with PSE, removes candidates whose predicted risk exceeds the safety threshold, and executes the remaining pair with the highest relative reliability score. Consequently, Relativein-Box or Absolute may be selected ahead of Index when the current layout makes the default format unreliable.

![](images/56e8e97bd6dadca92945510c72b403e28392be1d483701e4b0287aa8ec1e36ad.jpg)  
Figure 18: System pop-up handling. A type-aware policy selects safe dismissal actions and executes them with index-prioritized targeting; the figure shows concrete commands for notifications, risk alerts, and permission requests.

## E.3 Anomaly Handling Template

As shown in Figure 13, we frame anomaly handling as a concise, cross-task routine over prediction and verification. Given the current layout $L _ { t }$ and forecast $\hat { L } _ { t + 1 }$ , fast rules classify the risk, after which PEC selects a mitigation: (i) Pop-up/Overlay—dismiss via safe affordances (X/Close/Cancel/Later); (ii) Blank Screen—wait briefly, then Back/refresh or return to a stable hub; (iii) Freeze/Ineffective Action—retry once with safer targeting, else Back/re-plan; (iv) Off-Goal/Misdirection—abort the path and restore on-goal context. A compulsory re-check gates progress after mitigation.

## E.4 Role and Context Template

To structure the agent’s operational context, we define a clear set of responsibilities and a standardized input format for each reasoning step. As illustrated in Figure 14, the agent is prompted with persona as an expert GUI automation agent. For each step, it receives a tripartite input: (1) the current screenshot augmented with indexed, highlighted bounding boxes over interactable elements; (2) feedback on the execution status (e.g., success or failure) of the prior action; and (3) the current temporal step index. Crucially, the agent is explicitly instructed to ground its reasoning solely on visual evidence, judging task progression based on observable changes in the UI state rather than uncritically accepting the programmatic execution status. This mandate establishes a tight, closed-loop visual verification process for all decisionmaking.

## E.5 OS-Specific Hints

As shown in Figure 15, we encode platform conventions into structured hints that guide robust action execution on Android. These rules address common UI operations and context-sensitive behaviors: (i) app launching via centered icon clicks with optional open app flag; (ii) special system keys such as home, back, and recent for navigation control; (iii) text input handling by directly invoking input text when the ADB keyboard is active, avoiding redundant position specifications; (iv) game-specific flows where the back key may be ineffective, requiring strict adherence to in-game command order; and (v) app termination through swipe-off gestures in the recent-apps screen. Collectively, these hints ground agent actions in OS-level semantics, reducing execution ambiguity and improving crosscontext stability.

![](images/108cc8f7bc2d8635649e86702c44b15a8b7f3f21de107922cf91481f75e2e4ab.jpg)  
Figure 19: Environment disturbances. A lightweight routine—short waits plus index prioritized safe retries stabilizes black/white screens, delayed loads, and network stalls; the figure shows concrete wait and retry commands for representative cases.

## E.6 General Instructions

As shown in Figure 16, this template encodes cross-task discipline for structured reasoning and verifiable execution. It emphasizes (i) step-by-step task decomposition into checkable sub-steps; (ii) precise targeting using highlighted regions or indices while avoiding ambiguous clicks; (iii) progress verification strictly by on-screen evidence such as titles, messages, or control states; (iv) controlled waiting to accommodate delays or animations; (v) fallback to anomaly-handling rules when overlays appear; and (vi) termination only after explicit visual confirmation of success. When targeting remains uncertain, the agent is required to re-locate or choose safer alternatives, ensuring robustness against cascading errors. Collectively, these rules establish a disciplined action loop where correctness validation precedes task advancement.

## F Qualitative Analysis

## F.1 Apps Pop-up Handling

As shown in Figure 17, we deploy a type-aware policy that closes in-app pop-ups while preserving task context. The controller first classifies the pop-up—(i) announcement/notice panels, (ii) gift-package ads, (iii) event promotions, or (iv) confirmation/input dialogs—and selects the safest dismiss affordance. Execution follows our hierarchical position schema: prioritize element indices for X/Close/Cancel/Later; degrade to Relative-in-Box when the target is a sub-control; and use normalized absolute coordinates only when indexing is unreliable. Each thumbnail shows the predicted command (index or relative point) rendered beneath the image; progress continues only after the overlay is visually cleared.

## F.2 System-Level Pop-up Handling

As shown in Figure 18, we handle OS-mediated interruptions—system notifications, risk alerts, and permission requests—via a type-conditioned, safety-first policy. The controller classifies the pop-up and selects the safest affordance (e.g., Cancel/Close, Allow only while in use, Deny). Execution uses our hierarchical position scheme, prioritizing element indices and backing off to Relative-in-Box or normalized Absolute coordinates only when indexing is unreliable. Each panel displays the issued command (primarily index clicks), and progress resumes only after the overlay is visually cleared to preserve task context.

![](images/aea938d638b371a6367cbda28e433850d003db8f158785cd501ba073365ef00c.jpg)  
Figure 20: Layout-shift handling. PrecogUI rebuilds layout hashes and re-indexes targets under portrait/landscape transitions, executing with index-first targeting; the figure shows before/after screens with preserved action intent.

## F.3 Environment Perturbation Handling

As shown in Figure 19, we address environment-level disturbances (black/white screens, loading delays, and network stalls) with a lightweight stabilization routine. Detection relies on low-saliency/blank frames, near-identical consecutive layouts, or stalled progress indicators. Mitigation is minimal yet effective: inject a short wait (e.g., 200 ms) to absorb transient transitions, then issue a single index-prioritized safe retry of the previous action; progress resumes only after visual evidence of recovery, otherwise control is escalated to the general anomaly rules.

Layout-Shift Perturbations. As shown in Figure 20, we address orientation/gravity–induced reflows (portrait ↔ landscape) with an orientation-aware re-localization routine. Upon detecting a layout shift (aspect-ratio change and index invalidation), the agent reconstructs the symbolic layout hash, re-indexes targets, and remaps the current goal to the new arrangement by type/text cues. Execution then follows the hierarchical position policy (Index → Relative-in-Box → Absolute), and progress is gated by visual re-check to ensure the intended control is active after rotation.

## G PEC Algorithm Pseudocode

Algorithm 1 summarizes the inference-time control flow of PEC and complements the module description in Sec. 2.5. PEC first handles anomalies already visible in the current state by retrieving a remedy from PEP or invoking the structured anomaly-diagnosis policy. Otherwise, it generates candidate action-format pairs, filters those predicted to be highrisk by PSE, and executes the highest-scoring safe pair. Each outcome is then verified using visual and semantic feedback; failed pairs are temporarily blocked, while unexpected transitions trigger recovery or episode restart before candidate generation resumes.

## H Additional Discussions

Forecasting future layouts is central to PrecogUI: lookahead turns reactive ”observe–act” behavior into risk-aware planning that preempts pop-ups, freezes, and off-goal drifts, improving long-horizon stability. However, timeliness is a key constraint. Pre-execution simulation and verification add latency and compute, which can be costly for real-time use or very long tasks. In addition, experience priors can become stale as apps update; outdated remedies hurt reliability unless memory is refreshed. Future work should adopt lightweight, anytime forecasting and drift-aware memory maintenance to preserve the gains of look-ahead without sacrificing responsiveness.

Algorithm 1 Pre-cognitive Execution Controller   
Require: Goal g, current state $s _ { t } ,$ anomaly memory $M _ { a } ,$   
anomaly rules $\mathcal { R }$ anom   
Ensure: A verified action-format pair $( a , r )$ or an anomaly  
handling action   
$/ / - S t a g e \ : I ; i$ Pre-cognitive Anomaly Checks –   
1: $\ell _ { t } \gets \mathrm { L a y o u t } ( s _ { t } )$   
2: i $\mathbf { f } M _ { a } . \mathbf { Q } \mathbf { \bar { u e r y } } \mathbf { \vec { B } } \mathbf { y } \mathbf { \bar { L } }$ ayout $( \ell _ { t } )$ returns a remedy $a _ { \mathrm { h a n d l e } }$ then   
3: return a<sub>handle</sub>   
4: end if   
5: if CurrentStateAnomaly $\prime ( s _ { t } ; \mathcal { R } _ { \mathrm { a n o m } } )$ then ▷   
Current-state check, not action forecasting   
6: return $\pi _ { \mathrm { L L M } } ( s _ { t } , \mathcal { T } _ { \mathrm { a n o m a l y } } )$ ▷ Autonomous diagnosis   
for novel anomaly   
7: end if   
// – Stage $2 { : }$ Iterative Execution and Recovery Loop –   
8: $\mathcal { C } _ { t } \gets \dot { \bf M } { \bf L } { \bf L } { \bf M }$ .GenerateCandidatePairs $( s _ { t } , g )$   
9: $\mathcal { F } _ { t } \gets \emptyset$ ▷ Initialize temporary taboo list   
10: while $\mathcal C _ { t } \setminus \mathcal F _ { t }$ is not empty do   
11: $\begin{array} { r l r l r l r l r l } { \mathcal { C } _ { t } ^ { \mathrm { s a f e } } } & { { } } & { \langle - } & { } & { \{ ( \bar { a } , \bar { r } ) } & { { } } & { \in } & { } & { \mathcal { C } _ { t } } & { { } \backslash } & { \mathcal { F } _ { t } } \end{array}$   
w(PSE.ForecastLayout $( s _ { t } , a , r ) ) > 0 \}$   
12: if $\mathcal { C } _ { t } ^ { \mathrm { s a f e } }$ is empty then   
13: return $\pi _ { \mathrm { L L M } } ( s _ { t } , \mathcal { T } _ { \mathrm { a n o m a l y } } )$   
14: end if   
15: $( a ^ { * } , r ^ { * } ) \gets \arg \operatorname* { m a x } _ { ( a , r ) \in \mathcal { C } _ { t } ^ { \mathrm { s a f e } } } s ( a , r )$   
16: execute $( \boldsymbol { a } ^ { * } , \boldsymbol { r } ^ { * } )$ ; observe new state $s _ { t + 1 }$   
17: if VerifySuccess $( s _ { t } , ( a ^ { * } , r ^ { * } ) , s _ { t + 1 } )$ then   
18: return $( a ^ { * } , r ^ { * } )$ ▷ Success: terminate step   
19: end if   
20: ${ \mathcal { F } } _ { t } \gets { \mathcal { F } } _ { t } \cup \{ ( a ^ { * } , r ^ { * } ) \}$   
21: if $\dot { \mathcal { D } } _ { \mathrm { l a y o u t } } ( \mathrm { L a y o u t } ( s _ { t } ) \dot { , }$ Layou $( s _ { t + 1 } ) ) < \tau _ { \mathrm { c } }$ then ▷   
Stagnation   
22: continue   
23: else ▷ Unexpected Transition   
24: s<sub>t</sub> ← RECOVERORRESTART   
25: $\mathcal { C } _ { t } \gets \mathbf { M L L M }$ .GenerateCandidatePairs $( s _ { t } , g )$   
26: $\mathcal { F } _ { t } \gets \emptyset \ \triangleright$ Discard taboos tied to the failed state   
27: end if   
28: end while   
29: return $\pi _ { \mathrm { L L M } } ( s _ { t } , \mathcal { T } _ { \mathrm { a n o m a l y } } )$ ▷ Final diagnosis if all   
candidates fail

## H.1 Cross-Device Grounding and Mobile Navigation Analysis

PrecogUI represents UI elements with normalized coordinates and structured (type,bbox) layouts, reducing its dependence on any single screen resolution. The Desktop and Web subsets of ScreenSpot provide direct evidence for cross-form-factor grounding: as reported in Table 3, PrecogUI obtains 97.5% text and 82.2% icon accuracy on Desktop, and 94.6% text and 91.7% icon accuracy on Web. These results support transfer of the grounding component across mobile, desktop, and web screenshots, but they do not by themselves establish end-to-end task completion on desktop or web environments.

AndroidControl-High and GUI-Odyssey evaluate end-toend navigation in mobile application environments. PrecogUI achieves 76.4% SR on AndroidControl-High and 89.1% on GUI-Odyssey, as shown in Table 2. OSWorld provides complementary end-to-end evidence on desktop and web applications under a screenshot-only protocol, where PrecogUI obtains 20.7% and 27.5% SR with 15- and 50-step budgets (Table 5). We therefore interpret the evidence conservatively: ScreenSpot measures cross-form-factor grounding, AndroidControl and GUI-Odyssey measure mobile navigation, and OSWorld supplies an initial desktop/web endto-end evaluation. Broader testing with additional interactive desktop and web protocols remains future work.

## H.2 Rollback Scope and Empirical Statistics

Scope and Feasibility. The recovery branch in Sec. 2.5 and Algorithm 1 is restricted to reversible evaluation tasks. We distinguish three operations that were previously grouped under the term rollback. First, navigation recovery uses reversible UI actions such as Back, Close, or Home and then visually verifies whether the last stable task context has been recovered. Second, episode restart reinitializes a benchmark task from its known starting state when navigation recovery fails. Third, checkpoint restoration is used only in emulator environments that expose an explicit state-checkpoint API. We do not claim that an arbitrary physical-device application state can be restored after an external side effect.

Snapshot Storage and Restoration. Before executing a candidate, PEC stores an observation record containing the last confirmed screenshot, its parsed layout, the selected action-format pair, and the verification result. This record supports diagnosis and re-planning but is not itself a restorable application state. When an emulator checkpoint is available, the environment state is restored through the benchmark interface. Otherwise, PEC attempts navigation recovery and falls back to an episode restart if the stable context cannot be re-established. After any successful recovery, PEC regenerates candidate action-format pairs from the recovered state rather than reusing predictions made for the failed state.

Empirical Usage Statistics. To quantify the practical impact of recovery, we report usage statistics across three longhorizon evaluation splits:

As shown in Table 6, recovery is triggered in 12.4– 18.6% of all evaluated episodes. Among those episodes, 69.7–73.1% resume successfully after navigation recovery or checkpoint restoration. Episode restarts account for 1.8– 2.7% of all evaluated episodes. These statistics quantify recovery within our controlled evaluation protocol; they do not imply that irreversible external side effects can be undone.

Handling Irreversible Operations. When PEC repeatedly fails from the same stable state, it first retries with alternative safe action-format pairs and then invokes navigation recovery. If the task context cannot be recovered, the evaluation episode is restarted from its initial state. The current PSE detects visual and transition anomalies; it is not a semantic authorization mechanism for transactions or other irreversible actions. Accordingly, our autonomous evaluation excludes operations that can create real-world side effects, such as purchases or message transmission. A deployed system should require explicit user confirmation and application-specific safeguards for such actions. End-to-end protection for irreversible operations remains outside the present scope.

Table 6: Recovery usage statistics across long-horizon evaluation splits. “Episodes w/ recovery” and “Episodes requiring restart” are percentages of all evaluated episodes; “Success after recovery” is computed only within episodes where recovery was triggered.
<table><tr><td>Dataset</td><td>Episodes w/ recovery</td><td>Success after recovery</td><td>Episodes requiring restart</td></tr><tr><td>AndroidControl-High</td><td>12.4%</td><td>69.7%</td><td>1.8%</td></tr><tr><td>GUI-Odyssey</td><td>15.3%</td><td>73.1%</td><td>2.4%</td></tr><tr><td>InterfereBench</td><td>18.6%</td><td>71.2%</td><td>2.7%</td></tr></table>