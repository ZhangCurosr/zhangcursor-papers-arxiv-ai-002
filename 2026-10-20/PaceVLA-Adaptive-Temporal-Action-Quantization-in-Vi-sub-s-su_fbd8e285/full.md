![](images/6365e544a140cde387d47e3e3a639fbf4451d270f6d10cf20eb393f0c5bef52c.jpg)

# PaceVLA: Adaptive Temporal Action Quantization in Vi<sub>s</sub>i<sub>on-</sub>L<sub>anguage-</sub>A<sub>c</sub>ti<sub>on</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

Hojin Jeon<sup>1,\*</sup>, Jimin Lee<sup>1,\*</sup>, Taekwan Kim<sup>3</sup>, Nayoung Oh<sup>1</sup>, Taeyoung Kim<sup>1,2</sup>, Myungkyu Koo<sup>1,2</sup>, Ji<sub>nmo</sub> K<sub>oo</sub><sup>4</sup><sub>,</sub> J<sub>oonwoo</sub> Ah<sub>n</sub><sup>2</sup> <sub>an</sub>d Ji<sub>nwoo</sub> Shi<sub>n</sub><sup>1,2</sup>

<sup>1</sup>KAIST, <sup>2</sup>RLWRLD, <sup>3</sup>Korea University, <sup>4</sup>Seoul National University

Fast and reliable manipulation is crucial for practical robot deployment. Although Vision-Language-Action (VLA) models have recently shown remarkable progress as generalist robot policies, their execution often remains temporally ineficient. A key source of this ineficiency is that policies learn fine-grained temporal patterns from teleoperated demonstrations, requiring many low-level action steps. Temporal action quantization ofers a simple way to accelerate execution by aggregating consecutive low-level actions into temporally coarser actions. However, applying action quantization without considering when fine-grained control is required introduces a speed–accuracy trade-of: accelerated actions may eficiently reduce redundant steps in some phases, while discarding critical motion details in precision-sensitive interactions. To address this, we propose PaceVLA, a unified framework for adaptive temporal execution in VLAs that integrates two coupled components: (1) mode-specific action decoders built on a shared action expert, supporting multiple temporal granularities and prediction horizons, and (2) a quantizability-guided mode selection scheme which decides whether coarse execution is appropriate, and a context-conditioned router that selects an execution mode at each inference step. Experiments across three multi-task simulation benchmarks and six real-world tasks show that PaceVLA improves execution eficiency while maintaining task success rate across multiple embodiments, including long-horizon and dexterous manipulation. The project page is available.

(a) Accelerated execution on the real-robot conveyor task.  
![](images/44bbf1ac197d04c5b7f5e978885e182ac612235ad14e9498a2b1fefc148f8e70.jpg)  
(b) Speed–accuracy trade-of.  
Figure 1: A<sup>d</sup>aptive tempora<sup>l</sup> execution wit<sup>h</sup> PaceVLA. (a) Conveyor package handling task on the humanoid ALLEX: PaceVLA selects a temporal granularity per phase. (<sup>b</sup>) Success rate vs. speedup over the base VLA, averaged over six real-world tasks.

## 1<sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Vision-Language-Action models (VLAs) (Zitkovich et al., 2023; Kim et al., 2024; Intelligence et al., 2025; Pertsch et al., 2025; Bjorck et al., 2025) have emerged as a promising framework for generalist robot policies, achieving strong performance through imitation learning from diverse robot demonstrations (Walke et al., 2023; O’Neill et al., 2024; Khazatsky et al., 2024), many of which are collected through human teleoperation (Zhao et al., 2023; Wu et al., 2024). However, teleoperated demonstrations often exhibit conservative behavior: human-in-the-loop control is typically performed through limited-rate interfaces, and human operators adopt cautious motions to avoid failures during data collection (Mandlekar et al., 2018, 2021). Since imitation learning trains policies to reproduce the demonstrated trajectories, VLAs inherit these patterns and execute tasks through many slow action steps even in phases where faster motion is physically feasible.

Recent works have explored accelerating robot policies beyond demonstration speed, addressing two complementary questions: where to accelerate and how coarsely to execute actions. DemoSpeedup (Guo et al., 2025) segments per-frame action entropy with a proxy policy and trains a policy on the resulting accelerated trajectories. SAIL (Arachchige et al., 2025) combines high-fidelity tracking with switching between preset slow and fast speeds based on motion complexity and gripper events. A particularly simple mechanism is temporal action quantization (Figure AI, 2025; Tang et al., 2025), which merges consecutive actions into coarser ones. It fits naturally with chunk-based VLAs since each predicted action chunk can be executed at a chosen granularity.

Despite their promise, these approaches have two limitations. First, the acceleration decision is either fixed into the training data, using the action entropy of a fine-tuned proxy policy as an implicit indicator (Guo et al., 2025), or applied at execution time to the output of a policy trained at demonstration speed, switching between preset speeds according to kinematic cues (Arachchige et al., 2025). Both signals are statistics of the actions, yet how much precision a phase demands is determined by the robot’s interaction with the scene rather than by the motion itself. Second, existing temporal action quantization applies a fixed granularity across the entire trajectory, inducing a speed–accuracy trade-of: coarse actions sufice during free-space motion but discard trajectory details required in precision-critical phases such as grasping and contact-rich interaction. Moreover, granularity is closely coupled with the prediction horizon: phases that tolerate coarse execution can commit to longer predictions, whereas precision-critical phases benefit from shorter horizons with more frequent re-planning. We argue that both the temporal granularity and the prediction horizon of action execution should be treated as a policy decision, adapted to the precision each phase requires.

To this end, we propose PaceVLA, a unified framework for adaptive temporal action execution in VLAs. PaceVLA equips a shared action expert with mode-specific action decoders that support diferent temporal granularities and prediction horizons. At each inference step, a quantization judge and a router jointly select an execution mode from the visual and language context. Each execution mode is defined by a temporal action quantization level and a prediction horizon, allowing the policy to choose both the granularity of its commands and the temporal extent of its action predictions. To validate the efectiveness of PaceVLA, we conduct extensive experiments across Robocasa-Kitchen (Nasiriany et al., 2024), LIBERO (Liu et al., 2023), and DexJoCo (Wang et al., 2026) simulated robotic benchmarks and six real-world manipulation tasks including long-horizon, dexterous tasks covered on three robot platforms: Franka Research arm with gripper, OpenArm robot arm with two 6-DoF dexterous hands, and humanoid ALLEX robot with two 15-DoF dexterous hands. Compared with the baselines, PaceVLA achieves up to 1.8× faster than the base VLA, and 1.75× throughput on the conveyor package-handling task with the humanoid ALLEX, while maintaining task success rates.

Our contributions are:

• We highlight the tendency of VLA policies to inherit fine-grained execution patterns from teleoperated demonstrations as a key source of temporal ineficiency.

• We propose PaceVLA, a lightweight architecture that models execution modes with diferent temporal quantization levels and action horizons using mode-specific action decoders, a quantization judge and a context-conditioned router.

• We demonstrate consistent gains on simulation benchmarks and real-world tasks, achieving a stronger balance between task success and execution eficiency than the base VLA and fixed temporal quantization baselines.

## 2<sub>.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Vision-Language-Action mo<sup>d</sup>e<sup>l</sup>s (VLAs). VLAs have recently shown great promise in robotic policy learning by leveraging large-scale robot datasets. One line of VLA models (Kim et al., 2024; Pertsch et al., 2025; Driess et al., 2023) discretizes the continuous action space and predicts actions as tokens, showing that pre-trained VLMs (Beyer et al., 2024; Chen et al., 2026; Bai et al., 2025; Chen et al., 2023) can be efectively adapted to low-level action prediction. In contrast, a growing body of studies (Intelligence et al., 2025; Li et al., 2024) attaches action experts that generate continuous action chunks via difusion or flow-matching, enabling more accurate predictions. For example, GR00T N1.5 (Bjorck et al., 2025) and RLDX-1 (Kim et al., 2026) condition a flow-matching head on VLM features to produce action chunks. In this work, we build on a flow-matching action expert for action generation. While most difusion-based VLAs predict action chunks at a single, fixed temporal granularity set by the data collection frequency, this single granularity is not equally suited to all phases of a task. Motivated by this observation, we explore whether jointly learning actions at multiple temporal granularities and adapting the action granularity to the current context can lead to faster and more accurate task execution.

Faster-t<sup>h</sup>an-Demonstration Execution. The execution speed of imitation-learned policies is bounded by the pace of their demonstrations, and several lines of work have aimed to exceed it. Reinforcement-learning-based fine-tuning adjusts the policy using task-level rewards to encourage faster and more decisive behaviors (Yuan et al., 2025), but tends to push the policy toward uniformly faster execution. Another direction modifies the temporal structure of actions, as explored in DemoSpeedup (Guo et al., 2025) and SAIL (Arachchige et al., 2025). While these approaches improve execution speed, they determine the degree of acceleration either at implicit action entropy signals or through kinematic cues for switching between preset speeds, both derived from action statistics, rather than letting the policy itself decide how much to accelerate from the scene it perceives at inference time.

Router-Base<sup>d</sup> Ex ert Se<sup>l</sup>ection. Conditional computation employs a lightweight router that selects which sub-network to apply to each input, and has been widely adopted to scale capacity without proportional cost. In most designs the candidates are structurally identical and difer only in learned weights, whether FFN experts (Fedus et al., 2022) or attention heads (Jin et al., 2024), so a routing decision carries no predefined meaning. Within VLAs, MoLe-VLA (Zhang et al., 2026) routes over backbone layers to reduce the cost of each forward pass, leaving the action output unchanged. Mixture of Horizons (Jing et al., 2025) is the closest to our setting, as it constructs action heads at several chunk lengths, but it fuses their outputs through a gate, so each action is still emitted at a single fixed temporal granularity. None of these, however, treat the temporal granularity of the action as the routing target, where a router selects one decoder per state from a set of decoders that difer in temporal resolution and horizon.

## 3<sub>.</sub> M<sub>e</sub>th<sub>o</sub>d

In this section, we introduce PaceVL $\mathbf { A } ,$ a unified framework for accelerating robot policy execution through adaptive temporal action quantization. In a nutshell, PaceVLA leverages a shared action expert with multiple action decoders that support diferent temporal granularities and prediction horizons. At each inference step, a learned quantization judge determines whether to use fine-grained or quantized decoding, and a router selects a mode within the chosen group. We also describe the system-level controller configuration used to track the resulting action commands. The overall framework is illustrated in Figure 2.

Formally, given a language instruction $^ { I , }$ visual observations $V _ { t } ,$ , and a proprioceptive robot state $\mathbf { } _ { s _ { t } , }$ let $\Phi _ { t } = f _ { \theta } ( I , V _ { t } , \pmb { s } _ { t } )$ denote the conditioning context for predicting an action chunk $\pmb { A } _ { t } = ( \pmb { a } _ { t , 1 } , \dots , \pmb { a } _ { t , H } )$ with action horizon $H .$ . We build on a pretrained difusion-based VLA (Bjorck et al., 2025), whose action expert uses a difusion transformer (DiT) backbone (Peebles and Xie, 2023) trained with flow matching (Lipman et al., 2022). Given a clean chunk $\pmb { A } _ { t }$ , a sampled flow time �, and Gaussian noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , I )$ , we construct the noisy chunk $A _ { t } ^ { \tau } = \tau A _ { t } + ( 1 - \tau ) \epsilon$ . The velocity network ${ \mathbf { } } { \mathbf { } } \mathbf { } u _ { \theta } ( \Phi _ { t } , A _ { t } ^ { \tau } , \tau )$ is trained to predict $A _ { t } - \epsilon \mathrm { : }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { A _ { t } , \tau , \epsilon } \left[ \left\| u _ { \theta } ( \Phi _ { t } , A _ { t } ^ { \tau } , \tau ) - ( A _ { t } - \epsilon ) \right\| ^ { 2 } \right] . } \end{array}\tag{1}
$$

We extend this formulation to multiple temporal action modes using mode-specific output decoders with a shared action expert.

## 3.1. Temporal Action Quantization

Temporal action quantization represents a sequence of demonstration actions with fewer action commands by aggregating adjacent actions. We define $K$ action modes. For each mode $k \in \{ 1 , \ldots , K \}$ , we convert $H _ { k }$ demonstration steps into $T _ { k } \ \leq \ H _ { k }$ commands, with a temporal quantization ratio $q _ { k } = H _ { k } / T _ { k }$ . Here, $H _ { k }$ denotes the source horizon and $T _ { k }$ is the number of predicted commands. Note that action quantization refers to aggregating action values along time. We construct fine-grained modes with $q _ { k } = 1$ and quantized modes with $q _ { k } > 1$ . Both fine-grained and quantized modes further include shorter and longer source horizons to provide temporal variety.

We partition each source segment into $T _ { k }$ contiguous blocks and aggregate the actions within each block. In detail, action quantization depends on the action representation. For absolute actions, each block retains its final command. For delta-pose actions, we sum translations expressed in a fixed Cartesian frame, compose rotational increments, and retain the final gripper command. The resulting target is denoted by $A _ { t } ^ { ( k ) } \in \mathbb { R } ^ { T _ { k } \times d _ { a } }$ where $d _ { a }$ is the action dimension. Detailed aggregation rules are provided in Appendix A.

![](images/5650e40ae41e47a5b4ecc79d92a08ead8a50f3879070497c12210c4f20c42bdd.jpg)  
Figure 2: Overview o<sup>f</sup> PaceVLA. PaceVLA is a unified framework for adaptive temporal action execution in pretrained VLAs. It equips a shared action expert with mode-specific action decoders that support diferent temporal granularities and prediction horizons.

## 3<sub>.</sub>2<sub>.</sub> Sh<sub>are</sub>d A<sub>c</sub>ti<sub>on</sub> E<sub>xper</sub>t <sub>w</sub>ith M<sub>o</sub>d<sub>e-</sub>S<sub>pec</sub>ifi<sub>c</sub> A<sub>c</sub>ti<sub>on</sub> D<sub>eco</sub>d<sub>ers</sub>

We implement these temporal action modes using a shared flow-matching action expert with mode-specific output decoders. The action encoder and DiT backbone are shared across action modes, while each mode � has a lightweight MLP decoder $D _ { k }$ . For mode $k ,$ the shared encoder and DiT backbone process a noisy action sequence of length $T _ { k }$ . Decoder $D _ { k }$ maps the resulting features to a flow velocity prediction of shape $T _ { k } \times d _ { a }$ where $d _ { a }$ is the action dimension. This design allows each mode to learn its corresponding temporal granularity without requiring global temporal resampling of the dataset. During training, we compute the loss for all � modes using their respective targets and sequence lengths. At inference, action generation uses the sequence length $T _ { k }$ and decoder $D _ { k }$ associated with the selected action mode.

## 3.3. Quantizability-Guided Mode Selection

We select an action mode using two signals. The quantization judge first explicitly selects whether to use fine-grained or quantized action, and the router then selects a mode within the chosen group.

Group se<sup>l</sup>ection an<sup>d</sup> mo<sup>d</sup>e routing. We partition the modes into a fine-grained group, $K _ { \mathrm { f i n e } } = \{ k : q _ { k } = 1 \}$ , and a quantized group, $\mathcal { K } _ { \mathrm { q u a n t } } = \{ k : q _ { k } > 1 \}$ . The predicted quantizability score determines the action group. A lightweight MLP router produces a set of probabilities $\pmb { \alpha } _ { t } = ( \alpha _ { t , 1 } , \ldots , \alpha _ { t , K } ) ^ { \top }$ over the modes, and these select the mode within the eligible group.

Quantiza<sup>b</sup>i<sup>l</sup>ity supervision. To explicitly determine whether acceleration is appropriate, we supervise the quantization judge with annotations using task execution outcomes and a large vision-language model (LVLM). We first examine task behavior under several fixed action quantization settings applied to each task. We use these observations as task-level cues to guide the identification of phase-level patterns of quantizability within the demonstrations. LVLM then refines these patterns into criteria using demonstration samples and applies them to each time step of the original demonstrations, producing a per-step quantizability score $y \in [ 0 , 1 ]$ . These scores supervise the judge through Eq. (3), while mode-specific prediction errors supervise the router through Eq. (2). The details regarding the creation and actual annotation criteria are provided in the appendix C.5.

## 3.4. Training Objectives

We jointly optimize action prediction and routing, with an auxiliary regression objective for the quantization judge. For mode $k ,$ let $\ell _ { k }$ denote the per-sample prediction error in Eq. (1), evaluated using context $\Phi _ { t } ,$ aggregated target $A _ { t } ^ { ( k ) }$ , and the velocity network ${ \pmb u } _ { \theta } ^ { ( k ) }$ formed by the shared action expert and decoder $D _ { k } .$ Each mode uses independently sampled noise and flow times. Prediction errors are masked and averaged over valid action elements, which accounts for the diferent target action lengths across modes.

Action po<sup>l</sup>icy <sup>l</sup>earning. Let $\pmb { \alpha } = ( \alpha _ { 1 } , \ldots , \alpha _ { K } ) ^ { \top } \in \mathbb { R } ^ { K }$ denote the router probabilities from current context $\Phi _ { t }$ , let $\mathbf { w } = ( w _ { 1 } , \hdots , w _ { K } ) ^ { \intercal }$ denote its training weights, and $\mathbf { r } = ( r _ { 1 } , \ldots , r _ { K } ) ^ { \overline { { \mathsf { T } } } } \in \mathbb { R } ^ { K }$ a target probability vector. The policy objective is

$$
{ \mathcal { L } } _ { \mathrm { p o l i c y } } = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } w _ { k } \ell _ { k } + \lambda _ { \operatorname* { s u p } } \operatorname { K L } ( \mathbf { r } \parallel \alpha ) \right] + \lambda _ { \mathrm { b a l } } { \mathcal { L } } _ { \mathrm { b a l } }\tag{2}
$$

where the target $r _ { k } \propto \exp ( - \ell _ { k } / \beta )$ favors modes with lower prediction errors, with temperature $\beta .$ The training weights w mix � with a uniform distribution, and $\mathcal { L } _ { \mathrm { b a l } }$ penalizes their deviation from uniform, preventing collapse onto a single decoder (Fedus et al., 2022).

Quantiza<sup>b</sup>i<sup>l</sup>ity regression. The judge $J _ { \psi }$ regresses the LVLM-generated quantizability score � at the first frame of each training chunk:

$$
\mathcal { L } _ { \mathrm { c o n f } } = \mathbb { E } \Big [ \big ( J _ { \psi } ( \mathrm { s g } ( \Phi _ { t } ) ) - y \big ) ^ { 2 } \Big ] ,\tag{3}
$$

where sg denotes stop-gradient, so that ${ \mathcal { L } } _ { \mathrm { c o n f } }$ updates only the judge parameters �. At inference, group selection uses the judge’s predicted score.

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p o l i c y } } + \lambda _ { \mathrm { c o n f } } \mathcal { L } _ { \mathrm { c o n f } } . } \end{array}\tag{4}
$$

## 3<sub>.</sub>5<sub>.</sub> C<sub>on</sub>t<sub>ro</sub>ll<sub>er</sub> C<sub>on</sub>fi<sub>gura</sub>ti<sub>on</sub> f<sub>or</sub> A<sub>cce</sub>l<sub>era</sub>t<sub>e</sub>d E<sub>xecu</sub>ti<sub>on</sub>

Temporally quantized actions require controllers to track larger commanded changes. In our experiment, insuficient tracking leads to execution failures. We adjust the tracking and gripper gains to support accelerated execution. Controller-specific settings are reported in Appendix C.3.

## 4. Ex<sub>p</sub>eriments

We validate the performance of PaceVLA and the efect of the proposed components through extensive experiments. In particular, we design our experiments to investigate the following questions:

• Does PaceVLA improve the trade-of between task success and execution eficiency on simulation benchmarks with various action representations? (Table 1, 3)

• Can PaceVLA accelerate real-world dexterous and long-horizon manipulation while maintaining task success across diferent embodiments? (Table $2 , \ 4 )$

• How does each component of $\mathrm { P A C E V L A }$ contribute to performance? (Table 5)

## 4.1. Ex<sub>p</sub>erimental Setu<sub>p</sub>s

Datasets. For simulation, we use RoboCasa-Kitchen (Nasiriany et al., 2024), LIBERO (Liu et al., 2023), and DexJoCo (Wang et al., 2026). We evaluate 24 RoboCasa-Kitchen tasks using 300 demonstrations per task, and

## Simulation Benchmarks

![](images/05c39c93f8ab0165d2ee197cd05ebfb7d8e5d1d153f879ecf8ba3a8ae8976306.jpg)  
Figure 3: Experimenta<sup>l</sup> setup. Top: Three simulation benchmarks, LIBERO, RoboCasa-Kitchen, and DexJoCo. Bottom: Real-world manipulation tasks across three embodiments, a Franka Research Arm with a gripper, an OpenArm robot with a dexterous hand, and an ALLEX humanoid. The real-world evaluation includes long-horizon and dexterous manipulation tasks.

report the results separately for pick-and-place tasks and the remaining tasks. For LIBERO, we evaluate the Spatial, Object, Goal, and Long suites. For DexJoCo, we evaluate a subset of six single-arm tasks that require contact-rich dexterous manipulation. In our setup, RoboCasa-Kitchen and LIBERO use delta actions, whereas DexJoCo uses absolute actions, allowing us to assess temporal action quantization under both representations. For real-world evaluation, we evaluate six tasks across three embodiments. On Franka arm with a gripper, we evaluate Pick-and-Place, Air Fryer, and Stack Cup. Air Fryer is a long-horizon task that requires opening the basket, placing food inside, and closing it. Stack Cup requires two consecutive stacking operations. On a bimanual OpenArm equipped with 6-DoF dexterous hands, we evaluate two bimanual tasks, Table Sweeping and Pump-and-Scrub. Table Sweeping requires sweeping a table using a dustpan and brush, and Pump-and-Scrub involves dispensing soap and wiping a plate. On the ALLEX humanoid equipped with 15-DoF dexterous hands, we evaluate long-horizon Conveyor Package Handling, in which the robot handles 10 packages in a row per trial, re-orienting each arriving package so that its shipping label faces forward. Detailed task descriptions, datasets, and robot configurations are provided in Appendices C.1 and C.2.

Imp<sup>l</sup>ementation Detai<sup>l</sup>s. We implement PaceVLA by fine-tuning pre-trained Vision-Language-Action models (VLA). Specifically, we use GR00T N1.5 (Bjorck et al., 2025) and RLDX-1-IMG (Kim et al., 2026) by following their oficial implementation. We use four action decoders, corresponding to two fine-grained and two quantized modes, each with a longer and a shorter source horizon, and the quantization judge is supervised using Large-Vision-Language-Model (LVLM)-generated annotations. Training hyperparameters, action mode configurations, annotation procedures, and controller settings are provided in Appendix C.

Base<sup>l</sup>ines. We compare the original model with variants incorporating three acceleration methods and PaceVLA. Action Quantization (Tang et al., 2025) applies fixed temporal action quantization with a ratio of 2 over the entire trajectory on inference. DemoSpeedup (Guo et al., 2025) accelerates demonstrations through entropy-guided temporal downsampling, and SAIL (Arachchige et al., 2025) combines action scheduling, temporally consistent prediction, and speed modulation to support faster policy execution. We use GR00T N1.5 (Bjorck et al., 2025) as the backbone for all experiments, and additionally use RLDX-1-IMG (Kim et al.,

Ta<sup>bl</sup>e 1: Simu<sup>l</sup>ation <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup> resu<sup>l</sup>ts. We compare PaceVLA with baseline methods on RoboCasa-Kitchen and LIBERO. For RoboCasa-Kitchen, we report results on the pick-and-place tasks (PnP) and the remaining tasks (Others), with models trained using 300 demonstrations per task. For LIBERO, we report results on the short-horizon suites (Short, averaged over Spatial, Object, and Goal) and the long-horizon suite (Long). We report the success rate (SR, %), throughput-with-regret (TPR, $\times 1 0 ^ { - 3 } )$ , and the efective speedup (ES). Base models are listed for reference (ES = 1 by definition). Among the acceleration methods for each base model, the best and second-best results are in <sup>b</sup>o<sup>ld</sup> and underlined, respectively.
<table><tr><td rowspan="3">Method</td><td colspan="5">RoboCasa-Kitchen</td><td colspan="5">LIBERO</td></tr><tr><td colspan="3">PnP</td><td colspan="3">Others</td><td colspan="2">Short</td><td colspan="3">Long</td></tr><tr><td>SR↑</td><td>TPR↑</td><td>ES↑</td><td>SR↑</td><td>TPR↑</td><td>ES↑</td><td>SR↑</td><td>TPR↑</td><td>ES↑</td><td>SR↑</td><td>TPR ↑ ES↑</td></tr><tr><td>RLDX-1-IMG [18]</td><td>48.5</td><td>1.10</td><td>1.00</td><td>76.3</td><td>2.73</td><td>1.00</td><td>96.7</td><td>7.93</td><td>1.00</td><td>88.6</td><td>3.15 1.00</td></tr><tr><td>+ Fixed Quantization [30]</td><td>43.0</td><td>1.55</td><td>0.87</td><td>72.0</td><td>4.00</td><td>0.97</td><td>89.4</td><td>11.80</td><td>1.23</td><td>79.2 4.44</td><td>1.10</td></tr><tr><td>+ DemoSpeedUp [13]</td><td>28.7</td><td>0.80</td><td>0.49</td><td>62.7</td><td>3.37</td><td>0.68</td><td>93.5</td><td>14.03</td><td>1.54</td><td>75.2 4.95</td><td>1.06</td></tr><tr><td>+ SAIL [2]</td><td>38.5</td><td>1.03</td><td>0.72</td><td>66.6</td><td>2.80</td><td>0.75</td><td>96.5</td><td>10.34</td><td>1.27</td><td>76.6 3.38</td><td>0.92</td></tr><tr><td>+ PACEVLA</td><td>45.0</td><td>1.54</td><td></td><td>0.93 76.9</td><td>3.21</td><td>1.07</td><td>97.7</td><td>10.36</td><td>1.29</td><td>90.2</td><td>4.44 1.31</td></tr><tr><td>GR00T N1.5 [5]</td><td>52.3</td><td>1.05</td><td>1.00</td><td>72.4</td><td>2.20</td><td>1.00</td><td>97.8</td><td>8.36</td><td>1.00</td><td>89.2 3.12</td><td>1.00</td></tr><tr><td>+ Fixed Quantization [30]</td><td>49.0</td><td>1.56</td><td></td><td>0.96 63.1</td><td>2.92</td><td>0.81</td><td>91.2</td><td>12.41</td><td>1.25</td><td>82.0</td><td>4.78 1.21</td></tr><tr><td>+ DemoSpeedUp [13]</td><td>40.3</td><td>1.15</td><td>0.71</td><td>57.3</td><td>2.77</td><td>0.67</td><td>87.7</td><td>12.09</td><td>1.14</td><td>71.6 3.91</td><td>0.90</td></tr><tr><td>+ SAIL [2]</td><td>32.5</td><td>0.77</td><td>0.52</td><td>58.5</td><td>2.31</td><td>0.68</td><td>95.5</td><td>10.85</td><td>1.23</td><td>79.0 3.89</td><td>1.03</td></tr><tr><td>+ PACEVLA</td><td>55.0</td><td>1.64</td><td></td><td>1.16 72.9</td><td>3.30</td><td>1.14</td><td>97.1</td><td>10.95</td><td>1.27</td><td>89.2 4.27</td><td>1.26</td></tr></table>

2026) on the simulation benchmarks to verify that PaceVLA generalizes across VLAs. Baseline adaptations are detailed in Appendix C.6 and evaluation protocols are detailed in Appendix C.1 and C.3.

Metrics. We report the success rate (SR, %) for every task, with partial score on Stack Cup and OpenArm tasks as detailed in Appendix C. In simulation, we report throughput-with-regret (TPR) (Arachchige et al., 2025), which scores each trial by how quickly it succeeds and assigns every failure the same penalty, together with an efective speedup (ES). Following the mean picks per hour (MPPH) metric of Dex-Net (Mahler et al., 2019), we define ES as the ratio of a method’s success throughput to that of its base model, where each failed trial is charged the full episode budget �<sup>max</sup>. A method that moves faster but fails more often can therefore have ES below 1. On the Franka and OpenArm tasks, we report the mean wall-clock execution time (Time, s) over fully successful trials, and on the ALLEX conveyor task, the number of packages handled successfully per minute (Thpt.). Full definitions are given in Appendix B.

## 4.2. Main Results

Simu<sup>l</sup>ation Benc<sup>h</sup>mar<sup>k</sup>s. Tables 1 and 3 compare PaceVLA with the acceleration baselines across RoboCasa-Kitchen, LIBERO, and DexJoCo. Across both VLA backbones, PaceVLA largely preserves the success rate of the base policy while improving execution eficiency, achieving an efective speedup above 1× in all but one setting. In contrast, the baseline acceleration methods often attain high raw throughput by sacrificing task success, resulting in an efective speedup below 1× in several settings. This trend is consistent on DexJoCo, where PaceVLA improves both success rate and efective speedup. Overall, PaceVLA provides a more favorable balance between task success and execution eficiency across action representations and VLA backbones.

Rea<sup>l</sup>-Wor<sup>ld</sup> Tas<sup>k</sup>s. Tables 2 and 4 compare PaceVLA with the baselines across the three real-world embodiments. On the Franka tasks, PaceVLA reduces execution time by 37–45% while preserving a success rate comparable to or higher than that of the base VLA. Notably, on the Air Fryer task, it reduces execution time from 24.6 s to 14.6 s without degrading success. Fixed Action Quantization achieves comparable or slightly shorter execution on some tasks, but drops substantial success rate by 28.2 and 21.9 percentage points on Air Fryer and Stack Cup, respectively. On the bimanual OpenArm tasks, PaceVLA reduces execution time by 33–39% while keeping the base model’s success rate. Finally, on the ALLEX conveyor task, PaceVLA increases throughput

Ta<sup>bl</sup>e 2: Rea<sup>l</sup>-Wor<sup>ld</sup> eva<sup>l</sup>uation resu<sup>l</sup>ts. We report average success rate (SR, %, over 32 trials per task; partial scores of 0.5 are given on Stack Cup, Table Sweeping, and Pump and Scrub, see Appendix C) and execution time (Time, average wall-clock execution time on fully successful trials) on five real-world tasks across two embodiments: three Franka+Gripper tasks (Pick-and-Place, Air Fryer, Stack Cup) and two OpenArm+Hand tasks (Table Sweeping, Pump and Scrub).
<table><tr><td rowspan="2"></td><td colspan="6">Franka + Gripper</td><td colspan="4">OpenArm + Bimanual Hand</td></tr><tr><td colspan="2">Pick-and-Place</td><td colspan="2">Air Fryer</td><td colspan="2">Stack Cup</td><td colspan="2">Table Sweeping</td><td colspan="2">Pump and Scrub</td></tr><tr><td>Method</td><td>SR ↑ Time (s)↓</td><td></td><td>SR ↑ Time (s) ↓</td><td></td><td>SR ↑ Time (s) ↓</td><td></td><td></td><td>SR ↑ Time (s) ↓</td><td>SR ↑ Time (s) ↓</td><td></td></tr><tr><td>GR00T N1.5 [5]</td><td>81.3</td><td>12.1</td><td>81.3</td><td>24.6</td><td>84.4</td><td>35.3</td><td>62.5</td><td>21.9</td><td>64.1</td><td>30.0</td></tr><tr><td>+ Fixed Quantization [30]</td><td>62.5</td><td>7.7</td><td>53.1</td><td>14.2</td><td>62.5</td><td>18.7</td><td>48.4</td><td>13.3</td><td>42.2</td><td>19.9</td></tr><tr><td>+ DemoSpeedUp [13]</td><td>68.8</td><td>8.1</td><td>71.9</td><td>18.9</td><td>53.1</td><td>24.2</td><td>56.3</td><td>17.9</td><td>53.1</td><td>20.2</td></tr><tr><td>+ SAIL [2]</td><td>65.6</td><td>7.9</td><td>56.3</td><td>16.8</td><td>45.3</td><td>20.0</td><td>46.9</td><td>15.4</td><td>45.3</td><td>15.3</td></tr><tr><td>+ PACEVLA</td><td>84.4</td><td>7.6</td><td>81.3</td><td>14.6</td><td>81.3</td><td>19.4</td><td>59.4</td><td>14.6</td><td>65.6</td><td>18.2</td></tr></table>

Ta<sup>bl</sup>e 3: DexJoCo <sup>b</sup>enc<sup>h</sup>mar<sup>k</sup> resu<sup>l</sup>ts. We compare PaceVLA with baselines on DexJoCo, where a dexterous hand performs contact-rich manipulation tasks. We report the success rate (SR, %), throughput-with-regret (TPR, ×10<sup>−3</sup>), and the efective speedup (ES) over 6 single-arm tasks. Among the acceleration methods, the best and secondbest results are in <sup>b</sup>o<sup>ld</sup> and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td>DexJoCo (Single-arm)</td></tr><tr><td>SR↑ TPR↑ ES↑</td></tr><tr><td>GR00T N1.5 [5]</td><td>76.3 1.83 1.00</td></tr><tr><td>+ Fixed Quantization [30]</td><td>63.0 1.65 0.70</td></tr><tr><td>+ DemoSpeedUp [13]</td><td>65.7 1.75 0.76</td></tr><tr><td>+ SAIL [2]</td><td>60.3 1.58 0.65</td></tr><tr><td>+ PACEVLA</td><td>77.3 2.22 1.11</td></tr></table>

Ta<sup>bl</sup>e 4: Rea<sup>l</sup>-wor<sup>ld</sup> eva<sup>l</sup>uation resu<sup>l</sup>ts. We compare PaceVLA with baselines on Conveyor-based package handling task with a humanoid ALLEX (15-DoF hand). We report the success rate (SR, %) and throughput (Thpt., packages/min). Among the acceleration methods, the best and second-best results are in <sup>b</sup>o<sup>ld</sup> and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Conveyor Package Handling</td></tr><tr><td>SR↑</td><td>Thpt. ↑</td></tr><tr><td>GR00T N1.5 [5]</td><td>86.0</td><td>3.6</td></tr><tr><td>+ Fixed Quantization [30]</td><td>72.0</td><td>4.2</td></tr><tr><td>+ DemoSpeedUp [13]</td><td>86.0</td><td>4.7</td></tr><tr><td>+ SAIL [2]</td><td>78.0</td><td>4.3</td></tr><tr><td>+ PACEVLA</td><td>94.0</td><td>6.3</td></tr></table>

from 3.6 to 6.3 packages per minute while improving success rate from 86% to 94%. These results show that PaceVLA transfers across gripper-based, bimanual, and humanoid manipulation without the substantial success rate degradation observed with the baseline acceleration methods.

## 4.3. More Anal<sub>y</sub>sis

A<sup>bl</sup>ation Stu<sup>d</sup>y. First, we remove each of the two key components one at a time. (i) Without the mode-specific decoders, we use the base VLA’s single decoder and directly quantize its predicted chunk with the same aggregation rule when the judge selects coarse execution. (ii) Without the quantization judge, the router selects among all modes. As shown in Table 5a, removing the mode-specific decoders reduces the success rate from 66.9% to 63.3% and the efective speedup from 1.14 to 1.00, indicating that dedicated decoders are important for jointly preserving task success and execution eficiency under temporal quantization. This is consistent with Appendix D.1, where dedicated decoders predict quantized actions more accurately than directly quantizing the base model’s predictions. Removing the quantization judge lowers TPR from 2.75 to 2.01 and ES from 1.14 to 1.01, showing that explicit quantizability supervision is necessary to decide when coarse execution is appropriate. Next, we examine the loss design in Table 5b. Removing the KL objective causes the largest degradation, reducing SR, TPR, and ES to 63.7%, 2.04, and 0.98, respectively. Removing the balancing objective also degrades all three metrics, showing that both objectives contribute to learning efective mode selection. Lastly, Table 5c compares router input representations. Average pooling achieves the highest SR, TPR, and ES, outperforming both attention pooling and the last-token representation, and we therefore use it as the default router input.

Ta<sup>bl</sup>e 5: A<sup>bl</sup>ation stu<sup>d</sup>y. Success rate (SR, %), throughput-with-regret (TPR, ×10<sup>−3</sup>), and efective speedup (ES) on RoboCasa-Kitchen, averaged over all 24 tasks. ES is computed relative to the base model GR00T N1.5. Our default setting is highlighted in violet .
<table><tr><td>Method</td><td>SR ↑ TPR ↑</td><td>ES↑</td></tr><tr><td>w/o Mode-specific Decoders</td><td>63.3</td><td>2.44 1.00</td></tr><tr><td>w/o Quantization Judge</td><td>65.1</td><td>2.01 1.01</td></tr><tr><td>PACEVLA</td><td>66.9</td><td>2.75 1.14</td></tr></table>

(a) Component ablation.

<table><tr><td>Method</td><td>SR↑</td><td>TPR↑</td><td>ES↑</td></tr><tr><td>w/o KL</td><td>63.7</td><td>2.04</td><td>0.98</td></tr><tr><td>w/o Balance</td><td>64.8</td><td>2.38</td><td>1.04</td></tr><tr><td>Balance + KL</td><td>66.9</td><td>2.75</td><td>1.14</td></tr></table>

(b) Loss design.

<table><tr><td>Method</td><td>SR↑</td><td>TPR ↑</td><td>ES↑</td></tr><tr><td>Attn. Pool</td><td>64.6</td><td>2.38</td><td>1.04</td></tr><tr><td>Last Token</td><td>64.2</td><td>2.43</td><td>1.03</td></tr><tr><td>Avg. Pool</td><td>66.9</td><td>2.75</td><td>1.14</td></tr></table>

(c) Router input.  
![](images/9b216b6e8cf45456aa05a18f2bc5fb263706be3be4fb59fb55920a82b01ad7d1.jpg)  
Figure 4: Quantiza<sup>b</sup>i<sup>l</sup>ity supervision ana<sup>l</sup>ysis. We visualize the quantizability scores annotated on a RoboCasa-Kitchen demonstration. Shaded bands indicate manipulation phases. Scores are low around contact transitions (grasp and place) and higher during reach and carry, so coarse execution is suppressed exactly where precise control matters.

Qua<sup>l</sup>itative Ana<sup>l</sup>ysis. Figure 4 visualizes the quantizability scores assigned along a representative RoboCasa-Kitchen demonstration. The quantizability scores remain higher during reaching and carrying, where coarse execution is generally tolerable, and decrease around grasping and placement, where finer control is required. The score also drops during the final approach to the object, indicating that the annotation captures changes in temporal precision requirements across phases.

Limitation. Table 6 compares the per-step inference latency, averaged over 100 runs on an RTX 5090 GPU. PaceVLA incurs a slight overhead over the base model (1.05×), as the judge and router must run before action generation. However, this ∼2 ms cost is negligible compared to the execution time saved (e.g., 24.6 s → 14.6 s on Air Fryer), so the eficiency gain far outweighs the added inference cost.

Ta<sup>bl</sup>e 6: In<sup>f</sup>erence cost. Per-step latency on RoboCasa-Kitchen.
<table><tr><td>Method</td><td>Latency (ms)↓</td><td>Rel. Cost</td></tr><tr><td>GR00T N1.5</td><td>40.07</td><td>1.00×</td></tr><tr><td>PACEVLA</td><td>42.15</td><td>1.05×</td></tr></table>

## 5<sub>.</sub> Conclusion

In this work, we have presented PaceVLA, a unified framework for adaptive temporal action execution in Vision-Language-Action models (VLAs). Motivated by the observation that VLAs inherit the fine-grained execution patterns of teleoperated demonstrations, and that fixed temporal action quantization trades task success for speed, we treat the temporal granularity and prediction horizon of actions as a policy decision rather than a fixed design choice. PaceVLA equips a shared action expert with mode-specific decoders operating at diferent temporal granularities and prediction horizons, and selects among them at each inference step through a quantizability-guided judge and a context-conditioned router. Our experiments on three simulation benchmarks and six real-world tasks across gripper-based, dexterous-hand, and humanoid platforms show that PaceVLA substantially shortens execution while maintaining the task success of the base VLA, unlike fixed or rule-based acceleration methods, with only marginal inference overhead. We hope that our work facilitates future research toward generalist robot policies that adapt their execution pace to the task at hand, enabling faster and more reliable deployment in practice.

## AI use statement

In this work, we used generative AI tools as part of the quantizability supervision pipeline. Claude Opus 5 (Anthropic, 2026) was used to help develop and refine the quantizability criteria and annotation prompts based on task-level evaluation outcomes and demonstration examples. Gemini 3.8 Flash (Google DeepMind, 2026) was subsequently used through its API to annotate demonstration chunks according to these criteria. The resulting ratings were aggregated into quantizability scores used to supervise the quantization judge, as described in Appendix C.5. We also used generative AI tools for grammar checking and minor readability edits of author-written text. The authors reviewed the AI-assisted research process and resulting manuscript content and take responsibility for the final content of this work.

## R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y</sub> St<sub>a</sub>t<sub>emen</sub>t

We provide hyperparameter details in Section 4 and Appendix C.3. In addition, we provide prompts in Appendix E.

## Referen<sub>c</sub>e<sub>s</sub>

Anthropic. Claude opus 5 system card. Model card, July 2026. URL https://www.anthropic.com/ claude-opus-5-system-card.

Nadun Ranawaka Arachchige, Zhenyang Chen, Wonsuhk Jung, Woo Chul Shin, Rohan Bansal, Pierre Barroso, Yu Hang He, Yingyang Celine Lin, Benjamin Jofe, Shreyas Kousik, et al. Sail: Faster-than-demonstration execution of imitation learning policies. arXiv preprint arXiv:2506.11948, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Johan Bjorck, Valts Blukis, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1.5: An improved open foundation model for generalist humanoid robots. https://research.nvidia.com/labs/gear/gr00t-n1\_5/, June 2025. Accessed: 2025-09-09.

Kevin Black, Manuel Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. Advances in Neural Information Processing Systems, 38:33383–33407, 2026.

Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Max Ehrlich, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. Advances in Neural Information Processing Systems, 38:91077–91100, 2026.

Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023.

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Figure AI. Helix accelerating real-world logistics. https://www.figure.ai/news/helix-logistics, 2025.

Google DeepMind. Gemini 3.8 flash model card. Model card, September 2026. URL https://deepmind. google/models/model-cards/gemini-3-8-flash/.

Lingxiao Guo, Zhengrong Xue, Zijing Xu, and Huazhe Xu. Demospeedup: Accelerating visuomotor policies via entropy-guided demonstration acceleration. arXiv preprint arXiv:2506.05064, 2025.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. � : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moh: Multi-head attention as mixture-of-head attention. arXiv preprint arXiv:2410.11842, 2024.

Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, and Mingyu Ding. Mixture of horizons in action chunking. arXiv preprint arXiv:2511.19433, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, Daewon Choi, Dongsu Han, et al. Rldx-1 technical report. arXiv preprint arXiv:2605.03269, 2026.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning, 2024.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791, 2023.

Jefrey Mahler, Matthew Matl, Vishal Satish, Michael Danielczuk, Bill DeRose, Stephen McKinley, and Ken Goldberg. Learning ambidextrous robot grasping policies. Science robotics, 4(26):eaau4984, 2019.

Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879–893. PMLR, 2018.

Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What matters in learning from ofline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems, 2024.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In IEEE International Conference on Robotics and Automation, pages 6892–6903. IEEE, 2024.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, and Song Han. Vlash: Real-time vlas via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031, 2025.

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

Hanwen Wang, Weizhi Zhao, Xiangyu Wang, Siyuan Huang, He Lin, Boyuan Zheng, Rongtao Xu, Gang Wang, Yao Mu, He Wang, et al. Dexjoco: A benchmark and toolkit for task-oriented dexterous manipulation on mujoco. arXiv preprint arXiv:2605.16257, 2026.

Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12156–12163. IEEE, 2024.

David D Yuan, Tony Z Zhao, Kaylee Burns, and Chelsea Finn. Speedtuning: Speeding up policy execution with lightweight reinforcement learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1184–1192. IEEE, 2025.

Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, and Shanghang Zhang. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for eficient robot manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 18764–18772, 2026.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

## A. Temporal Action Quantization Details

The aggregation rule depends on the action representation. For a source segment of � actions quantized into $T$ commands, let $0 = s _ { 0 } < s _ { 1 } < \dots < s _ { T } = H$ denote the boundaries of the $T$ contiguous temporal blocks. The �-th block therefore contains $\{ \mathbf { a } _ { s _ { i - 1 } + 1 } , \dotsc , \mathbf { a } _ { s _ { i } } \}$ . When � is not divisible by $T _ { i }$ we partition the source segment into $T$ contiguous blocks whose sizes difer by at most one. For absolute actions, each block is represented by its final command:

$$
\bar { \mathbf { a } } _ { i } = \mathbf { a } _ { s _ { i } } .\tag{5}
$$

This preserves the endpoint while removing intermediate commands.

For delta-pose actions, directly summing all action dimensions is generally invalid because rotational increments do not compose additively. We write each action as $\mathbf { a } _ { j } = ( \Delta \mathbf { p } _ { j } , \mathbf { u } _ { j } , g _ { j } )$ , where $\Delta \mathbf { p } _ { j }$ is the translational increment, $\mathbf { u } _ { j }$ is the rotational command, and $g _ { j }$ is the gripper command. Translations expressed in a fixed Cartesian frame can be summed directly, whereas rotations must be composed on SO(3).

Let $\phi$ denote the mapping from the policy’s rotational command to the rotation vector used by the controller, and define ${ \bf R } _ { j } = \mathrm { E x p } ( \phi ( { \bf u } _ { j } ) )$ ). Here, Exp $: \mathbb { R } ^ { 3 }  \mathrm { S O } ( 3 )$ and $\mathrm { L o g : S O ( 3 ) \to \mathbb { R } ^ { 3 } }$ denote the mappings between rotation vectors and rotation matrices. Under the spatial-increment convention, the quantized action for block � is

$$
\bar { \mathbf { a } } _ { i } = \left( \sum _ { j = s _ { i - 1 } + 1 } ^ { s _ { i } } \Delta \mathbf { p } _ { j } , \phi ^ { - 1 } \bigl [ \mathrm { L o g } \bigl ( \mathbf { R } _ { s _ { i } } \cdot \cdot \cdot \mathbf { R } _ { s _ { i - 1 } + 1 } \bigr ) \bigr ] , g _ { s _ { i } } \right) .\tag{6}
$$

Because rotations are noncommutative, directly adding rotational increments does not in general reproduce their ordered composition. The SO(3) composition in Eq. (6) preserves the net commanded rotation of the original block under the controller convention, although quantization may alter the intermediate trajectory.

When actions are normalized for policy learning, we first transform them back to controller coordinates, perform the aggregation above, and then normalize the resulting actions again. The resulting quantized target chunk is denoted by $\mathbf { A } _ { t } ^ { ( k ) } \in \mathbb { R } ^ { T _ { k } \times d _ { a } }$

## B<sub>.</sub> Evaluation Metrics

In this section, we describe the two eficiency metrics used in the simulation benchmarks in detail.

T<sup>h</sup>roug<sup>h</sup>put-wit<sup>h</sup>-regret (TPR). Following SAIL (Arachchige et al., 2025), TPR is defined as

$$
\mathrm { T P R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \frac { S _ { i } } { t _ { i } } - \frac { 1 - S _ { i } } { t ^ { \operatorname* { m a x } } } \right) ,\tag{7}
$$

where $N$ is the number of trials, $S _ { i } \in \{ 0 , 1 \}$ indicates whether trial � succeeds, $t _ { i }$ is its number of executed action steps, and $t ^ { \mathrm { m a x } }$ is the maximum number of steps allowed per trial, after which the trial is terminated and counted as a failure. TPR rewards faster successes while penalizing all failures equally.

E<sup>f</sup>ective spee<sup>d</sup>up (ES). Since TPR penalizes a failure by only $1 / t ^ { \mathrm { m a x } }$ , it underweights the time lost to failed trials. We therefore additionally report

$$
\mathrm { E S } = \frac { T _ { \mathrm { b a s e } } } { T } , \qquad T = L + \frac { 1 - \mathrm { S R } } { \mathrm { S R } } \cdot t ^ { \mathrm { m a x } } ,\tag{8}
$$

where $\mathrm { S R } \in ( 0 , 1 ]$ denotes the success rate expressed as a fraction, $L$ is the mean number of action steps over successful trials, � is the expected number of action steps per success including the $t ^ { \mathrm { m a x } }$ steps consumed by each failed trial, and $T _ { \mathrm { b a s e } }$ is $T$ of the corresponding base model. Equivalently, $1 / T$ is the number of successes per action step, $\mathrm { i . e . , }$ the success throughput, following the mean-picks-per-hour (MPPH) metric of Dex-Net (Mahler et al., 2019). $\mathrm { E S } > 1$ indicates that a method completes tasks faster than its base model once failures are taken into account, and $\mathrm { E S } = 1$ for the base model by definition.

Evaluation horizon. We use �<sup>max</sup> = 1500 for RoboCasa-Kitchen and DexJoCo, and �<sup>max</sup> = 220/280/300/520 for LIBERO-Spatial/Object/Goal/Long, following the default settings.

## C. Ex<sub>p</sub>erimental Details

We collect 100 demonstrations for Pick and Place, 50 each for Stack Cup, Table Sweeping, Air Fryer, and Pump and Scrub, and 320 for Conveyor Package Handling across three embodiments: a single-arm Franka Research 3 with a parallel-jaw gripper (Pick and Place, Stack Cup, Air Fryer), a bimanual OpenArm with two 6-DoF hands (Table Sweeping, Pump and Scrub), and the ALLEX humanoid (Conveyor Package Handling). The Franka setup uses a third-person and a wrist-mounted camera, and the bimanual setup uses a stereo egocentric camera pair, all at 720×1280 without depth; the ALLEX setup uses a head-mounted stereo camera as its sole vision sensor. We use absolute joint positions as the action mode: 8-dim for the Franka tasks, 26-dim for the bimanual tasks, and 48-dim for the ALLEX task, alongside joint torque estimates from all 48 DoF as physical signals. Unless otherwise noted, each model is evaluated over 32 trials.

## C<sub>.</sub>1<sub>.</sub> Task S<sub>p</sub>ecifications

Pic<sup>k</sup> an<sup>d</sup> P<sup>l</sup>ace The task requires picking up a single target object from the table and placing it into the plate. To prevent memorization of one grasp pose, we vary the target object across three categories that difer in size, shape, and required grasp width and vary the object position across the workspace to evaluate spatial generalization. We evaluate each model over 32 trials, consisting of 10,11, and 11 trials per object. We use the instruction “Pick up the [ doll / green cube / blue cup ] and put it into the plate." for this task.

Stac<sup>k</sup> Cup The task requires two stacking operations in a fixed order within one episode: the left blue cup is stacked on the yellow cup in the middle, and then the right blue cup is stacked on the same yellow cup. Initial cup positions are varied across trials. We evaluate each model over 32 trials. Each successful stack is worth 0.5 points, so a trial that completes only the first stack scores 0.5. We use the instruction “Stack the left blue cup on the yellow cup in the middle, then stack the right blue cup on the yellow cup in the middle." for this task.

Air Fryer This task requires sequential multi-stage manipulation with a Franka arm with a gripper: opening the air fryer basket, picking up the food item from the plate and placing it inside the basket, and closing the basket. We evaluate each model over 32 trials. We use the instruction “Open the air fryer basket, put the food from the plate into the basket, and close the basket.” for this task.

Ta<sup>bl</sup>e Sweeping This bimanual task requires the two arms to play asymmetric and simultaneous roles: the left hand holds the dustpan steady against the floor while the right hand sweeps debris into it with a brush. The debris layout is varied across trials. We evaluate each model over 32 trials. A trial receives 0.5 points for grasping the dustpan correctly and another 0.5 points for sweeping the debris into it. We use the instruction “Hold the dustpan with the left hand and sweep the debris into it with the brush in the right hand" for this task.

Pump an<sup>d</sup> Scru<sup>b</sup> This bimanual task requires picking up a scrubber with the left hand and scrubbing the interior of a white bowl, i.e., maintaining sustained periodic contact with a curved surface rather than executing a single grasp-and-place. We evaluate each model over 32 trials. A trial receives 0.5 points for pumping the soap and 0.5 points for scrubbing the bowl. We use the instruction “Pick up the blue scrubber with the left hand and scrub the white bowl" for this task.

Conveyor Pac<sup>k</sup>age Han<sup>dl</sup>ing The task requires picking up a parcel from the conveyor belt and placing it on the unloading table. Parcels are fed onto the belt by a human operator, so the robot has to handle each one as it arrives. To prevent memorization of one grasp pose, we vary the parcel across three categories that difer in size, rigidity, and the required grasp type, and vary the parcel position and orientation along the belt to evaluate spatial generalization. We evaluate each model over 5 trials, and each trial consists 10 consecutive packages. We use the instruction “Bring the package over, orient barcode up, then place it on the conveyor." for this task.

## C.2. Hardware Setu<sub>p</sub>

We evaluate on three platforms: OpenArm, ALLEX, and Franka Research 3, as shown in Figure 5.

OpenArm. OpenArm is a bimanual platform with two 7-DoF arms, each equipped with a 6-DoF Inspire RH56F1 hand. A ZED stereo camera is mounted on the neck, and its left and right views are used as visual inputs. We collect demonstrations by controlling the arms with ExoArm-7, a wearable exoskeleton for OpenArm, and the hands with Manus Pro gloves, where finger joint angles are computed from the fingertip positions via inverse kinematics. The neck is fixed during data collection.

ALLEX. ALLEX is a high-DoF upper-body humanoid. We use 48 joints: two 7-DoF arms, two 15-DoF five-finger hands, a 2-DoF waist, and a 2-DoF neck. A head-mounted stereo camera is the only visual input, with no wrist or side cameras. We collect demonstrations using a multi-device teleoperation system with separate interfaces for diferent robot components. A Meta Quest VR device controls the head and waist, while Vive Trackers provide wrist poses from which arm joint angles are obtained via inverse kinematics. Finger joint angles are similarly recovered from fingertip positions captured by Manus Pro gloves.

Fran<sup>k</sup>a Researc<sup>h</sup> 3. Following DROID (Khazatsky et al., 2024), we use a 7-DoF Franka Research 3 arm with a Robotiq 2F-85 gripper. An external ZED camera and a wrist-mounted ZED camera are used as visual inputs. We collect demonstrations via VR teleoperation with a Meta Quest controller.

![](images/e71b312b92b61f55d11a518c27c9484bedfe575568247982f35cf2a7eb2b46f9.jpg)  
Figure 5: Hardware setups and task environments.

## C.3. Simulation H<sub>yp</sub>er<sub>p</sub>arameter Details

Trainin . We implement PaceVLA on two pretrained VLAs, GR00T N1.5 and RLDX-1-IMG, following their oficial implementations. For both models, we freeze the VLM backbone and train the action expert, the mode-specific decoders, the router, and the judge. We use �=4 temporal modes: two fine-grained modes (16, 16) and (8, 8), and two quantized modes whose configuration depends on the benchmark (Table 8). We set $\lambda _ { s \mathrm { u p } } { = } 0 . 1 , \lambda _ { \mathrm { b a l } } { = } 0 . 0 5$ , and $\lambda _ { \mathrm { c o n f } } { = } 0 . 1$ , where $\lambda _ { \mathrm { s u p } }$ and � increase linearly from zero over the first 5k iterations. In inference, the quantized group is selected when the judge’s predicted quantizability score is at least the threshold �. The remaining hyperparameters are summarized in Table 7.

Contro<sup>ll</sup>er con<sup>fi</sup>guration. Both RoboCasa-Kitchen and LIBERO use an operational space controller (OSC). Since temporal action quantization merges consecutive delta actions into a single command, the controller must track a larger displacement per step, and the default position gain $( k _ { p } \mathrm { = } 1 5 0 )$ can undershoot large merged motions. We therefore select $k _ { p } \in \{ 1 5 0 , 3 0 0 , 6 0 0 \}$ per task, and apply the same per-task gain to all acceleration methods for a fair comparison. For base GR00T N1.5, we use the default gain.

Ta<sup>bl</sup>e 7: Hyperparameter details for training PaceVLA on simulation benchmarks.
<table><tr><td></td><td>GR00T N1.5 + PACEVLA</td><td>RLDX-1-IMG + PACEVLA</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Optimizer momentum</td><td> $\beta _ { 1 } , \beta _ { 2 } = 0 . 9 5 , 0 . 9 9 9$ </td><td> $\beta _ { 1 } , \beta _ { 2 } = 0 . 9 , 0 . 9 9 9$ </td></tr><tr><td>Optimizer weight decay</td><td>1e-5</td><td>1e-5</td></tr><tr><td>Learning rate</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Learning rate scheduler</td><td>Cosine decay</td><td>Cosine decay</td></tr><tr><td>Warmup iterations</td><td>3,000</td><td>3,000</td></tr><tr><td>Training iterations</td><td>60,000</td><td>60,000</td></tr><tr><td> $\lambda _ { \mathrm { s u p } } , \lambda _ { \mathrm { b a l } } , \lambda _ { \mathrm { c o n f } }$ </td><td>0.1, 0.05, 0.1</td><td>0.1, 0.05, 0.1</td></tr><tr><td>Router weight w temperature</td><td>0.5</td><td>0.5</td></tr><tr><td>Target distribution r temperature</td><td>0.3</td><td>0.3</td></tr><tr><td>Minimum mode probability</td><td>0.05</td><td>0.05</td></tr></table>

Ta<sup>bl</sup>e 8: Benchmark-specific settings. Quantized modes are given as $( H _ { k } , T _ { k } )$
<table><tr><td></td><td>RoboCasa-Kitchen</td><td>LIBERO</td><td>DexJoCo</td></tr><tr><td>Batch size</td><td>64</td><td>32</td><td>64</td></tr><tr><td>Quantized modes</td><td>(16, 8), (8, 4)</td><td>(16, 9), (8, 5)</td><td>(16, 8), (8, 4)</td></tr></table>

## C.4. Real H<sub>yp</sub>er<sub>p</sub>arameter Details

Training. All real-world experiments use GR00T N1.5 + PaceVLA with the same training hyperparameters as in Table 7, except for the embodiment-specific settings in Table 9. For ALLEX, we use a longer source horizon, so the fine-grained modes are (24, 24) and (12, 12); the other embodiments use (16, 16) and (8, 8). On Franka, Pick-and-Place is trained for 20k iterations, while Air Fryer and Stack Cup are trained for 30k iterations.

Contro<sup>ll</sup>er con<sup>fi</sup>guration. On the real robots, temporally quantized commands require both the arm and the end-efector to follow larger changes within a single control step. For the Franka arm, we use Cartesian impedance control with a translational stifness of 600 N/m (800 N/m for stacking), and set the Robotiq 2F-85 to its maximum finger speed (150 mm/s) and minimum grasp force $( \sim 2 0 \mathrm { N } )$ . For OpenArm, we use joint PD control with gravity compensation, with shoulder and elbow gains of $K _ { p } \in \{ 2 4 0 , 4 8 0 \}$ Nm/rad across two configurations.

Ta<sup>bl</sup>e 9: Embodiment-specific settings for real-world tasks. Modes are given as $( H _ { k } , T _ { k } )$
<table><tr><td></td><td>Franka</td><td>OpenArm</td><td>ALLEX</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>256</td></tr><tr><td>Training iterations</td><td>30,000</td><td>40,000</td><td>30,000</td></tr><tr><td>Fine-grained modes Quantized modes</td><td>(16, 16), (8, 8) (16, 8), (8, 4)</td><td>(16, 16), (8, 8) (16, 8), (8, 4)</td><td>(24, 24), (12, 12) (24, 12), (12, 6)</td></tr></table>

## C.5. Quantizability Supervision Details

We design the annotation criteria with the assistance of Claude Opus 5 (Anthropic, 2026). We use per-task success rates under fixed quantization only as task-level cues for constructing the quantizability criteria, and they are never used as supervision targets for the judge. These task-level cues help reveal quantizability patterns that are dificult to infer from task information alone and guide the construction of the visual checks. The criterion is decomposed into four to six concrete visual checks. Each check is rated on a 1–5 scale and contributing either positively or negatively to quantizability (prompts in Appendix E).

We then annotate the full datasets with Gemini 3.8 Flash (Google DeepMind, 2026). For each action chunk, the model receives the first-frame camera observations, task instruction, and computed facts derived from the demonstration, and returns one integer rating per check. The ratings are combined with fixed weights and normalized to [0, 1] to obtain the quantizability score �.

## C.6. Com<sub>p</sub>arison to SAIL

We reimplement SAIL (Arachchige et al., 2025) on the same GR00T-N1.5 and RLDX-1-IMG backbones used in our experiments. We follow the original implementation and hyperparameters, with three changes to keep the comparison fair. First, we parameterize action chunks as end-efector deltas to match the delta control interface of simulation benchmarks. Second, for Error-Adaptive Guidance (EAG), we condition on the four actions of the previous chunk that immediately follow the replanning step, $i . e . ,$ actions replaced by the new chunk, so that the condition is temporally aligned with the prediction. Third, we disable action scheduling. Instead, SAIL follows the same inference setting as our method and the baselines: synchronous inference on all tasks except ALLEX conveyor task, where all methods use asynchronous inference with real-time chunking (RTC) (Black et al., 2026). Thus, speedups within each task come from faster motion rather than from the inference setting.

## D<sub>.</sub> Additi<sub>ona</sub>l A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub>

## D.1. Quantitative Analysis on Ground-Truth Data

Setup. We examine whether dedicated coarse decoders provide an advantage over directly quantizing the actions predicted by a standard policy. We evaluate on the RoboCasa-Kitchen simulation benchmark (Nasiriany et al., 2024), sampling trajectory points across all 24 tasks. Both PaceVLA and the baselines are fine-tuned on the same data and receive the same ground-truth observations. The two predictions are constructed as follows:

• Ours: The output of the quantized decoder $D _ { k }$

• Base: The output of decoder, post-processed using the same quantization rule as the target of $D _ { k }$

Both are compared against the corresponding ground-truth quantized target $\mathbf { a } ^ { ( k ) }$

Metrics. Let $\hat { \mathbf { a } } = ( \hat { \mathbf { a } } _ { 1 } , \hdots , \hat { \mathbf { a } } _ { T } )$ and $\mathbf { a } = ( \mathbf { a } _ { 1 } , \dots , \mathbf { a } _ { T } )$ denote the predicted and ground-truth quantized chunks, respectively, with $\mathbf { a } _ { i } \in \mathbb { R } ^ { d }$ over the continuous action dimensions. We report MSE, cosine similarity, and Dynamic Time Warping (DTW).

MSE measures pointwise magnitude error:

$$
\mathrm { M S E } ( \hat { \mathbf { a } } , \mathbf { a } ) = \frac { 1 } { T d } \sum _ { i = 1 } ^ { T } \left. \hat { \mathbf { a } } _ { i } - \mathbf { a } _ { i } \right. _ { 2 } ^ { 2 } .\tag{9}
$$

Cosine similarity measures directional agreement:

$$
\cos ( \hat { \mathbf { a } } , \mathbf { a } ) = \frac { \mathrm { v e c } ( \hat { \mathbf { a } } ) ^ { \top } \mathrm { v e c } ( \mathbf { a } ) } { \| \mathrm { v e c } ( \hat { \mathbf { a } } ) \| _ { 2 } \| \mathrm { v e c } ( \mathbf { a } ) \| _ { 2 } } .\tag{10}
$$

DTW measures sequence similarity while allowing temporal misalignment. With $D ( 0 , 0 ) = 0$ and $D ( i , 0 ) =$ $D ( 0 , j ) = \infty$ for $i , j \geq 1$ 2

$$
\begin{array} { r l } & { \mathrel { \phantom { = } } \bigl \| \hat { \mathbf { a } } _ { i } - \mathbf { a } _ { j } \bigr \| _ { 2 } } \\ & { \mathrel { \phantom { = } } + \operatorname* { m i n } \{ D ( i - 1 , j ) , D ( i , j - 1 ) , D ( i - 1 , j - 1 ) \} , } \end{array}\tag{11}
$$

and $\mathrm { D T W } ( \hat { \mathbf { a } } , \mathbf { a } ) = D ( T , T )$

Resu<sup>l</sup>ts. Table 10 shows that the dedicated coarse decoders consistently outperform post-hoc quantization of the baseline predictions across both configurations and all three metrics. This indicates that directly learning the quantized action target better preserves the ground-truth motion than quantizing a fine-grained prediction after inference.

Ta<sup>bl</sup>e 10: Quantize<sup>d</sup>-action pre<sup>d</sup>iction accuracy on Ro<sup>b</sup>oCasa-Kitc<sup>h</sup>en. Ours uses dedicated coarse decoders, while Base quantizes GR00T N1.5 predictions using the same SO(3)-aware rule. (�, �) denotes the source horizon and quantization level, and Both averages the two configurations.
<table><tr><td>Decoder (H, q)</td><td>Metric</td><td>Ours</td><td>Base</td></tr><tr><td>(16,2)</td><td>MSE↓ DTW↓ Cosine ↑</td><td>0.0273 3.209 0.850</td><td>0.0344 3.777 0.809</td></tr><tr><td>(8,2)</td><td>MSE↓ DTW↓ Cosine ↑</td><td>0.0163 1.359 0.886</td><td>0.0243 1.658 0.847</td></tr><tr><td>Both</td><td>MSE↓ DTW↓ Cosine ↑</td><td>0.0218 2.284 0.868</td><td>0.0294 2.718 0.828</td></tr></table>

## D.2. Rollout Trajectory Comparison

![](images/ecd41dff9a3d5a3612d4860c0a2b21fda403516b38c96d1413e2f11c98166015.jpg)  
Figure 6: En<sup>d</sup>-e<sup>f</sup>ector ro<sup>ll</sup>out trajectories. End-efector trajectories of the base policy (blue) and PaceVLA (orange) from the same initial state; PaceVLA traverses the same path with fewer, sparser steps.

## D.3. Failure Case Anal<sub>y</sub>sis

Across the failed rollouts we inspected (Figure 7), most failures seem to come from perception rather than from the temporal decision. All four decoders use the same visual and language features from the backbone, so if the scene is misread, the choice of decoder makes little diference. In Pick-and-Place, most failures happen during grasping. The judge assigns low quantizability in this phase and the router switches to fine-grained modes, but the trajectory is aimed slightly of the object, so the gripper knocks the cup over and closes on air, or holds the object loosely and drops it during transport. In Stack Cup, the gripper misses the first cup slightly and hits its edge, so the first stack fails and the robot cannot recover. The base model fails in the same way, so we believe these errors are mostly due to perception rather than temporal adaptation.

We also see a similar pattern in the OpenArm bimanual hand tasks. The only visual input is an egocentric stereo camera, and the hands and tools often block the view of the contact region. In Table Sweeping, the left hand grasps the dustpan at the wrong angle, so its lip does not sit flush against the table, and the debris swept by the right hand piles up against the lip instead of going in. In Pump and Scrub, the policy pumps soap onto the scrubber but never proceeds to scrubbing, possibly because the hands occlude the scrubber from the camera, so the policy cannot see that the soap has already been applied. In these cases the missing information is in the observation, not in the trajectory, so finer temporal granularity is unlikely to help. These cases suggest that PaceVLA is still limited by the quality of its perception, and that better sensing, such as depth or tactile input, would add to the gains from adaptive temporal execution.

Stack Cup (Failure case)  
![](images/c324564d34227b80487ac127dfacb9c6521266ebec6efbe800b0d68b47c50102.jpg)  
Grasping the scrubber..  
Scrubber never lowers to the bowl  
Scrub failed!

Figure 7: Failure cases of PaceVLA. Most failures come from small targeting errors (Pick-and-Place, Stack Cup) or from the hands and tools blocking the contact region in the egocentric view (Table Sweeping, Pump and Scrub).

## E. Prom<sub>p</sub>ts used on Dataset Annotation

Sim<sub>:</sub> RoboCasa-Kitchen   
[USER]   
<3 images>   
You are judging one moment of a kitchen robot's motion, to decide whether the arm could get through the stretch   
of motion ahead of it faster -- reaching the same places in fewer and longer steps -- without changing the˓→   
˓→ outcome.   
Going faster leaves the arm fewer chances to correct itself along the way. That costs most where the arm still   
˓→ has to arrive somewhere exactly: setting down what it is carrying onto the spot it must end up on, or   
˓→ closing the gripper on one exact point -- and, when it is the gripper closing, more still if that spot is   
˓→ hemmed in. It costs least where the arm only acts on something that is anchored in place -- a button or   
˓→ switch pressed, a knob or tap turned, a door pushed shut. There a longer step just presses or turns a   
˓→ little harder and nothing is lost.   
Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one moment   
˓→ to the next.   
You are shown 3 camera views: agentview-left, agentview-right, and a wrist (eye-in-hand) close-up. The wrist   
˓→ camera is mounted on the gripper, so objects normally look close in it -- general closeness is normal. Use   
the wrist view to judge the close moments: the gripper actually closing on something, and something being˓→   
set down onto the spot it must end up on.˓→   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at all   
A grade refers only to the check on that line.   
A) Is the gripper CLOSING on its target right now, or about to -- the fingers   
already around the handle, rim or object and only the squeeze left, so that a   
slip here loses it?   
B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or   
a sink, or lined up with a slot -- so the gripper has little room around it as   
it closes?   
C) Is the arm just pressing, turning or pushing something that is anchored in   
place -- a button or switch, a knob or tap, a door or drawer being shoved shut   
-- and not something it has to keep hold of while it travels, the way a handle   
is held on to and pulled to swing a door or drawer open?   
D) Is the object ALREADY RESTING where it is meant to end up, taking its own   
weight, with nothing left but to open the hand and draw back?   
E) Is the gripper carrying something through open space -- already holding it,   
still well away from wherever it is going?   
F) Is the arm STILL LOWERING the object onto its target right now -- the gripper   
not yet opened, the object not yet taking its own weight, coming down onto   
the spot it must end up on?   
Answer:

Sim: LIBERO   
[USER]   
<2 images>   
You are judging one moment of a table-top robot's motion, to decide whether the arm could get through the   
stretch of motion ahead of it faster -- reaching the same places in fewer and longer steps -- without˓→   
changing the outcome.˓→   
Going faster leaves the arm fewer chances to correct itself along the way. That costs most when the gripper has   
to close on one exact thing -- and more still when the way in is threaded: the object perched on top of˓→   
another, or wedged in among things standing close by, so that there is one line in and a longer step swings˓→   
wider than that line. It costs least where whatever is being moved only has to land somewhere that will˓→   
catch it, and at the moments either side of the work -- crossing open space, carrying something with room˓→   
all round it, drawing back after letting go.˓→   
Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one moment   
to the next: the same episode that threads a bowl out from between two dishes then carries it through open˓→   
air.˓→   
You are shown 2 camera views of this one moment: a scene view and a wrist (eye-in-hand) close-up. The wrist   
camera is mounted on the gripper, so objects normally look close in it -- general closeness is normal. Use˓→   
the wrist view to judge the close moments: the gripper actually closing on something, and something being˓→   
set down into the place it must end up.˓→   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at all   
A grade refers only to the check on that line.   
A) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open   
basket, a broad tray or an open stretch of table with slack all round -- rather   
than a plate, a rack, or a spot barely bigger than the thing itself?   
B) Is the gripper closing on the object right now, or about to -- the fingers   
already around it and only the squeeze left, so that a slip here loses it?   
C) Is that closing spot threaded -- the thing being taken perched on top of another   
object, or wedged in among things standing close by -- so that there is only one   
line in for the hand?   
D) Is the hand ALREADY OPEN and coming away, the object left sitting where it was   
meant to end up and holding its own weight without the fingers?   
E) Is this a job that needs no hold kept on anything -- a push, a knob turned, a   
drawer pressed shut -- so that once it is sent it carries on without the hand?   
Answer:

## Sim: DexJoCo

[USER]   
[IMAGE]   
[IMAGE]   
You are judging one moment of a robot's motion with a dexterous hand, to decide whether the arm could get   
˓→ through the stretch of motion ahead of it faster -- reaching the same places in fewer and longer steps --   
˓→ without changing the outcome.   
Going faster leaves the arm fewer chances to correct itself along the way. Changing the working configuration   
˓→ of a held object and directing a held tool onto a localized target depend on the relation between the   
˓→ fingers, object and target. Maintaining a supported press and carrying an already grasped object through   
˓→ clear space impose different constraints. Use the visible interaction to judge these checks; do not infer   
˓→ contact or a secure grasp from commanded motion alone.   
Judge the moment in front of you, not the task as a whole. One task passes through different interactions from   
˓→ one moment to the next.   
You are shown 2 camera views of this one moment: image 1 is the scene view (observation.images.ego\_right), and   
˓→ image 2 is a wrist (eye-in-hand) close-up (observation.images.wrist). The wrist camera is mounted on the   
˓→ hand, so objects normally look close in it -- general closeness is normal. Use the wrist view to judge the   
˓→ close moments: the fingers operating the held object, a tool approaching its target, and a finger pressing   
a supported target.˓→   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at al1   
A grade refers only to the check on that line.   
A) Are the fingers operating movable parts of the object being held—closing   
jaws or folding an articulated part—so that changing the relation between   
those parts is the operation, rather than merely holding the object?   
B) Is a held tool being directed into an impact on a localized target, rather   
than making a sustained supported press? Use visible held-tool/target geometry together with the command   
˓→ motion to judge   
preparation for or execution of a targeted strike. Command targets do not establish   
that an impact actually occurred; the presence of a hammer alone is not enough.   
C) Is a finger or hand pressing a target supported by the surrounding surface,   
without lifting it, striking it with a tool, or manipulating movable parts   
within a grasp? A surface supports the target; do not assume it is bolted down.   
D) Is an already grasped object being moved or tilted as a whole through clear   
space, without changing the working configuration of its movable parts,   
acquiring/releasing the grasp, or entering a visibly tight target region?   
Answer:

R<sub>ea</sub>l<sub>-</sub>W<sub>or</sub>ld<sub>:</sub> F<sub>ran</sub>k<sub>a</sub>+G<sub>r</sub>i<sub>pper</sub> T<sub>as</sub>k<sub>s</sub>   
[USER]   
<2 images>   
You are judging one moment of a table-top robot's motion, to decide whether the arm could get through the   
stretch of motion ahead of it faster -- reaching the same places in fewer and longer steps -- without˓→   
changing the outcome.˓→   
Going faster leaves the arm fewer chances to correct itself along the way. That costs most at the two moments   
where the object can be lost: when the hand is closing on it, and when it is being brought down into the˓→   
place it has to end up. Both get worse the less room there is -- letting a cube down into a cup barely˓→   
wider than itself leaves no slack, while letting the same cube down onto an open plate does.˓→   
Most moments are neither. Crossing empty space to reach for something, carrying something with room all round   
it, or drawing the hand back after letting go -- a longer step changes nothing there. That is the common˓→   
case.˓→   
Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one moment   
˓→ to the next.   
You are shown 2 camera views of this one moment: a scene view and a wrist (eye-in-hand) close-up. The wrist   
˓→ camera is mounted on the gripper, so objects normally look close in it -- general closeness is normal. Use   
the wrist view to judge the close moments: the hand actually closing on the object, and the object being˓→   
brought down into the place it must end up.˓→   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at all   
A grade refers only to the check on that line.   
A) Is the hand closing on the object right now, or about to -- the fingers   
already around it and only the squeeze left, so that a slip here loses it?   
B) Is the place this object has to end up ROOMY FOR IT -- a broad plate or an   
open dish with slack all round -- rather than a cup or a container barely   
wider than the thing itself?   
C) Is the hand ALREADY OPEN and coming away, the object left sitting where it   
was meant to end up and holding its own weight without the fingers?   
D) Is the arm STILL LOWERING the object onto its target right now -- the hand   
not yet opened, the object not yet taking its own weight, coming down onto   
the spot it must end up on?   
Answer:

R<sub>ea</sub>l<sub>-</sub>W<sub>or</sub>ld<sub>:</sub> O <sub>en</sub>A<sub>rm</sub>+Bi<sub>manua</sub>l H<sub>an</sub>d T<sub>as</sub>k<sub>s</sub>   
[USER]   
<2 images>   
You are judging one moment of a two-armed robot's motion, each arm ending in a dexterous hand, to decide   
whether it could get through the stretch of motion ahead faster -- reaching the same places in fewer and˓→   
longer steps -- without changing the outcome.˓→   
Going faster leaves the arms fewer chances to correct themselves along the way. That costs most where a hand   
has to arrive somewhere exactly or close on something exactly: taking hold of a tool, pushing a hand-held˓→   
thing down onto something that is only standing on the table and can tip or slide, and guiding what is held˓→   
into a small target region. It costs least where a hand is crossing open space, carrying something that is˓→   
˓→ already firmly in the grasp with room all round it.   
The two hands may be doing different things at the same moment -- one may hold something steady while the other   
works. Read both. Use the visible interaction; do not infer contact or a secure grasp from commanded motion˓→   
alone.˓→   
Judge the moment in front of you, not the task as a whole. One task passes through different interactions from   
˓→ one moment to the next.   
You are shown 2 camera views of this one moment, both scene views of the same workspace from slightly different   
angles (image 1 is observation.images.ego\_left, image 2 is observation.images.ego\_right). There is no wrist˓→   
camera; neither view is an eye-in-hand close-up. This robot has two arms, each with a dexterous hand, and˓→   
they may be doing different things at the same time -- read both hands.˓→   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at all   
A grade refers only to the check on that line.   
A) Is a hand closing on a tool or object to take hold of it right now, or about   
to -- the fingers already around it and only the squeeze left, so that a slip   
here loses it -- rather than already holding it securely?   
B) Is a hand pushing down on something that is only standing on the table and is   
not held flat by it -- a bottle, a pump head, an upright container -- so that   
going too far tips it over or slides off it, rather than pressing on something   
the table holds down?   
C) Is what a hand holds being guided into or over a small target region -- across   
a rim or lip, into a narrow opening, onto a spot only a little larger than the   
thing itself -- so that missing that line loses the attempt?   
D) Is an already grasped object being moved or tilted as a whole through clear   
space, without changing the working configuration of its movable parts,   
acquiring/releasing the grasp, or entering a visibly tight target region?   
E) Is a hand moving through open space touching nothing at all -- on its way   
somewhere, or drawing back after letting go?   
Answer:

R<sub>ea</sub>l<sub>-</sub>W<sub>or</sub>ld<sub>:</sub> All<sub>ex</sub> C<sub>onveyer</sub> P<sub>ac</sub>k<sub>age</sub> H<sub>an</sub>dli<sub>ng</sub> T<sub>as</sub>k   
[USER]   
<2 images>   
You are judging one instant of a two-armed robot with hands, to decide how much of the next stretch of its   
motion could be thinned out -- how many of its commanded poses could be dropped, letting the arms travel˓→   
˓→ further between the ones that remain, without changing the outcome.   
What thinning takes away is the arms' chance to correct on the way, and there are two ways that loses the   
object. One is a hold that exists only as a balance: the thing is trapped between two hands and nothing is˓→   
˓→ closed around it, so the hold IS the push of one hand against the other. If that thing is stiff it cannot   
˓→ give, and a coarser swing changes the balance and drops it. Something soft squashes and stays pinched.   
The other is a load that does not hold its own shape. A bag or a sagging thing hangs, swings and shifts while   
˓→ it is carried, and the further the arms travel between commanded poses the further it swings.   
Most moments are neither. Sliding something across to the other side, pushing something that is still resting   
˓→ on a surface, or moving with nothing in the hands -- a longer step changes nothing.   
Judge the moment in front of you, not the name of the job. One segment passes through several of these from one   
˓→ second to the next.   
The two images are the left and right camera views of this one moment.   
The robot was told: {instruction}   
{computed facts}   
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the   
˓→ cameras show about the MOMENT in front of you, read together with those measurements.   
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating   
˓→ how far that check describes this moment:   
5 = clearly true of this moment   
4 = mostly true   
3 = partly true   
2 = barely true   
1 = not true at all   
A grade refers only to the check on that line.   
A) Are the two hands SQUEEZING something between them to hold it, and is that   
thing HARD -- a box, a carton, something that will not squash?   
B) Is the item a POUCH OR PACKET -- a soft mailer, a padded envelope, a   
plastic parcel -- rather than a rigid carton with square sides?   
C) Is the thing only being SENT ACROSS -- slid or passed over to the other side,   
or shoved on its way -- with no particular spot it has to come to rest on?   
D) Are the hands ON THEIR WAY -- reaching out towards something, or drawing back   
from what they have finished, carrying nothing between them?   
Answer: