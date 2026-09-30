# METACTRL: YOUR LARGE LANGUAGE MODELS CAN REASON BETTER AND MORE CONCISELY WITH A METACOGNITIVE CONTROLLER

Zhibin Wen<sup>1</sup>, Tao Han<sup>2,3</sup>, Lei Bai<sup>3</sup>, Can Li<sup>4</sup>, Yang Xu<sup>1∗</sup>

<sup>1</sup>Southern University of Science and Technology

<sup>2</sup>The Hong Kong University of Science and Technology

<sup>3</sup>Shanghai Artificial Intelligence Laboratory

<sup>4</sup>Tongji University

## ABSTRACT

Large reasoning models improve performance on challenging problems by allocating additional computation before answering, but longer reasoning does not always lead to better results and can introduce substantial redundant reasoning on simple problems. Conversely, aggressively shortening reasoning can degrade performance on difficult ones. Effective reasoning therefore requires dynamically deciding when additional computation is useful based on the reasoner’s capabilities and evolving solution state. Existing approaches often rely on predefined budgets or intervention rules, retrain the target reasoner, or require additional supervision. We introduce MetaCtrl, a lightweight controller that adaptively regulates a frozen reasoner without predefined token budgets or reasoner retraining. We formulate reasoning regulation as a sequential metacognitive control problem: MetaCtrl observes the evolving reasoning trace and decides whether to continue, simplify, skip redundant steps, or conclude reasoning. It is trained directly with reinforcement learning using a reward that prioritizes correctness while favoring shorter trajectories among correct solutions, requiring neither supervised intervention trajectories nor problem-specific budgets. Across seven benchmarks spanning mathematics, science, and code, MetaCtrl consistently improves the accuracy of LRMs while reducing their reasoning length. On DeepSeek-R1-Distill-Qwen-7B, it improves average accuracy by 4.7 points while reducing generation length by 53.3%. Without further training, the same controller transfers to an unseen reasoner (e.g., Qwen3- 14B), improving average accuracy by 2.9 points and reducing generation length by 50.3%. These results establish MetaCtrl as a plug-and-play controller for improving reasoning accuracy while substantially reducing inference-time generation. The code is available at https://github.com/binbin2xs/MetaCtrl.

## 1 INTRODUCTION

Recent large reasoning models (LRMs) (Jaech et al., 2024; Guo et al., 2025; Yang et al., 2025a) have shown that scaling test-time computation (Snell et al., 2025) through chain of thought (Kojima et al., 2022; Wei et al., 2022; Yao et al., 2023; Sprague et al., 2025) reasoning can substantially improve performance on challenging tasks (Phan et al., 2025; Li et al., 2025; Cao et al., 2025; Yu et al., 2025b; Chen et al., 2026). However, longer reasoning steps do no always lead to better performance. LRMs often spend excessive computation on redundant verification and unnecessary exploration, and may even deviate from an initially correct solution after prolonged deliberation (Chen et al., 2025b; Zhou et al., 2026; Zhang et al., 2026; Li et al., 2026). Conversely, overly restricting reasoning can hurt difficult problems that genuinely require more computation (Madaan et al., 2023; Jin et al., 2024). The key challenge is therefore not simply to make reasoning shorter or longer, but to determine whether additional reasoning remains beneficial and to allocate test-time computation accordingly.

![](images/a2da7257cbde114dd6b80038cfa6bf5d329acaa82efc6a989a86f6d68d48057e.jpg)  
Figure 1: Left: Accuracy and generation length of the same reasoner when controlled by MetaCtrl, controlled by frontier LLMs, or run without external control across five benchmarks. Right: Com parison of MetaCtrl with representative efficient reasoning baselines across two reasoning models.

This challenge has a close analogue to the classical concept of “metacognition” in cognitive science, which is commonly characterized as the monitoring and control of one’s own cognitive processes (Ackerman & Thompson, 2017). The Nelson–Narens framework (Nelson, 1990) characterizes metacognition as the interaction between a meta-level and an object-level processes, with the former monitoring and steering the state of the latter. Applying this concept to reasoning LLMs, it is natural to propose the idea of metareasoning strategy that regulates cognitive effort and strategy, including whether to continue, redirect, or terminate a reasoning process (Lieder & Griffiths, 2017).

We posit that overthinking in LRMs can be understood as a failure of resource-rational metareasoning. At any point in a reasoning trajectory, additional computation is worthwhile only when its expected benefit outweighs its cost. Continuing after this marginal utility has vanished leads to overthinking, whereas terminating while further computation is still valuable leads to underthinking. Crucially, the computational demand of a problem is not determined by the input alone(Liu et al., 2026b; Jia et al., 2026): it depends on the reasoner’s capability and evolves dynamically as the reasoning trajectory unfolds. For example, the same problem may be trivial for a stronger reasoner yet require substantial deliberation for a weaker one. Moreover, the same problem may require substantial computation early on, yet little or none once a promising solution path or sufficient solution has been reached. Therefore, efficient reasoning should be viewed as a dynamic resource-allocation problem, where the value of further reasoning must be continually reassessed as the process unfolds.

Existing approaches only partially address this dynamic allocation problem. Prompt-based methods (Aytes et al., 2025; Ding et al., 2024; Han et al., 2025; Xu et al., 2025) estimate problem difficulty from the input and predict a reasoning budget/strategy before generation. However, the input alone provides only limited evidence about how much computation a particular reasoner will need, making the accurate allocation difficult. Model-based methods (Liu et al., 2026b; Yang et al., 2025d; Ma et al., 2025b; Xia et al., 2025) use supervised fine-tuning (SFT) or reinforcement learning (RL) to induce concise or adaptive reasoning, but they entangle object-level reasoning with meta-level regulation, which may cause disturbance to the reasoner. Output-based methods (Li et al., 2026; Wang et al., 2025; Yang et al., 2026; Huang et al., 2026) instead regulate computation during inference through compression, pruning, early termination, or other interventions, but often rely on predefined proxy signals or decision rules whose reliability may vary across scenarios. While effective, these approaches do not fully address the trajectory-level challenge of regulating computation online, without retraining the reasoner or relying on predefined allocation rules.

This gap naturally motivates a resource-rational metacognitive approach to reasoning control, where computation is allocated online according to the problem and the evolving reasoning trajectory rather than prescribed in advance or encoded into the reasoner itself. We therefore formulate efficient reasoning as a sequential metacognitive control problem and introduce MetaCtrl, a learned metalevel controller that regulates a frozen object-level reasoner. The controller repeatedly observes the evolving reasoning state and dynamically determines how reasoning should proceed through four lightweight interventions: continue leaves reasoning unchanged, fast-think encourages concise derivation, skip-think bypasses redundant intermediate reasoning, and stop-think triggers a bounded conclusion before terminating deliberation, while preserving the reasoning already generated. Rather than relying on supervised intervention trajectories, fixed human-specified reasoning budgets, or hand-crafted rules, we train the controller directly with Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to learn when and how to regulate reasoning from trajectory-level outcomes. The reward signal jointly reflects final correctness and reasoning efficiency, providing credit to the sequence of meta-level decisions according to their eventual effect on task success and computation cost.

![](images/9de446f6efa30c1f5d93029e79439340cc975af839269ad56940402d437a3e83.jpg)  
Figure 2: Comparison of four paradigms for efficient LRMs inference. In external control methods, some approaches train the reasoner, while others keep it frozen.

We evaluate MetaCtrl on seven reasoning benchmarks spanning mathematics, science, and code. On the seen DeepSeek-R1-Distill-Qwen-7B reasoner, MetaCtrl improves accuracy while reducing generation length by 47.3%, 58.2%, and 72.1% on mathematical, scientific, and code reasoning, respectively. More importantly, the controller transfers to multiple unseen reasoners without modelspecific training. On Qwen3-14B, it reduces generation length by 51.6%, 62.4%, and 33.2% across the three domains, respectively, while improving accuracy across all evaluated benchmarks. Notably, compared with DeepSeek-V4.1-Flash (552B parameters) (Xu et al., 2026) as the controller under the same frozen reasoner, MetaCtrl (4B parameters) improves average accuracy by 7.6 percentage points while reducing average generation length by 7.3% across the five benchmarks. Overall, our contributions can be outlined as follows:

(1) Resource-rational metacognitive formulation. We frame overthinking and underthinking in LRMs as failures to appropriately regulate inference-time computation, and introduce MetaCtrl, a resource-rational metacognitive framework that dynamically regulates reasoning effort according to the reasoner and its evolving reasoning trajectory.

(2) Learning metacognitive control from outcomes. To learn when and how to regulate reasoning, we introduce a metacognitive policy learning approach that uses GRPO to optimize the sequential interventions in a frozen reasoner. Final correctness and reasoning efficiency jointly guide policy learning, eliminating the need for supervised intervention trajectories.

(3) Effective and transferable reasoning regulation. Comprehensive experiments show that MetaCtrl consistently improves the reasoning accuracy and efficiency of LRMs over strong baselines, and generalizes well to unseen reasoning models without additional training.

## 2 RELATED WORK

Following Sui et al. (2025), we group efficient reasoning methods into three categories and additionally discuss external reasoning control methods that are most closely related to ours, as illustrated in Figure 2. Prompt-based methods improve efficiency without modifying reasoner parameters (Lee et al., 2025; Yu et al., 2025a; Zhang et al., 2025; Renze & Guven, 2024; Xu et al., 2025; Han et al., 2025; Aytes et al., 2025; Liang et al., 2025; Liu et al., 2026a). They shorten reasoning or select budgets and reasoning strategies from the input. However, it is difficult to determine from the question alone how much and how to reason, and decisions made before generation cannot adapt to subsequent changes in reasoning progress. MetaCtrl instead regulates computation online by conditioning interventions on the evolving reasoning trajectory. Model-based methods internalize efficient reasoning through post-training (Kang et al., 2025; Ma et al., 2025b; Yang et al., 2025b; Yu et al., 2024; Xia et al., 2025; Luo et al., 2026; Hou et al., 2025; Aggarwal & Welleck, 2025; Shen et al., 2025; Liu et al., 2026b). These approaches train the reasoner itself to shorten, compress, or adapt its reasoning, often through SFT, RL, or explicit length constraints. In contrast, MetaCtrl freezes the reasoner and decouples problem solving from computation control, enabling the same controller to transfer across reasoners without retraining them. This separation becomes increasingly advantageous as reasoner scale grows, since only the lightweight controller requires training. Output-based methods regulate ongoing generation using signals such as confidence, certainty, hidden states, attention, or logits (Liao et al., 2025; Liu & Wang, 2025; Tikhonov et al., 2026; Hao et al., 2024; Yang et al., 2026; Fu et al., 2025; Chen et al., 2025a; Li et al., 2026). Many rely on predefined decision rules or thresholds, or require access to internal model signals, making their behavior sensitive to the chosen control criteria. MetaCtrl instead learns trajectory-conditioned interventions directly from correctness–efficiency rewards using only the observable reasoning trace. External reasoning control uses an auxiliary model to regulate a separate reasoner (Pan et al., 2025; Yang et al., 2025c; Wang et al., 2026; Xia et al., 2026). Prior methods may rely on substantially larger external models or even human feedback, and often participate directly in problem solving by generating, verifying, correcting, or providing solution-specific guidance, effectively coupling reasoning across multiple models. Others additionally retrain the reasoner or require distilled supervision and predefined token budgets. MetaCtrl instead learns a lightweight, budget-free controller through reinforcement learning, using only task-agnostic control actions to regulate a frozen reasoner without supplying solution content or requiring supervised intervention trajectories. Appendix A provides a more detailed discussion.

![](images/d312bca017c4eb8e4ba212a81683701d76d09922d895f7520680b4426dc11625.jpg)  
Figure 3: Overview of MetaCtrl. (a) A learned meta-level controller regulates a frozen reasoner through four intervention actions, without an input budget. (b) At each turn, it selects an intervention based on the current reasoning state. (c) The controller is trained with GRPO using an efficiencyaware outcome reward. (d) The trained controller generalizes to unseen reasoners without retraining.

## 3 METACTRL: METACOGNITIVE REASONING CONTROL

## 3.1 GENERATION WITH METACOGNITIVE CONTROL

We formulate reasoning as a generation process controlled by metacognitive actions. Let x denote an input problem, $f _ { \phi }$ a frozen object-level reasoner, and π a trainable meta-level controller. We

assume that the final successful reasoning trajectory can be decomposed into multiple sub-traces:

$$
[ \tau _ { 0 } , a _ { 1 } , \tau _ { 1 } , \dots , a _ { T } , \tau _ { T } ] ,
$$

in which $a _ { t }$ is an action sampled from the controller’s discrete output space $\mathcal { A }$ (section 3.2), determined by all preceding traces and actions:

$$
a _ { t } \sim \pi _ { \theta } ( \cdot \mid x \oplus \tau _ { 0 } \oplus a _ { 1 } \cdot \cdot \cdot \oplus a _ { t - 1 } \oplus \tau _ { t - 1 } ) ,
$$

where $\oplus$ denotes concatenation of traces and actions with the relative order preserved. Let $s _ { t }$ denote the condition as a state, then the action selection essentially from a Markovian policy $\pi _ { \boldsymbol { \theta } } ( \cdot \ \vert \ \boldsymbol { s } _ { t } )$ The newly selected action $a _ { t }$ produces a new trace $\tau _ { t }$ from the reasoner $f _ { \phi } .$ . This process repeats for $T$ steps and generate a full controlled reasoning trajectory $\xi = ( \tau _ { 0 } , a _ { 1 } , \tau _ { 1 } , \dots , a _ { T } , \tau _ { T } )$ , which terminates with a final answer $y _ { \xi }$ . Therefore, given a dataset $\mathcal { D } .$ , we optimize the controller $\theta$ to balance reasoning performance and efficiency:

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { ( x , y ^ { * } ) \sim \mathcal { D } , \xi \sim p _ { \theta , \phi } ( \cdot | x ) } \left[ R ( \xi , y ^ { * } ) \right] ,
$$

where $y ^ { * }$ denotes the ground-truth answer and $R$ jointly evaluates answer correctness and objectlevel reasoning cost. The reasoner parameters $\phi$ remain frozen throughout training.

## 3.2 ACTION SPACE OF METACOGNITIVE CONTROL

The action space of metacognitive controls $a _ { t }$ matter to the final performance. MetaCtrl’s basic principle is to place a lightweight text-modality interventions on the evolving reasoning trace. Concretely, at each control point, we select from a discrete action space

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { C O N T I N U E } , \mathrm { F A S T ~ T H I N K } , \mathrm { S K I P ~ T H I N K } , \mathrm { S T O P ~ T H I N K } \} . } \end{array}
$$

CONTINUE preserves the context and lets the reasoner proceed naturally. FAST THINK encourages a more concise continuation, while $\mathbf { S K I P }$ THINK more aggressively directs the reasoner toward the solution with only the necessary intermediate reasoning. STOP THINK signals that the current reasoning is sufficient and prompts the reasoner to conclude and finalize its answer. Except for CONTINUE, each action is implemented by appending a short textual prompt to the current context. The exact prompts are provided in Appendix B.1.

Where to intervene. Control is exposed only at natural language boundaries. After the initial uncontrolled trace $\tau _ { 0 } .$ , we let generation pause at sentence-ending paragraph boundaries, implemented using $" \cdot \langle \mathrm { n } \rangle \mathrm { n } " , " ? \langle \mathrm { n } \rangle \mathrm { n } "$ , and two whitespace variants ". $\harpoonright n ^ { \prime } { \mathrm { n } ^ { \prime \prime } } , \ " ? \setminus \backslash { \mathrm { n } ^ { \prime \prime } }$ . The controller is invoked between coherent reasoning traces rather than at arbitrary token positions, providing semantically meaningful states while leaving token-level generation uninterrupted.

When to intervene. The action is sampled from the policy distribution $\pi _ { \boldsymbol { \theta } } ( \cdot \ \vert \ s _ { t } )$ conditioned on the evolving interaction. In contrast, problem-level allocation methods select a reasoning strategy or budget before generation according to $\pi ( \cdot \mid x )$ , while output-based methods often trigger early exit through predefined proxy criteria, e.g., stopping when the confidence $C _ { t }$ of an induced trial answer exceeds a fixed threshold $\delta .$ Conditioning on $s _ { t }$ allows MetaCtrl to determine when to intervene from the semantic content of the reasoning trace, enabling flexible control as reasoning evolves.

## 3.3 OUTCOME-GUIDED POLICY OPTIMIZATION FOR EFFICIENT METAREASONING

We train MetaCtrl directly from the outcomes of its interactions with the frozen reasoner, without expert intervention trajectories or intermediate action labels. For each problem x, we sample a group of G controlled reasoning trajectories $\{ \xi _ { i } \} _ { i = 1 } ^ { G }$ using the controller $\pi _ { \theta }$ , and evaluate each trajectory by both answer correctness and reasoner generation length.

Correctness-gated efficiency reward. Let $c _ { i } \in \{ 0 , 1 \}$ indicate whether $\xi _ { i }$ produces the correct answer, and let $L _ { i }$ be the total number of tokens generated by the object-level reasoner, including both reasoning and final answer generation.

To prevent efficiency optimization from favoring prematurely terminated but incorrect reasoning, we reward shorter computation only among correct trajectories. Define

$$
\mathcal { Z } ^ { \mathrm { c o r r e c t } } = \{ i \mid c _ { i } = 1 \} , \qquad L _ { \mathrm { m i n } } ^ { \mathrm { c o r r e c t } } = \operatorname* { m i n } _ { i \in \mathcal { Z } ^ { \mathrm { c o r r e c t } } } L _ { i } , \qquad L _ { \mathrm { m a x } } ^ { \mathrm { c o r r e c t } } = \operatorname* { m a x } _ { i \in \mathcal { Z } ^ { \mathrm { c o r r e c t } } } L _ { i } .
$$

Table 1: Comparison across reasoning models of different scales and and multiple baselines.
<table><tr><td rowspan="3">Methods</td><td colspan="9">Math Reasoning</td><td colspan="4">Scientific Reasoning</td></tr><tr><td colspan="3">MATH-500</td><td colspan="2">AIME2024</td><td colspan="3">OmniMath</td><td colspan="3">GPQA Diamond</td><td colspan="2">AVG</td></tr><tr><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc. Len.</td><td></td><td>Reduc.</td><td>Acc. Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.↑</td><td>Reduc.↑</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-7B [Seen Reasoner]</td><td></td><td></td><td></td><td>47.2</td><td></td></tr><tr><td>NoThinking</td><td>79.4</td><td>699</td><td>-75.5</td><td>40.0</td><td>4134</td><td>-60.9</td><td>40.0 2083</td><td>-63.7</td><td>29.3</td><td>2084</td><td>-61.0</td><td></td><td></td><td>-65.3</td></tr><tr><td>Vanilla</td><td>85.2</td><td>2857</td><td>-0%</td><td>50.0</td><td>10570</td><td>-0%</td><td>45.0</td><td>5736</td><td>-0%</td><td>32.8</td><td>5349</td><td>-0%</td><td>53.3</td><td>-0%</td></tr><tr><td>D-Prompt</td><td>86.0</td><td>2726</td><td>-4.6%</td><td>46.7</td><td>10209</td><td>-3.4%</td><td>45.0</td><td>5398</td><td>-5.9%</td><td>30.3</td><td>5117</td><td>-4.3%</td><td>52.0</td><td>-4.6%</td></tr><tr><td>TokenSkip CoT-Valve</td><td>80.2</td><td>2109</td><td>-26.2%</td><td>40.0</td><td>5559</td><td>-47.4%</td><td>38.3</td><td>3461</td><td>-39.7%</td><td>28.8</td><td>4001</td><td>–25.2%</td><td>46.8</td><td>-34.6%</td></tr><tr><td>AdaCtrl</td><td>78.6</td><td>747</td><td>–73.9%</td><td>43.3</td><td>5871</td><td>–44.5%</td><td>41.7</td><td>3311</td><td>-42.3%</td><td>28.3</td><td>3226</td><td>-39.7%</td><td>48.0</td><td>–50.1%</td></tr><tr><td></td><td>74.0</td><td>3196</td><td>+11.9%</td><td>21.3</td><td>16889</td><td>+59.8%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TH2T ACTS</td><td>86.8 85.0</td><td>1772</td><td>-38.0%</td><td>50.0</td><td>7490</td><td>-29.1%</td><td>48.3</td><td>3377</td><td>-41.1%</td><td>32.8</td><td>4166</td><td>-22.1%</td><td>54.5</td><td>-32.6%</td></tr><tr><td>MetaCtrl</td><td>88.0</td><td>2181</td><td>-23.7% –42.5%</td><td>36.7 56.7</td><td>5909 5899</td><td>-44.1%</td><td>46.4</td><td>4622</td><td>-19.4%</td><td>40.9</td><td>5358</td><td>+0.2%</td><td>52.3</td><td>–21.8%</td></tr><tr><td></td><td></td><td>1644</td><td></td><td></td><td></td><td>-44.2%</td><td>48.7</td><td>3197</td><td>-44.3%</td><td>42.4</td><td>2234</td><td>-58.2%</td><td>59.0</td><td>-47.3%</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-32B [Unseen Reasoner]</td><td colspan="7"></td></tr><tr><td>NoThinking</td><td>80.2</td><td>659</td><td>-72.0</td><td>53.3</td><td>3208</td><td>-66.6</td><td>45.0</td><td>2146</td><td>-61.3</td><td>50.5</td><td>1627</td><td>-62.9</td><td>57.3</td><td>-65.7</td></tr><tr><td>Vanilla</td><td>87.2</td><td>2357</td><td>-0%</td><td>60.0</td><td>9605</td><td>-0%</td><td>50.0</td><td>5540</td><td>-0%</td><td>52.0</td><td>4384</td><td>-0%</td><td>62.3</td><td>-0%</td></tr><tr><td>D-Prompt</td><td>86.8</td><td>2198</td><td>-6.8%</td><td>56.7</td><td>9445</td><td>-1.7%</td><td>48.3</td><td>5135</td><td>-7.3%</td><td>49.5</td><td>3877</td><td>-11.6%</td><td>60.3</td><td>-6.9%</td></tr><tr><td>TokenSkip</td><td>79.8</td><td>1567</td><td>-33.5%</td><td>50.0</td><td>5856</td><td>-39.0%</td><td>43.3</td><td>3730</td><td>-32.7%</td><td>47.4</td><td>2874</td><td>-34.4%</td><td>55.1</td><td>-34.9%</td></tr><tr><td>CoT-Valve</td><td>85.3</td><td>2263</td><td>-3.9%</td><td>53.3</td><td>7997</td><td>-16.7%</td><td>43.3</td><td>4663</td><td>-15.8%</td><td>49.0</td><td>3550</td><td>-19.0%</td><td>57.7</td><td>-13.9%</td></tr><tr><td>TH2T</td><td>88.0</td><td>1733</td><td>-26.5%</td><td>56.7</td><td>7004</td><td>-27.1%</td><td>48.3</td><td>2998</td><td>-45.9%</td><td>52.0</td><td>2367</td><td>-46.0%</td><td>61.3</td><td>-36.4%</td></tr><tr><td>ACTS</td><td>85.0</td><td>2132</td><td>-9.5%</td><td>43.3</td><td>6323</td><td>-34.2%</td><td>50.3</td><td>4539</td><td>-18.1%</td><td>52.5</td><td>5036</td><td>+14.9%</td><td>57.8</td><td>-11.7%</td></tr><tr><td>MetaCtrl</td><td>88.0</td><td>1528</td><td>–35.2%</td><td>56.7</td><td>4951</td><td>-48.5%</td><td>50.6 3113</td><td></td><td>-43.8%</td><td>53.2</td><td>1968</td><td>–55.1%</td><td>62.1</td><td>-45.6%</td></tr></table>

For a correct trajectory, its relative efficiency is

$$
e _ { i } = \frac { L _ { \mathrm { m a x } } ^ { \mathrm { c o r r e c t } } - L _ { i } } { L _ { \mathrm { m a x } } ^ { \mathrm { c o r r e c t } } - L _ { \mathrm { m i n } } ^ { \mathrm { c o r r e c t } } } ,
$$

with $e _ { i } = 0$ when all correct trajectories have the same length. The reward is

$$
R _ { i } = \left\{ \begin{array} { l l } { r _ { \mathrm { c o r r e c t } } + \lambda _ { \mathrm { e f f } } e _ { i } , } & { c _ { i } = 1 , } \\ { } & { } \\ { r _ { \mathrm { w r o n g } } , } & { c _ { i } = 0 , } \end{array} \right.
$$

where $r _ { \mathrm { c o r r e c t } } > 0$ is the base correctness reward, $\lambda _ { \mathrm { e f f } } > 0$ controls the strength of the efficiency bonus, and $r _ { \mathrm { w r o n g } } < 0$ is the penalty for an incorrect trajectory. Thus, shorter trajectories are favored only among correct solutions, while all incorrect trajectories receive the same length-independent penalty, encouraging efficient reasoning without trading correctness for premature termination.

Online GRPO update. We optimize the controller online with GRPO (Shao et al., 2024) over trajectories sampled for the same problem. Following the group-normalization treatment of Dr. GRPO (Liu et al., 2025), we omit within-group standard-deviation normalization and use the mean-centered advantage

$$
\hat { A } _ { i } = R _ { i } - \bar { R } , \qquad \bar { R } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R _ { j } .
$$

The trajectory-level advantage ${ \hat { A } } _ { i }$ is assigned to the controller generated action tokens in $\xi _ { i } ,$ while reasoner generated tokens are masked from the policy loss. Consequently, policy optimization updates only π<sub>θ</sub>, with $f _ { \phi }$ remaining fixed.

Through outcome-guided optimization, MetaCtrl learns trajectory-dependent interventions that improve both reasoning correctness and efficiency, without expert intervention trajectories or intermediate action labels. Rather than imitating a fixed reasoning schedule, the controller learns to allocate computation dynamically from the relative outcomes of its own interactions with the frozen reasoner.

Table 2: Comparison across the Qwen3 series at different model scales and multiple baselines.
<table><tr><td rowspan="3">Methods</td><td colspan="9">Math Reasoning</td><td colspan="4">Scientific Reasoning</td><td colspan="2"></td></tr><tr><td colspan="3">MATH-500</td><td colspan="3">AIME 2024</td><td colspan="3">OlympiadBench</td><td colspan="3">GPQA Diamond</td><td colspan="2">AVG</td></tr><tr><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.↑</td><td>Reduc.↑</td></tr><tr><td colspan="10">Qwen3-8B [Unseen Reasoner]</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NoThinking</td><td>87.4</td><td>1480</td><td>-70.0</td><td>23.3</td><td>7121</td><td>-41.2</td><td>48.7</td><td>5219</td><td>-43.7</td><td>48.5</td><td>2564</td><td>-72.7</td><td>52.0</td><td>-56.9</td></tr><tr><td>Vanilla</td><td>92.2</td><td>4926</td><td>-0%</td><td>63.3</td><td>12101</td><td>-0%</td><td>59.9</td><td>9268</td><td>-0%</td><td>52.5</td><td>9382</td><td>-0%</td><td>67.0</td><td>-0%</td></tr><tr><td>TALE</td><td>91.8</td><td>3682</td><td>-25.3%</td><td>60.0</td><td>11847</td><td>-2.1%</td><td>56.1</td><td>7306</td><td>-21.2%</td><td>52.0</td><td>4161</td><td>-55.6%</td><td>65.0</td><td>-26.0%</td></tr><tr><td>Dynasor</td><td>92.4</td><td>3361</td><td>-31.8%</td><td>60.0</td><td>11365</td><td>-6.1%</td><td>62.4</td><td>7640</td><td>-17.6%</td><td>55.1</td><td>5647</td><td>-39.8%</td><td>67.5</td><td>–23.8%</td></tr><tr><td>DEER ASAG</td><td>92.4</td><td>2966</td><td>-39.8%</td><td>63.3</td><td>8930</td><td>-26.2%</td><td>61.7</td><td>7367</td><td>-20.5%</td><td>54.5</td><td>5334</td><td>–43.1%</td><td>68.0</td><td>-32.4%</td></tr><tr><td></td><td>93.0</td><td>3152</td><td>-36.0%</td><td>66.7</td><td>8683</td><td>–28.3%</td><td>63.9</td><td>7296</td><td>–21.3%</td><td>56.1</td><td>5714</td><td>-39.1%</td><td>69.9</td><td>-31.2%</td></tr><tr><td>ACTS MetaCtrl</td><td>90.2</td><td>2736</td><td>–44.5% –53.4%</td><td>46.7</td><td>7148</td><td>-40.9%</td><td>58.0</td><td>5059</td><td>–45.4%</td><td>57.6</td><td>5102</td><td>–45.6%</td><td>63.1</td><td>-44.1%</td></tr><tr><td></td><td>94.0</td><td>2294</td><td></td><td>66.7</td><td>8645</td><td>-28.6%</td><td>64.0</td><td>4134</td><td>–55.4%</td><td>57.1</td><td>3518</td><td>–62.5%</td><td>70.5</td><td>–50.0%</td></tr><tr><td colspan="10">Qwen3-14B [Unseen Reasoner]</td><td colspan="7"></td></tr><tr><td>NoThinking</td><td>88.0</td><td>1403</td><td>-70.3</td><td>30.0</td><td>7726</td><td>-26.7</td><td>50.4</td><td>5462</td><td>-35.9</td><td>50.5</td><td>2308</td><td>-70.0</td><td>54.7</td><td>-50.7</td></tr><tr><td>Vanilla</td><td>93.4</td><td>4725</td><td>-0%</td><td>70.0</td><td>10537</td><td>-0%</td><td>61.9</td><td>8519</td><td>-0%</td><td>58.6</td><td>7688</td><td>-0%</td><td>71.0</td><td>-0%</td></tr><tr><td>TALE</td><td>93.2</td><td>3616</td><td>–23.5%</td><td>70.0</td><td>10358</td><td>-1.7%</td><td>58.4</td><td>7250</td><td>-14.9%</td><td>59.1</td><td>5391</td><td>-29.9%</td><td>70.2</td><td>-17.5%</td></tr><tr><td>Dynasor</td><td>93.8</td><td>3958</td><td>-16.2%</td><td>70.0</td><td>11156</td><td>+5.9%</td><td>63.0</td><td>7336</td><td>-13.9%</td><td>59.6</td><td>5387</td><td>-29.9%</td><td>71.6</td><td>-13.5%</td></tr><tr><td>DEER</td><td>94.0</td><td>3410</td><td>-27.8%</td><td>73.3</td><td>8402</td><td>-20.3%</td><td>62.7</td><td>7020</td><td>-17.6%</td><td>60.6</td><td>4998</td><td>-35.0%</td><td>72.7</td><td>–25.2%</td></tr><tr><td>ASAG</td><td>95.0</td><td>3505</td><td>-25.8%</td><td>73.3</td><td>7602</td><td>-27.9%</td><td>64.6</td><td>6819</td><td>-20.0%</td><td>63.1</td><td>5075</td><td>-34.0%</td><td>74.0</td><td>-26.9%</td></tr><tr><td>ACTS</td><td>92.6</td><td>2664</td><td>-43.6%</td><td>53.3</td><td>6861</td><td>-34.9%</td><td>60.4</td><td>4875</td><td>-42.8%</td><td>59.6</td><td>4859</td><td>–36.8%</td><td>66.5</td><td>-39.5%</td></tr><tr><td>MetaCtrl</td><td>95.2</td><td>1938</td><td>–59.0%</td><td>73.3</td><td>6627</td><td>-37.1%</td><td>64.4 3712</td><td></td><td>–56.4%</td><td>63.6 2890</td><td></td><td>-62.4%</td><td>74.1</td><td>-53.7%</td></tr></table>

Table 3: Baseline comparison with frontier LLMs as controllers. The reasoning model is DeepSeek-R1-Distill-Qwen-7B.
<table><tr><td rowspan="2">Controller Model</td><td colspan="3">MATH-500</td><td colspan="3">AMC</td><td colspan="3">AIME2024</td><td colspan="3">OlympiadBench</td><td colspan="3">GPQA Diamond</td></tr><tr><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td></tr><tr><td>Vanilla (No Controller)</td><td>85.2</td><td>2857</td><td></td><td>73.5</td><td>7279</td><td></td><td>50.0</td><td>10570</td><td></td><td>48.6</td><td>8347</td><td></td><td>32.8</td><td>5349</td><td></td></tr><tr><td>Qwen-3.8-Flash</td><td>71.4</td><td>6548</td><td>+129.2%</td><td>60.2</td><td>9229</td><td>+26.8%</td><td></td><td></td><td></td><td>一</td><td></td><td></td><td>30.3</td><td>10584</td><td>+97.9%</td></tr><tr><td>Qwen-3.8-Max</td><td>78.2</td><td>1605</td><td>-43.8%</td><td>48.8</td><td>2507</td><td>-65.6%</td><td>30.0</td><td>7780</td><td>-26.4%</td><td>39.3</td><td>2921</td><td>-65.0%</td><td>38.4</td><td>1813</td><td>-66.1%</td></tr><tr><td>GLM-5.2</td><td>82.0</td><td>1472</td><td>-48.5%</td><td>66.9</td><td>2450</td><td>-66.3%</td><td>36.7</td><td>5524</td><td>-47.7%</td><td>43.0</td><td>2355</td><td>-71.8%</td><td>40.4</td><td>1930</td><td>-63.9%</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>84.6</td><td>1781</td><td>-37.7%</td><td>68.7</td><td>2892</td><td>-60.3%</td><td>36.0</td><td>5949</td><td>-43.7%</td><td>48.7</td><td>3371</td><td>-59.6%</td><td>38.0</td><td>2822</td><td>-47.2%</td></tr><tr><td>GPT-5.6-Luna</td><td>76.4</td><td>1326</td><td>-53.6%</td><td>53.0</td><td>1767</td><td>-75.7%</td><td>30.0</td><td>4320</td><td>-59.1%</td><td>37.6</td><td>1835</td><td>-78.0%</td><td>38.9</td><td>1665</td><td>-68.9%</td></tr><tr><td>MetaCtrl</td><td>88.0</td><td>1644</td><td>-42.5%</td><td>74.7</td><td>3036</td><td>-58.3%</td><td>56.7</td><td>5899</td><td>-44.2%</td><td>52.3</td><td>2965</td><td>-64.5%</td><td>42.4</td><td>2234</td><td>-58.2%</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

We evaluate our method on seven benchmarks spanning mathematical reasoning, scientific reasoning, and code reasoning, using multiple target reasoners and comparing against a broad set of recent and strong baselines. Full details of the training data, evaluation benchmarks and metrics, backbone models, baselines and implementation settings are provided in Appendix C, while prompting details are provided in Appendix B.

## 4.2 MAIN RESULTS

MetaCtrl improves reasoning accuracy while reducing redundant reasoning. As shown in Tables 1, on the seen DeepSeek-R1-Distill-Qwen-7B reasoner, MetaCtrl achieves the highest accuracy on all four benchmarks, improving average accuracy from 53.3% to 59.0% over Vanilla while reducing total generated tokens by 47.3%. Compared with D-Prompt, TokenSkip, TH2T, and ACTS, MetaCtrl improves average accuracy by 7.0, 12.2, 4.5, and 6.7 percentage points, while achieving an additional 42.7, 12.7, 14.7, and 25.5 points of token reduction, respectively. NoThinking and CoT-Valve exhibit more aggressive compression, with 18.0 and 2.8 points greater token reduction than MetaCtrl, but at substantial accuracy costs of 11.8 and 11.0 points. Notably, ACTS is the closest external control baseline, but requires a predefined token budget that directly affects both accuracy and generation length (Xia et al., 2026) and can be difficult to specify a priori, as the computation required for a correct solution varies across problems and reasoners. Overall, MetaCtrl achieves a consistently stronger accuracy–efficiency trade-off, suggesting that trajectory-dependent metacognitive control can reduce redundant reasoning while preserving or improving reasoning accuracy. More detailed comparisons across different baseline categories, end to end latency measurements, and case studies are provided in Appendices D, F, and I, respectively.

![](images/384a53b9b57eef27c0e549472a6492742ba4a75de7dd54c17632bd09dde10a79.jpg)

![](images/fb04ed97877d7ad3c4ad5dd64a64e1df3b22f03479111453ad407942cdff0920.jpg)

![](images/c7c34531505b0a763a9462597c4d7c8f24ed1f4733e9633bde2c969e8d36087d.jpg)

![](images/ae27998e4dbe5561ed82d82b10f2347e52cc49de223190f38cf38916b5bb9c01.jpg)

![](images/be0f99d2c38a448fe907e245b022d369734316f31fa795dd618d26a60982e2fc.jpg)

Figure 4: Action distributions of frontier LLMs as controllers and MetaCtrl across MATH-500 difficulty levels. The reasoning model is DeepSeek-R1-Distill-Qwen-7B.  
![](images/e5df4b930770915ccb20b3be473b5bad24da1baa2925db064f84507c50793732.jpg)  
Figure 5: Action distribution change by MetaCtrl training. The reasoning model is DeepSeek-R1- Distill-Qwen-7B.

MetaCtrl generalizes effectively to unseen reasoners. Beyond substantially reducing redundant reasoning and improving accuracy on the training-time reasoner, MetaCtrl transfers directly to unseen reasoners without any further training, while retaining strong accuracy–efficiency trade-offs. As shown in Tables 2, on Qwen3-8B and Qwen3-14B, MetaCtrl improves average accuracy from 67.0% to 70.5% and from 71.0% to 74.1%, respectively, while reducing total generated tokens by 50.0% and 53.7%. Compared with ASAG, the strongest baseline on these reasoners, MetaCtrl achieves comparable or higher accuracy while substantially larger token reductions (50.0% vs. 31.2% on Qwen3-8B and 53.7% vs. 26.9% on Qwen3-14B). As shown in Tables 1, on DeepSeek-R1-Distill-Qwen-32B, MetaCtrl maintains accuracy close to Vanilla, achieving 62.1% versus 62.3%, while reducing generation length by 45.6%. These results indicate that MetaCtrl learns transferable metacognitive control that generalizes across different reasoners and model scales without additional post training. Generalization to additional reasoner families is provided in Appendix E.1.

MetaCtrl generalizes across benchmarks and domains. As shown in Tables 1 and 2, MetaCtrl consistently achieves strong accuracy with substantially shorter generation lengths across mathematical and scientific reasoning tasks. On MATH-500, it improves or maintains high accuracy across all four reasoners while reducing generation length by 35.2% to 59.0%. Similar gains are observed on more challenging mathematical benchmarks, including AIME 2024, OmniMath, and OlympiadBench, where MetaCtrl substantially shortens reasoning trajectories while maintaining competitive accuracy. The same trend extends to GPQA Diamond, where MetaCtrl achieves accuracies of 42.4%, 53.2%, 57.1%, and 63.6% across the four reasoners, with generation reductions ranging from 55.1% to 62.5%. These results suggest that MetaCtrl provides effective reasoning control across diverse domains and problem difficulty levels. Additional code reasoning results are reported in Appendix E.1.

Comparison with frontier LLM controllers. Table 3 shows that stronger general purpose LLMs do not necessarily provide more effective reasoning control. MetaCtrl achieves the highest accuracy on all five benchmarks, outperforming the best frontier LLM controller by 2.0 to 20.0 percentage points. Although some frontier LLM controllers achieve larger token reductions on individual benchmarks, these gains often coincide with substantial accuracy losses. As shown in Figure 4, GLM-5.2, GPT-5.6-Luna, and Qwen-3.8-Max frequently apply aggressive shortening actions to difficult MATH 500 problems, coinciding with marked accuracy drops, whereas MetaCtrl better preserves continued reasoning on difficult problems while still reducing computation. These results suggest that effective control requires selectively reducing unnecessary computation while retaining the reasoning needed for correctness, rather than simply shortening trajectories.

Table 4: Ablation study on reward-function coefficients. The reasoning model is DeepSeek-R1- Distill-Qwen-7B.
<table><tr><td colspan="2">Reward Setting</td><td rowspan="2"></td><td colspan="3">MATH-500</td><td colspan="3">AMC</td><td colspan="3">AIME2024</td><td colspan="3">OlympiadBench</td><td colspan="3">GPQA Diamond</td></tr><tr><td>Tcorrect  $r _ { \mathrm { w r o n g } }$ </td><td>λeff</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td></tr><tr><td>Vanilla (No Controller)</td><td></td><td></td><td>85.2</td><td>2857</td><td></td><td>73.5</td><td>7279</td><td></td><td>50.0</td><td>10570</td><td></td><td>48.6</td><td>8347</td><td></td><td>32.8</td><td>5349</td><td></td></tr><tr><td>1.5</td><td>-1.0</td><td>0.5</td><td>88.8</td><td>1739</td><td>-39.1%</td><td>76.1</td><td>3175</td><td>-56.4%</td><td>40.7</td><td>6261</td><td>-40.8%</td><td>52.3</td><td>3092</td><td>-63.0%</td><td>43.4</td><td>2789</td><td>-47.9%</td></tr><tr><td>2.0</td><td>-1.0</td><td>0.5</td><td>90.8</td><td>1921</td><td>-32.8%</td><td>77.6</td><td>4162</td><td>-42.8%</td><td>44.0</td><td>8106</td><td>-23.3%</td><td>53.3</td><td>3870</td><td>-53.6%</td><td>37.9</td><td>4818</td><td>-9.9%</td></tr><tr><td>1.0</td><td>-0.5</td><td>0.5</td><td>85.8</td><td>1553</td><td>-45.6%</td><td>71.1</td><td>2545</td><td>-65.0%</td><td>40.7</td><td>5013</td><td>-52.6%</td><td>45.9</td><td>2590</td><td>-69.0%</td><td>36.9</td><td>1998</td><td>-62.6%</td></tr><tr><td>1.0</td><td>-1.0</td><td>1.0</td><td>82.0</td><td>1284</td><td>-55.1%</td><td>59.0</td><td>2053</td><td>-71.8%</td><td>30.7</td><td>4626</td><td>-56.2%</td><td>40.9</td><td>1968</td><td>-76.4%</td><td>37.5</td><td>1463</td><td>-72.6%</td></tr><tr><td>1.0</td><td>-1.0</td><td>0.5</td><td>88.0</td><td>1644</td><td>-42.5%</td><td>74.7</td><td>3036</td><td>-58.3%</td><td>56.7</td><td>5899</td><td>-44.2%</td><td>52.3</td><td>2965</td><td>-64.5%</td><td>42.4</td><td>2234</td><td>-58.2%</td></tr></table>

Adaptive Reasoning Allocation by Difficulty. Figure 4 examines how control policies vary with problem difficulty on MATH-500. For MetaCtrl, Continue increases from 65.8% at Level 1 to 77.2% at Level 5, while Stop Think decreases from 15.0% to 5.0%, suggesting that the controller becomes increasingly likely to preserve ongoing reasoning as problem difficulty increases. A similar coarse trend appears across the frontier LLM controllers, all of which increase Continue and reduce Stop Think as difficulty rises. Among them, DeepSeek-V4.1-Flash exhibits the most similar action distributions to MetaCtrl across difficulty levels and also achieves the strongest accuracy among the frontier LLM controllers, matching MetaCtrl at both difficulty levels but with longer generations. Beyond this shared trend, the controllers differ in their finer grained adjustments, with GPT-5.6- Luna substantially increasing Fast Think, and Qwen3.8-Max increasing Skip Think. Overall, this difficulty-aware control suggests that MetaCtrl intervenes more conservatively on harder problem while applying stronger reasoning compression on easier ones, contributing to its strong accuracy with shorter generations across difficulty levels. Further analyses of control behavior across tasks and reasoners are provided in Appendix E.2.

MetaCtrl training learns effective reasoning control. Figure 5 compares the controller’s action distributions before and after training across MATH-500 difficulty levels. Before training, the controller is dominated by Fast Think, whereas training shifts the policy strongly toward Continue at every difficulty level. On the hardest Level 5 problems, for example, Continue increases from 0.6% to 77.2%, while Fast Think decreases from 86.9% to 7.6%. This policy shift improves accuracy while reducing generation length on Levels 2 through 5, with accuracy gains of 12.5 to 15.2 percentage points and generation reductions of 40.4% to 49.4% on the harder Levels 3 through 5. Overall, these results suggest that effective reasoning control is not achieved by simply encouraging shorter or faster reasoning, but by learning when to preserve computation and when to reduce it. Further analyses of training dynamics and policy evolution are provided in Appendix G.

## 4.3 ABLATION STUDY

Reward coefficients. Table 4 examines the effect of reward coefficients on accuracy and generation length. The default setting, $r _ { \mathrm { c o r r e c t } } = 1 . 0 , r _ { \mathrm { w r o n g } } = - 1 . 0$ , and $\lambda _ { \mathrm { e f f } } = 0 . 5$ , is the only tested configuration that improves accuracy over Vanilla on all five benchmarks, while reducing generation length by 42.5% to 64.5%. Weakening the wrong answer penalty to −0.5 yields stronger compression but lowers accuracy on several tasks, while increasing $\lambda _ { \mathrm { e f f } }$ to 1.0 further shortens generation at a larger accuracy cost. Increasing $r _ { \mathrm { c o r r e c t } }$ benefits some tasks but is less consistent across benchmarks. Overall, these results highlight the importance of balancing correctness preservation and reasoning compression. Additional ablation results are provided in Appendix E.3.

## 5 CONCLUSION

In this paper, we present MetaCtrl, a lightweight metacognitive controller that dynamically regulates a frozen reasoner based on its evolving reasoning trajectory. MetaCtrl learns when and how to intervene through GRPO with a correctness-gated efficiency objective, without supervised intervention trajectories, predefined reasoning budgets, or reasoner retraining. Extensive experiments across seven benchmarks show that MetaCtrl consistently improves the reasoning accuracy of controlled LRMs while substantially reducing their reasoning length, and generalizes effectively to unseen LRMs spanning different model scales and architectures as well as across diverse reasoning domains. With matched GPU memory, MetaCtrl also substantially reduces end to end LRM inference latency. These results demonstrate that decoupling metacognitive control from problem solving provides an effective and transferable mechanism for adaptively allocating reasoning computation.

## AI USE STATEMENT

Generative AI tools were used solely to assist with language editing and writing refinement, including improving grammar, clarity, readability, and presentation of the manuscript. They were not used to develop the research methodology, design experiments, generate or analyze experimental results, or formulate the scientific conclusions of this work. All AI-assisted text was carefully reviewed, revised, and verified by the authors to ensure that it accurately reflects the intended technical content. The authors take full responsibility for the final content of the paper, including all statements, claims, and conclusions.

## REFERENCES

Rakefet Ackerman and Valerie A Thompson. Meta-reasoning: Monitoring and control of thinking and reasoning. Trends in cognitive sciences, 21(8):607–617, 2017.

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

AI-MO. AMC 2023, 2024. URL https://huggingface.co/datasets/AI-MO/ aimo-validation-amc.

Simon A Aytes, Jinheon Baek, and Sung Ju Hwang. Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 24307–24331, 2025.

Lang Cao, Yingtian Zou, Chao Peng, Renhong Chen, Wu Ning, and Yitong Li. Step guided reasoning: Improving mathematical reasoning using guidance generation and step reasoning. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 21112–21129, 2025.

Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, and Zhangyang Wang. Seal: Steerable reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986, 2025a.

Xin Chen, Feng Jiang, Yiqian Zhang, Hardy Chen, Shuo Yan, Wenya Xie, Min Yang, and Shujian Huang. Reasoning while asking: Transforming reasoning large language models from passive solvers to proactive inquirers. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 35069–35090, 2026.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning, 2025b.

Mengru Ding, Hanmeng Liu, Zhizhang Fu, Jian Song, Wenbo Xie, and Yue Zhang. Break the chain: Large language models can be shortcut reasoners. arXiv preprint arXiv:2406.06580, 2024.

Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, and Hao Zhang. Efficiently scaling LLM reasoning programs with certaindex. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=nn51ewu5k2.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Zhengyang Tang, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. In International Conference on Learning Representations, volume 2025, pp. 100540–100569, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Tokenbudget-aware llm reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 24842–24855, 2025.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

Jiameng Huang, Baijiong Lin, Guhao Feng, Jierun Chen, Di He, and Lu Hou. Efficient reasoning for large reasoning language models via certainty-guided reflection suppression. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 31176–31184, 2026.

Shijue Huang, Hongru Wang, Wanjun Zhong, Zhaochen Su, Jiazhan Feng, Bowen Cao, and Yi R Fung. Adactrl: Towards adaptive and controllable reasoning via difficulty-aware budgeting. arXiv preprint arXiv:2505.18822, 2025.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https: //github.com/huggingface/open-r1.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Repre sentations, volume 2025, pp. 58791–58831, 2025.

Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Zhongyu Ouyang, Chiyu Ma, and Soroush Vosoughi. What makes a good curriculum? disentangling the effects of data ordering on llm mathematical reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 34472–34488, 2026.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. The impact of reasoning step length on large language models. In Findings ofthe Association for Computational Linguistics: ACL 2024, pp. 1830–1842, 2024.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-of-thought without compromising effectiveness. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 24312–24320, 2025.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

Ayeong Lee, Ethan Che, and Tianyi Peng. How well do llms compress their own chain-of-thought? a token complexity approach. arXiv preprint arXiv:2503.01141, 2025.

Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E Gonzalez, and Ion Stoica. S\*: Test time scaling for code generation. In EMNLP (Findings), pp. 15964–15978, 2025.

Jiakai Li, Ke Qin, Rongzheng Wang, Yizhuo Ma, Qizhi Chen, Muquan Li, and Shuang Liang. Stop when further reasoning won’t help: Attention-state adaptive generation in reasoning models. arXiv preprint arXiv:2606.15070, 2026.

Guosheng Liang, Longguang Zhong, Ziyi Yang, and Xiaojun Quan. Thinkswitcher: When to think hard, when to think fast. In EMNLP (Findings), pp. 5185–5201, 2025.

Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning. arXiv preprint arXiv:2501.19324, 2025.

Falk Lieder and Thomas L Griffiths. Strategy selection as rational metareasoning. Psychological review, 124(6):762, 2017.

Xiang Liu, Xuming Hu, Xiaowen Chu, and Eunsol Choi. Diffadapt: Difficulty-adaptive reasoning for token-efficient llm inference. In International Conference on Learning Representations, volume 2026, pp. 31640–31666, 2026a.

Xin Liu and Lu Wang. Answer convergence as a signal for early stopping in reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 17907–17918, 2025.

Yongjiang Liu, Haoxi Li, Xiaosong Ma, Jie Zhang, and Song Guo. Think how to think: Mitigating overthinking with autonomous difficulty cognition in large reasoning models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 38105–38126, 2026b.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Haotian Luo, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Xiaochun Cao, Dacheng Tao, Naiqiang Tan, and Li Shen. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 14242–14257, 2026.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858, 2025a.

Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. Cot-valve: Lengthcompressible chain-of-thought tuning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6025–6035, 2025b.

MAA Committees. AIME problems and solutions. URL https://artofproblemsolving. com/wiki/index.php/AIME\_Problems\_and\_Solutions.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

Thomas O Nelson. Metamemory: A theoretical framework and new findings. In Psychology of learning and motivation, volume 26, pp. 125–173. Elsevier, 1990.

Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. Specreason: Fast and accurate inference-time compute via speculative reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=wCbOKbZ7kf.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Qwen Team. Qwen3.8-max: A new bar for coding and cowork, August 2026. URL https: //qwen.ai/blog?id=qwen3.8.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview. net/forum?id=Ti67584b98.

Matthew Renze and Erhan Guven. The benefits of a concise chain of thought on problem-solving in large language models. arXiv preprint arXiv:2401.05618, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, and Shiguo Lian. Dast: Difficulty-adaptive slow-thinking for large reasoning models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 2322–2331, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations, volume 2025, pp. 10131–10165, 2025.

Zayne Sprague, Fangcong Yin, Juan Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. In International Conference on Learning Representations, volume 2025, pp. 94118–94162, 2025.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Pavel Tikhonov, Ivan Oseledets, and Elena Tutubalina. Confidence leaps in llm reasoning: Early stopping and cross-model transfer. In Proceedings ofthe 19th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 602–616, 2026.

Qianyue Wang, Jinwu Hu, Yaofo Chen, Yufeng Wang, Bailin Chen, Huanxiang Lin, Yu Rong, Yuanqing Li, Zhiquan Wen, and Mingkui Tan. Intervene when it doubts: Conjunction-guided interactive reasoning. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=zVjLO7jg9T.

Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, and Rui Wang. Sampling-efficient test-time scaling: Self-estimating the best-of-n sampling in early decoding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=BcKYVmh3yH.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 3351–3363, 2025.

Yu Xia, Zhouhang Xie, Xin Xu, Byungkyu Kang, Prarit Lamba, Xiang Gao, and Julian McAuley. Agentic chain-of-thought steering for efficient and controllable llm reasoning. arXiv preprint arXiv:2606.03965, 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weipinng Wang. Dynamic early exit in reasoning models. In International Conference on Learning Representations, volume 2026, pp. 88170–88210, 2026.

Junjie Yang, Ke Lin, and Xing Yu. Think when you need: Self-adaptive chain-of-thought learning. arXiv preprint arXiv:2504.03234, 2025b.

Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. Speculative thinking: Enhancing small-model reasoning with large model guidance at inference time. arXiv preprint arXiv:2504.12329, 2025c.

Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025d. URL https://openreview.net/forum?id= 6ICFqmixlS.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1. arXiv preprint arXiv:2407.06023, 2024.

Ye Yu, Yaoning Yu, and Haohan Wang. Premise: Scalable and strategic prompt optimization for efficient mathematical reasoning in large models. arXiv preprint arXiv:2506.10716, 2025a.

Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Mahmoud Khademi, Hany Hassan Awadalla, Junjie Wang, Yujiu Yang, et al. Chain-of-reasoning: Towards unified mathematical reasoning in large language models via a multi-paradigm perspective. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 24914–24937, 2025b.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Ruiqi Zhang, Changyi Xiao, and Yixin Cao. Long or short cot? investigating instance-level switch of large reasoning models. arXiv preprint arXiv:2506.04182, 2025.

Xinliang Frederick Zhang, Anhad Mohananey, Alexandra Chronopoulou, Pinelopi Papalampidi, Somit Gupta, Tsendsuren Munkhdalai, Lu Wang, and Shyam Upadhyay. Do llms really need 10+ thoughts for “find the time 1000 days later”? towards structural understanding of llm overthinking. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 17005–17030, 2026.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Shu Zhou, Rui Ling, Junan Chen, Xin Wang, Tao Fan, and Hao Wang. When more thinking hurts: Overthinking in llm test-time compute scaling. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 23967–23977, 2026.

Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025. GitHub repository. Corresponding author: Xin Lv.

A Related Work Details 17   
B Prompt Templates 18   
B.1 Intervention Prompts 18   
B.2 Controller Prompt . 18   
C Experimental Details 19   
C.1 Datasets and Metrics. 19   
C.2 Backbone Models. 19   
C.3 Baseline Details . 19   
C.4 Implementation Details 20   
D Detailed Comparison with Efficient Reasoning Baselines 20   
E Additional Experimental Results and Analysis 22   
E.1 Generalization Across Benchmarks, Domains, and Reasoning Model Architectures. 22   
E.2 Adaptive Control Emerges from Task–Reasoner Interactions. 23   
E.3 More Ablation Studies 24   
F The Analysis of Latency and Computation Cost of Methods 25   
G Training Dynamics and Efficiency 25   
H Future work 26   
I Case Study 27   
I.1 Preventing Post-Solution Overthinking . 27   
I.2 Reducing Errors from Prolonged Reasoning 27

## A RELATED WORK DETAILS

We provide a more detailed discussion of the efficient-reasoning paradigms summarized in Figure 2.

Prompt-based methods. Prompt-based approaches improve reasoning efficiency without modifying model parameters (Lee et al., 2025; Yu et al., 2025a; Zhang et al., 2025). Concise Chainof-Thought (Renze & Guven, 2024) and Chain of Draft (Xu et al., 2025) prompt reasoning models to produce shorter and more concise reasoning traces, while TALE (Han et al., 2025) estimates question-specific token budgets before generation. Sketch-of-Thought (Aytes et al., 2025), ThinkSwitcher (Liang et al., 2025), and DiffAdapt (Liu et al., 2026a) further adapt reasoning formats or modes according to input characteristics. These methods can adapt computation to the input, but their main control decisions are made before or at the beginning of generation, when it is difficult to determine from the question alone how much computation is needed or how reasoning should proceed. Moreover, such decisions cannot respond to later changes in reasoning progress as the solution develops. MetaCtrl instead conditions its interventions on the evolving reasoning trace and adapts computation online throughout generation.

Model-based methods. Model-based approaches internalize efficient reasoning through posttraining (Kang et al., 2025; Ma et al., 2025b; Yang et al., 2025b). System-2 distillation (Yu et al., 2024) and TokenSkip (Xia et al., 2025) compress deliberative reasoning, while O1-Pruner (Luo et al., 2026), ThinkPrune (Hou et al., 2025), and L1 (Aggarwal & Welleck, 2025) optimize reasoning length through reinforcement learning or explicit token constraints. DAST (Shen et al., 2025) and TH2T (Liu et al., 2026b) further incorporate difficulty or redundancy awareness to adapt computation across problems. These methods modify the reasoner itself and therefore require modelspecific post-training, coupling problem solving with computation control. This is particularly costly for difficulty-aware methods because perceived problem difficulty is relative to the capability of the underlying reasoner, potentially requiring adaptation when the reasoner changes. MetaCtrl instead freezes the reasoner and trains only a lightweight meta-level controller, decoupling reasoning control from problem solving and avoiding costly reasoner training. This advantage becomes more pronounced as reasoner scale grows, while also enabling transfer across different reasoners without retraining them.

Output-based methods. Output-based approaches regulate the ongoing generation process or intervene on its internal representations (Liao et al., 2025; Liu & Wang, 2025; Tikhonov et al., 2026). Coconut (Hao et al., 2024) performs reasoning in continuous latent states rather than discrete language tokens. DEER (Yang et al., 2026) and Certaindex (Fu et al., 2025) use confidence or answerstability signals to determine when additional computation is unnecessary, while SEAL (Chen et al., 2025a) intervenes through internal representations. ASAG (Li et al., 2026) combines model confidence and attention entropy to adapt generation strategies. These approaches often rely on prede fined decision rules or thresholds, or require access to internal states such as hidden representations, attention, or logits. Moreover, fixed confidence-based criteria can behave differently across reasoning states and problem difficulty. MetaCtrl instead learns when and how to intervene directly from correctness–efficiency rewards using only the observable reasoning trajectory, without hand-crafted control criteria or access to internal model states.

External reasoning control. External control methods regulate a reasoner using a separate model. SpecReason (Pan et al., 2025) and Speculative Thinking (Yang et al., 2025c) use stronger models to verify, correct, or guide weaker reasoners. Although this can reduce computation performed by the target reasoner, the external model participates directly in problem solving by providing solution-specific reasoning guidance, effectively coupling inference across multiple capable agents and increasing system complexity. CGI (Wang et al., 2026) injects external state-based feedback from humans or LLM proxies. Its evaluator assesses the rationality and completeness of the current reasoning and can provide task-specific corrective guidance, thereby participating in the problem solving process to some extent. When an LLM proxy is used, this effectively forms a multi-model problem-solving system in which a stronger external model guides the target reasoner. CGI additionally trains the reasoner to respond to such interventions and may rely on an LLM proxy substantially larger than the target reasoner. ACTS (Xia et al., 2026) is more closely related to our setting because it controls a frozen reasoner with a separate controller. However, ACTS relies on distilled SFT data and requires a predefined token budget, which can be difficult to determine a priori because the required computation depends jointly on problem difficulty and reasoner capability.

MetaCtrl differs from these approaches in three respects. First, it keeps the target reasoner fully frozen and requires no model-specific adaptation. Second, its controller does not participate in solving the task or provide solution-specific content. Instead, it selects a small set of task-agnostic control actions that determine how the reasoner should proceed. The controller is learned directly through reinforcement learning from evolving reasoning trajectories, without supervised intervention trajectories or distilled SFT data. Third, MetaCtrl requires neither a predefined reasoning budget nor a large external proxy, allowing a lightweight controller to dynamically regulate when and how reasoning proceeds as the solution unfolds.

## B PROMPT TEMPLATES

## B.1 INTERVENTION PROMPTS

MetaCtrl realizes its meta-level actions through lightweight textual prompts inserted into the reasoner’s context at each control point. Table 5 reports the exact prompts used in our experiments. CONTINUE introduces no additional text, leaving the reasoning context unchanged. For the other actions, the corresponding prompt is appended to the current context before the frozen reasoner resumes generation. The leading newline used in implementation is omitted from the table for readability.

Table 5: Textual intervention prompts associated with the MetaCtrl action space.  
Action Intervention Prompt   
CONTINUE No textual intervention.   
FAST THINK Let’s keep only the essential derivation, avoid   
repetition, and proceed efficiently.   
SKIP THINK Let’s solve this directly with the minimum necessary   
reasoning and move to the answer.   
STOP THINK The reasoning is sufficient; state only the final   
conclusion briefly and close the reasoning now.

Importantly, STOP THINK does not forcibly truncate generation. Instead, it instructs the frozen reasoner to conclude its current reasoning and produce the final answer, preserving the same textbased interaction interface used by the other interventions.

## B.2 CONTROLLER PROMPT

The controller receives the original problem and the reasoner’s latest reasoning trajectory. Its system prompt is:

You control a frozen mathematical reasoner while it solves one problem.

The reasoner always generates its first reasoning step without controller intervention. You are called only if reasoning is still active after that step. On every controller turn, you receive the reasoner’s latest step and select one action from continue, fast think, skip think, or stop think. Optimize for a correct answer first; among correct solutions, prefer the shortest sufficient reasoning.

Actions:

• continue: add no intervention and let the reasoner continue naturally.

• fast think: ask for only the essential derivation and no repetition.

• skip think: skip redundant intermediate work and move to the shortest remaining derivation.

• stop think: run one bounded conclusion continuation, close the thinking channel, and proceed to the final answer. Existing reasoning is preserved and never truncated.

Output exactly one line and no explanation:

Action: <continue|fast think|skip think|stop think>

## C EXPERIMENTAL DETAILS

## C.1 DATASETS AND METRICS.

We use 7,500 mathematical problems sampled from OpenR1-Math (Hugging Face, 2025; Xia et al., 2026) for online controller training. We evaluate across three reasoning domains: mathematics on MATH-500 (Hendrycks et al., 2021), AIME 2024 (MAA Committees), Omni-MATH (Gao et al., 2025), OlympiadBench (He et al., 2024), and AMC (AI-MO, 2024), science on GPQA Diamond (Rein et al., 2024), and code on LiveCodeBench (Jain et al., 2025). We report accuracy (pass@1), average token length, and length reduction ratio, which denotes the percentage reduction in token length relative to Vanilla. For controlled methods, token length includes generations from both the target reasoner and the controller.

## C.2 BACKBONE MODELS.

During training, we initialize the controller from Qwen3-4B-Instruct-2507 (Yang et al., 2025a) and pair it with a frozen DeepSeek-R1-Distill-Qwen-7B reasoner (Guo et al., 2025). At evaluation, we test the controller on the training-time reasoner and assess cross-reasoner generalization by directly transferring it, without further training, to three unseen reasoners: DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Llama-8B (Guo et al., 2025), Qwen3-8B, and Qwen3-14B (Yang et al., 2025a).

## C.3 BASELINE DETAILS

We compare against a broad set of recent and strong baselines across six categories: (i) Vanilla: standard reasoning without intervention; (ii) Prompt-based methods: D-Prompt (Liu et al., 2026b), NoThinking (Ma et al., 2025a), and TALE (Han et al., 2025); (iii) Model-based methods: Token-Skip (Xia et al., 2025), CoT-Valve (Ma et al., 2025b), AdaCtrl (Huang et al., 2025) and TH2T (Liu et al., 2026b); (iv) Output-based methods: Dynasor (Fu et al., 2025), DEER (Yang et al., 2026), and ASAG (Li et al., 2026); and (v) External reasoning control: ACTS (Xia et al., 2026); and (vi) Frontier LLM as controller: GLM-5.2 (Zeng et al., 2026), Qwen3.8-Flash, Qwen3.8-Max (Qwen Team, 2026), DeepSeek-V4.1-Flash (Xu et al., 2026), and GPT-5.6-Luna<sup>1</sup>,, each accessed through its official API and directly used as the controller under the same controller–reasoner interface.

Prompt-based methods. D-Prompt (Liu et al., 2026b) augments the input with a difficulty reminder and encourages the reasoner to adjust its response length according to its self-assessed problem difficulty. NoThinking (Ma et al., 2025a) instructs the model to bypass explicit intermediate reasoning and directly generate the answer. TALE (Han et al., 2025) controls reasoning length through token-budget prompting, assigning a constrained reasoning budget before generation.

Model-based methods. TokenSkip (Xia et al., 2025) fine-tunes the target reasoner to skip nonessential reasoning tokens during generation. CoT-Valve (Ma et al., 2025b) learns controllable directions in model parameter space to regulate chain-of-thought length. TH2T (Liu et al., 2026b) applies two-stage fine-tuning to induce difficulty and redundancy awareness, enabling adaptive reasoning depth while suppressing redundant reasoning. AdaCtrl (Huang et al., 2025) combines difficulty-aware fine-tuning with reinforcement learning to self-assess problem difficulty and adaptively allocate reasoning budgets.

Output-based methods. Dynasor (Fu et al., 2025) checks partial reasoning at regular intervals and stops generation once successive intermediate answers reach sufficient agreement. DEER (Yang et al., 2026) instead evaluates intermediate predictions at adaptive transition points in the reasoning process and terminates early when a high-confidence answer emerges. ASAG (Li et al., 2026) jointly uses model confidence and attention-state information to determine whether further reasoning is likely to be useful and adaptively controls generation.

External reasoning control. ACTS (Xia et al., 2026) employs a separately trained controller to steer a frozen reasoner step by step. Its controller conditions on the evolving reasoning trajectory and an explicit remaining token budget, and is first initialized through supervised fine-tuning on steering data constructed from expert reasoning trajectories, followed by budget-conditioned reinforcement learning.

For the baseline methods, we directly report the results from the original TH2T (Liu et al., 2026b) and ASAG (Li et al., 2026) papers. For all evaluations of our method, we adopt the same configurations used in these prior works, ensuring fair comparison under matched inference settings. For ACTS (Xia et al., 2026), we reproduce its evaluation using greedy decoding for the reasoner to ensure a consistent decoding protocol across baselines. Implementation details are described in the Appendix C.4.

## C.4 IMPLEMENTATION DETAILS

Training and Serving. We directly train the controller with GRPO, without supervised finetuning. We use a learning rate of $1 \stackrel { \cdot } { \times } 1 0 ^ { - 6 } ,$ , a GRPO group size of 8, a rollout batch size of 32, and a global batch size of 64. We set $r _ { \mathrm { c o r r e c t } } = 1 , r _ { \mathrm { w r o n g } } = - 1$ , and the efficiency coefficient $\lambda _ { \mathrm { e f f } } = 0 . 5$ . During training, the controller is sampled with temperature 1.0 and top-p 0.95, whereas the frozen reasoner uses greedy decoding. All experiments run on 8 NVIDIA H20 GPUs. RL training is implemented with SLIME (Zhu et al., 2025). At inference time, the controller and reasoner are hosted as separate SGLang (Zheng et al., 2024) servers.

Evaluation Protocol. For fair comparison, we directly report the baseline results from the original TH2T (Liu et al., 2026b) and ASAG (Li et al., 2026) papers, and evaluate our method using the same configurations adopted in these works across all experiments. Since TH2T and ASAG report results with greedy reasoner decoding, we also reproduce ACTS under greedy reasoner decoding rather than directly using its official evaluation setting, which samples from the reasoner with temperature 0.6 and top-p 0.95. Since ACTS requires an explicit input reasoning budget, we set the token budget to 2,000 for MATH-500, 5,000 for AIME 2024, and 4,000 for Omni-MATH, OlympiadBench, and GPQA Diamond. For our method, the controller follows the ACTS decoding configuration with temperature 0.7 and top-p 0.8, while the reasoner uniformly uses greedy decoding across all evaluations. The reasoner-side generation limits are matched to those of the corresponding baseline settings to avoid confounding performance differences with inference configuration.

## D DETAILED COMPARISON WITH EFFICIENT REASONING BASELINES

Our main results are presented in Table 1 and Table 2. We provide a more detailed comparison with the four major families of efficient reasoning methods considered in our experiments. We focus on both empirical performance and how each paradigm allocates test-time computation.

Prompt-based methods. Prompt-based approaches regulate reasoning through instructions or token budgets specified before generation, and therefore cannot adapt their decisions to how the reasoning trajectory unfolds. This limitation is reflected in our results. On DeepSeek-R1-Distill-Qwen-7B, D-Prompt achieves 52.0% average accuracy with only a 4.6% token reduction, whereas MetaCtrl reaches 59.0% accuracy while reducing generation by 47.3%. NoThinking obtains a larger reduction of 65.3%, but its average accuracy drops to 47.2%, illustrating the cost of uniformly suppressing deliberation. The same trend holds for other reasoning models: compared with TALE, MetaCtrl improves average accuracy from 65.0% to 70.5% on Qwen3-8B and from 70.2% to 74.1% on Qwen3-14B, while increasing token reduction from 26.0% to 50.0% and from 17.5% to 53.7%, respectively. These results suggest that the amount of useful computation is difficult to determine from the input alone and is better adjusted according to the evolving reasoning state.

Model-based methods. Model-based methods improve reasoning efficiency by modifying the target reasoner itself through supervised fine-tuning or reinforcement learning. Such approaches can learn adaptive reasoning behaviors, but couple computation control with problem solving and generally require model-specific post-training. In contrast, MetaCtrl keeps the reasoner frozen and learns a separate metacognitive policy. On the DeepSeek-R1-Distill-Qwen-7B reasoner, TH2T, the strongest model-based baseline in average accuracy, achieves 54.5% accuracy with a 32.6% token reduction, while MetaCtrl reaches 59.0% accuracy and a 47.3% reduction. TokenSkip and CoT-Valve achieve 46.8% and 48.0% average accuracy, respectively, despite reducing generation by 34.6% and 50.1%.

A further limitation of difficulty-aware model-based approaches such as TH2T is that perceived problem difficulty is inherently reasoner-dependent. Models at different scales can have substan tially different capability boundaries, such that the same problem may be easy for a stronger reasoner but difficult for a weaker one. Consequently, a difficulty criterion calibrated for one model or benchmark does not necessarily transfer faithfully to another. Applying a shared difficulty standard across reasoners therefore risks conflating task difficulty with model capability, whereas properly calibrating difficulty would require model- and benchmark-specific estimation. Moreover, TH2T requires constructing additional training data and post-training each target reasoner, further increasing the adaptation cost.

MetaCtrl avoids these requirements by estimating how much further computation is useful directly from the evolving reasoning trajectory of the current reasoner. More importantly, the controller is trained once as a smaller meta-level model and can then regulate multiple larger frozen reasoners without retraining them individually. This substantially reduces the cost of deploying efficient rea soning across a family of models: rather than separately collecting data and post-training every target reasoner, a single reusable controller can improve the inference efficiency of multiple larger models. Empirically, on DeepSeek-R1-Distill-Qwen-32B, the same controller achieves 62.1% average accuracy with a 45.6% token reduction, compared with 61.3% and 36.4% for TH2T. This decoupling enables a single controller to generalize across reasoners, eliminating the need for model-specific retraining of computation control.

Output-based methods. Output-based methods regulate computation during inference and are therefore more adaptive than static prompting. However, existing approaches typically rely on predefined stopping criteria, confidence estimates, agreement statistics, or privileged model-internal signals. MetaCtrl instead learns its intervention policy directly from trajectory-level correctness and efficiency outcomes and requires only the observable reasoning trace. On Qwen3-8B, ASAG, the strongest output-based baseline, obtains 69.9% average accuracy with a 31.2% token reduction, whereas MetaCtrl achieves 70.5% accuracy while reducing generation by 50.0%. On Qwen3-14B, MetaCtrl achieves comparable accuracy to ASAG (74.1% vs. 74.0%) but more than doubles the token reduction, from 26.9% to 53.7%. Compared with DEER and Dynasor, MetaCtrl likewise achieves higher average accuracy together with substantially larger reductions in generated tokens. These results indicate that learning trajectory-dependent interventions can provide more aggressive computation savings without relying on manually specified control signals. More broadly, the fixed criteria underlying many output-based methods may not transfer uniformly across reasoners with different capability boundaries or across problems with different computational demands. MetaCtrl instead conditions each control decision on the evolving reasoning trajectory, allowing the regulation strategy to adapt dynamically to both the current reasoner and its problem solving progress.

External reasoning control. ACTS is the most closely related baseline because it also separates the controller from a frozen reasoner and makes online decisions from the evolving trajectory. However, ACTS requires supervised steering data and explicitly conditions the controller on a predefined remaining token budget. MetaCtrl removes both requirements: it is trained directly from outcomelevel reinforcement and determines how much additional reasoning is useful without a user-specified budget. The performance gap is consistent across all four evaluated reasoners. On DeepSeek-R1- Distill-Qwen-7B, MetaCtrl improves average accuracy from 52.3% to 59.0% and increases token reduction from 21.8% to 47.3%. On the unseen 32B reasoner, it improves accuracy from 57.8% to 62.1% while increasing token reduction from 11.7% to 45.6%. Similar gains are observed on Qwen3-8B and Qwen3-14B, where MetaCtrl improves average accuracy over ACTS by 7.4 and 7.6 percentage points, respectively, while also achieving larger token reductions (50.0% vs. 44.1% and 53.7% vs. 39.5%). Although increasing the prescribed budget in ACTS may improve accuracy, the minimum budget sufficient for a correct solution is unknown a priori and varies across problems and reasoners, leaving users to choose it heuristically. MetaCtrl instead learns to allocate additional computation online from the evolving reasoning trajectory without requiring such a predefined budget.

Table 6: Performance and reasoning efficiency across unseen reasoners on AMC and Live-CodeBench.
<table><tr><td rowspan="2">Methods</td><td colspan="3">AMC</td><td colspan="3">LiveCodeBench</td></tr><tr><td>Acc.↑</td><td>Len.↓</td><td>Reduc.↑</td><td>Acc.↑</td><td>Len.↓</td><td>Reduc.↑</td></tr><tr><td colspan="7">DeepSeek-R1-Distill-Qwen-7B [Seen Reasoner]</td></tr><tr><td>Vanilla</td><td>73.5</td><td>7279</td><td></td><td>38.4</td><td>10,516</td><td></td></tr><tr><td>MetaCtrl (ours)</td><td>74.7</td><td>3036</td><td>-58.3%</td><td>42.3</td><td>2,932</td><td>-72.1%</td></tr><tr><td colspan="7">Qwen3-8B [Unseen Reasoner]</td></tr><tr><td>Vanilla</td><td>78.8</td><td>9083</td><td></td><td>64.7</td><td>8,923</td><td></td></tr><tr><td>MetaCtrl (ours)</td><td>81.9</td><td>4623</td><td>-49.1%</td><td>65.0</td><td>5855</td><td>-34.4%</td></tr><tr><td colspan="7">Qwen3-14B [Unseen Reasoner]</td></tr><tr><td>Vanilla</td><td>81.9</td><td>8505</td><td></td><td>72.6</td><td>8,112</td><td></td></tr><tr><td>MetaCtrl (ours)</td><td>84.3</td><td>3938</td><td>-53.7%</td><td>74.7</td><td>5416</td><td>-33.2%</td></tr><tr><td colspan="7">DeepSeek-R1-Distill-Llama-8B [Unseen Reasoner]</td></tr><tr><td>Vanilla</td><td>67.5</td><td>7982</td><td></td><td>41.5</td><td>11029</td><td></td></tr><tr><td>MetaCtrl (ours)</td><td>68.7</td><td>3357</td><td>-57.9%</td><td>43.1</td><td>3861</td><td>-65.0%</td></tr></table>

Table 7: Results of DeepSeek-R1-Distill-Llama-8B across five reasoning benchmarks. Acc. denotes pass@1 accuracy, Len. denotes the average total generation length, and Reduc. denotes the relative reduction in generation length compared with vanilla reasoning.
<table><tr><td rowspan="2">Methods</td><td colspan="3">MATH-500</td><td colspan="3">AIME 2024</td><td colspan="3">OlympiadBench</td><td colspan="3">GPQA Diamond</td></tr><tr><td></td><td></td><td>Acc.↑ Len.↓ Reduc.↑</td><td></td><td>Acc.↑ Len.↓</td><td>Reduc.↑</td><td></td><td></td><td>Acc.↑ Len.↓ Reduc.↑</td><td></td><td>Acc.↑ Len.↓ Reduc.↑</td><td></td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Llama-8B [Unseen Reasoner]</td></tr><tr><td></td><td></td><td>3928</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>9655</td><td></td></tr><tr><td>Vanilla MetaCtrl (ours)</td><td>87.6 89.0</td><td>1942</td><td>-50.6%</td><td>40.0 40.0</td><td>13472 7642</td><td>-43.3%</td><td>47.3 50.1</td><td>8416 3819</td><td>-54.6%</td><td>26.3 37.4</td><td>3192</td><td>-66.9%</td></tr></table>

Across the four paradigms, the results reveal complementary limitations of existing approaches: prompt-based methods decide computation too early, model-based methods require modifying each target reasoner, output-based methods rely on predefined control signals, and existing external controllers require additional supervision or explicit reasoning budgets. MetaCtrl instead learns a separate, budget-free, trajectory-conditioned control policy directly from correctness and efficiency outcomes while keeping the reasoner frozen. Rather than requiring a predefined computation budget, it dynamically determines how much additional reasoning is needed based on the evolving trajectory. This design yields a better accuracy–efficiency trade-off on the training-time reasoner and, critically, transfers to unseen reasoners without additional post-training.

## E ADDITIONAL EXPERIMENTAL RESULTS AND ANALYSIS

## E.1 GENERALIZATION ACROSS BENCHMARKS, DOMAINS, AND REASONING MODEL ARCHITECTURES.

We further evaluate MetaCtrl on additional benchmarks, the code reasoning domain, and an unseen reasoner architecture. As shown in Tables 6 and 7, the same controller transfers to unseen reasoners on AMC and LiveCodeBench without additional training, reducing generation length by 33.2% to 65.0% while maintaining or improving accuracy in all settings. This transfer extends to DeepSeek-R1-Distill-Llama-8B, whose Llama backbone differs from the Qwen-based reasoner used for controller training. On AMC and LiveCodeBench, MetaCtrl improves accuracy from 67.5% to 68.7% and from 41.5% to 43.1%, while reducing generation length by 57.9% and 65.0%, respectively. Across MATH-500, AIME 2024, OlympiadBench, and GPQA Diamond, it further reduces generation length by 43.3% to 66.9% while matching or improving Vanilla accuracy, including an 11.1 percentage point gain on GPQA Diamond. Together, these results show that MetaCtrl is not tied to the training benchmark, reasoning domain, or reasoner architecture, but transfers effectively across unseen tasks and reasoning models without additional training.

![](images/6bb1ae2a13663eddab85d7e36d3b55ffd7e17036460b8b32a6c5f8c8d4a08c1f.jpg)  
Figure 6: Controller action distributions across different reasoners and benchmarks. Each row corresponds to a reasoner and each column to a benchmark. The learned policy exhibits distinct task– reasoner-specific control patterns rather than a uniform intervention strategy. DeepSeek-R1-7B is used as shorthand for DeepSeek-R1-Distilled-Qwen2.5-7B.

## E.2 ADAPTIVE CONTROL EMERGES FROM TASK–REASONER INTERACTIONS.

Adaptive control across tasks and reasoners. Figure 6 shows that MetaCtrl maintains a selective control policy across 12 reasoner and benchmark combinations. Continue remains the dominant action, accounting for 61.9% to 86.8% of decisions, while Stop Think is used only 1.5% to 7.8% of the time. Skip Think is more frequent than Stop Think in every setting and is the most common intervention action in 10 of the 12 combinations. These statistics indicate that the controller more frequently applies local reasoning compression than explicit termination at individual control points. At the same time, intervention rates vary systematically across tasks. GPQA elicits substantially more intervention than AIME 2024 and AMC across all three reasoners, while MATH 500, despite being relatively easier, consistently shows lower Continue rates than AIME 2024 and AMC. This suggests that the controller is not governed by benchmark difficulty alone, but adapts to task specific reasoning patterns, intervening more when shorter or locally compressed reasoning may suffice and preserving longer reasoning when sustained computation remains useful. Overall, these patterns indicate that MetaCtrl largely preserves the reasoner’s native computation while adapting both the degree and form of intervention across tasks and reasoners.

Control demand is not explained by task difficulty or trajectory length alone. GPQA consistently induces the highest intervention rates across all three reasoners, reaching 30.8%, 34.2%, and 38.1%, compared with 20.9%, 18.1%, and 13.2% on AIME 2024. Importantly, this difference cannot be attributed simply to longer reasoning trajectories. Although all three reasoners receive substantially more controller decisions per problem on AIME 2024 than on GPQA, the controller intervenes considerably less often on AIME. These results suggest that control demand is not determined by task difficulty or trajectory length alone. Instead, the learned policy appears sensitive to the evolving reasoning trajectory, preserving continued computation in some settings while intervening more frequently in others.

Control demand depends on the model-task pair, not model scale alone. Increasing the reasoner size from Qwen3-8B to Qwen3-14B does not produce a consistent change in intervention rate. It decreases from 18.1% to 13.2% on AIME 2024, but increases from 34.2% to 38.1% on GPQA and from 15.5% to 19.0% on AMC. Meanwhile, MATH-500 remains nearly unchanged across DeepSeek-R1-7B, Qwen3-8B, and Qwen3-14B, with intervention rates of 24.1%, 26.0%, and 24.5%, respectively. These non-monotonic trends suggest that control demand depends on the interaction between the reasoner and the task rather than on model scale alone. This motivates adaptive external control, since a fixed token budget or uniform stopping rule cannot directly account for such variation across reasoner-task combinations.

Table 8: Ablation studies on intervention position, action space, and controller initialization. The reasoning model is DeepSeek-R1-Distill-Qwen-7B.
<table><tr><td colspan="2">Ablation Setting</td><td colspan="3">MATH-500</td><td colspan="3">AMC</td><td colspan="3">AIME2024</td><td colspan="2">OlympiadBench</td><td colspan="3">GPQA Diamond</td></tr><tr><td>Component</td><td>Setting</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td><td>Acc. Len.</td><td>Reduc.</td><td>Acc.</td><td>Len.</td><td>Reduc.</td></tr><tr><td colspan="2">Vanilla (No Controller)</td><td>85.2</td><td>2857</td><td></td><td>73.5</td><td>7279</td><td></td><td>50.0</td><td>10570</td><td></td><td>48.6 8347</td><td></td><td>32.8</td><td>5349</td><td></td></tr><tr><td rowspan="2">Intervention Position</td><td>At Transition Words</td><td>71.2</td><td>999</td><td>-65.0%</td><td>47.7</td><td>1236</td><td>-83.0%</td><td>22.7</td><td>2828</td><td>-73.2%</td><td>32.4 1315</td><td>-84.2%</td><td>35.5</td><td>1069</td><td>-80.0%</td></tr><tr><td>Every 300 Tokens</td><td>86.0</td><td>3638</td><td>+27.3%</td><td>68.0</td><td>9351</td><td>+28.5%</td><td>41.3</td><td>15500</td><td>+46.6% 50.7</td><td>9587</td><td>+14.9%</td><td>29.0</td><td>15362</td><td>+187.2%</td></tr><tr><td>Intervention Action</td><td>Add Check Action</td><td>87.2</td><td>2025</td><td>-29.1%</td><td>70.8</td><td>4000</td><td>-45.0%</td><td>42.7</td><td>8793 -16.8%</td><td>51.1</td><td>3888</td><td>-53.4%</td><td>38.2</td><td>3746</td><td>-30.0%</td></tr><tr><td>Controller Model</td><td>Initialized from Qwen3-1.7B</td><td>88.0</td><td>2295</td><td>-19.7%</td><td>72.5</td><td>4850</td><td>-33.4%</td><td>41.3</td><td>10179</td><td>-3.7%</td><td>51.6 4946</td><td>-40.7%</td><td>40.9</td><td>4600</td><td>-14.0%</td></tr><tr><td></td><td>MetaCtrl</td><td>88.0</td><td>1644</td><td>-42.5%</td><td>74.7</td><td>3036</td><td>-58.3%</td><td>56.7</td><td>5899</td><td>-44.2% 52.3</td><td>2965</td><td>-64.5%</td><td>42.4</td><td>2234</td><td>-58.2%</td></tr></table>

Table 9: End-to-end inference latency under Vanilla Reasoning and MetaCtrl Controlled Reasoning with different GPU memory allocations. All experiments are conducted on NVIDIA H20 GPUs. Mem. Frac. (R+C) denotes the configured static GPU memory fractions for the reasoner (R) and controller (C), respectively. Vanilla Reasoning uses a single GPU for the reasoner. For MetaCtrl with separate GPUs, the reasoner and controller run on two dedicated GPUs. For MetaCtrl with a shared GPU, the reasoner and controller are colocated on the same GPU.
<table><tr><td rowspan="2">Reasoner</td><td colspan="2">MATH500</td><td colspan="2">AMC</td><td colspan="2">GPQA Diamond</td></tr><tr><td>Avg. Time (s)</td><td>Mem. Frac. (R+C)</td><td>Avg. Time (s)</td><td>Mem. Frac. (R+C)</td><td>Avg. Time (s)</td><td>Mem. Frac. (R+C)</td></tr><tr><td colspan="7">Vanilla Reasoning</td></tr><tr><td>Qwen3-8B</td><td>31.62</td><td>0.90 + 0.00</td><td>74.93</td><td>0.90 + 0.00</td><td>81.28</td><td>0.90 + 0.00</td></tr><tr><td>Qwen3-14B</td><td>45.18</td><td>0.90 + 0.00</td><td>100.94</td><td>0.90 + 0.00</td><td>95.79</td><td>0.90 + 0.00</td></tr><tr><td colspan="7">MetaCtrl Controlled Reasoning (Two Separate GPUs)</td></tr><tr><td>Qwen3-8B</td><td>15.44</td><td>0.90 + 0.60</td><td>43.02</td><td>0.90 + 0.60</td><td>23.96</td><td>0.90 + 0.60</td></tr><tr><td>Qwen3-14B</td><td>20.63</td><td>0.90 + 0.60</td><td>39.57</td><td>0.90 + 0.60</td><td>36.74</td><td>0.90 + 0.60</td></tr><tr><td colspan="7">MetaCtrl Controlled Reasoning (Two Separate GPUs)</td></tr><tr><td>Qwen3-8B</td><td>16.21</td><td>0.90 + 0.40</td><td>38.39</td><td>0.90 + 0.40</td><td>23.77</td><td>0.90 + 0.40</td></tr><tr><td>Qwen3-14B</td><td>20.98</td><td>0.90 + 0.40</td><td>38.79</td><td>0.90 + 0.40</td><td>36.95</td><td>0.90 + 0.40</td></tr><tr><td colspan="7">MetaCtrl Controlled Reasoning (Two Separate GPUs)</td></tr><tr><td>Qwen3-8B</td><td>15.71</td><td>0.90 + 0.18</td><td>37.32</td><td>0.90 + 0.18</td><td>24.25</td><td>0.90 + 0.18</td></tr><tr><td>Qwen3-14B</td><td>21.21</td><td>0.90 + 0.18</td><td>46.93</td><td>0.90 + 0.18</td><td>39.69</td><td>0.90 + 0.18</td></tr><tr><td colspan="7">MetaCtrl Controlled Reasoning (One Shared GPU)</td></tr><tr><td>Qwen3-8B</td><td>15.95</td><td>0.72 + 0.18</td><td>37.83</td><td>0.72 + 0.18</td><td>21.74</td><td>0.72 + 0.18</td></tr><tr><td>Qwen3-14B</td><td>20.79</td><td>0.72 + 0.18</td><td>48.16</td><td>0.72 + 0.18</td><td>33.33</td><td>0.72 + 0.18</td></tr></table>

Overall, these results provide behavioral evidence for the central motivation of adaptive, budget-free reasoning control: useful computation should be regulated according to the evolving reasoning state, rather than predetermined from model size, task difficulty, or elapsed computation.

## E.3 MORE ABLATION STUDIES

Controller design. Table 8 examines the effects of intervention timing, action space, and controller initialization. Heuristic intervention schedules lead to substantially different tradeoffs. Intervening at transition words produces aggressive compression of 65.0% to 84.2%, but substantially reduces accuracy on most benchmarks, whereas intervening every 300 tokens increases generation length on all five benchmarks and underperforms MetaCtrl in accuracy. These results support conditioning intervention timing on the evolving reasoning state rather than using fixed heuristics. Adding a Check action also degrades both accuracy and length reduction relative to the original action space across all benchmarks, suggesting that a larger action space does not necessarily improve control. Similarly, initialization from Qwen3-1.7B results in weaker accuracy and substantially less compression than the default controller on most benchmarks. Overall, the full MetaCtrl design provides the most consistent balance between accuracy and generation length, supporting adaptive intervention timing, a compact action space, and the chosen controller initialization.

![](images/dd8a6ab6cd08d0792cb42139d98dc4ab98bfc2f71652ef5bd85d265973a9942e.jpg)  
(a) Training reward throughout GRPO optimization.

![](images/32a2076b7deb68fe08ddb6abd5fa4d0047deb6841b8a54846a31adc47fefc2e5.jpg)  
(b) Validation accuracy and average total generation tokens during training.  
Figure 7: Training dynamics of the MetaCtrl. (a) Training reward during GRPO optimization. (b) Validation accuracy and average total generation tokens throughout training. The reward and accuracy progressively improve and stabilize, while generation length converges to a relatively stable range, indicating that the controller learns to regulate reasoning computation without uniformly compressing reasoning trajectories.

## F THE ANALYSIS OF LATENCY AND COMPUTATION COST OF METHODS

End to end latency. Table 9 shows that MetaCtrl consistently reduces end to end inference latency across both reasoners and all three benchmarks. When the reasoner and controller are placed on two separate GPUs, MetaCtrl reduces mean latency by 42.6% to 70.8% relative to Vanilla Reasoning across the evaluated memory allocations. This indicates that the reduction in reasoner computation is large enough to offset the additional controller inference and communication overhead. Moreover, reducing the controller memory fraction from 0.60 to 0.18 largely preserves the latency improvement, suggesting that MetaCtrl does not require a heavily provisioned controller to achieve substantial acceleration. The shared GPU setting provides further evidence of practical efficiency. When the reasoner and controller share a single H20 with memory fractions of 0.72 and 0.18, respectively, MetaCtrl reduces latency by 49.5% to 73.3% across the six reasoner and benchmark combinations, with an average reduction of 57.3%. On GPQA Diamond, for example, latency decreases from 81.28s to 21.74s for Qwen3 8B and from 95.79s to 33.33s for Qwen3 14B. These results show that reductions in reasoning computation translate into substantial wall clock speedups, including when the reasoner and controller are colocated on a single GPU. MetaCtrl does introduce an additional controller model, which increases memory requirements and system complexity compared with Vanilla Reasoning. Nevertheless, the shared GPU results show that this overhead can be reduced substantially while retaining large latency gains.

## G TRAINING DYNAMICS AND EFFICIENCY

Stable policy improvement during training. We analyze the optimization dynamics of our MetaCtrl by tracking the training reward throughout GRPO training (Figure 7a). The reward increases rapidly during the early stage of training and continues to improve despite local fluctuations, reaching a substantially higher level after roughly 400 training steps. It subsequently stabilizes within a relatively narrow range, indicating that the controller progressively learns more effective intervention policies rather than relying on transient high-reward behaviors. The sustained improvement and stable late-stage reward further suggest that the proposed optimization objective provides a reliable learning signal for regulating the reasoning process.

![](images/3c3cbe86a658cab71fb57760afb701aa777b90490175ecd035f1fafafafefcf9.jpg)

![](images/0931a9e4f3166172b680470c8466730bd75f560c0fa153c288b70525baf65365.jpg)

![](images/7a7704bf6b2e21c7062b8573234449d98f0296ffbcc192940ab4112e2efab00a.jpg)

![](images/81c200339a4963e00e11382db9b12cd3959c394e620a41050040c87d2b44f70b.jpg)  
Figure 8: Evolution of the controller action distribution during training. The learned policy gradually shifts toward CONTINUE, while the frequencies of FAST-THINK, SKIP-THINK, and STOP-THINK decrease and stabilize at lower levels. This trend suggests that the controller progressively learns to intervene more selectively rather than frequently modifying the reasoner’s trajectory.

Balancing accuracy and reasoning length. We track validation accuracy and mean reasoner generation length throughout training in Figure 7b. Validation accuracy increases from approximately 79% at the beginning of training to around 87% in later stages, while mean reasoning length rises from roughly 1.1K to around 1.55K tokens. The two metrics exhibit broadly aligned trends during optimization, indicating that training does not collapse to a trivial policy that simply minimizes rea soning length. Instead, the controller learns to retain additional computation when it is useful for correctness, rather than aggressively compressing reasoning at the expense of accuracy.

Evolution of the controller policy. We further investigate how the learned control strategy evolves during training by tracking the distribution of the four controller actions in Figure 8. At the beginning of training, the controller frequently modifies the reasoner’s trajectory through FAST-THINK, SKIP-THINK, and STOP-THINK, while CONTINUE constitutes only a relatively small fraction of the decisions. As optimization proceeds, the action distribution changes markedly: the proportion of CONTINUE steadily increases and eventually dominates the policy, whereas FAST-THINK and SKIP-THINK decrease substantially and STOP-THINK remains comparatively infrequent. Notably, the intervention actions do not vanish completely, but converge to small yet non-zero frequencies. This behavior suggests that the controller does not learn to indiscriminately shorten the frozen reasoner’s reasoning process. Instead, it progressively learns to preserve the reasoner’s native trajectory in most cases and intervene selectively when modifying the ongoing reasoning is useful. Together with the improvements in reward and accuracy and the stabilization of generation length, these dynamics indicate the emergence of a stable and selective computation policy: the controller learns not only how to intervene, but also when intervention is unnecessary.

## H FUTURE WORK

A promising direction is to extend MetaCtrl to a collaborative multi-reasoner system, where multi ple reasoning models constitute the object-level and a shared meta-level controller coordinates the overall reasoning process. For each problem, the controller could first perform joint model routing and initial effort allocation, selecting which reasoner should begin solving the problem and under which reasoning effort configuration.

As reasoning unfolds, the controller could continuously monitor the evolving explicit reasoning trace and decide whether to let the current reasoner proceed, intervene through lightweight textual guidance to regulate its subsequent reasoning, or selectively invoke a stronger model for targeted guidance, verification, or correction. Routine reasoning could therefore remain with smaller and less expensive models, while stronger models are reserved for stages where additional capability is most valuable. The resulting system would jointly control computation at two levels: across models, by deciding which reasoner should handle each stage of the solution process, and within a selected reasoner, by regulating how its ongoing reasoning should proceed through trajectory-level interventions.

This setting raises an important resource allocation problem: when a difficult intermediate state should be handled by further regulating the current reasoner, and when it should instead be escalated to a more capable model, selectively invoked at critical stages to provide targeted guidance or correction. Learning this trade-off online could reduce the need for users to manually choose among model families and reasoning-effort settings, while avoiding the cost of running the strongest model throughout the entire solution process. More broadly, such a system would extend metacognitive control from regulating a single reasoning trajectory to coordinating capability and computation across a collaborative reasoning system, with the goal of achieving a desired level of reliability at lower end-to-end inference cost.

## I CASE STUDY

## I.1 PREVENTING POST-SOLUTION OVERTHINKING

As shown in Figures 9–18, across five representative reasoning problems, vanilla reasoning often continues with redundant verification or repeated hypothesis exploration after the key solution has already been established. In contrast, MetaCtrl adaptively regulates the evolving trajectory, preserving necessary derivations while accelerating or terminating further computation when sufficient evidence has accumulated. The consistent preservation of correct answers across these cases illustrates how MetaCtrl reduces diverse forms of overthinking without aggressively truncating reasoning.

## I.2 REDUCING ERRORS FROM PROLONGED REASONING

As shown in Figures 19–24, these cases further illustrate that longer reasoning is not necessarily more reliable. Vanilla reasoning can become trapped in prolonged hypothesis exploration or increasingly complex derivations, where additional deliberation may compound inconsistencies and ultimately lead to incorrect answers. In contrast, MetaCtrl regulates how the frozen reasoner proceeds through lightweight interventions, encouraging more direct reasoning and terminating unnecessary deliberation once a sufficient solution path has emerged. In these examples, such controlled trajectories are not only substantially shorter but also reach the correct answers, suggesting that simpler and more focused reasoning can sometimes be more reliable than unconstrained overthinking.

![](images/f9cfa863ea1d63138615fd33321b5fc8b9e358851c88845aaa28e219fc060c5b.jpg)  
Figure 9: Vanilla reasoning on a MATH-500 example. The model obtains the correct answer but continues with repetitive verification and alternative derivations, leading to substantial overthinking and 7,043 reasoning tokens.

![](images/305a8a79573f281a232843384baeb243b8460a522a2fdda05f1b3113a7c98094.jpg)  
Figure 10: MetaCtrl reasoning on the MATH-500 example. Both methods arrive at the correct answer, while MetaCtrl dynamically regulates the reasoning trajectory through lightweight interventions and terminates once the solution is sufficiently established, reducing the reasoning length from 7,043 to 957 tokens.

![](images/56d731c152ac331df7c68020d409037ec430a2f85bcb35879e1bc6931b8020ce.jpg)  
Figure 11: Vanilla reasoning on a MATH-500 example. The model reaches the correct answer early but continues with alternative counting strategies, repeated verification, and additional sanity checks, resulting in 2,479 reasoning tokens.

![](images/599c6e8e3ec21dc7d47e9f8bf676236f94c6334cc4e8b80f3db2efada79f55d6.jpg)  
Figure 12: MetaCtrl reasoning on the MATH-500 example. After identifying that all valid handshakes occur between the two groups, the controller skips unnecessary intermediate reasoning and terminates once sufficient information for the solution has been established, reducing the reasoning length from 2,479 to 409 tokens while preserving the correct answer.

![](images/c771d8cc36293290b62228329487840c8fbe15fafc58b4e83d72e092f7d669a6.jpg)  
Figure 13: Vanilla reasoning on a GPQA example. Although the model eventually selects the correct answer, it repeatedly revisits competing explanations and re-evaluates previously considered options, resulting in 2,305 reasoning tokens.

![](images/4ba3eb4812d59491abcd4777d933258d1a35ae0541b89de1ccfbb6e6af70b89f.jpg)  
Figure 14: MetaCtrl reasoning on the same GPQA example. The controller progressively regulates the reasoning trajectory through Fast Think, Skip Think, and Stop Think interventions, reducing repeated hypothesis exploration and reaching the same correct answer with 707 reasoning tokens.

![](images/c0abddd9f51f26c64062e8b850454af0270eff8076d8ba88ddede2d42374e835.jpg)  
Figure 15: Vanilla reasoning on an AIME 2024 example. The model derives the correct composition-based counting argument but continues to revisit the same calculation through repeated verification and alternative checks, resulting in 6,637 reasoning tokens.

![](images/ad24fad31bda717dd875b8edcd01f51f3972eafa08ad04420006af7a6eb49685.jpg)  
Figure 16: MetaCtrl reasoning on the same AIME 2024 example. The controller dynamically accelerates the reasoning trajectory and terminates once the composition-based counting argument is complete, reaching the same correct answer with 1,789 reasoning tokens.

![](images/9c85391578a47fe82987a578ef1052d1e9189a2257b5a0d6df2a6882d447812f.jpg)  
Figure 17: Vanilla reasoning on an OlympiadBench example. The model correctly derives the unique solution but continues with repeated substitution checks, alternative derivations, and unnecessary uniqueness verification, resulting in 6,064 reasoning tokens.

![](images/1007a8f78b9d22854b8e35153bdb5afb525b5b8fea446d39ed93ffdc9e476114.jpg)  
Figure 18: MetaCtrl reasoning on the same OlympiadBench example. After the key algebraic reduction establishes the unique solution, the controller terminates further deliberation, reaching the same correct answer with 1,463 reasoning tokens.

![](images/80fec675dfa13964477f2d1e5093b6e75a89c1100d382ed43b6af77f8b5f91f4.jpg)  
Figure 19: Vanilla reasoning on an AMC example. The model enters an inconsistent trigonometric detour and continues reasoning despite recognizing contradictions in its intermediate derivation, ultimately producing an incorrect answer after 8,535 reasoning tokens.

![](images/3ceeeb0b36ea469452a5185ca963996301cbde7aae185b2a2cf77d70ace62895.jpg)  
Figure 20: MetaCtrl reasoning on the same AMC example. By dynamically regulating how the frozen reasoner proceeds, MetaCtrl follows a more direct derivation and avoids prolonged unproductive deliberation, reaching the correct answer with 2,479 reasoning tokens.

![](images/3d66d4fbd8d23ae1dd20cc93a48b3c4912eaf5820f3828a807f666f64a44f324.jpg)  
Figure 21: Vanilla reasoning on a GPQA chemistry example. The model repeatedly revisits conflicting hypotheses about solvation, basicity, and nucleophilicity, and an early mischaracterization of the alkoxide contributes to a prolonged inconsistent trajectory, ultimately yielding an incorrect answer after 12,806 reasoning tokens.

![](images/b3a6daaf74a32a168621f0c75ae3961b622dd53318e0c7f8d62b692058d43148.jpg)  
Figure 22: MetaCtrl reasoning on the same GPQA example. By dynamically regulating how the frozen reasoner proceeds, MetaCtrl substantially reduces prolonged hypothesis exploration and reaches the correct answer with 2,001 reasoning tokens.

![](images/7c9c95cdd1a893a855ea74faf9aed232f206a6a719d3dfd803ba7b5cfe5de12a.jpg)  
Figure 23: Vanilla reasoning on a GPQA physics example. The model repeatedly interprets the “first two minima” as the symmetric first minima on opposite sides of the central maximum, leading to prolonged verification of the same interpretation and ultimately an incorrect answer after 3,702 reasoning tokens.

![](images/819a641316d1d8af834c1191f32d6b412bf96526de58b41cdc335987937a087a.jpg)  
Figure 24: MetaCtrl reasoning on the same GPQA example. The controlled trajectory revisits the interpretation of the diffraction minima and uses the first two zeros of the Airy pattern, reaching the correct answer with 3,117 reasoning tokens.