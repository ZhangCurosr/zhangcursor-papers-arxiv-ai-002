# PaceReasoner: Label-Free Learning to Answer Briefly from Long Reasoning

Changguo Jia<sup>1</sup>\* Zhiyou Xiao<sup>1</sup>\* Xinbei Ma<sup>2</sup>\* Geng Chen<sup>1</sup> Weiming Zhang<sup>3</sup> Yiming Xu<sup>1</sup> Tianqi Zhao<sup>4</sup> Minghui Zhou<sup>1†</sup>

<sup>1</sup>Peking University <sup>2</sup>National University of Singapore <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Zhongguancun Laboratory

jiachangguo@stu.pku.edu.cn xiaozhiyou@stu.pku.edu.cn xinbei.ma@nus.edu.sg zhmh@pku.edu.cn

## Abstract

Extended chain-of-thought reasoning improves problem-solving performance but increases inference cost and latency, motivating more accurate answers with shorter reasoning. We introduce PaceReasoner, a label-free test-time reinforcement learning framework that uses a model’s LONG responses to improve its SHORT-mode accuracy. Both modes share a trainable policy, creating a need for reliable supervision, sufficient SHORT training samples, and continued LONG responses to guide learning. PaceReasoner combines all-response and LONG-only answer consensus through progressive, confidence-weighted supervision. A behavior reward gradually shifts from encouraging SHORT exploration to rewarding agreement with LONG answers, while a ratio controller regulates the balance between modes. Across three model backbones and four benchmarks, PaceReasoner improves average SHORT accuracy over the baselines by 7.4–13.5 percentage points, with fewer than 25 reasoning tokens per response on average. Ablations support the key components, and crossdataset results demonstrate transfer beyond the adaptation set.

## 1 Introduction

Large reasoning models improve problem-solving performance by devoting additional test-time computation to extended chain-of-thought reasoning (Guo et al., 2025; Qwen Team, 2025). However, longer reasoning increases token usage and response latency, while unnecessarily extended reasoning may yield limited additional gains (Chen et al., 2025; Ma et al., 2025). For simpler questions or applications with latency or cost constraints, these gains may not justify the additional computation. Existing approaches manage this tradeoff by learning when to invoke explicit reasoning (Fang et al., 2025; Zhang et al., 2025; Kim et al., 2026), compressing reasoning traces through selfdistillation (Sang et al., 2026), or controlling and allocating reasoning budgets (Li et al., 2026a; Lin et al., 2026). These approaches primarily optimize the reasoning cost rather than the problem-solving capability of a low-cost mode itself. Therefore, they motivate a complementary question: how can we improve the model’s problem-solving accuracy when it operates in shorter reasoning mode?

Our starting point is the accuracy gap between LONG and SHORT modes of the same model, where zero-shot LONG achieves higher average accuracy than SHORT across all three evaluated backbones. This gap suggests that the model’s own LONG responses may provide useful supervision for improving SHORT without reference answers. Motivated by this observation, we formulate the problem as label-free test-time adaptation: given a set of test questions without access to their reference answers, our goal is to improve the problemsolving accuracy of SHORT mode while keeping its reasoning brief.

We introduce PaceReasoner, a test-time reinforcement learning framework that jointly optimizes LONG and SHORT within a shared policy, using LONG answers to guide SHORT learning without reference answers. This approach faces three challenges. First, supervision reliability: LONG responses may disagree or converge on an incorrect answer. Second, SHORT learning opportunities: the initially stronger LONG mode may dominate generation, limiting the samples available for improving SHORT. Third, continued LONG supervision: encouraging SHORT too strongly may leave too few LONG responses to support further learning.

PaceReasoner addresses these challenges through coordinated answer and behavior rewards. To improve supervision reliability, progressive supervision combines answer consensus across all responses with consensus from LONG responses alone, while confidence weighting reduces reliance on uncertain LONG consensus. To provide SHORT learning opportunities, a behavior reward initially encourages SHORT exploration and gradually shifts toward rewarding agreement with LONG answers. To preserve continued LONG supervision, a ratio controller adjusts this reward according to the proportion of SHORT responses in the batch, counteracting mode imbalance. Finally, Group Relative Policy Optimization (GRPO) updates the complete responses from both modes, including mode selection and subsequent generation.

Our evaluation on three model backbones and four benchmarks shows that PaceReasoner improves average SHORT-mode accuracy over TTRL by 7.4–13.5 percentage points, while using fewer than 25 reasoning tokens per response on average for each backbone. Our further ablation studies support the contributions of progressive supervision and behavior control, while cross-dataset experiments show that SHORT-mode improvements transfer beyond the adaptation dataset.

In summary, our contributions are threefold:

• We introduce PaceReasoner, which improves short reasoning using answer supervision from the same model’s LONG responses on unlabeled target questions.

• We coordinate progressive consensus supervision, an exploration-to-agreement reward, and ratio control to address noisy supervision, insufficient SHORT coverage, and loss of LONG evidence during joint training.

• Across three models and four benchmarks, PaceReasoner improves average SHORT accuracy over baselines at fewer than 25 reasoning tokens on average. Ablations support each key component, while cross-dataset results demonstrate transfer beyond the adaptation set.

## 2 Related Work

## 2.1 Reasoning Compression and Distillation

Length-constrained reinforcement learning and pruning improve the accuracy–cost trade-off (Aggarwal and Welleck, 2025a; Hou et al., 2025a). Progressive CoT removal internalizes intermediate steps (Deng et al., 2024), while on-policy distillation learns from teacher distributions on student-generated trajectories (Agarwal et al., 2024). Teacher conditioning includes privileged solutions in OPSD (Zhao et al., 2026a), conciseness instructions in CRISP (Sang et al., 2026), and hidden reasoning prefixes in Masked Self-Distillation (Kalwar et al., 2026). Alternatively, LONG answers guide learning without distribution matching. PaceReasoner supervises SHORT using sequencelevel rewards from LONG answer consensus.

Adaptive Reasoning. Qwen3 supports hybrid reasoning (Qwen Team, 2025). Thinkless decouples mode-selection and response optimization, while AdaptThink balances mode exploration through importance sampling (Fang et al., 2025; Zhang et al., 2025). RPO optimizes only the root token (Kim et al., 2026). Greater SHORT usage need not improve capability. Mode allocation governs supervision: LONG dominance restricts SHORT updates, whereas SHORT dominance reduces LONG evidence. PaceReasoner coordinates mixed-mode sampling to sustain SHORT learning and LONG supervision.

## 2.2 Label-Free Test-Time Adaptation

Self-consistency aggregates answers at inference (Wang et al., 2023), and TTRL converts majority agreement into reinforcement-learning rewards (Zuo et al., 2025); RENT and Intuitor use intrinsic confidence (Prabhudesai et al., 2025; Zhao et al., 2026b). U-OPSD distills disagreeing completions using consensus-supported pseudo-solutions (Li et al., 2026b), whereas TTPO distills agreeing rollouts and selectively penalizes disagreeing ones (Wang et al., 2026). Both improve nonthinking performance; TTPO studies thinkingto-non-thinking transfer. Joint adaptation couples supervision and mode allocation: updates change LONG evidence, while mode frequencies affect voting pools and training opportunities. Allresponse consensus offers broader coverage but risks domination by weaker SHORT predictions; LONG-only consensus isolates deliberative evidence but may be sparse or incorrect. PaceReasoner combines progressive, confidence-weighted dual-consensus rewards with mode regulation to sustain supervision during joint adaptation.

## 3 Problem Definition and Preliminaries

## 3.1 Test-time Adaptation

Test-time adaptation aims to update a policy at inference time without additional ground-truth supervision. Given an initial policy $\pi _ { \theta _ { 0 } }$ and a set of test questions $\mathcal { D } = \{ q _ { j } \} _ { j = 1 } ^ { M }$ , the adaptation stage updates the policy to π<sub>θ</sub> using rewards derived from its own inference. After the learning stage, we evaluate the updated policy on the same questions D, where the generates final answers are scored against the reference answer. In this study, our goal is to improve performance in the short chain-ofthought setting, defined in the following section.

## 3.2 Reasoning Mode

Chain-of-thought largely facilitates the reasoning capabilities of an LLM. Given a question q, an LLM with parameters θ generates reasoning text z followed by a final answer a. We denote the resulting output by $o = \left( z , a \right)$

$$
o = ( z , a ) \sim \pi _ { \theta } ( \cdot \mid q ) .\tag{1}
$$

To enable efficient reasoning, we use system prompts to elicit two reasoning modes, LONG and SHORT. The LLM is prompted either to choose between the two reasoning modes or to follow a specified mode, and to format its response accordingly.

(1) In LONG mode, the response contains explicit reasoning followed by the final answer:

$$
< r e a s o n > z < / r e a s o n > < a \cap s w e r > a < / a n s w e r > .
$$

(2) In SHORT mode, the response gives the final answer directly, without an explicit reasoning trace:

$$
< \mathsf { n o t } > < / \mathsf { n o t } > < \mathsf { a n s w e r } > a < / \mathsf { a n s w e r } > .
$$

Thus, under the specified output format, $z = \emptyset$ in SHORT mode. The full prompt used during adaptation is provided in Appendix A.5. Therefore, the reasoning policy is denoted as

$$
( z , a ) \sim \pi _ { \theta } ( \cdot \mid q , m ) ,
$$

where L and S denote LONG and SHORT and both modes share the same policy parameters θ.

We define the reasoning cost of a response as the number of generated reasoning tokens before the final answer, excluding answer tokens. Let $\ell ( z )$ denote this count. For mode $m ,$ , the expected reasoning cost is

$$
C _ { m } = \mathbb { E } _ { ( z , a ) \sim \pi _ { \theta } ( \cdot | q , m ) } [ \ell ( z ) ] .\tag{2}
$$

## 3.3 Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm for LLM posttraining that has been shown to be effective for reasoning tasks with outcome-based reward (Shao et al., 2024), where only the answer a is compared with a verifiable reference, without requiring supervision for the intermediate reasoning process.

Given a question q, GRPO samples a group of K responses $\{ o _ { i } \} _ { i = 1 } ^ { K }$ from the policy. Each response receives a scalar reward $r _ { i }$ , and its advantage is computed relative to the other responses in the group, used to update the policy optimization,

$$
\widehat { A } _ { i } = \frac { r _ { i } - \mathrm { m e a n } _ { j } ( r _ { j } ) } { \mathrm { s t d } _ { j } ( r _ { j } ) + \epsilon } .\tag{3}
$$

In our test-time adaptation setting, the reward $r _ { i } ,$ is derived from the policy’s LONG mode of reasoning, transforming self-generated signals to improve token-efficient reasoning of SHORT mode.

## 4 PaceReasoner

## 4.1 Method Overview

Building on GRPO-based test-time adaptation, the primary idea of PaceReasoner is to construct a reward that encourages SHORT to generate the same answer as LONG, enhancing short reasoning of the policy model. We propose a novel reward that uses answer consensus from LONG. As is shown in Figure 1, for each question, the current policy performs K times of rollout, where it chooses a reasoning mode and follows the format to generate K responses. Since LONG always reasons to superior answer accuracy than SHORT, we use the predicted answer of LONG’s consensus to align SHORT. Another key is the mode-choosing behavior: because both modes are updated by the same policy, the training requires enough SHORT for learning while preserving enough LONG as a source of supervision. Therefore, we have to prevent the reasoning mode from collapsing into any single one.

![](images/0cc49568b75d1de3a6b1022934fcd2593891044531907cba964b0f19d91e0ab2.jpg)  
Figure 1: Overview of PaceReasoner. LONG and SHORT responses supply all-response and LONG-only answer consensus. The accuracy-reward block implements progressive supervision with the additional confidence weight in Eq. (7). SHORT activation and ratio control construct the behavior reward; format, answer, and behavior rewards jointly drive policy optimization.

The reward for $o _ { i } = ( m _ { i } , z _ { i } , a _ { i } )$ is defined as

$$
r _ { i } = r _ { \mathrm { f m t } } ( o _ { i } ) + r _ { \mathrm { a c c } } ( q , a _ { i } , t ) + r _ { \mathrm { b e h } } ( o _ { i } , t ) .\tag{4}
$$

The format reward is 1 for a valid LONG or SHORT output and 0 otherwise. The answer reward supplies pseudo-supervision, while the behavior reward regulates reasoning mode exploration and prevalence.

## 4.2 Answer Reward: Long-CoT Consensus

Answer consensus. Let $A _ { L } ( q )$ contain those generated in LONG mode, whose majority voting yields a pseudo-label $a _ { L } ^ { * }$ . For response $o _ { i } .$ , let $a _ { i }$ denote its extracted final answer, and let $a _ { L } ^ { * }$ denote the majority-voted answer from the corresponding LONG answer pool. We define the answeragreement reward $R ( a _ { i } , a _ { L } ^ { * } )$ to be 1 if the taskspecific answer checker judges $a _ { i }$ equivalent to $a _ { L } ^ { * }$ and 0 otherwise. If either answer is missing, we set $R ( a _ { i } , a _ { L } ^ { * } ) = 0$

A<sub>L</sub>(q) voting isolates evidence produced through explicit deliberation, yet remains vulnerable to limited coverage, disagreement, and shared errors. We therefore consider All-response voting to provide feedback when LONG samples are scarce, using all extractable answers $\mathcal { A } _ { \mathrm { a l l } } ( \boldsymbol { q } )$ and the agreement $R ( a _ { i } , a _ { a l l } ^ { * } )$ with their majorityvoting label $a _ { \mathrm { a l l } } ^ { \ast }$ as a secondary supervision.

Progressive supervision. With training progress $s _ { t } = t / T$ , where $T$ is the total number of steps, the underlying supervision curriculum is

$$
\begin{array} { r l } { r _ { \mathrm { a c c } } ^ { \mathrm { p r o g } } = ( 1 - s _ { t } ) R ( a _ { i } , a _ { \mathrm { a l l } } ^ { * } ) } & { } \\ { + s _ { t } R ( a _ { i } , a _ { L } ^ { * } ) . } \end{array}\tag{5}
$$

Early updates rely more on all-response consensus, while later updates increasingly use LONG evidence. This schedule expresses a curriculum rather than a guarantee that LONG becomes correct as training proceeds.

Confidence weighting. Without ground-truth labels erroneous majority-vote pseudo-labels can steer policy updates toward incorrect answers (Zuo et al., 2025). We therefore weight answeragreement reward $R ( a _ { i } , a _ { L } ^ { * } )$ to reduce reliance on pseudo-labels, allowing varying degrees of supervision based on the “confidence”. Let $N _ { L } = | A _ { L } |$ let $v _ { 1 } , v _ { 2 }$ be the largest and second-largest answer counts, and set $v _ { 2 } = ~ 0$ when only one distinct answer occurs. For a minimum sample threshold m $\geq 1$ , define

$$
c _ { L } ( q ) = \left\{ \begin{array} { l l } { ( v _ { 1 } - v _ { 2 } ) / N _ { L } , } & { N _ { L } \geq m , } \\ { 0 , } & { N _ { L } < m . } \end{array} \right.\tag{6}
$$

The final weight is $w _ { q , t } = s _ { t } c _ { L } ( q )$ , giving

$$
\begin{array} { r l } & { r _ { \mathrm { a c c } } ( q , a _ { i } , t ) = ( 1 - w _ { q , t } ) R ( a _ { i } , a _ { \mathrm { a l l } } ^ { * } ) } \\ & { ~ + w _ { q , t } R ( a _ { i } , a _ { L } ^ { * } ) . } \end{array}\tag{7}
$$

Such a weighting makes insufficient LONG samples or a tied vote return the answer reward to allresponse consensus, avoiding rejecting the whole update. When the two pseudo-labels agree, the weight does not change this reward. We use this “consensus strength” as is a reliability “confidence”, not a calibrated correctness probability, and a confidently wrong consensus can still reinforce errors.

![](images/723bf53242dd0c4de489a83e44446fddb2cb3233d97c08fae241e84e2b45b382.jpg)  
Figure 2: Cosine ratio control for SHORT activation. The horizontal coordinate is $x = \pi \rho _ { S }$ and the vertical coordinate is $c _ { \mathrm { r a t i o } }$ . Positive coefficients encourage SHORT exploration below $\rho _ { S } = 0 . 5$ , while negative coefficients suppress further SHORT dominance above it. Pie charts illustrate mode proportions rather than measured distributions.

## 4.3 Behavioral Reward: SHORT Activation and Ratio Control

From exploration to answer agreement. Answer-agreement rewards alone may favor the currently stronger LONG behavior, leaving SHORT with few training opportunities. We introduce a behavior bonus that encourages SHORT exploration early and increasingly emphasizes agreement with LONG consensus later.

Let $I _ { S } ( o _ { i } )$ indicate a valid SHORT response. Its base behavior bonus is

$$
b _ { i } = I _ { S } ( o _ { i } ) \lambda _ { \mathrm { b r } } \big [ ( 1 - s _ { t } ) + s _ { t } R ( a _ { i } , a _ { L } ^ { * } ) \big ] .\tag{8}
$$

A SHORT answer that disagrees with LONG consensus receives the decaying bonus $\lambda _ { \mathrm { b r } } ( 1 - s _ { t } )$ while an agreeing SHORT answer receives $\lambda _ { \mathrm { b r } }$ , and other responses with wrong formats receive zero. This shifts learning from making SHORT available to improving the SHORT answers. Confidence weighting modifies Eq. (7). The behavior bonus retains the time-based curriculum. This time-based schedule the curriculum: first encourages SHORT explorations and then leans to answer agreement.

Regulating the mode mixture. Unrestricted activation can make SHORT dominate sampling, reducing LONG evidence and changing the all-response consensus. We compute the proportion $\rho _ { S }$ of SHORT responses in the current training batch and rescale the bonus:

$$
c _ { \mathrm { r a t i o } } = \cos ( \pi \rho _ { S } ) , \qquad r _ { \mathrm { b e h } } = b _ { i } c _ { \mathrm { r a t i o } } .\tag{9}
$$

When $\rho _ { S } < 0 . 5 $ , the bonus encourages SHORT. Above 0.5, the negative coefficient opposes further SHORT dominance. Figure 2 illustrates this regulation: it does not constrain the learned mixture to remain at 50% but change the direction of the bonus.

Together, the behavioral reward adjusts the ratio of SHORT and LONG reasoning mode, encouraging attempts at SHORT while retaining LONG to provide learning signals, thereby avoiding collapse into either extreme. It provides batch-level feedback against mode imbalance, while answer quality still depends on consensus and policy optimization.

## 4.4 Joint Full-Trajectory Optimization

We use the combined reward in Eq. (4) to optimize complete trajectories from both modes with GRPO. Writing $u _ { i , k } ( \boldsymbol { \theta } )$ for the token probability ratio between the current and rollout policies, the clipped policy objective is

$$
\begin{array} { r } { J _ { \mathrm { c l i p } } ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { K } \underset { i = 1 } { \overset { K } { \sum } } \frac { 1 } { \vert o _ { i } \vert } \underset { k = 1 } { \overset { \vert o _ { i } \vert } { \sum } } \operatorname* { m i n } \left( u _ { i , k } \widehat { A } _ { i } , \right. } \\ { \left. \mathrm { c l i p } ( u _ { i , k } , 1 - \delta , 1 + \delta ) \widehat { A } _ { i } \right) \Bigg ] . } \end{array}\tag{10}
$$

Here $\left| o _ { i } \right|$ counts the full generated trajectory for optimization. Both mode selection and subsequent generation receive updates. Algorithm 1 summarizes the learning procedure: no target-task reference answers, external teacher, or additional verifier is used.

After adaptation, the policy generates a response conditioned on the question and the requested LONG or SHORT mode. SHORT generation uses the adapted policy directly, without requiring LONG responses or consensus computation.

## 5 Experiments

We evaluate whether PaceReasoner improves SHORT-mode accuracy while reducing reasoning cost, and analyze the contribution of each method component.

## 5.1 Experimental Setup

Benchmarks. We evaluate on GSM8K and MATH-500 for mathematical reasoning and verifiable answers (Cobbe et al., 2021; Hendrycks et al., 2021), and on CSQA and GPQA-Diamond for commonsense and scientific reasoning (Talmor et al.,

<table><tr><td rowspan="2">Mode</td><td rowspan="2">Method</td><td colspan="2">CSQA</td><td colspan="2">GPQA-D</td><td colspan="2">MATH-500</td><td colspan="2">GSM8K</td><td colspan="2">Average</td></tr><tr><td>Acc.</td><td>R-tok.</td><td>Acc.</td><td>R-tok.</td><td>Acc.</td><td>R-tok.</td><td>Acc.</td><td>R-tok.</td><td>Acc.</td><td>R-tok.</td></tr><tr><td colspan="10">Qwen3-4B-Base</td><td></td><td></td><td></td></tr><tr><td rowspan="3">LONG</td><td>Zero-shot TTRL</td><td>50.9 80.9</td><td>413.4 159.1</td><td>21.3 31.8</td><td>461.5 386.3</td><td>50.0 79.4</td><td>510.0 501.2</td><td>60.4</td><td>328.8 203.0</td><td>45.7</td><td>428.4 312.4</td></tr><tr><td></td><td>39.9</td><td>158.6</td><td>33.3</td><td></td><td></td><td></td><td>93.1</td><td></td><td>71.3</td><td></td></tr><tr><td>Length-Penalty CRISP</td><td>69.6</td><td>527.7</td><td>32.5</td><td>283.2 699.3</td><td>56.2 47.6</td><td>510.6 759.0</td><td>85.8 53.4</td><td>201.5 186.6</td><td>53.8 50.8</td><td>288.5 543.1</td></tr><tr><td></td><td>PaceReasoner</td><td>80.8</td><td>92.0</td><td>36.9</td><td>168.0</td><td>77.2</td><td>450.2</td><td>93.7</td><td>173.6</td><td>72.2</td><td>221.0</td></tr><tr><td rowspan="4">SHORT</td><td>Zero-shot</td><td>25.8</td><td>482.1</td><td>11.6</td><td>306.0</td><td>31.6</td><td>401.0</td><td>30.9</td><td>286.0</td><td>25.0</td><td>368.8</td></tr><tr><td>TTRL</td><td>78.1 80.7</td><td>11.0</td><td>22.9</td><td>19.8</td><td>19.4</td><td>13.4</td><td>22.9</td><td>19.8</td><td>35.8</td><td>16.0</td></tr><tr><td>Length-Penalty CRISP</td><td>67.1</td><td>12.4</td><td>14.0</td><td>18.3</td><td>26.4</td><td>13.9</td><td>26.7</td><td>15.6</td><td>37.0</td><td>15.1</td></tr><tr><td>PaceReasoner</td><td>80.2</td><td>9.2 12.0</td><td>31.1 36.9</td><td>9.2 51.4</td><td>23.4 39.8</td><td>7.8 14.5</td><td>21.2 39.1</td><td>13.2 19.3</td><td>35.7 49.0</td><td>9.8 24.3</td></tr><tr><td colspan="10">Qwen3-8B-Base</td></tr><tr><td></td><td>Zero-shot TTRL</td><td>77.0 84.0</td><td>149.8 127.4</td><td>29.8 31.3</td><td>514.0</td><td>61.8</td><td>466.0</td><td>72.1</td><td>228.3</td><td>60.2</td><td>339.5</td></tr><tr><td colspan="10"></td></tr><tr><td>LONG</td><td>Length-Penalty CRISP</td><td>61.4 51.9</td><td>133.7 402.5</td><td>32.8 31.5</td><td>319.0 419.5 856.0</td><td>79.8 65.5 49.7</td><td>487.7 463.6 896.3</td><td>94.2 86.8 59.0</td><td>210.5 205.3</td><td>72.3 61.6 48.0</td><td>286.2 305.5 618.3</td></tr><tr><td colspan="10"></td></tr><tr><td></td><td>PaceReasoner Zero-shot</td><td>83.0 39.1</td><td>112.9 100.8</td><td>36.9</td><td>241.9</td><td>77.8</td><td>462.9</td><td>95.0</td><td>318.5 207.9</td><td>73.2</td><td>256.4</td></tr><tr><td>TTRL</td><td></td><td>81.2</td><td>11.0</td><td>24.2 31.3</td><td>100.7 11.0</td><td>34.6 27.2</td><td>206.0</td><td>28.5 8.1</td><td>201.3 13.2</td><td>31.6 37.0</td><td>152.2 12.2</td></tr><tr><td>SHORT</td><td>Length-Penalty</td><td>73.9</td><td>23.8</td><td>5.5</td><td>14.6</td><td>31.0</td><td>13.5 12.1</td><td>19.4</td><td>11.9</td><td>32.5</td><td>15.6</td></tr><tr><td colspan="10"></td></tr><tr><td></td><td>CRISP</td><td>68.4 82.5</td><td>17.1</td><td>28.7</td><td>11.2</td><td>26.8</td><td>13.9</td><td>25.0</td><td>14.8</td><td>37.2</td><td>14.3</td></tr><tr><td></td><td>PaceReasoner</td><td></td><td>11.0</td><td>36.4</td><td>11.0</td><td>37.2</td><td>14.0</td><td>46.0</td><td>13.2</td><td>50.5</td><td>12.3</td></tr><tr><td colspan="10">Llama3.1-8B-Instruct</td></tr><tr><td></td><td>Zero-shot</td><td>62.7 66.9</td><td>175.1</td><td>16.7</td><td>869.6</td><td>31.4</td><td>612.3</td><td>72.1</td><td>321.0</td><td>45.7</td><td>494.5</td></tr><tr><td colspan="10">TTRL</td></tr><tr><td>LONG</td><td>Length-Penalty</td><td>60.4 68.6</td><td>133.6 166.8</td><td>27.9 23.2</td><td>308.9 354.5</td><td>33.0 35.9</td><td>434.2 417.1</td><td>90.5 84.4</td><td>207.9 177.0</td><td>54.6 51.0</td><td>271.2 278.9</td></tr><tr><td>CRISP</td><td></td><td></td><td>127.3</td><td>26.7</td><td>357.3</td><td>25.9</td><td>291.4</td><td>76.7</td><td>156.4</td><td>49.5</td><td>233.1</td></tr><tr><td></td><td>PaceReasoner</td><td>68.7</td><td>145.2</td><td>28.8</td><td>214.6</td><td>33.8</td><td>400.2</td><td>89.6</td><td>225.1</td><td>55.2</td><td>246.3</td></tr><tr><td></td><td>Zero-shot</td><td>60.6</td><td>187.1</td><td>14.4</td><td>790.8</td><td>10.6</td><td>634.7</td><td>11.8</td><td>313.2</td><td>24.4</td><td>481.5</td></tr><tr><td></td><td>TTRL</td><td>66.3</td><td>11.0</td><td>20.1</td><td>30.7</td><td>1.0</td><td>13.0</td><td>7.3</td><td>13.1</td><td>23.7</td><td>17.0</td></tr><tr><td>SHORT CRISP</td><td>Length-Penalty</td><td>63.5 68.6</td><td>16.3</td><td>23.5</td><td>37.6</td><td>6.4</td><td>69.7</td><td>8.5</td><td>11.8</td><td>25.5</td><td>33.9</td></tr></table>

Table 1: Performance comparison under LONG and SHORT reasoning modes for Qwen3-4B/8B-Base and Llama3.1- 8B-Instruct models. Acc.: mean@8 accuracy (%); R-tok.: reasoning tokens before the final answer, excluding answer tokens. Average is the unweighted mean across four tasks. Shading highlights PaceReasoner. Bold and underlining indicate the best and second-best results within each backbone and mode, respectively.

2019; Rein et al., 2024). Together, these benchmarks cover both open-ended and multiple-choice questions.

Implementation. We evaluate Qwen3-4B-Base, Qwen3-8B-Base, and Llama3.1-8B-Instruct (Qwen Team, 2025; Grattafiori et al., 2024). Adaptation is performed independently on each target benchmark without reference-answer rewards. We use GRPO in EasyR1 (Shao et al., 2024; Zheng et al., 2025), AdamW with learning rate $1 0 ^ { - 6 }$ , 15 epochs, and 15 rollouts per question at temperature 1.0. Rollout batches contain 256 questions for CSQA/GSM8K and 64 for GPQA-Diamond/MATH-500. The configured actor update batch sizes are respectively 64 and 16 question groups, corresponding to 960 and 240 response trajectories after expansion by 15 rollouts. These are distinct from rollout batch sizes.

We use $\lambda _ { \mathrm { b r } } = 2$ and 8 NVIDIA H20 GPUs.

Baselines. We compare zero-shot CoT, TTRL (Zuo et al., 2025), Length-Penalty, and CRISP (Sang et al., 2026) from the same initial checkpoint for each backbone, using the same target data for adaptation and a unified mean@8 evaluation protocol. Length-Penalty is our adaptation of reasoninglength control ideas from ThinkPrune (Hou et al., 2025b), DLER (Liu et al., 2025), and L1 (Aggarwal and Welleck, 2025b) to the label-free target-task setting. Its reward is $R _ { \mathrm { f m t } } + R _ { \mathrm { a c c } } R _ { \mathrm { l e n g t h } }$ , where $R _ { \mathrm { l e n g t h } } = 1 / ( L - L _ { \mathrm { m e d } } )$ for response length L above the batch median $L _ { \mathrm { m e d } }$ , and 1 otherwise.

Evaluation Metrics. For each question and mode, we sample eight responses and score each final answer as correct (1) or incorrect (0). Accuracy is the mean of these scores across responses and questions, reported as mean@8 (%). We do not use voting or best-of-eight selection. R-tok. is the average number of generated reasoning tokens before the final answer, excluding answer tokens. We report unweighted averages across tasks. Table 1 reports accuracy and reasoning cost separately for LONG and SHORT. All main-table PaceReasoner results use the final method with confidenceweighted progressive supervision; the full-method reference results in the component studies match the main table.

<table><tr><td></td><td colspan="2">CSQA</td><td colspan="2">GPQA-D</td><td colspan="2">MATH-500</td><td colspan="2">GSM8K</td></tr><tr><td>Supervision</td><td>Acc.</td><td>Time</td><td>Acc.</td><td>Time</td><td>Acc.</td><td>Time</td><td>Acc.</td><td>Time</td></tr><tr><td>All-response</td><td>79.6</td><td>2h 24m</td><td>32.7</td><td>44m</td><td>30.2</td><td>2h 17m</td><td>26.1</td><td>4h 25m</td></tr><tr><td>LONG-only</td><td>78.9</td><td>2h 13m</td><td>28.2</td><td>44m</td><td>45.0</td><td>2h 22m</td><td>36.9</td><td>4h 38m</td></tr><tr><td>Progressive</td><td>80.2</td><td>1h 57m</td><td>36.9</td><td>40m</td><td>39.8</td><td>2h 9m</td><td>39.1</td><td>3h 51m</td></tr></table>

Table 2: Supervision ablation on Qwen3-4B-Base. Acc.: SHORT accuracy (%); time: RL training time.
<table><tr><td rowspan="2">SHORT exploration</td><td rowspan="2">LONG-answer SHORT ratio matching</td><td rowspan="2">control</td><td rowspan="2">Eval. mode</td><td colspan="3">CSQA</td><td colspan="3">GPQA-D</td><td colspan="3">MATH-500</td><td colspan="3">GSM8K</td></tr><tr><td> $\operatorname { A c c } .$ </td><td>R-tok. Score</td><td></td><td></td><td>Acc. R-tok. Score</td><td></td><td>Acc. R-tok. Score</td><td></td><td></td><td></td><td>Acc. R-tok. Score</td><td></td></tr><tr><td>No</td><td>No</td><td>No</td><td>LONG</td><td>79.2</td><td>128.1</td><td>0.6 36.0</td><td></td><td>107.5</td><td>0.3</td><td>76.7</td><td>479.8</td><td>0.2</td><td>93.3 206.6</td><td></td><td>0.5</td></tr><tr><td>Yes</td><td>No</td><td>Yes</td><td>SHORT 79.3</td><td></td><td>14.7</td><td></td><td>5.4 33.3</td><td>31.4</td><td>1.1</td><td>40.0</td><td>14.5</td><td>2.8</td><td>25.8</td><td>19.3</td><td>1.3</td></tr><tr><td>Yes</td><td>Yes</td><td>No</td><td>SHORT</td><td>77.3</td><td>12.0</td><td>6.4</td><td>0.0</td><td>11.0</td><td>0.0</td><td>21.4</td><td>14.5</td><td>1.5</td><td>16.8</td><td>19.3</td><td>0.9</td></tr><tr><td>Yes</td><td>No</td><td>No</td><td>SHORT</td><td>80.0</td><td>12.0</td><td>6.7</td><td>22.7</td><td>31.4</td><td>0.7</td><td>24.4</td><td>14.5</td><td>1.7</td><td>25.2</td><td>19.3</td><td>1.3</td></tr><tr><td>Yes</td><td>Yes</td><td>Yes</td><td>SHORT</td><td>80.2</td><td>12.0</td><td></td><td>6.7 36.9</td><td>51.4</td><td>0.7</td><td>39.8</td><td>14.5</td><td>2.7</td><td>39.1</td><td>19.3</td><td>2.0</td></tr></table>

Table 3: Behavior-reward ablation on Qwen3-4B-Base. SHORT exploration rewards choosing SHORT; LONG answer matching rewards SHORT answers that match LONG consensus (Eq. (8)). SHORT ratio control scales this bonus using the sampled SHORT proportion (Eq. (9)). The last row enables all three components. Score = Acc./R-tok.; accuracy is in percentage units. Among SHORT results, bold and underlining mark the best and second-best distinct reported values, respectively (higher Acc./Score, lower R-tok.). The LONG row is a reference.

Algorithm 1 PaceReasoner test-time adaptation   
Require: Initial policy $\pi _ { \theta _ { 0 } }$ , questions D, steps T   
1: for $t = 1 , \dots , T$ do   
2: Sample a question batch from D   
3: Generate K mixed-mode responses per   
question   
4: Parse modes and final answers   
5: Compute all/LONG consensus and $c _ { L } ( q )$   
6: Compute batch SHORT ratio $\rho _ { S }$ and $s _ { t } =$   
$t / T$   
7: Compute answer reward using Eq. (7)   
8: Compute behavior reward using Eqs. (8)–(9)   
9: Add format reward and normalize within   
each group   
10: Update both modes on their complete trajec  
tories   
11: end for   
12: return Adapted policy π<sub>θ</sub>

## 5.2 Short Reasoning Performance

Comparison with baselines. PaceReasoner achieves higher average SHORT accuracy than all baselines, exceeding TTRL by 13.2, 13.5, and 7.4 points on Qwen3-4B, Qwen3-8B, and Llama3.1-8B at 24.3, 12.3, and 17.4 reasoning tokens, respectively (Table 1). All three improve over zero-shot with fewer tokens. Qwen3-8B’s cost is similar to TTRL (12.3 versus 12.2 tokens) and below CRISP and Length-Penalty. Llama also uses fewer tokens than Length-Penalty. Qwen3-8B gains 37.9 GSM8K points over TTRL. Both average 13.2 reasoning tokens. These comparisons use average costs, not matched per-question budgets.

LONG also achieves higher average accuracy than all baselines, using fewer tokens than TTRL and Length-Penalty on all backbones and than CRISP on both Qwen models. SHORT improvements thus accompany a stronger LONG mode.

LONG–SHORT accuracy gap. The absolute LONG–SHORT accuracy gap is within 1 point in 41.7% (5/12) of model–task pairs and within 5 points in 50% (6/12). The latter cover CSQA and GPQA-D across all three models, with SHORT using 69.4%–95.5% fewer tokens. This supports SHORT for timely, sufficiently accurate answers on these tasks, while larger mathematical gaps motivate retaining LONG.

<table><tr><td>Reasoning mode</td><td>Model</td><td>CSQA</td><td>GPQA-D</td><td>MATH-500</td><td>GSM8K</td></tr><tr><td rowspan="3">LONG</td><td>Qwen3-4B (Thinking) 4B-Base (PaceReasoner)</td><td>56.8 80.8</td><td>15.2 36.9</td><td>65.4</td><td>92.3</td></tr><tr><td></td><td></td><td></td><td>77.2</td><td>93.7</td></tr><tr><td>Qwen3-8B (Thinking) 8B-Base (PaceReasoner)</td><td>82.0 83.0</td><td>16.7 36.9</td><td>61.4 77.8</td><td>93.6 95.0</td></tr><tr><td rowspan="4">SHORT</td><td>Qwen3-4B (Non-thinking)</td><td>76.7</td><td>37.8</td><td>18.2</td><td>18.4</td></tr><tr><td>4B-Base (PaceReasoner)</td><td>80.2</td><td>36.9</td><td>39.8</td><td>39.1</td></tr><tr><td>Qwen3-8B (Non-thinking)</td><td>79.4</td><td>37.4</td><td>24.2</td><td></td></tr><tr><td>8B-Base (PaceReasoner)</td><td>82.5</td><td>36.4</td><td>37.2</td><td>21.1 46.0</td></tr></table>

Table 4: Accuracy (%) of PaceReasoner adapted from Qwen3-4B/8B-Base and released Qwen3-4B/8B checkpoints. Thinking and non-thinking are paired with LONG and SHORT, respectively. Bold indicates the higher accuracy within each model-size and mode pair.

## 5.3 Ablation Study

Table 2 and Table 3 present ablation results to analyze the contribution of method components.

Supervision strategy. Progressive supervision gives the highest SHORT accuracy on three tasks and the shortest training time on all four (Table 2), supporting its accuracy and training-efficiency benefits. LONG-only nevertheless leads on MATH-500 (45.0%), so supervision preferences remain task-dependent.

Behavior reward and ratio control. Table 3 ablates SHORT exploration, LONG-answer matching, and ratio control. Score measures accuracy per reasoning token (Kaiser et al., 2026). Without behavior rewards, the model relies on LONG. Removing ratio control reduces GSM8K accuracy from 39.1% to 16.8% at the same 19.3 tokens. Removing LONG-answer matching reduces GSM8K accuracy to 25.8% and GPQA-D accuracy from 36.9% to 33.3%. GPQA-D length falls from 51.4 to 31.4 tokens, increasing Score from 0.7 to 1.1 despite lower accuracy. MATH-500 differs by only 0.2 points. The components thus provide task-dependent accuracy gains.

## 5.4 Comparison with Post-trained Qwen3

To assess our lable-free test-time adaptation with post-training method, Table 4 compares PaceReasoner with the released Qwen3-4B/8B checkpoints, using enable\_thinking=True/False for their thinking/non-thinking modes (Qwen Team, 2025). SHORT exceeds non-thinking accuracy on six of eight model–task pairs, gaining 13.0–24.9 points on mathematics. CSQA gains are smaller, at 3.1–3.5 points. GPQA-D is 0.9–1.0 points lower. LONG exceeds thinking accuracy on all eight pairs. This reference comparison involves models with different post-training histories.

<table><tr><td>Eval. dataset</td><td>Training data</td><td>LONG</td><td>SHORT</td></tr><tr><td rowspan="3">GPQA-D</td><td>None (zero-shot)</td><td>21.3</td><td>11.6</td></tr><tr><td>GPQA-D</td><td>36.9</td><td>36.9</td></tr><tr><td>CSQA</td><td>37.4</td><td>34.7</td></tr><tr><td rowspan="3">MATH-500</td><td>None (zero-shot)</td><td>50.0</td><td>31.6</td></tr><tr><td>MATH-500</td><td>77.2</td><td>39.8</td></tr><tr><td>GSM8K</td><td>74.2</td><td>36.8</td></tr></table>

Table 5: Cross-dataset accuracy (%) on Qwen3-4B-Base. Each row specifies the evaluation and training datasets. None (zero-shot) uses the initial model; all trained rows use PaceReasoner with unlabeled questions from the listed training dataset.

## 5.5 Cross-Dataset Transfer

To investigate whether SHORT-mode improvements transfer beyond the adaptation dataset, we further adapt Qwen3-4B-Base on CSQA or GSM8K and evaluate on GPQA-D or MATH-500, respectively shown in Table 5. SHORT gains 23.1 and 5.2 points over zero-shot, trailing direct targetset adaptation by 2.2 and 3.0 points. This retains 63–91% of the target-adaptation gains. LONG differs from target-set adaptation by +0.5 and −3.0 points. Both modes improve over zero-shot without target-set adaptation, suggesting that the gains are not solely due to same-dataset overfitting.

## 6 Conclusion

We presented PaceReasoner, a label-free test-time reinforcement learning framework that improves SHORT-mode accuracy using answer supervision from the same model’s LONG responses. Progressive, confidence-weighted supervision and behavior control coordinate answer learning and mode sampling within a shared policy. Across three models and four benchmarks, PaceReasoner improves average SHORT accuracy over TTRL by 7.4–13.5 percentage points with fewer than 25 reasoning tokens per response on average. Ablations support the key components, and cross-dataset results demonstrate transfer beyond the adaptation set. These findings support using a model’s own long-reasoning answers to strengthen its performance with short reasoning.

## Limitations

Our evaluation covers three 4B/8B models and four question-answering benchmarks, without coding or interactive deployment tests. Target-set adaptation and two transfer experiments provide limited evidence of generalization. Multi-seed variability is not measured.

Self-generated consensus can reinforce shared errors and depends on the model retaining useful LONG reasoning capability. Confidence weighting is heuristic, and its incremental benefit is not isolated by the available ablations. Ratio control does not guarantee per-question LONG coverage; sensitivity to controller and reward settings remains a limitation. SHORT is a prompted mode rather than a fixed token budget.

Our cost metric counts reasoning tokens before the final answer, excluding answer tokens and adaptation computation. These measurements do not establish lower end-to-end latency, total deployment cost, or energy use.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In The Twelfth International Conference on Learning Representations.

Pranjal Aggarwal and Sean Welleck. 2025a. L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning. In Second Conference on Language Modeling.

Pranjal Aggarwal and Sean Welleck. 2025b. L1: Controlling how long a reasoning model thinks with reinforcement learning. In Second Conference on Language Modeling.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Yuntian Deng, Yejin Choi, and Stuart Shieber. 2024. From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step. Preprint, arXiv:2405.14838.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. Thinkless: LLM learns when to think. In Advances in Neural Information Processing Systems, volume 38.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. 2025a. ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning. Preprint, arXiv:2504.01296.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. 2025b. ThinkPrune: Pruning long chain-of-thought of LLMs via reinforcement learning. arXiv preprint arXiv:2504.01296.

Daniel Kaiser, Arnoldo Frigessi, Ali Ramezani-Kebrya, and Benjamin Ricaud. 2026. Decomposing reasoning efficiency in large language models. arXiv preprint arXiv:2602.09805.

Durgesh Kalwar, Vardhan Palod, Jaya Adithya Pavuluri, and Subbarao Kambhampati. 2026. Masked Self-Distillation: Internalizing the Chain-of-Thought in Language Models. Preprint, arXiv:2607.22629.

Taehyeon Kim, Hyunsoo Lee, Youngsoo Jang, and Moontae Lee. 2026. Efficiently learning to reason or not to reason: Root-token policy optimization for adaptive thinking. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 17934– 17949. Association for Computational Linguistics.

Junyan Li, Wenshuo Zhao, Yang Zhang, and Chuang Gan. 2026a. Steering LLM thinking with budget guidance. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 37437–37451, San Diego, California, United States. Association for Computational Linguistics.

Yijiang Li, Bingyang Wang, Yijun Liang, Yunjie Tian, Di Fu, and Nuno Vasconcelos. 2026b. On-policy selfdistillation without any supervision. arXiv preprint arXiv:2608.06296.

Junhong Lin, Xinyue Zeng, Jie Zhu, Song Wang, Julian Shun, Jun Wu, and Dawei Zhou. 2026. Plan and budget: Effective and efficient test-time scaling on reasoning large language models. In The Fourteenth International Conference on Learning Representations.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, et al. 2025. DLER: Doing length penalty right—incentivizing more intelligence per token via reinforcement learning. arXiv preprint arXiv:2510.15110.

Xinyin Ma, Guangnian Wan, Runpeng Yu, Gongfan Fang, and Xinchao Wang. 2025. Cot-valve: Lengthcompressible chain-of-thought tuning. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 6025–6035.

Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. 2025. Maximizing Confidence Alone Improves Reasoning. Preprint, arXiv:2505.22660.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. GPQA: A graduate-level google-proof Q&A benchmark. In First Conference on Language Modeling.

Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. 2026. CRISP: Compressed reasoning via iterative self-policy distillation. arXiv preprint arXiv:2603.05433.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Aozhe Wang, Zhengxi Lu, Jianze Wang, Shangke Lv, Ying Liu, Weiming Lu, Jun Xiao, Yueting Zhuang, Hua Yang, Qianglong Chen, and Yongliang Shen. 2026. TTPO: Test-time policy optimization. arXiv preprint arXiv:2608.27448.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. 2025. AdaptThink: Reasoning models can learn when to think. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3716–3730, Suzhou, China. Association for Computational Linguistics.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026a. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. Preprint, arXiv:2601.18734.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. 2026b. Learning to Reason without External Rewards. In The Fourteenth International Conference on Learning Representations.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, and Richong Zhang. 2025. EasyR1: An efficient, scalable, multimodality RL training framework. https://github. com/hiyouga/EasyR1.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. 2025. TTRL: Test-time reinforcement learning. In Advances in Neural Information Processing Systems, volume 38.

## A Additional Experimental Details

## A.1 Benchmarks and Data

CSQA evaluates commonsense question answering with five answer options. GPQA-Diamond contains 198 graduate-level science questions. MATH-500 contains 500 problems from MATH, while GSM8K evaluates arithmetic word problems (Talmor et al., 2019; Rein et al., 2024; Hendrycks et al., 2021; Cobbe et al., 2021). The supplied question pools contain 1,221 CSQA, 198 GPQA-Diamond, 500 MATH-500, and 1,319 GSM8K items. For each task, adaptation and evaluation files contain the same question sequence. We use these questions as unlabeled target data for independent adaptation and evaluate their final answers. MATH-500 does not require a particular visible reasoning length for an answer to be scored correct. Multiple-choice and mathematical answers are parsed and normalized for voting and evaluation.

Evaluation generation settings. Zero-shot, TTRL, and PaceReasoner use forced LONG and SHORT instructions. For CRISP, the LONG group reports its native short-reasoning setting; the SHORT group reports native no-think for Qwen3-8B-Base and answer-only generation for the other backbones. These results use the unified LONG/SHORT labels in Table 1; actual reasoning lengths are measured for each method.

## A.2 Training Configuration

Table 6 distinguishes a rollout batch of questions from an actor update batch. With 15 responses per question, the training framework expands the configured actor batch by a factor of 15 before distributing it across workers. Thus, a configured actor batch of 64 denotes 960 response trajectories globally, not 64 trajectories per GPU. Main-table evaluation uses the unified mean@8 protocol described in Section 5.1.

## A.3 Consensus Construction

For each question, final-answer extraction forms an all-response answer pool and a LONG-only pool. We vote over normalized final answers, rather than entire reasoning traces. If a pool is empty, its pseudo-label is missing and its agreement reward is zero. A tie in the two leading LONG answer counts yields zero confidence in Eq. (6); this affects the answer-supervision weight but does not remove the time-based behavior bonus. The additional confidence weight therefore reduces reliance on weak

Algorithm 2 Majority Voting for Pseudo Ground   
Truth (Reasoning / All Rollouts)   
Input: rollout set ${ \mathcal O } ( q ) ~ = ~ \{ o _ { i } \} _ { i = 1 } ^ { N }$ for each   
query q   
Output: pseudo ground truths $g t _ { \mathrm { t h i n k } } ( q )$ and   
$g t _ { \mathrm { a l l } } ( q )$   
Initialize dictionaries $\mathcal { A } _ { \mathrm { t h i n k } }  \{ \} , \mathcal { A } _ { \mathrm { a l l } }  \{ \}$   
for each response $o _ { i } \in \mathcal { O } ( q )$ do   
o ← STRIPSPACESAROUNDTAGS $\left( o _ { i } \right)$   
$m _ { i } \gets$ DETECTMODE $\left( o _ { i } \right)$   
$a _ { i } \gets$ EXTRACTANSWER $\left( o _ { i } \right)$   
if $a _ { i } \neq \emptyset$ then   
Append $a _ { i }$ to $\mathcal { A } _ { \mathrm { a l l } } [ q ]$   
if $m _ { i } =$ think then   
Append $a _ { i }$ to A<sub>think</sub>[q]   
end if   
end if   
end for   
for each query q do   
if $| \mathcal { A } _ { \mathrm { t h i n k } } [ q ] | > 0$ then   
$g t _ { \mathrm { t h i n k } } ( q )$ ←   
arg max COUNT $( a , \mathcal { A } _ { \mathrm { t h i n k } } [ q ] )$   
a∈A<sub>think</sub>[q]   
else   
$g t _ { \mathrm { t h i n k } } ( q ) \gets \emptyset$   
end if   
if $| \mathcal { A } _ { \mathrm { a l l } } [ q ] | > 0$ then   
gt<sub>all</sub>(q) ← arg max COUNT(a, A<sub>all</sub>[q])   
a∈A<sub>all</sub>[q]   
else   
$g t _ { \mathrm { a l l } } ( q ) \gets \emptyset$   
end if   
end for

LONG consensus without establishing correctness of the fallback consensus.

Algorithm 2 retains the original construction of the LONG-only (think) and all-response answer pools. Confidence weighting is applied after this voting step.

## A.4 Mode Instructions

The following template states the LONG/SHORT generation contract used in the revised method description. At forced-mode evaluation, the corresponding opening marker conditions generation. A direct answer is allowed as the emptyreasoning special case of SHORT; there is no separate SHORT token cap.

<table><tr><td>Parameter</td><td>CSQA / GSM8K</td><td>GPQA-D / MATH-500</td></tr><tr><td>Rollout batch (questions)</td><td>256</td><td>64</td></tr><tr><td>Actor update batch (question groups, configured)</td><td>64</td><td>16</td></tr><tr><td>Responses per question</td><td>15</td><td>15</td></tr><tr><td>Rollout batch (response trajectories)</td><td>3840</td><td>960</td></tr><tr><td>Actor update batch (response trajectories)</td><td>960</td><td>240</td></tr><tr><td>Training epochs</td><td>15</td><td>15</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td></tr><tr><td>Learning-rate warmup ratio</td><td>0</td><td>0</td></tr><tr><td>Rollout temperature</td><td>1.0</td><td>1.0</td></tr><tr><td>Maximum prompt / response length</td><td>2048 / 2048</td><td>2048 / 2048</td></tr><tr><td>Reference KL loss coefficient</td><td>0.01</td><td>0.01</td></tr><tr><td>Behavior bonus  $\lambda _ { \mathrm { b r } }$ </td><td>2</td><td>2</td></tr><tr><td>GPUs / tensor parallel size</td><td> $8 / 2$ </td><td> $8 / 2$ </td></tr><tr><td>Actor update microbatch (responses per device)</td><td>1</td><td>1</td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td><td>Enabled</td></tr><tr><td>Rollout GPU memory utilization</td><td>0.6</td><td>0.6</td></tr></table>

Table 6: Training settings. Batch quantities are global unless explicitly marked per device. Maximum response length is an overall generation limit, not a separate SHORT budget.

## LONG/SHORT mode specification

Choose a solution mode for the question.   
LONG: develop the necessary explicit   
analysis,   
intermediate derivations, and checks. Use:   
<reason>reasoning</reason><answer>final   
answer</answer>   
SHORT: solve concisely and directly,   
retaining only   
essential steps when needed. Use:   
<not>brief reasoning</not><answer>final   
answer</answer>   
A direct answer may use an empty   
<not></not> block.   
Place only the final answer in   
<answer>...</answer>.

## A.5 Recorded Training Prompt

The recorded training prompt below uses the <not></not> marker for the direct-answer case of SHORT.

## Recorded training prompt

You are required to first decide whether to think or not based on the given question. The reasoning process MUST BE enclosed within <reason> </reason> tags, and the final answer MUST BE enclosed within <answer> </answer> tags.

\- If you decide NOT to think, you should ONLY output <not></not> and DIRECTLY output the final answer, and DO NOT contain any thinking process, the final answer MUST BE enclosed within <answer> </answer> tags. Output as

'<not></not><answer> </answer>'.   
ONLY include the final answer   
between '<answer> </answer>' tags   
and do not include any other texts.   
- If you decide to think, then you MUST   
enclose thinking process within   
<reason> </reason> tags, and the   
final answer MUST BE enclosed within   
<answer> </answer> tags. Output as   
'<reason> </reason><answer>   
</answer>'. Do not contain any   
<answer> </answer> tags within   
<reason> </reason> tokens, and all   
the texts MUST BE included in tags   
'<reason> </reason>' and '<answer>   
</answer>'.   
# Requirements   
ALL the output texts MUST BE enclosed   
within tags.

## B Additional Component Results

The tables in this section report Qwen3-4B-Base component studies, with full-method reference results matching Table 1. Gray entries identify runs evaluated using LONG after failing to activate the intended short behavior, and should not be read as SHORT measurements.

## B.1 Controller Shape and Bonus Strength

Table 7 compares the controller functions, and Table 8 retains the reported shifted-cosine settings. These choices differ in their sign, scale, and behavior near the endpoints, so the comparison does not isolate smoothness or establish a unique optimal controller. The unshifted cosine gives the highest SHORT accuracy among the controller settings on all four tasks, although its higher GPQA accuracy comes with longer reasoning and a lower accuracyper-token score. In the bonus-strength comparison (Table 9), a value of 1 fails to activate SHORT on the two mathematical tasks; a value of 2 gives the highest SHORT accuracy on three tasks, while 3 is slightly better on GSM8K (39.3% versus 39.1%) with more reasoning tokens. On GPQA, the default bonus of 2 improves accuracy at the cost of longer reasoning, rather than a higher accuracy-per-token score.

## B.2 Training Dynamics

Figure 3(a–b) shows that SHORT sampling approaches zero without activation and approaches one without ratio control. With both components, the rollout pool retains a mixture of modes, supplying SHORT training examples alongside LONG answers for supervision. These trajectories complement the behavior-reward ablation in Table 3.

On GPQA, all-response voting suffers an early accuracy drop and LONG-only voting declines later, whereas progressive supervision improves more steadily (Figure 3c). These trajectories complement the supervision comparison in Table 2; they do not separately isolate the additional effect of confidence weighting.

## C Recorded Qualitative Examples

We retain the original recorded outputs for CSQA, GPQA, MATH-500, and GSM8K below. Their generated text is unchanged; these selected examples illustrate behavior and are not additional measurements from the updated main-table configurations. Mode markers inside a recorded response are preserved as part of that response. GT denotes the ground-truth reference answer. Table 10 includes the original questions and SHORT-mode responses, including zero-shot outputs. The remaining tables retain the LONG-mode examples.

<table><tr><td rowspan="2">Ratio Controller  $f ( x )$ </td><td colspan="3">CSQA</td><td colspan="3">GPQA</td><td colspan="3">MATH-500</td><td colspan="3">GSM8K</td></tr><tr><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td></tr><tr><td>1</td><td>77.3</td><td>12.0</td><td>6.4</td><td>0.0</td><td>11.0</td><td>0.0</td><td>21.4</td><td>14.5</td><td>1.5</td><td>16.8</td><td>19.3</td><td>0.9</td></tr><tr><td> $1 - x$ </td><td>78.5</td><td>11.6</td><td>6.8</td><td>20.7</td><td>31.5</td><td>0.7</td><td>32.5</td><td>14.4</td><td>2.3</td><td>36.0</td><td>19.3</td><td>1.9</td></tr><tr><td> ${ \begin{array} { l } { { \frac { 1 } { 2 } } - x } \\ { { \frac { 1 } { x } } } \end{array} }$ </td><td>79.8</td><td>12.0</td><td>6.7</td><td>31.2</td><td>31.9</td><td>1.0</td><td>27.8</td><td>14.5</td><td>1.9</td><td>26.0</td><td>19.1</td><td>1.4</td></tr><tr><td></td><td>76.7</td><td>12.0</td><td>6.4</td><td>22.2</td><td>31.5</td><td>0.7</td><td>31.2</td><td>14.4</td><td>2.2</td><td>37.2</td><td>19.3</td><td>1.9</td></tr><tr><td> $\cos ( \pi x )$ </td><td>80.2</td><td>12.0</td><td>6.7</td><td>36.9</td><td>51.4</td><td>0.7</td><td>39.8</td><td>14.5</td><td>2.7</td><td>39.1</td><td>19.3</td><td>2.0</td></tr></table>

Table 7: Controller-function sensitivity on Qwen3-4B-Base. The reported score is accuracy divided by reasoning tokens; accuracy and length remain the primary measurements.

<table><tr><td rowspan="2">Ratio Controller f(x)</td><td colspan="3">CSQA</td><td colspan="3">GPQA</td><td colspan="3">MATH-500</td><td colspan="3">GSM8K</td></tr><tr><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td></tr><tr><td>COS  $\textstyle { \left( \pi ( x - { \frac { \pi } { 4 } } ) \right) }$ </td><td>74.1</td><td>11.5</td><td>6.4</td><td>28.8</td><td>31.6</td><td>0.9</td><td>25.2</td><td>13.8</td><td>1.8</td><td>90.3</td><td>198.3</td><td>0.5</td></tr><tr><td>COS  $\begin{array} { r } { ( \pi ( x + \frac { \pi } { 4 } ) ) } \end{array}$ </td><td>77.2</td><td>12.0</td><td>6.4</td><td>27.8</td><td>31.4</td><td>0.9</td><td>32.3</td><td>13.5</td><td>2.4</td><td>89.3</td><td>202.4</td><td>0.4</td></tr><tr><td>cos(πx)</td><td>80.2</td><td>12.0</td><td>6.7</td><td>36.9</td><td>51.4</td><td>0.7</td><td>39.8</td><td>14.5</td><td>2.7</td><td>39.1</td><td>19.3</td><td>2.0</td></tr></table>

Table 8: Shifted-controller sensitivity. Functions and values are retained as reported. Gray entries are LONG evaluations, not successful SHORT results.

<table><tr><td rowspan="2">Behavior Bonus  $\lambda _ { b r }$ </td><td colspan="3">CSQA</td><td colspan="3">GPQA</td><td colspan="3">MATH-500</td><td colspan="3">GSM8K</td></tr><tr><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td><td>acc</td><td>R-tok.</td><td>score</td></tr><tr><td>1</td><td>79.9</td><td>13.0</td><td>6.1</td><td>32.3</td><td>31.3</td><td>1.0</td><td>78.0</td><td>440.7</td><td>0.2</td><td>92.2</td><td>194.8</td><td>0.5</td></tr><tr><td>2</td><td>80.2</td><td>12.0</td><td>6.7</td><td>36.9</td><td>51.4</td><td>0.7</td><td>39.8</td><td>14.5</td><td>2.7</td><td>39.1</td><td>19.3</td><td>2.0</td></tr><tr><td>3</td><td>80.0</td><td>13.0</td><td>6.2</td><td>30.3</td><td>35.0</td><td>0.9</td><td>39.6</td><td>14.8</td><td>2.7</td><td>39.3</td><td>21.9</td><td>1.8</td></tr></table>

Table 9: Behavior-bonus sensitivity. Gray entries are LONG evaluations. The score is accuracy divided by reasoning tokens.

![](images/683392f86cabe61b2848736c2232f9f88b60f3816c3b150ad86ba614b8a45d49.jpg)  
(a) SHORT ratio on GPQA.

![](images/c863946e9cec3b194a553a69ca231e5c3844a544eabef82d5d8e04202a01da56.jpg)  
(b) SHORT ratio on MATH-500.

![](images/393567ca936f6bcf39c934737a1937c4902e5f3d4cc5c254e173970edf2857ef.jpg)  
(c) Supervision on GPQA.  
Figure 3: Training dynamics on Qwen3-4B-Base. (a–b) SHORT sampling with and without activation or ratio control; “Non-reasoning” denotes the direct-answer SHORT setting. (c) Validation accuracy under all-response (Naive), LONG-only (Reasoning), and progressive supervision.

Table 10: Recorded SHORT-mode responses of Qwen3-4B-Base. The questions, reference answers, and generated outputs are preserved from the original examples.
<table><tr><td></td><td>CSQA</td><td>GPQA</td><td>MATH-500</td><td>GSM8K</td></tr><tr><td>Question</td><td>Question: The man closed his eyes as the music played, what effect did the music have? (A)coma (B)enjoyable (C)soothing (D)universal</td><td>Question: Astronomers are observing a planet with a radius of 1 Earth radius in a circular orbit, ... What is the maximum orbital period of a second planet, with a radius of</td><td>Question:  $\begin{array} { r l r } { \frac { 1 } { \cos _ { 1 } ^ { 2 } 1 0 ^ { \circ } } } & { { } + } & { \frac { 1 } { \sin ^ { 2 } 2 0 ^ { \circ } } } \end{array} +$   $\scriptstyle { \overline { { \sin ^ { 2 } ~ 4 0 ^ { \circ } } } }$ </td><td>Compute Question:Two leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel</td></tr><tr><td>GT</td><td>C</td><td>B</td><td></td><td>train in the two days? 230</td></tr><tr><td>Zero- shot&#x27;s outputs</td><td>The man closed his eyes as the music played, what effect did the music have? &lt;not&gt;&lt;/not&gt;&lt;answer&gt; (C)soothing&lt;/answer&gt;</td><td>To solve this problem, we need to determine the maximum orbital pe- riod of the second planet that can both transit and occult the first planet. Here&#x27;s the step-by-step reasoning: ... &lt;answer&gt;</td><td>To compute the ex- pression  $\frac { 1 } { \cos ^ { 2 } 1 0 ^ { \circ } }$  十  $\textstyle { \frac { 1 } { \sin ^ { 2 } 2 0 ^ { \circ } } } \ + \ { \frac { 1 } { \sin ^ { 2 } 4 0 ^ { \circ } } } ,$  we can use trigonometric identities and properties. First, recall the iden-  $\mathrm { t i t y } \colon \ { \frac { 1 } { \cos ^ { 2 } \theta } } = \sec ^ { 2 } \theta ,$   $\begin{array} { r } { \frac { 1 } { \sin ^ { 2 } \theta } \ = \csc ^ { 2 } \theta . \ldots \ ( \mathrm { n o \ } } \end{array}$ </td><td>The total distance cov- ered by each train in the two days is calculated by adding the distance trav- eled on the first day (80 miles) and the distance traveled on the second day (150 miles). This gives us a total distance</td></tr><tr><td>TTRL&#x27;s outputs</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; C&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; C&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; 16&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; 460 miles&lt;/answer&gt;</td></tr><tr><td>Ours</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; C&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; B.33.5&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; 12&lt;/answer&gt;</td><td>&lt;not&gt;&lt;/not&gt;&lt;answer&gt; 230&lt;/answer&gt;</td></tr></table>

<table><tr><td>CSQA</td><td></td></tr><tr><td>Question</td><td>Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? (A)bank (B)library (C)department store (D)mall (E)new york</td></tr><tr><td>GT</td><td>A</td></tr><tr><td>Zero-shot's out- puts</td><td>To determine whether to think or not, I will analyze the question and the options provided. **Question:** A revolving door is convenient for two-direction travel, but it also serves as a security measure at a what? **Options:** (A) bank (B) library (C) department store (D) mall (E) New York **Reasoning:**A revolving door is commonly used in places where a high volume of people move in and out, and it helps to prevent the spread of germs and maintain a sense of security. This is particularly important in financial institutions like banks, where security is a top priority. Given this information, I can conclude that the correct answer is: &lt;answer&gt;(A) bank&lt;/answer&gt;</td></tr><tr><td>TTRL's outputs</td><td>&lt;reason&gt;The question asks which location a revolving door serves as a security measure. A revolving door is commonly used as a security measure at banks to control access and prevent unauthorized entry Libraries, department stores, and malls may use revolving doors for convenience, but banks are the most likely to use them for security purposes.&lt;/reason&gt;&lt;answer&gt;A&lt;/answer&gt;</td></tr><tr><td>PaceReasoner's outputs</td><td>&lt;reason&gt;A revolving door is typically used as a security measure at banks, as it helps prevent tailgating (where one person follows another into a secure area) and ensures that only one person can pass through at a time. While revolving doors can also be found in other places like department stores and malls, the primary security function is most commonly associated with banks.&lt;/reason&gt;&lt;answer&gt;A&lt;/answer&gt;</td></tr><tr><td>GPQA</td><td>Question: You are analyzing a small peptidic compound that has been chemically synthesized. The 1H</td></tr><tr><td>Question</td><td>NMR spectrum of the crude compound appears consistent with the expected molecule, with the exception of two peaks that both correspond to the same alpha-proton. These two peaks have similar chemical shifts and roughly equal integrals (together they integrate to 1H - the expected integral); based on the coupling pattern, you can rule out spin-spin coupling as an explanation for the duplicate peaks. LC-MS analysis of the crude compound at elevated temperature shows two clearly defined peaks of equal intensities. Both peaks have the same mass spectrum, which is consistent with the expected molecule. What is the most likely explanation for these observations? A. The compound is contaminated with a precursor B. The crude compound exists as a mixture of diastereoisomers C. 'Double coupling' has occurred during an amide-bond forming reaction D. The crude compound exists as a mixture of enantiomers. You are</td></tr><tr><td>GT B</td><td>required to first decide whether to think or not based on the given question.</td></tr><tr><td>Zero-shot's out- puts</td><td>You are a helpful assistant. &lt;reason&gt;The question describes a small peptidic compound with a 1H NMR spectrum showing two peaks for the same alpha-proton, which are consistent in chemical shift and integral. LC-MS analysis at elevated temperature shows two peaks of equal intensities with the same mass spectrum. The absence of spin-spin coupling and the consistency with the expected molecule suggest that the issue is not due to contamination or diastereoisomers. The most plausible explanation is that the compound exists as a mixture of enantiomers, as enantiomers have identical NMR spectra and mass spectra but different optical rotations. The observation of two peaks in the NMR spectrum could be due to the presence of both enantiomers in the sample.&lt;/reason&gt;&lt;answer&gt;D&lt;/answer&gt;</td></tr><tr><td>TTRL's outputs</td><td>&lt;reason&gt;The question describes a scenario where a synthetic peptidic compound shows two peaks in its 1H NMR spectrum that correspond to the same alpha-proton, with similar chemical shifts and integrals. The LC-MS analysis at elevated temperature shows two peaks of equal intensity with the same mass spectrum. The key points are: 1. **NMR Analysis**: Two peaks for the same proton, ruling out spin-spin coupling. 2. **LC-MS Analysis**: Two peaks of equal intensity with the same mass spectrum, indicating identical molecular species. The most likely explanation for these observations is that the compound exists in a form where these two peaks arise from different molecular conformations or isomers. Since the peaks have similar chemical shifts and integrals, they are likely not due to contamination or precursor molecules. Double coupling is not a plausible explanation given the context. Enantiomers would typically show distinct peaks in NMR due to their chirality, but here the peaks are identical. Diastereoisomers, however, can have similar NMR spectra due to their structural similarities, especially if they are in a conformational equilibrium.&lt;/reason&gt;&lt;answer&gt;B. The crude compound exists as a mixture of diastereoisomers &lt;/</td></tr><tr><td>PaceReasoner 's outputs</td><td>answer&gt; &lt;reason&gt;The question describes a situation where a synthetic peptidic compound shows two peaks in its 1H NMR spectrum that appear to be the same alpha-proton but do not couple to each other, and LC-MS analysis shows two peaks of equal intensity with identical mass spectra. The options provided are: A. Contamination with a precursor - This would likely show a different peak or pattern in the NMR or MS. B. Mixture of diastereoisomers - Diastereoisomers would have different physical properties and would show distinct peaks in NMR and MS. C. Double coupling during an amide-bond forming reaction - This would typically result in more complex splitting patterns due to the coupling constants involved. D. Mixture of enantiomers - Enantiomers would have identical physical properties and would show the same peaks in NMR and MS.&lt;/reason&gt;&lt;answer&gt;B. The crude compound exists as a mixture of diastereoisomers &lt;/ answer&gt;</td></tr><tr><td colspan="1" rowspan="1">MATH-500</td></tr><tr><td colspan="1" rowspan="1">Question          Question: What is the least positive integer multiple of 30 that can be written with only the digits 0 and2?</td></tr><tr><td colspan="1" rowspan="1">GT                2220</td></tr><tr><td colspan="1" rowspan="1">Zero-shot's out- To find the least positive integer multiple of 30 that can be written with only the digits 0 and 2, we need toputs               consider the properties of multiples of 30. A number is a multiple of 30 if and only if it is divisible byboth 2 and 15. This means the number must end in 0 (to be divisible by 2) and the sum of its digits mustbe divisible by 3 (to be divisible by 15). Given that the number can only contain the digits 0 and 2, let'sstart by checking the smallest possible numbers: 1. **Single-digit numbers**: None of the single-digitnumbers (0, 2) are multiples of 30. 2. **Two-digit numbers**: The only two-digit number that ends in 0is 20, but 20 is not divisible by 15 (since $2 + 0 = 2$ is not divisible by 3). 3. **Three-digit numbers**:We need to find a number ending in 0 where the sum of the digits is divisible by 3. Let's check: - 200: $( 2 + 0 + 0 = 2 )$ (not divisible by $3 ) - 2 2 0 \colon ( 2 + 2 + 0 = 4 )$ (not divisible by 3) - 202: $( 2 + 0 + 2 = 4 )$ (not divisible by $3 ) - 2 2 2 \colon ( 2 + 2 + 2 = 6 )$ (divisible by 3) Since 222 is divisible by 3 and ends in 0, it is amultiple of 30. Therefore, the least positive integer multiple of 30 that can be written with only the digits0 and 2 is 222. &lt;reason&gt;The number must end in 0 to be divisible by 2 and the sum of its digits must bedivisible by 3 to be divisible by 15. Checking the smallest possible numbers, we find that 222 meets bothcriteria.&lt;/reason&gt;&lt;answer&gt;222&lt;/answer&gt;</td></tr><tr><td colspan="1" rowspan="1">TTRL's outputs   &lt;reason&gt;To find the least positive integer multiple of 30 that can be written with only the digits 0 and2, we need to consider the properties of multiples of 30. A number is a multiple of 30 if and only if itis a multiple of both 3 and 10. 1. **Multiple of 10**: A number is a multiple of 10 if its last digit is 0.Therefore, the number we are looking for must end in 0. 2. **Multiple of $3 ^ { * * } { \mathrm { . } }$ A number is a multiple of3 if the sum of its digits is a multiple of 3. Since the only digits we can use are 0 and 2, we need to find thesmallest number ending in 0 where the sum of the digits is a multiple of 3. Let's start constructing such anumber: - The smallest number ending in 0 is $2 0 \left( 2 + 0 = 2 , \right.$ not a multiple of 3). - The next candidate is220 $( 2 + 2 + 0 = 4$ , not a multiple of 3). - The next candidate is 200 $( 2 + { \overset { \cdot } { 0 } } + 0 = 2$ , not a multiple of 3). -The next candidate is $2 2 0 0 ( 2 + 2 + 0 + 0 = 4$ , not a multiple of 3). - The next candidate is 2000 $( 2 + 0 +$  $0 + 0 = 2$ , not a multiple of 3). - The next candidate is $2 2 2 0 ( 2 + 2 + 2 + 0 = 6 .$  which is a multiple of3). Thus, 2220 is the smallest number that meets both criteria: it ends in 0 and the sum of its digits is amultiple of 3. Therefore, the least positive integer multiple of 30 that can be written with only the digits 0and 2 is 2220.&lt;/reason&gt;&lt;answer&gt;2220&lt;/answer&gt;</td></tr><tr><td colspan="1" rowspan="6">PaceReasoner 's  &lt;reason&gt;To find the least positive integer multiple of 30 that can be written with only the digits 0 and2, we need to consider the properties of multiples of 30. A number is a multiple of 30 if and only if itis a multiple of both 3 and 10. 1. **Multiple of 10**: A number is a multiple of 10 if its last digit is 0.Therefore, the number we are looking for must end in 0. 2. **Multiple of $3 ^ { \ast \ast }$ : A number is a multiple of3 if the sum of its digits is a multiple of 3. Since the only digits we can use are 0 and 2, the sum of thedigits must be a multiple of 3. Given that the number must end in 0, the possible forms are 2220, 220, 20,etc. We need to check the smallest ones first:  $\ast \ast 2 0 ^ { \ast \ast } ;$ Sum of $\mathrm { d i g i t s } = 2 + 0 = 2$ (not a multiple of 3) - $^ { * * } 2 2 0 ^ { * * }$ Sum of ${ \mathrm { d i g i t s } } = 2 + 2 + 0 = 4$ (not a multiple of $3 ) \_ * * _ { 2 2 } \bar { 2 } 0 ^ { * * } \colon$ Sum of digi $\mathrm { t s } = 2 + 2 + 2 + 0 =$ 6 (a multiple of 3) Since 2220 is the smallest number that meets both criteria (ends in 0 and the sum of itsdigits is a multiple of 3), it is the least positive integer multiple of 30 that can be written with only thedigits 0 and 2.&lt;/reason&gt;&lt;answer&gt;2220&lt;/answer&gt;</td></tr><tr><td colspan="1" rowspan="1">outputs</td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td>GSM8K</td><td></td></tr><tr><td>Question</td><td>Question: I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?</td></tr><tr><td>GT Zero-shot's out-</td><td>15 You are a helpful assistant. &lt;reason&gt;First, let's determine the amount of water in the original 10 liters of</td></tr><tr><td>puts</td><td>orange drink. Since it is two-thirds water, the amount of water in the orange drink is  $( \bar { 2 } / 3 ) \ast 1 0 = 6 . 6 7$  liters. Next, we need to account for the spill. Since one liter of the orange drink is spilled, we subtract that from the total amount of water in the orange drink:  $6 . 6 7 \cdot 1 = 5 . 6 7$  liters. Now, let's calculate the amount of water in the 15 liters of pineapple drink. Since it is three-fifths water, the amount of water in the pineapple drink is  $( 3 / 5 ) \ast 1 5 = 9$  liters. Finally, we add the remaining water from the orange drink and the water from the pineapple drink to find the total amount of water in the remaining 24 liters:  $5 . 6 7 + 9$   $= 1 4 . 6 7$  liters. Therefore, there are 14.67 liters of water in the remaining 24 liters.&lt;/reason&gt;&lt;answer&gt; 14.67&lt;/answer&gt;</td></tr><tr><td>TTRL's outputs</td><td>&lt;reason&gt;First, let's determine the amount of water in the original 10 liters of orange drink. Since it is two-thirds water, the amount of water in the orange drink is: Water in orange drink  $\textstyle { \overline { { \cdot } } } = { \frac { 2 } { 3 } } \times 1 0 = { \frac { 2 0 } { 3 } }$  ≈ 6.67 liters. Next, let's determine the amount of water in the 15 liters of pineapple drink. Since it is three- fifths water, the amount of water in the pineapple drink is: Water in pineapple  $\mathrm { \bar { d r i n k } = \frac { 3 } { 5 } \times 1 5 = 9 l i t e r s . }$  Since you spill 1 liter of the orange drink, the amount of orange drink left is: Remaining orange drink =  $1 0 - 1 = { \bar { 9 } } \mathrm { l i t e r s } .$  Now, let's calculate the amount of water in the remaining 9 liters of orange drink: Water in remaining orange  $\mathrm { d r i n k } = { \textstyle { \frac { 2 } { 3 } } } \times 9 = \mathrm { 6 l i t e r s }$  . Finally, the total amount of water in the combined mixture is the sum of the water from the remaining orange drink and the pineapple drink: Total water =  $6 + 9 = 1 5$  liters. So, the total amount of water in the remaining 24 liters is 15 liters.&lt;/reason&gt;&lt;answer&gt;  $1 5 { < } / \mathsf { a n s w e r } { > }$ </td></tr><tr><td>PaceReasoner 's outputs</td><td>&lt;reason&gt;First, let's determine the amount of water in the original 10 liters of orange drink. Since the orange drink is two-thirds water, the amount of water in it is: Water in orange drink  $\begin{array} { r } { \dot { \mathbf { \rho } } = \frac { 2 } { 3 } \times 1 0 = \frac { 2 0 } { 3 } } \end{array}$  ≈ 6.67liters. Next, let's determine the amount of water in the 15 liters of pineapple drink. Since the pineapple drink is three-fifths water, the amount of water in it is: Water in pineapple drink  $= { \frac { 3 } { 5 } } \times 1 5 = 9 \mathrm { l i t e r s } .$  Since you spill 1 liter of the orange drink, the amount of orange drink left is: Remaining orange drink =  $1 0 - 1 = \mathrm { \bar { 9 } l i t e r s }$  . Now, let's calculate the amount of water in the remaining 9 liters of orange drink: Water in remaining orange drink  $= { \begin{array} { l } { { \frac { 2 } { 3 } } } \end{array} } \times { \begin{array} { r } { 9 } } \end{array} = { \mathrm { ~ 6 l i t e r s } }$  The total amount of water in the combined mixture is the sum of the water in the remaining orange drink and the water in the pineapple drink: Total water  $= 6 + 9 = 1 5 { \mathrm { l i t e r s } } .$  The total volume of the mixture after spilling is: Total volume  $= 9 + 1 5 =$  24liters. Therefore, the amount of water in the remaining 24 liters is: Water in the remaining mixture = 15liters.&lt;/reason&gt;&lt;answer&gt;15&lt;/answer&gt;</td></tr></table>

Table 11: Illustration of generated responses under LONG mode on CSQA, including zero-shot, TTRL and PaceReasoner. The base model is Qwen3-4B-Base.

Table 12: Illustration of generated responses under LONG mode on GPQA, including zero-shot, TTRL and PaceReasoner. The base model is Qwen3-4B-Base.

Table 13: Illustration of generated responses under LONG mode on MATH-500, including zero-shot, TTRL and PaceReasoner. The base model is Qwen3-4B-Base.

Table 14: Illustration of generated responses under LONG mode on GSM8K, including zero-shot, TTRL and PaceReasoner. The base model is Qwen3-4B-Base.