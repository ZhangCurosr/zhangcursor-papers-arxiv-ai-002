# FROM IMITATION TO REWARD DISCOVERY: ON-POLICY WARMUP FOR AGENTIC RL

Yitong Qiao<sup>1,2,∗</sup>, Tiantian He<sup>2</sup>, Lei Liu<sup>1,2,†</sup>, Yue Shen<sup>2</sup>, Jian Wang<sup>2</sup>, Jinjie Gu<sup>2</sup>, Zhixuan Chu<sup>1,†</sup> <sup>1</sup>Zhejiang University <sup>2</sup>Ant Healthcare, Ant Group

qiaoyt@zju.edu.cn; liulei1497@gmail.com; zhixuanchu@zju.edu.cn

## ABSTRACT

Reinforcement learning with a verifiable reward (RLVR) offers a scalable approach to training language-model agents, yet sparse outcome rewards can leave early training with little signal for policy improvement. We identify an On-Policy Acceleration Phenomenon: in our main comparisons, RLVR initialized with onpolicy distillation reaches high performance earlier in training and achieves both higher average performance during subsequent RLVR and higher final performance than the alternative baselines. Motivated by this observation, we study On-Policy Warmup (OPW), a teacher-guided stage in which the student trains with teacher supervision on its own interaction trajectories before transitioning to RLVR. Unlike imitation on fixed teacher-generated trajectories, OPW targets states induced by the student’s own decisions, including imperfect actions and recovery situations. We provide a theoretical explanation by connecting on-policy reverse-KL distillation to trajectory-level distribution matching. Under a competent teacher and sufficiently small population distillation loss, this connection yields a lower bound on initial verifier success and a corresponding bound on reward-discovery complexity. For group-relative RLVR, we further characterize when increased success probability produces more reward-informative groups. Together, our findings support on-policy distillation as an effective warmup for agentic RLVR and identify initial reward discovery as a mechanism that can contribute to the observed acceleration.

![](images/498a21d1a4f58ddb8c95b2519db605bfbda4544baed7b90856ae0a0474e77f29.jpg)  
Less training time

Qwen3-1.7B  
![](images/8db691a2a247d4a5982f0cb6642b287b6891a8fcad289750f20d4b8ca678cce4.jpg)

![](images/a4c4573638f272831f202d757a7e4e9dc4b7407a8c79e3147ee8e5079f23f439.jpg)

Qwen3-4B  
![](images/ed875107e46487271581cf654599fb0979eec5ccb4784999ea8fbb458046d00f.jpg)  
Off-policy: SFT On-policy: GRPO OPW (Ours)  
Figure 1: OPW accelerates GRPO and improves repeated task success in ALFWorld. Left: Qwen3-4B Seen learning curves for GRPO with and without OPW. GPU-hours include teacher scoring and exclude evaluation. Right: Final Unseen repeated task success (pass^k).

## 1 INTRODUCTION

Reinforcement learning with verifiable rewards (RLVR) improves language-model reasoning using automatically checkable outcomes rather than human annotations (Shao et al., 2024; DeepSeek-AI et al., 2025). It is appealing for agents that interleave reasoning with tool use and environment feedback (Yao et al., 2023). Yet outcome-based feedback offers limited guidance about the intermediate decisions required for success. This creates a cold-start challenge: when success requires several coordinated decisions, an initial policy may produce mostly failures with identical rewards. Under binary outcome rewards, all-failure groups provide no within-group task-reward contrast for group-relative methods such as GRPO (Shao et al., 2024). Effective early training requires not just diverse trajectories, but successes frequent enough for the verifier to distinguish useful behavior.

Teacher supervision offers a way to improve this starting point, where knowledge distillation can transfer a denser supervision signal to a student (Hinton et al., 2015). However, imitation on fixed teacher-generated trajectories primarily supervises states visited by the teacher, whereas an agent must act on histories induced by its own decisions. This distribution mismatch is a central concern in imitation learning (Ross et al., 2011) and is particularly consequential in interactive tasks: an imperfect action can alter subsequent tool outputs, available information, and opportunities for recovery. On-policy distillation addresses the mismatch by applying teacher supervision to studentgenerated trajectories (Agarwal et al., 2024). Whether such supervision provides a useful initialization for subsequent RLVR, rather than merely improving imitation, remains a distinct question.

In this work, we identify an On-Policy Acceleration Phenomenon: after on-policy distillation warmup, RLVR reaches high performance earlier and achieves higher final performance than the compared baselines. Motivated by this observation, we propose On-Policy Warmup (OPW), a teacher-guided initialization stage for agentic RLVR. During warmup, the student generates trajectories by interacting with the environment and receives supervision from a fixed teacher. The student learns from this supervision before transitioning to RLVR, where training uses verifiable task outcomes without further teacher queries. Teacher supervision is confined to warmup.

The purpose of warmup is not simply to reduce distillation loss. The downstream objective is task success under sparse verifier feedback, and a student that imitates the teacher more closely need not, in general, learn faster during RLVR. Instead, we ask whether on-policy supervision makes successful trajectories accessible often enough to improve initial reward discovery (Chen et al., 2025). This question connects the distribution of trajectories induced by the warmed-up policy to the training signal available to the subsequent RLVR stage. It also separates the role of teacher guidance during initialization from the role of the verifier during later policy improvement.

We analyze this connection theoretically. Under shared environment dynamics, the cumulative reverse-KL distillation loss on student-visited states equals the KL divergence between student and teacher trajectory distributions. The data-processing inequality then bounds the gap between their success probabilities under the same binary verifier used to assess task completion. A sufficiently successful teacher and sufficiently small student population loss imply a positive lower bound on initial student success probability, and thus an upper bound on the expected rollouts to discover a success. For group-relative RLVR, we also characterize when higher success probability increases the likelihood of sampling a group containing both successes and failures. These results suggest initial reward discovery as a mechanism for the observed acceleration, but neither guarantee convergence nor imply lower total compute once warmup and teacher queries are included.

Our contributions are threefold:

• We identify the On-Policy Acceleration Phenomenon in agentic RLVR. In our main comparisons, on-policy distillation warmup leads to earlier high performance, higher average performance during subsequent RLVR, and higher final performance than the compared baselines.

• We study OPW as a teacher-guided initialization stage for RLVR. OPW provides teacher supervision along student-generated trajectories, including after unsuccessful actions, and is followed by RLVR without further teacher queries.

• We connect on-policy reverse-KL distillation to initial reward discovery. Through trajectorylevel distribution matching, we derive a lower bound on initial verifier success and bound rewarddiscovery complexity under sufficient teacher competence and small population distillation loss.

Table 1: Related work on warmup for RLVR.
<table><tr><td rowspan="2">Work</td><td colspan="2">Setting</td><td colspan="2">Evidence</td><td colspan="2">Training study</td></tr><tr><td>On-policy Environment warmup</td><td>interaction</td><td>Theoretical analysis</td><td>1 Action-level interventions</td><td>Training Warmup cost</td><td>duration</td></tr><tr><td>Sequential Beats Joint (Li et al., 2026a)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>RL Starts before RL (Dong et al., 2026)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>OPDSearch+ (Ye et al., 2026)</td><td></td><td>V</td><td>V</td><td>X</td><td></td><td>X</td></tr><tr><td>PRISM (Wang et al., 2026a)</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>PEAR (Zhang et al., 2026)</td><td>X</td><td>X</td><td></td><td>X</td><td>X</td><td>X</td></tr><tr><td>OPW (Ours)</td><td></td><td></td><td>L</td><td></td><td></td><td></td></tr></table>

## 2 RELATED WORK

Warmup for RLVR. RLVR trains language-model policies using verifiable outcome rewards (Shao et al., 2024). SFT is commonly used as warmup before RLVR in mathematical reasoning and logic tasks (Luong et al., 2024; Shrestha et al., 2025). Stronger SFT performance does not necessarily lead to better performance after RL (Kang et al., 2026; Zhang et al., 2026). PEAR reweights fixed offline supervision using importance sampling to improve subsequent RL (Zhang et al., 2026). TailSFT filters sequences according to their likelihood improvement during SFT to improve coverage and subsequent GRPO performance (Malladi et al., 2026). For on-policy distillation (OPD), Sequential Beats Joint compares sequential and joint OPD–RLVR training and examines when to switch to RL (Li et al., 2026a). RL Starts before RL shows that initial accuracy and pre-RL pass@k do not fully explain subsequent performance, and compares trajectory sources and divergence objectives (Dong et al., 2026). PRISM inserts a black-box, response-level adversarial alignment stage between SFT and multimodal RLVR (Wang et al., 2026a). Closest to our interactive setting, OPDSearch+ applies teacher supervision to student trajectories collected through live search interactions before RL refinement (Ye et al., 2026). Table 1 compares these warmup methods.

On-Policy Supervision for Interactive Agents. Knowledge distillation transfers teacher behavior to a student (Hinton et al., 2015). For language models and agents, a common approach is SFT on teacher-generated responses or interaction trajectories (DeepSeek-AI et al., 2025; Zeng et al., 2023). GKD instead applies teacher feedback to student-generated sequences, addressing the mismatch between training data and the student’s own outputs (Agarwal et al., 2024). MiniLLM studies reverse KL distillation to discourage the student from assigning excessive probability to regions that are unlikely under the teacher (Gu et al., 2024). Guided-OPD extends this line of work to multi-turn agents by mixing teacher- and student-generated turns and gradually reducing the probability of teacher intervention (Li et al., 2026b). Other methods combine ongoing RL with teacher trajectories (Yan et al., 2025), token-level guidance from interaction feedback (Wang et al., 2026b), or self-distillation from completed interactions (Wu et al., 2026). We study teacher supervision on student-generated trajectories as warmup before teacher-free RLVR in multi-turn environments. Under population reverse-KL assumptions, we connect trajectory distributions to initial verifier success and rewarddiscovery complexity. We remove supervision at repeated invalid actions, evaluate teacher-guided choices through task continuations from common histories, and track the retention of teacher-target preferences during GRPO. We also compare warmup durations and recorded training costs, including teacher supervision, to distinguish faster subsequent learning from lower total cost.

## 3 METHOD

We study OPD as warmup for RLVR. Our analysis connects distillation to trajectory-level distribution matching and informative reward observations during early RLVR.

## 3.1 THEORETICAL MOTIVATION FOR ON-POLICY WARMUP

Setup. Fix a task x and a finite-horizon interaction process with horizon H. Let $s _ { t }$ denote the full interaction history, including previous actions and environment observations, and let $a _ { t }$ denote the next policy action. Variable-length trajectories can be padded with an absorbing state and a shared dummy action, which contributes zero KL. For token-level language policies, t indexes policygenerated tokens, including reasoning and action tokens, and H bounds their number. The student policy $\pi$ and teacher policy $\pi _ { T }$ interact with the same environment and initial-state distribution. Let $P _ { \pi } ^ { x }$ and $P _ { T } ^ { x }$ denote their induced trajectory distributions. For a binary verifier $R _ { x } ( \tau ) \in \{ 0 , 1 \}$ indicating complete task success, define $\mathsf { \dot { p } } _ { \pi } ( x ) \mathsf { \dot { = } } \mathbb { E } _ { \tau \sim P _ { \pi } ^ { x } } [ R _ { x } ( \tau ) ]$ and $p _ { T } ( \mathbf { \bar { \boldsymbol { x } } } ) = \mathbb { E } _ { \tau \sim P _ { T } ^ { x } } [ \dot { R _ { x } } ( \tau ) ]$

All theoretical statements below are conditional on x; we suppress this dependence when unambiguous. We define the population on-policy distillation loss as

$$
{ \mathcal { L } } _ { \mathrm { O P D } } ( \pi ) = \sum _ { t = 1 } ^ { H } \mathbb { E } _ { s _ { t } \sim d _ { t } ^ { \pi } } \left[ D _ { \mathrm { K L } } \left( \pi ( \cdot \mid s _ { t } ) \parallel \pi _ { T } ( \cdot \mid s _ { t } ) \right) \right] ,\tag{1}
$$

where $d _ { t } ^ { \pi }$ is the student’s state distribution. We assume finite population loss and use natural logs. Proposition 3.1 (Trajectory-Level Interpretation). Under the shared-environment assumption, $D _ { \mathrm { K L } } \bar { ( } P _ { \pi } \| P _ { T } ) = \bar { \mathcal { L } } _ { \mathrm { O P D } } \bar { ( } \pi )$

Theorem 3.2 (Success Transfer from Teacher to Student). $I f \mathcal { L } _ { \mathrm { O P D } } ( \pi ) \leq \varepsilon$ , then $\mathrm { K L } ( p _ { \pi } \| p _ { T } ) \leq \varepsilon$ where KL denotes the KL divergence between Bernoulli distributions. In particular,

$$
| p _ { \pi } - p _ { T } | \leq \sqrt { \frac { \varepsilon } { 2 } } , \qquad p _ { \pi } \geq \left[ p _ { T } - \sqrt { \frac { \varepsilon } { 2 } } \right] _ { + } .\tag{2}
$$

Theorem 3.2 provides a sufficient condition for transferring verifier success from a competent teacher. When the loss is reported as an average over H steps, $\overline { { \mathcal { L } } } _ { \mathrm { O P D } } = \mathcal { L } _ { \mathrm { O P D } } / H$ , the corresponding deviation bound is $\sqrt { H \overline { { \mathcal { L } } } _ { \mathrm { O P D } } / 2 }$ . A small average token-level loss alone need not give a tight guarantee over long interactions, because loss accumulates across action positions.

Why On-Policy States Matter. The trajectory identity in Proposition 3.1 specifically requires distillation under the student’s distribution. To illustrate the role of state coverage, define $k _ { \pi } ( s ) =$ $D _ { \mathrm { K L } } ( \pi ( \cdot \textit { | s } ) \| \pi _ { T } ( \cdot \textit { | s } ) )$ and consider evaluating the same local reverse-KL discrepancy under an offline state distribution µ : $\begin{array} { r } { \mathcal { L } _ { \boldsymbol { \mu } } ( \boldsymbol { \pi } ) = \sum _ { t = 1 } ^ { H } \mathbb { E } _ { s \sim \mu _ { t } } [ k _ { \boldsymbol { \pi } } ( s ) ] } \end{array}$ . If $d _ { t } ^ { \pi }$ is absolutely continuous with respect to $\mu _ { t }$ and $\frac { d d _ { t } ^ { \pi } } { d \mu _ { t } } ( s ) \le C$ for every t and almost every $s ,$ then ${ \mathcal { L } } _ { \mathrm { O P D } } ( \pi ) \leq C { \mathcal { L } } _ { \mu } ( \pi )$ . The success-transfer bound obtained from this offline reverse-KL loss thus depends on an additional state-coverage coefficient C. If $d _ { t } ^ { \pi } \ll \mu _ { t }$ for any t, no finite $C$ exists. OPD directly targets the stateweighted discrepancy appearing in the trajectory KL, avoiding this additional distribution-transfer requirement at the population level. This is particularly relevant for agentic tasks, where imperfect actions can lead to novel tool outputs, execution failures, and recovery states.

## 3.2 IMPLICATIONS FOR EARLY RLVR REWARD DISCOVERY

For the student $\pi _ { \mathrm { w } }$ after OPW, Theorem 3.2 gives $p _ { \mathrm { w } } = p _ { \pi _ { \mathrm { w } } } \geq \ell = \left[ p _ { T } - \sqrt { { \mathcal L } _ { \mathrm { O P D } } ( \pi _ { \mathrm { w } } ) / 2 } \right] _ { + }$

Corollary 3.3 (Initial Reward-Discovery Complexity). Suppose $\ell > 0 .$ . For independent rollouts from the fixed warm-start policy $\pi _ { \mathrm { w } } ,$ , the probability of observing at least one successful trajectory among N rollouts satisfies

$$
\operatorname* { P r } ( a t l e a s t o n e s u c c e s s ) = 1 - ( 1 - p _ { \mathrm { w } } ) ^ { N } \geq 1 - ( 1 - \ell ) ^ { N } \geq 1 - e ^ { - N \ell } .\tag{3}
$$

For any $\begin{array} { r } { \delta \in ( 0 , 1 ) , N \geq \left\lceil \frac { \log ( 1 / \delta ) } { \ell } \right\rceil } \end{array}$ is sufficient to observe a success with probability at least $1 - \delta .$ The expected number ofrollouts until thefirst success is at most $1 / \ell$ under the samefixed policy.

If the pre-warmup success probability is $p _ { 0 }$ and $\ell > p _ { 0 }$ , OPD is certified to improve this initial discovery process. For group-relative methods with binary rewards, the relevant event is observing successes and failures in the same group. For $G \geq 2$ independent rollouts on the same task, its probability is $m _ { G } ( p ) = 1 - p ^ { G } - ( \check { 1 } - p \check { ) } ^ { G }$ . Such groups contain nonzero empirical reward variance and can support reward-based differentiation among trajectories. The derivative $m _ { G } ^ { \prime } ( p ) = G \big ( ( 1 -$ $p ) ^ { G - 1 } - p ^ { G - 1 } )$ is nonnegative on $[ 0 , 1 / 2 ]$ . Therefore, in the sparse-success regime $p _ { 0 } < p _ { \mathrm { w } } \le 1 / 2$ increasing success probability increases the frequency of reward-informative groups. The lower bound $p _ { \mathrm { w } } \geq \ell$ alone does not imply an increase in reward-informative groups. This monotonicity does not hold globally: near $p = 1$ , groups can become uninformative because all trajectories succeed. For partial-credit rewards, unsuccessful trajectories can still receive different scores.

Table 2: Final task performance across three training paths after 100 steps (Qwen3-4B).
<table><tr><td rowspan="2">Split</td><td></td><td>Task type</td><td colspan="6">ALFWorld</td><td colspan="8">ScienceWorld</td></tr><tr><td>Training</td><td></td><td>Pick Clean</td><td>Heat</td><td>Cool</td><td></td><td>Exam Pick2</td><td></td><td>All</td><td>Find</td><td>Life</td><td>Gene.</td><td>Mix</td><td>Prop.</td><td>Phys.</td><td>All</td></tr><tr><td rowspan="3">Seen</td><td>Direct OPD</td><td></td><td>88.6 30.1</td><td>9.4</td><td></td><td>12.5</td><td>51.9</td><td>57.3</td><td>45.9</td><td>61.4</td><td>46.2</td><td>29.6</td><td>30.4</td><td>52.1</td><td>44.2</td><td>48.9</td></tr><tr><td>Direct GRPO</td><td></td><td>100.0 83.3</td><td>78.9</td><td></td><td>83.5</td><td>85.6</td><td>95.8</td><td>89.4</td><td>80.3</td><td>39.8</td><td>55.3</td><td>25.1</td><td>61.1</td><td>62.1</td><td>61.2</td></tr><tr><td>OPW (Ours) + GRPO</td><td></td><td>98.9 98.1</td><td>86.7</td><td>93.5</td><td></td><td>97.1</td><td>97.9</td><td>96.1</td><td>97.0</td><td>53.6</td><td>78.7</td><td>30.4</td><td>98.0</td><td>82.4</td><td>87.3</td></tr><tr><td rowspan="3">Unseen</td><td>Direct OPD</td><td></td><td>90.6 54.0</td><td></td><td>12.0</td><td>48.2</td><td>56.9</td><td>59.6</td><td>53.5</td><td>64.8</td><td>45.4</td><td>47.7</td><td>8.1</td><td>60.8</td><td>47.1</td><td>55.2</td></tr><tr><td>Direct GRPO</td><td></td><td>88.5 82.3</td><td>83.7</td><td>90.5</td><td></td><td>83.3</td><td>94.1</td><td>86.6</td><td>74.0</td><td>50.7</td><td>50.4</td><td>13.6</td><td>63.8</td><td>61.0</td><td>61.2</td></tr><tr><td>OPW (Ours) + GRPO</td><td></td><td>93.8 100.0</td><td>92.4</td><td>99.4</td><td></td><td>97.2</td><td>97.1</td><td>96.7</td><td>71.8</td><td>62.5</td><td>52.4</td><td>15.4</td><td>75.8</td><td>74.9</td><td>68.2</td></tr></table>

## 3.3 ON-POLICY WARMUP FOR RLVR

Motivated by this connection, we propose On-Policy Warmup (OPW): a teacher-guided initialization stage preceding RLVR. Starting from the base model, the pipeline is $\pi _ { \theta _ { 0 } } \xrightarrow [ ] { \mathrm { O P D } } \pi _ { \theta _ { \mathrm { w } } } \xrightarrow [ ] { \mathrm { R L V R } } \pi _ { \theta _ { \mathrm { f i n a l } } } .$

Stage I: Teacher Supervision on Student Trajectories. Let $\mathcal { D } _ { \mathrm { w } }$ and $\mathcal { D } _ { \mathrm { R L } }$ denote the warmup and RLVR task distributions, and $\pi _ { T }$ a fixed teacher policy. Our experiments use disjoint training task pools. At warmup iteration $k ,$ we sample a task $x \sim \mathcal { D } _ { \mathrm { w } }$ and generate an interaction trajectory using the current student $\tau \sim P _ { \pi _ { \theta _ { k } } } ^ { x }$ . We query the teacher at interaction histories along these trajectories. Unlike offline imitation of teacher trajectories, OPW exposes the teacher to states induced by the student’s own decisions, including imperfect intermediate solutions, failed tool calls, and recovery.

The population reverse-KL objective considered in our analysis is ${ \mathcal { L } } _ { \mathrm { { O P W } } } ( \theta )$ = $\mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { w } } } \left[ \mathcal { \bar { L } } _ { \mathrm { O P D } } ( \pi _ { \theta } ; \boldsymbol { x } ) \right]$ , using the task-specific loss in Eq. 1. For this reverse-KL formulation, the corresponding fixed-history surrogate at iteration k is

$$
\widetilde { \mathcal { L } } _ { k } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } _ { \mathbf { w } } , \tau \sim P _ { \pi _ { \theta _ { k } } } ^ { x } } \left[ \sum _ { { t = 1 } } ^ { H } D _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot  { | } s _ { t } )  { \| } \pi _ { T } ( \cdot  { | } s _ { t } ) \right) \right] .\tag{4}
$$

Sampled histories are held fixed during each step, and trajectories are resampled as the student changes. Teacher supervision is applied to policy-generated tokens, not to environment observations.

Stage II: Verifier-Based Reinforcement Learning. After a prescribed warmup duration, we initialize RLVR with $\pi _ { \theta _ { \mathrm { w } } }$ and optimize

$$
J _ { \mathrm { V R } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { R L } } , \boldsymbol { \tau } \sim P _ { \pi _ { \theta } } ^ { x } } \left[ r _ { x } ( \boldsymbol { \tau } ) \right] , \qquad \theta \gets \theta _ { \mathrm { w } } ,\tag{5}
$$

where $r _ { x } ( \tau ) \in [ 0 , 1 ]$ is the training reward. It equals the binary verifier $R _ { x }$ in ALFWorld and includes partial progress in ScienceWorld. We use GRPO’s group-relative updates without probability-ratio clipping or additional KL or entropy terms (Table 8). In the basic $\mathrm { O \bar { P } W }$ pipeline, the distillation loss is removed after warmup and no further teacher queries are required.

Scope. The bounds concern initial reward discovery under task-specific population reverse-KL control, with success measured under the same policy distributions. The analysis does not establish that finite warmup meets this condition or that it transfers across tasks or decoding settings.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate agents on ALFWorld (Shridhar et al., 2021) and ScienceWorld (Wang et al., 2022), which require multi-step interaction to complete household and scientific tasks. ALFWorld covers placing, cleaning, heating, cooling, examining objects under a light, and placing two objects, with 3,553 training tasks and 140 Seen and 134 Unseen test tasks. ScienceWorld contains 2,294 training tasks, of which 64 are reserved for Seen evaluation, and 200 Unseen test tasks. We group its tasks into finding, growth/lifespan, genetics, mixing, properties, and mechanics/energy.

Baselines. We use Qwen3-1.7B and Qwen3-4B as students and Qwen3-32B as the teacher (Yang et al., 2025). All RLVR training uses GRPO. OPW uses cross-entropy on teacher top-1 targets at response-token positions in student-generated trajectories. We compare OPW with (1) off-policy SFT, (2) SFT with rejection-sampling (SFT-RS), and (3) teacher-free on-policy GRPO warmup as baselines. Each warmup method runs for 40 steps and is followed by 60 GRPO steps without further teacher supervision. We compare OPW + GRPO with direct OPD and direct GRPO over 100 steps.

Table 3: Final task performance after 40 warmup and 60 GRPO steps (Qwen3-4B).
<table><tr><td rowspan="2">Split</td><td></td><td>Task type</td><td colspan="6">ALFWorld</td><td colspan="6">ScienceWorld</td></tr><tr><td>Warmup</td><td>Pick</td><td>Clean</td><td>Heat</td><td>Cool</td><td></td><td>Exam Pick2</td><td>All</td><td>Find</td><td>Life</td><td>Gene.</td><td>Mix</td><td>Prop.</td><td>Phys. All</td></tr><tr><td rowspan="4">Seen</td><td>Off-policy: SFT</td><td>97.5</td><td>82.4</td><td>75.8</td><td>74.5</td><td>66.3</td><td>90.6</td><td>83.9</td><td>69.9</td><td>49.3</td><td>67.8</td><td>42.8</td><td>58.2 63.0</td><td>60.5</td></tr><tr><td>Off-policy: SFT-RS</td><td>98.9</td><td>90.7</td><td>78.1</td><td>78.5</td><td>88.5</td><td>91.1</td><td>89.0</td><td>79.7</td><td>50.2 58.5</td><td>34.6</td><td>67.1</td><td>62.0</td><td>64.6</td></tr><tr><td>On-policy: GRPO</td><td>99.3</td><td>74.1</td><td>78.9</td><td>69.5</td><td>70.2</td><td>91.1</td><td>82.7</td><td>81.5</td><td>51.2</td><td>61.1 35.3</td><td>67.2</td><td>73.3</td><td>68.8</td></tr><tr><td>OPW (Ours)</td><td>98.9</td><td>98.1</td><td>86.7</td><td>93.5</td><td>97.1</td><td>97.9</td><td>96.1</td><td>97.0</td><td>53.6</td><td>78.7 30.4</td><td>98.0</td><td>82.4</td><td>87.3</td></tr><tr><td rowspan="4">Unseen</td><td>Off-policy: SFT</td><td>88.0</td><td>74.6</td><td>82.1</td><td>81.5</td><td>72.2</td><td>84.6</td><td>80.3</td><td>71.2</td><td>53.0</td><td>45.3</td><td>14.4 68.6</td><td>61.6</td><td>61.9</td></tr><tr><td>Off-policy: SFT-RS</td><td>88.0</td><td>91.1</td><td>81.5</td><td>92.9</td><td>85.4</td><td>86.0</td><td>87.8</td><td>70.2</td><td>54.3</td><td>43.0 12.4</td><td>75.4</td><td>59.9</td><td>63.6</td></tr><tr><td>On-policy: GRPO</td><td>85.4</td><td>78.2</td><td>71.2</td><td>77.4</td><td>66.7</td><td>77.2</td><td>76.5</td><td>68.4</td><td>54.5</td><td>50.0 15.8</td><td>61.5</td><td>60.0</td><td>59.6</td></tr><tr><td>OPW (Ours)</td><td>93.8</td><td>100.0</td><td>92.4</td><td>99.4</td><td>97.2</td><td>97.1</td><td>96.7</td><td>71.8</td><td>62.5</td><td>52.4 15.4</td><td>75.8</td><td>74.9</td><td>68.2</td></tr></table>

Table 4: Final repeated task success (pass^8, %) for three training paths at 100 steps (Qwen3-4B).
<table><tr><td rowspan="2">Split</td><td></td><td>Task type</td><td colspan="6">ALFWorld</td><td colspan="6">ScienceWorld</td></tr><tr><td>Training</td><td></td><td>Pick Clean</td><td>Heat</td><td>Cool</td><td></td><td>Exam Pick2</td><td>All</td><td>Find</td><td>Life</td><td>Gene. 1</td><td>Mix</td><td>Prop.</td><td>Phys. All</td></tr><tr><td rowspan="3">Seen</td><td>Direct OPD</td><td></td><td>80.0 25.9</td><td>0.0</td><td>0.0</td><td>23.1</td><td>16.7</td><td>30.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0 0.0</td><td>4.8</td><td>1.6</td></tr><tr><td>Direct GRPO</td><td>100.0</td><td>48.1</td><td>68.8</td><td>60.0</td><td>38.5</td><td>87.5</td><td>71.4</td><td>0.0</td><td>0.0</td><td>0.0 0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>OPW (Ours) + GRPO</td><td>97.1</td><td>92.6</td><td>81.2</td><td>80.0</td><td>84.6</td><td>83.3</td><td>87.9</td><td>62.5</td><td>25.0 50.0</td><td>0.0</td><td>63.0</td><td>19.0</td><td>43.8</td></tr><tr><td rowspan="3">Unseen</td><td>Direct OPD</td><td></td><td>83.3 41.9</td><td>0.0</td><td>9.5</td><td>11.1</td><td>5.9</td><td>28.4</td><td>1.9</td><td>2.9</td><td>12.0</td><td>0.0 0.0</td><td>0.0</td><td>2.5</td></tr><tr><td>Direct GRPO</td><td>79.2</td><td>38.7</td><td>56.5</td><td>61.9</td><td>72.2</td><td>76.5</td><td>61.9</td><td>9.6</td><td>2.9</td><td>8.0 0.0</td><td>3.1</td><td>21.1</td><td>7.0</td></tr><tr><td>OPW (Ours) + GRPO</td><td>87.5</td><td>100.0</td><td>82.6</td><td>95.2</td><td>77.8</td><td>82.4</td><td>88.8</td><td>3.8</td><td>14.3 16.0</td><td>0.0</td><td>4.6</td><td>15.8</td><td>8.5</td></tr></table>

Evaluation metrics. Task performance is mean task success (pass@1, %) in ALFWorld and normalized task scores (0–100) in ScienceWorld. We measure success coverage by pass@k (at least one success in k rollouts) (Chen et al., 2021), and repeated task success by pass^k (success in every rollout) (Yao et al., 2024; Barres et al., 2025). ScienceWorld scores include partial progress, while success requires full task completion. Final results use eight trajectories per task, and the warmup success profile uses 64. Training and evaluation details are provided in Appendix B.

## 4.2 RESULTS

Warmup VS. Direct Training. OPW followed by GRPO achieves the best final task performance among the three training paths in both environments (Table 2). On ALFWorld Unseen, it reaches 96.7% success, compared with 86.6% for direct GRPO and 53.5% for direct OPD. The same ordering holds in ScienceWorld, where Unseen task scores reach 68.2, 61.2, and 55.2, respectively. These results support on-policy distillation as an effective warmup for agentic RLVR.

Comparison of Warmup Strategies. OPW yields the highest final task performance under the same GRPO procedure after warmup (Table 3). On ALFWorld Unseen, restricting SFT to successful teacher trajectories raises final success from 80.3% to 87.8%, while OPW reaches 96.7%, compared with 76.5% after GRPO warmup. In ScienceWorld, OPW exceeds the strongest alternative by 18.5 points on Seen tasks and 4.6 points on Unseen tasks. Gains over both SFT variants and teacher-free warmup support teacher supervision on student-generated trajectories during warmup.

Task-Specific Performance. In ALFWorld, Clean, Heat, and Cool require changing an object’s state before placement, adding intermediate requirements to the retrieval and placement needed for Pick (Shridhar et al., 2021). OPW improves these tasks in both comparisons (Tables 2 and 3). In the warmup comparison, Seen Pick success already exceeds 97% for all methods, while OPW improves Clean and Cool success over GRPO warmup by 24.0 percentage points each. In ScienceWorld, Mixing requires selecting and combining materials to obtain a target product (Wang et al., 2022).

Table 5: Success coverage and repeated task success after warmup (pass@k/pass^k, %).
<table><tr><td>Split / k</td><td colspan="4">Seen</td><td colspan="4">Unseen</td></tr><tr><td>Policy</td><td>k = 8</td><td>k = 16</td><td>k = 32</td><td>k = 64</td><td>k = 8</td><td>k = 16</td><td>k = 32</td><td>k = 64</td></tr><tr><td colspan="9">ALFWorld</td></tr><tr><td>Teacher (32B) Base (4B)</td><td>70.9/27.5 51.5/15.8</td><td>76.4/21.5 58.4/13.2</td><td>81.0/15.2 64.6/10.9</td><td>85.7/10.0 69.3/9.3</td><td>88.3/24.2 55.2/10.9</td><td>93.8/17.3 63.1/8.0</td><td>97.1/11.5 70.7/5.9</td><td>99.3/6.7 78.4/4.5</td></tr><tr><td>Off-policy: SFT</td><td>52.9/12.6</td><td>60.0/9.3</td><td>66.1/6.7</td><td>71.4/4.3</td><td>58.7/7.7</td><td>67.8/5.2</td><td>75.8/3.8</td><td>82.8/3.0</td></tr><tr><td>Off-policy: SFT-RS</td><td>53.2/12.6</td><td>60.7/9.1</td><td>67.5/6.4</td><td>73.6/4.3</td><td>59.8/7.0</td><td>69.0/3.7</td><td>76.8/1.9</td><td>84.3/1.5</td></tr><tr><td>On-policy: GRPO</td><td>78.6/28.4</td><td>83.6/23.0</td><td>87.9/18.6</td><td>91.4/15.0</td><td>80.1/19.8</td><td>84.8/13.9</td><td>88.1/9.0</td><td>90.3/5.2</td></tr><tr><td>OPW (Ours)</td><td>69.3/28.1</td><td>75.2/24.6</td><td>79.5/22.4</td><td>82.9/21.4</td><td>88.1/27.5</td><td>93.2/22.9</td><td>95.0/19.7</td><td>95.5/17.9</td></tr><tr><td colspan="9"></td></tr><tr><td>Teacher (32B) Base (4B)</td><td>75.8/0.6</td><td>85.0/0.0</td><td>ScienceWorld 91.9/0.0</td><td>95.3/0.0</td><td>76.3/3.1</td><td>84.5/1.1</td><td>88.6/0.2</td><td>90.5/0.0</td></tr><tr><td>Off-policy: SFT</td><td>23.9/0.0 25.6/0.0</td><td>28.9/0.0 31.6/0.0</td><td>32.9/0.0 37.0/0.0</td><td>35.9/0.0 42.2/0.0</td><td>35.9/2.9 37.1/2.1</td><td>44.6/1.7 45.8/0.7</td><td>53.5/0.8</td><td>62.0/0.0</td></tr><tr><td>Off-policy: SFT-RS</td><td>22.0/0.0</td><td>27.2/0.0</td><td>31.9/0.0</td><td>35.9/0.0</td><td>38.0/2.1</td><td>46.6/0.7</td><td>54.4/0.1 55.2/0.1</td><td>63.0/0.0 63.5/0.0</td></tr><tr><td>On-policy: GRPO</td><td>51.5/1.4</td><td>56.3/0.1</td><td>59.5/0.0</td><td>62.5/0.0</td><td>62.9/5.4</td><td>71.1/2.3</td><td>77.4/0.8</td><td>82.5/0.5</td></tr><tr><td>OPW (Ours)</td><td>73.7/7.0</td><td>82.3/3.5</td><td>87.6/1.7</td><td>90.6/0.0</td><td>66.8/4.5</td><td>75.9/2.0</td><td>82.4/0.9</td><td>85.5/0.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Each cell: pass@k/pass^k (%). Best and second-best nonzero values per metric and column, including Teacher.

![](images/92d25d7a3e71de5d6356eae2fabad02dcc769d97b6dd71c0443da7d7716476de.jpg)  
Figure 2: Task performance during warmup and GRPO.

OPW does not lead this category despite its higher scores on Properties and Mechanics/Energy, showing that its gains do not extend uniformly across scientific tasks (Table 3).

Repeated Task Success. OPW improves repeated task success in both environments (Table 4). On ALFWorld Unseen, final pass^8 reaches 88.8%, compared with 61.9% for direct GRPO and 28.4% for direct OPD. In ScienceWorld, OPW followed by GRPO reaches final pass^8 of 43.8% on Seen tasks and 8.5% on Unseen tasks, compared with 0.0% and 7.0% for direct GRPO. Thus, more tasks are completed successfully in every rollout, although repeated task success remains difficult on ScienceWorld Unseen. The comparison after warmup further distinguishes success coverage from repeated task success (Table 5). On ALFWorld Seen, OPW has lower pass@64 than GRPO warmup (82.9% versus 91.4%) but higher pass^64 (21.4% versus 15.0%). Its stronger final task performance therefore does not require the broadest initial success coverage. We next examine how warmup affects subsequent learning and the behavior generated during teacher-free GRPO.

![](images/1079c662369cdb7024dedc74687aa0ef545e797054c06453cbbc760df443f2a1.jpg)

Table 6: Invalid actions after warmup (Qwen3-4B). Values are task-mean shares of all actions (%) across Seen and Unseen.
<table><tr><td colspan="2">Type</td><td colspan="3">ALFWorld</td><td colspan="3">ScienceWorld</td></tr><tr><td>Warmup</td><td></td><td>Nonrep.</td><td>Repeat</td><td>Consec.</td><td>Nonrep.</td><td>Repeat</td><td>Consec.</td></tr><tr><td>SFT</td><td></td><td>3.05</td><td>10.10</td><td>5.90</td><td>16.23</td><td>24.91</td><td>13.68</td></tr><tr><td>SFT-RS</td><td></td><td>3.12</td><td>10.24</td><td>5.90</td><td>16.11</td><td>23.79</td><td>12.88</td></tr><tr><td>GRPO</td><td></td><td>3.52</td><td>6.78</td><td>5.25</td><td>29.05</td><td>23.15</td><td>8.01</td></tr><tr><td>OPW (Ours)</td><td></td><td>3.14</td><td>3.19</td><td>0.53</td><td>30.62</td><td>19.17</td><td>4.02</td></tr></table>

Table 7: Task performance for fixed and resampled trajectories (ALF-World 4B, Seen/Unseen).  
![](images/eb0ad6885e207207e6a07150d8f2eebb8b28d6f539f3b2093d587914be76fd59.jpg)

<table><tr><td>Metric</td><td>Fixed</td><td>OPW (Ours)</td></tr><tr><td>Early gain (pp)</td><td>38.6/29.2</td><td>41.8/31.4</td></tr><tr><td>Average (%)</td><td>81.1/81.4</td><td>84.7/86.6</td></tr><tr><td>Final (%)</td><td>97.4/95.1</td><td>96.1/96.7</td></tr></table>

Figure 3: Distinct successful action sequences among eight rollouts per task during GRPO.

## 5 MORE ANALYSIS

Learning Efficiency and Training Cost. OPW improves subsequent learning even from lower warmup success. On ALFWorld 4B Seen, it starts below GRPO warmup (46.8% versus 56.0%) but gains 41.8 versus 11.4 percentage points in the first 20 GRPO steps after warmup. Its average success over all 60 steps is 84.7% versus 69.5% (Fig. 2). OPW has the highest average task performance across environments, model sizes, and splits, even on ScienceWorld 1.7B Unseen, where GRPO warmup yields a larger early gain. Warmup scores alone do not explain subsequent learning.

OPW exceeds 80% Seen success on ALFWorld 4B at 19.6 recorded training GPU-hours including teacher supervision (Appendix C), versus at least 29.6 for other methods. Only OPW exceeds 90% within 100 steps. Over 100 steps, OPW costs less than GRPO warmup at 4B but more at 1.7B.

Repeated Invalid Actions. OPW’s main reduction in ALFWorld action errors concerns repeated invalid commands. We classify validity by membership in the current valid-action list and repetition by earlier use in the trajectory. Unparseable responses are invalid. Repeated invalid commands account for 3.19% of actions after OPW, versus 10.10% after SFT and 6.78% after GRPO warmup, while non-repeated invalid actions remain near 3% (Table 6). In ScienceWorld, OPW reduces repeated errors but produces more non-repeated invalid actions than SFT. This list-based measure can also mark accepted navigation aliases as invalid. Similar fractions of SFT and OPW trajectories contain an invalid action on ALFWorld Seen (50.9% and 50.0%), despite different levels of repetition. Appendix H illustrates these errors and valid repetition in two complete action sequences.

To test the contribution of supervision at these errors, we remove teacher supervision at repeated invalid actions. Early GRPO gain on ALFWorld 4B Seen falls to 34.6 percentage points, compared with 42.0 under random removal of a similar number of action tokens (Table 18). Warmup action validity remains similar. Supervision at repeated errors contributes to subsequent learning.

Teacher Supervision on Student-Generated Trajectories. Teacher supervision provides a learning signal on unsuccessful student trajectories. In all-failure groups on ALFWorld 4B, one optimizer update from Base raises mean teacher-target log probability at teacher–student disagreements by 0.183 with OPD, versus 0.128 with SFT and 0.004 with GRPO (Appendix D). Compared with reusing initial student trajectories, resampling improves average success during GRPO by 3.6 percentage points on Seen and 5.2 on Unseen (Table 7). Fixed trajectories still yield higher final Seen success, so resampling mainly benefits early learning and average success during GRPO.

![](images/f1435c26c45de98eb3333c00f914c914b042a2ee956f06645f2b24f04a827a91.jpg)  
Figure 4: Repeated task success (pass^8) across OPW durations (Qwen3-1.7B).

Task Value and Retention of Teacher Corrections. Teacher substitution raises continuation success from 26.48% to 29.70% at 311 selected action-token disagreements (Appendix E.1). Base continues both branches. Over 40 teacher-free GRPO steps, teacher-target probability rises from 39.3% to 47.2% at these positions but changes little on a broader disagreement set, from 43.8% to 43.2% (Table 19). The broader set shows retention, with further gains at selected action positions.

A complementary evaluation (Appendix E) samples complete actions from each policy at common histories, with Base taking all subsequent actions. At 447 active positions from 63 tasks, continuation success increases from 45.00% to 46.75% over 40 GRPO steps after OPW, compared with 45.93% to 46.47% after GRPO warmup. OPW again improves more from a lower starting score. With Base fixed, these gains measure action selection at the evaluated histories.

Diversity of Successful Action Sequences. OPW samples more distinct successful action sequences early in GRPO across both environments and model sizes (Fig. 3). After sequence normalization (Appendix F), its mean count among eight rollouts per task rises from 2.23 to 4.00 over 60 GRPO steps on ALFWorld 4B Seen. Failure-only sequences fall from 2.95 to 0.31, accounting for the lower total sequence count. Among four selected successful trajectories on common tasks with enough successes under every method and checkpoint, OPW’s expected distinct-sequence count falls from 2.54 to 2.14 on ALFWorld 4B but rises from 1.83 to 2.32 on ScienceWorld 4B. More successful sequences in eight rollouts need not mean more varied successful trajectories.

Choosing the Warmup Duration. Teacher supervision can improve repeated task success after most actions become valid. On ALFWorld 1.7B, extending OPW from 20 to 40 steps raises validity only from 89.4% to 91.8%, while pass^8 more than doubles from 10.8% to 22.0%. Fig. 4 compares 20, 40, 60, and 80 OPW steps followed by GRPO to 100 total steps, with varying seeds. The ALFWorld 20/40 comparison branches from the same warmup run. On Seen tasks, longer warmup raises final pass^8 from 69.3% to 80.7% and moves the first evaluation reaching 80% success from total step 75 to 60. In a separate ScienceWorld comparison from one 40-step OPW checkpoint, switching immediately gives higher Unseen pass^8 than adding 20 OPW steps (4.5% versus 3.5%), despite a lower mean task score. Choose warmup duration by subsequent validation on the required task outcome. Additional duration comparisons and warmup diagnostics are provided in Appendix G.

## 6 CONCLUSION

We study On-Policy Warmup (OPW), which uses teacher supervision on student-generated trajectories before RLVR. Our main comparisons in ALFWorld and ScienceWorld show the On-Policy Acceleration Phenomenon, with OPW reaching high performance earlier and achieving higher average and final task performance. Our theory connects on-policy reverse-KL distillation to trajectory-level distribution matching, initial verifier success, and reward-discovery complexity under its stated conditions. Empirical analyses show that teacher-target probability advantages persist during teacher-free GRPO, alongside improved action selection and repeated task success. Together, these findings support OPW as an effective warmup for agentic RLVR.

## AI USE STATEMENT

AI assistants were used to refine the language and improve the clarity of the manuscript. All scientific claims, theoretical analyses, experimental results, and references were reviewed and verified by the authors, who take full responsibility for the final manuscript.

## REPRODUCIBILITY STATEMENT

Section 3 defines OPW and connects population reverse-KL distillation to initial verifier success and reward discovery, with assumptions and complete proofs in Appendix A. Section 4 specifies the models, benchmarks, baselines, and evaluation metrics. Appendix B details task partitions, agent interaction, the teacher top-1 cross-entropy implementation, optimization settings, rollout sampling, and metric estimators. The analyses in Section 5 are supported by protocols for teacher-target probability measurements, trajectory resampling, and supervision removal in Appendix D. Appendix E describes token and complete-action interventions at common histories, including position selection, continuation policies, aggregation, and measurement of teacher-target preferences during teacher-free GRPO. Appendix F defines action-error and successful action-sequence measurements. Appendices C and G document additional training-length and seed comparisons, training-cost accounting including teacher supervision, and warmup-duration allocations. Appendix H provides complete action traces with selected environment feedback.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv preprint arXiv:2506.07982, 2025. URL https://arxiv.org/abs/2506.07982.

Fan Chen, Audrey Huang, Noah Golowich, Sadhika Malladi, Adam Block, Jordan T. Ash, Akshay Krishnamurthy, and Dylan J. Foster. The Coverage Principle: How Pre-Training Enables Post-Training. arXiv preprint arXiv:2510.15020, 2025. URL https://arxiv.org/abs/2510. 15020.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374, 2021. URL https: //arxiv.org/abs/2107.03374.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang

Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501.12948v1.

Shuai Dong, Yongfu Zhu, Yuqi Xu, Weichu Xie, Liuwenpu, Ziyue Wang, Kaiwen Tuo, Congcong Wang, Siyuan Wang, Wenqi Shao, Shuai Yang, Ji Zhao, Caoyuan Ma, Wenzheng Chang, Taiqiang Wu, Xinlei Yu, Hongrui Wu, Xiaoxuan He, Fangke Chen, Dianyi Wang, Kanghui Tian, Sirry Chen, Xingyu Liu, Xiangnan Wu, Jiawei Guo, Haowen Hou, LingHan Chen, Zhongyu Wei, and Jiaqi Wang. RL Starts before RL: On Policy Distillation for Better Reinforcement Learning. arXiv preprint arXiv:2609.28145, 2026. URL https://arxiv.org/abs/2609.28145.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge Distillation of Large Language Models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 8ac015d409635f196f9e3e9dcfb9a94e-Paper-Conference.pdf.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531, 2015. URL https://arxiv.org/abs/1503.02531.

Feiyang Kang, Michael Kuchnik, Karthik Padthe, Marin Vlastelica, Ruoxi Jia, Carole-Jean Wu, and Newsha Ardalani. Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 56876–56918, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 5d6ae8ba43ecb378030753c4408ef9bd-Paper-Conference.pdf.

Boyan Li, Bingsen Chen, Chenghao Yang, Ping Nie, Chen Zhao, and Xi Ye. Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR. arXiv preprint arXiv:2609.04108, 2026a. URL https://arxiv.org/abs/2609.04108.

Gengsheng Li, Mao Zheng, Mingyang Song, Ruiqi Liu, Tianyu Yang, Jie Sun, Qiyong Zhong, Haiyun Guo, Junfeng Fang, Dan Zhang, and Jinqiao Wang. On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents. arXiv preprint arXiv:2606.15912, 2026b. URL https://arxiv.org/abs/2606.15912.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum? id=Bkg6RiCqY7.

Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. ReFT: Reasoning with Reinforced Fine-Tuning. arXiv preprint arXiv:2401.08967, 2024. URL https: //arxiv.org/abs/2401.08967.

Sadhika Malladi, Samy Jelassi, Dylan Foster, Jordan T. Ash, and Akshay Krishnamurthy. TailSFT: Filtered Fine-Tuning Improves Post-Training Performance. arXiv preprint arXiv:2608.25756, 2026. URL https://arxiv.org/abs/2608.25756.

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Geoffrey Gordon, David Dunson, and Miroslav Dudík (eds.), Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, pp. 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr. press/v15/ross11a.html.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, and Keith W. Ross. Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 14380–14401, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176- 332-6. doi: 10.18653/v1/2025.emnlp-main.727. URL https://aclanthology.org/2025. emnlp-main.727/.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2010.03768.

Tim van Erven and Peter Harremoës. Rényi Divergence and Kullback-Leibler Divergence. IEEE Transactions on Information Theory, 60(7):3797–3820, 2014. doi: 10.1109/TIT.2014.2320500. URL https://doi.org/10.1109/TIT.2014.2320500.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. ScienceWorld: Is your Agent Smarter than a 5th Grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, Abu Dhabi, United Arab Emirates, dec 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https://aclanthology.org/ 2022.emnlp-main.775/.

Sudong Wang, Weiquan Huang, Xiaomin Yu, Zuhao Yang, Hehai Lin, Keming Wu, Chaojun Xiao, Chen Chen, Wenxuan Wang, Beier Zhu, Yunjian Zhang, and Chengwei Qin. Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL. arXiv preprint arXiv:2604.28123, 2026a. URL https://arxiv.org/abs/2604.28123.

Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. OpenClaw-RL: Train Any Agent Simply by Talking. arXiv preprint arXiv:2603.10165, 2026b. URL https://arxiv. org/abs/2603.10165.

Jinyang Wu, Shuo Yang, Zhengxi Lu, Fan Zhang, Yuhao Shen, Lang Feng, Haoran Luo, Zheng Lian, Shuai Zhang, Zhengqi Wen, and Jianhua Tao. SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning. arXiv preprint arXiv:2607.14777, 2026. URL https: //arxiv.org/abs/2607.14777.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to Reason under Off-Policy Guidance. arXiv preprint arXiv:2504.14945, 2025. URL https://arxiv.org/abs/2504.14945.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin

Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2210.03629.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv preprint arXiv:2406.12045, 2024. URL https://arxiv.org/abs/2406.12045.

Qinglin Ye, Zhiyuan Gu, Jingjie Xia, Yiheng Zhang, Kaiyan Zhao, Shunchao Zheng, Yuhang Mu, Wenchao Du, and Yiming Wang. OPDSearch+: On-Policy Distillation with RL Refinement for Search-Augmented Reasoning. arXiv preprint arXiv:2608.24310, 2026. URL https://arxiv. org/abs/2608.24310.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. AgentTuning: Enabling Generalized Agent Abilities for LLMs. arXiv preprint arXiv:2310.12823, 2023. URL https://arxiv.org/abs/2310.12823.

Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, and Hao Peng. Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning. arXiv preprint arXiv:2602.01058, 2026. URL https://arxiv.org/abs/2602.01058.

## APPENDIX CONTENTS

A Proofs and Theoretical Details 14   
A.1 Trajectory KL and Success Transfer 14   
A.2 Reward Discovery and Reward-Informative Groups 16   
B Training and Evaluation Protocol 16   
B.1 Data, Models, and Agent Interaction 16   
B.2 Training Objectives and Optimization 17   
B.3 Evaluation and Statistical Summaries . 17   
C Additional Warmup Comparisons 18   
C.1 Full Warmup Results 18   
C.2 Sensitivity to Training Length and Seed 18   
C.3 Training Computation and Target-Reaching Cost 20   
D Teacher Supervision on Student-Generated Trajectories 21   
D.1 Teacher-Target Probabilities after One Optimizer Step . 21   
D.2 Resampling Student Trajectories 21   
D.3 Removing Supervision at Repeated Invalid Actions 22   
E From Teacher Targets to Action Value 23   
E.1 Teacher Alternatives, Task Value, and Retention 23   
E.2 Complete-Action Value at Common Histories 23   
F Behavior after Warmup and during GRPO 24   
F.1 Execution Errors and Repetition 24   
F.2 Successful and Failure-Only Action Sequences 25   
G Warmup Duration and Repeated Task Success 25   
G.1 What Continues to Improve during Warmup? 25   
G.2 Learning Speed and Repeated Task Success 26   
H Long-Trajectory Examples 27   
I Limitations and Scope 29   
A PROOFS AND THEORETICAL DETAILS   
A.1 TRAJECTORY KL AND SUCCESS TRANSFER

Fix a task x and suppress its index. The student π and teacher $\pi _ { T }$ induce distributions $P _ { \pi }$ and $P _ { T }$ over horizon H, with the same initial distribution $\rho$ and transition kernels $K _ { t } ( \cdot \mid s _ { t } , a _ { t } )$ . States contain full interaction histories, and $d _ { t } ^ { \pi }$ is the student’s state distribution. Padding uses an absorbing state and a shared deterministic null action with zero KL. We use the cumulative objective in Eq. 1 and natural logarithms, with $0 \log ( 0 / q ) = 0$ and $p \log ( p / 0 ) = + \infty$ for $p > 0$

ProofofProposition 3.1. Write a trajectory as $\tau = \left( s _ { 1 } , a _ { 1 } , \dots , s _ { H } , a _ { H } , s _ { H + 1 } \right)$ . Using probabilitymass notation, its probability under the student factorizes as

$$
P _ { \pi } ( \tau ) = \rho ( s _ { 1 } ) \prod _ { t = 1 } ^ { H } \pi ( a _ { t } \mid s _ { t } ) K _ { t } ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) ,
$$

and the teacher trajectory probability $P _ { T } ( \tau )$ has the same factorization with π replaced by $\pi _ { T }$

First suppose that $P _ { \pi } \ll P _ { T }$ . The initial-state and environment-transition factors are identical under the two policies and therefore cancel in the likelihood ratio. Consequently, $P _ { \pi }$ -almost surely,

$$
\log { \frac { P _ { \pi } ( \tau ) } { P _ { T } ( \tau ) } } = \sum _ { t = 1 } ^ { H } \log { \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { T } ( a _ { t } \mid s _ { t } ) } } .
$$

Taking expectation under $P _ { \pi }$ and conditioning on $s _ { t }$ yields

$$
\begin{array} { l } { { \cal D } _ { \mathrm { K L } } ( P _ { \pi } \| P _ { T } ) = \mathbb { E } _ { \tau \sim P _ { \pi } } \left[ \displaystyle \sum _ { t = 1 } ^ { H } \log \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { T } ( a _ { t } \mid s _ { t } ) } \right] } \\ { = \displaystyle \sum _ { t = 1 } ^ { H } \mathbb { E } _ { s _ { t } \sim d _ { \tau } ^ { \pi } } \left[ \mathbb { E } _ { a _ { t } \sim \pi ( \cdot \mid s _ { t } ) } \left[ \log \frac { \pi ( a _ { t } \mid s _ { t } ) } { \pi _ { T } ( a _ { t } \mid s _ { t } ) } \right] \right] } \\ { = \displaystyle \sum _ { t = 1 } ^ { H } \mathbb { E } _ { s _ { t } \sim d _ { \tau } ^ { \pi } } \left[ { \cal D } _ { \mathrm { K L } } \left( \pi ( \cdot \mid s _ { t } ) \| \pi _ { T } ( \cdot \mid s _ { t } ) \right) \right] } \\ { = \mathcal { L } _ { \mathrm { O P D } } ( \pi ) . } \end{array}
$$

If $P _ { \pi } \not \ll P _ { T }$ , the shared initial distribution and transition kernels imply an action-support mismatch on a set of student-visited states with positive probability. Both the expected conditional and trajectory KL divergences are infinite. The trajectory identity therefore holds under these conventions. □

ProofofTheorem 3.2. Let $V ( \tau ) = R _ { x } ( \tau )$ be the shared binary verifier. Its success probabilities are $p _ { \pi }$ and $p _ { T }$ as defined in the main text. For Bernoulli distributions, $\mathrm { K L } ( p | | q ) = p \dot { \log } ( p / q ) + ( 1 -$ $p ) \log ( ( 1 - p ) / ( 1 - q ) )$ , using the same endpoint conventions.

Applying V maps $P _ { \pi }$ and $P _ { T }$ to $\mathrm { B e r n } ( p _ { \pi } )$ and $\mathrm { B e r n } ( p _ { T } )$ . The data-processing inequality (van Erven & Harremoës, 2014, Theorem 9) therefore gives

$$
\begin{array} { r l } & { \mathrm { K L } ( p _ { \pi } \| p _ { T } ) = D _ { \mathrm { K L } } \left( \mathrm { B e r n } ( p _ { \pi } ) \| \mathrm { B e r n } ( p _ { T } ) \right) } \\ & { \qquad \leq D _ { \mathrm { K L } } ( P _ { \pi } \| P _ { T } ) } \\ & { \qquad = \mathcal { L } _ { \mathrm { O P D } } ( \pi ) } \\ & { \qquad \leq \varepsilon , } \end{array}
$$

where the equality in the third line follows from Proposition 3.1.

For Bernoulli distributions, Pinsker’s inequality (van Erven & Harremoës, 2014, Theorem 31) yields

$$
2 | p _ { \pi } - p _ { T } | ^ { 2 } \leq \mathrm { K L } ( p _ { \pi } \| p _ { T } ) \leq \varepsilon .
$$

Taking square roots gives the gap bound in Eq. 2. Rearranging with $p _ { \pi } \geq 0$ gives the lower bound.

Offline state coverage. Evaluating the same local reverse KL under offline distributions $\mu _ { t }$ requires $d _ { t } ^ { \pi } \ll \mu _ { t }$ and a finite density-ratio bound $\begin{array} { r } { \frac { d d _ { t } ^ { \pi } } { d \mu _ { t } } \leq C } \end{array}$ to recover the bound. Nonnegativity of local KL then gives $\mathcal { L } _ { \mathrm { O P D } } ( \pi ) \leq C \mathcal { L } _ { \mu } ( \pi )$ . No finite C exists if absolute continuity fails.

Normalization and scope. For a fixed horizon, $\overline { { \mathcal { L } } } _ { \mathrm { O P D } } = \mathcal { L } _ { \mathrm { O P D } } / H$ gives $D _ { \mathrm { K L } } ( P _ { \pi } \Vert P _ { T } ) = H \overline { { \mathcal { L } } } _ { \mathrm { O P D } }$ Thus, $\overline { { \mathcal { L } } } _ { \mathrm { O P D } } \ \leq \ \delta$ implies $p _ { \pi } \geq [ p _ { T } - \sqrt { H \delta / 2 } ] .$ <sub>+</sub>. Normalization by realized trajectory length need not preserve this identity. The guarantee uses the population objective for the specified task, environment, verifier, and sampling policies. Certifying this premise from empirical estimates or under new decoding rules or evaluation distributions requires further error bounds.

## A.2 REWARD DISCOVERY AND REWARD-INFORMATIVE GROUPS

ProofofCorollary 3.3. For independent rollouts from the fixed policy $\pi _ { \mathrm { w } }$ , the first-success time $T _ { \mathrm { s u c c } }$ is geometric with parameter $p _ { \mathrm { w } }$ . When $p _ { \mathrm { w } } \geq \ell > 0$

$$
\operatorname* { P r } ( T _ { \mathrm { s u c c } } > N ) = ( 1 - p _ { \mathrm { w } } ) ^ { N } \leq ( 1 - \ell ) ^ { N } \leq e ^ { - N \ell } , \qquad \mathbb { E } [ T _ { \mathrm { s u c c } } ] = \frac { 1 } { p _ { \mathrm { w } } } \leq \frac { 1 } { \ell } .
$$

Taking complements gives the success-probability bound. For $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , choosing $N \geq$ $\lceil \log ( \bar { 1 } / \delta ) / \ell \rceil$ makes the failure probability at most δ. □

Binary reward groups. Under binary rewards, a group of $G \geq 2$ independent rollouts has all-success probability $p ^ { G }$ , all-failure probability $( 1 - p ) ^ { \dot { G } }$ , and mixed-group probability

$$
m _ { G } ( p ) = 1 - p ^ { G } - ( 1 - p ) ^ { G } , \qquad m _ { G } ^ { \prime } ( p ) = G \bigl [ ( 1 - p ) ^ { G - 1 } - p ^ { G - 1 } \bigr ] .\tag{6}
$$

This probability increases below $p = 0 . 5$ and decreases above it. The lower bound $p _ { \mathrm { w } } \geq \ell$ alone does not imply an increase in mixed groups, since it does not also ensure $p _ { \mathrm { w } } \leq 0 . 5$ . All-success groups indicate reliable completion even though their direct GRPO advantages are zero. Empirical group fractions average these per-task quantities over tasks with different success probabilities. In ScienceWorld, partial-credit rewards can differ even without complete success.

## B TRAINING AND EVALUATION PROTOCOL

## B.1 DATA, MODELS, AND AGENT INTERACTION

The students and teacher use released Qwen3 weights (Yang et al., 2025) without additional agent-task training. Table 8 lists model sizes and shared settings. Chat templates disable thinking, while task prompts request <reasoning> and <action> fields. ALFWorld supplies the current observation and admissible actions. ScienceWorld supplies the task, observation, action templates, and objects. Both include the two most recent observation–action pairs when available.

Table 8: Training and evaluation settings for the main warmup comparisons.
<table><tr><td colspan="2">Parameter Value</td></tr><tr><td colspan="2">Models and training schedule</td></tr><tr><td>Student</td><td>Qwen3-1.7B / Qwen3-4B</td></tr><tr><td>Teacher</td><td>Qwen3-32B</td></tr><tr><td>Environments</td><td>ALFWorld / ScienceWorld</td></tr><tr><td>Main training length</td><td>40 warmup + 60 GRPO steps</td></tr><tr><td>Shorter warmup comparison Teacher during GRPO</td><td>20 warmup + 80 GRPO steps</td></tr><tr><td colspan="2">Not used</td></tr><tr><td>SFT / SFT-RS</td><td>Objectives</td></tr><tr><td>OPD</td><td>Token cross-entropy on all / successful teacher trajectories Cross-entropy on teacher top-1 targets along student trajectories</td></tr><tr><td>GRPO</td><td>Group-relative log-probability objective</td></tr><tr><td>GRPO advantage normalization</td><td>Within-group mean and sample standard deviation,  $\varepsilon = 1 0 ^ { - 6 }$ </td></tr><tr><td>Probability-ratio clipping</td><td>Not used</td></tr><tr><td>Additional KL / entropy coefficient</td><td>0/0</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW (Loshchilov &amp; Hutter, 2019), β = (0.9, 0.999)</td></tr><tr><td>Learning rate / schedule</td><td> $1 0 ^ { - 6 } /$  constant</td></tr><tr><td>Weight decay / gradient clipping</td><td>0.01 / norm 1.0</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Loss within a microbatch</td><td>Mean over response tokens</td></tr><tr><td colspan="2">Table 8 (continued)</td></tr><tr><td colspan="2">Parameter Value</td></tr><tr><td colspan="2"></td></tr><tr><td rowspan="4">OPD / GRPO batch OPD / GRPO minibatch</td><td>Batching and step counts</td></tr><tr><td>16 tasks × 8 rollouts</td></tr><tr><td>8 tasks ×8 rollouts</td></tr><tr><td>1</td></tr><tr><td>SFT / SFT-RS global batch Optimizer steps per training step</td><td>128 teacher trajectories SFT: 1, OPD / GRPO: 2</td></tr><tr><td></td><td>Agent interaction</td></tr><tr><td rowspan="3">Prompt / trajectory response limit Response limit per turn Maximum interaction length</td><td>4,096 / 24,576 tokens</td></tr><tr><td>512 tokens</td></tr><tr><td>30 turns</td></tr><tr><td>Recent interactions in the prompt</td><td>2 observation-action pairs</td></tr><tr><td>Chat-template thinking mode</td><td>Disabled</td></tr><tr><td colspan="2">Sampling and evaluation</td></tr><tr><td colspan="2">Training temperature / top-p 1.0/1.0 0.7 / 0.95</td></tr><tr><td colspan="2">Evaluation temperature / top-p</td></tr><tr><td colspan="2">Learning-curve rollouts per task 8</td></tr><tr><td colspan="2">Rollouts per task for warmup success 64</td></tr><tr><td colspan="2">profile</td></tr><tr><td colspan="2">Compute</td></tr><tr><td colspan="2"></td></tr><tr><td>GPUs per training run 8</td><td></td></tr><tr><td>Student / teacher placement</td><td>Shared eight-GPU pool for OPD</td></tr><tr><td colspan="2">Sequence parallel size 4</td></tr></table>

ALFWorld reserves 64 of its 3,553 training tasks for analysis, leaving disjoint warmup and GRPO pools of 1,744 and 1,745 tasks. ScienceWorld reserves 64 of 2,294 tasks and uses two training pools of 1,115. Both partitions use seed 20260722 and task-instance identities, allowing task types to overlap. ALFWorld Seen and Unseen use separate benchmark instances. ScienceWorld uses it reserved 64 tasks as Seen and the original 200 test tasks as Unseen. Complete-action evaluation uses 32 tasks per benchmark test split, excluding ALFWorld’s 64 reserved tasks.

## B.2 TRAINING OBJECTIVES AND OPTIMIZATION

All warmup methods start from the same Base model. GRPO after warmup uses the second task pool and no teacher supervision. The GRPO warmup baseline also switches pools and loads its warmup weights into a new optimizer, as shown in Figs. 1 and 2. Direct GRPO instead uses the second pool for all 100 steps without restarting, while direct OPD uses the warmup pool throughout. Extended ALFWorld comparisons run 100 or 280 GRPO steps after 40 warmup steps.

In ALFWorld 4B, SFT processes 5,120 teacher trajectories once over 40 steps, using 9,662,640 supervised response tokens. SFT-RS retains 2,375 successful trajectories and repeats them in a fixed order, processing 7.07 million tokens over 40 steps. OPW processes 9.76 million student-generated response tokens. All loss counts exclude prompts, environment observations, and padding.

In OPD and GRPO, each microbatch’s response-token mean is weighted by its share of the minibatch’s trajectories before gradient accumulation. In ALFWorld 4B SFT, each data-parallel rank accumulates 64 one-trajectory microbatches. Their token-normalized gradients are summed before clipping, with division by the microbatch count applied only to the logged loss. Axes count training steps, each covering one batch and its optimization. A 40-step SFT warmup followed by 60 GRPO steps contains 160 optimizer steps, compared with 200 for the OPW and GRPO-warmup paths.

## B.3 EVALUATION AND STATISTICAL SUMMARIES

Table 9 lists the trajectory collections used for each analysis. Learning curves average eight rollouts per task and weight tasks equally. All eight rollouts come from the same trained checkpoint. Curves connect recorded evaluations without smoothing. A checkpoint can have several independently sampled evaluations, so each comparison uses its own initial evaluation.

Table 9: Trajectory collections used for evaluation.
<table><tr><td>Collection</td><td>Rollouts/task Measurements</td><td></td></tr><tr><td>Original evaluations</td><td>8</td><td>Learning curves, final task performance and pass^8</td></tr><tr><td>New warmup evaluations</td><td>64</td><td>Success coverage and repeated task success before GRPO</td></tr><tr><td>First eight of the new collection</td><td>8</td><td>Warmup action composition and standard OPW&#x27;s initial score in supervision removal</td></tr><tr><td>ALFWorld warmup diagnostics</td><td>32</td><td>Validity and repeated task success on 64 reserved tasks</td></tr><tr><td>ScienceWorld warmup diagnostics</td><td>8</td><td>Complete success and partial-credit reward contrast on 64 Seen tasks</td></tr><tr><td>Duration comparisons</td><td>8</td><td>Final performance under different OPW + GRPO allocations</td></tr></table>

Early gain and average task performance. Let $R _ { t }$ denote the task mean after t GRPO steps following warmup, measured as task success in ALFWorld or task score in ScienceWorld on a 0–100 scale. Early gain is $\Delta R _ { 2 0 } = R _ { 2 0 } - R _ { 0 }$ . For recorded evaluations at $0 = t _ { 0 } < \cdots < t _ { m } = T$ , average task performance during GRPO is

$$
\bar { R } _ { T } = \frac { 1 } { T } \sum _ { j = 1 } ^ { m } ( t _ { j } - t _ { j - 1 } ) \frac { R _ { t _ { j } } + R _ { t _ { j - 1 } } } { 2 } .\tag{7}
$$

This averages the recorded curve by the trapezoidal rule, with GRPO steps counted from the end of warmup. Full-training and duration comparisons count both stages. Target-reaching times use the first scheduled evaluation meeting the stated success target, without interpolation.

Task-level success probabilities. For independent rollouts on a task x with success probability $p _ { x }$ pass $\circledast k ( x ) = 1 - ( \bar { 1 } - p _ { x } ) ^ { k }$ measures at least one success, while pas $\begin{array} { r } { { \mathfrak { s } } ^ { \wedge } k ( x ) = p _ { x } ^ { k } } \end{array}$ measures success in all k rollouts. Given n rollouts with $c _ { x }$ successes, we use (Chen et al., 2021; Yao et al., 2024)

$$
\begin{array} { r } { \widehat { \mathrm { p a s s } @ k } ( x ) = 1 - \frac { \binom { n - c _ { x } } { k } } { \binom { n } { k } } , \qquad \widehat { \mathrm { p a s s } ^ { \wedge } \mathrm { k } } ( x ) = \frac { \binom { c _ { x } } { k } } { \binom { n } { k } } , } \end{array}\tag{8}
$$

then average the estimates over tasks. A binomial coefficient is zero when its upper argument is below k. Table 5 uses $n = 6 4$ and $k \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ $\operatorname { A t } k = n$ , these are the fractions of tasks solved at least once and in every rollout. ScienceWorld complete success requires a terminal trajectory with normalized task reward one, excluding partial progress.

## C ADDITIONAL WARMUP COMPARISONS

## C.1 FULL WARMUP RESULTS

OPW has the highest overall repeated task success in both environments after 40 warmup and 60 GRPO steps. Table 10 reports pass^8 by task type for the methods in Table 3.

Across environments, model sizes, and splits, OPW has the highest average task performance over 60 GRPO steps (Table 11). GRPO warmup yields a larger early gain on ScienceWorld 1.7B Unseen.

## C.2 SENSITIVITY TO TRAINING LENGTH AND SEED

Shorter warmup. With 20 warmup and 80 GRPO steps, OPW retains the highest final ALFWorld 4B success and pass^8 on both splits (Table 12). Fig. 5 uses the main comparison’s task sets and Base evaluations, counting total steps. Direct OPD is evaluated every 20 steps through step 100.

Longer GRPO training. After 40 warmup and 100 GRPO steps, SFT-RS approaches OPW’s final Seen success (95.2% versus 96.2%), but its average GRPO success remains lower (71.0% versus 89.4%). Unseen averages are 68.3% and 90.9%, respectively (Table 13). Fig. 6 extends GRPO to 280 steps and shows sustained high success after OPW.

Table 10: Final repeated task success (pass^8, %) after 40 warmup and 60 GRPO steps (Qwen3-4B).
<table><tr><td rowspan="2">Split</td><td></td><td>Task type</td><td colspan="6">ALFWorld</td><td colspan="8">ScienceWorld</td></tr><tr><td>Warmup</td><td></td><td>Pick</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Exam Pick2</td><td></td><td>All</td><td>Find</td><td>Life</td><td>Gene. Mix</td><td></td><td>Prop.</td><td>Phys.</td><td>All</td></tr><tr><td rowspan="4">Seen</td><td colspan="2">Off-policy: SFT</td><td>82.9</td><td>51.9</td><td>56.2</td><td>36.0</td><td>38.5</td><td>50.0</td><td>55.7</td><td>0.0</td><td>25.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>9.5</td><td>4.7</td></tr><tr><td colspan="2">Off-policy: SFT-RS</td><td>91.4</td><td>70.4</td><td>50.0</td><td>44.0</td><td>69.2</td><td>54.2</td><td>65.7</td><td>12.5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>7.4</td><td>4.8</td><td>6.2</td></tr><tr><td colspan="2">On-policy: GRPO</td><td>94.3</td><td>37.0</td><td>43.8</td><td>36.0</td><td>23.1</td><td>58.3</td><td>54.3</td><td>12.5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>19.0</td><td>7.8</td></tr><tr><td colspan="2">OPW (Ours)</td><td>97.1</td><td>92.6</td><td>81.2</td><td>80.0</td><td>84.6</td><td>83.3</td><td>87.9</td><td>62.5</td><td>25.0</td><td>50.0</td><td>0.0</td><td>63.0</td><td>19.0</td><td>43.8</td></tr><tr><td rowspan="4">Unseen</td><td colspan="2">Off-policy: SFT</td><td>75.0</td><td>45.2</td><td>52.2</td><td>52.4</td><td>27.8</td><td>52.9</td><td>51.5</td><td>3.8</td><td>2.9</td><td>12.0</td><td>0.0</td><td>1.5</td><td>26.3</td><td>6.0</td></tr><tr><td colspan="2">Off-policy: SFT-RS</td><td>87.5</td><td>64.5</td><td>65.2</td><td>76.2</td><td>55.6</td><td>58.8</td><td>68.7</td><td>1.9</td><td>8.6</td><td>4.0</td><td>0.0</td><td>3.1</td><td>10.5</td><td>4.5</td></tr><tr><td colspan="2">On-policy: GRPO</td><td>70.8</td><td>29.0</td><td>34.8</td><td>28.6</td><td>22.2</td><td>17.6</td><td>35.1</td><td>7.7</td><td>0.0</td><td>12.0</td><td>0.0</td><td>1.5</td><td>15.8</td><td>5.5</td></tr><tr><td colspan="2">OPW (Ours)</td><td>87.5</td><td>100.0</td><td>82.6</td><td>95.2</td><td>77.8</td><td>82.4</td><td>88.8</td><td>3.8</td><td>14.3</td><td>16.0</td><td>0.0</td><td>4.6</td><td>15.8</td><td>8.5</td></tr></table>

Table 11: Early gains and average task performance during 60 GRPO steps after warmup.
<table><tr><td rowspan="2">Benchmark</td><td colspan="4">ALFWorld</td><td colspan="4">ScienceWorld</td></tr><tr><td colspan="2">Seen</td><td colspan="2">Unseen</td><td colspan="2">Seen</td><td colspan="2">Unseen</td></tr><tr><td>Warmup</td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td></tr><tr><td colspan="9"></td></tr><tr><td>Off-policy: SFT</td><td>+9.7</td><td>23.6</td><td>Qwen3-1.7B +7.5</td><td>25.8</td><td>+0.5</td><td>13.6</td><td>+2.2</td><td>18.7</td></tr><tr><td>Off-policy: SFT-RS</td><td>+6.8</td><td>20.7</td><td>+5.0</td><td>21.1</td><td>+3.2</td><td>12.9</td><td>+4.6</td><td>19.4</td></tr><tr><td>On-policy: GRPO</td><td>+7.8</td><td>37.2</td><td>+6.8</td><td>37.8</td><td>+13.7</td><td>28.1</td><td>+14.0</td><td>34.1</td></tr><tr><td>OPW (Ours)</td><td>+26.9</td><td>73.7</td><td>+19.6</td><td>71.9</td><td>+14.2</td><td>62.0</td><td>+5.6</td><td>58.1</td></tr><tr><td colspan="9"></td></tr><tr><td>Off-policy: SFT</td><td>+14.1</td><td>56.5</td><td>Qwen3-4B +15.0</td><td>53.8</td><td>+15.9</td><td>43.5</td><td>+17.1</td><td>51.4</td></tr><tr><td>Off-policy: SFT-RS</td><td>+8.7</td><td>56.3</td><td>+7.3</td><td>54.5</td><td>+22.8</td><td>47.4</td><td>+19.5</td><td>53.1</td></tr><tr><td>On-policy: GRPO</td><td>+11.4</td><td>69.5</td><td>+11.9</td><td>65.0</td><td>+0.9</td><td>57.0</td><td>+0.2</td><td>59.4</td></tr><tr><td>OPW (Ours)</td><td>+41.8</td><td>84.7</td><td>+31.4</td><td>86.6</td><td>+22.6</td><td>78.9</td><td>+3.5</td><td>65.2</td></tr></table>

In Fig. 6, test panels (a,b) use unsmoothed evaluations every 20 steps, with eight rollouts on 140 Seen and 134 Unseen tasks. Training panels (c)–(f) use 16 tasks with eight rollouts per step. Validity pools recorded actions, reward-group fractions average task groups, and interaction length counts action turns. Training curves use exponential moving averages with coefficient 0.2, while interval means weight raw steps equally. Direct OPD remains on the warmup pool over the same total steps.

OPW’s all-success group fraction rises from 44.7% to 72.8% between the first two 20-step GRPO intervals. These groups indicate reliable completion but have zero direct GRPO advantages. Over the final 100 steps, OPW averages 97.3% training success with 1.6% all-failure groups, compared with 86.7% and 4.0% after GRPO warmup.

Second training seed. Both ALFWorld 4B seeds use 40 warmup and 100 GRPO steps, with eight rollouts per task on both test splits. Table 14 reports OPW minus GRPO warmup for initial success $R _ { 0 } .$ , early gain $\Delta R _ { 2 0 }$ , and average success $\bar { R } _ { 1 0 0 }$ . Under both seeds, OPW starts lower on Seen and higher on Unseen, then achieves larger early gains and higher averages on both splits.

Table 12: Final task performance and repeated task success for 20-step warmup (ALFWorld 4B).
<table><tr><td rowspan="2">Warmup</td><td colspan="2">Seen</td><td colspan="2">Unseen</td></tr><tr><td>pass@1</td><td>pass^8</td><td>pass@1</td><td>pass^8</td></tr><tr><td>SFT</td><td>91.2</td><td>72.1</td><td>83.3</td><td>63.4</td></tr><tr><td>SFT-RS</td><td>89.1</td><td>68.6</td><td>83.5</td><td>59.0</td></tr><tr><td>GRPO</td><td>72.3</td><td>35.0</td><td>69.9</td><td>26.1</td></tr><tr><td>OPW (Ours)</td><td>95.7</td><td>85.0</td><td>93.3</td><td>76.9</td></tr></table>

![](images/e8319e24b3ab30fdb8a5feb8129d64b53a7b31cf91927eaf053a2ba0c345cf01.jpg)  
Figure 5: Task performance during 20 warmup and 80 GRPO steps.

![](images/7c5be33e61c3f52b543d20258ccc7ccc02e7d6f3b1ebaedc6eca5613050ccd9d.jpg)  
Figure 6: Task performance (a,b) and training dynamics (c–f) during extended training.

## C.3 TRAINING COMPUTATION AND TARGET-REACHING COST

We multiply recorded training times by eight GPUs for each path with 40 warmup and 60 GRPO steps in Fig. 2. Timers include trajectory generation, environment interaction, teacher scoring, and student optimization. SFT and SFT-RS include 10.9 GPU-hours for collecting 5,120 shared teacher trajectories before filtering. Separate evaluations, research diagnostics, checkpoint saving, and startup are excluded. Student training and teacher scoring share the eight-GPU allocation, with fully sharded data parallelism, sequence-parallel size four, and data-parallel size two. Supplementary runs use L20X GPUs. Unknown GPU models in earlier runs prevent hardware normalization.

Table 13: Average and final task performance over 100 GRPO steps after warmup (ALFWorld 4B).
<table><tr><td rowspan="2">Task set Warmup</td><td colspan="2">Seen</td><td colspan="2">Unseen</td></tr><tr><td> $\bar { R } _ { 1 0 0 }$ </td><td> $R _ { 1 0 0 }$ </td><td> $\bar { R } _ { 1 0 0 }$ </td><td> $R _ { 1 0 0 }$ </td></tr><tr><td>Off-policy: SFT</td><td>69.3</td><td>90.4</td><td>65.6</td><td>88.6</td></tr><tr><td>Off-policy: SFT-RS</td><td>71.0</td><td>95.2</td><td>68.3</td><td>92.6</td></tr><tr><td>On-policy: GRPO</td><td>76.3</td><td>92.7</td><td>70.8</td><td>84.3</td></tr><tr><td>OPW (Ours)</td><td>89.4</td><td>96.2</td><td>90.9</td><td>96.5</td></tr></table>

Table 14: OPW minus GRPO warmup under two training seeds (percentage points).
<table><tr><td rowspan="2">Seed</td><td colspan="3">Seen</td><td colspan="3">Unseen</td></tr><tr><td> $R _ { 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 1 0 0 }$ </td><td> $R _ { 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 1 0 0 }$ </td></tr><tr><td>1</td><td>-9.2</td><td>+30.4</td><td>+13.0</td><td>+7.3</td><td>+19.5</td><td>+20.1</td></tr><tr><td>2</td><td>-0.8</td><td>+17.2</td><td>+6.6</td><td>+6.4</td><td>+9.0</td><td>+9.5</td></tr></table>

Table 15: Training cost and cost to reach target Seen success in ALFWorld (GPU-hours).
<table><tr><td rowspan="2">Cost Warmup</td><td colspan="4">100 training steps</td><td colspan="2">To reach</td></tr><tr><td>Data collection</td><td>Warmup</td><td>GRPO</td><td>Total</td><td>&gt; 80%</td><td>&gt; 90%</td></tr><tr><td colspan="9">Qwen3-1.7B</td></tr><tr><td>Off-policy: SFT</td><td>10.9</td><td>2.5</td><td>14.1</td><td>27.5</td><td>一</td><td>一</td></tr><tr><td>Off-policy: SFT-RS</td><td>10.9</td><td>1.9</td><td>13.9</td><td>26.7</td><td>一</td><td></td></tr><tr><td>On-policy: GRPO</td><td>0.0</td><td>9.3</td><td>12.9</td><td>22.2</td><td></td><td></td></tr><tr><td>OPW (Ours)</td><td>0.0</td><td>12.8</td><td>11.1</td><td>23.9</td><td>20.3</td><td></td></tr><tr><td colspan="9">Qwen3-4B</td></tr><tr><td>Off-policy: SFT</td><td>10.9</td><td>4.8</td><td>15.7</td><td>31.4</td><td>31.4</td><td>一</td></tr><tr><td>Off-policy: SFT-RS</td><td>10.9</td><td>3.2</td><td>16.8</td><td>30.9</td><td>30.9</td><td></td></tr><tr><td>On-policy: GRPO</td><td>0.0</td><td>13.0</td><td>16.6</td><td>29.6</td><td>29.6</td><td></td></tr><tr><td>OPW (Ours)</td><td>0.0</td><td>14.9</td><td>12.4</td><td>27.3</td><td>19.6</td><td>23.6</td></tr></table>

–: target not reached within 100 steps.

OPW reduces total cost relative to GRPO warmup at 4B (27.3 versus 29.6 GPU-hours) but increases it at 1.7B (23.9 versus 22.2). Table 15 uses the first evaluation at total step 40, 60, 80, or 100 exceeding the target, without interpolation. At 4B, OPW reaches both targets at the same evaluations on Seen and Unseen. GRPO warmup exceeds 80% only on Seen. At 1.7B, only OPW exceeds 80%, at step 80 on Seen and step 100 on Unseen. Only 4B OPW exceeds 90% within 100 steps.

## D TEACHER SUPERVISION ON STUDENT-GENERATED TRAJECTORIES

## D.1 TEACHER-TARGET PROBABILITIES AFTER ONE OPTIMIZER STEP

From Base, we take one AdamW step with learning rate $1 0 ^ { - 6 }$ , using 128 student trajectories for OPD/GRPO or 128 teacher trajectories for SFT. These batches contain 329,646 and 257,068 response tokens, respectively. OPD applies $\ell _ { t } = - \log \pi _ { \theta } ( v _ { t } \mid h _ { t } )$ at student token histories $h _ { t } ,$ with teacher top-1 targets $v _ { t } .$ SFT uses this loss with teacher tokens on teacher trajectories. We score log-probability changes at common histories where teacher targets disagree with student tokens.

The batch has 16 tasks and 22,279 disagreements. Eight groups contain only failures, six have mixed rewards, and two contain only successes. Table 16 averages changes over all disagreements, 11,839 positions in 64 all-failure trajectories, and 4,455 positions with the lowest Base probability for the teacher target. OPD and GRPO use all 16 task groups in one minibatch, compared with eight per minibatch in main training. In all-failure groups, SFT changes probabilities through transfer from teacher trajectories, while OPD supervises the student histories directly. GRPO advantages are zero in these groups, but other groups can change their probabilities through shared parameters. Across the batch, disagreements occupy 6.8% of response positions but account for 97.7% of teacher-target loss and 95.2% of the positive log-probability change after OPD.

## D.2 RESAMPLING STUDENT TRAJECTORIES

Both ALFWorld Qwen3-4B conditions use teacher top-1 cross-entropy for 40 warmup steps, followed by 60 teacher-free GRPO steps. The fixed version reuses initial-student trajectories while recomputing teacher targets and current student probabilities. OPW resamples trajectories as the student changes.

Table 16: Teacher-target log-probability changes after one optimizer step (ALFWorld 4B).
<table><tr><td>Objective</td><td>All disagreements</td><td>All-failure groups</td><td>Lowest-probability 20%</td></tr><tr><td>SFT</td><td>+0.1279</td><td>+0.128</td><td>+0.290</td></tr><tr><td>GRPO</td><td>-0.0004</td><td>+0.004</td><td>+0.017</td></tr><tr><td>OPD</td><td>+0.1812</td><td>+0.183</td><td>+0.392</td></tr></table>

Evaluations use eight rollouts per task on 140 Seen and 134 Unseen tasks. Table 17 reports the evaluations underlying Table 7. Resampling improves early gains and average success, while fixed trajectories yield higher final Seen success. Resampling jointly changes histories, diversity, and task coverage.

Table 17: Task performance after warmup on fixed or resampled trajectories (ALFWorld 4B).
<table><tr><td rowspan="2">Task set</td><td rowspan="2">Trajectories</td><td colspan="4">GRPO steps after warmup</td></tr><tr><td>0</td><td>20</td><td>40</td><td>60</td></tr><tr><td rowspan="2">Seen</td><td>Fixed</td><td>41.1</td><td>79.6</td><td>94.4</td><td>97.4</td></tr><tr><td>OPW (Ours)</td><td>46.8</td><td>88.6</td><td>94.1</td><td>96.1</td></tr><tr><td rowspan="2">Unseen</td><td>Fixed</td><td>50.5</td><td>79.7</td><td>91.7</td><td>95.1</td></tr><tr><td>OPW (Ours)</td><td>56.7</td><td>88.2</td><td>95.0</td><td>96.7</td></tr></table>

## D.3 REMOVING SUPERVISION AT REPEATED INVALID ACTIONS

Two ALFWorld Qwen3-4B runs share OPW’s initialization, tasks, teacher, batch size, seed, and optimizer for 40 warmup and 60 GRPO steps. Targeted removal zeros teacher-loss weights on action-content tokens when the action is invalid and has appeared earlier in the trajectory. Inputs, targets, and loss normalization stay unchanged. Random removal selects complete actions with the same token-length counts as repeated invalid actions in its current batch, allowing overlap. Branches generate their own trajectories, so total removal counts can differ. Targeted removal excludes 34,536 tokens in 6,489 actions (0.354% of response tokens), versus 32,132 tokens in 5,992 actions (0.330%) under random removal. Random removal includes 7,199 tokens in 1,474 repeated invalid actions.

Targeted and random removal yield similar warmup validity on Seen (93.8% versus 93.4%) and Unseen (93.5% versus 93.7%). Yet targeted removal produces smaller early GRPO gains on both splits (Table 18), supporting the contribution of supervision at repeated errors. All conditions use eight rollouts per task on 140 Seen and 134 Unseen tasks at 0, 20, 40, and 60 GRPO steps. OPW’s initial evaluation uses the first eight of the new 64-rollout collection, explaining differences from the main summary. Later OPW evaluations use the original collections. The table reports early gain $\Delta R _ { 2 0 }$ in percentage points, average success $\bar { R } _ { 6 0 }$ from Eq. 7, and final success $R _ { 6 0 }$ in percent.

Table 18: GRPO learning after targeted and random supervision removal (ALFWorld 4B).
<table><tr><td>Task set</td><td colspan="3">Seen</td><td colspan="4">Unseen</td></tr><tr><td>Warmup</td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td><td> $R _ { 6 0 }$ </td><td> $\Delta R _ { 2 0 }$ </td><td> $\bar { R } _ { 6 0 }$ </td><td> $R _ { 6 0 }$ </td></tr><tr><td>OPW (Ours)</td><td>41.5</td><td>84.7</td><td>96.1</td><td>32.2</td><td>86.5</td><td>96.7</td></tr><tr><td>Random removal</td><td>42.0</td><td>84.8</td><td>97.1</td><td>35.2</td><td>82.9</td><td>94.3</td></tr><tr><td>Targeted removal</td><td>34.6</td><td>80.1</td><td>95.4</td><td>29.2</td><td>80.9</td><td>92.9</td></tr></table>

## E FROM TEACHER TARGETS TO ACTION VALUE

## E.1 TEACHER ALTERNATIVES, TASK VALUE, AND RETENTION

In ALFWorld Qwen3-4B, we test 311 action-token alternatives from 97 tasks. We inspect eight Base rollouts per task from the first 128 distinct training tasks in trajectory order. Eligible positions have Base probability of at least 0.9 for the original token and a different teacher top-1 token, with both tokens containing letters or digits. We retain 21 positions from an eight-task analysis and add up to four distinct token histories per task, without using continuation outcomes or trained-policy probabilities for selection. At each position, we choose either token and let Base complete the action and remaining trajectory. Four continuations per choice yield 2,488 rollouts. We average continuations within positions, positions within tasks, and tasks equally. Teacher alternatives raise action validity from 67.44% to 82.22% and continuation success from 26.48% to 29.70%. On the 89 added tasks, continuation success still improves by 2.39 percentage points.

We track teacher-target probabilities at these action positions and on a separate set of 39,490 disagreements from 256 student trajectories across 32 tasks (Table 19). All checkpoints use the same token histories and teacher top-1 targets. Probabilities use the full softmax before temperature scaling or top-p truncation, averaged within tasks and then equally across tasks. Over 40 teacher-free GRPO steps, OPW’s mean probability rises from 39.3% to 47.2% at the selected action positions. On the broader set, it remains near its warmup level (43.8% to 43.2%) and highest on all 32 tasks. Selected action positions thus show further gains, while the broader set shows retention. The table reports warmup probabilities $P _ { 0 }$ in percent and changes $\Delta P _ { t } = P _ { t } - P _ { 0 }$ in percentage points.

Table 19: Teacher-target probabilities during teacher-free GRPO (ALFWorld 4B).
<table><tr><td rowspan="2">Positions Warmup</td><td colspan="3">All disagreements</td><td colspan="3">Action positions</td></tr><tr><td> $P _ { 0 }$ </td><td> $\Delta P _ { 2 0 }$ </td><td> $\Delta P _ { 4 0 }$ </td><td> $P _ { 0 }$ </td><td> $\Delta P _ { 2 0 }$ </td><td> $\Delta P _ { 4 0 }$ </td></tr><tr><td>Off-policy: SFT</td><td>17.7</td><td>+0.3</td><td>+1.2</td><td>3.8</td><td>+2.8</td><td>+6.2</td></tr><tr><td>Off-policy: SFT-RS</td><td>17.9</td><td>-1.4</td><td>+1.0</td><td>3.7</td><td>+4.7</td><td>+14.8</td></tr><tr><td>On-policy: GRPO</td><td>17.7</td><td>+1.6</td><td>-0.9</td><td>12.5</td><td>+6.2</td><td>+8.3</td></tr><tr><td>OPW (Ours)</td><td>43.8</td><td>-0.2</td><td>-0.6</td><td>39.3</td><td>+6.1</td><td>+7.9</td></tr></table>

## E.2 COMPLETE-ACTION VALUE AT COMMON HISTORIES

We take one trajectory per warmup method from each of 32 Seen and 32 Unseen tasks and inspect positions after four and twelve actions. Checkpoints share 512 records, including 447 active positions from 63 tasks and 65 records from already-ended trajectories. Each policy samples four actions per active position. Base supplies four continuations per distinct position–action pair, reusing continuations for identical actions at their sampled frequencies. We average positions and sources within tasks, then weight tasks equally. Ended records retain their outcomes only in all-position success.

We replay actions in a reset game, checking observations and admissible actions. Base receives only the sampled <action> field. Actions and continuations use temperature 0.7, top-p = 0.95, a 1,024-token response limit, a 32,768-token context limit, and at most 30 interaction turns. Table 20 reports validity $V _ { t }$ and continuation success $C _ { t } ~ ( \% )$ after t GRPO steps. Gains are in percentage points, with $\Delta C _ { 4 0 } = C _ { 4 0 } - C _ { 0 }$

At active positions, OPW starts with the highest validity (94.45%) but lower continuation success than GRPO warmup (45.00% versus 45.93%). Over 40 GRPO steps, its continuation success rises to 46.75%, compared with 46.47% after GRPO warmup. Including already-ended records gives the same ordering of gains, at 1.60 versus 0.49 points, with similar final means. Since Base supplies every continuation, these gains reflect improved action selection at the evaluated histories.

Table 20: Complete-action validity and continuation success under Base (ALFWorld 4B).
<table><tr><td rowspan="2">Warmup</td><td rowspan="2">Metric</td><td colspan="4">Active positions</td><td colspan="4">All positions: success</td></tr><tr><td colspan="2">Validity</td><td colspan="2">Success</td><td colspan="2">Seen</td><td colspan="2">Unseen</td></tr><tr><td></td><td></td><td> $V _ { 0 }$ </td><td> $V _ { 4 0 }$ </td><td> $C _ { 0 }$ </td><td> $C _ { 4 0 }$ </td><td> $C _ { 0 }$ </td><td> $\Delta C _ { 4 0 }$ </td><td> $C _ { 0 }$ </td><td> $\Delta C _ { 4 0 }$ </td></tr><tr><td>Off-policy: SFT</td><td></td><td>92.57</td><td>93.68</td><td>44.01</td><td>45.16</td><td>41.89</td><td>+0.61</td><td>51.56</td><td>+1.20</td></tr><tr><td>Off-policy: SFT-RS</td><td>92.64</td><td></td><td>93.74</td><td>44.40</td><td>45.56</td><td>42.33</td><td>+0.42</td><td>51.78</td><td>+1.34</td></tr><tr><td>On-policy: GRPO</td><td>93.92</td><td></td><td>93.78</td><td>45.93</td><td>46.47</td><td>43.60</td><td>+0.05</td><td>52.93</td><td>+0.93</td></tr><tr><td>OPW (Ours)</td><td>94.45</td><td></td><td>94.61</td><td>45.00</td><td>46.75</td><td>42.26</td><td>+1.68</td><td>52.61</td><td>+1.51</td></tr></table>

## F BEHAVIOR AFTER WARMUP AND DURING GRPO

## F.1 EXECUTION ERRORS AND REPETITION

The Qwen3-4B analysis uses the first eight trajectories per task on 140 Seen and 134 Unseen ALFWorld tasks and 64 Seen and 200 Unseen ScienceWorld tasks. Base and Teacher have one evaluation each. The four warmup methods are evaluated after warmup and after 20, 40, and 60 GRPO steps. Table 6 uses the warmup evaluations from the new 64-rollout collection. Tasks receive equal weight.

We lowercase action text and normalize whitespace. An action repeats if it appeared earlier, whether or not that occurrence was valid. Validity requires membership in the current valid-action list. ScienceWorld navigation aliases absent from that list count as invalid even if execution succeeds. Empty or unparseable actions are invalid format failures. Empty actions are excluded from repetition. Consecutive identical invalid actions require both adjacent actions to be invalid and identical, forming a subset of repeated invalid actions. Event shares divide by all actions in each task’s eight trajectories. Error incidence is the fraction of trajectories containing an invalid action.

On ALFWorld Seen, similar fractions of SFT and OPW trajectories contain invalid actions (50.9% and 50.0%), but consecutive identical invalid actions occupy 3.93% and 0.66% of all actions. On ScienceWorld Unseen, the consecutive-error share is 13.83% for SFT and 4.17% for OPW. Appendix H illustrates these patterns with complete action sequences.

Table 21 compares both model sizes using the original eight-rollout evaluations. Its initial means can differ slightly from the new warmup collection. $V _ { 0 } , L _ { 0 }$ , and $R _ { 0 }$ denote warmup validity, repetition, and task performance, with $\Delta X = X _ { 6 0 } - X _ { 0 }$ . Repetition includes valid actions, and ScienceWorld task scores include partial progress. For ALFWorld 4B, OPW’s validity rises by only 0.9 points during GRPO while success rises by 49.3, showing continued progress after most actions are valid.

Table 21: Execution behavior and task performance before and after 60 GRPO steps.
<table><tr><td rowspan="3">Warmup</td><td colspan="6">Qwen3-1.7B</td><td colspan="6">Qwen3-4B</td></tr><tr><td colspan="2">Action validity</td><td colspan="2">Repeated actions</td><td colspan="2">Task performance</td><td colspan="2">Action validity</td><td colspan="2">Repeated actions</td><td colspan="2">Task performance</td></tr><tr><td> $V _ { 0 }$ </td><td>∆V</td><td> $L _ { 0 }$ </td><td> $\Delta L$ </td><td> $R _ { 0 }$ </td><td>∆R</td><td> $V _ { 0 }$ </td><td>∆V</td><td> $L _ { 0 }$ </td><td>∆L</td><td> $R _ { 0 }$ </td><td>∆R</td></tr><tr><td colspan="10">ALFWorld, Seen</td><td></td><td></td><td></td><td></td></tr><tr><td>Off-policy: SFT</td><td>76.7</td><td>+6.5</td><td>60.1</td><td>-35.9</td><td>6.1</td><td>+39.2</td><td>87.4</td><td>+5.4</td><td>29.3</td><td>-9.6</td><td>29.6</td><td>+54.4</td></tr><tr><td>Off-policy: SFT-RS</td><td>77.1</td><td>+4.0</td><td>61.7</td><td>-29.5</td><td>6.2</td><td>+37.6</td><td>88.0</td><td>+7.5</td><td>29.7</td><td>-16.1</td><td>31.0</td><td> $+ 5 8 . 0$ </td></tr><tr><td>On-policy: GRPO</td><td>71.1</td><td>+12.7</td><td>38.5</td><td>-10.6</td><td>20.1</td><td>+45.4</td><td>90.1</td><td>-0.5</td><td>22.7</td><td>-5.1</td><td>56.0</td><td>+26.7</td></tr><tr><td>OPW (Ours)</td><td>92.0</td><td>+4.5</td><td>18.5</td><td>+1.6</td><td>46.5</td><td>+39.9</td><td>93.9</td><td>+0.9</td><td>16.4</td><td>-3.5</td><td>46.8</td><td>+49.3</td></tr><tr><td colspan="10">Science World, Seen</td><td colspan="3"></td></tr><tr><td>Off-policy: SFT</td><td>62.8</td><td>-39.6</td><td>87.8</td><td>-39.2</td><td>4.0</td><td>+29.4</td><td>63.5</td><td>-23.4</td><td>67.5</td><td>一</td><td>-31.021.8</td><td>+38.7</td></tr><tr><td>Off-policy: SFT-RS</td><td>61.5</td><td>-19.2</td><td>89.5</td><td>-28.4</td><td>3.3</td><td>+25.0</td><td>62.3</td><td>-13.1</td><td>67.8</td><td>-36.0</td><td>21.7</td><td>+42.9</td></tr><tr><td>On-policy: GRPO</td><td>35.0</td><td>-2.6</td><td>79.9</td><td>-29.0</td><td>9.0</td><td>+32.6</td><td>49.4</td><td>+2.4</td><td>41.7</td><td>-1.4</td><td>49.8</td><td>+19.0</td></tr><tr><td>OPW (Ours)</td><td>43.1</td><td>+3.7</td><td>53.8</td><td>-17.8</td><td>44.1</td><td>+27.6</td><td>51.7</td><td>+2.3</td><td>38.5</td><td>-11.3</td><td>57.3</td><td>+29.9</td></tr></table>

![](images/0d5e877b39e7a13d69b5d252678195ca61207bb5fbe4d15ca6339731ca806d28.jpg)  
Figure 7: All distinct action sequences among eight rollouts per task during GRPO.

Table 21 (continued)
<table><tr><td rowspan="3">Warmup</td><td colspan="6">Qwen3-1.7B</td><td colspan="6">Qwen3-4B</td></tr><tr><td colspan="2">Action validity</td><td colspan="2">Repeated actions</td><td colspan="2">Task performance</td><td colspan="2">Action validity</td><td colspan="2">Repeated actions</td><td colspan="2">Task performance</td></tr><tr><td>V0</td><td>ΔV</td><td>L0</td><td>∆L</td><td>R0</td><td>∆R</td><td>V0</td><td>ΔV</td><td>L0</td><td>∆L</td><td>R0</td><td>∆R</td></tr><tr><td colspan="9">ScienceWorld, Unseen</td><td></td><td></td><td></td><td></td></tr><tr><td>Off-policy: SFT</td><td>61.0</td><td>-31.4</td><td>88.3</td><td>-41.4</td><td>8.8</td><td>+31.6</td><td>57.1</td><td>-13.3</td><td>62.6</td><td>-27.0</td><td>32.2</td><td>+29.7</td></tr><tr><td>Off-policy: SFT-RS</td><td>59.8</td><td>-16.3</td><td>86.6</td><td>-27.6</td><td>8.5</td><td>+27.1</td><td>56.4</td><td>-8.3</td><td>59.1</td><td>-30.3</td><td>34.2</td><td>+29.3</td></tr><tr><td>On-policy: GRPO</td><td>36.8</td><td>-0.6</td><td>77.1</td><td>-24.0</td><td>15.0</td><td>+30.5</td><td>48.5</td><td>+4.2</td><td>38.5</td><td>-2.1</td><td>56.8</td><td>+2.8</td></tr><tr><td>OPW (Ours)</td><td>42.5</td><td>+3.4</td><td>44.8</td><td>-8.0</td><td>51.8</td><td>+11.2</td><td>50.3</td><td>+3.0</td><td>36.6</td><td>-3.6</td><td>61.6</td><td>+6.6</td></tr></table>

## F.2 SUCCESSFUL AND FAILURE-ONLY ACTION SEQUENCES

We evaluate both model sizes on 140 ALFWorld Seen and 200 ScienceWorld Unseen tasks, with eight rollouts per task after warmup and after 20, 40, and 60 GRPO steps. After lowercasing and whitespace normalization, we remove invalid actions and look, look around, and inventory, then merge consecutive identical actions. Object identities and order remain. We count each distinct sequence once per task, including the empty sequence. A sequence is successful if any occurrence succeeds and failure-only otherwise, even when filtering gives successful and failed trajectories the same sequence. Task averages include tasks with no successful trajectory.

Total sequence counts in Fig. 7 decrease after OPW, while successful counts in Fig. 3 increase in both environments and model sizes. In ALFWorld 4B, the total falls from 5.18 to 4.31, successful sequences rise from 2.23 to 4.00, and failure-only sequences fall from 2.95 to 0.31.

To compare equal numbers of successes, we retain tasks with at least $m \in \{ 2 , 4 \}$ successful trajectories under every method at all four evaluations. If a task has S successes and sequence j occurs c<sub>j</sub> times, its expected distinct count in m successes selected without replacement is $\textstyle { \overline { { \sum _ { j } [ 1 - } } }$ $\binom { S - c _ { j } } { m } / \binom { S } { m } ]$ . Table 22 averages over the N retained tasks, at warmup end and step 60.

For OPW 4B with $m = 4$ , the expected count falls from 2.54 to 2.14 in ALFWorld but rises from 1.83 to 2.32 in ScienceWorld. More successful sequences within eight rollouts can therefore accompany either less or more variation among equally many successes. All 1.7B cells except ALFWorld with m = 2 describe a single task.

## G WARMUP DURATION AND REPEATED TASK SUCCESS

## G.1 WHAT CONTINUES TO IMPROVE DURING WARMUP?

Table 23 evaluates Qwen3-1.7B OPW with data-order seed 4 every five steps on 64 reserved ALF-World analysis tasks, using 32 rollouts per task. These differ from the 140 Seen test tasks in the full-training comparison. Pass@8 and pass^8 use Eq. 8 with n = 32. From steps 20 to 40, validity

Table 22: Distinct action sequences among equal numbers of successful trajectories.
<table><tr><td>Environment</td><td></td><td></td><td>Size m N</td><td>Off-policy: SFT</td><td> $\scriptstyle \mathrm { O f f - p o l i c y : }$  SFT-RS</td><td>On-policy: GRPO</td><td>OPW (Ours)</td></tr><tr><td>ALFWorld</td><td>1.7B</td><td></td><td>2 10</td><td> $1 . 7 2  1 . 6 4$ </td><td> $1 . 8 1  1 . 5 4$ </td><td> $1 . 5 2  1 . 4 8$ </td><td> $1 . 5 1  1 . 3 4$ </td></tr><tr><td></td><td></td><td>4 1</td><td></td><td> $3 . 0 0  3 . 2 1$ </td><td> $2 . 0 0  2 . 6 0$ </td><td> $1 . 9 7  2 . 4 1$ </td><td> $1 . 0 0  1 . 9 3 $ </td></tr><tr><td></td><td>4B</td><td></td><td>2 49</td><td> $1 . 6 9  1 . 6 0$ </td><td> $1 . 7 0  1 . 5 0$ </td><td> $1 . 6 2  1 . 5 8$ </td><td> $1 . 6 7  1 . 5 2$ </td></tr><tr><td></td><td></td><td></td><td>4 39</td><td> $2 . 5 3  2 . 4 6$ </td><td> $2 . 6 5  2 . 0 8$ </td><td> $2 . 4 8  2 . 3 6$ </td><td> $2 . 5 4  2 . 1 4$ </td></tr><tr><td>ScienceWorld</td><td>1.7B</td><td>2</td><td>1</td><td> $1 . 6 7  1 . 0 0$ </td><td> $1 . 7 3  1 . 4 6$ </td><td> $1 . 6 8 \to 1 . 0 0$ </td><td> $1 . 2 9  1 . 0 0$ </td></tr><tr><td></td><td></td><td></td><td>4 1</td><td> $2 . 4 3  1 . 0 0$ </td><td> $2 . 6 0  2 . 0 0$ </td><td> $2 . 4 1  1 . 0 0$ </td><td> $1 . 5 7  1 . 0 0$ </td></tr><tr><td></td><td>4B</td><td></td><td>2 17</td><td> $1 . 4 7  1 . 5 7$ </td><td> $1 . 5 6  1 . 5 9$ </td><td> $1 . 5 0  1 . 6 0$ </td><td> $1 . 4 3  1 . 5 7$ </td></tr><tr><td></td><td></td><td></td><td>4 7</td><td> $1 . 9 8  2 . 0 2$ </td><td> $2 . 0 3  2 . 3 1$ </td><td> $1 . 5 2  2 . 0 7$ </td><td> $1 . 8 3  2 . 3 2$ </td></tr></table>

gains slow while repeated task success continues to improve. Tasks solved at least once in 32 rollouts increase from 52 to 53, while those solved in every rollout increase from 2 to 11.

Table 23: Action validity and repeated task success during OPW (ALFWorld 1.7B, $n = 3 2 , \% )$ .
<table><tr><td>OPW steps</td><td>Valid actions</td><td>pass@8</td><td> $\mathsf { p a s s } ^ { \wedge 8 }$ </td></tr><tr><td>10</td><td>72.9</td><td>39.4</td><td>0.4</td></tr><tr><td>15</td><td>85.1</td><td>52.4</td><td>1.3</td></tr><tr><td>20</td><td>89.4</td><td>69.7</td><td>10.8</td></tr><tr><td>25</td><td>90.4</td><td>71.4</td><td>13.4</td></tr><tr><td>30</td><td>91.1</td><td>71.2</td><td>14.5</td></tr><tr><td>35</td><td>91.6</td><td>73.7</td><td>20.7</td></tr><tr><td>40</td><td>91.8</td><td>74.8</td><td>22.0</td></tr></table>

For ScienceWorld Qwen3-1.7B with data-order seed 3, we evaluate 64 Seen tasks every five OPW steps from 10 to 40, with eight rollouts per task. These samples differ from initial GRPO evaluations. Fig. 8 compares partial-credit reward contrast with complete-task success. At both steps 15 and 40, 92.2% of groups contain different scores, while pass@8 rises from 18.8% to 56.2%. Success coverage thus continues to improve when partial-credit reward differences are already common.

![](images/67366cbb22ebf6f07981f45bef4725aa9095b6025e66d9eac108eeeb0a5c8397.jpg)  
Figure 8: Success coverage and partial-credit reward contrast during OPW (ScienceWorld 1.7B).

## G.2 LEARNING SPEED AND REPEATED TASK SUCCESS

Table 24 compares Qwen3-1.7B paths sharing a warmup run and ending at 100 total steps, evaluated on both benchmark splits with eight rollouts per task. ALFWorld branches after 20 or 40 OPW steps from the same seed-4 run. Longer warmup raises final Seen pass@1 from 88.3% to 94.9% and pass^8 from 69.3% to 80.7%. With evaluations every five steps, the first reaching 80% success moves from total step 75 to 60. Total steps are fixed, but GPU-hours differ.

ScienceWorld branches from one seed-1 checkpoint after 40 OPW steps. Switching to GRPO immediately yields higher Unseen pass@1 (36.6% versus 32.2%) and pass^8 (4.5% versus 3.5%) than 20 more OPW steps. Mean score favors longer warmup (63.84 versus 63.06). Longer warmup improves Seen mean score (71.72 to 76.65) and pass@8 (84.4% to 85.9%). The preferred allocation depends on the task set and whether evaluation rewards partial progress or complete success.

Table 24: Repeated task success after 100 total steps (Qwen3-1.7B, n = 8, %).
<table><tr><td>Environment / task set</td><td>OPW steps</td><td> $\mathsf { p a s s } ^ { \wedge 1 }$ </td><td> $\mathsf { p a s s } ^ { \wedge 2 }$ </td><td>pass^4</td><td>pass^8</td></tr><tr><td rowspan="2">ALFWorld / Seen</td><td>20</td><td>88.3</td><td>83.5</td><td>77.2</td><td>69.3</td></tr><tr><td>40</td><td>94.9</td><td>91.8</td><td>86.9</td><td>80.7</td></tr><tr><td rowspan="2">ScienceWorld / Unseen</td><td>40</td><td>36.6</td><td>20.1</td><td>8.7</td><td>4.5</td></tr><tr><td>60</td><td>32.2</td><td>17.8</td><td>8.5</td><td>3.5</td></tr><tr><td>ScienceWorld / Seen</td><td>40</td><td>51.8</td><td>37.4</td><td>24.4</td><td>10.9</td></tr><tr><td></td><td>60</td><td>56.1</td><td>42.7</td><td>30.7</td><td>20.3</td></tr></table>

Four-duration comparison across training runs. Fig. 4 compares 20, 40, 60, and 80 OPW steps followed by GRPO to 100 steps, evaluated every 20 steps. ALFWorld uses seed 4 for 20/40, seed 3 for 60, and seed 1 for 80 OPW steps. ScienceWorld uses seed 3 for 20/40 and seed 1 for 60/80. Rankings include training-run differences. ScienceWorld’s 40-step curve ends at 2.0% Unseen pass^8, while the seed-1 branch in Table 24 ends at 4.5%. The 60-step branch is shared.

Within the ScienceWorld seed-3 run, extending warmup from 20 to 40 steps raises Unseen pass@1 from 28.3% to 34.1% and pass^8 from 1.0% to 2.0%, while mean score changes little (60.94 to 60.64). Seen pass^8 also rises, from 3.1% to 10.9%. These comparisons support choosing duration by subsequent validation on the required outcome.

## H LONG-TRAJECTORY EXAMPLES

Two ALFWorld Unseen tasks illustrate long interactions and failure patterns. Comparisons use Qwen3-4B on the same task and the sixth evaluation trajectory. Base uses released weights, while SFT and OPW use their 40-step warmup models. Table 6 reports task-wide error frequencies.

The tables retain every action and quote selected environment responses, omitting reasoning and other observations. Orange marks invalid actions, red marks repeated invalid actions, and R marks any repetition. Other actions are valid. Highlights and turn labels are our annotations.

OPW completes the requested task while retaining useful repeated actions. Table 25 follows a task that requires cleaning a pan and placing it on a countertop. Base cleans and places a pot instead, then repeats finish without completing the task. OPW locates the pan and completes the task in 26 actions. Its returns to the sink and countertop serve cleaning and placement.

Table 25: Complete action sequences before and after OPW on the pan-cleaning task.
<table><tr><td>Task: “clean some pan and put it in countertop." Base</td><td>ALFWorld Unseen, Qwen3-4B.</td></tr><tr><td></td><td>OPW (Ours)</td></tr><tr><td>1 go to cabinet 1</td><td>1 go to cabinet 1</td></tr><tr><td>2 go to cabinet 2</td><td>2 go to cabinet 2</td></tr><tr><td>3 open cabinet 2</td><td>3 open cabinet 2</td></tr><tr><td>4 go to cabinet 3 5 go to cabinet 4</td><td>4 go to cabinet 3</td></tr><tr><td>6 go to cabinet 5</td><td>5 go to cabinet 4</td></tr><tr><td>7 open cabinet 5</td><td>6 go to cabinet 5 7 open cabinet 5</td></tr><tr><td>8 go to cabinet 6</td><td>8 go to cabinet 6</td></tr><tr><td>9 open cabinet 6</td><td>9 open cabinet 6</td></tr><tr><td>10 go to drawer 1</td><td>10 go to drawer 1</td></tr><tr><td>Task: “clean some pan and put it in countertop."</td><td>ALFWorld Unseen, Qwen3-4B.</td></tr><tr><td>Base</td><td>OPW (Ours)</td></tr><tr><td>11 open drawer 1</td><td>11 open drawer 1</td></tr><tr><td>12 go to drawer 2</td><td>12 go to drawer 2</td></tr><tr><td>13 open drawer 2</td><td>13 open drawer 2</td></tr><tr><td>14 go to drawer 3</td><td>14 go to drawer 3</td></tr><tr><td>15 open drawer 3</td><td>15 open drawer 3</td></tr><tr><td>16 go to shelf 1</td><td>16 go to countertop 1</td></tr><tr><td>17 go to shelf 2</td><td>17 go to countertop 2</td></tr><tr><td>18 go to shelf 3</td><td>18 go to countertop 3</td></tr><tr><td>19 go to sinkbasin 1 20 go to stoveburner 1</td><td>19 go to sinkbasin 1</td></tr><tr><td>21 take pot 1 from stoveburner 1</td><td>20 go to stoveburner 1</td></tr><tr><td>22 go to sinkbasin 1</td><td>21 go to stoveburner 2</td></tr><tr><td>23 clean pot 1 with sinkbasin 1</td><td>R 22 take pan 1 from stoveburner 2</td></tr><tr><td>24 go to countertop 1</td><td>23 go to sinkbasin 1 R</td></tr><tr><td>25 move pot 1 to countertop 1</td><td>24 clean pan 1 with sinkbasin 1</td></tr><tr><td>26 finish</td><td>25 go to countertop 1 R 26 move pan 1 to countertop 1</td></tr><tr><td>27 finish</td><td></td></tr><tr><td>28 finish</td><td></td></tr><tr><td>29 finish</td><td></td></tr><tr><td>ＲＲＲＲ 30 finish</td><td></td></tr><tr><td>Environment feedback</td><td>Environment feedback</td></tr><tr><td>Turn 21: “You pick up the pot 1 from the stoveburner 1."</td><td>Turn 22: “You pick up the pan 1 from the stoveburner 2."</td></tr><tr><td>Turns 26–30: "Nothing happens."</td><td>Turn 24: “You clean the pan 1 using the sinkbasin 1."</td></tr><tr><td>Outcome: Task not completed.</td><td>Turn 26: “You move the pan 1 to the countertop 1."</td></tr><tr><td>25/30 actions valid, 4 repeated invalid actions.</td><td>Outcome: Task completed. 26/26 actions valid, 0 repeated invalid actions.</td></tr></table>

OPW completes the task requirements after a longer search. In Table 26, SFT finds the egg earlier and places it in the microwave without cleaning it. It issues end 11 times with feedback “Nothing happens.” OPW searches longer but cleans and places the egg in 29 actions.

Table 26: Complete action sequences after SFT and OPW on the egg-cleaning task.
<table><tr><td>Task: “put a clean egg in microwave."</td><td colspan="3">ALFWorld Unseen, Qwen3-4B.</td></tr><tr><td colspan="2"></td><td>Off-policy: SFT</td><td>OPW (Ours)</td></tr><tr><td colspan="2"></td><td>1 go to fridge 1</td><td>1 go to fridge 1</td></tr><tr><td colspan="2"></td><td>2 open fridge 1</td><td>2 open fridge 1</td></tr><tr><td colspan="2"></td><td>3 go to cabinet 1</td><td>3 go to cabinet 1</td></tr><tr><td colspan="2"></td><td>4 go to cabinet 2</td><td>4 go to cabinet 2</td></tr><tr><td colspan="2"></td><td>5 open cabinet 2</td><td>5 open cabinet 2</td></tr><tr><td colspan="2"></td><td>6 go to drawer 1</td><td>6 go to cabinet 3</td></tr><tr><td colspan="2"></td><td>7 open drawer 1</td><td>7 go to cabinet 4</td></tr><tr><td colspan="2"></td><td>8 go to shelf 1</td><td>8 go to cabinet 5</td></tr><tr><td colspan="2"></td><td>9 go to shelf 2</td><td>9 open cabinet 5</td></tr><tr><td colspan="2"></td><td>10 go to shelf 3</td><td>10 go to cabinet 6</td></tr><tr><td colspan="2"></td><td>11 go to countertop 1</td><td>11 open cabinet 6</td></tr><tr><td colspan="2"></td><td>12 go to countertop 2</td><td>12 go to drawer 1</td></tr><tr><td colspan="2"></td><td>13 go to countertop 3</td><td>13 open drawer 1</td></tr><tr><td colspan="2"></td><td>14 take egg 1 from countertop 3</td><td>14 go to drawer 2</td></tr><tr><td colspan="2"></td><td>15 examine egg 1</td><td>15 open drawer 2</td></tr><tr><td colspan="2"></td><td>16 go to microwave 1</td><td>16 go to drawer 3</td></tr><tr><td colspan="2"></td><td>17 open microwave 1</td><td>17 open drawer 3</td></tr><tr><td colspan="2"></td><td>18 move egg 1 to microwave 1</td><td>18 go to shelf 1</td></tr><tr><td colspan="2"></td><td>19 close microwave 1</td><td>19 go to shelf 2</td></tr><tr><td colspan="2">Off-policy: SFT</td><td></td><td>OPW (Ours)</td></tr><tr><td>20 end 21 end</td><td></td><td>R</td><td>20 go to shelf 3 21 go to countertop 1</td></tr><tr><td>22 end</td><td></td><td></td><td>22 go to countertop 2</td></tr><tr><td>23 end</td><td></td><td></td><td>23 go to countertop 3</td></tr><tr><td>24 end</td><td></td><td></td><td>24 take egg 1 from countertop 3</td></tr><tr><td>25 end</td><td></td><td></td><td>25 go to sinkbasin 1</td></tr><tr><td>26 end</td><td></td><td></td><td>26 clean egg 1 with sinkbasin 1</td></tr><tr><td>27 end</td><td></td><td></td><td>27 go to microwave 1</td></tr><tr><td>28 end</td><td></td><td>ＲＲＲＲＲＲＲＲＲ</td><td>28 open microwave 1</td></tr><tr><td>29 end</td><td></td><td></td><td>29 move egg 1 to microwave 1</td></tr><tr><td>30 end</td><td></td><td></td><td></td></tr><tr><td colspan="2">Environment feedback Turn 18: “You move the egg 1 to the microwave 1."</td><td></td><td>Environment feedback Turn 26: “You clean the egg 1 using the sinkbasin 1."</td></tr></table>

## I LIMITATIONS AND SCOPE

Our experiments use ALFWorld and ScienceWorld as controlled testbeds for studying how OPW affects subsequent RLVR in multi-step interactive tasks. Extending this investigation to more diverse interaction settings is a natural direction for future work. Our theoretical analysis characterizes initial reward discovery under the stated assumptions, complementing the empirical evaluation of subsequent learning dynamics. In practice, the appropriate warmup duration and end-to-end training efficiency depend on the task, model configuration, and cost of teacher supervision. We examine multiple warmup durations and include teacher scoring in the recorded training cost, distinguishing faster subsequent learning from overall computational savings. Adaptive criteria for transitioning from warmup to RLVR could further improve the allocation of training resources.