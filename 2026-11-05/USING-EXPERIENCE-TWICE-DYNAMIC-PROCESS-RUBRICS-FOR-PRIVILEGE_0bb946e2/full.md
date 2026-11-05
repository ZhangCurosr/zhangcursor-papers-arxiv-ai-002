# USING EXPERIENCE TWICE: DYNAMIC PROCESS RUBRICS FOR PRIVILEGED TOKEN-LEVEL CREDIT AS-SIGNMENT

Shumao Wu<sup>1</sup>, Kongcheng Zhang<sup>2</sup>, Jianqiang Huang<sup>1,</sup> <sup>3</sup>, Baisheng Lai<sup>1,</sup> <sup>3∗</sup>

<sup>1</sup>Computer Network Information Center,University of Chinese Academy of Sciences,

<sup>2</sup>Zhejiang University

<sup>3</sup>Computer Network Information Center, Chinese Academy of Sciences

smwu@cnic.cn, zhangkc@zju.edu.cn, jqhuang@cnic.cn, bslai@cnic.cn

## ABSTRACT

Agent trajectories often contain both successful decisions and failed attempts, yet optimization with sparse trajectory-level rewards struggles to identify which critical decisions should be reinforced or suppressed. We observe that meaningful distinctions among on-policy trajectories can be distilled into natural-language process rubrics and used as process supervision signals. Based on this insight, we propose ONLINE RUBRIC SELF-DISTILLATION (ORSD), which leverages self-derived process rubrics for dense token-level supervision, thereby augmenting outcome-centric reinforcement learning. Specifically, the current policy independently generates multiple rubric bundles at each step according to the same group of trajectories, extracting candidate process rubrics from cross-trajectory behavioral differences. A frozen verifier then evaluates all rubric items against each trajectory in the group, and selects one candidate bundle based on bundlelevel alignment between rubric satisfaction and task success. ORSD treats the selected rubric bundle as privileged information during training, then compares the log-likelihoods of the original tokens with and without the rubric. The resulting likelihood gap provides token-specific process credit, which is added to the original trajectory-level GRPO advantage through additive advantage shaping. ORSD thus bridges trajectory-level outcome supervision and fine-grained process supervision, enabling token-level credit assignment using process signals. Extensive experiments on ALFWorld and WebShop show that ORSD accelerates policy improvement and boosts final task performance for both Qwen3-1.7B and Qwen3-4B under the same environment-interaction budget. Our code is available at https://github.com/AnnaXChun/ORSD.

## 1 INTRODUCTION

Large language model (LLM) agents have shown strong capabilities in web interaction (Zhou et al., 2024a) and software engineering (Yang et al., 2024), with applications extending to embodied exploration (Wang et al., 2024). Yet reliable decision-making in interactive environments remains challenging: agents must plan sequences of actions, explore unfamiliar situations, and reflect on feedback to revise unsuccessful strategies. Recent progress in reinforcement learning with verifiable rewards (RLVR) offers a promising training approach. DeepSeek-R1 (Guo et al., 2025) shows that verifiable reward feedback can substantially improve reasoning performance, with reflection and self-verification emerging during reinforcement learning. Building on these advances, agentic RL has also adopted Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to train agents through environment interaction, as demonstrated by RAGEN (Wang et al., 2025) in multi-turn tasks.

Despite these advances, relying solely on outcome rewards provides limited guidance about which intermediate decisions deserve credit. Under trajectory-level GRPO, all policy-generated tokens within a trajectory share the same advantage, so useful actions, redundant steps, and recoverable mistakes receive the same learning signal. To assign credit to individual steps, GiGPO (Feng et al., 2025) groups actions from matching environment states across trajectories and estimates step-level advantages through within-group comparisons. This refines credit across decisions, although tokens within each action still share the same step-level signal. More recently, natural-language rubrics have emerged as another source of supervision by making task-specific evaluation criteria explicit. Rubrics as Rewards (Gunjal et al., 2026) use these criteria to construct reward signals. RLCF (Viswanathan et al., 2025) constructs instruction-specific checklists in advance and aggregates criterion-level judgments into a response score, while OpenRubrics (Liu et al., 2026) derives rubrics from existing preference pairs and trains a rubric-guided reward model for preference optimization. Although rubric-based methods introduce richer evaluation criteria, their feedback usually still remains at the trajectory level, leaving credit assignment to specially individual tokens unresolved. Moreover, their rubric construction is separate from the policy’s ongoing learning process, so the criteria are not continually updated by the updated policy.

These limitations motivate the use of rubrics as privileged context for finer-grained credit assignment. Our core insight is that comparing the log-probabilities assigned to the same sampled tokens with and without rubrics can turn rubric-based guidance into token-level advantages. Such signals can refine trajectory-level advantages, allowing tokens within the same trajectory to receive different credits. Moreover, as the policy evolves during training, its exploration patterns and failure modes may also change. Generating rubrics online from on-policy trajectories allows them to reflect the policy’s current behavior, capturing task-specific constraints, failure patterns, and successful action sequences that static rubrics may overlook. This enables the credit assignment signal to adapt to the evolving policy rather than relying on fixed, predefined criteria. However, jointly deriving rubrics from on-policy experience and leveraging them as privileged context for token-level credit assignment remains underexplored.

In this work, we introduce ONLINE RUBRIC SELF-DISTILLATION (ORSD), a framework that turns on-policy experience into adaptive process supervision for token-level credit assignment. ORSD reuses these trajectories to construct process supervision in three steps. First, conditioned on both task prompt and the group of trajectories, the same policy independently generates multiple rubric bundles, each containing multiple criteria for evaluating intermediate states along the trajectories. Second, ORSD employs a verifier to assess the satisfaction of each rubric criterion on every trajectory, producing a satisfaction profile for each rubric bundle. It then measures the alignment between each satisfaction profile and the outcome rewards, and selects the rubric bundle with the highest correlation as privileged information. Third, ORSD augments the prompt with the selected privileged information and recomputes the token likelihoods of the sampled trajectories. The gap between the original and augmented token likelihoods is then used to compute the token-level advantage. As a result, ORSD transforms a group of on-policy trajectories into token-level credit, requiring neither additional rollouts nor additional environment interactions.

## Our contributions are summarized as follows:

• We propose ONLINE RUBRIC SELF-DISTILLATION (ORSD), a framework that uses onlinegenerated process rubrics to derive token-level credit. These rubrics are distilled from the policy’s own rollouts and serve as privileged information that augments the replay context, with token-level log-probability differences between replay with and without the rubrics underpinning this fine-grained credit allocation.

• To address the quality variation among generated rubrics, we propose a rubric-reward alignment mechanism that filters out low-value rubrics by measuring the alignment between rubric satisfaction and task success, thereby enhancing the quality of token-level credit.

• Experiments on ALFWorld and WebShop with Qwen3-1.7B and Qwen3-4B show that ORSD improves overall success over GRPO under the same environment-interaction budget. Ablations support the contributions of online rubric generation and rubric–reward alignment.

## 2 RELATED WORK

RLVR for LLM Agents. Reinforcement learning with verifiable rewards (RLVR) uses automatically verified outcomes to train language models. PPO and GRPO are commonly used for policy optimization, with GRPO estimating relative advantages within groups of sampled responses (Schulman et al., 2017; Shao et al., 2024). For LLM agents, training involves multiple rounds of interaction, during which policy’s decisions affect subsequent observations and the final reward. RAGEN examines learning dynamics in multi-turn agent RL, and Agent Lightning provides a framework for training agents from their execution traces (Wang et al., 2025; Luo et al., 2025). The environment interaction also supports additional learning signals. EnvRL uses state transitions for representation learning, while T<sup>2</sup>PO adjusts exploration through uncertainty estimates and turn-level resampling (Wang et al., 2026d;a).

Credit Assignment in Agentic RL. Credit assignment assigns different credit to individual steps or tokens in an agent trajectory. GiGPO groups steps across trajectories by state similarity and estimates relative advantages within each group (Feng et al., 2025). HGPO further groups steps according to their interaction contexts (He et al., 2026). ArCHer uses rewards from subsequent interactions to estimate the value of the entire response generated at the current turn, then uses this estimate to update the policy that generates the response (Zhou et al., 2024b). POAD further considers the role of individual tokens within an action: it treats action generation as a sequence of token-level decisions and estimates future rewards for each token, providing a finer-grained training signal (Wen et al., 2024). SC-GRPO conditions on verified solutions and weights token gradients according to the resulting distributional divergence (Shan et al., 2026).

Natural-language Rubrics. A natural-language rubric is a set of written criteria specifying which aspects of a response or reasoning process to evaluate. Prior work uses rubrics to provide feedback at different levels: Rubrics as Rewards assigns a reward to the entire response based on rubric satisfaction (Gunjal et al., 2026); Rubrics to Tokens estimates the relevance of each criterion to individual response tokens, providing token-level feedback (Xu et al., 2026); and Step-wise Rubric Rewards checks whether each reasoning step satisfies the criteria and assigns step-level rewards (Xie et al., 2026). These methods use rubrics to construct fine-grained supervision, with criteria generated offline and held fixed. We further investigate how to generate rubrics online that better match the current policy and use them for token-level credit assignment.

On-policy Distillation with Privileged Context. Knowledge distillation trains a student model using teacher predictions, with supervision applied to individual tokens or complete sequences (Hinton et al., 2015; Kim & Rush, 2016). Training can also use data generated by the learner itself: DAgger collects supervision at states visited by the learner’s policy (Ross et al., 2011), while GKD (Agarwal et al., 2024) trains on student-generated sequences with supervision provided by a teacher. OPSD (Zhao et al., 2026) uses the same model for self-distillation: the model first generates a response, then computes its log-probability given additional information to provide a training signal for the original policy. This additional information is called privileged context. Prior work also extracts such information from task execution records: Skill-SD (Wang et al., 2026b) summarizes experience from completed trajectories into natural-language skills, while SDPO (Hubotter et al.,¨ 2026) uses environmental feedback and successful attempts for self-distillation.

## 3 METHOD

In this section, we present the proposed ONLINE RUBRIC SELF-DISTILLATION (ORSD), a framework that turns on-policy rollouts into rubrics and then into token-level credit signals. As shown in Figure 1, at each training step, the current policy first samples a group of trajectories for the same task and obtains a reward for each trajectory. It then generates candidate rubric bundles from the task description and these trajectories. The rubric-reward alignment mechanism selects the bundle whose rubric satisfaction scores correlate most strongly with the rewards across the group. Finally, the token-wise log-likelihood difference between rollouts with and without the selected bundle enables per-token advantage shaping of the original trajectory-level advantage. In this way, ORSD reuses the same on-policy trajectories for rubric generation and privileged paired replay to derive token-specific credit without introducing additional environment interactions.

![](images/27cb223b79b5938ba0d82b3906da77dd5166724bdf3b094da3532356a5d08711.jpg)  
Figure 1: The overview of ORSD. (a) For each task, the policy first samples a group of trajectories. It then takes the task description and these trajectories as inputs to generate multiple rubric bundles. (b) Each Rubric bundle is evaluated on every trajectory, the rubric bundle with the largest rubric-reward alignment score is selected. (c) The selected rubric bundle is used to augment the context and compute the per-token likelihood gap between augmented and non-augmented contexts, enabling per-token advantage shaping of the original trajectory-level advantage.

## 3.1 PRELIMINARIES

Let $\pi _ { \mathrm { o l d } }$ denote the policy used to sample trajectories in the current step. For a task x, ORSD samples $N$ trajectories:

$$
\begin{array} { r } { \mathcal { T } = \{ \tau _ { i } \} _ { i = 1 } ^ { N } , \qquad \tau _ { i } \sim \pi _ { \mathrm { o l d } } ( { \cdot } \mid x ) . } \end{array}\tag{1}
$$

The reward for trajectory $\tau _ { i }$ is $r _ { i } = R _ { \mathrm { t r a i n } } ( \tau _ { i } )$ , as defined in Section 4.1. As in GRPO, ORSD normalizes rewards within each task group to obtain a trajectory-level advantage, which serves as the base advantage for subsequent token-level advantage shaping (Shao et al., 2024):

$$
A _ { i } ^ { \mathrm { G R P O } } = \frac { r _ { i } - \mu _ { r } } { \sigma _ { r } + \varepsilon _ { \mathrm { n u m } } } .\tag{2}
$$

Here, $\mu _ { r }$ and $\sigma _ { r }$ are the mean and sample standard deviation of the group rewards. ORSD will add a token-level correction to this advantage.

## 3.2 ONLINE RUBRIC BUNDLE GENERATION AND SELECTION

Building on the same-task rollout group mentioned above, ORSD further exploits the behavioral distinctions across trajectories to construct process supervision. For example, in WebShop, one trajectory may verify product attributes before purchasing, whereas another may skip this step and end up selecting an item that does not satisfy the user’s request. These behavioral differences can be summarized as natural-language process rubrics. At each training step, we provide the current policy with the task description x, the sampled trajectory group $\tau$ , and a rubric-generation instruction $p _ { r } .$ and independently sample M candidate rubric bundles:

$$
\begin{array} { r } { \mathcal { B } ^ { ( m ) } \sim \pi _ { \mathrm { o l d } } ( \cdot \mid x , \mathcal { T } , p _ { r } ) , \qquad m = 1 , \ldots , M . } \end{array}\tag{3}
$$

Each bundle contains a set of process criteria, denoted by ${ \cal B } ^ { ( m ) } = \{ \kappa _ { k } ^ { ( m ) } \} _ { k = 1 } ^ { K ^ { ( m ) } }$ , where the superscript (m) indexes the candidate bundle, k indexes a criterion within it, and $K ^ { ( m ) }$ is its number of criteria.

Different from prior works, our method produces candidate rubric bundles online using the policy instead of human-defined criteria or generation by stronger models. This mechanism enables the generated rubrics to capture the dynamics of on-policy trajectories, while the challenge is that candidate rubric bundles exhibit variable quality. To address this, we select one rubric bundle among the candidates based on a rubric–reward alignment score. Specifically, a verifier V first scores each trajectory against each criterion $\kappa _ { k } ^ { ( m ) }$

$$
S _ { i , k } ^ { ( m ) } = V ( x , \tau _ { i } , \kappa _ { k } ^ { ( m ) } ) \in [ 0 , 1 ] .\tag{4}
$$

For trajectory i, we average the criterion-level scores $S _ { i , k } ^ { ( m ) }$ to obtain the bundle-level satisfaction score $s _ { i } ^ { ( m ) }$ :

$$
s _ { i } ^ { ( m ) } = \frac { 1 } { K ^ { ( m ) } } \sum _ { k = 1 } ^ { K ^ { ( m ) } } S _ { i , k } ^ { ( m ) } .\tag{5}
$$

We then compute the Spearman correlation between bundle satisfaction and outcome rewards across all N trajectories for the task:

$$
\rho ^ { ( m ) } = \mathrm { S p e a r m a n } \Big ( \{ s _ { i } ^ { ( m ) } \} _ { i = 1 } ^ { N } , \{ r _ { i } \} _ { i = 1 } ^ { N } \Big ) .\tag{6}
$$

This correlation measures the agreement between the rankings in $\{ s _ { i } ^ { ( m ) } \} _ { i = 1 } ^ { N }$ and $\{ r _ { i } \} _ { i = 1 } ^ { N }$ . Higher alignment indicates that the bundle better distinguishes higher- and lower-reward trajectories. Note that to prevent the generated rubrics from degenerating into simple rules equivalent to outcome rewards, Equation 3 does not take outcome rewards as input when generating rubric bundles. We finally select the bundle with the highest correlation:

$$
m ^ { \star } = \arg \operatorname* { m a x } _ { m } \rho ^ { ( m ) } , \qquad B ^ { \star } = B ^ { ( m ^ { \star } ) } .\tag{7}
$$

## 3.3 RUBRIC-BASED ADVANTAGE SHAPING

The selected rubric bundle serves as privileged context for computing token-level credit. For each policy-generated token $y _ { i , \ell }$ at position ℓ in trajectory $\tau _ { i }$ , let $y _ { i , < \ell }$ denote the preceding interaction history, including earlier policy-generated tokens and environment observations. We compare the log-likelihoods assigned by $\pi _ { \mathrm { o l d } }$ to the same token with and without the selected rubric:

$$
\Delta _ { i , \ell } = \log \pi _ { \mathrm { o l d } } ( y _ { i , \ell } \mid x , \mathcal { B } ^ { \star } , y _ { i , < \ell } ) - \log \pi _ { \mathrm { o l d } } ( y _ { i , \ell } \mid x , y _ { i , < \ell } ) .\tag{8}
$$

Both forward passes evaluate the same pre-sampled trajectories, with no additional sampling. We compute the gap only for policy-generated tokens and mask environment observations. A positive $\Delta _ { i , \ell }$ increases the token’s advantage, while a negative value decreases it. The resulting token-level advantage is

$$
A _ { i , \ell } ^ { \mathrm { t o k e n } } = A _ { i } ^ { \mathrm { G R P O } } + \alpha \Delta _ { i , \ell } .\tag{9}
$$

The coefficient $\alpha \geq 0$ controls the correction magnitude. Note that the per-token advantage $A _ { i , \ell } ^ { \mathrm { t o k e n } }$ is derived from trajectories generated by the policy itself. For this reason, we categorize ORSD as a self-distillation mechanism that enables fine-grained credit assignment.

## 3.4 POLICY OPTIMIZATION

We optimize the policy using the token-level advantage $A _ { i , \ell } ^ { \mathrm { t o k e n } }$ defined in Eq. equation 9. For each policy-generated token, we define the importance sampling ratio as

$$
r _ { i , \ell } ( \theta ) = { \frac { \pi _ { \theta } ( y _ { i , \ell } \mid x , y _ { i , < \ell } ) } { \pi _ { \mathrm { o l d } } ( y _ { i , \ell } \mid x , y _ { i , < \ell } ) } } .\tag{10}
$$

The policy is optimized by minimizing the following clipped objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { O R S D } } ( \theta ) = \mathrm { ~ - ~ } \displaystyle \frac { 1 } { | \mathcal { B } _ { \mathrm { v a l i d } } | } \sum _ { ( i , \ell ) \in \mathcal { B } _ { \mathrm { v a l i d } } } \operatorname* { m i n } \Big \{ r _ { i , \ell } ( \theta ) A _ { i , \ell } ^ { \mathrm { t o k e n } } , } \\ & { ~ \mathrm { c l i p } ( r _ { i , \ell } ( \theta ) , 1 - \varepsilon _ { c } , 1 + \varepsilon _ { c } ) A _ { i , \ell } ^ { \mathrm { t o k e n } } \Big \} + \eta \mathcal { R } _ { \mathrm { K L } } ( \theta ) . } \end{array}\tag{11}
$$

where $\boldsymbol { B } _ { \mathrm { v a l i d } }$ denotes the set of valid policy-generated tokens in the micro-batch, $\varepsilon _ { c }$ is the clipping threshold, and $\mathcal { R } _ { \mathrm { K L } }$ denotes the KL regularization weighted by $\eta .$

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmarks and Metrics. We evaluate ORSD on ALFWorld (Shridhar et al., 2021) and Web Shop (Yao et al., 2022) two long-horizon interactive benchmarks using Qwen3-1.7B and Qwen3- 4B (Yang et al., 2025). For ALFWorld, we report task-wise and overall success rates on the unseen split. For WebShop, we report the official task score and success rate.

Prompting Baselines. We include published results for prompting methods that guide interaction without RL training. ReAct (Yao et al., 2023) interleaves reasoning and actions. Reflexion (Shinn et al., 2023) uses reflections on previous attempts to guide later attempts, while ReflAct (Kim et al., 2025) relates the current state to the task goal when choosing actions. ReAct w/ Self-Guidance (SG) adds self-generated guidance during interaction (Wang et al., 2026c).

Training Baselines. We compare ORSD with several baselines. (a) GRPO: its training data, reward definition, thinking mode, rollout count, and the number of training steps are strictly consistent with ORSD. (b) GiGPO (Feng et al., 2025): it estimates step-level advantages by grouping actions from similar states. We run GiGPO on both benchmarks with both Qwen3 backbones. (c) Selfdistillation methods, including OPSD (Zhao et al., 2026), GRPO+OPSD, and SDAR (Lu et al., 2026). GRPO+OPSD combines GRPO with an auxiliary distillation loss. The sources of published results and their evaluation protocols are specified in the notes to Table 1.

Training Details. We fine-tune all parameters of both backbones with VERL/HybridFlow (Sheng et al., 2025). Each run uses four A100 80GB GPUs. Every update contains eight tasks with eight rollouts per task. We train for 300 training steps at a learning rate of $1 0 ^ { - 6 }$ . WebShop episodes allow for at most 15 assistant actions, while ALFWorld interactions end according to the environment’s termination conditions. We generate M = 4 candidate rubric bundles per task and use $\alpha = 0 . 0 1$ $\varepsilon _ { c } = 0 . 2 ,$ and $\eta = 1 0 ^ { - 3 }$ by default. Appendix C provides the remaining implementation details.

Reward Definition For ALFWorld, the training reward is the binary task success indicator:

$$
R _ { \mathrm { t r a i n } } ( \tau ) = \mathbb { I } [ \mathrm { s u c c e s s } ] ,\tag{12}
$$

These rewards also form the outcome vector used for bundle selection. For WebShop, we use

$$
\begin{array} { r l } & { R _ { \mathrm { t r a i n } } ( \tau ) = 0 . 5 R _ { \mathrm { e n v } } ( \tau ) + 0 . 5 \mathbb { I } [ R _ { \mathrm { e n v } } ( \tau ) = 1 ] } \\ & { \qquad - 0 . 0 5 \frac { N _ { \mathrm { i n v a l i d } } ( \tau ) } { \operatorname* { m a x } ( T , 1 ) } - 0 . 0 2 5 \frac { N _ { \mathrm { l o o p } } ( \tau ) } { \operatorname* { m a x } ( T , 1 ) } . } \end{array}\tag{13}
$$

Here, $R _ { \mathrm { e n v } } ( \tau ) \in [ 0 , 1 ]$ is the official WebShop task score, and T counts all attempted environment actions, including invalid ones. $N _ { \mathrm { i n v a l i d } }$ and $N _ { \mathrm { l o o p } }$ count invalid actions and repeated actions that make no progress. This training reward is used to compute GRPO advantages and select rubrics on WebShop. Baselines run in this study use the same reward transformation. Evaluation only uses the original official metrics, namely the task score and success rate.

## 4.2 MAIN RESULTS

ORSD achieves the best overall results across both benchmarks and model scales. As shown in Table 1, ORSD ranks first in ALFWorld overall success and in both WebShop metrics with Qwen3- 1.7B and Qwen3-4B. On ALFWorld, ORSD exceeds the strongest baseline, GRPO, by 12.12 and 3.36 percentage points in overall success at the two model scales, respectively. On WebShop, ORSD improves success over the strongest baseline by 2.80 percentage points with Qwen3-1.7B, compared with GiGPO, and by 10.90 points with Qwen3-4B, compared with GRPO. The accompanying gains in WebShop score show that the improvement extends to satisfying the requested product attributes.

These results demonstrate the effectiveness of ORSD across household interaction and web shopping tasks. Its overall gains extend beyond GRPO to GiGPO and the reported self-distillation baselines, including SDAR. ORSD derives task-specific process criteria from rollout differences and uses the selected rubric to refine token-level learning signals through paired replay. The improvements over

Table 1: Performance on ALFWorld and WebShop. Bold indicates the best result for each metric.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>Overall</td><td>Score</td><td>Succ.</td></tr><tr><td rowspan="10">Qwen3 1.7B</td><td>ReAct$</td><td>24.20</td><td>12.50</td><td>0.00</td><td>0.00</td><td>0.00</td><td>12.10</td><td>10.20</td><td>41.60</td><td>4.70</td></tr><tr><td>Reflexion§</td><td>22.90</td><td>15.40</td><td>14.80</td><td>18.80</td><td>16.00</td><td>8.30</td><td>16.40</td><td>39.80</td><td>16.00</td></tr><tr><td>ReflAct§</td><td>28.60</td><td>23.10</td><td>7.40</td><td>6.30</td><td>8.00</td><td>0.00</td><td>12.90</td><td>37.50</td><td>4.40</td></tr><tr><td>ReAct w/ SG§</td><td>54.20</td><td>14.30</td><td>18.50</td><td>0.00</td><td>12.10</td><td>19.00</td><td>21.10</td><td>39.70</td><td>5.40</td></tr><tr><td>OPSD†</td><td>26.30</td><td>33.30</td><td>9.10</td><td>0.00</td><td>4.50</td><td>5.30</td><td>14.10</td><td>47.40</td><td>9.30</td></tr><tr><td>GRPO+OPSD†</td><td>38.20</td><td>50.00</td><td>30.80</td><td>28.60</td><td>30.00</td><td>21.10</td><td>32.00</td><td>70.70</td><td>38.30</td></tr><tr><td>SDAR†</td><td>73.50</td><td>25.00</td><td>76.90</td><td>33.30</td><td>40.00</td><td>36.80</td><td>53.90</td><td>76.80</td><td>58.60</td></tr><tr><td>GRPO</td><td>68.75</td><td>72.22</td><td>50.81</td><td>54.35</td><td>36.90</td><td>50.00</td><td>53.92</td><td>72.20</td><td>55.40</td></tr><tr><td>GiGPO</td><td>79.20</td><td>27.80</td><td>77.40</td><td>34.80</td><td>38.10</td><td>41.20</td><td>53.00</td><td>76.80</td><td>60.20</td></tr><tr><td>ORSD (ours)</td><td>81.25</td><td>68.06</td><td>69.35</td><td>79.35</td><td>53.57</td><td>41.18</td><td>66.04</td><td>78.60</td><td>63.00</td></tr><tr><td rowspan="6">Qwen3 4B</td><td>ReAct§</td><td>42.40</td><td>16.70</td><td>0.00</td><td>8.30</td><td>9.10</td><td>28.00</td><td>20.30</td><td>31.10</td><td>11.30</td></tr><tr><td>Reflexion§</td><td>68.60</td><td>38.50</td><td>37.00</td><td>25.00</td><td>32.00</td><td>4.20</td><td>37.10</td><td>57.80</td><td>24.00</td></tr><tr><td>ReflAct$</td><td>74.30</td><td>38.50</td><td>25.90</td><td>25.00</td><td>32.00</td><td>16.70</td><td>38.60</td><td>35.10</td><td>16.00</td></tr><tr><td>ReAct w/ SG§</td><td>87.80</td><td>50.00</td><td>29.20</td><td>58.30</td><td>27.30</td><td>80.00</td><td>58.60</td><td>42.70</td><td>17.60</td></tr><tr><td>GRPO</td><td>83.33</td><td>86.11</td><td>73.39</td><td>72.83</td><td>72.62</td><td>63.24</td><td>74.07</td><td>82.69</td><td>66.15</td></tr><tr><td>GiGPO ORSD (ours)</td><td>88.20</td><td>50.00 77.78</td><td>96.20 87.10</td><td>52.40 70.65</td><td>65.00 83.33</td><td>57.90 63.24</td><td>73.40 77.43</td><td>81.80 89.02</td><td>65.60</td></tr></table>

§: Prompting results from Co-Evolution ofPolicy and Internal Rewardfor Language Agents (Wang et al., 2026c). †: OPSD, GRPO+OPSD, and SDAR results from Self-Distilled Agentic Reinforcement Learning (Lu et al., 2026). Published results retain their source evaluation protocols.

GRPO under the same environment-interaction budget support this approach to extracting additional supervision from existing trajectories. The following ablations examine the contributions of online rubric generation and rubric-reward alignment.

## 4.3 ABLATION STUDY

In this section, we use ablation experiments to examine the effectiveness of ORSD’s components: online rubric generation, the number of candidate bundles and their selection, and the advantage shaping coefficient α. All ablations use WebShop. We study rubric generation and selection with both Qwen3 backbones and vary α with Qwen3-1.7B.

Effectiveness of Online Rubric Generation. To evaluate the effectiveness of online rubric generation, we construct an Offline Rubric variant. Before RL training, the initial policy generates one task-specific rubric bundle for each task, and these rubrics remain fixed throughout training. As shown in Table 2, Offline Rubric consistently improves over GRPO on both model scales, confirming that task-specific rubric supervision is already beneficial even when generated only once. However, it still falls short of online ORSD by 4.80 percentage points in success rate for both backbones. This gap highlights the value of refreshing the rubric from the current policy’s rollouts, so that the supervision remains aligned with the policy’s changing behavior and failure modes during training. Importantly, both variants generate rubrics automatically and require no manually authored criteria.

Effectiveness of Rubric-Reward Alignment. ORSD selects candidate rubric bundles based on their alignment with outcome rewards. We examine this design through two ablations. ORSD w/o Rubric-Reward Alignment generates a single bundle and uses it directly, omitting verifier scoring and candidate selection. Random Rubric retains the same candidate count as full ORSD (M = 4) but replaces alignment-based selection with random selection. Table 2 shows that full ORSD outperforms both variants on both model scales. Relative to the single-bundle variant, ORSD improves success by 2.15 and 9.45 percentage points on Qwen3-1.7B and Qwen3-4B, respectively. With the candidate count held fixed, alignment-based selection improves success over random selection by 3.40 and 7.15 percentage points, with corresponding score gains of 5.36 and 4.08 points. These consistent gains demonstrate the effectiveness of rubric-reward alignment in selecting process supervision for policy training, with the selected rubrics serving as privileged context for token-level advantage shaping.

Table 2: Component ablations on WebShop with two policy backbones. Baseline denotes GRPO. ORSD w/o Rubric–Reward Alignment generates one bundle and omits verifier scoring and candidate selection. Rubric-based variants use $\underline { { \alpha = 0 . 0 1 } }$
<table><tr><td></td><td colspan="2">Qwen3-1.7B</td><td colspan="2">Qwen3-4B</td></tr><tr><td>Method</td><td>Score</td><td>Success (%)</td><td>Score</td><td>Success (%)</td></tr><tr><td>Baseline</td><td>72.20</td><td>55.40</td><td>82.69</td><td>66.15</td></tr><tr><td>Offline Rubric</td><td>74.35</td><td>58.20</td><td>85.39</td><td>72.25</td></tr><tr><td>ORSD w/o Rubric–Reward Alignment</td><td>77.53</td><td>60.85</td><td>83.86</td><td>67.60</td></tr><tr><td>Random Rubric</td><td>73.24</td><td>59.60</td><td>84.94</td><td>69.90</td></tr><tr><td>ORSD  $( M = 4 )$ </td><td>78.60</td><td>63.00</td><td>89.02</td><td>77.05</td></tr></table>

Sensitivity to α. Figure 2 compares the default ORSD coefficient $\alpha = 0 . 0 1$ with $\alpha = 0 . 0 0 1$ and $\alpha = 0 . 1$ on WebShop using Qwen3-1.7B. We report the success rate and use the same GRPO result as the baseline at each setting; GRPO corresponds to removing the correction in Eq. (9). All three coefficients outperform GRPO, with the default $\alpha = 0 . 0 1$ achieving the highest success rate of 63.00%. These results indicate that $\alpha = 0 . 0 1$ provides the most effective correction strength among the tested values, supporting its use as the default setting.

![](images/5769db3d8f9bc4ef0cc939a7115d2bf4d5002d09f6b404a00cbe2f5a8233d470.jpg)  
Figure 2: Sensitivity to α on WebShop with Qwen3-1.7B. The default is $\alpha = 0 . 0 1$ . Each setting uses the same GRPO baseline (55.40% success).

## 4.4 FURTHER ANALYSIS

Training Dynamics Figure 3 compares the training rewards of ORSD and GRPO on both benchmarks and model scales. On WebShop, ORSD improves earlier with Qwen3-1.7B and maintains higher rewards through most of training with Qwen3-4B. On ALFWorld, the advantage is more pronounced at 1.7B and emerges later at 4B. Across all four settings, ORSD achieves higher mean rewards over the final 300 optimization steps under the same environment-interaction budget. Together with the gains in Table 1, these results support more effective use of sampled trajectories: ORSD reuses rollout experience to derive fine-grained supervision through self-distillation, improving learning without additional environment rollouts.

Analysis of Generated Rubric Bundles Table 3 compares a selected WebShop rubric with a rejected candidate. The selected bundle specifies product attributes and a concrete price constraint, providing more informative task-specific privileged context than the rejected candidate’s general criteria. This comparison illustrates how rubric-reward alignment selects more informative privileged context to guide token-level advantage shaping.

![](images/417e1ba30ac12fab075bb677b1c90fb5174c74aea9f4cf63ab57734e3ebbd641.jpg)  
Figure 3: Training reward curves comparing ORSD with GRPO on WebShop and ALFWorld using Qwen3-1.7B and Qwen3-4B. Both methods use the same environment-interaction budget.

Table 3: WebShop rubric comparison. The left column contains a rubric saved at step 300; the right column contains a rejected proposal.
<table><tr><td colspan="2">Saved rubric Rejected rubric</td></tr><tr><td colspan="2">WebShop: Buy a women&#x27;s top with the requested attributes, color, and size for less than $30.</td></tr><tr><td>1. Identify the correct product category and specific product attributes (e.g., slim fit, loose fit, color, size, price) needed to find the desired item. 2. Ensure the search query includes all relevant</td><td>1. Identify the appropriate product category and the key attributes specified in the user instruction. 2. Use relevant product keywords and available</td></tr><tr><td>keywords and filters to narrow down the results effectively. 3. Search using the relevant product attributes and narrow the candidates accordingly. Do not</td><td>search or filtering options to narrow down candidate items. 3. Select the appropriate product variant, such as the requested size, color, or other configurable</td></tr><tr><td>commit to a product based only on a partial match in the title or description. 4. Verify that the selected product meets the price criteria (lower than $30.00) before</td><td>attributes. 4. Verify that the selected product satisfies the</td></tr><tr><td>proceeding to purchase. 5. Confirm that the product is correctly selected and the purchase action is initiated.</td><td>important constraints in the instruction, including product attributes and price requirements. 5. Confirm that the final product matches the user&#x27;s request and complete the purchase</td></tr></table>

## 5 DISCUSSION AND LIMITATIONS

Rubric Knowledge is not explicitly accumulated across updates. ORSD generates task-specific rubrics from the current rollout group at each update, but does not maintain a persistent repository of previously discovered criteria. How to retain, refine, and reuse useful rubrics across training stages and related tasks remains an open question. Extending ORSD with a rubric memory could support cumulative process knowledge while preserving adaptation to the evolving policy.

Outcome alignment does not fully resolve credit assignment. ORSD selects rubric bundles by their correlation with trajectory-level rewards. This objective favors rubrics that distinguish stronger trajectories, but does not directly optimize their usefulness for assigning credit within a trajectory. Future work could develop selection objectives that account for how rubric guidance affects individual decisions and subsequent policy improvement.

## 6 CONCLUSION

We introduced ONLINE RUBRIC SELF-DISTILLATION (ORSD) to improve credit assignment in LLM agent reinforcement learning. ORSD generates task-specific process rubrics online from differences among on-policy trajectories, without manually authored or fixed offline criteria. Rubric-reward alignment selects the bundle whose satisfaction scores best correlate with outcomes. Through privileged paired replay, the selected rubric converts token-level likelihood gaps into token-level advantage corrections, providing fine-grained supervision without additional environment interactions. Experiments on ALFWorld and WebShop with Qwen3-1.7B and Qwen3-4B show the best overall performance among the compared methods, while ablations validate online rubric generation and candidate selection. These results highlight the value of deriving adaptive process supervision from the policy’s own experience for credit assignment in long-horizon agent tasks.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-Group Policy Optimization for LLM Agent Training. In Advances in Neural Information Processing Systems, volume 38, pp. 51797–51830, 2025. doi: 10.52202/085713-1544. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ cfd7bee7a651ee9af525098ef67a9e45-Abstract-Conference.html.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. doi: 10.48550/arXiv.2501.12948. URL https://arxiv.org/abs/2501.12948.

Shuo He, Lang Feng, Qi Wei, Xin Cheng, Lei Feng, and Bo An. Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2602.22817.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. URL https://arxiv.org/abs/1503.02531.

Jonas Hubotter, Frederike L¨ ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta,¨ Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement Learning via Self-Distillation. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, 2026. URL https://arxiv.org/abs/2601.20802.

Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, and Kyomin Jung. ReflAct: World-grounded decision making in LLM agents via goal-state reflection. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 33433–33465, 2025. doi: 10.18653/v1/2025.emnlp-main.1697. URL https: //aclanthology.org/2025.emnlp-main.1697/.

Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327. Association for Computational Linguistics, 2016. doi: 10.18653/v1/D16-1139. URL https: //aclanthology.org/D16-1139/.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17417–17437. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.791. URL https://aclanthology.org/2026.acl-long. 791/.

Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Self-distilled agentic reinforcement learning. arXivpreprint arXiv:2605.15155, 2026. URL https://arxiv.org/abs/2605.15155v1.

Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, and Yuqing Yang. Agent Lightning: Train ANY AI Agents with Reinforcement Learning. arXiv preprint arXiv:2508.03680, 2025. doi: 10.48550/arXiv.2508.03680. URL https://arxiv. org/abs/2508.03680.

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15, pp. 627–635. PMLR, 2011. URL https: //proceedings.mlr.press/v15/ross11a.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707. 06347. URL https://arxiv.org/abs/1707.06347.

Yingyu Shan, Yuhang Guo, Zihao Cheng, Zeming Liu, Xiangrong Zhu, Xinyi Wang, Jiashu Yao, Wei Lin, Hongru Wang, and Heyan Huang. Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards. arXiv preprint arXiv:2606.18810, 2026. doi: 10.48550/arXiv.2606.18810. URL https://arxiv.org/abs/2606.18810.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. URL https://arxiv.org/abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A Flexible and Efficient RLHF Framework. In Proceedings of the Twentieth European Conference on Computer Systems, 2025. doi: 10.1145/ 3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://papers.neurips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://arxiv.org/ abs/2010.03768.

Vijay Viswanathan, Yanchao Sun, Xiang Kong, Meng Cao, Graham Neubig, and Sherry Wu. Checklists Are Better Than Reward Models For Aligning Language Models. In Advances in Neural Information Processing Systems, volume 38, pp. 114728–114754. Curran Associates, Inc., 2025. doi: 10.52202/085713-3824. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ a6837c1dd021f76f1b4098e3722052a8-Paper-Conference.pdf.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=ehfRiF0R3a.

Haixin Wang, Hejie Cui, Chenwei Zhang, Xin Liu, Shuowei Jin, Shijie Geng, Xinyang Zhang, Nasser Zalmout, Zhenyu Shi, and Yizhou Sun. T<sup>2</sup>PO: Uncertainty-guided exploration control for stable multi-turn agentic reinforcement learning. In International Conference on Machine Learning, 2026a. URL https://arxiv.org/abs/2605.02178.

Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. Skill-SD: Skill-Conditioned Self-Distillation for Multiturn LLM Agents. arXiv preprint arXiv:2604.10674, 2026b. doi: 10.48550/arXiv.2604.10674. URL https://arxiv.org/abs/2604.10674.

Xinyu Wang, Hanwei Wu, Jingwei Song, Shuyuan Zhang, Jiayi Zhang, Fanqi Kong, Tung Sum Thomas Kwok, Xiao-Wen Chang, Yuyu Luo, Chenglin Wu, and Bang Liu. Co-evolution of policy and internal reward for language agents. arXiv preprint arXiv:2604.03098, 2026c. URL https://arxiv.org/abs/2604.03098v1.

Zhitong Wang, Songze Li, Hao Peng, Shuzheng Si, Yi Wang, Maosong Sun, and Juanzi Li. EnvRL: Learn from environment dynamics in agentic reinforcement learning. arXiv preprint arXiv:2606.17680, 2026d. doi: 10.48550/arXiv.2606.17680. URL https://arxiv.org/ abs/2606.17680.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025. URL https://arxiv.org/abs/2504.20073.

Muning Wen, Ziyu Wan, Jun Wang, Weinan Zhang, and Ying Wen. Reinforcing LLM agents via policy optimization with action decomposition. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://papers.neurips.cc/paper\_files/paper/2024/ hash/bc09efb501c801ed92e181e26a885c2d-Abstract-Conference.html.

Weichu Xie, Haozhe Zhao, Wenpu Liu, Yongfu Zhu, Liang Chen, Minghao Ye, Zirong Chen, Yuqi Xu, Shuai Dong, Ziyue Wang, Xinbo Xu, Kean Shi, Ruoyu Wu, Xiaoying Zhang, Wenqi Shao, Baobao Chang, Nan Duan, and Jiaqi Wang. Step-wise Rubric Rewards for LLM Reasoning. arXiv preprint arXiv:2605.17291, 2026. doi: 10.48550/arXiv.2605.17291. URL https://arxiv. org/abs/2605.17291.

Tianze Xu, Yanzhao Zheng, Pengrui Lu, Lyumanshan Ye, Yong Wu, Zhentao Zhang, Yuanqiang Yu, Chao Ma, Jihuai Zhu, Pengfei Liu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, and Gang Yu. Rubrics to Tokens: Bridging Response-Level Rubrics and Token-Level Rewards in Instruction Following Tasks. In Conference on Language Modeling, 2026. URL https: //arxiv.org/abs/2604.02795.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Advances in Neural Information Processing Systems, volume 37, pp. 50528–50652. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1601. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pp. 20744–20757, 2022. doi: 10.52202/068431-1508. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2601.18734.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations, volume 2024, pp. 15585–15606, 2024a. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 4410c0711e9154a7a2d26f9b3816d1ef-Paper-Conference.pdf.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. ArCHer: Training language model agents via hierarchical multi-turn RL. In Proceedings of the 41st International Conference on Machine Learning, pp. 62178–62209, 2024b. URL https://proceedings.mlr.press/ v235/zhou24t.html.

## A ALGORITHM

1. Let $\pi _ { \mathrm { o l d } }$ denote the policy used for trajectory sampling and paired replay in the current update.

2. For every task x, sample a group of $N = 8$ trajectories T and compute $A _ { i } ^ { \mathrm { G R P O } }$

3. Aggregate the task and all trajectories in the group into one rubric-generation prompt, then independently sample four complete rubric bundles with 5-8 criteria each using the current policy.

4. Use the frozen verifier to score every criterion in each complete bundle. Average all criterion scores within each trajectory, then compute bundle-level Spearman correlation with group outcomes and select the complete bundle with the highest score.

5. Replay each stored action turn with and without $B ^ { \star }$ . Check response masks and token alignment, compute the raw $\Delta _ { i , \ell } .$ , and add the valid-token correction in Eq. (9).

6. Accumulate the policy loss and reference-policy KL regularizer in Eq. (11); update the shared policy once.

## B PROMPT TEMPLATES

One aggregated prompt supplies the current policy with the task and all action-observation transcripts in the rollout group. With the group-hint instruction, the prompt requests a complete process rubric containing 5-8 observable criteria. The policy samples four independent complete proposals from the same prompt, and each proposal remains intact throughout selection.

For verification, the frozen verifier receives the task, the rollout transcripts, and one candidate bundle. Each criterion is scored independently from visible evidence and receives a satisfaction score in [0, 1]. All criterion scores in the bundle are averaged within each trajectory, then correlated with group outcomes across trajectories using Spearman correlation.

The following templates specify rubric generation and policy interaction in the two environments. Blue placeholders denote fields populated for each task. For rubric generation, k = 8 and the previous-attempts field contains the same-task rollout group.

## Rubric Generation Prompt

## [SYSTEM]

You are a task coach who writes process rubrics for an interactive environment.

You will see one task and several previous attempts at the same task.

Use them to write useful process criteria for a NEW agent that has never seen those attempts.

Write 5 to {k} short rubric criteria.

Useful rubric criteria can describe:

\- important subgoals needed to complete the task,

\- useful ordering of subgoals,

\- mistakes or no-progress behavior to avoid,

\- what must be true before the task is finished.

## Rules:

\- Use only information supported by the task or the attempts.

\- Do not mention attempts, trajectories, rollouts, rewards, sources, evidence, or IDs.

\- Do not copy a complete action sequence.

\- Do not invent object locations or hidden state.

\- Keep each rubric criterion short, concrete, and useful by itself.

\- Do not explain your reasoning.

Output only a numbered process rubric:

1. ...

2. ...

## Rubric Generation Prompt (continued)

3. ...

4. ...

5. ...

[USER]

Task:

{problem}

## Previous attempts:

{attempts}

Write a useful process rubric for a new agent solving this same task.

## ALFWorld Interaction Prompt

## [SYSTEM]

You are acting in ALFWorld. Observe the environment and its valid actions, maintain task state, and reply with exactly ‘ACTION: <one currently valid action>‘.

[USER]

Goal: {goal}

## WebShop Interaction Prompt

## [SYSTEM]

You are an agent interacting with a virtual text-based web shopping environment to test out your ability. Your job is to follow the Instruction provided and mimic the steps to buy the item that are closest to the Instruction provided.

## Action Format:

You should give both the action name and action arg like the format

‘action name[action arg]‘. You can execute two types of actions, search and click.

\- When the button ‘[button] Search [button ]‘ is available in the current observation, you can execute the action ‘search[xxx]‘ (you should type the query you want to search in the square brackets here).

\- You can click buttons ‘[button] xxx [button ]‘ that is available in the current observation, by execute the action ‘click[xxx]‘.

Below are some examples of action formats.

\- ACTION: search[white shoes]

\- ACTION: click[Buy Now]

## Notes:

You should ALWAYS take one action each step.

You should finish the task and buy the item within 15 steps.

## [USER]

DO NOT try to interact with the user at anytime. Finish the task and buy the item by yourself.

{instruction}

Table 4: Optimization and sampling configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Base models</td><td>Qwen3-1.7B; Qwen3-4B</td></tr><tr><td>Optimization</td><td>Full-parameter GRPO</td></tr><tr><td>Hardware</td><td>4 × A100 80GB</td></tr><tr><td>Prompts / rollouts per prompt</td><td>8/8</td></tr><tr><td>Optimizer steps</td><td>300</td></tr><tr><td>Rollout group size</td><td>8 (ALFWorld and WebShop)</td></tr><tr><td>Learning rate</td><td>10⁻6</td></tr><tr><td>Clipping parameter εc</td><td>0.2</td></tr><tr><td>KL regularizer / coefficient</td><td>low-variance estimator /  $1 0 ^ { - 3 }$ </td></tr><tr><td>Token-credit coefficient α</td><td>0.01</td></tr><tr><td>Rubric candidates per task</td><td>4 complete bundles</td></tr><tr><td>Criteria per bundle</td><td>5-8</td></tr><tr><td>Bundle selection</td><td>highest Spearman correlation with group outcomes</td></tr><tr><td>Policy micro-batch size per GPU</td><td>ALFWorld: 2; WebShop: 1</td></tr><tr><td>Replay eligibility</td><td>selected bundle; valid alignment and finite scores</td></tr><tr><td>Frozen verifier</td><td>Qwen3.5-Plus via DashScope API</td></tr><tr><td>Verifier decoding</td><td>temperature 0, thinking disabled, JSON only; 1,536-token cap</td></tr><tr><td>Policy / privileged-replay thinking</td><td>on</td></tr><tr><td>Generator thinking / prompt</td><td>off / aggregated numbered rubric</td></tr><tr><td>Generator temperature / top-p / cap</td><td>0.3 / 0.9 / 4,096</td></tr><tr><td>Verifier concurrency / retries / cap</td><td>8 /4 /1,536</td></tr><tr><td>Validation sampling</td><td>4 trajectories per task, temperature 0.6</td></tr><tr><td>Framework</td><td>VERL 0.8, FSDP, async vLLM rollout</td></tr></table>

Table 5: Environment-specific configuration. Full history denotes the complete sequence of previous reasoning responses, actions, and observations.
<table><tr><td>Parameter</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>Evaluation outcome</td><td>official won</td><td>official score and score = 1</td></tr><tr><td>Episode horizon</td><td>environment termination</td><td>15 assistant actions</td></tr><tr><td>Response budget</td><td>16,384</td><td>30,720</td></tr><tr><td>Interaction history</td><td>full history</td><td>full history</td></tr><tr><td>Bundle-selection outcome</td><td>official success</td><td>behavior-regularized return</td></tr><tr><td>Training reward</td><td>Eq. (12)</td><td>Eq. (13)</td></tr><tr><td>Invalid-action coefficient</td><td>0 0</td><td>0.05</td></tr><tr><td>No-progress coefficient</td><td></td><td>0.025</td></tr><tr><td>vLLM memory utilization</td><td>0.50</td><td>0.72</td></tr><tr><td>Agent-loop workers</td><td>8</td><td>4</td></tr></table>

## C IMPLEMENTATION DETAILS

Evaluation and learning curves. We evaluate every 50 optimizer steps on fixed held-out tasks. Comparison runs in this study use the same evaluation checkpoints and training seeds. Figure 3 plots training rewards against training steps. Each update collects 64 environment trajectories, so step 300 corresponds to 19,200 training trajectories. Verifier calls, replay computation, and wall-clock time are recorded separately.

## C.1 POLICY LOSS REDUCTION

The main text uses $A _ { \mathrm { m b } }$ to denote the masked micro-batch reduction. Let D be the number of dataparallel workers, G the number of gradient accumulation steps per worker, and $\mathcal { C } _ { d , g }$ the trajectories in the corresponding micro-batch, indexed across tasks for this reduction. For any token-level quantity $f ,$

$$
\mathcal { A } _ { \mathrm { m b } } [ f ] = \frac { 1 } { D G } \sum _ { d = 1 } ^ { D } \sum _ { g = 1 } ^ { G } \frac { \sum _ { i \in \mathcal { C } _ { d , g } } \sum _ { \ell } b _ { i , \ell } f _ { i , \ell } } { \sum _ { i \in \mathcal { C } _ { d , g } } \sum _ { \ell } b _ { i , \ell } } .\tag{14}
$$

Here, $b _ { i , \ell }$ is a binary mask for valid response tokens, excluding prompts, environment observations, and padding. This reduction first averages over tokens within each micro-batch, then over microbatches and workers; it does not separately average over tasks or turns. The term $\varepsilon _ { \mathrm { n u m } }$ ensures numerical stability when computing GRPO advantages.

## C.2 PAIRED REPLAY DATA

Paired replay uses records saved at each generation turn. A record contains the turn index, context messages and token IDs, complete response token IDs, action-only response history, subsequent observation, response mask, and rollout log probabilities. Plain replay packs $c _ { t } + y _ { t }$ independently for each turn. Rubric-conditioned replay inserts the selected complete bundle into $c _ { t }$ and reuses the same target response. The response mask selects the log probabilities used by the correction. The implementation mask $e _ { i , \ell }$ equals one when the tokenization round-trip succeeds, response tokens align, and both log-likelihoods are finite. A token with $e _ { i , \ell } = 0$ retains its GRPO advantage. The main-text equations omit this implementation mask. The policy ratio continues to use the original context and saved rollout log probabilities.

## C.3 TRAINING ENTRY POINTS AND RUN RECORDS

Two launchers under examples/agent rubric2 define the WebShop group-hint and ALFWorld first-version group-hint configurations. Both aggregate all trajectories in a same-task rollout group, sample four complete bundles with the current policy, select one complete proposal, and apply the raw additive token correction. GRPO uses the same four-GPU, eight-task, eight-rollout, 300-step protocol with rubric generation and token correction disabled.

Rubric generation uses temperature 0.3, $\mathrm { t o p } { \cdot } p = 0 . 9$ , and a 4,096-token cap. Verification calls qwen3.5-plus with concurrency 8, at most four retries, and a 1,536-token output cap. Each run record contains the API identifier and date, resolved configuration, task IDs, seeds, saved trajectories, bundle scores, and checkpoint evaluations. Runs are grouped under verl rubric 3.

Both replay branches preserve the per-turn context and response IDs, masks, chat templates, temperature settings, and rubric insertion rules. Before applying the advantage correction, token scores are written back to their original response positions. The micro-batch loss reduction and reference-policy KL settings match GRPO.

## D MATHEMATICAL FORM OF THE TOKEN CORRECTION

Effect on trajectory credit. For $\begin{array} { r } { n _ { i } = \sum _ { \ell } b _ { i , \ell } > 0 . } \end{array}$ , the actual mean advantage is

$$
\frac { 1 } { n _ { i } } \sum _ { \ell } b _ { i , \ell } A _ { i , \ell } ^ { \mathrm { t o k e n } } = A _ { i } ^ { \mathrm { G R P O } } + \frac { \alpha } { n _ { i } } \sum _ { \ell } b _ { i , \ell } e _ { i , \ell } \Delta _ { i , \ell } .
$$

The raw correction can therefore change both the trajectory mean and individual token signs. Its scale is controlled by α and the replay-alignment mask; no additional normalization or redistribution is applied.

Self-distillation through advantage shaping. Both terms in $\Delta _ { i , \ell }$ are computed by the same $\pi _ { \mathrm { o l d } }$ ORSD applies stop-gradient, denoted by sg below, to the likelihood gap and includes it only in $A _ { i , \ell } ^ { \mathrm { t o k e n } }$ . At the start of an update, the additional contribution to the policy-loss gradient is

$$
\begin{array} { r } { - \alpha \ A _ { \mathrm { m b } } [ e _ { i , \ell } \ \mathrm { s g } [ \Delta _ { i , \ell } ] \ \nabla _ { \theta } \log \pi _ { \theta } ( y _ { i , \ell } \ | \ x , y _ { i , < \ell } ) ] _ { \theta = \theta _ { \mathrm { o l d } } } . } \end{array}\tag{15}
$$

Here, $A _ { \mathrm { m b } }$ denotes the masked micro-batch reduction in $\mathrm { E q . } ( 1 4 )$ , applied to the enclosed token-level quantity. Privileged context changes the weight assigned to each sampled token in the RL update. ORSD implements privileged self-distillation through advantage shaping; $\mathcal { L } _ { \mathrm { t o t a l } }$ contains no auxiliary KL or cross-entropy matching objective. The likelihood gap measures the policy’s response to the selected process context and does not establish token correctness.