# When to Follow the Teacher: Advantage-Based On-Policy Distillation

Mahdi Sabbaghi<sup>\*1</sup>, George Pappas<sup>1</sup>, Hamed Hassani<sup>1</sup>, and Adel Javanmard<sup>2</sup>

<sup>1</sup>University of Pennsylvania <sup>2</sup>University of Southern California

## Abstract

Online Policy Distillation (OPD) often outperforms offline methods such as Supervised Fine-Tuning (SFT) by training on trajectories sampled directly from the student. However, standard OPD trains the student to follow the teacher at every state of the trajectory, assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed. We introduce Advantage-Based On-Policy Distillation (AOPD), which weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student’s expected outcome are reinforced, while those that do not are downweighted.

Directly computing these advantages requires trying alternative teacher actions at each step and running new rollouts from each. Instead, we show that the AOPD objective can be estimated using only standard student rollouts and their terminal rewards. Our estimator uses teacher importance weights on the student sampled actions to provide unbiased advantage-weighted supervision, without any additional rollouts or reward evaluations. This allows AOPD to focus teacher supervision on actions that help the current student while preserving the rollout efficiency of standard OPD. Experiments on math, coding, and Medqa show gains over distillation and RL baselines in both external-teacher and self-distillation settings.

<sup>§</sup> https://github.com/Helloworld10011/AOPD

## 1 Introduction

Knowledge distillation transfers capabilities from a stronger teacher to a smaller student model by training the student to imitate the teacher’s outputs. Offline distillation is a common approach that uses fixed teachergenerated trajectories. But this approach can lead to a mismatch between the training trajectories and the states the student generates itself. On-policy distillation (OPD) addresses this mismatch by querying the teacher along student-sampled trajectories, providing dense token-level supervision at the states the student actually visits (Agarwal et al., 2024; Lu and Lab, 2025; Song and Zheng, 2026; Ross et al., 2011; Li et al., 2026a). Although the trajectories now come from the student, the objective remains to match the teacher, without distinguishing which teacher actions improve the student’s chance of success (Liu et al., 2026).

This limitation also applies to on-policy self-distillation (OPSD), which replaces the external teacher with a privileged version of the student itself, such as conditioned on ground-truth solutions (Zhao et al., 2026; Hübotter et al., 2026). Whether external or privileged, the teacher may choose an action from which it can solve the problem, while the student cannot complete the remaining steps correctly. Conversely, distillation discourages a valid student solution when it differs from the teacher’s preferred approach. Thus the question is whether the teacher action at every state increases the student’s chance of success. An action should be credited according to its impact on the student’s task success, not simply the teacher’s preference (Chen et al., 2026; Saadi and Wang, 2026).

Prior attempts to mitigate this optimization misalignment rely on making the teacher supervision more selective. A common approach combines outcome-based reinforcement learning with distillation, but adding a reward term does not change which teacher actions the distillation term reinforces (Xu et al., 2025; Zhang et al., 2026b). Other methods downweight distillation when the teacher is uncertain, using entropy as a proxy, or when teacher tokens are implausible under the student (Jin et al., 2026; Wang et al., 2026). Alternative approaches validate teacher actions using auxiliary reward models (Saadi and Wang, 2026; Gu et al., 2026) or explicit intervention rollouts (Chen et al., 2026). These methods provide ways to identify useful teacher guidance, but rely on auxiliary signals or additional counterfactual trajectories. More importantly, they do not formulate the objective directly around the impact of the teacher action on the student’s expected task reward. Therefore, we ask:

![](images/e524bc37ba80a73028f289654b0041f6d453600bfc5bbef0e381b660d2540568.jpg)

![](images/be16032713768c77c2209a5ed9f08d8e80afd4b73958721a99914ddc4f3e2131.jpg)

![](images/89489cfa9a645195c4c33857fa650666e3f54c70a69423b7de7de5fd24d280ed.jpg)  
Figure 1: (left) The overall picture of our method. On the student sampled trajectories, not all the suggested actions by the teacher are useful to the student, and many suggestions are destructive. (middle) Performance comparison of our method with OPSD when training Qwen3-8B in non-thinking mode on Open-Thoughts. (right) Comparison with OPSD when training Qwen3-4B-Instruct on TACO for coding.

How should a distillation objective weight teacher supervision to improve the student’s eventual task success?

We propose Advantage-based On-Policy Distillation (AOPD), the first formulation of on-policy distillation that directly weights teacher actions by their advantage under the current student. Intuitively, the advantage measures whether taking a teacher action and then letting the student finish the response improves expected final reward over following the student’s own policy. Since this advantage is defined with respect to the current student, the same teacher action can be useful for one student but unhelpful for another. AOPD therefore reinforces teacher actions with positive advantage and suppresses those with negative advantage. This is illustrated in Figure 1 (left).

A direct estimate of these advantages would require generating new student continuations after alternative teacher actions. We instead derive an unbiased importance-sampling estimator of the AOPD objective using only the student’s on-policy trajectories and their terminal rewards. This allows AOPD to incorporate outcomelevel rewards into token-level teacher supervision without learning a value function, generating counterfactual continuations or teacher rollouts, or relying on an external process reward model.

Our contributions are summarized as follows:

• AOPD algorithm. In Section 3, we provide an advantage-weighted formulation of on-policy distillation and derive an unbiased estimator of its objective, without additional rollouts or auxiliary models. We also discuss clipping the importance sampling weights.

• Experimental results. In Section 4, we evaluate multiple model sizes on tasks such as math and coding under both external-teacher and self-distillation settings, showing improvements over OPD, OPSD, and GRPO. We further examine whether teacher actions are uniformly beneficial in an ablation study.

• Reward improvement beyond imitation. In Section 5, we establish a reward-improvement guarantee for AOPD under an exact KL-proximal analysis and show that forward-KL OPD can reduce reward even with a stronger teacher. We further construct a two-state counterexample showing that simply adding an RL objective to OPD does not guarantee reward improvement.

## 2 Related Work

Offline supervision. Supervised fine-tuning and offline knowledge distillation learn from fixed demonstrations or teacher-generated trajectories (Hinton et al., 2015; Kim and Rush, 2016; Wen et al., 2023; Gu et al., 2024). As established in imitation learning, offline supervision suffers from covariate shift and compounding errors when the learned policy visits states absent from the training distribution (Ross et al., 2011; Bengio et al., 2015; Agarwal et al., 2024). This mismatch is particularly relevant for autoregressive reasoning, where early deviations change all subsequent steps drastically (Gu et al., 2024; Agarwal et al., 2024).

On-policy reinforcement learning. On-policy RL instead learns from trajectories generated by the current policy and has become central to LLM alignment and reasoning through methods such as PPO, RLOO, GRPO, and related policy-gradient algorithms (Schulman et al., 2017; Ouyang et al., 2022; Ahmadian et al., 2024; Shao et al., 2024; Sabbaghi et al., 2026b). RLVR directly optimizes task reward but typically provides sparse outcome-level supervision and can struggle when useful trajectories are rarely sampled (Casper et al., 2023; Yue et al., 2025; Xu et al., 2025; Xiong et al., 2025).

On-policy distillation. On-policy distillation combines on-policy sampling with dense teacher supervision by matching teacher and student distributions along student-generated trajectories (Agarwal et al., 2024; Gu et al., 2024; Lu and Lab, 2025; Li et al., 2026a; Song and Zheng, 2026). OPSD further uses a privileged version of the student itself as the teacher, often conditioning it on ground-truth solutions or other privileged information (Zhao et al., 2026; Hübotter et al., 2026; Penaloza et al., 2026; Ye et al., 2026). These approaches correct where supervision is provided, but generally still apply teacher supervision uniformly and throughout the student’s trajectory.

Selective teacher supervision. Recent methods make this supervision selective using teacher uncertainty, teacher–student compatibility, local validity, verifier signals, or additional intervention rollouts (Jin et al., 2026; Wang et al., 2026; Saadi and Wang, 2026; Gu et al., 2026; Chen et al., 2026; Zhang et al., 2026a). These criteria identify when teacher guidance may be useful, but do not directly define usefulness by its effect on the student’s eventual task reward. AOPD instead introduces a new formulation for on-policy distillation that weights each teacher action by its advantage under the current student and estimates this signal from the existing on-policy rollouts and terminal rewards.

## 3 Method: Advantage-based On-Policy Distillation

Let S denote the frozen student policy used to generate the current rollout batch, $S _ { \theta }$ the policy being updated, and T the teacher policy. Given a prompt x, the student generates a response $y = \left( y _ { 0 } , \dots , y _ { H - 1 } \right)$ according to $\begin{array} { r } { S ( y \mid x ) = \prod _ { t = 0 } ^ { H - 1 } S ( y _ { t } \mid h _ { t } ) , h _ { t } = ( x , y _ { < t } ) } \end{array}$ , then $h _ { t } \sim S ( \cdot \mid x )$ denotes a sampled trajectory up to step t. The terminal reward $R ( x , y ) \in [ 0 , 1 ]$ indicates task success. We assume a shared token vocabulary V and access to teacher probabilities $T ( \cdot \mid h _ { t } )$ ) at student-generated histories.

Reverse-KL distillation. Reverse-KL distillation (Agarwal et al., 2024) minimizes:

$$
\mathcal { L } _ { \mathrm { R K L } } ( \theta ) = \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim S ( . ) } \left[ { \bf K L } \big ( S _ { \theta } ( \cdot \mid h _ { t } ) \| T ( \cdot \mid h _ { t } ) \big ) \right] .\tag{3.1}
$$

At each step, the actions come from $a \sim S ( \cdot \mid h _ { t } )$ and the expectation is over log-ratio log $\frac { T ( a | h _ { t } ) } { S ( a | h _ { t } ) }$ . Thus, the teacher evaluates actions already sampled by the student.

Forward-KL distillation. Forward-KL distillation instead samples the action from teacher’s distribution at each step of the student sampled trajectory and minimizes the log-ratio log $\frac { S ( a | h _ { t } ) } { T ( a | h _ { t } ) }$

$$
\mathcal { L } _ { \mathrm { F K L } } ( \theta ) = \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim S ( . ) } \left[ \mathbf { K L } \big ( T ( \cdot \mid h _ { t } ) \big \| S _ { \theta } ( \cdot \mid h _ { t } ) \big ) \right] .\tag{3.2}
$$

Dropping the teacher entropy that is independent of θ, Equation (3.2) is equivalent to the following maximization objective:

$$
J _ { \mathrm { F K L } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } \sim \boldsymbol { S } ( \cdot | \boldsymbol { x } ) } \left[ \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \boldsymbol { a } \sim \boldsymbol { T } ( \cdot | \boldsymbol { h } _ { t } ) } \left[ \log S _ { \theta } ( \boldsymbol { a } \mid \boldsymbol { h } _ { t } ) \right] \right] .\tag{3.3}
$$

The objective reinforces teacher-supported actions at student-generated histories $h _ { t } .$ . In practice, FKL and RKL objectives roughly reach the same performance, and we will develop our method for FKL. Critically, FKL reinforces the teacher actions without distinguishing whether they improve the student’s final task outcome.

On-policy self-distillation. On-policy self-distillation (OPSD) (Zhao et al., 2026) uses the same objective above, but it replaces the external teacher with a privileged version of the student itself. Given privileged information $y ^ { * }$ , such as the ground-truth solution, the teacher distribution becomes:

$$
T _ { \mathrm { O P S D } } ( a \mid h _ { t } ) = S ( a \mid h _ { t } , y ^ { * } )
$$

and the rollout is still generated from the unprivileged student $S ( \cdot \mid h _ { t } )$ . Our method makes no assumptions about how T is obtained and therefore applies directly to both OPD and OPSD.

Our method: Advantage-based on-policy distillation (AOPD). Starting from the FKL objective in Equation (3.3), we preserve the teacher supervision at the same student-generated histories but change how teacher actions are weighted. Rather than reinforcing every teacher-supported action, we weight each action by its effect on the student’s expected reward. For a history $h = \left( x , y _ { < t } \right)$ , define:

$$
\begin{array} { r l } & { Q ^ { S } ( h _ { t } , a ) = \mathbb { E } _ { y _ { > t } \sim S ( \cdot | h _ { t } a ) } \left[ R ( x , y _ { < t } a y _ { > t } ) \right] , \quad V ^ { S } ( h _ { t } ) = \mathbb { E } _ { a \sim S ( \cdot | h _ { t } ) } [ Q ^ { S } ( h _ { t } , a ) ] } \\ & { A ^ { S } ( h _ { t } , a ) = Q ^ { S } ( h _ { t } , a ) - V ^ { S } ( h _ { t } ) . } \end{array}\tag{3.4}
$$

These are the standard RL value and advantage function. The advantage $A ^ { S }$ compares the expected reward after taking a with the expected reward when the student follows its own action policy. We therefore modify the FKL objective by weighting each teacher action by $A ^ { S } ( h _ { t } , a )$

$$
J _ { \mathrm { A O P D } } ( \theta ) = \mathbb { E } _ { x , y \sim S ( \cdot \vert x ) } \left[ \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { a \sim T ( \cdot \vert h _ { t } ) } \left[ A ^ { S } ( h _ { t } , a ) \log S _ { \theta } ( a \mid h _ { t } ) \right] \right] .\tag{3.5}
$$

Unlike the forward KL objective, AOPD assigns positive weights to teacher actions that improve the student’s expected reward and negative weights to those that reduce it. Both the advantage and the history distribution are defined under the frozen rollout policy S. In Theorem 5.1, we show that our objective increases the average reward with guarantee, whereas OPD has an arbitrary sign that depends on the advantage of teacher’s actions.

Estimating AOPD from student rollouts. Directly evaluating $Q ^ { s }$ for teacher actions would require separate student continuations from the teacher interventions. Instead, we estimate the entire objective using a student rollout $y ^ { S } \sim S ( \cdot \mid x )$ and its terminal reward $R ^ { S } \ : = \ : R ( x , y ^ { S } )$ . At each step, let $a _ { t } \ = \ y _ { t } ^ { S }$ and define $\begin{array} { r } { w _ { t } = \frac { T ( a _ { t } | h _ { t } ) } { S ( a _ { t } | h _ { t } ) } } \end{array}$ . The ratio $w _ { t }$ converts expectations under the student into expectations under the teacher. We use centered rewards $\widetilde { R } ^ { S } = R ^ { S } - b$ for a group with G responses, where b is a baseline independent of $y ^ { S }$ (e.g., the leave-one-out mean). The resulting estimator is:

$$
\boxed  \widehat { J } _ { \mathrm { A O P D } } ( \theta ) = \frac { \widetilde { R } ^ { S } } { H } \sum _ { t = 0 } ^ { H - 1 } \left[ w _ { t } \log S _ { \theta } ( a _ { t } \mid h _ { t } ) - \mathbb { E } _ { a \sim T } \left[ \log S _ { \theta } ( a \mid h _ { t } ) \right] \right] , \quad w _ { t } : = \frac { T ( a _ { t } \mid h _ { t } ) } { S ( a _ { t } \mid h _ { t } ) } \left[ \begin{array} { l } { \hat { w } _ { 0 } ( a _ { t } \mid h _ { t } ) } \\ { \hat { w } _ { 0 } ( a _ { t } \mid h _ { t } ) } \end{array} \right] .\tag{3.6}
$$

In the following lemma, we show that the two terms within the sum estimate the contributions of $Q ^ { S } - b$ and $V ^ { S } - b$ , respectively. So their difference obtains the desired advantage in Equation (3.5).

Lemma 3.1 (Unbiased AOPD estimator). Assume that $T ( a \mid h ) > 0$ implies $S ( a \mid h ) > 0$ at every visited

Algorithm 1 AOPD with the unbiased estimator   
Require: Student $S _ { \theta } ,$ teacher T, reward function R, group size G   
1: for each training iteration do   
2: Freeze the rollout policy $S  S _ { \theta }$   
3: for each prompt $x \in B$ do   
4: Sample G independent responses $y ^ { ( i ) } \sim S ( \cdot \mid x )$   
5: Compute terminal rewards $R _ { i } = R ( x , y ^ { ( i ) } )$   
6: Set $\begin{array} { r } { b _ { i } = \frac { 1 } { G - 1 } \sum _ { j \neq i } R _ { j } } \end{array}$ for $G > 1 .$   
7: Evaluate student and teacher probabilities at every visited history.   
8: Compute $\widetilde { J } _ { \mathrm { A O P D } } ^ { ( i ) } ( \theta )$ using Equation (3.8).   
9: $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) \gets - \frac { 1 } { | \mathcal { B } | G } \sum _ { \boldsymbol { x } \in \mathcal { B } } \sum _ { i = 1 } ^ { G } \widetilde { J } _ { \mathrm { A O P D } } ^ { ( i ) } ( \boldsymbol { \theta } ) } \end{array}$ ▷ differentiating only through $S _ { \theta }$

step. We have:

$$
\mathbb { E } \left[ \widehat { J } _ { \mathrm { A O P D } } ( \theta ) \right] = J _ { \mathrm { A O P D } } ( \theta )
$$

And the same identity holds for their gradients.

The proof is given in Appendix B. When $\widetilde { R } ^ { S } > 0 ,$ AOPD reinforces the successful student action, with a larger update when the teacher also assigns it high probability. Unlike GRPO, this focuses the positive update on successful actions supported by the teacher. The second term suppresses alternative teacher actions, since the sampled student action already led to success. When $\widetilde R ^ { S } < 0$ , the signs reverse: AOPD suppresses the failed student action, especially when the teacher also favors it, while promoting alternative teacher actions. Thus, AOPD uses the teacher to distinguish which actions should be reinforced or corrected based on the rollout outcome.

Clipping the importance sampling. Large importance sampling weights can make rare sampled tokens dominate the update, so we cap the importance sampling as:

$$
\widetilde { w } _ { t } = \operatorname* { m i n } \left( \frac { T ( a _ { t } \mid h _ { t } ) } { S ( a _ { t } \mid h _ { t } ) } , c \right)\tag{3.7}
$$

In our experiments this constant is $c = 4$ . We apply the same truncation to teacher probabilities in the second term without re-normalization: $\widetilde { T } ( a | h ) = \mathrm { m i n } \bar { \{ } T ( a | h ) , c S ( a | h ) \}$ . The resulting estimator is:

$$
\widetilde { J } _ { \mathrm { A O P D } } ( \theta ) = \frac { \widetilde { R } ^ { S } } { H } \sum _ { t = 0 } ^ { H - 1 } \left[ \widetilde { w } _ { t } \log S _ { \theta } ( a _ { t } \mid h _ { t } ) - \sum _ { a \in \mathcal { V } } \widetilde { T } ( a \mid h _ { t } ) \log S _ { \theta } ( a \mid h _ { t } ) \right] .\tag{3.8}
$$

We show in Corollary 5.3 that applying the same truncation to both terms preserves the advantage-weighted objective and its reward-improvement guarantee. We further discuss the impact of clipping on training in Appendix A. Algorithm 1 summarizes the final version of our algorithm.

## 4 Experiments

We evaluate AOPD in two settings: (i) distillation from a stronger external teacher and (ii) self-distillation from a version of the student with privileged information. We compare AOPD against existing distillation and RL methods across math, coding, biology, and medical question answering.

## 4.1 Experimental Setup

Tasks. We conduct our experiments across four tasks:

Math: We train on Open-Thoughts (Guha et al., 2025) and evaluate on held-out Open-Thoughts problems, AIME24, AIMO-Validation-AMC (AMC), and HMMT (AI-MO, 2024a,b; Dekoninck et al., 2026).

Coding: We train on verified TACO problems (Li et al., 2023; Li, 2024) and evaluate on HumanEval, HumanEval+, and MultiPL-E (Chen et al., 2021; Cassano et al., 2022).

Biology: We train on SciKnowEval (Feng et al., 2024) and evaluate on 50 held-out biology questions and the biology subset of MMLU-Pro (Wang et al., 2024).

Medical QA: We train on MedMCQA (Pal et al., 2022) and evaluate on held-out MedMCQA and the separate MedQA benchmark (Jin et al., 2021).

Models and configurations. Following previous work (Zhao et al., 2026; Ye et al., 2026), we use Qwen2.5 and Qwen3 models of different sizes (Yang et al., 2025). For external-teacher distillation, we use Qwen3- 1.7B as the student, with Qwen3-4B-Instruct as the teacher for math training and Qwen3-8B as the teacher for coding. For self-distillation, we use Qwen3-1.7B and Qwen3-8B for math, and Qwen2.5-7B-Instruct and Qwen3-4B-Instruct for biology, medical question answering, and coding. All Qwen3 experiments use non-thinking mode for both the student and teacher.

Baselines. For external-teacher distillation, we compare AOPD with OPD using forward-KL (FKL) and reverse-KL (RKL) objectives (Agarwal et al., 2024; Lu and Lab, 2025), GRPO (Shao et al., 2024), SFT, and KDRL (Xu et al., 2025), which combines OPD and GRPO. For self-distillation, we compare with OPSD and GRPO, using the forward-KL objective for OPSD following previous work (Zhao et al., 2026; Jin et al., 2026). Within each comparison, we use the same number of training iterations, a batch size of 32, and the same LoRA adapters with $( r = 6 4 , \alpha = 1 2 8 )$ , while selecting method-specific hyperparameters.

Hyperparameters. To preserve the support required by the importance-sampling estimator in Equation (3.6), we disable top-p and top-k pruning for AOPD. We use a sampling temperature of $T = 0 . 8$ , which we found effective for generating coherent Qwen responses. For OPD and OPSD, we use $T = 1 . 1$ , top- $\cdot p = 0 . 9 5$ , and top-k = 20, following Zhao et al. (2026) and the settings that reproduce their best performance. For the main Qwen3 math and coding experiments, we use a learning rate of $2 \times 1 0 ^ { - 5 }$ for AOPD.

## 4.2 External-Teacher Distillation

We first evaluate AOPD with stronger external teachers on math and coding. For math, we use Qwen3-4B-Instruct as the teacher because it performs better on Open-Thoughts than Qwen3-8B in non-thinking mode. All methods train for 450 iterations on Open-Thoughts and 350 iterations on TACO. Math training uses a binary correctness reward. For coding, we use partial rewards based on five test cases, which we found more effective for training. We use G = 8 for GRPO and KDRL, following Xu et al. (2025), and G = 4 for OPD and AOPD. This is because our ablations show that larger groups provide little accuracy benefit for OPD and AOPD (Appendix A). For math, we use a maximum completion length of 8,192 tokens during training and 16,192 tokens at evaluation.

Results. Figure 2 compares OPD (FKL), GRPO, and AOPD during training. On math, OPD improves quickly and then plateaus, consistent with observations in previous work (Yang et al., 2026; Tan et al., 2026). On the other hand, GRPO improves more slowly and requires more iterations to converge. AOPD also improves rapidly at the beginning but also continues to improve later in training. This is because unlike standard OPD, which enforces uniform supervision, our objective dynamically adapts which teacher signals to follow as the student improves. Although OPD and GRPO achieve comparable Open-Thoughts evaluation rewards, OPD performs better on AIME24, as shown in Figure 2 (middle).

On TACO, OPD makes little progress in evaluation reward, whereas AOPD improves and remains above GRPO (Figure 2, right). Nevertheless, OPD (FKL) improves HumanEval and MultiPL-E performance over the base model (Table 1). Overall, our finding across the experiments is that the all distillation methods achieve better performance on benchmarks regardless of the training reward.

Table 1 shows the comparison for all methods across the math and coding benchmarks. For math, AOPD achieves the highest pass@1 on Open-Thoughts, AIME24, and AMC. It improves AIME24 by 2.4 percentage points and AMC by 3.3 points over the baselines. For coding, AOPD achieves the best scores on HumanEval. However, GRPO achieves the highest HumanEval pass@10, while OPD (FKL) performs the best on MultiPL-E.

![](images/a4cfb327b68dfd697b3d4dde0bfbe1dfb7e94f0d411377ebe32a9d95577f73c6.jpg)

![](images/255a24232d4776bf490b0e202ad95c678f5e5d4a371cd8411bdc182da4c625ec.jpg)

![](images/9af3e976acc8f03588fb6c245e5c47c6e903a007c9e92dd9b6483efd634ded89.jpg)

Figure 2: Training progress under external-teacher distillation with Qwen3-1.7B as the student. (Left/middle) Open-Thoughts evaluation reward and AIME24 performance. (Right) TACO evaluation reward for coding.
<table><tr><td rowspan="2">Models</td><td colspan="4">Math Training on Open-Thoughts</td><td colspan="4">Code Training on TACO</td></tr><tr><td>OpenThoughts pass@1</td><td>AIME24 pass@1</td><td>AMC pass@1</td><td>HMMT pass@1</td><td>HE</td><td>HumanEval HE+</td><td>pass@10</td><td>MultiPL-E Avg. (7 lang.)</td></tr><tr><td>Qwen3-1.7B</td><td>56.6</td><td>14.7</td><td>40.1</td><td>5.8</td><td>68.3</td><td>61.6</td><td>82.5</td><td>31.4</td></tr><tr><td>+ SFT</td><td>68.0</td><td>23.3</td><td>53.5</td><td>12.6</td><td>69.5</td><td>62.3</td><td>82.4</td><td>31.0</td></tr><tr><td>+ GRPO</td><td>71.8</td><td>22.3</td><td>51.4</td><td>10.6</td><td>71.8</td><td>64.0</td><td>87.9</td><td>31.1</td></tr><tr><td>+ KDRL</td><td>72.8</td><td>25.0</td><td>55.8</td><td>12.1</td><td>71.5</td><td>66.7</td><td>84.4</td><td>31.0</td></tr><tr><td>+ OPD (FKL)</td><td>73.2</td><td>32.1</td><td>61.8</td><td>16.0</td><td>71.3</td><td>66.5</td><td>85.3</td><td>33.3</td></tr><tr><td>+ OPD (RKL)</td><td>72.0</td><td>32.6</td><td>61.0</td><td>14.2</td><td>68.3</td><td>62.2</td><td>81.8</td><td>33.2</td></tr><tr><td>+ AOPD</td><td>78.4</td><td>35.0</td><td>65.1</td><td>15.8</td><td>74.4</td><td>68.9</td><td>86.5</td><td>32.5</td></tr></table>

Table 1: External-teacher distillation on math and coding with Qwen3-1.7B as the student. The teachers are Qwen3-4B-Instruct for math and Qwen3-8B for coding.

## 4.3 Self-Distillation

We next replace the external teacher with a version of the student conditioned on privileged information. For math, we use Qwen3-1.7B and Qwen3-8B, training for 400 iterations on Open-Thoughts with G = 4 for OPSD and AOPD. The maximum training completion length is 8,192 tokens. For the remaining tasks, we use Qwen2.5-7B-Instruct and Qwen3-4B-Instruct for biology, medical question answering, and coding. We train models for 300 iterations with a maximum completion length of 2,048 tokens on medical QA, and for 400 iterations with a maximum completion length of 4,096 tokens on biology and coding.

Math results. Table 2 compares GRPO, OPSD, and AOPD for both students. As Figure 3 shows, for Qwen3-1.7B, OPSD makes little progress on Open-Thoughts, consistent with previous work (Li et al., 2026b). This suggests limited benefit from the self-distillation signal in this setting. Nevertheless, AOPD reaches a comparable evaluation reward to GRPO and outperforms the baselines on all three math benchmarks. On AIME24, AOPD reaches 35.0%, improving over GRPO by 12.7 percentage points and OPSD by 19.4 points. For Qwen3-8B, OPSD is more effective and improves performance across the benchmarks. AOPD again achieves substantial gains in evaluation reward and benchmark performance, as illustrated in Table 2 and Figure 1 (middle). In particular, AOPD reaches 52.1% on AIME24, compared with 41.7% for OPSD, a gain of 10.4 percentage points.

Results with instruct models. We further evaluate AOPD across biology, medical question answering, and coding. As shown in Table 3, for biology, AOPD achieves the highest accuracy on SciKnowEval Biology and MMLU-Pro with Qwen2.5-7B, reaching 54.0% and 73.8%, respectively. For medical question answering, AOPD outperforms GRPO and OPSD on both MedQA and MedMCQA with both models. In particular, with Qwen2.5-7B-Instruct, AOPD reaches 74.8% on MedQA and 69.8% on MedMCQA, improving over OPSD by 2.7 and 3.2 percentage points. The training progress on MedQA is also shown in Figure 3. For coding, AOPD improves the evaluation reward significantly as shown in Figure 1 (right). AOPD achieves the best HumanEval+ scores with both models. On MultiPL-E, AOPD reaches 61.1% with Qwen3-4B-Instruct, outperforming GRPO by 5.9 points, although it does not outperform GRPO with Qwen2.5-7B-Instruct.

<table><tr><td></td><td colspan="4">Qwen3-1.7B</td><td colspan="4">Qwen3-8B</td></tr><tr><td>Method</td><td>OpenThoughts pass@1</td><td>AIME24 pass@1</td><td>AMC pass@1</td><td>HMMT pass@1</td><td>OpenThoughts pass@1</td><td>AIME24 pass@1</td><td>AMC pass@1</td><td>HMMT pass@1</td></tr><tr><td>Base</td><td>56.6</td><td>14.7</td><td>40.1</td><td>5.8</td><td>68.8</td><td>27.5</td><td>59.6</td><td>14.2</td></tr><tr><td>+ GRPO</td><td>71.8</td><td>22.3</td><td>51.4</td><td>10.6</td><td>78.5</td><td>30.6</td><td>63.8</td><td>13.6</td></tr><tr><td>+ OPSD</td><td>55.2</td><td>15.6</td><td>45.7</td><td>6.4</td><td>72.7</td><td>41.7</td><td>75.0</td><td>18.4</td></tr><tr><td>+ AOPD</td><td>70.2</td><td>35.0</td><td>61.4</td><td>14.2</td><td>84.6</td><td>52.1</td><td>78.6</td><td>20.8</td></tr></table>

Table 2: Math results for self-distillation on Open-Thoughts with Qwen3-1.7B and Qwen3-8B, both in nonthinking mode.
<table><tr><td></td><td>Model</td><td>Benchmark</td><td>Base</td><td>GRPO</td><td>OPSD</td><td>AOPD</td></tr><tr><td rowspan="3">Biology</td><td>Qwen2.5-7B-Instruct</td><td>SciKnowEval MMLU-Pro</td><td>20.0</td><td>42.0</td><td>41.0</td><td>54.0</td></tr><tr><td rowspan="2">Qwen3-4B-Instruct</td><td></td><td>71.3</td><td>72.5</td><td>71.6</td><td>73.8</td></tr><tr><td>SciKnowEval MMLU-Pro</td><td>36.0 84.5</td><td>38.0 84.1</td><td>44.0 84.9</td><td>32.0 84.8</td></tr><tr><td rowspan="4">Medical</td><td>Qwen2.5-7B-Instruct</td><td>MedQA</td><td>61.5</td><td>70.3</td><td>72.1</td><td>74.8</td></tr><tr><td rowspan="2"></td><td>MedMCQA</td><td>56.5</td><td>64.5</td><td>66.6</td><td>69.8</td></tr><tr><td>MedQA</td><td>72.7</td><td>71.1</td><td>70.6</td><td>76.8</td></tr><tr><td rowspan="2">Qwen3-4B-Instruct</td><td>MedMCQA</td><td>60.2</td><td>59.8</td><td>60.1</td><td>61.6</td></tr><tr><td>HE+</td><td>76.0</td><td>76.2</td><td>76.8</td><td>77.8</td></tr><tr><td rowspan="3">Coding</td><td>Qwen2.5-7B-Instruct</td><td>MultiPL-E</td><td>38.3</td><td>39.9</td><td>38.2</td><td>38.5</td></tr><tr><td rowspan="2">Qwen3-4B-Instruct</td><td>HE+</td><td>83.1</td><td>84.2</td><td>81.8</td><td>85.1</td></tr><tr><td>MultiPL-E</td><td>55.9</td><td>55.2</td><td>53.3</td><td>61.1</td></tr></table>

Table 3: Self-distillation results across biology, medical question answering, and coding. AOPD outperforms GRPO and OPSD across all three domains.

## 4.4 Ablation: Is the teacher always useful?

To test whether teacher-preferred actions necessarily benefit the student, we conduct a single-token intervention study with Qwen3-1.7B as the student and Qwen3-4B-Instruct-2507 as the teacher. We also consider a selfdistillation setting, where the teacher is Qwen3-1.7B with access to privileged information. For a set of 512 student response on Open-Thoughts, we select a position either (i) uniformly at random or (ii) by maximizing the KL divergence KL $\left( T ( \cdot | h ) \| S ( \cdot | h ) \right)$ ) between the teacher and student distributions. We replace the student’s token with the teacher’s highest-probability token, then let the student complete the response greedily under the same 8192-token total budget. We compare the final answer of this intervention with the original response.

The results are shown in Table 4. Teacher interventions cause both improvements and failures: with choosing the highest-disagreement position, teacher actions help (i.e., changing the reward to 1) in 9.6% of cases but harm (i.e., changing the reward to 0) in 9.8%. The high unchanged rate is expected as single-token interventions on the entire trajectory can rarely flip the final outcomes.

Interestingly, the privileged self-teacher provides more helpful interventions than the external teacher at uniformly random positions (8.6% vs. 6.9%). This suggests that self-distillation can provide more beneficial guidance in some settings, despite using a smaller model. This further highlights the importance of selecting teacher guidance based on its contribution to the student’s reward rather than the teacher’s preferences alone.

![](images/0bf3587b8be8f20a22f9ccc89a877abe3e3572c7d038ff1851bb629f5687a018.jpg)  
Figure 3: Training progress for GRPO, OPSD, and AOPD in self-distillation. (Left/middle) Open-Thoughts evaluation reward for Qwen3-1.7B and Qwen3-8B. (Right) MedQA test accuracy after training on MedMCQA with Qwen2.5-7B-Instruct.

<table><tr><td>Teacher</td><td>Intervention position</td><td>Helpful (%)</td><td>Unchanged (%)</td><td>Harmful (%)</td></tr><tr><td rowspan="2">External</td><td>Random</td><td>6.9</td><td>85.1</td><td>8.0</td></tr><tr><td>Highest KL disagreement</td><td>9.6</td><td>80.6</td><td>9.8</td></tr><tr><td rowspan="2">Self-distilled</td><td>Random</td><td>8.6</td><td>83.4</td><td>8.0</td></tr><tr><td>Highest KL disagreement</td><td>9.0</td><td>81.4</td><td>9.6</td></tr></table>

Table 4: Effect of replacing a student token with the teacher’s preferred token under external-teacher and self-distillation settings. Interventions can both improve and harm the student’s final outcome.

## 5 Theory: Reward Improvement with Advantage-based Distillation

Although a teacher policy may achieve a higher reward than the student when evaluated as a complete policy, this does not imply that every partial update toward the teacher improves the student. In practice, training does not replace the student policy with the teacher policy directly; rather, it proceeds through a finite sequence of updates, often with a student whose capacity is limited relative to that of the teacher. Consequently, the teacher’s eventual advantage does not establish that intermediate training updates improve task performance.

To characterize the effect of training on the student, we analyze a single update under different objectives and examine whether it increases the student’s expected final reward.

A proximal view of distillation. Following Section 3, let S denote the current, frozen student model, T the teacher model, and $A ^ { S }$ the student advantage function. Let $R ( x , y ) \in [ 0 , 1 ]$ denote the terminal reward obtained by response y for prompt x. We define the expected task reward of a policy π as $\mathcal { R } ( \pi ) =$ $\mathbb { E } _ { x , y \sim \pi ( \cdot | x ) } [ R ( x , y ) ]$

Our goal is to analyze a limited update from the current frozen student S to to the updated policy $\pi ,$ rather than assuming the policy can move far arbitrary. To model this limited update from the current student, we follow standard trust-region analyses (Schulman et al., 2015; Tomar et al., 2020) and the proximal analysis in Sabbaghi et al. (2026a). We restrict the KL movement from S with the following constraint:

$$
\operatorname { K L } ( S \| \pi ) = { \frac { 1 } { H } } \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \operatorname { K L } \bigl ( S ( \cdot \mid h _ { t } ) \| \pi ( \cdot \mid h _ { t } ) \bigr ) \right] \leq \epsilon
$$

Adding this constraint to the optimization for objective J provides a controlled notion of how much the student can change in one update. Writing the Lagrangian of this constrained problem gives the proximal update:

$$
S ^ { + } \in \arg \operatorname* { m a x } _ { \pi } \left\{ J ( \pi ) - \frac { 1 } { \eta } \mathrm { K L } ( S \| \pi ) \right\} ,\tag{5.1}
$$

where J is either the FKL-OPD objective in Equation (3.3) or the AOPD objective in Equation (3.5). Here $1 / \eta$ is the dual variable for ϵ, so smaller η gives a more conservative update. We conduct the analysis in policy space, characterizing the local update induced by each objective rather than the parameter-space dynamics of SGD. Accordingly, we write the student policy as π in place of $S _ { \theta }$ . Throughout, we hold the teacher, rollout distribution, and student advantage function $\bar { A ^ { S } }$ fixed at their values for the current student policy. Under this proximal-update framework, the following result compares FKL-OPD with AOPD:

Theorem 5.1 (Reward Improvement from a Single Proximal Update). (i) For sufficiently small $\eta > 0 ,$ , the exact proximal updates satisfy the following relation at every reachable history h:

$$
\mathbb { E } _ { a \sim S _ { \mathrm { F K L } } ^ { + } ( \cdot | h ) } [ A ^ { S } ( h , a ) ] = \frac { \eta } { 1 + \eta } \mathbb { E } _ { a \sim T ( \cdot | h ) } [ A ^ { S } ( h , a ) ] ,\tag{5.2}
$$

and its sign under OPD remains indeterminate (see Lemma 5.4 for concrete example). Whereas,

$$
\mathbb { E } _ { a \sim S _ { \mathrm { A O P D } } ^ { + } ( \cdot | h ) }  { [ A ^ { S } ( h , a ) ] } = \frac { \eta ~ \mathbb { E } _ { a \sim T ( \cdot | h ) }  { [ A ^ { S } ( h , a ) ^ { 2 } ] } } { 1 + \eta ~ \mathbb { E } _ { a \sim T ( \cdot | h ) }  { [ A ^ { S } ( h , a ) ] } } \geq 0 .\tag{5.3}
$$

(ii) We have the identity: $\begin{array} { r } { \mathcal { R } ( \pi ) - \mathcal { R } ( S ) = \mathbb { E } _ { x , y \sim \pi } \left[ \sum _ { t = 0 } ^ { H - 1 } A ^ { S } ( h _ { t } , a _ { t } ) \right] } \end{array}$ . Consequently ${ \mathcal R } ( S _ { \mathrm { A O P D } } ^ { + } ) \geq { \mathcal R } ( S )$ The inequality is strict when a teacher-supported action has nonzero advantage at a history reached with positive probability under $S _ { \mathrm { A O P D } } ^ { + } .$

The distinction follows directly from the two update expressions. FKL shifts probability toward the teacher’s actions irrespective of whether those actions improve the current student. Its resulting reward change therefore depends on the teacher’s average advantage under the student, which may be negative even when the teacher achieves a higher overall reward. In contrast, AOPD weights teacher actions by their student advantages, yielding a squared-advantage term and thereby ensuring a nonnegative local reward change.

The small-step condition guarantees that the AOPD proximal update remains a valid probability distribution. In particular, the update cannot remove more probability mass from a negative-advantage action than the current policy assigns to that action. Appendix B provides the exact proximal solutions and the associated condition on η.

Corollary 5.2 (Surpassing the teacher). Suppose $S = T$ . For a sufficiently small $\eta > 0 ,$ we have ${ \mathcal R } ( S _ { \mathrm { A O P D } } ^ { + } ) \geq$ $\mathcal { R } ( T )$ . Therefore, AOPD can still improve the policy and surpass the teacher. Hence, the teacher acts as a source ofaction-level supervision rather than afixed target.

Reward improvement with clipping. The proof of Theorem 5.1 does not require the teacher weights to form a normalized distribution. It only requires them to be nonnegative and that the proximal coefficients remains positive. The same argument therefore applies to the truncated weights $\widetilde { T } ( a \mid h ) \dot { = } \operatorname* { m i n } \{ T ( a \mid h ) , c S ( a \mid h ) \}$

Corollary 5.3 (Reward improvement with clipping). Under the assumptions ofTheorem 5.1, let $S _ { c } ^ { + }$ be the exact proximal updatefor the expected clipped objective in Equation (3.8). Let $c \geq 1$ and $0 < \eta < 1 / c ,$ we have: $\mathcal { R } ( S _ { c } ^ { + } ) \geq \mathcal { R } ( S )$

Adding RL does not resolve the imitation conflict. An alternative way to incorporate terminal rewards into distillation is to augment the FKL distillation objective with an explicit reward-maximization term:

$$
J _ { \mathrm { m i x } } ( \pi ) = \frac { 1 } { H } \mathcal { R } ( \pi ) + \lambda J _ { \mathrm { F K L } } ( \pi ) , \qquad \lambda > 0\tag{5.4}
$$

Where H is the horizon and is added to simply puts reward and FKL on the same per-token scale. This objective is studied in previous work such KDRL (Xu et al., 2025; Zhang et al., 2026b). The reward term favors successful behavior, but the FKL term still moves the student toward the teacher independently of the student advantage. The two objectives can therefore conflict as we show below:

Lemma 5.4 (RL plus FKL-OPD need not improve reward). For every fixed $\lambda > 0$ , there exists a two-step task with binary terminal rewards and full-support policies $S , T$ satisfying $\mathcal { R } ( T ) > \mathcal { R } ( S )$ such that,for all sufficiently small $\eta > 0$

$$
\mathcal { R } ( S _ { \mathrm { m i x } } ^ { + } ) < \mathcal { R } ( S ) , \quad \mathcal { R } ( S _ { \mathrm { O P D } } ^ { + } ) < \mathcal { R } ( S )
$$

Therefore, adding an RL objective does not necessarily mitigate the harmful aspect of imitation.

## 6 Conclusion

We introduced AOPD, an on-policy distillation objective that asks not only what the teacher prefers, but whether that preference actually helps the current student succeed. By weighting teacher actions with their advantage under the student, AOPD reinforces useful guidance and suppresses harmful imitation, while requiring no additional counterfactual rollouts or auxiliary value models. Across external-teacher and selfdistillation settings, AOPD consistently improves over typical distillation and RL baselines, showing that effective distillation should optimize for student success rather than teacher agreement.

## Acknowledgment

This research was supported by Coefficient Giving, EnCORE (the Institute for Emerging CORE Methods in Data Science), and NSF Award No. 2502489 through the Mathematical Foundations of Alignment in Generative Artificial Intelligence (MFAI) program. AJ was supported in part by the Sloan fellowship in mathematics, the NSF Award DMS-2311024, an Amazon Faculty Research Award, an Adobe Faculty Research Award and an iORB grant form USC Marshall School of Business.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246–21263, 2024.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, 2024.

AI-MO. AIMO Validation AIME. https://huggingface.co/datasets/AI-MO/aimo-validation-aime, 2024a. Hugging Face dataset.

AI-MO. AIMO Validation AMC. https://huggingface.co/datasets/AI-MO/aimo-validation-amc, 2024b. Hugging Face dataset.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al. Multipl-e: A scalable and extensible approach to benchmarking neural code generation. arXiv preprint arXiv:2208.08227, 2022.

Chishui Chen, Yaoyou Fan, Te Sun, Yi Yang, Chenghao Sun, Delin Mao, Hongbo Qiao, Zuowei Zhang, Junxi Wang, Chenxing Sun, et al. Look ahead before you distill: Future trajectory validation of teacher guidance for agentic on-policy distillation. arXiv preprint arXiv:2608.01953, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger, Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun, and´ Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. 2026. URL https://arxiv.org/abs/2605.00674.

Kehua Feng, Xinyi Shen, Weijie Wang, Xiang Zhuang, Yuqi Tang, Qiang Zhang, and Keyan Ding. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098, 2024.

Siyi Gu, Jialin Chen, Sophia Zhou, Arman Cohan, and Rex Ying. Rethinking reward supervision: Rubricconditioned self-distillation. arXiv preprint arXiv:2606.19327, 2026.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pages 32694–32717, 2024.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.

Seungyul Han and Youngchul Sung. Dimension-wise importance sampling weight clipping for sample-efficient reinforcement learning. In International conference on machine learning, pages 2586–2595. PMLR, 2019.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Wenjin Hou, Shangpin Peng, Weinong Wang, Zheng Ruan, Yue Zhang, Zhenglin Zhou, Mingqi Gao, Yifei Chen, Kaiqi Wang, Hongming Yang, et al. Uni-opd: Unifying on-policy distillation with a dual-perspective recipe. arXiv preprint arXiv:2605.03677, 2026.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021. doi: 10.3390/app11146421. URL https://www.mdpi.com/2076-3417/11/14/6421.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pages 1317–1327, 2016.

Kaixin Li. Verified taco problems. https://huggingface.co/datasets/likaixin/TACO-verified, 2024. URL https://huggingface.co/datasets/likaixin/TACO-verified.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026a.

Yijiang Li, Bingyang Wang, Yijun Liang, Yunjie Tian, Di Fu, and Nuno Vasconcelos. On-policy self-distillation without any supervision. arXiv preprint arXiv:2608.06296, 2026b.

Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, and Chaowei Xiao. When are teacher tokens reliable? position-weighted on-policy self-distillation for reasoning. arXiv preprint arXiv:2605.21606, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. Advances in neural information processing systems, 29, 2016.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pages 248–260. PMLR, 2022.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

Khouloud Saadi and Di Wang. Validity-calibrated reasoning distillation. arXiv preprint arXiv:2605.04078, 2026.

Mahdi Sabbaghi, George Pappas, Adel Javanmard, and Hamed Hassani. Infosft: Learn more and forget less with information-aware token weighting. arXiv preprint arXiv:2605.14967, 2026a.

Mahdi Sabbaghi, George Pappas, Adel Javanmard, and Hamed Hassani. Robust policy optimization to prevent catastrophic forgetting. arXiv preprint arXiv:2602.08813, 2026b.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026.

Qitai Tan, Zefang Zong, Yang Li, and Peng Chen. Atod: Annealed turn-aware on-policy distillation for multi-turn autonomous agents. arXiv preprint arXiv:2606.27814, 2026.

Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. Mirror descent policy optimization. arXiv preprint arXiv:2005.09814, 2020.

Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. Not all disagreement is learnable: Token teachability in on-policy distillation. arXiv preprint arXiv:2605.26844, 2026.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando De Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224, 2016.

Yuqiao Wen, Zichao Li, Wenyu Du, and Lili Mou. F-divergence minimization for sequence-level knowledge distillation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10817–10834, 2023.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. Kdrl: Post-training reasoning llms via unified knowledge distillation and reinforcement learning. arXiv preprint arXiv:2506.02208, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang, Mingxuan Xia, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Junbo Zhao, et al. Oprd: On-policy representation distillation. arXiv preprint arXiv:2606.06021, 2026.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, and Bo An. Incentivizing llms to self-verify their answers. Advances in Neural Information Processing Systems, 38:4800–4829, 2026a.

Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang, Zhuowen Tu, Wei Xia, and Stefano Soatto. Reinforcement-aware knowledge distillation for llm reasoning. arXiv preprint arXiv:2602.22495, 2026b.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

## A Additional Experiments

Group size effect on OPD and AOPD. For our main experiments in Section 4, we choose group-size $G = 4$ for AOPD and OPD. This is because as Figure 4 shows, increasing the group size leads to only a negligible reward improvement. We also found that this reward improvement does not transfer to better benchmark number. The same effect is also studied for OPD in previous work Hou et al. (2026).

![](images/b44baa8b2277acdf64dd4aebfd1f14d7e402c16aba5f3a150739b83ca256c28c.jpg)

![](images/7d668d6781baf89f29e225cb937e01acd3c9b78acd4084405daf24bbd9a25f66.jpg)  
Figure 4: Increasing the group size does not significantly improve the performance for AOPD. This is while for methods such GRPO, a larger group size is critical.

Clipping avoids performance fluctuations. As explained in Section 3, we clip the importance sampling weights to control the high variance caused by large importance sampling weight, a standard technique in importance-sampling-based optimization (Munos et al., 2016; Wang et al., 2016; Han and Sung, 2019). This effect is visible in Figure 5: without clipping, training exhibits substantially larger fluctuations, whereas clipping leads to a more stable optimization trajectory.

![](images/d4421e84ebcaed75d7dd0877496e6fe6ffea15ff139ecdc696223a9dca38277f.jpg)  
Figure 5: Clipping the importance sampling weight improves the stability and the performance of AOPD.

## B Proofs: Reward Improvement

ProofofLemma 3.1. Fix a visited history $h _ { t } = h$ . The sampled response comes from $S ,$ and b is independent of that response conditional on the prompt. Remember that:

$$
\mathbb { E } [ R ^ { S } \mid h , a _ { t } = a ] = Q ^ { S } ( h , a ) , \qquad \mathbb { E } [ R ^ { S } \mid h ] = V ^ { S } ( h ) .
$$

Therefore we have:

$$
\begin{array} { l } { { \mathbb { E } [ ( R ^ { S } - b ) w _ { t } \log S _ { \theta } ( a _ { t } \mid h ) \mid h ] = \mathbb { E } \left[ \mathbb { E } [ ( R ^ { S } - b ) w _ { t } \log S _ { \theta } ( a _ { t } \mid h ) \mid h , a ] \right] } } \\ { \ ~ = \mathbb { E } \left[ ( Q ^ { S } ( h , a ) - b ) w _ { t } \log S _ { \theta } ( a _ { t } \mid h ) \right] } \\ { \ ~ = \displaystyle \sum _ { a \in \mathcal { V } } S ( a \mid h ) \big ( Q ^ { S } ( h , a ) - b \big ) \frac { T ( a \mid h ) } { S ( a \mid h ) } \log S _ { \theta } ( a \mid h ) } \\ { \ ~ = \displaystyle \sum _ { a \in \mathcal { V } } T ( a \mid h ) \big ( Q ^ { S } ( h , a ) - b \big ) \log S _ { \theta } ( a \mid h ) } \end{array}
$$

where the support assumption $( T \ll S )$ lets us apply the Radon-Nikodym theorem in the last line. Meanwhile, $\mathbb { E } _ { a \sim T } [ \log S _ { \theta } ( a | h _ { t } ) ]$ ] depends only on the history and in independent of a, so:

$$
\begin{array} { r } { \mathbb { E } \left[ \left( R ^ { S } - b \right) \mathbb { E } _ { a \sim T } [ \log S _ { \theta } ( a \mid h _ { t } ) ] \mid h \right] = \left( V ^ { S } ( h ) - b \right) \mathbb { E } _ { a \sim T } [ \log S _ { \theta } ( a \mid h _ { t } ) ] } \\ { = \mathbb { E } _ { a \sim T } \left[ \left( V ^ { S } ( h ) - b \right) \log S _ { \theta } ( a \mid h _ { t } ) \right] } \end{array}
$$

Subtracting the two expressions cancels the baseline:

$$
\begin{array} { l } { { \mathbb { E } \left[ ( R ^ { S } - b ) \Big ( w _ { t } \log S _ { \theta } ( a _ { t } \mid h ) - { \mathbb { E } _ { a \sim T } } [ \log S _ { \theta } ( a \mid h _ { t } ) ] \Big ) ~ | ~ h , b \right] } } \\ { { \mathrm { } = \displaystyle \sum _ { a \in \mathcal { V } } T ( a \mid h ) \big ( Q ^ { S } ( h , a ) - V ^ { S } ( h ) \big ) \log S _ { \theta } ( a \mid h ) } } \end{array}
$$

Averaging over the baseline, histories, and token positions yields $J _ { \mathrm { A O P D } } ( \theta )$ . Differentiating this identity under the stated conditions gives the gradient result since the expectation and the gradient operations can be swapped. □

Proof of Theorem 5.1. We first prove Equations (5.2) and (5.3). Then we show a general identity is true: $\mathbb { E } _ { a \sim \pi } [ \tilde { A } ^ { S } ( h , a ) ] = \mathcal { R } ( \pi ) - \mathcal { R } ( \bar { S } )$ , and use it to prove the final statement.

Forward-KL. Recall that FKL optimizes the objective in Equation (3.3). Replacing the updated student $S _ { \theta }$ by the candidate policy π, the proximal objective in Equation (5.1) becomes

$$
\begin{array} { l } { { \displaystyle \int _ { \mathrm { F K L } } ( \pi ) - \frac { 1 } { \eta } \mathrm { K L } ( \mathcal { S } \| \pi ) = \mathrm { E } _ { x } [ \displaystyle \frac { 1 } { H } \displaystyle \sum _ { t = 0 } ^ { I - 1 } \mathbb { E } _ { h _ { t } \sim { S } ( \cdot ) } [ \displaystyle \sum _ { a \in \mathcal { V } } T ( a \mid h _ { t } ) \log \pi ( a \mid h _ { t } )   } } \\ { { \displaystyle  \qquad - \frac { 1 } { \eta } \displaystyle \sum _ { a \in \mathcal { V } } S ( a \mid h _ { t } ) \log \frac { S ( a \mid h _ { t } ) } { \pi ( a \mid h _ { t } ) } ] ] } } \\ { { \displaystyle = \mathrm { E } _ { x } [ \displaystyle \frac { 1 } { H } \displaystyle \sum _ { i = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim { S } ( \cdot ) } [ \displaystyle \sum _ { a \in \mathcal { V } } ( T ( a \mid h _ { t } ) + \displaystyle \frac { 1 } { \eta } S ( a \mid h _ { t } ) ) \log \pi ( a \mid h _ { t } )   } } \\ { { \displaystyle  \qquad - \frac { 1 } { \eta } \displaystyle \sum _ { a \in \mathcal { V } } S ( a \mid h _ { t } ) \log S ( a \mid h _ { t } ) ] ] . } } \end{array}
$$

We can drop the last term since it is independent of $\pi .$ . Then, this can be optimized for each h separately. Therefore, fixing a history $h ,$ the relevant objective is:

$$
{ \frac { 1 } { \eta } } \sum _ { a \in \mathcal { V } } \left[ S ( a \mid h ) + \eta T ( a \mid h ) \right] \log \pi ( a \mid h )\tag{B.1}
$$

The optimization separates across histories. Since:

$$
\sum _ { a \in \mathcal { V } } \left[ S ( a \mid h ) + \eta T ( a \mid h ) \right] = 1 + \eta
$$

maximizing over $\pi ( \cdot \mid h )$ subject to $\textstyle \sum _ { a } \pi ( a \mid h ) = 1$ gives:

$$
S _ { \mathrm { F K L } } ^ { + } ( a \mid h ) = \frac { S ( a \mid h ) + \eta T ( a \mid h ) } { 1 + \eta } .
$$

By definition of the student advantage:

$$
\mathbb { E } _ { a \sim S ( \cdot | h ) } [ A ^ { S } ( h , a ) ] = \mathbb { E } _ { a \sim S } [ Q ^ { S } ( h , a ) ] - V ^ { S } ( h ) = 0 .
$$

Therefore:

$$
\begin{array} { r l } & { \mathbb { E } _ { a \sim S _ { \mathrm { F K L } } ^ { + } ( \cdot | h ) } [ A ^ { S } ( h , a ) ] = \frac { \mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] } { 1 + \eta } } \\ & { \quad \quad \quad = \frac { \eta } { 1 + \eta } \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] . } \end{array}
$$

Thus, the reward effect of the FKL update depends on the teacher’s average advantage under the current student, which can have either sign.

AOPD. For AOPD, substituting Equation (3.5) into the same proximal objective gives

$$
\begin{array} { r l } & { J _ { \mathrm { A O P D } } ( \pi ) - \displaystyle \frac { 1 } { \eta } \mathrm { K L } ( S \| \pi ) = \mathbb { E } _ { x } \left[ \frac { 1 } { H } \displaystyle \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \displaystyle \sum _ { a \in \mathcal { V } } T ( a \mid h _ { t } ) A ^ { S } ( h _ { t } , a ) \log \pi ( a \mid h _ { t } ) \right. \right. } \\ & { ~ \displaystyle \left. \left. - \frac { 1 } { \eta } \displaystyle \sum _ { a \in \mathcal { V } } S ( a \mid h _ { t } ) \log \frac { S ( a \mid h _ { t } ) } { \pi ( a \mid h _ { t } ) } \right] \right] . } \end{array}
$$

Similar to Equation (B.1), dropping terms independent of $\pi ,$ the objective at a fixed history h becomes:

$$
{ \frac { 1 } { \eta } } \sum _ { a \in \mathcal { V } } \left[ S ( a \mid h ) + \eta T ( a \mid h ) A ^ { S } ( h , a ) \right] \log \pi ( a \mid h ) .
$$

For sufficiently small $\eta ,$ all coefficients are positive. Their sum is

$$
\begin{array} { l } { { { \displaystyle \sum _ { a \in \mathcal { V } } \left[ S ( a \mid h ) + \eta T ( a \mid h ) A ^ { S } ( h , a ) \right] = 1 + \eta \sum _ { a \in \mathcal { V } } T ( a \mid h ) A ^ { S } ( h , a ) } } } \\ { { { } } } \\ { { { = 1 + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] } . } } \end{array}
$$

Hence the proximal optimizer is

$$
S _ { \mathrm { A O P D } } ^ { + } ( a \mid h ) = \frac { S ( a \mid h ) + \eta T ( a \mid h ) A ^ { S } ( h , a ) } { 1 + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] } .\tag{B.2}
$$

Its expected student advantage is therefore

$$
\begin{array} { r l } { \mathbb { E } _ { a \sim S _ { \mathrm { A O P D } } ^ { + } ( \cdot | h ) } [ A ^ { S } ( h , a ) ] = \frac { \mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ^ { 2 } ] } { 1 + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] } } \\ { = \frac { \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ^ { 2 } ] } { 1 + \eta \mathbb { E } _ { a \sim T } [ A ^ { S } ( h , a ) ] } \ge 0 . } \end{array}
$$

The denominator is positive by the small-step condition, while the numerator is nonnegative because it is an expectation of squared advantages.

(ii) From local advantage to final reward. We finally show that the nonnegative advantage at each history implies a nonnegative change in the final task reward. Recall that $A ^ { S }$ is defined using continuations under the original student S, but it can be evaluated at any history. This is because we assumed S has full support and the sufficiently small proximal update preserves this support. For any history-action pair:

$$
Q ^ { S } ( h _ { t } , a _ { t } ) = V ^ { S } ( h _ { t + 1 } ) ,
$$

where $V ^ { S } ( h _ { H } ) = R ( x , y )$ at the terminal history. Thus, along any complete response, regardless of the policy that generated it,

$$
\begin{array} { l } { { \displaystyle \sum _ { t = 0 } ^ { H - 1 } A ^ { S } ( h _ { t } , a _ { t } ) = \sum _ { t = 0 } ^ { H - 1 } \left( Q ^ { S } ( h _ { t } , a _ { t } ) - V ^ { S } ( h _ { t } ) \right) } } \\ { { \displaystyle \qquad = \sum _ { t = 0 } ^ { H - 1 } \left( V ^ { S } ( h _ { t + 1 } ) - V ^ { S } ( h _ { t } ) \right) } } \\ { { \displaystyle \qquad = R ( x , y ) - V ^ { S } ( x ) . } } \end{array}
$$

Taking the expectation over responses generated by any policy π, and using

$$
\mathbb { E } _ { x } [ V ^ { S } ( x ) ] = \mathcal { R } ( S ) ,
$$

gives

$$
\begin{array} { r l } & { \mathcal { R } ( \pi ) - \mathcal { R } ( S ) = \mathbb { E } _ { x , y \sim \pi } \left[ \displaystyle \sum _ { t = 0 } ^ { H - 1 } A ^ { S } ( h _ { t } , a _ { t } ) \right] } \\ & { \quad \quad \quad = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { h _ { t } \sim \pi ( \cdot ) } \left[ \mathbb { E } _ { a \sim \pi ( \cdot | h _ { t } ) } [ A ^ { S } ( h _ { t } , a ) ] \right] . } \end{array}
$$

Setting $\pi = S _ { \mathrm { A O P D } } ^ { + }$ , every inner expectation is nonnegative by the local result above. Therefore,

$$
\begin{array} { r } { \mathcal { R } ( S _ { \mathrm { A O P D } } ^ { + } ) \geq \mathcal { R } ( S ) . } \end{array}
$$

The inequality is strict whenever $S _ { \mathrm { A O P D } } ^ { + }$ reaches with positive probability a history h for which $T ( a \mid h ) > 0$ and $A ^ { S } ( h , a ) \neq 0$ for some action a.

Proof of Corollary 5.3. Replacing T by $\widetilde { T }$ in the AOPD proximal objective (see (B.2)) gives

$$
S _ { c } ^ { + } ( a \mid h ) = \frac { S ( a \mid h ) + \eta \widetilde { T } ( a \mid h ) A ^ { S } ( h , a ) } { 1 + \eta \sum _ { b } \widetilde { T } ( b \mid h ) A ^ { S } ( h , b ) } .
$$

Since $R \in [ 0 , 1 ]$ , we have $A ^ { S } ( h , a ) \in [ - 1 , 1 ]$ . Moreover,

$$
{ \widetilde { T } } ( a \mid h ) \leq c S ( a \mid h ) ,
$$

so for $\eta < 1 / c$

$$
S ( a \mid h ) + \eta \widetilde { T } ( a \mid h ) A ^ { S } ( h , a ) \ge S ( a \mid h ) ( 1 - \eta c ) > 0 .
$$

Therefore the update is well defined. Using $\mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] = 0 .$

$$
\mathbb { E } _ { a \sim S _ { c } ^ { + } } [ A ^ { S } ( h , a ) ] = \frac { \eta \sum _ { a } \widetilde { T } ( a  { \lvert { \ h } \rvert } { h } ) A ^ { S } ( h , a ) ^ { 2 } } { 1 + \eta \sum _ { a } \widetilde { T } ( a  { \lvert { \ h } \rvert } { h } ) A ^ { S } ( h , a ) } \geq 0 .
$$

Applying the same performance-difference argument as in Theorem 5.1 gives

$$
\mathcal { R } ( S _ { c } ^ { + } ) \geq \mathcal { R } ( S ) .
$$

## B.1 Surpassing the teacher

Proof of Corollary 5.2. Setting $S = T$ in Theorem 5.1 gives

$$
\mathcal { R } ( S _ { \mathrm { A O P D } } ^ { + } ) \geq \mathcal { R } ( S ) = \mathcal { R } ( T ) .
$$

The strict-improvement condition follows directly from the theorem. Meanwhile, the FKL update becomes

$$
S _ { \mathrm { F K L } } ^ { + } = \frac { T + \eta T } { 1 + \eta } = T .
$$

This follows directly from Theorem 5.1 by setting $S = T$ . In contrast, FKL makes no update once the student matches the teacher. AOPD can therefore continue improving beyond the teacher by reweighting its actions according to their task advantage.

## B.2 Adding RL does not guarantee improvement

Proof of Lemma 5.4. In order to prove that FKL + RL (and FKL) can reduce the reward after the update, we first expand the updated policy pi for infinitesimal η. Under this setting, we write down the change in the student reward. We then construct a counter-example that the update reward is smaller, even though the teacher is stronger itself and has a high reward.

The hybrid update. Recall that

$$
J _ { \mathrm { m i x } } ( \pi ) = \frac { 1 } { H } \mathcal { R } ( \pi ) + \lambda J _ { \mathrm { F K L } } ( \pi ) .
$$

Its proximal update therefore maximizes

$$
\frac { 1 } { H } \mathcal { R } ( \pi ) + \lambda J _ { \mathrm { F K L } } ( \pi ) - \frac { 1 } { \eta } \mathrm { K L } ( S \| \pi ) .
$$

For sufficiently small η, the KL penalty keeps the updated policy close to $S ,$ so we write

$$
\pi ( a \mid h ) = S ( a \mid h ) + \eta v ( a \mid h ) + O ( \eta ^ { 2 } ) , \sum _ { a } v ( a \mid h ) = 0 .
$$

We now expand the three terms around S. From the performance-difference identity and $\mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] = 0 \mathrm { . }$

$$
\frac { 1 } { H } \bigl ( \mathcal { R } ( \pi ) - \mathcal { R } ( S ) \bigr ) = \frac { \eta } { H } \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \sum _ { a } v ( a \mid h _ { t } ) A ^ { S } ( h _ { t } , a ) \right] + O ( \eta ^ { 2 } )
$$

Similarly, using log $( 1 + z ) = z + O ( z ^ { 2 } )$

$$
\begin{array} { l } { \displaystyle { J _ { \mathrm { F K L } } ( \pi ) - J _ { \mathrm { F K L } } ( S ) = \frac { \eta } { H } \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \sum _ { a } \frac { T ( a \mid h _ { t } ) } { S ( a \mid h _ { t } ) } v ( a \mid h _ { t } ) \right] + O ( \eta ^ { 2 } ) , } } \\ { \displaystyle { \mathrm { K L } ( S | | \pi ) = \frac { \eta ^ { 2 } } { 2 H } \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \sum _ { a } \frac { v ( a \mid h _ { t } ) ^ { 2 } } { S ( a \mid h _ { t } ) } \right] + O ( \eta ^ { 3 } ) , } } \end{array}
$$

where the first-order KL term vanishes because $\Sigma _ { a } v ( a \mid h ) = 0 .$ . Therefore, at each history h, the leading-order direction maximizes:

$$
\sum _ { a } v ( a \mid h ) A ^ { S } ( h , a ) + \lambda \sum _ { a } { \frac { T ( a \mid h ) } { S ( a \mid h ) } } v ( a \mid h ) - { \frac { 1 } { 2 } } \sum _ { a } { \frac { v ( a \mid h ) ^ { 2 } } { S ( a \mid h ) } }
$$

subject to $\textstyle \sum _ { a } v ( a \mid h ) = 0$ . Introducing a multiplier ν, the first-order condition gives:

$$
A ^ { S } ( h , a ) + \lambda \frac { T ( a \mid h ) } { S ( a \mid h ) } - \frac { v ( a \mid h ) } { S ( a \mid h ) } - \nu = 0 .
$$

Summing over a and using $\mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] = 0$ and $\textstyle \sum _ { a } S ( a \mid h ) = \sum _ { a } T ( a \mid h ) = 1$ gives $\nu = \lambda$ . Hence

$$
v ( a \mid h ) = S ( a \mid h ) A ^ { S } ( h , a ) + \lambda \big ( T ( a \mid h ) - S ( a \mid h ) \big ) .\tag{B.3}
$$

The first term is the reward-improving direction, while the second moves the student toward the teacher.

Effect on the student’s reward. Substituting Equation (B.3) into the first-order reward expansion gives

$$
\begin{array} { l } { \displaystyle \mathcal { R } ( S _ { \mathrm { m i x } } ^ { + } ) - \mathcal { R } ( S ) = \eta \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \sum _ { a } S ( a \mid h _ { t } ) A ^ { S } ( h _ { t } , a ) ^ { 2 } + \lambda \sum _ { a } ( T ( a \mid h _ { t } ) - S ( a \mid h _ { t } ) ) A ^ { S } ( h _ { t } , a ) \right] + O ( \eta ^ { 2 } ) } \\ { \displaystyle \qquad = \eta \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \mathbb { E } _ { a \sim S } [ A ^ { S } ( h _ { t } , a ) ^ { 2 } ] + \lambda \mathbb { E } _ { a \sim T } [ A ^ { S } ( h _ { t } , a ) ] \right] + O ( \eta ^ { 2 } ) , } \end{array}
$$

where the second equality again uses $\mathbb { E } _ { a \sim S } [ A ^ { S } ( h , a ) ] = 0$ . Thus,

$$
\mathcal { R } ( S _ { \mathrm { m i x } } ^ { + } ) - \mathcal { R } ( S ) = \eta \sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \left[ \mathbb { E } _ { a \sim S } [ A ^ { S } ( h _ { t } , a ) ^ { 2 } ] + \lambda \mathbb { E } _ { a \sim T } [ A ^ { S } ( h _ { t } , a ) ] \right] + O ( \eta ^ { 2 } ) .\tag{B.4}
$$

The RL contribution is always nonnegative, but the FKL contribution can be negative.

A two-step counterexample. Consider a two-step task where the first action chooses a "safe" or "difficult" route and the second action determines success or failure. Let $0 < \varepsilon < 1 / 3$ . Define the student by:

$$
S ( \mathrm { d i f f i c u l t } ) = \varepsilon , \qquad S ( \mathrm { s u c c e s s } \mid \mathrm { d i f f i c u l t } ) = \varepsilon , \qquad S ( \mathrm { s u c c e s s } \mid \mathrm { s a f e } ) = 1 - \varepsilon
$$

and the teacher by:

$$
T ( \mathrm { d i f f i c u l t } ) = 1 - \varepsilon , \qquad T ( \mathrm { s u c c e s s } \mid \mathrm { d i f f i c u l t } ) = T ( \mathrm { s u c c e s s } \mid \mathrm { s a f e } ) = 1 - \varepsilon
$$

The expected rewards for the student and the teacher are:

$$
\mathcal { R } ( S ) = \varepsilon ^ { 2 } + ( 1 - \varepsilon ) ^ { 2 } = 1 - 2 \varepsilon + 2 \varepsilon ^ { 2 } , \qquad \mathcal { R } ( T ) = 1 - \varepsilon ,
$$

so

$$
\mathcal { R } ( T ) - \mathcal { R } ( S ) = \varepsilon ( 1 - 2 \varepsilon ) > 0 .
$$

Thus, the teacher is strictly better. At the initial history h<sub>0</sub>:

$$
Q ^ { S } ( h _ { 0 } , \mathrm { d i f f i c u l t } ) = \varepsilon , \qquad Q ^ { S } ( h _ { 0 } , \mathrm { s a f e } ) = 1 - \varepsilon ,
$$

and therefore

$$
A ^ { S } ( h _ { 0 } , \mathrm { d i f f i c u l t } ) = - ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) , \qquad A ^ { S } ( h _ { 0 } , { \mathrm { s a f e } } ) = \varepsilon ( 1 - 2 \varepsilon ) .
$$

Hence

$$
\mathbb { E } _ { a \sim T } [ A ^ { S } ( h _ { 0 } , a ) ] = - ( 1 - 2 \varepsilon ) ^ { 2 } .
$$

At the difficult-route history,

$$
\mathbb { E } _ { a \sim T } [ A ^ { S } ( h _ { \mathrm { d i f f i c u l t } } , a ) ] = 1 - 2 \varepsilon ,
$$

and this history is reached by $S$ with probability ε. At the safe-route history, teacher and student have the same success probability, so the expected teacher advantage is zero. Therefore:

$$
\sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \mathbb { E } _ { a \sim T } [ A ^ { S } ( h _ { t } , a ) ] = - ( 1 - 2 \varepsilon ) ^ { 2 } + \varepsilon ( 1 - 2 \varepsilon ) = ( 1 - 2 \varepsilon ) ( 3 \varepsilon - 1 ) .\tag{B.5}
$$

For the squared-advantage term, the initial history contributes

$$
\mathbb { E } _ { a \sim S } [ A ^ { S } ( h _ { 0 } , a ) ^ { 2 } ] = \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } .
$$

The two second-step histories together contribute $\varepsilon ( 1 - \varepsilon )$ . Thus,

$$
\sum _ { t } \mathbb { E } _ { h _ { t } \sim S ( \cdot ) } \mathbb { E } _ { a \sim S } [ A ^ { S } ( h _ { t } , a ) ^ { 2 } ] = \varepsilon ( 1 - \varepsilon ) \left[ 1 + ( 1 - 2 \varepsilon ) ^ { 2 } \right] .\tag{B.6}
$$

Substituting Equations (B.5) and (B.6) into Equation (B.4), the coefficient of $\eta$ is

$$
\varepsilon ( 1 - \varepsilon ) \left[ 1 + ( 1 - 2 \varepsilon ) ^ { 2 } \right] + \lambda ( 1 - 2 \varepsilon ) ( 3 \varepsilon - 1 ) .
$$

As $\varepsilon  0$ , this converges to $- \lambda < 0$ . Therefore, for every fixed $\lambda > 0$ , we can choose a sufficiently small $\varepsilon > 0$ such that the linear reward change is negative. Holding this task fixed, the negative $O ( \eta )$ term dominates the $O ( \eta ^ { 2 } )$ remainder for sufficiently small $\eta > 0 \mathrm { { ; } }$ , giving

$$
\mathcal { R } ( S _ { \mathrm { m i x } } ^ { + } ) < \mathcal { R } ( S ) .
$$