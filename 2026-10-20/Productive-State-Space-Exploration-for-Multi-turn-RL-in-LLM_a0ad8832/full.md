# Productive State-Space Exploration for Multi-turn RL in LLM Agents

Mohammed Haroon Dupty<sup>†</sup> Institute of Advanced Computing and Intelligence, A\*STAR haroon@u.nus.edu

Song Guo The Hong Kong University of Science and Technology songguo@cse.ust.hk

## Abstract

Reinforcement Learning (RL) for Large Language Model (LLM) agents in multi-turn, longhorizon environments remains challenging due to inefficient action selection and repetitive be havioral patterns. In particular, agents often become trapped in high-confidence yet unproductive behaviors, repeatedly executing invalid actions or entering action loops that hinder task completion. We observe that these frequently violate environment preconditions, revealing a mismatch between the agent’s representation of the internal state and the dynamics of the underlying environment. In addition, repetitive high-confidence actions lead to cyclical behaviors that provide little to no progress towards the goal. We argue that addressing these challenges requires more productive task-aligned exploration to improve the agent’s understanding of environment dynamics. Motivated by intrinsic curiosity-driven exploration, we propose a conditional state-space exploration framework that balances task completion with intrinsic information-seeking behavior. Specifically, we introduce an intrinsic reward that penalizes actions whose observation duplicates prior state memory. This provides the benefit of reducing redundant behaviors and encouraging state-space coverage without relying on a general information-theoretic measure. Strong experimental results demonstrate that our ap proach substantially improves policy learning in multi-turn environments, mitigating invalid actions and repetitive loops.

## 1 Introduction

Reinforcement learning (RL) has become an important training paradigm for large language model (LLM) agents that must operate over multiple turns in partially observable, long-horizon environments (Carta et al., 2023; Shani et al., 2024; Guan et al., 2026). In contrast to single-step prediction

Yong Liang Goh Grabtaxi Holdings Pte Ltd gyl@u.nus.edu

Wee Sun Lee School of Computing, National University of Singapore leews@comp.nus.edu.sg

tasks, these agents must select actions sequentially, update internal memory, and adapt to changing states while receiving sparse and delayed feedback. Recent work has shown that RL is increasingly relevant for improving reasoning, agentic control, and long-horizon decision making in LLMs, but it has also revealed a persistent difficulty: agents often become stuck in repetitive, high-confidence behaviors that are locally plausible yet unproductive for the task (Liu et al., 2025; Zhang et al., 2025; Xi et al., 2025).

In such settings, effective learning depends not only on exploiting promising actions but also on exploring the environment in a way that improves the agent’s understanding of its dynamics. In practice, agents often overcommit to high-confidence actions that are locally plausible but globally unhelpful, leading to invalid interactions or repetitive loops that consume budget without advancing the task. Such failures, as shown in Figure 1, typically reflect a mismatch between the agent’s internal state representation and the true environment dynamics, especially when the agent lacks sufficient feedback about which actions are feasible or informative. This motivates the need for a form of productive state-space exploration that helps the agent learn environment dynamics while remaining aligned with task completion.

A natural way to improve exploration is to encourage the agent to visit a broader range of states. In principle, increasing the entropy of the explored state space can improve coverage, reduce mode collapse, and expose the agent to more informative trajectories. However, directly maximizing exploration entropy can also be problematic. If encouraged too strongly, the agent may drift toward diverse but task-irrelevant behavior, thereby sacrificing progress on the underlying objective. In long-horizon environments, this tension between exploration and exploitation is especially important: exploration should broaden the agent’s experience, but it should not pull the policy away from useful action sequences that lead to task completion.

![](images/f2bef92af139e460e4a7f4c00b8da3eccfe51141906c44688375b5371a83080f.jpg)

![](images/07fe6d2c50aa823c85ababba136a2dd685f0cfbecacbb195cf181c002ccc0262.jpg)  
Figure 1: Failure modes of an RL-trained LLM agent in ALFWorld. (a) Repetitive action loop. The agent repeats an invalid action and receives the same failure feedback until the step budget runs out. (b) Repetitive reasoning loop. The agent’s reasoning ignores the current observation, and the agent cycles between the fridge and the microwave without completing the task. In both cases the agent’s internal state diverges from the environment dynamics, and the looping steps consume interaction budget while revealing no new observation. These are the steps our intrinsic reward penalizes.

This paper pursues productive exploration through intrinsic curiosity-driven rewards, with a conservative formulation that preserves task alignment. Rather than positively rewarding state-space entropy increase, we penalize any step whose observation has already appeared in the episode. The intuition is that unhelpful actions, such as invalid steps or redundant repetitions, consume interaction budget and policy probability without revealing anything new about the environment. They can be detected simply by checking whether each new observation adds any new information to the agent’s memory.

Our formulation reframes entropy increase as a consequence of avoiding unproductive actions rather than as an objective to be optimized directly. The penalty improves within-episode statespace coverage, while the task reward keeps exploration tied to task completion. We further formalize this argument by showing that the cumulative penalty measures how many distinct observations an episode visits, and that it bounds the episode’s visitation entropy (Section 3). A single-step analysis further shows that novel observations always increase entropy, while revisits of frequent observations decrease it.

Our approach is consistent with the broader literature on intrinsic motivation in reinforcement learning, where curiosity, novelty seeking, and uncertainty reduction have been used to improve learning under sparse rewards (Chentanez et al., 2004; Aubret et al., 2019; Gao et al., 2025). It also aligns with recent efforts to train long-horizon LLM agents, where robustness depends not only on reasoning ability but also on the capacity to avoid repetitive or uninformative interactions with the environment (Kalyan and Andrews, 2025). Within this setting, penalizing unhelpful actions provides a simple compromise between unrestricted entropy maximization and overly conservative exploitation. The resulting behavior promotes broader state coverage without losing contact with the task.

In summary, we propose a conditional intrinsic reward framework that supports productive exploration by penalizing actions that yield no new informative observation. Our key argument is that entropy expansion is useful, but only when it is induced indirectly through the suppression of redundant or uninformative behavior. This allows the agent to explore more broadly while maintaining a strong coupling to task progress. We evaluate our proposed exploration strategy on two prominent multi-turn benchmark environments: ALFWorld and WebShop. Empirical results show that our approach surpasses recent baselines in performance metrics of success rates and significantly mitigates both invalid and repetitive action rates.

## 2 Related Work

## 2.1 Prompting-Based LLM Agents

Early LLM agent frameworks rely on prompting for multi-step reasoning and action selection without updating model weights. ReAct (Yao et al., 2023) interleaves reasoning (Thought) and actions (Act), outperforming pure reasoning or action prompting on tasks such as ALFWorld and WebShop. Reflexion (Shinn et al., 2023) extends this with verbal reinforcement, storing self-reflections from failed episodes in episodic memory to guide future decisions. Although ReAct and Reflexion improve performance on multi-turn tasks, both methods depend heavily on self-evaluation quality, which degrades for smaller LLMs, and cannot structurally suppress invalid actions since prompting alone cannot alter action priors.

## 2.2 Reinforcement Learning for LLM Agents

Recently, reinforcement learning (RL) has emerged as a principled approach for post-training LLMs, with growing focus on agent-based LLMs in multiturn environments. PPO (Schulman et al., 2017) and its critic-free variant GRPO (Shao et al., 2024), which estimates baselines from group rollout scores, are widely adopted for RL-based LLM training. RLOO (Kool et al., 2019; Ahmadian et al., 2024) further reduces variance without a learned critic, and AGILE (Feng et al., 2024) applies PPO to LLM agents using a four-module architecture with proactive expert consultation, significantly outperforming prompting on long-horizon QA tasks. Retrospex (Xiang et al., 2024) combines an offline RL critic with LLM action likelihood through dynamic rescoring, improving performance on ALF-World and WebShop over prompting baselines.

GiGPO (Feng et al., 2025) captures global trajectory quality and group-level effectiveness without additional rollouts, providing rich signals for LLM agent learning. Recently, ProxMO (Fang et al., 2026) presents a framework to address GRPO’s context-agnostic normalization and GiGPO’s discrete state-grouping limitation. They explore the provision of gradient intensities based on task difficulty, and replace hard boundary partitioning with continuous semantic similarity weighting, so as to provide richer reward signals that are less taskagnostic. In totality, these works establish multiple avenues to improve vanilla RL for multi-turn LLM agents, such as stable baselines and supporting tools to provide more information about the world. However, they do not directly address the exploration failure modes we identify in this work. Specifically, we explore the notion of invalid actions that arise from environment precondition mismatches, resulting in action loops driven by high-confidence but unproductive behaviors. Our approach adds a conditional intrinsic reward that explicitly penalizes such behaviors during RL training.

Recent RL methods for multi-turn LLM agents focus on improving credit assignment, exploration, and training stability. ArCHer (Zhou and Zanette, 2024) introduces a hierarchical RL framework combining off-policy temporal difference learning at the utterance level with on-policy policy gradients at the token level, substantially improving sample efficiency over PPO while emphasizing information-seeking behavior in multi-turn settings. Tree-GRPO (Ji et al., 2026) improves training efficiency by replacing chain-based rollouts with tree-structured sampling, enabling step-level supervision from outcome rewards alone. Addressing exploration instability, EPO (Xu et al., 2025) identifies exploration-exploitation cascade failure in shared multi-turn policies and mitigates it via trajectory-level entropy regularization and entropy smoothing, yielding gains on ScienceWorld and ALFWorld. Similarly, RAGEN (Wang et al., 2025) studies multi-turn RL failure modes, identifying an “Echo Trap” caused by reward variance cliffs and gradient spikes, and proposes trajectory filtering with a critic for stabilization. While these works improve how agents learn—through efficient credit propagation, entropy stabilization, or robust optimization—our approach is complementary, focusing on what to explore by penalizing invalid or repetitive actions that fail to expand state-space knowledge, without itself performing credit assignment across turns.

## 2.3 Intrinsic Motivation and Exploration in RL

Intrinsic motivation has long been used to improve exploration under sparse rewards by encouraging state diversity and novelty. Curiosity-based methods like Pathak et al. (2017); Burda et al. (2019) reward novel states or high prediction errors, while count-based exploration formalizes revisitation penalties through information gain via KL-divergence between prior and posterior state beliefs (Bellemare et al., 2016). Extending stateentropy exploration to task-reward settings, Kim et al. (2023) propose value-conditional state entropy (VCSE), addressing the tendency of naive entropy maximization to overexplore low-value regions by maximizing entropy within value partitions. Mutti et al. (2021); Ashlag et al. (2025) further provide a theoretical foundation for state entropy regularization, showing that it improves robustness and promotes broader coverage across multiple viable trajectories, unlike policy entropy which mainly injects noise along a dominant path. These works motivate our approach – by targeting state-space entropy, our conditional intrinsic penalty discourages revisiting already-observed states. Additionally, by tying the penalty to information gain rather than raw visitation counts, we ensure that only transitions that fail to expand the agent’s knowledge are penalized, preserving the pull of the task reward toward goal-relevant states.

In LLM-based RL, intrinsic signals have recently been explored to complement sparse task rewards. MIRA (Nourzad and Joe-Wong, 2026) improves exploration by using memory graphs from agent trajectories and semantic reasoning to shape PPO advantages, mitigating poor early critic calibration. Similarly, our approach leverages accumulated experience for exploration, but instead defines an information-gain-based intrinsic signal that penalizes observations failing to expand state knowledge. This is simpler to compute, and is motivated by Shannon entropy (Proposition 2), while being aligned with repetitive and invalid behaviors in text-based environments. Further recent work on moral intrinsic rewards (Tennant et al., 2025) has shown that intrinsic signals can shape agent behavior toward desired norms without task-specific reward engineering, reinforcing the broader applicability of intrinsic motivation for LLM alignment. Our contribution fits within this emerging direction: intrinsic rewards as a practical mechanism for shaping LLM agent behavior in ways that extrinsic rewards alone cannot achieve.

## 3 Methodology

## 3.1 Problem Setup

We consider reinforcement learning for LLM agents operating in partially observable, longhorizon interactive environments. At each timestep t, the agent is in state $s _ { t } .$ , takes an action $a _ { t } \in { \mathcal { A } } .$ and receives an observation $O t$ from the environment. We define the agent’s state $s _ { t }$ as the set of all observations received so far along the trajectory, i.e., the information the agent has accumulated about the environment up to time t. The objective is to learn a policy $\pi ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t } )$ that maximizes expected cumulative reward while completing the task within a finite horizon $T$

Unlike standard short-horizon decision problems, this setting introduces two recurring failure modes. First, the agent may produce invalid actions whose preconditions are not satisfied, causing the transition to fail or yielding no meaningful progress. Second, the agent may enter action loops, repeatedly selecting a small set of actions that do not improve task state or move the agent closer to completion. Both issues reduce learning efficiency and often reflect an imperfect internal model of the environment dynamics. In practice, these failures are especially harmful in LLM agents because they waste interaction budget and reinforce unproductive behavior patterns.

## 3.2 State Representation

To support long-horizon decision making, we define the agent’s state as the accumulated observation history:

$$
s _ { t } = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { t } \} .
$$

This representation captures the information available to the agent at time t. Instead of relying only on the most recent observation, the agent maintains a memory of the full trajectory of unique observations, enabling it to reason about what has already been explored and what remains uncertain. This accumulated state serves as the basis for evaluating whether a new observation provides additional information about the environment.

At each step, the agent observes the current state $s _ { t } .$ , selects an action $a _ { t } .$ , and receives the next observation $o _ { t + 1 }$ . The state is then updated by appending the new observation:

$$
s _ { t + 1 } = s _ { t } \cup \{ o _ { t + 1 } \} .
$$

The intrinsic reward is computed from the change in information content between $s _ { t }$ and $o _ { t + 1 }$ . If the new observation does not improve the agent’s knowledge of the environment, a negative intrinsic reward is assigned. This signal is combined with the task reward and used to update the policy.

## 3.3 Conditional State-Space Exploration

We propose a conditional state-space exploration mechanism that penalizes unhelpful transitions. After taking action ${ \mathit { a } } _ { t } ,$ the agent receives a new observation $o _ { t + 1 }$ , and the information gain is computed by comparing this new observation with the previous state $s _ { t }$ . Let $I ( \cdot )$ denote a measure of information content or entropy associated with the agent’s accumulated state representation. We define the change in information as

$$
\Delta I ( s _ { t } , o _ { t + 1 } ) = I ( s _ { t } \cup \{ o _ { t + 1 } \} ) - I ( s _ { t } ) .
$$

This quantity measures whether the new observation expands the agent’s knowledge of the environment.

The intrinsic reward is then defined as:

$$
r _ { t } ^ { \mathrm { i n t r i n s i c } } = \left\{ { \begin{array} { l l } { - \gamma } & { { \mathrm { i f } } \Delta I ( s _ { t } , o _ { t + 1 } ) \leq 0 , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{1}
$$

where $\gamma > 0$ controls the penalty magnitude. This formulation penalizes actions that do not lead to a meaningful increase in the agent’s information about the environment, thereby discouraging redundant or cyclic behavior.

The formulation is general, and $I ( \cdot )$ can be any measure of the information in the agent’s memory. In practice, we use a simple instantiation, $I ( s ) = \log | s |$ , with new observations compared by string matching. Then $\Delta I ( s _ { t } , o _ { t + 1 } ) \ \leq \ 0$ iff $o _ { t + 1 } \in s _ { t }$ , and Equation 1 reduces to a duplicateobservation penalty (Algorithm 1). Exact matching suffices in many settings, including ALFWorld, where environment feedback is drawn from a fixed set of templated observations, so identical states produce identical text. It errs only toward underpenalization, since it may miss near-duplicates but never penalizes a genuinely new observation. Unlike similarity-based measures, it also requires no threshold and adds no hyperparameters beyond $\gamma .$ Moreover, since the agent controls actions rather than observation text, it can avoid the penalty only by reaching a different state. More sophisticated measures, such as embedding similarity, are needed only in environments with free-form or noisy observations, where the same state can be described in different ways.

How Does the Penalty Teach Preconditions? Let $\mathcal { F } ( s )$ be the set of actions whose preconditions fail in state s. We assume that (A1) every $a \in { \mathcal { F } } ( s )$ returns the same observation $o _ { \perp }$ , independent of a and s, and (A2) a failed action leaves the state unchanged. Under A1, every precondition violation after the first in an episode yields $o _ { t + 1 } = o _ { \perp } \in \mathfrak { s } _ { t }$ and is penalized, whichever action failed. The policy is thus pushed away from all precondition-violating actions, not only from repeating one. Under A2, avoiding them loses no information, so the penalty does not conflict with information gathering.

Algorithm 1 Computation of the intrinsic reward   
Require: state memory $s _ { t } = \{ o _ { 1 } , . . . , o _ { t } \}$ , new   
observation $^ { O _ { t + 1 } , }$ penalty weight $\gamma$   
1: $\mathbf { i f } \ O _ { t + 1 } \in s _ { t }$ then   
2: $r _ { t } ^ { \mathrm { i n t r i n s i c } }  - \gamma$   
3: $s _ { t + 1 } \gets s _ { t }$   
4: else   
5: $r _ { t } ^ { \mathrm { i n t r i n s i c } }  0$   
6: $s _ { t + 1 } \gets s _ { t } \cup \left\{ o _ { t + 1 } \right\}$   
7: end if   
8: return $r _ { t } ^ { \mathrm { i n t r i n s i c } } , s _ { t + 1 }$

## 3.4 Final Reward

The final reward used for policy optimization combines the task reward and the intrinsic reward:

$$
r _ { t } = r _ { t } ^ { \mathrm { t a s k } } + \lambda r _ { t } ^ { \mathrm { i n t r i n s i c } } ,
$$

where $\lambda \geq 0$ balances task performance and exploration pressure. The resulting objective is

$$
J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } \rho ^ { t } r _ { t } \right] ,
$$

where $\rho \in [ 0 , 1 ]$ is the discount factor. This objective encourages the agent to maximize task reward while avoiding actions that fail to improve statespace coverage.

## 3.5 Entropy-Oriented Exploration Objective

The motivation for this design is that directly maximizing state-space entropy can sometimes distract the agent from the task objective. An overly strong incentive for novelty may lead the policy to explore diverse but irrelevant trajectories. By contrast, penalizing non-informative actions creates a task-constrained pressure toward visiting more distinct observations within an episode. The policy is discouraged from remaining in low-diversity action cycles, but it is still constrained by the external task reward, which ensures that exploration remains useful for task completion.

This conditional structure is especially important in long-horizon environments, where the agent must balance exploration and exploitation over many sequential decisions. Actions that expand state coverage or contribute to task progress receive no intermediate positive incentive. However, actions that actively fail to improve the agent’s state knowledge are penalized. As a result, the learned policy is pushed toward trajectories that are both diverse and productive.

This penalization mechanism encourages productive exploration by discouraging repeated observations and promoting broader coverage of the agent’s experience. It is related to informationtheoretic exploration, which uses visitation entropy to characterize exploration diversity. Below, we formalize this connection through single-step changes in visitation entropy and episode-level coverage.

Let $P _ { t }$ denote the empirical observationvisitation distribution over $o _ { 1 } , \ldots , o _ { t }$ , with $p _ { i } ^ { ( t ) } = $ $c _ { i } ^ { ( t ) } / N$ , where $c _ { i } ^ { ( t ) }$ counts occurrences of the $i -$ th distinct observation, $\begin{array} { r c l } { n } & { = } & { | s _ { t } | } \end{array}$ , and $\begin{array} { r l } { N } & { { } = } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { n } c _ { i } ^ { ( t ) } ~ = ~ t ~ \geq ~ 1$ . Its Shannon entropy is $\begin{array} { r c l } { { { \cal H } ( P _ { t } ) } } & { { = } } & { { - \sum _ { i = 1 } ^ { n } p _ { i } ^ { ( t ) } \log p _ { i } ^ { ( t ) } } } \end{array}$ Unlike the memory-based measure $I ( s _ { t } ) = \log | s _ { t } |$ $H ( P _ { t } )$ accounts for observation frequencies.

Proposition 1. For the distribution P defined above, let $n = n _ { t }$ be the number ofdistinct states observed by time t. States are identified with observations in the memory $s _ { t } .$ . Then,

1. If the new observation introduces a previously unseen state, then $H ( P _ { t + 1 } ) > H ( P _ { t } )$

2. If the new observation repeats an alreadyvisited state k satisfying $p _ { k } ^ { ( \bar { t } ) } \geq e ^ { - H ( P _ { t } ) }$ , then $H ( P _ { t + 1 } ) \leq H ( P _ { t } )$

Proof. Let $\alpha = 1 / ( N + 1 )$ , where $N = t$ is the total count of observations. The update induced by a new observation can be written as

$$
P _ { t + 1 } = ( 1 - \alpha ) P _ { t } + \alpha e ,
$$

where e is a point mass on the observed state.

## Case 1: Introduction of a new state.

Suppose the observation corresponds to a previously unseen state. Then the support expands by one dimension and

$$
P _ { t + 1 } = \big ( ( 1 - \alpha ) P _ { t } , \alpha \big ) .
$$

The entropy of the updated distribution decomposes as

$$
H ( P _ { t + 1 } ) = ( 1 - \alpha ) H ( P _ { t } ) + h ( \alpha ) ,
$$

where $h ( \alpha ) = - ( 1 - \alpha ) \log ( 1 - \alpha ) - \alpha$ log α is the binary entropy term arising from introducing the new state. Therefore,

$$
H ( P _ { t + 1 } ) - H ( P _ { t } ) = h ( \alpha ) - \alpha H ( P _ { t } ) .
$$

Since the support size before the update is at most N,

$$
H ( P _ { t } ) \leq \log N .
$$

Substituting $\alpha = 1 / ( N + 1 )$ gives

$$
\begin{array} { l } { \displaystyle h ( \alpha ) - \alpha H ( P _ { t } ) \geq h ( \alpha ) - \alpha \log N } \\ { \displaystyle = \alpha \log \frac { N + 1 } { N } - ( 1 - \alpha ) \log ( 1 - \alpha ) } \\ { \displaystyle > 0 } \end{array}
$$

Hence $H ( P _ { t + 1 } ) > H ( P _ { t } )$

## Case 2: Repetition of an existing state.

Suppose the observation revisits an already-seen state $k \in \{ 1 , \ldots , n \}$ . Then

$$
P _ { t + 1 } = ( 1 - \alpha ) P _ { t } + \alpha e _ { k } ,
$$

where $e _ { k }$ is the point mass on state k. For $\lambda \in [ 0 , 1 ]$ define

$$
f ( \lambda ) = H ( ( 1 - \lambda ) P _ { t } + \lambda e _ { k } ) .
$$

If $P _ { t } = e _ { k }$ , then $f ( \lambda ) = 0$ for all $\lambda \in [ 0 , 1 ]$ . Otherwise, since Shannon entropy is strictly concave on the simplex and $P _ { t } \neq e _ { k } , f$ is strictly concave in λ. Its right derivative at $\lambda = 0$ is

$$
f ^ { \prime } ( 0 ) = - H ( P _ { t } ) - \log p _ { k } ^ { ( t ) } .
$$

Thus,

$$
f ^ { \prime } ( 0 ) \leq 0 \quad \Longleftrightarrow \quad p _ { k } ^ { ( t ) } \geq e ^ { - H ( P _ { t } ) } .
$$

By concavity, a non-positive initial slope implies

$$
f ( \lambda ) \leq f ( 0 )
$$

for all $\lambda \in ( 0 , 1 ]$ . If $\boldsymbol { P _ { t } } \neq \boldsymbol { e _ { k } }$ , strict concavity gives

$$
f ( \lambda ) < f ( 0 ) \qquad \mathrm { f o r \ a l l } \lambda \in ( 0 , 1 ] .
$$

Taking $\lambda = \alpha$ gives

$$
H ( P _ { t + 1 } ) \leq H ( P _ { t } ) ,
$$

with strict inequality unless $P _ { t } = e _ { k }$

Remark 1. Proposition 1formalizes the intuition underlying our intrinsic penalty at the level of a single visitation update. Introducing novel states strictly increases visitation entropy, whereas repeatedly revisiting sufficiently frequent states reduces entropy and concentrates probability mass. Penalizing repeated observations therefore discourages unproductive revisitation and is directionally aligned with broader intrinsic exploration.

Proposition 2 (Whole-episode coverage and entropy). Consider an episode with observations $o _ { 1 } , \ldots , o _ { L }$ and $D = | \{ 1 \leq t < L : o _ { t + 1 } \in s _ { t } \} |$ | duplicate steps. Then $| s _ { L } | = L - D$ , and the undiscounted cumulative intrinsic reward is

$$
R ^ { i n t } = \sum _ { t = 1 } ^ { L - 1 } r _ { t } ^ { i n t r i n s i c } = - \gamma D = - \gamma ( L - | s _ { L } | ) .
$$

Thus, at fixed episode length $L ,$ maximizing $R ^ { i n t }$ maximizes within-episode coverage $\left| s _ { L } \right|$ . Moreover,

$$
\begin{array} { c } { { H ( P _ { L } ) \leq \log ( L - D ) , } } \\ { { { } } } \\ { { H ( P _ { L } ) \geq \log L - \displaystyle \frac { D + 1 } { L } \log ( D + 1 ) . } } \end{array}
$$

The lower bound is attained, both bounds are decreasing in $D ,$ , and both equal log L if and only if $D = 0 ,$ , equivalently $R ^ { i n t } = 0$

Proof. Coverage. $o _ { 1 }$ is always novel, and each subsequent observation either adds one element to the memory or is a duplicate. Therefore,

$$
| s _ { L } | = 1 + ( L - 1 - D ) = L - D .
$$

By the definition of the intrinsic reward,

$$
R ^ { \mathrm { i n t } } = - \gamma D = - \gamma ( L - | s _ { L } | ) .
$$

Upper bound. Entropy is at most the logarithm of the support size:

$$
\begin{array} { r } { H ( P _ { L } ) \leq \log | s _ { L } | = \log ( L - D ) . } \end{array}
$$

Lower bound. For positive integer counts $c _ { 1 } , \ldots , c _ { L - D }$ summing to L,

$$
H ( P _ { L } ) = \log L - \frac { 1 } { L } \sum _ { i = 1 } ^ { L - D } c _ { i } \log c _ { i } .
$$

The sum $\textstyle \sum _ { i } c _ { i }$ log $c _ { i }$ is maximized when the counts are $( D + 1 , 1 , \ldots , 1 )$ . Indeed, whenever two counts satisfy $a \geq b > 1$ , moving one unit from b to a does not decrease the sum, since x log x is convex.

Repeating this operation leaves all but one count equal to 1. Therefore,

$$
\begin{array} { l } { { \displaystyle { \cal H } ( P _ { L } ) \geq { \cal H } \left( \frac { D + 1 } { L } , \frac { 1 } { L } , \cdots , \frac { 1 } { L } \right) } } \\ { { \displaystyle ~ = \log L - \frac { D + 1 } { L } \log ( D + 1 ) } , } \end{array}
$$

with equality when the count vector is $( D \mathrm { ~ + ~ }$ $1 , 1 , \ldots , 1 )$ , up to permutation.

Monotonicity. The upper bound $\log ( L - D )$ is strictly decreasing in $D .$ . The lower bound is also strictly decreasing, since $\left( D + 1 \right) \log ( D + 1 )$ is strictly increasing for $D \geq 0$ . Both bounds equal log $L$ when $D = 0$ and are strictly below log $L$ when $D \geq 1$ . Since $\gamma > 0 , D = 0$ if and only if $R ^ { \mathrm { i n t } } = 0 .$ □

Remark 2. Proposition 2 establishes that, for a fixed episode length, maximizing the cumulative intrinsic reward is equivalent to maximizing withinepisode coverage. It also bounds the visitation entropy in terms ofthe number ofduplicate steps, showing that fewer duplicates permit higher entropy. However, we do note that maximizing coverage does not necessarily maximize entropy across trajectories, since entropy also depends on the frequency distribution ofvisited observations.

## 4 Experiments

In this section, we report the experimental results of our proposed approach of productive state-space exploration. We evaluate our method against several recent baselines across two interactive agent benchmarks: ALFWorld (Shridhar et al., 2020) and WebShop (Yao et al., 2022). With this, we aim to demonstrate two main benefits of our proposed approach. First, the performance improvements due to the intrinsic reward signals. Second, the exploration behavior and its usefulness in overcoming the repeated actions and looping behavior of RL trained LLM agents.

## 4.1 Baselines

For the performance comparison, we benchmark against several state-of-the-art recent baseline models. These include prompting frameworks and those trained with RL. The prompting baselines are Gemini-2.5 (Team et al., 2023), GPT-4o (Achiam et al., 2023), Qwen-2.5 (Qwen et al., 2025), ReAct (Yao et al., 2023) and Reflexion (Shinn et al., 2023) frameworks. The RL trained baselines include PPO (Schulman et al.,

Table 1: Performance on ALFWorld and WebShop. Following GiGPO, for ALFWorld environment we report the average success rate (%) for each subtask along with the overall result. For WebShop environment, we report the average score and the average success rate (%). Prompting/PPO/RLOO/GRPO/GiGPO rows are quoted from Feng et al. (2025); ProxMO from Fang et al. (2026).
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>All</td><td>Score</td><td>Succ.</td></tr><tr><td colspan="2">Closed-Source Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>GPT-40</td><td>75.3</td><td>60.8</td><td>31.2</td><td>56.7</td><td>21.6</td><td>49.8</td><td>48.0</td><td>31.8</td><td>23.7</td></tr><tr><td>Prompting</td><td>Gemini-2.5-Pro</td><td>92.8</td><td>63.3</td><td>62.1</td><td>69.0</td><td>26.6</td><td>58.7</td><td>60.3</td><td>42.5</td><td>35.9</td></tr><tr><td colspan="2">Qwen2.5-1.5B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>Qwen2.5</td><td>5.9</td><td>5.5</td><td>3.3</td><td>9.7</td><td>4.2</td><td>0.0</td><td>4.1</td><td>23.1</td><td>5.2</td></tr><tr><td>Prompting</td><td>ReAct</td><td>17.4</td><td>20.5</td><td>15.7</td><td>6.2</td><td>7.7</td><td>2.0</td><td>12.8</td><td>40.1</td><td>11.3</td></tr><tr><td>Prompting</td><td>Reflexion</td><td>35.3</td><td>22.2</td><td>21.7</td><td>13.6</td><td>19.4</td><td>3.7</td><td>21.8</td><td>55.8</td><td>21.9</td></tr><tr><td>RL Training</td><td>PPO (with critic)</td><td> $6 4 . 8 _ { ( 3 . 5 ) }$ </td><td> $4 0 . 5 _ { ( 6 . 9 ) }$ </td><td> $5 7 . 1 _ { ( 4 . 9 ) }$ </td><td> $6 0 . 6 _ { ( 6 . 6 ) }$ </td><td> $4 6 . 4 _ { ( 4 . 0 ) }$ </td><td> $4 7 . 4 _ { ( 1 . 9 ) }$ </td><td> $5 4 . 4 _ { ( 3 . 1 ) }$ </td><td> $7 3 . 8 _ { ( 3 . 0 ) }$ </td><td> $5 1 . 5 _ { ( 2 . 9 ) }$ </td></tr><tr><td>RL Training</td><td>RLOO</td><td> $8 8 . 3 _ { ( 3 . 0 ) }$ </td><td> $5 2 . 8 _ { ( 8 . 6 ) }$ </td><td> $7 1 . 0 _ { ( 5 . 9 ) }$ </td><td> $6 2 . 8 _ { ( 8 . 7 ) }$ </td><td> $6 6 . 4 _ { ( 5 . 5 ) }$ </td><td> $5 6 . 9 _ { ( 4 . 7 ) }$ </td><td> $6 9 . 7 _ { ( 2 . 5 ) }$ </td><td> $7 3 . 9 _ { ( 5 . 6 ) }$ </td><td> $5 2 . 1 _ { ( 6 . 7 ) }$ </td></tr><tr><td>RL Training</td><td>GRPO</td><td> $8 5 . 3 _ { ( 1 . 5 ) }$ </td><td> $5 3 . 7 _ { ( 8 . 0 ) }$ </td><td> $8 4 . 5 _ { ( 6 . 8 ) }$ </td><td> $7 8 . 2 _ { ( 7 . 9 ) }$ </td><td> $5 9 . 7 _ { ( 5 . 0 ) }$ </td><td> $5 3 . 5 _ { ( 5 . 6 ) }$ </td><td> $7 2 . 8 _ { ( 3 . 6 ) }$ </td><td> $7 5 . 8 _ { ( 3 . 5 ) }$ </td><td> $5 6 . 8 _ { ( 3 . 8 ) }$ </td></tr><tr><td>RL Training</td><td>GiGPO</td><td> $9 4 . 4 _ { ( 5 . 9 ) }$ </td><td> $6 7 . 5 _ { ( 4 . 6 ) }$ </td><td> $9 4 . 8 _ { ( 3 . 8 ) }$ </td><td> $9 4 . 4 _ { ( 7 . 8 ) }$ </td><td> $7 9 . 8 _ { ( 4 . 7 ) }$ </td><td> $7 6 . 4 _ { ( 5 . 4 ) }$ </td><td> $8 6 . 7 _ { ( 1 . 7 ) }$ </td><td> $8 3 . 1 _ { ( 1 . 6 ) }$ </td><td> $6 5 . 0 _ { ( 3 . 2 ) }$ </td></tr><tr><td>RL Training</td><td>ProxMO</td><td> $9 4 . 3 _ { ( - ) }$ </td><td> $9 2 . 9 _ { ( - ) }$ </td><td> $8 9 . 3 _ { ( - ) }$ </td><td> $9 2 . 2 _ { ( - ) }$ </td><td> ${ \bf 8 9 . 5 } _ { ( - ) }$ </td><td> $8 7 . 0 _ { ( - ) }$ </td><td> $9 0 . 6 _ { ( - ) }$ </td><td> $8 5 . 3 _ { ( - ) }$ </td><td> $6 7 . 1 _ { ( - ) }$ </td></tr><tr><td>RL Training</td><td>OURS (PPO)</td><td> $\mathbf { 9 9 . 5 } _ { ( 0 . 7 ) }$ </td><td> $9 5 . 5 _ { ( 6 . 4 ) }$ </td><td> $\mathbf { 9 9 . 7 } _ { ( 0 . 4 ) }$ </td><td> $\mathbf { 9 7 . 1 } _ { ( 2 . 9 ) }$ </td><td> $8 7 . 4 _ { ( 1 . 6 ) }$ </td><td> $\mathbf { 8 3 . 9 } _ { ( 3 . 7 ) }$ </td><td> $\mathbf { 9 4 . 0 } _ { ( 1 . 2 ) }$  1</td><td> ${ \bf 8 9 . 2 } _ { ( 1 . 1 ) }$  </td><td> $7 8 . 3 _ { ( 1 . 2 ) }$ </td></tr></table>

Table 2: Exploration efficiency on ALFWorld and WebShop. We compare our method with GiGPO using three metrics: mean episode length, invalid action rate, and repetitive action rate. Both GiGPO and our method are run by us under the same setup. Lower values indicate more productive exploration.
<table><tr><td rowspan="2">Method</td><td colspan="3">ALFWorld</td><td colspan="3">WebShop</td></tr><tr><td>Mean Episode Length</td><td>Invalid Action Rate</td><td>Repetitive Action Rate</td><td>Mean Episode Length</td><td>Invalid Action Rate</td><td>Repetitive Action Rate</td></tr><tr><td>GiGPO</td><td>32.27</td><td>4.08</td><td>45.31</td><td>5.53</td><td>0.14</td><td>43.25</td></tr><tr><td>Ours</td><td>22.02</td><td>2.79</td><td>11.53</td><td>5.18</td><td>0.09</td><td>40.83</td></tr></table>

2017), RLOO (Kool et al., 2019; Ahmadian et al., 2024), GRPO (Shao et al., 2024), GiGPO (Feng et al., 2025), ProxMO (Fang et al., 2026). The most recent and relevant baselines for our approach are GiGPO and ProxMO, since we aim to address challenges in multi-turn LLM agent training with RL. We do not re-run these baselines, all Table 1 numbers except our method are quoted from their original publications. For the additional analyses (Table 3, Sections 4.4–4.5 and Appendix A.2), we re-train GiGPO with the latest official verl-agent implementation, without the intrinsic penalty, and evaluate it under exactly the same protocol as our agent.

Equation 1, is set $\textstyle { \frac { 1 } { 4 0 } }$ of the final success reward. Full reproducibility details (γ selection, reward normalization, splits) are in Appendix A.3.

## 4.3 Results

Table 1 compares our proposed method (labelled OURS) against existing prompting-based and reinforcement learning (RL) training approaches on the ALFWorld and WebShop benchmarks. Following prior work, we report the average success rate for each ALFWorld subtask together with the overall success rate. For WebShop, we report both the average task score and the average success rate.

## 4.2 Training Details:

We train Qwen-2.5-1.5B-Instruct model with the proposed state-space exploration rewards along with Proximal Policy Optimization (PPO) based RL training. We run all our experiments on top of verl-agent framework released by GiGPO (Feng et al., 2025). We train for 300 steps with all other hyperparameters kept same as in the framework. We set the maximum steps at 50 for ALFWorld and 15 for WebShop as with GiGPO. The strength of state-space exploration penalty, γ, as defined in

We primarily compare our approach against recent RL optimization methods, namely GiGPO (Group-in-Group Policy Optimization) and ProxMO (Fang et al., 2026), which represent strong baselines for LLM agent training. Compared to GiGPO, our method achieves consistent gains across all ALFWorld subtasks, improving the overall success rate from 86.7% to 94.0%. The improvements are particularly pronounced on challenging long-horizon tasks such as Cool (+7.6%) and Pick2 (+7.5%), indicating stronger sequential reasoning and more effective action planning. On WebShop, our intrinsic-rewards approach improves the average task score from 83.1 to 89.2 and substantially increases the success rate from 65.0% to 78.3%, demonstrating superior capability in complex web-based interaction tasks.

![](images/961b8c995fbaa34f45fa80f23ea06f636a9407a149d4f3b0a45e6cc27d6d6af3.jpg)  
Figure 2: A successful ALFWorld episode of the agent trained with our intrinsic reward, shown in two halves. The agent searches eight locations, revisiting only two, then takes, cleans and places the butterknife in 18 steps without a single invalid action.

Compared to ProxMO, a recent proximity-based multi-turn optimization framework designed for practical credit assignment in LLM agent training, our approach still achieves better overall performance across both benchmarks. Specifically, it improves the overall ALFWorld success rate from 90.6% to 94.0%, while increasing the WebShop success rate from 67.1% to 78.3%. The improvements on ALFWorld are especially significant, suggesting that our method generalizes more effectively in environments requiring long-horizon exploration, recovery from intermediate mistakes, and consistent multi-turn decision making.

Generalization to unseen scenes. To test whether these gains carry over to environments not seen during training, we evaluate our agent and the re-trained GiGPO agent (Section 4.1) on the ALFWorld out-of-distribution split, which contains unseen scenes, and on the in-distribution split under the same protocol (128 episodes each, same games for both agents; Appendix A.2). Since these are single evaluations of one checkpoint per method, the seen-scene numbers differ slightly from Table 1, where the GiGPO result is quoted from its original paper. Table 3 shows that our agent generalizes markedly better. Its success rate is nearly unchanged on unseen scenes, whereas GiGPO’s drops noticeably, so the gap between the two agents roughly doubles, and our agent leads on every task type.

Table 3: Success rate (%) on seen and unseen ALFWorld scenes for our agent and the re-trained GiGPO.
<table><tr><td>Split</td><td>Method</td><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>All</td></tr><tr><td>Seen</td><td>GiGPO</td><td>92.3</td><td>84.6</td><td>92.0</td><td>100.0</td><td>81.2</td><td>83.3</td><td>89.1</td></tr><tr><td></td><td>Ours</td><td>100.0</td><td>100.0</td><td>100.0</td><td>90.9</td><td>81.2</td><td>91.7</td><td>95.3</td></tr><tr><td></td><td>GiGPO</td><td>82.6</td><td>91.3</td><td>79.2</td><td>73.7</td><td>87.0</td><td>81.2</td><td>82.8</td></tr><tr><td>Unseen</td><td>Ours</td><td>95.7</td><td>100.0</td><td>91.7</td><td>89.5</td><td>100.0</td><td>87.5</td><td>94.5</td></tr></table>

## 4.4 Intrinsic Productive Exploration

To look beyond overall success rates, we evaluate the behavioral dynamics of our agent against the competitive baseline GiGPO (Feng et al., 2025). This analysis focuses on intrinsic productive exploration - the agent’s capacity to autonomously discover optimal state-action trajectories through targeted, purposeful environment interactions, rather than wasting computational steps on redundant or unproductive behaviors. We define three behavioral diagnostics across the ALFWorld and WebShop benchmarks to quantify the efficiency and productivity of the agent’s intrinsic exploration strategy:

• Mean Episode Length: The average number of sequential steps to complete a task instance.

• Invalid Action Rate: The frequency of actions that fail (violate action preconditions). It is detected as described in Appendix A.3.

• Repetitive Action Rate: The frequency of steps spent executing identical actions. We consider an action repetitive when either the action or the resulting observation is repeated in the trajectory.

The empirical results detailing these behavioral metrics are shown in Table 2. In ALFWorld, our method compresses the Mean Episode Length from ≈ 32 steps down to ≈ 22 relative to GiGPO. Similar trend persists in the WebShop, where the gains are not as profound as in ALFWorld, however help in score improvement compared to GiGPO. More broadly, the results suggest:

• Learning of Environment Dynamics: The improvement in Mean Episode Length and Invalid Action Rate shows that the agent better understands the environment dynamics and, in particular, rarely repeats actions that have already failed.

• Loop Resolution and Novelty Seeking: The most supporting evidence of productive exploration is revealed by the Repetitive Action Rate in ALFWorld. While GiGPO spends 45.31% of its execution trajectory repeating identical commands, our approach drops this rate dramatically to 11.53%. This absolute reduction of 33.78% demonstrates that our intrinsic penalty inherently disincentivizes stale, repeated states. Instead, it forces the agent to continuously break out of loops and seek novel, progressive environmental interactions.

Both the above inferences can also be seen in qualitative action trajectories as shown in Figure 2. The contrast in the decrease in the number of invalid actions relative to Figure 1 suggests our approach of penalizing actions which repeat states is indeed helpful in reducing the redundancy of the agent’s trajectory.

To examine information gathering more closely, we record every step of our agent and of the retrained GiGPO agent (Section 4.1). Our agent searches the partially observable environment more widely while wasting fewer steps: it visits more distinct locations, revisits fewer of them, obtains new information from a larger share of its information-gathering actions, and finds the target object more often. The two agents also fail differently. GiGPO’s failed episodes are dominated by repeated observations, whereas ours keep exploring new states until the step budget runs out. These advantages hold on both seen and unseen scenes. Per action, both agents satisfy preconditions for nearly all of their actions; our agent’s advantage lies in avoiding repeated failures rather than in avoiding a first one (Appendix A.2).

## 4.5 Effect of the Intrinsic Penalty on Exploration

To isolate the effect of the penalty, we train two PPO agents from the same base model with identical settings, differing only in the penalty weight $\begin{array} { r } { ( \gamma = 0 \mathrm { v s . } \gamma = \frac { 1 } { 4 0 } } \end{array}$ of the success reward), and evaluate them after 100 training steps (details in Appendix A.2). We measure exploration by the duplicate-observation rate (steps whose observation repeats an earlier one), the location revisit rate (go to actions to an already visited location), the new-information rate (information-gathering actions go to, open and examine that reveal a new observation), and whether the target object is ever observed. A precondition violation is an executed action outside the environment’s admissible set; we report violations after the first one in an episode and the recovery rate, i.e. how often the action following a violation is admissible.

The penalty broadens exploration during training. Starting from identical behavior, the penalized agent reduces repeated observations much faster and covers a substantially larger part of the state space (Figure 3). This broader exploration precedes its gain in success, which emerges after the first 50 steps.

The learned policy explores productively at test time. It repeats observations and revisits locations far less often, gains new information from more of its actions, and finds the target object more often, equally on seen and unseen scenes (Table 4).

Better exploration leads to better use offeedback and higher success. Since repeated failures reveal nothing new, the penalized agent makes far fewer repeated precondition violations, recovers more often, satisfies preconditions more often, and succeeds considerably more often on both splits. Absolute success is lower than in Table 1 because of the shorter training.

![](images/94c73d3c4d1bbf0f30fd925e528d9db8a2e8b41a7f6058b510763a5155e6c8a8.jpg)  
Figure 3: Controlled comparison of our model with and without the intrinsic penalty during the first 100 training steps (identical settings). Top: validation success and duplicate-observation rate in the training rollouts. Bottom: distinct observations per episode in the training rollouts and the cumulative number of distinct (game, observation) states visited during training. Rollout curves are 10-step moving averages.

Table 4: Effect of the intrinsic penalty: PPO with γ = 0 vs. ours (identical settings, 100 training steps), same 128 episodes per split.
<table><tr><td></td><td colspan="2">Seen scenes</td><td colspan="2">Unseen scenes</td></tr><tr><td>Metric</td><td>γ = 0</td><td>Ours</td><td>γ = 0</td><td>Ours</td></tr><tr><td>Exploration</td><td></td><td></td><td></td><td></td></tr><tr><td>Duplicate-observation rate</td><td>59.3%</td><td>37.0%</td><td>61.8%</td><td>36.6%</td></tr><tr><td>Location revisit rate</td><td>68.6%</td><td>41.3%</td><td>68.7%</td><td>42.3%</td></tr><tr><td>New-information rate of info. actions</td><td>43.5%</td><td>67.4%</td><td>40.4%</td><td>67.1%</td></tr><tr><td>Distinct locations visited</td><td>4.68</td><td>6.17</td><td>4.87</td><td>6.88</td></tr><tr><td>Target object found</td><td>76.6%</td><td>87.5%</td><td>64.8%</td><td>85.9%</td></tr><tr><td>Preconditions</td><td></td><td></td><td></td><td></td></tr><tr><td>Precondition satisfied</td><td>68.0%</td><td>82.4%</td><td>67.8%</td><td>81.9%</td></tr><tr><td>Violations after the first, per episode</td><td>9.87</td><td>4.65</td><td>10.12</td><td>5.09</td></tr><tr><td>Recovery after a violation</td><td>47.2%</td><td>59.9%</td><td>50.8%</td><td>64.2%</td></tr></table>

## 5 Conclusion

In this paper, we addressed the tendency of RLtrained LLM agent policies to repeat invalid actions and fall into action loops in multi-turn, longhorizon environments. We introduced a conditional state-space exploration framework driven by an information-aware intrinsic reward. Instead of directly maximizing unconstrained state entropy, which risks driving the agent toward task-irrelevant trajectories, our approach reframes exploration by penalizing actions that produce redundant state information. Our experimental evaluation demonstrates that this framework achieves substantial performance gains and significantly mitigates action repetition and circular reasoning patterns.

## Limitations

Our experiments are conducted primarily on smaller reasoning models. Because exploration strategies are inherently influenced by a model’s overall capacity and pre-trained priors, larger instruction-tuned models may exhibit fewer failures than those documented in this study. Growth in model scale often mitigates baseline reasoning bottlenecks; nevertheless, optimizing smaller models remains highly valuable. Enhancing the capabilities of these smaller architectures enables more practical, targeted applications, particularly in hardware-constrained environments where deployment efficiency is critical.

Additionally, this work focuses specifically on text-based environments within ALFWorld and WebShop, which feature relatively small, welldefined action spaces that lend themselves well to set representations. Environments with larger, continuous, or stochastic action spaces may exhibit vastly different dynamics under this set-based state representation, potentially limiting direct generalizability.

Regardless of these limitations, our approach provides a structurally sound framework for agent reasoning and state representation in multi-turn settings. Ultimately, the insights gained from our set-based representation and reward design establish a foundation for future scaling and adaptation across more complex, open-ended environments.

## Acknowledgments

This research was supported by fundings from the Hong Kong RGC General Research Fund (152228/23E, 162161/24E, 162116/25E, 162180/25E), National Natural Science Foundation of China (NSFC) Key Program (No.62532005), Collaborative Research Fund (No. C1042-23GF, No. 5097-25G), NSFC/RGC Collaborative Research Scheme (Grant No. 62461160332 & CRS\_HKUST602/24), Research Impact Fund (No. R5011-23), and Areas of Excellence Scheme (AoE/E-601/22-R).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12248–12267.

Yonatan Ashlag, Uri Koren, Mirco Mutti, Esther Derman, Pierre-Luc Bacon, and Shie Mannor. 2025. State entropy regularization for robust reinforcement learning. In Advances in Neural Information Processing Systems.

Arthur Aubret, Laetitia Matignon, and Salima Hassas. 2019. A survey on intrinsic motivation in reinforcement learning. arXiv preprint arXiv:1908.06976.

Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Rémi Munos. 2016. Unifying count-based exploration and intrinsic motivation. In Proceedings ofthe 30th International Conference on Neural Information Processing Systems, NIPS’16, page 1479–1487, Red Hook, NY, USA. Curran Associates Inc.

Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A. Efros. 2019. Large-scale study of curiosity-driven learning. In ICLR.

Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. 2023. Grounding large language models in interactive environments with online reinforcement learning. In International conference on machine learning, pages 3676–3713. PMLR.

Nuttapong Chentanez, Andrew Barto, and Satinder Singh. 2004. Intrinsically motivated reinforcement learning. Advances in neural information processing systems, 17.

Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, and Peilin Zhao. 2026. Proximitybased multi-turn optimization: Practical credit assignment for llm agent training. arXiv preprint arXiv:2602.19225.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems.

Peiyuan Feng, Yichen He, Guanhua Huang, Yuan Lin, Hanchong Zhang, Yuchen Zhang, and Hang Li. 2024. Agile: A novel reinforcement learning framework of

llm agents. Advances in Neural Information Processing Systems, 37:5244–5284.

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Maolin Wang, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. 2025. Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621.

Shengyue Guan, Jindong Wang, Jiang Bian, Bin Zhu, Jian-Guang Lou, and Haoyi Xiong. 2026. Evaluating llm-based agents for multi-turn conversations: A survey. ACM Transactions on Intelligent Systems and Technology, 17(4):1–40.

Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. 2026. Tree search for LLM agent reinforcement learning. In International Conference on Learning Representations.

Vivek Kalyan and Martin Andrews. 2025. Reinforcement learning for long-horizon multi-turn search agents. arXiv preprint arXiv:2510.24126.

Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, and Younggyo Seo. 2023. Accelerating reinforcement learning with value-conditional state entropy exploration. Advances in Neural Information Processing Systems, 36:31811–31830.

Wouter Kool, Herke van Hoof, and Max Welling. 2019. Buy 4 reinforce samples, get a baseline for free!

Keliang Liu, Dingkang Yang, Ziyun Qian, Weijie Yin, Yuchi Wang, Hongsheng Li, Jun Liu, Peng Zhai, Yang Liu, and Lihua Zhang. 2025. Reinforcement learning meets large language models: A survey of advancements and applications across the llm lifecycle. arXiv preprint arXiv:2509.16679.

Mirco Mutti, Lorenzo Pratissoli, and Marcello Restelli. 2021. Task-agnostic exploration via policy gradient of a non-parametric state entropy estimate. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, pages 9028–9036.

Narjes Nourzad and Carlee Joe-Wong. 2026. MIRA: Memory-integrated reinforcement learning agent with limited LLM guidance. In The Fourteenth International Conference on Learning Representations.

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. 2017. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778–2787. PMLR.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, and 1 others. 2024. Multi-turn reinforcement learning with preference human feedback. Advances in Neural Information Processing Systems, 37:118953–118993.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Elizaveta Tennant, Stephen Hailes, and Mirco Musolesi. 2025. Moral alignment for LLM agents. In International Conference on Learning Representations.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, and 1 others. 2025. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073.

Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, and 1 others. 2025. Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning. arXiv preprint arXiv:2509.08755.

Yufei Xiang, Yiqun Shen, Yeqin Zhang, and Nguyen Cam-Tu. 2024. Retrospex: Language agent meets offline reinforcement learning critic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4650–4666.

Wujiang Xu, Wentian Zhao, Zhenting Wang, Yu-Jhe Li, Can Jin, Mingyu Jin, Kai Mei, Kun Wan, and Dimitris N Metaxas. 2025. Epo: Entropy-regularized policy optimization for llm agents reinforcement learning. arXiv preprint arXiv:2509.22576.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, and 1 others. 2025. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827.

Yifei Zhou and Andrea Zanette. 2024. Archer: training language model agents via hierarchical multi-turn rl. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

## A Appendix

## A.1 Datasets:

ALFWorld. ALFWorld (Shridhar et al., 2020) is a text-based interactive environment designed for studying embodied decision making and longhorizon reasoning in language agents. Derived from the ALFRED benchmark, it converts household manipulation tasks into a partially observable textual setting where agents interact through natural language actions and observations. Tasks typically involve navigation, object manipulation, and sequential planning over multiple turns, making the environment particularly suitable for evaluating reinforcement learning methods for LLM agents under sparse rewards and delayed feedback. Concretely, ALFWorld contains 6 task types of various difficulty. The tasks are:

• Pick & Place: agent must find an object of a specified type, pick it up, find the correct location to place it, and place it down there.

• Examine in Light: agent must find an object of a specified type, locate and turn on a light source with the object still in hand.

• Clean & Place: agent must find an object of a specified type, pick it up, move to a sink or a basin, turn on the faucet and wash the object, find the correct location to place it, and place it down there.

• Heat & Place: agent must find an object of a specified type, pick it up, move to the microwave, heat the object by turning on the microwave, find the correct location to place it, and place it down there.

• Cool & Place: agent must find an object of a specified type, pick it up, move to the fridge, place the object in the fridge to cool it, find the correct location to place it, and place it down there.

• Pick Two & Place: agent must find an object of a specified type, find the correct location to place it, place it down there, look for another object of a specified type, pick up the object, return to the previous location, place it down there together with the first object.

WebShop. WebShop (Yao et al., 2022) is an ecommerce interaction benchmark that evaluates language agents on realistic online shopping tasks. In this environment, agents must search through large collections of products, inspect item descriptions and reviews, and iteratively refine decisions to satisfy user instructions. Unlike embodied environments, WebShop emphasizes retrieval, preference understanding, and multi-step reasoning in semistructured web settings. The benchmark is widely used for evaluating the effectiveness of sequential decision-making and reinforcement learning approaches for interactive LLM agents. Concretely, the WebShop dataset contains 1,000,000 products, 670 unique attributes, 842,849 unique options, extracted publicly from Amazon. It presents over 12,000 annotated trajectories and an action space of 8 valid actions.

## A.2 Additional Experiments

This subsection reports additional ALFWorld experiments that analyze (i) whether the trained agent gathers information more effectively at test time, and (ii) whether more of its actions satisfy their preconditions. Sections 4.4 and 4.5 summarize the main findings; here we give the full results.

## A.2.1 Test-Time Exploration and Preconditions

Setup. We compare our agent with a GiGPO agent that we trained ourselves using the released verl-agent code. Both agents play the same 128 evaluation episodes (91 distinct games) from the ALFWorld in-distribution split, with sampling temperature 0.4 and at most 50 steps. We record every step (observation, admissible actions, chosen action and next observation) and compute all metrics below per episode, then average over episodes.

Test-time exploration and information gathering. ALFWorld is partially observable: the contents of a receptacle are revealed only when the agent goes to it or opens it. Table 5 measures how the agents gather information. Our agent covers more of the environment with less effort: it visits more distinct locations per episode while taking fewer steps, revisits locations less often, and a larger share of its information-gathering actions (go to, open, examine) reveals a previously unseen observation. As a result, it finds the target object more often. When both agents find the target, our agent gets there in fewer steps despite searching more locations, i.e. it searches more widely and wastes fewer steps on revisits. The failure modes also differ. Failed episodes of both agents exhaust the step budget, but GiGPO’s failures are dominated by repetition, whereas our failed episodes keep exploring new states.

Table 5: Test-time information gathering (per-episode means). <sup>†</sup>Episodes in which both agents find the target.
<table><tr><td>Metric</td><td>GiGPO</td><td>Ours</td></tr><tr><td>Success rate</td><td>89.1%</td><td>95.3%</td></tr><tr><td>Episode length (steps)</td><td>14.7</td><td>12.5</td></tr><tr><td>Distinct locations visited</td><td>3.66</td><td>5.16</td></tr><tr><td>Location revisit rate</td><td>31.5%</td><td>18.9%</td></tr><tr><td>New-information rate of info. actions</td><td>81.2%</td><td>89.2%</td></tr><tr><td>Duplicate-observation rate</td><td>16.7%</td><td>8.7%</td></tr><tr><td>Target object found</td><td>91.4%</td><td>96.9%</td></tr><tr><td>Steps until target found†</td><td>4.46</td><td>4.19</td></tr><tr><td>Locations searched before target†</td><td>2.27</td><td>3.19</td></tr><tr><td>Duplicate-obs. rate in failed episodes</td><td>72%</td><td>27%</td></tr></table>

Precondition satisfaction at test time. ALFWorld provides the set of admissible actions in each state, which are exactly the actions whose preconditions are satisfied. We therefore measure, for each action type, how many actions each agent executes and the fraction of them that are admissible (Table 6). On the same episodes, our agent completes the tasks with about 16% fewer actions. The saving comes almost entirely from navigation, as it needs far fewer go to actions, while it uses information-gathering actions such as open and examine more often. Both agents satisfy preconditions for nearly all of their actions, with a small difference in GiGPO’s favor overall. The per-action picture is mixed: our agent is more reliable on take and examine, while GiGPO is more reliable on put. Against this baseline, our agent therefore does not show a higher precondition-satisfaction rate per action at test time. Its advantage lies in reaching the goal with fewer actions and in avoiding repeated uninformative actions (Table 5), which is the behavior the penalty directly targets.

Table 6: Actions per episode and precondition satisfaction by action type. <sup>‡</sup>Includes close and look.
<table><tr><td></td><td colspan="3">GiGPO</td><td colspan="3">Ours</td></tr><tr><td>Action</td><td>Per ep.</td><td>n</td><td>Satisfied</td><td>Per ep.</td><td>n</td><td>Satisfied</td></tr><tr><td>go to</td><td>10.24</td><td>1311</td><td>98.6%</td><td>7.26</td><td>929</td><td>98.3%</td></tr><tr><td>take</td><td>1.58</td><td>202</td><td>93.6%</td><td>1.43</td><td>183</td><td>100.0%</td></tr><tr><td>put</td><td>1.38</td><td>176</td><td>99.4%</td><td>1.40</td><td>179</td><td>92.2%</td></tr><tr><td>open</td><td>0.74</td><td>95</td><td>98.9%</td><td>1.34</td><td>171</td><td>99.4%</td></tr><tr><td>examine</td><td>0.07</td><td>9</td><td>66.7%</td><td>0.37</td><td>47</td><td>80.9%</td></tr><tr><td>clean</td><td>0.22</td><td>28</td><td>100.0%</td><td>0.24</td><td>31</td><td>96.8%</td></tr><tr><td>cool</td><td>0.13</td><td>17</td><td>100.0%</td><td>0.13</td><td>17</td><td>100.0%</td></tr><tr><td>heat</td><td>0.09</td><td>12</td><td>100.0%</td><td>0.09</td><td>11</td><td>100.0%</td></tr><tr><td>use</td><td>0.23</td><td>30</td><td>100.0%</td><td>0.11</td><td>14</td><td>100.0%</td></tr><tr><td>All actions‡</td><td>14.71</td><td>1883</td><td>98.0%</td><td>12.38</td><td>1584</td><td>97.4%</td></tr></table>

Repeated precondition violations. In ALFWorld, every action whose preconditions are not satisfied returns the same fixed observation, "Nothing happens." (Appendix A.3). The first violation in an episode therefore adds a new observation to the state memory and is not penalized, while every later violation in the same episode repeats this observation and is penalized. The penalty is thus not expected to prevent a first, exploratory violation, but to discourage the agent from violating preconditions again after it has already received this feedback. Table 7 measures this behavior on the same 128 episodes, using the following definitions:

• Violation: an executed action that is not in the admissible set of the state in which it is taken, i.e. an action whose preconditions are not satisfied. we report violations after the first, which are the violations which receive the intrinsic penalty. Per episode divides their number by the 128 episodes.

• Exact repeat of a failed action: a violation in which the agent executes the very same action string that has already failed earlier in the same episode.

• Share of all violations: the percentage of an agent’s violations that fall into the given category.

• Back-to-back violations: violations that immediately follow another violation.

• Recovery after a violation: the percentage of violations whose next action is admissible.

Our agent is less likely to violate preconditions again once it has received the failure feedback: it has fewer violations after the first in an episode. The clearest difference is in exact repeats. About half of GiGPO’s violations re-execute an action that has already failed in the same episode, compared with about a quarter of ours. Our agent also recovers more often, with the action following a violation more frequently being admissible. This pattern matches the mechanism of the penalty, which leaves the first violation unpenalized but penalizes repeating it.

## A.2.2 Generalization to Unseen Scenes

We repeat all of the above analyses on the ALFWorld out-of-distribution split, which contains scenes not seen during training. Both agents use the same checkpoints and evaluation settings as before, and both play the same 128 episodes (74 distinct games). Table 8 summarizes the results, and all metrics follow the definitions given above.

Table 7: Repeated precondition violations (shares of each agent’s violations).
<table><tr><td>Metric</td><td>GiGPO</td><td>Ours</td></tr><tr><td>Violations after the first (total)</td><td>26</td><td>23</td></tr><tr><td>per episode</td><td>0.20</td><td>0.18</td></tr><tr><td>share of all violations</td><td>70.3%</td><td>56.1%</td></tr><tr><td>Exact repeats of a failed action (total)</td><td>19</td><td>10</td></tr><tr><td>share of all violations</td><td>51.4%</td><td>24.4%</td></tr><tr><td>Back-to-back violations</td><td>4</td><td>3</td></tr><tr><td>Recovery after a violation</td><td>88.9%</td><td>92.5%</td></tr><tr><td>Maximum violations in one episode</td><td>8</td><td>7</td></tr></table>

The advantage of our agent grows on unseen scenes. Its success rate barely changes, while GiGPO’s drops noticeably, so the gap roughly doubles to about 12 points, and our agent is ahead on every task type. All information-gathering differences persist, mostly with larger margins. Its failed episodes keep exploring rather than looping, and it again completes the tasks with fewer actions. On unseen scenes, it also satisfies preconditions slightly more often than GiGPO and makes fewer violations in total.

Table 8: Results on unseen scenes (out-of-distribution split). <sup>†</sup>Episodes in which both agents find the target.
<table><tr><td>Metric</td><td>GiGPO</td><td>Ours</td></tr><tr><td>Information gathering</td><td></td><td></td></tr><tr><td>Episode length (steps)</td><td>17.4</td><td>14.7</td></tr><tr><td>Distinct locations visited</td><td>4.22</td><td>5.80</td></tr><tr><td>Location revisit rate</td><td>35.4%</td><td>22.1%</td></tr><tr><td>New-information rate of info. actions</td><td>76.7%</td><td>88.7%</td></tr><tr><td>Duplicate-observation rate</td><td>20.8%</td><td>9.4%</td></tr><tr><td>Target object found</td><td>88.3%</td><td>97.7%</td></tr><tr><td>Steps until target found†</td><td>4.96</td><td>4.44</td></tr><tr><td>Duplicate-obs. rate in failed episodes</td><td>67%</td><td>25%</td></tr><tr><td>Actions and preconditions</td><td></td><td></td></tr><tr><td>Actions per episode</td><td>17.4</td><td>14.6</td></tr><tr><td>Precondition satisfied (all actions)</td><td>96.7%</td><td>97.3%</td></tr><tr><td>take / put /examine</td><td>94.0 / 99.4 / 40.0%</td><td>98.7 / 94.1 / 93.0%</td></tr><tr><td>Repeated violations</td><td></td><td></td></tr><tr><td>Violations after the first, per episode</td><td>0.38</td><td>0.23</td></tr><tr><td>share of all violations</td><td>65.8%</td><td>58.0%</td></tr><tr><td>Exact repeats of a failed action (total)</td><td>31</td><td>19</td></tr><tr><td>share of all violations</td><td>42.5%</td><td>38.0%</td></tr><tr><td>Back-to-back violations</td><td>16</td><td>17</td></tr><tr><td>Recovery after a violation</td><td>78.1%</td><td>64.6%</td></tr><tr><td>Maximum violations in one episode</td><td>12</td><td>7</td></tr></table>

## A.2.3 Controlled Comparison with the Penalty Disabled

Here we detail the controlled comparison of Section 4.5. Both PPO agents start from Qwen2.5-1.5B-Instruct and share all hyperparameters, prompts and the random seed, so they encounter exactly the same training games at every step. We record all training rollouts (100 episodes per step) and evaluate both policies on the same validation episodes every 25 steps, using a single run per setting.

Exploration during training. Table 9 complements Figure 3 with further training-rollout metrics at selected steps. Besides repeating fewer observations, the penalized agent revisits fewer locations, obtains new information from a larger share of its information-gathering actions, and runs into fewer precondition failures throughout training. The number of distinct observations per episode eventually decreases for the penalized agent because it succeeds more often and its episodes become shorter.

Table 9: Training rollouts with $\gamma = 0$ vs. ours, averaged over the 10 steps ending at each step.
<table><tr><td></td><td colspan="2">Step 25</td><td colspan="2">Step 50</td><td colspan="2">Step 100</td></tr><tr><td>Metric</td><td>γ = 0</td><td>Ours</td><td>γ = 0</td><td>Ours</td><td>γ = 0</td><td>Ours</td></tr><tr><td>Duplicate-observation rate</td><td>69.6%</td><td>63.2%</td><td>66.1%</td><td>51.5%</td><td>56.9%</td><td>40.7%</td></tr><tr><td>Distinct observations / episode</td><td>14.5</td><td>17.6</td><td>14.7</td><td>21.6</td><td>15.1</td><td>18.8</td></tr><tr><td>Location revisit rate</td><td>67.4%</td><td>64.5%</td><td>65.6%</td><td>58.9%</td><td>68.9%</td><td>49.8%</td></tr><tr><td>New-information rate of info. actions</td><td>33.0%</td><td>38.2%</td><td>37.8%</td><td>48.8%</td><td>42.0%</td><td>59.7%</td></tr><tr><td>&quot;Nothing happens.&quot; rate</td><td>45.7%</td><td>40.2%</td><td>44.4%</td><td>31.1%</td><td>31.3%</td><td>20.2%</td></tr><tr><td>Cumulative distinct states (k)</td><td>28.3</td><td>30.4</td><td>50.4</td><td>65.7</td><td>84.9</td><td>116.0</td></tr><tr><td>Training success</td><td>11.4%</td><td>11.4%</td><td>21.3%</td><td>20.7%</td><td>41.9%</td><td>49.3%</td></tr><tr><td>Episode length (steps)</td><td>47.1</td><td>47.3</td><td>44.8</td><td>44.8</td><td>38.5</td><td>35.2</td></tr><tr><td>Validation success</td><td>15%</td><td>19%</td><td>28%</td><td>32%</td><td>45%</td><td>57%</td></tr></table>

Test-time behavior of the controlled agents. Table 10 extends Table 4 with further metrics, using the same 128 episodes per split as Tables 5–8. In addition to the effects discussed in Section 4.5, the penalized agent completes episodes with fewer actions, wins far more of the games that only one of the two agents solves, satisfies preconditions more often for almost every action type, and its failed episodes contain less repetition. At this stage of training, both agents still violate preconditions often, so absolute rates are lower than for the fully trained agent in Tables 5–8.

Table 10: Test-time behavior of the controlled agents after 100 training steps.
<table><tr><td></td><td colspan="2">Seen scenes</td><td colspan="2">Unseen scenes</td></tr><tr><td>Metric</td><td>γ = 0</td><td>Ours</td><td>γ = 0</td><td>Ours</td></tr><tr><td>Success rate</td><td>39.1%</td><td>57.8%</td><td>34.4%</td><td>52.3%</td></tr><tr><td>Episode length (steps)</td><td>37.9</td><td>31.0</td><td>40.5</td><td>32.7</td></tr><tr><td>Actions per episode</td><td>33.8</td><td>30.3</td><td>34.4</td><td>32.1</td></tr><tr><td>Distinct locations visited</td><td>4.68</td><td>6.17</td><td>4.87</td><td>6.88</td></tr><tr><td>Location revisit rate</td><td>68.6%</td><td>41.3%</td><td>68.7%</td><td>42.3%</td></tr><tr><td>New-information rate of info. actions</td><td>43.5%</td><td>67.4%</td><td>40.4%</td><td>67.1%</td></tr><tr><td>Duplicate-observation rate</td><td>59.3%</td><td>37.0%</td><td>61.8%</td><td>36.6%</td></tr><tr><td>Target object found</td><td>76.6%</td><td>87.5%</td><td>64.8%</td><td>85.9%</td></tr><tr><td>Duplicate-obs. rate in failed episodes</td><td>72%</td><td>56%</td><td>72%</td><td>57%</td></tr><tr><td>Precondition satisfied (all actions)</td><td>68.0%</td><td>82.4%</td><td>67.8%</td><td>81.9%</td></tr><tr><td>Violations after the first, per episode</td><td>9.87</td><td>4.65</td><td>10.12</td><td>5.09</td></tr><tr><td>Exact repeats, share of all violations</td><td>62.5%</td><td>55.4%</td><td>61.7%</td><td>54.1%</td></tr><tr><td>Recovery after a violation</td><td>47.2%</td><td>59.9%</td><td>50.8%</td><td>64.2%</td></tr></table>

## A.2.4 Relation to Theory and Qualitative Example

Relation to Proposition 2. Proposition 2 bounds the visitation entropy $H ( P _ { L } )$ of an episode by its number of duplicate observations D. Figure 4 plots both quantities for all full-length episodes of the controlled agents. Every episode of both agents lies between the two bounds, and across all episodes, entropy decreases almost monotonically with the number of duplicates (rank correlation close to −1). The penalized agent produces episodes with clearly fewer duplicates and correspondingly higher visitation entropy, both in failed full-length episodes and on average over all episodes, where the paired confidence interval of the entropy difference excludes zero on both splits. This confirms empirically that minimizing the cumulative penalty, i.e. the number of duplicate observations, translates into higher visitation entropy in practice.

![](images/7f62aa501bc13aa53805eb2c3a9065d2fddbc74fcde264884c99029b87ae7bf4.jpg)  
Figure 4: Visitation entropy $H ( P _ { L } )$ against the number of duplicate observations D for all full-length episodes $( L = 5 1$ observations) of the controlled agents after 100 training steps, seen and unseen splits combined. The dotted and dashed curves are the upper and lower bounds of Proposition 2.

Qualitative example. Figure 5 shows the two controlled agents on the same game from the unseen split, taken directly from the evaluation logs. Both agents start identically and pick up the mug. The agent trained without the penalty then repeatedly issues a command for the location it is already at, receives "Nothing happens." over and over, and never cools the mug before the step budget runs out. The penalized agent instead proceeds to the fridge, cools the mug and completes the task in eight steps. This is the looping failure mode of Figure 1, which the penalty removes.

<table><tr><td colspan="2">Task: cool some mug and put it in coffeemachine (unseen scene).</td></tr><tr><td>Penalty  $\begin{array} { r } { ( \gamma = \frac { 1 } { 4 0 } ) \colon } \end{array}$  success in 8 steps 1 go to sinkbasin 1 2 go to countertop 1 3 take mug 1 from countertop 1 4 go to fridge 1 5-6 cool mug 1 with fridge 1 7 go to coffeemachine 1</td><td>arrive; sinkbasin empty arrive; see mug 1 pick up mug 1 arrive; fridge closed cool mug 1 arrive</td></tr><tr><td>No penalty  $( \gamma = 0 ) \colon$  failure after 50 steps 1–3 same as above (mug 1 picked up)</td><td></td></tr><tr><td>4 go to sinkbasin 1</td><td>arrive</td></tr><tr><td>5-7 go to sinkbasin 1 (×3) 8–9 go to fridge 1, open fridge 1 10 go to sinkbasin 1</td><td>&quot;Nothing happens.&quot; fridge open arrive</td></tr></table>

Figure 5: The two controlled agents after 100 training steps on the same unseen game (actions and observations from the evaluation logs, abridged). In total, 31 of the 50 steps of the agent without the penalty return "Nothing happens.".

## A.2.5 Additional Qualitative Results

In Figure 6, we further show some of the example outputs of the LLM agent trained with intrinsic productive exploration rewards framework.

## A.3 Reproducibility Details

Penalty weight γ and reward weight λ. $\gamma =$ 1/10 destabilized training; $\gamma = 1 / 1 0 0$ matched the baseline; $\gamma = 1 / 4 0$ (used throughout) gave a clear, consistent improvement.

## Prompts. Shown in Figure 7.

Invalid actions: The two environments signal invalid actions differently. In ALFWorld, every action whose preconditions are not satisfied returns the fixed observation "Nothing happens.", and we count a step as invalid when its observation matches this string or when the agent’s output contains no well-formed action. WebShop provides no failure message: an action that is not available on the current page, such as clicking an element that is not displayed, is ignored, and the environment returns the unchanged page. For WebShop, we therefore count a step as invalid when the agent’s output contains no well-formed action. Because an ignored action returns the same page, it always produces a duplicate observation and is penalized by our intrinsic reward.

Rollout and optimization settings. Batch size, rollouts/step, PPO epochs, clip range, and learning rate are unmodified verl-agent/GiGPO defaults (Feng et al., 2025). Details are mentioned in

Table 11: Training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Rollouts per step</td><td>128</td></tr><tr><td>PPO clip range Learning rate</td><td>0.2 1e-6</td></tr><tr><td>Training steps</td><td>300</td></tr><tr><td>Max steps (ALFWorld)</td><td>50</td></tr><tr><td>Max steps (WebShop)</td><td>15</td></tr><tr><td>γ (intrinsic penalty)</td><td>1/40× success reward</td></tr></table>

Train/test splits. We use the standard ALFWorld seen/unseen split released by Shridhar et al. (2020) and the standard WebShop train/eval split released by Yao et al. (2022).

## A.4 Declaration

We have cited the models and code used in Section 4. We have used ChatGPT, Gemini, and Claude to assist with polishing writing.

![](images/4eb877edd720c2d86f156b0dc9b19ba9b2ed2929de7b03e7373684ae72606d8e.jpg)

![](images/2625e70fc5742541bd2691b34f29ee5013ab3f1533bf9200ab336eb725d5a308.jpg)  
Figure 6: Additional example outputs of the multi-turn interaction of the LLM agent on the ALFWorld dataset after training with conditional state-space exploration with intrinsic rewards.

![](images/be9bc5eef348f214039fa45da37f50c2321e956c5f7487f3ac24269281207af5.jpg)  
Figure 7: Prompts used for both ALFWorld and WebShop environments.