# Learning from Viable Failure Prefixes: Milestone Viability Potential Policy Optimization for Long-Horizon LLM Agents

Qi Zhou<sup>†</sup>, Yuanfan Li<sup>†,</sup> <sup>∗</sup>

Xi’an Jiaotong University

<sup>†</sup> Equal contribution, <sup>∗</sup> Correspondence to: liyuan7716@gmail.com

## Abstract

Long-horizon LLM agents require reinforcement learning methods that can assign credit to intermediate decisions under sparse and delayed rewards. Existing group-based methods such as GRPO and GiGPO alleviate this issue by comparing rollout returns or repeated anchor states, but they still fail when the compared returns have no variation. We identify this failure mode as zero-credit failure: during early training, many failed rollouts contain useful prefixes, yet existing methods assign them no task-discriminative advantage. To address this issue, we propose Milestone Viability Potential Policy Optimization (MVPO), a potentialrouted policy optimization algorithm that learns from viable failure prefixes. MVPO estimates prefix potential over Union-Find viability regions, repairs zero-credit groups with potentialdifference advantages, and attenuates the potential branch according to relative performance progress. Experiments with Qwen2.5- 1.5B-Instruct show that MVPO outperforms eight strong baselines, including GRPO and GiGPO. Under the same training length, MVPO improves over the GiGPO baseline by +4.4 success points on ALFWorld and +5.3 on WebShop, while adding only 0.16%–0.20% advantage-construction overhead.

## 1 Introduction

Large language models (LLMs) have achieved remarkable progress across language understanding, reasoning, and generation tasks (Vaswani et al., 2017; Brown et al., 2020; Singh et al., 2025), and are increasingly being developed as autonomous agents that interact with external environments, execute multi-step plans, and solve long-horizon decision-making problems (Bai et al., 2022; Shao et al., 2024; Feng et al., 2025). Such agentic LLMs have been studied in embodied environments, web navigation, tool-use tasks, and application-centered workflows, where success requires a sequence of environment-dependent actions rather than a single response (Shridhar et al., 2020; Yao et al., 2022a; Trivedi et al., 2024; Wei et al., 2025). Training these agents with reinforcement learning (RL) is challenging because rewards are often sparse and delayed: most early rollouts fail before reaching the goal, and non-zero feedback is observed only after completing the task. Recent critic-free policy optimization algorithms attempt to mitigate this issue by comparing multiple rollouts under the same prompt. GRPO (Shao et al., 2024) computes grouprelative advantages from trajectory-level returns, but its credit signal remains tied to final episode outcomes. GiGPO (Feng et al., 2025) further adapts group-based optimization to agentic tasks by constructing step-level advantages at repeated anchor states, comparing subsequent returns of rollouts that visit the same state. Although this provides denser supervision than trajectory-level GRPO, we find that it still fails to eliminate sparse-reward failure in long-horizon agentic training.

![](images/f57d803e8b8246a60e9985c84aece07fca12343402f46c43d4c33ecdd0363447.jpg)

![](images/9005552f2810bcdf50d5aa4b86b0ae3193a8f85dcc31379d8f1661c99ed7a7c0.jpg)

![](images/1d8cb1b0b44440b2ecd82a3d016fcbb511b6bfd2d65a740dabc68d49523a125f.jpg)

![](images/533b1be606b58dcc923a28fdb027d2cd87fb8a3e12e48f03c791a166f4bd8b66.jpg)

![](images/c4d48063312728f8ebfa71615db98ee666fb1a1e77e3269119fe09a1348447bd.jpg)  
Figure 1: Motivation and overview of MVPO. Top: an ALFWorld case of zero-creditfailure: all rollouts fail, making GRPO/GiGPO assign no task-discriminative advantage, while MVPO identifies viable prefixes using prefix potential. Bottom (a): zero-advantage steps affect over 90% of GRPO steps and about 65% of GiGPO steps in early WebShop training. Bottom (b): higher prefix potential predicts higher future success on ALF-World. Bottom (c): MVPO reaches 40% success earlier during training. Bottom (d): MVPO improves final success over GiGPO on both benchmarks.

The key issue is that both rollout-level and anchor-level comparisons require return variation, which is especially scarce at the beginning of training. When most sampled rollouts fail, GRPO receives nearly identical trajectory returns and therefore assigns no task-discriminative relative advantage. GiGPO alleviates this issue by comparing continuations from repeated anchor states, but its step-level advantage still collapses whenever all continuations from an anchor have identical returns. We call these cases zero-credit groups: groups in which existing credit estimators cannot distinguish useful decisions from useless ones because the observed returns contain no variation. As shown in Figure 1(a), zero-credit groups dominate the early training stage: over 90% of GRPO steps and more than 60% of GiGPO steps have zero advantage. This means that a large fraction of early interactions provides no task-discriminative policy-gradient signal, slowing down the initial rise of the training curve. Thus, early agent training is sparse not only at the trajectory level, but also at the prefix level.

To understand the cause of this prefix-level sparsity, we examine what is hidden inside failed rollouts. Figure 1 (top) shows an ALFWorld task, “put a cooled mug on the countertop,” where three rollouts all fail with zero returns. Yet their prefixes differ substantially: one rollout reaches the kitchen, takes the mug, and approaches the fridge, while the others pick up a wrong object or explore irrelevant locations. GRPO and GiGPO treat these rollouts as equally uninformative once their returns are all zero. This motivates our view that failed trajectories should not be uniformly discarded; their viable prefixes should be identified and reinforced.

Motivated by this diagnosis, we propose Milestone Viability Potential Policy Optimization (MVPO), a potential-routed LLM agent training algorithm that learns from viable failure prefixes. Instead of assigning credit only from final trajectory outcomes, MVPO estimates a prefix potential, which measures whether an intermediate state remains on a promising path toward task completion. For example, finding the apple or reaching the fridge should have higher potential than entering a shelf loop or holding an irrelevant object. To estimate this signal robustly, MVPO uses a Union-Find structure to merge semantically equivalent prefixes across rollouts into viability regions, where each region aggregates future success, loop frequency, and adaptive milestone progress. During optimization, MVPO routes the step-level signal according to credit availability: it preserves comparative anchor advantages when they are informative, and repairs zero-credit groups with potential-difference advantages that reward transitions toward higher-potential regions and suppress regressions or dead ends. Finally, MVPO learns milestone weights from rollout statistics and attenuates the potential branch according to relative performance progress, so that potential-based credit dominates early sparse-reward training but gradually gives way to anchor-based credit later.

Figure 1(b–d) validates both the mechanism and effectiveness of MVPO. On ALFWorld, future success increases from 34.6% in the lowest-potential bucket to 94.5% in the highest-potential bucket, showing that prefix potential captures meaningful task progress. Using this signal to repair zero-credit steps, MVPO reaches 40% ALFWorld success at step 52, earlier than GiGPO at step 62 and GRPO at step 73. Across ALFWorld (Shridhar et al., 2020) and WebShop (Yao et al., 2022a), MVPO outperforms eight strong baselines, including promptbased agents, actor-critic RL, GRPO (Shao et al., 2024), and GiGPO (Feng et al., 2025). Under the same training length, it improves ALFWorld/Web-Shop success over GiGPO by +4.4/+5.3 points with Qwen2.5-1.5B-Instruct and +1.1/+2.7 points with Qwen2.5-7B-Instruct, while adding only 0.16%– 0.20% advantage-construction overhead. Our contributions are as follows:

• Zero-Credit Prefix Failure Diagnosis. We identify a key limitation of existing groupbased policy optimization methods in longhorizon agent training: when rollout or anchor groups contain identical returns, their advantages collapse to zero. This prevents GRPO and GiGPO from distinguishing viable failure prefixes from completely unproductive trajectories during early sparse-reward training.

• Potential-Routed Prefix Credit Repair. We propose MVPO, a new potential-routed policy optimization algorithm that learns from viable failure prefixes. MVPO estimates prefix potential via Union-Find viability regions and repairs zero-credit groups with potential-difference advantages, while preserving anchor-based comparative credit when it is available.

• Strong Long-Horizon Agent Performance. We evaluate MVPO on ALFWorld and Web-Shop, where it outperforms eight strong baselines, including prompt-based agents, actorcritic RL methods, GRPO, and GiGPO. Under the same training length, MVPO improves ALFWorld/WebShop success over GiGPO by +4.4/+5.3 points with Qwen2.5-1.5B-Instruct and +1.1/+2.7 points with Qwen2.5- 7B-Instruct, with only 0.16%–0.20% additional advantage-construction overhead.

## 2 Related Work

LLMs as autonomous agents. LLMs are increasingly used as autonomous agents that interact with environments, invoke tools, and solve multi-step tasks. Early systems mainly relied on frozen LLMs with prompting, reasoning-action interleaving, memory, reflection, retrieval, and tool use, such as ReAct (Yao et al., 2022b), Reflexion (Shinn et al., 2023), WebGPT (Nakano et al., 2021), and Toolformer (Schick et al., 2023). Agent benchmarks have expanded from embodied and text-based tasks to web navigation, shopping, mobile-device control, and application-centered workflows (Shridhar et al., 2020; Yao et al., 2022a; Deng et al., 2023; Zhou et al., 2024a; Trivedi et al., 2024; Wen et al., 2024). These environments require long sequences of environment-dependent actions under delayed rewards, making policy learning harder than single-turn reasoning. Promptingbased agents exploit pretrained knowledge, but provide limited mechanisms for improving policies from failed interactions, motivating supervised and reinforcement learning for LLM agents.

Policy optimization for LLMs and agents. RL has been widely used to align and improve LLMs, from RLHF with PPO (Ziegler et al., 2019; Stiennon et al., 2020; Ouyang et al., 2022; Schulman et al., 2017) to critic-free estimators such as RE-INFORCE, RLOO, and GRPO (Williams, 1992; Ahmadian et al., 2024; Shao et al., 2024). Groupbased algorithms such as Dr.GRPO, DAPO, CPPO, and GSPO (Liu et al., 2025; Yu et al., 2025; Lin et al., 2026; Zheng et al., 2025) further improve scalable LLM RL and have been applied to reasoning, search, and tool use (Dong et al., 2025a,b). For multi-turn agents, ArCHer (Zhou et al., 2024b) uses hierarchical RL, Agent Q (Putta et al., 2024) combines MCTS with iterative fine-tuning, RAGEN/StarPO (Wang et al., 2025a) studies trajectory-level agent optimization, Tree-GRPO (Ji et al., 2025) derives process signals from tree-structured rollouts, and GiGPO (Feng et al., 2025) constructs step-level advantages from repeated anchor states. These methods improve trajectory collection or densify credit from sparse outcomes, but still rely on return variation within rollout or anchor groups. When all rollouts fail or all anchor continuations have identical returns, their advantages collapse to zero, leaving viable prefixes inside failed trajectories unlearned. In contrast, MVPO targets this zero-credit regime by routing such steps to prefix-potential advantages estimated over Union-Find viability regions, enabling agents to learn from viable failure prefixes while preserving comparative credit when available.

## 3 Preliminaries

Problem setup. We consider a long-horizon agentic RL setting where an LLM agent receives a task instruction $x \sim p ( \mathcal { X } )$ and interacts with an environment for multiple steps. At step $t ,$ the agent observes state $s _ { t } ,$ samples a textual action $a _ { t } \sim \pi _ { \theta } ( \cdot \mid s _ { t } , x )$ , receives reward $r _ { t } ,$ and transits to $s _ { t + 1 }$ . An episode is a trajectory

$$
\tau = \{ ( s _ { 1 } , a _ { 1 } , r _ { 1 } ) , \ldots , ( s _ { T } , a _ { T } , r _ { T } ) \} ,\tag{1}
$$

with return $R ( \tau )$ . In long-horizon agentic tasks, rewards are often sparse and delayed: most intermediate actions receive no direct supervision, and early rollouts frequently fail before reaching the goal. The central challenge is therefore to assign useful credit to intermediate decisions, especially inside failed trajectories.

Group-based credit estimation and zero-credit groups. Critic-free group-based RL estimates advantages by comparing multiple rollouts from the same task. Given N trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { N }$ sampled from the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ , GRPO (Shao et al., 2024) computes a trajectory-level group-relative advantage:

$$
A _ { i } ^ { \mathrm { e p i } } = \frac { R _ { i } - \mu _ { R } } { \sigma _ { R } + \epsilon } , \quad \mu _ { R } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R _ { j } ,\tag{2}
$$

where $R _ { i } = R ( \tau _ { i } )$ and $\sigma _ { R }$ is the standard deviation of trajectory returns. This avoids learning a value function, but all actions in the same trajectory share the same advantage. When all rollouts fail and obtain identical returns, $\sigma _ { R } = 0 $ , so GRPO provides no task-discriminative signal.

GiGPO (Feng et al., 2025) improves this by constructing step-level credit at repeated anchor states. For trajectory $\tau _ { i } .$ , the discounted future return from step t is

$$
G _ { i , t } = \sum _ { \ell = t } ^ { T _ { i } } \gamma ^ { \ell - t } r _ { i , \ell } .\tag{3}
$$

For an anchor state s, GiGPO collects its occurrence set

$$
{ \mathcal { T } } _ { s } = \{ ( i , t ) \mid s _ { i , t } = s \} ,\tag{4}
$$

and normalizes future returns within this anchor group:

$$
A _ { i , t } ^ { \mathrm { a n c h o r } } = \frac { G _ { i , t } - \mu _ { s } } { \sigma _ { s } + \epsilon } , \quad ( i , t ) \in \mathcal { T } _ { s } .\tag{5}
$$

The final advantage is

$$
\begin{array} { r } { \hat { A } _ { i , t } ^ { \mathrm { G i G P O } } = A _ { i } ^ { \mathrm { e p i } } + \omega A _ { i , t } ^ { \mathrm { a n c h o r } } . } \end{array}\tag{6}
$$

Although GiGPO provides denser credit than GRPO, it still requires return variation inside anchor groups. If all continuations from an anchor state have identical future returns, then $\sigma _ { s } = 0$ and the anchor advantage also collapses. We call such rollout or anchor groups zero-credit groups:

$$
\mathrm { Z e r o C r e d i t } ( g ) \Longleftrightarrow \mathrm { S t d } \left( \{ G _ { u } \mid u \in g \} \right) = 0 .\tag{7}
$$

Zero-credit groups are especially common in early sparse-reward training, when most rollouts fail. As a result, existing group-based PO methods treat viable failure prefixes and completely unproductive prefixes as equally uninformative. This motivates MVPO, which repairs zero-credit groups with prefix-potential credit in Section 4.

## 4 Methodology

As discussed in Section 3, group-based training methods are efficient because they avoid learning a value model and estimate advantages from rollout groups. However, they require return variation. When all rollouts fail, trajectory-level credit becomes uninformative; when all continuations from an anchor state obtain identical returns, anchorlevel credit also collapses. We call these cases zero-credit groups. They are especially common in early long-horizon training, where agents often make partial progress but fail to complete the task.

MVPO addresses this failure by learning from viable failure prefixes. As illustrated in Figure 2, MVPO first merges semantically equivalent prefixes into viability regions using Union-Find, then estimates a prefix potential for each region from adaptive milestone progress, empirical future success, and loop statistics. During optimization, MVPO preserves anchor-based comparative credit when it is informative, and routes zero-credit groups to potential-based repair. The potential signal is strong in the sparse-reward stage and gradually fades as the policy improves. We analyze why MVPO can accelerate early optimization under sparse rewards theoretically in Appendix C.

## 4.1 Viability Potential Estimation

We construct a prefix potential that estimates whether an intermediate state remains on a promising path toward task completion. For a task instruction x and state $s _ { i , t }$ , we extract a compact progress signature

$$
z _ { i , t } = \psi ( x , s _ { i , t } ) ,\tag{8}
$$

where $\psi ( \cdot )$ maps raw states to abstract progress descriptors, such as task phase, acquired objects, satisfied constraints, observed evidence, or repeated loop patterns.

Union-Find viability regions. Since raw states can be noisy and sparse, MVPO groups semantically equivalent prefixes into viability regions. We maintain these regions with Union-Find. Two state occurrences are merged if they share the same progress signature or satisfy an environment-level equivalence rule:

$$
\begin{array} { r } { \left( i , t \right) \sim \left( j , k \right) \iff z _ { i , t } = z _ { j , k } } \\ { \mathrm { o r ~ M e r g e R u l e } ( s _ { i , t } , s _ { j , k } ) = 1 . } \end{array}\tag{9}
$$

The second condition allows the method to merge repeated loop states or states corresponding to the same task milestone beyond exact string equality. Let $c _ { i , t }$ denote the Union-Find representative of occurrence $( i , t )$ . We denote the occurrence set and trajectory set of region c as

$$
\begin{array} { r } { \mathcal { O } _ { k } ( c ) = \{ ( i , t ) \mid c _ { i , t } = c \} , } \\ { \mathcal { T } _ { k } ( c ) = \{ i \mid \exists t , c _ { i , t } = c \} . } \end{array}\tag{10}
$$

Adaptive milestone potential. For each state occurrence, we extract a milestone vector

$$
h _ { i , t } = ( h _ { i , t , 1 } , \ldots , h _ { i , t , M } ) \in [ 0 , 1 ] ^ { M } ,\tag{11}
$$

![](images/5b309bdfbfdd5d6b491610b4127e786affcdc6f1caa406bd056f79af608cec56.jpg)  
Figure 2: Overview of MVPO. MVPO first estimates prefix potential over Union-Find viability regions using future success, loop statistics, and adaptive milestone progress (§4.1). It then routes step-level credit according to anchor-group availability: informative anchor groups use anchor-based comparative credit, while zero-credit groups are repaired with prefix-potential advantages (§4.2). Finally, the routed step-level credit is combined with the episode-level group advantage for critic-free policy optimization (§4.3).

where each dimension indicates whether a taskrelevant milestone has been reached. For example, ALFWorld milestones include observing the target object, holding it, completing an operation, or reaching the target receptacle; WebShop milestones include reaching a search page, opening an item page, matching attributes, or purchasing an item. Details can be found in Appendix A.3.

Unlike subgoal-reward methods that manually assign dense rewards to predefined milestones, our milestones are not used as fixed rewards. They are only statistical features for estimating prefix viability, and their importance is learned from rollout data. Let $Y _ { i , t }$ denote a future-progress target, such as future maximum milestone progress or final success. The marginal utility of milestone m is estimated as

$$
\begin{array} { r l } & { \hat { u } _ { k , m } = \Big [ \mathbb { E } ( Y _ { i , t } \mid h _ { i , t , m } = 1 ) } \\ & { \qquad - \mathbb { E } ( Y _ { i , t } \mid h _ { i , t , m } = 0 ) \Big ] _ { + } . } \end{array}\tag{12}
$$

The normalized milestone weights are updated by EMA:

$$
w _ { k , m } = ( 1 - \beta _ { \mathrm { a m p } } ) w _ { k - 1 , m } + \beta _ { \mathrm { a m p } } \frac { \hat { u } _ { k , m } } { \sum _ { m ^ { \prime } } \hat { u } _ { k , m ^ { \prime } } + \epsilon } .\tag{13}
$$

Thus, MVPO learns which milestones are most predictive of future progress instead of relying on hand-crafted progress rewards.

For each region $c ,$ we compute its milestone profile, empirical future success, and loop frequency:

$$
\begin{array} { l } { \displaystyle \bar { h } _ { k , m } ( c ) = \frac { 1 } { | \mathcal { O } _ { k } ( c ) | + \epsilon } \sum _ { ( i , t ) \in \mathcal { O } _ { k } ( c ) } h _ { i , t , m } , } \\ { \displaystyle \bar { S } _ { k } ( c ) = \frac { 1 } { | \mathcal { T } _ { k } ( c ) | + \epsilon } \sum _ { i \in \mathcal { T } _ { k } ( c ) } \mathbb { I } [ R _ { i } \ge \epsilon _ { \mathrm { s u c c } } ] , } \\ { \displaystyle \bar { L } _ { k } ( c ) = \frac { 1 } { | \mathcal { O } _ { k } ( c ) | + \epsilon } \sum _ { ( i , t ) \in \mathcal { O } _ { k } ( c ) } \ell _ { i , t } , } \end{array}\tag{14}
$$

where $\ell _ { i , t }$ indicates whether the occurrence belongs to a repeated or looping region. The viability potential is defined as

$$
\begin{array} { r l } { \displaystyle \Phi _ { k } ( c ) = \mathrm { N o r m } \Big ( \displaystyle \sum _ { m = 1 } ^ { M } w _ { k , m } \bar { h } _ { k , m } ( c ) } & { } \\ { \displaystyle + \alpha _ { s } \bar { S } _ { k } ( c ) - \alpha _ { l } \bar { L } _ { k } ( c ) \Big ) . } \end{array}\tag{15}
$$

This formulation remains informative even when most rollouts fail, because milestone progress and loop suppression can still distinguish promising prefixes from unproductive ones.

To reduce noise in low-count regions, we smooth the potential toward the batch average:

$$
\begin{array} { l } { \displaystyle \tilde { \Phi } _ { k } ( c ) = \eta _ { k } ( c ) \Phi _ { k } ( c ) + ( 1 - \eta _ { k } ( c ) ) \bar { \Phi } _ { k } , } \\ { \displaystyle \eta _ { k } ( c ) = \frac { | \mathcal { O } _ { k } ( c ) | } { | \mathcal { O } _ { k } ( c ) | + \lambda _ { \mathrm { c n t } } } . } \end{array}\tag{16}
$$

For a transition $( s _ { i , t } , a _ { i , t } , s _ { i , t + 1 } )$ , the potentialbased progress advantage is

$$
A _ { i , t } ^ { \mathrm { p o t } } = \gamma \tilde { \Phi } _ { k } ( c _ { i , t + 1 } ) - \tilde { \Phi } _ { k } ( c _ { i , t } ) .\tag{17}
$$

It is positive when the transition moves to a more viable region, negative when it regresses or enters a dead end, and close to zero when no meaningful progress is made. We normalize $A ^ { \mathrm { p o t } }$ within each task group to stabilize scale.

## 4.2 Progress-Aware Credit Routing

The potential signal should only repair missing comparative credit, not replace reliable anchorbased credit. For occurrence (i, t), let

$$
n _ { i , t } = | \mathcal { T } _ { s _ { i , t } } | , \quad \sigma _ { i , t } = \mathrm { S t d } \left( \{ G _ { j , k } \} _ { ( j , k ) \in \mathcal { T } _ { s _ { i , t } } } \right) .\tag{18}
$$

We route each occurrence according to the availability of anchor-level return variation:

$$
A _ { i , t } ^ { \mathrm { r o u t e } } = \left\{ \begin{array} { l l } { A _ { i , t } ^ { \mathrm { a n c h o r } } , } & { n _ { i , t } > 1 , \sigma _ { i , t } > \epsilon _ { r } , } \\ { \kappa _ { k } \bar { A } _ { i , t } ^ { \mathrm { p o t } } , } & { n _ { i , t } > 1 , \sigma _ { i , t } \leq \epsilon _ { r } , } \\ { 0 , } & { n _ { i , t } = 1 . } \end{array} \right.\tag{19}
$$

Here, the first case preserves standard anchor credit when repeated anchors contain meaningful return variation; the second case repairs repeated zerocredit anchors with prefix-potential credit; and the third case keeps singleton states neutral because no repeated-state comparison is available. Thus, potential repair is applied only when comparative anchor evidence exists but collapses due to identical returns.

The coefficient $\kappa _ { k }$ controls the strength of potential repair. Since potential is most useful before the policy becomes competent, we attenuate it according to relative performance progress. Let $\bar { s } _ { k }$ be the EMA of batch success rate and $s _ { 0 }$ be the initial EMA success rate:

$$
\begin{array} { l } { \displaystyle { g _ { k } = \mathrm { c l i p } \left( \frac { \bar { s } _ { k } - s _ { 0 } } { 1 - s _ { 0 } + \epsilon } , 0 , 1 \right) , } } \\ { \displaystyle { \kappa _ { k } = \mathrm { c l i p } ( 1 - g _ { k } , \kappa _ { \mathrm { m i n } } , 1 ) . } } \end{array}\tag{20}
$$

Thus, potential repair is strong when the policy has made little progress and naturally fades as successful rollouts become more frequent.

The final repaired advantage is

$$
\hat { A } _ { i , t } = A _ { i } ^ { \mathrm { e p i } } + \omega A _ { i , t } ^ { \mathrm { r o u t e } } ,\tag{21}
$$

where $A _ { i } ^ { \mathrm { { e p i } } }$ is the episode-level group advantage and ω controls the contribution of routed step-level credit.

## 4.3 Policy Optimization

The repaired advantage $\hat { A } _ { i , t }$ is assigned to all tokens of textual action ${ a } _ { i , t }$ . For token $y _ { i , t , \ell }$ with context $c _ { i , t , \ell } .$ , the probability ratio is

$$
q _ { i , t , \ell } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , t , \ell } \mid c _ { i , t , \ell } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t , \ell } \mid c _ { i , t , \ell } ) } .\tag{22}
$$

We optimize the policy with a clipped objective:

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { M v p o } } ( \theta ) = \mathbb { E } _ { i , t , \ell } \Big [ \operatorname* { m i n } \big ( q _ { i , t , \ell } ( \theta ) \hat { A } _ { i , t } , } \\ { \mathrm { c l i p } ( q _ { i , t , \ell } ( \theta ) , 1 - \epsilon _ { c } , 1 + \epsilon _ { c } ) \hat { A } _ { i , t } \big ) \Big ] } \\ { - \beta \mathbb { E } _ { i , t , \ell } \Big [ D _ { \mathrm { K L } } \big ( \pi _ { \theta } ( \cdot \mid c _ { i , t , \ell } ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid c _ { i , t , \ell } ) \big ) \Big ] . } \end{array}\tag{23}
$$

The resulting algorithm remains critic-free and does not introduce a learned value model. Its key difference lies in how the step-level signal is constructed: MVPO preserves comparative credit when return variation is available and repairs zero-credit transitions with a progress-aware potential signal that fades as the policy improves.

## 5 Experiments and Results

In this section, we evaluate whether MVPO can improve long-horizon agent training by repairing zero-credit steps with prefix-potential signals. We first report the main results on ALFWorld and Web-Shop, and then conduct ablations to verify the contribution of each component. Due to the limited space, we evaluate MVPO on Search-Augmented QA Tasks in Appendix B.2.

## 5.1 Results on Long-Horizon Agentic Tasks

Experimental setup. We evaluate MVPO on two long-horizon agentic benchmarks: ALF-World (Shridhar et al., 2020) and WebShop (Yao et al., 2022a). For ALFWorld, we report the average success rate over six subtasks: Pick, Clean, Cool, Look, Heat, and Pick2. For WebShop, we report both the average task score and success rate. All results are averaged over 3 random seeds. We use Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (Hui et al., 2024) as trainable base policies, and compare MVPO with three groups of baselines: closed-source LLM agents, including GPT-5.4 and Claude Opus 4.7; prompt-based opensource agents, including direct prompting, Re-Act (Yao et al., 2022b), and Reflexion (Shinn et al., 2023); and RL training baselines, including PPO with a critic (Schulman et al., 2017), RLOO (Ahmadian et al., 2024), GRPO (Shao et al., 2024), and GiGPO (Feng et al., 2025). Unless otherwise specified, we follow the training configuration of GiGPO, with a rollout group size of 8 and the same training length for all trainable methods. For MVPO, we use $\omega = 0 . 5$ for routed step-level credit, set $\kappa _ { \mathrm { m i n } } = 0 . 0 5$ by default and 0.01 on WebShop, and use $\alpha _ { \kappa } = 0 . 0 5$ for EMA-based performanceprogress tracking. MVPO uses the w/std setting for group normalization, following $\mathrm { G i G P O _ { w / s t d } }$ Full hyperparameter settings and implementation details are provided in Appendix A. We further analyze training dynamics in Appendix B.1 and training-time cost in Section 5.3.

Table 1: Main results on ALFWorld and WebShop. For ALFWorld, we report success rates for six subtasks and the overall success rate. For WebShop, we report the average task score and success rate. Results are averaged over 3 random seeds. The best result of each group is in bold and the second-best result is underlined. $\mathrm { G i G P O _ { w / s t d } }$ uses $F _ { \mathrm { n o r m } } = \mathrm { s t d }$ , while $\mathrm { G i G P O _ { w / o } \ s t d }$ uses $F _ { \mathrm { n o r m } } = 1$ . MVPO uses the $w / s t d$ setting.
<table><tr><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td>Pick2</td><td>All</td><td>Score</td><td>Success</td></tr><tr><td colspan="8">Closed-source LLMs</td></tr><tr><td>GPT 5.4</td><td> $8 8 . 3 { \scriptstyle \pm 2 . 4 }$ </td><td> $5 2 . 3 _ { \pm 8 . 6 }$ </td><td> $5 6 . 9 { \scriptstyle \pm 2 . 0 }$ </td><td> $6 6 . 7 _ { \pm 8 . 6 }$ </td><td> $7 0 . 8 { \scriptstyle \pm 2 0 . 6 }$ </td><td> $5 6 . 3 { \scriptstyle \pm 8 . 4 }$ </td><td> $6 4 . 1 { \scriptstyle \pm 3 . 4 }$ </td><td> $9 . 3 { \scriptstyle \pm 1 . 1 }$ </td><td> $7 . 0 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>Claude Opus 4.7</td><td> $9 2 . 8 { \scriptstyle \pm 1 . 9 }$ </td><td> $8 1 . 7 _ { \pm 3 . 2 }$ </td><td> $7 3 . 6 { \scriptstyle \pm 2 . 0 }$ </td><td> $7 2 . 7 _ { \pm 0 . 0 }$ </td><td> $7 2 . 9 _ { \pm 1 0 . 6 }$ </td><td> $6 6 . 2 { \scriptstyle \pm 0 . 6 }$ </td><td> $7 7 . 1 { \pm } 1 . 0$ </td><td> $2 3 . 6 { \scriptstyle \pm 6 . 1 }$ </td><td> $1 9 . 8 { \scriptstyle \pm 4 . 8 }$ </td></tr><tr><td colspan="8">Qwen2.5-1.5B-Instruct</td></tr><tr><td>Prompting</td><td>5.9</td><td>3.3</td><td>4.2</td><td>5.5</td><td>9.7</td><td>0.0</td><td>4.1</td><td>23.1</td><td>5.2</td></tr><tr><td>ReAct</td><td>17.4</td><td>15.7</td><td>7.7</td><td>20.5</td><td>6.2</td><td>2.0</td><td>12.8</td><td>40.1</td><td>11.3</td></tr><tr><td>Reflexion</td><td>35.3</td><td>21.7</td><td>19.4</td><td>22.2</td><td>13.6</td><td>3.7</td><td>21.8</td><td>55.8</td><td>21.9</td></tr><tr><td>PPO</td><td> $6 4 . 8 { \scriptstyle \pm 3 . 5 }$ </td><td> $5 7 . 1 _ { \pm 4 . 9 }$ </td><td> $4 6 . 4 _ { \pm 4 . 0 }$ </td><td> $4 0 . 5 { \scriptstyle \pm 6 . 9 }$ </td><td> $6 0 . 6 { \scriptstyle \pm 6 . 6 }$ </td><td> $4 7 . 4 _ { \pm 1 . 9 }$ </td><td> $5 4 . 4 _ { \pm 3 . 1 }$ </td><td> $7 3 . 8 { \scriptstyle \pm 3 . 0 }$ </td><td> $5 1 . 5 { \scriptstyle \pm 2 . 9 }$ </td></tr><tr><td>RLOO</td><td> $8 8 . 3 { \scriptstyle \pm 3 . 0 }$ </td><td> $7 1 . 0 { \scriptstyle \pm 5 . 9 }$ </td><td> $6 6 . 4 _ { \pm 5 . 5 }$ </td><td> $5 2 . 8 { \scriptstyle \pm 8 . 6 }$ </td><td> $6 2 . 8 { \scriptstyle \pm 8 . 7 }$ </td><td> $5 6 . 9 { \scriptstyle \pm 4 . 7 }$ </td><td> $6 9 . 7 _ { \pm 2 . 5 }$ </td><td> $7 3 . 9 { \scriptstyle \pm 5 . 6 }$ </td><td> $5 2 . 1 _ { \pm 6 . 7 }$ </td></tr><tr><td>GRPO</td><td> $8 5 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $8 4 . 5 { \scriptstyle \pm 6 . 8 }$ </td><td> $5 9 . 7 _ { \pm 5 . 0 }$ </td><td> $5 3 . 7 _ { \pm 8 . 0 }$ </td><td> $7 8 . 2 { \scriptstyle \pm 7 . 9 }$ </td><td> $5 3 . 5 { \scriptstyle \pm 5 . 6 }$ </td><td> $7 2 . 8 { \scriptstyle \pm 3 . 6 }$ </td><td> $7 5 . 8 { \scriptstyle \pm 3 . 5 }$ </td><td> $5 6 . 8 { \scriptstyle \pm 3 . 8 }$ </td></tr><tr><td> $\mathrm { G i G P O _ { w / s t d } }$ </td><td> $9 4 . 4 { \scriptstyle \pm 5 . 9 }$ </td><td> $\mathbf { 9 4 . 8 _ { \pm 3 . 8 } }$ </td><td> $7 9 . 8 _ { \pm 4 . 7 }$ </td><td> $6 7 . 5 { \scriptstyle \pm 4 . 6 }$ </td><td> $9 4 . 4 { \scriptstyle \pm 7 . 8 }$ </td><td> $7 6 . 4 _ { \pm 5 . 4 }$ </td><td> $8 6 . 7 _ { \pm 1 . 7 }$ </td><td> $8 3 . 1 _ { \pm 1 . 6 }$ </td><td> $6 5 . 0 { \scriptstyle \pm 3 . 2 }$ </td></tr><tr><td> $\mathrm { G i G P O _ { w / o \ s t d } }$ </td><td> $\mathbf { 9 6 . 0 _ { \pm 1 . 4 } }$ </td><td> $9 1 . 8 { \scriptstyle \pm 5 . 5 }$ </td><td> $7 1 . 7 _ { \pm 8 . 4 }$ </td><td> $7 6 . 5 { \scriptstyle \pm 3 . 9 }$ </td><td> $9 1 . 3 { \scriptstyle \pm 6 . 3 }$ </td><td> $7 9 . 5 { \scriptstyle \pm 7 . 7 }$ </td><td> $8 6 . 1 _ { \pm 4 . 7 }$ </td><td> $8 3 . 5 { \scriptstyle \pm 1 . 8 }$ </td><td> $6 7 . 4 _ { \pm 4 . 5 }$ </td></tr><tr><td>MVPO</td><td> $9 2 . 2 { \scriptstyle \pm 1 . 5 }$ </td><td> $9 2 . 4 { \scriptstyle \pm 1 . 5 }$ </td><td> $\mathbf { 8 9 . 9 } _ { \pm 2 . 1 }$ </td><td> $7 5 . 6 _ { \pm 1 1 . 0 }$ </td><td> $\mathbf { 9 4 . 7 { \scriptstyle \pm 0 . 0 } }$ </td><td> ${ \bf 9 0 . 0 _ { \pm 4 . 1 } }$ </td><td> ${ \bf 9 1 . 1 { \bf _ { \pm 1 . 0 } } }$ </td><td> $\mathbf { 8 4 . 6 } _ { \pm 1 . 0 }$ </td><td> ${ \bf 7 0 . 3 _ { \pm 1 . 9 } }$ </td></tr><tr><td colspan="8"></td></tr><tr><td>Qwen2.5-7B-Instruct Prompting</td><td>33.4</td><td>19.3</td><td>2.8</td><td>21.6</td><td>6.9</td><td>3.2</td><td>14.8</td><td>26.4</td><td>7.8</td></tr><tr><td>ReAct</td><td>48.5</td><td>34.3</td><td>18.2</td><td>35.4</td><td>13.2</td><td>17.6</td><td>31.2</td><td>46.2</td><td>19.5</td></tr><tr><td>Reflexion</td><td>62.0</td><td>44.9</td><td>36.3</td><td>41.6</td><td>30.9</td><td>23.8</td><td>42.7</td><td>58.1</td><td>28.8</td></tr><tr><td>PPO</td><td> $9 2 . 3 _ { \pm 4 . 0 }$ </td><td> $9 2 . 5 { \scriptstyle \pm 2 . 4 }$ </td><td> $8 0 . 3 { \scriptstyle \pm 2 . 0 }$ </td><td> $6 4 . 0 { \scriptstyle \pm 8 . 4 }$ </td><td> $8 9 . 5 { \scriptstyle \pm 7 . 0 }$ </td><td> $6 8 . 8 { \scriptstyle \pm 8 . 3 }$ </td><td> $8 0 . 4 { \scriptstyle \pm 2 . 7 }$ </td><td> $8 1 . 4 _ { \pm 3 . 1 }$ </td><td> $6 8 . 7 _ { \pm 5 . 1 }$ </td></tr><tr><td>RLOO</td><td> $8 7 . 6 { \scriptstyle \pm 4 . 3 }$ </td><td> $8 7 . 3 { \scriptstyle \pm 5 . 8 }$ </td><td> $7 1 . 9 { \scriptstyle \pm 5 . 2 }$ </td><td> $7 8 . 2 { \scriptstyle \pm 8 . 3 }$ </td><td> $8 1 . 3 { \scriptstyle \pm 7 . 6 }$ </td><td> $4 8 . 9 { \scriptstyle \pm 8 . 4 }$ </td><td> $7 5 . 5 { \scriptstyle \pm 4 . 6 }$ </td><td> $8 0 . 3 { \scriptstyle \pm 3 . 2 }$ </td><td> $6 5 . 7 _ { \pm 4 . 0 }$ </td></tr><tr><td>GRPO</td><td> $9 0 . 8 { \scriptstyle \pm 5 . 1 }$ </td><td> $8 9 . 3 _ { \pm 5 . 4 }$ </td><td> $7 2 . 5 { \scriptstyle \pm 5 . 4 }$ </td><td> $6 6 . 1 { \scriptstyle \pm 6 . 7 }$ </td><td> $7 4 . 7 _ { \pm 6 . 9 }$ </td><td> $6 4 . 7 _ { \pm 7 . 3 }$ </td><td> $7 7 . 6 { \scriptstyle \pm 5 . 2 }$ </td><td> $7 9 . 3 { \scriptstyle \pm 2 . 8 }$ </td><td> $6 6 . 1 _ { \pm 3 . 7 }$ </td></tr><tr><td> $\mathrm { G i G P O _ { w / s t d } }$ </td><td> $9 7 . 7 { \pm } 1 . 6 $ </td><td> $9 8 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td> $\mathbf { 8 9 . 3 } _ { \pm 8 . 2 }$ </td><td> $8 2 . 7 { \scriptstyle \pm 7 . 9 }$ </td><td> $8 3 . 7 \pm 7 . 2$ </td><td> $7 9 . 2 { \scriptstyle \pm 6 . 6 }$ </td><td> $9 0 . 8 { \scriptstyle \pm 1 . 3 }$ </td><td> $8 4 . 4 _ { \pm 2 . 9 }$ </td><td> $7 2 . 8 { \scriptstyle \pm 3 . 2 }$ </td></tr><tr><td> $\mathrm { G i G P O _ { w / o \ s t d } }$ </td><td> $9 1 . 8 { \scriptstyle \pm 5 . 4 }$ </td><td> $9 5 . 9 _ { \pm 3 . 2 }$ </td><td> $\underline { { 8 6 . 5 _ { \pm 5 . 5 } } }$ </td><td> ${ \bf 8 8 . 6 _ { \pm 6 . 3 } }$ </td><td> $\underline { { 9 0 . 2 \pm 2 . 6 } }$ </td><td> $\underline { { 8 5 . 2 \pm 7 . 5 } }$ </td><td> $9 0 . 2 { \scriptstyle \pm 2 . 3 }$ </td><td> $8 6 . 2 { \scriptstyle \pm 2 . 6 }$ </td><td> $\underline { { 7 5 . 2 \pm 3 . 8 } }$ </td></tr><tr><td>MVPO</td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $\mathbf { 1 0 0 . 0 } _ { \pm 0 . 0 }$ </td><td> $6 8 . 1 \pm 2 . 1$ </td><td> $8 7 . 8 { \scriptstyle \pm 8 . 8 }$ </td><td> ${ \bf 9 1 . 2 } _ { \pm 2 . 5 }$ </td><td> $\mathbf { 9 6 . 7 \pm 2 . 4 }$ </td><td> $\mathbf { 9 1 . 9 } _ { \pm 1 . 3 }$ </td><td> $\mathbf { 8 6 . 6 } _ { \pm 2 . 1 }$ </td><td> $7 5 . 5 { \scriptstyle \pm 4 . 5 }$ </td></tr></table>

Experiment results. Table 1 reports the main results on ALFWorld and WebShop. We summarize three findings. 1) MVPO accelerates early sparsereward learning. Our method targets the zerocredit regime where GRPO and GiGPO cannot distinguish useful failure prefixes from unproductive ones. By routing such steps to prefix-potential advantages, MVPO provides learning signals before final success becomes frequent. As shown in Figure 1(c), MVPO reaches 40% ALFWorld success at step 52, earlier than GiGPO at step 62 and GRPO at step 73, showing that viable-prefix learning helps the policy escape the sparse-reward stage faster. 2) Better early credit leads to stronger final agents. Under the same training length, MVPO consistently improves over $\mathrm { G i G P O } _ { \mathrm { w / s t d } } .$ . With Qwen2.5- 1.5B-Instruct, it improves ALFWorld from 86.7% to 91.1% and WebShop success from 65.0% to 70.3%. With Qwen2.5-7B-Instruct, it improves ALFWorld from 90.8% to 91.9% and WebShop success from 72.8% to 75.5%. These gains indicate that the improvement comes from more informative credit construction rather than longer optimization. 3) The effect is consistent across environments and scales. MVPO improves both embodied household control in ALFWorld and webbased decision making in WebShop, and the gains hold for both 1.5B and 7B backbones. This supports our central hypothesis: long-horizon agent training benefits from identifying viable prefixes inside failed trajectories, rather than treating all

Table 2: Ablation study on WebShop with Qwen2.5- 1.5B-Instruct. We report success rate and task score, averaged over 3 random seeds. “Anchor” denotes GiGPO anchor credit, “Potential” denotes MVPO prefixpotential credit, and “κ Decay” denotes performanceprogress attenuation. Best results are bolded.
<table><tr><td>Method</td><td>Anchor</td><td>Potential κ Decay</td><td></td><td>Success</td><td>Score</td></tr><tr><td> $\mathrm { G i G P O _ { w / s t d } }$ </td><td>√</td><td>一</td><td>一</td><td> $\underline { { 6 5 . 0 \pm 3 . 2 } }$ </td><td> $8 3 . 1 _ { \pm 1 . 6 }$ </td></tr><tr><td>w/o GiGPO anchor</td><td>一</td><td>√</td><td>√</td><td> $\overline { { 6 3 . 5 _ { \pm 2 . 1 } } }$ </td><td> $7 9 . 8 _ { \pm 0 . 7 }$ </td></tr><tr><td>w/o κ decay</td><td>√</td><td>√</td><td>一</td><td> $6 0 . 4 _ { \pm 4 . 1 }$ </td><td> $8 3 . 4 _ { \pm 2 . 4 }$ </td></tr><tr><td>MVPO</td><td>√</td><td>√</td><td>√</td><td> ${ \bf 7 0 . 3 _ { \pm 1 . 9 } }$ </td><td> $\overline { { 8 4 . 6 _ { \pm 1 . 0 } } }$ </td></tr></table>

zero-return rollouts as equally uninformative.

## 5.2 Ablation Study

Experimental setup. We conduct ablation studies on WebShop with Qwen2.5-1.5B-Instruct. All variants follow the same training and evaluation protocol as the main experiments, and results are averaged over 3 random seeds. We compare four variants: $\mathrm { G i G P O } _ { \mathrm { w / s t d } } .$ , which uses only anchorbased comparative credit; w/o GiGPO anchor, which removes anchor credit and relies only on MVPO prefix-potential repair; w/o κ decay, which keeps both anchor and potential branches but disables performance-progress attenuation; and the full MVPO with all components enabled.

Experiment results. Table 2 reports the ablation results. We summarize three findings. 1) Potential repair needs anchor routing. Removing GiGPO anchor credit reduces success rate from 70.3% to 63.5% and task score from 84.6 to 79.8. This shows that prefix potential is effective for repairing zerocredit steps, but should not replace comparative anchor credit when return variation is available. Thus, MVPO benefits from routing: anchor credit handles reliable comparisons, while potential credit repairs missing signals. 2) Potential signals must fade after early learning. Removing κ decay drops success rate to 60.4%, even though task score remains 83.4. This suggests that keeping potential repair fully active throughout training can introduce noisy late-stage guidance. The result supports our design that prefix potential should mainly help the early sparse-reward stage and gradually give way to anchor-based credit as the policy improves. 3) The full method best balances the two signals. The full MVPO achieves the best success rate and task score, reaching 70.3% and 84.6, respectively. Compared with $\mathrm { G i G P O _ { w / s t d } }$ , it improves success by 5.3 points and task score by 1.5 points. This confirms our central claim: learning from viable failure prefixes is beneficial when potential credit

Table 3: Per-step training time. Gen. denotes rollout generation, Logp+Ref denotes old-policy and referencepolicy log-probability computation, Adv. denotes advantage construction, and Actor denotes policy update.
<table><tr><td>Env.</td><td>Method</td><td></td><td>Gen. Logp+Ref Adv. Actor</td><td></td><td></td><td>Total</td></tr><tr><td rowspan="3">ALFWorld</td><td>GRPO</td><td>183.3</td><td>19.5</td><td>0.5</td><td>35.5</td><td>283.4</td></tr><tr><td>GiGPO</td><td>207.9</td><td>16.6</td><td>0.7</td><td>30.7</td><td>303.9</td></tr><tr><td>MVPO</td><td>227.5</td><td>20.4</td><td>1.2</td><td>35.7</td><td>340.0</td></tr><tr><td rowspan="3">WebShop</td><td>GRPO</td><td>60.0</td><td>10.7</td><td>0.1</td><td>19.6</td><td>108.1</td></tr><tr><td>GiGPO</td><td>58.8</td><td>8.0</td><td>0.1</td><td>14.7</td><td>100.0</td></tr><tr><td>MVPO</td><td>61.2</td><td>9.4</td><td>0.3</td><td>17.3</td><td>107.9</td></tr></table>

is selectively routed to zero-credit groups and progressively attenuated.

## 5.3 Training Time Analysis

Table 3 reports the per-step wall-clock time under the same Qwen2.5-1.5B training setting. For a fair comparison, all methods are profiled under the same environment and hardware setup within each benchmark, i.e., ALFWorld methods are compared under the same ALFWorld setting and WebShop methods under the same WebShop setting. We decompose each step into rollout generation (Gen.), old-policy and reference-policy logprobability computation (Logp+Ref), advantage construction (Adv.), and actor update (Actor).

The only extra computation introduced by MVPO is in the Adv. stage, where Union-Find grouping, region statistics, and potential routing are performed. Compared with GiGPO, this stage increases by only 0.5s on ALFWorld and 0.2s on WebShop, corresponding to merely 0.16% and 0.20% of the total GiGPO step time, respectively. Therefore, MVPO introduces almost no additional training-time overhead while preserving the efficiency of critic-free group-based RL.

## 6 Conclusion

We identify zero-credit failure as a key bottleneck in group-based policy optimization for longhorizon LLM agents: failed rollouts may contain useful prefixes, yet GRPO and GiGPO assign no task-discriminative credit when returns are identical. We propose MVPO, which repairs zero-credit steps with prefix-potential advantages estimated over Union-Find viability regions and attenuated by performance progress. Experiments on ALFWorld, WebShop, and search-augmented QA show consistent gains with only 0.16%–0.20% advantagecomputation overhead.

## Limitations

Although MVPO improves sparse-reward agent training by repairing zero-credit steps, it still has two limitations. First, MVPO relies on meaningful prefix abstraction to estimate viability potentials. When observations are highly unstructured or task progress cannot be reliably mapped into region signatures, the Union-Find viability regions may become noisy, reducing the quality of potentialbased repair. Second, MVPO is most beneficial in long-horizon sparse-reward settings where failed trajectories contain recoverable prefixes. Its gains may be smaller in short-horizon tasks where final rewards are already frequent or in environments with very weak progress indicators. Our experiments focus on ALFWorld, WebShop, and search-augmented QA with Qwen2.5-based agents; broader evaluation on more diverse tool-use, multiagent, and real-world interactive settings remains future work.

## Ethics Statement

This work studies reinforcement learning methods for training long-horizon LLM agents in benchmark environments. The proposed MVPO improves policy optimization by allowing agents to learn from viable prefixes inside failed trajectories, thereby making sparse-reward training more effective. Our experiments are conducted on standard research benchmarks, including ALFWorld, Web-Shop, and search-augmented QA tasks. These environments are simulated or benchmarked settings, and our method does not introduce new external tools, collect private user data, or grant agents additional real-world permissions beyond the evaluated tasks.

Dual-use considerations. Improving long-horizon agent training can have beneficial applications, such as more reliable web navigation, embodied assistance, tool-use automation, and interactive problem solving. However, stronger autonomous agents may also be misused if connected to real-world tools, APIs, websites, or physical systems. Potential risks include unsafe tool execution, automated manipulation of online services, unintended goal pursuit, and optimization toward incomplete or misaligned reward signals. Because MVPO can make agents learn more efficiently from partial progress, it may also improve agents in settings where the intended task is harmful or policyviolating. We therefore recommend that deployment of such agents be restricted to well-scoped tasks with explicit safety constraints, access control, sandboxed execution, logging, and human oversight.

Reward and specification risks. MVPO estimates prefix potentials from rollout statistics and milestone progress. Although this improves sparsereward learning, it may amplify biases or errors in the environment reward, milestone definitions, or progress signatures. If the task reward is misspecified, the agent may learn to optimize intermediate states that correlate with success in the benchmark but are unsafe or undesirable in real deployment. Thus, prefix-potential credit should be used together with careful reward design, validation on held-out scenarios, and monitoring for reward hacking or shortcut behaviors.

Data and privacy. This work does not use private or sensitive user data. All evaluations are performed on public or standard research benchmarks. For search-augmented QA, the agent interacts with a controlled search interface following benchmark protocols. If similar methods are applied to real user data or live web environments, data minimization, privacy protection, consent, and secure logging should be enforced.

Human oversight and deployment. Our results should not be interpreted as evidence that trained agents are safe for unrestricted autonomous deployment. In real-world applications, agents should operate under explicit permission boundaries, with human approval for high-impact actions and failsafe mechanisms for abnormal behavior. Particularly for domains involving finance, healthcare, legal decisions, cybersecurity, or physical control, additional safety evaluations and domain-specific constraints are necessary before deployment.

Environmental impact. Training LLM agents with reinforcement learning can require substantial computation. MVPO is designed to remain criticfree and introduces only lightweight advantageconstruction overhead compared with existing group-based RL methods. Nevertheless, largescale agent training should consider energy efficiency, hardware utilization, and reproducibility, and future work should further study computeefficient variants.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ah-

met Üstün, and Sara Hooker. 2024. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, and 1 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114.

Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. 2025a. Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning. arXiv preprint arXiv:2505.16410.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, and 1 others. 2025b. Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for llm agent training. Advances in Neural Information Processing Systems, 38:46375–46408.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609– 6625.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, and 1 others. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186.

Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. 2025. Tree search for llm agent reinforcement learning. arXiv preprint arXiv:2509.21240.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings ofthe 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1601–1611.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. 2026. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. Advances in Neural Information Processing Systems, 38:61043–61068.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings ofthe 61st annual meeting of the associationfor computational linguistics (volume 1: Long papers), pages 9802–9822.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, and 1 others. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. 2024. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2020. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008– 3021.

Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. 2025. Zerosearch: Incentivize the search capability of llms without searching. arXiv preprint arXiv:2505.04588.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16022–16076.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, and 1 others. 2025a. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073.

Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu. 2025b. Stepsearch: Igniting llms search ability via stepwise proximal policy optimization. arXiv preprint arXiv:2505.15107.

Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, and 1 others. 2025. Webagentr1: Training web agents via end-to-end multi-turn reinforcement learning. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 7920–7939.

Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2024. Autodroid: Llmpowered task automation in android. In Proceedings of the 30th annual international conference on Mobile computing and networking, pages 543–557.

Ronald J Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022a. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022b. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong

Liu, Rui Men, An Yang, and 1 others. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, and 1 others. 2024a. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. 2024b. Archer: Training language model agents via hierarchical multi-turn rl. arXiv preprint arXiv:2402.19446.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

## A Experimental Details

## A.1 Details of Training

Hyperparameters for ALFWorld. All methods are configured with the same basic rollout and optimization settings for fair comparison. The maximum prompt length is 2048 tokens, and the maximum response length is 512 tokens. Each episode allows up to 50 environment steps. The actor learning rate is set to $1 \times 1 0 ^ { - 6 }$ . For PPO, which uses an additional critic, the critic learning rate is set to $1 \times 1 0 ^ { - 5 }$ . We adopt the standard sparse task reward: a reward of 10 is assigned for task success and 0 for failure, with a penalty of −0.1 for invalid actions. For group-based RL methods, we use a rollout group size of $N = 8$ and sample 16 groups per rollout batch, resulting in $1 6 \times 8 = 1 2 8$ parallel environments. PPO uses 128 separate rollout environments for a comparable batch size. The rollout temperature is set to 1.0, and the validation temperature is set to 0.4. The mini-batch size is 256, and the KL-divergence coefficient is 0.01. For step-level credit methods, the discount factor is set to $\gamma = 0 . 9 5$ . For MVPO, we set the routed stepcredit weight to $\omega = 0 . 5$ , the minimum potential strength to $\kappa _ { \mathrm { m i n } } = 0 . 0 5$ , and the EMA coefficient for performance-progress tracking to $\alpha _ { \kappa } = 0 . 0 5$ Hyperparameters for WebShop. We use the same general training protocol as ALFWorld. The maximum prompt length is 4096 tokens, and the maximum response length is 512 tokens. Each episode is limited to 15 environment steps. The actor learning rate is $1 \times 1 0 ^ { - 6 } .$ , and the critic learning rate for PPO is $1 \times 1 0 ^ { - 5 }$ . The task reward is sparse: 10 for success and 0 for failure, with a penalty of −0.1 for invalid actions. All group-based RL methods use a rollout group size of $N = 8$ and sample 16 groups per rollout batch, giving 128 parallel environments. PPO also uses 128 rollout environments. The rollout temperature is $1 . 0 ,$ and the validation temperature is 0.4. The mini-batch size is 64, and the KL-divergence coefficient is 0.01. For steplevel credit methods, we set $\gamma = 0 . 9 5$ . For MVPO, we use $\omega = 0 . 5$ and $\alpha _ { \kappa } = 0 . 0 5$ . The minimum potential strength is set to $\kappa _ { \mathrm { m i n } } = 0 . 0 1$ on Web-Shop, allowing the potential branch to fade more completely after the policy becomes stronger.

Implementation of MVPO. For each rollout batch, MVPO first extracts progress signatures from environment observations and builds Union-Find viability regions over state prefixes. For each region, we collect milestone statistics, empirical future-success statistics, and loop statistics to estimate the prefix potential. The routed advantage is then constructed by preserving anchor-based credit when return variation exists and using potentialbased repair when the anchor group is zero-credit. The performance-progress coefficient $\kappa _ { t }$ is updated from the EMA of batch success rate and controls how strongly the potential branch contributes to the final advantage. All these computations are performed during advantage construction and do not require an additional value model, reward model, or extra policy forward/backward pass.

Computing details. For the reported experiments, Qwen2.5-1.5B models are trained with 8 A100- 80G GPUs, and Qwen2.5-7B models are trained with 16 A100-80G GPUs. Each run is trained for 150 iterations, and all reported numbers are averaged over 3 random seeds. The 1.5B experiments can also be run with at least 2 A100-80G GPUs by reducing rollout parallelism and using gradient accumulation. Compared with GRPO and GiGPO, MVPO introduces only lightweight CPUside Union-Find grouping and region-statistics computation during advantage construction. It does not introduce a critic or additional model inference, so its overall computational cost is nearly identical to existing critic-free group-based RL methods.

Closed-source LLM evaluation. We evaluate closed-source LLM agents, including GPT-5.4 and Claude Opus 4.7, as reference models under the same task protocol as open-source agents. They are not fine-tuned or trained with RL. For both ALFWorld and WebShop, we use the same evaluation splits, prompt templates, admissible-action interface, and maximum interaction steps as the open-source evaluation. Specifically, ALFWorld allows at most 50 environment steps and WebShop allows at most 15 steps. The maximum response length is 512 tokens, and the decoding temperature is set to 0.4, matching the validation setting used for open-source agents. At each step, the model receives the same task instruction as open-source models. All closed-source models are evaluated via their official API during May 2026.

## A.2 Prompt Templates

ALFWorld prompt. For ALFWorld, we use the following prompt template with recent interaction history and admissible actions.

ALFWorld Prompt Template   
You are an expert agent operating in the ALFRED Em  
bodied Environment. Your task is to: {task\_description}   
Prior to this step, you have already taken {step\_count}   
step(s). Below are the most recent {history\_length} ob  
servations and the corresponding actions you took: {ac  
tion\_history}   
You are now at step {current\_step} and your current ob  
servation is: {current\_observation}   
Your admissible actions of the current situation are: [{ad  
missible\_actions}].   
Now it’s your turn to take an action.   
You should first reason step-by-step about the current   
situation. This reasoning process MUST be enclosed   
within <think> </think> tags.   
Once you’ve finished your reasoning, you should choose   
an admissible action for current step and present it within   
<action> </action> tags.

WebShop prompt. For WebShop, we use the following prompt template with recent interaction history and available actions.

WebShop Prompt Template

You are an expert autonomous agent operating in the WebShop e-commerce environment.

Your task is to: {task\_description}. Prior to this step, you have already taken {step\_count} step(s). Below are the most recent {history\_length} observations and the corresponding actions you took: {action\_history}. You are now at step {current\_step} and your current observation is: {current\_observation}. Your admissible actions for the current situation are: [{available\_actions}].

Now it’s your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.

Once you’ve finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

## A.3 Implementation Details of Prefix Abstraction

This section provides the exact prefix abstraction used in our experiments. Importantly, MVPO does not depend on a specific hand-crafted state representation or dense reward design. The algorithm only requires a lightweight interface that maps raw interaction histories into coarse progress regions and extracts a small set of progress indicators. In our experiments, we instantiate this interface with deterministic string-matching and structural rules for reproducibility. These rules can be replaced by other environment metadata, learned state abstractions, or task-specific parsers without changing the policy optimization algorithm.

Progress signatures. For each occurrence (i, t), we compute a compact progress signature $z _ { i , t }$ = $\psi ( x , s _ { i , t } )$ . The signature is used only to group prefixes into viability regions; it is not used as a reward.

For ALFWorld, we use

$$
\psi ( x , s ) = 1 0 \mathsf { c } _ { - } \mathsf { t y p e } \mid \mathsf { t o p } 2 _ { - } \mathsf { o b j e c t s } \mid \mathsf { d e p t h \_ b i n } .
$$

Here, loc\_type is extracted from the observation by regular expressions over room or receptacle names, top2\_objects denotes the first two object keywords matched by a fixed vocabulary, and depth\_bin $= ~ 3 \lfloor t / 3 \rfloor$ For example, an observation in the kitchen containing an apple and a knife at depth 6 is mapped to kitchen|apple+knife|d6.

For WebShop, we use

$$
\psi ( x , s ) = { \mathsf { p a g e \_ t y p e } } \mid { \mathsf { d e p t h \_ b i n } } ,\tag{25}
$$

where page\_type is determined by exact keyword matching:

Page Type Extraction Rule   
init Observation contains [search]   
search\_result Observation contains search results or [back to search], without [buy now]   
item\_detail Observation contains [buy now], without size/color option buttons   
item\_sub Observation contains [buy now] and size/color option buttons   
bought Observation contains your order or you have bought

For search-augmented QA, the horizon is at most four turns, so we do not apply depth binning. We use

$$
\psi ( x , s ) = \mathsf { d } t \mid \mathsf { i n f o } _ { 0 / 1 } \mid \mathsf { s r c h } _ { 0 / 1 } \mid \mathsf { a n s } _ { 0 / 1 } .\tag{26}
$$

Here, info indicates whether the history contains an <information> block, srch indicates whether the current action contains <search>, and ans indicates whether the current action contains <answer>. All flags are extracted by exact string matching.

Milestone features. For each occurrence, we extract a binary milestone vector $h _ { i , t } \in \{ 0 , 1 \} ^ { M }$ These milestones are not optimized as manually assigned dense rewards. Instead, they serve as progress features whose weights are adapted online by AMP according to their empirical association with future progress.

For ALFWorld, we use five milestones:

Milestone Trigger Condition   
h<sub>target</sub> Observation contains you pick up or you take with the target object name   
h<sub>operation</sub> Observation contains clean, cool, heat, slice, or their past-tense forms   
h<sub>place</sub> Observation contains you put or you move   
h<sub>success</sub> Environment reward is positive   
h<sub>invalid</sub> Observation contains nothing happens

For WebShop, we instantiate the same idea with page-level progress milestones:

Milestone Trigger Condition   
h<sub>search</sub> Page type is search\_result, item\_detail, item\_sub, or bought   
h<sub>item</sub> Page type is item\_detail, item\_sub, or bought   
h<sub>option</sub> Page type is item\_sub or bought   
h<sub>success</sub> Environment reward is positive or page type is bought   
h<sub>invalid</sub> Action is invalid or the page does not change after an invalid operation

For search-augmented QA, we use five tool-use milestones:

Milestone Trigger Condition   
h<sub>retrieved</sub> Action contains <search>   
h<sub>relevant</sub> Observation contains a non-empty <information> block longer than 10 characters   
h<sub>new\_info</sub> Same as h in the current implementation   
h<sub>answer</sub> Action contains <answer>   
h<sub>success</sub> Environment reward is greater than 0.5

For ALFWorld and WebShop, the initial milestone weights are

$$
\begin{array} { r } { w _ { 0 } = [ 0 . 2 , 0 . 3 , 0 . 5 , 1 0 . 0 , 0 . 0 ] . } \end{array}\tag{27}
$$

These values only initialize the AMP estimator. During training, AMP updates the weights from rollout statistics, so the effective milestone contribution is learned online rather than fixed by manual reward shaping.

Union-Find merge rule. MVPO first merges occurrences with identical progress signatures. In addition, we use a structural cycle rule to merge regions that belong to the same local loop. For each trajectory, we scan its region signatures in temporal order. If the same signature is revisited, all regions along the intervening path are merged:

$$
\mathrm { i f } \mathrm { s i g } _ { t } = \mathrm { s i g } _ { t ^ { \prime } } \mathrm { f o r } t ^ { \prime } < t ,
$$

$$
\begin{array} { r } { \mathrm { t h e n ~ u n i o n ~ a l l ~ a d j a c e n t ~ r e g i o n s ~ o n ~ s i g } _ { t ^ { \prime } } , \ldots , \mathrm { s i g } _ { t } . } \\ { ( 2 8 ) } \end{array}
$$

Equivalently, if an agent leaves a region and later returns to the same signature, the intermediate regions are treated as part of a reversible local loop and share a viability estimate. This rule is structural and does not use task success labels or manually assigned reward values.

Potential computation. For each Union-Find region C, we compute

$$
\Phi ( C ) = \bar { p } ( C ) + \delta \big ( r _ { \mathrm { r e c } } ( C ) - r _ { \mathrm { l o o p } } ( C ) \big ) ,\tag{29}
$$

where $\bar { p } ( C )$ is the average normalized milestone return-to-go over visits to region $C , \ r _ { \mathrm { r e c } } ( C )$ is the fraction of visits after which future milestone progress improves, and

$$
r _ { \mathrm { l o o p } } ( C ) = { \frac { 1 0 0 { \mathsf { p } } _ { - } \mathsf { c o u n t } ( C ) } { { \mathsf { v i s i t } } _ { - } \mathsf { c o u n t } ( C ) } } .\tag{30}
$$

We set $\delta \ : = \ : 0 . 2$ in all experiments. The potential is normalized within each rollout batch before computing the potential-difference advantage.

Generality. The above rules are used to make the experiments deterministic and reproducible, not to restrict MVPO to these environments. The method itself is agnostic to the specific form of $\psi ( \cdot )$ and h. It only assumes that prefixes can be mapped to coarse regions and that some weak progress indicators are available. Such indicators naturally exist in many agentic settings, including simulator states, admissible actions, browser page types, tool-call traces, retrieved evidence, or learned state embeddings. Moreover, milestone features are not used as fixed rewards; their weights are adapted online from rollout statistics. Therefore, MVPO should be understood as a general potential-routing framework for repairing zero-credit groups, with the above prefix abstractions serving as one reproducible instantiation.

## B Additional Results

## B.1 Training Dynamics Analysis

The main motivation of MVPO is that early longhorizon agent training contains many zero-credit groups: GRPO and GiGPO cannot distinguish useful failure prefixes from unproductive ones when observed returns have no variation. MVPO repairs this missing signal with prefix-potential credit, but this design requires two properties to hold during training. First, the potential branch should be strong when the policy is weak, and should gradually fade after the policy becomes competent. Second, the learned potential should remain directionally correct, assigning higher values to prefixes that are more likely to lead to success. Figure 3 verifies both properties.

Adaptive retreat of potential credit. Figure 3a analyzes the performance-progress attenuation coefficient $\kappa _ { t }$ . At the beginning of training, the policy is far from competent: the EMA success rate is only about 13% on ALFWorld and 3% on WebShop during the first several training steps. Accordingly, $\kappa _ { t }$ remains at 1.0, meaning that the potential branch is fully active. This behavior is desirable because zero-credit groups are most frequent in this stage, and the policy needs dense prefix-level signals to learn from partially successful failed trajectories.

![](images/aaa1e7ae3a74700066ef5b4bbd882c475a51f0ba9c66e7dbcab55e4b04cd3353.jpg)  
(a) κ<sub>t</sub> follows relative performance progress.

![](images/6c6405d9ad44e08b15d43e1c9378ea0559a6c7813f84c1815d7978f7dc082e41.jpg)  
(b) Prefix potential separates successful and failed trajectories.  
Figure 3: Training dynamics of MVPO. (a) The performance-progress attenuation mechanism works as intended. At the beginning of training, the EMA success rate is low and $\kappa _ { t }$ stays close to 1, allowing prefix-potential credit to repair zero-credit groups with full strength. As the policy improves, the EMA success rate increases and $\kappa _ { t }$ decreases accordingly, causing the potential branch to fade and leaving more gradient space to anchor-based comparative credit. (b) The learned prefix potential remains directionally meaningful throughout training. Successful trajectories maintain higher mean potential than failed trajectories, while the Union-Find region statistics remain stable. This supports the central assumption of MVPO: potential can distinguish viable failure prefixes from unproductive prefixes before final rewards become frequent.

As training progresses, the EMA success rate steadily increases. By step $1 5 0 ,$ it reaches approximately 82.5% on ALFWorld and 61.2% on Web-Shop. Following the definition of performance progress, the gain term increases and $\kappa _ { t }$ decreases to about 0.20 on ALFWorld and 0.40 on WebShop. This negative correlation between EMA success rate and $\kappa _ { t }$ is consistent across both environments, showing that the attenuation mechanism is driven by actual policy improvement rather than by a manually specified training schedule. In other words, MVPO does not require a hand-crafted decay midpoint or fixed step-based schedule: when the policy is weak, potential repair is strong; when the policy becomes stronger, potential repair naturally fades.

This behavior is important for the overall creditrouting design. Prefix potential is useful for repairing missing credit, but it is not intended to replace anchor-based comparison throughout training. Once successful continuations become more common, anchor groups contain richer return variation, and GiGPO-style comparative credit becomes more informative. The decay of $\kappa _ { t }$ therefore prevents potential signals from dominating late-stage optimization, allowing the final policy update to rely more on direct anchor evidence. This supports the ablation results in Table 2, where disabling κ decay substantially reduces success rate.

Directional quality of prefix potential. Figure 3b evaluates whether the learned prefix potential Φ(s) provides a meaningful progress signal during training. On WebShop, the mean potential of successful trajectories remains positive throughout training, decreasing from +0.44 at step 1 to +0.09 at step 150. In contrast, the mean potential of failed trajectories remains negative, from −0.02 at step 1 to −0.12 at step 150. The sign of this separation never flips, indicating that $\Phi ( s )$ consistently assigns higher viability to prefixes that eventually lead to success and lower viability to prefixes that remain unsuccessful.

This result directly supports the core assumption behind potential-based repair. When a zero-credit group provides no return variation, MVPO uses potential differences to decide whether a transition should receive positive or negative step-level credit. The observed separation between successful and failed trajectories shows that this signal is directionally aligned with future task completion. Thus, potential repair is not simply adding dense noise to the advantage estimator; it provides a structured progress signal that distinguishes viable prefixes from dead-end or irrelevant prefixes.

The decreasing absolute gap between successful and failed trajectories is also expected. The gap shrinks from roughly 0.46 early in training to about 0.20 near the end. This does not indicate that the potential becomes invalid. Instead, as the policy improves, even failed trajectories increasingly visit more plausible states before making their final mistakes. Consequently, the global potential distribution becomes compressed. At the same time, $\kappa _ { t }$ decreases as shown in Figure 3a, reducing the influence of the potential branch precisely when the potential margin becomes smaller. This coupling between potential quality and potential strength is central to MVPO: the method exploits prefix potential when it is most needed and reduces its effect when anchor-based credit becomes more reliable. Stability of Union-Find viability regions. Figure 3b also reports the stability of the region construction process. The batch success rate on Web-Shop increases from approximately 2.3% to 56%, while the effective grouping ratio remains stable around 70%–78%. This indicates that the Union-Find viability regions remain usable across the entire training process. Even when the policy changes and trajectories become more successful, the region abstraction continues to provide sufficient coverage for estimating prefix potential. This is important because MVPO relies on cross-rollout aggregation: individual failed trajectories may be sparse, but semantically equivalent prefixes can be merged into regions with more stable statistics.

Together, these results validate the mechanism of MVPO. Early in training, many rollouts fail and existing group-based methods often produce zero advantages. During this stage, $\kappa _ { t } \approx 1$ and prefix potential supplies dense repair signals for zerocredit groups. The potential itself is directionally correct, assigning higher values to prefixes that are more likely to lead to success. As training proceeds, success becomes more frequent, anchor-based comparative credit becomes more informative, and $\kappa _ { t }$ automatically decreases. Therefore, MVPO follows the intended behavior: it learns from viable failure prefixes when sparse rewards dominate, and gradually returns control to comparative anchor credit once the policy becomes competent.

## B.2 Evaluation on Search-Augmented QA Tasks

Experimental setup. We further evaluate MVPO on search-augmented QA tasks, where an agent must decide when to issue search queries and when to stop with a final answer. Following prior work, the model is trained on NQ (Kwiatkowski et al., 2019) and HotpotQA (Yang et al., 2018), and evaluated on both in-domain and out-ofdomain datasets. The single-hop evaluation includes NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al., 2023); the multi-hop evaluation includes HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), MuSiQue, and Bamboogle (Press et al., 2023). We use Qwen2.5-3B-Instruct (Hui et al., 2024) as the base policy. We compare MVPO with R1-Instruct (Guo et al., 2025), Search-R1 (Jin et al., 2025), ZeroSearch (Sun et al., 2025), StepSearch (Wang et al., 2025b), and GiGPO (Feng et al., 2025). All results are evaluated at training step 200. This setting provides a complementary testbed for MVPO because search-augmented QA has shorter horizons than ALFWorld and WebShop, but still requires sequential tool-use decisions under sparse final-answer rewards.

Training details. The maximum prompt length is 4096 tokens and the maximum response length is 512 tokens. The maximum interaction turn is 4. We use a rule-based reward, assigning 1 for a correct final answer and 0 otherwise, with a penalty of −0.01 for invalid actions. The actor learning rate is $1 \times 1 0 ^ { - 6 }$ , the training data size is 256, and the rollout group size is 5. The rollout and validation temperatures are 1.0 and 0.0, respectively. The mini-batch size is 512, the KL coefficient is 0.001, the step-credit weight is $\omega = 1$ , and the discount factor is $\gamma = 0 . 9 5$ . Experiments are trained on 8 A100 GPUs for 200 iterations.

Prompt template. We use the following template for search-augmented QA:

Search-Augmented QA Prompt Template You are an expert agent tasked with answering the given question step-by-step. Your question: {task\_description}. Prior to this step, you have already taken {step\_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History: {memory\_context} Now it’s your turn to respond for the current step. You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. After completing your reasoning, choose only one of the following actions: (1) If you lack some knowledge, call a search engine using: <search> your query </search>. (2) If you have enough knowledge to answer confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.

Experiment results. Table 4 reports the results on search-augmented QA tasks. Overall, MVPO improves the average score from 42.1% under GiGPO to 44.3%, yielding a +2.2 point gain. The improvement is especially clear on single-hop outof-domain datasets: MVPO improves TriviaQA from 59.5% to 61.8% and PopQA from 42.4% to 45.7%. It also improves NQ, HotpotQA, 2Wiki-MultiHopQA, and MuSiQue, indicating that prefixpotential repair can benefit search and answer decisions even in shorter-horizon tool-use tasks.

Table 4: Performance on search-augmented QA tasks. Models are trained on NQ and HotpotQA with Qwen2.5- 3B-Instruct. † and ∗ indicate in-domain and out-of-domain datasets, respectively. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Hop QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ†</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2Wiki*</td><td>MuSiQue*</td><td>Bamboogle*</td></tr><tr><td>R1-Instruct</td><td>27.0</td><td>53.7</td><td>19.9</td><td>23.7</td><td>29.2</td><td>7.2</td><td>29.3</td><td>27.1</td></tr><tr><td>Search-R1</td><td>34.1</td><td>54.5</td><td>37.8</td><td>32.4</td><td>31.9</td><td>10.3</td><td>26.4</td><td>32.5</td></tr><tr><td>ZeroSearch</td><td>41.4</td><td>57.4</td><td>44.8</td><td>27.4</td><td>30.0</td><td>9.8</td><td>11.1</td><td>31.7</td></tr><tr><td>StepSearch</td><td></td><td></td><td></td><td>34.5</td><td>32.0</td><td>17.4</td><td>34.4</td><td></td></tr><tr><td>GiGPO</td><td>42.0</td><td>59.5</td><td>42.4</td><td>36.9</td><td>37.0</td><td>12.6</td><td>64.1</td><td>42.1</td></tr><tr><td>MVPO</td><td>45.0</td><td>61.8</td><td>45.7</td><td>37.2</td><td>37.1</td><td>13.4</td><td>30.4</td><td>44.3</td></tr></table>

These results complement the main ALFWorld and WebShop experiments. Search-augmented QA has fewer interaction turns, so zero-credit prefixes are less severe than in embodied or web-shopping tasks. Nevertheless, the agent still receives sparse final-answer rewards and must learn which intermediate search actions are useful. The modest but positive gains suggest that MVPO is not limited to environment navigation: its core mechanism, learning from viable prefixes when final outcomes are sparse, also transfers to search-based reasoning agents. The smaller gain compared with ALF-World and WebShop is consistent with our story that MVPO is most beneficial in longer-horizon settings where failed trajectories contain more recoverable partial progress.

## C Theoretical Analysis

This section explains why MVPO can accelerate early optimization under sparse rewards. The key difference from GRPO and GiGPO is not the policy objective itself, but the construction of the step-level advantage. When a rollout or anchor group has identical returns, GRPO and GiGPO produce zero task-discriminative credit. In contrast, MVPO repairs such zero-credit steps with a potential-difference signal. We show that, as long as the prefix potential is positively aligned with future task progress, this repair introduces an additional positive policy-gradient component, yielding a faster expected improvement during the early sparse-reward stage.

## C.1 Zero-Credit Gradient Decomposition

Let J(θ) denote the expected task return of policy $\pi _ { \theta } ,$ and let $g _ { \mathrm { B } }$ be the policy-gradient estimator of a baseline group-based method, e.g., GRPO or GiGPO. For a sampled step (i, t), the baseline advantage is denoted as $A _ { i , t } ^ { \mathrm { B } }$ . For GRPO, this advantage is shared by all steps in a trajectory; for GiGPO, it further includes anchor-level credit when repeated anchor states have non-zero return variation.

Let Z be the set of zero-credit occurrences:

$$
{ \mathcal { Z } } = \{ ( i , t ) \mid A _ { i , t } ^ { \mathrm { B } } = 0 { \mathrm { ~ d u e ~ t o ~ z e r o ~ r e t u r n ~ v a r i a t i o n } } \} .\tag{31}
$$

For these steps, the baseline estimator contributes no task-discriminative gradient. MVPO repairs them with a potential advantage

$$
A _ { i , t } ^ { \mathrm { p o t } } = \gamma \Phi ( c _ { i , t + 1 } ) - \Phi ( c _ { i , t } ) ,
$$

and forms the routed estimator:

$$
\begin{array} { r l } & { g _ { \mathrm { M V P O } } = g _ { \mathrm { B } } + \Delta g _ { \mathrm { p o t } } , } \\ & { \Delta g _ { \mathrm { p o t } } = \omega \kappa \mathbb { E } _ { ( i , t ) \in \mathcal { Z } } \big [ A _ { i , t } ^ { \mathrm { p o t } } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { i , t } \mid s _ { i , t } , x ) \big ] . } \end{array}\tag{32}
$$

Thus, MVPO differs from GiGPO exactly on the zero-credit region: when anchor credit is informative, it preserves the baseline signal; when the baseline signal collapses, it injects prefix-potential credit.

## C.2 Expected Improvement Bound

We analyze a single policy update

$$
\theta ^ { + } = \theta + \eta g ,
$$

where $\eta$ is the learning rate. Assume $J ( \theta )$ is $L _ { - }$ smooth:

$$
J ( \theta + \eta g ) \ge J ( \theta ) + \eta \langle \nabla J ( \theta ) , g \rangle - \frac { L \eta ^ { 2 } } { 2 } \| g \| ^ { 2 } .\tag{33}
$$

This standard smoothness inequality says that the improvement is controlled by the alignment between the update direction and the true policy gradient, penalized by a second-order step-size term.

Let

$$
q = \operatorname* { P r } [ ( i , t ) \in \mathcal { Z } ]
$$

be the zero-credit step ratio. We assume that the learned prefix potential is positively aligned with true future progress on zero-credit steps:

$$
\begin{array} { r } { \mathbb { E } _ { ( i , t ) \in \mathcal { Z } } \Big [ A _ { i , t } ^ { \mathrm { p o t } } \langle \nabla J ( \theta ) , \nabla _ { \theta } \log \pi _ { \theta } ( a _ { i , t } \mid s _ { i , t } , x ) \rangle \Big ] \geq \rho , } \end{array}\tag{34}
$$

where $\rho > 0$ measures the quality of the potential signal. This assumption is exactly what Figure 1(b) and Figure 3 support empirically: high-potential prefixes are more likely to lead to future success, and successful trajectories have higher mean potential than failed trajectories.

Let

$$
B _ { \mathrm { p o t } } = \lVert \Delta g _ { \mathrm { p o t } } \rVert , \quad B _ { \mathrm { B } } = \lVert g _ { \mathrm { B } } \rVert .
$$

Then the one-step improvement of MVPO over the baseline satisfies the following bound.

Proposition 1: Potential Repair Improves Early Update Speed

Under the smoothness condition in Eq. (33) and the positive-alignment condition in Eq. (34), the expected one-step improvement of MVPO over a baseline group-based estimator satisfies

$$
\begin{array} { r l } & { \eta \omega \kappa q \rho - L \eta ^ { 2 } \left( B _ { \mathrm { B } } B _ { \mathrm { p o t } } + \cfrac { 1 } { 2 } B _ { \mathrm { p o t } } ^ { 2 } \right) } \\ & { \leq \mathbb { E } [ \Delta J _ { \mathrm { M V P 0 } } - \Delta J _ { \mathrm { B } } ] } \\ & { \leq \eta \omega \kappa q \rho _ { \mathrm { m a x } } + L \eta ^ { 2 } \left( B _ { \mathrm { B } } B _ { \mathrm { p o t } } + \cfrac { 1 } { 2 } B _ { \mathrm { p o t } } ^ { 2 } \right) , } \end{array}\tag{35}
$$

where $q$ is the zero-credit step ratio, κ is the potential strength, $\omega$ is the step-credit weight, and $\rho _ { \mathrm { m a x } }$ upper-bounds the potential-gradient alignment.

Interpretation. The lower bound shows that the additional gain of MVPO increases with three factors: (i) the zero-credit ratio $q ,$ (ii) the potential strength κ, and (iii) the alignment quality $\rho .$ This explains why MVPO is most beneficial early in training: zero-credit groups are frequent, κ is close to 1, and potential repair supplies gradients where GRPO and GiGPO provide none. The second-order term is small when the learning rate is small and the potential gradient is bounded, so the first-order positive term dominates.

The upper bound also clarifies why MVPO should not keep potential repair fully active forever. As training progresses, zero-credit groups become less dominant and anchor-based return variation becomes more informative. If potential credit remained large, the second-order term could introduce unnecessary late-stage noise. This motivates the performance-progress attenuation:

$$
\kappa _ { k } = \mathrm { c l i p } ( 1 - g _ { k } , \kappa _ { \mathrm { m i n } } , 1 ) ,
$$

which reduces $B _ { \mathrm { p o t } }$ and the potential contribution as the policy improves.

## C.3 Optimization-Speed Consequence

Let $\mu _ { \mathrm { B } }$ denote the expected per-step performance increase of a baseline method in a given training phase. From Proposition 1, the expected per-step increase of MVPO is lower-bounded by

$$
\mu _ { \mathrm { M v P O } } \geq \mu _ { \mathrm { B } } + \eta \omega \kappa q \rho - L \eta ^ { 2 } \left( B _ { \mathrm { B } } B _ { \mathrm { p o t } } + \frac { 1 } { 2 } B _ { \mathrm { p o t } } ^ { 2 } \right) .\tag{36}
$$

For any target success level $S ^ { \star }$ , let $S _ { 0 }$ be the initial success rate. Ignoring higher-order stochastic fluctuations, the number of training steps required to reach $S ^ { \star }$ is bounded by

$$
\begin{array} { r l } & { \mu _ { \mathrm { M V P 0 } } = \displaystyle \mu _ { \mathrm { B } } + \eta \omega \kappa q \rho - L \eta ^ { 2 } \left( B _ { \mathrm { B } } B _ { \mathrm { p o t } } + \frac { 1 } { 2 } B _ { \mathrm { p o t } } ^ { 2 } \right) , } \\ & { T _ { \mathrm { M V P 0 } } ( S ^ { \star } ) \lesssim \displaystyle \frac { S ^ { \star } - S _ { 0 } } { \mu _ { \mathrm { M V P 0 } } } . } \end{array}
$$

Compared with the baseline hitting time

$$
T _ { \mathrm { B } } ( S ^ { \star } ) \approx \frac { S ^ { \star } - S _ { 0 } } { \mu _ { \mathrm { B } } } ,\tag{37}
$$

MVPO reaches the same target faster whenever the positive potential-repair term exceeds the secondorder penalty.

## Main Theoretical Takeaway

MVPO is expected to update faster than GRPO and GiGPO in the early sparse-reward stage because it converts zero-credit prefixes into aligned potential-gradient signals. The improvement is largest when the zero-credit ratio q is high and κ ≈ 1, and it naturally fades as q decreases and performance-progress attenuation reduces $\kappa .$

Table 5: Empirical verification of the theoretical prediction on ALFWorld 1.5B. We report validation success rates at representative training steps. $\Delta _ { \mathrm { G i } }$ denotes the improvement of MVPO over GiGPO.
<table><tr><td>Step</td><td>GRPO</td><td>GiGPO MVPO</td><td> $\Delta _ { \mathrm { G i } }$ </td></tr><tr><td>20</td><td>20.3</td><td>23.4 28.1</td><td>+4.7</td></tr><tr><td>30</td><td>22.7</td><td>19.5 32.0</td><td>+12.5</td></tr><tr><td>50</td><td>31.2</td><td>32.8 47.7</td><td>+14.9</td></tr><tr><td>60</td><td>21.1</td><td>37.5 57.0</td><td>+19.5</td></tr><tr><td>70</td><td>40.6</td><td>60.2 60.9</td><td>+0.7</td></tr><tr><td>90</td><td>46.9</td><td>76.6 78.1</td><td>+1.5</td></tr></table>

## C.4 Empirical Verification on ALFWorld

The theory predicts three observable patterns: (1) MVPO should improve faster in the early stage when zero-credit groups are frequent; (2) the advantage should be most visible before GiGPO obtains enough successful anchor variation; (3) late-stage performance should approach GiGPO as κ decays and anchor credit becomes reliable.

Table 5 verifies these predictions using ALF-World 1.5B training dynamics.

The early-stage results strongly match the bound in Eq. (35). At step 20, MVPO already outperforms GiGPO by +4.7 points. The gap grows to +12.5 points at step 30, +14.9 points at step 50, and +19.5 points at step 60. This is exactly the regime where q is large: many rollouts still fail, making GRPO and GiGPO unable to assign credit to useful prefixes. MVPO instead obtains a positive extra term through potential repair.

As training proceeds, GiGPO begins to receive more informative anchor-level return variation, while the potential branch in MVPO is attenuated by performance progress. Consequently, the gap narrows after step 70. At step 90, MVPO remains slightly ahead of GiGPO (78.1 vs. 76.6). This behavior is also consistent with the theory: MVPO is designed to accelerate the sparse-reward stage rather than permanently override anchorbased credit. Thus, the empirical dynamics support the theoretical explanation that potential-routed repair improves optimization speed when zero-credit groups dominate, and then fades as comparative anchor credit becomes reliable.

## D Pseudo Code of MVPO

Algorithm 1 presents the training procedure of MVPO.

Algorithm 1 MVPO Training Procedure   
1: Input: training task set $\mathcal { D } = \{ { x } \}$ , policy π , reference policy $\pi _ { \mathrm { r e f } } .$ , environment $\varepsilon ,$ group size $N ,$ maximum iterations   
$K$ , discount factor $\gamma ,$ step-credit weight ω, routing threshold $\epsilon _ { r } ,$ AMP update rate $\beta _ { \mathrm { a m p } } ,$ success and loop weights $\alpha _ { s } , \alpha _ { l } ,$   
count-smoothing coefficient $\lambda _ { \mathrm { { c n t } } } .$ minimum potential strength $\kappa _ { \mathrm { m i n } } .$ , PPO clipping threshold $\epsilon _ { c } ,$ KL coefficient $\beta .$   
2: Initialize: policy parameters θ, milestone weights $\{ w _ { 0 , m } \} _ { m = 1 } ^ { M }$ , initial success-rate EMA $s _ { 0 } ,$ current success-rate EMA   
$\begin{array} { r } { \bar { s } _ { 0 }  s _ { 0 } . } \end{array}$   
3: \*\*\* MVPO training begins \*\*\*   
4: for $k = 1$ to K do   
5: Set old policy $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta } .$   
6: for each task description $x \in \mathcal { D }$ do   
7: \*\*\* Step A: Group rollout collection. \*\*\*   
8: Sample N trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { N }$ from $\tau _ { \theta _ { \mathrm { o l d } } }$ in environment $\varepsilon .$   
9: Compute trajectory returns $\{ R _ { i } \} _ { i = 1 } ^ { N }$ and batch success rate $\mathrm { S R } _ { k }$   
10: Update success-rate EMA $\bar { s } _ { k }$   
11: Compute episode-level advantages $A _ { i } ^ { \mathrm { { e p i } } }$ by Eq. (2).   
12: \*\*\* Step B: Anchor construction. \*\*\*   
13: for each trajectory $\tau _ { i }$ do   
14: for each step $i = 1 , \ldots , T _ { i }$ do   
15: Compute future return $G _ { i , t }$ by Eq. $( 3 ) .$   
16: Add occurrence $( i , t )$ to anchor set $\mathcal { T } _ { s _ { i , t } } .$   
17: end for   
18: end for   
19: Compute anchor advantages $A _ { i , t } ^ { \mathrm { a n c h o r } }$ for all anchor groups by Eq. (5).   
20: \*\*\* Step C: Union-Find viability potential estimation. \*\*\*   
21: for each trajectory $\tau _ { i }$ do   
22: for each step $i = 1 , \ldots , T _ { i }$ do   
23: Extract progress signature $z _ { i , t } = \psi ( x , s _ { i , t } )$ by Eq. (8).   
24: Extract milestone vector $h _ { i , t }$ by Eq. (11).   
25: Compute loop indicator $\ell _ { i , t }$   
26: end for   
27: end for   
28: Initialize Union-Find over all state occurrences.   
29: for each pair or candidate pair of occurrences $( i , t ) , ( j , k )$ do   
30: $\mathbf { i f } \ z _ { i , t } = z _ { j , k }$ or MergeRule $( s _ { i , t } , s _ { j , k } ) = 1$ then   
31: Merge $( i , t )$ and $( \bar { j } , k )$ in Union-Find by Eq. (9).   
32: end if   
33: end for   
34: Obtain region representative $c _ { i , t }$ for each occurrence.   
35: Construct $\mathcal { O } _ { k } ( c )$ and $\mathcal { T } _ { k } ( c )$ for each region by Eq. (10).   
36: Estimate milestone utilities $\{ \hat { u } _ { k , m } \} _ { m = 1 } ^ { M }$ by Eq. (12).   
37: Update AMP milestone weights $\{ w _ { k , m } \} _ { m = 1 } ^ { M }$ by Eq. (13).   
38: for each viability region c do   
39: Compute region statistics $\bar { h } _ { k , m } ( c ) , \bar { S } _ { k } ( c ) .$ , and $\bar { L } _ { k } ( c )$ by Eq. (14).   
40: Compute viability potential $\Phi _ { k } ( c )$ by Eq. (15).   
41: Apply count smoothing to obtain $\tilde { \Phi } _ { k } ( c )$ by Eq. (16).   
42: end for   
43: for each occurrence (i, t) do   
44: Compute potential advantage $A _ { i , t } ^ { \mathrm { p o t } }$ by Eq. (17).   
45: end for   
46: Normalize potential advantages within the task group to obtain $\bar { A } _ { i , t } ^ { \mathrm { p o t } }$   
47: <sup>i,t</sup> \*\*\* Step D: Progress-aware credit routing. \*\*\*   
48: Compute relative performance gain g<sub>k</sub> and potential strength κ<sub>k</sub> by Eq. (20).   
49: for each occurrence (i, t) do   
50: Select the anchor, potential, or neutral branch using $n _ { i , t }$ and $\sigma _ { i , t }$ in Eq. (19).   
51: Compute routed credit $A _ { i , t } ^ { \mathrm { r o u t e } }$ by Eq. (19).   
52: Compute repaired advantage $\hat { A } _ { i , t }$ by Eq. (21).   
53: Assign $\hat { A } _ { i , t }$ to all tokens of action $_ { a _ { i , t } }$   
54: end for   
55: \*\*\* Step E: Policy optimization. \*\*\*   
56: Optimize $\pi _ { \theta }$ with the clipped objective in Eq. (23).   
57: end for   
58: end for   
59: Output: trained policy π<sub>θ</sub>.