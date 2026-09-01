# Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence

Zhiqin Yang\*<sup>1</sup>, Jingwen Fu\*<sup>2</sup>, Yuhan Liu\*<sup>3</sup>, Hengyu Liu\*<sup>4</sup>, Yonggang Zhang\*<sup>1</sup>, Kainan Cao<sup>5</sup>, Zizhuo Zhang<sup>6</sup>, Chenxin Li<sup>7</sup>, Ruibin Yuan<sup>1</sup>, Jiahao Pan<sup>1</sup>, Jiankai Sun<sup>4</sup>, Zhenyuan Zhang<sup>1</sup>, Yibo Li<sup>8</sup>, Yunlong Lin<sup>9</sup>, Jing Xiong<sup>5</sup>, Sida Lin<sup>1</sup>, Bo Han<sup>6†</sup>, Wei Xue<sup>1†</sup>, Yike Guo<sup>1†</sup>

<sup>1</sup> The Hong Kong University of Science and Technology <sup>2</sup> Zhongguancun Academy <sup>3</sup> Xi’an Jiaotong University <sup>4</sup> The Chinese University of Hong Kong <sup>5</sup> The University of Hong Kong <sup>6</sup> Hong Kong Baptist University <sup>7</sup> Hunyuan Tencent <sup>8</sup> National University of Singapore <sup>9</sup> Xiamen University

![](images/90bda3c97fc756c1047feb62a8c801a617b8927f5fbb99ef95c830fbd69bcb1e.jpg)

GitHub: https: //github.com/visitworld123/Awesome-Scaling-LRM-Beyond-Human-Supervision

Recent advances in large reasoning models (LRMs) have shown that reinforcement learning with verifiable rewards (RLVR) can substantially improve reasoning in mathematics and code, where outcomes can be checked automatically. Extending this progress to open-ended and agentic tasks remains dificult because reliable rewards are harder to obtain and direct human supervision cannot keep pace with the scale and complexity of model-generated experience. This paper studies how LRMs can continue to improve as human supervision gradually recedes from the learning loop. We examine two connected dimensions of this problem. The reward axis traces the development from per-instance human judgments to reusable verifiers and rewards that operate even without human feedback. The experience axis examines how learning can progress from human-curated tasks and environments toward self-generated curricula, constructed environments, and autonomous co-evolution. We connect these dimensions through a five-level ladder from L0 to L4 that identifies which parts of the learning process remain under continued human control. Our analysis further highlights the risks introduced by increasingly autonomous rewards and experience generation, including reward hacking, feedback drift, curriculum collapse, and environment errors. Consequently, we also provide the evaluation around three complementary objects: policy capability, feedback fidelity, and experience quality. This analysis provides a structured account of current approaches to scaling LRMs beyond human supervision and the open problems involved in developing self-sustaining learning systems toward superintelligence. Furthermore, we maintain a continuously updated GitHub repository to track the latest advances.

\* Core contribution <sup>†</sup> Corresponding author Contact information: zqccc.yang@connect.ust.hk

## 1. Introduction

The emergence of large reasoning models (LRMs) (Hui et al., 2024; Yang et al., 2025a; Li et al., 2025k; Chen et al., 2025e; Achiam et al., 2023; Comanici et al., 2025; Team et al., 2025a,b; Sun et al., 2025b) has substantially advanced the reasoning capabilities of language models. Recent systems combine large-scale pretraining with reinforcement learning and increased inference-time computation. Models such as DeepSeek-R1 (Guo et al., 2025a; Shao et al., 2024) and OpenAI o1 (Jaech et al., 2024) have achieved strong performance in mathematics and code generation, where answers can be checked or programs can be executed to produce informative training signals. These advances suggest that further progress depends not only on model scale and pretraining data, but also on the ability to generate trajectories, obtain feedback, and learn from interaction. This shift motivates the Experience Era (Silver & Sutton, 2025), in which experience becomes an increasingly important source of capability improvement.

Reinforcement learning with verifiable rewards (RLVR) provides a strong foundation for learning from experience. Mathematical answers can be compared with known solutions, while generated code can be compiled and tested automatically (Yu et al., 2025c; Zhou et al., 2025f; Lin et al., 2025b; Deng et al., 2025). As a result, large numbers of rollouts can be evaluated without asking a person to inspect each response. However, deterministic verification is unavailable for many tasks that matter in practice. Creative writing (Lu, 2025), open-ended question answering (Yu et al., 2025d; Ma et al., 2025), and complex agent interactions may admit several valid outcomes whose quality depends on context, preferences, or long-term consequences. Learned reward models and language-model judges extend feedback to these settings, but their scores remain proxies for the intended objective. Moreover, direct human evaluation becomes increasingly costly as reasoning traces grow longer and interactions span more steps (Casper et al., 2023). When tasks approach or exceed expert dificulty, evaluators may also require model assistance to identify errors and assess outputs reliably (Saunders et al., 2022). Consequently, the challenge is not simply to replace an expensive label with a cheaper score, but to sustain meaningful improvement when human judgment and human design can no longer cover the full stream of learning experience.

How can large reasoning models continue to improve when human supervision can no longer keep pace with the scale and dificulty of their experience?

We use scaling beyond human supervision to describe a shift in which fewer components of the learning process require continued human provision. Human intent remains essential and may be expressed through objectives, initial data, tools, environments, safety constraints, and independent audits. The central change lies in the recurring efort required during learning, as models increasingly need to obtain useful feedback and encounter informative experience without depending on people to evaluate or design every training instance. We study this shift through two connected axes. The reward axis concerns the evidence used to evaluate behavior, while the experience axis concerns the tasks and environments that produce the trajectories from which models learn.

Along the reward axis, verifiable tasks provide the most direct route to scalable feedback because correctness is grounded in answer checking or execution. For tasks without exact answers, learned reward models, rubrics, and language-model judges can encode evaluative criteria and apply them across many trajectories (Seed et al., 2025; Ren et al., 2025a; Liu et al., 2025j; Su et al., 2025). Other methods derive rewards from model-internal certainty, agreement across sampled responses, or consistency with available references (Prabhudesai et al., 2025; Zuo et al., 2025b; Zhang et al., 2025o; Tang et al., 2025; Zhou et al., 2025b; Yu et al., 2025d; Liu et al., 2025h). When success produces observable changes in an environment, interaction outcomes ofer another source of grounding. Together, these approaches broaden the range of experiences that can support reinforcement learning. At the same time, feedback becomes more vulnerable to miscalibration, correlated errors, and reward exploitation as it depends less on independent evidence. Scaling reward acquisition must therefore be accompanied by careful analysis of what grounds each signal and how that grounding behaves under optimization.

Reliable feedback alone does not ensure continued progress because learning also depends on the experience to which that feedback is applied. Initially, models may train on human-written tasks in fixed environments. They can subsequently generate reasoning traces, synthesize instructions, or adjust task dificulty through proposer-and-solver interaction (Fang et al., 2025b; Yu et al., 2025b). More integrated approaches construct executable environments, propose tasks from live interfaces, and retain reusable information from previous interactions (Yang et al., 2025b; Yu et al., 2025a; Cai et al., 2025). These mechanisms allow the curriculum to change with the policy and can expose weaknesses that a fixed dataset no longer reveals. Nevertheless, generated experience is useful only when tasks remain diverse, solvable, and relevant, and when synthesized environments preserve the dynamics and consequences of the target domain. The experience axis therefore shifts attention from the quantity of sampled trajectories to the quality and sustainability of the process that produces them.

![](images/98bd3f351782d48e2155a698dcb5ff85f58d6dee5e03329fa8b6fb464829d2d6.jpg)  
Figure 1: The receding-supervision ladder. Human involvement moves from judgments supplied for individual instances toward learning loops that increasingly obtain rewards and produce experience with less continued human provision. The ladder describes operational responsibility during learning and does not imply the absence of human-originated objectives, data, tools, or environments.

The two axes ultimately converge because a generated task cannot support learning unless its outcome can be evaluated, while a scalable reward has limited value if the model repeatedly encounters familiar or uninformative situations. Continued improvement therefore requires rewards and experience to develop together. In a mature learning loop, stronger policies reveal new weaknesses, those weaknesses guide the generation of new tasks and environments, and reliable evaluation turns the resulting trajectories into further policy updates. Such co-evolution ofers a possible route to capability growth beyond the bandwidth and expertise of human supervisors. However, it also creates tightly coupled failure modes because policies, generators, and evaluators may adapt to one another while performance under independent criteria stagnates or declines.

To organize this progression, we introduce a five-level ladder from L0 to L4 in Figure 1. The ladder begins with per-instance human supervision, proceeds through reusable human-grounded evaluation and rewards obtained beyond human evaluation, and then extends to experience generated beyond continued human design. Its final level represents an idealized loop in which policies, rewards, tasks, and environments co-evolve. By identifying which components still depend on continued human provision, the ladder characterizes operational responsibility during learning without serving as a measure of capability, reliability, alignment, or the historical amount of human knowledge embedded in the learning infrastructure.

Our paper develops this account in three ways. First, it formulates scaling beyond human supervision through the connected reward and experience axes and introduces a common framework for locating continued human provision within the learning loop. Second, it organizes existing methods according to how they obtain rewards and produce experience, while examining the assumptions and failure modes associated with decreasing human involvement. Third, it develops an evaluation framework that distinguishes policy capability, feedback fidelity, and experience quality. This distinction also informs our discussion of reliable rewards, sustainable curricula, valid environments, eficient learning, and safe autonomous improvement.

The remainder of the paper develops this account in stages. Section 2 introduces the formal setting and the receding-supervision ladder. Section 3 studies the sources and grounding of rewards as direct human evaluation recedes. Section 4 examines task generation, environment construction, and their eventual co-evolution with the policy. Section 5 presents datasets, benchmarks, and protocols for evaluating capability, feedback, and experience. Finally, Section 6 discusses the principal challenges and research directions, and Section 7 summarizes the scope and limitations of our analysis.

## 2. Preliminaries

This section introduces the notation and learning framework used throughout the paper. We first describe single-turn reasoning and sequential interaction within a common MDP and POMDP formulation. We then examine how reward can move from direct human supervision toward signals obtained from models, references, and environments. After introducing the policy-optimization objectives used in later sections, we define the uncertainty measures required by model-derived reward methods. These elements support the five-level ladder presented at the end of the section, which connects reward beyond human evaluation with experience beyond human design. Table 1 summarizes the notation in our paper.

## 2.1. From Single-Turn MDP to Sequential POMDP

Learning from experience can be represented as a Markov decision process (MDP) (Sutton et al., 1998), denoted by $M : = ( S , { \mathcal { A } } , { \mathcal { P } } , { \mathcal { R } } , \gamma )$ . At step �, the policy $\pi _ { \theta }$ selects an action $\pmb { a } _ { t } \in \mathcal { A }$ from state $\pmb { s } _ { t } \in S$ according to $\mathbf { \sigma } _ { \mathbf { \alpha } _ { t } } \sim \pi _ { \theta } ( \cdot \mid s _ { t } )$ . For an LRM, the action may contain a reasoning trace $z _ { t }$ together with an answer or tool command $\mathbf { } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf } _ { } { \mathbf } _ { } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf } { } _ { \mathbf } { } _ { \mathbf } _ { } { \mathbf } _ { } \mathbf { } _ { \mathbf } _ { }  _ { \mathbf { } } _ { \mathbf } _ { } _ { \mathbf } _ { } _ { \mathbf } _ { } _ { \mathbf } { } _ { \mathbf } _ { } _ { \mathbf }$ . The transition function determines the next state through $\mathbf { \boldsymbol { s } } _ { t + 1 } \sim \mathcal { P } ( \cdot \mid \mathbf { \boldsymbol { s } } _ { t } , \mathbf { \boldsymbol { a } } _ { t } )$ while the reward function assigns a scalar signal $r _ { t } = \mathcal { R } ( s _ { t } , \pmb { a } _ { t } )$ . Multiple evaluation criteria may contribute to this scalar reward. The discount factor $\gamma \in [ 0 , 1 ]$ controls the contribution of future rewards.

Table 1: Comprehensive notation table for large reasoning models and its corresponding descriptions.
<table><tr><td>Symbol Description</td><td></td></tr><tr><td> $\pi _ { \theta }$ </td><td>Policy model (large reasoning model) parameterized by  $\theta .$ </td></tr><tr><td> $\mathcal { M }$ </td><td>Markov Decision Process (MDP) tuple  $\{ S , { \mathcal { A } } , { \mathcal { R } } , { \mathcal { P } } , \gamma \} .$ </td></tr><tr><td> $s$ </td><td>State space.</td></tr><tr><td> $\mathcal { A }$ </td><td>Action space, representing the model&#x27;s response  $\pmb { a } _ { t }$  at time  $t .$ </td></tr><tr><td> $\pmb { a } _ { t }$ </td><td>Model&#x27;s action at time t, comprising reasoning  $z _ { t }$  and final answer</td></tr><tr><td> $\mathbf { } _ { \mathbf { } } \mathbf { s } _ { t }$ </td><td>State at time  $t ,$  including full history of interactions.</td></tr><tr><td> $\pmb { s } _ { 0 }$ </td><td>Initial state, user&#x27;s query, question, or request  $\pmb q .$ </td></tr><tr><td> $\mathbf { \delta } _ { \pmb { t } }$ </td><td>Observation at time t in the partially observable (POMDP) setting.</td></tr><tr><td> $\mathcal { R }$ </td><td>Reward function mapping state-action pair to a scalar or vector.</td></tr><tr><td> $\mathcal { R } _ { \phi }$ </td><td>Model-based reward model parameterized by  $\phi .$ </td></tr><tr><td> $\pi _ { \nu }$ </td><td>Verifier (e.g., an LLM-as-a-judge) that assigns rewards to rollouts.</td></tr><tr><td> $\mathcal { P }$ </td><td>Transition function governing state evolution.</td></tr><tr><td> $\gamma$ </td><td>Discount factor in MDP.</td></tr><tr><td> $\tau$ </td><td>Experience trajectory, a multi-turn interaction process.</td></tr><tr><td> $T$ </td><td>Total length of the experience trajectory.</td></tr><tr><td>q</td><td>User&#x27;s query, question, or request.</td></tr><tr><td>Ⅱ</td><td>Task (query) distribution from which  $\pmb q$  is drawn.</td></tr><tr><td> $z _ { t }$ </td><td>Reasoning component of action at time t.</td></tr></table>

Continued on next page

Table 1 (continued)
<table><tr><td colspan="2">Table 1 (continued)</td></tr><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathbf { } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf } _ { } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf } { \mathbf } _ { } _ { \mathbf { } } _ { \mathbf } { } _ { \mathbf } _ { } { \mathbf } _ { } _ { \mathbf } { } _ { \mathbf } _ { } _ { \mathbf } { } _ { \mathbf } _ { } _ { \mathbf } _ { } $ </td><td>Final answer component of action at time  $t .$ </td></tr><tr><td>D</td><td>Dataset for training or evaluation.</td></tr><tr><td> $y ^ { * }$ </td><td>Ground truth answer for query  $\pmb q .$  t</td></tr><tr><td> $\pmb { a } _ { i }$ </td><td>i-th rollout, comprising reasoning b  $z _ { i }$  and answer  $y _ { i }$ </td></tr><tr><td> $z _ { i }$ </td><td>Reasoning trace for the i-th rollout.</td></tr><tr><td> $\boldsymbol { y } _ { i }$ </td><td>Final answer for the i-th rollout.</td></tr><tr><td> $G$ </td><td>Number of rollouts sampled per query.</td></tr><tr><td> $\hat { A } _ { i }$ </td><td>Relative advantage based on rewards across rollouts.</td></tr><tr><td> $\pi _ { \theta _ { \mathrm { o l d } } }$ </td><td>Old policy model for comparison.</td></tr><tr><td> $\varepsilon$ </td><td>Clipping parameter in GRPO optimization.</td></tr><tr><td> $\beta$ </td><td>Coefficient for KL divergence penalty.</td></tr><tr><td> $\pi _ { \mathrm { r e f } }$ </td><td>Reference policy model.</td></tr><tr><td> $\mathbb { D } _ { \mathrm { K L } }$ </td><td>KL divergence between policies.</td></tr><tr><td> $\mathcal { T } _ { \mathrm { G R P O } }$ </td><td>Objective function for Group Relative Policy Optimization.</td></tr><tr><td> $\varepsilon$ </td><td>Environment for LLM interaction.</td></tr><tr><td> $_ \mathrm { ~ \textit ~ { ~ N ~ } ~ }$ </td><td>Vocabulary over which tokens are generated.</td></tr><tr><td> $H _ { j } ( \pi _ { \theta } )$ </td><td>Trajectory-level entropy of the policy.</td></tr><tr><td> $H _ { t } ( \pi _ { \theta } )$ </td><td>Token-level entropy of the policy.</td></tr><tr><td> $\pmb { o }$ </td><td>Output (completion) sequence.</td></tr><tr><td> $\mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta } \mathbf { \sigma } \mathbf { \delta \sigma } \mathbf { \delta } \mathbf { \sigma \delta } \mathbf { \sigma } \mathbf { \delta \sigma } \mathbf { \delta } \mathbf { \sigma \delta } \mathbf { \delta } \mathbf { \sigma \delta } \mathbf { \delta } \mathbf { \sigma \delta } \mathbf { \delta \sigma } \mathbf { \delta \sigma } \mathbf { \delta \delta } \mathbf \mathbf { \sigma } \delta \mathbf { \delta \sigma } \mathbf { \delta \delta } \mathbf \mathbf { \delta } \mathbf \mathbf { \delta \sigma } \mathbf { \delta \delta \delta \mathbf } \mathbf \mathbf  \delta \delta \mathbf { \sigma } \delta \delta \mathbf \mathbf { \delta } \delta \mathbf \mathbf \delta \mathbf { \delta } \delta \mathbf \delta \mathbf \delta \mathbf \delta \mathbf \mathbf  \delta \delta \delta \delta \mathbf \delta \delta \mathbf \delta \mathbf \delta \delta \delta \mathbf \delta \mathbf \delta \delta \delta \mathbf \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \mathbf \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta \delta $ </td><td>Sequence of tokens before time t.</td></tr></table>

The MDP formulation assumes that the policy observes the state needed for decision-making. Many agentic settings are only partially observable. A Partially Observable Markov Decision Process (POMDP) therefore augments the formulation with an observation space $o$ and an observation model. The environment retains a latent state $\mathbf { } _ { \mathbf { } } \mathbf { s } _ { t }$ , while the model receives an observation $\mathbf { } _ { \pmb { \theta } _ { t } }$ and selects its action from the interaction history $\pmb { h } _ { t } = \left( \pmb { o } _ { 0 } , \pmb { a } _ { 0 } , \ldots , \pmb { o } _ { t } \right)$ . GUI applications, tool systems, and conversations commonly have this structure because the model observes only part of the underlying environment state.

An experience trajectory � records the interaction between the model and its environment E. For a horizon �, the underlying trajectory can be written as $\tau = ( s _ { 0 } , \pmb { a } _ { 0 } , r _ { 0 } , \ldots , s _ { T - 1 } , \pmb { a } _ { T - 1 } , r _ { T - 1 } )$ , although the policy in a POMDP acts from $\mathbf { } _ { \mathbf { } ^ { h _ { t } } }$ rather than directly from $\mathbf { } _ { \mathbf { } } \mathbf { s } _ { t }$ . Given a task distribution Π and an initial-state distribution $\rho _ { 0 } ( \cdot \mid q )$ , the learning objective is:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { q \sim \Pi , s _ { 0 } \sim \rho _ { 0 } ( \cdot | q ) , \tau \sim p _ { \theta } ( \cdot | s _ { 0 } , \mathcal { E } ) } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } \mathcal { R } ( s _ { t } , a _ { t } ) \right] .\tag{1}
$$

The horizon � places single-turn reasoning and sequential interaction within the same objective. When �=1 and $\gamma { = } 1$ , a query produces one reasoning and answer trajectory, followed by an outcome reward. This setting captures common RLVR formulations in mathematics and code. When $T { > } 1$ , the model acts repeatedly in $\varepsilon ,$ and later outcomes depend on earlier decisions. Then the partial observability and long-horizon credit assignment become the central. Therefore, the single-turn setting is a special case of the broader sequential formulation.

## 2.2. The Reward Function as the Object of Scaling

Equation 1 separates the source of experience from the signal used to evaluate it. Among the components of M, the reward function R provides the most immediate starting point for tracing human supervision. In a common RLVR setting, a rule-based procedure compares the model output with an instance-specific answer $y ^ { * }$ or test suite supplied with the task. In preference-based training, a learned reward model $\mathcal { R } _ { \phi }$ or verifier �<sub>�</sub> applies criteria derived from human judgments. The first setting retains human influence in the target attached to each task. The second transfers human judgments into a reusable evaluator. Their diference therefore concerns the provenance and reusability of the evaluation criterion rather than when the score is produced.

Other reward sources reduce dependence on human evaluation in diferent ways. Model-derived methods use properties of the policy distribution, such as certainty or agreement among sampled answers. Reference-dependent methods automate scoring through an available target, while environment-grounded methods use consequences such as execution results, formal acceptance, game outcomes, or observable state changes. These signals may still depend on human-created references, rules, or environments. The relevant transition is that human judgment no longer serves as the primary mechanism for evaluating each new trajectory. Section 3 organizes these methods by reward provenance and external grounding. Section 2.5 connects this transition with the source of tasks and environments.

## 2.3. Policy Optimization

Once a reward source has been specified, policy optimization determines how that signal updates the model. For the single-turn case, let D denote an empirical approximation to the task distribution Π. Reward-driven methods optimize

$$
\mathcal { T } ( \theta ) = \mathbb { E } _ { \pmb { q } \sim \mathcal { D } , \pmb { a } \sim \pi _ { \theta } ( \cdot | \pmb { q } ) } \left[ \mathcal { R } ( \pmb { q } , \pmb { a } ) \right] ,\tag{2}
$$

and estimate its gradient as (Sutton et al., 1998)

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { T } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { q } \sim \mathcal { D } , \boldsymbol { a } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \vert \boldsymbol { q } ) } \left[ \hat { A } ( \boldsymbol { q } , \boldsymbol { a } ) \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { q } ) \right] .\tag{3}
$$

The advantage estimate $\hat { A } ( { \pmb q } , { \pmb a } )$ weights the log-likelihood gradient according to performance relative to a baseline. Reward provenance and optimizer design play diferent roles in this expression. The reward source defines ${ \mathcal { R } } .$ , while the optimizer determines how advantages are estimated and how far the policy may move during an update. PPO and GRPO illustrate two widely used choices: critic-based and critic-free methods.

PPO. Proximal Policy Optimization (PPO) (Schulman et al., 2017) commonly uses a learned value function $V _ { \psi }$ as its baseline. It estimates the token-level advantage $\hat { A } _ { t }$ and limits each update through the importance ratio $\begin{array} { r } { r _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { t } | s _ { t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { t } | s _ { t } \right) } } \end{array}$

$$
\mathcal { T } _ { \mathrm { P P O } } ( \theta ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( r _ { t } ( \theta ) \hat { A } _ { t } , \ \mathrm { c l i p } ( r _ { t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { t } \right) \right] .\tag{4}
$$

This clipped objective discourages large policy changes that could destabilize training. In common LRM implementations, PPO also requires training and storing a value model comparable in size to the policy, which adds substantial memory and computation costs.

GRPO. Group Relative Policy Optimization (GRPO) (Guo et al., 2025a) replaces the learned value baseline with statistics computed from a group of sampled rollouts. Given a query ${ \pmb q } ,$ it samples � rollouts $\pmb { a } _ { i } = \left( z _ { i } , \mathbf { y } _ { i } \right)$ and assigns each rollout a reward $r _ { i }$ . The normalized group-relative advantage is

$$
\hat { A } _ { i } = \frac { r _ { i } - \mathrm { m e a n } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } .\tag{5}
$$

Using the group as an implicit baseline avoids a separate value network. The resulting clipped objective is

$$
\begin{array} { l } { \displaystyle \mathcal { J } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \ \{ a } _ { i } ) _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o d } } } ( \cdot \vert q ) }  \\ { \displaystyle \left\{ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \vert a _ { i } \vert } \sum _ { t = 1 } ^ { \vert a _ { i } \vert } \left( \operatorname* { m i n } \left[ r _ { i , t } ( \theta ) \widehat { A } _ { i } , \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \widehat { A } _ { i } \right] - \beta \widehat { D } _ { \mathrm { K L } , i , t } \right) \right\} , } \end{array}\tag{6}
$$

where $\begin{array} { r } { r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { i , t } | \pmb q , \pmb { a } _ { i , < t } \right) } { \pi _ { \theta _ { 0 } | \mathrm { d } } \left( a _ { i , t } | \pmb q , \pmb { a } _ { i , < t } \right) } } \end{array}$ is the token-level importance ratio. The sampled KL estimator

$$
\widehat { D } _ { \mathrm { K L } , i , t } = \frac { \pi _ { \mathrm { r e f } } ( a _ { i , t } | { \pmb q } , \pmb { a } _ { i , < t } ) } { \pi _ { \theta } ( a _ { i , t } | { \pmb q } , \pmb { a } _ { i , < t } ) } - \log \frac { \pi _ { \mathrm { r e f } } ( a _ { i , t } | { \pmb q } , \pmb { a } _ { i , < t } ) } { \pi _ { \theta } ( a _ { i , t } | { \pmb q } , \pmb { a } _ { i , < t } ) } - 1\tag{7}
$$

penalizes deviation from the reference policy $\pi _ { \mathrm { r e f } } .$ . By combining group-relative advantages with clipped updates and KL regularization, GRPO provides a critic-free objective that is widely used in LRM training.

## 2.4. Uncertainty Measures

The optimizer in Eq. 2 does not require rewards to come from an external verifier. Several methods in Section 3 instead derive model-internal rewards from uncertainty over generated trajectories. Two quantities used by these methods require a clear distinction. Sequence surprisal aggregates the negative log-likelihood of a sampled completion �:

$$
U _ { \mathrm { s e q } } ( \pmb { \sigma } , \pi _ { \theta } ) = - \frac { 1 } { | \pmb { \sigma } | } \sum _ { t = 1 } ^ { | \pmb { \sigma } | } \log \pi _ { \theta } ( \pmb { \sigma } _ { t } \mid \pmb { \sigma } _ { < t } ) .\tag{8}
$$

Lower sequence surprisal means that the sampled completion is more probable under the policy, but it does not establish that the completion is correct. Predictive token entropy instead measures uncertainty over the full next-token distribution:

$$
H _ { t } ( \pi _ { \theta } ) = - \sum _ { \nu \in \mathcal { V } } \pi _ { \theta } ( \nu \mid \pmb { \sigma } _ { < t } ) \log \pi _ { \theta } ( \nu \mid \pmb { \sigma } _ { < t } ) .\tag{9}
$$

Lower token entropy indicates a more concentrated distribution at a particular generation step. A method may aggregate $H _ { t }$ across a trajectory, with its normalization determining how sequence length afects the result. Sequence surprisal and predictive entropy can both serve as internal signals, but neither should be interpreted as a correctness probability without additional calibration or grounding.

## 2.5. A Five-Level Ladder of Receding Supervision

The framework separates two questions: where the feedback used to improve a policy comes from, and who determines the tasks and environments that produce its experience. The reward axis addresses the first question, beginning with direct human judgments and extending to reusable evaluators and other sources of feedback. The experience axis addresses the second, covering the shift from fixed, human-designed training settings to experience that is generated or discovered during learning. Figure 1 groups this progression into five regimes according to which parts of the learning loop still require ongoing human involvement. The classification concerns how learning is sustained, not whether the models, data, tools, or environments were originally created by humans. We then formulate this five-level ladder as follows:

L0 Per-instance Human Supervision. Each training instance is paired with a human-provided answer, preference, or judgment that serves as its evaluation target.

L1 Human-grounded Evaluation. Human criteria are encoded in reusable reward models, rubrics, or language-model judges, allowing the same evaluative standard to be applied across many trajectories.

L2 Reward beyond Human Evaluation. Rather than relying primarily on human judgment, training derives rewards from model confidence, agreement across samples, available references, or observable environment outcomes, while tasks and environments remain largely externally provided.

L3 Experience beyond Human Design. The training distribution becomes adaptive as tasks, curricula, and environments are generated, discovered, selected, or reorganized during learning, while humans continue to set broad goals, constraints, and initial conditions.

![](images/7506f7f173f48b9579ec5eecf2074493320852cbf8f704385df3b60119a3aff1.jpg)  
Figure 2: Representative methods along the receding-supervision ladder. For brevity, the figure uses “Human-free Reward” and “Human-free Experience” for regimes in which continued learning no longer depends primarily on human evaluation or human-designed experience. These labels do not imply the complete absence of human influence.

L4 Autonomous Co-evolution. Feedback, experience, and policy improvement adapt together in a persistent loop, while human involvement is concentrated at the level of intent, safety constraints, and independent oversight.

The ladder classifies supervision sources rather than entire training pipelines. Standard RLHF spans L0 and L1 e.g., human preferences provide L0 supervision, while the resulting reward model serves as a reusable L1 evaluator. The boundaries between these levels are not always sharp because a method may reduce human involvement in one component while retaining it in another. For example, reward computation can be automated even when the reference answer was written by an expert. A method may also generate new tasks while continuing to use an environment designed by humans. We therefore classify each method by the part of the learning loop that becomes less dependent on human input, while noting the human input it still requires.

Each step away from direct human supervision also introduces a diferent source of error. Rewards based on model confidence or agreement can reinforce mistakes shared across the model’s outputs. Rewards based on tests or environment outcomes can favor unintended behavior when the evaluation rules are incomplete. Automatically generated tasks and environments may be invalid, too easy, or too dificult, while self-play may repeatedly produce a narrow range of experience. Sections 3.4 and 4.5 examine these failure modes together with the mechanisms proposed to mitigate them.

## 3. Reward beyond Human Evaluation

The reward axis introduced in Section 2.5 concerns the evidence used to evaluate a rollout. At Per-Instance Human Supervision, each task comes with a human-provided answer, preference, or acceptance criterion. Human-Grounded Evaluation turns such judgments into reusable evaluators that can score many outputs. Reward beyond Human Evaluation goes one step further by deriving learning signals from the model, an available reference, or observable consequences in an environment. These regimes difer in the source of evaluation. Whether the policy is optimized online or ofline is a separate design choice.

![](images/9cf005b2bc9e0511f4437fd2d22a87323e15a303006b8f851a5db714eed45082.jpg)  
Figure 3: Taxonomy of reward sources beyond per-instance human supervision. Human-grounded evaluators reuse human criteria. Beyond this regime, reward evidence comes from model statistics, stored references, or observable environment outcomes. Self-trained evaluators form a boundary between them.

## 3.1. Per-Instance Human Supervision

Per-instance human supervision associates each training instance with an evaluation target supplied by people or experts. In preference-based training, this target may be a comparison or quality judgment, while verifiable tasks may use a reference answer, formal specification, or task-specific test suite. This regime corresponds to L0 in our ladder. Once the target is available, reward computation can be automatic. Extending training to new tasks nevertheless requires new targets, so automatic scoring alone does not remove per-instance supervision (Ouyang et al., 2022; Guo et al., 2025a).

DeepSeekMath (Shao et al., 2024) and DeepSeek-R1 (Guo et al., 2025a) demonstrate the efectiveness of rule-based rewards when outputs can be checked automatically. Mathematical answers can be extracted and compared with a ground-truth answer through equivalence or matching rules. Generated programs can be compiled and executed against predefined tests. This approach has driven substantial progress in mathematical reasoning (Yu et al., 2025c; Lin et al., 2025b; Mu et al., 2025; Liu et al., 2026c) and code

generation (Guo et al., 2025a).

The limitation appears as the amount and dificulty of experience increase. Every new task still needs an answer, test, preference, or comparable evaluation criterion. Expert domains make these targets expensive to produce, and open-ended tasks present a deeper dificulty because several responses may be valid under context-dependent standards. Scaling reward beyond this regime consequently requires either reusing human judgment across many instances or drawing evaluation from a diferent source, corresponding to L1 and L2.

## 3.2. Human-Grounded Evaluation

At L1, human preferences, expert annotations, or explicit criteria are encoded in a reusable evaluator. A learned reward model $\mathcal { R } _ { \phi }$ or LLM judge $\pi _ { \nu }$ can apply the same standard across many queries � (Ouyang et al., 2022; Zhu et al., 2025a). Unlike exact matching, these evaluators can assign graded scores, assess intermediate reasoning, and evaluate responses without a unique reference answer. Their reach is broader than that of task-specific rules, but their standards remain grounded in the data and criteria provided by people. DeepSeekMath V2 illustrates this regime in natural-language theorem proving, where a generative verifier assesses correctness, completeness, and rigor and rewards the resolution of identified flaws (Shao et al., 2025b).

The need for a learned evaluator depends on the evaluation procedure, not on a fixed division between domains. Some task instances admit inexpensive deterministic checks, including short-form mathematics, executable code, and constraint-based puzzles (Cobbe et al., 2021; Chen et al., 2021; Lin et al., 2025a). Fully open-ended generation often lacks both a unique target and a rule-based checker, so evaluation relies on human preferences or learned proxies (Ouyang et al., 2022; Stiennon et al., 2020; Zheng et al., 2023). Between these cases are semi-verifiable tasks that support partial checking, verifiable reformulation, instruction-specific criteria, or reference-conditioned scoring (Tang et al., 2025; Zhang et al., 2025e; Viswanathan et al., 2025; Zhou et al., 2025b; Yu et al., 2025d; Liu et al., 2025h). As evaluation relies more heavily on learned proxies, bias, misspecification, reward hacking, and overoptimization become more consequential (Casper et al., 2023; Skalse et al., 2022; Gao et al., 2023).

## 3.2.1. Outcome and Process Reward Models

Learned reward models can supplement sparse outcome checks with partial-credit or process-level feedback. This is useful when a correct response may appear in an unexpected form or when the final outcome reveals little about where a reasoning trace failed. Process reward models and step-wise critiques were first developed extensively for mathematical reasoning (Uesato et al., 2022; Wang et al., 2024a; Lightman et al., 2023). Their central challenge is to obtain reliable step labels without exhaustive human annotation.

Math-Shepherd addresses this problem through automatic process annotation (Wang et al., 2024a). For a reasoning prefix ${ \boldsymbol { z } } _ { \leq t }$ , a completer model samples � continuations, and the step receives the fraction that reach the reference answer $\boldsymbol { y } ^ { * }$ :

$$
\mathcal { R } ( z _ { \leq t } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbb { I } \{ { \pmb { y } } _ { j } = { \pmb { y } } ^ { * } \} , \qquad { \pmb { y } } _ { j } \sim \pi _ { \theta } ( { \bf \cdot } \mid { \pmb { q } } , z _ { \leq t } ) .\tag{10}
$$

This Monte Carlo estimate removes direct human annotation at each step but remains anchored to the final target. PRIME removes the separate annotation stage by deriving an implicit process reward from outcome labels and updating the reward model with policy rollouts (Cui et al., 2025a). Comparative studies show that automatically generated step labels can still lag behind LLM judgments or human annotations, although consensus filtering can improve their reliability (Zhang et al., 2025n). Related work extends process rewards across domains and develops reasoning-based or entropy-regularized variants (Zeng et al., 2025b; She et al., 2025b; Zhang et al., 2024a).

Generative verifiers formulate evaluation as next-token prediction and produce an analysis before issuing a judgment (Zhang et al., 2025d). Long-reasoning and generative PRMs apply this formulation at the step level (Zhao et al., 2026; Khalifa et al., 2025). Compact systems such as xVerify and TinyV focus on answer extraction, semantic equivalence, and false negatives from rule-based matching (Chen et al., 2025a; Xu et al., 2025c). CompassVerifier extends answer checking across domains (Liu et al., 2025g), while RL Tango jointly trains a generator and a generative process verifier with outcome rewards for both components (Zha et al., 2025). The same approach now covers multimodal reasoning and multi-turn agents. Multimodal PRMs score intermediate reasoning (Zhao et al., 2026; Wang et al., 2025h; Chen et al., 2025h), multimodal reward models assess completed outputs (Wang et al., 2025j; Zhang et al., 2025m; Wang et al., 2026a), and agentic verifiers evaluate tool-use trajectories (Zhang et al., 2026a; Chae et al., 2025; Zhang et al., 2025a).

Several systems train verifiers that span both verifiable and open-ended tasks. Seed1.5 Thinking evaluates whether responses align with references and supplies graded feedback when exact matching is inadequate (Seed et al., 2025). General Reasoner trains a compact generative verifier on synthetic candidates derived from WebInstruct (Ma et al., 2025; Yue et al., 2024b). StructVRM represents partial correctness with a reward vector whose dimensions correspond to subquestions (Zhang et al., 2025h). Posterior GRPO adds a reasoning-quality reward for code but applies it only when the final program is correct (Fan et al., 2026). Together, these methods show that learned verification can enrich an available target without changing where the target ultimately originates.

General reward models take a broader approach by learning preferences across tasks. Skywork Reward uses a discriminative Bradley-Terry objective over curated preference pairs (Liu et al., 2024a), while Nemotron Reward predicts attributes such as helpfulness, correctness, coherence, complexity, and verbosity (Adler et al., 2024). Reasoning-oriented models generate an analysis before assigning a preference or score. For example, RM R1 combines reasoning distillation with RLVR (Chen et al., 2025g), J1 optimizes judgment reasoning through correctness and consistency rewards (Whitehouse et al., 2025), and RRM allocates additional inference computation through longer reasoning or response aggregation (Guo et al., 2025b). Think RM, S2J, and CAMEL respectively train long-form reward reasoning, couple problem solving with judging, and invoke reflection selectively for uncertain comparisons (Hong et al., 2025; Sun et al., 2025c; Zhu et al., 2026b). GRAM instead pretrains on unlabeled input-response pairs before preference tuning and predicts a label token without an explicit long reasoning trace (Wang et al., 2025a).

Reward-model development also depends on data scale and modality. UnifiedReward and R1 Reward extend reward modeling to image and video understanding and generation (Wang et al., 2025l; Zhang et al., 2025l). Skywork Reward V2 trains on a large filtered preference collection (Liu et al., 2025c), while HelpSteer3 Preference supports both discriminative and generative reward modeling with human annotations (Wang et al., 2025m). Prometheus 2 combines rubric-conditioned assessment with pairwise ranking and produces verbal feedback, although it is primarily evaluated as an open evaluator (Kim et al., 2024b). POLAR takes a diferent route by pretraining a reference-conditioned policy discriminator on same-policy and diferent-policy trajectory pairs before aligning it with human-defined criteria (Dou et al., 2025).

Task-specific reward models instantiate these ideas under narrower constraints. Self-Principled Critique Tuning generates task-specific principles and critiques, then optimizes their judgments with rule-based online RL (Liu et al., 2025j; Bai et al., 2022). Writing-Zero adapts the same procedure to the evaluation of creative writing (Lu, 2025). For instruction following, Ren et al. (Ren et al., 2025a) combine programmatic checks for hard constraints with a learned binary model for soft constraints. EvolvR synthesizes score-aligned comparison rationales and iteratively checks, refines, and attacks its own judgments (Wang et al., 2025i). In computer use, SEAgent trains a World State Model to describe GUI changes and provide action-level rewards over complete trajectories (Sun et al., 2025d).

## 3.2.2. LLM Judges and Rubric-Based Evaluators

Open-ended evaluation often requires several criteria at once. Factual responses may need claim decomposition and evidence checking, while writing and dialogue require judgments about relevance, coherence, style, and usefulness (Min et al., 2023; Lin et al., 2022; Fein et al., 2025). LLM judges provide a scalable way to apply such criteria. RLAIF uses an LLM to generate preference labels for reward-model training or to supply rewards directly during policy optimization (Lee et al., 2023). Related evaluators cover free-form text, reasoning, creative writing, and task-dependent RLHF (Gu et al., 2024b; Badshah & Sajjad, 2024; Saha et al., 2025; Fein et al., 2025; Dong et al., 2024).

Reference-based LLM verifiers can issue binary or graded rewards. Su et al. (Su et al., 2025) study verifiers that receive a prompt $x ,$ a reference $^ { a , }$ and the final step $y _ { i } ^ { T }$ of a response. Their binary reward is

$$
\begin{array} { r } { r _ { \phi } ^ { \mathrm { b i n } } ( x , a , y _ { i } ) = \mathbb { I } [ c _ { i } = 1 ] , \qquad c _ { i } \sim \pi _ { \phi } ( \cdot \mid x , a , y _ { i } ^ { T } ) , } \end{array}\tag{11}
$$

while the soft reward is

$$
\begin{array} { r } { r _ { \phi } ^ { \mathrm { s o f t } } ( x , a , y _ { i } ) = \left\{ \begin{array} { l l } { \pi _ { \phi } ( 1 \mid x , a , y _ { i } ^ { T } ) , } & { c _ { i } = 1 , } \\ { 1 - \pi _ { \phi } ( 0 \mid x , a , y _ { i } ^ { T } ) , } & { c _ { i } = 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{12}
$$

VDS-TTT uses a pretrained verifier to select a candidate for test-time training only when its score exceeds a confidence threshold (Moradi et al., 2025). SSR-Zero alternates the same pretrained model between actor and judge roles in machine translation, then rescales the judgments as policy rewards (Yang et al., 2025c).

The evaluator can also operate on richer evidence than text alone. ZeroGUI replaces handcrafted success scripts with a vision-language reward estimator that receives task instructions and trajectory screenshots (Yang et al., 2025b). It excludes the agent’s textual claim of success and requires unanimous judgments across repeated queries, which reduces false-positive rewards. For long-form factual reasoning, claim-level verification can be combined with a separate comparison against a reference answer to assess overall relevance and quality (Chen et al., 2025f).

Rubrics and checklists make multidimensional standards explicit before they are applied to candidate outputs. Omni Thinker combines rule-based rewards with rubric-aligned preferences for dialogue and writing (Li et al., 2025a). RLCF extracts instruction-specific checklists and evaluates each item with an AI judge or specialized program (Viswanathan et al., 2025). RaR synthesizes prompt-specific rubrics from expert answers or strong references (Gunjal et al., 2025), while Rubicon combines human-generated and model-generated criteria with defenses against recurring reward exploits (Huang et al., 2025d). RuscaRL uses checklists both to score and guide generation, then gradually reduces the guidance (Zhou et al., 2025d).

The criteria need not remain fixed because rubric construction can itself become adaptive. Kimi K2 uses explicit success, tool-use, and evaluation criteria during data synthesis and policy optimization (Team et al., 2025a). OpenRubrics derives prompt-specific criteria by contrasting preferred and rejected responses (Liu et al., 2026b). EvoRubrics jointly optimizes the policy and rubric generator (Ding et al., 2026), while Rubric ARM and RUBRIC ARROW alternate between training rubric generators and rubric-conditioned judges (Xu et al., 2026a; Jiang et al., 2026a). AMARIS revises rubrics during policy training using a persistent memory of rollout analyses (Wu et al., 2026b). When pointwise rubric scores cannot distinguish same-query rollouts reliably, Tournament-GRPO and ArenaRL convert repeated relative comparisons into group-wise rewards (Yang et al., 2026b; Zhang et al., 2026b).

Specialized judges can improve reliability when a general evaluator lacks domain knowledge. Baichuan M2 combines a patient simulator with a rubric generator and evaluates medical responses across multiple turns (Team et al., 2025b). The benefit comes from domain-specific evidence and criteria, although the resulting reward remains human-grounded because medical records, conversations, and rubric standards determine what the evaluator learns to value.

## 3.2.3. Self-Trained Evaluators and Boundary Cases

Self-trained evaluators sit between human-grounded evaluation and rewards obtained without human evaluation. Self-Rewarding Language Models use the same model to generate responses and judge them, then train on pairs formed from the highest- and lowest-scoring candidates (Yuan et al., 2024). Meta-Rewarding adds a meta-judge that compares sampled judgments and trains the judging role alongside the actor (Wu et al., 2025d). Self-Taught Evaluators begin with unlabeled human-written instructions, construct preferred and inferior responses, retain reasoning traces whose verdict matches the synthetic label, and iteratively fine-tune the evaluator (Wang et al., 2024b). Process-based Self-Rewarding applies the same idea to step-level mathematical preferences (Zhang et al., 2025g). These methods remove labeled preferences from diferent parts of the evaluator-training loop, but many retain supervised initialization, human-written prompts, or human-defined judging formats. Their position on the ladder consequently depends on the evidence that continues to anchor the evaluator.

Policy optimization does not provide a second ladder dimension. DPO, IPO, SimPO, and ORPO learn from fixed preference pairs, while KTO uses pointwise desirable or undesirable feedback (Rafailov et al., 2023; Azar et al., 2024; Ethayarajh et al., 2024; Meng et al., 2024; Hong et al., 2024). The pairs may come from people, reusable judges, or self-generated comparisons. SPIN, ReST, and Self-Rewarding Language Models likewise difer in how they refresh and filter policy data (Chen et al., 2024f; Gulcehre et al., 2023; Yuan et al., 2024). Ofline reuse reduces evaluator queries during optimization but freezes the feedback function and its data support, which can leave poor coverage as the policy changes (Casper et al., 2023; Xiong et al., 2024). For this reason, online and ofline objectives describe how feedback is consumed, not where its evaluative standard originates.

Human-grounded evaluators reduce the need for a new judgment on every rollout. However, their criteria still trace back to preferences, annotations, rubrics, or expert standards supplied by people. Moving beyond this dependence requires reward signals whose immediate evidence comes from another source.

## 3.3. Reward Sources beyond Human Evaluation

At L2, the reward no longer depends primarily on an evaluator trained to reproduce human judgments. Within this regime, we distinguish three sources of evidence. Model-derived signals use properties of the policy or its sampled outputs, including certainty and agreement. Reference-anchored signals measure how a reasoning trajectory supports or reconstructs an available target. Environment-grounded signals evaluate observable consequences such as execution, formal acceptance, or game outcomes.

These categories concern the evidence used when a rollout is scored, not the historical origin of every artifact in the pipeline. References may come from human-authored corpora, and people may have designed tests, tools, or game rules (Zhou et al., 2025b; Yu et al., 2025d; Liu et al., 2025h; Gehring et al., 2025; Liu et al., 2025a). The amount of remaining supervision consequently depends on how reusable that evidence is. A separate answer written for every task remains close to per-instance supervision, while fixed environment dynamics can evaluate many new trajectories. Retrieval occupies a similar boundary because search supplies evidence, but the terminal reward may still use a known answer or task rule (Jin et al., 2025; Song et al., 2025a).

## 3.3.1. Model-Derived Signals

Intrinsic certainty. Intrinsic certainty maps the model’s predictive state to a scalar reward without consulting a reference answer at scoring time. Its usefulness depends on a statistical relationship between confidence and correctness, not on confidence alone. Uncertainty is a broader concept than any single estimator because sequence likelihood, token entropy, semantic entropy, and verbalized confidence describe diferent objects and need not rank outputs consistently (Liu et al., 2025i; Shorinwa et al., 2025). SLiC calibrates sequence likelihood with references (Zhao et al., 2023), while UaIT trains verbalized confidence from probability-derived pseudo-targets (Liu et al., 2024c). Purely intrinsic rewards difer because their rollout scores are not anchored to a target answer, environment outcome, or human-grounded evaluator.

Entropy minimization becomes informative under two conditions: the base model must already place useful probability mass on good solutions, and its confidence must correlate with correctness on the target distribution (Agarwal et al., 2025). EM-RL implements these conditions through negative-entropy rewards computed at either the sequence or token level. For a sampled trajectory $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } ) \sim \pi _ { \theta } ( \cdot \mid q )$ , its sequence-level reward is

$$
r _ { \mathrm { s e q } } ( q , \mathbf { y } ) = \log \pi _ { \boldsymbol { \theta } } ( \mathbf { y } \mid q ) = \sum _ { t = 1 } ^ { T } \log \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid q , \mathbf { y } _ { < t } ) ,
$$

whose expectation is the negative trajectory entropy. The token-level variant accumulates the negative Shannon entropy of the full next-token distribution:

$$
r _ { \mathrm { { t o k } } } ( q , \mathbf { y } ) = - \sum _ { t = 1 } ^ { T } H ( \pi _ { \theta } ( \cdot \mid q , \mathbf { y } _ { < t } ) ) , \qquad H ( \pi _ { \theta } ( \cdot \mid q , \mathbf { y } _ { < t } ) ) = - \sum _ { \nu \in \mathcal { N } } \pi _ { \theta } ( \nu \mid q , \mathbf { y } _ { < t } ) \log \pi _ { \theta } ( \nu \mid q , \mathbf { y } _ { < t } ) .
$$

Both variants assign an outcome reward to a complete trajectory, but they derive it from diferent predictive distributions. Reported gains in mathematics and code are best understood as eliciting capabilities already represented by the base policy. They do not establish that low entropy guarantees correct reasoning (Agarwal et al., 2025).

Other methods construct confidence in semantic or token space. EMPO clusters rollouts by semantic similarity and rewards answers according to cluster mass (Zhang et al., 2025f). RENT and RSLC use token-level entropy or confidence to rank rollouts and smooth policy updates (Prabhudesai et al., 2025; Li et al., 2025e). Intuitor defines self-certainty as the average divergence between a uniform distribution U and the next-token distribution over vocabulary V (Zhao et al., 2025b; Kang et al., 2025):

$$
{ \mathcal { R } } _ { i } = { \frac { 1 } { | \pmb { \mathscr { o } } _ { i } | } } \sum _ { t = 1 } ^ { | \pmb { \mathscr { o } } _ { i } | } \mathrm { K L } \bigl ( \mathbb { U } \parallel \pi _ { \theta } ( \cdot \mid \pmb { \mathscr { o } } _ { i , < t } ) \bigr ) .\tag{13}
$$

A larger value indicates a greater departure from uniformity and is interpreted as higher certainty. Shop-R1 applies this signal in online shopping (Zhang et al., 2025j), and related work studies it in vision-language reasoning (Li et al., 2026f).

Confidence rewards remain available when ground-truth answers are absent, but optimization can make them increasingly detached from correctness. Entropy minimization is sensitive to the base model’s initial entropy and may eventually produce overconfidence, omitted transition tokens, and degraded reasoning (Zhang et al., 2025k). This limitation motivates calibration and external evaluation even when those signals are not used to compute the training reward.

Entropy can also influence optimization without becoming the reward source. Covariance-aware clipping, high-entropy token selection, and prompt-level semantic-entropy weighting use uncertainty to stabilize or focus learning (Cui et al., 2025b; Wang et al., 2025g; Chen et al., 2025d). Other methods combine uncertainty with an external correctness signal. ACE penalizes overconfident errors (Xu et al., 2026b), and calibrated semantic-entropy targets support verbalized uncertainty (Jenane et al., 2026). Reasoning with Exploration, EDGE-GRPO, RLCR, Archer, and CoDaPO use entropy or confidence in advantage estimation, correction, calibration, token constraints, or data selection while retaining verifier-based rewards (Cheng et al., 2026; Zhang et al., 2025i; Damani et al., 2025; Wang et al., 2025d; Zhou et al., 2025f). ETTRL uses consensus pseudo-labels and entropy-guided branching (Liu et al., 2025d), while CURE branches and reconnects trajectories at high-entropy tokens to preserve exploration (Li et al., 2025f). Because these methods retain a separate correctness signal, they are entropy-assisted RLVR methods, not standalone certainty rewards.

The term intrinsic here refers to confidence derived from the model’s predictive distribution. Classic RL uses the same term for novelty-seeking signals such as curiosity and count-based exploration (Pathak et al., 2017). The two objectives can exert opposite pressures because confidence concentrates probability mass, while novelty encourages the policy to visit unfamiliar states. EVOL RL combines semantic novelty with majority-vote selection to counteract collapse under pure confidence signals (Wu et al., 2025c).

Answer-level consensus. Consensus replaces an unavailable answer key with agreement across repeated samples. TTRL draws $G$ responses $\{ \pmb { o } _ { i } \} _ { i = 1 } ^ { G }$ for a question $\pmb q$ and extracts answers $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { G }$ (Zuo et al., 2025b). If A is the set of distinct answers, the majority pseudo-label is

$$
\hat { \pmb { y } } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { i = 1 } ^ { G } \mathbb { I } \{ \pmb { y } _ { i } = a \} ,\tag{14}
$$

and each rollout receives

$$
r _ { i } = \mathbb { I } \{ { \bf y } _ { i } = \hat { { \bf y } } \} .\tag{15}
$$

The pseudo-label follows the current model distribution and can be wrong. Consensus thus provides model-derived supervision but no independent guarantee of correctness.

Consensus rewards have been extended to multimodal reasoning through MM-UPT (Wei et al., 2025b). Majority-vote self-training can improve initially and then collapse as the policy learns to exploit its own pseudo-labels (Shafayat et al., 2025). SeRL incorporates consensus into a self-play task-generation pipeline (Fang et al., 2025b), while ETTRL couples it with entropy-guided rollout exploration (Liu et al., 2025d). Co-Rewarding adds agreement across semantically related questions and a slowly updated teacher (Zhang et al., 2025o).

Several methods refine hard majority voting when agreement is unreliable. SCOPE combines step-level confidence with subgroup consensus (Wang et al., 2026b), while SCRL filters uncertain positive pseudo-labels and introduces entropy-gated negative labels (Yan et al., 2026). TTRL-Guard detects when correct answers disappear from the vote distribution and intervenes before an incorrect majority becomes entrenched (Lin et al., 2026a). These safeguards reduce confirmation bias, although the reward remains tied to the model’s own answer distribution.

Trajectory-level consensus. Final-answer agreement discards information about how a solution was produced. Self-consistency can improve inference accuracy (Wang et al., 2023c), but answer-only statistics remain vulnerable to reward exploitation (Zhang et al., 2025c). CoVo compares likelihood distances from intermediate states to the trajectory’s answer and to answers drawn from other trajectories (Zhang et al., 2025c). It measures how consistently a trajectory supports its own answer and how long competing answers remain plausible. Aggregating these statistics across trajectories gives a richer reward than majority vote and is reported to improve stability. However, the statistics still come from the policy and do not independently certify correctness.

## 3.3.2. Reference-Anchored Signals

Reference-anchored methods retain an available target $y ^ { * }$ but avoid a separately trained evaluator for every rollout. Instead of relying only on exact answer matching, they measure how a sampled reasoning trace � changes the probability, reconstruction, or metric score of the target (Zhou et al., 2025b; Yu et al., 2025d; Liu et al., 2025h). The reference supplies external evidence and can prevent direct self-confirmation. At the same time, it may inherit substantial human or expert supervision, and a single target may cover only part of the acceptable answer space.

Variational objectives. One family treats the reasoning trace as a latent variable and optimizes the marginal likelihood of $\boldsymbol { y } ^ { * }$ . TRICE uses an MCMC EM procedure to sample from an answer-conditioned posterior over rationales (Phan et al., 2023). LaTRO rewards answer log-likelihood and regularizes the rationale policy toward a frozen pre-optimization model (Chen et al., 2024c). JEPO applies Jensen’s inequality to the latent-rationale factorization and obtains a policy-gradient term for reasoning together with an SFT-like answer-prediction term (Tang et al., 2025). Its one-sample core is

$$
\begin{array} { r l } & { \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { * } \mid \boldsymbol { q } ) : = \log \mathbb { E } _ { z \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { q } ) } \left[ \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { * } \mid \boldsymbol { q } , z ) \right] } \\ & { \qquad \ge \mathcal { L } _ { \boldsymbol { \mathrm { J } } } ^ { ( 1 ) } ( \boldsymbol { \theta } ) : = \mathbb { E } _ { z \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { q } ) } \left[ \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ^ { * } \mid \boldsymbol { q } , z ) \right] . } \end{array}\tag{16}
$$

Table 2: Policy gradient decomposition of methods that use the reference answer as an implicit incentive signal. Each update comprises a reasoning term weighted by $\textstyle { \mathcal { R } } _ { r }$ and a final answer term weighted by $\mathcal { R } _ { a } .$
<table><tr><td>Method</td><td>Update Policy</td><td colspan="2">Different Reward Term Final Answer</td></tr><tr><td>GRPO (Shao et al., 2024)</td><td> $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \pmb { q } ) + \mathscr { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } | \pmb { q } , z )$ </td><td></td><td></td></tr><tr><td>JEPO (Tang et al., 2025)</td><td> $\begin{array} { r } { \mathcal { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \boldsymbol { q } ) + \mathcal { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { * } | \boldsymbol { q } , z ) } \end{array}$ </td><td colspan="2"> $\mathbf { 1 } ( \mathbf { y } = \mathbf { y } ^ { * } )$   $\begin{array} { r } { \log \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \pi _ { \theta } ( y ^ { * } | q , z _ { j } ) } \end{array}$ </td></tr><tr><td>LaTRO (Chen et al., 2024c)</td><td> $\begin{array} { r } { \mathcal { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \boldsymbol { q } ) + \mathcal { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { * } | \boldsymbol { q } , z ) } \end{array}$ </td><td> $\begin{array} { r } { \pi _ { \theta } ( \mathbf { y } ^ { * } | \pmb { q } , z ) - \log \frac { \pi _ { \theta } ( z | \pmb { q } ) } { \pi _ { \mathrm { r e f } } ( z | \pmb { q } ) } } \end{array}$ </td><td>1 1</td></tr><tr><td>VeriFree (Zhou et al., 2025b)</td><td></td><td colspan="2"></td></tr><tr><td>RLPR (Yu et al., 2025d)</td><td> $\begin{array} { r } { \mathcal { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \boldsymbol { q } ) + \mathcal { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { * } | \boldsymbol { q } , z ) } \end{array}$ </td><td colspan="2"> $\pi _ { \boldsymbol { \theta } } ( \mathbf { y } ^ { * } | \boldsymbol { q } , z )$ </td></tr><tr><td>DRO (Xu et al., 2025b)</td><td> $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \pmb { q } ) + \mathscr { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } | \pmb { q } , z )$ </td><td colspan="2"> $\begin{array} { r } { \frac { 1 } { | \mathbf { y } ^ { * } | } \sum _ { i = 1 } ^ { | \mathbf { y } ^ { * } | } \left( \pi _ { \theta } \left( y _ { i } ^ { * } | \pmb { q } , z , \mathbf { y } _ { < i } ^ { * } \right) - \pi _ { \theta } \left( y _ { i } ^ { * } | \pmb { q } , \mathbf { y } _ { < i } ^ { * } \right) \right) } \end{array}$ </td></tr><tr><td>VR-CLI (Gurung &amp; Lapata, 2025)</td><td> $\begin{array} { r } { \mathcal { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \boldsymbol { q } ) + \mathcal { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { * } | \boldsymbol { q } , z ) } \end{array}$   $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | q )$ </td><td colspan="2"> $\begin{array} { r } { \sum _ { i = 1 } ^ { | { \pmb y } ^ { \bullet } | } \left( \omega ( \sigma _ { i } ) \log \pi _ { \theta } \left( y _ { i } ^ { \ast } | \pmb q , z , \pmb y _ { < i } ^ { \ast } \right) - \log \pi _ { \theta } \left( y _ { i } ^ { \ast } | \pmb q , z _ { \mathrm { m a s k } } , \pmb y _ { < i } ^ { \ast } \right) \right) } \end{array}$   $\begin{array} { r } { \big ( 1 - \frac { \mathrm { P P L } _ { \pi _ { g } } ( { \bf y } ^ { * } \mid { \bf q } , z ) } { \mathrm { P P L } _ { \pi _ { g } } ( { \bf y } ^ { * } \mid { \bf q } ) } \big ) \times 1 0 0 } \end{array}$ </td></tr><tr><td>NOVER (Liu et al., 2025h)</td><td> $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \pmb { q } ) + \mathscr { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } | \pmb { q } , z )$ </td><td colspan="2"> $\begin{array} { r } { \exp \left( - \frac { \sum _ { i = 1 } ^ { | { \pmb v } ^ { * } | } \log \pi _ { \theta } ( y _ { i } ^ { * } | { \pmb q } , z , { \pmb y } _ { < i } ^ { * } ) } { | { \pmb y } ^ { * } | \cdot N ( | { \pmb z } | ) } \right) } \end{array}$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Logprob (Kwiatkowski et al., 2026) P2S (Zhong et al., 2026)</td><td> $\begin{array} { r } { \mathcal { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \boldsymbol { q } ) + \mathcal { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { * } | \boldsymbol { q } , z ) } \end{array}$ </td><td> $\log \pi _ { \theta } ( \mathbf { y } ^ { * } | \boldsymbol { q } , z )$ </td><td>1</td></tr><tr><td></td><td> $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z _ { \le t } | \pmb { q } )$ </td><td> $\pi _ { \theta } ( z _ { > t } ^ { * } | \pmb q , z _ { \leq t } )$ </td><td></td></tr><tr><td>DARL (Huang et al., 2026a)</td><td> $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z | \pmb { q } ) + \mathscr { R } _ { a } \nabla _ { \theta } \log \pi _ { \theta } ( \pmb { y } | \pmb { q } , z )$   $\mathscr { R } _ { r } \nabla _ { \theta } \log \pi _ { \theta } ( z , y | \pmb { q } _ { k } )$ </td><td colspan="2"> $\begin{array} { r } { \frac { 1 } { \vert \mathbf { y } ^ { * } \vert } \sum _ { i = 1 } ^ { \vert \mathbf { y } ^ { * } \vert } \pi _ { \theta } \left( y _ { i } ^ { * } \vert \pmb { q } , z , \mathbf { y } _ { < i } ^ { * } \right) + \lambda \mathcal { R } _ { \mathrm { d i v } } } \end{array}$   $d ( \hat { q } _ { u } , q _ { u } )$ </td></tr></table>

This expression is a Jensen lower bound. A general ELBO would introduce an answer-conditioned variational posterior and a prior-posterior ratio. JEPO further tightens Eq. 16 with a multi-sample log-mean-probability bound and adds forward-KL regularization during training.

Likelihood and probability rewards. Direct likelihood methods avoid explicit variational machinery. VeriFree rewards the conditional probability $\pi _ { \boldsymbol { \theta } } ( \mathbf { y } ^ { * } \mid \pmb { q } , z )$ of the reference after a sampled trace (Zhou et al., 2025b). Under a unique answer string and exact matching, its Rao-Blackwellized estimator has no greater variance than an estimator that samples an answer and applies a binary verifier. RLPR averages reference-token probabilities, subtracts a score computed without the sampled trace, and clips the diference (Yu et al., 2025d). The subtraction controls a baseline association but does not identify a causal efect of the trace. Log-probability rewards are generally more practical than exact sequence probabilities for long references, whose probabilities can vanish. Even then, apparent gains may partly reflect shorter traces instead of uniformly better reasoning (Kwiatkowski et al., 2026). DARL adds a thresholded diversity term to encourage controlled exploration among reference-consistent variants (Huang et al., 2026a).

Perplexity rewards. NOVER converts reference-conditioned perplexity into a within-group ranking reward (Liu et al., 2025h). Given �, �, and $\boldsymbol { y } ^ { * }$ , a synchronized proxy model computes

$$
P _ { r } = \exp \left( - { \frac { \sum _ { i } \log \pi _ { \mathrm { p } } ( y _ { i } ^ { * } \mid q , z , y _ { < i } ^ { * } ) } { | y ^ { * } | N ( | z | ) } } \right) , \qquad N ( | z | ) = \operatorname * { m a x } ( 1 , 1 + \log | z | ) .
$$

Lower perplexity receives a higher rank. An additional eficiency score rewards a shorter trace only when it also improves reference perplexity. VR-CLI applies a related construction to next-chapter prediction (Gurung & Lapata, 2025). A frozen story generator evaluates whether a generated plan makes the observed continuation more predictable. The resulting reward measures support for that continuation without assuming that it is the only valid story.

Token-selective and process rewards. Some methods ask which parts of a reference are most informative about the sampled reasoning. Direct Reasoning Optimization weights reference tokens by how much their conditional probabilities vary across rollouts, then aggregates the weighted scores under rubric and query filters (Xu et al., 2025b). P2S generates candidate gold chains conditioned on the question and answer, filters them, and selects the chain with the highest reference likelihood (Zhong et al., 2026).

Segment-level rewards then measure how much each part of a sampled trace improves prediction of a later chain segment. This construction provides process shaping when a direct outcome reward is unavailable.

Reconstruction and cycle consistency. Other methods obtain reference-anchored signals from a reconstruction loop. DuPO decomposes a query into known and unknown components and rewards answers that support reconstruction of the missing component (She et al., 2025a). CycleReward converts cross-modal reconstruction scores into preferences for image and text generation (Bahng et al., 2025). TRANS ZERO trains translation from monolingual data through round-trip semantic consistency (Zou et al., 2025), while RTRL rewards chemical-language outputs according to the likelihood that a frozen backward model reconstructs the original input (Kong et al., 2026).

Reference metrics. Reference-based metrics can score sampled outputs without a fresh judgment on every rollout. Their training data may nevertheless encode human preferences (Qi et al., 2026a). Machine-translation methods combine reference metrics, learned quality estimators, source consistency, alignment, and format checks (Yang et al., 2025c; Rei et al., 2020; Feng et al., 2025c; He et al., 2025; Li et al., 2025j). Similar scorers compare generations with reference responses or explanations in other domains (Li et al., 2025i; Pappone et al., 2025). BLEUBERI optimizes BLEU against synthetic references (Chang et al., 2025), and RLVRR combines reference-derived content checks with automatically generated style criteria (Jiang et al., 2026b).

VeriFree illustrates how a single objective can influence both reasoning and answer prediction. In its leave-one-out estimator, the reasoning term is weighted by $\begin{array} { r } { A _ { i } \ = \ R _ { i } - { \frac { 1 } { G - 1 } } \sum _ { j \neq i } R _ { j } } \end{array}$ , while the answer term is weighted by $R _ { i } = \pi _ { \theta } ( y ^ { \star } \mid x , z _ { i } )$ (Zhou et al., 2025b). The decomposition clarifies why reference-anchored rewards can shape a trace even when no external verifier evaluates that trace directly.

## 3.3.3. Environment-Grounded Signals

Reference-anchored rewards ask whether a rollout supports an existing target. Environment-grounded rewards instead evaluate what happens when the output is executed or used in an interaction. Executable tests, formal kernels, game rules, and observable interface states can adjudicate success without a new human judgment for each trajectory. This external grounding is stronger than confidence or consensus because the evidence does not come from the policy’s own distribution. It is still imperfect, since people may have written the tests, specified the goals, or chosen the observable outcome.

Executable outcomes. Code execution provides the clearest example. For a program � and test suite T, a strict reward can be written as

$$
\mathcal { R } ( \mathbf { y } ) = \mathbb { I } \{ \operatorname { e x e c } ( \mathbf { y } , \mathcal { T } ) \ \operatorname { p a s s e s } \} .
$$

Other designs use the fraction of passed tests or a terminal success signal after several rounds of interaction (Li et al., 2026d). RLEF treats competitive programming as an iterative task in which execution feedback is appended to the dialogue (Gehring et al., 2025). Public-test success triggers evaluation on held-out private tests, and PPO optimizes the final all-tests-pass outcome with an invalid-code penalty and KL regularization. ACECoder synthesizes tasks and tests from seed code, uses test pass rates to form preference pairs, and trains either a learned reward model or a rule-based policy reward (Zeng et al., 2025a). When a model also generates the tests or tasks, the same executable outcome can support self-generated experience. Co-evolving coder and test-generator systems therefore connect this reward source to the experience axis (Wang et al., 2025k).

Tool execution during a rollout does not automatically make the reward environment-grounded. ReTool and ToRL call a Python interpreter while reasoning, but their main policy rewards are based on final-answer correctness (Feng et al., 2025a; Li et al., 2025h). ToolRL rewards agreement with ground-truth tool names and parameters (Qian et al., 2025). In these cases, execution changes the information available to the policy, while the terminal evaluation remains reference-based. The distinction matters because tool access and reward grounding address diferent parts of the learning loop.

Repository-level software engineering contains both reference-based and execution-based designs. SWE-RL scores generated patches by similarity to an oracle patch and uses oracle-localized context (Wei et al., 2026). Its reward is consequently anchored to a stored target even though the surrounding scafold includes tests and reranking. Kimi-Dev uses binary execution outcomes for Agentless code-edit RL, while its later end-to-end SWE-agent adaptation relies on supervised fine-tuning (Yang et al., 2026d). SkyRL Agent applies multi-turn RL to software engineering, although its reported objective combines several signals and does not isolate a single verified-completion reward (Cao et al., 2025).

Retrieval and interactive outcomes. Search and browsing provide environmental observations, but many systems still judge the final answer with a reference or model-based evaluator. Search R1 uses exact matching against a target answer and masks retrieved tokens from the policy loss (Jin et al., 2025). R1 Searcher separates retrieval incentives from answer rewards (Song et al., 2025a). WebDancer, WebSailor, WebExplorer, and ASearcher support long browsing trajectories but retain terminal judgments based on references or LLM evaluators (Wu et al., 2025a; Li et al., 2025b; Liu et al., 2025e; Gao et al., 2025b). Process rewards and simulated retrieval can improve search behavior under the same final-answer grounding (Zheng et al., 2025b; Nguyen et al., 2025b; Sun et al., 2025a). Computer-use agents likewise span programmatic state checks, learned visual outcome models, action matching, reference judgments, and ofline preferences (Li et al., 2025d; Ye et al., 2025; Wang et al., 2025c; Lu et al., 2026; Luo et al., 2025; Qin et al., 2025). Observable interaction is thus a source of experience, while only an outcome tied to the resulting state constitutes environment-grounded reward.

Game outcomes. Games provide sparse but reproducible rewards through wins, losses, and rule-defined scores. SPIRAL trains a shared policy through self-play in two-player zero-sum language games (Liu et al., 2025a). The game outcome supplies the reward, while evolving opponents create an adaptive curriculum. The method therefore belongs to the reward axis through its terminal signal and to the experience axis through self-play. Role-conditioned advantage estimation addresses positional asymmetry, and the resulting policy transfers beyond the games used for training. Vision Zero extends outcome-based self-play to multimodal reasoning through a visual social-deduction game (Wang et al., 2025e). Absolute Zero combines a Python executor with self-proposed tasks, using separate rewards for proposer learnability and solver correctness (Zhao et al., 2025a).

Formal verification. Formal theorem proving ofers one of the strongest environment-grounded signals because kernel acceptance gives a reproducible outcome under a fixed statement, environment, and resource budget. Formal tools can also provide verified step-level feedback (Rahman & Alsharari, 2026). Even so, kernel acceptance does not validate the natural-language formalization of a theorem, and a timeout is not a logical refutation.

DeepSeek Prover V2 trains with Lean-verified correctness and structural-consistency rewards after decomposing dificult goals into subgoals (Ren et al., 2025b). Kimina Prover scales RL with binary Lean acceptance (Wang et al., 2025b), while Seed Prover iteratively refines complete proofs using Lean feedback (Chen et al., 2025d). Goedel Prover V2 combines task synthesis with verifier-guided correction (Lin et al., 2026b), and Leanabell Prover V2 trains long trajectories through repeated interactions with Lean (Ji et al., 2025). STP couples a conjecturer and prover and selects frontier conjectures using proof outcomes and lemma-usage checks (Dong & Ma, 2025). InternLM2.5 StepProver provides a contrasting fixed-corpus regime based on formal libraries and expert iteration (Wu et al., 2025e).

Taken together, environment-grounded rewards ofer the clearest separation between the policy and the evidence used to score it. That separation reduces direct self-confirmation but does not eliminate specification error. Incomplete tests, inaccurate simulators, exploitable game rules, and mismatches between natural-language intent and formal statements can all preserve high reward without producing the intended behavior. Reward grounding must therefore be assessed together with the validity and coverage of the environment.

## 3.4. Limitations and Failure Modes

As evaluation moves from per-instance targets to reusable or automatically obtained signals, supervision becomes more scalable but the source of error also changes. Human-grounded evaluators may misrepresent the judgments they were trained to reproduce. Model-derived rewards can reinforce the policy’s own mistakes. Reference-anchored rewards can overfit an incomplete target, while environment-grounded rewards can exploit omissions in tests or rules. The relevant question is consequently not whether a reward is automatic, but what evidence makes it trustworthy under optimization.

## 3.4.1. Failures of the Reward Signal

Human-grounded evaluators. LLM judges exhibit position, verbosity, self-enhancement, and modelspecific biases (Chen et al., 2024b). Diverse judge panels can reduce some model-specific efects (Verga et al., 2024), and rubrics or checklists make the evaluation criteria more explicit (Gunjal et al., 2025; Viswanathan et al., 2025). Nevertheless, an explicit rubric can still omit substantive requirements, and automatic preference scores may favor longer responses or presentation features (Dubois et al., 2024; Singhal et al., 2023). Reward-model ensembles reduce overoptimization but do not remove it (Coste et al., 2023; Eisenstein et al., 2023). Constitution- and principle-guided pipelines clarify the normative source of supervision through critique and revision (Bai et al., 2022; Sun et al., 2023), although their reliability still depends on the chosen principles and evaluator.

Model-derived signals. Consensus can reinforce an initially dominant wrong answer because it lacks independent evidence of correctness. Under prolonged majority-vote training, proxy reward may separate from task accuracy and produce template answers or collapse (Shafayat et al., 2025). Entropy and self-certainty rewards show a related pattern. Some base models improve early and then become overconfident, while instruction-tuned models often benefit less (Zhang et al., 2025k). A broader analysis attributes this behavior to distribution sharpening and observes a recurring rise followed by decline (He et al., 2026). Hyperparameter changes can delay the failure, but they do not supply the missing external anchor. Structured self-verification may extend the useful regime, although current evidence does not establish a general solution.

Reference-anchored and hybrid rewards. Reference-likelihood methods optimize the probability or perplexity of a fixed target conditioned on a sampled trace (Zhou et al., 2025b; Liu et al., 2025h; Tang et al., 2025). When one reference represents only part of the valid answer space, the objective leaves other aspects of quality unspecified. Hybrid systems can combine complementary evidence without forming a separate ladder level. JEPO couples a policy-gradient reasoning objective with supervised prediction of the target answer (Tang et al., 2025). Seed1.5 Thinking uses verifiers for verifiable tasks and a learned pairwise reward model for open-ended tasks (Seed et al., 2025). Such combinations improve coverage only when their components fail in diferent ways.

Reward validity and verifier noise. An observed RLVR gain does not by itself show that the reward captured the intended semantics. Random or format-only rewards can improve performance in some settings because of base-model priors and optimization dynamics (Shao et al., 2025a). Controlled verifier noise can be tolerated, but false positives are generally more damaging than false negatives (Plesner et al., 2026). Systematic false negatives tend to delay learning, while exploitable false-positive patterns can create suboptimal plateaus or collapse (Egashira et al., 2026). Incorrect annotations likewise reduce performance relative to clean supervision (Zhu & Kang, 2026). External adjudication is reliable only when the verifier specification, coverage, and implementation are themselves sound.

Specification exploitation. Rule-based mathematical verifiers can reject equivalent answers in unfamiliar formats, while learned verifiers may admit false positives that a policy can exploit (Huang et al., 2025c). Generative judges can reward punctuation-only responses or generic reasoning openers (Zhao et al., 2025c). Similar shortcuts appear when extensional checks reward enumeration in inductive logic (Helf et al., 2026) or when rubric biases become strategies during RL (Wang et al., 2026c). Taxonomies of reward hacking, controlled testbeds, and verifier fuzzing ofer complementary ways to identify these weaknesses before large-scale optimization (Wang et al., 2026d; Khalifa et al., 2026; Ray, 2026). Executed rewards remain grounded only to the extent that the implementation represents the intended success criterion.

Coupled policies and evaluators. Self-judging creates an additional failure mode because the policy and evaluator may adapt to the same errors. A policy optimized against a candidate-conditioned, reference-free judge can learn to make incorrect answers more convincing without making them more accurate. These errors can transfer across judges from diferent model families (Zhou, 2026). A related collapse appears in SPELL when a verifier is updated only from its own majority-vote pseudo-labels. Errors can then accumulate until every responder output is labeled correct. SPELL limits this drift by checking verifier updates against rule-based outcomes on verifiable questions (Yang et al., 2026c). The result illustrates why recursive self-evaluation needs evidence that remains independent of both actor and judge.

## 3.4.2. Capability Expansion and Transfer

Reward improvement and capability expansion are related but not equivalent. RLVR can improve sampling eficiency while narrowing the set of problems solved by the trained policy (Yue et al., 2025). Under standard online optimization, exact support cannot extend beyond outputs that initially have nonzero probability, and finite sampling can shrink the empirical support observed during training (Wu & Choi, 2025). However, the severity of this limitation depends on the base policy, task, and training horizon. Prolonged training can shift from exploiting common solutions to increasing the probability of rare high-reward actions (Yao et al., 2025). ProRL likewise reports sustained boundary expansion on some dificult coding and out-of-distribution tasks, alongside plateaus on others (Liu et al., 2025f).

Transfer provides a second test of whether reward optimization changes general capability. In controlled mathematics experiments, RL tuning preserves or improves several nonmathematical abilities more reliably than SFT (Huan et al., 2026). This result does not imply that RLVR produces domaingeneral reasoning in every setting. Improvements can follow superficial heuristics in combinatorial optimization (Alam & Rastogi, 2025), and reasoning traces learned through RLVR contribute much less to general question answering than to verifiable reasoning (Li et al., 2026b). Apparently conflicting findings often reflect diferences in how well the base model is already aligned with the target task (Wu et al., 2026a).

Overall, rewards beyond human evaluation are best compared through the evidence that supports them. As the reward becomes less independent of the policy, measured gains depend more strongly on the base model, the task distribution, and exploitable regularities in the evaluator. Yet external execution is not suficient on its own because incomplete specifications can also reward unintended behavior. Reward cost, reward independence, and reward validity are distinct properties and must be evaluated separately. This section has focused on the source of reward. A method moves toward L3 only when it also generates or discovers the tasks, curricula, or environments that provide its experience, as discussed in Section 4.

## 4. Experience beyond Human Design

Section 3 examines how the source of evaluation changes along the reward axis. Obtaining rewards beyond human evaluation addresses only one part of the scaling problem. Even when trajectories can be evaluated without relying primarily on human judgment, training may remain confined to tasks selected by people and environments built in advance. Continued improvement then depends on whether the experience distribution can adapt with the policy, exposing the model to new problems and interactions without continual human curation. This motivates the central question of this section:

![](images/9ee9f9b45421dd0b5f06887e9a6ac5f438e7f049d41e63f148374c47c2443472.jpg)  
Figure 4: Taxonomy of experience supply beyond human design. The chapter begins with fixed humandesigned experience, then organizes adaptive experience methods into task generation, environment construction, integrated experience generation, and autonomous co-evolution. Systems are placed according to the experience component they most directly adapt; repeated systems span more than one component.

How can large reasoning models generate useful experience with less dependence on human-designed tasks and environments?

Changing the source of reward does not necessarily change the source of experience. Progress along the experience axis begins when the experience stream itself becomes adaptive, either through task generation in a fixed environment or through environment construction around external tasks and seeds. Combining these approaches can sustain a curriculum with less continual human curation. The progression culminates in a closed loop in which reward acquisition, experience generation, and policy improvement adapt together. As defined in Section 2.5, the ladder identifies the human input needed to sustain learning, not the historical origin of every dataset, tool, or environment.

Related work has approached this area from model-centric (Gao et al., 2025a; Fang et al., 2025a) and environment-centric (Huang et al., 2025b) perspectives. Our paper brings these perspectives together by tracking the reward R, the task distribution Π, and the environment dynamics (S, P) within the same learning loop. Table 3 records which components remain externally supplied at each stage. The discussion begins with fixed human-designed experience and then considers task generation and environment construction as two paths toward integrated experience generation. It concludes with autonomous co-evolution and the failure modes that arise as these components become coupled.

## 4.1. Fixed Human-Designed Experience

In the ladder, L2 marks a regime in which the reward source may change while experience remains fixed. Rewards may come from model behavior, available references, or environment outcomes. The queries are still sampled from a human-curated task distribution Π<sub>H</sub>, and interaction takes place in a human-provided

Table 3: Representative methods along the experience axis in Section 4. The symbols describe the operational source of the Reward R, Environment (S, P), and Task distribution Π. A filled circle indicates that the component is supplied or adapted within the learning loop, a half-filled circle indicates partial adaptation under an external scafold, and an empty circle indicates that the component remains fixed externally. These symbols do not imply the absence of human-designed seeds, rules, or tools. The three blocks correspond broadly to the transition from L2 to L3 and from L3 toward L4.
<table><tr><td>Method</td><td></td><td>R E</td><td>T</td><td>Mechanism</td><td>Domain</td></tr><tr><td colspan="6">Task generation in fxed environments</td></tr><tr><td>STaR (Zelikman et al., 2022)</td><td>O</td><td>O</td><td></td><td>rationale bootstrapping</td><td>math, commonsense</td></tr><tr><td>Self-Instruct (Wang et al., 2022)</td><td>O</td><td>O</td><td></td><td>self-instruction</td><td>instruction following</td></tr><tr><td>Evol-Instruct (Xu et al., 2023)</td><td>O</td><td>O</td><td></td><td>instruction evolution</td><td>general</td></tr><tr><td>SeRL (Fang et al., 2025b)</td><td></td><td>O</td><td></td><td>generation with consensus</td><td>math, general</td></tr><tr><td>CoT-Self-Instruct (Yu et al., 2025b)</td><td></td><td>O</td><td></td><td>domain-aware synthesis</td><td>reasoning</td></tr><tr><td>R-Zero (Huang et al., 2025a)</td><td></td><td></td><td></td><td>challenger and solver</td><td>general reasoning</td></tr><tr><td>SQLM (Chen et al., 2025c)</td><td></td><td></td><td></td><td>proposer and voting solvers</td><td>reasoning</td></tr><tr><td>LSP (Kuba et al., 2025)</td><td></td><td>O</td><td></td><td>minimax self-play</td><td>general</td></tr><tr><td>MAE (Chen et al., 2025j)</td><td></td><td>O</td><td></td><td>proposer, solver, and judge</td><td>general</td></tr><tr><td>Socratic-Zero (Wang et al., 2025f)</td><td></td><td>O</td><td></td><td>teacher, solver, and generator</td><td>math</td></tr><tr><td>G-Zero (Huang et al., 2026b)</td><td></td><td>O</td><td></td><td>intrinsic hint reward</td><td>open-ended</td></tr><tr><td>SPICE (Liu et al., 2025b)</td><td></td><td>O</td><td></td><td>corpus-grounded self-play</td><td>reasoning</td></tr><tr><td>OpenSIR (Kwan et al., 2025)</td><td></td><td>O</td><td></td><td>diversity and difficulty</td><td>math</td></tr><tr><td>AgentSynth (Xie et al., 2025)</td><td></td><td></td><td></td><td>subtask composition</td><td>computer use</td></tr><tr><td colspan="6">Environment construction and integrated experience generation</td></tr><tr><td>MemAgent (Yu et al., 2025a)</td><td></td><td></td><td></td><td>learned interaction memory</td><td>long-context</td></tr><tr><td>ZeroGUI (Yang et al., 2025b)</td><td></td><td></td><td></td><td>interface-grounded generation</td><td>GUI</td></tr><tr><td>SEAgent (Sun et al., 2025d)</td><td></td><td></td><td></td><td>learned world-state evaluator</td><td>computer use</td></tr><tr><td>Voyager (Wang et al., 2023a)</td><td></td><td></td><td></td><td>curriculum and skill library</td><td>embodied agents</td></tr><tr><td>Eureka (Ma et al., 2023)</td><td></td><td></td><td></td><td>reward-code evolution</td><td>robotics</td></tr><tr><td>Genie (Bruce et al., 2024)</td><td></td><td></td><td></td><td>generative world model</td><td>interactive worlds</td></tr><tr><td>PAIRED (Dennis et al., 2020)</td><td>O</td><td></td><td></td><td>regret-based environment design</td><td>RL environments</td></tr><tr><td>AWM (Wang et al., 2026e)</td><td></td><td></td><td></td><td>code-driven environment synthesis</td><td>tool use</td></tr><tr><td>ScaleEnv (Tu et al., 2026)</td><td></td><td></td><td></td><td>environment synthesis</td><td>tool use</td></tr><tr><td>CuES (Mai et al., 2025)</td><td></td><td></td><td></td><td>curiosity-driven task synthesis</td><td>agentic</td></tr><tr><td>EvoEnv (Shi et al., 2026)</td><td></td><td></td><td></td><td>validated environment synthesis</td><td>reasoning</td></tr><tr><td>Env-Tuning (Lu et al., 2025c)</td><td></td><td></td><td>O</td><td>environment augmentation</td><td>tool use</td></tr><tr><td colspan="6">Coupled curricula and co-evolving learning loops</td></tr><tr><td>AZR (Zhao et al., 2025a)</td><td></td><td></td><td></td><td>proposer, solver, and executor</td><td>code, math</td></tr><tr><td>CURE (Wang et al., 2025k)</td><td></td><td></td><td></td><td>code and test co-evolution</td><td>code</td></tr><tr><td>Self-Challenging (Zhou et al., 2025c)</td><td></td><td></td><td></td><td>task and verifier generation</td><td>tool use</td></tr><tr><td>SPIRAL (Liu et al., 2025a)</td><td></td><td></td><td></td><td>adaptive self-play</td><td>games, reasoning</td></tr><tr><td>POET (Wang et al., 2019)</td><td>O</td><td></td><td></td><td>open-ended environment generation</td><td>RL environments</td></tr><tr><td>OMNI-EPIC (Faldor et al., 2024)</td><td></td><td></td><td></td><td>open-ended code generation</td><td>RL environments</td></tr><tr><td>GenEnv (Guo et al., 2025c)</td><td></td><td></td><td></td><td>difficulty-aligned co-evolution</td><td>agentic</td></tr><tr><td>Agent-World (Dong et al., 2026)</td><td></td><td></td><td></td><td>adaptive arena construction</td><td>tool use</td></tr></table>

environment $\mathcal { E } _ { \mathrm { H } } = ( S _ { \mathrm { H } } , \mathcal { A } _ { \mathrm { H } } , \mathcal { P } _ { \mathrm { H } } )$ . The corresponding objective can be written schematically as

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \pmb { q } \sim \Pi _ { \mathrm { H } } , \tau \sim p _ { \theta } ( \cdot | \pmb { q } , \mathcal { E } _ { \mathrm { H } } ) } \left[ \mathcal { R } ( \pmb { q } , \tau ) \right] .\tag{17}
$$

Eq. 17 separates the reward mechanism from the source of experience. Changing how R is obtained does not alter the externally supplied task distribution or environment. The policy can explore the available tasks and dynamics, but the choice of new tasks and environments remains outside the learning loop. This fixed setting provides the departure point for the experience axis.

Externally supplied task distribution. Even when the reward no longer depends primarily on human evaluation, $\Pi _ { \mathrm { H } }$ still embodies human decisions about what the policy encounters and how often. Corpus selection and domain boundaries determine its coverage, while task templates, dificulty ranges, filtering rules, and sampling frequencies allocate training efort within that coverage. For mathematical reasoning, this may mean drawing from curated collections such as GSM8K (Cobbe et al., 2021). For coding, it may mean using human-written specifications and tests in the style of HumanEval (Chen et al., 2021). For instruction following, it may mean training on a fixed corpus whose topical and stylistic coverage was selected in advance. These design choices concentrate gradient signal on particular capabilities and determine which failures training can expose. Model-derived or automatically computed rewards may improve performance within a narrow $\Pi _ { \mathrm { H } }$ , but they cannot reveal domains that are absent from the data or repair a poorly calibrated curriculum on their own.

Externally supplied environment. Humans also determine the world in which trajectories unfold. In single-turn reasoning, the “environment” may be nearly invisible, consisting of a static prompt, a tokenizer-level action space, and a fixed answer interface. Coding tasks add a programming language, sandbox, libraries, and execution protocol. Retrieval and tool-use settings add corpora, APIs, tool schemas, and state transitions, while GUI-based, embodied, and game agents depend on applications or simulators that define their observations, legal actions, and interaction horizons. Even outcomes such as successful compilation, retrieval, or gameplay are produced within dynamics selected and engineered by humans. The model may explore those dynamics extensively, but it cannot broaden or repair a fixed environment when the necessary tools, states, or causal mechanisms are missing.

Reward signals under fixed experience. When applied to fixed tasks and environments, the methods in Section 3.3 change the source of evaluation without changing the source of experience. TTRL (Zuo et al., 2025b) and MM-UPT (Wei et al., 2025b) replace external labels with majority-vote pseudo-labels while retaining the supplied test instances. EM-RL (Agarwal et al., 2025) derives reward from the policy’s own entropy, but the questions on which certainty is optimized remain externally chosen. VeriFree, RLPR, and NOVER (Zhou et al., 2025b; Yu et al., 2025d; Liu et al., 2025h) use the likelihood or perplexity of a supplied reference as a soft consistency signal. They remove the need for a separate verifier while retaining the query and reference pairs that define the experience. Similarly, execution feedback can provide R through program outcomes while the interpreter, unit-test interface, or game rules remain fixed. Across these methods, the reward source changes while $( \Pi _ { \mathrm { H } } , { \cal E } _ { \mathrm { H } } )$ remains externally supplied.

Benefits and limits of fixed experience. Fixing the source of experience provides a stable basis for comparing reward signals because the target distribution does not change with the policy and an external source of grounding is preserved. With $\Pi _ { \mathrm { H } }$ and $\mathcal { P } _ { \mathrm { H } }$ held stationary, gains can be attributed more cleanly to policy optimization than to changes in the curriculum. This stability makes fixed experience a useful starting point, allowing a policy to learn from consensus, certainty, reference consistency, or world outcomes before it begins to generate new experience (Zuo et al., 2025b; Agarwal et al., 2025; Zhou et al., 2025b; Guo et al., 2025a).

The same stability also limits further improvement because the diversity and horizon of experience cannot exceed what $\Pi _ { \mathrm { H } }$ and $\varepsilon _ { \mathrm { H } }$ support. Repeated sampling and additional test-time compute can improve coverage on existing problems, but they do not broaden the task or environment distribution by themselves (Brown et al., 2024; Snell et al., 2024). Moving beyond this setting requires adaptation in what problems are encountered, where their consequences unfold, or both. Adaptive curricula and open-ended environment generation pursue this expansion by changing which tasks or environments enter learning (Chen et al., 2025i; Wang et al., 2019; Faldor et al., 2024). A new reward source alone does not provide that adaptation.

## 4.2. From Fixed to Generated Experience

Some recent methods generally relax one source of human design at a time. Some generate tasks while retaining a fixed environment, whereas others construct environments around externally supplied tasks or seeds, adapting Π and (S, P), respectively. Neither route by itself produces a continuing stream of both new tasks and the contexts needed to execute them, but each provides a building block for the integrated systems discussed in Section 4.3.

## 4.2.1. Task Generation in Fixed Environments

One way to reduce reliance on a human-curated task distribution is to let the model generate or select tasks while keeping the environment fixed. Depending on the domain, this fixed substrate may include a code interpreter, mathematical checker, or retrieval corpus. In the objective of Eq. 1, queries are drawn from a model-parameterized proposer $q \sim \Pi _ { \theta }$ instead of the fixed distribution $\Pi _ { \mathrm { H } }$ , while the transition dynamics $\mathcal { P }$ and the verifier remain externally supplied. Generated tasks must be solvable, diverse, and calibrated to the policy’s current ability. Proposer objectives often capture this requirement through learnability, which favors tasks that remain informative without becoming trivial or efectively unsolvable. Because validation still relies on external dynamics and rules, these methods adapt the task distribution without yet adapting the environment.

Self-generated reasoning traces and instruction data. STaR (Zelikman et al., 2022) provides an early template by keeping the questions, environment, and answer-checking reward fixed while generating intermediate reasoning traces � and retaining only those that reach the correct answer $\boldsymbol { y } ^ { * }$ . Later work extends generation from traces to the problems themselves. Self-Instruct (Wang et al., 2022) synthesizes instruction, input, and output tuples and removes near-duplicates before fine-tuning, while Evol-Instruct (Xu et al., 2023) rewrites seed instructions with harder constraints or new topics. SeRL (Fang et al., 2025b) generates reasoning instances from a limited seed set, filters them by content, length, similarity, and dificulty, and trains on the retained instances with consensus-based rewards. CoT-Self-Instruct (Yu et al., 2025b) conditions synthesis on the intended domain and complexity, using answer consistency to filter verifiable tasks and a preference model to filter unverifiable ones. Ren et al. (Ren et al., 2025a) control compositional dificulty by adding or removing constraints. The same pattern extends to agentic settings, where ZeroGUI (Yang et al., 2025b) proposes GUI tasks from the current interface and AgentSynth (Xie et al., 2025) combines individually solvable subtasks into longer computer-use tasks. These pipelines reduce dependence on human-authored corpora while leaving the verification environment fixed.

Proposer and solver self-play. Task generation can also be framed as an asymmetric game between a proposer $\pi _ { \boldsymbol { \theta _ { P } } }$ and a solver $\pi _ { \theta _ { S } }$ , which are often instantiated by the same base model in diferent roles. The proposer reward determines whether this game yields a useful curriculum. Rewarding only solver failure pushes Π toward unsolvable noise, while rewarding only successful solutions encourages trivial tasks. A learnability signal avoids both extremes by peaking at intermediate solver success. Writing $\bar { p } _ { \theta _ { S } } ( { \pmb q } )$ for the empirical solve rate of a proposed query over � solver rollouts,

$$
\bar { p } _ { \theta _ { S } } ( q ) = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \kappa [ y _ { i } = \hat { y } ^ { * } ( q ) ] , \qquad \mathcal { R } _ { \mathrm { p r o p o s e } } ( q ) = \bar { p } _ { \theta _ { S } } ( q ) \big ( 1 - \bar { p } _ { \theta _ { S } } ( q ) \big ) ,\tag{18}
$$

so that queries that are neither always solved $( \bar { p } { = } 1 )$ nor never solved $( \bar { p } { = } 0 )$ receive the highest proposer reward. Absolute Zero Reasoner (AZR) (Zhao et al., 2025a) lets the model act as both proposer and solver, combining a learnability score of the form in Eq. 18 with environment-validated feedback to target the solver’s competence frontier. R-Zero (Huang et al., 2025a) trains a challenger to pose questions near that frontier while the solver learns from majority-vote pseudo-labels. SQLM (Chen et al., 2025c) uses the same structure and obtains the proxy label $\hat { y } ^ { * }$ through voting across independent solvers. Language Self-Play (LSP) (Kuba et al., 2025) removes the initial dataset by formulating task generation as a minimax game:

$$
\operatorname* { m a x } _ { \theta _ { S } } \operatorname* { m i n } _ { \theta _ { P } } \mathbb { E } _ { { \pmb q } \sim \pi _ { \theta _ { P } } , { \pmb a } \sim \pi _ { \theta _ { S } } ( \cdot | { \pmb q } ) } [ \mathcal { R } ( { \pmb q } , { \pmb a } ) ] .\tag{19}
$$

The minimax coupling allows the task distribution to respond to the solver’s changing performance without an initial dataset. Recent work enriches the game with additional roles and internal signals. Multi-Agent Evolve (Chen et al., 2025j) uses proposer, solver, and judge roles from a shared backbone to estimate the dificulty and quality of open-ended questions without rule-checkable answers. Socratic-Zero (Wang et al., 2025f) coordinates teacher, solver, and generator roles, allowing the teacher to target the solver’s weaknesses and the generator to distill that teaching strategy into a scalable curriculum. AERO (Gao et al., 2026) uses entropy-based positioning and counterfactual correction to locate the solver’s learning frontier. G-Zero (Huang et al., 2026b) replaces an external judge with an intrinsic Hint-� reward that measures how a self-generated hint shifts the solver’s predictive distribution. Across these methods, the task supply adapts while the surrounding environment or validation mechanism remains fixed. Extending the loop further requires evaluation and experience generation to adapt together.

Corpus- and document-grounded self-play. Because ungrounded self-play tends to recycle concepts the solver already knows, corpus-grounded methods draw task content from external documents. SPICE (Liu et al., 2025b) alternates one model between a Challenger that mines documents and writes reasoning problems and a Reasoner that solves them. SCOPE (Kwan et al., 2026) applies a related design to open-ended tasks, with a Challenger that writes document-grounded questions near the Solver’s frontier and a frozen copy of the base model that produces task-specific rubrics and grades the responses. Search Self-play (Lu et al., 2025b) targets deep-search agents by generating search queries and using retrieval-augmented generation over each search trajectory to provide training targets. In each case, grounding expands the task content beyond the proposer’s existing priors.

Reward-free self-training and self-generated curricula. Several boundary methods leave the source of tasks and the environment essentially unchanged but adapt how the model trains on its own outputs. SePT (Li et al., 2025c) alternates between response sampling and fine-tuning on refreshed self-generated traces without a reward model, verifier, or teacher. Self-Evolving Curriculum (Chen et al., 2025i) models task ordering as a non-stationary multi-armed bandit whose arms represent dificulty levels or problem categories and whose payof is the absolute policy-gradient advantage. E2H Reasoner (Parashar et al., 2025) similarly orders tasks from easy to hard and gradually reduces exposure to easy problems. At test time, TTCS (Yang et al., 2026a) co-evolves a synthesizer and solver by generating local curriculum variants that help the solver bootstrap from easier neighboring problems under self-consistency rewards. OpenSIR (Kwan et al., 2025) combines diversity and dificulty rewards so that a teacher and student pair can expand beyond concepts represented by a minimal seed. These methods improve the organization of experience without independently changing both its task and environment sources.

## 4.2.2. Environment Constructionfor Fixed or Seeded Tasks

A complementary route leaves the task distribution largely intact but lets the model construct or expand the environment (S, P) in which experience is gathered. The limiting factor is often the cost and brittleness of the interactive world needed to produce and verify trajectories, not the availability of questions. This constraint is especially important for long-horizon agentic tasks (Li et al., 2026e) in the sequential POMDP regime (�>1) introduced in Section 2.1, where a human-authored simulator or verifier may account for much of the development efort. Current approaches use adversarial environment design, executable code synthesis, learned world models, live applications, and persistent memory to construct the dynamics and interfaces through which seeded tasks become useful experience.

Automatic environment design. Work on Unsupervised Environment Design (UED) predates LRMs, with PAIRED (Dennis et al., 2020) formalizing the approach through a game between an environmentgenerating adversary and a protagonist and antagonist pair. To avoid generating worlds that are merely impossible, the adversary maximizes regret instead of the protagonist’s failure. Regret is the gap between the return of the best-responding antagonist $\pi ^ { A }$ and the protagonist $\pi ^ { P }$ on a generated query �,

$$
\mathrm { R e g r e t } ( { \pmb q } ) = U ( \pi ^ { A } , { \pmb q } ) - U ( \pi ^ { P } , { \pmb q } ) ,\tag{20}
$$

which becomes informative when the antagonist can solve an environment that challenges the protagonist. For environment construction, regret serves the same purpose that the learnability signal in Eq. 18 serves for task generation. Both create an automatic curriculum that favors experience near the policy’s competence frontier. Subsequent UED variants refine the estimation of regret and the sampling of environment parameters, but retain the principle that useful environments should neither under-challenge nor overwhelm the current policy.

Environments as executable code. Representing the environment as code ofers a practical route for LRMs. Eureka (Ma et al., 2023) uses an LLM to write reward code and refine it through simulator feedback, thereby automating part of reward design within E. OMNI-EPIC (Faldor et al., 2024) generates environment and termination code and then applies an interestingness model to retain tasks that are both learnable and novel. For tool use, Self-Challenging agents (Zhou et al., 2025c) pair each generated task with an executable verification function, allowing code execution to evaluate the resulting trajectory.

Scalable synthesis of executable tool environments. Because interactive environments are often scarcer than questions in agentic RL, recent work seeks to increase their supply without sacrificing reliability. Agent World Model (Wang et al., 2026e) decomposes an environment into a stateful backend, tool interface, and success criteria, then uses a staged LLM pipeline to synthesize database-backed environments and correct invalid components. ScaleEnv (Tu et al., 2026) expands tool-dependency graphs and verifies actions through execution, producing interactive environments together with verifiable tasks. RandomWorld (Sullivan et al., 2025) procedurally generates tools and compositional tasks from a fine-grained type hierarchy. CuES (Mai et al., 2025) explores a new environment through intrinsic curiosity and abstracts observed interactions into reusable task schemas. EvoEnv (Shi et al., 2026) constructs Python environments through staged validation, semantic review, dificulty calibration, and novelty checks. Environment Tuning (Lu et al., 2025c) augments the environment with corrective feedback and a progress-based curriculum, allowing training to move beyond static trajectories.

Generative world models. An alternative to symbolic code is to learn the dynamics P directly. Genie (Bruce et al., 2024) trains an action-controllable world model from unlabeled videos without ground-truth action labels, showing how (S, P) can be learned from passive data instead of being specified entirely by hand.

Persistent and adaptive interaction contexts. Live applications and persistent memory ofer another way to extend the context in which experience is gathered. ZeroGUI (Yang et al., 2025b) uses a vision-language model to generate tasks from the current interface and estimate rewards from trajectory screenshots, reducing its reliance on handwritten verifier scripts and annotated success trajectories. SEAgent (Sun et al., 2025d) trains a World State Model to describe the state of a computer environment and evaluate actions along a trajectory. MemAgent (Yu et al., 2025a) learns memory that can be read and updated across conversation turns, extending the horizon over which experience can accumulate.

Voyager (Wang et al., 2023a) combines an automatic curriculum with a growing library of executable skills in an embodied environment. More broadly, experience-driven lifelong learning frameworks (Cai et al., 2025) emphasize the accumulation and reuse of past interactions, consistent with the Experience Era view that sustained interaction can support durable capability growth (Silver & Sutton, 2025).

## 4.3. Integrated Experience Generation

Integrated experience generation corresponds to L3 in the ladder, where the task and environment pipeline can continue to produce training trajectories after receiving only limited external seeds. The policy need not create every artifact itself because another model may synthesize the experience, a corpus may supply its content, or a live environment may provide the interactions. Such sources can continue to ground learning without requiring people to curate each new task and setting. A method enters this regime when its generated or discovered experience supports continued training with limited additional curation.

Integrated task and environment pipelines. Several recent systems combine automatically supplied tasks, executable environments, and automatic feedback in one pipeline. ZeroGUI (Yang et al., 2025b) illustrates a boundary case in the GUI domain by using a vision-language model to propose tasks from a live interface and estimate rewards from trajectory screenshots under majority confirmation. Because the application dynamics and pretrained visual evaluator remain externally supplied, the method reduces task curation and handcrafted reward annotation without independently generating its verifier or environment. Agent World Model (Wang et al., 2026e) uses a staged LLM pipeline with automated correction to synthesize stateful backends, tool interfaces, tasks, and success criteria from compact schemas. ScaleEnv (Tu et al., 2026) combines tool-dependency-graph expansion with executable action verification to construct solvable tasks and interactive environments across multiple domains. EvoEnv (Shi et al., 2026) admits a synthesized Python environment only after validation, semantic review, dificulty calibration, and novelty checks. These systems reduce continued human curation by generating both the setting and the tasks used for interaction. CuES (Mai et al., 2025) marks a boundary because it derives reusable task schemas through curiosity-driven exploration without handcrafted task seeds while retaining an externally supplied initial environment.

Boundary cases. When a proposer and solver loop invents mathematical questions while relying on a fixed Python interpreter, the task distribution adapts but execution and checking remain externally designed. A generative world model such as Genie (Bruce et al., 2024) changes the environment while leaving the downstream task distribution and reward externally specified. These cases reduce human design along only one part of the experience axis. Integrated experience generation requires the generated or discovered tasks and the contexts needed to execute them to provide an ongoing sequence of (�, �, R) for training. This criterion still permits seeds, corpora, and live applications to provide external grounding.

## 4.4. Autonomous Co-evolution

Autonomous co-evolution corresponds to L4 in the ladder, where reward acquisition, task generation, environment construction, and policy improvement form a persistent adaptive loop in which changes to one component reshape the others. Humans may still define broad objectives, operational boundaries, and independent audits without maintaining the curriculum or evaluator throughout training. This regime connects current LRM research with earlier work on open-ended learning. POET (Wang et al., 2019) co-evolves populations of environments and agents while transferring solutions across an expanding curriculum, although its reward remains externally specified. OMNI-EPIC (Faldor et al., 2024) generates environment and reward code before filtering for tasks that are both learnable and interesting. Self-Rewarding Language Models (Yuan et al., 2024) couple reward and policy by using the policy as its own judge across successive iterations, while leaving the task distribution and environment human-designed. These examples couple diferent subsets of the learning loop, whereas the full regime requires experience generation and evaluation to continue improving together without an externally maintained curriculum.

Coupled generator and verifier loops. CURE (Wang et al., 2025k) couples code and unit-test generation so that programs expose weaknesses in generated tests and stronger tests provide more informative feedback for subsequent programs. The executable runtime remains fixed, but the efective verification environment adapts with the generator. AZR (Zhao et al., 2025a) and R-Zero (Huang et al., 2025a) close the loop between proposer and solver while retaining a fixed execution or validation mechanism, making them examples of co-evolving curricula within an externally defined environment. Self-Challenging agents (Zhou et al., 2025c) synthesize Code-as-Task instances that bundle an instruction, executable verification function, reference solution, and failure cases, thereby coupling task generation with executable feedback while leaving the underlying tool system externally defined.

Self-play and adaptive curricula. SPIRAL (Liu et al., 2025a) trains a single policy in both roles of a two-player zero-sum language game against improving copies of itself. As the opponent strengthens under an objective of the form in Eq. 19, the sequence of matchups becomes an adaptive curriculum that can elicit transferable reasoning strategies. Because the game rules remain fixed, SPIRAL adapts its opponents and curriculum without generating the underlying environment. The shifting opponent also introduces non-stationarity and higher gradient variance. Role-conditioned advantage estimation addresses this instability by normalizing each rollout against the expected return of its role, analogous to the group-relative baseline of GRPO in Eq. 6. Closed-loop experience generation depends on the same recurrent relationship: policy improvements must induce more informative tasks and environments, which must then support further policy improvement. POET (Wang et al., 2019) and OMNI-EPIC (Faldor et al., 2024) pursue this form of open-ended coupling in RL settings (Silver & Sutton, 2025).

Dificulty-aligned agent and environment co-evolution. GenEnv (Guo et al., 2025c) makes the environment a learned component by pairing an agent LLM with a generative simulator. An �-curriculum reward steers the generated tasks toward the agent’s zone of proximal development, extending the regret objective in Eq. 20 and the learnability signal in Eq. 18 to a learned environment. Agent-World (Dong et al., 2026) constructs an arena from databases and executable toolsets, then uses successive rounds of multi-environment RL to diagnose capability gaps and synthesize targeted tasks. Competitive game engines provide another source of adaptive experience. TextArena (Guertler et al., 2025) ofers language games whose self-play dynamics create an automatic dificulty curriculum for social and strategic skills while providing a substrate for methods such as SPIRAL.

Open-ended self-improvement of the agent itself. Some systems extend co-evolution beyond tasks and environments to the design of the agent itself (Liu et al., 2026e,f,d). The Darwin Gödel Machine (Zhang et al., 2025b) maintains an archive of coding agents that modify their own code and branch from archived ancestors to explore alternative designs. Foundation Model Self-Play (Dharna et al., 2025) proposes and refines agent strategies through competitive play and quality-diversity search. Automated Capability Discovery (Lu et al., 2025a) uses one foundation model to propose and evaluate tasks for another, exposing capabilities and failures without relying on a fixed hand-designed benchmark.

## 4.5. Failure Modes across the Experience Axis

As task and environment generation become adaptive, errors in the generated experience can feed directly into subsequent policy updates. On the task side, poor dificulty calibration and limited diversity can deprive the policy of useful learning signals. On the environment side, inaccurate or expensive simulations can undermine the trajectories they produce. Once these components are coupled, non-stationarity and mutual reinforcement create additional paths for errors to persist.

Dificulty miscalibration. If an unconstrained proposer drifts toward trivial or unsolvable tasks, group rewards become nearly constant and the GRPO advantages $\hat { A } _ { i }$ in Eq. 6 vanish, causing learning to stall. The learnability signal in Eq. 18, the regret objective in Eq. 20, and the frontier-targeting challenger in

R-Zero (Huang et al., 2025a) all seek to preserve an informative dificulty range. Their efectiveness depends on estimates of the solve rate $\bar { p } _ { \theta _ { S } }$ , which may be noisy early in training.

Co-adaptive reward hacking. When R, Π, and (S, P) are generated within the learning loop, the solver may exploit blind spots in the verifier instead of acquiring genuine capability, extending the reward-hacking problem discussed in Section 3. Drift becomes harder to detect when the verifier changes with the policy, and the generator and grader may converge on conventions that are internally consistent but externally invalid. Unit tests in CURE (Wang et al., 2025k), code verification in Self-Challenging agents (Zhou et al., 2025c), and game outcomes in SPIRAL (Liu et al., 2025a) constrain this behavior without eliminating systematic bias in the generated tests or games. Internal grading creates a further asymmetry because models may improve at producing answers without becoming equally reliable at checking them (Chen et al., 2026b). Reliable verification may therefore require its own training objective and independent audits.

Distributional narrowing and mode collapse. Self-generated task streams can concentrate on regions the model already handles well, reducing the diversity of Π over successive generations. Multidimensional filtering in SeRL (Fang et al., 2025b), the adversarial objective of LSP (Kuba et al., 2025) in Eq. 19, and interestingness models in open-ended learning (Faldor et al., 2024; Wang et al., 2019) introduce diversity pressure that helps keep the curriculum expanding.

Non-stationarity and instability. In fully coupled loops, the environment or opponent changes as the policy learns, departing from the fixed data and reward assumptions common in single-turn RLVR. The resulting non-stationarity can increase gradient variance in multi-agent training, although role-conditioned or group-relative baselines (Liu et al., 2025a) can improve stability. It also compounds the long-horizon credit-assignment dificulty of the �>1 regime, where a single scalar outcome must be distributed over an entire self-generated trajectory.

Environment fidelity and cost. The reliability of a constructed environment depends on the models that build and evaluate it. Vision-language reward estimation (Yang et al., 2025b) and world-state modeling (Sun et al., 2025d) inherit errors and biases from their underlying models, which can propagate through every trajectory collected in the environment. Verifying a generated simulator or test suite may also reintroduce much of the supervision cost that autonomous experience generation seeks to reduce. Evidence from closed-loop self-evolution further suggests that internally generated supervision may plateau below oracle supervision (Qi et al., 2026b), supporting the use of independent anchors even in a largely autonomous experience pipeline.

Domain landscape and evaluation implications. The main constraint on autonomous experience generation difers by domain. In mathematics and code, executable checkers provide relatively strong grounding for generated tasks, although proposers may still collapse to recycled templates without corpus grounding (Liu et al., 2025b; Kwan et al., 2025). In tool use and computer control, environment fidelity is often a tighter constraint than question scarcity (Wang et al., 2026e; Tu et al., 2026; Zhou et al., 2025c). Evaluation must therefore measure simulator correctness and transfer to held-out applications as discussed in Section 5.3. In open-ended and social settings, self-play can produce an adaptive curriculum (Liu et al., 2025a; Guertler et al., 2025), but the same coupling can amplify collusion between the proposer and judge (Yuan et al., 2024).

The goal is not to maximize autonomy at the expense of grounding. Stable learning may still require consensus, execution feedback, corpus grounding, or occasional human audits. The relevant question is whether continual human curation can be reduced while enough independent evidence remains to detect drift and collapse, which motivates the evaluation criteria developed in the next section.

![](images/1c206dd8d49cf7bf2dc1e3700a080feb52a31ab80c9d0cb5d5ed4a58b7f3aaf2.jpg)  
Figure 5: Evaluation taxonomy for scaling beyond human supervision. A complete evaluation jointly assesses capability on frozen benchmarks, the fidelity of training feedback, the quality of generated experience, and reproducibility under contamination-aware protocols.

## 5. Evaluation, Datasets, and Benchmarks

Evaluating learning beyond human supervision requires evidence about three distinct objects: the policy, the feedback used to train it, and the experience from which it learns. A higher held-out score establishes a change in capability, but does not establish that the training reward remained faithful or that a generated curriculum remained broad and valid. Our evaluation framework consequently separates policy capability, feedback fidelity, and experience quality. These forms of evidence become jointly important as reward acquisition and experience generation move into the learning loop.

The balance among these three forms of evidence changes across the ladder in Sec. 2.5. At L0, human judgments directly define the held-out standard. At L1 and L2, evaluation must also test the reusable evaluators and reward sources used for learning. At L3, task and environment quality become explicit evaluation targets, while L4 requires longitudinal evidence about the stability of the coupled learning loop. Tables 4 and 5 summarize representative suites and metrics. Table 6 translates the three-part framework into reporting requirements for the five ladder levels.

## 5.1. Capability Evaluation

Capability evaluation measures whether the policy $\pi _ { \theta }$ solves the intended problems under a held-out standard. The source of the training reward does not determine the appropriate capability test. A policy trained with model-derived rewards may still be evaluated on human-authored mathematics problems, while an agent trained in evolving environments may still be tested in a fixed sandbox. We group capability benchmarks by how their outputs are assessed because this choice determines the reliability, cost, and scalability of evaluation.

## 5.1.1. Hard-verifiable reasoning

Hard-verifiable tasks admit deterministic checking through exact numerical or symbolic matching, unit tests, game outcomes, or constraint satisfaction. They support scalable evaluation and provide a comparatively reliable external anchor for methods trained with reduced supervision.

Mathematical reasoning. Mathematics is among the most widely used domains for evaluating LRMs (Ahn et al., 2024). Established suites include GSM8K (Cobbe et al., 2021), MATH and MATH500 (Hendrycks et al., 2021b; Lightman et al., 2023), and competition problems from AMC and AIME (Li et al., 2024b; Mathematical Association of America, 2024, 2025). Omni-MATH (Gao et al., 2024), OlympiadBench (He et al., 2024a), FrontierMath (Glazer et al., 2024), LiveMathBench (Liu et al., 2024b), and RandomCalculation (Wu et al., 2025b) extend coverage to harder or more dynamic settings. PutnamBench evaluates formal theorem proving (Tsoukalas et al., 2024), while SciBench tests college-level scientific problem solving (Wang et al., 2023b).

MathArena broadens this landscape with a continuously maintained collection spanning olympiad problems, proof-oriented competitions, research questions drawn from arXiv, and formal proof generation in Lean (Dekoninck et al., 2026). Its tasks distinguish final-answer accuracy from proof construction and formalization. This distinction matters because Pass@1 does not reveal whether a model is consistent across samples or whether its intermediate reasoning is valid. Pass@�, mean sample accuracy, G-Pass@�, and CoT-Pass@� provide complementary evidence, as discussed in Sec. 5.5. Results also benefit from dificulty and subset breakdowns. A single aggregate on MATH can conceal whether gains are confined to routine algebra or extend to olympiad-level problems (Gao et al., 2024; Glazer et al., 2024). Fixed grading scripts further improve comparisons across models. GSM-Symbolic adds a robustness check by changing names, values, and templates while preserving the underlying reasoning structure (Mirzadeh et al., 2024).

Code generation and software engineering. Executable tests provide a direct measure of functional correctness in code generation. HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and EvalPlus (Liu et al., 2023a) are widely used for this purpose. BigCodeBench (Zhuo et al., 2024), LiveCodeBench (Jain et al., 2025), CodeContests (Li et al., 2022), CRUXEval (Gu et al., 2024a), CodeElo (Quan et al., 2025), and MHPP (Dai et al., 2024) expand functional and distributional coverage. Similarity metrics such as CodeBLEU (Ren et al., 2020), CodeBERTScore (Zhou et al., 2023b), and ICE-Score (Zhuo, 2024) can complement execution, but they are less direct measures of task completion when reliable tests are available.

Repository-level benchmarks assess a broader form of software engineering. SWE-bench (Jimenez et al., 2024), Multi-SWE-bench (Zan et al., 2025), and SWE-bench Multimodal (Yang et al., 2024a) evaluate repository patches, multilingual issue resolution, and visually grounded tasks. Active-SWE removes the supplied issue report and requires agents to discover and repair defects (Li et al., 2026a). SWE-bench Science covers issue-driven repair, expert exploration, and system integration in scientific repositories (Xu et al., 2026c). These suites evaluate problem identification, tool use, environment interaction, and patch generation together. For methods trained with executable rewards, the evaluation harness must be disjoint from the training tests to limit leakage. SciCode provides an additional transfer test through scientist-curated research programming problems (Tian et al., 2024).

Table 4: Representative suites and protocols for evaluating large reasoning models. Target identifies the component being evaluated. Evidence records the external signal used to support the measurement. The selection emphasizes coverage across evaluation targets and is not exhaustive.
<table><tr><td>Suite or protocol</td><td>Target</td><td>Domain</td><td>Evidence</td><td>Representative metrics</td></tr><tr><td colspan="5">Policy capability</td></tr><tr><td>MATH / MATH500 (Hendrycks et al., 2021b; Lightman et al., 2023)</td><td>policy</td><td>mathematics</td><td>reference answers</td><td>Acc. / Pass@k</td></tr><tr><td>Omni-MATH / FrontierMath (Gao et al., 2024; Glazer et al., 2024)</td><td>policy</td><td>advanced mathematics</td><td>exact or expert grading</td><td>Acc. / Pass@k</td></tr><tr><td>LiveBench / DyVal (White et al., 2025; Zhu et al., 2024)</td><td>policy</td><td>multiple domains</td><td>refreshed or perturbed ground truth</td><td>Acc.</td></tr><tr><td>HumanEval / EvalPlus (Chen et al., 2021; Liu et al., 2023a)</td><td>policy</td><td>code generation</td><td>held-out unit tests</td><td>Pass@k</td></tr><tr><td>BigCodeBench / LiveCodeBench (Zhuo et al., 2024; Jain et al., 2025)</td><td>policy</td><td>code generation</td><td>execution</td><td>Pass@k</td></tr><tr><td>SWE-bench (Jimenez et al., 2024)</td><td>policy</td><td>software engineering</td><td>repository tests</td><td>resolve rate</td></tr><tr><td>GPQA / HLE (Rein et al., 2024; Phan et al., 2025)</td><td>policy</td><td>expert reasoning</td><td>expert-authored labels</td><td>Acc.</td></tr><tr><td>SimpleOA / CRAG (Wei et al., 2024: Yang et al., 2024b)</td><td>policy</td><td>factuality and RAG</td><td>references or retrieval-backed grading</td><td>Acc. / graded score</td></tr><tr><td>Arena-Hard / WildBench (Li et al., 2024d; Lin et al., 2024a)</td><td>policy</td><td>open-ended interaction</td><td>frozen judges or human preferences</td><td>win rate</td></tr><tr><td>LongBench / HELMET / RULER (Bai et al., 2024b; Yen et al., 2025; Hsieh et al., 2024) policy</td><td></td><td>long context</td><td>task-specific ground truth</td><td>Acc. / recall</td></tr><tr><td>WebArena / OSWorld (Zhou et al., 2024; Xie et al., 2024b)</td><td>policy</td><td>web and computer use</td><td>executable task success</td><td>success rate</td></tr><tr><td>AgentBench / GAIA (Liu et al., 2024d; Mialon et al., 2024)</td><td>policy</td><td>general agents</td><td>task completion</td><td>Acc. / success rate</td></tr><tr><td colspan="5">Feedback fidelity</td></tr><tr><td>RewardBench / RewardBench 2 (Lambert et al., 2024; Malik et al., 2025)</td><td>feedback</td><td>outcome reward models</td><td>disjoint preference data</td><td>pairwise / best-of-N Acc.</td></tr><tr><td>RM-Bench / PPE (Liu et al., 2024e; Frick et al., 2025)</td><td>feedback</td><td>reward robustness and utility</td><td>controlled rewrites / downstream outcomes</td><td>Acc. / correlation</td></tr><tr><td>ProcessBench / PRMBench (Zheng et al., 2025a; Song et al., 2025b)</td><td>feedback</td><td>process reward models</td><td>step-error annotations</td><td>step F1 / PRM-Score</td></tr><tr><td>Socratic-PRMBench / VILBENCH (Li et al., 2025g: Tu et al., 2025</td><td>feedback</td><td>text and multimodal reasoning</td><td>fine-grained error labels</td><td>PRM-Score</td></tr><tr><td>JudgeBench / CodeJudgeBench (Tan et al., 2025; Jiang et al., 2025)</td><td>feedback</td><td>LLM judges</td><td>verifiable response pairs</td><td>Acc. / consistency</td></tr><tr><td>JETTS (Zhou et al., 2025e)</td><td>feedback</td><td>judge utility</td><td>downstream search and refinement</td><td>quality gain</td></tr><tr><td>VerifyBench / LongJudgeBench (Li et al., 2026c; Chen et al., 2026a)</td><td>feedback</td><td>expert and long-form judging</td><td>expert answers / controlled protocols</td><td>Acc. / agreement</td></tr><tr><td>LLMBar / CriticBench (Zeng et al., 2024; Lin et al., 2024b)</td><td>feedback</td><td></td><td>instruction judging and critique adversarial pairs / correction targets</td><td>Acc. / correction gain</td></tr><tr><td colspan="5">Experience quality</td></tr><tr><td>Generated-task audit</td><td></td><td>experience self-generated curricula</td><td>frozen solver and external seeds</td><td>solve-rate profile / learnability / diversity</td></tr><tr><td>Corpus-grounded task audit (Liu et al., 2025b; Kwan et al., 2026)</td><td></td><td>experience document-grounded curricula</td><td>source attribution</td><td>novelty / source coverage</td></tr><tr><td>Synthesized-environment audit</td><td></td><td>experience interactive environments</td><td>executable validation and frozen agents</td><td>validity / fidelity / transfer</td></tr><tr><td>AgentWorldBench (Zuo et al., 2026)</td><td></td><td>experience learned world models</td><td>observed environment transitions</td><td>transition fidelity</td></tr><tr><td>PAST-Bench (Xue et al., 2026)</td><td></td><td>experience persistent agents</td><td>matched sequences with and without memory retained-experience gain</td><td></td></tr><tr><td>AI4AI-Bench (Chi et al., 2026)</td><td></td><td>experience algorithm improvement</td><td>fixed repository and evaluator</td><td></td></tr><tr><td>Closed-loop audit</td><td></td><td>experience co-evolving learning loops</td><td>frozen external tests and component ablations diversity / drift / stability</td><td>external improvement</td></tr></table>

Table 5: Evaluation metrics grouped by the object they measure. No metric is suficient in isolation when the feedback or experience stream is generated within the learning loop.
<table><tr><td>Family</td><td>Representative metrics</td><td>Primary use</td></tr><tr><td>Outcome correctness</td><td>Exact match, numerical equivalence, unit-test Pass@k, success rate</td><td>Verifiable capability</td></tr><tr><td>Sampling reliability</td><td>Avg@G, G-Pass@k (Liu et al., 2024b), CoT-Pass@k (Wen et al., 2025)</td><td>Typical performance, consistency, and reasoning validity</td></tr><tr><td>Preference agreement</td><td>Pairwise Acc. (Lambert et al., 2024), listwise Acc., best-of-N Acc.</td><td>Outcome reward models and LLM judges</td></tr><tr><td>Judge utility and bias</td><td>Reranking gain (Zhou et al., 2025e), human-judge agreement, position and style sensitivity</td><td>Practical value and robustness of πy</td></tr><tr><td>Process fidelity</td><td>Step F1, PRM-Score (Li et al., 2025g; Song et al., 2025b)</td><td>Error detection within reasoning traces</td></tr><tr><td>Open-ended quality</td><td>Win rate, rubric score, G-Eval (Liu et al., 2023b)</td><td>Open-ended capability under a specified evaluator</td></tr><tr><td>Task-supply quality</td><td>Solve-rate distribution, p(1 – p), diversity, novelty, rejection rate</td><td>Self-generated curricula</td></tr><tr><td>Environment validity</td><td>Executability, validation pass rate, transition fidelity, frozen-agent solve rate</td><td>Synthesized environments</td></tr><tr><td>Closed-loop health</td><td>Internal reward versus external capability, diversity over time, non-stationarity, component-freezing ablations</td><td>Co-evolving learning loops</td></tr><tr><td>Evaluation validity</td><td>Contamination (Singh et al., 2024; Dekoninck et al., 2024), position bias (Shi et al., 2024), protocol variance</td><td>Trustworthy comparison</td></tr></table>

Logic puzzles, games, and multiple choice. ZebraLogic (Lin et al., 2025a), AutoLogi (Zhu et al., 2025b), and Sudoku-Bench (Seely et al., 2025) evaluate structured reasoning through constraint satisfaction. TextArena uses competitive language games to assess strategic behavior through observable outcomes (Guertler et al., 2025). Multiple-choice suites such as AQuA (Ling et al., 2017), CodeMMLU (Nguyen et al., 2025a), LogiQA (Liu et al., 2021), and BBH (Suzgun et al., 2022) enable inexpensive and reproducible comparisons. Their format can understate the dificulty of generating a complete solution and can reward option elimination. General365 addresses a diferent confound by using reasoning problems that require only K-12 background knowledge (Liu et al., 2026a). Its controlled variants help separate reasoning dificulty from specialized factual knowledge.

## 5.1.2. Semi-verifiable and expert-graded tasks

Some tasks have concise answers but require expertise or retrieval-backed evidence for reliable grading. GPQA (Rein et al., 2024), Humanity’s Last Exam (Phan et al., 2025), MMLU (Hendrycks et al., 2021a), MMLU-Pro (Wang et al., 2024c), and SuperGPQA (Du et al., 2025) test broad knowledge and multi-step reasoning across disciplines. Their expensive labels make them useful external anchors for assessing transfer beyond mathematics and code.

Table 6: Minimum evaluation evidence for claims at each ladder level. Capability remains an evaluation target throughout the ladder. Feedback and experience require additional source-matched audits as their provision moves into the learning loop.
<table><tr><td>Level</td><td>Policy capability</td><td>Feedback fidelity</td><td>Experience quality</td></tr><tr><td>L0</td><td>fixed evaluation protocol</td><td>coverage</td><td>Held-out performance under a Annotation quality, agreement, and Separation of human-supplied training and evaluation instances</td></tr><tr><td>L1</td><td>ator</td><td>Domain-matched capability and Accuracy, calibration, robustness, transfer under an external evalu- and bias of the reusable evaluator on disjoint data</td><td>Coverage and distribution shift within externally supplied tasks and environ- ments</td></tr><tr><td>L2</td><td>Held-out capability with stability or reasoning-validity metrics when including calibration, consensus supplied tasks and environments relevant</td><td>fragility, reference dependence, and checker validity</td><td>Audits matched to the reward source, Generalization across fixed externally</td></tr><tr><td>L3</td><td>Capability and transfer on frozen training stream</td><td>suites disjoint from the generated ifier, executable checker, or sparse vironment validity, solvability, and trans- human audit</td><td>Agreement with an independent ver- Task diversity, learnability, novelty, en- fer to external experience</td></tr><tr><td>L4</td><td>External capability and transfer Drift between internal reward and in- Closed-loop stability, non-stationarity, tracked throughout co-evolution</td><td>dependent external evaluation, includ- diversity over time, the relation between ing generator-verifier asymmetry</td><td>internal reward and external capability, and component-freezing ablations</td></tr></table>

Domain-specific suites cover medicine (Jin et al., 2021; Chen et al., 2025b; Zuo et al., 2025a), finance (Mateega et al., 2025; Chen et al., 2024d; Xie et al., 2024a; Islam et al., 2023), law (Fei et al., 2024; Li et al., 2024a; Guha et al., 2023; Fan et al., 2025), commonsense reasoning (Talmor et al., 2019; Sun et al., 2024), and practical biology research (Laurent et al., 2024). SimpleQA isolates short-form factuality (Wei et al., 2024), while CRAG evaluates retrieval-augmented generation (Yang et al., 2024b). These benchmarks retain an external grading standard even when the model was trained with rewards that did not depend on direct human evaluation. They consequently measure transfer under residual expert grounding, not the independence of the training reward itself.

## 5.1.3. Open-ended generation and interactive preference

Open-ended tasks lack a unique reference answer, so evaluation relies on preferences, rubrics, or live comparison. This setting creates a particular risk of evaluator entanglement when related LLM judges are used for both training and testing.

Instruction following, chat, and arenas. IFEval (Zhou et al., 2023a), AlpacaEval (Li et al., 2023b; Dubois et al., 2024), MT-Bench (Zheng et al., 2023), MT-Bench-101 (Bai et al., 2024a), Arena-Hard (Li et al., 2024d), WildBench (Lin et al., 2024a), and MixEval (Ni et al., 2024) evaluate instruction following and conversational quality. HELM (Liang et al., 2022) and FLASK (Ye et al., 2023) provide broader evaluation frameworks. Arena-style win rates capture comparative preference at scale, but remain sensitive to verbosity, response order, and judge choice (Shi et al., 2024; Dubois et al., 2024). A stronger protocol fixes the judge panel and measures its agreement with an independent human subset.

Creative, social, and long-context evaluation. LitBench (Fein et al., 2025), CS4 (Atmakuru et al., 2024), and SS-GEN (Feng et al., 2025b) probe creative and narrative writing. EQ-Bench (Paech, 2023), ToMBench (Chen et al., 2024g), EgoSocialArena (Hou et al., 2024), and SocialEval (Zhou et al., 2025a) examine social reasoning and theory of mind. Long-context suites include LongBench (Bai et al., 2024b), LongBench v2 (Bai et al., 2025), ∞Bench (Zhang et al., 2024b), HELMET (Yen et al., 2025), RULER (Hsieh et al., 2024), and NoLiMa (Modarressi et al., 2025). Multimodal evaluation is represented by MMMU-Pro (Yue et al., 2024a), CharXiv (Wang et al., 2024e), and Video-MME (Fu et al., 2024a). These benchmarks commonly combine objective items with LLM-based rubric scores and selective human review (Zheng et al., 2023; Liu et al., 2023b). Their scale is valuable, but judge bias can cause stylistic fluency to be mistaken for substantive quality.

Safety and adversarial robustness. HarmBench standardizes automated red teaming and measures both harmful compliance and excessive refusal (Mazeika et al., 2024). JailbreakBench supplies reproducible attack, defense, and threat-model protocols (Chao et al., 2024). Fixed external safety suites remain important when the training curriculum adapts because self-generated experience can underrepresent rare harms or amplify unsafe behavior.

Capability benchmarks establish whether performance improved under a specified evaluator. They cannot by themselves determine whether the feedback used during training was faithful or whether the experience distribution remained informative. A method may raise GSM8K accuracy through a poorly calibrated intrinsic reward, improve HumanEval while generating progressively narrower tasks, or gain arena wins by exploiting judge preferences (Dubois et al., 2024; Liu et al., 2024e). The next two subsections examine these missing forms of evidence directly.

## 5.2. Feedback Fidelity

Feedback evaluation tests whether the learning signal rewards the behavior it is intended to promote. This requires examining the reward mechanism itself, not only the capability of the resulting policy. The relevant object may be an outcome reward model, a process reward model, an LLM judge, or a model-derived signal such as certainty or consensus.

## 5.2.1. Outcome reward models

RewardBench evaluates outcome reward models through pairwise accuracy on preference triples spanning chat, reasoning, and safety (Lambert et al., 2024). RewardBench 2 adds unseen human prompts, best-offour selection, and tests of downstream utility (Malik et al., 2025). Long-form RewardBench extends evaluation to responses that require document-level assessment and tests sensitivity to response length and error position (Huang et al., 2026c). Personalized RewardBench examines whether a reward model follows user-specific rubrics when responses have similar overall quality (Ma et al., 2026).

Several benchmarks target specific sources of error. RM-Bench rewrites semantically equivalent responses in diferent styles to expose spurious preferences for length and formatting (Liu et al., 2024e). PPE tests whether reward-model proxy metrics predict downstream RLHF outcomes (Frick et al., 2025). RewardMATH studies robustness and overoptimization in mathematical reward models (Kim et al., 2024a), and VL-RewardBench extends reward evaluation to vision-language responses (Li et al., 2024c). HelpSteer2 (Wang et al., 2024d) and RewardAnything (Yu et al., 2025e) support multi-attribute or principle-conditioned evaluation. Across these settings, the central question is whether $\mathcal { R } _ { \phi }$ represents the intended criterion or a convenient correlate of it.

## 5.2.2. Process reward models

Process reward models require evaluation at the level where credit is assigned. ProcessBench (Zheng et al., 2025a) and PRMBench (Song et al., 2025b) ask verifiers to identify erroneous steps in mathematical trajectories. Socratic-PRMBench introduces fine-grained errors and summarizes their detection with PRM-Score (Li et al., 2025g). VILBENCH extends process-reward evaluation to vision-language reasoning (Tu et al., 2025). Final-answer discrimination and process-error detection measure diferent properties, so outcome accuracy must be accompanied by step-level F1 or PRM-Score.

## 5.2.3. LLMjudges and verifier utility

Generative judges can be tested for agreement with people, factual discrimination, and utility within an inference or critique pipeline. G-Eval (Liu et al., 2023b), PandaLM (Wang et al., 2023d), JudgeLM (Zhu et al., 2025a), Auto-J (Li et al., 2023a), Prometheus 2 (Kim et al., 2024b), FLAMe (Vu et al., 2024), and BiGGen Bench (Kim et al., 2025) cover rubric scoring, pairwise comparison, and fine-grained skill assessment. JudgeBench uses correct and incorrect responses drawn from verifiable sources to reduce reliance on stylistic cues (Tan et al., 2025). CodeJudgeBench applies the same principle to execution-checked code (Jiang et al., 2025), while JETTS evaluates judge utility in reranking, beam search, and critique refinement (Zhou et al., 2025e).

Other suites probe judge reliability across modalities and contexts. MLLM-as-a-Judge covers multimodal inputs (Chen et al., 2024a). VerifyBench compares specialized verifiers with general LLM judges on expert responses in science and mathematics (Li et al., 2026c). LongJudgeBench evaluates long-form outputs under diferent judging protocols (Chen et al., 2026a), and JudgeArena supports controlled variation of the benchmark, judge, prompt, and inference backend (Lushtaku et al., 2026). LLMBar distinguishes instruction adherence from superficial preference (Zeng et al., 2024), while CriticBench evaluates generation, critique, and correction as a pipeline (Lin et al., 2024b). Position-bias tests quantify order efects through repeated and swapped comparisons (Shi et al., 2024).

When an LLM judge supplies the training reward, reusing the same judge family for final evaluation creates a circular test. A disjoint judge, a held-out human subset, or a verifiable proxy provides a more credible measure of improvement.

## 5.2.4. Intrinsic certainty and consensus signals

Rewards derived from uncertainty, self-consistency, or likelihood relative to a reference require diagnostics that difer from preference accuracy. Confidence can be calibrated against correctness on verifiable tasks. Consensus rewards can be tested across sample sizes, temperatures, and distribution shifts. Reference-based signals require sensitivity tests for changes in reference wording. In every case, an external checker can reveal whether optimization improves reasoning or merely amplifies patterns already favored by the model.

The feedback audit must match the training mechanism. Reward-model policies require evaluation of the frozen $\mathcal { R } _ { \phi }$ through RewardBench-style tests or downstream utility measures (Lambert et al., 2024; Frick et al., 2025). Judge-trained policies require direct evaluation of $\pi _ { \nu }$ and a secondary score from an independent evaluator (Tan et al., 2025; Jiang et al., 2025). Process-supervised policies require both step-error detection and final-outcome measures (Zheng et al., 2025a; Song et al., 2025b). Policies trained with intrinsic rewards require calibration and external-checker audits. Matching the audit to the reward source separates policy improvement from the reliability of the signal that produced it.

## 5.3. Experience Quality

Experience evaluation asks whether the tasks and environments supplied to learning remain valid, diverse, and informative. This issue arises when the task distribution Π or environment dynamics (S, P) are generated or adapted during training. A rising internal reward can coexist with a narrowing curriculum, an inaccurate environment, or a verifier that co-adapts to the policy. Evaluation must consequently examine the experience stream as well as the trained policy.

## 5.3.1. Agentic and interactive environments

Web and computer-use benchmarks provide externally specified success criteria for long-horizon agents. WebArena (Zhou et al., 2024), VisualWebArena (Koh et al., 2024), WorkArena (Drouin et al., 2024), WebVoyager (He et al., 2024b), AssistantBench (Yoran et al., 2024), BrowseComp (Wei et al., 2025a), and BrowserGym (Chezelles et al., 2025) cover navigation, enterprise knowledge work, multimodal browsing, and information seeking. OSWorld (Xie et al., 2024b), Windows Agent Arena (Bonatti et al., 2025),

AndroidWorld (Rawles et al., 2025), and AppWorld (Trivedi et al., 2024) extend evaluation to desktop, mobile, and multi-application workflows.

Tool-use suites include ToolLLM and ToolBench (Qin et al., 2024), StableToolBench (Guo et al., 2024), and the Berkeley Function Calling Leaderboard (Patil et al., 2025). AgentBench (Liu et al., 2024d), GAIA (Mialon et al., 2024), �-bench (Yao et al., 2024), TheAgentCompany (Xu et al., 2025a), ScienceAgentBench (Chen et al., 2024e), and BALROG (Paglieri et al., 2024) broaden coverage across general, scientific, workplace, and game settings. AgentDojo adds security evaluation through promptinjection attacks and defenses in dynamic tool environments (Debenedetti et al., 2024).

The same suites can evaluate both policy capability and environment quality. A unified framework can vary agent scafolds and environmental volatility to separate policy performance from execution conditions (Zhu et al., 2026a). Trainee-Bench tests scheduling, information gathering, and reuse of experience across workplace episodes (Fu et al., 2026). AgentWorldBench evaluates whether a language world model reproduces state transitions observed in established agent benchmarks (Zuo et al., 2026). For synthesized environments, relevant measures include validation pass rate, transition reproducibility, success-criterion correctness, solve rate under a frozen policy, and transfer to a public agent suite. These measures apply directly to the environment-generation methods discussed in Sec. 4.2.2.

## 5.3.2. Task quality under self-generated curricula

When the task distribution is generated during learning, solver accuracy must be accompanied by evidence about the proposed curriculum. Let $\bar { p } _ { \theta _ { S } } ( \pmb { q } )$ denote the empirical solve rate of query �. The learnability score $\bar { p } ( 1 - \bar { p } )$ in Eq. 18 is largest at intermediate solve rates and vanishes when every sampled solution succeeds or fails. It identifies tasks near the current solver’s learning frontier, although it does not measure novelty or semantic diversity.

Complementary statistics include the solve-rate distribution, embedding or �-gram diversity, novelty relative to the seed set, and rejection rates for invalid, toxic, or near-duplicate proposals. Corpus-grounded proposers enable document-level attribution as an external novelty check (Liu et al., 2025b; Kwan et al., 2026). Repeated concentration on a small source subset signals curriculum recycling. Trace-bootstrapping and instruction-evolution pipelines also need retention rates and filtering criteria because these choices shape the final task distribution (Zelikman et al., 2022; Wang et al., 2022; Xu et al., 2023; Fang et al., 2025b).

Capability leaderboards often conceal failures in the experience stream. Trivial or impossible tasks produce weak learning signals under group-relative optimization. A generator may repeatedly sample skills the policy has already mastered (Fang et al., 2025b; Kuba et al., 2025). In interactive settings, hallucinated rewards, flaky tests, and underspecified tools can corrupt every trajectory collected in a generated environment (Yang et al., 2025b; Shi et al., 2026). Curriculum and environment metrics are needed to distinguish genuine expansion of experience from these forms of collapse.

## 5.3.3. Closed-loop evaluation

The transition from experience beyond human design to autonomous co-evolution requires the strongest evaluation protocol. When policies, rewards, tasks, or environments adapt together, internal progress can diverge from externally measured capability. Relevant risks include distributional narrowing, nonstationary opponents or arenas (Liu et al., 2025a; Guertler et al., 2025; Dong et al., 2026), and asymmetry between generation and verification (Chen et al., 2026b). A credible closed-loop evaluation needs frozen external tests, trajectories of curriculum diversity, agreement between training and held-out verifiers, and ablations that freeze individual components of the loop.

PAST-Bench compares matched task sequences with and without retained experience to determine whether later gains follow the intended save, retrieve, and update pathway (Xue et al., 2026). AI4AI-Bench tests whether agents can improve learning algorithms inside frozen research repositories (Chi et al., 2026). The modified procedures are rerun from scratch and scored by a fixed evaluator. This protocol tests a concrete component of recursive improvement without treating success on that component as evidence of a fully autonomous loop.

Public agent suites and synthetic environments play complementary roles. Success in generated worlds can reflect generator-specific regularities if it does not transfer to environments such as WebArena or OSWorld (Zhou et al., 2024; Xie et al., 2024b). Public-suite gains are also incomplete evidence when the validity of the generated training environments is unknown. Credible evaluation combines learning curves in synthetic environments with transfer to frozen public environments under the same action interface.

## 5.4. Datasets and Benchmark Suites

The role of a dataset depends on how it enters the learning and evaluation pipeline. A verifiable collection can serve as training experience, a held-out capability test, or a source for constructing an evaluator. These roles must be stated explicitly because reuse across them can invalidate the apparent independence of evaluation.

Verifiable training and evaluation sets. GSM8K, MATH, HumanEval, MBPP, and competition-code corpora are attractive for training because their checkers are inexpensive and objective. Their broad use also creates saturation and contamination risks (Singh et al., 2024; Xu et al., 2024). LiveBench (White et al., 2025), LiveCodeBench (Jain et al., 2025), LiveMathBench (Liu et al., 2024b), DyVal (Zhu et al., 2024), AntiLeakBench (Wu et al., 2024), VarBench (Qian et al., 2024), and MMLU-CF (Zhao et al., 2024) reduce particular leakage risks through fresh data, controlled perturbations, or closed-test construction. They do not eliminate protocol bias, so static and refreshed suites remain useful as complementary views.

Preference and evaluator-development sets. Outcome and process reward models are developed from preference comparisons, step labels, and rubric annotations. RewardBench-style data (Lambert et al., 2024; Malik et al., 2025), HelpSteer2 (Wang et al., 2024d), ProcessBench (Zheng et al., 2025a), PRMBench (Song et al., 2025b), and Socratic-PRMBench (Li et al., 2025g) instantiate diferent standards of judgment. An evaluator trained on one of these distributions requires a disjoint test. Otherwise, the benchmark measures familiarity with its annotation scheme as much as general evaluator fidelity.

Interaction logs and environments. WebArena-style environments (Zhou et al., 2024; Xie et al., 2024b; Chezelles et al., 2025) and tool-use corpora (Qin et al., 2024; Patil et al., 2025) supply trajectories for long-horizon evaluation. Experience-generating methods additionally need a frozen split of synthesized environments, together with generator prompts and validation scripts (Mei et al., 2026; Zhu et al., 2026a). These artifacts make environment-fidelity claims auditable.

Table 6 maps these dataset roles to the ladder. The essential distinction is not whether a dataset was originally produced by a person. It is whether the same data, evaluator, or environment participates in both learning and final assessment. As more components enter the learning loop, frozen external suites and experience-quality measures become increasingly important.

Sampling, decoding, and comparability. Results on the same dataset can vary with temperature, sample count �, token budget, tool access, and requirements on the reasoning trace. Reports need a complete decoding profile that includes temperature, top-�, maximum tokens, stopping rules, and random seeds. When sampling is part of the method, greedy and sampled results answer diferent questions and require separate reporting. Agent evaluations also require the action budget, observation format, available tools, reset procedure, and environment version (Zhou et al., 2024; Xie et al., 2024b; Chezelles et al., 2025). Without these controls, protocol diferences can dominate the efect attributed to a learning signal.

Cross-domain transfer. Transfer provides a useful stress test for methods trained on a narrow verifiable domain. A complete report includes performance in the training domain, performance in the target domain, and the gap between them. Large in-domain gains with little transfer are compatible with several explanations, including overfitting, reward exploitation, and curriculum narrowing. Improvements across science QA, tool use, or open-ended interaction provide stronger evidence of generalization, although additional analysis is still needed to identify the cause (Wei et al., 2024; Wang et al., 2023b; Mialon et al., 2024).

## 5.5. Metrics, Protocols, and Reproducibility

Outcome and stability metrics. Exact match, numerical equivalence, and unit-test Pass@� remain the primary metrics for verifiable domains. Let � be the number of samples for each query and � the number of correct samples. The unbiased Pass@� estimator is

$$
\mathrm { P a s s } @ \mathit { k } = \mathbb { E } _ { q } \left[ 1 - \frac { { \binom { G - c } { \mathit { k } } } } { { \binom { G } { \mathit { k } } } } \right] .\tag{21}
$$

Mean sample accuracy is $c / G$ , sometimes denoted Avg@�. G-Pass@� measures consistent correctness across samples (Liu et al., 2024b),

$$
\mathrm { G } \mathrm { - } \mathrm { P a s s } @ k = \mathbb { E } _ { q } \left[ \binom { c } { k } / \binom { G } { k } \right] ,\tag{22}
$$

while CoT-Pass@� additionally requires a valid reasoning trace (Wen et al., 2025). The combination distinguishes occasional success from reliable reasoning across repeated samples.

Preference and judge metrics. Pairwise accuracy measures agreement between an outcome reward model and preference labels. Given triples $( \pmb q , \pmb y _ { c } , \pmb y _ { r } )$

$$
\mathrm { A c c u r a c y } = \frac { 1 } { | \mathcal { D } | } \sum _ { ( q , y _ { c } , y _ { r } ) \in \mathcal { D } } \mathbb { I } \big [ \mathcal { R } _ { \phi } ( q , y _ { c } ) > \mathcal { R } _ { \phi } ( q , y _ { r } ) \big ] ,\tag{23}
$$

RewardBench 2 extends this protocol to best-of-four selection, while PPE assesses metrics through their relationship with downstream RLHF outcomes (Malik et al., 2025; Frick et al., 2025). Open-ended evaluation commonly reports win rates, rubric scores, and structured G-Eval scores (Liu et al., 2023b). Process fidelity is measured with step-level F1 or PRM-Score (Li et al., 2025g; Song et al., 2025b). Every judge-based result also needs a diagnostic for position, length, and style bias (Shi et al., 2024; Liu et al., 2024e; Dubois et al., 2024).

Experience-quality metrics. Self-generated task streams require solve-rate distributions, mean learnability $\mathbb { E } [ \bar { p } ( 1 - \bar { p } ) ]$ , diversity, novelty, and filtering statistics. Synthesized environments require executability, validation pass rate, transition fidelity, and transfer to public agent benchmarks. Closed loops require trajectories of both internal reward and external held-out capability. Persistent divergence between the two signals warns of co-adaptive reward exploitation or collapse, although it does not identify a single cause on its own.

Contamination, bias, and tooling. ConTAM (Singh et al., 2024), ConStat (Dekoninck et al., 2024), and related analyses (Xu et al., 2024; Fu et al., 2024b) illustrate the vulnerability of static suites to leakage. Refreshed and procedurally varied benchmarks reduce this risk (White et al., 2025; Zhu et al., 2024; Wu et al., 2024; Qian et al., 2024; Zhao et al., 2024). LLM-based evaluators also require explicit audits for position and style bias (Shi et al., 2024; Liu et al., 2024e).

Community tools including lm-evaluation-harness,<sup>1</sup> LightEval,<sup>2</sup> ZeroEval,<sup>3</sup> Math-evaluation-harness,<sup>4</sup> and BigCode-evaluation-harness<sup>5</sup> standardize prompting, decoding, and aggregation. Reproducible reports must identify decoding seeds, judge prompts, environment settings, and exact harness revisions because these choices can materially change the comparison.

Recommended minimum scorecard. Our paper recommends frozen capability tests in the training domain and at least one transfer setting, together with an audit matched to the feedback source. Methods that generate experience also need direct measurements of task or environment quality. Claims based on heavily reused benchmarks require a contamination analysis, and all comparisons need compute-matched baselines under a shared protocol. Methods approaching L4 additionally require a longitudinal comparison between internal reward and frozen external evaluation. Table 6 presents the same requirements by ladder level.

## 5.6. Limitations and Open Evaluation Gaps

Current evaluation practice remains much stronger for policy capability than for feedback fidelity or experience quality. Public leaderboards rarely test whether a training judge agrees with an independent evaluator. Metrics for curriculum diversity, environment fidelity, and closed-loop stability are also fragmented across individual methods. These weaknesses make it possible for internal reward to rise while the evidence for general capability remains unchanged (Qi et al., 2026b; Chen et al., 2026b).

Several practical gaps compound this problem. Contamination controls and dynamic evaluation are unevenly adopted outside a small set of benchmarks (White et al., 2025; Zhu et al., 2024; Singh et al., 2024). Intrinsic-reward methods and verifier-based RLVR are seldom compared under matched compute, data access, and decoding budgets. Agent benchmarks ofer realistic interaction but remain expensive and partly non-reproducible because live websites and graphical interfaces change over time (Zhou et al., 2024; Chezelles et al., 2025; Xie et al., 2024b). Multimodal and long-context evaluation is also only weakly connected to work on learning with reduced supervision, despite the growing use of multimodal, tool-using agents (Chen et al., 2024a; Yen et al., 2025; Hsieh et al., 2024).

The three-part evaluation framework provides a practical response to these gaps. Capability is measured on frozen verifiable, expert, open-ended, or agentic suites. Feedback is tested with rewardmodel, process-model, judge, or intrinsic-signal audits matched to the training method. Experience is assessed through curriculum quality, environment validity, and closed-loop stability. Refreshed suites can reduce contamination in capability tests (White et al., 2025; Zhu et al., 2024; Jain et al., 2025), while expert checks and frozen judge panels provide sparse external anchors for feedback and experience.

No single leaderboard can establish progress toward self-sustaining learning. Claims must identify which components were evaluated externally and which remained inside the training loop. Missing evidence does not invalidate a method, but it limits the conclusion that can be drawn from its reported gains.

## 6. Challenges and Future Directions

The methods discussed in the preceding sections expand the sources of reward and experience available to large reasoning models. Reusable evaluators reduce the need for repeated human judgment, while self-generated tasks and environments allow training to continue beyond a fixed corpus. Yet the same mechanisms weaken some of the external constraints that make conventional training easy to audit. Proxy exploitation, curriculum collapse, generator-verifier collusion, and overlap between training and evaluation can all remain hidden behind rising internal reward. Accordingly, the central challenge is to sustain learning with less direct intervention while preserving evidence that remains independent of the learning loop.

## 6.1. Stabilizing Rewards beyond Per-Instance Human Evaluation

Section 3 reveals a tradeof between the scalability of a reward and the independence of its evidence. Entropy, self-certainty, consensus, and self-judgment can score large numbers of rollouts at low marginal cost. However, prolonged optimization can sharpen the policy distribution, reinforce majority errors, and decouple confidence from correctness. Execution, games, and formal verification provide stronger external grounding, but incomplete tests and exploitable rules still leave room for unintended behavior (Zhang et al., 2025k; Shafayat et al., 2025; He et al., 2026; Huang et al., 2025c; Zhao et al., 2025c; Wang et al., 2026d). Progress along the reward axis consequently depends on preserving independent evidence without recreating dense per-instance supervision.

Independent anchors for intrinsic and consensus rewards. Majority voting and confidence maximization avoid external labels at reward time, but neither signal can detect an error shared by the policy’s own samples (Shafayat et al., 2025; Zhang et al., 2025k). Sparse executable checks, frozen evaluators from an independent model family, corpus-grounded consistency tests, and periodic expert audits can provide complementary evidence. These anchors need to remain outside the optimization loop so that they detect divergence between internal reward and held-out capability. Their purpose is not to restore dense supervision. Instead, they establish a reference against which model-derived rewards can be calibrated and monitored.

Verifier robustness under optimization pressure. Learned and generative verifiers cover outputs that brittle string matching cannot assess (Chen et al., 2025a; Xu et al., 2025c; Zhang et al., 2025d). This flexibility creates a larger attack surface because false positives can become highly rewarding under policy optimization (Zhao et al., 2025c; Huang et al., 2025c; Helf et al., 2026; Wang et al., 2026c). Robustness therefore has to be tested after exposure to an optimizing policy, not only on a static benchmark. Adversarial evaluation, diferential testing against stricter checkers, and asymmetric objectives can expose or discourage collusion between generators and verifiers (Zha et al., 2025; Wang et al., 2025k). Ensembles provide an additional defense by reducing dependence on a single evaluator (Verga et al., 2024; Coste et al., 2023). Process-level rewards ofer evidence beyond final-answer matching, although their intermediate judgments require the same adversarial scrutiny (Cui et al., 2025a; She et al., 2025b). An open problem is how to certify that evaluator fidelity survives policy optimization when success on RewardBench-style probes does not guarantee robustness to newly optimized outputs (Lambert et al., 2024; Malik et al., 2025).

Semi-verifiable and open-ended tasks. Evidence for scalable reward remains concentrated in mathematics and code because these domains support inexpensive outcome checks. Scientific reasoning, long-form factuality, multimodal interaction, and preference-sensitive generation admit multiple valid responses and more context-dependent criteria. Extending RL to these settings requires structured evidence from rubrics, claim-level verification, retrieval, and calibrated judges (Viswanathan et al., 2025; Gunjal et al., 2025; Chen et al., 2025f; Fein et al., 2025). Such evaluators must recognize substantive variation without collapsing quality into style or verbosity. Hybrid designs provide a practical intermediate approach by sending verifiable subproblems to deterministic checkers and open-ended components to learned evaluators (Seed et al., 2025). The unresolved question is which criteria can safely be delegated and which still require an independent external standard.

## 6.2. Scaling Experience without Curriculum Collapse

As task and environment generation enter the learning loop, the main dificulty shifts from collecting more trajectories to preserving their educational value. A proposer can generate tasks that are uniformly easy, efectively impossible, or confined to patterns the solver already knows. A synthesized environment can also be internally consistent while failing to represent the external world. Dificulty miscalibration, mode collapse, non-stationarity, and co-adaptive reward exploitation already appear in proposer-solver systems and generated environments (Huang et al., 2025a; Zhao et al., 2025a; Fang et al., 2025b; Qi et al., 2026b).

Learnability and diversity over time. Frontier-targeting proposers and regret-based environment design seek tasks near the solver’s current learning boundary (Dennis et al., 2020; Huang et al., 2025a; Zhao et al., 2025a). Early solve-rate estimates �¯ are noisy, however, and a well-calibrated dificulty distribution can still lack semantic diversity. A useful curriculum therefore requires longitudinal measurements of both learnability and coverage. Corpus-grounded and interaction-grounded proposers can reduce template recycling by drawing on external content (Liu et al., 2025b; Kwan et al., 2025; Lu et al., 2025b). Archives of unsolved and out-of-distribution tasks provide a complementary mechanism because they preserve dificult experience for later policies instead of discarding it when the current solver fails.

Environment fidelity and executable grounding. Generated simulators, unit tests, and tool stacks inherit errors from the models that construct them (Yang et al., 2025b; Sun et al., 2025d; Zhou et al., 2025c). Environment generation must consequently be paired with validation. Staged executability checks can reject malformed components before training, while diferential testing against public harnesses such as SWE-bench, OSWorld, and WebArena can reveal discrepancies in task outcomes (Jimenez et al., 2024; Xie et al., 2024b; Zhou et al., 2024). Transfer from synthetic training environments to held-out real interactions provides a stronger test of fidelity than success inside the generated environment alone. Without these controls, a learning loop may optimize against artificial dynamics that do not support capability outside its own simulator.

Stable closed loops. CURE, Self-Challenging agents, SPIRAL, and the Darwin Gödel Machine couple diferent combinations of generators, verifiers, opponents, and agent modification (Wang et al., 2025k; Zhou et al., 2025c; Liu et al., 2025a; Zhang et al., 2025b). None yet closes the entire learning loop because fixed runtimes, game rules, task sources, or evaluation objectives continue to supply external structure. More complete co-evolution will require optimization and diagnostics that account for changing opponents and evaluators. Role-conditioned baselines can reduce variance in asymmetric self-play (Liu et al., 2025a). Frozen capability tests can reveal divergence from internal reward, while explicit verifier training can address the assumption that evaluation improves automatically with generation (Chen et al., 2026b). Stopping criteria are also needed when internal reward rises but external transfer declines. Existing closed-loop experiments show improvement over the initial policy followed by a plateau below oracle-supervised training (Qi et al., 2026b). Extending this limit without allowing the loop to validate its own errors remains a defining challenge for autonomous co-evolution.

## 6.3. Eficiency of Experience Sampling and Token Utility

Reducing annotation cost can increase inference and interaction cost. Group-based RL samples several rollouts per query, consensus rewards require repeated generation, and self-generated experience adds proposer, verifier, or environment-construction passes. Long reasoning traces further consume computation even when their tokens do not afect the final reward. Under Eq. 6, tasks with nearly constant group rewards can spend substantial compute while producing little learning signal.

Experience generation is consequently a resource-allocation problem in which every sampled trajectory must justify its computational cost. Samples can be concentrated near the learnability frontier, and low-information rollouts can be rejected before full generation. Shorter process feedback may preserve credit assignment without scoring every token, while replay bufers can reuse expensive interactions across proposer and solver updates. At inference time, uncertainty and estimated task dificulty can guide the allocation of test-time computation. These mechanisms connect eficiency to capability because the number and diversity of useful experiences available to a policy are limited by the compute budget.

Credible reports pair capability gains with the amount of generation, verification, and environment interaction required to obtain them.

## 6.4. Evaluation under Receding Supervision

Capability, feedback fidelity, and experience quality provide complementary evidence, as developed in Section 5. Current leaderboards emphasize capability on static test sets, while reward robustness, curriculum diversity, environment fidelity, and closed-loop stability remain less standardized (White et al., 2025; Qi et al., 2026b; Chen et al., 2026b). The reuse of related judges for training and evaluation creates an additional risk because the same bias can improve both the policy’s reward and its reported score.

Table 6 summarizes the minimum evidence required as more components enter the learning loop. Reward-generating methods need audits matched to the source of feedback. Methods that generate tasks or environments additionally need measurements of diversity, solvability, validity, and transfer. Live and procedurally refreshed benchmarks reduce dependence on fixed evaluation distributions (White et al., 2025; Zhu et al., 2024; Jain et al., 2025), although freshness alone cannot rule out contamination or protocol bias. Shared evaluation harnesses are equally important because comparisons among intrinsic-reward, verifier-based, and self-play methods require matched compute, data access, and decoding budgets. Automated evaluation remains credible only when its external anchors are frozen and independent of training.

## 6.5. Trust, Safety, and Value Preservation under Autonomy

Human values do not disappear when direct supervision becomes sparse. Their influence moves into initial data, constitutions, tools, environments, objectives, and evaluation criteria (Bai et al., 2022; Sun et al., 2023). As these components begin to co-adapt, the feedback channels that support learning can also amplify misalignment. Self-reward may reinforce sycophancy or unsafe high-confidence responses. Generated curricula may omit rare but important safety cases, and exploitable judges may favor persuasive behavior over correct or safe behavior (Zhou, 2026; Yuan et al., 2024).

Accordingly, safety evaluation must examine the same three objects as capability evaluation. Policy tests measure harmful behavior and over-refusal, feedback audits test evaluator reliability under adversarial optimization, and experience audits assess whether generated curricula cover risk-relevant states instead of selecting only convenient cases. Sparse human review and constitutional audits can serve as external vetoes, but only if they remain independent of the policy, reward model, and curriculum generator. The broader aim is not to eliminate human involvement. It is to reserve human judgment for setting objectives, detecting drift, and intervening when self-generated evidence becomes unreliable.

## 6.6. Toward Broader Training Paradigms and Superintelligence

Most methods considered in this paper apply RL after pretraining. A longer-term question is whether sparse external anchors, generated curricula, and closed experience loops can also support pretraining or continual mid-training, where direct annotation cannot cover the volume of data and interaction (Silver & Sutton, 2025). Extending these mechanisms will require reward signals that remain stable under large-scale optimization, curricula that expand semantic coverage, and evaluation procedures that detect drift before it compounds. Iterative self-distillation, emergent curricula, and agent-environment co-evolution provide pieces of this process, but none yet demonstrates safe, open-ended capability growth.

The five-level ladder describes a possible transfer of responsibility for sustaining learning. Humans may continue to specify goals, constraints, and external anchors even when they cannot demonstrate optimal behavior or inspect every reasoning step. Progress toward more autonomous learning is meaningful only if execution, interaction, frozen audits, and value constraints remain suficiently independent to distinguish genuine improvement from an increasingly persuasive proxy. Whether this independence can be preserved at scale will determine if learning beyond direct human supervision supports reliable progress toward superintelligence or amplifies ungrounded self-reinforcement.

## 7. Conclusion and Limitations

Our paper examines how large reasoning models can continue to learn when direct human supervision no longer scales with the volume or dificulty of their experience. We organize this problem along two connected axes. The reward axis concerns the evidence used to evaluate a rollout, while the experience axis concerns the tasks and environments from which learning proceeds. The five-level ladder connects these axes by tracking which components still require continued human provision.

Along the reward axis, reusable evaluators and model-derived, reference-anchored, or environmentgrounded signals reduce the need for direct judgment. Their scalability, however, is inseparable from the independence and validity of the evidence they use. Along the experience axis, task generation and environment construction allow the training distribution to adapt with the policy, but introduce dificulty miscalibration, curriculum narrowing, environment errors, and generator-verifier collusion. Continued improvement consequently requires rewards that remain faithful and experience streams that remain informative as both become more adaptive.

Together, these findings motivate a three-part evaluation framework that separates policy outcomes from the validity of the learning loop. Policy capability measures performance on frozen external tasks. Feedback fidelity tests whether the reward preserves its intended meaning under optimization. Experience quality assesses the diversity, validity, and transferability of generated tasks and environments. Capability gains alone are insuficient when feedback or experience is produced within the same learning loop. Scaling beyond human supervision also does not remove human intent. It relocates that intent into reusable evaluators, objectives, task seeds, environments, and sparse external audits. Such learning loops may contribute to progress toward superintelligence, but supervisory autonomy neither constitutes general intelligence nor guarantees reliable capability growth.

Limitations. The ladder is an analytical framework, not a set of mutually exclusive method classes. Methods often occupy transitional positions because reward acquisition, task generation, environment construction, and policy optimization do not become autonomous simultaneously. The framework also tracks the operational source of supervision during learning, not the historical amount of human knowledge embedded in pretrained models, references, corpora, tools, or environments.

Direct comparison across methods remains constrained by diferences in models, compute, data, decoding, verifiers, and evaluation protocols. Our coverage of safety, multilingual reasoning, multimodal interaction, and long-context learning is necessarily selective, and the discussion of superintelligence remains conceptual. The framework does not imply that current systems are superintelligent or that movement along the ladder will inevitably produce superintelligent behavior.

Despite these limitations, the dual-axis framework clarifies the evidence required for credible progress beyond direct human supervision. Reliable capability growth depends on rewards that withstand optimization pressure, experience that continues to support learning, and external anchors that preserve contact with intended objectives.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report, 2023.

Adler, B., Agarwal, N., Aithal, A., Anh, D. H., Bhattacharya, P., Brundyn, A., Casper, J., Catanzaro, B., Clay, S., Cohen, J., et al. Nemotron-4 340b technical report. arXiv preprint arXiv:2406.11704, 2024.

Agarwal, S., Zhang, Z., Yuan, L., Han, J., and Peng, H. The unreasonable efectiveness of entropy minimization in llm reasoning. Advances in Neural Information Processing Systems, 38:107150–107180, 2025.

Ahn, J., Verma, R., Lou, R., Liu, D., Zhang, R., and Yin, W. Large language models for mathematical reasoning: Progresses and challenges. In The 18th Conference of the European Chapter of the Association for Computational Linguistics, pp. 225, 2024.

Alam, M. T. and Rastogi, N. Limits of generalization in rlvr: Two case studies in mathematical reasoning. arXiv preprint arXiv:2510.27044, 2025.

Atmakuru, A., Nainani, J., Bheemreddy, R. S. R., Lakkaraju, A., Yao, Z., Zamani, H., and Chang, H.-S. Cs4: Measuring the creativity of large language models automatically by controlling the number of story-writing constraints. arXiv preprint arXiv:2410.04197, 2024.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences. In International conference on artificial intelligence and statistics, pp. 4447–4455. PMLR, 2024.

Badshah, S. and Sajjad, H. Reference-guided verdict: Llms-as-judges in automatic evaluation of free-form text. arXiv preprint arXiv:2408.09235, 2024.

Bahng, H., Chan, C., Durand, F., and Isola, P. Cycle consistency as reward: Learning image-text alignment without human preferences. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22934–22946, 2025.

Bai, G., Liu, J., Bu, X., He, Y., Liu, J., Zhou, Z., Lin, Z., Su, W., Ge, T., Zheng, B., and Ouyang, W. MT-Bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7421–7442, 2024a.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137, 2024b.

Bai, Y., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X., Cao, S., Xu, J., Hou, L., Dong, Y., et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3639–3664, 2025.

Bonatti, R., Zhao, D., Bonacci, F., Dupont, D., Abdali, S., Li, Y., Lu, Y., Wagle, J., Koishida, K., Bucker, A., et al. Windows agent arena: Evaluating multi-modal os agents at scale. In International Conference on Machine Learning, pp. 4874–4910. PMLR, 2025.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S., Behbahani, F., Chan, S., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., de Freitas, N., Singh, S., and Rocktäschel, T. Genie: Generative interactive environments. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

Cai, Y., Hao, Y., Zhou, J., Yan, H., Lei, Z., Zhen, R., Han, Z., Yang, Y., Li, J., Pan, Q., et al. Building self-evolving agents via experience-driven lifelong learning: A framework and benchmark. arXiv preprint arXiv:2508.19005, 2025.

Cao, S., Li, D., Zhao, F., Yuan, S., Hegde, S. R., Chen, C., Ruan, C., Griggs, T., Liu, S., Tang, E., et al. Skyrl-agent: Eficient rl training for multi-turn llm agent. arXiv preprint arXiv:2511.16108, 2025.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Chae, H., Kim, S., Cho, J., Kim, S., Moon, S., Hwangbo, G., Lim, D., Kim, M., Hwang, Y., Gwak, M., et al. Web-shepherd: Advancing prms for reinforcing web agents. Advances in Neural Information Processing Systems, 38:63314–63356, 2025.

Chang, Y., Kim, Y., Krumdick, M., Zadeh, A., Li, C., Tanner, C., and Iyyer, M. Bleuberi: Bleu is a surprisingly efective reward for instruction following. Advances in Neural Information Processing Systems, 38:15466–15501, 2025.

Chao, P., Debenedetti, E., Robey, A., et al. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. arXiv preprint arXiv:2404.01318, 2024.

Chen, D., Chen, R., Zhang, S., Wang, Y., Liu, Y., Zhou, H., Zhang, Q., Wan, Y., Zhou, P., and Sun, L. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning, 2024a.

Chen, D., Yu, Q., Wang, P., Hu, M., Zhang, W., Wang, Z., Tang, B., Xiong, F., Li, X., Wang, C., et al. xverify: Eficient answer verifier for reasoning model evaluations. arXiv preprint arXiv:2504.10481, 2025a.

Chen, G. H., Chen, S., Liu, Z., Jiang, F., and Wang, B. Humans or LLMs as the judge? a study on judgement bias. arXiv preprint arXiv:2402.10669, 2024b.

Chen, H., Feng, Y., Liu, Z., Yao, W., Prabhakar, A., Heinecke, S., Ho, R., Mui, P., Savarese, S., Xiong, C., et al. Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding. arXiv preprint arXiv:2411.04282, 2024c.

Chen, H., Fang, Z., Singla, Y., and Dredze, M. Benchmarking large language models on answering and explaining challenging medical questions. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3563–3599, 2025b.

Chen, J., Zhou, P., Hua, Y., Xin, L., Chen, K., Li, Z., Zhu, B., and Liang, J. Fintextqa: A dataset for long-form financial question answering. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6025–6047, 2024d.

Chen, J., Dong, Y., Li, H., Su, W., Zhou, Y., Zhang, M., Liu, Y., and Ai, Q. Benchmarking llm-as-a-judge for long-form output evaluation. arXiv preprint arXiv:2606.01629, 2026a.

Chen, L., Prabhudesai, M., Fragkiadaki, K., Liu, H., and Pathak, D. Self-questioning language models. arXiv preprint arXiv:2508.03682, 2025c.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Chen, M., Chen, G., Wang, W., and Yang, Y. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346, 2025d.

Chen, Q., Qin, L., Liu, J., Peng, D., Guan, J., Wang, P., Hu, M., Zhou, Y., Gao, T., and Che, W. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025e.

Chen, X., Kulikov, I., Berges, V.-P., Oğuz, B., Shao, R., Ghosh, G., Weston, J., and Yih, W.-t. Learning to reason for factuality. arXiv preprint arXiv:2508.05618, 2025f.

Chen, X., Li, G., Wang, Z., Jin, B., Qian, C., Wang, Y., Wang, H., Zhang, Y., Zhang, D., Zhang, T., et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025g.

Chen, X., Yue, C., Liu, B., Wang, X., Wang, Y., and Lu, C. Vrprm: Process reward modeling via visual reasoning. arXiv preprint arXiv:2508.03556, 2025h.

Chen, X., Zhang, J., Wang, S., Wu, Y., Le, H., et al. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025i.

Chen, Y., Wang, Y., Zhu, S., Yu, H., Feng, T., Zhang, M., Patwary, M., and You, J. Multi-agent evolve: Llm self-improve through co-evolution. arXiv preprint arXiv:2510.23595, 2025j.

Chen, Y., Wang, Y., Zhang, Y., Ye, Z., Cai, Z., Shi, Y., Gu, Q., Su, H., Cai, X., Wang, X., et al. Learning to self-verify makes language models better reasoners. arXiv preprint arXiv:2602.07594, 2026b.

Chen, Z., Chen, S., Ning, Y., et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. arXiv preprint arXiv:2410.05080, 2024e.

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335, 2024f.

Chen, Z., Wu, J., Zhou, J., Wen, B., Bi, G., Jiang, G., Cao, Y., Hu, M., Lai, Y., Xiong, Z., et al. Tombench: Benchmarking theory of mind in large language models. arXiv preprint arXiv:2402.15052, 2024g.

Cheng, D., Huang, S., Zhu, X., Dai, B., Zhao, X., Zhang, Z., and Wei, F. Reasoning with exploration: An entropy perspective. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 30377–30385, 2026.

Chezelles, T. L. S. d., Gasse, M., Lacoste, A., Caccia, M., Drouin, A., Boisvert, L., Thakkar, M., Marty, T., Assouel, R., Shayegan, S. O., et al. The browsergym ecosystem for web agent research. Transactions on Machine Learning Research, 2025.

Chi, Y., Li, W., Hong, D., Wang, X., Gao, M., Yang, K., He, B., Zheng, Y., Xiao, C., and Na, Q. Ai4ai-bench: Benchmarking llm agents in algorithmic design for recursive self-improvement. arXiv preprint arXiv:2608.20318, 2026.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743, 2023.

Cui, G., Yuan, L., Wang, Z., Wang, H., Zhang, Y., Chen, J., Li, W., He, B., Fan, Y., Yu, T., et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

Dai, J., Lu, J., Feng, Y., Huang, D., Zeng, G., Ruan, R., Cheng, M., Tan, H., and Guo, Z. Mhpp: Exploring the capabilities and limitations of language models beyond basic code generation. arXiv preprint arXiv:2405.11430, 2024.

Damani, M., Puri, I., Slocum, S., Shenfeld, I., Choshen, L., Kim, Y., and Andreas, J. Beyond binary rewards: Training lms to reason about their uncertainty. arXiv preprint arXiv:2507.16806, 2025.

Debenedetti, E., Zhang, J., Balunovic, M., et al. Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. arXiv preprint arXiv:2406.13352, 2024.

Dekoninck, J., Mueller, M. N., and Vechev, M. Constat: Performance-based contamination detection in large language models. arXiv preprint arXiv:2405.16281, 2024.

Dekoninck, J., Jovanović, N., Gehrunger, T., Rögnvaldsson, K., Petrov, I., Sun, C., and Vechev, M. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. arXiv preprint arXiv:2605.00674, 2026.

Deng, J., Chen, J., Chen, Z., Cheng, D., Bai, F., Zhang, B., Min, Y., Gao, Y., Zhao, W. X., and Wen, J.-R. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534, 2025.

Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S., Critch, A., and Levine, S. Emergent complexity and zero-shot transfer via unsupervised environment design. arXiv preprint arXiv:2012.02096, 2020.

Dharna, A., Lu, C., and Clune, J. Foundation model self-play: Open-ended strategy innovation via foundation models. arXiv preprint arXiv:2507.06466, 2025.

Ding, H., Huang, B., Fang, Y., Liao, W., Li, Z., Zhang, J., Wu, Z., Zhao, J., and Wang, Y. Evorubrics: Dynamic rubrics as rewards via adversarial co-evolution for llm reinforcement learning. arXiv preprint arXiv:2606.23038, 2026.

Dong, G. et al. Agent-world: Scaling real-world environment synthesis for evolving general agent intelligence. arXiv preprint arXiv:2604.18292, 2026.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. Rlhf workflow: From reward modeling to online rlhf a comprehensive practical alignment recipe of iterative preference learning. Transactions on Machine Learning Research, 2024.

Dong, K. and Ma, T. Stp: Self-play llm theorem provers with iterative conjecturing and proving. In International Conference on Machine Learning, pp. 14114–14136. PMLR, 2025.

Dou, S., Liu, S., Yang, Y., Zou, Y., Zhou, Y., Xing, S., Huang, C., Ge, Q., Song, D., Lv, H., et al. Pre-trained policy discriminators are general reward models. arXiv preprint arXiv:2507.05197, 2025.

Drouin, A., Gasse, M., Caccia, M., et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.

Du, X., Yao, Y., et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Egashira, K., Vero, M., Dekoninck, J., Dorner, F. E., Staab, R., and Vechev, M. Delay, plateau, or collapse: Evaluating the impact of systematic verification error on rlvr. arXiv preprint arXiv:2605.02909, 2026.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D’Amour, A., Dvijotham, K., Fisch, A., Heller, K., Pfohl, S., Ramachandran, D., et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244, 2023.

Ethayarajh, K., Xu, W., Muennighof, N., Jurafsky, D., and Kiela, D. KTO: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Faldor, M., Zhang, J., Cully, A., and Clune, J. Omni-epic: Open-endedness via models of human notions of interestingness with environments programmed in code. arXiv preprint arXiv:2405.15568, 2024.

Fan, L., Zhang, Y., Chen, M., and Liu, Z. Recode: Reinforcing code generation with reasoning-process rewards. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 43896–43914, 2026.

Fan, Y., Ni, J., Merane, J., Salimbeni, E., Tian, Y., Hermstrüwer, Y., Huang, Y., Akhtar, M., Geering, F., Dreyer, O., et al. Lexam: Benchmarking legal reasoning on 340 law exams. arXiv preprint arXiv:2505.12864, 2025.

Fang, J., Peng, Y., Zhang, X., Wang, Y., Yi, X., Zhang, G., Xu, Y., Wu, B., Liu, S., Li, Z., et al. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407, 2025a.

Fang, W., Liu, S., Zhou, Y., Zhang, K., Zheng, T., Chen, K., Song, M., and Tao, D. Serl: Self-play reinforcement learning for large language models with limited data. arXiv preprint arXiv:2505.20347, 2025b.

Fei, Z., Shen, X., Zhu, D., Zhou, F., Han, Z., Huang, A., Zhang, S., Chen, K., Yin, Z., Shen, Z., et al. Lawbench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7933–7962, 2024.

Fein, D., Russo, S., Xiang, V., Jolly, K., Rafailov, R., and Haber, N. Litbench: A benchmark and dataset for reliable evaluation of creative writing. arXiv preprint arXiv:2507.00769, 2025.

Feng, J., Huang, S., Qu, X., Zhang, G., Qin, Y., Zhong, B., Jiang, C., Chi, J., and Zhong, W. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025a.

Feng, Y., Song, M., Wang, J., Chen, Z., Bi, G., Huang, M., Jing, L., and Yu, J. Ss-gen: A social story generation framework with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 1300–1308, 2025b.

Feng, Z., Cao, S., Ren, J., Su, J., Chen, R., Zhang, Y., Wu, J., and Liu, Z. Mt-r1-zero: Advancing llm-based machine translation via r1-zero-like reinforcement learning. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, pp. 18685–18702, 2025c.

Frick, E., Li, T., Chen, C., Chiang, W.-L., Angelopoulos, A. N., Jiao, J., Zhu, B., Gonzalez, J. E., and Stoica, I. How to evaluate reward models for RLHF. In The Thirteenth International Conference on Learning Representations, 2025.

Fu, C., Dai, Y., Luo, Y., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. arXiv preprint arXiv:2405.21075, 2024a.

Fu, D., Mei, J., Wu, R., Yang, X., Xu, J., Wang, D., Cai, P., Liu, Y., Wen, L., and Shi, B. The agent’s first day: Benchmarking learning, exploration, and scheduling in the workplace scenarios. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 30094–30109, 2026.

Fu, Y., Uzuner, O., Yetisgen, M., and Xia, F. Does data contamination detection work (well) for LLMs? a survey and evaluation on detection assumptions. arXiv preprint arXiv:2410.18966, 2024b.

Gao, B., Song, F., Yang, Z., Cai, Z., Miao, Y., Dong, Q., Li, L., Ma, C., Chen, L., Xu, R., et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985, 2024.

Gao, H.-a., Geng, J., Hua, W., Hu, M., Juan, X., Liu, H., Liu, S., Qiu, J., et al. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 2025a.

Gao, J., Fu, W., Xie, M., Xu, S., He, C., Mei, Z., Zhu, B., and Wu, Y. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl. In First Workshop on Multi-Turn Interactions in Large Language Models, 2025b.

Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.

Gao, Z., Ma, J., Li, X., Li, P., Qu, N., Wu, Y., Liu, H., and Liu, J. Aero: Autonomous evolutionary reasoning optimization via endogenous dual-loop feedback. arXiv preprint arXiv:2602.03084, 2026.

Gehring, J., Zheng, K., Copet, J., Mella, V., Cohen, T., and Synnaeve, G. Rlef: Grounding code llms in execution feedback with reinforcement learning. In International Conference on Machine Learning, pp. 19034–19055. PMLR, 2025.

Glazer, E., Erdil, E., Besiroglu, T., Chicharro, D., Chen, E., Gunning, A., Falkman Olsson, C., Denain, J.-S., Ho, A., Santos, E. d. O., et al. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in AI. arXiv preprint arXiv:2411.04872, 2024.

Gu, A., Rozière, B., Leather, H., Solar-Lezama, A., Synnaeve, G., and Wang, S. I. Cruxeval: a benchmark for code reasoning, understanding and execution. In Proceedings of the 41st International Conference on Machine Learning, pp. 16568–16621, 2024a.

Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H., et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024b.

Guertler, L., Cheng, B., Yu, S., Liu, B., Choshen, L., and Tan, C. Textarena. arXiv preprint arXiv:2504.11442, 2025.

Guha, N., Nyarko, J., Ho, D., Ré, C., Chilton, A., Chohlas-Wood, A., Peters, A., Waldon, B., Rockmore, D., Zambrano, D., et al. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems, 36:44123–44279, 2023.

Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. Reinforced self-training (ReST) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

Gunjal, A., Wang, A., Lau, E., Nath, V., Liu, B., and Hendryx, S. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Guo, J., Chi, Z., Dong, L., Dong, Q., Wu, X., Huang, S., and Wei, F. Reward reasoning models. Advances in Neural Information Processing Systems, 38:150477–150510, 2025b.

Guo, J., Yang, L., Chen, P., Xiao, Q., Wang, Y., Juan, X., Qiu, J., Shen, K., and Wang, M. Genenv: Dificulty-aligned co-evolution between llm agents and environment simulators. arXiv preprint arXiv:2512.19682, 2025c.

Guo, Z., Cheng, S., Wang, H., Liang, S., Qin, Y., Li, P., Liu, Z., Sun, M., and Liu, Y. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. arXiv preprint arXiv:2403.07714, 2024.

Gurung, A. and Lapata, M. Learning to reason for long-form story generation. arXiv preprint arXiv:2503.22828, 2025.

He, B., Zuo, Y., Liu, Z., Zhao, S., Fu, Z., Yang, J., Qian, C., Zhang, K., Fan, Y., Cui, G., et al. How far can unsupervised rlvr scale llm training? In The Fourteenth International Conference on Learning Representations, 2026.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Liu, Y., Liu, Y., Zhong, W., et al. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024a.

He, H., Yao, W., Ma, K., et al. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024b.

He, M., Liu, Y., Tao, S., Luo, Y., Zeng, H., Su, C., Zhang, L., MaHongxia, Wei, D., Meng, W., et al. R1-t1: Fully incentivizing translation capability in llms via reasoning learning. arXiv preprint arXiv:2502.19735, 2025.

Helf, L., Delfosse, Q., Steinmann, D., Härle, R., Shindo, H., Schramowski, P., Stammer, W., Kersting, K., and Friedrich, F. LLMs gaming verifiers: RLVR can lead to reward hacking. arXiv preprint arXiv:2604.15149, 2026.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021b.

Hong, I., Yu, C., Qiu, L., Yan, W., Xu, Z., Jiang, H., Zhang, Q., Lu, Q., Liu, X., Zhang, C., et al. Think-rm: Enabling long-horizon reasoning in generative reward models. Advances in Neural Information Processing Systems, 38:174227–174251, 2025.

Hong, J., Lee, N., and Thorne, J. ORPO: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2024.

Hou, G., Zhang, W., Shen, Y., Tan, Z., Shen, S., and Lu, W. Egosocialarena: Benchmarking the social intelligence of large language models from a first-person perspective. arXiv preprint arXiv:2410.06195, 2024.

Hsieh, C.-P., Sun, S., Kryscinski, W., Cohen, N., Yu, T., Kaplan, J., Wu, F., et al. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Huan, M. Z., Li, Y., Zheng, T., Xu, X., Kim, S., Du, M., Poovendran, R., Neubig, G., and Yue, X. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. In Forty-third International Conference on Machine Learning, 2026.

Huang, C., Yu, W., Wang, X., Zhang, H., Li, Z., Li, R., Huang, J., Mi, H., and Yu, D. R-zero: Self-evolving reasoning llm from zero data. arXiv preprint arXiv:2508.05004, 2025a.

Huang, C., Lin, L., Shi, X., Hu, W., and Tang, R. Darl: Encouraging diverse answers for general reasoning without verifiers. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 30649–30665, 2026a.

Huang, C., Liu, H., Zheng, T., Dai, R., Huang, L., Li, J., Li, Z., Wei, Z., Meng, Y., and Huang, J. G-zero: Self-play for open-ended generation from zero data. arXiv preprint arXiv:2605.09959, 2026b.

Huang, H., He, Y., Liu, W., Yang, M., Liu, J., Chen, K., Xu, B., Zhu, C., Cao, H., and Zhao, T. Long-form rewardbench: Evaluating reward models for long-form generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 31149–31157, 2026c.

Huang, Y., Li, S., Liu, M., Liu, W., Huang, S., Fan, Z., Chan, H. P., and Fung, Y. R. Environment scaling for interactive agentic experience collection: A survey. arXiv preprint arXiv:2511.09586, 2025b.

Huang, Y., Zeng, W., Zeng, X., Zhu, Q., and He, J. From accuracy to robustness: A study of rule-and model-based verifiers in mathematical reasoning. arXiv preprint arXiv:2505.22203, 2025c.

Huang, Z., Zhuang, Y., Lu, G., Qin, Z., Xu, H., Zhao, T., Peng, R., Hu, J., Shen, Z., Hu, X., et al. Reinforcement learning with rubric anchors. arXiv preprint arXiv:2508.12790, 2025d.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., and Vidgen, B. Financebench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944, 2023.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025.

Jenane, A., Walha, N., Kuhn, L., and Buettner, F. From entropy to calibrated uncertainty: Training language models to reason about uncertainty. In Towards Trustworthy Predictions: Theory and Applications ofCalibrationfor Modern AI, 2026.

Ji, X., Liu, Y., Wang, Q., Zhang, J., Yue, Y., Shi, R., Sun, C., Zhang, F., Zhou, G., and Gai, K. Leanabell-prover-v2: Verifier-integrated reasoning for formal theorem proving via reinforcement learning. arXiv preprint arXiv:2507.08649, 2025.

Jiang, H., Chen, Y., Cao, Y., Lee, H.-y., and Tan, R. T. Codejudgebench: Benchmarking llm-as-a-judge for coding tasks. arXiv preprint arXiv:2507.10535, 2025.

Jiang, H., Dong, Z., Liu, T., Wang, W., Xu, R., Yu, T., Zhang, L., and Wang, H. Rubric-arrow: Alternating pointwise rubric reward modeling for llm post-training in non-verifiable domains. arXiv preprint arXiv:2605.29156, 2026a.

Jiang, Y., Wang, Y., Zhang, Q., Zeng, X., Li, L., Chen, J., Tao, C., Bai, H., and Shang, L. From verifiable dot to reward chain: Harnessing verifiable reference-based rewards for reinforcement learning of open-ended generation. In The Fourteenth International Conference on Learning Representations, 2026b.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. R. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024.

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S. O., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling, 2025.

Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11 (14):6421, 2021.

Kang, Z., Zhao, X., and Song, D. Scalable best-of-n selection for large language models via self-certainty. arXiv preprint arXiv:2502.18581, 2025.

Khalifa, M., Agarwal, R., Logeswaran, L., Kim, J., Peng, H., Lee, M., Lee, H., and Wang, L. Process reward models that think. arXiv e-prints, pp. arXiv–2504, 2025.

Khalifa, M., Khan, Z., Tafveez, O., Peng, H., and Wang, L. Countdown-code: A testbed for studying the emergence and generalization of reward hacking in RLVR. arXiv preprint arXiv:2603.07084, 2026.

Kim, S., Kang, D., Kwon, T., et al. Evaluating robustness of reward models for mathematical reasoning. arXiv preprint arXiv:2410.01729, 2024a.

Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., and Seo, M. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4334–4353, 2024b.

Kim, S., Suk, J., Cho, J. Y., Longpre, S., Kim, C., Yoon, D., Son, G., Cho, Y., Shafayat, S., Baek, J., et al. The biGGen bench: A principled benchmark for fine-grained evaluation of language models with language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025.

Koh, J. Y., Lo, R., Jang, L., Duvvur, V., Lim, M. C., Huang, P.-Y., Neubig, G., Zhou, S., Salakhutdinov, R., and Fried, D. Visualwebarena: Evaluating multimodal agents on realistic visually grounded web tasks. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 881–905, 2024.

Kong, L., Wang, X., Chen, Y., and Zhang, M. Round-trip reinforcement learning: Self-consistent training for better chemical llms. In Third Conference on Language Modeling, 2026.

Kuba, J. G., Gu, M., Ma, Q., Tian, Y., Mohan, V., and Chen, J. Language self-play for data-free training. arXiv preprint arXiv:2509.07414, 2025.

Kwan, W.-C., Gema, A. P., Leang, J. O. J., Vougiouklis, P., Pan, J. Z., Valentino, M., and Minervini, P. Opensir: Open-ended self-improving reasoner. arXiv preprint arXiv:2511.00602, 2025.

Kwan, W.-C., Gema, A. P., Leang, J. O. J., and Minervini, P. Scope: Self-play via co-evolving policies for open-ended tasks. arXiv preprint arXiv:2605.31433, 2026.

Kwiatkowski, A., Butt, N., Labiad, I., Kempe, J., and Ollivier, Y. Likelihood-based reward designs for general llm reasoning. arXiv preprint arXiv:2602.03979, 2026.

Lambert, N., Pyatkin, V., Morrison, J., Miranda, L., Lin, B. Y., Chandu, K., Dziri, N., Kumar, S., Zick, T., Choi, Y., et al. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

Laurent, J. M., Janizek, J. D., Ruzo, M., et al. LAB-Bench: Measuring capabilities of language models for biology research. arXiv preprint arXiv:2407.10362, 2024.

Lee, H., Phatale, S., Mansoor, H., Mesnard, T., Ferret, J., Lu, K., Bishop, C., Hall, E., Carbune, V., Rastogi, A., et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023.

Li, D., Zhou, J., Kazemi, A., Sun, Q., Ghaddar, A., Ma, L., Luo, Y., Li, D., HAO, J., and Zhang, Y. Omni-thinker: Scaling cross-domain generalization in llms via multi-task rl with hybrid rewards. In 2nd AIfor Math Workshop@ ICML, 2025a.

Li, H., Chen, Y., Ai, Q., Wu, Y., Zhang, R., and Liu, Y. Lexeval: A comprehensive chinese legal benchmark for evaluating large language models. Advances in Neural Information Processing Systems, 37:25061–25094, 2024a.

Li, H., Deng, P., Qian, W., Jiang, L., Huang, Z., Yang, M., and Peng, X. Active-swe: Benchmarking coding agents for proactive bug fixing without issue reports. arXiv preprint arXiv:2608.04682, 2026a.

Li, J., Sun, S., Yuan, W., Fan, R.-Z., Zhao, H., and Liu, P. Generative judge for evaluating alignment. arXiv preprint arXiv:2310.05470, 2023a.

Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S., Rasul, K., Yu, L., Jiang, A. Q., Shen, Z., et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024b.

Li, K., Zhang, Z., Yin, H., Zhang, L., Ou, L., Wu, J., Yin, W., Li, B., Tao, Z., Wang, X., et al. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025b.

Li, K., Pang, J.-C., and Yu, Y. Rlvr training of llms does not improve thinking ability for general qa: Evaluation method and a simple solution. In ICML 2026 Workshop on Foundations ofDeep Generative Models: Understanding Memorization, Generalization, and Reasoning, 2026b.

Li, L., Wei, Y., Xie, Z., et al. VL-RewardBench: A challenging benchmark for vision-language generative reward models. arXiv preprint arXiv:2411.17451, 2024c.

Li, M., Zhao, L., So, A. M.-C., Sun, R., and Li, X. A model can help itself: Reward-free self-training for llm reasoning. arXiv preprint arXiv:2510.18814, 2025c.

Li, P., Hu, Z., Shang, Z., Wu, J., Liu, Y., Liu, H., Gao, Z., Shi, C., Zhang, B., Zhang, Z., et al. Eficient multi-turn rl for gui agents via decoupled training and adaptive data curation. arXiv preprint arXiv:2509.23866, 2025d.

Li, P., Skripkin, M., Zubrey, A., Kuznetsov, A., and Oseledets, I. Confidence is all you need: Few-shot rl fine-tuning of language models. arXiv preprint arXiv:2506.06395, 2025e.

Li, Q., Xue, R., Wang, J., Zhou, M., Li, Z., Ji, X., Wang, Y., Liu, M., Yang, Z., Qiu, M., et al. Cure: Criticaltoken-guided re-concatenation for entropy-collapse prevention. arXiv preprint arXiv:2508.11016, 2025f.

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024d.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval: An automatic evaluator of instruction-following models, 2023b.

Li, X., Yu, H., Zhang, X., Huang, Z., He, S., Liu, K., Zhao, J., Huang, F., and Li, Y. Socraticprmbench: Benchmarking process reward models with systematic reasoning patterns. arXiv preprint arXiv:2505.23474, 2025g.

Li, X., Zou, H., and Liu, P. Torl: Scaling tool-integrated rl. arXiv preprint arXiv:2503.23383, 2025h.

Li, X., Li, X., Hu, S., Guo, Y., and Zhang, W. Verifybench: A systematic benchmark for evaluating reasoning verifiers across domains. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 31796–31804, 2026c.

Li, X.-Y., Liu, R.-B., Zhang, Y.-J., Sun, H., Xie, Z., and Li, M. Exploring pass-rate reward in reinforcement learning for code generation. arXiv preprint arXiv:2605.02944, 2026d.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

Li, Z., Chang, Y., Zhou, Y., Wu, X., Liang, Z., Sung, Y. Y., and Boyd-Graber, J. L. Semantically-aware rewards for open-ended r1 training in free-form generation. arXiv preprint arXiv:2506.15068, 2025i.

Li, Z., Zheng, M., Song, M., and Yang, W. Tat-r1: Terminology-aware translation with reinforcement learning and word alignment. arXiv preprint arXiv:2505.21172, 2025j.

Li, Z., Shi, Y., Li, Z., Wang, R., Li, A., Huang, Z., Yang, J., Ke, L., Liu, N., Mi, H., et al. Recursive synthesis for long-horizon terminal tasks. arXiv preprint arXiv:2608.05466, 2026e.

Li, Z., Yu, W., Huang, C., Liang, Z., Liu, R., Liu, F., Che, J., Yu, D., Boyd-Graber, J., Mi, H., et al. Self-rewarding vision-language model via reasoning decomposition. In ICLR, 2026f.

Li, Z.-Z., Zhang, D., Zhang, M.-L., Zhang, J., Liu, Z., Yao, Y., Xu, H., Zheng, J., Wang, P.-J., Chen, X., et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025k.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Lin, B. Y., Deng, Y., Chandu, K., Brahman, F., Ravichander, A., Pyatkin, V., Dziri, N., Bras, R. L., and Choi, Y. Wildbench: Benchmarking LLMs with challenging tasks from real users in the wild. arXiv preprint arXiv:2406.04770, 2024a.

Lin, B. Y., Bras, R. L., Richardson, K., Sabharwal, A., Poovendran, R., Clark, P., and Choi, Y. Zebralogic: On the scaling limits of llms for logical reasoning. arXiv preprint arXiv:2502.01100, 2025a.

Lin, H., Kuai, Z., Xue, E., and Wang, L. Detecting and mitigating the correct-answer extinction window in test-time reinforcement learning with majority voting. arXiv preprint arXiv:2605.19444, 2026a.

Lin, S., Hilton, J., and Evans, O. TruthfulQA: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2022.

Lin, Y., Tang, S., Lyu, B., Yang, Z., Chung, J.-H., Zhao, H., Jiang, L., Geng, Y., Ge, J., Sun, J., et al. Goedel-prover-v2: Scaling formal theorem proving with scafolded data synthesis and self-correction. In International Conference on Learning Representations, volume 2026, pp. 11793–11818, 2026b.

Lin, Z., Gou, Z., Liang, T., et al. Criticbench: Benchmarking LLMs for critique-correct reasoning. arXiv preprint arXiv:2402.14809, 2024b.

Lin, Z., Lin, M., Xie, Y., and Ji, R. Cppo: Accelerating the training of group relative policy optimizationbased reasoning models. arXiv preprint arXiv:2503.22342, 2025b.

Ling, W., Yogatama, D., Dyer, C., and Blunsom, P. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 158–167, 2017.

Liu, B., Guertler, L., Yu, S., Liu, Z., Qi, P., Balcells, D., Liu, M., Tan, C., Shi, W., Lin, M., Lee, W. S., and Jaques, N. Spiral: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. arXiv preprint arXiv:2506.24119, 2025a.

Liu, B., Jin, C., Kim, S., Yuan, W., Zhao, W., Kulikov, I., Li, X., Sukhbaatar, S., Lanchantin, J., and Weston, J. Spice: Self-play in corpus environments improves reasoning. arXiv preprint arXiv:2510.24684, 2025b.

Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., and Zhou, Y. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024a.

Liu, C. Y., Zeng, L., Xiao, Y., He, J., Liu, J., Wang, C., Yan, R., Shen, W., Zhang, F., Xu, J., et al. Skyworkreward-v2: Scaling preference data curation via human-ai synergy. arXiv preprint arXiv:2507.01352, 2025c.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pp. 3622–3628, 2021.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems, volume 36, 2023a.

Liu, J., Liu, H., Xiao, L., Wang, Z., Liu, K., Gao, S., Zhang, W., Zhang, S., and Chen, K. Are your llms capable of stable reasoning? arXiv preprint arXiv:2412.13147, 2024b.

Liu, J., He, C., Lin, Y., Yang, M., Shen, F., Liu, S., and Gao, T. Ettrl: Balancing exploration and exploitation in llm test-time reinforcement learning via entropy mechanism. arXiv preprint arXiv:2508.11356, 2025d.

Liu, J., Li, Y., Zhang, C., Li, J., Chen, A., Ji, K., Cheng, W., Wu, Z., Du, C., Xu, Q., et al. Webexplorer: Explore and evolve for training long-horizon web agents. arXiv preprint arXiv:2509.06501, 2025e.

Liu, J., An, S., Zhou, S., Ma, D., Luo, S., Xie, Y., Zhang, Y., Yuan, W., Zhou, Y., Li, X., et al. General365: Benchmarking general reasoning in large language models across diverse and challenging tasks. arXiv preprint arXiv:2604.11778, 2026a.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. Advances in Neural Information Processing Systems, 38:17998–18031, 2025f.

Liu, S., Li, Z., Liu, X., Zhan, R., Wong, D., Chao, L., and Zhang, M. Can llms learn uncertainty on their own? expressing uncertainty efectively in a self-training manner. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 21635–21645, 2024c.

Liu, S., Liu, H., Liu, J., Xiao, L., Gao, S., Lyu, C., Gu, Y., Zhang, W., Wong, D. F., Zhang, S., et al. Compassverifier: A unified and robust verifier for llms evaluation and outcome reward. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 33454–33482, 2025g.

Liu, T., Xu, R., Yu, T., Hong, I., Yang, C., Zhao, T., and Wang, H. Openrubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 17417–17437, 2026b.

Liu, W., Qi, S., Wang, X., Qian, C., Du, Y., and He, Y. Nover: Incentive training for language models via verifier-free reinforcement learning. arXiv preprint arXiv:2505.16022, 2025h.

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y., Ding, H., Men, K., Yang, K., et al. Agentbench: Evaluating LLMs as agents. In The Twelfth International Conference on Learning Representations, 2024d.

Liu, X., Chen, T., Da, L., Chen, C., Lin, Z., and Wei, H. Uncertainty quantification and confidence calibration in large language models: A survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 6107–6117, 2025i.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., and Zhu, C. G-eval: NLG evaluation using GPT-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023b.

Liu, Y., Yao, Z., Min, R., Cao, Y., Hou, L., and Li, J. Rm-bench: Benchmarking reward models of language models with subtlety and style. arXiv preprint arXiv:2410.16184, 2024e.

Liu, Y., Fu, P., Li, H., Qi, Y., Jiang, C., Fu, J., Liu, Z., Qin, B., Luo, Z., Luan, J., et al. Elva: Exploring ranking-driven universal multimodal retrieval. arXiv preprint arXiv:2606.20280, 2026c.

Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., Li, P., Liu, Y., and Wu, Y. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025j.

Liu, Z., Huang, W., Song, W., Liu, Y., Yang, Z., and Fu, J. An ai4ai framework for visual token pruning. arXiv preprint arXiv:2608.07193, 2026d.

Liu, Z., Liu, Y., Wang, J., Song, W., Liu, J., and Fu, J. Structured progressive knowledge activation for llm-driven neural architecture search. arXiv preprint arXiv:2605.04057, 2026e.

Liu, Z., Zhou, W., Bai, S., Liu, Y., Wang, J., and Fu, J. Graphir: Architecture-level search states for llm-guided neural architecture evolution. arXiv preprint arXiv:2608.01633, 2026f.

Lu, C., Hu, S., and Clune, J. Automated capability discovery via model self-exploration. arXiv preprint arXiv:2502.07577, 2025a.

Lu, H., Wen, Y., Cheng, P., Ding, R., Xu, H., Guo, J., Wang, C., Chen, H., Jiang, X., and Jiang, G. Search self-play: Pushing the frontier of agent capability without supervision. arXiv preprint arXiv:2510.18821, 2025b.

Lu, S., Wang, Z., Zhang, H., Wu, Q., Gan, L., Zhuang, C., Gu, J., and Lin, T. Don’t just fine-tune the agent, tune the environment. arXiv preprint arXiv:2510.10197, 2025c.

Lu, X. Writing-zero: Bridge the gap between non-verifiable problems and verifiable rewards. arXiv preprint arXiv:2506.00103, 2025.

Lu, Z., Chai, Y., Guo, Y., Yin, X., Liu, L., Wang, H., Xiao, H., Ren, S., Zhao, P., Liu, G., et al. Ui-r1: Enhancing eficient action prediction of gui agents by reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 17608–17616, 2026.

Luo, R., Wang, L., He, W., Chen, L., Li, J., and Xia, X. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.

Lushtaku, E., Kargi, B., Elganzory, A., Ferreira, F., Salamanca, A. R., Kreutzer, J., and Salinas, D. Judgearena: A unified framework for reproducible llm-judge evaluation. arXiv preprint arXiv:2608.02620, 2026.

Ma, Q., Gao, D., Cai, R., Zhao, B., Zhou, H., Zhang, J., and Zhao, Z. Personalized rewardbench: Evaluating reward models with human aligned personalization. In Conference on Language Modeling, 2026.

Ma, X., Liu, Q., Jiang, D., Zhang, G., Ma, Z., and Chen, W. General-reasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025.

Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., and Anandkumar, A. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Mai, S., Zhai, Y., Chen, Z., Chen, C., Zou, A., Tao, S., Liu, Z., and Ding, B. Cues: A curiosity-driven and environment-grounded synthesis framework for agentic rl. arXiv preprint arXiv:2512.01311, 2025.

Malik, S., Pyatkin, V., Land, S., Morrison, J., Smith, N. A., Hajishirzi, H., and Lambert, N. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

Mateega, S., Georgescu, C., and Tang, D. Financeqa: A benchmark for evaluating financial analysis capabilities of large language models. arXiv preprint arXiv:2501.18062, 2025.

Mathematical Association of America. AIME 2024, 2024.

Mathematical Association of America. AIME 2025, 2025.

Mazeika, M., Phan, L., Yin, X., et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

Mei, L., Yu, X., Chen, C., Liu, L., Chen, X., Ma, J., Feng, C., Huang, L., Mo, S., Kang, S., et al. Searchart: Training long-horizon search agent with scalable synthetic and verified task. arXiv preprint arXiv:2607.24850, 2026.

Meng, Y., Xia, M., and Chen, D. SimPO: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., and Scialom, T. GAIA: A benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations, 2024.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-t., Koh, P. W., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251, 2023.

Mirzadeh, I., Alizadeh, K., Shahrokhi, H., et al. GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models. arXiv preprint arXiv:2410.05229, 2024.

Modarressi, A., Deilamsalehy, H., Dernoncourt, F., et al. Nolima: Long-context evaluation beyond literal matching. arXiv preprint arXiv:2502.05167, 2025.

Moradi, M. M., Amer, H., Mudur, S., Zhang, W., Liu, Y., and Ahmed, W. Continuous self-improvement of large language models by test-time training with verifier-driven sample selection. arXiv preprint arXiv:2505.19475, 2025.

Mu, Y., Zeng, J., Li, B., Guan, X., Meng, F., Zhou, J., Xiao, T., and Zhu, J. Dissecting long-chain-of-thought reasoning models: An empirical study. arXiv preprint arXiv:2506.04913, 2025.

Nguyen, D. M., Phan, T. C., Le Hai, N., Doan, T.-T., Nguyen, N. V., Pham, Q., and Bui, N. D. Codemmlu: A multi-task benchmark for assessing code understanding & reasoning capabilities of codellms. In The Thirteenth International Conference on Learning Representations, 2025a.

Nguyen, X.-P., Pandit, S., Reddy, R. G., Xu, A., Savarese, S., Xiong, C., and Joty, S. Sfr-deepresearch: Towards efective reinforcement learning for autonomously reasoning single agents. arXiv preprint arXiv:2509.06283, 2025b.

Ni, J., Xue, F., Yue, X., Deng, Y., Shah, M., Jain, K., Neubig, G., and You, Y. Mixeval: Deriving wisdom of the crowd from LLM benchmark mixtures. In Advances in Neural Information Processing Systems, volume 37, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Paech, S. J. Eq-bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281, 2023.

Paglieri, D., Cupial, B., Coward, S., et al. BALROG: Benchmarking agentic LLM and VLM reasoning on games. arXiv preprint arXiv:2411.13543, 2024.

Pappone, F., Lazzaroni, R. M., Califano, F., Gentile, N., and Marras, R. Shaping explanations: Semantic reward modeling with encoder-only transformers for grpo. arXiv preprint arXiv:2509.13081, 2025.

Parashar, S., Gui, S., Li, X., Ling, H., Vemuri, S., Olson, B., Li, E., Zhang, Y., Caverlee, J., Kalathil, D., and Ji, S. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning. arXiv preprint arXiv:2506.06632, 2025.

Pathak, D., Agrawal, P., Efros, A. A., and Darrell, T. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pp. 2778–2787. PMLR, 2017.

Patil, S. G., Mao, H., Cheng-Jie Ji, C., Yan, F., Suresh, V., Stoica, I., and Gonzalez, J. E. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.

Phan, D., Hofman, M. D., Douglas, S., Le, T. A., Parisi, A. T., Sountsov, P., Sutton, C., Vikram, S., Saurous, R. A., et al. Training chain-of-thought via latent-variable inference. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Zhang, C. B. C., Shaaban, M., Ling, J., Shi, S., et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Plesner, A., Guzmán, F., and Athalye, A. An imperfect verifier is good enough: Learning with noisy rewards. arXiv preprint arXiv:2604.07666, 2026.

Prabhudesai, M., Chen, L., Ippoliti, A., Fragkiadaki, K., Liu, H., and Pathak, D. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660, 2025.

Qi, Y., Fu, P., Li, H., Liu, Y., Jiang, C., Qin, B., Luo, Z., and Luan, J. Patchcue: Enhancing vision-language model reasoning with patch-based visual cues. arXiv preprint arXiv:2603.05869, 2026a.

Qi, Z., Baby, S. M., Baby, S. A., Yuan, K., Tomkins, A., Vu, T., Juan, D.-C., and Rashtchian, C. On the generalization gap in self-evolving language model reasoning. arXiv preprint arXiv:2606.01075, 2026b.

Qian, C., Acikgoz, E. C., He, Q., Wang, H., Chen, X., Hakkani-Tur, D., Tur, G., and Ji, H. Toolrl: Reward is all tool learning needs. Advances in Neural Information Processing Systems, 38:105523–105553, 2025.

Qian, K., Wan, S., Tang, C., et al. Varbench: Robust language model benchmarking through dynamic variable perturbation. arXiv preprint arXiv:2406.17681, 2024.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In The Twelfth International Conference on Learning Representations, 2024.

Qin, Y., Ye, Y., Fang, J., Wang, H., Liang, S., Tian, S., Zhang, J., Li, J., Li, Y., Huang, S., et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Quan, S., Yang, J., Yu, B., Zheng, B., Liu, D., Yang, A., Ren, X., Gao, B., Miao, Y., Feng, Y., et al. Codeelo: Benchmarking competition-level code generation of llms with human-comparable elo ratings. arXiv preprint arXiv:2501.01257, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Rahman, A. and Alsharari, M. S. Veribound: Pac-bayesian generalization bounds for process reward models trained with formal verification tools. arXiv preprint arXiv:2606.20740, 2026.

Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations, 2025.

Ray, J. Before the model learns the bug: Fuzzing RLVR verifiers. arXiv preprint arXiv:2606.01066, 2026.

Rei, R., Stewart, C., Farinha, A. C., and Lavie, A. Comet: A neural framework for mt evaluation. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2685–2702, 2020.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Ren, Q., He, Q., Zhang, B., Zeng, J., Liang, J., Xiao, Y., Zhou, W., Sun, Z., and Yu, F. Beyond the trade-of: Self-supervised reinforcement learning for reasoning models’ instruction following. arXiv preprint arXiv:2508.02150, 2025a.

Ren, S., Guo, D., Lu, S., Zhou, L., Liu, S., Tang, D., Sundaresan, N., Zhou, M., Blanco, A., and Ma, S. Codebleu: a method for automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297, 2020.

Ren, Z., Shao, Z., Song, J., Xin, H., Wang, H., Zhao, W., Zhang, L., Fu, Z., Zhu, Q., Yang, D., et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025b.

Saha, S., Li, X., Ghazvininejad, M., Weston, J. E., and Wang, T. Learning to plan & reason for evaluation with thinking-llm-as-a-judge. In Forty-second International Conference on Machine Learning, 2025.

Saunders, W., Yeh, C., Wu, J., Bills, S., Ouyang, L., Ward, J., and Leike, J. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Seed, B., Chen, J., Fan, T., Liu, X., Liu, L., Lin, Z., Wang, M., Wang, C., Wei, X., Xu, W., et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

Seely, J., Imajuku, Y., Zhao, T., Cetin, E., and Jones, L. Sudoku-bench: Evaluating creative reasoning with sudoku variants. arXiv preprint arXiv:2505.16135, 2025.

Shafayat, S., Tajwar, F., Salakhutdinov, R., Schneider, J., and Zanette, A. Can large reasoning models self-train? arXiv preprint arXiv:2505.21444, 2025.

Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., et al. Spurious rewards: Rethinking training signals in rlvr. arXiv preprint arXiv:2506.10947, 2025a.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shao, Z., Luo, Y., Lu, C., Ren, Z., Hu, J., Ye, T., Gou, Z., Ma, S., and Zhang, X. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025b.

She, S., Bao, Y., Lu, Y., Xu, L., Li, T., Zhu, W., Huang, S., Cheng, S., Lu, L., and Wang, Y. Dupo: Enabling reliable llm self-verification via dual preference optimization. arXiv preprint arXiv:2508.14460, 2025a.

She, S., Liu, J., Liu, Y., Chen, J., Huang, X., and Huang, S. R-prm: Reasoning-driven process reward modeling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13449–13462, 2025b.

Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., and Vosoughi, S. Judging the judges: A systematic study of position bias in llm-as-a-judge. arXiv preprint arXiv:2406.07791, 2024.

Shi, Y., Liang, Z., Panaganti, K., Yu, D., Yu, W., and Mi, H. Learning to build the environment: Self-evolving reasoning rl via verifiable environment synthesis. arXiv preprint arXiv:2605.14392, 2026.

Shorinwa, O., Mei, Z., Lidard, J., Ren, A. Z., and Majumdar, A. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Computing Surveys, 2025.

Silver, D. and Sutton, R. S. Welcome to the era of experience. Google AI, 1, 2025.

Singh, A. K., Kocyigit, M. Y., Poulton, A., Esiobu, D., Lomeli, M., Szilvasy, G., and Hupkes, D. Evaluation data contamination in LLMs: How do we measure it and (when) does it matter? arXiv preprint arXiv:2411.03923, 2024.

Singhal, P., Goyal, T., Xu, J., and Durrett, G. A long way to go: Investigating length correlations in RLHF. arXiv preprint arXiv:2310.03716, 2023.

Skalse, J., Howe, N., Krasheninnikov, D., and Krueger, D. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35, pp. 9460–9471, 2022.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm test-time compute optimally can be more efective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Song, H., Jiang, J., Min, Y., Chen, J., Chen, Z., Zhao, W. X., Fang, L., and Wen, J.-R. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025a.

Song, M., Su, Z., Qu, X., Zhou, J., and Cheng, Y. PRMBench: A fine-grained and challenging benchmark for process-level reward models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025b.

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Su, Y., Yu, D., Song, L., Li, J., Mi, H., Tu, Z., Zhang, M., and Yu, D. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025.

Sullivan, M., Hartmann, M., and Koller, A. Procedural environment generation for tool-use agents. arXiv preprint arXiv:2506.11045, 2025.

Sun, H., Qiao, Z., Guo, J., Fan, X., Hou, Y., Jiang, Y., Xie, P., Zhang, Y., Huang, F., and Zhou, J. Zerosearch: Incentivize the search capability of llms without searching. arXiv preprint arXiv:2505.04588, 2025a.

Sun, J., Huang, W., Wu, J., Gu, C., Li, W., Zhang, S., Yan, H., and He, C. Benchmarking chinese commonsense reasoning of llms: From chinese-specifics to reasoning-memorization correlations. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 11205–11228, 2024.

Sun, J., Zheng, C., Xie, E., Liu, Z., Chu, R., Qiu, J., Xu, J., Ding, M., Li, H., Geng, M., et al. A survey of reasoning with foundation models: Concepts, methodologies, and outlook. ACM Computing Surveys, 57(11):1–43, 2025b.

Sun, S., Yu, J., Wang, Z., Yang, X., Gu, T., and Yang, Y. S2j: Bridging the gap between solving and judging ability in generative reward models. arXiv preprint arXiv:2509.22099, 2025c.

Sun, Z., Shen, Y., Zhou, Q., Zhang, H., Chen, Z., Cox, D., Yang, Y., and Gan, C. Principle-driven self-alignment of language models from scratch with minimal human supervision. arXiv preprint arXiv:2305.03047, 2023.

Sun, Z., Liu, Z., Zang, Y., Cao, Y., Dong, X., Wu, T., Lin, D., and Wang, J. Seagent: Self-evolving computer use agent with autonomous learning from experience. arXiv preprint arXiv:2508.04700, 2025d.

Sutton, R. S., Barto, A. G., et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Suzgun, M., Scales, N., Schaerli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging BIG-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Talmor, A., Herzig, J., Lourie, N., and Berant, J. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.

Tan, S., Zhuang, S., Montgomery, K., Tang, W., Cuadron, A., Wang, C., Popa, R., and Stoica, I. Judgebench: A benchmark for evaluating llm-based judges. In International Conference on Learning Representations, volume 2025, pp. 63277–63303, 2025.

Tang, Y., Wang, S., Madaan, L., and Munos, R. Beyond verifiable rewards: Scaling reinforcement learning for language models to unverifiable data. arXiv preprint arXiv:2503.19618, 2025.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025a.

Team, M., Dou, C., Liu, C., Yang, F., Li, F., Jia, J., Chen, M., Ju, Q., Wang, S., Dang, S., Li, T., Zeng, X., and Yijie Zhou, e. a. Baichuan-m2: Scaling medical capability with large verifier system, 2025b. URL https://arxiv.org/abs/2509.02208.

Tian, M., Gao, L., Zhang, S. D., et al. Scicode: A research coding benchmark curated by scientists. arXiv preprint arXiv:2407.13168, 2024.

Trivedi, H., Khot, T., Hartmann, M., Manku, R., Dong, V., Li, E., Gupta, S., Sabharwal, A., and Balasubramanian, N. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. arXiv preprint arXiv:2407.18901, 2024.

Tsoukalas, G., Lee, J., Jennings, J., et al. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214, 2024.

Tu, D., Hao, H., Yang, H., Chen, Y., Zhang, Y.-K., Xia, Z., Sun, Y., Liu, X., Shen, F., Gu, Q., Su, H., and Cai, X. Scaleenv: Scaling environment synthesis from scratch for generalist interactive tool-use agent training. arXiv preprint arXiv:2602.06820, 2026.

Tu, H., Feng, W., Chen, H., Liu, H., Tang, X., and Xie, C. Vilbench: A suite for vision-language process reward modeling. arXiv preprint arXiv:2503.20271, 2025.

Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N., Wang, L., Creswell, A., Irving, G., and Higgins, I. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Verga, P., Hofstatter, S., Su, S., Matzaris, P., Rocktäschel, T., and Lewis, M. Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. arXiv preprint arXiv:2404.18796, 2024.

Viswanathan, V., Sun, Y., Ma, S., Kong, X., Cao, M., Neubig, G., and Wu, T. Checklists are better than reward models for aligning language models. arXiv preprint arXiv:2507.18624, 2025.

Vu, T., Krishna, K., Alzubi, S., et al. Foundational autoraters: Taming large language models for better automatic evaluation. arXiv preprint arXiv:2407.10817, 2024.

Wang, C., Gan, Y., Huo, Y., Mu, Y., He, Q., Yang, M., Li, B., Xiao, T., Zhang, C., Liu, T., et al. Gram: A generative foundation reward model for reward generalization. In International Conference on Machine Learning, pp. 62916–62936. PMLR, 2025a.

Wang, C., Huo, Y., Gan, Y., He, Q., Meng, Q., Li, B., Wang, Y., Liu, J., Zhou, T., Zhu, J., et al. Msrl: Scaling generative multimodal reward modeling via multi-stage reinforcement learning. arXiv preprint arXiv:2603.25108, 2026a.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

Wang, H., Unsal, M., Lin, X., Baksys, M., Liu, J., Santos, M. D., Sung, F., Vinyes, M., Ying, Z., Zhu, Z., et al. Kimina-prover preview: Towards large formal reasoning models with reinforcement learning. arXiv preprint arXiv:2504.11354, 2025b.

Wang, H., Zou, H., Song, H., Feng, J., Fang, J., Lu, J., Liu, L., Luo, Q., Liang, S., Huang, S., et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025c.

Wang, J., Liu, R., Zhang, F., Li, X., and Zhou, G. Stabilizing knowledge, promoting reasoning: Dual-token constraints for rlvr. arXiv preprint arXiv:2507.15778, 2025d.

Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2024a.

Wang, Q., Liu, B., Zhou, T., Shi, J., Lin, Y., Chen, Y., Li, H. H., Wan, K., and Zhao, W. Vision-zero: Scalable vlm self-improvement via strategic gamified self-play. arXiv preprint arXiv:2509.25541, 2025e.

Wang, R., Lehman, J., Clune, J., and Stanley, K. O. Paired open-ended trailblazer (poet): Endlessly generating increasingly complex and diverse learning environments and their solutions. arXiv preprint arXiv:1901.01753, 2019.

Wang, S., Jiao, Z., Zhang, Z., Peng, Y., Xu, Z., Yang, B., Wang, W., Hu, W., and Zhang, L. Socratic-zero: Bootstrapping reasoning via data-free agent co-evolution. arXiv preprint arXiv:2509.24726, 2025f.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive efective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025g.

Wang, T., Kulikov, I., Golovneva, O., Yu, P., Yuan, W., Dwivedi-Yu, J., Pang, R. Y., Fazel-Zarandi, M., Weston, J., and Li, X. Self-taught evaluators. arXiv preprint arXiv:2408.02666, 2024b.

Wang, W., Gao, Z., Chen, L., Chen, Z., Zhu, J., Zhao, X., Liu, Y., Cao, Y., Ye, S., Zhu, X., et al. Visualprm: An efective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025h.

Wang, W., Wang, Y., Chen, K., and Huang, H. Beyond majority voting: Towards fine-grained and more reliable reward signal for test-time reinforcement learning. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 37251–37265, 2026b.

Wang, X., Hu, Z., Lu, P., Zhu, Y., Zhang, J., Subramaniam, S., Loomis, A. R., Zhou, B., et al. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023b.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023c.

Wang, X., Hou, Z., Zhang, Y., Yan, B., Yang, Z., Zhang, X., Xing, L., Zhou, Q., and Zhang, C. Evolvr: Self-evolving pairwise reasoning for story evaluation to enhance generation. arXiv preprint arXiv:2508.06046, 2025i.

Wang, X., Wang, P., Pei, J., Shen, W., Peng, Y., Hao, Y., Qiu, W., Jian, A., Xie, T., Song, X., et al. Skywork-vl reward: An efective reward model for multimodal understanding and reasoning. arXiv preprint arXiv:2505.07263, 2025j.

Wang, X., Hao, Z., Hou, S., Peng, H., Li, J., and Wang, X. Reproducing, analyzing, and detecting reward hacking in rubric-based reinforcement learning. arXiv preprint arXiv:2606.04923, 2026c. CHERRL.

Wang, X., Tian, M., Zeng, Y., Huang, Z., Yuan, J., Chen, B., Xu, J., Zhou, M., Liu, W., Wu, M., et al. Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges. arXiv preprint arXiv:2604.13602, 2026d.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.

Wang, Y., Yu, Z., Zeng, Z., Yang, L., Wang, C., Chen, H., Jiang, C., Xie, R., Wang, J., Xie, X., et al. PandaLM: An automatic evaluation benchmark for LLM instruction tuning optimization. arXiv preprint arXiv:2306.05087, 2023d.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024c.

Wang, Y., Yang, L., Tian, Y., Shen, K., and Wang, M. Co-evolving llm coder and unit tester via reinforcement learning. arXiv preprint arXiv:2506.03136, 2025k.

Wang, Y., Zang, Y., Li, H., Jin, C., and Wang, J. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025l.

Wang, Z., Dong, Y., Delalleau, O., Zeng, J., Shen, G., Egert, D., Zhang, J. J., Sreedhar, M. N., and Kuchaiev, O. Helpsteer2: Open-source dataset for training top-performing reward models. arXiv preprint arXiv:2406.08673, 2024d.

Wang, Z., Xia, M., He, L., et al. Charxiv: Charting gaps in realistic chart understanding in multimodal LLMs. arXiv preprint arXiv:2406.18521, 2024e.

Wang, Z., Zeng, J., Delalleau, O., Shin, H.-C., Soares, F., Bukharin, A., Evans, E., Dong, Y., and Kuchaiev, O. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. Advances in Neural Information Processing Systems, 38, 2025m.

Wang, Z., Xu, C., Liu, B., Wang, Y., Han, S., Yao, Z., Yao, H., and He, Y. Agent world model: Infinity synthetic environments for agentic reinforcement learning. arXiv preprint arXiv:2602.10090, 2026e.

Wei, J., Nguyen, K., Chung, H. W., Jiao, Y. J., Papay, S., Glaese, A., Schulman, J., and Fedus, W. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

Wei, J., Sun, Z., Papay, S., et al. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025a.

Wei, L., Li, Y., Wang, C., Wang, Y., Kong, L., Huang, W., and Sun, L. First sft, second rl, third upt: Continual improving multi-modal llm reasoning via unsupervised post-training. arXiv preprint arXiv:2505.22453, 2025b.

Wei, Y., Duchenne, O., Copet, J., Carbonneaux, Q., Zhang, L., Fried, D., Synnaeve, G., Singh, R., and Wang, S. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. Advances in Neural Information Processing Systems, 38:78500–78525, 2026.

Wen, X., Liu, Z., Zheng, S., Xu, Z., Ye, S., Wu, Z., Liang, X., Wang, Y., Li, J., Miao, Z., et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., Jain, S., Shwartz-Ziv, R., Jain, N., Saifullah, K., Dey, S., et al. Livebench: A challenging, contamination-limited llm benchmark. In International Conference on Learning Representations, volume 2025, pp. 91595–91631, 2025.

Whitehouse, C., Wang, T., Yu, P., Li, X., Weston, J., Kulikov, I., and Saha, S. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning. arXiv preprint arXiv:2505.10320, 2025.

Wu, F. and Choi, Y. The invisible leash: Why rlvr may not escape its origin. In 2nd AI for Math Workshop@ ICML 2025, 2025.

Wu, H., Wang, C., Zhao, W., and He, J. Mirage or method? how model–task alignment induces divergent rl conclusions. In International Conference on Learning Representations, volume 2026, pp. 123759–123783, 2026a.

Wu, J., Li, B., Fang, R., Yin, W., Zhang, L., Wang, Z., Tao, Z., Zhang, D.-C., Xi, Z., Tang, R., et al. Webdancer: Towards autonomous information seeking agency. Advances in Neural Information Processing Systems, 38:120957–120985, 2025a.

Wu, M., Zhang, Z., Dong, Q., Xi, Z., Zhao, J., Jin, S., Fan, X., Zhou, Y., Fu, Y., Liu, Q., et al. Reasoning or memorization? unreliable results of reinforcement learning due to data contamination. arXiv preprint arXiv:2507.10532, 2025b.

Wu, P., Zhang, X., Wan, K., Zhao, W., Wu, G., Du, X., and Chen, Z. Amaris: A memory-augmented rubric improvement system for rubric-based reinforcement learning. arXiv preprint arXiv:2605.18592, 2026b.

Wu, R., Wang, X., Mei, J., Cai, P., Fu, D., Yang, C., Wen, L., Yang, X., Shen, Y., Wang, Y., et al. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025c.

Wu, T., Yuan, W., Golovneva, O., Xu, J., Tian, Y., Jiao, J., Weston, J., and Sukhbaatar, S. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025d.

Wu, X., Pan, L., Xie, Y., et al. Antileakbench: Preventing data contamination by automatically constructing benchmarks with updated real-world knowledge. arXiv preprint arXiv:2412.13670, 2024.

Wu, Z., Huang, S., Zhou, Z., Ying, H., Yuan, Z., Zhang, W., Lin, D., and Chen, K. Internlm2. 5-stepprover: Advancing automated theorem proving via critic-guided search. In 2nd AIfor Math Workshop@ ICML 2025, 2025e.

Xie, J., Xu, D., Zhao, X., and Song, D. Agentsynth: Scalable task generation for generalist computer-use agents. arXiv preprint arXiv:2506.14205, 2025.

Xie, Q., Han, W., Chen, Z., Xiang, R., Zhang, X., He, Y., Xiao, M., Li, D., Dai, Y., Feng, D., et al. Finben: A holistic financial benchmark for large language models. Advances in Neural Information Processing Systems, 37:95716–95743, 2024a.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Toh, T. J., et al. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, volume 37, 2024b.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. arXiv preprint arXiv:2312.11456, 2024.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. Wizardlm: Empowering large pre-trained language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.

Xu, C., Guan, S., Greene, D., and Kechadi, M.-T. Benchmark data contamination of large language models: A survey. arXiv preprint arXiv:2406.04244, 2024.

Xu, F. F., Song, Y., Li, B., Tang, Y., Jain, K., Bao, M., Wang, Z. Z., Zhou, X., Guo, Z., Cao, M., et al. Theagentcompany: Benchmarking LLM agents on consequential real world tasks. In Advances in Neural Information Processing Systems, volume 38, 2025a.

Xu, R., Liu, T., Dong, Z., Yu, T., Hong, I., Yang, C., Zhang, L., Zhao, T., and Wang, H. Alternating reinforcement learning for rubric-based reward modeling in non-verifiable llm post-training. In Forty-third International Conference on Machine Learning, 2026a.

Xu, Y., Chakraborty, T., Sharma, S., Nunes, L., Kıcıman, E., Lu, S., and Chandra, R. Direct reasoning optimization: Token-level reasoning reflectivity meets rubric gates for unverifiable tasks. arXiv preprint arXiv:2506.13351, 2025b.

Xu, Y., Sang, H., Zhou, Z., He, R., and Wang, Z. Overconfident errors need stronger correction: Asymmetric confidence penalties for reinforcement learning. arXiv preprint arXiv:2602.21420, 2026b.

Xu, Z., Li, Y., Jiang, F., Ramasubramanian, B., Niu, L., Lin, B. Y., and Poovendran, R. Tinyv: Reducing false negatives in verification improves rl for llm reasoning. In 2nd AI for Math Workshop@ ICML 2025, 2025c.

Xu, Z., Lu, J., Zheng, Y., Wang, Y., and Qiu, X. Swe-bench science: Can coding agents resolve engineering tasks in science? arXiv preprint arXiv:2608.19799, 2026c.

Xue, S., Ding, Z., Shen, Y., Wang, Y., Yin, Z., Wu, Y., Chen, Y., Wang, M., and Yang, L. Past-bench: Benchmarking the foundations of recursive self-improvement in personal agents. arXiv preprint arXiv:2608.04003, 2026.

Yan, D., Liang, J., Wang, Y., Lu, S., He, R., and Tan, T. What if consensus lies? selective-complementary reinforcement learning at test time. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 28957–28970, 2026.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, C., Su, S., Liu, S., Dong, X., Yu, Y., Su, W., Wang, X., Liu, Z., Zhu, J., Li, H., et al. Zerogui: Automating online gui learning at zero human cost. arXiv preprint arXiv:2505.23762, 2025b.

Yang, C., Xiang, Z., Tang, Y., Teng, Z., Huang, C., Long, F., Liu, Y., and Su, J. Ttcs: Test-time curriculum synthesis for self-evolving. arXiv preprint arXiv:2601.22628, 2026a.

Yang, J., Jimenez, C. E., Zhang, A. L., et al. SWE-bench multimodal: Do AI systems generalize to visual software domains? arXiv preprint arXiv:2410.03859, 2024a.

Yang, W., Zheng, M., Song, M., Li, Z., and Wang, S. Ssr-zero: Simple self-rewarding reinforcement learning for machine translation. arXiv preprint arXiv:2505.16637, 2025c.

Yang, X., Sun, K., Xin, H., Sun, Y., Bhalla, N., Chen, X., Choudhary, S., Gui, R. D., Jiang, Z., Jiang, Z., et al. CRAG – comprehensive RAG benchmark. arXiv preprint arXiv:2406.04744, 2024b.

Yang, Z., Chen, Y., Yang, W., Zhang, E., Shen, Z., Wei, X., Gao, Y., Wu, Y., Hu, Y., and Mao, J. Tournament-grpo: Group-wise tournament rewards for reinforcement learning in open-ended long-form generation. arXiv preprint arXiv:2605.26958, 2026b.

Yang, Z., Shen, W., Li, C., Chen, R., Wan, F., Yan, M., Quan, X., and Huang, F. Spell: Self-play reinforcement learning for evolving long-context language models. In International Conference on Learning Representations, volume 2026, pp. 59910–59942, 2026c.

Yang, Z., Wang, S., Fu, K., He, W., Xiong, W., Liu, Y., Miao, Y., Gao, B., Wang, Y., Li, Y., et al. Kimi-dev: Agentless training as skill prior for swe-agents. In International Conference on Learning Representations, volume 2026, pp. 148369–148436, 2026d.

Yao, S., Shinn, N., Razavi, P., and Narasimhan, K. Tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Yao, X., Yu, L., Hu, X., Teng, F., Cui, Q., Zhou, J., and Liu, Y. The debate on rlvr reasoning capability boundary: Shrinkage, expansion, or both? a two-stage dynamic view. arXiv preprint arXiv:2510.04028, 2025.

Ye, J., Zhang, X., Xu, H., Liu, H., Wang, J., Zhu, Z., Zheng, Z., Gao, F., Cao, J., Lu, Z., et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.

Ye, S., Kim, D., Kim, S., Hwang, H., Kim, S., Jo, Y., Thorne, J., Kim, J., and Seo, M. FLASK: Fine-grained language model evaluation based on alignment skill sets. arXiv preprint arXiv:2307.10928, 2023.

Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., and Chen, D. Helmet: How to evaluate long-context language models efectively and thoroughly. In The Thirteenth International Conference on Learning Representations, 2025.

Yoran, O., Amouyal, S. J., Malaviya, C., et al. Assistantbench: Can web agents solve realistic and time-consuming tasks? arXiv preprint arXiv:2407.15711, 2024.

Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.-Q., Ma, W.-Y., Liu, J., Wang, M., et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025a.

Yu, P., Lanchantin, J., Wang, T., Yuan, W., Golovneva, O., Kulikov, I., Sukhbaatar, S., Weston, J., and Xu, J. Cot-self-instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks. arXiv preprint arXiv:2507.23751, 2025b.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025c.

Yu, T., Ji, B., Wang, S., Yao, S., Wang, Z., Cui, G., Yuan, L., Ding, N., Yao, Y., Liu, Z., et al. Rlpr: Extrapolating rlvr to general domains without verifiers. arXiv preprint arXiv:2506.18254, 2025d.

Yu, Z., Zeng, J., Gu, W., Wang, Y., Wang, J., Meng, F., Zhou, J., Zhang, Y., Zhang, S., and Ye, W. Rewardanything: Generalizable principle-following reward models. arXiv preprint arXiv:2506.03637, 2025e.

Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., and Weston, J. E. Self-rewarding language models. In International Conference on Machine Learning, pp. 57905–57923. PMLR, 2024.

Yue, X., Zheng, T., Ni, Y., et al. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024a.

Yue, X., Zheng, T., Zhang, G., and Chen, W. Mammoth2: Scaling instructions from the web. Advances in Neural Information Processing Systems, 37:90629–90660, 2024b.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Yue, Y., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? Advances in Neural Information Processing Systems, 38:57654–57689, 2025.

Zan, D., Huang, Z., Liu, W., et al. Multi-swe-bench: A multilingual benchmark for issue resolving. arXiv preprint arXiv:2504.02605, 2025.

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. D. Star: Bootstrapping reasoning with reasoning. arXiv preprint arXiv:2203.14465, 2022.

Zeng, H., Jiang, D., Wang, H., Nie, P., Chen, X., and Chen, W. Acecoder: Acing coder rl via automated test-case synthesis. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 12023–12040, 2025a.

Zeng, T., Zhang, S., Wu, S., Classen, C., Chae, D., Ewer, E., Lee, M., Kim, H., Kang, W., Kunde, J., et al. Versaprm: Multi-domain process reward model via synthetic reasoning data. In International Conference on Machine Learning, pp. 74197–74239. PMLR, 2025b.

Zeng, Z., Yu, J., Gao, T., et al. Evaluating large language models at evaluating instruction following. arXiv preprint arXiv:2310.07641, 2024.

Zha, K., Gao, Z., Shen, M., Hong, Z.-W., Boning, D., and Katabi, D. Rl tango: Reinforcing generator and verifier together for language reasoning. Advances in Neural Information Processing Systems, 38: 119283–119313, 2025.

Zhang, D., Zhang, S., Yang, Z., Zhu, Z., Zhao, Z., Cao, R., Chen, L., and Yu, K. Progrm: Build better gui agents with progress rewards. arXiv preprint arXiv:2505.18121, 2025a.

Zhang, H., Wang, P., Diao, S., Lin, Y., Pan, R., Dong, H., Zhang, D., Molchanov, P., and Zhang, T. Entropy-regularized process reward model. Transactions on Machine Learning Research, 2024a.

Zhang, J., Hu, S., Lu, C., Lange, R. T., and Clune, J. Darwin gödel machine: Open-ended evolution of self-improving agents. arXiv preprint arXiv:2505.22954, 2025b.

Zhang, J., Fu, Z., Xi, Z., Jing, W., Chai, M., He, W., Zhang, G., Fan, C., An, C., Chen, W., et al. Agentv-rl: Scaling reward modeling with agentic verifier. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 23078–23100, 2026a.

Zhang, K., Yao, Q., Liu, S., Wang, Y., Lai, B., Ye, J., Song, M., and Tao, D. Consistent paths lead to truth: Self-rewarding reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.08745, 2025c.

Zhang, L., Hosseini, A., Bansal, H., Kazemi, S. M., Kumar, A., and Agarwal, R. Generative verifiers: Reward modeling as next-token prediction. In International Conference on Learning Representations, volume 2025, pp. 12476–12505, 2025d.

Zhang, M., Ding, S., Yin, W., Sun, Y., and Wu, H. Extending rlvr to open-ended tasks via verifiable multiple-choice reformulation. arXiv preprint arXiv:2511.02463, 2025e.

Zhang, Q., Wu, H., Zhang, C., Zhao, P., and Bian, Y. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025f.

Zhang, Q., Chen, B., Zhang, F., Ding, R., Wang, S., Wang, Q., Huang, Y., Zhang, H., Zhu, R., Wang, P., et al. Arenarl: Scaling rl for open-ended agents via tournament-based relative ranking. arXiv preprint arXiv:2601.06487, 2026b.

Zhang, S., Liu, X., Zhang, X., Liu, J., Luo, Z., Huang, S., and Gong, Y. Process-based self-rewarding language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 18097–18110, 2025g.

Zhang, X., Chen, Y., Hu, S., Xu, Z., Chen, J., Hao, M. K., Han, X., Thai, Z. L., Wang, S., Liu, Z., et al. InfinityBench: Extending long context evaluation beyond 100k tokens. arXiv preprint arXiv:2402.13718, 2024b.

Zhang, X., Wei, J., Zhong, D., Chen, Q., Jia, C., Tan, C., Gu, J., Qin, X., Liu, Z., Hu, L., et al. Structvrm: Aligning multimodal reasoning with structured and verifiable reward models. arXiv preprint arXiv:2508.05383, 2025h.

Zhang, X., Wen, S., Wu, W., and Huang, L. Edge-grpo: Entropy-driven grpo with guided error correction for advantage diversity. arXiv preprint arXiv:2507.21848, 2025i.

Zhang, Y., Wang, T., Gesi, J., Wang, Z., Lu, Y., Lin, J., Zhan, S., Gao, V., Jiao, R., Liu, J., et al. Shop-r1: Rewarding llms to simulate human behavior in online shopping via reinforcement learning. arXiv preprint arXiv:2507.17842, 2025j.

Zhang, Y., Zhang, Z., Guan, H., Cheng, Y., Duan, Y., Wang, C., Wang, Y., Zheng, S., and He, J. No free lunch: Rethinking internal feedback for llm reasoning. arXiv preprint arXiv:2506.17219, 2025k.

Zhang, Y.-F., Lu, X., Hu, X., Fu, C., Wen, B., Zhang, T., Liu, C., Jiang, K., Chen, K., Tang, K., et al. R1-reward: Training multimodal reward model through stable reinforcement learning. arXiv preprint arXiv:2505.02835, 2025l.

Zhang, Y.-F., Yang, H., Zhang, H., Shi, Y., Chen, Z., Tian, H., Fu, C., Wang, H., Wu, K., Cui, B., et al. Basereward: A strong baseline for multimodal reward model. arXiv preprint arXiv:2509.16127, 2025m.

Zhang, Z., Zheng, C., Wu, Y., Zhang, B., Lin, R., Yu, B., Liu, D., Zhou, J., and Lin, J. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495–10516, 2025n.

Zhang, Z., Zhu, J., Ge, X., Zhao, Z., Zhou, Z., Li, X., Feng, X., Yao, J., and Han, B. Co-rewarding: Stable self-supervised rl for eliciting reasoning in large language models. arXiv preprint arXiv:2508.00410, 2025o.

Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Lin, M., Wang, S., Wu, Q., Zheng, Z., and Huang, G. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025a.

Zhao, J., Liu, R., Zhang, K., Zhou, Z., Gao, J., Li, D., Lyu, J., Qian, Z., Qi, B., Li, X., et al. Genprm: Scaling test-time compute of process reward models via generative reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 34932–34940, 2026.

Zhao, Q., Huang, Y., Lv, T., et al. MMLU-CF: A contamination-free multi-task language understanding benchmark. arXiv preprint arXiv:2412.15194, 2024.

Zhao, X., Kang, Z., Feng, A., Levine, S., and Song, D. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025b.

Zhao, Y., Khalman, M., Joshi, R., Narayan, S., Saleh, M., and Liu, P. J. Calibrating sequence likelihood improves conditional language generation. In The Eleventh International Conference on Learning Representations, 2023.

Zhao, Y., Liu, H., Yu, D., Kung, S., CHEN, M., Mi, H., and Yu, D. One token to fool llm-as-a-judge. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025c.

Zheng, C., Zhang, Z., Zhang, B., Lin, R., Lu, K., Yu, B., Liu, D., Zhou, J., and Lin, J. Processbench: Identifying process errors in mathematical reasoning. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 2025a.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Zheng, X., An, K., Wang, Z., Wang, Y., and Wu, Y. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 21816–21841, 2025b.

Zhong, W., Liu, C., Wu, Y., Tan, B., Sun, C., Wang, Y., Liu, X., and Kuang, K. P2s: Probabilistic process supervision for general-domain reasoning question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Zhou, C. More convincing, not more correct: Self-play reward hacking of reference-free llm judges. arXiv preprint arXiv:2607.05904, 2026.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023a.

Zhou, J., Chen, Y., Shi, Y., Zhang, X., Lei, L., Feng, Y., Xiong, Z., Yan, M., Wang, X., Cao, Y., et al. Socialeval: Evaluating social intelligence of large language models. arXiv preprint arXiv:2506.00900, 2025a.

Zhou, S., Alon, U., Agarwal, S., and Neubig, G. Codebertscore: Evaluating code generation with pretrained models of code. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13921–13937, 2023b.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., Alon, U., and Neubig, G. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024.

Zhou, X., Liu, Z., Sims, A., Wang, H., Pang, T., Li, C., Wang, L., Lin, M., and Du, C. Reinforcing general reasoning without verifiers. arXiv preprint arXiv:2505.21493, 2025b.

Zhou, Y., Levine, S., Weston, J., Li, X., and Sukhbaatar, S. Self-challenging language model agents. arXiv preprint arXiv:2506.01716, 2025c.

Zhou, Y., Li, S., Liu, S., Fang, W., Zhao, J., Yang, J., Lv, J., Zhang, K., Zhou, Y., Lu, H., et al. Breaking the exploration bottleneck: Rubric-scafolded reinforcement learning for general llm reasoning. arXiv preprint arXiv:2508.16949, 2025d.

Zhou, Y., Xu, A., Wang, P., Xiong, C., and Joty, S. Evaluating judges as evaluators: The jetts benchmark of llm-as-judges as test-time scaling evaluators. arXiv preprint arXiv:2504.15253, 2025e.

Zhou, Z., Lu, X., Cao, C., Miranda, B., Liu, T., Han, B., and Koyejo, S. Codapo: Confidence and dificultyadaptive policy optimization for post-training language models. In 2nd AIfor Math Workshop@ ICML, 2025f.

Zhu, K., Chen, J., Wang, J., Gong, N. Z., Yang, D., and Xie, X. Dyval: Dynamic evaluation of large language models for reasoning tasks. In The Twelfth International Conference on Learning Representations, 2024.

Zhu, L., Wang, X., and Wang, X. Judgelm: Fine-tuned large language models are scalable judges. In International Conference on Learning Representations, volume 2025, pp. 51257–51296, 2025a.

Zhu, P., Li, L., Lyu, Y., Luo, Q., Yang, J., Liu, Y., Hui, T., Yuan, X., Sun, L., Su, S., et al. A unified framework for the evaluation of llm agentic capabilities. arXiv preprint arXiv:2605.27898, 2026a.

Zhu, Q., Huang, F., Peng, R., Lu, K., Yu, B., Cheng, Q., Qiu, X., Huang, X., and Lin, J. Autologi: Automated generation of logic puzzles for evaluating reasoning abilities of large language models. arXiv preprint arXiv:2502.16906, 2025b.

Zhu, Y. and Kang, D. Noisy data is destructive to reinforcement learning with verifiable rewards. arXiv preprint arXiv:2603.16140, 2026.

Zhu, Z., Xu, H., Luo, Y., Liu, Y., Sarkar, K., Xu, K., and You, Y. Camel: Confidence-gated reflection for reward modeling. arXiv preprint arXiv:2602.20670, 2026b.

Zhuo, T. Y. Ice-score: Instructing large language models to evaluate code. In Findings of the Association for Computational Linguistics: EACL 2024, pp. 2232–2242, 2024.

Zhuo, T. Y., Vu, M. C., Chim, J., Hu, H., Yu, W., Widyasari, R., Yusuf, I. N. B., Zhan, H., He, J., Paul, I., et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877, 2024.

Zou, W., Yang, S., Bao, Y., Huang, S., Chen, J., and Cheng, S. Trans-zero: Self-play incentivizes large language models for multilingual translation without parallel data. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 12337–12347, 2025.

Zuo, Y., Qu, S., Li, Y., Chen, Z., Zhu, X., Hua, E., Zhang, K., Ding, N., and Zhou, B. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. arXiv preprint arXiv:2501.18362, 2025a.

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Long, X., Hua, E., Qi, B., et al. Ttrl: Test-time reinforcement learning. Advances in Neural Information Processing Systems, 38: 131459–131483, 2025b.

Zuo, Y., Xiao, Z., Sheng, L., Huang, F., Tu, J., Liu, Y., Tang, T., Hu, X., Su, Y., Lan, Q., et al. Qwen-agentworld: Language world models for general agents. arXiv preprint arXiv:2606.24597, 2026.