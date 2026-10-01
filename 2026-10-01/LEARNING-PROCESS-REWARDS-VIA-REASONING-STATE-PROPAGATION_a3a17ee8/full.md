# LEARNING PROCESS REWARDS VIA REASONING STATE PROPAGATION

Kai Gan<sup>1,2</sup>, Zi-Hao Zhou<sup>1,2</sup>, Bo Ye<sup>1,2,3</sup>, Jian Zhao<sup>3,4</sup>, Min-Ling Zhang<sup>1,2</sup>, Tong Wei<sup>1,2,†</sup>

<sup>1</sup>School of Computer Science and Engineering, Southeast University, Nanjing 210096, China

<sup>2</sup>Key Laboratory of Computer Network and Information Integration (Southeast University),

Ministry of Education, China

<sup>3</sup>Zhongguancun Academy

<sup>4</sup>Zhongguancun Institute of Artificial Intelligence

## ABSTRACT

Process reward models (PRMs) have demonstrated notable effectiveness in testtime scaling and reinforcement learning by providing fine-grained signals for evaluating intermediate reasoning states, but their training relies heavily on costly process annotations. A natural way to alleviate this dependence is to complement limited process supervision with scalable outcome supervision. However, existing PRMs often model reasoning prefixes independently, providing no explicit mechanism for effectively using final outcome to guide the learning of intermediate reasoning states. We introduce Reasoning State Propagation (RSP), which represents each reasoning prefix with a binary validity state and models transitions between successive states across the reasoning trajectory. Specifically, RSP predicts a break probability that a valid state becomes invalid and a repair probability that an invalid state returns to valid. By propagating these transitions, RSP connects intermediate states to the final state, allowing process annotations to supervise intermediate states while outcome labels supervise the final state and can provide learning signals to preceding steps. Across reasoning search, response selection, and reinforcement learning, RSP consistently outperforms representative PRM baselines, with average improvements over Qwen2.5-Math-PRM of 5.6% in beam search and 2.1% in reinforcement learning.

## 1 INTRODUCTION

Reinforcement learning with verifiable rewards (RLVR) (Jaech et al., 2024; Guo et al., 2025; Team et al., 2025; Yu et al., 2026) has driven substantial progress in large reasoning models (LRMs) (Yang et al., 2025a; Seed et al., 2025; Xiaomi et al., 2025). However, RLVR typically relies on finalanswer correctness as a trajectory-level reward, reducing a multi-step reasoning process to a single scalar signal. Such coarse supervision cannot distinguish the contributions of individual reasoning steps, making fine-grained credit assignment difficult. This limitation is especially problematic for all-incorrect groups on challenging problems. In group-relative methods such as GRPO (Shao et al., 2024), identical rewards within all-incorrect groups result in zero advantages after group normalization, leaving no learning signal for policy optimization. To address these limitations, PRMs (Lightman et al., 2024; Wang et al., 2024; Luo et al., 2024; Zhang et al., 2024; Li & Li, 2025; Zhang et al., 2025b; Yang et al., 2025b; Zhang et al., 2025a) have emerged to provide step-level feedback throughout the reasoning process.

Despite the notable success of PRMs in test-time scaling (Lightman et al., 2024; Lu et al., 2024; Snell et al., 2025) and reinforcement learning (Li et al., 2026; Agrawal et al., 2026), PRM training typically relies on process annotations, which are costly and difficult to obtain at scale. Such annotations provide supervision over intermediate reasoning steps and are particularly valuable for fine-grained process evaluation. In contrast, outcome labels are substantially cheaper and readily scalable in verifiable domains since they only require determining whether the complete reasoning trajectory ultimately reaches a correct answer. This disparity suggests a natural complementary supervision strategy: scarce process annotations provide guidance on intermediate reasoning states, while abundant outcome labels offer scalable supervision over a much larger collection of trajecto ries. However, how to effectively integrate these two signals within a PRM remains underexplored.

![](images/29f2589b5be8ce1e66e7ab549cffdc153a7e1b4079890a6ec7e57045f2ba633a.jpg)  
Figure 1: (a) Empirical performance across evaluation settings. (b) RSP propagates a binary reasoning state across steps using break and repair probabilities. G denotes a Good or Valid state with no unresolved error, while B denotes a Bad or Invalid state with an unresolved error. Transitions G→B and B →G correspond to error introduction (break) and error recovery (repair), respectively.

A key obstacle to effectively exploiting this complementary supervision is how to relate intermediate reasoning states to the final outcome. Existing discriminative PRMs optimize individual prefixes as independent classification targets (Lightman et al., 2024; Zhang et al., 2025b), providing no explicit structure for connecting reasoning states across steps. Recent methods such as PQM (Li & Li, 2025) and CRM (Zhang et al., 2025a) introduce sequential dependencies, but largely organize the reasoning process around the first erroneous step and effectively treat subsequent reasoning as remaining invalid. Such a formulation oversimplifies the relationship between intermediate reasoning states and final correctness. An intermediate error may be corrected by later reasoning and therefore does not necessarily lead to an incorrect final outcome. Similarly, a correct final outcome does not imply that every preceding reasoning step is valid. Another limitation is reflected in the popular process annotation protocol of PRM800K (Lightman et al., 2024), where annotation terminates once the first error is identified, leaving subsequent reasoning unannotated. In fact, modern LRMs can detect and revise earlier mistakes during continued reasoning (Guo et al., 2025; Lee et al., 2026). This motivates a more general formulation that explicitly propagates reasoning states across steps.

In this paper, we introduce Reasoning State Propagation (RSP) for PRM training. Rather than independently predicting the validity of each reasoning prefix, RSP associates each prefix with an effective reasoning state that indicates whether the currently maintained derivation is valid or contains an unresolved error and propagates this state across successive prefixes. Consider a reasoning trajectory composed of steps $( s _ { 1 } , \ldots , s _ { T } )$ , where $s _ { t }$ denotes the t-th reasoning step. When extending the preceding prefix $\mathbf { s } _ { < t }$ with step $s _ { t }$ , RSP estimates two conditional update probabilities: a break probability, which measures the likelihood that a previously valid state becomes invalid after incorporating $s _ { t } .$ , and a repair probability, which measures the likelihood that the new step resolves or supersedes an earlier error and returns the current reasoning state to a valid state. These probabilities determine how the state of the preceding prefix is updated after incorporating $s _ { t } .$ Repeating this process throughout the trajectory explicitly links the reasoning states across steps. The framework of RSP is shown in Figure 1(b).

RSP provides a natural interface for jointly leveraging process and outcome supervision. Process annotations can ground the validity of intermediate reasoning states, while outcome labels supervise the final propagated state of the complete trajectory. Since the final state depends on the sequence of preceding state updates, outcome supervision can supervise the final state and provide learning sig nals through the intervening state updates. This provides a way for outcome feedback to complement the model’s locally grounded validity predictions on trajectories without process labels.

Our main contributions are summarized as follows:

• We propose a new formulation RSP for process reward modeling that models transitions between successive reasoning states through break and repair, rather than treating each reasoning prefix independently, thus explicitly linking reasoning states across the trajectory.

• RSP jointly leverages scarce process annotations and scalable outcome supervision, with process labels grounding intermediate states and outcome labels constraining the final state. We further analyze how both signals provide training signals for state updates.

• Extensive experiments across test-time scaling and reinforcement learning demonstrate consistent improvements over representative PRMs. Comprehensive analyses further validate the effectiveness of reasoning state propagation and its key design choices.

## 2 RELATED WORK

## 2.1 ENHANCING REASONING IN LRMS

Recent studies improve the reasoning capabilities of LRMs through test-time scaling and reinforce ment learning. Lightman et al. (2024) use process reward models to rank sampled solutions in Best-of-N selection. Xie et al. (2023) introduce self-evaluation-guided beam search, which prunes intermediate reasoning prefixes during generation. REST-MCTS\* (Zhang et al., 2024) further incorporates process rewards into Monte Carlo search and uses the resulting trajectories for self-training. Beyond test-time scaling, DeepSeek-R1 (Guo et al., 2025) demonstrates that reinforcement learning with verifiable outcomes can substantially strengthen reasoning capabilities. PURE (Cheng et al., 2026) introduces min-form credit assignment, using future process rewards to provide denser supervision for intermediate reasoning steps. VeriGate (Agrawal et al., 2026) selectively activates PRM-based token-level supervision when outcome rewards fail to provide group-relative learning signal. These advances increasingly rely on reliable and fine-grained process feedback, motivating accurate and scalable PRMs for evaluating intermediate reasoning states.

## 2.2 PROCESS REWARD MODELING

PRMs rely on fine-grained signals about intermediate reasoning steps, which are commonly obtained through human annotation (Uesato et al., 2022; Lightman et al., 2024), rollout-based estima tion (Wang et al., 2024; Tan et al., 2025), or automated judging (Zhang et al., 2025b; She et al., 2025). PRM800K (Lightman et al., 2024) collects human judgments of step-level correctness and trains discriminative PRMs to classify each reasoning step as valid or invalid. Math-Shepherd (Wang et al., 2024) reduces annotation costs by estimating step correctness from whether sampled contin uations can reach the correct final answer. Zhang et al. (2025b) improve automatically constructed labels through consensus filtering. With these labels, conventional PRMs typically optimize stepwise classification objectives without modeling dependencies between adjacent reasoning prefixes. OVM (Yu et al., 2024) learns the probability that a partial trajectory can eventually reach a correct answer using outcome-annotated rollouts. PQM (Li & Li, 2025) formulates process reward modeling as Q-value ranking, capturing relative relationships among steps within a trajectory. CRM (Zhang et al., 2025a) models the first invalid state and links the process rewards to the final outcome using the probability chain rule. Other recent studies improve process reward modeling through dynamic evaluation criteria or retrieval-augmented verification (Yin et al., 2025; Zhu et al., 2025). However, existing PRMs do not jointly exploit process- and outcome-level supervision and explicitly mode flexible transitions between reasoning states.

## 3 METHOD

In this section, we present RSP, including its reasoning state propagation formulation, model parameterization, and learning objectives.

## 3.1 PRELIMINARIES

Task Definition Given a question $x ,$ an LRM policy π generates a reasoning trajectory $\begin{array} { r l } { \mathbf { s } } & { { } = } \end{array}$ $( s _ { 1 } , \ldots , s _ { T } )$ autoregressively, where $s _ { t } \ \sim \ \pi ( \cdot \ | \ x , \mathbf { s } _ { < t } )$ and $\mathbf { s } _ { < t } ~ = ~ \left( s _ { 1 } , \ldots , s _ { t - 1 } \right)$ denotes the preceding reasoning prefix. We consider two types of supervision. For process-annotated trajectories, binary process labels $y _ { t } ^ { \mathrm { s t e p } } \in \{ 0 , 1 \}$ are available for each step, where $y _ { t } ^ { \mathrm { s t e p } } = 1$ indicates that the reasoning prefix $\mathbf { s } _ { < t }$ is valid after incorporating step $s _ { t }$ . For outcome-annotated trajectories, only a final label $\bar { y } ^ { \mathrm { o u t } } \in \{ \bar { 0 } , 1 \}$ is available, indicating whether the complete trajectory reaches a correct answer. Let $N _ { \mathrm { s t e p } }$ and $N _ { \mathrm { o u t } }$ denote the numbers of process-annotated and outcome-annotated trajectories. For the i-th trajectory, $T _ { i }$ denotes its number of reasoning steps, and $\mathcal { A } _ { i } \subseteq \{ 1 , \dots , T _ { i } \}$ denotes the set of steps with available process annotations. The two training sets are defined as $\mathcal { D } _ { \mathrm { s t e p } } = \left\{ \left( x _ { i } , \mathbf { s } _ { i } , \{ y _ { i , t } ^ { \mathrm { s t e p } } \} _ { t \in \mathcal { A } _ { i } } \right) \right\} _ { i = 1 } ^ { N _ { \mathrm { s t e p } } }$ , and $\mathcal { D } _ { \mathrm { o u t } } = \{ ( x _ { i } , \mathbf { s } _ { i } , y _ { i } ^ { \mathrm { o u t } } ) \} _ { i = 1 } ^ { N _ { \mathrm { o u t } } }$ . For clarity, we omit the trajectory index i when discussing a single trajectory. Our goal is to learn a PRM that estimates the state validity of each reasoning prefix $\mathbf { s } _ { \leq t }$ while jointly leveraging process and outcome supervision.

Independent Prefix Classification A common discriminative PRM formulation (Lightman et al., 2024; Zhang et al., 2025b) treats each reasoning prefix as an independent binary classification target. For the i-th trajectory, a PRM with parameters θ predicts $q _ { i , t } = p _ { \theta } \left( y _ { i , t } ^ { \mathrm { s t e p } } = 1 \mid x _ { i } , \mathbf { s } _ { i , \le t } \right)$ , where ${ { q } _ { i , t } }$ denotes the predicted validity of the reasoning prefix after step $s _ { i , t } .$ . The model is trained on the available process annotations using the binary cross-entropy objective

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { \sum _ { i = 1 } ^ { N _ { \mathrm { s t e p } } } \left| A _ { i } \right| } \sum _ { i = 1 } ^ { N _ { \mathrm { s t e p } } } \sum _ { t \in A _ { i } } \left[ y _ { i , t } ^ { \mathrm { s t e p } } \log q _ { i , t } + \left( 1 - y _ { i , t } ^ { \mathrm { s t e p } } \right) \log \left( 1 - q _ { i , t } \right) \right] .\tag{1}
$$

Although each prediction is conditioned on the full reasoning prefix, the objective decomposes over individual prefixes and imposes no explicit relation between their predicted validity across successive steps. Moreover, this objective relies exclusively on process annotations and cannot naturally incorporate outcome-only trajectories.

## 3.2 REASONING STATE PROPAGATION

To explicitly relate reasoning states across successive steps, we represent the state of each reasoning prefix as $z _ { t } \in \mathcal { Z } = \{ \mathrm { G } , \mathrm { B } \}$ . Here, G denotes that the reasoning currently maintained after step $s _ { t }$ is valid and contains no unresolved error, whereas B denotes that an error remains unresolved in the current reasoning state. For each reasoning step $s _ { t } .$ , we parameterize two probabilities that describe its effect on the preceding reasoning state:

$$
\alpha _ { t } : = p _ { \theta } \left( z _ { t } = \mathrm { B } \mid z _ { t - 1 } = \mathrm { G } , x , \mathbf { s } _ { < t } , s _ { t } \right) ,\tag{2}
$$

$$
\beta _ { t } : = p _ { \theta } \left( z _ { t } = \operatorname { G } \ | \ z _ { t - 1 } = \operatorname { B } , x , \mathbf { s } _ { < t } , s _ { t } \right) .\tag{3}
$$

Here, $\alpha _ { t }$ denotes the break probability that step $s _ { t }$ introduces an unresolved error into a previously valid reasoning state, and $\beta _ { t }$ denotes the repair probability that step $s _ { t }$ resolves or supersedes an earlier error, returning the current reasoning state to a valid one. The break and repair probabilities define a local state propagation matrix for step $s _ { t } \colon$

$$
\mathbf { M } _ { t } = \left[ \begin{array} { c c } { 1 - \alpha _ { t } } & { \alpha _ { t } } \\ { \beta _ { t } } & { 1 - \beta _ { t } } \end{array} \right] .\tag{4}
$$

The rows correspond to the preceding state $z _ { t - 1 }$ , while the columns correspond to the updated state $z _ { t } , \ \mathbf { M } _ { t }$ describes a one-step propagation between adjacent reasoning states, conditioned on the preceding reasoning prefix and the newly incorporated step $s _ { t }$ . Let $\begin{array} { r } { \mathbf { { \bar { p } } } _ { t } ~ = ~ \left[ p _ { t } ^ { \mathrm { G } } ~ p _ { t } ^ { \mathrm { B } } \right] } \end{array}$ denote the probability distribution over reasoning states after step $s _ { t }$ . Before any reasoning step, we initialize the state as $\mathbf { p } _ { 0 } = [ 1 , 0 ]$ , and propagate it as

$$
\mathbf { p } _ { t } = \mathbf { p } _ { t - 1 } \mathbf { M } _ { t } = \mathbf { p } _ { 0 } \prod _ { j = 1 } ^ { t } \mathbf { M } _ { j } .\tag{5}
$$

We use $p _ { t } ^ { \mathrm { G } }$ as the process score for the reasoning prefix $\mathbf { s } _ { \leq t }$ . Unlike the independently predicted score $q _ { t } , p _ { t } ^ { \mathrm { G } }$ depends on all preceding state updates through sequential propagation. Here, $\mathrm { G }  \mathrm { G }$ preserves valid reasoning, $\mathrm { G }  \mathrm { B }$ captures error introduction, $\mathrm { B } \to \mathrm { B }$ captures error persistence, and $\mathrm { \bar { B } }  \mathrm { G }$ captures error recovery. The resulting state sequence therefore explicitly links reasoning states across successive steps.

Transition Parameterization To estimate the break and repair probabilities, we insert two special tokens, ⟨BREAK⟩ and ⟨REPAIR⟩, at each reasoning boundary. Boundary 0 is placed after the question x, and boundary t is placed after reasoning step $s _ { t } .$ . Specifically, at each boundary t, we insert a pair of update tokens $\mathbf { u } _ { t } = ( \langle \mathrm { B R E A K } \rangle _ { t } .$ ⟨REPAIR⟩ ), yielding

$$
\widetilde { \mathbf { s } } = ( x , \mathbf { u } _ { 0 } , s _ { 1 } , \mathbf { u } _ { 1 } , \dots , s _ { T } , \mathbf { u } _ { T } ) .
$$

Our PRM is built on a pretrained LRM backbone $f _ { \phi }$ , which encodes $\widetilde { \mathbf { s } }$ in a single forward pass. Here, ϕ denotes the backbone parameters, including the embeddings of the newly introduced special tokens. Let $\mathbf { h } _ { t } ^ { \mathrm { b r } }$ and ${ \bf h } _ { t } ^ { \mathrm { r p } }$ denote the hidden representations of ⟨BREAK⟩<sub>t</sub> and $\langle \mathrm { R E P A I } \bar { \mathrm { R } } \rangle _ { t }$ at boundary t. Since boundary t follows step $s _ { t }$ , these representations encode the reasoning prefix $( x , { \mathbf { s } } _ { \leq t } )$ . To characterize the state update induced by step $s _ { t } ,$ , we use the representations at the boundaries before and after $s _ { t }$ . The break and repair probabilities are computed as

$$
\alpha _ { t } = \sigma \left( g _ { \mathrm { b r } } \left( \mathbf { h } _ { t - 1 } ^ { \mathrm { b r } } \mid \mid \mathbf { h } _ { t } ^ { \mathrm { b r } } \right) \right) ,\tag{6}
$$

$$
\beta _ { t } = \sigma \left( g _ { \mathrm { r p } } \left( \mathbf { h } _ { t - 1 } ^ { \mathrm { r p } } \parallel \mathbf { h } _ { t } ^ { \mathrm { r p } } \right) \right) ,\tag{7}
$$

where ∥ denotes the vector concatenation. $g _ { \mathrm { b r } } ( \cdot )$ and $g _ { \mathrm { r p } } ( \cdot )$ are two independent MLP heads, and $\sigma ( \cdot )$ denotes the sigmoid function. All break and repair probabilities along the trajectory are obtained from a single forward pass. We denote all trainable parameters of RSP by θ.

## 3.3 LEARNING FROM PROCESS AND OUTCOME SUPERVISION

The propagated reasoning states enable RSP to naturally incorporate both process and outcome supervision. We define the corresponding learning objectives below.

For each process-annotated trajectory in $\mathcal { D } _ { \mathrm { s t e p } } ,$ we supervise the propagated state at every annotated step $t \in A _ { i }$ . Since $y _ { i , t } ^ { \mathrm { s t e p } } = 1$ indicates a valid reasoning step, we define

$$
\mathcal { L } _ { \mathrm { s t e p } } = - \frac { 1 } { \sum _ { i = 1 } ^ { N _ { \mathrm { s t e p } } } \left| A _ { i } \right| } \sum _ { i = 1 } ^ { N _ { \mathrm { s t e p } } } \sum _ { t \in A _ { i } } \left[ y _ { i , t } ^ { \mathrm { s t e p } } \log p _ { i , t } ^ { \mathrm { G } } + \left( 1 - y _ { i , t } ^ { \mathrm { s t e p } } \right) \log p _ { i , t } ^ { \mathrm { B } } \right] .\tag{8}
$$

Unlike BCE in equation 1 applied to independently predicted scores $q _ { i , t } ,$ , this loss supervises state probabilities that depend on the preceding reasoning states through sequential propagation.

For each outcome-annotated trajectory in $\mathcal { D } _ { \mathrm { o u t } }$ , we define

$$
\mathcal { L } _ { \mathrm { o u t } } = - \frac { 1 } { N _ { \mathrm { o u t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o u t } } } \left[ y _ { i } ^ { \mathrm { o u t } } \log p _ { i , T _ { i } } ^ { \mathrm { G } } + \left( 1 - y _ { i } ^ { \mathrm { o u t } } \right) \log p _ { i , T _ { i } } ^ { \mathrm { B } } \right] .\tag{9}
$$

Because the final state $\mathbf { p } _ { i , T _ { i } }$ depends on all state updates along the trajectory, the outcome loss is backpropagated through the propagation process and therefore trains the break and repair predictions at intermediate steps. We jointly optimize the two supervision signals as $\mathcal { L } _ { \mathrm { t o t a l } } = \bar { \mathcal { L } } _ { \mathrm { s t e p } } \bar { + } \mathcal { L } _ { \mathrm { o u t } }$

Effect of Outcome Supervision on State Transitions Process labels directly supervise intermediate states, whereas outcome supervision reaches earlier transitions through state propagation. For an outcome-annotated trajectory, let $c \in \{ \mathrm { G } , \mathrm { B } \}$ denote the terminal target and define $h _ { t } ( a ) = p _ { \theta } ( z _ { T } = c \mid z _ { t } = a , x , \mathbf { s } )$ as the likelihood of reaching c from state a through the remaining reasoning. Let $Z _ { \mathrm { o u t } } = p _ { T } ^ { c }$ and $\ell ^ { \mathrm { { o u t } } } = - \log Z _ { \mathrm { { o u t } } }$ . Then, with respect to the pre-sigmoid logits of $\alpha _ { t }$ and $\beta _ { t } .$

$$
\frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \log \mathrm { i t } ( \alpha _ { t } ) } = \frac { p _ { t - 1 } ^ { \mathrm { G } } \alpha _ { t } ( 1 - \alpha _ { t } ) } { Z _ { \mathrm { o u t } } } \big ( h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) \big ) , \quad \frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \log \mathrm { i t } ( \beta _ { t } ) } = \frac { p _ { t - 1 } ^ { \mathrm { B } } \beta _ { t } ( 1 - \beta _ { t } ) } { Z _ { \mathrm { o u t } } } \big ( h _ { t } ( \mathrm { B } ) - h _ { t } ( \mathrm { G } ) \big ) .
$$

These gradients show how the final outcome shapes intermediate state transitions. If $h _ { t } ( \mathrm { G } ) > h _ { t } ( \mathrm { B } )$ the outcome loss favors transitions toward G by decreasing break and increasing repair, with the opposite effect when $h _ { t } ( \mathrm { B } ) > h _ { t } ( \mathrm { G } )$ . Detailed analysis is provided in Appendix C.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Model Architecture We build RSP on the Qwen3 family, using Qwen3-4B as the default backbone unless otherwise specified. Following the formulation of RSP, we use two independent MLP heads to predict the break and repair probabilities. The backbone parameters, the embeddings of the ⟨BREAK⟩ and ⟨REPAIR⟩ tokens, and the two MLP heads are optimized during training.

Training Data We train RSP using both process and outcome supervision. For process supervision, we use PRM800K (Lightman et al., 2024), which contains human process annotations for trajectories. Unless otherwise specified, we randomly select 20% of PRM800K as the process-annotated training set. For outcome supervision, we use the filtered correctness labels from AceMath-RM (Liu et al., 2025b) as outcome supervision for the terminal reasoning state. AceMath-RM ranks and filters candidate responses with an LLM judge to retain high-confidence examples, reducing cases where final answer correctness is inconsistent with the validity of the terminal reasoning state. Each batch contains process-annotated and outcome-annotated examples at a default ratio of 1:3. We train models for two epochs, where one epoch corresponds to a complete pass over the process-annotated set, and outcome-annotated examples are sampled to maintain the ratio.

Baselines We compare RSP with representative process- and outcome-supervised reward models. Supervised PRM (Lightman et al., 2024) independently predicts the validity of each reasoning prefix using process annotations. OVM (Yu et al., 2024) uses outcome supervision by assigning the outcome as the target for each reasoning step. We additionally compare with Qwen2.5-Math-PRM (Zhang et al., 2025b), using its 7B checkpoint by default, and CRM (Zhang et al., 2025a) as another representative PRM. To compare different ways of combining process and outcome supervision, we construct additional baselines: Pseudo-Label PRM, Joint-Supervised PRM, and CRM<sup>†</sup>. Pseudo-Label PRM follows a standard semi-supervised strategy (Sohn et al., 2020), using highconfidence predictions as pseudo process labels. Joint-Supervised PRM extends Supervised PRM by additionally supervising the prediction at the final reasoning prefix with the outcome label using BCE loss. CRM<sup>†</sup> extends CRM with an additional outcome objective, allowing it to incorporate both process and outcome supervision. More details are provided in Appendix A.

## 4.2 PROCESS-GUIDED BEAM SEARCH

We evaluate whether process rewards can reliably rank intermediate reasoning prefixes in beam search. We consider PRMs built on Qwen3-1.7B and Qwen3-4B and evaluate them on MATH500 (Lightman et al., 2024) and Gaokao (Liao et al., 2024). The policy model is fixed to Qwen3-1.7B for all experiments. We fix the beam width to $b = 4$ and vary the expansion width $e \in \{ 4 , 8 , 1 2 \}$ . At each search step, each of the b retained prefixes is expanded into e candidate continuations. The resulting $b \times e$ candidates are ranked by the PRM, and the top-b prefixes are retained for the next step.

Analysis Table 1 shows that RSP achieves the strongest overall beam search performance across both Qwen3 PRM backbones and expansion widths. The advantage is particularly pronounced with the Qwen3-4B backbone at larger expansion widths. At e = 12, RSP outperforms OVM by 4.9% on both MATH500 and Gaokao. OVM remains competitive, suggesting that outcome supervision alone can provide useful signals for prefix ranking. In contrast, Pseudo-Label PRM performs substantially worse and exhibits larger variation across expansion widths, particularly on Gaokao. Consistent gains over Joint-Supervised PRM and CRM<sup>†</sup> further show the benefit of reasoning state propagation when jointly leveraging process and outcome supervision.

## 4.3 BEST-OF-N SELECTION

Best-of-N evaluates whether a reward model can select a correct response from multiple sampled candidates. For each generator and question, we sample N responses and select the response with the highest final score. We consider $N \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and report the average accuracy in these settings. All evaluations are conducted on MATH500. For PRMs, we use the process score of the final reasoning step as the response score. We consider six generators from the LLaMA, Qwen, and DeepSeek model families.

Table 1: Beam search accuracy (%) on MATH500 and Gaokao. The beam width is fixed to 4, and e denotes the expansion width. The best and second-best results are shown in bold and underlined.
<table><tr><td rowspan="2">PRM Backbone</td><td rowspan="2">Reward Model</td><td colspan="3">MATH500</td><td colspan="3">Gaokao</td></tr><tr><td> $e = 4$ </td><td> $e = 8$ </td><td> $e = 1 2$ </td><td> $e = 4$ </td><td> $e = 8$ </td><td> $e = 1 2$ </td></tr><tr><td>Qwen2.5-Math-7B</td><td>Qwen2.5-Math-PRM</td><td>74.8</td><td>74.0</td><td>75.8</td><td>58.5</td><td>55.6</td><td>59.3</td></tr><tr><td rowspan="6">Qwen3-1.7B</td><td>Supervised PRM</td><td>54.8</td><td>53.6</td><td>55.4</td><td>44.0</td><td>45.7</td><td>48.3</td></tr><tr><td>OVM</td><td>67.9</td><td>68.4</td><td>69.5</td><td>54.5</td><td>58.3</td><td>58.3</td></tr><tr><td>Pseudo-Label PRM</td><td>65.3</td><td>64.8</td><td>67.7</td><td>53.1</td><td>52.5</td><td>55.0</td></tr><tr><td>Joint-Supervised PRM</td><td>67.4</td><td>64.6</td><td>64.0</td><td>54.4</td><td>57.2</td><td>55.9</td></tr><tr><td>CRM†</td><td>62.2</td><td>66.2</td><td>63.4</td><td>55.6</td><td>53.5</td><td>56.6</td></tr><tr><td>RSP</td><td>68.8</td><td>70.5</td><td>70.0</td><td>55.6</td><td>58.5</td><td>59.3</td></tr><tr><td rowspan="6">Qwen3-4B</td><td>Supervised PRM</td><td>62.3</td><td>59.8</td><td>61.4</td><td>52.6</td><td>53.3</td><td>52.7</td></tr><tr><td>OVM</td><td>73.1</td><td>75.2</td><td>74.9</td><td>61.7</td><td>64.8</td><td>63.0</td></tr><tr><td>Pseudo-Label PRM</td><td>68.2</td><td>64.4</td><td>65.7</td><td>56.4</td><td>49.9</td><td>53.4</td></tr><tr><td>Joint-Supervised PRM</td><td>75.4</td><td>75.4</td><td>75.6</td><td>62.9</td><td>63.2</td><td>61.1</td></tr><tr><td>CRM†</td><td>68.5</td><td>68.2</td><td>69.9</td><td>55.3</td><td>55.3</td><td>56.8</td></tr><tr><td>RSP</td><td>76.0</td><td>78.6</td><td>79.8</td><td>63.5</td><td>65.8</td><td>67.9</td></tr></table>

Table 2: Average Best-of-N accuracy (%) over $N \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ . Proc./Out. denote process/outcome supervision, and $\checkmark ^ { + }$ indicates additional large-scale process supervision.
<table><tr><td></td><td></td><td>Out.</td><td>LLaMA3 70B</td><td>LLaMA3.3 70B</td><td>Qwen3  $1 . 7 8$ </td><td>Qwen3 4B</td><td>Qwen3 8B</td><td>DeepSeek-R1-0528 Qwen3-8B</td><td>Avg.</td></tr><tr><td>Method</td><td>Proc.</td><td></td><td></td><td></td><td></td><td>77.4</td><td>79.2</td><td></td><td>64.4</td></tr><tr><td>Pass@1</td><td>一 √</td><td>一</td><td>43.6 57.2</td><td>69.0 74.6</td><td>65.8 76.0</td><td>80.8</td><td>79.9</td><td>51.6 46.0</td><td>69.1</td></tr><tr><td>Supervised PRM OVM</td><td></td><td>一 √</td><td>60.5</td><td>75.5</td><td>77.9</td><td>84.1</td><td>83.2</td><td>62.4</td><td>73.9</td></tr><tr><td>CRM</td><td>一 √</td><td></td><td>58.2</td><td>77.8</td><td>79.1</td><td>83.1</td><td>82.5</td><td>52.1</td><td>72.1</td></tr><tr><td>Qwen2.5-Math-PRM</td><td>√+</td><td>一 一</td><td>60.6</td><td>77.1</td><td>79.9</td><td>83.5</td><td>82.4</td><td>61.0</td><td>74.1</td></tr><tr><td>Pseudo-Label PRM</td><td>√</td><td>√</td><td>43.6</td><td>69.2</td><td>65.8</td><td>77.6</td><td>80.2</td><td>59.9</td><td>66.1</td></tr><tr><td>Joint-Supervised PRM</td><td>√</td><td>√</td><td>60.7</td><td>77.1</td><td>78.8</td><td>82.4</td><td>82.9</td><td>59.9</td><td>73.6</td></tr><tr><td>CRM†</td><td>√</td><td>√</td><td>60.6</td><td>77.4</td><td>78.6</td><td>82.0</td><td>81.9</td><td>60.0</td><td>73.4</td></tr><tr><td>RSP</td><td>√</td><td>√</td><td>61.2</td><td>76.1</td><td>78.3</td><td>84.1</td><td>83.3</td><td>63.3</td><td>74.4</td></tr></table>

Analysis Table 2 shows that RSP achieves the highest average Best-of-N accuracy across the six generators. In particular, RSP slightly exceeds Qwen2.5-Math-PRM in average accuracy despite the latter using additional large-scale process supervision. Adding outcome supervision to CRM improves its average accuracy from 72.1% to 73.4%, while RSP further improves it to 74.4%. In contrast, Pseudo-Label PRM performs substantially worse, indicating that pseudo process labels provide a less effective way to combine the two supervision sources. Joint-Supervised PRM is also competitive but remains below RSP, showing that it does not fully exploit the two supervision signals. The results remain competitive across LLaMA, Qwen, and DeepSeek generators, indicating that the benefit of RSP is not limited to a single generator family.

## 4.4 REINFORCEMENT LEARNING WITH PROCESS REWARDS

We further evaluate whether process rewards can provide effective fine-grained feedback for policy optimization. Following the verifier-gated strategy of VeriGate (Agrawal et al., 2026), we use PRM feedback only when all sampled responses for a prompt receive zero verifier reward, while retaining standard verifier-based GRPO updates otherwise. Unlike VeriGate, we directly use the PRM score at each reasoning step as the step-level reward without aggregating rewards from subsequent steps. We train on 8K prompts from DAPO-MATH-17K (Yu et al., 2026) and evaluate the resulting policy using zero-shot Avg@16 accuracy on six mathematical reasoning benchmarks.

Table 3: Avg@16 accuracy (%) after reinforcement learning. All methods use the same policy initialization, training data, and optimization budget.
<table><tr><td>Method</td><td>AIME</td><td>AMC</td><td>GSM8K</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Base Policy</td><td>23.3</td><td>70.0</td><td>92.5</td><td>78.6</td><td>38.2</td><td>52.4</td><td>59.2</td></tr><tr><td>GRPO</td><td>23.3</td><td>72.3</td><td>94.2</td><td>88.2</td><td>40.2</td><td>59.6</td><td>63.0</td></tr><tr><td>Supervised PRM</td><td>16.7</td><td>68.3</td><td>92.3</td><td>83.4</td><td>38.2</td><td>54.8</td><td>59.0</td></tr><tr><td>OVM</td><td>31.1</td><td>71.3</td><td>92.6</td><td>86.2</td><td>37.6</td><td>58.4</td><td>62.9</td></tr><tr><td>Qwen2.5-Math-PRM</td><td>31.1</td><td>68.3</td><td>93.8</td><td>86.8</td><td>39.8</td><td>59.8</td><td>63.3</td></tr><tr><td>Pseudo-Label PRM</td><td>26.7</td><td>72.3</td><td>92.6</td><td>85.8</td><td>35.8</td><td>57.3</td><td>61.8</td></tr><tr><td>Joint-Supervised PRM</td><td>31.1</td><td>70.0</td><td>92.3</td><td>87.2</td><td>38.2</td><td>59.8</td><td>63.1</td></tr><tr><td>CRM†</td><td>23.3</td><td>70.0</td><td>92.5</td><td>86.2</td><td>37.0</td><td>57.3</td><td>61.1</td></tr><tr><td>RSP</td><td>38.9</td><td>72.5</td><td>92.9</td><td>88.4</td><td>39.3</td><td>60.2</td><td>65.4</td></tr></table>

Analysis Table 3 shows that RSP achieves the strongest overall performance, improving the average accuracy by 2.1% over PRM-guided with Qwen2.5-Math-PRM. The gain is particularly pronounced on AIME, where RSP outperforms Qwen2.5-Math-PRM by 7.8%. Supervised PRM reduces the average accuracy below that of the base policy, while OVM, Pseudo-Label PRM, and CRM<sup>†</sup> all remain below GRPO on average. RSP is the only PRM-guided method that substantially improves the average accuracy over GRPO, indicating that introducing process rewards alone does not necessarily benefit policy optimization and that the quality of the process signal is critical. These results highlight the benefit of process rewards from RSP for policy optimization.

## 4.5 ANALYSIS OF REASONING STATE PROPAGATION

Transition Predictions We examine whether the predicted transitions reflect meaningful changes in reasoning states. In particular, we compare the score distributions under the four state update patterns $\mathrm { G \bar {  } G , G  \bar { B } , B  B , }$ and $\mathrm { B } \to \mathrm { G }$ . Publicly available process-annotated data contain relatively few examples after reasoning has entered an invalid state, making B→B and B →G cases particularly scarce. We therefore use Qwen3-32B as an LLM judge to determine the validity of each reasoning step and construct approximately 10K step pairs covering the four state update patterns. As shown in Figure 2, for break predictions, G → G cases are concentrated at low scores, whereas G → B cases receive substantially higher scores. Similarly, repair scores are generally higher for B → G than for B → B. The overall separation indicates that the learned transition scores capture meaningful changes in reasoning states.

Scaling with Outcome Supervision We further study how RSP benefits from increasing the amount of outcome supervision while keeping process supervision fixed. We vary the processto-outcome data ratio from 1:1 to 1:9 and report the performance gain over the corresponding Process-Only model. Figure 2(c) shows that adding outcome supervision provides clear gains over Process-Only for both beam search and reinforcement learning. The gains generally increase as more outcome-annotated samples are introduced, showing that RSP can effectively benefit from additional outcome supervision for reasoning search and policy optimization.

## 4.6 ABLATION STUDY

Table 4 examines the contributions of the two supervision and the main design choices of RSP. Except for the results reported in Table 1, all subsequent beam search results use the Qwen3-4B PRM backbone with b = 4 and $e = 8 ,$ , and report the average accuracy on MATH500 and Gaokao.

Supervision Sources Process-Only and Outcome-Only use only process and outcome supervision, respectively. Process-Only shows large drops on beam search and ProcessBench (Zheng et al., 2024), whereas Outcome-Only nearly preserves Best-of-N performance but drops by 9.6% on ProcessBench. This contrast suggests that outcome supervision provides strong signals for response selection and search, while process annotations remain important for fine-grained process evaluation. Combining the two sources of supervision gives the best performance in the evaluated settings.

![](images/70cbbe974f0a273d6c8aff4bdb734b0d22dd929b9814694014cfe2188fb854c2.jpg)  
(a) Break prediction

![](images/8a9476473e1c43d002b734ed0fddab1a8d837e457bf36b09cb723f56970ce755.jpg)  
(b) Repair prediction

![](images/582d4c2bc43082c418b0257fe5a67d3c91adb7144fac8fd1e66b0a487c089cfc.jpg)  
(c) Outcome supervision scaling  
Figure 2: (a) Distributions of predicted break scores for G → G and G → B. (b) Distributions of predicted repair scores for B → B and B → G. (c) Performance gains over Process-Only under increasing amounts of outcome supervision for beam search and reinforcement learning.

Table 4: Ablation study across evaluation settings. Colored numbers indicate changes relative to Ours, with darker colors denoting larger degradation.
<table><tr><td>Variant</td><td>BoN</td><td>Beam Search</td><td>ProcessBench</td><td>RL</td></tr><tr><td>Process-Only</td><td>72.0 (-2.4)</td><td>59.7 (-12.5)</td><td>58.3 (-7.5)</td><td>59.8 (-5.6)</td></tr><tr><td>Outcome-Only</td><td>74.2 (-0.2)</td><td>69.5 (-2.7)</td><td>56.2 (-9.6)</td><td>61.6 (-3.8)</td></tr><tr><td>No Repair</td><td>73.6 (-0.8)</td><td>68.9 (-3.3)</td><td>64.5 (-1.3)</td><td>63.2 (-2.2)</td></tr><tr><td>Current Representation Only</td><td>73.5 (-0.9)</td><td>70.1 (-2.1)</td><td>64.2 (-1.6)</td><td>62.3 (-3.1)</td></tr><tr><td>Shared Token</td><td>73.5 (-0.9)</td><td>69.7 (-2.5)</td><td>63.5 (-2.3)</td><td>63.8 (-1.6)</td></tr><tr><td>No Outcome Propagation</td><td>73.9 (-0.5)</td><td>65.5 (-6.7)</td><td>61.8 (-4.0)</td><td>64.2 (-1.2)</td></tr><tr><td>RSP</td><td>74.4</td><td>72.2</td><td>65.8</td><td>65.4</td></tr></table>

Repair Modeling No Repair sets the repair probability to zero during training and inference, so an invalid reasoning state cannot return to a valid state. Removing repair degrades performance across all four settings, with the largest drop of 3.3% on beam search. This result shows that explicitly modeling error recovery is important for reasoning state propagation.

Adjacent Representations Current Representation Only predicts the break and repair probabilities using only the representation at the current reasoning boundary, rather than the representations before and after the reasoning step. The consistent degradation shows that using both boundary representations provides more informative signal for estimating how the step updates reasoning states.

Separate Break and Repair Tokens Shared Token replaces the separate ⟨BREAK⟩ and ⟨REPAIR⟩ tokens with a single shared token. The resulting performance drops across all four settings, particularly 2.5% drop on beam search and 2.3% on ProcessBench, indicating that separate token representations help the model distinguish the roles of break and repair, while sharing a single token may introduce semantic interference between two conceptually different state transitions.

Outcome Propagation We further examine whether outcome supervision benefits intermediate state learning through the propagation mechanism. Specifically, we preserve the same forward state propagation while blocking backpropagation from the terminal outcome loss to earlier state transitions. This variant consistently degrades performance, with particularly large drops of 6.7% on beam search and 4.0% on ProcessBench. Since the forward propagation remains unchanged, the results show that propagating outcome supervision to earlier state updates is important for learning effective process rewards.

## 5 CONCLUSION

We introduced RSP, a reasoning state propagation approach for process reward modeling. By modeling break and repair probabilities, RSP explicitly links reasoning states across successive steps and captures error introduction, persistence, and recovery. The propagated reasoning state also provides a natural way to combine scarce process supervision with scalable outcome supervision, allowing terminal outcome supervision to provide learning signals to earlier state updates. Our results demonstrate the effectiveness of this formulation and highlight the complementary roles of process and outcome supervision in learning reliable process rewards.

## AI USE STATEMENT

We used generative AI tools primarily to assist with language editing and improving the clarity and presentation of the manuscript. Generative AI was also used in a limited experimental analysis, where Qwen3-32B served as an LLM judge as described in Section 4.5. All AI-assisted content and outputs were reviewed and verified by the authors, who take full responsibility for the final content of the paper.

## REPRODUCIBILITY STATEMENT

We provide the main implementation and experimental details in Section 4 and Appendix A, including training data, hyperparameters, evaluation settings, and baseline implementations. Code is provided in the supplementary material.

## REFERENCES

Aakriti Agrawal, Minghui Liu, and Furong Huang. Verigate: Verifier-gated step-level supervision for grpo. arXiv preprint arXiv:2605.30451, 2026.

Jie Cheng, Gang Xiong, Ruixi Qiao, Lijun Li, Chao Guo, Junle Wang, Yisheng Lv, and Fei-Yue Wang. Stop summation: Min-form credit assignment is all process reward model needs for reasoning. Advances in Neural Information Processing Systems, 38:131646–131671, 2026.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jinu Lee, Shivam Agarwal, Amruta Parulekar, Siddarth Madala, Dilek Hakkani-Tur, and Julia Hockenmaier. Reasoningflow: Discourse structures for understanding llm reasoning traces. arXiv preprint arXiv:2606.05402, 2026.

Weijie Li, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. Step-grpo: Enhancing reasoning quality and efficiency via structured prm-based reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 31734–31742, 2026.

Wendi Li and Yixuan Li. Process reward model with q-value rankings. In International Conference on Learning Representations, volume 2025, pp. 14708–14726, 2025.

Minpeng Liao, Chengxi Li, Wei Luo, Wu Jing, and Kai Fan. Mario: Math reasoning with code interpreter output-a reproducible pipeline. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 905–924, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025a.

Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling, 2025b.

Jianqiao Lu, Zhiyang Dou, Hongru Wang, Zeyu Cao, Jianbo Dai, Yingjia Wan, Yunlong Feng, and Zhijiang Guo. Autopsv: Automated process-supervised verifier. Advances in Neural Information Processing Systems, 37:79935–79962, 2024.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.

ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shuaijie She, Junxiao Liu, Yifeng Liu, Jiajun Chen, Xin Huang, and Shujian Huang. R-prm: Reasoning-driven process reward modeling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13438–13451, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations, volume 2025, pp. 10131–10165, 2025.

Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33: 596–608, 2020.

Xingwei Tan, Marco Valentino, Mahmud Elahi Akhter, Maria Liakata, and Nikolaos Aletras. Enhancing logical reasoning in language models via symbolically-guided monte carlo process supervision. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 31874–31888, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, 2024.

LLM Xiaomi, Bingquan Xia, Bowen Shen, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, et al. Mimo: Unlocking the reasoning potential of language model–from pretraining to posttraining. arXiv preprint arXiv:2505.07608, 2025.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. Self-evaluation guided beam search for reasoning. Advances in Neural Information Processing Systems, 36:41618–41650, 2023.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, and Jing Tang. Treerpo: Tree relative policy optimization. arXiv preprint arXiv:2506.05183, 2025b.

Zhangyue Yin, Qiushi Sun, Zhiyuan Zeng, Qinyuan Cheng, Xipeng Qiu, and Xuan-Jing Huang. Dynamic and generalizable process reward modeling. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4203–4233, 2025.

Fei Yu, Anningzhe Gao, and Benyou Wang. Ovm, outcome-supervised value models for planning in mathematical reasoning. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 858–875, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts\*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 37:64735–64772, 2024.

Zheng Zhang, Ziwei Shan, Kaitao Song, Yexin Li, and Kan Ren. Linking process to outcome: Conditional reward modeling for llm reasoning. In International Conference on Learning Representations, volume 2026, pp. 87995–88016, 2025a.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 10495– 10516, 2025b.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Processbench: Identifying process errors in mathematical rea soning. arXiv preprint arXiv:2412.06559, 2024.

Jiachen Zhu, Congmin Zheng, Jianghao Lin, Kounianhua Du, Ying Wen, Yong Yu, Jun Wang, and Weinan Zhang. Retrieval-augmented process reward model for generalizable mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 8453– 8468, 2025.

## A ADDITIONAL EXPERIMENTAL DETAILS

## A.1 RSP TRAINING DETAILS

Unless otherwise specified, we initialize RSP from Qwen3-4B and jointly optimize the backbone, the embeddings of the two special tokens, and the break and repair MLP heads. Following Section 3, each head receives the concatenation of the boundary representations before and after a reasoning step. We randomly select 20% of PRM800K and mix its process-annotated trajectories with outcome-annotated trajectories from AceMath-RM at a ratio of 1:3. Training proceeds for two epochs over the selected PRM800K subset using AdamW with a learning rate of $5 \times 1 0 ^ { - 6 }$ We optimize only the process and outcome objectives in Section 3. Training is conducted in bfloat16 across eight NVIDIA H100 GPUs. We use the final checkpoint for evaluation. The code is provided in supplementary material.

Transition Heads. The break and repair predictors use two independent MLP heads with the same architecture. Given backbone hidden size d, each head takes the concatenated adjacent boundary representations in $\mathbb { R } ^ { 2 d }$ and applies a two-layer MLP, $2 d  d  1$ , with GELU activation and dropout of 0.1 after the input and hidden layer. The resulting scalar logits are converted to break and repair probabilities using a sigmoid.

## A.2 REINFORCEMENT LEARNING DETAILS

We initialize the policy from Qwen3-4B and perform full parameter training for one epoch on the 8K subset of DAPO-MATH-17K. For each prompt, we sample eight responses with a temperature of 1.0 and top-p of 1.0. We use a rollout batch size of 16 and a training batch size of 128, with corresponding micro-batch sizes of 4 and 2. The maximum prompt and generation lengths are 512 and 4096 tokens, respectively. The actor learning rate is $\mathrm { i ^ { \circ } \times 1 \bar { 0 } ^ { - 6 } }$ with a cosine schedule, and training is performed in bfloat16. During training, if all responses receive zero reward, we use RSP to score reasoning steps in each response. The step scores are normalized within the sampled group and assigned to the tokens of their corresponding steps, without accumulating scores from subsequent steps. Otherwise, the update uses only the outcome rewards. We use the Dr. GRPO (Liu et al., 2025a) advantage estimator with a zero KL coefficient. All RL methods use the same policy training settings.

## A.3 IMPLEMENTATION DETAILS OF ADDITIONAL BASELINES

For a fair comparison, Pseudo-Label PRM, Joint-Supervised PRM, and CRM<sup>†</sup> use the same backbone, training data mixture, and optimization settings as described above. Each baseline uses a single scalar step head evaluated at a [PRM] token appended after each reasoning step.

Pseudo-Label PRM For process-annotated trajectories, we apply BCE loss to all steps with available binary process labels. For outcome-annotated trajectories, the intermediate steps are treated as unlabeled, and hard pseudo labels are generated online from the current step predictions. A step is assigned a positive pseudo label when its predicted probability is at least 0.9 and a negative pseudo label when the probability is at most 0.1. The predictions between the two thresholds are ignored. Pseudo-labeling is enabled from the beginning of training.

Joint-Supervised PRM We use the same BCE loss as Supervised PRM on process-annotated trajectories. For each outcome-annotated trajectory, we additionally align the prediction at its final [PRM] position with the binary outcome label using BCE loss. Predictions at earlier steps of these trajectories are not supervised, and no pseudo labels are generated.

CRM<sup>†</sup> Following CRM (Zhang et al., 2025a), the scalar prediction at step t parameterizes the conditional probability $h _ { t }$ that the trajectory first enters an invalid state at that step. The probability of remaining valid through step $T$ is

$$
S _ { T } = \prod _ { t = 1 } ^ { T } ( 1 - h _ { t } ) .\tag{10}
$$

For a process-annotated trajectory, we set its trajectory label to positive when all available step labels are valid and to negative otherwise. For a negative trajectory, the first step labeled invalid is used as the observed first-error position z, and CRM optimizes both the trajectory outcome likelihood and the localization likelihood $\begin{array} { r } { \operatorname* { P r } ( z ) = h _ { z } \prod _ { t < z } ( 1 - h _ { t } ) } \end{array}$ . CRM<sup>†</sup> further applies the outcome objective to outcome-annotated trajectories without process labels: it minimizes − log S<sub>T</sub> for a positive outcome and $- \log ( 1 - S _ { T } )$ for a negative outcome, thereby marginalizing over the unknown first-error position.

## A.4 OUTCOME DATA CONSTRUCTION

We convert the original AceMath-RM (Liu et al., 2025b) reward model training data into the outcome-annotated trajectories used in our experiments. For each example, we extract the user question and the corresponding assistant response, and retain the original binary correctness label as the trajectory-level outcome. We remove multiple-choice examples, malformed conversations, and responses for which reliable reasoning steps or a final answer cannot be identified.

Outcome label reliability. The outcome labels in AceMath-RM are constructed with multiple stages of quality control. Candidate responses are first labeled as correct or incorrect by comparing their answers against reference labels using the Qwen-Math evaluation toolkit. AceMath-RM further ranks positive and negative candidates using Qwen2.5-Math-RM-72B and applies score-sorted sampling to retain high-confidence responses, reducing label noise introduced by heuristic answer matching (Liu et al., 2025b). This ranking and filtering also helps reduce cases where a response reaches the correct final answer despite containing unresolved reasoning errors. We do not assume that a correct final answer implies that all preceding reasoning steps are valid. Instead, the outcome label is used only to supervise the final propagated state, while intermediate states are determined through state propagation and process supervision. Therefore, the filtered outcome labels provide reliable supervision for the final state in most cases.

Table 5: F1 scores (%) on ProcessBench. Our method classifies a step as valid when $p _ { t } ^ { \mathrm { G } } \geq 0 . 5$ The last column reports the macro-average across all subsets.
<table><tr><td>Method</td><td>GSM8K</td><td>MATH</td><td>OlympiadBench Omni-MATH</td><td></td><td>Avg.</td></tr><tr><td>Supervised PRM</td><td>69.8</td><td>60.3</td><td>48.5</td><td>47.9</td><td>56.6</td></tr><tr><td>OVM</td><td>49.9</td><td>44.1</td><td>28.1</td><td>28.2</td><td>37.6</td></tr><tr><td>Qwen2.5-Math-PRM</td><td>81.8</td><td>67.3</td><td>66.8</td><td>66.2</td><td>70.5</td></tr><tr><td>Pseudo-Label PRM</td><td>1.9</td><td>2.1</td><td>2.0</td><td>2.2</td><td>2.1</td></tr><tr><td>Joint-Supervised PRM</td><td>70.1</td><td>66.2</td><td>57.0</td><td>52.8</td><td>61.5</td></tr><tr><td>CRM†</td><td>73.8</td><td>56.2</td><td>30.7</td><td>24.4</td><td>46.3</td></tr><tr><td>RSP</td><td>73.5</td><td>71.5</td><td>60.9</td><td>57.1</td><td>65.8</td></tr></table>

## B PROCESS ERROR IDENTIFICATION

We evaluate process error identification on ProcessBench (Zheng et al., 2024). ProcessBench requires a model to identify the first erroneous reasoning step or determine that the entire solution is correct. For RSP, we consider a reasoning prefix valid when $p _ { t } ^ { \mathrm { G } } \geq 0 . 5$ and predict the first step with $p _ { t } ^ { \mathrm { G } } < 0 . 5$ as the error location. If no reasoning step is identified as erroneous, the solution is predicted to be fully correct. Following ProcessBench, we report F1 as the harmonic mean of the accuracy on erroneous solutions and the accuracy on fully correct solutions.

Analysis Table 5 shows that, among the models trained under our shared experimental setup, RSP improves Supervised PRM by 9.2%. RSP also outperforms Qwen2.5-Math-PRM on MATH by 4.2%, although Qwen2.5-Math-PRM achieves the strongest overall result with additional large-scale process supervision. These results indicate that process annotations remain particularly important for ProcessBench, which explicitly evaluates fine-grained process error identification. The very low F1 of Pseudo-Label PRM further shows the difficulty of obtaining reliable fine-grained process supervision from pseudo labels.

## C ANALYSIS OF PROCESS AND OUTCOME SUPERVISION

We analyze how process annotations and outcome labels jointly supervise break and repair in RSP. We first relate the fit of adjacent annotated states to the corresponding head predictions. We then derive how a final outcome changes the relative support for intermediate states and how this conditioning enters the training gradients.

We retain the notation of Section 3 and omit the trajectory index i. For a fixed question and reasoning trajectory $( x , \mathbf { s } )$ , state propagation gives

$$
p _ { t } ^ { \mathrm { G } } = p _ { t - 1 } ^ { \mathrm { G } } \big ( 1 - \alpha _ { t } \big ) + p _ { t - 1 } ^ { \mathrm { B } } \beta _ { t } , \qquad p _ { t } ^ { \mathrm { B } } = 1 - p _ { t } ^ { \mathrm { G } } .\tag{11}
$$

All probabilities below refer to reasoning states along this observed trajectory.

## C.1 PROCESS SUPERVISION OF BREAK AND REPAIR

For an annotated step $t \in A ,$ the individual state loss is

$$
\ell _ { t } ^ { \mathrm { s t e p } } = - y _ { t } ^ { \mathrm { s t e p } } \log p _ { t } ^ { \mathrm { G } } - ( 1 - y _ { t } ^ { \mathrm { s t e p } } ) \log p _ { t } ^ { \mathrm { B } } .\tag{12}
$$

When $p _ { t - 1 } ^ { \mathrm { G } } = 1$ , propagation reduces to $p _ { t } ^ { \mathrm { G } } = 1 - \alpha _ { t }$ , so this loss is binary cross-entropy for $\alpha _ { t }$ with target $1 - y _ { t } ^ { \mathrm { s t e p } }$ . When $p _ { t - 1 } ^ { \mathrm { B } } = 1$ , it instead reduces to binary cross-entropy for $\beta _ { t }$ with target $y _ { t } ^ { \mathrm { s t e p } }$ . Thus, fitting the current state supervises break after a valid prefix and repair after an invalid prefix. The same relation holds approximately when the preceding state is not deterministic.

Proposition 1 (Adjacent-state supervision). Suppose $t - 1 , t \in { \mathcal { A } }$ , and let $a , b \in { \mathcal { Z } }$ denote the states specified by their process labels, with 1 mapped to G and 0 to B. If

$$
p _ { t - 1 } ^ { a } \geq 1 - \epsilon , \qquad p _ { t } ^ { b } \geq 1 - \delta , \qquad 0 \leq \epsilon , \delta < 1 ,\tag{13}
$$

then

$$
y _ { t - 1 } ^ { \mathrm { s t e p } } = 1 \quad \Longrightarrow \quad \big | \alpha _ { t } - ( 1 - y _ { t } ^ { \mathrm { s t e p } } ) \big | \leq \frac { \delta } { 1 - \epsilon } ,\tag{14}
$$

$$
y _ { t - 1 } ^ { \mathrm { s t e p } } = 0 \quad \Longrightarrow \quad \left| \beta _ { t } - y _ { t } ^ { \mathrm { s t e p } } \right| \leq \frac { \delta } { 1 - \epsilon } .
$$

Proof. Let $\bar { b }$ be the state other than b. Since all entries of ${ { \bf { M } } _ { t } }$ are nonnegative,

$$
\delta \geq p _ { t } ^ { \bar { b } } \geq p _ { t - 1 } ^ { a } \mathbf { M } _ { t } [ a , \bar { b } ] \geq ( 1 - \epsilon ) \big ( 1 - \mathbf { M } _ { t } [ a , b ] \big ) .\tag{15}
$$

Hence $1 - \mathbf { M } _ { t } [ a , b ] \leq \delta / ( 1 - \epsilon )$ . Substituting the appropriate entry of ${ { \bf { M } } _ { t } }$ gives both inequalities in equation 14. For example, when an invalid prefix is followed by a valid one and both states are well fitted, the bound requires $\beta _ { t }$ to be close to one. The result applies to the prediction conditioned on the annotated preceding state. It establishes how process supervision constrains the meaning of break and repair through state propagation, without requiring separate labels for the two heads.

## C.2 OUTCOME EVIDENCE FOR INTERMEDIATE REASONING STATES

For an outcome-annotated trajectory, let $c = \mathrm { G }$ when $y ^ { \mathrm { o u t } } = 1$ and $c = \mathrm { B }$ otherwise, following the terminal target used by ${ \mathcal { L } } _ { \mathrm { o u t } }$ . Expanding the propagation product gives

$$
Z _ { \mathrm { o u t } } : = p _ { T } ^ { c } = \sum _ { z _ { 1 : T - 1 } \in \mathcal { Z } ^ { T - 1 } } \prod _ { j = 1 } ^ { T } \mathbf { M } _ { j } [ z _ { j - 1 } , z _ { j } ] ,\tag{16}
$$

where $z _ { 0 } = \mathrm { G }$ and $z _ { T } ~ = ~ c$ are fixed. The individual outcome loss is $\ell ^ { \mathrm { { o u t } } } = - \log Z _ { \mathrm { { o u t } } }$ . The sum includes every possible sequence of intermediate states ending in c. A correct outcome can therefore be explained by reasoning that remains valid throughout or by reasoning that contains an error followed by recovery.

To compare these explanations at position t, define

$$
h _ { t } ( a ) : = \big ( \mathbf { M } _ { t + 1 } \cdot \cdot \cdot \mathbf { M } _ { T } \big ) [ a , c ] , \qquad a \in \mathcal { Z } ,\tag{17}
$$

with the empty product equal to the identity matrix. The quantity $h _ { t } ( a )$ is the probability of reaching the observed terminal state from state a through the remaining reasoning steps. It satisfies

$$
h _ { T } ( a ) = \mathbf { 1 } \{ a = c \} , \qquad h _ { t - 1 } ( a ) = \sum _ { b \in \mathcal { Z } } \mathbf { M } _ { t } [ a , b ] h _ { t } ( b ) .\tag{18}
$$

The joint probability of $z _ { t } = a$ and $z _ { T } = c \mathrm { i s } p _ { t } ^ { a } h _ { t } ( a )$ . Conditioning on the terminal label therefore gives

$$
\gamma _ { t } ^ { a } : = p _ { \theta } ( z _ { t } = a \mid z _ { T } = c , x , \mathbf { s } ) = \frac { p _ { t } ^ { a } h _ { t } ( a ) } { Z _ { \mathrm { o u t } } } , \qquad Z _ { \mathrm { o u t } } = \sum _ { a \in \mathcal { Z } } p _ { t } ^ { a } h _ { t } ( a ) .\tag{19}
$$

This expression combines the model’s prefix evaluation $p _ { t } ^ { a }$ with the likelihood of the final result under that state, $h _ { t } ( a )$ . In particular,

$$
\gamma _ { t } ^ { \mathrm { G } } - p _ { t } ^ { \mathrm { G } } = \frac { p _ { t } ^ { \mathrm { G } } p _ { t } ^ { \mathrm { B } } \big ( h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) \big ) } { Z _ { \mathrm { o u t } } } .\tag{20}
$$

The outcome increases the relative support for whichever intermediate state makes the observed final result more likely under the remaining reasoning. It leaves the state probabilities unchanged when the two explanations have equal likelihood. Thus, the terminal observation can support or revise a prefix prediction rather than simply reinforce its existing preference. These posterior probabilities describe conditioning at fixed model parameters, and the process score remains $p _ { t } ^ { \mathrm { G } }$ , computed from the prefix alone.

## C.3 LEARNING BREAK AND REPAIR FROM THE FINAL OUTCOME

The posterior reweighting above also determines the gradients of the outcome loss. Let $\eta _ { t } ^ { \mathrm { b r } }$ and $\eta _ { t } ^ { \mathrm { r p } }$ be the head logits, with $\bar { \alpha } _ { t } = \sigma ( \eta _ { t } ^ { \mathrm { b r } } )$ and $\beta _ { t } = \sigma ( \eta _ { t } ^ { \mathrm { r p } } )$ . The outcome-conditioned targets for break and repair are

$$
\widetilde { \alpha } _ { t } = \frac { \alpha _ { t } h _ { t } ( \mathrm { B } ) } { h _ { t - 1 } ( \mathrm { G } ) } , \qquad \widetilde { \beta } _ { t } = \frac { \beta _ { t } h _ { t } ( \mathrm { G } ) } { h _ { t - 1 } ( \mathrm { B } ) } .\tag{21}
$$

For finite logits, both denominators are positive. Where the corresponding preceding state has positive probability, these quantities are respectively $p _ { \theta } ( z _ { t } = \mathrm { ~ B ~ } | \bar { z _ { t - 1 } } = \mathbf { \bar { G } } , z _ { T } = \bar { c } , x , \mathbf { s } )$ and $p _ { \theta } ( z _ { t } =  { \mathrm { G } } \ | \ z _ { t - 1 } =  { \mathrm { B } } , z _ { T } = c , x ,  { \mathbf { s } } )$

Proposition 2 (Outcome supervision of the two heads). The local derivatives of the outcome loss are

$$
\begin{array} { r l r } & { \frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \eta _ { t } ^ { \mathrm { b r } } } = \gamma _ { t - 1 } ^ { \mathrm { G } } \big ( \alpha _ { t } - \widetilde { \alpha } _ { t } \big ) , } & \\ & { \frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \eta _ { t } ^ { \mathrm { r p } } } = \gamma _ { t - 1 } ^ { \mathrm { B } } \big ( \beta _ { t } - \widetilde { \beta } _ { t } \big ) . } & \end{array}\tag{22}
$$

Proof. Holding the other head outputs fixed, write

$$
\begin{array} { r l } & { Z _ { \mathrm { o u t } } = p _ { t - 1 } ^ { \mathrm { G } } \big ( ( 1 - \alpha _ { t } ) h _ { t } ( \mathrm { G } ) + \alpha _ { t } h _ { t } ( \mathrm { B } ) \big ) } \\ & { \qquad + p _ { t - 1 } ^ { \mathrm { B } } \big ( \beta _ { t } h _ { t } ( \mathrm { G } ) + ( 1 - \beta _ { t } ) h _ { t } ( \mathrm { B } ) \big ) . } \end{array}\tag{23}
$$

Differentiating − log $Z _ { \mathrm { o u t } }$ through the sigmoid gives

$$
\begin{array} { r l } & { \frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \eta _ { t } ^ { \mathrm { b r } } } = \frac { p _ { t - 1 } ^ { \mathrm { G } } \alpha _ { t } \left( 1 - \alpha _ { t } \right) } { Z _ { \mathrm { o u t } } } \big ( h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) \big ) , } \\ & { \frac { \partial \ell ^ { \mathrm { o u t } } } { \partial \eta _ { t } ^ { \mathrm { r p } } } = - \frac { p _ { t - 1 } ^ { \mathrm { B } } \beta _ { t } \left( 1 - \beta _ { t } \right) } { Z _ { \mathrm { o u t } } } \big ( h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) \big ) . } \end{array}\tag{24}
$$

Substituting equation 18, equation 19, and equation 21 yields equation 22.

Equation 22 has the form of a weighted binary cross-entropy gradient. Each head is compared with its outcome-conditioned target and weighted by the posterior probability of the relevant preceding state. For example, a repair target larger than $\beta _ { t }$ gives a negative derivative for the repair logit, encouraging a higher repair prediction. If $h _ { t } ( \mathrm { G } ) = h _ { t } ( \mathrm { B } ) > 0 .$ , the two targets equal the current predictions and both local derivatives vanish. The final result therefore supplies a learning signal according to how a step relates to the subsequent reasoning, rather than imposing the same validity target at every position.

These equalities describe the gradient of the existing outcome objective. In the weighted crossentropy interpretation, the posterior quantities are evaluated at the current parameters and treated as fixed targets and weights. Automatic differentiation of $\ell ^ { \mathrm { o u t } }$ produces the same local derivatives without an explicit posterior-labeling stage.

Implication for joint supervision. Process annotations provide direct evidence about intermediate validity through equation 14. On outcome-only data, the same model supplies the prefix probabilities used in equation 19, while the terminal label reweights these predictions through the remaining reasoning. The two losses consequently train the same break and repair heads using different levels of annotation.

A useful instance is a correct outcome following a predicted invalid prefix. If $y ^ { \mathrm { o u t } } = 1 , p _ { k } ^ { \mathrm { B } } \geq 1 - \epsilon$ and $p _ { T } ^ { \mathrm { G } } \geq 1 - \delta$ , with $0 \leq \epsilon , \delta < 1$ , then

$$
h _ { k } ( \mathrm { B } ) \geq 1 - \frac { \delta } { 1 - \epsilon } .\tag{25}
$$

This follows from $p _ { T } ^ { \mathrm { B } } \geq p _ { k } ^ { \mathrm { B } } ( 1 - h _ { k } ( \mathrm { B } ) )$ . Conditional on retaining the invalid prefix prediction, fitting the correct outcome requires the remaining steps to provide sufficient recovery. The outcome constrains their combined effect while leaving the location of recovery to be learned from the reasoning

context. This connects locally supervised validity judgments with terminal feedback on trajectories that lack process annotations. Section 4.5 and Table 4 evaluate the learned break and repair scores and the empirical contribution of the two supervision sources.

## C.4 STRENGTH OF STATE PROPAGATION

The analysis above shows that outcome supervision reaches intermediate transitions through the propagated reasoning states. We further examine the strength of this propagation in the learned model. From equation 11, using $p _ { t - 1 } ^ { \mathrm { B } } = 1 - p _ { t - 1 } ^ { \mathrm { G } }$ , we can write

$$
p _ { t } ^ { \mathrm { G } } = \beta _ { t } + \kappa _ { t } p _ { t - 1 } ^ { \mathrm { G } } , \qquad \kappa _ { t } : = 1 - \alpha _ { t } - \beta _ { t } .\tag{26}
$$

Therefore,

$$
\frac { \partial p _ { t } ^ { \mathrm { G } } } { \partial p _ { t - 1 } ^ { \mathrm { G } } } = \kappa _ { t } .\tag{27}
$$

The coefficient $\kappa _ { t }$ directly measures how strongly the propagated state after step t depends on the preceding propagated state. Across multiple steps,

$$
\frac { \partial p _ { t } ^ { \mathrm { G } } } { \partial p _ { s } ^ { \mathrm { G } } } = \prod _ { j = s + 1 } ^ { t } \kappa _ { j } , \qquad s < t .\tag{28}
$$

Thus, smaller $\left| \kappa _ { t } \right|$ weakens the influence of earlier reasoning states, whereas larger $\left| \kappa _ { t } \right|$ preserves more of the preceding state information.

The same coefficient also determines how outcome supervision is transmitted backward through state propagation. Recall that $h _ { t } ( a )$ denotes the probability of reaching the observed terminal state from state a. From the backward recursion in equation 18, the difference between the two possible current states satisfies

$$
h _ { t - 1 } ( \mathrm { G } ) - h _ { t - 1 } ( \mathrm { B } ) = \kappa _ { t } \big ( h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) \big ) .\tag{29}
$$

Consequently,

$$
| h _ { t } ( \mathrm { G } ) - h _ { t } ( \mathrm { B } ) | = \prod _ { j = t + 1 } ^ { T } | \kappa _ { j } | ,\tag{30}
$$

where the terminal difference has magnitude one. Since $h _ { t } ( \mathrm { G } ) \mathrm { ~ - ~ } h _ { t } ( \mathrm { B } )$ appears directly in the outcome gradients in equation $2 4 , \kappa _ { t }$ has a dual role: it controls both the forward dependence of later states on earlier states and the backward propagation of outcome supervision to earlier transitions. Importantly, this formulation does not require outcome supervision to propagate equally strongly to all preceding transitions. Its influence is modulated by the learned intervening state transitions, so that earlier states whose effect is weakly preserved receive correspondingly weaker terminal evidence.

Empirical Analysis Figure 3(a) shows that the learned state dependence is predominantly positive. The median value of $\kappa _ { t }$ is 0.092 and the mean is 0.124. Moreover, 91.5% of the transitions have $\kappa _ { t } > 0$ , while only 8.5% have $\kappa _ { t } < 0$ . Therefore, for the large majority of reasoning steps, a higher probability of being in the valid state before the step continues to increase the propagated valid state probability after the step. The distribution also contains a broad positive tail, showing that a subset of transitions retains substantially stronger dependence on preceding states.

Figure 3(b) further shows that the propagation strength varies considerably across reasoning steps. Only 3.3% of transitions satisfy $| \kappa _ { t } | < 0 . 0 1$ , indicating that transitions that almost completely remove dependence on the preceding propagated state are relatively uncommon. Meanwhile, 22.0% of transitions have $| \kappa _ { t } | < \mathrm { { \dot { 0 } } } . 0 5 , 5 1 { \mathrm { { \dot { . } 1 } \mathrm { { \dot { \% } } } } }$ have $| \kappa _ { t } | < 0 . 1$ , and 79.0% have $| \kappa _ { t } | < 0 . 2$ . Thus, RSP does not maintain the same propagation strength at every reasoning step. Instead, some transitions substantially reduce the influence of preceding states, while others preserve stronger dependence across successive steps.

These results suggest that reasoning state propagation is selective rather than uniform. This behavior is consistent with the formulation of RSP: later reasoning may preserve an earlier reasoning state, introduce an error, or recover from an earlier error, and these cases need not retain the same amount of dependence on the preceding state. The same learned transition structure also determines how outcome supervision is propagated backward. Therefore, RSP allows both state dependence and outcome supervision to vary across the reasoning trajectory according to the learned state transitions, rather than requiring equally strong propagation at every step.

![](images/2eaee6210c696a7bd4d54014b0616ae59128ee5814ec49d71569760647e8b376.jpg)  
(a) Distribution of $\kappa _ { t }$

![](images/8e61b38f6abba6a40463dc6755c2d087363146faa510b4ae2a1e5b906037354f.jpg)  
(b) Empirical cumulative distribution of $\left| \kappa _ { t } \right|$  
Figure 3: (a) Distribution of $\kappa _ { t } = 1 - \alpha _ { t } - \beta _ { t }$ . The signed value indicates the direction and strength of the dependence of the updated propagated state on the preceding state. (b) Empirical cumulative distribution of $\left| \kappa _ { t } \right|$ . Smaller $\left| \kappa _ { t } \right|$ indicates weaker dependence of the updated propagated state on the preceding propagated state.

## D SCALING ANALYSIS

We further examine how RSP behaves under different model capacities and amounts of training supervision. Unless otherwise specified, all training and evaluation settings follow the main experiments. Process-Only denotes the corresponding model trained with process supervision alone.

Scaling with Model Size. We first study whether the benefit of RSP persists as the PRM backbone scales. We vary the backbone size of the Qwen3 series models from 1.7B to 8B while keeping the training data and other settings fixed. As shown in Figure 4, RSP consistently improves as the model size increases across all four evaluation settings. Compared with Process-Only, RSP maintains a clear advantage across all model sizes. These results indicate that the benefit of incorporating outcome supervision is maintained as the PRM capacity increases.

Scaling with Outcome Supervision. We further study how RSP scales with the amount of outcome supervision by varying the process-to-outcome data ratio in each training batch from 1:1 to 1:9, while keeping the process supervision fixed. As shown in Figure 5, increasing outcome supervision provides substantial improvements in beam search and reinforcement learning. Beam search accuracy increases from approximately 69% to above 72%, while reinforcement learning improves from about 63.2% to 65.8%. In contrast, Best-of-N selection and ProcessBench remain relatively stable across different ratios. These results show that RSP can effectively benefit from additional outcome supervision, particularly for reasoning search and policy optimization.

## E TRAINING DATA AND EVALUATION OVERLAP

We further clarify the relation between the training data used in our experiments and the evaluation benchmarks.

Process Supervision. For process supervision, we use a deduplicated version of PRM800K (Lightman et al., 2024) before selecting the 20% subset used in our main experiments. Specifically, duplicated full reasoning trajectories are removed during preprocessing. The deduplication is performed at the trajectory level. Distinct trajectories that share partial reasoning prefixes are retained, since they correspond to different annotated reasoning trajectories. PRM800K uses the MATH split released by Lightman et al. (2024). In this split, 4,500 problems from the original MATH test set are included in the training set, while the remaining 500 problems are held out for evaluation. These held-out problems form MATH500 and are not included in the PRM800K training data used for process supervision. Therefore, our MATH500 evaluation is separated from the PRM800K process training data by construction.

![](images/517bed092075e55d4af41b8d79e7b1d4d0bed25f2b3bc5125d2bfdedb46a0e69.jpg)  
Figure 4: Performance with different PRM backbone sizes. We compare RSP with Process-Only on beam search, Best-of-N selection, reinforcement learning, and ProcessBench.

Outcome Supervision. For outcome supervision, we use the released AceMath-RM (Liu et al., 2025b) training data and convert it into outcome-annotated trajectories following Appendix A.4. Our preprocessing only filters examples and converts each retained response into reasoning steps with a trajectory-level outcome label. It does not introduce additional questions from any evaluation benchmark. AceMath-RM is constructed from the math training data of AceMath, where test-data filtering is applied before model training.

Reinforcement Learning. The verifier-gated update used in our reinforcement learning experiments follows VeriGate (Agrawal et al., 2026). For RL training data, we use an 8K subset of the public DAPO-MATH-17K dataset (Yu et al., 2026), following the setting described in Section 4.4 and Appendix A.2. DAPO-MATH-17K is released as a curated and processed math training dataset. We sample the 8K training prompts only from this released training set. All methods in Table 3 use exactly the same 8K training prompts, together with the same policy initialization and optimization budget.

Overall, the process, outcome, and reinforcement learning experiments use their corresponding public training data. Our preprocessing does not add examples from the evaluation benchmarks to these training sets. In particular, MATH500 is explicitly held out from the PRM800K process training split. The same training data are used across the compared methods under each experimental setting.

![](images/634c5e310d8803466137fdf45332a36218ca2625cae7a9167155cd93d683b0c7.jpg)  
Figure 5: Performance with different amounts of outcome supervision. The horizontal axis denotes the process-to-outcome data ratio within each training batch. For example, 1:3 indicates that the numbers of process-annotated and outcome-annotated examples are mixed at a ratio of 1:3 in a batch. The process supervision is kept fixed while the amount of outcome supervision is varied.

## F QUALITATIVE EXAMPLES

We provide qualitative examples to illustrate how RSP propagates reasoning validity across different reasoning patterns. Each step reports the propagated process score together with the predicted break and repair probabilities. Examples illustrate representative state patterns, including validity preservation, error introduction and persistence, and error recovery.

## Example 1

Question:

How many four-digit positive integers contain at least one 2 or 3 as a digit?

LLM Response:

To determine how many four-digit positive integers contain at least one 2 or 3 as a digit, we can use the principle of complementary counting. This involves first finding the total number of four-digit positive integers and then subtracting the number of four-digit positive integers that do not contain any 2s or 3s as digits.

Step 2:

Calculate the total number of four-digit positive integers.

A four-digit positive integer ranges from 1000 to 9999. Therefore, the total number of four-digit positive integers is:

$$
9 9 9 9 - 1 0 0 0 + 1 = 9 0 0 0 .
$$

Step 3:

✓ PRM 0.995 Break 0.005 Repair 0.905

Calculate the number of four-digit positive integers that do not contain any 2s or 3s as digits.

For a four-digit number, the digits can be chosen as follows:

• The thousands place can be any digit from 1 to 9 except 2 and 3, so there are $7$ choices (1, 4, 5, 6, 7, 8, 9).

• The hundreds, tens, and units places can each be any digit from 0 to 9 except 2 and 3, so there are 8 choices for each of these places (0, 1, 4, 5, 6, 7, 8, 9).

Step 4:

✓ PRM 0.996 Break 0.003 Repair 0.887

Thus, the total number of four-digit positive integers that do not contain any 2s or 3s as digits is:

$$
7 \times 8 \times 8 \times 8 = 7 \times 5 1 2 = 3 5 8 4 .
$$

Step 5:

✓ PRM 0.997 Break 0.002 Repair 0.853

Calculate the number of four-digit positive integers that contain at least one 2 or 3 as a digit.

Using the principle of complementary counting, we subtract the number of four-digit positive integers that do not contain any 2s or 3s from the total number of four-digit positive integers:

$$
9 0 0 0 - 3 5 8 4 = 5 4 1 6 .
$$

Step 6:

✓ PRM 1.000 Break 0.000 Repair 0.835

Therefore, the number of four-digit positive integers that contain at least one 2 or 3 as a digit is:

5416 .

G → G: Validity Preservation. Example 1 shows a fully valid reasoning trajectory. The break probability remains close to zero throughout the solution, and the propagated process score stays near one, correctly preserving the valid reasoning state across successive steps.

## Example 2

## Question:

Given that the function $f ( x )$ defined on R is an odd function and satisfies $f ( 1 + x ) =$ $f ( 3 + x )$ . When $0 \leq x \leq 1 , f ( x ) = x ^ { 3 } - x$ . Find

$$
f \left( { \frac { 1 1 } { 2 } } \right) + f ( 6 ) .
$$

LLM Response:

Step 1: ✓ PRM 0.991 Break 0.009 Repair 0.973 To find $f \left( { \frac { 1 1 } { 2 } } \right) + f ( 6 )$ , we need to use the properties of the function $f .$ The function $f$ is an odd function, which means $f ( - x ) = - f ( x )$ , and it satisfies the periodicity condition $f ( 1 + x ) = f ( 3 + x )$ . This implies that $f ( x ) = f ( x + 2 )$ , so f is periodic with period 2. Step 2: ✓ PRM 0.992 Break 0.008 Repair 0.998

First, let’s find $f ( 6 )$ . Since $f$ is periodic with period 2, we have:

$$
f ( 6 ) = f ( 6 - 2 \cdot 3 ) = f ( 0 ) .
$$

Given that $f ( x ) = x ^ { 3 } - x$ for $0 \leq x \leq 1$ , we substitute $x = 0 \colon$

$$
f ( 0 ) = 0 ^ { 3 } - 0 = 0 .
$$

So, $f ( 6 ) = 0 .$

Step 3:

× PRM 0.104 Break 0.895 Repair 0.011

Next, let’s find $f \left( { \frac { 1 1 } { 2 } } \right)$ . Using the periodicity of $f ,$ we have:

$$
f \left( { \frac { 1 1 } { 2 } } \right) = f \left( { \frac { 1 1 } { 2 } } - 2 \cdot 2 \right) = f \left( { \frac { 1 1 } { 2 } } - 4 \right) = f \left( { \frac { 3 } { 2 } } \right) .
$$

Since $f$ is odd, we can write:

$$
f \left( { \frac { 3 } { 2 } } \right) = f \left( 1 + { \frac { 1 } { 2 } } \right) = f \left( 3 + { \frac { 1 } { 2 } } - 2 \right) = f \left( 1 + { \frac { 1 } { 2 } } \right) = f \left( { \frac { 1 } { 2 } } \right) .
$$

Given that $f ( x ) = x ^ { 3 } - x$ for $0 \leq x \leq 1$ , we substitute $\begin{array} { r } { x = { \frac { 1 } { 2 } } } \end{array}$ :

$$
f \left( { \frac { 1 } { 2 } } \right) = \left( { \frac { 1 } { 2 } } \right) ^ { 3 } - { \frac { 1 } { 2 } } = { \frac { 1 } { 8 } } - { \frac { 1 } { 2 } } = { \frac { 1 } { 8 } } - { \frac { 4 } { 8 } } = - { \frac { 3 } { 8 } } .
$$

So,

$$
f \left( { \frac { 1 1 } { 2 } } \right) = - { \frac { 3 } { 8 } } .
$$

Step 4:

Finally, we add $f \left( { \frac { 1 1 } { 2 } } \right)$ and $f ( 6 )$

× PRM 0.018 Break 0.870 Repair 0.005

$$
f \left( { \frac { 1 1 } { 2 } } \right) + f ( 6 ) = - { \frac { 3 } { 8 } } + 0 = - { \frac { 3 } { 8 } } .
$$

Thus, the answer is:

$$
\boxed { - \frac { 3 } { 8 } } .
$$

G → B and $\mathrm { { B } }  \mathrm { { B } } \colon$ Error Introduction and Persistence. In Example 2, the reasoning is valid through Step 2, but Step 3 incorrectly reduces $f ( 3 / 2 )$ to $f ( 1 / 2 )$ . The high break probability sharply lowers the process score, while the subsequent step continues from the incorrect result with a very low repair probability, preserving the invalid reasoning state.

## Example 3

Question:

Rationalize the denominator of

$$
{ \frac { 7 } { \sqrt { 7 5 } } } .
$$

LLM Response:

Step 1:

To rationalize the denominator of the expression $\frac { 7 } { \sqrt { 7 5 } } .$ , we need to eliminate the square root from the denominator. We can do this by multiplying both the numerator and the denominator by ${ \sqrt { 7 5 } } .$

Here are the steps:

Start with the original expression:

$$
{ \frac { 7 } { \sqrt { 7 5 } } } .
$$

Step 2: PRM 0.995 Break 0.004 Repair 0.941 Multiply both the numerator and the denominator by ${ \sqrt { 7 5 } } \colon$

$$
{ \frac { 7 } { \sqrt { 7 5 } } } \cdot { \frac { \sqrt { 7 5 } } { \sqrt { 7 5 } } } .
$$

Step 3: ✓ PRM 0.987 Break 0.012 Repair 0.738 Perform the multiplication in the numerator and the denominator:

$$
{ \frac { 7 \cdot { \sqrt { 7 5 } } } { \sqrt { 7 5 } \cdot { \sqrt { 7 5 } } } } = { \frac { 7 { \sqrt { 7 5 } } } { 7 5 } } .
$$

Step 4:

× PRM 0.399 Break 0.600 Repair 0.358

Simplify the expression by canceling the common factor of 7 in the numerator and the denominator:

$$
{ \frac { 7 { \sqrt { 7 5 } } } { 7 5 } } .
$$

Step 5: PRM 0.802 Break 0.018 Repair 0.682 Simplify $\sqrt { 7 5 }$ . We know that $7 5 = 2 5 \cdot 3$ and ${ \sqrt { 2 5 } } = 5 ,$ , so:

$$
{ \sqrt { 7 5 } } = { \sqrt { 2 5 \cdot 3 } } = 5 { \sqrt { 3 } } .
$$

Step 6:

✓ PRM 0.924 Break 0.009 Repair 0.651

Substitute $\sqrt { 7 5 }$ with $5 \sqrt { 3 }$ in the expression:

$$
{ \frac { 7 \cdot 5 { \sqrt { 3 } } } { 7 5 } } = { \frac { 3 5 { \sqrt { 3 } } } { 7 5 } } .
$$

Step 7: ✓ PRM 0.925 Break 0.044 Repair 0.556 Simplify the fraction $\frac { 3 5 } { 7 5 }$ . The greatest common divisor of 35 and 75 is 5, so:

$$
{ \frac { 3 5 } { 7 5 } } = { \frac { 3 5 \div 5 } { 7 5 \div 5 } } = { \frac { 7 } { 1 5 } } .
$$

Step 8:

Therefore, the expression simplifies to:

✓ PRM 0.972 Break 0.024 Repair 0.929

$$
{ \frac { 7 { \sqrt { 3 } } } { 1 5 } } .
$$

Step 9:

The rationalized form of the expression is

✓ PRM 0.999 Break 0.000 Repair 0.972

$$
{ \boxed { 7 { \sqrt { 3 } } } } .
$$

B → G: Error Recovery. Example 3 illustrates recovery after an intermediate error. Step 4 introduces an invalid reasoning step and reduces the process score, whereas the subsequent valid derivation returns to a correct computation. Correspondingly, the repair probability increases and the propagated process score gradually recovers.

## Example 4

## Question:

Calculate the sum of all integral values of c with $c \leq 2 0$ for which the equation

$$
y = x ^ { 2 } - 9 x - c
$$

produces two rational roots.

## LLM Response:

## Step 1:

✓ PRM 0.955 Break 0.045 Repair 0.890

To determine the sum of all integral values of c with $c \leq 2 0$ for which the quadratic equation $y = x ^ { 2 } - 9 x - c$ has two rational roots, we start by using the discriminant. A quadratic equation $a x ^ { 2 } + b x + c = 0$ has two rational roots if and only if its discriminant

$$
\Delta = b ^ { 2 } - 4 a c
$$

is a perfect square.

## Step 2:

✓ PRM 0.985 Break 0.012 Repair 0.914

For the given equation $x ^ { 2 } - 9 x - c = 0$ , the discriminant is

$$
\Delta = ( - 9 ) ^ { 2 } - 4 \cdot 1 \cdot ( - c ) = 8 1 + 4 c .
$$

We need $8 1 + 4 c$ to be a perfect square. Let

$$
8 1 + 4 c = k ^ { 2 }
$$

for some integer k. Then

$$
c = { \frac { k ^ { 2 } - 8 1 } { 4 } } .
$$

Since c must be an integer and $8 1 \equiv 1$ (mod 4), we require

$$
k ^ { 2 } \equiv 1 { \pmod { 4 } } .
$$

Therefore, k must be an odd integer.

## Step 3:

× PRM 0.431 Break 0.570 Repair 0.500

Next, we need to find the values of k such that $c \leq 2 0$ . Substituting $\begin{array} { r } { c = \frac { k ^ { 2 } - 8 1 } { 4 } } \end{array}$ into the inequality gives

$$
{ \frac { k ^ { 2 } - 8 1 } { 4 } } \leq 2 0 \quad \Rightarrow \quad k ^ { 2 } \leq 1 6 1 .
$$

The largest odd integer k that satisfies $k ^ { 2 } \leq 1 6 1$ is 12, and the smallest odd integer is −13. Therefore, k can take any odd value from −13 to 13.

## Step 4:

✓ PRM 0.890 Break 0.008 Repair 0.812

We now list the odd integers in this range and calculate the corresponding values of c:

$$
k = - 1 3 \quad \Rightarrow \quad c = { \frac { 1 6 9 - 8 1 } { 4 } } = 2 2 \quad \quad ( { \mathrm { n o t ~ v a l i d ~ s i n c e ~ } } c \leq 2 0 ) ,
$$

$$
k = - 1 1 \quad \Rightarrow \quad c = { \frac { 1 2 1 - 8 1 } { 4 } } = 1 0 ,
$$

$$
k = - 9 \quad \Rightarrow \quad c = { \frac { 8 1 - 8 1 } { 4 } } = 0 ,
$$

$$
k = - 7 \quad \Rightarrow \quad c = { \frac { 4 9 - 8 1 } { 4 } } = - 8 ,
$$

$$
k = - 5 \quad \Rightarrow \quad c = { \frac { 2 5 - 8 1 } { 4 } } = - 1 4 ,
$$

$$
k = - 3 \quad \Rightarrow \quad c = { \frac { 9 - 8 1 } { 4 } } = - 1 8 ,
$$

$$
k = - 1 \quad \Rightarrow \quad c = { \frac { 1 - 8 1 } { 4 } } = - 2 0 .
$$

The positive values of k give the same values of c. In particular, $k = 1 3$ also gives $c = 2 2$ which is not valid since $c \leq 2 0$ . Therefore, the valid values of c are

$$
- 2 0 , - 1 8 , - 1 4 , - 8 , 0 , 1 0 .
$$

Step 5:

$$
\sqrt { \mathrm { ~ \bf ~ P R M ~ } 0 . 9 7 8 \mathrm { ~ \bf ~ B r e a k ~ } 0 . 0 0 0 \mathrm { ~ \bf ~ R e p a i r ~ } 0 . 8 0 5 }
$$

Summing these values, we get

$$
- 2 0 + ( - 1 8 ) + ( - 1 4 ) + ( - 8 ) + 0 + 1 0 = - 5 0 .
$$

Thus, the sum of all integral values of c is

$$
\boxed { - 5 0 } .
$$

$\mathrm { { B } }  \mathrm { { G } } \cdot$ : Error Recovery. Example 4 illustrates recovery after an intermediate error. Step 3 incorrectly includes $\pm 1 3$ among the possible values of $k ,$ even though $1 3 ^ { 2 } > 1 6 1$ . In Step 4, the response evaluates these values, obtains $c = 2 2$ , and explicitly excludes them because $c \leq 2 0$ . The derivation then returns to the correct set of values and produces the correct final answer.