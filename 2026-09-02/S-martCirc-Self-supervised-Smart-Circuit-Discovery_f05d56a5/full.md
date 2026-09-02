# S³martCirc: Self-supervised Smart Circuit Discovery

Wendy Zheng<sup>1∗</sup>, Yinhan He<sup>1∗</sup>, Liang Wu<sup>2</sup> Jundong Li<sup>1†</sup>

<sup>1</sup>University of Virginia

<sup>2</sup>Nokia

ncd9cf@virginia.edu,nee7ne@virginia.edu, liang.wu@nokia.com, jundong@virginia.edu

## Abstract

Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks, from text summarization to question answering. Despite these capabilities, their black-box nature obscures internal decision-making processes. Mechanistic interpretability (MI) aims to address this by reverse-engineering neural networks into humanunderstandable algorithms. Current MI approaches for LLMs typically follow a two-stage paradigm: first identifying important components (circuit discovery), where components are typically individual nodes such as an attention head or feedforward neuron, and second determining the role they play in a certain task (functional interpretation). However, this sequential approach overlooks a fundamental insight: a component’s importance and its functional role are inherently codependent. Unifying these stages presents two key challenges: (1) functional roles are often tied to specific nodes or components, limiting generalization, and (2) their identification relies on subjective interpretation rather than quantifiable metrics. To address these challenges, we propose S³martCirc (Self supervised Smart Circuit Discovery), a unified framework that simultaneously discovers circuits and interprets functionality. S³martCirc abstracts node behavior into two general computational roles that generalize across tasks and defines a quantitative metric for assigning them, enabling importance and functional role to be discovered jointly rather than in sequence. Extensive experiments show that our framework outperforms existing methods in circuit discovery.

Code — https://anonymous.4open.science/r/s3martcirc

## Introduction

Recently, Large Language Models (LLMs) have exhibited extraordinary performance across a wide range of diverse tasks, from text summarization (Basyal and Sanghvi 2023; Van Veen et al. 2023) and translation (Feng et al. 2024; Kleidermacher and Zou 2025) to question answering (Li et al. 2024; Tan et al. 2023; Yang et al. 2023a). However, these models are primarily black boxes whose internal reasoning mechanisms cannot be directly accessed or interpreted by hu mans. This opacity obscures their decision-making processes and raises safety concerns about potential undesired behaviors (Singh et al. 2024; Wagner et al. 2024). Consequently, it hinders the adoption of LLMs in critical fields such as healthcare and finance, where reliability and trustworthiness are paramount (Tatsat and Shater 2025; Yang et al. 2023b).

To better understand how LLMs operate, various interpretation techniques have been proposed (Chen et al. 2021; He et al. 2025; Hoover, Strobelt, and Gehrmann 2020; Singh et al. 2024), among which mechanistic interpretability (MI) has emerged as a particularly promising direction (Bereska and Gavves 2024; Sharkey et al. 2025). MI aims to reverse engineer neural networks into human-understandable algorithms by uncovering how components collaborate to implement a specific task. Current approaches for LLMs generally follow a two-stage paradigm: (1) identify important nodes (i.e., attention heads or feedforward neurons) through circuit discovery, and (2) determine the functional role each selected node performs through functional interpretation (Liu, Mao, and Wen 2025; Nanda et al. 2023; Wang et al. 2022).

An alternative perspective suggests that this sequential separation is not fundamental. The order of the two stages can be reversed: one may first identify the functional operations required for a task and subsequently search for the nodes that implement them (Méloux et al. 2025a). In this view, node importance and functionality are intrinsically interdependent: a node is important because of the function it performs, while its function is defined by its contribution to the task. Knowing a node’s functional role therefore provides direct evidence for whether it is relevant to the task, and conversely, knowing that a node is important constrains the role it is likely to play. This mutual dependence implies that circuit discovery and functional interpretation should not be treated as sequential steps, but as a single, jointly optimized problem. However, realizing this perspective faces two key challenges: (1) Node-specific interpretations. Prior works assign fine-grained, semantically rich roles (e.g., name mover or S-inhibition heads) that are tightly bound to a specific node and task, and therefore do not provide a general abstraction that can be shared across tasks or embedded into an automated pipeline. (2) Subjective functional roles. Current interpretation methods heavily rely on subjective human judgment, making functional roles dificult to formalize and integrate into automated discovery pipelines.

To address these challenges, we propose Self-supervised Smart Circuit Discovery (S³martCirc), a framework that jointly performs circuit discovery and functional interpretation. Our approach is motivated by a key observation about LLMs: their core computation, matrix multiplication, primarily enables two fundamental operations, computational transformation and information propagation.

We therefore abstract the fine-grained, task-specific interpretations of prior work into a coarser dichotomy of computational roles that describes how a node processes information rather than what content it processes. This abstraction deliberately trades semantic specificity for two properties that fine-grained roles lack: generality and quantifiability. These properties allow the abstraction to be embedded directly into an optimization objective, enabling S³martCirc to simultaneously identify important nodes and assign functional roles while explicitly modeling the interdependence between importance and functionality. Our contributions are summarized as follows:

• General functional roles: We introduce a dichotomy of computational roles that abstracts prior task-specific interpretations into two general, quantifiable categories.

• S³martCirc Framework: We propose a novel MI framework that couples discovery and interpretation through bidirectional, alternating optimization.

• Empirical validation: Extensive experiments across multiple LLM architectures demonstrate the superiority of S³martCirc in identifying task-relevant circuits and recovering known circuits aligned with human-validated mechanistic interpretations.

## Preliminaries and Definitions

In this section, we first introduce the notation used throughout the paper. We then examine circuits from two distinct tasks and define two general functional roles that characterize how nodes process information at a high level.

Given an LLM, let P be its output probability distribution and V be its vocabulary. We model the LLM as a computational graph in which each node is a model component (i.e. attention head or feedforward layer neuron) and each edge represents information flowing between components through the residual stream, as illustrated in Figure 2. A circuit is the subgraph within the computational graph responsible for the model’s performance on a given task. Each node takes the previous layer’s representation $X \in \mathbb { R } ^ { T \times d _ { m } }$ as input and produces an output $Y \in \mathbb { R } ^ { T \times d _ { n } }$ , where $T$ is the number of input tokens, $d _ { m }$ is the model’s hidden dimension, and $d _ { n }$ is the node’s output dimension. For attention heads, $d _ { n }$ is determined by the architecture. For feedforward neurons, $d _ { n } = 1$ All discovery methods we compare operate over the same node set and graph abstraction to ensure a fair comparison.

Current MI research typically identifies and interprets nodes in the context of a single task during functional interpretation. However, when comparing circuits across diferent tasks, we observe striking similarities in their functional interpretations, echoing evidence that circuit components are reused across tasks with consistent roles (Merullo, Eickhof, and Pavlick 2024). This suggests that the same small set of computational roles recurs across tasks. For example, consider the two tasks shown in Figure 1: the Indirect Object Identification (IOI) task (Wang et al. 2022), where the model identifies the indirect object in sentences (e.g., "When John and Mary went to the store, John gave a drink to [Mary]"), and the Acronym Prediction task (García-Carrasco, Maté, and Trujillo 2024b), where the model predicts the last letter of a three-letter acronym from a word sequence (e.g., "The Three Letter Acronym" → "TLA"). Although these tasks are distinct, their circuits share common functional interpretations, such as previous token and mover heads.

![](images/8d1a1a1bd49f6ae38e4b219a5cedbc0ac9c9a888c8f9eeccabeb4a4c0fd32836.jpg)  
Figure 1: Mapping between node-specific functional roles in discovered circuits and the defined general functional roles.

Thus, we propose grouping node functionalities into two broad categories based on their computational role: computational transformation and information propagation. We refer to nodes in the former category as functional nodes, which actively transform their inputs, producing outputs that difer substantially in representation or semantic content. In contrast, passthrough nodes primarily relocate input features to downstream components with minimal modification. Importantly, this distinction captures how nodes operate on information rather than what information they process, enabling a task-agnostic characterization of node behavior.

Revisiting Figure 1, applying this dichotomy reveals consistent patterns across both circuits. In the IOI circuit, the Sinhibition heads and previous token heads act as passthrough nodes: the former route subject-related information to downstream components, while the latter write information about the previous token into the residual stream. In contrast, the name mover heads and negative name mover heads act as functional nodes that transform this information to selectively amplify specific names in the output logit distribution. A similar decomposition arises in the acronym circuit, where previous token heads and bridge heads propagate letter information, and letter mover heads transform it to emphasize the correct output. This consistent functional decomposition across tasks with diferent objectives suggests systematic design principles underlying how LLMs decompose complex tasks into simpler computational components.

## Self-supervised Smart Circuit Discovery

In this section, we first present an overview of the proposed framework, S³martCirc. We then describe the implementation of each stage in detail and outline the training process.

![](images/37ddf06cabdec3a388a693c4f43b761c08dd5b021888e0d729e567a004ef6703.jpg)  
Figure 2: An overview of the proposed S³martCirc framework.

## S³martCirc Overview

To unify circuit discovery and functional interpretation, we jointly model both node importance and its computational role. Specifically, for every node n in the LLM, we introduce a set of learnable masking coeficients $c _ { n } = \{ c _ { n , u } , c _ { n , p } , c _ { n , f } \}$ where $c _ { n , u }$ denotes the probability that node n is unimportant, $c _ { n , p }$ denotes the probability that the node acts as a passthrough node, and $c _ { n , f }$ denotes the probability that it acts as a functional node. These coeficients are constrained such that $c _ { n , u } + c _ { n , p } + c _ { n , f } = 1$ . We optimize these coefficients through the two-stage procedure shown in Figure 2. In the first stage, Important Node Discovery, we identify nodes that contribute to task performance by distinguishing nodes assigned to either functional role introduced earlier $( c _ { n , p } + c _ { n , f } )$ from nodes classified as unimportant $( c _ { n , u } )$ Nodes satisfying $c _ { n , p } + c _ { n , f } > c _ { n , u }$ are selected as candidate circuit nodes. In the second stage, Important Node Classification, each candidate node is assigned to one of the two roles, determined using a task-dependent metric described later. During training, node masking is applied softly using the coeficients $c _ { n }$ , while hard assignments are used only during evaluation through the following indicator function:

$$
\hat { c } _ { i } = \left\{ \begin{array} { l l } { 1 , } & { i = \arg \operatorname* { m a x } _ { j } c _ { j } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{1}
$$

## Important Node Discovery

A circuit contains the nodes responsible for performing the task at hand. Consequently, removing any truly important node from the circuit should substantially degrade task performance. We identify important nodes by optimizing two complementary objectives: (1) minimizing the KL divergence between the circuit’s output distribution and that of the full model to ensure that the circuit faithfully reproduces the model’s original computation; and (2) minimizing crossentropy loss with respect to the ground-truth answer to ensure the circuit produces correct predictions.

Formally, let $P _ { \mathrm { o r i g } }$ denote the output probability distribution of the full LLM, and $P _ { \mathrm { { m a s k } } }$ denote the distribution obtained when unimportant nodes are masked (i.e., nodes where $c _ { n , u } > c _ { n , p } + c _ { n , f } )$ . The combined objective becomes

$$
L _ { \mathrm { d i s c . } } = \alpha \cdot \sum _ { v \in V } P _ { \mathrm { o r i g } } ( v ) \log \frac { P _ { \mathrm { o r i g } } ( v ) } { P _ { \mathrm { m a s k } } ( v ) } - ( 1 - \alpha ) \cdot ( \log P _ { \mathrm { m a s k } } ( y ) )\tag{2}
$$

Here, $v \in V$ represents a token in the LLM’s vocabulary, and $P _ { \mathrm { o r i g } } ( v )$ and $P _ { \mathrm { m a s k } } ( v )$ represent the probability assigned to token v by the full model and the model when unimportant nodes are masked, respectively. $y \in V$ is the ground truth token, and $\alpha \in [ 0 , 1 ]$ controls the trade-of between faithfulness to the original model behavior and task performance.

## Important Node Classification

As defined earlier, circuit nodes are categorized as functional or passthrough nodes. Prior work typically assigns these roles using qualitative analysis or heuristic criteria (Wang et al. 2022; Chandna, Bashir, and Sen 2025). In contrast, we propose a quantitative metric based on the following observation: nodes that perform nontrivial computation should alter the relational structure between tokens, whereas nodes that propagate information should preserve this structure. To formalize this idea, we measure how the pairwise similarity between tokens changes after applying a node. Intuitively, passthrough nodes induce minimal similarity changes, while functional nodes produce larger deviations.

We define a token-wise similarity operator Sim(A) for an activation matrix $A \in \mathbb { R } ^ { T \times d _ { r } }$ (i.e. the node’s input or output) as $\mathrm { S i m } ( A ) = \hat { A } \hat { A } ^ { T } \odot ( \mathbf { 1 1 } ^ { T } - I )$ . A<sup>ˆ</sup> denotes the row-wise $L _ { 2 }$ normalization of A, 1 is a column vector of ones, and I denotes an identity matrix. We apply elementwise multiplication with $( \mathbf { 1 1 } ^ { T } - I )$ to exclude self-similarity. Using this operator, we define the functional interpretation metric as the normalized change in token-wise similarity:

$$
\operatorname { F I } ( X , Y ) = { \frac { \| \operatorname { S i m } ( X ) - \operatorname { S i m } ( Y ) \| _ { F } } { \| \operatorname { S i m } ( X ) \| _ { F } } } ,\tag{3}
$$

where $X \in \mathbb { R } ^ { T \times d _ { m } }$ and $Y \in \mathbb { R } ^ { T \times d _ { n } }$ denote the input and output of a node, respectively. Let $N _ { i }$ denote the set of nodes identified as important (i.e., those satisfying $c _ { n , p } + c _ { n , f } >$ $c _ { n , u } )$ . We incorporate this metric into the objective for the second stage:

$$
L _ { \mathrm { c l a s s . } } = \sum _ { n _ { i } \in N _ { i } } \Big [ - c _ { n _ { i } , f } \cdot \mathrm { F I } ( X _ { n _ { i } } , Y _ { n _ { i } } ) - \frac { c _ { n _ { i } , p } } { \mathrm { F I } ( X _ { n _ { i } } , Y _ { n _ { i } } ) } \Big ] .\tag{4}
$$

This objective encourages nodes that induce larger changes in similarity structure to be assigned higher $c _ { n , f }$ values, while nodes that preserve similarity are assigned higher passthrough probability $c _ { n , p }$ . In this way, functional roles are inferred directly from measurable changes in representation, enabling a fully quantitative interpretation.

## S³martCirc Training Process

Due to the vast size of LLMs, we introduce coeficient regularization, applied to both stages, to encourage the identification of minimal circuits while maximizing task performance. The objective consists of three terms that promote (1) confident assignments by enforcing binary values, (2) valid probability distributions over node roles (i.e., coeficients sum to one), and (3) sparse circuit selection:

$$
\begin{array} { l } { { \displaystyle { { \cal L } _ { \mathrm { s p a r s i t y } } } = \frac { 1 } { | C | } \sum _ { c _ { n } \in C } \Big ( \underbrace { \sum _ { v \in { \cal C } _ { n } } \left[ \frac { 1 } { 2 } | v ( 1 - v ) | \right] } _ { \mathrm { c o n h i d e n c e } } \Big . } } \\ { { \displaystyle ~ + \sum _ { \mathrm { { \scriptsize ~ \sqrt { ~ a b t d i t y } } } } \left. + ( c _ { n , u } + c _ { n , p } + c _ { n , f } ) \right| } } \\ { { \displaystyle ~ + \frac { ( c _ { n , p } + c _ { n , f } ) } { \mathrm { s p a r s i t y } } \Big ) , } } \end{array}\tag{5}
$$

where C is the set of all masking coeficients in the model. The first term pushes each coeficient toward binary values (0 or 1), the second term enforces that the three probabilities sum to one, and the third term directly penalizes nodes being included in the circuit, encouraging sparsity.

As illustrated in Figure 2, training alternates between the two stages to capture the interdependence between node importance and functional role, allowing bidirectional influence. However, assigning functional roles with randomly initialized coeficients is unreliable at the start of training. Therefore, we introduce a warmup phase that optimizes only the discovery objective $\mathrm { ( } L _ { \mathrm { d i s c . } } )$ to first identify important nodes. After warmup, we alternate between both stages to train the masking coeficients.

## Experiments

In this section, we first introduce the experiment setup. Then, we answer the following research questions: RQ1. How effective is S³martCirc at identifying circuits compared to existing methods? RQ2. Is S³martCirc able to recover known circuits from prior work? RQ3. How sensitive is S³martCirc to varying hyperparameters? RQ4. How does each component contribute to the performance of S³martCirc?

## Experiment Setup

LLMs. We select three LLMs of varying sizes for our experiments: GPT-2 (Radford et al. 2019; Maintainers 2022), Llama 3.2 1B (Grattafiori et al. 2024; Meta 2024), and Qwen 3 1.7B (Team 2025b,a). GPT-2 has been extensively examined in prior MI work, making it a well-understood and widely used benchmark for circuit analysis. Llama 3.2 1B and Qwen 3 1.7B are included to evaluate whether S³martCirc generalizes across model architectures.

Evaluation Tasks. We collect two tasks to perform circuit discovery on: Indirect Object Identification (IOI) (Wang et al. 2022) and Acronym Prediction (Acronym) (García-Carrasco, Maté, and Trujillo 2024b). Both tasks are drawn from previous MI work where extensive experiments have been performed to uncover the task circuit in GPT-2.

Evaluation Metrics. We use three metrics: Circuit Size, Accuracy Drop (Acc.), and Logit Diference (Logit.). Each method selects a set of important nodes. We then remove nodes that are not connected to the input or output, yielding the final end-to-end circuit; Circuit Size is defined as the size of this circuit. Unlike during training, we evaluate circuit performance by masking only the circuit nodes in the model, denoted as $P _ { \mathrm { c i r c } } .$ Acc. measures the percent drop in accuracy between the original model and $P _ { \mathrm { c i r c } } ^ { \phantom { \dagger } } ,$ . Lastly, Logit. denotes the diference in the logit assigned to the original model’s predicted answer between the original model and $P _ { \mathrm { c i r c } } .$

Baselines. We adopt six circuit discovery methods as baselines. We first consider patching-based approaches: Activation Patching (Act.) (Heimersheim and Nanda 2024) measures each node’s contribution by ablating it (i.e., replacing its activation with a corrupted counterpart) and evaluating the resulting change in logit diference between the model’s original answer and an alternative response. Attribution Patching (Attr.) (Syed, Rager, and Conmy 2023) provides a linear approximation of each node’s contribution by multiplying the diference between its clean and corrupted activations with the gradient of a logit-based metric (i.e., logit diference between the original and corrupted outputs). Attribution Patching with Integrated Gradients (Attr. IG) (Hanna, Pezzelle, and Belinkov 2024) extends attribution patching by integrating gradients along a path between the corrupted and clean activations, yielding a more accurate attribution score. We next consider optimization-based approaches: IB-Circuit (Bian et al. 2026) performs circuit discovery by optimizing an information bottleneck objective to identify compact, task-relevant subgraphs. Pruning (Prune.) (Bhaskar et al. 2024) applies gradient-based pruning to remove less important edges; we adapt this method to operate at the node level for fair comparison. Random serves as a lower bound by randomly selecting k nodes as the circuit.

## RQ1: Performance of S³martCirc

In Table 1, we compare S³martCirc to current circuit discovery methods and observe the following:

Efectiveness. S³martCirc outperforms all baselines by substantial margins across all tasks and models. For example, on Acronym with GPT-2, S³martCirc achieves an accuracy drop of -97.50%, compared to -54.67% for the strongest baseline, Activation Patching. These results indicate that an optimizable mask is more efective in identifying nodes critical to model computation than patching-based methods, which evaluate nodes independently and potentially miss important interactions between components. Furthermore, although Prune. and IBCircuit also employ continuous masking strategies, both methods perform substantially worse than S³martCirc. This suggests that incorporating functional interpretation provides essential guidance for identifying circuit nodes, beyond what is achievable through masking alone. Circuit Size. Our goal is to identify a minimal circuit that achieves maximal accuracy drop and logit diference. While S³martCirc does not consistently produce the smallest circuits, it achieves substantially greater performance degradation for comparable circuit sizes, indicating more efective node selection than baseline methods. For example, on the

Table 1: Performance of S³martCirc compared to baselines over three runs. Best results are shown in bold, and runner-up results are italicized. Empty cells indicate that no circuit could be found within the set of nodes selected by the method. An asterisk (\*) indicates that only one experimental run was completed due to the time limit (48 hours).
<table><tr><td></td><td></td><td colspan="3">Acronym</td><td colspan="3">IOI</td></tr><tr><td></td><td></td><td>Circuit Size</td><td>Acc.</td><td>Logit.</td><td>Circuit Size</td><td>Acc.</td><td>Logit.</td></tr><tr><td rowspan="9"></td><td>Act.</td><td> $3 5 . 0 0 { \scriptstyle \pm 4 . 3 2 }$ </td><td> $- 5 4 . 6 7 _ { \pm 3 8 . 3 3 }$ </td><td> $- 3 . 5 4 { \scriptstyle \pm 3 8 . 3 3 }$ </td><td> $3 9 . 3 3 _ { \pm 1 0 . 4 0 }$ </td><td> $- 7 0 . 7 1 _ { \pm 2 5 . 8 9 }$ </td><td> $- 9 . 1 1 _ { \pm 2 5 . 8 9 }$ </td></tr><tr><td>Attr.</td><td> $3 7 . 3 3 { \scriptstyle \pm 2 . 0 5 }$ </td><td> $- 0 . 1 7 _ { \pm 0 . 2 4 }$ </td><td> $0 . 4 6 _ { \pm 0 . 2 4 }$ </td><td> $4 8 . 0 0 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $- 1 2 . 7 3 { \scriptstyle \pm 8 . 0 1 }$ </td><td> $- 5 . 1 4 _ { \pm 8 . 0 1 }$ </td></tr><tr><td>Attr. IG</td><td> $4 9 . 6 7 _ { \pm 1 1 . 4 4 }$ </td><td> $- 2 . 0 0 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $- 0 . 5 9 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $\mathbf { 3 8 . 0 0 { \scriptstyle \pm 5 . 1 0 } }$ </td><td> $- 0 . 1 7 _ { \pm 0 . 2 4 }$ </td><td> $- 0 . 2 9 _ { \pm 0 . 2 4 }$ </td></tr><tr><td>GPT-2 IBCircuit</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prune.</td><td> $\mathbf { 3 2 . 3 3 _ { \pm 3 7 . 9 7 } }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 7 _ { \pm 0 . 0 0 }$ </td><td> $3 8 . 3  { \beta } _ { \pm 3 3 . 8 1 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 2 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>Random</td><td> $6 0 . 3 3 { \scriptstyle \pm 7 . 7 6 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 4 _ { \pm 0 . 0 0 }$ </td><td> $7 7 . 3 3 _ { \pm 3 . 4 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 1 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td> $\mathrm { S ^ { 3 } m a r t C i r c }$ </td><td> $6 0 . 3 3 { \scriptstyle \pm 7 . 9 3 }$ </td><td> $\mathbf { - 9 7 . 5 0 { \scriptstyle \pm 1 . 0 8 } }$ </td><td> $\mathbf { - 5 . 0 0 { \scriptstyle \pm 1 . 0 8 } }$ </td><td> $7 6 . 0 0 { \scriptstyle \pm 5 . 7 2 }$ </td><td> $\mathbf { - 9 5 . 9 9 { \scriptstyle \pm 2 . 8 5 } }$ </td><td> $\mathbf { - 1 0 . 4 2 { \scriptstyle \pm 2 . 8 5 } }$ </td></tr><tr><td>Act.</td><td></td><td></td><td></td><td> $1 3 . 0 0 _ { \pm 2 . 0 0 }$ </td><td> $- 5 9 . 1 7 _ { \pm 4 1 . 9 0 }$ </td><td> $- 2 . 3 7 _ { \pm 4 1 . 9 0 }$ </td></tr><tr><td>Attr.</td><td></td><td></td><td></td><td> $7 1 . 6 7 _ { \pm 2 . 0 5 }$ </td><td> $- 1 . 5 0 _ { \pm 0 . 4 1 }$ </td><td> $- 0 . 3 5 _ { \pm 0 . 4 1 }$ </td></tr><tr><td rowspan="6"></td><td>Attr. IG</td><td> $5 4 . 6 7 { \scriptstyle \pm 2 . 6 2 }$ </td><td> $- 2 . 1 7 _ { \pm 0 . 2 4 }$ </td><td> $\ - 0 . 3 4 \pm 0 . 2 4$ </td><td> $4 7 . 6 7 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $- 5 . 6 7 _ { \pm 3 . 7 0 }$ </td><td> $- 0 . 3 5 { \scriptstyle \pm 3 . 7 0 }$ </td></tr><tr><td>Llama IBCircuit</td><td> $\mathbf { 2 9 . 6 7 _ { \pm 1 2 . 6 6 } }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 8 _ { \pm 0 . 0 0 }$ </td><td> ${ \bf 7 . 3 3 _ { \pm 0 . 9 4 } }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>Prune.</td><td> $7 0 . 6 7 { \scriptstyle \pm 3 . 0 9 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 1 _ { \pm 0 . 0 0 }$ </td><td> $7 0 . 6 7 _ { \pm 3 . 0 9 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>Random</td><td> $6 0 . 0 0 { \scriptstyle \pm 9 . 9 3 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $6 2 . 3 3 { \scriptstyle \pm 6 . 1 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td> $\mathrm { S ^ { 3 } m a r t C i r c }$ </td><td> $\it 4 7 . 3 3 2 . 0 5$ </td><td> $\mathbf { - 5 7 . 3 3 _ { \pm 3 . 4 0 } }$ </td><td> $\mathbf { - 3 . 1 7 _ { \pm 3 . 4 0 } }$ </td><td> $3 5 . 0 0 { \scriptstyle \pm 3 . 7 4 }$ </td><td> $\mathbf { 6 9 . 0 0 _ { \pm 4 2 . 4 3 } }$ </td><td> $\mathbf { - 2 . 9 6 _ { \pm 4 2 . 4 3 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Qwen</td><td>Act. Attr.</td><td> $3 2 . 0 0 _ { \pm 0 . 0 0 } ^ { \mathrm { ~ ~ } } ^ { \ast }$ </td><td> $\mathbf { - 1 0 0 . 0 0 _ { \pm 0 . 0 0 } } ^ { \mp }$ </td><td> $\Gamma \mathopen { } \mathclose \bgroup \left. \begin{array} { r l } { - 1 . 0 0 _ { \pm 0 . 0 0 } } & { \aftergroup \egroup \right. \mathclose \bgroup \left. \left. \begin{array} { r l } \end{array} \aftergroup \egroup \right. \kern - delimiterspace _ { - \infty } \right. } \end{array}$ </td><td> $\mathbf { 1 2 . 0 0 _ { \pm 0 . 0 0 } } ^ { \ast }$ </td><td> $- 3 3 . 0 0 _ { \pm 0 . 0 0 } ^ { \ast }$ </td><td> $- 0 . 3 3 _ { \pm 0 . 0 0 } ^ { \ ^ { \ast } }$ </td></tr><tr><td>Attr. IG</td><td> $6 4 . 3 3 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $- 0 . 1 7 _ { \pm 0 . 2 4 }$ </td><td> $- 0 . 4 8 _ { \pm 0 . 2 4 }$ </td><td> $2 9 . 0 0 { \scriptstyle \pm 3 . 5 6 }$ </td><td> $- 0 . 5 0 _ { \pm 0 . 4 1 }$ </td><td> $- 0 . 9 1 _ { \pm 0 . 4 1 }$ </td></tr><tr><td></td><td> $3 2 . 3 3 { \scriptstyle \pm 8 . 3 4 }$ </td><td> $- 0 . 1 7 _ { \pm 0 . 2 4 }$ </td><td> $- 0 . 1 2 _ { \pm 0 . 2 4 }$ </td><td> $3 5 . 0 0 { \scriptstyle \pm 5 . 3 5 }$ </td><td> $- 0 . 3 3 _ { \pm 0 . 4 7 }$ </td><td> $- 0 . 1 9 _ { \pm 0 . 4 7 }$ </td></tr><tr><td>IBCircuit</td><td> $1 4 . 6 \tilde { 7 } _ { \pm 0 . 9 4 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 2 9 { \scriptstyle \pm 0 . 0 0 }$ </td><td></td><td></td><td></td></tr><tr><td>Prune.</td><td> $4 6 . 0 0 { \scriptstyle \pm 4 . 2 4 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 1 _ { \pm 0 . 0 0 }$ </td><td> $4 6 . 0 0 { \scriptstyle \pm 4 . 2 4 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>Random</td><td> $5 2 . 6 7 { \scriptstyle \pm 9 . 7 4 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $- 0 . 0 0 _ { \pm 0 . 0 0 }$ </td><td> $4 8 . 0 0 { \scriptstyle \pm 6 . 9 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td> $\mathrm { S ^ { 3 } m a r t C i r c }$ </td><td> $\mathbf { 1 2 . 0 0 _ { \pm 1 . 6 3 } }$ </td><td> $\mathbf { - 1 0 0 . 0 0 _ { \pm 0 . 0 0 } }$ </td><td> $\mathbf { - 1 0 . 9 7 { \scriptstyle \pm 0 . 0 0 } }$ </td><td> $I 6 . 3 3 _ { \pm 4 . 5 0 }$ </td><td> $\mathbf { - 1 0 0 . 0 0 _ { \pm 0 . 0 0 } }$ </td><td> $\mathbf { - 1 2 . 0 1 _ { \pm 0 . 0 0 } }$ </td></tr></table>

IOI task with Llama, S³martCirc achieves a -69.00% accuracy drop using a circuit of 35 nodes. In contrast, Attr. IG produces a larger circuit (average size 47.67 nodes) but yields only $\mathrm { ~ a ~ } { - } 5 . 6 7 \%$ accuracy drop. This shows that S³martCirc identifies more functionally critical nodes and, again, implies the benefit of incorporating functional interpretation.

## RQ2: Recovering Prior Circuits

Previous work (Wang et al. 2022) manually identified and verified a ground-truth circuit for the IOI task through extensive analysis and validation. This circuit serves as a reliable reference for evaluating whether automated methods can recover the same critical components. Accordingly, a desirable property of circuit discovery methods is the ability to recover nodes from this known IOI circuit. Figure 3 reports the percentage of nodes from the original IOI circuit that are recovered by each method within the final end-to-end discovered circuits. We evaluate this across varying circuit sizes obtained by thresholding at 50, 100, 150, and 200 nodes. Note that perfect recovery (100%) is not achievable due to the vast number of possible nodes to choose from.

its performance is comparable only to Activation Patching. In contrast, other baselines, particularly optimization-based methods, achieve substantially lower recovery rates, even when allowed to select up to 200 nodes. These results provide evidence that S³martCirc not only improves circuit discovery performance (RQ1), but also aligns more closely with human-verified mechanistic interpretations of the IOI task. This increases the credibility of its discovered circuits, particularly in settings where ground-truth labels are unavailable.

Overall, S³martCirc consistently recovers a higher percent of the original circuit compared to most baselines, and

We further validate the performance of S³martCirc by comparing its functional interpretations against the groundtruth circuit for the Acronym Prediction task (García-Carrasco, Maté, and Trujillo 2024b), shown in Figure 4. In the figure, each node is labeled as layer.head (e.g., 8.11 represents layer 8, attention head 11). S³martCirc successfully recovers four of the eight nodes (50%) from the original circuit, and its functional classifications align with the mapping shown in Figure 1. Specifically, Previous Token node 1.0 is correctly identified as a passthrough node, while Letter Mover node 11.4 is correctly classified as a functional node. This alignment, again, indicates that S³martCirc not only identifies important nodes but also accurately captures their mechanistic roles, providing automated interpretations that correspond closely to those derived through human analysis.

![](images/a2321558db9d69be981f627075b8236792fd27f54ca005fa8030bc3e4c1c1eea.jpg)  
Figure 3: Percent of the IOI circuit recovered by diferent circuit discovery methods

However, an apparent discrepancy arises in the classification of nodes 8.11 and 10.10. S³martCirc classifies 8.11 as a passthrough node while classifying 10.10 as a functional node, even though prior work shows that both heads can act as either Bridge nodes or Letter Mover nodes depending on the specific intervention applied. This divergence highlights a fundamental diference in methodology: human-verified approaches evaluate node behavior under carefully isolated interventions to determine what functions a head can perform, while $S ^ { 3 }$ martCirc measures the dominant contribution of each head across multiple samples. In other words, ${ \mathbf S } ^ { 3 } ]$ martCirc assigns labels based on the function that most strongly influences the circuit’s overall behavior. The differing classifications therefore reflect the multi-functional nature of these heads rather than a misclassification.

## RQ3: Parameter Analysis

To answer RQ3, we perform a parameter analysis, shown in Figure 5, on the IOI task using GPT-2 to evaluate the sensitivity of S³martCirc to two key hyperparameters: α, which balances the two objectives in the Important Node Discovery stage, and $k ,$ which controls the number of steps in the Important Node Classification stage. Specifically, each iteration consists of one discovery step followed by k classification steps. We evaluate both accuracy (Acc.) and the percentage of recovered nodes from the original IOI circuit across k ∈ {2, 4, 6, 8, 10} and α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.

Figure 5a demonstrates that performance generally improves as k increases, and stabilizes at $k = 6$ . The accuracy drop decreases from -71% at $k = 2$ to approximately -96% at $k = 6 ,$ , and percent circuit recovery monotonically increases. The consistent improvement suggests that additional refinement steps in the classification stage improve the identification of important nodes and the circuit quality.

Figure 5b reveals that $\alpha = 0 . 7 5$ achieves the best performance in terms of both accuracy and circuit recovery. This indicates that, for the IOI task, the KL divergence objective should be weighted more heavily than the task loss during circuit discovery. Nevertheless, the cross-entropy term remains important, as performance degrades sharply when it is removed (α = 1.0). This helps explain the weaker performance of baselines, which rely primarily on output distribution diferences (e.g., KL divergence) to identify important nodes. Our results suggest that, for IOI, combining output faithfulness with task performance signals is more efective, as some critical components contribute to correct behavior without inducing large changes in logits.

![](images/6d10a2fbb6d5c99529ba78fc35f7230b4f8f50c22582c873b13f9b4b9ecfbb8a.jpg)  
Figure 4: Comparison between the Acronym circuit and the functional interpretations determined by $\check { \mathbf { S } } ^ { 3 } \mathrm { _ { 1 } }$ martCirc.

## RQ4: Ablation Study

We perform an ablation study to understand how each component of ${ \mathbf S } ^ { 3 } ]$ martCirc contributes to its overall performance. Figure 6 compares the accuracy diference achieved by the original S³martCirc method against three variants: S³martCirc-NW (No Warmup), which removes the initial warmup period and directly starts the alternating training process; S³martCirc-DF (Discovery First), which completes the entire Important Node Discovery stage before beginning Important Node Classification; and S³martCirc-IF (Interpretation First), which reverses the order by performing Important Node Classification before Important Node Discovery. These ablations isolate the contributions of the warmup period and the alternating training strategy.

Importance of Alternating Training. S³martCirc-DF performs substantially worse than the other variants, and achieves near-zero accuracy drop on Acronym (-4.33% compared to -97.5% for S³martCirc). This sharp degradation suggests that identifying all important nodes prior to functional interpretation leads to an overly broad candidate set, making it dificult to distinguish truly critical nodes from less relevant ones. As a result, applying functional interpretation only after discovery becomes less efective, as it is overwhelmed by noise from the large set of candidate nodes. This leads to circuits that include non-critical components while omitting key ones.

S³martCirc-IF performs significantly better than

![](images/a7a2f525a93a870b0e8521adde25bd2d68878eac3b10cb063bc20c5239c44486.jpg)  
(a)

![](images/bf474d4a39abf211d3a77326548ff7e7ba45d3b0ccba5e3ddf84d3206496c03a.jpg)  
(b)

Figure 5: Efect of α and k on $S ^ { 3 }$ martCirc Performance.  
![](images/5086f7e621850c3eabfb6b2a1e84db015fdafc9c5d54226398c4b42e132b9505.jpg)  
Figure 6: Ablation study to evaluate the efectiveness of different components in ${ \mathsf { S } } ^ { \mathsf { \check { 3 } } }$ martCirc and its training process.

S³martCirc-DF (approximately 88% on IOI and 42% on Acronym), indicating that incorporating functional information into the discovery process provides useful guidance for identifying important nodes. However, it still underperforms S³martCirc. This demonstrates that, while functional interpretation is beneficial as a prior, it is insuficient without the alternating training.

Overall, these results highlight the importance of bidirectional coupling between Important Node Discovery and Important Node Classification. The alternating optimization allows each stage to refine the other: discovery benefits from evolving functional understanding, while classification operates on progressively more focused candidate sets.

Role of Warmup Period. The removal of the warmup period (S³martCirc-NW) afects performance consistently across tasks (67% on IOI and 66% on Acronym). This suggests that directly starting alternating optimization from randomly initialized masking coeficients provides an unreliable signal for early-stage updates. In particular, without an initial phase that identifies a reasonable set of candidate nodes, the model struggles to generate informative gradients for distinguishing important from unimportant components. As a result, the optimization process is less stable and converges to suboptimal circuits, making it more dificult to recover the underlying task-relevant structure.

## Related Works

Mechanistic Interpretability and Circuit Discovery. Mechanistic interpretability (MI) explains model behavior by identifying the components responsible for it (Bereska and Gavves 2024; Sharkey et al. 2025; Rai et al. 2024), with the circuit as its central abstraction. Most work follows a where-then-what paradigm (Méloux et al. 2025b; García-Carrasco, Maté, and Trujillo 2024a; Chandna, Bashir, and Sen 2025), localizing components via patching-based interventions (Wang et al. 2022; Meng et al. 2022; García-Carrasco, Maté, and Trujillo 2024b; Heimersheim and Nanda 2024) or gradient-based approximations (Syed, Rager, and Conmy 2023; Hanna, Pezzelle, and Belinkov 2024) and then interpreting their roles manually. Such interventions are costly and evaluate components in isolation, while the interpretation step is subjective and hard to scale. Recent work further shows that functional roles are not task-specific: components are reused across tasks with consistent roles (Merullo, Eickhof, and Pavlick 2024), roles can be localized to individual neurons (Nikankin et al. 2026), and structurally distinct circuits can be functionally equivalent (Haklay et al. 2026). This motivates the abstract, quantitative notion of role that S³martCirc folds directly into discovery.

Optimization-Based Circuit Discovery. A complementary line optimizes continuous masks over components under sparsity constraints (Frankle and Carbin 2019; Bhaskar et al. 2024), e.g., IBCircuit’s information bottleneck objective (Bian et al. 2026) and behavior-specific mask learning (Li and Janson 2024). These better capture component interactions than patching, but most model only which components matter, not how they contribute. For example, IBCircuit formulates circuit extraction as an optimization problem balancing behavioral fidelity with an information bottleneck objective (Bian et al. 2026), while other approaches optimize masks to isolate behavior-specific components (Li and Janson 2024). S³martCirc difers in that (1) roles come from a task-agnostic, quantitative metric on activations rather than hand-specified desiderata or ablation signs; (2) importance and role are optimized jointly by alternating between stages, not read of a single mask; and (3) discovery spans both attention heads and MLP neurons.

## Conclusion

We introduce S³martCirc, a unified mechanistic interpretability framework that jointly discovers circuits and interprets node functionality in LLMs. We propose a generalized dichotomy of abstract functional roles that applies across tasks, together with a quantitative metric that integrates these interpretations into an end-to-end optimizable circuit discovery process. This enables S³martCirc to directly model the interdependence between node importance and functional role. Extensive experiments on GPT-2, Llama 3.2, and Qwen 3 demonstrate that S³martCirc substantially outperforms existing methods in identifying compact, taskrelevant circuits and recovering human-verified circuits, advancing automated and scalable MI. Our framework also has limitations that point to future work: the two roles are intentionally coarse and do not recover the fine-grained roles of manual analysis, and because roles are defined relative to a task’s activations rather than intrinsically, the same node may be labeled diferently across tasks.

## References

Basyal, L.; and Sanghvi, M. 2023. Text summarization using large language models: a comparative study of mpt-7b-instruct, falcon-7b-instruct, and openai chat-gpt models. arXiv preprint arXiv:2310.10449.

Bereska, L.; and Gavves, E. 2024. Mechanistic interpretability for AI safety–a review. arXiv preprint arXiv:2404.14082.

Bhaskar, A.; Wettig, A.; Friedman, D.; and Chen, D. 2024. Finding transformer circuits with edge pruning. Advances in Neural Information Processing Systems, 37: 18506–18534.

Bian, T.; Niu, Y.; Yuan, C.; Piao, C.; Wu, B.; Huang, L.-K.; Rong, Y.; Xu, T.; Cheng, H.; and Li, J. 2026. IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck. arXiv preprint arXiv:2602.22581.

Chandna, B.; Bashir, Z.; and Sen, P. 2025. Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective. arXiv preprint arXiv:2506.05166.

Chen, B.; Fu, Y.; Xu, G.; Xie, P.; Tan, C.; Chen, M.; and Jing, L. 2021. Probing BERT in Hyperbolic Spaces. In ICLR.

Feng, Z.; Zhang, Y.; Li, H.; Wu, B.; Liao, J.; Liu, W.; Lang, J.; Feng, Y.; Wu, J.; and Liu, Z. 2024. Tear: Improving llmbased machine translation with systematic self-refinement. arXiv preprint arXiv:2402.16379.

Frankle, J.; and Carbin, M. 2019. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In International Conference on Learning Representations.

García-Carrasco, J.; Maté, A.; and Trujillo, J. 2024a. Detecting and understanding vulnerabilities in language models via mechanistic interpretability. arXiv preprint arXiv:2407.19842.

García-Carrasco, J.; Maté, A.; and Trujillo, J. C. 2024b. How does gpt-2 predict acronyms? extracting and understanding a circuit via mechanistic interpretability. In International Conference on Artificial Intelligence and Statistics, 3322– 3330. PMLR.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Haklay, T.; Prakash, N.; Pandey, S.; Torralba, A.; Mueller, A.; Andreas, J.; Shaham, T. R.; and Belinkov, Y. 2026. Pitfalls in Evaluating Interpretability Agents. arXiv preprint arXiv:2603.20101.

Hanna, M.; Pezzelle, S.; and Belinkov, Y. 2024. Have faith in faithfulness: Going beyond circuit overlap when finding model mechanisms. arXiv preprint arXiv:2403.17806.

He, Z.; Zhao, H.; Qiao, Y.; Yang, F.; Payani, A.; Ma, J.; and Du, M. 2025. Saif: A sparse autoencoder framework for interpreting and steering instruction following of language models. arXiv preprint arXiv:2502.11356.

Heimersheim, S.; and Nanda, N. 2024. How to use and interpret activation patching. arXiv preprint arXiv:2404.15255.

Hoover, B.; Strobelt, H.; and Gehrmann, S. 2020. exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformer Models. In ACL.

Kleidermacher, H. C.; and Zou, J. 2025. Science across languages: Assessing llm multilingual translation of scientific papers. arXiv preprint arXiv:2502.17882.

Li, M.; and Janson, L. 2024. Optimal ablation for interpretability. Advances in Neural Information Processing Systems, 37: 109233–109282.

Li, Z.; Fan, S.; Gu, Y.; Li, X.; Duan, Z.; Dong, B.; Liu, N.; and Wang, J. 2024. Flexkbqa: A flexible llm-powered framework for few-shot knowledge base question answering. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 18608–18616.

Liu, Q.; Mao, J.; and Wen, J.-R. 2025. How do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective. arXiv preprint arXiv:2504.07898.

Maintainers, H. C. M. 2022. gpt2 (Revision 909a290).

Méloux, M.; Maniu, S.; Portet, F.; and Peyrard, M. 2025a. Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? In The Thirteenth International Conference on Learning Representations.

Méloux, M.; Maniu, S.; Portet, F.; and Peyrard, M. 2025b. Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? arXiv preprint arXiv:2502.20914.

Meng, K.; Bau, D.; Andonian, A.; and Belinkov, Y. 2022. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35: 17359–17372.

Merullo, J.; Eickhof, C.; and Pavlick, E. 2024. Circuit component reuse across tasks in transformer language models. In International Conference on Learning Representations, volume 2024, 18349–18377.

Meta. 2024. Llama-3.2-1B.

Nanda, N.; Chan, L.; Lieberum, T.; Smith, J.; and Steinhardt, J. 2023. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217.

Nikankin, Y.; Arad, D.; Gandelsman, Y.; and Belinkov, Y. 2026. Same task, diferent circuits: Disentangling modalityspecific mechanisms in vlms. Advances in Neural Information Processing Systems, 38: 66352–66385.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; and Sutskever, I. 2019. Language Models are Unsupervised Multitask Learners.

Rai, D.; Zhou, Y.; Feng, S.; Saparov, A.; and Yao, Z. 2024. A practical review of mechanistic interpretability for transformer-based language models. arXiv preprint arXiv:2407.02646.

Sharkey, L.; Chughtai, B.; Batson, J.; Lindsey, J.; Wu, J.; Bushnaq, L.; Goldowsky-Dill, N.; Heimersheim, S.; Ortega, A.; Bloom, J.; et al. 2025. Open problems in mechanistic interpretability. arXiv preprint arXiv:2501.16496.

Singh, C.; Inala, J. P.; Galley, M.; Caruana, R.; and Gao, J. 2024. Rethinking interpretability in the era of large language models. arXiv preprint arXiv:2402.01761.

Syed, A.; Rager, C.; and Conmy, A. 2023. Attribution patching outperforms automated circuit discovery. arXiv preprint arXiv:2310.10348.

Tan, Y.; Min, D.; Li, Y.; Li, W.; Hu, N.; Chen, Y.; and Qi, G. 2023. Can ChatGPT replace traditional KBQA models? An in-depth analysis of the question answering performance of the GPT LLM family. In International Semantic Web Conference, 348–367. Springer.

Tatsat, H.; and Shater, A. 2025. Beyond the black box: Interpretability of llms in finance. arXiv preprint arXiv:2505.24650.

Team, Q. 2025a. Qwen3-1.7B.

Team, Q. 2025b. Qwen3 Technical Report. arXiv:2505.09388.

Van Veen, D.; Van Uden, C.; Blankemeier, L.; Delbrouck, J.-B.; Aali, A.; Bluethgen, C.; Pareek, A.; Polacin, M.; Reis, E. P.; Seehofnerova, A.; et al. 2023. Clinical text summarization: adapting large language models can outperform human experts. Research Square, rs–3.

Wagner, N.; Desmond, M.; Nair, R.; Ashktorab, Z.; Daly, E. M.; Pan, Q.; Cooper, M. S.; Johnson, J. M.; and Geyer, W. 2024. Black-box uncertainty quantification method for llm-as-a-judge. arXiv preprint arXiv:2410.11594.

Wang, K.; Variengien, A.; Conmy, A.; Shlegeris, B.; and Steinhardt, J. 2022. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022. URL https://arxiv.org/abs/2211.00593, 2.

Yang, H.; Li, M.; Zhou, H.; Xiao, Y.; Fang, Q.; and Zhang, R. 2023a. One llm is not enough: Harnessing the power of ensemble learning for medical question answering. medRxiv.

Yang, R.; Tan, T. F.; Lu, W.; Thirunavukarasu, A. J.; Ting, D. S. W.; and Liu, N. 2023b. Large language models in health care: Development, applications, and challenges. Health Care Science.

## Implementation Details

## Computation Resources

All experiments were performed on clusters of NVIDIA RTX A4000 16GB and NVIDIA A16 16GB GPUs except for Attribution Patching and Attribution Patching IG on Qwen. Due to the large model size, these experiments were done on clusters of NVIDIA RTX A6000 48GB GPUs.

## Data Generation

We generate the data samples for each experiment due to the need to maintain token limits based on each model’s tokenizer. Below describes the data generation process for each task. given a tokenizer.

IOI. We adopt the names and place/object pairs from (Wang et al. 2022). For each model, we keep only the names and place/object pairs that tokenize as a single token, yielding a valid set of values. We then enumerate all valid combinations and insert each into the template “When [NAME\_1] and [NAME\_2] went to the [PLACE], [NAME\_1] handed the [OBJECT] to”. Corrupted counterparts are formed by swapping the second name, producing clean/corrupted pairs. Finally, to obtain a balanced set of 1000 samples, we draw 1000/p samples from each place/object pair, where p is the number of valid pairs.

Acronym. We generate samples using the RandomWord class from the wonderwords package. For each letter of the alphabet, we sample 500 words and keep a word only if its formatted version (i.e., prepending a space and capitalizing the first letter) tokenizes into exactly two tokens, the first of which is the space-prefixed capital letter. Letters with no valid words are discarded, leaving a set of valid letters. Within this set, we identify valid abbreviations, i.e., three-letter sequences whose concatenation tokenizes into three tokens. For each pair of valid abbreviations, we sample random words matching their letters and insert them into the template “The [A\_RAND\_WORD] [B\_RAND\_WORD] [C\_RAND\_WORD] (AB” to form clean and corrupted samples. Finally, we shufle the generated data and take the first 1000 samples.

## Hyperparameters

Below, we list the hyperparameters used for each model dataset combination in Table 1. All experiments used a random seed of 42.

• Acronym with GPT-2 - Learning rate is 0.01, weight decay is 0.005, number of epochs is 60, number of warmup epochs is 60, sparsity coeficient is 0.5, and the logit coeficient (alpha) is 0.25.

• IOI with GPT-2 - Learning rate is 0.005, weight decay is 0.0005, number of epochs is 60, number of warmup epochs is 5, sparsity coeficient is 0.3, and the logit coefficient is 0.75.

• Acronym with Llama - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 5, sparsity coeficient is 1.0, and the logit coefficient is 0.75.

Table 2: Runtime comparison between baselines and S³martCirc (in seconds)
<table><tr><td></td><td></td><td>Act.</td><td>Attr.</td><td>Attr. IG</td><td>IBCircuit</td><td>Prune.</td><td>Random</td><td>SªmartCirc</td></tr><tr><td rowspan="2">GPT-2</td><td>Acronym</td><td>6239.72</td><td>299.54</td><td>3452.48</td><td>426.70</td><td>3391.87</td><td>3.98</td><td>197.14</td></tr><tr><td>IOI</td><td>7943.71</td><td>267.94</td><td>2801.64</td><td>467.19</td><td>3732.88</td><td>0.13</td><td>264.44</td></tr><tr><td rowspan="2">Llama</td><td>Acronym</td><td>61510.09</td><td>1114.46</td><td>12770.96</td><td>1266.76</td><td>11470.94</td><td>2.22</td><td>507.67</td></tr><tr><td>IOI</td><td>64639.54</td><td>1128.62</td><td>12344.18</td><td>1617.78</td><td>11645.51</td><td>3.37</td><td>901.02</td></tr><tr><td rowspan="2">Qwen</td><td>Acronym</td><td>168689.44</td><td>829.43</td><td>11151.09</td><td>3845.65</td><td>19072.51</td><td>3.54</td><td>982.16</td></tr><tr><td>IOI</td><td>187663.88</td><td>783.17</td><td>11148.12</td><td>5152.79</td><td>21000.88</td><td>2.36</td><td>1047.59</td></tr></table>

![](images/2e6c30936f6a5f258dfef9ab675c22435ca29b4d1bcfe049406f730f93cbd355.jpg)  
Figure 7: Percentage of the Acronym circuit recovered by diferent circuit discovery methods

• IOI with Llama - Learning rate is 0.001, weight decay is 0.005, number of epochs is 75, number of warmup epochs is 5, sparsity coeficient is 1.0, and the logit coeficient is 0.25.

• Acronym with Qwen - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 5, sparsity coeficient is 0.7, and the logit coefficient is 0.75.

• IOI with Qwen - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 10, sparsity coeficient is 0.9, and the logit coeficient is 0.75.

## Run time

In Table 2, we compare the runtime of S³martCirc to the baselines. From the table, we see that S³martCirc is relatively eficient, achieving high performance without requiring an excessive amount of time.

## Additional Experiment Results

We also repeated the first experiment in Section 4.3 for the acronym task and show the results in Figure 7. We see that our claims still hold; S³martCirc maintains the largest percent recovered compared to the baselines.