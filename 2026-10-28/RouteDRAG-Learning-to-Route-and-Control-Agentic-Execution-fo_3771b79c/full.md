# RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG

Zekai Chen\* Beijing Institute of Technology Beijing, China

Xunkai Li Beijing Institute of Technology Beijing, China

Hao Zhu\* Beijing Institute of Technology Beijing, China

Zejian Kang Zhejiang University Hangzhou, China Westlake University Hangzhou, China

Quanjiang Li National University of Defense Technology Changsha, China

Yu Liu Chinese Academy of Sciences China

Rui Qian Fudan University Shanghai, China

Rong-Hua Li Beijing Institute of Technology Beijing, China

## Abstract

Web-based applications increasingly rely on understanding long multimodal documents, where answering questions often requires combining information across pages, tables, and figures. Agentic retrieval-augmented generation addresses these tasks through iterative retrieval and reasoning. However, Agentic execution is not always necessary, introducing additional computation and potentially incorrect answers. Our empirical analysis reveals that Agentic execution provides query-dependent benefits, while additional operations may improve, preserve, or degrade answer accuracy. These findings motivate controlling both Agent invocation and subsequent execution. To address these challenges, we propose RouteDRAG, a learning-based framework comprising four components. Unified Document Index organizes multimodal content into a graph-backed index and derives query represen tations from initial retrieval. Adaptive Path Routing preserves exact answering and predicts the relative benefit of Agent over shorter execution paths. Dynamic Agent Execution maintains an evolving workspace and refines Planner proposals to control tool use and termination. Sequential Policy Learning first trains operation control from observed outcomes and then trains path routing using the resulting Agent’s performance. Across two benchmarks, RouteDRAG achieves an average accuracy improvement of 3.47 percentage points over shorter-path-only execution while assigning only 79.3% of queries to Agent. Notably, it also surpasses uniform Agent execution on both benchmarks while requiring fewer answering calls. Our code is now available on https: //github.com/Ereshkigal1/RouteDRAG.

## CCS Concepts

• Information systems → Information retrieval.

Keywords multimodal document question answering, retrieval-augmented generation, selective execution, Agentic evidence acquisition

ACM Reference Format: Zekai Chen\*, Hao Zhu\*, Yu Liu, Xunkai Li, Zejian Kang, Rui Qian, Quanjiang Li, and Rong-Hua Li. 2027. RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG. In Proceedings ofThe Web Conference 2027 (WWW’27). ACM, New York, NY, USA, 13 pages.

## 1 Introduction

Web-based applications, such as financial analysis platforms [5, 30], government information portals, academic search engines [7], and online technical support [3, 11], increasingly rely on understanding long and visually rich documents. Answering questions often requires locating and combining information distributed across pages, tables, and figures. For example, explaining a company’s financial performance may require examining numerical results in a table and their explanations on other pages. A single retrieval step may locate only part of the required information, leaving the question unanswered. Agentic retrieval-augmented generation (Agentic RAG) addresses this challenge through iterative retrieval, visual understanding, and multi-step reasoning, allowing information from diferent document regions to be located and combined.

Recent studies have made Agentic RAG more adaptive through multimodal retrieval, graph-based exploration and iterative reasoning [15, 18, 22, 25, 32]. These methods allow Agents to dynamically select document content and adjust retrieval procedures. However, adapting how information is retrieved does not fully address how much Agentic execution is needed. As illustrated in Fig. 1, some questions require multiple retrieval steps to locate and combine information across diferent pages, while others can be answered directly from the initial retrieval results without invoking an Agent. Unnecessary Agent invocation increases computational cost and may introduce irrelevant or conflicting information, leading to incorrect answers. Yet deciding whether to invoke an Agent is only the first step. As the Agent gathers more information, it must further decide whether to retrieve additional content, inspect visual regions, or terminate execution. This raises two fundamental questions: when should Agentic execution be invoked, and how should its subsequent operations be controlled?

Our empirical analysis reveals three observations about Agentic execution. O1: Limited Overall Gains. Agent improves accuracy by 1.00 and 3.12 percentage points on DocBench and MMLongBench-Doc, respectively, but uses 1.65× and 1.81× as many answering tokens as Standard. O2: Query-Dependent Benefits. Agent alone answers 3.09% and 6.69% of questions correctly, while both Agent and Standard succeed on 69.24% and 44.18%. Conversely, Standard alone succeeds on 2.09% and 3.57%, indicating that Agent execution is not universally beneficial. O3: Non-Monotonic Operation Gains. Among 917 matched operation-level comparisons, 84.51% show unchanged correctness, while additional operations can improve or degrade answers. These observations motivate selective Agent invocation based on its expected benefit and adaptive operation control based on information collected during execution.

To address these challenges, we propose RouteDRAG, a learning-based framework for selective Agentic document RAG. First, Unified Document Index organizes multimodal content into a graph-backed index and derives query representations from initial retrieval. Next, Adaptive Path Routing predicts the benefit of invoking Agent over a shorter path while preserving exact answering. When Agent is invoked, Dynamic Agent Execution maintains an evolving workspace and refines Planner proposals to control subsequent operations and termination. Finally, Sequential Policy Learning first trains operation control from observed outcomes and then trains path routing using the resulting Agent’s performance. Together, these components adapt Agent invocation and execution to individual queries.

Our Contributions. (1) New Perspective. We identify limited overall gains, query-dependent benefits, and non-monotonic operation gains in Agentic document RAG, motivating execution decisions at both query and operation levels. (2) New Framework. We propose RouteDRAG, integrating Unified Document Index, Adaptive Path Routing, Dynamic Agent Execution, and Sequential Policy Learning to control Agent invocation and subsequent operations. (3) SOTA Performance. We evaluate RouteDRAG against nine RAG methods on DocBench and MMLongBench-Doc, achieving 75.86% and 53.80% accuracy, respectively, while reducing answering calls by 16.4% and 14.9% compared with uniform Agent execution.

## 2 Related Work

## 2.1 Multimodal and Agentic Document RAG

Document RAG has evolved from text-based retrieval toward richer multimodal and structural representations. LightRAG [13] organizes entities and relations into a graph, while RAG-Anything [12] extends graph-based retrieval to multimodal documents. GraphReader [17] further employs graph-guided Agent exploration for long-text understanding.

G<sup>2</sup>-Reader [8] maintains evolving content and planning graphs for multimodal document question answering.

VisRAG [29] and M3DocRAG [6] further explore visual retrieval for document understanding.

AgenticOCR [26] enables query-driven parsing of relevant visual regions to reduce redundant document processing.

ReAct [28] established a reasoning-and-acting paradigm that interleaves language-model reasoning with external actions.

Recent Agentic approaches support more flexible information acquisition. ViDoRAG [25] employs iterative exploration and reasoning, LAD-RAG [22] combines neural and structural document retrieval, and MAGE-RAG [32] dynamically explores and prunes graph-based evidence. TAP-RAG [15] further adapts graph expansion and visual inspection to query requirements. These methods improve the acquisition and organization of multimodal information. RouteDRAG instead focuses on whether such Agentic execution is necessary for a given query and how its subsequent operations should be controlled.

## 2.2 Adaptive Retrieval and Execution Selection

Adaptive RAG methods adjust retrieval and execution according to query requirements. Self-RAG [1] learns when retrieval is needed, while Adaptive-RAG [14] selects among no-retrieval, single-step, and iterative strategies based on query complexity.

Active Retrieval Augmented Generation [16] dynamically retrieves additional evidence during generation.

MBA-RAG [24] further uses bandit-based selection to balance retrieval quality and eficiency. More recent methods extend adaptation to multi-step execution. MAO-ARAG [4] dynamically organizes retrieval agents into query-specific workflows, while SPARKLE [9] learns a separate policy for controlling retrieval operations. These studies demonstrate the value of both query-level adaptation and step-wise control. RouteDRAG connects the two decisions by estimating Agent’s benefit relative to a designated shorter path and refining subsequent operations based on the evolving workspace. Its sequential learning process further aligns path routing with the behavior of the resulting Agent.

## 3 Problem Definition

Given a multimodal document � and a query $q ,$ let $\mathcal { K } _ { D }$ denote the unified document index. We consider four execution paths:

$$
C = \{ \mathrm { E x a c t , S t a n d a r d , D i r e c t V L M , A g e n t } \} ,\tag{1}
$$

where the first three provide shorter answering procedures, while Agent supports iterative retrieval and reasoning. Unlike executionpath selection alone, selective Agentic execution involves two related decisions. At the query level, a routing policy $\pi ( q , \mathcal { K } _ { D } )$ determines whether Agent is necessary or a shorter path is suficient. Once Agent is invoked, its execution policy selects an operation $a _ { t }$ from the legal action set $\Gamma _ { t }$ based on the evolving workspace $W _ { t } ,$ , including whether to continue acquiring information or terminate. Our objective is to maintain answer accuracy while avoiding unnecessary Agent invocation and subsequent operations.

## 4 Empirical Study

To understand when Agentic execution is necessary, we investigate its benefits from three perspectives. We use a standalone reference Agent for these analyses, distinct from the final RouteDRAG Actor evaluated in the main experiments. We first examine whether uniform Agent execution justifies its additional cost, then investigate how its benefits vary across queries, and finally assess whether additional operations consistently improve answers.

![](images/f6258db7605047084303d55e67c09cfb2a6a566964e53693dfc044f0f2c3ae07.jpg)  
Figure 1: Why selective Agentic execution? (a) Initial retrieval can sufice, making Agent execution redundant. (b) Others require linking information through Agent execution. (c) Further retrieval can distract an Agent with suficient information.

![](images/fed6f8a93987c1c883690e7196f959ce25ab7d052ec0e5419afcca62ead5bb8e.jpg)  
(a) Overall accuracy–cost

![](images/4605c60abebd9d37655f5d859dfcdbaf88e1600a707308402413f8dfb76bfebc.jpg)  
(b) Query-level outcomes

![](images/ff52991edf875ca34c4014f9b93d64e8ddb0c21878f2373c93526ad5e668a87d.jpg)  
(c) Operation-level outcomes  
Figure 2: Empirical analysis of Agentic execution. (a) Accuracy and answering-token usage across document categories for Standard and Agent. (b) Paired query outcomes, illustrating the query-dependent benefits of Agentic execution. (c) Answer correctness changes after an additional operation compared with immediate termination.

O1: Limited Overall Gains. Does Agentic execution justify its additional computation? Fig. 2(a) compares Standard and Agent across five DocBench domains and seven MMLongBench-Doc document types. Agent improves accuracy by 1.00 and 3.12 percentage points on DocBench and MMLongBench-Doc, respectively, while consuming 1.65× and 1.81× the answering tokens. These results reveal a mismatch between overall accuracy gains and computational cost: uniformly invoking the Agent incurs substantial overhead for modest improvements. However, aggregate results cannot identify which queries justify this additional computation, motivating a closer examination of query-level benefits.

O2: Query-Dependent Benefits. Which queries actually benefit from Agent execution? Fig. 2(b) compares the two paths on the same questions. Correctness remains unchanged for 94.83% of DocBench queries and 89.73% of MMLongBench-Doc queries. Meanwhile, Agent recovers answers missed by Standard on 3.09% and 6.69% of queries, but also turns correct answers into incorrect ones on 2.09% and 3.57%, respectively. Thus, Agent execution is neither universally necessary nor uniformly beneficial. Its value depends on the query, motivating Adaptive Path Routing to estimate the expected benefit of Agent execution over the designated short path.

O3: Non-Monotonic Operation Gains. Even when Agent execution is warranted, does acquiring more information necessarily improve the answer? We examine 917 matched comparisons between immediate FINISH and executing one additional operation before FINISH. As shown in Fig. 2(c), 84.51% of comparisons leave correctness unchanged, while the remaining operations can either improve or degrade answers. In particular, VISUAL operations degrade correctness more often than they improve it, suggesting that additional information may introduce noise rather than useful evidence. This motivates Dynamic Agent Execution, which adapts operation selection to the evolving workspace instead of assuming that continued execution is always beneficial.

## 5 Methodology

## 5.1 Overview

Fig. 3 illustrates the overall framework. RouteDRAG addresses selective Agentic execution through two related decisions: when to invoke Agent and how to control its subsequent operations. These decisions are learned from observed execution outcomes at the query and operation levels. The framework comprises four components: (1) Unified Document Index, (2) Adaptive Path Routing, (3) Dynamic Agent Execution, and (4) Sequential Policy Learning.

## 5.2 Unified Document Index

Motivation. Efective path routing requires considering not only the query but also the document content retrieved before execution. We therefore construct a shared document index and learn a queryconditioned representation that can be reused during execution.

Graph-Backed Document Organization. Following RAG-Anything and LightRAG [12, 13], we organize each multimodal document into the shared index $\mathcal { K } _ { D }$ defined in the problem formulation. The index connects text chunks, entities, relations, visual elements, and structured metadata with their original document locations. It supports four execution paths: Exact, Standard, Direct VLM, and Agent. The document index and pretrained retrieval encoder remain frozen during training.

Query-Conditioned Representation. Given a query �, we retrieve the most relevant text chunks and graph entities, retaining up to eight chunks and four entities. A frozen semantic encoder produces their embeddings, denoted by $\mathbf { e } _ { q } ,$ E<sub>�</sub>, and $\mathbf { E } _ { g } .$ A shared trainable projection $P _ { \psi }$ maps them into a 64-dimensional space. We then obtain query-conditioned summaries through masked attention:

$$
\begin{array} { r } { \mathbf { z } _ { q } = P _ { \psi } ( \mathbf { e } _ { q } ) , \mathbf { z } _ { c } = \mathrm { A t t n } _ { \psi } ( \mathbf { z } _ { q } , P _ { \psi } ( \mathbf { E } _ { c } ) ) , \mathbf { z } _ { g } = \mathrm { A t t n } _ { \psi } ( \mathbf { z } _ { q } , P _ { \psi } ( \mathbf { E } _ { g } ) ) . } \end{array}\tag{2}
$$

Here, the attention operation excludes unavailable entries through a validity mask. To adapt the contribution of each source to the current query, we compute

$$
\pmb { \alpha } = \mathrm { s o f t m a x } \left( G _ { \psi } ( [ \mathbf { z } _ { q } ; \mathbf { z } _ { c } ; \mathbf { z } _ { g } ] ) \right) , \mathbf { h } _ { 0 } = F _ { \psi } \left( [ \alpha _ { q } \mathbf { z } _ { q } ; \alpha _ { c } \mathbf { z } _ { c } ; \alpha _ { g } \mathbf { z } _ { g } ; U _ { \psi } ( \mathbf { u } _ { 0 } ) ] \right) ,\tag{3}
$$

where $\mathbf { u } _ { 0 }$ contains query characteristics, retrieved-content statistics, and similarity scores. $G _ { \psi }$ and $F _ { \psi }$ denote lightweight gating and

fusion networks. The resulting representation ${ \bf h } _ { 0 }$ summarizes the query and initial retrieval results before Agent invocation.

## 5.3 Adaptive Path Routing

Motivation. A complex question does not necessarily require Agentic execution, while a seemingly simple question may need multiple retrieval operations. We therefore learn whether Agent improves upon an available non-Agent path, rather than predicting execution requirements solely from query types.

Relative Path Prediction. RouteDRAG first applies a verified Exact gate. Otherwise, a fixed policy selects the available shorter path $C ( q ) \in$ {Standard, VLM}. Instead of independently estimating every path’s correctness, we predict the relative advantage of the Agent:

$$
\widehat { \Delta } _ { \theta } ( q ) = \mathbf { w } _ { h } ^ { \top } \mathbf { h } _ { 0 } + \mathbf { w } _ { u } ^ { \top } \widetilde { \mathbf { u } } _ { 0 } + b ,\tag{4}
$$

where $\widetilde { \mathbf { u } } _ { 0 }$ contains normalized query and retrieval statistics. The first term models interactions between the query and retrieved document content, while the second preserves direct access to lightweight descriptive features.

Execution-Path Selection. The resulting routing policy is

$$
\pi _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) = \left\{ \begin{array} { l l } { \mathrm { E x a c t , ~ } } & { g _ { \mathrm { e x a c t } } ( \boldsymbol { q } , \mathcal { K } _ { D } ) = 1 , } \\ { \mathrm { A g e n t , ~ } } & { \widehat { \Delta } _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) > \tau , } \\ { C ( \boldsymbol { q } ) , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{5}
$$

where $g _ { \mathrm { e x a c t } }$ verifies complete structured metadata and � is determined on the development set. This formulation turns Agent invocation into a relative execution decision: the Agent is selected when its predicted improvement over the designated shorter path exceeds the routing threshold.

## 5.4 Dynamic Agent Execution

Motivation. Agent operations should adapt to the information accumulated during execution. We integrate learned action refinement with the LLM Planner, using the evolving execution state to guide its proposals.

State-Conditioned Operation Modeling. Agent starts with native hybrid retrieval and maintains a workspace $W _ { t }$ containing retrieved content, source references, and operation history. The shared projection network produces its state representation h<sub>�</sub>. For each candidate operation � with embedding $\mathbf { e } _ { a } ,$ , we compute action-conditioned summaries of text chunks and graph entities:

$$
\begin{array} { r } { \mathbf { z } _ { t , a } ^ { m } = \mathrm { A t t n } _ { \phi } \left( \mathbf { z } _ { q } + \mathbf { e } _ { a } , P _ { \psi } ( \mathbf { E } _ { t } ^ { m } ) \right) , \quad m \in \{ c , g \} . } \end{array}\tag{6}
$$

Here, $\mathbf { E } _ { t } ^ { c }$ and $\mathbf { E } _ { t } ^ { g }$ denote current chunk and entity embeddings. We combine these summaries with the execution state and operation history:

$$
\begin{array} { r } { z _ { \phi } ( W _ { t } , a ) = H _ { \phi } \left( [ \mathbf { h } _ { t } ; \mathbf { z } _ { t , a } ^ { c } ; \mathbf { z } _ { t , a } ^ { g } ; \mathbf { e } _ { a } ; \pmb { \ell } _ { t } ] \right) , } \end{array}\tag{7}
$$

where $\ell _ { t }$ encodes previous operations. The relative operation score is defined against immediate termination:

$$
\widehat { d } _ { t } ( a ) = z _ { \phi } ( W _ { t } , a ) - z _ { \phi } ( W _ { t } , \mathrm { F I N I S H } ) .\tag{8}
$$

This provides a common reference for comparing continued execution with termination.

Constrained Action Refinement. At each step, native execution rules determine the legal action set $\Gamma _ { t } ,$ including retrieval, graph expansion, structured lookup, visual inspection, and FINISH

![](images/65d631f64df0a11c91612843b196c79a88dfa16855a6d74fbcb28e1a70f6731c.jpg)  
Figure 3: Overview of RouteDRAG. A shared document index supports learned path routing and Agent execution control. Actor-aligned policy learning uses paired execution outcomes to train routing and action decisions in stages.

when permitted. Given the Planner’s proposal $a _ { t } ^ { P }$ , the controller identifies

$$
a _ { t } ^ { * } = \arg \operatorname* { m a x } _ { a \in \Gamma _ { t } } \widehat { d } _ { t } ( a ) .\tag{9}
$$

The selected action is

$$
a _ { t } = \left\{ \begin{array} { l l } { a _ { t } ^ { * } , \quad \widehat { d } _ { t } ( a _ { t } ^ { * } ) - \widehat { d } _ { t } ( a _ { t } ^ { P } ) > m , } \\ { a _ { t } ^ { P } , \quad \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{10}
$$

where � is a fixed margin. This refinement applies to valid Planner proposals, while mandatory actions follow native rules. If the controller changes the operation, the Planner generates its arguments, preserving the original tool execution procedure.

State Transition. The selected operation returns $o _ { t }$ , updating the workspace as $W _ { t + 1 } = \mathrm { C o m m i t } ( W _ { t } , o _ { t } )$ . The update incorporates new content, preserves source references, and removes duplicates. Subsequent decisions use the updated workspace until FINISH or the execution limit is reached, followed by answer synthesis and validation.

## 5.5 Sequential Policy Learning

Motivation. Agent invocation depends on its actual performance, while operation utility varies with the execution state. We therefore

learn both decisions from paired outcomes, training the router after updating the Agent controller.

Operation-Level Learning. We compare immediate termination with one additional operation at observed Agent states. For state $W _ { i , t }$ and operation �, the target is

$$
d _ { i , t , a } = Y \left( \operatorname { S t o p } ( { \mathcal { T } } _ { a } ( W _ { i , t } ) ) \right) - Y \left( \operatorname { S t o p } ( W _ { i , t } ) \right) ,\tag{11}
$$

where $Y ( \cdot ) \in \{ 0 , 1 \}$ denotes correctness and Stop applies a fixed � − �answer-generation procedure. Only evaluated state–action pairs are used. The controller combines regression, action ranking, and Planner imitation:

$$
\mathcal { L } _ { A } = \mathcal { L } _ { \mathrm { r e g } } + \alpha \mathcal { L } _ { \mathrm { r a n k } } ^ { A } + \beta \mathcal { L } _ { \mathrm { B C } } ,\tag{12}
$$

where $\mathcal { L } _ { \mathrm { r e g } }$ fits $\widehat { d } _ { t } ( a )$ to the observed gain, and $\mathcal { L } _ { \mathrm { B C } }$ preserves Planner preferences. The ranking loss is

$$
\mathcal { L } _ { \mathrm { r a n k } } ^ { A } = \frac { 1 } { | \mathcal { P } _ { A } | } \sum _ { ( t , a , b ) \in \mathcal { P } _ { A } } \mathrm { s o f t p l u s } \left( - \left[ \widehat { d } _ { t } ( a ) - \widehat { d } _ { t } ( b ) \right] \right) ,\tag{13}
$$

where ${ \mathcal { P } } _ { A }$ contains evaluated same-state action pairs with $d _ { t } ( a ) >$ $d _ { t } ( b )$ .

Path-Level Learning. We execute the updated Agent $A _ { \phi }$ and compare its correctness with the fixed shorter path:

$$
\Delta _ { i } = Y ( A _ { \phi } ( q _ { i } ) ) - Y ( C ( q _ { i } ) ) \in \{ - 1 , 0 , 1 \} .\tag{14}
$$

The router predicts this relative gain using regression and crossquery ranking:

$$
\begin{array} { r } { \mathcal { L } _ { R } = \mathbb { E } _ { i } \big ( \hat { \Delta } _ { i } - \Delta _ { i } \big ) ^ { 2 } + \mu \mathbb { E } _ { ( i , j ) \in \mathcal { P } _ { R } } \ell \big ( \hat { \Delta } _ { j } - \hat { \Delta } _ { i } \big ) . } \end{array}\tag{15}
$$

where $\widehat { \Delta } _ { i } = \widehat { \Delta } _ { \theta } ( q _ { i } ) , \ell ( x ) = \mathrm { s o f t p l u s } ( x )$ , and $\mathcal { P } _ { R }$ contains query pairs satisfying $\Delta _ { i } > \Delta _ { j }$ .

Sequential Training. RouteDRAG first trains the shared repre sentation and Agent controller from observed operation outcomes. It then freezes both and trains the routing head using the updated Agent’s execution results. This ensures that path routing reflects the Agent’s actual behavior. All parameters remain fixed during evaluation.

## 6 Experiments

We conduct extensive experiments on two multimodal document benchmarks to answer the following research questions. Q1: Overall Performance. How does RouteDRAG compare with existing document RAG methods in answering accuracy? Q2: Ablation Study. How do adaptive path routing and dynamic Agent execution contribute to performance? Q3: Robustness Analysis. How robust is RouteDRAG to noisy inputs, hyperparameter variations, and diferent model backbones? Q4: Eficiency Analysis. Can RouteDRAG achieve a favorable accuracy–computation trade-of compared with existing RAG methods?

## 6.1 Experimental Setup

Datasets, Baselines, and Backbones. We evaluate RouteDRAG on two multimodal document benchmarks, DocBench [31] and MMLongBench-Doc [19]. To assess its efectiveness across diferent retrieval and execution paradigms, we compare against four categories of baselines: (1) direct inference using a budgeted Qwen3- VL model [2]; (2) text-based RAG, including BM25-based sparse retrieval [21] and BGE-M3-based dense retrieval [20]; (3) multimodal RAG, including M3DocRAG [6], and EVisRAG [23]; and (4) graph-based and Agentic RAG, including RAG-Anything [12], MoLoRAG [27], ViDoRAG [25], and TAP-RAG [15]. For a fair comparison, all baselines and RouteDRAG use the same Qwen3-VL-8B-Instruct answering model, with diferences arising from their retrieval and execution strategies rather than backbone capabilities. We additionally compare Short Path Only, Rule Routing, and Uniform Agent as controlled execution strategies. RouteDRAG employs a unified document index, frozen BGE-M3 [20] for semantic representation, and Qwen3-VL-8B-Instruct [2] for multimodal inference.

Evaluation Metrics and Protocol. We report answering accuracy (Acc.), Agent usage (Agent%), executed actions per query (Actions/Q), answering calls and tokens per query, inference latency, and peak GPU memory. Accuracy is assessed using a fixed Qwen-based judge, with failed responses counted as incorrect. For the main performance comparison, we report mean accuracy and standard deviation over five independent runs.

## 6.2 Overall Performance (Q1)

To answer Q1, we compare RouteDRAG with representative document RAG methods across direct inference, text-based RAG, multimodal RAG, and graph-based and Agentic RAG. All experiments are conducted with five independent runs, and we report mean accuracy with standard deviations. As shown in Tables 2 and 1, RouteDRAG achieves 75.86% accuracy on DocBench and 53.80% on MMLongBench-Doc, outperforming the competing RAG baselines on both benchmarks. Notably, RouteDRAG outperforms uniform Agent execution by 1.18 and 1.28 percentage points on DocBench and MMLongBench-Doc, respectively, while reducing answering calls by 16.4% and 14.9%. These results suggest that selective Agent invocation can improve the accuracy–eficiency trade-of under our evaluation settings.The gains also extend across evaluation categories: RouteDRAG achieves the best results in two of five document domains and two of four question types on DocBench, and leads across all five support-source categories and two of three support-requirement categories on MMLongBench-Doc. These results demonstrate the efectiveness of RouteDRAG across diverse document structures and question requirements, motivating further analysis of its component contributions and computational eficiency.

![](images/7c90b73cc1e191348583d988d9cb8327479d619c96072cbb62dcb7ffc0c1b1d8.jpg)

![](images/eb8d268524ec4c0ec6add56aa3195f4dd90a4baa74aafe03470be52c8d5ecdef.jpg)  
Figure 4: Sensitivity to regularization strength and routing threshold on (a) DocBench and (b) MMLongBench-Doc. Cells report accuracy (%); colors show percentage-point changes from � = 100 and zero threshold ofset.

## 6.3 Ablation Study (Q2)

To answer Q2, we investigate the contributions of adaptive path routing and dynamic Agent execution. As shown in Table 3, removing Agent execution reduces accuracy to 73.41%/49.31%, while uniform Agent execution achieves 74.68%/52.52%. In contrast, RouteDRAG achieves 75.86%/53.80% while invoking Agent for only 78.49%/80.02% of queries. RouteDRAG-Eficient ofers a lower-cost operating point, achieving 74.05%/50.87% accuracy with only 11.52%/16.41% Agent use, exceeding rule-based routing on both datasets. These results show how selective execution balances the benefits of Agentic reasoning against unnecessary or potentially harmful operations.

Comparison Settings. w/o Adaptive Path Routing replaces learned path selection with rule-based capability assignment. w/o Dynamic Agent Execution assigns every query to a non-adaptive Agent execution configuration. w/o Agent Routing answers through the short-path portfolio, using Exact for supported structured requests and Standard or Direct VLM for the remaining queries. All Agent Routing sends every query to the mature Agent. RouteDRAG combines selective path allocation with dynamic Agent execution. RouteDRAG-Eficient shares the same mechanism as RouteDRAG and yields the eficiency-oriented operating point in Table A1 solely through a diferent threshold configuration. The two all-Agent configurations compare execution policies at full Agent coverage;

Table 1: Answer accuracy (%) on MMLongBench-Doc. Best and second-best results are bold and underlined, respectively.
<table><tr><td colspan="2"></td><td colspan="5">Support source</td><td colspan="3">Support requirement</td><td rowspan="2">Overall</td></tr><tr><td>Type</td><td>Method</td><td>TXT</td><td>LAY</td><td>CHA</td><td>TAB</td><td>FIG</td><td>SIN</td><td>MUL</td><td>UNA</td></tr><tr><td>Direct</td><td>Direct MLLM</td><td> $3 0 . 7 6 _ { 1 . 8 8 }$ </td><td> $3 6 . 3 0 _ { 2 . 0 9 }$ </td><td> $2 8 . 5 8 _ { 1 . 7 2 }$ </td><td> $2 0 . 5 2 _ { 2 . 1 6 }$ </td><td> $2 8 . 4 6 _ { \ 1 . 9 4 }$ </td><td> $3 6 . 3 0 _ { \ 1 . 6 6 }$ </td><td> $2 1 . 3 6 _ { 2 . 3 1 }$ </td><td> $5 2 . 4 6 _ { 1 . 8 3 }$ </td><td>35.00</td></tr><tr><td rowspan="2">Text RAG</td><td>BM25</td><td> $3 4 . 8 6 _ { 1 . 1 9 }$ </td><td> $1 9 . 2 0 _ { 1 . 3 7 }$ </td><td> $1 7 . 3 3 _ { 1 . 1 2 }$ </td><td> $3 1 . 8 8 _ { 1 . 4 8 }$ </td><td> $1 4 . 2 2 _ { 1 . 0 6 }$ </td><td> $3 4 . 0 0 _ { 0 . 9 4 }$ </td><td> $1 6 . 9 8 _ { 1 . 5 6 }$ </td><td> $7 6 . 6 4 _ { 1 . 2 1 }$ </td><td>37.98</td></tr><tr><td>BGE-M3</td><td> $3 3 . 7 2 _ { 1 . 3 2 }$ </td><td> $2 3 . 1 7 _ { 1 . 1 8 }$ </td><td> $1 3 . 9 6 _ { \ 1 . 0 9 }$ </td><td> $3 5 . 8 2 _ { 1 . 4 1 }$ </td><td> $1 6 . 1 5 _ { \ 1 . 0 2 }$ </td><td> $3 3 . 1 5 _ { 0 . 9 6 }$ </td><td> $1 9 . 9 9 _ { 1 . 5 3 }$ </td><td> $\underline { { 7 3 . 3 6 } } _ { 1 . 2 4 }$ </td><td>37.86</td></tr><tr><td rowspan="2">Multimodal</td><td>M3DocRAG [6]</td><td> $\underline { { 4 3 . 4 7 } } _ { 1 . 7 9 }$ </td><td> $3 1 . 8 1 _ { 1 . 6 3 }$ </td><td> $\underline { { 3 9 . 2 3 } } _ { 1 . 5 1 }$ </td><td> $3 3 . 4 5 _ { 1 . 9 1 }$ </td><td> $3 7 . 3 3 _ { 1 . 4 3 }$ </td><td> $\underline { { 5 3 . 0 8 } } _ { 1 . 3 2 }$ </td><td> $2 5 . 3 8 _ { 1 . 7 2 }$ </td><td> $2 5 . 0 0 _ { \phantom { 1 . 9 8 } }$ </td><td>37.58</td></tr><tr><td>EVisRAG [23]</td><td> $3 8 . 2 9 _ { 1 . 2 2 }$ </td><td> $3 3 . 2 1 _ { 1 . 3 4 }$ </td><td> $3 8 . 1 7 _ { 1 . 1 1 }$ </td><td> $3 9 . 8 4 _ { 1 . 4 6 }$ </td><td> $\underline { { 3 8 . 9 0 } } _ { 0 . 9 8 }$ </td><td> $5 1 . 5 5 _ { 0 . 9 2 }$ </td><td> $2 5 . 0 3 _ { \ 1 . 2 8 }$ </td><td> $5 8 . 2 0 _ { 1 . 5 7 }$ </td><td>44.27</td></tr><tr><td rowspan="5">Graph/Agentic</td><td>RAG-Anything [12]</td><td> $4 0 . 0 0 _ { \ 2 . 0 8 }$ </td><td> $2 9 . 3 0 _ { 1 . 9 1 }$ </td><td> $2 9 . 2 1 _ { 1 . 7 6 }$ </td><td> $\underline { { 4 3 . 4 5 } } _ { 2 . 2 2 }$ </td><td> $3 3 . 7 3 _ { 1 . 5 9 }$ </td><td> $4 7 . 5 9 _ { 1 . 5 3 }$ </td><td> $\underline { { 2 6 . 9 3 } } _ { 1 . 8 2 }$ </td><td> $6 0 . 2 5 _ { 2 . 3 1 }$ </td><td>43.61</td></tr><tr><td>MoLoRAG [27]</td><td> $3 8 . 8 8 _ { 2 . 3 9 }$ </td><td> $\underline { { 3 6 . 8 2 } } _ { 2 . 1 7 }$ </td><td> $3 5 . 8 6 _ { \ 1 . 9 4 }$ </td><td> $3 7 . 5 5 _ { 2 . 4 8 }$ </td><td> $3 2 . 8 0 _ { 2 . 0 8 }$ </td><td> $5 1 . 2 8 _ { 1 . 8 5 }$ </td><td> $2 2 . 3 1 _ { 2 . 2 3 }$ </td><td> $5 2 . 4 6 _ { 2 . 6 1 }$ </td><td>41.96</td></tr><tr><td>ViDoRAG [25]</td><td> $3 8 . 2 1 _ { 1 . 9 3 }$ </td><td> $2 7 . 4 1 _ { 1 . 7 8 }$ </td><td> $2 7 . 4 5 _ { \ 1 . 6 4 }$ </td><td> $3 4 . 6 6 _ { 2 . 0 2 }$ </td><td> $3 3 . 3 7 _ { \ 1 . 4 8 }$ </td><td> $4 8 . 6 9 _ { \phantom { } 1 . 5 5 }$ </td><td> $1 8 . 7 4 _ { 1 . 8 6 }$ </td><td> $6 2 . 7 0 _ { 2 . 1 9 }$ </td><td>41.85</td></tr><tr><td>TAP-RAG [15]</td><td> $4 3 . 2 5 _ { \ 1 . 6 9 }$ </td><td> $2 8 . 3 6 _ { 1 . 5 5 }$ </td><td> $3 4 . 9 1 _ { 1 . 3 4 }$ </td><td> $3 6 . 7 9 _ { 1 . 8 2 }$ </td><td> $3 6 . 2 2 _ { 1 . 4 6 }$ </td><td> $5 2 . 2 0 _ { \ 1 . 2 7 }$ </td><td> $2 4 . 8 4 _ { 1 . 6 1 }$ </td><td> $6 0 . 4 5 _ { \ 1 . 9 8 }$ </td><td>45.01</td></tr><tr><td>RouteDRAG</td><td> ${ \bf 5 1 . 6 8 } _ { 0 . 9 7 }$ </td><td> $4 6 . 2 2 _ { \ 1 . 0 8 }$ </td><td> $\mathbf { 4 1 . 5 7 } _ { 0 . 8 2 }$ </td><td> $5 5 . 0 5 _ { \mathrm { ~ 1 . 1 9 } }$ </td><td> $4 2 . 6 6 _ { 0 . 7 2 }$ </td><td> $5 8 . 7 5 _ { \ 0 . 6 1 }$ </td><td> $3 7 . 9 9 _ { 0 . 9 1 }$ </td><td> $6 7 . 6 2 _ { 1 . 3 1 }$ </td><td>53.80</td></tr></table>

Table 2: Answer accuracy (%) on DocBench. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td colspan="2"></td><td colspan="5">Document domain</td><td colspan="4">Question type</td><td rowspan="2">Overall</td></tr><tr><td>Type</td><td>Method</td><td> $\mathbf { A c a d . }$ </td><td>Finance</td><td> $\mathbf { G o v . }$ </td><td>Law</td><td>News</td><td>TXT</td><td>MM</td><td>META</td><td> $\mathbf { U N A }$ </td></tr><tr><td>Direct</td><td>Direct MLLM</td><td> $3 0 . 3 6 _ { 2 . 1 4 }$ </td><td> $3 9 . 5 8 _ { 2 . 3 8 }$ </td><td> $6 8 . 9 2 _ { \ 1 . 9 2 }$ </td><td> $6 5 . 9 7 _ { \ 2 . 2 6 }$ </td><td> $3 3 . 1 4 _ { 2 . 4 7 }$ </td><td> $5 2 . 4 3 _ { \textrm { 1 . 8 5 } }$ </td><td> $2 2 . 4 0 _ { \ 2 . 5 5 }$ </td><td> $4 9 . 2 2 _ { 2 . 0 8 }$ </td><td> $6 3 . 7 1 _ { 2 . 3 1 }$ </td><td>44.55</td></tr><tr><td rowspan="2">Text RAG</td><td>BM25</td><td> $6 1 . 0 6 _ { \ 1 . 4 5 }$ </td><td> $5 7 . 6 4 _ { 1 . 6 2 }$ </td><td> $\underline { { 7 4 . 3 2 } } _ { 1 . 2 7 }$ </td><td> $\underline { { 7 3 . 8 2 } } _ { 1 . 7 3 }$ </td><td> $8 7 . 2 1 _ { 1 . 1 8 }$ </td><td> $\underline { { 8 9 . 8 1 } } _ { 1 . 0 6 }$ </td><td> $5 8 . 4 4 _ { 1 . 8 4 }$ </td><td> $3 9 . 5 3 _ { \ 1 . 9 1 }$ </td><td> $\underline { { 8 0 . 6 5 } } _ { 1 . 3 9 }$ </td><td>68.24</td></tr><tr><td>BGE-M3</td><td> $5 5 . 7 8 _ { 1 . 5 8 }$ </td><td> $5 4 . 5 1 _ { \ 1 . 4 2 }$ </td><td> $6 7 . 5 7 _ { 1 . 3 1 }$ </td><td> $7 1 . 2 0 _ { \ 1 . 6 4 }$ </td><td> $8 6 . 6 3 _ { \ 1 . 2 2 }$ </td><td> $8 2 . 7 7 _ { 1 . 1 5 }$ </td><td> $5 2 . 9 2 _ { \ 1 . 7 8 }$ </td><td> $4 2 . 2 5 _ { \ 1 . 9 6 }$ </td><td> $7 9 . 0 3 _ { \ 1 . 3 7 }$ </td><td>64.52</td></tr><tr><td rowspan="2">Multimodal</td><td>M3DocRAG [6]</td><td> $6 1 . 0 6 _ { \ 2 . 0 3 }$ </td><td> $6 5 . 2 8 _ { 1 . 8 7 }$ </td><td> $6 6 . 8 9 _ { \ 1 . 7 4 }$ </td><td> $7 1 . 2 0 _ { 2 . 1 8 }$ </td><td> $8 1 . 4 0 _ { \ 1 . 6 5 }$ </td><td> $8 5 . 6 8 _ { \ 1 . 5 2 }$ </td><td> $6 8 . 5 1 _ { 1 . 9 6 }$ </td><td> $5 3 . 1 0 _ { 2 . 2 6 }$ </td><td> $5 7 . 9 0 _ { \ : 2 . 0 7 }$ </td><td>67.88</td></tr><tr><td>EVisRAG [23]</td><td> $6 7 . 3 0 _ { \ 1 . 4 3 }$ </td><td> $\underline { { 6 5 . 7 5 } } _ { 1 . 5 6 }$ </td><td> $7 0 . 6 5 _ { \ 1 . 2 8 }$ </td><td> $7 2 . 9 2 _ { \ 1 . 7 2 }$ </td><td> $8 2 . 4 7 _ { 1 . 1 6 }$ </td><td> $8 2 . 4 4 _ { 1 . 0 8 }$ </td><td> $\underline { { 7 2 . 6 5 } } _ { 1 . 4 7 }$ </td><td> $4 5 . 8 4 _ { 1 . 8 5 }$ </td><td> $7 8 . 4 5 _ { 1 . 3 1 }$ </td><td>70.69</td></tr><tr><td rowspan="5">Graph/Agentic</td><td>RAG-Anything [12]</td><td> $\underline { { 6 9 . 6 4 } } _ { 2 . 3 2 }$ </td><td> $6 2 . 5 0 _ { 2 . 1 4 }$ </td><td> $6 7 . 5 7 _ { \ 1 . 9 7 }$ </td><td> $7 0 . 6 8 _ { 2 . 4 8 }$ </td><td> $8 8 . 3 7 _ { 1 . 8 1 }$ </td><td> $8 6 . 8 9 _ { 1 . 7 3 }$ </td><td> $7 2 . 4 0 _ { \ 1 . 9 2 }$ </td><td> $3 9 . 5 3 _ { \ : 2 . 5 7 }$ </td><td> $7 6 . 6 1 _ { 2 . 2 1 }$ </td><td>70.60</td></tr><tr><td>MoLoRAG [27]</td><td> $4 6 . 8 6 _ { 2 . 6 6 }$ </td><td> $6 5 . 2 8 _ { 2 . 4 3 }$ </td><td> $\underline { { 7 4 . 3 2 } } _ { 2 . 1 9 }$ </td><td> $7 3 . 3 0 _ { 2 . 7 5 }$ </td><td>74.882.34</td><td> $6 9 . 1 7 _ { \ 2 . 0 8 }$ </td><td> $5 4 . 8 7 _ { \ 2 . 4 9 }$ </td><td> $4 4 . 9 6 _ { \ 2 . 8 9 }$ </td><td> $5 6 . 4 5 _ { 2 . 6 1 }$ </td><td>64.32</td></tr><tr><td>ViDoRAG [25]</td><td> $6 4 . 0 3 _ { \ 2 . 1 7 }$ </td><td> $6 5 . 6 2 _ { 2 . 0 2 }$ </td><td> $6 6 . 2 2 _ { 1 . 8 8 }$ </td><td> $7 3 . 3 0 _ { 2 . 2 8 }$ </td><td> $\underline { { 8 7 . 7 9 } } _ { 1 . 6 9 }$ </td><td> $7 8 . 6 4 _ { 1 . 7 6 }$ </td><td> $6 8 . 1 8 _ { \ 2 . 0 9 }$ </td><td> $\underline { { 5 1 . 1 6 } } _ { 2 . 4 6 }$ </td><td> $8 5 . 4 8 _ { 1 . 6 1 }$ </td><td>70.05</td></tr><tr><td>TAP-RAG [15]</td><td> $6 7 . 0 7 _ { \ 1 . 9 2 }$ </td><td> $6 4 . 2 2 _ { 1 . 7 8 }$ </td><td> $7 4 . 8 7 _ { 1 . 5 4 }$ </td><td>74.662.06</td><td> $7 6 . 0 5 _ { \ 1 . 6 8 }$ </td><td> $8 7 . 4 4 _ { 1 . 4 6 }$ </td><td> $6 4 . 8 8 _ { \ 1 . 8 4 }$ </td><td> $4 6 . 3 9 _ { \ 2 . 2 4 }$ </td><td> $7 1 . 7 5 _ { \ 1 . 9 7 }$ </td><td>70.09</td></tr><tr><td>RouteDRAG</td><td> $7 4 . 2 6 _ { \ : 1 . 1 2 }$ </td><td>73.611.26</td><td> $7 3 . 6 5 _ { \ 0 . 9 8 }$ </td><td> $7 3 . 3 0 _ { 1 . 4 1 }$ </td><td> $8 7 . 2 1 _ { 0 . 8 6 }$ </td><td> $\mathbf { 9 1 . 7 5 } _ { 0 . 7 3 }$ </td><td> $7 7 . 2 7 _ { 1 . 0 5 }$ </td><td> $4 8 . 4 5 _ { \phantom { } 1 . 5 3 }$ </td><td> $7 6 . 6 1 _ { 1 . 1 8 }$ </td><td>75.86</td></tr></table>

Table 3: Component ablation. Accuracy and Agent use are percentages; tokens are in thousands. Dashes mark the no-Agent setting. Best and second-best measured values are bold and underlined.
<table><tr><td></td><td colspan="5">DocBench</td><td colspan="5">MMLongBench-Doc</td></tr><tr><td>Method</td><td> $\operatorname { A c c . } \uparrow$ </td><td> $\mathrm { A g e n t \% } \downarrow$ </td><td> $\mathrm { A c t . } / \mathrm { Q } \downarrow$ </td><td> $\mathrm { C a l l s / Q \downarrow }$ </td><td> $\mathrm { T o k . } / \mathrm { Q } \downarrow$ </td><td> $\operatorname { A c c . } \uparrow$ </td><td> $\mathrm { A g e n t \% } \downarrow$ </td><td> $\mathrm { A c t . } / \mathrm { Q } \downarrow$ </td><td> $\mathrm { C a l l s / Q \downarrow }$ </td><td> $\mathrm { T o k . } / \mathrm { Q } \downarrow$ </td></tr><tr><td>w/o Adaptive Path Routing</td><td>73.77</td><td>15.88</td><td>0.300</td><td>2.97</td><td>29.14</td><td>50.32</td><td>16.50</td><td>0.379</td><td>3.31</td><td>33.15</td></tr><tr><td>w/o Dynamic Agent Execution</td><td>74.19</td><td>100.00</td><td>1.418</td><td>7.11</td><td>49.30</td><td>50.25</td><td>100.00</td><td>1.850</td><td>8.23</td><td>56.53</td></tr><tr><td>w/o Agent Routing</td><td>73.41</td><td></td><td></td><td>2.01</td><td>24.07</td><td>49.31</td><td></td><td>一</td><td>2.23</td><td>27.14</td></tr><tr><td>All Agent Routing</td><td>74.68</td><td>100.00</td><td>1.685</td><td>7.46</td><td>52.08</td><td>52.52</td><td>100.00</td><td>2.148</td><td>8.25</td><td>57.02</td></tr><tr><td>RouteDRAG-Efficient</td><td>74.05</td><td>11.52</td><td>0.171</td><td>2.58</td><td>27.36</td><td>50.87</td><td>16.41</td><td>0.348</td><td>3.20</td><td>32.18</td></tr><tr><td>RouteDRAG</td><td>75.86</td><td>78.49</td><td>1.349</td><td>6.24</td><td>46.04</td><td>53.80</td><td>80.02</td><td>1.769</td><td>7.02</td><td>50.99</td></tr></table>

## 6.4 Robustness Analysis (Q3)

the routed configurations additionally expose the accuracy–cost trade-of of selective invocation.

To answer Q3, we examine the sensitivity of routing decisions to regularization and threshold choices, together with robustness across model backbones and noisy inputs.

Hyperparameters. Fig. 4 examines the interaction between regularization strength � and the routing threshold. For each �, the predictor is fitted on the training documents, and a reference threshold is calibrated on development documents. The horizontal axis varies a fixed ofset from this reference; each query is then routed independently. Accuracy varies modestly across the displayed settings on DocBench. MMLongBench-Doc is more sensitive when strong regularization is combined with a positive threshold ofset. Stronger regularization compresses the prediction scale, so the same ofset can exclude more Agent executions. Regularization and threshold calibration should therefore be considered together.

![](images/95220bcc2bd36ad1f4c9863fd3c406cb1fcd2bc7af033fe62d234eca5dc3019c.jpg)  
Figure 5: Robustness comparison under increasing noise ratios. (a–b) Retrieval noise. (c–d) Document distractors.

![](images/cdfc2ba6705799a4a72fdfb4f624daff98fe63bda6c284c61a1ce53375882c41.jpg)

![](images/65bec79c06605d26014f65d32461e7d2308cbfed816ad0a9cbec528f4be3c24d.jpg)

![](images/60cf1b4b1e949cad9d23c918ace3da0f3fb33e2ceba5389aa5f4edc94e756ece.jpg)

![](images/5a990f94c924494aac351334c23838ba7b324ba0a881bb39cc96100b072fb9c5.jpg)  
Figure 6: Practical eficiency comparison with representative RAG baselines. (a–b) Answer accuracy versus inference latency on DocBench and MMLongBench-Doc. (c–d) Peak GPU memory consumption across diferent methods.

Table 4: Backbone generalization on DocBench and MMLongBench-Doc. Accuracy (%) and relative reduction in answering model calls compared with Uniform Agent (%).
<table><tr><td>VLM Backbone</td><td>Agent</td><td>Random</td><td>Ours</td><td>Calls Red. (%)</td></tr><tr><td>Qwen3-VL-4B</td><td>69.60/45.92</td><td>68.20/43.21</td><td>70.15/46.65</td><td>13.8/14.2</td></tr><tr><td>Qwen3-VL-8B</td><td>74.68/52.52</td><td>73.66/49.88</td><td>75.86/53.80</td><td>16.4/14.9</td></tr><tr><td>Qwen2.5-VL-7B</td><td>72.78/48.49</td><td>70.40/45.92</td><td>73.32/48.30</td><td>14.1/14.4</td></tr><tr><td>InternVL3-8B</td><td>73.50/50.23</td><td>71.10/47.59</td><td>74.23/51.05</td><td>15.0/14.3</td></tr><tr><td>Gemma-3-12B</td><td>71.96/47.57</td><td>69.54/45.97</td><td>71.87/48.49</td><td>13.7/15.2</td></tr></table>

Backbone Generalization. Table 4 evaluates RouteDRAG across multimodal backbones of diferent scales and architectures. With Qwen3-VL-4B, RouteDRAG achieves 70.15%/46.65% accuracy on DocBench/MMLongBench-Doc, improving over uniform Agent execution by 0.55/0.73 percentage points while reducing answering calls by 13.8%/14.2%. With InternVL3-8B, it improves accuracy by 0.73/0.82 percentage points. Gains vary across backbones.

Robustness to Noisy Information. We consider two perturbations: Retrieval Noise, which adds irrelevant content to retrieved results, and Document Distractors, which adds unrelated pages to documents. Fig. 5 reports accuracy as the perturbation ratio increases from 0 to 0.5. At the highest ratio, RouteDRAG achieves 73.9%/52.3% accuracy on DocBench/MMLongBench-Doc under Retrieval Noise and 74.5%/51.3% under Document Distractors. It outperforms TAP-RAG by 5.9/13.5 and 8.5/12.4 percentage points under the two perturbations, respectively.

## 6.5 Eficiency Analysis (Q4)

To answer Q4, we evaluate answering calls and tokens, inference latency, and peak GPU memory consumption.

Eficiency of Selective Execution. Table 3 shows that RouteDRAG uses 6.24/7.02 answering calls and 46.04k/50.99k tokens per query on DocBench/MMLongBench-Doc. Compared with All Agent Routing, it improves accuracy by 1.18/1.28 percentage points while reducing calls by 16.4%/14.9% and tokens by 11.6%/10.6%. Compared with w/o Dynamic Agent Execution, it improves accuracy by 1.67/3.55 percentage points while reducing both calls and tokens. The lower costs of w/o Adaptive Path Routing and w/o Agent Routing come with reduced accuracy.

Practical Eficiency Comparison. Fig. 6(a–b) compares answer accuracy against inference latency on DocBench and MMLongBench-Doc; panels (c–d) compare peak GPU memory consumption. These measurements complement the answering-call and token costs in Table 3.

## 7 Conclusion

In this paper, we investigated a fundamental question in Agentic document RAG: when is Agentic execution necessary, and how should its operations be controlled? Our empirical findings reveal that Agentic execution provides query-dependent benefits, while additional operations yield gains, ties, and degradations in answer correctness. Motivated by these observations, we proposed RouteDRAG, a framework that combines adaptive path routing with dynamic Agent execution through sequential policy learning. By learning from observed execution outcomes, RouteDRAG selectively invokes Agent and refines its subsequent operations. Experiments on two multimodal document benchmarks demonstrate improved accuracy with fewer answering calls compared with uniform Agent execution. These results highlight the importance of controlling not only whether an Agent is invoked, but also how it executes. While this work focuses on single-turn document question answering, extending selective execution to multi-turn document interactions presents a promising direction for future research.

## References

[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations. OpenReview.net, Vienna, Austria, 30 pages. https://openreview.net/forum?id=hSyW5go0v8

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 https://arxiv.org/abs/2511.21631

[3] Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu Florian, Martin Franz, Dinesh Garg, Dinesh Khandelwal, Scott McCarley, Michael McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus, John Pitrelli, Saurabh Pujar, Salim Roukos, Andrzej Sakrajda, Avi Sil, Rosario Uceda-Sosa, Todd Ward, and Rong Zhang. 2020. The TechQA Dataset. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics. Association for Computationa Linguistics, 1269–1278. doi:10.18653/v1/2020.acl-main.117

[4] Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, and Jiaxin Mao. 2025. MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation. arXiv:2508.01005 https://arxiv.org/ abs/2508.01005

[5] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 3697– 3711. doi:10.18653/v1/2021.emnlp-main.300

[6] Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi document Understanding. arXiv:2411.04952 https://arxiv.org/abs/2411.04952

[7] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. In Proceedings of the 2021 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 4599–4610. doi:10.18653/v1/2021.naacl-main.365

[8] Yaxin Du, Junru Song, Yifan Zhou, Cheng Wang, Jiahao Gu, Zimeng Chen, Menglan Chen, Wen Yao, Yang Yang, Ying Wen, et al. 2026. �<sup>2</sup>-Reader: Dual Evolving Graphs for Multimodal Document QA. arXiv preprint arXiv:2601.22055 (2026).

[9] Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald. 2026. SPARKLE: A Structured and Plug-and-play Agentic Retrieval Policy for Adaptive RAG Models. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, San Diego, California, United States, 38699–38719. doi:10.18653/v1/2026.acl-long.1793

[10] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. Colpali: Eficient document retrieval with vision language models. In International Conference on Learning Representations, Vol. 2025. 61424–61449.

[11] Song Feng, Hui Wan, Chulaka Gunasekara, Siva Patel, Sachindra Joshi, and Luis Lastras. 2020. doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 8118–8128. doi:10.18653/ v1/2020.emnlp-main.652

[12] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. RAG Anything: All-in-One RAG Framework. arXiv:2510.12323 https://arxiv.org/abs/ 2510.12323

[13] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and Fast Retrieval-Augmented Generation. In Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Lin guistics, Suzhou, China, 10746–10761. doi:10.18653/v1/2025.findings-emnlp.568

[14] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, Mexico City, Mexico, 7036–7050. doi:10.18653/v1/2024.naacl-long.389

[15] Zhong Ji, Keqi Jin, Yan Zhang, and Jiasheng Li. 2026. TAP-RAG: Task-Aware Policy Control for Long-Document Multimodal Question Answering. arXiv:2607.18917 https://arxiv.org/abs/2607.18917

[16] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing. 7969–7992.

[17] Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, et al. 2024. Graphreader: Building graph-based agent to enhance long-context abilities of large language models. In Findings ofthe Association for Computational Linguistics: EMNLP 2024. 12758– 12786.

[18] Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, Weizhuo Chen, Jianjun Li, and Zhiyuan Ma. 2026. OPERA: A Reinforcement Learning– Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval. Proceedings of the AAAI Conference on Artificial Intelligence 40, 38 (2026), 32258–32266. doi:10.1609/aaai.v40i38.40499

[19] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems 37 (2024), 95963–96010.

[20] Multi-Linguality Multi-Functionality Multi-Granularity. 2024. M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216 (2024).

[21] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3, 4 (2009), 333–389. doi:10.1561/1500000019

[22] Zhivar Sourati, Zheng Wang, Marianne Menglin Liu, Yazhe Hu, Mengqing Guo, Sujeeth Bharadwaj, Kyu J Han, Tao Sheng, Sujith Ravi, Morteza Dehghani, et al. 2026. Lad-rag: Layout-aware dynamic rag for visually-rich document understanding. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15945–15968.

[23] Yubo Sun, Chunyi Peng, Yukun Yan, Shi Yu, Zhenghao Liu, Chi Chen, Zhiyuan Liu, and Maosong Sun. 2025. Visrag 2.0: Evidence-guided multi-image reasoning in visual retrieval-augmented generation. arXiv preprint arXiv:2510.09733 (2025).

[24] Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and Sihong Xie. 2025. MBA-RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through Question Complexity. In Proceedings ofthe 31st International Conference on Computational Linguistics. Association for Computational Linguistics, Abu Dhabi, UAE, 3248–3254. https://aclanthology.org/2025.coling-main.218/

[25] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. 2025. ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 9113–9134. doi:10.18653/v1/2025. emnlp-main.464

[26] Zhengren Wang, Dongsheng Ma, Huaping Zhong, Jiayu Li, Wentao Zhang, Bin Wang, and Conghui He. 2026. Agenticocr: Parsing only what you need for eficient retrieval-augmented generation. arXiv preprint arXiv:2602.24134 (2026).

[27] Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. 2025. Molorag: Bootstrapping document understanding via multi-modal logic-aware retrieval. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. 14035–14056.

[28] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 (2022).

[29] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2025. Visrag: Vision-based retrievalaugmented generation on multi-modality documents. In International Conference on Learning Representations, Vol. 2025. 21074–21098.

[30] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, 3277–3287. doi:10.18653/v1/2021.acl-long.254

[31] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. 2025. DocBench: A Benchmark for Evaluating LLM-based Document Reading Systems. In Proceedings ofthe 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing. Association for Computational Linguistics, Albuquerque, New Mexico, USA, 359–373. doi:10.18653/v1/2025.knowledgenlp-1.2

[32] Yilong Zuo, Xunkai Li, Jing Yuan, Qiangqiang Dai, Hongchao Qin, and Ronghua Li. 2026. MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA. arXiv:2606.15906 https://arxiv.org/ abs/2606.15906

## A Experimental Details

## A.1 Datasets, Readers, and Execution Paths

We evaluate RouteDRAG on all 1,102 DocBench questions from 229 documents and 1,091 MMLongBench-Doc questions from 135 documents. All paths share the document index and frozen BGE-M3 representations, using Qwen3-VL-8B-Instruct with an FP8 backend; final Agent synthesis is capped at 8,192 output tokens. The desig nated short path uses Standard or eligible Direct VLM, while the Actor denotes the Agent with its native Planner, Workspace, legal action filtering, and learned intervention. For controlled allocation comparisons, we reuse the recorded short-path and Actor answers for each query, keeping Exact Specialist outputs fixed across poli cies.

## A.2 Baseline Configurations

RAG-Anything uses upstream mixed retrieval with entity $\mathrm { { t o p } } \mathrm { { - } } k =$ 50 and chunk top-� = 10. It reuses the canonical graph index and uses a Qwen FP8 service with a one-sentence answer instruction. MoLoRAG retains native multimodal retrieval and traversal. BM25 and BGE-M3 share native MinerU text chunks, including native table text, with document-local top-20 retrieval and a one-sentence Qwen FP8 reader. The BGE-M3 baseline retrieves native text chunks using frozen embeddings.

M3DocRAG follows the comparison adapter: document-local ColPali [10] retrieval selects three pages, rendered at 144 DPI for the Qwen FP8 reader.

Budgeted Direct MLLM reads all pages up to 128. For the remaining baselines, we preserve their original retrieval and execution procedures while using the same Qwen3-VL-8B-Instruct answering model and evaluation protocol across all methods.

## A.3 Robustness Evaluation Protocol

We evaluate RouteDRAG and representative RAG baselines on DocBench and MMLongBench-Doc under two types of perturbations, with noise ratios $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ . A ratio of zero denotes the original setting. Distractors are sampled from other documents in the same benchmark, excluding the source document of the current query. The questions, reference answers, model parameters, and routing thresholds remain fixed throughout evaluation.

Retrieval Noise. For each query, we obtain the native retrieval results of each method and denote the number of retrieved units by �. We then append $\lfloor \rho K \rfloor$ irrelevant units sampled from the distractor pool, preserving the original results. Thus, $\rho = 0 . 5$ adds distractors equal to half the original number of retrieved units, while preserving the original retrieved content. The document index and retrieval model remain unchanged, isolating interference introduced at retrieval time.

Document Distractors. For a document containing � pages, we append $\lfloor \rho P \rfloor$ unrelated pages sampled from other documents, inserting them at randomly selected positions while preserving the relative order of the original pages. $\mathrm { A t } \rho = 0 . 5$ , the number of added pages is half the original page count. Each method processes the perturbed documents using its native indexing pipeline, including index reconstruction where required. This setting evaluates the efect of irrelevant pages present before retrieval.

## A.4 Inference Eficiency Measurement

We measure inference latency and peak GPU memory for Route-DRAG and representative RAG baselines under the same hardware and runtime conditions. All methods use batch size 1, with five warm-up queries followed by 100 timed queries drawn from the same question set. We report the mean end-to-end latency per query and peak GPU memory usage. Experiments run on a single NVIDIA A100 GPU, with inference precision and backend following each method’s configuration.

End-to-end latency includes online retrieval, routing, Agent planning and tool execution, and final answer generation. Ofline document parsing and index construction are excluded. Peak GPU memory includes model weights and runtime allocations. All methods retain their designated retrieval and answering pipelines; Route-DRAG additionally includes routing and learned execution-control overhead.

## A.5 Routing Parameters

For each query, a fixed policy first designates Standard or Direct VLM as the short path �(�). The learned router then predicts the signed benefit of Agent execution relative to $C ( q )$ , using the query representation and short-path identity. Agent is invoked when the predicted benefit exceeds the fixed threshold $\tau = 0 . 0 2 ;$ otherwise, �(�) is executed.

The router is trained on paired Agent and short-path outcomes. Its weights are fitted on training data, while model configurations and routing thresholds are selected using development data. The resulting routing function is applied independently to each query with its fixed threshold.

## A.6 Document-Disjoint Cross-Validation

We evaluate the learned Controller and router using five-fold crossvalidation at the source-PDF level, stratified by dataset. Questions from the same PDF are assigned to the same outer fold, with dupli cate copies grouped under a common PDF identity. In each round, one fold serves as the test set. Among the remaining documents, 20% form an inner development set and the rest form the training set. The resulting training/development/test proportions are approximately 64%/16%/20% of documents. Question counts vary with the number of questions associated with each document.

In each round, the shared projection and operation Controller are trained using operation-level outcomes from the training partition, with checkpoint selection based on the inner development set. They are then frozen, and a separate router is fitted using the training questions and their paired correctness targets, $\Delta _ { i } =$ $Y ( A _ { \phi } ( q _ { i } ) ) - Y ( C ( q _ { i } ) )$ . The inner development set determines the router configuration and decision threshold. The trained components and selected threshold remain fixed when applied to the outer test fold.

We pool the predictions from the five outer test folds, so each query in the routing evaluation contributes one out-of-fold routing decision. Dataset-level routing results are aggregated from these held-out predictions.

## B Method Implementation Details

## B.1 Shared Index and Source References

Figure 3 distinguishes the vector index $I _ { D } ;$ , the entity–relation graph $\mathcal { G } _ { D } ,$ , and source references $\rho _ { D }$ . The vector index supports similarity retrieval of text chunks and entity descriptions, while graph relations enable connected retrieval. Visual descriptions are indexed with links to their original pages and image assets. Source references retain document-local identifiers and locations, allowing retrieved descriptions to be traced to their source tables or figures. Retrieving a visual description and inspecting its original image are separate operations.

Frozen semantic vectors have 1,024 dimensions. A shared learned projection maps query, chunk, and entity representations to 64 dimensions. Each input contains up to eight chunk vectors and four entity vectors, with unavailable entries padded and masked before attention. The graph branch pools retrieved entity representations, while relations remain available for graph retrieval. Lightweight statistics describe query form, available content, and query–content similarity; these features are computed from information available at each decision point.

## B.2 Exact and Short-Path Interfaces

Exact is a deterministic metadata gate that returns an answer only when a supported structured request satisfies the required completeness checks. Otherwise, the fixed short-path policy selects Standard by default or Direct VLM when visual inspection is eligible for the query. Standard performs graph-backed retrieval and answering. Direct VLM first applies a text precheck, invokes visual reading when needed, and falls back to text when visual reading is unnecessary.

The designated short path is denoted by $C ( q )$ . Adaptive Path Routing predicts whether the Agent ofers suficient benefit over the designated shorter path $C ( q )$ to justify additional execution. Exact outputs remain fixed across allocation policies, and the Agent, denoted as the Actor, retains its native Planner, Workspace, legalaction constraints, and tool interfaces.

## B.3 Workspace, Legal Actions, and Tool Arguments

The query-specific Workspace stores retrieved text, entities, relations, source references, tool outcomes, and action history. Each tool result is committed with its provenance, and duplicate content is removed. Initial retrieval creates the starting state; subsequent operations update the same Workspace while reusing the shared document index.

At step �, execution rules construct the legal action set $\Gamma _ { t }$ from the available resources and current Workspace. Candidate operations include text search, graph expansion, structured lookup, visual inspection, and FINISH when permitted. The learned Controller scores only legal actions, while the Planner proposes an action and its arguments. If the proposed action is retained, its arguments are reused; otherwise, the Planner generates arguments for the selected tool.

The Controller receives projected query and content representations, state statistics, operation embeddings, and an eight-dimensional history vector comprising the step, history length, five operation counts, and failure count, transformed by log(1 + �). Operation embeddings have 64 dimensions. The action-value head uses a 32-unit hidden layer and a scalar output, with FINISH as the reference and a fixed refinement margin of $m = 0 . 1 5$ . The relative scores estimate operation-level correctness gains against the FINISH reference. After a permitted stop or execution limit, the original synthesis procedure generates the final answer from the accumulated content and source references.

## B.4 Learning Signals

The labels $Y _ { \mathrm { s t o p } }$ and $Y _ { \mathrm { c o n t i n u e } }$ in Figure 3 denote answer correctness before and after one additional operation, respectively, as defined in Equation 11. Only evaluated state–action pairs contribute to the regression and ranking losses. Planner imitation provides an auxiliary signal that preserves the original Planner’s operation preferences.

At the path level, $r _ { A } = Y ( A _ { \phi } ( q ) )$ and $r _ { C } = Y ( C ( q ) )$ denote the paired correctness outcomes of the Actor and designated short path. Their diference, $r _ { A } - r _ { C }$ , provides the signed routing target in Equation 14. The shared projection and Controller are trained first and then frozen while the routing heads learn from the updated Actor’s outcomes. Reference answers and correctness labels provide ofline supervision; inference uses only the learned representations, predicted scores, and legal execution rules.

## B.5 Training and Policy Selection

Sequential Training. The document index, semantic encoder, and language-model backbones remain frozen. In the first stage, the shared projection and operation Controller are trained using observed operation-level outcomes through regression, same-state ranking, and auxiliary Planner imitation. In the second stage, the shared projection and Controller are frozen, and the routing heads are trained using the paired correctness outcomes of the Actor and the designated short path. This two-stage procedure is repeated independently in each fold, using the training, development, and test partitions defined by source PDF in Appendix A.6.

Checkpoint and Threshold Selection. Each stage selects its checkpoint using the corresponding development objective. The operation-refinement margin is fixed at $m = 0 . 1 5 .$ The per-query routing threshold � is selected on the development set, and both model parameters and thresholds remain fixed throughout test evaluation.

Execution Policies. In per-query inference, the router invokes the Actor when the predicted relative benefit $\widehat { \Delta } _ { \theta } ( q )$ exceeds �; otherwise, it uses the designated short path. Once invoked, the Controller refines the Planner’s proposal only when a legal alternative exceeds the proposed action’s score by the fixed refinement margin.

The high-accuracy configuration uses the shared router and fixed threshold specified in Appendix A.5. Each query is routed independently: the Agent is invoked when $\widehat { \Delta } ( q ) > \tau ;$ otherwise, the designated short path is used.

![](images/3c02cb4dc80e274172e91f27c21308a2358c71683f57819135ca8267c3afced5.jpg)

![](images/0889f566a0b7bcbef03172888ea20f3fb2b0ca80811d635c5d767130022f15a6.jpg)

![](images/db150ca49a97f2eb509bc946bda68e51f1f559a2e7ff078509aab17da605b194.jpg)  
Figure A2: Benefit-predictor validation loss. The marker identifies the selected epoch, 47.  
Figure A1: Tool use across document-length groups. Counts show questions per bin; the MML 201+ bin contains five questions from one document.

## C Supplementary Execution Profiles

## C.1 Natural Document-Length Profile

Figure A1 groups the evaluated documents by page count and reports tool actions per query. RouteDRAG uses fewer actions than uniform execution across the displayed groups. Tool use also varies within the length range, reflecting query requirements as well as document size. Question counts accompany each bin; the longest MML bin contains five questions from one PDF.

## C.2 Validation-Loss Trace

Figure A2 plots the benefit predictor’s validation objective: signedbenefit regression plus a ranking term weighted by 0.25. Training uses paired execution outcomes, and validation selects epoch 47. The trace describes optimization of the routing objective; the workload and document-disjoint allocation results are reported separately.

Table A1: Accuracy–cost trade-ofs of per-query routing configurations. Accuracy and actual Agent usage are percentages.
<table><tr><td></td><td colspan="3">DocBench</td><td colspan="3">MMLongBench-Doc</td></tr><tr><td>Configuration</td><td></td><td>Acc.↑ Agent%↓</td><td>Calls/Q↓</td><td></td><td></td><td>Acc.↑ Agent%↓ Calls/Q↓</td></tr><tr><td>RouteDRAG-Efficient</td><td>74.05</td><td>11.52</td><td>2.58</td><td>50.87</td><td>16.41</td><td>3.20</td></tr><tr><td>RouteDRAG-Medium</td><td>74.73</td><td>42.65</td><td>4.22</td><td>52.29</td><td>57.56</td><td>5.72</td></tr><tr><td>RouteDRAG-High</td><td>75.18</td><td>59.53</td><td>5.09</td><td>52.93</td><td>71.13</td><td>6.57</td></tr><tr><td>RouteDRAG</td><td>75.86</td><td>78.49</td><td>6.24</td><td>53.80</td><td>80.02</td><td>7.02</td></tr></table>

## C.3 Per-Query Routing Operating Points

Table A1 compares four routing configurations with fixed thresholds calibrated on the development set. RouteDRAG-Eficient, RouteDRAG-Medium, RouteDRAG-High, and RouteDRAG share the accuracy-oriented router with diferent thresholds. RouteDRAG Eficient achieves accuracies of 74.05% and 50.87% on DocBench and MMLongBench-Doc, respectively, while limiting Agent usage to 11.52% and 16.41%, with 2.58 and 3.20 calls per query. Increasing the routing intensity improves accuracy at the cost of higher resource usage. RouteDRAG-Medium achieves 74.73%/52.29% accuracy at 42.65%/57.56% Agent usage, while RouteDRAG-High reaches 75.18%/52.93% accuracy at 59.53%/71.13% Agent usage on the two benchmarks, respectively. RouteDRAG achieves the highest accuracy, at 75.86% on DocBench and 53.80% on MMLongBench-Doc, but also incurs the highest Agent usage (78.49%/80.02%) and the most calls per query (6.24/7.02). Overall, these operating points illustrate the trade-of between accuracy and inference cost enabled by routing decisions.