# SynAct: A Reasoning-Acting Large Language Model Agent for Adaptive Synthesis Optimization

Fangzhou Liu, Peiyi Han, Jiawei Liu, Yuan Pu, Zhuolun He, Rongliang Fu, Tsung-Yi Ho, Bei Yu

Abstract—Logic synthesis transforms RTL designs into gatelevel netlists, where PPA results are highly sensitive to the choice of optimization commands, making synthesis tuning both highdimensional and expensive. Previous approaches fall into two categories: automated methods, which perform black-box search over fixed action spaces with limited decision-level interpretability, and LLM-based methods, which typically generate static scripts upfront and cannot adapt to evolving circuit states. We present SynAct, an adaptive closed-loop LLM reasoning–acting agent that iteratively diagnoses live synthesis reports and reasons over the current circuit state, retrieved tool knowledge, and historical optimization experience to issue targeted commands. SynAct focuses on improving timing, particularly worst negative slack (WNS), while maintaining balanced area and power tradeoffs. Experiments on a commercial synthesis tool across 14 designs show that SynAct reduces average WNS to 27% of that from bootstrap synthesis.

Index Terms—Electronic design automation, large language models, logic synthesis, retrieval-augmented generation.

## I. INTRODUCTION

L <sup>OGIC</sup> <sup>synthesis</sup> <sup>transforms</sup> <sup>RTL</sup> <sup>designs</sup> <sup>into</sup> <sup>gate-level</sup> netlists. As shown in Fig. 1, a typical flow comprises translation, logic optimization, and technology mapping, where the latter two stages involve extensive algorithmic choices that commercial tools encapsulate into large command sets with diverse configurable parameters. Prior studies show that the choice and ordering of commands significantly affect PPA outcomes [1], making synthesis tuning a highdimensional and high-cost design-space exploration challenge. Among the PPA metrics, timing is particularly critical, as violations can necessitate flow re-tuning and additional place-androute iterations, thereby substantially prolonging the design cycle [2].

Prior work on logic synthesis flow optimization falls into two paradigms, as shown in the upper right of Fig. 1. The first treats synthesis tuning as combinatorial search over AIGlevel operator sequences on ABC [3], with methods spanning manual tuning, reinforcement learning (RL) [4], Bayesian optimization (BO) [5], [6], Monte Carlo Tree Search (MCTS) [7], and bandits [8]. Although effective, these methods are confined to fixed action spaces, rely on black-box rewards, and provide limited interpretability regarding how their decisions relate to the current circuit state. These limitations become more pronounced on commercial tools with far richer command vocabularies. The second paradigm emerges from the growing use of large language models (LLMs) in EDA. Works such as ChatEDA [9], ChipNeMo [10], and ChatLS [11] generate complete Tcl scripts or command sequences for commercial tools directly from user intent and tool documentation, thereby avoiding a fixed action space. However, these end-to-end methods rely on “one-shot” generation and produce complete static scripts upfront. As a result, later commands cannot adapt to evolving circuit states after each command, potentially leading to suboptimal results.

![](images/0548696c11ac8ae857d77b888932ae5e3f976249f77180f484b53e3dff82ee26.jpg)  
Fig. 1 Traditional synthesis flow vs. our LLM-based agent SynAct. “IR” denotes intermediate representation.

A key observation is that commercial synthesis tools support a continuous interaction model: the tool session remains open across commands, allowing an agent to issue commands and observe updated PPA reports after each iteration, making synthesis a natural fit for a closed-loop LLM agent. This suggests that rather than generating a script upfront or searching over a fixed action space, the agent should interact directly with the tool, diagnosing the current circuit state and retrieving relevant knowledge at each iteration to guide its next decision. However, two challenges arise: (1) at each iteration, the LLM must locate relevant content within extensive command documentation, where irrelevant information can cause reasoning errors; (2) pure LLM reasoning lacks systematic reuse of prior experience, making it difficult to reuse commands proven effective in previous iterations and leading to inefficient longhorizon exploration.

To address these challenges, we present SynAct, an adaptive closed-loop LLM reasoning–acting agent that iteratively optimizes PPA on a commercial synthesis tool, with particular emphasis on WNS. At each iteration, SynAct conditions its decision on the latest timing reports and diagnostic outputs, mitigating information overload through a multilayer GraphRAG module [12] that retrieves scenario-relevant command documentation on demand. To improve exploration efficiency, Bayesian optimization over a GrammarVAE latent space leverages historical optimization experience accumulated in earlier iterations. Built on ReAct [13] for coupled reasoning and action and AutoGen-style multi-agent collaboration [14] for role coordination, SynAct continuously interprets circuit feedback and produces each optimization command with an explicit rationale, as illustrated in Fig. 1. Our main contributions are as follows:

1) A closed-loop LLM agent that iteratively optimizes PPA by adapting decisions to the current circuit state.

2) A multi-layer GraphRAG module for scenario-relevant knowledge retrieval, mitigating information overload.

3) A BO-guided experience refinement mechanism over a GrammarVAE latent space for experience-guided exploration.

4) SynAct reduces average WNS to 27% of that from bootstrap synthesis while maintaining balanced area and power trade-offs.

The remainder of this paper is organized as follows. Section II reviews the necessary background. Section III presents the SynAct framework, and Section IV details its knowledgeand experience-guidance modules. Section V provides a practical example of the framework in operation. Section VI reports the experimental results, and Section VII concludes the paper.

## II. PRELIMINARIES

## A. Logic Synthesis

Logic optimization and technology mapping are key stages in logic synthesis, respectively restructuring Boolean networks and mapping them to library cells under timing, area, and power constraints. Numerous algorithms have been proposed for these tasks [3]. Commercial synthesis tools offer a much richer command space that includes combinational and sequential optimization, retiming, power-aware optimization, and physically aware synthesis [15]. These capabilities create a larger and more context-dependent action space that must be explored effectively to achieve high-quality PPA results.

Because each command alters the circuit state and subsequent actions rely on the updated netlist and PPA reports, synthesis optimization can be formulated as a Markov Decision Process (MDP) $\mathfrak { M } = ( \mathfrak { X } , \mathcal { A } , \mathcal { F } , \mathcal { R } )$ , forming the basis of our agent design.

• X is the state space, where each state $x _ { t }$ includes the current netlist, PPA summary, and detailed timing, area, and power reports.

• A is the action space of executable synthesis commands.

$\mathcal { F } : \mathcal { X } \times \mathcal { A }  \mathcal { X }$ is the transition function realized by the synthesis tool under fixed settings.

$\mathcal { R } : \mathcal { X } \times \mathcal { A } $ R is the reward function, which uses a weighted aggregation of WNS and other PPA metrics to capture timing, area, and power trade-offs.

Bootstrap synthesis is the initialization flow that configures the technology library, reads and elaborates RTL, applies timing constraints, and runs the initial optimization. It excludes recipe-level tuning and yields the common initial state $x _ { 0 }$ and reference PPA for all compared methods on each design. Starting from $x _ { 0 } .$ , the agent iteratively selects one command $a _ { t } .$ executes it within the tool session, and observes the resulting reports as the next state $x _ { t + 1 }$ until the optimization target is met or the budget is exhausted.

## B. LLMs for EDA

LLM-based EDA increasingly combines knowledge retrieval with agentic tool interaction. Retrieval-Augmented Generation (RAG) [16] grounds generation in retrieved knowledge, and its retrieval models can be fine-tuned on EDA tool documentation to improve domain-specific retrieval [12]. Beyond retrieval, ReAct [13] interleaves reasoning and action so that observations inform later decisions. AutoGen [14] supports coordination among specialized agents through structured conversations in complex workflows. In practice, ChatEDA [9] decomposes user requests before generating and executing RTL-to-GDSII scripts, while ChipNeMo [10] combines domain adaptation and retrieval for tasks including EDA script generation. ChatLS [11] uses retrieved tool knowledge and chain-of-thought prompting to customize synthesis scripts, whereas LLSM [17] uses EDA-guided prompting to integrate circuit information extracted from RTL with structural features. These studies show that LLMs can retrieve domain knowledge, reason about design tasks, coordinate specialized agents, and invoke EDA tools, laying the foundation for SynAct’s closed-loop PPA optimization.

## III. SYNACT FRAMEWORK

## A. Overview

The left side of Fig. 2 shows the overall flow. Given an RTL design and designer inputs, SynAct first performs bootstrap synthesis to establish the initial circuit state. It then iterates through LLM-guided command generation, optimization execution, and termination checking until the request is satisfied or the iteration budget $n _ { \mathrm { i t e r } }$ is exhausted. The resulting netlist is then written out.

The right side lists the designer inputs, including a request, the command manual, and environment variables. The request specifies the optimization preferences and quantitative PPA targets, while the manual and variables provide tool-specific information for command generation. SynAct can prioritize any selected PPA metric. In this work, we use WNS as the primary objective and treat the remaining metrics as secondary objectives.

![](images/5ac47366a9b4f420bea137df7f683814526eb56bbfe98a036df9c3b8f8fcf747.jpg)  
Fig. 2 The overall framework of SynAct.

The center of Fig. 2 details three main components of SynAct. Inspired by ReAct [13], the Analysis Agent examines the current reports and produces a diagnosis. The Optimization Agent combines this diagnosis with retrieved tool knowledge and historical experience to generate command candidates. Candidate Selection chooses the best safe command from the evaluated candidates for execution. Executing the command updates the circuit state, and the resulting reports inform the next iteration, forming a closed reasoning–acting loop. The following subsections detail these three components.

## B. Analysis Agent

Existing search-based methods guide optimization primarily using scalar PPA summaries [5], [6], [8], but these summaries provide limited information, making it difficult to uncover the circuit-level causes of PPA violations. SynAct therefore introduces an Analysis Agent that inspects detailed timing, area, and power reports to diagnose potential performance bottlenecks through two stages: Preprobe and Postprobe.

In Preprobe, the agent parses raw report logs of the current design to identify performance bottlenecks. When the available data lacks sufficient detail, it proactively constructs follow-up probes using analyze\_<sub>\*</sub> commands described in the tool documentation to obtain the necessary information. For precise and context-aware diagnosis, the agent is guided to generate targeted commands for specific design objects such as clock domains, timing constraints, circuit instances, or buffer chains through the following prompt:

System: role = analysis agent (PREPROBE); diagnose root   
causes and propose analyze\_<sub>\*</sub> probes; prefer object-level Tcl  
braced probes   
User: user request, current/previous metrics, raw report log, and   
preprobe policy   
Expected JSON: { summary, recommended probes }

A typical output may include probes such as:   
analyze\_clock\_timing -type latency   
analyze\_inst -pin {...}   
analyze\_buffer\_chain -from {...}

In Postprobe, the outputs collected by executing the probes from the previous stage are incorporated into the user prompt to help the agent refine its diagnosis. The refined result is then returned as a structured diagnosis that forms part of the Optimization Agent’s input. For example, it may conclude that one path group dominates the timing bottleneck and that the next actions should focus on buffer optimization, cell sizing, or local restructuring under area and power guardrails. The corresponding Postprobe prompt is:

System: role = analysis agent (POSTPROBE); refine diagnosis   
using collected probe evidence   
User: user request, raw report text, extra analysis reports   
Expected JSON: summary, goals, violations,   
module hypotheses, suggested strategies, reflection }

## C. Optimization Agent

The Optimization Agent takes the structured diagnosis and user request as input and generates executable synthesis command candidates. This constitutes our second key design:

rather than optimizing over a small fixed action set, SynAct performs context-aware command generation within a large, structured command space. This generation process is supported by two complementary guidance modules, which are detailed in Section IV:

• RAG-grounded candidates ${ \mathcal { C } } _ { \mathrm { R A G } }$ , which are derived from scenario-relevant manual sections retrieved by GraphRAG;

• BO-seeded candidates $\mathcal { C } _ { \mathrm { B O } }$ , which adapt historically effective command patterns with minimal syntax changes. By combining these sources, the agent generates commands that reflect the current diagnosis while leveraging retrieved knowledge and prior experience. Each candidate is accompanied by a concise rationale. The corresponding structured prompt is:

System: role = optimization agent; generate command candidates User: user request, Postprobe diagnosis, RAG-retrieved manual sections, BO seed recommendation Expected JSON: { recommended candidates, rationale }

## D. Candidate Selection

Given the candidates produced by the Optimization Agent, SynAct first performs safety filtering, removing those whose WNS falls beyond a preset threshold $( \mathrm { e . g . , - 2 0 }$ ps) or whose reward drops below $0 . 2 \times r _ { \mathrm { b o o t } }$ , where $r _ { \mathrm { b o o t } }$ is the initial reward from bootstrap synthesis. Among the remaining safe candidates, it then selects the one with the highest score (defined below). If no candidate passes filtering, the previous checkpoint is retained.

Each candidate command C is evaluated by the score

$$
\operatorname { s c o r e } ( C ) = r ( C ) + \kappa \sigma ( z ) - \rho ( C ) ,\tag{1}
$$

where $r ( C )$ is the reward measuring objective satisfaction, κ controls exploration strength, and $\rho ( C )$ penalizes recent repetitions (waived for large WNS improvements). Here $\sigma ( z )$ represents a local uncertainty estimate from the BO surrogate at latent point z, encouraging exploration of under-explored regions. All candidates, whether from $\mathcal { C } _ { \mathrm { B O } }$ or ${ \mathcal { C } } _ { \mathrm { R A G } }$ , are encoded into this shared latent space, enabling the surrogate to generalize learned experience and provide consistent exploration guidance across both sources. The latent space and surrogate model are detailed in Section IV-B. This score balances high reward, moderate exploration, and redundancy avoidance.

The reward $r ~ \in ~ ( 0 , 1 ]$ aggregates violations across all objectives:

$$
r = \exp \left( - \sum _ { i } w _ { i } \log ( 1 + \operatorname { e r r } _ { i } ) \right) ,\tag{2}
$$

$$
\mathrm { e r r } _ { i } = \left\{ \begin{array} { l l } { \mathrm { m a x } ( 0 , \mathrm { t a r g e t } _ { i } - \mathrm { m e t r i c } _ { i } ) , } & { i \in \{ \mathrm { W N S } , \mathrm { T N S } \} , } \\ { \displaystyle \frac { \mathrm { m a x } ( 0 , \mathrm { m e t r i c } _ { i } - \mathrm { t a r g e t } _ { i } ) } { \vert \mathrm { t a r g e t } _ { i } \vert } , } & { i \in \{ \mathrm { a r e a } , P _ { \mathrm { d y n } } , P _ { \mathrm { s t a t } } \} , } \end{array} \right.\tag{3}
$$

![](images/e227a0e389a517465bfc6af668ea2e3210730e8764dc9a4f203a7014a6fa7610.jpg)  
Fig. 3 Three-layer structure of the proposed GraphRAG framework. The optimization scenario layer (center) connects related commands (left) and configurable variables (right), forming inter- and intra-layer relationships.

where target is the user-specified target and metric is the current measured value. Timing metrics (WNS, TNS) are maximized, while area and power are minimized and normalized by target magnitude. The $\log ( 1 + \exp _ { i } )$ term compresses large violations to reduce their influence in the aggregate reward. Each weight $w _ { i }$ is assigned a base value, upweighted for the user-selected primary objective, and then normalized to ensure $\textstyle \sum _ { i } w _ { i } = 1$ . Thus $r = 1$ when all targets are met and decreases faster for larger or higher-priority violations.

## IV. KNOWLEDGE- AND EXPERIENCE-GUIDED OPTIMIZATION

The Optimization Agent is supported by two complementary modules shown in Fig. 2: GraphRAG grounds candidate generation in scenario-relevant tool knowledge, while BO reuses historical synthesis experience in a GrammarVAE latent space. This section details these two modules.

## A. GraphRAG-Based Knowledge Grounding

The Optimization Agent requires precise, context-aware domain knowledge to generate effective synthesis commands. A natural approach is to retrieve relevant documentation via $\mathrm { R A G } ;$ however, forming effective synthesis strategies requires connecting applicable scenarios, specific commands, and configurable variables scattered across disparate documentation sections. This structured, multi-hop reasoning process goes beyond the ability of traditional vector-similarity retrieval. To address this, we develop a multi-layer GraphRAG module that unifies graph-based reasoning and retrieval augmentation [18], [19], detailed in the following four parts.

Document Preprocessing. Building the GraphRAG module begins with extracting relevant content from the synthesis tool documentation. We manually curate document sections related to optimization commands and categorize them into three functional roles:

1) Applicable scenarios: describe design goals (e.g., timing, area, power) with typical associated commands and variables;

2) Executable commands: define specific tool operations, including syntax, structure, and invocation patterns;

3) Configurable variables: specify tunable parameters, valid ranges, and default configurations.

This taxonomy follows practical EDA usage: optimizations are scenario-driven, executed through commands, and tuned by variables. Based on this categorization, we organize the extracted content into a three-layer graph of scenarios, commands, and variables.

Entity and Relationship Extraction. An LLM is prompted to convert the unstructured text into structured entities belonging to the three layers defined above. We introduce three disjoint sets of entities corresponding to the three layers, where $\mathcal { E } _ { S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n _ { S } } \} , \mathcal { E } _ { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { n _ { C } } \}$ , and ${ \mathcal { E } } _ { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n _ { V } } \}$ represent scenarios, commands, and variables, respectively.

The semantic structure of the knowledge graph is built through two complementary processes: intra-layer and interlayer relationship identification. For intra-layer relationships, we use an LLM-based identification function to detect semantic connections within a single layer, defined as follows:

$$
\mathcal { R } ^ { ( i n t r a ) } = \mathcal { R } _ { C C } \cup \mathcal { R } _ { V V } ,\tag{4}
$$

$$
\mathcal { R } _ { C C } = \{ ( c _ { i } , c _ { j } , r _ { i j } ) \mid c _ { i } , c _ { j } \in \mathcal { E } _ { C } , \mathbf { L L M } _ { r e l } ( c _ { i } , c _ { j } ) = \mathrm { T r u e } \} ,\tag{5}
$$

$$
\begin{array} { r } { \mathcal { R } _ { V V } = \{ \left( v _ { i } , v _ { j } , r _ { i j } ^ { \prime } \right) \mid v _ { i } , v _ { j } \in \mathcal { E } _ { V } , \mathbf { L } \mathbf { L } \mathbf { M } _ { r e l } ( v _ { i } , v _ { j } ) = \mathrm { T r u e } \} . } \end{array}\tag{6}
$$

Here, $\mathrm { L L M } _ { r e l } ( \cdot , \cdot )$ determines whether two entities share a semantic dependency based on their descriptions and usage patterns.

For inter-layer relationships, scenarios are linked to related commands and variables that co-occur in documentation examples, formalized as follows:

$$
\mathcal { R } ^ { \left( i n t e r \right) } = \mathcal { R } _ { S C } \cup \mathcal { R } _ { S V } ,\tag{7}
$$

$$
\mathcal { R } _ { S C } = \{ ( s , c ) \ | \ s \in \mathcal { E } _ { S } , \ c \in \mathcal { E } _ { C } , \ c \in \mathcal { C } _ { s } ^ { ( c m d s ) } \} ,\tag{8}
$$

$$
\mathcal { R } _ { S V } = \{ ( s , v ) ~ | ~ s \in \mathcal { E } _ { S } , v \in \mathcal { E } _ { V } , v \in \mathcal { C } _ { s } ^ { ( v a r s ) } \} .\tag{9}
$$

A link $( s , c ) \ \mathrm { o r } \ ( s , v )$ is established when the LLM identifies co-occurrence between scenario s and the corresponding command or variable in application examples.

Hierarchical Knowledge Graph Construction. Based on the extracted entities and relations, we construct a hierarchical knowledge graph defined as:

$$
{ \mathcal { G } } = ( \mathcal { E } , { \mathcal { R } } ) ,\tag{10}
$$

where $\mathcal { E } = \mathcal { E } _ { S } \cup \mathcal { E } _ { C } \cup \mathcal { E } _ { V }$ denotes the set of all entities, which is partitioned into disjoint subsets of scenarios, commands, and variables, and $\mathcal { R } \overset { \cdot } { = } \mathcal { R } ^ { ( \mathrm { i n t r a } ) } \cup \mathcal { R } ^ { ( \mathrm { i n t e r } ) }$ represents intra-layer and inter-layer relations. The scenario layer acts as the central hub connecting the command and variable layers, providing the foundation for subsequent reasoning and retrieval, as illustrated in Fig. 3.

Scenario-Driven Retrieval. Before retrieval, the Analysis Agent diagnoses optimization bottlenecks and generates an analysis report A, which serves as the input query. Given A, GraphRAG retrieves relevant synthesis knowledge via four compact steps:

1) Scenario description generation: An LLM summarizes an optimization scenario description $d _ { \mathrm { s c e n e } }$ from A.

2) Semantic embedding: scenario description $d _ { \mathrm { s c e n e } }$ and scenario entity s are encoded into vectors using a domaincustomized text embedding model [12], fine-tuned on EDA tool documentation via supervised contrastive learning. This model better captures EDA terminology and semantics than general-purpose embeddings, ensuring all entities/descriptions in GraphRAG share a consistent EDA-aware vector space for precise retrieval.

3) Scenario selection: Top-k relevant scenarios are retrieved by semantic similarity:

$$
\mathcal { S } ^ { * } = \operatorname * { a r g m a x } _ { \mathcal { S } \subseteq \mathcal { E } _ { S } , | \mathcal { S } | = k } \sum _ { s \in \mathcal { S } } \sin ( \mathrm { E m b e d } ( d _ { \mathrm { s c e n e } } ) , \mathrm { E m b e d } ( s ) ) .\tag{11}
$$

4) Knowledge expansion: Retrieve commands and variables directly connected to ${ \mathcal { S } } ^ { * }$ as the initial associated set ${ \mathcal { C } } _ { 0 } / { \mathcal { V } } _ { 0 }$ , and further retrieve their $\mathrm { t o p } { - } k _ { \mathrm { i n t r a } }$ intra-layer neighbors via k-nearest neighbor (KNN) to form the final expanded set. Specifically:

$$
\begin{array} { r } { \mathfrak { C } _ { 0 } = \{ c \mid ( s , c ) \in \mathcal { R } _ { S C } , s \in \mathcal { S } ^ { * } \} , \mathfrak { C } ^ { * } = \mathfrak { C } _ { 0 } \cup \mathrm { K N N } ( \mathfrak { C } _ { 0 } , k _ { \mathrm { i n t r a } } ) , } \end{array}\tag{12}
$$

$$
\begin{array} { r } { \mathcal { V } _ { 0 } = \{ v \mid ( s , v ) \in \mathcal { R } _ { S V } , s \in \mathcal { S } ^ { * } \} , \mathcal { V } ^ { * } = \mathcal { V } _ { 0 } \cup \mathrm { K N N } ( \mathcal { V } _ { 0 } , k _ { \mathrm { i n t r a } } ) . } \end{array}\tag{13}
$$

Here, $k _ { \mathrm { i n t r a } }$ denotes the number of intra-layer neighbors retrieved for each seed entity.

The final retrieved knowledge set is $\mathcal { K } ~ = ~ \mathcal { S } ^ { * } \cup \mathcal { C } ^ { * } \cup \mathcal { V } ^ { * }$ providing the Optimization Agent with concise, scenariospecific, non-redundant EDA synthesis knowledge.

The hierarchical GraphRAG mechanism tightly couples knowledge grounding with design diagnosis, which allows the Optimization Agent to leverage structured, reliable synthesis knowledge for improved reasoning accuracy and decision reliability.

## B. BO-Guided Experience Refinement

Commands validated in earlier iterations provide designspecific evidence for promising optimization directions [5], [6], [8]. Inspired by this, the Optimization Agent reuses executed commands and their feedback as experience. In practice, reapplying a previously effective command can yield further gains and sometimes outperform one newly generated from retrieved documentation. This motivates an experienceguided search based on Bayesian optimization (BO), which uses accumulated evaluations to efficiently identify promising commands.

Applying BO directly to discrete command strings remains inefficient because the search space is large and similar commands can exhibit different optimization behavior. Inspired by grammar-based VAEs that map discrete grammatical structures into continuous spaces [20], [21], we adapt GrammarVAE to encode synthesis commands for BO. This representation enables smooth optimization while preserving syntactic validity [22]. BO then iteratively fits a surrogate model, proposes and evaluates candidates, and updates its data to focus on empirically effective regions.

![](images/b52b9533c444ad394ed1d238e258c73269f4e1c63a9244fe89a5cdf22d5d37a4.jpg)  
Fig. 4 GrammarVAE encoding and latent-space BO. A command is parsed into a grammar tree and encoded into latent space Z; BO then identifies $z _ { \mathrm { b e s t } }$ and proposes $z _ { \mathrm { n e x t } }$ within the trust region T, which is decoded back into a command recommendation for the Optimization Agent.

Algorithm 1 BO-Guided Experience Refinement   
Input: Experience log D, trust region T, last commit reward   
$r _ { \mathrm { p r e v } } { \mathrm { . } }$ , diagnosis Φ, RAG docs R   
Output: Updated log $\mathrm { { \mathcal { D } ^ { \prime } } } ,$ updated trust region $\mathcal { T } ^ { \prime }$ , BO seed   
$\hat { C } _ { \mathrm { B O } }$   
1: $/ { } ^ { * }$ Surrogate Fitting \*/   
2: $w _ { i } \gets \operatorname* { m i n } ( w _ { \mathrm { m a x } } , 1 + \lambda$ max $\left( 0 , r _ { i } \right) )$ for each $( z _ { i } , r _ { i } ) \in \mathcal { D }$   
3: Refit RBF surrogate on D to obtain $\mu ( z ) , \sigma ( z )$   
4: $/ { } ^ { * }$ UCB Acquisition and Latent Decoding \*/   
5: Sample pool $\mathcal { P } \subset \mathcal { T }$   
6: $z _ { \mathrm { n e x t } } \gets \arg \operatorname* { m a x } _ { z \in \mathcal { P } } \left[ \mu ( z ) + \kappa _ { \mathrm { a c q } } \sigma ( z ) \right]$   
7: $\hat { C } _ { \mathrm { B O } }  \mathrm { D e c } ( z _ { \mathrm { n e x t } } )$ ▷ GrammarVAE decode   
8: C ← OptAgent(C<sup>ˆ</sup><sub>BO</sub>, Φ, R)   
9: $/ { } ^ { * }$ Evaluation and Trust Region Update $^ { * }$   
10: $\mathcal { D } ^ { \prime }  \mathcal { D }$   
11: for $C \in \mathcal { C }$ do   
12: Execute $C ;$ record $r ( C )$   
13: z ← Enc(Norm(C)) ▷ GrammarVAE encode   
14: $\mathcal { D } ^ { \prime }  \mathcal { D } ^ { \prime } \oplus ( z , r ( C ) )$ ▷ ⊕: insert or replace   
15: end for   
16: ${ \mathcal { C } } _ { \mathrm { s a f e } } \gets \{ C \in { \mathcal { C } } | C$ passes safety filter} ▷ Section III-D   
17: C<sup>⋆</sup> ← arg m $\mathrm { a x } _ { C \in { \mathcal { C } } _ { \mathrm { s a f e } } }$ score(C) ▷ see Equation (1)   
18: $z _ { \mathrm { b e s t } } \gets \mathrm { E n c } ( \mathrm { N o r m } ( C ^ { \star } ) )$   
19: Recenter $\mathcal { T } ^ { \prime }$ on $z _ { \mathrm { b e s t } } ;$ expand if $r ( C ^ { \star } ) > r _ { \mathrm { p r e v } }$ , else shrink   
20: return $( \mathbb { D } ^ { \prime } , \mathbb { T } ^ { \prime } , \hat { C } _ { \mathrm { B O } } )$

Latent Encoding. Each synthesis command C is normalized by Norm(·) and encoded into a latent vector $z =$ $\mathrm { E n c } ( \mathrm { N o r m } ( C ) ) \ \in \ \mathbb { R } ^ { d }$ via the GrammarVAE encoder [20], where z is a point in latent space $\mathcal { Z } \subseteq \mathbb { R } ^ { d }$ . As shown in Fig. 4, a command is first parsed into a context-free grammar tree, then compressed into z, such that nearby points in $\mathcal { Z }$ correspond to grammatically similar commands. Because synthesiscommand syntax encodes the optimization mode, transformation scope, effort level, and parameter configuration, these grammar-preserving neighborhoods provide a structured local prior that BO calibrates with measured synthesis rewards. GrammarVAE is pre-trained offline on a corpus of synthesis commands using the standard VAE objective (reconstruction and KL regularization); during BO, each command is mapped to the encoder’s posterior mean for a consistent latent representation.

Definition 1 (Trust Region T). BO limits its search to an axisaligned trust region ${ \mathcal { T } } \subseteq { \mathcal { Z } } ,$ centered at the latent point of the current best command $z _ { \mathrm { b e s t } } ,$ to focus optimization on the most promising area. The radius of T adapts across iterations: it expands when performance improves and shrinks otherwise.

Definition 2 (Experience Log D). The experience log $\mathcal { D } =$ $\{ ( z _ { i } , r _ { i } ) \} _ { i = 1 } ^ { N }$ stores the latent encoding and reward of each executed command $C _ { i } ,$ , with $z _ { i } ~ = ~ \operatorname { E n c } ( \operatorname { N o r m } ( C _ { i } ) )$ and $r _ { i }$ defined in Equation (2). During bootstrap synthesis, D is initialized from the default optimize command, and T is centered at its latent point.

Algorithm 1 illustrates how the refinement procedure is executed within one iteration: it takes D, T, last commit reward $r _ { \mathrm { p r e v } } .$ , the Analysis Agent’s diagnosis Φ, and RAGretrieved documents R as input, and returns updated D<sup>′</sup>, $\mathcal { T } ^ { \prime }$ and BO seed $\hat { C } _ { \mathrm { B O } }$

Surrogate Fitting. At each iteration, a reward-weighted RBF surrogate [23] is refit from scratch on the full D (lines 2– 3). Each observation $( z _ { i } , r _ { i } )$ is assigned a weight $\begin{array} { r l } { w _ { i } } & { { } = } \end{array}$ min $( w _ { \mathrm { m a x } } , 1 + \lambda \operatorname* { m a x } ( 0 , r _ { i } ) )$ , where $\lambda ~ \geq ~ 0$ scales how aggressively positive rewards upweight a point in the fit and $w _ { \mathrm { m a x } }$ caps each $w _ { i } .$ , so that high-reward regions receive greater influence while no single sample dominates. The surrogate estimates the expected reward and local uncertainty through kernel-weighted averages:

$$
\mu ( z ) = \frac { \sum _ { i } w _ { i } K ( z , z _ { i } ) r _ { i } } { \sum _ { i } w _ { i } K ( z , z _ { i } ) } , \ : \sigma ^ { 2 } ( z ) = \frac { \sum _ { i } w _ { i } K ( z , z _ { i } ) [ r _ { i } - \mu ( z ) ] ^ { 2 } } { \sum _ { i } w _ { i } K ( z , z _ { i } ) \sum _ { \ell = 1 } \ : \ : _ { i } } ,\tag{14}
$$

where $K ( z , z _ { i } ) = \exp ( - \| z - z _ { i } \| ^ { 2 } / 2 \ell ^ { 2 } )$ is the RBF kernel. The resulting $\sigma ( z )$ is smoothed and clipped for numerical stability and serves as the local uncertainty estimate used by the commitment score described in Section III-D.

UCB Acquisition and Latent Decoding. BO samples a pool $\Phi \subset \mathcal { T }$ and selects the next latent point by maximizing the UCB acquisition function [24] (lines 5–6):

$$
\alpha ( z ) = \mu ( z ) + \kappa _ { \mathrm { a c q } } \sigma ( z ) ,\tag{15}
$$

where $\mu ( z )$ exploits known high-reward regions and $\kappa _ { \mathrm { a c q } } \sigma ( z )$ promotes exploration of regions with high local uncertainty. The selected $z _ { \mathrm { n e x t } }$ is decoded by GrammarVAE into a command string $\hat { C } _ { \mathrm { B O } }$ (line 7) and passed to the Optimization Agent as the BO seed recommendation. While GrammarVAE decoding enforces grammatical validity, the decoded command may still contain tool-level errors (e.g., unsupported flags or invalid parameter ranges). The Optimization Agent retains $\hat { C } _ { \mathrm { B O } }$ as an explicit conditioning input and refines it with retrieved tool documentation to produce $\mathcal { C } _ { \mathrm { B O } }$ , while independently generating ${ \mathcal { C } } _ { \mathrm { R A G } }$ from the retrieved knowledge; together they form the full candidate set $\mathcal { C } = \mathcal { C } _ { \mathrm { B O } } \cup \mathcal { C } _ { \mathrm { R A G } }$ (line 8).

Evaluation and Trust Region Update. All candidates in C are executed from the same parent checkpoint and their rewards recorded (lines 11–12). Each candidate is then reencoded and logged into $\mathrm { \textmathcal { D } ^ { \prime } }$ (lines 13–14), including those that fail or yield poor results, as broader latent-space coverage helps BO distinguish effective from ineffective regions and improve future proposals. The committed command $C ^ { \star }$ is selected by first filtering C for safe candidates $\mathcal { C } _ { \mathrm { s a f e } }$ (line 16), and then maximizing the score function score(C) defined in Equation (1) (line 17). The trust region $\mathcal { T } ^ { \prime }$ is recentered on $z _ { \mathrm { b e s t } } = \operatorname { E n c } ( \operatorname { N o r m } ( C ^ { \star } ) )$ and expanded if $r ( C ^ { \star } ) > r _ { \mathrm { p r e v } } ,$ otherwise shrunk (lines 18–19).

## V. RUNNING EXAMPLE

To illustrate the practical use of SynAct, Fig. 5 presents a five-iteration run on arm9. Starting from the initial synthesis report, SynAct follows a reasoning, acting, and observation loop. At each iteration, the LLM interprets the tool feedback, explains the optimization intent, and produces an executable Tcl command. After execution, the updated circuit state is reflected in new synthesis reports, which serve as observations for the next iteration.

In the first two iterations, the LLM follows BO guidance and selects incremental timing optimization to limit changes to the current implementation. It then uses updated timing observations and retrieved tool knowledge to select tdopt, register retiming, and phase assignment. The figure records the reasoning, committed command, and PPA observation at each iteration. Over the five iterations, WNS changes from −20.85 ps to −3.70 ps.

## VI. EXPERIMENT

## A. Experimental Setup

All experiments are conducted on a workstation equipped with an Intel® Xeon® Gold 6226R CPU (2.90 GHz) and an NVIDIA GeForce RTX 3090 GPU. SynAct is implemented in Python with PyTorch for neural modules and connects to large language models via an OpenAI-compatible API supporting both commercial and internal endpoints. It integrates with $\mathrm { \ A l t i S y n ^ { \textregistered } }$ , the commercial logic synthesis tool from ZeniSyn Design Systems [25], for automated command execution and report collection. The synthesis flow employs the standard-cell library from the 7 nm ASAP7 edge-cut PDK [26].

SynAct is implemented as a multi-agent system comprising an Analysis Agent for report-driven diagnosis and an Optimization Agent for command generation. The Optimization

![](images/8ef3c490dae3f548a9e9b1aa9f629e0d22a0cd606e312b70527201416400ac50.jpg)  
Fig. 5 An actual SynAct run on arm9, showing the reasoning, committed action, and tool observation at each iteration.

Agent performs Bayesian optimization in a GrammarVAE latent space to efficiently explore discrete synthesis commands and utilizes a Neo4j-based multi-layer GraphRAG module for structured knowledge retrieval. Semantic retrieval within this module uses the transformer model fine-tuned for EDA documentation in [12].

For evaluation, we use 14 open-source RTL designs collected from OpenCores [27]. TABLE I summarizes the benchmark statistics obtained after importing each design and running a bootstrap synthesis pass, including the number of nets, leaf, sequential, buffer, and inverter cell instances, and the configured clock period (CPD). Appropriate clock periods are assigned according to design size and fixed across all methods to ensure sufficient optimization margin. Across all experiments, we set WNS as the primary optimization objective, with quantitative targets specified at the input stage (a configuration example is shown in Fig. 2). Additionally, we track TNS, area, dynamic power, and static power as secondary metrics to assess overall PPA quality. To account for stochastic variation, each experiment is repeated five times, and the average results are reported unless otherwise specified.

TABLE I Benchmark statistics.
<table><tr><td>Benchmarks</td><td>#Nets</td><td>#Leaf</td><td>#Seq.</td><td>#Buf.</td><td>#Inv.</td><td>CPD.</td></tr><tr><td>uart16550</td><td>3057</td><td>3006</td><td>605</td><td>143</td><td>336</td><td>310</td></tr><tr><td>picorv32</td><td>9193</td><td>9091</td><td>1556</td><td>268</td><td>688</td><td>390</td></tr><tr><td>yacc</td><td>10504</td><td>10363</td><td>1102</td><td>487</td><td>1285</td><td>560</td></tr><tr><td>aes_core</td><td>11816</td><td>11557</td><td>530</td><td>521</td><td>1091</td><td>365</td></tr><tr><td>wb2axip</td><td>12539</td><td>11381</td><td>1900</td><td>257</td><td>1221</td><td>335</td></tr><tr><td>wb_conmax</td><td>18170</td><td>17040</td><td>770</td><td>492</td><td>1354</td><td>365</td></tr><tr><td>arm9</td><td>19267</td><td>19195</td><td>1222</td><td>823</td><td>2634</td><td>1050</td></tr><tr><td>sha3</td><td>24927</td><td>24161</td><td>2949</td><td>674</td><td>2903</td><td>365</td></tr><tr><td>spimaster</td><td>38365</td><td>38313</td><td>8705</td><td>862</td><td>3909</td><td>320</td></tr><tr><td>ethernet</td><td>44765</td><td>44637</td><td>10543</td><td>1097</td><td>3390</td><td>385</td></tr><tr><td>ecg</td><td>53613</td><td>53071</td><td>7676</td><td>1116</td><td>7807</td><td>340</td></tr><tr><td>linkruncca</td><td>69138</td><td>69134</td><td>16606</td><td>1720</td><td>4879</td><td>480</td></tr><tr><td>xge_mac</td><td>81564</td><td>81358</td><td>21667</td><td>29</td><td>5896</td><td>135</td></tr><tr><td>fft256</td><td>198981</td><td>198897</td><td>42432</td><td>3129</td><td>21095</td><td>490</td></tr></table>

<sup>1</sup> CPD: Clock period (ps).

TABLE II Original PPA metrics after bootstrap synthesis.
<table><tr><td>Benchmarks</td><td>WNS(ps)</td><td>TNS(ps)</td><td>Area (µm2)</td><td>DP (mW)1</td><td>SP (µW)2</td></tr><tr><td>uart16550</td><td>-7.45</td><td>-10.10</td><td>400.94</td><td>1.572</td><td>0.392</td></tr><tr><td>picorv32</td><td>-15.70</td><td>-3132.97</td><td>1128.68</td><td>3.529</td><td>1.248</td></tr><tr><td>yacc</td><td>-26.86</td><td>-1592.03</td><td>1153.77</td><td>4.095</td><td>1.300</td></tr><tr><td>aes_core</td><td>-8.03</td><td>-482.04</td><td>1192.06</td><td>3.695</td><td>1.316</td></tr><tr><td>wb2axip</td><td>-20.54</td><td>-12344.79</td><td>1325.09</td><td>5.158</td><td>1.169</td></tr><tr><td>wb_conmax</td><td>-11.35</td><td>-3840.85</td><td>1698.47</td><td>4.399</td><td>1.481</td></tr><tr><td>arm9</td><td>-20.85</td><td>-419.62</td><td>2031.81</td><td>1.821</td><td>2.078</td></tr><tr><td>sha3</td><td>-14.41</td><td>-3446.74</td><td>2892.50</td><td>17.930</td><td>2.874</td></tr><tr><td>spimaster</td><td>-16.96</td><td>-151.50</td><td>5459.98</td><td>11.290</td><td>6.182</td></tr><tr><td>ethernet</td><td>-9.83</td><td>-213.09</td><td>6435.68</td><td>23.480</td><td>7.433</td></tr><tr><td>ecg</td><td>-5.04</td><td>-205.58</td><td>5873.96</td><td>22.760</td><td>5.765</td></tr><tr><td>linkruncca</td><td>-41.40</td><td>-146498.44</td><td>10380.14</td><td>28.330</td><td>12.290</td></tr><tr><td>xge_mac</td><td>-5.86</td><td>-43.86</td><td>12747.57</td><td>7.556</td><td>14.090</td></tr><tr><td>fft256</td><td>-17.14</td><td>-550.53</td><td>26638.81</td><td>72.630</td><td>31.600</td></tr><tr><td>Ratio Avg.</td><td>100.00%</td><td>100.00%</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr></table>

<sup>1</sup> Dynamic Power. <sup>2</sup> Static Power.

## B. Overall PPA Improvement

Comparison setup and method reproduction. TABLE II reports the PPA metrics after the mandatory bootstrap synthesis, which configures the technology library and timing constraints, reads and elaborates each RTL design, and runs the default optimize command without additional recipelevel tuning. These metrics serve as the reference for subsequent comparisons. The “Ratio Avg.” is the mean of the perdesign ratios between each method and bootstrap synthesis. Each ratio is computed by dividing the method result by its bootstrap value. For WNS and TNS, the reported value is first converted to its violation magnitude, max(0, −slack), so timing closure contributes zero. A lower “Ratio Avg.” indicates greater improvement.

TABLE III compares two baselines representing distinct paradigms of synthesis optimization. We set SynAct’s iteration limit to $n _ { \mathrm { { i t e r } } } = 5$ . To control optimization depth, all methods commit five actions, each consisting of a command or command sequence. This equalizes the committed-action budget but not the oracle-query budget, because the methods require different numbers of candidate evaluations, as detailed below. We therefore report the end-to-end runtime in TABLE IV alongside the PPA results.

ChatLS [11] is an LLM-based end-to-end script customization method that combines multimodal RAG and chain-ofthought reasoning. As its one-shot script cannot use feedback from intermediate synthesis results, it provides a natural counterpart to SynAct’s iterative optimization. Since its source code is unavailable, we reproduce ChatLS following the published methodology. Under its original seven-benchmark 45 nm FreePDK setup, our reproduction matches every reported result within 5%. We then port it without further tuning to our 14-design setup using ASAP7 and AltiSyn<sup>®</sup>, retaining the same prompts, retrieval configuration, and five-step script. The resulting comparison represents our controlled reproduction rather than the official implementation.

CBTune [8] is a LinUCB-based contextual bandit method originally developed for synthesis sequence generation in ABC [3]. We reproduce it using its public implementation. Unlike ChatLS and SynAct, LinUCB requires predefined, enumerable arms for reward estimation and action selection, so adapting CBTune to AltiSyn<sup>®</sup>requires bounding the tool’s much larger command space. We manually construct the seven entries in TABLE V from documented AltiSyn<sup>®</sup>commands and the optimization categories used by CBTune on ABC. They cover timing optimization, mapping, sizing, phase assignment, retiming, and resynthesis, and remain fixed across all 14 benchmarks without design-specific tuning. At each of five steps, CBTune evaluates the arms, updates LinUCB using short- and long-term rewards, and applies the selected arm.

SynAct performs five sequential iterations without a fixed action list, enabling it to reason over a broader command space than CBTune. In each iteration, the Analysis Agent diagnoses the latest synthesis reports, and the Optimization Agent combines this diagnosis with GraphRAG-retrieved documentation and BO-based experience to generate ten candidates. After synthesis evaluation, SynAct commits the best safe command to optimize the circuit and feeds the updated reports into the next iteration. Both agents use DeepSeek V3.1.

Optimization results with WNS as the primary objective. As shown in TABLE III, with WNS as the primary optimization objective, SynAct achieves the best overall timing improvement among all methods. In terms of “Ratio Avg.”, SynAct reduces the remaining WNS violation to 27.03% of that from bootstrap synthesis, substantially outperforming ChatLS [11] (71.73%) and CBTune [8] (66.67%). It achieves nonnegative WNS on two designs, whereas neither baseline does. Notably, for wb2axip, SynAct achieves WNS of -6.78 ps versus -9.23 ps (ChatLS) and -16.22 ps (CBTune); for arm9, it reaches -3.70 ps versus -8.25 ps and -8.44 ps, respectively. For picorv32 and wb conmax, SynAct further approaches timing closure with WNS values of 0.06 ps and 0.01 ps.

For secondary metrics, TNS is assigned lower priority than WNS and tends to improve as WNS becomes tighter. SynAct reduces TNS to 17.86% of the bootstrap synthesis result, compared with 96.43% for ChatLS and 77.14% for CBTune. In terms of area and power trade-offs, SynAct keeps metrics close to the bootstrap synthesis levels, with slight reductions in area and power (99.28% area, 98.92% dynamic power, 98.64% static power). These results show that SynAct achieves substantial improvements in both WNS and TNS while maintaining overall PPA balance and efficiency.

TABLE III Five-run average PPA comparison with WNS as the primary objective and TNS/area/power as secondary metrics.
<table><tr><td rowspan="2">Benchmarks</td><td colspan="2">WNS (ps)</td><td colspan="2"></td><td colspan="2">TNS (ps)</td><td colspan="3">Area (µm2)</td><td colspan="3">Dynamic Power (mW)</td><td colspan="3">Static Power (μW)</td></tr><tr><td>[11]</td><td>[8]</td><td>SynAct</td><td>[11]</td><td>[8]</td><td>SynAct</td><td>[11]</td><td>[8]</td><td>SynAct</td><td>[11]</td><td>[8]</td><td>SynAct</td><td>[11]</td><td>[8]</td><td>SynAct</td></tr><tr><td>uart16550</td><td>-6.62</td><td>-2.04</td><td>-1.18</td><td>-6.69</td><td>-2.04</td><td>-1.30</td><td>403.57</td><td>393.19</td><td>397.73</td><td>1.575</td><td>1.570</td><td>1.573</td><td>0.394</td><td>0.384</td><td>0.391</td></tr><tr><td>picorv32</td><td>-11.92</td><td>-2.94</td><td>0.06</td><td>-329.10</td><td>-268.32</td><td>0.00</td><td>1123.33</td><td>1146.73</td><td>1123.56</td><td>3.489</td><td>3.524</td><td>3.510</td><td>1.190</td><td>1.287</td><td>1.276</td></tr><tr><td>yacc</td><td>-12.42</td><td>-18.61</td><td>-4.38</td><td>-595.46</td><td>-996.20</td><td>-68.27</td><td>1155.38</td><td>1155.90</td><td>1159.37</td><td>4.115</td><td>4.162</td><td>4.150</td><td>1.306</td><td>1.333</td><td>1.304</td></tr><tr><td>aes_core</td><td>-9.83</td><td>-3.68</td><td>-3.41</td><td>-869.29</td><td>-108.98</td><td>-118.37</td><td>1249.87</td><td>1210.61</td><td>1194.67</td><td>4.020</td><td>3.765</td><td>3.703</td><td>1.390</td><td>1.369</td><td>1.316</td></tr><tr><td>wb2axip</td><td>-9.23</td><td>-16.22</td><td>-6.78</td><td>-4170.73</td><td>-8771.95</td><td>-3488.09</td><td>1393.82</td><td>1523.99</td><td>1328.59</td><td>5.498</td><td>5.573</td><td>5.162</td><td>1.222</td><td>1.687</td><td>1.173</td></tr><tr><td>wb_conmax</td><td>-12.20</td><td>-11.12</td><td>0.01</td><td>-3562.51</td><td>-4704.21</td><td>0.00</td><td>1700.23</td><td>1686.60</td><td>1684.22</td><td>4.141</td><td>4.392</td><td>4.040</td><td>1.368</td><td>1.469</td><td>1.314</td></tr><tr><td>arm9</td><td>-8.25</td><td>-8.44</td><td>-3.70</td><td>-120.85</td><td>-149.88</td><td>-37.42</td><td>2022.95</td><td>1971.41</td><td>2038.79</td><td>1.812</td><td>1.785</td><td>1.819</td><td>2.059</td><td>1.983</td><td>2.100</td></tr><tr><td>sha3</td><td>-14.33</td><td>-6.32</td><td>-3.82</td><td>-8183.41</td><td>-1849.45</td><td>-705.50</td><td>4079.15</td><td>2901.76</td><td>2837.75</td><td>30.540</td><td>15.380</td><td>17.210</td><td>3.979</td><td>3.261</td><td>2.761</td></tr><tr><td>spimaster</td><td>-13.51</td><td>-4.47</td><td>-3.04</td><td>-142.14</td><td>-35.01</td><td>-53.20</td><td>5448.90</td><td>5428.24</td><td>5415.65</td><td>11.280</td><td>11.270</td><td>11.240</td><td>6.165</td><td>6.166</td><td>6.128</td></tr><tr><td>ethernet</td><td>-3.56</td><td>-6.53</td><td>-2.85</td><td>-304.41</td><td>-67.81</td><td>-25.05</td><td>6438.15</td><td>6449.11</td><td>6406.14</td><td>23.620</td><td>23.280</td><td>23.320</td><td>7.080</td><td>7.465</td><td>7.378</td></tr><tr><td>ecg</td><td>-3.35</td><td>-4.56</td><td>-2.23</td><td>-22.11</td><td>-453.69</td><td>-33.09</td><td>5838.97</td><td>6304.30</td><td>5842.29</td><td>22.650</td><td>23.960</td><td>22.680</td><td>5.714</td><td>7.089</td><td>5.714</td></tr><tr><td>linkruncca</td><td>-6.40</td><td>-3.14</td><td>-6.19</td><td>-12595.18</td><td>-801.79</td><td>-5865.33</td><td>9832.23</td><td>9950.35</td><td>9876.56</td><td>27.390</td><td>27.770</td><td>27.660</td><td>11.520</td><td>11.870</td><td>11.730</td></tr><tr><td>xge_mac</td><td>-6.30</td><td>-7.04</td><td>-4.36</td><td>-34.54</td><td>-35.47</td><td>-25.77</td><td>12613.77</td><td>12622.96</td><td>12668.15</td><td>7.478</td><td>7.540</td><td>7.541</td><td>13.970</td><td>13.960</td><td>14.010</td></tr><tr><td>fft256</td><td>-12.68</td><td>-34.27</td><td>-7.90</td><td>-1805.46</td><td>-1794.78</td><td>-137.37</td><td>26628.94</td><td>26475.57</td><td>26617.73</td><td>72.660</td><td>72.700</td><td>72.660</td><td>31.590</td><td>31.010</td><td>31.570</td></tr><tr><td>Ratio Avg.</td><td>71.73%</td><td>66.67%</td><td>27.03%</td><td>96.43%</td><td>77.14%</td><td>17.86%</td><td>103.14%</td><td>101.02%</td><td>99.28%</td><td>105.33%</td><td>99.79%</td><td>98.92%</td><td>101.67%</td><td>105.51%</td><td>98.64%</td></tr></table>

<sup>1</sup> “Ratio Avg.” averages per-design metric ratios relative to bootstrap synthesis; lower values indicate greater improvement.

TABLE IV Runtime and token usage averaged over five runs.
<table><tr><td rowspan="2">Benchmark</td><td colspan="4">Total execution time (s)</td><td colspan="2">Token usage</td></tr><tr><td>Boot.1</td><td>[11]</td><td>[8]</td><td>SynAct</td><td>Anal.2</td><td>Opt.3</td></tr><tr><td>uart16550</td><td>19.02</td><td>187.93</td><td>428.59</td><td>425.20</td><td>43381</td><td>95823</td></tr><tr><td>picorv32</td><td>83.86</td><td>225.47</td><td>1617.28</td><td>783.49</td><td>43747</td><td>96444</td></tr><tr><td>yacc</td><td>101.57</td><td>225.55</td><td>2232.63</td><td>898.45</td><td>42262</td><td>96144</td></tr><tr><td>aes_core</td><td>93.88</td><td>351.18</td><td>2478.49</td><td>1078.54</td><td>43955</td><td>95322</td></tr><tr><td>wb2axip</td><td>84.40</td><td>265.93</td><td>2323.97</td><td>822.28</td><td>45103</td><td>95840</td></tr><tr><td>wb_conmax</td><td>121.26</td><td>258.71</td><td>3427.69</td><td>1383.27</td><td>44795</td><td>95416</td></tr><tr><td>arm9</td><td>617.47</td><td>980.17</td><td>10229.67</td><td>6941.71</td><td>45002</td><td>96493</td></tr><tr><td>sha3</td><td>248.03</td><td>515.75</td><td>5218.05</td><td>3242.30</td><td>43580</td><td>95474</td></tr><tr><td>spimaster</td><td>325.07</td><td>594.11</td><td>7005.90</td><td>2084.52</td><td>36585</td><td>95186</td></tr><tr><td>ethernet</td><td>327.66</td><td>848.57</td><td>6408.45</td><td>2706.38</td><td>51182</td><td>94415</td></tr><tr><td>ecg</td><td>253.50</td><td>567.90</td><td>5083.02</td><td>1850.21</td><td>37951</td><td>96071</td></tr><tr><td>linkruncca</td><td>1219.77</td><td>3379.13</td><td>24823.51</td><td>10079.57</td><td>42836</td><td>95464</td></tr><tr><td>xge_mac</td><td>381.86</td><td>866.17</td><td>7669.93</td><td>1573.92</td><td>40991</td><td>96447</td></tr><tr><td>fft256</td><td>1551.27</td><td>2169.11</td><td>29751.55</td><td>10167.44</td><td>41060</td><td>96651</td></tr><tr><td>Ratio Avg.</td><td>100.00%</td><td>289.83%</td><td>2174.18%</td><td>988.62%</td><td>=</td><td>=</td></tr></table>

<sup>1</sup> Bootstrap synthesis. <sup>2</sup> Analysis Agent. <sup>3</sup> Optimization Agent.

TABLE V Bounded action space for CBTune reproduction.  
ID Command   
1 optimize -mode timing -map\_effort high   
2 tdopt -flat -perturb -stop {1} -area 1.0;   
phase\_assign -mode area\_recovery   
3 optimize -mode timing -map\_effort high -incremental   
4 tdopt -flat -resyn2 -area 1.000000   
5 retime\_register -period 0 -effort medium; optimize   
-incremental   
6 impl\_select -mode down -scope s+ -tns; resize -mode   
normal -cost c>s>a -flat -area 1.0   
7 pre\_td; phase\_assign -flat -mode timing -effort   
medium; tdopt -flat -stop {10 1.0} -area 1.000000

Runtime and LLM API token cost. TABLE IV summarizes the end-to-end execution time of all methods. All methods begin with the same bootstrap synthesis stage, whose runtime is reported in the “Boot.” column and included in each method’s total runtime. The percentages represent end-to-end runtime normalized to the bootstrap time. ChatLS has the lowest normalized runtime, averaging 289.83% of the bootstrap synthesis time, because it generates the script in a single pass. CBTune has the highest runtime at 2174.18% due to its bandit-based exploration. SynAct strikes a favorable balance at 988.62%, less than half of CBTune’s runtime, while achieving the best WNS ratio of 27.03%, compared with 71.73% for ChatLS and 66.67% for CBTune. The additional runtime relative to single-pass ChatLS comes from feedback-driven candidate evaluation, which enables iterative refinement.

![](images/c642bc69f06c6110d5c520347ce2a051b18a4488162bb7ebe73f71df3b1b33ab.jpg)  
Fig. 6 Runtime breakdown of SynAct.

Fig. 6 further breaks down the internal runtime composition of SynAct. Candidate evaluation dominates the runtime at 84.85%. Each iteration evaluates ten LLM-generated candidates in parallel and commits the best safe action. This setting balances server capacity and robustness to LLM stochasticity: fewer candidates reduce synthesis cost, whereas more broaden exploration. Despite parallelism, repeated synthesis remains the primary bottleneck. Bootstrap synthesis, the required initialization step, accounts for the second-largest share at 10.4%. The Analysis and Optimization Agents account for 2.56% and 1.67%, respectively, while GrammarVAE encoding, decoding, and result processing account for the remaining 0.52%. Future work may consider early stopping upon WNS saturation or lightweight candidate filtering before execution to reduce unnecessary synthesis runs.

Token usage in TABLE IV is consistent across all 14 benchmarks. On average, SynAct consumes about 139k tokens per design, with 69% from the Optimization Agent and 31% from the Analysis Agent. The Optimization Agent uses more tokens because it processes more documents, even after RAG filtering. Total usage varies slightly from 132k to 146k, showing low cross-design variance and a stable ratio between the two agents.

![](images/4e9203fd91ddbeaed51f959aafac06ccd1de75f3d32ebea39d624dfac2ff5c6d.jpg)  
Fig. 7 Optimization trajectory of SynAct.

## C. Optimization Trajectory Analysis

Recall that SynAct incrementally optimizes a design over at most $n _ { \mathrm { i t e r } }$ LLM-guided iterations, where $n _ { \mathrm { i t e r } }$ is the tunable maximum number of iterations. We use $n _ { \mathrm { i t e r } } = 5$ in the main experiments in TABLE III to balance optimization quality and runtime cost. Fig. 7 evaluates the sensitivity to $n _ { \mathrm { i t e r } }$ by plotting optimization trajectories from $n _ { \mathrm { i t e r } } = 1$ onward. For readability, we show four representative benchmarks: ethernet, linkruncca, sha3, and wb2axip. Their clear WNS improvements and well-separated curves provide a concise view of SynAct’s overall optimization trend without the visual clutter of plotting all benchmarks.

The first iteration, labeled “LLM cold start,” refers to SynAct’s initial attempt to generate optimization commands that rely almost solely on RAG-retrieved documents without historical experience, often leading to suboptimal results. As iterations proceed, SynAct refines its strategy using updated analyses and synthesis feedback, improving WNS over time. Most trajectories trend upward, while minor fluctuations in linkruncca and sha3 reflect the exploratory nature of LLMguided optimization. Although additional iterations can further improve WNS when runtime is less constrained, $n _ { \mathrm { i t e r } } = 5$ provides a practical balance between optimization quality and runtime cost, supporting our default setting. Overall, these results demonstrate the effectiveness of iterative adaptation driven by cumulative synthesis feedback.

## D. Generalizability Across LLMs

To evaluate whether SynAct generalizes across different LLMs, we rerun the complete 14-design experiment using GPT-5.2 for both the Analysis and Optimization Agents under the same setup as TABLE III. As reported in TABLE VI, this configuration reduces the WNS and TNS Ratio Avg. to 20.37% and 13.69%, respectively, while keeping area and power within 1% of the bootstrap results. The default DeepSeek V3.1 configuration already achieves a WNS Ratio Avg. of 27.03% and substantially outperforms the baselines, confirming that SynAct’s closed-loop diagnosis, retrieval, and experience refinement remain effective with different LLMs. GPT-5.2 further enhances report interpretation, diagnosis, and command generation within this pipeline, yielding additional timing gains. These results show that SynAct provides the adaptive optimization mechanism, while a stronger LLM can exploit it more effectively and raise the performance ceiling.

TABLE VI Five-run average PPA results of SynAct with GPT-5.2 under the same setup as TABLE III.
<table><tr><td>Benchmarks</td><td>WNS(ps)</td><td>TNS(ps)</td><td>Area (µm2)</td><td>DP (mW)1</td><td>SP (µW)2</td></tr><tr><td>uart16550</td><td>-1.43</td><td>-1.53</td><td>401.39</td><td>1.584</td><td>0.395</td></tr><tr><td>picorv32</td><td>0.04</td><td>0.00</td><td>1131.68</td><td>3.542</td><td>1.303</td></tr><tr><td>yacc</td><td>-3.28</td><td>-59.29</td><td>1156.95</td><td>4.133</td><td>1.300</td></tr><tr><td>aes_core</td><td>-2.90</td><td>-158.80</td><td>1207.75</td><td>3.747</td><td>1.345</td></tr><tr><td>wb2axip</td><td>-6.46</td><td>-3352.07</td><td>1329.51</td><td>5.163</td><td>1.175</td></tr><tr><td>wb_conmax</td><td>-0.59</td><td>-9.60</td><td>1685.97</td><td>4.363</td><td>1.460</td></tr><tr><td>arm9</td><td>-4.42</td><td>-47.12</td><td>2034.10</td><td>1.817</td><td>2.077</td></tr><tr><td>sha3</td><td>-2.54</td><td>-246.83</td><td>2839.00</td><td>17.220</td><td>2.765</td></tr><tr><td>spimaster</td><td>0.02</td><td>0.00</td><td>5387.82</td><td>11.240</td><td>6.103</td></tr><tr><td>ethernet</td><td>-2.94</td><td>-17.15</td><td>6377.96</td><td>23.530</td><td>7.330</td></tr><tr><td>ecg</td><td>0.01</td><td>0.00</td><td>5846.38</td><td>22.690</td><td>5.722</td></tr><tr><td>linkruncca</td><td>0.00</td><td>0.00</td><td>9815.93</td><td>27.540</td><td>11.440</td></tr><tr><td>xge_mac</td><td>-4.36</td><td>-25.63</td><td>12683.96</td><td>7.540</td><td>14.020</td></tr><tr><td>fft256</td><td>-6.49</td><td>-151.60</td><td>26612.28</td><td>72.650</td><td>31.560</td></tr><tr><td>Ratio Avg.</td><td>20.37%</td><td>13.69%</td><td>99.36%</td><td>99.65%</td><td>99.41%</td></tr></table>

Dynamic Power. <sup>2</sup> Static Power.

TABLE VII Diagnostics of GrammarVAE latent locality and BO-seed utility over 14 benchmarks.
<table><tr><td>Category</td><td>Diagnostic</td><td>Result</td></tr><tr><td>Latent locality</td><td>Mean  $\overline { { \Delta r } } ,$  near pairs Mean  $\Delta r ,$  far pairs</td><td>0.188 0.238</td></tr><tr><td>BO seed utility</td><td>BO-seeded mean reward RAG-grounded mean reward Iterations where BO has highest mean</td><td>0.297 0.289 65.3% 47.2%</td></tr></table>

## E. Validation of Latent-Space BO Guidance

To validate BO-guided experience refinement, we test two properties required by its design. First, if GrammarVAE provides a useful local search prior, commands that are close in latent space should produce more similar synthesis rewards than distant commands. Using $r ( C )$ from Equation (2), we measure this similarity by $\Delta r ( C _ { i } , C _ { j } ) = | r ( C _ { i } ) - r ( C _ { j } ) |$ where a smaller value indicates more consistent optimization outcomes. As shown in TABLE VII, near command pairs have a mean $\Delta r$ of 0.188 versus 0.238 for far pairs, a 21.0% reduction. Second, the BO guidance should remain useful after the LLM converts a decoded seed into tool-valid candidates. BO-seeded candidates achieve a mean reward of 0.297 versus 0.289 for RAG-grounded candidates and obtain the highest mean reward in 65.3% of benchmark iterations. These results support the two design properties: locally consistent latent neighborhoods and BO guidance that remains effective after tool-aware LLM refinement.

## F. Ablation Study on BO and Retrieval Modules

Fig. 8 presents the ablation results using the full SynAct configuration (“BO+GraphRAG”) as the reference. In this configuration, candidate generation combines report diagnosis, documentation retrieved by GraphRAG, and GrammarVAE seeds guided by BO. All variants generate ten candidates per iteration, evaluate them through synthesis, and select the best safe command, thereby isolating the effects of BO and retrieval. The “w/o BO” variant removes BO guidance and the reuse of historical commands, while “vanilla RAG” uses flat retrieval and “w/o RAG” provides the full documentation directly.

![](images/c336dd7000995b06720413f2a363320b6830ad3b56fdd8a6979d6f12a48183ff.jpg)  
Fig. 8 Ablation of SynAct components on WNS. The full configuration (BO + GraphRAG) achieves the best timing improvement.

Without BO, timing performance clearly degrades, with the WNS Ratio Avg. rising from 27.0% to 37.5%, indicating less effective WNS improvement. The full configuration outperforms this variant on 13 of 14 benchmarks, with WNS improving from -7.16 ps to -6.78 ps on wb2axip and from - 8.65 ps to -7.90 ps on fft256. These results show that BOguided experience reuse steers exploration of the command space toward better optimization outcomes.

The “vanilla RAG” variant raises the WNS Ratio Avg. from 27.0% to 28.6%, while the “w/o RAG” variant further increases it to 38.3%. WNS degradations exceeding 2 ps on yacc and linkruncca suggest that GraphRAG’s structured representation captures richer contextual dependencies than flat retrieval or direct use of the full documentation. Overall, SynAct achieves the most effective timing improvement among all variants, with BO supporting effective exploration and GraphRAG providing structured contextual knowledge for optimization.

## VII. CONCLUSION

In summary, SynAct is an adaptive closed-loop LLM framework for iterative PPA optimization in commercial synthesis, addressing the limitations of fixed-action search and oneshot script generation. By combining synthesis feedback, prior experience, and retrieved tool knowledge, SynAct enables state-aware decisions with explicit rationales while improving timing and maintaining balanced area and power. Although evaluated only on AltiSyn<sup>®</sup>and open-source designs, SynAct is designed for portability to other logic synthesis tools with scripting interfaces, command documentation, and structured PPA and timing reports by replacing its tool-specific adapters and knowledge base. Future work will validate crosstool portability and industrial-scale performance and reduce candidate-evaluation overhead.

## ACKNOWLEDGMENT

The authors thank ZeniSyn Design Systems for providing access to AltiSyn<sup>®</sup> and related technical support. Regarding the use of generative AI, DeepSeek V3.1 and GPT-5.2 were used in SynAct to analyze synthesis reports and generate candidate commands, while OpenAI Codex was used solely for language editing and grammar checking. All technical content, ideas, and results are original work by the authors.

## REFERENCES

[1] C. Yu, H. Xiao, and G. De Micheli, “Developing synthesis flows with out human knowledge,” in ACM/IEEE Design Automation Conference (DAC). IEEE, 2018, pp. 1–6.

[2] A. B. Kahng, “New directions for learning-based IC design tools and methodologies,” in 2018 23rd Asia and South pacific design automation conference (ASP-DAC). IEEE, 2018, pp. 405–410.

[3] R. Brayton and A. Mishchenko, “ABC: An academic industrial-strength verification tool,” in International Conference on Computer-Aided Verification (CAV). Springer, 2010, pp. 24–40.

[4] A. Hosny, S. Hashemi, M. Shalan, and S. Reda, “DRiLLS: Deep reinforcement learning for logic synthesis,” in IEEE/ACM Asia and South Pacific Design Automation Conference (ASPDAC). IEEE, 2020, pp. 581–586.

[5] A. Grosnit, C. Malherbe, R. Tutunov, X. Wan, J. Wang, and H. B. Ammar, “BOiLS: Bayesian optimisation for logic synthesis,” in IEEE/ACM Design, Automation and Test in Europe (DATE). IEEE, 2022, pp. 1193–1196.

[6] Z. Pei, F. Liu, Z. He, G. Chen, H. Zheng, K. Zhu, and B. Yu, “AlphaSyn: Logic synthesis optimization with efficient Monte Carlo Tree Search,” in IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2023, pp. 1–9.

[7] C. Yu, “FlowTune: Practical multi-armed bandits in boolean optimization,” in IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2020, pp. 1–9.

[8] F. Liu, Z. Pei, Z. Yu, H. Zheng, Z. He, T. Chen, and B. Yu, “CBTune: Contextual bandit tuning for logic synthesis,” in IEEE/ACM Design, Automation and Test in Europe (DATE). IEEE, 2024, pp. 1–6.

[9] H. Wu, Z. He, X. Zhang, X. Yao, S. Zheng, H. Zheng, and B. Yu, “ChatEDA: A large language model powered autonomous agent for EDA,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), vol. 43, no. 10, pp. 3184–3197, 2024.

[10] M. Liu, T.-D. Ene, R. Kirby, C. Cheng, N. Pinckney, R. Liang, J. Alben, H. Anand, S. Banerjee, I. Bayraktaroglu et al., “ChipNemo: Domain-adapted LLMs for chip design,” 2023. [Online]. Available: https://arxiv.org/abs/2311.00176

[11] H. Zheng, H. Wu, and Z. He, “ChatLS: Multimodal retrieval-augmented generation and chain-of-thought for logic synthesis script customization,” in ACM/IEEE Design Automation Conference (DAC). IEEE, 2025, pp. 1–7.

[12] Y. Pu, Z. He, T. Qiu, H. Wu, and B. Yu, “Customized retrieval augmented generation and benchmarking for EDA tool documentation QA,” in IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2024, pp. 1–9.

[13] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations (ICLR), 2022.

[14] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu et al., “AutoGen: Enabling next-gen LLM applications via multi-agent conversations,” in First Conference on Language Modeling (CoLM), 2024.

[15] Synopsys, Design Compiler User Guide, Mountain View, CA, USA, 2022, available via Synopsys SolvNet.

[16] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel¨ et al., “Retrievalaugmented generation for knowledge-intensive NLP tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.

[17] S. Huang, J. Li, Z. Yu, J. Ye, J. Xu, N. Xu, and G. Dai, “LLSM: LLM-enhanced logic synthesis model with EDA-guided CoT prompting, hybrid embedding and AIG-tailored acceleration,” in IEEE/ACM Asia and South Pacific Design Automation Conference (ASPDAC). IEEE, 2025, pp. 974–980.

[18] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global: A graph RAG approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024.

[19] J. Wu, J. Zhu, Y. Qi, J. Chen, M. Xu, F. Menolascina, Y. Jin, and V. Grau, “Medical graph RAG: Evidence-based medical large language model via graph retrieval-augmented generation,” in Annual Meeting of the Association for Computational Linguistics (ACL), 2025, pp. 28 443– 28 467.

[20] M. J. Kusner, B. Paige, and J. M. Hernandez-Lobato, “Grammar varia-´ tional autoencoder,” in International Conference on Machine Learning (ICML). PMLR, 2017, pp. 1945–1954.

[21] D. Lynch, J. McDermott, and M. O’Neill, “Program synthesis in a continuous space using grammars and variational autoencoders,” in International Conference on Parallel Problem Solving from Nature. Springer, 2020, pp. 33–47.

[22] H. Dai, Y. Tian, B. Dai, S. Skiena, and L. Song, “Syntax-directed variational autoencoder for molecule generation,” in International Conference on Learning Representations (ICLR), 2018.

[23] H.-M. Gutmann, “A radial basis function method for global optimization,” Journal of global optimization, vol. 19, no. 3, pp. 201–227, 2001.

[24] N. Srinivas, A. Krause, S. Kakade, and M. Seeger, “Gaussian process optimization in the bandit setting: No regret and experimental design,” in International Conference on Machine Learning (ICML). PMLR, 2010, pp. 1015–1022.

[25] “ZeniSyn Design Systems,” https://www.zenisyn.com.

[26] L. T. Clark, V. Vashishtha, L. Shifren, A. Gujja, S. Sinha, B. Cline, C. Ramamurthy, and G. Yeric, “ASAP7: A 7-nm FinFET predictive process design kit,” Microelectronics Journal (MEJ), vol. 53, pp. 105– 115, 2016.

[27] “OpenCores,” https://opencores.org.