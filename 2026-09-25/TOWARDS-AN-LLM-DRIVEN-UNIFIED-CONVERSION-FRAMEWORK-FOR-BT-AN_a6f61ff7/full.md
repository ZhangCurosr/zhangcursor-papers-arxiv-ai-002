# TOWARDS AN LLM-DRIVEN UNIFIED CONVERSION FRAMEWORK FOR BT AND FSM IN AUTONOMOUS INTELLIGENT SYSTEMS ∗

Zhang Qi, Yang Shuo, Zhu Zhengqiu, Zhou Peng, Jiao Peng College of Systems Engineering, National University of Defense Technology CHANGSHA, HUNAN, China zhangqiy123@nudt.edu.cn

## ABSTRACT

Finite state machine (FSM) and behavior trees (BT) are widely adopted behavioral modeling paradigms for autonomous intelligent systems. While functionally equivalent and inter-convertible in principle, existing transformation methods between FSM and BT face major challenges in preserving behavioral completeness and avoiding model complexity explosion. To overcome these issues, we propose an LLM-driven unified conversion framework that enables automatic, efficient, and semantically consistent transformation between FSM and BT. Specifically, a novel loop execution BT structure is designed for LLM to accurately capture the loop structure in FSM, thereby preserving behavioral completeness. To mitigate the state explosion problem in BT-to-FSM conversion, a depth compression strategy is introduced with LLM prompt to eliminate redundant control nodes, complemented by differentiated hierarchical conversion rules that collectively reduce the number of required sub-FSM. Simulation experiments in multiple autonomous decision-making scenarios demonstrate that the proposed framework enables an accurate and automated bidirectional conversion between FSM and BT. Furthermore, it significantly enhances the scalability and maintainability of generated models compared to traditional approaches, providing a practical solution for behavior model conversion in consumer-grade autonomous intelligent systems such as service robots, game agents, and smart home devices

Keywords Behavior Tree · Finite State Machine · Behavioral Model Conversion · Large Language Model · Autonomous Intelligent Systems

## 1 Introduction

In consumer autonomous intelligent (AISs) applications such as home and service robots [1], smart home appliances, interactive game agents [2], and smart electric vehicles [3], behavioral modeling is a key component in achieving a closed-loop of autonomous perception, decision-making, and execution. Since these consumer electronics systems are typically deployed in dynamic environments and interact directly with end users, their behavioral logic must be interpretable, maintainable, verifiable, and easily portable across software architectures and hardware platforms. FSM [4-5] are widely used in such systems due to their advantages, including explicit state transitions, low runtime overhead, and verifiable execution paths. These characteristics are particularly important for consumer electronics, as systems typically require predictable behavior, convenient debugging capabilities, and safety-related traceability during product development, testing, and updates. However, their single architecture faces a state-explosion problem in dynamic environments[6], when there are n states, the number of state transitions is O(n²). This compromises explainability and safety compliance due to the inability to audit decision logic.

In contrast, behavior tree (BT) [6-8] demonstrate superior flexibility and maintainability in complex decision-making scenarios through modular sub-tree reuse. This enables independent development, testing, and deployment, high responsiveness in handling task interruptions, as well as hierarchical task decomposition. However, the introduction of unexplained polling latency and opaque sub-tree interactions is a fundamental barrier for applications that require human-in-the-loop transparency. These paradigms exhibit significant complementarity across dimensions, including real-time operation versus flexibility and determinism versus extensibility [9-10].

FSM remain widely used in robotic and gaming systems for their explicit state transitions and ease of verification, while BT are increasingly adopted in autonomous intelligent systems for their modularity and reactivity [6,11]. Therefore, seamless bidirectional FSM-BT conversion is essential for technology migration, hybrid system design, and long-term maintenance, especially in consumer electronics where behavioral logic frequently evolves through software iteration, firmware updates, platform migration, and behavioral logic refactoring. The conversion of FSM-to-BT improves the modularity and reactivity of legacy control modules, whereas the conversion of BT-to-FSM supports formal verification, debugging, runtime monitoring, certification and integration with existing FSM-based infrastructures [12-14]. However, manual conversion is time-consuming and prone to structural inconsistency, semantic deviation, and behavioral misalignment caused by the mismatch between event-driven FSM and poll-based BT. Thus, an automated and semantically consistent conversion framework is needed to reduce engineering costs and improve maintainability.

Conventional conversion methods[15-16] typically rely on fixed templates and mapping rules built on expert knowledge; therefore, they require the input model to adopt a predefined formal representation and satisfy specific structural assumptions. When the input is presented in non-prescribed formats—such as images, text, or structured tables—or employs new modeling conventions and transformation semantics, traditional methods often require additional format conversion and model preprocessing, or even manual extension or rewriting of transformation rules. This reliance on specific input representations and manually defined rules limits the generalizability of input representations and increases the cost of system adaptation. Furthermore, target models generated by traditional methods are prone to structural redundancy, resulting in low modularity and readability, which in turn increases the difficulty of subsequent feature expansion, debugging, and maintenance.

Recent advances in large language models (LLMs) [17-19] open new pathways for the conversion of intelligent behavioral models. With enhanced natural language understanding and logical reasoning capabilities, LLMs can parse control logic, preserve semantic equivalence, and verify behavioral consistency during the FSM–BT transformation. This substantially reduces the development threshold and improves cross-paradigm alignment accuracy. LLMs[20- 22] conversion methods face three primary challenges when applied to complex scenarios: (1) LLMs’ insufficient comprehension of intricate behavioral logic may result in semantic information loss; (2) the structural mapping between FSM and BT remains unresolved, leading to incomplete semantic conversion in certain instances; (3) Inconsistencies in conversion results, as LLMs may generate multiple viable conversion schemes. These challenges render current LLMbased approaches insufficient to meet the practical demands of efficient and reliable conversion between behavioral modeling paradigms.

To overcome these limitations, this paper proposes an LLM-driven unified conversion framework (LLMDUCF) for the bidirectional transformation between FSM and BT. LLMDUCF treats the LLM as a high-level behavioral model conversion engineer, whose reasoning process is constrained by domain-specific conversion rules, structural mappings, and task-oriented prompts. Under these constraints, the LLM parses the source behavioral model, reasons about its execution semantics, and synthesizes a structurally correct target model, thereby enabling automated and semantically consistent model conversion.

In summary, the key contributions of this paper are as follows.

1) We propose LLMDUCF, an LLM-driven unified framework that formulates bidirectional FSM-BT conversion as constrained semantic reasoning, enabling LLMs to perform structural recognition, semantic decomposition, execution logic reasoning, target-model synthesis, and structural correctness under domain-specific conversion constraints.

2) We propose a loop execution behavior tree (LEBT) for behavior-preserving FSM-to-BT conversion. The LEBT represents typical FSM control-flow patterns, including entry, loop, chain, and branch structures; guided by the proposed mapping specifications, the LLM recognizes these structures, reasons about FSM transition semantics, and synthesizes a complete BT that preserves loop behavior and maintains consistent execution logic.

3) We propose a BT depth-compression strategy and a differentiated hierarchical conversion mechanism for scalable BT-to-FSM conversion. These mechanisms guide the LLM to merge redundant homogeneous control nodes, decompose compressed BTs into semantic conversion units, infer the execution logic of sequence and selector nodes, and generate the corresponding FSM structures, thereby reducing BT depth, mitigating sub-FSM explosion, and improving the maintainability of the generated FSM.

The remainder of this article is organized as follows: Section II briefly discusses related studies. Section III presents the proposed bidirectional FSM and BT conversion framework for AISs. The simulation results are presented in Section IV, Section V concludes the article.

## 2 BACKGROUND AND RELATED WORKS

## 2.1 Conventional Model Conversion Approaches

Colledanchise et al. [15] have proposed explicit state-variable mapping by converting each FSM state to a BT node and transitioning to global variable-driven selector/sequence combinations to achieve semantic equivalence. Their state simulation method interprets BT execution states (running/success/failure) as FSM transition triggers, thereby enabling BT-to-FSM conversion through chained state transitions. Marzinotto et al. [16] have established a unified BT framework that maps FSM transitions to condition nodes and actions to action nodes for automated FSM-to-BT conversion. For reverse conversion, they have designed recursive embedding rules to translate BTs sequence, selection, and root nodes into FSM structures. Martín et al. [23] have integrated symbolic planning to map high-level BT structures to FSM planning states to generate executable sequences and reconstruct FSM logic as combinations of BT nodes. Hallen et al. [24] have achieved bidirectional conversion via functional decoupling, where FSMs govern when to act and BTs determine how to act.

Although existing conversion methods show certain effectiveness, several critical limitations must be addressed to meet the demands of autonomous intelligent systems. First, the generated FSM exhibit O(n²) state growth as BT depth increases, which not only undermines model maintainability but can also lead to state explosion and degraded conversion performance. Second, the resulting BT often suffers from limited expressiveness and structural overflattening, reducing readability, and introducing semantic inconsistencies. Third, these methods primarily provide transformation rules or algorithmic concepts and do not directly include complete behavioral model parsing modules or engineered transformation implementation modules.

## 2.2 LLM-based Automated Conversion Approaches

Recent research has explored the generation of LLM-based automated behavioral models. Gan et al. [25] have developed ChatFSM, which is a multiagent framework that reads FSM structures from codebases. They have used retrieval-augmented generation for context and guiding LLMs to modify the FSM code based on natural language instructions. Lin et al. [26] have proposed systematic markdown prompt templates with task-oriented prompt patches and step-by-step instructions for LLMs to focus on key design constraints. This improves the accuracy of FSM generation through chain-of-thought reasoning. Swick et al. [27] have proposed a state machine synthesis method based on LLMs to allow humans to plan robot movements. This method reduces the burden of robot programming while ensuring the safety and human-collaborative nature of conventional motion planning. Zhou et al. [28] have proposed an LLM–BT method that uses the reasoning task description steps of ChatGPT to construct a semantic map using a target recognition algorithm. They have designed a BERT-based parser module to parse the steps into initial BT. Then, the initial BT are dynamically expanded through a BT update algorithm to control the adaptive task execution of a robot. Lykov et al. [29] have proposed LLM-BRAIn, which is an AI-driven method based on an LLM. Existing research has demonstrated the potential of LLMs to generate behavioral models. However, their limited comprehension of complex logical relationships described in natural language often introduces ambiguity, inaccuracy, and instability into the resulting model. To overcome these shortcomings, this article introduces an LLM-driven framework for bidirectional FSM–BT conversion, designed to improve conversion efficiency, structural stability, and behavioral coherence.

## 3 PROPOSED METHODOLOGY

## 3.1 Framework Overview

The proposed framework shown in Fig.1 is initiated using a natural language conversion task to use LLM as a high-level behavioral model conversion engineer specialized in bidirectional FSM–BT transformations. It comprises of three integrated components: the FSM-to-BT conversion component on the left, which performs model conversion through structural mapping; and the BT-to-FSM conversion component on the right, which performs model conversion through differentiated hierarchical conversion rules; at the bottom layer of the framework, a multi-level verification module is designed to ensure the structural correctness and behavioral consistency of the converted model. For models found to contain errors during validation, the LLM can be prompted to regenerate the model by optimizing the prompts (e.g., adding constraints or strengthening structural rules).

![](images/e51d11303bb89e75df802813700361e2f159a6ffb975187a6d91c8e00a89f44f.jpg)  
Figure 1: LLM-based bidirectional conversion framework for FSM and BT behavioural models.

## 3.2 FSM-to-BT Conversion Workflow

This section describes the mapping template and workflow for the FSM-to-BT conversion.

## 3.2.1 Typical Structure Recognition

![](images/43d6422c7626bbc6678a81351bd2d0824230283cd4da2786c20b88c3633d8700.jpg)  
Figure 2: Basic FSM structure.

There are four typical structures in FSMs: entry states, loop states, chain links, and branch links. The entry state is the state connected to the start state, such as state A in Fig. 2. Loop states are those with an in-degree greater than 1 (e.g., C and D in Fig.2). The state s has n $( 0 < n )$ out-degrees. Should the $i ( 1 < i \leq n )$ out-degrees connect to the ordinary state, the link formed between state s and the ordinary states pointed to by these $i ( i = 1 )$ out-degrees constitutes a branch link(A → B/F). Should i out-degree connect to ordinary states, the link formed between state s and the ordinary state pointed to by this single out-degree constitutes a chain link $( \mathrm { E }  \mathrm { G }  \mathrm { H } )$ .

$\mathrm { L E B T } = ( S T _ { e n t r y } , S T _ { b r a n c h } , S T _ { s e q } , S T _ { s e l } )$ is a quadruple. $S T _ { e n t r y } = ( r , S T _ { s e l } ^ { i n i t } , S T _ { s e l } ^ { l o o p } )$ expresses the entry structure, where the root node r is of sequence node type, with its child nodes comprising two sets of selector sub-trees: $S T _ { s e l } ^ { i n i t }$ corresponding to the entry state set and $S T _ { s e l } ^ { l o o p }$ corresponding to the loop state set. This entry structure resides at the top level of the BT, serving to express the entry states and loop re-entry points of the FSM. $\dot { S } T _ { s e l } = ( r , c _ { l } , c _ { r } )$ is the selector subtree, where the root node r serves as the selector node. The left child node c denotes the condition node, which maps the trigger events for state transitions. The right child node $c _ { r } \in \{ A _ { s } , S T _ { b r a n c h } , S T _ { s e q } \}$ represents action nodes, branching structures, and sequential structures, designed to map states, branch links, and chained links within the FSM. $S T _ { b r a n c h } \mathbf { \bar { = } } ( r , A _ { s } , S T _ { s e l } ^ { 1 } , S T _ { s e l } ^ { \hat { 2 } } , . . . , S T _ { s e l } ^ { k } )$ represents the branching structure, where the root node r is a sequence node. A denotes the action node of state s, and each selector subtree in $S T _ { s e l } ^ { 1 } , S T _ { s e l } ^ { 2 } , . . . , S T _ { s e l } ^ { k }$ corresponds to a link in the FSM where state s behaves as an ordinary state, capturing the branching link within the FSM, where k represents the number of links. When $k = 1$ is a sequence structure $\bar { S T } _ { s e q } = ( r , A _ { s } , \bar { S } T _ { s e l } )$ , it corresponds to chained links in the FSM. If a subsequent chain link exists, it is implemented by recursively nesting $S T _ { s e l }$ through the right subtree of $S T _ { s e q } ,$ producing $c _ { r } ( S T _ { s e l } ) = S T _ { s e q } ^ { \prime }$

![](images/de33367c58d08a386b4185732766a7e8e9eaf64582a42ff548c1b7277b5fea90.jpg)  
Figure 3: LEBT structure.

## 3.2.2 FSMs and LEBT Mapping Relationship

Within the conversion framework, the states of the FSM are mapped to action nodes in the LEBT, whilst the events within the FSM are mapped to conditional nodes in the LEBT after being negated. Multiple events corresponding to the same action node are consolidated into a single conditional node. Furthermore, the entry states and loop states within the FSM are mapped to the selection subtrees within the LEBT entry structure. Branch links in the FSM are mapped to the branch structure in LEBT, whilst chained links are mapped to the sequence structure.

## 3.2.3 LEBT Construction Process

First, the root node is directly connected to the entry structure, with the entry state and loop states sequentially added as selected subtrees, while duplicate state structures are removed. Second, each selector subtree’s right child node is processed: If the corresponding state contains branch links, the node is converted into a branch structure. If chain links exist, they are converted into a sequential structure. Finally, the complete sequential structure is constructed recursively based on the order of the chained links.

## 3.2.4 Prompts for FSM-to-BT Conversion

The prompt module for FSM-to-BT conversion is structured into four units: Role, Task, Stepwise Conversion Procedure (SCP), and Constraints. This design systematically defines conversion tasks and clarifies constraints, thereby significantly enhancing the accuracy and stability of LLM outputs, as shown in Fig. 4. Within this framework, the LLM assumes the role of ‘Behavioral Model Expert’, whose main task is to achieve the correct conversion from FSM to BT, ensuring that the results generated exhibit consistent behavioral logic. The SCP first guides the model to recognize typical structures within the $\mathrm { F S M } ;$ subsequently, it learns the characteristic structures of the proposed LEBT; then it completes the structural conversion based on the mapping relationship between the FSM and the LEBT structures; finally, it follows the LEBT construction process to complete the final BT construction. Key constraints include: prohibiting the introduction of any additional logic, strictly adhering to the defined conversion steps, and ensuring that the output is directly a complete BT description without any interpretations or extraneous text beyond the specified content<sup>1</sup>.

## 3.2.5 FSM–LEBT Behavioral Equivalence Proof

We prove behavioral equivalence by constructing a macro-step bisimulation [30] relationship between the pre- and posttransformation models, modeling both FSM and BT as labeled transition systems. If there exist $M = ( S , \bar { \Sigma } , \delta , s _ { 0 } , \bar { \alpha } , F )$ and $B = ( Q _ { B } , \sum , \Delta , q _ { 0 } , \beta , \bar { F _ { B } } )$ such that there is a relation $R \subseteq \bar { S } \times Q _ { B }$ satisfying: 1. Initial preservation: $( s _ { 0 } , q _ { 0 } ) \in R ;$ 2. Preservation of action : $\operatorname { i f } \left( s , q \right) \in R$ , then $\alpha ( s ) = \beta ( q ) ; 3 .$ Transition preservation: if $( s , q ) \in R$ and $\delta ( s , e ) = s ^ { \prime }$ , then there exists $q ^ { \prime } = \Delta ( q , e )$ such that $( s ^ { \prime } , q ^ { \prime } ) \in R ;$ if $( s , q ) \in R$ and $\Delta ( q , e ) = q ^ { \prime }$ , then there exists $s ^ { \prime } = \operatorname { \bar { \delta } } ( s , e )$ such that $( s ^ { \prime } , q ^ { \prime } ) \bar { \in } R$ 4.Termination preservation: if $( s , q ) \in R ,$ , then $s \in F \Leftrightarrow q \in F _ { B }$ , then M and $B$ are said to be macro-step bisimulation equivalents, denoted by $\dot { M } \sim B$ . Where, M denotes an FSM, S is a finite non-empty set of states; $\displaystyle \sum ^ { }$ is a finite non-empty set of trigger events; $\delta : S \times \Sigma $ S is the state transition function; $\delta$ is defined for every $( s , \overline { { e } } ) \in S \times \Sigma$ and produces a unique successor state for each input pair; M is then referred to as a fully-defined deterministic $\mathbf { \mathrm { F S M } } ; s _ { 0 } \in S$ is the initial state; $\alpha : S  A c t$ is the state-action labeling function, denoting the observable actions executed when the system is in state s; $F \subseteq S$ is the set of terminal states. $Q _ { B }$ is a finite set of configurations; a running configuration of an LEBT is defined as $q _ { B } = ( n , \rho )$ , where $n \in N$ denotes the current position in the specification structure corresponding to the observable execution result; $\rho$ denotes the finite execution context required for a complete tick. In the LEBT specification discussed in this paper, as no parallel nodes, decorator nodes, or complex cross-tick local memory are introduced, all reachable running configurations form a finite set, denoted by $Q _ { B } ; \sum$ is the set of events; $\Delta : Q _ { B } \mathrm { { ' } } \times \Sigma  Q _ { B }$ is the macro-step transition function; $q _ { 0 } \in Q _ { B }$ is the initial configuration; $\overline { { \beta } } : Q _ { B } $ Act is the observable action labeling function in the configuration; $F _ { B } \subseteq Q _ { B }$ is the set of terminal configurations.

For any $s \in S$ , define $\phi ( s ) \in Q _ { B }$ as the stable configuration reached after the completion of a full LEBT tick. Specifically, $\phi ( s )$ denotes the stable macro-step configuration corresponding to state $s ,$ in which the action node $A _ { s }$ is activated. The initial state of the FSM is $s _ { 0 }$ . Since the direct child of the LEBT root node is unique and corresponds to the entry structure node, and since the entry state $s _ { 0 }$ is encoded as the initial entry branch in the entry structure, the first complete tick executed from the root node leads the system to the stable configuration corresponding to $s _ { 0 } .$ . Denote this initial stable configuration by $q _ { 0 }$ . By the definition of $\phi ,$ we have $\phi ( s _ { 0 } ) = q _ { 0 }$ , which establishes the initial-state condition.

![](images/13304bbaaa71e55e6b0ff692e3aaa923266b8cbce42d041f48fe4dcb830c96e2.jpg)  
Figure 4: Prompts for FSM-to-BT Conversion.

Let $s \in S$ be arbitrary. By the LLMDUCF construction rule, the FSM state s is uniquely mapped to the action node $A _ { s }$ whose observable action label is exactly α(s). By the definition of $\phi , \phi ( s )$ is the stable configuration in which A is currently active. Since $\beta : Q _ { B } $ Act assigns observable actions to LEBT configurations, the observable action of ϕ(s) is precisely the label of $A _ { s }$ . Hence,

$$
\beta ( \phi ( s ) ) = \alpha ( s ) ,
$$

which establishes the action-preservation condition.

Let $s \in S$ and $e \in \Sigma$ be arbitrary, and suppose that $s ^ { \prime } = \delta ( s , e )$ is defined. Since M is a fully defined and deterministic FSM, the successor state $s ^ { \prime }$ is uniquely determined. By the definition of $\phi , \phi ( s )$ denotes the stable configuration in which the action node corresponding to the currently active state is $A _ { s }$ . Therefore, during the next complete tick starting from $\phi ( s )$ , only the successor structure associated with $A _ { s }$ can be expanded. According to the LLMDUCF construction rule, for each state–event pair $( s , e )$ , there exists a unique corresponding conversion unit. The success condition of this unit is precisely that the current state is s and the input event is $e ,$ and its success result is the activation of the action node $A _ { s ^ { \prime } }$ corresponding to the target state $s ^ { \prime } = \delta ( s , \bar { e } )$ . For chained structures, the $S T _ { s e q }$ semantics ensures that, after a successful condition check, the execution proceeds to $A _ { s ^ { \prime } }$ in a fixed order. For branched structures, the $S T _ { b r a n c h }$ semantics ensures that, among all candidate branches, exactly the branch corresponding to $( s , e )$ succeeds and activates $A _ { s ^ { \prime } }$ . For loop cases, the $S T _ { b r a n c h }$ structure ensures that the valid re-entry target is uniquely $A _ { s ^ { \prime } }$ . Furthermore, since the mapping from state–event pairs to conversion units is unique, no two distinct conversion units can simultaneously correspond to the same state–event pair and lead to different target states. Therefore, a complete tick starting from $\phi ( s )$ under the input event e must, and can only, terminate in the stable configuration corresponding to $\phi ( s ^ { \prime } )$ . That is,

$$
\Delta ( \phi ( s ) , e ) = \phi ( s ^ { \prime } ) = \phi ( \delta ( s , e ) ) ,
$$

which establishes the transition-preservation condition.

The set of final states $F$ of the FSM is mapped to the set of final configurations $F _ { B }$ under the LEBT macro-step semantics. Specifically, if $s \in F$ , then the corresponding action node $A _ { s }$ either does not derive any further valid successor behavior in the construction, or its corresponding stable configuration is explicitly designated as a terminating configuration. Conversely, if $s \notin F$ , then there exists a valid successor structure triggered by some event in the construction, and hence the stable configuration corresponding to s is not a terminating configuration. Therefore, we have

$$
s \in F \Leftrightarrow \phi ( s ) \in F _ { B } ,
$$

which establishes the final-state preservation condition.

In summary, the mapping ϕ preserves the initial state, observable actions, transition behavior, and final states between the FSM and the LEBT macro-step semantics. Hence, the FSM and the LEBT are bisimilar at the macro-step level. Consequently, the LEBT constructed by LLMDUCF is behaviorally equivalent to the original FSM. The FSMs supported by the framework are deterministic finite transition systems with a finite set of states, where each state–input pair leads to a unique successor state.

## 3.3 BT-to-FSM Conversion Workflow

This section presents the workflow for the BT-to-FSM conversion.

## 3.3.1 BT Depth Compression strategy

If the main tree and sub-trees of a BT share the same type of control node, the sub-tree control nodes can be removed, and the sub-tree structure can be promoted to the main tree, as shown in Figs. 5 and 6. The deep compression strategy not only eliminate redundant BT control structures to simplify tree structures but also transform BT into alternating sequential and selection node control structures to accommodate differentiated hierarchical conversion rules.

![](images/4cdc2130d49676453c8ed9ed76d8cfad7b30ddd8f7d1e0f77f601379166d5133.jpg)  
(a) Sequence BT before simplification (b) Sequence BT after simplification  
Figure 5: Simplified flowchart for sequence BT.

## 3.3.2 Differentiated Hierarchical Conversion Rules

The differentiated hierarchical conversion rules first divide the BT into conversion units every three layers. Following deep compression processing, each conversion unit retains structural consistency. Subsequently, every conversion unit is transformed into a sub-FSM according to the respective execution logic of sequence nodes and selector nodes. Finally, all generated sub-FSM are integrated into a complete FSM. The conversion results for sequence BT and selector BT are shown in Figs 7 and 8.

## 3.3.3 FSM Construction Process

First, map the leaf nodes of the BT to states, and map the transition units to child FSM; Second, convert all transition units into their corresponding child FSM according to the established transition rules; Finally, integrate the child FSM generated from all transition units to form a complete FSM.

![](images/a3a52bdd05fadd80283b8ee302a776beb8516073afd1420cf9eab0cb2c345697.jpg)  
(a) Select BT before simplification

![](images/1b2a9b5333f4581d5cb9b7a2fd25ab18713f427b1367c22963dfea9639220e1b.jpg)  
(b) Select BT after simplification  
Figure 6: Simplified flowchart for selector BT.

![](images/b61cd9cd9f5369be4a0f6f6e5e17d709c77986c673307b311e441a6e47491045.jpg)  
(a) Sequence BT

![](images/0ffc4d6edc4f51a343f66008c35bcf6542d27dd8da39dd226ca364df0a2f60fa.jpg)  
(b) FSMs corresponding to sequence BT  
Figure 7: Structure Diagram of Sequence BT Conversion to FSM.

## 3.3.4 Prompts for BT-to-FSM Conversion

The prompt for converting BT to FSM is illustrated in Fig. 9. Here, the LLM assumes the role of a ‘Behavioral Model Expert’, with its main task being to achieve the correct conversion from BT to FSM, ensuring that the results generated exhibit consistent behavioral logic. The SCP first guides the model to compress the depth of BT and decompose the BT into multiple conversion units; second, it learns the conversion rules proposed in this paper; finally, it completes the FSM construction by adhering to the FSM construction process. Crucial constraints include prohibiting the introduction of any additional logic; strictly adhering to the defined conversion steps; and ensuring that the output is a complete FSM description without any explanatory text or extraneous content beyond the specified requirements.

## 3.3.5 BT-FSM Behavioral Equivalence Proof

This paper implements a direct conversion from BT to FSM based on execution semantics, and demonstrates from a linguistic-semantic perspective that the execution semantics of this FSM are trajectory-equivalent to those of the original BT. For nested sequence structures of type $S e q ( a , S e q ( b ) , c ) \equiv S e q ( a , b , c )$ , both visit child nodes in exactly the same left-to-right order and halt upon the first unsuccessful execution; consequently, the leaf node visit sequences, atomic observation sequences, return values, and memory updates are consistent. If execution is in progress, the control remains at the same leaf node. Similarly, the same holds for $S e l ( a , S e l ( b ) , c ) \equiv S e l ( a , b , c )$ ; therefore, the semantic integrity of the depth-first compression method is preserved. The sequential conversion unit visits child nodes in sequence; if a child node is a leaf node and its execution succeeds, it transitions to the next node, continuing until the last child node executes successfully, at which point the entire conversion unit is deemed to have executed successfully.

![](images/4a786a13de8229bc83e59b7a49ac3bacefa27ea5e719eca68890f1fbfb7bd2a2.jpg)

![](images/6c108d355c3f2da812f46718fad206995be0959c14c4b4547b967c5f85645aa2.jpg)  
(b) FSMs corresponding to selector BT  
Figure 8: Structure Diagram of Selector BT Conversion to FSM.

If a leaf node fails, the conversion unit is deemed to have failed; if execution is in progress, it resumes from the current node on the next iteration. If the child node is a selection sub-tree, the child nodes are visited sequentially. If a child node is a leaf node or the execution of a child transformation unit fails, the process moves to the next node in the selection sub-tree, until the last child node fails, at which point the entire transformation unit is deemed to have failed. If a leaf node in the selection sub-tree executes successfully, the process moves to the next leaf node in the transformation unit; this is consistent with the logic of the generated FSM. By the same way, it follows that the selector conversion unit is equivalent to the FSM generated by the transition; therefore, BT is equivalent to FSM. The BT supported by the framework are normalized behavior trees that contain only selector nodes, sequence nodes, and leaf nodes.

## 3.4 Verification Module

LLMDUCF employs a two-stage verification mechanism to ensure the reliability of the generated models. First, an LLM-based verifier checks the syntax and structure of the generated BT/FSM against predefined grammar rules and structural constraints, identifying errors such as dangling states or nodes, duplicated structures, missing nodes, invalid branches, and incomplete coverage. Second, NuSMV<sup>2</sup> is used to verify the behavioral consistency between the source and generated models. The original FSM/BT and the converted BT/FSM are abstracted into NuSMV-compatible finite-state transition systems, and formal properties are constructed to check whether the converted model preserves the execution behavior of the source model. When inconsistencies are detected, the feedback is returned to the LLM to guide regeneration through prompt refinement and additional structural constraints, thereby improving the correctness and stability of bidirectional FSM-BT conversion.

## 4 EXPERIMENTS AND VALIDATION

To quantitatively evaluate the performance and generalizability of our proposed framework, we conducted experiments in three distinct hypothetical scenarios, all scenarios using the same text input. This multi-scenario evaluation includes a basic symbolic scenario for foundational verification, a robotic manipulation scenario for physical interaction tasks, and a game AI scenario for dynamic decision-making . The performance of our framework is systematically compared against a normal LLM (NLLM) without SCP and traditional translation approaches.

## 4.1 Experimental Settings

## 4.1.1 Experimental Dataset

We constructed a behavioral-model dataset spanning multiple application scenarios. The dataset combines models randomly generated under formally defined syntactic constraints with behavioral models collected from open-source projects. Covers diverse node-naming conventions, nesting depths, branching and looping patterns, model sizes, and domain-specific semantic descriptions, enabling a representative and rigorous evaluation of the proposed framework across different behavioral-model structures and application contexts. The dataset contains only the source models to be converted; the generated target models are not included in the dataset and are evaluated separately using NuSMV to verify their behavioral consistency with the corresponding source models.

![](images/fe051eeb7d058817eb07cd06771b4137e407581f1eff817051f241c6ea47fe2a.jpg)  
Figure 9: Prompts for BT-to-FSM Conversion.

Scenario 1: The basic symbol scenario verifies the fundamental correctness and completeness of the formal core structures (such as sequence, selector, and loop) during the mutual conversion between FSM and BT. Scenario 2: The robotic grasping scenario tests the framework’s performance under typical physical interactions and embodied decision-making tasks. Scenario 3: The game AI scenario evaluates the framework’s generalizability and practicality within autonomous systems requiring high dynamics, real-time responses, and non-deterministic decision-making. The experimental scenarios described above are closely related to AISs. Robot grasping scenarios represent object selection, navigation, grasping, and placement tasks performed by service robots, home robots, and smart logistics devices in human-centered environments. Game AI scenarios represent interactive entertainment systems, in which non-player characters must possess the ability to make dynamic decisions, track targets, patrol, evade, and adaptively switch behaviors. These two scenarios cover embodied consumer electronics devices and software-based consumer entertainment systems, respectively, and reflect typical modeling and transformation requirements for autonomous behavior in the consumer electronics field.

Each scenario includes 200 BT and 200 FSM. The dataset<sup>1</sup> contains approximately 350 behavioral models from opensource projects. For example, the behavioral model shown in Fig. 14 is a model from the open-source robotics library<sup>3</sup>. The model implements the core task logic for autonomous cargo selection, path planning, target cargo grasping, and cargo placement at fixed locations in industrial logistics handling scenarios. The research covers boundary conditions, naming conventions, and structural constraints commonly found in actual robotic systems. For game AI scenarios, as shown in Figure 20, the NPC control code libraries are derived from open-source game engines<sup>4</sup>. These models implement the dynamic decision-making logic of predator-prey game agents, including patrol and exploration, target detection and tracking, prey capture, foraging, and danger avoidance/escape. The behavioral models for the remainder of the dataset were generated randomly. BT are composed of basic node types, with depth and breadth constrained to a range of 2 to 10. FSM incorporate chain, branch, and loop structures, with a number of states ranging from 2 to 30. According to expert studies, BT structures deeper than 8 layers lead to a sharp increase in comprehension difficulty, while FSM with more than 25 states become significantly harder to understand, test, and extend.

## 4.1.2 Experimental Environment

Scenario 1 employs manual inspection to verify the accuracy of the conversion. In Scenario 2, some models were simulated using the Robot Operating System (Humble ROS 2) and Gazebo 11 platform. In Scenario 3, some behavioral models were simulated using the open-source Predator-Prey Game platform. NuSMV (2.6.0-win64) is a symbolic model checker designed for formal verification of finite-state systems. It is used here to verify the converted FSM and BT models, ensuring that they satisfy behavioral consistency properties such as reachability, conversion completeness, deadlock freedom, and termination consistency. The experimental hardware configuration included an NVIDIA RTX 3060 graphics card and 16GB of memory.

ROS is an open-source robotics software framework that provides hardware abstraction, device drivers, messaging, etc., and supports modular development. Gazebo is a stand-alone 3D robotics simulation environment that focuses on high-fidelity physics and sensor simulations. The robot grasping task is implemented using the Gazebo simulator, as shown in Fig. 10.

![](images/907f1968486de672cb10f893fa34744d2480a0d2b9fa17964558e92f5babc766.jpg)  
Figure 10: Gazebo simulator.

A predator–prey game simulation platform based on C++ is used to simulate the predator-prey process, in which the behaviors of the predator and the prey are controlled by AI. The platform provides an AI modeling interface, FSM and

BT behavioral modeling, and a large number of basic actions such as roaming, tracking, capturing, and maneuvering.   
The pursuer capture task is implemented on this platform, as shown in Fig. 11.

Predator-Prey Game

![](images/e04158095bc97d1ca82c7a851c64076df9854b5e7da70f99312e8f323fda09c7.jpg)

![](images/537397a54d3000e013ec38ffa463dd3fe0bab03a15e10ee83cbe5c9a1ce87dcf.jpg)  
Figure 11: Predator-Prey Game Simulation Platform.

## 4.1.3 Experimental Setup for LLM

In this study, we conducted experiments using three different LLMs: ChatGPT (gpt-4o), DeepSeek (deepseek-reasoner) and Qwen (qwen3-max-thinking). The reason for this selection is as follows: 1. All three are reasoning-capable large models with native support for Chain-of-Thought logical reasoning, making them well-suited to the structured transformation requirements of behavioral models; 2. They encompass both internationally dominant closed-source models and domestically developed open-source models, validating the framework’s cross-model universality; 3. All provide stable API interfaces, supporting reproducible experiments with fixed hyperparameter, and represent the main choices for current behavioral modeling research in the field. All LLMs were accessed via the OpenAI API (version 1.30.0+), with the following key parameters fixed to ensure reproducibility: the temperature was set to 0.1 and the random seed was fixed; the Top-p to 0.9 to enhance output stability; the max retries was set to 3, and the request timeout was set to 180 seconds. The error output verified by the validation module may be reprocessed with optimized prompts.

## 4.2 Evaluation Metrics

Evaluating the behavioral model transformation framework from five perspectives: Conversion Accuracy, Path Coverage, Modularity, Readability, and Input Representation Generalization Ability.

## 4.2.1 Conversion Accuracy

Conversion accuracy is used to evaluate the reliability of the framework in performing behavioral model conversion tasks repeatedly. The conversion accuracy is defined as:

$$
A c c u r a c y = \frac { N _ { f i r s t - p a s s - s u c c e s s } } { N _ { m o d e l } }
$$

where $N _ { m o d e l }$ represents the total number of test cases; and $ { N _ { f i r s t - p a s s - s u c c e s s } }$ denotes the number of test cases for which the initially generated model passes both structural validation and NuSMV-based formal verification, without any retry or regeneration.

## 4.2.2 Behavioral Consistency

Behavioral consistency evaluates whether the converted model preserves the execution semantics of the source model. In this work, formal verification is used to check whether the converted model satisfies the following properties: all critical states, action nodes, and behavioral phases in the source model are reachable in the converted model; each valid transition relation or behavioral path in the source model has a corresponding execution path in the converted model; no additional critical behavior sequence absent from the source model is introduced; no deadlock exists in any non-terminal state; and, under the same input events or condition sequences, the source and converted models reach consistent success, failure, or terminal states. A converted model is regarded as behaviorally consistent if all these properties are satisfied.

## 4.2.3 Modularity

This paper compares the addition, deletion, and modification operations of behaviors within decision structures, focusing on their computational complexity and edit distance. Computational complexity refers to the computational resources and the number of steps required to perform addition, deletion, or modification operations on nodes or states within a BT or FSM. The edit distance denotes the minimum number of edit operations required to transform one structure into another target structure.

## 4.2.4 Readability

Comparing the scalability and maintainability of the behavior model from a readability perspective, treat behavior actions (represented as nodes in BT and states in FSM) as variables to calculate the number of graphical elements within the behavior model structure.Graphical elements constitute all visual components within the behavioral model, including nodes (or states) and edges (or events) that can be intuitively represented. Active elements denote core components within the behavioral model that are directly manipulable (add, delete, modify), serving as the fundamental elements that influence the behavioral model.

## 4.2.5 Input Representation generalizability

The ability of a method to accurately understand the semantic meaning of a behavior and correctly perform model transformation when the same behavioral model is presented in different input formats or representations. This ability is reflected mainly in the method’s adaptability to different input definitions, input formats, and transformation conventions.

## 4.3 Baselines

The experimental setup for NLLM is identical to that of LLMDUCF, employing prompts without SCP. Traditional behavior model conversion methods, including bidirectional structural mapping (BSD) [15] and CHDS–BT [16] method, are also selected to compare with LLMDUCF.

## 4.4 Performance of BT-to-FSM Conversion

Comparison of Conversion Accuracy Among Different LLM  
![](images/de80616ff3c7661b606ee0da65d50a4289447a1e7ea7a43b8496d9046ca927ff.jpg)  
Figure 12: Comparison of Conversion Accuracy Across Scenario.

In all evaluation scenarios, traditional conversion methods BSD and CHDS-BT achieved conversion accuracy of 100%, due to their reliance on fixed conversion rules, as shown in Fig. 12. The proposed LLMDUCF, when implemented with ChatGPT, significantly outperformed the NLLM in all three scenarios (Scenario $1 \colon 9 5 . 7 0 \% \scriptstyle \pm 4 . 2 1$ , Scenario 2: $9 6 . 9 5 \% _ { \pm 3 . 6 1 }$ , Scenario $3 \colon \dot { 9 3 } . 6 5 \dot { \% } _ { \pm 4 . 6 3 } )$ , where the confidence interval is 95%. This performance improvement stems primarily from two key design elements: first, the depth of the behavior tree is reduced through deep compression strategy, forming an alternating control structure; second, the proposed differential hierarchical transformation rules partition complex behavior trees into multiple transformation units for separate processing. This approach enhances conversion accuracy while effectively reducing the number of generated sub-FSM.

![](images/ad2b99da234c72830bb6b68fa60534f2563d889c7dd45ef7f0dd6e4d98a3905b.jpg)  
Figure 13: Comparison of Conversion Accuracy Among Different LLM.

Table 1: Conversion Accuracy with 95% confidence intervals
<table><tr><td></td><td>scenario1</td><td>scenario2</td><td>scenario3</td></tr><tr><td>ChatGPT</td><td> $9 5 . 7 0 { \scriptstyle \pm 4 . 2 1 }$ </td><td> $9 6 . 9 5 { \scriptstyle \pm 3 . 6 1 }$ </td><td> $9 3 . 6 5 { \scriptstyle \pm 4 . 6 3 }$ </td></tr><tr><td>DeepSeek</td><td> $9 6 . 3 5 _ { \pm 3 . 0 0 }$ </td><td> $9 7 . 8 0 _ { \pm 2 . 3 6 }$ </td><td> $9 0 . 6 5 _ { \pm 5 . 5 8 }$ </td></tr><tr><td>Qwen</td><td> $9 4 . 1 0 { \scriptstyle \pm 4 . 2 6 } $ </td><td> $9 4 . 7 5 { \scriptstyle \pm 4 . 9 9 }$ </td><td> $9 3 . 5 5 { \scriptstyle \pm 4 . 5 9 }$ </td></tr><tr><td>ChatGPT (without SCP)</td><td> $3 2 . 1 0 { \scriptstyle \pm 9 . 2 7 }$ </td><td> $2 8 . 7 0 { \scriptstyle \pm 8 . 1 6 }$ </td><td> $3 1 . 9 0 { \scriptstyle \pm 9 . 3 8 }$ </td></tr><tr><td>DeepSeek (without SCP)</td><td> $3 2 . 8 0 _ { \pm 9 . 1 0 }$ </td><td> $3 1 . 5 0 _ { \pm 8 . 1 4 }$ </td><td> $3 1 . 0 0 _ { \pm 8 . 0 0 }$ </td></tr><tr><td>Qwen (without SCP)</td><td> $3 2 . 6 0 { \scriptstyle \pm 9 . 3 4 }$ </td><td> $3 2 . 4 5 _ { \pm 8 . 1 4 }$ </td><td> $3 1 . 0 5 { \scriptstyle \pm 8 . 2 4 }$ </td></tr></table>

In simulation experiments, we evaluated the performance of different LLMs in converting BT to FSM, with the results shown in Fig. 13. In all test scenarios, LLMs employing SCP prompts achieved significantly higher conversion accuracy than baseline models without this strategy, maintaining accuracy consistently between 90.65% and 97.80%. This outcome demonstrates that SCP prompts, specifically designed for model conversion tasks, effectively guide LLMs in comprehending the semantic structure of BT, thereby enhancing conversion precision. Specifically, the main LLM that employs SCP exhibited robust performance in all scenarios, consistently achieving accuracy rates above 90% with minimal fluctuation. This confirms the strong versatility and compatibility of the model with the prompt framework. Moreover, the SCP strategy achieved high-precision, low-variance conversion results in three distinct test scenarios, further demonstrating its cross-scenario robustness in performance enhancement. In summary, the LLMDUCF framework achieves high-precision and stable conversion outcomes in various LLM and task scenarios by integrating SCP prompts, demonstrating excellent adaptability to mainstream models. SCP prompts successfully bridge the semantic gap between general-purpose LLM and structured formal model conversion tasks.

Table 2: Ablation Experiments
<table><tr><td></td><td>scenario1</td><td>scenario2</td><td>scenario3</td></tr><tr><td>Base</td><td> $3 2 . 1 0 { \scriptstyle \pm 9 . 2 7 }$ </td><td> $2 8 . 7 0 { \scriptstyle \pm 8 . 1 6 }$ </td><td> $3 1 . 9 0 { \scriptstyle \pm 9 . 3 8 }$ </td></tr><tr><td>Base+M1</td><td> $3 7 . 0 5 { \scriptstyle \pm 9 . 0 2 }$ </td><td> $3 2 . 5 5 { \scriptstyle \pm 8 . 9 9 }$ </td><td> $3 3 . 8 5 { \scriptstyle \pm 9 . 1 4 }$ </td></tr><tr><td>Base+M2</td><td> $8 3 . 4 0 { \scriptstyle \pm 7 . 0 8 }$ </td><td> $8 6 . 8 5 _ { \pm 6 . 8 9 }$ </td><td> $8 4 . 5 0 { \scriptstyle \pm 7 . 3 5 }$ </td></tr><tr><td> $\mathbf { B a s e + M 1 + M 2 }$ </td><td> $9 5 . 7 0 { \scriptstyle \pm 4 . 2 1 }$ </td><td> $9 6 . 9 5 { \scriptstyle \pm 3 . 6 1 }$ </td><td> $9 3 . 6 5 { \scriptstyle \pm 4 . 6 3 }$ </td></tr></table>

To elucidate the independent and synergistic effects of each module within the SCP framework of the LLMDUCF method, we conducted systematic ablation experiments. The conversion accuracy results under each module configuration are summarized in Table II. M1 comprises the deep compression strategy, while M2 involves differentiated hierarchical transformation rules. The experiments demonstrate that enabling the deep compression algorithm alone yields limited performance gains. This module primarily assists large language models in achieving correct conversions by simplifying behavior tree structures with depths of 3 to 4 layers; however, its capability to handle more complex nested structures remains constrained. Enabling differentiated hierarchical conversion rules alone yielded more pronounced performance gains. Its core mechanism involves decomposing the complex BT into multiple simple conversion units for separate processing. However, this rule is primarily suited to tree structures alternately controlled by sequential and selector nodes. The differentiated hierarchical conversion rules constitute the core module for enhancing overall conversion performance. The depth compression algorithm not only generates independent performance gains, but also synergizes with these conversion rules to further optimize the final conversion results.

To evaluate the conversion quality of LLMDUCF, we conducted simulation experiments under Scenario 2 to compare the performance of different conversion methods. The initial behavior tree employed in the experiments reuses the FSM behavior model of the ROS[8] platform, as shown in Fig. 14.

Table 3: BT to FSM Behavioral Consistency
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD</td><td>CHDS-BT</td></tr><tr><td>Reachability</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Completeness</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>No Additional Behavior</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Deadlock Absence</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Termination Consistency</td><td>√</td><td>X</td><td>√</td><td>√</td></tr></table>

![](images/b3c1e110d03d7252d2511da757823b895a9e7527906042d5397f84e4df16eda5.jpg)  
Figure 14: BT behavioral model of robot.

Behavioral Consistency: Figs. 15–17 show the FSMs generated from BT in Fig. 14 by LLMDUCF, BSD/CHDS-BT, and NLLM, respectively. NuSMV was used to verify reachability, completeness, absence of additional behavior, deadlock freedom, and termination consistency. As shown in Table III, LLMDUCF, BSD, and CHDS-BT satisfy all criteria, indicating behavioral equivalence with the original BT. This was further confirmed in Gazebo, where the original BT and converted FSM produced identical robot trajectories. In contrast, NLLM fails in completeness, absence of additional behavior, and termination consistency, showing that it misses expected paths, introduces redundant transitions, and cannot reliably preserve termination behavior due to insufficient reasoning over deeply nested BT semantics.

Modularity: When adding or removing a state in a standard FSM generated by the NLLM method, consistency between state outputs and transitions must be verified, potentially requiring processing of every state transition with O(n) complexity. For the Proposed and converted BSD/CHDS–BT approaches to HFSMs, adding a state incurs the same complexity as in BT. Once the insertion point is determined, the operation is performed only within the current sub-FSM, achieving O(1) time complexity.

![](images/7c2f5d64938893c83b2145b1cfd156f7897f6ed8c885ee2458c404faed3d2a9b.jpg)  
Figure 15: FSM generated by proposed method.

Table 4: Comparison of Structural Complexity
<table><tr><td colspan="2">LLMDUCF</td><td>NLLM</td><td>BSD/CHDS-BT</td></tr><tr><td>Computational Complexity</td><td>O(1)</td><td>O(n)</td><td>O(1)</td></tr><tr><td>Graphical Elements</td><td>4M+2k+1</td><td>1</td><td>≈ 36M</td></tr><tr><td>Active Elements</td><td>4M+2k+1</td><td>1</td><td>≈ 29M</td></tr></table>

The model scalability of the FSMs was validated through edit distance analysis. When adding Safe-Move-To behavior, Tuck Arm subtree, and Dock subtree behaviors to the FSM, both the Proposed and Baseline methods yielded an edit distance of 4, as illustrated in the structural diagram below. Adding states to the FSM only requires splitting two existing states and connecting the new state. In contrast, the edit distances for the FSMs generated by BSD/CHDS–BT were 4, 12, and 17, respectively. An action node is a subgraph with 1 vertex and 3 edges. The vertex being the node itself and the edges being the 3 transitions, one from each return status to the next child or the parent return statuses. A condition node is similar to an action node, with one edge less because conditions do not return RUNNING. A control node contributes with 1 vertex and 4 edges. The additional edge is the transition from the node to the first child. LLMDUCF exhibits shorter edit distances, greater model modularity, and greater scalability.

Readability: In the FSM generated by the LLMDUCF conversion, each state has two transition paths leading to subsequent states: a successful transition and a failed terminal state. To achieve full reactivity, each state must possess one active self-transition path (i.e., a transition where the state loops back to itself). Under this design, if a finite state machine contains M states, its total number of nodes is M. The transition paths in the FSM then comprise the following types: 1. M self-transitions; 2. M success transitions; 3. M failure transitions; 4. 2k start/end states in sub-FSM; 5. 1 final state. Thus, the total number of graphical elements in the FSM is 4M+2k+1.

In the FSM generated by the BSD/CHDS–BT conversion, each action corresponds to one conditional node and one back-to-start node, while the number of sequence nodes is approximately half that of the action nodes. This implies that for M actions, there will be M conditional nodes, M back-to-start nodes, and 0.5M sequence nodes. Each action node contributes 10 elements, each conditional node contributes 7 elements, and each control node contributes 8 elements, resulting in approximately 29M active elements. For calculating the number of graphical elements, two additional elements must be added per node to account for entering the IN state and a corresponding transition path. The total number of graphical elements is 36M. The FSM generated by the LLMDUCF conversion offers improved readability.

![](images/f96cce53579cdeb5cb697122d89e241575e73ed2aee78c5474f53dca92883f3c.jpg)  
Figure 16: FSM generated by BSD/CHDS-BT.

In addition, we invited 10 behavioral modeling engineers to evaluate converted behavioral models for cognitive load (scored 1–5, where lower scores indicate reduced cognitive load) and model comprehension time. The experimental results revealed that the LLMDUCF-generated model achieved an average cognitive load score of $1 . 8 { \scriptstyle \pm 0 . 3 } $ , compared to $4 . 3 { \scriptstyle \pm 0 . 6 }$ for traditional methods. The understanding time was reduced by 42% compared to conventional approaches, validating its superior readability.

## 4.5 Performance of FSM-to-BT Conversion

In all evaluation scenarios, traditional conversion methods BSD and CHDS-BT achieved conversion accuracy 100%, attributable to their deterministic mapping mechanisms based on fixed templates, as shown in Fig. 18. The LLMDUCF, when implemented with ChatGPT, achieved significantly higher accuracy than the NLLM in all three scenarios (Scenario $1 \colon 9 6 . 9 5 \bar { \% } _ { \pm 3 . 2 2 }$ , Scenario $2 \colon 9 8 . 3 5 \% _ { \pm 1 . 7 7 } .$ , Scenario $3 \colon 9 5 . { \bar { 5 } } 0 \% _ { \pm 4 . 3 7 } )$ . This performance improvement stems primarily from two key design elements: firstly, we rigorously formalized the typical structure of finite state machines, enabling large language models to clearly and consistently recognize various structural types within input models; secondly, our proposed LEBT templates possess comprehensive expressive power, precisely capturing and converting diverse structures within finite state machines, thereby preserving semantic correctness throughout the conversion process.

![](images/ad751fbfafd113f0e09edba532943bd1aa40943337b8a9f562a9fae7b4ad38df.jpg)  
Figure 17: FSM generated by NLLM.

Table 5: Edit Distance of FSM
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD/CHDS-BT</td></tr><tr><td>Tuck Arm subtree</td><td>4</td><td>4</td><td>12</td></tr><tr><td>Safe-Move- To behavior</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Dock subtree</td><td>4</td><td>4</td><td>17</td></tr></table>

The accuracy rates achieved by different LLMs in converting FSM to BT across various scenarios are shown in Fig. 19. Across all scenarios, LLM employing SCP prompts consistently outperformed baseline models without SCP. This confirms that SCP prompts specifically designed for conversion tasks effectively guide LLM in understanding semantics and enhancing conversion accuracy. Mainstream LLMs employing SCP demonstrate consistently high performance, with accuracy rates persistently exceeding 94% and minimal fluctuation. This indicates that the prompt framework possesses strong generalizability and maintains stable output quality when adapted to different models.

To elucidate the independent and synergistic effects of each model within the SCP framework of the LLMDUCF method, we conducted systematic ablation experiments. The conversion accuracy results are summarized in Table VII. M1 involved the recognition of the FSM structure, while M2 addressed the definition and mapping relationships of LEBT. The experiments demonstrate that adding only FSM structure recognition yields a limited performance improvement , demonstrating that pure structure recognition alone is insufficient to bridge the semantic conversion gap. Adding LEBT structure mapping alone produces a slightly greater improvement, yet overall accuracy remains below 50%, reflecting incomplete input comprehension. When both components are integrated, the conversion accuracy in all scenarios increases to high levels of 95.50% to 98.35%. This indicates complementary roles: M1 provides precise FSM input parsing, while M2 enables accurate LEBT structural mapping. Their integration enhances the conversion performance of LLMDUCF. In summary, LLMDUCF relies on the synergistic interaction between the FSM structural recognition and LEBT structural mapping components, rather than the contribution of any single component. Both are indispensable elements for achieving high-precision FSM to BT conversion.

Comparison of Conversion Accuracy Across Scenario  
![](images/4f2500109af72da20bbf274aba65ce2acd73e07bf44658fcb8bf29c40574b3f7.jpg)  
Figure 18: Comparison of Conversion Accuracy Across Scenario.

Table 6: Conversion Accuracy with 95% confidence intervals
<table><tr><td></td><td>scenario1</td><td>scenario2</td><td>scenario3</td></tr><tr><td>ChatGPT</td><td> $9 6 . 9 5 _ { \pm 3 . 2 2 }$ </td><td> $9 8 . 3 5 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $9 5 . 5 0 { \scriptstyle \pm 4 . 3 7 }$ </td></tr><tr><td>DeepSeek</td><td> $9 6 . 1 0 { \scriptstyle \pm 3 . 1 0 }$ </td><td> $9 7 . 9 5 { \scriptstyle \pm 2 . 8 2 }$ </td><td> $9 4 . 0 5 { \scriptstyle \pm 4 . 6 1 }$ </td></tr><tr><td>Qwen</td><td> $9 6 . 3 0 { \scriptstyle \pm 3 . 9 3 }$ </td><td> $9 6 . 5 0 { \scriptstyle \pm 3 . 0 9 }$ </td><td> $9 5 . 7 0 { \scriptstyle \pm 3 . 2 6 }$ </td></tr><tr><td>ChatGPT (without SCP)</td><td> $3 1 . 2 0 { \scriptstyle \pm 8 . 9 4 }$ </td><td> $3 8 . 7 0 { \scriptstyle \pm 9 . 1 5 }$ </td><td> $3 3 . 2 0 { \scriptstyle \pm 9 . 3 5 }$ </td></tr><tr><td>DeepSeek (without SCP)</td><td> $3 1 . 3 5 { \scriptstyle \pm 9 . 3 3 }$ </td><td> $3 2 . 2 0 { \scriptstyle \pm 9 . 5 5 }$ </td><td> $3 3 . 9 0 _ { \pm 1 0 . 6 5 }$ </td></tr><tr><td>Qwen (without SCP)</td><td> $3 2 . 7 0 { \scriptstyle \pm 9 . 1 0 }$ </td><td> $3 5 . 6 0 { \scriptstyle \pm 9 . 2 4 }$ </td><td> $3 2 . 1 0 { \scriptstyle \pm 9 . 3 0 }$ </td></tr></table>

To evaluate the conversion quality of LLMDUCF, we conducted simulation experiments under Scenario 3 to compare the performance of different conversion methods. The initial FSM employed in the experiments was designed manually , as shown in Fig. 20.

Behavioral Consistency: Figs. 21–24 show the BTs generated from the FSM in Fig. 20 by LLMDUCF, BSD, CHDS-BT and NLLM, respectively. As shown in Table VIII, NuSMV verification confirms that LLMDUCF, BSD, and CHDS-BT satisfy all behavioral consistency criteria and are equivalent to the original FSM. The predator-prey simulation further shows consistent trajectory behavior across these models. In contrast, NLLM fails in reachability, completeness, absence of additional behavior, and termination consistency, indicating semantic loss in FSM-to-BT conversion, particularly for loop structures. LLMDUCF preserves these behaviors by guiding the LLM to identify loop, chain, and branch structures and map them into the proposed LEBT representation.

Modularity: In BT, inserting a new node requires adding it to the child node list of its designated parent control node. The BT generated by the LLMDUCF and NLLM methods maintains clear decoupling between the child nodes, allowing insertion and deletion operations to be performed by directly accessing the parent node. This yields an optimal time complexity of O(1) for both operations.In contrast, BT produced by the BSD method incorporates transition relationships between nodes, where the next executable node is determined through explicit transition nodes. Inserting a new node necessitates traversing all transition nodes to update path connections, leading to a time complexity of O(n), in addition to requiring modifications to the transition node source code. The CHDS-BTs also maintain node transition links. However, in this case, it requires traversing all nodes and their associated transition paths to establish new connections, resulting in a time complexity of O(n²). The higher complexity underscores the maintenance challenges for dynamic and scalable autonomous systems, where frequent model updates are often required.

Comparison of Conversion Accuracy Among Different LLM  
![](images/90510f8c08d98a9fcdfdec8e7d994708867ea49551619261187c085897b7c431.jpg)  
Figure 19: Comparison of Conversion Accuracy Among Different LLM.

Table 7: Ablation Experiments
<table><tr><td></td><td>scenario1</td><td>scenario2</td><td>scenario3</td></tr><tr><td>Base</td><td> $3 1 . 2 0 { \scriptstyle \pm 8 . 9 4 }$ </td><td> $3 8 . 7 0 { \scriptstyle \pm 9 . 1 5 }$ </td><td> $3 3 . 2 0 { \scriptstyle \pm 9 . 3 5 }$ </td></tr><tr><td>Base+M1</td><td> $3 3 . 8 0 { \scriptstyle \pm 9 . 7 4 }$ </td><td> $3 8 . 6 5 _ { \pm 1 0 . 0 3 }$ </td><td> $3 6 . 0 5 { \scriptstyle \pm 1 0 . 9 6 }$ </td></tr><tr><td>Base+M2</td><td> $3 4 . 3 0 { \scriptstyle \pm 9 . 3 7 }$ </td><td> $4 6 . 2 0 { \scriptstyle \pm 1 0 . 8 6 }$ </td><td> $4 1 . 5 5 { \scriptstyle \pm 9 . 6 9 }$ </td></tr><tr><td>Base+M1+M2</td><td> $9 6 . 9 5 _ { \pm 3 . 2 2 }$ </td><td> $9 8 . 3 5 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $9 5 . 5 0 { \scriptstyle \pm 4 . 3 7 }$ </td></tr></table>

The edit distance serves as a key metric for evaluating the scalability and maintainability of BT models when extending their functionality. As illustrated in Fig. 25, when introducing new behaviors—namely the Gliding behavior, the Fire sub-tree, and Loading sub-tree behaviors into the base BT, the LLMDUCF and NLLM methods yield edit distances of 2, 6, and 8, respectively. This low modification cost stems from the node independence in these BT: adding a node or sub-tree only involves inserting the corresponding nodes and edges, without impacting existing structures. In contrast, the BSD-generated BT produces edit distances of 2, 8+k, and 10+k, where k denotes the number of existing transition relationships associated with the current node. Each action insertion requires adding 4 nodes and 4 edges. More importantly, incorporating a new sub-tree requires creating a new transition node and updating transition linkages across other sub-trees, increasing both structural and coding overhead. The CHDS-BT method results in even greater edit distances of 2, 8+6k, and 10+6k. Here, introducing a new sub-tree demands creating an entire conversion sub-tree to reestablish transition connections, significantly increasing operational complexity. These results highlight the advantages of LLMDUCF in supporting efficient and low-overhead model extension.

Readability: To quantitatively assess the readability of generated BT, we introduce a metric based on the number of graphical elements within the model structure. Let M denote the number of action nodes. In the proposed LLMDUCF method, each action node is associated with one selector node and one condition node. The number of sequence nodes is variable, but can be empirically estimated to be half the number of action behaviors. Thus, the total number of nodes is approximately 3.5M. To account for all graphical components, edges connecting the parent and child nodes must also be included, resulting in 3.5M-1 edges. This results in a total of 7M-1 graphical elements. In contrast, the

![](images/78a0a3ec92229ff5b3ada12fc9c2d6f55404434cf42954cc8e4618c808686057.jpg)  
Figure 20: FSM behavioral model of predator.

NLLM method produces non-deterministic BT structures, making it impossible to consistently quantify graphical elements. The BSD and CHDS-BT methods yield totals of 8M+1 and 8M+1+6k elements, respectively, where k reflects additional transition relations. The LLMDUCF method significantly reduces the graphical complexity compared to these alternatives, directly contributing to better readability.

## 4.6 Sensitivity and Robustness Evaluation

This section provides a systematic and comprehensive analysis of the proposed LLMDUCF framework, thoroughly examining its practical performance in terms of version sensitivity, edge case handling, and input-output compatibility.

Sensitivity: We conducted comparative experiments on the version sensitivity of large language models, contrasting the latest official stable versions with their preceding iterations, the results are shown in Table XI. Experimental results demonstrate that the proposed LLMDUCF framework exhibits less than 3% fluctuations in conversion accuracy between different versions of the same model. This deviation falls within the experimentally acceptable error range and does not affect the conclusions of the core research presented here. The low version sensitivity of this framework stems from the dual-constraint mechanism of SCP prompts and the post-processing structural verification layer. Their synergistic action effectively mitigates subtle logical deviations arising from large model version iterations, thereby ensuring the stability and consistency of bidirectional conversion results between FSM and BT across versions.

![](images/d643a3da05fd5cb5461f26bdc968d4726bcddc6e78d49c66ad0f5c42615a368c.jpg)  
Figure 21: BT model generated using proposed method.

Table 8: FSM to BT Behavioral Consistency
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD</td><td>CHDSs-BT</td></tr><tr><td>Reachability</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Completeness</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>No Additional Behavior</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Deadlock Absence</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Termination Consistency</td><td>√</td><td>X</td><td>√</td><td>√</td></tr></table>

Failure Cases and Recovery Methods: Table XII reports the number of first-pass failures, total retry attempts, and average retries per failed case in the three experimental scenarios. For both BT-to-FSM and FSM-to-BT conversion, the average number of retries ranges from 1.56 to 2.28, showing that most failed initial conversions can be corrected with a small number of feedback-guided regeneration attempts, achieving 100% accuracy. Behavioral model conversion is primarily an offline engineering activity performed during system design, migration, and refactoring. Its objective is to obtain a formally verified correct model, rather than to satisfy runtime latency constraints. Consequently, NuSMV-based verification, error feedback, and regeneration constitute an acceptable processing workflow.

Conversion failures in LLMDUCF primarily stem from hallucinations in LLM. Although this paper employs SCP prompts and conversion constraints, in a small number of complex or edge cases, the LLM may still fail to strictly follow certain prompts, leading to errors in structural mapping or the identification of conversion units. In the conversion from FSM-to-BT, a common error is the presence of redundant structures in the generated LEBT. For example, when a particular state is identified as both an entry state and a loop state, the LLM may generate duplicate sub-structures for the same state, leading to logical errors. This issue can be resolved by incorporating a node uniqueness constraint into the prompt, namely requiring that each state correspond to only one node in the LEBT, and that duplicate nodes be merged. In BT-to-FSM conversion, the primary error is the incorrect identification of transition units. LLMs sometimes treat every control node as an independent transition unit, resulting in the generation of too many sub-FSMs and an overly complex structure. This issue can be remedied by refining the definition of transition units and providing reference examples to guide the LLM in correctly partitioning transition units according to hierarchical rules. Therefore, although

![](images/2991709b667db5fb8f5931ce0d43e7082300d2d298d7904aed12d5ce928c8cdc.jpg)  
Figure 22: BT generated by BSD.

Table 9: Comparison of Structural Complexity
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD</td><td>CHDS-BT</td></tr><tr><td>Computational Complexity</td><td>O(1)</td><td>O(1)</td><td>O(n)</td><td>O(n2)</td></tr><tr><td>Graphical Elements</td><td>≈ 7M-1</td><td>1</td><td>8M+1</td><td>8M+1+6k</td></tr><tr><td>Active Elements</td><td>≈ 3.5M</td><td>1</td><td>4M+1</td><td>4M+1+3k</td></tr></table>

LLM outputs may still exhibit instability in a small number of complex structures, these errors can usually be corrected through targeted prompt optimization and enhanced structural constraints.

Input Representation Generalizability: LLMDUCF and NLLM are capable of processing three types of input representations—text files, images, and structured tables—and can adapt to unseen encoding schemes and new modeling conventions, demonstrating that methods based on large language models possess strong generalization capabilities regarding input representations. In contrast, BSD and CHDS-BT only support predefined text-based structured inputs; when inputs are presented as images, tables, unseen encodings, or new expression conventions, neither can perform the conversion directly and typically require additional format parsing, model preprocessing, or manually extended conversion rules. LLMDUCF can further improve the behavioral correctness and stability of conversion results across different input representations, thereby enhancing the flexibility of the model conversion process and interoperability between tools.

## 4.7 Experiment Summary and Discussion

The multi-scenario experimental evaluation comprehensively validates the performance and intelligent characteristics of the proposed LLMDUCF in bidirectional FSM-BT conversion. The results demonstrate a clear trade-off profile between traditional methods, NLLM, and our framework, highlighting its distinct value proposition.

Performance and Generality: While traditional rule-based methods (BSD, CHDS-BT) achieve 100% conversion accuracy through handcrafted deterministic templates, they inherently lack generalizability and require expert knowledge for each new model or scenario. In contrast, NLLM approaches offer automation but suffer from low and unstable accuracy due to insufficient comprehension of complex behavioral logic. The LLMDUCF strikes an effective balance: it achieves high conversion accuracy (ranging from 90.65% to 98.35% across all tasks) by leveraging the LLM’s parsing capability while constraining its reasoning with domain-specific rules and structural strategies. This enables robust performance across diverse scenarios—from basic symbols to embodied robotic tasks and dynamic game AI—without the need for scenario-specific template engineering. Furthermore, the framework offers advantages such as strong version stability, adaptability to complex structures, and compatibility with multimodal input and output. Although there are minor fluctuations in accuracy in a small number of edge cases, it demonstrates high overall reliability and practical value.

![](images/2fdf311a7756921da7f1c5b4d407e4c946d2828da17be098a701a255f5c26684.jpg)

Figure 23: BT generated by CHDS–BT.  
![](images/78500327e034be7d2688b3333302096085f77ab3bf8d90084be3566282bc731d.jpg)  
Figure 24: BT generated by NLLM.

Conversion Quality and Model Utility: Beyond accuracy, the quality of the generated models is paramount for practical application. In terms of behavioral consistency, the LLMDUCF-generated models fully satisfy all specified requirements (Tables III&VIII), demonstrating semantic equivalence to the original models. Regarding Modularity, operations such as adding or removing nodes in LLMDUCF-generated models exhibit lower computational complexity (often O(1)) and shorter edit distances compared to models from traditional methods (Tables V&X), indicating a superior ease of modification and extension. For Readability, LLMDUCF-generated models contain significantly fewer graphical and active elements than those produced by traditional methods (Tables IV&IX), leading to clearer visual representations and reduced cognitive load for engineers. Finally, LLMDUCF natively supports multimodal input, which improves flexibility and ease of use compared to traditional methods designed for specific formats.

Table 10: Edit Distance of BT
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD</td><td>CHDS-BT</td></tr><tr><td>Fire subtree</td><td>6</td><td>6</td><td>8 + k</td><td>8 + 6k</td></tr><tr><td>Gliding behavior</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Loading subtree</td><td>8</td><td>8</td><td>10 + k</td><td>10 + 6k</td></tr></table>

![](images/d82d7cd2bb26048b2671a0b14a60dfdb36517fbe075642bfd000f3430336d80a.jpg)  
Figure 25: Different cases of node addition in a BT. The added nodes are highlighted in red.

Cost Analysis: Although LLMDUCF has higher per-conversion latency than traditional rule-based methods, i.e., 60–70 s versus 0.1–0.5 s, behavioral model conversion in consumer-grade AIS is typically a low-frequency engineering task conducted during system design, migration, or refactoring rather than runtime execution. Therefore, overall engineering efficiency is more critical than single-instance latency. For 100 models, traditional methods require approximately 68 h in total, including approximately 48 h to develop a dedicated converter and 20 h to preprocess input-format, with an estimated cost of \$430–440. In contrast, LLMDUCF does not require dedicated converter development or strict preprocessing, completing the same workload in about 2 h with an API cost of approximately \$5–10(taking ChatGPT (gpt-4o) as an example, a single conversion consumes approximately 500 to 1000 tokens, with API call costs of roughly \$0.05 to 0.10 per call). Thus, LLMDUCF significantly reduces development time, labor effort, and overall cost, making it well suited for small-scale or occasional conversion of a behavioral model in consumer AIS scenarios. Furthermore, in the development of consumer-grade AIS, behavioral model conversion is primarily applied during the system design, model migration, and model refactoring phases. Once conversion is complete, the target behavior model can be used directly or in a further way maintained, and no additional model conversion is required during AIS runtime. Therefore, the requirements for real-time performance are relatively lenient.

In addition, we further validated LLMDUCF on consumer-grade hardware using a locally deployed Qwen-35B model via Ollama 3.1 on a PC with an RTX 3060 GPU and 16 GB RAM. In 20 representative behavioral models, local deployment achieved an accuracy of 80% for FSM-to-LEBT conversion and 85% for BT-to-FSM conversion, with an average latency of approximately 2 min per model. Although its performance is lower than that of cloud-based large-scale models due to weaker structured reasoning and hardware-constrained quantization/offloading, the results confirm the feasibility of low-cost, cloud-independent deployment for consumer-grade autonomous intelligent systems.

Table 11: Comparison of Version Sensitivity
<table><tr><td></td><td>BT to FSM</td><td>FSM to BT</td></tr><tr><td>GPT(gpt-4-turbo)</td><td>93.85%</td><td>96.20%</td></tr><tr><td>GPT(gpt-4o)</td><td>94.55%</td><td>97.05%</td></tr><tr><td>DeepSeek(v3.1)</td><td>94.25%</td><td>95.30%</td></tr><tr><td>DeepSeek(v3.2)</td><td>95.10%</td><td>96.15%</td></tr><tr><td>Qwen(v2)</td><td>92.90%</td><td>95.45%</td></tr><tr><td>Qwen(v3)</td><td>93.80%</td><td>95.75%</td></tr></table>

Table 12: Retry Statistics of Failed Conversion Cases
<table><tr><td></td><td>Failure Cases</td><td>Total Retries</td><td>Average Retries</td></tr><tr><td>BT-FSM(ChatGPT)</td><td>27</td><td>42</td><td>1.56</td></tr><tr><td>BT-FSM(DeepSeek)</td><td>31</td><td>69</td><td>2.23</td></tr><tr><td>BT-FSM(Qwen)</td><td>35</td><td>76</td><td>2.17</td></tr><tr><td>FSM-BT(ChatGPT)</td><td>18</td><td>41</td><td>2.28</td></tr><tr><td>FSM-BT(DeepSeek)</td><td>24</td><td>53</td><td>2.21</td></tr><tr><td>FSM-BT(Qwen)</td><td>23</td><td>51</td><td>2.22</td></tr></table>

Limitations and Future Prospects: LLMDUCF is only applicable to behavioral models with a finite set of behaviors, complete structures, and logical consistency. For models containing anomalies, ambiguities, or semantic vagueness, such issues must be resolved prior to formal transformation. In addition, the current framework supports only fundamental structural transformations of behavioral models. For more complex model variants, we shall refine and extend our approach in future work: Parallel nodes may be mapped as a set of concurrently executed sub-FSMs, or a runtime environment supporting concurrent semantics may be introduced to achieve synchronous scheduling and state synchronization for multi-branch behaviors; For decorator nodes (such as UntilSuccess, Repeat, etc.), these may be modeled as loop states within the FSM featuring specific exit conditions, with decorator logic expressed equivalently through customized transition constraints; The core of hierarchical FSM transformation lies in hierarchical nesting and cross-layer transitions. A hierarchical FSM can be decoupled into a set of FSMs in the order of top-level and sub-levels, starting from the lowest-level FSM and transforming it into a sub-tree according to the transformation rules, ultimately constructing a composite LEBT. The aforementioned extensions necessitate syntactic expansion of the existing LEBT and FSM template alongside the design of more refined LLM prompt engineering. This will guide the model in understanding and processing these advanced behavioral semantics while ensuring semantic consistency and executability of the conversion model.

Table 13: Input Representation Generalizability
<table><tr><td></td><td>LLMDUCF</td><td>NLLM</td><td>BSD</td><td>CHDS-BT</td></tr><tr><td>Text File</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Image</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Table</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Unseen Encoding</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>New Conventions</td><td>√</td><td>√</td><td>X</td><td>X</td></tr></table>

## 5 CONCLUSION

This paper presents an LLM-driven unified conversion framework that enables a fully automated and semantically consistent bidirectional transformation between FSM and BT. The proposed LEBT structure effectively preserves loop behavior in FSM-to-BT conversion, while a compression and hierarchical rule strategy successfully mitigates the sub-FSM explosion problem in the reverse direction. Comprehensive experiments in multiple autonomous decisionmaking scenarios demonstrate that LLMDUCF achieves high conversion accuracy (90.65%–98.35%) while generating models with superior structural quality—exhibiting full path coverage, lower modification complexity, and improved readability compared to traditional methods. This work provides a practical and automated solution for enhancing the interoperability of behavioral models in AIS, it provides practical tools for the transfer, verification, and maintenance of autonomous behavior logic in service robots, smart home devices, game agents, and smart electric vehicles.

## References

[1] Sarkar M, Pradhan J, Singh A K, et al. A novel hybrid quantum architecture for path planning in quantum-enabled autonomous mobile robots[J]. IEEE Transactions on Consumer Electronics, 2024, 70(3): 5597-5606.

[2] Abbasi AZ, Asif M, Hollebeek LD, Islam JU, Ting DH, Rehman U (2021), "The effects of consumer esports videogame engagement on consumption behaviors". Journal of Product & Brand Management, Vol. 30 No. 8 pp. 1194–1211.

[3] Haghighi M S, Farivar F, Jolfaei A, et al. Cyber attacks via consumer electronics: Studying the threat of covert malware in smart and autonomous vehicles[J]. IEEE Transactions on Consumer Electronics, 2023, 69(4): 825-832.

[4] Rahman M S, Guo R, Kamali H M, et al. ReTrustFSM: toward RTL hardware obfuscation-a hybrid FSM approach[J]. IEEE Access, 2023, 11: 19741-19761.

[5] Singh R P, Gergondet P, Kanehiro F. mc-mujoco: Simulating articulated robots with fsm controllers in mujoco[C]//2023 IEEE/SICE International Symposium on System Integration (SII). IEEE, 2023: 1-5.

[6] Iovino M, Scukins E, Styrud J, et al. A survey of behavior trees in robotics and ai[J]. Robotics and Autonomous Systems, 2022, 154: 104096.

[7] Ögren P, Sprague C I. Behavior trees in robot control systems[J]. Annual Review of Control, Robotics, and Autonomous Systems, 2022, 5(1): 81-107.

[8] Zhang Q, Yao J, Yin Q, et al. Learning behavior trees for autonomous agents with hybrid constraints evolution[J]. Applied Sciences, 2018, 8(7): 1077.

[9] Iovino M, Förster J, Falco P, et al. On the programming effort required to generate Behavior Trees and Finite State Machines for robotic applications[J]. arxiv preprint arxiv:2209.07392, 2022.

[10] Ghzouli R, Berger T, Johnsen E B, et al. Behavior trees and state machines in robotics applications[J]. IEEE Transactions on Software Engineering, 2023, 49(9): 4243-4267.

[11] Iovino M, Förster J, Falco P, et al. Comparison between behavior trees and finite state machines[J]. arxiv preprint arxiv:2405.16137, 2024.

[12] C. Pezzato, C. H. Corbato, S. Bonhof and M. Wisse, "Active Inference and Behavior Trees for Reactive Action Planning and Execution in Robotics," in IEEE Transactions on Robotics, vol. 39, no. 2, pp. 1050-1069, April 2023.

[13] Schulz-Rosengarten A, Ahmad A, Clement M, et al. Behavior trees with dataflow: Coordinating reactive tasks in lingua franca[C]//Proceedings of the 2024 IEEE/ACM 46th International Conference on Software Engineering: Companion Proceedings. 2024: 304-305.

[14] Zutell J M, Conner D C, Schillinger P. Flexible behavior trees: In search of the mythical HFSMBTH for collaborative autonomy in robotics[J]. arxiv preprint arxiv:2203.05389, 2022.

[15] Colledanchise M, Ögren P. Behavior trees in robotics and AI: An introduction[M]. CRC Press, 2018.

[16] Marzinotto A, Colledanchise M, Smith C, et al. Towards a unified behavior trees framework for robot control[C]//2014 IEEE international conference on robotics and automation (ICRA). IEEE, 2014: 5420-5427.

[17] Zhou H, Hu C, Yuan Y, et al. Large language model (llm) for telecommunications: A comprehensive survey on principles, key techniques, and opportunities[J]. IEEE Communications Surveys & Tutorials, 2024.

[18] Chkirbene Z, Hamila R, Gouissem A, et al. Large language models (llm) in industry: A survey of applications, challenges, and trends[C]//2024 IEEE 21st International Conference on Smart Communities: Improving Quality of Life using AI, Robotics and IoT (HONET). IEEE, 2024: 229-234.

[19] Friha O, Ferrag M A, Kantarci B, et al. Llm-based edge intelligence: A comprehensive survey on architectures, applications, security and trustworthiness[J]. IEEE Open Journal of the Communications Society, 2024.

[20] Yao Y, Duan J, Xu K, et al. A survey on large language model (llm) security and privacy: The good, the bad, and the ugly[J]. High-Confidence Computing, 2024, 4(2): 100211.

[21] Li B, Jiang Y, Gadepally V, et al. Llm inference serving: Survey of recent advances and opportunities[C]//2024 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2024: 1-8.

[22] Kumar P. Large language models (LLMs): survey, technical frameworks, and future challenges[J]. Artificial Intelligence Review, 2024, 57(10): 260.

[23] Martín F, Lera F J R, Ginés J, et al. Evolution of a cognitive architecture for social robots: Integrating behaviors and symbolic knowledge[J]. Applied Sciences, 2020, 10(17): 6067.

[24] Hallen M, Iovino M, Sander-Tavallaey S, et al. Behavior trees in industrial applications: a case study in underground explosive charging[C]//2024 IEEE 20th International Conference on Automation Science and Engineering (CASE). IEEE, 2024: 156-162.

[25] Gan X R, Song Y R, Walker N, et al. Can Large Language Models Help Developers with Robotic Finite State Machine Modification?[J]. arxiv preprint arxiv:2412.05625, 2024.

[26] Lin Q K, Hsu C, Chang T S. Enhancing Finite State Machine Design Automation with Large Language Models and Prompt Engineering Techniques[C]//2024 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS). IEEE, 2024: 475-478.

[27] Swick B, Donegan S, Gillman A, et al. Human Planning of Robot Actions through LLM-guided State Machine Synthesis[C]//2024 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN). IEEE, 2024: 430-437.

[28] Zhou H, Lin Y, Yan L, et al. Llm-bt: Performing robotic adaptive tasks based on large language models and behavior trees[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024: 16655- 16661.

[29] Lykov A, Tsetserukou D. Llm-brain: Ai-driven fast generation of robot behaviour tree based on large language model[C]//2024 2nd International Conference on Foundation and Large Language Models (FLLM). IEEE, 2024: 392-397.

[30] Koutavas V, Lin Y Y, Tzevelekos N. Fully abstract normal form bisimulation for call-by-value PCF[J]. Journal of the ACM, 2025, 72(6): 1-52.