# Towards Numerical TOHTN Planning with SMT-based HTN-SAT Encoding

Takudzwa Togarepi<sup>1</sup>, Gaspard Quenard<sup>1</sup>, Damien Pellier<sup>1</sup>, Humbert Fiorino<sup>1</sup>

<sup>1</sup>Univ. Grenoble Alpes, France

{takudzwa.togarepi, gaspard.quenard, damien.pellier, humbert.fiorino}@univ-grenoble-alpes.fr

## Abstract

While HTN planning has received significant attention in recent years, support for numerical reasoning remains very limited. In this paper, we investigate numerical Totally-Ordered HTN (TOHTN) planning and show how standard SAT-based encodings can be naturally extended with SMT to handle numeric fluents. In addition, we introduce a benchmark suite for numerical TOHTN planning, providing a first common basis for evaluation in this setting. Experimental results show that this simple encoding already constitutes a competitive baseline. This work opens the way to more expressive approaches to HTN planning.

## Introduction

Hierarchical Task Network (HTN) planning (Erol, Hendler, and Nau 1994) is a planning paradigm that decomposes complex tasks into simpler subtasks using domain-specific knowledge. Unlike classical planning, HTN introduces abstract tasks, which cannot be executed directly, and methods, which describe how these tasks can be refined into partially ordered sets of subtasks including both primitive tasks (i.e., executable actions) and additional abstract tasks that must themselves be recursively refined. The objective of an HTN planner is to iteratively decompose an initial abstract task into a valid plan (i.e., an executable sequence of primitive tasks). In this paper, we focus our investigation on Totally-Ordered HTN (TOHTN) planning, a highly popular subclass of HTN problems where the decomposition methods specify a totally-ordered list of primitive actions and abstract tasks to be executed in order to achieve an abstract task.

While HTN planning is a central topic in automated planning and has been included in recent editions of the International Planning Competition (IPC) (Behnke et al. 2019; Taitler et al. 2024), it still suffers from important modeling limitations compared to classical planning formalisms. In particular, numerical and temporal features (such as resource management, costs, and durations) introduced in classical planning with PDDL2.1 (Fox and Long 2003) are absent from most HTN planners. Although some work has investigated the integration of temporal aspects into HTN planning (Pellier et al. 2022), the formalization of temporal HTN remains an active research topic, and current proposals have not yet reached broad adoption. In contrast, numerical reasoning (e.g., through numerical constraints over the preconditions and effects of primitive tasks) can be naturally incorporated, but has received little attention so far. These features are essential in many real-world applications, including logistics, robotics, and scheduling, where reasoning about quantities is required. To the best of our knowledge, only Siadex (Castillo et al. 2006) and Aries (Bit-Monnot 2023) support numerical reasoning in HTN planning. This limitation restricts the applicability of HTN planning to realistic domains.

In this paper, we propose to extend SAT-based TOHTN planning to handle numerical constraints by leveraging Satisfiability Modulo Theories (SMT). SAT-based approaches have shown strong performance in recent years in TOHTN planning due to both the efficiency of modern solvers and improved encodings and search strategies (Schreiber et al. 2019; Behnke, Holler, and Biundo 2018; Schreiber 2021;¨ Behnke 2021; Quenard, Pellier, and Fiorino 2024, 2025), but they are inherently limited to propositional representations. In contrast to heuristic-search-based approaches, which typically require substantial adaptations to handle numerical reasoning, SAT-based methods can be more naturally extended by lifting the encoding to SMT, without fundamentally modifying the search procedure. By moving to SMT, we enable reasoning over numerical variables while preserving the benefits of logical encodings. In this paper, we introduce a new encoding that extends SAT-based TOHTN planning to handle numerical variables and constraints using an SMT solver. Additionally we design and provide seven numerical TOHTN benchmarks to evaluate these approaches. We experimentally show that our SMT-based encoding enables solving numerical TOHTN problems more efficiently than existing numerical HTN planners.

This paper is organized as follows: first, we introduce the concept of numerical TOHTN planning. Next, we describe the basic incremental encoding used by current SAT-based TOHTN planners to find a solution. Then, we explain how this encoding can be modified to support numerical constraints. Finally, we compare this approach with other numerical HTN planners.

## Numerical TOHTN Planning Problem

We present a formalization of numerical TOHTN planning, building on (Behnke, Holler, and Biundo 2018; Behnke¨ 2021; Quenard, Pellier, and Fiorino 2024) and following the

treatment of numeric fluents introduced in PDDL2.1 and discussed for HDDL 2.1 (Pellier et al. 2022).

## Tasks, Actions, Methods, Numeric Fluents, and Task Networks

Tasks are central to HTN planning. A task is defined by a name and parameters. Tasks are either primitive or abstract: primitive tasks directly affect the state of the world, while abstract tasks do not; instead, they must be decomposed into primitive tasks using methods before they can be executed.

We assume a finite set L of propositions and a finite set $F$ of numeric fluents. A numeric expression over $F$ is built from constants and fluents in $F$ using the arithmetic operators $\mathbf { \sigma } + \mathbf { , - } , \mathbf { \times } , / . \mathbf { A }$ numeric constraint is an expression of the form $f$ ▷◁ ξ, where $f \in F$ is a numeric fluent, ξ is a numeric expression, and ▷◁∈ $\{ < , \leq , = , \geq , > \}$

A primitive task a is analogous to an action in classical planning and is defined by a tuple $( n a m e ( a ) , p r e c o n d ( a ) , e f f e c t ( a ) )$ . Its preconditions $\begin{array} { l l l } { p r e c o n d ( a ) } & { = } & { ( p r e c o n d _ { L } ( a ) , p r e c o n d _ { N } ( a ) ) } \end{array}$ consist of a set of propositional preconditions precond (a) and a set of numeric constraints $p r e c o n d _ { N } ( a )$ . Its effects $e f f e c t ( a ) \ = \ ( e f f e c t ^ { + } ( a ) , e f f e c t ^ { - } ( a ) , e f f e c t _ { N } ( a ) )$ consist of add and delete effects over propositions and a set of numeric effects $e f f e c t _ { N } ( a )$ . In this work, we restrict numeric effects to assignment effects of the form $f : = \xi$ , where $f \in F$ and ξ is a numeric expression over $F$

A state s is defined as a pair $( l , v )$ where $l \subseteq L$ is the set of propositions true in the state, and ${ \dot { v } } : F \to \mathbb { R }$ is a valuation function assigning a real value to each numeric fluent. A task a is executable in $\boldsymbol { s } = ( l , \boldsymbol { v } )$ iff $p r e c o n d _ { L } ( a ) \subseteq l$ and v |= $p r e c o n d _ { N } ( a )$ , i.e., all numeric constraints in $p r e c o n d _ { N } ( a )$ are satisfied under v. If a task a is executable in a state s = $( l , v )$ , applying its effects yields a new state $s ^ { \prime } = ( l ^ { \prime } , v ^ { \prime } )$ where $l ^ { \prime } = ( l \setminus e f f e c t ^ { - } ( a ) ) \cup e f f e c t ^ { + } ( a )$ and $v ^ { \prime }$ is obtained by applying $e f f e c t _ { N } ( a )$ to v.

A method $m = ( n a m e ( m ) , c , w _ { m } )$ indicates how an abstract task c can be refined into a task network $w _ { m } ,$ , called the subtasks of m. For notation purposes, we define $M ( c ) =$ $\{ m = ( n a m e ( m ) , c , w _ { m } ) \mid m \stackrel { . } { \in } \bar { M } \}$ as the set of all methods that can be applied to decompose the abstract task c.

## Planning Problem and Solution

Definition 1 (TOHTN Planning Problem) A numerical TOHTN planning problem $\textit { P }  { \textit { i s } }  { \textit { a } }$ tuple $( L , F , C , A , M , c _ { I } , s _ { I } , g )$ where: L is a finite set of propositions; F is a finite set of numeric fluents; C is a finite set of abstract tasks; A is a finite set of primitive tasks; M is a finite set of decomposition methods; $c _ { I } \in C$ is the initial abstract task to be decomposed; $s _ { I } = ( l _ { I } , v _ { I } ) \in S$ is the initial state; and $\boldsymbol { g } = \left( g _ { L } , g _ { N } \right)$ is the goal condition, where $g _ { L } \subseteq L$ is a set of propositions and $g _ { N }$ is a set of numeric constraints.

Definition 2 (TOHTN Planning Problem Solution) A solution to a numerical TOHTN planning problem P is a primitive task network $\pi \in A ^ { * }$ such that:

1. π is obtained by refining $c _ { I }$ using methods,

2. π is executable in $s _ { I } ,$

3. π reaches the goal g after execution.

## SAT-based Search in TOHTN

HTN planning can be naturally represented as an AND/OR tree (Ghallab, Nau, and Traverso 2004), where the root contains the initial abstract task. This tree represents a finite fragment of the potentially infinite decompositions of the initial abstract task, and is incrementally expanded during search. OR nodes correspond to abstract tasks with multiple possible decompositions (methods), while AND nodes represent methods whose subtasks must all be achieved. A valid plan is obtained by selecting exactly one child for each OR node and all children for each AND node, such that the leaves form a sequence of primitive actions achieving the goal. An example of such AND/OR tree is given in the left side of Figure 1. This tree is not fully developed here, since the abstract task $T _ { 8 }$ is undeveloped.

In SAT-based HTN planning, the AND/OR tree is encoded into a SAT formula that is satisfiable iff a solution exists within it, with the satisfying assignment yielding the plan. However, the AND/OR tree alone is insufficient for current encodings. Indeed, clauses must capture action transitions, ensuring preconditions hold at execution time and effects are applied. All current SAT-based HTN planners address this by assigning each task a discrete time step indicating when it may execute; something the AND/OR tree’s structural representation does not provide.

That is why all current HTN-SAT planners rely on a structure introduced independently in TreeRex and totSAT (Schreiber et al. 2019; Behnke, Holler, and Biundo 2018),¨ which is equivalent to the AND/OR tree while making execution time steps explicit. We call this structure a Compact Path Decomposition Tree (cPDT); its nodes may contain multiple tasks and are organized as follows:

• The root node contains only the initial abstract task $c _ { I }$

• To expand a node $P ,$ , its children are built as follows:

– For each abstract task c in P and each method $m _ { i }$ decomposing c into subtasks $\langle t _ { 1 } , \ldots , t _ { n } \rangle$ , the k-th child of $\dot { P }$ contains task $t _ { k }$

– For each primitive task a in $P ,$ the first child of P contains a.

The cPDT guarantees that for any task t at leaf $l _ { i } ,$ all tasks that must precede t appear at leaves $l _ { j }$ with $j < i$ , and all that must follow appear at $l _ { r }$ with $r > i$ . Consequently, a left-toright scan of the leaves reveals all possible plans encoded in the cPDT. The cPDT corresponding to the AND/OR tree of Figure 1 is shown on its right side.

We now present an incremental encoding, capturing the core ideas shared by current HTN-SAT encodings (Schreiber et al. 2019; Behnke, Holler, and Biundo 2018; Schreiber¨ 2021), that determines whether a solution exists in a given cPDT. This encoding serves as the basis for the numerical extension introduced next. For each cPDT node $P ,$ , we use Boolean variables $P _ { t }$ indicating that task t is active at $P ,$ variables $P _ { p }$ indicating that proposition $p \in L$ holds at $P ,$ and an auxiliary variable $P _ { p r i m }$ indicating that a primitive task is active in P. When t is a primitive task, we write it as $a ;$ when t is an abstract task, we write it as c. For any leaf $P ,$ we denote by $P ^ { + } ~ = ~ s u c c e s s o r ( P )$ the leaf executed immediately after $P$ in the left-to-right order induced by the decomposition hierarchy. To handle the goal state, we introduce a special virtual node $G ,$ representing the goal. Thus, for any node $P$ that has no successor, we define $P ^ { + } = G$ . As shown in Figure 1, there may exist solutions in which no task is active at a given position. To handle this case, we introduce a special action ε such that $p r e c o n d ( \varepsilon ) = e f f e c t ( \varepsilon ) = \varnothing$ . This action is treated like a normal action in the encoding, but it is ignored in the final plan. We write $P _ { \varepsilon }$ to denote that no task is active at node $P .$

![](images/df42b445a3a172116e539ddb0e84d822e809bc53efa8d16833b57d8f8dd5fe8d.jpg)  
Figure 1: On the left, a simple decomposition schema in the form of an AND/OR tree, where $T _ { i }$ denotes abstract task $i ,$ $M _ { i }$ denotes method i and $A _ { i }$ denotes action i. On the right, we represent the corresponding cPDT. An identical potential solution for both structures is highlighted in blue.

Initial task, initial state, and goal. At the root node $R ,$ the initial task and initial state must hold, and the goal must hold at G:

$$
\begin{array} { r l } & { R _ { c _ { I } } } \\ & { \forall p \in s _ { I } : R _ { p } \qquad \forall p \in L \setminus s _ { I } : \neg R _ { p } } \\ & { \forall p \in g : G _ { p } } \end{array}
$$

Leaf constraints. For every leaf node $P ,$ exactly one task is active at $P \colon$

$$
\begin{array} { l } { \bigvee _ { t \in T a s k s ( P ) } P _ { t } } \\ { \bigwedge _ { \begin{array} { c } { t , t ^ { \prime } \in T a s k s ( P ) } \\ { t \not \in t ^ { \prime } } \end{array} } \neg ( P _ { t } \wedge P _ { t ^ { \prime } } ) } \end{array}
$$

If a primitive task a is active in $P ,$ then its preconditions must hold at $P$ and its effects at $P ^ { + }$ :

$$
\begin{array} { l l } { { \forall a \in T a s k s ( P ) \cap A : } } & { { P _ { a } \Rightarrow \displaystyle \bigwedge _ { p \in p r e c o n d ( a ) } P _ { p } } } \\ { { } } & { { } } \\ { { \forall a \in T a s k s ( P ) \cap A : } } & { { P _ { a } \Rightarrow \displaystyle \bigwedge _ { p \in e f f e c t ^ { + } ( a ) } P _ { p } ^ { + } } } \\ { { } } & { { } } \\ { { \forall a \in T a s k s ( P ) \cap A : } } & { { P _ { a } \Rightarrow \displaystyle \bigwedge _ { p \in e f f e c t ^ { - } ( a ) } \neg P _ { p } ^ { + } } } \end{array}
$$

Frame axioms ensure that if a fact changes value between a node and its successor, it must be explained by a specific

action that can change this fact or an abstract task:

$$
\begin{array} { r l } & { \forall p \in L : \quad \neg P _ { p } \land P _ { p } ^ { + } \Rightarrow ( \neg P _ { p r i m } \lor \negthickspace \negthickspace P _ { a } \land P _ { a } ) } \\ & { \forall p \in L : \quad \neg P _ { p } \land P _ { p } ^ { + } \Rightarrow ( \neg P _ { p r i m } \lor \negmedskip ) } \\ & { \forall p \in L : \quad P _ { p } \land \neg P _ { p } ^ { + } \Rightarrow ( \neg P _ { p r i m } \lor  \emptyset P _ { a } ) } \\ & { \forall p \in L : \quad P _ { p } \land \neg P _ { p } ^ { + } \Rightarrow ( \neg P _ { p r i m } \lor  \emptyset P _ { a } ) } \\ & { \forall e \notin P e t ^ { - } ( a ) } \end{array}
$$

Finally, if an action is chosen on this position, this node is primitive:

$$
\begin{array} { r l } { \forall a \in T a s k s ( P ) \cap A : } & { { } P _ { a } \Rightarrow P _ { p r i m } } \\ { \forall c \in T a s k s ( P ) \cap C : } & { { } P _ { c } \Rightarrow \neg P _ { p r i m } } \end{array}
$$

Hierarchical constraints. Let $P$ be an expanded node with children $\langle P ^ { 1 } , \ldots , P ^ { k } \rangle$ . Facts hold in $P$ iff they hold in its first child:

$$
\begin{array} { r l } { \forall p \in L : } & { { } \ P _ { p } \Leftrightarrow P _ { p } ^ { 1 } } \end{array}
$$

If an abstract task $c$ is selected at $P ,$ then exactly one applicable method must be chosen to decompose this task, its subtasks must be assigned to the children in order, and all remaining children must receive ε:

∀c ∈ T asks(P) ∩ C :

$$
P _ { c } \Rightarrow { \bigvee } _ { m \in M \left( c \right) } \left( \bigwedge _ { i = 1 \atop i } ^ { | s u b t a s k s ( m ) | } P _ { s u b t a s k s ( m ) \left[ i \right] } ^ { i } \right.
$$

If a primitive task a is selected at P (including ε), it is propagated to the first child and remaining children receive ε:

$$
\forall a \in T a s k s ( P ) \cap A : \quad P _ { a } \Rightarrow \left( P _ { a } ^ { 1 } \land \bigwedge _ { i = 2 } ^ { k } P _ { \varepsilon } ^ { i } \right)
$$

Primitivity of the leaves. Finally, in order to obtain a solution, all leaves must contain primitive actions.

$$
\underset { P \in l e a v e s ( c P D T ) } { \bigwedge } P _ { p r i m }
$$

The clauses ensuring the initial task, initial state, and goal are generated only during the first encoding of the cPDT, whereas the clause ensuring the primitivity of the leaves is provided only as assumptions (temporary hypotheses active during a single SAT call). All other clauses remain valid after extending the cPDT. When the cPDT is expanded, hierarchical constraints are generated for the newly expanded nodes, and leaf constraints are generated for the new leaf positions (i.e., the children of the expanded nodes).

## SMT Extension for Numerical TOHTN

To extend the previous SAT encoding to numerical TOHTN, we introduce for each node $P$ and each numeric fluent $f \in$ $F$ a numeric variable $P _ { f }$ denoting the value of $f$ at $P .$ . For any numeric expression or constraint $\xi ,$ we write $[ [ \xi ] ] _ { P }$ for the arithmetic expression or constraint obtained by replacing each fluent $f$ occurring in ξ by the numerical variable $P _ { f }$

Initial numeric state and numeric goal. The initial valuation of the numeric fluents is encoded at the root node:

$$
\forall f \in F : \quad R _ { f } = v _ { I } ( f )
$$

Each numeric goal constraint must hold at the goal node $G \colon$

$$
\forall \xi \in g _ { N } : \quad \mathbb { [ } \xi ] ] _ { G }
$$

Numeric leaf constraints. If a primitive task a is active at leaf node $P ,$ then its numeric preconditions must hold at $P \colon$

$$
\forall a \in T a s k s ( P ) \cap A , \forall \xi \in p r e c o n d _ { N } ( a ) : \quad P _ { a } \Rightarrow \ [ \xi ] _ { P }
$$

If a is active at $P ,$ then each numeric effect of the form $f = \xi$ determines the value of $f$ at $P ^ { + }$ :

$$
\begin{array} { c } { { \forall a \in T a s k s ( P ) \cap A , \forall ( f = \xi ) \in e f f e c t _ { N } ( a ) : } } \\ { { { } } } \\ { { P _ { a } \Rightarrow ( P _ { f } ^ { + } = \mathbb { [ } \xi ] | _ { P } ) } } \end{array}
$$

Numeric frame axioms. For each leaf node P and each fluent $f \in F _ { \ast }$ , the value of $f$ can change between $P$ and $P ^ { + }$ only if the selected task is abstract or a primitive action that can modify $f \colon$

$$
( P _ { f } \ne P _ { f } ^ { + } ) \Rightarrow \left( \neg P _ { p r i m } \ \vee \ \bigvee _ { \begin{array} { c } { { a \in T a s k s ( P ) \cap A } } \\ { { f \in e f f e c t { N } } } \end{array} } P _ { a } \right)
$$

Numeric hierarchical constraints. $\mathbf { A } \mathbf { s }$ in the propositional case, numeric fluent values are constrained to be equal between a node $P$ and its first child $P ^ { 1 }$ :

$$
\forall f \in F : \quad P _ { f } = P _ { f } ^ { 1 }
$$

All Boolean clauses from the SAT encoding remain unchanged. The numeric clauses described above are generated following the same incremental scheme: the initial numeric state and numeric goal clauses are generated once at the first encoding, while the numeric precondition, effect, and frame axiom clauses are generated for each new leaf node as the cPDT is extended and numerical hierarchical constraints are generated for the newly expanded nodes.

## Evaluation

We compare our approach with the two numerical HTN planners available to us. The evaluation involves three planners: SibylSmt, our planner using the encoding described above with the Z3 SMT solver (De Moura and Bjørner 2008) and a BFS exploration $( \mathrm { i . e . , }$ expanding all nodes of the cPDT if no solution is found)<sup>1</sup>; Siadex (Castillo et al. 2006), an HTN planner derived from SHOP (Nau et al.

1999, 2003) with constraint-based reasoning; and Aries (Bit-Monnot 2023), a hybrid CP/SAT planner. Experiments were run on an Intel Core i7-12700H with 32GB RAM, with a 10-minute time limit per instance.

To the best of our knowledge, no standard benchmark suite exists for numerical HTN planning. We therefore introduce seven numerical HTN benchmark families, each with an instance generator<sup>2</sup>. For example, TRANSPORT-FUEL models fuel-limited delivery with possible refueling, while TRANSVASEMENT captures water-jug-style transfers (reaching a target quantity in a specific container by pouring between containers of different capacities).

<table><tr><td>Domain</td><td>SibylSmt</td><td>Aries</td><td>Siadex</td></tr><tr><td>Alchemy 20 Backpack 20 Gripper-colors 20</td><td>10.54 9.99</td><td>7.79 4.59</td><td>12.82 0 6.81</td></tr><tr><td>Minecraft 20 Overcooked 20</td><td>9.54 9.11 12.58</td><td>7.82 4.09 0</td><td>19.91 1</td></tr><tr><td>Transport-Fuel 20 Transvasement 20 Coverage 140</td><td>7.89 7.17 91</td><td>1.61 5.82</td><td>0 0 42</td></tr></table>

Table 1: Performance of each planner on the numerical benchmark suite. Each cell reports the agile score per domain. Total coverage and agile score are shown in the last two rows. Agile score is computed as Agile score = min $\left( 1 , 1 - \frac { \log ( t ) } { \log ( T ) } \right)$ if a plan is found, and 0 otherwise, where T is the time limit and t the solving time (in seconds).

Results are shown in Table 1. Our approach achieves higher coverage and agile scores (reflecting how quickly a planner finds a solution) than the other planners on most benchmarks, suggesting that it is a promising direction for numerical HTN problems. In particular, it appears more robust, being able to solve multiple instances across all benchmark families, while the other planners fail on some domains. Siadex notably struggles on recursive domains with strong combinatorial branching, where the choice of decomposition methods is not easily guided by the current state. Finally, none of the planners saturate all the benchmarks, leaving significant room for improvement and the development of more powerful approaches.

## Conclusion

Numerical reasoning is essential in many planning applications, yet remains largely underexplored in HTN planning. We introduced a benchmark suite for numerical TOHTN and a simple SMT-based extension of HTN-SAT encodings to handle numeric fluents. This provides a baseline for future work and enables empirical comparison on shared instances. We hope this work encourages further research on expressive HTN models and stronger planners and benchmarks for numerical hierarchical planning.

## References

Behnke, G. 2021. Block compression and invariant pruning for SAT-based totally-ordered HTN planning. In Proceedings of the International Conference on Automated Planning and Scheduling, volume 31, 25–35.

Behnke, G.; Holler, D.; Bercher, P.; Biundo, S.; Pellier, D.;¨ Fiorino, H.; and Alford, R. 2019. Hierarchical planning in the IPC. In Workshop on HTN Planning (ICAPS).

Behnke, G.; Holler, D.; and Biundo, S. 2018. totSAT-¨ Totally-ordered hierarchical planning through SAT. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Bit-Monnot, A. 2023. Experimenting with Lifted Plan-Space Planning as Scheduling: Aries in the 2023 IPC. In 2023 International Planning Competition at the 33rd International Conference on Automated Planning and Scheduling.

Castillo, L. A.; Fernandez-Olivares, J.; Garcia-Perez, O.;´ and Palao, F. 2006. Efficiently Handling Temporal Knowledge in an HTN Planner. In ICAPS, 63–72.

De Moura, L.; and Bjørner, N. 2008. Z3: An efficient SMT solver. In International conference on Tools and Algorithms for the Construction and Analysis of Systems, 337– 340. Springer.

Erol, K.; Hendler, J. A.; and Nau, D. S. 1994. UMCP: A Sound and Complete Procedure for Hierarchical Tasknetwork Planning. In Aips, volume 94, 249–254.

Fox, M.; and Long, D. 2003. PDDL2. 1: An extension to PDDL for expressing temporal planning domains. Journal ofartificial intelligence research, 20: 61–124.

Ghallab, M.; Nau, D.; and Traverso, P. 2004. Automated Planning: theory and practice. Elsevier.

Nau, D.; Cao, Y.; Lotem, A.; and Munoz-Avila, H. 1999. SHOP: Simple hierarchical ordered planner. In Proceedings of the 16th international joint conference on Artificial intelligence-Volume 2, 968–973.

Nau, D. S.; Au, T.-C.; Ilghami, O.; Kuter, U.; Murdock, J. W.; Wu, D.; and Yaman, F. 2003. SHOP2: An HTN planning system. Journal of artificial intelligence research, 20: 379–404.

Pellier, D.; Fiorino, H.; Grand, M.; Albore, A.; and Bailon-Ruiz, R. 2022. HDDL 2.1: Towards Defining an HTN Formalism with Time. arXiv preprint arXiv:2206.01822.

Quenard, G.; Pellier, D.; and Fiorino, H. 2024. SibylSat: Using SAT as an Oracle to Perform a Greedy Search on TO-HTN Planning. In 27th European Conference on Artificial Intelligence, volume 392, 4157–4164.

Quenard, G.; Pellier, D.; and Fiorino, H. 2025. SibylSatOpt: a MaxSAT-based Greedy Optimal Search for TOHTN Planning. In Proceedings ofthe International Conference on Automated Planning and Scheduling, volume 35, 236–244.

Schreiber, D. 2021. Lilotane: A lifted SAT-based approach to hierarchical planning. Journal of artificial intelligence research, 70: 1117–1181.

Schreiber, D.; Pellier, D.; Fiorino, H.; et al. 2019. Tree-REX: SAT-based tree exploration for efficient and high-quality HTN planning. In Proceedings of the International Conference on Automated Planning and Scheduling, volume 29, 382–390.

Taitler, A.; Alford, R.; Espasa, J.; Behnke, G.; Fiser, D.;ˇ Gimelfarb, M.; Pommerening, F.; Sanner, S.; Scala, E.; Schreiber, D.; et al. 2024. The 2023 international planning competition.