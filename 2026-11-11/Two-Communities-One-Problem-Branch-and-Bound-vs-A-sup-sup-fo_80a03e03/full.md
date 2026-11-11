# Two Communities, One Problem: Branch-and-Bound vs. A<sup>∗</sup> for Resource-Constrained Project Scheduling

Izack Cohen<sup>1</sup>, Dominik Kress<sup>2</sup>, Lara Nehrke<sup>2</sup>, Erwin Pesch<sup>3</sup>, and Shimrit Shtern<sup>1</sup>

<sup>1</sup>Faculty of Engineering, Bar-Ilan University, Max and Anna Webb Street, Ramat Gan 5290002, Israel <sup>2</sup>Business Administration, especially Procurement and Production, Helmut Schmidt University/University of the Federal Armed Forces Hamburg, Friedrich-Ebert-Damm 245, D-22159 Hamburg, Germany <sup>3</sup>Management Information Science, University of Siegen, Kohlbettstr. 15, D-57072 Siegen, Germany izack.cohen@biu.ac.il, dominik.kress@hsu-hh.de, lara.nehrke@hsu-hh.de, erwin.pesch@uni-siegen.de, shimrit.shtern@biu.ac.il

## Abstract

Branch-and-Bound (B&B) and A<sup>∗</sup> are popular exact search methods originating from Operations Research and Artificial Intelligence, respectively. Although the two methods share fundamental search principles, their diferent ways of exploring the search space can lead to substantially diferent computational performance. We investigate these diferences using the Resource-Constrained Project Scheduling Problem (RCPSP) as an exemplary problem setting. We first establish the relationships between mixed-integer linear programming, state-action, and timed Petri-net representations. Since B&B and A<sup>∗</sup> admit numerous algorithmic enhancements, we focus on relatively basic implementations. This allows us to examine the fundamental diferences between the methods rather than compare the performance of state-of-the-art solvers. By using the same lower bound and best-first node selection, we isolate the efects of the search mechanisms, particularly the identification of equivalent states reached through diferent scheduling decision sequences. Experiments on 480 benchmark RCPSP instances show that graph-based A<sup>∗</sup> substantially outperforms the B&B variants considered. With additional pruning rules, A<sup>∗</sup> proves optimality for all 480 instances, compared with 445 for best-first B&B under the same time and memory limits. Depth-first B&B uses less memory and can find feasible solutions even when the search does not finish within the time limit. Stronger linear programming relaxations reduce the search space but introduce considerable computational overhead. Our findings highlight the importance of duplicate-state detection and clarify the computational trade-ofs between the search approaches. These insights can help researchers select an appropriate method and identify opportunities to incorporate efective mechanisms from one approach into the other.

Keywords: Scheduling; Project Management; Branch-and-Bound; A<sup>∗</sup>; Artificial Intelligence

## 1 Introduction

The Resource-Constrained Project Scheduling Problem (RCPSP) seeks a schedule that minimizes project completion time while honoring interdependencies among activities and limits on requested resources (see, e.g., Artigues et al., 2026, for a recent overview). The problem itself and its extensions model a common challenge in construction (see, e.g., Zhang et al., 2024), manufacturing (see, e.g., Čapek et al., 2012), maintenance (see, e.g., Chen et al., 2024), healthcare (see, e.g., Yousefzadeh and Harati Motlagh, 2026), software development (see, e.g., Li et al., 2024), and other project-based settings in which activities that could otherwise run in parallel must compete for scarce resources. Because even moderately sized instances can be dificult to solve optimally, the RCPSP is important both as a practical scheduling problem and as a benchmark for exact optimization methods.

Two mature research traditions approach this problem from diferent perspectives (details are presented in Section 2). In Operations Research (OR), exact methods commonly explore a tree of restricted subproblems or partial schedules using Branch-and-Bound (B&B). Lower bounds are used to eliminate subproblems that cannot improve the best complete schedule found so far. In Artificial

Intelligence (AI), the problem can instead be represented as a state-space graph and solved using A<sup>∗</sup>, which expands states in best-first order according to an admissible evaluation function and merges equivalent states reached through diferent decision sequences. At a suficiently abstract level, $A ^ { * }$ can be represented within general formulations of B&B (Nau et al., 1984). This theoretical relationship, however, does not explain how the concrete algorithms used in OR and AI difer when applied to the same scheduling problem. In practice, B&B in its ordinary variant (see, ordinary B&B in Nau et al., 1984) explores a tree of subsets of feasible solutions implicitly defined by partial schedules, whereas graph-based $A ^ { * }$ systematically searches a graph, detecting and merging duplicate states reached through diferent decision sequences. B&B may also use the best feasible solution found so far to prune nodes. Existing comparisons are often afected by diferences in problem representation, lower-bound strength, node-selection strategy, or algorithm-specific enhancements. Consequently, it remains unclear how much of the observed performance diference is caused by the diferent search mechanisms themselves.

This study aims to explain the fundamental computational trade-ofs between these two widely used exact-search paradigms and to provide a basis for selecting, improving, or combining them. Its purpose is therefore not to determine whether a basic implementation of either method outperforms state-of the-art RCPSP solvers. Instead, our contributions are threefold. First, we develop a common modeling foundation for the comparison by presenting mixed-integer linear programming (MILP), state-action, and timed Petri-net representations of the RCPSP and establishing the relationships between their schedules, states, and search graphs. Second, we provide a mechanistic and formal comparison of B&B and $A ^ { * }$ under identical lower-bound information and best-first node selection. The analysis identifies how duplicate states, pruning, repeated heuristic evaluations, and memory requirements afect the two searches. Finally, we conduct a controlled empirical study on benchmark RCPSP instances, comparing computational time, generated and expanded nodes, lower-bound evaluations, and memory use. The experiments examine how instance characteristics and the search mechanisms influence the relative performance of the two methods.

The remainder of the paper is organized as follows. Section 2 reviews the relevant literature and identifies the gap addressed by this study. Section 3 defines the RCPSP and introduces a running example. Section 4 presents the MILP, state-action, and timed Petri-net representations and establishes their connections. Section 5 describes the B&B and $A ^ { * }$ algorithms and compares their search mechanisms. Section 6 reports the empirical study. Finally, Section 7 concludes the paper and discusses directions for future research.

## 2 Related Literature

This section reviews three related research streams: B&B methods for the RCPSP (Section 2.1), heuristic state-space search for scheduling (Section 2.2), and theoretical studies on the relationship between B&B and $A ^ { * }$ (Section 2.3). Together, these streams position our comparison of optimization-based B&B and graph-based A<sup>∗</sup> (Section 2.4).

## 2.1 Branch-and-Bound Methods for the RCPSP

Exact RCPSP solution approaches in OR have mainly focused on MILP and specialized B&B algorithms. Early exact approaches were compared by Patterson (1984). Subsequent B&B algorithms introduced delay alternatives, schedule schemes, memory-eficient enumeration, and constraint propagation (De meulemeester and Herroelen, 1992; Brucker et al., 1998; Sprecher, 2000; Dorndorf et al., 2000). Their performance depends on the branching scheme, lower bounds, dominance rules, node-selection strategy, and the early discovery of good feasible schedules (Brucker and Knust, 2003; Coelho and Vanhoucke, 2018; Guo et al., 2023). In parallel, MILP formulations range from classical time-indexed models (Pritsker et al., 1969) to event-based formulations (Koné et al., 2011). Surveys document the continued development of these approaches (Brucker et al., 1999; Artigues et al., 2026).

In this OR interpretation, B&B is an optimization framework. Each node in the search tree represents a subset of feasible schedules, encoded either by a partial schedule or by a restricted optimization subproblem. Branching decisions partition this subset into smaller subsets. A valid lower bound, often but not necessarily obtained from a relaxation, bounds the best schedule attainable within a node, while the best complete schedule found so far provides an upper bound. A node can be discarded if it is infeasible or if its lower bound is not smaller than the upper bound. Node selection is flexible and may follow depth-first, best-first, breadth-first, or hybrid strategies. Equivalent nodes reached through diferent decision sequences are not systematically merged, although problem-specific dominance rules may remove some of this redundancy.

## 2.2 State-Space Search for Scheduling

In $A ^ { * }$ , nodes are selected according to ${ \hat { f } } = g + h$ , where $g$ is the cost accumulated so far and h is an admissible lower bound on the remaining cost (Hart et al., 1968). As originally defined, $A ^ { * }$ performs graph search: its Open and Closed structures detect when a state is regenerated along a diferent path, and only the cheapest path to each state is retained. This mechanism’s efect depends on the state representation. Ordinary B&B, in contrast, may retain the corresponding decision sequences as distinct nodes unless a dominance rule detects their equivalence.

Timed Petri nets have long been used to model scheduling problems, including job-shop systems (Hillion and Proth, 1989), manufacturing systems (Proth and Sauer, 1998; Lee and DiCesare, 1994; Mejía and Odrey, 2005), and data-driven project planning (Cohen, 2024). Scheduling through reachability graph search, particularly using $A ^ { * }$ , is a well-established approach in this literature (Lee and DiCesare, 1994; Huang et al., 2023).

Bell and Park (1990) developed an exact $A ^ { * } .$ -based algorithm for the RCPSP. Its nodes are subproblems obtained by adding precedence constraints to resolve resource conflicts, so equivalent project states reached through diferent repair sequences are never identified. Accordingly, Nazareth et al. (1999) later interpreted the method as a best-first B&B procedure formulated as an $A ^ { * }$ tree search. More recently, Lublin et al. (2026) formulated the RCPSP as an optimal search over a Petri-net reachability graph and showed that graph-based $A ^ { * }$ can compete with general-purpose MILP solvers. That comparison demonstrates the potential of $A ^ { * }$ , but it does not isolate the algorithmic diferences between $A ^ { * }$ and B&B under identical lower-bound information and search conditions.

## 2.3 Relationship between B&B and $A ^ { * }$

Earlier studies related B&B to dynamic programming and state-space representations (Morin and Marsten, 1976; Ibaraki, 1978), while Kumar and Kanal (1983) proposed a broad B&B formulation for AND/OR-tree search. Nau et al. (1984) introduced General Branch and Bound (GBB), defined through abstract representation, goal, selection, splitting, and pruning functions. These functions are deliberately general enough for both ordinary B&B and $A ^ { * }$ to be represented as special cases. We argue that this common abstraction should not be interpreted as showing that graph-based $A ^ { * }$ and ordinary optimization B&B are the same practical algorithm. In the terminology of Nau et al. (1984), ordinary B&B selects one active subproblem at a time and prunes using a lower bound and an upper bound which is given by the best solution found so far. $A ^ { * }$ is embedded in GBB through a diferent choice of representations, selection rules, and pruning functions. Similarly, Labat and Pomerol (2003) concluded that heuristic search and B&B apply the same principles and difer mainly at the interpretation level within an abstract view of B&B. Thus, the AI literature compares the methods within a highly general search abstraction, rather than comparing the concrete optimization-based B&B used in RCPSP research with graph-based $A ^ { * }$

## 2.4 Research Gap

Taken together, the existing literature does not provide a controlled, practical comparison of the two introduced search paradigms for the RCPSP. An admissible $A ^ { * }$ heuristic and a B&B lower bound serve the same mathematical purpose and may be computed from the same relaxation. Therefore, diferences in bound strength should be controlled when comparing the methods. To the best of our knowledge, no study compares ordinary $\mathrm { O R - s t y }$ le B&B and graph-based $A ^ { * }$ on identical RCPSP instances while holding the lower-bound function and node-selection conditions fixed. This paper addresses that gap and isolates the consequences of tree-based enumeration and upper-bound pruning in B&B versus systematic duplicate-state detection in graph-based $A ^ { * }$

## 3 The Resource-Constrained Project Scheduling Problem

A project consists of activities that must be executed subject to precedence relations and limited resources. In the RCPSP, the goal is to choose activity start times that minimize the makespan while satisfying all precedence and resource constraints. This section gives the formal problem definition (Section 3.1) and introduces the running example (Section 3.2) used throughout the paper.

## 3.1 Formal Definition

Let V represent the set of non-preemptive activities, with $d _ { i } \in \mathbb { N } _ { 0 } = \{ 0 , 1 , 2 , . . . \}$ denoting the duration of activity $i ~ \in ~ \nu$ . Let R be the set of renewable resources, with capacity $R _ { k }$ for each $k \in \mathcal { R }$ Additionally, let $\mathcal { E } \subset \mathcal { V } \times \mathcal { V }$ be the set of all direct precedence relations, meaning that $( i , j ) \in \mathcal { E }$ if the completion of activity i is a direct prerequisite for starting activity $j$ . For each activity $i \in \mathcal V$ , we define $\operatorname { P r e d } ( i ) = \{ j \in \mathcal { V } : ( j , i ) \in \mathcal { E } \} , \operatorname { S u c } ( i ) = \{ j \in \mathcal { V } : ( i , j ) \in \mathcal { E } \}$ . The elements of Pred(i) and $\operatorname { S u c } ( i )$ are the immediate predecessors and immediate successors of i, respectively.

Without loss of generality, assume that the unique starting point (no predecessors) of the project is at activity 0 and the unique ending point (no successors) of the project is activity $n ,$ with $0 , n \in \mathcal { V }$ and $d _ { 0 } = d _ { n } = 0$ . We refer to activities 0 and n as dummy activities. For any other activity i, we assume $d _ { i } > 0$ . The resource demand of activity $i \in \mathcal V$ for resource $k \in \mathcal { R }$ is denoted by $r _ { i k }$ , where $r _ { 0 k } = r _ { n k } = 0$ for all $k \in \mathcal { R }$ , and the start time of activity $i \in \mathcal V$ is denoted by $S _ { i }$ . We assume that $( \mathcal V , \mathcal E )$ is acyclic and that $0 \leq r _ { i k } \leq R _ { k }$ for all $i \in \mathcal V$ and $k \in \mathcal { R }$ . For each time $t \in  { \mathbb { N } } _ { 0 }$ , let $\mathbf { \mathcal { A } } ( t ) \subseteq \mathcal { V }$ denote the set of activities in process at time t, so that $i \in \mathcal { A } ( t )$ if $S _ { i } \leq t < S _ { i } + d _ { i }$ . A schedule $S = \{ S _ { i } \} _ { i \in \mathcal { V } }$ is an assignment of start times to all activities. A schedule is feasible if it satisfies all precedence constraints, and at each point in time the resource demand of the processed activities does not exceed the amount of available resources. A feasible schedule is optimal if its makespan, given by $S _ { n } .$ , is minimal among all feasible schedules.

Equivalently, the RCPSP can be stated as the following optimization problem:

$$
\begin{array} { l l } { \underset { S } { \operatorname* { m i n } } } & { S _ { n } } \\ { \mathrm { ~ s . t . ~ } } & { S _ { i } + d _ { i } \leq S _ { j } , } \\ & { \displaystyle \sum _ { i \in A ( t ) } r _ { i k } \leq R _ { k } , } \\ & { S _ { i } \in \mathbb { N } _ { 0 } , } \end{array}\tag{1a}
$$

$$
\forall ( i , j ) \in \mathcal { E } ,
$$

$$
\forall k \in \mathcal { R } , \ t \in \mathbb { N } _ { 0 } ,\tag{1b}
$$

(1c)

$$
\forall i \in \mathcal { V } .\tag{1d}
$$

A feasible schedule S is called active if there is no feasible schedule $S ^ { \prime } \ne S$ that starts every activity no later than in $S$ and starts at least one activity strictly earlier; that is, there does not exist a feasible schedule $S ^ { \prime }$ and an activity $j \in \mathcal V$ such that $S _ { i } ^ { \prime } \leq S _ { i } , \forall i \in \mathcal { V }$ , and $S _ { j } ^ { \prime } < S _ { j }$ . Equivalently, $S$ cannot be improved by moving one or more activities earlier without delaying any other activity or violating a precedence or resource constraint. Since at least one optimal schedule is active (Kolisch, 1996), focusing on earliest-start schedule generation preserves optimality. In the search formulations below, once an activity is selected, it is started at the earliest feasible time relative to the current state, subject to precedence and resource feasibility. Hence, the resulting state space contains at least one path corresponding to an optimal active schedule.

We use the term partial schedule to refer to a feasible schedule restricted to a subset of already scheduled activities. Formally, a partial schedule is a pair $( \mathcal { I } , S _ { \mathcal { I } } )$ , where $\mathcal { I } \subseteq \mathcal { V }$ is predecessor-closed, i.e., if $j \in \mathcal I$ and $( i , j ) \in \mathcal { E }$ , then $i \in \mathcal { I }$ , and ${ \cal S } _ { \mathcal { T } } = \{ { \cal S } _ { j } \} _ { j \in \mathcal { T } }$ assigns start times to all activities in $\mathcal { I }$ The assignment must satisfy all precedence and resource constraints induced by the activities in $\mathcal { I }$ Each partial schedule induces a timed system state that records the current time, which is the start time of the latest scheduled activity, the completed and ongoing activities, and the residual durations of the ongoing activities. This state contains all information required to determine the feasible future scheduling decisions. Consequently, any two partial schedules that induce the same timed state define the same remaining scheduling problem. Our B&B formulation searches the tree of partial schedules, whereas graph-based $A ^ { * }$ searches the corresponding state graph obtained by merging partial schedules that induce the same timed state. Thus, both methods represent the same underlying scheduling decisions but difer in whether equivalent scheduling histories are retained as separate nodes or merged into a single state.

![](images/853fc44434e8e55cfd1e3027cbece596f6d768ceaadea5d9dce04e38fecae8d5.jpg)  
Figure 1: Illustration of the running RCPSP example.  
Note: Each node shows the activity identifier, duration d, and demand r for the single renewable resource with capacity $R _ { 1 } = 2 { \it \Delta }$

The RCPSP is NP-hard in the strong sense (Blazewicz et al., 1983), which necessitates the use of efective bounding and pruning strategies within exact methods.

## 3.2 Running Example

Throughout this paper, we use a small RCPSP instance to illustrate the concepts and methods. The instance contains the dummy start and end activities 0 and n, respectively, and five non-dummy activities $a , b , c , d , e$ . It is illustrated in Figure 1.

The project comprises the seven activities $\mathcal { V } = \{ 0 , a , b , c , d , e , n \}$ . Activities 0 and n have zero duration. The durations of the non-dummy activities are $d _ { a } = 3 , d _ { b } = 2 , d _ { c } = 3 , d _ { d } = 1$ , and $d _ { e } = 1$ . The precedence relations are completely captured by $\mathrm { S u c } ( 0 ) = \{ a , b , c \} , \mathrm { P r e d } ( d ) = \{ a , b \}$ $\mathrm { P r e d } ( e ) = \{ b , c \}$ , and $\mathrm { P r e d } ( n ) = \{ d , e \}$ . Equivalently, using informal arrow notation, they can be written as $0  \{ a , b , c \} , \{ a , b \}  d , \{ b , c \}  e .$ , and $\{ d , e \}  n$ . A single renewable resource with capacity $R _ { 1 } = 2$ is available. The resource demands are $r _ { a } = 1 , r _ { b } = 2 , r _ { c } = 1 , r _ { d } = 1$ , and $r _ { e } = 1$ while the dummy activities 0 and n require no resources. Since the example contains a single resource, we omit the resource index; for example, $r _ { a 1 }$ is written as $r _ { a }$ . The example is small enough to be solved by hand, yet rich enough to exhibit key diferences between B&B and $A ^ { * }$ search. Concurrent activities create multiple equivalent execution orderings, while the resource constraint prevents full parallelism and forces nontrivial scheduling decisions.

## 4 Problem Representations

The RCPSP can be modeled from several computational perspectives. A MILP formulation (Section 4.1) represents complete schedules through decision variables and supports ordinary OR-style B&B. A state-action formulation (Section 4.2) represents scheduling as transitions between timed system states and supports graph-search methods such as $A ^ { * }$ . A Petri-net formulation (Section 4.3) provides an explicit token-based model of precedence, timing, and renewable-resource availability, whose timed reachability graph is equivalent to the state graph. This section introduces the three formulations and formalizes their relationships (Section 4.4).

## 4.1 Mixed-Integer Linear Programming Formulation

The RCPSP can be formulated as a MILP by introducing binary decision variables that encode the scheduling decisions. Several MILP formulations exist in the literature; here we present a standard time-indexed formulation.

Let H denote an upper bound on the project duration, usually obtained by scheduling all activities in sequence so $\textstyle H = \sum _ { i \in \mathcal { V } } d _ { i }$ , and let $\mathcal { T } = \{ 0 , 1 , \ldots , H \}$ denote the corresponding discrete time horizon. Define binary variables

$$
y _ { i t } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ a c t i v i t y ~ } } i { \mathrm { ~ s t a r t s ~ a t ~ t i m e ~ } } t , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{2}
$$

for all $i \in \mathcal V$ and $t \in \mathcal T$ . Then, the MILP formulation is as follows.

min $S _ { n }$

$$
H { - } d _ { i }\tag{3a}
$$

$$
\mathrm { s . t . } \sum _ { t = 0 } ^ { } y _ { i t } = 1
$$

$$
\forall i \in \mathcal { V }\tag{3b}
$$

$$
S _ { i } = \sum _ { t = 0 } ^ { } { \stackrel { . } { t } } y _ { i t }
$$

$$
\forall i \in \mathcal { V }\tag{3c}
$$

$$
S _ { i } + d _ { i } \leq S _ { j }
$$

$$
\forall ( i , j ) \in \mathcal { E }\tag{3d}
$$

$$
\operatorname* { m i n } \{ t , H - d _ { i } \}
$$

$$
\sum _ { i \in \mathcal { V } } r _ { i k } \sum _ { \tau = \operatorname* { m a x } \{ 0 , t - d _ { i } + 1 \} } y _ { i \tau } \leq R _ { k }
$$

$$
\forall t \in T , k \in \mathcal { R }\tag{3e}
$$

$$
S _ { i } \in  { \mathbb { N } } _ { 0 }
$$

$$
\forall i \in \mathcal { V }\tag{3f}
$$

$$
y _ { i t } \in \{ 0 , 1 \}
$$

$$
\forall i \in \mathcal { V } , \ t = 0 , \ldots , H - d _ { i }\tag{3g}
$$

The objective $\mathrm { ( 3 a ) }$ minimizes the makespan, represented by the start time of the dummy activity $n , S _ { n }$ . Constraints (3b) ensure that each activity starts exactly once, while constraints $( \mathrm { { 3 c } ) }$ define the corresponding start times $S _ { i } .$ . Constraints (3d) enforce the precedence relations. For a fixed activity i and time period t, activity i is ongoing during time period t if and only if it started at some time τ with $t - d _ { i } + 1 \leq \tau \leq t$ . Since start variables are defined only for feasible start times $\tau \in \{ 0 , \ldots , H - d _ { i } \}$ , the inner summation in (3e) ranges over the intersection of these two intervals, namely $\boldsymbol { \tau } = \operatorname* { m a x } \{ 0 , t - d _ { i } + 1 \} , \ldots , \operatorname* { m i n } \{ t , H - d _ { i } \}$ . Finally, constraints (3f) and $( \mathrm { 3 g } )$ impose integrality on the start-time and binary decision variables.

## 4.2 State-Action Representation

An alternative representation views the RCPSP as a shortest-path problem in a state space, where each state encodes the current execution status of the project and transitions (actions) correspond to scheduling decisions. More specifically, a state s captures the project status at a particular point in time. It records:

• The current system time $g ( s ) \in  { \mathbb { N } } _ { 0 }$ , i.e., the time represented by state s at which the system is observed and the next scheduling decision is considered.

• The set of completed activities $\mathcal { V } ^ { c } ( s ) \subseteq \mathcal { V }$ , i.e., activities that have finished no later than $g ( s )$

• The set of ongoing activities $\mathcal { V } ^ { o } ( s ) \subseteq \mathcal { V }$ that have started but have not yet completed at $g ( s )$ together with their residual durations $\tilde { d } _ { i } ( s ) \in  { \mathbb { N } } _ { 0 }$ relative to $g ( s )$ for all $i \in \mathcal { V } ^ { o } ( s )$ , representing the remaining processing time at the current system time $g ( s )$

Although $\mathcal { V } ^ { o } ( s )$ is a set, in the state tuple we represent its elements as an ordered tuple, with activities listed in a fixed order and their corresponding residual durations in $\tilde { d } ( s )$ listed in the same order. Activities in $\mathcal { V } ^ { u } ( s ) = \mathcal { V } \setminus ( \mathcal { V } ^ { c } ( s ) \cup \mathcal { V } ^ { o } ( s ) )$ are called not-yet-started activities.

We define an initial state s with $\mathcal V ^ { c } ( s _ { 0 } ) = \emptyset , \mathcal V ^ { o } ( s _ { 0 } ) = \emptyset , g ( s _ { 0 } ) = 0$ , and full resource availability. Furthermore, we define a set of goal states $ { \boldsymbol { S } } _ { f }$ as the set of all states s with $\mathcal { V } ^ { c } ( s ) = \mathcal { V }$ . The available resource profile at time $g ( s )$ can be deduced from resources consumed by ongoing activities and their capacities. A state s is called feasible if it satisfies the resource constraints $\begin{array} { r } { \sum _ { i \in \mathcal { V } ^ { o } ( s ) } r _ { i k } \le R _ { k } } \end{array}$ for all $k \in \mathcal { R }$ , and the precedence constraints, i.e., if $j \in \mathcal { V } ^ { o } ( s )$ and $( i , j ) \in \mathcal { E }$ , then $i \in \mathcal { V } ^ { c } ( s )$

At each state $s ,$ we can select an activity to be scheduled next, which leads to a new state $s ^ { \prime } .$ The set of selectable activities at state s, denoted by $\gamma ^ { a } ( s )$ , consists of the not-yet-started activities whose immediate predecessors have either completed or are currently ongoing: $\gamma ^ { a } ( s ) =$ $\{ i \in \mathcal { V } ^ { u } ( s ) : \mathrm { P r e d } ( i ) \subseteq \mathcal { V } ^ { c } ( s ) \cup \mathcal { V } ^ { o } ( s ) \}$ . Thus, $\mathcal { V } ^ { a } ( s ) \subseteq \mathcal { V } ^ { u } ( s )$ . If some predecessors of the selected activity are still ongoing, or if insuficient resources are currently available, the system time advances to the earliest time at which the selected activity can start. This earliest-start interpretation is used in the remainder of the paper. Hence, for a given activity $j \in \mathcal { V } ^ { a } ( s )$ , we get:

• Time transition:

$$
g ( s ^ { \prime } ) = \operatorname* { m i n } \left\{ t : \begin{array} { l l } & { t \geq g ( s ) , } \\ { t : \ t \geq g ( s ) + \tilde { d _ { i } } ( s ) } & { \forall i \in \mathcal { V } ^ { o } ( s ) \ \mathrm { w i t h } \ ( i , j ) \in \mathcal { E } , } \\ & { r _ { j k } \leq R _ { k } - \displaystyle \sum _ { i \in \mathcal { V } ^ { o } ( s ) : g ( s ) + \tilde { d _ { i } } ( s ) > t } r _ { i k } , \quad \forall k \in \mathcal { R } } \end{array} \right\} ,\tag{4}
$$

with $\Delta ( s , s ^ { \prime } ) = g ( s ^ { \prime } ) - g ( s )$ denoting the elapsed time in the transition.

• Ongoing activities:

$$
\begin{array} { r } { \mathcal { V } ^ { o } ( s ^ { \prime } ) = \{ i \in \mathcal { V } ^ { o } ( s ) : \Delta ( s , s ^ { \prime } ) < \tilde { d } _ { i } ( s ) \} \cup \left\{ \{ j \} , \mathrm { i f } \ d _ { j } > 0 , \right. } \end{array}\tag{5}
$$

Thus, an activity is ongoing in $s ^ { \prime }$ when it has positive residual duration after the transition. Residual durations of previously ongoing activities are reduced by the elapsed time, while the newly started activity j receives its full duration if it is positive. Hence, for all $i \in \mathcal { V } ^ { o } ( s ^ { \prime } ) \setminus \{ j \}$ we have $\tilde { d } _ { i } ( s ^ { \prime } ) = \tilde { d } _ { i } ( s ) - \Delta ( s , s ^ { \prime } )$ , and, if $d _ { j } > 0 , \tilde { d } _ { j } ( s ^ { \prime } ) = d _ { j }$

• Completed activities:

$$
\mathcal V ^ { c } ( s ^ { \prime } ) = \mathcal V ^ { c } ( s ) \cup \{ i \in \mathcal V ^ { o } ( s ) : \Delta ( s , s ^ { \prime } ) \geq \tilde { d } _ { i } ( s ) \} \cup \left\{ \{ j \} , \quad \mathrm { i f ~ } d _ { j } = 0 , \right.\tag{6}
$$

The reachable state space can be represented as a directed graph $G = ( S , \mathbb { E } )$ , which we call the state graph, where $s$ is the set of states reachable from $s _ { 0 }$ by feasible scheduling decisions, and E is the set of directed edges induced by these decisions. We write $s \stackrel { j } { \longrightarrow } s ^ { \prime }$ if choosing activity j in state s leads to the successor state $s ^ { \prime }$ under the above transition rules. Finding an optimal schedule is equivalent to finding a path from the initial state to a goal state that minimizes the system time of the goal state. Thus, the cost of a path is measured by the system time at its final state. This equivalence follows because an optimal schedule can be represented by ordering its activities according to their start times and applying the earliest-start transition rule along that order.

To illustrate this notation, consider the running example. We start from state s<sub>0</sub>. Since $\mathcal { V } ^ { a } ( s _ { 0 } ) = \{ 0 \}$ scheduling dummy activity 0 leads to $s _ { 1 } = ( g ( s _ { 1 } ) , \mathcal { V } ^ { c } ( s _ { 1 } ) , \mathcal { V } ^ { o } ( s _ { 1 } ) , \tilde { d } ( s _ { 1 } ) ) = ( 0 , \{ 0 \} , ( ) , ( ) )$ . At this state, the set of selectable activities is $\mathcal { V } ^ { a } ( s _ { 1 } ) = \{ a , b , c \}$ . If activity a is selected next, the search moves to $s _ { 2 } = ( 0 , \{ 0 \} , ( a ) , ( 3 ) )$ , and the selectable activities become $\mathcal { V } ^ { a } ( s _ { 2 } ) = \{ b , c \}$ . If activity b is then selected, the transition advances time to state $s _ { 5 }$ . Activity b requires two resource units, one of which is currently used by activity a, and therefore cannot start until a releases that unit. Consequently, $g ( s _ { 5 } ) = 3$ and $s _ { 5 } = ( 3 , \{ 0 , a \} , ( b ) , ( 2 ) )$ ). In contrast, if activity c is selected after activity a, then a and c can be processed in parallel because suficient resource capacity is available. The resulting state is $s _ { 6 } = ( 0 , \{ 0 \} , ( a , c ) , ( 3 , 3 ) )$ ). The same state $s _ { 6 }$ is reached by scheduling c before $^ { a , }$ illustrating how diferent scheduling decision sequences can lead to the same state. Figure 2 illustrates the subgraph of G induced by these scheduling decisions.

## 4.3 Petri Net Representation

A Petri net can be viewed as a directed bipartite graph with two types of nodes: places and transitions. Petri nets provide a formal mathematical framework for modeling concurrent systems, capturing synchronization, precedence relations, and shared resources (Murata, 1989). We use them to model concurrent project activities and resource availability, and to establish a formal correspondence between their timed reachability graph and the state graph used by $A ^ { * }$ . Places represent conditions, intermediate project states, or available resources. Transitions represent activities or events. Directed arcs connect only places to transitions or transitions to places. Therefore, there are no direct arcs between two places or between two transitions. A token is a marker residing in a place, and the distribution of tokens over the places represents the current state of the system. The movement of tokens through the net naturally represents synchronization and parallelism among activities. Petri nets can also represent XOR choices between alternative courses of action. The RCPSP considered here, however, contains only AND-type precedence relations. An activity may start only after all of its predecessors have completed. Accordingly, we do not use XOR constructs in the Petri-net representation. In our comparison, the Petri-net representation provides an established modeling framework for state-based search, rather than a requirement for applying $A ^ { * }$

![](images/ebedd17258f161384b7e48ba27580138c51b290422c4c99bc699d951aa69d4c3.jpg)  
Figure 2: Illustrative subgraph of the RCPSP state graph for the first scheduling decisions. Note: Each node is a state represented as $( g ( s ) , \mathcal { V } ^ { c } ( s ) , \mathcal { V } ^ { o } ( \underline { { s } } ) , \tilde { d } ( s ) )$ , where $g ( s )$ is the current time, $\mathcal { V } ^ { c }$ is the set of completed activities, $\mathcal { V } ^ { o }$ is the ordered tuple of ongoing activities, and d<sup>˜</sup> is the corresponding ordered tuple of residual processing times.

Petri nets support correctness checks such as whether all activities of a project are executable, the project can always progress to completion, and no residual work remains in the final state. They thus provide verification capabilities that go well beyond what alternative representations are capable of (Murata, 1989; van der Aalst and Carmona, 2022). Moreover, Petri nets can be used to model and visualize resource constraints.

## 4.3.1 Timed Transition Petri Nets with Resources

We model the RCPSP as a so-called Timed Transition Petri Net with Resources $( T T P N R )$ , i.e., a tuple $T T P N R = ( P , T , F , W , M _ { 0 } , L , \tau , \mathcal { R } )$ , where, in addition to the definitions in Section 3.1, we define:

$P = P _ { C } \cup P _ { R }$ is the finite set of places, depicted as circles, where $P _ { C } \cap P _ { R } = \emptyset$ . The set $P _ { C }$ contains the control-flow places $\{ p 1 , p 2 , \dots , p _ { m } \}$ , which represent the progress of the project. The set $P _ { R } = \{ p _ { r _ { k } } : k \in \mathcal { R } \}$ contains one resource place for each renewable resource. Tokens in resource places represent available resource units.

$T = \{ t _ { i } : i \in \mathcal { V } \}$ is the finite set of transitions, depicted as rectangles, with $P \cap T = \emptyset$ . Transition $t _ { i }$ represents activity i. In particular, the dummy start and end activities are represented by zero-duration transitions $t _ { 0 }$ and $t _ { n } ,$ respectively.

$F \subseteq ( P \times T ) \cup ( T \times P )$ is the set of directed arcs, also called the flow relation. We distinguish between control-flow arcs and resource arcs. Control-flow arcs connect control-flow places and transitions and encode the precedence/progress structure of the project. Resource arcs connect resource places and transitions and model the use of renewable resource units. For the RCPSP, the control-flow structure consists of a source place $p _ { \mathrm { s t a r t } }$ connected to $t _ { 0 }$ , a sink place $p _ { \mathrm { e n d } }$ connected from $t _ { n }$ , and one place $p _ { i j }$ connecting t<sub>i</sub> to $t _ { j }$ for each $( i , j ) \in \mathcal { E }$ . These are the only control-flow arcs, and initially only $p _ { \mathrm { s t a r t } }$ contains a token. In the running example (Figures 3 and 4), these places are denoted by $p _ { 1 } , \ldots , p _ { 1 1 }$

$W : F \to \mathbb { Z } ^ { + }$ is a weight function that assigns a positive integer weight to each arc. For an input arc $( p , t )$ , the value $W ( p , t )$ is the number of tokens that must be available in place $p$ for transition t to start. For an output arc $( t , p )$ , the value $W ( t , p )$ is the number of tokens deposited in place $p$ after transition t completes. In our RCPSP models, control-flow arcs have weight one. Resource arcs have weight equal to the number of resource units required by the corresponding activity, and the same number of resource tokens is returned to the resource place when the activity completes.

• A marking $M : P \to \mathbb { N } _ { 0 }$ specifies the number of tokens in each place and thereby encodes the current system state. Specifically, $M _ { 0 } : P \to \mathbb { N } _ { 0 }$ is the initial marking. For control-flow places,

![](images/d1f33d9441ff7320a87ac029b0718c2dc1b1a55c0743a99e599d9680c045b577.jpg)  
Figure 3: Timed transition Petri net with resources for the running RCPSP example.  
Note: Black solid arcs encode control-flow structure and dashed gray arcs encode consumption and release of the renewable resource. The resource place $p _ { r _ { 1 } }$ initially contains two tokens, corresponding to capacity $R _ { 1 } = 2$ . Each transition is labeled by its transition name and duration τ.

$M _ { 0 }$ encodes the initial project state. For each resource place $p _ { r _ { k } } \in P _ { R } , M _ { 0 } ( p _ { r _ { k } } ) = R _ { k }$ specifies the initial number of available units of resource k. In the timed state representation introduced in Section 4.3.2, this ordinary marking is refined by associating a residual availability time with each token, defined as the number of time units, measured from the current system time, until the token becomes available. The underlying distribution of tokens over places remains the marking defined here.

$L : T  \mathcal { V }$ is a labeling function that assigns each transition its corresponding project activity. Since transition $t _ { i }$ represents activity i, we have $L ( t _ { i } ) = i$ for all $i \in \mathcal V$

$\tau : T  \mathbb { N } _ { 0 }$ is a timing function that specifies the duration of each transition. Since transition $t _ { i }$ represents activity i, we set $\tau ( t _ { i } ) = d _ { i }$ for all $i \in \mathcal V$ . Starting, or firing, a transition means consuming the required input tokens and committing to the corresponding activity. When transition $t _ { i }$ fires, the required input tokens are consumed, and its output control-flow tokens and returned resource tokens are associated with their corresponding places with residual availability time $\tau ( t _ { i } )$ . These tokens can enable subsequent transitions only when their residual availability time reaches zero.

When no resources are present $( \mathcal { R } = \emptyset , P _ { R } = \emptyset )$ and all durations are zero $( \tau ( t ) = 0$ for all $t )$ , the $T T P N R$ reduces to a standard Petri Net. When resources are absent but timing is present, it reduces to a Timed Transition Petri Net.

For compactness, we often represent markings by listing only places with positive token counts. Thus, $\left[ p _ { 1 } \right]$ denotes the marking M with $M ( p _ { 1 } ) = 1$ and $M ( p ) = 0$ for all $p \neq p _ { 1 }$ . More generally, $p ^ { q }$ denotes q tokens in place $p ,$ while the exponent is omitted when $q = 1$ . For example, $[ p _ { 1 } , p _ { r _ { 1 } } ^ { 2 } ]$ denotes a marking with one token in the control-flow place $p _ { 1 }$ , two tokens in the resource place ${ { p } _ { { { r } _ { 1 } } } }$ , and zero tokens in all other places.

A place p is an input place of transition t if $( p , t ) \in F$ , and an output place if $( t , p ) \in F$ . We write $\bullet t = \{ p \in P : ( p , t ) \in F \}$ and $t \bullet = \{ p \in P : ( t , p ) \in F \}$ for the sets of input and output places of $t ,$ respectively. Similarly, •p and $p \bullet$ denote the input and output transitions of place $p .$

A transition $t \in T$ is enabled in marking M if $M ( p ) \geq W ( p , t )$ for all $p \in \bullet t .$ . This condition requires that every input place, including both control-flow and resource places, contains suficiently many tokens.

When t fires, it consumes tokens from its input places and produces tokens for its output places. The corresponding structural marking update $M ^ { \prime }$ is given by:

$$
M ^ { \prime } ( p ) = M ( p ) - \{ W ( p , t ) , \quad ( p , t ) \in F , + \{ W ( t , p ) , \quad ( t , p ) \in F ,  \qquad \forall p \in P .\tag{7}
$$

Equation (7) describes the structural change in the distribution of tokens over places. In the timed interpretation used in this paper, however, the presence of a token in a place does not necessarily imply that it is immediately available. The availability of produced control-flow tokens and returned resource tokens depends on the duration $\tau ( t )$ . This timing information is incorporated into the state representation introduced in Section 4.3.2.

In Figure 3, we instantiate the $T T P N R$ for the running RCPSP example with $P = \{ p _ { 1 } , \dots , p _ { 1 1 } , p _ { r _ { 1 } } \}$ and $T = \{ t _ { 0 } , t _ { a } , t _ { b } , t _ { c } , t _ { d } , t _ { e } , t _ { n } \}$ . The black solid arcs encode the control-flow structure, $\mathrm { i . e . }$ , the precedence relations between activities, while the dashed gray arcs encode consumption and release of the single renewable resource with capacity $R _ { 1 } = 2$ . The notation directly follows the general activityindexed transition notation $t _ { i } , i \in \mathcal { V }$ . Thus, $t _ { 0 }$ and $t _ { n }$ represent the dummy start and end activities, respectively, while $t _ { a } , \ldots , t _ { e }$ represent the five non-dummy activities of the running example. For every activity $i \in \{ 0 , a , b , c , d , e , n \}$ , the labeling and timing functions satisfy $L ( t _ { i } ) = i$ and $\tau ( t _ { i } ) = d _ { i }$ . The control-flow places encode the precedence structure of the running example. For instance, transition $t _ { d } ,$ , corresponding to activity $d ,$ requires tokens in both $p _ { 5 }$ and $p _ { 6 }$ , thereby enforcing that activities a and b must complete before d can start. Similarly, transition $t _ { e } ,$ corresponding to activity $e ,$ requires tokens from the output places of activities b and $c .$ . The resource place $p _ { r _ { 1 } }$ initially contains two tokens, represented by the initial marking $M _ { 0 } = [ p _ { 1 } , p _ { r _ { 1 } } ^ { 2 } ]$ . Resource arcs connect $p _ { r _ { 1 } }$ to the non-dummy activity transitions ${ t _ { a } , t _ { b } , t _ { c } , t _ { d } , t _ { e } }$ . Their weights satisfy $W ( p _ { r _ { 1 } } , t _ { b } ) = W ( t _ { b } , p _ { r _ { 1 } } ) = 2$ , reflecting that activity b requires two resource units, whereas each of the other non-dummy activities requires one resource unit.

## 4.3.2 State Representation and Search Graph Construction

The dynamic behavior of a TTPNR can be represented by a reachability graph $R G = ( S _ { R G } , \mathbb { E } _ { R G } )$ The node set $\cal { S } _ { R G }$ consists of reachable timed states. Each node $s \in S _ { R G }$ is represented by a current system time $g ( s )$ and a timed marking $M _ { s }$ . The edge set $\mathbb { E } _ { R G }$ contains directed edges induced by transition firings, where each edge corresponds to firing one transition, i.e., to one scheduling decision. A sequence of transition firings is called a trace and represents one possible project realization sequence. If several transitions can be selected, each such transition defines a diferent successor edge.

The timed marking $M _ { s }$ refines the ordinary marking $M : P  \mathbb { N } _ { 0 }$ defined in Section 4.3.1. It preserves the distribution of tokens over places while associating a residual availability time with each token. Formally, suppose that place p contains q tokens in the ordinary marking underlying state s. Let $\nu _ { p }$ denote the number of distinct residual availability times of these tokens, $\theta _ { p , 1 } , \ldots , \theta _ { p , \nu _ { p } }$ denote these times, and $q _ { p , \ell }$ denote the number of tokens with residual availability time $\theta _ { p , \ell }$ . The timed marking at $p$ is represented as

$$
M _ { s } ( p ) = \left\{ q _ { p , 1 } ^ { \theta _ { p , 1 } } , \dots , q _ { p , \nu _ { p } } ^ { \theta _ { p , \nu _ { p } } } \right\} , \qquad \sum _ { \ell = 1 } ^ { \nu _ { p } } q _ { p , \ell } = q .
$$

Here, $q _ { p , \ell } ^ { \theta _ { p , \ell } }$ denotes q tokens that become available after ${ q _ { p , \ell } }$ $\theta _ { p , \ell }$ time units. If $q = 0$ , then $M _ { s } ( p ) = \emptyset$ . At the initial state $s _ { 0 }$ , every token is immediately available. Thus, if $M _ { 0 } ( p ) = q > 0$ , then $M _ { s _ { 0 } } ( p ) = \{ q ^ { 0 } \}$

For compactness, we list only places containing at least one token. Tokens having the same residual availability time are grouped, and a superscript equal to zero is omitted. Thus, $p : 2 ^ { \theta }$ denotes two tokens in place $p$ that both become available after θ time units, whereas $p : 1 , 1 ^ { 3 }$ denotes one immediately available token and one token that becomes available after three time units. In particular, $M _ { s _ { 0 } } ( p ) = \{ q ^ { 0 } \}$ is written as $M _ { s _ { 0 } } ( p ) = \{ q \}$ . For example, $M _ { s } = [ p _ { 1 } : 1 , ~ p _ { r _ { 1 } } : 1 , 1 ^ { 3 } ]$ represents one immediately available token in place $p _ { 1 }$ and two tokens in resource place $p _ { r _ { 1 } }$ , one immediately available and one available after three time units.

Let us illustrate the notation using the TTPNR in Figure 3. At the initial node $s _ { 0 }$ , the timed marking is $M _ { s _ { 0 } } = [ p _ { 1 } : 1 , p _ { r _ { 1 } } : 2 ]$ , and the current system time is $g ( s _ { 0 } ) = 0$ . This marking indicates that there is one immediately available control-flow token in place $p _ { 1 }$ and two immediately available resource tokens in place $p _ { r _ { 1 } }$ . After $t _ { 0 }$ is fired, the system time remains 0, since $\tau ( t _ { 0 } ) = 0$ , and the timed marking becomes $M _ { s _ { 1 } } = [ p _ { 2 } : 1 , ~ p _ { 3 } : 1 , ~ p _ { 4 } : 1 , ~ p _ { r _ { 1 } } : 2 ]$ . The input places of activities $a , b ,$ and $c$ contain the required control-flow tokens, and the resource place contains two immediately available resource tokens. Hence, each of $t _ { a } , t _ { b } .$ and $t _ { c }$ can be selected as the next scheduling decision. Suppose, for example, that $t _ { a }$ is fired from $s _ { 1 }$ . This produces state $s _ { 2 }$ with timed marking $M _ { s _ { 2 } } = [ p _ { 3 } : 1 , p _ { 4 } : 1 , p _ { 5 } : 1 ^ { 3 } , p _ { r _ { 1 } } : 1 , 1 ^ { 3 } ]$ Transition $t _ { a }$ consumes the token in $p _ { 2 }$ and one resource token. A control-flow token becomes available in $p _ { 5 }$ after $\tau ( t _ { a } ) = 3$ time units, and the consumed resource token returns to ${ { p } _ { { { r } _ { 1 } } } }$ after the same delay. Although the control-flow token required by activity b is immediately available in $p _ { 3 }$ , only one resource token is immediately available. Therefore, selecting b next advances the system time to time 3, when the resource token used by activity a returns and two resource tokens are again available. Alternatively, if transition $t _ { b }$ is fired immediately after $t _ { 0 }$ , the resulting state $s _ { 3 }$ has timed marking

![](images/ff67d5db4a0c976e10c7b2b59a1edb11e1d5bc4306aef38906e9e2b699d03e7e.jpg)  
Figure 4: Partial timed reachability graph of the T T P NR for the running example.  
Note: Each state s is accompanied by its timed marking $M _ { s }$ and the tuple $( g ( s ) , h ( s ) , { \hat { f } } ( s ) )$ , where g(s) is the current system time, $h ( s )$ is an admissible lower bound on the remaining time to project completion, and ${ \hat { f } } ( s ) = g ( s ) + h ( s )$ is the resulting lower bound on the final project duration used by $A ^ { * }$ . Each directed edge is labeled by the activity whose transition is fired. Gray states are generated and expanded by $A ^ { * }$ , whereas white states are generated but not expanded. The double-bordered state s<sub>14</sub> is a goal state, in which all activities have completed. The dashed edges indicate transitions to previously generated states, illustrating duplicate-state detection. State indices reflect the generation order in the illustrative $A ^ { * }$ run. For Section 4.3.2, the relevant elements are the timed markings and transition firings. The A<sup>∗</sup> evaluation and expansion annotations are discussed in Section 5.2.

$M _ { s _ { 3 } } = [ p _ { 2 } : 1 , ~ p _ { 4 } : 1 , ~ p _ { 6 } : 1 ^ { 2 } , ~ p _ { 7 } : 1 ^ { 2 } , ~ p _ { r _ { 1 } } : 2 ^ { 2 } ]$ . Activity b is fully enabled at time 0 and has duration $\tau ( t _ { b } ) = 2$ . Its output control-flow tokens and the two resource tokens it consumes therefore become available after two time units.

Figure 4 illustrates the timed markings associated with the states of the reachability graph and the way these markings evolve under transition firings. The figure also contains the evaluation annotations and node-status shading, which are explicitly explained in the $A ^ { * }$ illustration of Section 5.2.

## 4.4 Mapping between Representations

The three representations, MILP, state-action, and Petri net, describe the same underlying RCPSP from diferent viewpoints. The MILP formulation encodes complete schedules, whereas the state-action and Petri-net representations encode intermediate execution states and their transitions. In particular, the state-action representation records explicitly which activities are completed, ongoing, or not yet started, while the Petri-net representation records the same information implicitly through timed control-flow and resource tokens.

## 4.4.1 Mapping between State-Action and Petri Net Representations

For the RCPSP-induced T T P NR considered in this paper, the reachability graph $R G = ( S _ { R G } , \mathbb { E } _ { R G } )$ of Section 4.3.2 is already defined at decision states. Each node is a timed state represented by a timed marking and the current system time, and each outgoing edge corresponds to choosing one activity to start next, with the clock advancing by the smallest amount needed to make the required control-flow and resource tokens available. Therefore, the Petri-net reachability graph and the state graph use the same transition granularity, difering only in how states are represented.

There is thus a natural correspondence between a state $\boldsymbol { s } = ( g ( s ) , \mathcal { V } ^ { c } ( s ) , \mathcal { V } ^ { o } ( s ) , \tilde { d } ( s ) )$ and a timed marking at the same system time. Completed activities correspond to immediately available control-flow tokens at the appropriate successor places, ongoing activities and their remaining durations correspond to delayed control-flow and resource tokens with positive residual duration, and currently available resources correspond to immediately available tokens in the resource places.

Proposition 1. The state graph $G = ( S , \mathbb { E } )$ and the T T P NR reachability graph $R G = ( S _ { R G } , \mathbb { E } _ { R G } )$ where $\cal { S } _ { R G }$ is the set of reachable timed states and $\mathbb { E } _ { R G }$ is the set of directed edges induced by transition firings, are isomorphic. That is, there exists a one-to-one correspondence between the states of G and the states of RG, $\phi : { \cal S }  { \cal S } _ { R G }$ , such that:

(i) $\phi ( s _ { 0 } )$ is the initial RG node represented by the timed marking $M _ { s _ { 0 } }$ and system time 0, where $M _ { s _ { 0 } }$ refines the ordinary initial marking $M _ { 0 }$

(ii) corresponding states have the same system time,

(iii) for every activity $j \in \mathcal V$ , there is an edge $s \stackrel { j } { \longrightarrow } s ^ { \prime }$ in G if and only if there is an edge $\phi ( s ) \xrightarrow { t _ { j } } \phi ( s ^ { \prime } )$ in RG with label $L ( t _ { j } ) = j$

Proof. The proof is presented in Appendix A.1.

## 4.4.2 Mapping between MILP and State-Action Representations

A partial schedule can be mapped to a state in the state graph induced by the state-action representation, but this mapping is generally many-to-one. Diferent partial schedules may lead to the same state. For example, in the running example, scheduling the independent activities a and c in either order may produce identical states. The following proposition formalizes this relation.

Proposition 2. Given an RCPSP, let G be the state graph associated with the state-action representation. Then, any partial schedule $( \mathcal { I } , \{ S _ { i } \} _ { i \in \mathcal { I } } )$ produced by earliest-start generation corresponds to a path in the graph G leading to a state $s \in { \mathcal { S } }$ satisfying $g ( s ) = \mathrm { m a x } _ { i \in \mathcal { T } } S _ { i }$ and $\mathcal { V } ^ { o } ( s ) = \{ i \in \mathcal { I } : S _ { i } \leq$ $g ( s ) < S _ { i } + d _ { i } \} , \tilde { d } _ { i } ( s ) = S _ { i } + d _ { i } - g ( s ) > 0$ for all $i \in \mathcal { V } ^ { o } ( s )$ , and $\mathcal { V } ^ { c } ( s ) = \mathcal { I } \setminus \mathcal { V } ^ { o } ( s )$ . Moreover, any state $s \in { \mathcal { S } }$ corresponds to at least one partial schedule $( \mathcal { I } , \{ S _ { i } \} _ { i \in \mathcal { I } } )$ such that $\mathcal { I } = \mathcal { V } ^ { o } ( s ) \cup \mathcal { V } ^ { c } ( s )$ , and $S _ { i } = g ( s ) + \tilde { d } _ { i } ( s ) - d _ { i }$ for all $i \in \mathcal { V } ^ { o } ( s )$ and $S _ { i } \leq g ( s ) - d _ { i }$ for $a l l i \in \mathcal { V } ^ { c } ( s )$

Proof. The proof is presented in Appendix A.2.

In particular, the above proposition implies that a complete path from $s _ { 0 }$ to a goal state $s _ { f }$ in the graph G defines a schedule $\{ S _ { i } \} _ { i \in \mathcal { V } }$ by recording the time at which each activity’s transition fires. Conversely, given a feasible MILP solution with start times $\{ S _ { i } ^ { \prime } \} _ { i \in \mathcal { V } }$ , the corresponding path in $G$ can be reconstructed by processing activities in non-decreasing order of their start times.

Note that since a state in S is not necessarily produced by a unique partial schedule, there may be multiple feasible schedules (paths in G) associated with a single state $s _ { f } \in S _ { f }$ . This many-to-one relationship between scheduling decisions and states may afect solution approaches that use these two representations.

Thus, while $A ^ { * }$ is naturally applied to the graph G associated with the state-action representation, the ordinary OR-style B&B approach is naturally applied to a MILP or partial-schedule representation. The distinction that $A ^ { * }$ searches over states, whereas B&B searches over a tree of partial scheduling decisions is central to the comparison below.

## 5 Solution Approaches

In this section, we describe the two exact solution methods B&B (Section 5.1) and $A ^ { * }$ (Section 5.2) and provide a structural comparison of the two approaches (Section 5.3). We first present each method in a generic form and then specialize it to the RCPSP. Both methods use lower bounds on the best achievable project completion time to guide their search. For the baseline comparison, we use the same critical-path lower bound and best-first (BF) node-selection rule for both methods. This design controls for the common lower bound and search order while retaining the algorithm-specific mechanisms that distinguish the two approaches. In Section 6, we additionally examine B&B variants with depth-first (DF) search and linear programming (LP) relaxations that produce stronger lower bounds.

## 5.1 Branch-and-Bound

In the field of OR, B&B is an exact solution framework for combinatorial optimization problems that systematically explores the solution space while avoiding exhaustive enumeration. In the context of the RCPSP, it is naturally applied to the MILP representation and corresponding partial schedules.

Within B&B, the explored solutions can be represented by a rooted search tree, in which the root node $N _ { 0 }$ corresponds to the optimization problem to be solved, i.e., its objective function and constraints. Each other node $N _ { j }$ of the B&B search tree is implicitly defined by the objective function of the problem to be solved and the constraints of the problem corresponding to its parent node $N _ { i } .$ as well as an additional attribute (usually a constraint) that every solution to the problem at $N _ { j }$ must satisfy, whereas this need not be the case for all solutions to the problem at $N _ { i }$ . A path in the search tree from the root node $N _ { 0 }$ to a leaf node thus corresponds to the initial problem in $N _ { 0 }$ , successively extended by the constraints added along the branches of this path. Consequently, the set of feasible solutions can only shrink from a node $N _ { i }$ to its child $N _ { j }$ , while the best achievable objective value of the corresponding subproblem cannot improve. Hence, B&B relies on partitioning the solution space (branching) and eliminating problems (nodes) that cannot lead to an optimal solution (bounding and pruning). The overall search is typically guided by strategies for selecting the next node to be evaluated.

Since the best achievable objective value of a node is hard to calculate in general, bounds are computed at the nodes of the search tree. First, a global upper bound is the objective value of a feasible solution of the problem in $N _ { 0 }$ , which may be improved several times during the search as new, better feasible solutions of $N _ { 0 }$ are found. Second, a local lower bound of the problem in $N _ { i }$ is computed at each node $N _ { i }$ , i.e., a value not exceeding the best-possible objective function value that any feasible solution in $N _ { i }$ can achieve. The computation of this bound can, for example, be performed by solving a relaxed (simplified) version of the optimization problem considered in $N _ { i }$ that can be solved eficiently. If this bound indicates that no better solution can be found within the corresponding subtree originating in $N _ { i }$ , the node is discarded (pruned, fathomed).

## 5.1.1 Generic Framework for Minimization Problems

Consider a combinatorial optimization problem mi $_ { 1 } { } _ { x \in \mathcal { X } } F ( x )$ with the set of feasible solutions $\mathcal { X }$ and objective function $F : \mathcal { X }  \mathbb { R }$ . Assume that X is too large to be enumerated explicitly, so that it is systematically explored by means of B&B. The root node $N _ { 0 }$ represents the original problem. The remaining nodes $N _ { i }$ are associated with subsets $\mathcal { X } ( N _ { i } ) \subseteq \mathcal { X }$ of feasible solutions. Hence, the branching of a node $N _ { i }$ corresponds to partitioning $\mathcal { X } ( N _ { i } )$ into smaller disjoint subsets $\begin{array} { r } { \mathcal { X } ( N _ { i } ) = \bigcup _ { i = i + 1 } ^ { k } \mathcal { X } ( N _ { j } ) } \end{array}$ where $N _ { i + 1 } , \ldots , N _ { k }$ are the child nodes of $N _ { i }$ . This partitioning is typically induced by fixing decision variables or imposing additional constraints.

The algorithm maintains a global upper bound F<sup>¯</sup>, which corresponds to the objective value of the best feasible solution of the original problem found so far. Additionally, for each node $N _ { i }$ , a lower bound $\begin{array} { r } { \underline { { \operatorname { F } } } ( N _ { i } ) \leq \operatorname* { m i n } _ { x \in \mathcal { X } ( N _ { i } ) } F ( x ) } \end{array}$ on the objective value of all solutions in $\mathcal { X } ( N _ { i } )$ is computed. The quality of these bounds afects the eficiency of the algorithm.

A node $N _ { i }$ is pruned if one of the following conditions holds:

$\underline { { \mathrm { F } } } ( N _ { i } ) \geq \bar { F }$ , since no solution in $\mathcal { X } ( N _ { i } )$ can improve the current best solution.

$\mathcal { X } ( N _ { i } ) = \emptyset$ , i.e., the subproblem at $N _ { i }$ is infeasible.

• The relaxation solved at $N _ { i }$ yields a feasible solution $x \in \mathcal { X }$

The order in which nodes are explored is determined by a search strategy. Common strategies include DF search and BF search. DF search completely explores a single branch before backtracking. It aims at quickly determining feasible solutions, and thus global upper bounds, to the original problem. BF search, in contrast, selects the next node to be explored based on the lower bounds of the not yet fathomed (active) nodes.

In summary, a generic version of B&B can be described as follows:

1. Initialize the set of active nodes with the root node (corresponding to the original problem).

2. Set $\bar { F } : = + \infty ,$ or ${ \bar { F } } : = F ( x ^ { 0 } )$ if a feasible solution $x ^ { 0 }$ is known in advance.

3. While there are active nodes:

(a) Select a node $N _ { i }$ according to the search strategy.

(b) Compute $\underline { { \mathrm { F } } } ( N _ { i } ) , \mathrm { e . g . }$ , by considering and solving a relaxed version of the problem in $N _ { i }$

(c) If $\underline { { \mathrm { F } } } ( N _ { i } ) \geq \bar { F }$ or if the problem in $N _ { i }$ has no feasible solution, prune $N _ { i } , \mathrm { i . e . }$ , delete $N _ { i }$ from the set of active nodes, and continue.

(d) Else, if the solution $x _ { i }$ of the relaxed version of the problem in $N _ { i }$ is feasible for the original problem, set $\bar { F } \gets \operatorname* { m i n } \{ \bar { F } , F ( x _ { i } ) \}$ , potentially update the best solution to $x _ { i } ,$ prune $N _ { i }$ and continue.

(e) Otherwise, branch on $N _ { i } , \mathrm { i . e . }$ , delete it from the set of active nodes and add its children to the set of active nodes.

The correctness of this generic version of B&B follows from the fact that only nodes that cannot contain better solutions than the currently best known solution are pruned. Therefore, an optimal solution is never discarded if no optimal solution is yet known. The eficiency of the method critically depends on the strength of the bounds, the efectiveness of the branching scheme and the search strategy. In the worst case, the algorithm may still explore the entire search space. However, in practice, strong bounds and good branching rules can lead to substantial reductions in computational efort.

## 5.1.2 A Branch-and-Bound Method for the RCPSP

As the goal of this paper is to compare baseline variants of $A ^ { * }$ and B&B, we refrain from adapting sophisticated B&B approaches for the RCPSP (see Section 2). Instead, we restrict ourselves to a fairly basic version which we describe in this section.

Each node $N _ { i }$ is associated with a partial schedule $( \mathcal { I } , S _ { \mathcal { I } } )$ of the considered RCPSP instance with the current time $g ( N _ { i } ) = \operatorname* { m a x } _ { j \in \mathcal { T } } S _ { j }$ . The state-consistent critical-path (CP) lower bound on the project duration reachable from node $N _ { i }$ denoted by $\hat { f } ( N _ { i } ) \ ( \mathrm { i . e . , } \ E ( N _ { i } )$ in the generic framework), corresponds to the optimal objective function value of the relaxation of problem $\mathrm { ( 3 a ) \mathrm { - } ( 3 g ) }$ that neglects all resource constraints under the given partial schedule with start times $S _ { \mathcal { T } } = \{ S _ { j } \} _ { j \in \mathcal { T } }$

$$
\hat { f } ( N _ { i } ) = \operatorname* { m i n } _ { S _ { j } , j \in \mathcal { V } \backslash \mathcal { I } } S _ { n }\tag{8a}
$$

$$
\forall ( j , l ) \in \mathcal { E }\tag{8b}
$$

$$
\forall j \in \mathcal { V } \setminus \mathcal { I }\tag{8c}
$$

Since this relaxation contains only precedence constraints, $\hat { f } ( N _ { i } )$ is obtained by a single forward critical-path pass and can be computed without an optimization solver.

In this relaxation, the start times of already scheduled activities are fixed. Equivalently, in the timed-state representation, completed activities require no further processing, ongoing activities contribute their residual durations, and not-yet-started activities may start no earlier than $g ( N _ { i } )$ , subject to precedence constraints but with resource constraints relaxed. The resulting bound satisfies $\hat { f } ( N _ { i } ) = g ( N _ { i } ) + h ( N _ { i } )$ , where $h ( N _ { i } )$ is a lower bound on the remaining time to project completion.

If a node $N _ { i }$ with partial schedule $( \mathcal { I } , S _ { \mathcal { I } } )$ cannot be pruned, we branch over all activities that can be started at time $t \geq g ( N _ { i } )$ , while respecting all precedence and resource constraints. Hence, a child $N _ { k }$ of $N _ { i }$ is associated with augmenting the partial schedule with an additional activity $j ,$ , such that all predecessors of $j$ belong to $\mathcal { I }$ , where activity $j$ is started at the earliest feasible time no earlier than $g ( N _ { i } )$ , resulting in the partial schedule $( \mathcal { I } \cup \{ j \} , S _ { \mathcal { I } \cup \{ j \} } )$ (see Section 3.1). Hence, $S _ { j }$ is given by the solution of the following mixed integer program, where the values $y _ { i t }$ are fixed to one if $S _ { i } = t$ and to zero otherwise for all $i \in \mathcal { I }$ :

$$
\operatorname* { m i n } _ { S _ { j } , y _ { j t } } S _ { j }\tag{9a}
$$

$$
\mathrm { s . t . } \sum _ { t = g ( N _ { i } ) } ^ { H - d _ { j } } y _ { j t } = 1
$$

$$
H { - } d _ { j }\tag{9b}
$$

$$
S _ { j } = \sum _ { \substack { t = g ( N _ { i } ) } } t y _ { j t }\tag{9c}
$$

$$
S _ { l } + d _ { l } \le S _ { j }
$$

$$
\operatorname* { m i n } \{ t , H - d _ { l } \}
$$

$$
\forall ( l , j ) \in \mathcal { E } , l \in \mathcal { I }\tag{9d}
$$

$$
\sum \begin{array} { r l r l r l } { r _ { l k } } & { { } } & { \sum } & { { } } & { y _ { l \tau } \le R _ { k } } \end{array}
$$

$$
l \in \mathcal { T } \cup \{ j \} \qquad \tau = \operatorname* { m a x } \{ 0 , t - d _ { l } + 1 \}\tag{9e}
$$

$$
\begin{array} { l } { { y _ { j t } = 0 } } \\ { { y _ { j t } \in \{ 0 , 1 \} , } } \end{array}
$$

$$
\forall t = 0 , \ldots , g ( N _ { i } ) - 1\tag{9f}
$$

$$
\forall t = g ( N _ { i } ) , \ldots , H - d _ { j }\tag{9g}
$$

This ensures that $g ( N _ { k } ) \ge g ( N _ { i } )$ and that the start time $S _ { j }$ of activity j takes into account both resource and precedence constraints. We note that in practice $S _ { j }$ can be computed analytically without the need to solve (9a)–(9g).

Scheduling all activities sequentially yields a feasible schedule with makespan H. We therefore initialize the upper bound as $\bar { F } = H$ . We apply a BF search strategy that selects an active node $N _ { i }$ with minimum value $\hat { f } ( N _ { i } )$ . As a tie-breaker, we choose the highest value of $g ( N _ { i } )$ . If both values are tied, we apply a First In, First Out (FIFO) strategy. B&B can also employ other node-selection strategies, but this choice aligns the B&B node-selection rule with the BF expansion rule of $A ^ { * }$ thereby controlling for diferences in search order and allowing the comparison to focus on the efects of tree-based enumeration and duplicate-state detection.

For the running example, the procedure generates the search tree shown in Figure 5. The initial upper bound is $\bar { F } = H = 1 0$ . From the root node $N _ { 0 }$ , the only possible branching decision is dummy activity 0, because it is the only activity without predecessors. This produces node $N _ { 1 }$ . Since activity 0 has zero duration, $( g ( N _ { 1 } ) , h ( N _ { 1 } ) , \hat { f } ( N _ { 1 } ) ) = ( 0 , 4 , 4 )$ . At $N _ { 1 }$ , activities $a , b ,$ and c become available, generating nodes $N _ { 2 } , N _ { 3 }$ , and $N _ { 4 } ,$ respectively. Each has evaluation value 4. Under the above tie-breaking rules, $N _ { 2 }$ is selected first, and activity a is scheduled at time 0. Branching at $N _ { 2 }$ generates $N _ { 5 }$ , corresponding to selecting activity $b ,$ and $N _ { 6 }$ , corresponding to selecting activity c. Activity b requires two resource units and therefore cannot start until activity a completes. Hence, $( g ( N _ { 5 } ) , h ( N _ { 5 } ) , \hat { f } ( N _ { 5 } ) ) = ( 3 , 4 , 7 )$ At this stage, $N _ { 5 }$ is not pruned because $\hat { f } ( N _ { 5 } ) = 7 < \bar { F } = 1 0$ . It stays active but is not selected while nodes with smaller evaluation values exist. Activity $^ { c , }$ in contrast, can start in parallel with activity a. Consequently, $\left( g ( N _ { 6 } ) , h ( N _ { 6 } ) , \hat { f } ( N _ { 6 } ) \right) = ( 0 , 4 , 4 )$ . Branching at $N _ { 3 }$ generates $N _ { 7 }$ and $N _ { 8 }$ , corresponding to selecting activities a and $c ,$ respectively; both have evaluation triple (2, 4, 6). Branching at $N _ { 4 }$ generates $N _ { 9 }$ by selecting activity a and $N _ { 1 0 }$ by selecting activity b. Since activity a can start in parallel with activity $c ,$ node $N _ { 9 }$ has evaluation triple $( 0 , 4 , 4 )$ . By the FIFO rule, $N _ { 6 }$ is expanded next. Selecting activity b from $N _ { 6 }$ subsequently generates $N _ { 1 1 }$ with evaluation triple $( 3 , 3 , 6 )$ . The search next processes the remaining node $N _ { 9 }$ with evaluation value 4, selecting activity b to produce $N _ { 1 2 }$ with evaluation triple (3, 3, 6). Expanding $N _ { 1 1 }$ generates $N _ { 1 3 }$ and $N _ { 1 4 }$ by scheduling activities d and $e ,$ respectively, both with $( g , h , \hat { f } ) = ( 5 , 1 , 6 )$ . Since $g ( N _ { 1 3 } ) = g ( N _ { 1 4 } ) = 5 > g ( N _ { 1 2 } ) = 3$ , the tie-breaking rule prefers $N _ { 1 3 }$ and $N _ { 1 4 }$ over $N _ { 1 2 }$ , and FIFO selects $N _ { 1 3 }$ , which generates $N _ { 1 5 }$ by scheduling e. Next, $N _ { 1 4 }$ is expanded, generating $N _ { 1 6 }$ by scheduling d. Nodes $N _ { 1 5 }$ and $N _ { 1 6 }$ represent the same partial schedule but remain distinct nodes in the B&B tree. By FIFO, $N _ { 1 5 }$ is selected next and generates the goal node $N _ { 1 7 }$ by scheduling the dummy end activity n. With $( g , h , \hat { f } ) = ( 6 , 0 , 6 )$ $N _ { 1 7 }$ is selected next, updating the upper bound to $\bar { F } = 6$ . All remaining active nodes have lower bounds of at least 6, and the search terminates with the optimal makespan of 6.

Our procedure is refined by two additional elements. Observe that the above implementation may create duplicate schedules due to diferent insertion of activities starting at the same time. To prevent such duplication, we prune nodes which insert activities starting at the same time in non-lexicographic order. In the example of Figure 5, node $N _ { 9 }$ would be pruned immediately as it schedules both a and c at time 0 in non-lexicographic order. Similarly, $N _ { 1 6 }$ would be pruned because activities e and d are scheduled at the same time in non-lexicographic order. Additionally, the schedule produced by the above implementation may not be active. Specifically, if Node $N _ { k }$ is created from node $N _ { i }$ by scheduling activity j, (9) only enforces earliest starting time from $g ( N _ { i } )$ onward. To verify that the schedule is active, we need to check whether or not j can be scheduled prior to time $g ( N _ { i } )$ . Thus, if replacing $g ( N _ { i } )$ by zero in (9), produces $S _ { j } < g ( N _ { i } )$ then the node $N _ { k }$ would be pruned, as it results in a nonactive schedule. In Figure 5, such pruning would have occurred if, for example, we expanded node $N _ { 5 }$ to schedule activity $^ { c , }$ which would have resulted in a nonactive schedule.

![](images/4aa2effddd76ac7549d819e4fc8d494379aee2a1ce5a0b19b36e6a033c0dffc6.jpg)  
Figure 5: B&B search tree generated by a best-first procedure.  
Note: Gray nodes are generated and expanded, and white nodes are only generated. Each node shows $( g , h , \hat { f } )$ values. Node numbers reflect the node generation order.

## 5.2 A<sup>∗</sup> Search

$A ^ { * }$ is a well-known BF graph-search algorithm for finding a minimum-cost path from an initial state to a goal state (Hart et al., 1968). We first describe the generic $A ^ { * }$ framework and then specialize it to the RCPSP state graph defined in Section 4.2.

## 5.2.1 Generic Framework

Consider a directed state-space graph with an initial state, a set of goal states, and nonnegative transition costs. For each generated state $s , A ^ { * }$ uses the evaluation function ${ \hat { f } } ( s ) = g ( s ) + h ( s )$ , where $g ( s )$ is the cost of the best path currently known from the initial state to s, and $h ( s )$ estimates the minimum remaining cost from s to a goal state. The heuristic h is admissible if it never overestimates this remaining cost. If, in addition, h is consistent, the standard graph-search version of $A ^ { * }$ guarantees that the first goal state selected for expansion lies on an optimal path.

A<sup>∗</sup> maintains two sets: Open, containing generated states that are candidates for expansion, and Closed, containing states that have already been expanded. Initially, the initial state is inserted into Open. At each iteration, $A ^ { * }$ selects a state from Open with minimum <sup>ˆ</sup>f-value. Ties may be resolved using a secondary rule, such as preferring the state with the larger g-value.

The selected state is expanded by generating its successor states, i.e., the states reachable through its outgoing edges. A newly generated state is inserted into Open; if the same state has already been generated through another path, only the better path needs to be retained. The expanded state is moved to Closed. The procedure continues until a goal state is selected or Open becomes empty.

## 5.2.2 A<sup>∗</sup> for the RCPSP

We now apply this generic framework to the RCPSP state graph G defined in Section 4.2. Its initial state is $s _ { 0 }$ , its goal states are the states in $ { \boldsymbol { S } } _ { f }$ , and each edge corresponds to selecting one activity to be scheduled next. The transition cost from s to its successor $s ^ { \prime }$ is $\Delta ( s , s ^ { \prime } ) = g ( s ^ { \prime } ) - g ( s )$ . Consequently, the accumulated path cost from $s _ { 0 }$ to state s equals the current system time $g ( s )$

For the RCPSP, the evaluation function is therefore ${ \hat { f } } ( s ) = g ( s ) + h ( s )$ , where $h ( s )$ is an admissible lower bound on the remaining time to project completion. When a scheduling decision is made, time advances to the earliest moment at which the selected activity can begin, namely when its precedence and resource requirements are satisfied. We use the same tie-breaking rule as for B&B: among states with equal ${ \hat { f } } _ { \mathrm { - V a l u e s } }$ , we select the state with the largest g-value and remaining ties are resolved by FIFO. Furthermore, we use the same CP lower bound defined in Section 5.1. Thus, $h ( s ) = { \hat { f } } ( s ) - g ( s )$ provides an admissible lower bound on the remaining time to project completion. We also consider the two pruning refinements introduced at the end of Section 5.1: lexicographic pruning of alternative orderings of activities starting at the same time, and active-schedule pruning of scheduling decisions that could have been made earlier without delaying previously scheduled activities.

In Figure $4 , A ^ { * }$ searches the reachability graph induced by the $T T P N R$ in Figure 3, which is equivalent to G by Proposition 1. Each node represents a timed state consisting of the current system time $g ( s )$ and a timed marking $M _ { s }$ specifying which control-flow and resource tokens are immediately available and which are delayed by ongoing activities. Each outgoing arc represents one feasible next scheduling decision: choosing an activity to start next. This decision may leave $g ( s )$ unchanged if the activity can start immediately, or advance time to the earliest moment at which its precedence and resource requirements are satisfied.

For every generated state $s , A ^ { * }$ computes ${ \hat { f } } ( s ) = g ( s ) + h ( s )$ and selects an open state with minimum $\hat { f }$ for expansion. From the initial state $s _ { 0 }$ , only dummy activity 0 is enabled, so expanding $s _ { 0 }$ generates only $s _ { 1 }$ . From $s _ { 1 }$ , activities $a , b ,$ and c are all feasible next decisions, and therefore three successor states $s _ { 2 } , s _ { 3 } .$ and $s _ { 4 }$ are generated. In this way, the general BF principle of $A ^ { * }$ is instantiated as a search over timed project states.

To illustrate the calculation of ${ \hat { f } } ( s )$ , consider state $s _ { 1 0 }$ in Figure 4, whose timed marking is represented by $M _ { s _ { 1 0 } } = [ p _ { 5 } : 1 , ~ p _ { 6 } : 1 ^ { 2 } , ~ p _ { 7 } : 1 ^ { 2 } , ~ p _ { 8 } : 1 , ~ p _ { r _ { 1 } } : 2 ^ { 2 } ]$ . At state $s _ { 6 }$ , the decisions to execute activities 0, $^ { a , }$ and $c$ have already been taken, and $( g ( s _ { 6 } ) , h ( s _ { 6 } ) ) = ( 0 , 4 )$ . Activity b is the only selectable activity with respect to the precedence constraints. It requires two resource tokens, which become available when activities a and c complete after three time units. Consequently, $g ( s _ { 1 0 } ) = S _ { b } = 3$ The remaining critical-path duration through activities $\{ b , d , e , n \}$ is three time units, so $h ( s _ { 1 0 } ) = 3$ and ${ \hat { f } } ( s _ { 1 0 } ) = g ( s _ { 1 0 } ) + h ( s _ { 1 0 } ) = 6 $

## 5.3 Structural Comparison of Branch-and-Bound and $A ^ { * }$

Although B&B and $A ^ { * }$ both systematically explore alternative scheduling decisions using lower bounds to guide the search, they difer in several fundamental respects. In this section, we identify the key structural diferences and discuss their implications for solving the RCPSP. The results are summarized in Table 1.

The B&B algorithm searches a tree whose nodes are generated by sequences of scheduling decisions, whereas $A ^ { * }$ searches a graph of timed project states. Every B&B node induces a timed state containing the information required to determine the remaining feasible scheduling decisions. Diferent branches of the B&B tree may induce the same timed state and therefore represent the same remaining scheduling problem. Tree-based B&B retains such branches as separate nodes. For example, the decision sequences $\langle 0 , a , c \rangle$ and $\langle 0 , c , a \rangle$ generate diferent paths in the B&B tree, although they may result in the same scheduled activities, current time, ongoing activities, and resource availability. Graph-based $A ^ { * }$ , in contrast, identifies nodes by their timed-state representation. When a generated successor is already present in Open or Closed, it is recognized as a duplicate and is not retained as a separate search node. The $A ^ { * }$ state graph can therefore be viewed as the graph obtained from the B&B tree by merging nodes that induce the same timed project state. Consequently, under otherwise identical search conditions, tree-based B&B may generate and evaluate the same remaining scheduling problem several times, whereas graph-based $A ^ { * }$ represents it once. The potential redundancy increases with project concurrency, because independent activities can be selected in many diferent orders while leading to the same timed state.

Table 1: Structural comparison of B&B and $A ^ { * }$ for the RCPSP.
<table><tr><td>Aspect</td><td>B&amp;B</td><td>A* Search</td></tr><tr><td>Search structure</td><td>Tree  $\mathrm { N o }$ </td><td>Graph</td></tr><tr><td>Duplicate detection</td><td></td><td>Yes (via OPEN and CLOSED lists)</td></tr><tr><td>Upper bound</td><td>Maintained and updated</td><td>Not used</td></tr><tr><td>Lower bound</td><td> $g + h$  (same)</td><td> $g + h$  (same)</td></tr><tr><td>Pruning mechanism</td><td>Explicit: bound  $( { \hat { f } } ( N _ { i } ) \geq { \bar { F } } )$  sibility, or a feasible solution</td><td>, infea- Implicit: best-first + admissibility</td></tr><tr><td>h evaluations per distinct state</td><td>Possibly &gt; 1</td><td>1 (with duplicate check before evaluation)</td></tr><tr><td>Node selection</td><td>Flexible  $( \mathrm { b e s t \mathrm { - } f i r s t } , \mathrm { d e p t h \mathrm { - } f i r s t } , \dots )$ </td><td>Best-first (required)</td></tr></table>

We now turn our attention to the bounding mechanisms of the algorithms. In their RCPSP specializations, both methods use the same evaluation ${ \hat { f } } = g + h$ , but they difer in how this bound afects the search. B&B maintains a global upper bound ${ \bar { F } } _ { ; }$ , updated whenever a complete schedule is found, and explicitly prunes any node $N _ { i }$ with $\hat { f } ( N _ { i } ) \ge \bar { F }$ . The interplay between the improving upper bound and the lower bounds of individual nodes is central to its efectiveness. Our $A ^ { * }$ implementation does not maintain an upper bound. Instead, BF expansion with an admissible and consistent heuristic guarantees that the first goal state selected from Open is optimal. Hence, $A ^ { * }$ does not exploit an incumbent solution for pruning in the way that B&B does.

The methods furthermore difer with respect to the heuristic computation eficiency. With state based duplicate detection, $A ^ { * }$ retains a single search record for each distinct state, and with a consistent heuristic, each state needs to be expanded at most once. If duplicate detection is performed before evaluating $h ,$ the heuristic is therefore computed only once per distinct retained state. In B&B, distinct tree nodes can induce the same timed state, so h may be evaluated multiple times for the same remaining scheduling problem.

B&B permits flexible node-selection strategies such as depth-first, breadth-first, or hybrid. This choice interacts with the upper-bound update mechanism. $A ^ { * }$ , on the other hand, requires BF selection for its optimality guarantee. When both use BF selection with the same heuristic, their expansion orders are closely related but not identical since $A ^ { * }$ may expand fewer nodes due to duplicate detection.

With respect to the memory requirements, A<sup>∗</sup> stores all expanded states in a Closed list and all frontier states in an Open list. DF B&B avoids most of this cost because it only keeps the children of the nodes on the current path from the root in memory. However, when B&B uses $\mathrm { B F }$ selection, it has to maintain all unexpanded nodes, resulting in memory requirements comparable to $A ^ { * }$ . In this setting, the Closed list is an additional cost of $A ^ { * }$ that can also save memory by preventing duplicate states from entering the Open list.

## 6 Empirical Study

The purpose of this section is to examine the diferences between the ordinary B&B and $A ^ { * }$ algorithms presented above across a broad range of instances. To this end, we consider the 480 instances in the J30 benchmark from PSPLIB (Kolisch and Sprecher, 1997), which consists of RCPSP instances with 30 activities and varying precedence structures and resource requirements. Note that we do not aim to establish that the algorithms are competitive with state-of-the-art RCPSP solvers.

All methods were implemented in $\mathrm { P y }$ thon 3.13.13, and experiments were conducted on a single core of an Intel Xeon Gold 6354 @ 3.00 GHz, without parallelization, with a time limit of 2 hours and a memory limit of 64 GB. All code, data, and results are available at https: $: / / \tt g i$ thub.com/ ShimShtern/rcpsp-astar-vs-branch-and-bound (version v1.0).

We start by comparing $A ^ { * }$ and B&B using the same CP lower bound described in Section 5.1. Thus, the only diference between BF B&B and $A ^ { * }$ is duplicate identification: $A ^ { * }$ abstracts partial schedules into states and tracks duplicate states. Figure 6 shows the efect of applying the lexicographic ordering and active-schedule pruning refinements discussed in Section 5.1 to both methods. The left panel depicts the number of instances proven optimal as a function of runtime, and the right panel depicts the number of instances whose maximum frontier size, i.e., the largest number of open nodes held simultaneously during the search, does not exceed a given value, which serves as a proxy for memory consumption.

![](images/6c9ab2c38b435baa0858fe79edba0209f93cd36003ba60734b8ee07b26447e0c.jpg)

![](images/fa0e44e5c34e491223d9c3b924408739844dbaf0529313cc228ab2fa7b51393a.jpg)  
Figure 6: Time and memory behavior of $A ^ { * }$ and best-first B&B under diferent pruning configurations.

Without incorporating the pruning rules, BF B&B results in 230 instances terminated upon reaching the memory limit and one upon reaching the time limit. With $A ^ { * }$ , in contrast, only four instances did not reach optimality, all due to the time limit. This indicates substantially higher memory consumption for B&B, partly due to equivalent partial schedules appearing in diferent parts of the search tree. Adding lexicographic pruning aided both methods. For $A ^ { * }$ , three of the four previously time-limited instances reached an optimal solution. For B&B, the number of instances terminating due to the memory limit decreased to 82, and the number terminating due to the time limit increased to 11. Lexicographic pruning therefore removes repeated occurrences of the same partial schedule across branches, but not distinct partial schedules corresponding to the same state. Among the instances reaching optimality after applying lexicographic pruning, the median runtime reduction was 93.3% for B&B, compared with 32.1% for $A ^ { * }$ . Active-schedule pruning further improved both methods. For $A ^ { * }$ the median reduction in runtime to optimality was 52.8% and all instances were solved to optimality. For B&B, it eliminated all memory-limit terminations, although the number of instances reaching the time limit increased to 35, and yielded a median runtime reduction of 82.4%. Because $A ^ { * }$ already reduces the search space through duplicate-state identification, these pruning mechanisms yield smaller additional gains than for B&B. The right panel of Figure 6 shows that, regardless of the pruning rules applied, $A ^ { * }$ requires a frontier that is no larger than that of BF B&B on almost all instances. The diferences become less pronounced as the pruning rules are introduced, since they reduce the frontier of B&B considerably more than that of $A ^ { * }$ . Specifically, the median ratio between the maximum frontier sizes of BF B&B and $A ^ { * }$ decreases from about 140 without pruning rules to about 10 with lexicographic pruning and to about 1.5 with both pruning rules. Nevertheless, on the hardest instances the gap remains large, as the largest frontier of BF B&B with both pruning rules is still more than 20 times that of $A ^ { * }$

A potential advantage of B&B over $A ^ { * }$ is its ability to prune using incumbent upper bounds. We therefore examine whether exploiting this mechanism can improve the relative performance of B&B. More specifically, we compare $A ^ { * }$ to B&B with DF search, where the child with the better lower bound is explored first. DF search may obtain an incumbent before reaching the optimum. Additionally, we utilize the partial schedule to obtain weak and strong LP relaxations, both solved using Gurobi 13.0.3. The weak LP does not add constraints to the integer formulation, whereas the strong LP models each precedence relation through one constraint per time period, resulting in tighter lower bounds. The two formulations are detailed in Appendix B.1. They provide stronger lower bounds for B&B and, when feasible, incumbent upper bounds. The status counts for the diferent methods are summarized in Figure 7. Without LP relaxations, DF B&B finds suboptimal incumbent solutions for 21 instances and optimal solutions without proof of optimality for 13 additional instances out of the 40 instances terminating without proving optimality. Thus, although DF search is less efective at proving optimality, it provides useful incumbent solutions for many instances that remain unsolved within the time limit. By construction, BF B&B without LP relaxations produces no incumbents before the optimum, except for the initial one. In our experiments, the weak LP also yielded none, while the strong LP yielded fewer than DF B&B. Additional results, including a breakdown of unresolved instances by PSPLIB parameter subclass and a comparison of the final upper and lower bounds of the B&B variants, are provided in Appendix B.2.

![](images/ac04759371115e62fd09c27a94fb6cbe6daaf9f29373b929fe4bc52f4dd2548c.jpg)  
Figure 7: Status counts for each method.

Figure 8 shows the time and memory profiles of the diferent methods over all instances, and Table 2 summarizes the performance on the 365 instances for which all methods proved optimality. The profiles in Figure 8 show that the DF search is generally much more memory eficient but slower than the BF search. Somewhat surprisingly, they also show that both LP variants reduce overall performance: although they strengthen the bounds, their computational overhead substantially increases runtime and causes more instances to terminate without proving optimality. Table 2 provides a closer look at the instances solved by all methods. It confirms the memory advantage of the DF search, whose maximum frontier is much smaller than that of the BF search. It further shows that the strong LP explores the tree much more eficiently than the weak LP and the no-LP variant, as reflected in the number of generated nodes, but incurs substantially higher runtime and memory usage.

Overall, A<sup>∗</sup> outperforms the B&B variants. Its state-space representation and duplicate elimination reduce the efective search space enough to outweigh the additional upper- and lower-bound mechanisms available to B&B.

![](images/c88ed3fce74b25d6f835b79a3de5c43254d7aa9cf342ef506b3ec42195f3dc85.jpg)  
Figure 8: Time and memory performance comparing A<sup>∗</sup> with BF and DF B&B, with and without LP relaxations.

## 7 Conclusions and Future Research

This paper investigates the computational diferences between B&B and A<sup>∗</sup> using the RCPSP as a common problem. We establish the relationships between MILP, state-action, and timed Petri-net representations and compare basic implementations of both methods. By using the same lower bound and BF node-selection rule, we examine the efect of their diferent search mechanisms. In particular, tree-based B&B may repeatedly explore the same remaining scheduling problem through diferent decision sequences, whereas graph-based A<sup>∗</sup> identifies equivalent states and avoids their repeated exploration.

Experiments on 480 PSPLIB J30 instances demonstrate the importance of this diference. With lexicographic and active-schedule pruning, A solved all 480 instances to optimality, compared with 445 for BF B&B under the same time and memory limits. Although these pruning rules substantially improved B&B, they did not eliminate the performance gap. DF B&B required less memory and found feasible schedules for many instances where the search could not be completed within the time limit. Stronger LP relaxations reduced the number of generated nodes but increased computational time, illustrating the trade-of between bound strength and evaluation efort.

These results highlight the importance of how the search space is represented and explored. They also illustrate diferent advantages of the two approaches: duplicate-state detection can considerably reduce redundant exploration, while DF search and incumbent-based pruning ofer opportunities to control memory use and obtain feasible solutions early. Such insights can help researchers select appropriate search methods according to the characteristics and requirements of the problem. However, our findings concern the implementations and benchmark instances considered here and do not claim the general superiority of either method.

Several directions for future research arise from this study. Since both approaches admit numerous algorithmic enhancements, an important direction is to investigate the transfer of efective mechanisms between them. Examples include introducing duplicate-state detection into B&B, incorporating incumbent-based pruning into A<sup>∗</sup>, applying constraint propagation to restrict scheduling decisions and strengthen bounds, and using lazy heuristic evaluation to reduce unnecessary computation. Finally, extending the analysis to other combinatorial optimization problems would help quantify the findings across additional problem domains.

Table 2: Computational summary on the 365 PSPLIB J30 instances for which every method proved optimality.
<table><tr><td>Method</td><td>Search</td><td>LP relaxation</td><td>Wall time (s) median [IQR]</td><td>Expanded median [IQR]</td><td>Generated median [IQR]</td><td>Max. frontier median [IQR]</td><td>Peak RSS (MiB) median [IQR]</td></tr><tr><td rowspan="2">A*</td><td></td><td></td><td>0.074 [0.026-0.300]</td><td>1,147 [465-3,967]</td><td>2,353 [922–8,076]</td><td>1,111</td><td>24.7</td></tr><tr><td></td><td></td><td>0.111</td><td>1,330</td><td>2,982</td><td>[463-4,070] 1,548</td><td>[23.8–28.2] 23.8</td></tr><tr><td rowspan="7">B&amp;B</td><td>BF</td><td>No</td><td>[0.033-0.560] 6.831</td><td>[472–6,622] 1,147</td><td>[986-13,263] 2,526</td><td>[498-6,493] 1,315</td><td>[23.4–26.0] 53.7</td></tr><tr><td>BF</td><td>Weak</td><td>[2.742–28.366] 8.963</td><td>[430-4,742] 69</td><td>[887–11,158] 207</td><td>[477–5,418] 81</td><td>[49.7–62.0] 270.8</td></tr><tr><td>BF</td><td>Strong</td><td>[2.413-79.996]</td><td>[0-951] 6,042</td><td>[1-2,445] 7,476</td><td>[1-672]</td><td>[231.4–312.8]</td></tr><tr><td>DF</td><td>No</td><td>0.433 [0.118-2.215]</td><td>[1,985–29,354]</td><td>[2,262–40,583]</td><td>63 [51-76]</td><td>22.9 [22.7–23.0]</td></tr><tr><td>DF</td><td>Weak</td><td>11.570 [4.554–59.114]</td><td>5,295 [1,748-19,337]</td><td>6,741 [1,905–31,502]</td><td>64 [51-79]</td><td>52.4 [49.0–60.1]</td></tr><tr><td>DF</td><td>Strong</td><td>11.727 [2.385-86.468]</td><td>224 [0-1,905]</td><td>420 [1-3,808]</td><td>32 [1-58]</td><td>271.0 [230.5-314.6]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Note: Every column uses the same common subset of 365 instances. Values in brackets are interquartile ranges. Wall time includes LP overhead and measures time until optimality was proven. Max. frontier is the largest candidate-node list observed during a run. DF denotes depth-first search with local best-lower-bound child ordering.

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft – Project number 524165670.

## Use of AI Tools

During the preparation of this work the authors used ChatGPT and Claude to assist with coding, writing, and editing. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the article.

## References

Christian Artigues, Sönke Hartmann, and Mario Vanhoucke. Fifty years of research on resourceconstrained project scheduling explored from diferent perspectives. European Journal of Operational Research, 328(2):367–389, 2026. doi: 10.1016/j.ejor.2025.03.024.

Colin E. Bell and Kwangho Park. Solving resource-constrained project scheduling problems by A<sup>∗</sup> search. Naval Research Logistics, 37(1):61–84, 1990. doi: 10.1002/1520-6750(199002)37:1<61:: AID-NAV3220370104>3.0.CO;2-S.

Jacek Blazewicz, Jan Karel Lenstra, and A. H. G. Kan Rinnooy. Scheduling subject to resource constraints: classification and complexity. Discrete Applied Mathematics, 5(1):11–24, 1983. doi: 10.1016/0166-218X(83)90012-4.

Peter Brucker and Sigrid Knust. Lower bounds for resource-constrained project scheduling problems. European Journal of Operational Research, 149(2):302–313, 2003. doi: 10.1016/S0377-2217(02) 00762-2.

Peter Brucker, Sigrid Knust, Arno Schoo, and Olaf Thiele. A branch and bound algorithm for the resource-constrained project scheduling problem. European Journal of Operational Research, 107(2): 272–288, 1998. doi: 10.1016/S0377-2217(97)00335-4.

Peter Brucker, Andreas Drexl, Rolf Möhring, Klaus Neumann, and Erwin Pesch. Resource-constrained project scheduling: Notation, classification, models, and methods. European Journal of Operational Research, 112(1):3–41, 1999. doi: 10.1016/S0377-2217(98)00204-5.

G. Chen, He; W., Y. Tian, and K. Ma. Resource-constrained project scheduling with multiple states: Bi-objective optimization model and case study of aircraft maintenance. Computers & Industrial Engineering, 191:110169, 2024. doi: 10.1016/j.cie.2024.110169.

José Coelho and Mario Vanhoucke. An exact composite lower bound strategy for the resourceconstrained project scheduling problem. Computers & Operations Research, 93:135–150, 2018. doi: 10.1016/j.cor.2018.01.017.

Izack Cohen. Data-driven project planning: An integrated network learning, process mining, and constraint relaxation approach in favor of scheduling recurring projects. IEEE Transactions on Engineering Management, 71:7719–7729, 2024. doi: 10.1109/TEM.2024.3382727.

Erik Demeulemeester and Willy Herroelen. A branch-and-bound procedure for the multiple resourceconstrained project scheduling problem. Management Science, 38(12):1803–1818, 1992. doi: 10.1287/ mnsc.38.12.1803.

Ulrich Dorndorf, Erwin Pesch, and Toàn Phan-Huy. A branch-and-bound algorithm for the resourceconstrained project scheduling problem. Mathematical Methods of Operations Research, 52(3): 413–439, 2000. doi: 10.1007/s001860000091.

Weikang Guo, Mario Vanhoucke, and José Coelho. A prediction model for ranking branch-and-bound procedures for the resource-constrained project scheduling problem. European Journal of Operational Research, 306(2):579–595, 2023. doi: 10.1016/j.ejor.2022.08.042.

Peter E. Hart, Nils J. Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2):100–107, 1968. doi: 10.1109/TSSC.1968.300136.

H. P. Hillion and J.-M. Proth. Using timed petri nets for the scheduling of job-shop systems. Engineering Costs and Production Economics, 17(1-4):149–154, 1989. doi: 10.1016/0167-188X(89)90064-5.

Bo Huang, Mengchu Zhou, Xiaoyu Sean Lu, and Abdullah Abusorrah. Scheduling of resource allocation systems with timed petri nets: A survey. ACM Computing Surveys, 55(11):1–27, 2023. doi: 10.1145/3570326.

Toshihide Ibaraki. Branch-and-bound procedure and state-space representation of combinatorial optimization problems. Information and Control, 36(1):1–27, 1978. doi: 10.1016/S0019-9958(78) 90197-3.

Rainer Kolisch. Serial and parallel resource-constrained project scheduling methods revisited: Theory and computation. European Journal of Operational Research, 90(2):320–333, 1996. doi: 10.1016/ 0377-2217(95)00357-6.

Rainer Kolisch and Arno Sprecher. PSPLIB - A project scheduling problem library: OR Software - ORSEP Operations Research Software Exchange Program. European Journal of Operational Research, 96(1):205–216, 1997. doi: 10.1016/S0377-2217(96)00170-1.

Oumar Koné, Christian Artigues, Pierre Lopez, and Marcel Mongeau. Event-based MILP models for resource-constrained project scheduling problems. Computers & Operations Research, 38(1):3–13, 2011. doi: 10.1016/j.cor.2009.12.011.

Vipin Kumar and Laveen N. Kanal. A general branch and bound formulation for understanding and synthesizing and/or tree search procedures. Artificial Intelligence, 21(1–2):179–198, 1983. doi: 10.1016/S0004-3702(83)80009-5.

Jean-Marc Labat and Jean-Charles Pomerol. Are branch and bound and A<sup>∗</sup> algorithms identical? Journal of Heuristics, 9(2):131–143, 2003. doi: 10.1023/A:1022573412940.

Doo Yong Lee and Frank DiCesare. Scheduling flexible manufacturing systems using petri nets and heuristic search. IEEE Transactions on Robotics and Automation, 10(2):123–132, 1994. doi: 10.1109/70.282537.

H. Li, H. Zhu, L. Zheng, and F. Xie. Software project scheduling under activity duration uncertainty. Annals of Operations Research, 338(1):477–512, 2024. doi: 10.1007/s10479-023-05343-0.

Ido Lublin, Dor Atzmon, and Izack Cohen. Petri net induced heuristic search for resource constrained scheduling. Proceedings of the International Symposium on Combinatorial Search, 19(1):234–239, 2026. doi: 10.1609/socs.v19i1.43094.

Gonzalo Mejía and Nicholas G. Odrey. An approach using Petri nets and improved heuristic search for manufacturing system scheduling. Journal of Manufacturing Systems, 24(2):79–92, 2005. doi: 10.1016/S0278-6125(05)80009-3.

Thomas L. Morin and Roy E. Marsten. Branch-and-bound strategies for dynamic programming. Operations Research, 24(4):611–627, 1976. doi: 10.1287/opre.24.4.611.

Tadao Murata. Petri nets: Properties, analysis and applications. Proceedings of the IEEE, 77(4): 541–580, 1989. doi: 10.1109/5.24143.

Dana S. Nau, Vipin Kumar, and Laveen N. Kanal. General branch and bound, and its relation to A<sup>∗</sup> and AO<sup>∗</sup>. Artificial Intelligence, 23(1):29–58, 1984. doi: 10.1016/0004-3702(84)90004-3.

Terence Nazareth, Sanjay Verma, Subir Bhattacharya, and Amitava Bagchi. The multiple resource constrained project scheduling problem: A breadth-first approach. European Journal of Operational Research, 112(2):347–366, 1999. doi: 10.1016/S0377-2217(97)00402-5.

James H. Patterson. A comparison of exact approaches for solving the multiple constrained resource, project scheduling problem. Management Science, 30(7):854–867, 1984. doi: 10.1287/mnsc.30.7.854.

A. Alan B. Pritsker, Lawrence J. Watters, and Philip M. Wolfe. Multiproject scheduling with limited resources: A zero-one programming approach. Management Science, 16(1):93–108, 1969. doi: 10.1287/mnsc.16.1.93.

Jean-Marie Proth and Nathalie Sauer. Scheduling of piecewise constant product flows: A petri net approach. European Journal of Operational Research, 106(1):45–56, 1998. doi: 10.1016/ S0377-2217(98)00273-2.

Arno Sprecher. Scheduling resource-constrained projects competitively at modest memory requirements. Management Science, 46(5):710–723, 2000. doi: 10.1287/mnsc.46.5.710.12044.

Wil M. P. van der Aalst and Josep Carmona, editors. Process Mining Handbook. Springer, Cham, 2022. doi: 10.1007/978-3-031-08848-3.

H. R. Yousefzadeh and S. Harati Motlagh. Multi-mode resource constrained project scheduling in home health care: a DEA framework. Journal of the Operational Research Society, 77(8):2311–2329, 2026. doi: 10.1080/01605682.2025.2564768.

J. Zhang, L. Li, E. Demeulemeester, and H. Zhang. A three-dimensional spatial resource-constrained project scheduling problem: Model and heuristic. European Journal of Operational Research, 319(3): 943–966, 2024. doi: 10.1016/j.ejor.2024.07.018.

R. Čapek, P. Šůcha, and Z. Hanzálek. Production scheduling with alternative process plans. European Journal of Operational Research, 217(2):300–311, 2012. doi: 10.1016/j.ejor.2011.09.018.

## Appendices

## A Proofs

In this appendix, we present the proofs of the propositions in Section 4.4.

## A.1 Proof of Proposition 1

Proposition 1. The state graph $G = ( S , \mathbb { E } )$ and the TTPNR reachability graph $R G = ( S _ { R G } , \mathbb { E } _ { R G } )$ ， where $\cal { S } _ { R G }$ is the set of reachable timed states and $\mathbb { E } _ { R G }$ is the set of directed edges induced by transition firings, are isomorphic. That is, there exists a one-to-one correspondence between the states of G and the states of RG, $\phi : S  S _ { R G }$ , such that:

(i) $\phi ( s _ { 0 } )$ is the initial RG node represented by the timed marking $M _ { s _ { 0 } }$ and system time 0, where $M _ { s _ { 0 } }$ refines the ordinary initial marking $M _ { 0 }$

(ii) corresponding states have the same system time,

(iii) for every activity $j \in \mathcal V _ { i }$ , there is an edge $s \ { \stackrel { j } { \to } } \ s ^ { \prime }$ in G if and only if there is an edge $\phi ( s ) \xrightarrow { t _ { j } } \phi ( s ^ { \prime } )$ in RG with label $L ( t _ { j } ) = j$

Proof. For each $s \in S .$ , we define $\phi ( s ) = ( g ( s ) , M _ { s } )$ , where $M _ { s }$ is constructed below.

For each control-flow place $p \in P _ { C } \setminus \{ p _ { \mathrm { s t a r t } } , p _ { \mathrm { e n d } } \}$ , let $\bullet p = \{ t _ { i } \}$ and $p \bullet = \{ t _ { j } \}$ denote its unique input and output transitions. We set

$$
M _ { s } ( p ) = \left\{ \begin{array} { l l } { \{ 1 ^ { \tilde { d } _ { i } ( s ) } \} , } & { \mathrm { i f ~ } i \in \mathcal { V } ^ { o } ( s ) , } \\ { \{ 1 ^ { 0 } \} , } & { \mathrm { i f ~ } i \in \mathcal { V } ^ { c } ( s ) \mathrm { ~ a n d ~ } j \notin \mathcal { V } ^ { c } ( s ) \cup \mathcal { V } ^ { o } ( s ) , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{A.1}
$$

For the source place, $M _ { s } ( p _ { \mathrm { s t a r t } } ) = \{ 1 ^ { 0 } \} { \mathrm { i f ~ } } 0 \not \in \mathcal { V } ^ { c } ( s )$ , and ∅ otherwise. For the sink place, $M _ { s } ( p _ { \mathrm { e n d } } ) =$ $\{ 1 ^ { 0 } \}$ if $n \in \mathcal { V } ^ { c } ( s )$ , and ∅ otherwise.

Thus, if activity i is ongoing, then its output token toward each successor is present as a delayed token with residual duration $\tilde { d } _ { i } ( s )$ . If activity i has completed and successor $j$ has not yet started, then the corresponding place contains one immediately available token. Otherwise the place is empty.

For each resource place $p _ { r _ { k } } \in P _ { R }$ , the timed multiset $M _ { s } ( p _ { r _ { k } } )$ consists of $\begin{array} { r } { R _ { k } - \sum _ { i \in \mathcal { V } ^ { o } ( s ) } r _ { i k } } \end{array}$ tokens with residual duration 0 and, for each ongoing activity $i \in \mathcal { V } ^ { o } ( s )$ with $r _ { i k } > 0$ , exactly $r _ { i k }$ tokens with residual duration $\tilde { d } _ { i } ( s )$ . Hence $M _ { s } ( p _ { r _ { k } } )$ records both the resource units that are immediately available at time $g ( s )$ and the resource units that are currently tied to ongoing activities and return after their remaining processing times.

The construction above determines $\phi ( s )$ uniquely from the state tuple $( g ( s ) , \mathcal { V } ^ { c } ( s ) , \mathcal { V } ^ { o } ( s ) , \tilde { d } ( s ) )$ $\mathrm { B y }$ construction, $\phi ( s )$ has the same system time as s, so part $( i i )$ holds. Moreover, for the initial state $s _ { 0 }$ there are no completed or ongoing activities and all resources are available, hence $\phi ( s _ { 0 } )$ is the initial RG node represented by $M _ { s _ { 0 } }$ and system time 0, where $M _ { s _ { 0 } }$ is obtained from $M _ { 0 }$ by assigning zero residual availability time to every token, proving part (i).

Conversely, for the RCPSP-induced nets considered here, every reachable timed state determines the corresponding state-action information uniquely. Delayed control-flow tokens identify activities that have started but whose output places are not yet available, and their residual times determine the corresponding values $\tilde { d } _ { i } ( s )$ . Immediately available control-flow tokens indicate completed predecessors whose successor activities have not yet consumed the corresponding tokens. The resource-place tokens distinguish between immediately available resource units and resource units committed to ongoing activities. Starting from activity 0, whose status is determined by the source-place token, we recover the remaining activity statuses in topological order where an activity has started if and only if all its predecessors have completed and all its input control-flow places are empty. A started activity that is not ongoing is completed. Since only reachable states are considered, these token patterns arise from a unique combination of completed activities, ongoing activities, residual durations, and system time. Thus the correspondence is one-to-one on the reachable state space.

It remains to prove part (iii). By the construction of $M _ { s }$ , the control-flow tokens required by $t _ { j }$ possibly delayed, are present if and only if $j \in \mathcal { V } ^ { a } ( s )$ . Resource constraints afect only the earliest firing time. Thus, the selectable transitions correspond exactly to $\gamma ^ { a } ( s )$

In the state graph, the successor $s ^ { \prime }$ is obtained by advancing the clock by the smallest amount $\Delta = g ( s ^ { \prime } ) - g ( s ) \ge 0$ such that all predecessors of $j$ have completed and enough resources are available to start $j .$

Now consider the RG node $\phi ( s )$ , represented by the timed marking $M _ { s }$ and current system time $g ( s )$ . For every predecessor i of $j$ , the corresponding control-flow place on an input arc of $t _ { j }$ contains exactly one token: it is immediately available when $i \in \mathcal { V } ^ { c } ( s )$ , and it is delayed by $\tilde { d } _ { i } ( s )$ when $i \in \mathcal { V } ^ { o } ( s )$ Likewise, each resource place contains exactly the immediately available resource tokens together with the delayed return tokens induced by the ongoing activities in $s .$ Therefore, the smallest time increment after which all input control-flow and resource tokens required by $t _ { j }$ are available according to $M _ { s }$ is exactly the same value $\Delta$ as in the state-action transition rule.

After advancing time by $\Delta$ and firing $t _ { j }$ , the Petri-net token dynamics match the state-action update exactly:

• the required predecessor and resource tokens for $j$ are consumed,

• if $d _ { j } > 0$ , activity $j$ becomes ongoing, so its output control-flow tokens and its used resource tokens reappear as delayed tokens with residual duration $d _ { j }$ ; if $d _ { j } = 0$ , activity $j$ completes immediately and its output tokens and resources are immediately available,

• for every previously ongoing activity, residual durations are reduced by $\Delta$ (but not below 0),

• activities whose residual duration becomes zero leave $\mathcal { V } ^ { o }$ and join $\mathcal { V } ^ { c }$ , and their successor places now contain immediately available tokens unless these have already been consumed by already started successors.

Hence the resulting timed marking is precisely $M _ { s ^ { \prime } }$ , and the resulting RG node is $\phi ( s ^ { \prime } )$ . This proves that every edge $s \stackrel { j } { \longrightarrow } s ^ { \prime }$ in $G$ corresponds to an edge $\phi ( s ) \xrightarrow { t _ { j } } \phi ( s ^ { \prime } )$ in $R G$

The converse follows by the same argument. Every edge in RG is defined by choosing one transition $t _ { j }$ , advancing time by the smallest amount needed to make its input control-flow and resource tokens available, and then firing it. This is exactly the state-transition rule used in Section 4.2 for the scheduling decision $j$ . Therefore each edge in RG corresponds to a unique edge in $G$ with the same label. Thus, $\phi$ preserves and reflects labeled adjacency. Since both graphs start from corresponding initial states and generate exactly the same labeled successors from corresponding states, induction on path length establishes that $\phi ( S ) = S _ { R G }$ □

## A.2 Proof of Proposition 2

Proposition 2. Given an RCPSP, let $G$ be the state graph associated with the state-action representation. Then, any partial schedule $( \mathcal { I } , \{ S _ { i } \} _ { i \in \mathcal { I } } )$ produced by earliest-start generation corresponds to a path in the graph G leading to a state $s \in { \mathcal { S } }$ satisfying $g ( s ) = \mathrm { m a x } _ { i \in \mathcal { T } } S _ { i }$ and $\mathcal { V } ^ { o } ( s ) = \{ i \in \mathcal { I } : S _ { i } \leq$ $g ( s ) < S _ { i } + d _ { i } \} , \tilde { d } _ { i } ( s ) = S _ { i } + d _ { i } - g ( s ) > 0$ for all $i \in \mathcal { V } ^ { o } ( s )$ , and $\mathcal { V } ^ { c } ( s ) = \mathcal { I } \setminus \mathcal { V } ^ { o } ( s )$ . Moreover, any state $s \in { \mathcal { S } }$ corresponds to at least one partial schedule $( \mathcal { I } , \{ S _ { i } \} _ { i \in \mathcal { I } } )$ such that $\mathcal { I } = \mathcal { V } ^ { o } ( s ) \cup \mathcal { V } ^ { c } ( s )$ , and $S _ { i } = g ( s ) + \tilde { d } _ { i } ( s ) - d _ { i }$ for all $i \in \mathcal { V } ^ { o } ( s )$ and $S _ { i } \leq g ( s ) - d _ { i }$ for all $i \in \mathcal { V } ^ { c } ( s )$

Proof. Since $( \mathcal { I } , S _ { \mathcal { I } } )$ is produced by earliest-start generation, there exists a sequence $( i _ { 0 } = 0 , i _ { 1 } , \dots , i _ { k } )$ of the activities in $\mathcal { I }$ such that, for each $\ell \geq 1 , S _ { i \rho }$ is the earliest start time of $i _ { \ell }$ that is no earlier than $S _ { i _ { \ell - 1 } }$ and feasible with respect to $( \{ i _ { 0 } , \dots , i _ { \ell - 1 } \} , \{ S _ { i _ { 0 } } , \dots , S _ { i _ { \ell - 1 } } \} )$ , i.e., $S _ { i _ { \ell } }$ is given by (9). We denote this ordered list as $\mathcal { T } ^ { o } = ( i _ { 0 } = 0 , i _ { 1 } , \dots , i _ { k } )$ . We can therefore construct the path in $G$ starting from $s _ { 0 }$ by applying the scheduling decisions from ${ \mathcal { I } } ^ { o }$ in order. Specifically, at state $s _ { \ell }$ we apply decision $i _ { \ell }$ and proceed to state $s _ { \ell + 1 }$ by applying the state transition formulas. $\mathrm { B y }$ induction on $\ell \geq 1$ , using (5) and (6), each state $s _ { \ell }$ satisfies $g ( s _ { \ell } ) = S _ { i _ { \ell - 1 } }$ , and $\mathcal { V } ^ { o } ( s _ { \ell } )$ consists of the activities among $i _ { 0 } , \ldots , i _ { \ell - 1 }$ that are in process at $g ( s _ { \ell } )$ , with $\tilde { d } _ { i } ( s _ { \ell } ) = S _ { i } + d _ { i } - g ( s _ { \ell } ) > 0$ , and $\mathcal { V } ^ { c } ( s _ { \ell } ) = \{ i _ { 0 } , \dots , i _ { \ell - 1 } \} \setminus \mathcal { V } ^ { o } ( s _ { \ell } )$ . Hence the conditions in (4) coincide with those in (9), and $g ( s _ { \ell + 1 } ) = S _ { i _ { \ell } }$ . Moreover, we terminate the path in $s _ { k + 1 }$ , which must satisfy $g ( s _ { k + 1 } ) = S _ { i _ { k } } = \operatorname* { m a x } _ { j \in \mathcal { T } } S _ { j }$ . The definitions of $\mathcal { V } ^ { o } ( s _ { k + 1 } )$ and $\mathscr { V } ^ { c } ( s _ { k + 1 } )$ follow from the recursive transition.

Conversely, let $s \in S .$ , and let $( s _ { 0 } , s _ { 1 } , \ldots , s _ { k } = s )$ be a path from $s _ { 0 }$ to s on graph $G ,$ then by definition there exist scheduling decisions $( j _ { 0 } = 0 , j _ { 1 } , j _ { 2 } , \dots , j _ { k - 1 } )$ such that $j _ { \ell }$ induces the transition from $s _ { \ell }$ to $s _ { \ell + 1 }$ . Moreover, since $j _ { \ell }$ is feasible for state $s _ { \ell }$ the set $\{ j _ { 0 } , j _ { 1 } , \ldots , j _ { \ell } \}$ is a predecessor-closed set, and together with $S _ { j _ { \ell } } = g ( s _ { \ell + 1 } )$ it defines a partial schedule. Since each $S _ { j _ { \ell } }$ is the earliest feasible start time given $\left( \{ j _ { 0 } , \dots , j _ { \ell - 1 } \} , \{ S _ { j _ { 0 } } , \dots , S _ { j _ { \ell - 1 } } \} \right)$ , this partial schedule is produced by earliest-start generation, and the first part of the proof yields the stated relations between $( \mathcal { I } , S _ { \mathcal { I } } )$ and s. □

## B Empirical study

In this appendix, we present the LP relaxations used in the empirical study of Section 6 as well as additional computational results.

## B.1 Weak and strong LP relaxations

Both LP relaxations are derived from the time-indexed MILP formulation of Section 4.1, whose notation we adopt throughout. In both relaxations, the integrality requirements are relaxed, i.e., $y _ { i t } \in \{ 0 , 1 \}$ is replaced by $0 \leq y _ { i t } \leq 1$ and $S _ { i } \in  { \mathbb { N } } _ { 0 }$ by $S _ { i } \geq 0$ , while the objective, the assignment constraints, and the start-time definitions are kept unchanged. The two relaxations difer only in how the precedence relations are modeled.

Node-specific LP and resulting bounds Consider a B&B node $N _ { i }$ with partial schedule $( \mathcal { I } , S _ { \mathcal { I } } )$ and current time $g ( N _ { i } )$ . Instead of the full horizon $\mathcal { T } = \{ 0 , \ldots , H \}$ , the LP at $N _ { i }$ only considers the time window

$$
\mathcal { T } ( N _ { i } ) = \{ g ( N _ { i } ) , \ldots , \bar { F } \} ,\tag{B.1}
$$

which runs from the current time $g ( N _ { i } )$ , a lower bound on the start time of every unscheduled activity, to the global upper bound $\bar { F } .$ , i.e., the makespan of the incumbent. The upper bound $\bar { F }$ is initialized to the horizon H and is updated whenever a new incumbent is found. Since we are only interested in schedules that start no unscheduled activity before $g ( N _ { i } )$ and have makespan at most ${ \bar { F } } ,$ , this restriction does not cut of any schedule in the subtree of $N _ { i }$ that can match or improve the incumbent. Accordingly, the LP at $N _ { i }$ has decision variables $y _ { j t }$ only for unscheduled activities $j \in \mathcal { V } \backslash \mathcal { I }$ and start times $t \in \{ g ( N _ { i } ) , \ldots , \bar { F } - d _ { j } \}$ . All other values of $y _ { j t }$ are set to constants:

• for scheduled activities $j \in \mathcal { I }$ , whose start times $S _ { j }$ are given, $y _ { j t } = 1$ if $t = S _ { j }$ and $y _ { j t } = 0$ otherwise;

• for unscheduled activities $j \in \mathcal { V } \setminus \mathcal { I } , y _ { j t } = 0$ for $t < g ( N _ { i } )$ and for $t > F - d _ { j }$

Substituting these values into the LP afects the constraints as follows.

• Assignment constraints are only needed for unscheduled activities, and read $\begin{array} { r } { \sum _ { t = g ( N _ { i } ) } ^ { \bar { F } - d _ { j } } y _ { j t } = 1 } \end{array}$ for all $j \in \mathcal { V } \backslash \mathcal { I }$

• Resource constraints are imposed only for $t \in \mathcal T ( N _ { i } )$ , with the capacity reduced by the scheduled activities still in process at time $t , \mathrm { i . e . }$ , the right-hand side becomes $\begin{array} { r } { R _ { k } - \sum _ { j \in \mathcal { I } : S _ { j } \leq t < S _ { j } + d _ { j } } r _ { j k } } \end{array}$ For $t < g ( N _ { i } )$ , only scheduled activities are in process, so these constraints are satisfied by the feasibility of the partial schedule and can be omitted.

• Precedence constraints $( i , j ) \in \mathcal { E }$ with $i , j \in \mathcal { I }$ involve only constants, are satisfied by the partial schedule, and can be omitted. Since an activity is only scheduled after all of its predecessors have been scheduled, the case $i \notin \mathcal { I } , j \in \mathcal { I }$ cannot occur. Hence, only two cases remain: precedence relations between two unscheduled activities, $i , j \notin { \mathcal { I } }$ , and precedence relations in which only the predecessor is scheduled, $i \in \mathcal { I }$ and $j \not \in \mathcal { I }$ . In the latter case, the constraint reduces to a release time for $j .$ . The resulting constraints are given below for each relaxation.

The purpose of the time window is to reduce the size of the $\mathrm { L P }$ formulation, as the number of variables, resource constraints, and (for the strong LP) precedence constraints all grow linearly with the length of the window, so a shorter window yields a smaller LP that is faster to solve.

Let $z _ { \mathrm { L P } } ( N _ { i } )$ denote the optimal value of the LP at node $N _ { i }$ (weak or strong, see below). Since every start time in the window is at least $g ( N _ { i } )$ , we have $z _ { \mathrm { L P } } ( N _ { i } ) \geq g ( N _ { i } )$ , so the LP provides a lower bound $z _ { \mathrm { L P } } ( N _ { i } ) - g ( N _ { i } )$ on the remaining time to project completion. Because makespans are integral, this bound can be rounded up and combined with the critical-path bound $h ( N _ { i } )$ of Section 5, yielding

$$
h _ { \mathrm { L P } } ( N _ { i } ) = \operatorname* { m a x } \Bigl \{ h ( N _ { i } ) , \ \left\lceil z _ { \mathrm { L P } } ( N _ { i } ) \right\rceil - g ( N _ { i } ) \Bigr \} .\tag{B.2}
$$

In the LP-based variants, the evaluation function of Section 5 is replaced by $\hat { f } ( N _ { i } ) = g ( N _ { i } ) + h _ { \mathrm { L P } } ( N _ { i } )$ which is used both for node selection and for pruning. A node whose LP is infeasible is pruned as well. Moreover, if the optimal $\mathrm { L P }$ solution at $N _ { i }$ is integral, i.e., $y _ { j t } \in \{ 0 , 1 \}$ for all $j \in \mathcal { V } \backslash \mathcal { I }$ and $t \in \mathcal { T } ( N _ { i } )$ , it satisfies all constraints of the MILP formulation and, together with the partial schedule $( \mathcal { I } , S _ { \mathcal { I } } )$ , defines a feasible schedule with makespan $z _ { \mathrm { L P } } ( N _ { i } )$ . In this case, the upper bound is updated to $\bar { F } \gets \operatorname* { m i n } \{ \bar { F } , z _ { \mathrm { L P } } ( N _ { i } ) \}$ , the incumbent is updated accordingly, and $N _ { i }$ is pruned, since no schedule in its subtree can have a smaller makespan.

Weak LP relaxation The weak LP keeps the precedence constraints of the MILP formulation as they are, i.e., each precedence relation $( i , j ) \in \mathcal { E } \mathrm { ~ y ~ }$ ields the single aggregated constraint

$$
S _ { i } + d _ { i } \le S _ { j } \quad \Longleftrightarrow \quad \sum _ { t = 0 } ^ { \bar { F } - d _ { i } } ( t + d _ { i } ) y _ { i t } \le \sum _ { t = 0 } ^ { \bar { F } - d _ { j } } t y _ { j t } \qquad \forall ( i , j ) \in \mathcal { E } .\tag{B.3}
$$

Hence, the weak LP is the plain LP relaxation of the MILP formulation.

Strong LP relaxation The strong LP replaces (B.3) by one cumulative constraint per precedence relation and time point $t \in \mathcal T ( N _ { i } )$ : by any time t, activity $j$ may have started only to the extent that activity i has already completed. If i is already scheduled, this simply prohibits $j$ from starting before i completes. Hence, (B.3) is replaced by

$$
\begin{array} { l } { { \displaystyle \sum _ { \tau = g ( N _ { i } ) } ^ { t - d _ { i } } y _ { i \tau } ~ \geq ~ \sum _ { \tau = g ( N _ { i } ) } ^ { \operatorname* { m i n } \{ t , \bar { F } - d _ { j } \} } y _ { j \tau } } } \\ { { \displaystyle \sum _ { t = g ( N _ { i } ) } ^ { S _ { i } + d _ { i } - 1 } y _ { j t } ~ \leq ~ 0 } } \end{array}
$$

$$
\forall ( i , j ) \in \mathcal { E } , \ i , j \not \in \mathcal { I } , \ t \in \mathcal { T } ( N _ { i } ) ,\tag{B.4}
$$

$$
\forall ( i , j ) \in \mathcal { E } , \ i \in \mathcal { I } , \ j \not \in \mathcal { I } .\tag{B.5}
$$

Together with the assignment constraints, summing (B.4) over $t \in \mathcal { T } ( N _ { i } )$ yields (B.3) for $i , j \notin { \mathcal { I } }$ and (B.5) implies (B.3) for $i \in { \mathcal { I } } , j \notin { \mathcal { I } }$ , since it allows j to start only at or after the completion time $S _ { i } + d _ { i }$ of i. Hence, every solution of the strong LP satisfies the weak one and the strong bound is at least as large. The price is up to $\left| \mathcal { E } \right| \left| \mathcal { T } ( N _ { i } ) \right|$ precedence constraints instead of |E|.

## B.2 Additional computational results

This section complements the empirical study of Section 6. Comparing the B&B variants among themselves is not the main goal of the paper, which focuses on the comparison between B&B and $A ^ { * }$ The following results are therefore reported for the interested reader. Figure B.1 compares the final upper and lower bounds obtained by each method with the optimal makespan (for $A ^ { * }$ , both bounds coincide with the optimum on all instances), and Figure B.2 reports, for each pair of B&B variants, on how many instances one variant obtained a better upper or lower bound than the other. Since $A ^ { * }$ proves optimality on all instances, it is not included in the pairwise comparison. Here, UB denotes the final upper bound $\bar { F } , \mathrm { i . e . , }$ the makespan of the best schedule found, and LB the final lower bound, i.e., the smallest lower bound $\hat { f } ( N _ { i } )$ among the nodes that remain open when the search terminates.

Both figures show that DF B&B is better at obtaining upper bounds, whereas BF B&B is better at obtaining lower bounds. For instance, without LP relaxations, DF B&B obtains a better upper bound than BF B&B on 31 instances and a worse one on only 4, whereas BF B&B obtains a better lower bound on 39 instances and never a worse one. The LP relaxations generally deteriorate the upper bounds, since their computational overhead slows down the search and fewer incumbents are found within the time limit. For example, DF B&B without LP obtains a better upper bound than with the weak and strong LP on 66 and 82 instances, respectively, and a worse one on only 4 and 7 instances. The LP relaxations also do not improve the lower bounds of BF B&B, as the no-LP variant obtains a better lower bound than both LP variants on at least 98 instances. For DF B&B, in contrast, the LP relaxations present a trade-of, which is why their lower-bound profiles cross that of the no-LP variant. The no-LP variant proves optimality on more instances, since solving an LP at every node is two to three orders of magnitude slower, but the LP variants obtain considerably tighter lower bounds on the instances that remain unsolved. This trade-of is sharper for the strong LP, which proves optimality on 69 fewer instances than the no-LP variant, compared with 52 for the weak LP, but also yields tighter lower bounds. Specifically, on the 40 instances that the no-LP variant does not solve, which the LP variants do not solve either, the median relative gap between the optimum and the final lower bound drops from 39.0% without LP relaxations to 34.2% with the weak LP and 28.1% with the strong LP. On these instances, the final lower bound of DF B&B usually remains at the root bound, as shallow open nodes are never revisited. The strong LP therefore benefits mainly from its tighter root bound, whereas the weak LP also raises its bound above the root bound on about a third of these instances, as its cheaper LPs allow more nodes to be processed. This suggests that, for DF B&B, it may be preferable to solve the strong LP only once at the root, to obtain a global lower bound, and to use the cheaper critical-path bound at all other nodes. Such a hybrid would retain most of the lower-bound advantage of the strong LP while preserving the speed of the no-LP variant.

![](images/38b8053b0fe7718b4d3c600d833c51d2be269e4b826b1f1d8002a38c4c94525d.jpg)  
Figure B.1: Profiles of the final upper bounds (left) and lower bounds (right) over all 480 instances,  
Note: Bounds measured as the relative gaps (UB−optimum)/optimum and (optimum−LB)/optimum, respectively, in percent. Each curve gives the fraction of instances whose gap does not exceed the given value. Since A<sup>∗</sup> proves optimality on all instances, its curves are at one for every gap.

Table B.1 reports the instances for which optimality was not proven, partitioned by PSPLIB parameter subclass. The J30 instances are generated by varying three parameters (Kolisch and Sprecher, 1997). The network complexity (NC) is the average number of immediate successors per activity. The resource factor (RF) is the average fraction of the resources required by an activity, so that a higher RF means that each activity uses more of the resources. The resource strength (RS) measures how scarce the resources are relative to the demand, where a lower RS means more scarce resources and hence a more constrained instance. The unresolved instances concentrate in subclasses with low resource strength $( R S = 0 . 2 5 )$ and high resource factor $( R F \ge 0 . 7 5 )$ , and the LP-based variants leave considerably more instances unresolved than the variants without LP relaxations.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>55</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>82</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>52</td></tr><tr><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1></td></tr></table>

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>106</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>101</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>102</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>68</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>18</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1></td></tr></table>

Figure B.2: Pairwise comparison of the final upper bounds (left) and lower bounds (right) obtained by the B&B variants.

Note: A<sup>∗</sup> is not included, as it proves optimality on all instances. Each cell gives the number of instances on which the row method obtained a better bound than the column method; darker cells indicate that the row method dominates the column method on more instances.

Table B.1: Instances for which optimality was not proven, partitioned by PSPLIB parameter subclass.
<table><tr><td colspan="4"></td><td colspan="3">Best-first B&amp;B</td><td colspan="3">Depth-first B&amp;B</td></tr><tr><td>RS</td><td>RF</td><td>NC</td><td>A*</td><td>No LP</td><td>Weak LP</td><td>Strong LP</td><td>No LP</td><td>Weak LP</td><td>Strong LP</td></tr><tr><td>0.25</td><td>0.25</td><td>1.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>0.25</td><td>0.25</td><td>1.8</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>0.25</td><td>0.50</td><td>1.5</td><td>0</td><td>1</td><td>9</td><td>9</td><td>1</td><td>8</td><td>9</td></tr><tr><td>0.25</td><td>0.50</td><td>1.8</td><td>0</td><td>0</td><td>5</td><td>9</td><td>0</td><td>3</td><td>7</td></tr><tr><td>0.25</td><td>0.50</td><td>2.1</td><td>0</td><td>0</td><td>4</td><td>6</td><td>0</td><td>4</td><td>6</td></tr><tr><td>0.25</td><td>0.75</td><td>1.5</td><td>0</td><td>5</td><td>10</td><td>10</td><td>6</td><td>10</td><td>10</td></tr><tr><td>0.25</td><td>0.75</td><td>1.8</td><td>0</td><td>5</td><td>10</td><td>10</td><td>5</td><td>10</td><td>10</td></tr><tr><td>0.25</td><td>0.75</td><td>2.1</td><td>0</td><td>2</td><td>10</td><td>10</td><td>3</td><td>10</td><td>10</td></tr><tr><td>0.25</td><td>1.00</td><td>1.5</td><td>0</td><td>9</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>0.25</td><td>1.00</td><td>1.8</td><td>0</td><td>9</td><td>10</td><td>10</td><td>9</td><td>10</td><td>10</td></tr><tr><td>0.25</td><td>1.00</td><td>2.1</td><td>0</td><td>4</td><td>10</td><td>10</td><td>4</td><td>10</td><td>10</td></tr><tr><td>0.50</td><td>0.50</td><td>1.8</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>0.50</td><td>0.75</td><td>1.5</td><td>0</td><td>0</td><td>4</td><td>4</td><td>1</td><td>4</td><td>3</td></tr><tr><td>0.50</td><td>0.75</td><td>1.8</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>0.50</td><td>0.75</td><td>2.1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>0.50</td><td>1.00</td><td>1.5</td><td>0</td><td>0</td><td>5</td><td>6</td><td>1</td><td>6</td><td>6</td></tr><tr><td>0.50</td><td>1.00</td><td>1.8</td><td>0</td><td>0</td><td>5</td><td>5</td><td>0</td><td>3</td><td>4</td></tr><tr><td>0.50</td><td>1.00</td><td>2.1</td><td>0</td><td>0</td><td>3</td><td>4</td><td>0</td><td>1</td><td>4</td></tr><tr><td>0.75</td><td>0.75</td><td>1.5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>0.75</td><td>1.00</td><td>1.5</td><td>0</td><td>0</td><td>2</td><td>2</td><td>0</td><td>2</td><td>2</td></tr><tr><td>0.75</td><td>1.00</td><td>1.8</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>2</td></tr></table>

Note: Each (NC, RF, RS) subclass contains ten instances, so every entry is a count out of ten. Subclasses omitted from the table have zero unresolved instances for every method.