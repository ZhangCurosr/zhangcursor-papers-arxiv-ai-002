# Reward-Guided RRT\* for Budgeted Task and Motion Planning

Vin´ıcius Araujo de Oliveira´ <sup>1</sup>, Gabriel Ferreira Mendes Lourenc¸o<sup>1</sup>, Paulo L. J. Drews-Jr<sup>2</sup>, Armando Alves Neto<sup>3</sup>, and Douglas G. Macharet<sup>1</sup>

Abstract— Robotic missions involving spatially distributed tasks often require jointly selecting rewarding locations and planning feasible motions under limited travel budgets. This paper proposes Reward-Guided RRT\*, a new sampling-based framework that constructs a budget-feasible mission through a sequence of locally joint task-selection-and-motion-planning stages, evaluated directly in the continuous state space, without requiring a precomputed task-transition graph. The method incrementally builds a mission using reward-guided RRT\* searches, while a return-feasibility procedure ensures that each selected task can be visited without violating the mission budget. Experiments with increasing obstacle complexity and number of tasks show solutions comparable to VNS-PRM\* while requiring substantially fewer sampled states. We also extend the framework to Dubins steering and evaluate it in CoppeliaSim with a scaled car-like robotic platform, demonstrating its ability to incorporate vehicle kinematic constraints.

## I. INTRODUCTION

Many robotic applications require an autonomous vehicle to perform a collection of spatially distributed tasks while operating under limited resources. Examples include inspection, environmental monitoring, search and rescue, and data collection. In these scenarios, the vehicle must simultaneously determine where to perform tasks and how to move between them. Visiting additional task locations may increase the mission reward, but it also consumes the travel budget and can make it impossible to reach the final destination. This coupling between discrete task selection and continuous motion planning becomes particularly challenging in environments with obstacles, nonholonomic constraints, or other motion constraints.

In this context, the Orienteering Problem (OP) [1] provides a natural abstraction for this class of missions. Given a set of reward-bearing locations and a limited travel budget, the objective is to select and visit a subset of locations that maximizes the collected reward (Fig. 1). Classical OP methods, however, assume that the underlying graph and its travel costs are available beforehand [2]. In Robotics, building such a graph may require a large number of pairwise motion-planning queries and may provide only a discrete approximation of the continuous state space. Moreover, task selection and motion planning are typically treated as separate stages, despite their strong interdependence.

![](images/5f55957c55ba66f3c5ecf1801bf0d479b4903c8ea8c9f353731f0ab08c70f0d1.jpg)  
Fig. 1: Illustration of budgeted task and motion planning, jointly selecting reward-bearing tasks and collision-free paths to maximize reward under a limited travel budget.

This paper investigates an alternative sampling-based scheme in which promising task locations and feasible motions are discovered jointly. Rather than first constructing a complete roadmap and subsequently solving a discrete OP, we incrementally grow a search tree from the initial state while directing exploration toward reward-bearing locations. Each branch represents a feasible partial mission, including both a continuous path and the set of tasks visited along it. The resulting tree therefore serves as an implicit representation of the relevant task-transition graph.

This formulation offers a natural advantage over roadmapbased approaches, such as the Probabilistic Roadmap Method (PRM), for single-mission budgeted planning [3]. PRM is well suited to multi-query motion planning, but its roadmap construction generally separates the exploration of the state space from the subsequent task-selection process. In contrast, RRT-based approaches grow directly from the current mission state and can incorporate motion feasibility and constraints during expansion. Collision avoidance, vehicle kinematics, nonholonomic motion, and other local feasibility conditions can be directly enforced when generating each tree edge [4]. Furthermore, the tree can maintain the accumulated travel cost and explicitly account for the cost required to reach a prescribed terminal state or return to the initial state. This enables task selection and motion feasibility to be evaluated within a unified incremental planning process.

Here we propose the Reward-Guided RRT\*, a samplingbased approach for budgeted task and motion planning with predefined rewards. Rewards are associated with specific task locations and are collected only when those locations are visited. Tree expansion is biased toward promising unvisited tasks, while the accumulated travel cost and terminal feasibility are explicitly considered. RRT\* [5] rewiring is used to improve the motion cost of promising branches, allowing the planner to discover increasingly efficient paths and potentially visit additional reward-bearing locations within the available budget. The approach can support both fixed-goal missions and closed-circuit missions in which the vehicle must return to its initial state.

## II. RELATED WORK

The Orienteering Problem (OP) is a classical NP-hard combinatorial optimization problem that combines the selection and sequencing of locations under a limited travel budget. Given a set of reward-bearing vertices, the goal is to determine a path between prescribed endpoints that visits a subset of these vertices and maximizes the collected reward without exceeding the available budget. It was formally introduced by Golden et al. [1], motivating the development of both exact and heuristic solution methods. Comprehensive surveys of the OP and its variants are provided by Vansteenwegen et al. [6] and Gunawan et al. [2].

A broad range of methods has been proposed to solve the OP, including exact mixed-integer programming formulations, dynamic programming, constructive heuristics, local search, and metaheuristics [2]. These developments have led to numerous generalizations involving multiple agents [7], time-varying [8] or spatial-correlated [9] rewards, clustered targets [10], and other application-specific constraints [11], [12], [13]. Such variants demonstrate the flexibility of the OP framework, but most formulations still assume that the travel cost between candidate locations is known through a predefined graph or distance matrix.

Meanwhile, in Robotics, motion planning addresses the problem of finding feasible paths for a robot operating in a continuous state space while satisfying environmental and vehicle constraints. Classical approaches include graph-search methods, potential-field methods, samplingbased methods, and path optimization techniques. Among sampling-based methods, Probabilistic Roadmaps (PRMs) construct a graph of collision-free states and feasible local connections, while Rapidly-exploring Random Trees (RRTs) incrementally explore the state space from an initial state [14], [15]. The RRT\* algorithm introduced by Karaman and Frazzoli [5] extends this framework with rewiring operations and provides asymptotic optimality under suitable conditions. These methods are particularly attractive for high-dimensional spaces and problems involving complex geometric or kinodynamic constraints.

The integration of reward-based task selection and motion planning has received increasing attention, motivated by the need to account for environmental and vehicle constraints that cannot be captured by simple distances between reward locations. The Physical Orienteering Problem (POP), introduced by Peniˇ ckaˇ et al. [3], extends the classical OP to environments with obstacles, requiring selected targets to be connected by feasible collision-free trajectories. The Dubins Orienteering Problem (DOP) further incorporates curvature constraints into the vehicle model [16], while the Minimal Exposure Dubins Orienteering Problem considers both reward collection and exposure to a known sensor field [17]. More recently, the Kinematic Orienteering Problem (KOP) integrates reward-maximizing node selection with time-optimal trajectory generation under velocity and acceleration constraints for multirotor UAVs [18].

These works illustrate the broad range of environmental objectives, vehicle models, and motion constraints that can be incorporated into robotic orienteering. However, although the literature establishes a clear connection between the OP and motion planning, existing approaches commonly retain a decomposition in which candidate task locations and their transitions are first represented in a discrete problem, while motion planning is subsequently used to validate or realize the selected route. In contrast, our proposed approach uses RRT\*-based trees to incrementally explore the continuous state space while simultaneously guiding the search toward rewarding task locations. Rather than introducing another problem-specific variant of the Orienteering Problem, we propose a general sampling-based planning framework that jointly accounts for task selection, path feasibility, motion constraints, and budget requirements.

## III. PROBLEM FORMULATION

We consider a vehicle operating in bounded environments with obstacles and a set of reward-bearing locations. Our goal is to compute a feasible path that maximizes the total reward collected by visiting a subset of these locations while satisfying a limited travel budget. The problem can be viewed as a motion-constrained extension of the Orienteering Problem, in which the reward-bearing locations are known, but the feasible paths and their associated costs are not explicitly available beforehand.

Let the environment be represented by a state space X and its collision-free subset $\mathcal { X } _ { \mathrm { f r e e } } \subseteq \mathcal { X } .$ . Let $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ denote the set of reward-bearing locations, where each location $v _ { i } \in \mathcal { X } _ { \mathrm { f r e e } }$ is associated with a non-negative reward $r _ { i } .$ The vehicle starts at position $\vec { x } _ { \mathrm { s t a r t } }$ and must reach a terminal position $\vec { x } _ { \mathrm { g o a l } }$ . The terminal position may be distinct from the initial position or may coincide with it, corresponding to a closed-circuit mission.

A possible solution is a continuous path $\sigma : [ 0 , 1 ]  \mathcal { X } _ { \mathrm { f r e e } }$ satisfying the vehicle’s motion constraints, with $\sigma ( 0 ) \ =$ $\vec { x } _ { \mathrm { s t a r t } }$ and $\sigma ( 1 ) = \vec { x } _ { \mathrm { g o a l } }$ . Also, let $c ( \sigma )$ denote the cost of this path, typically its length or travel time, and let B denote the available budget, such that σ must satisfy $c ( \sigma ) \leq B$

Now, let $r _ { g }$ denote the visitation radius associated with v . A reward location is considered visited if the path enters its corresponding visitation region. Accordingly, the set of locations visited by σ is defined as

$$
\mathcal { V } ( \sigma ) = \left\{ v _ { i } \in \mathcal { V } \mid \operatorname* { m i n } _ { t \in [ 0 , 1 ] } d \left( \sigma ( t ) , v _ { i } \right) \leq r _ { g } \right\} ,
$$

where $d ( \cdot , \cdot )$ is an appropriate distance metric in the state

space. The goal is to maximize the total collected reward:

$$
\begin{array} { r l r } { \displaystyle \operatorname* { m a x } _ { \sigma } } & { \displaystyle R ( \sigma ) = \sum _ { v _ { i } \in \mathcal { V } ( \sigma ) } r _ { i } , \quad \mathrm { s u b j e c t ~ t o } } & \\ & { } & \\ { \displaystyle \quad \sigma ( t ) \in \mathcal { X } _ { \mathrm { f r e e } } , \quad \forall t \in [ 0 , 1 ] , } & \\ & { \displaystyle c ( \sigma ) \leq B , } & \\ & { \displaystyle \sigma ( 0 ) = \vec { x } _ { \mathrm { s t a r t } } , \quad \sigma ( 1 ) = \vec { x } _ { \mathrm { g o a l } } . } & \end{array}
$$

Unlike the classical graph-based formulation of the OP, the cost of traveling between two reward locations is not assumed to be known. In particular, for two locations $v _ { i }$ and $v _ { j } .$ , the existence and cost of a feasible connection depend on the environment and the vehicle’s motion constraints. Thus, the underlying task-transition graph is not explicitly provided. A feasible transition can be represented by a motion-planning problem,

$$
\begin{array} { c } { { \sigma _ { i j } ^ { * } = \arg \underset { \sigma } { \operatorname* { m i n } } c ( \sigma ) , \quad \mathrm { s u b j e c t ~ t o } \hfill } } \\ { { \sigma ( 0 ) = v _ { i } , \quad \sigma ( 1 ) = v _ { j } , \quad \sigma ( t ) \in \mathcal { X } _ { \mathrm { f r e e } } , } } \end{array}
$$

together with the vehicle-specific motion constraints. However, computing such connections for all pairs of reward locations would require constructing the complete underlying graph, which may be computationally expensive and unnecessary for a single budget-constrained mission.

The central challenge addressed in this work is to jointly determine the subset and ordering of reward locations to visit while constructing feasible connections between them. Rather than assuming a precomputed transition graph, the proposed method incrementally explores the continuous state space and discovers relevant feasible task transitions during planning. The resulting search must balance three competing objectives: collecting additional reward, maintaining motion feasibility, and preserving sufficient budget to reach the required terminal position.

## IV. METHODOLOGY

## A. Algorithm Overview

The proposed method constructs a budget-feasible mission incrementally through a sequence of reward-guided RRT\* searches. At each stage, an outgoing tree $\tau _ { \mathrm { { o u t } } }$ is rooted at the endpoint of the currently confirmed mission and explores the collision-free state space in search of an unvisited reward-bearing location. When the outgoing tree reaches a candidate location, the corresponding path is not immediately incorporated into the mission. Instead, a return tree $\mathcal { T } _ { \mathrm { r e t } }$ is initialized at the candidate state and attempts to connect it to the prescribed terminal state, which in the closed-circuit case coincides with $\vec { x } _ { \mathrm { s t a r t } }$ . The candidate is committed only if the complete outgoing-and-return path satisfies the available travel budget.

After a successful validation, the outgoing segment is appended to the confirmed mission, the candidate reward is collected, and a new outgoing tree is initialized at the newly visited location. The new tree inherits the travel cost and reward accumulated by the mission. Therefore, the planner does not construct a single tree containing all possible task sequences. Instead, it generates a sequence of local trees that incrementally extend one feasible mission. Figure 2 illustrates this process. The solid green path represents the mission already committed to by the planner, whereas the gray branches represent the current outgoing search tree. Once a candidate is validated, the corresponding branch becomes part of the confirmed mission, and the next tree is rooted at that location.

Algorithm 1 summarizes the high-level procedure, while Algorithms 2 and 3 detail the outgoing-tree expansion and return-validation routines, respectively.

Algorithm 1: Reward-Guided Incremental Planning   
Input: state ${ \vec { x } } _ { \mathrm { s t a r t } } ,$ reward locations V, budget B   
Output: confirmed mission $\mathcal { P }$   
1 $\mathcal { P }  [ \vec { x } _ { \mathrm { s t a r t } } ] , \mathcal { V } _ { \mathrm { v i s i t e d } }  \emptyset , \mathcal { V } _ { \mathrm { f a i l e d } }  \emptyset ;$   
2 ${ \vec { x } } _ { \mathrm { r o o t } }  { \vec { x } } _ { \mathrm { s t a r t } } ;$   
3 $C _ { \mathrm { r o o t } }  0 , R _ { \mathrm { r o o t } }  0 ;$   
4 T<sub>out</sub> ← InitializeTree(⃗x<sub>root</sub>, C<sub>root</sub>, R<sub>root</sub>);   
5 while PlanningActive() do   
6 (g, ${ \vec { x } } _ { g } ,$ status) ←   
GrowOutgoingTree $( \tau _ { \mathrm { o u t } } , \mathcal { V } _ { \mathrm { v i s i t e d } } , \mathcal { V } _ { \mathrm { f a i l e d } } , B ) ;$   
7 if status = NoCandidate then   
8 break;   
9 (success, $\mathcal { P } _ { g } , \mathcal { P } _ { \mathrm { r e t } } ) $   
ValidateReturn $( \mathcal T _ { \mathrm { o u t } } , \vec { x } _ { g } , \vec { x } _ { \mathrm { g o a l } } , B )$   
10 if success then   
11 P ← CommitSegment(P, P<sub>g</sub>);   
12 $\mathcal { V } _ { \mathrm { v i s i t e d } }  \mathcal { V } _ { \mathrm { v i s i t e d } } \cup \{ g \} ;$   
13 (P, P<sub>ret</sub>) ← RefinePaths(P, P<sub>ret</sub>);   
14 $\vec { x } _ { \mathrm { r o o t } }  \vec { x } _ { g } ;$   
15 $C _ { \mathrm { r o o t } }  \subset \mathrm { c o s t } ( \mathcal { P } ) , R _ { \mathrm { r o o t } }  \sum \quad r _ { i } ;$   
v<sub>i</sub>∈V<sub>visited</sub>   
16 $\tau _ { \mathrm { o u t } } $   
InitializeTree(⃗x<sub>root</sub>, C<sub>root</sub>, R<sub>root</sub>);   
17 else   
18 V<sub>failed</sub> ← V<sub>failed</sub> ∪ {g};   
19 return P;

## B. Outgoing Tree Expansion

The outgoing tree is responsible for discovering the next task to be incorporated into the mission. At each planning stage, a new tree is rooted at the last confirmed task (or at the initial state during the first iteration) and expanded until either a new candidate task is reached or no additional feasible candidates can be identified.

Unlike conventional graph-based formulations of the Orienteering Problem (OP), the proposed method explores the continuous collision-free state space by incrementally growing an $\mathrm { R R T ^ { * } }$ . Each expansion follows the standard samplingbased planning procedure, consisting of sampling, nearestneighbor search, steering, parent selection, and rewiring [5]. The main adaptations are (i) a reward-guided sampling strategy that biases the exploration toward promising task locations, (ii) a budget feasibility test that prevents the generation of infeasible branches, and (iii) the detection of candidate tasks that trigger the return-validation phase. Note that because each tree is grown for a finite iteration budget I and is subsequently frozen once a task is committed, the classical asymptotic-optimality guarantee of RRT\* [5] does not directly apply to the overall mission; rewiring here serves as an anytime cost-improvement mechanism within each stage’s finite budget, not as a source of global optimality guarantees. Algorithm 2 summarizes one iteration of the outgoing-tree expansion. The following subsections describe the components that differ from the standard RRT\*.

![](images/f81f882c7807e1d882bb28a3bf70913e0fc7f0dd9bf2de48b4e3be257882712c.jpg)  
Fig. 2: The outgoing tree first searches from the initial state and reaches task $v _ { 1 } .$ . After the candidate is validated, its branch is committed to the route and the next outgoing tree is rooted at $v _ { 1 }$ . The same procedure is repeated when $v _ { 2 }$ is confirmed. Gray edges denote the current search tree, while green edges denote the already committed ones.

Algorithm 2: GrowOutgoingTree(·)   
Input: outgoing tree T , visited tasks $\mathcal { V } _ { v i s i t e d } .$ , failed   
tasks $\nu _ { f a i l e d } ,$ , budget B   
Output: candidate task g, candidate state ${ \vec { x } } _ { g } ,$ status   
1 while planning budget available do   
2 ⃗x<sub>rand</sub> ← RewardGuidedSample(V \   
(V<sub>visited</sub> ∪ V<sub>failed</sub>));   
3 ⃗x<sub>nearest</sub> ← Nearest(T , ⃗x<sub>rand</sub>);   
4 ⃗x<sub>new</sub> ← Steer(⃗x<sub>nearest</sub>, ⃗x<sub>rand</sub>);   
5 if ObstacleFree (⃗x<sub>nearest</sub>, ⃗x<sub>new</sub>) then   
6 X<sub>near</sub> ← Near(T , ⃗x<sub>new</sub>);   
7 ⃗x<sub>parent</sub> ← ChooseParent(X<sub>near</sub>, ⃗x<sub>new</sub>);   
8 if BudgetFeasible (⃗x<sub>parent</sub>, ⃗x<sub>new</sub>, B)   
then   
9 AddVertex(T , ⃗x<sub>new</sub>);   
10 AddEdge(T , ⃗x<sub>parent</sub>, ⃗x<sub>new</sub>);   
11 Rewire(T , ⃗x<sub>new</sub>, X<sub>near</sub>);   
12 if TaskReached (⃗x<sub>new</sub>) then   
13 g ← ReachedTask(⃗x<sub>new</sub>);   
14 ⃗x<sub>g</sub> ← ⃗x<sub>new</sub>;   
15 return (g, ⃗x<sub>g</sub>, Candidate);   
16 return (∅, ∅, NoCandidate);

1) Reward-Guided Sampling: To balance exploration and exploitation, the sampling strategy combines uniform random exploration with goal-directed sampling. With probability $p _ { r } ,$ the algorithm samples a random state from the collision free workspace, preserving the exploratory behavior of the original RRT\*. Otherwise, the algorithm selects one of the unvisited task locations according to a roulette-wheel strategy.

Let $r _ { i }$ denote the reward associated with task v and let $d _ { i }$ represent its Euclidean distance to the root of the current outgoing tree. The sampling weight assigned to $v _ { i }$ is $w _ { i } = r _ { i } / d _ { i }$ , which favors nearby high-reward tasks while still allowing every candidate to be selected. The probability of selecting task $v _ { i }$ is therefore

$$
P ( v _ { i } ) = { \frac { w _ { i } } { \sum _ { j } w _ { j } } } .\tag{1}
$$

This mechanism continuously biases the tree toward promising regions of the workspace without sacrificing the global exploration capability inherited from the RRT\*.

2) Budget-Constrained Expansion: Each node stores the cumulative travel cost from the root of the current mission. After selecting the parent node, the accumulated cost of the new state is updated as

$$
C ( \vec { x } _ { \mathrm { { n e w } } } ) = C ( \vec { x } _ { \mathrm { { p a r e n t } } } ) + c ( \vec { x } _ { \mathrm { { p a r e n t } } } , \vec { x } _ { \mathrm { { n e w } } } ) ,\tag{2}
$$

where $c ( \cdot )$ denotes the Euclidean length of the newly added edge. Nodes whose accumulated travel cost exceeds the available mission budget are immediately discarded. Consequently, every branch in the outgoing tree satisfies the travelbudget constraint, preventing the planner from expanding toward regions it cannot reach within the remaining mission budget.

3) Candidate Task Detection: After inserting a new node into the tree, the planner verifies whether the new state lies inside the capture region of an unvisited task. A task $v _ { i }$ becomes a candidate whenever

$$
\| \vec { x } _ { \mathrm { n e w } } - v _ { i } \| < r _ { g } ,\tag{3}
$$

where $r _ { g }$ denotes the task capture radius.

Reaching a task does not immediately incorporate it into the mission. Instead, the outgoing tree is temporarily suspended, and a second RRT\* is initialized from the candidate task to verify whether a feasible return to the $\vec { x } _ { \mathrm { s t a r t } }$ still exists within the available travel budget. Only tasks that satisfy this validation are permanently incorporated into the mission.

## C. Return Feasibility Validation

Reaching a task with the outgoing tree does not guarantee that the corresponding task can be incorporated into the mission. Since the proposed planner constructs paths subject to a limited travel budget, every candidate task must also admit a feasible return to the goal state (Alg. 3).

Algorithm 3: ValidateReturn(·)   
Input: outgoing tree $\mathcal { T } _ { o u t } .$ , candidate state ${ \vec { x } } _ { g } ,$ goal   
state ${ \vec { x } } _ { \mathrm { g o a l } } .$ remaining budget B   
Output: success flag, outgoing path $P _ { g } ,$ return $P _ { r e t }$   
1 $\mathcal { T } _ { r e t }  \mathrm { I n i t i a l i z e T r e e } ( \vec { x } _ { g } ) ;$   
2 while return planning active do   
3 ⃗x<sub>rand</sub> ← RewardGuidedSample(⃗x<sub>goal</sub>);   
4 ${ \vec { x } } _ { 1 }$ <sub>nearest</sub> ← Nearest(T<sub>ret</sub>, ⃗x<sub>rand</sub>);   
5 ⃗x<sub>new</sub> ← Steer(⃗x<sub>nearest</sub>, ⃗x<sub>rand</sub>);   
6 if ObstacleFree (⃗x<sub>nearest</sub>, ⃗x<sub>new</sub>) then   
7 $\mathcal X _ { n e a r } \gets \mathrm { N e a r } ( \mathcal T _ { r e t } , \vec { x } _ { \mathrm { n e w } } ) ;$   
8 ⃗x<sub>parent</sub> ← ChooseParent $( \mathcal { X } _ { n e a r } , \vec { x } _ { \mathrm { n e w } } ) ;$   
9 if BudgetFeasible (⃗x<sub>parent</sub>, ⃗x<sub>new</sub>, B)   
then   
10 AddVertex $( T _ { r e t } , \vec { x } _ { \mathrm { n e w } } ) ;$   
11 AddEdge $( T _ { r e t } , \vec { x } _ { p a r e n t } , \vec { x } _ { \mathrm { n e w } } ) ;$   
12 Rewire $( T _ { r e t } , \vec { x } _ { \mathrm { n e w } } , \mathcal { X } _ { n e a r } ) ;$   
13 if ReachedGoal $( \vec { x } _ { \mathrm { n e w } } )$ then   
14 $P _ { g } \gets \mathtt { E x t r a c t P a t h } ( \mathcal { T } _ { o u t } , \vec { x } _ { g } ) ;$   
15 $P _ { r e t } \gets \mathrm { E x t r a c t P a t h } ( \mathcal { T } _ { r e t } , \vec { x } _ { \mathrm { n e w } } ) ;$   
16 return (True, $P _ { g } , P _ { r e t } ) ;$   
17 return (False, None, None);

To verify this condition, the planner initializes a second RRT\* rooted at the candidate task and expands it toward the initial state. The return tree follows the same expansion procedure adopted by the outgoing planner, including nearestneighbor search, steering, parent selection, and rewiring. However, unlike the outgoing phase, the reward-guided sampling is performed using only the goal state, because the objective is no longer to discover additional tasks, but solely to determine the feasibility of the return path.

The accumulated travel cost stored in the root of the return tree corresponds to the travel cost already incurred by the outgoing mission. Consequently, each node in the return tree represents the total travel cost of completing the mission through that partial return path. Nodes exceeding the available mission budget are immediately discarded. Whenever the return tree reaches the initial state, the candidate task is considered feasible. The outgoing path segment leading to the task is permanently incorporated into the mission, the return path is stored as the current feasible terminal path, and the planner initializes a new outgoing tree rooted at the newly confirmed task.

Conversely, if the return tree fails to connect the candidate task to the goal state within the prescribed budget, the task is temporarily classified as unavailable until a new location is selected. The planner then resumes the outgoing-tree exploration in search of alternative task locations.

## D. Path Refinement

The paths generated by the outgoing and return trees are feasible but not necessarily locally optimal. Since both trees are constructed incrementally, the resulting paths may contain unnecessary intermediate nodes and detours introduced by the sampling process. Therefore, after each successful task insertion, a post-processing stage reduces the overall travel distance before the next planning stage begins.

The first refinement consists of a greedy shortcut procedure [4]. Starting from the end of each path segment, the algorithm iteratively attempts to replace a sequence of consecutive edges with a single straight-line connection. A shortcut is accepted whenever the resulting segment is collision-free and reduces the accumulated travel distance. The procedure is repeated until no additional local improvements can be found. Since only collision-free connections are accepted, path feasibility is always preserved.

After the shortcut stage, a 2-opt local optimization is applied to the complete mission [19]. The algorithm evaluates alternative edge pairings by reversing intermediate path segments whenever such a modification reduces the total travel distance while maintaining collision-free connections. Unlike the greedy shortcut, which removes unnecessary intermediate nodes, the 2-opt operator optimizes the ordering of consecutive path segments and eliminates local geometric inefficiencies that remain after the shortcut stage.

Both refinement procedures operate exclusively on the geometry of the confirmed mission and do not modify the set of visited tasks. Consequently, the collected reward remains unchanged while the resulting path becomes shorter, increasing the remaining travel budget available for subsequent planning stages.

## V. EXPERIMENTS

## A. Illustrative Examples

To evaluate our Reward-Guided RRT\* motion planning framework, we consider three 1000m × 1000m scenarios with increasing difficulty and different sets of reward-bearing locations, hereafter referred to as goals. All scenarios share the same initial and final position, [500, 500], resulting in a closed-circuit mission. We categorize them as LOW, MEDIUM, and HIGH, reflecting increasing obstacle density and complexity, as well as an increasing number of goals. Task positions and reward values are randomly generated for each scenario. Figure 3 shows the best solution over 30 runs for each scenario, defined as the solution with the highest reward and, among those, the lowest budget.

![](images/bdad4f33c7da25ac4cfe93769c83e5570f9a70d226d39c54efdf33f2eca74f0b.jpg)  
(a) LOW

![](images/b86a67170d5e5977d200f1f85a55dab3f9617b27668fd54a8992891eeb95b7c3.jpg)  
(b) MEDIUM

![](images/97d5c63500eab2a297897707bac2f41b507173c718d575a167a3d42049167eb0.jpg)  
(c) HIGH  
Fig. 3: Example solutions for the (a) LOW, (b) MEDIUM, and (c) HIGH scenarios, with 15, 25, and 40 goals, respectively. All results use $I = 2 5 0 0 0$ and $d = 1 0 \mathrm { m } ;$ each result shows the best run (maximum reward) obtained.

## B. Quantitative Analysis

To quantitatively evaluate the performance and stability of our proposed approach, we conducted a systematic set of experiments, varying the main planning parameters and scenario difficulty. For each state, we ran 30 independent trials, varying the maximum number of tree expansion iterations I, the steering step distance d, and the scenario difficulty. The evaluated values were I ∈ {1500, 5000, 10000, 15000, 25000}, $d ~ \in ~ \{ 1 0 , 2 5 , 5 0 , 1 0 0 \}$ and difficulty levels LOW, MEDIUM, and HIGH. Figure 4 summarizes the overall results and these trends.

Statistical analysis of the mean budget utilization B% and mean collected reward R% shows relatively low variability across runs while preserving the method’s non-deterministic behavior. The results also show the expected behavior: more challenging scenarios, characterized by more goals and increased obstacle complexity, generally require more iterations to obtain higher-quality solutions and better exploit the available budget. Conversely, a larger steering step can improve performance when the maximum number of iterations is limited, as it lets the tree explore the environment more quickly.

Interestingly, the results indicate that, beyond approximately 10,000 iterations, the performance tends to stabilize, with little to no further improvement in either budget utilization or collected reward, regardless of the steering step distance. This suggests the existence of a practical stability point beyond which increasing the number of iterations provides diminishing returns. Due to the non-deterministic nature of the sampling-based approach, however, solutions of higher quality may still occasionally be obtained beyond this stability point. In fact, all of the best solutions observed across the experiments were obtained in runs beyond this point. This behavior is consistent with the trade-off inherent to RRTbased planning: smaller steering steps with sufficiently large iteration budgets allow the tree to explore the state space more finely, potentially producing more sophisticated and precise paths, but at the cost of increased computational time. Therefore, the observed stability point provides a practical criterion for determining when further increases in computational effort are unlikely to yield significant improvements in solution quality.

## C. State-of-the-art Comparison

To assess the performance of our Reward-Guided RRT\*, we compare it with VNS-PRM\* [3], which represents the closest approach to our method. Both techniques address the integration of reward-based route selection and motion planning in environments with obstacles using samplingbased strategies. However, they differ fundamentally in how they use sampling. The VNS-PRM\* relies on a classical Probabilistic Roadmap to build a dense representation of the free state space, while our approach incrementally grows an RRT\* tree from the current mission state. In particular, the classical PRM formulation does not directly incorporate motion constraints into the sampling and connection process as an RRT-based planner can, allowing our approach to naturally account for motion feasibility during tree expansion in Sec. V-D.

For a fair comparison, we evaluated VNS-PRM\* in each proposed scenario and compared its resulting solutions with the best solutions obtained by Reward-Guided RRT\*, as reported in Table I. The best solutions obtained with Reward-Guided RRT\* used I = 25000 iterations and a steering step of d = 10 (Figure 3).

TABLE I: Comparison of the best results obtained with Reward-Guided RRT\* and VNS-PRM\* [3].
<table><tr><td rowspan="2"></td><td colspan="2">Reward-Guided RRT*</td><td colspan="2">VNS-PRM* [3]</td></tr><tr><td>%B</td><td>%R</td><td>%B</td><td>%R</td></tr><tr><td>LOW</td><td>0.690</td><td>1</td><td>0.668</td><td>1</td></tr><tr><td>MEDIUM</td><td>0.986</td><td>1</td><td>0.896</td><td>1</td></tr><tr><td>HIGH</td><td>0.966</td><td>0.929</td><td>0.987</td><td>0.967</td></tr></table>

Our approach achieved the same collected reward as VNS-PRM\* in both LOW and MEDIUM scenarios, while collecting only 4% less in the HIGH. However, a key aspect of this comparison is the sampling density each method requires. The VNS-PRM\* solutions are obtained from a substantially denser set of samples, resulting in a considerably larger roadmap than the number of nodes generated by Reward-Guided $\mathrm { R R T ^ { * } }$ during a single run. Figure 5 and Table II provide a visual and quantitative comparison of the number of samples/nodes used by the two approaches. This highlights an important trade-off: VNS-PRM\* can achieve slightly better solutions in the most challenging scenario through a much denser state-space representation, whereas Reward-Guided RRT\* obtains comparable solutions with substantially fewer generated nodes and without requiring the construction of a dense roadmap.

![](images/ced21d33429285406c503b06f8387305a9ab196362fee5821526377f130006a1.jpg)  
Fig. 4: Overall statistical analysis of the proposed Reward-Guided RRT\*. Each curve represents the mean over 30 runs for a fixed steering step $d ,$ with different steps indicated by distinct markers. Shaded regions represent one standard deviation.

TABLE II: Number of final-tree nodes generated by Reward-Guided RRT\* and samples used by VNS-PRM\* [3].
<table><tr><td></td><td>Reward-Guided RRT*</td><td>VNS-PRM* [3]</td></tr><tr><td>LOW</td><td>1230</td><td>13229</td></tr><tr><td>MEDIUM</td><td>2925</td><td>21803</td></tr><tr><td>HIGH</td><td>1695</td><td>57153</td></tr></table>

## D. Kinematic Constraints

Finally, to demonstrate the proposed framework’s ability to incorporate vehicle kinematic constraints, we extended the planner using Dubins curves as the local steering method [20]. The vehicle state is represented as $\vec { x } ~ = ~ ( x , y , \theta )$ where $( x , y )$ is the car position and θ is its orientation. Connections between RRT\* nodes are generated subject to a minimum turning radius $R _ { \mathrm { m i n } }$ , such that $| \kappa ( s ) | \leq 1 / R _ { \mathrm { m i n } }$ Consequently, edge costs correspond to Dubins-path lengths, and collision checking is performed along the resulting curved paths. The same steering constraint is applied to both the outward and return trees, ensuring that the complete budget-feasible mission is kinematically feasible.

The resulting path was evaluated in CoppeliaSim<sup>1</sup> using the scaled car-like robotic platform shown in Fig. 6(a). The vehicle has a mass of approximately 5 kg and $R _ { \mathrm { m i n } } \approx 2$ m. During the experiment, the vehicle was commanded to travel at $\mathrm { 1 m / s }$ with a travel budget of 250 meters. About 61% of the total reward was collected in this scenario. Fig. 6(b) shows the planned route and resulting execution of the mission while respecting the vehicle’s turning constraint.

![](images/1cb4d9a68b71328fa08d642c68797e0aba0e2863a804ebaebdb26b75cde82758.jpg)

![](images/77469bfd840de6f635aa289522891af6fd25bc42ed59b29f3c9794a939ba0879.jpg)

(a) LOW  
![](images/c75ff85969310eb5636f9080b3c5bae66a7f486a9963896791bcf87eeda314ba.jpg)

![](images/1510fe599688b7b0d062133ae299da3bdea42a0e52f11dc04dc4d5e0e9ebfb61.jpg)

(b) MEDIUM  
![](images/28277b634413e23b0dd51adb40fb7b9189f5c6141d7ddb42fd1951feb84c9e26.jpg)

![](images/254d8dbb99d183c33135d9f94dada06cf4aecdce3acce935721f7cc21fc1a98d.jpg)  
(c) HIGH  
Fig. 5: Comparison of the final states obtained by Reward-Guided $\mathrm { R R T ^ { * } }$ (left) and VNS-PRM\* [3] (right).

![](images/74a57b23bda0b53a7706bc3d9bb606d62e01b223ab380f509832674035409397.jpg)  
(a) Scaled car-like platform.

![](images/e99a79cf32e132b9c88b87acf1bd1a6a1ec361bf95c91334f4777baa992afc46.jpg)  
(b) Mission path.

![](images/8d542d292609bcaf13d50850a5323e06c57976fc04e5993d51a7b9c5510e03af.jpg)  
(c) Simulation.  
Fig. 6: Kinematic evaluation of the planned mission in CoppeliaSim. (a) Scaled car-like robotic platform used in the experiment; (b) planned route (orange) and path executed (dashed blue) by the vehicle through the selected rewardbearing tasks; (c) simulation environment and execution. Video: https://streamable.com/g7wgkh

We emphasize that this example illustrates the generality of the proposed approach, as planning for a Dubins vehicle in an obstacle-rich environment effectively combines the characteristics of the DOP [16] and POP [3].

## VI. CONCLUSION AND FUTURE WORK

This work introduced the Reward-Guided RRT\*, a sampling-based framework that jointly addresses task selection and continuous-space motion planning under a finite budget. The approach incrementally explores the state space while guiding tree expansion toward rewarding locations based on their reward and travel cost, with path optimization reducing the required budget while preserving reward and feasibility. The experimental results demonstrate the ability to obtain near-optimal solutions across scenarios of varying complexity. Compared with VNS-PRM\*, Reward-Guided RRT\* achieved the same reward in the LOW and MEDIUM scenarios and only 4% less in the HIGH scenario while using substantially fewer samples. These results highlight its potential as a general sampling-based framework for budgeted task and motion planning.

Future work includes investigating more advanced mission-level optimization strategies that can reconsider previously committed tasks and exploit the remaining budget more effectively. Extending the framework to 3D environments and replacing traveled distance with more realistic budget measures, such as energy consumption, would also enable evaluation with vehicles such as UAVs, whose motion constraints are naturally represented in higher-dimensional state spaces. A further direction is to relax the irrevocablecommitment assumption, $e . g .$ , through limited backtracking or a rolling-horizon re-evaluation of recently committed segments, to mitigate the greedy nature of the current construction.

## REFERENCES

[1] B. L. Golden, L. Levy, and R. Vohra, “The orienteering problem,” Naval Research Logistics (NRL), vol. 34, no. 3, pp. 307–318, 1987.

[2] A. Gunawan, H. C. Lau, and P. Vansteenwegen, “Orienteering Problem: A survey of recent variants, solution approaches and applications,” European Journal of Operational Research, vol. 255, no. 2, pp. 315–332, 2016.

[3] R. Peni ˇ cka, J. Faigl, and M. Saska, “Physical Orienteering Problem for ˇ Unmanned Aerial Vehicle Data Collection Planning in Environments With Obstacles,” IEEE Robotics and Automation Letters, vol. 4, no. 3, pp. 3005–3012, 2019.

[4] S. M. LaValle, Planning algorithms. Cambridge university press Cambridge, UK, 2006, vol. 1.

[5] S. Karaman and E. Frazzoli, “Sampling-based algorithms for optimal motion planning,” The International Journal of Robotics Research, vol. 30, no. 7, pp. 846–894, 2011.

[6] P. Vansteenwegen, W. Souffriau, and D. V. Oudheusden, “The orienteering problem: A survey,” European Journal of Operational Research, vol. 209, no. 1, pp. 1–10, 2011.

[7] I.-M. Chao, B. L. Golden, and E. A. Wasil, “The team orienteering problem,” European Journal of Operational Research, vol. 88, no. 3, pp. 464–474, 1996.

[8] Z. Ma, K. Yin, L. Liu, and G. S. Sukhatme, “A spatio-temporal representation for the orienteering problem with time-varying profits,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017, pp. 6785–6792.

[9] J. Yu, M. Schwager, and D. Rus, “Correlated Orienteering Problem and its Application to Persistent Monitoring Tasks,” IEEE Transactions on Robotics, vol. 32, no. 5, pp. 1106–1118, 10 2016.

[10] L. E. Almeida, C. A. Valle, and D. G. Macharet, “Clustered Orienteering Problem With Subgroups,” IEEE Robotics and Automation Letters, vol. 10, no. 12, pp. 12 413–12 420, 2025.

[11] R. F. dos Santos, E. R. Nascimento, and D. G. Macharet, “Anytime Fault-tolerant Adaptive Routing for Multi-Robot Teams,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 7936–7942.

[12] A. Mansfield, D. G. Macharet, and M. A. Hsieh, “Energy-efficient Orienteering Problem in the Presence of Ocean Currents,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 10 081–10 087.

[13] M. T. P. T. Tristao and D. G. Macharet, “Team Orienteering Problem˜ with Communication Constraints,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025, pp. 12 675–12 680.

[14] L. Kavraki, P. Svestka, J.-C. Latombe, and M. Overmars, “Probabilistic roadmaps for path planning in high-dimensional configuration spaces,” IEEE Transactions on Robotics and Automation, vol. 12, no. 4, pp. 566–580, 1996.

[15] S. M. LaValle, “Rapidly-Exploring Random Trees: A New Tool for Path Planning,” Iowa State University, Technical Report 98-11, 1998.

[16] R. Peniˇ cka, J. Faigl, P. Vˇ a´na, and M. Saska, “Dubins Orienteeringˇ Problem,” IEEE Robotics and Automation Letters, vol. 2, no. 2, pp. 1210–1217, 2017.

[17] D. G. Macharet, A. Alves Neto, and D. Shishika, “Minimal Exposure Dubins Orienteering Problem,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2280–2287, 2021.

[18] F. Meyer and K. Glock, “Kinematic Orienteering Problem With Time-Optimal Trajectories for Multirotor UAVs,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 402–11 409, 2022.

[19] G. A. Croes, “A method for solving traveling-salesman problems,” Operations research, vol. 6, no. 6, pp. 791–812, 1958.

[20] L. E. Dubins, “On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents,” American Journal of mathematics, vol. 79, no. 3, pp. 497–516, 1957.