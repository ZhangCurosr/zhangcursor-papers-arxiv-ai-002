# POLICY SEARCH IN CODE SPACE FOR MULTI-AGENT PATHFINDING

Matthias Eder<sup>1</sup>, Pantelis Emmanouil<sup>2</sup>, Vivek Farias<sup>2</sup>, Fabian Fischer<sup>1</sup>, Adam Jozefiak<sup>2</sup>, Tianyi Peng<sup>3</sup>, Thomas Ulz<sup>1</sup>, Raphael Watschinger<sup>1</sup>, Fangyu Wu<sup>3,✉</sup>, Andrew Zheng<sup>4</sup> <sup>1</sup>KNAPP <sup>2</sup>MIT <sup>3</sup>Columbia University <sup>4</sup>University of British Columbia

## ABSTRACT

Developing coordination policies for complex multi-agent systems relies on either labor-intensive algorithm design or opaque task-specific reinforcement learning. With recent advances in autonomous coding agents, we show that policy search directly in code space offers a compelling third path that combines automated optimization with explicit interpretability. We investigate a human-directed workflow in which coding agents iteratively propose, implement, evaluate, diagnose, and eventually refine algorithms guided by simulation feedback and explicit correctness verification. We conducted two case studies and found that this workflow robustly outperformed any manually engineered solutions. On a challenging map in the 2024 League of Robot Runners competition, a two-day search discovered a policy that exceeds the stronger locally reproduced competition entry by 35.0% on the densest instance and generalizes to unseen map instances. In simulated real-world warehouses with irregular lanes, continuous robot geometry, and asynchronous motion, a week-long search discovered an entirely new class of multiagent pathfinding algorithm that outperforms a calibrated proprietary solution in throughput by 58.9%. Under stated assumptions, we further prove that the new algorithm is collision-free and completes, in finite time, every accepted goal that is not withdrawn or replaced. Together, the studies show that policy search in code space powered by state-of-the-art coding agents offers a highly competitive path to building real engineering solutions that are both effective and analytically tractable.

![](images/e4aea44a0a0e95608538bb7cf40fc60b0a2c816daef5af158bcbbd65dca68e4b.jpg)  
(a) League of Robot Runners 2024. We exceed the top entry by roughly 35% in a dense map.

![](images/d42b0eb81e913b80f4c812c28f6df36e36c8685757ec201f0ea47188dc511773.jpg)  
(b) Industrial warehouse. We complete 58.9% more orders than the proprietary solution.  
Figure 1: Throughput of the discovered policies across milestone commits. Human-directed coding agents iteratively refine coordination algorithms, producing policies that outperform strong manually engineered baselines in competitive multi-agent pathfinding problems. Maps above are illustrative.

## 1 INTRODUCTION

Multi-agent coordination underlies warehouse fulfillment, factory automation, and parcel sorting. In lifelong multi-agent pathfinding (LMAPF), tasks arrive continuously, requiring a fleet of robots to coordinate both task allocation and collision-free navigation through a shared spatial network (Wurman et al., 2008; Ma et al., 2017; Li et al., 2021b). The operational objective is to maximize system throughput subject to collision avoidance and strict real-time compute budgets. Because uncoordinated routing decisions cause severe congestion and deadlocks, effective warehouse operation typically requires sophisticated joint optimization of task assignment and trajectory coordination across the fleet.

Existing methods generally fall into two paradigms. First, the standard approach relies on human experts to engineer sophisticated solvers, dispatch rules, and traffic guidance heuristics (Okumura et al., 2022; Jiang et al., 2024; Zhang et al., 2024). Although effective, manual engineering is bottlenecked by human engineering throughput and cannot scale algorithm discovery with available compute. Second, learning-based methods attempt to bypass this manual effort by training neural policies directly through simulated interaction (Damani et al., 2021; Skrynnik et al., 2024; Andreychuk et al., 2025). However, neural policies often lack formal safety guarantees, remain difficult to inspect and verify, and resist modular modification when operational constraints shift.

We study a third approach: policy search in code space (PSCS), driven by coding agents. In this paradigm, human designers specify the system objective, physical constraints, and scaffolding interfaces, while coding agents iteratively refine the implementation of the planner and scheduler directly in source code, which we refer to as code space. Guided by simulation benchmarks, unit tests, and execution traces, the agent discovers procedural algorithms that are computationally efficient, interpretable, and maintainable. Unlike manual design, the search scales directly with compute; unlike black-box learning, the resulting policy remains fully transparent and verifiable.

Automated code generation has demonstrated promise in discovering novel mathematical algorithms and discrete heuristics (Romera-Paredes et al., 2024; Novikov et al., 2025; Liu et al., 2024). However, existing studies focus primarily on idealized theoretical benchmarks or stylized combinatorial problems. It remains an open question whether PSCS can scale to demanding cyber-physical systems such as LMAPF, where task allocation, spatial planning, and low-latency runtime engineering must interact robustly under realistic constraints. Specifically, we investigate whether coding agents can navigate these coupled requirements across both standardized benchmarks and complex industrial environments.

We evaluate PSCS across two distinct domains. The first is the 2024 League of Robot Runners (LoRR) benchmark (Chan et al., 2024), a synchronous grid competition environment operating with 100 to 800 robots. In this domain, PSCS systematically refines existing baselines to discover layoutspecific routing heuristics, outperforming the stronger locally reproduced competition entry by up to 35.0% in mean throughput (Figure 1a). The second is a high-fidelity simulation of an operational warehouse from KNAPP characterized by irregular lanes, continuous vehicle geometry, and asynchronous execution. In a challenging production map, our synthesized algorithm outperforms an optimized proprietary baseline by 58.9% (Figure 1b) while preserving provable progress certificates against deadlock under stated assumptions.

The contributions of this work are as follows:

1. We demonstrate the feasibility and effectiveness of policy search in code space (PSCS) for automated, nontrivial combinatorial algorithm design in LMAPF.

2. On the LoRR 2024 benchmark, we show that PSCS effectively refines established planners to discover map-specific heuristics, outperforming the stronger locally reproduced competition entry by up to 35.0% in mean throughput.

3. In high-fidelity industrial warehouse environments, we show that PSCS discovers an entirely new class of algorithms that outperforms an optimized proprietary baseline by 58.9% while maintaining provable progress certification under stated assumptions.

## 2 RELATED WORK

This work sits at the intersection of multi-agent pathfinding (MAPF) and automated program synthesis. Rather than hand-crafting domain heuristics or optimizing parametric weights, we search for coordination policies directly in code space.

Multi-Agent Pathfinding. MAPF seeks collision-free trajectories for a team of agents (Stern et al., 2019). In lifelong MAPF (LMAPF), agents continuously receive new tasks upon completing previous assignments (Ma et al., 2017; Li et al., 2021b). Because optimal solvers such as conflictbased search (Sharon et al., 2015) do not scale to hundreds of agents, large-scale deployments rely on fast suboptimal planners that human experts design and tune for specific topologies. Representative rule-based methods include PIBT (Okumura et al., 2022) for one-step priority inheritance, Causal-PIBT (Okumura et al., 2021) for asynchronous execution, and WPPL (Jiang et al., 2024), which combines windowed PIBT with large neighborhood search (Li et al., 2021a; 2022) and traffic guidance graphs. Although guidance graph optimization (Zhang et al., 2024) automates edge-cost tuning over these graphs, the underlying planning algorithms remain fixed. Alternatively, learningbased formulations train neural policies via reinforcement or imitation learning (Sartoretti et al., 2019; Damani et al., 2021; Skrynnik et al., 2024; Andreychuk et al., 2025), with recent neural planners and schedulers outperforming earlier benchmark baselines (Jiang et al., 2025; Gaber et al., 2026). Large language models (LLMs) have also been explored as direct online planners; however, prompting LLMs to emit next-step moves sequentially struggles to scale beyond small stylized instances (Chen et al., 2025).

Automated Program Synthesis. Policy search in code space builds upon a broader lineage of program synthesis and algorithmic discovery. Early programmatic reinforcement learning formulated policies as interpretable programs, though restricted to small, domain-specific languages (Verma et al., 2018; Trivedi et al., 2021). Modern LLMs lift these grammar constraints, enabling coding agents to manipulate general-purpose programming languages such as C++ and Python directly. Recent frameworks leverage this capability to evolve individual functions or isolated files for discrete algorithms and combinatorial optimization heuristics (Romera-Paredes et al., 2024; Novikov et al., 2025; Liu et al., 2024; Ye et al., 2024). Complementary lines of work optimize prompt graphs and textual variables through verbal feedback and execution traces (Yuksekgonul et al., 2025; Agrawal et al., 2026). More recently, automated loops have advanced to whole programs and repositories, in SAT solving, model training, and optimization contests (Yu et al., 2025; Karpathy, 2026; Imajuku et al., 2025), as well as dispatch and charging policies for automated guided vehicles without spatial path planning (Li & Li, 2026).

Our study differs from existing literature in the following aspects. 1 ) Whereas existing automated MAPF methods predominantly tune heuristic parameters or train parametric neural weights, our work searches over planner and scheduler implementations directly in code space. 2) Furthermore, while prior program synthesis frameworks rely on sophisticated evolutionary harnesses for stylized benchmarks, we demonstrate that state-of-the-art coding agents discover effective algorithms for demanding LMAPF problems without rigid search harnesses.

## 3 PROBLEM SETUP

We consider an environment consisting of n agents, a finite directed configuration graph $G =$ $( V , E )$ , and a discrete planning horizon of T clock ticks. Each vertex $u \in V$ represents an agent configuration, such as a discrete position and orientation. Each directed edge $e ~ = ~ ( u , v ) ~ \in ~ E$ denotes a feasible motion with strictly positive traversal duration $\delta ( e ) \in \mathbb { Z } _ { > 0 }$ ticks. At any tick $t \in \{ 0 , \ldots , T \}$ , an agent is either stationary at a vertex $u \in V$ or traversing an edge $e \in E$ with $\tau \in \{ 1 , \ldots , \delta ( e ) \}$ ticks remaining before arrival. Waiting is permitted only at vertices; edge traversals are non-preemptive and must proceed until completion.

Collision avoidance is formalized through a spatial conflict map. Let Γ : $V \cup E \to 2 ^ { V \cup E }$ denote a conflict function mapping any occupied vertex or edge to the set of configurations and edges that no other agent may occupy simultaneously. A valid multi-agent configuration requires that for any pair of agents occupying elements $x , y \in V \cup E , y \notin \Gamma ( x )$ . A stationary agent at vertex u occupies only u. A moving agent traversing $\boldsymbol { e } = \left( u , v \right)$ occupies edge e throughout traversal, its departure vertex u at the initiation tick, and its arrival vertex v at the completion tick. This general mapping captures classical grid-based vertex and swap conflicts, as well as complex footprint conflicts in continuous warehouse layouts.

Tasks enter an available-task pool Q over time according to an exogenous stochastic arrival process. Each task consists of an ordered sequence of errands. An errand is a tuple $( W , \sigma )$ , where ${ \bar { W } } \subseteq V$ denotes the set of admissible goal configurations and $\sigma \in \mathbb { Z } _ { > 0 }$ is the required service duration. An agent completes an errand by dwelling at a chosen vertex $\bar { w } \in W$ for σ contiguous ticks before proceeding to the subsequent errand. Task assignment is constrained such that each agent serves at most one active task at any time, and reassignments are prohibited once execution of the initial errand begins.

![](images/cab54b2b5794b53bb6c647c9505cd74f79b95c88d51638b81498df1223734cc8.jpg)  
Figure 2: Policy search in code space. The user provides constraints, performance objectives, and an initial codebase. The coding agent iteratively modifies source code, benchmarks candidates in simulation, and inspects traces to guide subsequent revisions.

A coordination policy π assigns available tasks to idle agents and selects, for each agent stationed at a vertex, either a wait action or an outgoing edge transition. Let J(π) denote the cumulative number of tasks completed by horizon T. The primary objective is to synthesize a policy π that maximizes expected throughput E[J(π)], subject to collision avoidance, valid task assignments, kinematic motion constraints, and real-time execution budgets. For domain-specific instantiations and operational constraints, see Sections 5.1 and 6.1.

## 4 POLICY SEARCH IN CODE SPACE

In policy search in code space (PSCS), we search for the source code of the coordination policy itself (Figure 2).

We initiate the search by specifying the environment, operational constraints, performance objectives, and an initial algorithm or reference implementation. Rather than using a rigid, specialized evolutionary harness, we prompt the coding agent directly in natural language, with minimal steering or redirecting as the search progresses. Remarkably, we find state-of-the-art coding agents already effective without additional augmentation from an external search harness.

Once the search is initiated, the coding agent iteratively refines the codebase through a four-stage loop: 1 ) proposing algorithmic edits, 2) implementing source-level modifications, 3) evaluating candidates in closed-loop simulation, and 4) diagnosing bottlenecks from execution traces and profiler logs. Importantly, language models are queried strictly offline during development; policy execution at runtime is handled entirely by compiled binaries with zero model inference latency.

During the search, all candidate policies must pass the following hierarchical verification before acceptance. First, a candidate must compile cleanly and pass unit tests enforcing collision avoidance, deadlock prevention, and compute budgets. Next, the coding agent uses fast short-horizon simulations to filter unpromising candidates. Lastly, surviving candidates undergo longer, multi-seed runs to confirm throughput improvements across stochastic task arrivals. Any candidate that fails the verification is rejected.

We apply this workflow to two distinct case studies: refining competitive heuristics on an LMAPF competition benchmark (Section 5), and discovering a routing policy for industrial warehouses with a provable progress guarantee under stated assumptions (Section 6).

## 5 CASE STUDY 1: LEAGUE OF ROBOT RUNNERS

Our first case study investigates whether PSCS can systematically discover and refine highperformance heuristics for established LMAPF benchmarks. We evaluate the framework on five challenging instances from the 2024 League of Robot Runners (LoRR) competition, wherein a single policy must jointly perform task allocation and path planning (MAPF Competition Organizers, 2025). At the time of writing this paper, the post-2024 LoRR competition archives have not yet been made public.

![](images/f8f3fcee9f8bcc4a237cf120c0f9f871071d079ba5d840d78e86086bee17e85c.jpg)  
Figure 3: The five RANDOM instances at t = 0. Gray cells are obstacles, blue cells hold robots.

## 5.1 EXPERIMENT SETUP

The LMAPF setting in LoRR follows the classical instantiation of the formulation in Section 3. A vertex represents a free cell with one of four cardinal orientations, yielding $| V | = 3 { , } 2 7 6$ . Each directed edge represents either a forward translation to an adjacent open cell or a 90<sup>◦</sup> in-place rotation, each with duration $\delta ( e ) = 1$ tick. Two agents conflict when they occupy the same spatial cell or traverse an edge in opposing directions simultaneously; thus, Γ(u) contains all four directional vertices at the corresponding grid coordinate. An errand target W comprises the four directional vertices corresponding to a designated service cell, with service time $\sigma = 0$ . At step t = 0, 1.5n tasks are visible, and each fulfilled task immediately reveals a replacement task. The dispatcher selects assignments from the available-task pool Q, without a FIFO or task-level fairness requirement. The instances span n ∈ {100, 200, 400, 700, 800} robots, evaluated over planning horizons T ∈ {600, 600, 800, 1,000, 2,000} ticks, respectively.

The policy search was initiated by supplying the coding agent, Codex Astra, with standard planning baselines: an implementation of Priority Inheritance with Backtracking (PIBT) (Okumura et al., 2022), the windowed large neighborhood search framework with guidance graphs from the 2023 winning entry WPPL (Jiang et al., 2024), and task-dispatching routines from 2024 participants. The environments consist of the five instances RANDOM-01–05, which share a common 32 × 32 2D grid containing 819 traversable cells (Figure 3). The agent operated in a command-line environment equipped with compilation toolchains, local simulation scripts, and trajectory verifiers. The search objective was to maximize cumulative throughput while strictly respecting collision avoidance, valid assignment constraints, a 1.0 s per-step execution budget, and a 32 GB memory ceiling.

## 5.2 RESULTS AND ANALYSIS

We benchmark our synthesized policy, termed Pipelined Look-Ahead with Task Matching (PLATM), against No Man’s Sky (NMS) and Kitty Knight (KK) in the combined track, the top two winning entries of LoRR 2024. Both baseline entries were re-evaluated on identical maps, initial poses, task sequences, and horizons using identical hardware: 16 physical CPU cores, a 32 GB memory ceiling, and a strict 1.0 s execution budget per step. Evaluations comprise eight paired trials per instance using fixed initial poses alongside identical random seeds for both the planner and the task arrival process.

Table 1 summarizes the final throughput comparison across all five competition instances. PLATM consistently outperforms both expert-engineered baselines across all fleet densities, achieving performance gains of up to 35.0% over the stronger baseline. The performance advantage is particularly pronounced on the densest instance (n = 800), where mutual blocking and congestion collapse severely degrade standard heuristic solvers.

Figure 4 illustrates the development trajectory of PLATM on RANDOM-05. Initial candidates failed due to exponential backtracking timeouts and graph deadlocks under standard PIBT at only 54 tasks. The agent systematically resolved these bottlenecks in stages: having waiting robots turn toward their next move in advance (∼2,400 tasks), evaluating more candidate plans per step (∼3,000 tasks), and scoring candidate moves across short simulated rollouts (>4,000 tasks). To adapt to less crowded maps (RANDOM-01–04), the agent enabled windowed group replanning and eliminated thread-creation overhead to stay strictly within the one-second time limit. The entire search converged in 42.3 wall-clock hours, consuming 3.5M output tokens (≈\$700).

Table 1: Tasks completed by PLATM and the top LoRR entries, as $\mathrm { m e a n } \pm \mathrm { s . d . }$ over eight paired trials.
<table><tr><td rowspan="2"></td><td colspan="2">Top LoRR Entries</td><td colspan="3">PLATM</td><td rowspan="2">Gain</td></tr><tr><td>NMS</td><td>KK</td><td> $\mathbf { M e a n } \pm \mathbf { S . D } .$ </td><td>Min</td><td>Max</td></tr><tr><td>RANDOM-01</td><td> $6 4 4 . 8 \pm 8 . 1$ </td><td> $6 8 7 . 1 \pm 6 . 8$ </td><td> ${ \bf 7 3 6 . 8 \pm 1 0 . 4 }$ </td><td>724</td><td>754</td><td> $+ 7 . 2 2 \%$ </td></tr><tr><td>RANDOM-02</td><td> $1 , 2 0 7 . 6 \pm 1 6 . 1$ </td><td> $1 , 2 1 9 . 1 \pm 1 8 . 0$ </td><td> ${ \bf 1 } , 3 5 6 . 9 \pm { \bf 1 4 . 6 }$ </td><td>1,332</td><td>1,380</td><td> $+ 1 1 . 3 0 \%$ </td></tr><tr><td>RANDOM-03</td><td> $2 , 3 8 4 . 1 \pm 4 2 . 3$ </td><td> $1 , 9 5 6 . 6 \pm 2 0 . 1$ </td><td> $2 , 6 4 7 . 5 \pm 3 1 . 4$ </td><td>2,606</td><td>2,703</td><td> $+ 1 1 . 0 5 \%$ </td></tr><tr><td>RANDOM-04</td><td> $2 , 6 1 0 . 8 \pm 3 3 . 6$ </td><td> $1 , 2 7 9 . 9 \pm 4 6 . 4$ </td><td> ${ \bf 2 } , 7 9 0 . 1 \pm 5 4 . 9$ </td><td>2,716</td><td>2,886</td><td> $+ 6 . 8 7 \%$ </td></tr><tr><td>RANDOM-05</td><td> $3 , 1 5 0 . 5 \pm 6 9 . 7$ </td><td> $2 , 0 6 5 . 3 \pm 2 5 . 5$ </td><td> $\mathbf { 4 } , \mathbf { 2 5 3 . 3 \pm 5 9 . 8 }$ </td><td>4,170</td><td>4,331</td><td> $+ 3 5 . 0 0 \%$ </td></tr></table>

![](images/750d60e60688ea3c8400a7691cd3cc6d3cd915860baf3bb0884ea60befdc22a7.jpg)  
(a) Throughput progression across code milestone commits.

![](images/d7498b8ae476dbccddaa368a8bdcc78a1c32c1ae912ba7da357714045aa1e522.jpg)  
(b) Elapsed search time, token consumption, and API expenditure across milestones.  
Figure 4: Policy search trajectory and computational cost of PLATM on RANDOM-05 (800 robots).

The synthesized policy couples task allocation and spatial planning in a unified two-level architecture: At the upper level, the task dispatcher balances assignments using a rotation-aware cost metric rather than simple grid distance, while allowing robots to trade unstarted goals dynamically as spatial conditions change. At the lower level, the motion planner is matched to traffic density: it coordinates the fleet one step at a time on the crowded instances, but uses windowed group replanning on the sparser ones. Complementing this adaptive planning, waiting robots use idle ticks to rotate toward their intended departure headings in advance, eliminating rotational delay the moment a path opens.

## 6 CASE STUDY 2: INDUSTRIAL WAREHOUSE

Our second case study investigates whether PSCS can synthesize an entirely new algorithm for physically constrained, real-world LMAPF environments. We evaluate our approach on high-fidelity simulations of operational KNAPP industrial warehouses, wherein an autonomous mobile fleet operates under continuous 3D vehicle geometries, kinematic lane constraints, and asynchronous execution.

## 6.1 EXPERIMENT SETUP

The warehouse LMAPF setting generalizes the formulation in Section 3 to 3D continuous space. Figure 5 illustrates elements of this continuous warehouse layout. A storage node is the ground-level interface to a vertical column of storage locations. A configuration vertex $u \in V$ represents a 3D robot pose (position and orientation), located either on the warehouse floor or elevated at a storage node. Directed edges $e = ( u , v ) \in E$ encompass forward and backward lane translations, in-place rotations, lateral docking maneuvers at storage nodes, and vertical lift transitions. Edge durations $\delta ( e )$ are governed by physical kinematics, with bounded floor velocities and accelerations, rotational speed limits, and vertical movement speed limits. An errand target W comprises the vertices at a storage node or picking station with a few seconds of service time $\sigma > 0$

Spatial conflicts are evaluated via continuous volumetric occupancy. Let $S ( z ) \subset \mathbb { R } ^ { 3 }$ denote the swept 3D bounding volume occupied by an agent at vertex pose or along edge transition $z \in V \cup E$ The geometric conflict mapping is defined as $\bar { \Gamma } ( z ) = \{ z ^ { \prime } \in \mathrm { \cdot } \mathrm { \cdot } \mathrm { \cup } E \mid S ( z ) \bar { \cap } S \bar { ( } z ^ { \prime } ) \neq \emptyset \}$ . This relation captures spatial interference that grid abstractions cannot model, such as wide vehicle sweeps into adjacent corridors during turning maneuvers.

![](images/17d2b9bc251afea79590816468f1e4eb407a16a18889249f4f98f04032b294dc.jpg)  
Figure 5: An illustrative warehouse layout. Black lanes connect waypoints, green markers denote storage nodes, and orange diamonds denote picking stations.

![](images/3dd35f74c8f2a5867578f58bbdbd9d284491ef9e37f243fc67f6c4713ca6ec8c.jpg)  
Figure 6: The six warehouse layouts. Maps are mirrored, and some rotated, for anonymity.

Orders arrive via an exogenous stochastic stream at approximately 1,800 orders per hour. Each order consists of an alternating sequence of pickup and drop-off errands between storage nodes and picking stations. Simulations run for a horizon of T = 7,200 s, with throughput measured in completed orders per hour. Algorithms must make decisions faster than real time.

The search was initiated using Codex 5.6 Sol and subsequently Codex Astra (upon its release) within a command-line environment equipped with the warehouse simulator. Auxiliary models Fable 5.1, Fable 5, and Opus 5 were occasionally consulted for code review and diagnostic feedback at the discretion of the primary coding agents. The search was seeded solely with a design specification: the agent was required to build an algorithm that, under stated assumptions, provably completes, in finite time, every accepted goal that is not withdrawn or replaced, while leaving architectural freedom for throughput heuristics to develop. The user directed the coding agent to prototype on WAREHOUSE-01 first and later generalize across the remaining five warehouse environments (Figure 6). The objective was to maximize order throughput, subject to continuous geometric collision avoidance, legal order assignments, and real-time execution, with no explicit memory constraints.

Table 2: Orders completed per hour by CGAR and KNAPP, as mean $\pm \ : s . 0 .$ . over eight paired trials.
<table><tr><td></td><td></td><td></td><td>KNAPP</td><td colspan="4">CGAR</td><td></td></tr><tr><td></td><td> $n ^ { 1 }$ </td><td> $d ^ { 1 }$ </td><td> $\mathbf { M e a n } \pm \mathbf { S . D } .$ </td><td> $\mathbf { M e a n } \pm \mathbf { S . D } .$ </td><td>Min</td><td>Max</td><td> $\mu / n ^ { 1 }$ </td><td>Gain</td></tr><tr><td>WAREHOUSE-01</td><td>65</td><td>51.8</td><td> $4 8 7 . 9 \pm 5 1 . 9$ </td><td> $7 7 5 . 3 \pm 2 5 . 4$ </td><td>733.0</td><td>808.5</td><td>11.93</td><td> $+ 5 8 . 9 \%$ </td></tr><tr><td>WAREHOUSE-02</td><td>17</td><td>43.5</td><td> $( 1 2 7 . 0 \pm 4 9 . 6 ) ^ { 2 }$ </td><td> $2 7 8 . 6 \pm 2 . 7$ </td><td>275.0</td><td>282.0</td><td>16.39</td><td> $( + 1 1 9 . 4 \% ) ^ { 2 }$ </td></tr><tr><td>WAREHOUSE-03</td><td>8</td><td>21.4</td><td> $( 1 7 . 7 \pm 1 4 . 8 ) ^ { 2 }$ </td><td> ${ \bf 1 3 5 . 1 \pm 5 . 8 }$ </td><td>128.5</td><td>144.0</td><td>16.89</td><td> $( + 6 6 4 . 0 \% ) ^ { 2 }$ </td></tr><tr><td>WAREHOUSE-04</td><td>19</td><td>49.3</td><td> $( 7 9 . 8 \pm 2 4 . 6 ) ^ { 2 }$ </td><td> ${ \bf 2 6 9 . 4 \pm 8 . 7 }$ </td><td>254.5</td><td>285.0</td><td>14.18</td><td> $( + 2 3 7 . 6 \% ) ^ { 2 }$ </td></tr><tr><td>WAREHOUSE-05</td><td>13</td><td>38.3</td><td> $( 2 8 . 4 \pm 7 . 3 ) ^ { 2 }$ </td><td> ${ \bf 1 9 1 . 4 \pm 7 . 5 }$ </td><td>177.5</td><td>201.5</td><td>14.73</td><td> $( + 5 7 4 . 7 \% ) ^ { 2 }$ </td></tr><tr><td>WAREHOUSE-06</td><td>32</td><td>74.2</td><td> $( 2 5 7 . 2 \pm 2 2 . 5 ) ^ { 2 }$ </td><td> ${ \bf 3 0 4 . 8 \pm 1 2 . 6 }$ </td><td>290.0</td><td>320.5</td><td>9.52</td><td> $( + 1 8 . 5 \% ) ^ { 2 }$ </td></tr></table>

<sup>1</sup> n is the number of robots, d the mean shortest path length of an order in meters, and $\mu / n$ the CGAR mean throughput per robot.  
<sup>2</sup> Values in parentheses are not optimized, as KNAPP has not tuned these maps for their method.

## 6.2 RESULTS AND ANALYSIS

We benchmark our synthesized policy, named Core-Graph-Anchored Routing (CGAR), against an optimized proprietary production baseline. As in Section 5, all evaluations execute on identical hardware, pairing eight trials per map with fixed initial poses and shared random seeds across methods for order arrivals (CGAR’s decisions are deterministic).

Table 2 reports throughput across eight paired two-hour evaluation runs for all six warehouse facil ities. CGAR outperforms the proprietary solution by 58.9% on WAREHOUSE-01 and achieves higher throughput across the remaining five layouts where the baseline was uncalibrated. The proprietary solution relies on extensive manual calibration of lane directions, traversal costs, and vertex-level stopping rules for each specific floor plan, limiting its transferability across facilities. In contrast, because CGAR operates on topology-agnostic coordination, a single synthesized policy transfers directly across all layouts without any site-specific recalibration. Across fleet sizes from 8 to 65 robots and mean route lengths from 21 to 74 m, CGAR consistently sustains 9.5 to 16.9 orders per hour per robot.

Figure 7 illustrates how the coding agent synthesized and refined CGAR on WAREHOUSE-01. The initial seed implementation enforced the progress guarantee too rigidly, actuating robots in nearsequential fashion, completing almost zero orders. The agent then unlocked concurrent execution by certifying simultaneous transitions that preserved a return path to a position where progress is guaranteed (∼350 orders/hour). It next resolved stop-and-go patterns and fleet freezes by stabilizing lane admission decisions (∼440 orders/hour). Throughput then plateaued near 445 orders/hour because robots took unnecessary detours around temporarily occupied corridors. The upgraded coding models (Astra and Fable 5.1) resolved this by introducing route commitments and fair queue release (630.6 orders/hour). Finally, the agent improved long-distance yielding maneuvers and refined order dispatching (775.3 orders/hour). Across the entire synthesis process, the search utilized 177.8 elapsed development hours and consumed 22.0M output tokens (≈\$4,800).

The resulting CGAR architecture decouples formal progress certification from heuristic throughput optimization: At its foundation, an invariant ensures that every admitted motion keeps a recovery plan back to the core; under the assumptions of Theorem 1 (Appendix A), this guarantees that every accepted goal that is not withdrawn or replaced is completed in finite time. Operating above this safety layer, throughput heuristics optimize traffic flow without the risk of inducing deadlock under the same assumptions. Specifically, the planner treats occupied lanes as temporary delays rather than static obstacles whenever the current occupant has committed to departure, avoiding unnecessary detouring loops. Concurrently, the task dispatcher assigns orders based on predicted lane congestion and trip duration, preventing bottlenecks before traffic accumulates.

## 7 DISCUSSION AND CONCLUSION

Across two challenging case studies, PSCS demonstrates that agentic code synthesis can discove algorithms that significantly outperform mature, human-engineered baselines. On the LoRR 2024 benchmark, PLATM surpasses the top two competition entries across all five instances, achieving throughput gains of up to 35.0% in extreme congestion. In simulated industrial warehouses, CGAR establishes a new class of multi-agent routing algorithms, delivering a 58.9% throughput increase over the proprietary production baseline on WAREHOUSE-01 while operating across all six facilities within a single, uncalibrated configuration. Together, these findings establish PSCS as a viable, highly effective methodology for complex combinatorial algorithm design.

![](images/b2e9c84fe02251fd297810633010aec6c7253b6566b757b621478af5b4cc0363.jpg)  
(a) Throughput progression across code milestone commits.

![](images/56483e0a181e5718fa480233f27c03523562ff3f81d18e235240c385fa1403f8.jpg)  
(b) Elapsed search time, token consumption, and API expenditure across milestones.  
Figure 7: Policy search trajectory and computational cost of CGAR on WAREHOUSE-01 (65 robots).

Both the search methodology and the resulting policies exhibit strong generalization capabilities. Because competitive LoRR entries rely on map-specific calibration, we evaluate whether PLATM transfers effectively to unseen layouts when provided comparable tuning. As detailed in Appendix B.1, recalibrating PLATM on a newly generated layout yields throughput within 3.9% of the competition benchmark across all fleet densities. In the industrial warehouse domain, CGAR exhibits zero-shot generalization: evaluated on the unseen synthetic layout WAREHOUSE-00 without any parameter adjustment, it sustains 16.98 orders per hour per robot, matching the operational efficiency observed on production facilities of comparable scale (Appendix B.2).

A primary operational constraint of PSCS is the computational and financial cost of empirical evaluation. The synthesis of PLATM required 42.3 wall-clock hours and approximately \$700 in API usage, whereas CGAR required 177.8 active development hours and approximately \$4,800. This expenditure is driven largely by the simulation bottleneck: each candidate policy must be evaluated over extended temporal horizons and multiple stochastic realizations before returning informative performance feedback to the coding agent. Accelerating policy evaluation is therefore a critical research direction, motivating techniques such as temporal parallelization (Farias et al., 2025), im portance sampling, and efficient surrogate simulators.

A notable trait of policies synthesized through PSCS is their accumulated software complexity. Whereas human engineers prioritize modularity, architectural symmetry, and maintainability, the coding agent optimizes strictly for empirical performance. As the agent resolves successive edge cases and failure modes, the codebase accumulates tightly coupled heuristics, yielding monolithic implementations that are challenging for developers to audit or modify. Future iterations of PSCS should incorporate software engineering best practices into the search objective, such as penalizing cyclomatic complexity or introducing periodic refactorization.

We also note a methodological consideration in our experimental execution: the policies in Sections 5 and 6 were developed through organic, human-in-the-loop steering rather than a fully autonomous, hardcoded search harness. It remains an open question to what extent rigidly encapsulating search logic within a static agentic scaffold risks overfitting to the limitations of a specific model release. Systematically characterizing the optimal boundary between programmatic search scaffolding and open-ended, model-driven steering represents an important direction for future inquiry.

Looking forward, PSCS represents a compelling paradigm for combinatorial algorithm design in verifiable domains. Like traditional software engineering, PSCS produces source code that incurs zero neural inference latency at runtime and admits formal safety verification under stated assumptions, as demonstrated by the progress certificates of CGAR (Theorem 1). Simultaneously, like deep learning, it scales with compute and optimizes directly against empirical loss landscapes through automated iteration. By uniting the formal verifiability of symbolic programs with the scalability of data-driven optimization, PSCS offers a practical alternative to both manual engineering and end-to-end learning. In this paradigm, code serves as the new weights.

## AI USE STATEMENT

Generative AI is part of the subject of this paper. Coding agents wrote, tested, and diagnosed the policy code of both case studies. Codex Astra did this work for PLATM. For CGAR, Codex 5.6 Sol and Codex Astra did most of it, Fable 5.1 contributed two mechanisms, and Fable 5 and Opus 5 gave feedback (Sections 5.1 and 6.1). The authors set the objectives and constraints and directed the work, and the final results passed the validity checks of Section 4. We also used AI assistants to help draft and edit the text, to produce figures and tables from our experiment records, and to check citations against their primary sources. The authors reviewed all AI-assisted text, figures, and code. We take responsibility for the final content of this paper, including all claims.

## REFERENCES

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective prompt evolution can outperform reinforcement learning. In International Conference on Learning Representations, 2026.

Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov, and Alexey Skrynnik. MAPF-GPT: Imitation learning for multi-agent pathfinding at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 23126–23134, 2025. doi: 10.1609/aaai.v39i22.34477.

Shao-Hung Chan, Zhe Chen, Teng Guo, Han Zhang, Yue Zhang, Daniel Harabor, Sven Koenig, Cathy Wu, and Jingjin Yu. The League of Robot Runners: Competition goals, designs, and implementation. In Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS), System Demonstrations, 2024.

Weizhe Chen, Sven Koenig, and Bistra Dilkina. Solving multi-agent path finding as an LLM benchmark: How, how good and why. Transactions on Machine Learning Research, 2025. URL https://openreview.net/forum?id=8hAxEFRVQT.

Mehul Damani, Zhiyao Luo, Emerson Wenzel, and Guillaume Sartoretti. PRIMAL<sub>2</sub>: Pathfinding via reinforcement and imitation multi-agent learning - lifelong. IEEE Robotics and Automation Letters, 6(2):2666–2673, 2021. doi: 10.1109/LRA.2021.3062803.

Vivek Farias, Joren Gijsbrechts, Aryan I. Khojandi, Tianyi Peng, and Andrew Zheng. Speeding up policy simulation in supply chain RL. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 16161–16177, 2025.

Johannes Gaber, Meshal Alharbi, Daniele Gammelli, and Gioele Zardini. GRAND: Guidance, rebalancing, and assignment for networked dispatch in multi-agent path finding. IEEE Robotics and Automation Letters, 11(5):5470–5477, 2026. doi: 10.1109/LRA.2026.3673902.

Yuki Imajuku, Kohki Horie, Yoichi Iwata, Kensho Aoki, Naohiro Takahashi, and Takuya Akiba. ALE-Bench: A benchmark for long-horizon objective-driven algorithm engineering. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2025.

He Jiang, Yulun Zhang, Rishi Veerapaneni, and Jiaoyang Li. Scaling lifelong multi-agent path finding to more realistic settings: Research challenges and opportunities. In Proceedings of the International Symposium on Combinatorial Search, volume 17, pp. 234–242, 2024. doi: 10.1609/ socs.v17i1.31565.

He Jiang, Yutong Wang, Rishi Veerapaneni, Tanishq Duhan, Guillaume Sartoretti, and Jiaoyang Li. Deploying ten thousand robots: Scalable imitation learning for lifelong multi-agent path finding. In Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2025. doi: 10.1109/ICRA55743.2025.11127445.

Andrej Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically. https://github.com/karpathy/autoresearch, 2026. GitHub repository, first commit on March 6, 2026.

Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, and Sven Koenig. Anytime multi-agent path finding via large neighborhood search. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI), pp. 4127–4135, 2021a. doi: 10.24963/ijcai.2021/ 568.

Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar, and Sven Koenig. Lifelong multi-agent path finding in large-scale warehouses. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11272–11281, 2021b. doi: 10.1609/aaai.v35i13.17344.

Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, and Sven Koenig. MAPF-LNS2: Fast repairing for multi-agent path finding via large neighborhood search. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pp. 10256–10265, 2022. doi: 10.1609/aaai. v36i9.21266.

Jinbo Li and Chuanhao Li. LLM-guided heuristic design from simulation traces: A case study in dynamic production and AGV scheduling. arXiv preprint arXiv:2608.09343, 2026.

Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32201–32223, 2024.

Hang Ma, Jiaoyang Li, T. K. Satish Kumar, and Sven Koenig. Lifelong multi-agent path finding for online pickup and delivery tasks. In Proceedings of the 16th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pp. 837–845, 2017.

MAPF Competition Organizers. The League of Robot Runners benchmark archive: 2024 competition. https://github.com/MAPF-Competition/Benchmark-Archive, 2025. Archived evaluation instances, solutions, and implementation references.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt˜ Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Keisuke Okumura, Yasumasa Tamura, and Xavier Défago. Time-independent planning for multiple moving agents. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11299–11307, 2021. doi: 10.1609/aaai.v35i13.17347.

Keisuke Okumura, Manao Machida, Xavier Défago, and Yasumasa Tamura. Priority inheritance with backtracking for iterative multi-agent path finding. Artificial Intelligence, 310:103752, 2022. doi: 10.1016/j.artint.2022.103752.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024. doi: 10.1038/ s41586-023-06924-6.

Guillaume Sartoretti, Justin Kerr, Yunfei Shi, Glenn Wagner, T. K. Satish Kumar, Sven Koenig, and Howie Choset. PRIMAL: Pathfinding via reinforcement and imitation multi-agent learning. IEEE Robotics and Automation Letters, 4(3):2378–2385, 2019. doi: 10.1109/LRA.2019.2903261.

Lui Sha. Using simplicity to control complexity. IEEE Software, 18(4):20–28, 2001. doi: 10.1109/ MS.2001.936213.

Guni Sharon, Roni Stern, Ariel Felner, and Nathan R. Sturtevant. Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence, 219:40–66, 2015. doi: 10.1016/j.artint.2014.11. 006.

Alexey Skrynnik, Anton Andreychuk, Maria Nesterova, Konstantin Yakovlev, and Aleksandr Panov. Learn to follow: Decentralized lifelong multi-agent pathfinding via planning and learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17541–17549, 2024. doi: 10.1609/aaai.v38i16.29704.

Roni Stern, Nathan Sturtevant, Ariel Felner, Sven Koenig, Hang Ma, Thayne Walker, Jiaoyang Li, Dor Atzmon, Liron Cohen, T. K. Satish Kumar, Roman Barták, and Eli Boyarski. Multiagent pathfinding: Definitions, variants, and benchmarks. In Proceedings of the International Symposium on Combinatorial Search, volume 10, pp. 151–158, 2019. doi: 10.1609/socs.v10i1. 18510.

Dweep Trivedi, Jesse Zhang, Shao-Hua Sun, and Joseph J. Lim. Learning to synthesize programs as interpretable and generalizable policies. In Advances in Neural Information Processing Systems, volume 34, 2021.

Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, and Swarat Chaudhuri. Programmatically interpretable reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 5045–5054, 2018.

Peter R. Wurman, Raffaello D’Andrea, and Mick Mountz. Coordinating hundreds of cooperative, autonomous vehicles in warehouses. AI Magazine, 29(1):9–20, 2008. doi: 10.1609/aimag.v29i1. 2082.

Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. ReEvo: Large language models as hyper-heuristics with reflective evolution. In Advances in Neural Information Processing Systems, volume 37, 2024.

Cunxi Yu, Rongjian Liang, Chia-Tung Ho, and Haoxing Ren. Autonomous code evolution meets NP-completeness. arXiv preprint arXiv:2509.07367, 2025.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou. Optimizing generative AI by backpropagating language model feedback. Nature, 639(8055):609–616, 2025. doi: 10.1038/s41586-025-08661-4.

Yulun Zhang, He Jiang, Varun Bhatt, Stefanos Nikolaidis, and Jiaoyang Li. Guidance graph optimization for lifelong multi-agent path finding. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI), pp. 311–320, 2024. doi: 10.24963/ijcai.2024/ 35.

## A THE CGAR ALGORITHM

CGAR is briefly described in Section 6. This appendix gives the full algorithm (Section A.2). It proves that every execution is collision-free and that every goal the task layer does not withdraw or replace is completed in finite time (Section A.3).

## A.1 OVERVIEW

CGAR is a PIBT-style planner (Okumura et al., 2022) for robots with continuous geometry that move asynchronously. It has two layers. The throughput layer proposes motions using priority inheritance and heuristics. The safety layer decides which proposals are committed, and supplies certified motions when no proposal can be committed. This follows the Simplex architecture (Sha, 2001): an unverified controller runs freely as long as a verified fallback remains available.

Before a run, CGAR selects a set of ground poses on the map, called core nodes, and fixed motions that connect them. Together, they form the core graph. The robots are on the core graph when every robot occupies a different core node and no motion is pending. The safety layer rests on one invariant: at every moment, all robots can be brought back to the core graph. CGAR commits a motion only if two conditions hold. First, the motion is collision-free. Second, after the motion is committed, there exists a recovery plan: a plan that brings all robots back to the core graph. These two checks maintain the invariant. Once the robots are on the core graph, and there are more core nodes than robots, any robot can reach any core node by shifting other robots along a vacancy, like a sliding puzzle.

Goals are served oldest-first. If the oldest goal remains primary for D seconds, CGAR activates a deadline-triggered fallback at the next decision boundary. It stops all other work, executes the recovery plan, and moves the primary robot to its goal with vacancy shifts.

## A.2 ALGORITHM

We use the configuration graph $G = ( V , E )$ and the conflict map Γ of Sections 3 and 6.1. Execution is asynchronous: the actual duration of a motion may differ from its nominal duration, and robots start and finish their motions at different times. CGAR makes a decision at every decision boundary: when a motion or service finishes, when the task layer assigns, replaces or withdraws a goal, or at a periodic controller tick (1 s by default).

A path is a short sequence of motions of one robot, and each robot executes at most one committed path at a time. A new path is committed only if two conditions hold.

(i) The volume it sweeps, the union of $S ( e )$ over its motions e (Section 6.1), is disjoint from the volume swept by every other path in progress, so the two cannot collide, whatever their timing.

(ii) It does not intersect $S ( u )$ for any robot standing still at a vertex u.

The core graph is specified by the triple $( B , H , P )$ . The set $B \subseteq V$ consists of ground poses of G (a position on the floor and an orientation), called core nodes, and H and P are directed graphs on B.

• The support graph $H = ( B , E _ { H } )$ . Each core edge $u  v$ has an associated core path: a fixed path from u to v in G whose edges are lane traversals and in-place rotations.

• The promotion graph $P = ( B , E _ { P } )$ , with $E _ { P } \subseteq E _ { H }$

The two graphs have different roles. Core edges of H are used to move robots out of the way, and core edges of $P$ are used to move a robot toward its goal. P may leave out core edges of H that are safe for the first use but not for the second (condition (K5) below).

Before a run, CGAR builds the core graph automatically from the map and checks the following five conditions for a fleet of n robots.

(K1) Separation. Robots standing at distinct core nodes do not overlap: $S ( b ) \cap S ( b ^ { \prime } ) = \emptyset$ for any two distinct core nodes $\bar { b } , b ^ { \prime } \in B$

(K2) Clear core paths. The core path for every core edge $u  v$ is free of conflicts with any core node other than u and v.

![](images/6d453f922419d017d41ffc55bf16d95855087d9fffaaeac35125e72cbff17aa9.jpg)  
Figure 8: The core graph of WAREHOUSE-00 for a fleet of 16 robots. Hollow purple circles mark its 17 core nodes; purple curves show core paths projected onto the ground network, omitting headings and rotations.

(K3) Connectivity. H and P are strongly connected.

(K4) Vacancy. The core graph contains more nodes than robots: $| B | \geq n + 1$ . Thus, whenever all robots are on the core graph, at least one core node is vacant.

(K5) Room to advance. For every core edge u → v of P, at least n core nodes can be reached from v in H without passing through u.

The last condition ensures that while a robot stands at u, at least one core node reachable from v is vacant. The robots between v and that core node can then shift toward it one at a time without passing through u, which frees v for the robot at u.

If any check fails, the run does not start. The checks passed on all six tested warehouses. Figure 8 shows a certified core graph projected onto WAREHOUSE-00’s ground network.

A recovery plan is a sequence of steps that brings all robots back to the core graph, starting once every committed path has finished. Each step is a set of paths, at most one per robot, that satisfy (i) and (ii) with respect to each other and to every robot standing still, so they can run at the same time. CGAR always stores a recovery plan. It commits new paths only if a recovery plan exists from the vertices where the robots will be once all committed paths, including the new ones, have finished. CGAR searches for such a plan with a bounded search and checks it by simulating every step. If no plan is found, the new paths are rejected.

The task layer gives each robot one goal at a time: a vertex to reach and, if required, to perform a service at. Robots receive tickets from an increasing counter when they get assigned a goal, and the robot with the oldest goal is the primary.

At each decision boundary, every robot proposes candidate paths, such as prefixes of its shortest route, pushes of a blocking robot as in PIBT, and moves along the core graph. CGAR visits the robots in ticket order, starting with the primary, and adds each robot’s best candidate that satisfies (i) and (ii) with respect to the paths chosen so far. It commits the chosen paths if a recovery plan exists, and otherwise removes paths until one does. If nothing can be committed, it uses the steps of the fallback described below, without stopping other robots.

On top of this skeleton, the throughput layer uses many heuristics that shape which paths are proposed and in what order. For example:

• A robot blocked only by traffic that will clear by itself waits instead of detouring;

• Robots blocked at a service location back off to the core graph and retry;

• The primary can be overtaken only a bounded number of times before other robots must yield to it;

• Helper robots that make way for the primary may move in parallel when this does not delay it.

![](images/f174616ff6500019c49c31d1dc208cbaf0268d5549945c3890c528cdb5cbce3f.jpg)  
Figure 9: CGAR decision flow. Safety requires passing checks (i)–(ii); recoverability requires a valid recovery plan. The deadline also pauses work outside fallback.

These heuristics change only which paths are proposed and in what order. Every committed path still passes the same checks.

When a robot becomes primary, the fallback deadline of its goal is set to D seconds later. At the first decision boundary at or after this deadline, if the goal remains active, CGAR protects it and enters the following fallback. All work outside the fallback halts until the goal is completed, withdrawn or replaced.

1. Drain. Wait for committed motions to finish.

2. Recover. Execute the stored recovery plan step by step.

3. Promote. Move the primary along a shortest path in P to the entry node of its goal, a core node from which the primary can reach its goal and return without conflicting with any other core node. For each core edge u → v of this path, the robots between v and a vacant core node first move one core node forward, one at a time, as described after (K5). This frees v, and the primary moves from u to v.

4. Serve. Move the primary from its entry node to its goal, and keep the path back to the entry node as the recovery plan.

If the fallback cannot construct a step, the run stops with an error rather than returning to heuristic routing. Figure 9 summarizes one decision.

## A.3 SAFETY AND PROGRESS

Besides (K1)–(K5), which are checked before each run, the proof uses three assumptions.

(E1) Finite motions. Every motion or service, once started, finishes in finite time.

(E2) Recoverable start. The robots start at non-overlapping positions with a recovery plan, for example all on the core graph.

(S) Service coverage. Every goal has an entry node: a core node b with a path from b to the goal and back that conflicts with no other core node. <sup>1</sup>

The proof also takes the runtime checks of (i), (ii) and of the recovery plan to be correct as implemented.

Theorem 1. Under (K1)–(K5), (E1), (E2) and (S), every execution of CGAR is collision-free, a recovery plan exists at every decision, and every goal is completed in finite time unless the task layer withdraws or replaces it.

Lemma A.1 (Safe Commitment). Suppose the robots start at non-overlapping positions, and every committed path satisfies (i) and (ii) when it is committed. Then no two robots ever overlap, however long each motion takes.

Proof. A robot never leaves the volume swept by its last committed path, which includes the path’s end. By (i), this volume is disjoint from that of every path in progress when the path was committed, and by (ii), from every robot standing still at that time. Paths committed later are checked against this one, or against the robot standing at its end, in the same way. □

Lemma A.2 (Recovery Is Always Possible). CGAR always stores a recovery plan.

Proof. Initially, a recovery plan exists by (E2). We show that every decision keeps one. Between decisions, robots only carry out committed paths, so the stored plan stays valid. At a decision, CGAR commits one of the following.

• Outside the protected fallback, paths are committed only if a recovery plan exists afterwards.

• Recover executes the next step of the stored plan. The remaining steps form a recovery plan.

• Promote moves one robot along a core path while all robots are on the core graph. By (K1) and (K2), all robots are on the core graph again once it finishes, so no further recovery is needed.

• Serve moves the primary from its entry node to its goal while all other robots are on the core graph. By (S), this conflicts with no other core node, and the path back to the entry node is a recovery plan.

Drain commits nothing.

Lemma A.3 (Vacancy Shift). Let the robots be on the core graph, let robot r stand at core node u, and let u → v be a core edge of P. Then at most n − 1 other robots can each shift one core node along a path in H that avoids u, after which r moves to v. This executes at most n core paths, one at a time, and the robots are on the core graph after each core path.

Proof. By (K5), at least n core nodes are reachable from v in H without passing through u. At most $n - 1$ robots other than r stand on them, so at least one of them is vacant. Let h be a vacant one closest to v, and take a shortest path $v = q _ { 0 } , \ldots , q _ { k } = h$ in H that avoids u. Every core node on this path before h is occupied, so $k \leq n - 1$ . For $j = k , \ldots , 1$ , if a robot stands at $q _ { j - 1 }$ , move it to $q _ { j }$ , which is vacant at that point. Then v is vacant, and r moves from u to v. This uses at most $k + \bar { 1 } \leq n$ core paths. Each core path moves one robot while the others stand on core nodes, so by (K1) and (K2) it causes no conflict and ends with the robots on the core graph. □

Lemma A.4 (The Fallback Completes the Goal). Once a goal is protected, it is completed in finite time unless it is withdrawn or replaced.

Proof. New goals receive larger tickets, so the protected goal stays the primary’s goal until it ends.   
Only fallback steps are committed while it is protected, and each step finishes in finite time by (E1).

• Drain: all committed paths finish.

• Recover: the stored recovery plan (Lemma A.2) brings all robots back to the core graph.

• Promote: by (K3), the entry node is at most $| B | { - } 1$ core edges of P away, so by Lemma A.3 at most $( | B | - 1 )$ n core paths bring the primary there.

• Serve: by (S), the path from the entry node to the goal conflicts with no other core node, so the primary reaches its goal and performs the service.

ProofofTheorem 1. Safety. Paths committed in normal operation are checked for (i) and (ii). Fallback steps satisfy them too: recovery steps by definition, core paths by (K1) and (K2), and the path to a goal by (S). Lemma A.1 then gives collision freedom, and Lemma A.2 gives the recovery plan.

Progress. Let g be a goal that is never withdrawn or replaced. New goals receive larger tickets, so while g is active, the primary’s goal is g or one of the at most n − 1 goals that were active when g was accepted. Each primary goal ends within D seconds of becoming primary, or CGAR protects it at the first decision boundary at or after its fallback deadline and it ends by Lemma A.4. Hence at most n goals, each ending in finite time, precede the completion of g. □

## B EVALUATION ON ADDITIONAL LAYOUTS

## B.1 PLATM ON RANDOM-06 TO RANDOM-10

We generate one new $3 2 \times 3 2$ layout with the size and obstacle density of the competition map, and reuse the five fleet sizes and horizons of Section 5.1 to obtain RANDOM-06 to RANDOM-10 (Figure 10).

![](images/550251367b85d39775d8913c1af4db5226d93b38712c970f37711562505e3e66.jpg)  
RANDOM-06 100 robots

![](images/dc8c0c8a6eeb19890376c59a578f64b81ddc5f250dc309919b1f43939180c826.jpg)  
RANDOM-07 200 robots

![](images/e4f4dbd6b1d58494ba34c60c858db011769eb3842367b0808b6e810f09c963ed.jpg)  
RANDOM-08 400 robots

![](images/c77a92a07acbe2b900553ee32697e6236cb21e5c52f0ca828f5f610a7c3872e8.jpg)  
RANDOM-09 700 robots

![](images/c909fdd907773ad56147d4848c49e39c2d9b321804b39c28da118470c170cde3.jpg)  
RANDOM-10 800 robots

Figure 10: The five instances RANDOM-06 to RANDOM-10 at their start. Gray cells are obstacles, and blue cells hold robots.

Table 3 reports PLATM on both layouts over eight paired trials with fresh task streams. The new maps received map-specific calibration, as it is part of the algorithm. Their task counts are within 3.9% of those on the competition layout.

Table 3: Tasks completed by PLATM on the competition layout and, after map-specific calibration, on the additional layout, as mean $\pm \ \mathrm { s . d . }$ over eight paired trials. The RANDOM-09 and RANDOM-10 profiles received extended calibration.
<table><tr><td>n</td><td> $\mathbf { M e a n } \pm \mathbf { S } . \mathbf { D } .$ </td><td>Min</td><td>Max</td></tr><tr><td>RANDOM-01 100 RANDOM-02 200 RANDOM-03 400 RANDOM-04 700 RANDOM-05 800</td><td> $7 3 6 . 8 \pm 1 0 . 4$   $1 , 3 5 6 . 9 \pm 1 4 . 6$   $2 , 6 4 7 . 5 \pm 3 1 . 4$   $2 , 7 9 0 . 1 \pm 5 4 . 9$   $4 { , } 2 5 3 { . } 3 \pm 5 9 { . } 8$ </td><td>724 1,332 2,606 2,716 4,170</td><td>754 1,380 2,703 2,886 4,331</td></tr><tr><td>RANDOM-06 100 RANDOM-07 200 RANDOM-08 400 RANDOM-09 700</td><td> $7 4 6 . 4 \pm 1 0 . 9$   $1 , 3 7 3 . 0 \pm 1 6 . 7$   $2 , 5 6 1 . 9 \pm 5 0 . 4$   $2 , 7 0 7 . 6 \pm 3 1 . 6$  800  $4 , 0 8 9 . 4 \pm 6 3 . 9$ </td><td>724 1,345 2,496 2,658 4,008</td><td>761 1,400 2,625 2,749 4,205</td></tr></table>

## B.2 CGAR ON WAREHOUSE-00

We run CGAR unchanged on WAREHOUSE-00, the synthetic layout of Figure 11, with 16 robots and the demand seeds of the main comparison. Table 4 reports throughput per robot: WAREHOUSE-00 reaches $1 6 . 9 8 \pm 0 . 7 8$ orders per hour per robot, on par with the real warehouses of similar fleet size. All eight runs finish with no logged collisions or assignment failures.

![](images/7398a2d58ededf09127e7684bb12da0a0e2f3bd1cbf17aca387c7d05e0ec91a5.jpg)  
Figure 11: WAREHOUSE-00, the synthetic layout evaluated here. Black lanes connect waypoints, green markers denote storage nodes, and orange diamonds denote picking stations. Blue vehicles illustrate poses and are not the evaluation’s initial state.

Table 4: Orders completed per hour per robot by CGAR, as mean ± s.d. over eight trials. The symbols n and d are those of Table 2.
<table><tr><td></td><td>n</td><td> $d$ </td><td> $\mathbf { M e a n } \pm \mathbf { S } . \mathbf { D } .$ </td><td>Min</td><td>Max</td></tr><tr><td>WAREHOUSE-01</td><td>65</td><td>51.8</td><td> $1 1 . 9 3 \pm 0 . 3 9$ </td><td>11.28</td><td>12.44</td></tr><tr><td>WAREHOUSE-02</td><td>17</td><td>43.5</td><td> $1 6 . 3 9 \pm 0 . 1 6$ </td><td>16.18</td><td>16.59</td></tr><tr><td>WAREHOUSE-03</td><td>8</td><td>21.4</td><td> $1 6 . 8 9 \pm 0 . 7 2$ </td><td>16.06</td><td>18.00</td></tr><tr><td>WAREHOUSE-04</td><td>19</td><td>49.3</td><td> $1 4 . 1 8 \pm 0 . 4 6$ </td><td>13.39</td><td>15.00</td></tr><tr><td>WAREHOUSE-05</td><td>13</td><td>38.3</td><td> $1 4 . 7 3 \pm 0 . 5 7$ </td><td>13.65</td><td>15.50</td></tr><tr><td>WAREHOUSE-06</td><td>32</td><td>74.2</td><td> $9 . 5 2 \pm 0 . 3 9$ </td><td>9.06</td><td>10.02</td></tr><tr><td>WAREHOUSE-00</td><td>16</td><td>32.2</td><td> $1 6 . 9 8 \pm 0 . 7 8$ </td><td>15.88</td><td>17.97</td></tr></table>