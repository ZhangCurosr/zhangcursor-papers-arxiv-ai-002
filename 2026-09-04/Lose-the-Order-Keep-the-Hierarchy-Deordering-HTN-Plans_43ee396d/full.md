# Lose the Order, Keep the Hierarchy: Deordering HTN Plans

Takudzwa Togarepi<sup>1</sup>, Gaspard Quenard<sup>1</sup>, Damien Pellier<sup>1</sup>, Humbert Fiorino<sup>1</sup>

<sup>1</sup>Universite Grenoble Aples, France´

{takudzwa.togarepi, gaspard.quenard}@univ-grenoble-alpes.fr, {damien.pellier, humbert.fiorino}@imag.fr

## Abstract

Hierarchical Task Network (HTN) planning is a powerful planning formalism based on task decomposition. Although most of the literature studied plan generation, comparatively less attention has been paid to post-plan optimization. In particular, plan deordering has been extensively studied in classical planning but remains under-researched in the HTN setting. Plan deordering removes unnecessary ordering constraints between actions in a plan whilst keeping the plan valid. In this paper, we adapt two established plan deordering techniques from classical planning by extending the techniques to account for hierarchical decomposition constraints. We evaluate our proposed approaches on the IPC 2023 Partial-Order HTN benchmarks and we compare them against Optiplan, an HTN planner that generates partially ordered plans directly. Our results show a substantial reduction in number of ordering constraints in both our implementations. Although we also observe a reduction in critical path length, the improvements are less pronounced.

## Introduction

Hierarchical Task Network (HTN) planning (Erol, Hendler, and Nau 1994b) is planning by task decomposition; that is, recursively refining complex tasks until they become executable tasks. Among other planning formalisms, HTN planning is generally more expressive than STRIPS-style planning (Erol, Hendler, and Nau 1994a; Georgievski and Aiello 2015).

The main objective of an HTN planner is to determine how to refine an initial abstract (compound) task into a sequence of executable tasks. In simple terms, a plan is a sequence of primitive actions whose execution accomplishes a given task network. Existing HTN planners differ in the way they address an HTN problem. Some encode the given HTN problem into propositional logic in order to exploit existing specialized solvers (Behnke, Holler, and Biundo 2019;¨ Schreiber et al. 2019; Schreiber 2021; Quenard, Pellier, and Fiorino 2024, 2025). Alternatively, other planners make use of heuristic search (Holler et al. 2018, 2019, 2020), whereas¨ others translate HTN problems to a classical planning problem in order to exploit classical planners (Alford et al. 2016; Behnke et al. 2022).

While most HTN planning research has extensively studied plan generation, fewer works have addressed plan optimization (Behnke, Holler, and Biundo 2019; Behnke and¨

Speck 2021). In their recent survey, Bercher, Haslum, and Muise (2024) define plan optimization as the task of improving a plan that has already been generated, and they also highlight several optimization techniques.

In practice, one important aspect of plan quality is planorder flexibility. In some real-world applications of HTN planning, a totally ordered plan impose unnecessary ordering constraints at the expense of efficiency. For example, in a firefighting mission, a totally ordered plan could enforce that action dispatch-ground-crew be executed before dispatchhelicopter. However, these two actions do not interfere and thus removing the ordering constraint between these two actions results in a flexible partial-order plan. Executing actions in parallel might improve execution speed and flexibility. This approach of post-plan optimization by removing ordering constraints on a plan while keeping it a valid plan is known as deordering and has been extensively studied in the classical planning sense (Backstr¨ om 1998; Kambhampati¨ and Kedar 1994; Muise, McIlraith, and Beck 2012; Bercher and Olz 2020; Siddiqui and Haslum 2015; Muise, Beck, and McIlraith 2016). In particular, Muise, Beck, and McIlraith (2016) improved deordering methods by using partially weighted MaxSAT which computes the minimal deorderings. A minimum deordering is a deordered plan in which one cannot remove any additional ordering constraints without invalidating the plan.

To the best of our knowledge, no prior work has explicitly studied plan deordering under HTN decomposition constraints. Although there is not much done in terms of HTN plan deordering, a few planners try to pursue a similar goal of having a partially ordered plan. One such planner is Opti-Plan by Firsov (2025), which produces a partially ordered plan directly. This motivates our interest in studying deordering in HTN planning. Our contributions in this paper are twofold. First, we adapt the PRF algorithm of Backstr¨ om¨ (1998) to the HTN setting, taking into account the hierarchical decomposition constraints. Second, we compute minimal deorderings using a MaxSAT approach of Muise, McIlraith, and Beck (2012), which we also adapt to the HTN setting.

We evaluate both implementations on the IPC 2023 Partial-Order HTN benchmarks. We apply the deordering techniques to plans generated by PANDA a partial-order planner (Holler 2023). As a baseline, we compare the perfor-¨ mance of the post-plan optimization against Optiplan. The results show a significant decrease in both the number of ordering constraints and the critical-path length of the plans in the post-plan optimization.

The remainder of this paper is structured as follows: following this introduction, we introduce the formalisms used throughout the paper. Afterward, we present the two implemented algorithms followed by a discussion of the results. Finally, we provide directions for future work and conclude the paper.

## Formalism

We begin by introducing the formalisms for the techniques used throughout this paper

## STRIPS Planning Problem

For the purpose of this paper, whenever we mention propositional planning, we restrict it to the STRIPS formalism by Bylander (1996). In the STRIPS formalism, a planning task is defined as a tuple $\Pi = \langle \mathcal { V } , \mathcal { A } , \mathcal { T } , \mathcal { G } \rangle$ where V refers to a set of propositional state variables, A is a finite set of actions, $\bar { \mathcal { T } } \subseteq \mathcal { V }$ is the initial state and $\mathcal { G } \subseteq \mathcal { V }$ is the goal condition. Each action $a \ \in \ A$ is defined by a triple $\langle p r e ( a ) , a d d ( a ) , d e l ( a ) \rangle$ ⟩ where $p r e ( a ) \subseteq \mathcal { V }$ is a set of preconditions, $a d d ( a ) \subseteq \mathcal { V }$ is a set of add effects, and ${ \bar { d e l } } ( a ) \subseteq \mathcal V$ is a set of delete effects. A state $s \subseteq \nu$ is the set of state variables that are true. Under the closed-world assumption, every variable in $\mathcal { V } \backslash s$ is considered false. An action a is applicable in s if and only if $p r e ( a ) \subseteq s$ . The resulting successor state $s ^ { \prime } = ( s \backslash d e l ( \boldsymbol { a } ) ) \cup \bar { a } d \dot { d } ( \boldsymbol { \dot { a } } ) . \mathbf { A }$ plan (solution) $\pi = a _ { 1 } , \ldots , a _ { n } ,$ , is a sequence of actions which when successively applied from the initial state lead to a goal state. A partially ordered plan (PO plan) maintains a partial-order among actions (Backstr¨ om 1998).¨

## Hierarchical Task Network (HTN) Planning

In this section, we present the HTN planning formalism by Geier and Bercher (2011) that we are going to use for the rest of the paper. HTN planning is centered on task decomposition. A task consists of a task name and an associated list of parameters. In HTN, tasks are classified into two categories: abstract tasks and primitive tasks. A primitive task is analogous to an action in classical planning; that is, it can directly modify the state of the world. In contrast, abstract tasks cannot be executed directly and must be decomposed by applying decomposition methods.

A method m is defined as a tuple $( c , t n )$ where c is an abstract task and tn a task network. We refer to c as the abstract task decomposed by method m, and to tn as the task network that replaces c when m is applied. The resulting task network consists of primitive and/or abstract tasks.

We now formally define a task network as follows:

Definition 1. (Task network). A task network tn over a set of task names X is a tuple $( T , \prec , \alpha )$ , where

• $T$ is a finite set of tasks

$\prec \subseteq T \times T$ is a set of ordering constraints of the form $( t _ { i } < t _ { j } )$ where $t _ { i } , t _ { j } \in T$

• α: $T  X$ assigns a task name to each task in the network.

$T N _ { X }$ denotes a set of all task networks over X. A task network comprising only primitive tasks is known as a primitive task network. In their paper, Geier and Bercher (2011) defined an executable task network as one that contains a linearization of its tasks that is executable.

Next, we define a HTN planning problem.

Definition 2. (Planning problem). A planning problem is a tuple $\mathcal { P } = ( L , C , O , M , c _ { I } , s _ { I } , g )$ , where

• L is a finite set of propositions

• C is a finite set of abstract tasks

• O is a finite set of primitive tasks (actions)

${ M \subseteq C \times T N _ { C \cup O } }$ a finite set of decomposition methods

$c _ { I } \in C$ initial abstract task

$s _ { I } \subseteq L ,$ , the initial state

• g the goal condition (possibly empty).

We interpret each primitive task (action) $a \in O$ as an action with preconditions and effects over L, i.e, $p r e ( a )$ $a d d ( a ) , d e l ( \bar { a } ) \subseteq L$ . This allows us to properly reason about adders and deleters. For proposition $\bar { l } \in \bar { L }$ , let adders $\left( l \right) =$ $\{ a \in O \mid l \in a d d ( a ) \}$ , and deleters $\left. l \right. = \{ a \in O \mid l \in$ $\bar { d e l } ( a ) \}$

Definition 3. (Decomposition). A decomposition refines an abstract task t of a given task network $t n _ { 1 }$ using a method $m = ( c , t n _ { m } )$ with $\alpha ( t ) = c$ resulting in a new task network $t n _ { 2 } .$ The decomposition replaces t with $t n _ { m }$ . We write $t n _ { 1 }$ $ t n _ { 2 }$ for a single decomposition step and $t n _ { 1 }  _ { D } ^ { * } t n _ { 2 }$ for an arbitrary number of decompositions.

We use a standard notion of decomposition trees of Geier and Bercher (2011), however, we adopt a simplified formulation that is sufficient for our purposes.

Definition 4. (Decomposition tree). Let $D _ { t }$ be a decomposition tree that captures the refinement of an initial task network into a primitive task network. The root is the initial abstract task $c _ { I }$ the primitive tasks are the leaves of the tree. The inner nodes represent the abstract tasks together with the methods.The tree induces a partial-order ≺ over its nodes.

Definition 5. (HTN solution). A task network $t n _ { s }$ is a solution to an HTN planning problem $\mathcal { P }$ if and only if:

1. $t n _ { s }$ is a primitive task network ,

2. $t n ( c _ { I } )  _ { D } ^ { \ast } t n _ { s } .$

3. Any linearization of $t n _ { s }$ is executable in $s _ { I }$ and satisfies g.

In this paper, we represent the HTN solution as $\left. \bar { a } , \prec , D _ { t } \right.$ where $\bar { a } \subseteq O$ is a set of primitive tasks (actions); ≺ the ordering relations on these actions; $D _ { t }$ represents the decomposition tree which captures the refinement of abstract tasks. $\mathsf { L e t } , \prec _ { H } \subseteq D _ { t }$ denote the mandatory ordering constraints that are induced by the decomposition methods. Since the constraints are part of $D _ { t }$ they should be preserved.

## HTN-PRF Algorithm

In this section, we extend the classical PRF algorithm of Backstr ¨ om (1998) by integrating the mandatory constraints¨ induced by the decomposition hierarchy. The PRF algorithm relies on simple concurrency to determine whether two distinct actions, $a _ { i }$ and $a _ { j }$ can execute in parallel. Their ordering can only be removed if they are not constrained by nonconcurrency conditions.

Algorithm 1: HTN-PRF algorithm   
Input: Htn planning problem ${ \mathcal { P } } _ { : }$ , HTN solution $\left. \bar { a } , \prec , D _ { t } \right.$   
Output: A valid PO-HTN solution   
1: for all $a _ { i } , a _ { j } \in \bar { a }$ such that $a _ { i } \prec a _ { j }$ do   
2: if $( a _ { i } , a _ { j } ) \in \prec _ { H }$ then   
3: Order $a _ { i } \prec ^ { \prime } a _ { j }$   
4: else if $a _ { i } \# a _ { j }$ then   
5: Order $a _ { i } \prec ^ { \prime } a _ { j }$   
6: end if   
7: end for   
8: return $\langle \bar { a } , \prec ^ { \prime + } , D _ { t } \rangle$

Definition 6. (Simple concurrency). Let $a _ { i }$ and $a _ { j }$ be two distinct actions. Actions $a _ { i }$ and $a _ { j }$ are considered nonconcurrent, denoted by $a _ { i } \# a _ { j }$ if there exists an $l \in L$ such that:

$$
1 . ~ l \in { a d d ( a _ { i } ) } \land l \in { p r e ( a _ { j } ) }
$$

$$
2 . \ l \in a d d ( a _ { i } ) \land l \in d e l ( a _ { j } )
$$

$$
3 . \ l \in p r e ( a _ { i } ) \land l \in d e l ( a _ { j } )
$$

If any of the above conditions holds for any l, then we preserve the ordering constraint between $a _ { i }$ and $a _ { j }$ . In HTN planning, actions implicitly inherit the ordering constraints imposed by decomposition methods used during task refinement. These constraints should also therefore be conserved when deordering a plan.

Algorithm 1 iterates over all ordering constraints in the primitive HTN solution. For each pair of ordered primitive actions $a _ { i } ~ \prec ~ a _ { j }$ , the algorithm initially checks whether the ordering is induced by the decomposition tree. If so, the ordering is kept. Otherwise, the algorithm continues to perform the simple concurrency check. If the two actions are non-concurrent according to simple concurrency criterion, then we keep the ordering. Otherwise, the ordering between $a _ { i }$ and $a _ { j }$ is removed. The algorithm returns a partial-order HTN solution $( \langle \bar { a } , { \prec } ^ { \prime + } , D _ { t } \rangle )$ where a¯ is a set of actions, $\prec ^ { \prime + } \subseteq \prec$ is a transitive closure of the ordering constraints and $D _ { t }$ is the decomposition tree. Thus, the HTN-PRF algorithm only removes ordering constraints between $a _ { i }$ and $a _ { j }$ if the ordering is neither induced by the decomposition methods nor required by non-concurrency criteria.

Although the HTN-PRF deordering algorithm produces a partial-order plan, it does not guarantee an optimal deordering. To address this limitation, we propose an extension of the partial weighted MaxSAT algorithm of (Muise, McIlraith, and Beck 2012), which we refer to as HTN-MaxSAT deordering.

## HTN-MaxSAT deordering

To generate optimal deorderings, we encode the deordering problem as a partial weighted MaxSAT problem. Given an HTN planning problem $\bar { \mathcal { P } } = ( L , C , O , \bar { M ^ { } } , c _ { I } , s _ { I } , g )$ , let: $\cdot \bar { a } \subseteq O$ be the primitive tasks (actions) of an HTN plan, • $a _ { I }$ be an initial dummy action,

• $a _ { G }$ be a dummy goal action.

For every pair of actions $a _ { i } , a _ { j } \in \bar { a } ,$ , we represent an ordering constraint between them using a variable $\kappa ( a _ { i } , a _ { j } )$ The encodings of Muise, McIlraith, and Beck (2012) used an action-membership variable to represent an action present in the partial-order plan. However, in our setting, this is not necessary as every action in a¯ must remain present in the final partial-order plan after deordering. Since all actions in a¯ are part of a valid HTN solution generated through the decomposition hierarchy, they must be preserved to maintain correctness.

Partial weighted MaxSAT allows us to differentiate between two types of clauses, namely, hard clauses and soft clauses. The objective of partially weighted MaxSAT is to satisfy all hard clauses and maximize the total weight associated with the soft clauses. We begin by presenting the hard clauses of our encoding.

Let $\prec _ { H }$ be the set of mandatory ordering constraints induced by the decomposition methods. We must preserve all ordering constraints in $\prec _ { H } \colon$

$$
\kappa ( a _ { i } , a _ { j } ) , \quad \forall ( a _ { i } , a _ { j } ) \in \prec _ { H }\tag{1}
$$

Furthermore, we enforce additional structural properties of a valid partially ordered plan:

$$
\left( \neg \kappa ( a _ { i } , a _ { i } ) \right)\tag{2}
$$

$$
\kappa ( a _ { I } , a _ { i } ) \wedge \kappa ( a _ { i } , a _ { G } )
$$

$$
\kappa ( a _ { i } , a _ { j } ) \wedge \kappa ( a _ { j } , a _ { k } ) \to \kappa ( a _ { i } , a _ { k } )\tag{3}
$$

(4)

(2) forbids self loops; (3) ensures that each action is between $a _ { I }$ and $a _ { G }$ , we also assume $a _ { I } \neq a _ { i } \neq a _ { G } ; ( 4 )$ captures the transitive relations of the ordering constraints; the combination of (2) and (4) ensures our partial-order plan will be acyclic. For causal correctness, we define the following:

$$
\gamma ( a _ { i } , a _ { j } , l ) \equiv \bigwedge _ { a _ { k } \in \mathrm { d e l e t e r s } ( l ) } \left( \kappa { \left( a _ { k } , a _ { i } \right) } \vee \kappa { \left( a _ { j } , a _ { k } \right) } \right)\tag{5}
$$

which forbids any action $a _ { k }$ that deletes l from occurring between $a _ { i }$ and $a _ { j }$ when $a _ { i }$ produces l for $a _ { j }$ . Additionally, every precondition should be supported:

$$
\bigwedge _ { l \in p r e ( a _ { j } ) } \bigcup _ { a _ { i } \in \mathrm { a d d e r s } ( l ) } ( \kappa ( a _ { i } , a _ { j } ) \wedge \gamma ( a _ { i } , a _ { j } , l ) )\tag{6}
$$

A combination of (5) and (6) enforces the condition that for each precondition l of $a _ { j }$ , there exists an action $a _ { i }$ which adds l and precedes $a _ { j }$ , while forbidding any action $a _ { k }$ that deletes l from occurring between $a _ { i }$ and $a _ { j }$

To conclude the hard clauses, we ensure we only remove unnecessary ordering constraints without reversing the original ordering in our sequential plan. Assuming our sequential plan was originally ordered as $a _ { 0 } , \ldots , a _ { k }$ then,:

$$
\neg \kappa ( a _ { j } , a _ { i } ) , \quad 0 \leq i < j \leq k\tag{7}
$$

For optimization, we add a soft unit clause for each ordering constraint that neither contains $a _ { I }$ nor $a _ { G }$ and is not induced by the hierarchy:

$$
w ( \neg \kappa ( a _ { i } , a _ { j } ) ) = 1\tag{8}
$$

## Evaluation

We conducted the experiments on a Linux laptop with an Intel(R) Core(TM) Ultra 9 185H processor and 32GB of RAM. Each problem instance was limited to a runtime of 30 minutes and 10GB of memory. We evaluate the effectiveness of our implementations using two metrics: (i) the number of ordering constraints, (ii) the length of the critical path. In this paper, we consider the critical path length as the longest chain of sequential actions. We compare our resulting partial-order plans to those produced by Optiplan (Firsov 2025). For our implementations<sup>1</sup>, we also report the percentage improvements relative to the original plans denoted as %Diff in our tables.

We generate sequential plans using PANDA (Holler 2023)¨ which are then used as input to our HTN-PRF and HTN-MaxSAT implementations. We conduct our experiments using the IPC 2023 Partial-Order HTN benchmarks. When we compare Optiplan to our implementations we only report domains and problem instances that are solved by all the approaches.

The values shown in the tables are the mean value for each of the domains. The percentage improvement (%Diff) for both the number of ordering constraints and critical path length are independently computed per domain relative to the original sequential plan. Given an init value (Init) and a deordered value (Final), we calculate (Init − Final)/Init × 100. The last row of the tables contain weighted values over all problem instances computed by weighing the mean by the number of instances.

To enhance readability we adopt a short-hand notation for our implementations, $\mathrm { H } _ { P }$ refers to HTN-PRF, whereas $\mathrm { H } _ { M }$ and $\mathrm { O } \dot { \mathrm { P } }$ refer to HTN-MaxSAT and Optiplan respectively.

<table><tr><td rowspan="2">Domain</td><td>mean constraints</td><td>mean critical path</td></tr><tr><td> $\overline { { \mathbf { H } _ { P } } }$   $\overline { { \mathbf { H } _ { M } } }$  OP</td><td>HP HM OP</td></tr><tr><td rowspan="4">PCP (9) Rover (4) Satellite (25)</td><td>240.77240.77 243.56</td><td>21.11 21.11 22.33</td></tr><tr><td>32 28.50522.75</td><td>7.00 6.70 25.75</td></tr><tr><td>46.12 46.12 30.32</td><td>8.72 8.72 6.60</td></tr><tr><td>188.64188.64155.36</td><td>16.5516.5512.64</td></tr><tr><td>Um-Translog (22)</td><td>138.55138.55138.22</td><td>15.00 15.0015.81</td></tr><tr><td>ALL (71)</td><td>120.72 120.52137.90</td><td>13.3513.3413.46</td></tr></table>

Table 1: Mean ordering constraints and mean critical path of the partial-order plans. Numbers next to the domain names are the sum of problem instances evaluated for all methods

Table 1 reports the mean number of ordering constraints together with the mean critical path length of our two HTN approaches and Optiplan. In comparison, HTN-MaxSAT produces a plan with marginally lower mean number of ordering constraints to that of HTN-PRF. However, the difference is more pronounced when compared to Optiplan. This is despite the fact that Optiplan directly outputs a partial-order plan and the HTN approaches use a sequential plan generated by PANDA. The higher number of ordering constraints observed in Optiplan is due to Optiplan generating larger plans than those by PANDA. Across all approaches, the mean critical path length is comparable, with

HTN-MaxSAT producing a partial-order plan with a slightly shorter mean critical path.

<table><tr><td rowspan="2">Domain</td><td> $\overline { { \mathbf { H } _ { P } } }$ </td><td></td><td> $\overline { { \mathbf { H } _ { M } } }$ </td><td></td></tr><tr><td>Init Final</td><td>%Diff</td><td>Init Final</td><td>%Diff</td></tr><tr><td>PCP (9)</td><td>240.77 240.77</td><td>0</td><td>240.77 240.77</td><td>0</td></tr><tr><td>Rover (4)</td><td>39 32</td><td>17.95</td><td>39</td><td>28.50 26.92</td></tr><tr><td>Satellite (25)</td><td>63.7646.12</td><td>27.67</td><td>63.76 46.12 27.67</td><td></td></tr><tr><td>Transport (11)</td><td>214.73188.6412.15</td><td></td><td>214.73188.6412.15</td><td></td></tr><tr><td>Um-Translog (22)</td><td>146.73138.55</td><td>5.57</td><td>146.73138.55</td><td>5.57</td></tr><tr><td>ALL (71)</td><td>133.90120.72</td><td>9.84</td><td>133.90120.52</td><td>9.99</td></tr></table>

Table 2: Ordering constraints percentage improvements

Table 2 shows that apart from the PCP domain, both HTN-PRF and HTN-MaxSAT achieve significant reductions in the number of ordering constraints in the partial-order plan. In particular, the largest improvement is observed in the Satellite domain using HTN-MaxSAT algorithm with a mean reduction of 27.67%. HTN-MaxSAT performed at least as well as HTN-PRF in all domains and notably outperforms it in the Rover domain by close to 9%. In domains such as Transport, we observe a correlation between number of trucks and number of ordering constraints removed. However, PANDA did not find solutions for most of the harder problems in Transport, which limits the observed improvements.

<table><tr><td rowspan="2">Domain</td><td> $\overline { { \mathbf { H } _ { P } } }$ </td><td colspan="2"> $\overline { { \mathbf { H } _ { M } } }$ </td></tr><tr><td>Init Final %Diff</td><td>Init Final %Diff</td><td></td></tr><tr><td rowspan="3">PCP (9) Rover (4) Satellite (25) Transport (11)</td><td>21.11 21.11 0</td><td>21.11 21.11</td><td>0</td></tr><tr><td>9.25 7.00 24.32</td><td>9.25 6.70 27.57</td><td></td></tr><tr><td>10.96 8.72 20.44 20 16.55 17.25</td><td>10.96 8.72 20.44 20 16.55 17.25</td><td></td></tr><tr><td>ALL (71)</td><td>15.60 15.00 3.85 14.99 13.35 10.91</td><td>15.60 15.00 3.85 14.99 13.34 11.03</td><td></td></tr></table>

Table 3: Critical path percentage improvements

The reduction in critical path length in Table 3 follow a similar pattern to the reductions in ordering constraints in Table 2. The largest percentage reduction is observed in the Rover domain. In contrast, the PCP domain remains totally ordered, with no reduction in both ordering constraints or critical path length.

## Conclusion

In this paper, we proposed two approaches for HTN plan deordering. The first, HTN-PRF, preserves ordering constraints enforced by the hierarchy or required by simple concurrency and removes all others. The second, HTN-MaxSAT, encodes the problem as a partially weighted MaxSAT formulation and guarantees a minimal deordering for a given input plan.

Our results show that both approaches significantly reduce ordering constraints and the critical path length, resulting in more flexible partial-order plans. However, their effectiveness depends heavily on the quality of the input sequential plan. Although larger reductions were observed on harder instances, most harder instances were unsolved.

## Acknowledgments

This work has been supported by the French National Research Agency (ANR) under the project ANR-24-CE10- 0893 (Prod4Human).

## References

Alford, R.; Behnke, G.; Holler, D.; Bercher, P.; Biundo, S.;¨ and Aha, D. W. 2016. Bound to Plan: Exploiting Classical Heuristics via Automatic Translations of Tail-Recursive HTN Problems. In Proceedings ofthe Twenty-Sixth International Conference on Automated Planning and Scheduling, 20–28.

Backstr ¨ om, C. 1998. Computational Aspects of Reordering¨ Plans. Journal ofArtificial Intelligence Research, 9: 99–137.

Behnke, G.; Holler, D.; and Biundo, S. 2019. Bringing Or-¨ der to Chaos - A Compact Representation of Partial Order in SAT-Based HTN Planning. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence, 7520– 7529.

Behnke, G.; Holler, D.; and Biundo, S. 2019. Finding Opti-¨ mal Solutions in HTN Planning - A SAT-based Approach. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 5500–5508.

Behnke, G.; Pollitt, F.; Holler, D.; Bercher, P.; and Al-¨ ford, R. 2022. Making Translations to Classical Planning Competitive with Other HTN Planners. In Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence, 9687–9697.

Behnke, G.; and Speck, D. 2021. Symbolic Search for Optimal Total-Order HTN Planning. In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence, 11744–11754.

Bercher, P.; Haslum, P.; and Muise, C. 2024. A Survey on Plan Optimization. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, 7941– 7950.

Bercher, P.; and Olz, C. 2020. POP ≡ POCL, right? Complexity Results for Partial Order (Causal Link) Makespan Minimization. In Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence, 9785–9793.

Bylander, T. 1996. A Probabilistic Analysis of Propositional STRIPS Planning. Journal of Artificial Intelligence, 81: 241–271.

Erol, K.; Hendler, J. A.; and Nau, D. S. 1994a. HTN Planning: Complexity and Expressivity. In Proceedings of the twelfth National Conference on Artificial Intelligence, 1123–1128.

Erol, K.; Hendler, J. A.; and Nau, D. S. 1994b. UMCP: A Sound and Complete Procedure for Hierarchical Tasknetwork Planning. In Proceedings of the Second International Conference on Artificial Intelligence Planning Systems, 249–254.

Firsov, O. 2025. Automation of decision-making in Industry 4.0 with CSP-based temporal hierarchical planning. Ph.D. thesis, Universite Grenoble Alpes, Grenoble, France.´

Geier, T.; and Bercher, P. 2011. On the Decidability of HTN Planning with Task Insertion. In IJCAI 2011, Proceedings of the Twenty-second International Joint Conference on Artificial Intelligence, 1955–1961.

Georgievski, I.; and Aiello, M. 2015. HTN planning: Overview, comparison, and beyond. Artificial Intelligence, 222: 124–156.

Holler, D. 2023. The PANDA progression system for HTN¨ planning in the 2023 IPC. In Proceedings ofthe Eleventh International Planning Competition: Planner Abstracts-HTN Planning Track.

Holler, D.; Bercher, P.; Behnke, G.; and Biundo, S. 2018.¨ A Generic Method to Guide HTN Progression Search with Classical Heuristics. In Proceedings of the Twenty-Eighth International Conference on Automated Planning and Scheduling, 114–122.

Holler, D.; Bercher, P.; Behnke, G.; and Biundo, S. 2019.¨ On Guiding Search in HTN Planning with Classical Planning Heuristics. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 6171– 6175.

Holler, D.; Bercher, P.; Behnke, G.; and Biundo, S. 2020.¨ HTN Planning as Heuristic Progression Search. Journal of Artificial Intelligence Research, 67: 835–880.

Kambhampati, S.; and Kedar, S. 1994. A Unified Framework for Explanation-Based Generalization of Partially Ordered and Partially Instantiated Plans. Journal of Artificial Intelligence, 29–70.

Muise, C. J.; Beck, J. C.; and McIlraith, S. A. 2016. Optimal Partial-Order Plan Relaxation via MaxSAT. Journal of Artificial Intelligence Research, 57: 113–149.

Muise, C. J.; McIlraith, S. A.; and Beck, J. C. 2012. Optimally Relaxing Partial-Order Plans with MaxSAT. In Proceedings ofthe Twenty-Second International Conference on Automated Planning and Scheduling, 358–362.

Quenard, G.; Pellier, D.; and Fiorino, H. 2024. SibylSat: Using SAT as an Oracle to Perform a Greedy Search on TO-HTN Planning. In Twenty-Seventh European Conference on Artificial Intelligence,, 4157–4164.

Quenard, G.; Pellier, D.; and Fiorino, H. 2025. SibylSatOpt: a MaxSAT-based Greedy Optimal Search for TOHTN Planning. In Proceedings ofthe International Conference on Automated Planning and Scheduling, 236–244.

Schreiber, D. 2021. Lilotane: A Lifted SAT-Based Approach to Hierarchical Planning. Journal of Artificial Intelligence Research, 70: 1117–1181.

Schreiber, D.; Pellier, D.; Fiorino, H.; and Balyo, T. 2019. Tree-REX: SAT-Based Tree Exploration for Efficient and High-Quality HTN Planning. In Proceedings of the Twenty-Ninth International Conference on Automated Planning and Scheduling, 382–390.

Siddiqui, F. H.; and Haslum, P. 2015. Continuing Plan Quality Optimisation. Journal of Artificial Intelligence Research, 54: 369–435.