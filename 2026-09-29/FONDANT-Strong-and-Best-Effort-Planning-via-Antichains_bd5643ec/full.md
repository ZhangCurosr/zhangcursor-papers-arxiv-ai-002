# FONDANT: Strong and Best-Effort Planning via Antichains

Benjamin Aminof<sup>2</sup>, Tuan Khai Nguyen<sup>1</sup>, Sasha Rubin<sup>1</sup>

<sup>1</sup>School of Computer Science, The University of Sydney, Australia <sup>2</sup>Technical University of Vienna, Austria

## Abstract

A classical solution concept in fully observable nondeterministic (FOND) planning, is the strong policy (aka winning strategy in the closely related area of reactive synthesis), i.e., such a policy ensures that the goal is reached in an adversarial environment. When strong policies are not available or there is no evidence that the environment is adversarial, one can resort to best-effort policies, which always exist, and which follow the classic decision-theoretic principle that an agent should not use a dominated strategy. A typical positional besteffort policy works as follows: from every state, it follows a strong policy if one exists from that state (such states are called “strong-winning”), else a weak policy if one exists from that state (“weak-winning”), and else is unconstrained (“losing”). In this work, we introduce a sound and complete planner for both best-effort planning and strong planning. The algorithm that underpins the planner is quite simple: it repre sents certain sets of states, such as the winning regions, by their ⊆-minimal elements. We represent positional policies by a set of instructions of the form $\langle x , a , r \rangle$ where x is a state, a is an action, and r is a rank; the induced policy, from a given state s, finds an instruction $\langle x , a , r \rangle$ with smallest rank r amongst those for which $x \subseteq s ,$ and does action a. The algorithm returns uniform policies, i.e., it returns a policy π that is a strong solution starting in every strong-winning state, and it returns a policy $\pi _ { w }$ that is a weak solution starting in every weak-winning state, and it provides a certificate for the set of losing states. We implemented the algorithm with some simple optimizations (calling it FONDANT), and evaluated it on a benchmark set consisting of the instances that were used in the evaluation of leading strong planners PR2 and FOND-SAT, and the best-effort planner BeSyftP. On coverage, our implementation is at least as good on all domains, and outperforms on some domains; and on wall time, it is slower on small and medium-sized instances, and outperforms on larger instances.

## 1 Introduction

In fully observable nondeterministic (FOND) planning, an agent’s actions may have several possible effects. A classic solution concept for such models is the strong policy, which (if it exists) guarantees reaching the goal against every outcome. What should an agent do when there is no such policy? Rather than assuming the environment is co-operative (which would mean that weak policies are relevant), or fair (which would mean that strong-cyclic or stochastic besteffort policies are relevant), another line of work (that makes no such assumptions) has proposed and studied best-effort solutions (Berwanger 2007; Faella 2009; Brenguier, Raskin, and Sassolas 2014; Aminof, De Giacomo, and Rubin 2021; Muvvala, Amorese, and Lahijanian 2022; De Giacomo, Parretti, and Zhu 2023).

Best-effort policies follow the decision-theoretic principle of not using a dominated policy. That is, one should not use a policy for which there is a policy that performs just as well against all ”environments” (functions that resolve the nondeterminism at every history), and strictly better in some. Most importantly, unlike strong or strong-cyclic solutions, such solutions always exist. In fact, it turns out (cf. (Aminof, De Giacomo, and Rubin 2021)) that there is always a positional best-effort policy $\pi _ { b e }$ that is induced by two positional policies: a uniform strong policy $\pi _ { t } \ ( \mathrm { i . e . }$ , one that is strong from every state in the set $T$ of states starting from which there is a strong policy), and a uniform weak policy $\pi _ { w }$ (i.e., one that is weak from every state in the set $W$ of states starting from which there is a weak policy). Indeed, the policy is very simple to define from these uniform policies:

$$
\pi _ { b e } ( s ) : = { \left\{ \begin{array} { l l } { \pi _ { t } ( s ) } & { { \mathrm { ~ i f ~ } } s \in T } \\ { \pi _ { w } ( s ) } & { { \mathrm { ~ i f ~ } } s \in W \setminus T } \end{array} \right. }
$$

We introduce FONDANT, a planner for strong and besteffort FOND planning that relies on the observation that winning regions are upward-closed, and so can be represented by their set of minimal elements – called antichains. The planner grows an antichain of states, as well as one instruction per state in the antichain. An instruction is a triple $\langle x , a , r \rangle$ consisting of a set x of atoms, an action a that can be executed from every state s with $x \subseteq s ,$ , and a rank r. The produced set R of instructions induced a policy as follows: given a state $s ,$ find an instruction $\langle x , a , r \rangle$ with smallest rank amongst those for which $x \subseteq s ,$ , and do action a.

Strong planning and best-effort planning in FOND are EXPTIME-complete (Rintanen 2004; De Giacomo, Parretti, and Zhu 2023). These bounds are tight, so the problem is to achieve the best-possible speed under the these bounds. We implemented the algorithm, and incorporated some classic optimizations, i.e., mutex groups, $h ^ { 2 }$ pruning, and ${ h ^ { \mathrm { m a x } } }$ style heuristic selection.

For strong planning, we evaluated our tool (called FON-DANT) on a set of 222 instances from eight strong-planning domains that were used in the evaluation of leading and influential strong-planners, i.e., PR2 (Muise, McIlraith, and Beck 2024) and FOND-SAT (Geffner and Geffner 2018). For best-effort planning, we evaluated our tool on a set of 47 instances from the public repository of the best-effort synthesizer BeSyftP. (De Giacomo, Parretti, and Zhu 2023).

## 1.1 Related Work

Classical solution concepts for FOND planning include strong policies, weak policies, and strong-cyclic policies. The winning regions for such solutions (i.e., the set of states from which there is such a solution), have fixpoint characterizations (Cimatti et al. 2003).

An early planner for such solution concepts, MBP, was built on top of the symbolic model checker NuSMV (Cimatti et al. 2003).

Some later planners did not aim to compute the winning regions. Methods for strong FOND have since been based on explicit AND/OR and regression search (Ram´ırez and Sardina 2014), on classical-planning reformulations solved˜ by repeated replanning (Muise, McIlraith, and Beck 2012, 2024), and on SAT-based controller synthesis (Geffner and Geffner 2018). The runtime of the latest replanning planner dominates in practice (Muise, McIlraith, and Beck 2024).

PRP leverages the concept of grouping states into families for which the same action can be applied (Muise, McIlraith, and Beck 2012), and PR2 represents policies by controller DAGs whose nodes are partial state-action pairs (Muise, McIlraith, and Beck 2024). Similarly, our representation of policies also groups states, but the action to be executed from state s requires some search (to find an instruction, amongst those that are subsumed by s, with the smallest rank).

SAT-based synthesis (Geffner and Geffner 2018) decides, through a compact encoding whose size does not grow with the state space, whether a controller within a given controller-size bound exists (taking the bound to be the size of the state space, it is also complete).

Best-effort solutions were studied in the games-ongraphs community (Berwanger 2007; Faella 2009; Brenguier, Raskin, and Sassolas 2014), and later studied into the context of temporal synthesis (Aminof, De Giacomo, and Rubin 2021), and robotics (Muvvala, Amorese, and Lahijanian 2022). A recent work provided a tool for finding besteffort policies for FOND problems with LTLf goals (De Giacomo, Parretti, and Zhu 2023).

Finally, antichain/upward-closed representations have been used in synthesis, automata, and games on graphs. E.g., De Wulf, Doyen, Henzinger, and Raskin (Wulf et al. 2006) proposed a least fixed point algorithm on the lattice of antichains of state sets to check the universality of finite automata, avoiding the cost of explicit determinization. Later, an incremental antichain algorithm that makes use of the problem structure is proposed in the context of LTL synthesis to solve the LTL realizability problem (Filiot, Jin, and Raskin 2009). That algorithm reduces LTL realizability to a safety game, and both it and our work are built on the same antichain concept: the winning region is closed under a game-simulation order, so antichains fully represent it; in particular, their safety objective is the counterpart of our strong objective. We differ in domain and how the antichain is grown at each step.

## 2 Preliminaries

In this section, we fix our notation, provide preliminaries on FOND planning and the relevant solution concepts, and some elementary facts about upward-closed sets and antichains.

For a sequences $x = x _ { 0 } x _ { 1 } x _ { 2 } \cdot \cdot \cdot$ , we write $x _ { < i }$ for the prefix $x _ { 0 } x _ { 1 } \cdot \cdot \cdot x _ { i - 1 }$ of length i.

## 2.1 FOND planning problems

A fully-observable non-deterministic (FOND) planning problem is a tuple $\Pi = \langle A t , I , G , A \rangle$ , where At is a finite set of propositional atoms, $I \subseteq A t$ is the initial condition, $G \subseteq A t$ is the goal condition, and A is the set of actions. Every action $\textit { a } = \langle p r e _ { a } , E _ { a } \rangle$ consists of a precondition $p r e _ { a } \subseteq A t$ and a finite non-empty set $E _ { a } = \{ e _ { 1 } , \ldots , e _ { K _ { a } } \}$ of effects. Each effect $\boldsymbol { e } \ : = \ : \langle a d d _ { e } , d e l _ { e } \rangle$ consists of delete atoms $d e l _ { e } \subseteq A t$ and add atoms $a d d _ { e } \subseteq A t$

The semantics of FOND are as follows. A domain state (aka state) is a function $s : A t  \{ \mathbf { t r u e } , \mathbf { f a l s e } \}$ . The set of all states is denoted S. The initial state, denoted $s ^ { I } .$ maps p to true iff $p \in I . \mathrm { ~ A ~ }$ state s is a goal state if $s ( p ) = \mathbf { t r u e }$ for every $p \in G$ . From now on, we will also identify a state s with the set $s ^ { - 1 } ( \mathbf { t r u e } )$ of atoms in it that are true. An action a is applicable in a state s if $p r e _ { a } \subseteq s .$ The successor of a state s under effect $\boldsymbol { e } = \langle a d d _ { e } , d e l _ { e } \rangle$ is $s [ e ] : = ( s \setminus d e l _ { e } ) \cup a d d _ { e } . \mathrm { A }$ trajectory is an alternating sequence $\tau = s _ { 0 } , a _ { 0 } , e _ { 0 } , s _ { 1 } , . . .$ . where $( \mathrm { i } ) a _ { i }$ is applicable in $s _ { i } .$ (ii) $e _ { i } \in E _ { a _ { i } }$ , and (iii) $\begin{array} { r } { s _ { i + 1 } = s _ { i } [ e _ { i } ] ; } \end{array}$ ; a history h is a finite trajectory. Write last(h) for the last character of h. A state s is reachable from $s _ { 0 }$ if there is a history h starting from $s _ { 0 }$ with $l a s t ( h ) = s . \operatorname { A }$ trajectory $\tau = s _ { 0 } , a _ { 0 } , e _ { 0 } , . . .$ is goal reaching if there is some i such that $s _ { i }$ is a goal state.

An agent history is one that ends in a state. An agent policy (aka policy) π is a mapping from agent histories to applicable actions, i.e., for an agent history h, the action $\pi ( h )$ needs to be applicable in the state last(h). A trajectory $\tau = s _ { 0 } , a _ { 0 } , e _ { 0 } , . . .$ . and a policy π are consistent if for every agent history $\tau _ { < i }$ that is a prefix of $\tau ,$ we have that $\pi ( \tau _ { < i } ) = \tau _ { i } , \mathrm { i . e . }$ , the policy selects the next action as specified on the trajectory. A policy is positional (aka memoryless) $\operatorname { i f } \pi ( h )$ only depends on $l a s t ( h )$ , for every agent history $h ;$ in this case, we overload notation and write $\pi : S  A .$ For convenience, we allow policies (resp. positional policies) to be partial functions π: such a function π is assumed to denote a policy that agrees with π on its domain, and is arbitrarily defined on all histories (resp. states) not in the domain of π.

## 2.2 Solution concepts for FOND planning problems

In this section we recall the following solution concepts for FOND planning problems: strong solutions and weak solutions (Cimatti et al. 2003), best-effort solutions (Berwanger 2007; Aminof, De Giacomo, and Rubin 2021).

An environment history is a history that ends in an action. An environment choice-function (aka environment) δ maps environment histories to effects, i.e., if last(h) is the action $^ { a , }$ then $\delta ( h ) \in E _ { a \cdot } \operatorname { A }$ trajectory $\tau = s _ { 0 } , a _ { 0 } , e _ { 0 } , . . .$ . and an environment δ are consistent if for every environment history $\tau _ { < i }$ that is a prefix of $\tau ,$ we have that $\pi ( { \boldsymbol { \tau } } _ { < i } ) ~ = ~ \tau _ { i } ,$ $\mathrm { i . e . } .$ , the environment resolves the nondeterminism as specified on the trajectory. An agent policy $\pi ,$ an environment choice-function $\delta ,$ and a state s together determine a trajectory $p l a y ( \pi , \delta , s )$ called the play (resulting from $\pi , \delta , s ) , \mathrm { i . e . }$ the unique trajectory consistent with both π, δ starting from s.

An agent policy π is strong (resp. weak) winning from s iff every (resp. some) environment choice-function $\delta ,$ , the trajectory $p l a y ( \pi , \delta , s )$ is goal reaching.<sup>1</sup>

The strong winning-region is the set $T$ of states s for which there exists an agent policy π that is strong winning from s. The states in $\check { T }$ are called strong-winning states. A uniformly strong-winning policy is a policy that is strong winning from every state $s \in T$ . By a classic fixpoint contruction (cf. (Cimatti et al. 2003; Fijalkow 2026)), T can be constructed by: $T = \cup _ { i > 0 } T _ { i }$ where $T _ { 0 }$ is the set of goal states, and $T _ { i + 1 }$ is defined as the union of $T _ { i }$ and the universal preimage of $T _ { i }$ :

$$
\left\{ s \in S : \exists a \in A , p r e _ { a } \subseteq s \land \forall e \in E _ { a } , s [ e ] \in T _ { i } \right\}\tag{1}
$$

We call T the i-th layer of the strong-winning region. There exists $j$ such that $\dot { T _ { j + 1 } } = T _ { j }$ . For state $s \bar { \in } \bar { T _ { i } } \backslash T _ { i - 1 }$ for $i > 0$ , an action a is a witnessing action of s in $T$ iff $\forall e \in$ $E _ { a } : s [ e ] \in T _ { i - 1 }$ . Denote the set of witnessing action of s in $T$ to be $W i t _ { T } ( s )$ . A positional policy π that satisfies $\pi ( s ) \in W i t _ { T } ( s )$ is a uniformly strong winning policy.

Similarly, the weak winning-region is the set W of states s such that there exists an agent policy π that is weak winning from s. The states in W are called weak-winning states. A uniformly weak policy is a policy that is weak winning from every state $s \in W$ . The attractor construction of $\breve { W }$ is as follow: $W = \cup _ { i > 0 } W _ { i }$ where $W _ { 0 }$ is the set of goal states, and $W _ { i + 1 }$ is defined as the union of $W _ { i }$ and the existential preimage of $W _ { i } { \mathrm { : } }$

$$
W _ { i } \cup \{ s : \exists a , \ p r e _ { a } \subseteq s \ \land \ \exists e \in E _ { a } , \ s [ e ] \in W _ { i } \}\tag{2}
$$

We call $W _ { i }$ the i-th layer of weak-winning region. There exists j such that $W _ { j + 1 } = W _ { j }$ . For state $s \in W _ { i } \backslash W _ { i - 1 }$ for $i > 0$ , an action a is a witnessing action of s in $\dot { W }$ iff $\exists e \in E _ { a } : s [ e ] \in W _ { i - 1 } .$ Denote the set of witnessing action of s in W to be $W i t _ { W } ( s )$ . A positional policy π such that $\pi ( s ) \in W i t _ { W } ( s )$ is a uniformly weak winning.

Fix the initial FOND domain state $s ^ { I }$ as a starting state. For policies $\pi _ { 1 } , \pi _ { 2 }$ , write $\pi _ { 1 } ~ \leq ~ \pi _ { 2 }$ (read $^ { 6 6 } \pi _ { 2 }$ dominates $\pi _ { 1 } \mathrm { ^ { * } ) }$ iff for every environment choice-function $\delta ,$ , if $p l a y ( \pi _ { 1 } , \delta , s ^ { I } )$ is goal reaching then also $p l a y ( \pi _ { 2 } , \delta , s ^ { I } )$ is goal reaching. Write $\pi _ { 1 } < \pi _ { 2 }$ (read $^ { 6 6 } \pi _ { 2 }$ strictly dominates $\bar { \pi } _ { 1 } { } ^ { , \ast } ) \mathrm { i f } \pi _ { 1 } \leq \bar { \pi } _ { 2 }$ and it is not the case that $\pi _ { 2 } \leq \pi _ { 1 }$ . A policy π is best-effort if no policy strictly dominates it. Best-effort policies always exist (Berwanger 2007; Aminof, De Giacomo, and Rubin 2021). In particular, since strong-winning policies dominate all others, if a strong-winning policy from $\bar { s } ^ { I }$ exists then the best-effort policies are exactly the strongwinning policies. Given a positional uniform strong-winning policy $\pi _ { t } ,$ , and position uniform weak-winning policy π<sub>w</sub>, define the following positional policy

$$
\pi _ { b e } ( s ) : = { \left\{ \begin{array} { l l } { \pi _ { t } ( s ) } & { { \mathrm { ~ i f ~ } } s \in T } \\ { \pi _ { w } ( s ) } & { { \mathrm { ~ i f ~ } } s \in W \setminus T } \end{array} \right. }\tag{3}
$$

Then $\pi _ { b e }$ is a best-effort policy. This follows from the graph-based algorithm (Aminof, De Giacomo, and Rubin 2021) that constructs a best-effort strategy for a linear-time temporal logic objective.

## 2.3 Upward-closed sets and Antichains

As discussed in the introduction, we will represent certain sets of interest by their minimal elements. Here we provide some basic definitions and Lemmas.

Consider the subset ordering $\subseteq \mathbf { o n } ~ S ~ = ~ 2 ^ { A t }$ (recall our convention of viewing states as sets of atoms). For a set $X \subseteq$ S of states:

1. Let $U P ( X )$ denote the upwards-closure of $X$ , i.e., $\textstyle \bigcup _ { x \in X } \{ s \in S \ : \ x \ \subseteq \ s \}$ . We overload the notation, for single state s, and simply write $U P ( s )$ instead of $U P ( \{ s \} )$

2. Let $M I N ( X )$ be the set of states in X that are minimal in the ordering, ${ \mathrm { i . e . , } } M I N ( X ) = \{ s \in X : \lnot \exists s ^ { \prime } \in X . s ^ { \prime } \subsetneqq$ $s \}$ . We call $M I N ( X )$ the basis of X.

3. Call X an antichain if for all distinct $s , s ^ { \prime } \in X$ , it is not the case that $s \subseteq s ^ { \prime } .$

The first Lemma says that upward closed sets can be represented by their minimal elements.

Lemma 1 (Bases). Let X be upward closed. Then MIN(X) is the unique antichain B with $\mathsf { \bar { U } } P ( B ) = X$

Proof. Elements of MIN(X) are ⊆-incomparable by minimality, so the family is an antichain. We have $U P ( M I N ( { \mathbf { \bar { X } } } ) ) ~ \subseteq ~ X$ holds because X is upward closed and $M I N ( X ) \subseteq X$ . On the other side, we have $X \subseteq$ $U P ( M I N ( \dot { X } ) )$ because for every $s \in X$ if s /∈ MIN(X), then there is $s ^ { \prime } \in M I N ( X )$ such that $s ^ { \prime } \subsetneq s , \ \bar { 1 } . { \mathrm e } . , \ s \in$ $U P ( M I N ( X ) )$

For uniqueness, take an antichain B with $U P ( B ) = X$ . If $y \in X$ satisfied $y \subsetneq b$ for some $b \in B$ , then $y \in U P ( B )$ would give $b ^ { \prime } \in \bar { B }$ with $b ^ { \prime } \subseteq y \subsetneq b ,$ which is a contradiction. Hence ${ \mathrm { ~ \bar { \textit { B } } \subseteq \mathop { M I N } } } ( X )$ . Conversely, $x \in M I N ( X ) \subseteq X =$ $U P ( B )$ there is $b \in B$ with $b \subseteq x ,$ , and since $b \in X$ while x is minimal in $X , b = x ;$ so $M I N ( X ) \subseteq B$ □

The second Lemma says how to build up the minimal elements of an upward closed set.

Lemma 2 (Bases Extension). Let X be upward closed, B be an antichain with $U P ( B ) \subseteq X , x \in S$ such that $U P ( x ) \subseteq$ X and x $\notin U P ( B )$ . Then $B ^ { \prime } = ( B \backslash \{ b \in B : x \subsetneq b \} ) \cup \{ x \}$ is an antichain and $U P ( B ) \subsetneq U P ( \dot { B ^ { \prime } } ) = U P ( B ) \cup U P ( x ) \ \stackrel { \circ } { \subseteq }$ X.

Proof. $B ^ { \prime }$ consists of x and of the elements of B that do not strictly contain x. The latter are pairwise incomparable, and x is comparable with none of them: $b \nsubseteq$ x because $x \notin$ $U P ( B )$ and x ̸⊆ b because of the set exclusion operation.

We have $B ^ { \prime } \subseteq B \cup \{ x \}$ gives $U P ( B ^ { \prime } ) \subseteq U P ( \bar { B } ) \cup U P ( x ) ;$ conversely, every $s \in \bar { U P } ( B ) \cup U P ( x )$ lies in $\dot { U } P ( B ^ { \prime } )$ : If $x \subseteq s$ this holds because $x \in B ^ { \prime }$ . Otherwise $s \in U P ( B )$ and there is $b \subseteq s$ that survives in $B ^ { \prime }$ because ${ \mathrm { i f ~ } } x \subsetneq { \mathit { b } }$ then $x \subsetneq s ,$ and thus $s \in U P ( x )$ which is a contradiction. Therefore $s \in U P ( B ^ { \prime } )$ in both cases. Since $x \notin U P ( B )$ , we also have that $U P ( { \overline { { B } } } ) \subsetneq U P ( B ^ { \prime } )$ □

## 3 Representing policies by instruction-sets

In this work we represent a memoryless policy by a set R of ranked instructions. A ranked instruction (or simply instruction) is a 3-tuple $\langle x , a , r \rangle ;$ : a state $x ,$ an action a that can be executed from every state s with $x \subseteq s ,$ and a non-negative integer rank r. The size of R is the number of instructions in it.

For a state s denote

$$
\mathcal { R } ( s ) = \{ \langle x , a , r \rangle \in \mathcal { R } : x \subseteq s \}\tag{4}
$$

We call $\mathcal { R } ( s )$ the applicable instructions (from R) for s. Let $R a n k _ { \mathcal { R } } ( s )$ min $\{ r : \langle x , a , r \rangle \in { \mathcal { R } } ( s ) \}$ be its minimum rank. Then the induced memoryless policy $\pi _ { \mathcal { R } }$ is

$$
\pi _ { \mathcal { R } } ( s ) = a { \mathrm { ~ s . t . ~ } } \exists \langle x , a , R a n k _ { \mathcal { R } } ( s ) \rangle \in { \mathcal { R } } ( s ) ,\tag{5}
$$

and if multiple such a exist we tie-break arbitrarily.

Now we show that this representation is complete for memoryless policies:

Proposition 1 (Completeness of the Representation). Given a memoryless policy π, there is a set R of instructions such that the induced policy $\pi _ { \mathcal { R } }$ is exactly π.

Proof. Suppose we have an assignment $r ( s )$ from each state s to a non-negative integer such that if $s \subsetneq s ^ { \prime }$ then $r ( s ) >$ $r ( s ^ { \prime } )$ .

For each state s, add the tuple $\langle s , \pi ( s ) , r ( s ) \rangle$ to R. For every state s, and for every instruction $\langle x , a , r \rangle \in \mathcal { R } ( s )$ , we have $r ( s ) < r ( x )$ if $x \neq s .$ . Thus Rank $\pi ( s ) = r ( s )$ , and $s , \pi ( s ) { \dot { , } } { \dot { r ( s ) } } \rangle$ is the only intruction in $\mathcal { R } ( s )$ with rank $r ( s )$ Therefore $\pi _ { \mathcal { R } } ( s ) = \pi ( s )$

One possible assignment is $d ( s ) = \vert A t \ \backslash$ s|, the number of atoms absent from s: ${ \mathrm { i f ~ } } s \subsetneq s ^ { \prime }$ then ${ \dot { A } } t \setminus { \dot { s } } \supset A t \setminus s ^ { \prime }$ and hence $d ( s ) > d ( s ^ { \prime } )$ □

The representation $\mathcal { R } _ { \pi }$ of $\pi$ in this proof is naive: it has an instruction for each domain state. However, for some domains and classes of policies, there are smaller instructions sets (compared to the number of states in the domain).

## 4 Planning with antichains

In this section we observe that since the winning regions are upward closed, there is the potential for compact representations of both the winning regions (as the set of their minimal elements), and of uniformly winning policies (as certain instruction sets). We first introduce an algorithm to efficiently compute the representations of the winning regions, and then show how to modify the algorithm to also retrieve a potentially compact ranked instructions set for positional uniformwinning policies. We do this for both strong-winning and weak-winning.

## 4.1 Representing Winning Region By Bases

The first Theorem states that the layers and winning regions are upwards closed. It uses the property of FOND planning problems that preconditions are monotone and that the successor updates are monotone: $( { \star } ) \operatorname { i f } s \subseteq s ^ { \prime }$ then $\mathrm { ( i ) } \ p r e _ { a } \subseteq s$ implies ${ \bar { p } } r e _ { a } \subseteq s ^ { \prime }$ , and (ii) for every effect $e , s [ e ] \subseteq s ^ { \prime } [ e ] .$ Theorem 1 (Upward closure). $F o r X \in \{ T , W \}$ and $i \geq 0 ,$ $s \in X _ { i }$ and $s \subseteq s ^ { \prime }$ imply $s ^ { \prime } \in X _ { i }$ . Hence, so are the regions $T$ and W.

Proof. Induct on the layer i.

Base case: $T _ { 0 } = { \tilde { W _ { 0 } } } = \{ s : G \subseteq s \}$ , and a superset of a state containing G contains G too.

Inductive step: let $s \subseteq s ^ { \prime }$ with $s \in T _ { i + 1 } . \mathrm { I f } \ s \in T _ { i } \ \subseteq$ $T _ { i + 1 }$ then we are done. Otherwise, there is some applicable action such that for every effect $e \in E _ { a }$ we have that $s [ e ] \in$ $T _ { i }$ . By property $( \star ) \ p r e _ { a } \subseteq s ^ { \prime }$ (and so a is applicable in $\bar { s } ^ { \prime } )$ and $s [ e ] \subseteq s ^ { \prime } [ e ]$ <sup>′</sup>[e], so $\bar { s } ^ { \prime } [ e ] \in T _ { i }$ by the induction hypothesis. So $s ^ { \prime } \in T _ { i + 1 }$

The weak-winning case is the same argument with the existential condition of (2): if $s \in W _ { i }$ we are done, otherwise one outcome $s [ e ] ~ \in ~ W _ { i }$ of an applicable action gives $s ^ { \prime } [ e ] \ \in \ W _ { i }$ by the hypothesis (a applies in $s ^ { \prime } \ \mathrm { t o o } )$ so $s ^ { \prime } \in \dot { W } _ { i + 1 }$

Since upward-closed sets are closed under union, also the region X is upwards-closed. □

Therefore by lemma 1, we can represent winning regions $T$ and W by only using their bases $\bar { M I N } ( T )$ and $M I N ( W )$ .

## 4.2 Computing the bases of the winning regions

In this section we provide an algorithm to compute the basis of the strong winning-region and then show how to modify it to compute the basis of the weak winning region.

We first introduce some notation. For a set $\bar { X }$ of states, let $P R E _ { \forall } ( X )$ consist of states of the form

$$
p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \setminus a d d _ { k } )\tag{6}
$$

where $a \in A$ is an action, $K : = | E _ { a } |$ , and for $1 \leq k \leq K$ we have $\left( \mathrm { i } \right) b _ { k } \in X , \left( \mathrm { i i } \right) b _ { k } \cap d e l _ { k } = \emptyset$ , and (iii) $b _ { k } \cap a d d _ { k } \neq \emptyset .$

Remark 1. While we will not use thisfact, it might be helpful to observe that the $U P ( P R E _ { \forall } ( X ) )$ ) is exactly the universal preimage (see (1)) of $U P ( X )$ $i . e . , f o r$ a state $s ,$ there exists an action a such that for all $e _ { k } \in E _ { a }$ we have that $s [ e _ { k } ] \in U P ( X )$ if and only if s is a superset of a state in $\mathring { P } R \mathring { E } _ { \forall } ( X )$

Algorithm I: Compute the basis of the strong-winning (resp. weak-winning) region.

The algorithm will compute a sequence of sets of states: $B _ { 0 } , B _ { 1 } , \bar { B } _ { 2 } , \cdot \cdot .$

1. Let $B _ { 0 } = \{ G \} , \mathrm { i } . \mathbf { e } . , B _ { 0 }$ consists of the set of goal atoms.

2. For $i > 0 ,$ we define $B _ { i + 1 }$ in two stages: state selection and basis extension.

(a) Select a state $x _ { i }$ from the set of candidates $C a n d ( B _ { i } ) : = M I N ( P R E _ { \forall } ( B _ { i } ) ) \backslash U P ( B _ { i } ) . ^ { 2 }$

(b) Extend the basis as follows:

$$
B _ { i + 1 } = \left( B _ { i } \setminus \{ b \in B _ { i } : x _ { i } \subsetneq b \} \right) \cup \{ x _ { i } \}\tag{7}
$$

3. We stop at iteration i when there are no candidates at that iteration, and let $B = B _ { i }$ be the resulting set of states.

By construction, B is an antichain. Moreover, as we wil prove, B is the basis of the strong-winning region.

Similarly, we have a variation of this algorithm which results in the basis B of the weak-winning region: instead of using $P R E _ { \forall } ( B _ { i } )$ , we use $P R E _ { \exists } ( B _ { i } )$ . For a set X of states $P R { \bar { E } } { \ni } ( X )$ consisting of states of the form

$$
p r e _ { a } \cup \ ( b _ { k } \setminus a d d _ { k } )
$$

where a is an arbitrary action, $K = | E _ { a } | , 1 \leq k \leq K$ $b _ { k } \in X , b _ { k } \cap d e l _ { k } = { \bar { \varnothing } }$ and $b _ { k } \cap a d d _ { k } \neq \emptyset .$ . In this case, $U P ( P R E \exists ( X ) )$ is exactly existential preimage (see (2)) of $U P ( X )$ , i.e., for a state s, there exists an action a and an effect $e _ { k } \in E _ { a }$ such that $s [ e _ { k } ] \in U P ( X )$ if and only if s is a superset of some state in $\therefore \dot { P R E } _ { \exists } ( X )$ . Again, by construction, $B$ is an antichain, and we will prove that it is the basis of the weak-winning region.

Theorem 2 (Correctness of Algorithm I). Fix any selection of candidates at each iteration. The set B produced by the algorithm is the basis of $T \left( r e s p . \ W \right)$

The rest of this section proves this Theorem.

Lemma 3 (Candidates states are strong/weak winning). Let B be an antichain such that $U P ( B ) \subseteq T ( r e s p . W ) ,$ , and let $x \in P R E _ { \forall } ( B ) \ ( r e s p . \ x \in P R E _ { \exists } ( B ) )$ . Then $U P ( x ) \subseteq T$ (resp. W).

Proof. For strong winning case: Let $x \ \in \ P R E _ { \forall } ( B _ { i } ) \ =$ $p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \setminus a d d _ { k } )$ . We have $U P ( b _ { k } ) \subseteq T$ for every k by the hypothesis on B. Take any state $s \supseteq x$ . Then $p r e _ { a } \subseteq s ,$ , so a applies in s. For every effect k:

$$
\begin{array} { l } { s [ e _ { k } ] = ( s \setminus d e l _ { k } ) \cup a d d _ { k } } \\ { \supseteq ( ( b _ { k } \setminus a d d _ { k } ) \setminus d e l _ { k } ) \cup a d d _ { k } } \\ { \qquad = ( ( b _ { k } \setminus d e l _ { k } ) \setminus a d d _ { k } ) \cup a d d _ { k } } \end{array}
$$

and because $b _ { k } \cap d e l _ { k } = \emptyset$ we have:

$$
s [ e _ { k } ] \supseteq ( b _ { k } \backslash a d d _ { k } ) \cup a d d _ { k } = b _ { k }
$$

so $s [ e _ { k } ] \in T$ for every k. Since T is closed under the operator of $( 1 ) , \mathrm { { s o \ } } s \in T$

For the weak winning case: the proof is analogous. We have that $x \in P R E _ { \exists } ( \hat { B _ { i } } ) = p r e _ { a } \ \hat { \cup } \ ( b _ { k } \ \backslash \ a d d _ { k } )$ and thus $s [ e _ { k } ] \supseteq b _ { k }$ . Therefore $s [ e _ { k } ] \in W$ . Since W is closed under the operator of $( 2 )$ , so $s \in \bar { W }$

The state s was arbitrary, we have $U p ( x ) \subseteq T \left( { \mathrm { r e s p . } } W \right)$ for the strong (resp. weak) winning case.

Now the correctness proof of our algorithm:

Proof of Theorem 2. For the convention of the proof define $B _ { - 1 } = \varnothing$ . First we prove that our algorithm terminates:

By induction on iteration $i \geq 0$ , we will show that $U P \big ( B _ { i - 1 } \big ) \subsetneq U P ( B _ { i } ) \subsetneq T \left( \mathrm { r e s p . } W \right) ( \star \star )$

Base case: For $i = 0$ , we have inductive hypothesis trivially true by definition.

Inductive case: For $i > 0$ , since $U p ( B _ { i - 1 } ) \subseteq T$ by inductive hypothesis, by Lemma 3, we have that $U P ( x _ { i - 1 } ) \subseteq$ $T .$ Because $x _ { i - 1 } \notin \forall D P ( B _ { i - 1 } )$ , by Lemma 2, we get $U P ( B _ { i - 1 } ) \subsetneq U P ( B _ { i } ) \subsetneq T { \mathrm { ~ ( r e s p . ~ } } W )$

Therefore, $| U P ( B _ { i } ) |$ grows at every iteration and $| U P ( B _ { i } ) | \ \le \ | \dot { T } | \ ( \mathrm { r e s p . } \ | \dot { W } | )$ which is finite. Thus the algorithm terminate with at most |T| (resp. |W|) iterations.

For completeness, we first provide the proof for the strong winning case: We show by induction on i that $T _ { i } \subseteq U P ( B )$ For $i = 0$ , we have $T _ { 0 } = U P ( G )$ . By applying (⋆⋆) repetitively, we have $T = U P ( G ) = U P ( \dot { B _ { 0 } } ) \subseteq U P \bar { ( } B _ { 1 } ) \subseteq \cdots \subseteq$ $U P ( { \bar { B } } )$ . For the inductive step, let $s \in T _ { i + 1 } \backslash T _ { i } ; \mathrm { b y } \left( 1 \right)$ there is an action a with $p r e _ { a } \subseteq s$ and $\forall e _ { k } \in E _ { a } , s [ e _ { k } ] \in T _ { i }$ By inductive hypothesis there are $b _ { 1 } , \dots b _ { k } \in B$ such that $\bar { b _ { k } } \subseteq s [ e _ { k } ]$ . Let

$$
x = p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \setminus a d d _ { k } )
$$

We have that $x \subseteq s$ because $b _ { k } \ \backslash \ a d d _ { k } \ \subseteq \ s [ e _ { k } ] \ \backslash $ $\scriptstyle { a d d } _ { k } \ =$ $( ( s \cup a d d _ { k } ) \setminus d e l _ { k } ) \setminus a d d _ { k } \subset s$ for every k, and $p r e _ { a } \subseteq s .$ There are two cases:

(1) Some $b _ { k }$ is disjoint from $\boldsymbol { a d d } _ { k }$ : then $b _ { k } \setminus a d d _ { k } = b _ { k }$ so $b _ { k } \subseteq x \subseteq s$ and thus $s \in U P ( B )$

(2) Otherwise $x \in P R E \exists ( B )$ . Since the algorithm terminates, $C a n d ( B ) = \varnothing , \operatorname { i . e . , } P R E \exists = \varnothing { \mathrm { ~ o r ~ } } P R E _ { \exists } ( B ) \subseteq$ $U P ( B )$ . Therefore in this case $x \in P R E _ { \exists } ( B ) \subseteq U P ( B )$ and since $x \subseteq s ,$ , we have $s \in U P ( B )$

In both case we have $s \in U P ( B )$ for any $s \in T _ { i + 1 } \setminus T _ { i }$ By inductive hypothesis we already have $T _ { i } \subseteq U P ( B )$ , and thus $T _ { i + 1 } \subseteq { \dot { U P } } ( B )$

The argument never inspects which state is chosen, so the conclusion holds under every per-iteration choice.

For the completeness proof of the weak winning case, we following the same inductive proof. For state $s \in W _ { i + 1 } \backslash W _ { i } ,$ by 2, there is and action a with $p r e _ { a } \subseteq$ s such that ∃e with $s [ e _ { k } ] \in W _ { i }$ . By inductive hypothesis there is $b _ { k }$ such that $\bar { b _ { k } } \ \bar { \subseteq } \ s [ e _ { k } ]$ , then let $x = { \overrightarrow { p r e } } _ { a } \cup \left( b _ { k } \ \backslash \ a d d _ { k } \right)$ and follow the same argument to show that $s \in U P ( B )$ and $W _ { i + 1 } \subseteq$ $U P ( B )$ □

## 4.3 Computing an instructions-set that induces a uniform strong-winning (resp. weak-winning) policy

We can modify the algorithm(s) in Section 4.2 to produce an instruction set that induces a positional uniformly strongwinning (resp. weak-winning) policy. In particular, we keep track of three more properties of each state in the basis: the rank, the witness action, and the witness state. Furthermore, the removed states from the basis at each iteration (together with the listed properties associated with those states) is still kept in memory (just not be used in the algorithm anymore) instead of removing them completely.

Algorithm II: Compute an instructions-set of a uniformly strong-winning (resp. weak-winning) policy.

1. Modify Algorithm I to store the following additional data:

(a) Rank: If candidate $x _ { i }$ is chosen at iteration i then $R a n k ( x _ { i } ) = i .$

(b) Witness action and state: For the strong-winning version. If x is a chosen candidate at iteration i, i.e.,

$$
x = p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \setminus a d d _ { k } )
$$

store alongside x the action a and all the states $b _ { k }$ for $k = 1 , \ldots , K$ . For the weak-winning version, we have $x = p r e _ { a } \cup \left( \boldsymbol { b } _ { k } \setminus a d d _ { k } \right)$ , store alongside x the action a and the state $b _ { k }$ . If there are multiple action-basis tuples that satisfy the defining property (6) for a given $x ,$ choose arbitrarily among them.

(c) Record of removed states: In the basis extension stage (7), while we may remove some states from $B _ { i } ,$ we still keep the record of these states, their ranks, and their witnessing actions and states.

2. When the iterations end, the instruction set R is constructed as follows. Add each tuple $\langle x , a , R a n k ( x ) \rangle$ where x is the final basis elements or the removed elements from the basis in some iteration, a is the recorded action of x and Rank(x) is the recorded rank.

For goal states, we use the following convention. We assign ${ \overline { { R a n k ( G ) } } } = 0$ and let a be an arbitrary action such that $p r e _ { a } \subseteq \dot { G } . $ 3

Theorem 3 (Correctness of Algorithm II). Let R be the constructed instructions-set from the strong-winning (resp. weak-winning) version of Algorithm II. Then $\pi _ { \mathcal { R } }$ is a uniformly strong-winning (resp. weak-winning) policy.

First we prove our policy is sound in the following lemma Lemma 4 (Rank descent). Let R be the constructed instructions-set from the strong-winning (resp. weakwinning) version of Algorithm II. Let s be a state with $\mathcal { R } ( s ) \ \ne \ \emptyset$ . Then every (resp. some) play of π<sub>R</sub> from s reaches a goal state within Rank<sub>R</sub>(s) steps.

Proof. Induction on $R a n k _ { \mathcal { R } } ( s )$ . If $R a n k _ { \mathcal { R } } ( s ) = 0 .$ then the only instruction of rank 0 is the one with G as its state.Thus, $G \subseteq s$ and s is a goal state, reached in zero steps. Let $R a n k _ { \mathcal { R } } ( s ) = r > 0$ and let $\langle x , a , r \rangle$ be a minimum-rank instruction such that $x \subseteq s .$ . The policy plays a, which applies in s because $\begin{array} { l l l } { p r e _ { a } } & { \subseteq } & { x } \end{array}$ . For every (resp. there is a) effect $e _ { k }$ of $^ { a , }$ the recorded basis $b _ { k }$ of x satisfies $b _ { k } \subseteq x [ e _ { k } ]$ by the same argument used in Lemma 3. Since we also have $x [ e _ { k } ] \subseteq s [ e _ { k } ]$ , we get $b _ { k } \subseteq s [ e _ { k } ]$ . Therefore

$R a n k _ { \mathcal { R } } ( s [ e _ { k } ] ) \ \leq \ R a n k ( b _ { k } ) \ < \ R a n k ( x ) \ = \ r$ . By applying inductive hypothesis to s[e<sub>k</sub>] we get that for every (resp. there exists a) play of $\pi _ { R }$ from $s [ e _ { k } ]$ reachs a goal state within $r - 1$ steps. Since this holds for every (resp. for some) effect, every (resp. there exists a) play of $\pi _ { R }$ from s reachs a goal state with in r steps. □

Now we provide the proof of Theorem 3.

Proof of Theorem 3. By Theorem 2, we have $U P ( B ) = T$ (resp. W). By the construction of R, for every $s \in U P ( B ) =$ $T _ { \ast }$ , we have $\operatorname { \mathcal { R } } ( s ) \neq \varnothing$ . Therefore, by Lemma 4, π<sub>R</sub> is strongwinning (resp. weak-winning) from s. Hence, $\pi _ { \mathcal { R } }$ is strongwinning (resp. weak-winning) from every state s in $T$ (resp. W). □

Remark 2 (Early termination for the strong-winning case). For the strong-winning case, Algorithm II may be terminated at the end ofan iteration in which $x \in C a n d ( B _ { i } )$ is selected such that $x \subseteq s ^ { I }$

## 4.4 An optimization that prunes unreachable candidate states

In this section, we discuss why for best-effort planning we do not need the whole winning regions but just the states in the regions that are reachable from $s ^ { I }$ . In the rest of this section, whenever we say reachable without specifying from which state, we mean reachable from $s ^ { I }$ . We then provide Algorithm III which is an optimized version of Algorithm II, and that removes from the candidate sets those states that are not reachable from $s ^ { I }$ . Lastly, we show that the resulting policy is still best-effort.

Let $\pi$ be a positional policy, and let U be a set that only consists of states that are unreachable from $s ^ { I }$ . Let $\pi ^ { \prime }$ denote the restriction of π to the domain $S \backslash U$ (recall our convention that a policy can be a partial function). Then, it follows from the definition of $\geq$ , that $\pi ^ { \prime } \geq \pi$ . Thus, if π is best-effort, then so is $\pi ^ { \prime } . \mathrm { S o }$ , we show how to detect if a state is unreachable from $s ^ { I }$ , and how to exclude such states from Algorithm II.

First, we introduce some terminology from (Haslum and Geffner 2000): We call a set m of atoms a mutex group iff for every reachable state s, there is an atom in m that is not in s, i.e., m $\nsubseteq s$ . We say that a state x violates the mutex group m if $m \subseteq x$ . Observe that, in particular, if x violates a mutex group m, then x is not reachable. Various techniques to compute mutex groups will be discussed in Section 5.

Algorithm III. This algorithm proceeds as in Algorithm II, except that at each iteration i, we only select a state $x \in C a n d ( \hat { B _ { i } } )$ if for all mutex groups m, the state x does not violate m.

Here, we show that removing a set of unreachable states from the candidate set at each iteration of the selection stage of algorithm II does not affect its correctness.

Lemma 5 (Violation is hereditary). Let m be a mutex group. In the strong-winning (resp. weak-winning) version $o \bar { f } A l g \bar { o } -$ rithm II, let $x \ = \ p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \ \backslash \ a d d _ { k } ) \ \in \ P R E _ { \forall } ( B _ { i } )$ (resp. $x = p r e _ { a } \cup ( b _ { k } \setminus a \bar { d } \tilde { d } _ { k } ) \ \tilde { \in } P R E _ { \exists } ( B _ { i } ) )$ . Suppose that for some $k ,$ we have that the state $b _ { k }$ violates m. Then x also violates m.

Proof. Suppose $x \subseteq s$ for some reachable (from $s ^ { I } )$ state s. Then $p r e _ { a } \subseteq x \subseteq s ,$ so a applies in s and $s [ e _ { k } ]$ is reachable. But since $b _ { k } \subseteq s [ e _ { k } ]$ and $b _ { k }$ violates m, therefore $s [ e _ { k } ]$ also violates m which is a contradiction. □

Corollary 1. Let B and R be the basis and the instruction sets obtained by using the strong-winning (resp. weakwinning) version of Algorithm III. Then there exists a set U of unreachable states such that $U P ( B ) = T \setminus U$ (resp. $W \setminus U )$ and $\pi _ { \mathcal { R } }$ is the restriction of a uniformly strongwinning (resp. weak-winning) policy on $U P ( B )$

Therefore, let $\pi _ { t } ~ ( \mathrm { r e s p . } ~ \pi _ { w } )$ be restriction of uniformly strong-winning (resp. weak-winning) policies on $U P ( B ) = { \frac { \mathbf { \bar { \tau } } } { \mathbf { \bar { \tau } } } }$ $T \setminus \bigcup _ { 1 } ^ { \backprime } \left( W \setminus \bigcup _ { 2 } \right)$ that is induced by the instruction-set from Algorithm III, where $U _ { 1 }$ (resp. $U _ { 2 } )$ is a set that only consists of unreachable states. Then,

$$
\pi _ { r b e } ( s ) : = { \left\{ \begin{array} { l l } { \pi _ { t } ( s ) } & { { \mathrm { ~ i f ~ } } s \in T \backslash U _ { 1 } } \\ { \pi _ { w } ( s ) } & { { \mathrm { ~ i f ~ } } s \in ( W \setminus U _ { 2 } ) \backslash T } \end{array} \right. }\tag{8}
$$

is also best-effort.

## 5 Implementation

We provide an implementation to produce a best-effort policy together with a classification of whether $s ^ { I }$ is strongwinning, weak-winning, or neither. The implementation combines both the strong-winning and weak-winning variations of algorithm III with some implementation optimisations to achieve more efficient runtimes.

The implementation can be broken into three major parts: precomputation, core, and finalisation. The pseudocode is given in Pseudocode 1.

## 5.1 Precomputation

To work with the pratical FOND benchmarking sets where instances are given in the PDDL format, we follow the standard approach of previous works $( \mathrm { e . g }$ . FOND-SAT (Geffner and Geffner 2018), PR2 (Muise, McIlraith, and Beck 2024)) and use the Fast-Downward system (Helmert 2006) to translate PDDL to SAS+ before parsing the SAS+ file as our input. We then compute mutex groups and the heuristic function for the selection stage of Algorithm III.

Mutex groups The implementation obtains mutex groups in two ways:

1. Structural groups: the Fast-Downward translation already produces some mutex groups during the translation process which is written in the SAS+ file that the implementation only need to parse.

2. Pairfixpoint (h2) (Haslum and Geffner 2000): The algorithm by (Haslum and Geffner 2000) is a delete-relaxed greatest fixpoint over atom pairs to produce the set of size-2 mutex groups, i.e., atoms pair $( p , q )$ such that no reachable state contains both. It is sound but incomplete for size-2 mutex groups, i.e., some pair $( p , q )$ can be a mutex group but is not produced by this algorithm. We runs this algorithm under a time-abort budget of 1% of the instance time cap and a memory-abort: its tables are quadratic in atoms, so the peak memory is estimated (for abandoning if its size exceed memory limit) before allocating anything. On either abort nothing is registered, because a partial greatest fixpoint is unsound. Pairs copresent in $\mathit { \Pi } _ { \overline { { s } } ^ { I } }$ are cleared from the start, since $s ^ { I }$ itself contains both.

Heuristic computation The heuristic function for the selection stages will be compute as follows:

For every atom $p ,$ the value $h _ { s } ( p )$ is the delete-relaxation cost of making p true from $s ^ { I }$ under ${ h ^ { \mathrm { m a x } } }$ aggregation (Bonet and Geffner 2001): atoms true in $s ^ { I }$ cost nothing, an action costs one more than the cost of its most expensive precondition, and an atom costs the least of the actions one of whose effects adds it. A single Dial-bucket Dijkstra pass over the delete-relaxed actions computes all values at once. Symmetrically, $h _ { b } ( p )$ is the delete-relaxed cost of reaching the goal from p over the reversed graph (an action’s relaxed adds become its backward preconditions, its preconditions the backward adds, and delete effects are ignored), by the same Dial pass from the goal atoms.

## 5.2 Core Algorithm

The implementation first executes the strong-winning variation of Algorithm III.

Data Structure. For the basis B, we maintain an appendonly set-trie. We use set-trie because it supports efficient insertion in $O ( | A t | )$ and coverage check (i.e., check if $x \supseteq b$ for some $b \in B )$ in $O ( | A t | )$ on average. As we can see in the description of Algorithm III, these are the two most frequently used operators. Since the removal operation for a set-trie is expensive, and since in Algorithm III we need to keep a record of basis states, instead of removing them we only label the subsumed state of x in B as “removed”. These states are not used in future operations.

Since $\begin{array} { r l r } { B _ { i } } & { { } \subseteq } & { B _ { i + 1 } , } \end{array}$ we have that $\begin{array} { r l } { P R E _ { \forall } ( B _ { i } ) } & { { } \subseteq } \end{array}$ $P R E _ { \forall } ( B _ { i + 1 } )$ Therefore, instead of regenerating $P R E _ { \forall } ( B _ { i } )$ for each iteration $i ,$ we keep a single set (implemented as a min-key heap) of $P R E _ { \forall } ( B _ { i } )$ and update it across iterations. We call it the pool. It is an invariant that at the end of every iteration i, the pool contains a set $P$ of states such that ${ \dot { P R E } } _ { \forall } ( B _ { i } ) \setminus U P ( { \dot { B } } _ { i } ) \subseteq P \subseteq P R E _ { \forall } ( B _ { i } )$ Every $x ~ \in ~ { \cal M I N } ( { \cal P R E } _ { \forall } ( B _ { i } ) ) ~ \backslash ~ { \cal U P } ( B _ { i } )$ is a valid state for selection theoretically. However, in practice, we want to select a state that is most promising with respect to the initial state since the early-stop condition of the strong variation of Algorithm III is $x \subseteq s ^ { I }$ . We do so by using a heuristic key function to order the pool.

We order the min-heap with a key function (denote as key(x)) such that key(x) is ⊊-monotone: if $x ^ { \prime } \subsetneq x$ then $\ker ( x ^ { \prime } ) \leq \ker ( x )$ with a tie-break mechanism that favors the smaller state. So, it is never the case that $x ^ { \prime } \subsetneq$ x is selected while both are in the pool. Thus, we can ensure that the top of the heap is an element $x \in M I N ( P )$ which implies that $x \in M I N ( P R E _ { \forall } ( B _ { i } ) \setminus U P ( B _ { i } ) )$ if $x \not \in U P ( B _ { i } )$ Checking if x $\notin U P ( B _ { i } )$ can be done efficiently as discussed above. The heap-pop operator is done in $O ( l o g ( n ) )$ time (instead of searching and checking minimality against each element in the pool as in naive set). Insertion is also done in $O ( l o g ( n ) )$ of time.

For constructing candidates for the pool efficiently, instead of iterating over all basis elements for each effect of each action, we keep a per-effect live list of states. A state $b \in B$ belong to the live list of an effect e iff $b \cap a d d _ { e } \neq \emptyset$ and $b \cap d e l _ { e } = \emptyset$

Initialisation. We initialise $B$ and the live list of every effect to be the emptyset, and put $G$ into the pool of candidates. This allows $B = B _ { 0 }$ at the end of iteration 0 to match with Algorithm III.

Each Iteration At each iteration i starting with $i = 0 ,$ , the min-key state $x _ { i }$ in the pool is selected to extend the basis (note that for $i = 0$ the only state in the pool is G).

In the strong variation, we order the pool by the potential of a state, $\begin{array} { r } { \dot { \mathrm { p o t } } ( x ) = \sum _ { p \in x } \operatorname* { m a x } ( 0 , \dot { h _ { s } } ( p ) \dot { - } h _ { b } ( \dot { p ) } ) } \end{array}$ (i.e. $\ker ( x ) = \operatorname { p o t } ( x )$ until one of the steering mechanisms of Runtime steering (below) changes it). Intuitively, the potential allows states whose atoms are cheap from $s ^ { \check { I } }$ and expensive from the goal to be selected first, so that the basis grows from the goal toward the $s ^ { I }$ side. We tie-break by selecting states with fewer atoms outside $s ^ { I }$ , and if it is still a tie, we select the smaller state.

Selection stage. For the selection stage, pop the minimum key x in the pool and check if $x \notin \hat { U P } ( \hat { B } )$ already and x does not violate all mutex groups. If both are true, we select x and move to the basis extension stage, otherwise repeatedly pop the pool until there is such x or until the pool is empty (which we terminate since there is no candidate as described in Algorithm III).

It is easy to see that the stated selection mechanism is ⊊-monotone. Since if $x ^ { \prime } \subsetneq x$ then $\ker ( x ) \geq \ker ( x ^ { \prime } )$ . For the tie-braking case, since we select states with fewer atoms outside $s ^ { I }$ and then, if it is still a tie, select the smaller state, x will never be selected over $x ^ { \prime } .$ . Thus, we ensure that $x \in$ $M I N ( P R E _ { \forall } ( B _ { i } ) ) \setminus U P ( B _ { i } )$

Basis extension stage. For the selected state x, we add $x$ to B by using the standard set-trie add operator then mark all the subsumed states by x in B as “removed”.

Now we update the live list of every effect e by adding x to it if $x \cap a d d _ { e } \neq \emptyset$ and $x \cap d e l _ { e } = \emptyset$ and removing any state $x ^ { \prime }$ from the live list if $x \subsetneq x ^ { \prime }$

Finally, we update the pool by adding new candidates to it. Each candidate is a state constructed as follows: pick an action a with effects $e _ { 1 } , \ldots , e _ { K } $ , for each effect $\mathit { e } _ { k } ,$ pick a state $b _ { k }$ in its live list. If the live list is non-empty for all the effects and at least one chosen $b _ { i } = x$ , then a new candidate is $x _ { n } = p r e _ { a } \cup \bigcup _ { k = 1 } ^ { K } ( b _ { k } \setminus a d d _ { k } )$ . We enumerate over all actions and all such combination of states in the live list.<sup>4</sup>We add all candidates to the pool.

If there is a state x in the basis such that $x \subseteq s ^ { I }$ , then we move directly to the finalisation step. Otherwise, if the pool is empty and there is no such x, we stop and run the weakwinning version of Algorithm III. The implementation of the weak-winning version is the same except in two places:

• We replace the equation in the candidate constructing step to $x _ { n } = p r e _ { a } \cup \left( x \backslash a d d _ { e } \right)$ . Notice that the constraint of at least one $b _ { k }$ must be x in this case exists directly in the formula.

• We change the heuristic function to key $( s ) = | s |$ and tiebreak the same way. It is easy to see that this selection mechanism is still $\subsetneq - { \mathrm { m o n o t o n e } }$

There are more micro optimisation for this core implementation that can be found in our source code. For the strong-winning variation, we implement an additional mechanism for the pool/selection heuristic.

Runtime steering. We add two adaptive mechanisms that change the pool order while the strong-winning variation is running.<sup>5</sup>

Drift penalties: the strong-winning variation should move the potential toward the $s ^ { I }$ side, we achieve it by introducing this mechanism: During the candidates construction from x process, an action a may put several candidates into the pool. Denote ch(a) the set of candidates constructed by a. If $\operatorname { c h } ( a ) > 1$ , let $\begin{array} { r } { \dot { \delta _ { a } } = \frac { 1 } { | \mathrm { c h } ( a ) | } \sum _ { c \in \mathrm { c h } ( a ) } \mathrm { p o t } ( c ) - \mathrm { p o t } ( x ) } \end{array}$ be the mean potential change. Let $Q$ be the sliding median of $\left| \mathrm { p o t } ( x _ { i + 1 } ) - \mathrm { p o t } ( x _ { i } ) \right.$ | over the last 1024 iterations, bounded below by 1. Action a drifts when $| \delta _ { a } | \leq Q / 2 .$ , that is, when the action moves the potential by less than half a typical step in either direction. Each drift event adds one penalty unit to pen(p), for every precondition atom p of $^ { a , }$ up to four units per atom. Every 1024 iterations, if any drift event occurred since the last update, we recompute the pool keys as ke $\begin{array} { r } {  { \boldsymbol { \mathbf { \rho } } } ^ { \prime } (  { \boldsymbol { { x } } } ) =  { \operatorname { p o t } } (  { \boldsymbol { { x } } } ) + \sum _ { p \in  { \boldsymbol { { x } } } }  { \operatorname { p e n } } ( p ) } \end{array}$ . The intuition is that penalties add a small synthetic gradient on the flat parts of the potential function, so the selection moves away from actions that keep producing flat candidates.

Depth window: the strong-winning variation should prefer states that gets contructed in fewer steps from $B _ { 0 } { : }$ at iterations $F , 2 F , \bar { 4 } F , . . . \ ( F = 1 0 2 4 )$ we add depth term in the key for exactly $\frac { F } { 2 }$ iterations and then drops it. In particular it add depth(x) to key(x) where depth(x) is the length of the longest states-chain that derives x from the $G , G \in B _ { 0 }$ has depth 0 and a state is one deeper than the deepest state used to derive it.

The potential and the penalty sum are sums of nonnegative per-atom terms, so neither of them can order a subset behind a superset. The depth term can, so while a window is engaged the selection discards any state x at the top of the pool that strictly contains another state in the pool.

## 5.3 Finalisation

There are two ways the algorithm can terminate:

If pool runs out of candidate then we get both the bases for the strong-winning and weak-winning region excluding some unreachable states. Then we check if there is a state $x \subseteq s ^ { I }$ in the weak-winning basis. Note that there is certainly no such state in the strong-winning otherwise the algorithm halted before getting to the weak-winning variation. If there is such x, we can construct a best-effort policy by following the construction as described in Algorithm III and report that the instance is weak-winning. Otherwise it reports that the instance is losing.

```latex
Pseudocode 1 Core algorithm
Require: FOND task $( S , s ^ { I } , A , G )$ and the potential pot;
the pool is ordered by key (ties: fewer atoms outside $s ^ { I } ,$
then the smaller state).
Ensure: The verdict; a policy when the instance is strong
winning/weak-wining.
1: $B  \varnothing ;$ pool $ \{ \bar { G } \}$ ; live lists empty;
2: repeat
3: pop the min-key element x of the pool;
4: if x is covered by B or violates a mutex group then
5: skip x and try the next pooled element;
6: end if
7: if x is not exists because the pool is empty then
8: break
9: end if
10: add x to $B ;$ mark the states of B that are superset of
x as removed; update the live lists;
11: for each candidate $\begin{array} { r } { y = p r e _ { a } \cup \bigcup _ { k } ( b _ { k } \setminus a d d _ { k } ) } \end{array}$ gener
ated from x do
12: if $\cdot _ { y } \subseteq s ^ { I }$ then
13: halt: report strong-winning and return the wit
ness chain;
14: else
15: push y into the pool with key $\operatorname { p o t } ( y ) ;$
16: end if
17: end for
18: until the pool is empty
19: rerun the loop from the seed (fresh basis) with key(x) =
|x| and $y = \bar { p } r e _ { \underline { { a } } } \cup ( x \setminus a d d _ { e } ) ;$ (weak variation)
20: if the weak basis contains $y \subseteq s ^ { I }$ then report weak
winning and construct a best-effort policy else report
losing.
```

If there is a state x in the strong-winning variation’s basis such that $x \subseteq s ^ { I }$ , the algorithm reports that the instance is strong-winning, together with a strong-winning plan.

## 6 Experiments

We have compared our planner with some of the best existing strong and best-effort FOND planners: PR2 (Muise, McIlraith, and Beck 2024) and FOND-SAT (Geffner and Geffner 2018) for strong planning, and BeSyftP (De Giacomo, Parretti, and Zhu 2023) for best-effort planning. The four planners were run on an AMD Ryzen 7 8845HS@4.5GHz with per-instance time and memory limits of 1800s and 16GiB memory. PR2 runs in its default strong mode (width-1..5 loop). Its design halves each allocatted budget between the core search and the repair rounds that finalize the plan and extract the policy. All of the planners are the authors’ shipped public versions, unmodified. We use domains and instances available from previous publications: For strong planning, the exact 8 domains are used in FOND-SAT benchmarking (Geffner and Geffner 2018); for best-effort planning, the benchmarking set is taken directly from the public GitHub repository of BeSyftP (De Giacomo, Parretti, and Zhu 2023).

## 6.1 Results for strong-winning

The benchmark covers eight domains with total of 222 instances: doors (15), miner (51), elevators (15), tireworld (15), tireworld-spiky (11), triangle-tireworld (40), islands (60), and zenotravel (15). Zenotravel’s empty outcomes and tireworld’s “spare change fails” outcome are stripped for every planner because: zenotravel is not strong-winning at all, and only 3 of the 15 tireworld instances are strong-winning if we keep the original domain. After stripped, all zenotravel and twelve tireworld instances are strong-winning. This also appears to be the approach taken in the FOND-SAT paper (Geffner and Geffner 2018). Table 1 is therefore restricted to the strong-winning instances of each domain (elevators 9 of 15, tireworld 12 of 15); the remaining nine (elevators p08/p10–p13/p15, tireworld p01/p09/p15) are provably not strong-winning, and all are weak-winning.

Overall, our planner decides all the instances within a 1800 s cap, the slowest in 12.2 s, outperforming the strong FOND planners PR2 and FOND-SAT in coverage. PR2 is faster on the instances it solves in under a second, where our fixed precomputation dominates the core algorithm’s runtime.

Figure 1 shows the solved strong-winning instances versus time curves. PR2’s curve rises first (139 vs. 92 solved at 0.2s); the FONDANT overtakes just past 0.4 s and stops 213 (its last solved instances, zenotravel/p15, at 12.2s) while PR2 stops at 204 with the remaining instances timeout; FOND-SAT (dash-dot) stops at 162, its time spread over 0.13–1311.8s.

Table 1 summarizes the performance on strong-winning instances. Our planner solved all 213 strong-winning instances (the slowest is zenotravel/p15 at 12.2 s) instances, PR2 204 of the 213 in scope, and FOND-SAT 162. PR2’s rows without a winning answer are nine timeouts (doors p15; triangle p33–p40).

Analysis of the runtime on small and medium instances. Where PR2 is fastest (islands 0.11–0.22 s, miner 0.13– 0.35 s, spiky 0.12–0.14 s, zenotravel 0.14–5.0 s) our times are within a small factor (0.17–0.34, 0.23–1.4, 0.18–0.21,

Table 1: Coverage table for strong winning instances: FON-DANT (ours) vs. PR2 and FOND-SAT within the 1800s and 16GB cap; cells are solved/total within scope, bold marks the highest number of solved instances per row.
<table><tr><td>domain</td><td>FONDANT (ours)</td><td>PR2</td><td>FOND-SAT</td></tr><tr><td>doors</td><td>15/15</td><td>14/15</td><td>15/15</td></tr><tr><td>miner</td><td>51/51</td><td>51/51</td><td>51/51</td></tr><tr><td>elev-strong</td><td>9/9</td><td>9/9</td><td>7/9</td></tr><tr><td>tire-strong</td><td>12/12</td><td>12/12</td><td>12/12</td></tr><tr><td>tire-spiky</td><td>11/11</td><td>11/11</td><td>10/11</td></tr><tr><td>tri-tire</td><td>40/40</td><td>32/40</td><td>2/40</td></tr><tr><td>islands</td><td>60/60</td><td>60/60</td><td>60/60</td></tr><tr><td>zeno</td><td>15/15</td><td>15/15</td><td>5/15</td></tr><tr><td>TOTAL (213)</td><td>213/213</td><td>204/213</td><td>162/213</td></tr></table>

Table 2: Wall times on the 204 strong-winning instances both solve, per domain: mean and slowest wall (s) over the same instance set for each solver. PR2’s no-answer rows are excluded (doors p15; triangle p33–p40). Bold = faster (lower).
<table><tr><td>domain</td><td>instances</td><td colspan="2">FONDANT</td><td colspan="2">PR2</td></tr><tr><td></td><td></td><td>avg (s)</td><td>max (s)</td><td>avg (s)</td><td>max (s)</td></tr><tr><td>doors</td><td>14 51</td><td>0.19</td><td>0.20 1.35</td><td>52.3 0.20</td><td>560.3 0.35</td></tr><tr><td>miner elev-strong</td><td>9</td><td>0.41 0.20</td><td></td><td></td><td>0.13</td></tr><tr><td>tire-strong</td><td>12</td><td>0.18</td><td>0.20 0.20</td><td>0.12 0.12</td><td>0.14</td></tr><tr><td>tire-spiky</td><td>11</td><td>0.19</td><td>0.21</td><td>0.13</td><td>0.14</td></tr><tr><td>tri-tire</td><td>32</td><td>0.80</td><td>2.97</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>129.6</td><td>802.7</td></tr><tr><td>islands</td><td>60</td><td>0.21</td><td>0.34</td><td>0.14</td><td>0.22</td></tr><tr><td>zeno TOTAL</td><td>15 204</td><td>2.32 0.50</td><td>12.2 12.2</td><td>1.05 24.1</td><td>5.01 802.7</td></tr></table>

![](images/f20d0dd7895a7b1a38ab85ae22b31a109a40b82fbecf465e1fe9b1c583db345a.jpg)  
Figure 1: Solved instances versus wall time (log scale), FONDANT (ours, solid), PR2 (dashed), and FOND-SAT (dash-dot) over 213 strong-winning instances under the 1800 s box, the vertical line marks the cap.

0.22–12.2 s).

The time gap is because of per-instance precomputation (especially mutex groups computation); the core is millisecond-scale on these rows.

Our tool is also complete, and provides a certificate of unsolvability. To illustrate that our tool is complete, we compared it to PR2 and FOND-SAT on nine instances (elevators p08/p10–p13/p15; tireworld p01/p09/p15) for which there is no strong-winning solution. Our strong-winning phase of the core algorithm terminates within 0.17–3.8s (elevators 0.25–3.8s; tireworld 0.17–0.19s), and provide a certificate (the strong-winning region) that the instance is not strong-winning. PR2 either expires a time limit mid-search (elevators p11/p12/p15; tireworld p15) or drains its width loop and proves that there is no strong plan of width ≤ 5 (which is not a valid certifcate to indicate that the instace is not strong-winning). FOND-SAT times out on all nine.

Demonstration of scaling beyond the benchmark set. Some of the domains are too small to give a good runtime comparasion between FONDANT and PR2. We grew each family to a larger size while preserving the structure and statistics of the originals in the benchmark set: zenotravel to 36 cities, miner to 224 grid rows, doors to a 24-location corridor, tireworld-spiky to g6, elevators to P = 24, islands to 162 locations, tireworld to n = 51.

Both planners were re-run on all of them (Figure 2). On doors FONDANT stays flat (0.19–0.21 s, terminated within 34–46 iterations) while PR2 climbs to 1,164s at p15; on triangle-tireworld FONDANT answers all forty in under 7.3 s while PR2’s budget expires from p33. Tireworld-spiky has the same results as in the benchmark set — both sides finished in a fraction of a second. On miner PR2 leads up to x64 but then grows steeply over the three largest points, where our rows stay in the tens of seconds. The dip at miner x112 is the mutex precomputation switching: below the cap it spends its whole budget before giving up, above it the memory guard aborts at once, so x112 is cheap and x160/x224 resume the rise. Zenotravel when scaled past roughly twenty cities both planners need tens to hundreds of seconds, and at 36 cities PR2 exhausts widths 1–5 while we still provide a strong plan. For the three remaining families, both stay relative fast as the domain size increases.

## 6.2 Best-effort results

The best-effort benchmark runs on the benchmark families published with BeSyftP (De Giacomo, Parretti, and Zhu 2023): the crafted BestEffortTests tireworld suite, Triangle-TireWorld p1–p10, Elevators, and RectangleTireworld p1– p15 (47 instances).

Table 3: Wall times on the 34 best-effort instances both planners answer, per family (mean/slowest, s). The twelve instances only FONDANT answers (elevators p09/p11–p15, rectangle p9–p14) are not included this table. Bold means faster (lower) per statistic.
<table><tr><td rowspan="2">domain</td><td rowspan="2"></td><td colspan="2">#instances FONDANT (ours)</td><td colspan="2">BeSyftP</td></tr><tr><td>avg (s)</td><td>max (s)</td><td>avg (s)</td><td>max (s)</td></tr><tr><td>BestEffortTests</td><td>7</td><td>0.24</td><td>0.30</td><td>0.06</td><td>0.06</td></tr><tr><td>TriangleTireWorld</td><td>10</td><td>0.18</td><td>0.21</td><td>60.1</td><td>315.6</td></tr><tr><td>Elevators</td><td>9</td><td>0.30</td><td>0.71</td><td>178.3</td><td>1001.7</td></tr><tr><td>RectangleTireworld</td><td>8</td><td>0.51</td><td>0.79</td><td>43.2</td><td>236.8</td></tr><tr><td>TOTAL</td><td>34</td><td>0.30</td><td>0.79</td><td>75.0</td><td>1001.7</td></tr></table>

Side by side with BeSyftP on its 47 published instances (Table 4), FONDANT answers 46, BeSyftP 34, its 13 misses are timed out instances (elevators p09/p11–p15, rectangle p9–p15); on every instance that both answer, the classifications agree. Rectangle p15 is the one neither answers: our Fast-Downward translation exceeds the memory cap and the synthesizer times out. On the 34 instances that both planners answer (Table 3), our walls are 0.15–0.79s everywhere and BeSyftP’s span 0.05–1,001.7s: BeSyftP is faster on the seven crafted tireworld variants (0.06 s) and ours is faster on other domains by two to three orders of magnitude (triangle 0.18 vs. 60.1 s; elevators 0.30 vs. 178.3 s; rectangle 0.51 vs. 43.2 s means). The instances BeSyftP leaves unanswered are all decided by FONDANT in 0.19–44.6 s (six elevators rows; rectangle p9–p14 at 2.6–44.6 s).

![](images/2786aa6501b3ecdd2e2db495c0eda26ad15a822425528d1baee3906948abe6a0.jpg)  
Figure 2: Scaling: wall time (s, log scale) against size, ours vs PR2; dotted line = 1,800 s cap. Filled = strong plan, cross = no strong-plan answer (timeouts/expiries).

Table 4: Best-effort three-way classification (strong / weak / losing) of FONDANT side by side with the synthesizer BeSyftP (De Giacomo, Parretti, and Zhu 2023) on its published families; cells count instances per verdict class $( ^ { \mathfrak { s } \mathfrak { c } } \mathrm { a n s } ^ { \mathfrak { s } \mathfrak { z } } = \mathrm { a n s w e r e d } )$ . Bold = our totals.
<table><tr><td rowspan="2">domain</td><td rowspan="2">#instances</td><td colspan="4">FONDANT (ours)</td><td colspan="4">BeSyftP</td></tr><tr><td>strong</td><td>coop</td><td>losing</td><td>ans</td><td>strong</td><td>coop</td><td>losing</td><td>ans</td></tr><tr><td>BestEffortTests</td><td>7</td><td>3</td><td>2</td><td>2</td><td>7</td><td>3</td><td>2</td><td>2</td><td>7</td></tr><tr><td>TriangleTireWorld (p1-p10)</td><td>10</td><td>10</td><td>0</td><td>0</td><td>10</td><td>10</td><td>0</td><td>0</td><td>10</td></tr><tr><td>Elevators</td><td>15</td><td>9</td><td>6</td><td>0</td><td>15</td><td>7</td><td>2</td><td>0</td><td>9‡</td></tr><tr><td>RectangleTireworld</td><td>15</td><td>14</td><td>0</td><td>0</td><td>14†</td><td>8</td><td>0</td><td>0</td><td>8‡</td></tr><tr><td>TOTAL</td><td>47</td><td>36</td><td>8</td><td>2</td><td>46</td><td>28</td><td>4</td><td>2</td><td>34</td></tr></table>

<sup>†</sup> Rectangle p15: neither tool verdicts it: our PDDL front end’s Fast Downward grounding exceeds the memory cap, the synthesizer BeSyftP times out. <sup>‡</sup> synthesis caps within the box.

## 7 Conclusion and future work

We described an antichain-based algorithm for best-effort and strong FOND planning, that can also classify instances as strong-winning or weak-winning or neither. Compared with PR2, FONDANT does at least as well on some domains and has better scaling on large instances in some domains. Future work will focus on proving that for uniformly strongwinning policies, there is an instructions-set that induces a uniformly strong-winning policy whose size is as small as the smallest uniformly strong-winning acyclic finite-state strategy. We will also investigate whether there are better selection mechanisms, and we will extend the best-effort and strong-winning test domains/instances.

## References

Aminof, B.; De Giacomo, G.; and Rubin, S. 2021. Best-Effort Synthesis: Doing Your Best Is Not Harder Than Giving Up. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI), 1766–1772.

Berwanger, D. 2007. Admissibility in Infinite Games. In Proceedings of the 24th Annual Symposium on Theoretical Aspects ofComputer Science (STACS), 188–199. Springer.

Bonet, B.; and Geffner, H. 2001. Planning as heuristic search. Artificial Intelligence, 129(1–2): 5–33.

Brenguier, R.; Raskin, J.; and Sassolas, M. 2014. The complexity of admissibility in Omega-regular games. In Henzinger, T. A.; and Miller, D., eds., Joint Meeting of the Twenty-Third EACSL Annual Conference on Computer Science Logic (CSL) and the Twenty-Ninth Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), CSL-LICS 2014, Vienna, Austria, July 14 - 18, 2014, 23:1–23:10. ACM.

Cimatti, A.; Pistore, M.; Roveri, M.; and Traverso, P. 2003. Weak, Strong, and Strong Cyclic Planning via Symbolic Model Checking. Artificial Intelligence, 147(1–2): 35–84.

De Giacomo, G.; Parretti, G.; and Zhu, S. 2023. LTLf Best-Effort Synthesis in Nondeterministic Planning Domains. In Proceedings of the 26th European Conference on Artificial Intelligence (ECAI), 533–540. IOS Press.

Faella, M. 2009. Admissible Strategies in Infinite Games over Graphs. In Kralovic, R.; and Niwinski, D., eds.,´ Mathematical Foundations of Computer Science 2009, 34th International Symposium, MFCS 2009, Novy Smokovec, High Tatras, Slovakia, August 24-28, 2009. Proceedings, volume 5734 of Lecture Notes in Computer Science, 307–318. Springer.

Fijalkow, e. a., ed. 2026. Games on Graphs: From Logic and Automata to Algorithms. Cambridge University Press.

Filiot, E.; Jin, N.; and Raskin, J. 2009. An Antichain Algorithm for LTL Realizability. In Bouajjani, A.; and Maler, O., eds., Computer Aided Verification, 21st International Conference, CAV2009, Grenoble, France, June 26 - July 2, 2009. Proceedings, volume 5643 of Lecture Notes in Computer Science, 263–277. Springer.

Geffner, T.; and Geffner, H. 2018. Compact Policies for Fully Observable Non-Deterministic Planning as SAT. In Proceedings ofthe Twenty-Eighth International Conference on Automated Planning and Scheduling (ICAPS), 88–96.

Haslum, P.; and Geffner, H. 2000. Admissible Heuristics for Optimal Planning. In Proceedings ofthe Fifth International Conference on Artificial Intelligence Planning and Scheduling (AIPS), 70–79.

Helmert, M. 2006. The Fast Downward Planning System. Journal ofArtificial Intelligence Research, 26: 191–246.

Muise, C.; McIlraith, S. A.; and Beck, J. C. 2012. Improved Non-Deterministic Planning by Exploiting State Relevance. In Proceedings of the Twenty-Second International Conference on Automated Planning and Scheduling (ICAPS), 172– 180.

Muise, C.; McIlraith, S. A.; and Beck, J. C. 2024. PRP Rebooted: Advancing the State of the Art in FOND Planning. In Proceedings ofthe Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI), 20212–20221.

Muvvala, K.; Amorese, P.; and Lahijanian, M. 2022. Let’s Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation. In 2022 International Conference on Robotics and Automation, ICRA 2022, Philadelphia, PA, USA, May 23-27, 2022, 4340–4346. IEEE.

Ram´ırez, M.; and Sardina, S. 2014. Directed Fixed-˜ Point Regression-Based Planning for Non-Deterministic Domains. In Proceedings of the Twenty-Fourth International Conference on Automated Planning and Scheduling (ICAPS), 235–243.

Rintanen, J. 2004. Complexity of Planning with Partial Observability. In ICAPS 2024.

Wulf, M. D.; Doyen, L.; Henzinger, T. A.; and Raskin, J. 2006. Antichains: A New Algorithm for Checking Universality of Finite Automata. In Ball, T.; and Jones, R. B., eds., Computer Aided Verification, 18th International Conference, CAV 2006, Seattle, WA, USA, August 17-20, 2006, Proceedings, volume 4144 of Lecture Notes in Computer Science, 17–30. Springer.