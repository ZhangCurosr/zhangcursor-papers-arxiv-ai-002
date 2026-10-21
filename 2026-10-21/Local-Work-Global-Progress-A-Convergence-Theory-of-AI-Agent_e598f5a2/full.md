# Local Work, Global Progress: A Convergence Theory of AI Agent Collaboration

Ivan Ilin

KAUST

ivan.ilin@kaust.edu.sa

## Abstract

Teams of large language model (LLM) agents are increasingly applied to complex tasks, but their collaboration protocols are mostly designed empirically, without convergence guarantees. We formalize collaborative problem solving as block optimization: each agent improves one part of a shared solution, guided by its local objective plus a correction for how its changes afect the other parts. A coordinator checks every proposal against the global objective and merges proposals that do not interfere with each other. We prove linear convergence up to an error floor that vanishes when the corrections are exact. Interfering proposals can also be merged once their combination has been checked; the rate then depends on how well the proposals actually combine, not on worst-case interactions. With GPT-OSS-120B agents on robot routing, a three-agent team finds collision-free plans in every run, better than those of a single agent that controls all robots. On a larger routing task, a six-agent team completes its search about twice as fast as the single agent (in median). Merging the best checked combination of proposals, rather than only proposals that do not interfere, cuts the team’s search time by about 30%. With constants measured on the runs, our convergence bound holds in every round and follows the observed progress. We also test the theory on query planning in SQLite and on a synthetic task.

## 1 Introduction

Peter Richtárik

Large language model (LLM) agents are increasingly asked to produce artifacts that are too large for a single request, such as software with many modules or plans for many robots. Human organizations handle such work by division of labor, and dividing work among agents promises two benefits. First, each request becomes smaller and easier, so a task that overwhelms one agent can become solvable. Second, the parts can be searched in parallel, so a solution arrives sooner. We do not aim to reduce monetary cost: like a construction crew compared with a single builder, a team spends more in total but completes work that one agent would finish slowly or not at all.

Not every task benefits from a team, and not every team needs coordination. A simple task, like changing a light bulb, is best done by one worker. If the parts do not interact, for example when one agent waters a garden while another goes shopping, agents can simply work in parallel. The interesting and common case lies in between: large tasks whose parts are coupled, such as the modules of a program, the components of an aircraft, or the routes of robots that share a floor. Each

KAUST

peter.richtarik@kaust.edu.sa

part is manageable alone, but a change to one part can change what is best for the others.

Coupling creates a coordination problem: improving each part in isolation need not improve the whole. A module can pass its own tests while violating another module’s interface; a robot can reach its goal along a route that collides with a robot controlled by another agent. A global check can reject a harmful proposal, but rejection does not tell the agent which alternative to try next: it needs information that changes how it ranks its own candidates.

We study this problem through finite block optimization. An agent is a local search procedure, for example an LLM, that owns one block of a shared solution and searches it while the other blocks stay frozen. A correction added to its local objective tells the agent how its changes afect the other blocks. Agents search their corrected objectives in parallel. A verifier then scores each proposal against the current shared solution, and a coloring of the dependency graph selects proposals that do not interfere (Figure 1). The correction steers the proposals, while verification and coloring ensure that the shared solution never gets worse.

We test the protocol with GPT-OSS-120B agents on robot routing, where robots of diferent agents can collide. A team of three agents finds collision-free plans in every run, better than those of a single agent that controls all robots at once. The same team without corrections rarely avoids collisions. On a larger task, a team of six agents also searches about twice as fast as the single agent in median. The coordinator can even merge interfering proposals once the verifier confirms that together they lower the loss more than the non-interfering ones alone. This rule, which our theory covers, cuts the team’s search time by about 30% and lets it reach low losses sooner than single agents searching in parallel. With constants measured on the runs, the convergence bound holds in every round, so theory and practice agree. Two smaller studies, on query planning in SQLite and on a synthetic objective, explore the theory and its limits further.

Contributions.

(1) We formalize collaborative problem solving as block optimization with corrected local objectives, and introduce a protocol of parallel private search, exact verification and compatible merging (Section 3).

(2) We prove linear convergence with an explicit error floor (Theorem 2). The rate separates task geometry, local search quality and merge compatibility.

(3) We show that the coordinator can accept the best verified combination of the agents’ proposals, keeping the compatible work of interfering blocks, with the same guarantees and a rate set by how well the proposals actually combine (Section 5).

![](images/17515dadaab092a39d8c8792f9041d9862969772d4c6955a63fc606356ece885.jpg)  
Figure 1: One epoch of Algorithm 1 on a constructed example with illustrative gains. Staircases show each agent’s best corrected score so far, which need not reach the minimum $\widehat { f _ { b } } ^ { * }$ of the corrected objective. Each proposal is verified alone, and the coordinator accepts the color with the largest total gain: blocks 2 and 3 (3 + 2 = 5) rather than the single largest gain (4).

(4) We show how agents can learn reusable corrections from global evaluations, with a joint convergence rate for the solution and the correction (Appendix A.4), and account for calibration, local work and synchronization costs (Sections 6 and 7).

(5) We evaluate the protocol with teams of three and six LLM agents on robot routing and relate the observed convergence to the theorem; a database study shows when acquiring correction information pays of, and a synthetic study confirms the error floor (Section 8).

The appendices contain further theory and examples (Appendix A), extended related work (Appendix B), and complete experimental protocols and outcomes.

The implementation and code for reproducing our experiments are available at https://github.com/vectozavr/ local-work-global-progress.

## 1.1 Notation

All key notation used in this paper is summarized in tabular form in the Notation section before the appendices; see Table 2.

## 2 Related work

Block optimization and distributed coordination. Block coordinate descent and its convergence theory are classical (Bertsekas and Tsitsiklis, 1989; Tseng, 2001; Nesterov, 2012), as are its parallel and inexact variants (Richtárik and Takáč, 2016; Tappenden et al., 2016). When parallel coordinate updates interfere through shared terms, prior analyses bound the interference (Bradley et al., 2011; Richtárik and Takáč, 2016) or group weakly coupled coordinates (Scherrer et al., 2012). We instead verify every proposal and then either merge only blocks of one color, whose gains add exactly, or check the merged combination as a whole. Distributed constraint optimization assigns coupled variables to agents: MGM coordinates improving moves, DSA randomizes local updates, and DPOP propagates conditional utility tables (Maheswaran et al., 2004; Zhang et al., 2005; Petcu and Faltings, 2005). Our addition is an explicit interface for correcting candidate values and a bound that converts the correction error into lost global progress.

Persistent corrections and identification. ProxSkip and SCAFFOLD correct local work through control variates (Mishchenko et al., 2022; Karimireddy et al., 2020). Our corrections play the analogous role for discrete search, adjust ing candidate scores of disjoint blocks rather than gradients; whether they also admit ProxSkip-type acceleration remains open. Our update for learning corrections (Section 6) is a randomized Kaczmarz, or sketch-and-project, step (Strohmer and Vershynin, 2009; Gower and Richtárik, 2015), and additive Bayesian optimization and linear-bandit identification also exploit known structure and informative measurements (Kandasamy et al., 2015; Soare et al., 2014).

LLMs as evaluator-guided optimizers. LLMs can serve as search operators when paired with an automatic evaluator. OPRO proposes new solutions from a history of scored ones, and FunSearch and AlphaEvolve evolve programs under automated evaluation (Yang et al., 2024; Romera-Paredes et al., 2024; Novikov et al., 2025). Our agents are such evaluatorguided searchers, each restricted to one block. The theory asks what feedback the evaluator must provide so that parallel block searches add up to global progress.

LLM collaboration and multi-robot planning. MetaGPT and ChatDev coordinate software-development artifacts, and AutoGen composes conversing agents (Hong et al., 2024; Qian et al., 2024; Wu et al., 2024). GPTSwarm optimizes agent graphs, and AgentPrune removes redundant communication (Zhuge et al., 2024; Zhang et al., 2025b). Reflexion retains feedback-derived text across trials, and EvoMAC adapts collaboration using verification feedback (Shinn et al., 2023; Hu et al., 2025). Decomposed prompting splits complex tasks into simpler subtasks (Khot et al., 2023). In robotics, one LLM agent per robot can negotiate a joint plan through dialogue and re-plan from motion-planner feedback such as collisions (Mandi et al., 2024), and LLMs can decompose instructions and allocate the subtasks among robots (Kannan et al., 2024). Centralized, decentralized and hybrid LLM planners have also been compared as teams grow (Chen et al., 2024a). LLMs solve multi-agent path finding on empty maps but still fail on standard maps with obstacles (Chen et al., 2025), which is the dificulty our routing study targets. These works show empirically that decomposition and feedback help; our analysis states which properties of the agents and of the feedback yield convergence. Appendix B gives a broader comparison.

## 3 Model and protocol

This section turns the coordination problem from the introduction into an optimization model. We first define the shared task and the corrected objective that each agent searches (Section 3.1), then the dependency graph that determines which changes can be combined safely (Section 3.2), and finally the protocol that combines them (Section 3.3, Algorithm 1).

## 3.1 Task and corrected objectives

The shared solution is $x = ( x _ { 1 } , \dots , x _ { m } )$ : agent � owns block $x _ { b } ,$ which takes values in a nonempty finite set $\mathcal { X } _ { b }$ . The team solves

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } F ( x ) , \qquad \mathcal { X } = \mathcal { X } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { X } _ { m } ,\tag{1}
$$

for a loss $F : \mathcal { X }  \mathbb { R }$ . An exact verifier returns $F ( x )$ for any solution, although its formula need not be known to the agents. Each agent can also cheaply evaluate a local objective $L _ { b } : \mathcal { X } _ { b }  \mathbb { R }$ of its candidate blocks $u \in \mathcal { X } _ { b } .$ which may ignore how � interacts with the other blocks.

The optimal loss is $F ^ { * }$ , and the conditional change $f _ { b } ( u ; x )$ is how much the global loss changes if agent � alone replaces its block by �:

$$
F ^ { * } : = \operatorname* { m i n } _ { x \in \mathcal { X } } F ( x ) , \qquad f _ { b } ( u ; x ) : = F ( x _ { - b } , u ) - F ( x ) .\tag{2}
$$

These quantities serve the analysis; agents need not know them. The exact correction $R _ { b }$ is the part of the conditional change that the local objective misses, and an agent searches the corrected objective $\widehat { f _ { b } }$ built from an available approximation $\widehat { R } _ { b }$ :

$$
R _ { b } ( u ; x ) : = f _ { b } ( u ; x ) - [ L _ { b } ( u ) - L _ { b } ( x _ { b } ) ] ,\tag{3}
$$

$$
\widehat { f } _ { b } ( u ; x ) : = L _ { b } ( u ) - L _ { b } ( x _ { b } ) + \widehat { R } _ { b } ( u ; x ) .\tag{4}
$$

The four functions $f _ { b } , R _ { b } , \widehat { R } _ { b } , \widehat { f } _ { b } : \mathcal { X } _ { b } \times \mathcal { X } $ R take a candidate block and the current solution. If $\widehat { R } _ { b } = R _ { b }$ , then ${ \widehat { f _ { b } } } = f _ { b }$ We require anchoring, $\widehat { R } _ { b } ( x _ { b } ; x ) = 0$ , so that keeping the current block always scores zero. A useful correction $\widehat { R } _ { b } ( u ; x )$ must vary with the candidate �, because adding the same constant to every candidate cannot change their ranking; constructing and communicating it costs work, which Section 7 accounts for.

Neither $L _ { b }$ nor $\widehat { R } _ { b }$ has to be given to the agent as a formula. An LLM agent receives its task and the information behind the correction in its prompt, for example which collisions the verifier has found, and proposes candidates that it judges better. The functions $L _ { b }$ and $\widehat { R } _ { b }$ describe the preferences that this information conveys; they let us state what the search must achieve (Assumption 2) without modeling how the agent reasons.

Example. In the routing task of Section $8 . 1 , \ x _ { b }$ lists the moves of the three robots that agent � controls, and $L _ { b }$ adds their delivery costs and the collisions among them. The exact correction �<sub>�</sub> is the change in the penalty for collisions with the other agents’ robots. The implemented correction $\widehat { R } _ { b }$ computes this change only for the robot pairs whose collisions the verifier has already reported, using the other robots’ current routes.

## 3.2 Dependency graph and coloring

We write the loss as a sum of � real-valued terms, called factors:

$$
F ( x ) = \sum _ { a = 1 } ^ { M } \ell _ { a } ( x _ { N ( a ) } ) ,\tag{5}
$$

where factor $\ell _ { a }$ depends only on the blocks in a set $N ( a ) \subseteq$ $\{ 1 , \ldots , m \}$ , and $x _ { N ( a ) } : = ( x _ { b } ) _ { b \in N ( a ) }$ denotes these blocks. Every loss can be written in this form, trivially as the single factor $\ell _ { 1 } = F$ with $N ( 1 ) = \{ 1 , \dots , m \}$ . Only the sets $N ( a )$ must be known; the factors themselves, like the formula of $F ,$ may stay hidden in the verifier.

For example, the loss $F ( x ) = ( x _ { 1 } - x _ { 2 } ) ^ { 2 } + ( x _ { 1 } - x _ { 3 } ) ^ { 2 }$ of three blocks has two factors, $\ell _ { 1 } ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } - x _ { 2 } ) ^ { 2 }$ and $\ell _ { 2 } ( x _ { 1 } , x _ { 3 } ) = ( x _ { 1 } - x _ { 3 } ) ^ { 2 }$ , so $N ( 1 ) = \{ 1 , 2 \}$ and $N ( 2 ) = \{ 1 , 3 \}$

The dependency graph has one vertex per block and joins blocks $b \neq b ^ { \prime }$ whenever some factor depends on both, that is, $b , b ^ { \prime } \in N ( a )$ for some �. It records how the objective couples the blocks, not which agents communicate. A proper coloring with � colors is a map $\chi : \{ 1 , \ldots , m \}  \{ 1 , \ldots , \kappa \}$ that gives adjacent blocks diferent colors; it splits the blocks into the color classes $\mathcal { C } _ { j } : = \{ b : \chi ( b ) = j \}$ . In the example above, the graph has the edges {1, 2} and {1, 3}, and $\chi ( 1 ) = 2$ $\chi ( 2 ) = \chi ( 3 ) = 1$ is a proper coloring with $\kappa = 2 ,$ as in step 4 of Figure 1.

The blocks of one factor are pairwise adjacent, so they all have diferent colors. If several blocks of one color change at once, each factor therefore sees at most one of the changes, and the individual changes add exactly: for every set $S \subseteq { \mathcal { C } } _ { j }$ and candidates $U _ { b } , b \in S$

$$
F ( x _ { - S } , U _ { S } ) - F ( x ) = \sum _ { b \in S } f _ { b } ( U _ { b } ; x ) ,\tag{6}
$$

where $( x _ { - S } , U _ { S } )$ is � with each block $x _ { b } , b \in S _ { \mathrm { \Gamma } }$ , replaced by $U _ { b }$ . This makes the merge of Section 3.3 safe.

Algorithm 1 Corrected local search with verified merging   
Input: $x ^ { 0 } \in { \mathcal { X } } ;$ local objectives ${ L } _ { b } ;$ proper coloring � with �   
colors; private horizon $H ;$ number of epochs �   
1: for $k = 0 , 1 , \ldots , K - 1$ do   
2: Update the correction with verified values (Section 6)   
3: for $b = 1 , \ldots , m$ in parallel do   
4: $\widehat { f _ { b } } ( u ; x ^ { k } ) \gets L _ { b } ( u ) - L _ { b } ( x _ { b } ^ { k } ) + \widehat { R } _ { b } ( u ; x ^ { k } )$   
5: $U _ { b } \gets$ Search<sub>�</sub> $\left( \widehat { f _ { b } } ( \cdot ; x ^ { k } ) , x _ { b } ^ { k } \right)$ ◁ � private steps   
6: $g _ { b } \gets \operatorname* { m a x } \{ 0 , F ( x ^ { k } ) - F ( x _ { - b } ^ { k } , U _ { b } ) \}$ ◁ verify   
7: end for   
8: $j ^ { k } \gets \arg \operatorname* { m a x } _ { 1 \leq j \leq \kappa } \sum _ { b \in \mathcal { C } _ { j } } g _ { b }$ ◁ best color   
9: $S ^ { k } \gets \{ b \in \mathcal { C } _ { j ^ { k } } : g _ { b } > 0 \}$   
10: $x ^ { k + 1 }  x ^ { k }$   
11: $x _ { b } ^ { k + 1 }  U _ { b }$ for all $b \in S ^ { k }$ ◁ merge (or Alg. 2)   
12: end for

Fewer colors are better, because this merge keeps at least a fraction $1 / \kappa$ of the total verified gain (see (8) in Section 3.3); a greedy coloring needs at most one more color than the maximum degree of the graph. At one extreme, a separable task with exact local objectives, $\begin{array} { r } { F ( x ) = \sum _ { b } L _ { b } ( x _ { b } ) } \end{array}$ up to a constant, has a graph without edges: $\kappa = 1$ , the exact correction is zero, every agent can optimize alone and all improvements can be merged at once. At the other extreme, the single factor $\ell _ { 1 } = F$ gives a complete graph and $\kappa = m$ so only one block can safely be replaced per epoch: changes of two blocks that share a factor may each lower the loss and yet raise it together (Appendix A.1). Corrections and coloring matter exactly when blocks share factors. This safety has a price: the coloring separates any two blocks that could interact for some candidates, even when the proposals of an epoch do not. Section 5 lets the coordinator accept proposals of diferent colors together whenever the verifier confirms that they lower the loss more than the best color alone.

## 3.3 Search, verification and merging

Algorithm 1 states the protocol, and Figure 1 illustrates one epoch: a round of private search, verification and merging. Within an epoch, all agents search from the same frozen solution $x ,$ called the base. Before the search, the correction is updated with the latest verified values, in one of the ways described in Section 6. Each agent returns a proposal $U _ { b }$ after � private steps: the best candidate it has found, or its current block if no candidate improves on it (Search<sub>�</sub> in Algorithm 1). The verifier then records its gain over the base,

$$
g _ { b } : = \operatorname* { m a x } \{ 0 , - f _ { b } ( U _ { b } ; x ) \} .\tag{7}
$$

The coordinator then accepts the strictly improving proposals of the color with the largest total gain, which gives the new base $x ^ { + }$ . By (6), these gains add exactly:

$$
F ( x ) - F ( x ^ { + } ) = \operatorname* { m a x } _ { 1 \leq j \leq \kappa } \sum _ { b \in \mathcal { C } _ { j } } g _ { b } \geq \frac { 1 } { \kappa } \sum _ { b } g _ { b } .\tag{8}
$$

The coordinator thus keeps at least the average gain of a color, and the shared objective never increases. Applying all individually helpful proposals at once could increase it, because adjacent blocks share factors.

The best color discards the verified improvements of all other colors, even when they would combine well. Section 5 shows that the coordinator may instead accept the best verified combination of the proposals (Algorithm 2), with the same guarantees and a rate set by how well the proposals actually combine.

## 4 Convergence and the error floor

The guarantee rests on three ingredients: accurate corrections $\widehat { R } _ { b }$ in (4), efective local search and favorable task geometry. Throughout, ℋ denotes everything fixed before a search: the base, correction and past randomness. Expectations are over the agents’ randomness, and all constants must hold at every base the run reaches.

Correction accuracy where it matters. An approximate correction $\widehat { R } _ { b }$ can be badly wrong for poor candidates that no reasonable search returns. What matters is its accuracy at two candidates: a true block minimizer $\boldsymbol { u } _ { b } ^ { * }$ of $f _ { b } ( \cdot ; x )$ , which the agent aims to find, and the proposal $U _ { b }$ that it actually returns.

Assumption 1 (Anchored correction accuracy). There are constants $\delta _ { b } ^ { * } , \delta _ { b } ^ { U } \geq 0$ such that at every epoch the correction is anchored, $\widehat { R } _ { b } ( x _ { b } ; x ) = 0 .$ and its errors at some true block minimizer $\boldsymbol { u } _ { b } ^ { * }$ and at the returned proposal $U _ { b }$ satisfy

$$
\begin{array} { r l } & { ~ | \widehat { R } _ { b } ( u _ { b } ^ { * } ; \boldsymbol { x } ) - R _ { b } ( u _ { b } ^ { * } ; \boldsymbol { x } ) | \leq \delta _ { b } ^ { * } , } \\ & { ~ \mathbb { E } \big [ | \widehat { R } _ { b } ( U _ { b } ; \boldsymbol { x } ) - R _ { b } ( U _ { b } ; \boldsymbol { x } ) | ~ \big | ~ \mathcal { H } \big ] \leq \delta _ { b } ^ { U } . } \end{array}\tag{9}
$$

A uniform bound $\begin{array} { r } { \operatorname* { s u p } _ { u \in \mathcal { X } _ { b } } | \widehat { R } _ { b } ( u ; x ) - R _ { b } ( u ; x ) | \leq \delta _ { b } } \end{array}$ implies (9) with $\delta _ { b } ^ { * } = \delta _ { b } ^ { U } = \delta _ { b }$ , but it can be far larger. In the routing experiments of Section 8.1, for example, a uniform bound exceeds $1 0 ^ { 4 }$ , whereas the errors in (9) measured during the runs are zero at the true block minimizers and about 40 on average at the returned proposals. To verify the assumption, one need not find $u _ { b } ^ { * } .$ , which may be hard or impossible to compute; a bound on the error there sufices, and it can follow from the structure of the objective.

Next, local search must make progress on the corrected objective. Let $\widehat { f } _ { b } ^ { * } : = \mathrm { m i n } _ { u \in { \mathcal { X } } _ { b } } \widehat { f } _ { b } ( u ; x )$ . Since the unchanged block scores zero, an agent can gain at most $0 - \widehat { f } _ { b } ^ { * }$ , and we require its � private steps to capture, in expectation, a fixed fraction of this amount. The staircases in step 2 of Figure 1 depict such progress.

Assumption 2 (Conditional local contraction). The solver starts from the unchanged block, and for some $0 < \rho _ { b } < 1$ its � steps satisfy

$$
\mathbb { E } \big [ \widehat { f } _ { b } ( U _ { b } ; x ) - \widehat { f } _ { b } ^ { * } \mid \mathcal { H } \big ] \le ( 1 - \rho _ { b } ) ^ { H } \big ( 0 - \widehat { f } _ { b } ^ { * } \big ) .\tag{10}
$$

With $q _ { b } ( H ) : = 1 - ( 1 - \rho _ { b } ) ^ { H }$ and $\begin{array} { r } { q ( H ) : = \operatorname* { m i n } _ { b } q _ { b } ( H ) } \end{array}$ $q _ { b } ( H )$ is the guaranteed fraction of the available corrected improvement that agent � captures in � steps; in the routing task of Section 8.1, LLM agents capture on average about 2/3 of this improvement in $H = 3$ steps.

Finally, improving single blocks must be enough to make global progress. Let the best block decrease $D _ { b } ( x )$ be the largest decrease that agent � alone can achieve, and $\Delta ( x )$ the remaining gap to the optimum:

$$
D _ { b } ( x ) : = - \operatorname* { m i n } _ { u \in { \mathcal { X } } _ { b } } f _ { b } ( u ; x ) \geq 0 , \qquad \Delta ( x ) : = F ( x ) - F ^ { * } .\tag{11}
$$

Whenever the solution is suboptimal, the agents together must be able to improve it in proportion to the remaining gap.

Assumption 3 (Block error bound). There is a constant $\mu \in ( 0 , 1 ]$ such that every reached base satisfies

$$
\sum _ { b } D _ { b } ( x ) \geq \mu \Delta ( x ) .\tag{12}
$$

This excludes single-block traps, where only a coordinated change of several blocks helps, for example two modules whose shared interface can be changed only by editing both at once. On a finite domain, nothing more is needed: some $\mu > 0$ exists if and only if every solution that no single agent can improve is optimal. Rearranged, (12) bounds the unknown gap, $\begin{array} { r } { \Delta ( x ) \le \mu ^ { - 1 } \sum _ { b } D _ { b } ( x ) } \end{array}$ , so upper bounds on the block improvements certify how close a solution is to the optimum, although $F ^ { * }$ is unknown (Proposition $\mathrm { A . 1 } )$ In the routing experiments, the ratio $\sum _ { b } D _ { b } ( x ) / \Delta ( x )$ is at least 0.43 at every recorded base from which the team starts a round (Section 8.1), and on the synthetic task all three assumptions are certified (Section 8.3).

Lemma 1 (Correction error becomes lost progress). Under Assumptions 1 and 2,

$$
\begin{array} { r } { \mathbb { E } [ g _ { b } \ | \ \mathcal { H } ] \ge q _ { b } ( H ) D _ { b } ( x ) - q _ { b } ( H ) \delta _ { b } ^ { * } - \delta _ { b } ^ { U } . } \end{array}\tag{13}
$$

Proof. Condition on ℋ, and write the correction error as $\varepsilon _ { b } ( u ) : = \widehat { R } _ { b } ( u ; x ) - R _ { b } ( u ; x )$ . By (3) and $( 4 ) , \widehat { f _ { b } } = f _ { b } + \varepsilon _ { b }$ At the true minimizer, $\widehat { f } _ { b } ^ { * } \leq \widehat { f } _ { b } ( u _ { b } ^ { * } ; x ) = - D _ { b } ( x ) + \varepsilon _ { b } ( u _ { b } ^ { * } ) \leq$ $- D _ { b } ( x ) + \delta _ { b } ^ { * }$ , and local contraction with anchoring gives $\begin{array} { r } { \mathbb { E } \widehat { f } _ { b } ( U _ { b } ; x ) \le q _ { b } ( H ) \widehat { f } _ { b } ^ { * } } \end{array}$ . Hence

$$
\begin{array} { r l } & { \mathbb { E } f _ { b } ( U _ { b } ; \boldsymbol { x } ) = \mathbb { E } [ \widehat { f } _ { b } ( U _ { b } ; \boldsymbol { x } ) - \varepsilon _ { b } ( U _ { b } ) ] \leq q _ { b } ( H ) \widehat { f } _ { b } ^ { * } + \delta _ { b } ^ { U } } \\ & { \qquad \leq - q _ { b } ( H ) D _ { b } ( \boldsymbol { x } ) + q _ { b } ( H ) \delta _ { b } ^ { * } + \delta _ { b } ^ { U } , } \end{array}
$$

and $g _ { b } \ge - f _ { b } ( U _ { b } ; x )$ proves the claim. Verification also ensures $g _ { b } \geq 0$ when this lower bound is negative. □

Theorem 2 (Convergence with a correction-error floor). Under the factor structure, proper coloring and Assumptions 1–3, let

$$
\alpha : = \frac { \mu q ( H ) } { \kappa } , \qquad B : = \frac { 1 } { \kappa } \sum _ { b } \bigl ( q _ { b } ( H ) \delta _ { b } ^ { * } + \delta _ { b } ^ { U } \bigr ) .\tag{14}
$$

From a deterministic $x ^ { 0 }$ , the iterates of Algorithm 1 satisfy, for every $K \geq 0$

$$
\boxed { \mathbb { E } \Delta ( x ^ { K } ) \leq ( 1 - \alpha ) ^ { K } \Delta ( x ^ { 0 } ) + \frac { B } { \alpha } \big [ 1 - ( 1 - \alpha ) ^ { K } \big ] } .\tag{15}
$$

Exact corrections give geometric decay to zero expected gap.

Proof. By (8), Lemma 1 and the block error bound,

$$
\mathbb { E } [ F ( x ) - F ( x ^ { + } ) \mid \mathcal { H } ] \ge \frac { 1 } { \kappa } \sum _ { b } q _ { b } ( H ) D _ { b } ( x ) - B \ge \alpha \Delta ( x ) - B .
$$

Total expectation gives $\begin{array} { r } { \mathbb { E } \Delta ( x ^ { k + 1 } ) \leq ( 1 - \alpha ) \mathbb { E } \Delta ( x ^ { k } ) + B . } \end{array}$ Iterating and summing the geometric series proves (15). □

The rate has three distinct factors: useful block directions $( \mu )$ , finding them $( q )$ , and retaining compatible gains $( 1 / \kappa ) ;$ Section 5 sharpens the third. More local work improves only the second. The floor $B / \alpha$ comes from correction error: more private search cannot bring it below $\sum _ { b } ( \delta _ { b } ^ { * } + \delta _ { b } ^ { U } ) / \mu$ , but a more accurate correction lowers it. The floor belongs to an upper bound, not a level at which the search must stall. If the correction error does not change which candidate scores best, the agent returns the same proposal as with an exact correction and loses nothing; in the synthetic task of Section 8.3, for example, small correction biases leave every run at the optimum. Figure 4 evaluates the recurrence be hind (15) with per-round values measured on LLM runs, and Figure 5(b) compares observed mean convergence with (15) using certified constants.

## 5 Merging interfering proposals

Algorithm 1 accepts only the improving proposals of one color. Blocks of one color share no factors, so this merge is safe for every candidate and keeps at least a fraction $1 / \kappa$ of the verified gain (8). The price is that the verified improvements of all other colors are thrown away. The coloring is built for the worst case: it separates every pair of blocks that could interact for some candidates, whether or not the proposals of a given epoch actually do. In practice, proposals of adjacent blocks often do not interfere at all, and even proposals that do interfere can lower � more together than either does alone. Throwing all of this work away is wasteful. This section shows how the coordinator can keep it and what the theory then guarantees.

Accepting the best combination. A merge accepts the proposals of some set $S \subseteq \{ 1 , \dots , m \}$ of blocks and keeps the other blocks unchanged, giving $( x _ { - S } , U _ { S } )$ as in $( 6 ) ; S = \varnothing$ leaves � unchanged. Algorithm 1 uses $S = S ^ { k }$ , the improving blocks of the best color (line 9). Instead, the coordinator can evaluate $\Phi ( S ) : = F ( x _ { - S } , U _ { S } )$ for every � and accept the set with the lowest loss, even if it contains proposals that interfere (Algorithm 2). Because $S ^ { k }$ is one of these candidates, this best-combination merge never gains less than the best color: the equality in (8) becomes an inequality, and the gain is still at least $\kappa ^ { - 1 } \sum _ { b } ^ { } g _ { b }$

With many blocks, checking all $2 ^ { m }$ sets is impractical, so the coordinator can use a greedy search instead. Starting from $S ^ { k }$ , it repeatedly adds the block whose proposal lowers Φ most, as long as one does; it never raises the loss of the best-color plan, so the same bound holds. Theorem 2 and the convergence results for learned corrections in Appendix A.4 use the merge only through the inequality

$F ( x ) - F ( x ^ { + } ) \geq \kappa ^ { - 1 } \sum _ { b } g _ { b }$ in (8), so they hold for both versions of Algorithm 2.

Algorithm 2 Best-combination merge   
Input: base $x ^ { k }$ , proposals $U _ { 1 } , \ldots , U _ { m } , S ^ { k }$ (line 9 of Algorithm 1)   
1: if all $2 ^ { m }$ sets can be checked then   
2: $S ^ { k } \gets \operatorname { a r g m i n } _ { S } \Phi ( S )$ ◁ all combinations   
3: else   
4: while min ${ } _ { b \notin S ^ { k } } \Phi ( S ^ { k } \cup \{ b \} ) < \Phi ( S ^ { k } )$ do ◁ greedy   
5: $S ^ { k }  S ^ { \dot { k } } \cup$ {︀arg min<sub>�</sub> <sub>/∈�</sub>� $\Phi ( S ^ { k } \cup \{ b \} ) \}$   
6: end while   
7: end if   
8: $x ^ { k + 1 }  ( x _ { - S ^ { k } } ^ { k } , U _ { S ^ { k } } )$ ◁ replaces lines 10–11 of Alg. 1

Realized number of colors. Theorem 2 needs the merge to keep at least a fraction $1 / \kappa$ of the total verified gain $\sum _ { b }$ �<sub>�</sub>. How much Algorithm 2 keeps in epoch � is measured by its realized number of colors

$$
\kappa _ { k } ^ { * } : = \frac { \sum _ { b } g _ { b } } { F ( x ^ { k } ) - F ( x ^ { k + 1 } ) } ,\tag{16}
$$

and $\kappa _ { k } ^ { * } : = 1$ if no block improves. In words, the merge gains $\sum _ { b } g _ { b } / \kappa _ { k } ^ { * }$ , whereas the best color is only guaranteed $\sum _ { b } ^ { - } g _ { b } / \kappa ;$ since the merge never gains less than the best color, $\kappa _ { k } ^ { * } \leq \kappa$ When all $2 ^ { m }$ sets are checked, the merge accepts the combination with the largest gain, so no other way of combining this epoch’s proposals gives a smaller $\kappa _ { k } ^ { * } .$ . The realized number $\kappa _ { k } ^ { * }$ is not the number of accepted groups: if three proposals that gain 100 each gain only 110 together, accepting all three can still be best, and then $\kappa _ { k } ^ { * } = 3 0 0 / 1 1 0 \approx 2 . 7 ;$ if proposals help each other, $\kappa _ { k } ^ { * } < 1$ . In the routing experiments of Section 8.1, where any two agents’ robots can collide and Algorithm 1 accepts one block per round on one map, Algorithm 2 accepts on average 1.7 blocks in each round that makes progress, and $\kappa _ { k } ^ { * }$ averages about 1.2 against $\kappa = 3$

Corollary 3 (Convergence with realized colors). Under the assumptions of Theorem 2, suppose that Algorithm 2 has $\kappa _ { k } ^ { * } \leq \bar { \kappa }$ in every epoch, for some $\bar { \kappa } \geq 1$ . Then its iterates satisfy (15) with � replaced by �¯ in � and � of (14).

Proof. By (16), $\begin{array} { r } { F ( x ) - F ( x ^ { + } ) \geq \sum _ { b } g _ { b } / \kappa _ { k } ^ { * } \geq \bar { \kappa } ^ { - 1 } \sum _ { b } g _ { b } . } \end{array}$ The first relation is an equality unless no block improves, in which case $\begin{array} { r } { \sum _ { b } g _ { b } = 0 } \end{array}$ and the merge never raises the loss; the second holds because the gains are nonnegative. This is (8) with �¯ in place of $\kappa ,$ and the proof of Theorem 2 uses the coloring only through (8). □

Efective number of colors. Corollary 3 holds for any $F ,$ but $\kappa _ { k } ^ { * }$ can only be measured after the epoch. We can, however, bound it from above in advance. Let us assume that blocks interact only in pairs, that is, every factor of � in (5) depends on at most two blocks. This is the case whenever the coupling consists of pairwise conflicts, as in the routing task of Section 8.1, where each collision term involves one pair of robots. For a base � with proposals $U _ { b }$ and $b \neq b ^ { \prime }$ the interaction

$$
\xi _ { b b ^ { \prime } } : = F ( x _ { - \{ b , b ^ { \prime } \} } , U _ { b } , U _ { b ^ { \prime } } ) - F ( x ) - f _ { b } ( U _ { b } ; x ) - f _ { b ^ { \prime } } ( U _ { b ^ { \prime } } ; x )\tag{17}
$$

is zero unless � and $b ^ { \prime }$ share a factor, and expanding the pairwise factors gives, for every set � of blocks,

$$
F ( x _ { - S } , U _ { S } ) - F ( x ) = \sum _ { b \in S } f _ { b } ( U _ { b } ; x ) + \sum _ { \{ b , b ^ { \prime } \} \subseteq S } \xi _ { b b ^ { \prime } } .\tag{18}
$$

Let $I ^ { k } : = \{ b : g _ { b } > 0 \}$ be the improving blocks of epoch �. Two improving blocks � and �<sup>′</sup> conflict if $\xi _ { b b ^ { \prime } } > 0$ , that is, if their proposals together gain less than the sum of their gains, and a set of improving blocks is conflict-free if no two of its blocks conflict. The efective number of colors $\kappa _ { k }$ is the smallest number of conflict-free sets that cover $I ^ { k } \left( \kappa _ { k } : = 1 \right.$ if $I ^ { k } = \varnothing )$ , and $\deg _ { k } ( b )$ is the number of other improving blocks that conflict with �. Whereas � must separate every pair of blocks that could interact for some candidates, � counts only the groups that epoch � needs. Two blocks of one color share no factor and therefore never conflict, so the � color classes already split $I ^ { k }$ into at most � conflict-free groups; and one group per improving block always works. Hence $\kappa _ { k } \leq \operatorname* { m i n } \{ \kappa , | I ^ { k } | \}$ whenever $I ^ { k }$ is nonempty. Moreover, if every improving block conflicts with at most � others, � + 1 groups are always enough: assign the blocks to groups one at a time; a block conflicts with at most � blocks, so at least one of $d + 1$ groups contains none of them, and the block joins that group. Hence $\kappa _ { k } \leq 1 + \operatorname* { m a x } _ { b \in I ^ { k } } \deg _ { k } ( b )$ for nonempty $I ^ { k }$

Proposition 4 (Realized and effective colors). If every factor of � in (5) depends on at most two blocks, then every conflict-free set � of improving blocks satisfies

$$
F ( x ) - F ( x { - } s , U s ) \geq \sum _ { b \in S } g _ { b } .\tag{19}
$$

Consequently, Algorithm 2 with all combinations checked gains at least $\kappa _ { k } ^ { - 1 } \sum _ { b } g _ { b }$ , that is, $\kappa _ { k } ^ { * } \leq \kappa _ { k }$

Proof. For an improving block, $f _ { b } ( U _ { b } ; x ) = - g _ { b }$ . By (18), accepting a set � therefore changes $F \mathrm { \Delta b y - } \sum _ { b \in S } { g _ { b } }$ plus the interactions $\xi _ { b b ^ { \prime } }$ of its pairs. If � is conflict-free, all these interactions are at most zero, which proves (19). Now split the improving blocks into $\kappa _ { k }$ conflict-free groups. Their gains add up to $\sum _ { b } g _ { b }$ , so the best group alone gains at least $\kappa _ { k } ^ { - 1 } \sum _ { b } g _ { b }$ The merge checks this group along with all other sets and accepts a plan at least as good, so $F ( x ) - F ( x ^ { + } ) \geq \kappa _ { k } ^ { - 1 } \sum _ { b } g _ { b } ,$ which by (16) is $\kappa _ { k } ^ { * } \leq \kappa _ { k }$ □

What this means. The three numbers of colors are ordered, $\kappa _ { k } ^ { * } \leq \kappa _ { k } \leq \kappa .$ . The number $\kappa$ in Theorem 2 is the price of throwing away all colors but one, and it is fixed by how the objective could couple the blocks. Corollary 3 replaces it by an upper bound �¯ on $\kappa _ { k } ^ { * }$ , which reflects how well the proposals of each epoch actually combine; the error floor $B / \alpha$ stays the same. The efective number $\kappa _ { k }$ gives a guarantee that can be stated in advance: if no improving proposal conflicts with more than � others, then $\kappa _ { k } ^ { * } \leq \kappa _ { k } \leq d + 1$ , and Corollary 3 holds with $\bar { \kappa } = d + 1$ . This matters most for large teams: on a densely coupled task � grows with the number of agents, whereas $d + 1$ need not. The realized number can be smaller still, because accepting conflicting proposals together counts whenever it lowers $F$ most.

Cost of checking and limits. Checking all $2 ^ { m }$ combinations is afordable for a few agents, as in our experiments with three. For more agents, pairwise interactions help: by (18), the loss of every combination of improving blocks follows from one check per improving block and one per pair of them, so about $| I ^ { k } | ^ { 2 } / 2$ checks replace $2 ^ { m }$ . Appendix E.9 gives a simple rule that picks a combination from these checks and keeps Corollary 3 with $\bar { \kappa } = d + 1$ . Combining proposals cannot, however, remove the need for Assumption 3. At a single-block trap with exact corrections no agent finds an improvement, so every agent returns its current block and no combination of the returned proposals changes anything; Appendix E.9 discusses ways around this.

## 6 Learning corrections

Before every epoch, Algorithm 1 updates the correction with verified values (line 2). This can happen in two ways. First, agents can receive the verifier’s reports in their prompts; in the routing task of Section 8.1, for example, every collision that the verifier reports becomes a known pair whose penalty the correction adds.

Second, when the hidden part of the objective has a known structure, the correction can be learned numerically. Suppose that $F ( x )$ equals $G ( x ) + \langle \vartheta , \phi ( x ) \rangle$ up to a constant, with a known function � and known features � but unknown weights �. Every evaluated trial solution � then gives one linear equation, $F ( z ) - F ( x ) - [ G ( z ) - G ( x ) ] = \langle \phi ( z ) - \phi ( x ) , \vartheta \rangle$ The estimate �̂︀ moves by the smallest amount that satisfies it, a sketch-and-project step (Strohmer and Vershynin, 2009; Gower and Richtárik, 2015). Updating the estimate in this way before an epoch is called calibration. Agents keep the learned weights across epochs and recompute the known terms when the base changes. If the trial solutions explore every unknown direction on average, the solution and the correction contract jointly at a linear rate set by the slower of search and learning, with no error floor (Theorem A.5 in Appendix A.4). Already evaluated proposals, including rejected ones, can be reused as measurements (Proposition A.6). Without such structure, sparse evaluations cannot determine the values of unqueried candidates (Appendix A.5.2). Appendix G.7 tests numerical learning on the synthetic task.

The cost of learning. Learning repays its queries when the needed information is sparse and reused; otherwise measuring the alternatives in advance is cheaper (Appendix A.5 accounts for the cost of calibration). If the unknown terms are linear in individual binary variables, one extra verifier query identifies one coeficient exactly, so at most one query per variable is ever needed, and a computable bound on the remaining correction error contracts linearly together with the gap (Proposition A.7, Corollary A.8).

## 7 Private work and synchronization

Longer private search makes each epoch more productive but also slower, whereas frequent synchronization spends time on verification and context transfer. Let one parallel private step cost $c > 0$ , and let each synchronization, including correction construction, context transfer and verification, cost $C \geq 0$ . With exact corrections, Theorem 2 shows that $K _ { \varepsilon } ( H ) = \lceil \kappa \log ( \Delta ( x ^ { 0 } ) / \varepsilon ) / ( \mu q ( H ) ) \rceil$ epochs reach expected gap �, at cost at most $( c H + C ) K _ { \varepsilon } ( H )$ , where ⌈·⌉ rounds up to an integer. With $\lambda = - \log ( 1 - \operatorname* { m i n } _ { b } \rho _ { b } )$ , dropping the ceiling and the constant factor � log(Δ(�<sup>0</sup>)/�)/� and using $q ( H ) =$ $1 - e ^ { - \lambda H }$ gives the relaxed cost $J ( H ) : = ( c H + C ) / ( 1 - e ^ { - \lambda H } )$ for $H > 0$ , whose unique minimizer for $C \ > \ 0$ satisfies (Proposition A.3)

$$
e ^ { \lambda H _ { * } } = 1 + \lambda H _ { * } + \lambda C / c .\tag{20}
$$

Costlier synchronization therefore calls for longer private work: $H _ { * } ~ \approx ~ \sqrt { 2 C / ( c \lambda ) }$ when $\lambda C / c$ is small. Section 8.3 compares this choice with observed completion costs.

## 8 Experiments

We test the theory in three settings. The main study, robot routing with LLM agents (Section 8.1), asks three questions. (Q1) Can decomposition with corrections let LLM agents solve a coupled task that a single agent struggles with? (Q2) How much does the team gain by merging the best checked combination of proposals (Section 5) instead of a single color? (Q3) Does the bound of Theorem 2, evaluated round by round with constants measured on the runs, hold for the team’s progress, and how closely? Two smaller studies test the remaining parts of the theory. Query planning in SQLite (Section 8.2) asks (Q4) when acquiring correction information pays of (Section 6). A synthetic task with certified constants (Section 8.3) asks (Q5) how tight the bound is, whether a persistent correction error leaves the predicted floor, and how long agents should search between synchronizations (Section 7).

## 8.1 Robot routing with LLM agents

Each robot must reach its goal while avoiding walls and other robots, a multi-agent pathfinding (MAPF) task (Stern et al., 2019). The rules are simple and visual, yet the task remains hard for LLMs (Chen et al., 2025). Specialized solvers such as conflict-based search (Sharon et al., 2012) can solve this problem exactly, and we use them only to certify the optimum. Most importantly, coupling is built in: robots of diferent agents can collide, so the best routes for one agent’s robots can block another agent’s.

Software was our first target, and the introduction uses it as an example. In code, however, strong coupling between the parts that diferent agents would own appears mainly in large systems, because good design keeps modules apart (Parnas, 1972). Experiments at that scale are hard to control and to analyze: the best program is unknown, so one cannot say how far a team is from it or which interactions slowed it down. Routing, on the other hand, provides a certified optimum, lower bounds and exact block minima, so every constant of the theory can be measured. This small, fully analyzable setting showed us what actually happens during a run. Observations made in it inspired two results of this paper: the realized number of colors (Section 5) and the error condition (9), which concerns only two candidates per agent instead of all of them (Section 4). Nothing in the protocol is specific to routing, so it applies to code as well, at a scale where the parts are strongly coupled (Appendix C.1). Below we describe the task and the agents, compare the team with single agents (Q1), test the merge of Section 5 (Q2) and finally confront the runs with the theory (Q3).

![](images/b1ff2f9b4f27a29778e663329cef64f69957957c7749ef99578d39d56b1e6e19.jpg)  
(a) One robot: reach its goal

![](images/3e35d3d1f45258016bb80a2f2115b2607f8b608f9b09b105d114b0aa970d909b.jpg)  
(b) Full task: nine robots

![](images/4a2a3d17d2e66b100e37e5ad2bca3ce64ecb65a8e903c07c21f66ec924f6bbda.jpg)  
(c) One agent: three robots  
Figure 2: Recorded plan at the optimum (first map, team seed 19 after round 4): nine delivered, zero collisions, ${ \boldsymbol { F } } = { \boldsymbol { F } } ^ { * } = 1 4 4$ (a) Robot 1: arrival at tick 20, cost 20. (b) All nine routes. (c) Block $\{ r _ { 0 } , r _ { 1 } , r _ { 6 } \}$ with peers faded. Markers: starts; flags: goals. Overlapping segments are ofset within cells; spatial crossings need not coincide in time.

Rules. Agents write explicit command lists for nine robots on a 16 × 16 grid with walls, over 48 time steps, which we call ticks. At each tick a robot moves one cell up, down, left or right, or stays in place with wait; directions refer to the map, not to the robot. A route may not enter a wall, and a robot that reaches its goal stays there. The pictures below show, from left to right, the commands, a blocked move, a legal route that arrives at tick 4, and a swap.

![](images/edcab9d5b6738605e5a25ded6a76a4344e4a087ab64d980ab052317e963f8f20.jpg)

Collisions couple the robots. Two robots collide when they occupy the same cell at the same tick. They also collide when they exchange cells during one tick, as in the rightmost picture above, since they would have to pass through each other. Two routes can each avoid every wall and still collide. In the example below, if A and B both move, they meet in the center at tick 1. If B waits one tick instead, A passes first and both arrive safely.

![](images/0232cdc0afc17c7e8350e089776886e0677eeddabc0a1bc6a4e670e840e0e529.jpg)

Loss, local objectives and corrections. The delivery cost of robot � is $c _ { i } ( x ) = \tau _ { i } ,$ , where $\tau _ { i }$ is the tick at which it reaches its goal. If the robot does not arrive within the horizon, its cost is $c _ { i } ( x ) = 1 0 0 + 4 8 + d _ { i }$ instead, where $d _ { i }$ is its remaining shortest-path distance to the goal. Here 100 is the penalty for non-delivery; we also add the horizon 48, so that an undelivered robot always costs at least 100 more than a delivered one, and $d _ { i }$ rewards getting closer. Each collision adds a penalty of 100:

$$
F ( x ) = \sum _ { i = 0 } ^ { 8 } c _ { i } ( x ) + 1 0 0 \sum _ { 0 \leq i < j \leq 8 } C _ { i j } ( x ) ,\tag{21}
$$

where $C _ { i j } ( x )$ counts the ticks at which robots � and � collide. In the example above, the safe plan costs $2 + 3 = 5 ,$ since A arrives at tick 2 and B at tick 3. If both move at once, each arrives at tick 2, but the collision raises the loss to 104. We call a plan safe if it delivers every robot without collisions.

Agent � controls a set $\mathcal { T } _ { b }$ of three robots, and its block � lists their commands. Its local objective keeps what it controls, the delivery costs of its robots and the collisions among them:

$$
L _ { b } ( u ) = \sum _ { i \in \mathcal { T } _ { b } } c _ { i } ( u _ { i } ) + 1 0 0 \sum _ { i < j \atop i , j \in \mathcal { T } _ { b } } C _ { i j } ( u _ { i } , u _ { j } ) .\tag{22}
$$

![](images/ddd55f75595d0285e13a3fa64be1858e64c74c4f87dea42f83b6551632f56233.jpg)  
(a) Map 1, nine robots

![](images/7a3c154b5107c559bbe284ddded3ba4e005aa80758ea65e45c2f9856e857d9f6.jpg)  
(b) Map 2, nine robots

![](images/36eee1b5daa7ec4bf66e273f23b07ca8106eec564891e2fa92f8f865904343fa.jpg)  
(c) Two floors, 18 robots  
Figure 3: Mean gap $F - F ^ { * }$ against search time (logarithmic gap axis); 20 runs per curve, 10 without correction. $( \mathrm { a } , \mathrm { b } )$ Maps 1 and 2: nine robots, three team agents; (c) both maps as two floors: 18 robots, six team agents. Best-of-�: � parallel single agents, as many as team agents. Dotted: final mean of the team without correction. Variants: Section 8.1.

Collisions between robots of diferent agents are the reason why the blocks are coupled. The correction accounts for them, but only for the pairs that the agent already knows about. Let $\mathcal { P } ^ { k }$ contain the pairs of robots of diferent agents whose collision the verifier has found before round �. With every other robot $j$ kept on its current route $x _ { j } ^ { k }$ , agent �’s correction is the change in the penalty for these pairs:

$$
\widehat { R } _ { b } ( u ; x ^ { k } ) = 1 0 0 \sum _ { \{ i , j \} \in { \mathscr { P } } ^ { k } } \left[ C _ { i j } ( u _ { i } , x _ { j } ^ { k } ) - C _ { i j } ( x _ { i } ^ { k } , x _ { j } ^ { k } ) \right] .\tag{23}
$$

The agent learns these pairs through its prompt: the verifier reports every collision it finds, and from then on the agent sees the other robot’s current route. The LLM never computes (22) or (23); it searches with what its prompt shows. The system scores each of its attempts by $L _ { b } + \widehat { R } _ { b }$ and keeps the best one as the proposal $U _ { b }$ . Any two agents’ robots may collide, so the dependency graph is complete, $\kappa = 3$ , and Algorithm 1 accepts one block per round.

Maps. Figure 2 shows the first map, with three $2 \times 2$ walls: (a) one robot’s route and cost, (b) an optimal plan found by the team, and (c) the three robots of one agent. Every run starts from the plan in which all robots wait at their starts, which costs $F = 1 4 7 6$ on this map. No plan can cost less than the sum of the robots’ shortest-path distances, 144 here, and collision-free plans with exactly this cost exist, such as the one in Figure $2 ( \mathrm { b } )$ , so the optimum is $F ^ { * } = 1 4 4$ . A plan with a collision or an undelivered robot costs at least $F ^ { * } + 1 0 0$ , so any plan within 100 of the optimum is safe. We did not rely on a single map. We repeated every experiment on a second map, on which the robots start from diferent positions, head to diferent goals and the walls are placed diferently (Figure C.1 in Appendix C). Its optimum is $F ^ { * } = 1 4 7$ , and we report the results on both maps.

Agents. All agents are the open-weight model GPT-OSS-120B (OpenAI, 2025) with medium reasoning, temperature 0.6 and $\mathrm { t o p } { - } p = 0 . 9 5$ , served by Groq through OpenRouter (Appendix C.6). Every prompt states the rules once, then shows what the agent can see of the current plan: a labeled map, the routes and costs of its own robots, a timeline of every robot it can see, drawings of the collisions it knows about, and the attempts already checked. A team agent sees its own robots and the robots in its correction (23). The prompt gives no strategy advice. We tried several prompts and found that the way the plan is presented matters a great deal; we use the best one we found (Appendix D). The team has three agents, each owning three robots (Figure 2(c)). Each round is one epoch of Algorithm 1: every agent gives up to $H = 3$ answers on the frozen plan, and the coordinator accepts the best checked block.

What the correction does. Recall robots A and B from the collision example, and suppose they belong to diferent agents. Once B’s agent knows about A, its correction shows that moving now costs a collision, 100, while waiting costs only one tick. The optimal plan in Figure 2(b) came from such a repair. In team seed 19 on the first map, robot $r _ { 1 }$ swapped cells at tick 11 with robot �<sub>8</sub> of another agent, a pair that an earlier check had already reported. In round 4, the agent owning $r _ { 1 }$ moved it to another shortest route with the same arrival tick 20. This removed the collision and lowered $F$ from 244 to the optimum 144 (Appendix E.6). The agent’s local loss did not change; only the correction, built from $r _ { 8 }$ ’s route, made the new route better.

Baselines. We compare the team with two baselines that use the same model and the same kind of prompt. The single agent sees and scores all nine robots. Each answer replaces the route of one robot of its choice; the agent accepts every improvement at once and stops after 15 answers in a row without improvement. Of the single-agent protocols we tried, this one was the most reliable (Appendix D). The best-of-� runs � copies of the single agent in parallel on the same plan and accepts the best improving answer of each round. With � equal to the number of team agents, three on one map, it matches the team’s parallelism. It stops after five rounds without improvement. So does the team; since each round gives every agent three answers, a team agent, like the single agent, has 15 answers to find an improvement before the run stops. Every variant also stops once the plan is safe and within 9 of the optimum (� ≤ 153 on the first map and $F \le 1 5 6$ on the second). This threshold only ends a run early, and every run is judged by its final plan. Search time adds up the latencies of the model’s answers, network overhead included; in a parallel round only the slowest agent counts (Appendix C.6). Unless stated otherwise, each variant has 20 runs per map, which difer in the model’s sampling seed.

<table><tr><td></td><td colspan="3">Map 1</td><td colspan="3">Map 2</td><td colspan="3">Two floors</td></tr><tr><td>Variant</td><td>Safe runs ↑</td><td> $\boldsymbol { F } - \boldsymbol { F } ^ { * } \downarrow$ </td><td>Search (s) ↓ Safe runs ↑</td><td></td><td> $F - F ^ { * } \downarrow$ </td><td>Search (s) ↓ Safe runs ↑</td><td></td><td> $F - F ^ { * } \downarrow$ </td><td>Search (s) ↓</td></tr><tr><td>Single agent</td><td>100%</td><td> $1 2 . 0 \pm 1 0 . 5$ </td><td> $6 1 3 \pm 3 9 8$ </td><td>95%</td><td> $1 9 . 5 \pm 4 4 . 4$ </td><td> $5 3 8 \pm 3 6 6$ </td><td>100%</td><td> $2 8 . 6 \pm 2 3 . 3$ </td><td> $1 1 4 9 \pm 4 2 0$ </td></tr><tr><td>Best-of-W</td><td>95%</td><td> $1 8 . 9 \pm 4 8 . 8$ </td><td> ${ \bf 3 2 6 \pm 1 9 8 }$ </td><td>90%</td><td> $2 1 . 6 \pm 3 3 . 9$ </td><td> ${ 3 2 6 \pm 1 4 7 }$ </td><td>100%</td><td> $2 1 . 4 \pm 1 6 . 3$ </td><td> ${ \bf 5 1 7 \pm 1 2 4 }$ </td></tr><tr><td>Team (Alg. 1)</td><td>100%</td><td> $5 . 9 \pm 2 . 7$ </td><td> $4 8 1 \pm 2 2 7$ </td><td>100%</td><td> ${ \bf 7 . 2 \pm 1 2 . 6 }$ </td><td> $4 8 2 \pm 2 0 2$ </td><td>100%</td><td> ${ \bf 1 3 . 1 \pm 1 2 . 7 }$ </td><td> $6 5 6 \pm 2 0 6$ </td></tr><tr><td>Team (Alg. 2)</td><td>100%</td><td> $5 . 4 \pm 3 . 2$ </td><td> ${ \bf 3 4 0 \pm 1 9 7 }$ </td><td>100%</td><td> $1 2 . 8 \pm 1 7 . 2$ </td><td> $\mathbf { 3 1 0 \pm 2 0 4 }$ </td><td>100%</td><td> $1 8 . 2 \pm 1 7 . 4$ </td><td> ${ \bf 4 5 2 } \pm 2 2 7$ </td></tr><tr><td>Team unsafe</td><td>100%</td><td> ${ \bf 5 . 0 \pm 2 . 6 }$ </td><td> $6 0 1 \pm 2 8 4$ </td><td>95%</td><td> $1 0 . 1 \pm 2 2 . 3$ </td><td> $4 2 4 \pm 2 3 2$ </td><td>95%</td><td> $1 5 . 1 \pm 2 2 . 5$ </td><td> $7 1 7 \pm 2 6 3$ </td></tr><tr><td>Team, no correction</td><td>0%</td><td> $3 1 3 . 1 \pm 1 0 9 . 2$ </td><td> $3 2 6 \pm 1 7 2$ </td><td>20%</td><td> $1 8 1 . 4 \pm 1 5 3 . 4$ </td><td> $1 7 8 \pm 8 7$ </td><td>0%</td><td> $4 9 4 . 5 \pm 1 8 1 . 9$ </td><td> $3 8 2 \pm 1 7 6$ </td></tr></table>

Table 1: All outcomes per variant and task (20 runs, 10 without correction); mean ± sample SD. Safe runs: share of runs whose final plan delivers every robot without collisions. $F - F ^ { * }$ : final gap to the optimum $\left( F ^ { \ast } = 1 4 4 , 1 4 7 , 2 9 1 \right)$ . Bold: best per column (mean and SD separately), excluding the team without correction. Best-of-�: $W = 3 ,$ or 6 on two floors.

The team finds the best plans. Table 1 lists the outcomes of every variant, and Figure 3(a,b) shows how the mean gap $F - F ^ { * }$ falls with search time on the two maps. Both maps tell the same story. All three protocols reach near-optimal plans, but the team’s plans are the best and the most consistent. The team is safe in all 40 runs and meets the threshold in 39 of them, whereas the single agent meets it in 32 and the best-of-three in 31. The team also ends with the smallest gap on both maps: its mean is lower on map 1 (permutation tests, $p < 0 . 0 0 1$ against the single agent and $p = 0 . 0 1 1$ against the best-of-three) and its median on map $2 \ ( p \leq 0 . 0 0 2 )$ , where a few baseline runs that end with collisions make the diference in means less clear $( p = 0 . 2 6$ and $p = 0 . 0 8 )$ . Of the three, only the team’s mean gap falls below 10. The single agent and the team need similar time in median, but the single agent is less predictable: on the first map, seven of its runs took longer than 700 seconds, whereas only two team runs did. Three single agents in parallel are the fastest, with median search times of about four and a half minutes, against seven for the team $( p = 0 . 0 1 1$ on map 1 and $p = 0 . 0 2 2$ on map 2). They pay for it with worse plans, and 3 of their 40 runs end unsafe. At nine robots, parallel search thus buys speed, whereas decomposition with corrections buys the best plans. Part of the team’s own speed comes from parallelism: played one after another, its recorded answers would take 1.8 times as long on the first map (Appendix E.10). The team also asks the model more often than the single agent, 47 answers against 28 on the first map, and costs USD 0.23 per run against 0.16 (Table D.7).

Is the correction needed? If the agents’ parts were independent, the correction would not matter. To test this, we removed it: agents are told nothing about other agents’ robots, so they receive neither the collision reports nor the peers routes that carry the correction, while verification and merging stay unchanged (Appendices E and D.2). We ran this variant 10 times per map, and only 2 of the 20 runs end safe (Table 1). Every run on the first map delivers all nine robots but keeps one to five collisions, all between robots of diferent agents. Without the collision penalties its final gap would average 3.1, close to the corrected team’s 5.9: each agent solves its own part well, and the remaining loss comes entirely from interactions between the parts. In most runs, the search also ends early, but not because the agents run out of answers. In 6 of 10 runs on the first map, every agent’s robots soon take shortest routes without colliding with each other. Each block is then optimal for its own objective, so no agent can improve it, and the plan freezes with its collisions; 7 of 10 runs on the second map end this way. The coupling is real, and the correction is what resolves it.

A larger task: two floors. To test a larger task, we stack both maps as the two floors of one building. Each floor keeps its own nine robots, so the task has 18 robots, loss $F = F _ { 1 } + F _ { 2 }$ and optimum $F ^ { * } = 1 4 4 + 1 4 7 = 2 9 1$ . Robots on diferent floors never meet. The team has six agents, three per floor, with the same blocks as before. Its dependency graph consists of two separate triangles, so it still needs $\kappa = 3$ colors and can accept one block on each floor per round. Since each team agent sees only its own floor, a two-floor team run is simply a run on each floor, played side by side round by round; each round lasts as long as its slowest agent on either floor. We therefore build the two-floor team runs from the recorded runs of both maps, pairing seed � of the first map with seed � of the second, without new model calls. The single agent, in contrast, sees both floors in one prompt and changes one route on either floor per answer. The parallel baseline now uses six such agents, as many as the team has (Appendix E.4).

![](images/7906170434a90478cdec58013ba60b7d8a7ccc01570152c3b285199361b79360.jpg)  
(a) Bounds on two floors

![](images/b98e59f5d4f5a931b2f20851fe3a350d38e47a06f4b2991da8f1f18aaefb22c7.jpg)  
(b) Correction error

![](images/2fa955f24455b000f75bf1232dfc2276487380b7e33e37416c69093ca1ab1965.jpg)  
(c) Conflicts  
Figure 4: (a) Two floors: mean gap of each team and its bounds with the divisors of Section 5. (b) Mean correction error at the agents’ proposals, maps 1 and 2; 100 is one collision unknown to the correction. (c) Rounds of Team (Alg. 2) with severa improving proposals, by their $\kappa _ { k }$ , or with one or none; bar height: active runs. All tasks: Figure E.4.

With 18 robots the team’s advantage grows (Table 1, Figure 3(c)). All three protocols stay safe in every run, but the team meets both thresholds in 19 runs, against 9 for the single agent and 16 for the best-of-six. The team also ends with the best plans: its median final gap is 11, against 22.5 for the single agent and 15.5 for the best-of-six $( p \leq 0 . 0 0 4 )$ The single agent now struggles with the size of the task. Still changing one route per answer, it needs 63 answers on average, more than twice as many as on one map, and its median search takes 1200 seconds, twice the team’s 595 $( p < 0 . 0 0 1 )$ The team’s time grows less, because both floors are searched at the same time. The best-of-six is faster than Team (Alg. 1) on average, though not significantly in median (means 517 and 656 seconds, $p = 0 . 0 1$ ; medians 511 and 595, $p = 0 . 1 9 )$ ， and it ends with worse plans. Without correction, no run is safe.

Keeping the discarded work. Because the dependency graph is complete, Algorithm 1 accepts one block per round and throws the other agents’ checked proposals away. In the runs above this happened in most rounds that made progress: in 79 of 121 such rounds on the first map and in 64 of 115 on the second, at least one discarded proposal would also have lowered the loss. On average, the improving proposals of a round, each counted on its own, gained 1.5 times as much as the accepted one. This waste led us to Algorithm 2. In Team (Alg. 2), the coordinator checks every nonempty combination of the changed blocks’ proposals, at most $2 ^ { 3 } - 1 =$ 7, and accepts the one with the lowest loss; with three agents, checking them all is cheap. Theorem 2 still applies, because the best combination is never worse than the best color. For comparison, Team unsafe merges all changed blocks without any check and keeps the best plan it has seen. Both variants difer from Team (Alg. 1) only in the merge rule and in the one prompt sentence that describes it (Appendix E.5).

Keeping the work pays of (Table 1, Figure 3). In rounds that make progress, Team (Alg. 2) updates 1.7 blocks on average, whereas Algorithm 1 updates only one. This makes it about as fast as the best-of-�, the fastest protocol so far, on map 1 and faster on map 2 and two floors. Its mean gap lies below that of the best-of-� almost throughout the search, so it reaches almost every level of loss sooner, and it ends lower on every task (Figure 3). Against Team (Alg. 1), its mean gap falls faster on every task as well. It is also the first to reach a safe plan, with a median under four minutes on every task. That is 40–50% sooner than Team (Alg. 1) $( p \leq 0 . 0 0 7 )$ and 22–39% sooner than the best-of-� $( p < 0 . 0 0 1$ on map 2 and two floors, $p = 0 . 2 7$ on map 1). Its whole search is about 30% shorter in mean than with Algorithm 1 $( p < 0 . 0 5$ on every task), with fewer rounds and a lower cost. The final plans stay close: all are safe, and their median gaps are within 2 of those of Algorithm 1 on every task, with no significant diference.

Two caveats remain. First, on map 2 the mean gap of Team (Alg. 2) levels of near 13, above that of Team (Alg. 1). Six of its runs stop above the threshold there, against one with Algorithm 1 $( p = 0 . 0 9 )$ ; the same runs raise its final gap on two floors. These stops were not caused by the merge discarding improvements. In each of these runs some block could still be improved, but for five rounds its agent either found nothing or proposed a change that collided with a robot missing from its correction, and the check rejected that change. Any improvement that passed the check would have been accepted (Appendix E.5). Second, merging without a check does not help. Team unsafe is not significantly faster than Team (Alg. 1), and it is slower in mean on map 1 and two floors. Its plan gets worse in 32 of 185 rounds on map 1 and in 37 of 148 on map 2, and one of its final plans keeps a collision. What makes merging pay of is the check: every merged plan is evaluated before it is accepted. The check need not cover every combination, since the greedy version of Algorithm 2, which needs far fewer checks when there are many agents, keeps the guarantee of Theorem 2 (Section 5).

![](images/767170835c7d6441eabbb05072c3c7a2cc8ea67e0af01ad901ad2650a20f56d6.jpg)  
(a) SQLite feedback cost

![](images/da8a29f738dd8ec6c0983c327df6386f0403a11692ad6e84bd34cefd8f2cefbb.jpg)  
(b) Synthetic convergence

![](images/3f0fcfd1dac649346c70a886625849d931b712a028b1f727bc135ec8519063c7.jpg)  
(c) Completion cost

![](images/6dcd9dd157e44b1adca3f85309630bbd2f1c4b448638b796c84f92951f65b68c.jpg)  
(d) Error floor  
Figure 5: (a) SQLite controller time relative to measuring in advance (below 1: faster) on favorable (Fav.) and balanced (Bal.) workloads; points: databases. (b–d) Synthetic task, 2,000 runs per curve or point: (b) mean gap and Theorem 2 bound (dashed); (c) modeled cost, triangles at $H _ { * }$ (Section 7); (d) final gap under a correction bias � and the bound (dashed).

Measured constants. To compare the runs with the theory, we audited every search of Team (Alg. 1) ofline, computing the exact block minima, with and without the correction, at every recorded base (Appendix E.7). The audit first confirms that the coupling is strong: about a third of the proposals that deliver all three of an agent’s robots without collisions among them still hit another agent’s robot (30% on map 1, 32% on map 2). It then measures the constants of Theorem 2. The block error bound (Assumption 3) holds with $\mu = 0 . 4 3$ on map 1 and 0.70 on map 2 at every state that starts a round. Six final states, all within 3 of the optimum, are single-block traps, where only a joint change of two blocks helps. A search captures on average about two thirds of the available corrected improvement (0.66 and 0.65), which plays the role of $q _ { b } ( H )$ . The correction error is zero at the true block minimizer of every recorded base and averages about 40 at the proposals the agents return (45.5 and 36.5), whereas a uniform bound over all routes would exceed $1 0 ^ { 4 }$ . This gap is what the two-point condition (9) exploits.

The bound, round by round. We then apply the recurrence $\Delta _ { k + 1 } \leq ( 1 - \alpha _ { k } ) \Delta _ { k } + B _ { k }$ behind Theorem 2 to each run, computing $\alpha _ { k }$ and $B _ { k }$ from that round’s measurements, and average the resulting bounds over runs (Appendix E.8, Proposition A.2). Nothing is fitted to the loss curve. The bound holds in every round of every run with either merge rule, on all three tasks. It also carries over between runs: on both maps, coeficients averaged over 10 runs of Team (Alg. 1) bound the mean gap of the other 10 in every round. As the agents learn more collision pairs, the correction becomes accurate (Figure 4(b)). The error at the true block minimizers is zero, so $B _ { k }$ comes only from the error at the returned proposals. That error averages about 100 in round 1, one collision the correction did not know about, and falls to a few units within six or seven rounds, an empirical counterpart of Theorem A.5. It falls faster with Algorithm 2, to 19 by round 3 against 56 with Algorithm 1: its coordinator checks combinations of proposals, and every collision found in these checks also becomes known to the correction.

With $\kappa = 3 ,$ , however, the bound is loose late in a run. On two floors it ends at 396 for Algorithm 1 and 605 for Algorithm 2 (gray curves in Figure 4(a)), while the observed mean gaps are 13 and 18. The cause is the discarded work seen above: with $\kappa = 3$ the bound credits a round with only a third of the blocks’ gains, whereas Algorithm 1 accepts the best block and Algorithm 2 often more. The smaller divisors of Section 5 tighten the bound step by step. Dividing by min $\{ \kappa , | I ^ { k } | \}$ , which bounds the number of colors that contain an improving block, and by 1 in rounds without one, lowers the final bounds to $^ { 8 7 }$ and 191. The efective number of colors $\kappa _ { k } ,$ which bounds the realized one by Proposition 4, needs checked combinations and therefore applies only to Algorithm 2; it lowers that bound further to 133. The realized number of colors $\kappa _ { k } ^ { * }$ (Corollary 3), which averages about 1.5 for Algorithm 1 and 1.2 for Algorithm 2, gives the tightest bounds, 50 and 70. On all three tasks, the bound with $\kappa _ { k } ^ { * }$ ends within a factor of 3 to 7 of the observed mean gap, instead of 30 to 60 with � = 3 (Appendix E.9).

Coupling during a run. Why is the realized number of colors so much smaller than $\kappa = 3 ?$ Because the coupling changes as the plan improves. Team (Alg. 2) checks every combination, so its runs show which proposals actually conflict (Table E.10). Figure 4(c) sorts its rounds by the efective number of colors $\kappa _ { k }$ that their improving proposals need; rounds in which only one proposal improves, or none, are shown apart, since they leave nothing to combine. The worst case $\kappa _ { k } = 3$ is rare: it occurs in 8 of the 105 rounds with several improving proposals, 6 of them in round 1. A run passes through three phases. At the start, nearly every agent improves its block, and the new routes often collide: in round 1, 34 of 58 pairs of proposals conflict on map 1 and 19 of 56 on map 2, and most rounds need two colors. Conflicting proposals are still worth combining, because delivering robots gains far more than a collision costs; the best combination accepts a conflicting pair in more than half of the first rounds. The plan therefore goes from no collisions to 2.3 and 3.3 on average after round 1. In the repair phase that follows, while the plan still has a collision or an undelivered robot, fewer agents improve per round. Their proposals still conflict often, in half of the pairs on map 1 and in more than two thirds on map 2, frequently because both remove the same collision. Yet in rounds 2 to 4, 16 of the 60 rounds with several improving proposals have no conflict at all, and the best combination accepts all of them $( \kappa _ { k } = 1 )$ . Once the plan is safe, the task is nearly decoupled: in most rounds at most one agent improves, and $\kappa _ { k }$ averages 1.03 and 1.04; from round 5 on, nearly every round has one improving proposal or none. Coupling is thus strongest in the first rounds, when most of the loss is removed, and fades near the optimum, whereas the coloring, which must hold for every candidate, assumes three colors throughout.

In short. Decomposition with corrections gives the most reliable and the best plans, and its advantage grows with the size of the task (Q1). Checking combinations of proposals keeps the agents’ work, so the team reaches small gaps sooner than parallel single agents and is the first to reach a safe plan on every task (Q2). The bound of Theorem 2, in its round-by-round form (Proposition A.2), holds in every round of every run with a checked merge, and with realized colors it ends within a factor of 3 to 7 of the observed gap (Q3).

## 8.2 Execution feedback in SQLite

This study isolates the acquisition trade-of of Section 6 on a real database engine; the controller is a deterministic program, not an LLM. A query joins two groups of rows through a large edge table, and SQLite can start from either group; the cheaper side depends on the groups’ edge counts, which are unknown in advance. The controller’s correction is a table of learned edge counts: after a query runs, SQLite reports how many edges it visited (SQLite, 2026), and the controller reuses this count for later queries on the same group, as execution feedback does in LEO and Bao (Stillger et al., 2001; Marcus et al., 2021). The alternative is to count the candidate groups in advance, which costs extra queries but avoids mistakes.

Figure 5(a) compares the two on three workloads over three generated databases with 4.9 million edges, charging all measurement time. On a favorable stream of 48 queries, where a few costly groups recur, learning from feedback takes 0.65 of the time of counting in advance (4.92 versus 7.53 ms). Over 192 queries, amortized advance counting narrows this to 0.84. On a balanced stream, where costly groups are frequent and rarely recur, learning loses and takes 1.64 times as long. Learning a correction thus pays of when the needed information is sparse and reused (Appendix F).

## 8.3 Synthetic task with certified constants

Objective. Six blocks of eight binary variables, $x _ { b } \in \{ 0 , 1 \} ^ { 8 }$ interact along a path. With targets $t _ { b j }$ , unary weights $a _ { b j } \in$ [1, 1.5] and coupling weight $w = 0 . 2$ , the loss is

$$
F ( \boldsymbol { x } ) = \sum _ { b = 1 } ^ { 6 } \sum _ { j = 1 } ^ { 8 } a _ { b j } \lvert x _ { b j } - t _ { b j } \rvert + w \sum _ { b = 1 } ^ { 5 } \sum _ { j = 1 } ^ { 8 } \lvert x _ { b j } - x _ { b + 1 , j } \rvert .\tag{24}
$$

The fixed task starts at $x ^ { 0 } = 1 - t ;$ its unique minimizer is �, with $F ^ { * } = 3 . 4$ and $\Delta _ { 0 } ~ = ~ 5 8 . 6 0$ . The private score $\begin{array} { r } { L _ { b } ( u ) = \sum _ { i } a _ { b j } | u _ { j } - \tilde { t } _ { b j } | } \end{array}$ uses the wrong target $\tilde { t } _ { b } = 1 - t _ { b }$ in blocks 1, 3 and $5 ,$ a deliberate local/global mismatch that the exact correction removes.

Search and verified assumptions. Each private step picks a random bit of the block and, with a block-specific probability $\pi _ { b }$ between 0.5 and 1.0, sets it to minimize the corrected score; the two colors are {1, 3, 5} and {2, 4, 6}. With exact corrections, $\delta _ { b } ^ { * } = \delta _ { b } ^ { U } = 0 , \rho _ { b } = \pi _ { b } / 8$ and $\kappa = 2 .$ , and an exact computation certifies $\mu = 0 . 7 1 9 3$ , rounded down (Appendix G). Theorem 2 then gives, with no fitted coeficient,

$$
\mathbb { E } F ( x ^ { K } ) \le F ^ { * } + \left( 1 - \frac { \mu q ( H ) } { 2 } \right) ^ { K } \Delta _ { 0 } .
$$

Results and the cost of private work. For every horizon � from 1 to 128 (powers of two), all 2,000 exact-correction runs reach the optimum, whereas none of 2,000 uncorrected runs at $H = 8$ does: they stall at mean gap 31 (Figure 5(b)). The certified bound holds at every epoch and is within a factor of two of the observed mean gap for three epochs at $H = 8 ;$ later, convergence is faster than the worst case. Figure $5 ( \mathrm { c } )$ charges $H + C$ modeled units per epoch, where � is the cost of one synchronization. Its triangles mark the horizon $H _ { * }$ that minimizes $J ( H )$ (Section 7), computed from the certified constants alone: 1, 11, 20 and 35 for $C = 0 , 5 , 2 0$ , 100. Like the observed minima (1, 16, 64, 64), $H _ { * }$ grows with $C ,$ and its interpolated cost is at most 1.29 times the best tested one. The observed minima lie further right because long private search solves each block exactly, so runs finish in about two epochs, faster than the geometric rate behind �.

Approximate corrections and the floor. Every exact correction coeficient here receives a fixed bias ±� with random signs, unknown to the agents. Below the smallest coeficient magnitude, 0.605, the bias changes no decision and every run reaches the optimum. Larger biases leave a floor that grows with $\nu ,$ from 3.08 at $\nu = 1$ to 30.0 at $\nu = 2$ for $H = 8$ , and is essentially the same for � = 32 (Figure 5(d)): longer private search cannot remove it. The recurrence with measured two point errors bounds the floor in every run (Appendix G.6).

## 9 Conclusions and limitations

Dividing a shared task among agents works when each agent can see, through a correction, how its changes afect the others. We formalized this as corrected block optimization with verified, compatible merging, whose rate and error floor are set by correction accuracy, local search quality and block geometry. A coordinator that accepts the best verified combination of proposals keeps these guarantees, and its rate depends on how well the proposals actually combine. In practice, agents need corrections built from verified interactions, given as numbers or as information in their prompts, proposals verified against a common base, and merges that are either verified as a whole or limited to non-interacting changes.

With GPT-OSS-120B, a team of three agents found collision-free plans in all 40 runs on two maps, better than those of a single agent, which was safe in all but one run; three single agents searching in parallel were faster than this team but ended with worse plans more often. On a larger task made of both maps, a team of six agents kept its better plans and needed half the single agent’s median search time. Merging the best verified combination of proposals keeps the agents’ compatible work: it cut the team’s search time by about 30%, let it reach low losses sooner than the parallel single agents and made it, in median, the first to reach a safe plan, with the same guarantees, while merging all proposals unchecked did not help.

Coupling also rose and fell during the solve: the plan had no collisions at the start, several after the first round, when nearly all agents changed their parts at once, and none at the end, when at most one agent still improved in most rounds. We expect the same arc in other shared tasks, because parts that are first changed independently must then be reconciled, and near a good solution few changes remain; a merge that adapts to the conflicts that actually occur, as the best combination does, then pays for coupling only while it lasts.

Limitations. The block error bound excludes tasks that require coordinated changes of several blocks; in routing it holds at every audited state that starts a round, but a few final states near the optimum are single-block traps. For LLM agents we measure the constants on visited histories but cannot certify the uniform conditional contraction of Assumption 2. Our LLM evidence comes from two maps and their two-floor combination, one model and up to 20 runs per variant. On a single map, a single agent with a well-presented state also solves the task and parallel single agents search faster than Team (Alg. 1). The prompt strongly afects both protocols (Appendix D), so prompts must be matched before protocols are compared.

## Acknowledgments

The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) CRG Grant ORFS-CRG12-2024-6460, and iii) Center of Excellence for Generative AI, under award number 5940.

## References

D. P. Bertsekas and J. N. Tsitsiklis. 1989. Parallel and Distributed Computation: Numerical Methods. Prentice-Hall, Englewood Clifs, NJ. https://web.mit.edu/dimitrib/www/pdc.html

E. Bowring, J. P. Pearce, C. Portway, M. Jain, and M. Tambe. 2008. On K-Optimal Distributed Constraint Optimization Algorithms: New Bounds and Algorithms. In AAMAS, Vol. 2, pp. 607–614. https://www.ifaamas.org/Proceedings/aamas08/proceedings/pdf/ paper/AAMAS08\_0112.pdf

J. K. Bradley, A. Kyrola, D. Bickson, and C. Guestrin. 2011. Parallel Coordinate Descent for L1-Regularized Loss Minimization. Proceedings of the 28th International Conference on Machine Learning (ICML), pages 321–328. https://icml.cc/2011/papers/231\_icmlpa per.pdf

M. Cemri, M. Z. Pan, S. Yang, L. A. Agrawal, B. Chopra, R. Tiwari, K. Keutzer, A. Parameswaran, D. Klein, K. Ramchandran, M. A.

Zaharia, J. E. Gonzalez, and I. Stoica. 2025. Why Do Multi-Agent LLM Systems Fail? Advances in Neural Information Processing Systems, 38, Datasets and Benchmarks Track, pp. 135676–135729. https://doi.org/10.52202/085713-4082

W. Chen, S. Koenig, and B. Dilkina. 2025. Solving Multi-Agent Path Finding as an LLM Benchmark: How, How Good and Why. Transactions on Machine Learning Research; preprint arXiv:2401.03630. https://arxiv.org/abs/2401.03630

W. Chen, Y. Su, J. Zuo, C. Yang, C. Yuan, C.-M. Chan, H. Yu, Y. Lu, Y.-H. Hung, C. Qian, Y. Qin, X. Cong, R. Xie, Z. Liu, M. Sun, and J. Zhou. 2024b. AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors. ICLR. https: //openreview.net/forum?id=EHg5GDnyq1

Y. Chen, J. Arkin, Y. Zhang, N. Roy, and C. Fan. 2024a. Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems? IEEE International Conference on Robotics and Automation (ICRA), pages 4311–4317. https://doi.org/10.110 9/ICRA57147.2024.10610676

Y. Dang, C. Qian, X. Luo, J. Fan, Z. Xie, R. Shi, W. Chen, C. Yang, X. Che, Y. Tian, X. Xiong, L. Han, Z. Liu, and M. Sun. 2025. Multi-Agent Collaboration via Evolving Orchestration. Advances in Neural Information Processing Systems, 38, pp. 182975–183009. https://doi.org/10.52202/085713-5502

Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch. 2024. Improving Factuality and Reasoning in Language Models through Multiagent Debate. Proceedings of ICML, PMLR 235:11733–11763. https://proceedings.mlr.press/v235/du24e.html

Y. Fan, J. Zhang, K. Cai, J. Yang, C. Tang, J. Wang, and K. Wang. 2026. Cost-Efective Communication: An Auction-based Method for Language Agent Interaction. Proceedings of AAAI, 40(35):29412– 29420. https://doi.org/10.1609/aaai.v40i35.40182

R. M. Gower and P. Richtárik. 2015. Randomized Iterative Methods for Linear Systems. SIAM Journal on Matrix Analysis and Applications, 36(4):1660–1690. https://doi.org/10.1137/15M1025487

Groq. 2025. What is a Language Processing Unit? Groq blog, 7 March 2025. Accessed 17 October 2026. https://groq.com/blog/the-groqlpu-explained

Groq. 2026. OpenAI GPT-OSS 120B. Accessed 7 October 2026. https://console.groq.com/docs/model/openai/gpt-oss-120b

S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, C. Zhang, J. Wang, Z. Wang, S. K. S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber. 2024. MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. ICLR. https://arxiv.org/ abs/2308.00352

Y. Hu, Y. Cai, Y. Du, X. Zhu, X. Liu, Z. Yu, Y. Hou, S. Tang, and S. Chen. 2025. Self-Evolving Multi-Agent Collaboration Networks for Software Development. ICLR. https://arxiv.org/abs/2410.16946

K. Kandasamy, J. Schneider, and B. Póczos. 2015. High Dimensional Bayesian Optimisation and Bandits via Additive Models. In ICML, PMLR 37:295–304. https://proceedings.mlr.press/v37/kandasam y15.html

S. S. Kannan, V. L. N. Venkatesh, and B.-C. Min. 2024. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12140–12147. https://doi.org/10.1109/ IROS58592.2024.10802322

S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh. 2020. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. ICML, PMLR 119:5132–5143. https: //proceedings.mlr.press/v119/karimireddy20a.html

T. Khot, H. Trivedi, M. Finlayson, Y. Fu, K. Richardson, P. Clark, and A. Sabharwal. 2023. Decomposed Prompting: A Modular Approach for Solving Complex Tasks. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=\_nGg zQjzaRy

Y. Kim, K. Gu, C. Park, C. Park, S. Schmidgall, A. A. Heydari, Y. Yan, Z. Zhang, Y. Zhuang, Y. Liu, M. Malhotra, P. P. Liang, H. W. Park, Y. Yang, X. Xu, Y. Du, S. Patel, T. Althof, D. McDuf, and X. Liu. 2026. Towards a Science of Scaling Agent Systems. arXiv:2512.08296v3, revised April 2026 (first posted 2025). https: //arxiv.org/abs/2512.08296v3

Z. Liu, Y. Zhang, P. Li, Y. Liu, and D. Yang. 2024. A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration. COLM. https://openreview.net/forum?id=XII0Wp1XA9

R. T. Maheswaran, J. P. Pearce, and M. Tambe. 2004. Distributed Algorithms for DCOP: A Graphical-Game-Based Approach. In Proceedings of the 17th International Conference on Parallel and Distributed Computing Systems (PDCS).

Z. Mandi, S. Jain, and S. Song. 2024. RoCo: Dialectic Multi-Robot Collaboration with Large Language Models. IEEE International Conference on Robotics and Automation (ICRA), pages 286–299. https://doi.org/10.1109/ICRA57147.2024.10610855

R. Marcus, P. Negi, H. Mao, N. Tatbul, M. Alizadeh, and T. Kraska. 2021. Bao: Making Learned Query Optimization Practical. In SIGMOD, pp. 1275–1288. https://doi.org/10.1145/3448016.345283 8

K. Mishchenko, G. Malinovsky, S. U. Stich, and P. Richtárik. 2022. ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally! ICML, PMLR 162:15750–15769. https://proceedings.mlr.press/v162/mishchenko22b.html

D. Monderer and L. S. Shapley. 1996. Potential Games. Games and Economic Behavior, 14(1):124–143. https://doi.org/10.1006/game .1996.0044

K. Murota. 2003. Discrete Convex Analysis. SIAM Monographs on Discrete Mathematics and Applications, Vol. 10. SIAM, Philadelphia, PA. https://doi.org/10.1137/1.9780898718508

Yu. Nesterov. 2012. Eficiency of Coordinate Descent Methods on Huge-Scale Optimization Problems. SIAM Journal on Optimization, 22(2):341–362. https://doi.org/10.1137/100802001

A. Novikov, N. V u, M. Eisenberger, E. Dupont, P.-S. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog. 2025. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131. https://arxiv.org/abs/2506.13131

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925. doi:10.48550/arXiv.2508.10925

OpenRouter. 2026. Provider Routing. Accessed 7 October 2026. https://openrouter.ai/docs/guides/routing/provider-selection

J. Park, V. Kontonis, S. Garg, A. Krishnamurthy, and D. Papailiopoulos. 2026. Scaling discovery through test-time communication. arXiv:2609.21032. https://arxiv.org/abs/2609.21032

D. L. Parnas. 1972. On the Criteria To Be Used in Decomposing Systems into Modules. Communications of the ACM, 15(12):1053– 1058. https://doi.org/10.1145/361598.361623

A. Petcu and B. Faltings. 2005. A Scalable Method for Multiagent Constraint Optimization. IJCAI, pp. 266–271. https://www.ijcai. org/Proceedings/05/Papers/0445.pdf

C. Qian, W. Liu, H. Liu, N. Chen, Y. Dang, J. Li, C. Yang, W. Chen, Y. Su, X. Cong, J. Xu, D. Li, Z. Liu, and M. Sun. 2024. ChatDev: Communicative Agents for Software Development. Proceedings of ACL, pages 15174–15186. https://doi.org/10.18653/v1/2024.acllong.810

P. Richtárik and M. Takáč. 2016. Parallel Coordinate Descent Methods for Big Data Optimization. Mathematical Programming, 156(1– 2):433–484. https://doi.org/10.1007/s10107-015-0901-6

B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi. 2024. Mathematical discoveries from program search with large language models. Nature, 625(7995):468– 475. https://doi.org/10.1038/s41586-023-06924-6

S. Sakai, M. Togasaki, and K. Yamazaki. 2003. A note on greedy algorithms for the maximum weighted independent set problem. Discrete Applied Mathematics 126, 2–3 (2003), 313–322.

C. Scherrer, A. Tewari, M. Halappanavar, and D. Haglin. 2012. Feature Clustering for Accelerating Parallel Coordinate Descent. Advances in Neural Information Processing Systems 25 (NIPS 2012). https: //proceedings.neurips.cc/paper/2012/hash/4e732ced3463d06de0c a9a15b6153677-Abstract.html

G. Sharon, R. Stern, A. Felner, and N. Sturtevant. 2012. Conflict-Based Search For Optimal Multi-Agent Path Finding. In Proceedings of the AAAI Conference on Artificial Intel ligence, Vol. 26. AAAI Press, Palo Alto, CA, 563–569. doi:10.1609/aaai.v26i1.8140

N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems, 36, pp. 8634– 8652. https://proceedings.neurips.cc/paper\_files/paper/2023/hash /1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

M. Soare, A. Lazaric, and R. Munos. 2014. Best-Arm Identification in Linear Bandits. In Advances in Neural Information Processing Systems, 27, pp. 828–836. https://arxiv.org/abs/1409.6110

SQLite. 2026. Prepared Statement Scan Status; Prepared Statement Scan Status Opcodes. Documentation, accessed September 2026. https://www.sqlite.org/c3ref /stmt\_scanstatus.html; https: //www.sqlite.org/c3ref/c\_scanstat\_est.html.

R. Stern, N. Sturtevant, A. Felner, S. Koenig, H. Ma, T. Walker, J. Li, D. Atzmon, L. Cohen, T. K. S. Kumar, E. Boyarski, and R. Barták. 2019. Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks. In Proceedings of the Twelfth International Symposium on Combinatorial Search (SoCS), Vol. 10. AAAI Press, Palo Alto, CA, 151–158. doi:10.1609/socs.v10i1.18510

M. Stillger, G. M. Lohman, V. Markl, and M. Kandil. 2001. LEO— DB2’s LEarning Optimizer. In VLDB, pp. 19–28. https://research .ibm.com/publications/leo-db2s-learning-optimizer

T. Strohmer and R. Vershynin. 2009. A Randomized Kaczmarz Algorithm with Exponential Convergence. Journal of Fourier Analysis and Applications, 15(2):262–278. https://doi.org/10.1007/s00041- 008-9030-4

R. Tappenden, P. Richtárik, and J. Gondzio. 2016. Inexact Coordinate Descent: Complexity and Preconditioning. Journal of Optimization Theory and Applications, 170(1):144–176. https://doi.org/10.100 7/s10957-016-0867-4

P. Tseng. 2001. Convergence of a Block Coordinate Descent Method for Nondiferentiable Minimization. Journal of Optimization Theory and Applications, 109(3):475–494. https://doi.org/10.1023/A: 1017501703105

Z. Wang, Y. Wang, X. Liu, L. Ding, M. Zhang, J. Liu, and M. Zhang. 2025. AgentDropout: Dynamic Agent Elimination for Token-Eficient and High-Performance LLM-Based Multi-Agent Collaboration. Proceedings of ACL, pages 24013–24035. https: //doi.org/10.18653/v1/2025.acl-long.1170

Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations. COLM. https://openreview.net/for um?id=BAakY1hNKS

C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen. 2024. Large Language Models as Optimizers. International Conference on Learning Representations (ICLR). https://openreview.net/forum ?id=Bb4VGOWELI

G. Zhang, Y. Yue, Z. Li, S. Yun, G. Wan, K. Wang, D. Cheng, J. X. Yu, and T. Chen. 2025b. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems. ICLR. https: //arxiv.org/abs/2410.02506

G. Zhang, Y. Yue, X. Sun, G. Wan, M. Yu, J. Fang, K. Wang, T. Chen, and D. Cheng. 2025c. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks. Proceedings of ICML, PMLR 267:76678–76692. https://proceedings.mlr.press/ v267/zhang25cu.html

J. Zhang, J. Xiang, Z. Yu, F. Teng, X. Chen, J. Chen, M. Zhuge, X. Cheng, S. Hong, J. Wang, B. Zheng, B. Liu, Y. Luo, and C. Wu. 2025a. AFlow: Automating Agentic Workflow Generation. ICLR. https://arxiv.org/abs/2410.10762

W. Zhang, G. Wang, Z. Xing, and L. Wittenburg. 2005. Distributed stochastic search and distributed breakout: properties, comparison and applications to constraint optimization problems in sensor networks. Artificial Intelligence, 161(1–2):55–87. https://doi.org/ 10.1016/j.artint.2004.10.004

X. Zhang, Z. Wang, Z. Li, J. Yao, and J. Li. 2026. CommCP: Eficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction. ICRA, 2026; arXiv:2602.06038. https: //arxiv.org/abs/2602.06038

M. Zhuge, W. Wang, L. Kirsch, F. Faccio, D. Khizbullin, and J. Schmidhuber. 2024. GPTSwarm: Language Agents as Optimizable Graphs. Proceedings of ICML, PMLR 235:62743–62767. https://proceedings.mlr.press/v235/zhuge24a.html

## Notation

This table collects the notation used in the main text and the appendices. Each entry gives the meaning of a symbol. The same letter sometimes has a diferent meaning in a proof or experiment; these uses are listed separately. Block subscripts and fixed arguments are occasionally suppressed in proofs, so $f , \widehat { f } , R , \widehat { R } , D , g , U , q , \delta , \delta ^ { * } , \delta ^ { U }$ , � then denote the corresponding block quantities. The index $j$ denotes a coordinate within a block, a color class, or a single-agent answer, as stated. A hat denotes an implemented approximation or an estimate, not a diferent task.

Table 2: Notation used in the paper.  
Symbol Meaning   
Shared task, local objectives and corrections   
� Number of blocks or agents.   
$^ { b }$ Block or agent index.   
$\mathcal { X } _ { b } , \mathcal { X }$ Finite block domain and product domain $\mathcal { X } _ { 1 } \times \cdots \times \mathcal { X } _ { m } .$   
$x , x _ { b } , x _ { b j }$ Shared candidate solution, its block $^ { b , }$ and coordinate $j$ of that block.   
$x _ { - b } , ( x _ { - b } , u )$ All blocks other than $b ,$ and the solution with block � replaced by $u .$   
$x ^ { 0 } , x ^ { k } , x ^ { + }$ Initial solution, solution at epoch $k ,$ and solution after the next commit.   
$u , U _ { b } , U _ { k b }$ Trial block value, returned block proposal, and the proposal in epoch �.   
$u _ { b } ^ { * } , x ^ { * }$ A true conditional block minimizer and a global minimizer.   
$F ( x ) , F ^ { * }$ Global loss $F : \mathcal { X }  \mathbb { R }$ and its minimum over $\mathcal { X } .$   
$\Delta ( x ) , \Delta _ { 0 }$ Global objective gap $\boldsymbol { F } ( \boldsymbol { x } ) - \boldsymbol { F } ^ { * }$ and its initial value $\Delta ( x ^ { 0 } ) .$   
$L _ { b } ( u )$ Available local objective $L _ { b } : \mathcal { X } _ { b }  \mathbb { R } ; L$ is its one-block example.   
$f _ { b } ( u ; x )$ Exact conditional global change $F ( x _ { - b } , u ) - F ( x ) ; f _ { b } : \mathcal { X } _ { b } \times \mathcal { X }  \mathbb { R } .$   
$D _ { b } ( x )$ Best true block decrease − min � (�; �).   
$R _ { b } ( u ; x )$ Exact anchored correction $f _ { b } ( \boldsymbol { u } ; \boldsymbol { x } ) - [ L _ { b } ( \boldsymbol { u } ) - L _ { b } ( \boldsymbol { x } _ { b } ) ] ; R _ { b } : \mathcal { X } _ { b } \times \mathcal { X }  \mathbb { R } .$   
$\widehat { R } _ { b } ( u ; x )$ Implemented approximation to the exact correction; $\widehat { R } _ { b } : { \mathcal { X } } _ { b } \times { \mathcal { X } }  \mathbb { R } .$   
$\widehat { f _ { b } } ( u ; x )$ Corrected local change $L _ { b } ( u ) - L _ { b } ( x _ { b } ) + \widehat { R } _ { b } ( u ; x ) ; \widehat { f } _ { b } : \mathcal { X } _ { b } \times \mathcal { X }  \mathbb { R } .$   
$\hat { f } _ { b } ^ { * }$ Minimum of ${ \widehat { f } } _ { b } ( \cdot ; x )$ over $\mathcal { X } _ { b } .$   
$\widehat F ( x )$ Learned global surrogate $G ( x ) + \langle \widehat \vartheta ^ { + } , \phi ( x ) \rangle$   
$M , \ell _ { a } , N ( a )$ Number of objective factors, factor $^ { a , }$ and the blocks on which it depends.   
$x _ { N \left( a \right) }$ Subvector indexed by the factor’s block set.   
$\chi ( b ) , \kappa$ Color assigned to block � and number of colors.   
$\mathcal { C } _ { j } , j ^ { k }$ Color class $\{ b : \chi ( b ) = j \}$ , and the color selected in epoch $k .$   
$g _ { b }$ Verified nonnegative gain max $\{ 0 , - f _ { b } ( U _ { b } ; x ) \}$   
$S ^ { k }$ Blocks committed in epoch �: the improving blocks of the selected color (Algorithm 1) or the best verified   
combination (Algorithm 2).   
$( x _ { - S } , U _ { S } )$ Solution � with every block $b \in S$ replaced by its proposal $U _ { b } .$   
$\Phi ( S )$ Loss $F ( x _ { - S } , U _ { S } )$ of accepting the proposals of the blocks in $S$ (Section 5).   
$\dot { I } ^ { \dot { k } }$ Improving blocks of epoch $k ,$ those with $g _ { b } > 0 .$   
$\xi _ { b b ^ { \prime } }$ Interaction of the proposals of blocks � and $b ^ { \prime } ;$ their joint change minus the sum of their single changes; they   
conflict i $\dot { \xi } _ { b b ^ { \prime } } > 0 .$   
$\kappa _ { k } ^ { * }$ Realized number of colors of the best-combination merge in epoch �: $\sum _ { b } g _ { b }$ over the realized gain (Section 5).   
$\kappa _ { k } , \deg _ { k } ( b )$ Efective number of colors, the smallest number of conflict-free sets covering $I ^ { k }$ , and the number of improving   
blocks that conflict with � (Section 5); deg(�) in the synthetic task is the degree in the dependency graph.   
�¯ Bound on $\kappa _ { k } ^ { * }$ in every epoch (Corollary 3).   
Convergence and stopping certificates   
$k , K$ Current epoch index and prescribed number of epochs.   
� Number of private search steps per epoch.   
$\mathcal { H } , \mathcal { H } _ { k }$ History conditioning a local search, or history before epoch $k ^ { \prime } \mathrm { s }$ calibration.   
$\delta _ { b } ^ { * } , \delta _ { b } ^ { U }$ Two-point correction-error bounds (Assumption 1): at a true block minimizer, and in conditional expectation at   
the returned proposal.   
$\delta _ { b }$ Uniform absolute correction-error bound over block candidates.   
$\varepsilon ( u ) , \varepsilon _ { b } ( u )$ Correction error ${ \widehat { f } } ( u ) - f ( u )$ of a candidate in the proofs, and of block � in Proposition $\mathrm { { A . 2 . } }$   
$\boldsymbol { Q _ { b } } , \boldsymbol { e } _ { b } ^ { * } , \boldsymbol { e } _ { b } ^ { U } , \boldsymbol { \widehat { D } _ { b } }$ In one recorded epoch (Proposition $_ { \mathrm { A . 2 ) : } }$ : captured fraction of the corrected improvement, correction errors at a   
true block minimizer and at the returned proposal, and the available corrected improvement.   
$\rho _ { b } , \rho , \rho _ { \mathrm { m i n } }$ Per-step local contraction parameter and its minimum across blocks.   
$q _ { b } ( H ) , q ( H )$ Captured-improvement guarantee $1 - ( 1 - \rho _ { b } ) ^ { H }$ , and its minimum over blocks.   
� Block error-bound constant in $\begin{array} { r } { \sum _ { b } D _ { b } ( x ) \ge \mu \Delta ( x ) } \end{array}$

Continued on the next page

```latex
Symbol Meaning (continued)
$\mu _ { \mathrm { f i n } }$ Capped minimum improvement-to-gap ratio on the finite domain.
$\underline { { f } } _ { b }$ Computable lower bound on min<sub>�</sub> $\widehat { f _ { b } } ( u ; x ) .$
$_ \alpha$ Descent coeficient $\mu q ( H ) / \kappa .$
$B$ Correction-error term $\overset { \cdot } { \kappa } { ^ { - 1 } } \sum _ { b } ( q _ { b } ( H ) \delta _ { b } ^ { * } + \delta _ { b } ^ { U } )$ ; at most $\kappa ^ { - 1 } \sum _ { b } ( 1 + q _ { b } ( H ) ) \delta _ { b }$ under a uniform bound.
$B _ { \delta }$ Uniform-error counterpart $\begin{array} { r } { \overline { { \kappa ^ { - 1 } } } \sum _ { b } ( 1 + q _ { b } ( \Breve { H } ) ) \delta _ { b } } \end{array}$ of � (Appendix $\mathrm { A . 2 } )$
$B / \alpha$ Error floor in the expected-gap bound.
$\Delta _ { k } , \alpha _ { k } , B _ { k }$ Gap after round � and the round’s measured coeficients in $\Delta _ { k + 1 } \leq ( 1 - \alpha _ { k } ) \Delta _ { k } + B _ { k } .$
$\zeta$ Relative-error fraction in $( 1 + q _ { b } ) \delta _ { b } \leq \zeta q _ { b } D _ { b } .$
� Requested objective-gap tolerance; an expected-gap tolerance where expectation is specified.
Learning corrections from global evaluations
$G ( x )$ Known part of the structured global objective.
$\phi ( x ) , r$ Known feature vector in $\mathbb { R } ^ { r }$ and its dimension.
$\vartheta , \widehat { \vartheta }$ Fixed unknown coeficients and their retained estimate.
${ \widehat { \vartheta } } ^ { k } , { \widehat { \vartheta } } ^ { + }$ Estimate indexed by epoch and estimate immediately after calibration.
$c _ { 0 }$ Unknown additive constant in the objective.
$\Lambda _ { b }$ Known bound on $\lVert \phi ( x _ { - b } , u ) - \phi ( x ) \rVert _ { 2 }$
$z , z _ { b }$ Feasible calibration probe and singleton proposal $( x _ { - b } , U _ { b } )$
$a , a _ { b }$ Feature displacement $\phi ( z ) - \phi ( x )$ , or that of proposal $z _ { b } .$
$y , y _ { b }$ Residual global diference after subtracting the known-part diference.
$y - \langle a , \vartheta \rangle$ Prediction residual for the current coeficient estimate.
$\dot { P }$ Measurement projector �� $^ \top / \| a \| _ { 2 } ^ { 2 } ;$ zero for a skipped or zero-length probe.
� Identity matrix on the coeficient space.
$\sigma$ Excitation constant in $\mathbb { E } [ P \mid { \mathcal { H } } _ { k } ] \subseteq \sigma I .$
� Learning contraction factor ${ \sqrt { 1 - \sigma } } .$
�,̃︀ �̃︀<sup>+</sup>, �̃︀<sup>next</sup> Parameter error ${ \widehat { \vartheta } } - \vartheta ,$ after calibration, and after extra reused measurements.
$E _ { k }$ Upper envelope of the gap along a recorded run (Proposition $\mathrm { A . 2 } )$
$\beta$ Conversion coeficient $\bar { \kappa } ^ { - 1 } \sum _ { b } ( \bar { 1 } + q _ { b } ( H ) ) \Lambda _ { b } .$
$\omega$ Potential weight $2 \beta / ( 1 - \eta ) \overline { { { \it \Omega } } }$
� Joint rate min{�, $( 1 - \eta ) / 2 \}$
$\nu _ { k }$ Joint potential $\Delta ( x ^ { k } ) + \omega \| \widehat { \vartheta ^ { k } } - \vartheta \| _ { 2 }$
Binary calibration and observable uncertainty
$d _ { b } , d _ { \mathrm { m a x } }$ Number of binary coordinates in block $^ { b , }$ and the largest block dimension.
$\Psi ( x )$ Known coupling function in the coordinate residual model.
$\vartheta _ { b j } , \dot { \vartheta _ { b j } }$ True and estimated coeficient of binary coordinate $( b , j )$
$M _ { b j }$ Known initial bound on $| \vartheta _ { b j } - \widehat { \vartheta } _ { b j } ^ { 0 } | .$
� Bit displacement: signed one-bit change $z _ { b j } - x _ { b j } \in \{ - 1 , 1 \}$
$p$ Probe probability: probability that a block with unknown coordinates performs calibration
${ \mathcal { U } } _ { b } ^ { k } , n _ { b }$ Remaining unknown coordinates before epoch �, and their number in a block.
$\Gamma _ { k } , \Gamma _ { b , k }$ Radii: sum of remaining bounds $M _ { b j }$ over all blocks, or over block �.
$d _ { \mathrm { t o t } }$ Query budget: total number of binary coordinates $\sum _ { b } d _ { b } .$
$\gamma$ Radius contraction parameter $p / d _ { \mathrm { m a x } } .$
$\beta _ { 0 }$ Coordinate-model conversion factor $\kappa ^ { - 1 } \operatorname* { m a x } _ { b } ( 1 + q _ { b } ( H ) )$
$\omega _ { 0 } , \tau _ { 0 }$ Radius weight $2 \beta _ { 0 } / \gamma$ and joint rate min $\{ \alpha , \gamma / 2 \}$
$\mathcal { W } _ { k }$ Observable-radius potential $\Delta ( x ^ { k } ) + \omega _ { 0 } \dot { \Gamma } _ { k } .$
� Value range: width of a known interval of objective values in the impossibility argument.
Private work and cost accounting
� Private cost: modeled cost of one parallel private-search slot.
� Other per-epoch cost, including synchronization and context preparation; it includes verifier calls in
Appendix A.3 and excludes them, charged through �, in Appendix $_ { \mathrm { A . 5 . } }$
� Cost of one scalar global evaluation.
$K _ { \varepsilon } ( H )$ Suficient epoch count for expected gap at most $\varepsilon .$
$T _ { \varepsilon } ( H ) , T _ { K }$ Suficient modeled completion cost and cost through � epochs.
$\lambda { \mathrm { ~ - ~ } } \log ( 1 - \rho )$ , the exponential form of the worst-block local rate.
$J ( H )$ Relaxed cost ratio $\hat { ( c H + C ) / ( 1 - e ^ { - \lambda H } ) }$
$H _ { * }$ Positive continuous minimizer of $J ( H )$ for $C > 0 .$
$\psi ( H )$ Proof helper: $e ^ { \lambda H } - 1 - \lambda H - \lambda C / \dot { c } .$
$N _ { \mathrm { q } } , K _ { \mathrm { e n d } }$ Synthetic experiments: number of counted global evaluations, and the epoch at which a run completes.
```

Continued on the next page

<table><tr><td colspan="2">Symbol Meaning (continued</td></tr><tr><td colspan="2">Robot-routing task and its offline references</td></tr><tr><td> $r _ { i } , i$ </td><td>Robot label and robot index,  $i = 0 , \ldots , 8 .$ </td></tr><tr><td> $\mathcal { T } _ { b }$ </td><td>Set of robots owned by team block b.</td></tr><tr><td> $W$ </td><td>Number of single agents searching in parallel in the best-of-W baseline, equal to the number of team agents.</td></tr><tr><td> $x _ { i } , u _ { i }$ </td><td>Routes: accepted and candidate command arrays for robot i.</td></tr><tr><td> $( x , y ) , p$ </td><td>Grid coordinates and a free cell in a reference-search state; x increases rightward and y downward.</td></tr><tr><td></td><td>t Routing tick; the fixed horizon is 48 ticks.</td></tr><tr><td> $s _ { i } , g _ { i } , p _ { i } ( t )$ </td><td>Start, goal and position of robot i at tick t.</td></tr><tr><td></td><td>τi First goal-arrival tick of a delivered robot.</td></tr><tr><td> $d _ { i } ( p , g _ { i } )$ </td><td>Shortest legal grid distance from cell  $p$  to robot  $i \mathrm { { } s }$  goal.</td></tr><tr><td> $c _ { i } ( x _ { i } )$ </td><td>Delivery cost of robot i. It equals the arrival tick for a delivered robot, otherwise  $1 0 0 + 4 8 + d _ { i } ( p _ { i } ( 4 8 ) , g _ { i } )$ </td></tr><tr><td> $C _ { i j } ( x _ { i } , x _ { j } )$   $\scriptstyle { \dot { \mathcal { P } } } ^ { k }$ </td><td>Number of simultaneous vertex and edge-swap conflicts for a robot pair.</td></tr><tr><td></td><td>External robot pairs discovered before round  $k .$ </td></tr><tr><td> $\widehat { R } _ { b } ( u ; x ^ { k } )$ </td><td>Routing correction: anchored collision-penalty change over the retained external pairs.</td></tr><tr><td> $\Delta L _ { b } , \Delta \widehat { R } _ { b }$ </td><td>Change in local loss and in the retained correction for a proposed route block.</td></tr><tr><td> $A _ { b } ^ { \mathcal { P } ^ { k } } ( u ; x ^ { k } )$ </td><td>Absolute private score using the retained external pairs.</td></tr><tr><td> $A _ { b } ^ { \mathrm { a l l } }$   $d _ { b }$ </td><td>Absolute block score using every external pair.</td></tr><tr><td></td><td>Routing bound: sum of shortest distances for robots owned by block  $^ { b . }$ </td></tr><tr><td> $A _ { b } ^ { \mathrm { C B \bar { S } } }$ </td><td>Reference score: absolute score of a collision-free delivered reference plan in the soft-penalty certificate.</td></tr><tr><td> $\underline { { A } } _ { i } , w _ { b }$ </td><td>Single-robot relaxation minimum and joint block witness program.</td></tr><tr><td> $C _ { b , \mathrm { m i s s } } ^ { k } ( u )$ </td><td>Conflicts involving undiscovered external pairs against frozen peers.</td></tr><tr><td> $\varepsilon _ { k b } ( u )$ </td><td>Realized score error  ${ \widehat { f } } _ { b } ( u ; x ^ { k } ) - f _ { b } ( u ; x ^ { k } )$ </td></tr><tr><td> $Q _ { k b } , Q _ { s , k , b }$ </td><td>Fraction of the corrected improvement captured by a returned block; the counterpart of  $q _ { b } ( H )$ </td></tr><tr><td> $\mu _ { \mathrm { o b s } } , \mu _ { s }$ </td><td>Minimum capped improvement-to-gap ratio over recorded states, or within run s.</td></tr><tr><td> $e _ { s , k , b } ^ { * } , e _ { s , k , b } ^ { U }$ </td><td>Measured correction errors at a true block minimizer and at the returned block; realized counterparts of</td></tr><tr><td> $\Delta _ { s , k }$ </td><td>Objective gap for run s after round  $k .$ </td></tr><tr><td> $\alpha _ { s , k } , B _ { s , k }$ </td><td>Round-specific descent and correction-error coefficients computed from all returned blocks.</td></tr><tr><td> $\bar { \alpha } , \bar { \boldsymbol { B } }$ </td><td>Pooled arithmetic means of the measured round coefficients.</td></tr><tr><td> $x _ { s } ^ { k } , K _ { s }$ </td><td>Accepted state at round k in run s, and that run&#x27;s stopping round.</td></tr><tr><td> $E _ { s , k }$ </td><td>Upper envelope for run  $s \mathrm { ^ { \circ } s }$  objective gap at round k.</td></tr><tr><td> $\overline { { E } } _ { k } , \bar { \alpha } _ { k } , \bar { B } _ { k }$ </td><td>Envelope propagated with round-k coefficients averaged over the runs of one half (check on the other half).</td></tr><tr><td> $\overline { { U } } _ { K }$ </td><td>Mean loss envelope: the optimum plus the mean of  $E _ { s , \mathrm { m i n } \{ K , K _ { s } \} }$  over runs.</td></tr><tr><td></td><td>s (replicate), j, k, h, b Run index; single-agent step, team round, private step and block indices in the API-seed scheme.</td></tr><tr><td> $a _ { j } , a _ { k b h }$ </td><td>Measured duration of a checkable model answer.</td></tr><tr><td> $t _ { k b } , t _ { \mathrm { t e a m } }$   $T _ { \mathrm { p a r a l l e l } }$ </td><td>Sum of a block&#x27;s checked durations in a round, and the team&#x27;s search time.</td></tr><tr><td> $T _ { \mathrm { s e q u e n t i a l } }$ </td><td>Search time of the parallel team and of the sequential replay of the same answers.</td></tr><tr><td></td><td>p Top-p: nucleus-sampling parameter, fixed to 0.95 in routing.</td></tr><tr><td colspan="2">Synthetic optimization experiment</td></tr><tr><td> $t _ { b j } , \tilde { t } _ { b j }$ </td><td>Desired target bit and private target bit</td></tr><tr><td> $a _ { b j }$ </td><td>Unary mismatch weight for bit  $( b , j )$ </td></tr><tr><td></td><td>w Neighbor-coupling weight, fixed at  $_ { 0 . 2 } .$ </td></tr><tr><td> $\pi _ { b }$ </td><td>Activity probability of block b&#x27;s randomized coordinate solver.</td></tr><tr><td> $A _ { b j } ( x )$ </td><td>Exact affine coefficient of candidate bit  $u _ { j }$  with peers frozen.</td></tr><tr><td> $h _ { b j } ( x )$ </td><td>Difference between the exact and private affine coefficients.</td></tr><tr><td></td><td></td></tr><tr><td> $\hat { A _ { b j } }$ </td><td>Private coefficient plus learned unary correction and refreshed coupling coefficient.</td></tr><tr><td> $c _ { b } , \deg ( b )$ </td><td>Total incident coupling weight w deg(b) and block degree on the path.</td></tr><tr><td> $I ( x )$ </td><td>Set of coordinates whose bits differ from the desired target.</td></tr><tr><td> $\mu _ { \mathrm { d o m } }$ </td><td>Unary-dominance certificate mir  $\boldsymbol { \mathrm { 1 } } _ { b , j } ( a _ { b j } - c _ { b } ) / ( a _ { b j } + c _ { b } )$ </td></tr><tr><td> $\widehat { D } ( x )$ </td><td>Sum of all exact affine surrogate improvements at the committed base.</td></tr><tr><td> $\Gamma ( x )$ </td><td>Sum of remaining coordinate uncertainty bounds at the current base.</td></tr><tr><td> $\overline { { \Delta } } ( x )$ </td><td>Stopping: certified gap bound  $( \widehat { D } ( x ) + \Gamma ( x ) ) / \mu _ { \mathrm { d o m } } .$ </td></tr><tr><td> $\underline { { \nu } } , s _ { b j }$ </td><td>Persistent correction bias and its random sign in the approximate-correction series.</td></tr><tr><td> $e _ { b } ^ { * } , e _ { b } ^ { U } , Q _ { b }$ </td><td>Realized correction errors at a true block minimizer and at the returned proposal, and the realized captured</td></tr><tr><td>Database experiment</td><td>fraction, in the same series.</td></tr><tr><td> $s , s _ { \mathrm { d b } }$ </td><td>Database source index and database-generation seed.</td></tr></table>

Continued on the next page

<table><tr><td></td><td>Symbol Meaning (continued)</td></tr><tr><td> $g _ { s } , h _ { s }$ </td><td>Selected left and right endpoint-group IDs.</td></tr><tr><td></td><td>Join directions: left-first and right-first traversal choices.</td></tr><tr><td> $q , q _ { s } , a _ { s }$ </td><td>Query/report index, selected predicates for source  $s ,$  and its join-direction choice.</td></tr><tr><td> $v _ { s } ( \ell , g ) , v _ { s } ( r , h )$ </td><td>Edge fan-out of a selected left or right group.</td></tr><tr><td> $J _ { q } ( a ) , a$ </td><td>Separable fan-out surrogate and vector of three join directions.</td></tr><tr><td colspan="2">Mathematical operations and summary conventions</td></tr><tr><td> $\mathbb { R } , \mathbb { R } ^ { r }$ </td><td>Real numbers and real vectors of dimension r.</td></tr><tr><td> $\langle a , a ^ { \prime } \rangle , a ^ { \top }$ </td><td>Euclidean inner product  $\sum _ { j } a _ { j } a _ { j } ^ { \prime }$  and vector or matrix transpose.</td></tr><tr><td> $\| \cdot \| _ { 2 } , \| \cdot \| _ { 2 } ^ { 2 }$ </td><td>Euclidean norm and its square.</td></tr><tr><td> $\| z - x \| _ { 1 }$ </td><td></td></tr><tr><td> $| z _ { j } - x _ { j } |$ </td><td>Sum of absolute coordinate differences and scalar absolute value.</td></tr><tr><td> $\mathbb { E } [ \breve { Z } ]$ </td><td></td></tr><tr><td> $\mathbb { E } [ Z \mid \mathcal { H } ]$ </td><td>Expectation and conditional expectation of a random quantity</td></tr><tr><td> $A \succeq B$ </td><td>Positive-semidefinite matrix order:  $A - B$  is positive semidefinite.</td></tr><tr><td> $: = , = , \leq , \geq$ </td><td>Definition, equality and scalar inequality relations.</td></tr><tr><td>min, max, sup</td><td>Minimum, maximum and supremum over the stated index or candidate set.</td></tr><tr><td> $\textstyle \sum , \times$ </td><td>Sum and multiplication or Cartesian product as indicated.</td></tr><tr><td> $\in , \subseteq , \emptyset$ </td><td>Membership, subset relation and empty set.</td></tr><tr><td> $\log , \exp , \sqrt { \cdot }$ </td><td>Natural logarithm, exponential and square root.</td></tr><tr><td></td><td>[z] Smallest integer greater than or equal to z.</td></tr><tr><td>min  $\{ \cdot \} , \operatorname* { m a x } \{ \cdot \}$ </td><td>Minimum or maximum of listed scalar quantities.</td></tr><tr><td> $\approx , O ( n )$ </td><td>Approximation and asymptotic upper-bound notation.</td></tr><tr><td> $n , \pm , \mathrm { S D }$ </td><td>, SE Sample count, displayed variation, sample standard deviation and standard error.</td></tr><tr><td> $+ , *$ </td><td>A plus superscript marks an update; a star marks an optimum. Hats mark approximations or estimates; a tilde marks the private target of the synthetic task and, in Appendix  $_ { \mathrm { A . 4 , } }$  a parameter error; bars and underlines mark</td></tr></table>

## A Further theory and examples

This appendix extends Sections 3–7 and uses their notation; the table on page 16 lists every symbol. All guarantees concern the specified objective �, not requirements that its evaluation omits, and private work means computation without communication, not confidential data. Section A.1 works through small examples of the model. Section A.2 treats a uniform correction-error bound, characterizes the block error bound, gives an optimality certificate and proves the bound along a recorded run used in Section 8.1. Section A.3 proves the private-work horizon of Section 7. Sections A.4 and A.5 develop the learning theory summarized in Section 6, with its certificate and cost. The best-combination merge and its guarantees are stated in Section 5, with a conflict-greedy variant in Appendix E.9.

## A.1 Examples

Why local improvement may not help. A module can pass its isolated tests while returning a format that another module cannot read. An agent searching only with those tests has no reason to repair the format. We call the diference between what the local and global objectives reward an objective mismatch. More local search does not necessarily make this mismatch larger; it may simply solve the wrong local problem more accurately. Checking a proposal globally can reject it, but rejection alone does not tell the agent how to search better.

The following one-decision example isolates this issue. Let $u = 0$ mean retaining an incompatible output format and � = 1 mean repairing it. Let � be the global objective and � the local objective, both minimized:

$$
F ( u ) = 1 - u , \qquad L ( u ) = u , \qquad u \in \{ 0 , 1 \} .\tag{A.1}
$$

Starting at zero, even an exact minimizer of � keeps the incompatible format, whereas � is minimized at one.

Verification therefore protects the current solution but does not resolve the cause of the failed search. To produce a useful proposal, the agent needs a diferent criterion for comparing its candidates, and the correction supplies it. At base $u = 0$ the local change from choosing one is +1, while the global change is −1. A correction of −2 to that candidate’s local change makes the corrected score prefer the repair, and the correction at the unchanged candidate is zero, as anchoring requires.

A two-block example. Let $x _ { 1 } , x _ { 2 } \in \{ 0 , 1 \}$ encode two decisions. Each local objective prefers zero, but the global objective prefers one and also penalizes disagreement:

$$
\begin{array} { c } { { L _ { 1 } ( u ) = L _ { 2 } ( u ) = u , } } \\ { { { } } } \\ { { F ( x ) = ( 1 - x _ { 1 } ) + ( 1 - x _ { 2 } ) + \frac 1 4 ( x _ { 1 } - x _ { 2 } ) ^ { 2 } . } } \end{array}\tag{A.2}
$$

Starting from $x = ( 0 , 0 )$ , the loss is 2. Changing either block to one lowers it to 1.25, although its local objective rises by 1. Thus the exact anchored correction at that candidate is −1.75: adding it to the local change gives the true change −0.75. For either block, the corrected score of leaving it unchanged is zero.

<table><tr><td>Block-1 choice Local change</td><td></td><td>Correction</td><td>Sum</td></tr><tr><td>Keep  $u = 0$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td>Choose u = 1</td><td>1</td><td>-1.75</td><td>-0.75</td></tr></table>

Section A.4 returns to it to show how objective evaluations reveal its unknown correction coeficients.

One update. The two blocks share the disagreement factor, so they have diferent colors. With exact local minimization, both propose one, with singleton gain 0.75. Either color is best. Break the tie in favor of block 1: the next solution is (1, 0) and its loss is 1.25. At this new base, changing block 2 to one improves the loss by 1.25, and the next update reaches (1, 1) with loss zero. Notice that the conditional score changed when its neighboring block changed; a correction must be evaluated at the current base.

Changes that clash. Two blocks that share a factor can each lower the loss alone and yet raise it together. On two binary blocks, let

$$
F ( x ) = ( 1 - x _ { 1 } ) + ( 1 - x _ { 2 } ) + 3 x _ { 1 } x _ { 2 } .
$$

From $x = ( 0 , 0 )$ , with loss 2, changing either block to one lowers the loss to 1, but changing both raises it to 3, like two robots that take the same shortcut and collide. The factor 3�<sub>1</sub>�<sub>2</sub> joins the two blocks, so they have diferent colors, and Algorithm 1 accepts only one of the two changes.

A single-block trap. Even a perfect correction cannot resolve a task that violates Assumption 3. For example, on two binary blocks let $F ( 0 , 0 ) = 1 , F ( 1 , 0 ) = F ( 0 , 1 ) = 2$ , and $F ( 1 , 1 ) = 0 . { \mathrm { ~ A t ~ } } ( 0 , 0 )$ , neither single-block change helps, so $D _ { 1 } + D _ { 2 } = 0$ while the global gap is 1. Joint search or acceptance of intermediate worsening is needed. The parameter � describes the task, rather than the agent or communication protocol.

## A.2 Further convergence results

A uniform correction-error bound. A bound on the correction error at every candidate,

$$
\operatorname* { s u p } _ { u \in \mathcal { X } _ { b } } | \widehat { R } _ { b } ( u ; x ) - R _ { b } ( u ; x ) | \leq \delta _ { b } ,\tag{A.3}
$$

implies Assumption 1 with $\delta _ { b } ^ { * } = \delta _ { b } ^ { U } = \delta _ { b }$ . Lemma 1 then gives

$$
\begin{array} { r } { \mathbb { E } [ g _ { b } \ | \ \mathcal { H } ] \ge q _ { b } ( H ) D _ { b } ( x ) - ( 1 + q _ { b } ( H ) ) \delta _ { b } , } \end{array}\tag{A.4}
$$

and Theorem 2 holds with $\begin{array} { r } { B = B _ { \delta } : = \kappa ^ { - 1 } \sum _ { b } ( 1 + q _ { b } ( H ) ) \delta _ { b } } \end{array}$ Such a bound must also hold for poor candidates that no reasonable search returns, so it can be far larger than the two-point constants; in routing it exceeds $1 0 ^ { 4 }$ (Appendix E.7). Lemma 1 and the first step of the proof of Theorem 2 hold epoch by epoch, conditionally on ${ \mathcal { H } } ,$ so they remain valid when $\delta _ { b }$ changes between epochs:

$$
\begin{array} { r } { \mathbb { E } [ F ( x ) - F ( x ^ { + } ) \mid \mathcal { H } ] \geq \alpha \Delta ( x ) - B _ { \delta } . } \end{array}\tag{A.5}
$$

The learning results of Sections A.4 and A.5 use them in this form.

A relative correction error. If instead $q _ { b } ( H ) \delta _ { b } ^ { * } + \delta _ { b } ^ { U } \leq$ $\zeta q _ { b } ( H ) D _ { b } ( x )$ for a fixed $\zeta < 1$ at every epoch, the proof of Theorem 2 gives the contraction factor $1 - ( 1 - \zeta ) \mu q ( H ) / \kappa$ and no floor. This relative certificate is useful analytically but is not automatically observable: $D _ { b }$ is usually unknown.

How restrictive is the block error bound? On a finite domain, existence of some positive constant has a simple characterization. The size of that constant is a separate question: it may be arbitrarily small. The following observation also explains when a computable stopping certificate is available.

Proposition A.1 (Block geometry and an optimality certificate). On the full finite product domain, a constant $\mu \in ( 0 , 1 ]$ satisfying (12) exists if and only if every point with $D _ { b } ( x ) = 0$ for all � is globally optimal. When this local-to-global property holds and there are suboptimal points, one valid choice is

$$
\mu _ { \mathrm { f i n } } : = \operatorname* { m i n } \left\{ 1 , \operatorname* { m i n } _ { x : \Delta ( x ) > 0 } \frac { \sum _ { b } D _ { b } ( x ) } { \Delta ( x ) } \right\} .\tag{A.6}
$$

If every point is optimal, take $\mu _ { \mathrm { f i n } } = 1$ . Moreover, at any base where a known $\mu > 0$ satisfies (12), exact block improvements give

$$
\Delta ( x ) \stackrel { ( 1 2 ) } { \leq } \frac { 1 } { \mu } \sum _ { b } D _ { b } ( x ) .\tag{A.7}
$$

More generally, suppose certified bounds $\delta _ { b }$ satisfy (A.3) and computable lower bounds $\underline { { f } } _ { b } \le \operatorname* { m i n } _ { u } \widehat { f } _ { b } ( u ; x )$ are available. Then

$$
\Delta ( x ) \overset { ^ { ( 1 2 ) , ( \mathrm { A . 3 } ) , ( 3 ) , ( 4 ) } } { \leq } \frac { 1 } { \mu } \sum _ { b } ( \delta _ { b } - \underline { { { f } } } _ { b } ) .\tag{A.8}
$$

Proof idea. Finiteness turns strictly positive improvement ratios into a positive minimum. For certification, the error bound converts upper bounds on exact block improvements into an upper bound on the global gap.

Proof. If (12) holds and all $D _ { b } ( x )$ vanish, then $\Delta ( x ) = 0$ Conversely, the stated local-to-global property makes every ratio in (A.6) positive; the nonempty set of ratios is finite, so its minimum is positive. At a global optimum every $D _ { b }$ is zero, and hence the resulting constant satisfies (12) everywhere. For the approximate certificate, uniform error gives

$$
\begin{array} { c } { \displaystyle { \operatorname* { m i n } _ { u } f _ { b } ( u ; x ) \sum _ { \hphantom { ( } } ^ { ( \mathrm { A . 3 } ) , ( 3 ) , ( 4 ) } \operatorname* { m i n } _ { u } \widehat { f _ { b } } ( u ; x ) - \delta _ { b } \geq \underline { { { f } } } _ { b } - \delta _ { b } , } } \\ { \displaystyle { D _ { b } ( x ) \sum _ { \hphantom { ( } } ^ { ( 1 1 ) , ( \mathrm { A . 3 } ) , ( 3 ) , ( 4 ) } \delta _ { b } - \underline { { { f } } } _ { b } . } } \end{array}\tag{A.9}
$$

Combining these bounds yields

$$
\Delta ( x ) \stackrel { ( 1 2 ) } { \leq } \frac { 1 } { \mu } \sum _ { b } D _ { b } ( x ) \stackrel { ( \mathrm { A . 9 } ) } { \leq } \frac { 1 } { \mu } \sum _ { b } ( \delta _ { b } - \underline { { { f } } } _ { b } ) ,
$$

which proves $\left( \mathrm { A } . 8 \right)$ . The same bound holds with $\delta _ { b }$ replaced by a bound $\delta _ { b } ^ { * }$ on the error at a true block minimizer. □

Neither enumerating (A.6) nor solving all block subproblems is automatically cheap. Approximate proposals give lower bounds on available improvement, whereas stopping needs upper bounds. An exact block solve or a certified lower bound on each surrogate minimum supplies the missing direction. A known valid $\mu$ is also required. Thus the generic local solver does not automatically provide a stopping test, but the theory does permit one when this additional information is available.

Local-to-global results in discrete convex analysis depend on the allowed neighborhood: coordinated subset moves or exchanges may be essential (Murota, 2003). Merely calling an objective discrete convex does not establish (12) for the fixed disjoint blocks used here.

A bound along a recorded run. The argument of Lemma 1 also applies to a single realized epoch, with realized local progress in place of $q _ { b } ( H )$ and realized two-point errors in place of $\delta _ { b } ^ { * } , \delta _ { b } ^ { U }$

Proposition A.2 (Bound along a recorded run). Fix a realized epoch � with base $x = x ^ { k }$ and $\Delta ( x ) > 0 .$ . Suppose that every returned block scores no worse than the current block, that is, $\widehat { f _ { b } } ( U _ { b } ; x ) \le 0$ . Write

$$
\varepsilon _ { b } ( u ) : = \widehat { f } _ { b } ( u ; x ) - f _ { b } ( u ; x ) , \qquad \widehat { D } _ { b } : = - \operatorname* { m i n } _ { u } \widehat { f } _ { b } ( u ; x ) .
$$

Let $Q _ { b } : = - { \widehat { f } } _ { b } ( U _ { b } ; x ) / { \widehat { D } } _ { b } \ \in \ [ 0 , 1 ] \ i f \ \widehat { D } _ { b } > \ 0$ and $Q _ { b } : = 0$ otherwise, and let $e _ { b } ^ { * } : = | \varepsilon _ { b } ( u _ { b } ^ { * } ) |$ for a true block minimizer $\boldsymbol { u } _ { b } ^ { * }$ and $e _ { b } ^ { U } : = | \varepsilon _ { b } ( U _ { b } ) |$ |. Then the merge of Algorithm 1 gives

$$
\Delta ( x ^ { + } ) \leq \operatorname* { m i n } \{ \Delta ( x ) , ( 1 - \alpha _ { k } ) \Delta ( x ) + B _ { k } \} ,\tag{A.10}
$$

where

$$
\alpha _ { k } : = \operatorname* { m i n } \biggl \{ 1 , \frac { \sum _ { b } Q _ { b } D _ { b } ( x ) } { \kappa \Delta ( x ) } \biggr \} , \qquad B _ { k } : = \frac { 1 } { \kappa } \sum _ { b } ( Q _ { b } e _ { b } ^ { * } + e _ { b } ^ { U } ) .
$$

Consequently, if every epoch with $\Delta ( x ^ { k } ) > 0$ satisfies these hypotheses, and with $\alpha _ { k } : = B _ { k } : = 0$ once $\Delta ( x ^ { k } ) = 0$ , which then persists because the objective never increases, the sequence $E _ { 0 } = \Delta ( x ^ { 0 } ) , E _ { k + 1 } =$ min $\{ E _ { k } , ( 1 - \alpha _ { k } ) E _ { k } + B _ { k } \}$ satisfies $\Delta ( x ^ { k } ) \leq E _ { k }$ for every �.

Proof. If $\widehat { D } _ { b } \ > \ 0$ , then $- \widehat { D } _ { b } \leq \widehat { f } _ { b } ( u _ { b } ^ { \ast } ; x ) = - D _ { b } ( x ) +$ $\varepsilon _ { b } ( u _ { b } ^ { * } ) , \mathrm { { s o } } - \widehat { f } _ { b } ( U _ { b } ; x ) = Q _ { b } \widehat { D } _ { b } \geq Q _ { b } ( D _ { b } ( x ) - e _ { b } ^ { * } )$ . Hence

$$
\begin{array} { c } { { g _ { b } \geq - f _ { b } ( U _ { b } ; x ) = - \widehat { f } _ { b } ( U _ { b } ; x ) + \varepsilon _ { b } ( U _ { b } ) } } \\ { { \geq Q _ { b } D _ { b } ( x ) - Q _ { b } e _ { b } ^ { * } - e _ { b } ^ { U } . } } \end{array}
$$

If $\widehat { \cal D } _ { b } = 0 .$ , the right-hand side is $- e _ { b } ^ { U } \leq 0 \leq g _ { b }$ . Summing over blocks and using (8), $\begin{array} { r } { \Delta ( x ) - \Delta ( x ^ { + } ) \geq \kappa ^ { - 1 } \sum _ { b } g _ { b } \geq } \end{array}$ $\alpha _ { k } \Delta ( x ) - B _ { k } ;$ capping $\alpha _ { k }$ at one only weakens this inequality, and $\Delta ( x ^ { + } ) \leq \Delta ( x )$ because the objective never increases. Since $1 - \alpha _ { k } \ge 0$ , induction on (A.10) gives $\Delta ( x ^ { k } ) \leq E _ { k }$ □

The proposition uses only quantities realized in one run, so it holds on every recorded trajectory without assumptions on the local solver beyond returning a block that scores no worse than the current one. Under the best-combination merge, � can be replaced by the realized number of colors $\kappa _ { k } ^ { * }$ (Corollary 3), and, with pairwise factors and all combinations checked, by the efective number of colors $\kappa _ { k }$ (Proposition $^ { 4 , }$ Appendix E.9). Appendix E.8 evaluates it on the recorded LLM runs.

## A.3 How much private work is worth doing

Section 7 charges at most $c > 0$ for one parallel private step, which lasts as long as its slowest agent, and at most $C \geq 0$ for each synchronization: correction construction, certification, message transfer, loading, singleton verification and applying the selected edits. These bounds are uniform, and � does not grow with �. With exact corrections, let $T _ { \varepsilon } ( H )$ denote the modeled cost of running enough epochs to make the expected gap at most �. For $0 < \varepsilon < \Delta ( x ^ { 0 } )$ ),

$$
\begin{array} { c } { { \displaystyle { K _ { \varepsilon } ( H ) : = \left\lceil \frac { \kappa } { \mu q ( H ) } \log \frac { \Delta ( x ^ { 0 } ) } { \varepsilon } \right\rceil } , } } \\ { { \displaystyle { T _ { \varepsilon } ( H ) \leq ( c H + C ) K _ { \varepsilon } ( H ) . } } } \end{array}\tag{A.11}
$$

Indeed, using $1 - t \leq e ^ { - t }$

$$
\begin{array} { r c l } { \mathbb { E } \Delta ( x ^ { K _ { \varepsilon } } ) } & { \stackrel { ( 1 5 ) } { \leq } } & { ( 1 - \alpha ) ^ { K _ { \varepsilon } } \Delta ( x ^ { 0 } ) } \\ & { \leq } & { e ^ { - \alpha K _ { \varepsilon } } { \Delta ( x ^ { 0 } ) } \stackrel { ( 1 4 ) , ( \mathrm { A . 1 1 } ) } { \leq } \varepsilon . } \end{array}
$$

If correction size or evaluation cost grows with the horizon, one must replace � by that actual bound before optimizing. Token work also grows with the total number of agents private steps.

Let $\rho : = \operatorname* { m i n } _ { b } \rho _ { b }$ and $\begin{array} { r } { \lambda : = - \log ( 1 - \rho ) > 0 , } \end{array}$ . Dropping the outer ceiling in (A.11) and allowing a real-valued horizon yields the objective

$$
J ( H ) : = \frac { c H + C } { 1 - e ^ { - \lambda H } } , \qquad H > 0 .\tag{A.12}
$$

Proposition $\mathrm { A . 3 \ ( A }$ finite useful private-work horizon). For $C > 0$ , � has a unique positive minimizer $H _ { * } ,$ characterized by (20). Over integer $H \geq 1$ , it sufices to compare the neighboring integers around $H _ { * }$ , clamped to one. For $C = 0 _ { \cdot }$ , the integer optimum is $H = 1$ . These optimize the relaxed bound, not necessarily the ceiled bound or actual runtime.

Proof idea. The derivative has the sign of a single strictly increasing function. Its unique zero balances the extra cost of one more private step against the additional improvement that step buys.

Proof. Diferentiating (A.12) gives

$$
J ^ { \prime } ( H ) \stackrel { ( \mathrm { A . 1 2 } ) } { = } \frac { c [ 1 - e ^ { - \lambda H } ] - \lambda ( c H + C ) e ^ { - \lambda H } } { ( 1 - e ^ { - \lambda H } ) ^ { 2 } } .\tag{A.13}
$$

Its sign is the sign of $\psi ( H ) = e ^ { \lambda H } - 1 - \lambda H - \lambda C / c$ . For $C > 0 , \psi ( 0 ) < 0 , \psi ^ { \prime } ( H ) = \lambda ( e ^ { \lambda H } - 1 ) > 0$ for $H > 0$ , and $\psi ( H )$ tends to infinity. Hence there is exactly one zero, with � decreasing before and increasing after it. This proves both the continuous and integer claims. For $C = 0 , e ^ { t } > 1 + t$ for $t > 0$ gives $J ^ { \prime } ( H ) > 0$ □

As � increases, $q ( H )$ approaches one: the analysis interpolates between inexact and exact block minimization. It supplies a batching calculation, not ProxSkip-type communication acceleration.

## A.4 Learning the correction

A global objective evaluation can teach the correction if it reveals something reusable. We obtain this property by assuming that the unknown part of the objective has fixed coeficients multiplying known features. Evaluations then give linear equations for those coeficients. Learning them once can help evaluate many later candidates, even as the solution changes.

In the next assumption, $G ( x )$ is the known part of the objective. The vector $\phi ( x )$ contains � known numerical features, and the unknown vector � gives their weights. A probe is a feasible candidate evaluated to learn these weights; it need not become the team’s next solution. We call these learning evaluations calibration.

Assumption A.1 (Static identifiable residual). For a positive integer �, there are a known function �, known features $\phi : \mathcal { X }  \mathbb { R } ^ { r }$ , and an unknown fixed vector $\vartheta \in \mathbb { R } ^ { r }$ such that

$$
F ( x ) = G ( x ) + \langle \vartheta , \phi ( x ) \rangle + c _ { 0 } .\tag{A.14}
$$

The unknown constant $c _ { 0 }$ cancels in diferences. Known terms and features can be evaluated locally after context loading; their costs are charged. Exact scalar verifier queries are permitted on feasible speculative solutions. Feature changes obey known bounds

$$
\| \phi ( x _ { - b } , u ) - \phi ( x ) \| _ { 2 } \leq \Lambda _ { b } .\tag{A.15}
$$

The representation helps only if � and the cost of evaluating � and $\phi$ are small. The unknown residual here means $F -$ $G - c _ { 0 } ;$ the measurement residual introduced next instead means an observed value minus its prediction.

An objective diference gives a linear measurement. At base $x ,$ query a feasible probe � and compute

$$
\begin{array} { c } { { a : = \phi ( z ) - \phi ( x ) , } } \\ { { \ } } \\ { { y : = F ( z ) - F ( x ) - [ G ( z ) - G ( x ) ] } } \\ { { \overset { \mathrm { ( A . 1 4 ) ~ } } { = } \langle a , \vartheta \rangle . } } \end{array}\tag{A.16}
$$

Let $\widehat { \vartheta }$ denote the current estimate of the unknown weights. Its predicted measurement is $\langle a , { \widehat { \vartheta } } \rangle$ , so $y - \langle a , \widehat { \vartheta } \rangle$ is the prediction error, or residual. For $a \neq 0 ,$ , the following update moves the estimate to the nearest vector consistent with the observed measurement. The superscript + denotes the updated estimate, which is retained for later epochs:

$$
\widehat { \vartheta } ^ { + } : = \widehat { \vartheta } + \frac { y - \langle a , \widehat { \vartheta } \rangle } { \| a \| _ { 2 } ^ { 2 } } a .\tag{A.17}
$$

For intuition, return to the one-bit example and take the known part to be $G ( u ) = L ( u ) = u$ and the feature to be $\phi ( u ) = u$ . The unknown coeficient is $\vartheta = - 2$ and the constant is $c _ { 0 } = 1$ . Evaluating the change from zero to one gives $a = 1$ and $y = - 2 \colon$ the global change is −1 and the known local change is +1. Starting from an estimate of zero, the update sets it $\mathrm { t o ~ } - 2$ . The corrected local objective then prefers one. This example requires only one measurement because it has one unknown coeficient; larger models need enough informative directions.

Learning the two-block example. The task (A.2) has known part and features

$$
\begin{array} { c } { { G ( x ) = x _ { 1 } + x _ { 2 } + \frac 1 4 ( x _ { 1 } - x _ { 2 } ) ^ { 2 } , \quad \phi ( x ) = ( x _ { 1 } , x _ { 2 } ) , } } \\ { { { } } } \\ { { \vartheta = ( - 2 , - 2 ) , \qquad c _ { 0 } = 2 . } } \end{array}\tag{A.18}
$$

The algorithm knows � and $\phi ,$ but must learn �. Start with $\widehat { \vartheta } = ( 0 , 0 ) \mathrm { ~ a t ~ } x = ( 0 , 0 )$ . Evaluating the probe $z = ( 1 , 0 )$ gives global change −0.75 and known-part change 1.25. Their diference is $y = - 2$ , while $a = ( 1 , 0 )$ . Update (A.17) therefore sets the first estimated coeficient $\mathrm { t o } - 2$ and leaves the second at zero. The probe is a measurement; it does not itself change the shared solution.

With that estimate, block 1’s corrected candidate now has score −0.75, so local search can propose it and verification can accept it. After this accepted update, the base is (1, 0). A probe at (1, 1) gives global change −1.25, known-part change 0.75, and hence the same unknown coeficient value −2 for block 2. Its exact correction is now −2.25, rather than the value −1.75 at the original base. The coeficient is static; the complete correction function is not. Known coupling terms must still be recomputed at the current context.

This two-query sequence illustrates coeficient reuse. It is an explanatory sequence; a general rate needs the excitation assumption below.

The update is the established Kaczmarz/sketch-andproject estimator (Strohmer and Vershynin, 2009; Gower and Richtárik, 2015). Ordinary verified edits can supply such measurements too, but they need not explore all parameter directions. We therefore permit deliberate calibration probes that are evaluated without being committed.

Measurements must reveal every unknown direction, not repeatedly test the same one. The next assumption expresses this coverage, often called excitation. Here � is the identity matrix, ⪰ is the positive-semidefinite matrix order, and $\mathcal { H } _ { k }$ is the history before epoch �’s calibration. A larger � means a stronger guaranteed coverage of parameter directions.

Assumption A.2 (Exciting calibration). Conditionally on all past information before calibration, the measurement projector $P = a a ^ { \top } / \| a \| _ { 2 } ^ { 2 }$ satisfies

$$
\begin{array} { r } { \mathbb { E } [ P \mid \mathcal { H } _ { k } ] \succeq \sigma I , \sigma \in ( 0 , 1 / r ] . } \end{array}\tag{A.19}
$$

A skipped probe has $P = 0$ and leaves the estimate unchanged; zero-length measurements are skipped. Their probability is included in �.

For example, uniformly sampling one of the � coordinate directions gives $\mathbb { E } [ P ] = I / r , \mathrm { s o } \sigma = 1 / r$ is valid. Taking traces also shows that a single rank-one probe per epoch cannot give $\sigma > 1 / r$ . Thus the amount and diversity of measurement information matter. This assumption is a property of an implementable probe distribution, not a consequence of successful task edits. The coordinate specialization below instead samples the remaining uncertain coordinates and uses an observable-radius analysis.

From the estimate to private work. After calibration, replace the unknown vector in the exact correction by its

estimate:

$$
\begin{array} { c } { { \widehat { R } _ { b } ( u ; x ) : = G ( x _ { - b } , u ) - G ( x ) } } \\ { { - \left[ L _ { b } ( u ) - L _ { b } ( x _ { b } ) \right] + \langle \widehat { \vartheta } ^ { + } , \phi ( x _ { - b } , u ) - \phi ( x ) \rangle . } } \end{array}\tag{A.20}
$$

Substituting into the surrogate makes the implemented local score explicit:

$$
\begin{array} { r l } & { \widehat { f } _ { b } ( u ; x ) \overset { ( 4 ) , ( \mathtt { A . 2 0 } ) } { = } G ( x _ { - b } , u ) - G ( x ) } \\ & { ~ + \langle \widehat { \vartheta } ^ { + } , \phi ( x _ { - b } , u ) - \phi ( x ) \rangle . } \end{array}\tag{A.21}
$$

Thus the implemented method is block descent on the learned global surrogate $\widehat { F } ( x ) = G ( x ) + \langle \widehat { \vartheta } ^ { + } , \phi ( x ) \rangle$ . The local objec tive cancels algebraically; in this representation it supplies a modeling decomposition, not an extra source of optimization power. This exposes the actual computational requirement: after receiving the needed context and coeficients, the agent must evaluate the known terms and features cheaply. The saving comes from this representation and reuse, not from subtracting and adding �<sub>�</sub>. A coeficient vector whose evaluation is as expensive as the verifier would not solve the cost problem.

Then run the same �-step private solver, verify singleton gains, and merge the best color. Write ${ \widehat { \vartheta } } = { \widehat { \vartheta } } ^ { k }$ for the estimate before the calibration of epoch � and $\widehat { \vartheta } ^ { + } = \widehat { \vartheta } ^ { k + 1 }$ for the estimate after it. With $\widetilde { \vartheta } : = \widehat { \vartheta } - \vartheta$ and $\widetilde { \vartheta } ^ { + } : = \widehat { \vartheta } ^ { + } - \vartheta$ , the surrogate error is bounded by

$$
\operatorname* { s u p } _ { u } | \widehat { R } _ { b } ( u ; x ) - R _ { b } ( u ; x ) | \overset { ( 3 ) , ( \mathrm { A . 1 4 } ) , ( \mathrm { A . 1 5 } ) , ( \mathrm { A . 2 0 } ) } { \leq } \Lambda _ { b } \| \widetilde { \vartheta } ^ { + } \| _ { 2 } .\tag{A.22}
$$

The unknown error norm is only used in analysis. Section A.5 supplies an observable certificate for coordinate residuals.

Lemma A.4 (Learning contracts a persistent error). Under Assumptions A.1 and A.2, let $\eta : = \sqrt { 1 - \sigma } < 1$ Then

$$
\mathbb { E } [ \lVert \widetilde { \vartheta } ^ { + } \rVert _ { 2 } \mid \mathcal { H } _ { k } ] \leq \eta \lVert \widetilde { \vartheta } \rVert _ { 2 } .\tag{A.23}
$$

Proof idea. The update removes precisely the component of coeficient error along the measured direction. The excitation assumption ensures that, on average, every direction loses a definite fraction of its squared error.

Proof. An active update is an orthogonal projection of the error:

$$
\begin{array} { r l } & { \widetilde { \vartheta } ^ { + } \stackrel { ( \mathrm { A . 1 6 } ) , ( \mathrm { A . 1 7 } ) } { = } ( I - P ) \widetilde { \vartheta } , } \\ & { \qquad \quad \| \widetilde { \vartheta } ^ { + } \| _ { 2 } ^ { 2 } \stackrel { ( \mathrm { A . 1 6 } ) , ( \mathrm { A . 1 7 } ) } { = } \| \widetilde { \vartheta } \| _ { 2 } ^ { 2 } - \widetilde { \vartheta } ^ { \top } P \widetilde { \vartheta } . } \end{array}\tag{A.24}
$$

The same identities hold for a skipped update with $P = 0$ Conditioning and applying the excitation bound gives

$$
\begin{array} { r } { \mathbb { E } [ \| \widetilde { \vartheta } ^ { + } \| _ { 2 } ^ { 2 } \ | \ \mathcal { H } _ { k } ] \overset { ( \mathrm { A . 2 4 } ) , ( \mathrm { A . 1 9 } ) } { \leq } ( 1 - \sigma ) \| \widetilde { \vartheta } \| _ { 2 } ^ { 2 } . } \end{array}\tag{A.25}
$$

Jensen’s inequality for the square root proves (A.23). Since � is fixed, committing a new solution does not change the parameter error. This persistence is the crucial benefit of the representation. □

We can now track both processes together. Searching with an inaccurate correction leaves an additional task-error term proportional to the parameter error; calibration reduces that parameter error. Weighting the two errors appropriately gives one quantity that decreases geometrically. This quantity is an analysis tool: it contains the unknown optimum and unknown true weights, so the algorithm cannot use it as a stopping test.

What is coupled to what? The potential $\nu _ { k }$ below adds � times the coeficient error $\| \widehat { \vartheta } ^ { k } - \vartheta \| _ { 2 }$ to the task gap. Calibration reduces its conditional expectation by a factor $\eta < 1$ Meanwhile, optimization reduces task error by a factor $1 - \alpha$ but adds a penalty proportional to the coeficient error remaining after calibration. The proof combines these two statements. The coeficient $\beta$ converts parameter error into objective-error units; the weight � makes their weighted sum decrease. A potential is simply this weighted quantity used to track the analysis, not an additional objective that an agent minimizes.

Theorem A.5 (A joint task–correction contraction). Assume known graph structure, Assumption 2 for every learned surrogate, Assumption 3, and Assumptions $A . I -$ A.2. Set

$$
\begin{array} { c } { { \alpha : = \mu q ( H ) / \kappa , \quad \beta : = \kappa ^ { - 1 } \displaystyle \sum _ { b } ( 1 + q _ { b } ( H ) ) \Lambda _ { b } > 0 , } } \\ { { \nonumber } } \\ { { \omega : = 2 \beta / ( 1 - \eta ) , \quad \tau : = \operatorname* { m i n } \{ \alpha , ( 1 - \eta ) / 2 \} , } } \\ { { \gamma _ { k } : = \Delta ( x ^ { k } ) + \omega \| \widehat { \vartheta } ^ { k } - \vartheta \| _ { 2 } . } } \end{array}\tag{A.26}
$$

With deterministic initialization,

$$
\begin{array} { r } { \mathbb { E } \mathcal { V } _ { K } \leq ( 1 - \tau ) ^ { K } \mathcal { V } _ { 0 } . } \end{array}\tag{A.27}
$$

Thus solution error and correction error vanish together; no externally supplied shrinking-error schedule is needed.

Proof idea. Optimization contracts the task error but pays a penalty proportional to the remaining coeficient error. Calibration contracts that coeficient error. The weight � is chosen so that the latter decrease pays for the former penalty.

Proof. Condition first on the calibration outcome. The drift bound (A.5) applies with the current errors bounded by (A.22). Averaging over calibration yields

$$
\begin{array} { r l } & { \mathbb { E } [ \Delta ( x ^ { k + 1 } ) \mid \mathcal { H } _ { k } ] } \\ { \ } & { } \\ { ( \mathrm { A . 5 } ) , ( \mathrm { A . 2 2 } ) , ( \mathrm { A . 2 6 } ) } \\ { \ } & { \leq \ ( 1 - \alpha ) \Delta ( x ^ { k } ) + \beta \mathbb { E } [ \| \widetilde { \vartheta } ^ { + } \| _ { 2 } \mid \mathcal { H } _ { k } ] . } \end{array}\tag{A.28}
$$

Adding the weighted learning error gives

$$
\begin{array}{c} \begin{array} { r l } & { ~ \mathbb { E } [ \mathcal { V } _ { k + 1 } \mid \mathcal { H } _ { k } ] } \\ & { ( \mathrm { A . 2 8 } ) , ( \mathrm { A . 2 3 } ) } \\ & { \leq } \\ & { ( \mathrm { A . 2 6 } ) } \\ & { \leq } \end{array} \textstyle ( 1 - \alpha ) \Delta ( x ^ { k } ) + ( \beta + \omega ) \eta \| \widetilde { \vartheta } \| _ { 2 }  \\ & { ~ \leq \textstyle ( 1 - \tau ) \mathcal { V } _ { k } . } \end{array}\tag{A.29}
$$

For the second step, $\tau \leq \alpha$ and $( \beta + \omega ) \eta \leq \omega [ 1 - ( 1 - \eta ) / 2 ] ;$ the latter follows because the diference equals $\beta ( 1 - \eta ) \geq 0 .$ Taking expectations and induction prove (A.27). □

In this theorem, Lemma 1 is applied epoch by epoch, after conditioning on the calibration result. Its error bound may therefore change from epoch to epoch; no fixed nonzero error floor is imposed. The rate � is controlled by the slower of two processes: optimizing the solution with the current correction, and identifying the correction itself. The method retains an estimate of fixed coeficients, not a permanently valid table of candidate values: after � changes, known features and coupling terms must still be evaluated at the new context. This is the precise sense in which the correction is reusable.

Can already-paid verification teach the correction? Yes. Every proposal verified in line 6 of Algorithm 1, including a rejected one, supplies a measurement at the common base. With ${ z } _ { b } = ( x _ { - b } , U _ { b } )$ , define

$$
\begin{array} { c } { a _ { b } : = \phi ( z _ { b } ) - \phi ( x ) , } \\ { y _ { b } : = F ( z _ { b } ) - F ( x ) - [ G ( z _ { b } ) - G ( x ) ] \stackrel { { \scriptscriptstyle ( \mathrm { A . 1 4 } ) } } { = } \langle a _ { b } , \vartheta \rangle . } \end{array}\tag{A.30}
$$

One can apply (A.17) to these measurements after all proposals and merge decisions have been fixed, retaining the resulting estimate for the next epoch. Each equation must use the base at which its two objective values were measured. Reuse needs no additional verifier call; feature evaluation, projection work, and communicating the updated estimate still cost work.

Proposition A.6 (Safe reuse of proposal measurements). Under the hypotheses of Theorem A.5, keep its deliberate calibration and add any finite sequence of exact projection updates from the already-evaluated proposals, after the current search and merge decision. The bound (A.27) remains valid for the estimate after these extra updates. No excitation or independence assumption is needed for the extra measurements.

Proof idea. Extra projections cannot increase parameter error. They do not change the proposals used to prove this epoch’s task-error bound.

Proof. Let ${ \widetilde { \vartheta } } ^ { + }$ be the error just after deliberate calibration, and let $\widetilde { \vartheta } ^ { \mathrm { n e x t } }$ be its value after all proposal measurements have been processed. Every nonzero measurement gives an orthogonal projection; a zero measurement is skipped. Repeated use of the pathwise identity (A.24) gives

$$
\begin{array} { r l r } & { } & { \| \widetilde { { \vartheta } } ^ { \mathrm { n e x t } } \| _ { 2 } \begin{array} { c } { ( \mathrm { A } . 2 4 ) } \\ { \le } \end{array} \| \widetilde { { \vartheta } } ^ { + } \| _ { 2 } . } \end{array}\tag{A.31}
$$

The task update still obeys (A.28). Consequently

$$
\begin{array} { r l } & { ( \mathbb { A } . 2 8 ) , ( \mathbb { A } . 3 1 ) , ( \mathbb { A } . 2 3 ) } \\ & { \qquad \leq } \\ & { \qquad ( \mathbb { A } . 2 6 ) \qquad ( 1 - \tau ) \mathbb { A } ( x ^ { k } ) + ( \beta + \omega ) \eta \| \widetilde { \vartheta } \| _ { 2 } } \\ & { \qquad \leq } \end{array}
$$

Induction proves the claim. The next epoch’s hypotheses must continue to hold for the augmented history, as required by their uniform conditional formulation. □

This extension justifies learning from verification, but does not justify removing deliberate calibration: proposals can all have zero measurement vectors or fail to explore an unknown direction.

The joint theorem shows that learning and optimization can converge together. Yet its estimation error involves the unknown true coeficients, so the algorithm cannot read that error from its current state. Nor does a rate in epochs alone reveal the price of the measurements. A more concrete specialization lets us address both issues.

## A.5 Certifying and paying for learning

For binary blocks with unknown individual-bit penalties, each targeted objective evaluation identifies one coeficient exactly. This gives both a computable bound on the remaining correction error and a finite total number of learning queries. We now derive those guarantees. Unlike the preceding parameter-error norm, the remaining uncertainty can be tracked without knowing the true weights. For this specialization, assume $\mathcal { X } _ { b } = \{ 0 , 1 \} ^ { d _ { b } }$ with positive integer $d _ { b }$ . Thus every one-bit calibration probe is feasible; a constrained subset of the binary cube need not have this property. Let Ψ be a known function describing interactions between blocks. Assume all remaining unknown requirements are linear penalties on individual bits:

$$
F ( x ) = \sum _ { b } L _ { b } ( x _ { b } ) + \Psi ( x ) + \sum _ { b } \langle \vartheta _ { b } , x _ { b } \rangle + c _ { 0 } .\tag{A.32}
$$

The unknown � cancels in diferences. Assume known coordinate bounds $| \vartheta _ { b j } - \widehat { \vartheta } _ { b j } ^ { 0 } | \leq M _ { b j }$ . A probe flips bit � of block � at the current base, producing � and $s = z _ { b j } - x _ { b j } \in \{ - 1 , 1 \}$ Its residual is

$$
\begin{array} { r l } & { y : = F ( z ) - F ( x ) - [ L _ { b } ( z _ { b } ) - L _ { b } ( x _ { b } ) ] } \\ & { \qquad - \left[ \Psi ( z ) - \Psi ( x ) \right] \stackrel { ( \mathrm { A . 3 2 } ) } { = } s \vartheta _ { b j } , } \\ & { \widehat { \vartheta } _ { b j } ^ { + } : = \widehat { \vartheta } _ { b j } + ( y - s \widehat { \vartheta } _ { b j } ) / s . } \end{array}\tag{A.33}
$$

This identifies one coeficient exactly from one extra scalar query when $F ( x )$ is cached. Keep it forever. Conditionally on the current history, each block with unobserved coeficients probes with probability $p \in ( 0 , 1 ]$ , choosing uniformly among its unobserved coordinates. Independence between blocks is not required. Probes are not committed.

Let $\mathcal { U } _ { b } ^ { k }$ be the remaining unobserved coordinates before epoch �. Until identified, each coeficient estimate is kept at its initial value and retains its error bound $M _ { b j }$ , while a measured coeficient has zero error. Summing these remaining bounds gives a computable uncertainty radius:

$$
\Gamma _ { b , k } : = \sum _ { j \in \mathcal { U } _ { b } ^ { k } } M _ { b j } , \qquad \Gamma _ { k } : = \sum _ { b } \Gamma _ { b , k } , \qquad d _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { b } d _ { b } .\tag{A.34}
$$

This sampling rule stops probing already identified coefficients, so it need not satisfy uniform excitation over all original parameter directions. Its analysis instead uses the uncertainty remaining in $\Gamma _ { k }$ . Known coupling terms are recomputed at the current context; only the static unknown coeficients are retained. The anchored correction error for block � is at most $\Gamma _ { b , k + 1 }$ , its radius after this epoch’s probes, because each binary coordinate displacement has absolute value at most one.

Proposition A.7 (Observable uncertainty and bounded calibration). Write $\gamma : = p / d _ { \mathrm { { m a x } } }$ . The above calibration satisfies

$$
\mathbb { E } [ \Gamma _ { k + 1 } \mid \mathcal { H } _ { k } ] \leq ( 1 - \gamma ) \Gamma _ { k } .\tag{A.35}
$$

There are at most $\begin{array} { r } { d _ { \mathrm { t o t } } : = \sum _ { b } d _ { b } } \end{array}$ additional calibration queries over the entire run. After $K$ epochs their expected number is at most min $\{ d _ { \mathrm { t o t } } , m p K \}$

Proof idea. Each successful probe removes one previously unmeasured coeficient from the uncertainty sum. Bounding the expected removed uncertainty gives contraction; counting distinct coeficients gives the finite query budget.

Proof. If a block has $n _ { b } ~ > ~ 0$ unobserved coeficients and remaining radius $\Gamma _ { b , k ; }$ , its expected removed radius is $p \Gamma _ { b , k } / n _ { b } \geq p \Gamma _ { b , k } / d _ { \mathrm { m a x } }$ . Empty sets contribute zero. Summing gives $( \mathrm { A . 3 5 } ) \mathrm { { ; } }$ ; solution updates do not change the learned static parameters or their uncertainty. Every calibration identifies a previously unobserved coeficient, so its count is at most $d _ { \mathrm { t o t } } . \mathrm { A t }$ most �� queries occur in expectation per epoch, proving both bounds. □

Corollary A.8 (Joint contraction with observable correction uncertainty). Assume the full binary domains, coordinate model (A.32), known initial coeficient bounds, and exact calibration protocol above. Assume a valid dependency-graph coloring, Assumptions 2 and 3 for the resulting surrogates and iterates, and deterministic initialization. Define

$$
\begin{array} { r } { \beta _ { 0 } : = \kappa ^ { - 1 } \operatorname* { m a x } _ { b } ( 1 + q _ { b } ( H ) ) , \quad \omega _ { 0 } : = 2 \beta _ { 0 } / \gamma , \ } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \tau _ { 0 } : = \operatorname* { m i n } \{ \mu q ( H ) / \kappa , \gamma / 2 \} , \quad \mathcal { W } _ { k } : = \Delta ( x ^ { k } ) + \omega _ { 0 } \Gamma _ { k } . } \end{array}\tag{A.36}
$$

Then $\begin{array} { r } { \mathbb { E } \mathcal { W } _ { K } \le ( 1 - \tau _ { 0 } ) ^ { K } \mathcal { W } _ { 0 } } \end{array}$ for every integer $K \geq 0$ . The radius $\Gamma _ { k }$ is observable, although $\Delta ( x ^ { k } )$ need not be. In particular, $\Gamma _ { k } = 0$ certifies that the unknown coeficients have been learned, not that the current solution is globally optimal.

Proof idea. Repeat the joint-potential argument with the computable uncertainty radius replacing the unknown coeficient-error norm. The same weighting principle turns two coupled inequalities into one contraction.

Proof. Set $\alpha = \mu q ( H ) / \kappa$ . Here $\Gamma _ { k + 1 }$ is the radius after this epoch’s deliberate probes; the ensuing solution update does not change it. Conditional on those probes, each block’s correction error is at most its remaining coordinate-radius sum. The gain and merge bounds therefore give

$$
\begin{array} { r l } & { \mathbb { E } [ \Delta ( x ^ { k + 1 } ) \mid \mathcal { H } _ { k } ] } \\ & { ( \mathrm { A . 4 } ) , ( 8 ) , ( 1 2 ) , ( \mathrm { A . 3 6 } ) } \\ & { \leq \qquad ( 1 - \alpha ) \Delta ( x ^ { k } ) + \beta _ { 0 } \mathbb { E } [ \Gamma _ { k + 1 } \mid \mathcal { H } _ { k } ] . } \end{array}\tag{A.37}
$$

Adding the weighted radius yields

$$
\mathbb { E } [ \mathcal { W } _ { k + 1 } \mid \mathcal { H } _ { k } ]
$$

$$
\begin{array} { r l } { ( \mathbf { A } . 3 7 ) , ( \mathbf { A } . 3 5 ) , ( \mathbf { A } . 3 6 ) } & { { } } \\ { \leq } & { { } ( 1 - \alpha ) \Delta ( x ^ { k } ) } \\ { \quad } & { { } \quad + ( \beta _ { 0 } + \omega _ { 0 } ) ( 1 - \gamma ) \Gamma _ { k } } \\ { ( \mathbf { A } . 3 6 ) } & { { } } \\ { \leq } & { { } ( 1 - \tau _ { 0 } ) \mathcal { W } _ { k } . } \end{array}\tag{A.38}
$$

For the last step, $\tau _ { 0 } \leq \alpha , \gamma / 2$ and the coeficient margin is

$$
\omega _ { 0 } ( 1 - \gamma / 2 ) - ( \beta _ { 0 } + \omega _ { 0 } ) ( 1 - \gamma )
$$

$$
\stackrel { ( \mathrm { A . 3 6 } ) } { = } \beta _ { 0 } \gamma \ge 0 .
$$

Total expectation and induction establish the stated bound. □

Reusing verification while preserving the radius. For a binary proposal, subtract the contributions of all already identified coeficients from its residual. If its displacement contains exactly one still-unknown coeficient, the remaining residual identifies that coeficient by division by its nonzero bit displacement. Mark it identified, just as after a deliberate bit probe. Doing this after proposal generation only removes extra nonnegative terms from the final radius. In the proof, retain the radius just after deliberate probing for the taskerror bound, and bound the final radius by that intermediate radius in the potential. The same coeficient inequality proves Corollary A.8; Proposition A.7 also remains valid. Local contraction must hold for the augmented histories. A general multi-coordinate projection need not preserve the individual bounds $M _ { b j } ;$ it belongs to Proposition A.6’s norm analysis instead.

A.5.1 The cost ofacquiring the correction. Let a scalar verifier call cost $v ,$ a parallel private slot cost $^ { c , }$ and other per-epoch overhead cost �. Here � excludes verifier calls, which are charged separately through $v ;$ in the exact-correction model of Section A.3, they were included in its aggregate overhead. All terms below use the same modeled cost units. Evaluating each agent’s proposal separately costs at most � calls; color additivity supplies the new base value without an extra union query. The bound below is for Algorithm 1; the exact best-combination merge adds up to $2 ^ { m } - m - 1$ calls per epoch, and with pairwise factors $\vert I ^ { k } \vert ( \vert I ^ { k } \vert - 1 ) / 2$ calls find the best combination of improving blocks (Appendix E.9). For a prescribed �,

$$
\mathbb { E } T _ { K } \le K ( C + c H ) + v [ 1 + m K + \operatorname* { m i n } \{ d _ { \mathrm { t o t } } , m p K \} ] .\tag{A.39}
$$

The four terms count initial evaluation, local computation and synchronization, proposal evaluations, and calibration queries. The initial base query is charged once. Feature evaluation, coupling-context refresh, and correction-message transfer belong in $C$ or $c ;$ they are not free. A suficient horizon for $\mathbb { E } \Delta \leq \varepsilon$ is

$$
K = \lceil \tau _ { 0 } ^ { - 1 } \log ( \mathcal { W } _ { 0 } / \varepsilon ) \rceil , \qquad 0 < \varepsilon < \mathcal { W } _ { 0 } .
$$

This is a fixed-horizon guarantee. Since $\tau _ { 0 } \le p / ( 2 d _ { \mathrm { m a x } } )$ , larger � cannot speed it up beyond the calibration rate; with $p = 1$ all coeficients are identified within $d _ { \mathrm { m a x } }$ <sub>x</sub> epochs, after which Section A.3 applies.

A.5.2 Why arbitrary objectives need structure. Without structure, evaluations at previously visited candidates need not reveal anything about an unvisited one. On an unrestricted finite candidate set, suppose some candidate � has never been queried. Two objectives can agree on every observed value and difer arbitrarily at �. They generate the same transcript and correction estimate, but its error at � cannot be bounded uniformly for both objectives. If values are known only to lie in an interval of width �, the same argument precludes a guaranteed radius below $M / 2$ there. Adaptive queries do not change this indistinguishability argument. To infer unmeasured candidate values, the method needs a low-dimensional identifiable model, another regularity assumption, or enough queries to cover the candidates. Sparse evaluations alone cannot provide a uniformly accurate correction for an arbitrary objective.

## B Detailed related work

This appendix complements the related work of the main paper. It discusses corrected local computation, distributed optimization, surrogate learning, the organization of LLM collaboration and LLM planning for several robots.

Relation to local training. ProxSkip uses a persistent control variate to counter the drift caused by skipping an expensive proximal operation (Mishchenko et al., 2022). SCAF-FOLD uses control variates to address client drift in federated learning (Karimireddy et al., 2020). We borrow the question these mechanisms answer: what makes private steps point toward the shared objective? Here the answer is a correction to candidate values, not a gradient vector or an averaging rule. Agents own disjoint blocks, external context is frozen within an epoch, and synchronization combines accepted block replacements instead of averaging full models. Our theorem is therefore a block-descent result; whether ProxSkip-type acceleration carries over remains open.

Table B.1 makes the distinction explicit. Appendix A.4 supplies a persistent correction learned from objective-evaluation residuals and a joint error analysis for a restricted identifiable model. Appendix A.5 makes its uncertainty observable and charges calibration explicitly.

Classical optimization and distributed local search. Parallel numerical iteration and its communication costs have a long history (Bertsekas and Tsitsiklis, 1989). Inexact coordinate descent explicitly permits approximate block solves (Tappenden et al., 2016); our local contraction is an assumption on such a solve, not a new solver. Theorem 2 is an inexact block-descent argument under a quantitative local-to-global condition. Its finite domains and candidate-value errors difer from the smooth convex setting of those coordinate-descent analyses, but the descent principle is shared.

MGM chooses improving agents through neighbor gain comparisons, excluding simultaneous conflicting moves; DSA randomizes local updates (Maheswaran et al., 2004; Zhang et al., 2005). Our coordinator evaluates all proposals at a common base and accepts either the improving proposals of the best color (Algorithm 1) or the best verified combination of proposals (Algorithm 2). The monotonicity principle is closely related, while the selection rule and oracle interface difer. Work on �-optimal DCOP solutions enlarges the coordinated neighborhood and studies its solution guarantees (Bowring et al., 2008); this is directly relevant when our single-block error bound fails. Assigning every agent utility −� also gives an exact potential game with potential −� (Monderer and Shapley, 1996). Unilateral improvement then reaches a coordinatewise equilibrium; global optimality still needs the additional geometry in Proposition A.1.

Table B.1: Relation to ProxSkip (Mishchenko et al., 2022). Our correction can be learned from verifier feedback; its update and rate difer from ProxSkip’s.
<table><tr><td></td><td>ProxSkip</td><td>Present construction</td></tr><tr><td>Correction</td><td>Updated recursively through occasional proxi- Learned persistently from scalar verifier residuals mal steps</td><td>in an identifiable model</td></tr><tr><td>Private work</td><td>Corrected gradient steps</td><td>Search within an assigned artifact block</td></tr><tr><td>Analysis</td><td>Joint iterate-correction error</td><td>Joint task-correction contraction; observable un- certainty for coordinate probes</td></tr><tr><td>fit</td><td>Communication bene- Accelerated communication under its optimiza- tion assumptions</td><td>Batching and a finite calibration budget; no Prox- Skip acceleration claim</td></tr></table>

Expected separable overapproximations in parallel coordinate descent handle interference through randomized update models and upper bounds (Richtárik and Takáč, 2016). Our exact-additivity coloring argument is simpler and more conservative. The best-combination merge replaces 1/� by the fraction of the verified gain it actually keeps, which is at least one over the number of conflict-free groups for pairwise factors (Section 5).

Surrogate learning and measurements. Additive Bayesian optimization uses structural decompositions to make expensive objective optimization tractable (Kandasamy et al., 2015). Linear-bandit best-arm identification studies which measurements reveal enough about a shared unknown parameter to select a good action (Soare et al., 2014). Our representation is a noiseless, known-feature surrogate with deterministic candidate-error bounds and block descent. Learning an objective from scalar evaluations is not new; we add a bound that converts the remaining correction error into lost progress and an accounting of measurement cost (Section 7).

Established ingredients and proposed combination. Block optimization supplies the local-to-global framework (Tseng, 2001; Nesterov, 2012; Richtárik and Takáč, 2016). Distributed constraint optimization studies variables owned by agents, with costs that couple their choices. Its DPOP algorithm communicates tables summarizing the best achievable cost for specified neighboring choices, exposing the importance of message size (Petcu and Faltings, 2005). The contribution of this paper is the explicit combination of a candidate-value correction, its approximation error, a verified merge, and a communication-horizon calculation for an agent-artifact model.

## B.1 AI-agent collaboration

The following work studies how LLM agents divide tasks, communicate and use feedback. We compare these mechanisms with the objective correction and convergence analysis developed here.

Shared artifacts and dialogue. MetaGPT and ChatDev organize specialized agents around software-development workflows and intermediate artifacts (Hong et al., 2024; Qian et al., 2024). AutoGen builds applications by connecting agents that exchange messages, while AgentVerse studies collaborative agent groups (Wu et al., 2024; Chen et al., 2024b). Multiagent debate improves answers through exchanged critiques (Du et al., 2024). These works motivate specialization and information exchange. Our model instead makes a particular optimization interface explicit: owned solution blocks, candidate scores, and an exact scalar task verifier.

Communication topology and cost. In a communication graph, nodes represent agents or operations and edges specify who sends information to whom. GPTSwarm optimizes such language-agent graphs; DyLAN adapts the participating agent network; G-Designer selects communication patterns for a given task (Zhuge et al., 2024; Liu et al., 2024; Zhang et al., 2025c). AgentPrune removes redundant messagepassing edges, AgentDropout eliminates agents dynamically, and DALA allocates opportunities to speak through auctions (Zhang et al., 2025b; Wang et al., 2025; Fan et al., 2026). Thus, optimizing communication expenditure is an established research direction. Here the question is how omitted global requirements distort private search, and which persistent correction permits useful work between synchronizations. Our dependency graph describes objective factors and compatible edits; it is not a learned message-routing graph.

Feedback and evolving collaboration. EvoMAC uses verification-derived textual feedback to evolve agents and their connections (Hu et al., 2025). AFlow searches over workflows—programs that specify the sequence and connections of model calls—using feedback from executing them (Zhang et al., 2025a), and evolving orchestration adapts how agents are selected and coordinated (Dang et al., 2025). Test-time communication for discovery is another closely related setting (Park et al., 2026). These works already connect feedback, accumulated experience, and collaboration. Our narrower contribution is to estimate the missing part of a numerical objective, bound the estimation error, and prove joint convergence of the solution and the estimate when the unknown coeficients can be identified.

LLM planning for several robots. In RoCo, one LLM agent per robot negotiates a joint plan through dialogue and replans from motion-planner feedback such as collisions (Mandi et al., 2024). SMART-LLM decomposes instructions and assigns the subtasks to robots (Kannan et al., 2024); centralized, decentralized and hybrid LLM planners have been compared as teams grow (Chen et al., 2024a); and LLMs still fail at multi-agent path finding on maps with obstacles (Chen et al., 2025). Our routing agents also receive collision feedback, but as a numerical correction to their own objective, computed for the robot pairs whose collisions the exact verifier has reported, from those robots’ current routes, and only verified combinations of proposals are accepted.

Calibration, reliability, and scaling. CommCP calibrates LLM-generated messages through conformal prediction for robots cooperating to answer questions about their physical surroundings (Zhang et al., 2026). Its message-reliability objective difers from our correction requirement, which concerns only a true block minimizer and the returned proposal (Assumption 1). Empirical analyses of multi-agent failures and scaling further motivate checking coordination and verification rather than presuming that more agents or messages help (Cemri et al., 2025; Kim et al., 2026). Our assumptions state what such checks must deliver for parallel work to add up to global progress.

## C Routing: task, verifier and measurement

This appendix defines everything the routing experiments share: why we chose the task, the two maps and their two-floor combination, commands, loss, local objective and correction, how feedback reaches the agents, the protocols, the measurement of search time and the exact references that certify optima and block minima. Appendix D explains how we chose the prompt and the protocols, trying two versions of the prompt, and Appendix E reports the results of the main text.

## C.1 Which tasks need corrections

Three regimes. A team of agents is useful when a task is too large or too hard for one agent, and the parts can be searched in parallel. Whether the team also needs coordination depends on how the parts interact. A simple task, such as changing a light bulb, gains nothing from a team: splitting it only adds overhead. A task whose parts are independent, such as one agent watering a garden while another goes shopping, can be split without coordination; each agent optimizes its own part, and the team finishes sooner. In the language of Section 3, such a task is separable: if $\begin{array} { r } { F ( x ) = \sum _ { b } L _ { b } ( x _ { b } ) } \end{array}$ up to a constant, then the exact correction $R _ { b }$ is zero, the dependency graph has no edges, a single color contains every block, and all improving proposals can be merged in one epoch. Theorem 2 then holds with $\delta _ { b } ^ { * } = \delta _ { b } ^ { U } = 0$ and � = 1, but it says nothing that independent parallel search does not already achieve.

The case that motivates this paper lies between these extremes: tasks whose parts are coupled. Examples include the modules of a large program, the structure, wiring and plumbing of a building, the subsystems of an aircraft, and the routes of robots that share a floor. Each part is manageable alone, but changing it alters what is best for its neighbors, so independent local improvements need not combine into a better whole. Here corrections tell each agent how its candidates afect the others, and coloring decides which improvements can be applied together. The strength of the coupling, not the number of agents, determines how much these mechanisms matter.

Why a routing task. We wanted an experiment with few agents in which this coupling is strong, controlled and measurable. Multi-robot routing meets these needs. Its rules are simple: five commands, walls and a fixed horizon, and a reader can check a plan by eye. It is nevertheless hard for current LLMs, which must track coordinates, obstacles and timing; LLMs still fail at multi-agent path finding on maps with obstacles (Chen et al., 2025), and in our runs with prompt v1 a single model writing all nine routes reached a safe plan in only one run of ten (Appendix D). Most importantly, coupling is built into the task: robots owned by diferent agents can collide, so an agent’s best plan for its own robots can harm another agent’s robots. The exact verifier measures this coupling directly; in the team runs of the main text, 30% of the proposals that delivered all three of an agent’s robots without internal collisions still collided with another agent’s robot on the first map, and 32% on the second (Appendix E.7). A certified optimum and exact block minima also let us measure every constant of the theory.

Why not software. Code generation was the first application we considered, and the introduction uses it as an example. It is harder to use for a controlled study of coupling. Good software is designed to limit coupling between modules (Parnas, 1972), so small coding tasks rarely exhibit strong interactions between the parts that diferent agents would own; those interactions mainly appear in large systems. A coding benchmark would also need test suites whose failures can be attributed to particular interactions, and a numerical objective whose optimum is known. Building such a benchmark at a scale where coupling is strong and measurable is a natural next step.

## C.2 Maps, robots and the two-floor task

The task is discrete multi-agent pathfinding with vertex and edge-swap conflicts and a stay-at-goal convention (Stern et al., 2019). Figure C.1 and Table C.1 specify both instances. Coordinates are $( x , y ) \in \{ 0 , \ldots , 1 5 \} ^ { 2 }$ , with � increasing rightward and � downward. On the first map, three $2 \times 2$ obstacles occupy $\{ 5 , 6 \} \times \{ 4 , 5 \} , \{ 9 , 1 0 \} \times \{ 7 , 8 \}$ and $\{ 5 , 6 \} \times \{ 1 0 , 1 1 \}$ On the second map, the twelve wall cells form a horizontal wall of four cells, a vertical wall of four cells and two pieces of two cells. Every other cell is traversable. The simulation horizon is 48 ticks.

A model writes one JSON object with a programs field containing a literal command array for every robot it controls. Allowed commands are up, down, left, right, and wait; each consumes one tick and changes position by $( 0 , - 1 ) , ( 0 , 1 )$ ， $( - 1 , 0 ) , ( 1 , 0 ) , \mathrm { o r } ( 0 , 0 )$ , respectively. The model constructs the arrays rather than selecting a precomputed route. Arrays have at most 48 commands. After an array ends, the robot waits in its last cell through tick 48. Upon its first arrival at its goal, the robot must remain there. Every evaluation restarts all robots at their original starts at tick zero; optimization iterations do not advance a physical journey.

The second map has the same size, horizon, loss and number of wall cells as the first. Starts and goals were drawn at random, and the map was kept because its statistics match the first map’s: the sum of shortest distances is 147 against 144, and letting every robot take a shortest path produces five collisions against four. On it, the first agent’s three robots move from left to right, the second’s from top to bottom and the third’s from bottom to top. The two-floor task places both maps on the two floors of one building: eighteen robots, $F = F _ { 1 } + F _ { 2 }$ with optimum 291 and all-waiting value 2955; robots on diferent floors never collide, and a plan meets the threshold when both floors are safe and within 9 of their optima.

## C.3 Global loss, local loss and correction

The loss, the local objective and the correction are (21)–(23) of Section 8.1; this subsection gives their exact conventions. Let $p _ { i } ( t )$ be robot $i \mathrm { { s } }$ position at tick $t ,$ and let $\tau _ { i } \leq$ 48 denote its first arrival at its goal $g _ { i }$ if it arrives. For a robot that does not arrive, $d _ { i }$ is the shortest four-neighbor grid distance from $p _ { i } ( 4 8 )$ to $^ { g _ { i } , }$ respecting walls. Arrival time, including waits, costs one unit per tick. The horizon term applies to every undelivered robot even when its submitted array is short.

For each pair $i < j , C _ { i j }$ counts a vertex conflict whenever $p _ { i } ( t ) = p _ { j } ( t )$ , and an edge-swap conflict whenever the robots exchange distinct neighboring cells during the same tick. Conflicts are checked over ticks zero through 48; repeated colocation contributes one event per pair and tick. Geometric path intersection at diferent ticks is safe. The collision weight is 100 for both conflict types. A conflict adds a penalty but does not stop or redirect either submitted trajectory. Walls, in contrast, are hard constraints. They restrict only each robot’s own route, so they define the block domains $\mathcal { X } _ { b }$ rather than coupling blocks; collisions between robots are penalized instead, because they are the coupling under study. Empty arrays initialize the search with all robots stationary: $F ( x ^ { 0 } ) = 9 ( 1 0 0 + 4 8 ) + 1 4 4 = 1 4 7 6 { \mathrm { ~ o n } }$ the first map and 1479 on the second. The sums of individual shortest-path distances are 144 and 147. Separately computed collision-free witnesses attain these sums, certifying the global optima $F ^ { * } = 1 4 4$ and $F ^ { * } = 1 4 7$ . Appendix C.7 describes the reference algorithm and its certificates.

The local loss (22) of a team block counts its three robots’ delivery costs and all conflicts among them, and the correction (23) covers the external pairs in $\mathcal { P } ^ { k }$ , those discovered before global round $k ,$ with the other blocks frozen at $x ^ { k }$ . The checked private score change is $L _ { b } ( u ) - L _ { b } ( x _ { b } ^ { k } ) + \widehat { R } _ { b } ( u ; x ^ { k } )$ . A discovered pair remains known, but the peer’s time-indexed positions are refreshed from the current accepted plan at the next round. If every external pair is known, this score change equals the conditional global loss change; before then it can miss conflicts. The experiment learns which pairs to check; the penalty coeficients are known.

## C.4 How feedback works

The prompts state the rules and show the checked state visible to the agent; their two versions are described in Appendix D. The following mechanics are common to both.

Initially $\mathcal { P } ^ { 0 } = \varnothing$ . If global verification discovers an external collision, the following round supplies that pair’s event information and the peer’s refreshed trajectory. Unrelated robots remain hidden. A spatial line alone would not sufice: the correction uses simultaneous occupancy and movement at specific ticks. Agents receive their previous proposal even if it was not accepted, and the prompt distinguishes it from the current incumbent. Repairs must be new complete arrays from the original start; appending a wait after an earlier collision cannot repair that collision.

Format failure versus route failure. A missing or malformed JSON answer, wrong set of IDs, non-array output, or unknown command token is not a checked search answer. It is retried with the identical prompt and settings and a deterministic new retry seed. Format failures are re-asked at most three times per logical attempt, and transport failures at most three times with prompt v1 and twenty times with prompt v2; exhaustion pauses execution for diagnosis. In contrast, a readable array that enters a wall, leaves the grid, exceeds 48 commands, or moves away after arrival is a checked answer. Any such error rejects the entire owned proposal, retains the incumbent and returns route feedback; its candidate global loss is not evaluated. Otherwise, unfinished deliveries and collisions yield a finite score under (21). These answers count toward search attempts and time even when rejected as worse.

## C.5 Protocols

All protocols start from the same empty plan and accept only improvements of the global loss.

Team. A global round is one batch of three concurrent block searches (six on two floors) followed by verification and a commit: one block per round on one map with Algorithm 1, one per floor on two floors, and the accepted combination with the other merge rules. All searches use the same frozen $x ^ { k }$ and $\mathcal { P } ^ { k }$

![](images/dd86a722c381eb00f76a616c20118830b107a5debffb60218020702d3e046996.jpg)

(b) Map 2 (F¤ = 147), second floor  
![](images/18931166000d8d7e2a660651d2e44eb355cd5995a6ec68c6c2f14238ce108837.jpg)

Figure C.1: The two maps. Circles: starts; squares: goals; numbers: robot IDs; colors: the team agent that owns the robot (the single agent controls all of them); arrows point from start to goal; dark cells are walls. On two floors, map 1 is the first floor and map 2 the second, with its robots named s0–s8.
<table><tr><td colspan="5">Map 1 (first floor)</td><td colspan="5">Map 2 (second floor, s0-s8 on two floors)</td></tr><tr><td>Robot</td><td>Agent</td><td>Start</td><td>Goal</td><td>Dist.</td><td>Robot</td><td>Agent</td><td>Start</td><td>Goal</td><td>Dist.</td></tr><tr><td>r0</td><td>1</td><td>(2,3)</td><td>(13, 10)</td><td>18</td><td>r0</td><td>1</td><td>(2,1)</td><td>(14,12)</td><td>23</td></tr><tr><td>r1</td><td>1</td><td>(3, 1)</td><td>(10, 14)</td><td>20</td><td>r1</td><td>1</td><td>(1,9)</td><td>(13, 10)</td><td>13</td></tr><tr><td>r2</td><td>2</td><td>(2, 7)</td><td>(13, 5)</td><td>13</td><td>r2</td><td>1</td><td>(2, 4)</td><td>(14,5)</td><td>13</td></tr><tr><td>r3</td><td>2</td><td>(7, 1)</td><td>(5, 14)</td><td>15</td><td>r3</td><td>2</td><td>(6, 2)</td><td>(5, 13)</td><td>14</td></tr><tr><td>r4</td><td>3</td><td>(2, 12)</td><td>(13, 8)</td><td>15</td><td>r4</td><td>2</td><td>(14, 1)</td><td>(11, 13)</td><td>15</td></tr><tr><td>r5</td><td>3</td><td>(12, 1)</td><td>(8, 14)</td><td>17</td><td>r5</td><td>2</td><td>(12, 1)</td><td>(6, 14)</td><td>19</td></tr><tr><td>r6</td><td>1</td><td>(13, 3)</td><td>(2, 10)</td><td>18</td><td>r6</td><td>3</td><td>(4, 13)</td><td>(10, 2)</td><td>17</td></tr><tr><td>r7</td><td>2</td><td>(13, 7)</td><td>(2,5)</td><td>13</td><td>r7</td><td>3</td><td>(8, 14)</td><td>(9,2)</td><td>15</td></tr><tr><td>r8</td><td>3</td><td>(13, 12)</td><td>(2, 8)</td><td>15</td><td>r8</td><td>3</td><td>(12, 14)</td><td>(7, 1)</td><td>18</td></tr><tr><td>Sum</td><td></td><td></td><td></td><td>144</td><td>Sum</td><td></td><td></td><td></td><td>147</td></tr></table>

Table C.1: Assignments and shortest legal grid distances on both maps. On the first map the blocks are $\{ r _ { 0 } , r _ { 1 } , r _ { 6 } \} , \{ r _ { 2 } , r _ { 3 } , r _ { 7 } \}$ and $\{ r _ { 4 } , r _ { 5 } , r _ { 8 } \}$ ; on the second, $\{ r _ { 0 } , r _ { 1 } , r _ { 2 } \} , \{ r _ { 3 } , r _ { 4 } , r _ { 5 } \}$ and $\{ r _ { 6 } , r _ { 7 } , r _ { 8 } \}$ . The sums are the optima.

(1) Each agent proposes arrays for its three robots, receives private feedback, and may make up to $H = 3$ checked answers. Python retains its best strictly improving proposal for $L _ { b } + \widehat { R } _ { b }$ , including the unchanged block as an option.

(2) Check the three proposals globally: substitute one proposed block into $x ^ { k }$ while keeping both other blocks unchanged. Compare these three full-plan losses with $F ( x ^ { k } )$

(3) Accept only the proposal with the largest positive global decrease. Ties favor the lower block index; if none improves, retain $x ^ { k }$ . All block pairs may interact, so a conservative complete dependency graph has three colors, each containing one block.

(4) Retain all external collision pairs found during global checks, including those in unselected proposals. Peer trajectories for the next round come from the new accepted plan, rather than a discarded proposal.

This is parallel proposal generation with serial block commitment. Within each private search, the correction and peer routes stay frozen. A deterministic routine computes a certified minimum or a valid lower bound for the available three-robot objective. Search stops early only when a submitted score equals that bound; otherwise it uses its remaining � attempts. Appendix C.7 describes this routine and the independent ofline certificates. The team’s stopping rule differs between the two prompt versions (Appendices D and E); restarting a run never resets a counter.

Single agent and best-of-�. A single agent controls all robots, sees and scores the full plan and receives full-plan feedback; every checked answer is evaluated immediately, and any strictly lower-loss legal plan is accepted at once. There is no grouping into rounds and no hidden-pair correction. The answer scope (all routes, up to � routes or exactly one route per answer) and the stopping rules are given with each variant. The best-of-� runs � copies of the one-route single agent in parallel on the same accepted plan and prompt, with their own seeds; the best strictly improving answer of a round is accepted, its round time is the slowest of the � answers, and � equals the number of team agents.

## C.6 Model, seeds and measurement

The model is openai/gpt-oss-120b, an open-weight mixtureof-experts reasoning model described in the OpenAI model card (OpenAI, 2025). All routing runs, with both prompt versions, use the same OpenRouter/Groq endpoint, pinned without fallback, with reasoning efort medium, temperature 0.6 and top $- p = 0 . 9 5$ . No max\_tokens, max\_completion\_tokens, or reasoning-token cap is sent. The network timeout is 3600 seconds; there is no absolute client deadline or automatic reasoning escalation. The API does not expose a controlled accelerator allocation.

Inference service and hardware. OpenRouter supplies the API and routes requests to the selected inference provider (OpenRouter, 2026); Groq performs inference. The saved payloads restrict routing to groq with allow\_fallbacks=false, and returned provider metadata identifies Groq. Groq documents GPT-OSS-120B serving through its LPU (Language Processing Unit) inference platform (Groq, 2025, 2026). This identifies the accelerator family; the chip generation, accelerator count, location, batching and service load are not exposed by the API.

The client measures each request with a monotonic timer around the HTTP call, response download and response processing. Consequently, the durations below include network, routing and service latency, as well as model generation; these components are not separated. Search time aggregates only checkable-answer durations according to (C.1); elapsed time and failed-request costs are recorded separately. All routing variants use the same provider restriction.

For replicate �, the base API seed is 4,000,000 + 100,000�. Single-agent checked step $j ,$ starting at zero, adds � with prompt v1 and 1000� with prompt v2 (copy � of a best-of-� adds $1 0 0 0 j + c )$ ; team round $k ,$ private step ℎ and block $b ,$ also zero-indexed, add $1 0 0 0 k + 1 0 0 h + b$ . Retry seeds are deterministic hashes of the logical seed and retry index. Seed settings are recorded, but provider reproducibility is best efort and the protocols induce diferent request streams. Map, settings, thresholds and limits were fixed before the runs of each campaign, and failed seeds are not rerolled.

Let $a _ { j }$ be the latency of a returned answer whose command object can be checked. The single-agent search time is $\textstyle \sum _ { j } a _ { j }$ If agent � has checked latencies $a _ { k b h }$ in round �, the team search time is

$$
t _ { \mathrm { t e a m } } = \sum _ { k } \operatorname* { m a x } _ { b \in \{ 0 , 1 , 2 \} } \sum _ { h } a _ { k b h } .\tag{C.1}
$$

We call this the search time, as in the main text. The three agents’ times are not added as elapsed parallel time. Geomet ric errors and collisions count; missing or malformed answers and transport failures do not count toward search time. Actual elapsed time, preparation, stopping checks, verification, all physical requests, usage and reported charges are recorded separately. The simulator is inexpensive and has no artificial evaluation delay; this study does not measure a saving from expensive global evaluations.

## C.7 Reference routes and certificates

Reference computations certify the global optimum on the loss curves and the block minima used in the local-quality and block-geometry diagnostics. They use the model evaluator’s commands, horizon, goal convention and weights.

Global optimum. Breadth-first search on the obstacle grid computes each robot’s shortest distance $d _ { i } ( s _ { i } , g _ { i } )$ , as listed in Table C.1. On the first map, their sum is a lower bound on every full-plan loss:

$$
F ( x ) \geq \sum _ { i = 0 } ^ { 8 } d _ { i } ( s _ { i } , g _ { i } ) = 1 4 4 .\tag{C.2}
$$

A delivered robot cannot arrive before its grid distance. An undelivered robot costs at least 148, exceeding every individual distance (at most 20 on the first map and 23 on the second), and conflict penalties are nonnegative. Conflictbased search (CBS) (Sharon et al., 2012) finds a nine-robot plan with all deliveries, no conflicts and total arrival cost 144. Independent replay confirms that it attains the bound, proving $F ^ { * } = 1 4 4$ even when unfinished or colliding plans are considered. The saved search expands seven high-level CBS nodes. The same argument certifies $F ^ { * } = 1 4 7$ on the second map, where CBS again finds a delivered, collision-free plan whose arrival cost equals the sum of shortest distances, and $F ^ { * } = 1 4 4 + 1 4 7 = 2 9 1$ on two floors, whose floors do not interact.

Search algorithms. For one robot, breadth-first search uses states $( t , p ) \colon$ : tick $t \in \{ 0 , \ldots , 4 8 \}$ and free cell $p .$ A move or wait advances time by one; each transition checks obstacles, bounds and imposed vertex or directed-edge constraints. Goal arrival is accepted only if remaining there through tick 48 respects the constraints. Each state has one predecessor record, merging alternative command prefixes. With 244 free cells on either map, there are at most $4 9 \times 2 4 4 = 1 1 . 9 5 6$ states per single-robot search.

CBS starts with independently shortest constrained routes and orders its high-level nodes by total arrival cost. The earliest vertex conflict creates two branches, forbidding that cell at that tick for one robot or the other. An edge swap similarly branches on the directed transitions. Only the afected route is recomputed. The cheapest conflict-free node gives the minimum cost among delivered, conflict-free plans satisfying the constraints. CBS avoids enumerating all joint command arrays, but its tree can grow exponentially; computation-budget exhaustion certifies neither optimality nor impossibility.

Fixed-base block references. For a fixed base $x ^ { k }$ , let $A _ { b } ^ { \mathcal { P } ^ { k } } ( u ; x ^ { k } )$ be the absolute private score: $L _ { b } ( u )$ plus the external conflict penalties for retained pairs $\mathcal { P } ^ { k }$ . Then

$$
\widehat { f _ { b } } ( \boldsymbol { u } ; \boldsymbol { x } ^ { k } ) = \boldsymbol { A } _ { b } ^ { \mathcal { P } ^ { k } } ( \boldsymbol { u } ; \boldsymbol { x } ^ { k } ) - \boldsymbol { A } _ { b } ^ { \mathcal { P } ^ { k } } ( \boldsymbol { x } _ { b } ^ { k } ; \boldsymbol { x } ^ { k } ) .\tag{C.3}
$$

The other six robots’ programs remain fixed. Only retained external pairs enter the reference for the agent’s available objective; using all hidden peers here would change the denominator of $Q .$ Separately, let $A _ { b } ^ { \mathrm { a l l } }$ include every external pair. Costs involving only the other blocks cancel, so the true conditional global improvement is

$$
D _ { b } ( x ^ { k } ) = A _ { b } ^ { \mathrm { a l l } } ( x _ { b } ^ { k } ; x ^ { k } ) - \operatorname* { m i n } _ { u } A _ { b } ^ { \mathrm { a l l } } ( u ; x ^ { k } ) .\tag{C.4}
$$

For either reference, the relevant frozen peer positions become vertex constraints, and peer moves become reverse-edge constraints, for a collision-free witness search. CBS resolves conflicts among the three owned robots. All minima range over legal owned arrays, including waits and unfinished deliveries, rather than only a route catalogue.

Certifying the soft-penalty optimum. Finding the best collision-free delivered plan alone need not minimize the soft loss. Let $\begin{array} { r } { \bar { d } _ { b } = \sum _ { i \in \mathcal { T } _ { h } } d _ { i } \big ( s _ { i } , g _ { i } \big ) } \end{array}$ and let $A _ { b } ^ { \mathrm { C B S } }$ be that plan’s absolute score. Any plan with a penalized conflict costs at least $\bar { d } _ { b } + 1 0 0$ , and any unfinished plan costs at least

$$
\bar { d } _ { b } + \operatorname* { m i n } _ { i \in \mathcal { T } _ { b } } \{ 1 4 8 - d _ { i } ( s _ { i } , g _ { i } ) \} .
$$

The online routine certifies its CBS value only when $A _ { b } ^ { \mathrm { C B S } }$ does not exceed either bound. Otherwise it uses a valid single-robot relaxation bound for the early-stop test, without claiming that the CBS value is the soft minimum.

The ofline audit additionally uses independently implemented dynamic programming over $( t , p )$ . For each owned robot it minimizes delivery cost plus the relevant frozen-peer conflict penalties, including conflicts after arrival, and retains the non-delivery terminal cost of (21). Here conflicts remain penalized rather than forbidden. Dropping only conflicts among owned robots relaxes the joint block objective. If $\underline { { A } } _ { i }$ is this single-robot minimum and $w _ { b }$ is the joint witness, then, for either choice of external pairs,

$$
\sum _ { i \in \mathbb { Z } _ { b } } \underline { { A } } _ { i } \ \le \ \operatorname* { m i n } _ { u } A _ { b } ( u ; x ^ { k } ) \ \le \ A _ { b } ( w _ { b } ; x ^ { k } ) .\tag{C.5}
$$

A separate replay checks the witness’s legality and score. On the first map both inequalities are equalities in every audited subproblem, for the v1 and v2 team runs. On the second map, where conflicts among owned robots bind, the singlerobot bound lies 1 or 2 below the witness in 83 of the 1,074 subproblems of the v2 team runs (40 of 786 and 81 of 1,008 in the runs with best combination and unsafe merging); their minima are certified instead by the CBS optimum together with the penalty and non-delivery bounds above. Every block minimum used in the audit of Appendix E.7 is therefore certified. Diagnostic witnesses and lower bounds are saved separately from model answers. These certificates support the assumption checks of Appendix E.7; they do not supply routes to agents or alter any recorded decision.

## D Routing: prompt and protocol design

We wrote two versions of the prompt. With prompt v1 we tried several protocols for the single agent and the team on the first map, ten runs each. Where these runs stalled, the model had to simulate the other robots’ commands itself; prompt v2 presents the full checked state instead and made both protocols faster. For the main text we chose the combination that worked best: prompt v2, a single agent that replaces one route per answer, the best-of-� such agents as the parallel control, and the team that rewrites its three routes. Task, loss, correction, protocols and measurement are those of Appendix C. Figure D.1 shows the mean curves of all variants and Table D.7 their outcomes.

## D.1 Prompt v1

An initial prompt gives the command rules, coordinate orientation, horizon, loss weights, an ASCII obstacle map, owned robot IDs and start/goal coordinates, and their current command arrays. Team agents initially do not receive other blocks assignments or routes. The single agent receives all nine assignments and checks every pair internally. Each request states that the output must contain complete literal arrays for exactly the owned IDs.

After a checked answer, Python reports its score and perrobot arrival or remaining distance. For an illegal route it identifies the first rejected command, its tick, the previous and attempted cells, and legal directions from that cell. Feedback includes the legal position prefix and an ASCII route drawing for a failed or unfinished route. For visible conflicts it gives the robot pair, tick, vertex cell or swapped edge, and time-labeled map frames. A safe arrived route does not need a repeated full drawing. A supplied external peer is accompanied by its complete time-indexed positions, with only a constant trailing wait sufix compressed; the last listed position persists thereafter. At most three illustrative collision frames are shown, while complete event lists and peer motion remain available.

![](images/56292f42f25be6bafc83c6d4fa65fb95a1657cd6b37a2774d7cd3e72e3792d99.jpg)  
Figure D.1: How the protocols and the prompt were chosen (first map; mean gap $F - F ^ { * }$ against search time). (a) Single-agent protocols with prompt v1, ten runs each. (b) Team protocols with prompt v1, ten runs each; the sequential replay plays the team’s answers one agent at a time. (c) Prompt v1 against prompt v2: twenty runs for the v2 one-route single agent and team, ten for the other curves. Stars mark the end of recorded support.

Format failures were re-asked at most three times and transport failures at most three times. The team stopped upon reaching the threshold $( F \le 1 5 3$ with a safe plan), after five rounds without an improving commit, or at a ceiling of eight global rounds.

## D.2 Protocols tried with prompt v1

Team and whole-plan single agent. The team is the protocol of Appendix C. The first single agent rewrote all nine routes in every answer. It controls all nine arrays and receives full-plan feedback. Each checked answer is evaluated immediately; any strictly lower-loss legal plan becomes the incumbent immediately. There is no grouping into three-answer batches, no external hidden-pair correction, and no globalround counter. The chain stops upon reaching the stopping threshold, after five consecutive checked answers without an accepted improvement, or after sixty checked answers. A route-invalid answer contributes to that five-answer streak. Actual chains used 5–13 checked answers; sixty was a safety ceiling, not the number performed. Both variants also stop early once all nine robots are delivered without conflicts and $F \le 1 5 3 ;$ this rule only ends a run.

The twenty runs of these two protocols complete under their stopping rules. The team safely delivers every robot in 10/10 runs, with final losses 149–167; the whole-plan single agent does so in 1/10. Table D.7 reports final loss, search time and reported cost for these and all other variants. Table D.1 reports elapsed time and answer counts; Table D.2 retains every seed.

Figure D. $1 ( \mathbf { a } , \mathbf { b } )$ plots the mean gap $F - F ^ { * }$ over all ten runs of each variant; an accepted plan is held until the next accepted change, and a stopped run keeps its final loss. The team and its sequential replay end at mean $F = 1 5 6 . 5 0 \mathrm { _ { \cdot } }$ versus 841.00 for the whole-plan single agent.

<table><tr><td>Metric</td><td>Single agent (whole plan)</td><td>Team</td></tr><tr><td>Actual elapsed (s)</td><td> $3 1 5 . 2 5 \pm 1 3 2 . 3 6$ </td><td> $1 0 8 9 . 5 5 \pm 8 9 6 . 4 7$ </td></tr><tr><td>Checked answers</td><td> $8 . 3 0 \pm 2 . 7 1$ </td><td>59.30 ± 7.92</td></tr><tr><td>Safe delivery of all nine</td><td>1/10</td><td>10/10</td></tr></table>

Table D.1: All ten outcomes per variant; mean ± sample SD where applicable. Team answer counts sum across agents, whereas its search time uses the per-round maximum. Actual elapsed time includes failed requests and recovery pauses.

The journals contain 676 returned, checkable model responses and 142 HTTP 429 failures, for 818 physical requests. All returned responses include API-returned reasoning. Checked responses include 120 route-invalid proposals and 437 legal candidate proposals with collisions. Across both variants, the provider reports USD 3.553312 in charges; charges for the 142 HTTP failures are unknown, so reported costs are not a certified total bill. Diagnosed rate-limit interruptions resume after cooldowns by reusing saved answers and identical prompts; they do not restart a seed or change the model, map, loss or reasoning. Elapsed time includes recovery pauses. Independent replay checks every returned command, frozen base, local score, correction, block selection and final plan.

A flat curve can conceal attempted repairs. Single-agent seed 0 stops at the initial loss 1476 after five checked answers. It does not simply return stationary plans: the first answer has three illegal routes, the third has one, the fourth is legal but worse with $F = 2 0 9 2$ , and the fifth has one illegal route. Entire-plan rejection retains the incumbent, while the next prompt still contains attempted arrays and ofending commands. With the eficiency paragraph and patience fifteen (below), seed 0 again ends at $F = 1 4 7 6$

<table><tr><td colspan="5">Single agent, whole plan</td><td colspan="4">Team (Alg. 1)</td></tr><tr><td>Seed</td><td>Final F</td><td>Delivered/conflicts</td><td>Answers</td><td>Time (s)</td><td>Final F</td><td>Rounds</td><td>Answers</td><td>Time (s)</td></tr><tr><td>0</td><td>1476</td><td>0/0</td><td>5</td><td>165.6</td><td>152</td><td>7</td><td>45</td><td>608.0</td></tr><tr><td>1</td><td>1476</td><td>0/0</td><td>5</td><td>135.4</td><td>158</td><td>8</td><td>58</td><td>650.2</td></tr><tr><td>2</td><td>455</td><td>9/3</td><td>8</td><td>264.4</td><td>152</td><td>7</td><td>57</td><td>494.2</td></tr><tr><td>3</td><td>490</td><td>8/2</td><td>10</td><td>372.6</td><td>155</td><td>8</td><td>59</td><td>673.6</td></tr><tr><td>4</td><td>722</td><td>9/5</td><td>9</td><td>253.7</td><td>155</td><td>8</td><td>66</td><td>578.9</td></tr><tr><td>5</td><td>959</td><td>9/7</td><td>8</td><td>307.7</td><td>159</td><td>8</td><td>59</td><td>633.3</td></tr><tr><td>6</td><td>1476</td><td>0/0</td><td>5</td><td>176.1</td><td>164</td><td>8</td><td>58</td><td>582.3</td></tr><tr><td>7</td><td>609</td><td>8/3</td><td>9</td><td>294.0</td><td>149</td><td>8</td><td>51</td><td>486.0</td></tr><tr><td>8</td><td>150</td><td>9/0</td><td>11</td><td>513.3</td><td>167</td><td>8</td><td>72</td><td>544.5</td></tr><tr><td>9</td><td>597</td><td>9/4</td><td>13</td><td>330.9</td><td>154</td><td>8</td><td>68</td><td>642.9</td></tr></table>

Table D.2: Every completed seed, without selection. Time is the search time defined above. All ten team outcomes deliver $9 / 9$ without conflicts. Single-agent delivered/conflict counts refer to its final accepted plan, including an unchanged initial plan.

The nine failed runs each end with five nonimproving answers, which take 162.98 ± 24.31 seconds per run (mean ± sample SD): 34 of these 45 answers are route-invalid. All error durations already count toward search time. The last response, at 513.28 seconds, belongs to the successful seed; no later attempts were measured. Counts and clocks are verified by independent replay.

Team without correction. This ablation disables only the cross-block correction channel of the team; Appendix E repeats it with prompt v2. Agents receive no acquired external pairs, no peer timelines and no global collision feedback in their local searches; feedback about their own robots, including internal collisions, is unchanged. The coordinator still evaluates the full objective, logs collisions and commits the single best improving block. Model, provider, sampling parameters, prompt, $H = 3$ , the eight-round ceiling, the fiveround plateau rule, the safe target and the ten prescribed seed streams are those of the v1 team. The first request payloads are byte-identical, so the two variants diverge only when the corrected controller would first have revealed a conflict. Local early stopping is unchanged: a block that is optimal for its local objective may stop requesting answers while the joint plan still contains external collisions.

<table><tr><td>Seed</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>Final F</td><td>452</td><td>444</td><td>152</td><td>451</td><td>448</td><td>294</td><td>448</td><td>444</td><td>548</td><td>248</td></tr><tr><td>Collisions</td><td>3</td><td>3</td><td>0</td><td>3</td><td>3</td><td>1</td><td>3</td><td>3</td><td>4</td><td>1</td></tr><tr><td>Rounds</td><td>8</td><td>8</td><td>4</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr></table>

Table D.3: All ten no-correction team outcomes. Every final plan delivers all nine robots; collisions are cross-block events in the final plan.

All ten runs complete, and every final plan delivers all nine robots (Table D.3). Nine retain one to four cross-block collisions, 2.4 on average over all ten runs; only seed 2 reaches a safe plan, with $F = 1 5 2$ . Final loss is $3 9 2 . 9 0 \pm 1 2 0 . 6 6$ , search time $3 3 7 . 9 7 \pm 9 5 . 3 4$ seconds, with $3 9 . 8 \pm 1 0 . 4$ checked answers over $7 . 6 \pm 1 . 3$ rounds (mean ± sample SD). Subtracting the collision penalty of 100 per event leaves a mean of 152.9, close to the corrected v1 team’s 156.50. Delivery quality within blocks is therefore comparable, and the remaining loss consists of exactly the cross-block interactions that the uncorrected local objective omits. Because only improving blocks are committed, the uncorrected team never increases $F ,$ but its agents cannot tell which of their candidates would remove an external collision, so search stalls once every block is locally good. Reported charges total USD 1.32 over 476 requests, 78 of which return no charge. Independent replay checks every route, acceptance decision and outcome.

Whole-plan single agent with an eficiency paragraph. This variant adds to the prompt of the whole-plan single agent the paragraph

Work through the routes eficiently, avoiding repeated calculations or restating the same plan. Once you have a concrete candidate, return its JSON so the checker can provide feedback for the next improvement.

and raises its patience to fifteen answers without improvement, with a ceiling of sixty answers. All ten runs end normally and are independently replayed. One run reaches a safe plan, with $F = 1 5 4 ;$ ; five deliver all nine robots but keep one to five collisions, and four never accept a route. The mean final loss is $8 6 0 . 0 0 \pm 5 5 5 . 4 0$ and the search time $7 2 8 . 5 2 \pm 3 6 6 . 7 6$ seconds, with $2 6 . 4 \pm 1 5 . 5$ answers per run (Table D.7). Table D.4 lists every run, next to the agents that replace only some routes (below).

Reported costs total USD 2.22; one request returned no charge.

With patience five and no eficiency paragraph, the wholeplan agent ends at 841.00 ± 483.14 after 281.36 ± 111.67 seconds, also with one safe run. The longer patience therefore buys more search time but not a better final plan: writing all nine routes in one answer remains the bottleneck. Letting the agent change only some routes per answer, with the same prompt and patience, raises safe completion to seven or eight runs of ten with up to two or three routes, and to nine with one route per answer (below).

<table><tr><td></td><td colspan="4">1 route per answer</td><td colspan="4">Up to 2 routes</td><td colspan="4">Up to 3 routes</td><td colspan="4">All nine routes</td></tr><tr><td>Seed</td><td>F</td><td>Coll.</td><td>Ans.</td><td>Search (s)</td><td>F</td><td>Coll.</td><td>Ans.</td><td>Search (s)</td><td>F</td><td>Coll.</td><td>Ans.</td><td>Search (s)</td><td>F</td><td>Coll.</td><td>Ans.</td><td>Search (s)</td></tr><tr><td>0</td><td>152</td><td>0</td><td>16</td><td>241.8</td><td>152</td><td>0 1</td><td>18</td><td>298.3</td><td>160</td><td>0</td><td>27</td><td>637.7</td><td>1476*</td><td>0</td><td>15</td><td>403.4</td></tr><tr><td>1</td><td>161</td><td>0</td><td>35</td><td>742.6</td><td>267</td><td></td><td>25</td><td>448.3</td><td>153</td><td>0</td><td>19</td><td>402.9</td><td>259</td><td>1</td><td>60</td><td>1502.2</td></tr><tr><td>2</td><td>152</td><td>0</td><td>32</td><td>577.1</td><td>170</td><td>0</td><td>25</td><td>778.1</td><td>152</td><td>0</td><td>17</td><td>331.2</td><td>1476*</td><td>0</td><td>15</td><td>395.3</td></tr><tr><td>3</td><td>150</td><td>0</td><td>43</td><td>814.7</td><td>152</td><td>0</td><td>22</td><td>396.1</td><td>152</td><td>0</td><td>9</td><td>161.9</td><td>1476*</td><td>0</td><td>15</td><td>469.2</td></tr><tr><td>4</td><td>151</td><td>0</td><td>51</td><td>1153.7</td><td>152</td><td>0</td><td>13</td><td>194.5</td><td>152</td><td>0</td><td>10</td><td>168.6</td><td>691</td><td>5</td><td>18</td><td>395.7</td></tr><tr><td>5</td><td>156</td><td>0</td><td>91</td><td>1750.3</td><td>277</td><td>1</td><td>41</td><td>822.9</td><td>150</td><td>0</td><td>47</td><td>1269.7</td><td>365</td><td>2</td><td>29</td><td>917.9</td></tr><tr><td>6</td><td>272</td><td>1</td><td>26</td><td>372.1</td><td>150</td><td>0</td><td>22</td><td>384.4</td><td>164</td><td>0</td><td>21</td><td>682.0</td><td>656</td><td>5</td><td>22</td><td>755.4</td></tr><tr><td>7</td><td>149</td><td>0</td><td>26</td><td>289.0</td><td>153</td><td>0</td><td>12</td><td>255.0</td><td>263</td><td>1</td><td>31</td><td>745.0</td><td>154</td><td>0</td><td>47</td><td>1120.0</td></tr><tr><td>8</td><td>172</td><td>0</td><td>37</td><td>676.1</td><td>152</td><td>0</td><td>26</td><td>565.8</td><td>196</td><td>0</td><td>32</td><td>631.8</td><td>1476*</td><td>0</td><td>15</td><td>566.0</td></tr><tr><td>9</td><td>153</td><td>0</td><td>46</td><td>999.7</td><td>257</td><td>1</td><td>42</td><td>751.3</td><td>264</td><td>1</td><td>27</td><td>560.0</td><td>571</td><td>4</td><td>28</td><td>760.0</td></tr></table>

Table D.4: Single agents with the eficiency paragraph and patience fifteen, by the number of routes an answer may replace: independently replayed final plans of all forty runs. Coll.: collisions in the final plan; Ans.: checked answers. <sup>\*</sup>: no route was ever accepted, so the plan is the initial one and delivers no robot; every other final plan delivers all nine. Runs with a safe plan and $F \le 1 5 3$ stopped early, seed 1 of the whole-plan agent at its ceiling of sixty answers, and all others after fifteen answers without improvement. The whole-plan agent’s elapsed time, which adds retries, exceeds its search time by less than 5 s in every run except seed 2 (75 s).

Single agents that replace some routes. These agents still observe and score the complete nine-robot problem but replace at most $K \in \{ 1 , 2 , 3 \}$ routes per answer and keep the others unchanged. At every answer the agent sees all nine starts, goals and accepted arrays, plus the full global route and collision feedback. It chooses up to � distinct robot IDs, independent of the team’s fixed groups. Arrays for omitted IDs remain exactly unchanged. The selected replacement arrays are merged literally into the accepted plan and checked together. A geometry error in any selected array rejects the entire proposal. Otherwise, the merged plan is accepted immediately if and only if its full global � strictly decreases. Collisions can occur in accepted intermediate plans if their total loss improves; safe completion always requires zero collisions.

A run stops after fifteen consecutive checkable answers without an accepted improvement, or once its plan is safe with $F \le 1 5 3$ . The safety ceiling is sixty checked answers for � = 3 and 120 for $K \in \{ 1 , 2 \}$ , because one route per answer needs more answers; the $K = 3$ runs used at most 47, so the ceiling never stopped them. There are no client output or reasoning-token caps. Geometry-invalid but schemavalid answers consume checked attempts and search time; malformed answers and transport failures are retried with the same prompt and do not.

The shared coordinate, map, loss, collision and global feedback instructions are those of the team prompt, and the eficiency paragraph quoted above is included. The scope instruction reads:

You are one agent controlling all nine robots in robots. Propose a partial update to their delivery programs. . . . All nine robots, their starts, goals and current programs are visible to you.

Only the following sentences depend on �. For � = 3 they are

Choose from one to three distinct robot IDs in each response. Return complete replacement arrays ONLY for those chosen IDs. . . . Final answer: exactly one JSON object with key programs mapping one to three selected robot IDs to their new complete literal arrays of command names.

For � = 2 they read “Choose one or two distinct robot IDs in each response” and “mapping one or two selected robot IDs”. For � = 1 they read “Choose exactly one robot ID in each response. Return a complete replacement array ONLY for that chosen ID” and “mapping exactly one selected robot ID”. The structured task records max\_updated\_robots= � and the matching output\_scope. Before the $K \in \{ 1 , 2 \}$ runs, an ofline check confirmed that the code produces, for $K = 3$ exactly the first prompt recorded in the � = 3 runs.

Table D.7 summarizes the five single-agent protocols and Table D.4 lists every partial-update run.

∙ One route per answer: nine of ten runs are safe and six stop early; final loss $1 6 6 . 8 0 \pm 3 7 . 6 0 ;$ search time 761.71±456.76 seconds with $4 0 . 3 { \pm } 2 0 . 6 $ answers.

∙ Up to two routes: seven runs are safe and six stop early; final loss $1 8 8 . 2 0 \pm 5 4 . 8 7 ;$ search time $4 8 9 . 4 7 \pm$ 228.26 seconds with 24.6 ± 10.1 answers.

∙ Up to three routes: eight runs are safe and five stop early; final loss 180.60 ± 45.75; search time 559.08 ± 325.35 seconds with $2 4 . 0 \pm 1 1 . 4$ answers.

All partial-update protocols reduce the gap at a similar rate during the first 300 seconds; afterwards the one-route agent continues to improve, while runs with larger answers more often stall with a collision. The whole-plan agents stall far higher.

Requiring a complete answer for all nine routes proved dificult: the whole-plan single agent is safe in one run of ten, with or without the eficiency paragraph. Smaller answers make the single agent more reliable: with one route per answer it is safe in nine runs of ten and reaches the lowest mean loss of all single-agent protocols. The price is time: it needs about

40 answers per run, and its median time to a safe plan with � ≤ 153 is 696 seconds, against 341 and 331 seconds with up to two and three routes. We therefore use the one-route protocol as the single-agent baseline, with prompt v2 in the main text. With prompt v1, it reduces the gap faster at first than the v1 team, but the team’s mean loss is lower after about 560 seconds; the v1 team is safe in all ten runs, its search takes 589 ± 65 seconds against 762 ± 457, and its final loss varies far less (156.50 ± 5.60).

Team with the eficiency paragraph. This variant adds the same paragraph to the team prompt and changes nothing else; without the paragraph, its first three requests are those of the v1 team. All ten runs end normally and are independently replayed. Every final plan delivers all nine robots, and nine are safe; seed 8 keeps one collision. The mean final loss is 167.40 ± 34.15 and the search time 436.65 ± 90.79 seconds, with 51.8 ± 10.8 answers per run (Table D.7). Table D.5 lists every run, together with the team with partial updates (below).

The paragraph is the only diference from the v1 team. It shortens the mean search time from 589 to 437 seconds and lowers the reported cost per run from USD 0.28 to 0.23, while nine of ten runs remain safe; the mean final loss rises from 156.5 to 167.4, mostly because of the one unsafe run. The prompt therefore changes the team’s results only slightly.

Team with partial route updates. Here each agent may replace the routes of one, two or three of its own robots instead of returning all three; arrays for omitted robots are copied exactly from the frozen plan, separate subset proposals are not combined, and correction, verification and the single best commit are unchanged. The original instruction was:

Return complete arrays for EVERY owned ID, even if unchanged.

Its replacement is:

Choose from one to three distinct IDs from owned in each response. Return complete replacement arrays ONLY for those chosen IDs. All omitted owned robots keep their installed current arrays exactly unchanged. You cannot change robots outside owned. Each private attempt starts from the same current arrays shown below.

The opening instruction becomes:

Propose a partial update to delivery programs for your owned robots in robots.

The final-answer instruction becomes:

Final answer: exactly one JSON object with key programs mapping one to three selected owned IDs to a literal array of command names. No shorthand, loops, code or alternative answers.

The structured task adds two fields:

∙ max\_updated\_robots is 3;

∙ output\_scope has the exact value “One to three selected owned IDs only; omitted arrays remain exactly current.”

All other instructions, including ownership, objective and correction, are unchanged.

All ten runs end normally, six by early stop and four at the eight-round cap. All deliver nine robots, but replicates 2, 7 and 9 each retain one collision, giving seven safe runs. Replicates 2 and 7 retain an edge swap between $r _ { 4 }$ and $r _ { 8 } .$ which belong to the same agent, at $t = 6 ;$ replicate 9 retains a collision between $r _ { 6 }$ and �<sub>7</sub>, which belong to diferent agents, at � = 13. The mean final loss is 182.40±48.08 and the search time 310.40 ± 94.67 seconds (Tables D.7 and D.5).

Partial updates halve the team’s mean search time (310 versus 589 seconds). In exchange, three runs end with a collision, and the mean final loss rises from 156.5 to 182.4. Partial updates thus trade reliability for speed for the team. Because a team’s value lies in solving the task reliably, we keep the team that rewrites all its routes as the main protocol.

## D.3 Why prompt v1 stalled

With prompt v1, six of the thirty partial-update single-agent runs ended with exactly one collision and fifteen answers without improvement. An ofline check with the exact singlerobot best response (Appendix C.7) shows that in each of these six final plans one of the two colliding robots could be rerouted, with the other eight routes fixed, removing the collision without delaying that robot’s arrival; in three of them the best such route even arrives earlier than the accepted one. Five of the six collisions are head-on: two robots move toward each other along the same row or column, so waiting only moves the point where they meet. The ninety answers given in these stalled phases show two failure modes. First, 56 of the 113 edited routes keep exactly the same cells and only insert or remove waits. Second, the 20 answers that did remove the original collision all created a new one with another robot. Prompt v1 listed every robot’s commands but gave tick-by-tick positions only for the robots involved in a reported collision, and drew collision frames only for the previous attempt; to check a detour against the other robots, the model had to simulate their commands itself. The map had no column labels, and the rule that a swap between ticks � − 1 and � is reported at tick � was not stated.

In prompt v2 we therefore changed the presentation, not the information structure, of both prompts. Asked again in the six stalled states, the single-agent prompt v2 produced an accepted improvement in 21 of 60 answers (ten per state, with and without the list of rejected attempts), including at least one in every state, against none of the ninety recorded answers with prompt v1. The team prompt had the same presentation, and only 22 of the 207 private searches that could improve their block reached the exact corrected block minimum.

## D.4 Prompt v2

Both prompts state the command, timing, collision and loss rules once, without strategy advice: no route, distance to the optimum or repair rule is supplied. They then present the checked state:

<table><tr><td colspan="6"></td><td colspan="6">Partial route updates</td></tr><tr><td>Seed</td><td>F</td><td>Coll.</td><td>Rounds</td><td>Search (s)</td><td>USD</td><td>F</td><td>Coll.</td><td>Rounds</td><td>Ans.</td><td>Search (s)</td><td>USD</td></tr><tr><td>0</td><td>150</td><td>0</td><td>7</td><td>400.19</td><td>0.23792</td><td>152</td><td>0</td><td>5</td><td>43</td><td>257.44</td><td>0.15622</td></tr><tr><td>1</td><td>149</td><td>0</td><td>5</td><td>276.32</td><td>0.16257</td><td>152</td><td>0</td><td>6</td><td>51</td><td>320.67</td><td>0.19878</td></tr><tr><td>2</td><td>154</td><td>0</td><td>8</td><td>501.97</td><td>0.31963</td><td>248</td><td>1</td><td>8</td><td>58</td><td>489.37</td><td>0.26790</td></tr><tr><td>3</td><td>188</td><td>0</td><td>8</td><td>567.83</td><td>0.32965</td><td>150</td><td>0</td><td>4</td><td>33</td><td>212.23</td><td>0.12381</td></tr><tr><td>4</td><td>161</td><td>0</td><td>8</td><td>429.93</td><td>0.17183</td><td>153</td><td>0</td><td>6</td><td>44</td><td>268.90</td><td>0.16628</td></tr><tr><td>5</td><td>157</td><td>0</td><td>8</td><td>514.01</td><td>0.24439</td><td>164</td><td>0</td><td>8</td><td>57</td><td>373.95</td><td>0.20821</td></tr><tr><td>6</td><td>152</td><td>0</td><td>6</td><td>382.34</td><td>0.19034</td><td>148</td><td>0</td><td>4</td><td>27</td><td>156.81</td><td>0.08337</td></tr><tr><td>7</td><td>152</td><td>0</td><td>8</td><td>515.22</td><td>0.27382</td><td>248</td><td>1</td><td>8</td><td>46</td><td>291.64</td><td>0.14034</td></tr><tr><td>8</td><td>259</td><td>1</td><td>8</td><td>444.37</td><td>0.19429</td><td>150</td><td>0</td><td>8</td><td>39</td><td>360.10</td><td>0.15127</td></tr><tr><td>9</td><td>152</td><td>0</td><td>6</td><td>334.36</td><td>0.19404</td><td>259</td><td>1</td><td>8</td><td>49</td><td>372.92</td><td>0.19586</td></tr></table>

Table D.5: Two variants of the v1 team: independently replayed final plans of all ten runs of each. Every final plan delivers all nine robots. Coll.: collisions in the final plan; Ans.: checked answers, summed over agents; USD: reported cost. With partial updates, seed 0 is the adopted pilot; the runs with a safe plan and $F \le 1 5 3$ stopped early, the others at the eight-round cap, and elapsed time exceeds search time by less than 3 s in every run except seed 0 (31 s).

![](images/2df967a1ab1973d8bb34045c72479ea546ec5d007f5224057e562ede454818ee.jpg)  
(a) Search quality

![](images/c785776c5ac3b4b1f92a0416f92c9638d8b907fe8f3f675efb19f4d2f41c7cfa.jpg)  
(b) Correction error

![](images/75f24ff042f0ed225d412f4aaac28a84821f0d432cd5a26e814608ba685bb713.jpg)  
(c) Descent coefficient

![](images/81b2b5529bc39d67b631fa08627f8a5d63bf67add3e3a4623086b4fb8f63384f.jpg)  
C One run Mean ± sample SD  
(d) Block geometry  
Figure D.2: Measured constants of the team per run for four prompt settings: v1 and v1 with the eficiency paragraph on map 1 (ten runs each), v2 on map 1 and on map 2 (twenty runs each). Panels (a)–(c) average within a run; (d) takes the smallest ratio $\sum _ { b } D _ { b } / \Delta$ over the states of the run that start a round. Diamonds and bars: mean ± sample SD, not confidence intervals. Since every true-minimizer error is zero, a run’s mean � equals its mean error at returned proposals.

∙ the map with column and row labels, and the list of wall cells;

∙ every controlled robot’s start, goal, current program, arrival tick and cost;

∙ a timeline with the position of every visible robot at every tick, marking robots that have arrived;

∙ every visible collision of the accepted plan, described in words (for example, “tick 7: r2 moves $[ 7 , 6 ]  [ 8 , 6 ]$ while r7 moves $[ 8 , 6 ]  [ 7 , 6 ] ^ { 5 ) } )$ and drawn as maps of all visible robots at the tick before and the tick of the collision;

∙ the attempts already checked against the accepted plan, with their losses and collisions.

The single agent sees all nine robots, the global loss and the stopping threshold, and lists the attempts rejected since the plan was last improved. A team agent sees only its three robots and its known peers, exactly as with prompt v1: robots of other agents whose collisions with its robots the coordinator has already found, with their current fixed timelines (Section 3.1). Other robots appear in no table, drawing or collision list. A team agent sees neither the global loss nor the threshold; it lists its earlier attempts in the current round and, if its previous proposal collided in the coordinator’s global check, those collisions. Without correction, a team agent sees only its own robots. Answers have the same format as with prompt v1: complete literal command arrays for the chosen robots.

## D.5 Efect of the prompt version

Figure D.1(c) compares the two prompt versions. For the single agent with one route per answer, prompt v2 raises safe completion from nine of ten runs to all twenty, lowers the final loss from 166.80 ± 37.60 to $1 5 6 . 0 0 \pm 1 0 . 5 3$ , and shortens the search from 761.7 to 613.0 seconds on average with fewer answers (27.8 against 40.3). For the team, it shortens the search from 589.4 to 481.0 seconds and lowers the final loss from $1 5 6 . 5 0 \pm 5 . 6 0$ to $1 4 9 . 9 0 \pm 2 . 6 7 \mathrm { : }$ ; both are safe in every run, and the v2 team reaches the threshold in all twenty runs against three of ten. The team’s stopping rule changed as well: the v1 team stopped after at most eight rounds, which ended seven of its ten runs before the plan reached $F \leq 1 5 3 ,$ four of them right after an improving round, whereas the v2 team may run up to thirty rounds; this comparison therefore also reflects that change. Geometrically invalid answers did not become rarer: they are 17% of the checked answers for both v2 protocols, against 12% for the single agent and 11% for the team with v1. The gains come from repairs: the v2 team’s searches reach the exact corrected block minimum in 82 of the 343 searches that can improve their block (22 of 207 with v1) and capture on average a fraction 0.66 of the available corrected improvement, pooled over searches (0.58 with v1).

<table><tr><td>Per-run statistic</td><td>v1</td><td> ${ \mathrm {  ~ v 1 ~ } } + { \mathrm {  ~ p a r . } }$ </td><td>v2, map 1 v2, map 2</td><td></td></tr><tr><td>Mean Q</td><td> $0 . 5 9 \pm 0 . 0 8$ </td><td> $0 . 6 3 \pm 0 . 1 1$ </td><td> $0 . 7 0 \pm 0 . 1 5$ </td><td> $0 . 6 9 \pm 0 . 1 4$ </td></tr><tr><td>Mean B</td><td> $4 7 . 4 \pm 1 2 . 7$ </td><td> $4 9 . 2 \pm 1 8 . 3$ </td><td> $4 9 . 7 \pm 1 6 . 5$ </td><td> $4 0 . 4 \pm 1 9 . 5$ </td></tr><tr><td>Mean α</td><td> $0 . 2 5 \pm 0 . 0 4$ </td><td> $0 . 2 6 \pm 0 . 0 7$ </td><td> $0 . 2 7 \pm 0 . 0 7$ </td><td> $0 . 2 6 \pm 0 . 0 8$ </td></tr><tr><td>Smallest ratio</td><td> $0 . 8 1 \pm 0 . 1 5$ </td><td> $0 . 8 8 \pm 0 . 1 2$ </td><td> $0 . 9 2 \pm 0 . 1 4$ </td><td> $0 . 9 3 \pm 0 . 1 0$ </td></tr></table>

Table D.6: Mean ± sample SD over runs (ten for v1 settings, twenty for v2). Smallest ratio: the minimum of $\sum _ { b } D _ { b } / \Delta$ over the states of a run that start a round; its mean is not a certificate over the union of runs.

Applying the audit of Appendix E.7 to every team run with a full set of block references shows how the prompt moves the measured constants (Figure D.2, Table D.6): the v1 team, the v1 team with the eficiency paragraph (72 rounds, 216 returned searches, 518 checked answers) and the v2 team on both maps. Averaged per run, the realized search quality rises from 0.59±0.08 with v1 to 0.63±0.11 with the paragraph and to 0.70 ± 0.15 and 0.69 ± 0.14 with v2 on the two maps, while the error term stays at the same level $( 4 7 . 4 \pm 1 2 . 7 , 4 9 . 2 \pm 1 8 . 3$ 49.7±16.5 and 40.4±19.5) and the descent coeficient changes little (0.25, 0.26, 0.27 and 0.26). The error at every true block minimizer is zero for every prompt, so � comes entirely from returned proposals. The smallest geometry ratio at the states that start a round is at least 0.6, 0.69, 0.43 and 0.70 in every run, with means between 0.81 and 0.93. With the eficiency paragraph, the per-round recurrence holds in all 72 rounds and ends at a mean gap bound of 267.46 against an observed mean gap of 23.40. A better prompt thus raises search quality and changes which states are visited, while the correction errors stay at the same level.

## D.6 The protocols of the main text

Three observations explain the choice of protocols in the main text. First, a single agent that rewrites all nine routes in every answer rarely reaches a safe plan, with or without the eficiency paragraph and longer patience. Letting it update only some robots per answer, while it still sees and scores the whole plan, raises safe completion to seven or eight runs of ten with up to two or three robots, and to nine with exactly one; with prompt v2, one route per answer is safe in all twenty runs and exactly three routes per answer in nine of ten. The one-route agent is therefore the single-agent baseline. Second, with prompt v1 a partial-update option makes the team faster but less reliable, with three runs ending in a collision, two of them between robots of the same agent; the team that rewrites all its routes is kept for the main text. The eficiency paragraph shortens the v1 team’s search (437 versus 589 seconds) but leaves one run unsafe and raises its mean loss to 167.4. Third, prompt v2 shortens the search of both the one-route single agent and the team, and the team without correction is rarely safe with either prompt. With prompt v2, three one-route single agents searching in parallel are faster than the team with Algorithm 1 on both maps, while the team ends with the best plans. On two floors, six such agents are faster than the team in mean but not in median, the single agent needs about twice the team’s search time, and the team again ends with the best plans. The main text therefore uses prompt v2 with the one-route single agent, its best-of-� as the parallelism-matched control, the team that rewrites all three of an agent’s routes, with Algorithm 1 and with the best-combination and unsafe merges, and the team without correction; the sequential replay is reported in Appendix E.10.

## E Routing results with prompt v2

This appendix reports every run behind Figure 3 and Table 1, all with prompt v2 (Appendix D): on the first map, twenty runs each for the single agent with one route per answer, the best-of-three such agents and the team, and ten each for the single agent with exactly three routes per answer and the team without correction; on the second map, twenty each for the single agent, the best-of-three and the team, and ten for the team without correction; on two floors, twenty each for the single agent and the best-of-six, with the team variants evaluated on the runs of both maps; and for two other merge rules of the team, best combination and unsafe, twenty runs each on both maps. It then shows recorded collision repairs, audits the assumptions of Theorem 2 on the team runs, and evaluates the trajectory-wise bound.

## E.1 Protocols and measurement

The team protocol is that of Appendix C: three agents with fixed blocks, up to $H = 3$ answers per agent and round on a frozen base, one accepted block per round, and the certificate-based early stop of a private search. The team stops once the plan is safe and within 9 of the optimum $( F \leq 1 5 3$ on the first map), after five rounds without an improving commit (fifteen answers per agent, as many as the single agent), or after thirty rounds. The single agent stops once the plan is safe and within the threshold, after fifteen consecutive answers without improvement, or after 120 answers; it either replaces one route per answer or returns complete programs for exactly three robots of its choice per answer, the size of a team agent’s answer (a program may be repeated unchanged). An answer with another number of robots is a format failure. The best-of-three runs three copies of the one-route single agent in parallel: every round, all copies receive the same prompt with their own seeds (copy � in round � adds $1 0 0 0 r + c )$ , the best strictly improving answer is accepted and the other two are discarded; if no answer improves, all three enter the list of rejected attempts. Its round time is the slowest of the three answers, as for the team, and it stops once the plan is safe and within the threshold, after five rounds without improvement (fifteen answers) or after forty rounds.

Local Work, Global Progress
<table><tr><td>Variant</td><td>Appendix</td><td>Safe</td><td>Final F</td><td>Search time (s)</td><td>Answers</td><td>Reported USD/run</td></tr><tr><td colspan="7">Prompt v2, map 1 (main text)</td></tr><tr><td>Single agent, 1 route per answer</td><td>EEE</td><td>20/20</td><td> $1 5 6 . 0 0 \pm 1 0 . 5 3$ </td><td> $6 1 3 . 0 1 \pm 3 9 7 . 8 2$ </td><td> $2 7 . 8 0 \pm 1 1 . 6 7$ </td><td> $0 . 1 6 1 \pm 0 . 1 0 1$ </td></tr><tr><td>Single agent, exactly 3 routes per answer</td><td></td><td>9/10</td><td> $2 2 0 . 8 0 \pm 2 2 0 . 3 8$ </td><td> $5 6 1 . 2 4 \pm 2 5 3 . 4 5$ </td><td> $2 0 . 4 0 \pm 9 . 8 9$ </td><td> $0 . 1 4 7 \pm 0 . 0 6 5$ </td></tr><tr><td>Best-of-3 one-route single agents</td><td></td><td>19/20</td><td>162.90 ± 48.82</td><td> $3 2 6 . 4 1 \pm 1 9 8 . 1 0$ </td><td> $3 9 . 3 0 \pm 1 1 . 3 0$ </td><td> $0 . 1 8 8 \pm 0 . 1 0 8$ </td></tr><tr><td>Team (Alg. 1)</td><td>E</td><td>20/20</td><td> $1 4 9 . 9 0 \pm 2 . 6 7$ </td><td> $4 8 1 . 0 4 \pm 2 2 7 . 3 2$ </td><td> $4 7 . 1 0 \pm 1 6 . 0 8$ </td><td> $0 . 2 2 8 \pm 0 . 1 0 6$ </td></tr><tr><td>Team (Alg. 2)</td><td>E.5</td><td>20/20</td><td> $1 4 9 . 4 0 \pm 3 . 2 2$ </td><td> $3 3 9 . 7 4 \pm 1 9 6 . 9 1$ </td><td> $3 6 . 2 0 \pm 1 1 . 8 8$ </td><td> $0 . 1 8 1 \pm 0 . 0 7 4$ </td></tr><tr><td>Team unsafe</td><td>E.5</td><td>20/20</td><td> $1 4 9 . 0 0 \pm 2 . 5 5$ </td><td> $6 0 1 . 4 3 \pm 2 8 4 . 4 9$ </td><td> $6 4 . 5 5 \pm 2 5 . 9 3$ </td><td> $0 . 3 5 7 \pm 0 . 1 5 4$ </td></tr><tr><td>Team without correction</td><td>E</td><td>0/10</td><td> $4 5 7 . 1 0 \pm 1 0 9 . 1 7$ </td><td> $3 2 6 . 1 9 \pm 1 7 1 . 9 6$ </td><td> $3 7 . 0 0 \pm 2 1 . 4 2$ </td><td> $0 . 1 3 3 \pm 0 . 0 7 7$ </td></tr><tr><td>Sequential replay of the team</td><td>E</td><td>20/20</td><td> $1 4 9 . 9 0 \pm 2 . 6 7$ </td><td> $8 5 3 . 9 5 \pm 3 7 4 . 2 3$ </td><td> $4 7 . 1 0 \pm 1 6 . 0 8$ </td><td></td></tr><tr><td colspan="7">Prompt v2, map 2</td></tr><tr><td>Single agent, 1 route per answer</td><td>E.3</td><td>19/20</td><td> $1 6 6 . 5 0 \pm 4 4 . 3 8$ </td><td> $5 3 7 . 9 1 \pm 3 6 5 . 7 1$ </td><td> $2 4 . 5 0 \pm 1 1 . 2 5$ </td><td> $0 . 1 3 7 \pm 0 . 0 9 3$ </td></tr><tr><td>Best-of-3 one-route single agents</td><td>E.3</td><td>18/20</td><td> $1 6 8 . 5 5 \pm 3 3 . 9 2$ </td><td> $3 2 6 . 2 9 \pm 1 4 7 . 0 1$ </td><td> $3 8 . 7 0 \pm 9 . 5 8$ </td><td> $0 . 1 9 5 \pm 0 . 0 8 9$ </td></tr><tr><td>Team (Alg. 1)</td><td>E.3</td><td>20/20</td><td> $1 5 4 . 1 5 \pm 1 2 . 5 6$ </td><td> $4 8 2 . 2 6 \pm 2 0 1 . 5 8$ </td><td> $4 4 . 7 5 \pm 1 6 . 0 6$ </td><td> $0 . 2 2 1 \pm 0 . 0 9 3$ </td></tr><tr><td>Team (Alg. 2)</td><td>E.5</td><td>20/20</td><td> $1 5 9 . 7 5 \pm 1 7 . 2 1$ </td><td> $3 1 0 . 4 4 \pm 2 0 4 . 0 6$ </td><td> $3 3 . 0 0 \pm 1 5 . 5 1$ </td><td> $0 . 1 7 4 \pm 0 . 1 3 5$ </td></tr><tr><td>Team unsafe</td><td>E.5</td><td>19/20</td><td> $1 5 7 . 0 5 \pm 2 2 . 2 6$ </td><td> $4 2 3 . 6 6 \pm 2 3 2 . 4 1$ </td><td> $4 7 . 9 5 \pm 2 2 . 6 3$ </td><td> $0 . 2 3 5 \pm 0 . 1 3 0$ </td></tr><tr><td>Team without correction Sequential replay of the team</td><td>E.3 E.3</td><td>2/10 20/20</td><td> $3 2 8 . 4 0 \pm 1 5 3 . 4 1$ </td><td> $1 7 8 . 0 9 \pm 8 7 . 2 4$ </td><td> $1 9 . 0 0 \pm 7 . 1 8$ </td><td> $0 . 0 6 9 \pm 0 . 0 3 1$ </td></tr><tr><td></td><td></td><td></td><td> $1 5 4 . 1 5 \pm 1 2 . 5 6$ </td><td> $7 7 8 . 1 6 \pm 3 0 4 . 6 2$ </td><td> $4 4 . 7 5 \pm 1 6 . 0 6$ </td><td></td></tr><tr><td colspan="7">Prompt v2, two floors (eighteen robots)</td></tr><tr><td>Single agent, 1 route per answer</td><td>E.4</td><td>20/20</td><td> $3 1 9 . 6 0 \pm 2 3 . 3 1$ </td><td> $1 1 4 8 . 6 7 \pm 4 1 9 . 6 1$ </td><td> $6 3 . 2 0 \pm 1 6 . 1 6$ </td><td> $0 . 3 6 5 \pm 0 . 1 2 7$ </td></tr><tr><td>Best-of-6 one-route single agents</td><td>E.4</td><td>20/20</td><td> $3 1 2 . 4 0 \pm 1 6 . 2 6$ </td><td> $5 1 6 . 8 4 \pm { 1 2 3 . 8 7 }$ </td><td> $1 4 2 . 5 0 \pm 1 7 . 0 7$ </td><td> $0 . 6 8 0 \pm 0 . 1 5 4$ </td></tr><tr><td>Team  $\left( \mathrm { A l g } . \ 1 \right)$ </td><td>E.4</td><td>20/20</td><td> $3 0 4 . 0 5 \pm 1 2 . 7 3$ </td><td> $6 5 6 . 3 7 \pm 2 0 6 . 1 1$ </td><td> $9 1 . 8 5 \pm 2 0 . 7 3$ </td><td> $0 . 4 4 9 \pm 0 . 1 4 1$ </td></tr><tr><td>Team  $( { \mathrm { A l g } } , 2 )$ </td><td>E.5</td><td>20/20</td><td> $3 0 9 . 1 5 \pm 1 7 . 3 9$ </td><td> $4 5 1 . 6 4 \pm 2 2 7 . 1 0$ </td><td> $6 9 . 2 0 \pm 1 9 . 7 7$ </td><td> $0 . 3 5 5 \pm 0 . 1 5 8$ </td></tr><tr><td>Team unsafe</td><td>E.5</td><td>19/20</td><td> $3 0 6 . 0 5 \pm 2 2 . 5 4$ </td><td> $7 1 6 . 8 2 \pm 2 6 2 . 9 7$ </td><td> $1 1 2 . 5 0 \pm 2 8 . 7 8$ </td><td> $0 . 5 9 2 \pm 0 . 1 7 2$ </td></tr><tr><td>Team without correction</td><td>E.4</td><td>0/10</td><td> $7 8 5 . 5 0 \pm 1 8 1 . 9 0$ </td><td> $3 8 2 . 0 5 \pm 1 7 6 . 1 4$ </td><td> $5 6 . 0 0 \pm 1 9 . 9 0$ </td><td> $0 . 2 0 2 \pm 0 . 0 7 9$ </td></tr><tr><td>Sequential replay of the team</td><td>E.4</td><td>20/20</td><td> $3 0 4 . 0 5 \pm 1 2 . 7 3$ </td><td> $1 6 3 2 . 1 0 \pm 4 6 8 . 5 2$ </td><td> $9 1 . 8 5 \pm 2 0 . 7 3$ </td><td></td></tr><tr><td colspan="7">Prompt v1, map 1</td></tr><tr><td>Single agent, whole plan, patience 5</td><td>D</td><td>1/10</td><td> $8 4 1 . 0 0 \pm 4 8 3 . 1 4$ </td><td> $2 8 1 . 3 6 \pm 1 1 1 . 6 7$ </td><td> $8 . 3 0 \pm 2 . 7 1$ </td><td> $0 . 0 7 1 \pm 0 . 0 2 7$ </td></tr><tr><td>Single agent, whole plan, efficiency paragraph</td><td>D</td><td>1/10</td><td> $8 6 0 . 0 0 \pm 5 5 5 . 4 0$ </td><td> $7 2 8 . 5 2 \pm 3 6 6 . 7 6$ </td><td> $2 6 . 4 0 \pm 1 5 . 5 4$ </td><td> $0 . 2 2 2 \pm 0 . 1 1 1$ </td></tr><tr><td>Single agent, up to 3 routes per answer</td><td>D</td><td>8/10</td><td> $1 8 0 . 6 0 \pm 4 5 . 7 5$ </td><td> $5 5 9 . 0 8 \pm 3 2 5 . 3 5$ </td><td> $2 4 . 0 0 \pm 1 1 . 3 7$ </td><td> $0 . 1 5 2 \pm 0 . 0 8 7$ </td></tr><tr><td>Single agent, up to 2 routes per answer</td><td>D</td><td>7/10</td><td> $1 8 8 . 2 0 \pm 5 4 . 8 7$ </td><td> $4 8 9 . 4 7 \pm 2 2 8 . 2 6$ </td><td> $2 4 . 6 0 \pm 1 0 . 1 3$ </td><td> $0 . 1 3 6 \pm 0 . 0 6 6$ </td></tr><tr><td>Single agent, 1 route per answer</td><td>D</td><td>9/10</td><td> $1 6 6 . 8 0 \pm 3 7 . 6 0$ </td><td> $7 6 1 . 7 1 \pm 4 5 6 . 7 6$ </td><td> $4 0 . 3 0 \pm 2 0 . 6 3$ </td><td> $0 . 2 0 8 \pm 0 . 1 2 5$ </td></tr><tr><td>Team (Alg. 1)</td><td>D</td><td>10/10</td><td> $1 5 6 . 5 0 \pm 5 . 6 0$ </td><td> $5 8 9 . 3 9 \pm 6 4 . 8 5$ </td><td> $5 9 . 3 0 \pm 7 . 9 2$ </td><td> $0 . 2 8 5 \pm 0 . 0 4 2$ </td></tr><tr><td>Team (Alg. 1), efficiency paragraph</td><td>D</td><td>9/10</td><td> $1 6 7 . 4 0 \pm 3 4 . 1 5$ </td><td> $4 3 6 . 6 5 \pm 9 0 . 7 9$ </td><td> $5 1 . 8 0 \pm 1 0 . 7 9$ </td><td> $0 . 2 3 2 \pm 0 . 0 6 0$ </td></tr><tr><td>Team (Alg. 1), partial updates</td><td>D</td><td>7/10</td><td> $1 8 2 . 4 0 \pm 4 8 . 0 8$ </td><td> $3 1 0 . 4 0 \pm 9 4 . 6 7$ </td><td> $4 4 . 7 0 \pm 9 . 8 6$ </td><td> $0 . 1 6 9 \pm 0 . 0 5 1$ </td></tr><tr><td>Team without correction</td><td>D</td><td>1/10</td><td> $3 9 2 . 9 0 \pm 1 2 0 . 6 6$ </td><td> $3 3 7 . 9 7 \pm 9 5 . 3 4$ </td><td> $3 9 . 8 0 \pm 1 0 . 3 7$ </td><td> $0 . 1 3 2 \pm 0 . 0 3 7$ </td></tr><tr><td>Sequential replay of the team</td><td>D</td><td>10/10</td><td> $1 5 6 . 5 0 \pm 5 . 6 0$ </td><td> $1 2 0 7 . 1 4 \pm 1 6 5 . 6 1$ </td><td> $5 9 . 3 0 \pm 7 . 9 2$ </td><td></td></tr></table>

Table D.7: All routing variants, with prompt v2 of the main text and with prompt v1; twenty runs for the v2 one-route single agent, its best-of-three (best-of-six on two floors), the team with each of its three merge rules and its sequential replay on each task, ten for every other variant; mean ± sample SD over all runs, including unsafe ones. Safe: all robots delivered without collisions in the final plan. Search time ends at the stopping rule; for an unsafe run it is not a time to a solution. Answers counts checked answers per run, summed over agents for teams. Reported costs omit charges that the provider did not return. Each sequential replay reuses its team’s recorded answers, so it has no new cost.

maps, the provider stopped an answer at its 65,536-token output limit after the visible answer had degenerated into repetition; each such answer was kept, counted as a format failure, and re-asked once with the next retry seed. Search time is defined by (C.1); the sequential replay sums the three agents’ times in every round instead of taking the maximum. Every run was replayed independently from the recorded answers: proposals, block selection, accepted plans, losses, known pairs and clocks.

Model, provider and sampling settings are those of Appendix C.6. Replicate � uses base seed 4,000,000 + 100,000�; the team adds 1000� + 100ℎ + �, and the single agent adds 1000� for its �-th answer. Format failures are re-asked with a deterministic new seed, at most three times, and transport failures at most twenty times; neither counts toward search time. Fourteen times in about 5,400 requests on the two

## E.2 Results on the first map

Table E.1 lists every run; Table D.7 gives means and costs. Seeds 10–19 of the single agent with one route and of the team were run after seeds 0–9 with byte-identical code and settings; the best-of-three ran all twenty seeds with the same code.

∙ Single agent, one route per answer: safe in all twenty runs, sixteen of which stop at $F \le 1 5 3 ;$ final loss $1 5 6 . 0 0 \pm 1 0 . 5 3$ (median 153); search time $6 1 3 . 0 \pm 3 9 7 . 8$ seconds (median 448) with $2 7 . 8 \pm 1 1 . 7$ answers. Fifteen runs finish within 900 seconds; five need 955 to 1,532 seconds, three of which stop above the threshold, at 157, 160 and 191; the fourth run above the threshold stops at 180 after 855 seconds.

∙ Best-of-three one-route single agents: safe in nineteen runs, sixteen of which stop at $F \le 1 5 3 ;$ final loss $1 6 2 . 9 0 \pm 4 8 . 8 2$ (median 152); search time $3 2 6 . 4 \pm 1 9 8 . 1$ seconds (median 256) over $1 3 . 1 \pm 3 . 8$ rounds with $3 9 . 3 \pm 1 1 . 3$ answers. Seed 10 stops at $F = 3 7 0$ with two collisions; seeds 0, 11 and 12 stop at 154 to 158 after 587 to 934 seconds.

∙ Single agent, exactly three routes per answer (ten runs): safe in nine runs, all of which stop at $F \le 1 5 3 ;$ final loss $2 2 0 . 8 0 \pm 2 2 0 . 3 8$ (median 152); search time $5 6 1 . 2 \pm 2 5 3 . 5$ seconds with $2 0 . 4 \pm 9 . 9$ answers. In seed 2 the agent stopped at $F = 8 4 8 ,$ with four robots undelivered and one collision, after fifteen answers without improvement, eight of them geometrically invalid. Overall 37% of its answers are invalid, against 17% for the one-route agent, because one invalid route rejects the whole three-route answer.

∙ Team: safe in all twenty runs, all of which stop at $F \ \leq \ 1 5 3 ;$ final loss $1 4 9 . 9 0 \pm 2 . 6 7$ (median 150.5); search time $4 8 1 . 0 \pm 2 2 7 . 3$ seconds (median 440; two runs take longer than 700 seconds, at most 987) over $7 . 3 \pm 2 . 7$ rounds with $4 7 . 1 \pm 1 6 . 1$ answers. Seed 19 reaches the optimum $F = 1 4 4 ;$ ; the two longest runs need 12 and 13 rounds.

∙ Team without correction (ten runs): no run is safe; final loss $4 5 7 . 1 0 \pm 1 0 9 . 1 7 ,$ , search time 326.2 ± 172.0 seconds over 10.8 rounds on average. Every run delivers all nine robots and keeps one to five collisions, all between robots of diferent agents. Six runs end at a fixed point: every agent’s own robots take shortest routes without colliding with each other, so all three blocks are certified at the start of a round, no model is called, and every later round would repeat the plan. The other four stop after five rounds without improvement, still answering.

∙ Sequential replay of the team: $8 5 3 . 9 \pm 3 7 4 . 2$ seconds against 481.0±227.3 in parallel, a ratio of means of 1.78; answers, selections and losses are unchanged.

Statistical comparison. Two-sided permutation tests over the twenty runs of each protocol compare final losses and search times. The team’s final loss is lower than the single agent’s (diference of means 6.1, $p < 0 . 0 0 1$ ; of medians 2.5, $p = 0 . 0 0 5 )$ ; on every seed the single agent ends at least as high as the team, and higher on eighteen. Against the best-of-three, the diference of means is 13.0 $( p = 0 . 0 1 1 )$ and of medians 1.5 $( p = 0 . 0 9 )$ . The team reaches the threshold in all twenty runs, the two single-agent protocols in sixteen each $( p = 0 . 1 1 $ Fisher’s exact test). In search time, the single agent and the team have the same median (diference 7 seconds, $p = 0 . 9 7 )$ the single agent’s mean is higher by 132 seconds $( p = 0 . 2 1 )$ and its spread is larger (standard deviation 1.75 times the team’s, $p = 0 . 0 4 4 )$ . The best-of-three is faster than the team (diference of medians 184 seconds, $p = 0 . 0 1 1 $ ; of means 155 seconds, $p = 0 . 0 2 7 )$ , with a similar spread $( p = 0 . 6 2 )$ ; it is faster on fifteen of the twenty seeds.

## E.3 A second map

The second map (Figure C.1, Appendix C) is run with the code, prompts, model, settings, stopping rules and seed streams of the first map: twenty runs each for the one-route single agent, its best-of-three and the team, and ten for the team without correction. Its threshold is $F \le 1 5 6$ , nine above the optimum $F ^ { * } = 1 4 7$

Tables E.3 and E.2 and Figure 3(b) summarize the runs. The team is safe in all twenty runs and reaches the threshold in nineteen; the remaining run stops at $F = 2 0 6 .$ , and seed 10 reaches the optimum. Its final loss has median 151, lower than the single agent’s 155 and the best-of-three’s 156 (permutation tests on the medians, $p = 0 . 0 0 2$ and $p = 0 . 0 0 1 )$ ; on eighteen of twenty seeds it ends at least as low as each of them. The means difer less clearly $( p = 0 . 2 6$ and $p = 0 . 0 8 )$ because one single-agent run stops at $F = 3 5 3$ with two collisions and two best-of-three runs stop with one collision each. Only the team’s mean gap falls below 10, at about 520 seconds. In search time, the best-of-three is fastest (median 272 against 416 seconds for the team, $p = 0 . 0 2 )$ ; the single agent and the team do not difer significantly in median or mean, but the single agent’s spread is larger (standard deviation 1.8 times the team’s, $p = 0 . 0 3 )$ , with six runs longer than 700 seconds against three. Played in turn, the team’s answers would take $7 7 8 \pm 3 0 5$ seconds instead of $4 8 2 \pm 2 0 2$ . Without correction, eight of ten runs end with collisions between agents. Seven of them end at the fixed point described for the first map, with no model calls in their last five rounds; the eighth also repeats itself, since in each of its last five rounds one agent returns routes that are optimal for its own robots but collide with other agents’ robots, and they are rejected. This is why the uncorrected curve stops after about 310 seconds: more rounds would add no answers, or only rejected ones.

## E.4 Two floors

The two-floor task (Appendix C) places the first map (robots r0–r8) and the second map (robots renamed s0–s8) on the two floors of one building; both instances are copied unchanged from the single-floor campaigns apart from the renaming. Robots on diferent floors never collide, so $F = F _ { 1 } + F _ { 2 }$ with optimum $F ^ { * } = 2 9 1$ and all-waiting value 2955. A plan meets the threshold when both floors are safe with $F _ { 1 } \leq 1 5 3$ and $F _ { 2 } \leq 1 5 6 .$

Team variants. The team has six agents with the blocks of the single-floor runs. Its dependency graph consists of two disjoint triangles; each color class holds one block of each floor, and the coordinator commits, in each round, the best improving block of each floor, if any. This is Algorithm 1 with the pairing of the two floors’ blocks into colors chosen anew in each round: any such pairing is a proper three-coloring, and the committed gain is at least that of the best color of any fixed coloring, so the analysis applies with $\kappa = 3 .$ Each agent’s prompt shows only its own floor, exactly as in the single-floor runs. A two-floor team run is therefore a run on each floor played in lockstep: round � lasts max $( T _ { 1 , k } , T _ { 2 , k } )$ where $T _ { f , k }$ is the slowest agent’s search on floor � in that round, and a floor that has met its threshold or gone five rounds without improvement makes no further calls; the run ends when both floors have stopped. The team, its sequential replay, which sums the times of all six agents, and the team without correction are evaluated on the recorded single-floor runs, pairing seed � of the first map with seed � of the second, without new model calls.

<table><tr><td></td><td colspan="3">Single agent, 1 route</td><td colspan="2">Best-of-3</td><td colspan="3">Single agent, exactly 3 routes</td><td colspan="3">Team (Alg. 1)</td><td colspan="2">Team, no correction</td></tr><tr><td>Seed</td><td>F</td><td>Ans.</td><td>Time (s)</td><td></td><td> $F \mathrm { T i m e } \left( \mathrm { s } \right)$ </td><td>F</td><td>Ans.</td><td>Time (s)</td><td>F</td><td>Rounds</td><td>Time (s)</td><td> $F$ </td><td>Time (s)</td></tr><tr><td>0</td><td>148</td><td>20</td><td>318.9</td><td>157</td><td>673.7</td><td>153</td><td>10</td><td>280.6</td><td>145</td><td>6</td><td>304.2</td><td> $4 4 4 ^ { \dagger }$ </td><td>209.3</td></tr><tr><td>1</td><td>153</td><td>19</td><td>276.9</td><td>151</td><td>250.8</td><td>151</td><td>26</td><td>554.3</td><td>152</td><td>7</td><td>542.3</td><td> $4 5 2 ^ { \dagger }$ </td><td>533.2</td></tr><tr><td>2</td><td>153</td><td>20</td><td>323.4</td><td>152</td><td>195.3</td><td>848†</td><td>20</td><td>448.2</td><td>152</td><td>5</td><td>253.0</td><td> $5 4 6 ^ { \dagger }$ </td><td>357.1</td></tr><tr><td>3</td><td>151</td><td>14</td><td>209.8</td><td>152</td><td>213.4</td><td>153</td><td>14</td><td>504.3</td><td>147</td><td>11</td><td>666.7</td><td> $4 4 4 ^ { \dagger }$ </td><td>163.7</td></tr><tr><td>4</td><td>150</td><td>22</td><td>334.7</td><td>148</td><td>156.3</td><td>148</td><td>13</td><td>267.0</td><td>150</td><td>12</td><td>987.2</td><td> $4 4 4 ^ { \dagger }$ </td><td>270.8</td></tr><tr><td>5</td><td>152</td><td>17</td><td>260.1</td><td>153</td><td>449.7</td><td>149</td><td>24</td><td>477.0</td><td>145</td><td>13</td><td>911.9</td><td> $4 4 4 ^ { \dagger }$ </td><td>183.0</td></tr><tr><td>6</td><td>191</td><td>48</td><td>1106.0</td><td>152</td><td>190.5</td><td>152</td><td>16</td><td>769.4</td><td>152</td><td>6</td><td>386.5</td><td> $3 6 1 ^ { \dag }$ </td><td>290.4</td></tr><tr><td>7</td><td>153</td><td>47</td><td>1080.7</td><td>148</td><td>279.5</td><td>149</td><td>43</td><td>1124.9</td><td>150</td><td>3</td><td>121.5</td><td> $2 4 4 ^ { \dagger }$ </td><td>125.0</td></tr><tr><td>8</td><td>157</td><td>34</td><td>955.1</td><td>152</td><td>363.3</td><td>152</td><td>12</td><td>478.3</td><td>149</td><td>8</td><td>625.9</td><td> $5 4 4 ^ { \dagger }$ </td><td>525.2</td></tr><tr><td>9</td><td>153</td><td>26</td><td>581.9</td><td>153</td><td>212.0</td><td>153</td><td>26</td><td>708.4</td><td>152</td><td>9</td><td>673.7</td><td> $6 4 8 ^ { \dagger }$ </td><td>604.3</td></tr><tr><td>10</td><td>152</td><td>12</td><td>144.0</td><td>370†</td><td>325.6</td><td></td><td></td><td></td><td>149</td><td>6</td><td>366.0</td><td></td><td></td></tr><tr><td>11</td><td>153</td><td>22</td><td>407.3</td><td>154</td><td>586.8</td><td></td><td></td><td></td><td>151</td><td>8</td><td>442.7</td><td></td><td></td></tr><tr><td>12</td><td>153</td><td>18</td><td>312.5</td><td>158</td><td>934.2</td><td></td><td></td><td></td><td>152</td><td>7</td><td>353.1</td><td></td><td></td></tr><tr><td>13</td><td>152</td><td>36</td><td>810.1</td><td>153</td><td>214.2</td><td></td><td></td><td></td><td>151</td><td>4</td><td>302.9</td><td></td><td></td></tr><tr><td>14</td><td>160</td><td>41</td><td>1257.6</td><td>152</td><td>360.2</td><td></td><td></td><td></td><td>153</td><td>9</td><td>638.1</td><td></td><td></td></tr><tr><td>15</td><td>153</td><td>20</td><td>382.2</td><td>153</td><td>221.3</td><td></td><td></td><td></td><td>152</td><td>8</td><td>596.0</td><td></td><td></td></tr><tr><td>16</td><td>153</td><td>50</td><td>1532.0</td><td>148</td><td>188.4</td><td></td><td></td><td></td><td>152</td><td>6</td><td>438.2</td><td></td><td></td></tr><tr><td>17</td><td>150</td><td>31</td><td>623.4</td><td>149</td><td>273.8</td><td></td><td></td><td></td><td>150</td><td>5</td><td>323.3</td><td></td><td></td></tr><tr><td>18</td><td>153</td><td>25</td><td>488.2</td><td>151</td><td>177.9</td><td></td><td></td><td></td><td>150</td><td>8</td><td>513.0</td><td></td><td></td></tr><tr><td>19</td><td>180</td><td>34</td><td>855.4</td><td>152</td><td>261.4</td><td></td><td></td><td></td><td>144</td><td>4</td><td>174.7</td><td></td><td></td></tr></table>

Table E.1: Every run on the first map with prompt v2, independently replayed; — marks a seed that was not run for that variant. <sup>†</sup> marks a final plan that is not safe. Ans.: checked answers; team times use the per-round maximum (C.1).

<table><tr><td></td><td colspan="3">Single agent</td><td colspan="2">Best-of-3</td><td colspan="3">Team  $\left( \mathrm { A l g . ~ 1 } \right)$ </td><td colspan="2"> $\operatorname { T e a m } ,$  no correction</td></tr><tr><td>Seed</td><td>F</td><td>Ans.</td><td>Time (s)</td><td>F</td><td>Time (s)</td><td>F</td><td>Rounds</td><td>Time (s)</td><td> $F$ </td><td>Time (s)</td></tr><tr><td>0</td><td>156</td><td>20</td><td>458.2</td><td>153</td><td>173.3</td><td>151</td><td>12</td><td>916.2</td><td> $4 4 7 ^ { \dagger }$ </td><td>93.0</td></tr><tr><td>1</td><td>154</td><td>24</td><td>526.9</td><td>152</td><td>203.3</td><td>148</td><td>5</td><td>341.1</td><td> $2 4 7 ^ { \dagger }$ </td><td>89.4</td></tr><tr><td>2</td><td>155</td><td>14</td><td>278.0</td><td>156</td><td>409.8</td><td>155</td><td>7</td><td>396.9</td><td> $3 4 7 ^ { \dagger }$ </td><td>307.1</td></tr><tr><td>3</td><td>153</td><td>45</td><td>1236.5</td><td>185</td><td>476.3</td><td>151</td><td>5</td><td>260.7</td><td> $6 4 7 ^ { \dagger }$ </td><td>68.0</td></tr><tr><td>4</td><td>156</td><td>20</td><td>324.7</td><td>156</td><td>379.3</td><td>155</td><td>6</td><td>367.6</td><td> $4 4 7 ^ { \dagger }$ </td><td>194.0</td></tr><tr><td>5</td><td>154</td><td>19</td><td>293.9</td><td>150</td><td>199.0</td><td>148</td><td>4</td><td>180.0</td><td> $^ \mathrm { 1 5 2 }$ </td><td>218.2</td></tr><tr><td>6</td><td>154</td><td>15</td><td>222.6</td><td>156</td><td>218.8</td><td>152</td><td>7</td><td>384.1</td><td> $2 4 7 ^ { \dagger }$ </td><td>114.1</td></tr><tr><td>7</td><td>155</td><td>41</td><td>746.1</td><td>156</td><td>212.0</td><td>149</td><td>9</td><td>434.1</td><td> $2 5 1 ^ { \dagger }$ </td><td>304.9</td></tr><tr><td>8</td><td>161</td><td>43</td><td>1224.3</td><td>149</td><td>243.4</td><td>206</td><td>10</td><td>625.8</td><td> $3 4 7 ^ { \dagger }$ </td><td>164.9</td></tr><tr><td>9</td><td>156</td><td>33</td><td>838.8</td><td>162</td><td>701.0</td><td>148</td><td>8</td><td>651.4</td><td> $^ \mathrm { 1 5 2 }$ </td><td>227.3</td></tr><tr><td>10</td><td>155</td><td>17</td><td>322.5</td><td>152</td><td>257.8</td><td>147</td><td>7</td><td>310.1</td><td></td><td></td></tr><tr><td>11</td><td>150</td><td>20</td><td>372.1</td><td>155</td><td>185.0</td><td>156</td><td>8</td><td>524.2</td><td></td><td></td></tr><tr><td>12</td><td>154</td><td>17</td><td>271.7</td><td>156</td><td>197.5</td><td>151</td><td>8</td><td>521.9</td><td></td><td></td></tr><tr><td>13</td><td>353†</td><td>31</td><td>681.4</td><td>269†</td><td>452.9</td><td>151</td><td>6</td><td>344.5</td><td></td><td></td></tr><tr><td>14</td><td>152</td><td>15</td><td>258.5</td><td>153</td><td>286.3</td><td>149</td><td>13</td><td>823.9</td><td></td><td></td></tr><tr><td>15</td><td>156</td><td>12</td><td>178.7</td><td>191</td><td>456.7</td><td>156</td><td>5</td><td>290.9</td><td></td><td></td></tr><tr><td>16</td><td>167</td><td>35</td><td>1178.9</td><td>155</td><td>326.7</td><td>155</td><td>10</td><td>638.1</td><td></td><td></td></tr><tr><td>17</td><td>153</td><td>14</td><td>197.6</td><td>153</td><td>179.9</td><td>148</td><td>10</td><td>518.9</td><td></td><td></td></tr><tr><td>18</td><td>156</td><td>14</td><td>207.5</td><td>156</td><td>433.0</td><td>154</td><td>12</td><td>780.6</td><td></td><td></td></tr><tr><td>19</td><td>180</td><td>41</td><td>939.5</td><td>256†</td><td>533.7</td><td>153</td><td>7</td><td>334.2</td><td></td><td></td></tr></table>

Table E.2: Every run on the second map with prompt v2, independently replayed. <sup>†</sup> marks a final plan that is not safe. Ans.: checked answers; team times use the per-round maximum (C.1).

<table><tr><td>Variant</td><td>Safe</td><td> $F \le 1 5 6$ </td><td>Final F</td><td> $\mathrm { S e a r c h \ t i m e \ ( s ) }$ </td><td>Answers</td><td>USD/run</td></tr><tr><td>Single agent, 1 route per answer</td><td>19/20</td><td>16/20</td><td> $1 6 6 . 5 0 \pm 4 4 . 3 8$ </td><td> $5 3 7 . 9 \pm 3 6 5 . 7$ </td><td> $2 4 . 5 \pm 1 1 . 2$ </td><td> $0 . 1 3 7 \pm 0 . 0 9 3$ </td></tr><tr><td>Best-of-3 one-route single agents</td><td>18/20</td><td>15/20</td><td> $1 6 8 . 5 5 \pm 3 3 . 9 2$ </td><td> $3 2 6 . 3 \pm 1 4 7 . 0$ </td><td> $3 8 . 7 \pm 9 . 6 $ </td><td> $0 . 1 9 5 \pm 0 . 0 8 9$ </td></tr><tr><td>Team  $\left( \mathrm { A l g . ~ 1 } \right)$ </td><td>20/20</td><td>19/20</td><td> $1 5 4 . 1 5 \pm 1 2 . 5 6$ </td><td> $4 8 2 . 3 \pm 2 0 1 . 6$ </td><td> $4 4 . 8 \pm 1 6 . 1$ </td><td> $0 . 2 2 1 \pm 0 . 0 9 3$ </td></tr><tr><td>Team without correction</td><td>2/10</td><td>2/10</td><td> $3 2 8 . 4 0 \pm 1 5 3 . 4 1$ </td><td> $1 7 8 . 1 \pm 8 7 . 2$ </td><td> $1 9 . 0 \pm 7 . 2$ </td><td> $0 . 0 6 9 \pm 0 . 0 3 1$ </td></tr><tr><td>Sequential replay of the team</td><td>20/20</td><td>19/20</td><td> $1 5 4 . 1 5 \pm 1 2 . 5 6$ </td><td> $7 7 8 . 2 \pm 3 0 4 . 6$ </td><td> $4 4 . 8 \pm 1 6 . 1$ </td><td></td></tr></table>

Table E.3: All runs on the second map; mean ± sample SD over all runs, including unsafe ones.

Single agent and best-of-six. The single-agent prompt states the rules once, adding that each floor is its own grid and that robots on diferent floors never collide, and shows each floor exactly as the single-floor prompt does: the map with its labels and wall list, every robot’s program and cost, the timeline and every current collision with its maps. The accepted changes and the rejected attempts are listed once for the whole plan, with the floor of each changed robot; the loss is shown per floor and in total, and the goal names both thresholds. Each answer replaces the route of one robot on either floor, and a change is accepted only if � strictly decreases. The single agent stops once both floors meet their thresholds, after fifteen answers without improvement or after 240 answers; the best-of-six runs six copies of it as the best-of-three does and stops after five rounds without improvement (thirty answers) or after eighty rounds. Neither cap was reached. Seeds, retries and the clock are those of the single-floor runs; the best-of-six ran seeds 0–9 first and seeds 10–19 afterwards with byte-identical code. The provider stopped two of the 4,207 requests at its output limit, and each was re-asked once as before.

Results. Table E.4 summarizes the variants and Table E.5 lists every run.

∙ Single agent: safe in all twenty runs; nine meet both thresholds, thirteen the first floor’s and eleven the second’s. Final loss $3 1 9 . 6 0 \pm 2 3 . 3 1$ (median 313.5, at most 409); search time 1148.7±419.6 seconds (median 1200; eleven runs take longer than 1000 seconds) with $6 3 . 2 \pm 1 6 . 2$ answers. Eleven runs stop after fifteen answers without improvement.

∙ Best-of-six: safe in all twenty runs, sixteen of which meet both thresholds; final loss 312.40±16.26 (median 306.5, at most 363); search time $5 1 6 . 8 \pm 1 2 3 . 9$ seconds (median 511, at most 739) over 23.8 rounds with $1 4 2 . 5 \pm 1 7 . 1$ answers.

∙ Team: safe in all twenty runs, nineteen of which meet both thresholds; final loss 304.05±12.73 (median 302); search time 656.4 ± 206.1 seconds (median 595, at most 1013) over 9.3 rounds with $9 1 . 9 \pm 2 0 . 7$ answers summed over both floors.

∙ Team without correction (ten runs): no run is safe; final loss 785.50 ± 181.90, search time $3 8 2 . 1 \pm$ 176.1 seconds.

∙ Sequential replay of the team: $1 6 3 2 . 1 \pm 4 6 8 . 5$ seconds against $6 5 6 . 4 \pm 2 0 6 . 1$ in parallel, a ratio of means of 2.49 (1.78 on the first map, with three agents).

Statistical comparison. We apply the tests used on the first map. Against the single agent, the team’s final loss is lower (diference of means 15.6, $p = 0 . 0 0 2 \mathrm { : }$ of medians 11.5, $p < 0 . 0 0 1 ;$ ; the single agent ends higher on seventeen seeds and equal on two), the team meets both thresholds more often (19 against $9 , p = 0 . 0 0 1$ , Fisher’s exact test), and it is faster (diference of medians 606 seconds and of means 492 seconds, both $p < 0 . 0 0 1 $ ; faster on seventeen seeds), with half the spread (standard deviation ratio 2.04, $p = 0 . 0 0 2 )$ Against the best-of-six, the team’s median final loss is lower by 4.5 $( p = 0 . 0 0 4 )$ , the means difer by 8.4 $( p = 0 . 0 9 )$ , and the best-of-six ends higher on sixteen seeds; the threshold counts do not difer significantly (19 against $1 6 , p = 0 . 3 4 )$ The best-of-six is faster in mean (by 140 seconds, $p = 0 . 0 1 3 )$ but not in median (by 84 seconds, $p = 0 . 1 9 )$ , with a smaller spread (ratio 0.60, $p = 0 . 0 2 )$ , and it is faster on fifteen seeds.

Bound. The bound defined in Appendix E.8 below applies with six blocks and $\kappa = 3 \colon \alpha _ { s , k }$ adds $Q _ { s , k , b } D _ { b }$ over all six blocks and divides by $3 \Delta _ { s , k ; }$ and $B _ { s , k }$ adds the error terms of both floors; a floor that has stopped contributes neither. The one-step inequality (E.5) holds in all 186 rounds. Propagated per seed from $E _ { s , 0 } = 2 6 6 4$ , held after each run stops and averaged, the envelope ends at mean gap 395.9 after thirteen rounds, against 13.1 observed; it stays within a factor of 2.2 of the observed mean gap through round 3 (1.11, 1.46 and 2.15). The pooled means of the coeficients are $\bar { \alpha } = 0 . 2 2 2$ and $\bar { B } = 6 6 . 7$

## E.5 Best-combination and unsafe merges

Algorithm 1 accepts the best color; on one map the dependency graph is complete, so one block per round. Two variants change only this step. With best combination, the coordinator checks every combination of the blocks whose proposal difers from the base, at most seven on one map, and accepts the one with the lowest verified loss, or nothing if none improves (the exact branch of Algorithm 2). Because the improving blocks of each color are among these combinations, the round gain is at least that of the best color, and Theorem 2 applies unchanged (Section 3.3). The unsafe team accepts every changed block together, whatever the loss, and the run returns the best plan it has seen. Everything else is that of the team with Algorithm 1: code, prompt v2, model, settings, seed streams, the five-round stopping rule (which counts rounds that do not improve the best plan) and the measurement. The prompt states the merge rule in one sentence, so that sentence difers: with Algorithm 1 the coordinator “installs the best block proposal of the round”, with best combination it “checks every combination of the block proposals of the round on the full plan and installs the combination that lowers the loss most”, and in the unsafe team it “installs the block proposals of all agents of the round together”. Collision pairs found in any checked combination become known pairs. Before the runs, an ofline test with scripted answers confirmed that the code with the merge rule of Algorithm 1 reproduces the decisions and prompts of the main runs exactly; the independent replay checks every combination, decision and clock. On two floors both variants are composed from the two maps as for Algorithm 1: the best combination of six blocks on independent floors is the best combination of each floor. The provider stopped 19 of the 3,664 requests at its output limit, eight of them in one run; each was re-asked once.

<table><tr><td>Variant</td><td>Safe</td><td>Both thresholds</td><td>Final F</td><td>Search time (s)</td><td>Answers</td><td> $\mathrm { U S D / r u n }$ </td></tr><tr><td>Single agent, 1 route per answer</td><td>20/20</td><td>9/20</td><td> $3 1 9 . 6 0 \pm 2 3 . 3 1$ </td><td> $1 1 4 8 . 7 \pm 4 1 9 . 6$ </td><td> $6 3 . 2 \pm 1 6 . 2$ </td><td> $0 . 3 6 5 \pm 0 . 1 2 7$ </td></tr><tr><td>Best-of-6 one-route single agents</td><td>20/20</td><td>16/20</td><td> $3 1 2 . 4 0 \pm 1 6 . 2 6$ </td><td> $5 1 6 . 8 \pm { 1 2 3 . 9 }$ </td><td> $1 4 2 . 5 \pm 1 7 . 1$ </td><td> $0 . 6 8 0 \pm 0 . 1 5 4$ </td></tr><tr><td>Team (Alg. 1)</td><td>20/20</td><td>19/20</td><td> $3 0 4 . 0 5 \pm 1 2 . 7 3$ </td><td> $6 5 6 . 4 \pm 2 0 6 . 1$ </td><td> $9 1 . 9 \pm 2 0 . 7$ </td><td> $0 . 4 4 9 \pm 0 . 1 4 1$ </td></tr><tr><td>Team without correction</td><td>0/10</td><td>0/10</td><td> $7 8 5 . 5 0 \pm 1 8 1 . 9 0$ </td><td> $3 8 2 . 1 \pm 1 7 6 . 1$ </td><td> $5 6 . 0 \pm 1 9 . 9$ </td><td> $0 . 2 0 2 \pm 0 . 0 7 9$ </td></tr><tr><td>Sequential replay of the team</td><td>20/20</td><td>19/20</td><td> $3 0 4 . 0 5 \pm 1 2 . 7 3$ </td><td> $1 6 3 2 . 1 \pm 4 6 8 . 5$ </td><td> $9 1 . 9 \pm 2 0 . 7$ </td><td></td></tr></table>

Table E.4: All variants on two floors $( F ^ { * } = 2 9 1 )$ ; mean ± sample SD over all runs. Team answers and costs are summed over both floors.

<table><tr><td rowspan="2">Seed</td><td colspan="3">Single agent</td><td colspan="3">Best-of-6</td><td colspan="4">Team (Alg. 1)</td><td colspan="2">Team, no correction</td></tr><tr><td> $F _ { 1 } + F _ { 2 }$ </td><td>Ans.</td><td>Time (s)</td><td> $F _ { 1 } + F _ { 2 }$ </td><td>Rounds</td><td>Time (s)</td><td> $F _ { 1 } + F _ { 2 }$ </td><td>Rounds</td><td>Time (s)</td><td> $\mathrm { S e q . \ ( s ) }$ </td><td> $F _ { 1 } + F _ { 2 }$ </td><td>Time (s)</td></tr><tr><td>0</td><td> $1 5 0 + 1 6 4 ^ { * }$ </td><td>69</td><td>1778.2</td><td> $1 5 1 + 1 7 8 ^ { * }$ </td><td>26</td><td>676.8</td><td> $1 4 5 { + } 1 5 1$ </td><td>12</td><td>916.2</td><td>1906.9</td><td> $4 4 4 + 4 4 7 ^ { \ast }$ </td><td>219.5</td></tr><tr><td>1</td><td> $1 5 0 + 1 5 5$ </td><td>68</td><td>1518.6</td><td> $1 5 1 + 1 5 5$ </td><td>31</td><td>739.3</td><td> $1 5 2 + 1 4 8$ </td><td>7</td><td>578.4</td><td>1586.1</td><td> $4 5 2 + 2 4 7 ^ { \ast }$ </td><td>533.2</td></tr><tr><td>2</td><td> $1 5 2 + 1 6 5 ^ { * }$ </td><td>62</td><td>1380.5</td><td> $2 0 0 { + } 1 6 3 ^ { * }$ </td><td>28</td><td>638.8</td><td> $1 5 2 + 1 5 5$ </td><td>7</td><td>423.9</td><td>1186.9</td><td> $5 4 6 + 3 4 7 ^ { \ast }$ </td><td>435.2</td></tr><tr><td>3 4</td><td> $1 5 3 + 1 6 1 ^ { \ast }$ </td><td>67</td><td>1356.9</td><td> $^ { 1 5 2 + 1 5 3 }$ </td><td>24</td><td>491.6</td><td> $1 4 7 + 1 5 1$ </td><td>11</td><td>723.4</td><td>1335.1</td><td> $4 4 4 + 6 4 7 ^ { \ast }$ </td><td>163.7</td></tr><tr><td></td><td> $1 5 2 + 1 5 6$ </td><td>32</td><td>473.4</td><td> $1 5 0 + 1 5 3$ </td><td>21</td><td>390.4</td><td> $1 5 0 + 1 5 5$ </td><td>12</td><td>1012.8</td><td>2412.9</td><td> $4 4 4 + 4 4 7 ^ { \ast }$ </td><td>324.2</td></tr><tr><td>5</td><td> $1 5 3 + 1 6 5 ^ { * }$ </td><td>56</td><td>1255.8</td><td> $1 5 1 + 1 5 6$ </td><td>21</td><td>428.9</td><td> $1 4 5 { + } 1 4 8$ </td><td>13</td><td>935.2</td><td>1804.7</td><td> $4 4 4 + 1 5 2 \AA ^ { * }$ </td><td>257.3</td></tr><tr><td>6</td><td> $1 5 2 + 1 5 6$ </td><td>68</td><td>912.8</td><td> $^ { 1 4 8 + 1 5 3 }$ </td><td>20</td><td>319.7</td><td> $^ { 1 5 2 + 1 5 2 }$ </td><td>7</td><td>452.5</td><td>1690.2</td><td> $3 6 1 + 2 4 7 ^ { \ast }$ </td><td>303.9</td></tr><tr><td>7</td><td> $1 7 1 + 1 5 7 ^ { * }$ </td><td>82</td><td>1463.3</td><td> $1 5 2 + 1 5 6$ </td><td>26</td><td>578.7</td><td> $1 5 0 { + } 1 4 9$ </td><td>9</td><td>446.9</td><td>1014.6</td><td> $2 4 4 \substack { + 2 5 1 } ^ { \ast }$ </td><td>304.9</td></tr><tr><td>8</td><td>153+156</td><td>46</td><td>659.9</td><td> $^ { 1 5 2 + 1 5 3 }$ </td><td>19</td><td>344.3</td><td> $1 4 9 + 2 0 6 ^ { * }$ </td><td>10</td><td>754.6</td><td>2242.5</td><td> $5 4 4 + 3 4 7 ^ { \ast }$ </td><td>549.6</td></tr><tr><td>9</td><td>151+155</td><td>79</td><td>1291.7</td><td> $1 5 2 + 1 6 6 ^ { * }$ </td><td>25</td><td>692.7</td><td> $1 5 2 + 1 4 8$ </td><td>9</td><td>828.2</td><td>1957.3</td><td> $6 4 8 + 1 5 2 \AA ^ { * }$ </td><td>729.1</td></tr><tr><td>10</td><td> $1 5 6 + 1 5 8 ^ { * }$ </td><td>79</td><td>1576.7</td><td> $1 5 2 + 1 5 1$ </td><td>21</td><td>381.7</td><td> $1 4 9 + 1 4 7$ </td><td>7</td><td>423.2</td><td>1031.3</td><td></td><td></td></tr><tr><td>11</td><td>151+156</td><td>49</td><td>789.7</td><td> $1 5 3 + 1 5 5$ </td><td>23</td><td>496.2</td><td> $1 5 1 + 1 5 6$ </td><td>8</td><td>570.9</td><td>1501.8</td><td></td><td></td></tr><tr><td>12</td><td> $1 5 8 + 1 5 5 ^ { \ast }$ </td><td>91</td><td>1971.7</td><td> $1 5 2 + 1 5 5$ </td><td>24</td><td>455.3</td><td> $1 5 2 + 1 5 1$ </td><td>8</td><td>541.8</td><td>1462.1</td><td></td><td></td></tr><tr><td>13</td><td> $1 7 3 + 1 6 2 ^ { * }$ </td><td>72</td><td>1144.8</td><td> $1 5 2 + 1 5 4$ </td><td>21</td><td>395.8</td><td> $1 5 1 + 1 5 1$ </td><td>6</td><td>440.5</td><td>1283.9</td><td></td><td></td></tr><tr><td>14</td><td> $1 5 2 + 1 5 5$ </td><td>44</td><td>689.9</td><td> $1 8 4 + 1 6 5 ^ { * }$ </td><td>24</td><td>524.9</td><td> $1 5 3 + 1 4 9$ </td><td>13</td><td>963.3</td><td>2131.4</td><td></td><td></td></tr><tr><td>15</td><td> $1 5 2 + 1 5 6$ </td><td>52</td><td>832.4</td><td> $1 5 0 + 1 5 5$ </td><td>24</td><td>535.7</td><td> $1 5 2 + 1 5 6$ </td><td>8</td><td>596.0</td><td>1693.5</td><td></td><td></td></tr><tr><td>16</td><td> $1 6 6 + 1 7 4 ^ { * }$ </td><td>53</td><td>903.9</td><td> $1 5 2 + 1 5 6$ </td><td>25</td><td>655.1</td><td> $1 5 2 + 1 5 5$ </td><td>10</td><td>731.6</td><td>1951.8</td><td></td><td></td></tr><tr><td>17</td><td> $2 2 5 { + } 1 8 4 ^ { * }$ </td><td>49</td><td>604.1</td><td> $1 5 0 + 1 5 4$ </td><td>24</td><td>597.5</td><td> $1 5 0 { + } 1 4 8$ </td><td>10</td><td>593.4</td><td>1252.0</td><td></td><td></td></tr><tr><td>18</td><td> $1 5 0 + 1 5 6$ </td><td>55</td><td>889.0</td><td> $1 5 2 + 1 5 6$ </td><td>23</td><td>426.5</td><td> $1 5 0 + 1 5 4$ </td><td>12</td><td>831.6</td><td>2396.7</td><td></td><td></td></tr><tr><td>19</td><td> $1 7 7 + 1 4 9 ^ { * }$ </td><td>91</td><td>1480.1</td><td> $^ { 1 5 2 + 1 5 3 }$ </td><td>25</td><td>566.9</td><td> $^ { 1 4 4 + 1 5 3 }$ </td><td>7</td><td>362.9</td><td>800.3</td><td></td><td></td></tr></table>

Table E.5: Every run on two floors. $F _ { 1 } + F _ { 2 } \colon$ final loss of each floor; <sup>\*</sup> marks a run that misses a threshold. Every final plan of the single agent, the best-of-six and the team is safe; no final plan without correction is. Team runs pair seed � of both maps; Seq.: sequential replay.

plan. The median search time until the gap first falls below 100 is 185.1, 154.7 and 223.2 seconds on the three tasks with best combination, against 313.7, 288.1 and 444.7 with Algorithm 1, 236.1, 207.2 and 362.9 for the best-of-�, 329.1, 274.7 and 675.4 for the single agent and 377.6, 256.6 and 425.7 for the unsafe team; a run that never gets there counts as infinitely slow. In permutation tests on the medians, only the best-of-three on map 1 is not significantly slower $( p = 0 . 2 7 ;$ $p \leq 0 . 0 0 7$ for every other comparison). The final losses do not difer significantly from those of Algorithm 1 on any task $( p \geq 0 . 2 4 )$ , with medians 150, 153 and 302.5 against 150.5, 151 and 302. On map 1 nineteen of twenty runs meet the threshold; on map 2 fourteen do, against nineteen with Algorithm 1 $( p = 0 . 0 9$ , Fisher’s exact test), and six runs stop above it, at � between 157 and 203, against one.

Results. Table E.6 summarizes the runs and Table E.7 lists them. With best combination, a round that changes the plan accepts on average 1.67 blocks on map 1 and 1.72 on map 2, and in every run the first round accepts two or three. The team is faster than with Algorithm 1 on every task: on map 1 the search takes 339.7 ± 196.9 instead of 481.0 ± 227.3 seconds (diference of means $p = 0 . 0 4$ , of medians $p = 0 . 0 3 )$ on map 2 310.4 ± 204.1 instead of 482.3 ± 201.6 (� = 0.01 and $p = 0 . 0 4 )$ , and on two floors 451 $. 6 \pm 2 2 7 . 1$ instead of 656.4 ± 206.1 $( p = 0 . 0 0 5$ and $p = 0 . 0 8 )$ , with fewer rounds and answers and USD 0.18 and 0.17 instead of 0.23 and 0.22 per run on the two maps. In median, it also reaches a safe plan first. A collision or an undelivered robot adds at least 100 to the loss, so a gap below 100 implies a safe

Why six runs stall on map 2. In each of the six, the last five rounds accept nothing although a block can still improve: at the final state the exact corrected block minimum lies 43 to 52 below the current value in three of the runs and 2 to 14 below in the other three, so no run stops at a single-block trap. In five runs the other two blocks are certified optimal for their agents’ corrected objectives, so those agents make no calls; in seed 1 two agents keep searching. In four runs (seeds

<table><tr><td>Task and merge rule</td><td>Safe</td><td>Threshold</td><td>Final F</td><td>Search time (s)</td><td>Sequential (s)</td><td>Rounds</td><td>Answers</td><td>USD/run</td></tr><tr><td> $\mathrm { M a p \ 1 , \ T e a m \ ( A l g . \ 2 ) }$ </td><td>20/20</td><td>19/20</td><td> $1 4 9 . 4 0 \pm 3 . 2 2$ </td><td> $3 3 9 . 7 \pm 1 9 6 . 9$ </td><td> $6 0 1 . 7 \pm 2 5 5 . 5$ </td><td>5.7</td><td> $3 6 . 2 \pm { 1 1 . 9 }$ </td><td> $0 . 1 8 1 \pm 0 . 0 7 4$ </td></tr><tr><td>Map 1, Team unsafe</td><td>20/20</td><td>20/20</td><td> $1 4 9 . 0 0 \pm 2 . 5 5$ </td><td> $6 0 1 . 4 \pm 2 8 4 . 5$ </td><td> $1 1 8 0 . 2 \pm 5 1 5 . 9$ </td><td>9.3</td><td> $6 4 . 6 \pm 2 5 . 9$ </td><td> $0 . 3 5 7 \pm 0 . 1 5 4$ </td></tr><tr><td>Map 2, Team  $\left( \mathrm { A l g } . \ 2 \right)$ </td><td>20/20</td><td>14/20</td><td> $1 5 9 . 7 5 \pm 1 7 . 2 1$ </td><td> $3 1 0 . 4 \pm 2 0 4 . 1$ </td><td> $5 2 0 . 0 \pm 2 8 7 . 0$ </td><td>5.6</td><td> $3 3 . 0 \pm 1 5 . 5$ </td><td> $0 . 1 7 4 \pm 0 . 1 3 5$ </td></tr><tr><td>Map 2, Team unsafe</td><td>19/20</td><td>19/20</td><td> $1 5 7 . 0 5 \pm 2 2 . 2 6$ </td><td> $4 2 3 . 7 \pm 2 3 2 . 4$ </td><td> $7 6 4 . 2 \pm 4 1 0 . 3$ </td><td>7.4</td><td> $4 8 . 0 \pm 2 2 . 6$ </td><td> $0 . 2 3 5 \pm 0 . 1 3 0$ </td></tr><tr><td>Two floors, Team  $( \mathrm { A l g } . ~ 2 )$ </td><td>20/20</td><td>14/20</td><td> $3 0 9 . 1 5 \pm 1 7 . 3 9$ </td><td> $4 5 1 . 6 \pm 2 2 7 . 1$ </td><td> $1 1 2 1 . 7 \pm 4 2 1 . 2$ </td><td>7.1</td><td> $6 9 . 2 \pm 1 9 . 8$ </td><td> $0 . 3 5 5 \pm 0 . 1 5 8$ </td></tr><tr><td>Two floors, Team unsafe</td><td>19/20</td><td>19/20</td><td> $3 0 6 . 0 5 \pm 2 2 . 5 4$ </td><td> $7 1 6 . 8 \pm 2 6 3 . 0$ </td><td> $1 9 4 4 . 3 \pm 5 6 2 . 7$ </td><td>10.6</td><td> $1 1 2 . 5 \pm 2 8 . 8$ </td><td> $0 . 5 9 2 \pm 0 . 1 7 2$ </td></tr></table>

Table E.6: The team with the best-combination and the unsafe merge; mean ± sample SD over twenty runs. The team with Algorithm 1 is in Table 1. On two floors, answers and costs are summed over both floors.

<table><tr><td rowspan="3"></td><td colspan="5">Map 1</td><td colspan="5">Map 2</td></tr><tr><td colspan="3">Team (Alg. 2)</td><td colspan="2">Team unsafe</td><td colspan="3">Team (Alg. 2)</td><td colspan="2">Team unsafe</td></tr><tr><td>F</td><td>Rounds</td><td>Time (s)</td><td>F</td><td>Time (s)</td><td>F</td><td>Rounds</td><td>Time (s)</td><td>F</td><td>Time (s)</td></tr><tr><td>0</td><td>152</td><td>3</td><td>191.1</td><td>147</td><td>843.9</td><td>148</td><td>2</td><td>75.3</td><td>153</td><td>948.6</td></tr><tr><td>1</td><td>152</td><td>8</td><td>595.0</td><td>149</td><td>646.0</td><td>157</td><td>8</td><td>489.1</td><td>155</td><td>319.2</td></tr><tr><td>2</td><td>153</td><td>6</td><td>385.7</td><td>150</td><td>807.0</td><td>150</td><td>9</td><td>502.5</td><td>153</td><td>539.6</td></tr><tr><td>3</td><td>148</td><td>5</td><td>262.6</td><td>144</td><td>342.9</td><td>153</td><td>4</td><td>203.9</td><td>149</td><td>938.4</td></tr><tr><td>4</td><td>148</td><td>5</td><td>252.1</td><td>152</td><td>420.9</td><td>152</td><td>3</td><td>149.9</td><td>149</td><td>365.8</td></tr><tr><td>5</td><td>145</td><td>4</td><td>197.8</td><td>150</td><td>359.3</td><td>151</td><td>2</td><td>75.9</td><td>149</td><td>594.7</td></tr><tr><td>6</td><td>145</td><td>5</td><td>292.5</td><td>147</td><td>427.2</td><td>154</td><td>4</td><td>295.7</td><td>155</td><td>381.8</td></tr><tr><td>7</td><td>144</td><td>4</td><td>198.3</td><td>150</td><td>507.2</td><td>153</td><td>3</td><td>122.1</td><td>251†</td><td>429.5</td></tr><tr><td>8</td><td>152</td><td>3</td><td>176.6</td><td>148</td><td>1263.0</td><td>192</td><td>10</td><td>566.3</td><td>154</td><td>157.4</td></tr><tr><td>9</td><td>153</td><td>9</td><td>565.6</td><td>148</td><td>321.4</td><td>155</td><td>4</td><td>248.6</td><td>152</td><td>238.9</td></tr><tr><td>10</td><td>153</td><td>4</td><td>227.7</td><td>150</td><td>818.6</td><td>149</td><td>2</td><td>116.2</td><td>151</td><td>399.1</td></tr><tr><td>11</td><td>149</td><td>6</td><td>383.8</td><td>150</td><td>670.7</td><td>155</td><td>7</td><td>410.1</td><td>149</td><td>181.4</td></tr><tr><td>12</td><td>150</td><td>4</td><td>230.2</td><td>146</td><td>677.8</td><td>151</td><td>3</td><td>159.4</td><td>155</td><td>414.5</td></tr><tr><td>13</td><td>146</td><td>4</td><td>223.7</td><td>145</td><td>381.9</td><td>203</td><td>13</td><td>765.8</td><td>156</td><td>260.7</td></tr><tr><td>14</td><td>154</td><td>14</td><td>952.3</td><td>147</td><td>352.6</td><td>161</td><td>8</td><td>476.9</td><td>150</td><td>531.8</td></tr><tr><td>15</td><td>150</td><td>4</td><td>216.6</td><td>148</td><td>448.0</td><td>149</td><td>4</td><td>149.5</td><td>152</td><td>672.2</td></tr><tr><td>16</td><td>150</td><td>9</td><td>549.0</td><td>152</td><td>86.1</td><td>153</td><td>5</td><td>247.5</td><td>153</td><td>340.8</td></tr><tr><td>17</td><td>144</td><td>8</td><td>443.2</td><td>153</td><td>978.5</td><td>149</td><td>3</td><td>164.1</td><td>150</td><td>427.1</td></tr><tr><td>18</td><td>149</td><td>4</td><td>262.8</td><td>151</td><td>909.5</td><td>201</td><td>8</td><td>344.8</td><td>156</td><td>273.0</td></tr><tr><td>19</td><td>151</td><td>4</td><td>188.2</td><td>153</td><td>766.1</td><td>159</td><td>9</td><td>645.2</td><td>149</td><td>58.8</td></tr></table>

Table E.7: Every run of the two merge variants, independently replayed. <sup>†</sup> marks a final plan that is not safe.

1, 8, 14 and 18) a searching agent did return a proposal that improved its corrected objective, but the proposal collided with robots of pairs that the correction did not yet know, so verification rejected it and the collisions became known pairs; in seed 14 the agent’s next proposal collided with yet another unknown pair, and in seeds 1, 8 and 18 the agent found no further improvement before the stop. In seeds 13 and 19 the agent found no improvement of its corrected objective in fifteen answers. The merge rule would accept any single improving block, so the stalls come from local search with an incomplete correction, not from the merge. These states difer from those of Algorithm 1 because the first rounds difer: after round 1 the plan has 3.3 collisions on average, against 0.1, since two or three agents’ routes are accepted together, and it is safe after 2.8 rounds instead of 5.2. Once no collision is left, the realized search quality falls for every merge rule; on map 2 it is 0.28 with best combination (57 of 85 such searches capture nothing) against 0.34 with Algorithm 1 (47 of 87). On map 1 the diference does not occur (one run above the threshold against none).

Unsafe merging. The unsafe team accepts 2.2 blocks per changing round on map 1 and 2.1 on map 2, but its accepted plan becomes worse in 32 of its 185 rounds on map 1 and 37 of 148 on map 2, when blocks changed together create new collisions; after round 1 its plan has 6.3 collisions on average on map 1 and 3.0 on map 2. Its search time does not difer significantly from that of Algorithm 1 on any task (601.4 against 481.0 seconds on map 1, 423.7 against 482.3 on map 2, 716.8 against 656.4 on two floors; $p > 0 . 1 )$ , it needs more answers, and its final losses are similar $( p \ge 0 . 1 8 )$ ; one final plan on map 2 keeps a collision. Accepting worse plans did not significantly lower its final loss, and one final state on each map is a single-block trap.

Audit and bound. The audit of Appendix E.7, defined below, applies to both variants (Table E.9). With best combination, the realized search quality averages 0.65 and 0.60 on the two maps, 26% and 28% of the locally safe proposals collide with another agent’s robot, and the geometry ratio is at least 0.55 and 0.75 at every state that starts a round; four and two final states are single-block traps, all of them final states at which a run had met its threshold and stopped. The one-step inequality (E.5) holds in all 113 and 111 rounds and in all 142 on two floors, and the averaged envelope ends at mean gap 328.7, 399.4 and 604.7 against 5.4, 12.8 and 18.2 observed (gray curves in Figure E.4); coeficients from seeds 0–9 again bound the mean gap of seeds 10–19 in every round, and vice versa. The envelope ends higher than with Algorithm 1 because these runs stop after fewer rounds; $\mathrm { A p \mathrm { - } }$ pendix E.9 sharpens it by counting only the conflicts that occur. For the unsafe team the recurrence does not apply, since the loss can increase; its search quality averages 0.63 and 0.68 and its share of colliding locally safe proposals 29% and 34%.

## E.6 Recorded collision repairs

The loss curves hide what an accepted change looks like. The plan of Figure 2 arose from a repair shown in detail in Figure E.1. In team seed 19 on the first map, the plan accepted after round 3 has $F = 2 4 4$ and one collision: $r _ { 1 }$ and $r _ { 8 } .$ , which belong to diferent agents, exchange cells at tick 11. The pair was known from an earlier global check, so in round 4 the agent owning $r _ { 0 } , r _ { 1 } , r _ { 6 }$ saw $r _ { 8 } \mathrm { { ^ 3 s } }$ current timeline. Its first answer moved $r _ { 1 }$ to another shortest route with the same arrival tick $2 0 ;$ it also appended waits after arrival to �<sub>0</sub> and $r _ { 6 } ,$ which leave their cells unchanged. Its local loss did not change, $\Delta L _ { b } = 0$ , while the correction fell by 100. The verifier accepted the block, and the plan reached $F = 1 4 4$ the optimum, with all nine robots delivered and no collision. On the first map, 15 of the 121 accepted team rounds remove the last collision between agents in this way, through a pair the agent knew from the correction.

Figure E.2 shows two further repairs at the collision tick. In team seed 11, round 7, the agent of $r _ { 0 } , r _ { 1 } , r _ { 6 }$ knew the swap of $r _ { 0 }$ with $r _ { 7 }$ at tick 8; its third answer changed only $r _ { 0 }$ ’s route, kept its arrival tick 20 (local change $0 ,$ correction change −100) and removed the plan’s last collision (� from 259 to 159; the run ends at 151). The single agent repairs the same kind of conflict without a correction, because it sees every robot: in seed 4, its 22nd answer replaced $r _ { 4 } \mathrm { { ^ { * } s } }$ route, kept its arrival tick 15 and removed the swap of �<sub>4</sub> with �<sub>8</sub> at tick 6 (� from 250 to 150, the run’s final plan). In all three cases the paths still cross in space but no longer at the same tick, so spatial intersections alone do not determine safety.

## E.7 Audit of the convergence assumptions

The team runs allow a direct comparison with Assumptions 1– 3. Using the exact references of Appendix C.7, an ofline audit computes, at every recorded base of every team run, the exact corrected and true block minima. On the first map it covers 145 global rounds, 435 returned block proposals and 942 checked answers (160 of them geometrically invalid) at 165 state occurrences, one of which is optimal; on the second map, 159 rounds, 477 proposals and 895 checked answers (274 invalid) at 179 state occurrences. It includes unselected proposals, searches that return the incumbent and routeinvalid answers. Table E.9 compares the measured constants across prompts and tasks.

Coupling. Of the 382 returned proposals that deliver all three of an agent’s robots without internal collisions, 115 (30%) still collide with another agent’s robot on the first map, and 147 of 461 (32%) on the second; with prompt v1 the share was 83 of 211 (39%). Local delivery and internal safety therefore do not sufice for global compatibility. We count simultaneous vertex or edge-swap events, not intersections of spatial route drawings.

<table><tr><td></td><td colspan="4">Map 1</td><td colspan="4">Map 2</td></tr><tr><td>Block</td><td></td><td>n Mean Q</td><td> $Q = 0 Q = 1$ </td><td></td><td></td><td>n Mean Q</td><td> $Q = 0 Q = 1$ </td><td></td></tr><tr><td>Agent 1 125</td><td></td><td>0.67</td><td>26</td><td>40</td><td>79</td><td>0.72</td><td>18</td><td>31</td></tr><tr><td>Agent 2 114</td><td></td><td>0.63</td><td>28</td><td></td><td>16 115</td><td>0.73</td><td>23</td><td>58</td></tr><tr><td>Agent 3 104</td><td></td><td>0.68</td><td>23</td><td></td><td>26 144</td><td>0.56</td><td>46</td><td>22</td></tr><tr><td>All</td><td>343</td><td>0.66</td><td>77</td><td></td><td>82 338</td><td>0.65</td><td>87</td><td>111</td></tr></table>

Table E.8: Realized search quality � after each private search of the team with prompt v2, including unselected proposals; searches with zero initial corrected gap are excluded. Agents 1, 2 and 3 own $\{ r _ { 0 } , r _ { 1 } , r _ { 6 } \} , \{ r _ { 2 } , r _ { 3 } , r _ { 7 } \}$ and $\{ r _ { 4 } , r _ { 5 } , r _ { 8 } \}$ on map 1 and $\{ r _ { 0 } , r _ { 1 } , r _ { 2 } \} , \{ r _ { 3 } , r _ { 4 } , r _ { 5 } \}$ and $\left\{ r _ { 6 } , r _ { 7 } , r _ { 8 } \right\}$ on map 2.

Local search quality: Assumption 2. For each returned block $U _ { k b }$ with a positive initial surrogate gap, compute

$$
Q _ { k b } = \frac { - \widehat { f } _ { b } ( U _ { k b } ; x ^ { k } ) } { - \operatorname* { m i n } _ { u } \widehat { f } _ { b } ( u ; x ^ { k } ) } \in [ 0 , 1 ] .\tag{E.1}
$$

The unchanged block is a retained option, so this is the fraction of available corrected local improvement captured by the entire $H = 3$ search. It is not the quality of only the winning block or the fraction of the global gap closed. Early termination at a certified optimum can be viewed as leaving that optimum unchanged for the remaining private steps. A route-invalid answer consumes a private step and leaves the best score unchanged; all legal answers are replayed, including worse ones.

There are 343 searches with a positive corrected gap on the first map and 338 on the second; the others start at a surrogate optimum and are reported separately rather than assigned $Q = 1$ . The pooled mean is 0.66 on the first map and 0.65 on the second, with $7 7$ and 87 zero-gain searches and 82 and 111 that reach the exact corrected optimum (Table E.8); weighting by the initial local gap gives captured fractions of 0.82 and 0.85. The mean falls from 0.93 and 0.99 in round 1 to below 0.5 from round 5 on in the rounds with at least ten searches (Figure E.3(a)). This compares diferent histories, and later rounds exclude runs that have stopped. Assumption 2 requires a lower bound on the conditional expected fraction at every history, with $q _ { b } ( H ) = 1 - ( 1 - \rho _ { b } ) ^ { H }$ ， so Appendix E.8 uses each round’s realized � in the pathwise recurrence rather than a pooled constant.

Correction accuracy: Assumption 1. In the theory, the retained correction (23) is $\widehat { R } _ { b } ;$ the true correction also includes the undiscovered external pairs. Let $C _ { b , \mathrm { m i s s } } ^ { k } ( u )$ count conflicts with those pairs against the frozen peers. The exact score error is

$$
\begin{array} { r l } & { \varepsilon _ { k b } ( u ) = \widehat { f _ { b } } ( u ; \boldsymbol { x } ^ { k } ) - f _ { b } ( u ; \boldsymbol { x } ^ { k } ) } \\ & { \phantom { \varepsilon } = - 1 0 0 \big [ C _ { b , \mathrm { m i s s } } ^ { k } ( u ) - C _ { b , \mathrm { m i s s } } ^ { k } ( x _ { b } ^ { k } ) \big ] . } \end{array}\tag{E.2}
$$

Positions at ticks t-3 to t+1 around the collision tick t (larger dot: tick t); red circle: the swap

Agent 1 Agent 2 Agent 3 Old route Start (robot ID) Goal (robot ID) Collision  
![](images/b22f334cd904e5523ca2934ec0396b761c23ae83dc78b1de8c734d8f8924ef7f.jpg)

![](images/5ec99615d4b16b5e3651b4f899d1328d5de24e1211f5d82acfb9c9bdd01838b7.jpg)  
(b) One block: robots 0, 1, 6  
Only collision: r1 and r8 swap cells at tick $1 1 ; \mathsf { F } = 2 4 4$  
Only r1 changes its cells, same arrival; global F = 144 (optimum)

Figure E.1: The recorded repair behind Figure 2 (prompt v2, first map, team seed 19). (a) All nine accepted routes after round $_ { 3 ; }$ colors identify the three agents; filled circles mark starts and open diamonds goals. The red circle marks the only collision of this plan: $r _ { 1 }$ and $r _ { 8 }$ exchange cells at tick 11. (b) The block of the agent that owns $r _ { 0 } , r _ { 1 } , r _ { 6 }$ after its accepted answer in round 4: only $r _ { 1 }$ changes its cells (old route dashed), with the same arrival tick, and the plan reaches the optimum $F = 1 4 4$ . Coincident segments are slightly ofset for visibility; all event checks use the original cells and ticks.  
![](images/d72a0ca4070ed7958ecbadac8c7a1054ad4ee5b1cc2eaa88aad44378a530b3de.jpg)  
Figure E.2: Two more recorded repairs with prompt v2 on the first map, as positions of the two robots at ticks $t - 3$ to $t + 1$ around the collision tick � (larger dot: tick $t ;$ red circle: the swap). Left: team seed 11, round $7 ;$ the agent of $r _ { 0 } , r _ { 1 } , r _ { 6 }$ knows the swap of $r _ { 0 }$ with $r _ { 7 }$ from the correction, and only $r _ { 0 } \mathrm { { ^ { 2 } s } }$ route changes, with the same arrival tick $( F$ from 259 to 159). Right: single agent seed $^ { 4 , }$ answer $^ { 2 2 ; }$ only $r _ { 4 }$ ’s route changes, with the same arrival tick, and the swap with $r _ { 8 }$ disappears (� from 250 to 150). Python replays the recorded answers without editing them.

Both corrections are anchored at zero. At every recorded base, the missing-pair conflict count is zero; any existing conflict has already been discovered. Hence a nonzero error at these bases is optimistic: a missing conflict makes the private score understate the true global change.

Among the 782 legal answers on the first map, the mean absolute error is 94.5 and the maximum 3,500; among the

![](images/89894f7e6825c51bdd87e18c945893cc00cfce2095d846f68fa4d820de53b1b5.jpg)

![](images/a7c6624391f39c875dc8c2ca4164e311414a08befe4618e13eb3c0091bd910ce.jpg)  
Figure E.3: Ofline assumption diagnostics for the team with prompt v2. (a) Mean realized $Q$ over searches with a positive corrected gap, in the rounds with at least ten such searches. (b) Exact ratio $\sum _ { b } D _ { b } ( x ) / \Delta ( x )$ at every recorded state, one line per run; dashed lines mark the smallest value over the states that start a round, $\mu = 0 . 4 3$ on map 1 and 0.70 on map $2 ;$ dots mark final states, crosses the single-block traps (ratio 0).

435 returned proposals, 106 have a nonzero error, with mean absolute error 45.5 and maximum 500. On the second map the 621 legal answers have mean absolute error 53.3 (maximum 600), and 114 of the 477 returned proposals have a nonzero error, with mean 36.5 (maximum 600). In 28 and 40 searches the private score improves without a positive verified global gain, which distinguishes local search progress from progress after the correction and the global check.

These observed errors difer from a uniform bound $\delta _ { b } ,$ which takes a supremum over all legal block programs, including poor collision-seeking routes; such a bound implies Assumption 1 but is not needed for it. For each recorded state and block we compute that supremum exactly. A timeexpanded min/max dynamic program counts undiscovered external conflicts for each owned robot; because those counts add across robots, their extrema add over the product domain. Obstacles and absorbing goals remain enforced. Internal conflicts are penalized in $F ,$ rather than prohibited in the candidate domain. Independent replay verifies the extremizing joint programs.

Over the recorded states and blocks, these exact suprema average 12,933 on the first map (from 11,400 to 14,000) and 12,518 on the second (from 10,900 to 13,600). A deterministic bound covering any history is $\delta _ { b } = 8 6 , 4 0 0 \colon$ a block has at most $3 \cdot 6 = 1 8$ external pairs, each contributing at most 48 events because starts are distinct, and an anchored count diference is at most 18 · 48; multiplying by the collision weight gives the bound. Finite uniform boundedness and anchoring are therefore certified, but these worst-case bounds far exceed the initial global gap of 1332 and give no useful floor in Theorem 2. Assumption 1 avoids this obstacle: it needs the error only at a true block minimizer, where it is zero at every audited state, and at the returned proposal, where it averages 45.5 and 36.5.

Block geometry: Assumption 3. Using the exact true min ima in (C.4), compute over the recorded states � that start a round

$$
\mu _ { \mathrm { o b s } } = \operatorname* { m i n } _ { x } \operatorname* { m i n } \left\{ 1 , \frac { \sum _ { b } D _ { b } ( x ) } { F ( x ) - F ^ { * } } \right\} .\tag{E.3}
$$

It equals 0.43 on the first map, where the ratio is below 0.6 at one of the 145 states that start a round, and 0.70 on the second (Figure E.3(b)). Among final states, two on the first map, at $F = 1 4 7$ and $F = 1 4 5$ , and four on the second, at $F = 1 4 8$ or 149, are single-block traps: every $D _ { b } \ = \ 0$ although the gap is positive, so only a joint change of two blocks can improve them. All six are final states at which the run had met its threshold and stopped; no round starts from a trap, and the recurrence is applied only at states that start a round, where $\mu$ holds. The certificate covers the recorded states, not the full product domain; with prompt v1, $\mu = 0 . 6$ held at all 88 recorded states, including the final ones.

Scope of the connection to the theory. The task has a finite product domain of legal command arrays, exact global verification, unary and pairwise factors, and a conservative three-color dependency graph. The frozen parallel searches and best singleton commit implement the theory’s merge with $\kappa = 3 .$ The audit certifies the block error bound on the states that start a round and quantifies actual corrected local progress and score error. Assumptions 1 and 2 are statements about conditional expectations; the audit measures their realized counterparts on the recorded states. The two-point errors of Assumption 1 are small, whereas a uniform bound over all routes would be far larger, and Appendix E.8 uses the measured values. Because conflict weights are known, the experiment does not exercise Assumptions A.1–A.2.

<table><tr><td></td><td>µ</td><td>Q</td><td>α</td><td>B</td><td>Coupled</td><td>Gap (bound)</td></tr><tr><td>Team (Alg. 1)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Map 1, prompt v1</td><td>0.60</td><td>0.58</td><td>0.247</td><td>47.4</td><td>39%</td><td>12.5 (211.7)</td></tr><tr><td>Map 1</td><td>0.43</td><td>0.66</td><td>0.246</td><td>45.5</td><td>30%</td><td>5.9 (254.2)</td></tr><tr><td>Map 2</td><td>0.70</td><td>0.65</td><td>0.242</td><td>36.5</td><td>32%</td><td>7.2 (226.8)</td></tr><tr><td>Two floors</td><td></td><td>0.66</td><td>0.222</td><td>66.7</td><td>31%</td><td>13.1 (395.9)</td></tr><tr><td>Team  $( A l g . \ 2 )$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Map 1</td><td>0.55</td><td>0.65</td><td>0.244</td><td>36.3</td><td>26%</td><td>5.4 (328.7)</td></tr><tr><td>Map 2</td><td>0.75</td><td>0.60</td><td>0.222</td><td>40.2</td><td>28%</td><td>12.8 (399.4)</td></tr><tr><td>Two floors</td><td></td><td>0.63</td><td>0.210</td><td>60.3</td><td>27%</td><td>18.2 (604.7)</td></tr><tr><td>Team unsafe</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Map 1</td><td>0.73 0.63</td><td></td><td></td><td></td><td>29%</td><td>5.0 (—)</td></tr><tr><td>Map 2</td><td>0.60 0.68</td><td></td><td></td><td></td><td>34%</td><td>10.1 ()</td></tr></table>

Table E.9: Measured constants of the team runs: $\mu$ at states that start a round, mean search quality �<sup>¯</sup>, pooled means of the round coeficients, share of locally safe proposals that collide with another agent’s robot, and final mean gap with the averaged bound in parentheses. Prompt v1 values use all states and its ten runs; the two-floor rows pool the searches of both maps, and their $\mu$ is not computed. For the unsafe team the round coeficients and the bound are not defined, because its loss can increase. The correction error at true block minimizers is zero in every case, so $\bar { B }$ is the mean error at returned proposals (summed over both floors on two floors).

## E.8 The bound, round by round

This subsection defines the retrospective upper bound of Figures 4(a) and E.4 and evaluates it on every team run. The bound uses actual search quality and correction errors in each round, rather than full-domain worst-case errors or constant coeficients obtained by pooling diferent histories. For seed � and an active round $k ,$ define

$$
\begin{array} { r l } & { \alpha _ { s , k } = \frac { { \displaystyle \sum _ { b } Q _ { s , k , b } D _ { b } ( x _ { s } ^ { k } ) } } { { 3 \Delta \left( x _ { s } ^ { k } \right) } } , } \\ & { B _ { s , k } = \frac { 1 } { 3 } \displaystyle \sum _ { b } \left( Q _ { s , k , b } e _ { s , k , b } ^ { * } + e _ { s , k , b } ^ { U } \right) . } \end{array}\tag{E.4}
$$

Here $Q$ is the realized corrected-local progress fraction, $D _ { b }$ is the exact available true block improvement, $e ^ { * }$ is the absolute anchored score error at a certified true block minimizer, and $e ^ { U }$ is that error at the returned proposal. If the initial surrogate gap is zero, $D _ { b } \ = \ 0$ in these data, so its $Q D _ { b }$ contribution is zero. All true-minimizer errors $e ^ { * }$ are zero; returned-candidate errors include the actual missing-pair penalties and the zero-error cases.

For each recorded search, the argument of Lemma 1, applied to realized rather than expected quantities, gives

$$
g _ { s , k , b } \geq Q _ { s , k , b } D _ { b } - Q _ { s , k , b } e _ { s , k , b } ^ { * } - e _ { s , k , b } ^ { U } .
$$

The best of the three singleton colors has at least their mean verified gain. Hence

$$
\Delta ( x _ { s } ^ { k + 1 } ) \leq ( 1 - \alpha _ { s , k } ) \Delta ( x _ { s } ^ { k } ) + B _ { s , k } .\tag{E.5}
$$

Independent replay verifies this inequality and $0 \leq \alpha _ { s , k } \leq 1$ in every active round, using all returned block proposals, not just selected blocks. Averaging over blocks is conservative here: with $\kappa = 3$ each color is a single block on one map, so the committed gain is the largest singleton gain. The largest perblock lower bound gives the sharper recurrence $\Delta ( \bar { x } _ { s } ^ { k + 1 } ) \leq$ max $\begin{array} { r l } { \{ 0 , \Delta ( x _ { s } ^ { k } ) - \operatorname* { m a x } _ { b } ( Q _ { s , k , b } D _ { b } - Q _ { s , k , b } e _ { s , k , b } ^ { * } - e _ { s , k , b } ^ { U } ) \} } & { { } } \end{array}$ , which we also check.

Propagate the bound separately for each seed. Initialize ${ { E } _ { s , 0 } } = \Delta ( { { x } _ { s } ^ { 0 } } )$ (1476 − 144 = 1332 on the first map) and set

$$
E _ { s , k + 1 } = \operatorname* { m i n } \{ E _ { s , k } , ( 1 - \alpha _ { s , k } ) E _ { s , k } + B _ { s , k } \} .\tag{E.6}
$$

The minimum uses the accepted loss’s monotonicity. Since $1 - \alpha _ { s , k } \ge 0$ , induction with (E.5) proves $\Delta ( x _ { s } ^ { k } ) \le E _ { s , k }$ at every recorded state. After run � stops at $K _ { s } ,$ hold both its actual loss and envelope fixed. Only then average the twenty envelopes:

$$
\overline { { U } } _ { K } = F ^ { * } + \frac { 1 } { 2 0 } \sum _ { s = 0 } ^ { 1 9 } E _ { s , \operatorname* { m i n } \{ K , K _ { s } \} } .\tag{E.7}
$$

Results. On the first map, the one-step inequality (E.5) and $0 \leq \alpha _ { s , k } \leq 1$ hold in all 145 active rounds. Held after each run stops and averaged over the twenty runs, the envelope (E.7) ends at mean gap 254.2 after thirteen rounds, against 5.9 observed; it stays within a factor of 2.1 of the observed mean gap through round 3 (1.13, 1.49 and 2.03). With the largest per-block lower bound instead of the mean, the final mean gap is 6.8. The mean error term $B _ { s , k }$ falls from 137 in round 1 to 0 in round 7, with small values in later rounds over the few runs still active. On the second map the inequality holds in all 159 rounds, and the envelope ends at 226.8 against 7.2 observed, or 8.0 with the largest per-block lower bound. On two floors it holds in all 186 rounds and ends at 395.9 against 13.1 (Appendix E.4). The pooled means of the coeficients are $\bar { \alpha } = 0 . 2 4 6 , 0 . 2 4 2$ and 0.222 and $\bar { B } = 4 5 . 5 , 3 6 . 5$ and 66.7 on the three tasks; they summarize the logs and are not substituted into the recurrence, because keeping each round’s coeficients preserves their dependence on the current gap and history. Figure E.4 shows the pointwise range of the observed gap over runs. With prompt v1 on the first map, the bound held in all 78 rounds of the ten team runs and ended at 211.7 against 12.5, or 15.3 with the largest per-block lower bound.

This is a valid upper bound on the recorded trajectories: the pathwise form of Theorem 2, with realized � in place of $q _ { b } ( H )$ and measured two-point errors in place of $\delta _ { b } ^ { * } , \delta _ { b } ^ { U }$ . It takes no supremum over collision-seeking routes and fits no decay rate to the loss curve. Because its coeficients come from the same runs, it explains the recorded progress rather than predicting fresh runs.

Coeficients from other runs. The recurrence can also be evaluated with coeficients measured on diferent runs: average $\alpha _ { s , k }$ and $B _ { s , k }$ over the runs of one half that are active in round �, and propagate $\overline { { E } } _ { k + 1 } = \operatorname* { m i n } \{ \overline { { E } } _ { k } , ( 1 - \bar { \alpha } _ { k } ) \overline { { E } } _ { k } + \bar { B } _ { k } \}$ from $\overline { { E } } _ { 0 } = 1 3 3 2 .$ . On the first map, coeficients from seeds 0–9 give a curve above the mean gap of seeds 10–19 in every round, ending at 141.5 against 6.4, and with the halves exchanged it ends at 167.9 against 5.4, again above in every round; on the second map the curves also lie above in every round in both directions, and with prompt v1, splitting the ten seeds into halves gave the same result.

![](images/5d2480dc5adcca638c58f91b0b00f6f9ed9c260f8ebec16e1a9e6d798ce9c8fc.jpg)  
Figure E.4: Observed mean gap of each team and its averaged trajectory-wise bound (Appendix E.8) for every task, with the divisors of Section $5 \colon \kappa = 3 ;$ ; min $\{ \kappa , | I ^ { k } | \}$ (1 if no block improves), which bounds the number of colors that contain an improving block; the efective number of colors � (Proposition 4), which needs checked combinations and therefore applies only to Team $\left( \mathrm { A l g . ~ 2 } \right)$ ; and the realized number of colors $\kappa _ { k } ^ { * }$ (Corollary 3). Bands: pointwise minimum and maximum of the observed gap over the twenty runs, not confidence intervals; a run at the optimum is drawn at gap 1. On two floors each floor uses its own divisor.

## E.9 Greedy merging and conflicts

Section 5 shows that the best-combination merge keeps the guarantees of Theorem 2 and that its rate depends on the realized number of colors $\kappa _ { k } ^ { * }$ , which is at most the efective number of colors $\kappa _ { k }$ . This subsection gives a greedy version that needs only quadratically many checks, measures the conflicts and the resulting envelope on the recorded runs, explains why the block error bound is still needed, and lists ways to weaken it.

A greedy rule. With pairwise factors, (18) determines the loss of every combination from the single checks and one check per pair, so $| I ^ { k } | ( | I ^ { k } | - 1 ) / 2$ further checks sufice to find all conflicts and the best combination of improving blocks. When even the search over these combinations is too costly, a greedy rule sufices. Write $\deg ^ { \prime } ( b )$ for the number of remaining blocks that conflict with $b ;$ the rule repeatedly keeps the remaining block � with the largest $g _ { v } / ( 1 + \deg ^ { \prime } ( v ) )$ and deletes � and the blocks that conflict with it. It returns a conflict-free set with gain at least $\sum _ { b \in I ^ { k } } g _ { b } / ( 1 + \deg _ { k } ( b ) )$ (Sakai et al., 2003). Indeed, each of the $1 + \deg ^ { \prime } ( v )$ blocks � deleted in that step, � included, satisfies $g _ { u } / ( 1 + \deg ^ { \prime } ( u ) ) \leq g _ { v } / ( 1 + \deg ^ { \prime } ( v ) )$ , so together they contribute at most $g _ { v }$ to ${ \textstyle \sum _ { u } } g _ { u } / ( 1 + \deg ^ { \prime } ( u ) )$ since $\deg ^ { \prime } ( u ) \leq \deg _ { k } ( u )$ and every block is deleted exactly once, the kept gains add up to at least $\sum _ { b \in I ^ { k } } g _ { b } / ( 1 { + } \deg _ { k } ( b ) )$ Accepting the better of this set and the best color keeps Theorem 2, and Corollary 3 when every improving proposal conflicts with at most �¯ − 1 others, using a quadratic number of checks.

Recorded runs. The runs with best combination checked all combinations of changed blocks, which measures every $\xi _ { b b ^ { \prime } }$ . The checks confirm (18) in all 44 and 29 rounds in which all three blocks changed on the two maps, and Proposition $^ { 4 , }$ together with the bound ${ \sum } _ { b } g _ { b } / ( 1 + \deg _ { k } ( b ) )$ of the greedy rule above, in all 113 and 111 rounds. Of the pairs of improving proposals, 71 of 135 conflict on map 1 and 42 of 90 on map 2. In 56 and 28 of these pairs, accepting both proposals creates a collision between their robots that neither proposal causes alone; in the other 15 and 14, both proposals remove the same collision. Another 11 and 15 pairs help each other $( \xi _ { b b ^ { \prime } } < 0 )$ . Rounds with two or more improving proposals have no conflict in 16 of 63 cases on map 1 and 10 of 42 on map 2. With the rounds that have fewer than two improving proposals, $\kappa _ { k } = 1$ in 66 of the 113 rounds on map 1 and 79 of the 111 on map 2, $\kappa _ { k } = 2$ in 41 and 30, and $\kappa _ { k } = 3$ in 6 and 2. The realized number of colors $\kappa _ { k } ^ { * }$ averages 1.19 and 1.15 over the rounds that change the plan, never exceeds 1.98 and 2.00, and is below 1 in 8 and 9 rounds. In the rounds with two or more improving proposals it averages 1.29 and 1.26, against 1.84 and 1.81 for $\kappa _ { k }$ , and it is smaller than $\kappa _ { k }$ in 49 of 63 and 35 of 42 of them: conflicting proposals are accepted together because that still lowers the loss most. Table E.10 splits the rounds into the start, the repair phase, in which the base still has an undelivered robot or a collision, and the refinement of a safe base. At the start every conflict is a new collision between the agents’ new routes, yet the best combination often accepts both proposals; during repair many conflicts come from two proposals that remove the same collision; once the base is safe, conflicts are rare because at most one agent improves in most rounds.

With $g _ { b } \geq Q _ { b } D _ { b } - Q _ { b } e _ { b } ^ { * } - e _ { b } ^ { U }$ (Appendix E.8), (16) gives $\begin{array} { r } { F ( \boldsymbol { x } ) - F ( \boldsymbol { x } ^ { + } ) = \sum _ { b } g _ { b } / \kappa _ { k } ^ { * } \geq \sum _ { b } ( Q _ { b } D _ { b } - Q _ { b } e _ { b } ^ { * } - e _ { b } ^ { U } ) / \kappa _ { k } ^ { * } } \end{array}$ , and Proposition 4 gives the same with $\kappa _ { k }$ in place of $\kappa _ { k } ^ { * }$ . Hence the recurrence (E.5) holds with the divisor 3 in (E.4) replaced by either value in round � of run $s ,$ with $\alpha _ { s , k }$ capped at one; for Algorithm 1 the realized number is $\sum _ { b } { g _ { b } } /$ max<sub>�</sub> $g _ { b }$ . It exceeds one whenever the round discards an improving proposal, which happens in 79 of the 121 rounds of Team (Alg. 1) that change the plan on map 1 and in 64 of 115 on map 2; over all rounds that change the plan it averages 1.50 and 1.53 and reaches 2.85 and 2.94. The replay checks this inequality in every round, and the envelope (E.6) with these coeficients bounds every recorded gap. Figure E.4 separates three efects. At most min $\{ \kappa , | I ^ { k } | \}$ colors contain an improving block, so dividing by this number is valid for any merge that keeps at least the best color. If no block improves, then $| I ^ { k } | = 0$ and every �<sub>�</sub> is zero, so $\begin{array} { r } { \sum _ { b } ( Q _ { b } D _ { b } - Q _ { b } e _ { b } ^ { * } - e _ { b } ^ { U } ) \le 0 ; } \end{array}$ : the gain bound holds with any positive divisor. As for $\kappa _ { k }$ and $\kappa _ { k } ^ { * }$ in Section 5, we then divide by 1. It lowers the final envelope of the best-combination runs from 328.7 to 107.2 on map 1, from 399.4 to 162.5 on map 2 and from 604.7 to 190.6 on two floors, where each floor uses its own divisor, which is valid because the best combination is optimal on each floor; for the runs with Algorithm 1 it lowers the envelope from 254.2 to 54.6 and from 226.8 to 62.2. Dividing by the efective number of colors $\kappa _ { k } ,$ , which needs the best combination, lowers the best-combination envelopes further to 54.8, 116.3 and 133.0. Dividing by the realized number $\kappa _ { k } ^ { * } .$ , which also credits conflicting proposals accepted together, lowers them to 18.1, 87.6 and 69.9, against observed mean gaps of 5.4, 12.8 and 18.2; for Algorithm 1, its realized number gives 27.5, 33.5 and 50.0, against 5.9, 7.2 and 13.1. Through round 3, the best-combination envelope stays within a factor of 1.5, 3.7 and 1.7 of the observed mean gap with $\kappa _ { k } ^ { * }$ and 2.7, 5.7 and 3.1 with $\kappa _ { k }$ , against 6.4, 17.8 and 8.9 with $\kappa = 3$

What still needs the block error bound. The best combination combines the agents’ proposals, and an agent returns its current block unless it finds an improvement on its corrected objective. At a single-block trap with exact corrections no block improvement exists, so every agent returns its current block and every merge of the returned proposals stays at the trap. No merge rule over the returned proposals can therefore replace Assumption 3: with exact corrections, a reached trap keeps a positive gap forever, whereas Theorem 2 shows that the assumption sufices together with Assumption 2. Four and two final states of the best-combination runs on the two maps are such traps (Appendix E.5), and the unsafe team, which also accepts worse plans, ends in one such trap on each map.

Directions. Three changes could weaken the assumption. First, a pairwise error bound $\begin{array} { r } { \sum _ { b } D _ { b } ( x ) + \sum _ { b < b ^ { \prime } } D _ { b b ^ { \prime } } ( x ) \ge } \end{array}$ $\mu _ { 2 } \Delta ( x )$ , where $D _ { b b ^ { \prime } } ( x )$ is the best improvement from changing blocks � and $b ^ { \prime }$ together, holds at every trap that a joint change of two blocks can leave. Exploiting it requires proposals that are not individually improving: each agent could return a short list of candidates, such as its best proposal and alternatives that reroute its robots involved in reported collisions, and the coordinator would check combinations across the lists. A rate would then need a joint counterpart of Assumption 2: with some probability, the lists contain both parts of a good two-block change. Second, blocks can be coarsened after a stall: when a round accepts nothing, two agents’ blocks are merged for the next round, so one agent controls six robots. A two-block change of the fine partition is a single-block change of the coarse one, and Theorem 2 applies to the coarse partition with its own $\mu .$ Third, conflicts could be bounded in expectation: Corollary 3 needs $\kappa _ { k } \le \bar { \kappa }$ in every epoch, whereas a condition such as $\begin{array} { r } { \mathbb { E } \big [ \sum _ { b } g _ { b } / ( 1 + \deg _ { k } ( b ) ) \mid \mathcal { H } \big ] \ge \bar { \kappa } ^ { - 1 } \mathbb { E } \big [ \sum _ { b } g _ { b } \mid \mathcal { H } \big ] } \end{array}$ would allow occasional dense conflicts; it couples the gains with the conflicts and would have to be measured like the other constants.

## E.10 Parallelism and decomposition

We distinguish the efect of simplifying an individual search from the efect of executing searches concurrently. The discussion relates these mechanisms to the recorded comparison and the sequential timing replay.

Reducing the dificulty of a model request. Each team search writes three routes; with prompt v1 the whole-plan single agent writes nine in every answer and reaches a safe plan in one run of ten. A single agent that still sees and scores the whole plan but replaces only one route per answer is safe in nine runs of ten with prompt v1 and in all twenty with prompt ${ \mathrm { v } } 2$ on the first map (Appendix D), so much of the dificulty lies in the size of each answer. Within the team, the correction removes collisions between agents: without it, the team is safe in one run of ten with prompt v1, and in none of ten, two of ten and none of ten runs with prompt v2 on map 1, map 2 and two floors.

<table><tr><td>Map</td><td>Phase</td><td>Rounds</td><td>Mean  $\vert I ^ { k } \vert$ </td><td>Conflicting pairs</td><td>New collision</td><td>Mean κk</td><td>Mean accepted</td><td>Accepting a conflicting pair</td></tr><tr><td>Map 1</td><td>Start (round 1)</td><td>20</td><td>2.95</td><td>34/58</td><td>34</td><td>2.20</td><td>2.35</td><td>11</td></tr><tr><td></td><td>Repair (unsafe base)</td><td>57</td><td>1.89</td><td>36/72</td><td>21</td><td>1.49</td><td>1.42</td><td>2</td></tr><tr><td></td><td>Refinement (safe base)</td><td>36</td><td>0.61</td><td>1/5</td><td>1</td><td>1.03</td><td>0.61</td><td>0</td></tr><tr><td>Map 2</td><td>Start (round 1)</td><td>20</td><td>2.90</td><td>19/56</td><td>19</td><td>1.75</td><td>2.85</td><td>13</td></tr><tr><td></td><td>Repair (unsafe base)</td><td>35</td><td>1.63</td><td>21/30</td><td>7</td><td>1.49</td><td>1.17</td><td>1</td></tr><tr><td></td><td>Refinement (safe base)</td><td>56</td><td>0.32</td><td>2/4</td><td>2</td><td>1.04</td><td>0.30</td><td>0</td></tr></table>

Table E.10: Realized coupling in the best-combination runs by phase: number of rounds, mean number of improving proposals, conflicting pairs among pairs of improving proposals, conflicts in which the two proposals together create a collision that neither causes alone, mean efective number of colors, mean number of accepted blocks, and rounds whose accepted combination contains a conflicting pair.

![](images/ebcffe646256810f70a501bcba73579f21f9eeba4ad77b0be5c86346da006af1.jpg)  
(a) Map 1

![](images/ef83d1fd1d80008f79ebde9309aef367e1387fd2d819871fdeb22db5309c8022.jpg)  
(b) Map 2

![](images/ffb81ceba8d36b47d5b8b87cfab7c91ed37ba20ac0994cae5cfc4450178c23fd.jpg)  
(c) Two floors  
Figure E.5: Parallel search time (solid) against the sequential replay of the same answers (dashed) for Team (Alg. 1), Team (Alg. 2) and Team unsafe; mean gap ${ \boldsymbol { F } } - { \boldsymbol { F } } ^ { * }$ on a logarithmic axis, twenty runs per curve; the single agent for reference.

The same model in both policies. Full-plan search can revise several ownership groups jointly and has access to all assignments and full-plan feedback. On one map, our block protocol with Algorithm 1 instead commits at most one proposed block per round, even though three searches run concurrently. External conflicts omitted from the available objective must first be exposed by verification; revising a route against the refreshed peer trajectory can require another response in a later round. Block commitment, conflict discovery and repair are therefore serial steps that parallel search does not remove; from the all-waiting start, the team needs at least three rounds to accept routes for all three agents. A model that reliably writes a correct full plan in one answer avoids this overhead, which is why a single agent that improves one route at a time can fall faster at first.

A sequential block control. The sequential replay replays the recorded team histories on a sequential worker (Figure E.5). It completes the three private searches in turn with the same �, objective and feedback. The base, retained pairs and peer trajectories remain frozen until all proposals are available, then the verifier applies the team’s merge rule.

Committing before the next search would change the policy. Replay preserves every answer, selection and round loss, changing only timestamps; it makes no new model requests.

For a fixed sequence of checked responses, let $t _ { k b } =$ $\sum _ { h }$ � be block �’s response time in round �. The two search times would be

$$
\begin{array} { r } { T _ { \mathrm { s e q u e n t i a l } } = \displaystyle \sum _ { k } \sum _ { b } t _ { k b } , } \\ { T _ { \mathrm { p a r a l l e l } } = \displaystyle \sum _ { k } \operatorname* { m a x } _ { b } t _ { k b } . } \end{array}\tag{E.8}
$$

Thus $T _ { \mathrm { p a r a l l e l } } \leq T _ { \mathrm { s e q u e n t i a l } }$ , and for fixed durations �<sub>sequential</sub> is at most three times $T _ { \mathrm { p a r a l l e l } }$ (six times on two floors). With prompt v2 the replay gives 853.9±374.2 against 481.0±227.3 seconds on the first map (ratio of means 1.78), 778.2 ± 304.6 against 482.3±201.6 on the second (1.61), and 1632.1±468.5 against 656.4 ± 206.1 on two floors (2.49), where six agents search at once; with prompt v1 it gave 1207.14 ± 165.61 against 589.39 ± 64.85 on the first map (2.05). For the other merge rules the ratios are 1.77, 1.68 and 2.48 with best combination and 1.96, 1.80 and 2.71 for the unsafe team on map 1, map 2 and two floors (Figure E.5). Played in turn, the team with Algorithm 1 would be slower than the one-route single agent on every task, so parallel execution is what makes the team fast; its better plans do not depend on it. Both follow (C.1). Actual sequential execution could change serving latencies and samples, so the replay is not a measured variant. Per-round durations, clocks, loss sequences and curves are retained with the analyses.

## F SQLite: acquiring and reusing feedback

This appendix gives the complete database experiment: the data and query choices, feedback acquisition policies, timing procedure and recorded results. The study isolates the cost of obtaining and reusing execution information.

## F.1 Purpose of the SQLite study

The robot-routing study (Section 8.1, Appendix E) is our primary experiment with live LLMs. This second study isolates a diferent question: can execution of a useful task supply reusable information more cheaply than calibrating alternative decisions in advance? We test this question in a real SQLite engine, with constructed data and deterministic planning procedures. The procedures are not language models, and the experiment does not measure parallel-agent inference. The synthetic study in Appendix G supplies the controlled connection to the convergence assumptions; this database study supplies a measured application of feedback acquisition.

Prior work already uses execution feedback to improve query planning. LEO corrects cost estimates from observed executions; Bao learns to guide an existing optimizer (Stillger et al., 2001; Marcus et al., 2021). We do not propose a new optimizer; we use this setting to measure the cost of acquiring and processing selected, reusable observations.

## F.2 Data, queries and available decisions

Each report combines three aggregate queries, one per source; the main text calls a report simply a query. For source $s \in$ {0, 1, 2}, tables $A _ { s }$ and $C _ { s }$ hold endpoint objects with group identifiers, and table $B _ { s }$ holds edges with endpoint IDs and an integer amount. $\mathrm { A }$ query selects a group $g _ { s }$ in $A _ { s }$ and a group $h _ { s }$ in $C _ { s }$ , then returns the count and sum of amounts of edges connecting the selected objects. The report is one SQL UNION ALL of the three aggregate branches. Correctness is independent of the selected join direction.

Each endpoint table has 4,096 groups and two objects per group, hence 8,192 rows. Half of the $A _ { s }$ groups are randomly designated high-degree: each of their objects has 400 outgoing edges, while every other object has one. Thus $B _ { s }$ contains

$$
2 \cdot 2 , 0 4 8 \cdot 4 0 0 + 2 \cdot 2 , 0 4 8 = 1 , 6 4 2 , 4 9 6
$$

edges. Independently, a randomly chosen half of the $C _ { s }$ groups receives 97% of the generated edges. Membership is not encoded in group IDs. Edge amounts are integers sampled from 1 through 9,999. Indexes cover endpoint groups and both edge endpoint IDs; ordinary ANALYZE is performed. The three-source database contains 4,927,488 edge rows in total and remains static throughout a workload.

For each branch, the controller chooses one of two physical plans: start from the selected $A _ { s }$ group and traverse toward $C _ { s } \ \mathrm { ( l e f t ) }$ , or start from $C _ { s }$ and traverse toward $A _ { s }$ (right). CROSS JOIN and the relevant edge index enforce these alternatives; the native control lets SQLite choose. All alternatives compute the same aggregate, including when no edges match.

Let $v _ { s } ( \ell , g )$ and $v _ { s } ( r , h )$ be the numbers of edges incident to the selected left and right groups. Their sum over the three selected directions defines a separable work surrogate,

$$
J _ { q } ( a ) = \sum _ { s = 0 } ^ { 2 } v _ { s } ( a _ { s } , q _ { s } ) , \qquad a \in \{ \ell , r \} ^ { 3 } .\tag{F.1}
$$

Fan-out predicts work, but is not identical to elapsed time or total VM steps. The experimental outcomes are the independently measured execution and controller costs, rather than treating (F.1) as exact runtime. There is no collision-like coupling between branches: a centralized factorized controller can make the same choices without enumerating all eight joint plans.

## F.3 Acquiring a persistent correction

The controller stores estimates for each source, side and group. Unknown entries start at 401, the source’s average fan-out. It chooses the direction with the lower estimate, breaking ties in the order preferred by the native planner. After execution, the engine’s NVISIT counter reports how many edge-index entries the chosen branch visited (SQLite, 2026). This replaces the estimate for the executed side’s group. The estimate is retained for later queries with that group, even if the opposite endpoint predicate changes. Unexecuted alternatives are not revealed automatically.

The selective-feedback policy retrieves the detailed profile only if the completed report used more than 1,000 SQLite virtual-machine steps. A trigger updates all three executed branches. The engine collects counters even on reports below this threshold; the gate saves retrieval and processing, not collection. Both instrumentation and retrieval costs remain in the learner’s measured time. The threshold was selected during development and frozen before confirmation on the three fresh database seeds.

The observation changes the decision model rather than the answer to the query: it is reused cost information, the learned counterpart of a correction. Because each report brings new predicates, the stream is not a descent on one fixed solution; the study therefore measures the acquisition trade-of of Section $7$ rather than a convergence rate.

## F.4 Workloads and controls

The favorable stream has 48 reports. In each source, fivesixths of branch queries pair a low-degree $A _ { s }$ group with a high-degree $C _ { s }$ group, favoring the initial left choice. The remaining sixth repeatedly uses two high-degree $A _ { s }$ groups with varying low-degree $C _ { s }$ groups. Outliers are staggered across sources: half of reports contain one such branch and half contain none. A 192-report stream repeats this same cycle four times, testing how advance calibration amortizes. A balanced 48-report stream mixes the two patterns equally within each source, with less reuse of recurring costly groups. The workloads were deliberately constructed and developed adaptively.

<table><tr><td>Quantity</td><td>Fixed value</td></tr><tr><td>Sources per report</td><td>3</td></tr><tr><td>Groups per endpoint table</td><td>4,096</td></tr><tr><td>Objects per group</td><td>2</td></tr><tr><td>High/low outgoing degree per object Edges per source</td><td>400  / 1 1,642,496</td></tr><tr><td>Probability of a high-degree right group</td><td>0.97</td></tr><tr><td>Unknown fan-out estimate</td><td>401</td></tr><tr><td>Profile trigger (strictly greater than)</td><td>1,000 VM steps</td></tr><tr><td>Fresh database seeds</td><td></td></tr><tr><td>Timing repetitions per seed and workload</td><td>3 3</td></tr></table>

Table F.1: Fixed database and feedback settings. The 0.97 probability governs generation of edge endpoints, not a success probability of the controller.

The main controls include native planning, a fixed left plan, samples of 128 or 2,048 edges per source, complete joined histograms, exact cached active-group counts, and bounded cached counting. The strongest counting control first counts the predicted better side, then limits counting of its alternative once that alternative cannot improve the decision. It batches requests across sources and retains exact counts and reusable lower bounds. Calibration queries and their execution costs are charged. This control does not repeatedly identify every possible decision.

Cached plan trials execute both local directions when a predicate pair first appears, retain the lower-VM-step direction and reuse it. The first trial supplies the requested answer and is charged as productive work, while the other is calibration. This avoids charging both alternatives as pure overhead. Further controls retrieve feedback on every report, use centralized selective feedback with the same observations, or freeze feedback after four reports. The centralized controller in the main timing table enumerates eight joint surrogate choices; a separate factorized implementation removes that unnecessary enumeration and independently checks decision equivalence.

## F.5 Execution, timing and correctness

The engine is SQLite 3.46.1, compiled from pinned public source with statement scan-status support. Fresh database seeds are 51031, 68437 and 89759. Source-specific generation uses seed $s _ { \mathrm { d b } } + 1 8 , 1 8 1 s .$ and the query generator uses $s _ { \mathrm { d b } }$ + 818,181. There are three timing repetitions per database and workload, with randomized policy order. The instrumentationcontrolled record contains $3 \times 3 \times 3 \times 1 2 = 3 2 4$ complete workload timings. These are three independent datasets, not nine independent datasets per table entry.

The timer covers the complete Python/SQLite controller: calibration, execution, instrumentation, profile processing, correctness assertions and trace bookkeeping. Data generation, index construction, ordinary ANALYZE, connection opening and initial native-plan inspection lie outside this timer. Initial database preparation took 10.76–12.21 seconds per seed and produced approximately 193.7 MB per database; it is separately recorded. Measurements use warm local execution, without artificial network or evaluator delays. Baselines that do not consume profiles have scan-status collection disabled in the timing comparison. VM steps are engine instruction counts, not CPU instructions or a measured hardware time conversion.

During confirmation, every reported answer is checked against a Python reference accumulated independently while generating records. The instrumentation-control run replays those already verified answers. A separate ofline audit validates all 483 extracted group-degree measurements in six 48-report streams. A factorized centralized control with the same observations matches all local choices and VM totals in 27 replayed streams. Regenerating the appendix independently recomputes the summary statistics from the 324 saved timings; it does not rerun the benchmark.

## F.6 Results and a recorded correction

Selective feedback takes 4.92 ms on the favorable 48-report stream, compared with 7.53 ms for bounded cached counting: 0.65 of the time, an absolute saving of 2.61 ms. Per database, the relative time ranges from 0.62 to 0.71. Only six reports per database trigger detailed profiles. At 192 reports, the relative time rises to 0.84 as advance calibration amortizes. Feedback after every report costs 7.15 ms on the shorter stream, showing why retrieval and processing should be charged.

The balanced stream reverses the result: selective feedback costs 17.13 ms versus 10.45 ms for bounded counting, or 1.64 times as much. It retrieves profiles on 48, 47 and 48 reports for the three databases, respectively. Learning from feedback thus pays of when costly groups are few and recur, and advance calibration is preferable when they are frequent; this is the trade-of predicted in Section 7.

Time and VM work need not rank policies identically. On the favorable 48-report stream, bounded counting has lower median VM work (34,433 steps versus 48,106 for selective feedback), but performs 96 additional calibration statements per run. Selective feedback needs no calibration statements. Its lower controller time is consistent with saving these invocations; lower VM work alone does not explain the ranking. Figure F.1 shows both measurements: controller time in panel (a) and VM work in panels (b, c).

One concrete trace illustrates how reuse changes later choices. On database seed 51031, zero-indexed report 0 uses source-0 groups (0, 2470) and chooses left. Its edge-index scan visits 800 entries, replacing the initial left-group estimate 401 with 800. At report 12, group 0 recurs with the diferent right group 2199; the retained estimate makes the controller choose right. Total work of the three-branch report is 337 VM steps, compared with 6,521 for report 0. These are diferent predicate pairs, so the diference is a trace illustration, not a controlled single-query speedup. Group 1 shows the same pattern at reports 6 and 18, with 6,521 and 345 VM steps. The later reports stay below the profile threshold and do not request new detailed feedback.

<table><tr><td>Policy</td><td>Favorable, 48 reports</td><td>Favorable, 192 reports</td><td>Balanced, 48 reports</td></tr><tr><td>Native planner</td><td>6.64</td><td>23.44</td><td>13.27</td></tr><tr><td>Always left</td><td>6.54</td><td>25.05</td><td>15.77</td></tr><tr><td>128-edge sample</td><td>13.61</td><td>32.40</td><td>22.76</td></tr><tr><td>2,048-edge sample</td><td>36.47</td><td>51.29</td><td>40.16</td></tr><tr><td>Full joined histograms</td><td>433.34</td><td>440.16</td><td>439.11</td></tr><tr><td>Exact counts, cached</td><td>10.78</td><td>21.19</td><td>10.88</td></tr><tr><td>Bounded counts, cached</td><td>7.53</td><td>18.71</td><td>10.45</td></tr><tr><td>Plan trials, cached</td><td>152.60</td><td>158.85</td><td>105.29</td></tr><tr><td>Feedback every report</td><td>7.15</td><td>21.18</td><td>16.64</td></tr><tr><td>Selective feedback</td><td>4.92</td><td>15.74</td><td>17.13</td></tr><tr><td>Centralized selective feedback</td><td>4.86</td><td>16.31</td><td>17.08</td></tr><tr><td>Feedback frozen after four</td><td>7.08</td><td>24.39</td><td>15.30</td></tr></table>

Table F.2: Controller time of all policies, in milliseconds. Each entry is the pooled median of nine timings: three repetitions on each of three independent databases. All answers pass the recorded correctness checks. Calibration, feedback processing and policy-specific instrumentation are included.

![](images/b302c34244345f2b7eff1dbfd271fac4c83716e85150c76d7053f0acc4794874.jpg)  
(a) Controller time

![](images/415bc89c5e2993b7d971234c7573c2710a3382cf840105ecbf883cd699365f57.jpg)  
(b) VM work, favorable stream

![](images/7e51c2332badc82728d13655ef4d74c862c243f2b7b32c05a00cd689ad43c2e2.jpg)  
(c) VM work, balanced stream  
Figure F.1: (a) Controller time for the principal controls: bars show the mean of the three per-database timing medians; white points are those medians and whiskers their minimum and maximum. Table F.2 reports pooled medians, and all policies compute the same correct answers. (b, c) Accumulated SQLite VM work for the favorable and balanced 48-report streams, including calibration, which is charged before its associated report. Lines average the three database trajectories and light bands show their minimum and maximum; repeated timing runs have identical VM counts within a database. VM work measures engine work, not elapsed time: the archive records complete controller time only at the end of a workload.

The centralized selective learner makes the same decisions. Its separately audited factorized version has pooled-median controller times 5.35, 16.28 and 16.99 ms for the three workloads. These replays check implementation and information equivalence. A supplementary SQLite STAT4 control, with richer statistics granted for free and profiling disabled, records 6.14, 22.46 and 12.90 ms of SQL-only time; it excludes the outer controller and is therefore not directly comparable with Table F.2.

## F.7 Relation to the assumptions

For a fixed report, the two-direction action spaces are finite, the known additive surrogate has independent source factors, and its blockwise minimization is exact. Thus the planner selects a surrogate minimizer, rather than exhibiting stochastic LLM search quality. If exact fan-outs were supplied for that fixed report, the additive fan-out objective would have block error-bound constant one. These structural facts concern the work surrogate in (F.1), not the controller’s actual runtime over the workload.

Assumption 2 holds for every $\rho _ { b } < 1$ , since each two-choice surrogate is minimized exactly. Successive reports change predicates instead of improving one solution under a fixed $F ,$ so Theorem 2 does not apply to the stream, and the learned table is updated by direct replacement of observed entries, the coordinate case of the sketch-and-project update of Appendix $_ { \mathrm { A . 4 , } }$ from per-branch counters rather than from scalar objective diferences. The study instead tests the cost model of Section 6: whether the information a correction needs is cheaper to acquire from useful work or by measuring alternatives in advance. Theorem 2 is checked with measured constants in the routing study and with certified constants in the synthetic study.

## F.8 Scope and retained data

The workload is constructed to make the trade-of visible. Its joins are highly selective: 431 of 432 audited favorable branch aggregates and 429 of 432 balanced aggregates contain no matching edges, although finding no match still incurs execution work. There are three independent databases, totals are in milliseconds, the branches are separable, and host hardware details were not recorded.

The result is a measured acquisition trade-of: useful executions can reveal a small recurring set of missing costs cheaply enough to outperform a strong cached calibration control, whereas frequent or poorly transferable observations reverse the comparison. The appendix uses the final instrumentationcontrolled results.

## G Synthetic binary optimization

This appendix specifies the synthetic task and its four experimental series. It provides the assumption certificates, the cost of private work, the error floor under approximate corrections, and results for learning and reusing corrections.

## G.1 Purpose and experimental scope

This third study directly instantiates the finite blockoptimization protocol with an analytically tractable randomized local solver. It asks how an objective correction changes progress, how many local steps are worthwhile between synchronizations, and whether learning or reusing the correction repays its query cost. Unlike the routing experiments (Appendices C–E), it contains no LLM inference. Unlike Appendix F, its convergence constants can be established on the entire task domain. All series use one fixed 48-variable task.

The four series are: exact correction with varying private horizon; approximate correction with a persistent bias; correction learned through charged scalar evaluations; and reuse of already-paid proposal evaluations with an observable stopping certificate. Their stopping rules and cost conventions are stated separately. Each series has its own script, and the tables and figures are rebuilt from the saved results.

<table><tr><td>Block</td><td>True target  $t _ { b }$ </td><td>Private target  $\tilde { t } _ { b }$   $\pi _ { b }$ </td></tr><tr><td>1</td><td>10100000</td><td>01011111 0.5</td></tr><tr><td>2</td><td>01000001</td><td>01000001 0.6</td></tr><tr><td>3</td><td>00100010</td><td>11011101 0.7</td></tr><tr><td>4</td><td>10000010</td><td>10000010 0.8</td></tr><tr><td>5</td><td>11011001</td><td>00100110 0.9</td></tr><tr><td>6</td><td>11111101</td><td>11111101 1.0</td></tr></table>

Table G.1: The single fixed 48-bit task. Bit strings list coordinates 1 through 8 in order. Random solver trajectories change search randomness, not this task.

G.2 Objective, blocks and all fixed settings The objective is (24): six blocks $x _ { b } \in \{ 0 , 1 \} ^ { 8 }$ on the path 1–2– $3 { - } 4 { - } 5 { - } 6$ , where neighboring blocks pay for disagreements of corresponding bits and each bit pays for disagreeing with its target $t _ { b j }$ . The coupling weight is $w = 0 . 2$ . Targets are sampled first as independent binary values, then unary weights are sampled uniformly from [1, 1.5] using NumPy’s default random generator with seed 20260925. The actual targets, all 48 weights and the source are retained; subsequent experiments read this same instance rather than sampling another task. Initialization is $x ^ { 0 } = 1 - t .$ . The unique minimizer is �, with $F ^ { * } = 3 . 4 _ { \mathrm { : } }$ and the initial objective gap is $\Delta _ { 0 } = 5 8 . 6 0 4 7 7 7 5 1 3 5$

The private objective $\begin{array} { r } { L _ { b } ( u ) = \sum _ { i } a _ { b j } | u _ { j } - \tilde { t } _ { b j } | } \end{array}$ omits coupling and uses the private target $\tilde { t } _ { b }$ . Blocks 1, 3 and 5 use $\tilde { t } _ { b } = 1 - t _ { b }$ , while the other blocks have $\tilde { t } _ { b } = t _ { b }$ . This deliberately inserts a local/global mismatch. Activity probabilities �<sub>�</sub> range from 0.5 to 1.0. The complete target and activity assignment is given in Table G.1.

## G.3 Exact correction and search

With neighbors frozen at �, the coeficient of candidate bit $u _ { j }$ in the global conditional objective is

$$
A _ { b j } ( x ) = a _ { b j } ( 1 - 2 t _ { b j } ) + w \sum _ { r : | r - b | = 1 } ( 1 - 2 x _ { r j } ) .\tag{G.1}
$$

The corrected score is afine. Defining

$$
h _ { b j } ( x ) = A _ { b j } ( x ) - a _ { b j } ( 1 - 2 \tilde { t } _ { b j } ) ,
$$

the anchored correction is $\begin{array} { r } { R _ { b } ( u ; x ) = \sum _ { i } h _ { b j } ( x ) ( u _ { j } - x _ { b j } ) } \end{array}$ . It exactly equals the global conditional change minus the local change. A constant ofset could not reverse the wrong private preference; the candidate-dependent vector does. Eight coeficients per block replace a table of 256 candidate values. Neighbor contributions are recomputed at each synchronization and remain fixed during local search.

Each epoch starts from the same shared base for all six searches. At a private step, block � chooses a coordinate uniformly and, with probability $\pi _ { b } ,$ sets it to minimize its afine corrected score; otherwise it leaves the bit unchanged. After � steps, all six proposed singleton replacements are checked with the exact global objective. Their nonnegative verified gains are summed over the two colors {1, 3, 5} and {2, 4, 6}.

The larger-gain color is committed, using only strictly improving blocks. Ties select the first color. Same-color edits share no objective factor, so their verified gains add exactly. This difers from routing’s complete three-block dependency graph, where each accepted color contains only one block.

The exact-correction series uses the horizons $H = 1 , 2$ 4, 8, 16, 32, 64 and 128, with 2,000 solver trajectories per horizon. Solver randomness uses seed $2 0 2 6 0 9 2 5 + H$ . Each corrected run has a ceiling of 512 epochs and stops on exact attainment of the ofline-known optimum; every run succeeds before that ceiling. The uncorrected $H = 8$ ablation uses only the private unary scores and a ceiling of 128 epochs. It solves none of 2,000 runs and leaves mean gap 30.979, so the correction is necessary on this task.

## G.4 Certifying the assumptions

We establish correction accuracy, local contraction and block geometry for this fixed task. These certificates determine the constants used in the comparison between the theoretical bound and observed convergence.

Correction accuracy. The exact vector satisfies Assumption 1 with $\delta _ { b } ^ { * } = \delta _ { b } ^ { U } = 0 ;$ indeed its error is zero for every binary candidate and neighbor context. The saved implementation checks 200 sampled exact-correction identities, with maximum discrepancy $1 . 3 3 \times 1 0 ^ { - 1 4 }$ , and 10,000 anchored approximate-correction inequalities. The algebraic identity supplies the uniform guarantee; sampled checks audit its implementation.

If a correction computed at neighbor context � is reused after neighbors change, reanchoring at the current owned block gives the valid bound

$$
\operatorname* { s u p } _ { u } | \widehat { R } _ { b } ( u ; x ) - R _ { b } ( u ; x ) | \leq 2 w \sum _ { r : | r - b | = 1 } \| z _ { r } - x _ { r } \| _ { 1 } .
$$

The experiment refreshes context every epoch, so this staleness error is zero. Obtaining changed neighbor context is part of synchronization, not an assumed free oracle.

Conditional local contraction. Each unresolved coordinate’s nonnegative contribution to afine suboptimality disappears with probability $\pi _ { b } / 8$ per private step. Linearity of conditional expectation gives Assumption 2 with $\rho _ { b } = \pi _ { b } / 8$ , uniformly over bases and coeficient vectors. Thus

$$
\rho _ { \mathrm { m i n } } = 0 . 0 6 2 5 , \qquad q ( H ) = 1 - ( 1 - 0 . 0 6 2 5 ) ^ { H } , \qquad \kappa = 2 .\tag{G.2}
$$

This is an analytical property of the coordinate solver, not a value fitted to observed trajectories.

Block geometry and reference optimum. Let $c _ { b } = w \deg ( b )$ which is 0.2 for endpoints and 0.4 for interior blocks. Every $a _ { b j } > c _ { b }$ . Repairing a wrong bit to $t _ { b j }$ therefore saves more unary cost than it can add in incident coupling cost. This proves uniqueness of � and gives a uniform valid block error bound

$$
\mu _ { \mathrm { d o m } } = \operatorname* { m i n } _ { b , j } { \frac { a _ { b j } - c _ { b } } { a _ { b j } + c _ { b } } } = 0 . 4 3 0 7 8 5 5 7 6 4 .\tag{G.3}
$$

Indeed, for wrong-bit set $I ( x )$

$$
\sum _ { b } D _ { b } ( x ) \geq \sum _ { ( b , j ) \in I ( x ) } ( a _ { b j } - c _ { b } ) , \quad \Delta ( x ) \leq \sum _ { ( b , j ) \in I ( x ) } ( a _ { b j } + c _ { b } ) .
$$

Their ratio establishes Assumption 3 on the full domain.

The tighter finite-state certificate is $\mu = 0 . 7 1 9 3 5 5 0 1 5 8 .$ It does not require enumeration of $2 ^ { 4 8 }$ joint configurations. The objective separates across the eight bit positions: each feature has just $2 ^ { 6 } = 6 4$ block patterns. For each feature we enumerate its exact minimum and all singleton improvements, then minimize the improvement-to-gap ratio over its nonoptimal states. Full-state gaps and best-block improvements sum over features, so the full-state ratio is a weighted average of their ratios; the least feature ratio is valid globally and is attained by making only that feature nonoptimal. The total certificate uses $8 \times 6 4 = 5 1 2$ states. A separate appendix builder independently recomputes $F ^ { * }$ and this constant from the retained coeficients.

Theory-to-data comparison. Using this certified $\mu ,$ exact corrections and (G.2), Theorem 2 gives

$$
\mathbb { E } F ( x ^ { K } ) \le F ^ { * } + \left( 1 - \frac { \mu q ( H ) } { 2 } \right) ^ { K } \Delta _ { 0 } .\tag{G.4}
$$

Figure 5(b) compares this sequence with the sample means for $H = 8$ and 32 at integer epochs; no coeficient is fitted. The mean gap stays below the bound at every epoch. Completed trajectories remain at $F ^ { * }$

## G.5 Private work and synchronization

One parallel private slot costs $c = 1 ;$ ; shared preparation, transfer, verification and integration cost � per epoch. In the known-correction series, verification is included in this common overhead. A run finishing at epoch $K _ { \mathrm { e n d } }$ has modeled cost $( H + C ) K _ { \mathrm { e n d } }$ . We show $C \in \{ 0 , 5 , 2 0 , 1 0 0 \}$ , including the idealized zero-overhead control. Costs are in modeled work units; six agents use six local slots per private step in total.

Figure 5(c) plots these costs. At $C = 2 0$ , mean cost decreases from 1296.4 at $H = 1$ to 176.7 at $H = 6 4$ , then rises to 296.0 at $H = 1 2 8$ . The observed best horizons among the eight tested are 1, 16, 64 and 64 for $C = 0 , 5 , 2 0 , 1 0 0$ . The relaxed cost criterion of Section A.3 gives integer minimizers $H _ { * } = 1$ , 11, 20 and 35; the ceiled expected-gap criterion gives � = 22 at $C = 2 0$ when $\varepsilon = 0 . 0 1 \Delta _ { 0 }$ and $H = 1 , \ldots , 2 5 6$ are searched. Both move right with $C ,$ as the observed minima do. Figure 5(c) places each �<sub>\*</sub> on its cost curve by log-log interpolation between the tested horizons; the interpolated cost at � is 1.00, 1.03, 1.16 and 1.29 times the best tested cost for $C = 0 , 5 , 2 0 , 1 0 0$ . The criterion is smaller for large � because it minimizes a worst-case bound on the expected gap, whereas the experiment measures first exact completion: with $H \geq 6 4$ nearly every block is solved exactly and runs finish in $2 . 0 { - } 2 . 1$ epochs on average, one per color, faster than the geometric rate of the bound.

<table><tr><td>H</td><td>Completion epochs, mean ± SE</td><td>Cost, C = 0</td><td> $\mathrm { C o s t } , C = 5$ </td><td> $\mathrm { C o s t } , C = 2 0$ </td><td> $\mathrm { C o s t } , C = 1 0 0$ </td></tr><tr><td>1</td><td> $6 1 . 7 3 \pm 0 . 3 7$ </td><td>61.7</td><td>370.4</td><td>1296.4</td><td>6235.1</td></tr><tr><td>2</td><td> $3 3 . 6 0 \pm 0 . 1 8$ </td><td>67.2</td><td>235.2</td><td>739.2</td><td>3427.2</td></tr><tr><td>4</td><td> $1 8 . 1 9 \pm 0 . 0 9$ </td><td>72.8</td><td>163.7</td><td>436.6</td><td>1891.8</td></tr><tr><td>8</td><td> $1 0 . 3 4 \pm 0 . 0 5$ </td><td>82.7</td><td>134.4</td><td>289.4</td><td>1116.4</td></tr><tr><td>16</td><td> $6 . 0 1 \pm 0 . 0 3$ </td><td>96.1</td><td>126.2</td><td>216.3</td><td>696.9</td></tr><tr><td>32</td><td> $3 . 4 9 \pm 0 . 0 2$ </td><td>111.6</td><td>129.1</td><td>181.4</td><td>460.5</td></tr><tr><td>64</td><td> $2 . 1 0 \pm 0 . 0 1$ </td><td>134.6</td><td>145.1</td><td>176.7</td><td>345.0</td></tr><tr><td>128</td><td> $2 . 0 0 \pm 0 . 0 0$ </td><td>256.0</td><td>266.0</td><td>296.0</td><td>456.0</td></tr></table>

Table G.2: Exact correction: all 2,000 trajectories per horizon reach the optimum on the same task. Standard errors reflect solver randomness. Cost is modeled parallel work plus synchronization overhead.

## G.6 Biased corrections and the floor

Theorem 2 allows a correction error that does not vanish to leave a floor that more private work cannot remove; this series shows that such a floor appears. The exact-correction series cannot show this, because its error is zero. This series keeps the task, solver, verification and merge above, but agents search the exact conditional coeficients plus a fixed bias,

$$
\widehat { A } _ { b j } ( x ) = A _ { b j } ( x ) + \nu s _ { b j } , \qquad s _ { b j } \in \{ - 1 , 1 \} ,
$$

with independent uniform signs drawn once per run and unknown to the agents. The corrected objective remains afine in $\boldsymbol { u } - \boldsymbol { x } _ { b }$ and hence anchored; its error at a candidate � is $\sum _ { j } \nu s _ { b j } ( u _ { j } - x _ { b j } )$ . We test $\nu \in \{ 0 , 0 . 2 5 , \ldots , 2 \}$ and $H \in \{ 8 , { \bar { 3 2 } } \}$ with 2,000 runs per configuration and 160 epochs, using seed 20261009 ofset by $1 0 0 0 H + 1 0 0 \nu .$

For every search the series records the correction errors $e _ { b } ^ { * } = | \widehat { f } _ { b } ( u _ { b } ^ { * } ) - f _ { b } ( u _ { b } ^ { * } ) |$ at a true block minimizer $u _ { b } ^ { * }$ and $e _ { b } ^ { U } =$ $| \widehat { f _ { b } } ( U _ { b } ) - f _ { b } ( U _ { b } ) |$ at the returned proposal $U _ { b } ,$ , the realized fraction $Q _ { b }$ of the corrected improvement, and the exact block improvement $D _ { b } ; \ e _ { b } ^ { * }$ and $e _ { b } ^ { U }$ are the realized counterparts of $\delta _ { b } ^ { * }$ and $\delta _ { b } ^ { U }$ . The argument of Lemma 1, applied to these realized quantities, gives $g _ { b } \ge \operatorname* { m a x } \{ 0 , Q _ { b } ( D _ { b } - e _ { b } ^ { * } ) - e _ { b } ^ { U } \}$ }; the merge commits at least the larger color sum of these lower bounds. Propagating this inequality separately for each run from the initial gap yields an envelope that the observed gap never exceeds. The script asserts the local inequality, the merge inequality and the envelope for every run and epoch, and confirms that the envelope equals the observed gap when $\nu = 0$

The smallest magnitude of an exact coeficient is min<sub>�</sub> $_ { , j } ( a _ { b j } \ : - \ : c _ { b } ) = 0 . 6 0 5 \ : \geq \ : 0 . 6$ . A bias below it never reverses the sign of a coeficient, so decisions and trajectories coincide with exact correction and every run reaches the optimum. Above it, a bit whose coeficient magnitude is below the bias and whose bias opposes the required change is never repaired, and the gap settles at a floor that grows with �. Once the bias exceeds every coeficient magnitude (at most $a _ { b j } + 2 w \leq 1 . 9 )$ , about half of the coeficients are reversed and the floor approaches the uncorrected level 31. The floors for � = 8 and $H = 3 2$ difer by at most 0.49 at every $\nu ,$ against values up to 30: longer private search reaches the floor sooner but does not lower it. The envelope is conservative for small biases because it charges both two-point errors in every search, even when they do not change a decision. A uniform bound would be $\delta _ { b } = 8 \nu ,$ attained at the initial state in which every bit is wrong, and its floor $B _ { \delta } / \alpha$ is already 64 at � = 8 and 37 at $H = 3 2$ for $\nu = 0 . 2 5$ and exceeds the initial gap for every $\nu \geq 0 . 5 ;$ only the two-point errors make the bound informative here.

<table><tr><td rowspan="2">ν</td><td colspan="3"> $H = 8$ </td><td colspan="3"> $H = 3 2$ </td></tr><tr><td>Final gap</td><td>Solved</td><td>Bound</td><td>Final gap</td><td>Solved</td><td>Bound</td></tr><tr><td>0.00</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>0.0</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>0.0</td></tr><tr><td>0.25</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>10.6</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>6.8</td></tr><tr><td>0.50</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>20.2</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>2000</td><td>13.4</td></tr><tr><td>0.75</td><td> $0 . 3 0 \pm 0 . 0 1$ </td><td>1340</td><td>26.8</td><td> $0 . 2 7 \pm 0 . 0 1$ </td><td>1187</td><td>19.2</td></tr><tr><td>1.00</td><td> $3 . 0 8 \pm 0 . 0 4$ </td><td>48</td><td>30.4</td><td> $2 . 7 9 \pm 0 . 0 3$ </td><td>40</td><td>24.7</td></tr><tr><td>1.25</td><td> $1 3 . 1 2 \pm 0 . 0 7$ </td><td>0</td><td>36.9</td><td> $1 3 . 6 1 \pm 0 . 0 8$ </td><td>0</td><td>33.8</td></tr><tr><td>1.50</td><td> $2 3 . 4 2 \pm 0 . 1 0$ </td><td>0</td><td>45.8</td><td> $2 3 . 5 5 \pm 0 . 1 0$ </td><td>0</td><td>44.5</td></tr><tr><td>1.75</td><td> $2 6 . 6 6 \pm 0 . 1 0$ </td><td>0</td><td>51.5</td><td> $2 6 . 8 8 \pm 0 . 1 0$ </td><td>0</td><td>50.5</td></tr><tr><td>2.00</td><td> $3 0 . 0 1 \pm 0 . 0 9$ </td><td>0</td><td>55.6</td><td> $2 9 . 7 9 \pm 0 . 1 0$ </td><td>0</td><td>55.3</td></tr></table>

Table G.3: Approximate corrections: final mean gap after 160 epochs (± one standard error), runs reaching the optimum out of 2,000, and the mean per-run recurrence envelope.

## G.7 Learning the correction

This series tests the learning mechanism of Section 6 and Appendix A.4: agents start without the correction and learn its fixed coeficients from charged scalar evaluations. It keeps the same task, coupling and coordinate solver but hides the static unary correction coeficients. Writing

$$
\vartheta _ { b j } = a _ { b j } \big ( 1 - 2 t _ { b j } \big ) - a _ { b j } \big ( 1 - 2 \tilde { t } _ { b j } \big ) ,
$$

the algorithm starts at $\widehat { \vartheta } = 0$ with known bounds $M _ { b j } = 2 a _ { b j }$ The local scores and coupling are available; target bits and true coeficients are used only by the verifier and ofline diagnostics. A one-bit probe supplies exactly the scalar residual in (A.33), with displacement ±1, and identifies that coordinate. Coeficients are retained forever; changed neighbors afect only the recomputed coupling context.

Before each epoch’s � private steps, every block with unobserved coordinates probes with probability �, choosing uniformly among those coordinates. The tested probabilities are 1, 0.5, 0.25 and horizons are 8, 32, 64. The refresh baseline queries all 48 coeficients at every epoch; identify-once queries them only at initialization. Both use the same scalar interface and pay acquisition costs. An uncorrected baseline retains known coupling but never calibrates the unary correction; it difers from the earlier private-score-only ablation.

<table><tr><td>Policy</td><td>H</td><td>Completion epochs, mean ± SE</td><td>Mean verifier queries</td><td> $\mathrm { C o s t } , \mathrm { m e a n } \pm \mathrm { S E }$ </td></tr><tr><td>Learn  $( p = 1 )$ </td><td>8</td><td> $1 2 . 7 8 \pm 0 . 0 7$ </td><td>87.1</td><td> $4 4 4 . 8 \pm 1 . 9 6$ </td></tr><tr><td>Learn  $( p = 0 . 5 )$ </td><td>8</td><td> $2 1 . 3 2 \pm 0 . 1 3$ </td><td>85.5</td><td> $6 8 2 . 5 \pm 3 . 5 8$ </td></tr><tr><td>Learn  $( p = 0 . 2 5 )$ </td><td>8</td><td> $4 2 . 3 5 \pm 0 . 2 9$ </td><td>83.1</td><td> $1 2 6 8 . 9 \pm 8 . 0 9$ </td></tr><tr><td>Refresh every round</td><td>8</td><td> $1 0 . 2 4 \pm 0 . 0 7$ </td><td>524.8</td><td> $8 1 1 . 6 \pm 5 . 0 6$ </td></tr><tr><td>Identify once</td><td>8</td><td> $1 0 . 2 4 \pm 0 . 0 7$ </td><td>81.1</td><td> $3 6 7 . 9 \pm 1 . 8 9$ </td></tr><tr><td>Learn  $( p = 1 )$ </td><td>32</td><td> $8 . 8 6 \pm 0 . 0 2$ </td><td>80.9</td><td> $5 4 1 . 7 \pm 1 . 3 1$ </td></tr><tr><td>Learn  $( p = 0 . 5 )$ </td><td>32</td><td> $1 9 . 5 7 \pm 0 . 1 2$ </td><td>77.8</td><td> $1 0 9 5 . 4 \pm 6 . 3 9$ </td></tr><tr><td>Learn  $( p = 0 . 2 5 )$ </td><td>32</td><td> $4 0 . 4 5 \pm 0 . 2 9$ </td><td>76.4</td><td> $2 1 7 9 . 6 \pm 1 4 . 9 0$ </td></tr><tr><td>Refresh every round</td><td>32</td><td> $3 . 4 4 \pm 0 . 0 3$ </td><td>177.9</td><td> $3 5 6 . 8 \pm 2 . 5 9$ </td></tr><tr><td>Identify once</td><td>32</td><td> $3 . 4 4 \pm 0 . 0 3$ </td><td>60.8</td><td> $2 3 9 . 7 \pm 1 . 3 7$ </td></tr><tr><td>Learn  $( p = 1 )$ </td><td>64</td><td> $8 . 0 8 \pm 0 . 0 1$ </td><td>76.4</td><td> $7 5 5 . 3 \pm 0 . 8 1$ </td></tr><tr><td>Learn (p = 0.5)</td><td>64</td><td> $1 9 . 5 5 \pm 0 . 1 2$ </td><td>75.7</td><td> $1 7 1 8 . 0 \pm 9 . 7 8$ </td></tr><tr><td>Learn  $( p = 0 . 2 5 )$ </td><td>64</td><td> $4 0 . 4 4 \pm 0 . 2 8$ </td><td>75.1</td><td> $3 4 7 2 . 4 \pm 2 3 . 4 5$ </td></tr><tr><td>Refresh every round</td><td>64</td><td> $2 . 1 0 \pm 0 . 0 1$ </td><td>111.1</td><td> $2 8 7 . 7 \pm 1 . 2 8$ </td></tr><tr><td>Identify once</td><td>64</td><td> $2 . 1 0 \pm 0 . 0 1$ </td><td>58.1</td><td> $2 3 4 . 8 \pm 0 . 8 2$ </td></tr></table>

Table G.4: Every calibrated configuration in the learned-correction series: 1,000 trajectories each on the fixed task. Queries include initial evaluation, deliberate calibration and changed-proposal verification. Costs use $C = 2 0$ , unit parallel private-slot cost and one additional unit per counted scalar evaluation.

Each configuration has 1,000 trajectories with seed 20260926 + � + 100�. Controls at $p = 1$ use that same seed convention. Every calibrated configuration reaches the optimum. The uncorrected $H = 3 2$ case solves none of 1,000 trajectories within 256 epochs. These trajectories stop at the ofline-known optimum; the next series replaces this rule with an observable certificate.

The remaining coeficient radius $\Gamma _ { k }$ is observable and upper-bounds candidate correction error. With deliberate probing, Proposition A.7 applies with $\gamma = p / 8$ and at most $d _ { \mathrm { t o t } } = 4 8$ extra calibration queries over a whole run. The all-coordinate excitation assumption is unnecessary here, because an identified coeficient is not probed again; the binary specialization supplies the uncertainty analysis. Separate projection and coeficient-inequality checks audit the relevant identities.

Here a completed run costs $( H + 2 0 ) K _ { \mathrm { e n d } } + N _ { \mathrm { q } }$ , where $N _ { \mathbf { q } }$ counts the initial evaluation, calibration calls and verification of changed proposals. The base score is cached, unchanged proposals need no new call, and the selected color’s value follows from exact additivity. These query charges are explicit, unlike the exact-correction series where verification is included in $C ,$ so costs are compared within a series. Ofline diagnostic evaluations are excluded.

At $H = 8 , p = 1$ , gradual learning costs 444 $^ { . 8 , }$ compared with 811.6 for repeated refresh and 367.9 for identify-once (Figure G.1(a, b)). At this horizon, retaining learned coeficients thus nearly halves the cost of reidentifying them every epoch; at � = 32 and 64, where refresh runs finish in 3.4 and 2.1 epochs while gradual learning needs at least eight epochs to probe each block, repeated refresh is cheaper than gradual learning (356.8 against 541.7 and 287.7 against 755.3). When the hidden part is static and all 48 coeficients can be queried at once, identifying them once is cheapest; at larger � the gap widens because private steps taken before calibration is complete are wasted (gradual learning 541.7 and 755.3, identify-once 239.7 and 234.8). Reducing $p$ reduces total queries only slightly but increases epochs and cost. Gradual learning pays of instead when only a few of many unknown values are ever needed and they recur, as in the favorable SQLite workloads (Appendix F).

## G.8 Reuse and observable stopping

The fourth series compares dedicated calibration, dedicated calibration plus verification reuse, and identify-once at $p = 1$ and $H = 8 , 3 2 , 6 4$ , with 1,000 trajectories per configuration. After proposals are fixed, a singleton evaluation can identify one still-unknown coeficient when its displacement contains exactly one unknown coordinate after known contributions are subtracted. Both accepted and rejected proposals are eligible. Measurements with several unknown coordinates are skipped. Updating the estimator does not regenerate this epoch’s proposals; identified coeficients are used from the next epoch on. Identified coordinates are removed from future deliberate calibration, and calibration plus reused identifications never exceeds 48 per trajectory.

Stopping now uses an observable gap certificate. At the new committed base let $\widehat { A } _ { b j }$ be the private coeficient plus learned unary correction and refreshed coupling coeficient. Exact afine minimization gives the total surrogate improvement

$$
\widehat { D } ( x ) = \sum _ { b , j } \left[ \widehat { A } _ { b j } x _ { b j } - \operatorname* { m i n } \{ 0 , \widehat { A } _ { b j } \} \right] .
$$

<table><tr><td>Policy</td><td>H</td><td>Epochs, mean ± SE</td><td>Queries</td><td>Probes</td><td>Reused</td><td>Cost, mean ± SE</td></tr><tr><td>Dedicated</td><td>8</td><td> $1 2 . 7 8 \pm 0 . 0 7$ </td><td>87.1</td><td>48.0</td><td>0.000</td><td> $4 4 4 . 8 \pm 1 . 9 6$ </td></tr><tr><td>Verification reuse</td><td>8</td><td> $1 2 . 7 8 \pm 0 . 0 7$ </td><td>83.9</td><td>44.8</td><td>3.186</td><td> $4 4 1 . 6 \pm 1 . 9 6$ </td></tr><tr><td>Identify once</td><td>8</td><td> $1 0 . 2 4 \pm 0 . 0 7$ </td><td>81.1</td><td>48.0</td><td>0.000</td><td> $3 6 7 . 9 \pm 1 . 8 9$ </td></tr><tr><td>Dedicated</td><td>32</td><td> $8 . 8 6 \pm 0 . 0 2$ </td><td>80.9</td><td>48.0</td><td>0.000</td><td> $5 4 1 . 7 \pm 1 . 3 1$ </td></tr><tr><td>Verification reuse</td><td>32</td><td> $8 . 8 6 \pm 0 . 0 2$ </td><td>80.2</td><td>47.3</td><td>0.703</td><td> $5 4 1 . 0 \pm 1 . 3 0$ </td></tr><tr><td>Identify once</td><td>32</td><td> $3 . 4 4 \pm 0 . 0 3$ </td><td>60.8</td><td>48.0</td><td>0.000</td><td> $2 3 9 . 7 \pm 1 . 3 7$ </td></tr><tr><td>Dedicated</td><td>64</td><td> $8 . 0 8 \pm 0 . 0 1$ </td><td>76.4</td><td>48.0</td><td>0.000</td><td> $7 5 5 . 3 \pm 0 . 8 1$ </td></tr><tr><td>Verification reuse</td><td>64</td><td> $8 . 0 8 \pm 0 . 0 1$ </td><td>76.3</td><td>48.0</td><td>0.046</td><td> $7 5 5 . 2 \pm 0 . 8 1$ </td></tr><tr><td>Identify once</td><td>64</td><td> $2 . 1 0 \pm 0 . 0 1$ </td><td>58.1</td><td>48.0</td><td>0.000</td><td> $2 3 4 . 8 \pm 0 . 8 2$ </td></tr></table>

Table G.5: All verification-reuse configurations, with 1,000 trajectories each. Queries count every charged scalar evaluation, probes count deliberate calibration, and reused counts coeficients identified from already-paid proposal evaluations. Epoch and cost errors are standard errors of their means. Identify-once remains the strongest control.

![](images/a8629d731a040440825b8763310ca184333115a94d8c7740eb1bafbd2c98122d.jpg)  
(a) Learning, H= 8

![](images/def38815433c0eed76c08876a2530a59b03901e81d0ddc2e82318922eea08a99.jpg)  
(b) Acquisition cost

![](images/35478453cca4ca5c11b362e62ff14eb6e42a66a2124ec90f904f517a2e541c72.jpg)  
(c) Observable stopping

![](images/631a6c784d0372fad7cdf38a1a51c5e496ef2ede8b32797bf7a2781be272da88.jpg)  
(d) Verifier queries  
Figure G.1: Learning the correction and reusing verification. (a) Mean loss at � = 8 for the acquisition policies; refresh and identify-once have the same trajectory. (b) Mean completion costs of all acquisition policies at the three horizons, with one-standard-error bars. (c) True mean loss (solid) and $F ^ { * }$ plus the mean online stopping certificate (dashed) at $H = 8$ , for dedicated calibration and verification reuse; $F ^ { * }$ is added only to display the ofline comparison. (d) All counted verifier queries, averaged over 1,000 trajectories, including calibration and the initial evaluation.

If Γ(�) sums the remaining coordinate radii, (A.8) supplies

$$
\Delta ( x ) \leq \overline { { \Delta } } ( x ) : = \frac { \widehat { D } ( x ) + \Gamma ( x ) } { \mu _ { \mathrm { d o m } } } .\tag{G.5}
$$

The known weights and coupling determine $\mu _ { \mathrm { d o m } }$ in (G.3); the stopping decision does not use the hidden target or optimum. It costs $O ( 4 8 )$ arithmetic and no new verifier query, and stops when $\overline { { \Delta } } ( x ) < 1 0 ^ { - 9 }$ . This arithmetic and residual processing are included in $C = 2 0$ . All 9,000 trajectories stop with zero certified gap, and saved mean certificates dominate saved mean true gaps (Figure G.1(c)). The simulator’s run-time assertions check domination separately for each trajectory.

At � = 8, reuse lowers mean cost from 444.8 to 441.6 and identifies 3.186 coeficients without extra queries. At $H = 3 2$ and 64, the corresponding reused counts fall to 0.703 and 0.046 and cost savings become negligible. Mean completion epochs are unchanged, and identify-once remains cheaper at every horizon (Figure G.1(d)). Reuse is limited because an uncalibrated block with the wrong private target proposes no change in the coordinates needing correction, so its evaluation reveals nothing about them; deliberate probing remains necessary.

Summary. On a task where every constant is certified, the observed mean gap stays below the bound of Theorem 2, a persistent correction error leaves a floor that longer private search does not lower, the best private horizon grows with the synchronization cost as �(�) predicts, and learned coeficients are retained so that calibration stops after at most 48 queries. Costs are in modeled units on one fixed instance.