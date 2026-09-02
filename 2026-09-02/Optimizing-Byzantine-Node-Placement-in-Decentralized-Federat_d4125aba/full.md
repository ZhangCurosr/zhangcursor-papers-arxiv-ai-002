Strategic Byzantine placement  
![](images/2ca7fd94caca15da1a531d3c94f1e628159c23b2f320281e2aa99aae79148dac.jpg)

# Optimizing Byzantine Node Placement in Decentralized Federated Learning

Edoardo Gabrielli

Dipartimento di Ingegneria Informatica,

Automatica e Gestionale

Sapienza University of Rome

Rome, Italy

Gabriele Tolomei

Dipartimento di Informatica

Sapienza University of Rome

Rome, Italy

Abstract—Security evaluations of decentralized federated learning (DFL) typically focus on how Byzantine participants behave, while largely overlooking which participants are compromised. Yet, because aggregation is distributed over a communication graph, the placement of Byzantine nodes determines how malicious influence propagates through the network. We therefore treat Byzantine placement as an explicit adversarial decision and formulate the attacker’s objective as selecting, under a fixed compromise budget, the set of participants that maximizes its finite-time impact on honest nodes. To approximate this objective without executing the learning process for every candidate placement, we introduce Byzantine Placement Influence (BPI), a set-level measure derived from the actual gossip dynamics that quantifies the cumulative exposure of honest nodes to Byzantine sources over the training horizon. Unlike placement criteria based on node centrality heuristics, BPI directly accounts for weighted multi-hop propagation and interactions among compromised nodes. We develop efficient algorithms for optimizing BPI and evaluate them across six heterogeneous graph families, untargeted model poisoning, and backdoor attacks. BPI-guided placements consistently identify highly damaging configurations across different network structures and remain effective when the linear gossip assumption is relaxed through Byzantine-robust aggregation. Our results show that Byzantine placement is a critical but under-modeled dimension of DFL threat models and robustness evaluations.

## I. INTRODUCTION

Federated learning (FL) enables multiple participants to collaboratively train a machine learning model while keeping their training data local [1]. The participants repeatedly train on their local data and exchange model information according to a communication protocol. In conventional centralized federated learning (CFL), this communication structure can be represented as a star graph and clients communicate with a central server, which is the only node performing aggregation. Decentralized federated learning (DFL) generalizes this setting by distributing aggregation across participants, which exchange and combine models according to a general communication topology [2].

This generalization introduces an additional adversarial dimension because, while in CFL every client reaches all the others in one aggregation step, in the general DFL topology different participants occupy different positions in the communication graph, and malicious information propagates through successive aggregations before reaching distant nodes.

Fig. 1: Byzantine placement as an adversarial dimension in decentralized federated learning. The two executions use the same communication topology, attack, and number of Byzantine participants; only the compromised nodes differ. A poorly positioned Byzantine set exposes a limited portion of the network, whereas a strategic placement allows malicious influence to reach substantially more honest participants within the same training horizon.

Consequently, two Byzantine sets of the same cardinality may have substantially different effects on the honest network; Figure 1 illustrates this mechanism.

This raises a basic question: if an adversary can compromise m participants, which ones should it choose? Existing Byzantine threat models and evaluations provide no common answer. Byzantine participants are variously selected at random, fixed beforehand, positioned ad hoc, or chosen according to topological structural properties [3], [4], [5], [6]. Recent work has started to consider strategic node placement [7], but a general criterion for selecting a Byzantine set according to its actual finite-time influence on decentralized training is still missing.

In this paper, we argue that this is not only an experimentaldesign detail. If the adversary is able to choose which participants to compromise, evaluating an attack under an arbitrary Byzantine placement may substantially underestimate its actual capability. Conversely, a defense may appear effective because the selected Byzantine nodes are poorly positioned to influence the honest network, rather than because the defense successfully limits malicious updates. Thus, the placement of Byzantine participants should itself be considered part of the threat model.

To address these issues, we formulate Byzantine node placement as an optimization problem. Given a communication topology, a Byzantine budget m, and a training horizon T, the adversary seeks the set of compromised participants that maximizes its impact on honest nodes. We therefore introduce Byzantine Placement Influence (BPI), a topology-based proxy that measures the cumulative finite-time exposure of honest participants to a Byzantine set without observing the actual training dynamics. BPI models how malicious influence propagates over multiple communication hops and evaluates the compromised set jointly. Based on this objective, we develop two efficient placement strategies that identify high-influence Byzantine sets without executing the underlying training task.

Our main contributions are:

• We formalize Byzantine node placement as an explicit adversarial dimension in DFL, where the attacker selects a set of m compromised participants to maximize its finite-time impact on honest nodes.

• We introduce Byzantine Placement Influence (BPI), a finite-time, set-level measure of Byzantine exposure derived from the gossip communication dynamics, together with efficient algorithms for identifying high-influence placements.

• We show experimentally that BPI-guided placement consistently strengthens attacks across heterogeneous communication topologies and transfers across different poisoning objectives.

• We demonstrate that Byzantine placement can substantially alter the apparent robustness of Byzantine-robust aggregation, showing that fixed or arbitrary attacker configurations can lead to overly optimistic security evaluations.

## II. RELATED WORK

The communication topology is a central component of decentralized learning, as it determines which nodes exchange models and how information propagates. Nevertheless, the literature on Byzantine decentralized learning typically treats the position of Byzantine nodes as a secondary experimental detail. Recent works [2], [7], [8], however, explicitly raise the question of which nodes an attacker should corrupt in order to maximize the attack’s spread over the network. In the following, we review the topology and placement assumptions adopted by works in Byzantine distributed learning that propose attacks or robust aggregation strategies.

## A. Topology and Byzantine Placement in Decentralized Learning

In fully-connected topologies, Byzantine placement is essentially irrelevant. For instance, LEARN [9] considers a fullyconnected network, where all nodes are structurally equivalent and the attacker does not gain any topological advantage. Similarly, SelfishAttack [10] mainly assumes a fully-connected network, with selfish clients modeled as a fixed subset of participants, namely the last m clients in the indexing. SelfishAttack also considers a partially connected random graph, but the selfish clients are not selected according to their topological position.

Other works evaluate Byzantine robustness on sparse or random decentralized networks, but without optimizing the Byzantine set. UBAR [3] uses random graphs generated from a connection probability and randomly adds Byzantine nodes to the benign network. BALANCE [5] primarily evaluates regular graphs and additionally considers fully-connected, Erdos–˝ Renyi, small-world, ring, and random graphs with differ-´ ent fractions of malicious–benign edges. However, malicious nodes are fixed in the evaluated graph instances, with no topology-aware selection rule.

Some works use topology-specific or ad-hoc placements. ClippedGossip [4] evaluates dumbbell graphs, small-world graphs, torus graphs, and rings. In the small-world and torus experiments, Byzantine workers are not selected among the existing honest nodes; instead, the honest topology is first constructed, and additional Byzantine workers are attached to randomly selected regular workers. In other experiments, Byzantine nodes are added in ad-hoc positions to illustrate specific robustness or breakdown phenomena. Notably, its analysis emphasizes local Byzantine influence through the fraction of incident aggregation weight controlled by Byzantine neighbors, rather than only the global number of Byzantine nodes. Bhattacharya et al. [6] study small-world and scalefree networks, placing Byzantine nodes on nodes incident to rewired Watts–Strogatz edges and on high-degree nodes in scale-free graphs; these strategies remain specific to the considered graph families. $\mathbf { C S _ { + } { - } R G }$ [8] analyzes Byzantinerobust gossip on Two Worlds and Erdos–R˝ enyi graphs; in the´ former, one whole clique is Byzantine, while in the latter each node is adjacent to four malicious nodes.

We summarize these choices in Table I. Overall, there is no common placement convention when benchmarking Byzantine attacks and defenses, making results harder to compare and, more importantly, generally leaving the choice of which participants to compromise outside the attacker’s optimization problem.

Strategic Byzantine placement. Two recent works explicitly study Byzantine placement as an adversarial decision. Syros et al. [11] investigate the effect of compromising structurally influential participants on backdoor propagation in peer-to-peer federated learning. They consider node-wise graph measures, including degree, Effective Network Size, PageRank, and clustering coefficient, and show that the choice of compromised participants can substantially affect backdoor effectiveness. Their placement criteria, however, are static structural heuristics and are studied specifically in the context of backdoor attacks.

Piaseczny et al. [7] formulate Byzantine placement as a coordinated adversarial problem and introduce MaxSpAN-FL, a heuristic strategy that favors well-separated attackers. MaxSpAN-FL constructs BFS-based influence regions around candidate nodes and greedily minimizes their overlap, using shortest-path separation as a proxy for adversarial influence. Their evaluation is limited to a single FGSM-based datapoisoning attack and three graph families (directed geometric, Erdos–R˝ enyi, and preferential-attachment graphs).´

<table><tr><td>Paper</td><td>Topology</td><td>Byzantine Placement</td></tr><tr><td>LEARN [9]</td><td>FC</td><td>Irrelevant</td></tr><tr><td>UBAR [3]</td><td>Random Graph</td><td>Random</td></tr><tr><td>ClippedGossip [4]</td><td>Dumbbell, SW, Torus, Ring</td><td>Random, Ad hoc</td></tr><tr><td>Bhattacharya et al. [6]</td><td>SW, SF</td><td>Random, Ad hoc (SW), Degree (SF)</td></tr><tr><td>BALANCE [5]</td><td>Regular, FC, Erdős-Rényi, SW, Ring</td><td>Ad hoc</td></tr><tr><td> $\mathrm { C S _ { + } { - } R G \ [ 8 ] }$ </td><td>TW, Erdős-Rényi</td><td>Ad hoc</td></tr><tr><td>SelfishAttack [10]</td><td>FC, Random Graph</td><td>Last m nodes</td></tr></table>

TABLE I: Topologies and Byzantine placements considered in the literature on Byzantine distributed learning. Acronyms stand for: Fully-Connected (FC), Scale-Free (SF), Small World (SW), and Two Worlds (TW).

Our work builds on this observation but takes a different approach. Instead of using node centrality or shortest-path separation as a proxy for attack influence, BPI derives the placement objective from the gossip dynamics themselves, providing a common finite-time criterion for evaluating a Byzantine set across different communication topologies and poisoning objectives.

## III. BACKGROUND AND PRELIMINARIES

We consider a network composed of n nodes disposed on an undirected graph $\mathcal { G } = ( \nu , \mathcal { E } )$ where $| \nu | = n$ . Each node optimizes a local objective function $f _ { i } : \mathbb { R } ^ { p }  \mathbb { R }$ and holds a local parameter vector $\pmb { \theta } \in \mathbb { R } ^ { p }$

Participants are interested in solving

$$
\pmb { \theta } _ { \mathcal { H } } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { \theta } } \left[ \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } f _ { i } ( \pmb { \theta } ) \right]\tag{1}
$$

where $\mathcal { H } \subseteq \mathcal { V }$ denotes the set of honest nodes.

At each round $t = 0 , \ldots , T - 1$ , every node i produces its local model update $\pmb { \theta } _ { i } ^ { ( t ) }$ , e.g., by running SGD locally. Node i then receives all parameters from its neighbors $\mathcal { N } ( i )$ and linearly combines them:

$$
\pmb { \theta } _ { i } ^ { + } = \sum _ { j \in \mathcal { N } ( i ) \cup \{ i \} } w _ { i j } \pmb { \theta } _ { j } .\tag{2}
$$

Each $w _ { i j }$ forms a doubly-stochastic gossip matrix $\textrm { \textbf { W } } =$ $[ w _ { i j } ] \in [ \bar { 0 } , 1 ] ^ { n \times n }$ and is governed by the following equation:

$$
w _ { i j } = \left\{ \begin{array} { l l } { \displaystyle { \frac { 1 } { 1 + \operatorname* { m a x } ( d _ { i } , d _ { j } ) } } } & { ( i , j ) \in \mathcal { E } , } \\ { \displaystyle { 1 - \sum _ { \ell \in \mathcal { N } ( i ) } w _ { i \ell } } } & { i = j , } \\ { \displaystyle { 0 } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{3}
$$

where $d _ { i }$ denotes the degree of node i. By construction $\mathbf { W 1 } ~ = ~ \mathbf { 1 }$ and $\mathbf { 1 } ^ { \top } \mathbf { W } \mathbf { \Phi } = \mathbf { \bar { 1 } } ^ { \top }$ , and lim $\begin{array} { r } { \mathbf { \ i } _ { t  \infty } \mathbf { W } ^ { t } \ = \ \frac { 1 } { n } \mathbf { 1 } \mathbf { 1 } ^ { \top } ; } \end{array}$ meaning that messages eventually spread uniformly across the network, assuming $\mathcal { G }$ is connected.

Remark 1 (CFL as a special case). CFL corresponds to the special case where W is uniform over all clients – i.e., physically a star topology, but functionally equivalent to a fully-connected graph – since the server places every client at one-hop distance from every other. Under this W, Byzantine placement is irrelevant, because if the server fails to filter an attack, the poisoned update reaches the entire network in a single round regardless of which clients are compromised. However, this assumption breaks down as soon as W is no longer uniform, i.e., in genuinely decentralized topologies.

## A. Byzantine Threat Model

For the remainder of this work, we consider the scenario in which an external attacker selects a set $B \subset { \mathcal { V } } ;$ , with $| B | = m$ and $\mathcal { H } \cap B = \emptyset .$ , of Byzantine nodes before training starts. Byzantine nodes deviate from the protocol described above by performing model and data poisoning attacks.

The attacker operates under the following assumptions:

(A1) Fixed topology. The topology does not change after training starts and cannot be modified by the attacker.

(A2) Topology knowledge. The adversary knows the com munication topology ${ \mathcal { G } } ,$ and every Byzantine node knows the complete set $\boldsymbol { B }$ of Byzantine participants.

(A3) Non-robust aggregation. Honest nodes do not employ robust aggregation functions, so the mixing matrix W remains linear and depends only on the topology.

Justifying (A1). We consider a static communication graph, consistently with the standard formulation of decentralized gossip learning [8], [4]. This assumption allows us to isolate the effect of the placement of Byzantine nodes. If communication links could change during training or be modified by the adversary, the resulting attack effectiveness would conflate Byzantine placement with topology manipulation, and the propagation dynamics would be governed by a time-varying sequence of mixing matrices rather than by a single matrix W.

Justifying (A2). Consistent with prior work on Byzantine robustness in DFL [8], [9], [4], [12], [13], and to avoid relying on security through obscurity, we assume that the adversary has complete knowledge of the communication topology. In practice, the adversary’s ability to reconstruct the network depends on the actual implementation and characterizing such discovery mechanisms is orthogonal to our objective; therefore we leave it outside the scope of this work.

Justifying (A3). We adopt (A3) to isolate the effect of Byzantine placement from the dynamics introduced by robust aggregators and filtering mechanisms [14], [15], [16], [17], [18], [19]. Under plain averaging, communication is described by the fixed mixing matrix W, allowing us to separate the topological propagation of Byzantine influence from the particular attack realization. With robust aggregation, instead, whether a received model is accepted, rejected, clipped, or reweighted depends on its relation to the models observed during training. The resulting effective mixing operator is therefore nonlinear and time-varying, and depends on the attack, local data, optimization trajectory, and possibly previous rounds.

Moreover, robust aggregation does not provide a universal elimination of Byzantine influence, since adaptive poisoning attacks can circumvent several distance- and statistics-based defenses [20], [?], [21], while history-aware attacks and recent theoretical results expose broader limitations of robust aggregation strategies [22], [23]. Modeling a specific robust aggregator inside BPI would consequently make the placement criterion defense- and attack-specific rather than topologyonly. We therefore derive BPI under linear aggregation and separately evaluate in Section VI-D whether its placement recommendations transfer to Byzantine-robust aggregation.

Perturbation model. To keep the placement problem separate from the choice of poisoning strategy, Byzantine nodes transmit messages of the form $\tilde { \pmb { \theta } } _ { i } ^ { ( t ) } = \pmb { \theta } _ { i } ^ { ( \bar { t } ) } + \mathrm { \alpha ~ } \mathrm { \bar { \alpha } } \mathbf { u }$ , where u is a unit direction $( \left\| \mathbf { u } \right\| = 1 )$ and $\alpha > 0$ is the attack strength. Since we can decompose any poisoning attack into a clean local update plus an adversarial perturbation, and we are concerned only with how that perturbation propagates through the network, we abstract away from how α and u are generated.

Byzantine drift. We model the effect of an attack on node i’s model as $\delta _ { i } ^ { ( t ) } = \tilde { \pmb { \theta } } _ { i } ^ { ( t ) } - \pmb { \theta } _ { i } ^ { ( t ) }$ , the deviation between the model obtained under a placement $\boldsymbol { B }$ and the model obtained in a clean run with identical initialization, data partition, and randomness but $B = \varnothing$ . Stacking all node drifts gives $\Delta ^ { ( t ) } =$ $[ \delta _ { 1 } ^ { ( t ) } , \ldots , \delta _ { n } ^ { ( t ) } ] ^ { \intercal }$ , and the empirical honest-node drift of the network is

$$
D _ { \mathcal { H } } ^ { ( t ) } ( B ) = \frac { 1 } { \lvert \mathcal { H } \rvert } \sum _ { i \in \mathcal { H } } \left. \left. \delta _ { i } ^ { ( t ) } \right. \right. .\tag{4}
$$

Attacker’s objective. Putting (A1)–(A3) together, the attacker chooses the placement $B ,$ with $| B | = m$ fixed, that maximizes the expected honest-node drift after $T$ rounds:

$$
\beta ^ { * } = \underset { | \boldsymbol { \mathcal { B } } | = m } { \arg \operatorname* { m a x } } ~ \mathbb { E } \left[ D _ { \mathcal { H } } ^ { ( T ) } ( \boldsymbol { \mathcal { B } } ; \boldsymbol { \xi } ) \right] ,\tag{5}
$$

where $\xi$ collects the exogenous randomness of training (initialization, mini-batch sampling, data partitioning).

Directly solving Eq. 5 is generally computationally intractable. Evaluating the objective for a single candidate placement B requires estimating the expected honest-node drift $\overset { \cdot } { D } _ { \mathcal { H } } ^ { ( T ) } ( B ; \xi )$ , which entails running training twice – once with $B \stackrel { \cdot } { = } \varnothing$ and once with the candidate Byzantine placement B – and repeating this process across multiple random seeds to average over the randomness ξ. Moreover, since there are $\binom { n } { m }$ possible placements, exhaustive evaluation and optimization quickly become infeasible even for moderately sized graphs.

To address this challenge, Section IV introduces Byzantine Placement Influence (BPI), a tractable proxy objective for identifying influential Byzantine placements.

## IV. BYZANTINE PLACEMENT INFLUENCE

## A. From Empirical Drift to a Tractable Proxy

In practical scenarios, DFL training operates over a finite number of rounds $T ,$ , and the effectiveness of a Byzantine placement depends critically on how quickly poisoned mass propagates through the gossip network within that finite time horizon. This is particularly relevant when attacks add only a small bias over the clean baseline, which may not survive the training dynamics if Byzantine sources are located in lowinfluence regions of the graph. For example, backdoor attacks are sensitive to their positioning in the graph, as their effect survives only for some hops from the source (see Fig. 2). This is the decentralized analogue of the client-selection problem in CFL, where a client-injected backdoor must survive over several communication rounds [24], [25], [26], [27], [28].

![](images/24d26ece286aed494e4f0d91adbd587e750f96600133b535574cb6b87b4226ae.jpg)  
Fig. 2: BadNet backdoor propagation on the ring topology. The Byzantine source node is shown in red, while honest nodes are shown in white. Values around honest nodes report final local attack success rate (ASR) on triggered test samples; label color maps lower ASR to green and higher ASR to red.

These observations motivate us to define a tractable, topology-aware score that captures finite-time Byzantine exposure to honest nodes without requiring any training runs with the aim of maximizing the adversarial objective.

## B. Drift Dynamics and the BPI Score

Under assumptions (A1)–(A3) and the perturbation model of Section III-A, to obtain a topology exposure proxy we consider the linearized gossip component of the drift dynamics:

$$
\hat { \mathbf { \Delta } } \hat { \mathbf { \Delta } } ^ { ( t + 1 ) } = \mathbf { W } \hat { \mathbf { \Delta } } ^ { ( t ) } + \alpha \left( \mathbf { W } \mathbf { 1 } _ { B } \right) \otimes \mathbf { u } ,\tag{6}
$$

where $\mathbf { 1 } _ { B } \in \{ 0 , 1 \} ^ { n }$ is the indicator vector of the Byzantine set. The first term propagates the drift accumulated up to round t through one gossip step; the second term injects a fresh perturbation αu at each Byzantine source, weighted by how much each honest node listens to it. Specifically, $\begin{array} { r } { ( \mathbf { W 1 } _ { B } ) _ { i } \ : = \ : \sum _ { k \in \mathcal { B } } w _ { i k } } \end{array}$ is the total mixing weight that node i assigns to its Byzantine neighbors. Assuming no drift at initialization $( \hat { \pmb { \Delta } } ^ { ( 0 ) } = { \bf 0 }$ , since both the clean and poisoned runs share identical initialization under $B = \emptyset$ and $B \neq \emptyset$ respectively), the recursion unrolls in closed form to:

$$
\hat { \mathbf { \Delta } } \hat { ( \cal T ) } = \alpha \sum _ { t = 1 } ^ { T } ( \mathbf { W } ^ { t } \mathbf { 1 } _ { B } ) \mathbf { u } .\tag{7}
$$

Applying the same honest-node drift measure of Eq. 4 to the linearized dynamics, we define the corresponding reference drift as:

$$
\hat { D } _ { \mathcal { H } } ^ { ( T ) } ( \boldsymbol { \mathcal { B } } ) = \frac { \alpha } { | \mathcal { H } | } \mathbf { 1 } _ { \mathcal { H } } ^ { \top } \left( \sum _ { t = 1 } ^ { T } \mathbf { W } ^ { t } \right) \mathbf { 1 } _ { \boldsymbol { \mathcal { B } } } .\tag{8}
$$

This motivates the following definition.

Definition 1 (Byzantine Placement Influence (BPI)). Given a mixing matrix W, a Byzantine set $B \subseteq \nu ,$ and a training horizon T, the BPI score is computed as:

$$
\Phi _ { T } ( \mathbf { W } , \boldsymbol { B } ) = \frac { 1 } { \left| \mathcal { H } \right| } \mathbf { 1 } _ { \mathcal { H } } ^ { \intercal } \left( \sum _ { t = 1 } ^ { T } \mathbf { W } ^ { t } \right) \mathbf { 1 } _ { \boldsymbol { B } } .\tag{9}
$$

The BPI score has a natural interpretation: the (i, k)-entry of $\mathbf { W } ^ { t }$ measures the cumulative mixing weight that flows from node k to node i over exactly t gossip steps, and the sum $\scriptstyle \sum _ { t = 1 } ^ { T } \mathbf { W } ^ { t }$ aggregates this over the entire training horizon. BPI is therefore the average total exposure of honest nodes to Byzantine sources accumulated over T rounds, weighted by the gossip matrix. Note that BPI depends only on W, $B ,$ and $T ;$ it requires no training runs and no knowledge of the attack direction u or local data distributions.

Comparison with graph-centrality heuristics. Standard heuristics for identifying influential nodes – i.e., degree, betweenness, eigenvector centrality – score nodes independently of one another and of the training horizon. BPI differs in two key respects. First, it is set-valued: BPI evaluates the joint diffusion of the entire Byzantine set, capturing overlap effects that per-node scores miss. For instance, two high-degree nodes adjacent to the same region may expose fewer honest nodes than two lower-degree nodes placed in well-separated parts of the graph. Second, it is finite-time: it accounts for how far Byzantine mass actually travels within T rounds.

## C. Relaxing the Attacker’s Objective with BPI

From Eq. 8, maximizing $D _ { \mathcal { H } } ^ { ( T ) } ( B )$ over placements of size m is approximated by maximizing $\Phi _ { T } ( \mathbf { W } , B )$ . This yields a tractable reformulation of the attacker’s objective (Eq. 5):

$$
B ^ { * } \approx \operatorname * { a r g m a x } _ { B \subseteq \nu , \ | B | = m } \Phi _ { T } ( \mathbf { W } , B ) .\tag{10}
$$

The approximation sign reflects two sources of gap relative to Eq. 5: (i) BPI is a linearization that ignores the training dynamics beyond the gossip step (local SGD noise, data heterogeneity), and (ii) it abstracts away the exogenous randomness ξ by construction. We show in Section VI that larger BPI values correspond to stronger empirical influence and that BPI-guided placements remain effective across topologies and attack strategies, supporting its use as a placement criterion.

```perl
Algorithm 1 GREEDY-BPI SELECTION
Input: Mixing matrix W; node set $\nu ;$ budget $m < | \nu | ;$
horizon T.
Output: $B \subset \mathcal { V } , | B | = m .$
1: procedure GREEDY-BPI(W, V, m, T)
2: $B  \emptyset$
3: for in $1 , \ldots , m$ do
4: $s ^ { * }  \infty ; r ^ { * }  \mathrm { n u 1 1 }$
5: for all $r \in \mathcal { V } \setminus B$ do
6: $s \gets \Phi _ { T } ( \mathbf { W } , B \cup \{ r \} )$
7: if $s > s ^ { * }$ then
8: $s ^ { * } \gets s ; r ^ { * } \gets r$
9: $B  B \cup \{ r ^ { * } \}$
10: return B
```

Nevertheless, exact optimization of Eq. 10 quickly becomes computationally impractical. As with Eq. 5, it requires evaluating all $\binom { n } { m }$ possible Byzantine placements, which is feasible only for small graphs. Section V presents two efficient heuristic algorithms for approximating the optimal solution.

## V. BPI-GUIDED BYZANTINE PLACEMENT

Section IV established that optimizing Byzantine placement for an attacker can be relaxed to maximizing $\Phi _ { T } ( \mathbf { W } , B )$ over all subsets $B \subseteq \nu$ of size m (Eq. 10). Since exact optimization is combinatorially intractable for large graphs, we propose two efficient approximation algorithms: a greedy forward selection procedure and a local search refinement built on top of it.

## A. Greedy BPI Selection

The first algorithm builds B incrementally. Starting from $B = \varnothing .$ , at each step it adds the candidate node $r \in \mathcal { V } \setminus B$ that yields the largest marginal increase in $\Phi _ { T } ( \mathbf { W } , B )$ , repeating until $| B | = m$ . The procedure is described in Algorithm 1.

At each step, the greedy procedure selects the node that maximizes the immediate gain in finite-time exposure. However, this selection is locally optimal with respect to the current partial solution, for two reasons. First, the marginal gain of adding a candidate r depends on the Byzantine nodes already selected. Indeed, adding r both creates new exposure from r to the remaining honest nodes and removes r from H, so any exposure previously directed at r no longer contributes to $\Phi _ { T } ( \mathbf { W } , B )$ . Second, different Byzantine sources may induce partially overlapping diffusion patterns: an early greedy choice may block a more effective global configuration by steering the set toward a locally concentrated region of the graph. Consequently, greedy selection is not guaranteed to reach the globally optimal $B ^ { * }$ of size m.

Computational complexity. Each evaluation of $\Phi _ { T } ( \mathbf { W } , B )$ costs $O ( T \cdot | \mathcal { E } | )$ if $\mathbf { W } ^ { t } \mathbf { 1 } _ { B }$ is computed iteratively (one sparse matrix-vector product per round) rather than materializing $\mathbf { W } ^ { t }$ explicitly. The greedy loop performs m outer iterations, each evaluating at most n candidates, giving an overall cost of $\mathcal { O } ( m ^ { . }$ $n \cdot T \cdot \left| { \mathcal { E } } \right| )$

Algorithm 2 SWAP-BPI SELECTION   
Input: Mixing matrix W; node set V; budget $m < | \nu | ;$   
horizon T.   
Output: $B \subset \mathcal { V } , | B | = m .$   
1: procedure $\mathbf { S W A P - B P I } ( \mathbf { W } , \mathcal { V } , m , T )$   
2: $\boldsymbol { B } \gets \mathrm { G R E E D Y - B P I } ( \mathbf { W } , \boldsymbol { \mathcal { V } } , m , T )$   
3: improved ← true   
4: while improved do   
5: improved ← false   
6: $b ^ { * } \gets \mathrm { n u l l } ; r ^ { * } \gets \mathrm { n u l l } ; s ^ { * } \gets \Phi _ { T } ( \mathbf { W } , \boldsymbol { B } )$   
7: for all $b \in B$ do   
8: for all $r \in \mathcal { V } \setminus B$ do   
9: $\tilde { \mathcal { B } } \gets ( \mathcal { B } \setminus \{ b \} ) \cup \{ r \}$   
10: $s \gets \Phi _ { T } ( \mathbf { W } , \tilde { \mathcal { B } } )$   
11: if $s > s ^ { * }$ then   
12: $s ^ { * }  s ; b ^ { * }  b ; r ^ { * }  r$   
13: improved ← true   
14: if improved then   
15: $\bar { B ^ { } }  ( B \setminus \{ b ^ { * } \} ) \cup \{ r ^ { * } \}$   
16: return B

## B. Local Search Refinement via 1-Swap

To address the limitations of greedy selection, we initialize with the greedy solution and refine it through an exhaustive local 1-swap search. At each iteration, the procedure considers all single node exchanges of the form $\tilde { \mathcal { B } } = ( \mathcal { B } \setminus \{ b \} ) \cup \{ r \}$ with $b \in B$ and $r \in \mathcal { V } \backslash B$ , and accepts the swap that yields the largest improvement in $\Phi _ { T } ( \mathbf { W } , B )$ . The search repeats until no improving exchange exists. The resulting set is locally optimal with respect to all 1-swap moves. For every $b \in B$ and every $r \in \mathcal { V } \setminus B ,$

$$
\begin{array} { r } { \Phi _ { T } ( \mathbf { W } , \boldsymbol { B } ) \geq \Phi _ { T } ( \mathbf { W } , ( \boldsymbol { B } \setminus \{ b \} ) \cup \{ r \} ) . } \end{array}\tag{11}
$$

Since each accepted swap strictly increases $\Phi _ { T } ( \mathbf { W } , B )$ and the number of feasible placements of size m is finite, the procedure terminates in a finite number of iterations. The full algorithm is given in Algorithm 2.

Computational complexity. Each swap iteration evaluates $\Phi _ { T } ( \mathbf { W } , B )$ for all $m ( n - m )$ candidate exchanges at cost $O ( T \cdot | \mathcal { E } | )$ each, giving $\mathcal { O } ( m ( n - m ) \cdot T \cdot | \mathcal { E } | )$ per pass. The number of passes until local optimality is bounded by the number of feasible placements.

## VI. EXPERIMENTS

In this section, we address the following research questions (RQs):

RQ1: Does finite-time Byzantine exposure reflect the empirical   
influence of a placement during decentralized training?   
RQ2: Does BPI provide a consistent placement for Byzantine   
nodes across heterogeneous communication topologies?

RQ3: Do BPI-guided placements remain effective when the attacker adopts an objective different from the additive untargeted perturbation used to derive BPI?

RQ4: Does the advantage of BPI-guided placements persist when honest nodes employ a nonlinear Byzantine-robust aggregation?

Dataset and model. Unless otherwise stated, we use MNIST [29] and distribute the training samples i.i.d. across participants. Each client trains a two-layer multilayer perceptron consisting of a 128-unit ReLU hidden layer followed by a 10-unit output layer. Local training minimizes the crossentropy loss using Adam with learning rate 0.01, batch size 64, and one local epoch per communication round.

Setup. Our main experiments use $n \ = \ 5 0$ participants, of which $m = 5$ (10%) are Byzantine, and run for $T = 3 0$ communication rounds. Honest nodes aggregate neighbor models using Metropolis weights. We evaluate five experiment seeds, {33, 99, 5, 42, 123}.

Byzantine nodes do not aggregate models received from other Byzantine nodes. This removes Byzantine-to-Byzantine self-reinforcement from the experimental setting and isolates the effect of how malicious sources expose honest participants.

We select six topologies described in Table II. We selected the topology parameters so that all graph families yield broadly comparable sparse networks, with average degrees approximately between four and five. This prevents differences in attack effectiveness from being trivially explained by large differences in network density; at the same time, the resulting graphs retain substantially different structures. DC-SBM, Core-Periphery, Random-Geometric, and Scale-Free are regenerated using the experiment seed; disconnected realizations are resampled until a connected graph is obtained. Dragonfly and Ring-of-Cliques are deterministic and therefore identical across seeds.

Byzantine placement. We compare Greedy-BPI and Swap-BPI with five baselines: uniform random placement, Degree, Betweenness, Eigenvector centrality, and MaxSpAN-FL [7]. Centrality-based strategies select the m nodes with the largest respective centrality score.

For Random, one Byzantine set of m distinct nodes is sampled uniformly without replacement for each topology and experiment seed. Hence, each table entry for Random contains five independently generated placements, one for each seed.

Poisoning attacks. For the untargeted evaluation, Byzantine nodes inject an aligned additive perturbation. At the beginning of each run, we sample a direction u with entries $u _ { k } \sim$ $\mathcal { N } ( 0 , 1 )$ using the experiment seed and normalize it such that $\| \mathbf { u } \| _ { 2 } = 1$ . The same direction is shared by all Byzantine nodes.

When Byzantine node i attacks for the first time, we record its model norm $\lVert \pmb { \theta } _ { i , 1 } \rVert _ { 2 }$ and freeze this value for the remainder of the run. At every subsequent communication round, the node publishes

$$
\widetilde { \pmb { \theta } } _ { i , t } = \pmb { \theta } _ { i , t } + \alpha \Vert \pmb { \theta } _ { i , 1 } \Vert _ { 2 } \mathbf { u } ,
$$

where $\alpha = 6 4$ in the main experiments. Thus, all malicious perturbations are directionally aligned while their magnitude is normalized to the initial model scale of the corresponding attacker. We refer to this attack as FixedBias.

TABLE II: Communication topologies used in the main experiments. All graphs contain $n = 5 0$ nodes.
<table><tr><td>Topology</td><td>Configuration</td></tr><tr><td>Ring-of-Cliques</td><td>10 cliques of size  $^ { 5 , }$  each forming a  $K _ { 5 } ,$  connected in a ring by one inter-clique edge per consecutive pair (110 edges) [30].</td></tr><tr><td>Dragonfly</td><td>10 groups of 5 nodes, each forming a  $K _ { 5 } .$  The group-level graph is the 4-regular circulant  $\bar { C } _ { 1 0 } ( 1 , \bar { 2 } )$  , with one physical inter-group link for each group-level edge and endpoints</td></tr><tr><td>Scale-Free</td><td>assigned round-robin (120 edges) [31]. Barabási-Albert preferential attachment in which each new node attaches to two existing nodes [32].</td></tr><tr><td>DC-SBM</td><td>Five communities of 10 nodes, with  $p _ { \mathrm { i n } } = 0 . 4$  and  $p _ { \mathrm { o u t } } = 0 . 0 2 5$  . Node propensities are sampled from a log-normal distribution with σ = 0.6 and normalized to unit mean within each community; edge probabilities are  $p _ { i j } = \mathrm { m i n } ( 1 , \theta _ { i } \theta _ { j } p _ { \mathrm { i n / o u t } } )$  [33].</td></tr><tr><td>Core-Periphery</td><td>Two-block SBM with 10 core and 40 peripheral nodes,  $p _ { c c } \dot { ~ } = ~ 0 . 6 , ~ p _ { c p } ~ = ~ 0 . 1 5$  and  $p _ { p p } = 0 . 0 3 \ [ 3 4 ] .$ </td></tr><tr><td>Random-Geometric Nodes sampled uniformly in</td><td> $[ 0 , 1 ] ^ { 2 }$  and connected when their Euclidean distance is at most  $r = 0 . 1 7 \ [ { \bar { 3 } } 5 ] .$ </td></tr></table>

For the targeted evaluation, we employ a classical backdoor attack called BadNets [36]. Each Byzantine participant poisons a fixed 20% subset of its local training data. Poisoned samples are stamped with a $3 \times 3$ white square in the bottom-right corner and relabeled to target class $y ^ { * } = 0$ . Malicious clients otherwise follow the same local optimization configuration as honest clients, i.e., one local epoch of Adam training with learning rate 0.01 and batch size 64 per communication round. Evaluation metrics. For RQ1, we use the empirical honestnode drift $D _ { \mathcal { H } } ^ { ( t ) }$ from Eq. 4 and examine whether placements with progressively larger $\Phi _ { T } ( \mathbf { W } , B )$ induce correspondingly larger drift trajectories. For untargeted attacks, we measure the honest-node accuracy drop at round t as

$$
\Delta A _ { t } = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i \in \mathcal { H } } ( A _ { i , t } - \tilde { A } _ { i , t } ) ,
$$

where $A _ { i , t }$ is node $i \gamma _ { \mathrm { s } }$ clean accuracy and $\tilde { A } _ { i , t }$ its accuracy under attack. The main RQ2 comparison reports the finalround value $\Delta A _ { T }$ with $T = 3 0 ;$ the long-horizon FixedBias experiment in RQ4 reports $\Delta A _ { t }$ over training.

For backdoor attacks, we measure the honest-node attack success rate (ASR) on triggered test samples:

$$
A S R _ { \mathcal { H } } ^ { ( t ) } = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \frac { \sum _ { ( x , y ) \in D _ { \mathrm { t e s t } } : y \neq y ^ { * } } \mathbb { 1 } \left[ f _ { i } ^ { ( t ) } ( \tau ( x ) ) = y ^ { * } \right] } { | \{ ( x , y ) \in D _ { \mathrm { t e s t } } : y \neq y ^ { * } \} | } .\tag{12}
$$

Here, $f _ { i } ^ { ( t ) } ( \cdot )$ is the classifier held by node i at round $t , D _ { \mathrm { t e s t } }$ is the clean test set, τ (x) adds the backdoor trigger, and $y ^ { * }$ is the attacker’s target label.

## A. Empirical Validation of Finite-Time Exposure (RQ1)

We first investigate whether the ordering induced by BPI is reflected in the actual influence of Byzantine perturbations during training. Figure 3 compares four placements spanning progressively larger values of $\Phi _ { T } ( \mathbf { W } , B )$ on a 50-node Ringof-Cliques topology under FixedBias.

The empirical trajectories exhibit the same ordering as BPI. The minimum-exposure placement induces only limited honest-model drift, while Eigenvector, Random, and Swap-BPI produce progressively faster and larger drift accumulation as their finite-time exposure increases. Swap-BPI, which has the largest $\Phi _ { T } ( \mathbf { W } , B )$ , also produces the largest empirical drift throughout training.

![](images/2a83cd50ec8bc914d77acf4f8d12d26df87c6c10f65d6e4c11b5bf7614e2ef25.jpg)  
Fig. 3: Evolution of the empirical drift $D _ { \mathcal { H } } ^ { ( t ) }$ across communication rounds for placements spanning increasing values of $\Phi _ { T } ( \mathbf { W } , B )$ . Greater exposure results in faster and larger drift of honest models.

## B. Cross-Topology Consistency of BPI (RQ2)

1) BPI Across Placement Strategies: We next study whether a placement strategy that performs well on one graph family remains effective on structurally different networks.

Figure 4 first isolates the topology-level objective by comparing Φ<sub>T</sub> obtained by each deterministic placement strategy with the distribution induced by 5000 uniformly sampled Byzantine sets. The relative quality of conventional structural heuristics changes markedly across graph families. A criterion that produces a high-exposure placement on one topology may fall considerably closer to the random-placement distribution on another.

BPI exhibits a qualitatively different behavior. Greedy-BPI and, in particular, Swap-BPI consistently locate placements at the extreme high-exposure end of the distribution across all six graph families.

![](images/342d596821e6226a68e4e59e746862187c5edd0a7b0633e91bf27f296b37e18d.jpg)  
Fig. 4: Histograms show $\Phi _ { T } ( W , B )$ for 5000 uniformly sampled placements of five Byzantine nodes on each topology; vertical lines indicate deterministic placement strategies. Lines farther to the right correspond to greater finite-time exposure. We use $n = 5 0$ and $T = 3 0$

2) Accuracy Drop on Untargeted Attacks: Table III shows that no conventional structural heuristic provides a uniformly strong placement across graph families.

Degree and Betweenness are particularly effective on topologies containing pronounced degree heterogeneity or structurally privileged regions. On Scale-Free, they obtain 19.6 and 19.4 percentage points of accuracy degradation, respectively, compared with only 14.1 for Random. A similar pattern appears on DC-SBM and Core-Periphery, where both heuristics approach the BPI-guided placements. This is expected: in these graphs, hubs, high-connectivity nodes, and inter-community positions are informative proxies for malicious influence.

However, the same heuristics transfer poorly to other structures. On Random-Geometric, Degree falls to 15.3 and Betweenness to 14.0, compared with 20.8 for Greedy-BPI. They also provide little advantage over Random on Ring-of-Cliques and Dragonfly.

MaxSpAN-FL exhibits an almost complementary behavior. It performs strongly on Ring-of-Cliques (20.9), Dragonfly (20.7), and Random-Geometric (18.4), but is substantially weaker on Scale-Free (17.1), DC-SBM (16.5), and Core-Periphery (17.1). Eigenvector centrality is less reliable overall and is the weakest strategy on Ring-of-Cliques, Dragonfly, and Random-Geometric.

In contrast, BPI does not rely on any one of these structural signatures. Either Greedy-BPI or Swap-BPI achieves the largest mean degradation on every evaluated topology. This is the main cross-topology result: different structural heuristics become appropriate on different graph families, whereas the propagation-based BPI objective remains consistently effective.

Interestingly, the difference between Greedy-BPI and Swap-BPI is small. Greedy-BPI already attains the strongest observed mean on Dragonfly and Random-Geometric and ties Swap-BPI on DC-SBM, while Swap-BPI is marginally stronger on Ring-of-Cliques, Scale-Free, and Core-Periphery. Their topology-averaged degradations are 20.8 and 20.9 percentage points, respectively.

Recall that Swap-BPI is guaranteed to improve or preserve the BPI objective, not the downstream accuracy degradation. Small inversions between Greedy-BPI and Swap-BPI are therefore expected because BPI is a proxy for the complete learning dynamics. The small empirical gap also indicates that most of the benefit comes from optimizing the BPI objective itself.

## C. Transferability Across Attack Objectives (RQ3)

The cross-topology instability of the structural baselines becomes even more pronounced under BadNets (Table IV).

Degree is highly effective on Scale-Free (49.6%), DC-SBM (56.0%), and Core-Periphery (50.4%), but its effectiveness drops to 29.3% on Random-Geometric. Betweenness follows a similar pattern: it is competitive on Core-Periphery (54.2%) and DC-SBM (53.4%), but reaches only 31.6% on Dragonfly and 19.6% on Random-Geometric.

MaxSpAN-FL again exhibits a different topology preference. It is the strongest strategy on Dragonfly (46.0%) and performs well on Ring-of-Cliques and Random-Geometric, but drops to 34.3% on Scale-Free, 29.7% on DC-SBM, and 31.9% on Core-Periphery. Eigenvector centrality never provides the strongest placement and is particularly weak on Random-Geometric, where it reaches only 14.7% ASR.

Random placement follows the same topology-dependent pattern observed for FixedBias. It remains relatively competitive on Ring-of-Cliques and Dragonfly, but performs very poorly on Scale-Free (19.7%), DC-SBM (27.2%), and especially Core-Periphery (17.0%).

BPI is substantially more stable. A BPI-guided strategy achieves the largest mean ASR on five of the six graph families. On the only exception, Dragonfly, MaxSpAN-FL reaches 46.0% while Swap-BPI reaches 45.3%, a difference of only 0.7 percentage points. Conversely, whenever one of the structural baselines fails to match the graph structure, its degradation can be severe.

This result is particularly significant because BadNets is not described by the aligned additive perturbation model used to derive BPI. The persistence of the same cross-topology behavior therefore indicates that BPI is capturing a property of the communication process itself rather than a peculiarity of FixedBias.

Figure 5 provides a node-level view of this propagation process. Although only Byzantine nodes directly inject poisoned training samples, the learned backdoor is not confined to their immediate neighborhood. Honest participants that aggregate a malicious model partially inherit the backdoor and subsequently propagate it through their own outgoing models. As a result, non-neighboring honest nodes can acquire substantial ASR after several communication rounds.

This behavior illustrates why Byzantine placement cannot be characterized solely through one-hop quantities such as

TABLE III: Final honest-node accuracy drop $\Delta A _ { 3 0 }$ under FixedBias for different Byzantine placement strategies. Values are the mean ± 95% Student-t confidence interval over five seeds, in percentage points. Higher values indicate a stronger attack. The highest mean in each row is bold; the lowest is underlined.
<table><tr><td>Topology</td><td>Random</td><td>Degree</td><td>Betweenness</td><td>Eigenvector</td><td>MaxSpAN-FL | Greedy-BPI</td><td></td><td>Swap-BPI</td></tr><tr><td>Ring-of-Cliques</td><td> $1 8 . 2 \pm 1 . 0$ </td><td> $1 8 . 4 \pm 2 . 4$ </td><td> $1 8 . 4 \pm 2 . 4$ </td><td> $1 4 . 3 \pm 4 . 5$ </td><td> $2 0 . 9 \pm 1 . 1$ </td><td> $2 1 . 0 \pm 1 . 0$ </td><td> ${ \bf 2 1 . 2 \pm 1 . 1 }$ </td></tr><tr><td>Dragonfly</td><td> $1 9 . 8 \pm 1 . 3$ </td><td> $1 9 . 9 \pm 1 . 6$ </td><td> $1 8 . 8 \pm { 1 . 2 }$ </td><td> $\overline { { 1 8 . 5 \pm 2 . 3 } }$ </td><td> $2 0 . 7 \pm 1 . 2$ </td><td> ${ \bf 2 0 . 9 \pm 1 . 1 }$ </td><td> $2 0 . 8 \pm 1 . 0$ </td></tr><tr><td>Scale-Free</td><td> $\underline { { 1 4 . 1 \pm 3 . 0 } }$ </td><td> $1 9 . 6 \pm 1 . 0$ </td><td> $1 9 . 4 \pm 0 . 9$ </td><td> $\overline { { 1 8 . 4 \pm 1 . 1 } }$ </td><td> $1 7 . 1 \pm 1 . 3$ </td><td> $2 0 . 2 \pm 1 . 0$ </td><td> ${ \bf 2 0 . 5 \pm 0 . 7 }$ </td></tr><tr><td>Degree-Corrected-SBM</td><td> $\overline { { 1 6 . 1 \pm 0 . 5 } }$ </td><td> $2 0 . 6 \pm 1 . 8$ </td><td> $2 0 . 4 \pm 1 . 5$ </td><td> $1 9 . 7 \pm 2 . 4$ </td><td> $1 6 . 5 \pm 3 . 8$ </td><td> ${ \bf 2 1 . 3 \pm 1 . 5 }$ </td><td> ${ \bf 2 1 . 3 \pm 1 . 6 }$ </td></tr><tr><td>Core-Periphery</td><td> $\underline { { 1 3 . 1 \pm 3 . 7 } }$ </td><td> $1 9 . 3 \pm 0 . 8$ </td><td> $1 9 . 8 \pm 0 . 9$ </td><td> $1 8 . 8 \pm 0 . 9$ </td><td> $1 7 . 1 \pm 2 . 4$ </td><td> $2 0 . 6 \pm 0 . 9$ </td><td> ${ \bf 2 0 . 8 \pm 0 . 9 }$ </td></tr><tr><td>Random-Geometric</td><td> $\overline { { 1 5 . 2 \pm 1 . 5 } }$ </td><td> $1 5 . 3 \pm 1 . 9$ </td><td> $1 4 . 0 \pm 2 . 7$ </td><td> $\underline { { 9 . 1 \pm 4 . 5 } }$ </td><td> $1 8 . 4 \pm 2 . 4$ </td><td> ${ \bf 2 0 . 8 \pm 0 . 6 }$ </td><td> $2 0 . 6 \pm 0 . 8$ </td></tr></table>

TABLE IV: $A S R _ { \mathcal { H } }$ under BadNets for different Byzantine placement strategies. Entries report the $\mathrm { m e a n } \pm 9 5 \%$ Student-t confidence interval across five seeds. Higher values indicate a stronger attack. The highest mean in each row is bold; the lowest is underlined.
<table><tr><td>Topology</td><td>Random</td><td>Degree</td><td>Betweenness</td><td>Eigenvector</td><td>MaxSpAN-FL |Greedy-BPI</td><td></td><td>Swap-BPI</td></tr><tr><td>Ring-of-Cliques</td><td> $3 8 . 6 \pm 6 . 8$ </td><td> $3 7 . 0 \pm 6 . 6$ </td><td> $3 7 . 0 \pm 6 . 6$ </td><td> $2 6 . 9 \pm 1 2 . 6$ </td><td> $4 2 . 7 \pm 0 . 9$ </td><td> ${ \bf 4 4 . 6 \pm 1 . 6 }$ </td><td> $4 4 . 0 \pm 2 . 5$ </td></tr><tr><td>Dragonfly</td><td> $4 1 . 4 \pm { 1 . 8 }$ </td><td> $3 9 . 1 \pm 7 . 4$ </td><td> $3 1 . 6 \pm 1 . 8$ </td><td> $\overline { { 3 2 . 6 \pm 9 . 5 } }$ </td><td> ${ \bf 4 6 . 0 \pm 2 . 0 }$ </td><td> $4 4 . 2 \pm 2 . 2$ </td><td> $4 5 . 3 \pm 4 . 4$ </td></tr><tr><td>Scale-Free</td><td> $\underline { { 1 9 . 7 \pm 8 . 1 } }$ </td><td> $4 9 . 6 \pm 2 . 1$ </td><td> $\overline { { 4 7 . 7 \pm 2 . 1 } }$ </td><td> $3 9 . 2 \pm 5 . 4$ </td><td> $3 4 . 3 \pm 7 . 4$ </td><td> $5 6 . 1 \pm 9 . 5$ </td><td> ${ \bf 5 9 . 5 \pm 1 0 . 4 }$ </td></tr><tr><td>Degree-Corrected-SBM</td><td> $2 7 . 2 \pm 1 0 . 2$ </td><td> $5 6 . 0 \pm 1 3 . 4$ </td><td> $5 3 . 4 \pm 1 0 . 1$ </td><td> $4 7 . 1 \pm 1 5 . 7$ </td><td> $2 9 . 7 \pm 2 2 . 8$ </td><td> $6 3 . 1 \pm 8 . 1$ </td><td> ${ \bf 6 4 . 7 \pm 8 . 8 }$ </td></tr><tr><td>Core-Periphery</td><td> $1 7 . 0 \pm 9 . 8$ </td><td> $5 0 . 4 \pm 9 . 0$ </td><td> $5 4 . 2 \pm 7 . 5$ </td><td> $4 6 . 6 \pm 8 . 2$ </td><td> $3 1 . 9 \pm 1 1 . 9$ </td><td> $6 0 . 3 \pm 4 . 8$ </td><td> ${ \bf 6 0 . 6 \pm 5 . 7 }$ </td></tr><tr><td>Random-Geometric</td><td> $3 2 . 3 \pm 7 . 8$ </td><td> $2 9 . 3 \pm 6 . 8$ </td><td> $1 9 . 6 \pm 8 . 4$ </td><td> $\underline { { 1 4 . 7 \pm 9 . 2 } }$ </td><td> $3 5 . 6 \pm 8 . 6$ </td><td> ${ \bf 4 9 . 5 \pm 5 . 5 }$ </td><td> $4 8 . 9 \pm 6 . 9$ </td></tr></table>

The mean of the accepted neighboring models is then combined with the local intermediate model with weights 0.75 and 0.25, respectively.

Figure 6 reveals a substantial placement-induced vulnerability. Under BadNets, the placement used in the original BALANCE evaluation yields a mean honest-node ASR of 39.2% on Erdos–R˝ enyi and´ 16.0% on the Ring. Changing only the identities of the Byzantine participants to the Swap-BPI placement raises these values to 99.8% and 82.4%, respectively.

degree. The effect of compromising a node depends not only on how many honest neighbors it directly reaches, but also on how its influence propagates through sequences of honest relays over the finite training horizon. These multi-hop paths are precisely the contribution captured by the powers $\mathbf { W } ^ { t }$ in BPI.

## D. Beyond Linear Gossip (RQ4)

Our previous experiments use Metropolis aggregation, matching the linear communication model from which BPI is derived. We finally test whether BPI-guided placement remains effective when the actual aggregation rule is nonlinear.

$$
\left\| \theta _ { i } ^ { t + 1 / 2 } - \theta _ { j } ^ { t + 1 / 2 } \right\| _ { 2 } \leq 1 . 5 e ^ { - 0 . 5 t / T } \left\| \theta _ { i } ^ { t + 1 / 2 } \right\| _ { 2 } .
$$

We consider the Ring and Erdos–R˝ enyi graph configurations´ and Byzantine placements used in the BALANCE evaluation and compare those placements with Swap-BPI. These experiments contain n = 20 nodes and m = 4 Byzantine participants (20%).

We use BALANCE [5] with $\alpha _ { \mathrm { B A L } } ~ = ~ 0 . 2 5 , ~ \kappa ~ = ~ 0 . 5 ,$ $\lambda ( t ) = t / T$ , and $\gamma = 1 . 5 .$ . A neighbor j is accepted by node i whenever

Thus, the observed robustness of BALANCE is strongly conditional on where those Byzantine nodes are located. The original placement does not expose this worst-case behavior, whereas topology-aware placement reveals a substantially more damaging attack configuration.

Surprisingly, robust aggregation does not necessarily attenuate the topology-induced propagation of the backdoor. Under BadNets, BALANCE can yield higher $A S R _ { \mathcal { H } }$ than plain averaging. This indicates that filtering neighboring models may alter their relative influence in a way that favors malicious updates that remain within the acceptance threshold.

We additionally evaluate FixedBias with scale factor $\alpha =$ 2.5 for $T ~ = ~ 3 0 0$ rounds. Figure 7 compares the original BALANCE placement with Swap-BPI. While the original placement produces limited degradation, the BPI-guided placement causes a progressively larger accuracy gap over training, showing that the placement effect also persists for an untargeted attack under nonlinear aggregation.

## VII. CONCLUSION

This work studies Byzantine node placement as an explicit adversarial dimension in DFL. We formalize the attacker’s placement objective and introduce Byzantine Placement Influence (BPI), a finite-time, set-level measure of how strongly a Byzantine set can expose honest participants. BPI separates the structural contribution of the communication topology from the details of a particular poisoning strategy, and the resulting placement algorithms provide a practical way to identify high-influence Byzantine sets without executing training for every candidate placement. Our experiments show that placement can substantially change attack effectiveness across heterogeneous graph families, poisoning objectives, and even under nonlinear Byzantine-robust aggregation. These findings indicate that Byzantine placement should be specified and optimized as part of DFL security evaluation.

![](images/41f33194f71026630b4e1997ccecddd1371650d6d3526904bdf5fbfaeab2db85.jpg)  
Fig. 5: Final node-level ASR across network topologies and Byzantine node placement strategies. Node halos encode the final ASR at round $T = 3 0 ;$ nodes with ASR below 1% are left uncoloured. Each panel reports the mean ASR over honest nodes, $A S R _ { \mathcal { H } } .$ , and the corresponding $\Phi _ { T } ( \mathbf { W } , B )$ score.

Our analysis focuses on static, known communication topologies and derives BPI from nominal linear gossip dynamics. Extending the placement problem to time-varying or partially known topologies, and defense-specific propagation dynamics remains an important direction for future work. Additional deployment constraints can also be incorporated by restricting the feasible placement set; for example, the proposed optimization algorithms can be extended to enforce a bounded local Byzantine population around honest nodes.

## ACKNOWLEDGMENT

We thank Anthony Di Pietro for contributions to the experimental codebase, including implementations of attack, method, and aggregation components, debugging assistance, and preliminary experiments that informed our investigation of the echo effect.

## REFERENCES

[1] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, and others, “Advances and open problems in federated learning,” Foundations and Trends® in Machine Learning, vol. 14, no. 1–2, pp. 1–210, 2021.

[2] E. Gabrielli, A. D. Pietro, D. Fenoglio, G. Pica, and G. Tolomei, “A survey on decentralized federated learning,” 2026. [Online]. Available: https://arxiv.org/abs/2308.04604

![](images/f0ec603f724d3315b5a16a7b3c19cf135e45c4f0fd6d166e86d69e4761e06b7b.jpg)  
Fig. 6: Both the Erdos–R˝ enyi and Ring graphs, together´ with the Byzantine positioning in the first column, reproduce the topologies and attacker placement used in the original BALANCE paper. Columns compare this positioning with Swap-BPI. Colored halos report final local ASR, while red crosses identify Byzantine nodes. Each panel reports the mean ASR over honest nodes and the corresponding $\Phi _ { T } ( \mathbf { W } , B )$ score.

![](images/3478538ef4c0ceaf808b74748480e2878d2b7453634a4ba10e407e9b8f8fa318.jpg)  
Fig. 7: Mean $\Delta A _ { t }$ among honest nodes under BALANCE aggregation for the Byzantine placement used in the original BALANCE paper and Swap-BPI on the Erdos–R ˝ enyi and Ring´ topologies.

[3] S. Guo, T. Zhang, H. Yu, X. Xie, L. Ma, T. Xiang, and Y. Liu, “Byzantine-Resilient Decentralized Stochastic Gradient Descent,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 6, pp. 4096–4106, Jun. 2022. [Online]. Available: https: //ieeexplore.ieee.org/document/9555632/

[4] L. He, S. P. Karimireddy, and M. Jaggi, “Byzantine-Robust Decentralized Learning via ClippedGossip,” Apr. 2023, arXiv:2202.01545 [cs]. [Online]. Available: http://arxiv.org/abs/2202.01545

[5] M. Fang, Z. Zhang, Hairi, P. Khanduri, J. Liu, S. Lu, Y. Liu, and N. Gong, “Byzantine-Robust Decentralized Federated Learning,” in Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. Salt Lake City UT USA: ACM, Dec. 2024, pp. 2874–2888. [Online]. Available: https://dl.acm.org/doi/10.1145/3658644.3670307

[6] S. Bhattacharya, D. Helo, and J. Siegel, “Impact of Network Topology

on Byzantine Resilience in Decentralized Federated Learning,” Jul. 2024, arXiv:2407.05141 [cs]. [Online]. Available: http://arxiv.org/abs/ 2407.05141

[7] A. Piaseczny, E. Ruzomberka, R. Parasnis, and C. G. Brinton, “Adversarial Node Placement in Decentralized Federated Learning: Maximum Spanning-Centrality Strategy and Performance Analysis,” IEEE Internet of Things Journal, vol. 12, no. 1, pp. 45–60, Jan. 2025. [Online]. Available: https://ieeexplore.ieee.org/document/10681487/

[8] R. Gaucher, A. Dieuleveut, and H. Hendrikx, “Unified Breakdown Analysis for Byzantine Robust Gossip.”

[9] E. M. El-Mhamdi, S. Farhadkhani, R. Guerraoui, A. Guirguis, L.-N. Hoang, and S. Rouault, “Collaborative Learning in the Jungle (Decentralized, Byzantine, Heterogeneous, Asynchronous and Nonconvex Learning),” in Advances in Neural Information Processing Systems, vol. 34. Curran Associates, Inc., 2021, pp. 25 044– 25 057. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2021/hash/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Abstract.html

[10] Y. Jia, M. Fang, and N. Z. Gong, “Competitive Advantage Attacks to Decentralized Federated Learning,” 2023, version Number: 2. [Online]. Available: https://arxiv.org/abs/2310.13862

[11] G. Syros, G. Yar, S. Boboila, C. Nita-Rotaru, and A. Oprea, “Backdoor attacks in peer-to-peer federated learning,” ACM Trans. Priv. Secur., vol. 28, no. 1, Dec. 2024. [Online]. Available: https://doi.org/10.1145/3691633

[12] M. Raynal, D. Pasquini, and C. Troncoso, “Can Decentralized Learning be more robust than Federated Learning?”

[13] D. Pasquini, M. Raynal, and C. Troncoso, “On the (In)security of Peerto-Peer Decentralized Machine Learning,” Nov. 2023, arXiv:2205.08443 [cs]. [Online]. Available: http://arxiv.org/abs/2205.08443

[14] P. Blanchard, E. M. E. Mhamdi, R. Guerraoui, and J. Stainer, “Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.”

[15] E. M. E. Mhamdi, R. Guerraoui, and S. Rouault, “The Hidden Vulnerability of Distributed Learning in Byzantium,” Jul. 2018, arXiv:1802.07927 [stat]. [Online]. Available: http://arxiv.org/abs/1802. 07927

[16] D. Yin, Y. Chen, K. Ramchandran, and P. Bartlett, “Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates,” Feb. 2021, arXiv:1803.01498 [cs]. [Online]. Available: http://arxiv.org/abs/1803. 01498

[17] X. Cao, M. Fang, J. Liu, and N. Z. Gong, “FLTrust: Byzantinerobust Federated Learning via Trust Bootstrapping,” Apr. 2022, arXiv:2012.13995 [cs]. [Online]. Available: http://arxiv.org/abs/2012. 13995

[18] Z. Zhang, X. Cao, J. Jia, and N. Z. Gong, “FLDetector: Defending federated learning against model poisoning attacks via detecting malicious clients,” in Proc. of KDD ’22. ACM, 2022, pp. 2545–2555, number of pages: 11.

[19] E. Gabrielli, D. Belli, Z. Matrullo, V. Miori, and G. Tolomei, “Securing Federated Learning Against Extreme Model Poisoning Attacks via Multidimensional Time Series Anomaly Detection on Local Updates,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 9610–9624, 2025. [Online]. Available: https://ieeexplore.ieee.org/document/11164345/

[20] M. Fang, X. Cao, J. Jia, and N. Z. Gong, “Local Model Poisoning Attacks to Byzantine-Robust Federated Learning.”

[21] V. Shejwalkar and A. Houmansadr, “Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning,” in Proceedings 2021 Network and Distributed System Security Symposium. Virtual: Internet Society, 2021. [Online]. Available: https://www.ndss-symposium.org/wp-content/ uploads/ndss2021 6C-3 24498 paper.pdf

[22] S. P. Karimireddy, L. He, and M. Jaggi, “Learning from History for Byzantine Robust Optimization.”

[23] M. Raynal and C. Troncoso, “On the Conflict Between Robustness and Learning in Collaborative Machine Learning,” in 2025 IEEE Symposium on Security and Privacy (SP), May 2025, pp. 2171–2189, iSSN: 2375-1207. [Online]. Available: https://ieeexplore.ieee.org/document/ 11023472/

[24] H. Faraoun, R. Bellafqira, G. Coatrieux, and K. Kallas, “FLARE: Federated Learning Attack via Robust Expectation-Based Backdooring Using GAN,” in 2025 International Conference on Emerging Technologies and Computing (IC ETC), Jun. 2025, pp. 1–6. [Online]. Available: https://ieeexplore.ieee.org/document/11141092/

[25] Z. Zhang, A. Panda, L. Song, Y. Yang, M. Mahoney, P. Mittal, R. Kannan, and J. Gonzalez, “Neurotoxin: Durable Backdoors in Federated Learning,” in Proceedings of the 39th International Conference on Machine Learning. PMLR, Jun. 2022, pp. 26 429–26 446. [Online]. Available: https://proceedings.mlr.press/v162/zhang22w.html

[26] H. Wang, K. Sreenivasan, S. Rajput, H. Vishwakarma, S. Agarwal, J.-y. Sohn, K. Lee, and D. Papailiopoulos, “Attack of the Tails: Yes, You Really Can Backdoor Federated Learning,” Jul. 2020, arXiv:2007.05084 [cs, stat]. [Online]. Available: http://arxiv.org/abs/2007.05084

[27] Z. Sun, P. Kairouz, A. T. Suresh, and H. B. McMahan, “Can You Really Backdoor Federated Learning?” Dec. 2019, arXiv:1911.07963 [cs]. [Online]. Available: http://arxiv.org/abs/1911.07963

[28] C. Xie, K. Huang, P.-Y. Chen, and B. Li, “DBA: Distributed backdoor attacks against federated learning,” in International conference on learning representations, 2020. [Online]. Available: https://openreview. net/forum?id=rkgyS0VFvr

[29] Y. LeCun, C. Cortes, and C. J. Burges, “Mnist handwritten digit database,” ATT Labs Research, vol. 2, 2010. [Online]. Available: http://yann.lecun.com/exdb/mnist/

[30] S. Li and T. Tian, “Resistance Between Two Nodes of a Ring Clique Network,” Circuits, Systems, and Signal Processing, vol. 41, no. 3, pp. 1287–1298, Mar. 2022. [Online]. Available: https://doi.org/10.1007/s00034-021-01859-7

[31] J. Kim, W. J. Dally, S. Scott, and D. Abts, “Technology-driven, highlyscalable dragonfly topology,” in Proceedings of the 35th International Symposium on Computer Architecture, 2008, pp. 77–88.

[32] A.-L. Barabasi and R. Albert, “Emergence of scaling in random networks,” Science, vol. 286, no. 5439, pp. 509– 512, Oct. 1999, arXiv:cond-mat/9910332. [Online]. Available: http://arxiv.org/abs/cond-mat/9910332

[33] B. Karrer and M. E. J. Newman, “Stochastic blockmodels and community structure in networks,” Physical Review E, vol. 83, no. 1, p. 016107, 2011.

[34] S. P. Borgatti and M. G. Everett, “Models of core/periphery structures,” Social Networks, vol. 21, no. 4, pp. 375–395, 2000.

[35] M. Penrose, Random Geometric Graphs. Oxford University Press, 2003.

[36] T. Gu, B. Dolan-Gavitt, and S. Garg, “BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.”