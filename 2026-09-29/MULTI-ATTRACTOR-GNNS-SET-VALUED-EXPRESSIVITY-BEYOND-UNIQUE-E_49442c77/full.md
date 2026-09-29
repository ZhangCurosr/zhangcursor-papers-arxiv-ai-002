# MULTI-ATTRACTOR GNNS: SET-VALUED EXPRESSIVITY BEYOND UNIQUE EQUILIBRIA

JIALIN LIU

Abstract. Recurrent and equilibrium graph neural networks (GNNs) are often designed to have a unique fixed point or trained against a single target per graph. Yet many combinatorial and scientific problems are inherently set-valued: several valid solutions coexist for the same graph, and nothing in the problem singles one out. Training against a single designated target can then make the model fit an arbitrary selection rule rather than the full solution set. For example, when a task is invariant to node relabeling, a symmetric graph has a symmetric solution set, yet may admit no symmetric solution; isolating a single target then imposes an arbitrary symmetry-breaking choice.

In this paper, we show that multiple equilibria of recurrent GNNs are a resource rather than a defect: one weight-tied message-passing GNN can represent such set-valued equivariant maps through its attractor landscape, where diferent initializations approach diferent valid solutions. Under the stated regularity assumptions, we establish this expressive power in two steps. First, we prove the existence of globally Lipschitz, permutation-equivariant multi-attractor dynamics that converge almost surely to valid solutions while reaching every solution branch with positive probability. Second, we establish their approximate realization by recurrent message passing with continuous component maps, with arbitrarily small update and limiting errors and arbitrarily high probability. This goes beyond standard universality arguments, since message passing cannot by itself distinguish symmetric nodes: we show that the evolving state keeps nodes distinguishable at every finite step, without auxiliary node identifiers. In practice, such dynamics can be learned without solution labels from problemspecific energies. On three scientific tasks (ground states of Ising models, structural module detection in protein graphs, and steady states of chemical reaction networks), the learned updates produce multiple high-quality predictions and high numerical convergence rates, with better average solution quality than the tested unique-equilibrium, single-target, and feedforward baselines, while remaining competitive with far larger difusion-based solvers.

## 1. Introduction

Recurrent graph neural networks (GNNs) repeatedly apply a shared message-passing update, allowing computation to deepen without introducing new parameters at every step [35]. This iterative structure provides a natural mechanism for capturing long-range dependencies, while equilibrium-based variants support implicit diferentiation without storing the full sequence of intermediate states [24, 6, 67].

For an n-node graph $G \ : = \ : ( A , X )$ , A encodes connectivity and edge attributes, and $X _ { i }$ contains node i’s features. A prediction $y \in \mathbb { R } ^ { n \times q }$ assigns each node a q-dimensional output $y _ { i }$ A feedforward GNN computes $y = \mathrm { G N N } _ { \theta } ( G )$ directly; a recurrent predictor updates a state $y _ { t } \in \mathbb { R } ^ { n \times q }$ :

$$
y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } ) , \qquad t = 0 , 1 , 2 , \ldots .
$$

From a deterministic or random $y _ { 0 } .$ , each step receives $( X _ { i } , ( y _ { t } ) _ { i } )$ at node i and exchanges messages along edges to produce $( y _ { t + 1 } ) _ { i }$ . The graph and parameters θ remain fixed. The usual goal is a limit $y _ { \infty }$ approximating a prescribed target f(G).

![](images/a265a26c45f3c16d2a1407e9ee8984fbfbe4422cf317a05f2d3e51211f133549.jpg)  
Figure 1. Diferent initializations can lead to diferent equilibria under the same recurrent GNN update, with the graph and model parameters held fixed.

An equilibrium, or fixed point, is a state unchanged by the update. Implicit GNNs often seek well-posedness: existence and uniqueness of this state for each input [24, 39, 47]. A standard suficient condition is global contraction: each update shrinks the distance between any two states by a common factor $\kappa \ : < 1$ . Iteration then converges to the same fixed point from every initialization [53]. Strong monotonicity of an associated operator ofers another route to uniqueness [63, 6]; uniqueness alone need not ensure convergence of direct iteration. Singletarget training instead encourages one designated output without guaranteeing uniqueness. Contraction rules out initialization-dependent solutions regardless of the training objective.

Many graph or combinatorial tasks admit multiple valid solutions for the same input. Selecting one target can discard meaningful alternatives or conflict with input symmetries. For example:

• Two coloring. Consider proper two-coloring of the four-node cycle $1 - 2 - 3 - 4 - 1$ , with identical node features: each node receives a color in {0, 1}, and adjacent nodes must have diferent colors. There are exactly two valid assignments, (0, 1, 0, 1) and (1, 0, 1, 0), which are exchanged by a one-step rotation that leaves the input graph unchanged. A vanilla GNN receiving only the graph must assign identical outputs to all nodes and therefore cannot produce either coloring. Symmetry breaking can distinguish these nodes and enable a valid coloring, but single-target training still arbitrarily favors one equally valid solution.

• Nonlinear equilibria. Chemical reaction networks are represented as bipartite graphs connecting species to participating reactions, with reaction parameters as attributes; the task is to predict stationary species concentrations. Even one dynamic species can admit multiple solutions for the same graph and parameters: a Schl¨ogl model has steady-state equation $0 = 8 - 1 4 c + 7 c ^ { 2 } - c ^ { 3 }$ and three positive solutions $c \in \{ 1 , 2 , 4 \}$ . All three satisfy the same physical equation, which selects no preferred target. Discovering multiple steady states can reveal coexisting stable states and help explain switching between them.

These examples motivate recovering multiple valid solutions per graph without a preferred target. Recurrent GNNs provide a natural mechanism: initializations y<sub>0</sub> and $y _ { 0 } ^ { \prime }$ can converge to diferent limits $y _ { \infty }$ and $y _ { \infty } ^ { \prime }$ under the same learned update (Figure 1). Only the initial node states change between runs; G and θ remain fixed. We seek convergence to a valid solution within each run and access to the graph’s diferent solutions across runs.

Building on the set-valued dynamical viewpoint of Jore and Liu [30], we study permutationequivariant representation and message-passing realization with recurrent GNNs. Our contributions address three questions.

• Can one shared update represent many solutions? Under the stated assumptions, we prove that a single recurrent message-passing GNN can converge arbitrarily close to the solution set, with arbitrarily high probability over graphs and initializations (Theorems 1 and 2).

• Is a unique equilibrium merely less diverse? Under suitable conditions, every continuous, equivariant, contractive update misses all valid solutions with positive probability, even on almost-surely asymmetric graphs. This applies to MP-GNNs of arbitrary finite depth and width, regardless of training (Theorem 3).

• Can such dynamics be learned without solution labels, and do they pay of ? On Ising ground states, protein module detection, and chemical steady states, energy or residual training from random starts yields multiple high-quality predictions and high numerical convergence rates. Average solution quality exceeds the tested single-target, unique equilibrium, and feedforward baselines and remains competitive with much larger difusionbased solvers.

## 2. Main results

Existing expressivity theory studies single-valued invariant or equivariant maps [66, 33, 1]. Can one GNN instead represent solution sets through initialization-dependent limits? We establish existence, illustrate the mechanism, and identify an obstruction under contraction.

2.1. Can Recurrent GNNs Represent Set-Valued Equivariant Maps? To make the question precise, let $Y _ { n } = \mathbb { R } ^ { n \times q }$ be the space of node-level outputs. Let G be a family of n-node graphs that is closed under node permutations. A permutation π acts on a graph by relabeling its nodes, and on $Y _ { n }$ by permuting rows. A finite solution map assigns to each $G \in { \mathcal { G } }$ a set

$$
S ( G ) = \left\{ f _ { 1 } ( G ) , \dots , f _ { M } ( G ) \right\} \subseteq Y _ { n } ,
$$

where the branches $f _ { a }$ may coincide on some G. We require the map to be set-valued equivariant,

$$
S ( \pi G ) = \pi S ( G ) .
$$

Individual branches need not be equivariant. In the two-coloring example (Section 1), a onenode rotation preserves the graph but exchanges (0, 1, 0, 1) and $( 1 , 0 , 1 , 0 )$ . The solution set is unchanged; selecting either coloring alone violates equivariance.

A recurrent GNN represents solutions through the limits of its update $y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } )$ with $y _ { 0 } \sim \mu _ { n }$ , where $\mu _ { n }$ is an initialization distribution on $Y _ { n }$ with a density and full support, such as i.i.d. standard Gaussian entries. We collect the limits reached with positive probability in the attractor set

$$
\mathcal { A } _ { \theta } ( G ) : = \left\{ s \in Y _ { n } : \mathbb { P } _ { y _ { 0 } \sim \mu _ { n } } \left[ \operatorname* { l i m } _ { t \to \infty } \mathrm { G N N } _ { \theta } ^ { t } ( G , y _ { 0 } ) = s \right] > 0 \right\} ,
$$

where ${ \mathrm { G N N } } _ { \theta } ^ { t } ( G , \cdot )$ denotes t repeated applications of the block. Ideally, one shared update satisfies, for every G: (i) almost every trajectory $\{ y _ { t } \}$ <sub>t</sub> converges to a limit in $S ( G )$ ; and (ii) every $s \in S ( G )$ is reached with positive probability. Together, these ensure

$$
{ \mathcal { A } } _ { \theta } ( G ) = { \mathcal { S } } ( G ) ,
$$

with no probability mass on nonconvergent trajectories or invalid limits.

Theorem 1 constructs a globally Lipschitz, equivariant operator satisfying (i)–(ii) on suitably regular graphs. Theorem 2 realizes its dynamics approximately by recurrent message passing:

limits approach the solution set with arbitrarily high joint probability, and each branch with positive joint probability over graphs and initializations.

Theorem 1 (Existence of equivariant multi-attractor dynamics). Let G be a permutationinvariant family of n-node graphs, and let $S ( G ) = \{ f _ { 1 } ( G ) , \dots , f _ { M } ( G ) \} \subseteq Y _ { n } = \mathbb { R } ^ { n \times q }$ be setvalued equivariant, with branch labels as in Definition 1. Let $\mathcal { G } _ { \star }$ consist of the graphs where the branches are locally Lipschitz and which branches coincide does not change under suficiently small graph perturbations.

Then there exists a single operator $T : ( G , y ) \mapsto T _ { G } ( y )$ that is jointly globally Lipschitz and permutation equivariant, $T _ { \pi G } ( \pi y ) = \pi T _ { G } ( y )$ . For every $G \in \mathcal G _ { \star }$ , the iteration $y _ { t + 1 } = T _ { G } ( y _ { t } )$ initialized from any distribution $\mu _ { n }$ with a density and full support on $Y _ { n . }$ , satisfies

$$
y _ { t } \to y _ { \infty } \in S ( G ) \qquad { } \quad a l m o s t ~ s u r e l y ,
$$

$$
\mathbb { P } _ { y _ { 0 } \sim \mu _ { n } } ( y _ { \infty } = s ) > 0 \quad f o r \ e v e r y \ s \in \mathcal { S } ( G ) .
$$

Thus Lipschitz regularity, which bounds sensitivity without requiring contraction, permits multiple solutions and convergent trajectories (Appendix $\mathrm { A } )$ . We next realize this behavior by message passing.

Message-passing GNNs. For a graph $G = ( A , X )$ and state $y \in Y _ { n }$ , define a GNN block ${ \mathrm { G N N } } _ { \theta } ( G , y )$ with finitely many layers and finite hidden dimensions:

$$
h _ { i } ^ { ( 0 ) } = \operatorname { e n c } ( X _ { i } , y _ { i } ) ,\tag{1}
$$

$$
h _ { i } ^ { ( \ell + 1 ) } = U _ { \ell } \left( h _ { i } ^ { ( \ell ) } , \sum _ { j = 1 } ^ { n } M _ { \ell } \big ( h _ { i } ^ { ( \ell ) } , h _ { j } ^ { ( \ell ) } , A _ { j i } \big ) \right) ,
$$

$$
[ \mathrm { G N N } _ { \theta } ( G , y ) ] _ { i } = O \left( h _ { i } ^ { ( L ) } , \sum _ { j = 1 } ^ { n } R _ { L } ( h _ { j } ^ { ( L ) } ) \right) ,
$$

where $\ell = 0 , \dots , L - 1$ and $i = 1 , \ldots , n$ . The encoder and component maps $M _ { \ell } , U _ { \ell } , R _ { L } , O$ are arbitrary continuous maps between finite-dimensional Euclidean spaces, shared across nodes. The edge features $A _ { j i }$ encode edge attributes and edge presence; messages can be chosen to vanish on absent edges. Our theory uses arbitrary continuous component maps, rather than a fixed MLP parameterization; its guarantee is representational, not a guarantee for training.

Theorem 2 (Approximate realization by a recurrent MP-GNN). Let S and $\mathcal { G } _ { \star }$ be as in Theorem $^ { 1 , }$ and assume $q \geq 2$ . Let G follow a tight Borel probability on $\mathcal { G } _ { \star } ,$ , and independently draw $y _ { 0 } \sim \mu _ { n }$ , where $\mu _ { n }$ has a density and full support on $Y _ { n }$ . Let $T _ { G }$ be the ideal operator in Theorem 1.

For any $\varepsilon , \rho > 0$ and $\delta \in ( 0 , 1 )$ , there exists a single GNN of the form (1) such that, with probability at least $1 - \delta$ over (G, y<sub>0</sub>), the iteration $y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } )$ converges to a limit $y _ { \infty }$ satisfying

$$
\operatorname* { s u p } _ { t \geq 0 } \bigl \| \mathrm { G N N } _ { \theta } ( G , y _ { t } ) - T _ { G } ( y _ { t } ) \bigr \| < \rho \ a n d \ \mathrm { d i s t } ( y _ { \infty } , \mathcal { S } ( G ) ) < \varepsilon .
$$

Moreover, every solution branch is reached within ε with positive joint probability:

$$
\mathbb { P } ( y _ { t } \to y _ { \infty } , \ \lVert y _ { \infty } - f _ { a } ( G ) \rVert < \varepsilon ) > 0 , \qquad a = 1 , \ldots , M .
$$

The branch guarantee is joint over graphs and initializations, not per graph. Vanilla GNNs lack universality for continuous equivariant operators [66]; random initialization can provide unique node identifiers to overcome this [1]. But subsequent states depend on the graph and each other, even from i.i.d. $y _ { 0 }$ . Do they still distinguish nodes?

Our proof establishes that independence is unnecessary: the ideal trajectories $z _ { t } : = T _ { G } ^ { t } ( y _ { 0 } )$ retain distinct node states at every finite time,

$$
\mathbb { P } ( z _ { t , i } \neq z _ { t , j } \mathrm { ~ f o r ~ a l l ~ } i \neq j \mathrm { ~ a n d ~ a l l ~ } t \in \mathbb { N } _ { 0 } ) = 1 .
$$

These evolving identifiers enable recurrent message passing to approximate the ideal $\mathrm { d y }$ namics (Appendix B).

2.2. Toy Example: Single-Source Graph Difusion. On a problem with exactly three solutions, we test whether one trained recurrent GNN approaches diferent solutions from different random initial states and recovers all three across runs. We compare against contraction and single-target supervision.

Problem. The input is a three-node undirected graph $G = ( A , 0 )$ with independent edge weights $w _ { 1 2 } , w _ { 2 3 } , w _ { 3 1 } \sim \mathrm { U n i f } [ 0 . 5 , 1 . 5 ]$ and no distinguishing static node features. Here A is the symmetric weighted adjacency matrix with zero diagonal. Define $L _ { G } = \mathrm { d i a g } ( A { \bf 1 } ) - A$ and $H _ { G } = I _ { 3 } + L _ { G }$ . We seek a source indicator $b \in \{ 0 , 1 \} ^ { 3 }$ selecting exactly one node, $\mathbf { 1 } ^ { \top } b = 1$ , and its difusion field $u \in \mathbb { R } ^ { 3 }$ satisfying $H _ { G } u = b$ . Since $H _ { G }$ is positive definite, each source choice determines one difusion field, giving exactly three solutions:

$$
\begin{array} { r } { \mathcal { S } ( G ) = \left\{ [ H _ { G } ^ { - 1 } e _ { k } ~ e _ { k } ] : k = 1 , 2 , 3 \right\} \subset \mathbb { R } ^ { 3 \times 2 } , } \end{array}\tag{2}
$$

where $e _ { k }$ is the kth standard basis vector. The source is an output to discover, not an input supplied to the model. Figure 2 illustrates one graph and its three solutions.

Model and training. The recurrent state $y _ { t } ~ = ~ [ u _ { t } ~ b _ { t } ] ~ \in ~ \mathbb { R } ^ { 3 \times 2 }$ stores two values per node. From six independent standard Gaussian entries, we iterate $y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } )$ using a shared block with two message-passing layers, following (1). For each fixed graph and model, we seek convergent trajectories reaching each solution in $S ( G )$ with positive probability across initializations. We train using the energy function

$$
E _ { G } ( u , b ) = \frac 1 3 \| H _ { G } u - b \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { 3 } b _ { i } ( 1 - b _ { i } ) + ( \mathbf { 1 } ^ { \top } b - 1 ) ^ { 2 } , \qquad b \in [ 0 , 1 ] ^ { 3 } .\tag{3}
$$

The three terms are all nonnegative and $E _ { G } ( u , b ) = 0$ holds exactly on $S ( G )$ . Training uses eight random starts per graph and minimizes the terminal energy $\mathbb { E } _ { G , y _ { 0 } } [ E _ { G } ( u _ { T } , b _ { T } ) ]$ with $T = 1 0$ This objective permits diferent starts to approach diferent solutions without prescribing branch labels.

Baselines. We compare two controls with the same architecture, changing either the update constraint or training target. The contractive GNN uses the same energy objective but projects its parameters throughout training to enforce $\mathrm { L i p } _ { y } ( \mathrm { G N N } _ { \theta } ( G , \cdot ) ) \leq 0 . 9$ , ensuring a unique fixed point for each graph. The single-target-supervised GNN has no contraction constraint. For each training graph, we uniformly select one of its three exact solutions and retain it across all initializations and training visits. Squared-error training asks all starts to predict that target without guaranteeing a unique equilibrium. Details are in Appendix D.

Results. Figure 3 tracks the source coordinates $b _ { t }$ of 128 trajectories on one held-out graph, with identical initial states across all three models. The multi-attractor model forms three clusters near the valid source indicators $e _ { 1 } , e _ { 2 } , e _ { 3 }$ , whereas the supervised and contractive controls each concentrate near a single point, visibly separated from all three. The difusion coordinates $u _ { t }$ exhibit the same qualitative pattern (Appendix D, Figure 4). Thus energy training can produce multiple attractors near diferent solutions, consistent with Theorems 1 and 2, while contraction or single-target supervision can concentrate trajectories far from every valid solution.

![](images/e2fc70c1e3ebc9499c2e14a36354572377db3b8ef800ae5ad0c7e6ee8fd8a67d.jpg)  
Figure 2. One graph, three valid solutions to single-source difusion. Left: the input graph, with edge weights labeled. Right: each choice of source gives a solution $f _ { k } ( G ) = [ H _ { G } ^ { - 1 } e _ { k } e _ { k } ]$ . Node labels show $( u _ { i } , b _ { i } )$ , with difusion values rounded to four decimal places; orange marks the selected source $( b _ { i } = 1 )$ . All three outputs are valid for the same input, which specifies no source.

![](images/2d48225426dade5e434520dc7ba5aa091779b59ed8a7c06aef09d84961af35b3.jpg)  
Figure 3. Diferent initial states can approach diferent solutions under the same learned update. Each blue point represents one of 128 trajectories, plotted in source coordinates $( ( b _ { t } ) _ { 1 } , ( b _ { t } ) _ { 2 } , ( b _ { t } ) _ { 3 } )$ ; magenta circles mark the three valid source indicators $e _ { 1 } , e _ { 2 } , e _ { 3 }$ Columns show recurrent steps $t = 0 , 2 , 4 , 6 , 8 .$ . The multi-attractor model (top) forms three clusters near these targets, whereas the single-target-supervised (middle) and contractive (bottom) controls each concentrate away from them. All rows use the same held-out graph (test graph 0), training seed 1, and initial states.

2.3. Can a Unique Equilibrium Miss Every Valid Solution? Do these failures reflect training or a structural limitation? A unique equilibrium cannot represent multiple solutions through diferent initializations, but can it always recover one valid solution? The following theorem answers no for continuous, permutation-equivariant, globally contractive updates: even with independently sampled edge weights, their equilibrium remains separated from every valid solution with positive probability, regardless of training or initialization.

Theorem 3. Let $\mathcal { G } _ { \triangle }$ be the family of weighted triangles in (2), with edge weights in $[ w _ { - } , w _ { + } ]$ where $0 < w _ { - } < w _ { + } < \infty$ . Let ν be the graph law induced by sampling the three edge weights independently and uniformly from $[ w _ { - } , w _ { + } ]$ . Consider any jointly continuous, permutationequivariant operator $F : \mathcal { G } _ { \triangle } \times Y _ { 3 }  Y _ { 3 }$ , where $Y _ { 3 } = \mathbb { R } ^ { 3 \times 2 }$ , satisfying

$$
\| F _ { G } ( y ) - F _ { G } ( z ) \| _ { * } \le \kappa \| y - z \| _ { * } \qquad ( G \in { \mathcal { G } } _ { \triangle } , \ y , z \in Y _ { 3 } )\tag{4}
$$

for some fixed norm $\| \cdot \| ,$ <sub>∗</sub> and $\kappa \in [ 0 , 1 )$ . This includes any MP-GNN of (1) with continuous component maps satisfying (4), regardless of its finite depth and width.

For every $0 < \varepsilon < \sqrt { 2 / 3 }$ , there exists $\delta _ { F , \varepsilon } > 0$ such that, under any joint law $o f \left( G , Y _ { 0 } \right)$ with graph marginal $\nu ,$ the recurrence $Y _ { t + 1 } = F _ { G } ( Y _ { t } )$ converges to a limit $Y _ { \infty }$ and

$$
\mathbb { P } \big ( \mathrm { d i s t } \big ( Y _ { \infty } , \boldsymbol { \mathcal { S } } ( G ) \big ) > \varepsilon \big ) \geq \delta _ { F , \varepsilon } ,
$$

where distance is measured in the Euclidean norm. In particular, no such operator converges to a valid solution almost surely over independently weighted graphs.

Random initialization distinguishes nodes even on symmetric graphs; Figure 3 shows its influence in the early iterations. Contraction eventually erases these diferences: all trajectories approach one equilibrium. On the symmetric triangle, equivariance forces this equilibrium to preserve the graph’s symmetries, preventing single-source selection. For independently weighted graphs, the equilibrium need not be symmetric, but its continuous dependence on the graph extends the failure to nearby graphs with positive probability. The failure probability depends on the operator; no uniform lower bound across models is claimed. See Appendix C for the proof and extension.

## 3. Scientific applications

We test whether energy or residual training learns multiple high-quality predictions and numerically convergent dynamics. These empirical tests, including scalar states outside Theorem 2’s $q \geq 2$ setting, do not certify asymptotic convergence or complete coverage. Baselines use the reported budgets; inference computation is not matched.

## 3.1. Ising Models.

Problem. The Ising model is a classical model of interacting magnetic spins and is a standard discrete optimization problem. In the zero-field setting, each node of the weighted graph $G =$ $( V , E , J )$ carries a spin $s _ { i } \in \{ - 1 , + 1 \}$ , with energy and ground states

$$
{ \cal S } ( G ) = \underset { s \in \{ - 1 , + 1 \} ^ { | V | } } { \arg \operatorname* { m i n } } \quad E _ { G } ( s ) , \quad \quad E _ { G } ( s ) = - \sum _ { \{ i , j \} \in E } J _ { i j } s _ { i } s _ { j } .\tag{5}
$$

We use open triangular lattices with antiferromagnetic couplings $( J _ { i , j } < 0 )$ , which favor opposite neighboring spins. These preferences conflict on triangles, creating competing low-energy configurations; global spin reversal also preserves energy. This multiplicity naturally tests set-valued prediction: we seek diverse low-energy configurations per graph rather than one arbitrarily selected target.

Models and baselines. Our multi-attractor model (Ours in the tables) uses $y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } )$ from $y _ { 0 } \sim \mathcal { N } ( 0 , I )$ , refining continuous spin estimates and applying the sign function to $y _ { T }$ to predict discrete spins ˆs. Training minimizes $\mathbb { E } [ E _ { G } ( y _ { T } ) ]$ with $T = 1 0$ shared updates, encouraging low energy without prescribing a solution for each initialization. Testing uses 200 updates.

Table 1. Ising results on 500 test graphs with 20 candidates per graph (mean ± standard deviation over three training seeds). ¯e: mean energy per node (lower is better); Hit10: percentage of candidates within 10% of the certified minimum energy; $C _ { 2 0 0 } \mathrm { : }$ percentage of trajectories converged by step 200; $D _ { 2 0 } \colon$ mean number of distinct optimal configurations found per graph, counting global spin flips separately; $G _ { 2 0 } \colon$ percentage of graphs with at least two such configurations, up to numerical tolerance. Details in Appendix E.
<table><tr><td>Method</td><td>ē↓</td><td> $\mathrm { H i t 1 0 ~ \uparrow }$ </td><td> $C _ { 2 0 0 } \uparrow$ </td><td> $D _ { 2 0 } \uparrow$ </td><td> $G _ { 2 0 } \uparrow$ </td></tr><tr><td>Ours</td><td> $- 1 . 0 7 9 4 \pm 0 . 0 0 2 8$ </td><td> $8 3 . 9 8 \pm 0 . 9 3$ </td><td> $9 9 . 9 6 \pm 0 . 0 2$ </td><td> $2 . 1 6 \pm 0 . 2 3$ </td><td> $6 8 . 7 3 \pm 5 . 1 9$ </td></tr><tr><td>Single-target Supervision</td><td> $0 . 2 0 7 7 \pm 0 . 5 8 7 8$ </td><td> $9 . 4 5 \pm 4 . 3 7$ </td><td> $1 0 . 1 1 \pm 3 . 3 8$ </td><td> $0 . 1 0 \pm 0 . 0 5$ </td><td> $0 . 8 0 \pm 0 . 9 9$ </td></tr><tr><td>IGNN adapter (H16)</td><td> $- 0 . 4 5 8 1 \pm 0 . 2 5 3 8$ </td><td> $5 . 2 0 \pm 3 . 3 1$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN adapter (H32)</td><td> $- 0 . 7 5 1 6 \pm 0 . 0 1 8 5$ </td><td> $8 . 0 0 \pm 0 . 7 1$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 6 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN adapter (H64)</td><td> $- 0 . 7 9 5 4 \pm 0 . 0 1 4 3$ </td><td> $1 3 . 2 0 \pm 2 . 1 4$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 8 \pm 0 . 0 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Feedforward (×1)</td><td> $- 0 . 9 7 2 0 \pm 0 . 0 0 3 5$ </td><td> $3 6 . 1 3 \pm 0 . 7 5$ </td><td></td><td> $1 . 3 1 \pm 0 . 0 3$ </td><td> $3 6 . 6 7 \pm 1 . 2 3$ </td></tr><tr><td>Feedforward (×3)</td><td> $- 1 . 0 5 1 5 \pm 0 . 0 0 1 6$ </td><td> $6 9 . 0 7 \pm 0 . 6 6$ </td><td></td><td> $0 . 9 5 \pm 0 . 9 5$ </td><td> $2 2 . 4 0 \pm 3 1 . 6 8$ </td></tr><tr><td>Feedforward (×10)</td><td> $- 1 . 0 5 3 8 \pm 0 . 0 0 1 2$ </td><td> $7 0 . 9 3 \pm 1 . 5 2$ </td><td></td><td> $0 . 2 8 \pm 0 . 0 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Mean-field GNN</td><td> $- 0 . 9 7 3 6 \pm 0 . 0 0 2 6$ </td><td> $3 6 . 7 1 \pm 0 . 7 8$ </td><td></td><td> $1 . 3 4 \pm 0 . 0 6$ </td><td> $3 7 . 8 0 \pm 1 . 3 1$ </td></tr><tr><td>Annealed Bernoulli GNN</td><td> $- 0 . 9 7 6 6 \pm 0 . 0 0 1 1$ </td><td> $3 7 . 7 1 \pm 0 . 8 6$ </td><td></td><td> $1 . 3 7 \pm 0 . 0 3$ </td><td> $3 8 . 2 7 \pm 0 . 6 6$ </td></tr><tr><td>DIFUSCO (matched)</td><td> $- 0 . 9 0 3 3 \pm 0 . 0 0 2 3$ </td><td> $3 7 . 6 4 \pm 0 . 2 3$ </td><td></td><td> $1 . 5 6 \pm 0 . 0 3$ </td><td> $4 5 . 8 7 \pm 0 . 0 9$ </td></tr><tr><td>DIFUSCO (large)</td><td> $- 1 . 0 3 6 2 \pm 0 . 0 0 2 4$ </td><td> $6 9 . 8 2 \pm 1 . 6 4$ </td><td></td><td> $2 . 6 0 \pm 0 . 1 0$ </td><td> $6 8 . 0 0 \pm 2 . 9 5$ </td></tr><tr><td>VAG (matched)</td><td> $- 0 . 8 8 9 3 \pm 0 . 0 0 7 5$ </td><td> $1 7 . 1 8 \pm 2 . 0 2$ </td><td></td><td> $0 . 6 5 \pm 0 . 1 1$ </td><td> $1 7 . 4 0 \pm 4 . 9 8$ </td></tr><tr><td>VAG (large)</td><td> $- 0 . 8 8 1 4 \pm 0 . 0 5 4 7$ </td><td> $1 7 . 3 4 \pm 5 . 6 9$ </td><td></td><td> $0 . 5 6 \pm 0 . 1 7$ </td><td> $1 4 . 0 0 \pm 5 . 7 4$ </td></tr></table>

We compare four baseline groups. (A) Single-target supervision minimizes MSE to one Gurobi solution per graph using the same recurrent architecture, testing the efect of a designated target on accuracy and diversity. (B) Unique-equilibrium models are energy-trained IGNNs [24] with contraction constraints and increasing hidden widths, comparing quality and multiplicity under enforced uniqueness at several capacities. (C) Feedforward models stack 1, 3, or 10 independently parameterized copies of our block with the same energy objective, compar ing untied depth with shared recurrence. (D) Distributional models learn sampling distributions or generative processes, whereas ours learns a shared update with initialization-dependent solutions. We compare quality and diversity with Mean-field GNN, Annealed Bernoulli GNN, DIFUSCO, and VAG (VAG-CO) [32, 58, 59, 51] as alternative approaches to set-valued prediction. Appendix E provides all details.

Quality and multiple solutions. Our model achieves the lowest mean energy and highest Hit10 (Table 1): 83.98% of candidates are within 10% of optimum. It finds 2.16 distinct optima per graph on average and at least two on 68.73% of graphs. Numerical convergence reaches 99.96% by step 200 and 100% by step 500 for the same trajectories (Appendix E). Singletarget supervision has substantially poorer quality and diversity; the IGNN adapters converge reliably but return one configuration per graph with low Hit10. Deeper untied models improve quality but lag in Hit10 and distinct-optimum discovery. DIFUSCO (large) finds more optima $( D _ { 2 0 } = 2 . 6 0 )$ , but has worse mean energy despite approximately 606 times as many parameters. Consistent with our theory, one shared update trained on energy can converge to multiple highquality solutions, ofering a favorable combination of convergence, quality, and diversity over single-target training or enforced uniqueness in these experiments.

## 3.2. Structural module detection in protein graphs.

Problem. We seek structural modules in protein graphs: groups more internally connected than expected from their node degrees. We replace graph classification on the 1,113 PROTEINS graphs [9, 43] with instance-wise modularity maximization [46]. For an undirected graph $G =$ (V, E) with node degree $d _ { i }$ and node assignments $z _ { i } \in \{ 1 , \ldots , K \}$ , the objective and solution set are

$$
Q _ { G } ( z ) = \frac { 1 } { | E | } \sum _ { \{ i , j \} \in E } \mathbf { 1 } [ z _ { i } = z _ { j } ] - \sum _ { k = 1 } ^ { K } \left( \frac { \sum _ { i } d _ { i } \mathbf { 1 } [ z _ { i } = k ] } { 2 | E | } \right) ^ { 2 } , \ S ( G ) = \underset { z \in \{ 1 , \ldots , K \} ^ { | V | } } { \arg \operatorname* { m a x } } Q _ { G } ( z ) .
$$

We allow at most four modules $( K = 4$ , with empty modules permitted). Partitions can have similar energies even modulo module-label permutations, motivating distinct low-energy predictions without one selected target.

Models and baselines. Our shared recurrent update $P _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , P _ { t } )$ refines a randomly initialized soft partition, where $P _ { t , i } \in \mathbb { R } ^ { K }$ contains node i’s assignment weights over K modules. After $T$ iterations, each node selects its highest-weight module, $z _ { i } = \mathrm { a r g } \operatorname* { m a x } _ { k } ( P _ { T , i } ) _ { k }$ Training minimizes $\mathbb { E } [ e _ { G } ( P _ { T } ) ]$ with $T = 5$ over training graphs and initializations, where e<sub>G</sub> is a diferentiable soft-partition energy derived from modularity. At test time, we apply the learned update for 200 steps from 20 independent initializations per graph.

The four baseline groups (Section 3.1) test training objectives, equilibrium uniqueness, weight sharing, and alternative solution generation. Appendix F details the soft energy, architectures, training, and evaluation protocols.

Results. Our model has the highest mean modularity (0.5560) and Hit10 $( 7 7 . 4 9 \% )$ in Table 2, finding 5.83 distinct qualified partitions per graph on average and at least two on 80.78% of graphs. Single-target supervision and unique-equilibrium IGNNs have poorer quality and diversity. Untied depth improves performance, but five independent blocks still lag in mean modularity, Hit10, and partition discovery. DIFUSCO (large) finds more qualified partitions $( D _ { 2 0 } ^ { . 1 0 } = 1 1 . 0 2 )$ but has lower mean modularity with approximately 155 times as many parameters. Together with the convergence results in Appendix F (Table 10), these findings support our theoretical premise: energy training without prescribed targets lets one shared update recover multiple high-quality solutions from diferent initializations.

3.3. Chemical reaction-network multistationarity. Problem. A chemical reaction network has s species i and R reactions r. The matrices $\alpha , \breve { \beta } \in \mathbb { Z } _ { \geq 0 } ^ { R \times s }$ give the numbers of molecules of species i consumed $\left( \alpha _ { r i } \right)$ and produced $\left( \beta _ { r i } \right)$ by reaction $r ; \kappa \in \mathbb R _ { > 0 } ^ { R }$ contains rate constants. The graph $G = ( \alpha , \beta , \kappa )$ has one node per species and reaction, connected bidirectionally when $\alpha _ { r i } + \beta _ { r i } > 0$ . Edges carry $( \alpha _ { r i } , \beta _ { r i } ) ;$ ; reaction nodes encode log $\kappa _ { r }$ . Under mass-action kinetics, the species concentrations $c \in \mathbb { R } _ { > 0 } ^ { s }$ evolve according to

$$
\dot { c } = f _ { G } ( c ) = N v ( c ) , \qquad N _ { i r } = \beta _ { r i } - \alpha _ { r i } , \qquad v _ { r } ( c ) = \kappa _ { r } \prod _ { i = 1 } ^ { s } c _ { i } ^ { \alpha _ { r i } } .
$$

We seek stationary solutions $\mathcal { S } ( G ) = \{ c \in \mathbb { R } _ { > 0 } ^ { s } : f _ { G } ( c ) = 0 \}$ . Multiple positive solutions for one G constitute multistationarity, a set-valued prediction task. The solver seeks roots of $f _ { G }$ without requiring physical stability under the chemical ODE. Our kinetic instances use published network structures [68], with two numerically verified stationary witnesses per instance.

Models and baselines. Species log concentrations follow the shared update $z _ { t + 1 } =$ $\mathrm { G N N } _ { \theta } ( G , z _ { t } )$ , with output $c _ { T } = \exp z _ { T }$ . Training minimizes equation residuals without prescribed targets, using randomized unrolls of at most 32 steps; the main comparison uses $T = 2 0 0$ . The four baseline groups (Section 3.1) are single-target MSE, physics-trained contractive IGNNs, untied feedforward depth controls, and distributional models. For continuous states, the latter comprise DDPM, DDIM, and DIS with matched and larger upstream-recipe graph adaptations; details are in Appendix G.

Table 2. Structural module detection on 111 PROTEINS test graphs with 20 candidates per graph. $\overline { { Q } } _ { G }$ averages modularity over candidates and then graphs. For optimal modularity $Q _ { G } ^ { \star }$ , Hit10 is the percentage of candidates with relative gap $( Q _ { G } ^ { \star } - Q _ { G } ( z ) ) / | Q _ { G } ^ { \star } | \leq 0 . 1 0 . \ D _ { 2 0 } ^ { . 1 0 } .$ : mean number of distinct partitions per graph with absolute gap $Q _ { G } ^ { \star } - Q _ { G } ( z ) \leq 0 . 1 0 ; \ : G _ { 2 0 } ^ { . 1 0 }$ : percentage of graphs with at least two such partitions; C<sub>200</sub>: percentage of trajectories numerically converged at $t = 2 0 0 .$ . Results: mean ± standard deviation over training seeds $0 / 1 / 2$ on the same test set. Full protocol in Appendix F.
<table><tr><td>Method</td><td> $\overline { { Q } } _ { G } \mathrm { \uparrow }$ </td><td>Hit10 ↑</td><td> $C _ { 2 0 0 } \uparrow$ </td><td> $D _ { 2 0 } ^ { . 1 0 } \uparrow$ </td><td> $G _ { 2 0 } ^ { . 1 0 } \uparrow$ </td></tr><tr><td>Ours</td><td> $0 . 5 5 6 0 \pm 0 . 0 0 2 4$ </td><td> $7 7 . 4 9 \pm 2 . 0 6$ </td><td> $8 8 . 0 5 \pm 2 . 2 3 $ </td><td> $5 . 8 3 \pm 0 . 3 8$ </td><td> $8 0 . 7 8 \pm 0 . 4 2$ </td></tr><tr><td>Single-target Supervision</td><td> $0 . 4 9 1 1 \pm 0 . 0 1 1 4$ </td><td> $4 1 . 2 8 \pm 7 . 1 6$ </td><td> $5 0 . 8 1 \pm 1 0 . 8 4$ </td><td> $1 . 1 1 \pm 0 . 0 8$ </td><td> $3 3 . 0 3 \pm 2 . 2 5$ </td></tr><tr><td>IGNN (H16)</td><td> $0 . 4 1 4 7 \pm 0 . 0 0 7 9$ </td><td> $8 . 1 1 \pm 2 . 6 5$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 0 \pm 0 . 0 4$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN (H32)</td><td> $0 . 4 1 9 6 \pm 0 . 0 0 4 4$ </td><td> $1 1 . 7 1 \pm 2 . 2 1$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 2 \pm 0 . 0 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN (H64)</td><td> $0 . 4 2 3 5 \pm 0 . 0 1 2 1$ </td><td> $8 . 4 1 \pm 2 . 3 6$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 2 \pm 0 . 0 6$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Feedforward (×1)</td><td> $0 . 4 4 3 6 \pm 0 . 0 0 4 8$ </td><td> $1 6 . 5 2 \pm 2 . 2 2$ </td><td></td><td> $1 . 3 5 \pm 0 . 2 3$ </td><td> $3 0 . 9 3 \pm 6 . 2 6$ </td></tr><tr><td>Feedforward (×2)</td><td> $0 . 4 8 4 6 \pm 0 . 0 1 5 6$ </td><td> $2 9 . 5 0 \pm 3 . 6 1$ </td><td></td><td> $2 . 0 9 \pm 0 . 2 6$ </td><td> $5 3 . 1 5 \pm 6 . 6 2$ </td></tr><tr><td>Feedforward (×3)</td><td> $0 . 5 0 8 8 \pm 0 . 0 1 0 8$ </td><td> $3 9 . 9 2 \pm 5 . 1 7$ </td><td></td><td> $2 . 9 5 \pm 0 . 6 2$ </td><td> $7 1 . 4 7 \pm 9 . 7 1$ </td></tr><tr><td>Feedforward (×4)</td><td> $0 . 5 2 8 5 \pm 0 . 0 0 2 2$ </td><td> $5 1 . 3 5 \pm 1 . 6 8$ </td><td></td><td> $4 . 2 4 \pm 0 . 5 1$ </td><td> $8 2 . 8 8 \pm 6 . 6 2$ </td></tr><tr><td>Feedforward (×5)</td><td> $0 . 5 3 0 5 \pm 0 . 0 0 4 4$ </td><td> $5 4 . 9 7 \pm 3 . 5 9$ </td><td></td><td> $4 . 3 2 \pm 0 . 4 7$ </td><td> $7 8 . 6 8 \pm 4 . 4 9$ </td></tr><tr><td>Mean-field GNN</td><td> $0 . 4 0 2 0 \pm 0 . 0 1 2 2$ </td><td> $8 . 5 3 \pm 2 . 0 1$ </td><td></td><td> $0 . 6 9 \pm 0 . 0 5$ </td><td> $1 6 . 2 2 \pm 0 . 0 0$ </td></tr><tr><td>Annealed Bernoulli GNN</td><td> $0 . 4 1 3 1 \pm 0 . 0 1 0 3$ </td><td> $9 . 9 1 \pm 2 . 3 5$ </td><td></td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $1 8 . 3 2 \pm 1 . 5 3$ </td></tr><tr><td>DIFUSCO (matched)</td><td> $0 . 5 1 1 5 \pm 0 . 0 1 2 7$ </td><td> $4 1 . 8 5 \pm 7 . 9 7$ </td><td></td><td> $8 . 4 7 \pm 1 . 2 0$ </td><td> $9 7 . 3 0 \pm 1 . 2 7$ </td></tr><tr><td>DIFUSCO (large)</td><td> $0 . 5 5 1 5 \pm 0 . 0 0 1 4$ </td><td> $7 4 . 4 9 \pm 1 . 6 6$ </td><td></td><td> $1 1 . 0 2 \pm 0 . 0 3$ </td><td> $9 7 . 0 0 \pm 0 . 4 2$ </td></tr><tr><td>VAG (matched)</td><td> $0 . 3 9 5 2 \pm 0 . 0 0 2 4$ </td><td> $8 . 0 5 \pm 0 . 1 4$ </td><td></td><td> $0 . 5 7 \pm 0 . 0 1$ </td><td> $1 2 . 0 1 \pm 0 . 8 5$ </td></tr><tr><td>VAG (large)</td><td> $0 . 2 5 7 7 \pm 0 . 0 2 3 7$ </td><td> $0 . 4 4 \pm 0 . 3 1$ </td><td></td><td> $0 . 3 5 \pm 0 . 1 5$ </td><td> $7 . 2 1 \pm 1 . 4 7$ </td></tr></table>

Results. Our model achieves 83.77% candidate hits and 3.506 accepted prediction clusters per graph on average, with at least two on 41.67% of graphs (Table 3). Numerical settling increases from 91.99% at $T = 2 0 0 \ : \mathrm { t o 9 6 . 1 5 \% }$ at $T = 5 { , } 0 0 0 { ; }$ remaining oscillations and unresolved cases appear in Table 13. Single-target supervision reaches only 18.78% hits; IGNNs settle in their hidden-state operators but yield no accepted predictions, illustrating that convergence alone does not ensure validity. Untied depth improves accuracy, but five independent stages reach only 0.58% hits; tested distributional models also have low hit rates. These results demonstrate multiple residual-qualified predictions and high numerical settling rates. They do not certify distinct exact roots or complete root coverage. Baselines difer in objectives and inference budgets, so the comparison does not isolate recurrence at equal computational cost.

## 4. Conclusions

Multi-attractor dynamics provide a mechanism for representing set-valued graph tasks with recurrent, weight-tied GNNs. Under the stated assumptions, one shared update can converge arbitrarily close to diferent valid solutions, while continuous, equivariant contraction can obstruct recovery of even one. Across three scientific tasks, energy or residual training produces multiple high-quality predictions and high numerical convergence rates without solution labels. These findings motivate preserving access to multiple equilibria when tasks admit multiple answers. This general framework opens opportunities across graph applications, with success depending on problem-specific objectives and optimization. Understanding how training shapes convergence and solution coverage remains an important direction.

Table 3. Chemistry results on 236 test graphs with 20 candidates per graph. ${ \overline { { E } } } _ { \mathrm { r e s } } .$ mean Huber penalty on the mass-action residual (lower is better); $\operatorname { H i t } _ { 1 0 ^ { - 3 } } \colon$ percentage of final states with normalized physical residual $R \leq 1 0 ^ { - 3 }$ $C { : }$ percentage of recurrent trajectories with next-step log-concentration RMS change below $1 0 ^ { - 3 }$ at $T \ = \ 2 0 0 \colon$ $D _ { 2 0 } ^ { \mathrm { r a w } }$ : mean number of distinct accepted prediction clusters per graph; $G _ { \mathrm { 2 0 } } ^ { \mathrm { r a w } }$ : percentage of graphs with at least two. These count separated accepted predictions, not certified distinct roots. Entries: mean ± population standard deviation over training seeds $0 / 1 / 2$
<table><tr><td>Method</td><td> ${ \overline { { E } } } _ { \mathrm { r e s } } \downarrow$ </td><td> $\operatorname { H i t } _ { 1 0 ^ { - 3 } } \uparrow$ </td><td>C↑</td><td> $D _ { 2 0 } ^ { \mathrm { r a w } } \uparrow$ </td><td></td><td> $G _ { 2 0 } ^ { \mathrm { r a w } } \uparrow$ </td></tr><tr><td>Ours</td><td> $0 . 0 1 8 8 \pm 0 . 0 0 4 6$ </td><td> $8 3 . 7 7 \pm 3 . 4 4$ </td><td> $9 1 . 9 9 \pm 0 . 3 5$ </td><td> $3 . 5 0 6 \pm 0 . 3 1 4$ </td><td></td><td> $4 1 . 6 7 \pm 1 7 . 0 1$ </td></tr><tr><td>Single-target Supervis</td><td> $0 . 0 3 7 3 \pm 0 . 0 0 5 9$ </td><td> $1 8 . 7 8 \pm 3 . 8 8$ </td><td> $8 8 . 4 8 \pm 4 . 2 9$ </td><td> $1 . 3 7 1 \pm 0 . 1 3 7$ </td><td></td><td> $1 1 . 3 0 \pm 0 . 7 2$ </td></tr><tr><td>IGNN (H16)</td><td> $0 . 1 2 1 4 \pm 0 . 0 8 2 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN (H32)</td><td> $0 . 1 1 8 0 \pm 0 . 0 7 0 9$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>IGNN (H64)</td><td> $0 . 0 9 6 0 \pm 0 . 0 0 5 6$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Feedforward (×1)</td><td> $0 . 1 0 8 2 \pm 0 . 0 0 8 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td></td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Feedforward (×2)</td><td> $0 . 0 5 8 6 \pm 0 . 0 0 5 0$ </td><td> $0 . 0 1 \pm 0 . 0 2$ </td><td></td><td> $0 . 0 0 3 \pm 0 . 0 0 4$ </td><td></td><td> $0 . 1 4 \pm 0 . 2 0$ </td></tr><tr><td>Feedforward (×3)</td><td> $0 . 0 4 4 1 \pm 0 . 0 0 6 5$ </td><td> $0 . 1 1 \pm 0 . 1 0$ </td><td></td><td> $0 . 0 2 1 \pm 0 . 0 1 9$ </td><td></td><td> $0 . 2 8 \pm 0 . 2 0$ </td></tr><tr><td>Feedforward (×4)</td><td> $0 . 0 2 9 5 \pm 0 . 0 0 1 6$ </td><td> $0 . 4 2 \pm 0 . 3 5$ </td><td></td><td> $0 . 0 7 6 \pm 0 . 0 6 1$ </td><td></td><td> $0 . 4 2 \pm 0 . 3 5$ </td></tr><tr><td>Feedforward (×5)</td><td> $0 . 0 2 5 5 \pm 0 . 0 0 2 2$ </td><td> $0 . 5 8 \pm 0 . 4 1$ </td><td></td><td> $0 . 1 1 0 \pm 0 . 0 7 8$ </td><td></td><td> $0 . 7 1 \pm 0 . 5 3$ </td></tr><tr><td>DDPM / matched</td><td> $0 . 0 5 7 6 \pm 0 . 0 0 4 7$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td></td><td> $0 . 0 0 6 \pm 0 . 0 0 2$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>DDIM / matched</td><td> $0 . 0 7 3 2 \pm 0 . 0 0 5 5$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td></td><td> $0 . 0 0 3 \pm 0 . 0 0 2$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>DIS / matched</td><td> $0 . 9 3 1 5 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td></td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>DDPM / large</td><td> $0 . 1 1 9 7 \pm 0 . 0 0 4 0$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td></td><td> $0 . 0 0 3 \pm 0 . 0 0 2$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>DDIM / large</td><td> $0 . 1 3 1 7 \pm 0 . 0 0 3 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td></td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>DIS / large</td><td> $0 . 0 4 6 3 \pm 0 . 0 0 0 3$ </td><td> $0 . 0 2 \pm 0 . 0 2$ </td><td></td><td> $0 . 0 0 4 \pm 0 . 0 0 3$ </td><td></td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr></table>

## References

[1] Ralph Abboud, Ismail Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The surprising power of graph neural networks with random node initialization. arXiv preprint arXiv:2010.01179, 2020.

[2] Lada A Adamic and Eytan Adar. Friends and neighbors on the web. Social networks, 25 (3):211–230, 2003.

[3] Cem Anil, Ashwini Pokle, Kaiqu Liang, Johannes Treutlein, Yuhuai Wu, Shaojie Bai, J Zico Kolter, and Roger B Grosse. Path independent equilibrium models can better exploit test-time computation. Advances in Neural Information Processing Systems, 35: 7796–7809, 2022.

[4] Waiss Azizian and marc lelarge. Expressive power of invariant and equivariant graph neural networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=lxHgXYN4bwl.

[5] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Deep equilibrium models. Advances in neural information processing systems, 32, 2019.

[6] Justin Baker, Qingsong Wang, Cory D Hauck, and Bao Wang. Implicit graph neural networks: A monotone operator viewpoint. In International conference on machine learning, pages 1521–1548. PMLR, 2023.

[7] Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. Advances in Neural Information Processing Systems, 35:20232–20242, 2022.

[8] Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on difusion-based generative modeling. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=oYIjw37pTP.

[9] Karsten M. Borgwardt, Cheng Soon Ong, Stefan Sch¨onauer, S. V. N. Vishwanathan, Alex J. Smola, and Hans-Peter Kriegel. Protein function prediction via graph kernels. Bioinformatics, 2005.

[10] Paul Breiding and Sascha Timme. HomotopyContinuation.jl: A package for homotopy continuation in Julia. In Mathematical Software – ICMS 2018, volume 10931 of Lecture Notes in Computer Science, 2018.

[11] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=F72ximsx7C1.

[12] Chen Cai and Yusu Wang. A simple yet efective baseline for non-attributed graph classification. arXiv preprint arXiv:1811.03508, 2018.

[13] Qian Chen, Tianjian Zhang, Linxin Yang, Qingyu Han, Akang Wang, Ruoyu Sun, Xiaodong Luo, and Tsung-Hui Chang. Symilo: A symmetry-aware learning framework for integer linear optimization. Advances in Neural Information Processing Systems, 37:24411– 24434, 2024.

[14] Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph isomorphism testing and function approximation with gnns. Advances in neural information processing systems, 32, 2019.

[15] Kyunghyun Cho, Bart Van Merri¨enboer, C¸ a˘glar Gul¸cehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder–decoder for statistical machine translation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1724–1734, 2014.

[16] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Li\`o, and Petar Veliˇckovi´c. Principal neighbourhood aggregation for graph nets. Advances in neural information processing systems, 33:13260–13271, 2020.

[17] Hanjun Dai, Zornitsa Kozareva, Bo Dai, Alex Smola, and Le Song. Learning steady-states of iterative algorithms over graphs. In International conference on machine learning, pages 1106–1114. PMLR, 2018.

[18] Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=wTTjnvGphYj.

[19] Nadav Dym, Hannah Lawrence, and Jonathan W. Siegel. Equivariant frames and the impossibility of continuous canonicalization. In International Conference on Machine Learning, 2024.

[20] Patrick E. Farrell, Asgeir Birkisson, and Simon W. Funke. Deflation techniques for finding<sup>´</sup> distinct solutions of nonlinear partial diferential equations. SIAM Journal on Scientific Computing, 37(4):A2026–A2045, 2015.

[21] Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural networks. arXiv preprint arXiv:2204.04661, 2022.

[22] Abhinav Goel, Derek Lim, Hannah Lawrence, Stefanie Jegelka, and Ningyuan Huang. Anysubgroup equivariant networks via symmetry breaking. arXiv preprint arXiv:2603.19486, 2026.

[23] Lukas Gonon, Thilo Meyer-Brandis, and Niklas Weber. Universality and approximation rates of graph neural networks with random features. arXiv preprint arXiv:2607.26699, 2026.

[24] Fangda Gu, Heng Chang, Wenwu Zhu, Somayeh Sojoudi, and Laurent El Ghaoui. Implicit graph neural networks. Advances in neural information processing systems, 33:11984– 11995, 2020.

[25] Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2026. URL https: //www.gurobi.com.

[26] Aric Hagberg, Pieter J Swart, and Daniel A Schult. Exploring network structure, dynamics, and function using networkx. Technical report, Los Alamos National Laboratory (LANL), 2007.

[27] Fleur Hendriks, Ondˇrej Rokoˇs, Martin Doˇsk´aˇr, Marc GD Geers, and Vlado Menkovski. Equivariant flow matching for symmetry-breaking bifurcation problems. arXiv preprint arXiv:2509.03340, 2025.

[28] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[29] John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554–2558, 1982.

[30] Caleb Jore and Jialin Liu. Bifurcation models: Learning set-valued solution maps with weight-tied dynamics. arXiv preprint arXiv:2605.07277, 2026.

[31] S´ekou-Oumar Kaba and Siamak Ravanbakhsh. Symmetry breaking and equivariant neural networks. arXiv preprint arXiv:2312.09016, 2023.

[32] Nikolaos Karalias and Andreas Loukas. Erdos goes neural: an unsupervised learning framework for combinatorial optimization on graphs. Advances in Neural Information Processing Systems, 33:6659–6672, 2020.

[33] Nicolas Keriven and Gabriel Peyr´e. Universal invariant and equivariant graph neural networks. Advances in neural information processing systems, 32, 2019.

[34] Hannah Lawrence, Vasco Portilheiro, Yan Zhang, and S´ekou-Oumar Kaba. Improving equivariant networks with probabilistic symmetry breaking. In International Conference on Learning Representations, 2025.

[35] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493, 2015.

[36] Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Combinatorial optimization with graph convolutional networks and guided tree search. Advances in neural information processing systems, 31, 2018.

[37] David Liben-Nowell and Jon Kleinberg. The link prediction problem for social networks. In Proceedings of the twelfth international conference on Information and knowledge management, pages 556–559, 2003.

[38] Jialin Liu, Lisang Ding, Stanley J Osher, and Wotao Yin. Expressive power of implicit models: Rich equilibria and test-time scaling. In International Conference on Learning Representations, volume 2026, pages 17164–17212, 2026.

[39] Juncheng Liu, Kenji Kawaguchi, Bryan Hooi, Yiwei Wang, and Xiaokui Xiao. Eignn: Eficient infinite-depth graph neural networks. Advances in Neural Information Processing Systems, 34:18762–18773, 2021.

[40] Andreas Loukas. What graph neural networks cannot learn: depth vs width. arXiv preprint arXiv:1907.03199, 2019.

[41] Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. Advances in neural information processing systems, 32, 2019.

[42] Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, 2019.

[43] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663, 2020.

[44] James R. Munkres. Topology. Pearson Education Limited, Harlow, England, 2 edition, 2014. ISBN 978-1-292-02362-5.

[45] Yatin Nandwani, Deepanshu Jindal, Parag Singla, et al. Neural learning of one-ofmany solutions for combinatorial problems in structured output spaces. arXiv preprint arXiv:2008.11990, 2020.

[46] Mark EJ Newman. Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23):8577–8582, 2006.

[47] Junyoung Park, Jinhyun Choo, and Jinkyoo Park. Convergent graph solvers. arXiv preprint arXiv:2106.01680, 2021.

[48] Hubert Ramsauer, Bernhard Sch¨afl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlovi´c, Geir Kjetil Sandve, et al. Hopfield networks is all you need. arXiv preprint arXiv:2008.02217, 2020.

[49] Federico Ricci-Tersenghi. The bethe approximation for solving the inverse ising problem: a comparison with other inference methods. Journal of Statistical Mechanics: Theory and Experiment, 2012(08):P08015, 2012.

[50] Eran Rosenbluth and Martin Grohe. Repetition makes perfect: Recurrent graph neural networks match message passing limit. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[51] Sebastian Sanokowski, Wilhelm Berghammer, Sepp Hochreiter, and Sebastian Lehner. Variational annealing on graphs for combinatorial optimization. Advances in Neural Information Processing Systems, 36:63907–63930, 2023.

[52] Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks. In Proceedings of the 2021 SIAM international conference on data mining (SDM), pages 333–341. SIAM, 2021.

[53] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1): 61–80, 2008.

[54] Martin JA Schuetz, J Kyle Brubaker, and Helmut G Katzgraber. Combinatorial optimization with physics-inspired graph neural networks. Nature Machine Intelligence, 4(4): 367–377, 2022.

[55] Daniel Selsam, Matthew Lamm, Benedikt B¨unz, Percy Liang, Leonardo de Moura, and David L Dill. Learning a sat solver from single-bit supervision. arXiv preprint arXiv:1802.03685, 2018.

[56] Tess E. Smidt, Mario Geiger, and Benjamin Kurt Miller. Finding symmetry breaking order parameters with Euclidean neural networks. Physical Review Research, 3:L012002, 2021.

[57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=St1giarCHLP.

[58] Haoran Sun, Etash K Guha, and Hanjun Dai. Annealed training for combinatorial optimization on graphs. arXiv preprint arXiv:2207.11542, 2022.

[59] Zhiqing Sun and Yiming Yang. Difusco: Graph-based difusion solvers for combinatorial optimization. Advances in neural information processing systems, 36:3706–3731, 2023.

[60] Hao Tang, Zhiao Huang, Jiayuan Gu, Bao-Liang Lu, and Hao Su. Towards scale-invariant graph-related problem solving by iterative homogeneous gnns. Advances in Neural Information Processing Systems, 33:15811–15822, 2020.

[61] Jan Toenshof, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Graph neural networks for maximum constraint satisfaction. Frontiers in artificial intelligence, 3:580607, 2021.

[62] Petar Veliˇckovi´c, Rex Ying, Matilde Padovano, Raia Hadsell, and Charles Blundell. Neural execution of graph algorithms. arXiv preprint arXiv:1910.10593, 2019.

[63] Ezra Winston and J Zico Kolter. Monotone operator equilibrium networks. Advances in neural information processing systems, 33:10718–10728, 2020.

[64] Dian Wu, Lei Wang, and Pan Zhang. Solving statistical mechanics using variational autoregressive networks. Physical review letters, 122(8):080602, 2019.

[65] YuQing Xie and Tess Smidt. Equivariant symmetry breaking sets. Transactions on Machine Learning Research, 2024.

[66] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.

[67] Yongyi Yang, Tang Liu, Yangkun Wang, Zengfeng Huang, and David Wipf. Implicit vs unfolded graph neural networks. Journal of Machine Learning Research, 26(82):1–46, 2025.

[68] Shenghao Yao, AmirHosein Sadeghimanesh, and Matthew England. Understanding multistationarity of fully open reaction networks: S. yao, ah sadeghimanesh, m. england. Bulletin of Mathematical Biology, 87(12):176, 2025.

[69] Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron C Courville, Yoshua Bengio, and Ling Pan. Let the flows tell: Solving graph combinatorial problems with gflownets. Advances in neural information processing systems, 36:11952–11969, 2023.

[70] Tao Zhou, Linyuan L¨u, and Yi-Cheng Zhang. Predicting missing links via local information. The European Physical Journal B, 71(4):623–630, 2009.

## Appendix Contents

Appendix A. Ideal Operator Existence 16   
A.1. Basic concepts and main result 17   
A.2. Bad set and displacement functions 18   
A.3. Proof of the main result 21   
Appendix B. Family-wide asymptotic realization by a recurrent MP-GNN 25   
B.1. Architecture and main result 25   
B.2. Exact realization using the current node states 27   
B.3. Node separation and a common collection of boxes 30   
B.4. Proof of the main result 33   
Appendix C. An obstruction to almost-sure validity for contractive equivariant   
dynamics 38   
C.1. The operator-level obstruction 39   
C.2. Application to independently weighted triangles 41   
C.3. Consequence for recurrent message-passing GNNs 43   
Appendix D. Single-source difusion: protocol and additional results 43   
Appendix E. Ising Experiments: Details and Additional Results 45   
Appendix F. Structural module detection in protein graphs: details and additional   
results 50   
Appendix G. Chemical reaction networks: details and additional results 55   
Appendix H. Related Work 61

## Appendix A. Ideal Operator Existence

In the appendices, $S ( G )$ and $\mathcal { \partial } _ { n }$ denote $S ( G )$ and $Y _ { n }$ from the main text. Fix the number of nodes n. We represent a graph instance as

$$
G = ( A , X ) \in \mathcal { E } _ { n } : = \mathbb { R } ^ { n \times n \times d _ { e } } \times \mathbb { R } ^ { n \times d _ { v } } ,
$$

where

$$
A = ( A ^ { ( 1 ) } , \dots , A ^ { ( d _ { e } ) } )
$$

contains the edge features and X contains the node features. The output space is

$$
\mathcal { V } _ { n } = \mathbb { R } ^ { n \times q } \cong \mathbb { R } ^ { m } .
$$

Both spaces are equipped with their Euclidean norms.

For a permutation π of the n nodes, let $P _ { \pi }$ denote its permutation matrix, with $P _ { \pi } e _ { i } = e _ { \pi ( i ) }$ Relabeling the nodes of a graph and an output is defined by

$$
\pi G : = \left( \left( P _ { \pi } A ^ { ( r ) } P _ { \pi } ^ { \top } \right) _ { r = 1 } ^ { d _ { e } } , P _ { \pi } X \right) , \qquad \pi y : = P _ { \pi } y .
$$

Since permutation matrices are orthogonal, these relabelings preserve Euclidean distances.

Throughout, ${ \mathcal { G } } \subseteq { \mathcal { E } } _ { n }$ is closed under node relabeling:

$$
G \in { \mathcal { G } } \quad \Longrightarrow \quad \pi G \in { \mathcal { G } }
$$

for every node permutation π. The domain $\mathcal { G }$ need not be bounded, closed, connected, or compact.

## A.1. Basic concepts and main result.

Definition 1 (Permutation-equivariant finite-branch solution map). A finite-branch solution map on $\mathcal { G }$ is represented, for some integer $M \geq 1$ , by labeled branches

$$
f _ { 1 } , \ldots , f _ { M } : \mathcal { G } \to \mathcal { V } _ { n }
$$

through

$$
S ( G ) : = \{ f _ { 1 } ( G ) , \ldots , f _ { M } ( G ) \} .
$$

Distinct branch indices are allowed to have coincident values. Since $S ( G )$ is a set, if $f _ { i } ( G ) =$ $f _ { j } ( G )$ for $i \neq j$ , their common value appears only once in $S ( G )$

We say that this representation is permutation equivariant if, for every node permutation $\pi ,$ there exists a permutation $\tau _ { \pi }$ of the branch indices $\{ 1 , \dots , M \}$ such that

$$
f _ { \tau _ { \pi } ( i ) } ( \pi G ) = \pi f _ { i } ( G ) \qquad f o r e v e r y G \in \mathcal { G } \ a n d \ i \in \{ 1 , \ldots , M \} .
$$

Thus, relabeling the nodes may relabel the solution branches, but it does not change the underlying solution set. In particular,

$$
S ( \pi G ) = \pi S ( G ) : = \{ \pi s : s \in S ( G ) \} .
$$

Definition 2 (Regular and stable graph instances). For each branch, let

$$
D _ { i } : = \{ G \in \mathcal { G } : f _ { i } \ i s \ n o t \ l o c a l l y \ L i p s c h i t z \ a t \ G \} , \qquad D : = \bigcup _ { i = 1 } ^ { M } D _ { i } .
$$

A graph $G _ { 0 }$ has a stable collapse pattern $i f$ some relatively open neighborhood V $o f G _ { 0 }$ admits a partition

$$
\{ 1 , \dots , M \} = C _ { 1 } \dot { \cup } \cdot \cdot \cdot \dot { \cup } C _ { r }
$$

such that, for every $G \in V$

$$
f _ { i } ( G ) = f _ { j } ( G ) \quad \Longleftrightarrow \quad i , j \in C _ { \alpha } \ f o r \ s o m e \ \alpha .
$$

Let $U$ be the set of graphs without a stable collapse pattern and define

$$
{ \mathcal { G } } _ { \star } : = { \mathcal { G } } \setminus ( D \cup U ) .
$$

We call G<sub>⋆</sub> the regular stable domain. Both D and U are relatively closed in ${ \mathcal { G } } .$ . No emptiness, finiteness, or measure-zero assumption is imposed on $D .$

Remark (Equivariant branch labels). At the level of the underlying set-valued map, the fixed label permutations in Definition 1 impose no additional restriction. Indeed, given any finite representation $S ( G ) = \left\{ f _ { 1 } ( G ) , \dots , f _ { M } ( G ) \right\}$ satisfying $S ( \pi G ) = \pi S ( G )$ , define

$$
\widetilde { f } _ { ( i , \sigma ) } ( G ) : = \sigma f _ { i } ( \sigma ^ { - 1 } G ) , \qquad \widetilde { \tau } _ { \pi } ( i , \sigma ) : = ( i , \pi \sigma ) , \qquad \sigma \in \mathfrak { S } _ { n } .
$$

These Mn! labels represent the same solution set and satisfy Definition 1. If D and $\mathcal { G } _ { \star }$ <sub>⋆</sub> are computed from the original labels, the symmetrized representation has

$$
\widetilde D = \bigcup _ { \sigma \in { \mathfrak { S } } _ { n } } \sigma D , \qquad \widetilde { \mathcal { G } } _ { \star } = \bigcap _ { \sigma \in { \mathfrak { S } } _ { n } } \sigma { \mathcal { G } } _ { \star } .
$$

For the second identity, a stable partition restricts to each σ-block; conversely, local continuity and separation of the distinct solution values make matches between these blocks locally constant. Thus symmetrization may shrink the regular stable domain, and probability assumptions on that domain must be rechecked. If the original representation already satisfies Definition 1, both domains are invariant by Lemma 1 and remain unchanged.

Definition 3 (Switching set and strict basins). For $G \in \mathcal G _ { \star }$ , the reduced switching set is

$$
\Sigma _ { G } ^ { \circ } : = \left\{ y \in \mathcal { V } _ { n } : \# \underset { s \in S ( G ) } { \operatorname { a r g m i n } } \Vert y - s \Vert \geq 2 \right\} ,
$$

where coincident branch values are counted only once. $F o r \ : s \in S ( G )$ , its strict Voronoi basin is

$$
V _ { s } ( G ) : = \{ y \in \mathcal { Y } _ { n } : \| y - s \| < \| y - r \| \ f o r \ e v e r y \ r \in S ( G ) \setminus \{ s \} \} .
$$

Theorem 4 (Jointly globally Lipschitz equivariant multi-attractor dynamics). Let $S$ be an equivariant finite-branch solution map in the sense of Definition 1, with regular stable domain and switching geometry given by Definitions 2 and 3. Then there exists

$$
T : \mathcal { G } \times \mathcal { Y } _ { n }  \mathcal { Y } _ { n } , \qquad ( G , y ) \mapsto T _ { G } ( y ) ,
$$

with the following properties.

(1) Regularity and symmetry. The map $T$ is jointly globally Lipschitz and permutation equivariant. In particular, for some $L _ { T } < \infty$

$$
\| T _ { G } ( y ) - T _ { H } ( z ) \| \leq L _ { T } \sqrt { \| G - H \| ^ { 2 } + \| y - z \| ^ { 2 } } , \qquad T _ { \pi G } ( \pi y ) = \pi T _ { G } ( y ) .\tag{6}
$$

(2) Exceptional configurations. One may choose $T _ { G } ( y ) = y$ whenever $G \in D \cup U$ , or whenever $G \in { \mathcal { G } } _ { \star }$ and $y \in \Sigma _ { G } ^ { \circ }$

(3) Basin-wise convergence. Fix $G \in { \mathcal { G } } _ { \star }$ <sub>⋆</sub> and y<sub>0</sub> $\not \in \Sigma _ { G } ^ { \circ }$ , and let

$$
s : = \underset { r \in S ( G ) } { \mathrm { a r g m i n } } \Vert y _ { 0 } - r \Vert .
$$

Then $y _ { t + 1 } = T _ { G } ( y _ { t } )$ converges to $s ,$ and there is a trajectory-dependent constant $c ( G , y _ { 0 } ) \in$ (0, 1) such that

$$
\begin{array} { r } { \| y _ { t } - s \| \leq \big ( 1 - c ( G , y _ { 0 } ) \big ) ^ { t } \| y _ { 0 } - s \| . } \end{array}\tag{7}
$$

No uniform positive lower bound on $c ( G , y _ { 0 } )$ is claimed.

(4) Almost-everywhere recovery. For every $G \in { \mathcal { G } } _ { \star } , { \mathcal { L } } ^ { m } ( \Sigma _ { G } ^ { \circ } ) = 0$ . Let $\mu$ be a Borel probability measure on ${ \mathcal { V } } _ { n }$ with $\mu \ll \mathcal { L } ^ { m }$ , and draw $y _ { 0 } \sim \mu$ . Then the iteration $y _ { t }$ converges to S(G) for µ-almost every initialization. $I f$ in addition $\mu ( V _ { s } ( G ) ) > 0$ for every $s \in S ( G )$ , then

$$
\operatorname { s u p p } { \mathcal { L } } ( y _ { \infty } \mid G ) = S ( G ) ,\tag{8}
$$

where $y _ { \infty }$ denotes the almost-sure iteration limit.

A.2. Bad set and displacement functions. Before the proof, we state some definitions and lemmas.

Definition 4 (Joint bad set). Let $Z : = \mathcal { G } \times \mathcal { Y } _ { n }$ carry the Euclidean product metric, and define

$$
\Sigma ^ { \circ } : = \{ ( G , y ) : G \in { \mathcal { G } } _ { \star } , \ y \in \Sigma _ { G } ^ { \circ } \} .
$$

The joint bad set and its complement are

$$
B : = \left( \left( D \cup U \right) \times \mathcal { Y } _ { n } \right) \cup \overline { { \Sigma ^ { \circ } } } ^ { Z } , \qquad \Omega : = Z \backslash B .\tag{9}
$$

Here, B contains singular and unstable graph instances and the closed joint switching graph.

Definition 5 (Selector and normalized displacement). Define d : $Z \to [ 0 , 1 ]$ by

$$
d ( G , y ) : = \left\{ { \begin{array} { l l } { \operatorname* { m i n } \{ 1 , \operatorname { d i s t } ( ( G , y ) , B ) \} , } & { B \neq \emptyset , } \\ { 1 , } & { B = \emptyset . } \end{array} } \right.
$$

For $( G , y ) \in \Omega$ , the nearest distinct solution is unique; define the branch selector P and normalized displacement H as follows:

$$
P ( G , y ) : = \operatorname * { a r g m i n } _ { s \in S ( G ) } \| y - s \| , \qquad H ( G , y ) : = \frac { P ( G , y ) - y } { d ( G , y ) } .\tag{10}
$$

First, let’s consider the invariance/equivariance properties of the above definitions.

## Lemma 1. The bad set B and its complement Ω are both permutation invariant.

Proof. Fix a node permutation $\pi .$ . Its action on $\mathcal { G }$ is an isometric bijection and therefore maps relatively open neighborhoods to relatively open neighborhoods. Definition 1 gives

$$
f _ { \tau _ { \pi } ( i ) } = \pi \circ f _ { i } \circ \pi ^ { - 1 } .
$$

Consequently, $f _ { i }$ is locally Lipschitz at $G$ if and only if $f _ { \tau _ { \pi } ( i ) }$ is locally Lipschitz at $\pi G$ . Since $\tau _ { \pi }$ is a bijection of the labels, $\pi D = D$ . If V and $\{ C _ { \alpha } \}$ give a stable collapse pattern at $G _ { 0 } .$ then $\pi V$ and $\{ \tau _ { \pi } ( C _ { \alpha } ) \}$ give one at π $\mathbf { \nabla } \cdot G _ { 0 }$ . Applying the same argument to $\pi ^ { - 1 }$ yields $\pi U = U$ In particular, $\pi \mathcal { G } _ { \star } = \mathcal { G } _ { \star }$ , and

$$
\pi ( ( D \cup U ) \times \mathcal { Y } _ { n } ) = ( D \cup U ) \times \mathcal { Y } _ { n } .
$$

Moreover, solution-set equivariance and distance preservation imply $\pi ( { \bf Z } ^ { \circ } ) = { \bf Z } ^ { \circ }$ . Indeed, let $( G , y ) \in \Sigma ^ { \circ }$ . Then there exist distinct $s _ { 1 } , s _ { 2 } \in S ( G )$ such that

$$
\| y - s _ { 1 } \| = \| y - s _ { 2 } \| = \operatorname { d i s t } ( y , S ( G ) ) .
$$

Since $S ( \pi G ) = \pi S ( G )$ and the permutation action is an isometry, $\pi s _ { 1 }$ and $\pi s _ { 2 }$ are distinct nearest elements of $S ( \pi G )$ to πy. Since $\pi G \in \mathcal G _ { \star }$ , it follows that

$$
\pi ( G , y ) = ( \pi G , \pi y ) \in \Sigma ^ { \circ } .
$$

Thus

$$
\pi ( \Sigma ^ { \circ } ) \subseteq \Sigma ^ { \circ } .
$$

Applying the same argument to $\pi ^ { - 1 }$ gives the reverse inclusion and therefore

$$
\pi ( \mathbf { \boldsymbol { \Sigma } } ^ { \circ } ) = \mathbf { \boldsymbol { \Sigma } } ^ { \circ } .
$$

Since the permutation mapping is a homeomorphism,

$$
\pi \left( { \overline { { \Sigma ^ { \circ } } } } ^ { Z } \right) = { \overline { { \pi ( \Sigma ^ { \circ } ) } } } ^ { Z } = { \overline { { \Sigma ^ { \circ } } } } ^ { Z } .
$$

Using $B = ( ( D \cup U ) \times \mathcal { Y } _ { n } ) \cup \overline { { \Sigma ^ { \circ } } } ^ { Z }$ now gives π $B = B$ . In addition,

$$
\pi \Omega = \pi ( Z \setminus B ) = \pi Z \setminus \pi B = Z \setminus B = \Omega .
$$

which shows the invariance of Ω and finishes the proof.

## Lemma 2. Function d is permutation invariant, and $P$ and H are permutation equivariant.

Proof. Since the permutation action is an isometry and $\pi B = B$ ，

$$
d ( \pi G , \pi y ) = d ( G , y ) .
$$

Let $s _ { \star } = P ( G , y )$ . By solution-set equivariance,

$$
S ( \pi G ) = \pi S ( G ) .
$$

Thus every $\widetilde s \in S ( \pi G )$ has the form $\widetilde s = \pi s$ for some $s \in S ( G )$ . Since permutations preserve distances,

$$
\| \pi y - { \widetilde { s } } \| = \| y - s \| \geq \| y - s _ { \star } \| = \left\| \pi y - \pi s _ { \star } \right\| .
$$

Hence $\pi s _ { \star }$ is a nearest solution to πy. Its uniqueness on Ω gives

$$
P ( \pi G , \pi y ) = \pi P ( G , y ) .
$$

Finally, by Definition $5 ,$

$$
{ \cal H } ( \pi G , \pi y ) = { \frac { P ( \pi G , \pi y ) - \pi y } { d ( \pi G , \pi y ) } } = \pi { \frac { P ( G , y ) - y } { d ( G , y ) } } = \pi H ( G , y ) ,
$$

which finishes the proof.

To further establish the properties of displacement H, we state and use the following lemma.

Lemma 3 ([38, Theorem $\mathrm { A . 4 l } )$ . Let X be an arbitrary subset of a Euclidean space, equipped with the induced metric, and let $h : X \to \mathbb { R }$ be locally Lipschitz intrinsically on $X ,$ . Then there exists a function $\varepsilon : X \to ( 0 , 1 )$ such that both ε and εh are globally Lipschitz on X.

Here intrinsic local Lipschitzness means Lipschitzness on a relative neighborhood $X \cap B ( x , r )$ of each $x \in X$ . The cited theorem requires no openness, closedness, boundedness, or local compactness of X, so it applies to $X = \Omega$

We first extend this result to the vector-valued and equivariant form required below. Consider a general group setting where Γ is a finite group with orthogonal representations:

$$
\rho _ { X } : \Gamma  { \mathrm O } ( p ) , \qquad \rho _ { Y } : \Gamma  { \mathrm O } ( m ) .
$$

For brevity, given $\gamma \in \Gamma , x \in \mathbb { R } ^ { p }$ , and $v \in \mathbb { R } ^ { m }$ , we write:

$$
\gamma x : = \rho _ { X } ( \gamma ) x , \qquad \gamma v : = \rho _ { Y } ( \gamma ) v .
$$

The previously defined permutation $\pi$ is an example of such an action; it applies to both G and y, acting as an orthogonal transformation on each domain.

Lemma 4 (Equivariant bounded flattening). Let $X \subseteq \mathbb { R } ^ { p }$ be Γ-invariant. Suppose that H : $X ~ \to ~ \mathbb { R } ^ { m }$ is locally Lipschitz and Γ-equivariant. Then there exists a Γ-invariant function $\beta : X  ( 0 , 1 )$ such that both $\beta$ and $\beta H$ are bounded and globally Lipschitz on X. Moreover, βH is Γ-equivariant.

Proof. Write

$$
H = ( H _ { 1 } , \dots , H _ { m } ) .
$$

For every coordinate $H _ { j }$ , apply Lemma 3 to obtain a function

$$
\varepsilon _ { j } : X \to ( 0 , 1 )
$$

such that $\varepsilon _ { j }$ and

$$
u _ { j } : = \varepsilon _ { j } H _ { j }
$$

are globally Lipschitz. Define

$$
a _ { j } ( x ) : = \frac { \varepsilon _ { j } ( x ) } { 1 + | u _ { j } ( x ) | } .
$$

The scalar map $r \mapsto ( 1 + | r | ) ^ { - 1 }$ is bounded and globally Lipschitz. Since $\varepsilon _ { j }$ is also bounded and globally Lipschitz, $a _ { j }$ is bounded and globally Lipschitz, and

$$
0 < a _ { j } < 1 .
$$

Moreover,

$$
a _ { j } H _ { j } = \frac { u _ { j } } { 1 + | u _ { j } | } .
$$

Because $r \mapsto r / ( 1 + | r | )$ is bounded and globally Lipschitz, $a _ { j } H _ { j }$ is bounded and globally Lipschitz as well. Set

$$
a _ { 0 } ( x ) : = \prod _ { j = 1 } ^ { m } a _ { j } ( x ) .
$$

A finite product of bounded globally Lipschitz functions is bounded and globally Lipschitz. For every coordinate $j ,$

$$
a _ { 0 } H _ { j } = ( a _ { j } H _ { j } ) \prod _ { k \ne j } a _ { k } ,
$$

which is again bounded and globally Lipschitz. Hence $a _ { 0 } H$ is bounded and globally Lipschitz.

Finally, define

$$
\beta ( x ) : = \prod _ { \gamma \in \Gamma } a _ { 0 } ( \gamma x ) .
$$

Since X is Γ-invariant, each $a _ { 0 } \mathrm { { } } ^ { \mathrm { { O } } \mathrm { { \gamma } } }$ is well defined on $X .$ . Since the representation $\rho _ { X }$ is orthogonal,

$$
\mathrm { L i p } ( a _ { 0 } \circ \gamma ) \leq \mathrm { L i p } ( a _ { 0 } ) , \qquad \| a _ { 0 } \circ \gamma \| _ { \infty } \leq \| a _ { 0 } \| _ { \infty } .
$$

The finiteness of Γ therefore implies that $\beta$ is bounded and globally Lipschitz.

For every $\gamma _ { 0 } \in \Gamma$

$$
\beta ( \gamma _ { 0 } x ) = \prod _ { \gamma \in \Gamma } a _ { 0 } ( \gamma \gamma _ { 0 } x ) = \prod _ { \gamma \in \Gamma } a _ { 0 } ( \gamma x ) = \beta ( x ) ,
$$

where the second equality follows because right multiplication by $\gamma _ { 0 }$ permutes the elements of Γ. Thus $\beta$ is Γ-invariant. Furthermore,

$$
\beta ( x ) H ( x ) = a _ { 0 } ( x ) H ( x ) \prod _ { \gamma \in \Gamma \atop \gamma \neq e } a _ { 0 } ( \gamma x ) ,
$$

which is a finite product of bounded globally Lipschitz factors. Thus $\beta H$ is bounded and globally Lipschitz. Moreover, the equivariance of $\beta H$ follows from the invariance of $\beta$ and equivariance of $H .$ . This proves the lemma. □

## A.3. Proof of the main result.

Proof of Theorem 4. We construct T and verify the four asserted properties.

## Step 1: The set B is closed and permutation-invariant.

For each branch, the locus on which it is locally Lipschitz is relatively open. Therefore D is relatively closed in ${ \mathcal { G } } .$ The set of points having a stable collapse pattern is also relatively open by definition, so $U$ is relatively closed. It follows that

$$
( D \cup U ) \times \mathcal y _ { n }
$$

is relatively closed in $Z .$ The set $\overline { { \mathbf { \Sigma } \Sigma ^ { \circ } } } ^ { Z }$ is closed by definition. Hence B is closed in $Z .$

In addition, Lemma 1 provides the permutation-invariance of the bad set B and its complement.

## Step 2: The nearest-solution selector is locally Lipschitz on $Z \backslash B ,$

Fix $( G _ { 0 } , y _ { 0 } ) \in \Omega$ . Then $G _ { 0 } \not \in D \cup U$ , the collapse pattern is stable near $G _ { 0 } .$ , and $y _ { 0 }$ has a unique nearest distinct branch value. Let $C _ { \star }$ be the stable cluster realizing this nearest value, and choose any representative $i _ { \star } ~ \in ~ C _ { \star }$ . Branches in $C _ { \star }$ agree identically throughout some neighborhood of $G _ { 0 }$

If $C _ { \star }$ is the only stable cluster, then all branch values agree throughout that neighborhood, and hence

$$
P ( G , y ) = f _ { i _ { \star } } ( G )
$$

there; the local Lipschitz conclusion follows immediately.

Otherwise, choose one representative $i _ { \alpha }$ from every other stable cluster. Since the nearest branch value at $( G _ { 0 } , y _ { 0 } )$ is unique, the margin

$$
\delta : = \operatorname* { m i n } _ { \alpha \neq \star } \left( \| y _ { 0 } - f _ { i _ { \alpha } } ( G _ { 0 } ) \| - \| y _ { 0 } - f _ { i _ { \star } } ( G _ { 0 } ) \| \right)
$$

is strictly positive. All regular branches are continuous near $G _ { 0 }$ . After shrinking the neighborhood of $( G _ { 0 } , y _ { 0 } )$ if necessary, $C _ { \star }$ remains the unique nearest cluster throughout that neighborhood. Hence

$$
P ( G , y ) = f _ { i _ { \star } } ( G )
$$

there. Because $G _ { 0 } \notin D$ , the representative branch $f _ { i , \astrosun }$ is locally Lipschitz near $G _ { 0 }$ . Thus $P$ is jointly locally Lipschitz near $( G _ { 0 } , y _ { 0 } )$ . Since $( G _ { 0 } , y _ { 0 } )$ was arbitrary,

$$
P : \Omega  \mathcal { V } _ { n }
$$

is jointly locally Lipschitz. This is the same local mechanism used in [30, Lemma A.12], but no boundedness or compactness of the graph domain is required here.

## Step 3: Construction of a damping factor that removes the singularities.

Since B is relatively closed in $Z ,$ for every $z \in Z \backslash B$ we have dist $\left( z , B \right) > 0 ;$ otherwise $z \in \overline { { B } } ^ { Z } = B$ . Hence it holds that

$$
d ( z ) > 0 \quad { \mathrm { f o r ~ } } z \in \Omega , \qquad d ( z ) = 0 \quad { \mathrm { f o r ~ } } z \in B .
$$

Moreover, $d$ is bounded, 1-Lipschitz, and permutation-invariant. Since d is 1-Lipschitz and strictly positive on Ω, it is locally bounded away from zero there. Hence $d ^ { - 1 }$ is locally Lipschitz on Ω. Together with the local Lipschitzness of $P ,$ this implies that $H = ( P - y ) / d$ is locally Lipschitz. In addition, thanks to Lemma 2, H is permutation equivariant.

Apply Lemma 4 on Ω and permutation action group. There exists a permutation-invariant function

$$
\beta : \Omega \to ( 0 , 1 )
$$

such that

$$
A : = \beta H
$$

is bounded and globally Lipschitz on $\Omega .$ . Define

$$
q ( G , y ) : = \left\{ { \begin{array} { l l } { d ( G , y ) A ( G , y ) , } & { ( G , y ) \in \Omega , } \\ { 0 , } & { ( G , y ) \in B . } \end{array} } \right.
$$

For $( G , y ) \in \Omega ,$

$$
q ( G , y ) = \beta ( G , y ) \bigl ( P ( G , y ) - y \bigr ) .
$$

Finally, define

$$
T _ { G } ( y ) : = y + q ( G , y ) .
$$

Equivalently,

$$
T _ { G } ( y ) = \left\{ \begin{array} { l l } { ( 1 - \beta ( G , y ) ) y + \beta ( G , y ) P _ { G } ( y ) , } & { ( G , y ) \notin B , } \\ { y , } & { ( G , y ) \in B . } \end{array} \right.\tag{11}
$$

The freezing property on B is immediate from this definition.

Step 4: Joint global Lipschitz continuity of $T .$

Let

$$
\| A \| _ { \infty } \leq M _ { A } , \qquad \mathrm { L i p } ( A ) \leq L _ { A } .
$$

For any $z , z ^ { \prime } \in \Omega$

$$
\begin{array} { r l } & { \| q ( \boldsymbol { z } ) - q ( \boldsymbol { z } ^ { \prime } ) \| = \| d ( \boldsymbol { z } ) A ( \boldsymbol { z } ) - d ( \boldsymbol { z } ^ { \prime } ) A ( \boldsymbol { z } ^ { \prime } ) \| } \\ & { \qquad \leq | d ( \boldsymbol { z } ) - d ( \boldsymbol { z } ^ { \prime } ) | \| A ( \boldsymbol { z } ) \| + d ( \boldsymbol { z } ^ { \prime } ) \| A ( \boldsymbol { z } ) - A ( \boldsymbol { z } ^ { \prime } ) \| } \\ & { \qquad \leq ( M _ { A } + L _ { A } ) \| \boldsymbol { z } - \boldsymbol { z } ^ { \prime } \| , } \end{array}
$$

where we used $0 \leq d \leq 1$ (This follows from Definition 5) and the 1-Lipschitz property of $d .$

I $\therefore z \in \Omega$ and $z ^ { \prime } \in B$ , then

$$
\begin{array} { r } { \| q ( z ) - q ( z ^ { \prime } ) \| = \| q ( z ) \| = d ( z ) \| A ( z ) \| \leq M _ { A } \operatorname { d i s t } ( z , B ) \leq M _ { A } \| z - z ^ { \prime } \| . } \end{array}
$$

If both points lie in B, the diference is zero. Thus $q$ is globally Lipschitz on all of $Z .$ . Since the coordinate projection $( G , y ) \mapsto y$ is 1-Lipschitz and

$$
T _ { G } ( y ) = y + q ( G , y ) ,
$$

$T$ is jointly globally Lipschitz. In particular, one may take

$$
L _ { T } \leq 1 + M _ { A } + L _ { A } . 
$$

## Step 5: Permutation equivariance.

The functions d and $\beta$ are invariant, whereas H is equivariant. Therefore

$$
A ( \pi G , \pi y ) = \pi A ( G , y )
$$

on Ω. Since B is invariant, it follows that

$$
q ( \pi G , \pi y ) = \pi q ( G , y )
$$

on all of $Z .$ Consequently,

$$
T _ { \pi G } ( \pi y ) = \pi y + q ( \pi G , \pi y ) = \pi y + \pi q ( G , y ) = \pi T _ { G } ( y ) .
$$

Step 6: On a regular stable graph, taking the closure does not enlarge the switching section.

Fix $G \in \mathcal G _ { \star }$ . We claim that

$$
\{ y : ( G , y ) \in B \} = \Sigma _ { G } ^ { \circ } .\tag{12}
$$

The inclusion $\ ` 2 '$ follows immediately from the definition of the closure. For the reverse inclusion, it sufices to show that any point in the G-section of $\overline { { \pmb { \Sigma } ^ { \circ } } } ^ { Z }$ belongs to $\Sigma _ { G } ^ { \circ }$ . Suppose that

$$
( G _ { k } , y _ { k } ) \in \Sigma ^ { \circ } , \qquad ( G _ { k } , y _ { k } ) \to ( G , y ) .
$$

For every $k ,$ choose two labels whose distinct branch values attain the common minimum distance from $y _ { k }$ . Because there are only finitely many pairs of labels, after passing to a subsequence we may assume that these labels are fixed, say $i \neq j$

Since G is a stable-collapse point, if i and $j$ belonged to the same stable cluster, then the two branches would agree identically throughout a neighborhood of $G .$ For all suficiently large $k ,$ this would contradict the fact that they represent distinct branch values at $G _ { k }$ . Therefore i and $j$ belong to diferent stable clusters, and in particular

$$
f _ { i } ( G ) \neq f _ { j } ( G ) .
$$

Because $G \notin D$ , every branch is continuous at $G .$ . Passing to the limit in the equal-minimum relations gives

$$
\| y - f _ { i } ( G ) \| = \| y - f _ { j } ( G ) \| = \operatorname* { m i n } _ { \ell } \| y - f _ { \ell } ( G ) \| .
$$

Thus $y \in \Sigma _ { G } ^ { \circ }$ , proving the reverse inclusion.

## Step 7: Invariance of the strict Voronoi segment.

Fix

$$
{ \cal G } \in { \mathcal G } _ { \star } , \qquad y _ { 0 } \notin \Sigma _ { \cal G } ^ { \circ } ,
$$

and let

$$
s = P _ { G } ( y _ { 0 } ) .
$$

For any other distinct solution $r \in S ( G ) \setminus \{ s \}$

$$
\lVert y _ { 0 } - s \rVert < \lVert y _ { 0 } - r \rVert .
$$

For $a \in [ 0 , 1 ]$ , define

$$
z _ { a } = ( 1 - a ) y _ { 0 } + a s .
$$

Then

$$
\lVert z _ { a } - s \rVert = ( 1 - a ) \lVert y _ { 0 } - s \rVert .
$$

On the other hand, the reverse triangle inequality yields

$$
\begin{array} { r l } & { \| z _ { a } - r \| \geq \| y _ { 0 } - r \| - \| z _ { a } - y _ { 0 } \| } \\ & { \qquad = \| y _ { 0 } - r \| - a \| y _ { 0 } - s \| } \\ & { \qquad > ( 1 - a ) \| y _ { 0 } - s \| } \\ & { \qquad = \| z _ { a } - s \| . } \end{array}
$$

Therefore

for every $a \in [ 0 , 1 ]$ . Hence

$$
P _ { G } ( z _ { a } ) = s \qquad \mathrm { a n d } \qquad z _ { a } \not \in \Sigma _ { G } ^ { \circ }
$$

$$
P _ { G } ( z ) = s \qquad \mathrm { f o r ~ e v e r y ~ } z \in [ y _ { 0 } , s ] ,
$$

and Step 6 implies

$$
\{ G \} \times [ y _ { 0 } , s ] \subseteq \Omega .
$$

Step 8: Convergence.

The set $\{ G \} \times [ y _ { 0 } , s ]$ is a compact subset of Ω. Since $\beta$ is continuous and strictly positive on

$$
\Omega ,
$$

$$
c ( G , y _ { 0 } ) : = \operatorname* { m i n } _ { z \in [ y _ { 0 } , s ] } \beta ( G , z ) > 0 .
$$

Furthermore, $0 < \beta < 1$ . If $y _ { t } \in [ y _ { 0 } , s ]$ , then Step $7$ gives $P _ { G } ( y _ { t } ) = s ,$ and therefore

$$
y _ { t + 1 } = ( 1 - \beta ( G , y _ { t } ) ) y _ { t } + \beta ( G , y _ { t } ) s \in [ y _ { t } , s ] \subseteq [ y _ { 0 } , s ] .
$$

By induction, every iterate remains on $[ y _ { 0 } , s ] .$ . Moreover,

$$
y _ { t + 1 } - s = \bigl ( 1 - \beta ( G , y _ { t } ) \bigr ) ( y _ { t } - s ) .\tag{13}
$$

It follows that

$$
\left\| y _ { t + 1 } - s \right\| \leq \left( 1 - c ( G , y _ { 0 } ) \right) \left\| y _ { t } - s \right\| .
$$

Iterating this estimate gives

$$
\| y _ { t } - s \| \leq \big ( 1 - c ( G , y _ { 0 } ) \big ) ^ { t } \| y _ { 0 } - s \| .
$$

Hence $y _ { t } \to s .$

Step 9: Almost-everywhere convergence and full support.

Fix $G \in \mathcal G _ { \star }$ . For any two distinct solutions $s , r \in S ( G )$ , the equidistance set

$$
\{ y : \| y - s \| = \| y - r \| \}
$$

is an afine hyperplane, because the defining equality is equivalent to

$$
2 \langle y , r - s \rangle = \| r \| ^ { 2 } - \| s \| ^ { 2 } .
$$

Therefore

$$
\Sigma _ { G } ^ { \circ } \subseteq \bigcup _ { \stackrel { s , r \in S ( G ) } { s \neq r } } \{ y : \| y - s \| = \| y - r \| \}
$$

is contained in a finite union of Lebesgue-null afine hyperplanes. Thus

$$
\begin{array} { r } { \mathcal { L } ^ { m } ( \Sigma _ { G } ^ { \circ } ) = 0 . } \end{array}
$$

If $\mu \ll \mathcal { L } ^ { m }$ , then

$$
\mu ( \Sigma _ { G } ^ { \circ } ) = 0 .
$$

Step 8 therefore shows that the iteration converges to a point of $S ( G )$ for µ-almost every initialization.

Finally, if every strict Voronoi basin has positive µ-mass, then for every $s \in S ( G )$

$$
\operatorname* { P r } ( y _ { \infty } = s \mid G ) = \mu ( V _ { s } ( G ) ) > 0 .
$$

Since $S ( G )$ is finite and $y _ { \infty } \in S ( G )$ almost surely,

$$
\operatorname { s u p p } { \mathcal { L } } ( y _ { \infty } \mid G ) = S ( G ) .
$$

This proves all the conclusions.

## Appendix B. Family-wide asymptotic realization by a recurrent MP-GNN

We now realize the ideal dynamics of Appendix A using one message-passing GNN block that is reused at every recurrent step:

$$
y _ { t + 1 } = \operatorname { G N N } ( G , y _ { t } ) .\tag{14}
$$

We suppress the parameter subscript θ in this appendix. The random initialization supplies both the initial node addresses and the choice of solution basin. At later steps, the block receives only the graph and the current state. Throughout this appendix, write $d _ { y } : = q \ge 2$ and $m = n d _ { y }$ , using the graph and output spaces of Appendix A.

B.1. Architecture and main result. The recurrent block. For one call, write $y ^ { + } =$ $\operatorname { G N N } ( G , y )$ . Choose a finite number of hidden layers $L ,$ finite hidden widths, and a continuous encoder

$$
\mathrm { e n c : } \mathbb { R } ^ { d _ { v } } \times \mathbb { R } ^ { d _ { y } } \longrightarrow \mathbb { R } ^ { d _ { 0 } } .
$$

The block has the form

$$
\begin{array} { r l r } {  { h _ { i } ^ { ( 0 ) } = \mathrm { e n c } ( X _ { i } , y _ { i } ) , } } \\ & { } & { ~ } \\ & { m _ { i } ^ { ( \ell ) } = \displaystyle \sum _ { j = 1 } ^ { n } M _ { \ell } \Big ( h _ { i } ^ { ( \ell ) } , h _ { j } ^ { ( \ell ) } , A _ { j i } \Big ) , } \\ & { } & { ~ } \\ & { } & { h _ { i } ^ { ( \ell + 1 ) } = U _ { \ell } \Big ( h _ { i } ^ { ( \ell ) } , m _ { i } ^ { ( \ell ) } \Big ) , \qquad 0 \le \ell < L , } \\ & { } & { ~ } \\ & { } & { g ^ { ( L ) } = \displaystyle \sum _ { j = 1 } ^ { n } R _ { L } \Big ( h _ { j } ^ { ( L ) } \Big ) , } \\ & { } & { ~ } \\ & { } & { y _ { i } ^ { + } = O \Big ( h _ { i } ^ { ( L ) } , g ^ { ( L ) } \Big ) . } \end{array}\tag{15}
$$

Here the aggregation runs over all n senders, including $j = i ,$ , since the edge tensor A assigns a feature vector to every ordered pair. The component maps, $\{ M _ { \ell } , U _ { \ell } \} _ { \ell = 0 } ^ { \overline { { { L } } } - 1 } , R _ { L } , O _ { \ell }$ , are arbitrary continuous maps between finite-dimensional Euclidean spaces and are shared across nodes. The only global operation is the final invariant sum: no global aggregate is formed or broadcast in the hidden layers. The entire L-layer block, including its encoder and readout, is denoted as $\operatorname { G N N } ( G , \cdot )$ and is reused with the same parameters at every recurrent time. Hidden representations are recomputed within each call; only y<sub>t</sub> is carried to the next call. Remark (Sparse graphs). For an edge set $E ( G ) \subseteq [ n ] \times [ n ]$ , encode $A _ { j i } = \left( e _ { j i } , a _ { j i } \right)$ , where $e _ { j i } = \mathbf { 1 } _ { \left\{ ( j , i ) \in E ( G ) \right\} }$ and $a _ { j i } = 0$ whenever $e _ { j i } = 0$ . The class of such tensors is closed under relabeling, so Appendix A applies to any permutation-invariant domain of these graphs. The block constructed in Lemma 5 has $M _ { 0 } = c _ { i } \otimes c _ { j } \otimes A _ { j i }$ , which vanishes on every absent pair, even outside the compact realization family. Hence its all-pairs message sum agrees exactly with the neighbor sum over $( j , i ) \in E ( G )$ . On the compact realization family, $\mathcal { R } _ { \mathrm { e d g e } }$ records $( 1 , a _ { j i } )$ in present-edge slots and zero in absent-edge slots. Consequently, the block in Theorem 5 can use neighbor-only hidden message passing followed by the readout in (15).

It’s straightforward that reindexing the message and readout sums gives the equivariance:

$$
\mathrm { G N N } ( \pi G , \pi y ) = \pi \mathrm { G N N } ( G , y ) .\tag{16}
$$

Facts about the ideal update. Fix the operator $T$ constructed in Theorem 4. Let Ω and P be as in Definitions 4 and 5. The proof in Appendix A gives

$$
T _ { G } ( y ) = ( 1 - \beta ( G , y ) ) y + \beta ( G , y ) P ( G , y ) , \qquad 0 < \beta ( G , y ) < 1 , \qquad ( G , y ) \in \Omega ,\tag{17}
$$

where β is continuous. The selector is continuous on Ω and is constant along the segment from y to $P ( G , y )$ . We will also use the bound

$$
\| T _ { G } ( y ) - y \| \leq \mathrm { d i s t } ( y , S ( G ) ) \qquad ( G \in \mathcal { G } _ { \star } , ~ y \in \mathcal { V } _ { n } ) .\tag{18}
$$

Indeed, of the switching set this follows from (17); on the switching set $T _ { G } ( y ) = y$

Theorem 5 (Family-wide asymptotic realization from random initialization). Assume $d _ { y } \geq 2$ Let $( G , Y _ { 0 } )$ have a tight Borel probability law P on $\mathcal G \times \mathcal { V } _ { n }$ , and let $\mu _ { G }$ be a regular conditional distribution of $Y _ { 0 }$ given G. Such a Borel kernel exists by disintegrating the joint law on the ambient Euclidean product and restricting the graph variable to $\mathcal { G } .$ . Assume:

(1) $G \in { \mathcal { G } } _ { \star }$ almost surely;

(2) conditionally on $G _ { i }$ , the law µ<sub>G</sub> of $Y _ { 0 }$ satisfies $\mu _ { G } \ll \mathcal { L } ^ { m }$ almost surely;

(3) for almost every G and every $s \in S ( G )$

$$
\mu _ { G } ( V _ { s } ( G ) ) > 0 .
$$

All solution-dependent probability events are understood on the full-measure set $G \in \mathcal G _ { \star }$ . Fix a limit tolerance $\varepsilon > 0$ , an operator tolerance $\rho > 0$ , and a probability loss $\delta \in ( 0 , 1 )$ . Then there exist a compact permutation-invariant event $E _ { \varepsilon , \rho , \delta } \subseteq \mathcal G \times \mathcal y _ { n }$ with

$$
\mathbb { P } ( E _ { \varepsilon , \rho , \delta } ) \geq 1 - \delta\tag{19}
$$

and one deterministic, autonomous, weight-tied MP-GNN block such that, for every $( G , y _ { 0 } ) \in$ $E _ { \varepsilon , \rho , \delta ; }$ , the trajectory

$$
{ \widehat { y } } _ { 0 } = y _ { 0 } , \qquad { \widehat { y } } _ { t + 1 } = \operatorname { G N N } ( G , { \widehat { y } } _ { t } ) , \qquad t \geq 0 ,
$$

converges to a limit $\widehat { y } _ { \infty }$ satisfying

$$
\begin{array} { r } { \mathrm { d i s t } ( \widehat { y } _ { \infty } , S ( G ) ) < \varepsilon , } \\ { \displaystyle \mathrm { s u p } _ { t \geq 0 } \| \mathrm { G N N } ( G , \widehat { y } _ { t } ) - T _ { G } ( \widehat { y } _ { t } ) \| < \rho . } \end{array}\tag{20}
$$

The block receives only $( G , \widehat { y } _ { t } )$ at every recurrent time.

The graph projection

$$
K _ { \varepsilon , \rho , \delta } : = \mathrm { p r o j } _ { \mathcal { G } } E _ { \varepsilon , \rho , \delta }
$$

is compact, and the same fixed collection of continuous component maps works for every graph in this projection on its corresponding retained initialization fiber. The block is permutation equivariant as in (16). In particular,

$$
\begin{array} { r } { \mathbb { P } ( \widehat { y } _ { t } \ c o n v e r g e s \ a n d \ \mathrm { d i s t } ( \widehat { y } _ { \infty } , S ( G ) ) < \varepsilon ) \geq 1 - \delta . } \end{array}
$$

Moreover, the good event may be chosen so that every labeled branch has positive joint capture probability:

$$
\begin{array} { r } { ( 2 1 ) \quad \mathbb { P } \big ( Y _ { 0 } \in V _ { f _ { a } ( G ) } ( G ) , \quad \widehat { y } _ { t } \mathrm { ~ } c o n v e r g e s ~ t o ~ \widehat { y } _ { \infty } , \quad \| \widehat { y } _ { \infty } - f _ { a } ( G ) \| < \varepsilon \big ) > 0 , \qquad a = 1 , \dots , M . } \end{array}
$$

Coincident labels refer to the same distinct target. The learned limit is required to be ε-close to the solution set; it need not belong exactly to $S ( G )$

The probability law need not be permutation invariant. Equivariance is a property of the block, and the retained event is made invariant by its construction. The branch-capture guarantee is joint over graphs and initializations; it does not assert positive retained mass in every basin of every individual graph.

B.2. Exact realization using the current node states. In this subsection, we establish conditions under which a continuous equivariant operator can be exactly realized by a messagepassing GNN in our setting. We begin with the relevant symmetry-breaking condition. Without suitable symmetry-breaking features, standard message-passing GNNs may fail to distinguish graph inputs that the target operator distinguishes, limiting their ability to represent arbitrary continuous equivariant operators (see [66]). We therefore introduce a condition under which the current node states provide distinct node addresses, and prove that these addresses enable exact realization on a compact family. This result provides the basis for constructing a messagepassing GNN that follows the ideal dynamics introduced in Appendix A.

Definition 6 (Robust state-based node symmetry breaking). Let $K \subseteq \mathcal G \times \mathcal y _ { n }$ be compact and permutation invariant. We say that K admits robust state-based node symmetry breaking if there exist finitely many compact address regions $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { J } \subseteq \mathbb { R } ^ { d _ { y } }$ and $\eta > 0$ such that

$$
\mathrm { d i s t } ( A _ { j } , A _ { k } ) \geq \eta \qquad ( j \neq k ) ,
$$

and, for every $( G , y ) \in K$ , the n node arguments $y _ { 1 } , \ldots , y _ { n }$ belong to n diferent address regions.

Because there are finitely many compact address regions, pairwise disjointness already guarantees a positive separation margin; η simply denotes a common margin. The regions $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { J }$ and their separation margin η are fixed for the whole family $K$ , while the assignment of nodes to regions may vary between inputs. Since the node states $y _ { 1 } , \ldots , y _ { n }$ belong to diferent address regions, each $y _ { i }$ serves as a unique node identifier within the current input $( G , y )$ .

For example, take $d _ { y } = 2$ and a fixed two-node graph $G _ { 0 }$ that is unchanged by swapping its nodes. Write $u _ { \theta } = \left( \cos \theta _ { \mathrm { : } } \right.$ , sin θ) and define two opposite arcs:

$$
\begin{array} { r } { A _ { 1 } = \{ u _ { \theta } : \theta \in [ - \pi / 4 , \pi / 4 ] \} , \qquad A _ { 2 } = - A _ { 1 } . } \end{array}
$$

These compact regions lie on the right and left sides of the unit circle, respectively, with dist $( \mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } ) = \sqrt { 2 }$ . The compact permutation-invariant family

$$
K = \left\{ ( G _ { 0 } , ( u , - u ) ) , \ ( G _ { 0 } , ( - u , u ) ) : u \in \mathcal { A } _ { 1 } \right\}
$$

therefore satisfies Definition $6 \ ( \mathrm { i . e . }$ , admits robust state-based node symmetry breaking) with $J = 2$ and $\eta = { \sqrt { 2 } } { \mathrm { : } }$ one node lies in each region, although their assignments can be swapped.

In contrast, the following full-circle family does not satisfy Definition 6

$$
K _ { \mathrm { c i r c l e } } = \left\{ \left( G _ { 0 } , ( u _ { \theta } , - u _ { \theta } ) \right) : \theta \in \left[ 0 , 2 \pi \right] \right\} .
$$

The two node states are still always distance 2 apart, but each now ranges over the entire circle. There is no finite collection of compact regions that simultaneously (i) are positively separated, (ii) cover the entire unit circle, and (iii) place $u _ { \theta }$ and $- u _ { \theta }$ in diferent regions for every θ. Indeed, because the circle is connected, conditions (i) and (ii) force it to lie entirely in one region, contradicting (iii).

In the remainder of this subsection, we show how continuous equivariant operators can be exactly realized by message-passing GNNs under robust state-based node symmetry breaking. Subsections B.3 and B.4 then establish this condition for the compact families used in our construction.

Lemma 5 (Exact MP-GNN realization on a compact family). Let $K \subseteq \mathcal G \times \mathcal y _ { n }$ be compact, permutation invariant, and admit robust state-based node symmetry breaking. Fix integers $u , v \geq 0$ with $u + v \geq 1$ and, when $v > 0$ , positive integers $p _ { 1 } , \ldots , p _ { v }$ . For $\ell = 1 , \ldots , u _ { i }$ let $\varphi _ { \ell } : K  \mathbb { R }$ be continuous and permutation invariant. For $b = 1 , \dots , v _ { ; }$ , let $F _ { b } : K  \mathbb { R } ^ { n \times p _ { b } }$ be continuous and permutation equivariant. Then there exists a $M P  – G N N ,$ with output width $\textstyle u + \sum _ { b = 1 } ^ { v } p _ { b }$ and the structural form (15) realizes all these maps simultaneously and exactly on K: its output row at node i is

$$
\tau ( G , y , i ) : = \Big ( \varphi _ { 1 } ( G , y ) , \ldots , \varphi _ { u } ( G , y ) , [ F _ { 1 } ( G , y ) ] _ { i } , \ldots , [ F _ { v } ( G , y ) ] _ { i } \Big ) .
$$

Empty output blocks are omitted. The component maps and all widths are chosen once for the whole compact family.

Proof. We construct one message-passing layer that records the graph and state in node-code order, and then define a continuous readout on that record.

Step 1: Construct the message-passing representation. Fix one collection of compact address regions $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { J }$ satisfying Definition 6 simultaneously for all $( G , y ) \in K$ These regions remain fixed throughout the construction; only the region containing each node state may vary with the input. The map that equals the jth standard basis vector $\boldsymbol { e } _ { j } \in \mathbb { R } ^ { J }$ on $A _ { j }$ is continuous on their closed union. The Tietze extension theorem [44, Theorem $3 5 . 1 ( \mathrm { b } ) ]$ , applied coordinatewise, gives a continuous map

$$
c : \mathbb { R } ^ { d _ { y } } \longrightarrow \mathbb { R } ^ { J } , \qquad c ( w ) = e _ { j } \quad ( w \in \mathcal { A } _ { j } ) .
$$

For $( G , y ) \in K$ , the codes $c _ { i } : = c ( y _ { i } )$ are distinct standard basis vectors. Set $d _ { 0 } = d _ { v } + d _ { y } + J$ and define

$$
\operatorname { e n c } ( X _ { i } , y _ { i } ) : = ( X _ { i } , y _ { i } , c ( y _ { i } ) ) .
$$

For the single local layer, take

$$
\begin{array} { r l } & { M _ { 0 } \Bigl ( h _ { i } ^ { ( 0 ) } , h _ { j } ^ { ( 0 ) } , A _ { j i } \Bigr ) : = c _ { i } \otimes c _ { j } \otimes A _ { j i } , } \\ & { \qquad U _ { 0 } \Bigl ( h _ { i } ^ { ( 0 ) } , m _ { i } ^ { ( 0 ) } \Bigr ) : = \bigl ( c _ { i } , \ c _ { i } \otimes ( 1 , X _ { i } , y _ { i } ) , \ m _ { i } ^ { ( 0 ) } \bigr ) . } \end{array}
$$

Thus $h _ { i } ^ { ( 1 ) } = \left( c _ { i } , N _ { i } , E _ { i } \right)$ , where

$$
N _ { i } : = c _ { i } \otimes ( 1 , X _ { i } , y _ { i } ) , \qquad E _ { i } : = \sum _ { j = 1 } ^ { n } c _ { i } \otimes c _ { j } \otimes A _ { j i } .
$$

Set $R _ { 1 } ( c _ { i } , N _ { i } , E _ { i } ) : = ( N _ { i } , E _ { i } )$ . The final global sum is

$$
g ^ { ( 1 ) } = ( \mathcal { R } _ { \mathrm { n o d e } } , \mathcal { R } _ { \mathrm { e d g e } } ) , \qquad \mathcal { R } _ { \mathrm { n o d e } } : = \sum _ { i } c _ { i } \otimes ( 1 , X _ { i } , y _ { i } ) , \qquad \mathcal { R } _ { \mathrm { e d g e } } : = \sum _ { i , j } c _ { i } \otimes c _ { j } \otimes A _ { j i } .
$$

All tensors are vectorized before concatenation. The displayed component maps are continuous and shared across nodes.

Step 2: Recover the graph up to relabeling. For a standard basis vector $e _ { k } .$ , the tensor $e _ { k }$ ⊗z places z in its kth block and zero in the remaining blocks. Hence $\mathcal { R } _ { \mathrm { n o d e } }$ records $( 1 , X _ { i } , y _ { i } )$ in the slot indexed by $c _ { i } ,$ and $\mathcal { R } _ { \mathrm { e d g e } }$ records $A _ { j i }$ in the slot indexed by the ordered pair $( c _ { i } , c _ { j } )$

Note that, the roles of the constant-one coordinates are therefore as follows: The 1 in $( 1 , X _ { i } , y _ { i } )$ is a node-occupancy indicator. Without it, a node satisfying $X _ { i } ~ = ~ 0$ and $y _ { i } = 0$ would produce a zero block, indistinguishable from an unused address slot. This ensures that the global tensors reconstruct nodes and edges even when their features are zero; they do not carry solution information or participate in symmetry breaking.

Suppose $( G , y )$ and $( G ^ { \prime } , y ^ { \prime } )$ produce the same two global tensors. Both inputs have the same number n of nodes, each occupying a distinct address slot marked by the constant-one coordinate. Matching these occupied slots therefore defines a unique permutation $\pi \in { \mathfrak { S } } _ { n }$ such that

$$
c _ { \pi ( i ) } ^ { \prime } = c _ { i } , \qquad i \in [ n ] .
$$

Equality of the corresponding blocks then implies

$$
X _ { \pi ( i ) } ^ { \prime } = X _ { i } , \qquad y _ { \pi ( i ) } ^ { \prime } = y _ { i } , \qquad A _ { \pi ( j ) , \pi ( i ) } ^ { \prime } = A _ { j i } .
$$

Thus $( G ^ { \prime } , y ^ { \prime } ) = ( \pi G , \pi y )$ . If the receiver codes also agree, $c _ { i ^ { \prime } } ^ { \prime } = c _ { i }$ , their distinctness gives $i ^ { \prime } = \pi ( i )$ . Therefore

$$
\Gamma ( G , y , i ) : = ( c _ { i } , \mathcal { R } _ { \mathrm { n o d e } } , \mathcal { R } _ { \mathrm { e d g e } } )
$$

determines the input and its receiving node up to simultaneous relabeling.

Step 3: The target is determined by this representation. $\mathrm { I f } \ \Gamma ( G , y , i ) = \Gamma ( G ^ { \prime } , y ^ { \prime } , i ^ { \prime } )$ Step 2 supplies a permutation matching both inputs and receivers. Invariance of each $\varphi _ { \ell }$ and equivariance of each $F _ { b }$ imply

$$
\tau ( G , y , i ) = \tau ( G ^ { \prime } , y ^ { \prime } , i ^ { \prime } ) .
$$

Consequently, the prescription

$$
D _ { 0 } \bigl ( \Gamma ( G , y , i ) \bigr ) : = \tau ( G , y , i )
$$

defines a single-valued map on the reconstruction image $\mathcal { T } : = \Gamma ( K \times [ n ] )$

Step 4: Construct a continuous output map. Both Γ and τ are continuous on the compact space $K \times \lceil n \rceil$ , where [n] has the discrete topology. The induced map $D _ { 0 }$ is continuous. For completeness, if $w _ { k } \to u$ in $\mathcal { T } _ { : }$ choose preimages $z _ { k } \in K \times [ n ]$ . Every subsequence of these preimages has a further convergent subsequence, say $z _ { k _ { j } }  z$ . Continuity gives $\Gamma ( z ) = w$ and

$$
D _ { 0 } ( w _ { k _ { j } } ) = \tau ( z _ { k _ { j } } ) \longrightarrow \tau ( z ) = D _ { 0 } ( w ) .
$$

If $D _ { 0 } ( w _ { k } )$ did not converge to $D _ { 0 } ( w )$ , a subsequence staying a positive distance away would contradict this argument. Thus $D _ { 0 }$ is continuous. This is the compact-domain quotient argument in explicit form.

The image I is compact and hence closed in its Euclidean ambient space. Coordinatewise Tietze extension gives a continuous ambient map $\widetilde { D }$ agreeing with $D _ { 0 }$ on $\mathcal { Z } .$ . Define

$$
O \big ( ( c _ { i } , N _ { i } , E _ { i } ) , g ^ { ( 1 ) } \big ) : = \widetilde D ( c _ { i } , \mathcal { R } _ { \mathrm { n o d e } } , \mathcal { R } _ { \mathrm { e d g e } } ) .
$$

On $K ,$ this output equals $\tau ( G , y , i )$ , as required.

Lemma 5 shows that, once robust node-state symmetry breaking is available, any prescribed finite collection of continuous permutation-invariant scalar maps and continuous permutationequivariant node maps can be realized simultaneously and exactly on K. This proof difers from the Stone–Weierstrass density arguments commonly used in GNN universality theory, e.g., [33, 4, 21]. It constructs a finite-dimensional representation $( c _ { i } , \mathcal { R } _ { \mathrm { n o d e } } , \mathcal { R } _ { \mathrm { e d g e } } )$ that reconstructs $( G , y , i )$ up to simultaneous node relabeling. The target therefore factors continuously through this representation, and the Tietze extension theorem provides an ambient continuous readout. Because the present MP-GNN class allows arbitrary continuous component maps, this construction yields exact realization. By contrast, Stone–Weierstrass uses a function algebra that contains the constants and separates the relevant points or equivalence classes to establish density in in the relevant continuous-function space, yielding uniform approximation rather than exact equality.

B.3. Node separation and a common collection of boxes. To apply Lemma $5 ,$ we need the node states to provide robust addresses (symmetry breaking). The next two lemmas establish this condition: the first derives node separation from random initialization, and the second places the separated states in common address boxes, allowing arbitrarily small probability losses.

Lemma 6 (Finite-time node separation for the ideal dynamics). Assume $d _ { y } \geq 2 , G \in { \mathcal { G } } ,$ almost surely, and $\mu _ { G } \ll \mathcal { L } ^ { m }$ almost surely. Put $Y _ { t } : = T _ { G } ^ { t } ( Y _ { 0 } )$ . Then

(22)

$\mathbb { P } ( ( Y _ { t } ) _ { i } \neq ( Y _ { t } ) _ { j }$ for all $t \in  { \mathbb { N } } _ { 0 }$ and $i \neq j \vert G ) = 1$ for almost every $G .$

In particular, this event has joint probability one.

Let $E \subseteq \mathcal G _ { \star } \times \mathcal y _ { n }$ be compact and permutation invariant, let $H \in  { \mathbb { N } } _ { 0 }$ , and let $\gamma > 0$ . There exist $\eta > 0$ and a compact invariant $E ^ { \prime } \subseteq E$ such that

$$
\begin{array} { r } { \mathbb { P } ( E \setminus E ^ { \prime } ) < \gamma , } \\ { \| [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { i } - [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { j } \| \ge \eta \quad ( ( G , y _ { 0 } ) \in E ^ { \prime } , \ 0 \le t \le H , \ i \ne j ) . } \end{array}
$$

If finitely many measurable subsets $C _ { a } \subseteq E$ have positive probability, $E ^ { \prime }$ may also be chosen with $\mathbb { P } ( C _ { a } \cap E ^ { \prime } ) > 0$ for every a.

Proof. For $n = 1$ there are no node pairs; take $E ^ { \prime } = E$ and any $\eta > 0$ . Assume $n \geq 2 .$ and fix a graph $G \in { \mathcal { G } } $ for which the conditional initialization law is absolutely continuous. The key observation is that a collision forces an initial node state diference onto a fixed line determined by a solution. In two or more dimensions, such an event has zero measure.

For $i < j$ , the node-state-diference map

$$
D _ { i j } : \mathcal { V } _ { n }  \mathbb { R } ^ { d _ { y } } , \qquad D _ { i j } ( y ) = y _ { i } - y _ { j } ,
$$

is linear and surjective. For any $s \in S ( G )$ , define

$$
N _ { s , i j } : = D _ { i j } ^ { - 1 } \big ( \mathrm { s p a n } \{ s _ { i } - s _ { j } \} \big ) ,
$$

which consists of states whose node diference $y _ { i } - y _ { j }$ is a scalar multiple of the target diference $s _ { i } - s _ { j }$ . The span on the right has dimension at most one.

Write $W = \operatorname { s p a n } \{ s _ { i } - s _ { j } \}$ . Since $D _ { i j }$ is surjective, the rank–nullity theorem gives dim ker $D _ { i j } =$ $m - d _ { y }$ . Its restriction to $N _ { s , i j } = D _ { i i } ^ { - 1 } ( W )$ is a surjective linear map onto $W$ with the same kernel. Applying rank–nullity to this restriction yields

$$
\dim N _ { s , i j } = \dim \ker D _ { i j } + \dim W = m - d _ { y } + \dim \operatorname { s p a n } \{ s _ { i } - s _ { j } \} .
$$

The span has dimension zero if $s _ { i } = s _ { j }$ and one otherwise. Thus, since $d _ { y } \geq 2$

$$
\dim N _ { s , i j } \leq m - d _ { y } + 1 \leq m - 1 < m .
$$

Thus $N _ { s , i j }$ is a proper linear subspace and has zero Lebesgue measure. Define set

$$
N _ { G } : = \Sigma _ { G } ^ { \circ } \bigcup \left( \bigcup _ { s \in S ( G ) } \bigcup _ { i < j } N _ { s , i j } \right)
$$

where $\Sigma _ { G } ^ { \circ } \subseteq \mathcal { V } _ { n }$ is the switching set for the fixed graph $G ,$ as defined in Definition 3. The set $N _ { G }$ collects exceptional initializations excluded to ensure a unique selected branch from $\{ f _ { 1 } ( G ) , \cdot \cdot \cdot , f _ { M } ( G ) \}$ and rule out node-state collisions at every finite time. The set $N _ { G }$ is Lebesgue null: the switching set is Lebesgue null by Theorem 4, and the remaining union is finite. Absolute continuity gives $\mu _ { G } ( N _ { G } ) = 0$

For $y _ { 0 } \notin N _ { G }$ , let $s = P ( G , y _ { 0 } )$ . Basin preservation and (17) give

$$
y _ { t } = s + a _ { t } ( y _ { 0 } - s ) , \qquad a _ { t } : = \prod _ { k = 0 } ^ { t - 1 } ( 1 - \beta ( G , y _ { k } ) ) > 0 , \qquad a _ { 0 } : = 1 .\tag{23}
$$

If $( y _ { t } ) _ { i } = ( y _ { t } ) _ { j }$ at a finite time, then

$$
0 = a _ { t } { \big ( } ( y _ { 0 } ) _ { i } - ( y _ { 0 } ) _ { j } { \big ) } + ( 1 - a _ { t } ) ( s _ { i } - s _ { j } ) .
$$

Since $a _ { t } > 0$ , the initial row diference belongs to span $\{ s _ { i } - s _ { j } \}$ , contradicting $y _ { 0 } \notin N _ { G }$ . The same null set works for all finite times. This proves the conditional claim (22), and integration over $G$ proves the joint claim.

For the compact restriction, define

$$
\Delta _ { H } ( G , y _ { 0 } ) : = \operatorname* { m i n } _ { 0 \leq t \leq H } \| [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { i } - [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { j } \| .
$$

Finite iterates of $T$ are jointly continuous and equivariant, so $\Delta _ { H }$ is continuous and invariant. The probability-one statement implies $\mathbb { P } ( E \cap \{ \Delta _ { H } = 0 \} ) = 0 .$ , and hence

$$
\mathbb { P } ( E \cap \{ \Delta _ { H } < \eta \} ) \longrightarrow 0 \qquad ( \eta \downarrow 0 ) .
$$

Choose $\eta ~ > ~ 0$ so that this loss is below $\gamma$ and, when subsets $C _ { a }$ are prescribed, below $\frac { 1 } { 2 } \operatorname* { m i n } _ { a } \mathbb { P } ( C _ { a } )$ . Then $E ^ { \prime } : = E \cap \left\{ \Delta _ { H } \geq \eta \right\}$ is compact, invariant, and has all the required properties. □

Lemma 7 (Separated boxes for finitely many state maps). Let E be a compact metric space with a finite Borel measure $\mu { }$ . Let $\tau$ be a nonempty finite index set, and let

$$
V _ { t } = ( v _ { t , 1 } , \ldots , v _ { t , n } ) : E \longrightarrow \mathbb { R } ^ { n \times d _ { y } } , \qquad t \in \mathcal { T } ,
$$

be continuous. (Here i denotes the node index.) Suppose, for some $\eta > 0$

$$
\begin{array} { r } { \| v _ { t , i } ( x ) - v _ { t , j } ( x ) \| \geq \eta \qquad ( x \in E , \ t \in \mathcal { T } , \ i \neq j ) . } \end{array}
$$

For any $\omega , \gamma > 0$ , there exist a compact $E ^ { \prime } \subseteq E$ and finitely many pairwise positively separated compact axis-aligned boxes $Q _ { 1 } , \ldots , Q _ { J } \subset \mathbb { R } ^ { d _ { y } }$ , with centers $b _ { 1 } , \dots , b _ { J } $ , such that:

(1) $\mu ( E \setminus E ^ { \prime } ) < \gamma ;$

(2) every $v _ { t , i } ( x )$ lies in exactly one box, and the n items of each $V _ { t } ( x )$ lie in diferent boxes;

(3) replacing every item in $V _ { t } ( x )$ by its box center changes the full state by less than ω:

$$
\| V _ { t } ( x ) - b ( V _ { t } ( x ) ) \| < \omega \qquad ( x \in E ^ { \prime } , \ t \in \mathcal T ) .
$$

If finitely many measurable subsets $C _ { a } \subseteq E$ have positive measure, $E ^ { \prime }$ can preserve positive measure in every $C _ { a }$ . If E carries a continuous node-permutation action and $V _ { t } ( \pi x ) = \pi V _ { t } ( x )$ then $E ^ { \prime }$ can also be chosen permutation invariant.

Proof. We construct one common grid in the node-state space $\mathbb { R } ^ { d _ { y } }$ . First, we choose its mesh small enough to separate diferent nodes and control the error from replacing states by cell centers. Second, we shift the grid so that the set of inputs placing any node state on a grid boundary has measure zero. Third, we remove inputs whose node states lie in a suficiently thin neighborhood of these boundaries, losing arbitrarily little measure. The remaining node states then lie in finitely many separated compact boxes.

Step 1: Choose a suficiently small mesh. Since $E$ is compact and there are only finitely many continuous maps $v _ { t , i }$ , the union of all their ranges is compact and hence bounded. Choose a grid mesh ℓ satisfying

$$
0 < \ell < \operatorname* { m i n } \left\{ \frac { 2 \omega } { \sqrt { n d _ { y } } } , \frac { \eta } { \sqrt { d _ { y } } } \right\} .
$$

For a shift vector $a \in [ 0 , \ell ) ^ { d _ { y } }$ , the grid cells are

$$
a + \ell z + [ 0 , \ell ] ^ { d _ { y } } , \qquad z \in \mathbb { Z } ^ { d _ { y } } .
$$

Each cell has diameter $\sqrt { d _ { y } } \ell < \eta$ , so it cannot contain two node states from the same $V _ { t } ( x )$ whose distance is at least η. The other bound on ℓ will control the center-replacement error in Step 4.

Step 2: Shift the grid to give its boundaries zero measure. Fix a coordinate $s \in [ d _ { y } ]$ We first identify the coordinate values that occur on a set of inputs of positive measure:

$$
B _ { s } : = \left\{ b \in \mathbb { R } : \mu \{ x \in E : ( v _ { t , i } ( x ) ) _ { s } = b \} > 0 \mathrm { ~ f o r ~ s o m e ~ } t \in \mathcal { T } , \ i \in [ n ] \right\} .
$$

This set is at most countable. Indeed, for any fixed $( t , i )$ and positive integer $p ,$ only finitely many values b can satisfy

$$
\mu \{ x \in E : ( v _ { t , i } ( x ) ) _ { s } = b \} \geq \frac 1 p .
$$

The corresponding level sets are disjoint, and infinitely many such sets would contradict $\mu ( E ) <$ ∞. Every positive-measure level has measure at least $1 / p$ for some $p ,$ so there are at most countably many such levels. Taking the finite union over $( t , i )$ shows that $B _ { s }$ is countable.

The grid boundaries in coordinate s occur at the values $a _ { s } + k \ell , k \in \mathbb { Z }$ . We want none of these values to belong to $B _ { s }$ . A shift that places some $b \in B _ { s }$ on a boundary must satisfy $a _ { s } = b - k \ell$ for some integer k. Thus all such shifts belong to the countable set

$$
\{ b - k \ell : b \in B _ { s } , \ k \in \mathbb { Z } \} .
$$

Since the interval $[ 0 , \ell )$ is uncountable, it contains a shift $a _ { s }$ outside this set. For this choice, every boundary value $a _ { s } + k \ell$ lies outside $B _ { s }$ . By the definition of $B _ { s }$

$$
\mu \{ x \in E : ( v _ { t , i } ( x ) ) _ { s } = a _ { s } + k \ell \} = 0 \qquad ( t \in \mathcal { T } , \ i \in [ n ] , \ k \in \mathbb { Z } ) .
$$

Choose each coordinate shift in this way and fix $a = ( a _ { 1 } , \ldots , a _ { d _ { y } } )$ for the entire family. The union of all grid-boundary hyperplanes is

$$
\mathcal { H } : = \bigcup _ { s = 1 } ^ { d _ { y } } \bigcup _ { k \in \mathbb { Z } } \{ v \in \mathbb { R } ^ { d _ { y } } : v _ { s } = a _ { s } + k \ell \} .
$$

By the countable union bound, the set of inputs for which any $v _ { t , i } ( x )$ lies on H has measure zero.

Step 3: Remove a thin neighborhood of the boundaries. For $0 < \tau < \ell / 2$ , remove the inputs

$$
N _ { \tau } : = \left\{ x \in E : \operatorname* { m i n } _ { t \in \mathcal { T } , i \in [ n ] } \operatorname { d i s t } ( v _ { t , i } ( x ) , \mathcal { H } ) < \tau \right\} .
$$

The set $\mathcal { H }$ is closed: for each coordinate $s ,$ the lattice $a _ { s } + \ell \mathbb { Z }$ is closed, and there are only finitely many coordinates. The displayed minimum is continuous, so $N _ { \tau }$ is relatively open in $E .$ $\mathrm { A s } \ \tau \downarrow 0$ , these sets decrease to the zero-measure event that some node state lies on $\mathcal { H } .$ Since $\mu$ is finite, continuity of measure from above gives $\mu ( N _ { \tau } ) \to 0$

Choose τ suficiently small that

$$
\mu ( N _ { \tau } ) < \gamma
$$

and, if the subsets $C _ { a }$ are prescribed, also

$$
\begin{array} { r } { \mu ( N _ { \tau } ) < \frac { 1 } { 2 } \operatorname* { m i n } _ { a } \mu ( C _ { a } ) . } \end{array}
$$

Then $E ^ { \prime } : = E \backslash N _ { \tau }$ is compact, loses less than $\gamma$ measure, and retains positive measure in every prescribed $C _ { a } .$ . Node permutations reorder the node states without changing their internal coordinates. Since every node is tested against the same fixed grid, equivariance gives

$$
\operatorname* { m i n } _ { t , i } \mathrm { d i s t } ( v _ { t , i } ( \pi x ) , \mathcal { H } ) = \operatorname* { m i n } _ { t , i } \mathrm { d i s t } ( v _ { t , \pi ^ { - 1 } ( i ) } ( x ) , \mathcal { H } ) = \operatorname* { m i n } _ { t , j } \mathrm { d i s t } ( v _ { t , j } ( x ) , \mathcal { H } ) .
$$

Thus $N _ { \tau }$ and $E ^ { \prime } = E \setminus N _ { \tau }$ are permutation invariant, regardless of whether $\mu$ is invariant.

Step 4: Obtain separated boxes and bound the center error. Every retained node state lies at distance at least $\tau$ from every grid boundary, and therefore belongs to one of the shrunken boxes

$$
\begin{array} { r } { Q _ { z } : = a + \ell z + [ \tau , \ell - \tau ] ^ { d _ { y } } . } \end{array}
$$

Keep only boxes containing a node state $v _ { t , i } ( x )$ for some $x \in E ^ { \prime }$ , and enumerate them as $Q _ { 1 } , \ldots , Q _ { J }$ . Only finitely many are needed because all node-state ranges are bounded.

These boxes are compact, and distinct boxes have distance at least $2 \tau > 0$ . Their diameters satisfy

$$
\mathrm { d i a m } ( Q _ { j } ) = \sqrt { d _ { y } } ( \ell - 2 \tau ) < \eta .
$$

Consequently, the n node states of each retained $V _ { t } ( x )$ belong to diferent boxes.

Finally, each node state is within $\sqrt { d _ { y } } ( \ell - 2 \tau ) / 2$ of its box center. Summing the squared errors over the n nodes gives

$$
\| V _ { t } ( x ) - b ( V _ { t } ( x ) ) \| \le \frac { \sqrt { n d _ { y } } } { 2 } ( \ell - 2 \tau ) < \omega .
$$

This proves all the claims.

In the application below, $E ^ { \prime }$ consists of retained graph–initialization pairs, and $V _ { t } ( G , y _ { 0 } ) =$ $T _ { G } ^ { t } ( y _ { 0 } )$ for finitely many times. The boxes therefore provide one fixed collection of address regions for all retained graphs, initializations, and these times. Subsection B.4 then constructs terminal dynamics that keep each node in its box and contract toward its center, preserving the addressing condition while ensuring convergence near the selected solution.

B.4. Proof of the main result. Given the preparations, we are now ready to prove Theorem 5.

Proof of Theorem 5. Choose positive probability budgets satisfying

$$
\delta _ { \mathrm { l o c } } + \delta _ { \mathrm { s t o p } } + \delta _ { \mathrm { s e p } } + \delta _ { \mathrm { g r i d } } < \delta .
$$

These budgets allow us to exclude graph–initialization pairs with total probability less than $\delta ,$ while preserving positive probability for each labeled solution branch. The remaining pairs form a compact family whose ideal trajectories all approach their selected solutions within a common finite number of steps. During these steps, diferent nodes have uniformly separated states, which can be placed in small, separated boxes from one fixed collection shared across all retained inputs.

We then construct an update rule that follows $T _ { G }$ exactly until the state is suficiently close to its selected solution. From that point onward, each node moves halfway toward the center of its current box at every step. This ensures convergence while keeping diferent nodes distinguishable. Switching close to the solution and using small boxes controls both the final solution error and the deviation from the ideal update. Lemma 5 then realizes this rule with a single fixed GNN block, whose repeated application generates all retained trajectories.

Step 1: Localize the initializations and the ideal trajectories. For $G \in \mathcal G _ { \star }$ , the proof of Theorem 4 identifies the G-section of the joint bad set with $\Sigma _ { G } ^ { \circ }$ , even after taking the joint closure. Its switching-set nullity and the first two probability assumptions therefore give $\mathbb { P } ( \Omega ) = 1$ . For each label $^ { a , }$ define

$$
B _ { a } : = \{ ( G , y _ { 0 } ) \in \Omega : P ( G , y _ { 0 } ) = f _ { a } ( G ) \} .
$$

These sets are Borel, since P and the regular branches are continuous on their domains. For $G \in \mathcal G _ { \star }$ , the G-section of $B _ { a }$ is $V _ { f _ { a } ( G ) } ( G )$ , and the section is empty otherwise. Integration of the indicator of $B _ { a }$ against the conditional kernel is measurable in $G ,$ , so disintegration and the positive-basin assumption give, with $\mathbb { P } _ { G }$ denoting the graph marginal,

$$
\mathbb { P } ( B _ { a } ) = \int _ { \mathcal { G } _ { \star } } \mu _ { G } ( V _ { f _ { a } ( G ) } ( G ) ) d \mathbb { P } _ { G } ( G ) > 0 .
$$

A tight finite Borel measure on a metric space is inner regular by compact sets: restrict first to a compact set of arbitrarily high probability and use Borel regularity there. Hence choose compact sets $C _ { 0 } \subseteq \Omega$ and $C _ { a } \subseteq B _ { a }$ such that

$$
\mathbb { P } ( C _ { 0 } ) > 1 - \delta _ { \mathrm { l o c } } , \qquad \mathbb { P } ( C _ { a } ) > 0 \quad ( a = 1 , \ldots , M ) .
$$

Finite symmetrization gives the compact invariant event

$$
E ^ { 0 } : = \bigcup _ { \pi \in \mathfrak { S } _ { n } } \pi \left( C _ { 0 } \cup \bigcup _ { a = 1 } ^ { M } C _ { a } \right) \subseteq \Omega , \qquad \mathbb { P } ( E ^ { 0 } ) > 1 - \delta _ { \mathrm { l o c } } .
$$

The compact sets $C _ { a }$ are the branch subsets whose mass we preserve.

Write $s ( G , y _ { 0 } ) : = P ( G , y _ { 0 } )$ . The union of the full ideal segments,

$$
\begin{array} { r } { \mathcal { C } : = \left\{ \big ( G , ( 1 - a ) y _ { 0 } + a s ( G , y _ { 0 } ) \big ) : ( G , y _ { 0 } ) \in E ^ { 0 } , \mathrm { ~ } a \in [ 0 , 1 ] \right\} , } \end{array}
$$

is a compact subset of Ω: it is the continuous image of $E ^ { 0 } \times [ 0 , 1 ]$ , and Appendix A shows that each segment stays in its initial strict basin. Set

$$
b _ { * } : = \operatorname* { m i n } _ { ( G , y ) \in { \mathscr C } } \beta ( G , y ) \in ( 0 , 1 ) , \qquad R _ { 0 } : = \operatorname* { m a x } _ { ( G , y _ { 0 } ) \in E ^ { 0 } } \| y _ { 0 } - s ( G , y _ { 0 } ) \| < \infty .
$$

Equation (17) then gives the uniform bound

$$
\begin{array} { r } { \| T _ { G } ^ { t } ( y _ { 0 } ) - s ( G , y _ { 0 } ) \| \le ( 1 - b _ { * } ) ^ { t } R _ { 0 } \qquad ( ( G , y _ { 0 } ) \in E ^ { 0 } , t \ge 0 ) . } \end{array}\tag{24}
$$

Step 2: Choose an entrance threshold with a positive margin. We use distance to the solution set to decide when to switch from the ideal update $T _ { G } ( \cdot )$ to the final convergent update. By excluding a small-probability set of graph–initialization pairs, we ensure that this distance stays uniformly away from the chosen threshold at every time up to a common finite horizon. The resulting gap separates the states where the two updates are used, allowing us to combine them into a continuous rule.

For $x = ( G , y _ { 0 } ) \in E ^ { 0 }$ , define

$$
y _ { t } ( x ) : = T _ { G } ^ { t } ( y _ { 0 } ) , \qquad d _ { t } ( x ) : = \mathrm { d i s t } ( y _ { t } ( x ) , S ( G ) ) .
$$

Recall that $E ^ { 0 }$ is a compact subset of Ω, so its graph projection is a compact subset of $\mathcal { G } _ { \star }$ . Each branch $f _ { a }$ is locally Lipschitz on ${ \mathcal { G } } _ { \mathrm { \lambda } }$ <sub>⋆</sub> and hence continuous on this projection. Moreover, y<sub>0</sub>(x) is a coordinate projection, and the recurrence $y _ { t + 1 } ( x ) = T _ { G } ( y _ { t } ( x ) )$ uses the jointly continuous map $( G , y ) \mapsto T _ { G } ( y )$ . Induction therefore shows that $y _ { t } ( x )$ is continuous in $x = ( G , y _ { 0 } )$ for every finite t. Consequently,

$$
d _ { t } ( x ) = \operatorname* { m i n } _ { a = 1 , \dots , M } \| y _ { t } ( x ) - f _ { a } ( G ) \|
$$

is continuous as the minimum of finitely many continuous functions.

The distance $d _ { t }$ is also permutation invariant. Indeed, equivariance of $T$ gives $y _ { t } ( \pi x ) =$ $\pi y _ { t } ( x )$ , while $S ( \pi G ) = \pi S ( G )$ . Since node permutations preserve Euclidean distances,

$$
d _ { t } ( \pi x ) = \operatorname { d i s t } ( \pi y _ { t } ( x ) , \pi S ( G ) ) = \operatorname { d i s t } ( y _ { t } ( x ) , S ( G ) ) = d _ { t } ( x ) .
$$

Finally, Appendix A shows that the ideal trajectory remains in the strict basin of its selected solution s(x). Thus $s ( x )$ remains the unique nearest solution, and

$$
d _ { t } ( x ) = \| y _ { t } ( x ) - s ( x ) \| .
$$

Choose the radius before choosing a time horizon. For each fixed $t \in  { \mathbb { N } } _ { 0 }$ , there are at most countably many distance values $b \geq 0$ for which

$$
\mathbb { P } \{ x \in E ^ { 0 } : d _ { t } ( x ) = b \} > 0 ,
$$

by the same countability argument used in Step 2 of the proof of Lemma 7. Since there are only countably many times t, the collection of all such values across all finite times is still countable. The interval $( 0 , \operatorname* { m i n } \{ \varepsilon , \rho / 2 \} )$ is uncountable, so we can choose a radius r in this interval outside that collection. This ensures

$$
\mathbb { P } \{ x \in E ^ { 0 } : d _ { t } ( x ) = r \} = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } t \in \mathbb { N } _ { 0 } .
$$

Now choose $H \in  { \mathbb { N } } _ { 0 }$ with

$$
( 1 - b _ { * } ) ^ { H } R _ { 0 } < r / 4 .
$$

In particular, every trajectory from $E ^ { 0 }$ is within $r / 4$ of its selected solution by time H.

For $\zeta > 0$ , the events

$$
N _ { \mathrm { s t o p } } ( \zeta ) : = \{ x \in E ^ { 0 } : | d _ { t } ( x ) - r | < \zeta { \mathrm { ~ f o r ~ s o m e ~ } } 0 \leq t \leq H \}
$$

decrease to a null event as $\zeta \downarrow 0$ . Choose $0 < \zeta < r / 2$ such that

$$
\begin{array} { r } { \mathbb { P } ( N _ { \mathrm { s t o p } } ( \zeta ) ) < \operatorname* { m i n } \left\{ \delta _ { \mathrm { s t o p } } , \frac { 1 } { 2 } \operatorname* { m i n } _ { a } \mathbb { P } ( C _ { a } ) \right\} . } \end{array}
$$

The event

$$
E _ { \mathrm { s t o p } } : = E ^ { 0 } \setminus N _ { \mathrm { s t o p } } ( \zeta )
$$

is compact and permutation invariant, with

$$
\begin{array} { r } { \mathbb { P } ( E _ { \mathrm { s t o p } } ) > 1 - \delta _ { \mathrm { l o c } } - \delta _ { \mathrm { s t o p } } , \qquad \mathbb { P } ( C _ { a } \cap E _ { \mathrm { s t o p } } ) > \frac 1 2 \mathbb { P } ( C _ { a } ) > 0 \quad ( a = 1 , \ldots , M ) . } \end{array}
$$

On $E _ { \mathrm { s t o p } }$ , every ideal trajectory stays at least ζ away from the distance threshold r through time $H \colon$

$$
| d _ { t } ( x ) - r | \geq \zeta \qquad ( x \in E _ { \mathrm { s t o p } } , 0 \leq t \leq H ) .
$$

Thus, at each of these times, the distance to the solution set is either at most $r - \zeta$ or at least $r + \zeta .$

On $E _ { \mathrm { s t o p } } ,$ define the entrance time

$$
h ( x ) : = \operatorname* { m i n } \{ 0 \leq t \leq H : d _ { t } ( x ) \leq r - \zeta \} .
$$

It exists because $d _ { H } < r / 4 < r - \zeta$ . Every retained trajectory satisfies

$$
d _ { t } ( x ) \geq r + \zeta \quad ( 0 \leq t < h ( x ) ) , \qquad d _ { h ( x ) } ( x ) \leq r - \zeta .\tag{25}
$$

Thus it first visits the inner neighborhood of radius $r - \zeta$ after remaining outside the larger neighborhood of radius $r + \zeta$ . For each $k \in \{ 0 , \ldots , H \}$ , its entrance-time level set is

$$
L _ { k } : = E _ { \mathrm { s t o p } } \cap \{ d _ { k } \leq r - \zeta \} \cap \bigcap _ { t < k } \{ d _ { t } \geq r + \zeta \} .
$$

Thus $L _ { k }$ consists of graph–initialization pairs, $( G , y _ { 0 } )$ , whose ideal trajectories first enter the $( r - \zeta )$ -neighborhood of the solution set at time k. This is compact and invariant. When $k = 0 .$ there are no pre-entrance inequalities. The sets $L _ { k }$ form a finite partition of $E _ { \mathrm { s t o p } }$ . The horizon

H is a uniform upper bound on this retained family, whereas $h ( x )$ may be smaller and depends on the trajectory.

Step 3: Obtain one collection of distinct node-address boxes. Apply Lemma 6 to $E _ { \mathrm { s t o p } }$ , horizon H, loss $\delta _ { \mathrm { s e p } }$ , and the subsets $C _ { a } \cap E _ { \mathrm { { s t o p } } }$ . It gives a compact invariant $E _ { \mathrm { s e p } } \subseteq E _ { \mathrm { s t o p } }$ and $\eta > 0$ such that

$$
\mathbb { P } ( E _ { \mathrm { s t o p } } \setminus E _ { \mathrm { s e p } } ) < \delta _ { \mathrm { s e p } } , \qquad \mathbb { P } ( C _ { a } \cap E _ { \mathrm { s e p } } ) > 0 ,
$$

and, uniformly over all retained inputs and times up to $H _ { ; }$ , distinct nodes have states at least η apart:

$$
\| [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { i } - [ T _ { G } ^ { t } ( y _ { 0 } ) ] _ { j } \| \geq \eta \qquad \big ( ( G , y _ { 0 } ) \in E _ { \mathrm { s e p } } , 0 \leq t \leq H , \ i \neq j \big ) .
$$

Then choose any $0 < \omega < \zeta ,$ apply Lemma 7 on $E _ { \mathrm { s e p } }$ with the restricted measure $\mathbb { P } ,$ tolerance $\omega ,$ loss $\delta _ { \mathrm { g r i d } }$ , subsets $C _ { a } \cap E _ { \mathrm { s e p } } ,$ and state maps

$$
V _ { t } ( G , y _ { 0 } ) : = T _ { G } ^ { t } ( y _ { 0 } ) , \qquad t = 0 , \ldots , H .
$$

The maps are continuous and equivariant, and the row-separation hypothesis was just established. The lemma gives boxes $Q _ { 1 } , \ldots , Q _ { J }$ and a compact invariant final event

$$
E : = E _ { \varepsilon , \rho , \delta } \subseteq E _ { \mathrm { s e p } } , \qquad \mathbb { P } ( E ) > 1 - \delta , \qquad \mathbb { P } ( C _ { a } \cap E ) > 0 \quad ( a = 1 , \ldots , M ) .\tag{26}
$$

The probability bound follows by adding the four losses. For every $x \in E$ and $t \in \{ 0 , \ldots , H \}$ , there exist pairwise distinct indices $j _ { 1 } , \ldots , j _ { n } \in [ J ]$ such that

$$
[ y _ { t } ( x ) ] _ { i } \in Q _ { j _ { i } } \qquad ( i = 1 , \ldots , n ) .\tag{27}
$$

Replacing each node state by the center of its box changes the full state by less than ω:

$$
\| y _ { t } ( x ) - ( b _ { j _ { 1 } } , \ldots , b _ { j _ { n } } ) \| < \omega ,
$$

where $b _ { j }$ is the center of $Q _ { j }$ . The indices $j _ { i }$ may depend on $( x , t )$ , while the boxes and their centers are fixed for the whole family.

Using the final retained event $E ,$ define two families of graph–state pairs:

$$
\begin{array} { r l } & { K _ { \mathrm { t r } } : = \displaystyle \bigcup _ { k = 0 } ^ { H } \bigcup _ { t = 0 } ^ { k - 1 } \{ ( G , y _ { t } ( x ) ) : x = ( G , y _ { 0 } ) \in E \cap L _ { k } \} , } \\ & { K _ { \mathrm { e n t } } : = \displaystyle \bigcup _ { k = 0 } ^ { H } \{ ( G , y _ { k } ( x ) ) : x = ( G , y _ { 0 } ) \in E \cap L _ { k } \} . } \end{array}
$$

Here $K _ { \mathrm { t r } }$ collects the states of each retained trajectory before its first visit to the $( r - \zeta ) .$ neighborhood of $S ( G )$ ; these are the transient states. The family $K _ { \mathrm { e n t } }$ collects the state at that first visit for each trajectory; these are the entrance states. $\mathrm { B y }$ the definition of $L _ { k }$

$$
\mathrm { d i s t } ( y , S ( G ) ) \geq r + \zeta \quad \mathrm { o n ~ } K _ { \mathrm { t r } } , \qquad \mathrm { d i s t } ( y , S ( G ) ) \leq r - \zeta \quad \mathrm { o n ~ } K _ { \mathrm { e n t } } .
$$

Both families are compact, being finite unions of continuous images of the compact sets $E \cap L _ { k }$ They are permutation invariant because these sets are invariant and the maps $y _ { t }$ are equivariant. In every state from either family, diferent nodes belong to diferent boxes from the same collection $Q _ { 1 } , \ldots , Q _ { J } { \mathrm { - i . e . } }$ , see (27). Hence, $K _ { \mathrm { t r } }$ and $K _ { \mathrm { e n t } }$ admit robust node-state symmetry breaking (Definition 6).

Step 4: Construct the terminal dynamics and realize both regimes. Let ${ \cal Q } : =$ $\textstyle \bigcup _ { j = 1 } ^ { J } Q _ { j }$ , and define $b ( v ) : = b _ { j } \ ( v \in Q _ { j } )$ , where $b _ { j }$ is the center of $Q _ { j }$ . This map is continuous on $Q$ because it is constant on each box and the boxes are positively separated. For a state y with every node state $y _ { i } \in Q$ , define

$$
b ( y ) : = ( b ( y _ { 1 } ) , \ldots , b ( y _ { n } ) ) , \qquad \Psi ( y ) : = b ( y ) + \textstyle { \frac { 1 } { 2 } } ( y - b ( y ) ) .
$$

The update $\Psi$ moves each node state halfway toward the center of its current box. Since the boxes are convex, each node remains in that same box, so

$$
b ( \Psi ( y ) ) = b ( y ) .
$$

Repeated updates therefore converge to the fixed center pattern $b ( y )$ , halving each node’s distance to its box center at every step.

For each entrance state, include the entire line segment joining it to its box-center pattern. These segments form the terminal family:

$$
K _ { \mathrm { t e r m } } : = \left\{ \left( G , b ( y ) + a ( y - b ( y ) ) \right) : \left( G , y \right) \in K _ { \mathrm { e n t } } , \ a \in [ 0 , 1 ] \right\} .\tag{28}
$$

Here $a \ = \ 1$ gives the entrance state $y ,$ while $a ~ = ~ 0$ gives the center pattern $b ( y )$ toward which the terminal updates converge. Fix an entrance state $y .$ Its node states belong to distinct boxes: there are pairwise distinct indices $j _ { 1 } , \dots , j _ { n }$ such that $y _ { i } \in Q _ { j _ { i } }$ . For any state $z = b ( y ) + a ( y - b ( y ) )$ on the segment,

$$
z _ { i } = b _ { j _ { i } } + a ( y _ { i } - b _ { j _ { i } } ) \in Q _ { j _ { i } } , \qquad a \in [ 0 , 1 ] ,
$$

because $Q _ { j _ { i } }$ is convex and contains both $y _ { i }$ and its center $b _ { j _ { i } }$ . Thus each node remains in the box it occupied at entrance, and diferent nodes continue to occupy diferent boxes—i.e., $K _ { \mathrm { t e r m } }$ admits robust node-state symmetry breaking (Definition $6 )$

The family $K _ { \mathrm { t e r m } }$ is compact as a continuous image of $K _ { \mathrm { e n t } } \times [ 0 , 1 ]$ , and it is permutation invariant. Moreover,

$$
\Psi ( z ) = b ( y ) + \frac { a } { 2 } ( y - b ( y ) ) \in K _ { \mathrm { t e r m } } ( G ) ,
$$

where $K _ { \mathrm { t e r m } } ( G ) : = \{ w : ( G , w ) \in K _ { \mathrm { t e r m } } \}$ . Thus every terminal update stays in this family, which contains all terminal iterates and their limits.

For a terminal state $z = b ( y ) + a ( y - b ( y ) )$ ) arising from an entrance state $y = y _ { h ( x ) } ( x )$ , let $s = s ( x )$ be that initialization’s selected solution. The grid bound and (25) imply

$$
\begin{array} { r } { \| { z } - { y } \| < \omega , \qquad \| { z } - { b } ( { y } ) \| < \omega , \qquad \| { z } - { s } \| \le r - \zeta + \omega < r . } \end{array}\tag{29}
$$

These estimates hold also at both endpoints of the segment. In particular, the solution-set distance on $K _ { \mathrm { t e r m } }$ is below $r ,$ whereas it is at least $r + \zeta$ on $K _ { \mathrm { t r } }$ . The two compact families, $K _ { \mathrm { t e r m } }$ and $K _ { \mathrm { t r } }$ , are disjoint and, when both are nonempty, positively separated.

Set $K : = K _ { \mathrm { t r } } \cup K _ { \mathrm { t e r m } }$ and prescribe

$$
F ^ { \ast } ( G , y ) : = \left\{ \begin{array} { l l } { T _ { G } ( y ) , } & { ( G , y ) \in K _ { \mathrm { t r } } , } \\ { \Psi ( y ) , } & { ( G , y ) \in K _ { \mathrm { t e r m } } . } \end{array} \right.\tag{30}
$$

The two pieces are continuous and equivariant, and their domains are disjoint compact subsets of $K$ . Thus $F ^ { * }$ is a well-defined continuous equivariant map on $K$

Since $K _ { \mathrm { t r } }$ and $K _ { \mathrm { t e r m } }$ admit robust node-state symmetry breaking, K admits robust statebased symmetry breaking, with address regions $A _ { j } = Q _ { j }$ . Apply Lemma 5 with $u = 0 , v = 1$ 2 and $F _ { 1 } = F ^ { * }$ . It gives one MP-GNN satisfying

$$
\operatorname { G N N } ( G , y ) = F ^ { * } ( G , y ) \qquad ( ( G , y ) \in K ) .\tag{31}
$$

The block is defined on the entire input domain by the extension steps in Lemma 5. Its proof extends the address encoder to all of $\mathbb { R } ^ { d _ { y } }$ and uses the Tietze extension theorem to extend the decoder from the compact reconstruction image to its entire Euclidean ambient space. The resulting GNN is therefore defined for every $( G , y ) \in \mathcal { G } \times \mathcal { Y } _ { n } .$ , while its agreement with $F ^ { \star }$ is guaranteed on $K .$ In particular, it already realizes Ψ on $K _ { \mathrm { t e r m } } .$ so no separate extension of Ψ is needed. Both regimes are implemented by the same fixed block using only the current input $( G , y )$ , without an additional time input or stopping flag.

Step 5: Verify the full trajectory and branch capture. Fix $x = ( G , y _ { 0 } ) \in E$ and write $h = h ( x )$ . For $t < h ,$ , each ideal update from the transient family leads either to its next transient state or to the entrance state. From the entrance state onward, Ψ stays in $K _ { \mathrm { t e r m } }$ Thus exact realization and induction give

$$
\widehat { y } _ { t } = T _ { G } ^ { t } ( y _ { 0 } ) \quad ( 0 \leq t \leq h ) , \qquad \widehat { y } _ { h + k } = b ( \widehat { y } _ { h } ) + 2 ^ { - k } \big ( \widehat { y } _ { h } - b ( \widehat { y } _ { h } ) \big ) \quad ( k \geq 0 ) .\tag{32}
$$

This includes the case $h = 0$ . The trajectory converges to ${ \widehat { y } } _ { \infty } = b ( { \widehat { y } } _ { h } )$ , and (29) gives

$$
\begin{array} { r } { \| \widehat { y } _ { \infty } - s ( G , y _ { 0 } ) \| \le r - \zeta + \omega < r < \varepsilon . } \end{array}
$$

The learned update equals $T _ { G }$ on the transient. On the terminal family, $b ( z ) = b ( y )$ for its generating entrance state, so

$$
\begin{array} { r } { \| \Psi ( z ) - z \| = \frac { 1 } { 2 } \| z - b ( z ) \| < \omega / 2 . } \end{array}
$$

Using (18) and (29), uniformly on $K _ { \mathrm { t e r m } }$

$$
\begin{array} { r l } & { \| \operatorname { G N N } ( G , z ) - T _ { G } ( z ) \| \leq \| \Psi ( z ) - z \| + \| z - T _ { G } ( z ) \| } \\ & { \qquad \leq \frac { 1 } { 2 } \| z - b ( z ) \| + \| z - s \| } \\ & { \qquad \leq \omega / 2 + r - \zeta + \omega < r + \omega / 2 < 2 r < \rho . } \end{array}
$$

The right-hand bounds are constants independent of the trajectory and time, proving the supremum bound in (20).

The preceding argument establishes convergence and the uniform operator-error bound for every $( G , y _ { 0 } ) \in E$ . In particular, every retained trajectory converges to a limit within ε of its selected solution, and hence of $S ( G )$ . Together with (26), this gives

$$
\begin{array} { r } { \mathbb { P } ( \widehat { y } _ { t }  \widehat { y } _ { \infty } , \mathrm { ~ d i s t } ( \widehat { y } _ { \infty } , S ( G ) ) < \varepsilon ) \geq \mathbb { P } ( E ) \geq 1 - \delta . } \end{array}
$$

To verify branch capture, fix a label a. By construction, $C _ { a } \subseteq B _ { a }$ , so every $( G , y _ { 0 } ) \in C _ { a } \cap E$ satisfies

$$
y _ { 0 } \in V _ { f _ { a } ( G ) } ( G ) , \qquad s ( G , y _ { 0 } ) = f _ { a } ( G ) .
$$

The limit estimate therefore gives

$$
\begin{array} { r } { \| \widehat { y } _ { \infty } - f _ { a } ( G ) \| = \| \widehat { y } _ { \infty } - s ( G , y _ { 0 } ) \| < \varepsilon \qquad ( ( G , y _ { 0 } ) \in C _ { a } \cap E ) . } \end{array}
$$

Since each restriction preserved positive probability in $C _ { a }$

$$
\begin{array} { r } { \mathbb { P } \big ( \boldsymbol { Y } _ { 0 } \in V _ { f _ { a } ( G ) } ( G ) , \ \widehat { y } _ { t } \to \widehat { y } _ { \infty } , \ \| \widehat { y } _ { \infty } - f _ { a } ( G ) \| < \varepsilon \big ) \geq \mathbb { P } ( C _ { a } \cap E ) > 0 . } \end{array}
$$

Thus every labeled branch has positive capture probability under the joint law of graphs and initializations.

Finally, the graph projection $K _ { \varepsilon , \rho , \delta } : = \mathrm { p r o j } _ { \mathcal { G } } E$ is compact because it is the continuous image of the compact set E. Lemma 5 gives a single GNN that agrees with $F ^ { \star }$ on all of K. For every retained pair $( G , y _ { 0 } ) \in E$ , the entire learned trajectory stays in $K$ . Thus the same GNN, with all component maps fixed, generates every retained trajectory by repeated application. Moreover, the shared nodewise maps and sum aggregations commute with node relabeling, giving the permutation equivariance: $\mathrm { G N N } ( \pi G , \pi y ) = \pi \mathrm { G N N } ( G , y )$ . This concludes the proof. □

## Appendix C. An obstruction to almost-sure validity for contractive

## equivariant dynamics

We use the graph space ${ \mathcal { E } } _ { n }$ , output space $\mathcal { Y } _ { n } = \mathbb { R } ^ { n \times q }$ , Euclidean norms, and permutation actions of Appendix $\mathrm { A } .$ . In particular, $\pi y = P _ { \pi } y$ , and the graph domain ${ \mathcal { G } } \subseteq { \mathcal { E } } _ { n }$ is closed under node relabeling. The Euclidean norm on a matrix is its Frobenius norm. All continuity and neighborhood statements on $\mathcal { G }$ refer to its relative topology.

For a graph $G \in { \mathcal { G } }$ , define its automorphism group and the corresponding fixed subspace by

$$
\Gamma _ { G } : = \{ \pi \in \mathfrak { S } _ { n } : \pi G = G \} , \qquad \mathrm { F i x } ( \Gamma _ { G } ) : = \{ y \in \mathcal { V } _ { n } : \pi y = y \mathrm { ~ f o r ~ e v e r y ~ } \pi \in \Gamma _ { G } \} .
$$

The latter is a closed linear subspace, since

$$
\mathrm { F i x } ( \Gamma _ { G } ) = \bigcap _ { \pi \in \Gamma _ { G } } \ker \bigl ( y \mapsto P _ { \pi } y - y \bigr ) .
$$

Let $\nu$ be a Borel probability measure on G. We write $G _ { 0 } \in$ supp ν when every relatively open neighborhood of $G _ { 0 }$ has positive ν-measure. Thus a full-support graph law satisfies this condition at every $G _ { 0 } \in \mathcal G$

## C.1. The operator-level obstruction.

Theorem 6 (Positive-probability failure of continuous contractive equivariant dynamics). Let $S ( G ) = \{ f _ { 1 } ( G ) , \dots , f _ { M } ( G ) \}$ be a permutation-equivariant finite-branch solution map in the sense of Definition 1. For this theorem, assume additionally that every branch $f _ { a } : \mathcal { G }  \mathcal { V } _ { n }$ is continuous. Suppose there is a graph $G _ { 0 } \in$ supp ν such that

$$
S ( G _ { 0 } ) \cap \operatorname { F i x } ( \Gamma _ { G _ { 0 } } ) = \varnothing .\tag{33}
$$

Let

$$
F : { \mathcal { G } } \times { \mathcal { Y } } _ { n } \to { \mathcal { Y } } _ { n } , \qquad ( G , y ) \mapsto F _ { G } ( y ) ,
$$

be jointly continuous and permutation equivariant. Assume that, for some $\kappa \in [ 0 , 1 )$

$$
\| F _ { G } ( y ) - F _ { G } ( z ) \| \le \kappa \| y - z \| , \qquad F _ { \pi G } ( \pi y ) = \pi F _ { G } ( y ) ,\tag{34}
$$

for all $G \in { \mathcal { G } } , y , z \in { \mathcal { V } } _ { n }$ , and node permutations π. Then the following statements hold.

(1) A unique, initialization-independent limit. For every $G \in { \mathcal { G } }$ , there is a unique fixed point $p _ { F } ( G ) \in \mathcal { V } _ { n }$ . Every trajectory $y _ { t + 1 } = F _ { G } ( y _ { t } )$ satisfies

$$
\| y _ { t } - p _ { F } ( G ) \| \le \kappa ^ { t } \| y _ { 0 } - p _ { F } ( G ) \| , \qquad t \ge 1 .\tag{35}
$$

The map $p _ { F } : \mathcal { G }  \mathcal { V } _ { n }$ is continuous and permutation equivariant. In particular,

$$
p _ { F } ( G ) \in { \mathrm { F i x } } ( \Gamma _ { G } ) \qquad ( G \in { \mathcal { G } } ) .
$$

(2) Failure on a neighborhood of the obstruction graph. The number

$$
\gamma _ { 0 } : = \operatorname* { m i n } _ { 1 \leq a \leq M } \mathrm { d i s t } \big ( f _ { a } ( G _ { 0 } ) , \mathrm { F i x } ( \Gamma _ { G _ { 0 } } ) \big )\tag{36}
$$

is strictly positive and does not depend on $F .$ For every $\varepsilon \in ( 0 , \gamma _ { 0 } )$ , there is a relatively open neighborhood $V _ { F , \varepsilon }$ of $G _ { 0 }$ such that

$$
\mathrm { d i s t } \big ( p _ { F } ( G ) , S ( G ) \big ) > \varepsilon \qquad ( G \in V _ { F , \varepsilon } ) .\tag{37}
$$

(3) Positive-probability failure under random initialization. Let $( G , Y _ { 0 } )$ have any Borel probability law P on $\mathcal { G } \times \mathcal { V } _ { n }$ with graph marginal $\nu ,$ and define

$$
Y _ { t + 1 } = F _ { G } ( Y _ { t } ) , \qquad t \geq 0 .
$$

For every $\varepsilon \in ( 0 , \gamma _ { 0 } )$ , set $\delta _ { F , \varepsilon } : = \nu ( V _ { F , \varepsilon } ) > 0$ . Then $Y _ { t }$ converges for every $( G , Y _ { 0 } )$ , and its limit $Y _ { \infty } = p _ { F } ( G )$ satisfies

$$
\mathbb { P } ( \mathrm { d i s t } ( Y _ { \infty } , S ( G ) ) > \varepsilon ) \geq \delta _ { F , \varepsilon } > 0 .\tag{38}
$$

Consequently,

$$
\mathbb { P } \big ( Y _ { \infty } \in S ( G ) \big ) < 1 .
$$

No independence, density, or full-support assumption is needed for the initialization law.

Proof. We verify the three conclusions.

Step 1: Existence, uniqueness, and convergence for each graph. Fix $G \in { \mathcal { G } }$ and $y _ { 0 } \in \mathcal { V } _ { n }$ . Repeated use of (34) gives

$$
\lVert y _ { t + 1 } - y _ { t } \rVert \leq \kappa ^ { t } \lVert y _ { 1 } - y _ { 0 } \rVert .
$$

For integers $r > s \geq 0$ , it follows that

$$
\| y _ { r } - y _ { s } \| \leq \sum _ { t = s } ^ { r - 1 } \kappa ^ { t } \| y _ { 1 } - y _ { 0 } \| .
$$

The right-hand side tends to zero as $s  \infty ,$ , uniformly in $r > s ,$ because the geometric series is summable. Hence $\left( y _ { t } \right)$ is Cauchy. The space ${ \mathcal { V } } _ { n }$ is complete, so $y _ { t }$ has a limit $p .$ Continuity of $F _ { G }$ yields

$$
F _ { G } ( p ) = \operatorname* { l i m } _ { t \to \infty } F _ { G } ( y _ { t } ) = \operatorname* { l i m } _ { t \to \infty } y _ { t + 1 } = p .
$$

If $p$ and $\widetilde { p }$ are fixed points, then

$$
\begin{array} { r } { \| \boldsymbol { p } - \boldsymbol { \widetilde { p } } \| \leq \kappa \| \boldsymbol { p } - \boldsymbol { \widetilde { p } } \| , } \end{array}
$$

which forces $p = \widetilde { p }$ because $\kappa < 1$ . Write this unique fixed point as $p _ { F } ( G )$ . Its uniqueness makes the limit independent of $y _ { 0 }$ , and iteration of

$$
\| y _ { t + 1 } - p _ { F } ( G ) \| \leq \kappa \| y _ { t } - p _ { F } ( G ) \|
$$

proves (35). If $\kappa = 0$ , the trajectory reaches the fixed point after one step.

Step 2: Continuity of the fixed-point map. For any $G , H \in { \mathcal { G } }$ , the fixed-point equations and (34) imply

$$
\begin{array} { r l } & { \| p _ { F } ( G ) - p _ { F } ( H ) \| = \| F _ { G } ( p _ { F } ( G ) ) - F _ { H } ( p _ { F } ( H ) ) \| } \\ & { \qquad \le \kappa \| p _ { F } ( G ) - p _ { F } ( H ) \| + \| F _ { G } ( p _ { F } ( H ) ) - F _ { H } ( p _ { F } ( H ) ) \| . } \end{array}
$$

Therefore

$$
\| p _ { F } ( G ) - p _ { F } ( H ) \| \leq \frac { \| F _ { G } ( p _ { F } ( H ) ) - F _ { H } ( p _ { F } ( H ) ) \| } { 1 - \kappa } .\tag{39}
$$

Fixing H and letting $G  H ,$ , joint continuity of $F$ makes the numerator tend to zero. This proves continuity of $p _ { F }$ on ${ \mathcal { G } } .$

Step 3: Equivariance and preservation of graph automorphisms. By definition, $p _ { F } ( \pi G )$ is a fixed point of $\pi G$ . For any node permutation $\pi ,$

$$
F _ { \pi G } ( \pi p _ { F } ( G ) ) = \pi F _ { G } ( p _ { F } ( G ) ) = \pi p _ { F } ( G )
$$

which implies $\pi p _ { F } ( G )$ is also a fixed point of $\pi G .$ . By uniqueness of the fixed point at $\pi G$

$$
p _ { F } ( \pi G ) = \pi p _ { F } ( G ) .
$$

If $\pi \in \Gamma _ { G }$ , then $\pi G = G .$ , and consequently $\pi p _ { F } ( G ) = p _ { F } ( G )$ . Thus $p _ { F } ( G ) \in { \mathrm { F i x } } ( \Gamma _ { G } )$

Step 4: A positive error persists on a graph neighborhood. The fixed subspace $\mathrm { F i x } ( \Gamma _ { G _ { 0 } } )$ is closed. By (33), each of the finitely many points $f _ { a } ( G _ { 0 } )$ lies outside this subspace. Its distance to the subspace is therefore strictly positive, and the finite minimum in (36) satisfies $\gamma _ { 0 } > 0$ . Step 3 gives

$$
\| p _ { F } ( G _ { 0 } ) - f _ { a } ( G _ { 0 } ) \| \ge \gamma _ { 0 } \qquad ( 1 \le a \le M ) .
$$

Fix $\varepsilon \in ( 0 , \gamma _ { 0 } )$ and put $\eta : = ( \gamma _ { 0 } - \varepsilon ) / 2 > 0$ . By continuity of $p _ { F }$ and of the finitely many branches, there is a relatively open neighborhood $V _ { F , \varepsilon }$ of $G _ { 0 }$ such that

$$
\| p _ { F } ( G ) - p _ { F } ( G _ { 0 } ) \| < \eta , \qquad \operatorname* { m a x } _ { 1 \leq a \leq M } \| f _ { a } ( G ) - f _ { a } ( G _ { 0 } ) \| < \eta \qquad ( G \in V _ { F , \varepsilon } ) .
$$

For every such $G$ and every $^ { a , }$ the reverse triangle inequality gives

$$
\begin{array} { r l } & { \| p _ { F } ( G ) - f _ { a } ( G ) \| \ge \| p _ { F } ( G _ { 0 } ) - f _ { a } ( G _ { 0 } ) \| - \| p _ { F } ( G ) - p _ { F } ( G _ { 0 } ) \| - \| f _ { a } ( G ) - f _ { a } ( G _ { 0 } ) \| } \\ & { \qquad > \gamma _ { 0 } - 2 \eta = \varepsilon . } \end{array}
$$

Taking the minimum over the finite branch set proves (37).

Step 5: The probability statement. Because $G _ { 0 } \in$ supp ν, $\delta _ { F , \varepsilon } ~ = ~ \nu ( V _ { F , \varepsilon } ) ~ > ~ 0$ . The function

$$
G \longmapsto \mathrm { d i s t } ( p _ { F } ( G ) , S ( G ) ) = \operatorname* { m i n } _ { 1 \leq a \leq M } \| p _ { F } ( G ) - f _ { a } ( G ) \|
$$

is continuous, so the events in the statement are measurable. By Step $1 , Y _ { \infty } = p _ { F } ( G )$ for every initial state. Consequently,

$$
\{ G \in V _ { F , \varepsilon } \} \subseteq \{ \mathrm { d i s t } ( Y _ { \infty } , S ( G ) ) > \varepsilon \} .
$$

Taking P-probabilities and using its graph marginal ν proves (38). The last assertion follows because the event on the right contains no valid limit. □

Remark (Scope of the assumptions). The graph domain need not be compact or connected, and the graph law need not be permutation invariant. Full support of $\nu$ is suficient but not necessary: the proof only uses $G _ { 0 } \in \operatorname { s u p p } \nu .$ The neighborhood argument only needs continuity of the branches at $G _ { 0 } ;$ continuity on $\mathcal { G }$ is imposed above to ensure that all global error events are measurable without additional assumptions. In particular, the branch assumption holds on a domain where the branches are locally Lipschitz, as in the regular stable setting of Appendix A. The failure is uniform over initial states on $V _ { F , \varepsilon }$ only at the level of their limits; no common finite convergence time over the unbounded space ${ \mathcal { V } } _ { n }$ is asserted.

C.2. Application to independently weighted triangles. Fix $0 < w _ { - } < w _ { + } < \infty$ and let $W = [ w _ { - } , w _ { + } ] ^ { 3 }$ . For $w = ( w _ { 1 2 } , w _ { 1 3 } , w _ { 2 3 } ) \in W$ , define

$$
A ( w ) = \left( \begin{array} { c c c } { { 0 } } & { { w _ { 1 2 } } } & { { w _ { 1 3 } } } \\ { { w _ { 1 2 } } } & { { 0 } } & { { w _ { 2 3 } } } \\ { { w _ { 1 3 } } } & { { w _ { 2 3 } } } & { { 0 } } \end{array} \right) , \qquad G ( w ) = ( A ( w ) , 0 ) .
$$

Here $n = 3 , d _ { e } = d _ { v } = 1$ , and $X = 0 \in \mathbb { R } ^ { 3 \times 1 }$ . Set

$$
\mathcal { G } _ { \triangle } : = \{ G ( w ) : w \in W \} , \qquad L _ { G } : = \mathrm { d i a g } ( A { \bf 1 } ) - A , \qquad H _ { G } : = I _ { 3 } + L _ { G } .
$$

We take $q = 2$ and write $\boldsymbol { y } = [ u \ b ] \in \mathcal { V } _ { 3 } = \mathbb { R } ^ { 3 \times 2 }$ , where $u , b \in \mathbb { R } ^ { 3 }$ are its two columns. The single-source difusion solution set is

$$
S ( G ) = \{ f _ { 1 } ( G ) , f _ { 2 } ( G ) , f _ { 3 } ( G ) \} , \qquad f _ { k } ( G ) : = [ H _ { G } ^ { - 1 } e _ { k } e _ { k } ] .\tag{40}
$$

Thus a valid output satisfies $H _ { G } u = b$ , with b a one-hot source indicator.

Corollary 1 (Impossibility under independent edge weights). Let the three edge weights be independent and uniform on $[ w _ { - } , w _ { + } ]$ , and let ν be the induced law on $\mathcal { G } _ { \triangle }$ . Suppose $F : { \mathbf { \sigma } }$ $\mathcal { G } _ { \triangle } \times \mathcal { V } _ { 3 }  \mathcal { V } _ { 3 }$ satisfies the continuity, equivariance, and global contraction assumptions of Theorem 6. For every

$$
0 < \varepsilon < { \sqrt { \frac { 2 } { 3 } } } ,
$$

there is a number $\delta _ { F , \varepsilon } > 0$ such that, under any joint law of $( G , Y _ { 0 } )$ with graph marginal $\nu ,$ the recurrence converges and

$$
\mathbb { P } ( \mathrm { d i s t } ( Y _ { \infty } , S ( G ) ) > \varepsilon ) \ge \delta _ { F , \varepsilon } .
$$

In particular, no such operator converges to a valid solution almost surely over independently weighted graphs. The same conclusion holds for any full-support Borel law on $\mathcal { G } _ { \triangle }$

Proof. We verify the solution-map, support, and symmetry hypotheses.

Step 1: The graph family and solution branches. Relabeling the nodes permutes the three edge coordinates, so $\mathcal { G } _ { \triangle }$ is permutation invariant. For every $v \in \mathbb { R } ^ { 3 }$

$$
v ^ { \top } H _ { G } v = \| v \| _ { 2 } ^ { 2 } + \sum _ { 1 \leq i < j \leq 3 } w _ { i j } ( v _ { i } - v _ { j } ) ^ { 2 } \geq \| v \| _ { 2 } ^ { 2 } .
$$

Hence $H _ { G }$ is invertible and $\| H _ { G } ^ { - 1 } \| _ { \mathrm { o p } } \leq 1$ . The map $G \mapsto H _ { G }$ is linear in the edge matrix up to the constant identity. For any $G , \tilde { G } \in \mathcal { G } _ { \triangle }$ , the resolvent identity gives

$$
H _ { G } ^ { - 1 } - H _ { \widetilde { G } } ^ { - 1 } = H _ { G } ^ { - 1 } ( H _ { \widetilde { G } } - H _ { G } ) H _ { \widetilde { G } } ^ { - 1 } .
$$

Consequently,

$$
\| H _ { G } ^ { - 1 } - H _ { \widetilde { G } } ^ { - 1 } \| _ { \mathrm { o p } } \leq \| H _ { \widetilde { G } } - H _ { G } \| _ { \mathrm { o p } } .
$$

Since linear maps between finite-dimensional Euclidean spaces are globally Lipschitz, every branch in (40) is globally Lipschitz.

Furthermore,

$$
H _ { \pi G } = P _ { \pi } H _ { G } P _ { \pi } ^ { \top } , \qquad H _ { \pi G } ^ { - 1 } = P _ { \pi } H _ { G } ^ { - 1 } P _ { \pi } ^ { \top } .
$$

Using $P _ { \pi } e _ { k } = e _ { \pi ( k ) }$ yields

$$
f _ { \pi ( k ) } ( \pi G ) = [ P _ { \pi } H _ { G } ^ { - 1 } e _ { k } P _ { \pi } e _ { k } ] = \pi f _ { k } ( G ) .
$$

Thus Definition 1 holds with $\tau _ { \pi } ( k ) = \pi ( k )$ . The branches are distinct for every graph because their second columns are distinct; indeed,

$$
\| f _ { k } ( G ) - f _ { \ell } ( G ) \| \geq \| e _ { k } - e _ { \ell } \| _ { 2 } = { \sqrt { 2 } } \qquad ( k \neq \ell ) .
$$

Their collapse partition consists of three singleton sets throughout the domain. Hence $D =$ $U = \varnothing$ and $( \mathcal { G } _ { \triangle } ) _ { \star } = \mathcal { G } _ { \triangle }$ in the notation of Definition 2.

Step 2: The graph law has full support. The map $w \mapsto G ( w )$ is a homeomorphism from W onto $\mathcal { G } _ { \triangle }$ . In fact,

$$
\lVert G ( w ) - G ( \widetilde { w } ) \rVert = \sqrt { 2 } \lVert w - \widetilde { w } \rVert _ { 2 } .
$$

Every nonempty relatively open subset of the cube $W$ has positive three-dimensional Lebesgue measure: it contains the intersection of W with a suficiently small Euclidean ball, and this intersection has positive volume. The uniform product law has constant positive density on W. Its pushforward ν therefore has full support on $\mathcal { G } _ { \triangle }$

Step 3: A symmetric graph provides a fixed positive margin. Choose any $c \in ( w _ { - } , w _ { + } )$ and set $G _ { 0 } = G ( c , c , c )$ . Every node permutation fixes $G _ { 0 } .$ so $\Gamma _ { G _ { 0 } } = { \mathfrak { S } } _ { 3 }$ and

$$
\operatorname { F i x } ( \Gamma _ { G _ { 0 } } ) = \{ [ \alpha \mathbf { 1 } \beta \mathbf { 1 } ] : \alpha , \beta \in \mathbb { R } \} .
$$

For any $k \in \{ 1 , 2 , 3 \}$ and $\beta \in \mathbb { R }$

$$
\| e _ { k } - \beta \mathbf { 1 } \| _ { 2 } ^ { 2 } = ( 1 - \beta ) ^ { 2 } + 2 \beta ^ { 2 } = 3 \left( \beta - { \frac { 1 } { 3 } } \right) ^ { 2 } + { \frac { 2 } { 3 } } \geq { \frac { 2 } { 3 } } .
$$

It follows that

$$
S ( G _ { 0 } ) \cap \mathrm { F i x } ( \Gamma _ { G _ { 0 } } ) = \varnothing , \qquad \gamma _ { 0 } \ge \sqrt { \frac { 2 } { 3 } } .
$$

Step 2 gives $G _ { 0 } \in \mathrm { s u p p } \nu .$ . Applying Theorem 6 proves the claimed probability bound for every $\varepsilon \in ( 0 , \sqrt { 2 / 3 } )$ and for any initialization law. The proof uses only full support of the graph law, so the last assertion follows as well. □

## C.3. Consequence for recurrent message-passing GNNs.

Corollary 2 (Contractive recurrent MP-GNNs). Consider a deterministic, autonomous, weighttied MP-GNN block of the form (15), with continuous component maps and only y<sub>t</sub> carried between recurrent calls. Suppose its complete update satisfies

$$
\begin{array} { r l r } { \| \operatorname { G N N } ( G , y ) - \operatorname { G N N } ( G , z ) \| \leq \kappa \| y - z \| } & { } & { ( G \in { \mathcal { G } } , \ y , z \in { \mathcal { V } } _ { n } ) } \end{array}
$$

for some $\kappa < 1$ . Then the conclusions of Theorem 6 hold whenever its solution-map and graphlaw hypotheses hold. In particular, Corollary 1 applies on $\mathcal { G } _ { \triangle }$ , for every finite number of hidden layers and every choice of finite hidden widths.

Proof. Set $F _ { G } ( y ) : = \mathrm { G N N } ( G , y )$ . The encoder is continuous in $( G , y )$ . Inductively, if all $h _ { i } ^ { ( \ell ) }$ are continuous, then each message is continuous by continuity of $M _ { \ell }$ , its finite sum is continuous, and $h _ { i } ^ { ( \ell + 1 ) }$ is continuous by continuity of $U _ { \ell }$ . The final finite sum and continuous output map preserve continuity. Thus F is jointly continuous.

Under the simultaneous relabeling $( G , y ) \mapsto ( \pi G , \pi y )$ , the encoder satisfies $h _ { \pi ( i ) } ^ { ( 0 ) } ( \pi G , \pi y ) =$ $h _ { i } ^ { ( 0 ) } ( G , y )$ . If this identity holds at layer $\ell ,$ reindexing the sender sum by $j \mapsto \pi ( j )$ gives the same identity for messages and then for layer $\ell + 1$ . The final global sum is invariant, and the node outputs are permuted. Hence ${ \cal F } _ { \pi G } ( \pi y ) = \pi { \cal F } _ { G } ( y )$ , as also stated in (16). The assumed bound is exactly the remaining contraction hypothesis. The two cited results now apply. □

Relation to the multi-attractor results. For the triangle family, the preceding proof established $( \mathcal G _ { \triangle } ) _ { \star } = \mathcal G _ { \triangle }$ . Consequently, Theorem 4 applies. If the initialization law has a density and full support on $\mathcal { \mathrm { { y } _ { 3 } } }$ , every strict Voronoi basin has positive probability. Indeed, for each solution s, a suficiently small open ball centered at s lies in $V _ { s } ( G )$ because the other solutions are finitely many and distinct. The ideal operator therefore has almost-sure validity and positive capture probability for every solution at every fixed graph in $\mathcal { G } _ { \triangle }$

By contrast, every contractive operator above has one limit $p _ { F } ( G )$ for each fixed G. Its limiting law under any initialization distribution is a point mass. It cannot reach two distinct solutions with positive probability at the same graph. Moreover, Corollary 1 rules out ex act almost-sure validity under the independent-weight graph law, even if coverage of multiple solutions is not required.

## Appendix D. Single-source diffusion: protocol and additional results

Datasets. Each instance is a complete undirected graph on three nodes. Its fixed input consists only of the three positive edge weights; static node features are identically zero and omitted from the implemented encoder. Draw $w _ { 1 2 } , w _ { 2 3 } , w _ { 3 1 }$ independently from Unif[0.5, 1.5]. Randomly permute node labels consistently with the adjacency matrix. No node indices, drawing coordinates, prescribed source, or exact difusion fields are supplied as input features. We generate $5 0 0 / 1 0 0 / 1 0 0$ train/validation/test instances.

Model architecture. Across the three models, we use the same architecture:

$$
\begin{array} { l } { { \displaystyle h _ { i } ^ { ( 0 ) } = \mathrm { e n c } ( y _ { i } ) , \qquad } } \\ { { \displaystyle m _ { i } ^ { ( \ell ) } = \sum _ { j \neq i } M _ { \ell } \big ( h _ { i } ^ { ( \ell ) } , h _ { j } ^ { ( \ell ) } , w _ { i j } \big ) , } } \\ { { \displaystyle h _ { i } ^ { ( \ell + 1 ) } = U _ { \ell } \big ( h _ { i } ^ { ( \ell ) } , m _ { i } ^ { ( \ell ) } \big ) , \qquad \ell = 0 , 1 , \qquad } } \\ { { \displaystyle [ N _ { \theta } ( G , y ) ] _ { i } = O ( h _ { i } ^ { ( 2 ) } ) , \qquad \mathrm { G N N } _ { \theta } ( G , y ) = P ( a y + N _ { \theta } ( G , y ) ) . } } \end{array}\tag{41}
$$

Each component is a two-afine-layer MLP with a width-32 hidden layer and a SiLU activation between the afine layers. Messages use sum aggregation over the two neighbors. Layers within a block have separate parameters, but the entire block is shared across recurrent time t. The scalar skip a is also shared over nodes, coordinates, and time. The output projection is $P ( [ u \ b ] ) =$ [u $\mathrm { c l i p } _ { [ 0 , 1 ] } ( b ) ]$ ; it does not impose the mass constraint or choose a source. Both columns of y<sub>0</sub> are i.i.d. $\mathcal { N } ( 0 , 1 )$ samples, independently across nodes and trajectories.

Training. For supervised training, sample $k _ { G }$ uniformly from $\{ 1 , 2 , 3 \}$ once per training graph, independently of the neural initialization, and retain $s _ { G } = [ H _ { G } ^ { - 1 } e _ { k _ { G } } \quad e _ { k _ { G } } ]$ for every subsequent visit. The target is not resampled between updates or epochs. We train the model by minimizing the loss function $\| y _ { T } - s _ { G } \| _ { F } ^ { 2 } / 6$ with $T = 1 0$ . For energy based training, the training loss is $E _ { G } ( u _ { T } , b _ { T } )$ , where energy function $E _ { G }$ is defined in (3) and $T = 1 0$ . Our multiattractor model is directly trained with this energy function without constraints on its weights, while contractive models are trained over the same energy function but applying the Lipschitz control technique described below.

During training, a batch contains 32 diferent training graphs and eight starts per graph (256 trajectories); the final batch has 20 graphs (160 trajectories). With 500 graphs, there are 16 optimizer updates per epoch. We train 1,200 epochs (19,200 updates) with Adam, its default momentum parameters (0.9, 0.999), no weight decay, and no gradient clipping. The initial learning rate is $1 0 ^ { - 3 }$ , multiplied by 0.2 after epoch 600 and by 0.05 relative to the initial rate after epoch 960.

Lipschitz control. We enforce global contraction with respect to the recurrent state in the infinity norm. For a node-state array $h ,$ , this means $\| h \| _ { \infty } = \operatorname* { m a x } _ { i } \| h _ { i } \| _ { \infty }$ . For a weight matrix, we use the corresponding induced norm $\begin{array} { r } { \| W \| _ { \infty } = \operatorname* { m a x } _ { r } \sum _ { s } | W _ { r s } | } \end{array}$

Let $\sigma _ { \mathrm { S } } ( r ) = r / ( 1 + e ^ { - r } )$ denote SiLU. We use the fixed, conservative Lipschitz bound $c _ { \mathrm { S } } : =$ $1 + 1 / e$ , since $| \sigma _ { \mathrm { S } } ^ { \prime } ( r ) | \le 1 + | r | e ^ { - | r | } \le c _ { \mathrm { S } }$ where $r \in \mathbb { R }$ . Consequently, a two-afine-layer MLP $g ( z ) = W _ { 2 } \sigma _ { \mathrm { S } } ( W _ { 1 } z + \beta _ { 1 } ) + \beta _ { 2 }$ satisfies

$$
\begin{array} { r } { \mathrm { L i p } _ { \infty } ( g ) \leq c _ { \mathrm { S } } \| W _ { 2 } \| _ { \infty } \| W _ { 1 } \| _ { \infty } . } \end{array}
$$

For a message MLP $M _ { \ell } ( h _ { i } , h _ { j } , w _ { i j } )$ , the graph is held fixed when measuring state sensitivity. Thus its first-layer norm is taken only over the combined columns multiplying $h _ { i }$ and $h _ { j } ;$ the edge-weight columns and all biases are excluded from this norm bound. In particular, $L _ { M _ { \ell } }$ bounds the joint Lipschitz constant in $[ h _ { i } ; h _ { j } ]$ , rather than a separate constant for each argument.

Concatenation uses a maximum in the infinity norm:

$$
\begin{array} { r } { \| [ h _ { i } ; h _ { j } ] - [ \widetilde { h } _ { i } ; \widetilde { h } _ { j } ] \| _ { \infty } = \operatorname* { m a x } \{ \| h _ { i } - \widetilde { h } _ { i } \| _ { \infty } , \| h _ { j } - \widetilde { h } _ { j } \| _ { \infty } \} . } \end{array}
$$

Therefore, summing the two incoming messages gives a bound $2 L _ { M _ { \ell } } ,$ with no additional factor for the two message arguments. The update MLP receives $[ h _ { i } ; m _ { i } ]$ , giving the layer bound $L _ { U _ { \ell } } \operatorname* { m a x } \{ 1 , 2 L _ { M _ { \ell } } \}$ . Writing $L _ { \mathrm { e n c } }$ and $L _ { O }$ for the encoder and output MLP bounds, the complete residual block satisfies

$$
\mathrm { L i p } _ { y } ( N _ { \theta } ) \leq L _ { O } L _ { \mathrm { e n c } } \prod _ { \ell = 0 } ^ { 1 } L _ { U _ { \ell } } \operatorname* { m a x } \{ 1 , 2 L _ { M _ { \ell } } \} .
$$

We impose the following constraints once after parameter initialization and immediately after every Adam update, before the next forward pass. Set $\rho = 0 . 9 , c = c _ { \mathrm { S } } ^ { - 1 / 2 }$ , and $\eta = 1 - 1 0 ^ { - 5 }$ First clip a to $[ - \rho + 0 . 0 5 , \rho - 0 . 0 5 ]$ ], then use this clipped value to set the output-layer budget.

The absolute row-sum caps and resulting MLP bounds are
<table><tr><td>MLP</td><td>First-layer cap</td><td>Second-layer cap</td><td>Lipschitz bound</td></tr><tr><td>enc</td><td> $\eta c$ </td><td>ηc</td><td> $\overline { { L _ { \mathrm { e n c } } \leq \eta ^ { 2 } } }$ </td></tr><tr><td> $M _ { \ell }$ </td><td> $\eta c / 2$ </td><td>ηc</td><td> $L _ { M _ { \ell } } \leq \eta ^ { 2 } / 2$ </td></tr><tr><td> $U _ { \ell }$   $O$ </td><td> $\eta c$ </td><td>ηc</td><td> $L _ { U _ { \ell } } \le \eta ^ { 2 }$ </td></tr><tr><td></td><td> $\eta c$ </td><td> $\eta ( \rho - | a | ) c$ </td><td> $L _ { O } \leq \eta ^ { 2 } ( \rho - | a | )$ </td></tr></table>

For $M _ { \ell } ,$ the first-layer cap applies to the sum of absolute entries across both state-input blocks together, not to each block separately. All other caps apply to the full weight-matrix rows. For a constrained row or row segment w with cap $B > 0 ,$ , we rescale

$$
w  \frac { w } { \operatorname* { m a x } \{ 1 , \| w \| _ { 1 } / B \} } .
$$

Thus rows already satisfying the cap are unchanged; oversized rows are scaled radially to the boundary. This projection is an optimizer-side operation, performed without diferentiating through it.

Since $2 L _ { M _ { \ell } } \leq \eta ^ { 2 } < 1$ , the two message-passing layers yield $\mathrm { L i p } _ { y } ( N _ { \theta } ) \leq \eta ^ { 8 } ( \rho - | a | )$ . The coordinatewise clipping map $P$ is nonexpansive in the infinity norm, so the full update $\operatorname { G N N } _ { \theta } ( G , y ) =$ $P ( a y + N _ { \theta } ( G , y ) )$ satisfies

$$
\begin{array} { r } { \mathrm { L i p } _ { y } ( \mathrm { G N N } _ { \theta } ) \leq | a | + \eta ^ { 8 } ( \rho - | a | ) \leq \rho . } \end{array}
$$

The constraints therefore certify contraction throughout training, rather than only after training or at sampled states. The multi-attractor and supervised models have no such projection. Single-target supervision itself does not guarantee a unique equilibrium.

Additional visualizations. Figure 3 in the main text shows the evolution of the source coordinates $b _ { t } \in \mathbb { R } ^ { 3 }$ . Since the full recurrent state is $y _ { t } = \left[ u _ { t } \ b _ { t } \right]$ , Figure 4 complements this visualization by showing the difusion coordinates $u _ { t } \in \mathbb { R } ^ { 3 }$ . The same pattern emerges: the multi-attractor trajectories form three distinct clusters, whereas the supervised and contractive controls each concentrate near a single point that remains visibly separated from all three exact difusion solutions.

## Appendix E. Ising Experiments: Details and Additional Results

This section provides details and additional results regarding the Ising model experiments. Datasets. We generate lattice graphs by networkx.triangular lattice graph with open boundaries, following [30]. Each undirected edge receives an independent uniform coupling from $\{ - 0 . 5 , - 1 , - 1 . 5 \}$ ; both directed orientations are stored with equal weight. We generate graphs with controlled sizes to assess scalability. We consider four levels of sizes (see Table 4). Every size has 2,500 train-source and 500 test graphs; the final 250 train-source graphs are held out for validation.

For each graph used in supervised training, Gurobi [25] provides one optimal spin configuration with a reported MIP gap of zero. Enumerating the full solution set is computationally infeasible. Labeling extra-large graphs can take more than $5 { , } 0 0 0$ seconds per instance, so we omit supervised methods at that scale. Certified test references are still used to evaluate the extra-large graphs.

Model Architecture. For the recurrent dynamics $y _ { t + 1 } = \mathrm { G N N } _ { \theta } ( G , y _ { t } )$ , we use a graph convolutional backbone augmented with physical node features.

At each step, we compute for each node i (here $N ( i )$ represents the neighbors of i):

$$
f _ { i } = \sum _ { j \in N ( i ) } J _ { i j } y _ { j } , \qquad a _ { i } = \sum _ { j \in N ( i ) } | J _ { i j } | , \qquad q _ { i } = f _ { i } / \operatorname* { m a x } ( a _ { i } , 1 0 ^ { - 8 } ) ,\tag{42}
$$

![](images/f12bb321cc4b24ebc8ff386e1e2253f964e029b4e6dd5a2c6236193156384033.jpg)  
Figure 4. u-counterpart of Figure 3: the same test graph, seed, 128 trajectories, methods, and time steps. Blue points show $[ ( u _ { t } ) _ { 1 } , ( u _ { t } ) _ { 2 } , ( u _ { t } ) _ { 3 } ]$ ; magenta circles mark the three true solutions.

Table 4. Graph sizes.
<table><tr><td>Level</td><td>Nodes</td><td>Mean nodes</td><td>Edges</td><td>Mean edges</td></tr><tr><td>Compact</td><td>6-55</td><td>26.972</td><td>9-135</td><td>61.068</td></tr><tr><td>Medium</td><td>136-253</td><td>189.174</td><td>360-693</td><td>510.918</td></tr><tr><td>Large</td><td>561-990</td><td>770.580</td><td>1584-2838</td><td>2195.904</td></tr><tr><td>Extra-large</td><td>2016-3916</td><td>2965.198</td><td>5859-11484</td><td>8667.174</td></tr></table>

$$
c _ { i } ^ { ( \beta ) } = \operatorname { t a n h } \left[ \frac { \sum _ { j \in N ( i ) } \mathrm { a t a n h } ( \mathrm { c l i p } ( \operatorname { t a n h } ( \beta J _ { i j } ) y _ { j } , - 1 + 1 0 ^ { - 6 } , 1 - 1 0 ^ { - 6 } ) ) } { \operatorname* { m a x } ( a _ { i } , 1 0 ^ { - 8 } ) } \right] ,\tag{43}
$$

with $\beta \in \{ 0 . 5 , 1 , 2 , 4 \}$ , giving the seven features $\phi _ { i } = [ f _ { i } , q _ { i } , a _ { i } , c _ { i } ^ { ( 0 . 5 ) } , c _ { i } ^ { ( 1 ) } , c _ { i } ^ { ( 2 ) } , c _ { i } ^ { ( 4 ) } ]$ . Here, $f _ { i }$ is the local interaction field, $a _ { i }$ measures the total incident coupling strength, and $q _ { i }$ provides a normalized field. The four $c _ { i } ^ { ( \beta ) }$ features provide nonlinear summaries of neighboring states at diferent coupling scales. Their functional form is inspired by the cavity equations for Ising belief propagation [49], with node states replacing directed cavity messages and an additional normalization by the incident coupling strength. Clipping keeps the argument of atanh within (−1, 1), including for Gaussian initial states. These quantities are supplied as input features; they do not define a separate belief-propagation solver or prescribe the update direction.

Given these features, the recurrent cell $y ^ { + } = \mathrm { G N N } _ { \theta } ( G , y )$ , with $\phi = \phi ( G , y )$ , is defined as follows. The encoder is

$$
h _ { i } ^ { ( 0 ) } = \mathrm { L a y e r N o r m } ( W _ { e , 2 } \sigma ( W _ { e , 1 } [ y _ { i } , \phi _ { i } ] + b _ { e , 1 } ) + b _ { e , 2 } ) ,
$$

with $\sigma = \mathrm { S i L U }$ , dimensions $8  1 6  1 6 .$ and afine LayerNorm with epsilon $1 0 ^ { - 5 }$ . No zeropadding coordinates or random identifiers are supplied. The three weighted GraphConv layers

are

$$
h _ { i } ^ { ( \ell + 1 ) } = \sigma \left( W _ { r , \ell } h _ { i } ^ { ( \ell ) } + W _ { n , \ell } \sum _ { j \in N ( i ) } J _ { i j } h _ { j } ^ { ( \ell ) } + b _ { n , \ell } \right) , \qquad \ell = 0 , 1 , 2 .
$$

They use sum aggregation, a biased neighbor transform, and a bias-free root transform, without intermediate normalization or hidden residual connections. Independent $1 6  1 6  1 6$ SiLU MLPs produce $u _ { i } = U ( h _ { i } ^ { ( 3 ) } )$ and $v _ { i } = R ( h _ { i } ^ { ( 3 ) } )$ . The readout combines local and graph-level information:

$$
\bar { v } = \frac { 1 } { n } \sum _ { j } v _ { j } , \qquad \Delta _ { i } = w _ { o } ^ { \top } [ u _ { i } , \bar { v } ] + b _ { o } , \qquad y _ { i } ^ { + } = \operatorname { t a n h } ( y _ { i } + \Delta _ { i } ) .\tag{44}
$$

Training. For minibatch B with $\begin{array} { r } { N _ { \mathfrak { B } } = \sum _ { \mathfrak { G } \in \mathfrak { B } } | V _ { \mathfrak { G } } | , } \end{array}$ , the proposed model uses the following loss in training: $( 1 / N _ { B } ) \sum _ { B } E _ { G } ( y _ { T } )$ with $T = 1 0$ Training uses such node-weighted energy within each minibatch, whereas test energies are averaged equally across graphs. In particular, we use Adam for 200 epochs, LR $1 0 ^ { - 3 }$ , default betas (0.9, 0.999), epsilon $1 0 ^ { - 8 }$ , and zero weight decay. There is no LR schedule, clipping, dropout, rollback, or early stopping. Graph batches are $3 2 / 3 2 / 4 / 1$ for compact/medium/large/extra-large; each update uses one fresh trajectory per graph. The loader shufles each epoch. The best validation checkpoint is saved after every completed epoch.

Baselines. We compare single-target supervision, unique-equilibrium models, untied feedforward networks, and learned solution distributions. All methods are evaluated on the same test graphs. Table 5 summarizes model capacities and requested training budgets; resource limits and completed budgets are reported below.

• Single-target supervision. We use the same architecture as our multi-attractor GNN, replacing the energy objective with MSE against one Gurobi-certified optimal spin configuration per training graph. Obtaining even one certified solution can exceed 5,000 seconds per graph at the extra-large scale, so we omit this supervised baseline at that scale. We allocate 2,000 training epochs and select the checkpoint with the lowest validation MSE. This extended budget, compared with 200 epochs for our energy-trained model, accommodates the slower optimization observed under single-target supervision: validation MSE continued to improve beyond 200 epochs.

• Unique-equilibrium controls. We adopt the implicit graph neural network (IGNN) of Gu et al. [24], using its released implementation for the implicit layers and enforcing a contraction bound of $\rho = 0 . 9$ to ensure a unique equilibrium. To ensure fairness, we use the same physical-feature family and scalar spin-output interface as our model and train the adapters with the same energy objective. We evaluate hidden widths $H \in \{ 1 6 , 3 2 , 6 4 \}$ , including a capacity-comparable model and two larger variants, to assess whether increased capacity improves performance under the unique-equilibrium constraint.

• Untied feedforward depth controls. We stack independently parameterized copies of our recurrent block to obtain $y _ { t + 1 } = \mathrm { G N N } _ { \theta _ { t } } ( G , y _ { t } )$ , with distinct parameters $\theta _ { t }$ at each stage. We consider 1, 3, and 10 blocks, corresponding to 3, 9, and 30 message-passing layers. These models retain the block architecture, physical features, Gaussian node initialization, and terminal energy objective of our model. They test whether increasing feedforward depth and parameter count can match recurrent performance under the reported budgets.

• Distributional and difusion-based models. We include a mean-field GNN, an annealed Bernoulli GNN, DIFUSCO, and Variational Annealing on Graphs (VAG-CO; denoted VAG). The mean-field GNN uses our width-16, three-layer backbone to map a fresh Gaussian initialization $y _ { 0 } \sim \mathcal { N } ( 0 , I _ { n } )$ to predicted spin means $y _ { 1 } \in ( - 1 , 1 ) ^ { n }$ in one forward pass. It defines conditionally independent spins with $\operatorname* { P r } ( s _ { i } = + 1 \mid G , y _ { 0 } ) = ( 1 + y _ { 1 , i } ) / 2$ and is trained by diferentiating their analytic expected energy, $\mathbb { E } [ E _ { G } ( s ) \mid G , y _ { 0 } ] = E _ { G } ( y _ { 1 } )$ , following the training of Karalias and Loukas [32]. The otherwise identical annealed Bernoulli GNN additionally rewards conditional Bernoulli entropy, with its coeficient decreasing linearly from 0.5 to zero over 200 training epochs, following the annealed variational training of Sun et al. [58]. DIFUSCO [59] is adapted to Ising using binary categorical difusion and a denoiser conditioned on noisy spins, difusion time, and graph features. It is trained with clean-label cross entropy against one Gurobi-certified spin configuration and generates candidates by reverse difusion from uniform random bits. VAG [51] generates spins autoregressively in a randomized breadth-first order, jointly sampling five spins at each decoding step. It is trained with an energy-minus-entropy objective using score-function gradients and a leaveone-out baseline, with temperature annealed during training. For both DIFUSCO and VAG, the matched setting replaces their GNN backbone with our width-16, three-layer GraphConv backbone and local/global-mean readout, while retaining method-specific inputs and output heads. DIFUSCO additionally receives difusion-time and structural features and predicts binary denoising probabilities. VAG additionally receives partial spin assignments, assignment masks, and decoding priorities, and predicts a distribution over the $2 ^ { \bar { 5 } }$ assignments of each five-spin block. These additions yield 3,250 and 5,904 parameters, respectively, compared with 3,153 for our model. The large settings adopt the model-size configurations from their original GitHub repositories: width 256 with 12 layers for DIFUSCO and width 120 with three layers for $\operatorname { V A G } ;$ see Table 5.

Table 5. Learned configurations. H: hidden width; $L \colon$ total learned message passing depth.
<table><tr><td>Model</td><td>H</td><td>L Parameters</td><td>LR Epochs</td><td></td></tr><tr><td>Ours</td><td>16</td><td>3 3,153</td><td> $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Single-target Supervision</td><td>16</td><td>3</td><td>3,153  $1 0 ^ { - 3 }$ </td><td>2000</td></tr><tr><td>IGNN adapter (H16)</td><td>16</td><td>3 2,851</td><td> $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>IGNN adapter (H32)</td><td>32</td><td>3</td><td>10,819  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>IGNN adapter (H64)</td><td>64</td><td>3 42,115</td><td> $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Feedforward  $( \times 1 )$ </td><td>16</td><td>3</td><td>3,153  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Feedforward  $\left( \times 3 \right)$ </td><td>16</td><td></td><td>9,459  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Feedforward (×10)</td><td>16 30</td><td></td><td>31,530  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Mean-field GNN</td><td>16</td><td>3</td><td>3,153  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>Annealed Bernoulli GNN</td><td>16</td><td>3</td><td>3,153  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>DIFUSCO (matched)</td><td>16</td><td>3</td><td>3,250  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>DIFUSCO (large)</td><td>256 12</td><td></td><td>1,909,762  $2 \cdot 1 0 ^ { - 4 }$ </td><td>100</td></tr><tr><td>VAG (matched)</td><td>16</td><td>3</td><td>5,904  $1 0 ^ { - 3 }$ </td><td>200</td></tr><tr><td>VAG (large)</td><td>120</td><td>3</td><td>281,192  $5 \cdot 1 0 ^ { - 4 }$ </td><td>30</td></tr></table>

Resource limits. Structural models, matched DIFUSCO, and energy difusion use training batches $3 2 / 3 2 / 4 / 1$ by size; large DIFUSCO uses $8 / 8 / 1 / 1$ , and VAG uses $3 2 / 1 / 1 / 1$ . Training samples for each graph are processed in parallel, but VAG tokens are sequential. Large VAG is OOM on large and extra-large graphs even alone on a 24-GB GPU at graph batch one. Larger-graph training is capped at 30 accumulated wall-clock hours, including validation and retries; capped rows use the best validated checkpoint. Matched VAG completes $6 8 / 1 7 / 3$ of 200 epochs at medium/large/extra-large sizes.

Metrics. We evaluate $K = 5 0 0$ test graphs, generating $M = 2 0$ candidate spin configurations per graph. Here, $g \in \{ 1 , \ldots , K \}$ indexes test graphs and $m \in \{ 1 , \ldots , M \}$ indexes candidates. Let $s _ { g , m } \in \{ - 1 , + 1 \} ^ { n _ { g } }$ denote the mth candidate for graph $G _ { g } ,$ , where $n _ { g }$ is its number of nodes. For models with continuous outputs, we set $s _ { g , m } = \mathrm { s i g n } _ { + } ( y _ { g , m , T } )$ , where $T$ is the evaluation horizon and $\mathrm { s i g n } _ { + } ( 0 ) = + 1 ;$ discrete generators use their native spin outputs.

We measure energy per node, $e _ { g , m } = E _ { G _ { g } } ( s _ { g , m } ) / n _ { g }$ . Let $e _ { g } ^ { * }$ be the Gurobi-certified optimal energy per node for $G _ { g }$ . The relative optimality gap is

$$
\gamma _ { g , m } = \operatorname* { m a x } \biggl ( 0 , \frac { e _ { g , m } - e _ { g } ^ { * } } { \operatorname* { m a x } ( | e _ { g } ^ { * } | , \epsilon ) } \biggr ) ,
$$

where $\epsilon > 0$ safeguards the denominator. We report three quality metrics:

$$
\bar { e } = \frac { 1 } { K M } \sum _ { g = 1 } ^ { K } \sum _ { m = 1 } ^ { M } e _ { g , m } , \mathrm { B e s t } _ { 2 0 } = \frac { 1 } { K } \sum _ { g = 1 } ^ { K } \operatorname* { m i n } _ { 1 \leq m \leq M } e _ { g , m } , \mathrm { H i t } 1 0 = \frac { 1 0 0 } { K M } \sum _ { g = 1 } ^ { K } \sum _ { m = 1 } ^ { M } \mathbf { 1 } \{ \gamma _ { g , m } \leq 0 . 1 0 \} .
$$

Thus, ¯e averages energy over all candidates, $\mathrm { B e s t _ { 2 0 } }$ averages the best candidate energy within each graph, and Hit10 is the percentage of candidates within 10% of the certified optimum. All graphs receive equal weight. For iterative models, candidates are evaluated at the terminal step; we do not select intermediate states along a trajectory.

For the experiment in Table 1, we measure diversity among optimal spin configurations, allowing a numerical tolerance of $1 0 ^ { - 6 }$

$$
D _ { M } ( g ) = \big | \{ s _ { g , m } : 1 \leq m \leq M , \ \gamma _ { g , m } \leq 1 0 ^ { - 6 } \} \big | , \qquad G _ { M } = \frac { 1 0 0 } { K } \sum _ { g = 1 } ^ { K } \mathbf { 1 } \{ D _ { M } ( g ) \geq 2 \} .
$$

Here, $D _ { M } ( g )$ counts the distinct optimal configurations found among $M$ candidates for graph $G _ { g }$ . We report its graph-average $\begin{array} { r } { D _ { M } = K ^ { - 1 } \sum _ { q = 1 } ^ { K } D _ { M } ( g ) } \end{array}$ . The percentage $G _ { M }$ measures how often the model finds at least two diferent optimal configurations for the same graph, indicating its ability to recover multiple solutions.

For larger graphs, we relax the quality threshold and count distinct configurations within 10% of the certified optimum among 20 candidates:

$$
V _ { 2 0 } ^ { 1 0 \mathcal { Y } } = \frac { 1 } { K } \sum _ { g = 1 } ^ { K } \bigl | \{ s _ { g , m } : 1 \leq m \leq 2 0 , \ \gamma _ { g , m } \leq 0 . 1 0 + 1 0 ^ { - 6 } \} \bigr | .
$$

For example, $V _ { 2 0 } ^ { 1 0 \% } = 2 0 $ means that every graph has 20 distinct candidates meeting this threshold. Repeated configurations count only once, while global spin flips are counted as distinct configurations. These metrics measure the diversity of suficiently accurate outputs; they do not require the underlying trajectories to have converged.

Recurrence is numerically converged at T when $\sqrt { n ^ { - 1 } \| y _ { T } - y _ { T - 1 } \| _ { 2 } ^ { 2 } } < 1 0 ^ { - 4 } ; \ C _ { T }$ is the percentage meeting this criterion. Among nonconverged trajectories, periods $p = 2 , \ldots , 1 0$ are tested using

$$
\left[ \frac { 1 } { p n } \sum _ { k = 0 } ^ { p - 1 } \| y _ { T - k } - y _ { T - k - p } \| _ { 2 } ^ { 2 } \right] ^ { 1 / 2 } < 1 0 ^ { - 4 } ,
$$

assigning the smallest qualifying period. Remaining trajectories are unclassified at the tested horizon, not necessarily divergent or chaotic. Convergence is not applicable to feedforward or time-inhomogeneous generative models.

Longer test-time trajectories. Table 1 reports a convergence rate slightly below 100% at $t = 2 0 0$ . To determine whether the remaining trajectories exhibit persistent oscillations or simply require more iterations, we extend the same ${ 3 0 , 0 0 0 \ ( 5 0 0 \times 2 0 \times 3 ) }$ trajectories across three training seeds to $t = 2 0 0 0$ , without changing the learned weights. Only 11 trajectories fail the numerical fixed-point criterion at $t = 2 0 0 \mathrm { : }$ ; all satisfy it by the measured horizon $t = 5 0 0$ and remain numerically stationary at subsequent reported horizons through t = 2000 (Table 6). The extended checks also confirm sustained numerical stationarity at the final horizon, supporting the interpretation that these exceptions reflect longer transients.

Although training uses only $T = 1 0$ iterations, the learned update remains stable and settles to numerical fixed points when applied for substantially more iterations than seen during training. These are numerical observations rather than asymptotic guarantees. The longer runs serve only to examine convergence: all reported energy, Hit10, $D _ { 2 0 } .$ , and $G _ { 2 0 }$ metrics use the terminal outputs at $t = 2 0 0$ . Thus, the reported solution quality and diversity are already obtained by stopping at $t = 2 0 0 .$ without waiting for every trajectory to satisfy the convergence criterion.

Table 6. Ising test-time dynamics. Percentages are means ± population standard deviations across three training seeds.
<table><tr><td>t</td><td>Ct (%)</td><td>Period 2–10 (%)</td><td>Unclassified (%)</td></tr><tr><td>5</td><td> $3 . 3 7 3 3 \pm 0 . 6 3 7 0$ </td><td></td><td></td></tr><tr><td>10</td><td> $3 2 . 5 8 3 3 \pm 2 . 0 9 5 6$ </td><td></td><td></td></tr><tr><td>20</td><td> $7 7 . 2 7 3 3 \pm 2 . 0 1 2 9$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $2 2 . 7 2 6 7 \pm 2 . 0 1 2 9$ </td></tr><tr><td>50</td><td> $9 7 . 7 4 0 0 \pm 0 . 2 6 1 7$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $2 . 2 6 0 0 \pm 0 . 2 6 1 7$ </td></tr><tr><td>100</td><td> $9 9 . 7 1 0 0 \pm 0 . 0 6 3 8$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 9 0 0 \pm 0 . 0 6 3 8$ </td></tr><tr><td>200</td><td> $9 9 . 9 6 3 3 \pm 0 . 0 1 7 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 3 6 7 \pm 0 . 0 1 7 0$ </td></tr><tr><td>500</td><td> $1 0 0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td></tr><tr><td>1000</td><td> $1 0 0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td></tr><tr><td>2000</td><td> $1 0 0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td></tr></table>

Larger Graphs. Table 7 evaluates scalability on larger graphs while retaining the same width-16 architecture and 3,153 trainable parameters. Our model achieves the lowest mean and best-of-20 energies, the highest ${ \mathrm { H i t 1 0 } } ,$ and the largest number of distinct valid configurations among the evaluated methods at every scale. Its Hit10 reaches 96.88%, 99.96%, and 100% on medium, large, and extra-large graphs, respectively, with an average of 19.376, 19.992, and 20.000 distinct configurations within 10% of optimum among 20 candidates. Thus, a small shared recurrent model continues to produce both accurate and diverse solutions as graph size increases. This is consistent with the set-valued representation established by our theory: a single shared update can accommodate multiple solutions for the same input graph. Here, the empirical evidence concerns finite-horizon outputs, rather than convergence to distinct optimal attractors.

Increasing baseline capacity also helps in some cases, but does not consistently recover this combination of quality and diversity. The 3-block feedforward model substantially improves over one-shot prediction, and large DIFUSCO is competitive on medium and large graphs, yet both yield worse mean and best-of-20 energies than our model. Further increasing feedforward depth to 10 untied blocks reduces Hit10 and produces the same rounded configuration across all 20 initializations for each tested graph. These results support recurrent weight sharing as an efective way to scale solution generation without increasing parameter count, while the time-capped and OOM entries identify practical resource limits of the evaluated baseline configurations.

## Appendix F. Structural module detection in protein graphs: details and additional results

This appendix gives the experiment details for Section 3.2.

Table 7. Scaling to larger Ising graphs with fixed model capacity. The proposed model retains its width-16 architecture and 3,153 parameters across medium, large, and extra-large datasets, each containing 500 test graphs. Metrics use 20 candidates per graph: ¯e and Best<sub>20</sub> are mean and best-of-20 energies per node; Hit10 is the percentage of candidates within 10% of the certified optimum; and $V _ { 2 0 } ^ { 1 0 \% }$ is the mean number of distinct configurations meeting this threshold. <sup>†</sup> marks training capped at 30 hours; OOM denotes a memory failure. <sup>‡</sup> marks supervised methods omitted because of training-labeling costs.
<table><tr><td>Method</td><td>ē↓</td><td>Best20 ↓</td><td>Hit10 (%) ↑</td><td> $\overline { { V _ { 2 0 } ^ { 1 0 \% } \uparrow } }$ </td></tr><tr><td colspan="5">Medium</td></tr><tr><td>Ours</td><td>-1.2431</td><td>-1.2862</td><td>96.88</td><td>19.376</td></tr><tr><td>Single-target Supervision</td><td>1.5248</td><td>1.2913</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H16)</td><td>0.5713</td><td>0.5713</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H32)</td><td>-0.7917</td><td>-0.7917</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H64)</td><td>0.0052</td><td>0.0052</td><td>0.00</td><td>0.000</td></tr><tr><td>Feedforward (×1)</td><td>-1.0970</td><td>-1.1792</td><td>1.60</td><td>0.320</td></tr><tr><td>Feedforward (×3)</td><td>-1.1990</td><td>-1.2590</td><td>58.78</td><td>11.756</td></tr><tr><td>Feedforward (×10)</td><td>-1.1966</td><td>-1.1966</td><td>56.60</td><td>0.566</td></tr><tr><td>Mean-field GNN</td><td>-1.0963</td><td>-1.1781</td><td>1.61</td><td>0.322</td></tr><tr><td>Annealed Bernoulli GNN</td><td>-1.1002</td><td>-1.1811</td><td>1.73</td><td>0.346</td></tr><tr><td>DIFUSCO (matched)</td><td>-1.0864</td><td>-1.2224</td><td>10.49</td><td>2.098</td></tr><tr><td>DIFUSCO (large)</td><td>-1.1934</td><td>-1.2653</td><td>55.25</td><td>11.050</td></tr><tr><td>VAG (matched)†</td><td>-0.9949</td><td>-1.0899</td><td>0.01</td><td>0.002</td></tr><tr><td>VAG (large)</td><td>0.0306</td><td>-0.2003</td><td>0.00</td><td>0.000</td></tr><tr><td colspan="5">Large</td></tr><tr><td>Ours</td><td>-1.2789</td><td>-1.3023</td><td>99.96</td><td>19.992</td></tr><tr><td>Single-target Supervision†</td><td>0.4071</td><td>-0.2762</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H16)</td><td>0.0864</td><td>0.0864</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H32)</td><td>-0.9817</td><td>-0.9817</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H64)</td><td>0.1216</td><td>0.1216</td><td>0.00</td><td>0.000</td></tr><tr><td>Feedforward (×1)</td><td>-1.1296</td><td>-1.1722</td><td>0.00</td><td>0.000</td></tr><tr><td>Feedforward (×3)</td><td>-1.2444</td><td>-1.2734</td><td>75.85</td><td>15.170</td></tr><tr><td>Feedforward (×10)</td><td>-1.2289</td><td>-1.2289</td><td>41.00</td><td>0.410</td></tr><tr><td>Mean-field GNN</td><td>-1.1287</td><td>-1.1720</td><td>0.00</td><td>0.000</td></tr><tr><td>Annealed Bernoulli GNN</td><td>-1.1352</td><td>-1.1778</td><td>0.00</td><td>0.000</td></tr><tr><td>DIFUSCO (matched)</td><td>-1.1285</td><td>-1.2057</td><td>0.59</td><td>0.118</td></tr><tr><td>DIFUSCO (large)</td><td>-1.2514</td><td>-1.2876</td><td>80.26</td><td>16.052</td></tr><tr><td>VAG (matched)†</td><td>-0.9778</td><td>-1.0318</td><td>0.00</td><td>0.000</td></tr><tr><td>VAG (large)</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td colspan="5">Extra-large</td></tr><tr><td>Ours</td><td>-1.3111</td><td>-1.3227</td><td>100.00</td><td>20.000</td></tr><tr><td>Single-target Supervision‡</td><td></td><td></td><td></td><td></td></tr><tr><td>IGNN adapter (H16)</td><td>0.2937</td><td>0.2937</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H32)</td><td>-0.8143</td><td>-0.8143</td><td>0.00</td><td>0.000</td></tr><tr><td>IGNN adapter (H64)</td><td>2.9178</td><td>2.9178</td><td>0.00</td><td>0.000</td></tr><tr><td>Feedforward (×1)</td><td>-1.1443</td><td>-1.1665</td><td>0.00</td><td>0.000</td></tr><tr><td>Feedforward (×3)</td><td>-1.2753</td><td>-1.2912</td><td>98.85</td><td>19.770</td></tr><tr><td>Feedforward (×10)</td><td>-1.2390</td><td>-1.2390</td><td>5.00</td><td>0.050</td></tr><tr><td>Mean-field GNN</td><td>-1.1435</td><td>-1.1658</td><td>0.00</td><td>0.000</td></tr><tr><td>Annealed Bernoulli GNN</td><td>-1.1549</td><td>-1.1767</td><td>0.00</td><td>0.000</td></tr><tr><td>DIFUSCO (matched)‡</td><td></td><td></td><td></td><td></td></tr><tr><td>DIFUSCO (large)‡</td><td></td><td></td><td></td><td></td></tr><tr><td>VAG (matched)†</td><td>0.0980</td><td>0.0355</td><td>0.00</td><td>0.000</td></tr><tr><td>VAG (large)</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

Dataset. We download the PROTEINS dataset through PyTorch Geometric’s TUDataset. For each graph we remove self-loops, collapse duplicate directed entries to one undirected edge, and use both directions only for message passing. The original graph classification label is retained for audit purposes but is never supplied to the models. The task graph contains 1,113 graphs, 43,471 nodes, and 81,044 undirected objective edges.

We compute exact graph-isomorphism groups, randomly order the groups with split seed 2027, and allocate complete groups to $8 0 / 1 0 / 1 0$ splits. This gives 891 training, 111 validation, and 111 test graphs. No exact topology is shared across validation and test or with another split; duplicate topologies inside training remain grouped. Feature-normalization statistics are fit on the 891 training graphs only; standardized added topology features are clipped to [−8, 8].

Table 8. Graph statistics. Entries for nodes and edges are min/mean/median/max.
<table><tr><td>Split</td><td>Graphs</td><td>Nodes</td><td>Undirected edges</td></tr><tr><td>Train</td><td>891</td><td>4 / 36.53 /24 620</td><td>5 / 67.98 3 / 46 / 1049</td></tr><tr><td>Validation</td><td>111</td><td>5 / 52.56 31 504</td><td>6/ 98.33 / 61 / 894</td></tr><tr><td>Test</td><td>111</td><td>8 / 45.84 32 /285</td><td>16 / 86.15 / 57 / 424</td></tr></table>

For exact references, binary $x _ { i k }$ assign node i to group $k , \ \sum _ { k } x _ { i k } \ = \ 1$ . Auxiliary $w _ { i j k }$ linearize $x _ { i k } x _ { j k }$ on each observed edge, and $\begin{array} { r } { { v _ { k } } = \sum _ { i } { d _ { i } } x _ { i k } } \end{array}$ . Gurobi maximizes

$$
4 m \sum _ { \{ i , j \} \in E } \sum _ { k } w _ { i j k } - \sum _ { k } v _ { k } ^ { 2 } ,\tag{45}
$$

whose value divided by $4 m ^ { 2 }$ is $Q _ { G }$ . All 1,113 instances terminate with Gurobi’s optimal status.   
One certified partition per graph is retained as the supervised target.

Node features. All neural models in this experiment receive the same node and edge features. We augment the original node attributes with standard structural descriptors drawn from local degree profiles [12], random-walk encodings [18], and graph statistics documented in NetworkX [26]. The input $\bar { \boldsymbol { s } } _ { i } \in \mathbb { R } ^ { 1 9 }$ comprises five original-attribute/degree coordinates and fourteen additional structural coordinates:

$$
s _ { i } = [ \widetilde { a } _ { i } , \mathrm { o n e h o t 3 } ( \ell _ { i } ) , \widehat { d } _ { i } , \mathcal { Z } _ { \mathrm { n o d e } } ( \psi _ { i } ) ] .
$$

Here $\boldsymbol { a } _ { i }$ is the original continuous node attribute, $\ell _ { i }$ is its original three-category node label, and $\widehat { d } _ { i } = d _ { i } / \operatorname* { m a x } ( 1 , \operatorname* { m a x } _ { j } d _ { j } )$ . The standardized attribute $\widetilde { a } _ { i }$ uses the training-node mean and sample standard deviation (denominator floored at $1 0 ^ { - 8 } )$ . The original node labels are attributes, not target module assignments.

Let $N ( i )$ be the neighbor set, $n = | V | , W _ { i j } = A _ { i j } / \operatorname* { m a x } ( d _ { i } , 1 ) , \tau _ { i }$ the triangle count at $i , \kappa _ { i }$ its core number, and $\pi _ { i }$ its PageRank. The fourteen raw structural coordinates are

$$
\begin{array} { r } { \psi _ { i } = \big [ \log ( 1 + d _ { i } ) , ( W ^ { 2 } ) _ { i i } , ( W ^ { 3 } ) _ { i i } , ( W ^ { 4 } ) _ { i i } , ( W ^ { 8 } ) _ { i i } , c _ { i } , \log ( 1 + \tau _ { i } ) , \frac { \kappa _ { i } } { \operatorname* { m a x } ( 1 , \operatorname* { m a x } _ { j } \kappa _ { j } ) } , n \pi _ { i } , } \\ { \log ( 1 + \mu _ { i } ) , \log ( 1 + \sigma _ { i } ) , \frac { | N _ { 2 } ( i ) | } { \operatorname* { m a x } ( n - 1 , 1 ) } , \frac { | \mathcal { C } ( i ) | } { n } , \frac { b _ { i } } { \operatorname* { m a x } ( d _ { i } , 1 ) } \big ] . } \end{array}\tag{46}
$$

The four diagonal powers are random-walk return probabilities at lengths 2, 3, 4, 8. $c _ { i } =$ $2 \tau _ { i } / [ d _ { i } ( d _ { i } - 1 ) ]$ is the local clustering coeficient, set to zero when $d _ { i } < 2$ . The neighbor-degree mean and population standard deviation are $\mu _ { i }$ and $\sigma _ { i }$ (both zero for an empty neighborhood). $N _ { 2 } ( i )$ contains nodes reachable in two hops after excluding i and its one-hop neighbors; $\mathcal { C } ( i )$ is the connected component containing $i ; ~ b _ { i }$ counts incident bridges. Core numbers use the undirected simple graph. PageRank is approximated by 80 iterations from a uniform initialization, with damping .85, uniform teleportation, and uniform redistribution of dangling-node mass. The operator ${ \mathcal { Z } } _ { \mathrm { n o d e } }$ standardizes each added coordinate using all training nodes, floors population standard deviations at $1 0 ^ { - 6 }$ , and clips the result to $[ - 8 , 8 ]$

Edge features. We combine endpoint attributes with standard neighborhood-similarity and connectivity descriptors. For a directed message edge $j  i ,$ , set $C _ { j i } = N ( j ) \cap N ( i )$ and $U _ { j i } = N ( j ) \cup N ( i )$ . The thirteen raw coordinates are

$$
\begin{array} { r l } & { \chi _ { j i } = \big [ \log ( 1 + d _ { j } ) , \log ( 1 + d _ { i } ) , \widehat { d } _ { j } , \widehat { d } _ { i } , \mathbf { 1 } [ \ell _ { j } = \ell _ { i } ] , | \widetilde { a } _ { j } - \widetilde { a } _ { i } | , \log ( 1 + | C _ { j i } | ) , \frac { | C _ { j i } | } { \operatorname* { m a x } ( | U _ { j i } | , 1 ) } , } \\ & { \qquad \displaystyle \sum _ { w \in C _ { j i } } \frac { 1 } { \log ( \operatorname* { m a x } ( d _ { w } , 2 ) + 1 0 ^ { - 1 2 } ) } , \displaystyle \sum _ { w \in C _ { j i } } \frac { 1 } { \operatorname* { m a x } ( d _ { w } , 1 ) } , \mathbf { 1 } [ \{ j , i \} \mathrm { ~ i s ~ a ~ b r i d g e } ] , } \\ & { \qquad \displaystyle \frac { 1 } { \sqrt { \operatorname* { m a x } ( d _ { j } d _ { i } , 1 ) } } , \frac { | C _ { j i } | } { \operatorname* { m a x } ( \operatorname* { m i n } ( d _ { j } , d _ { i } ) , 1 ) } ] . } \end{array}
$$

The neighborhood-similarity terms include common-neighbor counts and Jaccard similarity [37], the Adamic–Adar index [2], and the resource-allocation and hub-promoted indices [70]. The last coordinate is the hub-promoted index. Standard implementations of Jaccard, Adamic–Adar, and resource-allocation scores are also available in NetworkX [26]. We use $e _ { j i } = \mathcal { Z } _ { \mathrm { e d g e } } ( \chi _ { j i } )$ with training-edge population moments, standard-deviation floor $1 0 ^ { - 6 }$ , and clipping to $[ - 8 , 8 ]$ Both edge orientations are included in message passing. All added features are deterministic functions of graph topology and original attributes. They contain no node identifiers, target partitions, or reference optimizer outputs, and do not depend on the current assignment state.

State and initialization. The recurrent variable is the soft partition $P _ { t } \in \mathbb { R } ^ { n \times 4 }$ . For each trajectory we independently draw node-wise and graph-wise Gaussian logits and initialize

$$
P _ { 0 , i } = \mathrm { s o f t m a x } \left( \frac { \epsilon _ { i } + b _ { G } } { \sqrt { 2 } } \right) , \qquad \epsilon _ { i } , b _ { G } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , I _ { 4 } ) .\tag{47}
$$

The same $b _ { G }$ is shared by the nodes of a graph within a trajectory, but is redrawn for each trajectory.

Message-passing cell. We use a width-32, three-layer PNA-style cell [16]. In particular, let $\sigma = \mathrm { R e L U }$ , at each time step the hidden embeddings are recomputed as

$$
h _ { i } ^ { ( 0 ) } = \mathrm { L a y e r N o r m } \big ( W _ { e , 2 } \sigma ( W _ { e , 1 } [ s _ { i } , P _ { t , i } ] + b _ { e , 1 } ) + b _ { e , 2 } \big ) ,
$$

with encoder dimensions $2 3  3 2  3 2$ . The state-pair edge vector is

$$
r _ { j i } ( P _ { t } ) = [ P _ { t , j } , P _ { t , i } , P _ { t , j } \odot P _ { t , i } , | P _ { t , j } - P _ { t , i } | ] \in \mathbb { R } ^ { 1 6 } .\tag{48}
$$

For $\ell = 0 , 1 , 2$ , compute

$$
\begin{array} { r l } & { m _ { j i } ^ { ( \ell ) } = M _ { \ell } \bigl ( [ h _ { j } ^ { ( \ell ) } , h _ { i } ^ { ( \ell ) } , e _ { j i } , r _ { j i } ( P _ { t } ) ] \bigr ) , } \\ & { \quad a _ { i } ^ { ( \ell ) } = \bigl [ \operatorname* { m e a n } _ { j \in N ( i ) } m _ { j i } ^ { ( \ell ) } , \mathrm { s t d } _ { j \in N ( i ) } m _ { j i } ^ { ( \ell ) } , \operatorname* { m a x } _ { j \in N ( i ) } m _ { j i } ^ { ( \ell ) } , \operatorname* { s u m } _ { j \in N ( i ) } m _ { j i } ^ { ( \ell ) } \bigr ] , } \\ & { h _ { i } ^ { ( \ell + 1 ) } = \mathrm { L a y e r N o r m } \bigl ( h _ { i } ^ { ( \ell ) } + U _ { \ell } \bigl ( [ h _ { i } ^ { ( \ell ) } , a _ { i } ^ { ( \ell ) } , P _ { t , i } ] \bigr ) \bigr ) . } \end{array}\tag{49}
$$

$M _ { \ell }$ is a $, 9 3 \to 3 2 \to 3 2$ MLP with ReLU after both afine maps; $U _ { \ell }$ is a $1 6 4  3 2  3 2$ MLP with ReLU only between afine maps. Aggregations are elementwise.

Global readout and update. The graph bottleneck and correction head are

$$
g _ { G } = \frac { 1 } { n } \sum _ { i } R ( h _ { i } ^ { ( 3 ) } ) \in \mathbb { R } ^ { 4 } , \qquad \Delta _ { \theta , i } = O ( [ h _ { i } ^ { ( 3 ) } , g _ { G } ] ) ,\tag{50}
$$

where R is a $3 2  3 2  4 ~ \mathrm { M L P }$ and O is $\mathrm { ~ a ~ 3 6  3 2  4 ~ M L P } ,$ each with ReLU between its two afine maps. We update

$$
P _ { t + 1 } = \mathrm { s o f t m a x } \left[ \log ( P _ { t } \lor 1 0 ^ { - 6 } ) + \Delta _ { \theta } ( G , P _ { t } ) \right] ,\tag{51}
$$

with row-wise softmax and coordinate-wise clipping before the logarithm. The three MP layers have distinct parameters within a cell; the entire cell is tied across time. There are 35,784 trainable parameters: 1,888 in the encoder, $3 \times 1 0 , 4 6 4$ in message passing, 1,188 in the global map, and 1,316 in the output head. The full model sizes are provided in Table 9.

Training and objectives. For an undirected graph $G = ( V , E )$ with $m = | E |$ edges and node degrees $d _ { i } .$ , let $z _ { i } \in \{ 1 , \ldots , K \}$ denote the module of node i. At resolution $\gamma = 1$ , we minimize the negative modularity:

$$
E _ { G } ( z ) = - Q _ { G } ( z ) = \sum _ { k = 1 } ^ { K } \left( \frac { \sum _ { i } d _ { i } { \bf 1 } [ z _ { i } = k ] } { 2 m } \right) ^ { 2 } - \frac { 1 } { m } \sum _ { \{ i , j \} \in E } { \bf 1 } [ z _ { i } = z _ { j } ] .
$$

The edge term rewards placing connected nodes in the same module; alone, it is minimized by assigning all nodes to one module. The degree-volume term accounts for the within-module connectivity expected from node degrees under the modularity null model. Combining the two terms therefore favors modules with more internal connectivity than this baseline predicts.

For a soft partition $P ,$ where each $P _ { i }$ is a probability vector over the K modules, we define the diferentiable energy

$$
e _ { G } ( P ) = - { \frac { 1 } { m } } \sum _ { \{ i , j \} \in E } \langle P _ { i } , P _ { j } \rangle + { \frac { 1 } { 4 m ^ { 2 } } } \left[ \left\| \sum _ { i } d _ { i } P _ { i } \right\| _ { 2 } ^ { 2 } - \sum _ { i } d _ { i } ^ { 2 } \| P _ { i } \| _ { 2 } ^ { 2 } \right] .\tag{52}
$$

Specifically, for independent assignments $Z _ { i } \sim$ Categorical(P<sub>i</sub>),

$$
e _ { G } ( P ) = \mathbb { E } [ E _ { G } ( Z ) ] - \frac { 1 } { 4 m ^ { 2 } } \sum _ { i } d _ { i } ^ { 2 } .
$$

Thus, the soft objective equals the expected discrete energy up to a graph-dependent constant. During training, each update samples ten training graphs with replacement and four independent initial soft partitions $P _ { 0 }$ per graph. We unroll $T = 5$ tied cells and minimize $\mathbb { E } [ e _ { G } ( P _ { T } ) ]$ ]. We use Adam with learning rate $3 \times 1 0 ^ { - 4 } , \beta _ { 1 } = . 9 , \beta _ { 2 } = . 9 9 9 , \epsilon = 1 0 ^ { - 8 }$ , zero weight decay, global gradient-norm clipping at 1, 3,500 optimizer updates, batch size ten base graphs, and no learning-rate decay. The final update-3,500 checkpoint is used; there is no best-test or bestvalidation checkpoint substitution. Our model and all baselines in Table 2 use training seeds 0, 1, and 2. The unsupervised model never receives the Gurobi labels.

Baselines. Table 9 summarizes the learned configurations used in the reported comparisons. In this PROTEINS experiment, we use the same baseline groups as in the Ising experiment:

• Single-target supervision. We use the same architecture as our multi-attractor GNN, replacing the energy objective $e _ { G } ( P )$ with cross entropy against one Gurobi-optimal partition per graph.

• Unique-equilibrium models. As in the Ising experiments, we adopt the IGNN with hidden widths $H \in \{ 1 6 , 3 2 , 6 4 \}$ and enforce a contraction bound of $\rho = 0 . 9$ . To train IGNN, we use the same unsupervised loss function $\mathbb { E } [ e _ { G } ( P _ { T } ) ]$ . Again, the input features are the same as in the multi-attractor setting.

• Untied feedforward depth controls. The feedforward controls replace the same three-layer rich-PNA cell by $B \in \{ 1 , 2 , 3 , 4 , 5 \}$ independent copies. They receive the same random $P _ { 0 }$ losses, four training trajectories, optimizer, and 3,500-update budget.

• Distributional and difusion-based models. We include the same distributional models as in the Ising experiments: a mean-field GNN, an annealed Bernoulli GNN, DIFUSCO, and Variational Annealing on Graphs (VAG-CO; denoted VAG). They are still applicable here. “Matched” setting uses the same model backbone, learning rate, and training updates.

“Large” setting uses the upstream original repository’s configurations. Details are in Table 9.

Table 9. Learned configurations on PROTEINS. H: hidden width; L: total untied depth.
<table><tr><td>Model</td><td>H</td><td>L Parameters</td><td></td><td>LR Updates</td></tr><tr><td>Ours</td><td>32</td><td>3</td><td>35,784  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Single-target Supervision</td><td>32</td><td>3</td><td>35,784  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>IGNN (H16)</td><td>16</td><td>3</td><td>11,340  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>IGNN (H32)</td><td>32</td><td>3</td><td>39,564  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>IGNN (H64)</td><td>64</td><td>3</td><td>146,700  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Feedforward (×1)</td><td>32</td><td>3</td><td>35,784 3· 10−4</td><td>3,500</td></tr><tr><td>Feedforward (×2)</td><td>32</td><td>6</td><td>71,568  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Feedforward (×3)</td><td>32</td><td>9</td><td>107,352 3· 10−4</td><td>3,500</td></tr><tr><td>Feedforward (×4)</td><td>32 12</td><td></td><td>143,136  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Feedforward (×5)</td><td>32</td><td>15</td><td>178,920  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Mean-field GNN</td><td>32</td><td>3</td><td>35,784  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>Annealed Bernoulli GNN</td><td>32</td><td>3</td><td>35,784  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>DIFUSCO (matched)</td><td>32</td><td>3</td><td>35,880  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>DIFUSCO (large)</td><td>256</td><td>12</td><td>5,555,466  $2 \cdot 1 0 ^ { - 4 }$ </td><td>2,800</td></tr><tr><td>VAG (matched)</td><td>32</td><td>3</td><td>40,270  $3 \cdot 1 0 ^ { - 4 }$ </td><td>3,500</td></tr><tr><td>VAG (large)</td><td>40</td><td>3</td><td>101,790  $5 \cdot 1 0 ^ { - 4 }$ </td><td>10,000</td></tr></table>

Metrics. The caption of Table 2 defines every displayed metric.

Longer test-time trajectories. Table 2 reports a one-step convergence rate of $C _ { 2 0 0 } =$ $8 8 . 0 5 \pm 2 . 2 3 \%$ at $t = 2 0 0$ . To determine whether the remaining trajectories exhibit persistent oscillations or simply require more iterations, we extend the same 6,660 $( 1 1 1 \times 2 0 \times 3 )$ trajectories across three training seeds to $t = 1 0 0 0 0$ , without changing the learned weights. Here t denotes test-time recurrence steps, and results are reported in Table 10.

The convergence rate increases to $C _ { 8 0 0 } = 9 8 . 7 5 \pm 0 . 6 2 \%$ and $C _ { 3 0 0 0 } = 9 9 . 4 9 \pm 0 . 2 9 \%$ . By $t = 5 0 0 0$ , every trajectory is either convergent or classified as periodic at the reported horizons, with 6,632 trajectories satisfying the numerical convergence criterion. The extended checks confirm that the vast majority of non-converged trajectories at $t = 2 0 0$ simply reflect longer transients, while a small remainder settles into stable periodicity rather than divergent behavior.

Although training uses only T = 5 unrolled iterations, the learned update remains stable and largely settles to numerical fixed points when applied for substantially more iterations than seen during training. The longer runs serve only to examine convergence: all reported modularity, Hit10, and discovery metrics use the terminal outputs at t = 200. Thus, the reported solution quality and diversity are already obtained by stopping at $t = 2 0 0$ , without waiting for every trajectory to strictly satisfy the convergence criterion.

## Appendix G. Chemical reaction networks: details and additional results

This appendix provides the experiment details and additional results for Section 3.3.

Dataset. We construct kinetic instances on the published reaction-network structures from [68]. We select networks labeled Multi from the Atom and Joshi construction families, preserving their original train/validation/test assignments. These labels indicate structural capacity for multistationarity but do not provide kinetic rates or stationary concentrations. For each selected structure, we add one inflow $\mathcal { O }  X _ { i }$ and one outflow $X _ { i } \to \emptyset$ per species. We then randomly propose two positive concentration vectors $c ^ { ( 1 ) } , c ^ { ( 2 ) }$ in log space and solve a linear feasibility problem for shared positive rate constants κ satisfying $f _ { G } ( c ^ { ( 1 ) } ) = f _ { G } ( c ^ { ( 2 ) } ) = 0 ;$ the equations are linear in κ when the concentrations are fixed. We retain instances whose two witnesses have normalized residuals at most $1 0 ^ { - 7 }$ , concentration-Jacobian condition numbers at most $1 0 ^ { 9 }$ , and log-RMS separation at least 0.25. After feasibility filtering, we balance the two construction families within each split, obtaining 1,242 training, 266 validation, and 236 test instances, with 2–5 species and 6–19 reactions. Each instance contains the network structure, constructed rate constants, and two numerically verified stationary witnesses, which need not exhaust its solution set. Physics-only training uses only $( \alpha , \beta , \kappa ) ;$ single-target supervision and supervised difusion use one witness selected once and held fixed as the label for each training instance.

Table 10. PROTEINS test-time dynamics. Metrics are the same as in $\mathrm { A p - }$ pendix $\mathrm { E } ,$ Table 6.
<table><tr><td>t</td><td>Ct (%)</td><td>Period 2-10 (%) Unclassified (%)</td></tr><tr><td>10</td><td> $3 . 8 3 \pm 1 . 7 8$ </td></tr><tr><td> $1 6 . 6 5 \pm 3 . 4 5$ </td><td></td></tr><tr><td>20 50</td><td></td></tr><tr><td> $4 7 . 1 8 \pm 1 . 9 2$ </td><td></td></tr><tr><td>100  $6 8 . 7 2 \pm 0 . 8 2$   $8 8 . 0 5 \pm 2 . 2 3 $ </td><td> $1 1 . 9 4 \pm 2 . 2 5$ </td></tr><tr><td>200 500  $9 7 . 5 1 \pm 0 . 4 5$ </td><td> $0 . 0 2 \pm 0 . 0 2$   $0 . 3 9 \pm 0 . 2 8$ </td></tr><tr><td></td><td> $2 . 1 0 \pm 0 . 3 4$   $0 . 4 2 \pm 0 . 2 8$ </td></tr><tr><td>800  $9 8 . 7 5 \pm 0 . 6 2$   $9 9 . 0 7 \pm 0 . 3 0$ </td><td> $0 . 8 3 \pm 0 . 4 8$   $0 . 4 2 \pm 0 . 2 8$   $0 . 5 1 \pm 0 . 2 4$ </td></tr><tr><td>1000 2000  $9 9 . 4 9 \pm 0 . 3 1 $ </td><td> $0 . 4 2 \pm 0 . 2 8$   $0 . 0 9 \pm 0 . 0 7$ </td></tr><tr><td></td><td> $0 . 4 2 \pm 0 . 2 8$   $0 . 0 9 \pm 0 . 0 4$ </td></tr><tr><td>3000  $9 9 . 4 9 \pm 0 . 2 9$   $9 9 . 5 8 \pm 0 . 2 8$ </td><td> $0 . 4 2 \pm 0 . 2 8$   $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>5000</td><td></td></tr><tr><td></td><td> $0 . 4 2 \pm 0 . 2 8$   $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>10000  $9 9 . 5 8 \pm 0 . 2 8$ </td><td></td></tr></table>

Table 11. Dataset construction. Balance is enforced within each split after feasibility filtering. Every retained instance has two constructed witnesses.
<table><tr><td>Split</td><td>Atom attempts</td><td>Joshi attempts</td><td>Atom feasible</td><td>Joshi feasible</td><td>Retained</td><td>Reactions</td></tr><tr><td>Train</td><td>790</td><td>2560</td><td>771</td><td>621</td><td>1242</td><td>6-19</td></tr><tr><td>Validation</td><td>180</td><td>570</td><td>172</td><td>133</td><td>266</td><td>6-18</td></tr><tr><td>Test</td><td>300</td><td>600</td><td>293</td><td>118</td><td>236</td><td>6-18</td></tr></table>

Nodes and edges. There is one node per species and one per reaction, including inflow/outflow reactions. For every pair $( i , r )$ with $\alpha _ { r i } > 0$ or $\beta _ { r i } > 0$ , we include both directed incidence edges. There is no extra node for ∅, no stoichiometric-copy expansion, and no direct species–species edge. The six static node coordinates are

$$
\begin{array} { l } { \displaystyle { \boldsymbol { x } _ { i } = \left( 1 , 0 , 0 , 0 , 0 , 0 \right) , } } \\ { \displaystyle } \\ { \displaystyle { \boldsymbol { x } _ { r } = \left( 0 , 1 , \log \kappa _ { r } , \sum _ { i } \alpha _ { r i } , \sum _ { i } \beta _ { r i } , \sum _ { i } \vert \beta _ { r i } - \alpha _ { r i } \vert \right) , } } \end{array}
$$

species,

(53)

reaction.

The first two coordinates distinguish node types; the remaining coordinates encode the log rate, total reactant/product orders, and net stoichiometric magnitude. Apart from the log-rate

transform, these graph inputs are used directly, without dataset-fitted standardization. The four edge coordinates are

$$
e _ { i \to r } = ( \alpha _ { r i } , \beta _ { r i } , 1 , 0 ) , \qquad e _ { r \to i } = ( \alpha _ { r i } , \beta _ { r i } , 0 , 1 ) .
$$

The last two entries encode message direction, not reactant versus product. A species can be both a reactant and a product, with its two coeficients retained on the same incidence pair. Reaction roles are thus edge attributes, not a single global species label. All compared neural models receive these same base graph features.

State and initialization. The recurrent state is only the species vector $z _ { t } = \log c _ { t } \in$ $[ - 2 0 , 2 0 ] ^ { s }$ . With width $w = 3 2$ , compute the static encoding

$$
b _ { v } = \operatorname { t a n h } \big ( \operatorname { L a y e r N o r m } ( W _ { x } x _ { v } + b _ { x } ) \big ) .
$$

Each candidate draws independent node vectors $\xi _ { v } \sim \mathcal { N } ( 0 , I _ { 8 } )$ . A bias-free $8  3 2$ map and a scalar afine head initialize

$$
z _ { 0 , i } = \mathrm { c l i p } _ { [ - 2 0 , 2 0 ] } \left( w _ { \mathrm { i n i t } } ^ { \top } ( b _ { i } + W _ { \xi } \xi _ { i } ) + b _ { \mathrm { i n i t } } \right) .
$$

Subsequent calls receive no identifier forcing. At each call, reconstruct a temporary node matrix $H ^ { ( 0 ) }$ from $b ,$ replace coordinate zero of each species row with its current $z _ { i } ,$ and leave reaction rows equal to $b _ { r }$ . All temporary hidden coordinates are discarded after the call except the new species log concentrations. There is no auxiliary carried hidden state.

Message-passing backbone. One update contains three distinct learned GATv2 [11] blocks; the entire stack is tied across recurrent iterations. For $\ell = 1 , 2 , 3 ,$ , compute

$$
U ^ { ( \ell ) } = \mathrm { L a y e r N o r m } _ { \ell } \left( \mathrm { G A T v 2 } _ { \ell } ( H ^ { ( \ell - 1 ) } , E , e ) \right) , \qquad H ^ { ( \ell ) } = \left\{ \operatorname { t a n h } U ^ { ( \ell ) } , \quad \ell < 3 , \ldots \right\} ,
$$

Each block is PyG GATv2Conv(32,32,heads=4,concat=False,edge dim=4). Attention heads are averaged, not concatenated. Self-loops are enabled, with mean incident edge features for their attributes; these internal loops are not chemical reactions. Other settings are zero attention dropout, LeakyReLU slope 0.2, separate source/target transforms, and biases enabled. LayerNorm is afine with epsilon $1 0 ^ { - 5 }$ . There is no spectral constraint on these attention blocks.

Dynamic physical features. For $c = \exp z$ , compute $\begin{array} { r } { \ell _ { r } = \log \kappa _ { r } + \sum _ { i } \alpha _ { r i } z _ { i } } \end{array}$ and protected fluxes $\widetilde { v } _ { r } = \exp ( \mathrm { c l i p } _ { [ - 3 0 , 3 0 ] } ( \ell _ { r } ) )$ . Set $\widetilde { a } = | N | \widetilde { v }$ and $\widetilde { q } _ { i } = ( N \widetilde { v } ) _ { i } / ( \widetilde { a } _ { i } + 1 0 ^ { - 1 2 } )$ . The node-wise dynamic features are

$$
\psi _ { i } = ( z _ { i } , \widetilde { q } _ { i } , \log ( \widetilde { a } _ { i } + 1 0 ^ { - 1 2 } ) ) , \qquad \psi _ { r } = ( \ell _ { r } , 0 , 0 ) .
$$

A shared afine $3  3 2$ map followed by tanh encodes $\psi _ { v }$ . The reaction log-flux feature itself is not clipped; clipping protects the exponentials used to construct the internal residual. The independent evaluation residual uses original coeficients and unclipped mass-action fluxes.

Gated readout and update. A bias-free conditioning map $W _ { b }$ and the physical-feature encoder give

$$
\widehat { H } = \mathrm { G R U C e l l } \Big ( \mathrm { t a n h } \Big [ H ^ { ( 3 ) } + W _ { b } b + \mathrm { t a n h } ( W _ { \psi } \psi + b _ { \psi } ) \Big ] , H ^ { ( 0 ) } \Big ) .
$$

This width-32 GRU [15] is a within-call gating computation, not an additional inter-iteration memory. Three scalar afine heads applied to each species row of $\widehat { H }$ produce

$$
\begin{array} { r } { d _ { i } = \frac 1 2 \sigma ( w _ { d } ^ { \mathsf T } \widehat H _ { i } + b _ { d } ) , \qquad g _ { i } = 2 \operatorname { t a n h } ( w _ { g } ^ { \mathsf T } \widehat H _ { i } + b _ { g } ) , \qquad a _ { i } ^ { \mathrm { N N } } = \operatorname { t a n h } ( w _ { a } ^ { \mathsf T } \widehat H _ { i } + b _ { a } ) . } \end{array}
$$

The update is

$$
z _ { i } ^ { + } = \mathrm { c l i p } _ { [ - 2 0 , 2 0 ] } \left[ z _ { i } + d _ { i } \left( g _ { i } \widetilde { q } _ { i } + \mathrm { t a n h } ( \widetilde { R } ) a _ { i } ^ { \mathrm { N N } } \right) \right] , \qquad \widetilde { R } = \sqrt { s ^ { - 1 } \sum _ { i } \widetilde { q } _ { i } ^ { 2 } + 1 0 ^ { - 2 4 } } .\tag{54}
$$

The signed-gain head starts with zero weights and bias atanh $( 1 / 2 )$ , so its initial gain is one. The model has 35,716 trainable parameters. It is a physics-assisted learned solver, not a generic GNN with only a physics loss, and is not guaranteed contractive. All aggregation/readout operations are permutation-equivariant. Initialization is equivariant when ξ is permuted with the nodes, and equivariant in distribution under independent identifier sampling. After initialization, the shared map depends only on $G$ and z.

Training and residual objective. Using the original mass-action fluxes, define

$$
q _ { i } ( c ) = \frac { ( N v ( c ) ) _ { i } } { ( | N | v ( c ) ) _ { i } + 1 0 ^ { - 1 2 } } , R ( c ) = \sqrt { \frac { 1 } { s } \sum _ { i } q _ { i } ( c ) ^ { 2 } } , E _ { \mathrm { r e s } } ( c ) = \frac { 1 } { s } \sum _ { i } \phi _ { 0 . 1 } ( q _ { i } ( c ) ) ,\tag{55}
$$

where $\phi _ { 0 . 1 } ( u ) = u ^ { 2 } / 0 . 2$ when $| u | < 0 . 1$ , and $| u | - 0 . 0 5$ otherwise. We train the model by minimizing $\mathbb { E } [ E _ { \mathrm { r e s } } ( c _ { T } ) ]$ . In particular, we use AdamW with LR 0.002, weight decay $1 0 ^ { - 5 }$ , default $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ ), epsilon $1 0 ^ { - 8 }$ , gradient-norm clipping at $5 ,$ at most 50 epochs, and earlystopping patience 12. Accumulate gradients over four graphs, with 20 candidate initializations per graph. Graph’s recurrent unroll length $T$ is sampled uniformly from {4, 8, 16, 24, 32}.

Baselines. We retain the four baseline groups used in the Ising and PROTEINS experiments: single-target supervision, unique-equilibrium models, untied feedforward depth controls, and distributional models. Table 12 summarizes their configurations. All models receive the same base graph inputs $x _ { i } , x _ { r } , e _ { i  r } , e _ { r  i }$ and are evaluated on the same 236 test graphs.

• Single-target supervision. We use the same architecture as our multi-attractor model, with the same random initialization, randomized unroll lengths, optimizer, and stopping rule, but replace the residual objective by the concentration-space MSE $s ^ { - 1 } \lVert c _ { T } - c ^ { \mathrm { t a r g e t } } \rVert _ { 2 } ^ { 2 }$ . For each training instance, $c ^ { \mathrm { t a r g e t } }$ is drawn uniformly from the two witnesses once and then held fixed throughout training. The supervised difusion models below use the same label.

• Unique-equilibrium controls. As in the other two experiments, we adopt IGNN [24] and enforce a contraction bound $\rho = 0 . 9 ,$ which guarantees a unique equilibrium. It is trained with the same residual objective $\mathbb { E } [ E _ { \mathrm { r e s } } ]$ as our model. We evaluate hidden widths $H \in$ {16, 32, 64}, which include a capacity-comparable model $\left( H = 3 2 \right)$ and a smaller and a larger variant. Because its equilibrium is uniquely determined by $G ,$ IGNN produces a single candidate per graph; replicated copies are not counted toward diversity.

• Untied feedforward depth controls. We replace the tied recurrence by $B \in \{ 1 , 2 , 3 , 4 , 5 \}$ independently parameterized copies of our update, $z _ { t + 1 } = \mathrm { G N N } _ { \theta _ { t } } ( G , z _ { t } )$ , with distinct parameters $\theta _ { t }$ at each stage. Each stage has its own input encoder, three GATv2 blocks, GRU, physical-feature encoder, and scalar heads. Only the identifier projection and initialization head, which produce $z _ { \mathrm { 0 } } ,$ are shared. The model thus has 3B message-passing blocks and $2 8 9 + 3 5 , 4 2 7 B$ parameters, and it executes exactly B stages at both training and test time. It uses the same random initialization, 20 candidates per graph, residual objective, and optimizer as our model. These controls compare increasing untied depth and parameter count with shared recurrence; inference computation is not matched. None is iterated beyond its trained depth, so no settling rate is reported.

• Distributional models. Since concentrations are continuous, we replace the discrete generators used for Ising and PROTEINS by DDPM [28], DDIM [57], and the time-reversed difusion sampler (DIS) [8]. DDPM is trained with the standard noise-prediction loss against the same fixed witness used for single-target supervision, after mapping log concentrations to the unbounded latent $u = 2 0 \mathrm { a t a n h } ( \log c / 2 0 )$ . It uses 1,000 linear-beta noise levels and 100 reverse steps. DDIM shares the trained DDPM checkpoint and samples deterministically $( \eta = 0 )$ with 100 steps. DIS uses no witness. Instead, it learns to sample from the residual-based density $\propto \exp [ - E _ { \mathrm { r e s } } ( c ) / \tau ]$ with $\tau = 1 0 ^ { - 4 }$ , using 100 steps. The matched setting keeps our width-32, three-block GATv2 backbone, GRU, physical-feature encoder, and heads. The only change is that the input encoder also receives the noisy species state and the normalized difusion time, giving 35,780 parameters compared with 35,716 for our model. It uses the same learning rate, 50-epoch budget, and early-stopping patience as our model. The large setting follows the upstream repositories. DDPM/DDIM use DIFUSCO’s gated residual graph backbone with width 256 and 12 layers (5,205,249 parameters). DIS uses the original FourierMLP controller, conditioned on a width-32, three-block GAT encoder (59,330 parameters), and is trained with its native log-variance loss. Every generator returns 20 candidates per graph after 100 sampling steps.

Table 12. Learned configurations for chemistry. H: hidden width; L: total untied depth.
<table><tr><td>Model</td><td>H</td><td>L Parameters</td><td>LR Epochs</td></tr><tr><td>Ours</td><td>32</td><td>3</td><td>35,716  $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Single-target Supervision</td><td>32</td><td>3</td><td>35,716  $2 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>IGNN (H16)</td><td>16</td><td>3 10,273</td><td>50  $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>IGNN (H32)</td><td>32</td><td>3 37,441</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>IGNN (H64)</td><td>64</td><td>3 142,465</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Feedforward  $( \times 1 )$ </td><td>32</td><td>3 35,716</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Feedforward  $( \times 2 )$ </td><td>32</td><td>6 71,143</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Feedforward  $\left( \times 3 \right)$ </td><td>32 9</td><td>106,570</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Feedforward (×4)</td><td>32 12</td><td>141,997</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>Feedforward (×5)</td><td>32 15</td><td>177,424</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>DDPM / matched</td><td>32 3</td><td>35,780</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>DDIM / matched</td><td>32</td><td>3 35,780</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>DIS / matched</td><td>32 3</td><td>35,780</td><td> $2 \cdot 1 0 ^ { - 3 }$  50</td></tr><tr><td>DDPM / large</td><td>256 12</td><td>5,205,249</td><td> $2 \cdot 1 0 ^ { - 4 }$  200</td></tr><tr><td>DDIM / large</td><td>256 12</td><td>5,205,249</td><td> $2 \cdot 1 0 ^ { - 4 }$  200</td></tr><tr><td>DIS / large</td><td>32/643/4</td><td>59,330</td><td> $5 \cdot 1 0 ^ { - 3 }$  100</td></tr></table>

Metrics. All metrics use the $n = 2 3 6$ test graphs with equal weight per graph. Graph g has $M _ { g }$ candidates $c _ { g m } .$ , with $M _ { g } = 2 0 . \mathrm { ~ A ~ }$ candidate is accepted, $I _ { g m } = 1$ , if it is finite, all its coordinates exceed $1 \bar { 0 } ^ { - 1 2 }$ , and $\bar { R ( c _ { g m } ) } \leq 1 0 ^ { - 3 }$ . Here R and $E _ { \mathrm { r e s } }$ are defined as in the training objective, but are recomputed in float64 from the original $\alpha , \beta ,$ κ rather than from the clipped internal features of the cell. We report

$$
\overline { E } _ { \mathrm { r e s } } = \frac { 1 } { n } \sum _ { g } \frac { 1 } { M _ { g } } \sum _ { m } E _ { \mathrm { r e s } } ( c _ { g m } ) , \qquad \mathrm { H i t } _ { 1 0 ^ { - 3 } } = \frac { 1 0 0 } { n } \sum _ { g } \frac { 1 } { M _ { g } } \sum _ { m } I _ { g m } .
$$

Nonfinite residuals are kept in these means, and acceptance involves no polishing or matching to reference roots. To count distinct predictions, we process the accepted candidates of each graph in saved order. Each one joins the first existing representative within distance ∥ log c − log $\bar { c } ^ { \prime } \| _ { 2 } / \sqrt { s _ { g } } \leq 1 0 ^ { - 3 }$ , or otherwise becomes a new representative. With $K _ { g }$ the resulting number of representatives $( K _ { g } = 0$ if no candidate is accepted), we report

$$
D _ { 2 0 } ^ { \mathrm { r a w } } = \frac { 1 } { n } \sum _ { g } K _ { g } , \qquad G _ { 2 0 } ^ { \mathrm { r a w } } = \frac { 1 0 0 } { n } \sum _ { g } { \bf 1 } \{ K _ { g } \geq 2 \} .\tag{56}
$$

Numerically separated accepted predictions can approximate the same root; these counts therefore do not certify distinct stationary solutions. For recurrent models, we measure numerical

settling at horizon $T$ by

$$
C _ { T } = \frac { 1 0 0 } { n } \sum _ { g } \frac { 1 } { M _ { g } } \sum _ { m } \mathbf { 1 } \left\{ \frac { \lVert z _ { T + 1 } ^ { ( g , m ) } - z _ { T } ^ { ( g , m ) } \rVert _ { 2 } } { \sqrt { s _ { g } } } < 1 0 ^ { - 3 } \right\} ,\tag{57}
$$

and the column $C$ in Table 3 reports $C _ { 2 0 0 }$ . This one-step test on the log-state is independent of the physical residual and does not certify asymptotic convergence. For IGNN, C is instead the percentage of graphs for which all three implicit layers satisfy $\| F _ { \ell , g } ( H _ { \ell , 2 0 0 } ) - H _ { \ell , 2 0 0 } \| _ { \infty , p } <$ $1 0 ^ { - 3 }$ after 200 updates per layer. This measures hidden-state settling, not chemical validity. Feedforward and generative models have no autonomous state update, so $C$ is not reported for them.

Longer test-time dynamics. To investigate the trajectories that fail the $T = 2 0 0$ one-step criterion, we extend the same three frozen multi-attractor models and initialization seeds to $t =$ 20,000. No model weights, initialization distributions, or physical acceptance thresholds change. The main table’s quality and diversity remain evaluated at $T = 2 0 0$ , not at a retrospectively chosen longer horizon.

Table 13. Chemistry test-time dynamics, mean ± population SD across three training seeds. $C _ { t }$ is the fraction with next-step species log-state RMS change below $1 0 ^ { - 3 }$ . Oscillatory denotes detected approximate periods 2–64 among trajectories failing that criterion; unresolved is the remainder. All percentages use the full trajectory population, and the three categories partition that population for $t \geq 2 0 0$ , up to rounding. Dashes denote unperformed early-horizon oscillation tests.
<table><tr><td>t</td><td> $C _ { t } ~ ( \% )$ </td><td>Oscillatory (%)</td><td>Unresolved (%)</td></tr><tr><td>10</td><td> $2 . 2 6 \pm 0 . 6 5$ </td><td></td><td></td></tr><tr><td>20</td><td> $1 3 . 7 7 \pm 3 . 7 2$ </td><td></td><td></td></tr><tr><td>50</td><td> $6 3 . 1 1 \pm 2 . 0 5$ </td><td>—</td><td></td></tr><tr><td>100</td><td> $8 3 . 9 9 \pm 0 . 7 9$ </td><td></td><td></td></tr><tr><td>200</td><td> $9 1 . 9 9 \pm 0 . 3 5$ </td><td> $1 . 7 5 \pm 1 . 5 0$ </td><td> $6 . 2 6 \pm 1 . 2 7$ </td></tr><tr><td>500</td><td> $9 5 . 2 0 \pm 1 . 7 2$ </td><td> $2 . 2 3 \pm 1 . 7 0$ </td><td> $2 . 5 6 \pm 0 . 5 4$ </td></tr><tr><td>1000</td><td> $9 5 . 6 4 \pm 1 . 6 7$ </td><td> $2 . 4 9 \pm 2 . 0 4$ </td><td> $1 . 8 8 \pm 0 . 4 6$ </td></tr><tr><td>2000</td><td> $9 5 . 9 5 \pm 1 . 8 3$ </td><td> $2 . 7 7 \pm 2 . 0 9$ </td><td> $1 . 2 8 \pm 0 . 2 6$ </td></tr><tr><td>5000</td><td> $9 6 . 1 5 \pm 2 . 0 1$ </td><td> $2 . 7 7 \pm 2 . 0 9$ </td><td> $1 . 0 8 \pm 0 . 2 7$ </td></tr><tr><td>10000</td><td> $9 6 . 1 3 \pm 2 . 0 1$ </td><td> $2 . 7 7 \pm 2 . 0 9$ </td><td> $1 . 1 0 \pm 0 . 2 7$ </td></tr><tr><td>20000</td><td> $9 6 . 1 2 \pm 2 . 0 1$ </td><td> $2 . 7 7 \pm 2 . 0 9$ </td><td> $1 . 1 1 \pm 0 . 3 1$ </td></tr></table>

Oscillation test: For $t \geq 2 0 0 .$ , use the 257 saved log states $w _ { j } = z _ { t + j } , j = 0 , \dotsc , 2 5 6$ . Let $A = \operatorname* { m a x } _ { j } \| w _ { j } - \bar { w } \| _ { 2 } / \sqrt { s } .$ . Among trajectories failing the one-step settling criterion, select the smallest $p \in \{ 2 , \ldots , 6 4 \}$ satisfying

$$
\operatorname* { m a x } _ { 0 \leq j \leq 2 5 6 - p } \frac { \| w _ { j + p } - w _ { j } \| _ { 2 } } { \sqrt { s } } \leq \operatorname* { m i n } ( 1 0 ^ { - 3 } , 0 . 0 2 A ) .
$$

Additionally require $A > 0$ , half-window amplitude ratio in [0.8, 1.25], and RMS distance at most $1 0 ^ { - 3 }$ between the first/last cycle centers. Half-window amplitudes are maximum RMS deviations from their respective means over $w _ { 0 } , \ldots , w _ { 1 2 7 }$ and $w _ { 1 2 8 } , \ldots , w _ { 2 5 6 } ;$ cycle centers average the first and last $p$ states. The window contains at least four cycles for every searched period.

Findings: Settling increases from $9 1 . 9 9 \pm 0 . 3 5 \%$ at t = 200 to 96.15 ± 2.01% at $t = 5 { , } 0 0 0$ and then plateaus. $\mathrm { A t } ~ t = 2 0 , 0 0 0 , 9 6 . 1 2 \pm 2 . 0 1 \%$ are step-settled, $2 . 7 7 \pm 2 . 0 9 \%$ are oscillatory, and $1 . 1 1 \pm 0 . 3 1 \%$ are unresolved. The unresolved group contains 157 trajectories across the three seeds. The oscillatory group includes 40 small-amplitude trajectories excluded by the earlier amplitude-gated diagnostic. No nonfinite states occurred. Settling need not increase monotonically because each horizon is tested independently. The log-state box precludes finitestate norm divergence but not bounded nonconvergence. Unresolved does not establish slow convergence, chaos, or divergence: the category can include drift, irregular fluctuations, periods beyond 64, or transients. These observations concern the learned solver, not physical chemical kinetics, and do not establish asymptotic attraction.

## Appendix H. Related Work

GNNs and their theory. Message-passing GNNs are at most as expressive as the 1-WL test [66, 42], and a line of work characterizes which invariant and equivariant maps GNNs can approximate [41, 14, 33, 4, 21]. Unique node identifiers or i.i.d. random node features restore universality with high probability [40, 52, 1]. Recent work extends this to measurable targets with explicit approximation rates [23], and to recurrent GNNs, which with random initialization can uniformly express any polynomial-time computable function on connected graphs [50]. All of these results concern single-valued targets $G \mapsto f ( G )$ , with randomness serving only to tell nodes apart. In our setting, the same random initial state must do double duty: it identifies nodes and selects the solution branch. Only the output-dimensional state $y _ { t }$ is carried between steps, so node distinguishability must be preserved by dynamics that also converge to a solution (Lemma 6, Theorem 2).

Recurrent/equilibrium GNNs. The original GNN was defined as the fixed point of a contraction [53]. Later weight-tied models unroll a shared update [35, 17], while several implicit GNNs impose suficient conditions for equilibrium existence and uniqueness, including contraction and strong monotonicity [24, 39, 47, 6]. Yang et al. [67] also analyze unfolded GNN regimes with stationarity guarantees that do not require uniqueness. Deep equilibrium models compute fixed points [5], with monotone variants guaranteeing uniqueness [63]; path independence, meaning convergence to the same state across initializations, has also been advocated [3]. Weight-tied message passing is also used for algorithmic reasoning and size extrapolation [62, 60, 7] and for SAT and constraint satisfaction [55, 61]. RUN-CSP in particular starts from random states and keeps the best of several parallel runs [61], but without a representational account of the solutions reached. On the theory side, Liu et al. [38] show that regular implicit operators (contractive in the state, Lipschitz in the input) represent exactly the locally Lipschitz maps, so that iterations rather than parameters supply expressivity. Jore and Liu [30] represent set-valued maps through attractor landscapes and demonstrate multi-solution discovery with recurrent GNNs on Ising problems, but do not establish permutation-equivariant dynamics or their message-passing realization. We develop this theory on graphs, where symmetry changes the picture in three ways: (i) the solution set is equivariant although individual branches need not be (Section 2.1); (ii) our message-passing construction realizes the dynamics through symmetry breaking carried by the state itself, without auxiliary inputs (Theorem 2, Lemma 6); and (iii) continuous, equivariant, globally contractive updates can miss every valid solution with positive probability (Theorems 3 and 6). As in Liu et al. [38], complexity comes from iteration: each step of the ideal update is globally Lipschitz, while its limit selector jumps across basin boundaries (Theorem 1). Multiple attractors have long served as a resource in Hopfield networks [29, 48], where they store fixed patterns rather than solutions of an input-dependent problem.

Distributional models. Another way to handle multiple solutions is to learn a distribution over them. Unsupervised neural combinatorial optimization trains GNNs from an energy, either as mean-field relaxations [32, 54] or as annealed, autoregressive, difusion, or GFlowNet samplers

[58, 51, 69]; variational autoregressive networks play the same role for statistical-mechanics models [64]. Supervised difusion solvers and difusion samplers generate candidates through time-inhomogeneous reverse processes [59, 28, 57, 8]. Other work keeps supervision but relaxes the single target: multiple output heads with a hindsight loss [36], learned selection among oneof-many solutions [45], label realignment under formulation symmetry [13], and equivariant flow matching with symmetry-aware target matching for bifurcation problems [27]. Our approach learns a single autonomous update whose attracting limits are intended to approximate valid solutions. Numerical stationarity and task validity are evaluated separately; iterations can be extended at test time without retraining, and training uses energies or residuals rather than solution labels. Classical multi-solution methods such as deflation and homotopy continuation [20, 10] operate instance by instance, whereas our model is trained once and amortized over a graph family.

Symmetry breaking. Deterministic equivariant networks cannot map a symmetric input to a less symmetric output [56, 31]. Existing remedies break symmetry through augmented inputs: random or positional node features [52, 1, 18], symmetry-breaking sets and objects [56, 65, 22], and probabilistic symmetry breaking through equivariant conditional distributions and sampled canonicalizations [34], the notion closest to our set-valued equivariance. We instead break symmetry through the model’s own evolving state, with no auxiliary input: randomness enters once through $y _ { 0 } ,$ and Lemma 6 shows that the ideal dynamics preserve node distin guishability at every finite step. The two mechanisms are compatible. Augmented features can be appended to $X$ or used to seed $y _ { 0 }$ (our PROTEINS model uses random-walk encodings, and our chemistry model seeds $y _ { 0 }$ from random node vectors), and the construction in Lemma 5 only needs distinct node codes, whatever their source. Augmentation alone, however, does not settle which solution the network should output: deterministic encodings alone do not provide a mechanism for sampling multiple valid outputs, and continuous canonicalization is impos sible in general [19]. Theorems 3 and 6 concern updates without auxiliary inputs; combining multi-attractor dynamics with augmented features is a natural direction for future work.

(JL) School of Data, Mathematical, and Statistical Sciences, University of Central Florida, Orlando<sub>,</sub> FL 32826.

Email address: jialin.liu@ucf.edu