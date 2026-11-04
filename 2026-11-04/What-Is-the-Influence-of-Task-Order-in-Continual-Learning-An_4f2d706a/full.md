# What Is the Influence of Task Order in Continual Learning? An Empirical Study and Order-Aware Evaluation for Motion Forecasting

Zirui Li, Member, IEEE, Yunlong Lin, Haohan Yang, Xiaocong Zhao, Member, IEEE, and Chen Lv, Senior Member, IEEE

Abstract—Aggregate scores for continual learning algorithms rest on a distribution over task orders that is rarely stated. We formulate algorithm comparison as an expectation over a declared order distribution, which separates training variation from order variation, and apply it to motion forecasting with four road-layout tasks from the INTERACTION dataset, all 24 task orders, four algorithms, and three training replicates (288 runs). Task order accounts for 65% to 91% of the variation in current-task error and backward transfer, and 90% to 99% of this order effect comes from the identity of the starting task: the smallest task cannot be learned from a cold start. Beyond the first stage, the order effect is comparable to training variability. In most orders, the leading algorithm in current-task error and backward transfer is within one replicate standard deviation of the runner-up. Backward transfer depends on the order mainly through its reference term, which weights the first task most heavily, so relative improvement and final error identify different leading algorithms for displacement error. A transitiondirection descriptor computed from task statistics tracks the position of the smallest task, and weighting orders by it leaves the algorithm ranking unchanged. These results call for benchmarks that declare their order distribution and report order-resolved, starting-task-resolved, warm-start, and final errors alongside aggregate scores. The code and data are available here.

Index Terms—Benchmark testing, continual learning, evaluation protocol, motion forecasting, task order.

## I. INTRODUCTION

ELF-DRIVING vehicles plan their motion around fore-S casts of where surrounding road users will go. A forecaster trained on urban merges meets highway ramps, intersections, and roundabouts as the fleet expands, and must keep learning in an order no designer controls. Continual learning (CL) provides this capability and is moving into safety-relevant applications. Human-guided CL already uses small amounts

## a Order changes the algorithm ranking

![](images/0ebbb6bd0bb1dbcad1db2371b9292da36cb3f9d043119f4733a3a6d640963ae9.jpg)

![](images/6857ac3576707e5814138e26e32c8b78f866925ee9a378180039abea4fe9a864.jpg)  
Fig. 1. Task order and algorithm comparison. (a) Current-task final displacement error (FDE-CT, m), averaged over three replicates, and algorithm ranking for two orders that differ only in their final two tasks. In both orders the leader lies within one pooled replicate standard deviation of the runnerup. (b) A common order policy q(o), uniform or weighted by the transition descriptor D(o), aggregates the order-resolved outcomes of every algorithm into a uniform mean or weighted score, shown with the range across orders.

of driving guidance to improve self-driving performance in dimensions such as social compliance and rare-case handling [1], and continual forecasting methods adapt motion predictors across changing traffic scenarios [2]–[6]. Learning a new task can overwrite earlier knowledge [7]–[9]. Replay, regularisation, and gradient constraints counter this forgetting through different retention mechanisms [10]–[14], so their outcomes depend on the learning history fixed by the task order. Yet the algorithm to deploy is chosen from benchmark rankings. If a ranking reverses when the same tasks arrive in another sequence, that choice hinges on an arbitrary benchmark order.

Because outcomes depend on the order in which tasks arrive, any aggregate score for a CL algorithm rests on a distribution over task orders. Evaluating continual forecasting under varying data-stream sequences has been recommended [15], yet many protocols leave this distribution implicit, repeating one or a few prescribed orders while varying initialisation or within-task sample order [5], [14]. Studies of order-dependent forgetting, order-wise performance disparity, ordering principles, and sequence transferability show that task transitions influence retention and transfer [16]–[19], and use this to explain the outcomes of particular orders, quantify the disparity among them, or select favourable ones. How the variation of a metric across all N! orders of N tasks, several algorithms, and repeated training runs divides between order, algorithm, their interaction, and training randomness, and which property of an order carries the order effect, remain open.

This paper answers both questions for a fixed task set and tests whether algorithm rankings persist across orders. We formulate algorithm comparison as an expectation over an explicit order policy, with the uniform policy as the equalorder reference, and decompose the sum of squares of each metric into order, algorithm, interaction, and training components, splitting the order term by starting task. A replicate bootstrap attaches a probability to every leading algorithm. A warm-start current-task error and a decomposition of backward transfer into subsequent error and initial reference isolate the first learning stage. A transition-direction descriptor computed from task statistics defines a descriptor-weighted policy, an example of a nonuniform policy shared by all algorithms (Figs. 1(b) and 2).

The experiments use the UQnet endpoint-heatmap backbone [20] and four road-layout tasks from the INTERACTION dataset: Urban Merge (A), Highway Merge (B), Intersection (C), and Roundabout (D). Vanilla sequential fine-tuning and three memory-based algorithms learn the tasks in all 4! = 24 orders, with three training replicates per order and algorithm (288 runs). The memory-based algorithms are dark experience replay (DER), synergetic memory rehearsal (SyReM), and gradient scenario memory (GSM). In Fig. 1(a), A→B→C→D and A→B→D→C differ only in their final two tasks, yet give exactly reversed rankings of the four algorithms by current-task final displacement error. Section VI-A traces both rankings to run-to-run variation in the cold-start error, which arises on the shared first task before the two orders diverge. The main contributions are summarised as follows.

1) Formulation and benchmark. An expectation over an explicit order policy defines a common comparison target, and the law of total variance separates within-order training variation from across-order variation. All 24 orders are enumerated with released per-run results, and the decomposition, adjustment, and bootstrap procedures form a reusable analysis toolkit.

2) Empirical finding. Task order accounts for 65% to 91% of the variation in current-task error and backward transfer, and 90% to 99% of this order effect comes from the identity of the starting task, acting through the cold start of the smallest task. Beyond the first stage, the residual order effect is comparable to training variability. The per-order leader’s gap to the runner-up exceeds the pooled replicate standard deviation in only 2 to 11 of the 24 orders for current-task error and backward transfer, against 16 or 17 for final error.

3) Metric diagnosis. The first task enters the reference term of stage-averaged backward transfer with weight 11/18 for four tasks, so a poor first fit simultaneously raises current-task error and drives backward transfer towards large negative values. Relative improvement and absolute final error therefore identify different leading algorithms for displacement error. The transitiondirection descriptor tracks the position of the smallest task, and the descriptor-weighted policy leaves the full ranking unchanged.

## II. RELATED WORK

## A. Task Order and Curriculum Effects

Curriculum learning investigates the effect of example or task presentation on optimisation [21]. In CL, ordering also affects retention of previous knowledge. Bell and Lawrence [16] examine order-dependent forgetting and task-distance heuristics, and Li and Hiratani [18] derive ordering principles in a teacher–student model. Analyses of task similarity and overparameterisation further characterise interference [22]. Nguyen et al. [23] relate error rates to the total complexity of a task sequence and to its sequential heterogeneity, the sum of distances between consecutive tasks. The path length L in Section IV-D is such a sum.

Yoon et al. [17] quantify order-wise performance disparity, establishing order fairness as an evaluation criterion. Measures of sequence transferability relate task transitions to performance using learned predictors [19]. The present analysis uses marginal task statistics to define a shared order policy and asks how much of an observed ranking survives a change of order.

## B. Continual Learning for Motion Forecasting

Continual forecasting addresses adaptation across changing traffic environments [15] through generative replay [24]– [26], regularisation under distribution shift [27], and task-free streams without supplied task boundaries [2], [5], [28].

D-GSM uses traffic divergence and scenario-organised memory to constrain gradient updates [3], H2C combines diversity-based selection with equiprobable replay [5], SyReM integrates a memory-loss constraint with gradient-based rehearsal selection [4], and Dual-LS coordinates long- and shortterm knowledge through complementary rehearsal [6]. These studies evaluate each method along one or a few task orders, so the dependence of the resulting ranking on the order is still unexamined.

## C. Evaluation Protocols for Continual Learning

Average performance over all tasks, current-task performance, and backward transfer are the standard quantities of continual learning evaluation [12], [29], and the appropriate protocol differs between task-, domain-, and class-incremental settings [30]. Early protocol studies showed that reported forgetting depends on the evaluation design, including the task sequences used [31], [32], and proposed metric sets beyond final accuracy [33]. Hyperparameter selection on the evaluation scenario can bias comparisons [34], and unequal update costs can alter conclusions about online CL [35]. Recent benchmarks treat the order either as an adversarial choice to be searched [36] or as a sampled factor whose uncertainty is reported [37].

![](images/d0fa7b193068c7d4de08457be28a9a35ee5fbe29fb2aa7799e0a0f42563d5654.jpg)

![](images/239fc674f3e87e5e7de51786befd03f5da922e8d50a908cec4ab7e4a85e0424d.jpg)  
Fig. 2. Order-aware evaluation workflow. (1) Tasks differ in size and speed distribution. (2) Each algorithm learns every order, starting cold in the first stage. (3) Testing all tasks after each stage gives the stage–task errors ${ \bar { R } } _ { r , i } ,$ summarised by CT, $\bar { \mathrm { C T _ { 2 : N } } }$ $\mathrm { B W T } = P { \overset { \cdot } { - } } B$ , and the final error. (4) W1 speed distances between consecutive tasks, $\bar { \Delta } _ { r } ,$ define the descriptor $D ( o )$ . (5) A common order policy q(o) yields policy scores, and a sum-of-squares decomposition (FDE-CT) and a replicate bootstrap assess them. Networks, policy weights, and comparison marks are schematic.

Order coverage is therefore a protocol choice. We distinguish repeated training within an order from evaluation across orders, separate the starting task from the remaining order, and assess backward transfer alongside initial, warm-start, and final error.

## III. PROBLEM FORMULATION

## A. Sequential Learning Conditioned on Tasks

Let $[ N ] = \{ 1 , \dots , N \}$ index a task set T . Task j is specified by a distribution $\mathcal { D } _ { j }$ on $\mathcal { X } \times \mathcal { V }$ and a finite training set $\mathcal { S } _ { j } = \{ ( \mathbf { X } _ { j k } , \mathbf { Y } _ { j k } ) \} _ { k = 1 } ^ { n _ { j } ^ { - } }$ . The tasks share the input and output spaces, while their data distributions may differ. An order (task sequence) $o = \left( o _ { 1 } , \ldots , o _ { N } \right)$ belongs to the permutation group ${ \mathfrak { S } } _ { N }$ , with $| \mathfrak { S } _ { N } | \ = \ N !$ . Evaluation uses a common subset $\mathcal { O } \subseteq \mathfrak { S } _ { N }$ for every algorithm.

For algorithm $a \in { \mathcal { A } } .$ let $z _ { r } ^ { ( o , a , s ) }$ denote its state after stage r in training replicate s, comprising model parameters, optimiser state, and algorithm-specific persistent information such as the replay memory $\mathcal { M } _ { r } ^ { ( o , a , s ) } \subseteq \bar { z } _ { r } ^ { ( o , a , s ) }$ . Sequential learning is represented by

$$
\begin{array} { r } { z _ { r } ^ { ( o , a , s ) } = \mathscr { U } _ { a } \left( z _ { r - 1 } ^ { ( o , a , s ) } , S _ { o _ { r } } , \xi _ { r } ^ { ( o , a , s ) } \right) , } \end{array}\tag{1}
$$

where $\mathcal { U } _ { a }$ is the update operator and $\xi _ { r } ^ { ( o , a , s ) }$ denotes the stochastic inputs for that run and stage. The initial state is likewise run-specific. The memory satisfies $| \mathcal { M } _ { r } ^ { ( o , a , s ) } | \leq B _ { a }$ for a record capacity $B _ { a }$ , a record may include a stored model output, and historical training examples are accessible only through this memory. Task boundaries are provided, and all tasks share one prediction head.

## B. Risk and Knowledge Retention

For loss ℓ and predictor $f _ { \pmb { \theta } } ,$ , the risk on task $j$ is

$$
\begin{array} { r } { \mathcal { R } _ { j } ( \pmb { \theta } ) = \mathbb { E } _ { ( \mathbf { X } , \mathbf { Y } ) \sim \mathcal { D } _ { j } } [ \ell ( f _ { \pmb { \theta } } ( \mathbf { X } ) , \mathbf { Y } ) ] . } \end{array}\tag{2}
$$

A reference objective for stage $r > 1$ is

$$
\pmb { \theta } _ { r } ^ { \star } \in \arg \operatorname* { m i n } _ { \pmb { \theta } } \left\{ \mathcal { R } _ { o _ { r } } ( \pmb { \theta } ) + \frac { \lambda _ { r } } { r - 1 } \sum _ { i < r } \mathcal { R } _ { o _ { i } } ( \pmb { \theta } ) \right\} ,\tag{3}
$$

where $\lambda _ { r } ~ \geq ~ 0$ balances current-task fit and past-task risk. Stage 1 contains only the current-task term, so the first stage is single-task learning from the initial state. Replay, regularisation, and gradient constraints approximate the historical risk through different mechanisms [10]–[12], [14]. Absolute historical risk and its change since initial learning are distinct quantities, which final error and backward transfer measure separately.

## C. Evaluation over Task Orders

Let $Z _ { m } ( o , a , \xi )$ be the scalar outcome of a complete run under metric m, where ξ collects initialisation and subsequent training randomness for the specified order and algorithm. For an order policy q satisfying $q ( o ) \geq 0$ and $\begin{array} { r } { \sum _ { o \in \mathcal { O } } q ( o ) = 1 } \end{array}$ , the comparison target is

$$
\mu _ { m } ( a , q ) = \sum _ { o \in \mathcal { O } } q ( o ) \mathbb { E } _ { \xi } [ Z _ { m } ( o , a , \xi ) ] .\tag{4}
$$

For S training replicates, define $y _ { m } ( o , a )$ = $\begin{array} { r } { S ^ { - 1 } \sum _ { s = 1 } ^ { S } Z _ { m } \big ( o , \bar { a } , \xi ^ { ( o , \bar { a } , s ) } \big ) } \end{array}$ . The empirical policy score is

$$
{ \widehat { \mu } } _ { m } ( a , q ) = \sum _ { o \in { \mathcal { O } } } q ( o ) y _ { m } ( o , a ) .\tag{5}
$$

All algorithms share $\mathcal { O } , q .$ , and the metric definition. For fixed a, m, and $q ,$ assume $Z _ { m }$ has a finite second moment and draw $o \sim q .$ . The law of total variance separates the two sources of variability:

$$
\begin{array} { r l } & { \mathrm { V a r } ( Z _ { m } ) = \mathbb { E } _ { o \sim q } [ \mathrm { V a r } _ { \xi } ( Z _ { m } \mid o ) ] } \\ & { \quad \quad \quad + \mathrm { V a r } _ { o \sim q } ( \mathbb { E } _ { \xi } [ Z _ { m } \mid o ] ) . } \end{array}\tag{6}
$$

The first term describes stochastic training variation within orders, and the second describes variation in expected performance across orders. Repeating a single order estimates only the first component, so order-aware evaluation combines an explicit order policy with order-resolved outcomes.

For algorithms a and $b ,$ let $\delta _ { m } ( o , a , b ) = y _ { m } ( o , a ) - y _ { m } ( o , b )$ be the measured difference. Equation (5) gives

$$
{ \widehat { \mu } } _ { m } ( a , q ) - { \widehat { \mu } } _ { m } ( b , q ) = \sum _ { o \in { \mathcal { O } } } q ( o ) \delta _ { m } ( o , a , b ) .\tag{7}
$$

A common sign of $\delta _ { m }$ across orders preserves the weak ranking under every policy supported on O, whereas opposite signs permit a reversal under policies concentrating on different orders. Order-wise differences thus distinguish uniform dominance from dependence on the evaluation distribution. Section IV-D quantifies how replicate variability affects such comparisons.

## IV. ORDER-AWARE EVALUATION

## A. Distances Between Task Distributions

We describe each task through empirical distributions of target-agent speed and acceleration. Speed is the norm of the two velocity components at each of nine historical time steps, and acceleration is the adjacent speed difference divided by 0.1 s, retaining values in $[ - 5 , 5 ] \mathrm { m } \bar { / \mathrm { s } } ^ { 2 }$ . Observations are pooled over time steps and trajectories within each task.

The candidate measures probe complementary aspects of distribution shift. Wasserstein-1 (W1) quantifies displacement of probability mass along the physical feature axis [38]. Kullback–Leibler (KL) divergence measures directed probability mismatch, whereas Jensen–Shannon (JS) divergence provides a symmetric, bounded comparison of the same histograms [39]. Maximum mean discrepancy (MMD) compares kernel mean embeddings without histogram discretisation [40]. Applying all four measures to speed holds the feature representation fixed while varying the notion of discrepancy, whereas applying W1 additionally to acceleration examines a different aspect of motion with the same distance functional.

For feature $f \in$ {speed, acceleration}, let $F _ { i } ^ { f }$ denote its empirical cumulative distribution function in task i. We compute

$$
d _ { i j } ^ { \mathrm { W } } ( f ) = \int _ { \mathbb { R } } | F _ { i } ^ { f } ( t ) - F _ { j } ^ { f } ( t ) | \mathrm { d } { t } ,\tag{8}
$$

using every retained observation. The distance is expressed in m/s for speed and $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$ for acceleration.

KL and JS divergences are computed on $H = 1 0 0$ shared equal-width speed histograms with additive smoothing, and MMD on a fixed standardised sample of 2,000 speed observations per task with a Gaussian kernel (Appendix A-A). W1, JS, and MMD are symmetric, whereas KL is directed.

The primary distance is W1 on speed, which has an interpretable physical scale. It was chosen in an exploratory scan of 20 distance representations and 15 aggregation operators against the CL outcomes (Appendix A-B), from which the five candidates above are reported.

## B. A Descriptor of Transition Direction

For order o, let $\Delta _ { r } ( o ) \ = \ d _ { o _ { r } , o _ { r + 1 } }$ be the distance at the rth transition. The descriptor is its least-squares slope against transition position:

$$
D ( o ) = \frac { \sum _ { r = 1 } ^ { N - 1 } ( r - \bar { r } ) \Delta _ { r } ( o ) } { \sum _ { r = 1 } ^ { N - 1 } ( r - \bar { r } ) ^ { 2 } } , \qquad \bar { r } = N / 2 .\tag{9}
$$

It is defined for $N \geq 3$ . For the four-task experiment this reduces exactly to

$$
D ( o ) = \frac { d _ { o _ { 3 } , o _ { 4 } } - d _ { o _ { 1 } , o _ { 2 } } } { 2 } .\tag{10}
$$

The middle transition has zero coefficient, and a positive value means that the final transition is larger than the first. We therefore call D a transition-direction descriptor.

For symmetric distances, $D ( o ^ { \mathrm { r e v } } ) ~ = ~ - D ( o )$ , where $o ^ { \mathrm { { r e v } } }$ reverses the order. There are only three partitions of four tasks into two unordered pairs, and each partition gives two signed contrasts, so (10) takes at most six distinct values. Swapping the tasks within either endpoint pair leaves D unchanged, so each value is shared by four orders. The slope is equivariant to a common positive affine transformation of the distances, $d _ { i j } ^ { \prime } = a d _ { i j } + b$ with $a > 0$ giving $D ^ { \prime } ( o ) = a D ( o )$ , so its rank and the min–max weights below are unchanged by a common linear rescaling.

In this task set, the first and last transitions of any order form one of the three pair partitions, and in every partition the pair containing task A has the larger speed distance $( d _ { A B } = 3 . 2 5 >$ $d _ { C D } = 1 . 3 6 , d _ { A C } = 2 . 6 3 > d _ { B D } = 2 . 1 5$ , and $d _ { A D } = 2 . 1 2 >$ $d _ { B C } = 1 . 7 4 ~ \mathrm { m / s } )$ . The sign of D in (10) is therefore set by the position of task A: $D ( o ) < 0$ exactly when A occupies position 1 or 2, and the Spearman correlation between D and the position of A is 0.79. Section VI-C accordingly examines associations with both.

## C. Uniform and Descriptor-Weighted Policies

The uniform policy $q ( o ) = 1 / | \mathcal { O } |$ assigns equal probability to each evaluated order and gives the equal-order reference

$$
\bar { y } _ { m } ( a ) = \frac { 1 } { | \mathcal { O } | } \sum _ { o \in \mathcal { O } } y _ { m } ( o , a ) .\tag{11}
$$

We report it together with the order-resolved means, ranks, and replicate spread.

The descriptor-weighted policy rescales the descriptor to the unit interval, with $D _ { \mathrm { m i n } } = \mathrm { m i n } _ { o \in { \mathcal O } } D ( o )$ and $D _ { \mathrm { m a x } } =$ $\operatorname* { m a x } _ { o \in { \mathcal { O } } } D ( o )$

$$
\widetilde D ( o ) = \frac { D ( o ) - D _ { \mathrm { m i n } } } { D _ { \mathrm { m a x } } - D _ { \mathrm { m i n } } } .\tag{12}
$$

If D is constant, the weights are uniform and the weighted result equals (11). Otherwise, following the directions of

association in Section VI-C, the common weights for the current-task error (CT) and backward transfer (BWT), both defined in Section V-C, are

$$
w _ { \mathrm { C T } } ( o ) = 1 - \widetilde { D } ( o ) , \qquad w _ { \mathrm { B W T } } ( o ) = \widetilde { D } ( o ) ,\tag{13}
$$

and, for each metric separately,

$$
\bar { y } _ { m } ^ { w } ( a ) = \frac { \sum _ { o } w _ { m } ( o ) y _ { m } ( o , a ) } { \sum _ { o } w _ { m } ( o ) } .\tag{14}
$$

The normalised weights define $\begin{array} { r } { q _ { m } ( o ) = w _ { m } ( o ) / \sum _ { o ^ { \prime } } w _ { m } ( o ^ { \prime } ) } \end{array}$ in (5). Because the complete order set contains the reversal of every order, a nonconstant symmetric-distance descriptor gives $D _ { \operatorname* { m i n } } = - D _ { \operatorname* { m a x } } .$ , so both weight vectors sum to $| { \mathcal { O } } | / 2 .$ . The policy emphasises the descriptor regions associated with larger current-task errors and less favourable backward transfer.

## D. Analysis Protocol

The decomposition and the bootstrap operate on individual runs, and the other analyses use the replicate means $y _ { m } ( o , a )$

Decomposition. For each metric, the total centred sum of squares over all runs is split into order, algorithm, order– algorithm interaction, and within-cell training components (Appendix A-C). The order component is further split into starting-task identity and order within starting task, and the decomposition is repeated for the 18 orders that do not start with task A.

Warm-start error. To separate the first stage, which involves no previous task, from the continual part of a run, the currenttask error is also reported without its first term (Section V-C).

Associations. Spearman correlations with the 16 algorithm– metric series use average ranks for ties. Starting-task dependence is assessed by excluding A-first orders and by correlating ranked variables after removing starting-task group means. The same residualisation with position-of-A group means isolates what D adds beyond the position of task A.

Replicate bootstrap. The three replicates of every order– algorithm cell are resampled with replacement (5,000 draws, fixed seed), and the order means, policy scores, and per-order leaders are recomputed. This yields the probability that each algorithm attains the smallest policy score and an interval for its number of per-order wins. A per-order leader is called separable when its gap to the runner-up exceeds the pooled sample standard deviation of the two algorithms. Because the 24 orders form the complete order set, order sampling adds no uncertainty, and the resampling reflects training variation alone (Appendix A-C).

Paths. We also compare the orders with the largest and smallest total transition distance,

$$
L ( o ) = \sum _ { r = 1 } ^ { N - 1 } d _ { o _ { r } , o _ { r + 1 } } ,\tag{15}
$$

within each fixed starting task. Path length and slope are different summaries: L uses all three transitions, whereas $D$ omits the middle one.

![](images/9b64b5694bf75cce7142779dcf2160fd27536d6dfa508db54ccae9e016fdb110.jpg)

(a) Target-agent speed (m/s)  
![](images/d016e1192e8cfc294931a7d907b17544dbf3a3a0efac7330a7be99b19a1c6195.jpg)  
(b) Acceleration (m/s²)  
Fig. 3. Task feature distributions. Dots, bands, and whiskers indicate medians, interquartile ranges, and 5th–95th percentiles. Acceleration is restricted to [−5, 5] m/s<sup>2</sup>.

## V. EXPERIMENTAL SETUP

## A. Motion Forecasting Tasks and Backbone

Each forecasting task comprises trajectory records from one category of road layout in INTERACTION [41]. Twelve location subsets are grouped into four tasks. Task A, Urban Merge, covers lane convergence and zipper merging at the German MT site. Task B, Highway Merge, covers ramp entry and lane convergence at the Chinese ZS highway site (archived subsets ZS0 and ZS2). Task C, Intersection, covers crossing and turning interactions at the US subsets MA, GL, EP0, and EP1. Task D, Roundabout, covers entry, circulation, and exit at EP, SR, and FT in the US, OF in Germany, and LN in China. The task arrays contain 5,378, 50,905, 134,269, and 103,540 trajectory records, so task A is 10 to 25 times smaller than the others.

Fig. 3 compares the locations and spreads of the empirical target-agent feature distributions. Urban merging has the highest median speed (6.29 m/s), whereas highway merging combines a lower median (2.87 m/s), reflecting queuing on the ramp, with the largest 95th percentile (17.56 m/s). Intersections have the lowest median speed (2.27 m/s), and intersections and roundabouts have the wider acceleration interquartile ranges.

The forecasting input is $\mathbf { X } = ( \mathbf { X } _ { \mathrm { T A } } , \mathbf { X } _ { \mathrm { S A } } , \mathbf { M } )$ , containing the target-agent history, surrounding-agent histories, and map context. The ground-truth output $\textbf { Y } ~ \in ~ \mathbb { R } ^ { 2 }$ is the target endpoint. UQnet [20] combines agent and map encoders with a spatial prediction head to produce a heatmap over future endpoints, from which the predictor samples W candidate endpoints $\{ \hat { \mathbf { Y } } _ { w } \} _ { w = 1 } ^ { W }$ . The training loss follows the heatmapbased forecasting formulation in the backbone work.

Fig. 4 details the encoder and prediction branches. Lane attention encodes the map into ${ \bf { H } } _ { \mathrm { { M } } }$ , a multilayer perceptron embeds spatial coordinates as $\operatorname { H } _ { \mathrm { { C } } } .$ and a trajectory subnet extracts target-vehicle and surrounding-agent features, $\mathbf { H } _ { \mathrm { T V } }$ and $\mathbf { H } _ { \mathrm { S } }$ . The interaction module fuses vehicle and map features, which the decoder combines with coordinate features

![](images/24cc17ac8a9b1149a0feafa4b81dffbbfd5580d0fece7fa8dc0f7ac4bc30704a.jpg)  
Fig. 4. UQnet backbone (adapted from [20]): (a) encoder, (b) regularisation network, used during training and removed at inference, and (c) predictor, which decodes fused map and vehicle features with position features into an endpoint heatmap.

## TABLE I

PUBLISHED MECHANISMS OF THE COMPARED ALGORITHMS. STORED EXAMPLES SERVE OUTPUT MATCHING, REHEARSAL, OR GRADIENT CONSTRAINTS, AND SCENARIO PARTITION ALLOCATES MEMORY BY PAST TASK.
<table><tr><td>Algorithm</td><td>Stored examples</td><td>Output matching</td><td>Scenario partition</td><td>Gradient constraint</td><td>Gradient-based selection</td></tr><tr><td>Vanilla</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DER</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>SyReM</td><td>√</td><td>X</td><td>×</td><td>√</td><td>√</td></tr><tr><td>GSM</td><td>√</td><td>X</td><td>√</td><td>√</td><td>X</td></tr></table>

to predict an endpoint heatmap. The regularisation network applies two cross-attention blocks to map, target-vehicle, and coordinate features during training and is removed at inference [20]. All compared algorithms use this predictor, and evaluation concerns the predicted endpoints.

## B. Algorithms and Experimental Coverage

We compare four algorithms under the common prediction architecture (Table I). Vanilla optimises the current task without memory and provides the reference for unconstrained adaptation. DER (dark experience replay) [14] rehearses stored inputs with output matching to preserve previous predictions, in a forecasting adaptation of the original classification method. SyReM (synergetic memory rehearsal) [4] combines gradient-based rehearsal selection with an average-memory loss constraint. GSM [3] is the equal-allocation variant of dynamic gradient scenario memory (D-GSM), with scenarioorganised memory (a scenario is one of the four tasks here) and gradient constraints derived from previous scenarios. The selection spans output distillation, selective constrained rehearsal, and scenario-wise constraints, and all memory-based algorithms use a capacity of 2,000 records.

The experiment evaluates all $4 ! = 2 4$ orders of $N = 4$ tasks, with four algorithms and three training replicates per combination (288 runs). Testing every task after each stage yields 1,152 stage logs and 4,608 stage–task records. The evaluation data and analysis code are available online.<sup>1</sup>

## C. Forecasting and Continual Learning Metrics

Minimum final displacement error (FDE) measures the distance from the ground-truth endpoint to the closest predicted candidate:

$$
\mathrm { F D E } = \operatorname* { m i n } _ { 1 \leq w \leq W } \| \hat { \mathbf { Y } } _ { w } - \mathbf { Y } \| _ { 2 } .\tag{16}
$$

Miss rate (MR) reports the percentage of evaluated cases whose closest candidate lies beyond the endpoint tolerance of the evaluation protocol.

For either error metric, let $E _ { r , j } ^ { ( o , a , s ) }$ denote the result on task identity j after stage r. We reorder columns by presentation position and define

$$
\begin{array} { r } { R _ { r , i } ^ { ( o , a , s ) } = E _ { r , o _ { i } } ^ { ( o , a , s ) } . } \end{array}\tag{17}
$$

Thus $R _ { r , r }$ always evaluates the task just learned, even when its identity is not r. Suppressing the superscripts, the summaries are

$$
\mathrm { C T } = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } R _ { r , r } ,\tag{18}
$$

$$
\mathrm { B W T } = \frac { 1 } { N - 1 } \sum _ { r = 2 } ^ { N } \frac { 1 } { r - 1 } \sum _ { i = 1 } ^ { r - 1 } ( R _ { r , i } - R _ { i , i } ) ,\tag{19}
$$

$$
{ \mathrm { F i n a l } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } R _ { N , i } .\tag{20}
$$

The current-task error CT measures error upon initial learning of each task. The backward transfer BWT averages subsequent error changes over stages: positive values indicate degradation and negative values improvement relative to the initial fit, reversing the accuracy-based sign convention. FDE is in metres, MR in percent, and MR-BWT in percentage points. Lower CT and Final indicate better absolute predictions, and lower BWT indicates greater improvement relative to the initial fit.

The first term of CT, $R _ { 1 , 1 } { \mathrm { : } }$ , is the error of single-task learning from the initial state and involves no previous task. We therefore also report the warm-start current-task error

$$
\mathrm { C T } _ { 2 : N } = \frac { 1 } { N - 1 } \sum _ { r = 2 } ^ { N } R _ { r , r } ,\tag{21}
$$

so that $\mathrm { C T } = [ R _ { 1 , 1 } + ( N - 1 ) \mathrm { C T } _ { 2 : N } ] / N$ . We write FDE-CT, MR-BWT, and so on for the metric–summary combinations.

## VI. EXPERIMENTAL RESULTS

## A. Order-Resolved Algorithm Comparisons

Tables III and IV report all orders. Each entry is a mean and sample standard deviation across three replicates, and bold marks the per-order minima that are separable from the runnerup.

Table II decomposes the total centred sum of squares. Task order accounts for 71.3%, 64.6%, 76.3%, and 91.3% in FDE-CT, MR-CT, FDE-BWT, and MR-BWT, against 1.3%, 1.7%,

TABLE II  
PERCENTAGE OF THE TOTAL CENTRED SUM OF SQUARES PER COMPONENT. THE ORDER TERM IS SPLIT INTO STARTING-TASK IDENTITY (START) AND ORDER WITHIN STARTING TASK, AND THE LAST THREE COLUMNS GIVE THE 18 ORDERS THAT DO NOT START WITH A.
<table><tr><td></td><td colspan="5">All 24 orders (%)</td><td colspan="3">Excl. A-first (%)</td></tr><tr><td></td><td colspan="2">Order|</td><td colspan="3">Order×</td><td colspan="3"></td></tr><tr><td>Metric</td><td>Start</td><td>start</td><td>Alg.</td><td>alg.</td><td>Within</td><td>Order</td><td>Alg.</td><td>Within</td></tr><tr><td>FDE-CT</td><td>69.6</td><td>1.7</td><td>1.3</td><td>11.3</td><td>16.0</td><td>30.6</td><td>2.3</td><td>50.8</td></tr><tr><td>MR-CT</td><td>58.3</td><td>6.3</td><td>1.7</td><td>10.3</td><td>23.4</td><td>24.5</td><td>2.4</td><td>51.2</td></tr><tr><td>FDE-BWT</td><td>75.7</td><td>0.5</td><td>0.9</td><td>9.7</td><td>13.1</td><td>19.3</td><td>17.4</td><td>45.9</td></tr><tr><td>MR-BWT</td><td>89.8</td><td>1.5</td><td>3.4</td><td>1.3</td><td>4.0</td><td>39.5</td><td>26.5</td><td>25.6</td></tr><tr><td>FDE-CT2:N</td><td>4.9</td><td>20.7</td><td>2.3</td><td>20.2</td><td>52.0</td><td>24.6</td><td>1.8</td><td>53.1</td></tr><tr><td> $\mathbf { M R - C T } _ { 2 : N }$ </td><td>7.9</td><td>15.9</td><td>2.8</td><td>24.3</td><td>49.2</td><td>21.3</td><td>2.1</td><td>51.2</td></tr></table>

0.9%, and 3.4% for the algorithm and 11.3%, 10.3%, 9.7%, and 1.3% for their interaction. Starting-task identity alone accounts for 69.6%, 58.3%, 75.7%, and 89.8%, which is 90% to 99% of the order term. Among the 18 orders that do not start with A, the order term falls to 30.6%, 24.5%, 19.3%, and 39.5%. Within-cell training variation then becomes the largest component for FDE-CT, MR-CT, and FDE-BWT (50.8%, 51.2%, and 45.9%), whereas for MR-BWT the order term remains largest and the algorithm term (26.5%) matches within-cell variation (25.6%). Relative to the share expected from within-cell noise alone (5.8%, 8.4%, 4.7%, and 1.4%, derived in Appendix A-C), the interaction is clearly elevated for FDE-CT and FDE-BWT.

For FDE-CT, the algorithm-averaged order means range from 0.97 to 6.92 m, a spread of 5.95 m. The median gap between the best and worst algorithm within an order is 0.87 m, and the gap between the largest and smallest uniformorder algorithm means is 0.62 m. Changing the order thus moves FDE-CT far more than changing the algorithm, and most of this effect is the identity of the first task.

Every algorithm has the smallest replicate-mean FDE-CT on at least three orders: Vanilla on 3, DER on 9, SyReM on 9, and GSM on 3, with bootstrap intervals of [1, 8], [5, 12], [5, 12], and [1, 6]. All four also lead on at least one MR-CT order (2, 8, 11, and 3), whereas DER never attains a perorder minimum of either BWT metric. Fig. 1(a) illustrates such a reversal: A→B→C→D and A→B→D→C differ only in their final two tasks, and the FDE-CT ranking of the first (GSM, DER, SyReM, Vanilla) is exactly reversed in the second. The replicates resolve the per-order leader from the runner-up in 4, 2, 10, and 11 of the 24 orders for FDE-CT, MR-CT, FDE-BWT, and MR-BWT, and the separable FDE-BWT and MR-BWT leaders are nearly all SyReM (9 of 10 and 10 of 11). In the two orders of Fig. 1(a) the leaders lie within 0.63 pooled standard deviations of the runners-up. The clearest separable FDE-CT reversal is between A→C→B→D, where SyReM leads with 3.31 m, and A→C→D→B, where DER leads with 3.48 m, with gaps of 3.65 and 1.10 pooled standard deviations. All four of these orders start with task A, and their rankings largely follow the first-stage error on A, for example 10.7 m for SyReM against 21.5 to 24.8 m for the other algorithms on A→C→B→D, whereas the warmstart errors of the four algorithms lie between 0.9 and 1.8 m.

Because the first stage is the same for every A-first order, these reversals arise mainly from run-to-run variation in that stage, which the next subsection examines.

## B. The Starting Task and the Cold Start of Task A

Table V resolves the FDE summaries by starting task. When task A is learned first, the first-stage error $R _ { 1 , 1 }$ averages 21.52 m over the 72 runs, ranging from 4.7 to 42.9 m, with a first-stage miss rate of 93.8%. For the other starting tasks, $R _ { 1 , 1 }$ averages 1.04 to 2.77 m. When A is learned at positions 2, 3, and 4 instead, its own stage error is 3.30, 1.87, and 1.18 m, against 0.88 to 1.32 m for the other tasks at warm positions, and its final error is 0.96 to 1.18 m in every position. The algorithm-averaged FDE-CT accordingly falls from 6.25 m with A in position 1 to 1.76, 1.42, and 1.20 m with A in positions 2, 3, and 4. Task A is both the smallest task and the most distinct in speed, and its failure in the first stage may stem from either property or both. The algorithms already differ within this stage: DER and SyReM reach lower A-first firststage errors than Vanilla and GSM (17.6 and 18.5 m against 26.4 and 23.6 m), consistent with their memory losses acting while the buffer fills.

Removing the first-stage term changes the picture. For the warm-start error $\mathrm { C T } _ { 2 : N }$ , the order component drops to 25.6% (FDE) and 23.8% (MR), the starting-task share to 4.9% and 7.9%, and within-cell training variation rises to 52.0% and 49.2% (Table II). A-first orders are no longer the worst starting group (Table V), but only because their $\mathrm { C T } _ { 2 : N }$ averages tasks B, C, and D, whereas for the other starting tasks it also includes the warm-position error of A, about twice that of the other tasks. Without that term, A-first orders have the highest warm-start error (1.16 m against 0.95 to 1.05 m). The residual order effect is thus comparable to training variation and is organised by the position of A, whose group means account for 13.3% (FDE) and 15.7% (MR) of the total sum of squares, half to two-thirds of the residual order term.

## C. Descriptor Associations and the Position of Task A

Table VI shows how the associations of D with CL performance change across the five candidate task distances. W1 on speed, the primary distance, was selected against these same outcomes in the exploratory scan (Appendix A-B). MMD and W1 on speed order the descriptor values identically and therefore give identical Spearman correlations. Under the acceleration distance, the pair containing A is the closer pair in two of the three partitions, so D changes sign for the 16 orders built on those partitions and no longer tracks the position of A (Spearman −0.61 against +0.79 for speed). The correlations in the last column reverse in sign accordingly, with smaller magnitudes for BWT.

Across the 16 algorithm–metric series, absolute Spearman correlations with D range from 0.53 to 0.81, with algorithmaveraged values of −0.64, −0.63, +0.61, and +0.74 for FDE-CT, MR-CT, FDE-BWT, and MR-BWT. Fig. 5 resolves these associations for each algorithm and metric across the six tied descriptor levels, and the two clusters visible in every

CURRENT-TASK ERRORS BY TASK ORDER (MEAN ± SAMPLE STANDARD DEVIATION OVER THREE REPLICATES). FDE IS IN METRES AND MR IN PERCENT. BOLD: PER-ORDER MINIMUM SEPARABLE FROM THE RUNNER-UP (SECTION IV-D).
<table><tr><td rowspan="2">Order</td><td colspan="2">Vanilla</td><td colspan="2">DER</td><td colspan="2">SyReM</td><td colspan="2">GSM</td></tr><tr><td>FDE-CT</td><td>MR-CT</td><td>FDE-CT</td><td>MR-CT</td><td>FDE-CT</td><td>MR-CT</td><td>FDE-CT</td><td>MR-CT</td></tr><tr><td>A→B→C→D</td><td> $8 . 2 5 { \scriptstyle \pm 0 . 4 6 }$ </td><td>32.83±2.59</td><td>6.17±2.13</td><td>31.75±1.97</td><td>7.17±3.71</td><td>31.74±1.65</td><td>6.08±1.36</td><td>31.84±0.56</td></tr><tr><td> $\mathrm { A {  } B {  } D {  } C }$ </td><td> $4 . 9 8 _ { \pm 2 . 4 9 }$ </td><td> $3 0 . 8 3 { \scriptstyle \pm 4 . 4 3 }$ </td><td> $6 . 6 7 \pm 2 . 3 9$ </td><td>32.20±1.14</td><td> $6 . 5 5 _ { \pm 2 . 4 8 } ^ { ^ { \perp } }$ </td><td> $2 8 . 7 6 { \scriptstyle \pm 2 . 2 7 }$ </td><td>8.00±3.49</td><td>34.38±0.52</td></tr><tr><td>A→C→B→D</td><td> $7 . 3 3 { \scriptstyle \pm 3 . 0 1 }$ </td><td>33.28±4.02</td><td> $6 . 6 9 { \scriptstyle \pm 2 . 5 7 }$ </td><td>29.44±1.99</td><td>3.31±0.31</td><td>25.33±0.67</td><td> $6 . 2 3 { \scriptstyle \pm 1 . 0 9 }$ </td><td>31.86±2.69</td></tr><tr><td>A→C→D→B</td><td> $9 . 2 6 _ { \pm 1 . 9 5 }$ </td><td> $2 9 . 9 6 _ { \pm 1 . 8 1 }$ </td><td> $\mathbf { 3 . 4 8 _ { \pm 1 . 0 2 } }$ </td><td> $2 8 . 8 5 { \scriptstyle \pm 3 . 1 5 }$ </td><td> $4 . 7 1 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $2 7 . 9 2 _ { \pm 4 . 0 5 }$ </td><td>5.00±1.84</td><td> $3 0 . 2 1 { \scriptstyle \pm 1 . 1 7 }$ </td></tr><tr><td>A→D→B→C</td><td> $9 . 2 0 { \scriptstyle \pm 1 . 9 8 }$ </td><td>31.51±2.50</td><td> $3 . 3 8 _ { \pm 0 . 7 0 }$ </td><td>28.18±4.41</td><td> $3 . 9 0 { \scriptstyle \pm 1 . 9 7 }$ </td><td> $2 6 . 0 8 _ { \pm 6 . 6 7 }$ </td><td> $9 . 1 7 { \scriptstyle \pm 2 . 1 1 }$ </td><td> $3 5 . 8 3 _ { \pm 0 . 4 4 }$ </td></tr><tr><td>A→D→C→B</td><td> $5 . 9 2 _ { \pm 2 . 5 4 }$ </td><td> $2 6 . 3 4 { \scriptstyle \pm 4 . 4 0 }$ </td><td> $5 . 1 0 { \scriptstyle \pm 3 . 7 3 }$ </td><td> $2 5 . 0 8 _ { \pm 1 . 6 5 }$ </td><td> $6 . 5 3 { \scriptstyle \pm 2 . 4 4 }$ </td><td> $2 9 . 3 8 _ { \pm 0 . 5 4 }$ </td><td> $6 . 9 7 _ { \pm 0 . 3 4 }$ </td><td> $3 2 . 3 7 { \scriptstyle \pm 2 . 8 2 }$ </td></tr><tr><td>B→A→C→D</td><td> $2 . 4 0 { \scriptstyle \pm 1 . 1 3 }$ </td><td>21.10±8.23</td><td> $2 . 9 7 _ { \pm 1 . 7 7 }$ </td><td> $2 1 . 9 2 _ { \pm 1 3 . 6 2 }$ </td><td> $2 . 1 6 _ { \pm 0 . 2 0 }$ </td><td> $2 1 . 0 2 _ { \pm 3 . 6 3 }$ </td><td>2.66±1.38</td><td>21.34±10.22</td></tr><tr><td>B→A→D→C</td><td> $2 . 1 3 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $2 0 . 7 5 { \scriptstyle \pm 6 . 8 1 }$ </td><td> $2 . 0 9 _ { \pm 0 . 6 5 }$ </td><td> $1 9 . 1 1 _ { \pm 8 . 9 8 }$ </td><td> $1 . 6 1 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 5 . 9 8 _ { \pm 5 . 3 6 }$ </td><td> $2 . 7 0 { \scriptstyle \pm 1 . 2 7 }$ </td><td>23.88±7.66</td></tr><tr><td>B→C→A→D</td><td> $1 . 4 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 2 . 3 1 { \scriptstyle \pm 4 . 2 1 }$ </td><td> $2 . 1 4 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 2 . 5 7 { \scriptstyle \pm 1 3 . 9 2 }$ </td><td> $1 . 2 1 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $8 . 4 6 { \scriptstyle \pm 4 . 8 7 }$ </td><td>1.52±0.32</td><td> $1 3 . 2 4 { \scriptstyle \pm 4 . 2 4 }$ </td></tr><tr><td>B→C→D→A</td><td> $1 . 2 1 _ { \pm 0 . 3 2 }$ </td><td> $9 . 1 6 _ { \pm 1 . 5 1 }$ </td><td> $1 . 1 2 _ { \pm 0 . 3 0 }$ </td><td> $7 . 3 4 { \scriptstyle \pm 2 . 9 8 }$ </td><td> $1 . 2 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 0 . 6 5 { \scriptstyle \pm 2 . 0 5 }$ </td><td>1.72±0.57</td><td> $1 6 . 5 1 { \scriptstyle \pm 9 . 9 1 }$ </td></tr><tr><td>B→D→A→C</td><td> $1 . 6 2 _ { \pm 0 . 1 9 }$ </td><td> $1 5 . 0 8 _ { \pm 3 . 6 3 }$ </td><td> $1 . 7 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td>13.61±1.82</td><td> $1 . 4 8 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $1 5 . 6 7 _ { \pm 7 . 4 4 }$ </td><td> $1 . 8 0 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $1 6 . 0 2 _ { \pm 7 . 9 9 }$ </td></tr><tr><td>B→D→C→A</td><td> $1 . 5 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 1 . 7 1 { \scriptstyle \pm 3 . 1 1 }$ </td><td> $1 . 3 3 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $1 2 . 8 7 { \scriptstyle \pm 7 . 7 3 }$ </td><td> $1 . 6 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 2 . 8 5 { \scriptstyle \pm 3 . 6 9 }$ </td><td>1.84±0.19</td><td>17.31±3.29</td></tr><tr><td>C→A→B→D</td><td> $1 . 2 7 _ { \pm 0 . 2 8 }$ </td><td> $1 4 . 7 4 { \scriptstyle \pm 5 . 9 0 }$ </td><td> $1 . 1 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 0 . 4 9 _ { \pm 4 . 6 1 }$ </td><td> $1 . 2 8 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 4 . 3 4 { \scriptstyle \pm 9 . 4 4 }$ </td><td>1.01±0.20</td><td> $7 . 8 1 { \scriptstyle \pm 3 . 5 7 }$ </td></tr><tr><td>C→A→D→B</td><td> $1 . 6 6 { \scriptstyle \pm 0 . 4 2 }$ </td><td>22.08±6.90</td><td> $1 . 0 7 _ { \pm 0 . 3 5 }$ </td><td> $9 . 7 5 { \scriptstyle \pm 7 . 2 9 }$ </td><td> $1 . 1 5 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 1 . 3 2 { \scriptstyle \pm 5 . 5 1 }$ </td><td>0.97±0.15</td><td>7.47±3.60</td></tr><tr><td>C→B→A→D</td><td> $1 . 0 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $9 . 6 4 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $1 . 0 7 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $6 . 7 8 { \scriptstyle \pm 2 . 1 1 }$ </td><td> $0 . 9 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $7 . 3 4 \pm 2 . 3 5$ </td><td> $1 . 5 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $1 7 . 0 8 { \scriptstyle \pm 8 . 0 5 }$ </td></tr><tr><td>C→B→D→A</td><td> $1 . 0 2 _ { \pm 0 . 1 1 }$ </td><td>8.37±1.34</td><td> $1 . 1 2 _ { \pm 0 . 3 7 }$ </td><td> $1 1 . 6 7 _ { \pm 8 . 1 3 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $8 . 7 6 { \scriptstyle \pm 3 . 7 4 }$ </td><td> $1 . 0 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $7 . 8 5 { \scriptstyle \pm 3 . 4 2 }$ </td></tr><tr><td>C→D→A→B</td><td> $1 . 3 2 _ { \pm 0 . 7 9 }$ </td><td>12.34±10.76</td><td> $\mathbf { 0 . 9 1 { \scriptstyle \pm 0 . 1 3 } }$ </td><td> $6 . 6 8 { \scriptstyle \pm 3 . 2 3 }$ </td><td> $1 . 2 2 _ { \pm 0 . 6 3 }$ </td><td> $9 . 6 2 _ { \pm 8 . 5 2 }$ </td><td>1.16±0.20</td><td> $1 1 . 0 7 { \scriptstyle \pm 2 . 8 8 }$ </td></tr><tr><td>C→D→B→A</td><td> $1 . 0 5 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $7 . 8 8 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $0 . 9 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $7 . 5 8 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $0 . 8 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\mathbf { 5 . 9 1 } _ { \pm 2 . 2 1 }$ </td><td>0.99±0.18</td><td> $8 . 2 0 { \scriptstyle \pm 4 . 2 4 }$ </td></tr><tr><td>D→A→B→C</td><td> $1 . 7 6 { \scriptstyle \pm 0 . 9 3 }$ </td><td>18.29±13.10</td><td>1.08±0.20</td><td> $1 0 . 4 6 _ { \pm 4 . 8 7 }$ </td><td> $1 . 8 4 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $1 7 . 5 4 { \scriptstyle \pm 1 1 . 5 8 }$ </td><td>2.04±0.82</td><td>21.88±10.08</td></tr><tr><td>D→A→C→B</td><td> $1 . 7 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 3 . 7 2 { \scriptstyle \pm 5 . 0 6 }$ </td><td> $1 . 1 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 3 . 0 3 _ { \pm 4 . 8 8 }$ </td><td> $1 . 3 7 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $1 2 . 0 3 _ { \pm 4 . 4 2 }$ </td><td>2.92±2.14</td><td> $2 3 . 6 9 _ { \pm 7 . 2 9 }$ </td></tr><tr><td>D→B→A→C</td><td> $1 . 2 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $9 . 6 8 { \scriptstyle \pm 3 . 1 2 }$ </td><td> $2 . 3 9 { \scriptstyle \pm 1 . 7 6 }$ </td><td>21.68±11.80</td><td> $1 . 3 6 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 4 . 1 1 { \scriptstyle \pm 9 . 0 7 }$ </td><td> $1 . 4 9 { \scriptstyle \pm 0 . 5 6 }$ </td><td>19.62±11.26</td></tr><tr><td>D→B→C→A</td><td> $1 . 1 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 3 . 1 5 { \scriptstyle \pm 3 . 2 4 }$ </td><td> $1 . 1 0 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 0 . 3 9 { \scriptstyle \pm 3 . 8 8 }$ </td><td> $1 . 1 8 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 1 . 0 6 _ { \pm 1 . 6 4 }$ </td><td> $1 . 3 6 _ { \pm 0 . 4 5 }$ </td><td> $1 5 . 9 5 { \scriptstyle \pm 7 . 9 4 }$ </td></tr><tr><td>D→C→A→B</td><td> $1 . 0 4 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 0 . 1 2 _ { \pm 1 . 5 7 }$ </td><td> $2 . 2 4 _ { \pm 0 . 9 0 }$ </td><td> $2 4 . 8 2 _ { \pm 7 . 9 4 }$ </td><td> $1 . 0 1 _ { \pm 0 . 1 3 }$ </td><td> $7 . 8 9 { \scriptstyle \pm 2 . 0 5 }$ </td><td> $1 . 0 2 _ { \pm 0 . 1 1 }$ </td><td> $9 . 9 8 _ { \pm 2 . 7 2 }$ </td></tr><tr><td>D→C→B→A</td><td> $1 . 0 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $9 . 4 4 { \scriptstyle \pm 2 . 9 6 }$ </td><td> $1 . 0 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td>9.01±2.85</td><td> $1 . 0 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 1 . 1 7 { \scriptstyle \pm 2 . 4 8 }$ </td><td>1.32±0.52</td><td> $1 4 . 5 6 { \scriptstyle \pm 7 . 8 5 }$ </td></tr></table>

BACKWARD TRANSFER BY TASK ORDER (MEAN ± SAMPLE STANDARD DEVIATION OVER THREE REPLICATES). FDE IS IN METRES AND MR IN PERCENTAGE POINTS. BOLD AS IN TABLE III.
<table><tr><td rowspan="2">Order</td><td colspan="2">Vanilla</td><td colspan="2">DER</td><td colspan="2">SyReM</td><td colspan="2">GSM</td></tr><tr><td>FDE-BWT</td><td>MR-BWT</td><td>FDE-BWT</td><td>MR-BWT</td><td>FDE-BWT</td><td>MR-BWT</td><td>FDE-BWT</td><td>MR-BWT</td></tr><tr><td> $\mathrm { A {  } B {  } C {  } D }$ </td><td> $- 1 6 . 3 9 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $- 4 6 . 1 9 { \scriptstyle \pm 3 . 1 5 }$ </td><td>-10.99±5.05</td><td>-41.41±0.95</td><td>-14.77±8.97</td><td> $- 5 2 . 9 5 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $- 1 2 . 0 0 { \scriptstyle \pm 3 . 6 3 }$ </td><td>-54.18±2.86</td></tr><tr><td>A→B→D→C</td><td>−8.41±5.26</td><td>-42.47±2.39</td><td>-12.28±5.50</td><td> $- 4 4 . 1 1 { \scriptstyle \pm 2 . 9 3 }$ </td><td>-13.55±5.89</td><td>-54.58±3.96</td><td>-15.61±9.07</td><td>-48.23±11.43</td></tr><tr><td>A→C→B→D</td><td>-13.97±7.56</td><td>-47.29±3.65</td><td>−13.49±6.02</td><td> $- 4 6 . 4 6 { \scriptstyle \pm 2 . 7 7 }$ </td><td>-5.96±0.70</td><td> $- 5 0 . 2 0 { \scriptstyle \pm 1 . 1 1 }$ </td><td>-12.35±2.34</td><td>-49.80±3.44</td></tr><tr><td>A→C→D→B</td><td> $- 1 9 . 8 4 _ { \pm 4 . 7 7 }$ </td><td> $- 4 2 . 4 7 _ { \pm 1 1 . 1 9 }$ </td><td>-5.36±2.77</td><td> $- 3 7 . 4 5 { \scriptstyle \pm 1 2 . 6 4 }$ </td><td>-9.18±3.12</td><td> $- 5 3 . 1 4 _ { \pm 5 . 2 3 }$ </td><td>-8.97±5.24</td><td>-48.22±7.97</td></tr><tr><td>A→D→B→C</td><td> $- 1 9 . 4 2 _ { \pm 4 . 9 4 }$ </td><td>-45.86±2.94</td><td>-5.44±1.35</td><td> $- 4 0 . 0 2 _ { \pm 8 . 0 1 }$ </td><td>-7.15±4.53</td><td>-45.12±8.76</td><td>-18.75±5.52</td><td>-50.08±2.73</td></tr><tr><td>A→D→C→B</td><td> $- 1 1 . 5 0 { \scriptstyle \pm 6 . 6 8 }$ </td><td> $- 3 3 . 6 9 { \scriptstyle \pm 2 1 . 4 6 }$ </td><td> $- 1 0 . 0 1 { \scriptstyle \pm 9 . 2 2 }$ </td><td>-41.18±7.96</td><td> $- 1 3 . 7 3 { \scriptstyle \pm 5 . 8 6 }$ </td><td> $\mathbf { - 5 6 . 9 7 { \pm } 0 . 5 6 }$ </td><td>-14.32±0.59</td><td>-49.42±2.57</td></tr><tr><td>B→A→C→D</td><td> $0 . 3 8 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $- 4 . 7 7 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $0 . 3 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $2 . 8 0 { \scriptstyle \pm 7 . 9 1 }$ </td><td> $- 0 . 3 5 { \scriptstyle \pm 0 . 6 4 }$ </td><td>-12.22±2.50</td><td>-0.68±1.66</td><td>-8.45±10.69</td></tr><tr><td>B→A→D→C</td><td> $- 0 . 9 5 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $- 8 . 3 4 _ { \pm 1 0 . 3 4 }$ </td><td>-0.13±0.46</td><td> $- 3 . 3 8 { \scriptstyle \pm 7 . 2 3 }$ </td><td> $- 0 . 1 8 _ { \pm 1 . 0 7 }$ </td><td> $- 7 . 8 8 _ { \pm 1 2 . 8 3 }$ </td><td>-0.36±1.37</td><td>-5.76±7.80</td></tr><tr><td>B→C→A→D</td><td> $0 . 1 7 { \scriptstyle \pm 0 . 8 4 }$ </td><td>4.50±2.97</td><td> $0 . 5 7 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 . 1 4 { \scriptstyle \pm 4 . 9 9 }$ </td><td> $- 0 . 5 1 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $- 4 . 4 3 { \scriptstyle \pm 9 . 0 9 }$ </td><td>−0.63±0.44</td><td>-3.01±2.01</td></tr><tr><td>B→C→D→A</td><td> $0 . 9 7 { \scriptstyle \pm 1 . 1 8 }$ </td><td>12.43±9.35</td><td> $1 . 0 8 _ { \pm 0 . 2 9 }$ </td><td> $1 0 . 7 4 { \scriptstyle \pm 3 . 7 8 }$ </td><td>−0.47±0.65</td><td>-3.43±7.03</td><td>−0.72±0.92</td><td> $1 . 5 8 { \scriptstyle \pm 3 . 4 7 }$ </td></tr><tr><td>B→D→A→C</td><td> $0 . 4 4 _ { \pm 0 . 8 5 }$ </td><td>4.86±1.12</td><td> $- 0 . 0 1 _ { \pm 0 . 8 0 }$ </td><td> $5 . 3 9 { \scriptstyle \pm 4 . 1 7 }$ </td><td> $- 0 . 3 8 { \scriptstyle \pm 0 . 6 6 }$ </td><td>-5.72±3.49</td><td> $- 0 . 8 5 _ { \pm 0 . 3 2 }$ </td><td> $- 2 . 7 3 { \scriptstyle \pm 2 . 7 0 }$ </td></tr><tr><td>B→D→C→A</td><td> $0 . 4 3 { \scriptstyle \pm 1 . 2 5 }$ </td><td>7.10±11.32</td><td> $0 . 8 7 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $1 1 . 6 2 { \scriptstyle \pm 2 . 7 7 }$ </td><td>-1.02±1.18</td><td> $- 0 . 1 1 { \scriptstyle \pm 3 . 7 9 }$ </td><td>-0.81±0.70</td><td> $1 . 2 7 { \scriptstyle \pm 4 . 8 0 }$ </td></tr><tr><td>C→A→B→D</td><td> $0 . 6 6 _ { \pm 0 . 1 4 }$ </td><td>6.36±4.90</td><td> $0 . 6 8 _ { \pm 0 . 1 3 }$ </td><td> $9 . 9 1 _ { \pm 3 . 5 8 }$ </td><td> $\mathbf { - 0 . 1 0 { \scriptstyle \pm 0 . 3 8 } }$ </td><td>-2.38±7.88</td><td>0.22±0.05</td><td> $5 . 5 7 { \scriptstyle \pm 1 . 9 8 }$ </td></tr><tr><td>C→A→D→B</td><td> $0 . 3 8 _ { \pm 0 . 1 5 }$ </td><td>3.60±5.30</td><td> $0 . 4 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $8 . 9 1 { \scriptstyle \pm 3 . 9 4 }$ </td><td> $\mathbf { 0 . 0 5 _ { \pm 0 . 1 2 } }$ </td><td> $1 . 9 2 _ { \pm 3 . 4 5 }$ </td><td> $0 . 4 2 _ { \pm 0 . 3 0 }$ </td><td> $8 . 8 7 _ { \pm 6 . 1 6 }$ </td></tr><tr><td>C→B→A→D</td><td> $0 . 9 7 { \scriptstyle \pm 0 . 3 1 }$ </td><td>11.47±3.74</td><td> $0 . 8 0 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 4 . 1 1 _ { \pm 1 . 9 4 }$ </td><td>0.11±0.17</td><td> $\mathbf { 0 . 2 2 _ { \pm 4 . 5 1 } }$ </td><td> $0 . 4 6 _ { \pm 0 . 6 3 }$ </td><td> $1 0 . 3 0 { \scriptstyle \pm 4 . 4 8 }$ </td></tr><tr><td>C→B→D→A</td><td> $1 . 1 2 _ { \pm 0 . 1 9 }$ </td><td> $1 3 . 3 5 { \scriptstyle \pm 3 . 4 1 }$ </td><td> $1 . 2 9 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $1 5 . 8 7 _ { \pm 2 . 0 3 }$ </td><td> $\mathbf { - 0 . 0 7 \pm 0 . 2 5 }$ </td><td> $- 2 . 1 4 { \scriptstyle \pm 6 . 7 8 }$ </td><td> $0 . 2 6 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 . 5 2 _ { \pm 6 . 0 5 }$ </td></tr><tr><td>C→D→A→B</td><td> $0 . 6 4 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $1 4 . 7 7 { \scriptstyle \pm 3 . 6 7 }$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 4 . 6 9 { \scriptstyle \pm 2 . 5 8 }$ </td><td> $\mathbf { 0 . 2 1 \pm 0 . 2 5 }$ </td><td> $\mathbf { 3 . 2 9 _ { \pm 3 . 0 1 } }$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td>13.43±1.53</td></tr><tr><td>C→D→B→A</td><td> $0 . 8 2 _ { \pm 0 . 1 2 }$ </td><td> $1 6 . 9 1 _ { \pm 1 . 5 5 }$ </td><td> $0 . 7 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td>13.16±8.55</td><td>0.21±0.15</td><td>4.24±3.76</td><td> $0 . 8 5 { \scriptstyle \pm 0 . 5 1 }$ </td><td>15.54±6.66</td></tr><tr><td>D→A→B→C</td><td> $0 . 4 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $1 0 . 6 5 { \scriptstyle \pm 7 . 3 5 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 4 . 0 6 _ { \pm 2 . 0 5 }$ </td><td> $- 0 . 4 9 { \scriptstyle \pm 1 . 3 4 }$ </td><td> $\mathbf { - 3 . 5 9 _ { \pm 1 2 . 0 0 } }$ </td><td> $1 . 3 8 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $1 5 . 0 6 _ { \pm 9 . 2 4 }$ </td></tr><tr><td>D→A→C→B</td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td>5.37±3.07</td><td> $0 . 6 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 1 . 9 1 _ { \pm 4 . 4 4 }$ </td><td> $0 . 2 9 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $2 . 6 2 _ { \pm 1 . 8 0 }$ </td><td> $- 0 . 7 5 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $0 . 8 4 { \scriptstyle \pm 2 . 5 6 }$ </td></tr><tr><td>D→B→A→C</td><td> $0 . 8 6 _ { \pm 0 . 4 9 }$ </td><td>17.83±2.77</td><td> $1 . 0 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $1 5 . 5 3 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $\mathbf { 0 . 2 1 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $\mathbf { 3 . 2 1 \pm 1 . 6 4 }$ </td><td> $0 . 9 0 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $8 . 4 4 { \scriptstyle \pm 5 . 0 8 }$ </td></tr><tr><td>D→B→C→A</td><td> $1 . 0 3 { \scriptstyle \pm 0 . 1 7 }$ </td><td>17.66±3.60</td><td> $0 . 9 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 6 . 6 3 { \scriptstyle \pm 3 . 1 7 }$ </td><td> $\mathbf { 0 . 0 1 _ { \pm 0 . 1 9 } }$ </td><td>2.55±1.02</td><td> $0 . 8 4 { \scriptstyle \pm 0 . 5 8 }$ </td><td>13.79±5.05</td></tr><tr><td> $\mathrm { D {  } C {  } A {  } B }$ </td><td> $0 . 7 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 5 . 6 1 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $0 . 7 3 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 2 . 9 9 { \scriptstyle \pm 5 . 9 9 }$ </td><td> $\mathbf { 0 . 2 1 { \scriptstyle \pm 0 . 2 2 } }$ </td><td>6.13±6.09</td><td> $0 . 5 4 { \scriptstyle \pm 0 . 2 5 }$ </td><td>10.47±4.67</td></tr><tr><td>D→C→B→A</td><td> $0 . 8 5 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $1 5 . 1 6 { \scriptstyle \pm 6 . 6 3 }$ </td><td> $0 . 9 1 _ { \pm 0 . 2 7 }$ </td><td>18.19±3.59</td><td> $\mathbf { 0 . 0 3 _ { \pm 0 . 1 2 } }$ </td><td> $\mathbf { - 0 . 2 2 _ { \pm 4 . 1 4 } }$ </td><td>0.53±0.41</td><td> $1 1 . 1 8 { \scriptstyle \pm 5 . 7 8 }$ </td></tr></table>

$$
( \mathbf { C T } _ { 2 : N } )
$$

<table><tr><td>Start</td><td> $R _ { 1 , 1 }$ </td><td>CT</td><td> $\mathrm { C T } _ { 2 : N }$ </td><td>P</td><td>B</td><td>BWT</td><td>Final</td></tr><tr><td>A</td><td>21.52</td><td>6.25</td><td>1.16</td><td>1.46</td><td>13.69</td><td>-12.23</td><td>1.28</td></tr><tr><td>B</td><td>2.77</td><td>1.81</td><td>1.48</td><td>2.34</td><td>2.46</td><td>-0.12</td><td>1.52</td></tr><tr><td>C</td><td>1.04</td><td>1.12</td><td>1.15</td><td>1.64</td><td>1.13</td><td>0.52</td><td>1.50</td></tr><tr><td>D</td><td>1.32</td><td>1.45</td><td>1.50</td><td>2.04</td><td>1.50</td><td>0.54</td><td>1.48</td></tr></table>

Table VII quantifies this reading. The position of A alone correlates more strongly than D with each of the four primary metrics $( - 0 . 7 7 , - 0 . 7 9 , + 0 . 7 3 ,$ and +0.77). Excluding A-first orders reduces the mean association with D in each of these metrics, removing starting-task means leaves −0.48, −0.49, +0.29, and +0.62, and removing position-of-A means leaves −0.23, −0.11, +0.22, and +0.53. For the warm-start error the marginal associations are weaker (−0.37 and −0.27 with D, and −0.20 and −0.10 with the position of A), and excluding A-first orders strengthens them because these orders combine the most negative D with low warm-start error. D therefore acts mainly as an index of the position of task A, the most panel coincide with A in positions 1–2 (negative D) and 3–4 (positive D).

![](images/c1e0c73b15b1c27d94dee9c8d4b35cdbef4245fa7e5c279deca4b7def36d69a9.jpg)

![](images/a118acdd78d31b7944a7041277bcd28939d034e94313855d0c81364a84125f31.jpg)

![](images/13107d9823211506f0bb1f80f04192100918a5fc7cd1c2215726062e31037773.jpg)  
Transition-direction descriptor, D(o) (m/s)

![](images/379a99f3a874bbcd246f0576b663ccd3572ab3ed198af1eab84fde8b812208e2.jpg)  
Fig. 5. Transition-direction descriptor versus performance for each algorithm. Points are replicate means for individual orders, coloured by the position of task A, and ρ is the Spearman correlation with D. Grey guides mark the six distinct descriptor values.

TABLE VI  
SPEARMAN CORRELATIONS BETWEEN CANDIDATE TRANSITION-DIRECTION DESCRIPTORS AND ORDER-LEVEL PERFORMANCE. W1: WASSERSTEIN-1. BOLD: $| \rho | \geq 0 . 7 .$
<table><tr><td>Algorithm</td><td>Metric</td><td>W1 speed</td><td>KL speed</td><td>JS speed</td><td>MMD speed</td><td>W1 accel.</td></tr><tr><td>Vanilla</td><td>FDE-CT</td><td>-0.75</td><td>-0.23</td><td>-0.20</td><td>-0.75</td><td>+0.69</td></tr><tr><td></td><td>MR-CT</td><td>-0.75</td><td>-0.09</td><td>-0.12</td><td>-0.75</td><td>+0.69</td></tr><tr><td></td><td>FDE-BWT</td><td>+0.66</td><td>+0.13</td><td>+0.10</td><td>+0.66</td><td>-0.55</td></tr><tr><td></td><td>MR-BWT</td><td>+0.81</td><td>+0.26</td><td>+0.32</td><td>+0.81</td><td>-0.46</td></tr><tr><td>DER</td><td>FDE-CT</td><td>-0.57</td><td>-0.29</td><td>-0.36</td><td>-0.57</td><td>+0.46</td></tr><tr><td></td><td>MR-CT</td><td>-0.54</td><td>-0.22</td><td>-0.27</td><td>-0.54</td><td>+0.48</td></tr><tr><td></td><td>FDE-BWT</td><td>+0.67</td><td>+0.10</td><td>+0.12</td><td>+0.67</td><td>-0.52</td></tr><tr><td></td><td>MR-BWT</td><td>+0.74</td><td>+0.28</td><td>+0.32</td><td>+0.74</td><td>-0.43</td></tr><tr><td>SyReM</td><td>FDE-CT</td><td>-0.67</td><td>-0.29</td><td>-0.26</td><td>-0.67</td><td>+0.74</td></tr><tr><td></td><td>MR-CT</td><td>-0.72</td><td>-0.22</td><td>-0.23</td><td>-0.72</td><td>+0.76</td></tr><tr><td></td><td>FDE-BWT</td><td>+0.55</td><td>+0.31</td><td>+0.36</td><td>+0.55</td><td>-0.35</td></tr><tr><td></td><td>MR-BWT</td><td>+0.72</td><td>+0.35</td><td>+0.43</td><td>+0.72</td><td>-0.42</td></tr><tr><td>GSM</td><td>FDE-CT</td><td>-0.57</td><td>-0.30</td><td>-0.29</td><td>-0.57</td><td>+0.66</td></tr><tr><td></td><td>MR-CT</td><td>-0.53</td><td>-0.23</td><td>-0.23</td><td>-0.53</td><td>+0.69</td></tr><tr><td></td><td>FDE-BWT</td><td>+0.55</td><td>+0.32</td><td>+0.30</td><td>+0.55</td><td>-0.37</td></tr><tr><td></td><td>MR-BWT</td><td>+0.69</td><td>+0.40</td><td>+0.44</td><td>+0.69</td><td>-0.43</td></tr><tr><td>Mean |ρ|</td><td></td><td>0.66</td><td>0.25</td><td>0.27</td><td>0.66</td><td>0.54</td></tr></table>

TABLE VII

distinct and smallest task. Once that position is removed, a substantial association remains only for MR-BWT (+0.53).

## D. Backward Transfer and the Initial Reference

The large negative BWT values for A-first orders motivate a decomposition of the initial reference. Expanding (19) gives

$$
\mathrm { B W T } = P - B ,\tag{22}
$$

MEAN SPEARMAN ASSOCIATIONS ACROSS ALGORITHMS. $\rho ( D )$ DESCRIPTOR AND ORDER MEANS. EXCL. A-FIRST: 18 ORDERS. START-ADJ.: RANK RESIDUALS AFTER REMOVING STARTING-TASK MEANS. $\rho ( p _ { A } )$ : POSITION OF TASK A. p<sub>A</sub>-ADJ.: RANK RESIDUALS AFTER REMOVING POSITION-OF-A MEANS.

<table><tr><td>Metric</td><td> $\rho ( D )$ </td><td>Excl. A-first</td><td>Start-adj.</td><td> $\rho ( p _ { A } )$ </td><td>pA-adj.</td></tr><tr><td>FDE-CT</td><td>-0.64</td><td>-0.47</td><td>-0.48</td><td>-0.77</td><td>-0.23</td></tr><tr><td>MR-CT</td><td>-0.63</td><td>-0.43</td><td>-0.49</td><td>-0.79</td><td>-0.11</td></tr><tr><td>FDE-BWT</td><td>+0.61</td><td>+0.40</td><td>+0.29</td><td>+0.73</td><td>+0.22</td></tr><tr><td>MR-BWT</td><td>+0.74</td><td>+0.64</td><td>+0.62</td><td>+0.77</td><td>+0.53</td></tr><tr><td>FDE-CT2: N</td><td>-0.37</td><td>-0.49</td><td>-0.58</td><td>-0.20</td><td>-0.27</td></tr><tr><td>MR-CT2:N</td><td>-0.27</td><td>-0.47</td><td>-0.50</td><td>-0.10</td><td>-0.20</td></tr></table>

where

$$
P = \frac { 1 } { N - 1 } \sum _ { r = 2 } ^ { N } \frac { 1 } { r - 1 } \sum _ { i < r } R _ { r , i } ,\tag{23}
$$

$$
B = \frac { 1 } { N - 1 } \sum _ { r = 2 } ^ { N } \frac { 1 } { r - 1 } \sum _ { i < r } R _ { i , i } .\tag{24}
$$

Both P and B are error levels: the average subsequent error on previously seen tasks and the corresponding weighted initial reference. A high B makes BWT very negative even when $P$ is moderate. The first task is referenced at every later stage, so for N = 4 its coefficient in B is 11/18, compared with 5/18 and $1 / 9$ for the second and third tasks.

For A-first runs, the subsequent-error and reference terms are 1.46 and 13.69 m, yielding BWT = −12.23 m with a final mean error of 1.28 m (Table V). Across algorithms, the correlation of D with the FDE reference term $B$ is between −0.69 and −0.57, whereas its correlation with the subsequenterror term P is only +0.02 to +0.19. The association of D with FDE-BWT therefore comes mainly from the orderdependent reference term B rather than from the subsequent error P.

TABLE VIII  
FINAL ERRORS: MEAN [MINIMUM, MAXIMUM] OVER ORDER-LEVEL REPLICATE MEANS, AND THE BOOTSTRAP PROBABILITY (PROB.) OF THE SMALLEST UNIFORM MEAN. FDE IS IN METRES AND MR IN PERCENT. BOLD: SMALLEST MEAN.
<table><tr><td>Algorithm</td><td>FDE: mean [min, max]</td><td>Prob.</td><td>MR: mean [min, max]</td><td>Prob.</td></tr><tr><td>Vanilla</td><td>1.63 [1.18, 2.14]</td><td>0.00</td><td>17.22 [10.84, 25.67]</td><td>0.00</td></tr><tr><td>DER</td><td>1.63 [1.17, 2.34]</td><td>0.00</td><td>17.80 [11.47, 29.09]</td><td>0.00</td></tr><tr><td>SyReM</td><td>1.08 [0.87, 1.99]</td><td>1.00</td><td>9.23 [4.38, 24.69]</td><td>1.00</td></tr><tr><td>GSM</td><td>1.43 [1.03, 2.44]</td><td>0.00</td><td>15.86 [8.05, 36.25]</td><td>0.00</td></tr></table>

Final error gives a different comparison (Table VIII). SyReM has the lowest uniform-order final FDE (1.08 m) and MR (9.23%) with bootstrap probability 1.00 for both, and its per-order lead is separable in 17 and 16 of the 24 orders. GSM’s smallest uniform-order FDE-BWT (−3.33 m), by contrast, is within 0.03 m of Vanilla, with probabilities 0.51 and 0.46 (Table IX), and this lead comes entirely from the six A-first orders, where the first-stage reference is largest. Over the 18 remaining orders SyReM has the smallest score in all four metrics, with bootstrap probabilities of 0.90, 0.89, 0.98, and 1.00. Improvement relative to an algorithm-specific initial checkpoint and absolute final prediction quality thus lead to different comparisons, and the final-error comparison is the one the replicates resolve. Because CT and BWT share the initial errors $R _ { i , i }$ with opposite signs, a high initial error raises CT and lowers BWT, which accounts for part of their opposite associations with D.

## E. Different Paths from the Same Starting Task

Fixing the first task leaves six orders of the remaining three. Fig. 6 compares the largest- and smallest-L paths within each group under the W1 matrix on speed. For every starting task the maximum-distance path has the smaller D. The paired paths differ in measured performance, with a direction that depends on the metric. The two B-first paths, for example, have algorithm-averaged FDE-CT values of 2.55 and 1.33 m. Order thus matters beyond the starting task: in the B-, C-, and D-first groups the maximum-distance path also places A earlier (position 2, 2, and 3 against 4) and has the higher current-task error in both FDE and MR, consistent with the position-of-A pattern of the residual order effect in Section VI-B.

## F. Effect of the Order Policy

Table IX compares the uniform and descriptor-weighted scores. Weighting increases the reported CT error for every algorithm: SyReM’s FDE-CT changes from 2.33 to 3.01 m and Vanilla’s from 2.94 to 3.76 m, and BWT shifts upward, for example from −13.79 to −6.04 percentage points for SyReM’s MR-BWT. These shifts follow from the weighting directions: 74.2% of the CT weight falls on the 12 orders with A in positions 1–2 and 37.1% on the six A-first orders, against 50% and 25% under the uniform policy, so the policy emphasises orders that place A early. The effective number of weighted orders is 17.54 of 24.

The complete four-algorithm ranking is identical under the two policies in all four metrics. The bootstrap probabilities identify the leaders that the replicates resolve, SyReM for MR-CT (0.94) and MR-BWT (1.00), while the FDE-CT leader (SyReM, 0.71) and the FDE-BWT leader (GSM, 0.51) remain close to their runners-up. The descriptor weighting thus shifts the aggregate scores and leaves the selected algorithm unchanged in every metric.

TABLE IX  
POLICY SCORES (UNITS AS IN TABLES III AND IV) UNDER THE UNIFORM POLICY, THE DESCRIPTOR-WEIGHTED POLICY, AND THE UNIFORM POLICY  
OVER THE 18 ORDERS THAT DO NOT START WITH A, WITH THE BOOTSTRAP PROBABILITY OF THE SMALLEST SCORE UNDER THAT POLICY IN PARENTHESES. BOLD: SMALLEST SCORE PER ROW.
<table><tr><td>Metric</td><td>Policy</td><td>Vanilla</td><td>DER</td><td>SyReM</td><td>GSM</td></tr><tr><td>FDE-CT</td><td>Uniform</td><td>2.94 (0.00)</td><td>2.42 (0.29)</td><td>2.33 (0.71)</td><td>2.94 (0.00)</td></tr><tr><td></td><td>D-weighted</td><td>3.76 (0.00)</td><td>3.06 (0.43)</td><td>3.01 (0.57)</td><td>3.76 (0.00)</td></tr><tr><td></td><td>Excl. A-first</td><td>1.43 (0.06)</td><td>1.48 (0.04)</td><td>1.31 (0.90)</td><td>1.62 (0.00)</td></tr><tr><td>MR-CT</td><td>Uniform</td><td>18.10 (0.00)</td><td>17.30 (0.05)</td><td>16.04 (0.94)</td><td>19.58 (0.00)</td></tr><tr><td></td><td>D-weighted</td><td>21.87 (0.00)</td><td>20.12 (0.21)</td><td>19.34 (0.79)</td><td>22.98 (0.00)</td></tr><tr><td></td><td>Excl. A-first</td><td>13.86 (0.01)</td><td>13.32 (0.09)</td><td>11.98 (0.89)</td><td>15.19 (0.00)</td></tr><tr><td>FDE-BWT Uniform</td><td></td><td>-3.30 (0.46)</td><td>-1.88 (0.00)</td><td>-2.77 (0.03)</td><td>-3.33 (0.51)</td></tr><tr><td></td><td>D-weighted</td><td>-1.46 (0.36)</td><td>-0.43 (0.00)</td><td>−1.23 (0.02)</td><td>-1.55 (0.61)</td></tr><tr><td></td><td>Excl. A-first</td><td>0.57 (0.00)</td><td>0.69 (0.00)</td><td>-0.13 (0.98)</td><td>0.12 (0.02)</td></tr><tr><td>MR-BWT</td><td>Uniform</td><td>-3.89 (0.00)</td><td>-2.10 (0.00)</td><td>-13.79 (1.00)</td><td>-8.17 (0.00)</td></tr><tr><td></td><td>D-weighted</td><td>5.17 (0.00)</td><td>6.01 (0.00)</td><td>-6.04 (1.00)</td><td>0.74 (0.00)</td></tr><tr><td></td><td>Excl. A-first</td><td>9.14 (0.00)</td><td>11.13 (0.00)</td><td>-1.00 (1.00)</td><td>5.77 (0.00)</td></tr></table>

## VII. DISCUSSION

## A. Interpreting the Influence of Task Order

Across the 24 orders, task order matters chiefly through the learning quality of the first task. Starting-task identity explains 90% to 99% of the order term in each primary metric, so the across-order component of (6) is essentially a startingtask component, driven by the cold start of the smallest task. Beyond the first stage, the order effect becomes comparable to training variability and is organised by where task A is placed: the smallest task stays harder to learn at later positions, and its position accounts for half to two-thirds of the residual order term. Within the first stage, DER and SyReM already reach lower cold-start errors than Vanilla and GSM.

The transition-direction descriptor inherits this structure. Because the pair containing task A is the more distant pair in every partition of the task set, D tracks the position of task A, and that position correlates more strongly with every primary metric than D does.

The metric decomposition explains why the same first stage moves CT and BWT in opposite directions: the cold-start error enters CT with weight 1/N, whereas in the reference term of BWT = P − B the first task carries coefficient 11/18 for N = 4. For A-first orders the reference term is an order of magnitude larger than the subsequent error, so the strongly negative BWT records recovery from a poor start, and the order dependence of BWT is largely a reference-term effect.

## B. Implications for Algorithm Comparison and Benchmark Design

Four reporting items follow for continual learning benchmarks.

![](images/0af44784313d59048288a451fd8e0f72aa370c3b58df884551137fe94aaf5a0b.jpg)

![](images/97f29047061bedc67e1543cc9df3156dd480e85c07afb2ad75255833e1df809c.jpg)

![](images/89532ea609471db9e95d28d5e9d7d0766e6fc085f0dc94dc86b6189889b7f2a8.jpg)

![](images/836f69ccbc9ca44bb01a33d39fb704ffa0f32661fa1bd5b461d5b8c7997bac15.jpg)

![](images/00a1d5a37c190a814e1510ef8f303b0e328ecfa07f3b7448a09e94aab91c5dd4.jpg)  
Fig. 6. Paths with the largest and smallest total transition distance L for each starting task. Order strips identify the paths, and paired points compare algorithm-averaged scores, standardised separately for each metric across task orders.

1) State the order distribution behind every aggregate score, the policy q in (4).

2) Report order-resolved and starting-task-resolved outcomes, since changing the order can move a score by far more than changing the algorithm (Tables III and V).

3) Report the warm-start error $\mathrm { C T } _ { 2 : N }$ and the final error alongside CT and BWT.

4) Fix any descriptor and weighting rule before the outcomes are seen, so that the policy expresses an intended emphasis. Here, the descriptor-weighted policy took its weighting directions from the present outcomes, put most of its CT weight on orders with task A in the first two positions, and left the ranking unchanged (Table IX).

For the compared algorithms, the clearest result is that SyReM attains the smallest final error and the smallest MR-BWT with bootstrap probability 1.00, leads MR-CT with probability 0.94, and leads all four metrics once the A-first orders are set aside. Final error also separates per-order leaders far more often than CT and BWT (16 or 17 of 24 orders against 2 to 11), and every separable final-error leader is SyReM, so final error is the comparison these replicates resolve most clearly. The interaction term exceeds its within-cell noise level for FDE-CT and FDE-BWT, indicating that algorithms differ in their sensitivity to order. For example, the FDE-CT range across orders is 5.8 m for DER and 6.3 m for SyReM, against 8.2 m for GSM and 8.3 m for Vanilla (Table III).

## C. Extension to Other Task Sets

The protocol applies to any task set, whereas the magnitudes reported here belong to one benchmark with four tasks, one backbone, and three replicates per cell. For larger N, sampling from a declared order distribution replaces enumeration, and repeated samples separate coverage uncertainty from training variability. Samples should balance starting tasks, which carried most of the order term here, and the warmstart decomposition should accompany CT. Pre-specifying the descriptor and weights turns the descriptor-weighted policy from the exploratory form used here into a confirmatory one.

Task representations beyond vehicle speed, such as distances on the inputs or on backbone features, would extend the protocol to domains without an application-specific variable, and independently constructed task sets will show how far the starting-task effect generalises beyond motion forecasting.

## VIII. CONCLUSION

This study formulates continual learning evaluation as an expectation over a declared distribution of task orders and applies it to motion forecasting across all 24 orders of four tasks, four algorithms, and three replicates. Task order accounts for 65% to 91% of the variation in current-task error and backward transfer, and almost all of this effect comes from the identity of the starting task, acting through the cold start of the smallest task. Beyond the first stage, the order effect is comparable to training variability, and most rank reversals between orders lie within it. The first task dominates the reference term of backward transfer, so relative improvement and absolute final error identify different leading algorithms for displacement error, and the transition-direction descriptor tracks where the smallest task is placed, leaving the ranking unchanged under descriptor weighting. Benchmarks that declare their order distribution and report order-resolved, starting-task-resolved, warm-start, and final errors make such effects visible and their algorithm comparisons interpretable.

## APPENDIX A

## ANALYSIS DETAILS AND REPRODUCIBILITY

## A. Feature and Distance Reconstruction

Target-agent speed and acceleration are computed from the trajectory arrays of all 12 location subsets. Feature counts and source hashes accompany the released inputs. Equation (8) gives the W1 distance. For KL and JS, let $c _ { i b }$ count speed observations in bin b of $H = 1 0 0$ shared equal-width bins spanning the pooled range. With $\epsilon = 1 0 ^ { - 1 0 }$ , the smoothed probabilities are

$$
p _ { i b } = \frac { c _ { i b } / \sum _ { h = 1 } ^ { H } c _ { i h } + \epsilon } { 1 + H \epsilon } ,\tag{25}
$$

and, writing $m _ { i j b } = ( p _ { i b } + p _ { j b } ) / 2 ,$

$$
d _ { i j } ^ { \mathrm { K L } } = \sum _ { b = 1 } ^ { H } p _ { i b } \log \frac { p _ { i b } } { p _ { j b } } ,\tag{26}
$$

$$
d _ { i j } ^ { \mathrm { J S } } = \frac { 1 } { 2 } \sum _ { b = 1 } ^ { H } \left( p _ { i b } \log \frac { p _ { i b } } { m _ { i j b } } + p _ { j b } \log \frac { p _ { j b } } { m _ { i j b } } \right) ,\tag{27}
$$

with natural logarithms. JS denotes the divergence rather than its square root. For MMD, a fixed sample of $n ~ = ~ 2 { , } 0 0 0$ speed observations per task is standardised with the pooled mean and population standard deviation of all retained speed observations, $z _ { i u } = ( v _ { i u } - \mu _ { v } ) / { \sigma _ { v } } ,$ and compared with the kernel $k ( z , z ^ { \prime } ) = \exp [ - ( z - z ^ { \prime } ) ^ { 2 } ]$ . The same samples are reused across pairs, and $d _ { i j } ^ { \mathrm { M M D } } ~ = ~ \sqrt { \operatorname* { m a x } ( 0 , Q _ { i j } ) }$ , where $Q _ { i j }$ is the biased squared discrepancy $\begin{array} { r } { n ^ { - 2 } \sum _ { u , v } [ k ( z _ { i u } , z _ { i v } ) + } \end{array}$ $\begin{array} { r } { k ( z _ { j u } , z _ { j v } ) \ - \ 2 k ( z _ { i u } , z _ { j v } ) ] } \end{array}$ including diagonal terms. The clipping removes roundoff below zero. MMD samples are drawn without replacement using NumPy PCG64 with seed 0, proceeding through tasks A–D in order. Exact bin edges, standardisation constants, and sampled indices accompany the data.

## B. Exploratory Descriptor Scan

Before the five candidates of Section IV-A were fixed, an exploratory scan compared 20 distance representations (W1, KL, JS, and MMD on speed, acceleration, relative distance, and relative speed, plus four multivariate distances) combined with 15 aggregation operators over the three transitions (the individual transitions, their total, mean, minimum, maximum, range, standard deviation, slope, and six further shape summaries) against the same 24-order outcomes, giving 300 candidates. The complete scan table is included with the released inputs. The W1-on-speed slope was retained for its interpretable physical scale and consistent sign pattern across the 16 algorithm–metric series.

## C. Decomposition, Adjustment, and Bootstrap

For a fixed metric, let $Z _ { o a s }$ be a run outcome with $K = | \mathcal { O } |$ orders, $J = | { \mathcal { A } } |$ algorithms, and S replicates per combination. The balanced two-way decomposition is $S S _ { T } = S S _ { O } + S S _ { A } +$ $S S _ { O A } + S S _ { W }$ , with $S S _ { O } \ { \stackrel { - } { = } } \ J S \sum _ { o } ( \bar { Z } _ { o \cdot \cdot } \ - \ \bar { Z } ) ^ { 2 } , \ S S _ { A } \ =$ $\begin{array} { r } { K S \sum _ { a } ( \bar { Z } . _ { a } . - \bar { Z } ) ^ { 2 } , S S _ { O A } = S \sum _ { o . a } ( \overline { { \bar { Z } } } _ { o a } . - \bar { Z } _ { o } . - \bar { Z } . _ { a } . + \bar { Z } ) ^ { 2 } } \end{array}$ and $\begin{array} { r } { S S _ { W } = \sum _ { o . a . s } ( Z _ { o a s } - \bar { Z } _ { o a . } ) ^ { 2 } } \end{array}$ , where dots denote averaging and Z<sup>¯</sup> is the grand mean. Writing $\bar { Z } _ { g ( o ) }$ for the mean over orders with the same starting task as $^ { O , }$ the order term splits as $\begin{array} { r } { S S _ { O } = J S \sum _ { o } ( \bar { Z } _ { g ( o ) } - \bar { Z _ { } } ) ^ { 2 } + J S \sum _ { o } ( \bar { Z } _ { o \cdot } - \bar { Z } _ { g ( o ) } ) ^ { 2 } . } \end{array}$ Under within-cell noise alone, a term with ν degrees of freedom is expected to take $\nu / \nu _ { W }$ of the within-cell share, with $\nu _ { W } = K J ( S - 1 )$ and $\nu = ( K - 1 ) ( J - 1 )$ for the interaction.

Conditional rank association for an algorithm–metric series assigns average ranks to D and to the order mean, subtracts the respective mean rank within each group, and correlates the two residual vectors. The groups are the four starting tasks or the four positions of task A. The bootstrap resamples the three replicates of every order–algorithm cell with replacement (5,000 draws, NumPy PCG64 seed 0), recomputes replicate means, uniform and weighted scores, and per-order minima, and reports the fraction of draws in which each algorithm attains the smallest score together with 2.5th and 97.5th percentiles of its per-order win count. Rescaling the resampled deviations by $\sqrt { S / ( S - 1 ) }$ , which corrects the plug-in variance of a three-replicate mean, changes the reported probabilities by at most 0.05. All weighted means are convex combinations of order means, and the effective number of weighted orders, $( \sum _ { o } w _ { o } ) ^ { 2 } / \sum _ { o } w _ { o } ^ { 2 }$ , measures weight concentration.

## REFERENCES

[1] H. Yang, Y. Zhou, X. Hu, H. Liu, C. Chen, and C. Lv, “Humanguided continual learning for multifaceted improvement of self-driving vehicles,” Nature Communications, Sep. 2026, early access.

[2] Y. Lin, Z. Li, C. Gong, Q. Liu, C. Lu, and J. Gong, “Rethinking trajectory prediction in real-world applications: An online task-free continual learning perspective,” in 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2023, pp. 5020–5026.

[3] Y. Lin, Z. Li, C. Gong, C. Lu, X. Wang, and J. Gong, “Continual interactive behavior learning with traffic divergence measurement: A dynamic gradient scenario memory approach,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 3, pp. 2355–2372, 2024.

[4] Y. Lin, C. Lu, T. Wu, X. Zhao, G. Du, Y. Sun, Z. Li, and J. Gong, “Escaping stability-plasticity dilemma in online continual learning for motion forecasting via synergetic memory rehearsal,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2026, early access.

[5] Y. Lin, Z. Li, G. Du, X. Zhao, C. Gong, X. Wang, C. Lu, and J. Gong, “H2C: Hippocampal circuit-inspired continual learning for lifelong trajectory prediction in autonomous driving,” IEEE Transactions on Intelligent Transportation Systems, vol. 27, no. 8, pp. 9834–9851, 2026.

[6] Z. Li, Y. Lin, G. Du, X. Zhao, C. Gong, C. Lv, C. Lu, and J. Gong, “Complementary learning system empowers online continual learning of vehicle motion forecasting in smart cities,” arXiv preprint arXiv:2508.19597, 2025.

[7] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Psychology of Learning and Motivation. Elsevier, 1989, vol. 24, pp. 109–165.

[8] R. M. French, “Catastrophic forgetting in connectionist networks,” Trends in Cognitive Sciences, vol. 3, no. 4, pp. 128–135, 1999.

[9] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.

[10] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[11] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 12, pp. 2935– 2947, 2018.

[12] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[13] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with A-GEM,” in 7th International Conference on Learning Representations, ICLR 2019, 2019.

[14] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 15.920–15930.

[15] Z. Li, C. Gong, Y. Lin, G. Li, X. Wang, C. Lu, M. Wang, S. Chen, and J. Gong, “Continual driver behaviour learning for connected vehicles and intelligent transportation systems: Framework, survey and challenges,” Green Energy and Intelligent Transportation, vol. 2, no. 4, p. 100103, 2023.

[16] S. J. Bell and N. D. Lawrence, “The effect of task ordering in continual learning,” arXiv preprint arXiv:2205.13323, 2022.

[17] J. Yoon, S. Kim, E. Yang, and S. J. Hwang, “Scalable and orderrobust continual learning with additive parameter decomposition,” in International Conference on Learning Representations, 2020.

[18] Z. Li and N. Hiratani, “Optimal task order for continual learning of multiple tasks,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 34 578–34 603.

[19] T. Nguyen, C. N. Nguyen, Q. Pham, B. T. Nguyen, S. Ramasamy, X. Li, and C. V. Nguyen, “Sequence transferability and task order selection in continual learning,” arXiv preprint arXiv:2502.06544, 2025.

[20] G. Li, Z. Li, V. L. Knoop, and H. van Lint, “Unravelling uncertainty in trajectory prediction using a non-parametric approach,” Transportation Research Part C: Emerging Technologies, vol. 163, p. 104659, 2024.

[21] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in Proceedings of the 26th International Conference on Machine Learning (ICML), 2009, pp. 41–48.

[22] D. Goldfarb, I. Evron, N. Weinberger, D. Soudry, and P. Hand, “The joint effect of task similarity and overparameterization on catastrophic forgetting—an analytical model,” in International Conference on Learning Representations, 2024.

[23] C. V. Nguyen, A. Achille, M. Lam, T. Hassner, V. Mahadevan, and S. Soatto, “Toward understanding catastrophic forgetting in continual learning,” arXiv preprint arXiv:1908.01091, 2019.

[24] P. Bao, Z. Chen, J. Wang, D. Dai, and H. Zhao, “Lifelong vehicle trajectory prediction framework based on generative replay,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 12, pp. 13 729–13 741, 2023.

[25] Y. Wu, A. Bighashdel, G. Chen, G. Dubbelman, and P. Jancura, “Continual pedestrian trajectory learning with social generative replay,” IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 848–855, 2023.

[26] H. Ma, Y. Sun, J. Li, M. Tomizuka, and C. Choi, “Continual multi-agent interaction behavior prediction with conditional generative memory,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8410–8417, 2021.

[27] L. Knoedler, C. Salmi, H. Zhu, B. Brito, and J. Alonso-Mora, “Improving pedestrian prediction models with self-supervised continual learning,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4781–4788, 2022.

[28] R. Aljundi, K. Kelchtermans, and T. Tuytelaars, “Task-free continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 11 246–11 255.

[29] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 7, pp. 3366–3385, 2022.

[30] G. M. van de Ven, T. Tuytelaars, and A. S. Tolias, “Three types of incremental learning,” Nature Machine Intelligence, vol. 4, no. 12, pp. 1185–1197, 2022.

[31] R. Kemker, M. McClure, A. Abitino, T. L. Hayes, and C. Kanan, “Measuring catastrophic forgetting in neural networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 3390–3398.

[32] S. Farquhar and Y. Gal, “Towards robust evaluations of continual learning,” in Lifelong Learning: A Reinforcement Learning Approach Workshop at ICML, 2018, arXiv:1805.09733.

[33] N. D´ıaz-Rodr´ıguez, V. Lomonaco, D. Filliat, and D. Maltoni, “Don’t forget, there is more than forgetting: new metrics for Continual Learning,” in NeurIPS 2018 Workshop on Continual Learning, 2018, arXiv:1810.13166.

[34] S. Cha and K. Cho, “Hyperparameters in continual learning: A reality check,” Transactions on Machine Learning Research, 2025.

[35] Y. Ghunaim, A. Bibi, K. Alhamoud, M. Alfarra, H. A. A. K. Hammoud, A. Prabhu, P. H. S. Torr, and B. Ghanem, “Real-time evaluation in online continual learning: A new hope,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 11 888–11 897.

[36] S. Chen, Y. Liao, X. Sun, K. Ma, and Y. Wei, “CLDyB: Towards dynamic benchmarking for continual learning with pre-trained models,” in International Conference on Learning Representations, 2025.

[37] M. Zeng, S. Zhou, Z. Zhan, and R. Zhang, “MedCL-Bench: Benchmarking stability-efficiency trade-offs and scaling in biomedical continual learning,” arXiv preprint arXiv:2603.16738, 2026.

[38] G. Peyre and M. Cuturi, “Computational optimal transport: With appli-´ cations to data science,” Foundations and Trends in Machine Learning, vol. 11, no. 5–6, pp. 355–607, 2019.

[39] J. Lin, “Divergence measures based on the Shannon entropy,” IEEE Transactions on Information Theory, vol. 37, no. 1, pp. 145–151, 1991.

[40] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Scholkopf, and A. Smola,¨ “A kernel two-sample test,” Journal of Machine Learning Research, vol. 13, no. 25, pp. 723–773, 2012.

[41] W. Zhan, L. Sun, D. Wang, H. Shi, A. Clausse, M. Naumann, J. Kummerle, H. K¨ onigshof, C. Stiller, A. de La Fortelle, and¨ M. Tomizuka, “INTERACTION dataset: An INTERnational, adversarial and cooperative moTION dataset in interactive driving scenarios with semantic maps,” arXiv preprint arXiv:1910.03088, 2019.